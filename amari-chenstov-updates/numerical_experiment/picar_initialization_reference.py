"""Independent model-label MC diagnostic; never replaces Plan19's primary D0.

Launch only after the source campaign has a sealed orchestration summary:
    python -m numerical_experiment.picar_initialization_reference \
        --source-campaign cache/.../initialization-fix1-v1 --output cache/.../reference-v1

The public CPU-testable kernel is FisherReference(...).run(). All scores use
the accepted LTM anchor, full causal histories and the frozen joint likelihood.
The CLI has no fit, replay, retirement, online condition or source-write path.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from numerical_experiment.picar.config import canonical_json, config_from_dict
from numerical_experiment.picar.report import read_completed_unit
from numerical_experiment.picar.runtime import StopController, sha256_file, utc_now
from src.seeding import derive_component_seed


SCHEMA = 1
VERSION = "plan19-initialization-mc-reference-v1"
SHARED_SECONDS = 36000.
MAX_SECONDS = 1200.
PREPARATION_SECONDS = 180.
CLOSEOUT_SECONDS = 60.
BLOCK_SECONDS = 30.
BLOCK_ITEMS = 16
SEED_COMPONENT = "initialization_mc_reference.independent_model_labels.v1"
PROJECT = Path(__file__).resolve().parents[1]


class ReferenceLimit(TimeoutError):
    pass


@contextmanager
def _deadline(seconds):
    """Bound preparation/kernel blocks too, including a blocking model load."""
    if seconds <= 0:
        raise ReferenceLimit("reference work allowance exhausted")
    previous = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    if timer[0]:
        raise RuntimeError("reference CLI requires ownership of the process alarm")

    def expired(signum, frame):
        raise ReferenceLimit("reference block/deadline reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _pending(reason, *, main_charge=None, active=0., charged=0.):
    return {"schema_version": SCHEMA, "kind": "initialization_mc_reference", "phase": 3,
        "status": "pending", "reason": reason, "smoke_only": True, "development_only": True,
        "active_seconds": active, "sidecar_charged_seconds": charged,
        "mainrepaircharged_seconds": main_charge,
        "combinedrepaircharged_seconds": None if main_charge is None else main_charge + charged,
        "primary_unchanged": True, "reference_is_oracle": False}


def inspect_source(source_campaign):
    """Verify scalar seals without importing Torch or touching the GPU."""
    root = Path(source_campaign).resolve()
    orchestration = root / "orchestration"
    if not (orchestration / "COMPLETED").is_file():
        return None
    main = read_completed_unit(orchestration)
    summary, resolved = main["summary"], main["stored_config"]
    if summary.get("kind") != "initialization_campaign" or resolved.get("version") != "plan19-phase3-fix1-v1":
        raise ValueError("reference requires the sealed Phase3Fix1 v1 campaign")
    charge = summary.get("repair_active_seconds")
    if isinstance(charge, bool) or not isinstance(charge, (int, float)) or not math.isfinite(charge) or not 0 <= charge <= SHARED_SECONDS:
        raise ValueError("main campaign has an invalid shared repair charge")
    units = {}
    for name, key in (("bootstrap", "initial_asset"), ("fit", "selected_fit")):
        path = Path(summary[key]).resolve()
        if not path.is_relative_to(root / "units"):
            raise ValueError("selected source unit is outside the source campaign")
        unit = read_completed_unit(path)
        if unit["summary"].get("kind") != "initialization_" + name:
            raise ValueError("selected source unit has the wrong kind")
        units[name] = unit
    if Path(units["bootstrap"]["summary"]["settings"]["fit"]).resolve() != Path(units["fit"]["path"]):
        raise ValueError("bootstrap and selected fit identities disagree")
    return {"root": root, "main": main, "main_charge": float(charge), **units}


def _completed_state(unit):
    # RunStore opens a completed source read-only and verifies its journal and
    # recovery payload before unpickling. Source files are never repaired.
    from numerical_experiment.picar.storage import RunStore
    with RunStore(unit["path"], unit["stored_config"], resume=True,
                  checkpoint_policy="journal-v1") as store:
        if not store.completed:
            raise ValueError("source checkpoint is incomplete")
        return store.load_checkpoint(trusted=True)


def reconstruct_training(source, fitted, bootstrap):
    """Restore only the collection prefix needed by verified selected-fit IDs."""
    from numerical_experiment.picar_initialization.data import namespace_data
    from numerical_experiment.picar_initialization.numerics import InitializationFit
    from numerical_experiment.picar_phase2_rescue.data import restore_demonstrations

    refs = fitted["identity"]["data"]["train"]
    if (fitted.get("stage") != "fit_done"
            or fitted["accepted_identity"] != bootstrap["accepted_identity"]):
        raise ValueError("selected fit and primary accepted identities differ")
    if not refs or len({r["record_id"] for r in refs}) != len(refs):
        raise ValueError("selected fit training references are empty or duplicated")
    if canonical_json(fitted["identity"]["data"]) != canonical_json(bootstrap["identity"]["data"]):
        raise ValueError("selected fit and primary Fisher conditioning differ")
    if canonical_json(fitted["identity"]["config"]) != canonical_json(bootstrap["identity"]["config"]):
        raise ValueError("selected fit and primary likelihood differ")
    needed_chunks = set()
    for ref in refs:
        namespace = ref["record_id"].split("/", 1)[0]
        if namespace != "original":
            if not namespace.startswith("collection-"):
                raise ValueError("unknown selected training namespace")
            index = int(namespace[len("collection-"):])
            if index < 1:
                raise ValueError("invalid selected collection index")
            needed_chunks.add(index)
    resolved = source["main"]["stored_config"]
    teacher = Path(resolved["teacher_source"])
    if sha256_file(teacher / "COMPLETED") != resolved["teacher_completion_sha256"]:
        raise ValueError("teacher source completion differs from the main campaign")
    original = namespace_data(restore_demonstrations(teacher), "original")
    records, acquired = list(original["records"]), original["acquired_count"]
    collections = {}
    # Earlier chunks are needed for the exact global-arrival offset even when
    # none of their records belongs to this selected training prefix.
    maximum = max(needed_chunks, default=0)
    for marker in sorted((source["root"] / "units").glob("collection-*/COMPLETED")):
        settings = _read_json(marker.parent / "config.json")["settings"]
        index = settings["index"]
        if 1 <= index <= maximum:
            if index in collections:
                raise ValueError("duplicate collection chunk index")
            collections[index] = read_completed_unit(marker.parent)
    if set(collections) != set(range(1, maximum + 1)):
        raise ValueError("selected fit needs a missing/unsealed collection prefix")
    seals = {}
    for index in range(1, maximum + 1):
        unit = collections[index]
        chunk = _completed_state(unit)
        if chunk.get("stage") != "collection_done":
            raise ValueError("collection checkpoint is not completed stationary data")
        restored = namespace_data({"records": chunk["eligible"], "episodes": chunk["episodes"],
            "acquired_count": chunk["acquired_count"]}, f"collection-{index}", arrival_offset=acquired)
        records.extend(restored["records"])
        acquired += restored["acquired_count"]
        seals[str(index)] = sha256_file(Path(unit["path"]) / "COMPLETED")
    lookup = {r["record_id"]: r for r in records}
    if len(lookup) != len(records):
        raise ValueError("reconstructed data has duplicate record identities")
    try:
        selected = sorted((lookup[r["record_id"]] for r in refs), key=lambda r: r["record_id"])
    except KeyError as exc:
        raise ValueError("selected training history is missing") from exc
    actual = InitializationFit._references(selected)
    if canonical_json(actual) != canonical_json(sorted(refs, key=lambda r: r["record_id"])):
        raise ValueError("selected training history/target hash or arrival differs")
    heldout_episodes = {r["episode_id"] for r in fitted["identity"]["data"]["validation"]}
    if heldout_episodes & {r["episode_id"] for r in selected}:
        raise ValueError("validation episode entered the MC reference")
    return selected, {"collection_completion_sha256": seals,
        "train_references_sha256": hashlib.sha256(canonical_json(actual).encode()).hexdigest(),
        "history_count": len(selected), "validation_used": False}


def _validate_primary(bootstrap, records):
    import torch
    from numerical_experiment.picar_initialization.numerics import _tensor_hash

    learner = bootstrap["learner"]
    names = learner["banks"]["ltm"]
    parameters = {n: learner["parameters"][n] for n in names}
    anchor_hash = _tensor_hash(parameters)
    n = len(records)
    if (bootstrap.get("stage") != "done" or not n or bootstrap["eligible"]
            or learner["retired_count"] != n
            or bootstrap["retired"] != sorted(r["arrival"] for r in records)
            or bootstrap["fisher"]["count"] != n
            or bootstrap["fisher"]["anchor_sha256"] != anchor_hash
            or bootstrap["accepted_identity"]["parameters_sha256"] != anchor_hash):
        raise ValueError("primary Fisher count, archive or immutable fitted anchor differs")
    subset = bootstrap["fisher"]["subset_ids"]
    if len(subset) != len(set(subset)) or not set(subset) <= {r["record_id"] for r in records}:
        raise ValueError("primary diagnostic subset differs from selected training")
    for key in ("diagonal", "anchor"):
        if set(learner[key]) != set(names):
            raise ValueError("primary parameter names differ")
        for name in names:
            value = learner[key][name]
            if value.shape != parameters[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError("primary parameter shape/value differs")
    for name in names:
        if not torch.equal(learner["anchor"][name], parameters[name]):
            raise ValueError("primary archive is not at the accepted anchor")
        if bool((learner["diagonal"][name] < 0).any()):
            raise ValueError("primary diagonal is negative")
        expected = bootstrap["fisher"]["nested_sums"]["4"][name] / n
        if not torch.equal(expected.to(learner["diagonal"][name]), learner["diagonal"][name]):
            raise ValueError("primary diagonal is not the equal-weight four-draw mean")
    return anchor_hash


class FisherReference:
    """Independent equal-weight reference using the public sampled-Fisher kernel.

    The immutable bootstrap is the only learner source, never a resume target.
    A record (all M draws) is atomic; checkpoints do not embed its raw history.
    The optional callbacks let the CLI bound/checkpoint work and CPU tests stop
    at exact block boundaries without altering the frozen initialization policy.
    """

    def __init__(self, model, config, records, bootstrap, store, stopped=lambda: None,
                 *, draws=16, block_items=BLOCK_ITEMS, block_seconds=BLOCK_SECONDS,
                 checkpoint_fields=None, before_record=None):
        import torch
        from numerical_experiment.picar.learner import MemoryLearner
        from numerical_experiment.picar_initialization.numerics import _global_rng, _restore_global, _tensor_hash

        if type(draws) is not int or draws < 1 or type(block_items) is not int or block_items < 1:
            raise ValueError("draws and block_items must be positive integers")
        if not math.isfinite(block_seconds) or block_seconds <= 0:
            raise ValueError("block_seconds must be positive")
        self.records = sorted(records, key=lambda r: r["record_id"])
        if len({r["record_id"] for r in self.records}) != len(self.records):
            raise ValueError("reference record IDs are duplicated")
        if canonical_json(config.to_dict()) != canonical_json(bootstrap["identity"]["config"]):
            raise ValueError("reference likelihood/configuration differs from primary")
        from numerical_experiment.picar_initialization.numerics import InitializationFit
        refs = InitializationFit._references(self.records)
        expected_refs = sorted(bootstrap["identity"]["data"]["train"], key=lambda r: r["record_id"])
        if canonical_json(refs) != canonical_json(expected_refs):
            raise ValueError("reference training conditioning differs from the selected fit")
        self.anchor_hash = _validate_primary(bootstrap, self.records)
        self.model, self.config, self.primary, self.store, self.stopped = model, config, bootstrap, store, stopped
        self.draws, self.block_items, self.block_seconds = draws, block_items, block_seconds
        self.checkpoint_fields, self.before_record = checkpoint_fields, before_record
        self.learner = MemoryLearner(model, config.learning, config.seeds(0))
        self.learner.load_state_dict(bootstrap["learner"])
        self.model.distillation_receipt = copy.deepcopy(bootstrap["distill_receipt"])
        self.model.eval()
        self.ltm = {n: self.learner.params[n] for n in self.learner.banks["ltm"]}
        self.seed = derive_component_seed(config.seed, SEED_COMPONENT)
        self.identity = {"schema_version": SCHEMA, "version": VERSION, "anchor_sha256": self.anchor_hash,
            "all_parameters_sha256": _tensor_hash(bootstrap["learner"]["parameters"]),
            "config": config.to_dict(), "draws": draws, "seed": self.seed,
            "train_references_sha256": hashlib.sha256(canonical_json(refs).encode()).hexdigest()}
        self.s = store.load_checkpoint(trusted=True)
        if self.s is None or self.s.get("stage") == "preparing":
            previous = self.s or {}
            self.s = {**previous, "schema_version": SCHEMA, "identity": self.identity,
                "stage": "reference", "cursor": 0, "count": 0,
                "sums": {n: torch.zeros(p.shape, dtype=torch.float64) for n, p in self.ltm.items()},
                "subset_sum": {n: torch.zeros(p.shape, dtype=torch.float64) for n, p in self.ltm.items()},
                "seeds": {"model_labels": self.seed, "component": SEED_COMPONENT,
                    "per_history": "derive_component_seed(model_labels, record_id); whole-history M draws atomic"},
                "anchor_sha256": self.anchor_hash, "started_at": previous.get("started_at", utc_now())}
        else:
            if canonical_json(self.s["identity"]) != canonical_json(self.identity):
                raise ValueError("reference resume identity differs")
            if (type(self.s["count"]) is not int or self.s["cursor"] != self.s["count"]
                    or not 0 <= self.s["count"] <= len(self.records)
                    or self.s["anchor_sha256"] != self.anchor_hash):
                raise ValueError("reference resume cursor/count/anchor differs")
            for key in ("sums", "subset_sum"):
                if set(self.s[key]) != set(self.ltm) or any(
                        v.shape != self.ltm[n].shape or not bool((torch.isfinite(v) & (v >= 0)).all())
                        for n, v in self.s[key].items()):
                    raise ValueError("reference checkpoint sum shape/value differs")
            _restore_global(self.s["rng"], self.learner.device)
        self._global_rng = _global_rng
        self._parameter_hash = _tensor_hash
        self._saved_count, self._saved_time = self.s["count"], time.monotonic()

    def _one(self, record):
        import torch
        from numerical_experiment.picar.learner import LearningBatch, predict_histories
        from src.continual import picar_math as pm

        batch = LearningBatch([record["history"]],
            torch.tensor([record["baseline"]], device=self.learner.device, dtype=self.learner.dtype),
            torch.tensor([record["bin"]], device=self.learner.device, dtype=torch.long),
            record["logits"].detach().reshape(1, 7).to(self.learner.device, self.learner.dtype),
            torch.tensor([record["arrival"]], device=self.learner.device))
        generator = torch.Generator(device=self.learner.device).manual_seed(
            derive_component_seed(self.seed, record["record_id"]))
        return pm.model_fisher_diagonal(
            lambda p: predict_histories(self.model, p, batch.histories), self.ltm,
            self.learner.design(batch), method="sampled", draws=self.draws, generator=generator)

    def save(self):
        if self._parameter_hash(self.learner.params) != self.identity["all_parameters_sha256"]:
            raise ValueError("reference moved the accepted model")
        self.s["rng"] = self._global_rng(self.learner.device)
        if self.checkpoint_fields is not None:
            self.s.update(self.checkpoint_fields())
        self.store.checkpoint(self.s, progress={"schema_version": SCHEMA, "phase": 3,
            "status": "initialization independent MC reference: " + self.s["stage"],
            "counts": {"step": self.s["count"], "target": len(self.records),
                "histories": self.s["count"], "draws_per_history": self.draws,
                "model_score_draws": self.s["count"] * self.draws},
            "anchor_sha256": self.anchor_hash, "reference_is_oracle": False})
        self._saved_count, self._saved_time = self.s["count"], time.monotonic()
        self.stopped()

    def run(self):
        import torch
        if self.s["stage"] == "done":
            return self.summary()
        if getattr(self.store, "completed", False):
            raise ValueError("completed reference cannot compute scores")
        if "rng" not in self.s:
            self.save()
        subset = set(self.primary["fisher"]["subset_ids"])
        while self.s["count"] < len(self.records):
            if self.before_record is not None:
                self.before_record()
            record = self.records[self.s["cursor"]]
            contribution = self._one(record)
            if set(contribution) != set(self.ltm) or any(
                    v.shape != self.ltm[n].shape or not bool((torch.isfinite(v) & (v >= 0)).all())
                    for n, v in contribution.items()):
                raise FloatingPointError("invalid raw reference Fisher contribution")
            sums = {n: self.s["sums"][n] + v.detach().cpu().double() for n, v in contribution.items()}
            subset_sum = self.s["subset_sum"]
            if record["record_id"] in subset:
                subset_sum = {n: subset_sum[n] + v.detach().cpu().double() for n, v in contribution.items()}
            # An asynchronous deadline cannot publish half an admitted history.
            self.s = {**self.s, "sums": sums, "subset_sum": subset_sum,
                "cursor": self.s["cursor"] + 1, "count": self.s["count"] + 1}
            if (self.s["count"] - self._saved_count >= self.block_items
                    or time.monotonic() - self._saved_time >= self.block_seconds):
                self.save()
        self.s = {**self.s, "stage": "done", "completed_at": utc_now(),
            "reference_diagonal": {n: v / len(self.records) for n, v in self.s["sums"].items()}}
        self.save()
        return self.summary()

    def summary(self):
        import torch
        from numerical_experiment.picar_initialization.numerics import _disagreement, _trace

        if self.s["stage"] != "done" or self.s["count"] != len(self.records):
            raise ValueError("incomplete reference has no completed comparison")
        if self._parameter_hash(self.learner.params) != self.identity["all_parameters_sha256"]:
            raise ValueError("reference moved the accepted model")
        reference, count = self.s["reference_diagonal"], len(self.records)
        primary = self.primary["learner"]["diagonal"]

        def compare(left, right):
            left = {n: v.detach().cpu().double() for n, v in left.items()}
            return {"primary_trace": _trace(left), "reference_trace": _trace(right),
                **_disagreement(left, right, 1e-12), "relative_denominator": "reference_L2_norm"}

        comparisons = {"primary_4_vs_reference": compare(primary, reference)}
        nested = self.primary["fisher"]["nested_sums"]
        if "2" in nested:
            two = {n: v / count for n, v in nested["2"].items()}
            comparisons["primary_2_vs_reference"] = compare(two, reference)
            comparisons["nested_primary_2_vs_primary_4"] = compare(two, {n: v.double() for n, v in primary.items()})
        subset_count = len(self.primary["fisher"]["subset_ids"])
        if subset_count:
            subset = {n: v / subset_count for n, v in self.s["subset_sum"].items()}
            for key, label in (("subset_sum", "primary_4_subset_vs_reference"),
                               ("independent_sum", "original_independent_4_subset_vs_reference")):
                comparisons[label] = compare({n: v / subset_count for n, v in self.primary["fisher"][key].items()}, subset)
        return {"schema_version": SCHEMA, "version": VERSION, "kind": "initialization_mc_reference",
            "phase": 3, "status": "completed", "smoke_only": True, "development_only": True,
            "counts": {"N": count, "M": self.draws, "histories": count,
                "model_score_draws": count * self.draws, "new_observations": 0, "validation_histories": 0},
            "anchor_sha256": self.anchor_hash, "seeds": copy.deepcopy(self.s["seeds"]),
            "execution": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "device": str(self.learner.device), "dtype": str(self.learner.dtype)},
            "normalization": {"history_weight": 1. / count, "draw_weight": 1. / self.draws,
                "reference_per_history_mean": "(1/N) sum_i (1/M) sum_m score_L(i,m)^2",
                "summed_information_multiplier": count, "pi": None, "ridge": 0., "projection_or_floor": False},
            "reference_per_history_mean": {"artifact": "final journal state.pt", "key": "reference_diagonal",
                "dtype": "torch.float64", "trace": _trace(reference)},
            "score_convention": "s_L = grad_L log p = -grad_L NLL; squared per-model-label, not observed-label or batch-mean gradients",
            "comparisons": comparisons, "primary_unchanged": True, "model_parameters_unchanged": True,
            "reference_is_oracle": False,
            "interpretation": "Independent model-label MC estimate at the same fixed histories/anchor, not ground truth. Nested primary 2/4 draws are dependent; the new stream is independent. No new conditions or inference about history-sampling variability.",
            "started_at": self.s["started_at"], "completed_at": self.s["completed_at"]}


def _load_model(config, source, model_path, offline):
    from numerical_experiment.picar.models import load_qwen_model
    from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent

    expected = source["bootstrap"]["summary"]["execution"]["model_provenance"]
    protocol = source["main"]["stored_config"]["protocol"]
    raw = load_qwen_model(model_path if model_path is not None else config.model.model_id,
        revision=config.model.revision, local_files_only=offline, seed=protocol["model_seed"],
        quantized=config.model.quantized, device=config.runtime.device,
        top_layers=config.model.top_layers, ltm_rank=config.model.ltm_rank, stm_rank=config.model.stm_rank,
        hidden_size=config.model.hidden_size, context_length=config.model.context_length,
        return_bins=config.model.return_bins, min_pixels=expected["processor_pixels"][0],
        max_pixels=expected["processor_pixels"][1], max_tokens=expected["max_tokens"])
    actual = raw.encoder.provenance
    for key in ("checkpoint_manifest", "compute_policy", "selected_layers", "adapter_targets",
                "factor_scales", "seeds", "quantization", "software"):
        if canonical_json(actual[key]) != canonical_json(expected[key]):
            raise ValueError("reference backbone identity differs: " + key)
    raw.encoder.enable_prefix_cache = True
    return NormalizedStudent(raw, mode=protocol["normalization"], eps=1e-6).eval()


def run(source_campaign, output, *, model_path=None, offline=False, resume=False, draws=16):
    """CLI entry: pending without GPU if main is unsealed or budget insufficient."""
    started = time.monotonic()
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    root, output = Path(source_campaign).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("reference output must be a new unit outside the frozen campaign")
    if (output / "COMPLETED").is_file():
        if not resume:
            raise FileExistsError("reference completed; use --resume or a new unique output")
        unit = read_completed_unit(output)
        identity = unit["stored_config"]
        if identity["source_campaign"] != str(root) or identity["draws"] != draws:
            raise ValueError("completed reference request identity differs")
        return unit["summary"]
    source = inspect_source(root)
    if source is None:
        return _pending("main campaign is not sealed; no GPU work admitted")
    main_charge = source["main_charge"]
    allowance = min(MAX_SECONDS, SHARED_SECONDS - main_charge)
    primary_seconds = source["bootstrap"]["summary"]["active_seconds"]
    estimate = 1.15 * primary_seconds * draws / 4. + PREPARATION_SECONDS + CLOSEOUT_SECONDS
    if allowance < estimate:
        return _pending("remaining shared budget cannot admit conservative model-load/reference/closeout estimate",
                        main_charge=main_charge, active=time.monotonic() - started)
    from numerical_experiment.picar.storage import RunStore

    identity = {"schema_version": SCHEMA, "version": VERSION,
        "source_campaign": str(root), "draws": draws,
        "main_completion_sha256": sha256_file(root / "orchestration/COMPLETED"),
        "bootstrap_completion_sha256": sha256_file(Path(source["bootstrap"]["path"]) / "COMPLETED"),
        "fit_completion_sha256": sha256_file(Path(source["fit"]["path"]) / "COMPLETED"),
        "source_sha256": sha256_file(Path(__file__)), "mainrepaircharged_seconds": main_charge,
        "allowance_seconds": allowance, "closeout_reserve_seconds": CLOSEOUT_SECONDS,
        "model_source": str(model_path) if model_path is not None else "pinned_project_cache_default"}
    stop = StopController(output)
    stop.install()
    try:
        with RunStore(output, identity, resume=resume, checkpoint_policy="journal-v1") as store:
            saved = store.load_checkpoint(trusted=True)
            base = saved.get("charged_seconds", 0.) if saved else 0.
            base_active = saved.get("active_seconds", 0.) if saved else 0.
            if not 0 <= base <= allowance or not 0 <= base_active <= base:
                raise ValueError("reference recovery resource ledger differs")

            def spent():
                return base + time.monotonic() - started

            def observed():
                return base_active + time.monotonic() - started

            def fields():
                return {"active_seconds": observed(),
                    "charged_seconds": min(allowance, spent() + BLOCK_SECONDS),
                    "mainrepaircharged_seconds": main_charge, "allowance_seconds": allowance}

            def checkpointed():
                if stop.check():
                    raise InterruptedError("reference STOP requested after committed block")
                if spent() + CLOSEOUT_SECONDS >= allowance:
                    raise ReferenceLimit("reference closeout reserve reached")

            def next_block():
                checkpointed()
                # Renew the alarm only after a durable cursor/sum transaction.
                signal.setitimer(signal.ITIMER_REAL, min(BLOCK_SECONDS, allowance - spent() - CLOSEOUT_SECONDS))

            reference = None
            try:
                if spent() + PREPARATION_SECONDS + CLOSEOUT_SECONDS >= allowance:
                    return _pending("resumed reference lacks preparation/closeout allowance", main_charge=main_charge,
                                    active=observed(), charged=spent())
                preparation = {**(saved or {}), "stage": (saved or {}).get("stage", "preparing"),
                    "started_at": (saved or {}).get("started_at", utc_now()),
                    "active_seconds": observed(), "charged_seconds": spent() + PREPARATION_SECONDS}
                store.checkpoint(preparation, progress={"schema_version": SCHEMA, "phase": 3,
                    "status": "independent MC reference: preparing; primary unchanged"})
                checkpointed()
                with _deadline(min(PREPARATION_SECONDS, allowance - spent() - CLOSEOUT_SECONDS)):
                    # Verify all scientific dependencies before loading Qwen.
                    for name, expected in source["main"]["stored_config"]["source_sha256"].items():
                        if sha256_file(PROJECT / name) != expected:
                            raise ValueError("frozen scientific source differs: " + name)
                    fitted, bootstrap = _completed_state(source["fit"]), _completed_state(source["bootstrap"])
                    records, data_identity = reconstruct_training(source, fitted, bootstrap)
                    config = config_from_dict(bootstrap["identity"]["config"])
                    import torch
                    torch.set_num_threads(config.runtime.threads)
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA unavailable; reference CLI never silently falls back to CPU")
                    model = _load_model(config, source, model_path, offline)
                    reference = FisherReference(model, config, records, bootstrap, store, next_block,
                        draws=draws, block_seconds=BLOCK_SECONDS / 2,
                        checkpoint_fields=fields, before_record=checkpointed)
                    reference.s["data_identity"] = data_identity
                # Each checkpoint precharges one future block. Alarm bounds that
                # uncommitted block on process stalls; record admission is atomic.
                while reference.s["stage"] != "done":
                    with _deadline(min(BLOCK_SECONDS, allowance - spent() - CLOSEOUT_SECONDS)):
                        reference.run()
                result = reference.summary()
            except (ReferenceLimit, InterruptedError) as exc:
                active = observed()
                closing = {**fields(), "active_seconds": active,
                    "charged_seconds": min(allowance, spent() + CLOSEOUT_SECONDS)}
                result = _pending(str(exc), main_charge=main_charge, active=active,
                                  charged=closing["charged_seconds"])
                result["counts"] = {"histories": reference.s["count"] if reference else 0, "M": draws}
                with _deadline(max(.001, allowance - spent())):
                    if reference is not None:
                        reference.stopped = lambda: None
                        reference.checkpoint_fields = lambda: closing
                        reference.save()
                    else:
                        state = store.load_checkpoint(trusted=True) or preparation
                        store.checkpoint({**state, **closing})
                return result
            # Charge a fixed closeout/report reserve, not the observed partial
            # duration of the final fsync. Completed artifacts stay immutable.
            active = observed()
            charge = spent() + CLOSEOUT_SECONDS
            if charge > allowance:
                raise ReferenceLimit("reference completion would exceed shared allowance")
            result.update(active_seconds=active, sidecar_charged_seconds=charge,
                mainrepaircharged_seconds=main_charge, combinedrepaircharged_seconds=main_charge + charge,
                allowance_seconds=allowance, closeout_reserve_charged_seconds=CLOSEOUT_SECONDS,
                source_campaign=str(root), initial_asset=source["bootstrap"]["path"],
                selected_fit=source["fit"]["path"], admission_estimate_seconds=estimate,
                resource_policy="Observed active seconds plus conservatively precharged interrupted preparation/blocks; reserve 60 seconds for closeout/reporting. No allowance reset on resume.")
            reference.s.update(active_seconds=active, charged_seconds=charge, summary=copy.deepcopy(result))
            with _deadline(max(.001, allowance - spent())):
                # Do not call save(): fields() deliberately reserves an *open*
                # next block, whereas this transaction is the final closeout.
                store.checkpoint(reference.s, progress={"schema_version": SCHEMA, "phase": 3,
                    "status": "independent MC reference complete; primary unchanged",
                    "counts": result["counts"]})
                store.complete(result)
            return result
    finally:
        stop.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--draws", type=int, default=16)
    args = parser.parse_args()
    result = run(args.source_campaign, args.output, model_path=args.model_path,
                 offline=args.offline, resume=args.resume, draws=args.draws)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
