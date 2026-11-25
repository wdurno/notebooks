"""Restore recorded Fix 5 paths; budget contributions do not change treatment."""

import argparse
import json
from pathlib import Path

from numerical_experiment.picar_head_awac.config import DEFAULT_OUTPUT
from numerical_experiment.picar.storage import _atomic_bytes


def recorded_options(root, *, max_hours=None, stage="all", continue_from=None):
    root=Path(root).resolve()
    path=(Path(continue_from).resolve() if continue_from else root)/"orchestration/config.json"
    if not path.is_file():
        raise FileNotFoundError("No Fix 5 configuration at "+str(path))
    config=json.loads(path.read_text())
    if config.get("numerical_revision") not in ("head-awac-v2","head-awac-v3"):
        raise ValueError("launcher requires the corrected Fix 5 configuration")
    limit_path=root/"runtime_budget.json"
    if not limit_path.exists() and continue_from:
        limit_path=Path(continue_from)/"runtime_budget.json"
    recorded=json.loads(limit_path.read_text()) if limit_path.exists() else {"max_hours":13.}
    limit=recorded["max_hours"] if max_hours is None else max_hours
    options={"output":root,"source":Path(config["source"]),
            "model_path":None if config["model_path"] is None else Path(config["model_path"]),
            "reuse_profile":None if config["reuse_profile"] is None else Path(config["reuse_profile"]),
            "resume":True,"offline":bool(config["model_path"]),"max_hours":limit,"stage":stage}
    lineage=continue_from or config.get("continuation",{}).get("source")
    if lineage:
        options["continue_from"]=Path(lineage).resolve()
    return options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--max-hours",type=float)
    parser.add_argument("--continue-from",type=Path)
    parser.add_argument("--stage",choices=("profile","development","all"),default="all")
    parser.add_argument("--clear-stop",action="store_true")
    parser.add_argument("--resume",action="store_true",help="Accepted explicitly; resumption is this launcher's default.")
    args=parser.parse_args()
    completed=args.output/"orchestration/COMPLETED"
    if completed.exists():
        print((args.output/"orchestration/summary.json").read_text())
        return
    kwargs=recorded_options(args.output,max_hours=args.max_hours,stage=args.stage,continue_from=args.continue_from)
    if not 0 < kwargs["max_hours"] <= 20:
        raise ValueError("compute limit must be in (0,20] hours")
    args.output.mkdir(parents=True,exist_ok=True)
    _atomic_bytes(args.output/"runtime_budget.json",
                  (json.dumps({"max_hours":kwargs["max_hours"]})+"\n").encode())
    if args.clear_stop:
        (args.output/"STOP").unlink(missing_ok=True)
    from numerical_experiment.picar_head_awac.campaign import run
    print(json.dumps(run(**kwargs),indent=2),flush=True)


if __name__=="__main__":
    main()
