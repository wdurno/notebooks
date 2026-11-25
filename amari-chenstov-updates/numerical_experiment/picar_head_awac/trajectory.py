"""Resumable A-only episodes, episode-ready labels and bounded replay."""

import copy
from dataclasses import asdict
import time

import torch

from numerical_experiment.picar.control import ControlSession
from numerical_experiment.picar.fix3_student import _History
from numerical_experiment.picar.fix2_data import observation_row
from numerical_experiment.picar.fix2_pipeline import eligible_history
from numerical_experiment.picar.runtime import ModelAccess
from src.continual.picar_math import discounted_returns, time_discounts
from src.robotics.worker import SimulatorWorker
from src.seeding import derive_component_seed
from .config import configuration, simulator
from .data import append_history, pack_history, unpack_history
from .likelihood import data_loss


class Trajectory:
    def __init__(self, learner, features, scorer, store, check, *, seed, total=1024,
                 evaluation_episodes=8, initial_episodes=4, development=False,
                 initial_evaluations=None, midpoint_episodes=None):
        self.learner, self.features, self.scorer, self.store, self.check = learner, features, scorer, store, check
        self.seed, self.total = seed, total
        self.eval_count, self.initial_count = evaluation_episodes, initial_episodes
        self.mid_count = min(4,evaluation_episodes) if midpoint_episodes is None else midpoint_episodes
        self.development = development
        self.cfg = configuration("width6_signed_square")
        self.s = store.load_checkpoint(trusted=True) or {
            "stage": "evaluate", "n": 0, "target": 0, "episodes": [], "evaluations": [],
            "metrics": [], "replay": [], "ready": [], "retired_ids": [], "current": None,
            "session": None, "history": None, "episode_counter": 0, "round": 0,
            "replay_rng": torch.Generator().manual_seed(derive_component_seed(seed,"replay")).get_state(),
            "label_cursor": 0, "acquired_total": 0, "timings": {"scoring":0., "fitting":0., "collection":0.}}
        if "learner" in self.s:
            learner.load_state_dict(self.s["learner"])
        elif initial_evaluations:
            self.s["evaluations"]=copy.deepcopy(initial_evaluations)
            self.s["stage"]="collect"

    def save(self, session=None, history=None):
        if session is not None:
            snapshot = session.snapshot(capture_external=lambda: {"history": history.snapshot(),
                "learner": self.learner.state_dict()})
            self.s["session"] = snapshot
            self.s["history"] = snapshot["external"]["history"]
            self.s["learner"] = snapshot["external"]["learner"]
        else:
            self.s["learner"] = self.learner.state_dict()
        self.store.checkpoint(self.s, progress={"stage": self.s["stage"],"n":self.s["n"],
            "total":self.total,"fits":len(self.s["metrics"]),"method":self.learner.method,
            "rank":self.learner.recipe.rank,"evaluation_panels":len(self.s["evaluations"]),
            "latest_fit": self.s["metrics"][-1] if self.s["metrics"] else None,
            "evaluation_preview":[{"n":p["n"],"environment":p["environment"],
                "episodes":len(p["episodes"]),
                "success":sum(e["success"] for e in p["episodes"])/len(p["episodes"]),
                "raw_return":sum(e["raw_return"] for e in p["episodes"])/len(p["episodes"])}
                for p in self.s["evaluations"] if p["episodes"]]})
        self.check()

    def episode(self, seed, *, environment=None, video=False):
        s = self.s
        evaluation = environment is not None
        cfg = simulator(s["n"], self.total, fixed=environment)
        if s["current"] is not None:
            from src.robotics.simulator import SimulatorConfig
            cfg = SimulatorConfig.from_dict(s["current"]["simulator_config"])
        history = _History(self.cfg, cfg, s["history"])
        access = ModelAccess()
        access.version = self.learner.version
        worker = SimulatorWorker(cfg, run_id=f"head-awac-{self.seed}")
        def infer(observation, version):
            causal = history(observation)
            with history.lock:
                # The returned causal batch travels with its action receipt.
                # Old inference batches are not an extra replay store.
                history.state["histories"].clear()
            feature = self.features(causal)
            z = self.learner.choose(feature, explore=not evaluation)
            return {"logits":z,"probabilities":torch.cat((z,torch.zeros(1))).softmax(-1),
                    "feature":feature,"causal":causal,"model_version":version}
        session = ControlSession(worker, access, infer=infer, pending_capacity=512, own_worker=True)
        try:
            if s["session"] is not None:
                session.restore(s["session"])
                if not s["session"]["terminal"]:
                    session.resume()
            else:
                s["current"] = {"seed":seed,"evaluation":evaluation,"environment":environment,
                    "n":s["n"],"rows":[],"scores":[],"complete":False,"frames":[],
                    "episode_id":f"{'eval' if evaluation else 'train'}:{seed}:{s['episode_counter']}",
                    "simulator_config":asdict(cfg)}
                session.start(seed=seed, exposure_intervals=float(s["n"]), training=not evaluation)
                s["current"]["layout"] = worker.diagnostics()["layout"]
                self.save(session,history)
            episode = s["current"]
            if episode["complete"]:
                return episode
            while True:
                interval = session.next_interval(timeout=120.)
                result, action = interval.result, interval.action
                post = observation_row(result.observation,previous_time=interval.observation.capture_time_s)
                row = {"frame_ref":self.store.frames.put(post["rgb"]),"elapsed":result.reward_components.elapsed_s,
                    "bonus":result.reward_components.terminal_bonus,"terminal":result.terminated,
                    "truncated":result.truncated,"end_reason":result.end_reason,
                    "fallback":interval.fallback_reason,"timing":interval.timing,
                    "action":None if action is None else action["logits"].cpu(),
                    "feature":None if action is None else action["feature"].cpu()}
                if action is not None:
                    causal = action["causal"]
                    nxt = append_history(causal,post,self.cfg.model.context_length)
                    row["history"] = pack_history(causal,self.store.frames)
                    row["next_history"] = pack_history(nxt,self.store.frames)
                    row["next_feature"] = None
                if not evaluation:
                    row["arrival"] = s["n"]
                    s["n"] += 1
                episode["rows"].append(row)
                episode["scores"].append(None)
                if video:
                    episode["frames"].append(worker.snapshot()["backend"])
                if result.terminated or result.truncated:
                    session.pause(reason="episode_boundary")
                    episode.update(complete=True,success=result.end_reason=="success",
                                   end_reason=result.end_reason)
                    self.save(session,history)
                    break
                self.save(session,history)
            return episode
        finally:
            session.close(timeout=120.)

    def label(self):
        ep = self.s["current"]
        if not ep["complete"] or ep["rows"][-1]["truncated"]:
            raise RuntimeError("incomplete/infrastructure episode cannot receive task returns")
        for i,row in enumerate(ep["rows"]):
            if ep["scores"][i] is None:
                tick=time.monotonic()
                ep["scores"][i] = self.scorer.score_rgb(self.store.frames.get(row["frame_ref"]))
                self.s["timings"]["scoring"] += time.monotonic()-tick
                self.save()
            if row.get("history") and row["next_feature"] is None:
                row["next_feature"] = self.features(unpack_history(row["next_history"],self.store.frames))
                self.save()
        dt=torch.tensor([r["elapsed"] for r in ep["rows"]],dtype=torch.float64)
        q=torch.tensor([r["score"] for r in ep["scores"]],dtype=torch.float64)
        b=torch.tensor([r["bonus"] for r in ep["rows"]],dtype=torch.float64)
        raw=dt*(.01*q.square()-.04)+b*b.abs()
        terminal=torch.tensor([r["terminal"] for r in ep["rows"]])
        discount=time_discounts(dt)
        returns=discounted_returns(raw,discount,terminal)
        ep.update(raw_stage_rewards=raw.tolist(),coded_stage_rewards=(raw/10).tolist(),
                  raw_returns=returns.tolist(),coded_returns=(returns/10).tolist(),
                  raw_return=float(returns[0]),coded_return=float(returns[0]/10),
                  raw_qwen_mean=float(q.mean()))
        records=[]
        for i,row in enumerate(ep["rows"]):
            if row["action"] is None:
                continue
            records.append({"id":ep["episode_id"]+f":{i}","episode_id":ep["episode_id"],
                "feature":row["feature"],"next_feature":row["next_feature"],"action":row["action"],
                "reward":(raw[i]/10).float(),"return":(returns[i]/10).float(),
                "discount":(discount[i]*(~terminal[i])).float(),
                "arrival":row.get("arrival",0),"visits":0,
                "history":row["history"],"next_history":row["next_history"]})
        if ep["evaluation"]:
            # Refresh targets on a copy: held-outs must not change learner RNG/state.
            before=copy.deepcopy(self.learner.state_dict())
            try:
                data,design,_=self.learner.design(records)
                output=self.learner.coordinates.predict(self.learner.vector,data["feature"],data["action"])
                losses=data_loss(output,data["action"],design)
                ep["heldout_nll"]=float(losses["nll"].mean().detach())
                q_pred=(output["q1"]+output["q2"])/2
                ep["critic_mse"]=float((q_pred-data["return"]).square().mean().detach())
            finally:
                self.learner.load_state_dict(before)
        return records

    def finish_episode(self):
        self.s["current"],self.s["session"],self.s["history"]=None,None,None
        self.s["episode_counter"]+=1
        self.features.clear()

    def trim(self):
        eligible=self.s["replay"]
        for record in eligible:
            times={float(r["history"]["decision_time"][0,-1]) for r in eligible
                   if r["episode_id"]==record["episode_id"]}
            for raw_key,feature_key in (("history","feature"),("next_history","next_feature")):
                packed=record[raw_key]
                packed_times=packed["decision_time"][0].tolist()
                if raw_key=="next_history":
                    times=times|{float(record["history"]["decision_time"][0,-1])}
                first=len(packed_times)-1
                while first>0 and packed_times[first-1] in times:
                    first-=1
                if first==0:
                    continue
                raw=unpack_history(packed,self.store.frames)
                trimmed=eligible_history(raw,times)
                if trimmed["images"].shape[1]!=raw["images"].shape[1]:
                    record[raw_key]=pack_history(trimmed,self.store.frames)
                    record[feature_key]=self.features(trimmed)
        self.features.clear()

    def work(self):
        s=self.s
        while s["ready"]:
            current=next(r for r in s["replay"] if r["id"]==s["ready"][0])
            while s["round"]<self.learner.recipe.rounds:
                rng=torch.Generator()
                rng.set_state(s["replay_rng"])
                others=[r for r in s["replay"] if r is not current and r["visits"]>0]
                indices=torch.randperm(len(others),generator=rng)[:self.learner.recipe.batch_size]
                selected=[current]+[others[int(i)] for i in indices]
                # Accepted event commits learner, selection RNG, visits and round together.
                metrics=self.learner.fit(selected,s["n"],check=self.check)
                s["replay_rng"]=rng.get_state()
                for record in selected:
                    record["visits"]+=1
                s["round"]+=1
                metrics["visits"]=sum(r["visits"] for r in selected)/len(selected)
                s["metrics"].append(metrics)
                s["timings"]["fitting"]+=metrics["update_seconds"]
                self.save()
            s["ready"].pop(0)
            s["round"]=0
            while len(s["replay"])>self.learner.recipe.capacity and s["replay"][0]["visits"]>0:
                old=s["replay"][0]
                if old["id"] in s["retired_ids"]:
                    raise ValueError("duplicate EWC admission")
                self.learner.retire(old)
                s["retired_ids"].append(old["id"])
                s["replay"].pop(0)
                self.trim()
                self.save()
            self.save()

    def evaluate(self):
        n=self.s["target"]
        count=self.initial_count if n==0 else self.eval_count if n==self.total else self.mid_count
        for environment in ("initial_A","moderate_A"):
            panel=next((p for p in self.s["evaluations"] if p["n"]==n and p["environment"]==environment),None)
            if panel is None:
                panel={"n":n,"acquired_count":self.s["n"],"environment":environment,"episodes":[]}
                self.s["evaluations"].append(panel)
            for i in range(len(panel["episodes"]),count):
                seed=derive_component_seed(199601 if not self.development else 199701,f"{environment}:{i}")
                self.episode(seed,environment=environment,video=i==0 and n in (0,self.total))
                self.label()
                ep=self.s["current"]
                if ep["frames"]:
                    from types import SimpleNamespace
                    from numerical_experiment.picar_evaluation_video import render_video
                    ep["videos"]=render_video(SimpleNamespace(store=self.store),ep,environment)
                panel["episodes"].append({k:v for k,v in ep.items() if k not in ("rows","frames")})
                self.finish_episode()
                self.save()
        self.s["stage"]="done" if n==self.total else "collect"
        self.save()

    def run(self):
        while self.s["stage"]!="done":
            stage=self.s["stage"]
            if stage=="evaluate":
                self.evaluate()
            elif stage=="collect":
                if self.s["current"] is None and self.s["n"]>=self.total:
                    self.s.update(stage="evaluate",target=self.total)
                elif (self.s["current"] is None and self.s["n"]>=self.total//2
                      and not any(p["n"]==self.total//2 for p in self.s["evaluations"])):
                    self.s.update(stage="evaluate",target=self.total//2)
                else:
                    seed=derive_component_seed(self.seed,f"episode:{len(self.s['episodes'])}")
                    self.episode(seed)
                    self.s["stage"]="label"
                self.save()
            elif stage=="label":
                records=self.label()
                known={r["id"] for r in self.s["replay"]}
                for r in records:
                    if r["id"] not in known:
                        self.s["replay"].append(r)
                        self.s["ready"].append(r["id"])
                self.s["episodes"].append({k:v for k,v in self.s["current"].items() if k not in ("frames","rows")})
                self.finish_episode()
                self.s["stage"]="fit"
                self.save()
            elif stage=="fit":
                self.work()
                self.s["stage"]="collect"
                self.save()
            else:
                raise ValueError("unknown trajectory stage: "+stage)
        return {"method":self.learner.method,"seed":self.seed,"n":self.s["n"],
                "metrics":self.s["metrics"],"evaluations":self.s["evaluations"],
                "episodes":self.s["episodes"],"archive":self.learner.archive.diagnostics(),
                "retired":len(self.s["retired_ids"]),"timings":self.s["timings"]}
