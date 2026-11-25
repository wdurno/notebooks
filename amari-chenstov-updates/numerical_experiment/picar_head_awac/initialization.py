"""Equal-weight head initialization from sealed stationary A observations."""

import copy
import math
import time
import torch

from numerical_experiment.picar.storage import FrameStore
from numerical_experiment.picar.fix2_data import history_batch, TASK
from .data import source_data, apply_parameters, append_history, feature_record
from .model import Heads, Coordinates
from .likelihood import fisher_factor
from .learner import Learner, Recipe, batch


def prepare(model, source, features, store, check):
    data,labeled=source_data(source)
    apply_parameters(model,data["distilled_learner"])
    model.requires_grad_(False)
    features.clear()
    state=store.load_checkpoint(trusted=True) or {"train":[],"validation":[],"cursor":0}
    from numerical_experiment.picar_phase2.campaign import read_json
    scores_path=next((source/"units").glob("scoring-*/summary.json"))
    scores=read_json(scores_path)["scores"]
    episodes={}
    for item in data["episodes"]:
        rows=item["episode"]["intervals"]
        for index,row in enumerate(rows):
            episodes[(item["episode"]["episode_id"],row["arrival"])]=(item,index)
    for split in ("train","validation"):
        for record in labeled[split][len(state[split]):]:
            item,index=episodes[(record["episode_id"],record["source_arrival"])]
            ep=item["episode"]
            row=ep["intervals"][index]
            rgb=FrameStore(item["frames_root"],readonly=True).get(row["frame_ref"])
            at=float(record["history"]["decision_time"][0,-1])+row["elapsed_s"]
            post={"rgb":rgb,"time":at,"elapsed":row["elapsed_s"],
                  "action":tuple(row["controls"].values()) if isinstance(row["controls"],dict) else row["controls"]}
            nxt=append_history(record["history"],post)
            # Derive stage reward from complete labeled returns to avoid a
            # second scorer or a changed score-to-reward transformation.
            gamma=.99**(row["elapsed_s"]/.5)
            b=ep["terminal_bonuses"][index]
            q=scores[row["frame_ref"]]["score"]
            reward=(row["elapsed_s"]*(.01*q*q-.04)+b*abs(b))/10
            feature=features(record["history"])
            state[split].append(feature_record(record,feature,next_feature=features(nxt),reward=reward,
                                               discount=0. if row["terminated"] else gamma))
            state["cursor"]+=1
            store.checkpoint(state,progress={"stage":"frozen initialization features","n":state["cursor"],"total":580})
            check()
    state["source_parameters"]=copy.deepcopy(data["distilled_learner"]["parameters"])
    store.checkpoint(state,progress={"stage":"feature dataset complete","n":580,"total":580})
    return state


def fit(prepared, store, check, device="cpu"):
    heads=Heads().to(device)
    coordinates=Coordinates(heads)
    source=prepared["source_parameters"]
    with torch.no_grad():
        weight=source["action_head.ltm.weight"].clone()
        if "action_head.stm.B" in source:
            weight+=source["action_head.stm.B"]@source["action_head.stm.A"]
        heads.actor.ltm.weight.copy_(weight.to(device))
        heads.actor.ltm.bias.copy_(source["action_head.ltm.bias"].to(device))
    saved=store.load_checkpoint(trusted=True)
    optimizer=torch.optim.Adam([p for n,p in heads.named_parameters() if n.startswith("q") and ".ltm." in n],lr=.003)
    if saved:
        heads.load_state_dict(saved["heads"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch=saved["epoch"]
        curves=saved["curves"]
        best=copy.deepcopy(saved["best"])
    else:
        epoch,curves=0,[]
        best={"mse":float("inf"),"epoch":0,"heads":copy.deepcopy(heads.state_dict())}
    train=batch(prepared["train"],device)
    validation=batch(prepared["validation"],device)
    while epoch<160:
        heads.train()
        # Deterministic full-batch fitting is cheap for these cached heads.
        optimizer.zero_grad()
        out=heads(train["feature"],train["action"])
        loss=((out["q1"]-train["return"]).square()+(out["q2"]-train["return"]).square()).mean()/2
        loss.backward()
        optimizer.step()
        epoch+=1
        with torch.no_grad():
            outv=heads(validation["feature"],validation["action"])
            val=((outv["q1"]-validation["return"]).square()+(outv["q2"]-validation["return"]).square()).mean()/2
            value=float(val)
        if value<best["mse"]:
            best={"mse":value,"epoch":epoch,"heads":copy.deepcopy(heads.state_dict())}
        curves.append({"epoch":epoch,"train_mse":float(loss.detach()),"validation_mse":value})
        state={"epoch":epoch,"heads":copy.deepcopy(heads.state_dict()),"optimizer":optimizer.state_dict(),
               "best":best,"curves":curves}
        if epoch%10==0:
            store.checkpoint(state,progress={"stage":"initial critic fit","epoch":epoch,"total":160,
                                            "validation_mse":value})
            check()
        if epoch>=60 and epoch-best["epoch"]>=30:
            break
    heads.load_state_dict(best["heads"])
    state.update(selected_heads=copy.deepcopy(heads.state_dict()),inventory=coordinates.inventory())
    store.checkpoint(state,progress={"stage":"initial heads fitted","epoch":epoch})
    return state


def initial_archive(head_state,records,recipe,store,check,device="cpu"):
    from .archive import Archive
    heads=Heads().to(device)
    heads.load_state_dict(head_state["selected_heads"])
    learner=Learner(heads,recipe,199001,"td_replay")
    saved=store.load_checkpoint(trusted=True)
    if saved:
        learner.load_state_dict(saved["learner"])
        cursor=saved["cursor"]
    else:
        cursor=0
    tick=time.monotonic()
    for i,record in enumerate(records[cursor:],cursor):
        data,design,_=learner.design([record],generator="fisher")
        draws=torch.randn((recipe.integration_draws,7),generator=learner.generators["fisher"],device=device)
        factor=fisher_factor(learner.coordinates,learner.vector,data["feature"][0],design.weights[0],
                              recipe.action_variance,recipe.critic_variance,draws)[learner.ltm]
        # Running mean includes every unique initial history equally.
        learner.archive.update(factor,alpha=i/(i+1),beta=1/(i+1))
        if (i+1)%16==0 or i+1==len(records):
            store.checkpoint({"cursor":i+1,"learner":learner.state_dict()},progress={
                "stage":"equal-weight initial Fisher","n":i+1,"total":len(records),
                "archive":learner.archive.diagnostics()})
            check()
    return {"learner":learner.state_dict(),"cursor":len(records),
            "archive":learner.archive.diagnostics(),"seconds":time.monotonic()-tick}
