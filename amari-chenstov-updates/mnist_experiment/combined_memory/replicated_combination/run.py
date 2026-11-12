"""Autonomous, immutable and resumable Plan 18 Phase 9 execution."""

import argparse
import json
import math
import os
from pathlib import Path
import time
import torch

from mnist_experiment.fisher_memory import artifacts
from mnist_experiment.fisher_memory.control import StopRequested,ResourceLimit,utc_now
from mnist_experiment.fisher_memory.learner import StopRequested as LearnerStop
from mnist_experiment.fisher_memory_grid_registration.data import MNISTSource
from mnist_experiment.fisher_memory_stm_capacity.runner import trajectory_id
from mnist_experiment.fisher_memory_regression.runtime import (
    read_json,setup_torch,provenance,dataset_manifest,content_hash,
)
from mnist_experiment.combined_memory.reproduction.audit import load_checkpoint,compare_array
from . import PROTOCOL
from .config import (REPO,FAMILY,HISTORY,ROOT,DATA,POLICIES,CALIBRATED,SEED,PROCESS_SCALES,
                     BOOTSTRAP_SEED,Policy,configuration)
from .runtime import authorize,atomic_json,sources,peer_seconds,RuntimeBudget,external_bytes
from .runner import Runner,record


class Study:
    def __init__(self,args,store,budget,history):
        self.args,self.store,self.budget,self.history=args,store,budget,history
        self.source=MNISTSource(args.data_root)
        self.runners={}
        self.records=[record(store,e['unit_id']) for e in store.iter_units(completed_only=True)
                      if e['metadata'].get('kind')=='grid_registration_trajectory']
        self.stage='9A numerical acceptance'
        self.state='running'; self.current=None; self.reason=None
        self.last_report=0.; self.new_paths=0

    def decision(self,name):
        unit='decision-'+name
        return read_json(self.store.unit_path(unit)/'result.json') if self.store.is_complete(unit) else None

    def seal(self,name,value):
        old=self.decision(name)
        if old is not None:
            if old!=value:
                raise artifacts.ImmutableError('sealed decision changed: '+name)
            return old
        unit='decision-'+name
        self.store.begin(unit,dict(kind='phase9_decision',name=name,protocol=PROTOCOL))
        self.store.write_json(unit,'result.json',value)
        self.store.complete(unit)
        return value

    def progress(self,force=False,**updates):
        self.budget.check()
        self.current=updates.get('current',self.current)
        fresh=[r for r in self.records if r['namespace']=='fresh']
        development=[r for r in self.records if r['namespace']=='development']
        blocks=sum({r['policy'] for r in fresh if r['replica']==i}==set(POLICIES)
                   for i in range(self.args.replicas))
        status=dict(status=self.state,stage=self.stage,current=self.current,reason=self.reason,
            pid=os.getpid(),heartbeat=utc_now(),fresh_completed=len(fresh),planned_fresh=6*self.args.replicas,
            fresh_blocks_completed=blocks,planned_blocks=self.args.replicas,
            development_completed=len(development),accepted_checkpoint=updates.get('novel_count'),
            charged_hours=self.budget.used/3600,remaining_hours=self.budget.remaining/3600,
            forecast_hours=0. if self.state=='completed' else self.forecast()/3600)
        self.store.write_json(None,'status.json',status,finalizing=True)
        if not force and time.monotonic()-self.last_report<900:
            return
        analysis=self.decision('analysis')
        if analysis is None and blocks>=2 and self.state!='failed':
            from .analysis import analyze
            analysis=analyze(fresh,draws=100,seed=BOOTSTRAP_SEED,partial=True,check=self.budget.check)
        design=self.decision('design')
        selected=(design or {}).get('selection',{})
        settings=selected.get('policies') or {p:Policy(p,math.sqrt(10) if p=='A' else 1.).mapping() for p in POLICIES}
        settings=dict(settings)
        settings.setdefault('R',Policy('R',process_scale=selected.get('process_scale',1e-4)).mapping())
        candidates=[]
        from .calibration import summarize_record
        grouped={}
        for row in development:
            grouped.setdefault((row['policy'],row['recipe_key']),[]).append(row)
        for (policy,key),rows in grouped.items():
            measured=[dict(replica=r['replica'],**summarize_record(r)) for r in rows]
            means={field:sum(r[field] for r in measured)/len(measured)
                   for field in ('Havg','H100','novel_auc','learning_seconds')}
            ridge=rows[0]['recipe']['ridge']
            candidate=dict(policy=policy,ridge=ridge,candidate_id=policy+'-'+key,
                members=sorted(r['replica'] for r in rows),n_members=len(rows),per_replica=measured,
                acquisition=means['novel_auc'],**means,
                selected=bool(design and (policy=='R' or settings[policy]['ridge']==ridge)))
            for name,members in (('first_pair',(0,1)),('second_pair',(2,3))):
                subset=[r for r in measured if r['replica'] in members]
                candidate[name]={field:sum(r[field] for r in subset)/len(subset) for field in means} if subset else None
            candidates.append(candidate)
        selected_rows=dict(selected.get('calibration',{}))
        if 'representation_reference' in selected:
            selected_rows['R']=selected['representation_reference']
        snapshot=dict(protocol=PROTOCOL,progress=status,settings=settings,analysis=analysis,
            fidelity=self.decision('fidelity') or dict(status='pending'),
            calibration=dict(status=(design or {}).get('status','pending'),selected=selected_rows,
                selection=selected,candidates=candidates,process_scale=self.decision('process-scale')),
            development=dict(candidates=candidates),diagnostics=self.diagnostics(fresh),
            changes=self.decision('changes') or [])
        self.store.write_json(None,'report_snapshot.json',snapshot,finalizing=True)
        from .report import publish
        publish(snapshot,self.args.notebook,self.store)
        self.last_report=time.monotonic()

    def diagnostics(self,fresh):
        complete={i for i in range(self.args.replicas)
                  if {r['policy'] for r in fresh if r['replica']==i}==set(POLICIES)}
        result={}
        for policy in ('C','D','E'):
            records=[r for r in fresh if r['replica'] in complete and r['policy']==policy]
            if not records:
                continue
            grouped=[]
            for record in records:
                groups={}
                for row in read_json(self.store.unit_path(record['physical_unit'])/'fits.json'):
                    groups.setdefault(row['group'],[]).append(row)
                grouped.append(groups)
            common=sorted(set.intersection(*(set(g) for g in grouped)))
            series=dict(observations=[])
            fields=(('applied_pi','kalman_velocity_gain_mean','motion_energy','noise_scale')
                    if policy in ('D','E') else ())
            if policy in ('C','E'):
                fields+=('lfu_relative_correction','negative_fraction')
            for field in fields:
                values=[]
                for group in common:
                    per_replica=[]
                    for mapping in grouped:
                        rows=mapping[group]
                        present=[r[field] for r in rows if r.get(field) is not None]
                        if not present:
                            break
                        # Action fields are fixed in a group; corrections average its segments.
                        per_replica.append(sum(present)/len(present))
                    if len(per_replica)!=len(grouped):
                        values=[];break
                    values.append(sum(per_replica)/len(per_replica))
                if values:
                    series[field]=values
            series['observations']=[sum(max(r['observations'] for r in g[group]) for g in grouped)/len(grouped)
                                    for group in common]
            if len(series)>1:
                result[policy]=series
        return result

    def runner(self,namespace):
        if namespace not in self.runners:
            cfg=configuration(self.history,namespace,self.args.device,self.args.smoke)
            self.runners[namespace]=Runner(cfg,self.store,self.source,self.budget,namespace,
                progress=self.progress,stop_at=self.args.smoke_stop_at)
        return self.runners[namespace]

    def job(self,namespace,replica,policy):
        runner=self.runner(namespace)
        partition='development' if namespace=='development' else 'confirmation'
        recipe=policy.recipe(runner.cfg)
        unit=runner.engine(policy).store.key(trajectory_id('head_r1',partition+'--grid',replica,recipe,runner.cfg.novel_count))
        old=next((r for r in self.records if r['physical_unit']==unit),None)
        if old is not None:
            return old
        self.current=unit
        self.progress()
        value=runner.run(policy,replica,partition)
        self.records.append(value); self.new_paths+=1
        print(json.dumps(dict(event='trajectory_completed',namespace=namespace,replica=replica,
            policy=policy.name,ridge=policy.ridge,novel_auc=value['novel_auc'],seconds=value['active_seconds'])),flush=True)
        if self.args.max_paths and self.new_paths>=self.args.max_paths:
            raise StopRequested('bounded invocation completed')
        return value

    def costs(self):
        defaults=dict(R=600.,A=450.,B=550.,C=800.,D=800.,E=1000.)
        for p in POLICIES:
            rows=[r['active_seconds'] for r in self.records if r['policy']==p]
            if rows:
                defaults[p]=max(rows)
        return defaults

    def forecast(self):
        costs=self.costs()
        completed={(r['replica'],r['policy']) for r in self.records if r['namespace']=='fresh'}
        return sum(costs[p] for i in range(self.args.replicas) for p in POLICIES if (i,p) not in completed)+1800.

    def reserve(self,extra=(),validation=True):
        costs=self.costs()
        remaining=self.forecast()-1800
        initial=self.args.replicas*120.
        remaining+=sum(costs[p] for p in extra)
        if validation:
            remaining+=2*sum(costs.values())
        required=1.25*(remaining+initial)+1800
        return dict(required_seconds=required,remaining_seconds=self.budget.remaining,
                    passed=required<=self.budget.remaining,measured_path_costs=costs)

    def fidelity(self):
        existing=self.decision('fidelity')
        if existing:
            if existing['status']!='passed':
                raise ValueError('failed fidelity seal')
            return
        if self.args.smoke:
            self.seal('fidelity',dict(status='passed',smoke_only=True))
            return
        self.seal('acceptance-contract',dict(historical_replica=0,
            tolerances=dict(rtol=1e-5,atol=1e-6,risk_atol=1e-4),
            fisher_comparison='represented operator on eight fixed probes and exact diagonal',
            source_hash=self.store.identity['source_hash']))
        self.stage='9A original F1 fidelity'
        current=self.job('acceptance',0,Policy('R'))
        old_paths=list((HISTORY/'units').glob('fresh-blur--head_r1-trajectory-confirmation--grid-0000-fd_replay_ewc-*-n0100'))
        if len(old_paths)!=1:
            raise ValueError('unique historical F1 replica zero required')
        old=old_paths[0]
        artifacts._unit_record(HISTORY,old.name,artifacts._identity(HISTORY)['identity_hash'],{},self.budget.check,True)
        new=self.store.unit_path(current['physical_unit'])
        old_metrics=read_json(old/'metrics.json'); new_metrics=read_json(new/'metrics.json')
        checks=[]
        for field in ('n','observations','base_observations'):
            if [r[field] for r in old_metrics]!=[r[field] for r in new_metrics]:
                raise ValueError('fidelity clock mismatch: '+field)
        for field in ('base_mse','novel_mse','base_nll','novel_nll','base_accuracy1','novel_accuracy1'):
            checks.append(compare_array([r[field] for r in old_metrics],[r[field] for r in new_metrics],field,atol=1e-4))
        old_fits=read_json(old/'fits.json'); new_fits=read_json(new/'fits.json')
        for field in ('step','n','observations','selected_replay_ids','selected_count','solver_reason'):
            if [r.get(field) for r in old_fits]!=[r.get(field) for r in new_fits]:
                raise ValueError('fidelity update mismatch: '+field)
        for field in ('initial_objective','final_objective','gradient_inf_norm'):
            checks.append(compare_array([r[field] for r in old_fits],[r[field] for r in new_fits],field))
        old_state=load_checkpoint(HISTORY,old.name)
        state=self.store.load_checkpoint(current['physical_unit'])
        for field in ('theta','anchor_L','ltm_indices'):
            checks.append(compare_array(old_state['learner'][field],state['learner'][field],field))
        def compare_fisher(a,b,label):
            generator=torch.Generator().manual_seed(SEED)
            probes=torch.randn((len(a['residual_diagonal']),8),dtype=torch.float64,generator=generator)
            def apply(value):
                factor=value['factor'].double(); diagonal=value['residual_diagonal'].double()
                return factor@(factor.T@probes)+diagonal[:,None]*probes
            checks.append(compare_array(apply(a),apply(b),label+'_operator'))
            checks.append(compare_array(a['factor'].double().square().sum(1)+a['residual_diagonal'].double(),
                                        b['factor'].double().square().sum(1)+b['residual_diagonal'].double(),label+'_diagonal'))
        compare_fisher(old_state['learner']['fisher_L'],state['learner']['fisher_L'],'final_fisher')
        old_summary=read_json(old/'summary.json')
        old_initial=load_checkpoint(HISTORY,'fresh-shared--'+old_summary['asset_id'])
        new_initial=self.store.load_checkpoint('acceptance-shared--'+current['asset_id'])
        checks.append(compare_array(old_initial['theta'],new_initial['theta'],'initial_theta'))
        compare_fisher(old_initial['learner']['fisher_L'],new_initial['learner']['fisher_L'],'initial_fisher')
        if old_state['retirements']!=state['retirements'] or old_state['learner']['replay']!=state['learner']['replay']:
            raise ValueError('fidelity retirement or FIFO mismatch')
        if current['plan_hash']!=old_summary['plan_hash']:
            raise ValueError('historical data plan changed')
        self.seal('fidelity',dict(status='passed',historical_unit=old.name,rerun=current['physical_unit'],checks=checks))

    def development_records(self):
        return [r for r in self.records if r['namespace']=='development']

    @staticmethod
    def policy(mapping):
        return Policy(mapping['policy'],float(mapping['ridge']),float(mapping['process_scale']))

    def calibrate(self):
        from . import calibration as cal
        design=self.decision('design')
        if design:
            return design
        self.stage='9B shadow observer calibration'
        selected=self.decision('process-scale')
        if selected is None:
            measurements=[]
            for replica in (0,1):
                result=self.runner('development').shadows(replica,'development',PROCESS_SCALES)
                measurements.extend(dict(namespace='development',replica=replica,process_scale=float(k),mean_log_score=v)
                                    for k,v in result['scores'].items())
            selected=self.seal('process-scale',cal.select_process_scale(measurements))
        scale=selected['process_scale']
        initial=cal.initial_recipes(scale)
        self.stage='9B first development pair'
        for replica in (0,1):
            for p in POLICIES:
                self.job('development',replica,self.policy(initial[p]))
            self.progress(force=True)
        for round_index in range(2):
            if cal.choose_tuple(self.development_records(),members=(0,1),process_scale=scale):
                break
            for p in CALIBRATED:
                name=f'proposal-first-{round_index}-{p}'
                proposal=self.decision(name)
                if proposal is None:
                    proposal=cal.ridge_proposal(p,self.development_records(),members=(0,1),process_scale=scale,
                        stage='first_pair',proposals_used=round_index)
                    if proposal is None:
                        self.seal(name,dict(skipped='no remaining proposal'))
                        continue
                    reservation=self.reserve(extra=(p,p),validation=True)
                    if not reservation['passed']:
                        self.seal(name,dict(skipped='resource reserve',reservation=reservation))
                        continue
                    self.seal(name,proposal)
                if 'skipped' in proposal:
                    continue
                for replica in (0,1):
                    self.job('development',replica,self.policy(proposal['policy_recipe']))
                self.progress(force=True)
        proposed=self.decision('validation-proposal')
        if proposed is None:
            proposed=cal.choose_tuple(self.development_records(),members=(0,1),process_scale=scale,require_match=False)
            if proposed is None:
                raise ValueError('missing initial development tuple')
            self.seal('validation-proposal',proposed)
        policies=dict(proposed['policies']); policies['R']=initial['R']
        self.stage='9B four-member validation'
        for replica in (2,3):
            for p in POLICIES:
                self.job('development',replica,self.policy(policies[p]))
            self.progress(force=True)
        chosen=cal.choose_tuple(self.development_records(),process_scale=scale)
        if chosen is None:
            for p in CALIBRATED:
                name='proposal-repair-'+p
                proposal=self.decision(name)
                if proposal is None:
                    proposal=cal.ridge_proposal(p,self.development_records(),process_scale=scale,stage='repair',proposals_used=0)
                    if proposal is None:
                        self.seal(name,dict(skipped='no remaining proposal'))
                        continue
                    reservation=self.reserve(extra=(p,)*4,validation=False)
                    if not reservation['passed']:
                        self.seal(name,dict(skipped='resource reserve',reservation=reservation))
                        continue
                    self.seal(name,proposal)
                if 'skipped' in proposal:
                    continue
                for replica in range(4):
                    self.job('development',replica,self.policy(proposal['policy_recipe']))
                self.progress(force=True)
            chosen=cal.choose_tuple(self.development_records(),process_scale=scale)
        if chosen is None:
            proposal=cal.choose_tuple(self.development_records(),process_scale=scale,require_match=False)
            return self.seal('design',dict(status='calibration_infeasible',selection=proposal))
        reservation=self.reserve(validation=False)
        return self.seal('design',dict(status='passed' if reservation['passed'] else 'calibration_budget_limited',
            selection=chosen,reservation=reservation,source_hash=self.store.identity['source_hash'],
            fresh_seed=configuration(self.history,'fresh',self.args.device).seed,replicas=self.args.replicas))

    def execute(self):
        self.progress(force=True)
        self.seal('cohort',dict(protocol=PROTOCOL,conditions=list(POLICIES),
            fresh_replicas=list(range(self.args.replicas)),development_members=list(range(4)),
            fresh_seed=configuration(self.history,'fresh',self.args.device,self.args.smoke).seed,
            development_seed=configuration(self.history,'development',self.args.device,self.args.smoke).seed,
            novel_budget=self.runner('fresh').cfg.novel_count,smoke=self.args.smoke,
            source_hash=self.store.identity['source_hash']))
        self.fidelity()
        if self.args.stage=='fidelity':
            self.state='fidelity_passed';self.stage='9A fidelity complete'
            return
        if self.args.smoke:
            policies={p:Policy(p,math.sqrt(10) if p=='A' else 1.) for p in POLICIES}
        else:
            design=self.calibrate()
            if design['status']!='passed':
                self.state=design['status'];self.stage='9B calibration closeout';self.current=None
                self.progress(force=True)
                return
            policies={p:self.policy(value) for p,value in design['selection']['policies'].items()}
            policies['R']=Policy('R',process_scale=design['selection']['process_scale'])
        self.stage='9C fresh paired confirmation'
        for replica in range(self.args.replicas):
            order=POLICIES[replica%6:]+POLICIES[:replica%6]
            for p in order:
                self.job('fresh',replica,policies[p])
            self.progress(force=True)
        self.stage='9D inference and closeout'
        if self.decision('analysis') is None:
            from .analysis import analyze
            result=analyze([r for r in self.records if r['namespace']=='fresh'],draws=100 if self.args.smoke else 2000,
                seed=BOOTSTRAP_SEED,check=self.budget.check,
                load_bootstrap=lambda start:self.decision(f'bootstrap-{start:04d}'),
                save_bootstrap=lambda start,block:self.seal(f'bootstrap-{start:04d}',block))
            self.seal('analysis',result)
        self.integrity()
        self.state='completed';self.stage='Phase 9 complete';self.current=None
        self.progress(force=True)
        self.store.write_json(None,'study_completed.json',dict(status='completed',protocol=PROTOCOL,
            completed_at=utc_now(),charged_seconds=self.budget.used,trajectories=len(self.records),
            smoke=self.args.smoke,analysis_unit='decision-analysis'))

    def integrity(self):
        if self.decision('integrity'):
            return
        if content_hash(sources())!=self.store.identity['source_hash']:
            raise artifacts.ImmutableError('scientific sources changed during execution')
        units=list(artifacts.iter_units(self.store.root,full=True,check=self.budget.check))
        if any(u['status']!='completed' for u in units):
            raise ValueError('incomplete unit at final audit')
        fresh=[r for r in self.records if r['namespace']=='fresh']
        for replica in range(self.args.replicas):
            rows=[r for r in fresh if r['replica']==replica]
            if len(rows)!=6 or {r['policy'] for r in rows}!=set(POLICIES) or len({r['plan_hash'] for r in rows})!=1:
                raise ValueError('confirmation pairing failure')
            for row in rows:
                saved=self.store.load_checkpoint(row['physical_unit'])
                retired=[r['exposure'] for r in saved['retirements']]
                if len(set(retired))!=len(retired):
                    raise ValueError('duplicate archived observation')
                if row['policy']!='R' and (saved['pending_scores'] or saved['learner']['fisher_L']['kind']!='diagonal'):
                    raise ValueError('invalid final diagonal/controller state')
        self.seal('integrity',dict(status='passed',checked_units=len(units),paired_blocks=self.args.replicas,
                                   sources=sources()))


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--notebook',type=Path,default=REPO/'mnist_experiment/plan18_results.ipynb')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--stage',choices=('all','fidelity','status','stop','report'),default='all')
    p.add_argument('--resume',action='store_true');p.add_argument('--clear-stop',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--replicas',type=int,default=8)
    p.add_argument('--max-paths',type=int)
    p.add_argument('--smoke-stop-at',choices=('pretrain','accepted','mid_solve','eviction','split_group'))
    return p.parse_args(argv)


def main(argv=None):
    args=arguments(argv)
    if args.stage in ('status','stop','report'):
        if args.stage=='status':
            print(json.dumps(read_json(args.root/'status.json'),indent=2))
        elif args.stage=='stop':
            (args.root/'STOP').touch()
        else:
            from .report import publish
            publish(read_json(args.root/'report_snapshot.json'),args.notebook, type('ReadOnlyRoot',(),{'root':args.root})())
        return 0
    if (args.device!='cuda' or args.replicas!=8) and not args.smoke:
        raise ValueError('production requires CUDA and eight fresh paired blocks')
    if args.smoke_stop_at and not args.smoke:
        raise ValueError('artificial interruptions are smoke-only')
    if args.smoke and (args.root.resolve()==ROOT.resolve() or args.notebook.resolve()==(REPO/'mnist_experiment/plan18_results.ipynb').resolve()):
        raise ValueError('smoke requires a separate root and notebook')
    if args.replicas<2 or (args.max_paths is not None and args.max_paths<1):
        raise ValueError('at least two replicas and positive bounded path count required')
    approval=authorize()
    history=read_json(HISTORY/'resolved_config.json')
    cfg=configuration(history,'fresh',args.device,args.smoke)
    setup_torch(cfg)
    code=sources(); data=dataset_manifest(args.data_root)
    if data!=read_json(HISTORY/'provenance.json')['dataset_manifest']:
        raise ValueError('historical MNIST source changed')
    identity=dict(protocol=PROTOCOL,allocation_family='plan18',role='smoke' if args.smoke else 'production',
        source_hash=content_hash(code),sources=code,config_hash=content_hash(dict(config=cfg.mapping(),replicas=args.replicas)),
        data_hash=content_hash(data),historical_identity=artifacts._identity(HISTORY)['identity_hash'],
        authorization_sha256=approval['sha256'])
    if not args.smoke and args.stage=='all':
        accepted=read_json(FAMILY/'phase9_acceptance.json')
        if accepted.get('status')!='passed' or accepted.get('source_hash')!=identity['source_hash']:
            raise ValueError('matching Phase 9 CPU/CUDA/recovery acceptance required')
    other=peer_seconds(args.root)
    with artifacts.Store(args.root,identity,resume=args.resume,clear_stop=args.clear_stop,
        external_bytes=external_bytes(args.root,args.data_root),free_reserve_bytes=20_000_000_000,
        finalization_reserve_bytes=64_000_000,max_active_seconds=max(0.,72000-other)) as store:
        budget=RuntimeBudget(store,other_seconds=other,allocation_seconds=72000.)
        study=None
        try:
            if not (args.root/'provenance.json').exists():
                detail=provenance(cfg,data)
                detail.update(protocol=PROTOCOL,sources=code,source_hash=identity['source_hash'],authorization=approval)
                store.write_json(None,'provenance.json',detail)
                store.write_json(None,'resolved_config.json',cfg.mapping())
            with store.control.signal_handlers():
                study=Study(args,store,budget,history)
                study.execute()
        except (StopRequested,LearnerStop,ResourceLimit) as exc:
            if study:
                study.state='paused';study.reason=str(exc)
            print(json.dumps(dict(status='paused',reason=str(exc))),flush=True)
            return 2
        except BaseException as exc:
            if study:
                study.state='failed';study.reason=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if study:
                try:
                    study.progress(force=True)
                except (StopRequested,LearnerStop,ResourceLimit):
                    atomic_json(args.root/'status.json',dict(status=study.state,stage=study.stage,reason=study.reason))
            budget.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
