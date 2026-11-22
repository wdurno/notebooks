# Plan 19 Handoff: Phase 3 Development

## Latest: Fix 5 Conditional Likelihood And Signed-Square Rewards Agreed (2026-11-22)

**Do not resume the old campaign or start a new experiment from this note.**
Fix 4 is complete: its orchestration `COMPLETED` marker is present, status is
`complete`, all four trajectories / two paired streams are sealed, and charged
time is 70,871.7245 seconds / 19.6866 hours. Its final notebook was refreshed on
2026-11-20. The older live-run and pause entries below are historical.

The user requested a new repair draft to protect the agreed design during
continued mathematical discussion, not implementation or GPU execution. They
called it Fix 4; preserve that completed identifier and use **Phase 3 Fix 5**.
The append-only section at the foot of `plan19.md` has explicit anchor
`plan19-phase3-fix5`; the end of the old Fix 4 section points to it.

Preserve these design decisions:
- AWAC with refreshed one-step Bellman targets, Gaussian action coordinates,
  small action-conditioned critics, and refreshed actor advantage weights.
- Use the explicit frozen-reference conditional likelihood in Fix 5 section 5
  for training data NLL, EWC Fisher and STM joint predictive KL. The user has
  accepted this practical contract; the self-consistent-model efficiency proof
  is deferred and is not an implementation gate.
- All three learners use signed-square rewards: raw stage reward is
  dt*(.01*q*abs(q)-.04) + b*abs(b), then divide by ten. Raw success/collision/
  timeout bonuses are +100/-25/-1; encoded bonuses are +10/-2.5/-.1.
  Keep the frozen six-example scorer unchanged. Calibrate advantage temperature
  and critic variance in encoded units; rebuild compatible heads and Fisher.
- Freeze Qwen/adapters, the input projector and LSTM/adapters after compatible
  initialization. Only output/action and critic heads are tunable; use head-only
  rank-one STM. The user now requests LTM EWC ranks 0, 3, 5, 10 using
  low-rank-plus-diagonal representations; admission remains eviction-only.
- Memoize the exact frozen history representation; history trimming, language,
  controls, time and encoder identity must invalidate incompatible cache entries.
- A-only appearance/steering, with modest continuous barrier-difficulty change,
  not a return to the reliably blocked B endpoint. No LFU/adaptive pi/physical run.
- Broader development-only replay/optimization/rank survey, then FD/ridge retention
  calibration and fresh assessment with settings frozen. Development may learn
  from the entire new A curriculum; it does not mean old-data-only calibration.
- The draft proposes MC-Replay, TD-Replay and TD-FD controls, matched episode-end
  admission, a 1,024-interval moderate-A path, two fresh paired replicas, and a
  roughly 5--13-hour compute forecast after implementation. These specifics are
  reviewable proposals, not authorized numerical settings or measured timings.
- Keep the notebook concise and chronological, with live progress near the top,
  held-out observed rewards/losses, replay age/reuse and update-time diagnostics,
  top-down videos, immutable units and tested `--resume` semantics.

**Theory is consolidated into one local framework.** Fix 5 sections 1--4
define one conditional likelihood, its Fisher and efficient local estimate,
an exact Gaussian Bellman example, and a sufficient score-equation criterion
for a numerical solver to inherit MLE efficiency. The user's goal is local
data-use efficiency, not global trajectory-wide MLE efficiency. The lengthy
sequence of alternative formulations has been replaced. Section 5 is now the
accepted operational contract, not an unresolved choice: differentiate only
candidate parameters in the normalized conditional working likelihood.
Hold the reference, targets and realized per-transition advantage weight fixed
through every inner fit, Fisher integration and both FD endpoints. Do not
recompute a weight from a newly sampled model action. The working actor uses
covariance Gamma/w, distinct from the executed policy's Gamma. Integrate the
critic part of the joint KL over the source working actor, preserving its
candidate-parameter dependence when differentiating the FD objective.
The twin-disagreement regularizer is outside data NLL/Fisher. EWC admits the
LTM block only on retirement. The MC control has the same conditional family
with encoded observed return as response and continuation offset zero.
Refresh references only between local fits; retain snapshot identities and
RNG state across resume. Do not implement total continuation derivatives or
claim the abstract result proves this sequence achieves MLE efficiency.
All theory remains in `plan19.md`; the mathematical overview is unchanged.

**Low-rank plan:** prefer incremental thin SVD of old factor plus weighted
fresh Fisher columns, returning discarded diagonal mass to the residual.
Preserve the full represented diagonal/trace. Reuse `src/representations.py`
quadratics/Woodbury solves; no production dense reconstruction. This is not
Plan 18's exact dense recompression: old discarded correlations are lost.
Profile small Gram versus thin SVD/QR, audit compression, and retain wider
sketch/Frequent Directions/Lanczos as recorded fallbacks. Track the spectrum
of L^T L (nonzero LL^T spectrum), trace share, residual/PSD integrity, operator
and solve errors, timing and bytes. Distinguish EWC rank from STM LoRA rank.
The main fresh TD-FD versus TD-Replay contrast shares a development-selected
EWC rank; a small development rank comparison checks zero versus a shortlisted
positive rank under both methods. No four-rank full online factorial is
authorized. Reprice this survey in the resource pilot; the 5--13 h forecast
is still provisional. No implementation or GPU execution has begun.

## Latest: User Authorized Detached Fix 4 Completion (2026-11-20)

The user requested the remaining experiment run in `tmux` so discussion of
the next repair can proceed concurrently. Restarted the existing campaign in
`plan19-fix4` with `--resume --clear-stop` and its recorded optional model
override. CUDA, the 1,027-capture / 938-event checkpoint, finite parameters and
Fisher, accepted-center equality, solver counts, retirement ledger, all 102
frozen numerical source hashes and unchanged theory passed preflight.
The recovery ledger starts from 13.1649 h against the same 28 h ceiling;
approximately 544 GB was free. No scientific settings changed.

Let the runner finish seed 198102 FD, then its matched control, audit/seal
the campaign and refresh the notebook automatically. There is no automatic
Phase 4 or physical operation. Status and checkpoint-boundary STOP commands
remain available below. New buffer/capacity designs are discussion only and
must not alter this frozen run. The pause record below is historical.

Architecture clarification from the saved parameter inventory: the input
projection is 2,053 -> 64 (2,048 Qwen channels plus four executed controls and
elapsed time), not the backbone's visual projector. A single 64-input /
64-hidden LSTM cell has 33,280 dense LTM coordinates plus 640 rank-one STM
coordinates, total 33,920 excluding the separate input/output maps. Total
STM is 55,021, of which 51,968 belongs to the final transformer block.
Replay capacity is eight transitions with one replay draw per new fitting
event, three steps on that selected pair. In the first sealed FD trajectory,
626/1,781 fitted histories (35.15%) were never selected for replay afterward;
they were still initially fitted and admitted to EWC upon retirement.

## Latest: Paused For User Interim Review (2026-11-20)

**Do not resume until requested.** The user interrupted the monitoring turn
and requested an interim review. On inspection no tmux server or experiment
GPU process remained; the parent heartbeat was stale, not a current running
status. A STOP marker was placed to preserve the requested pause. Do not
describe this absence as an acknowledged clean runner stop. No numerical
traceback explains the earlier disappearance in the saved log.

The third main trajectory (seed 198102 / FD),
`units/trajectory-ddc969b05575`, has a verified committed checkpoint at
**1,027 observations / 938 accepted events / 2,814 optimizer steps**, in
checkpoint evaluation near nominal 1,024. Targets 0, 448 and 768 are complete;
the 1,024 A panel was partially scored. Replay has eight eligible histories,
no pending labels or ready updates, and 1,445 retired histories including
initialization. Recovery seal, finite parameters and nonnegative Fisher,
accepted-center equality, event counts, all 102 frozen source hashes and the
unchanged theory passed the audit. The parent recovery charge is **13.1649 h**
including its conservative 60-second unclean-recovery reserve; the last
displayed heartbeat charge was 13.1482 h. Do not charge the inactive gap.

Two of four main trajectories are sealed; nominal acquired training workload
is 62.54% complete, with all setup/calibration done. Measured remaining FD
time plus one full control suggests about **7--9 additional active hours**
including closeout; the original 28 h cumulative ceiling remains unchanged.
No automatic Phase 4 or physical operation is authorized.

Interim artifact-only layout review of the completed seed-198101 pair:
final A has eight clear and four blocked layouts. FD succeeds on four clear
layouts and zero blocked layouts; control succeeds on none. Both B panels
have twelve blocked layouts and zero successes for both methods. This is
evidence of retained clear-path A behavior without demonstrated detour skill,
not established robotics sample savings. A/B labels denote fixed test
environments evaluated separately at every checkpoint, not segments of the
smooth training curriculum. Current Fix 4 uses linear rewards only and
diagonal-only LTM EWC in both learner conditions; signed-square results are
historical Fix 3. The first-step training plot is not held-out loss: it is
data-only new/replay NLL before each three-step fit, in 32-observation bins.
Whole-episode overshoots and the final bin center explain x=2,064.

## Latest: First Fix 4 Pair Sealed, Second Seed Running (2026-11-19)

Both seed-198101 trajectories are complete. FD at
`units/trajectory-a4860df2bfdb` captured 2,056 observations, accepted 1,781
events, recorded 5,226 FD evaluations and nonzero first-step EWC influence in
all events, with no pending updates or incomplete episode. Its active time
was 14,805.347 s / 4.1126 h versus control 5.0225 h. Final successes are FD
A 4/12, B-initial 0/12, B-final 0/12; control 0/12 in each environment.

One-pair budget-averaged fixed-reference joint NLL differences (FD minus
control) are A +.74465, B-initial -.55544, B-final -2.22989. No between-replica
CI is estimable from one pair, and reference NLL is not navigation or sample
savings. Retention accuracy and reference NLL do not tell the same story.
FD/control total-time ratio .81883 is policy/history dependent: fit time per
accepted event was about 1.647 s versus 1.583 s, while FD had shorter evaluation
rollouts and more collision-ended training episodes. Do not infer a cheaper
FD primitive from total time, or add overlapping component timers.

The campaign admitted seed 198102 and runs FD first at
`units/trajectory-ddc969b05575`, then its control, as frozen. Initial evaluation
is underway. Cumulative active charge was approximately 11.2 h / 28 h at this
milestone; consult committed progress for current values. The notebook is
refreshed from the two sealed main units. Continue Fix 4 only; earlier
milestones and pauses below are history, not a current stop request.

## Latest: First Fix 4 Control Sealed, FD Started (2026-11-19)

Main control seed 198101 is complete and immutable at
`units/trajectory-1b1d7fd79434`: 2,053 captures (five over nominal), 1,898
accepted events, 2,405 retired histories including initialization, zero FD
evaluations, and nonzero first-step EWC influence in all 1,898 events. Active
trajectory time is 18,081.021 s / 5.0225 h, inclusive of evaluations/scoring
and checkpointing; overlapping component timers must not be added.
All final A/B-initial/B-final panels have zero successes out of 12 each.
This is a completed control result, not an FD efficacy conclusion.

The same campaign automatically started the first positive-FD trajectory,
seed 198101, at `units/trajectory-a4860df2bfdb`, with strength
4498.478657338156 and ridge .1. Its initial evaluation is underway. The notebook
has been refreshed with the sealed control curves and provisional active
progress. There is not yet a complete paired result. Continue the authorized
four-trajectory Fix 4 campaign under the existing 28 h cumulative ceiling;
no Phase 4 or physical operation. Earlier pause counts below are history.

## Latest: Fix 4 Resumed From Final-B Hold (2026-11-19)

The user requested resumption after the second pause. Restarted the existing
campaign in detached `plan19-fix4` tmux with `--resume --clear-stop` and its
recorded optional model override. Preflight reverified the 1,787-capture /
1,636-event checkpoint seal, accepted-center equality, finite parameters and
nonnegative Fisher diagonal, retirement ledger, three-step counts, all 102
frozen numerical source hashes and unchanged theory. PyTorch CUDA passed on
the RTX 4070; approximately 545 GB was free. No numerical settings changed.

Continue the saved final-B hold, including the ten ready updates and five
pending-return intervals, then the final evaluation and remaining main runs.
The cumulative allowance starts from 5.7929 h charged out of 28 h, not zero.
Reuse sealed preparation/calibration. Monitor the notebook and committed
progress; stop after Fix 4 closeout, without Phase 4 or physical operation.
The pause entries below are historical, not current stop requests.

## Latest: Fix 4 Paused Again At User Request (2026-11-19)

**Stopped cleanly; wait for the next explicit resume request.** The STOP
marker is retained, `plan19-fix4` has exited, and the GPU has no experiment
compute process. The final `InterruptedError` in `runner.log` is the expected
checkpoint-boundary stop. No Phase 4 or physical motion is authorized.

The first main trajectory, **seed 198101 / Replay/EWC control**, is saved at
**1,787 captured training intervals / 1,636 accepted events / 4,908 optimizer
steps**, in the final-B hold (which starts at 1,664). It has 2,143 retired
histories including the 515-record initializer, 18 eligible histories, five
pending-return intervals and ten ready updates. An episode is in progress;
the simulator, recurrent history, learner, optimizer and RNG states are saved.
Do not discard or manually drain these queues during recovery.

Checkpoints 0, 448, 768, 1,024 and 1,664 have complete three-environment task
and fixed-reference evaluations. Actual acquisition boundaries are 0, 450,
771, 1,032 and 1,667. The final 2,048 evaluation remains to be run. Zero of
four main trajectories is sealed, so there is no complete FD/control pair.
At the latest panel the control has zero successes in all three environments;
these are interim control results, not a treatment conclusion.

Charged active compute is **20,854.472 seconds / 5.7929 h** against the same
28 h ceiling (22.2071 h remains). User pauses are excluded. Checkpoint seals,
finite parameters and nonnegative Fisher diagonal, accepted-center equality,
retirement ledger, three-step event counts, all 102 frozen numerical source
hashes and unchanged theory passed the post-exit audit. The canonical notebook
has been refreshed from scalar artifacts; no training was run by reporting.

On a later authorized resume, use the existing `run --resume --clear-stop`
command and the recorded optional model override, recreating detached tmux
session `plan19-fix4` if desired. Current unit:
`cache/numerical_experiment/picar/phase3/curriculum-fix4-v1/units/trajectory-1b1d7fd79434`.
Reuse all sealed preparation/calibration and the partial trajectory. Frozen FD
strength remains 4498.478657338156 / ridge .1, control FD=0 / ridge .1. Earlier
resumption and pause entries below are historical.

## Latest: Fix 4 Resumed After User Pause (2026-11-19)

The user requested resumption. The existing campaign was restarted in detached
`plan19-fix4` tmux with `--resume --clear-stop` and its recorded optional model
override, without changing the scientific sources or selected settings.
Preflight verified the sealed 517-capture / 463-event checkpoint, accepted
parameter center, finite/nonnegative diagonal Fisher, retirement accounting,
all 102 numerical source hashes and the unchanged theory notebook. Actual
PyTorch CUDA execution passed on the RTX 4070; approximately 546 GB was free.
The saved charge is 2.9522 h of the same 28 h ceiling, not a reset allowance.

Continue the four main trajectories and close out Fix 4 only. Completed
initialization, reference banks and calibration are reused. Preserve the
partial episode and pending labels; monitor scalar progress and notebook
updates. The pause record below is historical, not a current stop request.
No physical operation or automatic Phase 4 is authorized.

## Latest: User-Requested Fix 4 Pause (2026-11-19)

**Stopped cleanly; do not resume until requested.** The checkpoint-boundary
STOP was acknowledged, the `plan19-fix4` tmux runner exited, and NVIDIA reports
no experiment compute process. The STOP marker is intentionally retained.
The interruption traceback in `runner.log` is the expected stop acknowledgement,
not a failed numerical update. No Phase 4 or physical motion is authorized.

Saved first main trajectory: **seed 198101, Replay/EWC control**, 517 newly
captured training intervals out of nominal 2,048, still in the blocking ramp.
It has 463 accepted learning events / 1,389 optimizer steps; 970 retired
histories (including the 515-record initializer), eight eligible histories,
16 pending-return intervals, no ready updates, and an in-progress episode.
The simulator, recurrent history, learner/optimizer and RNG state are saved.
The initial and nominal-448 evaluations are complete (the latter at 450 actual
captures), including all three fixed-reference banks. Zero of four main
trajectories is sealed; shared preparation and calibration remain complete.
Charged active compute is **10,628.005 seconds / 2.9522 hours** of the approved
28-hour ceiling; this user pause is not compute time.

After runner exit, the trusted checkpoint load verified its recovery seal,
finite model parameters, unique/disjoint retirement ledger, accepted event and
three-step counts, and all 102 frozen numerical source hashes. The current
unit is `units/trajectory-1b1d7fd79434` under
`cache/numerical_experiment/picar/phase3/curriculum-fix4-v1`.
The canonical notebook was refreshed from saved scalar artifacts.

For a later authorized restart, first check status and GPU availability, then
use the existing `run --resume --clear-stop` command below with the same
optional model override recorded in `orchestration/config.json`. A detached
`plan19-fix4` tmux session may be recreated. Do not restart initialization,
repeat completed calibration, discard the partial episode, change frozen
settings, or launch duplicate runners. Earlier active-run entries are history.

## Latest: Fix 4 Main Runs Started (2026-11-19)

The three reference banks are sealed: A 147 usable histories / 155 captures,
B initial 68 / 76, B final 53 / 61, each from eight complete episodes. No bank
has a target shortfall or enters learning. The real STOP/resume test completed
without restarting those collections. FD resolution passed at .005/.01/.02.
The separate 128-interval timing run and three FD calibration runs completed,
followed by the matched control on seed 198301. No extra ridge trial was needed.

Frozen selection: **FD strength 4498.478657338156, ridge .1**, versus control
FD=0/ridge=.1. This multiplier targets the probed FD/live-gradient norm ratio
1; it is not a learning rate. All ordinary optimizer/exploration settings stay
unchanged. The tiny two-layout A matching diagnostic passed relative to the
control; this is not absolute retention, equivalence, or main efficacy evidence.

Preparation charged 1.942 h before main admission. Measured forecast for all
four main trajectories is **17.634--22.924 additional active hours**, or about
**19.576--24.866 total**, below the approved 28-hour ceiling. Actual timing and
artifacts, not this estimate, determine the remaining allowance. Main boundaries
remain 0,128,768,1024,1664,2048; no ramp/hold was shortened. Seed 198101 runs
Replay/EWC then FD; 198102 reverses order. The `plan19-fix4` tmux campaign and
the same status/stop/resume commands below remain current. Do not launch a
duplicate runner, Phase 4, or physical motion.

Reporting-only additions compute paired budget-averaged fixed-reference risk
and final task gaps from complete pairs. They retain the original normalized
likelihood, and no samples-saved statistic is invented for flat task curves.
Training-clock conversion can verify its 752-observation initialization offset
from the full contiguous saved arrival ledger, rather than requiring a redundant
summary field. The shared summary reader remains bounded at 128 MiB per summary
and 512 MiB total here, with all seals/schema checks intact. These reporting
changes are outside the 102 frozen numerical sources; all those hashes match.

**Main reporting reload:** a clean checkpoint/resume at seed 198101/control,
414 captures / 363 accepted events / 1,089 optimizer steps activated the tested
reporting refinements in the long-lived Python process. The checkpoint passed
its payload seal, finite-parameter and accepted-center checks, unique/disjoint
retirement ledger checks, and exact event/step counts. It retained 870 retired
histories, 28 eligible records and one pending-return interval at the saved
boundary (verify live state rather than using these historical counts).
No numerical source, selected hyperparameter, seed or curriculum changed.
The same tmux session name and command resume the existing unit, not a new
replica. This also exercises real-model recovery during ongoing learning.

## Current: Fix 4 Authorized, Implementation Underway (2026-11-19)

The user approved Phase 3 Fix 4 and a **28 additional active-hour ceiling**.
The earlier draft-only and Fix 3 stop instructions below are historical.
Implement/run `numerical_experiment.picar_curriculum_repair` under the separate
`cache/numerical_experiment/picar/phase3/curriculum-fix4-v1` root. Preserve the
completed Fix 3 artifacts and the theory notebook. The GPU was available at
preflight, with about 509 GB free on the working disk. No physical motion or
automatic Phase 4 is authorized. New CPU recovery/prediction tests and a
bounded real-model pilot must pass before the four main trajectories.

The staged simulator is opt-in (`staged_curriculum=True`); its default remains
historical. Main stage boundaries are 0,128,768,1024,1664,2048. Compressed
128-interval calibration runs are engineering/development diagnostics, never
substitutes for the main slow curriculum or members of its paired outcomes.
Fixed-reference data never enter fitting, replay, EWC or baseline updates.

Implemented entry point (run from the checkout, using the user's venv):

```bash
~/.venv/bin/python -m numerical_experiment.picar_curriculum_repair status
~/.venv/bin/python -m numerical_experiment.picar_curriculum_repair stop
~/.venv/bin/python -u -m numerical_experiment.picar_curriculum_repair run \
  --resume --clear-stop --model-path "$MODEL_PATH" --offline
```

Set `MODEL_PATH` to the **same explicit optional model override recorded in
the orchestration configuration** for this local campaign. The default for a
fresh campaign remains a pinned Hub download into the ignored project cache;
no sibling-project software import is used. Do not start a second runner if
one already owns the orchestration lock. The commands never move the physical
robot. `--stage software`, `references`, or `calibration` provides bounded
check-ins without sealing the whole campaign; later `--resume` continues the
same identities. Reporting-only refresh uses the `report` command.

Acceptance so far: eight deterministic trajectory/recovery tests, 27 control-
plane tests, the standalone real-PyBullet CPU smoke (12 accepted events, all
with first-step EWC influence, 21 FD evaluations), plus the bounded predictive
and staged-simulator suites. The canonical software unit is running the
combined acceptance suite before any new Qwen learning. Main trajectories have
not started. Five minutes of prelaunch checks are conservatively charged once;
subsequent runner wall time is charged continuously, with a 60-second recovery
margin on the latest uncleanly interrupted heartbeat. User pauses are excluded.

**Reporting-only restart (2026-11-19):** the eight-episode A reference bank
produced 147 usable histories / 155 captures. This showed that main summaries
could exceed the old 16 MiB reader limit. After a clean STOP while scoring the
B-initial bank, the scalar reader now allows at most 128 MiB per sealed summary
(configuration/other JSON still 16 MiB) and 512 MiB cumulatively for this report.
All checksum, schema and nonfinite checks remain. Fifty-five shared/Fix 4
reporting tests pass; all 102 frozen scientific source hashes still match.
The same `plan19-fix4` tmux command resumed with `--resume --clear-stop` and
the same optional model override; no initialization, completed bank or sampled
episode was discarded. This is an operational reporting correction, not a
change in the likelihood, model, curriculum, optimizer or treatment settings.
Current scalar previews are provisional; complete paired results remain pending.

## New Draft: Phase 3 Fix 4 (2026-11-19)

The user requested a written repair for review, **not execution**. The new
Fix 4 section in `plan19.md` proposes one persistent learner progressing
through A hold, blockage-only ramp, B-initial hold, steering/texture ramp, and
full-B learning. The provisional budget is 2,048 new intervals per trajectory,
with a small pilot before freezing common stage lengths. Linear rewards are
fixed; the main comparison explicitly restores positive FD versus the
capacity-matched Replay/EWC control. Held-out NLL is required.

The plan and a draft-only notebook introduction share the diagram in
`docs/figures/plan19_phase3_fix4_transition.png`; regenerate it with
`~/.venv/bin/python docs/plan19_fix4_diagram.py`. This is a documentation
renderer, not implemented experimental scheduling. Fix 3 reporting preserves
the tagged Fix 4 draft after its historical conclusion on refresh.
The initial forecast is 16--26 additional active hours and the proposed hard
ceiling is 28 hours, all awaiting user approval. Do not start implementation
of the experiment, new GPU work, Phase 4 or physical motion from this draft.

## Current: Fix 3 Complete And Stopped (2026-11-17)

All four authorized pilots and both paired seeds are sealed; none was omitted.
The GPU runner exited successfully and released its compute resources. **Do
not resume learning, spend unused allowance, start Phase 4, or actuate the
physical robot without a new request.** Earlier active/pause entries below
are historical. Campaign: `cache/numerical_experiment/picar/phase3/reward-fix3-v1`.

Charged runner compute is **28,578.719 seconds (7.9385 h)** against the approved
twelve-hour ceiling; additional closeout checks are lightweight CPU work.
The campaign occupies about 1.2 GiB. The notebook has been regenerated from
sealed artifacts, with chronological Fix 3 results and a final conclusion.

**Reporting follow-up (2026-11-19):** the user requested online training NLL
trajectories, despite the intentionally changing sampling distributions. Plot
first-inner-step data-only NLL from the sealed pilot metrics, with equal replica
weights in 32-captured-interval windows and the 752-interval initialization
offset removed. The reward recipes have different likelihood targets; held-out
online NLL was not recorded. Record held-out NLL and actor/return components in
the next authorized experiment, with explicit targets/conditioning and outputs
sufficient for an audit. No new GPU work or Phase 4 is authorized by this
reporting follow-up. Fixed A and B evaluations use the same frozen checkpoint,
A then B, without learning between them; their differences are not texture-only.

| Seed | Reward | Captured | A initial / mid / final | B initial / mid / final | Active min |
|---|---|---:|---|---|---:|
| 197101 | Linear | 620 | 5/20 / 5/20 / 5/20 | 0/10 / 0/10 / 0/10 | 66.73 |
| 197101 | Signed square | 630 | 5/20 / 4/20 / 3/20 | 0/10 / 0/10 / 0/10 | 68.85 |
| 197102 | Signed square | 622 | 5/20 / 2/20 / 5/20 | 0/10 / 0/10 / 0/10 | 64.63 |
| 197102 | Linear | 625 | 5/20 / 0/20 / 4/20 | 0/10 / 0/10 / 0/10 | 78.56 |

No hard-task acquisition is established, and neither recipe is ready for
promotion. Mean final A success is 22.5% linear and 20% signed square, versus
25% shared initialization. These are two development replicas, not significance
or retention-equivalence evidence. Only 2/187 training episodes succeeded;
successful-path coverage and the documented Qwen false-zero issue are plausible
bottlenecks, not established causes. FD was off in both conditions, so this
does not test an FD benefit or establish a sample-efficiency result.

Closeout verified all 21 completed envelopes (20 units plus orchestration),
20 distinct evaluation panels, reward arithmetic on 487 episodes / 5,305
intervals, 20 videos, and 5,773 frame blobs. All 2,299 learning events have
unique IDs and nonzero first-step EWC influence; optimizer/retirement counts
match three steps per event and exactly-once admission. No score retries were
needed. The 93-source numerical fingerprint, frozen theory snapshot and current
user-updated theory notebook all pass their respective hash checks.
The focused regression suite passes **317 tests**. Report-only fixes populate
collision counts from saved end reasons and average success by checkpoint,
including mean captured counts, rather than joining independent replicas.

The operational resume wrapper, its exact source snapshot and its successful
outcome receipt are preserved under `resume_provenance/`. The current theory
notebook was not edited by this work. No mathematical treatment or hyperparameter
was changed after launch. Completed resumption is verification only, not a way
to extend this pilot. Discuss reward support, behavior and coverage before
designing the next repair.

## Current: Fix 3 Resumed (2026-11-17)

The user explicitly requested continuation. Parent/child recovery hashes and
all 17 sealed units passed preflight; the GPU was available. The run is active
again through `numerical_experiment.picar_reward_resume` (exec session 25955),
appending to the existing `reward-fix3-v1-run.log`. The original twelve-hour
ceiling, saved seeds, numerical sources, prompt and comparison are unchanged.
Finish the partial seed-197101 signed-square pilot, then seed 197102 in
signed-square/linear order. The paused record below is historical.

The ordinary command correctly refused resumption because the user committed
new STM theory to `mathematical_overview.ipynb` while paused. No numerical
source changed. The separately versioned operational wrapper recovers the
exact original theory notebook from Git (`dd447dce15779ee4408037be48dd4729d01f8322`)
into ignored `resume_provenance/`, verifies its frozen hash, and redirects
only the campaign's theory-provenance reads to that snapshot. It does not
replace the current notebook, fake a checksum, or relax other identity checks.
Receipts record both document hashes, current Git state and wrapper hash.
Seven CPU tests cover isolation, corruption, identity rejection and cleanup.
The snapshot's bytes and the original 93-source fingerprint remain enforced.

Subsequent restart command (the saved model-data override and cap are read
from the original resolved configuration):

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_reward_resume \
  --output cache/numerical_experiment/picar/phase3/reward-fix3-v1 \
  --resume --clear-stop --offline
```

Use two CPU threads and CUDA as before. Stop via the original campaign's
`stop` subcommand; it is the same checkpoint protocol. Do not launch a second
writer. Phase 4 and physical robot motion remain deferred.

## Current: Fix 3 Paused By User (2026-11-17)

The user requested the node back. The checkpoint-boundary stop completed;
the runner exited and `nvidia-smi` shows no compute process. **Do not resume
until requested.** The expected `InterruptedError` at the end of the log is
the deliberate stop, not a numerical failure. `STOP` remains in the campaign.

Campaign: `cache/numerical_experiment/picar/phase3/reward-fix3-v1`.
Charged additional active compute is **18,718.244 seconds (5.200 h)**, leaving
**6.800 h** of the approved twelve-hour ceiling; idle time is not charged.
Both reward-specific initializers/Fishers, both initial panels, and both
readiness checks are sealed. Initial A/B success counts are 5/20 and 0/10 for
each condition. Linear retained NLL pass five of sixteen; signed square
retained pass nine of twelve.

Seed 197101 linear is sealed (`pilot-b628efa4825a`): 620 captured transitions,
A 5/20 -> 5/20 -> 5/20 and B 0/10 at every panel. Its first-step EWC proposal
changed in all 568 recorded learning events. The corresponding signed-square
pilot is paused at **544 captured transitions**, with initial/midpoint A/B
panels fully scored. Its unfinished episode labels resume from their saved
per-image cursor. Active unit: `units/pilot-b49beea18668`. Then finish the final
panel and run seed 197102 in reversed order (signed square, then linear).
Do not rerun a completed unit or treat the partial pilot as a final result.
Approximately 2.5--3.5 further active hours are plausible; admission checks
still enforce the original cap.

Parent and child CURRENT manifests, progress/evidence files, and recovery
payload hashes verified after exit; all **17 sealed units** validate. The
active child journal sequence is 1875, commit
`8c4aa5c702ac451ca0ad6429b9e3e6c0`. Scientific fingerprints remain unchanged.
The focused suite passed 307 tests before the final report-only plot-label
test; the reporting suite now passes 53 tests. The notebook was refreshed to
show the pause and only sealed outcomes. Keep the small-visible-ball/zero-score
qualification below. No prompt, numerical recipe, or learning scope changed.

On explicit resumption, use the same optional model-data override recorded
as `model_source` in `orchestration/config.json`:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_reward_repair run \
  --output cache/numerical_experiment/picar/phase3/reward-fix3-v1 \
  --max-hours 12 --resume --clear-stop \
  --model-path <same-local-model-directory> --offline
```

Use the user's venv, two CPU threads, and actual CUDA access. Preserve/append
the existing log rather than truncating it. A fresh reporting process picks up
the report-only legend/import fixes; regenerate the canonical notebook at
closeout. Phase 4 and physical robot motion remain unauthorized.

## Phase 3 Fix 3: Reward Repair (2026-11-17)

Execution is authorized with **12 additional active compute hours**. The new
`numerical_experiment.picar_reward_repair` package implements the six-example
frozen width evaluator and linear/signed-square comparison, holding FD off.
Historical numerical files and completed campaigns remain unchanged.

New campaign root: `cache/numerical_experiment/picar/phase3/reward-fix3-v1`.
The exact 515 training/65 validation histories and common post-KL checkpoint
were recovered; no new demonstrations or physical motion are needed. Re-score
stationary images once, refit each reward likelihood and its equal-weight
four-draw Fisher, then run readiness and paired seeds 197101/197102. Keep the
three-step proximal solver. The reward/return-unit contract is in Phase 3 Fix 3
of `plan19.md`; raw squared bonuses are +100/-25/-1 and coded returns divide
by ten. Both models use 161 coded bins on [-6,12] and the common coded
temperature. Never transplant the old reward's Fisher or targets.

The numerical/scoring and reporting agents finished their bounded changes.
Integration/recovery acceptance passed 287 tests, including fake-clock complete
pilots and exact accepted-state recovery at reward-label and midpoint/final
evaluation-score boundaries. The GPU campaign is now active. Its log is
`cache/numerical_experiment/picar/phase3/reward-fix3-v1-run.log`.
The report snapshots existing notebook history rather than rebuilding every
historical campaign. The new notebook section is chronological, with progress
near the top. Save both training and held-out rewards. Do not start Phase 4.

Planned restart command from the repository root (same optional model override
as the initial invocation, if used):

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_reward_repair run \
  --output cache/numerical_experiment/picar/phase3/reward-fix3-v1 \
  --max-hours 12 --resume
```

Add `--model-path <local-model-directory> --offline` to reuse local model data;
without an override the pinned model downloads into the project cache.
`status`, `stop`, and `report` replace the `run` subcommand. A graceful stop
leaves `STOP`; use `--clear-stop --resume` only for intentional continuation.
An unannounced restart resumes from committed scoring, fit, Fisher, learner,
simulation and evaluation cursors. Do not launch a second writer.

**Initialization milestone (2026-11-17):** all 659 unique stationary images
are scored without retries. Linear fitting completed 16 passes and retained
validation-best pass five; signed-square fitting completed 12 passes. Both
equal-weight Fisher assets are sealed. Linear initial A/B counts are 5/20 and
0/10, and its readiness run is complete; signed-square initial evaluation is
in progress. The expanded focused suite passes 307 tests. All 16 notebook code
cells also execute from the notebook directory with model/simulator imports
blocked. Numerical source hashes remain unchanged.

The linear initial B panel has zero Qwen scores on all 58 saved frames. A
qualitative spot check of its first frame, `sha256:bc04b838327c1797d1575c053c99a90b280e314ac85f080d8c3030909957c73c`,
shows a small red ball above the blue barrier despite that zero judgement.
Do not infer that every zero means true absence. This remaining evaluator
blind spot is a qualification, not permission to tune the frozen prompt or
change the learning comparison. Continue the approved curriculum/pilots.

## Training Reward Review (2026-11-16)

The user requested quick reward diagnostics, then explicitly declined new
evaluation-image scoring. Use saved training rewards only. The canonical
notebook now adds a compact review before the final conclusion: cumulative
reward by recorded clear/blocked path and early/middle/late exposure third,
plus visual-score/dense-rate plots. These are not frozen A/B test rewards or
initial/mid/final policy snapshots. The plotting module is
`numerical_experiment/picar_reward_review.py`; no scientific source or sealed
run is modified. Reward handling preserves ended episodes in cumulative means
and weights replicas equally. Empty groups have no fabricated curves.

Training successes are 3/115 episodes Replay/EWC and 1/96 FD; collisions are
98/115 and 71/96. The equal-replica fraction of visible observations assigned
proximity 1/10 is 98.4% and 96.8%. Collision penalties are already -5, versus
-1 timeout and +10 success. Full completed-episode discounted returns supply
training targets, not one-step TD. Weak distance shaping and insufficient
coherent successful detours are hypotheses, not causal diagnoses.
Acceptance passed 129 focused tests; all 14 notebook code cells execute with
model/training/runtime imports blocked. Scientific fingerprints, completion
envelopes and the reserved theory notebook remain unchanged. No GPU work ran.

**Future requirement:** record and plot test rewards by method, A/B and
initial/mid/final. See the new Phase 4 requirement in the plan. Implement a
separately versioned, resumable post-rollout scorer before new evaluations;
retain strict training isolation and include reward-scoring cost in forecasts.
Do not backfill old runs, start the declined 2,090-image scoring job, or launch
another experiment without authorization. Fix 2 remains complete and stopped.

## Current: Phase 3 Fix 2 Complete And Stopped (2026-11-15)

All four authorized development trajectories, both readiness checks, twelve
recorded forks and selection are sealed. The guarded GPU runner (exec 53949,
host PID 5111) exited successfully. Do not restart learning, spend the unused
allowance, start Phase 4, or actuate the physical robot without a new request.
The active campaign is now complete at `phase3/solver-fix2-v3`; v1/v2 stay
preserved and excluded. Completed `--resume` is verification only.

Selected `proximal_3`, alpha .001, changed first proposals through EWC in
256/256 recorded events. Isolated cost is 1.43x/1.65x proximal-one for control/FD;
three iterations do not establish convergence. Both lagged-anchor objectives
failed at both scales. Current, historical and supervisor source fingerprints
and `mathematical_overview.ipynb` are unchanged.

| Seed | Method | Captured | A initial / mid / final | B initial / mid / final | Cumulative active / fit min |
|---|---|---:|---|---|---:|
| 196101 | Replay/EWC | 621 | 5/20 / 5/20 / 5/20 | 0/10 / 0/10 / 0/10 | 43.49 / 13.40 |
| 196101 | FD | 621 | 5/20 / 4/20 / 3/20 | 0/10 / 0/10 / 0/10 | 49.83 / 16.77 |
| 196102 | Replay/EWC | 622 | 5/20 / 4/20 / 5/20 | 0/10 / 0/10 / 0/10 | 44.96 / 14.01 |
| 196102 | FD | 621 | 5/20 / 5/20 / 5/20 | 0/10 / 0/10 / 0/10 | 49.25 / 17.70 |

This verifies the numerical repair, not hard-task learning or FD efficacy.
Both conditions share trainable banks/replay/EWC/ridge. Their inherited penalty
strengths were not freshly retention-matched; no significance or sample-savings
claim follows. Coverage/exploration is a next-step hypothesis for review.

The whole repair charge is 4.3381678497 h (v3 charge 14,117.4043 s plus 1,500 s
superseded-attempt carry); the 300 s closeout reserve is included. Cumulative
study charge is 22.2496703707 h. Four accepted-center/Fisher/queue/count audits,
25 completion envelopes, 4,989 frame blobs, 16 videos and 120 focused tests pass.
All 275 pre-resumption completion markers are preserved. The canonical notebook
is `numerical_experiment/picar/plan19_results.ipynb`; its 13 code cells run with
model/training/runtime imports blocked and its conclusion is last. Reporting
uses cumulative learner timers across resume, not last-invocation callback
timers. Scientific artifacts remain immutable. See the plan's Fix 2 execution
closeout for the scope and limitations; older notes below are historical.

## Historical: Restart In Progress (2026-11-15)

After the node reboot, CUDA was available and no experiment writer remained.
The guarded v3 `--stage all --resume` command is running in exec session 53949
(host PID 5111); its new log is
`cache/numerical_experiment/picar/phase3/solver-fix2-v3-resume-20261115.log`.
Both seed-196101 pilots remain sealed. The unfinished seed-196102 Replay/EWC
pilot is `units/pilot-260b692cb82f` (last parent counter 490/640; child mirror
495/640). The final seed-196102 FD pilot follows it. Do not start a second
writer. The nine-hour repair ceiling, scientific settings, and Phase 4 hold
are unchanged. Regenerate the canonical notebook in a fresh process at closeout.

The third pilot has now sealed: seed 196102 Replay/EWC captured 622 transitions;
A initial/mid/final counts are 5/20, 4/20, 5/20 and all B panels are 0/10.
Its accepted-center/queue/retirement/Fisher audit passed (566 events, 1,698
gradients, 1,073 archived records, eight eligible records). The final FD pilot
is `units/pilot-95149ae18f3b` and is active in the same session. Its parameters
are unchanged. Reporting now prefers checkpointed `timings.active_seconds`
over callback `active_seconds` for forecasts/throughput: the third pilot's
values are 2,697.7063 s cumulative versus 764.6256 s for the resumed invocation
only. Sealed artifacts are unmodified; this is a reader-side correction.
Forty solver-report tests pass. The notebook is refreshed after the third seal.

## Historical: Phase 3 Fix 2 Authorized; Implementation (2026-11-14)

The user authorized execution of Phase 3 Fix 2. The next action is the bounded
three-solver comparison, then selection of one solver for two paired 640-interval
development seeds. The separate ceiling is nine additional active compute
hours. Phase 4 and physical robot motion remain deferred. The earlier suggestion
to change steering coverage is superseded by the agreed solver repair first.

New scientific code lives in `numerical_experiment/picar_solver_repair/`.
Historical numerical files and completed runs must remain recoverable. The new
recipes are `proximal_1`, `lagged_1`, and `proximal_3`; all use the specified
live-gradient second moment and implicit ridge. The accepted recurrent center
advances once per statistical event; Fisher admission remains retirement-only.

Reuse `phase3/initialization-fix1-v1`'s sealed 515-history initializer and primary
four-draw Fisher, not the independent reference. The recorded source is its
Replay/EWC seed 195101 pilot. Early/late 64-event forks both start from the
common pre-stream initializer, rather than a future historical checkpoint.
Validation episodes remain disjoint. Candidate solver states, center history,
RNGs, cursor and resource ledger must survive `--resume`.

Current campaign root is
`cache/numerical_experiment/picar/phase3/solver-fix2-v3`. The entry point is
`python -m numerical_experiment.picar_solver_repair.campaign` with optional
`--source`, `--output`, `--model-path`, `--offline`, `--resume`, and
`--stage software|screen|all`. Supply a reused model directory explicitly or
allow the project-local pinned download. The canonical reporting notebook is
still `numerical_experiment/picar/plan19_results.ipynb`.

The first `solver-fix2-v1` diagnostic was stopped at 88 committed events to
correct validation context trimming before live admission. Its four completed
administrative units, partial diagnostic, source archive and STOP marker remain
preserved. Never resume v1 for selection or live learning. The corrected code
is `plan19-phase3-fix2-v2`; the active root will be `solver-fix2-v2`, and 700
seconds are charged to the same nine-hour allocation for the first attempt.
The remaining corrected allocation is 31,700 seconds. Full held-out contexts
are now evaluated without replay trimming; training/replay trimming is unchanged.

The numerical agent completed its implementation/tests and is closed. The
reporting agent is finalizing artifact-only diagnostics and version handling.
The main agent is integrating and running the corrected comparison. GPU access works in an
escalated execution session; the ordinary sandbox reports CUDA unavailable.
Use the user's `~/.venv/bin/python`, with `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`, and a `/tmp` Matplotlib cache. Do not silently run the VLM
experiment on CPU. Record the launched session and subsequent milestones below.

**Active execution:** corrected v2 is running the bounded `--stage screen`
command in the main agent's exec session. Its persistent log is
`cache/numerical_experiment/picar/phase3/solver-fix2-v2-screen.log`. The optional
existing-model/offline override is the same one recorded in the v2 resolved
configuration, not a new source dependency. At the next stage boundary, inspect
the sealed comparison, then use the same flags with `--stage all --resume` to
continue within the existing authorization. Do not start a second writer while
the screen process is active. Both coding agents have completed and are closed.
Focused checks passed 54 numerical/campaign tests and 34 reporting tests before
the corrected comparison. Full historical/scientific source hashes and the
reserved theory notebook were verified unchanged.

**Latest correction:** v2 subsequently stopped during `lagged_1` after nonfinite
predictions reached its retirement Fisher sampler. The new v3 supervisor records
that as a failed candidate while retaining exceptions for unrelated schema/data
defects. V2's two complete, full-context `proximal_1` measurements are explicitly
reused by seal after numerical-source/protocol checks; they are not extra
replicas. V1 remains excluded and stopped. V3 gives every method the one common
`.003 -> .001` scale correction if the base screen has numerical failure.
The first two attempts cost a conservative 1,500 seconds in the shared nine-hour
ceiling; corrected v3 has 30,900 seconds. The current screen log will be
`cache/numerical_experiment/picar/phase3/solver-fix2-v3-screen.log`.
Never resume v1 or v2 for live work. Start/resume v3 only, with identical model
override flags and `--stage screen` until its diagnostic review is complete,
then `--stage all --resume` under the same execution authorization.

**Current launch uses the recorded supervisor guard:** run
`python -m numerical_experiment.picar_solver_supervisor` with the same source,
v3 output, model/offline and resume/stage flags. Its source is sealed separately
in `solver-fix2-v3/supervisor-guard-v1`; all frozen v3 scientific fingerprints
are unchanged. It represents undefined scalar diagnostics as null plus signed
field metadata and isolates finite-output numerical failures without changing
updates or suppressing schema defects. Five guard tests and 34 reporting tests
passed. The active main-agent exec session is running the screen and writes
`cache/numerical_experiment/picar/phase3/solver-fix2-v3-supervised-screen.log`.
Use this guarded entry point for subsequent live continuation too. The older
v3 log preserves the strict JSON rejection of an infinite lagged diagnostic.

**Screen complete; live work admitted (2026-11-14):** all twelve recorded forks
and the selection are sealed. Both lagged objectives fail at both scales. After
visual review, use the guide's `proximal_3`, alpha .001, for BOTH live objectives.
Average old-pool gap is 9.8097 versus one-step 9.8476; live NLL is 23.4717 versus
23.7325. The three-step mean fit times are 1.1356 s control and 1.5163 s FD,
1.43x/1.65x one-step cost. Residuals do not establish convergence. No guide
override, new hyperparameters, or efficacy claim is added. Pre-live planning
range is roughly 3.4--4.8 h for four pilots plus short checks/closeout, inside
the same nine-hour additional ceiling. The guarded screen process exited cleanly.
Continue v3 through the guarded entry point with `--stage all --resume`; do not
restart the screen, any historical campaign, or another writer. The new live log
is `phase3/solver-fix2-v3-supervised-live.log`. Reporting includes compact tables,
an objective residual panel, and live-derived remaining-time ranges. Scientific
sources are still frozen; report-only source is excluded from those hashes.

**Live checks complete; pilots active:** both 48-interval checks sealed with
38 captured transitions, 35 optimizer events, zero inference-deadline misses
and no non-startup fallback commands. Mean fit service times are 1.1573 s
Replay/EWC and 1.3223 s FD; maximum actor model-lease waits are .3307 s and
.3837 s. Wall costs are 117.6051 s and 126.0766 s. These are latency checks,
not acquisition evidence. The live-derived four-pilot planning range is
2.38--4.76 h; 7.96 h remains under the shared repair ceiling at admission.
The main agent's active exec session is 49101, logging to the guarded live log
above; first pilot is Replay/EWC seed 196101. Both agents remain closed.
The latest focused acceptance is 96 tests in 11.35 s. Canonical notebook
progress shows stage counts, completed pilots, and the live-derived forecast.

**First pair sealed; second seed running:** seed 196101 captured 621 transitions
in each condition. Replay/EWC's A counts were 5/20 -> 5/20 -> 5/20; FD's were
5/20 -> 4/20 -> 3/20. All B panels were 0/10. Wall costs were 2,609.5907 s and
2,990.0184 s. Both had three non-startup deadline-related fallback commands.
The observed full-pilot FD/control wall ratio is 1.146, not an isolated causal
FD overhead estimate because visited histories and episode lengths differ.
Do not infer an FD benefit or matched retention from this pair. Settings remain
unchanged for seed 196102, whose Replay/EWC run is now active in session 49101.
At second-pair admission the whole repair charge was 2.604 h and the remaining
two-pilot planning range was 1.17--2.33 h, below the 6.40 h remaining ceiling.

Both completed recovery states passed accepted-center/Fisher/queue/counter
checks. They contain 559/567 events and 1,677/1,701 gradients. Retirement IDs
are unique and include the 515 initial bulk records: total archived counts are
1,066/1,074; both have eight eligible replay records. The archival counter must
equal the complete retired-ID list length, not 515 plus that length. An initial
ad hoc audit incorrectly double-counted initialization; its corrected audit
passed without any artifact or numerical change. Current reporting additionally
shows endpoint preselected clips, compact runtime health and an outcome-based
conclusion; final fresh regeneration is needed because the running process has
its earlier report module cached. The 38 solver-report tests passed before the
last small quantile-label refinement; rerun focused acceptance at closeout.

## Historical: Phase 3 Fix 1 Complete And Stopped (2026-11-13, 18:43 PST)

**No experiment tmux session or CUDA compute process remains.** The bounded
initialization repair and both two-seed development comparisons are complete.
The independent Fisher reference is also sealed. Do not resume these completed
campaigns for more learning. Phase 4 and physical actuation remain unauthorized;
the development decision is not ready to propose confirmation.

### Results And Limits

Initialization used nested, independently refitted 259- and 515-history
training sets with the same 65-history, 12-episode validation pool. Accepted
validation NLL was 19.3439 and 19.3282, respectively; the sealed growth rule
stopped before 1,024 because improvement was below 1%. Initial fitting kept
STM neutral, fitted LTM over multiple KL/NLL passes, and selected validation
checkpoints explicitly. Both full selected-checkpoint A panels were 5/20:
joint-NLL fitting did not erase that initial skill, but did not improve it.
Training plateau/selection passed the declared rule, not proof of a global
MLE, sufficient data, or a high-quality navigation policy. Teacher argmax
actions remained forward-only; the coverage limitation is preserved.

The selected `units/bootstrap-0b5531537490` bulk Fisher equally weights all
515 training histories at the accepted anchor (four model-label draws/history),
with no initial pi/EMA/ridge. All 515 histories enter the archive once, and
online replay starts empty. The independent reference used 16 draws/history
on the SAME histories/anchor without changing primary D0 or model parameters:
primary trace 176.1032, reference trace 200.3178, global relative diagonal L2
difference .24687. The eight-history subset remains substantially noisier
(primary-four versus reference-sixteen relative difference 1.2166). The new
reference is not truth and does not retroactively change the failed initial
small-subset stability guide or the experimental settings.

| Seed | Method | Captured intervals | Endpoint A / 20 | Endpoint B / 10 | Training successes / episodes |
|---|---|---:|---:|---:|---:|
| 195101 | Replay/EWC | 634 | 0 | 0 | 0 / 37 |
| 195101 | STM FD | 619 | 4 | 0 | 1 / 44 |
| 195102 | Replay/EWC | 628 | 4 | 0 | 1 / 40 |
| 195102 | STM FD | 628 | 4 | 0 | 1 / 38 |

Every pilot started at A 5/20 and B 0/10; B remained 0/10 at midpoint too.
FD preserved more endpoint A skill on the first seed and tied on the second.
Retention matching passed only the second seed. There are only two development
seeds sharing one initialization and fixed panels: no fresh significance,
retention-equivalence or sample-savings claim is supported. Training had only
three successes in 159 episodes. Prioritize a separately versioned diagnostic
of steering coverage, successful exploration and online acquisition before
confirmation; this repair does not isolate the acquisition bottleneck.

### Final Resource Ledger

- Main repair: 25,536.977911302 s = 7.093605 h, including its 600 s closeout allowance.
- Reference: 354.395144109 s charged, including its 60 s closeout allowance.
- Combined repair: **25,891.373055411 s = 7.192048 h**, within the ten-hour cap.
- Historical study charge remains 38,590.03602203 s; cumulative study is 17.911503 h.
- Unspent repair allowance is 2.807952 h; it is not authorization for a new experiment.

### Integrity And Continuity

All 45 completion seals across the two historical Phase 3 campaigns, the main
repair and the reference passed scalar verification. All 74 frozen scientific
source hashes and the reserved theory notebook are unchanged. Each completed
pilot recovery checkpoint is `done`, with no pending labels/ready updates,
unique exactly-once retired IDs, and no initialization records in online replay.
Completed main `--resume` preserved all 25 seals and returned without importing
Torch. The full canonical notebook's 12 code cells executed with Torch, model,
training and numerical runtime imports blocked. Final focused acceptance:
**146 tests passed in 38.10 seconds**. `git diff --check` passed.

The main process logged a post-seal reporting failure: generic held-out curves
treated metadata-only campaign versions as efficacy versions, then the frozen
handler attempted to checkpoint the already completed parent. Scientific
artifacts were unaffected. Reporting-only filtering and non-confirmatory
campaign classification are repaired and regression tested; the traceback
remains historical evidence. Do not edit frozen code or reopen completed runs
to erase it. The notebook was rebuilt from a fresh process with all final
outcomes, an independent-reference table, a concise retention summary and a
qualified next-action recommendation.

Canonical report: `numerical_experiment/picar/plan19_results.ipynb`.
Main artifacts: `cache/numerical_experiment/picar/phase3/initialization-fix1-v1`.
Reference: `cache/numerical_experiment/picar/phase3/initialization-reference-v1`.
Both numerical and reporting agents are closed. The notes below are historical;
their old resume commands are not instructions to restart completed learning.

## Historical: Main Repair Sealed; Fisher Reference Running (2026-11-13, 18:36 PST)

All four repaired development pilots and the parent orchestration are sealed.
Main repair charge: **25,536.977911302 seconds = 7.094 hours**, including the
600-second closeout allowance. No learning remains to resume in this campaign;
never reopen its completed units. Initial A success was 5/20 in all pilots.
Endpoint results: Replay/EWC 0/20 and 4/20; STM FD 4/20 and 4/20 for seeds
195101/195102, respectively. Every initial/midpoint/final B panel was 0/10.
The recorded development decision is NOT ready for confirmation. Retention
matching passed only the second seed. No sample-savings claim is supported.

The scientific seals were written before a final notebook-refresh exception.
The generic held-out renderer treated metadata-only campaign versions as
efficacy versions, then the frozen exception handler tried to checkpoint the
already completed parent. This did not change any scientific payload or seal.
Reporting-only version filtering is repaired, completed campaign metadata is
explicitly non-confirmatory, and all 85 reporting regressions pass. Do not edit
the frozen campaign to erase its historical traceback. Rebuild the notebook
from a fresh reporting process after the reference finishes.

The sole current GPU writer is tmux `plan19-fisher-reference`, running
`python -m numerical_experiment.picar_initialization_reference` with
`--source-campaign cache/numerical_experiment/picar/phase3/initialization-fix1-v1`,
`--output cache/numerical_experiment/picar/phase3/initialization-reference-v1`,
`--draws 16 --offline --resume` and the same optional model-path override.
Its log is `cache/numerical_experiment/picar/phase3/initialization-reference-v1.log`.
This is an independent model-label diagnostic at the accepted anchor and all
515 selected training histories; no new data, fit, replay or primary-Fisher
replacement. It has a 1,200-second local cap and shares the ten-hour repair
ceiling. If interrupted, resume only this sidecar with identical flags and
preserve its charged ledger. Phase 4 and physical actuation remain unauthorized.

## Historical: Resuming Phase 3 Fix 1 (2026-11-13, 16:26 PST)

Update after the first repaired development pair sealed: control seed 195101
captured 634 intervals, A 5/20 -> 0/20 -> 0/20, B 0/10 throughout. FD on the
same seed captured 619 intervals, A 5/20 -> 2/20 -> 4/20, B 0/10 throughout.
These are initial/midpoint/final fixed-panel counts. FD has a descriptive
retention lead, not retention matching, new-task efficacy or sample savings.
Its accumulated trajectory time is 43.31 minutes versus 51.61 for the control;
do not use the resumed callback's shorter `active_seconds` as total cost.
The second-seed control `units/pilot-42a8581497e7` is now running; the second
FD trajectory remains pending. At 139/640 intervals the repair charge was
5.522 hours. Both conditions still use the same frozen settings.

Reporting-only tests passed 67 checks after adding a compact completed-pilot
table and correctly recognizing the independently versioned optional Fisher
reference. The four diagnostic plot groups were inspected. Scientific hashes
remain unchanged. The live worker imported an earlier reporting module, so
rebuild the canonical notebook from a fresh process at closeout to include
the newest reporting-only corrections.

The user authorized resumption. The saved parent/active journal checkpoints,
21 completed-unit scalar seals, all 74 scientific source hashes and the
unchanged theory notebook passed verification. Actual PyTorch CUDA works in
the execution environment. Only the new initialization repair's STOP marker
was intentionally removed; historical campaign stop markers remain untouched.
Resume uses the identical optional model-path/offline override and `--resume`
in tmux `plan19-phase3-fix1`. Continue FD seed 195101's midpoint evaluation
from 329 captures, then the remaining admitted development pilots. Preserve
the existing 17,376.676289215-second repair charge and ten-hour cap. Do not
start Phase 4, additional collection/refitting, or physical robot operation.

The pause record below is historical; its checkpoint details still describe
the resume boundary, not a new trajectory.

## Historical: User-Requested Pause (2026-11-13, 12:09 PST)

The user needs the node. Phase 3 Fix 1 stopped cooperatively, with its STOP
marker deliberately left in place. No tmux server or CUDA compute process
remains. Do not restart anything until the user asks. Phase 4 and physical
actuation remain unauthorized. Earlier running-state notes below are historical.

The saved active unit is `units/pilot-f97f546f5194` under
`cache/numerical_experiment/picar/phase3/initialization-fix1-v1`:
STM FD, seed 195101, **329/640 incoming intervals**, stage
`midpoint_evaluation`. The midpoint panel is not complete; its live simulator
session/history and learner/RNG checkpoint are preserved. No pending labels
or ready updates remain at this boundary. The preceding Replay/EWC trajectory
`units/pilot-68cb4a663a6c` is complete (634 captures); the second-seed pair has
not started. Do not interpret an unfinished FD panel as an outcome.

The parent checkpoint preserves **17,376.676289215 seconds = 4.827 hours** of
repair compute, leaving about **5.173 hours** of the separate ten-hour cap,
including the still-required closeout allowance. Preserve the historical
38,590.03602203-second study charge separately. Initialization is sealed at
515 training histories with 65 fixed validation histories; no new collection
or refitting is needed on resume.

Handover verification passed: 21 completed-unit scalar seals, all 74 frozen
scientific source hashes, trusted parent/active journal recovery checkpoints,
and the unchanged mathematical overview. Opening the journals for verification
did not advance training or write a new checkpoint. The stop traceback is the
expected cooperative `InterruptedError`, not a numerical failure.

### Resume Only On Request

From the checkout root, retrieve the same optional model-path override from
the ignored `orchestration/config.json` (`model_source`). Then remove only
this repair's root STOP marker intentionally and run the same command/flags,
preferably in tmux `plan19-phase3-fix1`:

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  MPLCONFIGDIR=/tmp/plan19-fix1-matplotlib \
  ~/.venv/bin/python -u -m numerical_experiment.picar_initialization.campaign \
  --output cache/numerical_experiment/picar/phase3/initialization-fix1-v1 \
  --teacher-source cache/numerical_experiment/picar/phase1/fix3-v2/units/bootstrap-5f42cbe3df8a \
  --model-path "<same-existing-model-directory>" --offline --resume \
  >> cache/numerical_experiment/picar/phase3/initialization-fix1-v1/run.log 2>&1
```

Do not clear historical campaign STOP files or reopen their completed runs.
Continue the saved FD midpoint evaluation and remaining admitted pilots before
closeout. Both coding agents are closed. The optional independent 16-draw
Fisher reference entry point and tests are implemented in
`numerical_experiment/picar_initialization_reference.py`, outside the frozen
scientific roots, but **have not run**. Admit it only after the main campaign
finishes and only within the shared remaining repair cap; never replace the
primary four-draw Fisher or launch it alongside policy trajectories.

## Historical: Phase 3 Fix 1 Started (2026-11-13, 07:28 PST)

The user authorized executing the new **Phase 3 Fix 1** initialization repair.
This supersedes the historical instruction below not to begin new learning,
but does NOT authorize Phase 4 or physical actuation. The repair has its own
ten-hour additional active-compute ceiling; the old whole-study 20-hour target
has been relaxed. Preserve the historical 10.719-hour charge separately.

Implementation and acceptance passed in `numerical_experiment/picar_initialization`.
The new campaign is running in tmux `plan19-phase3-fix1`; progress and `run.log`
are under `cache/numerical_experiment/picar/phase3/initialization-fix1-v1`.
Both historical Phase 3 campaigns are
complete and must never be reopened for learning. Their 19 completion seals,
scientific source hashes and the reserved theory notebook passed read-only
verification. Actual PyTorch CUDA works with the updated driver and venv.

The new runner is `python -m numerical_experiment.picar_initialization.campaign`:
use `--teacher-source` pointing to the original completed Fix3 bootstrap,
`--output cache/numerical_experiment/picar/phase3/initialization-fix1-v1`,
and `--resume`. Default model loading downloads the pinned project-local
checkpoint; optional `--model-path`/`--offline` overrides must match on resume.
The numerical/reporting components are integrated and scientific sources are
frozen. All 69 combined repair tests passed, including the tiny CPU
collection/fit/bulk-Fisher/online-handoff smoke. Sealed runner acceptance also
passed 69 tests. Both coding agents are finished and closed. Report-only
changes may continue without changing the scientific fingerprint.

The coordinator owns campaign/config/data and their tests; the two authorized
agents supplied numerical fitting/Fisher tests and reporting/tests.
Initial collection stays at A; split entire episodes, keep a fixed validation
panel, fit only LTM while STM is neutral, then bulk-average model scores at a
fixed accepted anchor without initial pi/EMA. All selected training records
enter EWC once; online replay starts empty. Offline block checkpoints retain
optimizer/order/RNG/Fisher cursors. Reused online collection safety is unchanged.

The data-size ladder is approximately 256/512/1024 training histories, with
at least two rungs attempted when admitted. Select among completed fits using
minimum joint NLL on the same fixed validation pool, not navigation luck.
Publish diagnostic epochs while fitting, plus distillation/accepted-NLL
easy-A checks. Reserve budget for paired development when initialization is
usable; report omissions rather than fabricate failures. STOP/SIGTERM and
abrupt-halt recovery remain mandatory. The campaign includes a conservative
600-second pre-execution validation charge, in addition to timed acceptance,
and a conservative 600-second closeout charge. Data-growth decisions are
sealed units: resumption must not reinterpret a prior four-hour cutoff and
silently skip already admitted data-size rungs. STOP/SIGTERM checkpoint the
active unit; a restart can use the same command and flags with `--resume`.

The older notes below are historical, not instructions to resume their writers.

### Fix 1 Initialization Sealed (2026-11-13, 10:59 PST)

Two fitted rungs completed: 259 and 515 training histories, sharing a fixed
65-history, 12-episode validation set. Selected validation NLL was 19.3439
and 19.3282 respectively. The sealed growth rule stopped before 1,024 because
the improvement was below 1%; no extra collection is planned. The selected
515-history asset is `units/bootstrap-0b5531537490` under the new campaign.
The first Replay/EWC pilot is running (seed 195101),
with common LR .003, one optimizer step, pi .025, variance 16, exploration
SD 4 and ridge .1. Positive FD passed numerical checks; the middle initial
gradient-ratio strength is 1742.42640280057 at epsilon .01. No efficacy result
is yet available.

Bulk Fisher trace is 176.1032, with four model-label draws on every training
history, no pi/EMA, and empty initial replay. Numerical validity passed but
MC stability remains qualified: two-versus-four draws relative L2 .2001;
an independent four-draw check on only eight histories differed by .7945.
Do not call this proof of Fisher convergence. Initial full A/B evaluations
and their clips are sealed separately; inspect their summaries for outcomes.

A resumed numerical sidecar is implementing a separate, optional 16-draw
reference entry point outside all frozen source roots. It must run ONLY
after the main GPU campaign exits and only inside the remaining shared
ten-hour repair cap. It must not change primary D0 or either comparison.
This is a diagnostic follow-up to the observed MC flag, not a new training
condition. Do not launch it concurrently with policy trajectories.

### Fix 1 Controlled Recovery Check (2026-11-13, 08:00 PST)

The first new collection sealed 152 incoming intervals and 123 usable teacher
histories in 520.8 seconds. The first selected training rung has 259 histories
across 48 episodes; its fixed validation has 65 histories across 12 episodes.
All 370 available teacher histories still use forward as the dominant action;
this is an explicit coverage limitation, not an ignored success filter.

A planned STOP at the first live GPU fit committed KL epoch 7, next cursor
223, with 1,777 KL exposures and seven diagnostic rows. The worker exited
cleanly, CUDA contexts were released, and trusted journal/state verification
found no source mismatch. The same command/flags restarted in tmux; check
that exposures continue from that cursor, never from zero. This is a recovery
test, not a new scientific version or extra data. Reporting-only fixes make
return bins numeric and KL tail convergence visible; 79 reporting regressions
passed. Mathematical/numerical sources remain frozen.

## Phase 3 Complete And Stopped (2026-11-12, 21:35 PST)

**No tmux worker or CUDA compute process remains.** Phase 3 completed its
bounded development/check-in scope; its scientific readiness guide FAILED.
Do not resume learning or start Phase 4 without a new user instruction.
Both orchestration versions are immutable: original
`superseded_after_ridge_screen`, supervisor `ready_for_checkin`.

Five completed control trajectories are preserved. Easy-A success began at
5/20 throughout. Ridge .1 and .03 ended at 0/20; ridge .3 ended at 2/20 on
seed 193101 and 0/20 on 193102. The gentle variance-16 repair ended at 2/20,
but preserved 5/20 at midpoint. Its average A deterioration was 3.54 points
versus 16.11 for original ridge .3 on the same seed. This is a descriptive
multi-setting lead, not an isolated causal effect or replicated repair win.
Hard-B success remained 0/10 at all measured checkpoints in every candidate.
The recorded rule retained original ridge .3 for replication; its mean
endpoint A loss was 20 points, above the ten-point guide.

FD numerical resolution PASSED at epsilon .005/.01/.02; selected epsilon
.01, initial gradient-ratio strengths .8177/8.1773/81.7725. Reversible scratch
fits measured .2653 s control versus .3799 s middle-FD (1.43x update cost),
not whole-trajectory overhead. No FD trajectory was admitted: estimated
69.8 minutes before safety margin versus 63.8 minutes remaining excluding
closeout. Do not treat this omission as an FD failure or claim sample savings.

The final parent charge is **17,870.41121297 s = 4.964 h for Phase 3**,
including one five-minute closeout allowance; cumulative study charge is
**38,590.03602203 s = 10.719 h**, leaving 9.281 h of the 20-hour target.
Use this final parent ledger, not the earlier forecast's slightly lower
snapshot. Selected control processing measured 3.12 s/captured interval.
Full FD trajectory cost remains unmeasured, so no certified total fresh-study
forecast is available; the forecast's panel-time rows are evaluation-only.

Final checks: all 19 completion seals, 72 frozen scientific sources and the
unchanged mathematical overview verified. Actual completed resume verified
unchanged seals without Torch. All 11 notebook code cells executed without
Torch; final reporting acceptance passed 53 tests. Supervisor runtime
acceptance passed 79 tests. Both new trajectories drained to zero pending
updates. The Phase 3 artifact tree occupies about 1001 MB. Notebook conclusions
now include a compact outcome table, the versioned repair rationale and
explicit budget omissions. No Phase 4 or physical actuation occurred.

The live-stage notes below are historical. Completed `--resume` may verify
and return, but must not reopen either campaign for additional learning.

## Selected Control Replication (2026-11-12, 20:38 PST)

The variance-16 gentle repair is complete at
`development-retention-v1/units/pilot-fc99666062ff`: 623 captures, easy-A
5/20 initially, 5/20 at midpoint, 2/20 finally; hard-B 0/10 throughout.
Its endpoint retention did not improve over the original ridge-.3 control,
so the recorded conservative selection rule kept `control_ridge_0.3`.
Do not change that rule after observing outcomes. The second-seed control is
running at `units/pilot-b3f65d164c07` (seed 193102), using the ORIGINAL
variance-1 initialization/Fisher/panels, LR .003, two steps, pi .025 and SD 8.
This fallback is intentional; the variance-16 Fisher must not be used for it.

The supervisor remains the sole writer. Resume ONLY its command below.
The new initialization (`bootstrap-6d4c1a77310c`) sealed successfully:
247 original eligible demonstrations, 239 retired and eight still eligible;
zero new teacher transitions, no old Fisher reuse. Supervisor acceptance
passed 79 tests in 153.69 s. The reporting-only rationale update passed
35 tests; all 72 supervisor scientific source hashes remain unchanged.

## Retention Supervisor Owns Continuation (2026-11-12, 19:33 PST)

All three ridge candidates are complete. Ridge .3 ended at 2/20 easy-A
successes versus 5/20 initially; hard-B stayed 0/10. Thus none met the
retained-acquisition guide. The original orchestration is now immutable and
sealed as `superseded_after_ridge_screen`, charged 10,388.198198782002 s.
No plasticity-repair trajectory was started. Its STOP must remain in place.

ONLY tmux `plan19-phase3-retention` is running, at
`phase3/development-retention-v1`. Resume ONLY its command below after any
later interruption; never restart the original campaign. The supervisor is
performing acceptance before the fresh variance-16 initialization and shared
retention-directed repair (ridge .3, LR .001, one step, pi .01, SD 4).
Historical controls remain explicit references, not newly acquired data.
The old and new orchestration charges still share one six-hour ceiling.

## Resumed (2026-11-12, 19:02 PST)

The user authorized resumption. Both STOP files were intentionally cleared,
and tmux `plan19-phase3` plus `plan19-phase3-retention` restarted with the
same model-path/offline overrides and `--resume`. The supervisor uses
`--wait`; it has not taken over yet. The original runner must finish the
saved ridge-.3 trajectory before that decision. All seven completed units,
both partial journal chains/recovery snapshots, and all 69 scientific source
hashes passed read-only verification before launch. PyTorch CUDA is available
in the tmux execution environment (RTX 4070, torch 2.10.0+cu128). The restricted
sandbox alone cannot access CUDA; do not mistake that for a driver failure.

The prior pause record below is historical, not the current running state.
Resume carries forward the durable 8,658.508928239-second Phase 3 charge;
the five-hour node-borrowing pause is excluded. Continue monitoring to the
development check-in only. Phase 4 and physical operation remain unauthorized.

## Paused For Node Handover (2026-11-12, 21:56 UTC)

**Both tmux workers have exited cleanly and CUDA has no remaining compute
process.** The user requested the pause; do not resume until instructed.
STOP files exist in BOTH `phase3/development-v1` and
`phase3/development-retention-v1`. The latter stopped while waiting; it has
not taken over, sealed the source, initialized a model or acquired data.

Two controls are complete and immutable: ridge .1 (`pilot-e50a89383d05`,
627 captures) and .03 (`pilot-6ff65bb6993d`, 625 captures). Both went from
5/20 easy-A successes to 0/20 at midpoint/final, with 0/10 hard-B throughout.
The ridge-.3 control (`pilot-470d6557397f`) is checkpointed at **330/640
incoming intervals, midpoint evaluation**. Resume its saved evaluation and
trajectory; do not restart it or classify its missing endpoints as failures.

The source parent durable charge is **8,658.508928239 s (2.405 h)**; prior
study plus this charge is **29,378.133737569 s (8.161 h)**. About 3.595 h of
the shared six-hour Phase 3 ceiling remains, including final verification.
Idle node-borrowing time must not be charged. This is a remaining allowance,
not an estimate that every proposed candidate will fit.

Restart: intentionally remove BOTH STOP files, launch the original campaign
using the command below, and launch the conditional supervisor with `--wait`
using its command below. Supply the same optional model-path/offline overrides
from the source `orchestration/config.json`. Use separate tmux sessions as
before. The source must finish its third ridge candidate; the supervisor then
decides whether to leave the original running or checkpoint/seal it and take
over. After a takeover, never resume the original writer again.

Shutdown verification checked all seven completed unit seals, all 516 parent
journal entries and all 1,035 active-pilot journal entries, including the
required recovery snapshots. All 69 frozen historical scientific sources and
the mathematical overview remain unchanged. The notebook was refreshed by
the runner's clean-stop handler. Independent supervisor/reporting acceptance
passed **59 tests in 49.26 s**; its agent also passed 78 combined runtime,
analysis and probe checks. All coding agents are closed. No FD development,
Phase 4 or physical robot operation has started.

## Phase 3 Authorized (2026-11-12)

The user authorized Phase 3 only. The campaign started at 19:39 UTC in tmux
session `plan19-phase3`, at
`cache/numerical_experiment/picar/phase3/development-v1`. Acceptance passed
109 tests in 159.47 s. CUDA initialization completed and is sealed at
`units/bootstrap-715cd4052b9c` (850.57 s, fresh KL/NLL/Fisher). The recorded
temperature is 2.8375266913896, action variance 1. All 247 original actions
were forward; no new teacher data were acquired. Scratch calibration selected
LR .003, two optimizer steps and exploration SD 8. The first ridge-.1 pilot
is complete at `units/pilot-e50a89383d05`: 627 captures, no pending updates;
easy-A 5/20 initially, 0/20 at midpoint/final; hard-B 0/10 throughout.
These are NEW development layouts, not Phase 2's 7/20 panel. The lower-ridge
.03 candidate is now running at `units/pilot-6ff65bb6993d`; .3 follows.
The new independent package is
`numerical_experiment/picar_phase3`; the scientific sources from Phases 1/2
and the mathematical overview remain frozen. Two reused Sol sidecars own
the development runtime/tests and reporting/tests. The coordinator owns
calibration, selection, budget, scheduler and GPU execution. Both sidecars
have finished and are closed. Runtime tests passed 30 checks and 42 earlier
regressions; the coordinator independently passed 47 runtime/probe checks.
Seven minutes of conservative pre-execution CPU validation and five minutes
of closeout allowance count inside the six-hour phase ceiling.

### Conditional Retention Repair Armed

After the first pilot's 25-point retention loss and zero hard-task gain, the
coordinator is preparing a separate `picar_phase3_retention` supervisor. The
original three ridge candidates CONTINUE UNCHANGED. Independent retention and
reporting acceptance passed 59 tests in 49.26 s. Before the user-requested
pause, the frozen supervisor waited in tmux `plan19-phase3-retention`
(launched about 21:28 UTC), logging to
`phase3/development-retention-v1/run.log` under the PiCar cache root.
It will take over only after all three are complete and none
provides a retained hard-task gain; otherwise the original schedule continues.
Do not resume the original past that gate without reviewing this amendment.

The refined common repair is LR .001, steps 1, pi .01, exploration SD 4 and
action variance 16 (same tau 2.8375266913896/ridge selected from original
controls). This replaces the unexecuted higher-plasticity repair. It needs
fresh KL/NLL/Fisher initialization from the recorded demonstrations because
Sigma_A changes; old Fisher/panels cannot be transplanted. No teacher queries
or retired-data online replay are added. The same six-hour Phase 3 budget
carries across the source handover; waiting overlaps existing work and is
not charged twice. No Phase 4 is authorized. See the appended plan refinement.

Supervisor resume command (same model-path/offline overrides as the source):

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase3_retention.campaign \
  --output cache/numerical_experiment/picar/phase3/development-retention-v1 \
  --source-campaign cache/numerical_experiment/picar/phase3/development-v1 \
  --resume --wait
```

Before the three-control gate, resume both the original runner and waiting
supervisor. After the source orchestration is sealed as
`superseded_after_ridge_screen`, resume ONLY the supervisor. Never remove the
source STOP after takeover. New-run STOP/SIGTERM preserves the current repair
checkpoint. The waiting supervisor performs cheap discovery and validates all
seals before taking ownership; idle waiting is not charged twice.

The appended Phase 3 execution recipe records the exact calibration and
candidate sequence. Use feature plus projection RMS, a fresh initial Fisher
for any changed likelihood scale, three ridge candidates and up to two
documented repair rounds. Attempt two development seeds and report any budget
omissions. Initial/midpoint/final fixed-layout evaluation is descriptive, not
confirmation or a sample-savings estimate. The maximum Phase 3 charge is six
hours inside the cumulative 20-hour target; prior charge is 20,719.62480906 s
(5.76 h). No automatic Phase 4 or physical actuation is authorized.

The standalone resume command, once acceptance passes, is:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase3.campaign \
  --output cache/numerical_experiment/picar/phase3/development-v1 \
  --teacher-source cache/numerical_experiment/picar/phase1/fix3-v2/units/bootstrap-5f42cbe3df8a \
  --prior-review cache/numerical_experiment/picar/phase2/readiness-rms-v1/reviews/closeout-v1 \
  --resume
```

Supply the same optional model-path/offline overrides recorded in the resolved
configuration. Default loading downloads the pinned model into project cache.
Never run a duplicate writer. STOP/SIGTERM checkpoint; remove STOP only before
intentional resume. All old completed runs remain immutable. The runtime
source/config fingerprint is frozen when the campaign starts; report-only
edits may continue. Check `orchestration/progress.json` and `run.log` for state.

## Phase 2 Complete (2026-11-12)

**Latest state, 17:56 UTC: complete and stopped for review. GPU is free; no
tmux job remains.** Both original baseline pilots and both normalization
rescues retained 7/20 easy and 0/10 hard successes. Captures were 620/619 for
the baselines, 635 for feature RMS and 623 for feature plus projection RMS.
Each finished with zero pending updates. The original orchestration is sealed
as `superseded_after_baselines`; the rescue orchestration is sealed as
`ready_for_checkin` with 12 completed child units. Completion is NOT a
readiness pass: the operational hard-panel improvement guide was not met.

Feature RMS alone left the initialization projection saturated. Adding
projection RMS produced zero saturated coordinates on the archived probes,
median local tanh slope .462 and maximum instruction logit difference .051
(originally about 4.77e-7). This repaired measured local sensitivity, not
instruction following or navigation learning. No FD comparison was run.

Phase 2 campaigns charged 12,290 seconds (3.41 h). Including a five-minute
conservative final-verification allowance, Phase 2 totals 3.50 h, inside its
shared five-hour cap. The conservative cumulative study charge is 5.76 h, leaving 14.24 h
of the 20 h target before additional development. Measured processing was
2.92 s/captured interval. Four pairs of 1,200 intervals/condition would take
7.8--11.7 h processing plus 1.41 h evaluation under hypothetical FD cost
multipliers 1--2; six pairs of 2,400 would take 23.4--35.1 h plus 2.12 h.
FD overhead is UNMEASURED and later development/closeout are extra. These are
conditional forecasts, not admitted studies. First resolve common-controller
learning. No Phase 3, additional rescue or physical robot work is authorized.

Closeout passed 124 focused Phase 2 tests and 51 final reporting tests. All
23 pre-closeout seals verified, and all ten notebook code cells executed
without importing Torch. Completed `--resume` verified unchanged seals without
Torch or GPU work. All 63 scientific sources match the archived fingerprint;
the mathematical overview is unchanged. Final outcome and sensitivity plots
were visually inspected. The separately sealed `reviews/closeout-v1` records
the verification and extra allowance without mutating the campaign forecast.
The complete Phase 2 artifact tree occupies about 854 MB.
The final report refresh exposed a metadata-only closeout review being treated
as a scientific experiment version. A reporting-only classification fix and
regression test resolved it; no sealed artifact or scientific source changed.
All ten notebook cells and all 24 seals (including closeout) verified afterward.

### Completed Execution Details And Recovery History

The user authorized Phase 2 only, ending at the readiness/forecast check-in.
Implementation is in `numerical_experiment/picar_phase2`, separate from the
frozen Phase 1 numerical sources. See the appended Phase 2 execution recipe
in `plan19.md`: compatible initial asset reuse, 20 easy/10 hard fixed layouts,
two 640-interval baseline pilots, at most two bounded common rescue recipes,
five active compute hours maximum. No FD comparison, Phase 3 or physical
operation is authorized. Preserve all prior artifacts and the theory notebook.

Two bounded Sol 6.1 Extra High sidecars implement the evaluated runtime/tests
and artifact-only reporting/tests. The coordinator owns configuration,
scientific integration, scheduler, resource limits and GPU execution. Their
write sets do not overlap. This records the plan's explicit routing settings;
token/credit usage is unavailable, not inferred.

Both sidecars are now complete/closed. Runtime recovery tests and reporting
acceptance passed; integration additionally fixed the language unit's full
checkpoint publication and action-variance compatibility checks before
freezing sources. The original campaign ran in tmux session `plan19-phase2`:
`cache/numerical_experiment/picar/phase2/readiness-v1`. Its `run.log` captures
the invocation, and `orchestration/progress.json` tracks the current stage.
Five minutes of conservative pre-execution validation reserve are included
in the charged phase allowance. The runtime source archive and resolved
recipe are immutable child artifacts; report-only improvements are excluded
from the scientific fingerprint.

Resume command from the checkout (also supply the same optional model-path
and offline overrides recorded in `orchestration/config.json`):

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase2.campaign \
  --output cache/numerical_experiment/picar/phase2/readiness-v1 \
  --bootstrap-source cache/numerical_experiment/picar/phase1/fix3-v2/units/bootstrap-5f42cbe3df8a \
  --resume
```

Do not launch a second writer while tmux is running. A root STOP file or
SIGTERM requests a checkpointed halt; remove STOP before intentional resume.
The initial asset has passed the compatibility/disjointness audit. No
repeated KL or unrecorded use of retired data is allowed. The original
52 scientific sources and mathematical overview retain their hashes.

### Targeted Conditioning Rescue Handover (2026-11-12)

The first 640-interval baseline completed at its episode boundary after 620
captures: easy-A remained 7/20, hard-B remained 0/10. The second planned seed
finished after 619 captures with the same outcomes. Initial frozen teacher readiness was also 7/20 easy-A; both
teacher and student succeeded on 7/11 clear and 0/9 blocked layouts.

The language probe changes Qwen features but barely changes action logits
(maximum difference about 4.77e-7). An artifact-only CPU reconstruction found
all projected coordinates saturated at abs(tanh(z)) > .99; median absolute
preactivation was 11.59 and median local tanh derivative zero in float32.
The sealed diagnostic is `readiness-v1/reviews/feature-health-v1`. This is a
concrete conditioning concern, not isolated causal proof or an STM result.

The appended plan amendment replaces the two UNEXECUTED ridge/floor rescues
with feature RMS normalization, then optionally feature plus projection RMS
normalization. All other scientific settings and parameter counts stay fixed.
Each new model reconstructs the 247 original demonstrations and receives fresh
weights, optimizer state, one-time KL, likelihood fit and Fisher. No historical
model-dependent Fisher is transplanted into a different forward model.

The supervisor ran in tmux session `plan19-phase2-rms` and is now complete:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase2_rescue.campaign \
  --output cache/numerical_experiment/picar/phase2/readiness-rms-v1 \
  --source-campaign cache/numerical_experiment/picar/phase2/readiness-v1 \
  --teacher-source cache/numerical_experiment/picar/phase1/fix3-v2/units/bootstrap-5f42cbe3df8a \
  --resume --wait
```

Use the same optional model-path/offline overrides as the original command.
Its log is `readiness-rms-v1/run.log`. It waits for BOTH baseline seals, stops
the original writer before the old rescues, and seals that orchestration as
`superseded_after_baselines`. It then owns the remaining Phase 2 work. Both
versions share ONE five-hour active-compute ceiling; this is not a fresh
five-hour allowance. Waiting overlaps the original campaign and is not charged
twice. Neither campaign automatically starts Phase 3.

The following describes interruption recovery during execution; both runs
are now complete and must not be reopened for additional learning. A completed
`--resume` verifies and returns without importing Torch or starting new work.
After a reboot before both baselines finish, resume the original command above
AND this waiting supervisor. After the original orchestration is sealed,
resume only the rescue command. Never launch duplicate writers. Remove the
appropriate root STOP file only for intentional resumption. Both numerical
source sets are frozen now; use a new version for any material correction.
Only reporting-only changes can be made without changing their fingerprints.
Both implementation sidecars have finished and are closed; 62 focused rescue,
health and manager tests passed before the supervisor was launched.

## Previous Check-In (2026-11-11)

**Fix 3 is complete and stopped for review. GPU is free; no worker remains.**
`cache/numerical_experiment/picar/phase1/fix3-v2/orchestration/COMPLETED`
seals 23 child units. Both real student pilots ran completely. Each retained
2/2 easy and 2/2 middle successes; hard success remained 0/2 before and after.
This supplies a useful approach bootstrap, not demonstrated learning of the
new obstacle task. The common learner must acquire the changed skill before
an STM comparison is informative. Phases 2--5 and physical operation remain
pending. No further parameter or prompt sweep was started.

The notebook has the complete chronological Fix 3 section, compact success
plot, reward contact sheet, before/after clips and a concise conclusion.
Final reporting fixes prevent a null live timestamp on the old stopped
attempt from superseding the completed campaign, verify pairing using the
recorded seed/geometry, and keep legends off the data.

Closeout reran all 19 focused reporting tests and executed all nine notebook
code cells without importing Torch. The final two-student success figure was
visually checked. Completed resume verifies/skips without importing Torch;
the GPU has no remaining compute process. The mathematical overview retains
its original SHA-256.

Scientific Python sources remain frozen and match the campaign fingerprint.
`units/provenance-*/sources.tar.gz` preserves the runtime code. Keep unrelated
dirty Plan 18 files and `.gitignore` changes. The mathematical overview is
unchanged. A completed resume verifies/skips without importing Torch:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar.phase1_fix3 \
  --output cache/numerical_experiment/picar/phase1/fix3-v2 --resume
```

Supply the same optional `--model-path` and `--offline` values recorded in
`orchestration/config.json`. A root `STOP` file requests a checkpointed halt;
remove it before an intentional resume. Pending episodes retain their state
and never receive fabricated failure returns. The CLI ends after the two
bounded Replay/EWC student pilots and the task/resource check-in, not the
full STM comparison or physical operation.

- The stopped `fix3-v1` attempt and its exact source snapshot are retained:
  combined proximity/centering judgments saturated almost all visible scores
  at 10. It charged 264.7 seconds and trained no student.
- The permitted targeted repair separates the exact Fix 2 contact/proximity
  query from a horizontal-image-coordinate query. Camera pan discounts the
  half-point centering addition. Development approach ordering is 4/6 and
  centering ordering 4/6; validation approach ordering is 4/6, centering 2/6.
  The signal is useful but coarse, with off-axis detection errors. No further
  prompt search is authorized by this amendment.
- Software acceptance passed 705 tests. A preceding incomplete software unit
  retains `pytest-initial-report-failure.txt` for a reporting-string assertion;
  it was corrected without changing scientific settings.
- The geometry audit has 58.2% clear early layouts and 100% blocked late
  layouts. Privileged diagnostic pursuit reached contact in 12/12 checked
  level/gain cases; none of those controls enter demonstrations.
- Return-bin calibration selected 161 bins to preserve more of the small
  centering signal. Model context 4, hidden width 64, top-L 1, LTM/STM ranks
  4/1, replay capacity 8, new/replay batch 1/1, FD strength zero in this pilot.
- Prior repairs/recognition plus the stopped attempt were 37.0 minutes at
  v2 admission. V2 has its own two-hour ceiling inside the cumulative
  three-hour repair envelope and 20-hour overall study ledger.
- Actual CUDA recovery acceptance halted one-time distillation at step 95
  after 292 captured transitions (44 episodes, 247 executed teacher examples,
  16 successes). A fresh GPU process restored parameters, KL Adam tensors and
  named RNG states exactly; the check is in `fix3-v2/gpu-recovery-check.json`.
  The STOP flag was removed and the same immutable configuration resumed.
- Corrected campaign compute was 80.5 min; both Fix 3 attempts total 84.9 min.
  Cumulative recorded repairs/recognition were 117.5 min; a conservative
  further 5 min reserve covers development/verification work outside those
  invocations. Including earlier acceptance/profiling, the conservative
  overall study charge is about 2.22 h. Budget caps remain unchanged.
- The bootstrap took 33.2 min across invocations; do not use only its final
  invocation's 12.2 min as its cost. Pilots took 17.2 and 17.6 min, captured
  305/300 intervals, and completed 276 updates each (255 online, 21 stopped).
  Ordinary non-startup misses were 2/278 and 1/277. Final replay has 8 entries,
  with no pending updates and exactly-once retirement accounting.
- Review artifacts: `review/integrity.json`, `review/video-checks.json`, and
  `review/resource-forecast.json`. Audit verified 32 completed units across
  attempts, 1,898 frame blobs, 203 media files and two source archives with
  unchanged completion markers. All six clips decode and show motion.
- Conditional Replay/EWC processing throughput is 2.95 s/captured interval:
  four pairs x 1,200/arm would take about 7.9 h processing; six x 2,400 about
  23.6 h. These exclude additional FD cost, development, evaluation and
  closeout, and do not admit a study whose new-task learning gate is unmet.

## Reader-Requested Hard-Task Recording (2026-11-12)

The old `seed-8-B/scene.mp4` gallery clips are privileged scripted feasibility
runs, not learned-student runs. The latest Fix 1 version records success at
13.5 s. Original Fix 3 final hard evaluations instead all collided at
4.8--5.7 s; only the easy evaluation panel originally had video capture.

At the user's request, `numerical_experiment.picar_evaluation_video` recorded
a separate no-learning evaluation of student 193701's final checkpoint on
hard layout 195803. Output: `diagnostics/fix3-final-hard-video-v1` (completed).
The synchronized camera/top-down MP4 reproduced a collision at 7.23 s versus
5.30 s in the original trial. This is explicitly supplemental footage with
fresh real-time scheduling, not recovery of the original trial's footage.
All learner parameters, optimizer, Fisher, anchors, RNGs and counters were
checked unchanged. Six focused tests pass, the original campaign numerical
sources remain unchanged, and the GPU is free. The successful invocation took
22.8 s; a preceding sandbox-restricted CUDA load failed before any rollout.
Original results and completion seals are untouched; no main-study result
or treatment comparison was added. This separate CLI stays outside the
frozen campaign's source tree and accepts an optional model path as usual.

## Historical Fix 2 Check-In

**Fix 2 complete, stopped at the task/resource check-in (2026-11-09).**
`phase1/fix2-v2/orchestration/COMPLETED` seals the corrected campaign. All six
software deliverables are present; integrated acceptance passed **637 tests**.
The notebook is artifact-only and includes the full Fix 2 diagnostic section
and conclusion. No Phase 2--5, physical motion or scientific STM comparison
has started. No worker needs to remain running. Preserve the already-dirty
worktree; do not revert earlier implementations or artifacts.

- Contact-aligned generated scores are valid and separate absence from close
  approach, but strict far/intermediate/close ordering passed only 5/12
  development and 8/12 held-out pairs (80% guide unmet).
- Ported Qwen: 3/6 unobstructed successes. Longer grounded prompt: 0/6.
  Full obstacle task: 0/12 across both interfaces, all collisions. Teacher
  rollouts are accelerated diagnostics, not real-time-control admission.
  Fresh 20-layout validation was not admitted.
- Each real-time student/control pilot completed three episodes, 44 updates
  (28 online, 16 stopped drain/sleep), 36 retirements and zero non-startup
  fallbacks. Peak label backlog 17, final zero. Students were untrained;
  all episodes collided. These are execution tests, not learning results.
- Return/bin gradient checks pass with 41 bins. Retired-data sleep is
  implemented/default off and crash-tested. Pi hardware/speech interfaces
  have injected-input acceptance; real calibration/STT/actuation untested.
- Corrected campaign charged 17.9 minutes; both Fix 2 attempts 21.3 minutes;
  all Fix 1+2 repairs 31.1 minutes, below the three-hour repair cap.
- Updated conditional study forecasts: **21.4--30.8 h** nominal, or
  **11.6--17.2 h** at the smaller permitted scope. Notebook totals explicitly
  add 10.6 minutes of earlier acceptance/profiling omitted by the sealed
  forecast. Fit timings include cooperative waits; these are conditional
  serial forecasts, not readiness or compute approval.

Next decision: settle teacher/reward readiness and a study scope with the
user. Do not automatically start Phase 2. Completed `phase1_fix2 --resume`
must use the same configuration/model/source identity and only verifies/skips
the campaign. New numerical/role changes require a new campaign version.
Exact source snapshots are retained in both attempts under
`source-snapshots/verified-source-v1`; the completed corrected campaign also
has its final provenance source archive. Rebuild the notebook with the
artifact-only `picar.run report` command in README.

Closeout verified 60 completed units, three source archives, unchanged
completion markers and a completed-resume skip without importing Torch.
All eight notebook code cells execute artifact-only. Focused reporting tests
pass (24), all four preselected videos decode with motion, and the GPU has
no remaining compute process. Fix 1's section and the theory notebook retain
their original hashes. Report-only grouping/figure-label changes are outside
the frozen scientific source identity; all 45 scientific source files match.

**Recognition follow-up (2026-11-10):** at the user's request, the new
`visual_questions` entry point queried four saved initial images plus a
ball-absent control, six independent questions each. The separate immutable
run `diagnostics/visual-questions-v1` contains all 30 raw answers and a sealed
CSV (89.3 active seconds). Spontaneous red-sphere recognition passed on all
four positive images. Direct presence/location questions rejected the absent
control, while distance questions hallucinated its position. Apparent depth
relationships and metric distances remain unreliable. No main experiment
scope changed. Three focused unit tests pass; saved frames/CSV/seals verify,
and completed resume skips without importing Torch. The GPU is free.

**First Fix 2 attempt retained:** `phase1/fix2-v1` checkpointed a requested
stop after 201.8 active seconds, with 20 complete units. Software acceptance
passed 635 tests. Qwen often generated valid action/reward JSON inside a
single Markdown fence, which the initial strict parser rejected. Those
teacher rows are interface failures, not measured navigation failures.
The new parser accepts only bare JSON or one complete JSON fence, keeping
keys/numeric validation strict. The timing pilot now masks retired context
predecessors. Both corrections receive a new source/configuration version
`phase1-fix2-v2` under `phase1/fix2-v2`; never overwrite v1 artifacts.
All three Sol sidecars have finished and closed. New resource forecasting
uses measured generated-role, fit, Fisher and durable journal costs.

### Historical Planning And Prior Acceptance Notes

The following notes describe earlier states and are superseded by the
completed Fix 2 status above; they are retained for continuity.

**Planning update (2026-11-09):** the user requested a further amendment
as a separate **Phase 1 Fix 2** for review, with Sol to execute subsequently.
Read "Phase 1 Fix 2: Restore Dense Visual Rewards And The Robot Workflow"
in `plan19.md` before making changes. It is **planned, not implemented or
executed**. This turn changes documentation only; no experiment was launched.
The earlier unexecuted draft was nested under Fix 1; the user corrected that
numbering. Fix 1's specification, execution history and artifact names are
preserved. Append future material amendments rather than rewriting that work.

The draft restores generated 0--10 RGB reward scores, proposes explicit
`ball-contact-v1` success semantics and accumulated `dense-visual-v1` rewards,
and tests simple approach before the full obstacle task. It requires causal
conversation/coaching context, a public task countdown, return bounds/bin
resolution checks, and sustained queue/timing accounting. Historical safe-stop
and potential-shaping versions remain intact. The normalized joint likelihood,
STM FD and diagonal eviction-only EWC mathematics remain the same family;
new reward labels/scales and initialization need their own version.

All related corrections are explicit implementation deliverables. Item 5
implements default-off retired-data sleep with supplemental-rehearsal
accounting and exact resume tests; primary scientific conditions keep it off.
Item 6 implements the actual actuator/camera adapter and typed/speech event
bridge with injected hardware and recorded-input tests. Real calibration,
microphone use and physical actuation are not part of this acceptance run.

Expected additional active compute is 0.9--2.4 h, with 4--9 h of implementation
estimated separately. Keep Fix 2 charges distinct, counting them together
with Fix 1's 9.8 minutes against the existing three-hour repair envelope and
overall study budget. Proposed output family:
`phase1/fix2-v1`. There is no new runnable command yet; the existing
`phase1_fix --output .../fix1-v2 --resume` only verifies the old campaign.
After implementation/readiness, stop for the resource check-in; no automatic
Phase 2--5 or physical motion. The added software features do not authorize
additional scientific treatment arms or real-world deployment tests.

**Completed prior work:** Phase 1 Fix 1 is implemented and its two bounded campaigns are
complete: `phase1/fix1-v1` and `phase1/fix1-v2`. Native/split parity and the
revised stop/left/right instruction checks pass. Context-4 cooperative pilots
had zero fallbacks at .75/1/1.25 s; scripted feasibility passed 10/10 at .75/1 s
but only 8/10 at 1.25 s. These are two-update burst diagnostics, not sustained
online-load admission. Teacher navigation and the visual potential remain
unready. No fresh teacher validation or Phase 2--5 learning was launched.

Repair compute was 590.4 recorded seconds across both passes. A Tk backend
abort during v2 was recovered through `--resume` with Agg; completed units
were preserved/skipped and the incomplete short timing pilot retried. The
new notebook section and Plan 19 execution record describe both successes and
remaining blockers. The separate Fix 2 draft above addresses that scope decision;
its execution still awaits the user's next instruction.

New code: `role_repair.py` (versioned grounding, causal multi-image input,
FP32 answer readout, native reference/generation); `phase1_fix*.py` (bounded
resumable diagnostics and artifact-only reporting). `MemoryLearner.fit`
accepts a graph-safe `cooperate` callback; `ControlSession.update(cooperative=True)`
uses it and skips expired work before GPU inference. Parameters remain
unchanged at every yield; failed updates preserve actor exploration RNG
consumption. Multi-step events stop yielding before their first mutation.

`RunStore(..., checkpoint_policy="journal-v1")` is opt-in; default legacy
behavior stays unchanged. It retains append-only events and current/previous
full states plus explicit milestones. Journal resume must select the same
policy. Only superseded state payloads in incomplete runs are pruned after
durable CURRENT publication. Never prune completed data. The v2 provenance
unit includes a checksummed source archive and recovery notes.

Phase 0 acceptance passed, and the user subsequently authorized Phase 1 only,
followed by a resource check-in. Phase 1 diagnostics are finished on
2026-11-09, **but readiness gates failed**. The original Phase 1 suite passed
361 tests; the final Fix 1 regression suite passes **442 tests**. No Phase 2--5 learning or physical motor operations
were started. Do not start them automatically. All Phase 1 processes and
coding agents have exited; the GPU is free.

The measured whole-study ranges are **38.9--55.7 h** nominal (six pairs,
2,400 new transitions/arm), or **19.0--28.0 h** at the smaller permitted size
(four pairs, 1,200/arm). Neither is safely admitted under 20 hours. These
conditional engineering forecasts include evaluation/sleep/checkpoint I/O;
teacher rescue or more optimizer steps can increase them.

Report: [plan19_results.ipynb](plan19_results.ipynb), pre-rendered from artifacts.

`mathematical_overview.ipynb` must remain unchanged. Initial SHA-256:
`eae6e00ce327334cca56755714d1965f21e0ff1dc4e8d3d14487eaa2b7ec8487`.

The worktree was clean when this Phase 1 turn began. Current uncommitted
changes are the Phase 1 implementation, exact model optimizations, tests,
notebook and notes. Do not revert unrelated later user changes.

## Phase 1 Findings And Next Decision

- Main immutable campaign:
  `cache/numerical_experiment/picar/phase1/resource-v2`. Its `orchestration`
  directory is complete; separate `units/` retain nine CUDA probes, scene
  audit, role/control checks and two scope forecasts. Re-running this completed
  campaign verifies/skips it; changed diagnostics require new units/versions.
  Main campaign execution was 364 s, plus earlier probes and engineering/tests.
- Memory shortlist only: top-L=1, hidden 64, LTM/STM ranks 4/1, context 8,
  replay batch 1, one score draw and one FD probe, 160x120 input. Peak reserved
  7.11 GiB, headroom 3.85 GiB. Top-2, batch-4 and context-16 exhausted VRAM;
  top-4/batch-8 were skipped. This does not establish student competence.
- Cold inference .790 s, control fit 1.714 s, FD fit 1.980 s, full-context
  eviction Fisher .866 s, frozen roles ~.17 s each. One learner checkpoint
  is 8.22 MB; a durable transaction measured .056 s. Draw counts 2/4 and
  two FD probes were measured, not selected for efficacy.
- Exact opt-in frozen-prefix CPU memoization (two entries/256 MB bound)
  reuses only frozen computations within events; all trainable suffix and
  FD-endpoint gradients remain live. Probes clear caches before every event
  and use distinct new/replay inputs. Frozen role extraction now avoids
  computing discarded earlier RGB rows while preserving the final row's
  entire recorded text/action context. Equivalence and invalidation tests
  pass; initial uncached `resource-v1` remains retained.
- Privileged scripted controller passed 10/10 endpoint/mirrored layouts and
  camera/action mapping checks. Its synchronized clips were visually checked.
  It is never a source of unreported primary teacher demonstrations.
- Frozen Qwen: 0/2 A and 0/2 B teacher rollouts succeeded (all collided).
  Completion potential ranked collisions above stopped success. These small
  diagnostics identify an inadequate current role path, not a precise estimate
  of Qwen's general capability. Bootstrap must not claim retention of an
  unlearned skill. Next development needs bounded role/prompt/observation
  repairs before substantive learning, as allowed by the plan.
- Real-time worker reached 20 simulated seconds/4,800 physics ticks while
  two ~2.1 s updates ran. At a 1.25 s control period, 6/17 non-startup intervals
  used safe fallback (35.3%). Worker wall time was 22.68 s. This is not admission
  for continuous one-update-per-acquisition operation; improve/profile the
  scheduling envelope without pausing physics or fabricating policy labels.
- Storage: raw training records/frames project to 13.5 / 4.7 GB nominal/small,
  but every-full-snapshot retention projects to 1.55 TB / 393 GB for fresh
  training alone. A bounded checkpoint/event-journal policy and explicit
  archive root are needed before large work. Do not silently use the drive
  or delete anything under an existing COMPLETED marker. Current Phase 1
  artifacts occupy about 95 MB.
- No physical measurements supplied. No sim-to-real validation claimed.

Probes use model initialization seed 191101, simulated profile history seed
190101 and named learner streams from `resolved(recipe).seeds(0)`; source and
model-file checksums are in immutable unit provenance. Role rollouts use
seeds 4/5 in both endpoint domains; the live worker uses 191107.

The next conversation should settle task readiness repairs and the acceptable
compute envelope, not automatically launch a 39--56 h run. The smoke FD
coefficient remains uncalibrated for scientific use. Full production-stage
orchestration still belongs to later work.

## Delegation

| Slice | Requested model/effort | Agent | State |
|---|---|---|---|
| 0A/B and bounded numerical integration review | `gpt-6-astra`, `xhigh` | Kierkegaard | Complete; closed |
| 0C model/language wiring and 0F reporting | `gpt-6.1-sol`, `xhigh` | Hubble | Complete; closed |
| 0D/E robot, storage, independent worker and control | `gpt-6.1-sol`, `xhigh` | Goodall | Complete; closed |
| Configuration, learner, acceptance, CLI and handoff | Coordinator | Main thread | Accepted |

Worker token/credit usage is not exposed by the current tool response and is
therefore unknown. Do not infer usage from output length.

## Environment Notes

- Use the user's existing `~/.venv/bin/python`; do not upgrade the model stack.
- GPU: RTX 4070, 12 GB. PyTorch 2.10.0+cu128 sees CUDA outside the sandbox;
  sandbox-only `is_available=False` is not a driver failure.
- Qwen defaults to a pinned download retained in the project's ignored
  `cache/numerical_experiment/picar/models/`, independent of working directory.
  `--model-path` is an optional explicit reuse override; `--offline` forbids
  downloads. The default Hub route is unit-tested with a mocked download.
  Actual GPU acceptance used explicitly supplied existing weights and their
  verified file checksums, without downloading a duplicate copy.
- Installed PyBullet 3.2.7, Gymnasium 1.4.0, imageio 2.38.1 and
  imageio-ffmpeg 0.8.1. PyBullet's local source build used two compiler workers;
  physics code was unchanged. Torch/Transformers/bitsandbytes were not upgraded.
  `pip check` was clean after installation. Full versions are in run manifests.
- Phase 0 artifacts belong under `cache/numerical_experiment/picar/`. No
  private external-drive path should enter source or tracked configs.

## Next-Phase Checklist

1. Read `plan19.md` Phase 0 and `NUMERICAL_CONTRACT.md`.
2. Phase 1 authorization has been used; await the requested check-in before
   later learning. Its resource shortlist fits but task/deadline gates failed.
3. Use `preflight --cuda` after node changes. Context-8/batch-1 is profiled;
   context-8/batch-4 exhausted VRAM and is not an admitted recipe.
4. Establish physical mapping, scripted feasibility, teacher/evaluator skill,
   inference period, fallback rates, context and label eligibility before
   binding the real-time controller to scientific acquisition.
5. Calibrate FD strength by gradient ratios and trajectories, not the smoke
   default. Match retention in development before sealing fresh recipes.
6. Enforce cumulative campaign time/storage accounting in the stage scheduler,
   in addition to the implemented per-run quota and free-space admission.
   Pending/context bytes count; replay/sleep do not create new evidence.

## Acceptance Artifacts

Paths below are relative to `cache/numerical_experiment/picar/phase0/`:

- `acceptance-v3`: final immutable bundle. 280 tests passed;
  paired bootstrap/control/FD smoke and CUDA check passed. Total invocation
  was 52.52 s, including tests, loading and artifact work. Its summary is
  authoritative; child timing is subtracted from parent accounting to avoid
  double-counting in the report.
- `acceptance-v1` and `acceptance-v2`: earlier bundles retained. The final
  revision also executes notebook cells from the notebook's own directory.
- `qwen-acceptance-v1`: retained first diagnostic. Forward/update/Fisher worked,
  but reduced-precision top-block arithmetic did not support stable FD.
- `qwen-acceptance-v2`: corrected check. Across half/base/double epsilon,
  penalty max/min was 1.0472; gradient-norm ratio 1.0398; gradient cosines
  exceeded .9999. Both banks received gradients. Reserved memory was 6.715 GiB;
  inference .618 s, one update .517 s, eviction Fisher .179 s. The final bundle
  independently repeated the corrected check against its source snapshot.
- `visual-acceptance-v1`: synchronized fixed-seed ego/top-down video and PNGs,
  visually inspected. The top-down stream is never a learner input.

Qwen probes use one 64x48 image/history step, top-L=1, hidden size 16, and
processor limits 56x56--112x112. They do not admit the full production workload.
The CPU smoke uses accelerated physics and a fake teacher/zero potential.
These are software results, not robot competence or sample-efficiency results.

## Corrections Before Science

1. FP32 adapted Qwen suffix, rotary embeddings and final norm; frozen lower
   blocks remain BF16 and packed weights remain quantized. Small-output KL
   arithmetic uses FP64 to avoid cancellation near equality.
2. Each condition deep-copies bootstrap optimizer state. Fresh replica RNGs
   are distinct from bootstrap streams and paired between treatments; resumed
   streams retain their consumed state.
3. Zero-based record indices become observation counts for replay ages, giving
   the newest item age zero. Retired predecessors are masked in replay context.
4. Physics restore does not invent frame IDs. Completed artifacts also reject
   new child runs or frame stores beneath them.
5. Control mixtures accept float32 normalization roundoff but reject invalid
   distributions. No implicit steering-triggered forward motion remains.
6. Relative artifact paths resolve from the project, not the notebook's
   working directory. The resource inspector is named `preflight.py` to avoid
   shadowing Python's standard `profile` module during kernel startup.

Tests cover numerical identities, both-endpoint/both-bank gradients, frozen
roles, independent physics under blocked updates, delayed labels, exactly-once
retirement, and forced halts around action/label/optimizer/retirement commits.
No interrupted episode gets a fabricated zero return. Startup/missed-deadline
safe-neutral intervals are tagged, not fabricated Gaussian action labels.

## Entry Points And Scope

- Numerical contract: `NUMERICAL_CONTRACT.md`, `src/continual/picar_math.py`.
- Models and one-time recorded teacher KL: `models.py`, `teacher.py`.
- Likelihood/optimizer/FD/EWC integration: `learner.py`.
- Shared backend, independent clocked physics, physical client: `src/robotics/`.
- Maintained lightweight onboard service with fake-driver tests: `demo/picar/`.
- Real-time control/update ownership and bounded queues: `control.py`, `runtime.py`.
- Transactional artifacts/replay: `storage.py`, `schemas.py`.
- Acceptance trajectory, CLI and profiler: `engine.py`, `run.py`, `acceptance.py`,
  `preflight.py`, `qwen_acceptance.py`.
- Artifact-only statistics/notebook/video: `analysis.py`, `report.py`, `preview.py`.

The smoke runner deliberately rejects production configurations. Its short
deadlines, fake roles and accelerated physics must not become a research
recipe. `ControlSession` supplies the independent real-time acquisition and
update machinery for later stage runners; full scientific acquisition,
eligibility, calibration and study orchestration remain Phase 1 onward work.
Do not treat `configs/plan19.json` as a selected or resource-admitted recipe.

## Commands

Run from the repository root. Fresh-clone installation is in README; no other
checkout is required. Use new output names for changed code/configurations.

```bash
~/.venv/bin/python -m numerical_experiment.picar.run status \
  --output cache/numerical_experiment/picar/phase0

~/.venv/bin/python -m numerical_experiment.picar.run report \
  --output cache/numerical_experiment/picar/phase0

~/.venv/bin/python -m numerical_experiment.picar.preflight --cuda
```

The README acceptance command supports `--resume`. `run smoke` additionally
supports `--max-events` for recovery checks. `run stop --output <run-directory>`
requests a checkpoint-boundary stop; explicit smoke resume clears that request.
Reporting never unpickles checkpoints, trains, downloads, or repairs artifacts.
Fake-service tests bind localhost only and may need socket permission outside
a restrictive sandbox. No hardware adapter has been actuated or calibrated.
