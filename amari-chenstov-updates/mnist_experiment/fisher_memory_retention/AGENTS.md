# Plan 15 Phase 8.1

Read Phase 8.1 in `../plan15.md` and the handoff before execution. The user
authorized implementation and execution on 2026-10-25, up to 20 active hours.

- Reuse repaired Phase 8 learning code unchanged. This package changes the
  search, selection and reporting, not the likelihood or archive estimand.
- Delta is 1.3342351463 degrees; epsilon is 0.20 degrees. Mean forgetting is
  mean(sqrt(replica base A) - sqrt(replica initial base MSE)). A common ceiling
  is not matching, and failure to reject a difference is not equivalence.
- All efficacy candidates run through 100 novel observations. Use four common
  development replicas, two more for finalists, then 12 fresh confirmation
  replicas unless a prospective timing-only admission decision requires 8/10.
- Give both replay methods C=256 and b in {4,16}; EWC-only is C=b=0. Tune ridge
  independently, cross FD strength with ridge, and preserve all tested points.
- Never tune on confirmation, exclude replicas for forgetting, or alter delta
  after results. Infeasibility is a result, not permission to hide a control.
- Preserve original Phase 8 units, source identity and frozen notebook. Publish
  only Phase 8.1-owned cells in the main notebook, preserving earlier cells.
- Use immutable decisions and accepted-update checkpoints; support --resume,
  STOP/SIGTERM and unattended operation. The shared storage cap remains 500 GB.
