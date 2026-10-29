# Plan 15 Phase 11

The user authorized Phase 11 implementation and execution on 2026-10-28.
Follow ../plan15.md, the root scientific contract, and the single 20-hour active
compute allowance with the shared 500 GB artifact ceiling. Resume preserves both.

- One observation is a pair of 84x84 mosaics containing nine independent affine
  digit pairs. Predict 54 cell-local landmark displacement coordinates.
- Gaussian covariance is 4 I. Sum coordinate likelihood/Fisher contributions,
  average grids, and divide squared prediction errors by 27 for reported risk.
- One branch/severity per grid; blur queries cellwise. Source pools and named
  seeds are paired, disjoint by role, and fixed before development selection.
- FIFO32 grids/replay2; EWC-only retires immediately. Retire each grid once with
  exact Gaussian-integrated model Fisher at the accepted parameter.
- Freeze architecture, compression backend/rank, FD step and policy before
  confirmation. Preserve failed artifacts and record material repairs.
- Historical packages and completed artifacts stay immutable. Reporting loads
  artifacts only, appends chronologically and preserves historical notebook cells.
- Profile the larger Fisher path within the bounded allowance before screening.
  Every approximate backend must preserve diagonal/trace and pass exact audits.
