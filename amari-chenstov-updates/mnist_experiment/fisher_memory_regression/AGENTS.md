# Plan 15 Phase 8

Read Phase 8 in `../plan15.md` and `../PLAN15_HANDOFF.md`. This package is a
separate scientific protocol, not a revision of categorical Plan 15 artifacts.

- The model has 287 LTM and 135 STM coordinates. Its scalar output is the mean
  angle divided by 45 degrees; the Gaussian standard deviation is fixed at 45
  degrees. Per-input model Fisher is the mean-output Jacobian outer product,
  not the observed residual-weighted gradient outer product.
- Pretraining and background arrivals use Uniform[-10,10] angles. Novel angles
  use a fair sign and 40*exp(.25*Z), rejecting magnitudes above 90. Novel count
  means generating-component membership, not a threshold on angle.
- Replay subsamples and FD probes stay fixed within each solve. Subsample means
  retain the 0.5 replay weight. All replay/FD gradients reach both memory blocks.
- Archive only actual FIFO retirements once, at accepted parameters. Preserve
  the nominal-four-observation archive clock despite variable segment sizes.
  The C=b=0 boundary retires immediately after its arrival fit.
- Protocol v2 uses exact symmetric eigh on the existing small dense LTM block,
  retaining eight modes plus the residual diagonal. Compression must preserve
  the source diagonal and trace. Do not restore the legacy Lanczos compressor:
  near rank-one inputs produced duplicated directions and explosive archives.
  The likelihood, eviction EMA, ridge formula and intended Fisher are unchanged.
- All comparisons have exactly aligned novel-budget checkpoints. Held-out
  panels never train or supply Fisher/curvature inputs. Never pool development
  and confirmation or reselect after observing confirmation.
- One persistent Phase 8 compute ledger includes acceptance and repairs; the
  Plan 15 shared storage cap is 500 GB. Completed artifacts are immutable.
- Phase 8A authorizes unit tests and tiny cached-MNIST recovery smokes only.
  Production 8B--8E requires the user's subsequent execution instruction.
- Keep the notebook chronological and concise, with progress at the top. It
  reads artifacts only. Sol may fix defects within the plan's autonomy rules,
  versioning mathematical changes and preserving prior evidence.
