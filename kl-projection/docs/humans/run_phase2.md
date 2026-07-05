# Run Phase 2

Phase 2 has two workflows:

1. Fit and replay the KL-projection LSTM: `train_phase2.md`.
2. Run the fitted hierarchy on the robot: `run_phase2_robot.md`.

The fitting path writes ignored model artifacts under `artifacts/models/phase2/` and compact summaries under `experiments/runs/phase2/`.
The live robot path writes ignored observations under `artifacts/data/phase2-live/`.
