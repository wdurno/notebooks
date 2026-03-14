
## phase 1 sampling command 

```
PICAR_V_HOST=10.0.0.223:5000 python -m src.experiments.experiment_interface \
  --phase init \
  --fixed-t 0.0 \
  --log-level INFO \
  --data-root data/phase1 \
  --model-root model/phase1
```

## phase 1 finalizing 

```
python -m src.experiments.phase1_finalize \
  --data-runs data/phase1/04f639c9-478a-4cb2-ab4d-086813915793 data/phase1/f0725eab-fe64-479e-8f88-995d38c24dba \
  --epochs 3 \
  --batch-size 3 \
  --prompt-token-window 512 \
  --run-uuid phase1-finalized \
  --log-level INFO
```

