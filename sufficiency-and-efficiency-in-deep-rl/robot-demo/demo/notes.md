
## phase 1 sampling command 

```
PICAR_V_HOST=10.0.0.223:5000 python -m src.experiments.experiment_interface \
  --phase init \
  --fixed-t 0.0 \
  --log-level INFO \
  --data-root data/phase1 \
  --model-root model/phase1
```

## phase 1 finalizing: memorization  

```
python -m src.experiments.phase1_finalize \
  --data-runs data/phase1/04f639c9-478a-4cb2-ab4d-086813915793 data/phase1/f0725eab-fe64-479e-8f88-995d38c24dba \
  --epochs 0 \
  --batch-size 3 \
  --learning-rate 0.00001 \
  --scale-ssr 1000. \
  --prompt-token-window 512 \
  --run-uuid phase1-memorized \
  --log-level INFO
```

## phase 1 finalizing : tuning 

```
python -m src.experiments.phase1_finalize \
  --data-runs data/phase1/04f639c9-478a-4cb2-ab4d-086813915793 data/phase1/f0725eab-fe64-479e-8f88-995d38c24dba \
  --load-snapshot model/phase1-memorized \
  --skip-memorization \
  --epochs 5 \
  --batch-size 3 \
  --learning-rate 0.000001 \
  --grad-clip 1.0 \
  --scale-ssr 1000. \
  --prompt-token-window 512 \
  --model-root model \
  --run-uuid phase1-tuned \
  --log-level INFO
```

## phase 2 

First call: 
```
PICAR_V_HOST=10.0.0.223:5000 ~/.venv/bin/python -m src.experiments.experiment_interface \
  --phase tune \
  --update-mode online \
  --load-snapshot model/phase1-tuned \
  --model-root model/phase2 \
  --t-step 0.001 \
  --train-every-steps 1 \
  --memorize-every-steps 1 \
  --min-replay-size 1 \
  --epochs 1 \
  --pi 0.001 \
  --log-level DEBUG
```

Follow-up phase 2 calls: 
```
PICAR_V_HOST=10.0.0.223:5000 python -m src.experiments.experiment_interface \
  --phase tune \
  --model-root model/phase2 \
  --load-latest-from-model-root \
  --init-t [CHOOSE T] \
  --t-step 0.001 \
  --t-log-every 1 \
  --epochs 10 \
  --train-every-steps 100 \
  --memorize-every-steps 100 \
  --memorize-n 100 \
  --min-replay-size 100 \
  --log-level INFO
```


