#!/bin/bash

python3 training_sweep.py \
    --save_dir /share/u/can/sae_eval/sae_training_can/saes \
    --architecture vanilla \
    --layers 5 \
    --width_exponents 12 \
    --num_tokens 400_000_000 \
    --device cuda:0 \
    # --no_wandb_logging \
    # --dry_run \

# Run in background with
# nohup ./run_training_sweep.sh > X.out 2>&1 &