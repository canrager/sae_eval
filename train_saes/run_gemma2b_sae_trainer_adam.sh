#!/bin/bash

python3 gemma2b_sae_trainer_adam.py \
    --save_dir /workspace/sae_eval/dictionary_learning/dictionaries/gemma2b_sweep_topk_ctx128_0816 \
    --layers 16 18 20 \
    # --no_wandb_logging \
    # --dry_run \