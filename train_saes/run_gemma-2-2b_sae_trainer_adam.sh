#!/bin/bash

python3 gemma2b_sae_trainer_adam.py \
    --save_dir /workspace/sae_eval/dictionary_learning/dictionaries/gemma-2-2b_sweep_topk_ctx128_0817 \
    --layers 12 16 20 \
    # --no_wandb_logging \
    # --dry_run \