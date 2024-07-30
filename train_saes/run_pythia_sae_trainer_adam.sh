#!/bin/bash

python3 pythia_sae_trainer_adam.py \
    --save_dir /workspace/sae_eval/dictionary_learning/dictionaries/pythia70m_sweep_standard_ctx128_0712 \
    --layers 0 1 2 \
    # --no_wandb_logging \
    # --dry_run \