#!/bin/bash

python3 pythia_sae_trainer_can.py \
    --save_dir /share/u/can/shift_eval/train_saes/trained_saes/pythia70m_sweep0711 \
    --layers 3 4 \
    # --no_wandb_logging \
    # --dry_run \