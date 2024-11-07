#!/bin/bash

# Define arrays for different configurations
ARCH="vanilla topk"
LAYERS=(5 12 19)
WIDTH=12
NUM_TOKENS=200_000_000
DEVICES=("cuda:0", "cuda:1", "cuda:2")

# Loop through the configurations
for i in {0..2}; do
    nohup python3 training_sweep.py \
        --save_dir saes \
        --architecture ${ARCH[$i]} \
        --layers ${LAYERS[$i]} \
        --width_exponents ${WIDTH} \
        --num_tokens ${NUM_TOKENS}\
        --device ${DEVICES[$i]} \
        > "vanilla_topk_l${LAYERS[$i]}_w${WIDTH}_${DEVICE//:/_}.out" 2>&1 &

        # --no_wandb_logging \
        # --dry_run \

    echo "Started job ${i+1}/6: ${ARCH[$i]} with ${LAYERS[$i]} layers"
    
    # Optional: add a small delay between job submissions
    sleep 2
done

echo "All jobs submitted!"




## Earlier version

# python3 training_sweep.py \
#     --save_dir /share/u/can/sae_eval/sae_training_can/saes \
#     --architecture vanilla \
#     --layers 5 \
#     --width_exponents 12 \
#     --num_tokens 400_000_000 \
#     --device cuda:0 \
    # --no_wandb_logging \
    # --dry_run \

# Run in background with
# nohup ./run_training_sweep.sh > X.out 2>&1 &