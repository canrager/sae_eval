#!/bin/bash
# Define arrays for different configurations
ARCH=("vanilla" "vanilla" "vanilla" "topk" "topk" "topk")
LAYERS=(5 12 19 5 12 19)
WIDTH=12
NUM_TOKENS=200_000_000
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5")

# Start only the first two jobs
for i in {0..1}; do
    nohup python3 training_sweep.py \
        --save_dir saes \
        --architecture ${ARCH[$i]} \
        --layers ${LAYERS[$i]} \
        --width_exponents ${WIDTH} \
        --num_tokens ${NUM_TOKENS} \
        --device ${DEVICES[$i]} \
        > "${ARCH[$i]}_l${LAYERS[$i]}_w${WIDTH}_${DEVICES[$i]//:/_}.out" 2>&1 &
    
    # Capture the PID of the last background job (nohup)
    PID=$!
    echo "Started job ${i+1}/2: ${ARCH[$i]} with ${LAYERS[$i]} layers (PID: $PID)"
    
    # Optional: add a small delay between job submissions
    sleep 2

    # Immediately cancel the job
    kill $PID
    echo "Cancelled job ${i+1}/2 with PID: $PID"
done

echo "First two jobs started and then cancelled."
