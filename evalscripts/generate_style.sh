#!/bin/bash
CONCEPT="Van Gogh"
MODEL_ID="CompVis/stable-diffusion-v1-4"
OCE_MODEL_PATH="$CONCEPT/$CONCEPT.safetensors"
LIST=("Van Gogh" "Picasso" "Monet")
SAVE_PATH="eval_final_$CONCEPT"
NUM_IMAGES_PER_PROMPT=10
NUM_INFERENCE_STEPS=50
DEVICE="cuda:0"

for cp in "${LIST[@]}"; do
    python generate_style.py \
        --model_id "$MODEL_ID" \
        --oce_model_path "$OCE_MODEL_PATH" \
        --prompt "$cp" \
        --save_path "$SAVE_PATH" \
        --exp_name "$cp" \
        --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --device "$DEVICE"
done