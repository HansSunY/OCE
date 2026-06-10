#!/bin/bash
MODEL_ID="black-forest-labs/FLUX.1-dev/"
OCE_MODEL_PATH=""
PROMPT=""
SAVE_PATH=""
EXP_NAME=""

NUM_IMAGES_PER_PROMPT=5
NUM_INFERENCE_STEPS=25
DEVICE="cuda:0"

python generate_flux.py \
    --model_id "$MODEL_ID" \
    --oce_model_path "$UCE_MODEL_PATH" \
    --prompt "$PROMPT" \
    --save_path "$SAVE_PATH" \
    --exp_name "$EXP_NAME" \
    --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --device "$DEVICE"
