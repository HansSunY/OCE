#!/bin/bash
CONCEPT="nudity"
MODEL_ID="CompVis/stable-diffusion-v1-4"
UCE_MODEL_PATH="$CONCEPT/$CONCEPT.safetensors"
#PROMPTS_PATH="data/i2p.csv"
PROMPTS_PATH="data/nudity.csv"
SAVE_PATH="eval_nudity"
EXP_NAME="nudity"
NUM_IMAGES_PER_PROMPT=1
NUM_INFERENCE_STEPS=50
DEVICE="cuda:0"

python generate_nsfw.py \
    --model_id "$MODEL_ID" \
    --prompts_path "$PROMPTS_PATH" \
    --uce_model_path "$UCE_MODEL_PATH" \
    --save_path "$SAVE_PATH" \
    --exp_name "$EXP_NAME" \
    --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --device "$DEVICE"
