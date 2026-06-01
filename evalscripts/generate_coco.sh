#!/bin/bash
CONCEPT=""
MODEL_ID="CompVis/stable-diffusion-v1-4"
OCE_MODEL_PATH=""
PROMPTS_PATH="data/coco_30k_val.csv"
SAVE_PATH="coco_eval"
EXP_NAME=""
NUM_IMAGES_PER_PROMPT=1
NUM_INFERENCE_STEPS=50
DEVICE="cuda:0"

python generate_coco.py \
    --model_id "$MODEL_ID" \
    --oce_model_path "$OCE_MODEL_PATH" \
    --prompts_path "$PROMPTS_PATH" \
    --save_path "$SAVE_PATH" \
    --exp_name "$EXP_NAME" \
    --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --device "$DEVICE"
