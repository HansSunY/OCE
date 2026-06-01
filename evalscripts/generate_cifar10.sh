#!/bin/bash
CONCEPT="airplane"
MODEL_ID="CompVis/stable-diffusion-v1-4"
OCE_MODEL_PATH="$CONCEPT/$CONCEPT.safetensors"
LIST=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")
PROMPT="a photo of $CONCEPT"
SAVE_PATH="eval_cifar_$CONCEPT"
NUM_IMAGES_PER_PROMPT=200
NUM_INFERENCE_STEPS=50
DEVICE="cuda:0"

for cp in "${LIST[@]}"; do
    python generate_object.py \
        --model_id "$MODEL_ID" \
        --oce_model_path "$OCE_MODEL_PATH" \
        --prompt "a photo of $cp" \
        --save_path "$SAVE_PATH" \
        --exp_name "$cp" \
        --num_images_per_prompt $NUM_IMAGES_PER_PROMPT \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --device "$DEVICE"
done