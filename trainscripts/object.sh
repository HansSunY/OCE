#!/bin/bash
MODEL_ID="CompVis/stable-diffusion-v1-4"
EDIT_CONCEPTS="airplane"
GUIDE_CONCEPTS="sky"
PRESERVE_CONCEPTS="sky"
DEVICE="cuda:0"
CONCEPT_TYPE="object"
EXP_NAME="$EDIT_CONCEPTS"
SAVE_DIR="./$EDIT_CONCEPTS"

ERASE_SCALE=2000
PRESERVE_GLOBAL_SCALE=10
PRESERVE_CONCEPT_SCALE=0
LAMB=10

python oce.py \
    --model_id "$MODEL_ID" \
    --edit_concepts "$EDIT_CONCEPTS" \
    --guide_concepts "$GUIDE_CONCEPTS" \
    --preserve_concepts "$PRESERVE_CONCEPTS" \
    --device "$DEVICE" \
    --concept_type "$CONCEPT_TYPE" \
    --exp_name "$EXP_NAME" \
    --save_dir "$SAVE_DIR" \
    --erase_scale $ERASE_SCALE \
    --preserve_global_scale $PRESERVE_GLOBAL_SCALE \
    --preserve_concept_scale $PRESERVE_CONCEPT_SCALE \
    --lamb $LAMB \
    --expand_prompts "true"