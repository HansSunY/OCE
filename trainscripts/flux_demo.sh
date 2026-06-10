#!/bin/bash
MODEL_ID="black-forest-labs/FLUX.1-dev"
EDIT_CONCEPTS="Elon Musk"
GUIDE_CONCEPTS="man"
PRESERVE_CONCEPTS=""
DEVICE="cuda:0"
CONCEPT_TYPE="object"
EXP_NAME="Elon_Musk"
SAVE_DIR="./Elon_Musk_FLUX"

ERASE_SCALE=1000
PRESERVE_GLOBAL_SCALE=10
PRESERVE_CONCEPT_SCALE=5
LAMB=10

python oce_flux.py \
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
    #--expand_prompts "true" \
