#!/bin/bash
MODEL_ID="CompVis/stable-diffusion-v1-4"
EDIT_CONCEPTS="Adam Driver; Adriana Lima; Amber Heard; Amy Adams; Andrew Garfield; Angelina Jolie; Anjelica Huston; Anna Faris; Anna Kendrick; Anne Hathaway"
GUIDE_CONCEPTS="person; woman; man"
PRESERVE_CONCEPTS="Aaron Paul; Alec Baldwin; Amanda Seyfried; Amy Poehler; Amy Schumer; Amy Winehouse; Andy Samberg; Aretha Franklin; Avril Lavigne; Aziz Ansari; Barry Manilow; Ben Affleck; Ben Stiller; Benicio Del Toro; Bette Midler; Betty White; Bill Murray; Bill Nye; Britney Spears; Brittany Snow; Bruce Lee; Burt Reynolds; Charles Manson; Christie Brinkley; Christina Hendricks; Clint Eastwood; Countess Vaughn; Dakota Johnson; Dane Dehaan; David Bowie; David Tennant; Denise Richards; Doris Day; Dr Dre; Elizabeth Taylor; Emma Roberts; Fred Rogers; Gal Gadot; George Bush; George Takei; Gillian Anderson; Gordon Ramsey; Halle Berry; Harry Dean Stanton; Harry Styles; Hayley Atwell; Heath Ledger; Henry Cavill; Jackie Chan; Jada Pinkett Smith; James Garner; Jason Statham; Jeff Bridges; Jennifer Connelly; Jensen Ackles; Jim Morrison; Jimmy Carter; Joan Rivers; John Lennon; Johnny Cash; Jon Hamm; Judy Garland; Julianne Moore; Justin Bieber; Kaley Cuoco; Kate Upton; Keanu Reeves; Kim Jong Un; Kirsten Dunst; Kristen Stewart; Krysten Ritter; Lana Del Rey; Leslie Jones; Lily Collins; Lindsay Lohan; Liv Tyler; Lizzy Caplan; Maggie Gyllenhaal; Matt Damon; Matt Smith; Matthew Mcconaughey; Maya Angelou; Megan Fox; Mel Gibson; Melanie Griffith; Michael Cera; Michael Ealy; Natalie Portman; Neil Degrasse Tyson; Niall Horan; Patrick Stewart; Paul Rudd; Paul Wesley; Pierce Brosnan; Prince; Queen Elizabeth; Rachel Dratch; Rachel Mcadams; Reba Mcentire; Robert De Niro"
DEVICE="cuda:0"
CONCEPT_TYPE="object"
EXP_NAME="celeb_10"
SAVE_DIR="./celeb"

#10
ERASE_SCALE=3500
PRESERVE_GLOBAL_SCALE=50
PRESERVE_CONCEPT_SCALE=5
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
    #--expand_prompts "true" \
