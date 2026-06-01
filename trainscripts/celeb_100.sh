#!/bin/bash
MODEL_ID="CompVis/stable-diffusion-v1-4"
EDIT_CONCEPTS="Adam Driver; Adriana Lima; Amber Heard; Amy Adams; Andrew Garfield; Angelina Jolie; Anjelica Huston; Anna Faris; Anna Kendrick; Anne Hathaway; Arnold Schwarzenegger; Barack Obama; Beth Behrs; Bill Clinton; Bob Dylan; Bob Marley; Bradley Cooper; Bruce Willis; Bryan Cranston; Cameron Diaz; Channing Tatum; Charlie Sheen; Charlize Theron; Chris Evans; Chris Hemsworth; Chris Pine; Chuck Norris; Courteney Cox; Demi Lovato; Drake; Drew Barrymore; Dwayne Johnson; Ed Sheeran; Elon Musk; Elvis Presley; Emma Stone; Frida Kahlo; George Clooney; Glenn Close; Gwyneth Paltrow; Harrison Ford; Hillary Clinton; Hugh Jackman; Idris Elba; Jake Gyllenhaal; James Franco; Jared Leto; Jason Momoa; Jennifer Aniston; Jennifer Lawrence; Jennifer Lopez; Jeremy Renner; Jessica Biel; Jessica Chastain; John Oliver; John Wayne; Johnny Depp; Julianne Hough; Justin Timberlake; Kate Bosworth; Kate Winslet; Leonardo Dicaprio; Margot Robbie; Mariah Carey; Melania Trump; Meryl Streep; Mick Jagger; Mila Kunis; Milla Jovovich; Morgan Freeman; Nick Jonas; Nicolas Cage; Nicole Kidman; Octavia Spencer; Olivia Wilde; Oprah Winfrey; Paul Mccartney; Paul Walker; Peter Dinklage; Philip Seymour Hoffman; Reese Witherspoon; Richard Gere; Ricky Gervais; Rihanna; Robin Williams; Ronald Reagan; Ryan Gosling; Ryan Reynolds; Shia Labeouf; Shirley Temple; Spike Lee; Stan Lee; Theresa May; Tom Cruise; Tom Hanks; Tom Hardy; Tom Hiddleston; Whoopi Goldberg; Zac Efron; Zayn Malik"
GUIDE_CONCEPTS="tree"
PRESERVE_CONCEPTS="Aaron Paul; Alec Baldwin; Amanda Seyfried; Amy Poehler; Amy Schumer; Amy Winehouse; Andy Samberg; Aretha Franklin; Avril Lavigne; Aziz Ansari; Barry Manilow; Ben Affleck; Ben Stiller; Benicio Del Toro; Bette Midler; Betty White; Bill Murray; Bill Nye; Britney Spears; Brittany Snow; Bruce Lee; Burt Reynolds; Charles Manson; Christie Brinkley; Christina Hendricks; Clint Eastwood; Countess Vaughn; Dakota Johnson; Dane Dehaan; David Bowie; David Tennant; Denise Richards; Doris Day; Dr Dre; Elizabeth Taylor; Emma Roberts; Fred Rogers; Gal Gadot; George Bush; George Takei; Gillian Anderson; Gordon Ramsey; Halle Berry; Harry Dean Stanton; Harry Styles; Hayley Atwell; Heath Ledger; Henry Cavill; Jackie Chan; Jada Pinkett Smith; James Garner; Jason Statham; Jeff Bridges; Jennifer Connelly; Jensen Ackles; Jim Morrison; Jimmy Carter; Joan Rivers; John Lennon; Johnny Cash; Jon Hamm; Judy Garland; Julianne Moore; Justin Bieber; Kaley Cuoco; Kate Upton; Keanu Reeves; Kim Jong Un; Kirsten Dunst; Kristen Stewart; Krysten Ritter; Lana Del Rey; Leslie Jones; Lily Collins; Lindsay Lohan; Liv Tyler; Lizzy Caplan; Maggie Gyllenhaal; Matt Damon; Matt Smith; Matthew Mcconaughey; Maya Angelou; Megan Fox; Mel Gibson; Melanie Griffith; Michael Cera; Michael Ealy; Natalie Portman; Neil Degrasse Tyson; Niall Horan; Patrick Stewart; Paul Rudd; Paul Wesley; Pierce Brosnan; Prince; Queen Elizabeth; Rachel Dratch; Rachel Mcadams; Reba Mcentire; Robert De Niro"
DEVICE="cuda:0"
CONCEPT_TYPE="object"
EXP_NAME="celeb_100"
SAVE_DIR="./celeb"

#100
ERASE_SCALE=800
PRESERVE_GLOBAL_SCALE=70
PRESERVE_CONCEPT_SCALE=2
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
