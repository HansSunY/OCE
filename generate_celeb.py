from diffusers import DiffusionPipeline
import torch
from PIL import Image
import pandas as pd
import argparse
import os
torch.set_grad_enabled(False)
from safetensors.torch import load_file

painting_templates = ["a portrait of {}", "a sketch of {}", "an oil painting of {}", "{} in an official photo", "an image capturing {} at a public event"]

PRESERVE_LIST = [
    "Aaron Paul", "Alec Baldwin", "Amanda Seyfried", "Amy Poehler", "Amy Schumer",
    "Amy Winehouse", "Andy Samberg", "Aretha Franklin", "Avril Lavigne", "Aziz Ansari",
    "Barry Manilow", "Ben Affleck", "Ben Stiller", "Benicio Del Toro", "Bette Midler",
    "Betty White", "Bill Murray", "Bill Nye", "Britney Spears", "Brittany Snow",
    "Bruce Lee", "Burt Reynolds", "Charles Manson", "Christie Brinkley",
    "Christina Hendricks", "Clint Eastwood", "Countess Vaughn", "Dakota Johnson",
    "Dane Dehaan", "David Bowie", "David Tennant", "Denise Richards", "Doris Day",
    "Dr Dre", "Elizabeth Taylor", "Emma Roberts", "Fred Rogers", "Gal Gadot",
    "George Bush", "George Takei", "Gillian Anderson", "Gordon Ramsey", "Halle Berry",
    "Harry Dean Stanton", "Harry Styles", "Hayley Atwell", "Heath Ledger",
    "Henry Cavill", "Jackie Chan", "Jada Pinkett Smith", "James Garner",
    "Jason Statham", "Jeff Bridges", "Jennifer Connelly", "Jensen Ackles",
    "Jim Morrison", "Jimmy Carter", "Joan Rivers", "John Lennon", "Johnny Cash",
    "Jon Hamm", "Judy Garland", "Julianne Moore", "Justin Bieber", "Kaley Cuoco",
    "Kate Upton", "Keanu Reeves", "Kim Jong Un", "Kirsten Dunst", "Kristen Stewart",
    "Krysten Ritter", "Lana Del Rey", "Leslie Jones", "Lily Collins",
    "Lindsay Lohan", "Liv Tyler", "Lizzy Caplan", "Maggie Gyllenhaal",
    "Matt Damon", "Matt Smith", "Matthew Mcconaughey", "Maya Angelou", "Megan Fox",
    "Mel Gibson", "Melanie Griffith", "Michael Cera", "Michael Ealy",
    "Natalie Portman", "Neil Degrasse Tyson", "Niall Horan", "Patrick Stewart",
    "Paul Rudd", "Paul Wesley", "Pierce Brosnan", "Prince", "Queen Elizabeth",
    "Rachel Dratch", "Rachel Mcadams", "Reba Mcentire", "Robert De Niro"
]
E1_LIST = ["Angelina Jolie"]
E10_LIST = [
    "Adam Driver", "Adriana Lima", "Amber Heard", "Amy Adams", "Andrew Garfield",
    "Angelina Jolie", "Anjelica Huston", "Anna Faris", "Anna Kendrick", "Anne Hathaway"
]
E50_LIST = [
    "Adam Driver", "Adriana Lima", "Amber Heard", "Amy Adams", "Andrew Garfield",
    "Angelina Jolie", "Anjelica Huston", "Anna Faris", "Anna Kendrick", "Anne Hathaway",
    "Arnold Schwarzenegger", "Barack Obama", "Beth Behrs", "Bill Clinton", "Bob Dylan",
    "Bob Marley", "Bradley Cooper", "Bruce Willis", "Bryan Cranston", "Cameron Diaz",
    "Channing Tatum", "Charlie Sheen", "Charlize Theron", "Chris Evans",
    "Chris Hemsworth", "Chris Pine", "Chuck Norris", "Courteney Cox", "Demi Lovato",
    "Drake", "Drew Barrymore", "Dwayne Johnson", "Ed Sheeran", "Elon Musk",
    "Elvis Presley", "Emma Stone", "Frida Kahlo", "George Clooney", "Glenn Close",
    "Gwyneth Paltrow", "Harrison Ford", "Hillary Clinton", "Hugh Jackman",
    "Idris Elba", "Jake Gyllenhaal", "James Franco", "Jared Leto", "Jason Momoa",
    "Jennifer Aniston", "Jennifer Lawrence"
]
E100_LIST = [
    "Adam Driver", "Adriana Lima", "Amber Heard", "Amy Adams", "Andrew Garfield",
    "Angelina Jolie", "Anjelica Huston", "Anna Faris", "Anna Kendrick", "Anne Hathaway",
    "Arnold Schwarzenegger", "Barack Obama", "Beth Behrs", "Bill Clinton", "Bob Dylan",
    "Bob Marley", "Bradley Cooper", "Bruce Willis", "Bryan Cranston", "Cameron Diaz",
    "Channing Tatum", "Charlie Sheen", "Charlize Theron", "Chris Evans",
    "Chris Hemsworth", "Chris Pine", "Chuck Norris", "Courteney Cox", "Demi Lovato",
    "Drake", "Drew Barrymore", "Dwayne Johnson", "Ed Sheeran", "Elon Musk",
    "Elvis Presley", "Emma Stone", "Frida Kahlo", "George Clooney", "Glenn Close",
    "Gwyneth Paltrow", "Harrison Ford", "Hillary Clinton", "Hugh Jackman",
    "Idris Elba", "Jake Gyllenhaal", "James Franco", "Jared Leto", "Jason Momoa",
    "Jennifer Aniston", "Jennifer Lawrence", "Jennifer Lopez", "Jeremy Renner",
    "Jessica Biel", "Jessica Chastain", "John Oliver", "John Wayne", "Johnny Depp",
    "Julianne Hough", "Justin Timberlake", "Kate Bosworth", "Kate Winslet",
    "Leonardo Dicaprio", "Margot Robbie", "Mariah Carey", "Melania Trump",
    "Meryl Streep", "Mick Jagger", "Mila Kunis", "Milla Jovovich", "Morgan Freeman",
    "Nick Jonas", "Nicolas Cage", "Nicole Kidman", "Octavia Spencer", "Olivia Wilde",
    "Oprah Winfrey", "Paul Mccartney", "Paul Walker", "Peter Dinklage",
    "Philip Seymour Hoffman", "Reese Witherspoon", "Richard Gere",
    "Ricky Gervais", "Rihanna", "Robin Williams", "Ronald Reagan",
    "Ryan Gosling", "Ryan Reynolds", "Shia Labeouf", "Shirley Temple",
    "Spike Lee", "Stan Lee", "Theresa May", "Tom Cruise", "Tom Hanks",
    "Tom Hardy", "Tom Hiddleston", "Whoopi Goldberg", "Zac Efron", "Zayn Malik"
]

def generate_images(pipe, oce_model_path, prompt, save_path, exp_name='test', device='cuda:0', torch_dtype=torch.bfloat16, guidance_scale = 7.5, num_inference_steps=100, num_images_per_prompt=10, from_case=0, till_case=1000000):
    generator = torch.manual_seed(42)
    # 1. Load the pipe

    if oce_model_path is not None:
        try:
            oce_weights = load_file(oce_model_path)
            pipe.unet.load_state_dict(oce_weights, strict=False)
        except:
            pipe.unet.load_state_dict(torch.load(oce_model_path, map_location='cpu'), strict=False)
    

    folder_path = f'{save_path}/{exp_name}'
    os.makedirs(folder_path, exist_ok=True)

        
    pil_images = pipe(prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=num_images_per_prompt,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        ).images
                          
                          
    for num, im in enumerate(pil_images):
        im.save(f"{folder_path}/{prompt}_{num}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_id', help='hf repo id for the model you want to test', type=str, required=False, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--oce_model_path', help='path for oce model', type=str, required=False, default=None)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='./oce_results/')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--exp_name', help='foldername to save the results', type=str, required=False, default='test_images')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_images_per_prompt', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--num_inference_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    args = parser.parse_args()
    
    model_id = args.model_id
    oce_model_path = args.oce_model_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    exp_name = args.exp_name
    num_images_per_prompt= args.num_images_per_prompt
    from_case = args.from_case
    till_case = args.till_case
    num_inference_steps = args.num_inference_steps
    pipe = DiffusionPipeline.from_pretrained(model_id, 
                                         torch_dtype=torch.bfloat16, 
                                         safety_checker=None).to(device)
    if exp_name=="P":
        concept_list = PRESERVE_LIST
    elif exp_name=="10":
        concept_list = E10_LIST
    elif exp_name=="50":
        concept_list = E50_LIST
    elif exp_name=="100":
        concept_list = E100_LIST
    elif exp_name=="1":
        concept_list = E1_LIST
    for concept in concept_list:
        for template in painting_templates:
            full_prompt = template.format(concept)
            generate_images(pipe=pipe, oce_model_path=oce_model_path, prompt=full_prompt, save_path=save_path, exp_name=concept, device=device, torch_dtype=torch.bfloat16, guidance_scale = guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt, from_case=from_case, till_case=till_case)
