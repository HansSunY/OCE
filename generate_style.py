from diffusers import DiffusionPipeline
import torch
from PIL import Image
import pandas as pd
import argparse
import os
torch.set_grad_enabled(False)
from safetensors.torch import load_file

painting_templates = [
    "{} style painting of the night sky with bold strokes.",
    "{} style landscape of rolling hills with dramatic brushwork.",
    "Self-portrait in {} style, showcasing distinctive color and texture.",
    "{} style rural scene with dynamic movement and intensity.",
    "Wheat field under a swirling sky in the style of {}.",
    "{} style transformation of simple cottages into profound scenes.",
    "{} style portrait conveying inner turmoil and vitality.",
    "Seascape in {} style capturing tumultuous waves and ocean movement.",
    "{} style depiction of light on water with delicate brushstrokes.",
    "Garden scene in {} style, emphasizing sunlight and shadow effects.",
    "{} style atmospheric haze enveloping the landscape.",
    "Haystacks in {} style, capturing light and color changes.",
    "Cathedral facade in {} style with shifting light and shadow.",
    "{} style vibrant flower garden in full bloom.",
    "{} style painting of boats as a study of color and atmosphere.",
    "{} style cliffs along the coastline with blended sea and sky.",
    "{} style bustling Paris streets with movement and vitality.",
    "Sunrise scene in {} style, capturing the beauty of dawn.",
    "{} style exploration of fog and mist in the landscape.",
    "Olive trees painted in {} style.",
    "Tree painted in {} style.",
    "Women working in the garden, in {} style.",
    "Wheat field painting by {}.",
    "Trees in bloom painted in {} style.",
    "{} style painting of a tree.",
    "Wheat field in {} style.",
    "{} style painting of a field with mountains in the background.",
    "Trees in bloom in the style of {}.",
    "Rocks in the ocean, in {} style.",
    "City painting in {} style.",
]

#painting_templates = ["a painting in the style of {}"]

def generate_images(pipe, oce_model_path, prompt, save_path, exp_name='test', device='cuda:0', torch_dtype=torch.bfloat16, guidance_scale = 7.5, num_inference_steps=100, num_images_per_prompt=10, from_case=0, till_case=1000000):
    generator = torch.manual_seed(0)
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
    parser.add_argument('--prompt', help='prompt', type=str, required=True)
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
    prompt = args.prompt
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
    for template in painting_templates:
        full_prompt = template.format(prompt)
        generate_images(pipe=pipe, oce_model_path=oce_model_path, prompt=full_prompt, save_path=save_path, exp_name=exp_name, device=device, torch_dtype=torch.bfloat16, guidance_scale = guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt, from_case=from_case, till_case=till_case)