import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from PIL import Image
import argparse
import os

torch.set_grad_enabled(False)

def generate_flux_single_prompt(
    model_id,
    oce_model_path,
    prompt,
    save_path,
    exp_name='test',
    device='cuda:0',
    torch_dtype=torch.bfloat16,
    guidance_scale=7.5,
    num_inference_steps=25,
    num_images=5
):
    generator = torch.manual_seed(42)
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None
    ).to(device)

    if oce_model_path is not None:
        oce_weights = load_file(oce_model_path)
        pipe.transformer.load_state_dict(oce_weights, strict=False)
        print(f"Loaded OCE weights from {oce_model_path}")

    folder_path = os.path.join(save_path, exp_name)
    os.makedirs(folder_path, exist_ok=True)

    pil_images = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=generator
    ).images

    for i, im in enumerate(pil_images[:num_images]):
        filename = f"{prompt.replace(' ', '_')[:50]}_{i}.png"
        im.save(os.path.join(folder_path, filename))
        print(f"Saved: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='flux_generate_single',
        description='Generate images from Flux model'
    )
    parser.add_argument('--model_id', type=str, required=True, help='Flux model repo id')
    parser.add_argument('--oce_model_path', type=str, default=None, help='Path to OCE weights')
    parser.add_argument('--prompt', type=str, required=True, help='Single prompt text')
    parser.add_argument('--save_path', type=str, default='./flux_results', help='Folder to save images')
    parser.add_argument('--exp_name', type=str, default='test_images', help='Subfolder name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=25)
    parser.add_argument('--num_images_per_prompt', type=int, default=1)

    args = parser.parse_args()

    generate_flux_single_prompt(
        model_id=args.model_id,
        oce_model_path=args.oce_model_path,
        prompt=args.prompt,
        save_path=args.save_path,
        exp_name=args.exp_name,
        device=args.device,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_images=args.num_images_per_prompt
    )
