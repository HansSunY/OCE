import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from argparse import ArgumentParser


@torch.no_grad()
def clip_classification_stats(image_dir, prompt_list):
    """
    Given a directory of images and a list of class prompts,
    compute per-class probability statistics using CLIP.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # collect images
    image_filenames = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]
    if not image_filenames:
        raise ValueError("No valid images found in the directory.")

    image_filenames = sorted(image_filenames)

    class_count = len(prompt_list)
    probs_collect = [[] for _ in range(class_count)]

    print(f"Evaluating {len(image_filenames)} images on {class_count} classes...\n")

    # compute probabilities
    for img_name in tqdm(image_filenames):
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        inputs = processor(
            text=prompt_list,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits_per_image  # [1, num_classes]
        probs = logits.softmax(dim=1).detach().cpu().numpy()[0]

        # append per-class probability
        for i in range(class_count):
            probs_collect[i].append(probs[i])

    # compute mean & std
    print("\n---------------- Classification Stats ----------------\n")
    for i, cls_prompt in enumerate(prompt_list):
        cls_probs = np.array(probs_collect[i])
        mean_p = cls_probs.mean()
        std_p = cls_probs.std()

        print(f"Class: {cls_prompt}")
        print(f"  Mean prob: {mean_p:.4f}")
        print(f"  Std prob : {std_p:.4f}\n")

    print("-------------------------------------------------------\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    args = parser.parse_args()
    prompt_list = ["photo of a airplane", "photo of a automobile", "photo of a bird", "photo of a cat", "photo of a deer", "photo of a dog", "photo of a frog", "photo of a horse", "photo of a ship", "photo of a truck"]
    clip_classification_stats(args.image_folder, prompt_list)

