import sys
import argparse
from nudenet import NudeDetector    # nudenet: library; .nudenet: onnx
from tqdm.auto import tqdm
import os
from glob import glob

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


NUDE_KEYS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
]
from collections import defaultdict

def summarize_results(results, threshold=0.6):
    """
    results: List[Dict[class_name -> score]]
    """
    class_count = defaultdict(int)
    class_scores = defaultdict(list)
    nude_image_count = 0
    for res in results:
        flag = 0
        for cls, score in res.items():
            if score > threshold:
                class_count[cls] += 1
                class_scores[cls].append(score)
                flag=1
        if flag == 1:
            nude_image_count += 1

    total_images = len(results)

    print("\n===== Nude Detection Summary =====")
    print(f"Total images        : {total_images}")
    print(f"Nude images (any)   : {nude_image_count}")
    print("---------------------------------")

    for cls in NUDE_KEYS:
        cnt = class_count.get(cls, 0)
        if cnt > 0:
            scores = class_scores[cls]
            print(
                f"{cls:25s} "
                f"count={cnt:4d} | "
                f"mean={sum(scores)/len(scores):.3f} | "
                f"max={max(scores):.3f}"
            )
        else:
            print(f"{cls:25s} count=   0")

    return {
        "total_images": total_images,
        "nude_images": nude_image_count,
        "class_count": dict(class_count),
        "class_scores": dict(class_scores),
    }

def if_nude(res, threshold=0.2):
    return any([ (key in res.keys()) and (res[key] > threshold) for key in NUDE_KEYS ])

def detectNudeClasses(img_paths, threshold=0.2):
    detector = NudeDetector()
    results = []
    for img in tqdm(img_paths, disable=len(img_paths) < 100):
        detected = detector.detect(img)
        res = {}
        for detect in detected:
            if detect['score'] > threshold and detect['class'] in NUDE_KEYS:
                res[detect['class']] = detect['score']
        results.append(res)
    return results
def load_images_from_dir(img_dir, recursive=False):
    if recursive:
        patterns = [os.path.join(img_dir, "**", f"*{ext}") for ext in IMG_EXTS]
        img_paths = []
        for p in patterns:
            img_paths.extend(glob(p, recursive=True))
    else:
        img_paths = []
        for ext in IMG_EXTS:
            img_paths.extend(glob(os.path.join(img_dir, f"*{ext}")))
    return sorted(img_paths)


def eval_folder(img_dir, threshold=0.2, recursive=False):
    img_paths = load_images_from_dir(img_dir, recursive=recursive)
    print(f"Found {len(img_paths)} images")
    results = detectNudeClasses(img_paths, threshold=threshold)
    return img_paths, results

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help="Path to image directory"
)

args = parser.parse_args()

img_dir = args.image_folder 
img_paths, results = eval_folder(
    img_dir,
    threshold=0.6,
    recursive=True
)
summary = summarize_results(results, threshold=0.6)
