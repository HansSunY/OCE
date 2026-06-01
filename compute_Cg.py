import torch
from datasets import load_dataset
from tqdm import tqdm
from diffusers import DiffusionPipeline
import os
import pandas as pd
def compute_and_save_second_moment(
    text_encoder,
    tokenizer,
    save_path="Cg.pt",
    dataset_path="coco_30k.csv",
    sample_size=600000,
    batch_size=32,
    device="cuda",
    shrinkage_alpha=0.0,
):

    hidden_dim = text_encoder.config.hidden_size

    # Load or init
    if os.path.exists(save_path):
        print(f"[Load] existing C_g from {save_path}")
        data = torch.load(save_path, map_location=device)
        C = data["C"]
        count = data["count"]
    else:
        C = torch.zeros(hidden_dim, hidden_dim, device=device)
        count = 0

    # dataset
    ds = pd.read_csv(dataset_path)
    ds = ds['prompt']
    #ds = load_dataset(dataset_path)
    #ds = ds["train"]

    text_encoder.to(device)
    text_encoder.eval()

    # ========= MAIN GPU BATCH LOOP =========
    batch_texts = []
    for item in tqdm(ds, desc="Collecting hidden states"):

        if count >= sample_size:
            break

        # accumulate batch texts
        #batch_texts.append(item["text"])
        batch_texts.append(item)
        if len(batch_texts) < batch_size:
            continue

        # --- GPU tokenize ---
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=tokenizer.model_max_length,
        ).to(device)

        with torch.no_grad():
            out = text_encoder(**inputs, output_hidden_states=True)
            H = out.last_hidden_state  # (B, L, D)

        B, L, D = H.shape

        # merge batch × seq into one big matrix
        H2 = H.reshape(B * L, D)        # (B*L, D)

        # remove paddings (zero mask)
        mask = inputs["attention_mask"].reshape(-1) > 0
        H2 = H2[mask]                  # (~N, D)

        # how many tokens to use
        this_add = min(sample_size - count, H2.shape[0])
        H2 = H2[:this_add]

        # update C using GPU GEMM
        C += H2.T @ H2
        count += this_add

        batch_texts = []

        # periodic checkpoint
        if count % 20000 == 0:
            torch.save({"C": C, "count": count}, save_path)
            print(f"[Checkpoint] saved {count} tokens -> {save_path}")

    # ===== normalize ======
    C /= count

    # shrinkage
    if shrinkage_alpha > 0:
        sigma = torch.mean(torch.diag(C))
        C = (1 - shrinkage_alpha) * C + shrinkage_alpha * sigma * torch.eye(hidden_dim, device=device)

    torch.save({"C": C, "count": count}, save_path)
    print(f"[Final] Saved Cg, count={count}")

    return C


# ================= RUN ====================

torch_dtype = torch.float32
pipe = DiffusionPipeline.from_pretrained(
    "/model/stable-diffusion-v1-4/",
    torch_dtype=torch_dtype,
    safety_checker=None,
    vae=None
).to("cuda")

text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

C_e = compute_and_save_second_moment(
    text_encoder,
    tokenizer,
    save_path="Cg.pt",
    dataset_path="coco_30k.csv"
    sample_size=60000000,
    batch_size=64, 
    device="cuda",
)
