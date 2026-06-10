import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
import os
import pandas as pd

def compute_and_save_Ce_flux(
    model_id,
    save_dir=".",
    dataset_path="coco_30k.csv",
    sample_size=600000,
    batch_size=16,
    device="cuda",
    shrinkage_alpha=0.0,
):
  
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        vae=None,
        transformer=None,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)

    text_encoder  = pipe.text_encoder    # CLIP ViT-L (768-dim)
    text_encoder_2 = pipe.text_encoder_2  # T5-XXL     (4096-dim)
    tokenizer     = pipe.tokenizer        # CLIP tokenizer
    tokenizer_2   = pipe.tokenizer_2      # T5 tokenizer

    clip_dim = text_encoder.config.hidden_size        # 768
    t5_dim   = text_encoder_2.config.d_model          # 4096

    max_seq_len = 512

    text_encoder.to(device).eval()
    text_encoder_2.to(device).eval()

    save_path_clip = os.path.join(save_dir, "Cg_clip.pt")
    save_path_t5   = os.path.join(save_dir, "Cg_t5.pt")

    if os.path.exists(save_path_clip):
        data = torch.load(save_path_clip, map_location=device)
        C_clip = data["C"]
        count_clip = data["count"]
        print(f"[Load] existing Cg_clip from {save_path_clip}, count={count_clip}")
    else:
        C_clip = torch.zeros(clip_dim, clip_dim, device=device)
        count_clip = 0

    if os.path.exists(save_path_t5):
        data = torch.load(save_path_t5, map_location=device)
        C_t5 = data["C"]
        count_t5 = data["count"]
        print(f"[Load] existing Cg_t5 from {save_path_t5}, count={count_t5}")
    else:
        C_t5 = torch.zeros(t5_dim, t5_dim, device=device)
        count_t5 = 0

    ds = pd.read_csv(dataset_path)
    texts = ds['prompt']

    batch_texts = []
    for item in tqdm(texts, desc="Computing Cg for Flux"):

        if count_clip >= sample_size and count_t5 >= sample_size:
            break

        batch_texts.append(item)
        if len(batch_texts) < batch_size:
            continue

        clip_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=tokenizer.model_max_length,
        ).to(device)

        with torch.no_grad():
            clip_out = text_encoder(**clip_inputs)
            pooled = clip_out.pooler_output  # (B, 768)

        this_add_clip = min(sample_size - count_clip, pooled.shape[0])
        pooled = pooled[:this_add_clip]
        C_clip += pooled.T @ pooled   # (768, B) @ (B, 768) → (768, 768)
        count_clip += this_add_clip

        t5_inputs = tokenizer_2(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            t5_out = text_encoder_2(**t5_inputs)
            H_t5 = t5_out.last_hidden_state  # (B, L, 4096)

        B_t5, L_t5, D_t5 = H_t5.shape
        H2 = H_t5.reshape(B_t5 * L_t5, D_t5)  # (B*L, 4096)

        mask = t5_inputs["attention_mask"].reshape(-1) > 0
        H2 = H2[mask]

        this_add_t5 = min(sample_size - count_t5, H2.shape[0])
        H2 = H2[:this_add_t5]
        C_t5 += H2.T @ H2   # (4096, N) @ (N, 4096) → (4096, 4096)
        count_t5 += this_add_t5

        batch_texts = []

        if count_clip % 20000 < batch_size or count_t5 % 20000 < batch_size:
            torch.save({"C": C_clip, "count": count_clip}, save_path_clip)
            torch.save({"C": C_t5, "count": count_t5}, save_path_t5)
            print(f"[Checkpoint] CLIP: {count_clip} embeddings, T5: {count_t5} tokens")

    if count_clip > 0:
        C_clip /= count_clip
    if count_t5 > 0:
        C_t5 /= count_t5

    if shrinkage_alpha > 0:
        sigma_clip = torch.mean(torch.diag(C_clip))
        C_clip = (1 - shrinkage_alpha) * C_clip + shrinkage_alpha * sigma_clip * torch.eye(clip_dim, device=device)
        sigma_t5 = torch.mean(torch.diag(C_t5))
        C_t5 = (1 - shrinkage_alpha) * C_t5 + shrinkage_alpha * sigma_t5 * torch.eye(t5_dim, device=device)

    torch.save({"C": C_clip, "count": count_clip}, save_path_clip)
    torch.save({"C": C_t5, "count": count_t5}, save_path_t5)
    print(f"[Final] Saved Cg_clip.pt ({clip_dim}×{clip_dim}, count={count_clip})")
    print(f"[Final] Saved Cg_t5.pt  ({t5_dim}×{t5_dim}, count={count_t5})")

    del pipe, text_encoder, text_encoder_2
    torch.cuda.empty_cache()

    return C_clip, C_t5


# ================= RUN ====================
if __name__ == "__main__":
    torch_dtype = torch.float32

    model_id = "./FLUX.1-dev/"

    C_clip, C_t5 = compute_and_save_Cg_flux(
        model_id=model_id,
        save_dir=".",
        dataset_path="coco_30k.csv",
        sample_size=600000,
        batch_size=16,  
        device="cuda",
        shrinkage_alpha=0.0,
    )
