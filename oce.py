import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time

from safetensors.torch import save_file
from diffusers import DiffusionPipeline
from safetensors.torch import load_file

def build_erase_subspace(W0, erase_embs,eps=1e-8):
    V = []
    for K in erase_embs:
        v = W0 @ K
        v = v / (v.norm() + eps)
        V.append(v)
    E = torch.stack(V, dim=1)
    E_orth, _ = torch.linalg.qr(E, mode="reduced")
    return E_orth

def build_guide_subspace(W0, guide_embs, eps=1e-8):
    G = []
    for g in guide_embs:
        v = W0 @ g
        v = v / (v.norm() + eps)
        G.append(v)
    G = torch.stack(G, dim=1)
    G_orth, _ = torch.linalg.qr(G, mode="reduced")
    return G_orth

def build_preserve_subspace(W0, preserve_embs, eps=1e-8):
    V = []
    for Kp in preserve_embs:
        v = W0 @ Kp
        v = v / (v.norm() + eps)
        V.append(v)
    P = torch.stack(V, dim=1)
    P_orth, _ = torch.linalg.qr(P, mode="reduced")
    return P_orth

def align_guides_with_edits(edit_concepts, guide_concepts, seed=None):
    if seed is not None:
        random.seed(seed)

    n_edit = len(edit_concepts)
    n_guide = len(guide_concepts)

    if n_guide == 0:
        raise ValueError("no guide_concepts")

    if n_guide < n_edit:
        extra = random.choices(guide_concepts, k=n_edit - n_guide)
        guide_concepts = guide_concepts + extra
    elif n_guide > n_edit:
        guide_concepts = guide_concepts[:n_edit]

    return guide_concepts

def Orthogonal_Erase(pipe, edit_concepts, guide_concepts, preserve_concepts,
                        erase_scale, preserve_scale, preserve_scale_2,
                        lamb, save_dir, exp_name):
    start_time = time.time()
    print(f"Editing with {edit_concepts}")

    ce_stats = torch.load("Ce.pt", map_location=pipe.device)
    C_e = ce_stats["C"].to(device, dtype=torch_dtype)

    # ===== collect target modules =====
    target_modules, module_names = [], []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v')):
            target_modules.append(module)
            module_names.append(name)

    updated_modules = copy.deepcopy(target_modules)

    # ===== collect embeddings =====
    embed_cache = {}
    all_prompts = edit_concepts + guide_concepts + preserve_concepts
    for e in all_prompts:
        if not e:
            continue
        t_emb = pipe.encode_prompt(prompt=e, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        last_idx = (pipe.tokenizer(e, padding="max_length",
                                   max_length=pipe.tokenizer.model_max_length,
                                   truncation=True,
                                   return_tensors="pt")["attention_mask"]).sum() - 2
        embed_cache[e] = t_emb[0][:, last_idx, :].squeeze(0).to(device=device, dtype=torch_dtype)

    # ===== main loop =====
    for module_idx, module in enumerate(target_modules):
        W0 = module.weight.detach().to(device=device, dtype=torch_dtype)
        out_dim, in_dim = W0.shape
        guide_embs= [embed_cache[e] for e in guide_concepts if e]
        erase_embs = [embed_cache[e] for e in edit_concepts if e]
        preserve_embs = [embed_cache[e] for e in preserve_concepts if e]

        if not erase_embs:
            continue

        E_star = build_guide_subspace(W0, guide_embs)      # (out_dim, r_guide)

        P = build_preserve_subspace(W0, preserve_embs)     # (out_dim, s)

        # === (2) build M_total ===
        M_total = torch.zeros(out_dim, out_dim, device=device, dtype=torch_dtype)
        G_star = E_star @ E_star.T
        E = build_erase_subspace(W0, erase_embs)
        G = E @ E.T
        I= torch.eye(G.shape[0], device=device, dtype=torch_dtype)
        M_total += -erase_scale * G @ (I - G_star)
        for Kp in preserve_embs:
            v = W0 @ Kp
            M_total += preserve_scale_2 * (v.unsqueeze(1) @ v.unsqueeze(0))
        M_total += preserve_scale * (W0 @ C_e @ W0.T)
        M_total += lamb * (W0 @ W0.T)
        # === (3) Procrustes ===
        U, _, Vh = torch.linalg.svd(M_total, full_matrices=False)
        R = U @ Vh
        if torch.det(R) < 0:
            R[:, -1] *= -1
        W_new = R @ W0
        updated_modules[module_idx].weight = torch.nn.Parameter(W_new)
        print(f"Updated {module_names[module_idx]}")
    print(f"\nErased concepts using Subspace Orthogonal Procrustes (RW)\nModel edited in {time.time() - start_time:.2f} seconds")

    # ===== save =====
    state_dict = {name + ".weight": mod.weight.detach().cpu()
                  for name, mod in zip(module_names, updated_modules)}
    save_file(state_dict, os.path.join(save_dir, exp_name + ".safetensors"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainOrthogonalErase',
                    description = 'Orthogonal Concept Erasure for Stable Diffusion')
    parser.add_argument('--edit_concepts', help='prompts corresponding to concepts to erase separated by ;', type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide the erased concepts towards seperated by ;', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve seperated by ;', type=str, default=None)
    
    parser.add_argument('--concept_type', help='type of concept being erased', choices=['art', 'object'], type=str, required=True)

    parser.add_argument('--model_id', help='Model to run on', type=str, default="CompVis/stable-diffusion-v1-4",)
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='cuda:0')

    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--preserve_global_scale', help='scale to preserve global knowledge', type=float, required=False, default=0.5)
    parser.add_argument('--preserve_concept_scale', help='scale to preserve concepts', type=float, required=False, default=0.3)

    parser.add_argument('--lamb', help='lambda regularization term', type=float, required=False, default=1.5)

    parser.add_argument('--expand_prompts', help='do you wish to expand your prompts?', choices=['true', 'false'], type=str, required=False, default='false')

    parser.add_argument('--save_dir', help='where to save your edited model weights', type=str, default='./oce_models')
    parser.add_argument('--exp_name', help='Use this to name your saved filename', type=str, default=None)


    args = parser.parse_args()

    device = args.device
    torch_dtype = torch.float32
    model_id = args.model_id

    preserve_scale = args.preserve_global_scale
    preserve_scale_2 = args.preserve_concept_scale
    erase_scale = args.erase_scale
    lamb = args.lamb

    concept_type = args.concept_type
    expand_prompts = args.expand_prompts

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'orthogonal_erase'

    # parse prompts
    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(';') if concept.strip()!='']
    guide_concepts = args.guide_concepts
    if guide_concepts is None:
        guide_concepts = [' ']
        if concept_type == 'art':
            guide_concepts = ['art']
    else:
        guide_concepts = [concept.strip() for concept in guide_concepts.split(';') if concept.strip()!='']

        
    if len(guide_concepts) != len(edit_concepts):
        guide_concepts = align_guides_with_edits(
            edit_concepts,
            guide_concepts,
            seed=42
        )

    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(';') if concept.strip()!='']

    if expand_prompts == 'true':
        edit_concepts_ = copy.deepcopy(edit_concepts)
        guide_concepts_ = copy.deepcopy(guide_concepts)

        for concept, guide_concept in zip(edit_concepts_, guide_concepts_):
            if concept_type == 'art':
                extra_e = [f'painting by {concept}',
                           f'art by {concept}',
                           f'artwork by {concept}',
                           f'picture by {concept}',
                           f'style of {concept}']
                extra_g = [f'painting by {guide_concept}',
                           f'art by {guide_concept}',
                           f'artwork by {guide_concept}',
                           f'picture by {guide_concept}',
                           f'style of {guide_concept}']
            else:
                extra_e = [f'image of {concept}',
                           f'photo of {concept}',
                           f'portrait of {concept}',
                           f'picture of {concept}',
                           f'painting of {concept}']
                extra_g = [f'image of {guide_concept}',
                           f'photo of {guide_concept}',
                           f'portrait of {guide_concept}',
                           f'picture of {guide_concept}',
                           f'painting of {guide_concept}']

            edit_concepts.extend(extra_e)
            guide_concepts.extend(extra_g)

    print(f"\n\nErasing: {edit_concepts}\n")
    print(f"Guiding: {guide_concepts}\n")
    print(f"Preserving: {preserve_concepts}\n")

    pipe = DiffusionPipeline.from_pretrained(model_id,
                                             torch_dtype=torch_dtype,
                                             safety_checker=None,
                                             vae=None).to(device)
    Orthogonal_Erase(pipe, edit_concepts, guide_concepts, preserve_concepts, erase_scale, preserve_scale, preserve_scale_2,lamb, save_dir, exp_name)
