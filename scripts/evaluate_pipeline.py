#!/usr/bin/env python3
"""
Evaluate CDViews: Regular vs. Novel-View-Augmented Pipeline
============================================================

Runs both the **regular** CDViews inference and a **modified** pipeline
(with novel views generated for coverage gaps) on x% of the SQA
validation set, then prints a comparative analysis.

Modified pipeline approach:
  - Original scene images are augmented with novel-view images.
  - Features are computed for novel images using the LLaVA vision encoder.
  - The ViewSelector ranks ALL images (original + novel) together.
  - viewNMS selects the final 9 images from this combined pool.
  - We track how many of the final 9 are novel views (per question).

Usage
-----
    python evaluate_pipeline.py --pct 10
    python evaluate_pipeline.py --pct 5 --seed 42
    python evaluate_pipeline.py --pct 20 --cfg_file ../cfgs/QA.yaml
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# ── Make cdviews/ importable ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "cdviews"))
sys.path.insert(0, str(SCRIPT_DIR))

from qa_utils import load_and_update, get_sqa
from novel_view_generator import (
    generate_novel_views_for_scene,
    load_qwen_pipeline,
)
from analysis import compare_and_report


# ---------------------------------------------------------------------------
#  Cache env vars  (must precede any HF import)
# ---------------------------------------------------------------------------
def _setup_caches():
    HF_ROOT = "/ocean/projects/cis250206p/aanugu"
    for k, v in {
        "HF_HOME": f"{HF_ROOT}/hf-cache",
        "HF_HUB_CACHE": f"{HF_ROOT}/hf-cache/hub",
        "TRANSFORMERS_CACHE": f"{HF_ROOT}/tf-cache",
        "HF_DATASETS_CACHE": f"{HF_ROOT}/ds-cache",
        "DIFFUSERS_CACHE": f"{HF_ROOT}/diff-cache",
        "XDG_CACHE_HOME": f"{HF_ROOT}/.cache",
        "PIP_CACHE_DIR": f"{HF_ROOT}/.cache/pip",
        "TORCH_HOME": f"{HF_ROOT}/.cache/torch",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }.items():
        os.environ[k] = v
        if "/" in v:
            Path(v).mkdir(parents=True, exist_ok=True)

_setup_caches()

# ── Now safe to import HF-dependent modules ──
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

import transformers
import re
import pandas as pd

from ViewSelector import ViewSelector
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from view_distance_calculation import calculate_view_distance
from dataset import ViewLabelDataset
from torch.utils.data import DataLoader
from qa_utils import custom_collate_fn


# ---------------------------------------------------------------------------
#  Prompt construction (mirrors qa_inference.py)
# ---------------------------------------------------------------------------
def preprocess_qwen(sources, tokenizer, has_image=False, max_len=2048,
                    system_message="You are a helpful assistant."):
    """Tokenise a conversation for the Qwen-based LLaVA model."""
    # In transformers >=5.x the TokenizersBackend may not expose
    # additional_special_tokens_ids; use convert_tokens_to_ids instead.
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        im_start, im_end = tokenizer.additional_special_tokens_ids
    else:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    input_id = (
        [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    )

    for sentence in sources:
        role = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall("<image>", sentence["value"]))
            texts = sentence["value"].split("<image>")
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = (
                    tokenizer(role).input_ids + nl_tokens
                    + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                )
        input_id += _input_id

    return torch.tensor([input_id], dtype=torch.long)


# ---------------------------------------------------------------------------
#  View ranking helpers  (mirrors qa_inference.py)
# ---------------------------------------------------------------------------
def split_list(lst, n=50):
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def ranking_views(pair_dataloader, tokenizer, model, view_selector, save_path,
                  chunk_size=50):
    """Rank all images for each question using the ViewSelector."""
    view_selector.eval()
    output_dict = {}
    with torch.no_grad():
        for qs_id_list, qs_list, image_embeds, labels, image_file_list in tqdm(
            pair_dataloader, desc="Ranking views"
        ):
            text_embeds_list = []
            for qs, image_files in zip(qs_list, image_file_list):
                line = {"from": "human", "value": qs}
                input_ids = preprocess_qwen(
                    [line, {"from": "gpt", "value": None}], tokenizer, has_image=False
                ).to(model.device)
                text_embed = model.get_model().embed_tokens(input_ids)
                text_embeds_list += [text_embed.squeeze(0)] * len(image_files)

            padded = pad_sequence(text_embeds_list, batch_first=True).float()
            image_embeds = image_embeds.float()

            if padded.size(0) < chunk_size:
                te, ie = view_selector(image_embeds.to(model.device), padded)
                scores = F.cosine_similarity(te, ie)
            else:
                te_chunks = split_list(padded)
                ie_chunks = split_list(image_embeds)
                scores_list = []
                for te_c, ie_c in zip(te_chunks, ie_chunks):
                    te_out, ie_out = view_selector(ie_c.to(model.device), te_c)
                    scores_list.append(F.cosine_similarity(te_out, ie_out))
                scores = torch.cat(scores_list)

            paired = sorted(zip(scores, image_files), key=lambda x: x[0], reverse=True)
            _, sorted_files = zip(*paired)
            output_dict[str(qs_id_list[0])] = list(sorted_files)

    with open(save_path, "w") as f:
        json.dump(output_dict, f)
    return output_dict


def viewNMS(image_list, neighbour_df, num_images, distance_threshold=0.5):
    """Non-maximum suppression on ranked views for spatial diversity.

    Gracefully handles images not in neighbour_df (e.g. novel views) by
    skipping the NMS distance check for them.
    """
    selected, remaining = [], list(image_list)
    while len(selected) < num_images and remaining:
        current = remaining.pop(0)
        selected.append(current)
        if current in neighbour_df.index:
            dists = neighbour_df.loc[current].sort_values()
            neighbours = set(dists[dists < distance_threshold].index.tolist())
            remaining = [img for img in remaining if img not in neighbours]
    return selected


# ---------------------------------------------------------------------------
#  Feature extraction for novel views
# ---------------------------------------------------------------------------
def extract_novel_view_features(
    novel_image_paths: List[str],
    model,
    image_processor,
) -> Dict[str, torch.Tensor]:
    """Extract LLaVA vision features for novel-view images.

    Mirrors visual_feature_processing.py:
        features = model.encode_images(tensors)
        features = cat(features, image_newline)
    Returns {filename: tensor[730, 3584]} on CPU.
    """
    if not novel_image_paths:
        return {}

    feature_dict = {}
    batch_size = 20  # smaller batches to avoid OOM

    for i in range(0, len(novel_image_paths), batch_size):
        batch_paths = novel_image_paths[i:i + batch_size]
        tensors = []
        names = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            t = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
            tensors.append(t.half().to(model.device))
            # Use novel-color/filename.jpg as the key to distinguish from originals
            names.append(f"novel-color/{os.path.basename(p)}")

        image_tensors = torch.cat(tensors, dim=0)
        num_img = image_tensors.shape[0]

        with torch.inference_mode():
            image_features = model.encode_images(image_tensors)
            image_features = torch.cat(
                (
                    image_features,
                    model.model.image_newline[None]
                    .unsqueeze(0)
                    .repeat(num_img, 1, 1),
                ),
                dim=1,
            )

        for name, feat in zip(names, image_features):
            feature_dict[name] = feat.cpu()

    return feature_dict


# ---------------------------------------------------------------------------
#  Augmented ranking: ViewSelector on original + novel features
# ---------------------------------------------------------------------------
def rank_augmented_views(
    question: str,
    original_feats: Dict[str, torch.Tensor],
    novel_feats: Dict[str, torch.Tensor],
    tokenizer,
    model,
    view_selector,
    chunk_size: int = 50,
) -> List[str]:
    """Rank all images (original + novel) for a single question.

    Returns list of image filenames sorted by descending ViewSelector score.
    Original files are like '380.jpg', novel files like 'novel-color/195.jpg'.
    """
    view_selector.eval()

    # Combine features
    all_files = list(original_feats.keys()) + list(novel_feats.keys())
    all_feats = [original_feats[f] for f in original_feats] + \
                [novel_feats[f] for f in novel_feats]

    if not all_feats:
        return []

    # Stack: [N, 730, 3584]
    image_embeds = torch.stack(all_feats, dim=0).unsqueeze(0)  # [1, N, 730, 3584]

    # Text embedding
    line = {"from": "human", "value": question}
    input_ids = preprocess_qwen(
        [line, {"from": "gpt", "value": None}], tokenizer, has_image=False
    ).to(model.device)
    text_embed = model.get_model().embed_tokens(input_ids)  # [1, seq_len, 3584]

    N = len(all_files)
    text_embeds_list = [text_embed.squeeze(0)] * N
    padded_text = pad_sequence(text_embeds_list, batch_first=True).float()  # [N, seq, 3584]
    image_embeds_flat = image_embeds.squeeze(0).float()  # [N, 730, 3584]

    with torch.no_grad():
        if N <= chunk_size:
            te, ie = view_selector(
                image_embeds_flat.to(model.device),
                padded_text.to(model.device),
            )
            scores = F.cosine_similarity(te, ie)
        else:
            te_chunks = split_list(padded_text)
            ie_chunks = split_list(image_embeds_flat)
            scores_parts = []
            for te_c, ie_c in zip(te_chunks, ie_chunks):
                te_out, ie_out = view_selector(
                    ie_c.to(model.device), te_c.to(model.device)
                )
                scores_parts.append(F.cosine_similarity(te_out, ie_out))
            scores = torch.cat(scores_parts)

    paired = sorted(zip(scores.cpu().tolist(), all_files), key=lambda x: -x[0])
    return [f for _, f in paired]


# ---------------------------------------------------------------------------
#  Core inference for one question
# ---------------------------------------------------------------------------
def run_inference_for_question(
    line: dict,
    image_files: List[str],
    image_folder: str,
    scene_id: str,
    tokenizer,
    model,
    image_processor,
    temperature: float = 0.2,
    top_p: float = None,
    num_beams: int = 1,
) -> str:
    """Run the LLaVA VLM on the given images + question.  Returns answer string.

    Mirrors the inference logic in qa_inference.py eval_model() lines 199-226.
    """
    num_image = len(image_files)

    question = line["situation"] + line["question"]
    value = "<image>" * num_image + question

    conv = [
        {"from": "human", "value": value},
        {"from": "gpt", "value": None},
    ]
    input_ids = preprocess_qwen(conv, tokenizer, has_image=True).to(model.device)

    image_tensors = []
    for img_file in image_files:
        # Support original (color/) and novel (novel-color/) paths
        if os.path.isabs(img_file):
            img_path = img_file
        elif img_file.startswith("novel-color/"):
            img_path = os.path.join(image_folder, scene_id, img_file)
        else:
            img_path = os.path.join(image_folder, scene_id, "color", img_file)

        img = Image.open(img_path).convert("RGB")
        tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
        image_tensors.append(tensor.half().to(model.device))

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=1024,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output.strip().lower().split("\n")[0]


# ---------------------------------------------------------------------------
#  Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(args):
    # ── 1) Load models ──────────────────────────────────────────────────
    disable_torch_init()
    print(f"\n{'='*60}")
    print(f"  CDViews Evaluation — Regular vs. Novel-View Augmented")
    print(f"  SQA val  |  sample = {args.pct}%  |  seed = {args.seed}")
    print(f"{'='*60}\n")

    model_path = os.path.expanduser(args.LVLM_ckpt)
    model_name = get_model_name_from_path(model_path)
    print(f"Loading VLM: {model_name} …")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        attn_implementation="sdpa",          # flash_attn binary is incompatible
    )

    print("Loading ViewSelector …")
    view_selector = ViewSelector().to(model.device)
    ckpt_file = args.pretrained_view_selector_ckpt.format("SQA")
    view_selector.load_state_dict(torch.load(ckpt_file)["model"])
    print(f"  Loaded from {ckpt_file}")

    # ── 2) Load & sample SQA val ────────────────────────────────────────
    sqa_all = get_sqa(args, mode="val")
    n_total = len(sqa_all)
    n_sample = max(1, int(n_total * args.pct / 100))

    rng = random.Random(args.seed)
    sqa_subset = rng.sample(sqa_all, n_sample)
    print(f"\nSampled {n_sample}/{n_total} questions ({args.pct}%)")

    scene_ids = sorted(set(q["scene_id"] for q in sqa_subset))
    print(f"Covering {len(scene_ids)} unique scenes")

    # ── 3) View ranking for regular pipeline (or load cached) ───────────
    rank_file = args.ranked_view_file.format("SQA", "val")
    if os.path.exists(rank_file):
        print(f"Loading cached view rankings from {rank_file}")
        with open(rank_file) as f:
            image_file_list = json.load(f)
    else:
        print("Computing view rankings (this may take a while) …")
        pair_dataset = ViewLabelDataset(args, mode="val")
        pair_loader = DataLoader(pair_dataset, batch_size=1, shuffle=False,
                                 collate_fn=custom_collate_fn)
        image_file_list = ranking_views(
            pair_loader, tokenizer, model, view_selector, rank_file
        )

    # ── 4) Generate novel views for all relevant scenes ─────────────────
    print(f"\n{'─'*60}")
    print("Generating novel views for coverage gaps …")
    print(f"{'─'*60}")

    qwen_pipe = None
    novel_view_map: Dict[str, List[str]] = {}  # scene_id → list of abs paths

    for scene_id in tqdm(scene_ids, desc="Novel view generation"):
        scene_dir = os.path.join(args.image_folder, scene_id)
        novel_dir = os.path.join(scene_dir, "novel-color")

        existing = sorted(Path(novel_dir).glob("*.jpg")) if os.path.isdir(novel_dir) else []
        if len(existing) >= args.max_novel_views:
            novel_view_map[scene_id] = [str(p) for p in existing[:args.max_novel_views]]
            continue

        if qwen_pipe is None:
            print("\n  Loading Qwen pipeline for novel-view generation …")
            qwen_pipe = load_qwen_pipeline()

        paths = generate_novel_views_for_scene(
            scene_dir,
            max_views=args.max_novel_views,
            bin_deg=args.bin_deg,
            pipe=qwen_pipe,
            seed=args.seed,
        )
        novel_view_map[scene_id] = paths

    total_novel = sum(len(v) for v in novel_view_map.values())
    print(f"\nNovel views ready: {total_novel} images across {len(novel_view_map)} scenes")

    # Free Qwen GPU memory
    if qwen_pipe is not None:
        del qwen_pipe
        torch.cuda.empty_cache()
        print("Freed Qwen pipeline GPU memory")

    # ── 5) Pre-compute novel-view features per scene ────────────────────
    print(f"\n{'─'*60}")
    print("Extracting LLaVA features for novel views …")
    print(f"{'─'*60}")

    novel_feats_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    for scene_id in tqdm(scene_ids, desc="Novel feature extraction"):
        paths = novel_view_map.get(scene_id, [])
        if not paths:
            novel_feats_cache[scene_id] = {}
            continue
        novel_feats_cache[scene_id] = extract_novel_view_features(
            paths, model, image_processor
        )

    print(f"  Features extracted for {sum(len(v) for v in novel_feats_cache.values())} novel images")

    # ── 6) Run inference: regular & augmented ───────────────────────────
    answers_regular = []
    answers_augmented = []
    novel_selection_tracker = []  # per-question tracking

    print(f"\n{'─'*60}")
    print(f"Running inference on {n_sample} questions …")
    print(f"{'─'*60}")

    pbar = tqdm(sqa_subset, desc="Inference", unit="q")
    for line in pbar:
        scene_id = line["scene_id"]
        question_id = str(line["question_id"])
        pbar.set_postfix(scene=scene_id, novel_sel=sum(
            t["novel_selected"] for t in novel_selection_tracker
        ), refresh=False)

        # -- View distance for NMS --
        vd_file = os.path.join(args.view_distance_folder, f"{scene_id}.csv")
        if os.path.exists(vd_file):
            distance_df = pd.read_csv(vd_file, index_col=0)
        else:
            distance_df = calculate_view_distance(scene_id, args)
            os.makedirs(args.view_distance_folder, exist_ok=True)
            distance_df.to_csv(vd_file)

        # -- Regular pipeline: rank + NMS on original images only --
        if question_id in image_file_list:
            ranked = image_file_list[question_id]
        else:
            color_dir = os.path.join(args.image_folder, scene_id, "color")
            ranked = sorted(os.listdir(color_dir))

        selected_regular = viewNMS(ranked, distance_df, num_images=args.input_views)
        selected_regular = selected_regular[:args.input_views]

        ans_reg = run_inference_for_question(
            line, selected_regular, args.image_folder, scene_id,
            tokenizer, model, image_processor,
            temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams,
        )
        answers_regular.append({
            "scene_id": scene_id,
            "question_id": question_id,
            "answer_top10": [ans_reg] * 10,
        })

        # -- Augmented pipeline: ViewSelector ranks original + novel --
        # Load original features
        feat_file = os.path.join(args.feature_folder, f"{scene_id}.pth")
        if os.path.exists(feat_file):
            original_feats = torch.load(feat_file, map_location="cpu")
        else:
            # Fallback: skip augmented if no features
            answers_augmented.append({
                "scene_id": scene_id,
                "question_id": question_id,
                "answer_top10": [ans_reg] * 10,
            })
            novel_selection_tracker.append({
                "question_id": question_id,
                "scene_id": scene_id,
                "total_novel_available": 0,
                "novel_selected": 0,
                "novel_selected_names": [],
                "selected_images": list(selected_regular),
            })
            continue

        novel_feats = novel_feats_cache.get(scene_id, {})

        # Build the question text for ranking
        q_text = line["situation"] + line["question"]

        # Rank all images together
        combined_ranked = rank_augmented_views(
            q_text, original_feats, novel_feats,
            tokenizer, model, view_selector,
        )

        # Apply viewNMS (novel views won't be in distance_df — viewNMS
        # handles this gracefully by skipping the distance check)
        selected_augmented = viewNMS(
            combined_ranked, distance_df, num_images=args.input_views
        )
        selected_augmented = selected_augmented[:args.input_views]

        # Track novel views selected
        novel_in_selection = [f for f in selected_augmented if f.startswith("novel-color/")]
        n_novel_selected = len(novel_in_selection)

        novel_selection_tracker.append({
            "question_id": question_id,
            "scene_id": scene_id,
            "total_novel_available": len(novel_feats),
            "novel_selected": n_novel_selected,
            "novel_selected_names": novel_in_selection,
            "selected_images": list(selected_augmented),
        })

        # Run augmented inference
        ans_aug = run_inference_for_question(
            line, selected_augmented, args.image_folder, scene_id,
            tokenizer, model, image_processor,
            temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams,
        )
        answers_augmented.append({
            "scene_id": scene_id,
            "question_id": question_id,
            "answer_top10": [ans_aug] * 10,
        })

    # ── 7) Save outputs ────────────────────────────────────────────────
    output_dir = os.path.join(str(PROJECT_ROOT), "analysis_outputs")
    os.makedirs(output_dir, exist_ok=True)

    tag = f"SQA_val_{args.pct}pct_seed{args.seed}"

    reg_file = os.path.join(output_dir, f"{tag}_regular_answers.json")
    aug_file = os.path.join(output_dir, f"{tag}_augmented_answers.json")
    tracker_file = os.path.join(output_dir, f"{tag}_novel_selection_tracker.json")

    with open(reg_file, "w") as f:
        json.dump(answers_regular, f, indent=2)
    with open(aug_file, "w") as f:
        json.dump(answers_augmented, f, indent=2)
    with open(tracker_file, "w") as f:
        json.dump(novel_selection_tracker, f, indent=2)

    print(f"\nAnswers saved:")
    print(f"  Regular:   {reg_file}")
    print(f"  Augmented: {aug_file}")
    print(f"  Tracker:   {tracker_file}")

    # ── 8) Comparative analysis ─────────────────────────────────────────
    compare_and_report(
        predictions_regular=answers_regular,
        predictions_augmented=answers_augmented,
        ground_truth=sqa_subset,
        novel_tracker=novel_selection_tracker,
        output_dir=output_dir,
        tag=tag,
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CDViews regular vs. novel-view-augmented pipeline"
    )
    parser.add_argument("--pct", type=float, default=10,
                        help="Percentage of SQA val set to evaluate (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset sampling (default: 42)")
    parser.add_argument("--max_novel_views", type=int, default=10,
                        help="Max novel views per scene (default: 10)")
    parser.add_argument("--bin_deg", type=float, default=30.0,
                        help="Azimuth bin width for coverage analysis (default: 30)")
    parser.add_argument("--cfg_file", type=str, default="../cfgs/QA.yaml",
                        help="CDViews config file (default: ../cfgs/QA.yaml)")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()
    args = load_and_update(args)
    args.dataset = "SQA"

    evaluate(args)


if __name__ == "__main__":
    main()
