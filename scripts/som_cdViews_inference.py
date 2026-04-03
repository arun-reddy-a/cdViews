"""cdViews QA inference with Set-of-Mark (SoM) prompted views.

Mirrors the standard cdViews pipeline (view ranking → viewNMS → LLaVA-OV QA)
but after the nine views are selected, each one is run through SAM2 to produce
a SoM-annotated image on the fly.  Generated SoM images are saved to:
    {som_output_folder}/{scene_id}/{image_file}
so repeated runs skip already-processed frames.

Usage (from scripts/):
    python som_cdViews_inference.py --cfg_file ../cfgs/QA.yaml
    python som_cdViews_inference.py --cfg_file ../cfgs/QA.yaml --sam-model sam2.1_l.pt
"""

import argparse
import colorsys
import json
import os
import re
import sys
import time
from typing import Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CDVIEWS_DIR = os.path.join(REPO_ROOT, "cdviews")
for _p in (REPO_ROOT, CDVIEWS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from qa_utils import custom_collate_fn, get_scanqa, get_sqa, load_and_update
from dataset import ViewLabelDataset
from ViewSelector import ViewSelector
from view_distance_calculation import calculate_view_distance


# ── SAM2 / SoM helpers ─────────────────────────────────────────────────────

def make_sam2_predictor(model_name: str = "sam2.1_s.pt", device: str = "cuda:0"):
    from ultralytics.models.sam import SAM2Predictor

    return SAM2Predictor(overrides=dict(
        task="segment",
        mode="predict",
        imgsz=1024,
        model=model_name,
        device=device,
        verbose=False,
        save=False,
    ))


def generate_som_image(
    image_path: str,
    predictor,
    alpha: float = 0.3,
    min_area_frac: float = 0.0005,
    points_stride: int = 16,
    crop_n_layers: int = 1,
    conf_thres: float = 0.86,
    stability_score_thresh: float = 0.92,
) -> np.ndarray:
    """Run SAM2 segmentation on *image_path* and return an RGB array with
    colour-coded mask overlays and numbered labels at each segment centroid."""
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    min_area = min_area_frac * h * w

    results = predictor(
        source=image_path,
        points_stride=points_stride,
        crop_n_layers=crop_n_layers,
        conf_thres=conf_thres,
        stability_score_thresh=stability_score_thresh,
    )

    if not results or results[0].masks is None:
        return img

    masks_raw = results[0].masks.data.cpu().numpy()
    scores = (results[0].boxes.conf.cpu().numpy()
              if results[0].boxes is not None else None)

    processed = []
    for i, m in enumerate(masks_raw):
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST)
        mask_bool = m.astype(bool)
        area = mask_bool.sum()
        if area >= min_area:
            score = (float(scores[i])
                     if scores is not None and i < len(scores) else 1.0)
            processed.append((mask_bool, area, score))

    processed.sort(key=lambda x: x[1], reverse=True)
    n = len(processed)
    if n == 0:
        return img

    golden_ratio_inv = 0.618033988749895
    hue = 0.0
    colors = []
    for _ in range(n):
        r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.75, 0.95)
        colors.append(np.array([int(r * 255), int(g * 255), int(b * 255)],
                               dtype=np.float64))
        hue += golden_ratio_inv

    overlay = img.astype(np.float64)
    contour_layer = np.zeros_like(img)

    for (mask, _area, _score), color in zip(processed, colors):
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_layer, contours, -1,
                         color.astype(int).tolist(), 2)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    contour_pixels = contour_layer.any(axis=2)
    overlay[contour_pixels] = contour_layer[contour_pixels]

    som_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(som_pil)

    font_size = max(12, min(h, w) // 45)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for idx, ((mask, _area, _score), color) in enumerate(zip(processed, colors)):
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(idx + 1)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        draw.rounded_rectangle(
            [cx - tw // 2 - pad, cy - th // 2 - pad,
             cx + tw // 2 + pad, cy + th // 2 + pad],
            radius=4, fill=(0, 0, 0, 210))
        draw.text((cx - tw // 2, cy - th // 2), label,
                  fill=(255, 255, 255), font=font)

    return np.array(som_pil)


def get_or_create_som(
    scene_id: str,
    image_file: str,
    image_folder: str,
    som_output_folder: str,
    sam_predictor,
) -> Image.Image:
    """Return a SoM-prompted PIL image for a selected view.

    Checks the cache directory first; if the SoM image does not yet exist it is
    generated on the fly with SAM2 and saved for future reuse.
    """
    som_scene_dir = os.path.join(som_output_folder, scene_id)
    som_path = os.path.join(som_scene_dir, image_file)

    if os.path.exists(som_path):
        return Image.open(som_path).convert("RGB")

    color_path = os.path.join(image_folder, scene_id, "color", image_file)
    som_arr = generate_som_image(color_path, sam_predictor)

    os.makedirs(som_scene_dir, exist_ok=True)
    som_img = Image.fromarray(som_arr)
    som_img.save(som_path, quality=95)
    return som_img


# ── Qwen tokenizer preprocessing (same as qa_inference.py) ─────────────────

def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len: int = 2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    if hasattr(tokenizer, "additional_special_tokens_ids"):
        im_start, im_end = tokenizer.additional_special_tokens_ids
    else:
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    input_ids, targets = [], []
    source = sources

    input_id, target = [], []
    system = ([im_start] + _system
              + tokenizer(system_message).input_ids + [im_end] + nl_tokens)
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)

    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split("<image>")
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([tok == IMAGE_TOKEN_INDEX for tok in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = (tokenizer(role).input_ids + nl_tokens
                             + tokenizer(sentence["value"]).input_ids
                             + [im_end] + nl_tokens)
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = ([im_start] + [IGNORE_INDEX] * (len(_input_id) - 3)
                       + [im_end] + nl_tokens)
        elif role == "<|im_start|>assistant":
            _target = ([im_start]
                       + [IGNORE_INDEX] * len(tokenizer(role).input_ids)
                       + _input_id[len(tokenizer(role).input_ids) + 1: -2]
                       + [im_end] + nl_tokens)
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


# ── View selection helpers ──────────────────────────────────────────────────

def split_list(input_list, chunk_size=50):
    return [input_list[i:i + chunk_size]
            for i in range(0, len(input_list), chunk_size)]


def ranking_views(pair_dataloader, tokenizer, model, view_selector,
                  save_path, chunk_size=50):
    view_selector.eval()
    output_dict = {}

    with torch.no_grad():
        for qs_id_list, qs_list, image_embeds, labels, image_file_list in tqdm(pair_dataloader):
            text_embeds_list = []
            for qs, image_files in zip(qs_list, image_file_list):
                line = {"from": "human", "value": qs}
                input_ids = preprocess_qwen(
                    [line, {"from": "gpt", "value": None}],
                    tokenizer, has_image=False,
                ).to(model.device)
                text_embed = model.get_model().embed_tokens(input_ids)
                text_embeds_list += [text_embed.squeeze(0)] * len(image_files)

            padded_text_embeds = pad_sequence(text_embeds_list, batch_first=True).float()
            image_embeds = image_embeds.float()

            if len(image_files) < chunk_size:
                text_embeds, image_embeds = view_selector(
                    image_embeds.to(model.device), padded_text_embeds)
                scores = F.cosine_similarity(text_embeds, image_embeds)
            else:
                text_embeds_chunks = split_list(padded_text_embeds)
                image_embeds_chunks = split_list(image_embeds)
                scores_list = []
                for te, ie in zip(text_embeds_chunks, image_embeds_chunks):
                    te, ie = view_selector(ie.to(model.device), te)
                    scores_list.append(F.cosine_similarity(te, ie))
                scores = torch.cat(scores_list)

            paired = list(zip(scores, image_files))
            paired.sort(key=lambda x: x[0], reverse=True)
            sorted_scores, sorted_image_files = zip(*paired)
            qs_id = str(qs_id_list[0])
            output_dict[qs_id] = list(sorted_image_files)

    json.dump(output_dict, open(save_path, "w"))


def viewNMS(list_of_images, neighbour_df, num_images, distance_threshold=0.5):
    selected_images = []
    remaining_images = list_of_images.copy()

    while len(selected_images) < num_images and remaining_images:
        current_image = remaining_images.pop(0)
        selected_images.append(current_image)
        sorted_distances = neighbour_df.loc[current_image].sort_values()
        filtered_images = sorted_distances[sorted_distances < distance_threshold]
        neighbours_to_remove = set(filtered_images.index.tolist())
        remaining_images = [img for img in remaining_images
                            if img not in neighbours_to_remove]

    return selected_images


# ── Main evaluation ─────────────────────────────────────────────────────────

def eval_model(args, shard_id=0, num_shards=1):
    disable_torch_init()
    tag = f"[shard {shard_id}]" if num_shards > 1 else ""

    # ── Load LVLM ───────────────────────────────────────────────────────────
    model_path = os.path.expanduser(args.LVLM_ckpt)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        attn_implementation="sdpa")

    # ── View selector is loaded lazily (only when rank file is missing) ────
    view_selector = None

    # ── Initialise SAM2 for on-the-fly SoM generation ──────────────────────
    sam_device = str(model.device)
    print(f"{tag} Loading SAM2 predictor ({args.sam_model}) on {sam_device} …")
    sam_predictor = make_sam2_predictor(args.sam_model, device=sam_device)

    som_output_folder = args.som_output_folder
    os.makedirs(som_output_folder, exist_ok=True)
    print(f"{tag} SoM images will be saved to {som_output_folder}")

    # ── Per-mode evaluation loop ────────────────────────────────────────────
    test_mode = (["test_w_obj", "test_wo_obj"]
                 if args.dataset == "ScanQA" else ["test"])

    for mode in test_mode:
        save_rank_file = args.ranked_view_file.format(args.dataset, mode)

        # Only shard 0 computes rankings if missing; other shards wait.
        if not os.path.exists(save_rank_file):
            if shard_id == 0:
                if view_selector is None:
                    view_selector = ViewSelector().to(model.device)
                    ckpt_file = args.pretrained_view_selector_ckpt.format(args.dataset)
                    view_selector.load_state_dict(
                        torch.load(ckpt_file, map_location=model.device)["model"])
                    print(f"{tag} Loaded view_selector from {ckpt_file}")
                pair_dataset_test = ViewLabelDataset(args, mode=mode)
                pair_dataloader_test = DataLoader(
                    pair_dataset_test, batch_size=1, shuffle=False,
                    collate_fn=custom_collate_fn)
                print(f"{tag} Ranking images by view selector for {args.dataset}_{mode}")
                ranking_views(pair_dataloader_test, tokenizer, model,
                              view_selector, save_path=save_rank_file)
            else:
                print(f"{tag} Waiting for rank file from shard 0 …")
                while not os.path.exists(save_rank_file):
                    time.sleep(5)
                time.sleep(1)

        raw_ranking = json.load(open(save_rank_file))
        if "view" in raw_ranking:
            image_file_list = raw_ranking["view"]
        else:
            image_file_list = raw_ranking

        if args.dataset == "ScanQA":
            qa_data = get_scanqa(args, mode=mode)
        elif args.dataset == "SQA":
            qa_data = get_sqa(args, mode=mode)

        # ── Shard the datapoints across GPUs ─────────────────────────────
        full_len = len(qa_data)
        qa_data = [item for i, item in enumerate(qa_data)
                   if i % num_shards == shard_id]

        answers_file = args.som_answers_file.format(args.dataset, mode)
        if num_shards > 1:
            base, ext = os.path.splitext(answers_file)
            answers_file = f"{base}_shard{shard_id}{ext}"

        answers = []
        print(f"{tag} Evaluating {len(qa_data)}/{full_len} items for {mode}")

        for line in tqdm(qa_data, desc=f"shard {shard_id}" if num_shards > 1 else mode):
            scene_id = line["scene_id"]

            # ── View distance for NMS ───────────────────────────────────────
            os.makedirs(args.view_distance_folder, exist_ok=True)

            view_distance_file = os.path.join(
                args.view_distance_folder, f"{scene_id}.csv")
            if os.path.exists(view_distance_file):
                distance_df = pd.read_csv(view_distance_file, index_col=0)
            else:
                distance_df = calculate_view_distance(scene_id, args)
                distance_df.to_csv(view_distance_file)

            question_id = str(line["question_id"])
            image_files = image_file_list[question_id]
            image_files = viewNMS(image_files, distance_df,
                                  num_images=args.input_views)
            image_files = (image_files if len(image_files) < args.input_views
                           else image_files[:args.input_views])
            num_image = len(image_files)

            # ── Build prompt ────────────────────────────────────────────────
            line["from"] = "human"
            question = (line["question"] if args.dataset == "ScanQA"
                        else line["situation"] + line["question"])
            line["value"] = "<image>" * num_image + question
            input_ids = preprocess_qwen(
                [line, {"from": "gpt", "value": None}],
                tokenizer, has_image=True,
            ).to(model.device)

            # ── Generate SoM images on the fly for selected views ───────────
            image_tensors = []
            for image_file in image_files:
                som_img = get_or_create_som(
                    scene_id, image_file, args.image_folder,
                    som_output_folder, sam_predictor,
                )
                image_tensor = image_processor.preprocess(
                    som_img, return_tensors="pt")["pixel_values"]
                image_tensors.append(image_tensor.half().to(model.device))

            # ── Generate answer ─────────────────────────────────────────────
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip().lower()
            ans = outputs.split("\n")[0]

            answers.append({
                "scene_id": scene_id,
                "question_id": question_id,
                "answer_top10": [ans for _ in range(10)],
            })

        json.dump(answers, open(answers_file, "w"))
        print(f"{tag} Saved {len(answers)} answers → {answers_file}")


# ── Multi-GPU helpers ───────────────────────────────────────────────────────

def _gpu_worker(gpu_id, num_gpus, args):
    """Spawn target: restrict CUDA visibility then run eval_model on one shard."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    eval_model(args, shard_id=gpu_id, num_shards=num_gpus)


def merge_answer_shards(args, num_shards):
    """Combine per-shard answer files into the final answers file."""
    test_mode = (["test_w_obj", "test_wo_obj"]
                 if args.dataset == "ScanQA" else ["test"])
    for mode in test_mode:
        answers_file = args.som_answers_file.format(args.dataset, mode)
        base, ext = os.path.splitext(answers_file)
        merged = []
        for s in range(num_shards):
            shard_path = f"{base}_shard{s}{ext}"
            with open(shard_path) as f:
                merged.extend(json.load(f))
            os.remove(shard_path)
        merged.sort(key=lambda x: x["question_id"])
        with open(answers_file, "w") as f:
            json.dump(merged, f)
        print(f"Merged {len(merged)} answers → {answers_file}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cdViews QA inference with SoM-prompted views")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg_file", type=str, default="../cfgs/QA.yaml")

    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset from config (ScanQA or SQA)")
    parser.add_argument(
        "--sam-model", type=str, default="sam2.1_l.pt",
        help="SAM2 model weights for SoM generation")
    parser.add_argument(
        "--som-output-folder", type=str,
        default="../data/qa/scannetv2/som_views",
        help="Directory to save generated SoM images")
    parser.add_argument(
        "--num-gpus", type=int, default=1, choices=[1, 2],
        help="Number of GPUs: 1 = single-GPU (default), "
             "2 = data-parallel split across cuda:0 and cuda:1")

    args = parser.parse_args()
    dataset_override = args.dataset
    args = load_and_update(args)

    if dataset_override is not None:
        args.dataset = dataset_override

    if not hasattr(args, "som_answers_file"):
        args.som_answers_file = args.answers_file.replace(
            "_answers.json", "_som_answers.json"
        )
        if args.som_answers_file == args.answers_file:
            args.som_answers_file = "../data/qa/{}_{}_som_answers.json"

    if args.num_gpus == 1:
        eval_model(args)
    else:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
        processes = []
        for gpu_id in range(args.num_gpus):
            p = mp.Process(target=_gpu_worker, args=(gpu_id, args.num_gpus, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        merge_answer_shards(args, args.num_gpus)
