"""Qwen3-VL inference on SoM-prompted views for cdViews SQA evaluation.

Uses pre-existing SoM images (data/qa/scannetv2/som_views/) and
pre-computed view rankings.  Applies viewNMS then runs Qwen3-VL for QA.

Usage (from scripts/):
    python qwen_som_inference.py --cfg_file ../cfgs/QA.yaml
    python qwen_som_inference.py --cfg_file ../cfgs/QA.yaml --qwen-model-id Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_CACHE", "/ocean/projects/cis250206p/aanugu/hf-cache/hub")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CDVIEWS_DIR = os.path.join(REPO_ROOT, "cdviews")
for _p in (REPO_ROOT, CDVIEWS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from qa_utils import get_sqa, load_and_update
from view_distance_calculation import calculate_view_distance


def viewNMS(list_of_images, neighbour_df, num_images, distance_threshold=0.5):
    """Select diverse views via non-maximum suppression on view distances."""
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


def eval_model(args):
    model_id = args.qwen_model_id
    print(f"Loading model: {model_id}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    som_views_folder = os.path.abspath(args.som_views_folder)

    test_mode = ["test"]
    for mode in test_mode:
        save_rank_file = args.ranked_view_file.format(args.dataset, mode)
        if not os.path.exists(save_rank_file):
            raise FileNotFoundError(
                f"Ranked view file not found: {save_rank_file}. "
                "Run the view selector pipeline first.")

        raw_ranking = json.load(open(save_rank_file))
        image_file_list = raw_ranking.get("view", raw_ranking)

        qa_data = get_sqa(args, mode=mode)

        answers_file = args.qwen_som_answers_file.format(args.dataset, mode)
        answers = []
        print(f"Evaluating {len(qa_data)} items for {mode}")

        for line in tqdm(qa_data, desc=mode):
            scene_id = line["scene_id"]
            question_id = str(line["question_id"])

            # ── View distance + NMS ──────────────────────────────────────
            os.makedirs(args.view_distance_folder, exist_ok=True)
            view_distance_file = os.path.join(
                args.view_distance_folder, f"{scene_id}.csv")
            if os.path.exists(view_distance_file):
                distance_df = pd.read_csv(view_distance_file, index_col=0)
            else:
                distance_df = calculate_view_distance(scene_id, args)
                distance_df.to_csv(view_distance_file)

            image_files = image_file_list[question_id]
            image_files = viewNMS(image_files, distance_df,
                                  num_images=args.input_views)
            image_files = image_files[:args.input_views]

            # ── Load pre-existing SoM images ─────────────────────────────
            som_images = []
            for f in image_files:
                p = os.path.join(som_views_folder, scene_id, f)
                if os.path.exists(p):
                    som_images.append(Image.open(p).convert("RGB"))

            if not som_images:
                answers.append({
                    "scene_id": scene_id,
                    "question_id": question_id,
                    "answer_top10": ["" for _ in range(10)],
                })
                continue

            # ── Build Qwen3-VL prompt ────────────────────────────────────
            question = line["situation"] + " " + line["question"]
            content = [{"type": "image"} for _ in som_images]
            content.append({
                "type": "text",
                "text": (question
                         + "\nAnswer the question using a single word or phrase."),
            })
            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(
                text=[text],
                images=som_images,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            # ── Generate ─────────────────────────────────────────────────
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                )

            generated_ids = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, output_ids)
            ]
            ans = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip().lower().split("\n")[0]

            answers.append({
                "scene_id": scene_id,
                "question_id": question_id,
                "answer_top10": [ans for _ in range(10)],
            })

        with open(answers_file, "w") as f:
            json.dump(answers, f)
        print(f"Saved {len(answers)} answers -> {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL inference with SoM-prompted views on SQA")
    parser.add_argument("--cfg_file", type=str, default="../cfgs/QA.yaml")
    parser.add_argument("--dataset", type=str, default="SQA")
    parser.add_argument(
        "--qwen-model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID for Qwen VL")
    parser.add_argument(
        "--som-views-folder", type=str,
        default="../data/qa/scannetv2/som_views",
        help="Folder with pre-generated SoM images")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()

    dataset_override = args.dataset
    qwen_model_id = args.qwen_model_id
    som_views_folder = args.som_views_folder
    temp = args.temperature
    top_p = args.top_p
    num_beams = args.num_beams

    args = load_and_update(args)

    args.dataset = dataset_override
    args.qwen_model_id = qwen_model_id
    args.som_views_folder = som_views_folder
    args.temperature = temp
    args.top_p = top_p
    args.num_beams = num_beams

    if not hasattr(args, "qwen_som_answers_file"):
        args.qwen_som_answers_file = "../data/qa/{}_{}_qwen_som_answers.json"

    eval_model(args)
