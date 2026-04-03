#!/usr/bin/env python3
"""
Batch 3D Object Localization for ScanNet scenes.

For every scene referenced in the SQA dataset, runs the two-phase pipeline:
  Phase 1 — CLIP + SAM2 auto-discovers which object categories are present
  Phase 2 — YOLO-World (with discovered vocab) + SAM2 detects & segments,
            depth + intrinsics + extrinsics back-project to 3D,
            DBSCAN fuses multi-view detections into unique object instances

Outputs one JSON per scene and a combined JSON with all scenes.

Usage:
    python scripts/localize_objects_3d.py                    # all scenes
    python scripts/localize_objects_3d.py --max-scenes 5     # first 5 only
    python scripts/localize_objects_3d.py --resume           # skip already-done
"""

import argparse
import glob
import json
import os
import ssl
import sys
import time
from collections import defaultdict
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from ultralytics import YOLOWorld
from ultralytics.models.sam import SAM2Predictor

ssl._create_default_https_context = ssl._create_unverified_context

# ─── Config ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
IMAGE_FOLDER = ROOT / "data" / "qa" / "scannetv2" / "frames_square"
SQA_TEST = ROOT / "data" / "qa" / "SQA" / "SQA_test.json"
OUTPUT_DIR = ROOT / "object_3d_maps"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

N_KEYFRAMES = 25
N_DISCOVERY_FRAMES = 5
DET_CONF = 0.10
MIN_MASK_AREA = 150
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2
SINGLETON_MIN_CONF = 0.25
CLIP_THRESHOLD = 0.20

CANDIDATE_CLASSES = [
    "chair", "office chair", "armchair", "table", "dining table", "coffee table",
    "sofa", "couch", "loveseat", "desk", "bed", "nightstand", "dresser",
    "counter", "countertop", "stool", "bar stool", "bench", "ottoman", "futon",
    "recliner", "rocking chair",
    "shelf", "bookshelf", "cabinet", "drawer", "wardrobe", "closet", "cupboard",
    "filing cabinet", "storage bin", "chest",
    "door", "window", "curtain", "blinds", "shutter",
    "lamp", "floor lamp", "table lamp", "ceiling light", "chandelier",
    "light fixture", "sconce",
    "monitor", "computer monitor", "tv", "television", "screen", "keyboard",
    "computer keyboard", "laptop", "printer", "speaker", "projector",
    "refrigerator", "fridge", "microwave", "oven", "stove", "dishwasher",
    "toaster", "coffee maker",
    "sink", "toilet", "bathtub", "shower", "shower curtain",
    "mirror", "picture", "picture frame", "painting", "clock", "poster",
    "whiteboard", "bulletin board", "calendar",
    "pillow", "cushion", "blanket", "towel", "rug", "carpet", "mat",
    "curtain", "tablecloth",
    "bottle", "cup", "mug", "bowl", "plate", "glass", "jar", "can",
    "pitcher", "vase",
    "bag", "backpack", "suitcase", "box", "cardboard box", "bin",
    "trash can", "recycling bin", "basket", "crate",
    "book", "paper", "notebook", "magazine", "newspaper",
    "phone", "cell phone", "remote", "remote control",
    "pen", "pencil", "scissors", "tape", "stapler",
    "fan", "heater", "radiator", "air conditioner", "humidifier",
    "vacuum cleaner", "iron",
    "plant", "potted plant", "flower", "flower pot", "tree",
    "piano", "keyboard instrument", "guitar", "violin", "drum",
    "bookcase", "shoe rack", "coat rack", "umbrella stand", "wine rack",
    "tv stand", "entertainment center", "sideboard",
    "umbrella", "hat", "helmet", "shoe", "boot", "coat", "jacket",
    "toy", "stuffed animal", "ball", "board game",
    "candle", "tissue box", "soap dispenser", "toothbrush holder",
    "fire extinguisher", "smoke detector",
    "column", "pillar", "beam", "pipe", "vent", "electrical outlet",
    "light switch", "thermostat", "doorknob",
    "stairs", "staircase", "railing", "banister",
    "wall", "floor", "ceiling",
]


# ─── ScanNet data loaders ────────────────────────────────────────────

def load_intrinsics(scene_dir):
    K = np.loadtxt(os.path.join(scene_dir, "intrinsic_depth.txt"))
    sample_depth = sorted(glob.glob(os.path.join(scene_dir, "depth", "*.png")))[0]
    dh, dw = np.array(Image.open(sample_depth)).shape[:2]
    if K[0, 2] > dw:
        scale = dw / (K[0, 2] * 2)
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
    return K


def load_pose(scene_dir, frame_id):
    pose = np.loadtxt(os.path.join(scene_dir, "pose", f"{frame_id}.txt"))
    if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
        return None
    return pose


def load_depth(scene_dir, frame_id):
    path = os.path.join(scene_dir, "depth", f"{frame_id}.png")
    return np.array(Image.open(path)).astype(np.float32) / 1000.0


def get_frame_ids(scene_dir):
    files = glob.glob(os.path.join(scene_dir, "color", "*.jpg"))
    return sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in files)


def select_keyframes(frame_ids, n):
    if len(frame_ids) <= n:
        return frame_ids
    indices = np.linspace(0, len(frame_ids) - 1, n, dtype=int)
    return [frame_ids[i] for i in indices]


# ─── Back-projection ─────────────────────────────────────────────────

def backproject_mask(mask, depth, K, cam2world, max_depth=5.0, subsample=500):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vs, us = np.where(mask)
    depths = depth[vs, us]
    valid = (depths > 0.01) & (depths < max_depth)
    us, vs, depths = us[valid], vs[valid], depths[valid]
    if len(us) < 10:
        return None
    if len(us) > subsample:
        idx = np.random.choice(len(us), subsample, replace=False)
        us, vs, depths = us[idx], vs[idx], depths[idx]
    X = (us - cx) * depths / fx
    Y = (vs - cy) * depths / fy
    pts_cam = np.stack([X, Y, depths, np.ones_like(depths)], axis=1)
    return (cam2world @ pts_cam.T).T[:, :3]


def mask_to_3d_centroid(mask, depth, K, cam2world):
    pts = backproject_mask(mask, depth, K, cam2world)
    return np.median(pts, axis=0) if pts is not None else None


# ─── CLIP discovery ──────────────────────────────────────────────────

def discover_scene_classes(
    scene_dir, sam_predictor, clip_model, clip_preprocess, device=DEVICE,
):
    frame_ids = get_frame_ids(scene_dir)
    sample_ids = select_keyframes(frame_ids, N_DISCOVERY_FRAMES)

    text_tokens = clip.tokenize(CANDIDATE_CLASSES).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    discovered = defaultdict(float)

    for fid in sample_ids:
        color_path = os.path.join(scene_dir, "color", f"{fid}.jpg")
        img_pil = Image.open(color_path).convert("RGB")
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        min_area = 0.005 * h * w

        results = sam_predictor(
            source=color_path, points_stride=32, crop_n_layers=0,
            conf_thres=0.88, stability_score_thresh=0.92,
        )
        if not results or results[0].masks is None:
            continue

        masks = results[0].masks.data.cpu().numpy()
        for m in masks:
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
            mask_bool = m.astype(bool)
            if mask_bool.sum() < min_area:
                continue

            ys, xs = np.where(mask_bool)
            pad = 5
            y1, x1 = max(0, ys.min() - pad), max(0, xs.min() - pad)
            y2, x2 = min(h, ys.max() + pad), min(w, xs.max() + pad)
            crop = img_pil.crop((x1, y1, x2, y2))

            crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = clip_model.encode_image(crop_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                probs = (img_feat @ text_feats.T).squeeze(0).softmax(dim=0).cpu().numpy()

            best_idx = probs.argmax()
            if probs[best_idx] >= CLIP_THRESHOLD:
                name = CANDIDATE_CLASSES[best_idx]
                discovered[name] = max(discovered[name], float(probs[best_idx]))

    return sorted(discovered.keys(), key=lambda k: discovered[k], reverse=True)


# ─── Per-frame detection ─────────────────────────────────────────────

def detect_and_locate_3d(scene_dir, frame_id, yolo_model, sam_predictor, K):
    color_path = os.path.join(scene_dir, "color", f"{frame_id}.jpg")
    cam2world = load_pose(scene_dir, frame_id)
    if cam2world is None:
        return []

    depth = load_depth(scene_dir, frame_id)

    det_results = yolo_model(color_path, conf=DET_CONF, verbose=False)
    boxes = det_results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    bboxes_np = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = [yolo_model.names[c] for c in class_ids]

    sam_results = sam_predictor(source=color_path, bboxes=bboxes_np)
    if not sam_results or sam_results[0].masks is None:
        return []

    masks_raw = sam_results[0].masks.data.cpu().numpy()
    h, w = depth.shape
    detections = []

    for i, (name, conf) in enumerate(zip(names, confs)):
        if i >= len(masks_raw):
            break
        mask = masks_raw[i]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        if mask.sum() < MIN_MASK_AREA:
            continue
        centroid = mask_to_3d_centroid(mask, depth, K, cam2world)
        if centroid is None:
            continue
        detections.append({
            "name": name,
            "confidence": float(conf),
            "centroid_3d": centroid.tolist(),
            "frame_id": int(frame_id),
        })

    return detections


# ─── Multi-view fusion ───────────────────────────────────────────────

def fuse_detections(all_detections):
    by_class = defaultdict(list)
    for det in all_detections:
        by_class[det["name"]].append(det)

    objects = []
    for name, dets in by_class.items():
        centroids = np.array([d["centroid_3d"] for d in dets])
        confs = np.array([d["confidence"] for d in dets])

        if len(centroids) == 1:
            if confs[0] >= SINGLETON_MIN_CONF:
                objects.append({
                    "name": name,
                    "position_3d": centroids[0].tolist(),
                    "n_views": 1,
                    "avg_confidence": round(float(confs[0]), 4),
                })
            continue

        clustering = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=min(DBSCAN_MIN_SAMPLES, len(centroids)),
        ).fit(centroids)

        for label in set(clustering.labels_):
            if label == -1:
                continue
            mask = clustering.labels_ == label
            objects.append({
                "name": name,
                "position_3d": centroids[mask].mean(axis=0).tolist(),
                "n_views": int(mask.sum()),
                "avg_confidence": round(float(confs[mask].mean()), 4),
            })

    objects.sort(key=lambda o: (-o["n_views"], -o["avg_confidence"]))
    return objects


# ─── Workaround for ultralytics set_classes device bug ────────────────
# clip.tokenize() returns CPU tensors, but after GPU inference the cached
# CLIP model's nn.Embedding is on CUDA. index_select(CUDA, CPU) crashes.
# Fix: move cached CLIP to CPU before encoding, then put everything back.

def safe_set_classes(yolo_model, classes):
    clip_cache = getattr(yolo_model.model, "clip_model", None)
    if clip_cache is not None:
        clip_cache.cpu()
    yolo_model.set_classes(classes)
    yolo_model.model.txt_feats = yolo_model.model.txt_feats.to(DEVICE)
    if clip_cache is not None:
        clip_cache.to(DEVICE)


# ─── Process one scene ───────────────────────────────────────────────

def process_scene(scene_id, yolo_model, sam_predictor, clip_mdl, clip_prep):
    scene_dir = str(IMAGE_FOLDER / scene_id)
    K = load_intrinsics(scene_dir)
    frame_ids = get_frame_ids(scene_dir)

    # Phase 1: discover classes
    scene_classes = discover_scene_classes(
        scene_dir, sam_predictor, clip_mdl, clip_prep,
    )
    if not scene_classes:
        scene_classes = CANDIDATE_CLASSES
    safe_set_classes(yolo_model, scene_classes)

    # Phase 2: detect on keyframes
    keyframes = select_keyframes(frame_ids, N_KEYFRAMES)
    all_detections = []
    for fid in keyframes:
        dets = detect_and_locate_3d(scene_dir, fid, yolo_model, sam_predictor, K)
        all_detections.extend(dets)

    objects = fuse_detections(all_detections)

    return {
        "scene_id": scene_id,
        "n_frames_total": len(frame_ids),
        "n_keyframes_used": len(keyframes),
        "n_discovery_classes": len(scene_classes),
        "discovered_classes": scene_classes,
        "n_raw_detections": len(all_detections),
        "n_objects": len(objects),
        "objects": objects,
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch 3D object localization")
    parser.add_argument("--sqa-file", type=str, default=str(SQA_TEST),
                        help="Path to SQA JSON (determines which scenes to process)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Process only the first N scenes (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip scenes that already have output JSONs")
    parser.add_argument("--sam-model", type=str, default="sam2.1_s.pt",
                        help="SAM2 model checkpoint name")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique scenes
    with open(args.sqa_file) as f:
        sqa_data = json.load(f)
    scene_ids = sorted({s["scene_id"] for s in sqa_data})
    print(f"Found {len(scene_ids)} unique scenes in {args.sqa_file}")

    if args.max_scenes:
        scene_ids = scene_ids[:args.max_scenes]
        print(f"  → limited to first {args.max_scenes}")

    if args.resume:
        already_done = {
            p.stem for p in output_dir.glob("*.json") if p.stem != "all_scenes"
        }
        before = len(scene_ids)
        scene_ids = [s for s in scene_ids if s not in already_done]
        print(f"  → resuming: {before - len(scene_ids)} already done, "
              f"{len(scene_ids)} remaining")

    if not scene_ids:
        print("Nothing to do.")
        return

    # Load models
    print(f"\nLoading models on {DEVICE}...")
    sam_predictor = SAM2Predictor(overrides=dict(
        task="segment", mode="predict", imgsz=1024,
        model=args.sam_model, device=DEVICE, verbose=False, save=False,
    ))
    clip_mdl, clip_prep = clip.load("ViT-B/32", device=DEVICE)
    yolo_model = YOLOWorld("yolov8x-worldv2.pt")
    print("Models loaded.\n")

    # Process each scene
    combined = {}
    failed = []

    for scene_id in tqdm(scene_ids, desc="Scenes", unit="scene"):
        scene_dir = IMAGE_FOLDER / scene_id
        if not scene_dir.exists():
            tqdm.write(f"  SKIP {scene_id}: directory not found")
            failed.append({"scene_id": scene_id, "error": "directory not found"})
            continue

        try:
            t0 = time.time()
            result = process_scene(
                scene_id, yolo_model, sam_predictor, clip_mdl, clip_prep,
            )
            elapsed = time.time() - t0
            result["elapsed_seconds"] = round(elapsed, 1)

            # Save per-scene JSON
            out_path = output_dir / f"{scene_id}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            combined[scene_id] = result
            tqdm.write(
                f"  {scene_id}: {result['n_objects']} objects, "
                f"{result['n_raw_detections']} raw dets, "
                f"{elapsed:.1f}s"
            )

        except Exception as e:
            tqdm.write(f"  FAIL {scene_id}: {e}")
            failed.append({"scene_id": scene_id, "error": str(e)})

    # Save combined output
    combined_path = output_dir / "all_scenes.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done: {len(combined)} scenes processed")
    print(f"  Failed: {len(failed)}")
    print(f"  Per-scene JSONs: {output_dir}/")
    print(f"  Combined JSON:   {combined_path}")
    total_objects = sum(r["n_objects"] for r in combined.values())
    print(f"  Total objects:   {total_objects}")
    if failed:
        print(f"\n  Failed scenes:")
        for f_info in failed:
            print(f"    {f_info['scene_id']}: {f_info['error']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
