"""
Novel View Generator for ScanNet Scenes
========================================

Analyses camera extrinsics for a scene, identifies angular coverage gaps,
and uses the Qwen Image-Edit Camera-Control model to synthesise novel views
that fill those gaps.

Public API
----------
    generate_novel_views_for_scene(scene_dir, max_views=10, pipe=None)
    load_qwen_pipeline()          # one-time model load
    get_coverage_gaps(scene_dir)   # analysis only, no generation

The generated images are saved under:
    <scene_dir>/novel-color/<angle>.jpg

Subsequent calls skip already-generated files (idempotent).
"""

import os
import math
import glob
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image


# ---------------------------------------------------------------------------
#  Camera-extrinsic analysis
# ---------------------------------------------------------------------------

def load_poses(pose_dir: str) -> Dict[int, np.ndarray]:
    """Load all valid 4×4 camera-to-world poses from a directory of .txt files.

    Returns {frame_id: 4×4 ndarray}.
    """
    poses = {}
    for f in sorted(Path(pose_dir).glob("*.txt")):
        frame_id = int(f.stem)
        try:
            mat = np.loadtxt(f)
        except Exception:
            continue
        if mat.shape == (4, 4) and np.isfinite(mat).all():
            poses[frame_id] = mat
    return poses


def extract_viewing_directions(
    poses: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (frame_ids, azimuths, elevations, positions) for all poses.

    Convention: ScanNet poses are camera-to-world in OpenCV convention
    (camera looks along +Z).  World frame has Z-up.
    """
    frame_ids, azimuths, elevations, positions = [], [], [], []

    for fid in sorted(poses.keys()):
        T = poses[fid]
        R = T[:3, :3]
        t = T[:3, 3]

        # Viewing direction = camera +Z in world
        d = R @ np.array([0.0, 0.0, 1.0])
        d = d / (np.linalg.norm(d) + 1e-12)

        az = math.degrees(math.atan2(d[1], d[0])) % 360
        el = math.degrees(math.asin(np.clip(d[2], -1.0, 1.0)))

        frame_ids.append(fid)
        azimuths.append(az)
        elevations.append(el)
        positions.append(t)

    return (
        np.array(frame_ids),
        np.array(azimuths),
        np.array(elevations),
        np.array(positions),
    )


def _angular_distance(a: float, b: float) -> float:
    """Signed shortest angular distance from *a* to *b* (degrees), in (-180, 180]."""
    d = (b - a) % 360
    return d - 360 if d > 180 else d


def find_coverage_gaps(
    azimuths: np.ndarray,
    bin_deg: float = 30.0,
    max_gaps: int = 10,
) -> List[float]:
    """Return up to *max_gaps* azimuth centres (°) sorted by ascending count.

    Bins with zero frames come first, then the smallest-count bins, etc.
    """
    n_bins = int(360 / bin_deg)
    edges = np.linspace(0, 360, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(azimuths, bins=edges)

    # Sort bins by frame count (ascending), take up to max_gaps
    order = np.argsort(counts)
    gap_centres = centres[order][:max_gaps].tolist()
    return gap_centres


def pick_source_frame(
    target_az: float,
    frame_ids: np.ndarray,
    azimuths: np.ndarray,
) -> Tuple[int, float, float]:
    """Find the frame closest to *target_az*.

    Returns (frame_id, source_azimuth, signed_delta_degrees).
    """
    deltas = np.array([_angular_distance(az, target_az) for az in azimuths])
    best = int(np.argmin(np.abs(deltas)))
    return int(frame_ids[best]), float(azimuths[best]), float(deltas[best])


def get_coverage_gaps(scene_dir: str, bin_deg: float = 30.0, max_gaps: int = 10):
    """High-level: analyse a scene and return gap info without generating.

    Returns list of dicts:
        [{"target_az", "source_fid", "source_az", "rotate_deg"}, ...]
    """
    pose_dir = os.path.join(scene_dir, "pose")
    poses = load_poses(pose_dir)
    if len(poses) < 3:
        return []

    fids, azs, els, _ = extract_viewing_directions(poses)
    gap_centres = find_coverage_gaps(azs, bin_deg=bin_deg, max_gaps=max_gaps)

    results = []
    for target_az in gap_centres:
        src_fid, src_az, delta = pick_source_frame(target_az, fids, azs)
        results.append(
            {
                "target_az": round(target_az, 1),
                "source_fid": src_fid,
                "source_az": round(src_az, 1),
                "rotate_deg": round(delta, 1),
            }
        )
    return results


# ---------------------------------------------------------------------------
#  Qwen camera-control pipeline  (lazy-loaded)
# ---------------------------------------------------------------------------

_pipe = None  # module-level singleton


def load_qwen_pipeline():
    """Load (once) and return the Qwen camera-control diffusion pipeline."""
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    # ── Cache config (must be set before any HF download) ──
    HF_ROOT = "/ocean/projects/cis250206p/aanugu"
    _cache = {
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
    }
    for k, v in _cache.items():
        os.environ[k] = v
        if "/" in v:
            Path(v).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16

    print("[novel_view_generator] Loading distilled transformer …")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print("[novel_view_generator] Loading pipeline …")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")

    print("[novel_view_generator] Loading camera-angle LoRA …")
    pipe.load_lora_weights(
        "dx8152/Qwen-Edit-2509-Multiple-angles",
        weight_name="\u955c\u5934\u8f6c\u6362.safetensors",
        adapter_name="angles",
    )
    pipe.set_adapters(["angles"], adapter_weights=[1.0])
    pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
    pipe.unload_lora_weights()
    print("[novel_view_generator] Pipeline ready.")

    _pipe = pipe
    return _pipe


# ---------------------------------------------------------------------------
#  Prompt building & single-image generation
# ---------------------------------------------------------------------------

def _build_camera_prompt(rotate_deg: float) -> str:
    """Bilingual (Chinese + English) camera rotation prompt."""
    if abs(rotate_deg) < 1:
        return ""
    direction = "left" if rotate_deg > 0 else "right"
    cn_dir = "左" if rotate_deg > 0 else "右"
    return (
        f"将镜头向{cn_dir}旋转{abs(rotate_deg):.0f}度 "
        f"Rotate the camera {abs(rotate_deg):.0f} degrees to the {direction}."
    )


def _generate_single_view(
    pipe,
    image: Image.Image,
    rotate_deg: float,
    seed: int = 42,
    steps: int = 4,
) -> Image.Image:
    """Run one hop of Qwen camera control (clamped to ±90°)."""
    import torch

    clamped = max(-90.0, min(90.0, rotate_deg))
    prompt = _build_camera_prompt(clamped)
    if not prompt:
        return image

    w, h = image.size
    if w > h:
        out_w, out_h = 1024, int(1024 * h / w)
    else:
        out_h, out_w = 1024, int(1024 * w / h)
    out_w = (out_w // 8) * 8
    out_h = (out_h // 8) * 8

    gen = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        image=[image.convert("RGB")],
        prompt=prompt,
        negative_prompt=" ",
        height=out_h,
        width=out_w,
        num_inference_steps=steps,
        generator=gen,
        true_cfg_scale=1.0,
        num_images_per_prompt=1,
    ).images[0]
    return result


def generate_view_for_angle(
    pipe,
    source_image: Image.Image,
    total_rotate_deg: float,
    seed: int = 42,
) -> Image.Image:
    """Multi-hop generation if |rotation| > 90°."""
    remaining = total_rotate_deg
    current = source_image
    hop = 0
    while abs(remaining) > 5:
        hop += 1
        step = max(-90.0, min(90.0, remaining))
        current = _generate_single_view(pipe, current, step, seed=seed)
        remaining -= step
    return current


# ---------------------------------------------------------------------------
#  High-level: generate all novel views for a scene
# ---------------------------------------------------------------------------

def generate_novel_views_for_scene(
    scene_dir: str,
    max_views: int = 10,
    bin_deg: float = 30.0,
    pipe=None,
    seed: int = 42,
) -> List[str]:
    """Generate novel views filling coverage gaps for one ScanNet scene.

    Saves images to ``<scene_dir>/novel-color/<target_az>.jpg``.
    Skips any that already exist on disk.

    Parameters
    ----------
    scene_dir : str
        e.g. ``data/qa/scannetv2/frames_square/scene0000_00``
    max_views : int
        Maximum number of novel views to generate (≤ 12).
    bin_deg : float
        Azimuth histogram bin width in degrees.
    pipe : diffusion pipeline (or None to auto-load)
    seed : int
        Reproducibility seed.

    Returns
    -------
    list of str : paths to novel-view images (new + already-existing).
    """
    novel_dir = os.path.join(scene_dir, "novel-color")
    os.makedirs(novel_dir, exist_ok=True)

    # 1) Analyse coverage
    gaps = get_coverage_gaps(scene_dir, bin_deg=bin_deg, max_gaps=max_views)
    if not gaps:
        print(f"  [skip] No valid poses for {os.path.basename(scene_dir)}")
        return []

    # 2) Generate (or reuse) each novel view
    if pipe is None:
        pipe = load_qwen_pipeline()

    color_dir = os.path.join(scene_dir, "color")
    generated_paths = []
    scene_name = os.path.basename(scene_dir)

    from tqdm import tqdm as _tqdm
    pbar = _tqdm(gaps, desc=f"  {scene_name}", unit="view", leave=False)
    for gap in pbar:
        target_az = gap["target_az"]
        src_fid = gap["source_fid"]
        rotate = gap["rotate_deg"]
        pbar.set_postfix(az=f"{target_az:.0f}°", delta=f"{rotate:+.0f}°", refresh=False)

        out_path = os.path.join(novel_dir, f"{target_az:.0f}.jpg")
        if os.path.exists(out_path):
            generated_paths.append(out_path)
            continue  # already generated

        src_path = os.path.join(color_dir, f"{src_fid}.jpg")
        if not os.path.exists(src_path):
            continue

        src_img = Image.open(src_path).convert("RGB")
        novel_img = generate_view_for_angle(pipe, src_img, rotate, seed=seed)
        novel_img.save(out_path, quality=95)
        generated_paths.append(out_path)

    return generated_paths


# ---------------------------------------------------------------------------
#  CLI for standalone use
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate novel views for a ScanNet scene")
    parser.add_argument("scene_dir", type=str,
                        help="Path to scene directory (e.g. data/…/scene0000_00)")
    parser.add_argument("--max_views", type=int, default=10)
    parser.add_argument("--bin_deg", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis_only", action="store_true",
                        help="Only print coverage gaps, don't generate images")
    args = parser.parse_args()

    if args.analysis_only:
        gaps = get_coverage_gaps(args.scene_dir, bin_deg=args.bin_deg,
                                 max_gaps=args.max_views)
        for g in gaps:
            print(g)
    else:
        paths = generate_novel_views_for_scene(
            args.scene_dir,
            max_views=args.max_views,
            bin_deg=args.bin_deg,
            seed=args.seed,
        )
        print(f"\n{len(paths)} novel views ready.")
