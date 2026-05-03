#!/usr/bin/env python3
"""
Resolution analysis for obfuscated images.

This script analyzes the "effective resolution" of obfuscation methods by comparing
how well obfuscated images preserve semantic relationships while obscuring identity.

For each reference image u, the script:
1. Creates an obfuscated version X(u)
2. Randomly selects 'trials' other test images v
3. Computes resolution = d2 - d1 where:
   - d1 = 1 - similarity(X(u), v)    (distance from obfuscated to other's original)
   - d2 = 1 - similarity(X(u), X(v)) (distance between obfuscated versions)

Example:
    resolution-analysis ./images --objects face person --mode redact --trials 5
"""

import argparse
import csv
import json
import math
import random
import shlex
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
import numpy as np

from contrastive_privacy.scripts.anonymize import (
    CLIPSegModels,
    DEFAULT_FAL_VISION_MODEL,
    DEFAULT_FAL_VISION_TEMPERATURE,
    GroundedSAMModels,
    SAM3Models,
    anonymize,
    load_clipseg_models,
    load_groundedsam_models,
    load_sam3_models,
)
from contrastive_privacy.scripts.compare_images import (
    EVA02_CLIP_EMBEDDER_MODEL,
    compare_images,
    compute_embeddings_batch,
    load_clip_model,
    similarity_from_embeddings,
)
from contrastive_privacy.reporting import generate_analysis_artifacts


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

# Default minimum coverage threshold (as a fraction, not percentage)
# If less than this fraction of the image was detected, consider the obfuscation invalid
DEFAULT_MIN_COVERAGE = 0.001  # 0.1%

# Default maximum coverage threshold (as a fraction, not percentage)
# If more than this fraction of the image was altered, the image cannot be made private
DEFAULT_MAX_COVERAGE = 1.0  # 100% (no limit by default)

# Default HuggingFace CLIP model for resolution similarity (image embeddings)
DEFAULT_EMBEDDER_MODEL = "apple/DFN5B-CLIP-ViT-H-14-378"


def parse_report_skipped_paths(report_path: Path) -> list[Path]:
    """
    Parse report.txt and return the full paths of skipped reference images
    from the "Skipped reference images (detailed):" section.
    """
    paths: list[Path] = []
    content = report_path.read_text()
    in_section = False
    for line in content.splitlines():
        if "Skipped reference images (detailed):" in line:
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            if stripped.startswith("Path:"):
                path_str = stripped[5:].strip()
                if path_str:
                    paths.append(Path(path_str))
            # Section ends at next major heading or empty block (e.g. "  Trials requested")
            if stripped.startswith("Trials requested") or (stripped and not stripped.startswith("-") and not stripped.startswith("Path:") and not stripped.startswith("Reason:")):
                break
    return paths


def load_params_from_output(output_folder: Path) -> dict:
    """Load params.json from a previous run; raises FileNotFoundError if missing."""
    params_file = output_folder / "params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Params file not found: {params_file}. Run without --retry first.")
    with open(params_file) as f:
        return json.load(f)


def _argv_contains_any(argv: list[str], option_strings: list[str]) -> bool:
    """Return True if any of the option strings appear in argv (for override detection)."""
    for opt in option_strings:
        if opt in argv:
            return True
    return False


def get_image_files(folder: Path) -> list[Path]:
    """Get all image files from a folder."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(images)


def create_obfuscated_image(
    input_path: Path,
    output_path: Path,
    target_labels: list[str],
    mode: str,
    replacement_prompt: Optional[str] = None,
    device: Optional[str] = None,
    segmenter: str = "groundedsam",
    threshold: float | list[float] = 0.4,
    dilate: int = 5,
    blur: int = 8,
    strength: float = 0.85,
    model: str = "schnell",
    num_inference_steps: int = 28,
    redact_blur_radius: int = 30,
    seed: Optional[int] = None,
    clipseg_models: Optional[CLIPSegModels] = None,
    groundedsam_models: Optional[GroundedSAMModels] = None,
    sam3_models: Optional[SAM3Models] = None,
    adaptive_blur: bool = False,
    blur_scale: float = 1.0,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
    sequential_labels: bool = False,
    convex_hull: bool = False,
    skip_empty_labels: bool = False,
    refinements: int = 0,
    fal_image_model: str = "gpt-image-1.5",
    fal_vision_model: str = DEFAULT_FAL_VISION_MODEL,
    fal_vision_temperature: float = DEFAULT_FAL_VISION_TEMPERATURE,
    privacy_concept: Optional[str] = None,
    base_concepts: Optional[list[str]] = None,
) -> tuple[Path, float, float, float]:
    """
    Create an obfuscated version of an image.
    
    Args:
        input_path: Path to the original image.
        output_path: Path where the obfuscated image will be saved.
        target_labels: Objects to detect and obfuscate.
        mode: "redact" (blur), "blackout" (black pixels), or "replace" (inpaint).
        replacement_prompt: Prompt for replacement mode.
        device: Device to run on.
        segmenter: Segmentation model ("groundedsam", "clipseg", "sam3", "ai-gen", or "vlm-bounding-box").
        threshold: Detection threshold (0.0-1.0). Can be a single float (applied to all labels)
            or a list of floats (one per label, used when sequential_labels=True).
        dilate: Pixels to expand the mask (ignored if adaptive_blur=True).
        blur: Blur radius for mask edges (ignored if adaptive_blur=True).
        strength: Inpainting strength (0.0-1.0).
        model: FLUX model ("schnell" or "dev").
        num_inference_steps: Number of inference steps.
        redact_blur_radius: Blur radius for redaction mode.
        seed: Random seed for reproducibility (used in replace mode for inpainting).
        clipseg_models: Pre-loaded CLIPSeg models (optional).
        groundedsam_models: Pre-loaded GroundedSAM models (optional).
        sam3_models: Pre-loaded SAM3 models (optional).
        adaptive_blur: If True, scale blur/dilation based on object size.
        blur_scale: When adaptive_blur=True, controls overall blur intensity.
        size_exponent: Controls size-dependence (0.0=same for all, 1.0=linear).
        scaling_factor: Constant multiplier on effective size (default 1.0).
        sequential_labels: If True, process each label separately and combine masks.
            Ensures strictly additive behavior (adding labels never reduces coverage).
        convex_hull: If True, expand each object's mask to its convex hull.
        skip_empty_labels: If True, skip obfuscation of objects where GroundingDINO
            returned an empty label (''). Only applies when using groundedsam segmenter.
        refinements: Number of refinement passes for fal.ai vision polygon detection (default: 0).
            Only applies when using vlm-bounding-box segmenter.
        fal_vision_model: When segmenter is "vlm-bounding-box", which fal.ai OpenRouter vision model to use.
        fal_vision_temperature: OpenRouter sampling temperature for fal vision (vlm-bounding-box segmenter).
        fal_image_model: When segmenter is "ai-gen", fal.ai image edit model: model path (e.g. fal-ai/gpt-image-1)
            or short name (gpt-image-1.5, nano-banana-pro, flux-2-pro, flux-2-dev). Requires FAL_KEY.
        privacy_concept: When segmenter is "ai-gen", optional privacy concept to redact from the image.
            If provided, fal.ai will use it instead of `target_labels` to construct the edit prompt.
        base_concepts: Optional list of base concepts (words/phrases) to obfuscate using SAM3.
            If provided, they are always segmented with SAM3 (regardless of the selected `segmenter`).
            The primary obfuscation (for `target_labels` / `privacy_concept`) is performed first,
            and then the SAM3 base concept obfuscation is applied on top.
    
    Returns:
        Tuple of (path to the obfuscated image, final coverage percentage, base coverage percentage, objects coverage percentage).
    """
    if base_concepts:
        # Base concepts are always handled by SAM3.
        # To ensure the SAM3 step runs on the final primary-obfuscated image
        # (not on the original, which may be altered by image-generation models),
        # we run the primary obfuscation first, then apply SAM3 for the base concepts.
        if sam3_models is None:
            print("  WARNING: base_concepts provided but sam3_models not preloaded; SAM3 will be loaded on demand.")

        # SAM3 segmentation requires threshold list length to match label length.
        # When the user provided per-object thresholds (list) for the *main* labels,
        # we fall back to the first value for the base concepts unless lengths match.
        base_threshold: float | list[float]
        if isinstance(threshold, list):
            base_threshold = threshold if len(threshold) == len(base_concepts) else threshold[0]
        else:
            base_threshold = threshold

        primary_tmp_output_path = output_path.parent / f"primary_obfuscated_{output_path.name}"
        base_mask_path = output_path.parent / f"base_mask_{output_path.stem}.png"

        # Stage 1: run the chosen primary obfuscation mechanism on the original image.
        object_mask_path: Optional[Path] = None
        can_save_object_mask = segmenter not in {"ai-gen", "vlm-bounding-box"}
        if can_save_object_mask:
            object_mask_path = output_path.parent / f"object_mask_{output_path.stem}.png"

        _, coverage_objects = anonymize(
            input_path=input_path,
            output_path=primary_tmp_output_path,
            target_labels=target_labels,
            privacy_concept=privacy_concept,
            replacement_prompt=replacement_prompt,
            redact=(mode == "redact"),
            blackout=(mode == "blackout"),
            device=device,
            segmenter=segmenter,
            threshold=threshold,
            dilate=dilate,
            blur=blur,
            strength=strength,
            model=model,
            num_inference_steps=num_inference_steps,
            redact_blur_radius=redact_blur_radius,
            seed=seed,
            clipseg_models=clipseg_models,
            groundedsam_models=groundedsam_models,
            sam3_models=sam3_models,
            adaptive_blur=adaptive_blur,
            blur_scale=blur_scale,
            size_exponent=size_exponent,
            scaling_factor=scaling_factor,
            sequential_labels=sequential_labels,
            convex_hull=convex_hull,
            skip_empty_labels=skip_empty_labels,
            refinements=refinements,
            fal_image_model=fal_image_model,
            fal_vision_model=fal_vision_model,
            fal_vision_temperature=fal_vision_temperature,
            save_mask=object_mask_path,
        )

        # Stage 2: apply SAM3 base concept obfuscation on top of the primary-obfuscated image.
        _, coverage_base = anonymize(
            input_path=primary_tmp_output_path,
            output_path=output_path,
            target_labels=base_concepts,
            privacy_concept=None,
            replacement_prompt=replacement_prompt,
            redact=(mode == "redact"),
            blackout=(mode == "blackout"),
            device=device,
            segmenter="sam3",
            threshold=base_threshold,
            dilate=dilate,
            blur=blur,
            strength=strength,
            model=model,
            num_inference_steps=num_inference_steps,
            redact_blur_radius=redact_blur_radius,
            seed=seed,
            clipseg_models=None,
            groundedsam_models=None,
            sam3_models=sam3_models,
            adaptive_blur=adaptive_blur,
            blur_scale=blur_scale,
            size_exponent=size_exponent,
            scaling_factor=scaling_factor,
            sequential_labels=sequential_labels,
            convex_hull=convex_hull,
            skip_empty_labels=False,
            refinements=0,
            fal_image_model=fal_image_model,
            fal_vision_model=fal_vision_model,
            fal_vision_temperature=fal_vision_temperature,
            save_mask=base_mask_path,
        )

        # Combined coverage:
        # - If we saved both masks, compute pixelwise union for a closer estimate.
        # - Otherwise, fall back to a conservative approximation (sum, capped at 100).
        final_coverage: float
        if object_mask_path is not None and base_mask_path.exists():
            base_mask_arr = np.array(Image.open(base_mask_path).convert("L"))
            object_mask_arr = np.array(Image.open(object_mask_path).convert("L"))
            union_mask = (base_mask_arr > 127) | (object_mask_arr > 127)
            final_coverage = float(np.sum(union_mask)) / union_mask.size * 100.0
        else:
            final_coverage = min(100.0, coverage_base + coverage_objects)

        # Cleanup intermediate files
        try:
            if primary_tmp_output_path.exists():
                primary_tmp_output_path.unlink()
        except Exception:
            pass
        try:
            if base_mask_path.exists():
                base_mask_path.unlink()
        except Exception:
            pass
        if object_mask_path is not None:
            try:
                if object_mask_path.exists():
                    object_mask_path.unlink()
            except Exception:
                pass

        return output_path, final_coverage, coverage_base, coverage_objects

    # No base concepts: use existing single-pass behavior.
    _, coverage = anonymize(
        input_path=input_path,
        output_path=output_path,
        target_labels=target_labels,
        privacy_concept=privacy_concept,
        replacement_prompt=replacement_prompt,
        redact=(mode == "redact"),
        blackout=(mode == "blackout"),
        device=device,
        segmenter=segmenter,
        threshold=threshold,
        dilate=dilate,
        blur=blur,
        strength=strength,
        model=model,
        num_inference_steps=num_inference_steps,
        redact_blur_radius=redact_blur_radius,
        seed=seed,
        clipseg_models=clipseg_models,
        groundedsam_models=groundedsam_models,
        sam3_models=sam3_models,
        adaptive_blur=adaptive_blur,
        blur_scale=blur_scale,
        size_exponent=size_exponent,
        scaling_factor=scaling_factor,
        sequential_labels=sequential_labels,
        convex_hull=convex_hull,
        skip_empty_labels=skip_empty_labels,
        refinements=refinements,
        fal_image_model=fal_image_model,
        fal_vision_model=fal_vision_model,
        fal_vision_temperature=fal_vision_temperature,
    )
    return output_path, coverage, 0.0, coverage


def create_compound_image(
    xu_path: Path,
    v_path: Path,
    xv_path: Path,
    output_path: Path,
) -> Path:
    """
    Create a compound image with four sub-images arranged in a 2x2 grid.
    
    Layout:
        +--------+--------+
        |  X(u)  |   v    |
        +--------+--------+
        |  X(u)  |  X(v)  |
        +--------+--------+
    
    Args:
        xu_path: Path to the obfuscated version X(u).
        v_path: Path to the original test image v.
        xv_path: Path to the obfuscated version X(v).
        output_path: Path where the compound image will be saved.
    
    Returns:
        Path to the created compound image.
    """
    # Load all images
    xu_img = Image.open(xu_path).convert("RGB")
    v_img = Image.open(v_path).convert("RGB")
    xv_img = Image.open(xv_path).convert("RGB")
    
    # Find a common size (use the size of X(u) as reference)
    target_size = xu_img.size

    def fit_on_white_canvas(img: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
        """Scale image to the largest size that fits while preserving aspect ratio."""
        scale = min(canvas_size[0] / img.width, canvas_size[1] / img.height)
        fitted_size = (
            max(1, round(img.width * scale)),
            max(1, round(img.height * scale)),
        )
        fitted = img.resize(fitted_size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))
        x_offset = (canvas_size[0] - fitted.width) // 2
        y_offset = (canvas_size[1] - fitted.height) // 2
        canvas.paste(fitted, (x_offset, y_offset))
        return canvas

    # Keep each image in its original aspect ratio and center it on white.
    xu_tile = fit_on_white_canvas(xu_img, target_size)
    v_tile = fit_on_white_canvas(v_img, target_size)
    xv_tile = fit_on_white_canvas(xv_img, target_size)
    
    # Create the compound image (2x2 grid)
    width, height = target_size
    compound = Image.new("RGB", (width * 2, height * 2), color=(255, 255, 255))
    
    # Place images: upper-left=X(u), upper-right=v, lower-left=X(u), lower-right=X(v)
    compound.paste(xu_tile, (0, 0))
    compound.paste(v_tile, (width, 0))
    compound.paste(xu_tile, (0, height))
    compound.paste(xv_tile, (width, height))
    
    # Save the compound image
    compound.save(output_path, quality=95)
    
    return output_path


def create_summary_grid(
    image_paths: list[Union[Path, None]],
    output_path: Path,
    thumb_size: int = 256,
) -> Path:
    """
    Create a summary grid image from multiple images.
    
    The grid dimensions are calculated to be as close to square as possible.
    None entries in image_paths are rendered as white (empty) cells.
    
    Args:
        image_paths: List of paths to images to include in the grid. Use None for
            empty/white cells (e.g. skipped reference images in obfuscated summary).
        output_path: Path where the grid image will be saved.
        thumb_size: Size of each thumbnail (square).
    
    Returns:
        Path to the created grid image.
    """
    if not image_paths:
        raise ValueError("No images to create grid from")
    
    # Calculate grid dimensions to be as close to square as possible
    num_images = len(image_paths)
    # Use ceiling of square root for columns to get a near-square grid
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    # Create the grid image
    grid_width = cols * thumb_size
    grid_height = rows * thumb_size
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    
    # Place each image in the grid (None = leave cell white)
    for idx, img_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        
        if img_path is None:
            # Leave cell white (e.g. skipped reference in obfuscated summary)
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize to thumbnail while maintaining aspect ratio, then crop to square
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            
            # Center the thumbnail in its cell
            x_offset = col * thumb_size + (thumb_size - img.width) // 2
            y_offset = row * thumb_size + (thumb_size - img.height) // 2
            
            grid.paste(img, (x_offset, y_offset))
        except Exception as e:
            print(f"  Warning: Could not add {img_path.name} to grid: {e}")
            continue
    
    # Save the grid
    grid.save(output_path, quality=90)
    return output_path


def format_resolution_histogram(
    resolutions: list[float],
    num_bins: int = 20,
    width: int = 50,
) -> list[str]:
    """
    Format a text-based histogram of resolution values.
    
    Args:
        resolutions: List of resolution values.
        num_bins: Number of bins for the histogram.
        width: Width of the histogram bars in characters.
    
    Returns:
        List of formatted lines for the histogram.
    """
    lines = []
    
    if not resolutions:
        lines.append("  No data to display.")
        return lines
    
    min_res = min(resolutions)
    max_res = max(resolutions)
    mean_res = sum(resolutions) / len(resolutions)
    
    # Handle edge case where all values are the same
    if min_res == max_res:
        lines.append(f"  All {len(resolutions)} values are {min_res:.4f}")
        return lines
    
    # Create bins
    bin_width = (max_res - min_res) / num_bins
    bins = [0] * num_bins
    
    for r in resolutions:
        # Find the appropriate bin
        bin_idx = int((r - min_res) / bin_width)
        # Handle edge case where r == max_res
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        bins[bin_idx] += 1
    
    # Find max count for scaling
    max_count = max(bins) if bins else 1
    
    # Statistics
    lines.append(f"  Min: {min_res:+.4f}  Max: {max_res:+.4f}  Mean: {mean_res:+.4f}")
    lines.append(f"  Median: {sorted(resolutions)[len(resolutions) // 2]:+.4f}")
    lines.append("")
    
    # Count positive and negative
    num_positive = sum(1 for r in resolutions if r > 0)
    num_negative = sum(1 for r in resolutions if r < 0)
    num_zero = sum(1 for r in resolutions if r == 0)
    lines.append(f"  Negative (good): {num_negative} ({num_negative / len(resolutions) * 100:.1f}%)")
    lines.append(f"  Zero:            {num_zero} ({num_zero / len(resolutions) * 100:.1f}%)")
    lines.append(f"  Positive (bad):  {num_positive} ({num_positive / len(resolutions) * 100:.1f}%)")
    lines.append("")
    
    # Histogram bars
    for i, count in enumerate(bins):
        bin_start = min_res + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_length
        
        # Mark if this bin contains zero
        zero_marker = " *" if bin_start <= 0 < bin_end else "  "
        
        lines.append(f"  [{bin_start:+.3f}, {bin_end:+.3f}){zero_marker} {bar} {count}")
    
    lines.append("")
    lines.append("  (* indicates bin containing zero)")
    
    return lines


def print_resolution_histogram(
    resolutions: list[float],
    num_bins: int = 20,
    width: int = 50,
) -> None:
    """
    Print a text-based histogram of resolution values.
    
    Args:
        resolutions: List of resolution values.
        num_bins: Number of bins for the histogram.
        width: Width of the histogram bars in characters.
    """
    for line in format_resolution_histogram(resolutions, num_bins, width):
        print(line)


def compute_resolution(
    u_path: Path,
    v_path: Path,
    xu_path: Path,
    xv_path: Path,
    device: Optional[str] = None,
    clip_model=None,
    clip_processor=None,
) -> float:
    """
    Compute the resolution for a pair of images.
    
    Resolution = d2 - d1 where:
    - d1 = 1 - similarity(X(u), v)  (distance from obfuscated to other's original)
    - d2 = 1 - similarity(X(u), X(v))  (distance between obfuscated versions)
    
    Args:
        u_path: Path to reference image u.
        v_path: Path to test image v (unused but kept for clarity).
        xu_path: Path to obfuscated version X(u).
        xv_path: Path to obfuscated version X(v).
        device: Device to run on.
        clip_model: Pre-loaded CLIP model (optional).
        clip_processor: Pre-loaded CLIP processor (optional).
    
    Returns:
        The resolution value (d2 - d1).
    """
    # d1 = 1 - similarity(X(u), v)
    sim_xu_v = compare_images(xu_path, v_path, device=device, model=clip_model, processor=clip_processor)
    d1 = 1 - sim_xu_v
    
    # d2 = 1 - similarity(X(u), X(v))
    sim_xu_xv = compare_images(xu_path, xv_path, device=device, model=clip_model, processor=clip_processor)
    d2 = 1 - sim_xu_xv
    
    resolution = d2 - d1
    return resolution


def run_resolution_analysis(
    image_folder: Path,
    objects: list[str],
    mode: str,
    trials: int,
    output_folder: Path,
    samples: Optional[int] = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_coverage: float = DEFAULT_MAX_COVERAGE,
    replacement_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    segmenter: str = "groundedsam",
    threshold: float | list[float] = 0.4,
    dilate: int = 5,
    blur: int = 8,
    strength: float = 0.85,
    model: str = "schnell",
    num_inference_steps: int = 28,
    redact_blur_radius: int = 30,
    adaptive_blur: bool = False,
    blur_scale: float = 1.0,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
    sequential_labels: bool = False,
    convex_hull: bool = False,
    skip_empty_labels: bool = False,
    refinements: int = 0,
    continue_from_output: bool = False,
    fal_image_model: str = "gpt-image-1.5",
    fal_vision_model: str = DEFAULT_FAL_VISION_MODEL,
    fal_vision_temperature: float = DEFAULT_FAL_VISION_TEMPERATURE,
    privacy_concept: Optional[str] = None,
    base_concepts: Optional[list[str]] = None,
    command_line: str | None = None,
    retry_skipped_paths: Optional[list[Path]] = None,
    generate_comparisons: bool = True,
    comparison_top_n: Optional[int] = None,
    obfuscate_missing_in_continue: bool = True,
    embedder_model: str = DEFAULT_EMBEDDER_MODEL,
    embed_batch_size: int = 8,
    verbose: bool = False,
    write_analysis_artifacts: bool = True,
) -> None:
    """
    Run resolution analysis on a folder of images.
    
    Args:
        image_folder: Folder containing input images.
        objects: List of objects to detect and obfuscate (ignored by `ai-gen` when `privacy_concept` is provided).
        privacy_concept: When segmenter is "ai-gen", optional privacy concept to redact from the image.
            If provided, `fal_image_model` is instructed to black out regions that can reveal this concept.
        base_concepts: Optional list of base concepts to obfuscate using SAM3 after the primary obfuscation.
        mode: "redact" or "replace".
        trials: Number of test images to sample per reference image.
        output_folder: Base folder for all outputs. Will contain:
            - comparisons/: Compound comparison images
            - obfuscated/: Individual obfuscated images
            - summary_obfuscated.jpg: Grid of all obfuscated images
            - summary_originals.jpg: Grid of original images (same positions)
            - results.csv: Analysis results
            - params.json: Parameters used for this run (for reproducibility)
            - report.txt: Summary report with privacy analysis
        samples: Number of reference images to sample (None = use all).
        min_coverage: Minimum mask coverage fraction (0.0-1.0) for valid obfuscation.
        max_coverage: Maximum mask coverage fraction (0.0-1.0) for valid obfuscation.
            Images requiring more than this fraction to be altered are discarded.
        replacement_prompt: Prompt for replacement mode.
        seed: Random seed for reproducibility.
        device: Device to run on.
        segmenter: Segmentation model ("groundedsam", "clipseg", "ai-gen", or "vlm-bounding-box").
        threshold: Detection threshold (0.0-1.0). Can be a single float (applied to all labels)
            or a list of floats (one per label, used when sequential_labels=True).
        dilate: Pixels to expand the mask (ignored if adaptive_blur=True).
        blur: Blur radius for mask edges (ignored if adaptive_blur=True).
        strength: Inpainting strength (0.0-1.0).
        model: FLUX model ("schnell" or "dev").
        num_inference_steps: Number of inference steps.
        redact_blur_radius: Blur radius for redaction mode.
        adaptive_blur: If True, scale blur/dilation based on object size.
        blur_scale: When adaptive_blur=True, controls overall blur intensity.
        size_exponent: Controls size-dependence (0.0=same for all, 1.0=linear).
        scaling_factor: Constant multiplier on effective size (default 1.0).
        sequential_labels: If True, process each label separately and combine masks.
            Ensures strictly additive behavior (adding labels never reduces coverage).
        convex_hull: If True, expand each object's mask to its convex hull.
        skip_empty_labels: If True, skip obfuscation of objects where GroundingDINO
            returned an empty label (''). Only applies when using groundedsam segmenter.
        refinements: Number of refinement passes for GPT-5.2 polygon detection (default: 0).
            Only applies when using vlm-bounding-box segmenter.
        continue_from_output: If True, load existing obfuscated images from output_folder/obfuscated/
            and only obfuscate images that are not yet present. Used to resume after a failed run.
        fal_image_model: When segmenter is "ai-gen", which fal.ai image edit model to use.
        fal_vision_model: When segmenter is "vlm-bounding-box", which fal.ai OpenRouter vision model to use.
        fal_vision_temperature: OpenRouter sampling temperature when using the fal vision segmenter.
        command_line: Full command line as issued (e.g. shlex.join(sys.argv)) for the report. Optional.
        retry_skipped_paths: When set, only obfuscate these image paths (e.g. skipped refs from report.txt).
            Used with continue_from_output to retry obfuscation for previously skipped reference images.
        generate_comparisons: If False, skip creating compound comparison images while still
            computing and reporting resolution values.
        comparison_top_n: If set, only create comparison images for the N highest-resolution
            comparisons across the full run. Resolution metrics are still computed for all pairs.
        obfuscate_missing_in_continue: When continue_from_output=True, controls whether images
            missing from output/obfuscated are newly obfuscated. Set False to only reuse existing
            obfuscations and skip missing ones.
        embedder_model: HuggingFace model ID for image embeddings used to compute
            similarity (resolution). Standard CLIP or Eva-02 CLIP (e.g.
            ``microsoft/LLM2CLIP-EVA02-B-16``). Same entry point as ``compare-images --model``.
        embed_batch_size: Batch size when pre-computing image embeddings (Step 3).
            Lower this value if you hit CUDA OOM with larger embedding models.
        verbose: If True, include per-comparison resolution details (including d1/d2)
            in report.txt.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle random seed: generate one if not provided
    if seed is None:
        # Generate a random seed and use it
        seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {seed}")
    else:
        print(f"Using provided seed: {seed}")
    random.seed(seed)
    
    # Get all image files
    image_files = get_image_files(image_folder)
    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 images in folder, found {len(image_files)}")
    
    print(f"Found {len(image_files)} images in {image_folder}")
    print(f"Obfuscation mode: {mode}")
    if privacy_concept is not None:
        print(f"Privacy concept: {privacy_concept}")
    else:
        print(f"Target objects: {', '.join(objects)}")
    if base_concepts:
        print(f"Base concepts (SAM3-after-primary): {', '.join(base_concepts)}")
    print(f"Segmenter: {segmenter}")
    if segmenter == "vlm-bounding-box":
        print(f"Fal vision temperature (OpenRouter): {fal_vision_temperature}")
    print(f"Trials per reference image: {trials}")
    print(f"Device: {device}")
    print(f"Embedder: {embedder_model}")
    print(f"Embed batch size: {embed_batch_size}")
    print(f"Anonymization settings: threshold={threshold if isinstance(threshold, (int, float)) else ', '.join(map(str, threshold))}, dilate={dilate}, blur={blur}")
    if adaptive_blur:
        print(f"Adaptive blur enabled: blur_scale={blur_scale}, size_exponent={size_exponent}, scaling_factor={scaling_factor}")
    if sequential_labels:
        print(f"Sequential labels enabled: each object processed separately for additive detection")
    if convex_hull:
        print(f"Convex hull enabled: object masks expanded to convex hull")
    if skip_empty_labels:
        print(f"Skip empty labels enabled: objects with empty labels will be filtered out")
    if mode == "redact":
        print(f"Redact blur radius: {redact_blur_radius}")
    elif mode == "replace":
        print(f"Inpainting: model={model}, strength={strength}, steps={num_inference_steps}")
    
    # Set up output directory structure
    output_folder.mkdir(parents=True, exist_ok=True)
    comparisons_folder = output_folder / "comparisons"
    obfuscated_folder = output_folder / "obfuscated"
    output_csv = output_folder / "results.csv"
    params_file = output_folder / "params.json"
    report_file = output_folder / "report.txt"
    
    if generate_comparisons:
        comparisons_folder.mkdir(parents=True, exist_ok=True)
    obfuscated_folder.mkdir(parents=True, exist_ok=True)
    
    # Save parameters for reproducibility
    params = {
        "image_folder": str(image_folder.absolute()),
        "objects": objects,
        "privacy_concept": privacy_concept,
        "base_concepts": base_concepts,
        "mode": mode,
        "trials": trials,
        "samples": samples,
        "min_coverage": min_coverage,
        "max_coverage": max_coverage,
        "replacement_prompt": replacement_prompt,
        "seed": seed,
        "device": device,
        "segmenter": segmenter,
        "threshold": threshold,
        "dilate": dilate,
        "blur": blur,
        "strength": strength,
        "model": model,
        "num_inference_steps": num_inference_steps,
        "redact_blur_radius": redact_blur_radius,
        "adaptive_blur": adaptive_blur,
        "blur_scale": blur_scale,
        "size_exponent": size_exponent,
        "scaling_factor": scaling_factor,
        "sequential_labels": sequential_labels,
        "convex_hull": convex_hull,
        "skip_empty_labels": skip_empty_labels,
        "refinements": refinements,
        "continue_from_output": continue_from_output,
        "generate_comparisons": generate_comparisons,
        "comparison_top_n": comparison_top_n,
        "fal_image_model": fal_image_model,
        "fal_vision_model": fal_vision_model,
        "fal_vision_temperature": fal_vision_temperature,
        "embedder_model": embedder_model,
        "embed_batch_size": embed_batch_size,
        "verbose": verbose,
        "write_analysis_artifacts": write_analysis_artifacts,
        "timestamp": datetime.now().isoformat(),
    }
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    
    if continue_from_output:
        print("Resume mode (--continue): will reuse existing obfuscated images and only obfuscate missing ones.")
    print(f"Output folder: {output_folder}")
    if generate_comparisons:
        print(f"  Comparisons: {comparisons_folder}")
        if comparison_top_n is None:
            print("  Comparison selection: all computed comparisons")
        else:
            print(f"  Comparison selection: top {comparison_top_n} highest-resolution comparisons")
    else:
        print(f"  Comparisons: disabled")
    print(f"  Obfuscated: {obfuscated_folder}")
    print(f"  Results CSV: {output_csv}")
    print(f"  Parameters: {params_file}")
    
    # Pre-load models for efficiency (loaded once, reused for all images)
    print("\n" + "=" * 60)
    print("Loading models (this will be reused for all images)")
    print("=" * 60)
    
    # Load segmentation models based on selected segmenter
    clipseg_models = None
    groundedsam_models = None
    sam3_models = None
    needs_sam3_for_base = bool(base_concepts)
    if segmenter == "ai-gen":
        print(f"Using fal.ai image edit API (model: {fal_image_model}, no local models to load)")
        if needs_sam3_for_base:
            print("Loading SAM3 models for base concepts...")
            sam3_models = load_sam3_models(device)
    elif segmenter == "vlm-bounding-box":
        print("Using GPT-5.2 vision API for polygon detection (no models to load)")
        if needs_sam3_for_base:
            print("Loading SAM3 models for base concepts...")
            sam3_models = load_sam3_models(device)
    elif segmenter == "clipseg":
        print("Loading CLIPSeg models...")
        clipseg_models = load_clipseg_models(device)
        if needs_sam3_for_base:
            print("Loading SAM3 models for base concepts...")
            sam3_models = load_sam3_models(device)
    elif segmenter == "sam3":
        sam3_models = load_sam3_models(device)
    else:  # groundedsam
        print("Loading GroundingDINO + SAM models...")
        groundedsam_models = load_groundedsam_models(device)
        if needs_sam3_for_base:
            print("Loading SAM3 models for base concepts...")
            sam3_models = load_sam3_models(device)
    
    # Delay loading embedder until Step 3 to keep max VRAM available for segmentation.
    clip_model = None
    clip_processor = None
    print("Segmentation models loaded successfully!")
    
    # Use a temporary directory for intermediate obfuscated images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Sample reference images and determine which images need obfuscation
        print()
        print("=" * 60)
        print("Step 1: Selecting images to process")
        print("=" * 60)
        
        # Sample reference images if requested
        if samples is not None and samples < len(image_files):
            reference_images = random.sample(image_files, samples)
            print(f"Randomly selected {samples} reference images from {len(image_files)} available")
        else:
            reference_images = list(image_files)
            print(f"Using all {len(reference_images)} images as reference images")
        
        # For each reference image, pre-select the test images
        # This allows us to know exactly which images need obfuscation
        reference_to_tests: dict[Path, list[Path]] = {}
        all_images_to_obfuscate: set[Path] = set()
        
        for u_path in reference_images:
            # Get other images (from the full set, not just reference images)
            other_images = [p for p in image_files if p != u_path]
            num_test_samples = min(trials, len(other_images))
            test_images = random.sample(other_images, num_test_samples)
            reference_to_tests[u_path] = test_images
            
            # Track all images that need obfuscation
            all_images_to_obfuscate.add(u_path)
            all_images_to_obfuscate.update(test_images)
        
        images_to_obfuscate = sorted(all_images_to_obfuscate)
        # When retrying skipped refs, only obfuscate those images (match by filename)
        if retry_skipped_paths:
            by_name = {p.name: p for p in image_files}
            images_to_obfuscate = sorted(
                by_name[p.name] for p in retry_skipped_paths if p.name in by_name
            )
            print(f"Retry mode: obfuscating {len(images_to_obfuscate)} skipped reference image(s) only")
        print(f"Total images to obfuscate: {len(images_to_obfuscate)}")
        
        # Step 2: Create obfuscated versions of required images only
        print()
        print("=" * 60)
        print("Step 2: Creating obfuscated versions (in temporary folder)")
        print("=" * 60)
        
        obfuscated_paths: dict[Path, Path] = {}
        valid_obfuscations: set[Path] = set()  # Images where obfuscation actually changed something
        
        # When --continue or --retry: load existing obfuscated images from output folder
        image_files_set = set(image_files)
        if (continue_from_output or retry_skipped_paths) and obfuscated_folder.exists():
            for obf_file in obfuscated_folder.iterdir():
                if not obf_file.is_file() or not obf_file.name.startswith("obfuscated_"):
                    continue
                original_name = obf_file.name[len("obfuscated_"):]
                # Match to an input image by name (same name under image_folder)
                original_path = next((p for p in image_files_set if p.name == original_name), None)
                if original_path is not None:
                    obfuscated_paths[original_path] = obf_file
                    valid_obfuscations.add(original_path)
            if obfuscated_paths:
                print(f"Loaded {len(obfuscated_paths)} existing obfuscated images from {obfuscated_folder}")
        
        for i, img_path in enumerate(images_to_obfuscate):
            print(f"\n[{i+1}/{len(images_to_obfuscate)}] Processing: {img_path.name}")
            
            # Skip if we already have this image (from --continue). In retry mode we never skip.
            existing_obf = obfuscated_folder / f"obfuscated_{img_path.name}"
            if (
                continue_from_output
                and not retry_skipped_paths
                and existing_obf.exists()
                and img_path in obfuscated_paths
            ):
                print(f"  Using existing obfuscated image (skip)")
                continue
            
            if (
                continue_from_output
                and not obfuscate_missing_in_continue
                and img_path not in obfuscated_paths
            ):
                print(
                    "  Skipping: missing existing obfuscated image and "
                    "--continue-only mode is enabled"
                )
                continue

            obf_path = temp_path / f"obfuscated_{img_path.stem}{img_path.suffix}"
            
            try:
                obf_path, coverage, base_coverage, objects_coverage = create_obfuscated_image(
                    input_path=img_path,
                    output_path=obf_path,
                    target_labels=objects,
                    privacy_concept=privacy_concept,
                    base_concepts=base_concepts,
                    mode=mode,
                    replacement_prompt=replacement_prompt,
                    device=device,
                    segmenter=segmenter,
                    threshold=threshold,
                    dilate=dilate,
                    blur=blur,
                    strength=strength,
                    model=model,
                    num_inference_steps=num_inference_steps,
                    redact_blur_radius=redact_blur_radius,
                    seed=seed,
                    clipseg_models=clipseg_models,
                    groundedsam_models=groundedsam_models,
                    sam3_models=sam3_models,
                    adaptive_blur=adaptive_blur,
                    blur_scale=blur_scale,
                    size_exponent=size_exponent,
                    scaling_factor=scaling_factor,
                    sequential_labels=sequential_labels,
                    convex_hull=convex_hull,
                    skip_empty_labels=skip_empty_labels,
                    refinements=refinements,
                    fal_image_model=fal_image_model,
                    fal_vision_model=fal_vision_model,
                    fal_vision_temperature=fal_vision_temperature,
                )
                # Check if obfuscation actually changed the image based on coverage
                # Only add to obfuscated_paths (and thus save to output folder) when valid
                coverage_fraction = coverage / 100.0
                base_coverage_fraction = base_coverage / 100.0
                base_modified = bool(base_concepts and base_coverage_fraction > 0)
                if coverage_fraction < min_coverage:
                    if base_modified:
                        print(
                            f"  WARNING: Total coverage for {img_path.name} is below --min-coverage "
                            f"(coverage={coverage_fraction:.4f}, min={min_coverage:.4f}), but base concepts modified the image; saving anyway."
                        )
                        # Save to output folder immediately so --continue can resume after a fail-fast
                        output_filename = f"obfuscated_{img_path.name}"
                        output_path = obfuscated_folder / output_filename
                        shutil.copy2(obf_path, output_path)
                        obfuscated_paths[img_path] = output_path
                        valid_obfuscations.add(img_path)
                        print(f"  Obfuscation successful (coverage={coverage_fraction:.4f})")
                    else:
                        print(
                            f"  WARNING: Obfuscation did not change {img_path.name} "
                            f"(coverage={coverage_fraction:.4f}) - not saving to obfuscated folder"
                        )
                elif coverage_fraction > max_coverage:
                    print(f"  WARNING: Image {img_path.name} cannot be made private because privacy "
                          f"requires discarding more than {max_coverage:.1%} of the image "
                          f"(coverage={coverage_fraction:.4f}) - not saving to obfuscated folder")
                else:
                    # Save to output folder immediately so --continue can resume after a fail-fast
                    output_filename = f"obfuscated_{img_path.name}"
                    output_path = obfuscated_folder / output_filename
                    shutil.copy2(obf_path, output_path)
                    obfuscated_paths[img_path] = output_path
                    valid_obfuscations.add(img_path)
                    print(f"  Obfuscation successful (coverage={coverage_fraction:.4f})")
                    
            except Exception as e:
                print(f"  WARNING: Failed to obfuscate {img_path.name}: {e}")
                continue
        
        # Fail fast: when not resuming and not retrying, any reference image that failed obfuscation must stop the run
        if not continue_from_output and not retry_skipped_paths:
            missing_refs = [u_path.name for u_path in reference_images if u_path not in obfuscated_paths]
            if missing_refs:
                print("\n" + "=" * 60)
                print("ERROR: Obfuscation failed for one or more reference images.")
                print("The following reference images could not be obfuscated:")
                for name in missing_refs:
                    print(f"  - {name}")
                print("Re-run with --continue and the same output folder to resume (existing obfuscated images will be reused).")
                print("=" * 60)
                sys.exit(1)
        
        print(f"\nSuccessfully obfuscated {len(obfuscated_paths)} images")
        print(f"Valid obfuscations (image changed): {len(valid_obfuscations)}")
        
        # Save obfuscated images to output folder (only those not already there, e.g. newly created in temp)
        print(f"\nSaving obfuscated images to: {obfuscated_folder}")
        saved_obfuscated_paths = []
        corresponding_original_paths = []
        for original_path, obf_path in obfuscated_paths.items():
            output_filename = f"obfuscated_{original_path.name}"
            output_path = obfuscated_folder / output_filename
            if obf_path.resolve().parent != obfuscated_folder.resolve():
                # Newly created in temp folder: copy to output
                shutil.copy2(obf_path, output_path)
            # else: already in output folder (from --continue), no copy needed
            saved_obfuscated_paths.append(output_path)
            corresponding_original_paths.append(original_path)
        print(f"Saved {len(saved_obfuscated_paths)} obfuscated images")
        
        # Create summary grids by reference image order (one cell per reference).
        # Originals: all reference images. Obfuscated: obfuscated when available, white cell when skipped.
        if reference_images:
            summary_originals_list = list(reference_images)
            summary_obfuscated_list: list[Union[Path, None]] = []
            for u_path in reference_images:
                if u_path in obfuscated_paths:
                    output_filename = f"obfuscated_{u_path.name}"
                    summary_obfuscated_list.append(obfuscated_folder / output_filename)
                else:
                    summary_obfuscated_list.append(None)  # White cell for skipped reference
            
            summary_obfuscated_path = output_folder / "summary_obfuscated.jpg"
            print(f"\nCreating obfuscated summary grid: {summary_obfuscated_path}")
            create_summary_grid(
                image_paths=summary_obfuscated_list,
                output_path=summary_obfuscated_path,
            )
            num_obfuscated_cells = sum(1 for p in summary_obfuscated_list if p is not None)
            print(f"Obfuscated summary grid saved ({len(reference_images)} cells, {num_obfuscated_cells} obfuscated, {len(reference_images) - num_obfuscated_cells} skipped/white)")
            
            summary_originals_path = output_folder / "summary_originals.jpg"
            print(f"Creating originals summary grid: {summary_originals_path}")
            create_summary_grid(
                image_paths=summary_originals_list,
                output_path=summary_originals_path,
            )
            print(f"Originals summary grid saved with {len(summary_originals_list)} reference images")
        
        # Step 3: Pre-compute image embeddings for all needed images
        print("\n" + "=" * 60)
        print("Step 3: Pre-computing image embeddings (batched for efficiency)")
        print("=" * 60)
        print(f"Loading image embedder ({embedder_model})...")
        clip_model, clip_processor, _ = load_clip_model(embedder_model, device=device)
        
        # Get list of images with valid obfuscations (for selecting test images)
        valid_test_candidates = [p for p in image_files if p in valid_obfuscations]
        print(f"Available test images with valid obfuscations: {len(valid_test_candidates)}")
        
        # Collect all image paths we need embeddings for:
        # - Original images v (for d1 = 1 - sim(X(u), v))
        # - Obfuscated images X(u) and X(v) (for both d1 and d2)
        all_paths_for_embedding: set[Path] = set()
        for v_path in valid_test_candidates:
            all_paths_for_embedding.add(v_path)  # Original v
            all_paths_for_embedding.add(obfuscated_paths[v_path])  # X(v)
        
        all_paths_list = sorted(all_paths_for_embedding)
        print(f"Computing embeddings for {len(all_paths_list)} images...")
        
        # Compute all embeddings in batches
        embeddings = compute_embeddings_batch(
            image_paths=all_paths_list,
            model=clip_model,
            processor=clip_processor,
            device=device,
            batch_size=max(1, embed_batch_size),
        )
        print(f"Embeddings computed for {len(embeddings)} images")
        
        # Step 4: Compute resolutions using cached embeddings
        print("\n" + "=" * 60)
        print("Step 4: Computing resolutions")
        print("=" * 60)
        
        results = []
        next_compound_index = 0
        references_processed = 0
        skipped_references: list[tuple[Path, str]] = []  # (path, reason) for report
        total_trials_requested = 0
        total_trials_actual = 0
        
        for i, u_path in enumerate(reference_images):
            print(f"\n[{i+1}/{len(reference_images)}] Processing reference: {u_path.name}")
            
            # Skip if reference image failed to obfuscate
            if u_path not in obfuscated_paths:
                reason = "obfuscation failed"
                print(f"  WARNING: Skipping {u_path.name} - {reason}")
                skipped_references.append((u_path, reason))
                continue
            
            # Skip if reference image obfuscation didn't change anything
            if u_path not in valid_obfuscations:
                reason = "obfuscation did not change the image (u = X(u))"
                print(f"  WARNING: Skipping {u_path.name} - {reason}")
                skipped_references.append((u_path, reason))
                continue
            
            xu_path = obfuscated_paths[u_path]
            
            # Get valid test images (excluding the reference image itself)
            available_tests = [p for p in valid_test_candidates if p != u_path]
            
            if len(available_tests) == 0:
                reason = "no valid test images available"
                print(f"  WARNING: Skipping {u_path.name} - {reason}")
                skipped_references.append((u_path, reason))
                continue
            
            # Sample test images from valid candidates (use fewer if not enough available)
            actual_trials = min(trials, len(available_tests))
            if actual_trials < trials:
                print(f"  WARNING: Only {actual_trials} valid test images available "
                      f"(requested {trials})")
            test_images = random.sample(available_tests, actual_trials)
            
            # Track trial counts
            references_processed += 1
            total_trials_requested += trials
            total_trials_actual += actual_trials
            
            # Compute resolution for each test image
            ref_results = []  # Results for this reference image
            for v_path in test_images:
                xv_path = obfuscated_paths[v_path]
                
                try:
                    # Use cached embeddings for fast similarity computation
                    xu_emb = embeddings[xu_path]
                    v_emb = embeddings[v_path]
                    xv_emb = embeddings[xv_path]
                    
                    # d1 = 1 - similarity(X(u), v)
                    sim_xu_v = similarity_from_embeddings(xu_emb, v_emb)
                    d1 = 1 - sim_xu_v
                    
                    # d2 = 1 - similarity(X(u), X(v))
                    sim_xu_xv = similarity_from_embeddings(xu_emb, xv_emb)
                    d2 = 1 - sim_xu_xv
                    
                    res = d2 - d1
                    
                    ref_results.append({
                        "comparison_index": next_compound_index,
                        "resolution": res,
                        "d1": d1,
                        "d2": d2,
                        "compound": "",
                        "u": str(u_path.absolute()),
                        "v": str(v_path.absolute()),
                        "u_name": u_path.name,
                        "v_name": v_path.name,
                        "xu_path": xu_path,
                        "xv_path": xv_path,
                        "compound_filename": "",
                    })
                    next_compound_index += 1
                except Exception as e:
                    print(f"  WARNING: Failed to compute resolution for {v_path.name}: {e}")
                    continue
            
            # Sort and print results for this reference image by increasing resolution
            ref_results.sort(key=lambda x: x["resolution"])
            for r in ref_results:
                print(f"    {r['resolution']:+.4f}  vs {r['v_name']}")
            
            # Add to overall results
            results.extend(ref_results)
        
        references_skipped = len(skipped_references)

        if generate_comparisons and results:
            print("\n" + "=" * 60)
            print("Step 5: Creating comparison images")
            print("=" * 60)
            sorted_results = sorted(results, key=lambda x: x["resolution"], reverse=True)
            results_for_comparisons = (
                sorted_results if comparison_top_n is None else sorted_results[:comparison_top_n]
            )
            print(f"Creating {len(results_for_comparisons)} comparison image(s)...")

            for result in results_for_comparisons:
                compound_index = result["comparison_index"]
                res_str = f"{result['resolution']:.4f}".replace("-", "n").replace(".", "p")
                compound_filename = f"cmp_{compound_index:04d}_res{res_str}.jpg"
                compound_path = comparisons_folder / compound_filename
                create_compound_image(
                    xu_path=result["xu_path"],
                    v_path=Path(result["v"]),
                    xv_path=result["xv_path"],
                    output_path=compound_path,
                )
                result["compound"] = str(compound_path.absolute())
                result["compound_filename"] = compound_filename
                print(
                    f"  [{result['comparison_index'] + 1}/{next_compound_index}] "
                    f"{result['u_name']} vs {result['v_name']} -> {compound_filename}"
                )
    
    # Clean up models to free GPU memory
    print("\n" + "=" * 60)
    print("Cleaning up models")
    print("=" * 60)
    
    if clipseg_models is not None:
        del clipseg_models
    if groundedsam_models is not None:
        del groundedsam_models
    if sam3_models is not None:
        del sam3_models
    if clip_model is not None:
        del clip_model
    if clip_processor is not None:
        del clip_processor
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Step 6: Write results to CSV
    print("\n" + "=" * 60)
    print("Step 6: Writing results to CSV")
    print("=" * 60)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort all results by increasing resolution for the CSV
    results.sort(key=lambda x: x["resolution"])
    
    # Prepare rows for CSV (exclude temporary display fields)
    fieldnames = ["resolution", "compound", "u", "v"]
    csv_rows = [{k: r[k] for k in fieldnames} for r in results]
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Build report content (will be written to both stdout and report.txt)
    report_lines = []
    
    def report(line: str = "") -> None:
        """Print a line and add it to the report."""
        print(line)
        report_lines.append(line)
    
    report("=" * 60)
    report("Resolution Analysis Report")
    report("=" * 60)
    report()
    if command_line is not None:
        report("Command line:")
        report(f"  {command_line}")
        report()
    report(f"Timestamp: {datetime.now().isoformat()}")
    report(f"Input folder: {image_folder}")
    report(f"Output folder: {output_folder}")
    report()
    report("Configuration:")
    report(f"  Mode: {mode}")
    if privacy_concept is not None:
        report(f"  Privacy concept: {privacy_concept}")
    else:
        report(f"  Objects: {', '.join(objects)}")
    if base_concepts:
        report(f"  Base concepts: {', '.join(base_concepts)}")
    report(f"  Segmenter: {segmenter}")
    report(f"  Embedder: {embedder_model}")
    report(f"  Fal image model: {fal_image_model}")
    report(f"  Fal vision model: {fal_vision_model}")
    report(f"  Fal vision temperature: {fal_vision_temperature}")
    report(f"  Verbose report: {'enabled' if verbose else 'disabled'}")
    report(f"  Trials per reference: {trials}")
    report(f"  Samples: {samples if samples else 'all'}")
    report(f"  Min coverage: {min_coverage}")
    report(f"  Max coverage: {max_coverage}")
    report(f"  Threshold: {threshold if isinstance(threshold, (int, float)) else ', '.join(map(str, threshold))}")
    report(f"  Dilate: {dilate}")
    report(f"  Blur: {blur}")
    if adaptive_blur:
        report(f"  Adaptive blur: enabled (blur_scale={blur_scale}, scaling_factor={scaling_factor}, exponent={size_exponent})")
    if sequential_labels:
        report(f"  Sequential labels: enabled (additive detection)")
    if convex_hull:
        report(f"  Convex hull: enabled (fills concave regions)")
    if skip_empty_labels:
        report(f"  Skip empty labels: enabled (filters ambiguous detections)")
    if mode == "redact":
        report(f"  Redact blur radius: {redact_blur_radius}")
    elif mode == "replace":
        report(f"  Replacement prompt: {replacement_prompt}")
        report(f"  Strength: {strength}")
        report(f"  Model: {model}")
        report(f"  Inference steps: {num_inference_steps}")
    report(f"  Seed: {seed}")
    report()
    report("Output files:")
    report(f"  Results CSV: {output_csv}")
    if generate_comparisons:
        report(f"  Comparison images: {comparisons_folder}")
        if comparison_top_n is None:
            report("  Comparison selection: all computed comparisons")
        else:
            report(f"  Comparison selection: top {comparison_top_n} highest-resolution comparisons")
    else:
        report("  Comparison images: disabled (--skip-comparisons)")
    report(f"  Obfuscated images: {obfuscated_folder}")
    report(f"  Parameters: {params_file}")
    report(f"  Report: {report_file}")
    report()
    report("Summary:")
    report(f"  Reference images processed: {references_processed}")
    report(f"  Reference images skipped: {references_skipped}")
    if skipped_references:
        report()
        report("Skipped reference images (detailed):")
        for u_path, reason in skipped_references:
            report(f"  - {u_path.name}")
            report(f"    Path: {u_path}")
            report(f"    Reason: {reason}")
    report(f"  Trials requested: {total_trials_requested}")
    report(f"  Trials completed: {total_trials_actual} ({len(results)} successful comparisons)")
    if total_trials_requested > 0 and total_trials_actual < total_trials_requested:
        report(f"  NOTE: {total_trials_requested - total_trials_actual} trials could not be completed "
               f"due to insufficient valid test images")
    report()
    report(f"To reproduce this run, use: --seed {seed}")

    if verbose and results:
        report()
        report("Verbose Comparison Details:")
        report("  (sorted by increasing resolution)")
        for r in results:
            report(
                "  "
                f"{r['resolution']:+.6f} "
                f"(d1={r['d1']:.6f}, d2={r['d2']:.6f}) | "
                f"u={r['u_name']} | v={r['v_name']}"
            )
    
    # Step 6: Report privacy violations and histogram
    report()
    report("=" * 60)
    report("Privacy Analysis")
    report("=" * 60)
    
    if results:
        # Group results by reference image to detect violations
        results_by_reference: dict[str, list[dict]] = {}
        for r in results:
            u_key = r["u"]
            if u_key not in results_by_reference:
                results_by_reference[u_key] = []
            results_by_reference[u_key].append(r)
        
        # A privacy violation occurs if any comparison for an image has resolution > 0
        # Track the worst (max) violation for each reference image
        violations = []
        for u_key, ref_results in results_by_reference.items():
            # Find the result with max resolution
            max_result = max(ref_results, key=lambda x: x["resolution"])
            max_res = max_result["resolution"]
            if max_res > 0:
                compound_filename = max_result["compound_filename"]
                violations.append((u_key, max_res, compound_filename))
        
        num_violations = len(violations)
        num_references = len(results_by_reference)
        
        report()
        report("Privacy Violations:")
        report(f"  {num_violations} of {num_references} reference images have at least one resolution > 0")
        report(f"  Violation rate: {num_violations / num_references * 100:.1f}%")
        if references_skipped > 0:
            report()
            report(f"  Note: {references_skipped} reference image(s) could not be analyzed (see 'Skipped reference images (detailed)' in Summary above)")
        
        if violations:
            report()
            report("  Images with violations (showing max resolution):")
            # Sort by max resolution (worst violations first)
            violations.sort(key=lambda x: x[1], reverse=True)
            for u_key, max_res, compound_filename in violations[:10]:  # Show top 10
                u_name = Path(u_key).name
                report(f"    {u_name}: max resolution = {max_res:+.4f} ({compound_filename})")
            if len(violations) > 10:
                report(f"    ... and {len(violations) - 10} more")
        
        # Create histogram of all resolutions
        all_resolutions = [r["resolution"] for r in results]
        report()
        report(f"Resolution Histogram ({len(all_resolutions)} total comparisons):")
        histogram_lines = format_resolution_histogram(all_resolutions)
        for line in histogram_lines:
            report(line)
    else:
        report()
        report("No results to analyze.")
    
    # Write report to file
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    
    print(f"\nReport saved to: {report_file}")
    if write_analysis_artifacts:
        try:
            artifacts = generate_analysis_artifacts(
                output_folder,
                title=f"Contrastive Privacy Analysis: {output_folder.name}",
                threshold=0.0,
                top_n=6,
                compute_similarity=True,
                device=device,
                batch_size=embed_batch_size,
                image_model=embedder_model,
                image_folder=image_folder,
                refresh=True,
            )
            print(f"Analysis page saved to: {artifacts['html_path']}")
            print(f"Analysis bundle saved to: {artifacts['json_path']}")
        except Exception as exc:
            print(f"WARNING: Failed to generate analysis artifacts: {exc}")


def main() -> None:
    """CLI entry point for resolution analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze effective resolution of image obfuscation methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze face redaction with 5 trials per image
  %(prog)s ./images --objects face --mode redact --trials 5 --output ./results
  
  # Redact with very heavy blur (more privacy)
  %(prog)s ./images --objects face --mode redact --trials 5 --redact-blur 80 --output ./results
  
  # Analyze face and person replacement with 10 trials
  %(prog)s ./photos --objects face person --mode replace --trials 10 \\
      --replace-prompt "a different person" --output ./results
  
  # Sample only 10 reference images from the folder
  %(prog)s ./images --objects face --mode redact --trials 3 --samples 10 --output ./results
  
  # Use CLIPSeg (faster) with lower detection threshold
  %(prog)s ./images --objects face --mode redact --trials 5 --segmenter clipseg \\
      --threshold 0.3 --output ./results
  
  # ADAPTIVE BLUR: Scale blur based on object size (great for blackout)
  %(prog)s ./images --objects person --mode blackout --trials 5 --adaptive-blur --output ./results
  
  # ADAPTIVE BLUR: Increase blur intensity for stronger obfuscation
  %(prog)s ./images --objects face --mode blackout --trials 5 --adaptive-blur \\
      --blur-scale 1.5 --output ./results
  
  # SEQUENTIAL LABELS: Process each object separately for additive detection
  # Use this if adding more objects reduces coverage instead of increasing it
  %(prog)s ./images --objects face person --mode redact --trials 5 \\
      --sequential-labels --output ./results
  
  # CONVEX HULL: Expand masks to convex hull to hide object silhouettes
  %(prog)s ./images --objects person --mode blackout --trials 5 \\
      --convex-hull --output ./results
  
  # SAM3: Use SAM3 for state-of-the-art text-prompted segmentation
  %(prog)s ./images --objects face person --mode redact --trials 5 \\
      --segmenter sam3 --output ./results
  
  # GPT-5.2: Use GPT-5.2 vision API for polygon detection and blackout
  %(prog)s ./images --objects face person --mode blackout --trials 5 \\
      --segmenter vlm-bounding-box --output ./results

  # Retry obfuscation for skipped reference images (uses params from output folder)
  %(prog)s --retry --output ./results

  # Retry with overrides (e.g. lower threshold)
  %(prog)s ./images --retry --output ./results --threshold 0.3

Output folder structure:
  The --output folder will contain:
    - comparisons/: Compound images showing X(u), v, and X(v) side by side
    - obfuscated/: Individual obfuscated versions of each input image
    - summary_obfuscated.jpg: Grid of all obfuscated images
    - summary_originals.jpg: Grid of original images (same positions)
    - results.csv: Resolution analysis results
    - params.json: All parameters used (for reproducibility)
    - report.txt: Summary report with privacy analysis

Resolution interpretation:
  The resolution (d2 - d1) measures how well the obfuscation preserves
  semantic relationships between images while obscuring identity:
  
  - d1 = 1 - similarity(X(u), v)
    Distance from obfuscated to another image's original
  
  - d2 = 1 - similarity(X(u), X(v))
    Distance between obfuscated versions
  
  Negative resolution means obfuscated images are MORE similar to each
  other than originals are to other obfuscated images (good for privacy).
        """,
    )
    
    parser.add_argument(
        "image_folder",
        type=str,
        nargs="?",
        default=None,
        help="Path to folder containing images to analyze. With --retry, defaults to the path stored in params.json.",
    )
    parser.add_argument(
        "--objects", "-o",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="Objects to detect and obfuscate (e.g., 'face' 'person' 'car').",
    )
    parser.add_argument(
        "--base-concepts",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help=(
            "Optional base concepts (words/phrases) to obfuscate using SAM3 after the "
            "primary obfuscation (selected by --segmenter) is applied to --objects / "
            "--privacy-concept."
        ),
    )
    parser.add_argument(
        "--privacy-concept",
        type=str,
        default=None,
        help="When --segmenter ai-gen: privacy concept to redact from the image (e.g., 'identity', 'sensitive information'). "
             "If provided, --objects is not needed.",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["replace", "redact", "blackout"],
        required=True,
        help="Obfuscation mode: 'replace' (AI inpainting), 'redact' (blur), or 'blackout' (black pixels).",
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        required=True,
        help="Number of test images to sample per reference image.",
    )
    parser.add_argument(
        "--output", "-O",
        type=str,
        required=True,
        help="Output folder. Will contain: comparisons/, obfuscated/, summary images, results.csv, params.json, and report.txt.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_from_output",
        action="store_true",
        help="Resume from existing output folder: reuse obfuscated images in output/obfuscated/ and only obfuscate missing images. Use after a run that failed due to obfuscation failure.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry obfuscation for skipped reference images listed in report.txt. Uses parameters from params.json in the output folder; command-line arguments override. Requires --output to point to a previous run.",
    )
    parser.add_argument(
        "--skip-comparisons",
        dest="generate_comparisons",
        action="store_false",
        help="Skip creating compound comparison images while still computing resolution metrics.",
    )
    parser.set_defaults(generate_comparisons=True)
    parser.add_argument(
        "--comparison-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Only create comparison images for the top N highest-resolution comparisons. Metrics are still computed for all comparisons.",
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=None,
        help="Number of reference images to sample (default: use all images).",
    )
    parser.add_argument(
        "--replace-prompt",
        type=str,
        default="a different object",
        help="Prompt for replacement mode (default: 'a different object').",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=f"Minimum mask coverage fraction (0.0-1.0) for valid obfuscation (default: {DEFAULT_MIN_COVERAGE}).",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=DEFAULT_MAX_COVERAGE,
        help=f"Maximum mask coverage fraction (0.0-1.0) for valid obfuscation (default: {DEFAULT_MAX_COVERAGE}). "
             "Images requiring more than this fraction to be altered are discarded.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect).",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="groundedsam",
        choices=["groundedsam", "clipseg", "sam3", "ai-gen", "vlm-bounding-box"],
        help=(
            "Segmentation model: 'groundedsam' (GroundingDINO+SAM, better quality), "
            "'clipseg' (faster), 'sam3' (state-of-the-art text-prompted segmentation), "
            "'ai-gen' (fal.ai image edit API, handles both detection and obfuscation), "
            "or 'vlm-bounding-box' (fal.ai OpenRouter vision for polygon detection + blackout). "
            "Default: groundedsam."
        ),
    )
    parser.add_argument(
        "--fal-image-model",
        type=str,
        default="gpt-image-1.5",
        help="When --segmenter ai-gen: fal.ai image edit model. Pass the model path (e.g. fal-ai/gpt-image-1) "
             "or a short name: gpt-image-1.5, nano-banana-pro, flux-2-pro, flux-2-dev. Default: gpt-image-1.5. Requires FAL_KEY.",
    )
    parser.add_argument(
        "--fal-vision-model",
        type=str,
        default=DEFAULT_FAL_VISION_MODEL,
        choices=["gpt-5.4", "gemini-3.1-pro", "opus-4.6"],
        help=f"When --segmenter vlm-bounding-box: which fal.ai OpenRouter vision model to use. Default: {DEFAULT_FAL_VISION_MODEL}. Requires FAL_KEY.",
    )
    parser.add_argument(
        "--fal-vision-temperature",
        type=float,
        default=DEFAULT_FAL_VISION_TEMPERATURE,
        help=f"When --segmenter vlm-bounding-box: OpenRouter sampling temperature for vision calls "
             f"(default: {DEFAULT_FAL_VISION_TEMPERATURE}).",
    )
    # Anonymization parameters (passed through to anonymize.py)
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.4],
        help="Detection threshold (default: 0.4). Lower = more inclusive. "
             "Can specify multiple thresholds (one per object). "
             "For clipseg: per-label thresholds are applied natively. "
             "For groundedsam: requires --sequential-labels to use per-label thresholds. "
             "If multiple thresholds are provided, they must match the number of --objects.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=5,
        help="Pixels to expand detected regions (default: 5).",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=8,
        help="Blur radius for mask edges (default: 8). Reduces harsh seams.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="Inpainting strength (default: 0.85). Lower = better blending.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="schnell",
        choices=["schnell", "dev"],
        help="FLUX model: 'schnell' (fast) or 'dev' (better quality). Default: schnell.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Inference steps (default: 28).",
    )
    parser.add_argument(
        "--redact-blur",
        type=int,
        default=30,
        help="Blur radius for redaction mode (default: 30). Higher = more blurred.",
    )
    parser.add_argument(
        "--adaptive-blur",
        action="store_true",
        help="Scale mask blur and dilation based on object size. "
             "Larger objects get more blur to effectively hide their shape. "
             "Particularly useful for --mode blackout. When enabled, --dilate "
             "and --blur are ignored; use --blur-scale to control intensity.",
    )
    parser.add_argument(
        "--blur-scale",
        type=float,
        default=1.0,
        help="Blur intensity when using --adaptive-blur (default: 1.0). "
             "Higher values (e.g., 1.5, 2.0) increase blur proportionally.",
    )
    parser.add_argument(
        "--size-exponent",
        type=float,
        default=1.0,
        help="Controls size-dependence of blur when using --adaptive-blur. "
             "1.0 (default) = linear scaling (larger objects get more blur). "
             "0.0 = same blur for all objects regardless of size. "
             "0.5 = square root scaling (less difference between sizes).",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Constant multiplier for --adaptive-blur (default: 1.0). "
             "Formula: effective_size = scaling_factor × char_size^exponent.",
    )
    parser.add_argument(
        "--sequential-labels",
        action="store_true",
        help="Process each object label separately and combine masks (only for groundedsam). "
             "This ensures strictly additive behavior: adding more objects to detect can never "
             "reduce the total obfuscated area. Use this when you observe that adding objects "
             "actually reduces coverage. Increases runtime proportionally with the number of labels.",
    )
    parser.add_argument(
        "--convex-hull",
        action="store_true",
        help="Expand each detected object's mask to its convex hull. "
             "This fills in concave regions (e.g., space between arms and body), "
             "better hiding the object's silhouette. Particularly useful with "
             "--mode blackout to prevent shape recognition.",
    )
    parser.add_argument(
        "--skip-empty-labels",
        action="store_true",
        help="Skip obfuscation of objects where GroundingDINO returned an empty label (''). "
             "This filters out ambiguous detections that couldn't be assigned a specific label. "
             "Only applies when using --segmenter groundedsam.",
    )
    parser.add_argument(
        "--refinements",
        type=int,
        default=0,
        help="Number of refinement passes for fal.ai vision polygon detection (default: 0). "
             "Only applies when using --segmenter vlm-bounding-box. Each refinement pass shows "
             "the vision model its previous polygon detections and asks it to improve the boundaries.",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=DEFAULT_EMBEDDER_MODEL,
        help=(
            "HuggingFace model ID for image embeddings in resolution similarity "
            f"(default: {DEFAULT_EMBEDDER_MODEL}). "
            f"EVA-02-CLIP example: {EVA02_CLIP_EMBEDDER_MODEL} (needs einops). "
            "Same as compare-images --model."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write detailed per-comparison values to report.txt (resolution with d1 and d2).",
    )
    parser.add_argument(
        "--skip-analysis-artifacts",
        dest="write_analysis_artifacts",
        action="store_false",
        help="Do not auto-generate analysis_report.html and analysis_report.json at the end of the run.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Batch size when pre-computing image embeddings (Step 3). "
            "Large models on ~16GB GPUs often need 1-2. Default: 8."
        ),
    )
    
    args = parser.parse_args()
    argv = sys.argv[1:]

    # Retry mode: load params and skipped paths from report, merge with CLI overrides, then run
    if args.retry:
        output_folder = Path(args.output)
        if not output_folder.exists():
            parser.error(f"Output folder not found: {output_folder}")
        try:
            params = load_params_from_output(output_folder)
        except FileNotFoundError as e:
            parser.error(str(e))
        report_file = output_folder / "report.txt"
        if not report_file.exists():
            parser.error(f"Report not found: {report_file}. Run without --retry first.")
        skipped_paths = parse_report_skipped_paths(report_file)
        if not skipped_paths:
            print("No skipped reference images found in report.txt. Nothing to retry.")
            return
        print(f"Retry mode: found {len(skipped_paths)} skipped reference image(s) in report.txt")
        # Merge: use params from file; override with args when the option was passed on the command line
        def _use(from_args, from_params, option_strings):
            return from_args if _argv_contains_any(argv, option_strings) else from_params

        image_folder = Path(args.image_folder or params["image_folder"])
        objects = _use(args.objects, params.get("objects"), ["--objects", "-o"]) or params.get("objects") or []
        privacy_concept = _use(args.privacy_concept, params.get("privacy_concept"), ["--privacy-concept"]) or params.get("privacy_concept")
        base_concepts = _use(args.base_concepts, params.get("base_concepts"), ["--base-concepts"]) or params.get("base_concepts")
        mode = _use(args.mode, params["mode"], ["--mode", "-m"]) or params["mode"]
        trials = _use(args.trials, params["trials"], ["--trials", "-t"]) or params["trials"]
        samples = _use(args.samples, params.get("samples"), ["--samples", "-s"])
        if samples is None and "samples" in params:
            samples = params["samples"]
        min_coverage = _use(args.min_coverage, params["min_coverage"], ["--min-coverage"])
        max_coverage = _use(args.max_coverage, params["max_coverage"], ["--max-coverage"])
        replacement_prompt_merged = _use(args.replace_prompt, params.get("replacement_prompt", "a different object"), ["--replace-prompt"])
        seed = _use(args.seed, params.get("seed"), ["--seed"])
        device = _use(args.device, params.get("device"), ["--device"])
        segmenter = _use(args.segmenter, params["segmenter"], ["--segmenter"])
        threshold_merged = _use(args.threshold, params["threshold"], ["--threshold"])
        dilate = _use(args.dilate, params["dilate"], ["--dilate"])
        blur = _use(args.blur, params["blur"], ["--blur"])
        strength = _use(args.strength, params["strength"], ["--strength"])
        model = _use(args.model, params["model"], ["--model"])
        num_inference_steps = _use(args.steps, params["num_inference_steps"], ["--steps"])
        redact_blur_radius = _use(args.redact_blur, params["redact_blur_radius"], ["--redact-blur"])
        adaptive_blur = _use(args.adaptive_blur, params.get("adaptive_blur", False), ["--adaptive-blur"])
        blur_scale = _use(args.blur_scale, params.get("blur_scale", 1.0), ["--blur-scale"])
        size_exponent = _use(args.size_exponent, params.get("size_exponent", 1.0), ["--size-exponent"])
        scaling_factor = _use(args.scaling_factor, params.get("scaling_factor", 1.0), ["--scaling-factor"])
        sequential_labels = _use(args.sequential_labels, params.get("sequential_labels", False), ["--sequential-labels"])
        convex_hull = _use(args.convex_hull, params.get("convex_hull", False), ["--convex-hull"])
        skip_empty_labels = _use(args.skip_empty_labels, params.get("skip_empty_labels", False), ["--skip-empty-labels"])
        refinements = _use(args.refinements, params.get("refinements", 0), ["--refinements"])
        generate_comparisons = _use(
            args.generate_comparisons,
            params.get("generate_comparisons", True),
            ["--skip-comparisons"],
        )
        comparison_top_n = _use(
            args.comparison_top_n,
            params.get("comparison_top_n"),
            ["--comparison-top-n"],
        )
        fal_image_model = _use(args.fal_image_model, params.get("fal_image_model", "gpt-image-1.5"), ["--fal-image-model"])
        fal_vision_model = _use(args.fal_vision_model, params.get("fal_vision_model", DEFAULT_FAL_VISION_MODEL), ["--fal-vision-model"])
        fal_vision_temperature = _use(
            args.fal_vision_temperature,
            params.get("fal_vision_temperature", DEFAULT_FAL_VISION_TEMPERATURE),
            ["--fal-vision-temperature"],
        )
        embedder_model = _use(
            args.embedder_model,
            params.get("embedder_model", DEFAULT_EMBEDDER_MODEL),
            ["--embedder-model"],
        )
        embed_batch_size = _use(
            args.embed_batch_size,
            params.get("embed_batch_size", 8),
            ["--embed-batch-size"],
        )
        verbose = _use(
            args.verbose,
            params.get("verbose", False),
            ["--verbose"],
        )
        write_analysis_artifacts = _use(
            args.write_analysis_artifacts,
            params.get("write_analysis_artifacts", True),
            ["--skip-analysis-artifacts"],
        )

        if not image_folder.exists():
            parser.error(f"Image folder not found: {image_folder}")
        if not image_folder.is_dir():
            parser.error(f"Not a directory: {image_folder}")
        if trials < 1:
            parser.error("--trials must be at least 1")
        if samples is not None and samples < 1:
            parser.error("--samples must be at least 1")
        if not (0.0 <= min_coverage <= 1.0):
            parser.error("--min-coverage must be between 0.0 and 1.0")
        if not (0.0 <= max_coverage <= 1.0):
            parser.error("--max-coverage must be between 0.0 and 1.0")
        if min_coverage > max_coverage:
            parser.error("--min-coverage cannot be greater than --max-coverage")
        if embed_batch_size < 1:
            parser.error("--embed-batch-size must be at least 1")
        if comparison_top_n is not None and comparison_top_n < 1:
            parser.error("--comparison-top-n must be at least 1")
        if mode == "replace" and not replacement_prompt_merged:
            parser.error("--replace-prompt is required when using --mode replace")

        if privacy_concept is not None and segmenter != "ai-gen":
            parser.error("--privacy-concept is only valid when --segmenter ai-gen")
        if segmenter == "ai-gen":
            if privacy_concept is not None and args.objects is not None:
                parser.error("Use either --privacy-concept or --objects with --segmenter ai-gen (not both).")
            if privacy_concept is None and not objects:
                parser.error("When --segmenter ai-gen, you must provide either --objects or --privacy-concept.")
        else:
            if privacy_concept is not None:
                parser.error("--privacy-concept requires --segmenter ai-gen.")
            if not objects:
                parser.error("--objects is required unless --segmenter ai-gen with --privacy-concept.")

        using_privacy_concept = segmenter == "ai-gen" and privacy_concept is not None
        if isinstance(threshold_merged, list):
            if len(threshold_merged) == 1:
                threshold_value = threshold_merged[0]
            else:
                if using_privacy_concept:
                    print(f"WARNING: Multiple thresholds provided with --segmenter ai-gen and --privacy-concept; using first threshold {threshold_merged[0]}.")
                    threshold_value = threshold_merged[0]
                elif len(threshold_merged) != len(objects):
                    parser.error(
                        f"Number of thresholds ({len(threshold_merged)}) must match number of objects ({len(objects)})."
                    )
                else:
                    threshold_value = threshold_merged
            if segmenter == "groundedsam" and not sequential_labels and len(threshold_merged) > 1:
                threshold_value = threshold_merged[0]
        else:
            threshold_value = threshold_merged
        replacement_prompt_final = replacement_prompt_merged if mode == "replace" else None

        run_resolution_analysis(
            image_folder=image_folder,
            objects=objects,
            privacy_concept=privacy_concept,
            base_concepts=base_concepts,
            mode=mode,
            trials=trials,
            output_folder=output_folder,
            samples=samples,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            replacement_prompt=replacement_prompt_final,
            seed=seed,
            device=device,
            segmenter=segmenter,
            threshold=threshold_value,
            dilate=dilate,
            blur=blur,
            strength=strength,
            model=model,
            num_inference_steps=num_inference_steps,
            redact_blur_radius=redact_blur_radius,
            adaptive_blur=adaptive_blur,
            blur_scale=blur_scale,
            size_exponent=size_exponent,
            scaling_factor=scaling_factor,
            sequential_labels=sequential_labels,
            convex_hull=convex_hull,
            skip_empty_labels=skip_empty_labels,
            refinements=refinements,
            continue_from_output=True,
            generate_comparisons=generate_comparisons,
            comparison_top_n=comparison_top_n,
            fal_image_model=fal_image_model,
            fal_vision_model=fal_vision_model,
            fal_vision_temperature=fal_vision_temperature,
            command_line=shlex.join(sys.argv),
            retry_skipped_paths=skipped_paths,
            embedder_model=embedder_model,
            embed_batch_size=embed_batch_size,
            verbose=verbose,
            write_analysis_artifacts=write_analysis_artifacts,
        )
        return

    # Validate arguments
    if not args.image_folder:
        parser.error("the following arguments are required: image_folder")
    image_folder = Path(args.image_folder)
    if not image_folder.exists():
        parser.error(f"Image folder not found: {image_folder}")
    if not image_folder.is_dir():
        parser.error(f"Not a directory: {image_folder}")
    
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    
    if args.samples is not None and args.samples < 1:
        parser.error("--samples must be at least 1")
    
    if not (0.0 <= args.min_coverage <= 1.0):
        parser.error("--min-coverage must be between 0.0 and 1.0")
    
    if not (0.0 <= args.max_coverage <= 1.0):
        parser.error("--max-coverage must be between 0.0 and 1.0")
    
    if args.min_coverage > args.max_coverage:
        parser.error("--min-coverage cannot be greater than --max-coverage")
    if args.embed_batch_size < 1:
        parser.error("--embed-batch-size must be at least 1")
    if args.comparison_top_n is not None and args.comparison_top_n < 1:
        parser.error("--comparison-top-n must be at least 1")
    
    if args.mode == "replace" and not args.replace_prompt:
        parser.error("--replace-prompt is required when using --mode replace")

    objects = args.objects or []
    privacy_concept = args.privacy_concept
    base_concepts = args.base_concepts

    if privacy_concept is not None and args.segmenter != "ai-gen":
        parser.error("--privacy-concept is only valid when --segmenter ai-gen")
    if args.segmenter == "ai-gen":
        if privacy_concept is not None:
            if args.objects is not None:
                parser.error("Use either --privacy-concept or --objects with --segmenter ai-gen (not both).")
        else:
            if not objects:
                parser.error("When --segmenter ai-gen, you must provide either --objects or --privacy-concept.")
    else:
        if privacy_concept is not None:
            parser.error("--privacy-concept requires --segmenter ai-gen.")
        if not objects:
            parser.error("--objects is required unless --segmenter ai-gen with --privacy-concept.")

    using_privacy_concept = args.segmenter == "ai-gen" and privacy_concept is not None
    
    # Normalize threshold: convert list to single float if only one value, or validate list length
    threshold_value = args.threshold
    if isinstance(threshold_value, list):
        if len(threshold_value) == 1:
            # Single threshold provided as list, convert to float
            threshold_value = threshold_value[0]
        else:
            if using_privacy_concept:
                print(f"WARNING: Multiple thresholds provided with --segmenter ai-gen and --privacy-concept; using first threshold {threshold_value[0]}.")
                threshold_value = threshold_value[0]
            elif len(threshold_value) != len(objects):
                parser.error(
                    f"Number of thresholds ({len(threshold_value)}) must match number of objects "
                    f"({len(objects)}). Got thresholds: {threshold_value}, objects: {objects}"
                )
            else:
                threshold_value = threshold_value
        # For groundedsam without sequential_labels, multiple thresholds won't work
        # (CLIPSeg and groundedsam with sequential_labels both support per-label thresholds)
        if args.segmenter == "groundedsam" and not args.sequential_labels and len(args.threshold) > 1:
            print(f"WARNING: Multiple thresholds provided but --sequential-labels is not enabled. "
                  f"For groundedsam, only the first threshold ({args.threshold[0]}) will be used. "
                  f"Use --sequential-labels to enable per-label thresholds, or use --segmenter clipseg "
                  f"which supports per-label thresholds natively.")
            threshold_value = args.threshold[0]
    
    # Only pass replacement_prompt for replace mode
    replacement_prompt = args.replace_prompt if args.mode == "replace" else None
    
    # Run analysis
    run_resolution_analysis(
        image_folder=image_folder,
        objects=objects,
        privacy_concept=privacy_concept,
        base_concepts=base_concepts,
        mode=args.mode,
        trials=args.trials,
        output_folder=Path(args.output),
        samples=args.samples,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        replacement_prompt=replacement_prompt,
        seed=args.seed,
        device=args.device,
        segmenter=args.segmenter,
        threshold=threshold_value,
        dilate=args.dilate,
        blur=args.blur,
        strength=args.strength,
        model=args.model,
        num_inference_steps=args.steps,
        redact_blur_radius=args.redact_blur,
        adaptive_blur=args.adaptive_blur,
        blur_scale=args.blur_scale,
        size_exponent=args.size_exponent,
        scaling_factor=args.scaling_factor,
        sequential_labels=args.sequential_labels,
        convex_hull=args.convex_hull,
        skip_empty_labels=args.skip_empty_labels,
        refinements=args.refinements,
        continue_from_output=args.continue_from_output,
        generate_comparisons=args.generate_comparisons,
        comparison_top_n=args.comparison_top_n,
        fal_image_model=args.fal_image_model,
        fal_vision_model=args.fal_vision_model,
        fal_vision_temperature=args.fal_vision_temperature,
        command_line=shlex.join(sys.argv),
        embedder_model=args.embedder_model,
        embed_batch_size=args.embed_batch_size,
        verbose=args.verbose,
        write_analysis_artifacts=args.write_analysis_artifacts,
    )


if __name__ == "__main__":
    main()


