#!/usr/bin/env python3
"""
Concept resolution analysis for measuring distinguishability between two concepts.

This script measures the resolution necessary to distinguish between two concepts c1 and c2
by analyzing how obfuscation affects the semantic distance between images.

For each pair of images (u, v) where u is from folder F1 (exhibiting concept c1) and
v is from folder F2 (exhibiting concept c2), the script computes:
    - d1 = 1 - similarity(u, v)        Distance between original images
    - d2 = 1 - similarity(u, X(v))     Distance from u to obfuscated v (v obfuscated for c2)
    - d3 = 1 - similarity(X(u), v)     Distance from obfuscated u to v (u obfuscated for c1)

The script reports the distributions of:
    - d2 - d1: How obfuscating c2 in F2 images affects their distance from F1 images
    - d3 - d1: How obfuscating c1 in F1 images affects their distance to F2 images

Example:
    concept-resolution --folder1 ./faces --folder2 ./bodies \\
        --concept1 face --concept2 person --mode redact --output ./results

    concept-resolution --folder1 ./faces --folder2 ./bodies \\
        --concept1 face --concept2 person --mode redact \\
        --embedder-model BAAI/EVA-CLIP-18B --output ./results

    concept-resolution --histogram-only --output ./results --bins 30
"""

import argparse
import csv
import json
import math
import random
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from contrastive_privacy.scripts.anonymize import (
    CLIPSegModels,
    GroundedSAMModels,
    SAM3Models,
    anonymize,
    load_clipseg_models,
    load_groundedsam_models,
    load_sam3_models,
)
from contrastive_privacy.scripts.compare_images import (
    EVA02_CLIP_EMBEDDER_MODEL,
    compute_embeddings_batch,
    load_clip_model,
    similarity_from_embeddings,
)
from contrastive_privacy.scripts.compare_texts import (
    DEFAULT_QWEN_EMBEDDER_MODEL,
    load_text_embedder,
)
from contrastive_privacy.scripts.recognize_entities import GLiNER2Recognizer, load_recognizer
from contrastive_privacy.scripts.text_anonymize import anonymize_text


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Default HuggingFace model for image similarity (same as resolution_analysis)
DEFAULT_EMBEDDER_MODEL = "apple/DFN5B-CLIP-ViT-H-14-378"
EVA_CLIP_IMAGE_EMBEDDER_EXAMPLE = "BAAI/EVA-CLIP-18B"

# Text embedding defaults (match text_resolution_analysis / similarity_analysis)
DEFAULT_TEXT_SBERT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default minimum coverage threshold (as a fraction, not percentage)
DEFAULT_MIN_COVERAGE = 0.001  # 0.1%

# Default maximum coverage threshold (as a fraction, not percentage)
DEFAULT_MAX_COVERAGE = 1.0  # 100% (no limit by default)


def get_image_files(folder: Path) -> list[Path]:
    """Get all image files from a folder."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(images)


def split_text_into_sentences(text: str) -> list[str]:
    """Split raw text into non-empty sentences using simple punctuation boundaries."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    parts = SENTENCE_SPLIT_RE.split(normalized)
    return [p.strip() for p in parts if p and p.strip()]


def create_obfuscated_text_sentence(
    sentence: str,
    target_labels: list[str],
    mode: str,
    recognizer: Optional[GLiNER2Recognizer] = None,
    threshold: float = 0.3,
    sequential_labels: bool = False,
    propagate: bool = True,
    placeholder: str = "[REDACTED]",
) -> tuple[str, float]:
    """Obfuscate one sentence and return (obfuscated_sentence, coverage_percent)."""
    if mode not in {"redact", "blackout"}:
        raise ValueError(
            f"Text inputs support --mode redact/blackout only (got {mode!r})."
        )
    result = anonymize_text(
        text=sentence,
        entity_types=target_labels,
        mode=mode,
        placeholder=placeholder,
        threshold=threshold,
        recognizer=recognizer,
        sequential_labels=sequential_labels,
        propagate=propagate,
    )
    return result.anonymized_text, result.coverage * 100.0


def create_obfuscated_image(
    input_path: Path,
    output_path: Path,
    target_labels: list[str],
    mode: str,
    replacement_prompt: Optional[str] = None,
    device: Optional[str] = None,
    segmenter: str = "sam3",
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
) -> tuple[Path, float]:
    """
    Create an obfuscated version of an image.
    
    Args:
        input_path: Path to the original image.
        output_path: Path where the obfuscated image will be saved.
        target_labels: Objects to detect and obfuscate.
        mode: "redact" (blur), "blackout" (black pixels), or "replace" (inpaint).
        replacement_prompt: Prompt for replacement mode.
        device: Device to run on.
        segmenter: Segmentation model ("sam3", "groundedsam", "clipseg", "openai-gen", or "gpt-5.2").
        threshold: Detection threshold (0.0-1.0).
        dilate: Pixels to expand the mask.
        blur: Blur radius for mask edges.
        strength: Inpainting strength (0.0-1.0).
        model: FLUX model ("schnell" or "dev").
        num_inference_steps: Number of inference steps.
        redact_blur_radius: Blur radius for redaction mode.
        seed: Random seed for reproducibility.
        clipseg_models: Pre-loaded CLIPSeg models (optional).
        groundedsam_models: Pre-loaded GroundedSAM models (optional).
        sam3_models: Pre-loaded SAM3 models (optional).
        adaptive_blur: If True, scale blur/dilation based on object size.
        blur_scale: When adaptive_blur=True, controls overall blur intensity.
        size_exponent: Controls size-dependence.
        scaling_factor: Constant multiplier on effective size.
        sequential_labels: If True, process each label separately and combine masks.
        convex_hull: If True, expand each object's mask to its convex hull.
        skip_empty_labels: If True, skip obfuscation of objects with empty labels.
        refinements: Number of refinement passes for GPT-5.2 polygon detection.
    
    Returns:
        Tuple of (path to the obfuscated image, coverage percentage).
    """
    _, coverage = anonymize(
        input_path=input_path,
        output_path=output_path,
        target_labels=target_labels,
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
    )
    return output_path, coverage


def create_summary_grid(
    image_paths: list[Path],
    output_path: Path,
    thumb_size: int = 256,
) -> Path:
    """
    Create a summary grid image from multiple images.
    
    The grid dimensions are calculated to be as close to square as possible.
    
    Args:
        image_paths: List of paths to images to include in the grid.
        output_path: Path where the grid image will be saved.
        thumb_size: Size of each thumbnail (square).
    
    Returns:
        Path to the created grid image.
    """
    if not image_paths:
        raise ValueError("No images to create grid from")
    
    # Calculate grid dimensions to be as close to square as possible
    num_images = len(image_paths)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    
    # Create the grid image
    grid_width = cols * thumb_size
    grid_height = rows * thumb_size
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    
    # Place each image in the grid
    for idx, img_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            
            # Center the thumbnail in its cell
            x_offset = col * thumb_size + (thumb_size - img.width) // 2
            y_offset = row * thumb_size + (thumb_size - img.height) // 2
            
            grid.paste(img, (x_offset, y_offset))
        except Exception as e:
            print(f"  Warning: Could not add {img_path.name} to grid: {e}")
            continue
    
    grid.save(output_path, quality=90)
    return output_path


def format_cumulative_distribution(
    values: list[float],
    num_bins: int = 20,
    width: int = 50,
    title: str = "Cumulative Distribution",
) -> list[str]:
    """
    Format a text-based cumulative distribution of values.
    
    Args:
        values: List of values.
        num_bins: Number of bins for the distribution.
        width: Width of the bars in characters.
        title: Title for the distribution.
    
    Returns:
        List of formatted lines for the cumulative distribution.
    """
    lines = []
    lines.append(f"{title}:")
    lines.append("")
    
    if not values:
        lines.append("  No data to display.")
        return lines
    
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        lines.append(f"  All {len(values)} values are {min_val:.4f}")
        return lines
    
    # Compute standard deviation
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_val = math.sqrt(variance)
    
    # Create bins
    bin_width = (max_val - min_val) / num_bins
    bins = [0] * num_bins
    
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        bins[bin_idx] += 1
    
    # Statistics
    lines.append(f"  Min: {min_val:+.4f}  Max: {max_val:+.4f}  Mean: {mean_val:+.4f}")
    sorted_values = sorted(values)
    median_val = sorted_values[len(sorted_values) // 2]
    lines.append(f"  Median: {median_val:+.4f}  Std: {std_val:.4f}")
    lines.append("")
    
    # Count positive and negative
    num_positive = sum(1 for v in values if v > 0)
    num_negative = sum(1 for v in values if v < 0)
    num_zero = sum(1 for v in values if v == 0)
    lines.append(f"  Negative: {num_negative} ({num_negative / len(values) * 100:.1f}%)")
    lines.append(f"  Zero:     {num_zero} ({num_zero / len(values) * 100:.1f}%)")
    lines.append(f"  Positive: {num_positive} ({num_positive / len(values) * 100:.1f}%)")
    lines.append("")
    
    # Cumulative distribution bars
    total_count = len(values)
    cumulative_count = 0
    
    for i, count in enumerate(bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        cumulative_count += count
        cumulative_pct = cumulative_count / total_count * 100 if total_count > 0 else 0
        bar_length = int(cumulative_pct / 100 * width)
        bar = "█" * bar_length
        
        # Mark if this bin contains zero
        zero_marker = " *" if bin_start <= 0 < bin_end else "  "
        
        lines.append(f"  [{bin_start:+.3f}, {bin_end:+.3f}){zero_marker} {bar} {cumulative_pct:5.1f}%")
    
    lines.append("")
    lines.append("  (* indicates bin containing zero)")
    
    return lines


def run_concept_resolution_analysis(
    input1: Path,
    input2: Path,
    concept1: list[str],
    concept2: list[str],
    mode: str,
    output_folder: Path,
    samples1: Optional[int] = None,
    samples2: Optional[int] = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_coverage: float = DEFAULT_MAX_COVERAGE,
    replacement_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    segmenter: str = "sam3",
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
    num_bins: int = 20,
    embedder_model: str = DEFAULT_EMBEDDER_MODEL,
    text_embedder_type: str = "clip",
    text_embedder_quantization: Optional[str] = None,
    embed_batch_size: int = 8,
) -> None:
    """
    Run concept resolution analysis between two inputs.
    
    For each pair (u, v) where u is from folder1 and v is from folder2, computes:
        - d1 = 1 - similarity(u, v)
        - d2 = 1 - similarity(u, X(v))  (v obfuscated for concept2)
        - d3 = 1 - similarity(X(u), v)  (u obfuscated for concept1)
    
    Reports distributions of d3-d1 and d3-d2.
    
    Args:
        input1: Folder of images OR file of text sentences exhibiting concept1.
        input2: Folder of images OR file of text sentences exhibiting concept2.
        concept1: Objects to detect and obfuscate in folder1 images.
        concept2: Objects to detect and obfuscate in folder2 images.
        mode: "redact", "blackout", or "replace".
        output_folder: Base folder for all outputs.
        samples1: Number of images to sample from folder1 (None = use all).
        samples2: Number of images to sample from folder2 (None = use all).
        min_coverage: Minimum mask coverage fraction for valid obfuscation.
        max_coverage: Maximum mask coverage fraction for valid obfuscation.
        replacement_prompt: Prompt for replacement mode.
        seed: Random seed for reproducibility.
        device: Device to run on.
        segmenter: Segmentation model (default: sam3).
        threshold: Detection threshold (0.0-1.0).
        dilate: Pixels to expand the mask.
        blur: Blur radius for mask edges.
        strength: Inpainting strength (0.0-1.0).
        model: FLUX model ("schnell" or "dev").
        num_inference_steps: Number of inference steps.
        redact_blur_radius: Blur radius for redaction mode.
        adaptive_blur: If True, scale blur/dilation based on object size.
        blur_scale: When adaptive_blur=True, controls overall blur intensity.
        size_exponent: Controls size-dependence.
        scaling_factor: Constant multiplier on effective size.
        sequential_labels: If True, process each label separately and combine masks.
        convex_hull: If True, expand each object's mask to its convex hull.
        skip_empty_labels: If True, skip obfuscation of objects with empty labels.
        refinements: Number of refinement passes for GPT-5.2 polygon detection.
        embedder_model: For image mode: HuggingFace CLIP / EvaCLIP for image embeddings
            (same as ``compare-images --model``). For text mode: model ID for the loader chosen by
            ``text_embedder_type`` (CLIP text tower, Qwen3 embedding, or SBERT).
        text_embedder_type: For text-file inputs only: ``clip``, ``qwen``, or ``sbert``.
        text_embedder_quantization: For ``qwen`` / ``sbert`` text embedders: ``none``, ``half``,
            ``4bit``, or ``8bit`` (bitsandbytes; CUDA). Ignored for ``clip`` and for image mode.
        embed_batch_size: Batch size when embedding text sentences (text-file mode only).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle random seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {seed}")
    else:
        print(f"Using provided seed: {seed}")
    random.seed(seed)

    if embed_batch_size < 1:
        raise ValueError("embed_batch_size must be at least 1")

    is_image_mode = input1.is_dir() and input2.is_dir()
    is_text_mode = input1.is_file() and input2.is_file()
    if not (is_image_mode or is_text_mode):
        raise ValueError(
            "Inputs must both be directories (image mode) or both be files (text mode)."
        )

    items1: list[Path] = []
    items2: list[Path] = []
    text_rows1: list[tuple[str, str]] = []
    text_rows2: list[tuple[str, str]] = []
    mode_name = "image" if is_image_mode else "text"

    resolved_text_embedder_model = embedder_model
    text_quant_display = text_embedder_quantization or "none"
    if is_text_mode:
        if text_embedder_type == "qwen" and embedder_model == DEFAULT_EMBEDDER_MODEL:
            resolved_text_embedder_model = DEFAULT_QWEN_EMBEDDER_MODEL
        elif text_embedder_type == "sbert" and embedder_model == DEFAULT_EMBEDDER_MODEL:
            resolved_text_embedder_model = DEFAULT_TEXT_SBERT_EMBEDDER_MODEL

    if is_image_mode:
        items1 = get_image_files(input1)
        items2 = get_image_files(input2)
        if len(items1) == 0:
            raise ValueError(f"No images found in folder1: {input1}")
        if len(items2) == 0:
            raise ValueError(f"No images found in folder2: {input2}")
        print(f"Found {len(items1)} images in folder1: {input1}")
        print(f"Found {len(items2)} images in folder2: {input2}")
    else:
        text1 = input1.read_text(encoding="utf-8", errors="replace")
        text2 = input2.read_text(encoding="utf-8", errors="replace")
        sents1 = split_text_into_sentences(text1)
        sents2 = split_text_into_sentences(text2)
        if len(sents1) == 0:
            raise ValueError(f"No sentences found in file1: {input1}")
        if len(sents2) == 0:
            raise ValueError(f"No sentences found in file2: {input2}")
        text_rows1 = [(f"{input1.name}#sent_{i+1:04d}", s) for i, s in enumerate(sents1)]
        text_rows2 = [(f"{input2.name}#sent_{i+1:04d}", s) for i, s in enumerate(sents2)]
        print(f"Found {len(text_rows1)} sentences in file1: {input1}")
        print(f"Found {len(text_rows2)} sentences in file2: {input2}")

    print(f"Concept1 (for folder1): {', '.join(concept1)}")
    print(f"Concept2 (for folder2): {', '.join(concept2)}")
    print(f"Obfuscation mode: {mode}")
    if is_image_mode:
        print(f"Segmenter: {segmenter}")
    print(f"Device: {device}")
    if is_image_mode:
        print(f"Embedder (image): {embedder_model}")
    else:
        print(f"Text embedder type: {text_embedder_type}")
        print(f"Text embedder model: {resolved_text_embedder_model}")
        if text_embedder_type in ("qwen", "sbert"):
            print(f"Text embedder quantization: {text_quant_display}")
        print(f"Text embed batch size: {embed_batch_size}")
    print(f"Input mode: {mode_name}")
    
    # Sample images if requested
    if samples1 is not None:
        if is_image_mode and samples1 < len(items1):
            items1 = random.sample(items1, samples1)
            print(f"Sampled {samples1} images from folder1")
        if is_text_mode and samples1 < len(text_rows1):
            text_rows1 = random.sample(text_rows1, samples1)
            print(f"Sampled {samples1} sentences from file1")
    
    if samples2 is not None:
        if is_image_mode and samples2 < len(items2):
            items2 = random.sample(items2, samples2)
            print(f"Sampled {samples2} images from folder2")
        if is_text_mode and samples2 < len(text_rows2):
            text_rows2 = random.sample(text_rows2, samples2)
            print(f"Sampled {samples2} sentences from file2")
    
    total_pairs = (
        len(items1) * len(items2)
        if is_image_mode
        else len(text_rows1) * len(text_rows2)
    )
    print(f"Total pairs to analyze: {total_pairs}")
    
    # Set up output directory structure
    output_folder.mkdir(parents=True, exist_ok=True)
    obfuscated_f1_folder = output_folder / ("obfuscated_f1" if is_image_mode else "obfuscated_text_f1")
    obfuscated_f2_folder = output_folder / ("obfuscated_f2" if is_image_mode else "obfuscated_text_f2")
    obfuscated_f1_folder.mkdir(parents=True, exist_ok=True)
    obfuscated_f2_folder.mkdir(parents=True, exist_ok=True)
    output_csv = output_folder / "results.csv"
    params_file = output_folder / "params.json"
    report_file = output_folder / "report.txt"
    
    # Save parameters for reproducibility
    params = {
        "input1": str(input1.absolute()),
        "input2": str(input2.absolute()),
        "input_mode": mode_name,
        "concept1": concept1,
        "concept2": concept2,
        "mode": mode,
        "samples1": samples1,
        "samples2": samples2,
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
        "embedder_model": embedder_model if is_image_mode else resolved_text_embedder_model,
        "timestamp": datetime.now().isoformat(),
    }
    if is_text_mode:
        params["text_embedder_type"] = text_embedder_type
        params["embedder_quantization"] = text_quant_display
        params["embed_batch_size"] = embed_batch_size
        if embedder_model != resolved_text_embedder_model:
            params["embedder_model_cli"] = embedder_model
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    
    print(f"\nOutput folder: {output_folder}")
    print(f"  Results CSV: {output_csv}")
    print(f"  Parameters: {params_file}")
    print(f"  Report: {report_file}")
    print(f"  Obfuscated folder1: {obfuscated_f1_folder}")
    print(f"  Obfuscated folder2: {obfuscated_f2_folder}")
    
    # Pre-load models for efficiency
    print("\n" + "=" * 60)
    print("Loading models")
    print("=" * 60)

    clipseg_models = None
    groundedsam_models = None
    sam3_models = None
    clip_model = None
    clip_processor = None
    text_embedder = None

    if is_image_mode:
        if segmenter == "openai-gen":
            print("Using OpenAI image edit API (no models to load)")
        elif segmenter == "gpt-5.2":
            print("Using GPT-5.2 vision API for polygon detection (no models to load)")
        elif segmenter == "clipseg":
            print("Loading CLIPSeg models...")
            clipseg_models = load_clipseg_models(device)
        elif segmenter == "sam3":
            sam3_models = load_sam3_models(device)
        else:  # groundedsam
            print("Loading GroundingDINO + SAM models...")
            groundedsam_models = load_groundedsam_models(device)
        print(f"Loading image embedder ({embedder_model})...")
        clip_model, clip_processor, _ = load_clip_model(embedder_model, device=device)
    else:
        print(
            f"Loading text embedder: {text_embedder_type} / {resolved_text_embedder_model}..."
        )
        text_embedder = load_text_embedder(
            model_type=text_embedder_type,
            model_name=resolved_text_embedder_model,
            device=device,
            embedder_quantization=(
                text_embedder_quantization
                if text_embedder_type in ("qwen", "sbert")
                else None
            ),
        )

    print("All models loaded successfully!")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create obfuscated versions of all images
        print("\n" + "=" * 60)
        print("Step 1: Creating obfuscated versions")
        print("=" * 60)

        obfuscated1: dict[Path, Path] = {}
        obfuscated2: dict[Path, Path] = {}
        valid1: set[Path] = set()
        valid2: set[Path] = set()
        text_original1: dict[str, str] = {}
        text_original2: dict[str, str] = {}
        text_obfuscated1: dict[str, str] = {}
        text_obfuscated2: dict[str, str] = {}
        valid_text_ids1: set[str] = set()
        valid_text_ids2: set[str] = set()

        if is_image_mode:
            print(f"\nObfuscating folder1 images for concept1: {concept1}")
            for i, img_path in enumerate(items1):
                print(f"  [{i+1}/{len(items1)}] {img_path.name}")
                obf_path = temp_path / f"obf1_{img_path.stem}{img_path.suffix}"
                try:
                    obf_path, coverage = create_obfuscated_image(
                        input_path=img_path,
                        output_path=obf_path,
                        target_labels=concept1,
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
                    )
                    obfuscated1[img_path] = obf_path
                    coverage_fraction = coverage / 100.0
                    if coverage_fraction < min_coverage:
                        print(f"    WARNING: Coverage too low ({coverage_fraction:.4f})")
                    elif coverage_fraction > max_coverage:
                        print(f"    WARNING: Coverage too high ({coverage_fraction:.4f})")
                    else:
                        valid1.add(img_path)
                        print(f"    OK (coverage={coverage_fraction:.4f})")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
            print(f"\nFolder1: {len(obfuscated1)} obfuscated, {len(valid1)} valid")

            print(f"\nObfuscating folder2 images for concept2: {concept2}")
            for i, img_path in enumerate(items2):
                print(f"  [{i+1}/{len(items2)}] {img_path.name}")
                obf_path = temp_path / f"obf2_{img_path.stem}{img_path.suffix}"
                try:
                    obf_path, coverage = create_obfuscated_image(
                        input_path=img_path,
                        output_path=obf_path,
                        target_labels=concept2,
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
                    )
                    obfuscated2[img_path] = obf_path
                    coverage_fraction = coverage / 100.0
                    if coverage_fraction < min_coverage:
                        print(f"    WARNING: Coverage too low ({coverage_fraction:.4f})")
                    elif coverage_fraction > max_coverage:
                        print(f"    WARNING: Coverage too high ({coverage_fraction:.4f})")
                    else:
                        valid2.add(img_path)
                        print(f"    OK (coverage={coverage_fraction:.4f})")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
            print(f"\nFolder2: {len(obfuscated2)} obfuscated, {len(valid2)} valid")
        else:
            ner_thr = (
                float(threshold[0])
                if isinstance(threshold, list) and threshold
                else float(threshold)
            )
            print("\nLoading GLiNER2 for text obfuscation (shared across all sentences)...")
            text_ner_recognizer = load_recognizer(model_name=None, device=device)

            print(f"\nObfuscating file1 sentences for concept1: {concept1}")
            for i, (sent_id, sentence) in enumerate(text_rows1):
                print(f"  [{i+1}/{len(text_rows1)}] {sent_id}")
                text_original1[sent_id] = sentence
                try:
                    obf_sentence, coverage = create_obfuscated_text_sentence(
                        sentence=sentence,
                        target_labels=concept1,
                        mode=mode,
                        recognizer=text_ner_recognizer,
                        threshold=ner_thr,
                        sequential_labels=sequential_labels,
                    )
                    text_obfuscated1[sent_id] = obf_sentence
                    (obfuscated_f1_folder / f"obfuscated_{sent_id}.txt").write_text(
                        obf_sentence, encoding="utf-8"
                    )
                    coverage_fraction = coverage / 100.0
                    if coverage_fraction < min_coverage:
                        print(f"    WARNING: Coverage too low ({coverage_fraction:.4f})")
                    elif coverage_fraction > max_coverage:
                        print(f"    WARNING: Coverage too high ({coverage_fraction:.4f})")
                    else:
                        valid_text_ids1.add(sent_id)
                        print(f"    OK (coverage={coverage_fraction:.4f})")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
            print(f"\nFile1: {len(text_obfuscated1)} obfuscated, {len(valid_text_ids1)} valid")

            print(f"\nObfuscating file2 sentences for concept2: {concept2}")
            for i, (sent_id, sentence) in enumerate(text_rows2):
                print(f"  [{i+1}/{len(text_rows2)}] {sent_id}")
                text_original2[sent_id] = sentence
                try:
                    obf_sentence, coverage = create_obfuscated_text_sentence(
                        sentence=sentence,
                        target_labels=concept2,
                        mode=mode,
                        recognizer=text_ner_recognizer,
                        threshold=ner_thr,
                        sequential_labels=sequential_labels,
                    )
                    text_obfuscated2[sent_id] = obf_sentence
                    (obfuscated_f2_folder / f"obfuscated_{sent_id}.txt").write_text(
                        obf_sentence, encoding="utf-8"
                    )
                    coverage_fraction = coverage / 100.0
                    if coverage_fraction < min_coverage:
                        print(f"    WARNING: Coverage too low ({coverage_fraction:.4f})")
                    elif coverage_fraction > max_coverage:
                        print(f"    WARNING: Coverage too high ({coverage_fraction:.4f})")
                    else:
                        valid_text_ids2.add(sent_id)
                        print(f"    OK (coverage={coverage_fraction:.4f})")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
            print(f"\nFile2: {len(text_obfuscated2)} obfuscated, {len(valid_text_ids2)} valid")
        
        # Save obfuscated images to output folders
        if is_image_mode:
            print("\nSaving obfuscated images to output folders...")
        
            # Save folder1 obfuscated images
            saved_obf1_paths: list[Path] = []
            saved_orig1_paths: list[Path] = []
            for orig_path, obf_path in sorted(obfuscated1.items(), key=lambda x: x[0]):
                output_path = obfuscated_f1_folder / f"obfuscated_{orig_path.name}"
                shutil.copy2(obf_path, output_path)
                saved_obf1_paths.append(output_path)
                saved_orig1_paths.append(orig_path)
            print(f"  Saved {len(saved_obf1_paths)} images to {obfuscated_f1_folder}")
        
            # Save folder2 obfuscated images
            saved_obf2_paths: list[Path] = []
            saved_orig2_paths: list[Path] = []
            for orig_path, obf_path in sorted(obfuscated2.items(), key=lambda x: x[0]):
                output_path = obfuscated_f2_folder / f"obfuscated_{orig_path.name}"
                shutil.copy2(obf_path, output_path)
                saved_obf2_paths.append(output_path)
                saved_orig2_paths.append(orig_path)
            print(f"  Saved {len(saved_obf2_paths)} images to {obfuscated_f2_folder}")
        
            # Create summary grid images
            print("\nCreating summary grid images...")

            if saved_orig1_paths:
                # Summary of originals from folder1
                summary_orig_f1 = output_folder / "summary_originals_f1.jpg"
                create_summary_grid(saved_orig1_paths, summary_orig_f1)
                print(f"  Created: {summary_orig_f1} ({len(saved_orig1_paths)} images)")

                # Summary of obfuscated from folder1
                summary_obf_f1 = output_folder / "summary_obfuscated_f1.jpg"
                create_summary_grid(saved_obf1_paths, summary_obf_f1)
                print(f"  Created: {summary_obf_f1} ({len(saved_obf1_paths)} images)")

            if saved_orig2_paths:
                # Summary of originals from folder2
                summary_orig_f2 = output_folder / "summary_originals_f2.jpg"
                create_summary_grid(saved_orig2_paths, summary_orig_f2)
                print(f"  Created: {summary_orig_f2} ({len(saved_orig2_paths)} images)")

                # Summary of obfuscated from folder2
                summary_obf_f2 = output_folder / "summary_obfuscated_f2.jpg"
                create_summary_grid(saved_obf2_paths, summary_obf_f2)
                print(f"  Created: {summary_obf_f2} ({len(saved_obf2_paths)} images)")
        
        # Step 2: Pre-compute embeddings
        print("\n" + "=" * 60)
        print("Step 2: Pre-computing embeddings")
        print("=" * 60)
        embeddings: dict = {}
        if is_image_mode:
            all_paths_for_embedding: set[Path] = set()
            for u in valid1:
                all_paths_for_embedding.add(u)
                all_paths_for_embedding.add(obfuscated1[u])
            for v in valid2:
                all_paths_for_embedding.add(v)
                all_paths_for_embedding.add(obfuscated2[v])
            all_paths_list = sorted(all_paths_for_embedding)
            print(f"Computing embeddings for {len(all_paths_list)} images...")
            embeddings = compute_embeddings_batch(
                image_paths=all_paths_list,
                model=clip_model,
                processor=clip_processor,
                device=device,
                batch_size=8,
            )
            print(f"Embeddings computed for {len(embeddings)} images")
        else:
            valid_rows1 = [row for row in text_rows1 if row[0] in valid_text_ids1]
            valid_rows2 = [row for row in text_rows2 if row[0] in valid_text_ids2]
            keyed_rows: list[tuple[str, str, str]] = []
            for sent_id, sentence in valid_rows1:
                keyed_rows.append((sent_id, "orig", sentence))
                keyed_rows.append((sent_id, "obf", text_obfuscated1[sent_id]))
            for sent_id, sentence in valid_rows2:
                keyed_rows.append((sent_id, "orig", sentence))
                keyed_rows.append((sent_id, "obf", text_obfuscated2[sent_id]))
            ordered_embeddings = text_embedder.embed_batch_ordered(
                [t for _, _, t in keyed_rows],
                batch_size=max(1, embed_batch_size),
            )
            embeddings = {
                (sent_id, role): emb
                for (sent_id, role, _), emb in zip(keyed_rows, ordered_embeddings)
            }
            print(f"Embeddings computed for {len(embeddings)} sentence slots")
        
        # Step 3: Compute distances for all pairs
        print("\n" + "=" * 60)
        print("Step 3: Computing distances for all pairs")
        print("=" * 60)
        
        results = []
        d2_minus_d1_values: list[float] = []
        d3_minus_d1_values: list[float] = []
        
        if is_image_mode:
            valid_rows_left = sorted(valid1)
            valid_rows_right = sorted(valid2)
        else:
            valid_rows_left = sorted(valid_text_ids1)
            valid_rows_right = sorted(valid_text_ids2)
        total_valid_pairs = len(valid_rows_left) * len(valid_rows_right)
        
        print(f"Processing {total_valid_pairs} valid pairs...")
        
        pair_count = 0
        for u in valid_rows_left:
            if is_image_mode:
                xu = obfuscated1[u]
                u_emb = embeddings[u]
                xu_emb = embeddings[xu]
                u_id = str(u.absolute())
            else:
                u_emb = embeddings[(u, "orig")]
                xu_emb = embeddings[(u, "obf")]
                u_id = u

            for v in valid_rows_right:
                if is_image_mode:
                    xv = obfuscated2[v]
                    v_emb = embeddings[v]
                    xv_emb = embeddings[xv]
                    v_id = str(v.absolute())
                else:
                    v_emb = embeddings[(v, "orig")]
                    xv_emb = embeddings[(v, "obf")]
                    v_id = v
                
                # Compute similarities
                sim_u_v = similarity_from_embeddings(u_emb, v_emb)
                sim_u_xv = similarity_from_embeddings(u_emb, xv_emb)
                sim_xu_v = similarity_from_embeddings(xu_emb, v_emb)
                
                # Compute distances
                d1 = 1 - sim_u_v       # d1 = 1 - similarity(u, v)
                d2 = 1 - sim_u_xv      # d2 = 1 - similarity(u, X(v))
                d3 = 1 - sim_xu_v      # d3 = 1 - similarity(X(u), v)
                
                # Compute delta values
                d2_minus_d1 = d2 - d1
                d3_minus_d1 = d3 - d1
                
                d2_minus_d1_values.append(d2_minus_d1)
                d3_minus_d1_values.append(d3_minus_d1)
                
                results.append({
                    "u": u_id,
                    "v": v_id,
                    "d1": d1,
                    "d2": d2,
                    "d3": d3,
                    "d2_minus_d1": d2_minus_d1,
                    "d3_minus_d1": d3_minus_d1,
                })
                
                pair_count += 1
                if pair_count % 100 == 0:
                    print(f"  Processed {pair_count}/{total_valid_pairs} pairs...")
        
        print(f"  Completed {pair_count} pairs")
    
    # Clean up models
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
    if text_embedder is not None:
        del text_embedder
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Step 4: Write results to CSV
    print("\n" + "=" * 60)
    print("Step 4: Writing results to CSV")
    print("=" * 60)
    
    fieldnames = ["u", "v", "d1", "d2", "d3", "d2_minus_d1", "d3_minus_d1"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to: {output_csv}")
    
    # Build report
    report_lines = []
    
    def report(line: str = "") -> None:
        """Print a line and add it to the report."""
        print(line)
        report_lines.append(line)
    
    report("=" * 60)
    report("Concept Resolution Analysis Report")
    report("=" * 60)
    report()
    report(f"Timestamp: {datetime.now().isoformat()}")
    report()
    report("Input:")
    report(f"  Input mode: {mode_name}")
    report(f"  Input1: {input1}")
    report(f"  Input2: {input2}")
    report(f"  Concept1: {', '.join(concept1)}")
    report(f"  Concept2: {', '.join(concept2)}")
    report()
    report("Configuration:")
    report(f"  Mode: {mode}")
    if is_image_mode:
        report(f"  Segmenter: {segmenter}")
    if is_image_mode:
        report(f"  Embedder (image): {embedder_model}")
    else:
        report(f"  Text embedder type: {text_embedder_type}")
        report(f"  Text embedder model: {resolved_text_embedder_model}")
        if text_embedder_type in ("qwen", "sbert"):
            report(f"  Text embedder quantization: {text_quant_display}")
        report(f"  Text embed batch size: {embed_batch_size}")
    report(f"  Threshold: {threshold if isinstance(threshold, (int, float)) else ', '.join(map(str, threshold))}")
    if mode == "redact":
        report(f"  Redact blur radius: {redact_blur_radius}")
    elif mode == "replace":
        report(f"  Replacement prompt: {replacement_prompt}")
    report(f"  Seed: {seed}")
    report()
    report("Summary:")
    if is_image_mode:
        report(f"  Images in input1: {len(items1)} (valid: {len(valid1)})")
        report(f"  Images in input2: {len(items2)} (valid: {len(valid2)})")
    else:
        report(f"  Sentences in input1: {len(text_rows1)} (valid: {len(valid_text_ids1)})")
        report(f"  Sentences in input2: {len(text_rows2)} (valid: {len(valid_text_ids2)})")
    report(f"  Total pairs analyzed: {len(results)}")
    report()
    report("Output files:")
    report(f"  Results CSV: {output_csv}")
    report(f"  Report: {report_file}")
    report(f"  Parameters: {params_file}")
    report(f"  Obfuscated folder1: {obfuscated_f1_folder}")
    report(f"  Obfuscated folder2: {obfuscated_f2_folder}")
    if is_image_mode:
        report(f"  Summary images:")
        report(f"    - summary_originals_f1.jpg")
        report(f"    - summary_obfuscated_f1.jpg")
        report(f"    - summary_originals_f2.jpg")
        report(f"    - summary_obfuscated_f2.jpg")
    report()
    report("=" * 60)
    report("Distance Definitions")
    report("=" * 60)
    report()
    report("  d1 = 1 - similarity(u, v)        Distance between original images")
    report("  d2 = 1 - similarity(u, X(v))     Distance from u to obfuscated v")
    report("  d3 = 1 - similarity(X(u), v)     Distance from obfuscated u to v")
    report()
    report("  X(u) = u obfuscated for concept1")
    report("  X(v) = v obfuscated for concept2")
    report()
    
    # Print histograms
    report("=" * 60)
    report("Distribution of d2 - d1")
    report("=" * 60)
    report()
    report("Interpretation: Measures how obfuscating concept2 in folder2 images")
    report("affects their distance from folder1 images.")
    report("  Positive: Obfuscation increases distance (images become more different)")
    report("  Negative: Obfuscation decreases distance (images become more similar)")
    report()
    for line in format_cumulative_distribution(d2_minus_d1_values, num_bins=num_bins, title="d2 - d1"):
        report(line)
    
    report()
    report("=" * 60)
    report("Distribution of d3 - d1")
    report("=" * 60)
    report()
    report("Interpretation: Measures how obfuscating concept1 in folder1 images")
    report("affects their distance to folder2 images.")
    report("  Positive: Obfuscation increases distance (images become more different)")
    report("  Negative: Obfuscation decreases distance (images become more similar)")
    report()
    for line in format_cumulative_distribution(d3_minus_d1_values, num_bins=num_bins, title="d3 - d1"):
        report(line)
    
    report()
    report("=" * 60)
    report(f"To reproduce this run, use: --seed {seed}")
    
    # Write report to file
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    
    print(f"\nReport saved to: {report_file}")


def _load_concept_resolution_params(params_path: Path) -> Optional[dict]:
    """Load params.json from a concept-resolution output folder; None if missing or invalid."""
    if not params_path.is_file():
        return None
    try:
        with open(params_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def regenerate_concept_resolution_report(output_folder: Path, num_bins: int = 20) -> None:
    """
    Rebuild report.txt cumulative histograms from results.csv (and params.json when present).

    Does not re-run obfuscation or embeddings. Overwrites output_folder/report.txt.
    """
    if num_bins < 1:
        raise ValueError("--bins must be at least 1")

    output_folder = Path(output_folder).resolve()
    if not output_folder.is_dir():
        raise FileNotFoundError(f"Output folder not found or not a directory: {output_folder}")

    results_csv = output_folder / "results.csv"
    report_file = output_folder / "report.txt"
    params_path = output_folder / "params.json"
    params = _load_concept_resolution_params(params_path)

    if not results_csv.is_file():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    rows: list[dict[str, str]] = []
    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("results.csv has no header row")
        required_cols = {"d2_minus_d1", "d3_minus_d1"}
        if not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"results.csv must contain columns {sorted(required_cols)}; got {list(reader.fieldnames)}"
            )
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("results.csv contains no data rows")

    d2_minus_d1_values: list[float] = []
    d3_minus_d1_values: list[float] = []
    for row in rows:
        d2_minus_d1_values.append(float(row["d2_minus_d1"]))
        d3_minus_d1_values.append(float(row["d3_minus_d1"]))

    unique_u = {row["u"].strip() for row in rows if row.get("u", "").strip()}
    unique_v = {row["v"].strip() for row in rows if row.get("v", "").strip()}

    def folder1_str() -> str:
        if params and params.get("input1"):
            return str(params["input1"])
        if params and params.get("folder1"):
            return str(params["folder1"])
        return "(see results.csv column u; params.json missing)"

    def folder2_str() -> str:
        if params and params.get("input2"):
            return str(params["input2"])
        if params and params.get("folder2"):
            return str(params["folder2"])
        return "(see results.csv column v; params.json missing)"

    def concept1_str() -> str:
        if params and params.get("concept1") is not None:
            c1 = params["concept1"]
            if isinstance(c1, list):
                return ", ".join(str(x) for x in c1)
            return str(c1)
        return "(unknown)"

    def concept2_str() -> str:
        if params and params.get("concept2") is not None:
            c2 = params["concept2"]
            if isinstance(c2, list):
                return ", ".join(str(x) for x in c2)
            return str(c2)
        return "(unknown)"

    mode = str(params.get("mode", "unknown")) if params else "unknown"
    segmenter = str(params.get("segmenter", "unknown")) if params else "unknown"
    embedder_model = str(
        params.get("embedder_model", DEFAULT_EMBEDDER_MODEL) if params else DEFAULT_EMBEDDER_MODEL
    )
    text_embedder_type = str(params.get("text_embedder_type", "")) if params else ""
    embedder_quantization = params.get("embedder_quantization") if params else None
    embed_batch_size = params.get("embed_batch_size")
    threshold = params.get("threshold", "unknown") if params else "unknown"
    replacement_prompt = params.get("replacement_prompt") if params else None
    redact_blur_radius = params.get("redact_blur_radius") if params else None
    seed = params.get("seed") if params else None

    report_lines: list[str] = []

    def report(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    report("=" * 60)
    report("Concept Resolution Analysis Report")
    report("=" * 60)
    report()
    report("Note: Regenerated from results.csv (--histogram-only). Obfuscation/embeddings were not re-run.")
    report()
    report(f"Timestamp: {datetime.now().isoformat()}")
    report()
    report("Input:")
    report(f"  Folder1: {folder1_str()}")
    report(f"  Folder2: {folder2_str()}")
    report(f"  Concept1: {concept1_str()}")
    report(f"  Concept2: {concept2_str()}")
    report()
    report("Configuration:")
    report(f"  Mode: {mode}")
    report(f"  Segmenter: {segmenter}")
    if text_embedder_type:
        report(f"  Text embedder type: {text_embedder_type}")
        report(f"  Text embedder model: {embedder_model}")
        if text_embedder_type in ("qwen", "sbert") and embedder_quantization is not None:
            report(f"  Text embedder quantization: {embedder_quantization}")
        if embed_batch_size is not None:
            report(f"  Text embed batch size: {embed_batch_size}")
    else:
        report(f"  Embedder: {embedder_model}")
    th_display = (
        threshold
        if isinstance(threshold, (int, float))
        else ", ".join(map(str, threshold))
        if isinstance(threshold, list)
        else str(threshold)
    )
    report(f"  Threshold: {th_display}")
    if mode == "redact" and redact_blur_radius is not None:
        report(f"  Redact blur radius: {redact_blur_radius}")
    elif mode == "replace" and replacement_prompt:
        report(f"  Replacement prompt: {replacement_prompt}")
    if seed is not None:
        report(f"  Seed: {seed}")
    report()
    report("Summary (from results.csv):")
    report(f"  Unique input1 items (u): {len(unique_u)}")
    report(f"  Unique input2 items (v): {len(unique_v)}")
    report(f"  Total pairs analyzed: {len(rows)}")
    report()
    report("Output files:")
    report(f"  Results CSV: {results_csv}")
    report(f"  Report: {report_file}")
    if params_path.is_file():
        report(f"  Parameters: {params_path}")
    report(f"  Obfuscated folder1: {output_folder / 'obfuscated_f1'}")
    report(f"  Obfuscated folder2: {output_folder / 'obfuscated_f2'}")
    report(f"  Summary images:")
    report(f"    - summary_originals_f1.jpg")
    report(f"    - summary_obfuscated_f1.jpg")
    report(f"    - summary_originals_f2.jpg")
    report(f"    - summary_obfuscated_f2.jpg")
    report()
    report("=" * 60)
    report("Distance Definitions")
    report("=" * 60)
    report()
    report("  d1 = 1 - similarity(u, v)        Distance between original images")
    report("  d2 = 1 - similarity(u, X(v))     Distance from u to obfuscated v")
    report("  d3 = 1 - similarity(X(u), v)     Distance from obfuscated u to v")
    report()
    report("  X(u) = u obfuscated for concept1")
    report("  X(v) = v obfuscated for concept2")
    report()
    report("=" * 60)
    report("Distribution of d2 - d1")
    report("=" * 60)
    report()
    report("Interpretation: Measures how obfuscating concept2 in folder2 images")
    report("affects their distance from folder1 images.")
    report("  Positive: Obfuscation increases distance (images become more different)")
    report("  Negative: Obfuscation decreases distance (images become more similar)")
    report()
    for line in format_cumulative_distribution(
        d2_minus_d1_values, num_bins=num_bins, title="d2 - d1"
    ):
        report(line)

    report()
    report("=" * 60)
    report("Distribution of d3 - d1")
    report("=" * 60)
    report()
    report("Interpretation: Measures how obfuscating concept1 in folder1 images")
    report("affects their distance to folder2 images.")
    report("  Positive: Obfuscation increases distance (images become more different)")
    report("  Negative: Obfuscation decreases distance (images become more similar)")
    report()
    for line in format_cumulative_distribution(
        d3_minus_d1_values, num_bins=num_bins, title="d3 - d1"
    ):
        report(line)

    report()
    report("=" * 60)
    if seed is not None:
        report(f"To reproduce this run, use: --seed {seed}")
    else:
        report("To reproduce the original run, use the same CLI as before (seed unknown).")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nReport saved to: {report_file}")


def main() -> None:
    """CLI entry point for concept resolution analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze resolution needed to distinguish between two concepts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare faces vs bodies
  %(prog)s --folder1 ./faces --folder2 ./bodies \\
      --concept1 face --concept2 person --mode redact --output ./results
  
  # Compare cars vs motorcycles with blackout
  %(prog)s --folder1 ./cars --folder2 ./motorcycles \\
      --concept1 car --concept2 motorcycle --mode blackout --output ./results
  
  # Sample a subset of images for faster analysis
  %(prog)s --folder1 ./faces --folder2 ./bodies \\
      --concept1 face --concept2 person --mode redact \\
      --samples1 10 --samples2 10 --output ./results
  
  # Use GroundingDINO+SAM instead of default SAM3
  %(prog)s --folder1 ./faces --folder2 ./bodies \\
      --concept1 face --concept2 person --mode blackout \\
      --segmenter groundedsam --output ./results

  # Use EVA-CLIP-18B for image similarity embeddings
  %(prog)s --folder1 ./faces --folder2 ./bodies \\
      --concept1 face --concept2 person --mode redact \\
      --embedder-model BAAI/EVA-CLIP-18B --output ./results

  # Regenerate report.txt histograms from an existing run (no GPU work)
  %(prog)s --histogram-only --output ./results --bins 30

Distance definitions:
  d1 = 1 - similarity(u, v)        Distance between original images
  d2 = 1 - similarity(u, X(v))     Distance from u to obfuscated v
  d3 = 1 - similarity(X(u), v)     Distance from obfuscated u to v

Output distributions:
  d2 - d1: How obfuscating concept2 in folder2 images affects their
           distance from folder1 images
  d3 - d1: How obfuscating concept1 in folder1 images affects their
           distance to folder2 images

Output folder structure:
  The --output folder will contain:
    - results.csv: All pair-wise distance measurements
    - params.json: Parameters used (for reproducibility)
    - report.txt: Summary with histograms
    - obfuscated_f1/: Obfuscated images from folder1
    - obfuscated_f2/: Obfuscated images from folder2
    - summary_originals_f1.jpg: Grid of original images from folder1
    - summary_obfuscated_f1.jpg: Grid of obfuscated images from folder1
    - summary_originals_f2.jpg: Grid of original images from folder2
    - summary_obfuscated_f2.jpg: Grid of obfuscated images from folder2
        """,
    )
    
    parser.add_argument(
        "--folder1", "-f1",
        type=str,
        default=None,
        help="Input1 path. Directory for image mode, or file for text mode. Not used with --histogram-only.",
    )
    parser.add_argument(
        "--folder2", "-f2",
        type=str,
        default=None,
        help="Input2 path. Directory for image mode, or file for text mode. Not used with --histogram-only.",
    )
    parser.add_argument(
        "--concept1", "-c1",
        type=str,
        nargs="+",
        default=None,
        help="Objects to detect and obfuscate in input1 items (images or sentences). Not used with --histogram-only.",
    )
    parser.add_argument(
        "--concept2", "-c2",
        type=str,
        nargs="+",
        default=None,
        help="Objects to detect and obfuscate in input2 items (images or sentences). Not used with --histogram-only.",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["replace", "redact", "blackout"],
        default=None,
        help="Obfuscation mode: image mode supports replace/redact/blackout; text mode supports redact/blackout. Not used with --histogram-only.",
    )
    parser.add_argument(
        "--output", "-O",
        type=str,
        required=True,
        help="Output folder for results. With --histogram-only, must be an existing run containing results.csv.",
    )
    parser.add_argument(
        "--histogram-only",
        action="store_true",
        help="Only rebuild report.txt from results.csv (and params.json when present). "
        "Does not require --folder1/--folder2/--concept1/--concept2/--mode. Use --bins to change bin count.",
    )
    parser.add_argument(
        "--samples1",
        type=int,
        default=None,
        help="Number of images to sample from folder1 (default: use all).",
    )
    parser.add_argument(
        "--samples2",
        type=int,
        default=None,
        help="Number of images to sample from folder2 (default: use all).",
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
        help=f"Minimum mask coverage fraction (default: {DEFAULT_MIN_COVERAGE}).",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=DEFAULT_MAX_COVERAGE,
        help=f"Maximum mask coverage fraction (default: {DEFAULT_MAX_COVERAGE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of bins for cumulative distribution (default: 20). Applies to full runs and --histogram-only.",
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
        default="sam3",
        choices=["groundedsam", "clipseg", "sam3", "openai-gen", "gpt-5.2"],
        help="Segmentation model: 'sam3' (Meta's Segment Anything 3, default), 'groundedsam', 'clipseg', 'openai-gen', or 'gpt-5.2'.",
    )
    
    # Anonymization parameters
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.4],
        help="Detection threshold (default: 0.4).",
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
        help="Blur radius for mask edges (default: 8).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="Inpainting strength (default: 0.85).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="schnell",
        choices=["schnell", "dev"],
        help="FLUX model (default: schnell).",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=DEFAULT_EMBEDDER_MODEL,
        help=(
            "Image mode: CLIP / EvaCLIP checkpoint for image embeddings (default: "
            f"{DEFAULT_EMBEDDER_MODEL}). "
            f"EVA-02-CLIP example: {EVA02_CLIP_EMBEDDER_MODEL} (needs einops). "
            f"EVA-CLIP-18B example: {EVA_CLIP_IMAGE_EMBEDDER_EXAMPLE}. "
            "Text-file mode: model for --text-embedder (default checkpoint is CLIP; "
            "with --text-embedder qwen the default becomes "
            f"{DEFAULT_QWEN_EMBEDDER_MODEL} when this flag is left at the image default)."
        ),
    )
    parser.add_argument(
        "--text-embedder",
        type=str,
        default="clip",
        choices=["clip", "qwen", "sbert"],
        help=(
            "Text-file inputs only: sentence embedding backend via compare_texts.load_text_embedder "
            "(default: clip). Use qwen for Qwen3 embedding models."
        ),
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Text-file mode only: batch size for sentence embeddings (default: 8).",
    )
    parser.add_argument(
        "--embedder-quantization",
        type=str,
        default=None,
        choices=["none", "half", "4bit", "8bit"],
        help=(
            "Text-file mode; --text-embedder qwen or sbert only: weight precision "
            "(default: none). half=fp16 weights; 4bit/8bit require CUDA and bitsandbytes. "
            "Ignored for clip and for image mode."
        ),
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
        help="Blur radius for redaction mode (default: 30).",
    )
    parser.add_argument(
        "--adaptive-blur",
        action="store_true",
        help="Scale mask blur and dilation based on object size.",
    )
    parser.add_argument(
        "--blur-scale",
        type=float,
        default=1.0,
        help="Blur intensity when using --adaptive-blur (default: 1.0).",
    )
    parser.add_argument(
        "--size-exponent",
        type=float,
        default=1.0,
        help="Controls size-dependence of blur (default: 1.0).",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Constant multiplier for --adaptive-blur (default: 1.0).",
    )
    parser.add_argument(
        "--sequential-labels",
        action="store_true",
        help="Process each object label separately and combine masks.",
    )
    parser.add_argument(
        "--convex-hull",
        action="store_true",
        help="Expand each detected object's mask to its convex hull.",
    )
    parser.add_argument(
        "--skip-empty-labels",
        action="store_true",
        help="Skip obfuscation of objects with empty labels.",
    )
    parser.add_argument(
        "--refinements",
        type=int,
        default=0,
        help="Number of refinement passes for GPT-5.2 (default: 0).",
    )
    
    args = parser.parse_args()

    if args.bins < 1:
        parser.error("--bins must be at least 1")

    if args.histogram_only:
        try:
            regenerate_concept_resolution_report(Path(args.output), num_bins=args.bins)
        except (FileNotFoundError, ValueError) as e:
            parser.error(str(e))
        return

    # Validate arguments (full analysis)
    if args.folder1 is None:
        parser.error("the following arguments are required: --folder1 (unless --histogram-only)")
    if args.folder2 is None:
        parser.error("the following arguments are required: --folder2 (unless --histogram-only)")
    if args.concept1 is None:
        parser.error("the following arguments are required: --concept1 (unless --histogram-only)")
    if args.concept2 is None:
        parser.error("the following arguments are required: --concept2 (unless --histogram-only)")
    if args.mode is None:
        parser.error("the following arguments are required: --mode (unless --histogram-only)")

    folder1 = Path(args.folder1)
    folder2 = Path(args.folder2)

    if not folder1.exists():
        parser.error(f"Folder1 not found: {folder1}")
    if not (folder1.is_dir() or folder1.is_file()):
        parser.error(f"Input1 must be a directory or file: {folder1}")
    if not folder2.exists():
        parser.error(f"Folder2 not found: {folder2}")
    if not (folder2.is_dir() or folder2.is_file()):
        parser.error(f"Input2 must be a directory or file: {folder2}")
    if folder1.is_dir() != folder2.is_dir():
        parser.error(
            "Inputs must both be directories (image mode) or both be files (text mode)."
        )
    if folder1.is_file() and args.mode == "replace":
        parser.error("--mode replace is not supported for text-file inputs")

    if folder1.is_file():
        if args.embed_batch_size < 1:
            parser.error("--embed-batch-size must be at least 1")
        _eq = args.embedder_quantization
        if _eq in ("4bit", "8bit"):
            if not torch.cuda.is_available():
                parser.error(
                    f"--embedder-quantization {_eq} requires CUDA (bitsandbytes is not available on CPU)."
                )
            if args.device == "cpu":
                parser.error(
                    f"--embedder-quantization {_eq} cannot be used with --device cpu. "
                    "Omit --device or use --device cuda."
                )
    
    if args.samples1 is not None and args.samples1 < 1:
        parser.error("--samples1 must be at least 1")
    if args.samples2 is not None and args.samples2 < 1:
        parser.error("--samples2 must be at least 1")
    
    if not (0.0 <= args.min_coverage <= 1.0):
        parser.error("--min-coverage must be between 0.0 and 1.0")
    if not (0.0 <= args.max_coverage <= 1.0):
        parser.error("--max-coverage must be between 0.0 and 1.0")
    if args.min_coverage > args.max_coverage:
        parser.error("--min-coverage cannot be greater than --max-coverage")
    
    if args.mode == "replace" and not args.replace_prompt:
        parser.error("--replace-prompt is required when using --mode replace")
    
    # Normalize threshold
    threshold_value = args.threshold
    if isinstance(threshold_value, list) and len(threshold_value) == 1:
        threshold_value = threshold_value[0]
    
    replacement_prompt = args.replace_prompt if args.mode == "replace" else None
    
    # Run analysis
    run_concept_resolution_analysis(
        input1=folder1,
        input2=folder2,
        concept1=args.concept1,
        concept2=args.concept2,
        mode=args.mode,
        output_folder=Path(args.output),
        samples1=args.samples1,
        samples2=args.samples2,
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
        num_bins=args.bins,
        embedder_model=args.embedder_model,
        text_embedder_type=args.text_embedder,
        text_embedder_quantization=args.embedder_quantization,
        embed_batch_size=args.embed_batch_size,
    )


if __name__ == "__main__":
    main()
