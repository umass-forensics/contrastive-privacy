#!/usr/bin/env python3
"""
Similarity analysis for resolution-analysis output (images or text).

Reads the output folder from resolution_analysis.py (images) or text_resolution_analysis.py
(text). Detects whether the obfuscated/ directory contains images or text files, then
for each original finds its obfuscated counterpart and computes the cosine similarity
between embeddings. Reports a histogram plus mean and median.

- **Images:** CLIP image encoder (default: apple/DFN5B-CLIP-ViT-H-14-378, same as resolution_analysis).
- **Text:** Same embedders as text_resolution_analysis (default: Qwen/Qwen3-Embedding-8B via
  ``--embedder qwen``). Optional ``--embedder-quantization``: ``half`` (fp16), or 4-bit/8-bit
  (bitsandbytes on CUDA).

Example:
    similarity-analysis ./results
    similarity-analysis ./text_results --bins 15
    similarity-analysis ./text_results --embedder qwen --embedder-quantization half
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal, Optional

import torch

from contrastive_privacy.scripts.compare_images import (
    EVA02_CLIP_EMBEDDER_MODEL,
    compute_embeddings_batch,
    load_clip_model,
    similarity_from_embeddings,
)
from contrastive_privacy.scripts.compare_texts import (
    DEFAULT_QWEN_EMBEDDER_MODEL,
    load_text_embedder,
    compute_embeddings_batch as compute_text_embeddings_batch,
)

# Match text_resolution_analysis / reanalyze_text_resolution defaults for sbert.
DEFAULT_SBERT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Supported extensions (must match resolution_analysis and text_resolution_analysis)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".md", ".text", ".csv", ".json"}

# Default/eample image embedders for similarity scoring.
DEFAULT_CLIP_MODEL = "apple/DFN5B-CLIP-ViT-H-14-378"
EVA_CLIP_IMAGE_EMBEDDER_EXAMPLE = "BAAI/EVA-CLIP-18B"


def _path_is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


def _path_is_text(p: Path) -> bool:
    return p.suffix.lower() in TEXT_EXTENSIONS


def discover_pairs(
    output_folder: Path,
    image_folder_override: Optional[Path] = None,
    text_folder_override: Optional[Path] = None,
) -> tuple[Literal["image", "text"], list[tuple[Path, Path]]]:
    """
    Discover original–obfuscated pairs from a resolution_analysis or text_resolution_analysis output folder.

    Detects content type from file extensions in obfuscated/. Uses params.json
    (image_folder or text_folder) and the obfuscated/ directory, or falls back
    to results.csv. If image_folder_override or text_folder_override is provided
    for the detected content type, that folder is used instead of the one in the report.

    Returns:
        (content_type, list of (original_path, obfuscated_path) tuples).
    """
    output_folder = Path(output_folder).resolve()
    obfuscated_folder = output_folder / "obfuscated"
    params_file = output_folder / "params.json"
    results_csv = output_folder / "results.csv"

    pairs: list[tuple[Path, Path]] = []
    content_type: Literal["image", "text"] = "image"

    if not obfuscated_folder.is_dir():
        return content_type, pairs

    # Detect content type from obfuscated files
    has_image = False
    has_text = False
    for obf_file in obfuscated_folder.iterdir():
        if not obf_file.is_file() or not obf_file.name.startswith("obfuscated_"):
            continue
        if _path_is_image(obf_file):
            has_image = True
        if _path_is_text(obf_file):
            has_text = True
    if has_image and not has_text:
        content_type = "image"
    elif has_text and not has_image:
        content_type = "text"
    elif has_image and has_text:
        # Mixed: prefer image if more image files
        image_count = sum(1 for f in obfuscated_folder.iterdir() if f.is_file() and f.name.startswith("obfuscated_") and _path_is_image(f))
        text_count = sum(1 for f in obfuscated_folder.iterdir() if f.is_file() and f.name.startswith("obfuscated_") and _path_is_text(f))
        content_type = "image" if image_count >= text_count else "text"
    else:
        return content_type, pairs

    # Strategy 1: use override folder if provided, else params.json (image_folder or text_folder)
    source_folder: Optional[Path] = None
    if content_type == "image" and image_folder_override is not None:
        source_folder = Path(image_folder_override).resolve()
    elif content_type == "text" and text_folder_override is not None:
        source_folder = Path(text_folder_override).resolve()
    elif params_file.exists():
        try:
            with open(params_file) as f:
                params = json.load(f)
            folder_key = "image_folder" if content_type == "image" else "text_folder"
            source_folder = Path(params.get(folder_key, ""))
        except (json.JSONDecodeError, OSError):
            source_folder = None

    if source_folder is not None and source_folder.is_dir():
        for obf_file in sorted(obfuscated_folder.iterdir()):
            if not obf_file.is_file() or not obf_file.name.startswith("obfuscated_"):
                continue
            original_name = obf_file.name[len("obfuscated_"):]
            original_path = source_folder / original_name
            if original_path.is_file():
                pairs.append((original_path.resolve(), obf_file.resolve()))
        if pairs:
            return content_type, pairs

    # Strategy 2: from results.csv, unique "u" (original) paths
    if results_csv.exists():
        seen: set[str] = set()
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            if "u" not in (reader.fieldnames or []):
                return content_type, pairs
            for row in reader:
                u_str = row.get("u", "").strip()
                if not u_str or u_str in seen:
                    continue
                seen.add(u_str)
                u_path = Path(u_str)
                obf_path = obfuscated_folder / f"obfuscated_{u_path.name}"
                if u_path.is_file() and obf_path.is_file():
                    pairs.append((u_path.resolve(), obf_path.resolve()))
        return content_type, pairs

    return content_type, pairs


def _resolve_text_embedder_settings(
    output_folder: Path,
    embedder: str | None,
    embedder_model: str | None,
    embedder_quantization: str | None,
) -> tuple[str, str, str]:
    """
    Resolve text embedder type, model name, and quantization.

    CLI arguments override values from params.json (when present). Defaults match
    text_resolution_analysis: qwen + Qwen3-Embedding-8B, quantization ``none``.
    """
    params: dict = {}
    params_path = output_folder / "params.json"
    if params_path.is_file():
        try:
            with open(params_path, encoding="utf-8") as f:
                params = json.load(f)
        except (json.JSONDecodeError, OSError):
            params = {}

    et = embedder if embedder is not None else params.get("embedder_type", "qwen")
    em = embedder_model if embedder_model is not None else params.get("embedder_model")
    if em is None:
        if et == "sbert":
            em = DEFAULT_SBERT_EMBEDDER_MODEL
        elif et == "qwen":
            em = DEFAULT_QWEN_EMBEDDER_MODEL
        else:
            em = DEFAULT_CLIP_MODEL

    eq = (
        embedder_quantization
        if embedder_quantization is not None
        else params.get("embedder_quantization", "none")
    )
    return et, em, eq


def format_similarity_histogram(
    similarities: list[float],
    num_bins: int = 20,
    width: int = 50,
) -> list[str]:
    """
    Format a text-based histogram of similarity values (e.g. CLIP cosine similarities).

    Args:
        similarities: List of similarity values (typically in [-1, 1] or [0, 1]).
        num_bins: Number of bins.
        width: Width of the histogram bars in characters.

    Returns:
        List of formatted lines for the histogram.
    """
    lines = []

    if not similarities:
        lines.append("  No data to display.")
        return lines

    min_val = min(similarities)
    max_val = max(similarities)
    mean_val = sum(similarities) / len(similarities)
    sorted_vals = sorted(similarities)
    median_val = sorted_vals[len(sorted_vals) // 2] if sorted_vals else 0.0

    if min_val == max_val:
        lines.append(f"  All {len(similarities)} values are {min_val:.4f}")
        return lines

    bin_width = (max_val - min_val) / num_bins
    bins = [0] * num_bins
    for s in similarities:
        bin_idx = int((s - min_val) / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        bins[bin_idx] += 1

    max_count = max(bins) if bins else 1

    lines.append(f"  Min: {min_val:.4f}  Max: {max_val:.4f}  Mean: {mean_val:.4f}  Median: {median_val:.4f}")
    lines.append("")

    for i, count in enumerate(bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_length
        lines.append(f"  [{bin_start:.3f}, {bin_end:.3f})  {bar} {count}")

    return lines


def run_analysis(
    output_folder: Path,
    device: str | None = None,
    batch_size: int = 8,
    embed_batch_size: int | None = None,
    num_bins: int = 20,
    histogram_width: int = 50,
    model_name: str = DEFAULT_CLIP_MODEL,
    image_folder: Optional[Path] = None,
    text_folder: Optional[Path] = None,
    text_embedder: Optional[str] = None,
    text_embedder_model: Optional[str] = None,
    text_embedder_quantization: Optional[str] = None,
) -> None:
    """
    Load pairs from output folder, compute embedding similarities, print histogram and stats.

    For images, ``model_name`` is the CLIP checkpoint. For text, embedder settings come from
    ``text_embedder*`` and params.json (see ``_resolve_text_embedder_settings``).
    ``embed_batch_size`` overrides ``batch_size`` for text embedding only.

    If image_folder or text_folder is provided, it overrides the folder stored in
    the report (params.json) for resolving original files by name.
    """
    output_folder = Path(output_folder).resolve()
    if not output_folder.is_dir():
        print(f"Error: output folder not found: {output_folder}", file=sys.stderr)
        sys.exit(1)
    image_override = None
    if image_folder is not None:
        image_override = Path(image_folder).resolve()
        if not image_override.is_dir():
            print(f"Error: image folder not found or not a directory: {image_override}", file=sys.stderr)
            sys.exit(1)
    text_override = None
    if text_folder is not None:
        text_override = Path(text_folder).resolve()
        if not text_override.is_dir():
            print(f"Error: text folder not found or not a directory: {text_override}", file=sys.stderr)
            sys.exit(1)

    content_type, pairs = discover_pairs(
        output_folder,
        image_folder_override=image_override,
        text_folder_override=text_override,
    )
    if not pairs:
        print(
            "No original–obfuscated pairs found. Ensure the folder is from resolution_analysis.py or "
            "text_resolution_analysis.py and contains obfuscated/ (and params.json or results.csv).",
            file=sys.stderr,
        )
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size < 1:
        print("Error: --batch-size must be at least 1", file=sys.stderr)
        sys.exit(1)
    if embed_batch_size is not None and embed_batch_size < 1:
        print("Error: --embed-batch-size must be at least 1", file=sys.stderr)
        sys.exit(1)

    label = "image" if content_type == "image" else "text"
    print(f"Detected {label} content: {len(pairs)} original–obfuscated pairs")

    similarities: list[float] = []
    resolved_text_et: Optional[str] = None
    resolved_text_em: Optional[str] = None

    if content_type == "image":
        print(f"Loading CLIP image encoder ({model_name})...")
        clip_model, clip_processor, device = load_clip_model(model_name, device)
        try:
            all_paths = [p[0] for p in pairs] + [p[1] for p in pairs]
            embeddings = compute_embeddings_batch(
                all_paths,
                model=clip_model,
                processor=clip_processor,
                device=device,
                batch_size=batch_size,
            )
            for orig_path, obf_path in pairs:
                sim = similarity_from_embeddings(embeddings[orig_path], embeddings[obf_path])
                similarities.append(sim)
        finally:
            del clip_model, clip_processor
            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        resolved_text_et, resolved_text_em, eq = _resolve_text_embedder_settings(
            output_folder,
            text_embedder,
            text_embedder_model,
            text_embedder_quantization,
        )
        q_label = ""
        if eq and eq != "none":
            q_label = (
                ", half precision (fp16)"
                if eq == "half"
                else f", quantization={eq}"
            )
        print(f"Loading text embedder: {resolved_text_et} / {resolved_text_em}{q_label}...")
        text_batch_size = embed_batch_size if embed_batch_size is not None else batch_size
        # Text: read files once, batch embed unique texts, then compute similarity per pair
        text_embedder = load_text_embedder(
            model_type=resolved_text_et,
            model_name=resolved_text_em,
            device=device,
            embedder_quantization=eq,
        )
        try:
            pair_texts = [
                (
                    orig_path.read_text(encoding="utf-8", errors="replace"),
                    obf_path.read_text(encoding="utf-8", errors="replace"),
                )
                for orig_path, obf_path in pairs
            ]
            unique_texts = list(dict.fromkeys(t for pair in pair_texts for t in pair))
            text_embeddings = compute_text_embeddings_batch(
                unique_texts,
                embedder=text_embedder,
                batch_size=text_batch_size,
            )
            for t_orig, t_obf in pair_texts:
                sim = similarity_from_embeddings(text_embeddings[t_orig], text_embeddings[t_obf])
                similarities.append(sim)
        finally:
            del text_embedder
            if device == "cuda":
                torch.cuda.empty_cache()

    # Stats and histogram
    mean_sim = sum(similarities) / len(similarities)
    sorted_sim = sorted(similarities)
    median_sim = sorted_sim[len(sorted_sim) // 2]

    print()
    print("=" * 60)
    if content_type == "image":
        title = f"Original vs. obfuscated CLIP {label} similarity (cosine)"
    else:
        title = (
            f"Original vs. obfuscated text similarity "
            f"({resolved_text_et}: {resolved_text_em}, cosine)"
        )
    print(title)
    print("=" * 60)
    print(f"  Pairs:    {len(similarities)}")
    print(f"  Mean:     {mean_sim:.4f}")
    print(f"  Median:   {median_sim:.4f}")
    print()
    print("Similarity histogram:")
    for line in format_similarity_histogram(similarities, num_bins=num_bins, width=histogram_width):
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarity between original and obfuscated pairs (images via CLIP, "
            "text via the same embedders as text_resolution_analysis, default Qwen3-Embedding-8B)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder from resolution_analysis.py or text_resolution_analysis.py (contains obfuscated/ and params.json or results.csv). Content type (image vs text) is auto-detected from file extensions.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=None,
        metavar="PATH",
        help="Folder containing original images. Overrides the path stored in the report (params.json). Use when originals have been moved or the report was produced elsewhere.",
    )
    parser.add_argument(
        "--text-folder",
        type=str,
        default=None,
        metavar="PATH",
        help="Folder containing original text files. Overrides the path stored in the report (params.json). Use when originals have been moved or the report was produced elsewhere.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device for models (default: auto-detect).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embeddings (default: 8). Used for images and as default for text.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override batch size for text embeddings only (default: use --batch-size).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins (default: 20).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=50,
        help="Histogram bar width in characters (default: 50).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CLIP_MODEL,
        help=(
            "CLIP model for **image** runs only (default: same as resolution_analysis). "
            f"EVA-02-CLIP example: {EVA02_CLIP_EMBEDDER_MODEL} (needs einops). "
            f"EVA-CLIP-18B example: {EVA_CLIP_IMAGE_EMBEDDER_EXAMPLE}. "
            "For text, use --embedder / --embedder-model."
        ),
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        choices=["clip", "sbert", "qwen"],
        help=(
            "Text embedder type for **text** runs (default: params.json embedder_type if present, "
            "else qwen). Ignored for image runs."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Text embedder model name (default: params.json embedder_model if present, else "
            f"{DEFAULT_SBERT_EMBEDDER_MODEL} for sbert, {DEFAULT_QWEN_EMBEDDER_MODEL} for qwen, "
            f"or {DEFAULT_CLIP_MODEL} for clip)."
        ),
    )
    parser.add_argument(
        "--embedder-quantization",
        type=str,
        default=None,
        choices=["none", "half", "4bit", "8bit"],
        help=(
            "For sbert/qwen text embedders: half=float16 weights (no bitsandbytes); "
            "4bit/8bit=bitsandbytes (CUDA). "
            "Default when omitted: value from params.json if present, else none. "
            "Ignored for image runs and for --embedder clip."
        ),
    )
    args = parser.parse_args()

    run_analysis(
        output_folder=Path(args.output_folder),
        device=args.device,
        batch_size=args.batch_size,
        embed_batch_size=args.embed_batch_size,
        num_bins=args.bins,
        histogram_width=args.width,
        model_name=args.model,
        image_folder=Path(args.image_folder) if args.image_folder else None,
        text_folder=Path(args.text_folder) if args.text_folder else None,
        text_embedder=args.embedder,
        text_embedder_model=args.embedder_model,
        text_embedder_quantization=args.embedder_quantization,
    )


if __name__ == "__main__":
    main()
