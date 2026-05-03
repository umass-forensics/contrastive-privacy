#!/usr/bin/env python3
"""
Resolution analysis for obfuscated texts.

This script analyzes the "effective resolution" of text obfuscation methods by comparing
how well obfuscated texts preserve semantic relationships while obscuring identity.

For each reference text u, the script:
1. Creates an obfuscated version X(u)
2. Randomly selects 'trials' other test texts v
3. Computes resolution = d2 - d1 where:
   - d1 = 1 - similarity(X(u), v)    (distance from obfuscated to other's original)
   - d2 = 1 - similarity(X(u), X(v)) (distance between obfuscated versions)

Example:
    text-resolution-analysis ./texts --entities person organization --mode blackout --trials 5
"""

import argparse
import csv
import json
import math
import os
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import textwrap

import torch
from PIL import Image, ImageDraw, ImageFont

from contrastive_privacy.scripts.text_anonymize import (
    anonymize_text,
    AnonymizationResult,
    BLOCK_CHAR,
    DEFAULT_CONCEPT_MODEL,
    DEFAULT_CONCEPT_TEMPERATURE,
    OPENROUTER_CONCEPT_MODELS,
)
from contrastive_privacy.scripts.recognize_entities import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_MODEL,
    load_recognizer,
    GLiNER2Recognizer,
)
from contrastive_privacy.scripts.compare_texts import (
    load_text_embedder,
    similarity_from_embeddings,
    TextEmbedder,
    EVA_CLIP_TEXT_EMBEDDER_EXAMPLE,
    DEFAULT_QWEN_EMBEDDER_MODEL,
)
from contrastive_privacy.reporting import generate_analysis_artifacts


# Supported text file extensions (tuple: stable iteration order across processes)
TEXT_EXTENSIONS = (".csv", ".json", ".md", ".text", ".txt")

# Default minimum coverage threshold (as a fraction)
# If less than this fraction of the text was detected, consider the obfuscation invalid
DEFAULT_MIN_COVERAGE = 0.001  # 0.1%

# Default maximum coverage threshold (as a fraction)
# If more than this fraction of the text was altered, the text cannot be made private
DEFAULT_MAX_COVERAGE = 1.0  # 100% (no limit by default)
DEFAULT_SBERT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keys written to params.json / printed under "Configuration:" in report.txt (stable order).
_PARAMS_REPORT_KEYS = (
    "text_folder",
    "approach",
    "entities",
    "concept",
    "concept_model",
    "concept_temperature",
    "base_concepts",
    "instances",
    "mode",
    "trials",
    "samples",
    "min_coverage",
    "max_coverage",
    "seed",
    "device",
    "ner_model",
    "threshold",
    "embedder_type",
    "embedder_model",
    "embedder_quantization",
    "embed_batch_size",
    "placeholder",
    "compact_blackout_words",
    "sequential_labels",
    "propagate",
    "continue_from_output",
    "obfuscate_missing_in_continue",
    "generate_comparisons",
    "write_analysis_artifacts",
    "retry_skipped_paths_count",
    "timestamp",
    "command_line",
)


def _params_report_label(key: str) -> str:
    """Human-readable label for report.txt (matches legacy Configuration wording where possible)."""
    labels = {
        "text_folder": "Text folder",
        "approach": "Approach",
        "entities": "Entities",
        "concept": "Concept",
        "concept_model": "Concept model",
        "concept_temperature": "Concept temperature",
        "base_concepts": "Base concepts",
        "instances": "Instances",
        "mode": "Mode",
        "trials": "Trials per reference",
        "samples": "Samples",
        "min_coverage": "Min coverage",
        "max_coverage": "Max coverage",
        "seed": "Seed",
        "device": "Device",
        "ner_model": "NER model",
        "threshold": "Threshold",
        "embedder_type": "Embedder",
        "embedder_model": "Embedder model",
        "embedder_quantization": "Embedder quantization",
        "embed_batch_size": "Embed batch size",
        "placeholder": "Placeholder",
        "compact_blackout_words": "Compact blackout words",
        "sequential_labels": "Sequential labels",
        "propagate": "Propagate",
        "continue_from_output": "Continue from output",
        "obfuscate_missing_in_continue": "Obfuscate missing in continue",
        "generate_comparisons": "Generate comparisons",
        "retry_skipped_paths_count": "Retry skipped paths count",
        "timestamp": "Timestamp (params save)",
        "command_line": "Command line",
    }
    return labels.get(key, key.replace("_", " ").title())


def _format_params_report_value(key: str, value) -> str:
    """String form for one params entry in report.txt (includes explicit empty / default semantics)."""
    if key == "samples" and value is None:
        return "all"
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        if key == "sequential_labels" and value:
            return "enabled (additive detection)"
        if key == "sequential_labels" and not value:
            return "disabled"
        if key == "propagate" and value:
            return "enabled (all occurrences anonymized)"
        if key == "propagate" and not value:
            return "disabled"
        return "true" if value else "false"
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        return ", ".join(str(x) for x in value)
    return str(value)


def parse_report_skipped_paths(report_path: Path) -> list[Path]:
    """
    Parse report.txt and return the full paths of skipped reference texts
    from the "Skipped reference texts (detailed):" section.
    """
    paths: list[Path] = []
    content = report_path.read_text(encoding="utf-8")
    in_section = False
    for line in content.splitlines():
        if "Skipped reference texts (detailed):" in line:
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            # Report lines look like:
            #   "  - Path: /abs/path/to/file.txt"
            # but we also accept:
            #   "Path: /abs/path/to/file.txt"
            path_key = "Path:"
            if path_key in stripped:
                idx = stripped.find(path_key)
                path_str = stripped[idx + len(path_key) :].strip()
                if path_str:
                    paths.append(Path(path_str))
            # Section ends at next major heading or empty block (e.g. "  Trials requested")
            if stripped.startswith("Trials requested") or (
                stripped
                and not stripped.startswith("-")
                and not stripped.startswith("Path:")
                and not stripped.startswith("Reason:")
            ):
                break
    return paths


def load_params_from_output(output_folder: Path) -> dict:
    """Load params.json from a previous run; raises FileNotFoundError if missing."""
    params_file = output_folder / "params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Params file not found: {params_file}. Run without --retry first.")
    with open(params_file, encoding="utf-8") as f:
        return json.load(f)


def _argv_contains_any(argv: list[str], option_strings: list[str]) -> bool:
    """Return True if any of the option strings appear in argv (for override detection)."""
    for opt in option_strings:
        if opt in argv:
            return True
    return False


def get_text_files(folder: Path) -> list[Path]:
    """Get all text files from a folder."""
    texts = []
    for ext in TEXT_EXTENSIONS:
        texts.extend(folder.glob(f"*{ext}"))
        texts.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(texts)


def create_obfuscated_text(
    text: str,
    entity_types: list[str],
    mode: str,
    threshold: float = 0.3,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    recognizer: Optional[GLiNER2Recognizer] = None,
    placeholder: str = "[REDACTED]",
    sequential_labels: bool = False,
    propagate: bool = True,
    approach: str = "entity",
    concept: Optional[str] = None,
    concept_model: Optional[str] = None,
    concept_temperature: float = DEFAULT_CONCEPT_TEMPERATURE,
    base_concepts: Optional[list[str]] = None,
    base_recognizer: Optional[GLiNER2Recognizer] = None,
    instances: Optional[list[str]] = None,
    compact_blackout_words: bool = False,
) -> tuple[str, float, float, float]:
    """
    Create an obfuscated version of a text (entity or concept approach).
    
    Args:
        text: Original text.
        entity_types: Entity types to detect and obfuscate (entity approach).
        mode: "blackout" (block chars) or "redact" (placeholder).
        threshold: Detection threshold (0.0-1.0).
        model_name: GLiNER2 model name (uses default if None).
        device: Compute device for GLiNER2 when entity detection is used.
        recognizer: Pre-loaded GLiNER2Recognizer (optional, for reuse).
        placeholder: Placeholder text for redact mode.
        sequential_labels: If True, process each entity type separately.
        propagate: If True (default), redact all occurrences of each detected entity.
        approach: "entity" (NER) or "concept" (fal.ai OpenRouter by description).
        concept: Privacy concept description (required when approach is "concept").
        concept_model: fal.ai OpenRouter model for concept approach.
        concept_temperature: OpenRouter sampling temperature when approach is "concept".
        base_concepts: Optional base concepts to obfuscate in a second pass using GLiNER2.
        base_recognizer: Optional pre-loaded GLiNER2 recognizer for base concept pass.
        instances: Literal substrings to always obfuscate: case-insensitive matching
            anywhere in the text (regex finditer with re.escape), including inside larger
            words; each matched span is obfuscated in full. No GLiNER2 for these spans.
        compact_blackout_words: If True and mode is "blackout", replace each
            obfuscated non-whitespace token with a single block character.
    
    Returns:
        Tuple of (obfuscated text, final coverage percentage, base coverage percentage, primary coverage percentage).
    """
    result = anonymize_text(
        text=text,
        entity_types=entity_types,
        mode=mode,
        placeholder=placeholder,
        threshold=threshold,
        model_name=model_name,
        device=device,
        recognizer=recognizer,
        sequential_labels=sequential_labels,
        propagate=propagate,
        approach=approach,
        concept=concept,
        concept_model=concept_model,
        concept_temperature=concept_temperature,
        instances=instances,
        compact_blackout_words=compact_blackout_words,
    )

    primary_text = result.anonymized_text
    primary_coverage_percent = result.coverage * 100.0

    # Optional second pass: base concept obfuscation with GLiNER2 (entity approach).
    if base_concepts:
        base_result = anonymize_text(
            text=primary_text,
            entity_types=base_concepts,
            mode=mode,
            placeholder=placeholder,
            threshold=threshold,
            model_name=model_name,
            device=device,
            recognizer=base_recognizer,
            sequential_labels=sequential_labels,
            propagate=propagate,
            approach="entity",
            concept=None,
            concept_model=None,
            compact_blackout_words=compact_blackout_words,
        )
        base_text = base_result.anonymized_text
        base_coverage_percent = base_result.coverage * 100.0

        # Compute net final coverage from original->final character changes.
        max_len = max(len(text), len(base_text))
        if max_len == 0:
            final_coverage_percent = 0.0
        else:
            changed = sum(
                1 for i in range(max_len)
                if (text[i] if i < len(text) else "") != (base_text[i] if i < len(base_text) else "")
            )
            final_coverage_percent = (changed / max_len) * 100.0

        return base_text, final_coverage_percent, base_coverage_percent, primary_coverage_percent

    return primary_text, primary_coverage_percent, 0.0, primary_coverage_percent


def create_comparison_text(
    u_text: str,
    xu_text: str,
    v_text: str,
    xv_text: str,
    u_name: str,
    v_name: str,
    resolution: float,
    output_path: Path,
) -> Path:
    """
    Create a comparison text file showing reference and target texts
    before and after obfuscation.

    Layout:
        === REFERENCE (u): <filename> ===
        <original text u>

        === REFERENCE OBFUSCATED X(u): <filename> ===
        <obfuscated text X(u)>

        === TARGET (v): <filename> ===
        <original text v>

        === TARGET OBFUSCATED X(v): <filename> ===
        <obfuscated text X(v)>

    Args:
        u_text: Original reference text.
        xu_text: Obfuscated reference text.
        v_text: Original target text.
        xv_text: Obfuscated target text.
        u_name: Filename of the reference text.
        v_name: Filename of the target text.
        resolution: The computed resolution value for this pair.
        output_path: Path where the comparison file will be saved.

    Returns:
        Path to the created comparison file.
    """
    separator = "=" * 60
    lines = [
        separator,
        f"Resolution: {resolution:+.4f}",
        separator,
        "",
        separator,
        f"REFERENCE (u): {u_name}",
        separator,
        u_text,
        "",
        separator,
        f"REFERENCE OBFUSCATED X(u): {u_name}",
        separator,
        xu_text,
        "",
        separator,
        f"TARGET (v): {v_name}",
        separator,
        v_text,
        "",
        separator,
        f"TARGET OBFUSCATED X(v): {v_name}",
        separator,
        xv_text,
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Text-to-image rendering and summary grids
# ---------------------------------------------------------------------------

def _load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospace font, falling back to the default bitmap font."""
    # Common paths for monospace fonts on various systems
    mono_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in mono_candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Last resort: Pillow built-in (small, but always available)
    return ImageFont.load_default()


def _measure_cell_height(
    text: str,
    title: str,
    width: int,
    font_size: int,
    padding: int = 4,
    line_spacing: int = 1,
) -> int:
    """Return the natural content height (in pixels) for a text cell."""
    font = _load_font(font_size)
    title_font = _load_font(font_size + 1)

    tmp = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(tmp)

    usable_w = width - 2 * padding
    sample = "abcdefghijklmnopqrstuvwxyz0123456789"
    sample_bbox = draw.textbbox((0, 0), sample, font=font)
    avg_char_w = (sample_bbox[2] - sample_bbox[0]) / len(sample)
    chars_per_line = max(int(usable_w / avg_char_w), 20)

    y = padding
    if title:
        wrapped_title = textwrap.fill(title, width=chars_per_line)
        bbox = draw.textbbox((0, 0), wrapped_title, font=title_font)
        y += (bbox[3] - bbox[1]) + 2 + 2  # title + gap + separator + gap

    wrapped = textwrap.fill(text, width=chars_per_line)
    for line in wrapped.split("\n"):
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing

    return y + padding


def render_text_to_image(
    text: str,
    title: str = "",
    width: int = 480,
    height: int = 480,
    font_size: int = 10,
    padding: int = 4,
    line_spacing: int = 1,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0),
    title_color: tuple[int, int, int] = (40, 40, 120),
) -> Image.Image:
    """
    Render a text string into a PIL Image of exactly *width* x *height*.

    The text is word-wrapped to fit the image width.  If the wrapped text
    exceeds the image height it is truncated with an ellipsis indicator.

    Args:
        text: The text content to render.
        title: Optional title drawn at the top in a distinct colour.
        width: Image width in pixels.
        height: Image height in pixels (exact).
        font_size: Font size for the body text.
        padding: Padding around the text in pixels.
        line_spacing: Extra pixels between lines.
        bg_color: Background colour (R, G, B).
        text_color: Body text colour (R, G, B).
        title_color: Title colour (R, G, B).

    Returns:
        A PIL Image with the rendered text.
    """
    font = _load_font(font_size)
    title_font = _load_font(font_size + 1)

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    usable_w = width - 2 * padding

    # Measure average character width using the actual font
    sample = "abcdefghijklmnopqrstuvwxyz0123456789"
    sample_bbox = draw.textbbox((0, 0), sample, font=font)
    avg_char_w = (sample_bbox[2] - sample_bbox[0]) / len(sample)
    chars_per_line = max(int(usable_w / avg_char_w), 20)

    y = padding

    # Draw title
    if title:
        wrapped_title = textwrap.fill(title, width=chars_per_line)
        bbox = draw.textbbox((0, 0), wrapped_title, font=title_font)
        title_h = bbox[3] - bbox[1]
        draw.text((padding, y), wrapped_title, fill=title_color, font=title_font)
        y += title_h + 2
        draw.line([(padding, y), (width - padding, y)], fill=(200, 200, 200), width=1)
        y += 2

    # Word-wrap body text
    wrapped = textwrap.fill(text, width=chars_per_line)
    body_lines = wrapped.split("\n")

    max_body_y = height - padding
    for line in body_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_h = bbox[3] - bbox[1] + line_spacing
        if y + line_h > max_body_y:
            draw.text((padding, y), "...", fill=text_color, font=font)
            break
        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_h

    return img


def create_text_summary_grid(
    texts: list[str],
    labels: list[str],
    output_path: Path,
    font_size: int = 10,
    min_cell_width: int = 120,
    max_cell_width: int = 600,
) -> Path:
    """
    Create a roughly-square summary grid of rendered text cells (PDF).

    The cell width is chosen automatically so that the overall grid is as
    close to square as possible.  A binary search over cell widths is used:
    narrower cells produce more line-wrapping and therefore taller cells,
    which balances the grid's aspect ratio.

    Args:
        texts: The text strings to render.
        labels: Per-cell titles (typically filenames).
        output_path: Where to save the grid (PDF).
        font_size: Font size used for body text.
        min_cell_width: Lower bound for cell width search (pixels).
        max_cell_width: Upper bound for cell width search (pixels).

    Returns:
        Path to the saved grid PDF.
    """
    if not texts:
        raise ValueError("No texts to create grid from")

    num = len(texts)
    cols = math.ceil(math.sqrt(num))
    rows = math.ceil(num / cols)
    border = 1

    def grid_aspect(cell_w: int) -> float:
        """Return width/height ratio for a given cell width."""
        # Measure the natural content height of every cell at this width
        max_row_heights: list[int] = []
        for r in range(rows):
            start = r * cols
            end = min(start + cols, num)
            row_h = 0
            for i in range(start, end):
                h = _measure_cell_height(texts[i], labels[i], cell_w, font_size)
                if h > row_h:
                    row_h = h
            max_row_heights.append(row_h)
        gw = cols * cell_w + (cols - 1) * border
        gh = sum(max_row_heights) + (rows - 1) * border
        return gw / gh if gh > 0 else 999.0

    # Binary search for the cell width that gives aspect ratio closest to 1.0
    lo, hi = min_cell_width, max_cell_width
    for _ in range(20):
        mid = (lo + hi) // 2
        if mid == lo:
            break
        ratio = grid_aspect(mid)
        if ratio > 1.0:
            # Grid is wider than tall -> need narrower cells
            hi = mid
        else:
            # Grid is taller than wide -> need wider cells
            lo = mid

    # Pick the width closest to aspect 1.0
    best_w = lo
    best_diff = abs(grid_aspect(lo) - 1.0)
    for candidate in (lo, hi, (lo + hi) // 2):
        diff = abs(grid_aspect(candidate) - 1.0)
        if diff < best_diff:
            best_diff = diff
            best_w = candidate

    cell_width = best_w

    # Now measure row heights at the chosen width to get actual cell height
    row_heights: list[int] = []
    for r in range(rows):
        start = r * cols
        end = min(start + cols, num)
        row_h = 0
        for i in range(start, end):
            h = _measure_cell_height(texts[i], labels[i], cell_width, font_size)
            if h > row_h:
                row_h = h
        row_heights.append(row_h)

    # Render all cells at the chosen width, using the row height as exact height
    grid_w = cols * cell_width + (cols - 1) * border
    grid_h = sum(row_heights) + (rows - 1) * border
    grid = Image.new("RGB", (grid_w, grid_h), color=(220, 220, 220))

    y_offset = 0
    for r in range(rows):
        x_offset = 0
        for c in range(cols):
            idx = r * cols + c
            if idx >= num:
                break
            cell = render_text_to_image(
                text=texts[idx],
                title=labels[idx],
                width=cell_width,
                height=row_heights[r],
                font_size=font_size,
            )
            grid.paste(cell, (x_offset, y_offset))
            x_offset += cell_width + border
        y_offset += row_heights[r] + border

    grid.save(output_path, "PDF", resolution=150)
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
    """
    for line in format_resolution_histogram(resolutions, num_bins, width):
        print(line)


def run_text_resolution_analysis(
    text_folder: Path,
    entities: list[str],
    mode: str,
    trials: int,
    output_folder: Path,
    samples: Optional[int] = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_coverage: float = DEFAULT_MAX_COVERAGE,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    ner_model: Optional[str] = None,
    threshold: float = 0.3,
    embedder_type: str = "sbert",
    embedder_model: Optional[str] = None,
    embedder_quantization: Optional[str] = None,
    embed_batch_size: int = 8,
    placeholder: str = "[REDACTED]",
    sequential_labels: bool = False,
    propagate: bool = True,
    approach: str = "entity",
    concept: Optional[str] = None,
    concept_model: Optional[str] = None,
    concept_temperature: float = DEFAULT_CONCEPT_TEMPERATURE,
    base_concepts: Optional[list[str]] = None,
    instances: Optional[list[str]] = None,
    compact_blackout_words: bool = False,
    continue_from_output: bool = False,
    retry_skipped_paths: Optional[list[Path]] = None,
    obfuscate_missing_in_continue: bool = True,
    generate_comparisons: bool = True,
    write_analysis_artifacts: bool = True,
    command_line: Optional[str] = None,
) -> None:
    """
    Run resolution analysis on a folder of text files.
    
    Args:
        text_folder: Folder containing input text files.
        entities: List of entity types to detect and obfuscate (entity approach).
        mode: "blackout" or "redact".
        trials: Number of test texts to sample per reference text.
        output_folder: Base folder for all outputs.
        samples: Number of reference texts to sample (None = use all).
        min_coverage: Minimum coverage fraction for valid obfuscation.
        max_coverage: Maximum coverage fraction for valid obfuscation.
        seed: Random seed for reproducibility.
        device: Unused (kept for API compatibility).
        ner_model: GLiNER2 model name (uses default if None).
        threshold: Detection threshold (0.0-1.0).
        embedder_type: "clip", "sbert", or "qwen".
        embedder_model: Embedder model name (uses default if None).
        embedder_quantization: For sbert/qwen: ``\"none\"``, ``\"half\"`` (fp16), ``\"4bit\"``, or ``\"8bit\"`` (bnb on CUDA).
        embed_batch_size: Batch size for Step 3 embedding (lower if CUDA OOM on large models).
        placeholder: Placeholder text for redact mode.
        sequential_labels: If True, process each entity type separately.
        propagate: If True (default), redact all occurrences of each detected entity.
        approach: "entity" (NER) or "concept" (fal.ai OpenRouter by description).
        concept: Privacy concept description (required when approach is "concept").
        concept_model: fal.ai OpenRouter model for concept approach.
        concept_temperature: OpenRouter sampling temperature when approach is "concept".
        base_concepts: Optional base concepts to obfuscate after primary obfuscation using GLiNER2.
        instances: Literal substrings to always obfuscate: case-insensitive matching
            anywhere in the text (regex finditer with re.escape), including inside larger
            words; each matched span is obfuscated in full. No GLiNER2 for these spans.
        compact_blackout_words: If True and mode is "blackout", replace each
            obfuscated non-whitespace token with a single block character.
        obfuscate_missing_in_continue: When continue_from_output=True, if False, skip texts with no
            existing obfuscated file instead of running primary obfuscation.
        generate_comparisons: If False, skip writing per-pair comparison files while still computing
            resolution and results.csv.
        write_analysis_artifacts: If True, automatically write analysis_report.html and
            analysis_report.json after the run completes.
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reduce backend-level nondeterminism for reproducible embeddings/resolution.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if device == "cuda":
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    else:
        # CPU thread scheduling/order can introduce small run-to-run numeric drift.
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    
    # Get all text files
    text_files = get_text_files(text_folder)
    if len(text_files) < 2:
        raise ValueError(f"Need at least 2 text files in folder, found {len(text_files)}")
    
    ner_model_display = ner_model or DEFAULT_MODEL
    if embedder_model is None:
        if embedder_type == "sbert":
            embedder_model = DEFAULT_SBERT_EMBEDDER_MODEL
        elif embedder_type == "qwen":
            embedder_model = DEFAULT_QWEN_EMBEDDER_MODEL
    print(f"Found {len(text_files)} text files in {text_folder}")
    print(f"Approach: {approach}")
    print(f"Obfuscation mode: {mode}")
    if approach == "entity":
        print(f"Target entities: {', '.join(entities)}")
        print(f"NER model: {ner_model_display}")
        print(f"Detection threshold: {threshold}")
        if sequential_labels:
            print("Sequential labels: enabled (each entity type processed separately)")
        if propagate:
            print("Propagate: enabled (all occurrences will be anonymized)")
    else:
        print(f"Privacy concept: {concept}")
        print(f"Concept model: {concept_model or DEFAULT_CONCEPT_MODEL}")
        print(f"Concept temperature: {concept_temperature}")
    if base_concepts:
        print(f"Base concepts (GLiNER2-after-primary): {', '.join(base_concepts)}")
    if instances:
        print(f"Instances (literal substring obfuscation): {', '.join(instances)}")
    if mode == "blackout" and compact_blackout_words:
        print("Blackout style: compact (one box per obfuscated word)")
    print(f"Embedder: {embedder_type}")
    q_display = embedder_quantization or "none"
    print(f"Embedder quantization: {q_display}")
    print(f"Trials per reference text: {trials}")
    print(f"Embed batch size: {embed_batch_size}")
    print(f"Device: {device}")
    
    # Set up output directory structure
    output_folder.mkdir(parents=True, exist_ok=True)
    obfuscated_folder = output_folder / "obfuscated"
    comparisons_folder = output_folder / "comparisons"
    output_csv = output_folder / "results.csv"
    params_file = output_folder / "params.json"
    report_file = output_folder / "report.txt"
    
    obfuscated_folder.mkdir(parents=True, exist_ok=True)
    comparisons_folder.mkdir(parents=True, exist_ok=True)
    
    # Save parameters for reproducibility (and mirror into report.txt Configuration section).
    params = {
        "text_folder": str(text_folder.absolute()),
        "approach": approach,
        "entities": entities,
        "concept": concept,
        "concept_model": concept_model or DEFAULT_CONCEPT_MODEL,
        "concept_temperature": concept_temperature,
        "base_concepts": base_concepts,
        "instances": instances,
        "mode": mode,
        "trials": trials,
        "samples": samples,
        "min_coverage": min_coverage,
        "max_coverage": max_coverage,
        "seed": seed,
        "device": device,
        "ner_model": ner_model or DEFAULT_MODEL,
        "threshold": threshold,
        "embedder_type": embedder_type,
        "embedder_model": embedder_model,
        "embedder_quantization": embedder_quantization or "none",
        "embed_batch_size": embed_batch_size,
        "placeholder": placeholder,
        "compact_blackout_words": compact_blackout_words,
        "sequential_labels": sequential_labels,
        "propagate": propagate,
        "continue_from_output": continue_from_output,
        "obfuscate_missing_in_continue": obfuscate_missing_in_continue,
        "generate_comparisons": generate_comparisons,
        "write_analysis_artifacts": write_analysis_artifacts,
        "retry_skipped_paths_count": len(retry_skipped_paths) if retry_skipped_paths else 0,
        "timestamp": datetime.now().isoformat(),
        "command_line": command_line,
    }
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    
    print(f"Output folder: {output_folder}")
    print(f"  Obfuscated: {obfuscated_folder}")
    print(f"  Comparisons: {comparisons_folder}")
    print(f"  Results CSV: {output_csv}")
    print(f"  Parameters: {params_file}")
    
    # Pre-load models for efficiency
    print("\n" + "=" * 60)
    print("Loading models (this will be reused for all texts)")
    print("=" * 60)
    
    recognizer = None
    if approach == "entity" or base_concepts:
        print(f"Loading GLiNER2 NER model ({ner_model_display})...")
        recognizer = load_recognizer(model_name=ner_model, device=device)
    
    # Load text embedder
    print(f"Loading {embedder_type} embedder for text comparison...")
    embedder = load_text_embedder(
        model_type=embedder_type,
        model_name=embedder_model,
        device=device,
        embedder_quantization=embedder_quantization,
    )
    
    print("All models loaded successfully!")
    
    # Step 1: Sample reference texts and determine which texts need obfuscation
    print()
    print("=" * 60)
    print("Step 1: Selecting texts to process")
    print("=" * 60)
    
    # Sample reference texts if requested
    if samples is not None and samples < len(text_files):
        reference_texts = random.sample(text_files, samples)
        print(f"Randomly selected {samples} reference texts from {len(text_files)} available")
    else:
        reference_texts = list(text_files)
        print(f"Using all {len(reference_texts)} texts as reference texts")
    
    # For each reference text, pre-select the test texts
    reference_to_tests: dict[Path, list[Path]] = {}
    all_texts_to_obfuscate: set[Path] = set()
    
    for u_path in reference_texts:
        other_texts = [p for p in text_files if p != u_path]
        num_test_samples = min(trials, len(other_texts))
        if num_test_samples == len(other_texts):
            # Deterministic: no randomness when using every other text as a trial
            test_texts = sorted(other_texts, key=lambda p: p.name)
        else:
            test_texts = random.sample(other_texts, num_test_samples)
        reference_to_tests[u_path] = test_texts
        
        all_texts_to_obfuscate.add(u_path)
        all_texts_to_obfuscate.update(test_texts)
    
    texts_to_obfuscate = sorted(all_texts_to_obfuscate)
    print(f"Total texts to obfuscate: {len(texts_to_obfuscate)}")
    
    # Step 2: Load texts and create obfuscated versions
    print()
    print("=" * 60)
    print("Step 2: Creating obfuscated versions")
    print("=" * 60)
    
    # Store original and obfuscated texts
    original_texts: dict[Path, str] = {}
    obfuscated_texts: dict[Path, str] = {}
    valid_obfuscations: set[Path] = set()

    retry_set: set[str] = set()
    if retry_skipped_paths:
        # Match by filename, to be resilient to absolute path changes
        retry_set = {Path(p).name for p in retry_skipped_paths}

    # When --continue or --retry: load existing obfuscated texts from output folder
    preloaded_obfuscated: dict[str, str] = {}
    if (continue_from_output or retry_skipped_paths) and obfuscated_folder.exists():
        for obf_path in obfuscated_folder.glob("obfuscated_*"):
            try:
                original_name = obf_path.name[len("obfuscated_") :]
                preloaded_obfuscated[original_name] = obf_path.read_text(encoding="utf-8")
            except Exception:
                continue
    
    for i, text_path in enumerate(texts_to_obfuscate):
        print(f"\n[{i+1}/{len(texts_to_obfuscate)}] Processing: {text_path.name}")
        
        try:
            # Read original text
            original_text = text_path.read_text(encoding="utf-8")
            original_texts[text_path] = original_text

            # Skip if we already have this text (from --continue). In retry mode we never skip.
            if (
                preloaded_obfuscated
                and text_path.name in preloaded_obfuscated
                and text_path.name not in retry_set
            ):
                obf_text = preloaded_obfuscated[text_path.name]
                obfuscated_texts[text_path] = obf_text
                # Assume previously validated; we at least ensure it changed the text.
                if obf_text != original_text:
                    valid_obfuscations.add(text_path)
                print("  Reusing obfuscated text from output folder")
                continue

            if (
                continue_from_output
                and not retry_skipped_paths
                and not obfuscate_missing_in_continue
                and text_path.name not in preloaded_obfuscated
            ):
                print(
                    "  Skipping: missing existing obfuscated text and "
                    "continue-only mode is enabled"
                )
                continue
            
            # Create obfuscated version
            obf_text, coverage, base_coverage, _primary_coverage = create_obfuscated_text(
                text=original_text,
                entity_types=entities,
                mode=mode,
                threshold=threshold,
                model_name=ner_model,
                device=device,
                recognizer=recognizer,
                placeholder=placeholder,
                sequential_labels=sequential_labels,
                propagate=propagate,
                approach=approach,
                concept=concept,
                concept_model=concept_model,
                concept_temperature=concept_temperature,
                base_concepts=base_concepts,
                base_recognizer=recognizer,
                instances=instances,
                compact_blackout_words=compact_blackout_words,
            )
            obfuscated_texts[text_path] = obf_text
            
            # Check if obfuscation actually changed the text
            coverage_fraction = coverage / 100.0
            base_coverage_fraction = base_coverage / 100.0
            base_modified = bool(base_concepts and base_coverage_fraction > 0)
            if coverage_fraction < min_coverage:
                if base_modified:
                    print(
                        f"  WARNING: Total coverage for {text_path.name} is below --min-coverage "
                        f"(coverage={coverage_fraction:.4f}, min={min_coverage:.4f}), but base concepts modified the text; saving anyway."
                    )
                    valid_obfuscations.add(text_path)
                else:
                    print(f"  WARNING: Obfuscation did not change {text_path.name} (coverage={coverage_fraction:.4f})")
            elif coverage_fraction > max_coverage:
                print(
                    f"  WARNING: Text {text_path.name} cannot be made private: estimated "
                    f"redaction/changed fraction {coverage_fraction:.2%} exceeds --max-coverage "
                    f"limit {max_coverage:.2%}."
                )
            else:
                valid_obfuscations.add(text_path)
                print(f"  Obfuscation successful (coverage={coverage_fraction:.4f})")
            
            # Save obfuscated text
            obf_output_path = obfuscated_folder / f"obfuscated_{text_path.name}"
            obf_output_path.write_text(obf_text, encoding="utf-8")
            
        except Exception as e:
            print(f"  WARNING: Failed to obfuscate {text_path.name}: {e}")
            continue
    
    print(f"\nSuccessfully obfuscated {len(obfuscated_texts)} texts")
    print(f"Valid obfuscations (text changed): {len(valid_obfuscations)}")
    
    # Create summary grid images (originals and obfuscated in same order)
    if obfuscated_texts:
        # Sort by filename so both grids are in the same order
        sorted_paths = sorted(obfuscated_texts.keys(), key=lambda p: p.name)
        grid_originals = [original_texts[p] for p in sorted_paths if p in original_texts]
        grid_obfuscated = [obfuscated_texts[p] for p in sorted_paths]
        grid_labels = [f"passage {i + 1}" for i in range(len(sorted_paths))]

        summary_originals_path = output_folder / "summary_originals.pdf"
        print(f"\nCreating originals summary grid: {summary_originals_path}")
        create_text_summary_grid(
            texts=grid_originals,
            labels=grid_labels,
            output_path=summary_originals_path,
        )
        print(f"Originals summary grid saved with {len(grid_originals)} texts")

        summary_obfuscated_path = output_folder / "summary_obfuscated.pdf"
        print(f"Creating obfuscated summary grid: {summary_obfuscated_path}")
        create_text_summary_grid(
            texts=grid_obfuscated,
            labels=[f"passage {i + 1} (obfuscated)" for i in range(len(sorted_paths))],
            output_path=summary_obfuscated_path,
        )
        print(f"Obfuscated summary grid saved with {len(grid_obfuscated)} texts")
    
    # Step 3: Pre-compute embeddings for all needed texts
    print("\n" + "=" * 60)
    print("Step 3: Pre-computing text embeddings")
    print("=" * 60)
    
    # Get list of texts with valid obfuscations
    valid_test_candidates = [p for p in text_files if p in valid_obfuscations]
    print(f"Available test texts with valid obfuscations: {len(valid_test_candidates)}")
    
    # One embedding per (path, role): same string in two files must not share a cache slot
    # (image resolution uses Path keys; text previously keyed by string, which was wrong).
    keyed_rows: list[tuple[Path, str, str]] = []
    for text_path in sorted(valid_test_candidates, key=lambda p: p.name):
        orig = original_texts.get(text_path)
        if orig is not None:
            keyed_rows.append((text_path, "orig", orig))
        obf = obfuscated_texts.get(text_path)
        if obf is not None:
            keyed_rows.append((text_path, "obf", obf))

    flat_texts = [t for _, _, t in keyed_rows]
    print(
        f"Computing embeddings for {len(flat_texts)} text slots "
        f"({len(valid_test_candidates)} files × original + obfuscated)..."
    )

    ordered_embeddings = embedder.embed_batch_ordered(
        flat_texts, batch_size=max(1, embed_batch_size)
    )
    path_tag_embeddings: dict[tuple[Path, str], torch.Tensor] = {
        (path, tag): emb for (path, tag, _), emb in zip(keyed_rows, ordered_embeddings)
    }
    print(f"Embeddings computed for {len(path_tag_embeddings)} path/role pairs")
    
    # Step 4: Compute resolutions
    print("\n" + "=" * 60)
    print("Step 4: Computing resolutions")
    print("=" * 60)
    
    results = []
    references_processed = 0
    references_skipped = 0
    skipped_reference_details: list[tuple[Path, str]] = []
    total_trials_requested = 0
    total_trials_actual = 0
    comparison_index = 0
    
    for i, u_path in enumerate(reference_texts):
        print(f"\n[{i+1}/{len(reference_texts)}] Processing reference: {u_path.name}")
        
        # Skip if reference text failed to obfuscate
        if u_path not in obfuscated_texts:
            print(f"  WARNING: Skipping {u_path.name} - obfuscation failed")
            references_skipped += 1
            skipped_reference_details.append((u_path, "obfuscation failed"))
            continue
        
        # Skip if reference text obfuscation didn't change anything
        if u_path not in valid_obfuscations:
            print(f"  WARNING: Skipping {u_path.name} - obfuscation did not change the text")
            references_skipped += 1
            skipped_reference_details.append((u_path, "obfuscation did not change the text"))
            continue
        
        xu_text = obfuscated_texts[u_path]
        
        # Get valid test texts (excluding the reference text itself)
        available_tests = [p for p in valid_test_candidates if p != u_path]
        
        if len(available_tests) == 0:
            print(f"  WARNING: Skipping {u_path.name} - no valid test texts available")
            references_skipped += 1
            skipped_reference_details.append((u_path, "no valid test texts available"))
            continue
        
        # Sample test texts from valid candidates
        actual_trials = min(trials, len(available_tests))
        if actual_trials < trials:
            print(f"  WARNING: Only {actual_trials} valid test texts available "
                  f"(requested {trials})")
        if actual_trials == len(available_tests):
            test_texts = sorted(available_tests, key=lambda p: p.name)
        else:
            test_texts = random.sample(available_tests, actual_trials)
        
        # Track trial counts
        references_processed += 1
        total_trials_requested += trials
        total_trials_actual += actual_trials
        
        # Compute resolution for each test text
        ref_results = []
        for v_path in test_texts:
            v_text = original_texts[v_path]
            xv_text = obfuscated_texts[v_path]
            
            try:
                # Get embeddings (by path + role; not by raw string — avoids collisions)
                xu_emb = path_tag_embeddings[(u_path, "obf")]
                v_emb = path_tag_embeddings[(v_path, "orig")]
                xv_emb = path_tag_embeddings[(v_path, "obf")]
                
                # d1 = 1 - similarity(X(u), v)
                sim_xu_v = similarity_from_embeddings(xu_emb, v_emb)
                d1 = 1 - sim_xu_v
                
                # d2 = 1 - similarity(X(u), X(v))
                sim_xu_xv = similarity_from_embeddings(xu_emb, xv_emb)
                d2 = 1 - sim_xu_xv
                
                res = d2 - d1
                
                u_text = original_texts[u_path]
                if generate_comparisons:
                    res_str = f"{res:.4f}".replace("-", "n").replace(".", "p")
                    comparison_filename = f"cmp_{comparison_index:04d}_res{res_str}.txt"
                    comparison_path = comparisons_folder / comparison_filename
                    create_comparison_text(
                        u_text=u_text,
                        xu_text=xu_text,
                        v_text=v_text,
                        xv_text=xv_text,
                        u_name=u_path.name,
                        v_name=v_path.name,
                        resolution=res,
                        output_path=comparison_path,
                    )
                    comparison_index += 1
                    comparison_abs = str(comparison_path.absolute())
                else:
                    comparison_filename = ""
                    comparison_abs = ""

                ref_results.append({
                    "resolution": res,
                    "comparison": comparison_abs,
                    "comparison_filename": comparison_filename,
                    "u": str(u_path.absolute()),
                    "v": str(v_path.absolute()),
                    "v_name": v_path.name,
                    "sim_xu_v": sim_xu_v,
                    "sim_xu_xv": sim_xu_xv,
                })
            except Exception as e:
                print(f"  WARNING: Failed to compute resolution for {v_path.name}: {e}")
                continue
        
        # Sort and print results for this reference text
        ref_results.sort(key=lambda x: x["resolution"])
        for r in ref_results:
            print(f"    {r['resolution']:+.4f}  vs {r['v_name']}")
        
        # Add to overall results
        results.extend(ref_results)
    
    # Step 5: Write results to CSV
    print("\n" + "=" * 60)
    print("Step 5: Writing results to CSV")
    print("=" * 60)
    
    # Sort all results by resolution
    results.sort(key=lambda x: x["resolution"])
    
    # Prepare rows for CSV
    fieldnames = ["resolution", "comparison_filename", "u", "v", "sim_xu_v", "sim_xu_xv"]
    csv_rows = [{k: r.get(k, "") for k in fieldnames} for r in results]
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # Build report content
    report_lines = []
    
    def report(line: str = "") -> None:
        """Print a line and add it to the report."""
        print(line)
        report_lines.append(line)
    
    report("=" * 60)
    report("Text Resolution Analysis Report")
    report("=" * 60)
    report()
    report(f"Timestamp: {datetime.now().isoformat()}")
    report(f"Input folder: {text_folder}")
    report(f"Output folder: {output_folder}")
    report()
    report("Configuration:")
    for _param_key in _PARAMS_REPORT_KEYS:
        _label = _params_report_label(_param_key)
        report(
            f"  {_label}: {_format_params_report_value(_param_key, params.get(_param_key))}"
        )
    _extra_param_keys = sorted(k for k in params if k not in _PARAMS_REPORT_KEYS)
    for _param_key in _extra_param_keys:
        _label = _params_report_label(_param_key)
        report(
            f"  {_label}: {_format_params_report_value(_param_key, params.get(_param_key))}"
        )
    report()
    report("Output files:")
    report(f"  Results CSV: {output_csv}")
    report(f"  Comparisons: {comparisons_folder}")
    report(f"  Obfuscated texts: {obfuscated_folder}")
    report(f"  Summary originals: {output_folder / 'summary_originals.pdf'}")
    report(f"  Summary obfuscated: {output_folder / 'summary_obfuscated.pdf'}")
    report(f"  Parameters: {params_file}")
    report(f"  Report: {report_file}")
    report()
    report("Summary:")
    report(f"  Reference texts processed: {references_processed}")
    report(f"  Reference texts skipped: {references_skipped}")
    report(f"  Trials requested: {total_trials_requested}")
    report(f"  Trials completed: {total_trials_actual} ({len(results)} successful comparisons)")
    if total_trials_requested > 0 and total_trials_actual < total_trials_requested:
        report(f"  NOTE: {total_trials_requested - total_trials_actual} trials could not be completed "
               f"due to insufficient valid test texts")
    if skipped_reference_details:
        report()
        report("Skipped reference texts (detailed):")
        for p, reason in skipped_reference_details:
            report(f"  - Path: {str(p.absolute())}")
            report(f"    Reason: {reason}")
    report()
    report(f"To reproduce this run, use: --seed {seed}")
    if command_line:
        report(f"Command line: {command_line}")
    
    # Step 6: Report privacy violations and histogram
    report()
    report("=" * 60)
    report("Privacy Analysis")
    report("=" * 60)
    
    if results:
        # Group results by reference text to detect violations
        results_by_reference: dict[str, list[dict]] = {}
        for r in results:
            u_key = r["u"]
            if u_key not in results_by_reference:
                results_by_reference[u_key] = []
            results_by_reference[u_key].append(r)
        
        # A privacy violation occurs if any comparison has resolution > 0
        violations = []
        for u_key, ref_results in results_by_reference.items():
            max_result = max(ref_results, key=lambda x: x["resolution"])
            max_res = max_result["resolution"]
            if max_res > 0:
                cmp_file = max_result.get("comparison_filename", "")
                violations.append((u_key, max_res, max_result["v_name"], cmp_file))
        
        num_violations = len(violations)
        num_references = len(results_by_reference)
        
        report()
        report("Privacy Violations:")
        report(f"  {num_violations} of {num_references} reference texts have at least one resolution > 0")
        report(f"  Violation rate: {num_violations / num_references * 100:.1f}%")
        if references_skipped > 0:
            report()
            report(f"  Note: {references_skipped} reference texts could not be analyzed because obfuscation failed")
        
        if violations:
            report()
            report("  Texts with violations (showing max resolution):")
            violations.sort(key=lambda x: x[1], reverse=True)
            for u_key, max_res, v_name, cmp_file in violations[:10]:
                u_name = Path(u_key).name
                report(f"    {u_name}: max resolution = {max_res:+.4f} (vs {v_name}) [{cmp_file}]")
            if len(violations) > 10:
                report(f"    ... and {len(violations) - 10} more")
        
        # Create histogram
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
                text_embedder=embedder_type,
                text_embedder_model=embedder_model,
                text_embedder_quantization=embedder_quantization,
                text_folder=text_folder,
                refresh=True,
            )
            print(f"Analysis page saved to: {artifacts['html_path']}")
            print(f"Analysis bundle saved to: {artifacts['json_path']}")
        except Exception as exc:
            print(f"WARNING: Failed to generate analysis artifacts: {exc}")


def main() -> None:
    """CLI entry point for text resolution analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze effective resolution of text obfuscation methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze person name redaction with 5 trials per text
  %(prog)s ./texts --entities person --mode blackout --trials 5 --output ./results
  
  # Analyze multiple entity types
  %(prog)s ./texts --entities person organization location --mode blackout --trials 5 --output ./results
  
  # Use CLIP embeddings instead of SBERT
  %(prog)s ./texts --entities person --mode blackout --trials 5 --embedder clip --output ./results

  # Use Qwen3 8B embeddings
  %(prog)s ./texts --entities person --mode blackout --trials 5 --embedder qwen --output ./results

  # Qwen3 8B with float16 weights (no bitsandbytes; less VRAM than fp32)
  %(prog)s ./texts --entities person --mode blackout --trials 5 --embedder qwen \\
      --embedder-quantization half --output ./results

  # Qwen3 8B with 4-bit weights (CUDA + bitsandbytes; less VRAM, often faster)
  %(prog)s ./texts --entities person --mode blackout --trials 5 --embedder qwen \\
      --embedder-quantization 4bit --output ./results

  # Use EVA-CLIP text encoder (e.g. BAAI/EVA-CLIP-18B)
  %(prog)s ./texts --entities person --mode blackout --trials 5 --embedder clip \\
      --embedder-model BAAI/EVA-CLIP-18B --output ./results
  
  # Use the larger GLiNER2 model
  %(prog)s ./texts --entities person --mode blackout --trials 5 --ner-model fastino/gliner2-large-v1 --output ./results
  
  # Sample only 10 reference texts
  %(prog)s ./texts --entities person --mode blackout --trials 3 --samples 10 --output ./results

  # Retry obfuscation for skipped reference texts (uses params from output folder)
  %(prog)s --retry --output ./results

  # Retry with overrides (e.g. lower threshold)
  %(prog)s ./texts --retry --output ./results --threshold 0.2

Output folder structure:
  The --output folder will contain:
    - comparisons/: Comparison files showing u, X(u), v, X(v) side by side
    - obfuscated/: Obfuscated versions of each input text
    - summary_originals.pdf: Grid of all original texts (PDF)
    - summary_obfuscated.pdf: Grid of all obfuscated texts (PDF, same positions)
    - results.csv: Resolution analysis results
    - params.json: All parameters used (for reproducibility)
    - report.txt: Summary report with privacy analysis

Resolution interpretation:
  The resolution (d2 - d1) measures how well the obfuscation preserves
  semantic relationships between texts while obscuring identity:
  
  - d1 = 1 - similarity(X(u), v)
    Distance from obfuscated to another text's original
  
  - d2 = 1 - similarity(X(u), X(v))
    Distance between obfuscated versions
  
  Negative resolution means obfuscated texts are MORE similar to each
  other than originals are to other obfuscated texts (good for privacy).
        """,
    )
    
    parser.add_argument(
        "text_folder",
        type=str,
        nargs="?",
        default=None,
        help="Path to folder containing text files to analyze. With --retry, defaults to the path stored in params.json.",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="entity",
        choices=["entity", "concept"],
        help="Obfuscation approach: 'entity' (NER with GLiNER2) or 'concept' (fal.ai OpenRouter by description). Default: entity.",
    )
    parser.add_argument(
        "--entities", "-e",
        type=str,
        nargs="+",
        default=None,
        help="Entity types to detect and obfuscate (required when --approach entity). E.g. 'person' 'organization'.",
    )
    parser.add_argument(
        "--base-concepts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional base concepts (words/phrases) to obfuscate after primary obfuscation "
            "using GLiNER2 entity detection."
        ),
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Literal substrings to obfuscate (case-insensitive; regex with re.escape). "
            "Every occurrence is obfuscated in full, including inside larger words. "
            "Deterministic; does not invoke GLiNER2."
        ),
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Privacy concept description (required when --approach concept). "
             "E.g. 'anything that can identify the movie discussed in this passage'.",
    )
    parser.add_argument(
        "--concept-model",
        type=str,
        default=None,
        help=f"fal.ai OpenRouter model: preset (gpt-5.4, gemini-3.1-pro, opus-4.6) or exact OpenRouter ID (e.g. google/gemini-3.1-pro-preview). Default: {DEFAULT_CONCEPT_MODEL}. Requires FAL_KEY.",
    )
    parser.add_argument(
        "--concept-temperature",
        type=float,
        default=DEFAULT_CONCEPT_TEMPERATURE,
        help=f"OpenRouter sampling temperature for --approach concept (default: {DEFAULT_CONCEPT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["blackout", "redact"],
        required=True,
        help="Obfuscation mode: 'blackout' (block chars) or 'redact' (placeholder).",
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        required=True,
        help="Number of test texts to sample per reference text.",
    )
    parser.add_argument(
        "--output", "-O",
        type=str,
        required=True,
        help="Output folder. Will contain: obfuscated/, results.csv, params.json, and report.txt.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_from_output",
        action="store_true",
        help="Resume from existing output folder: reuse obfuscated texts in output/obfuscated/ and only obfuscate missing texts.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry obfuscation for skipped reference texts listed in report.txt. Uses parameters from params.json in the output folder; command-line arguments override. Requires --output to point to a previous run.",
    )
    parser.add_argument(
        "--skip-analysis-artifacts",
        dest="write_analysis_artifacts",
        action="store_false",
        help="Do not auto-generate analysis_report.html and analysis_report.json at the end of the run.",
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=None,
        help="Number of reference texts to sample (default: use all texts).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=f"Minimum coverage fraction (0.0-1.0) for valid obfuscation (default: {DEFAULT_MIN_COVERAGE}).",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=DEFAULT_MAX_COVERAGE,
        help=f"Maximum coverage fraction (0.0-1.0) for valid obfuscation (default: {DEFAULT_MAX_COVERAGE}).",
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
        "--ner-model",
        type=str,
        default=None,
        help="GLiNER2 model name (default: fastino/gliner2-base-v1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Entity detection threshold (default: 0.3). Lower = more detections.",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="sbert",
        choices=["clip", "sbert", "qwen"],
        help=(
            "Text embedder: 'sbert' (sentence-transformers), "
            "'qwen' (Qwen/Qwen3-Embedding-8B), or 'clip'. Default: sbert."
        ),
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help=(
            "Embedder model name (default: sentence-transformers/all-MiniLM-L6-v2 when "
            "--embedder sbert; Qwen/Qwen3-Embedding-8B when --embedder qwen; "
            "CLIP default when --embedder clip). "
            f"EVA-CLIP example: {EVA_CLIP_TEXT_EMBEDDER_EXAMPLE}."
        ),
    )
    parser.add_argument(
        "--embedder-quantization",
        type=str,
        default=None,
        choices=["none", "half", "4bit", "8bit"],
        help=(
            "For --embedder sbert or qwen only: none (default dtype), half (float16 weights, "
            "no bitsandbytes), or bitsandbytes 4-bit/8-bit (CUDA; dependency of this package). "
            "Reduces VRAM for large models (e.g. Qwen3-Embedding-8B). Default: none."
        ),
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Batch size when pre-computing embeddings (Step 3). Large models on ~16GB GPUs often "
            "need 1–2. Default: 8."
        ),
    )
    parser.add_argument(
        "--placeholder",
        type=str,
        default="[REDACTED]",
        help="Placeholder text for redact mode (default: [REDACTED]).",
    )
    parser.add_argument(
        "--compact-blackout-words",
        action="store_true",
        help=(
            "For --mode blackout: use one block per obfuscated word/token "
            "(e.g. 'John Smith' -> '█ █') instead of one block per character."
        ),
    )
    parser.add_argument(
        "--sequential-labels",
        action="store_true",
        help="Process each entity type separately and merge results. "
             "This ensures strictly additive behavior: adding more entity types can never "
             "reduce the number of detections. Use this when you observe that combining entity "
             "types causes some detections to drop below the threshold. "
             "Increases runtime proportionally with the number of entity types.",
    )
    parser.add_argument(
        "--propagate",
        action="store_true",
        default=True,
        help=(
            "Find ALL occurrences of each detected entity and anonymize every one "
            "(default). NER models often tag only one occurrence of a repeated name; "
            "propagation ensures consistent anonymization."
        ),
    )
    parser.add_argument(
        "--no-propagate",
        action="store_false",
        dest="propagate",
        help="Only anonymize the specific spans returned by the NER model (do not "
             "propagate to other occurrences of the same entity text).",
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

        # In retry mode we should default to the input folder from params.json
        # if the user-provided positional folder is missing or invalid.
        text_folder_effective = Path(args.text_folder) if args.text_folder else Path(params["text_folder"])
        if not text_folder_effective.exists() or not text_folder_effective.is_dir():
            print(
                f"WARNING: Provided text_folder for --retry not found/invalid: {text_folder_effective}. "
                f"Falling back to params.json text_folder: {params['text_folder']}"
            )
            text_folder_effective = Path(params["text_folder"])

        skipped_paths = parse_report_skipped_paths(report_file)
        if not skipped_paths:
            # Backward compatibility: older report.txt files may only contain a count
            # ("Reference texts skipped: N") without a detailed list. In that case,
            # infer skipped references by reconstructing the sampled reference set
            # and subtracting the reference texts that produced rows in results.csv.
            text_folder_for_inference = text_folder_effective
            seed_for_inference = params.get("seed")
            samples_for_inference = params.get("samples")

            if seed_for_inference is not None:
                try:
                    text_files_for_inference = get_text_files(text_folder_for_inference)
                    random.seed(seed_for_inference)
                    if (
                        samples_for_inference is not None
                        and isinstance(samples_for_inference, int)
                        and samples_for_inference < len(text_files_for_inference)
                    ):
                        reference_texts_for_inference = random.sample(
                            text_files_for_inference, samples_for_inference
                        )
                    else:
                        reference_texts_for_inference = list(text_files_for_inference)

                    results_csv = output_folder / "results.csv"
                    if results_csv.exists():
                        processed_u: set[str] = set()
                        with open(results_csv, "r", encoding="utf-8", newline="") as f:
                            reader = csv.DictReader(f)
                            if reader.fieldnames and "u" in reader.fieldnames:
                                for row in reader:
                                    u_val = (row.get("u") or "").strip()
                                    if u_val:
                                        processed_u.add(u_val)

                        # "Skipped references" are those sampled reference texts without any produced rows.
                        if processed_u:
                            skipped_paths = [
                                p for p in reference_texts_for_inference if str(p.absolute()) not in processed_u
                            ]
                    else:
                        # Fallback if results.csv isn't present: treat missing/unchanged obfuscated files as skipped.
                        obfuscated_folder_for_inference = output_folder / "obfuscated"
                        for u_path in reference_texts_for_inference:
                            obf_path = obfuscated_folder_for_inference / f"obfuscated_{u_path.name}"
                            if not obf_path.exists():
                                skipped_paths.append(u_path)
                                continue
                            try:
                                obf_text = obf_path.read_text(encoding="utf-8")
                                orig_text = u_path.read_text(encoding="utf-8")
                                if obf_text == orig_text:
                                    skipped_paths.append(u_path)
                            except Exception:
                                # If we can't read/compare, be conservative and retry.
                                skipped_paths.append(u_path)
                except Exception:
                    skipped_paths = []

            if not skipped_paths:
                print(
                    "No skipped reference texts found (neither detailed report nor inferred). Nothing to retry."
                )
                return

            print(
                f"Retry mode: inferred {len(skipped_paths)} skipped reference text(s) from params/obfuscated folder."
            )
        else:
            print(f"Retry mode: found {len(skipped_paths)} skipped reference text(s) in report.txt")

        def _use(from_args, from_params, option_strings):
            return from_args if _argv_contains_any(argv, option_strings) else from_params

        text_folder = text_folder_effective
        approach = _use(args.approach, params.get("approach", "entity"), ["--approach"])
        mode = _use(args.mode, params["mode"], ["--mode", "-m"])
        trials = _use(args.trials, params["trials"], ["--trials", "-t"])
        samples = _use(args.samples, params.get("samples"), ["--samples", "-s"])
        if samples is None and "samples" in params:
            samples = params["samples"]
        min_coverage = _use(args.min_coverage, params["min_coverage"], ["--min-coverage"])
        max_coverage = _use(args.max_coverage, params["max_coverage"], ["--max-coverage"])
        seed = _use(args.seed, params.get("seed"), ["--seed"])
        device = _use(args.device, params.get("device"), ["--device"])
        ner_model = _use(args.ner_model, params.get("ner_model"), ["--ner-model"])
        threshold = _use(args.threshold, params.get("threshold", 0.3), ["--threshold"])
        embedder_type = _use(args.embedder, params.get("embedder_type", "sbert"), ["--embedder"])
        embedder_model = _use(args.embedder_model, params.get("embedder_model"), ["--embedder-model"])
        embedder_quantization = _use(
            args.embedder_quantization,
            params.get("embedder_quantization", "none"),
            ["--embedder-quantization"],
        )
        if embedder_quantization in ("4bit", "8bit"):
            if not torch.cuda.is_available():
                parser.error(
                    f"--embedder-quantization {embedder_quantization} requires CUDA "
                    "(bitsandbytes is not available on CPU)."
                )
            if args.device == "cpu":
                parser.error(
                    f"--embedder-quantization {embedder_quantization} cannot be used with --device cpu. "
                    "Omit --device or use --device cuda."
                )
            if device == "cpu":
                print(
                    "Note: bitsandbytes 4/8-bit embedders require CUDA; using device=cuda "
                    f"(overriding prior params device={params.get('device')!r})."
                )
                device = "cuda"
        placeholder = _use(args.placeholder, params.get("placeholder", "[REDACTED]"), ["--placeholder"])
        compact_blackout_words = _use(
            args.compact_blackout_words,
            params.get("compact_blackout_words", False),
            ["--compact-blackout-words"],
        )
        sequential_labels = _use(
            args.sequential_labels, params.get("sequential_labels", False), ["--sequential-labels"]
        )

        # propagate can be overridden by either flag
        if _argv_contains_any(argv, ["--no-propagate"]):
            propagate = False
        elif _argv_contains_any(argv, ["--propagate"]):
            propagate = True
        else:
            propagate = params.get("propagate", True)

        entities_list = _use(args.entities, params.get("entities", []), ["--entities", "-e"]) or params.get(
            "entities", []
        )
        base_concepts = _use(args.base_concepts, params.get("base_concepts"), ["--base-concepts"]) or params.get(
            "base_concepts"
        )
        instances = _use(args.instances, params.get("instances"), ["--instances"]) or params.get(
            "instances"
        )
        concept = _use(args.concept, params.get("concept"), ["--concept"])
        concept_model = _use(args.concept_model, params.get("concept_model"), ["--concept-model"])
        concept_temperature = _use(
            args.concept_temperature,
            params.get("concept_temperature", DEFAULT_CONCEPT_TEMPERATURE),
            ["--concept-temperature"],
        )
        embed_batch_size = _use(
            args.embed_batch_size,
            params.get("embed_batch_size", 8),
            ["--embed-batch-size"],
        )
        write_analysis_artifacts = _use(
            args.write_analysis_artifacts,
            params.get("write_analysis_artifacts", True),
            ["--skip-analysis-artifacts"],
        )

        if not text_folder.exists():
            parser.error(f"Text folder not found: {text_folder}")
        if not text_folder.is_dir():
            parser.error(f"Not a directory: {text_folder}")
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

        if approach == "entity":
            if not entities_list and not instances:
                parser.error(
                    "When --approach is 'entity', provide --entities and/or --instances "
                    "(either in params.json or on the CLI)."
                )
        else:
            if not concept or not str(concept).strip():
                parser.error("When --approach is 'concept', --concept is required and must be non-empty.")

        run_text_resolution_analysis(
            text_folder=text_folder,
            entities=entities_list if approach == "entity" else [],
            mode=mode,
            trials=trials,
            output_folder=output_folder,
            samples=samples,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
            seed=seed,
            device=device,
            ner_model=ner_model,
            threshold=threshold,
            embedder_type=embedder_type,
            embedder_model=embedder_model,
            embedder_quantization=embedder_quantization,
            embed_batch_size=embed_batch_size,
            placeholder=placeholder,
            compact_blackout_words=compact_blackout_words,
            sequential_labels=sequential_labels,
            propagate=propagate,
            approach=approach,
            concept=concept,
            concept_model=concept_model,
            concept_temperature=concept_temperature,
            base_concepts=base_concepts,
            instances=instances,
            continue_from_output=True,
            retry_skipped_paths=skipped_paths,
            command_line=shlex.join(sys.argv),
            write_analysis_artifacts=write_analysis_artifacts,
        )
        return

    # Validate arguments (non-retry)
    if args.approach == "entity":
        if not args.entities and not args.instances:
            parser.error("When --approach is 'entity', provide --entities and/or --instances.")
    else:
        if not args.concept or not args.concept.strip():
            parser.error("When --approach is 'concept', --concept is required and must be non-empty.")

    if not args.text_folder:
        parser.error("the following arguments are required: text_folder")
    text_folder = Path(args.text_folder)
    if not text_folder.exists():
        parser.error(f"Text folder not found: {text_folder}")
    if not text_folder.is_dir():
        parser.error(f"Not a directory: {text_folder}")

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

    _eq = args.embedder_quantization or "none"
    if _eq in ("4bit", "8bit"):
        if not torch.cuda.is_available():
            parser.error(
                f"--embedder-quantization {_eq} requires CUDA "
                "(bitsandbytes is not available on CPU)."
            )
        if args.device == "cpu":
            parser.error(
                f"--embedder-quantization {_eq} cannot be used with --device cpu. "
                "Omit --device or use --device cuda."
            )

    entities_list = args.entities if args.approach == "entity" else []
    run_text_resolution_analysis(
        text_folder=text_folder,
        entities=entities_list,
        mode=args.mode,
        trials=args.trials,
        output_folder=Path(args.output),
        samples=args.samples,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        seed=args.seed,
        device=args.device,
        ner_model=args.ner_model,
        threshold=args.threshold,
        embedder_type=args.embedder,
        embedder_model=args.embedder_model,
        embedder_quantization=args.embedder_quantization,
        embed_batch_size=args.embed_batch_size,
        placeholder=args.placeholder,
        compact_blackout_words=args.compact_blackout_words,
        sequential_labels=args.sequential_labels,
        propagate=args.propagate,
        approach=args.approach,
        concept=args.concept,
        concept_model=args.concept_model,
        concept_temperature=args.concept_temperature,
        base_concepts=args.base_concepts,
        instances=args.instances,
        continue_from_output=args.continue_from_output,
        command_line=shlex.join(sys.argv),
        write_analysis_artifacts=args.write_analysis_artifacts,
    )


if __name__ == "__main__":
    main()
