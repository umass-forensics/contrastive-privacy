#!/usr/bin/env python3
"""
Re-run resolution analysis on a prior run's obfuscated images.

Given a previous `resolution-analysis` output folder, this script:

- **Without** ``--base-concepts``: copies prior ``obfuscated/`` images into the new output
  folder and recomputes resolution (CLIP embeddings / similarity only), e.g. to try a
  different ``--embedder-model`` (e.g. EVA-02-CLIP).

- **With** ``--base-concepts``: applies SAM3 obfuscation for those concepts on top of the
  prior obfuscations, then recomputes resolution.

In both cases the original primary obfuscation is not re-run.
"""

import argparse
import shlex
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch

from contrastive_privacy.scripts.anonymize import (
    DEFAULT_FAL_VISION_TEMPERATURE,
    anonymize,
    load_sam3_models,
)
from contrastive_privacy.scripts.resolution_analysis import (
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_FAL_VISION_MODEL,
    EVA02_CLIP_EMBEDDER_MODEL,
    get_image_files,
    load_params_from_output,
    run_resolution_analysis,
)


def _resolve_image_folder(prior_output_folder: Path, params_image_folder: str) -> Path:
    """
    Resolve image folder from params.json with relocation-friendly fallbacks.

    Preference order:
    1) Stored path from params.json as-is (backward compatible).
    2) Relocation candidates derived from the provided input output folder.
       This handles copied runs where absolute paths in params.json are stale.
    """
    stored_path = Path(params_image_folder)
    if stored_path.exists() and stored_path.is_dir():
        return stored_path

    candidates: list[Path] = []
    if stored_path.is_absolute():
        # Rebase absolute path under the provided run folder when relocating across filesystems.
        candidates.append(prior_output_folder.joinpath(*stored_path.parts[1:]))
    else:
        candidates.append(prior_output_folder / stored_path)

    # Common relocation layouts.
    candidates.extend(
        [
            prior_output_folder,
            prior_output_folder / stored_path.name,
            prior_output_folder.parent / stored_path.name,
        ]
    )

    # Deduplicate while preserving order.
    deduped_candidates: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped_candidates.append(c)

    # Prefer candidates that look like the original corpus:
    # choose directory with best filename overlap against obfuscated_* outputs.
    expected_names = {
        p.name.removeprefix("obfuscated_")
        for p in (prior_output_folder / "obfuscated").glob("obfuscated_*")
        if p.is_file()
    }
    best_candidate: Optional[Path] = None
    best_score = -1
    for candidate in deduped_candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        candidate_images = get_image_files(candidate)
        if not candidate_images:
            continue
        if expected_names:
            score = sum(1 for p in candidate_images if p.name in expected_names)
        else:
            score = len(candidate_images)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is not None:
        print(
            "Stored image folder from params.json is unavailable; "
            f"using relocated image folder: {best_candidate}"
        )
        return best_candidate

    raise FileNotFoundError(
        "Could not resolve image folder from params.json.\n"
        f"  stored: {stored_path}\n"
        "  tried relocation candidates:\n"
        + "".join(f"    - {c}\n" for c in deduped_candidates)
        + "Provide/copy the source images so one of these paths exists."
    )


def _normalize_threshold_for_base(
    threshold: float | list[float], base_concepts: list[str]
) -> float | list[float]:
    """Match base-concept threshold behavior in resolution_analysis.create_obfuscated_image."""
    if isinstance(threshold, list):
        return threshold if len(threshold) == len(base_concepts) else threshold[0]
    return threshold


def _argv_contains_any(argv: list[str], option_strings: list[str]) -> bool:
    """Return True if any option string appears in argv."""
    for opt in option_strings:
        if opt in argv:
            return True
    return False


def _apply_base_concepts_on_prior_obfuscations(
    *,
    prior_output_folder: Path,
    new_output_folder: Path,
    image_folder: Path,
    base_concepts: list[str],
    mode: str,
    replacement_prompt: Optional[str],
    device: str,
    threshold: float | list[float],
    dilate: int,
    blur: int,
    strength: float,
    model: str,
    num_inference_steps: int,
    redact_blur_radius: int,
    seed: Optional[int],
    adaptive_blur: bool,
    blur_scale: float,
    size_exponent: float,
    scaling_factor: float,
    refinements: int,
    fal_image_model: str,
    fal_vision_model: str,
) -> None:
    prior_obfuscated = prior_output_folder / "obfuscated"
    new_obfuscated = new_output_folder / "obfuscated"
    new_obfuscated.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(image_folder)
    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 images in folder, found {len(image_files)}")

    print(f"Loading SAM3 models on {device} for base-concept obfuscation...")
    sam3_models = load_sam3_models(device)
    base_threshold = _normalize_threshold_for_base(threshold, base_concepts)

    total = len(image_files)
    processed = 0
    reused = 0
    missing = 0

    for idx, original_path in enumerate(image_files, start=1):
        prior_obf_path = prior_obfuscated / f"obfuscated_{original_path.name}"
        if not prior_obf_path.exists():
            print(f"[{idx}/{total}] Missing prior obfuscated image, skipping: {prior_obf_path.name}")
            missing += 1
            continue

        out_path = new_obfuscated / prior_obf_path.name
        if out_path.exists():
            print(f"[{idx}/{total}] Reusing existing reanalysis obfuscation: {out_path.name}")
            reused += 1
            continue

        print(f"[{idx}/{total}] Applying base-concept obfuscation: {prior_obf_path.name}")
        anonymize(
            input_path=prior_obf_path,
            output_path=out_path,
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
            sequential_labels=False,
            convex_hull=False,
            skip_empty_labels=False,
            refinements=refinements,
            fal_image_model=fal_image_model,
            fal_vision_model=fal_vision_model,
        )
        processed += 1

    print(
        f"Prep complete: processed={processed}, reused={reused}, "
        f"missing_prior_obfuscations={missing}"
    )


def _copy_prior_obfuscations_to_new_output(
    prior_output_folder: Path,
    new_output_folder: Path,
) -> None:
    """Copy obfuscated_* files from a prior run into the new output folder (no SAM3)."""
    prior_obfuscated = prior_output_folder / "obfuscated"
    new_obfuscated = new_output_folder / "obfuscated"
    if not prior_obfuscated.is_dir():
        raise FileNotFoundError(f"Missing obfuscated folder: {prior_obfuscated}")
    new_obfuscated.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for src in sorted(prior_obfuscated.iterdir()):
        if not src.is_file() or not src.name.startswith("obfuscated_"):
            continue
        dst = new_obfuscated / src.name
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(
        f"Copied {copied} prior obfuscated image(s) to {new_obfuscated} "
        f"(skipped {skipped} already present)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run resolution-analysis on a prior run: optionally add SAM3 base-concept "
            "obfuscation, or only recompute resolution with a chosen CLIP embedder."
        )
    )
    parser.add_argument(
        "input_output_folder",
        type=str,
        help="Path to an existing resolution-analysis output folder (must contain params.json and obfuscated/).",
    )
    parser.add_argument(
        "--base-concepts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional. If set, obfuscate these concepts with SAM3 on top of the prior run's "
            "obfuscated images. If omitted, prior obfuscations are copied as-is and only "
            "resolution is recomputed (e.g. with --embedder-model)."
        ),
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        required=True,
        help="Output folder for the new resolution analysis.",
    )
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Skip generation of comparison images while still computing resolution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for sampling/reanalysis (default: reuse prior run seed).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect).",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help=(
            "HuggingFace model for resolution image similarity (default: prior params.json, "
            f"else {DEFAULT_EMBEDDER_MODEL}). "
            f"EVA-02-CLIP example: {EVA02_CLIP_EMBEDDER_MODEL} (needs einops)."
        ),
    )
    # Accept resolution_analysis.py options as optional overrides.
    parser.add_argument("--objects", "-o", type=str, nargs="+", default=None)
    parser.add_argument("--privacy-concept", type=str, default=None)
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["replace", "redact", "blackout"],
        default=None,
    )
    parser.add_argument("--trials", "-t", type=int, default=None)
    parser.add_argument("--samples", "-s", type=int, default=None)
    parser.add_argument("--replace-prompt", type=str, default=None)
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--max-coverage", type=float, default=None)
    parser.add_argument(
        "--segmenter",
        type=str,
        choices=["groundedsam", "clipseg", "sam3", "ai-gen", "vlm-bounding-box"],
        default=None,
    )
    parser.add_argument("--fal-image-model", type=str, default=None)
    parser.add_argument(
        "--fal-vision-model",
        type=str,
        choices=["gpt-5.4", "gemini-3.1-pro", "opus-4.6"],
        default=None,
    )
    parser.add_argument("--fal-vision-temperature", type=float, default=None)
    parser.add_argument("--threshold", type=float, nargs="+", default=None)
    parser.add_argument("--dilate", type=int, default=None)
    parser.add_argument("--blur", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--model", type=str, choices=["schnell", "dev"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--redact-blur", type=int, default=None)
    parser.add_argument("--adaptive-blur", action="store_true")
    parser.add_argument("--blur-scale", type=float, default=None)
    parser.add_argument("--size-exponent", type=float, default=None)
    parser.add_argument("--scaling-factor", type=float, default=None)
    parser.add_argument("--sequential-labels", action="store_true")
    parser.add_argument("--convex-hull", action="store_true")
    parser.add_argument("--skip-empty-labels", action="store_true")
    parser.add_argument("--refinements", type=int, default=None)

    args = parser.parse_args()
    argv = sys.argv[1:]

    prior_output_folder = Path(args.input_output_folder)
    if not prior_output_folder.exists() or not prior_output_folder.is_dir():
        parser.error(f"Input output folder not found or not a directory: {prior_output_folder}")

    params = load_params_from_output(prior_output_folder)
    try:
        image_folder = _resolve_image_folder(prior_output_folder, params["image_folder"])
    except FileNotFoundError as e:
        parser.error(str(e))

    new_output_folder = Path(args.output)
    new_output_folder.mkdir(parents=True, exist_ok=True)

    def _use(from_args, from_params, option_strings):
        return from_args if _argv_contains_any(argv, option_strings) else from_params

    threshold_merged = _use(args.threshold, params.get("threshold", 0.4), ["--threshold"])
    mode = _use(args.mode, params["mode"], ["--mode", "-m"])
    if mode not in {"replace", "redact", "blackout"}:
        parser.error(f"Unsupported mode in prior params: {mode}")

    replacement_prompt = _use(
        args.replace_prompt,
        params.get("replacement_prompt"),
        ["--replace-prompt"],
    )
    if mode != "replace":
        replacement_prompt = None

    objects = _use(args.objects, params.get("objects"), ["--objects", "-o"]) or []
    privacy_concept = _use(
        args.privacy_concept,
        params.get("privacy_concept"),
        ["--privacy-concept"],
    )
    trials = _use(args.trials, params.get("trials"), ["--trials", "-t"])
    samples = _use(args.samples, params.get("samples"), ["--samples", "-s"])
    min_coverage = _use(args.min_coverage, params.get("min_coverage", 0.001), ["--min-coverage"])
    max_coverage = _use(args.max_coverage, params.get("max_coverage", 1.0), ["--max-coverage"])
    device = _use(args.device, params.get("device"), ["--device"]) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed = _use(args.seed, params.get("seed"), ["--seed"])
    segmenter = _use(args.segmenter, params.get("segmenter", "groundedsam"), ["--segmenter"])
    dilate = _use(args.dilate, params.get("dilate", 5), ["--dilate"])
    blur = _use(args.blur, params.get("blur", 8), ["--blur"])
    strength = _use(args.strength, params.get("strength", 0.85), ["--strength"])
    model = _use(args.model, params.get("model", "schnell"), ["--model"])
    num_inference_steps = _use(args.steps, params.get("num_inference_steps", 28), ["--steps"])
    redact_blur_radius = _use(
        args.redact_blur,
        params.get("redact_blur_radius", 30),
        ["--redact-blur"],
    )
    adaptive_blur = _use(
        args.adaptive_blur,
        params.get("adaptive_blur", False),
        ["--adaptive-blur"],
    )
    blur_scale = _use(args.blur_scale, params.get("blur_scale", 1.0), ["--blur-scale"])
    size_exponent = _use(
        args.size_exponent,
        params.get("size_exponent", 1.0),
        ["--size-exponent"],
    )
    scaling_factor = _use(
        args.scaling_factor,
        params.get("scaling_factor", 1.0),
        ["--scaling-factor"],
    )
    sequential_labels = _use(
        args.sequential_labels,
        params.get("sequential_labels", False),
        ["--sequential-labels"],
    )
    convex_hull = _use(args.convex_hull, params.get("convex_hull", False), ["--convex-hull"])
    skip_empty_labels = _use(
        args.skip_empty_labels,
        params.get("skip_empty_labels", False),
        ["--skip-empty-labels"],
    )
    refinements = _use(args.refinements, params.get("refinements", 0), ["--refinements"])
    fal_image_model = _use(
        args.fal_image_model,
        params.get("fal_image_model", "gpt-image-1.5"),
        ["--fal-image-model"],
    )
    fal_vision_model = _use(
        args.fal_vision_model,
        params.get("fal_vision_model", DEFAULT_FAL_VISION_MODEL),
        ["--fal-vision-model"],
    )
    fal_vision_temperature = _use(
        args.fal_vision_temperature,
        params.get("fal_vision_temperature", DEFAULT_FAL_VISION_TEMPERATURE),
        ["--fal-vision-temperature"],
    )
    embedder_model = (
        args.embedder_model
        if args.embedder_model is not None
        else params.get("embedder_model", DEFAULT_EMBEDDER_MODEL)
    )

    if trials is None:
        parser.error("Could not resolve --trials from CLI or params.json.")
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

    using_privacy_concept = segmenter == "ai-gen" and privacy_concept is not None
    if isinstance(threshold_merged, list):
        if len(threshold_merged) == 1:
            threshold = threshold_merged[0]
        else:
            if using_privacy_concept:
                print(
                    "WARNING: Multiple thresholds provided with --segmenter ai-gen and "
                    f"--privacy-concept; using first threshold {threshold_merged[0]}."
                )
                threshold = threshold_merged[0]
            elif len(threshold_merged) != len(objects):
                parser.error(
                    f"Number of thresholds ({len(threshold_merged)}) must match number of "
                    f"objects ({len(objects)})."
                )
            else:
                threshold = threshold_merged
        if segmenter == "groundedsam" and not sequential_labels and len(threshold_merged) > 1:
            threshold = threshold_merged[0]
    else:
        threshold = threshold_merged

    if args.base_concepts:
        _apply_base_concepts_on_prior_obfuscations(
            prior_output_folder=prior_output_folder,
            new_output_folder=new_output_folder,
            image_folder=image_folder,
            base_concepts=args.base_concepts,
            mode=mode,
            replacement_prompt=replacement_prompt,
            device=device,
            threshold=threshold,
            dilate=dilate,
            blur=blur,
            strength=strength,
            model=model,
            num_inference_steps=num_inference_steps,
            redact_blur_radius=redact_blur_radius,
            seed=seed,
            adaptive_blur=adaptive_blur,
            blur_scale=blur_scale,
            size_exponent=size_exponent,
            scaling_factor=scaling_factor,
            refinements=refinements,
            fal_image_model=fal_image_model,
            fal_vision_model=fal_vision_model,
        )
    else:
        print(
            "No --base-concepts: copying prior obfuscated images; "
            f"recomputing resolution with embedder: {embedder_model}"
        )
        _copy_prior_obfuscations_to_new_output(prior_output_folder, new_output_folder)

    run_resolution_analysis(
        image_folder=image_folder,
        objects=objects,
        privacy_concept=privacy_concept,
        base_concepts=None,
        mode=mode,
        trials=trials,
        output_folder=new_output_folder,
        samples=samples,
        min_coverage=min_coverage,
        max_coverage=max_coverage,
        replacement_prompt=replacement_prompt,
        seed=seed,
        device=device,
        segmenter=segmenter,
        threshold=threshold,
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
        fal_image_model=fal_image_model,
        fal_vision_model=fal_vision_model,
        fal_vision_temperature=fal_vision_temperature,
        command_line=shlex.join(sys.argv),
        generate_comparisons=not args.skip_comparisons,
        obfuscate_missing_in_continue=False,
        embedder_model=embedder_model,
    )


if __name__ == "__main__":
    main()
