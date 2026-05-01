#!/usr/bin/env python3
"""
Re-run text resolution analysis on a prior run's obfuscated texts.

Given a previous text-resolution-analysis output folder, this script:

- **Without** ``--base-concepts``: copies prior ``obfuscated/`` text files into the new output
  folder and recomputes resolution (embeddings / similarity only), e.g. to try a different
  ``--embedder``, ``--embedder-model``, or ``--embedder-quantization`` (``none``, ``half``,
  ``4bit``, ``8bit``).

- **With** ``--base-concepts``: applies GLiNER2 obfuscation for those concepts on top of the
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

from contrastive_privacy.scripts.recognize_entities import load_recognizer
from contrastive_privacy.scripts.text_anonymize import (
    DEFAULT_CONCEPT_TEMPERATURE,
    anonymize_text,
)
from contrastive_privacy.scripts.text_resolution_analysis import (
    DEFAULT_SBERT_EMBEDDER_MODEL,
    DEFAULT_QWEN_EMBEDDER_MODEL,
    get_text_files,
    load_params_from_output,
    run_text_resolution_analysis,
)


def _resolve_text_folder(prior_output_folder: Path, params_text_folder: str) -> Path:
    """
    Resolve text folder from params.json with relocation-friendly fallbacks.

    Preference order:
    1) Stored path from params.json as-is (backward compatible).
    2) Relocation candidates derived from the provided input output folder.
       This handles copied runs where absolute paths in params.json are stale.
    """
    stored_path = Path(params_text_folder)
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
    # choose directory with best filename overlap against obfuscated_*.txt outputs.
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
        candidate_texts = get_text_files(candidate)
        if not candidate_texts:
            continue
        if expected_names:
            score = sum(1 for p in candidate_texts if p.name in expected_names)
        else:
            score = len(candidate_texts)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is not None:
        print(
            "Stored text folder from params.json is unavailable; "
            f"using relocated text folder: {best_candidate}"
        )
        return best_candidate

    raise FileNotFoundError(
        "Could not resolve text folder from params.json.\n"
        f"  stored: {stored_path}\n"
        "  tried relocation candidates:\n"
        + "".join(f"    - {c}\n" for c in deduped_candidates)
        + "Provide/copy the source texts so one of these paths exists."
    )


def _apply_base_concepts_on_prior_obfuscations(
    *,
    prior_output_folder: Path,
    new_output_folder: Path,
    text_folder: Path,
    base_concepts: list[str],
    mode: str,
    placeholder: str,
    device: str,
    threshold: float,
    ner_model: Optional[str],
    sequential_labels: bool,
    propagate: bool,
) -> None:
    """Run a GLiNER2 entity pass on each prior obfuscated text (second pass, entity-only)."""
    prior_obfuscated = prior_output_folder / "obfuscated"
    new_obfuscated = new_output_folder / "obfuscated"
    new_obfuscated.mkdir(parents=True, exist_ok=True)

    text_files = get_text_files(text_folder)
    if len(text_files) < 2:
        raise ValueError(f"Need at least 2 text files in folder, found {len(text_files)}")

    print(f"Loading GLiNER2 on {device} for base-concept obfuscation...")
    recognizer = load_recognizer(model_name=ner_model, device=device)

    total = len(text_files)
    processed = 0
    reused = 0
    missing = 0

    for idx, original_path in enumerate(text_files, start=1):
        prior_obf_path = prior_obfuscated / f"obfuscated_{original_path.name}"
        if not prior_obf_path.exists():
            print(
                f"[{idx}/{total}] Missing prior obfuscated text, skipping: {prior_obf_path.name}"
            )
            missing += 1
            continue

        out_path = new_obfuscated / prior_obf_path.name
        if out_path.exists():
            print(f"[{idx}/{total}] Reusing existing reanalysis obfuscation: {out_path.name}")
            reused += 1
            continue

        primary_text = prior_obf_path.read_text(encoding="utf-8")
        print(f"[{idx}/{total}] Applying base-concept obfuscation: {prior_obf_path.name}")
        base_result = anonymize_text(
            text=primary_text,
            entity_types=base_concepts,
            mode=mode,
            placeholder=placeholder,
            threshold=threshold,
            model_name=ner_model,
            device=device,
            recognizer=recognizer,
            sequential_labels=sequential_labels,
            propagate=propagate,
            approach="entity",
            concept=None,
            concept_model=None,
        )
        out_path.write_text(base_result.anonymized_text, encoding="utf-8")
        processed += 1

    print(
        f"Prep complete: processed={processed}, reused={reused}, "
        f"missing_prior_obfuscations={missing}"
    )


def _copy_prior_obfuscations_to_new_output(
    prior_output_folder: Path,
    new_output_folder: Path,
) -> None:
    """Copy obfuscated_* files from a prior run into the new output folder."""
    prior_obfuscated = prior_output_folder / "obfuscated"
    new_obfuscated = new_output_folder / "obfuscated"
    if not prior_obfuscated.is_dir():
        raise FileNotFoundError(f"Missing obfuscated folder: {prior_obfuscated}")
    new_obfuscated.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for src in sorted(prior_obfuscated.glob("obfuscated_*")):
        if not src.is_file():
            continue
        dst = new_obfuscated / src.name
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(
        f"Copied {copied} prior obfuscated text file(s) to {new_obfuscated} "
        f"(skipped {skipped} already present)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run text-resolution-analysis on a prior run: optionally add GLiNER2 base-concept "
            "obfuscation, or only recompute resolution with a chosen text embedder."
        )
    )
    parser.add_argument(
        "input_output_folder",
        type=str,
        help=(
            "Path to an existing text-resolution-analysis output folder "
            "(must contain params.json and obfuscated/)."
        ),
    )
    parser.add_argument(
        "--base-concepts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional. If set, obfuscate these concepts with GLiNER2 on top of the prior run's "
            "obfuscated texts. If omitted, prior obfuscations are copied as-is and only "
            "resolution is recomputed (e.g. with --embedder / --embedder-model / "
            "--embedder-quantization)."
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
        help="Skip generation of comparison text files while still computing resolution.",
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
        help="Device to run on (default: reuse prior params, else auto-detect).",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=None,
        choices=["clip", "sbert", "qwen"],
        help="Text embedder type (default: prior params.json, else sbert).",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help=(
            "Embedder model name for resolution similarity (default: prior params.json, "
            f"else {DEFAULT_SBERT_EMBEDDER_MODEL} when embedder is sbert, "
            f"or {DEFAULT_QWEN_EMBEDDER_MODEL} when embedder is qwen)."
        ),
    )
    parser.add_argument(
        "--embedder-quantization",
        type=str,
        default=None,
        choices=["none", "half", "4bit", "8bit"],
        help=(
            "For sbert/qwen: half (float16, no bitsandbytes) or bitsandbytes 4-bit/8-bit (CUDA). "
            "Default when omitted: value from prior params.json, else none. "
            "4/8-bit requires bitsandbytes (included when you pip install -e this project)."
        ),
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Batch size for embedding precompute in resolution step. "
            "Default when omitted: value from prior params.json, else 8."
        ),
    )
    args = parser.parse_args()

    prior_output_folder = Path(args.input_output_folder)
    if not prior_output_folder.exists() or not prior_output_folder.is_dir():
        parser.error(f"Input output folder not found or not a directory: {prior_output_folder}")

    params = load_params_from_output(prior_output_folder)
    try:
        text_folder = _resolve_text_folder(prior_output_folder, params["text_folder"])
    except FileNotFoundError as e:
        parser.error(str(e))

    new_output_folder = Path(args.output)
    new_output_folder.mkdir(parents=True, exist_ok=True)

    approach = params.get("approach", "entity")
    mode = params["mode"]
    if mode not in {"blackout", "redact"}:
        parser.error(f"Unsupported mode in prior params: {mode}")

    placeholder = params.get("placeholder", "[REDACTED]")
    device = args.device or params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed if args.seed is not None else params.get("seed")
    embedder_type = (
        args.embedder if args.embedder is not None else params.get("embedder_type", "sbert")
    )
    embedder_model = (
        args.embedder_model
        if args.embedder_model is not None
        else params.get("embedder_model")
    )
    if embedder_model is None:
        if embedder_type == "sbert":
            embedder_model = DEFAULT_SBERT_EMBEDDER_MODEL
        elif embedder_type == "qwen":
            embedder_model = DEFAULT_QWEN_EMBEDDER_MODEL

    if args.embedder_quantization is not None:
        embedder_quantization = args.embedder_quantization
    else:
        embedder_quantization = params.get("embedder_quantization", "none")
    if args.embed_batch_size is not None:
        embed_batch_size = args.embed_batch_size
    else:
        embed_batch_size = int(params.get("embed_batch_size", 8))
    if embed_batch_size < 1:
        parser.error("--embed-batch-size must be at least 1")

    # bitsandbytes 4/8-bit requires CUDA; prior params often have device=cpu from the original run.
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

    entities = params.get("entities") or []
    if approach == "entity" and not entities:
        parser.error("Prior params missing non-empty 'entities' for approach entity.")

    if args.base_concepts:
        _apply_base_concepts_on_prior_obfuscations(
            prior_output_folder=prior_output_folder,
            new_output_folder=new_output_folder,
            text_folder=text_folder,
            base_concepts=args.base_concepts,
            mode=mode,
            placeholder=placeholder,
            device=device,
            threshold=params.get("threshold", 0.3),
            ner_model=params.get("ner_model"),
            sequential_labels=params.get("sequential_labels", False),
            propagate=params.get("propagate", True),
        )
    else:
        print(
            "No --base-concepts: copying prior obfuscated texts; "
            f"recomputing resolution with embedder: {embedder_type} / {embedder_model}"
        )
        _copy_prior_obfuscations_to_new_output(prior_output_folder, new_output_folder)

    run_text_resolution_analysis(
        text_folder=text_folder,
        entities=entities if approach == "entity" else [],
        mode=mode,
        trials=params["trials"],
        output_folder=new_output_folder,
        samples=params.get("samples"),
        min_coverage=params.get("min_coverage", 0.001),
        max_coverage=params.get("max_coverage", 1.0),
        seed=seed,
        device=device,
        ner_model=params.get("ner_model"),
        threshold=params.get("threshold", 0.3),
        embedder_type=embedder_type,
        embedder_model=embedder_model,
        embedder_quantization=embedder_quantization,
        embed_batch_size=embed_batch_size,
        placeholder=placeholder,
        sequential_labels=params.get("sequential_labels", False),
        propagate=params.get("propagate", True),
        approach=approach,
        concept=params.get("concept"),
        concept_model=params.get("concept_model"),
        concept_temperature=params.get("concept_temperature", DEFAULT_CONCEPT_TEMPERATURE),
        base_concepts=None,
        continue_from_output=True,
        obfuscate_missing_in_continue=False,
        generate_comparisons=not args.skip_comparisons,
        command_line=shlex.join(sys.argv),
    )


if __name__ == "__main__":
    main()
