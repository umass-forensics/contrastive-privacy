from __future__ import annotations

import csv
import html
import json
import os
import statistics
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import torch

from contrastive_privacy.scripts.compare_images import (
    compute_embeddings_batch as compute_image_embeddings_batch,
    load_clip_model,
    similarity_from_embeddings as image_similarity_from_embeddings,
)
from contrastive_privacy.scripts.compare_texts import (
    DEFAULT_QWEN_EMBEDDER_MODEL,
    load_text_embedder,
    similarity_from_embeddings as text_similarity_from_embeddings,
)
from contrastive_privacy.scripts.similarity_analysis import (
    DEFAULT_CLIP_MODEL,
    discover_pairs,
)


TEXT_EXTENSIONS = {".txt", ".md", ".text", ".csv", ".json"}
LOW_UTILITY_SIMILARITY = 0.3
HIGH_UTILITY_SIMILARITY = 0.7
DEFAULT_ANALYSIS_HTML = "analysis_report.html"
DEFAULT_ANALYSIS_JSON = "analysis_report.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _rehydrate_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    if not bundle:
        return bundle
    if bundle.get("output_folder") is not None:
        bundle["output_folder"] = Path(bundle["output_folder"])
    if bundle.get("analysis_json_path") is not None:
        bundle["analysis_json_path"] = Path(bundle["analysis_json_path"])
    if bundle.get("analysis_html_path") is not None:
        bundle["analysis_html_path"] = Path(bundle["analysis_html_path"])
    if "pairs" in bundle:
        bundle["pairs"] = [
            (Path(original), Path(obfuscated)) for original, obfuscated in bundle["pairs"]
        ]
    for examples in bundle.get("examples", {}).values():
        for example in examples:
            for key in ("u_path", "v_path", "u_obfuscated_path", "v_obfuscated_path"):
                if example.get(key) is not None:
                    example[key] = Path(example[key])
    return bundle


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_ready(item) for item in value]
    return value


def _read_text(path: Path) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _coerce_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": _mean(values),
        "median": _median(values),
    }


def _histogram(values: list[float], bins: int = 12) -> list[dict[str, float | int]]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        return [{"start": low, "end": high, "count": len(values)}]

    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = int((value - low) / width)
        if index >= bins:
            index = bins - 1
        counts[index] += 1

    hist: list[dict[str, float | int]] = []
    for index, count in enumerate(counts):
        start = low + index * width
        end = start + width
        hist.append({"start": start, "end": end, "count": count})
    return hist


def _resolve_image_model(params: dict[str, Any], image_model: Optional[str]) -> str:
    if image_model:
        return image_model
    return str(params.get("embedder_model") or DEFAULT_CLIP_MODEL)


def _resolve_text_settings(
    params: dict[str, Any],
    text_embedder: Optional[str],
    text_embedder_model: Optional[str],
    text_embedder_quantization: Optional[str],
) -> tuple[str, str, str]:
    embedder_type = text_embedder or str(params.get("embedder_type") or "qwen")
    embedder_model = text_embedder_model or str(
        params.get("embedder_model") or DEFAULT_QWEN_EMBEDDER_MODEL
    )
    quantization = text_embedder_quantization or str(
        params.get("embedder_quantization") or "none"
    )
    return embedder_type, embedder_model, quantization


def _compute_image_similarities(
    pairs: list[tuple[Path, Path]],
    model_name: str,
    device: Optional[str],
    batch_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    model, processor, active_device = load_clip_model(model_name, device=device)
    try:
        all_paths = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
        embeddings = compute_image_embeddings_batch(
            all_paths,
            model=model,
            processor=processor,
            device=active_device,
            batch_size=batch_size,
        )
        similarity_by_original: dict[str, float] = {}
        scores: list[float] = []
        for original_path, obfuscated_path in pairs:
            score = image_similarity_from_embeddings(
                embeddings[original_path],
                embeddings[obfuscated_path],
            )
            similarity_by_original[str(original_path)] = score
            scores.append(score)
        return similarity_by_original, {
            "enabled": True,
            "model": model_name,
            "device": active_device,
            "summary": _summarize_values(scores),
            "histogram": _histogram(scores),
        }
    finally:
        del model, processor
        if active_device == "cuda":
            torch.cuda.empty_cache()


def _compute_text_similarities(
    pairs: list[tuple[Path, Path]],
    embedder_type: str,
    embedder_model: str,
    quantization: str,
    device: Optional[str],
    batch_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    embedder = load_text_embedder(
        model_type=embedder_type,
        model_name=embedder_model,
        device=device,
        embedder_quantization=quantization,
    )
    originals = [_read_text(path) for path, _ in pairs]
    obfuscated = [_read_text(path) for _, path in pairs]
    original_embeddings = embedder.embed_batch_ordered(originals, batch_size=batch_size)
    obfuscated_embeddings = embedder.embed_batch_ordered(obfuscated, batch_size=batch_size)

    similarity_by_original: dict[str, float] = {}
    scores: list[float] = []
    for index, (original_path, _) in enumerate(pairs):
        score = text_similarity_from_embeddings(
            original_embeddings[index],
            obfuscated_embeddings[index],
        )
        similarity_by_original[str(original_path)] = score
        scores.append(score)

    return similarity_by_original, {
        "enabled": True,
        "model": embedder_model,
        "embedder": embedder_type,
        "quantization": quantization,
        "summary": _summarize_values(scores),
        "histogram": _histogram(scores),
    }


def _relative_url(target: Path, output_html: Path) -> str:
    relative = os.path.relpath(target, output_html.parent)
    return quote(relative.replace(os.sep, "/"), safe="/")


def _badge_class(resolution: float, threshold: float) -> str:
    if resolution > threshold:
        return "bad"
    if resolution > threshold - 0.05:
        return "warn"
    return "good"


def _status_bucket(resolution: float, threshold: float) -> str:
    if resolution > threshold:
        return "leak"
    if resolution > threshold - 0.05:
        return "borderline"
    return "pass"


def _utility_bucket(
    similarity: Optional[float],
    low_utility_threshold: float,
    high_utility_threshold: float,
) -> str:
    if similarity is None:
        return "unknown"
    if similarity < low_utility_threshold:
        return "low"
    if similarity >= high_utility_threshold:
        return "high"
    return "medium"


def _explanation(example: dict[str, Any], threshold: float) -> str:
    resolution = float(example["resolution"])
    margin = abs(resolution - threshold)
    similarity = example.get("reference_similarity")
    if resolution > threshold:
        base = (
            f"Leak candidate. The sanitized reference remains closer to the peer original than "
            f"to the peer sanitized version by {resolution - threshold:+.4f} beyond the threshold."
        )
    elif margin < 0.05:
        base = (
            f"Borderline pass. This pair stays on the private side, but only by {margin:.4f}, "
            f"so it is worth inspecting when tuning the sanitizer."
        )
    else:
        base = (
            f"Strong pass. The sanitized reference is closer to the sanitized peer than to the peer original "
            f"by {margin:.4f}, which is the direction the contrastive test wants."
        )
    if similarity is None:
        return base
    return base + f" Original-to-obfuscated reference similarity is {similarity:.4f}."


def _cache_metadata(
    *,
    threshold: float,
    top_n: int,
    compute_similarity: bool,
    device: Optional[str],
    batch_size: int,
    image_model: Optional[str],
    text_embedder: Optional[str],
    text_embedder_model: Optional[str],
    text_embedder_quantization: Optional[str],
    image_folder: Optional[str | Path],
    text_folder: Optional[str | Path],
    low_utility_threshold: float,
    high_utility_threshold: float,
) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "top_n": top_n,
        "compute_similarity": compute_similarity,
        "device": device,
        "batch_size": batch_size,
        "image_model": str(image_model) if image_model is not None else None,
        "text_embedder": text_embedder,
        "text_embedder_model": text_embedder_model,
        "text_embedder_quantization": text_embedder_quantization,
        "image_folder": str(Path(image_folder).resolve()) if image_folder else None,
        "text_folder": str(Path(text_folder).resolve()) if text_folder else None,
        "low_utility_threshold": low_utility_threshold,
        "high_utility_threshold": high_utility_threshold,
    }


def _trim_text(text: str, limit: int = 900) -> str:
    compact = text.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _load_results_rows(results_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(results_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "resolution": float(row["resolution"]),
                    "u": row.get("u", ""),
                    "v": row.get("v", ""),
                    "comparison_filename": row.get("comparison_filename")
                    or row.get("compound")
                    or "",
                    "sim_xu_v": _coerce_float(row.get("sim_xu_v")),
                    "sim_xu_xv": _coerce_float(row.get("sim_xu_xv")),
                }
            )
    return rows


def _unordered_pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((str(row.get("u", "")), str(row.get("v", "")))))


def _unique_unordered_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first directed row encountered for each unordered image/text pair."""
    seen: set[tuple[str, str]] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        key = _unordered_pair_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def load_run_report(
    output_folder: str | Path,
    *,
    threshold: float = 0.0,
    top_n: int = 6,
    compute_similarity: bool = True,
    device: Optional[str] = None,
    batch_size: int = 8,
    image_model: Optional[str] = None,
    text_embedder: Optional[str] = None,
    text_embedder_model: Optional[str] = None,
    text_embedder_quantization: Optional[str] = None,
    image_folder: Optional[str | Path] = None,
    text_folder: Optional[str | Path] = None,
    low_utility_threshold: float = LOW_UTILITY_SIMILARITY,
    high_utility_threshold: float = HIGH_UTILITY_SIMILARITY,
) -> dict[str, Any]:
    output_path = Path(output_folder).resolve()
    params = _read_json(output_path / "params.json")
    report_text = _read_text(output_path / "report.txt") if (output_path / "report.txt").is_file() else ""
    rows = _load_results_rows(output_path / "results.csv")
    content_type, pairs = discover_pairs(
        output_path,
        image_folder_override=Path(image_folder).resolve() if image_folder else None,
        text_folder_override=Path(text_folder).resolve() if text_folder else None,
    )

    original_to_obfuscated = {str(original): obfuscated for original, obfuscated in pairs}
    resolutions = [float(row["resolution"]) for row in rows]
    unique_references = sorted({row["u"] for row in rows if row.get("u")})
    violating_rows = [row for row in rows if float(row["resolution"]) > threshold]
    violating_refs = {row["u"] for row in violating_rows if row.get("u")}

    similarity_summary: dict[str, Any] = {"enabled": False}
    similarity_by_original: dict[str, float] = {}
    if compute_similarity and pairs:
        if content_type == "image":
            similarity_by_original, similarity_summary = _compute_image_similarities(
                pairs,
                _resolve_image_model(params, image_model),
                device,
                batch_size,
            )
        else:
            embedder_type, embedder_model, quantization = _resolve_text_settings(
                params,
                text_embedder,
                text_embedder_model,
                text_embedder_quantization,
            )
            similarity_by_original, similarity_summary = _compute_text_similarities(
                pairs,
                embedder_type,
                embedder_model,
                quantization,
                device,
                batch_size,
            )

    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        example = dict(row)
        example["u_path"] = Path(row["u"]).resolve()
        example["v_path"] = Path(row["v"]).resolve()
        example["u_obfuscated_path"] = original_to_obfuscated.get(row["u"])
        example["v_obfuscated_path"] = original_to_obfuscated.get(row["v"])
        example["reference_similarity"] = similarity_by_original.get(row["u"])
        example["peer_similarity"] = similarity_by_original.get(row["v"])
        example["status_class"] = _badge_class(float(row["resolution"]), threshold)
        example["status_bucket"] = _status_bucket(float(row["resolution"]), threshold)
        example["low_utility"] = (
            example["reference_similarity"] is not None
            and float(example["reference_similarity"]) < low_utility_threshold
        )
        example["utility_bucket"] = _utility_bucket(
            example["reference_similarity"],
            low_utility_threshold,
            high_utility_threshold,
        )
        example["reference_name"] = example["u_path"].name
        example["peer_name"] = example["v_path"].name
        example["search_text"] = (
            f"{example['reference_name']} {example['peer_name']} {example['status_bucket']} {example['utility_bucket']}"
        ).lower()
        example["explanation"] = _explanation(example, threshold)
        return example

    sorted_rows = sorted(rows, key=lambda row: float(row["resolution"]))
    all_examples = [enrich(row) for row in sorted_rows]
    worst_rows = _unique_unordered_pairs(
        sorted(rows, key=lambda row: float(row["resolution"]), reverse=True)
    )
    borderline_rows = _unique_unordered_pairs(
        sorted(rows, key=lambda row: abs(float(row["resolution"]) - threshold))
    )
    strongest_rows = _unique_unordered_pairs(
        [row for row in sorted_rows if float(row["resolution"]) < threshold]
    )
    leak_rows = _unique_unordered_pairs(
        sorted(
            [row for row in rows if float(row["resolution"]) > threshold],
            key=lambda row: float(row["resolution"]),
            reverse=True,
        )
    )
    low_utility_rows = _unique_unordered_pairs(
        sorted(
            [row for row in rows if similarity_by_original.get(row["u"], 1.0) < low_utility_threshold],
            key=lambda row: similarity_by_original.get(row["u"], 1.0),
        )
    )

    worst_examples = [enrich(row) for row in worst_rows[:top_n]]
    borderline_examples = [enrich(row) for row in borderline_rows[:top_n]]
    strongest_examples = [enrich(row) for row in strongest_rows[:top_n]]
    leak_examples = [enrich(row) for row in leak_rows[:top_n]]
    low_utility_examples = [enrich(row) for row in low_utility_rows[:top_n]]

    resolution_summary = {
        **_summarize_values(resolutions),
        "negative": sum(1 for value in resolutions if value < threshold),
        "zero": sum(1 for value in resolutions if value == threshold),
        "positive": sum(1 for value in resolutions if value > threshold),
        "reference_count": len(unique_references),
        "violation_rows": len(violating_rows),
        "violation_refs": len(violating_refs),
        "violation_rate": (len(violating_refs) / len(unique_references) * 100.0)
        if unique_references
        else 0.0,
        "leak_pairs_rate": (len(violating_rows) / len(rows) * 100.0) if rows else 0.0,
        "threshold": threshold,
        "histogram": _histogram(resolutions),
    }

    filter_summary = {
        "leaks": len([row for row in rows if float(row["resolution"]) > threshold]),
        "borderline": len(
            [row for row in rows if threshold >= float(row["resolution"]) > threshold - 0.05]
        ),
        "passes": len([row for row in rows if float(row["resolution"]) <= threshold - 0.05]),
        "low_utility": len(
            [row for row in rows if similarity_by_original.get(row["u"], 1.0) < low_utility_threshold]
        ),
        "high_utility": len(
            [row for row in rows if similarity_by_original.get(row["u"], 0.0) >= high_utility_threshold]
        ),
    }

    return {
        "content_type": content_type,
        "output_folder": output_path,
        "params": params,
        "report_text": report_text,
        "results": sorted_rows,
        "pairs": pairs,
        "reference_count": len(unique_references),
        "resolution_summary": resolution_summary,
        "similarity_summary": similarity_summary,
        "filter_summary": filter_summary,
        "utility_thresholds": {
            "low": low_utility_threshold,
            "high": high_utility_threshold,
        },
        "examples": {
            "all": all_examples,
            "leaks": leak_examples,
            "worst": worst_examples,
            "borderline": borderline_examples,
            "strongest": strongest_examples,
            "low_utility": low_utility_examples,
        },
    }


def load_cached_run_report(
    json_path: str | Path,
    *,
    expected_cache_metadata: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    json_path = Path(json_path).resolve()
    if not json_path.is_file():
        return None
    bundle = _read_json(json_path)
    if not bundle:
        return None
    if expected_cache_metadata is not None and bundle.get("cache_metadata") != expected_cache_metadata:
        return None
    return _rehydrate_bundle(bundle)


def write_json_report(bundle: dict[str, Any], output_json: str | Path) -> Path:
    output_path = Path(output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_json_ready(bundle), indent=2), encoding="utf-8")
    return output_path


def prepare_run_report(
    output_folder: str | Path,
    *,
    threshold: float = 0.0,
    top_n: int = 6,
    compute_similarity: bool = True,
    device: Optional[str] = None,
    batch_size: int = 8,
    image_model: Optional[str] = None,
    text_embedder: Optional[str] = None,
    text_embedder_model: Optional[str] = None,
    text_embedder_quantization: Optional[str] = None,
    image_folder: Optional[str | Path] = None,
    text_folder: Optional[str | Path] = None,
    low_utility_threshold: float = LOW_UTILITY_SIMILARITY,
    high_utility_threshold: float = HIGH_UTILITY_SIMILARITY,
    json_output: Optional[str | Path] = None,
    refresh: bool = False,
) -> tuple[dict[str, Any], bool, Path]:
    output_path = Path(output_folder).resolve()
    output_json = Path(json_output).resolve() if json_output else output_path / DEFAULT_ANALYSIS_JSON
    cache_metadata = _cache_metadata(
        threshold=threshold,
        top_n=top_n,
        compute_similarity=compute_similarity,
        device=device,
        batch_size=batch_size,
        image_model=image_model,
        text_embedder=text_embedder,
        text_embedder_model=text_embedder_model,
        text_embedder_quantization=text_embedder_quantization,
        image_folder=image_folder,
        text_folder=text_folder,
        low_utility_threshold=low_utility_threshold,
        high_utility_threshold=high_utility_threshold,
    )
    if not refresh:
        cached = load_cached_run_report(output_json, expected_cache_metadata=cache_metadata)
        if cached is not None:
            cached["analysis_json_path"] = output_json
            return cached, True, output_json

    bundle = load_run_report(
        output_path,
        threshold=threshold,
        top_n=top_n,
        compute_similarity=compute_similarity,
        device=device,
        batch_size=batch_size,
        image_model=image_model,
        text_embedder=text_embedder,
        text_embedder_model=text_embedder_model,
        text_embedder_quantization=text_embedder_quantization,
        image_folder=image_folder,
        text_folder=text_folder,
        low_utility_threshold=low_utility_threshold,
        high_utility_threshold=high_utility_threshold,
    )
    bundle["cache_metadata"] = cache_metadata
    bundle["analysis_json_path"] = output_json
    write_json_report(bundle, output_json)
    return bundle, False, output_json


def _overview_text(bundle: dict[str, Any]) -> str:
    summary = bundle["resolution_summary"]
    similarity_summary = bundle["similarity_summary"]
    if summary["violation_refs"] == 0:
        text = (
            f"All sampled references passed the contrastive test at threshold {summary['threshold']:.3f}. "
            f"The mean resolution was {summary['mean']:+.4f}, which is comfortably on the private side."
        )
    else:
        text = (
            f"{summary['violation_refs']} of {summary['reference_count']} references produced at least one leak candidate. "
            f"The highest observed resolution was {summary['max']:+.4f}."
        )
    if similarity_summary.get("enabled"):
        sim_mean = similarity_summary["summary"]["mean"]
        if sim_mean < 0.3:
            text += " Original-to-obfuscated similarity is low, so privacy may be coming with heavy distortion."
        elif sim_mean > 0.7:
            text += " Original-to-obfuscated similarity stays high, which suggests the sanitizer preserved a lot of structure while still changing the contrastive signal."
        else:
            text += " Original-to-obfuscated similarity lands in the middle range, which is a reasonable utility/privacy tradeoff to inspect case by case."
    return text


def _format_path_label(path: Path) -> str:
    return path.name


def _render_histogram(histogram: list[dict[str, float | int]], *, value_kind: str) -> str:
    if not histogram:
        return "<p class=\"muted\">No values available.</p>"
    max_count = max(int(bin_info["count"]) for bin_info in histogram) or 1
    rows: list[str] = []
    for bin_info in histogram:
        count = int(bin_info["count"])
        width = max(4, int(count / max_count * 100)) if count else 0
        label = f"{bin_info['start']:+.3f} to {bin_info['end']:+.3f}" if value_kind == "resolution" else f"{bin_info['start']:.3f} to {bin_info['end']:.3f}"
        rows.append(
            "<div class=\"hist-row\">"
            f"<div class=\"hist-label\">{html.escape(label)}</div>"
            f"<div class=\"hist-bar-wrap\"><div class=\"hist-bar\" style=\"width:{width}%\"></div></div>"
            f"<div class=\"hist-count\">{count}</div>"
            "</div>"
        )
    return "".join(rows)


def _render_stats_grid(title: str, stats: dict[str, Any], *, similarity: bool = False) -> str:
    if not stats:
        return ""
    cards = []
    if similarity:
        items = [
            ("Pairs", stats["count"]),
            ("Mean similarity", f"{stats['mean']:.4f}"),
            ("Median similarity", f"{stats['median']:.4f}"),
            ("Range", f"{stats['min']:.4f} to {stats['max']:.4f}"),
        ]
    else:
        items = [
            ("Comparisons", stats["count"]),
            ("Reference items", stats["reference_count"]),
            ("Leak candidates", stats["violation_rows"]),
            ("Leak rate", f"{stats['violation_rate']:.1f}%"),
            ("Leaky pairs", f"{stats['leak_pairs_rate']:.1f}%"),
            ("Mean resolution", f"{stats['mean']:+.4f}"),
            ("Worst resolution", f"{stats['max']:+.4f}"),
        ]
    for label, value in items:
        cards.append(
            "<div class=\"metric-card\">"
            f"<div class=\"metric-label\">{html.escape(str(label))}</div>"
            f"<div class=\"metric-value\">{html.escape(str(value))}</div>"
            "</div>"
        )
    return f"<section><h2>{html.escape(title)}</h2><div class=\"metrics\">{''.join(cards)}</div></section>"


def _format_similarity(value: Optional[float]) -> str:
    if value is None:
        return "(n/a)"
    return f"{value:.4f}"


def _render_examples_table(bundle: dict[str, Any]) -> str:
    rows = []
    for index, example in enumerate(bundle["examples"].get("all", []), start=1):
        rows.append(
            "<tr class=\"example-row\" "
            f"data-status=\"{example['status_bucket']}\" "
            f"data-low-utility=\"{'true' if example['low_utility'] else 'false'}\" "
            f"data-search=\"{html.escape(example['search_text'])}\" "
            f"data-resolution=\"{example['resolution']:.8f}\" "
            f"data-similarity=\"{(example['reference_similarity'] if example['reference_similarity'] is not None else -1.0):.8f}\" "
            f"data-reference=\"{html.escape(example['reference_name'].lower())}\" "
            f"data-peer=\"{html.escape(example['peer_name'].lower())}\" "
            f"data-utility=\"{html.escape(example['utility_bucket'])}\">"
            f"<td>{index}</td>"
            f"<td>{html.escape(example['reference_name'])}</td>"
            f"<td>{html.escape(example['peer_name'])}</td>"
            f"<td>{example['resolution']:+.4f}</td>"
            f"<td>{_format_similarity(example['reference_similarity'])}</td>"
            f"<td>{html.escape(example['status_bucket'])}</td>"
            f"<td>{html.escape(example['utility_bucket'])}</td>"
            "</tr>"
        )
    if not rows:
        return "<p class=\"muted\">No pairwise rows available.</p>"
    return (
        "<div class=\"table-tools\">"
        "<input id=\"example-search\" class=\"search-input\" type=\"search\" placeholder=\"Search reference, peer, status, or utility band\">"
        "<div id=\"table-count\" class=\"muted\"></div>"
        "</div>"
        "<div class=\"table-wrap\"><table class=\"examples-table\">"
        "<thead><tr>"
        "<th>#</th>"
        "<th><button class=\"sort-btn\" data-sort=\"reference\">Reference</button></th>"
        "<th><button class=\"sort-btn\" data-sort=\"peer\">Peer</button></th>"
        "<th><button class=\"sort-btn\" data-sort=\"resolution\">Resolution</button></th>"
        "<th><button class=\"sort-btn\" data-sort=\"similarity\">Ref similarity</button></th>"
        "<th><button class=\"sort-btn\" data-sort=\"status\">Status</button></th>"
        "<th><button class=\"sort-btn\" data-sort=\"utility\">Utility</button></th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_image_example(example: dict[str, Any], output_html: Path, threshold: float) -> str:
    paths = [
        (f"X_c(x): {_format_path_label(example['u_path'])}", example["u_obfuscated_path"]),
        (f"y: {_format_path_label(example['v_path'])}", example["v_path"]),
        (f"X_c(x): {_format_path_label(example['u_path'])}", example["u_obfuscated_path"]),
        (f"X_c(y): {_format_path_label(example['v_path'])}", example["v_obfuscated_path"]),
    ]
    tiles: list[str] = []
    for label, path in paths:
        body = "<div class=\"missing\">Missing artifact</div>"
        if isinstance(path, Path) and path.is_file():
            body = (
                f"<img src=\"{_relative_url(path, output_html)}\" alt=\"{html.escape(label)}\">"
            )
        tiles.append(
            "<figure class=\"image-tile\">"
            f"{body}<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )
    return (
        f"<article class=\"example-card {example['status_class']}\" data-status=\"{example['status_bucket']}\" data-low-utility=\"{'true' if example['low_utility'] else 'false'}\">"
        f"<div class=\"example-head\"><div><h3>{html.escape(_format_path_label(example['u_path']))}</h3>"
        f"<p class=\"muted\">Compared against {html.escape(_format_path_label(example['v_path']))}</p></div>"
        f"<div class=\"score-pill {example['status_class']}\">resolution {example['resolution']:+.4f}</div></div>"
        f"<p>{html.escape(example['explanation'])}</p>"
        "<div class=\"image-grid\">" + "".join(tiles) + "</div>"
        "</article>"
    )


def _render_text_block(label: str, path: Path, obfuscated: bool) -> str:
    if not path.is_file():
        body = "Missing artifact"
    else:
        body = _trim_text(_read_text(path))
    css_class = "text-tile obfuscated" if obfuscated else "text-tile"
    return (
        f"<div class=\"{css_class}\">"
        f"<div class=\"tile-label\">{html.escape(label)}</div>"
        f"<pre>{html.escape(body)}</pre>"
        "</div>"
    )


def _render_text_example(example: dict[str, Any], output_html: Path, threshold: float) -> str:
    del output_html, threshold
    return (
        f"<article class=\"example-card {example['status_class']}\" data-status=\"{example['status_bucket']}\" data-low-utility=\"{'true' if example['low_utility'] else 'false'}\">"
        f"<div class=\"example-head\"><div><h3>{html.escape(_format_path_label(example['u_path']))}</h3>"
        f"<p class=\"muted\">Compared against {html.escape(_format_path_label(example['v_path']))}</p></div>"
        f"<div class=\"score-pill {example['status_class']}\">resolution {example['resolution']:+.4f}</div></div>"
        f"<p>{html.escape(example['explanation'])}</p>"
        "<div class=\"text-grid\">"
        + _render_text_block("Reference original", example["u_path"], False)
        + _render_text_block("Reference obfuscated", example["u_obfuscated_path"], True)
        + _render_text_block("Peer original", example["v_path"], False)
        + _render_text_block("Peer obfuscated", example["v_obfuscated_path"], True)
        + "</div></article>"
    )


def render_html_report(
    bundle: dict[str, Any],
    output_html: str | Path,
    *,
    title: Optional[str] = None,
) -> str:
    output_html = Path(output_html).resolve()
    output_folder = Path(bundle["output_folder"])
    params = bundle["params"]
    summary = bundle["resolution_summary"]
    similarity_summary = bundle["similarity_summary"]
    report_title = title or "Contrastive Privacy Analysis"
    run_name = output_folder.name
    threshold = float(summary["threshold"])

    example_renderer = _render_image_example if bundle["content_type"] == "image" else _render_text_example
    example_sections = []
    labels = {
        "leaks": "Leak candidates",
        "worst": "Worst examples",
        "borderline": "Closest to the threshold",
        "strongest": "Strongest passes",
        "low_utility": "Low-utility examples",
    }
    for key in ("leaks", "worst", "borderline", "strongest", "low_utility"):
        cards = [example_renderer(example, output_html, threshold) for example in bundle["examples"][key]]
        if cards:
            example_sections.append(
                f"<section class=\"example-section\" data-section=\"{key}\"><h2>{labels[key]}</h2>{''.join(cards)}</section>"
            )

    config_rows = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        config_rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        )

    similarity_section = ""
    if similarity_summary.get("enabled"):
        similarity_section = (
            _render_stats_grid("Similarity summary", similarity_summary["summary"], similarity=True)
            + "<section><h2>Similarity histogram</h2><div class=\"histogram\">"
            + _render_histogram(similarity_summary["histogram"], value_kind="similarity")
            + "</div></section>"
        )

    filter_summary = bundle.get("filter_summary", {})
    utility_thresholds = bundle.get("utility_thresholds", {"low": LOW_UTILITY_SIMILARITY, "high": HIGH_UTILITY_SIMILARITY})
    filter_controls = (
        "<section class=\"panel\"><h2>Filters</h2><div class=\"filters\">"
        f"<button class=\"filter-btn active\" data-filter=\"all\">All ({summary['count']})</button>"
        f"<button class=\"filter-btn\" data-filter=\"leak\">Leaks ({filter_summary.get('leaks', 0)})</button>"
        f"<button class=\"filter-btn\" data-filter=\"borderline\">Borderline ({filter_summary.get('borderline', 0)})</button>"
        f"<button class=\"filter-btn\" data-filter=\"pass\">Strong passes ({filter_summary.get('passes', 0)})</button>"
        f"<button class=\"filter-btn\" data-filter=\"low-utility\">Low utility ({filter_summary.get('low_utility', 0)})</button>"
        "</div>"
        f"<p class=\"muted\">Filters apply to both the example cards and the table below. Low utility is similarity below {utility_thresholds['low']:.2f}; high utility starts at {utility_thresholds['high']:.2f}.</p></section>"
    )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(report_title)}</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500&display=swap\" rel=\"stylesheet\">
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255, 253, 249, 0.88);
      --panel-strong: #fffdf9;
      --ink: #1f1d1a;
      --muted: #665f57;
      --line: #d9cfbf;
      --good: #2f6b49;
      --warn: #b3731d;
      --bad: #ad3434;
      --good-soft: #e1f0e4;
      --warn-soft: #f7ead2;
      --bad-soft: #f7dfdc;
      --shadow: 0 20px 60px rgba(73, 56, 35, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Instrument Sans', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(182, 117, 58, 0.12), transparent 32%),
        radial-gradient(circle at top right, rgba(47, 107, 73, 0.1), transparent 28%),
        var(--bg);
    }}
    .page {{ max-width: 1220px; margin: 0 auto; padding: 40px 24px 72px; }}
    .hero {{
      background: var(--panel);
      border: 1px solid rgba(217, 207, 191, 0.9);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px 30px;
      backdrop-filter: blur(12px);
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 12px;
    }}
    h1, h2, h3 {{ font-family: 'Fraunces', serif; line-height: 1.1; margin: 0; }}
    h1 {{ font-size: clamp(2.2rem, 4vw, 4rem); max-width: 12ch; margin-bottom: 10px; }}
    h2 {{ font-size: 1.65rem; margin-bottom: 16px; }}
    h3 {{ font-size: 1.2rem; margin-bottom: 6px; }}
    p {{ line-height: 1.6; margin: 0 0 14px; }}
    .muted {{ color: var(--muted); }}
    .run-name {{
      color: var(--muted);
      font-family: 'IBM Plex Mono', monospace;
      font-size: clamp(0.92rem, 1.7vw, 1.15rem);
      line-height: 1.35;
      margin-bottom: 16px;
      overflow-wrap: anywhere;
    }}
    .hero-grid {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 28px; align-items: start; }}
    .hero-copy {{ max-width: 60ch; }}
    .hero-meta {{ display: grid; gap: 14px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(217, 207, 191, 0.9);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-top: 24px;
    }}
        .filters {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .filter-btn {{
            border: 1px solid var(--line);
            background: var(--panel-strong);
            border-radius: 999px;
            padding: 10px 14px;
            font: inherit;
            cursor: pointer;
            color: var(--ink);
        }}
        .filter-btn.active {{ background: #1f1d1a; color: #fffdf9; border-color: #1f1d1a; }}
        .table-tools {{ display: flex; gap: 14px; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }}
        .search-input {{
            min-width: min(420px, 100%);
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--panel-strong);
            padding: 12px 14px;
            font: inherit;
        }}
        .table-wrap {{ overflow: auto; max-height: min(640px, 70vh); border: 1px solid var(--line); border-radius: 18px; background: var(--panel-strong); }}
        .examples-table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
        .examples-table th, .examples-table td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; }}
        .examples-table th {{ font-size: 0.84rem; color: var(--muted); background: #f8f4ee; position: sticky; top: 0; }}
        .sort-btn {{ background: none; border: 0; padding: 0; font: inherit; color: inherit; cursor: pointer; }}
        .examples-table tbody tr:last-child td {{ border-bottom: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; }}
    .metric-card {{ background: var(--panel-strong); border: 1px solid var(--line); border-radius: 18px; padding: 16px; }}
    .metric-label {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 8px; }}
    .metric-value {{ font-size: 1.3rem; font-weight: 700; }}
    .split {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 24px; }}
    .histogram {{ display: grid; gap: 8px; }}
    .hist-row {{ display: grid; grid-template-columns: 130px 1fr 36px; gap: 12px; align-items: center; }}
    .hist-label, .hist-count {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }}
    .hist-bar-wrap {{ background: rgba(126, 101, 72, 0.12); border-radius: 999px; overflow: hidden; min-height: 12px; }}
    .hist-bar {{ background: linear-gradient(90deg, #cf8d46, #8d4e2d); height: 12px; border-radius: 999px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 0; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ width: 220px; color: var(--muted); font-weight: 600; }}
    .example-card {{ background: var(--panel-strong); border: 1px solid var(--line); border-radius: 24px; padding: 22px; margin-bottom: 18px; }}
    .example-card.good {{ border-color: rgba(47, 107, 73, 0.35); }}
    .example-card.warn {{ border-color: rgba(179, 115, 29, 0.35); }}
    .example-card.bad {{ border-color: rgba(173, 52, 52, 0.35); }}
    .example-head {{ display: flex; gap: 18px; justify-content: space-between; align-items: start; margin-bottom: 12px; }}
    .score-pill {{ border-radius: 999px; padding: 8px 12px; font-size: 0.82rem; font-weight: 700; white-space: nowrap; }}
    .score-pill.good {{ background: var(--good-soft); color: var(--good); }}
    .score-pill.warn {{ background: var(--warn-soft); color: var(--warn); }}
    .score-pill.bad {{ background: var(--bad-soft); color: var(--bad); }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .image-tile, .text-tile {{ margin: 0; background: #f8f4ee; border-radius: 18px; border: 1px solid var(--line); overflow: hidden; }}
    .image-tile img {{ width: 100%; height: 260px; object-fit: contain; display: block; background: #ebe2d5; }}
    figcaption, .tile-label {{ padding: 10px 12px; font-size: 0.84rem; color: var(--muted); border-top: 1px solid var(--line); }}
    .missing {{ min-height: 260px; display: grid; place-items: center; color: var(--muted); }}
    .text-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .text-tile pre {{ margin: 0; padding: 14px; white-space: pre-wrap; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; line-height: 1.55; max-height: 320px; overflow: auto; }}
    .text-tile.obfuscated {{ background: #f1ece4; }}
    .report-box {{ background: #1f1d1a; color: #f4efe7; border-radius: 18px; padding: 18px; overflow: auto; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; line-height: 1.55; }}
    @media (max-width: 900px) {{
      .hero-grid, .split, .text-grid, .image-grid {{ grid-template-columns: 1fr; }}
      .example-head {{ flex-direction: column; align-items: start; }}
      th {{ width: 140px; }}
    }}
  </style>
</head>
<body>
  <main class=\"page\">
    <section class=\"hero\">
      <div class=\"hero-grid\">
        <div class=\"hero-copy\">
          <div class=\"eyebrow\">Contrastive privacy run review</div>
          <h1>{html.escape(report_title)}</h1>
          <div class=\"run-name\">{html.escape(run_name)}</div>
          <p>{html.escape(_overview_text(bundle))}</p>
          <p class=\"muted\">Output folder: {html.escape(str(output_folder))}</p>
        </div>
        <div class=\"hero-meta\">
          <div class=\"metric-card\">
            <div class=\"metric-label\">Content type</div>
            <div class=\"metric-value\">{html.escape(bundle['content_type'])}</div>
          </div>
          <div class=\"metric-card\">
            <div class=\"metric-label\">Threshold</div>
            <div class=\"metric-value\">{summary['threshold']:.3f}</div>
          </div>
          <div class=\"metric-card\">
            <div class=\"metric-label\">Run seed</div>
            <div class=\"metric-value\">{html.escape(str(params.get('seed', '(none)')))}</div>
          </div>
        </div>
      </div>
    </section>

    {_render_stats_grid('Resolution summary', summary)}
        {filter_controls}

    <section class=\"panel split\">
      <div>
        <h2>How to read this page</h2>
        <p>Positive resolution means a leak candidate: the sanitized reference still stays closer to a peer original than to the peer sanitized version.</p>
        <p>Negative resolution means the pair is moving in the right direction. The more negative it gets, the more separation the sanitizer created for that pair.</p>
        <p>The example cards below show the reference item, its obfuscated version, the comparison peer, and the peer obfuscated version, with a short explanation for why the pair matters.</p>
      </div>
      <div>
        <h2>Resolution histogram</h2>
        <div class=\"histogram\">{_render_histogram(summary['histogram'], value_kind='resolution')}</div>
      </div>
    </section>

    {similarity_section}

    <section class=\"panel\">
      <h2>Configuration</h2>
      <table>{''.join(config_rows)}</table>
    </section>

        <section class=\"panel\">
            <h2>Reusable analysis files</h2>
            <p>This page is backed by a structured JSON bundle saved alongside the run. Reuse it for downstream plotting, dashboards, or notebooks without recomputing similarity.</p>
            <p class=\"muted\">JSON file: {html.escape(str(bundle.get('analysis_json_path') or (output_folder / DEFAULT_ANALYSIS_JSON)))}</p>
        </section>

        <section class=\"panel\">
            <h2>All Pairwise Rows</h2>
            <p class=\"muted\">Use the search box for fast inspection and click the column labels to sort.</p>
            {_render_examples_table(bundle)}
        </section>

    {''.join(example_sections)}

    <section class=\"panel\">
      <h2>Raw report</h2>
      <div class=\"report-box\">{html.escape(bundle['report_text'])}</div>
    </section>
  </main>
    <script>
        const buttons = Array.from(document.querySelectorAll('.filter-btn'));
        const cards = Array.from(document.querySelectorAll('.example-card'));
        const sections = Array.from(document.querySelectorAll('.example-section'));
        const searchInput = document.getElementById('example-search');
        const tableCount = document.getElementById('table-count');
        const tableBody = document.querySelector('.examples-table tbody');
        const tableRows = tableBody ? Array.from(tableBody.querySelectorAll('.example-row')) : [];
        const sortButtons = Array.from(document.querySelectorAll('.sort-btn'));
        let activeFilter = 'all';
        let searchTerm = '';
        let currentSort = {{ key: 'resolution', direction: 'desc' }};

        function rowMatchesFilter(element) {{
            const status = element.dataset.status;
            const lowUtility = element.dataset.lowUtility === 'true';
            return activeFilter === 'all'
                || (activeFilter === 'low-utility' && lowUtility)
                || (activeFilter !== 'low-utility' && status === activeFilter);
        }}

        function rowMatchesSearch(element) {{
            return !searchTerm || (element.dataset.search || '').includes(searchTerm);
        }}

        function updateTableCount() {{
            if (!tableCount) return;
            const visible = tableRows.filter((row) => row.style.display !== 'none').length;
            tableCount.textContent = `${{visible}} visible row${{visible === 1 ? '' : 's'}}`;
        }}

        function applyFilter(filter) {{
            activeFilter = filter;
            buttons.forEach((button) => button.classList.toggle('active', button.dataset.filter === filter));
            cards.forEach((card) => {{
                const show = rowMatchesFilter(card) && rowMatchesSearch(card);
                card.style.display = show ? '' : 'none';
            }});
            tableRows.forEach((row) => {{
                const show = rowMatchesFilter(row) && rowMatchesSearch(row);
                row.style.display = show ? '' : 'none';
            }});
            sections.forEach((section) => {{
                const visibleCards = section.querySelectorAll('.example-card:not([style*="display: none"])');
                section.style.display = visibleCards.length ? '' : 'none';
            }});
            updateTableCount();
        }}

        function sortRows(key) {{
            if (!tableBody) return;
            if (currentSort.key === key) {{
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            }} else {{
                currentSort = {{ key, direction: key === 'resolution' || key === 'similarity' ? 'desc' : 'asc' }};
            }}
            const direction = currentSort.direction === 'asc' ? 1 : -1;
            const sorted = [...tableRows].sort((left, right) => {{
                const a = left.dataset[key] || '';
                const b = right.dataset[key] || '';
                const aNum = Number(a);
                const bNum = Number(b);
                if (!Number.isNaN(aNum) && !Number.isNaN(bNum) && a !== '' && b !== '') {{
                    return (aNum - bNum) * direction;
                }}
                return a.localeCompare(b) * direction;
            }});
            sorted.forEach((row) => tableBody.appendChild(row));
        }}

        buttons.forEach((button) => {{
            button.addEventListener('click', () => applyFilter(button.dataset.filter));
        }});
        if (searchInput) {{
            searchInput.addEventListener('input', (event) => {{
                searchTerm = event.target.value.trim().toLowerCase();
                applyFilter(activeFilter);
            }});
        }}
        sortButtons.forEach((button) => {{
            button.addEventListener('click', () => sortRows(button.dataset.sort));
        }});
        sortRows('resolution');
        applyFilter('all');
    </script>
</body>
</html>
"""


def write_html_report(
    bundle: dict[str, Any],
    output_html: str | Path,
    *,
    title: Optional[str] = None,
) -> Path:
    output_path = Path(output_html).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_html_report(bundle, output_path, title=title)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def generate_analysis_artifacts(
    output_folder: str | Path,
    *,
    title: Optional[str] = None,
    threshold: float = 0.0,
    top_n: int = 6,
    compute_similarity: bool = True,
    device: Optional[str] = None,
    batch_size: int = 8,
    image_model: Optional[str] = None,
    text_embedder: Optional[str] = None,
    text_embedder_model: Optional[str] = None,
    text_embedder_quantization: Optional[str] = None,
    image_folder: Optional[str | Path] = None,
    text_folder: Optional[str | Path] = None,
    low_utility_threshold: float = LOW_UTILITY_SIMILARITY,
    high_utility_threshold: float = HIGH_UTILITY_SIMILARITY,
    html_output: Optional[str | Path] = None,
    json_output: Optional[str | Path] = None,
    refresh: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_folder).resolve()
    html_path = Path(html_output).resolve() if html_output else output_path / DEFAULT_ANALYSIS_HTML
    bundle, used_cache, json_path = prepare_run_report(
        output_path,
        threshold=threshold,
        top_n=top_n,
        compute_similarity=compute_similarity,
        device=device,
        batch_size=batch_size,
        image_model=image_model,
        text_embedder=text_embedder,
        text_embedder_model=text_embedder_model,
        text_embedder_quantization=text_embedder_quantization,
        image_folder=image_folder,
        text_folder=text_folder,
        low_utility_threshold=low_utility_threshold,
        high_utility_threshold=high_utility_threshold,
        json_output=json_output,
        refresh=refresh,
    )
    bundle["analysis_html_path"] = html_path
    html_written = write_html_report(bundle, html_path, title=title)
    return {
        "bundle": bundle,
        "html_path": html_written,
        "json_path": json_path,
        "used_cache": used_cache,
    }