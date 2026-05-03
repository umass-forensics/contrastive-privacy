#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

from contrastive_privacy.reporting import generate_analysis_artifacts


def build_parser(*, default_open: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an HTML analysis page from a contrastive-privacy output folder.",
    )
    parser.add_argument("output_folder", help="Folder created by resolution_analysis or text_resolution_analysis.")
    parser.add_argument(
        "--output",
        help="Path to the HTML file to write. Defaults to <output_folder>/analysis_report.html.",
    )
    parser.add_argument(
        "--json-output",
        help="Path to the JSON file to write. Defaults to <output_folder>/analysis_report.json.",
    )
    parser.add_argument("--title", help="Optional custom page title.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Resolution threshold to treat as a leak boundary (default: 0.0).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=6,
        help="How many examples to show in each example section (default: 6).",
    )
    parser.add_argument(
        "--low-utility-threshold",
        type=float,
        default=0.3,
        help="Reference similarity below this value is labeled low utility (default: 0.30).",
    )
    parser.add_argument(
        "--high-utility-threshold",
        type=float,
        default=0.7,
        help="Reference similarity at or above this value is labeled high utility (default: 0.70).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for optional similarity recomputation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size for similarity recomputation (default: 8).",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip original-vs-obfuscated similarity recomputation and only use resolution results.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore any existing cached JSON bundle and recompute the analysis.",
    )
    parser.add_argument(
        "--image-model",
        help="Override the image similarity model. Defaults to the model recorded in params.json.",
    )
    parser.add_argument(
        "--text-embedder",
        choices=["clip", "sbert", "qwen"],
        help="Override the text embedder type used for similarity recomputation.",
    )
    parser.add_argument(
        "--text-embedder-model",
        help="Override the text embedder model used for similarity recomputation.",
    )
    parser.add_argument(
        "--text-embedder-quantization",
        help="Override the text embedder quantization mode used for similarity recomputation.",
    )
    parser.add_argument(
        "--image-folder",
        help="Override the original image folder when resolving original/obfuscated pairs.",
    )
    parser.add_argument(
        "--text-folder",
        help="Override the original text folder when resolving original/obfuscated pairs.",
    )
    open_group = parser.add_mutually_exclusive_group()
    open_group.add_argument(
        "--open",
        dest="open_report",
        action="store_true",
        help="Open the generated HTML report in the default browser.",
    )
    open_group.add_argument(
        "--no-open",
        dest="open_report",
        action="store_false",
        help="Do not open the generated HTML report in the browser.",
    )
    parser.set_defaults(open_report=default_open)
    return parser


def generate_from_args(args: argparse.Namespace) -> dict[str, object]:
    output_folder = Path(args.output_folder).resolve()
    output_html = Path(args.output).resolve() if args.output else output_folder / "analysis_report.html"

    return generate_analysis_artifacts(
        output_folder,
        title=args.title,
        threshold=args.threshold,
        top_n=args.top_n,
        low_utility_threshold=args.low_utility_threshold,
        high_utility_threshold=args.high_utility_threshold,
        compute_similarity=not args.skip_similarity,
        device=args.device,
        batch_size=args.batch_size,
        image_model=args.image_model,
        text_embedder=args.text_embedder,
        text_embedder_model=args.text_embedder_model,
        text_embedder_quantization=args.text_embedder_quantization,
        image_folder=args.image_folder,
        text_folder=args.text_folder,
        html_output=output_html,
        json_output=args.json_output,
        refresh=args.refresh,
    )


def open_report_in_browser(html_path: str | Path) -> bool:
    return webbrowser.open(Path(html_path).resolve().as_uri())


def cli_main(*, default_open: bool = False) -> None:
    parser = build_parser(default_open=default_open)
    args = parser.parse_args()
    artifacts = generate_from_args(args)
    cache_note = " (from cache)" if artifacts["used_cache"] else ""
    print(f"Wrote analysis page to: {artifacts['html_path']}{cache_note}")
    print(f"Wrote analysis bundle to: {artifacts['json_path']}")
    if args.open_report:
        opened = open_report_in_browser(artifacts["html_path"])
        if opened:
            print(f"Opened analysis page in browser: {artifacts['html_path']}")
        else:
            print(f"Could not automatically open a browser. Report is ready at: {artifacts['html_path']}")


def main() -> None:
    cli_main(default_open=False)


if __name__ == "__main__":
    main()