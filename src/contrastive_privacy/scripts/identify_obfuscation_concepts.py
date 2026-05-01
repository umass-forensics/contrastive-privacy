#!/usr/bin/env python3
"""
Identify concepts to obfuscate in images to preserve privacy of a target concept.

This script uses the fal.ai OpenRouter Vision API to analyze images and identify
which concepts should be obfuscated to preserve the privacy of a target concept.

For example, if the target concept is "the identity of the restaurant", the script
will identify concepts like "logo", "sign", "burger", "menu" that should be obfuscated.

Examples:
    identify-obfuscation-concepts ./images "the identity of the restaurant"
    identify-obfuscation-concepts photo.jpg "the identity of the person"
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

# fal.ai OpenRouter Vision endpoint and model mapping (same LLM choices as text_anonymize)
OPENROUTER_VISION_ENDPOINT = "openrouter/router/vision"
OPENROUTER_VISION_MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "opus-4.6": "anthropic/claude-opus-4.6",
}
DEFAULT_VISION_MODEL = "gpt-5.4"


# Characters to strip from concept boundaries (quotes, commas, spaces)
_CONCEPT_JUNK = " \t\","


def _clean_concept(s: str) -> list[str]:
    """
    Remove superfluous quotes and commas from boundaries, then split on comma
    so one malformed string can yield multiple concepts. Returns a list of
    cleaned, non-empty concept strings.
    """
    s = str(s).strip()
    while s and s[0] in _CONCEPT_JUNK:
        s = s[1:]
    while s and s[-1] in _CONCEPT_JUNK:
        s = s[:-1]
    s = s.strip()
    if not s:
        return []
    # One string may contain multiple concepts (e.g. "a", "b")
    out = []
    for part in s.split(","):
        part = part.strip()
        while part and part[0] in _CONCEPT_JUNK:
            part = part[1:]
        while part and part[-1] in _CONCEPT_JUNK:
            part = part[:-1]
        part = part.strip()
        if part:
            out.append(part)
    return out


def _extract_code_fence_body(text: str) -> str:
    """Extract body from a markdown code fence when present."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.split("\n")
    json_lines = []
    in_json = False
    for line in lines:
        if line.strip().startswith("```"):
            if in_json:
                break
            in_json = True
            continue
        if in_json:
            json_lines.append(line)
    return "\n".join(json_lines).strip()


def _parse_concepts_fallback(response_text: str) -> list[str]:
    """
    Best-effort parser for malformed model outputs.
    Handles truncated JSON arrays and line-based outputs.
    """
    concepts: list[str] = []

    # 1) Recover complete quoted strings even from malformed JSON.
    for quoted in re.findall(r'"([^"\n]+)"', response_text):
        for part in _clean_concept(quoted):
            concepts.append(part.lower())

    # 2) If array-like output is truncated, split by commas and clean fragments.
    if not concepts and "[" in response_text:
        bracket_content = response_text.split("[", 1)[1]
        for fragment in bracket_content.split(","):
            fragment = fragment.strip().lstrip("]").rstrip("]")
            if not fragment:
                continue
            # Remove an unmatched leading quote from truncated fragments.
            if fragment.startswith('"') and fragment.count('"') == 1:
                fragment = fragment[1:]
            for part in _clean_concept(fragment):
                concepts.append(part.lower())

    # 3) Final fallback: old line-based parsing.
    if not concepts:
        for line in response_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                for part in _clean_concept(line.lstrip("-•*")):
                    concepts.append(part.lower())
            elif line and not line.startswith("[") and not line.startswith("]"):
                for part in _clean_concept(line):
                    concepts.append(part.lower())

    # Preserve order while deduplicating.
    deduped = list(dict.fromkeys(concepts))
    return deduped[:20]


def _resolve_openrouter_model(model_name: str) -> str:
    """Resolve user-facing name or raw OpenRouter ID to OpenRouter model ID."""
    if model_name in OPENROUTER_VISION_MODELS:
        return OPENROUTER_VISION_MODELS[model_name]
    if "/" in model_name:
        return model_name.strip()
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Use a preset ({list(OPENROUTER_VISION_MODELS.keys())}) or an OpenRouter model ID (e.g. google/gemini-3.1-pro-preview)."
    )


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def analyze_image_for_obfuscation(
    image_path: Path,
    target_concept: str,
    fal_key: Optional[str] = None,
    model: str = DEFAULT_VISION_MODEL,
) -> list[str]:
    """
    Use fal.ai OpenRouter Vision API to identify concepts in an image that should be obfuscated.

    Args:
        image_path: Path to the image file.
        target_concept: The target concept to preserve privacy of (e.g., "the identity of the restaurant").
        fal_key: fal.ai API key. If None, uses FAL_KEY environment variable.
        model: Preset (gpt-5.4, gemini-3.1-pro, opus-4.6) or exact OpenRouter model ID (e.g. google/gemini-3.1-pro-preview).

    Returns:
        List of concepts that should be obfuscated.
    """
    import tempfile
    import fal_client

    if fal_key is None:
        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            raise ValueError(
                "fal.ai API key not provided. Set FAL_KEY environment variable "
                "or pass --api-key argument."
            )

    openrouter_model = _resolve_openrouter_model(model)

    prompt = f"""Analyze this image and identify specific concepts (objects, text, visual elements) that should be obfuscated to preserve the privacy of the following target concept: "{target_concept}".

Please provide a list of general, reusable concepts that appear in this image and would reveal information about the target concept if left visible. Use general, broad concepts whenever possible (e.g., "logo" instead of "McDonald's logo", "sign" instead of "Joe's Restaurant sign", "menu" instead of "specific menu item names").

Return ONLY a JSON array of concept strings, one per line, with no additional text or explanation. Each concept should be a single word or short phrase (2-3 words maximum). Focus on concepts that are directly related to revealing the target concept.

Example format:
["logo", "sign", "menu", "brand name", "text"]

If no relevant concepts are found, return an empty array: []
"""

    try:
        # Work around non-ASCII path issues in fal_client by copying the image
        # to a temporary file with an ASCII-only name before upload.
        ascii_image_path: Optional[Path] = None
        image_path_str = str(image_path)

        # If the original path contains non-ASCII characters, skip the initial
        # upload attempt and go straight to an ASCII-safe temp filename to avoid
        # noisy fal_client fallback warnings.
        if image_path_str.isascii():
            try:
                image_url = fal_client.upload_file(image_path_str)
            except Exception:
                with tempfile.NamedTemporaryFile(
                    prefix="fal_upload_", suffix=image_path.suffix, delete=False
                ) as tmp:
                    ascii_image_path = Path(tmp.name)
                    with image_path.open("rb") as src:
                        tmp.write(src.read())
                image_url = fal_client.upload_file(str(ascii_image_path))
        else:
            with tempfile.NamedTemporaryFile(
                prefix="fal_upload_", suffix=image_path.suffix, delete=False
            ) as tmp:
                ascii_image_path = Path(tmp.name)
                with image_path.open("rb") as src:
                    tmp.write(src.read())
            image_url = fal_client.upload_file(str(ascii_image_path))

        arguments = {
            "image_urls": [image_url],
            "prompt": prompt,
            "model": openrouter_model,
            "temperature": 0.3,
            "max_tokens": 500,
        }
        lower_model = openrouter_model.lower()
        if (
            "gemini-2" in lower_model
            or "gemini-3" in lower_model
            or "gpt-5.4-pro" in lower_model
        ):
            arguments["reasoning"] = True

        result = fal_client.subscribe(
            OPENROUTER_VISION_ENDPOINT,
            arguments=arguments,
        )
        response_text = (result.get("output") or "").strip()

        try:
            response_text = _extract_code_fence_body(response_text)

            concepts = json.loads(response_text)
            if isinstance(concepts, list):
                result = []
                for c in concepts:
                    if c is None:
                        continue
                    for part in _clean_concept(c):
                        result.append(part.lower())
                return result
            else:
                print(f"Warning: Expected list but got {type(concepts)} for {image_path}")
                return []
        except json.JSONDecodeError as e:
            recovered = _parse_concepts_fallback(response_text)
            if recovered:
                return recovered

            print(f"Warning: Failed to parse JSON response for {image_path}: {e}")
            print(f"Response was: {response_text[:200]}...")
            return recovered

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []
    finally:
        # Clean up any temporary ASCII-only file we created.
        try:
            if "ascii_image_path" in locals() and ascii_image_path is not None:
                if ascii_image_path.exists():
                    ascii_image_path.unlink()
        except Exception:
            # Best-effort cleanup; ignore failures.
            pass


def find_images_in_folder(folder_path: Path) -> list[Path]:
    """
    Find all image files in a folder.

    Args:
        folder_path: Path to the folder.

    Returns:
        List of image file paths.
    """
    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a directory")

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder_path.glob(f"*{ext}"))
        images.extend(folder_path.glob(f"*{ext.upper()}"))

    return sorted(images)


def get_image_paths(path: Path) -> list[Path]:
    """
    Resolve a path to a list of image paths. Accepts either a single image file
    or a folder containing images.

    Args:
        path: Path to an image file or a folder containing images.

    Returns:
        List of image file paths (one element if path is a file).
    """
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        raise ValueError(
            f"Not an image file: {path}. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return find_images_in_folder(path)


def identify_obfuscation_concepts(
    path: Path,
    target_concept: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_VISION_MODEL,
    output_format: str = "text",
) -> dict:
    """
    Identify concepts to obfuscate in one or more images.

    Args:
        path: Path to a single image file or a folder containing images.
        target_concept: The target concept to preserve privacy of.
        api_key: fal.ai API key (FAL_KEY). If None, uses FAL_KEY environment variable.
        model: User-facing model name: gpt-5.4, gemini-3.1-pro, or opus-4.6.
        output_format: Output format ("text", "json", or "simple").

    Returns:
        Dictionary with results including unique concepts and per-image concepts.
    """
    image_paths = get_image_paths(path)

    if not image_paths:
        print(f"No images found in {path}")
        return {
            "unique_concepts": [],
            "per_image": {},
            "total_images": 0,
        }
    
    print(f"Found {len(image_paths)} image(s)")
    print(f"Target concept: {target_concept}")
    print(f"Analyzing images...\n")
    
    all_concepts = []
    per_image_concepts = {}
    
    # Process each image
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing {image_path.name}...", end=" ", flush=True)
        
        concepts = analyze_image_for_obfuscation(
            image_path=image_path,
            target_concept=target_concept,
            fal_key=api_key,
            model=model,
        )
        
        per_image_concepts[str(image_path)] = concepts
        all_concepts.extend(concepts)
        
        print(f"Found {len(concepts)} concept(s)")
    
    # Clean each concept (strip quotes/commas), flatten, then deduplicate
    cleaned = []
    for c in all_concepts:
        for part in _clean_concept(c):
            part = part.strip().lower()
            if part:
                cleaned.append(part)
    unique_concepts = sorted(set(cleaned), key=str.lower)
    
    print(f"\nTotal unique concepts identified: {len(unique_concepts)}")
    
    return {
        "unique_concepts": unique_concepts,
        "per_image": per_image_concepts,
        "total_images": len(image_paths),
        "target_concept": target_concept,
    }


def format_output(results: dict, output_format: str = "text") -> str:
    """
    Format results for output.
    
    Args:
        results: Results dictionary from identify_obfuscation_concepts.
        output_format: Output format ("text", "json", or "simple").
        
    Returns:
        Formatted output string.
    """
    def format_concepts_one_line(concepts: list[str]) -> str:
        # Clean first (strip quotes/commas, split on comma), then deduplicate
        cleaned: list[str] = []
        for c in concepts:
            for part in _clean_concept(c):
                part = part.strip().lower()
                if part:
                    cleaned.append(part)
        unique = sorted(set(cleaned), key=str.lower)
        return ", ".join(f'"{c}"' for c in unique)

    if output_format == "json":
        return json.dumps(results, indent=2)
    
    elif output_format == "simple":
        return format_concepts_one_line(results["unique_concepts"])
    
    else:  # text format (default)
        lines = []
        lines.append("=" * 70)
        lines.append("OBFUSCATION CONCEPTS")
        lines.append("=" * 70)
        lines.append(f"Target concept: {results['target_concept']}")
        lines.append(f"Images analyzed: {results['total_images']}")
        lines.append(f"Unique concepts to obfuscate: {len(results['unique_concepts'])}")
        lines.append("")
        lines.append(f"Final concepts (one line): {format_concepts_one_line(results['unique_concepts'])}")
        lines.append("")
        lines.append("Unique concepts:")
        for concept in results["unique_concepts"]:
            lines.append(f"  • {concept}")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


def main() -> None:
    """CLI entry point for identifying obfuscation concepts."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single image file or folder containing images to analyze",
    )
    parser.add_argument(
        "target_concept",
        type=str,
        help='Target concept to preserve privacy of (e.g., "the identity of the restaurant")',
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="fal.ai API key (default: uses FAL_KEY environment variable)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VISION_MODEL,
        help=f"fal.ai OpenRouter Vision model: preset (gpt-5.4, gemini-3.1-pro, opus-4.6) or exact OpenRouter ID (e.g. google/gemini-3.1-pro-preview). Default: {DEFAULT_VISION_MODEL}. Requires FAL_KEY.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json", "simple"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: print to stdout)",
    )
    
    args = parser.parse_args()

    # Validate path exists
    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")

    # Resolve to image list for validation (single file or folder)
    try:
        get_image_paths(args.path)
    except ValueError as e:
        parser.error(str(e))

    # Identify concepts
    results = identify_obfuscation_concepts(
        path=args.path,
        target_concept=args.target_concept,
        api_key=args.api_key,
        model=args.model,
        output_format=args.output_format,
    )
    
    # Format and output
    output_text = format_output(results, args.output_format)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + output_text)


if __name__ == "__main__":
    main()
