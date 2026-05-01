#!/usr/bin/env python3
"""
Anonymize or obfuscate specific entities in text automatically.

This script combines GLiNER2 named entity recognition with text obfuscation
to automatically detect and redact specified entity types in text.
Perfect for privacy applications like PII removal.

Supports multiple obfuscation modes:
- blackout: Replace entities with block characters (\u2588)
- redact: Replace entities with [REDACTED] or custom placeholder

Example: anonymize person names and phone numbers in a document.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from contrastive_privacy.scripts.recognize_entities import (
    DEFAULT_MODEL,
    RecognizedEntity,
    GLiNER2Recognizer,
    load_recognizer,
)


# Unicode block character for blackout
BLOCK_CHAR = "\u2588"


@dataclass
class AnonymizationResult:
    """Result of text anonymization."""
    
    original_text: str
    anonymized_text: str
    entities_found: list[RecognizedEntity]
    entities_redacted: int
    coverage: float  # Fraction of text that was redacted
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "entities_found": [e.to_dict() for e in self.entities_found],
            "entities_redacted": self.entities_redacted,
            "coverage_percent": round(self.coverage * 100, 2),
        }


def blackout_text(text: str, start: int, end: int) -> str:
    """
    Replace a portion of text with block characters.
    
    Args:
        text: Original text.
        start: Start index of region to blackout.
        end: End index of region to blackout.
    
    Returns:
        Text with the specified region replaced by block characters.
    """
    region_length = end - start
    blackout = BLOCK_CHAR * region_length
    return text[:start] + blackout + text[end:]


def blackout_text_compact_words(text: str, start: int, end: int) -> str:
    """
    Replace each non-whitespace token in a span with a single block character.

    Example:
        "John Smith" -> "█ █"
    """
    region = text[start:end]
    compact = re.sub(r"\S+", BLOCK_CHAR, region)
    return text[:start] + compact + text[end:]


def redact_text(text: str, start: int, end: int, placeholder: str = "[REDACTED]") -> str:
    """
    Replace a portion of text with a placeholder.
    
    Args:
        text: Original text.
        start: Start index of region to redact.
        end: End index of region to redact.
        placeholder: Replacement text.
    
    Returns:
        Text with the specified region replaced by the placeholder.
    """
    return text[:start] + placeholder + text[end:]


# fal.ai OpenRouter endpoint for concept-based redaction
OPENROUTER_ENDPOINT = "openrouter/router"

# User-facing model names -> OpenRouter model IDs (via fal.ai openrouter/router)
OPENROUTER_CONCEPT_MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "opus-4.6": "anthropic/claude-opus-4.6",
}

# Default concept model (must be one of OPENROUTER_CONCEPT_MODELS or a raw OpenRouter ID)
DEFAULT_CONCEPT_MODEL = "gpt-5.4"

# Sampling temperature for fal.ai OpenRouter concept redaction (openrouter/router)
DEFAULT_CONCEPT_TEMPERATURE = 0.1


def _build_instance_entities(text: str, instances: Optional[list[str]]) -> list[RecognizedEntity]:
    """
    Build case-insensitive entity spans for explicitly provided instances.

    Each instance is matched as a literal substring (via re.escape) anywhere in
    the text, including inside larger words. Every non-overlapping regex match
    is obfuscated in full.

    This path does not invoke GLiNER2: it performs deterministic matching only.
    """
    if not instances:
        return []

    normalized_instances: list[str] = []
    seen: set[str] = set()
    for instance in instances:
        token = instance.strip()
        if not token:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized_instances.append(token)

    if not normalized_instances:
        return []

    entities: list[RecognizedEntity] = []
    for token in normalized_instances:
        pattern = re.compile(re.escape(token), flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            obf_start, obf_end = match.start(), match.end()
            entities.append(
                RecognizedEntity(
                    text=text[obf_start:obf_end],
                    label="instance",
                    start=obf_start,
                    end=obf_end,
                    confidence=1.0,
                )
            )

    return GLiNER2Recognizer._merge_entities(entities)


def _apply_entities_to_text(
    text: str,
    entities: list[RecognizedEntity],
    mode: str,
    placeholder: str,
    compact_blackout_words: bool = False,
) -> tuple[str, int]:
    """Apply span obfuscation from end-to-start and return (text, chars_redacted)."""
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
    result_text = text
    chars_redacted = 0

    for entity in sorted_entities:
        original_length = entity.end - entity.start
        if mode == "blackout":
            if compact_blackout_words:
                result_text = blackout_text_compact_words(result_text, entity.start, entity.end)
            else:
                result_text = blackout_text(result_text, entity.start, entity.end)
            chars_redacted += original_length
        elif mode == "redact":
            result_text = redact_text(result_text, entity.start, entity.end, placeholder)
            chars_redacted += original_length
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'blackout' or 'redact'.")

    return result_text, chars_redacted


def _coverage_from_texts(original_text: str, anonymized_text: str) -> float:
    """Compute coverage as fraction of changed character positions."""
    max_len = max(len(original_text), len(anonymized_text))
    if max_len == 0:
        return 0.0
    changed = sum(
        1
        for i in range(max_len)
        if (original_text[i] if i < len(original_text) else "")
        != (anonymized_text[i] if i < len(anonymized_text) else "")
    )
    return changed / max_len


def _resolve_openrouter_model(model_name: str) -> str:
    """Resolve user-facing name or raw OpenRouter ID to OpenRouter model ID."""
    if model_name in OPENROUTER_CONCEPT_MODELS:
        return OPENROUTER_CONCEPT_MODELS[model_name]
    if "/" in model_name:
        return model_name.strip()
    raise ValueError(
        f"Unknown concept model: {model_name}. "
        f"Use a preset ({list(OPENROUTER_CONCEPT_MODELS.keys())}) or an OpenRouter model ID (e.g. google/gemini-3.1-pro-preview)."
    )


def redact_by_concept(
    text: str,
    concept: str,
    model_name: str = DEFAULT_CONCEPT_MODEL,
    temperature: float = DEFAULT_CONCEPT_TEMPERATURE,
) -> AnonymizationResult:
    """
    Redact text using a fal.ai/OpenRouter LLM and a privacy concept description.

    The model is prompted to replace any words consistent with the privacy concept
    with block characters (█), preserving length and layout like blackout mode.

    Args:
        text: Input text to redact.
        concept: Description of the privacy concept (e.g. "anything that can
            identify the movie discussed in this passage").
        model_name: Preset (gpt-5.4, gemini-3.1-pro, opus-4.6) or exact OpenRouter model ID (e.g. google/gemini-3.1-pro-preview).
        temperature: OpenRouter sampling temperature for the redaction call.

    Returns:
        AnonymizationResult with anonymized_text and coverage derived from
        the proportion of block characters in the result.
    """
    import fal_client
    import os

    if not os.getenv("FAL_KEY"):
        raise ValueError(
            "fal.ai API key not set. Set FAL_KEY for concept (OpenRouter) redaction."
        )

    openrouter_model = _resolve_openrouter_model(model_name)
    print(f"  Using concept model: {model_name} (fal.ai OpenRouter: {openrouter_model})")

    system_prompt = (
        "You redact text for privacy by replacing sensitive parts with block characters (█). "
        "You output only the redacted text."
    )
    user_prompt = (
        "Please redact the following passage by replacing with a single block character (█, "
        "Unicode U+2588) every word or phrase that is consistent with this privacy concept: "
        f'"{concept}".\n\n'
        "Rules: (1) Replace only the sensitive tokens with █, one character per original character "
        "so that the length and layout of the passage are preserved. (2) Do not redact words that "
        "do not match the concept. (3) Return only the redacted passage, no explanation. (4) You must "
        "redact something from the passage.\n\n"
        "Passage:\n"
        f"{text}"
    )

    # Gemini 2.x/3.x models (e.g. gemini-2.5-flash, gemini-3-pro-preview) and some frontier OpenAI models
    # (e.g. gpt-5.4-pro) require reasoning=True on the OpenRouter fal.ai endpoint.
    arguments = {
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "model": openrouter_model,
        "temperature": temperature,
    }
    lower_model = openrouter_model.lower()
    if (
        "gemini-2" in lower_model
        or "gemini-3" in lower_model
        or "gpt-5.4-pro" in lower_model
    ):
        arguments["reasoning"] = True

    result = fal_client.subscribe(
        OPENROUTER_ENDPOINT,
        arguments=arguments,
    )

    anonymized = (result.get("output") or "").strip()
    # Remove any surrounding markdown code fences if present
    if anonymized.startswith("```"):
        lines = anonymized.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        anonymized = "\n".join(lines)
    # Coverage: fraction of characters that are block characters
    block_char = "\u2588"
    num_blocks = sum(1 for c in anonymized if c == block_char)
    coverage = num_blocks / len(text) if len(text) > 0 else 0.0
    return AnonymizationResult(
        original_text=text,
        anonymized_text=anonymized,
        entities_found=[],
        entities_redacted=0,
        coverage=coverage,
    )


def anonymize_text(
    text: str,
    entity_types: Optional[list[str]] = None,
    mode: str = "blackout",
    placeholder: str = "[REDACTED]",
    threshold: float = 0.3,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    recognizer: Optional[GLiNER2Recognizer] = None,
    sequential_labels: bool = False,
    propagate: bool = True,
    approach: str = "entity",
    concept: Optional[str] = None,
    concept_model: Optional[str] = None,
    concept_temperature: float = DEFAULT_CONCEPT_TEMPERATURE,
    instances: Optional[list[str]] = None,
    compact_blackout_words: bool = False,
) -> AnonymizationResult:
    """
    Anonymize or redact text by either entity detection (GLiNER2) or concept (fal.ai OpenRouter).
    
    Args:
        text: Input text to anonymize.
        entity_types: List of entity types to detect and anonymize (entity approach).
            If None or empty and approach is entity, no obfuscation is performed.
        mode: "blackout" (block chars) or "redact" (placeholder text). Used only for entity approach.
        placeholder: Replacement text for redact mode (entity approach only).
        threshold: Confidence threshold for entity detection (0.0-1.0).
        model_name: GLiNER2 model name (uses default if None).
        device: Unused (kept for API compatibility).
        recognizer: Pre-loaded GLiNER2Recognizer (optional, for reuse across calls).
        sequential_labels: If True, process each entity type separately and merge.
        propagate: If True (default), find ALL occurrences of each detected entity text.
        approach: "entity" (NER-based) or "concept" (fal.ai OpenRouter LLM by description).
        concept: Privacy concept description (required when approach is "concept").
        concept_model: fal.ai OpenRouter model for concept approach (preset or exact OpenRouter ID, e.g. google/gemini-3.1-pro-preview).
        concept_temperature: OpenRouter sampling temperature when approach is "concept".
        instances: Literal substrings to obfuscate (case-insensitive; every match in the text,
            including inside larger words; see _build_instance_entities).
            This matching is deterministic and does not invoke GLiNER2.
        compact_blackout_words: If True and mode is "blackout", replace each obfuscated
            non-whitespace token with a single block character instead of one per character.
    
    Returns:
        AnonymizationResult with original text, anonymized text, and metadata.
    """
    if approach == "concept":
        if not concept or not concept.strip():
            raise ValueError("When approach is 'concept', concept must be a non-empty string.")
        concept_result = redact_by_concept(
            text=text,
            concept=concept.strip(),
            model_name=concept_model or DEFAULT_CONCEPT_MODEL,
            temperature=concept_temperature,
        )
        instance_entities = _build_instance_entities(concept_result.anonymized_text, instances)
        if not instance_entities:
            return concept_result

        final_text, _ = _apply_entities_to_text(
            text=concept_result.anonymized_text,
            entities=instance_entities,
            mode=mode,
            placeholder=placeholder,
            compact_blackout_words=compact_blackout_words,
        )
        coverage = _coverage_from_texts(text, final_text)
        return AnonymizationResult(
            original_text=text,
            anonymized_text=final_text,
            entities_found=instance_entities,
            entities_redacted=len(instance_entities),
            coverage=coverage,
        )

    # Entity approach
    if not entity_types and not instances:
        return AnonymizationResult(
            original_text=text,
            anonymized_text=text,
            entities_found=[],
            entities_redacted=0,
            coverage=0.0,
        )

    entity_entities: list[RecognizedEntity] = []
    if entity_types:
        # Load recognizer only when GLiNER2 entity detection is requested.
        if recognizer is None:
            recognizer = load_recognizer(model_name=model_name, device=device)
        entity_entities.extend(
            recognizer.recognize(
                text=text,
                entity_types=entity_types,
                threshold=threshold,
                sequential_labels=sequential_labels,
                propagate=propagate,
            )
        )

    # Always apply GLiNER2 entity obfuscation first, then run deterministic
    # instance matching on the already-obfuscated text.
    current_text = text
    if entity_entities:
        current_text, _ = _apply_entities_to_text(
            text=current_text,
            entities=entity_entities,
            mode=mode,
            placeholder=placeholder,
            compact_blackout_words=compact_blackout_words,
        )

    instance_entities = _build_instance_entities(current_text, instances)
    if instance_entities:
        current_text, _ = _apply_entities_to_text(
            text=current_text,
            entities=instance_entities,
            mode=mode,
            placeholder=placeholder,
            compact_blackout_words=compact_blackout_words,
        )

    entities_found = entity_entities + instance_entities
    if not entities_found:
        return AnonymizationResult(
            original_text=text,
            anonymized_text=text,
            entities_found=[],
            entities_redacted=0,
            coverage=0.0,
        )

    coverage = _coverage_from_texts(text, current_text)

    return AnonymizationResult(
        original_text=text,
        anonymized_text=current_text,
        entities_found=entities_found,
        entities_redacted=len(entities_found),
        coverage=coverage,
    )


def anonymize_file(
    input_path: str | Path,
    output_path: str | Path,
    entity_types: Optional[list[str]] = None,
    mode: str = "blackout",
    placeholder: str = "[REDACTED]",
    threshold: float = 0.3,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    sequential_labels: bool = False,
    propagate: bool = True,
    approach: str = "entity",
    concept: Optional[str] = None,
    concept_model: Optional[str] = None,
    concept_temperature: float = DEFAULT_CONCEPT_TEMPERATURE,
    instances: Optional[list[str]] = None,
    compact_blackout_words: bool = False,
) -> AnonymizationResult:
    """
    Anonymize or redact a text file (entity or concept approach).
    
    Args:
        input_path: Path to input text file.
        output_path: Path where anonymized text will be saved.
        entity_types: List of entity types (entity approach).
        mode: "blackout" or "redact" (entity approach only).
        placeholder: Placeholder for redact mode.
        threshold: Detection threshold (entity approach).
        model_name: GLiNER2 model name (entity approach).
        device: Unused (kept for API compatibility).
        sequential_labels: Process each entity type separately (entity approach).
        propagate: Redact all occurrences of each entity (entity approach). Default True.
        approach: "entity" or "concept".
        concept: Privacy concept description (required when approach is "concept").
        concept_model: fal.ai OpenRouter model for concept approach.
        concept_temperature: OpenRouter sampling temperature when approach is "concept".
        instances: Literal substrings to obfuscate (case-insensitive; every match in the text,
            including inside larger words; see _build_instance_entities).
        compact_blackout_words: If True and mode is "blackout", replace each obfuscated
            non-whitespace token with a single block character.
    
    Returns:
        AnonymizationResult with metadata.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Read input
    text = input_path.read_text(encoding="utf-8")
    
    # Anonymize
    result = anonymize_text(
        text=text,
        entity_types=entity_types,
        mode=mode,
        placeholder=placeholder,
        threshold=threshold,
        model_name=model_name,
        device=device,
        sequential_labels=sequential_labels,
        propagate=propagate,
        approach=approach,
        concept=concept,
        concept_model=concept_model,
        concept_temperature=concept_temperature,
        instances=instances,
        compact_blackout_words=compact_blackout_words,
    )
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.anonymized_text, encoding="utf-8")
    
    return result


def main() -> None:
    """CLI entry point for text anonymization."""
    parser = argparse.ArgumentParser(
        description="Automatically detect and anonymize entities in text using GLiNER2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Blackout person names
  %(prog)s "John Smith called from 555-1234." --entities person --output result.txt
  
  # Blackout multiple entity types
  %(prog)s "John works at Google." --entities person organization --output result.txt
  
  # Anonymize a file with [REDACTED] placeholders
  %(prog)s --input document.txt --output anonymized.txt --entities person --mode redact
  
  # Use custom placeholder
  %(prog)s "John Smith" --entities person --mode redact --placeholder "[NAME REMOVED]" --output out.txt
  
  # Use the larger GLiNER2 model
  %(prog)s "John Smith works at Apple." --entities person organization --model fastino/gliner2-large-v1 --output result.txt

MODES:
  blackout (default):
    Replaces entities with block characters
    Preserves text length and layout
    Example: "John Smith" -> "XXXXXXXXXX"
    
  redact:
    Replaces entities with a placeholder text
    Changes text length
    Example: "John Smith" -> "[REDACTED]"

ABOUT GLiNER2:
  GLiNER2 is a unified, open-vocabulary NER model that can detect ANY
  entity type you specify at inference time. It is CPU-first and requires
  no GPU or external API dependencies.

  Available models:
    fastino/gliner2-base-v1  (205M params, default)
    fastino/gliner2-large-v1 (340M params, more accurate)

DEFAULT ENTITY TYPES:
  person, organization, location, address, phone number, email,
  date, time, money, credit card number, social security number,
  ip address, url, product, medical condition, medication

  You can use ANY entity type (GLiNER2 is open-vocabulary):
  "car model", "case number", "defendant name", etc.

TIPS:
  - You MUST specify --entities for obfuscation to occur
  - Lower --threshold (0.1-0.2) to catch more potential entities
  - Higher --threshold (0.5-0.7) for higher confidence matches only
        """,
    )
    
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        default=None,
        help="Text to anonymize (or use --input for file input).",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input text file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save anonymized text.",
    )
    parser.add_argument(
        "--entities", "-e",
        type=str,
        nargs="+",
        default=None,
        help="Entity types to detect and anonymize. Required for obfuscation to occur. "
             "Examples: 'person', 'organization', 'phone number', 'email', 'location'.",
    )
    parser.add_argument(
        "--instances",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Literal substrings to obfuscate (case-insensitive; regex search with re.escape). "
            "Every occurrence is obfuscated, including inside larger words. "
            "This path does not invoke GLiNER2."
        ),
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="blackout",
        choices=["blackout", "redact"],
        help="Anonymization mode: 'blackout' (block chars) or 'redact' (placeholder). Default: blackout.",
    )
    parser.add_argument(
        "--placeholder", "-p",
        type=str,
        default="[REDACTED]",
        help="Placeholder text for redact mode (default: [REDACTED]).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Entity detection threshold (default: 0.3). Lower = more detections.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="GLiNER2 model name (default: fastino/gliner2-base-v1).",
    )
    parser.add_argument(
        "--sequential-labels",
        action="store_true",
        help="Process each entity type separately and merge results. "
             "This ensures strictly additive behavior: adding more entity types can never "
             "reduce the number of detections. Use this when you observe that combining entity "
             "types causes some detections to drop below the threshold.",
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
    parser.add_argument(
        "--approach",
        type=str,
        default="entity",
        choices=["entity", "concept"],
        help="Obfuscation approach: 'entity' (NER with GLiNER2) or 'concept' (fal.ai OpenRouter by description). Default: entity.",
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
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including detected entities.",
    )
    
    args = parser.parse_args()
    
    if args.approach == "concept":
        if not args.concept or not args.concept.strip():
            parser.error("When --approach is 'concept', --concept is required and must be non-empty.")
    elif args.approach == "entity" and not args.entities and not args.instances:
        parser.error("When --approach is 'entity', provide --entities and/or --instances.")
    
    # Get input text
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {input_path}")
        text = input_path.read_text(encoding="utf-8")
        print(f"Reading from: {input_path}")
    elif args.text:
        text = args.text
    else:
        parser.error("Either text argument or --input file is required")
    
    model_display = args.model or DEFAULT_MODEL
    print(f"Input length: {len(text)} characters")
    print(f"Approach: {args.approach}")
    if args.approach == "entity":
        print(f"Mode: {args.mode}")
        print(f"Model: {model_display}")
        if args.entities:
            print(f"Entity types: {', '.join(args.entities)}")
        if args.instances:
            print(f"Instances: {', '.join(args.instances)}")
        if args.sequential_labels:
            print("Sequential labels: enabled (each entity type processed separately)")
        if args.propagate:
            print("Propagate: enabled (all occurrences will be anonymized)")
    else:
        print(f"Concept: {args.concept}")
        print(f"Concept model: {args.concept_model or DEFAULT_CONCEPT_MODEL}")
        print(f"Concept temperature: {args.concept_temperature}")
    
    # Anonymize
    result = anonymize_text(
        text=text,
        entity_types=args.entities,
        mode=args.mode,
        placeholder=args.placeholder,
        threshold=args.threshold,
        model_name=args.model,
        sequential_labels=args.sequential_labels,
        propagate=args.propagate,
        approach=args.approach,
        concept=args.concept,
        concept_model=args.concept_model,
        concept_temperature=args.concept_temperature,
        instances=args.instances,
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.anonymized_text, encoding="utf-8")
    
    # Report results
    print(f"\nResults:")
    print(f"  Entities found: {result.entities_redacted}")
    print(f"  Coverage: {result.coverage * 100:.1f}% of text anonymized")
    print(f"  Output saved to: {output_path}")
    
    if args.verbose and result.entities_found:
        print(f"\nDetected entities:")
        for ent in result.entities_found:
            print(f"  \u2022 \"{ent.text}\" ({ent.label}, confidence: {ent.confidence:.1%})")
    
    # Preview
    preview_len = 200
    if len(result.anonymized_text) > preview_len:
        preview = result.anonymized_text[:preview_len] + "..."
    else:
        preview = result.anonymized_text
    print(f"\nPreview:\n{preview}")


if __name__ == "__main__":
    main()
