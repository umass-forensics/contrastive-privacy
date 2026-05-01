#!/usr/bin/env python3
"""
Recognize named entities in text using GLiNER2.

This script uses GLiNER2 for flexible, open-vocabulary named entity recognition.
GLiNER2 can detect arbitrary entity types specified at inference time, making it
ideal for privacy-focused applications. It also supports text classification,
structured data extraction, and relation extraction in a single model.

GLiNER2 is CPU-first and requires no GPU or external API dependencies.

Example:
    recognize-entities "John Smith works at Google in San Francisco."
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Default model for GLiNER2
DEFAULT_MODEL = "fastino/gliner2-base-v1"

# Default entity types for privacy-related detection
DEFAULT_ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "address",
    "phone number",
    "email",
    "date",
    "time",
    "money",
    "credit card number",
    "social security number",
    "ip address",
    "url",
    "product",
    "medical condition",
    "medication",
]


@dataclass
class RecognizedEntity:
    """Represents a recognized entity in text."""
    
    text: str
    label: str
    start: int
    end: int
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4),
        }


class GLiNER2Recognizer:
    """Entity recognizer using GLiNER2 model."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        """
        Initialize GLiNER2 model.
        
        Args:
            model_name: HuggingFace model ID for GLiNER2.
                Available models: "fastino/gliner2-base-v1" (205M params),
                "fastino/gliner2-large-v1" (340M params).
            device: Unused (GLiNER2 is CPU-first and manages device internally).
                Kept for API compatibility.
        """
        from gliner2 import GLiNER2
        
        self.device = device
        self.model = GLiNER2.from_pretrained(model_name)
    
    def _extract_and_convert(
        self,
        text: str,
        entity_types: list[str],
        threshold: float,
    ) -> list[RecognizedEntity]:
        """Run a single extract_entities call and convert to RecognizedEntity list."""
        raw_result = self.model.extract_entities(
            text,
            entity_types,
            threshold=threshold,
            include_confidence=True,
            include_spans=True,
        )
        results = []
        entities_by_type = raw_result.get("entities", {})
        for label, items in entities_by_type.items():
            for item in items:
                confidence = item.get("confidence", 1.0)
                if confidence < threshold:
                    continue
                results.append(RecognizedEntity(
                    text=item["text"],
                    label=label,
                    start=item["start"],
                    end=item["end"],
                    confidence=confidence,
                ))
        return results

    @staticmethod
    def _merge_entities(entities: list[RecognizedEntity]) -> list[RecognizedEntity]:
        """
        Merge entities from multiple passes, resolving overlaps.

        Uses a greedy approach: sort by confidence (descending), then accept
        each entity only if it does not overlap with any already-accepted entity.
        This is analogous to non-maximum suppression for bounding boxes.
        """
        # Sort by confidence descending so higher-confidence entities win
        candidates = sorted(entities, key=lambda e: e.confidence, reverse=True)
        accepted: list[RecognizedEntity] = []
        for ent in candidates:
            if not any(ent.start < a.end and ent.end > a.start for a in accepted):
                accepted.append(ent)
        accepted.sort(key=lambda e: e.start)
        return accepted

    @staticmethod
    def _propagate_entities(
        text: str,
        entities: list[RecognizedEntity],
    ) -> list[RecognizedEntity]:
        """
        Propagate detected entities to ALL occurrences in the text.

        NER models typically return one span per entity mention, even if the
        same string (e.g. "Infinity War") appears multiple times.  This method
        searches for every occurrence of each detected entity text and creates
        additional RecognizedEntity entries so that all of them are covered.

        The search is case-insensitive so that "Infinity War", "infinity war",
        and "INFINITY WAR" are all matched.  The actual text span from the
        original document is preserved in each RecognizedEntity.

        Overlaps are resolved via _merge_entities afterwards.
        """
        # Collect unique entity texts (lowercased) with best confidence
        seen: dict[str, RecognizedEntity] = {}
        for ent in entities:
            key = ent.text.lower()
            if key not in seen or ent.confidence > seen[key].confidence:
                seen[key] = ent

        # For each unique entity text, find all occurrences (case-insensitive)
        text_lower = text.lower()
        propagated: list[RecognizedEntity] = []
        for ent_lower, template in seen.items():
            start = 0
            while True:
                idx = text_lower.find(ent_lower, start)
                if idx == -1:
                    break
                # Use the actual text from the document, not the template
                matched_text = text[idx : idx + len(ent_lower)]
                propagated.append(RecognizedEntity(
                    text=matched_text,
                    label=template.label,
                    start=idx,
                    end=idx + len(ent_lower),
                    confidence=template.confidence,
                ))
                start = idx + 1

        return GLiNER2Recognizer._merge_entities(propagated)

    def recognize(
        self,
        text: str,
        entity_types: Optional[list[str]] = None,
        threshold: float = 0.3,
        sequential_labels: bool = False,
        propagate: bool = False,
    ) -> list[RecognizedEntity]:
        """
        Recognize entities in text.
        
        Args:
            text: Input text to analyze.
            entity_types: List of entity types to detect.
            threshold: Confidence threshold (0.0-1.0). Entities below this
                confidence are filtered out.
            sequential_labels: If True, process each entity type in a separate
                model call and merge results.  This ensures strictly additive
                behaviour: adding more entity types can never reduce the number
                of detections.  Use this when you observe that combining entity
                types causes some detections to drop below the threshold.
            propagate: If True, find ALL occurrences of each detected entity
                text in the input and redact every one, not just the span the
                model returned.  Useful when the same name (e.g. "Infinity War")
                appears many times but the model only tags one occurrence.
        
        Returns:
            List of RecognizedEntity instances sorted by position.
        """
        if entity_types is None:
            entity_types = DEFAULT_ENTITY_TYPES
        
        if sequential_labels and len(entity_types) > 1:
            # Process each entity type independently so the model gives
            # full attention to one type at a time.
            all_entities: list[RecognizedEntity] = []
            for entity_type in entity_types:
                entities = self._extract_and_convert(text, [entity_type], threshold)
                all_entities.extend(entities)
            results = self._merge_entities(all_entities)
        else:
            # Default: send all entity types in a single call
            results = self._extract_and_convert(text, entity_types, threshold)
            results.sort(key=lambda x: x.start)
        
        if propagate:
            results = self._propagate_entities(text, results)
        
        return results


def load_recognizer(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> GLiNER2Recognizer:
    """
    Load an entity recognizer.
    
    Args:
        model_name: GLiNER2 model name (uses default if None).
        device: Unused (kept for API compatibility).
    
    Returns:
        GLiNER2Recognizer instance.
    """
    model_name = model_name or DEFAULT_MODEL
    print(f"  Loading GLiNER2 model: {model_name}")
    return GLiNER2Recognizer(model_name=model_name, device=device)


def recognize_entities(
    text: str,
    entity_types: Optional[list[str]] = None,
    threshold: float = 0.3,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    sequential_labels: bool = False,
    propagate: bool = False,
) -> list[RecognizedEntity]:
    """
    Recognize named entities in text.
    
    Args:
        text: Input text to analyze.
        entity_types: List of entity types to detect.
        threshold: Confidence threshold (0.0-1.0).
        model_name: GLiNER2 model name (uses default if None).
        device: Unused (kept for API compatibility).
        sequential_labels: If True, process each entity type separately.
        propagate: If True, redact all occurrences of each detected entity.
    
    Returns:
        List of RecognizedEntity instances.
    """
    recognizer = load_recognizer(model_name=model_name, device=device)
    return recognizer.recognize(
        text=text,
        entity_types=entity_types,
        threshold=threshold,
        sequential_labels=sequential_labels,
        propagate=propagate,
    )


def recognize_entities_from_file(
    file_path: str | Path,
    entity_types: Optional[list[str]] = None,
    threshold: float = 0.3,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    sequential_labels: bool = False,
    propagate: bool = False,
) -> list[RecognizedEntity]:
    """
    Recognize named entities in a text file.
    
    Args:
        file_path: Path to the text file.
        entity_types: List of entity types to detect.
        threshold: Confidence threshold (0.0-1.0).
        model_name: GLiNER2 model name (uses default if None).
        device: Unused (kept for API compatibility).
        sequential_labels: If True, process each entity type separately.
        propagate: If True, redact all occurrences of each detected entity.
    
    Returns:
        List of RecognizedEntity instances.
    """
    text = Path(file_path).read_text(encoding="utf-8")
    return recognize_entities(
        text=text,
        entity_types=entity_types,
        threshold=threshold,
        model_name=model_name,
        device=device,
        sequential_labels=sequential_labels,
        propagate=propagate,
    )


def format_output(
    text: str,
    entities: list[RecognizedEntity],
    output_format: str = "text",
) -> str:
    """Format recognized entities for output."""
    unique_labels = sorted(set(ent.label for ent in entities))
    
    if output_format == "json":
        data = {
            "text": text,
            "count": len(entities),
            "unique_labels": unique_labels,
            "entities": [ent.to_dict() for ent in entities],
        }
        return json.dumps(data, indent=2)
    
    elif output_format == "simple":
        # Just list the entities
        return ", ".join(f"{ent.text} ({ent.label})" for ent in entities) or "No entities detected"
    
    elif output_format == "annotated":
        # Show text with inline annotations
        if not entities:
            return text
        
        # Build annotated text
        result = []
        last_end = 0
        for ent in sorted(entities, key=lambda x: x.start):
            # Add text before this entity
            result.append(text[last_end:ent.start])
            # Add annotated entity
            result.append(f"[{ent.text}]({ent.label})")
            last_end = ent.end
        # Add remaining text
        result.append(text[last_end:])
        
        return "".join(result)
    
    else:  # text format (default)
        lines = []
        lines.append(f"Found {len(entities)} entities ({len(unique_labels)} unique types):\n")
        
        # Group by label
        from collections import defaultdict
        by_label = defaultdict(list)
        for ent in entities:
            by_label[ent.label].append(ent)
        
        for label in sorted(by_label.keys()):
            ents = by_label[label]
            lines.append(f"  {label.upper()}:")
            for ent in ents:
                conf_str = f" ({ent.confidence:.1%})" if ent.confidence < 1.0 else ""
                lines.append(f"    • \"{ent.text}\"{conf_str}")
        
        if not entities:
            lines.append("  No entities detected.")
        
        return "\n".join(lines)


def main() -> None:
    """CLI entry point for entity recognition."""
    parser = argparse.ArgumentParser(
        description="Recognize named entities in text using GLiNER2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recognize entities in a text string
  %(prog)s "John Smith works at Google in San Francisco."
  
  # Recognize entities from a file
  %(prog)s --file document.txt
  
  # Specify entity types to detect
  %(prog)s "Call me at 555-1234" --types "phone number" "person"
  
  # JSON output
  %(prog)s "John Smith works at Google." --format json
  
  # Show annotated text
  %(prog)s "John Smith works at Google." --format annotated
  
  # Use the larger model
  %(prog)s "John Smith works at Google." --model fastino/gliner2-large-v1

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

THRESHOLD GUIDE:
  0.1  - Very sensitive, may have false positives
  0.3  - Default, good balance
  0.5  - Conservative
  0.7+ - High confidence only
        """,
    )
    
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        default=None,
        help="Text to analyze (or use --file for file input).",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a text file to analyze.",
    )
    parser.add_argument(
        "--types", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Entity types to detect (default: privacy-related types).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3). Lower = more detections.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="GLiNER2 model name (default: fastino/gliner2-base-v1).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json", "simple", "annotated"],
        help="Output format: text (detailed), json, simple, or annotated.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save output to file instead of printing.",
    )
    parser.add_argument(
        "--sequential-labels",
        action="store_true",
        default=False,
        help="Process each entity type in a separate model call and merge results.",
    )
    parser.add_argument(
        "--propagate",
        action="store_true",
        default=False,
        help=(
            "After detecting entities, find ALL occurrences of each detected "
            "entity text and tag every one (not just the span the model "
            "returned). Useful when the same name appears many times."
        ),
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
        print(f"Reading from: {args.file}")
    elif args.text:
        text = args.text
    else:
        parser.error("Either text argument or --file is required")
    
    # Recognize entities
    model_display = args.model or DEFAULT_MODEL
    print(f"Using GLiNER2 model: {model_display}")
    if args.sequential_labels:
        print("Sequential-labels: enabled")
    if args.propagate:
        print("Propagate: enabled (all occurrences will be tagged)")
    
    entities = recognize_entities(
        text=text,
        entity_types=args.types,
        threshold=args.threshold,
        model_name=args.model,
        sequential_labels=args.sequential_labels,
        propagate=args.propagate,
    )
    
    # Format output
    output = format_output(text=text, entities=entities, output_format=args.format)
    
    # Output results
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\nOutput saved to: {args.output}")
    else:
        print("\n" + output)


if __name__ == "__main__":
    main()
