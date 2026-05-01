#!/usr/bin/env python3
"""
Recognize objects in images using OWL-ViT (Open-Vocabulary Object Detection).

This script uses Google's OWL-ViT model to detect objects from a large
vocabulary of categories. Unlike COCO-based detectors (91 categories),
OWL-ViT can detect hundreds of object types with confidence thresholds.

Example: detect all common objects in a photo with adjustable sensitivity.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection


# Comprehensive list of common object categories (300+)
DEFAULT_CATEGORIES = [
    # People
    "person", "man", "woman", "child", "baby", "face", "hand", "crowd",
    
    # Animals
    "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra",
    "giraffe", "lion", "tiger", "monkey", "rabbit", "duck", "chicken", "fish",
    "whale", "dolphin", "shark", "turtle", "snake", "frog", "insect", "butterfly",
    "bee", "spider", "squirrel", "deer", "fox", "wolf", "pig", "goat",
    
    # Vehicles
    "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "helicopter",
    "boat", "ship", "train", "subway", "tram", "scooter", "skateboard",
    "surfboard", "kayak", "canoe", "jet ski", "rv", "van", "taxi", "ambulance",
    "fire truck", "police car", "tractor", "bulldozer", "crane",
    
    # Outdoor objects
    "tree", "flower", "grass", "bush", "rock", "mountain", "hill", "beach",
    "sand", "water", "ocean", "sea", "lake", "river", "waterfall", "sky",
    "cloud", "sun", "moon", "star", "rainbow", "snow", "ice", "rain",
    "bridge", "road", "street", "sidewalk", "path", "fence", "gate", "wall",
    "building", "house", "skyscraper", "tower", "church", "temple", "castle",
    "fountain", "statue", "monument", "sign", "traffic light", "street lamp",
    "bench", "picnic table", "umbrella", "tent", "flag", "pole", "antenna",
    
    # Indoor objects
    "chair", "table", "desk", "sofa", "couch", "bed", "pillow", "blanket",
    "curtain", "carpet", "rug", "lamp", "chandelier", "mirror", "window",
    "door", "stairs", "elevator", "shelf", "cabinet", "drawer", "closet",
    "fireplace", "radiator", "fan", "air conditioner",
    
    # Kitchen
    "refrigerator", "oven", "stove", "microwave", "toaster", "blender",
    "coffee maker", "kettle", "sink", "faucet", "dishwasher", "counter",
    "cutting board", "knife", "fork", "spoon", "plate", "bowl", "cup",
    "mug", "glass", "bottle", "jar", "pan", "pot", "lid",
    
    # Food
    "apple", "banana", "orange", "lemon", "grape", "strawberry", "watermelon",
    "pineapple", "mango", "peach", "pear", "cherry", "tomato", "carrot",
    "broccoli", "lettuce", "onion", "potato", "corn", "pepper", "cucumber",
    "mushroom", "bread", "sandwich", "pizza", "hamburger", "hot dog", "taco",
    "sushi", "rice", "pasta", "soup", "salad", "cake", "pie", "cookie",
    "donut", "ice cream", "chocolate", "candy", "cheese", "egg", "meat",
    "steak", "chicken", "fish", "shrimp", "lobster",
    
    # Electronics
    "phone", "smartphone", "cell phone", "telephone", "computer", "laptop",
    "tablet", "keyboard", "mouse", "monitor", "screen", "television", "tv",
    "remote control", "camera", "video camera", "speaker", "headphones",
    "microphone", "radio", "clock", "watch", "calculator", "printer",
    "scanner", "projector", "drone", "robot", "game controller", "console",
    
    # Clothing & accessories
    "shirt", "pants", "jeans", "dress", "skirt", "jacket", "coat", "sweater",
    "hoodie", "suit", "tie", "hat", "cap", "helmet", "glasses", "sunglasses",
    "shoe", "boot", "sandal", "sneaker", "sock", "glove", "scarf", "belt",
    "bag", "purse", "backpack", "suitcase", "wallet", "watch", "jewelry",
    "necklace", "bracelet", "ring", "earring",
    
    # Sports & recreation
    "ball", "soccer ball", "basketball", "football", "baseball", "tennis ball",
    "golf ball", "volleyball", "tennis racket", "baseball bat", "golf club",
    "hockey stick", "ski", "snowboard", "skateboard", "surfboard", "frisbee",
    "kite", "parachute", "weights", "dumbbell", "treadmill", "bicycle",
    "swimming pool", "hot tub", "playground", "swing", "slide", "trampoline",
    
    # Tools & equipment
    "hammer", "screwdriver", "wrench", "pliers", "saw", "drill", "ladder",
    "shovel", "rake", "hose", "sprinkler", "lawn mower", "wheelbarrow",
    "toolbox", "tape measure", "level", "paintbrush", "roller", "bucket",
    
    # Office & school
    "book", "notebook", "pen", "pencil", "marker", "eraser", "ruler",
    "scissors", "tape", "stapler", "paper clip", "folder", "binder",
    "calendar", "whiteboard", "blackboard", "desk", "chair", "bookshelf",
    "globe", "map", "poster", "certificate", "diploma",
    
    # Medical
    "medicine", "pill", "syringe", "bandage", "thermometer", "stethoscope",
    "wheelchair", "crutch", "hospital bed", "iv bag", "mask", "gloves",
    
    # Music
    "piano", "guitar", "violin", "drum", "flute", "trumpet", "saxophone",
    "microphone", "speaker", "headphones", "record", "cd", "music stand",
    
    # Toys & games
    "toy", "doll", "teddy bear", "action figure", "lego", "puzzle",
    "board game", "cards", "dice", "balloon", "kite", "yo-yo",
    
    # Nature & plants
    "plant", "flower", "rose", "tulip", "sunflower", "daisy", "tree",
    "palm tree", "pine tree", "leaf", "branch", "trunk", "root", "seed",
    "fruit", "vegetable", "cactus", "fern", "moss", "mushroom", "grass",
    
    # Containers
    "box", "crate", "basket", "bucket", "barrel", "tank", "container",
    "package", "envelope", "bag", "sack", "bin", "trash can", "recycling bin",
]


@dataclass
class DetectedObject:
    """Represents a detected object in an image."""
    
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": {
                "x_min": round(self.box[0], 2),
                "y_min": round(self.box[1], 2),
                "x_max": round(self.box[2], 2),
                "y_max": round(self.box[3], 2),
            },
        }


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def recognize_objects(
    image: Image.Image,
    threshold: float = 0.1,
    categories: Optional[list[str]] = None,
    device: Optional[str] = None,
    model_name: str = "google/owlvit-base-patch32",
) -> list[DetectedObject]:
    """
    Detect objects in an image using OWL-ViT.
    
    Args:
        image: PIL Image to analyze
        threshold: Confidence threshold (0.0-1.0). Lower = more detections.
        categories: List of object categories to search for (default: 300+ common objects)
        device: Device to run on (auto-detect if None)
        model_name: HuggingFace model identifier
        
    Returns:
        List of DetectedObject instances
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if categories is None:
        categories = DEFAULT_CATEGORIES
    
    # Load model and processor
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Process in batches (OWL-ViT has limits on text queries)
    batch_size = 50  # Process 50 categories at a time
    all_detections = []
    
    for i in range(0, len(categories), batch_size):
        batch_categories = categories[i:i + batch_size]
        
        # Process image with text queries
        inputs = processor(
            text=[batch_categories],
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )[0]
        
        # Collect detections
        for score, label_idx, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
        ):
            label = batch_categories[label_idx]
            all_detections.append(
                DetectedObject(
                    label=label,
                    confidence=float(score),
                    box=tuple(box.tolist()),
                )
            )
    
    # Sort by confidence (highest first)
    all_detections.sort(key=lambda x: x.confidence, reverse=True)
    
    # Clean up
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return all_detections


def recognize_objects_from_file(
    image_path: str | Path,
    threshold: float = 0.1,
    categories: Optional[list[str]] = None,
    device: Optional[str] = None,
    model_name: str = "google/owlvit-base-patch32",
) -> list[DetectedObject]:
    """
    Detect objects in an image file.
    """
    image = load_image(image_path)
    return recognize_objects(
        image=image,
        threshold=threshold,
        categories=categories,
        device=device,
        model_name=model_name,
    )


def format_output(
    detected_objects: list[DetectedObject],
    output_format: str = "text",
    image_path: Optional[str] = None,
) -> str:
    """Format detected objects for output."""
    unique_labels = sorted(set(obj.label for obj in detected_objects))
    
    if output_format == "json":
        data = {
            "image": str(image_path) if image_path else None,
            "count": len(detected_objects),
            "unique_count": len(unique_labels),
            "unique_labels": unique_labels,
            "objects": [obj.to_dict() for obj in detected_objects],
        }
        return json.dumps(data, indent=2)
    
    elif output_format == "simple":
        return ", ".join(unique_labels) if unique_labels else "No objects detected"
    
    else:  # text format (default)
        lines = []
        if image_path:
            lines.append(f"Image: {image_path}")
        lines.append(f"Detected {len(detected_objects)} object(s) ({len(unique_labels)} unique):\n")
        
        # Group by label and show counts
        from collections import Counter
        label_counts = Counter(obj.label for obj in detected_objects)
        
        for label, count in sorted(label_counts.items()):
            # Get max confidence for this label
            max_conf = max(
                obj.confidence for obj in detected_objects if obj.label == label
            )
            if count > 1:
                lines.append(f"  • {label} (×{count}, max conf: {max_conf:.1%})")
            else:
                lines.append(f"  • {label} ({max_conf:.1%})")
        
        if not detected_objects:
            lines.append("  No objects detected above threshold.")
        
        return "\n".join(lines)


def main() -> None:
    """CLI entry point for object recognition."""
    parser = argparse.ArgumentParser(
        description="Detect objects in images using OWL-ViT (300+ categories).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect objects with default threshold (0.1)
  %(prog)s photo.jpg
  
  # Higher threshold = fewer but more confident detections
  %(prog)s photo.jpg --threshold 0.2
  
  # Lower threshold = more detections (more sensitive)
  %(prog)s photo.jpg --threshold 0.05
  
  # Simple comma-separated list
  %(prog)s photo.jpg --format simple
  
  # JSON output with bounding boxes
  %(prog)s photo.jpg --format json

THRESHOLD GUIDE:
  0.05  - Very sensitive, may have false positives
  0.10  - Default, good balance (recommended)
  0.15  - More conservative
  0.20  - High confidence only
  0.30+ - Very strict, only obvious objects

ABOUT OWL-ViT:
  OWL-ViT searches for 300+ common object categories including:
  - People, faces, body parts
  - Animals (pets, wildlife, farm animals)
  - Vehicles (cars, bikes, boats, planes)
  - Furniture, electronics, appliances
  - Food, plants, clothing
  - Sports equipment, tools, toys
  - And much more!

  Unlike COCO-based detectors (91 categories), OWL-ViT can find
  many more object types in your images.
        """,
    )
    
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        help="Path(s) to the image(s) to analyze.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.1,
        help="Confidence threshold (default: 0.1). Lower = more detections.",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="text",
        choices=["text", "json", "simple"],
        help="Output format: text (detailed), json, or simple.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save output to file instead of printing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/owlvit-base-patch32",
        help="Model to use (default: google/owlvit-base-patch32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect).",
    )
    
    args = parser.parse_args()
    
    all_outputs = []
    
    for image_path in args.images:
        path = Path(image_path)
        if not path.exists():
            print(f"Error: Image not found: {image_path}")
            continue
        
        print(f"Processing: {image_path} (searching 300+ categories)...", flush=True)
        
        try:
            detected = recognize_objects_from_file(
                image_path=path,
                threshold=args.threshold,
                device=args.device,
                model_name=args.model,
            )
            
            unique_count = len(set(obj.label for obj in detected))
            print(f"Found {len(detected)} detection(s) ({unique_count} unique)")
            
            output = format_output(
                detected_objects=detected,
                output_format=args.format,
                image_path=image_path,
            )
            all_outputs.append(output)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine outputs
    if args.format == "json" and len(all_outputs) > 1:
        combined = "[" + ",\n".join(all_outputs) + "]"
        result = combined
    else:
        result = "\n\n".join(all_outputs)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result)
        print(f"\nOutput saved to: {args.output}")
    else:
        print("\n" + result)


if __name__ == "__main__":
    main()
