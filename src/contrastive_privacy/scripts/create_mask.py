#!/usr/bin/env python3
"""
Create segmentation masks from text descriptions using CLIPSeg.

This script uses CLIPSeg to automatically generate masks for objects or regions
described in natural language (e.g., "face", "person", "car", "sky").
The generated masks can be used with flux-inpaint for targeted image editing.

Model: https://huggingface.co/CIDAS/clipseg-rd64-refined
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def create_mask(
    image_path: str | Path,
    labels: list[str],
    output_path: str | Path,
    threshold: float = 0.5,
    dilate: int = 0,
    blur: int = 0,
    invert: bool = False,
    device: Optional[str] = None,
) -> Image.Image:
    """
    Create a segmentation mask from text descriptions.

    Args:
        image_path: Path to the input image.
        labels: List of text descriptions of objects to segment (e.g., ["face", "hair"]).
        output_path: Path where the mask will be saved.
        threshold: Confidence threshold for segmentation (0.0-1.0). Lower = more inclusive.
        dilate: Number of pixels to expand the mask (useful for including edges).
        blur: Blur radius for softening mask edges.
        invert: If True, invert the mask (white becomes black and vice versa).
        device: Device to run on ('cuda', 'cpu', or None for auto-detect).

    Returns:
        The generated mask as a PIL Image.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    original_size = image.size

    # Load CLIPSeg model
    print("Loading CLIPSeg model...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = model.to(device)

    # Process image with all labels
    print(f"Segmenting: {', '.join(labels)}")
    inputs = processor(
        text=labels,
        images=[image] * len(labels),
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Combine predictions from all labels (take maximum)
    preds = outputs.logits
    preds = torch.sigmoid(preds)  # Convert to probabilities
    
    # Combine all label predictions (union of all masks)
    combined = torch.max(preds, dim=0)[0]
    
    # Apply threshold
    mask_array = (combined > threshold).cpu().numpy().astype(np.uint8) * 255

    # Convert to PIL Image and resize to original dimensions
    mask = Image.fromarray(mask_array, mode='L')
    mask = mask.resize(original_size, Image.Resampling.LANCZOS)

    # Apply dilation if requested
    if dilate > 0:
        from PIL import ImageFilter
        # Dilate by applying maximum filter
        for _ in range(dilate):
            mask = mask.filter(ImageFilter.MaxFilter(3))

    # Apply blur if requested
    if blur > 0:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))

    # Invert if requested
    if invert:
        mask = Image.fromarray(255 - np.array(mask), mode='L')

    # Ensure binary mask after processing
    mask_array = np.array(mask)
    if blur == 0:  # Only binarize if no blur applied
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        mask = Image.fromarray(mask_array, mode='L')

    # Save mask
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    print(f"Mask saved to: {output_path}")

    # Print coverage statistics
    coverage = np.sum(mask_array > 127) / mask_array.size * 100
    print(f"Mask coverage: {coverage:.1f}% of image")

    return mask


def main() -> None:
    """CLI entry point for mask creation."""
    parser = argparse.ArgumentParser(
        description="Create segmentation masks from text descriptions using CLIPSeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a mask for faces
  %(prog)s image.png mask.png --labels "face"
  
  # Create a mask for multiple objects (combined)
  %(prog)s image.png mask.png --labels "face" "hair" "skin"
  
  # Create mask and expand it slightly for better coverage
  %(prog)s image.png mask.png --labels "person" --dilate 10
  
  # Create an inverted mask (mask everything EXCEPT the object)
  %(prog)s image.png mask.png --labels "face" --invert
  
  # Lower threshold for more inclusive mask
  %(prog)s image.png mask.png --labels "text" "sign" --threshold 0.3

COMMON LABELS:
  People:   "face", "person", "human", "body", "hair", "skin", "eyes", "mouth"
  Objects:  "sign", "text", "car", "building", "tree", "chair", "table"
  Regions:  "sky", "background", "foreground", "ground", "wall", "floor"
  
TIPS:
  - Use multiple labels to capture variations (e.g., "face" "human face" "head")
  - Lower --threshold (e.g., 0.3) for more inclusive masks
  - Use --dilate to expand masks and include edges
  - Use --blur for soft/feathered mask edges
  - Use --invert to mask everything EXCEPT the detected objects

WORKFLOW WITH INPAINTING:
  1. Create mask:  flux-create-mask photo.png face_mask.png --labels "face"
  2. Inpaint:      flux-inpaint photo.png face_mask.png "an old man" result.png
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to the input image.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path for the generated mask.",
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        nargs="+",
        required=True,
        help="Text descriptions of objects to segment (e.g., 'face' 'person').",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5). Lower = more inclusive mask.",
    )
    parser.add_argument(
        "--dilate", "-d",
        type=int,
        default=0,
        help="Pixels to expand the mask (default: 0). Helps include edges.",
    )
    parser.add_argument(
        "--blur", "-b",
        type=int,
        default=0,
        help="Blur radius for soft mask edges (default: 0).",
    )
    parser.add_argument(
        "--invert", "-i",
        action="store_true",
        help="Invert mask (white↔black). Use to mask everything EXCEPT the labels.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect).",
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")

    create_mask(
        image_path=args.input,
        labels=args.labels,
        output_path=args.output,
        threshold=args.threshold,
        dilate=args.dilate,
        blur=args.blur,
        invert=args.invert,
        device=args.device,
    )


if __name__ == "__main__":
    main()

