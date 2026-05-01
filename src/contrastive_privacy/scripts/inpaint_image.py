#!/usr/bin/env python3
"""
Inpaint specific regions of an image using a text prompt with FLUX.1-schnell.

This script uses mask-based inpainting to edit only specific parts of an image
while preserving the rest. White areas in the mask are edited, black areas
are preserved.

Model: https://huggingface.co/black-forest-labs/FLUX.1-schnell
License: Apache-2.0
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from diffusers import FluxInpaintPipeline
from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def load_mask(mask_path: str | Path) -> Image.Image:
    """
    Load a mask image.
    
    The mask should be:
    - WHITE (255) for areas to EDIT/REPAINT
    - BLACK (0) for areas to PRESERVE
    
    Can be RGB or grayscale; will be converted to single channel.
    """
    mask = Image.open(mask_path)
    # Convert to grayscale if needed
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask


def ensure_same_size(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Ensure image and mask have the same dimensions."""
    if image.size != mask.size:
        print(f"Resizing mask from {mask.size} to match image size {image.size}")
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    
    # Ensure dimensions are multiples of 8
    w, h = image.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    
    if (new_w, new_h) != (w, h):
        print(f"Adjusting dimensions from {w}x{h} to {new_w}x{new_h} (must be multiples of 8)")
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    
    return image, mask


def inpaint_image(
    input_path: str | Path,
    mask_path: str | Path,
    prompt: str,
    output_path: str | Path,
    strength: float = 0.95,
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
    num_inference_steps: int = 28,
    max_sequence_length: int = 512,
    device: Optional[str] = None,
    cpu_offload: bool = True,
) -> Image.Image:
    """
    Inpaint specific regions of an image based on a mask and text prompt.

    Args:
        input_path: Path to the input image.
        mask_path: Path to the mask image. White=edit, Black=preserve.
        prompt: Text description of what to generate in the masked area.
        output_path: Path where the edited image will be saved.
        strength: How much to transform the masked region (0.0-1.0). Default: 0.95.
        guidance_scale: How strongly the model follows the prompt (default: 7.0).
        seed: Random seed for reproducibility. If None, uses random seed.
        num_inference_steps: Number of denoising steps. Default: 28.
        max_sequence_length: Maximum length of the text encoding sequence.
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect).
        cpu_offload: Whether to offload model to CPU to save VRAM.

    Returns:
        The inpainted PIL Image object.
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images
    print(f"Loading input image: {input_path}")
    image = load_image(input_path)
    
    print(f"Loading mask: {mask_path}")
    mask = load_mask(mask_path)
    
    # Ensure same size
    image, mask = ensure_same_size(image, mask)
    print(f"Image size: {image.size}")

    print(f"Loading FLUX.1-schnell inpainting model...")

    # Load the inpainting pipeline
    pipe = FluxInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )

    # Enable CPU offload to save VRAM if requested and on CUDA
    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"Inpainting with prompt: '{prompt}'")
    print(f"Strength: {strength}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Inference steps: {num_inference_steps}")

    # Generate the inpainted image
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=generator,
    )

    inpainted_image = result.images[0]

    # Save the inpainted image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inpainted_image.save(output_path)
    print(f"Inpainted image saved to: {output_path}")

    return inpainted_image


def main() -> None:
    """CLI entry point for inpainting."""
    parser = argparse.ArgumentParser(
        description="Inpaint specific regions of an image using FLUX.1-schnell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png mask.png "A smiling woman with blonde hair" output.png
  %(prog)s photo.jpg face_mask.png "An elderly man with a beard" result.png --strength 0.9
  %(prog)s scene.png sky_mask.png "A dramatic sunset with orange clouds" edited.png --guidance 10

HOW TO CREATE A MASK:
  The mask tells the model what to edit:
  - WHITE (255) = Areas to EDIT/REPAINT (e.g., the face)
  - BLACK (0)   = Areas to PRESERVE (e.g., the sign with text)
  
  You can create masks using:
  - Any image editor (Photoshop, GIMP, Paint, etc.)
  - Python with PIL/OpenCV
  - Online mask editors
  
  Quick mask creation with Python:
    from PIL import Image, ImageDraw
    mask = Image.new('L', (1024, 1024), 0)  # Black background
    draw = ImageDraw.Draw(mask)
    draw.ellipse([300, 200, 700, 600], fill=255)  # White circle for face
    mask.save('mask.png')

TIPS:
  - Make the mask slightly larger than the area you want to edit
  - Use soft/feathered edges for smoother blending
  - Higher strength (0.95+) for more dramatic changes
  - Higher guidance (7-10) for better prompt adherence
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to the input image.",
    )
    parser.add_argument(
        "mask",
        type=str,
        help="Path to the mask image (white=edit, black=preserve).",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing what to generate in the masked area.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path for the inpainted image.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.95,
        help="How much to transform masked region (default: 0.95). Range: 0.0-1.0.",
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=7.0,
        dest="guidance_scale",
        help="Prompt adherence strength (default: 7.0). Higher = follows prompt more strictly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps (default: 28).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for text encoding (default: 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect).",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable CPU offloading (uses more VRAM but may be faster).",
    )

    args = parser.parse_args()

    # Validate strength
    if not 0.0 <= args.strength <= 1.0:
        parser.error("Strength must be between 0.0 and 1.0")

    inpaint_image(
        input_path=args.input,
        mask_path=args.mask,
        prompt=args.prompt,
        output_path=args.output,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_inference_steps=args.steps,
        max_sequence_length=args.max_seq_length,
        device=args.device,
        cpu_offload=not args.no_cpu_offload,
    )


if __name__ == "__main__":
    main()

