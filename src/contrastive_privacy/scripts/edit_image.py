#!/usr/bin/env python3
"""
Edit an image using a text prompt with the FLUX.1-schnell model.

This script uses Black Forest Labs' FLUX.1-schnell model for image-to-image
generation, allowing you to modify an existing image based on a text prompt.
The strength parameter controls how much the original image influences the result.

Model: https://huggingface.co/black-forest-labs/FLUX.1-schnell
License: Apache-2.0
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image


def load_and_resize_image(
    image_path: str | Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Image.Image:
    """
    Load an image and optionally resize it.

    Args:
        image_path: Path to the input image.
        width: Target width (if None, uses original or matches height aspect ratio).
        height: Target height (if None, uses original or matches width aspect ratio).

    Returns:
        The loaded (and optionally resized) PIL Image.
    """
    image = Image.open(image_path).convert("RGB")

    if width is not None or height is not None:
        original_width, original_height = image.size

        if width is not None and height is not None:
            # Both specified - resize to exact dimensions
            new_size = (width, height)
        elif width is not None:
            # Only width specified - maintain aspect ratio
            aspect_ratio = original_height / original_width
            new_size = (width, int(width * aspect_ratio))
        else:
            # Only height specified - maintain aspect ratio
            aspect_ratio = original_width / original_height
            new_size = (int(height * aspect_ratio), height)

        # Ensure dimensions are multiples of 8 (required by many diffusion models)
        new_size = (new_size[0] // 8 * 8, new_size[1] // 8 * 8)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def edit_image(
    input_path: str | Path,
    prompt: str,
    output_path: str | Path,
    strength: float = 0.95,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
    num_inference_steps: int = 20,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_sequence_length: int = 256,
    device: Optional[str] = None,
    cpu_offload: bool = True,
) -> Image.Image:
    """
    Edit an image based on a text prompt using FLUX.1-schnell.

    Note: FLUX has a unique strength curve different from other diffusion models:
        - 0.0 to 0.90: Almost no visible changes
        - 0.91 to 0.94: Slight modifications begin
        - 0.95 to 0.97: Optimal range for balanced edits (default: 0.95)
        - 0.98 to 1.0: Complete reimagining, may lose coherence

    Args:
        input_path: Path to the input image to edit.
        prompt: Text description of the desired edit/transformation. Use detailed,
                natural language descriptions for best results.
        output_path: Path where the edited image will be saved.
        strength: How much to transform the image (0.0-1.0). FLUX requires
                  high values (0.95-0.97) for visible edits. Default: 0.95.
        guidance_scale: How strongly the model follows the prompt (default: 3.5).
                        Higher values = stricter prompt adherence but less realism.
                        Range 1.5-5.0 recommended. Use 0.0 to disable guidance.
        seed: Random seed for reproducibility. If None, uses random seed.
        num_inference_steps: Number of denoising steps. Higher values give
                             more granular control over strength. Default: 20.
        width: Target width for the output image (None = use input size).
        height: Target height for the output image (None = use input size).
        max_sequence_length: Maximum length of the text encoding sequence.
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect).
        cpu_offload: Whether to offload model to CPU to save VRAM.

    Returns:
        The edited PIL Image object.
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess the input image
    print(f"Loading input image: {input_path}")
    init_image = load_and_resize_image(input_path, width, height)
    print(f"Image size: {init_image.size}")

    print(f"Loading FLUX.1-schnell img2img model...")

    # Load the img2img pipeline with bfloat16 for efficiency
    pipe = FluxImg2ImgPipeline.from_pretrained(
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

    print(f"Editing image with prompt: '{prompt}'")
    print(f"Strength: {strength} (FLUX optimal range: 0.95-0.97)")
    print(f"Guidance scale: {guidance_scale} (higher = stricter prompt following)")
    print(f"Inference steps: {num_inference_steps} (effective: ~{int(num_inference_steps * strength)})")

    # Generate the edited image
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=generator,
    )

    edited_image = result.images[0]

    # Save the edited image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited_image.save(output_path)
    print(f"Edited image saved to: {output_path}")

    return edited_image


def main() -> None:
    """CLI entry point for image editing."""
    parser = argparse.ArgumentParser(
        description="Edit an image using a text prompt with FLUX.1-schnell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.png "A sunset sky with orange and pink clouds behind the scene" output.png
  %(prog)s photo.jpg "Transform into a watercolor painting with soft edges" watercolor.png --strength 0.96
  %(prog)s portrait.png "Add round glasses and a red baseball cap" edited.png --guidance 5.0

PROMPTING TIPS:
  FLUX works best with detailed, natural language descriptions:
  
  ✗ BAD:  "add sunset"
  ✓ GOOD: "A dramatic sunset sky with vibrant orange and pink clouds"
  
  ✗ BAD:  "make it painting"  
  ✓ GOOD: "Transform into an oil painting with visible brushstrokes and rich colors"

FLUX Strength Curve:
  | Strength     | Effect                                    |
  |--------------|-------------------------------------------|
  | 0.00 - 0.90  | Almost NO visible changes                 |
  | 0.91 - 0.94  | Slight modifications begin to appear      |
  | 0.95 - 0.97  | OPTIMAL range for balanced edits          |
  | 0.98 - 1.00  | Complete reimagining, may lose coherence  |

Guidance Scale:
  Controls how strictly the model follows your prompt:
  | Value | Effect                                      |
  |-------|---------------------------------------------|
  | 0.0   | No guidance (ignore prompt weight)          |
  | 1.5   | Subtle prompt influence, more creative      |
  | 3.5   | Balanced (default)                          |
  | 5.0+  | Strong prompt adherence, may reduce realism |
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to the input image to edit.",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the desired edit.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path for the edited image (e.g., edited.png).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.95,
        help="Transformation strength 0.0-1.0 (default: 0.95). FLUX needs 0.91+ for visible changes. Optimal: 0.95-0.97.",
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=3.5,
        dest="guidance_scale",
        help="Prompt adherence strength (default: 3.5). Higher = follows prompt more strictly. Range: 0.0-7.0.",
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
        default=20,
        help="Number of inference steps (default: 20). More steps = finer strength control.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target width (default: use input image width).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target height (default: use input image height).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Maximum sequence length for text encoding (default: 256).",
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

    edit_image(
        input_path=args.input,
        prompt=args.prompt,
        output_path=args.output,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        max_sequence_length=args.max_seq_length,
        device=args.device,
        cpu_offload=not args.no_cpu_offload,
    )


if __name__ == "__main__":
    main()

