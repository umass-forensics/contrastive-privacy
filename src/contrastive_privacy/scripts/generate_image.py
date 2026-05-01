#!/usr/bin/env python3
"""
Generate an image from a text prompt using the FLUX.1-schnell model.

This script uses Black Forest Labs' FLUX.1-schnell model, a 12 billion parameter
rectified flow transformer capable of generating high-quality images from text
descriptions in just 1-4 inference steps.

Model: https://huggingface.co/black-forest-labs/FLUX.1-schnell
License: Apache-2.0
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from diffusers import FluxPipeline
from PIL import Image


def generate_image(
    prompt: str,
    output_path: str | Path,
    seed: Optional[int] = None,
    num_inference_steps: int = 4,
    width: int = 1024,
    height: int = 1024,
    max_sequence_length: int = 256,
    device: Optional[str] = None,
    cpu_offload: bool = True,
) -> Image.Image:
    """
    Generate an image from a text prompt using FLUX.1-schnell.

    Args:
        prompt: The text description of the image to generate.
        output_path: Path where the generated image will be saved.
        seed: Random seed for reproducibility. If None, uses random seed.
        num_inference_steps: Number of denoising steps (1-4 recommended for schnell).
        width: Width of the generated image in pixels.
        height: Height of the generated image in pixels.
        max_sequence_length: Maximum length of the text encoding sequence.
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect).
        cpu_offload: Whether to offload model to CPU to save VRAM.

    Returns:
        The generated PIL Image object.
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading FLUX.1-schnell model...")
    
    # Load the pipeline with bfloat16 for efficiency
    pipe = FluxPipeline.from_pretrained(
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

    print(f"Generating image for prompt: '{prompt}'")
    
    # Generate the image
    # Note: guidance_scale=0.0 is recommended for schnell model
    result = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        max_sequence_length=max_sequence_length,
        generator=generator,
    )

    image = result.images[0]

    # Save the image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")

    return image


def main() -> None:
    """CLI entry point for image generation."""
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using FLUX.1-schnell.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "A cat holding a sign that says hello world" output.png
  %(prog)s "A serene mountain landscape at sunset" landscape.png --seed 42
  %(prog)s "Abstract art with vibrant colors" art.png --steps 4 --width 1024 --height 768
        """,
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="The text prompt describing the image to generate.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output path for the generated image (e.g., output.png).",
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
        default=4,
        help="Number of inference steps (default: 4, recommended: 1-4).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the generated image in pixels (default: 1024).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the generated image in pixels (default: 1024).",
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

    generate_image(
        prompt=args.prompt,
        output_path=args.output,
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

