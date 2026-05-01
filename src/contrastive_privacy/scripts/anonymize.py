#!/usr/bin/env python3
"""
Anonymize or replace specific objects in images automatically.

This script combines semantic segmentation with inpainting (FLUX)
to automatically detect and replace specified objects in images.
Perfect for privacy applications like face anonymization.

Supports two segmentation backends:
- GroundingDINO + SAM (default): State-of-the-art quality, better boundaries
- CLIPSeg: Faster, lower memory usage

Example: anonymize faces in a photo by replacing them with different faces.
"""

import argparse
import base64
import io
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from openai import OpenAI

# fal.ai OpenRouter Vision endpoint for VLM polygon/bbox detection (same as identify_obfuscation_concepts)
OPENROUTER_VISION_ENDPOINT = "openrouter/router/vision"
OPENROUTER_VISION_MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "opus-4.6": "anthropic/claude-opus-4.6",
}
DEFAULT_FAL_VISION_MODEL = "gpt-5.4"

# OpenRouter sampling temperature for fal.ai vision (openrouter/router/vision)
DEFAULT_FAL_VISION_TEMPERATURE = 0.1

# fal.ai image edit endpoint IDs for ai-gen segmenter (short name -> full endpoint)
FAL_IMAGE_MODEL_ENDPOINTS = {
    "gpt-image-1.5": "fal-ai/gpt-image-1.5/edit",
    "gpt-image-1-mini": "fal-ai/gpt-image-1-mini/edit",
    "gpt-image-1": "fal-ai/gpt-image-1/edit-image",
    "nano-banana-pro": "fal-ai/nano-banana-pro/edit",
    "flux-2-pro": "fal-ai/flux-2-pro/edit",
    "flux-2-dev": "fal-ai/flux-2/edit",  # Cheapest option (~$0.012/MP) for testing
    "flux-2": "fal-ai/flux-2/edit",
    "flux-2-max": "fal-ai/flux-2-max/edit",
    "gemini-25-flash-image": "fal-ai/gemini-25-flash-image/edit",
    "gemini-3.1-flash-image-preview": "fal-ai/gemini-3.1-flash-image-preview/edit",
    "gemini-3-pro-image-preview": "fal-ai/gemini-3-pro-image-preview/edit",
}
# When user passes a 2-segment path (e.g. fal-ai/gpt-image-1), which operation to append.
# Only gpt-image-1 uses edit-image; all others use edit.
FAL_IMAGE_EDIT_OPERATION = {
    "fal-ai/gpt-image-1-mini": "edit",
    "fal-ai/gpt-image-1": "edit-image",
    "fal-ai/gpt-image-1.5": "edit",
    "fal-ai/gemini-25-flash-image": "edit",
    "fal-ai/gemini-3.1-flash-image-preview": "edit",
    "fal-ai/gemini-3-pro-image-preview": "edit",
    "fal-ai/flux-2": "edit",
    "fal-ai/flux-2-pro": "edit",
    "fal-ai/flux-2-max": "edit",
}
DEFAULT_FAL_EDIT_OPERATION = "edit"

# fal.ai upload limit (25MB); use slightly less for safety
MAX_FAL_UPLOAD_BYTES = 24_000_000


@dataclass
class CLIPSegModels:
    """Container for CLIPSeg model and processor."""
    model: CLIPSegForImageSegmentation
    processor: CLIPSegProcessor
    device: str


@dataclass
class GroundedSAMModels:
    """Container for GroundingDINO and SAM models."""
    dino_model: Any  # AutoModelForZeroShotObjectDetection
    dino_processor: Any  # AutoProcessor
    sam_model: Any  # SamModel
    sam_processor: Any  # SamProcessor
    device: str


@dataclass
class SAM3Models:
    """Container for SAM3 model and processor."""
    model: Any  # Sam3Model
    processor: Any  # Sam3Processor
    device: str


def load_clipseg_models(
    device: str = "cuda",
) -> CLIPSegModels:
    """
    Load CLIPSeg model and processor.
    
    Args:
        device: Device to run on.
    
    Returns:
        CLIPSegModels container with model, processor, and device.
    """
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = model.to(device)
    
    return CLIPSegModels(model=model, processor=processor, device=device)


def load_groundedsam_models(
    device: str = "cuda",
    sam_model_name: str = "facebook/sam-vit-huge",
    dino_model_name: str = "IDEA-Research/grounding-dino-base",
) -> GroundedSAMModels:
    """
    Load GroundingDINO and SAM models.
    
    Args:
        device: Device to run on.
        sam_model_name: SAM model variant (base/large/huge).
        dino_model_name: GroundingDINO model variant.
    
    Returns:
        GroundedSAMModels container with all models and processors.
    """
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        SamModel,
        SamProcessor,
    )
    
    # Load GroundingDINO
    dino_processor = AutoProcessor.from_pretrained(dino_model_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_name)
    dino_model = dino_model.to(device)
    
    # Load SAM
    sam_processor = SamProcessor.from_pretrained(sam_model_name)
    sam_model = SamModel.from_pretrained(sam_model_name)
    sam_model = sam_model.to(device)
    
    return GroundedSAMModels(
        dino_model=dino_model,
        dino_processor=dino_processor,
        sam_model=sam_model,
        sam_processor=sam_processor,
        device=device,
    )


def load_sam3_models(
    device: str = "cuda",
    model_name: str = "facebook/sam3",
    use_half_precision: bool = True,
) -> SAM3Models:
    """
    Load SAM3 model and processor.
    
    SAM3 is a unified foundation model for promptable segmentation that can
    segment objects using text prompts directly, without needing a separate
    detection model like GroundingDINO.
    
    Args:
        device: Device to run on.
        model_name: SAM3 model name on HuggingFace.
        use_half_precision: If True, load model in bfloat16 to reduce memory usage.
            Reduces VRAM from ~3.4GB to ~1.7GB with minimal quality impact.
    
    Returns:
        SAM3Models container with model, processor, and device.
    """
    from transformers import Sam3Model, Sam3Processor
    
    print(f"  Loading SAM3 model: {model_name}")
    processor = Sam3Processor.from_pretrained(model_name)
    
    if use_half_precision and device == "cuda":
        model = Sam3Model.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        model = Sam3Model.from_pretrained(model_name)
    
    model = model.to(device)
    
    return SAM3Models(model=model, processor=processor, device=device)


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def redact_with_blur(
    image: Image.Image,
    mask: Image.Image,
    blur_radius: int = 30,
) -> Image.Image:
    """
    Redact masked regions by applying gaussian blur.
    
    Args:
        image: Input image
        mask: Mask image (white = regions to blur)
        blur_radius: Radius of gaussian blur (higher = more blurred)
    
    Returns:
        Image with masked regions blurred
    """
    from PIL import ImageFilter
    
    # Ensure mask is same size as image
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    
    # Create heavily blurred version of the image
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Convert mask to proper mode for compositing
    # Mask should be 'L' mode (grayscale) where white = blurred, black = original
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # Composite: use mask to blend original and blurred
    # Where mask is white (255), use blurred; where black (0), use original
    result = Image.composite(blurred, image, mask)
    
    return result


def blackout_with_black(
    image: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Blackout masked regions by replacing them with black pixels.
    
    Args:
        image: Input image
        mask: Mask image (white = regions to blackout)
    
    Returns:
        Image with masked regions replaced by black
    """
    # Ensure mask is same size as image
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    
    # Create a black image of the same size
    black = Image.new('RGB', image.size, (0, 0, 0))
    
    # Convert mask to proper mode for compositing
    # Mask should be 'L' mode (grayscale) where white = black, black = original
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    # Composite: use mask to blend original and black
    # Where mask is white (255), use black; where black (0), use original
    result = Image.composite(black, image, mask)
    
    return result


def get_polygons_from_fal_vision(
    image: Image.Image,
    target_labels: list[str],
    fal_key: Optional[str] = None,
    model: str = DEFAULT_FAL_VISION_MODEL,
    previous_polygons: Optional[list[dict]] = None,
    temperature: float = DEFAULT_FAL_VISION_TEMPERATURE,
) -> list[dict]:
    """
    Use fal.ai OpenRouter Vision API to get bounding polygon coordinates for specified objects.
    
    Args:
        image: Input PIL image.
        target_labels: List of objects to detect (e.g., ["face", "person"]).
        fal_key: fal.ai API key. If None, uses FAL_KEY environment variable.
        model: User-facing vision model name: gpt-5.4, gemini-3.1-pro, or opus-4.6.
        previous_polygons: Optional list of polygons from a previous pass for refinement.
            If provided, the prompt will ask the vision model to refine these polygons.
        temperature: OpenRouter sampling temperature for this vision call.
    
    Returns:
        List of polygon dictionaries, each containing:
            - "label": The detected object label
            - "polygon": List of [x, y] coordinate pairs defining the polygon vertices
              (coordinates are in pixels, relative to image dimensions)
    """
    import fal_client

    if model not in OPENROUTER_VISION_MODELS:
        raise ValueError(
            f"Unknown vision model: {model}. Choose from: {list(OPENROUTER_VISION_MODELS.keys())}"
        )
    if fal_key is None:
        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            raise ValueError(
                "fal.ai API key not provided. Set FAL_KEY environment variable "
                "or pass fal_key parameter."
            )

    openrouter_model = OPENROUTER_VISION_MODELS[model]
    width, height = image.size

    # If we have previous polygons, create visualization for refinement
    if previous_polygons:
        from PIL import ImageDraw
        
        # Create the blackout mask from previous polygons to show what's being covered
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        for poly_data in previous_polygons:
            polygon = poly_data.get("polygon", [])
            if len(polygon) >= 3:
                coords = [(int(p[0]), int(p[1])) for p in polygon]
                mask_draw.polygon(coords, fill=255)
        
        # Create image showing the original with RED filled polygons (not black)
        # This way GPT-5.2 can see both the original content AND what we're trying to cover
        overlay_image = image.copy()
        
        # Create a red overlay for each polygon
        for i, poly_data in enumerate(previous_polygons):
            polygon = poly_data.get("polygon", [])
            if len(polygon) >= 3:
                coords = [(int(p[0]), int(p[1])) for p in polygon]
                
                # Draw a semi-transparent red fill so we can see what's underneath
                red_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
                red_draw = ImageDraw.Draw(red_layer)
                red_draw.polygon(coords, fill=(255, 0, 0, 180))  # Red with alpha
                
                # Also draw a thick white border so the polygon boundary is clear
                for thickness in range(3):
                    for dx in range(-thickness, thickness + 1):
                        for dy in range(-thickness, thickness + 1):
                            if abs(dx) == thickness or abs(dy) == thickness:
                                offset_coords = [(x + dx, y + dy) for x, y in coords]
                                red_draw.polygon(offset_coords, outline=(255, 255, 255, 255))
                
                overlay_image = Image.alpha_composite(overlay_image.convert('RGBA'), red_layer).convert('RGB')
                
                # Add number label
                overlay_draw = ImageDraw.Draw(overlay_image)
                if coords:
                    # Calculate center of polygon for label placement
                    center_x = sum(c[0] for c in coords) // len(coords)
                    center_y = sum(c[1] for c in coords) // len(coords)
                    label_text = f"{i+1}"
                    # Draw label with background
                    overlay_draw.rectangle([center_x-12, center_y-12, center_x+12, center_y+12], fill=(0, 0, 0))
                    overlay_draw.text((center_x-5, center_y-8), label_text, fill=(255, 255, 0))
        
        # Build refinement prompt - simpler and more direct
        objects_list = ", ".join(target_labels)
        previous_json = json.dumps(previous_polygons, indent=2)
        prompt = f"""This image shows my attempt to locate all "{objects_list}" objects.

The RED SHADED AREAS with WHITE BORDERS show the regions I'm currently detecting as "{objects_list}".
Each region is numbered (1, 2, etc.).

Current polygon coordinates for an image of size {width}x{height} pixels:
{previous_json}

Please look carefully at the RED regions and answer:
1. Does each RED region correctly cover a "{objects_list}"? 
2. Is any "{objects_list}" in the image NOT covered by a RED region?
3. Does any RED region need to be adjusted to better fit the actual "{objects_list}" boundary?

Based on your analysis, provide CORRECTED coordinates. For each polygon:
- If it's correctly placed, keep similar coordinates
- If it needs to be larger/smaller/moved, adjust the coordinates
- If it's a false detection, remove it
- If a "{objects_list}" was missed, add a new polygon

Return the corrected polygons as a JSON array:
[
  {{"label": "{objects_list}", "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}},
  ...
]

Coordinates are pixels where (0,0) is top-left. Image size: {width}x{height}.

IMPORTANT: Return ONLY the JSON array, nothing else."""
    else:
        overlay_image = image
        # Build the prompt for initial polygon detection
        objects_list = ", ".join(target_labels)
        prompt = f"""Analyze this image and identify all instances of the following objects: {objects_list}

For each detected object, provide a bounding polygon as a list of [x, y] coordinate pairs that outline the object.
The coordinates should be in pixels, where (0, 0) is the top-left corner of the image.
The image dimensions are {width}x{height} pixels (width x height).

Return your response as a JSON array with the following structure:
[
  {{
    "label": "object_type",
    "polygon": [[x1, y1], [x2, y2], [x3, y3], ...]
  }},
  ...
]

Provide at least 4 points per polygon to accurately outline each object.
If no objects are found, return an empty array: []

IMPORTANT: Return ONLY the JSON array, no other text or explanation."""

    # Save image to temp file for fal upload (refinement uses overlay_image)
    img_to_upload = overlay_image if previous_polygons else image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_to_upload.save(tmp.name)
        try:
            image_url = fal_client.upload_file(tmp.name)
        finally:
            os.unlink(tmp.name)

    arguments = {
        "image_urls": [image_url],
        "prompt": prompt,
        "model": openrouter_model,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    lower_model = openrouter_model.lower()
    if (
        "gemini-2" in lower_model
        or "gemini-3" in lower_model
        or "gpt-5.4-pro" in lower_model
    ):
        arguments["reasoning"] = True

    try:
        result = fal_client.subscribe(
            OPENROUTER_VISION_ENDPOINT,
            arguments=arguments,
        )
        response_text = (result.get("output") or "").strip()

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        polygons = json.loads(response_text)

        # Validate and normalize the response
        validated_polygons = []
        for item in polygons:
            if isinstance(item, dict) and "label" in item and "polygon" in item:
                polygon_coords = item["polygon"]
                if isinstance(polygon_coords, list) and len(polygon_coords) >= 3:
                    valid_coords = []
                    for coord in polygon_coords:
                        if isinstance(coord, (list, tuple)) and len(coord) == 2:
                            x = max(0, min(width, coord[0]))
                            y = max(0, min(height, coord[1]))
                            valid_coords.append([x, y])

                    if len(valid_coords) >= 3:
                        validated_polygons.append({
                            "label": item["label"],
                            "polygon": valid_coords,
                        })

        return validated_polygons

    except json.JSONDecodeError as e:
        print(f"  WARNING: Failed to parse fal vision response as JSON: {e}")
        print(f"  Response was: {response_text[:500]}...")
        return previous_polygons if previous_polygons else []
    except Exception as e:
        print(f"  WARNING: Error calling fal.ai vision API: {e}")
        return previous_polygons if previous_polygons else []


def get_polygons_from_fal_vision_with_refinements(
    image: Image.Image,
    target_labels: list[str],
    refinements: int = 0,
    fal_key: Optional[str] = None,
    model: str = DEFAULT_FAL_VISION_MODEL,
    temperature: float = DEFAULT_FAL_VISION_TEMPERATURE,
) -> list[dict]:
    """
    Use fal.ai OpenRouter vision to get bounding polygon coordinates with ensemble selection.
    
    Approach:
    1. Run multiple independent detections to gather candidate boxes
    2. Draw all candidates on one image with numbers
    3. Ask the vision model to select the best boxes that correctly cover the target objects
    
    Args:
        image: Input PIL image.
        target_labels: List of objects to detect (e.g., ["face", "person"]).
        refinements: Number of additional detection passes (0 = single detection only).
        fal_key: fal.ai API key. If None, uses FAL_KEY environment variable.
        model: User-facing vision model name (gpt-5.4, gemini-3.1-pro, opus-4.6).
        temperature: OpenRouter sampling temperature for each vision call.
    
    Returns:
        List of polygon dictionaries after selection.
    """
    all_polygons = []
    num_passes = refinements + 1

    for i in range(num_passes):
        print(f"    Pass {i + 1}/{num_passes}: Gathering candidates")
        polygons = get_polygons_from_fal_vision(
            image=image,
            target_labels=target_labels,
            fal_key=fal_key,
            model=model,
            previous_polygons=None,
            temperature=temperature,
        )

        if polygons:
            print(f"      Found {len(polygons)} candidates")
            all_polygons.extend(polygons)
        else:
            print(f"      No objects detected")

    if not all_polygons:
        print(f"    No candidates found in any pass")
        return []

    print(f"    Total candidates: {len(all_polygons)}")

    if num_passes == 1 or len(all_polygons) <= 3:
        return all_polygons

    print(f"    Asking vision model to select best boxes from {len(all_polygons)} candidates...")
    selected = select_best_polygons_with_fal_vision(
        image=image,
        candidate_polygons=all_polygons,
        target_labels=target_labels,
        fal_key=fal_key,
        model=model,
        temperature=temperature,
    )

    print(f"    Selected {len(selected)} boxes")
    return selected


def select_best_polygons_with_fal_vision(
    image: Image.Image,
    candidate_polygons: list[dict],
    target_labels: list[str],
    fal_key: Optional[str] = None,
    model: str = DEFAULT_FAL_VISION_MODEL,
    temperature: float = DEFAULT_FAL_VISION_TEMPERATURE,
) -> list[dict]:
    """
    Show the fal.ai vision model all candidate boxes and ask it to select the best ones.
    
    Args:
        image: Original PIL image.
        candidate_polygons: List of all candidate polygon dictionaries.
        target_labels: List of objects we're trying to detect.
        fal_key: fal.ai API key. If None, uses FAL_KEY environment variable.
        model: User-facing vision model name (gpt-5.4, gemini-3.1-pro, opus-4.6).
        temperature: OpenRouter sampling temperature for this vision call.
    
    Returns:
        List of selected polygon dictionaries.
    """
    import re

    import fal_client

    from PIL import ImageDraw

    if fal_key is None:
        fal_key = os.getenv("FAL_KEY")
    if model not in OPENROUTER_VISION_MODELS:
        model = DEFAULT_FAL_VISION_MODEL
    openrouter_model = OPENROUTER_VISION_MODELS[model]
    
    # Create image with all candidates drawn and numbered
    annotated = image.copy().convert('RGB')
    draw = ImageDraw.Draw(annotated)
    
    # Colors for different boxes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    ]
    
    # Draw each candidate with a number
    for i, poly_data in enumerate(candidate_polygons):
        polygon = poly_data.get("polygon", [])
        if len(polygon) >= 3:
            coords = [(int(p[0]), int(p[1])) for p in polygon]
            color = colors[i % len(colors)]
            
            # Draw thick polygon outline
            for thickness in range(4):
                for dx in range(-thickness, thickness + 1):
                    for dy in range(-thickness, thickness + 1):
                        if abs(dx) == thickness or abs(dy) == thickness:
                            offset_coords = [(x + dx, y + dy) for x, y in coords]
                            draw.polygon(offset_coords, outline=color)
            
            # Draw number label at center of polygon
            center_x = sum(c[0] for c in coords) // len(coords)
            center_y = sum(c[1] for c in coords) // len(coords)
            
            # Draw label background
            label_text = str(i + 1)
            bbox = [center_x - 15, center_y - 15, center_x + 15, center_y + 15]
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.rectangle(bbox, outline=color, width=2)
            draw.text((center_x - 8, center_y - 10), label_text, fill=(255, 255, 255))
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        annotated.save(tmp.name)
        try:
            image_url = fal_client.upload_file(tmp.name)
        finally:
            os.unlink(tmp.name)
    
    # Build candidate list for prompt
    candidate_list = []
    for i, poly_data in enumerate(candidate_polygons):
        coords = poly_data.get("polygon", [])
        # Get bounding box for description
        if coords:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            candidate_list.append(f"  Box {i+1}: [{min(xs):.0f}, {min(ys):.0f}] to [{max(xs):.0f}, {max(ys):.0f}]")
    
    objects_list = ", ".join(target_labels)
    candidates_text = "\n".join(candidate_list)
    
    prompt = f"""This image shows {len(candidate_polygons)} numbered candidate boxes. I need to identify ALL boxes that cover a "{objects_list}".

Each box is drawn with a colored outline and has a number in its center.

Candidate boxes:
{candidates_text}

IMPORTANT: There may be MULTIPLE "{objects_list}" objects in this image. I need to cover ALL of them.

Your task:
1. Look at EVERY numbered box
2. For EACH box, determine if it covers (even partially) a "{objects_list}"
3. KEEP a box if:
   - It covers a "{objects_list}" (even if not perfectly aligned)
   - It would help hide/obfuscate a "{objects_list}"
4. Only REJECT a box if:
   - It clearly does NOT contain any part of a "{objects_list}"
   - It is an exact duplicate of another box covering the same object (in which case, keep the better-fitting one)

Remember: It's better to include extra boxes than to miss a "{objects_list}". When in doubt, KEEP the box.

How many distinct "{objects_list}" objects do you see in the image? List the box numbers that cover each one.

Then return a JSON array of ALL box numbers to keep, like: [1, 2, 3, 5, 7]

IMPORTANT: Return ONLY the JSON array of numbers as your final answer."""

    try:
        arguments = {
            "image_urls": [image_url],
            "prompt": prompt,
            "model": openrouter_model,
            "temperature": temperature,
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
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines: list[str] = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)
        print(f"    Vision selection response: {response_text}")

        match = re.search(r"\[[\d,\s]*\]", response_text)
        if match:
            selected_indices = json.loads(match.group())
            selected_polygons = []
            for idx in selected_indices:
                if isinstance(idx, int) and 1 <= idx <= len(candidate_polygons):
                    selected_polygons.append(candidate_polygons[idx - 1])
            return selected_polygons
        print("    WARNING: Could not parse selection, returning all candidates")
        return candidate_polygons

    except Exception as e:
        print(f"    WARNING: Error in selection: {e}")
        return candidate_polygons


def create_mask_from_polygons(
    polygons: list[dict],
    image_size: tuple[int, int],
    dilate: int = 0,
    blur: int = 0,
) -> Image.Image:
    """
    Create a segmentation mask from polygon coordinates.
    
    Args:
        polygons: List of polygon dictionaries from get_polygons_from_gpt52().
        image_size: Tuple of (width, height) for the output mask.
        dilate: Pixels to expand the mask (optional).
        blur: Blur radius for soft edges (optional).
    
    Returns:
        Grayscale mask image (white = detected regions).
    """
    from PIL import ImageDraw, ImageFilter
    
    width, height = image_size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for poly_data in polygons:
        polygon = poly_data.get("polygon", [])
        if len(polygon) >= 3:
            # Convert to list of tuples for PIL
            coords = [(int(p[0]), int(p[1])) for p in polygon]
            draw.polygon(coords, fill=255)
    
    # Apply dilation if requested
    if dilate > 0:
        for _ in range(dilate):
            mask = mask.filter(ImageFilter.MaxFilter(3))
    
    # Apply blur for soft edges if requested
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    
    return mask


def anonymize_with_gpt52_polygons(
    image: Image.Image,
    target_labels: list[str],
    fal_key: Optional[str] = None,
    fal_vision_model: str = DEFAULT_FAL_VISION_MODEL,
    dilate: int = 5,
    blur: int = 8,
    refinements: int = 0,
    fal_vision_temperature: float = DEFAULT_FAL_VISION_TEMPERATURE,
) -> tuple[Image.Image, float, list[dict]]:
    """
    Anonymize objects in an image using fal.ai OpenRouter vision for polygon detection and blackout.
    
    This function:
    1. Uses fal.ai OpenRouter vision (e.g. gpt-5.4, gemini-3.1-pro, opus-4.6) to detect bounding polygons
    2. Optionally refines the polygons through additional passes
    3. Creates a mask from the polygons
    4. Applies blackout (black pixels) to the masked regions
    
    Args:
        image: Input image to anonymize.
        target_labels: List of objects to detect and black out.
        fal_key: fal.ai API key. If None, uses FAL_KEY environment variable.
        fal_vision_model: Vision model name (gpt-5.4, gemini-3.1-pro, opus-4.6).
        fal_vision_temperature: OpenRouter sampling temperature for vision API calls.
        dilate: Pixels to expand the mask (default: 5).
        blur: Blur radius for mask edges (default: 8).
        refinements: Number of refinement passes for polygon detection (default: 0).
            Each refinement pass shows the vision model its previous polygons and asks it to
            improve the boundaries.
    
    Returns:
        Tuple of:
            - Anonymized PIL Image
            - Coverage percentage (fraction of image that was masked)
            - List of detected polygons
    """
    print(f"  Using fal.ai vision ({fal_vision_model}) to detect polygons for: {', '.join(target_labels)}")
    if refinements > 0:
        print(f"  Refinement passes: {refinements}")
    
    polygons = get_polygons_from_fal_vision_with_refinements(
        image=image,
        target_labels=target_labels,
        refinements=refinements,
        fal_key=fal_key,
        model=fal_vision_model,
        temperature=fal_vision_temperature,
    )
    
    print(f"  Vision model detected {len(polygons)} objects (final)")
    for poly in polygons:
        print(f"    - {poly['label']}: {len(poly['polygon'])} vertices")
    
    if not polygons:
        print("  WARNING: No objects detected by vision model")
        # Return original image with 0 coverage
        return image.copy(), 0.0, []
    
    # Create mask from polygons
    mask = create_mask_from_polygons(
        polygons=polygons,
        image_size=image.size,
        dilate=dilate,
        blur=blur,
    )
    
    # Calculate coverage
    mask_array = np.array(mask)
    coverage = np.sum(mask_array > 127) / mask_array.size * 100
    
    # Apply blackout
    result = blackout_with_black(image, mask)
    
    return result, coverage, polygons


def create_segmentation_mask_clipseg(
    image: Image.Image,
    labels: list[str],
    threshold: float | list[float] = 0.4,
    dilate: int = 5,
    blur: int = 8,
    device: str = "cuda",
    models: Optional[CLIPSegModels] = None,
    adaptive_blur: bool = False,
    blur_scale: float = 1.0,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
    convex_hull: bool = False,
) -> Image.Image:
    """Create a segmentation mask using CLIPSeg.
    
    Args:
        image: Input image
        labels: Text descriptions of objects to segment
        threshold: Confidence threshold (lower = more inclusive).
            Can be a single float (applied to all labels) or a list of floats
            (one per label). When a list is provided, each label's prediction
            is thresholded independently before combining.
        dilate: Pixels to expand mask (ignored if adaptive_blur=True)
        blur: Blur radius for soft edges (ignored if adaptive_blur=True)
        device: Device to run on
        models: Pre-loaded CLIPSeg models (optional, for reuse across calls)
        adaptive_blur: If True, scale blur/dilation based on overall mask size.
            For CLIPSeg, this is applied to the combined mask since individual
            objects are not tracked separately.
        blur_scale: When adaptive_blur=True, controls overall blur intensity.
        size_exponent: Controls size-dependence of blur (0.0=same for all, 1.0=linear).
        scaling_factor: Constant multiplier on effective size (default 1.0).
        convex_hull: If True, expand each connected component to its convex hull.
    """
    original_size = image.size
    
    # Normalize threshold: convert to list if needed
    if isinstance(threshold, (int, float)):
        threshold_list = [float(threshold)] * len(labels)
    else:
        threshold_list = list(threshold)
        if len(threshold_list) != len(labels):
            raise ValueError(
                f"When threshold is a list, it must have the same length as labels. "
                f"Got {len(threshold_list)} thresholds for {len(labels)} labels."
            )
    
    # Track whether we loaded models ourselves (for cleanup)
    loaded_locally = models is None

    # Load CLIPSeg model if not provided
    if models is None:
        models = load_clipseg_models(device)
    
    processor = models.processor
    model = models.model

    # Process image with all labels
    inputs = processor(
        text=labels,
        images=[image] * len(labels),
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(models.device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions for each label (shape: [num_labels, H, W])
    preds = torch.sigmoid(outputs.logits)
    
    # Apply per-label thresholds and combine with OR operation
    # This allows different sensitivity for each object type
    combined_mask = torch.zeros_like(preds[0], dtype=torch.bool)
    for i, thresh in enumerate(threshold_list):
        label_mask = preds[i] > thresh
        combined_mask = combined_mask | label_mask
    
    mask_array = combined_mask.cpu().numpy().astype(np.uint8) * 255

    # Convert to PIL and resize
    mask = Image.fromarray(mask_array, mode='L')
    mask = mask.resize(original_size, Image.Resampling.LANCZOS)

    # Clean up only if we loaded models ourselves
    if loaded_locally:
        del model, processor
        torch.cuda.empty_cache() if device == "cuda" else None

    # Apply convex hull if enabled (before blur/dilation)
    if convex_hull:
        mask_array = apply_convex_hull(np.array(mask))
        mask = Image.fromarray(mask_array, mode='L')

    # Apply blur and dilation - either adaptive or fixed
    if adaptive_blur:
        # Apply size-adaptive blur based on the combined mask's characteristic size
        mask = apply_size_adaptive_effects(
            mask, blur_scale=blur_scale, size_exponent=size_exponent, scaling_factor=scaling_factor
        )
    else:
        # Dilate mask to include edges
        if dilate > 0:
            from PIL import ImageFilter
            for _ in range(dilate):
                mask = mask.filter(ImageFilter.MaxFilter(3))

        # Apply blur for soft edges (critical for natural blending!)
        if blur > 0:
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))

    return mask


def compute_mask_characteristic_size(mask_array: np.ndarray) -> float:
    """
    Compute the characteristic size of a mask region.
    
    The characteristic size is the square root of the mask's area in pixels,
    which gives a linear measurement that scales well for blur calculations.
    
    Args:
        mask_array: Binary mask array (non-zero = mask region)
    
    Returns:
        Characteristic size (sqrt of area in pixels), or 0 if mask is empty.
    """
    area = np.sum(mask_array > 0)
    return np.sqrt(area) if area > 0 else 0.0


def apply_size_adaptive_effects(
    mask: Image.Image,
    blur_scale: float,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
) -> Image.Image:
    """
    Apply dilation and blur to a mask, scaled by the mask's characteristic size.
    
    Formula:
        effective_size = scaling_factor × char_size^exponent
        dilation_radius = blur_scale × effective_size
    
    With defaults (scaling_factor=1, exponent=1):
        dilation_radius = blur_scale × char_size
    
    Args:
        mask: Grayscale mask image (white = detected regions)
        blur_scale: Multiplier for dilation radius.
        size_exponent: Power to raise char_size to (default 1.0 = linear).
        scaling_factor: Constant multiplier on effective size (default 1.0).
    
    Returns:
        Mask with size-adaptive dilation and blur applied.
    """
    from scipy import ndimage
    from PIL import ImageFilter
    
    mask_array = np.array(mask)
    char_size = compute_mask_characteristic_size(mask_array)
    
    if char_size == 0:
        return mask
    
    # Simple formula: effective_size = scaling_factor × char_size^exponent
    effective_size = scaling_factor * (char_size ** size_exponent)
    
    # dilation_radius = blur_scale × effective_size
    dilate_radius = int(blur_scale * effective_size)
    
    # Edge blur is proportional to dilation (soften the edges)
    blur_radius = max(1, dilate_radius // 4)
    
    # Ensure minimum values
    dilate_radius = max(0, dilate_radius)
    
    # Apply dilation using scipy's maximum_filter (single O(1) operation)
    if dilate_radius > 0:
        kernel_size = 2 * dilate_radius + 1
        mask_array = ndimage.maximum_filter(mask_array, size=kernel_size)
    
    # Convert back to PIL and apply blur for soft edges
    result = Image.fromarray(mask_array, mode='L')
    if blur_radius > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return result


def apply_convex_hull(mask_array: np.ndarray) -> np.ndarray:
    """
    Expand a binary mask to its convex hull.
    
    This is useful for hiding object silhouettes by filling in concave regions.
    For example, a person with arms akimbo would have the space between their
    arms and body filled in.
    
    Args:
        mask_array: Binary mask array (non-zero = mask region)
    
    Returns:
        Binary mask array with each connected component replaced by its convex hull.
    """
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    
    # Handle empty mask
    if np.sum(mask_array > 0) == 0:
        return mask_array
    
    # Label connected components
    binary_mask = (mask_array > 127).astype(np.uint8)
    labeled, num_features = ndimage.label(binary_mask)
    
    # Create output mask
    result = np.zeros_like(mask_array)
    
    for i in range(1, num_features + 1):
        # Get coordinates of this component
        component_mask = (labeled == i)
        coords = np.argwhere(component_mask)
        
        if len(coords) < 3:
            # Need at least 3 points for convex hull, just copy original
            result[component_mask] = 255
            continue
        
        try:
            # Compute convex hull
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            
            # Fill the convex hull polygon
            # Create a polygon mask using the hull vertices
            from PIL import Image, ImageDraw
            h, w = mask_array.shape
            hull_mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(hull_mask)
            
            # Convert to (x, y) format for PIL (note: coords are (row, col) = (y, x))
            polygon = [(int(p[1]), int(p[0])) for p in hull_points]
            draw.polygon(polygon, fill=255)
            
            # Add to result
            result = np.maximum(result, np.array(hull_mask))
        except Exception:
            # If convex hull fails (e.g., collinear points), just use original
            result[component_mask] = 255
    
    return result


def create_segmentation_mask_groundedsam(
    image: Image.Image,
    labels: list[str],
    threshold: float | list[float] = 0.3,
    dilate: int = 5,
    blur: int = 8,
    device: str = "cuda",
    sam_model: str = "facebook/sam-vit-huge",
    dino_model: str = "IDEA-Research/grounding-dino-base",
    models: Optional[GroundedSAMModels] = None,
    adaptive_blur: bool = False,
    blur_scale: float = 1.0,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
    sequential_labels: bool = False,
    convex_hull: bool = False,
    skip_empty_labels: bool = False,
) -> Image.Image:
    """Create a segmentation mask using GroundingDINO + SAM.
    
    This combination provides significantly better segmentation quality:
    - GroundingDINO: State-of-the-art text-to-bounding-box detection
    - SAM: State-of-the-art mask generation with precise boundaries
    
    Args:
        image: Input image
        labels: Text descriptions of objects to segment
        threshold: Detection confidence threshold (lower = more detections).
            Can be a single float (applied to all labels) or a list of floats
            (one per label, used when sequential_labels=True).
        dilate: Pixels to expand mask (ignored if adaptive_blur=True)
        blur: Blur radius for soft edges (ignored if adaptive_blur=True)
        device: Device to run on
        sam_model: SAM model variant (base/large/huge) - ignored if models provided
        dino_model: GroundingDINO model variant - ignored if models provided
        models: Pre-loaded GroundedSAM models (optional, for reuse across calls)
        adaptive_blur: If True, scale dilation/blur per-object based on object size.
            This is useful for blackout mode where larger objects need more dilation
            to effectively hide their shape (silhouette).
        blur_scale: When adaptive_blur=True, this controls the overall dilation intensity.
            A value of 1.0 gives significant dilation for most objects. Higher values
            increase dilation proportionally.
        size_exponent: Controls size-dependence of blur/dilation.
            - 1.0 (default): Linear - larger objects get proportionally more blur
            - 0.0: Same blur for all objects regardless of size
            - 0.5: Square root scaling - moderate difference between sizes
        scaling_factor: Constant multiplier on effective size (default 1.0).
        sequential_labels: If True, process each label separately and combine masks.
            This ensures strictly additive behavior (adding labels never reduces the
            obfuscated area) but may increase runtime with many labels.
        convex_hull: If True, expand each object's mask to its convex hull.
            This fills in concave regions, hiding the object's silhouette better.
        skip_empty_labels: If True, skip obfuscation of objects where GroundingDINO
            returned an empty label (''). This filters out ambiguous detections
            that couldn't be assigned a specific label.
    
    Returns:
        Grayscale mask image (white = detected regions)
    """
    original_size = image.size
    
    # Normalize threshold: convert to list if needed
    if isinstance(threshold, (int, float)):
        threshold_list = [float(threshold)] * len(labels)
    else:
        threshold_list = list(threshold)
        if len(threshold_list) != len(labels):
            raise ValueError(
                f"When threshold is a list, it must have the same length as labels. "
                f"Got {len(threshold_list)} thresholds for {len(labels)} labels."
            )
    
    # Track whether we loaded models ourselves (for cleanup)
    loaded_locally = models is None
    
    # Load models if not provided
    if models is None:
        print(f"  Loading GroundingDINO ({dino_model})...")
        print(f"  Loading SAM ({sam_model})...")
        models = load_groundedsam_models(device, sam_model, dino_model)
    
    dino_model_obj = models.dino_model
    dino_processor = models.dino_processor
    sam_model_obj = models.sam_model
    sam_processor = models.sam_processor
    
    # Collect all boxes from all labels
    all_boxes = []
    all_detected_labels = []
    
    # Initialize combined mask for sequential processing
    combined_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
    
    if sequential_labels and len(labels) > 1:
        # Process each label separately to ensure strictly additive behavior.
        # When labels are combined, GroundingDINO may distribute confidence
        # across object types, causing some detections to fall below threshold.
        # Processing sequentially ensures adding a label never reduces detections.
        print(f"  Processing {len(labels)} labels sequentially for additive detection...")
        for i, label in enumerate(labels):
            label_threshold = threshold_list[i]
            text_query = label + "."
            inputs = dino_processor(images=image, text=text_query, return_tensors="pt")
            inputs = {k: v.to(models.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = dino_model_obj(**inputs)
            
            results = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=label_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            
            boxes = results["boxes"]
            detected = results["labels"]
            
            if len(boxes) > 0:
                # Filter empty labels if requested
                label_boxes = []
                label_detected = []
                for box, det_label in zip(boxes.cpu().tolist(), detected):
                    if not skip_empty_labels or (det_label and det_label.strip()):
                        label_boxes.append(box)
                        label_detected.append(det_label)
                
                if len(label_boxes) > 0:
                    # Process this label's boxes through SAM
                    label_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
                    
                    for box in label_boxes:
                        # SAM expects boxes as [[x1, y1, x2, y2]] for each image
                        sam_inputs = sam_processor(
                            image,
                            input_boxes=[[[box]]],  # Nested list format
                            return_tensors="pt",
                        )
                        sam_inputs = {k: v.to(models.device) for k, v in sam_inputs.items()}
                        
                        with torch.no_grad():
                            sam_outputs = sam_model_obj(**sam_inputs)
                        
                        # Get the mask (SAM outputs multiple masks, take the best one)
                        masks = sam_processor.image_processor.post_process_masks(
                            sam_outputs.pred_masks.cpu(),
                            sam_inputs["original_sizes"].cpu(),
                            sam_inputs["reshaped_input_sizes"].cpu(),
                        )
                        
                        # Take the mask with highest IoU prediction
                        scores_sam = sam_outputs.iou_scores.cpu().numpy()[0, 0]
                        best_mask_idx = np.argmax(scores_sam)
                        object_mask = masks[0][0, best_mask_idx].numpy().astype(np.uint8) * 255
                        
                        # Apply convex hull per-object if enabled (before blur/dilation)
                        if convex_hull:
                            object_mask = apply_convex_hull(object_mask)
                        
                        # Apply per-object adaptive blur/dilation if enabled
                        if adaptive_blur:
                            object_mask_pil = Image.fromarray(object_mask, mode='L')
                            object_mask_pil = apply_size_adaptive_effects(
                                object_mask_pil,
                                blur_scale=blur_scale,
                                size_exponent=size_exponent,
                                scaling_factor=scaling_factor,
                            )
                            object_mask = np.array(object_mask_pil)
                        
                        # Combine with label mask
                        label_mask = np.maximum(label_mask, object_mask)
                    
                    # Calculate coverage for this label
                    label_coverage = np.sum(label_mask > 127) / label_mask.size * 100
                    print(f"    '{label}': detected {len(label_boxes)} objects (threshold={label_threshold}), coverage: {label_coverage:.1f}%")
                    
                    # Combine with overall mask
                    combined_mask = np.maximum(combined_mask, label_mask)
                    all_boxes.extend(label_boxes)
                    all_detected_labels.extend(label_detected)
                else:
                    print(f"    '{label}': detected {len(boxes)} objects but all were filtered (threshold={label_threshold})")
            else:
                print(f"    '{label}': no objects detected (threshold={label_threshold})")
    
    if not (sequential_labels and len(labels) > 1):
        # Default: combine labels into a single query (original behavior)
        # Use ". " to separate different object categories
        # Use the first threshold (or single threshold) when combining labels
        combined_threshold = threshold_list[0] if threshold_list else 0.3
        text_query = ". ".join(labels) + "."
        
        # Process with GroundingDINO
        inputs = dino_processor(images=image, text=text_query, return_tensors="pt")
        inputs = {k: v.to(models.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = dino_model_obj(**inputs)
        
        # Post-process to get bounding boxes
        # Note: API changed in transformers 4.51+ from box_threshold/text_threshold to just threshold
        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=combined_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]
        
        boxes = results["boxes"]
        detected_labels_result = results["labels"]
        
        if len(boxes) > 0:
            all_boxes = boxes.cpu().tolist()
            all_detected_labels = detected_labels_result
        
        # Non-sequential processing: collect all boxes first, then process
        print(f"  GroundingDINO detected {len(all_boxes)} total objects: {all_detected_labels}")
        
        # Filter out empty labels if requested
        if skip_empty_labels:
            filtered_boxes = []
            filtered_labels = []
            skipped_count = 0
            for i, (box, label) in enumerate(zip(all_boxes, all_detected_labels)):
                if label and label.strip():  # Non-empty label
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
                else:
                    skipped_count += 1
            if skipped_count > 0:
                print(f"  Skipped {skipped_count} objects with empty labels")
                all_boxes = filtered_boxes
                all_detected_labels = filtered_labels
        
        if len(all_boxes) == 0:
            print("  WARNING: No objects detected by GroundingDINO (or all were filtered)")
            # Return empty mask
            return Image.new('L', original_size, 0)
        
        # Process each box through SAM and combine masks
        for i, box in enumerate(all_boxes):
            # SAM expects boxes as [[x1, y1, x2, y2]] for each image
            inputs = sam_processor(
                image,
                input_boxes=[[[box]]],  # Nested list format
                return_tensors="pt",
            )
            inputs = {k: v.to(models.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = sam_model_obj(**inputs)
            
            # Get the mask (SAM outputs multiple masks, take the best one)
            masks = sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            
            # Take the mask with highest IoU prediction
            scores_sam = outputs.iou_scores.cpu().numpy()[0, 0]
            best_mask_idx = np.argmax(scores_sam)
            object_mask = masks[0][0, best_mask_idx].numpy().astype(np.uint8) * 255
            
            # Apply convex hull per-object if enabled (before blur/dilation)
            if convex_hull:
                object_mask = apply_convex_hull(object_mask)
            
            # Apply per-object adaptive blur/dilation if enabled
            if adaptive_blur:
                object_mask_pil = Image.fromarray(object_mask, mode='L')
                object_mask_pil = apply_size_adaptive_effects(
                    object_mask_pil,
                    blur_scale=blur_scale,
                    size_exponent=size_exponent,
                    scaling_factor=scaling_factor,
                )
                object_mask = np.array(object_mask_pil)
            
            # Combine with existing mask (OR operation)
            combined_mask = np.maximum(combined_mask, object_mask)
    
    # Convert to PIL
    mask = Image.fromarray(combined_mask, mode='L')
    
    # Clean up only if we loaded models ourselves
    if loaded_locally:
        del dino_model_obj, dino_processor, sam_model_obj, sam_processor
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Apply global dilate/blur only when NOT using adaptive blur
    # (adaptive blur applies per-object blur/dilation instead)
    if not adaptive_blur:
        # Dilate mask to include edges
        if dilate > 0:
            from PIL import ImageFilter
            for _ in range(dilate):
                mask = mask.filter(ImageFilter.MaxFilter(3))
        
        # Apply blur for soft edges
        if blur > 0:
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    
    # Calculate and report total coverage after blur/dilation
    if sequential_labels and len(labels) > 1:
        mask_array = np.array(mask)
        total_coverage = np.sum(mask_array > 127) / mask_array.size * 100
        print(f"  Total coverage across all objects: {total_coverage:.1f}%")
    
    return mask


def create_segmentation_mask_sam3(
    image: Image.Image,
    labels: list[str],
    threshold: float | list[float] = 0.5,
    dilate: int = 5,
    blur: int = 8,
    device: str = "cuda",
    models: Optional[SAM3Models] = None,
    adaptive_blur: bool = False,
    blur_scale: float = 1.0,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
    sequential_labels: bool = False,
    convex_hull: bool = False,
) -> Image.Image:
    """Create a segmentation mask using SAM3.
    
    SAM3 is a unified foundation model that can segment objects using text
    prompts directly. It combines the capabilities of detection and segmentation
    in a single model, providing high-quality masks without needing separate
    models like GroundingDINO.
    
    Args:
        image: Input image
        labels: Text descriptions of objects to segment
        threshold: Detection confidence threshold (lower = more detections).
            Can be a single float (applied to all labels) or a list of floats
            (one per label, used when sequential_labels=True).
        dilate: Pixels to expand mask (ignored if adaptive_blur=True)
        blur: Blur radius for soft edges (ignored if adaptive_blur=True)
        device: Device to run on
        models: Pre-loaded SAM3 models (optional, for reuse across calls)
        adaptive_blur: If True, scale dilation/blur per-object based on object size.
        blur_scale: When adaptive_blur=True, this controls the overall dilation intensity.
        size_exponent: Controls size-dependence of blur/dilation.
        scaling_factor: Constant multiplier on effective size (default 1.0).
        sequential_labels: If True, process each label separately and combine masks.
            This ensures strictly additive behavior (adding labels never reduces the
            obfuscated area).
        convex_hull: If True, expand each object's mask to its convex hull.
    
    Returns:
        Grayscale mask image (white = detected regions)
    """
    original_size = image.size
    
    # Normalize threshold: convert to list if needed
    if isinstance(threshold, (int, float)):
        threshold_list = [float(threshold)] * len(labels)
    else:
        threshold_list = list(threshold)
        if len(threshold_list) != len(labels):
            raise ValueError(
                f"When threshold is a list, it must have the same length as labels. "
                f"Got {len(threshold_list)} thresholds for {len(labels)} labels."
            )
    
    # Track whether we loaded models ourselves (for cleanup)
    loaded_locally = models is None
    
    # Load models if not provided
    if models is None:
        models = load_sam3_models(device)
    
    sam3_model = models.model
    sam3_processor = models.processor
    sam3_model.eval()
    
    # Initialize combined mask
    combined_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
    
    if sequential_labels and len(labels) > 1:
        # Process each label separately to ensure strictly additive behavior
        print(f"  Processing {len(labels)} labels sequentially for additive detection...")
        for i, label in enumerate(labels):
            label_threshold = threshold_list[i]
            inputs = None
            outputs = None
            results = None
            masks = []
            try:
                # Process with SAM3
                inputs = sam3_processor(images=image, text=label, return_tensors="pt")
                inputs = {k: v.to(models.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    outputs = sam3_model(**inputs)

                # Post-process to get masks
                results = sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=label_threshold,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                masks = results.get("masks", [])

                if len(masks) > 0:
                    label_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)

                    for mask_tensor in masks:
                        # Convert mask tensor to numpy array
                        object_mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255

                        # Apply convex hull per-object if enabled
                        if convex_hull:
                            object_mask = apply_convex_hull(object_mask)

                        # Apply per-object adaptive blur/dilation if enabled
                        if adaptive_blur:
                            object_mask_pil = Image.fromarray(object_mask, mode='L')
                            object_mask_pil = apply_size_adaptive_effects(
                                object_mask_pil,
                                blur_scale=blur_scale,
                                size_exponent=size_exponent,
                                scaling_factor=scaling_factor,
                            )
                            object_mask = np.array(object_mask_pil)

                        # Combine with label mask
                        label_mask = np.maximum(label_mask, object_mask)

                    # Calculate coverage for this label
                    label_coverage = np.sum(label_mask > 127) / label_mask.size * 100
                    print(f"    '{label}': detected {len(masks)} objects (threshold={label_threshold}), coverage: {label_coverage:.1f}%")

                    # Combine with overall mask
                    combined_mask = np.maximum(combined_mask, label_mask)
                else:
                    print(f"    '{label}': no objects detected (threshold={label_threshold})")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(
                        f"  WARNING: SAM3 ran out of memory for '{label}'. "
                        "Skipping this label and continuing."
                    )
                else:
                    raise
            finally:
                del inputs, outputs, results, masks
                if models.device == "cuda":
                    torch.cuda.empty_cache()
    else:
        # Process all labels together (default behavior)
        # For SAM3, we can process labels one at a time and combine
        combined_threshold = threshold_list[0] if threshold_list else 0.5
        
        for label in labels:
            inputs = None
            outputs = None
            results = None
            masks = []
            try:
                inputs = sam3_processor(images=image, text=label, return_tensors="pt")
                inputs = {k: v.to(models.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    outputs = sam3_model(**inputs)

                # Post-process to get masks
                results = sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=combined_threshold,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                masks = results.get("masks", [])

                print(f"  SAM3 detected {len(masks)} objects for '{label}'")

                for mask_tensor in masks:
                    # Convert mask tensor to numpy array
                    object_mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255

                    # Apply convex hull per-object if enabled
                    if convex_hull:
                        object_mask = apply_convex_hull(object_mask)

                    # Apply per-object adaptive blur/dilation if enabled
                    if adaptive_blur:
                        object_mask_pil = Image.fromarray(object_mask, mode='L')
                        object_mask_pil = apply_size_adaptive_effects(
                            object_mask_pil,
                            blur_scale=blur_scale,
                            size_exponent=size_exponent,
                            scaling_factor=scaling_factor,
                        )
                        object_mask = np.array(object_mask_pil)

                    # Combine with overall mask
                    combined_mask = np.maximum(combined_mask, object_mask)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(
                        f"  WARNING: SAM3 ran out of memory for '{label}'. "
                        "Skipping this label and continuing."
                    )
                else:
                    raise
            finally:
                del inputs, outputs, results, masks
                if models.device == "cuda":
                    torch.cuda.empty_cache()
    
    # Convert to PIL
    mask = Image.fromarray(combined_mask, mode='L')
    
    # Clean up only if we loaded models ourselves
    if loaded_locally:
        del sam3_model, sam3_processor
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Apply global dilate/blur only when NOT using adaptive blur
    if not adaptive_blur:
        # Dilate mask to include edges
        if dilate > 0:
            from PIL import ImageFilter
            for _ in range(dilate):
                mask = mask.filter(ImageFilter.MaxFilter(3))
        
        # Apply blur for soft edges
        if blur > 0:
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    
    # Calculate and report total coverage after blur/dilation
    if sequential_labels and len(labels) > 1:
        mask_array = np.array(mask)
        total_coverage = np.sum(mask_array > 127) / mask_array.size * 100
        print(f"  Total coverage across all objects: {total_coverage:.1f}%")
    
    return mask


def create_segmentation_mask(
    image: Image.Image,
    labels: list[str],
    threshold: float | list[float] = 0.4,
    dilate: int = 5,
    blur: int = 8,
    device: str = "cuda",
    segmenter: str = "clipseg",
    clipseg_models: Optional[CLIPSegModels] = None,
    groundedsam_models: Optional[GroundedSAMModels] = None,
    sam3_models: Optional[SAM3Models] = None,
    adaptive_blur: bool = False,
    blur_scale: float = 1.0,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
    sequential_labels: bool = False,
    convex_hull: bool = False,
    skip_empty_labels: bool = False,
) -> Image.Image:
    """Create a segmentation mask using the specified segmenter.
    
    Args:
        image: Input image
        labels: Text descriptions of objects to segment
        threshold: Confidence threshold. Can be a single float (applied to all labels)
            or a list of floats (one per label, used when sequential_labels=True).
        dilate: Pixels to expand mask (ignored if adaptive_blur=True)
        blur: Blur radius for soft edges (ignored if adaptive_blur=True)
        device: Device to run on
        segmenter: Which segmentation model to use:
            - "clipseg": CLIPSeg (fast, decent quality)
            - "groundedsam": GroundingDINO + SAM (slower, much better quality)
            - "sam3": SAM3 (state-of-the-art, text-prompted segmentation)
        clipseg_models: Pre-loaded CLIPSeg models (optional, for reuse)
        groundedsam_models: Pre-loaded GroundedSAM models (optional, for reuse)
        sam3_models: Pre-loaded SAM3 models (optional, for reuse)
        adaptive_blur: If True, scale blur/dilation based on object size.
            For GroundedSAM and SAM3, this is applied per-object. For CLIPSeg, it's
            applied to the combined mask based on overall detected size.
        blur_scale: When adaptive_blur=True, controls the overall blur intensity.
            A value of 1.0 gives reasonable blur for most objects. Higher values
            increase blur proportionally.
        size_exponent: Controls size-dependence of blur/dilation.
            - 1.0 (default): Linear - larger objects get proportionally more blur
            - 0.0: Same blur for all objects regardless of size
            - 0.5: Square root scaling - moderate difference between sizes
        scaling_factor: Constant multiplier on effective size (default 1.0).
        sequential_labels: If True (and using groundedsam or sam3), process each label
            separately and combine masks. This ensures strictly additive behavior
            (adding labels never reduces the obfuscated area) but increases runtime.
            Has no effect when using clipseg (which already processes labels
            independently).
        convex_hull: If True, expand each object's mask to its convex hull.
            This fills in concave regions, hiding the object's silhouette better.
        skip_empty_labels: If True, skip obfuscation of objects where GroundingDINO
            returned an empty label (''). Only applies to groundedsam segmenter.
    
    Returns:
        Grayscale mask image (white = detected regions)
    """
    if segmenter == "clipseg":
        return create_segmentation_mask_clipseg(
            image=image,
            labels=labels,
            threshold=threshold,
            dilate=dilate,
            blur=blur,
            device=device,
            models=clipseg_models,
            adaptive_blur=adaptive_blur,
            blur_scale=blur_scale,
            size_exponent=size_exponent,
            scaling_factor=scaling_factor,
            convex_hull=convex_hull,
        )
    elif segmenter == "groundedsam":
        return create_segmentation_mask_groundedsam(
            image=image,
            labels=labels,
            threshold=threshold,
            dilate=dilate,
            blur=blur,
            device=device,
            models=groundedsam_models,
            adaptive_blur=adaptive_blur,
            blur_scale=blur_scale,
            size_exponent=size_exponent,
            scaling_factor=scaling_factor,
            sequential_labels=sequential_labels,
            convex_hull=convex_hull,
            skip_empty_labels=skip_empty_labels,
        )
    elif segmenter == "sam3":
        return create_segmentation_mask_sam3(
            image=image,
            labels=labels,
            threshold=threshold,
            dilate=dilate,
            blur=blur,
            device=device,
            models=sam3_models,
            adaptive_blur=adaptive_blur,
            blur_scale=blur_scale,
            size_exponent=size_exponent,
            scaling_factor=scaling_factor,
            sequential_labels=sequential_labels,
            convex_hull=convex_hull,
        )
    else:
        raise ValueError(
            f"Unknown segmenter: {segmenter}. Choose 'clipseg', 'groundedsam', 'sam3', or "
            "'ai-gen' (note: 'ai-gen' is handled separately in anonymize function)"
        )


def anonymize_with_openai(
    image: Image.Image,
    target_labels: list[str],
    api_key: Optional[str] = None,
) -> Image.Image:
    """
    Anonymize objects in an image using OpenAI's image edit API.
    
    OpenAI handles both identification and obfuscation in one step.
    The objects are blacked out (replaced with solid black).
    
    Args:
        image: Input image to anonymize.
        target_labels: List of objects to detect and black out.
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
    
    Returns:
        Anonymized PIL Image.
    """
    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    client = OpenAI(api_key=api_key)
    
    # Build the prompt
    objects_list = "\n".join([f"- {label}" for label in target_labels])
    prompt = f"""Black out all regions containing:
{objects_list}

The blacked-out regions must be solid black.
Do not alter any other pixels.
When the blacked-out object can be recognized by silhouette alone, 
blur the edges to ensure it cannot be recognized.
"""
    
    # Save image to a temporary file so OpenAI API can detect MIME type
    # The API needs a file with proper extension to determine the image format
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        image.save(tmp_path, format="PNG")
    
    try:
        # Call OpenAI image edit API with file path
        with open(tmp_path, "rb") as image_file:
            result = client.images.edit(
                model="gpt-image-1.5",
                image=image_file,
                prompt=prompt,
                size="auto"  # "auto" preserves aspect ratio, other options: "1024x1024", "1024x1536", "1536x1024"
            )
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    
    # Decode base64 image
    # The API returns result.data[0].b64_json for base64-encoded images
    if result.data and len(result.data) > 0:
        image_obj = result.data[0]
        if hasattr(image_obj, 'b64_json') and image_obj.b64_json:
            image_data = base64.b64decode(image_obj.b64_json)
            result_image = Image.open(io.BytesIO(image_data))
            return result_image.convert("RGB")
        elif hasattr(image_obj, 'url') and image_obj.url:
            # If URL is provided instead, we'd need to download it
            # For now, raise an error asking for base64 format
            raise ValueError("OpenAI API returned URL instead of base64. Please ensure the API returns base64-encoded images.")
        else:
            raise ValueError("OpenAI API did not return image data in expected format")
    else:
        raise ValueError("OpenAI API returned no image data")


def anonymize_with_fal(
    image: Image.Image,
    target_labels: list[str],
    privacy_concept: Optional[str] = None,
    fal_key: Optional[str] = None,
    image_model: str = "gpt-image-1.5",
) -> Image.Image:
    """
    Anonymize objects in an image using fal.ai image edit API.

    Uses the specified fal.ai edit endpoint to both identify and black out the requested regions.

    Args:
        image: Input image to anonymize.
        target_labels: List of objects to detect and black out.
        fal_key: fal.ai API key. If None, uses FAL_KEY environment variable.
        image_model: Either a fal.ai model path (e.g. fal-ai/gpt-image-1; /edit is appended)
            or a short name: "gpt-image-1.5", "nano-banana-pro", "flux-2-pro", "flux-2-dev".

    Returns:
        Anonymized PIL Image.
    """
    import fal_client

    if "/" in image_model:
        # User passed a fal.ai path (e.g. fal-ai/gpt-image-1). Append operation if not given.
        path = image_model.strip("/")
        parts = path.split("/")
        if len(parts) == 2:
            op = FAL_IMAGE_EDIT_OPERATION.get(path, DEFAULT_FAL_EDIT_OPERATION)
            endpoint = f"{path}/{op}"
        else:
            endpoint = image_model
    elif image_model in FAL_IMAGE_MODEL_ENDPOINTS:
        endpoint = FAL_IMAGE_MODEL_ENDPOINTS[image_model]
    else:
        raise ValueError(
            f"Unknown fal image model: {image_model}. "
            f"Use a fal.ai model path (e.g. fal-ai/gpt-image-1) or one of: {list(FAL_IMAGE_MODEL_ENDPOINTS.keys())}"
        )
    if fal_key is None:
        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            raise ValueError(
                "fal.ai API key not provided. Set FAL_KEY environment variable "
                "or pass fal_key parameter."
            )
    if privacy_concept is not None:
        # When a privacy concept is provided, fal.ai should infer what to redact from the concept.
        prompt = (
            f"Black out all regions in this image that can reveal {privacy_concept}. "
            "The blacked-out regions must be solid black. "
            "Do not alter any other pixels. "
            "When the blacked-out object can be recognized by silhouette alone, "
            "blur the edges to ensure it cannot be recognized."
        )
    else:
        objects_list = "\n".join([f"- {label}" for label in target_labels])
        prompt = f"""Black out all regions containing:
{objects_list}

The blacked-out regions must be solid black.
Do not alter any other pixels.
When the blacked-out object can be recognized by silhouette alone,
blur the edges to ensure it cannot be recognized.
"""

    # Save as JPEG for upload so size stays close to originals (PNG would often exceed 25MB for photos).
    original_size = image.size
    image_for_upload = image.convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        image_for_upload.save(tmp_path, format="JPEG", quality=95)

    try:
        while os.path.getsize(tmp_path) > MAX_FAL_UPLOAD_BYTES:
            current_size = os.path.getsize(tmp_path)
            scale = (MAX_FAL_UPLOAD_BYTES * 0.95 / current_size) ** 0.5
            new_w = max(64, int(image_for_upload.width * scale))
            new_h = max(64, int(image_for_upload.height * scale))
            image_for_upload = image_for_upload.resize((new_w, new_h), Image.Resampling.LANCZOS)
            image_for_upload.save(tmp_path, format="JPEG", quality=95)
            print(f"  Image downsized to {new_w}x{new_h} for upload (file was {current_size / 1e6:.1f}MB, limit {MAX_FAL_UPLOAD_BYTES / 1e6:.0f}MB)")

        image_url = fal_client.upload_file(tmp_path)
        result = fal_client.subscribe(
            endpoint,
            arguments={
                "prompt": prompt,
                "image_urls": [image_url],
            },
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not result.get("images") or len(result["images"]) == 0:
        raise ValueError("fal.ai API returned no image data")

    out_url = result["images"][0].get("url")
    if not out_url:
        raise ValueError("fal.ai API did not return image URL in expected format")

    with urllib.request.urlopen(out_url) as resp:
        data = resp.read()
    result_image = Image.open(io.BytesIO(data)).convert("RGB")
    if result_image.size != original_size:
        result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
    return result_image


def inpaint_with_flux(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    strength: float = 0.95,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 28,
    seed: Optional[int] = None,
    device: str = "cuda",
    cpu_offload: bool = True,
    model: str = "schnell",
) -> Image.Image:
    """Inpaint masked regions using FLUX.
    
    Args:
        model: Which FLUX model to use:
            - "schnell": Fast but lower quality (Apache 2.0 license)
            - "dev": Higher quality but slower (non-commercial license)
    """
    from diffusers import FluxInpaintPipeline

    # Ensure same size and multiples of 8
    w, h = image.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    # Select model
    model_id = {
        "schnell": "black-forest-labs/FLUX.1-schnell",
        "dev": "black-forest-labs/FLUX.1-dev",
    }.get(model, "black-forest-labs/FLUX.1-schnell")
    
    print(f"Using model: {model_id}")

    # Load pipeline
    pipe = FluxInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )

    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Set up generator
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    # Inpaint
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    return result.images[0]


def anonymize(
    input_path: str | Path,
    output_path: str | Path,
    target_labels: list[str],
    replacement_prompt: Optional[str] = None,
    threshold: float | list[float] = 0.4,
    dilate: int = 5,
    blur: int = 8,
    strength: float = 0.85,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    save_mask: Optional[str | Path] = None,
    realistic: bool = True,
    model: str = "dev",
    redact: bool = False,
    redact_blur_radius: int = 30,
    blackout: bool = False,
    device: Optional[str] = None,
    cpu_offload: bool = True,
    segmenter: str = "groundedsam",
    clipseg_models: Optional[CLIPSegModels] = None,
    groundedsam_models: Optional[GroundedSAMModels] = None,
    sam3_models: Optional[SAM3Models] = None,
    adaptive_blur: bool = False,
    blur_scale: float = 1.0,
    size_exponent: float = 1.0,
    scaling_factor: float = 1.0,
    sequential_labels: bool = False,
    convex_hull: bool = False,
    skip_empty_labels: bool = False,
    refinements: int = 0,
    fal_image_model: str = "gpt-image-1.5",
    fal_vision_model: str = DEFAULT_FAL_VISION_MODEL,
    fal_vision_temperature: float = DEFAULT_FAL_VISION_TEMPERATURE,
    privacy_concept: Optional[str] = None,
) -> tuple[Image.Image, float]:
    """
    Automatically detect and replace objects in an image.

    Args:
        input_path: Path to the input image.
        output_path: Path where the result will be saved.
        target_labels: Text descriptions of objects to replace (e.g., ["face", "person"]).
        replacement_prompt: What to replace the objects with (not needed if redact/blackout=True).
        threshold: Segmentation threshold (0.0-1.0). Lower = more inclusive.
            Can be a single float (applied to all labels) or a list of floats
            (one per label, used when sequential_labels=True).
        dilate: Pixels to expand the mask (ignored if adaptive_blur=True).
        blur: Blur radius for mask edges (ignored if adaptive_blur=True).
        strength: Inpainting strength (0.0-1.0). Default 0.85 for natural blending.
        guidance_scale: Prompt adherence strength.
        num_inference_steps: Number of denoising steps.
        seed: Random seed for reproducibility.
        save_mask: Optional path to save the generated mask.
        realistic: If True, enhance prompt for photorealistic results.
        redact: If True, blur the masked region instead of inpainting.
        redact_blur_radius: Blur radius for redaction (higher = more blurred).
        blackout: If True, replace masked region with black pixels.
        device: Device to run on.
        cpu_offload: Whether to use CPU offloading.
        segmenter: Segmentation model to use:
            - "groundedsam": GroundingDINO + SAM (better quality, recommended)
            - "clipseg": CLIPSeg (faster, lower quality)
            - "sam3": SAM3 (state-of-the-art text-prompted segmentation)
            - "ai-gen": fal.ai image edit API (handles both detection and obfuscation in one step)
            - "vlm-bounding-box": fal.ai OpenRouter vision for polygon detection + blackout obfuscation
        fal_image_model: When segmenter is "ai-gen", fal.ai image edit model: model path (e.g. fal-ai/gpt-image-1)
            or short name (gpt-image-1.5, nano-banana-pro, flux-2-pro, flux-2-dev). Requires FAL_KEY.
        fal_vision_temperature: OpenRouter sampling temperature when segmenter is "vlm-bounding-box".
        clipseg_models: Pre-loaded CLIPSeg models (optional, for reuse across calls).
        groundedsam_models: Pre-loaded GroundedSAM models (optional, for reuse across calls).
        sam3_models: Pre-loaded SAM3 models (optional, for reuse across calls).
        adaptive_blur: If True, scale mask dilation and blur based on object size.
            This is particularly useful for blackout mode where larger objects
            need more dilation to effectively hide their shape/silhouette. When
            enabled, the dilate and blur parameters are ignored and computed
            automatically based on object size.
        blur_scale: When adaptive_blur=True, controls the overall dilation intensity.
            A value of 1.0 gives significant dilation (50% of object's characteristic
            size). Higher values (e.g., 1.5, 2.0) increase dilation proportionally.
        size_exponent: Controls size-dependence of blur/dilation when adaptive_blur=True.
            - 1.0 (default): Linear - larger objects get proportionally more blur
            - 0.0: Same blur for all objects regardless of size
            - 0.5: Square root scaling - moderate difference between sizes
        scaling_factor: Constant multiplier on effective size (default 1.0).
            Formula: effective_size = scaling_factor × char_size^exponent
            Then: dilation_radius = blur_scale × effective_size
        sequential_labels: If True (and using groundedsam), process each label
            separately and combine masks. This ensures strictly additive behavior
            (adding labels never reduces the obfuscated area) but increases runtime.
            Recommended when you notice that adding objects reduces coverage.
        convex_hull: If True, expand each object's mask to its convex hull.
            This fills in concave regions (e.g., space between arms and body),
            better hiding the object's silhouette. Applied per-object before
            any blur or dilation.
        skip_empty_labels: If True, skip obfuscation of objects where GroundingDINO
            returned an empty label (''). This filters out ambiguous detections
            that couldn't be assigned a specific label. Only applies when using
            groundedsam segmenter.
        refinements: Number of refinement passes for fal.ai vision polygon detection (default: 0).
            Only applies when using vlm-bounding-box segmenter. Each refinement pass shows
            GPT-5.2 its previous polygon detections and asks it to improve the boundaries.

    Returns:
        Tuple of (anonymized PIL Image, coverage percentage).
        Coverage is the percentage of the image that was detected and modified.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    print(f"Loading image: {input_path}")
    image = load_image(input_path)
    print(f"Image size: {image.size}")

    # Handle ai-gen segmenter (fal.ai image edit: detection + obfuscation in one step)
    if segmenter == "ai-gen":
        if privacy_concept is not None:
            print(f"\nUsing fal.ai image edit API (ai-gen) for privacy concept: {privacy_concept}")
        else:
            print(f"\nUsing fal.ai image edit API (ai-gen) for: {', '.join(target_labels)}")
        print(f"  Model: {fal_image_model} (fal.ai handles both detection and obfuscation)")
        
        result = anonymize_with_fal(
            image=image,
            target_labels=target_labels,
            privacy_concept=privacy_concept,
            fal_key=os.getenv("FAL_KEY"),
            image_model=fal_image_model,
        )
        
        # Resize result to match original image size (OpenAI may return different size)
        if result.size != image.size:
            print(f"  Resizing result from {result.size} to {image.size} to match original")
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        
        # Calculate coverage by comparing original vs modified image
        # Count pixels that changed to black (or very dark)
        original_array = np.array(image.convert("RGB"))
        result_array = np.array(result.convert("RGB"))
        
        # Find pixels that are black (or very dark) in result but not in original
        # A pixel is considered "blacked out" if it's very dark (sum < 30) in result
        black_threshold = 30  # Sum of RGB values
        result_dark = np.sum(result_array, axis=2) < black_threshold
        original_dark = np.sum(original_array, axis=2) < black_threshold
        
        # Coverage is pixels that became dark (were not dark before)
        changed_to_black = result_dark & ~original_dark
        coverage = np.sum(changed_to_black) / changed_to_black.size * 100
        
        print(f"Detected region coverage: {coverage:.1f}%")
        
        # Save result
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"\nAnonymized image saved to: {output_path}")
        
        return result, coverage

    # Handle vlm-bounding-box segmenter (fal.ai OpenRouter vision for polygon detection, then blackout)
    if segmenter == "vlm-bounding-box":
        print(f"\nUsing fal.ai OpenRouter vision ({fal_vision_model}) for polygon detection: {', '.join(target_labels)}")
        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            raise ValueError("fal.ai API key not set. Set FAL_KEY for vision-based polygon detection (segmenter vlm-bounding-box).")
        if not blackout:
            print("  Note: Vision segmenter currently only supports blackout mode")
            print("  Automatically using blackout obfuscation")
        result, coverage, polygons = anonymize_with_gpt52_polygons(
            image=image,
            target_labels=target_labels,
            fal_key=fal_key,
            fal_vision_model=fal_vision_model,
            dilate=dilate,
            blur=blur,
            refinements=refinements,
            fal_vision_temperature=fal_vision_temperature,
        )
        
        print(f"Detected region coverage: {coverage:.1f}%")
        
        # Save result
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"\nAnonymized image saved to: {output_path}")
        
        return result, coverage

    # Create segmentation mask (for non-OpenAI segmenters)
    print(f"\n[Step 1/2] Creating segmentation mask for: {', '.join(target_labels)}")
    print(f"  Using segmenter: {segmenter}")
    if adaptive_blur:
        print(f"  Adaptive blur enabled (blur_scale={blur_scale}, scaling_factor={scaling_factor}, exponent={size_exponent})")
    if sequential_labels and len(target_labels) > 1:
        print(f"  Sequential labels enabled (additive detection)")
    if convex_hull:
        print(f"  Convex hull enabled (fills concave regions)")
    if skip_empty_labels:
        print(f"  Skip empty labels enabled (filters ambiguous detections)")
    mask = create_segmentation_mask(
        image=image,
        labels=target_labels,
        threshold=threshold,
        dilate=dilate,
        blur=blur,
        device=device,
        segmenter=segmenter,
        clipseg_models=clipseg_models,
        groundedsam_models=groundedsam_models,
        sam3_models=sam3_models,
        adaptive_blur=adaptive_blur,
        blur_scale=blur_scale,
        size_exponent=size_exponent,
        scaling_factor=scaling_factor,
        sequential_labels=sequential_labels,
        convex_hull=convex_hull,
        skip_empty_labels=skip_empty_labels,
    )

    # Check if anything was detected
    mask_array = np.array(mask)
    coverage = np.sum(mask_array > 127) / mask_array.size * 100
    print(f"Detected region coverage: {coverage:.1f}%")

    if coverage < 0.1:
        print("WARNING: Very little was detected. Try:")
        print("  - Lower threshold (--threshold 0.3)")
        print("  - Different labels (e.g., 'human face' instead of 'face')")
        print("  - Check if the object is clearly visible in the image")

    # Save mask if requested
    if save_mask:
        mask.save(save_mask)
        print(f"Mask saved to: {save_mask}")

    # Either redact (blur), blackout (black pixels), or inpaint (replace object)
    if redact:
        # Simple blur redaction - fast and doesn't require FLUX
        print(f"\n[Step 2/2] Redacting with gaussian blur (radius={redact_blur_radius})")
        result = redact_with_blur(image, mask, blur_radius=redact_blur_radius)
    elif blackout:
        # Replace with black pixels - fast and doesn't require FLUX
        print(f"\n[Step 2/2] Blacking out masked region")
        result = blackout_with_black(image, mask)
    else:
        # AI inpainting with FLUX to replace with user-specified content
        if not replacement_prompt:
            raise ValueError("replacement_prompt is required when not using --redact or --blackout")
        
        # Enhance prompt for realism if requested
        final_prompt = replacement_prompt
        if realistic:
            # Add modifiers for photorealistic, natural-looking results
            realism_suffix = ", photorealistic, natural lighting, seamless blend with surroundings, matching skin tone and lighting of the scene"
            final_prompt = f"{replacement_prompt}{realism_suffix}"
            print(f"(Realistic mode: prompt enhanced for natural results)")

        # Inpaint
        print(f"\n[Step 2/2] Inpainting with prompt: '{final_prompt}'")
        result = inpaint_with_flux(
            image=image,
            mask=mask,
            prompt=final_prompt,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            device=device,
            cpu_offload=cpu_offload,
            model=model,
        )

    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"\nAnonymized image saved to: {output_path}")

    return result, coverage


def main() -> None:
    """CLI entry point for anonymization."""
    parser = argparse.ArgumentParser(
        description="Automatically detect and replace objects in images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # REDACT: Simply blur faces (fast, no AI needed)
  %(prog)s photo.png result.png --target "face" --redact
  
  # REDACT: Blur with stronger effect
  %(prog)s photo.png result.png --target "face" --redact --redact-blur 50
  
  # INPAINT: Replace faces with AI-generated faces
  %(prog)s photo.png result.png --target "face" --replace "a different person's face"
  
  # Replace all people with silhouettes
  %(prog)s crowd.jpg anon.jpg --target "person" "people" --replace "dark silhouette of a person"
  
  # Remove text/signs from image
  %(prog)s street.png clean.png --target "text" "sign" --replace "plain wall"
  
  # Save the generated mask for inspection
  %(prog)s photo.png result.png --target "face" --redact --save-mask mask.png
  
  # ADAPTIVE BLUR: Scale blur based on object size (great for blackout)
  %(prog)s photo.png result.png --target "person" --blackout --adaptive-blur
  
  # ADAPTIVE BLUR: Increase blur intensity for stronger obfuscation
  %(prog)s photo.png result.png --target "face" --blackout --adaptive-blur --blur-scale 1.5
  
  # GPT-5.2: Use GPT-5.2 vision API for polygon detection and blackout
  %(prog)s photo.png result.png --target "face" "license plate" --segmenter vlm-bounding-box --blackout

COMMON USE CASES:
  Face redaction (blur):
    --target "face" "human face" --redact
    
  Face replacement (AI):
    --target "face" "human face" --replace "a different person's face"
    
  Person removal:
    --target "person" --replace "empty background, no people"
    
  License plate redaction:
    --target "license plate" "car plate" --redact --redact-blur 40
    
  Text removal:
    --target "text" "words" "sign" --replace "plain surface"

TIPS FOR NATURAL RESULTS:
  - Use multiple target labels for better detection
  - Lower --threshold (0.3-0.4) if objects aren't fully detected
  - Increase --dilate (10-20) to ensure edges are covered
  - Increase --blur (15-20) for smoother blending with background
  - Lower --strength (0.7-0.8) to preserve more context
  - Use descriptive prompts: "a middle-aged woman with brown hair"
  - Use --save-mask to inspect what was detected
  
AVOIDING "FRANKENSTEIN" EFFECT:
  - Use --model dev for MUCH better quality (slower, non-commercial license)
  - The --blur option softens mask edges (default: 8)
  - Lower --strength preserves more of the original lighting/context
  - Realistic mode (on by default) adds photorealism to prompts
  - Match the style: "photo of a person" vs "painted portrait"
  - Use specific prompts: "bearded man with glasses" not "random face"

SEGMENTATION MODELS:
  --segmenter groundedsam (default):
    Uses GroundingDINO + SAM for state-of-the-art segmentation.
    - Much more precise mask boundaries
    - Better at detecting specified objects
    - Slower, requires more VRAM
    
  --segmenter clipseg:
    Uses CLIPSeg for faster, simpler segmentation.
    - Faster and uses less memory
    - Good for simple cases
    - May produce less precise boundaries

  --segmenter sam3:
    Uses SAM3 (Segment Anything Model 3) for text-prompted segmentation.
    - State-of-the-art unified model for promptable segmentation
    - Directly segments objects using text prompts (no separate detection model)
    - Handles 270K+ unique concepts (50x more than prior models)
    - High-quality masks with excellent boundary precision
    - See https://huggingface.co/facebook/sam3 for details

ADAPTIVE BLUR (--adaptive-blur):
  Automatically scales dilation based on object size.
  
  Formula:
    effective_size = scaling_factor × char_size^exponent
    dilation_radius = blur_scale × effective_size
  
  With defaults (all = 1.0): dilation_radius = char_size
  
  Parameters:
    --blur-scale      Overall multiplier (default: 1.0)
    --scaling-factor  Constant multiplier (default: 1.0)
    --size-exponent   Power for char_size (default: 1.0)
        0.0 = same dilation for all objects
        0.5 = square root (less size difference)
        1.0 = linear (default)
  
  With GroundedSAM, adaptive blur is applied per-object.

SEQUENTIAL LABELS (--sequential-labels):
  When using GroundedSAM with multiple target labels (e.g., --target face person),
  the model processes them as a single combined query. This can cause "competition"
  where the model distributes confidence across object types, potentially causing
  some detections to fall below threshold that would have passed individually.
  
  With --sequential-labels, each label is processed separately and the resulting
  masks are combined with OR logic. This ensures STRICTLY ADDITIVE behavior:
  adding more labels can never reduce the total obfuscated area.
  
  Trade-off: Runtime increases proportionally with the number of labels, as
  GroundingDINO must run once per label instead of once total.
  
  Example:
    # Without sequential-labels: "face" detections may be lost when "person" is added
    %(prog)s photo.png out.png --target "face" "person" --redact
    
    # With sequential-labels: guaranteed to detect at least as much as each label alone
    %(prog)s photo.png out.png --target "face" "person" --redact --sequential-labels

CONVEX HULL (--convex-hull):
  Expands each detected object's mask to its convex hull. This fills in any
  concave regions of the mask, such as:
  - Space between a person's arms and body
  - Gaps between fingers
  - Indentations in irregular shapes
  
  This is particularly useful with --blackout mode, where the goal is to hide
  not just the object's appearance but also its recognizable shape/silhouette.
  
  Example:
    # Blackout people with convex hull to hide their pose
    %(prog)s photo.png out.png --target "person" --blackout --convex-hull
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
        help="Output path for the anonymized image.",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        nargs="+",
        required=True,
        help="Objects to detect and replace (e.g., 'face' 'person').",
    )
    parser.add_argument(
        "--replace", "-r",
        type=str,
        default=None,
        help="What to replace the detected objects with (not needed if using --redact).",
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Blur the detected region instead of AI inpainting (faster, no GPU needed).",
    )
    parser.add_argument(
        "--redact-blur",
        type=int,
        default=30,
        help="Blur radius for redaction (default: 30). Higher = more blurred.",
    )
    parser.add_argument(
        "--blackout",
        action="store_true",
        help="Replace the detected region with black pixels (faster, no GPU needed).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.4],
        help="Detection threshold (default: 0.4). Lower = more inclusive. "
             "Can specify multiple thresholds (one per target object). "
             "For clipseg: per-label thresholds are applied natively. "
             "For groundedsam: requires --sequential-labels to use per-label thresholds. "
             "If multiple thresholds are provided, they must match the number of --target objects.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=5,
        help="Pixels to expand detected regions (default: 5).",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=8,
        help="Blur radius for mask edges (default: 8). Reduces harsh seams.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="Inpainting strength (default: 0.85). Lower = better blending.",
    )
    parser.add_argument(
        "--no-realistic",
        action="store_true",
        help="Disable automatic prompt enhancement for realism.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="dev",
        choices=["schnell", "dev"],
        help="FLUX model: 'dev' (better quality, non-commercial) or 'schnell' (fast, Apache license). Default: dev.",
    )
    parser.add_argument(
        "--segmenter", "-s",
        type=str,
        default="groundedsam",
        choices=["groundedsam", "clipseg", "sam3", "ai-gen", "vlm-bounding-box"],
        help=(
            "Segmentation model: 'groundedsam' (GroundingDINO+SAM, better quality), "
            "'clipseg' (faster), 'sam3' (state-of-the-art text-prompted segmentation), "
            "'ai-gen' (fal.ai image edit API, handles both detection and obfuscation), "
            "or 'vlm-bounding-box' (fal.ai OpenRouter vision for polygon detection + blackout). "
            "Default: groundedsam."
        ),
    )
    parser.add_argument(
        "--fal-image-model",
        type=str,
        default="gpt-image-1.5",
        help="When --segmenter ai-gen: fal.ai image edit model. Pass the model path (e.g. fal-ai/gpt-image-1) "
             "or a short name: gpt-image-1.5, nano-banana-pro, flux-2-pro, flux-2-dev. Default: gpt-image-1.5. Requires FAL_KEY.",
    )
    parser.add_argument(
        "--fal-vision-model",
        type=str,
        default=DEFAULT_FAL_VISION_MODEL,
        choices=list(OPENROUTER_VISION_MODELS.keys()),
        help=f"When --segmenter vlm-bounding-box: which fal.ai OpenRouter vision model to use. "
             f"Default: {DEFAULT_FAL_VISION_MODEL}. Requires FAL_KEY.",
    )
    parser.add_argument(
        "--fal-vision-temperature",
        type=float,
        default=DEFAULT_FAL_VISION_TEMPERATURE,
        help=f"OpenRouter sampling temperature when --segmenter vlm-bounding-box "
             f"(default: {DEFAULT_FAL_VISION_TEMPERATURE}).",
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=7.0,
        help="Prompt adherence (default: 7.0). Higher = stricter.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Inference steps (default: 50 for dev model quality).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-mask",
        type=str,
        default=None,
        help="Optional: save the generated mask to this path.",
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
        help="Disable CPU offloading.",
    )
    parser.add_argument(
        "--adaptive-blur",
        action="store_true",
        help="Scale mask blur and dilation based on object size. "
             "Larger objects get more blur to effectively hide their shape. "
             "Particularly useful for --blackout mode. When enabled, --dilate "
             "and --blur are ignored; use --blur-scale to control intensity.",
    )
    parser.add_argument(
        "--blur-scale",
        type=float,
        default=1.0,
        help="Blur intensity when using --adaptive-blur (default: 1.0). "
             "Higher values (e.g., 1.5, 2.0) increase dilation proportionally.",
    )
    parser.add_argument(
        "--size-exponent",
        type=float,
        default=1.0,
        help="Controls size-dependence of blur when using --adaptive-blur. "
             "1.0 (default) = linear scaling (larger objects get more blur). "
             "0.0 = same blur for all objects regardless of size. "
             "0.5 = square root scaling (less difference between sizes).",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=1.0,
        help="Constant multiplier for --adaptive-blur (default: 1.0). "
             "Formula: effective_size = scaling_factor × char_size^exponent. "
             "Then: dilation_radius = blur_scale × effective_size.",
    )
    parser.add_argument(
        "--sequential-labels",
        action="store_true",
        help="Process each target label separately and combine masks (only for groundedsam). "
             "This ensures strictly additive behavior: adding more labels can never reduce "
             "the total obfuscated area. Use this when you notice that adding objects to "
             "detect actually reduces coverage. Increases runtime proportionally with the "
             "number of labels.",
    )
    parser.add_argument(
        "--convex-hull",
        action="store_true",
        help="Expand each detected object's mask to its convex hull. "
             "This fills in concave regions (e.g., space between arms and body), "
             "better hiding the object's silhouette. Particularly useful with "
             "--blackout mode to prevent shape recognition.",
    )
    parser.add_argument(
        "--skip-empty-labels",
        action="store_true",
        help="Skip obfuscation of objects where GroundingDINO returned an empty label (''). "
             "This filters out ambiguous detections that couldn't be assigned a specific label. "
             "Only applies when using --segmenter groundedsam.",
    )
    parser.add_argument(
        "--refinements",
        type=int,
        default=0,
        help="Number of refinement passes for GPT-5.2 polygon detection (default: 0). "
            "Only applies when using --segmenter vlm-bounding-box. Each refinement pass shows "
            "the fal.ai vision model its previous polygon detections and asks it to improve the boundaries.",
    )

    args = parser.parse_args()

    # Validate: need exactly one of --redact, --blackout, or --replace
    mode_count = sum([args.redact, args.blackout, bool(args.replace)])
    if mode_count == 0:
        parser.error("One of --replace, --redact, or --blackout must be specified")
    if mode_count > 1:
        parser.error("Only one of --replace, --redact, or --blackout can be specified")
    
    # Normalize threshold: convert list to single float if only one value, or validate list length
    threshold_value = args.threshold
    if isinstance(threshold_value, list):
        if len(threshold_value) == 1:
            # Single threshold provided as list, convert to float
            threshold_value = threshold_value[0]
        elif len(threshold_value) != len(args.target):
            parser.error(
                f"Number of thresholds ({len(threshold_value)}) must match number of target objects "
                f"({len(args.target)}). Got thresholds: {threshold_value}, targets: {args.target}"
            )
        # For groundedsam without sequential_labels, multiple thresholds won't work
        # (CLIPSeg and groundedsam with sequential_labels both support per-label thresholds)
        if args.segmenter == "groundedsam" and not args.sequential_labels and len(args.threshold) > 1:
            print(f"WARNING: Multiple thresholds provided but --sequential-labels is not enabled. "
                  f"For groundedsam, only the first threshold ({args.threshold[0]}) will be used. "
                  f"Use --sequential-labels to enable per-label thresholds, or use --segmenter clipseg "
                  f"which supports per-label thresholds natively.")
            threshold_value = args.threshold[0]

    anonymize(
        input_path=args.input,
        output_path=args.output,
        target_labels=args.target,
        replacement_prompt=args.replace,
        threshold=threshold_value,
        dilate=args.dilate,
        blur=args.blur,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
        save_mask=args.save_mask,
        realistic=not args.no_realistic,
        model=args.model,
        redact=args.redact,
        redact_blur_radius=args.redact_blur,
        blackout=args.blackout,
        device=args.device,
        cpu_offload=not args.no_cpu_offload,
        segmenter=args.segmenter,
        adaptive_blur=args.adaptive_blur,
        blur_scale=args.blur_scale,
        size_exponent=args.size_exponent,
        scaling_factor=args.scaling_factor,
        sequential_labels=args.sequential_labels,
        convex_hull=args.convex_hull,
        skip_empty_labels=args.skip_empty_labels,
        refinements=args.refinements,
        fal_image_model=args.fal_image_model,
        fal_vision_model=args.fal_vision_model,
        fal_vision_temperature=args.fal_vision_temperature,
    )


if __name__ == "__main__":
    main()

