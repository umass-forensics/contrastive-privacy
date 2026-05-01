#!/usr/bin/env python3
"""
Compare two images using CLIP embeddings and compute their cosine similarity.

This script uses OpenAI's CLIP model to encode images into a shared embedding space
and computes the cosine similarity between them. Higher scores indicate more
semantically similar images.

Example:
    compare-images image1.png image2.png
"""

import argparse
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
)

# HuggingFace Eva-02–based CLIP (vision tower); loads with trust_remote_code (EvaCLIPModel).
EVA02_CLIP_EMBEDDER_MODEL = "microsoft/LLM2CLIP-EVA02-B-16"


def _is_evaclip_config(config: Any) -> bool:
    arch = getattr(config, "architectures", None) or []
    return any(a and "EvaCLIP" in a for a in arch)


def _inputs_for_get_image_features(inputs: dict) -> dict[str, Any]:
    """Vision-only kwargs for get_image_features (avoids passing text keys from CLIPProcessor)."""
    if "pixel_values" not in inputs:
        return inputs
    return {"pixel_values": inputs["pixel_values"]}


def _is_evaclip_model(model: Any) -> bool:
    n = type(model).__name__
    return "EvaCLIP" in n or "EvaClip" in n


def _processor_inputs(processor: Any, images: list, *, model: Any) -> Any:
    """CLIPProcessor accepts ``padding=True``; CLIPImageProcessor (Eva fallback) does not."""
    if _is_evaclip_model(model):
        return processor(images=images, return_tensors="pt")
    return processor(images=images, return_tensors="pt", padding=True)


def _evaclip_get_image_feature_tensor(model: Any, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Hub Eva checkpoints differ: Microsoft LLM2CLIP exposes ``get_image_features``;
    BAAI/EVA-CLIP-18B uses ``encode_image``. Fall back to ``vision_model`` + ``visual_projection``.
    """
    if hasattr(model, "get_image_features"):
        out = model.get_image_features(pixel_values=pixel_values)
    elif hasattr(model, "encode_image"):
        out = model.encode_image(pixel_values=pixel_values)
    else:
        vm = getattr(model, "vision_model", None)
        if vm is None:
            raise TypeError(
                f"{type(model).__name__} has no get_image_features and no vision_model; "
                "cannot compute image embeddings."
            )
        vm_out = vm(pixel_values=pixel_values, return_dict=True)
        pooled = vm_out.pooler_output
        if pooled is None:
            pooled = vm_out.last_hidden_state[:, 0, :]
        proj = getattr(model, "visual_projection", None)
        out = proj(pooled) if proj is not None else pooled

    if hasattr(out, "image_embeds"):
        out = out.image_embeds
    elif hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state[:, 0]
    return out


def _evaclip_resize_pixel_values(model: Any, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Eva vision embeddings use fixed position ids for ``(image_size // patch_size) ** 2 + 1`` tokens.
    Inputs must be exactly ``image_size × image_size`` or the patch grid no longer matches
    ``position_embedding`` (IndexError in ``nn.Embedding``).
    """
    vc = model.config.vision_config
    size = int(getattr(vc, "image_size", 224))
    _, _, h, w = pixel_values.shape
    if h == size and w == size:
        return pixel_values
    return F.interpolate(pixel_values, size=(size, size), mode="bicubic", align_corners=False)


def _get_image_features(
    model: Any,
    pixel_values: torch.Tensor,
    device: str,
) -> Any:
    """
    Run the vision tower. EvaCLIP on CUDA can hit cuBLAS/TF32/driver issues; we use fp32 inputs,
    disable autocast, and temporarily turn off TF32 for matmul (more stable, slightly slower).
    """
    if _is_evaclip_model(model):
        pixel_values = pixel_values.to(device=device, dtype=torch.float32)
        pixel_values = _evaclip_resize_pixel_values(model, pixel_values)
        if device == "cuda":
            old_mm = torch.backends.cuda.matmul.allow_tf32
            old_cd = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            try:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=False):
                        return _evaclip_get_image_feature_tensor(model, pixel_values)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_mm
                torch.backends.cudnn.allow_tf32 = old_cd
        with torch.no_grad():
            return _evaclip_get_image_feature_tensor(model, pixel_values)

    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        return model.get_image_features(pixel_values=pixel_values)


def _image_processor_for_eva(_model_name: str, config: Any) -> CLIPImageProcessor:
    """
    Build preprocessing from ``vision_config`` only.

    Do not use ``AutoProcessor.from_pretrained`` on the model repo: repos like
    ``microsoft/LLM2CLIP-EVA02-B-16`` may resolve to a processor whose output spatial
    size does not match the Eva vision tower, which breaks ``position_embedding``.
    """
    vc = config.vision_config
    size = int(getattr(vc, "image_size", 224))
    patch = int(getattr(vc, "patch_size", 16))
    if patch >= 16:
        ref = "openai/clip-vit-base-patch16"
    elif size >= 300:
        ref = "openai/clip-vit-large-patch14-336"
    else:
        ref = "openai/clip-vit-large-patch14"
    return CLIPImageProcessor.from_pretrained(
        ref,
        size={"shortest_edge": size},
        crop_size={"height": size, "width": size},
    )


def _fix_evaclip_vision_position_ids(model: Any) -> None:
    """
    ``position_ids`` must match ``position_embedding.num_embeddings``. A desynced buffer
    (e.g. wrong length or max index vs. weight rows) causes ``IndexError`` in ``nn.Embedding``.
    """
    emb = model.vision_model.embeddings
    n = emb.position_embedding.num_embeddings
    dev = emb.position_embedding.weight.device
    emb.position_ids = torch.arange(n, device=dev, dtype=torch.long).view(1, -1)


def _patch_evaclip_vision_embeddings_forward(model: Any) -> None:
    """
    Upstream adds ``[B, seq, D] + position_emb([1, npos, D])`` relying on ``seq == npos``.
    Slice positional embeddings to ``seq`` (same pattern as EvaCLIP text embeddings) so
    lengths always match and we fail with a clear error if the patch grid is wrong.
    """
    emb = model.vision_model.embeddings

    def forward(pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = emb.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = emb.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        seq_len = embeddings.shape[1]
        npos = emb.position_embedding.num_embeddings
        if seq_len > npos:
            raise ValueError(
                f"EvaCLIP vision: seq_len={seq_len} but position_embedding has {npos} rows. "
                f"pixel_values={tuple(pixel_values.shape)}, "
                f"image_size={emb.image_size}, patch_size={emb.patch_size}."
            )
        pos_ids = emb.position_ids[:, :seq_len]
        return embeddings + emb.position_embedding(pos_ids)

    emb.forward = forward  # type: ignore[method-assign]


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image and convert to RGB."""
    return Image.open(image_path).convert("RGB")


def compute_cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    # Normalize the embeddings
    embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    similarity = torch.sum(embedding1 * embedding2, dim=-1)
    return similarity.item()


def compute_embedding(
    image_path: str | Path,
    model: Any,
    processor: Any,
    device: str,
) -> torch.Tensor:
    """
    Compute the CLIP embedding for a single image.
    
    Args:
        image_path: Path to the image.
        model: Pre-loaded CLIP model.
        processor: Pre-loaded CLIP processor.
        device: Device the model is on.
    
    Returns:
        Normalized embedding tensor (1D).
    """
    image = load_image(image_path)
    
    inputs = _processor_inputs(processor, [image], model=model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pv = _inputs_for_get_image_features(inputs)["pixel_values"]
    features = _get_image_features(model, pv, device)
    
    # Handle both old API (returns tensor) and new API (returns object with image_embeds)
    if hasattr(features, 'image_embeds'):
        features = features.image_embeds
    elif hasattr(features, 'last_hidden_state'):
        features = features.last_hidden_state[:, 0]  # Take CLS token
    # else: features is already a tensor (old API)
    
    # Normalize and return
    embedding = features[0]
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def compute_embeddings_batch(
    image_paths: list[str | Path],
    model: Any,
    processor: Any,
    device: str,
    batch_size: int = 8,
) -> dict[Path, torch.Tensor]:
    """
    Compute CLIP embeddings for multiple images efficiently.
    
    Args:
        image_paths: List of paths to images.
        model: Pre-loaded CLIP model.
        processor: Pre-loaded CLIP processor.
        device: Device the model is on.
        batch_size: Number of images to process at once.
    
    Returns:
        Dictionary mapping image paths to their normalized embeddings.
    """
    embeddings = {}
    image_paths = [Path(p) for p in image_paths]
    # EvaCLIP: smaller batches reduce peak memory and avoid some cuBLAS edge cases on CUDA.
    eff_bs = batch_size
    if _is_evaclip_model(model) and device == "cuda":
        eff_bs = max(1, min(batch_size, 4))

    # Process in batches for GPU efficiency
    for i in range(0, len(image_paths), eff_bs):
        batch_paths = image_paths[i : i + eff_bs]
        batch_images = [load_image(p) for p in batch_paths]
        
        inputs = _processor_inputs(processor, batch_images, model=model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        pv = _inputs_for_get_image_features(inputs)["pixel_values"]
        features = _get_image_features(model, pv, device)
        
        # Handle both old API (returns tensor) and new API (returns object with image_embeds)
        if hasattr(features, 'image_embeds'):
            features = features.image_embeds
        elif hasattr(features, 'last_hidden_state'):
            features = features.last_hidden_state[:, 0]  # Take CLS token
        # else: features is already a tensor (old API)
        
        # Normalize and store
        features = features / features.norm(dim=-1, keepdim=True)
        
        for j, path in enumerate(batch_paths):
            embeddings[path] = features[j].cpu()  # Move to CPU to save GPU memory
    
    return embeddings


def similarity_from_embeddings(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
) -> float:
    """
    Compute cosine similarity from pre-computed normalized embeddings.
    
    Args:
        embedding1: First normalized embedding.
        embedding2: Second normalized embedding.
    
    Returns:
        Cosine similarity (float).
    """
    # Embeddings should already be normalized, but ensure they're on same device
    similarity = torch.sum(embedding1 * embedding2, dim=-1)
    return similarity.item()


def load_clip_model(
    model_name: str = "apple/DFN5B-CLIP-ViT-H-14-378",
    device: str | None = None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any, str]:
    """
    Load a CLIP or EvaCLIP model and image processor.

    Standard CLIP uses ``CLIPModel`` / ``CLIPProcessor``. Eva-02 CLIP checkpoints
    (e.g. ``microsoft/LLM2CLIP-EVA02-B-16``) use ``EvaCLIPModel`` with
    ``trust_remote_code=True`` and a CLIP-style image processor.

    Args:
        model_name: HuggingFace model ID for CLIP or EvaCLIP.
        device: Device to run on (auto-detected if None).
        trust_remote_code: Passed through for EvaCLIP and other custom checkpoints.

    Returns:
        Tuple of (model, processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    except OSError:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if _is_evaclip_config(config):
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        model = model.to(device)
        model.eval()
        # fp32 avoids half-precision / cuBLAS issues on some GPUs with Eva attention.
        if device == "cuda":
            model = model.float()
        _fix_evaclip_vision_position_ids(model)
        _patch_evaclip_vision_embeddings_forward(model)
        processor = _image_processor_for_eva(model_name, config)
        return model, processor, device

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return model, processor, device


def compare_images(
    image1_path: str | Path,
    image2_path: str | Path,
    model_name: str = "apple/DFN5B-CLIP-ViT-H-14-378",
    device: str | None = None,
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
) -> float:
    """
    Compare two images using CLIP and return their cosine similarity.
    
    Args:
        image1_path: Path to the first image.
        image2_path: Path to the second image.
        model_name: HuggingFace model ID for CLIP (ignored if model/processor provided).
        device: Device to run on (auto-detected if None).
        model: Pre-loaded CLIP model (optional, for reuse across calls).
        processor: Pre-loaded CLIP processor (optional, for reuse across calls).
    
    Returns:
        Cosine similarity between the two images (range: -1 to 1, typically 0 to 1).
    """
    # Track whether we loaded models ourselves (for cleanup)
    loaded_locally = model is None or processor is None
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)
    
    # Load CLIP model and processor if not provided
    if model is None or processor is None:
        model, processor, device = load_clip_model(model_name, device)
    
    # Process images
    inputs = _processor_inputs(processor, [image1, image2], model=model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pv = _inputs_for_get_image_features(inputs)["pixel_values"]
    image_features = _get_image_features(model, pv, device)
    
    # Handle both old API (returns tensor) and new API (returns object with image_embeds)
    if hasattr(image_features, 'image_embeds'):
        image_features = image_features.image_embeds
    elif hasattr(image_features, 'last_hidden_state'):
        image_features = image_features.last_hidden_state[:, 0]  # Take CLS token
    # else: image_features is already a tensor (old API)
    
    # Compute cosine similarity
    similarity = compute_cosine_similarity(image_features[0], image_features[1])
    
    # Clean up only if we loaded models ourselves
    if loaded_locally:
        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return similarity


def main() -> None:
    """CLI entry point for image comparison."""
    parser = argparse.ArgumentParser(
        description="Compare two images using CLIP embeddings and compute cosine similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two images
  %(prog)s photo1.png photo2.png
  
  # Use an OpenAI CLIP model
  %(prog)s photo1.png photo2.png --model openai/clip-vit-large-patch14
  
  # Force CPU inference
  %(prog)s photo1.png photo2.png --device cpu

Interpretation of similarity scores:
  0.90 - 1.00: Nearly identical images or very similar content
  0.80 - 0.90: Very similar images (same subject, different angle/lighting)
  0.70 - 0.80: Similar images (related content or style)
  0.60 - 0.70: Somewhat related images
  0.50 - 0.60: Loosely related images
  < 0.50:      Different images

Available CLIP models:
  - apple/DFN5B-CLIP-ViT-H-14-378 (default, high quality)
  - openai/clip-vit-base-patch32 (fastest)
  - openai/clip-vit-base-patch16 (more accurate)
  - openai/clip-vit-large-patch14 (most accurate OpenAI model)
  - microsoft/LLM2CLIP-EVA02-B-16 (EVA-02 CLIP vision encoder, trust_remote_code)
        """,
    )
    
    parser.add_argument(
        "image1",
        type=str,
        help="Path to the first image.",
    )
    parser.add_argument(
        "image2",
        type=str,
        help="Path to the second image.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="apple/DFN5B-CLIP-ViT-H-14-378",
        help="CLIP model to use (default: apple/DFN5B-CLIP-ViT-H-14-378).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect).",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    image1_path = Path(args.image1)
    image2_path = Path(args.image2)
    
    if not image1_path.exists():
        parser.error(f"Image not found: {image1_path}")
    if not image2_path.exists():
        parser.error(f"Image not found: {image2_path}")
    
    # Compare images
    print(f"Loading CLIP model: {args.model}")
    print(f"Comparing images:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    
    similarity = compare_images(
        image1_path=image1_path,
        image2_path=image2_path,
        model_name=args.model,
        device=args.device,
    )
    
    print(f"\nCosine Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()

