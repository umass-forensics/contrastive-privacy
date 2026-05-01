#!/usr/bin/env python3
"""
Compare two texts using embeddings and compute their cosine similarity.

This script supports multiple embedding models for text comparison:
- CLIP: Uses OpenAI's CLIP model text encoder for semantic embeddings
- SBERT: Uses Sentence-BERT for specialized sentence embeddings

Higher scores indicate more semantically similar texts.

Example:
    compare-texts "The cat sat on the mat." "A feline rested on the rug."
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch


def compute_cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    # Normalize the embeddings
    embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
    embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    similarity = torch.sum(embedding1 * embedding2, dim=-1)
    return similarity.item()


EVA_CLIP_TEXT_EMBEDDER_EXAMPLE = "BAAI/EVA-CLIP-18B"
DEFAULT_QWEN_EMBEDDER_MODEL = "Qwen/Qwen3-Embedding-8B"


def _is_evaclip_config(config: Any) -> bool:
    arch = getattr(config, "architectures", None) or []
    return any(a and "EvaCLIP" in a for a in arch)


def _is_evaclip_model(model: Any) -> bool:
    name = type(model).__name__
    return "EvaCLIP" in name or "EvaClip" in name


def _fix_evaclip_text_position_ids(model: Any) -> None:
    """
    Sync the text model's ``position_ids`` buffer with ``position_embedding.num_embeddings``.
    A desynced buffer causes ``IndexError`` in ``nn.Embedding`` (same issue as the vision tower).
    """
    text_model = getattr(model, "text_model", None)
    if text_model is None:
        return
    emb = getattr(text_model, "embeddings", None)
    if emb is None:
        return
    pos_emb = getattr(emb, "position_embedding", None)
    if pos_emb is None:
        return
    n = pos_emb.num_embeddings
    dev = pos_emb.weight.device
    emb.position_ids = torch.arange(n, device=dev, dtype=torch.long).view(1, -1)


def _evaclip_max_text_length(config: Any) -> int:
    """Return the maximum text context length from an EvaCLIP config."""
    tc = getattr(config, "text_config", None)
    if tc is not None:
        val = getattr(tc, "max_position_embeddings", None)
        if val is not None:
            return int(val)
    return 77


class TextEmbedder(ABC):
    """Abstract base class for text embedding models."""
    
    @abstractmethod
    def embed(self, text: str) -> torch.Tensor:
        """Compute embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Compute embeddings for multiple texts."""
        pass

    def embed_batch_ordered(
        self, texts: list[str], batch_size: int = 32
    ) -> list[torch.Tensor]:
        """
        Compute one embedding per list slot (duplicate strings are embedded separately).

        Default is slow (one forward per text); CLIP/SBERT override with batched code.
        """
        del batch_size
        return [self.embed(t) for t in texts]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return similarity_from_embeddings(emb1, emb2)


class CLIPTextEmbedder(TextEmbedder):
    """Text embedder using CLIP's text encoder."""
    
    def __init__(
        self,
        model_name: str = "apple/DFN5B-CLIP-ViT-H-14-378",
        device: Optional[str] = None,
    ):
        """
        Initialize CLIP text embedder.
        
        Args:
            model_name: HuggingFace model ID for CLIP.
            device: Device to run on (auto-detect if None).
        """
        from transformers import AutoConfig, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device

        # EVA-CLIP checkpoints (e.g., BAAI/EVA-CLIP-18B) require AutoModel with
        # trust_remote_code, while standard CLIP uses CLIPModel/CLIPProcessor.
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        except OSError:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        self._is_eva = _is_evaclip_config(config)
        if self._is_eva:
            self._max_length = _evaclip_max_text_length(config)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            try:
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                self.processor = CLIPProcessor.from_pretrained(model_name)
        else:
            self._max_length = 77
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model = self.model.to(device)
        self.model.eval()

        if self._is_eva:
            _fix_evaclip_text_position_ids(self.model)
            if device == "cuda":
                self.model = self.model.float()
    
    def _evaclip_text_features_inner(self, text_inputs: dict) -> torch.Tensor:
        """Run the EVA text tower, trying multiple API patterns."""
        if hasattr(self.model, "get_text_features"):
            return self.model.get_text_features(**text_inputs)
        if hasattr(self.model, "encode_text"):
            return self.model.encode_text(**text_inputs)
        text_model = getattr(self.model, "text_model", None)
        if text_model is None:
            raise TypeError(
                f"{type(self.model).__name__} has no get_text_features/encode_text/text_model; "
                "cannot compute text embeddings."
            )
        text_outputs = text_model(
            input_ids=text_inputs.get("input_ids"),
            attention_mask=text_inputs.get("attention_mask"),
        )
        pooled = text_outputs[1]
        projection = getattr(self.model, "text_projection", None)
        return projection(pooled) if projection is not None else pooled

    def _evaclip_text_features(self, text_inputs: dict) -> torch.Tensor:
        """Run EVA text tower with CUDA stability measures (fp32, no TF32)."""
        text_inputs = {k: v.to(device=self.device, dtype=torch.float32)
                       if v.is_floating_point() else v.to(self.device)
                       for k, v in text_inputs.items()}
        if self.device == "cuda":
            old_mm = torch.backends.cuda.matmul.allow_tf32
            old_cd = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            try:
                with torch.autocast(device_type="cuda", enabled=False):
                    return self._evaclip_text_features_inner(text_inputs)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_mm
                torch.backends.cudnn.allow_tf32 = old_cd
        return self._evaclip_text_features_inner(text_inputs)

    def _get_text_features(self, inputs: dict) -> torch.Tensor:
        """
        Extract text features from CLIP, handling both old and new
        transformers versions.

        Newer transformers versions may return a BaseModelOutputWithPooling
        from get_text_features instead of a plain tensor.  When that happens
        we fall back to calling the text model + projection explicitly.
        """
        text_inputs = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}

        if _is_evaclip_model(self.model):
            features = self._evaclip_text_features(text_inputs)
        else:
            features = self.model.get_text_features(**text_inputs)

        if isinstance(features, torch.Tensor):
            return features
        # Fallback: manually run text model -> projection
        text_outputs = self.model.text_model(
            input_ids=text_inputs.get("input_ids"),
            attention_mask=text_inputs.get("attention_mask"),
        )
        return self.model.text_projection(text_outputs[1])

    def _tokenize(self, texts: list[str]) -> dict:
        """Tokenize texts with the correct max_length and processor kwargs."""
        kwargs: dict[str, Any] = dict(
            text=texts,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        if not self._is_eva:
            kwargs["padding"] = True
        else:
            kwargs["padding"] = "max_length"
        inputs = self.processor(**kwargs)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def embed(self, text: str) -> torch.Tensor:
        """Compute CLIP embedding for a single text."""
        inputs = self._tokenize([text])

        with torch.no_grad():
            features = self._get_text_features(inputs)

        embedding = features[0]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu()

    def embed_batch_ordered(
        self, texts: list[str], batch_size: int = 32
    ) -> list[torch.Tensor]:
        """Compute CLIP embeddings for multiple texts, preserving list order."""
        out: list[torch.Tensor] = []
        eff_bs = batch_size
        if self._is_eva and self.device == "cuda":
            eff_bs = max(1, min(batch_size, 4))

        for i in range(0, len(texts), eff_bs):
            batch = texts[i : i + eff_bs]
            inputs = self._tokenize(batch)

            with torch.no_grad():
                features = self._get_text_features(inputs)

            features = features / features.norm(dim=-1, keepdim=True)

            for j in range(len(batch)):
                out.append(features[j].cpu())

        return out

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> dict[str, torch.Tensor]:
        """Compute CLIP embeddings for multiple texts (dict keys: last slot wins on duplicate text)."""
        ordered = self.embed_batch_ordered(texts, batch_size=batch_size)
        return {t: ordered[i] for i, t in enumerate(texts)}


def _normalize_embedder_quantization(q: Optional[str]) -> Optional[str]:
    """
    Return None if off, else ``\"4bit\"``, ``\"8bit\"``, or ``\"half\"`` (fp16, no bitsandbytes).
    """
    if q is None or str(q).strip().lower() in ("", "none", "off", "false"):
        return None
    s = str(q).strip().lower()
    if s in ("4bit", "4-bit", "nf4"):
        return "4bit"
    if s in ("8bit", "8-bit"):
        return "8bit"
    if s in ("half", "fp16", "float16"):
        return "half"
    raise ValueError(
        f"Unknown embedder quantization {q!r}; use 'none', 'half', '4bit', or '8bit'."
    )


class SBERTEmbedder(TextEmbedder):
    """Text embedder using Sentence-BERT."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        quantization: Optional[str] = None,
    ):
        """
        Initialize Sentence-BERT embedder.

        Args:
            model_name: Model name from sentence-transformers.
            device: Device to run on (auto-detect if None).
            quantization: None for full precision; ``\"half\"`` for float16 weights (no
                bitsandbytes); ``\"4bit\"`` / ``\"8bit\"`` for bitsandbytes (CUDA only).
        """
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        q = _normalize_embedder_quantization(quantization)
        trust_remote_code = "qwen" in model_name.lower()

        if q is None:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=trust_remote_code,
            )
        elif q == "half":
            # Do not use ``device_map="auto"`` here: it can leave weights on meta / CPU offload,
            # then SentenceTransformer calls ``.to(device)`` and fails (meta tensor copy).
            # Plain fp16 load onto ``device`` matches the non-quantized path and stays on one device.
            self.model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs={"torch_dtype": torch.float16},
                trust_remote_code=trust_remote_code,
            )
        elif q in ("4bit", "8bit"):
            if device != "cuda":
                raise ValueError(
                    f"Sentence-transformers {q} quantization requires CUDA (got device={device!r})."
                )
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError(
                    "Quantized embedders require transformers with BitsAndBytesConfig."
                ) from e
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes is required for 4-bit/8-bit --embedder-quantization. "
                    "Install: pip install bitsandbytes  (or reinstall this package: pip install -e .)"
                ) from e

            compute_dtype = torch.float16
            if q == "4bit":
                bnb_kwargs: dict[str, Any] = {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    ),
                    "device_map": "auto",
                }
            else:
                # Use BitsAndBytesConfig for 8-bit too: raw ``load_in_8bit`` in model_kwargs
                # can be forwarded into ``Qwen3Model.__init__`` and raise TypeError on newer stacks.
                # ``torch_dtype=float16`` aligns with bitsandbytes MatMul8bitLt (fp16 path) and avoids
                # noisy casts when the checkpoint would otherwise default to bfloat16.
                bnb_kwargs = {
                    "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                    "device_map": "auto",
                    "torch_dtype": compute_dtype,
                }

            self.model = SentenceTransformer(
                model_name,
                device="cuda",
                model_kwargs=bnb_kwargs,
                trust_remote_code=trust_remote_code,
            )
        else:
            raise ValueError(f"Unsupported quantization mode: {q!r}")
    
    def embed(self, text: str) -> torch.Tensor:
        """Compute SBERT embedding for a single text."""
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embedding.cpu()
    
    def embed_batch_ordered(
        self, texts: list[str], batch_size: int = 32
    ) -> list[torch.Tensor]:
        """Compute SBERT embeddings for multiple texts, preserving list order."""
        if not texts:
            return []
        embeddings_tensor = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
        )
        return [embeddings_tensor[i].cpu() for i in range(len(texts))]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> dict[str, torch.Tensor]:
        """Compute SBERT embeddings for multiple texts (dict keys: last slot wins on duplicate text)."""
        ordered = self.embed_batch_ordered(texts, batch_size=batch_size)
        return {t: ordered[i] for i, t in enumerate(texts)}


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
    similarity = torch.sum(embedding1 * embedding2, dim=-1)
    return similarity.item()


def load_text_embedder(
    model_type: str = "sbert",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    embedder_quantization: Optional[str] = None,
) -> TextEmbedder:
    """
    Load a text embedding model.

    Args:
        model_type: "clip", "sbert", or "qwen".
        model_name: Model name (uses default if None).
        device: Device to run on.
        embedder_quantization: For sbert/qwen only: ``\"half\"`` (fp16, no bitsandbytes),
            ``\"4bit\"`` / ``\"8bit\"`` (bitsandbytes, CUDA), or None / ``\"none\"``.

    Returns:
        TextEmbedder instance.
    """
    q = _normalize_embedder_quantization(embedder_quantization)
    if model_type == "clip":
        if q is not None:
            raise ValueError(
                "embedder_quantization is only supported for sbert and qwen, not clip."
            )
        model_name = model_name or "apple/DFN5B-CLIP-ViT-H-14-378"
        return CLIPTextEmbedder(model_name=model_name, device=device)
    elif model_type == "sbert":
        model_name = model_name or "all-MiniLM-L6-v2"
        return SBERTEmbedder(model_name=model_name, device=device, quantization=q)
    elif model_type == "qwen":
        model_name = model_name or DEFAULT_QWEN_EMBEDDER_MODEL
        return SBERTEmbedder(model_name=model_name, device=device, quantization=q)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Choose 'clip', 'sbert', or 'qwen'."
        )


def compute_embedding(
    text: str,
    embedder: TextEmbedder,
) -> torch.Tensor:
    """
    Compute embedding for a single text.
    
    Args:
        text: Input text.
        embedder: Text embedder instance.
    
    Returns:
        Normalized embedding tensor (1D).
    """
    return embedder.embed(text)


def compute_embeddings_batch(
    texts: list[str],
    embedder: TextEmbedder,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Compute embeddings for multiple texts efficiently.
    
    Args:
        texts: List of texts.
        embedder: Text embedder instance.
        batch_size: Number of texts to process at once.
    
    Returns:
        Dictionary mapping texts to their normalized embeddings.
    """
    return embedder.embed_batch(texts, batch_size=batch_size)


def compare_texts(
    text1: str,
    text2: str,
    model_type: str = "sbert",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    embedder: Optional[TextEmbedder] = None,
) -> float:
    """
    Compare two texts using embeddings and return their cosine similarity.
    
    Args:
        text1: First text.
        text2: Second text.
        model_type: "clip" or "sbert" (ignored if embedder provided).
        model_name: Model name (ignored if embedder provided).
        device: Device to run on (ignored if embedder provided).
        embedder: Pre-loaded embedder (optional, for reuse across calls).
    
    Returns:
        Cosine similarity between the two texts (range: -1 to 1, typically 0 to 1).
    """
    # Track whether we loaded embedder ourselves (for cleanup)
    loaded_locally = embedder is None
    
    if embedder is None:
        embedder = load_text_embedder(
            model_type=model_type,
            model_name=model_name,
            device=device,
        )
    
    similarity = embedder.similarity(text1, text2)
    
    # Note: Unlike CLIP models, we don't need to explicitly clean up
    # since PyTorch will handle garbage collection
    
    return similarity


def main() -> None:
    """CLI entry point for text comparison."""
    parser = argparse.ArgumentParser(
        description="Compare two texts using embeddings and compute cosine similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two texts using SBERT (default)
  %(prog)s "The cat sat on the mat." "A feline rested on the rug."
  
  # Compare using CLIP text encoder
  %(prog)s "A photo of a cat" "An image of a feline" --model-type clip
  
  # Use a specific SBERT model
  %(prog)s "Hello world" "Hi there" --model all-mpnet-base-v2
  
  # Compare texts from files
  %(prog)s --file1 text1.txt --file2 text2.txt

MODEL TYPES:
  sbert (default):
    - Specialized for sentence similarity
    - Better for comparing similar length texts
    - Faster and more lightweight
    - Models: all-MiniLM-L6-v2 (default), all-mpnet-base-v2, etc.
    
  clip:
    - Trained for image-text matching
    - Good for shorter descriptive texts
    - Same embedding space as CLIP images
    - Limited to 77 tokens
    - Models: apple/DFN5B-CLIP-ViT-H-14-378 (default)

Interpretation of similarity scores:
  0.90 - 1.00: Nearly identical meaning
  0.80 - 0.90: Very similar (paraphrases)
  0.70 - 0.80: Similar topics/content
  0.60 - 0.70: Related content
  0.50 - 0.60: Loosely related
  < 0.50:      Different topics

SBERT MODELS (sentence-transformers):
  - all-MiniLM-L6-v2 (default, fast, good quality)
  - all-mpnet-base-v2 (better quality, slower)
  - paraphrase-MiniLM-L6-v2 (optimized for paraphrase detection)
  - multi-qa-MiniLM-L6-cos-v1 (optimized for Q&A)
        """,
    )
    
    parser.add_argument(
        "text1",
        type=str,
        nargs="?",
        default=None,
        help="First text to compare (or use --file1).",
    )
    parser.add_argument(
        "text2",
        type=str,
        nargs="?",
        default=None,
        help="Second text to compare (or use --file2).",
    )
    parser.add_argument(
        "--file1", "-f1",
        type=str,
        default=None,
        help="Path to first text file.",
    )
    parser.add_argument(
        "--file2", "-f2",
        type=str,
        default=None,
        help="Path to second text file.",
    )
    parser.add_argument(
        "--model-type", "-t",
        type=str,
        default="sbert",
        choices=["clip", "sbert", "qwen"],
        help=(
            "Embedding model type: 'sbert' (sentence-transformers), "
            "'qwen' (Qwen/Qwen3-Embedding-8B via sentence-transformers), or 'clip'. "
            "Default: sbert."
        ),
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (default depends on model type).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect).",
    )
    
    args = parser.parse_args()
    
    # Get input texts
    if args.file1:
        text1 = Path(args.file1).read_text(encoding="utf-8").strip()
    elif args.text1:
        text1 = args.text1
    else:
        parser.error("Either text1 argument or --file1 is required")
    
    if args.file2:
        text2 = Path(args.file2).read_text(encoding="utf-8").strip()
    elif args.text2:
        text2 = args.text2
    else:
        parser.error("Either text2 argument or --file2 is required")
    
    # Compare texts
    print(f"Using {args.model_type} model: {args.model or '(default)'}")
    print(f"Comparing texts:")
    print(f"  Text 1: \"{text1[:80]}{'...' if len(text1) > 80 else ''}\"")
    print(f"  Text 2: \"{text2[:80]}{'...' if len(text2) > 80 else ''}\"")
    
    similarity = compare_texts(
        text1=text1,
        text2=text2,
        model_type=args.model_type,
        model_name=args.model,
        device=args.device,
    )
    
    print(f"\nCosine Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
