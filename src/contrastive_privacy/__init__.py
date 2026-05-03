"""
Contrastive Privacy: a semantic privacy test for AI-based sanitization.

The simplest entry point is :func:`contrastive_privacy_test`, which lets you
plug in any sanitizer and any embedder and measure whether your AI actually
hides what you wanted hidden.

    from contrastive_privacy import contrastive_privacy_test

    result = contrastive_privacy_test(
        items=my_proxy_set,
        sanitize=my_sanitizer,
        embed=my_embedder,
    )
    print(result.summary())

For end-to-end image and text experiments — including the frontier-model
benchmarks reported in the paper — see the ``resolution-analysis`` and
``contrastive_privacy.scripts.text_resolution_analysis`` console scripts.
"""

from contrastive_privacy.api import (
    ContrastivePrivacyResult,
    FailurePair,
    contrastive_privacy_test,
    cosine_distance,
)

__version__ = "0.1.0"

__all__ = [
    "contrastive_privacy_test",
    "ContrastivePrivacyResult",
    "FailurePair",
    "cosine_distance",
    "__version__",
]
