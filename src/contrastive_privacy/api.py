"""
High-level Python API for the contrastive privacy test.

This module exposes a single function, ``contrastive_privacy_test``, that lets
you measure whether *your* sanitizer actually hides *your* abstract privacy
concept on *your* data — without writing a CLI command, without manual labels,
and without committing to a particular model or modality.

Example
-------

    from contrastive_privacy import contrastive_privacy_test

    def my_sanitizer(item):
        ...   # your AI: blackout, blur, inpaint, LLM rewrite, anything

    def my_embedder(item):
        ...   # any model that returns a vector for cosine similarity

    result = contrastive_privacy_test(
        items=load_my_proxy_set(),
        sanitize=my_sanitizer,
        embed=my_embedder,
        delta=0.0,
    )

    print(result.summary())
    for f in result.top_failures(5):
        print(f)

The test compares each sanitized item against every other item's original and
sanitized counterpart. A pair (i, j) is a privacy leak iff

    distance(X(i), original(j))  <  distance(X(i), X(j))

i.e. the sanitized rendering still resembles the un-sanitized peer more than
its sanitized peer — which means something semantic about the secret survived
the sanitization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "contrastive_privacy_test",
    "ContrastivePrivacyResult",
    "FailurePair",
    "cosine_distance",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailurePair:
    """A single privacy leak: two items in the proxy set whose sanitized forms
    are bound together by something that should have been removed."""

    i: int
    j: int
    distance_to_original: float
    distance_to_sanitized: float

    @property
    def leak(self) -> float:
        """Positive => leak. Equals ``distance_to_sanitized - distance_to_original``."""
        return self.distance_to_sanitized - self.distance_to_original


@dataclass
class ContrastivePrivacyResult:
    """Outcome of the contrastive privacy test on a proxy set."""

    passed: bool
    delta: float
    worst_leak: float
    failures: List[FailurePair] = field(default_factory=list)
    n_items: int = 0
    n_pairs: int = 0

    def top_failures(self, k: int = 5) -> List[FailurePair]:
        """Return the ``k`` worst failures, sorted by leak magnitude."""
        return sorted(self.failures, key=lambda f: -f.leak)[:k]

    def summary(self) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        return (
            f"Contrastive privacy test {verdict} at delta={self.delta:.3f}\n"
            f"  items checked        : {self.n_items}\n"
            f"  pairs compared       : {self.n_pairs}\n"
            f"  worst observed leak  : {self.worst_leak:.4f}\n"
            f"  privacy failures     : {len(self.failures)}"
        )


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1-D vectors. Returns a value in [0, 2]."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine_distance received a zero-norm vector.")
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def contrastive_privacy_test(
    items: Sequence[Any],
    sanitize: Callable[[Any], Any],
    embed: Callable[[Any], np.ndarray],
    delta: float = 0.0,
    distance: Callable[[np.ndarray, np.ndarray], float] = cosine_distance,
    sanitized_items: Optional[Sequence[Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ContrastivePrivacyResult:
    """Run the contrastive privacy test on a proxy set.

    Parameters
    ----------
    items
        A proxy set: a sequence of renderings (images, text, audio paths,
        whatever) that all capture the abstract concept you want to hide.
        30–50 representative items is usually enough.
    sanitize
        A callable that turns one item into its sanitized form. The framework
        treats this as a black box; it can be a blackout pipeline, an LLM
        rewrite, a SAM-based redactor, an API call, anything.
    embed
        A callable that turns one item (original or sanitized) into a 1-D
        ``np.ndarray``. Must be deterministic and live in the same vector
        space for both originals and sanitized outputs.
    delta
        Privacy resolution. ``0.0`` is ideal privacy: the test fails on any
        leak. Larger values are more lenient.
    distance
        Distance over embeddings. Defaults to cosine distance.
    sanitized_items
        If you've already run your sanitizer, pass the outputs here to avoid
        running it again. Must align 1-1 with ``items``.
    progress
        Optional ``(current, total)`` callback for long sanitizations.

    Returns
    -------
    ContrastivePrivacyResult
        Includes the pass/fail verdict, the worst leak observed, and the full
        list of failing pairs you can inspect to diagnose what your sanitizer
        missed.
    """
    items = list(items)
    n = len(items)
    if n < 2:
        raise ValueError("Proxy set must have at least 2 items.")
    if delta < 0:
        raise ValueError("delta must be non-negative.")

    # 1. Sanitize.
    if sanitized_items is None:
        sanitized: List[Any] = []
        for k, x in enumerate(items):
            sanitized.append(sanitize(x))
            if progress is not None:
                progress(k + 1, n)
    else:
        sanitized = list(sanitized_items)
        if len(sanitized) != n:
            raise ValueError("sanitized_items must align 1-1 with items.")

    # 2. Embed both views.
    E_orig = np.stack([np.asarray(embed(x)).ravel() for x in items])
    E_san = np.stack([np.asarray(embed(x)).ravel() for x in sanitized])

    # 3. All-pairs check of Inequality 4 in cosine-distance form.
    failures: List[FailurePair] = []
    worst = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_to_orig = distance(E_san[i], E_orig[j])
            d_to_san = distance(E_san[i], E_san[j])
            leak = d_to_san - d_to_orig
            if leak > worst:
                worst = leak
            if d_to_orig + delta <= d_to_san:
                failures.append(
                    FailurePair(
                        i=i,
                        j=j,
                        distance_to_original=d_to_orig,
                        distance_to_sanitized=d_to_san,
                    )
                )

    return ContrastivePrivacyResult(
        passed=len(failures) == 0,
        delta=delta,
        worst_leak=worst,
        failures=failures,
        n_items=n,
        n_pairs=n * (n - 1),
    )
