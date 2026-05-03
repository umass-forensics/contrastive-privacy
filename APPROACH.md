# Contrastive Privacy — Method & Algorithm

A self-contained explanation of the approach. Read this once and you should
have everything you need to reimplement the test against your own sanitizer
and your own embedder. For the full theory and experiments see the paper;
this document is the operational core.

---

## TL;DR

You have an AI that hides a sensitive concept (a face, a brand, a movie
title) inside a corpus of media. You want to know whether it actually
worked. **Contrastive privacy** answers that with one inequality:

> For every pair of items `x, y` in your dataset:
> the distance from sanitized `x` to original `y` should be **at least as
> large as** the distance from sanitized `x` to sanitized `y`.

If it isn't, the two items are still semantically tied together by something
the sanitizer should have removed — and you can open the failing pair to
see what.

The test needs no hand-labeled ground truth. It works for any sanitizer
(blackout, blur, inpaint, LLM rewrite) and any modality with a usable
embedding model (image, text, audio).

---

## 1. The setup

Four pieces have to be in your head before any code:

| Symbol | Plain English | Example |
|---|---|---|
| `γ` | the **abstract** concept you actually want to hide | "the identity of Leonardo DiCaprio" |
| `c` | the **natural** concept your AI can locate and remove | `face`, `logo`, `embedded text` |
| `X_c` | the sanitization mechanism you wrote/picked | SAM3 + blackout, GPT-Image edit, LLM redact |
| `D`  | a distance over renderings | cosine over CLIP / EVA / sentence embeddings |

Two more:

| Symbol | Meaning |
|---|---|
| `X̃` | the **proxy set** — a small representative sample (≈30–50 items) that all capture `γ` |
| `δ` | **resolution** — how strict the test is. `δ = 0` is ideal. Larger `δ` = more lenient |

The whole point of the framework is that you usually **cannot** enumerate
`γ` directly (you can't list every property that identifies DiCaprio), so
instead you sanitize a natural concept `c` whose semantic closure covers
`γ` — and the test certifies you picked a good enough `c`.

---

## 2. Core definitions

### 2.1 Distance mechanism `D`

Any deterministic similarity function over renderings, satisfying:

- **Symmetry**: `D(x, y) == D(y, x)`
- **Proximity**: `D(x, x) == 0`
- **Positivity**: `D(x, y) > 0` whenever `x ≠ y`

In practice: cosine distance over normalized embeddings.

### 2.2 Privacy mechanism `X_c`

A function that takes a rendering and returns a modified rendering with
concept `c` either obfuscated or replaced by another instance not in `c`.
The framework treats this as a black box. Idempotent in the sense that
`X_c(X_c(x)) == X_c(x)`.

### 2.3 Capture, abstract vs. natural

- A rendering `x` **captures** a property `p` if `p` can be inferred from
  `x` algorithmically. Equivalently, `x ≠ X_{p}(x)`.
- A **natural** concept is one whose properties can be enumerated efficiently
  (`face`, `text`, `logo`).
- An **abstract** concept is one whose properties cannot
  (`identity of person X`, `the McDonald's brand`, `the movie discussed`).

### 2.4 Semantic connection and closure

Two property sets `c, d` are **semantically connected** under `(X, D)` —
written `c ∼ d` — when including or omitting either of them measurably moves
the distance. Formally, for all `x ∈ X(c)` and `y ∈ X(d)`:

```
D(x, y) + δ < D(X_c(x), y)
D(x, y) + δ < D(x, X_d(y))
```

The **semantic closure** `cl(X, D, c)` is everything connected to `c`. The
whole approach hinges on a single working assumption:

> **`γ` lies inside the semantic closure of `c`.**

In plain English: the natural concept you remove must be rich enough that
the embedding model treats it as semantically tied to the abstract concept
you actually care about. If `c = face` and `γ = identity of John Doe`, this
is usually fine. If `c = face` and `γ = the McDonald's brand`, it isn't.

---

## 3. The privacy test (Inequality 4)

The whole framework reduces to this. For sanitization mechanism `X_c` to
offer **contrastive privacy with respect to `γ` at resolution `δ`**, the
following must hold for every pair of renderings `x, y` in `X̃(γ ∩ c)`:

```
D(X_c(x), y) + δ  >  D(X_c(x), X_c(y))
```

Read it as: a sanitized item should be at least as close to a peer's
**sanitized** version as to that peer's **original**. If that flips for
any pair, the sanitization left a residual semantic tie — a leak — and the
pair tells you what.

### Why this works (intuition)

If the sanitizer truly erased everything in `γ`, the only properties
`X_c(x)` and `X_c(y)` share with each other but **not** with the
originals are properties outside `cl(c)`. By assumption, `γ ⊆ cl(c)`,
so anything left bridging `X_c(x)` and `X_c(y)` more strongly than `y`
itself does is something inside `cl(c)` that the mechanism failed to
remove — i.e., a privacy failure.

(The paper makes this rigorous: see Theorem 1 and Corollary 1.)

### Cosine form

When `D(x, y) = 1 − cos(E(x), E(y))` for embedding `E`, the inequality
simplifies to a **bias check**:

```
E(X_c(x)) · ( E(y) − E(X_c(y)) )  <  δ
```

The vector `E(y) − E(X_c(y))` encodes "the part of `y` that the sanitizer
removed". If `X_c(x)` has any leftover bias along that direction, you've
leaked.

---

## 4. Algorithm 1

Below is the algorithm in its strict form (all-pairs). For large proxy
sets you can swap the inner loop for a FAISS / HNSW lookup; the published
implementation in this repo currently uses random-trial sampling, which
checks a subset rather than every pair (see §7.2).

```text
ALGORITHM contrastive_privacy_test(X̃, X_c, E, δ):
    S, T  ←  empty lists
    for x in X̃ that captures γ ∩ c:
        S.append( E(X_c(x)) )                # sanitized embedding
        T.append( E(x) − E(X_c(x)) )         # "what was removed" vector

    I ← VectorDB.index(T)                    # or O(n²) loop

    for u in S:
        v ← VectorDB.nearest(I, u)           # closest "removed" vector
        if dot(u, v) ≥ δ:
            return FAIL  (with the (u, v) pair as the diagnostic)

    return PASS
```

### Complexity

| Variant | Per-item cost |
|---|---|
| Brute force | `O(n)` |
| FAISS exact | `O(n)` |
| HNSW approximate | `O(log n)` (with a tunable bound on slop, which acts as additional `δ`) |

For typical proxy set sizes (`n ≈ 30–50`) the brute-force version is
trivially fast.

### Equivalent formulation in the code

The reference implementation (`compute_resolution()` in
`src/contrastive_privacy/scripts/resolution_analysis.py`) computes:

```
d1 = 1 − sim(X_c(u), v)            # distance to peer's original
d2 = 1 − sim(X_c(u), X_c(v))       # distance to peer's sanitized
resolution = d2 − d1               # > 0 means leak
```

A pair leaks iff `resolution ≥ δ`. This is algebraically the same as
Inequality 4. The aggregate "privacy resolution" reported per cell in the
paper's tables is the **maximum leak observed** across the sampled pairs,
i.e. the smallest `δ` that would let the test pass on this dataset.

---

## 5. Reimplementing it — what you need to bring

You need exactly four things. Everything else is bookkeeping.

1. **A proxy set `X̃`.** 30–50 items that all capture the abstract concept
   `γ`. Bigger sets find more failures but become over-conservative; they
   also cost more API calls if your sanitizer is paid.
2. **A sanitizer `X_c`** — any function `rendering → rendering`.
3. **An embedder `E`** — any function `rendering → vector` that lives in a
   space where cosine similarity is meaningful.
4. **A resolution `δ`** — start with `0.0`. Loosen only if you have to.

Once you have these, the entire test fits in a couple of dozen lines:

```python
import numpy as np

def cos_dist(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(1.0 - np.dot(a, b))

def contrastive_privacy_test(items, sanitize, embed, delta=0.0):
    sanitized = [sanitize(x) for x in items]
    E_orig = np.stack([embed(x) for x in items])
    E_san  = np.stack([embed(x) for x in sanitized])

    failures, worst = [], 0.0
    for i in range(len(items)):
        for j in range(len(items)):
            if i == j: continue
            d_to_orig = cos_dist(E_san[i], E_orig[j])
            d_to_san  = cos_dist(E_san[i], E_san[j])
            leak = d_to_san - d_to_orig
            worst = max(worst, leak)
            if d_to_orig + delta <= d_to_san:
                failures.append((i, j, leak))

    return len(failures) == 0, worst, failures
```

The packaged version in `contrastive_privacy.api` is the same loop with a
typed result object and a progress hook.

---

## 6. Practical considerations

### 6.1 Concept ambiguity penalty

If you remove **more** than is needed (e.g. all faces when only Bob's face
matters), then a stranger who happens to appear next to Bob can drag her
own semantic neighbors (Christmas trees, sports jerseys, anything) into
`cl(c)`. Suddenly unrelated images count as failures. Three coping
strategies:

- Flag concept-ambiguous renderings and don't release them.
- Add additional natural properties `d` to remove from just those problem
  items.
- Edit `c` to exclude the broad neighbors that aren't actually related to
  your secret.

### 6.2 The distance mechanism `D` usually needs fine-tuning

`D` is assumed to capture every semantic link relevant to `γ`. That's only
roughly true when `γ` is well-represented in the embedder's training data.
For a non-famous person, an internal product, an in-house brand, you must
fine-tune the embedder on examples that capture `γ` before the test means
anything. You only have to label `γ` itself; you do **not** need to label
the natural concepts that `c` covers.

### 6.3 Picking `δ`

`δ = 0` is ideal but often unreachable. When you have to loosen, the
paper provides a calibration: at what resolution does `D` stop drawing a
link between two named concepts at the 90 / 95 / 99th percentile?

| Compared concepts | 90% | 95% | 99% |
|---|---|---|---|
| dog vs. orange | 0.075 | 0.093 | 0.127 |
| dog vs. cat    | 0.103 | 0.136 | 0.191 |
| orange vs. lemon | 0.135 | 0.198 | 0.309 |

(Values from the paper's Table 1 using EVA-CLIP for images.) So a result
"private at `δ = 0.136`" translates to "private against an adversary who
can't tell dogs from cats at the 95th percentile." Useful for reporting.

### 6.4 The proxy set is an approximation

`X̃` is assumed to cover `γ ∩ c`. It usually doesn't, perfectly. The test
gives you an empirical privacy measurement, not a formal guarantee. This
is the same trade-off as any practical adaptation of differential privacy
to unstructured data; the upside is that the result has a natural
explanation in semantic terms.

### 6.5 LLMs as privacy oracles don't work

A natural-seeming alternative is to ask a frontier VLM "is this image
private?" The paper documents that the same VLM gives contradictory
verdicts on the same image depending on prompt phrasing. Don't use this
as your test — use it as a sanitizer if you like, but evaluate it
externally with the contrastive test.

---

## 7. Map to this repo

| Concept | Where it lives in the code |
|---|---|
| Public API | `src/contrastive_privacy/api.py` (`contrastive_privacy_test`) |
| Image distance `D` | `src/contrastive_privacy/scripts/compare_images.py` (default model: `microsoft/LLM2CLIP-EVA02-B-16`; `BAAI/EVA-CLIP-18B` selectable) |
| Text distance `D` | `src/contrastive_privacy/scripts/compare_texts.py` (default: `Qwen/Qwen3-Embedding-8B`) |
| Sanitizers `X_c` | `src/contrastive_privacy/scripts/anonymize.py` (SAM3, GroundedSAM, CLIPSeg, ai-gen, vlm-bounding-box) and `text_anonymize.py` |
| Concept generators | `src/contrastive_privacy/scripts/identify_obfuscation_concepts.py` (LLM-based) |
| Image experiment driver | `src/contrastive_privacy/scripts/resolution_analysis.py` (CLI: `resolution-analysis`) |
| Text experiment driver | `src/contrastive_privacy/scripts/text_resolution_analysis.py` |
| Reproducible commands | `experiments.sh` |

### 7.1 The published CLI gives you a working sanitization pipeline

If you want the "concept generation → sanitization → resolution analysis"
loop and not just the test, use:

```bash
resolution-analysis data/dicaprio \
    --segmenter ai-gen \
    --fal-image-model fal-ai/gpt-image-1.5 \
    --privacy-concept "the identity of the celebrities" \
    --output runs/dicaprio_gpt-image-1.5
```

Output: an `obfuscated/` folder of sanitized images, `results.csv` with
per-pair resolutions, `report.txt` summarizing the worst leaks, and a
side-by-side summary grid.

### 7.2 One implementation note worth knowing

Inequality 4 is universally quantified ("for every pair"). The published
CLI samples a fixed number of comparison items per reference (`--trials`,
default `5`). That means a passing run guarantees the inequality on a
**subset** of pairs, not all of them. For a strict test, increase
`--trials` toward the full proxy size, or use the `contrastive_privacy_test`
function in `api.py`, which checks every ordered pair.

---

## 8. Diagnosing failures

When the test fails, sort `failures` by leak magnitude and look at the top
few pairs. The paper's most informative results came from this step:

- A McDonald's window with a silhouetted "M" the sanitizer missed.
- Text reading "Leonardo DiCaprio" visible behind a redacted face.
- A Margot Robbie poster from a film DiCaprio starred in — semantically
  bound to the actor through the movie.
- An Avengers passage that redacted "Iron Man" but left "superhero movie",
  which a peer passage included.

The framework gives you a **reproduction case** for every leak it finds.
Treat each top-`k` failure as a bug report and either:

1. Expand `c` (add the missing natural concepts to the redaction list);
2. Improve the sanitizer (different model, different prompts, more
   aggressive coverage);
3. Fine-tune `D` if it's missing semantic links latent in your data;
4. Loosen `δ` if reaching the strictest setting is provably impossible.

---

## 9. References

- Paper: *Contrastive Privacy: A Semantic Approach to Measuring Privacy of
  AI-based Sanitization* — Bissias, Bagdasarian, Levine (UMass Amherst).
- Code: <https://github.com/umass-forensics/contrastive-privacy>
- Citation: see `index.html` or the paper's BibTeX.
