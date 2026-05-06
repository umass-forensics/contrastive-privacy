# Contrastive Privacy

Contrastive Privacy contains image and text anonymization tools plus experiment drivers for measuring how much semantic information remains after obfuscation. The main experiment scripts create obfuscated copies of a dataset, compute an "effective resolution" score across sampled pairs, and then optionally run a similarity analysis between each original item and its obfuscated version.

The current workflow is centered on:

- `resolution-analysis` for image datasets.
- `python -m contrastive_privacy.scripts.text_resolution_analysis` for text datasets.
- `similarity-analysis` for post-run original-vs-obfuscated similarity histograms.
- `results-webpage` for a filterable HTML report plus a cached JSON analysis bundle.
- `experiments.sh` as a command catalog for the paper-style image and text experiments.

## Paper

- PDF: https://arxiv.org/pdf/2605.02977.pdf

If you use this repository, please cite:

```bibtex
@article{bissias2026contrastive,
  title   = {Contrastive Privacy: A Semantic Approach to Measuring Privacy of AI-based Sanitization},
  author  = {Bissias, George and Bagdasarian, Eugene and Levine, Brian Neil},
  journal = {arXiv preprint arXiv:2605.02977},
  year    = {2026}
}
```

## Setup

### Requirements

What works locally without external services:

- Python 3.9 or newer.
- CPU is enough for small `clipseg` image runs and small text entity runs.
- A CUDA GPU is strongly recommended for larger local image and embedding workloads.

What may require authentication or API keys:

- A Hugging Face account/token for gated or licensed model downloads such as `sam3`.
- A fal.ai API key for `--segmenter ai-gen`, `--segmenter vlm-bounding-box`, `identify-obfuscation-concepts`, or text `--approach concept`.
- Some older single-image OpenAI editing paths use `OPENAI_API_KEY`, but the main experiment workflow does not depend on it.

### Install

```bash
conda create -n priv python=3.11
conda activate priv
pip install -e .
```

For this repo, the validated local environment in this workspace has been `cp0`. If you already have that environment, the equivalent install path is:

```bash
conda activate cp0
pip install -e .
```

`pip install -e .` uses `pyproject.toml` and installs the console scripts listed there. Prefer it over `requirements.txt`, which is older and does not list every current dependency.

If you change `[project.scripts]` in `pyproject.toml`, rerun `pip install -e .` so new console commands are registered in the active environment.

If you use Hugging Face gated models, log in before the first run:

```bash
hf auth login
```

Set your fal.ai key when using cloud image editing, OpenRouter vision, or concept-based text redaction:

```bash
export FAL_KEY="your-fal-api-key"
```

Some legacy single-image OpenAI editing code paths use `OPENAI_API_KEY`, but the experiment commands in `experiments.sh` use fal.ai/OpenRouter through `FAL_KEY`.

## Data Layout

Experiment commands expect datasets under `data/`:

```text
data/
  dicaprio/          # image dataset expected by image experiments
  mcdonalds_large/   # image dataset expected by image experiments
  mcdonalds_small/   # image dataset expected by image experiments
  avengers_large/    # text dataset
  avengers_small/    # text dataset
```

Text files can use `.txt`, `.md`, `.text`, `.csv`, or `.json`. Image files can use `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, or `.tiff`.

The image experiment commands require the corresponding image folders under `data/`, and the text experiment commands use the `avengers_*` folders.

Generated analysis runs are expected to live outside the tracked source tree, typically under `runs/`. The repo now ignores that directory so local experiment outputs, HTML reports, cached JSON bundles, and generated media do not show up as unstaged changes.

## What Needs What

### Local-Only Paths

These do not require `FAL_KEY` and can be run entirely locally once dependencies are installed:

- Image runs with `--segmenter clipseg`.
- Image runs with `--segmenter groundedsam`, assuming the local model weights are available.
- Text runs with `--approach entity`.
- Report generation with `results-webpage` and `open-results-webpage`.

### Hugging Face Access Needed

- `--segmenter sam3` is local at inference time, but the `facebook/sam3` weights are gated on Hugging Face.
- Some other model backends may also require Hugging Face authentication or license acceptance before first download.

### API Keys Needed

- `FAL_KEY`: required for `ai-gen`, `vlm-bounding-box`, `identify-obfuscation-concepts`, and text `--approach concept`.
- `OPENAI_API_KEY`: only needed for the older OpenAI-specific single-image utilities, not for the default local workflow.

## Simple Local Run

The smallest no-external-service image path is a `clipseg` run with explicit objects:

```bash
resolution-analysis data/dicaprio \
  --mode blackout \
  --trials 1 \
  --samples 2 \
  --segmenter clipseg \
  --objects face person \
  --output runs/dicaprio_clipseg_local_cp0 \
  --seed 1 \
  --min-coverage 0.00001 \
  --max-coverage 1.0 \
  --threshold 0.2 \
  --blur 5 \
  --dilate 5 \
  --embedder-model openai/clip-vit-base-patch32 \
  --embed-batch-size 2 \
  --device cpu \
  --skip-comparisons
```

After that run completes, open the local HTML report with:

```bash
open-results-webpage runs/dicaprio_clipseg_local_cp0 --skip-similarity
```

The `--skip-similarity` flag is optional here. It keeps the local review loop fast by rendering from the resolution results only, without recomputing original-vs-obfuscated utility metrics.

## Core Commands

### Image Resolution Analysis

Use `resolution-analysis` to obfuscate image datasets and evaluate effective resolution.

Local-only example:

```bash
resolution-analysis data/dicaprio \
  --mode blackout \
  --trials 1 \
  --samples 2 \
  --segmenter clipseg \
  --objects face person \
  --output runs/dicaprio_clipseg_local_cp0 \
  --seed 1 \
  --min-coverage 0.00001 \
  --max-coverage 1.0 \
  --threshold 0.2 \
  --blur 5 \
  --dilate 5 \
  --embedder-model openai/clip-vit-base-patch32 \
  --embed-batch-size 2 \
  --device cpu \
  --skip-comparisons
```

API-backed example from the experiment catalog:

```bash
resolution-analysis data/dicaprio \
  --mode blackout \
  --trials 29 \
  --samples 30 \
  --segmenter ai-gen \
  --fal-image-model fal-ai/gpt-image-1.5 \
  --privacy-concept "the identity of the celebrities" \
  --output dicaprio_large/gpt-image-1.5_remove_t3 \
  --seed 34518474335 \
  --min-coverage 0.000001 \
  --max-coverage 1.0 \
  --blur 5 \
  --dilate 25 \
  --skip-empty-labels \
  --convex-hull \
  --embedder-model BAAI/EVA-CLIP-18B \
  --skip-comparisons
```

Important flags:

- `image_folder`: folder of original images.
- `--mode`: `blackout`, `redact`, or `replace`.
- `--trials`: number of comparison items sampled per reference item.
- `--samples`: number of reference items to sample. Omit it to use all items.
- `--output`: output directory. It will contain `obfuscated/`, `results.csv`, `params.json`, `report.txt`, and usually summary/comparison images.
- `--segmenter`: `groundedsam`, `clipseg`, `sam3`, `ai-gen`, or `vlm-bounding-box`.
- `--objects`: explicit object/concept labels to detect and obfuscate.
- `--privacy-concept`: concept-level instruction for `--segmenter ai-gen`; this can replace `--objects`.
- `--base-concepts`: additional labels always obfuscated with SAM3 after the primary obfuscation.
- `--min-coverage` and `--max-coverage`: discard runs that change too little or too much of the item.
- `--continue`: reuse existing `output/obfuscated/` files and fill in missing ones.
- `--retry`: retry skipped references from a previous `report.txt` using the saved `params.json`.
- `--skip-comparisons`: skip compound comparison images while still computing metrics. Some commands in `experiments.sh` use `--skip-comparison`; argparse accepts that as an abbreviation, but new commands should use the plural spelling.

Segmenter notes:

- `clipseg`: easiest local path for a first run.
- `groundedsam`: local path with stronger object grounding, but heavier than `clipseg`.
- `sam3`: local inference path, but requires gated Hugging Face access before first download.
- `ai-gen` and `vlm-bounding-box`: external-service paths that require `FAL_KEY`.

### Image Concept Identification

Use `identify-obfuscation-concepts` to ask a fal.ai OpenRouter vision model which visible concepts should be removed to protect a high-level privacy target:

```bash
identify-obfuscation-concepts data/dicaprio \
  --model gpt-5.4 \
  --output-format simple \
  "the identity of the celebrities"
```

The command accepts either one image file or a folder of images. It uploads each image, asks the selected vision model for reusable obfuscation concepts, deduplicates the concepts across the dataset, and prints the result.

Important flags:

- `path`: a single image or folder of images to analyze.
- `target_concept`: the privacy target to protect, such as `the identity of the celebrities`.
- `--model`: `gpt-5.4`, `gemini-3.1-pro`, `opus-4.6`, or a raw OpenRouter model ID.
- `--output-format`: `text`, `json`, or `simple`. The experiments use `simple`, which prints a quoted, comma-separated list that can be copied into a later `resolution-analysis --objects ...` invocation.
- `--output`: optional file path for saving the concept list instead of only printing it.

This command requires `FAL_KEY`.

### Text Resolution Analysis

There is no installed console script for text resolution analysis in `pyproject.toml`, so run it as a module.

Local-only example:

```bash
python -m contrastive_privacy.scripts.text_resolution_analysis data/avengers_small \
  --mode blackout \
  --trials 2 \
  --samples 3 \
  --approach entity \
  --entities movie person organization \
  --output runs/avengers_entity_local_cp0 \
  --embedder sbert \
  --embedder-model sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu \
  --embed-batch-size 2
```

API-backed concept example:

```bash
python -m contrastive_privacy.scripts.text_resolution_analysis data/avengers_small \
  --mode blackout \
  --trials 8 \
  --samples 9 \
  --approach concept \
  --concept "anything that can identify the movie discussed in this passage" \
  --concept-model openai/gpt-5.4 \
  --output avg_small_gpt-5.4_t3 \
  --embedder qwen \
  --embedder-model Qwen/Qwen3-Embedding-8B \
  --embedder-quantization none \
  --device cuda \
  --embed-batch-size 1 \
  --min-coverage 0.00001 \
  --max-coverage 1.0 \
  --sequential-labels
```

Text analysis supports two obfuscation approaches:

- `--approach entity`: uses GLiNER2 NER with `--entities`, optionally plus literal `--instances`.
- `--approach concept`: uses fal.ai OpenRouter with `--concept` and `--concept-model`; this requires `FAL_KEY`.

Common text flags:

- `--mode blackout`: replaces sensitive spans with block characters.
- `--mode redact`: replaces sensitive spans with `--placeholder`, defaulting to `[REDACTED]`.
- `--entities`: entity types for GLiNER2, such as `person`, `organization`, `movie`, or custom open-vocabulary labels.
- `--instances`: literal substrings to obfuscate without NER.
- `--threshold`: NER detection threshold.
- `--sequential-labels`: process entity labels separately and merge results, so adding labels cannot reduce detections.
- `--propagate` / `--no-propagate`: control whether repeated occurrences of detected text are anonymized.
- `--embedder`: `sbert`, `clip`, or `qwen`; the experiments use Qwen embeddings.
- `--embedder-quantization`: `none`, `half`, `4bit`, or `8bit`.

### Similarity Analysis

After a resolution run, run `similarity-analysis` on the output folder:

```bash
similarity-analysis dicaprio_large/gpt-image-1.5_remove_t3 --model BAAI/EVA-CLIP-18B
```

For text runs:

```bash
similarity-analysis avg_small_gpt-5.4_t3 \
  --embedder qwen \
  --embedder-model Qwen/Qwen3-Embedding-8B
```

`similarity-analysis` auto-detects image vs. text outputs from `obfuscated/`. For image outputs, `--model` selects the CLIP image embedder. For text outputs, use `--embedder` and `--embedder-model`, or omit them and let the script reuse the settings saved in `params.json`. Some text follow-up commands in `experiments.sh` pass `--model Qwen/Qwen3-Embedding-8B`; that flag is image-only in the current CLI, so the text embedder still comes from `params.json` unless you pass `--embedder-model`.

Use `--image-folder` or `--text-folder` if the original data has moved since the resolution run.

### HTML Report Generation

Image and text resolution runs now auto-generate two extra files in the output folder unless you pass `--skip-analysis-artifacts`:

- `analysis_report.html`: a static webpage with summary metrics, histograms, a searchable/sortable all-pairs table, and example pairs with explanations.
- `analysis_report.json`: a cached structured bundle for notebooks, dashboards, or downstream plotting without recomputing similarity.

You can also regenerate or customize the report later with `results-webpage`:

```bash
results-webpage dicaprio_large/gpt-image-1.5_remove_t3
results-webpage avg_small_gpt-5.4_t3 --title "Avengers Text Run"
results-webpage some_output_folder --refresh --threshold 0.02 --top-n 8
results-webpage some_output_folder --low-utility-threshold 0.25 --high-utility-threshold 0.8 --open
open-results-webpage some_output_folder
```

Useful flags:

- `--refresh`: ignore any cached `analysis_report.json` and recompute the bundle.
- `--skip-similarity`: render the report from resolution results only, without recomputing original-vs-obfuscated similarity.
- `--low-utility-threshold` and `--high-utility-threshold`: control the utility bands shown in the report and the low-utility example filter.
- `--json-output` and `--output`: override the default JSON/HTML output paths.
- `--open`: open the generated report immediately in your default browser.
- `--image-folder` or `--text-folder`: resolve original files from a different location if the dataset moved.

Recommended local review loop:

```bash
open-results-webpage runs/dicaprio_clipseg_local_cp0 --skip-similarity
```

Use `--skip-similarity` when you only want to inspect an existing run and do not need to recompute original-vs-obfuscated utility metrics. Omit it when you want the report to include the similarity summary, low-utility examples, and utility bands based on fresh original-vs-obfuscated embeddings.

If you usually want the report opened after generation, `open-results-webpage` is a thin wrapper around `results-webpage` with browser opening enabled by default. Pass `--no-open` when you only want the artifacts refreshed.

For interactive inspection, see the notebook at `notebooks/inspect_results.ipynb`, which reuses the same JSON/HTML artifact pipeline.

## Understanding `experiments.sh`

`experiments.sh` is a flat list of commands, not a parameterized launcher. Running the entire file will attempt every experiment. New users should copy one command pair at a time: first the `resolution-analysis` or `python -m ...text_resolution_analysis` command, then the following `similarity-analysis` command against the output folder that was just created.

Each experiment usually has this shape:

```bash
# 1. Create obfuscated outputs and compute effective resolution.
resolution-analysis ... --output some_output_folder ...

# 2. Measure original-vs-obfuscated similarity for that output folder.
similarity-analysis some_output_folder ...
```

For text:

```bash
python -m contrastive_privacy.scripts.text_resolution_analysis ... --output some_output_folder ...
similarity-analysis some_output_folder ...
```

Some follow-up similarity commands point at suffixed folders such as `_t2`, `_t3`, or `_t4`. Those suffixes reflect iterative runs and retries. If you adapt a command, make sure the `similarity-analysis` argument exactly matches the `--output` folder you produced.

### Image Experiment Blocks

The image commands cover three datasets:

- `data/dicaprio` with outputs under `dicaprio_large/`, usually `--trials 29 --samples 30`.
- `data/mcdonalds_large` with outputs under `mcdonalds_large/`, usually `--trials 48 --samples 49`.
- `data/mcdonalds_small` with outputs under `mcdonalds_small/`, usually `--trials 8 --samples 9`.

Within each dataset, `experiments.sh` compares several image obfuscation backends:

- `--segmenter ai-gen --fal-image-model fal-ai/gpt-image-1.5`
- `--segmenter ai-gen --fal-image-model fal-ai/gpt-image-1-mini`
- `--segmenter ai-gen --fal-image-model fal-ai/gemini-3-pro-image-preview`
- `--segmenter ai-gen --fal-image-model fal-ai/gemini-3.1-flash-image-preview`
- `--segmenter ai-gen --fal-image-model fal-ai/flux-2-pro`
- `--segmenter ai-gen --fal-image-model fal-ai/flux-2`
- `--segmenter sam3` for several `dicaprio` manual/object-list runs

Before the resolution runs, each image dataset now has a concept-identification block:

```bash
identify-obfuscation-concepts data/dicaprio --model gpt-5.4 --output-format simple "the identity of the celebrities"
identify-obfuscation-concepts data/dicaprio --model gemini-3.1-pro --output-format simple "the identity of the celebrities"
identify-obfuscation-concepts data/dicaprio --model opus-4.6 --output-format simple "the identity of the celebrities"
```

The same pattern appears for `data/mcdonalds_large` and `data/mcdonalds_small` with the target concept `the identity of the fast food restaurant`. These calls are for generating model-specific candidate object lists. The resulting concepts correspond to the later `--objects` lists in output folders named with the model family, such as `*_remove_gpt-5.4_t3`, `*_remove_gemini-3.1-pro_t3`, and `*_remove_opus-4.6_t3`.

The `ai-gen` runs ask a fal.ai image editing model to handle detection and obfuscation together. They either use:

- `--privacy-concept`, for high-level redaction such as "the identity of the celebrities" or "the identity of the fast food restaurant".
- `--objects`, for explicit target lists produced manually or by different LLMs.

The output folder names encode the backend and target-list source:

- `*_remove_t3`: concept-level removal using `--privacy-concept`.
- `*_remove_manual_t3`: manually chosen target objects.
- `*_remove_gpt-5.4_t3`, `*_remove_gemini-3.1-pro_t3`, `*_remove_opus-4.6_t3`: target object lists generated by those LLM families.

The common image settings in the experiment file are:

- `--mode blackout`: replace selected regions with black pixels.
- `--blur 5 --dilate 25`: expand and soften masks before blackout.
- `--convex-hull`: hide silhouettes more aggressively by filling concave mask gaps.
- `--skip-empty-labels`: ignore ambiguous empty-label detections from GroundingDINO-style paths.
- `--fal-vision-temperature 0.1`: low-temperature OpenRouter vision calls where applicable.
- `--embedder-model BAAI/EVA-CLIP-18B`: use EVA-CLIP embeddings for resolution scoring.
- `--embed-batch-size 1`: used with large embedders to fit GPU memory.

### Text Experiment Blocks

The text commands cover two datasets:

- `data/avengers_small` with outputs under `avg_small_*`, usually `--trials 8 --samples 9`.
- `data/avengers_large` with outputs under `avg_large_*`, usually `--trials 48 --samples 49`.

Each text block starts with one manual/entity baseline:

```bash
python -m contrastive_privacy.scripts.text_resolution_analysis data/avengers_small \
  --mode blackout \
  --trials 8 \
  --samples 9 \
  --sequential-labels \
  --output avg_small_manual \
  --entities comic movie publisher author superhero abbreviation archetype year \
  --threshold 0.05 \
  --embedder qwen \
  --embedder-model Qwen/Qwen3-Embedding-8B \
  --embedder-quantization none \
  --device cuda \
  --embed-batch-size 1 \
  --instances remake blockbuster battle first powerful defeated movies die
```

That baseline combines GLiNER2 entity labels (`--entities`) with literal substrings (`--instances`).

The rest of each text block uses concept redaction:

```bash
--approach concept \
--concept "anything that can identify the movie discussed in this passage" \
--concept-model <provider/model>
```

The experiment file compares OpenRouter models from OpenAI, Google, and Anthropic, including `openai/gpt-5.4`, `openai/gpt-5.4-mini`, `openai/gpt-5.2`, `openai/gpt-4o`, `google/gemini-3.1-pro-preview`, `google/gemini-2.5-flash`, `anthropic/claude-opus-4.6`, and related variants.

All text experiment commands use Qwen embeddings:

```bash
--embedder qwen \
--embedder-model Qwen/Qwen3-Embedding-8B \
--embedder-quantization none \
--embed-batch-size 1
```

If you run out of GPU memory, try `--embedder-quantization half`, reduce `--embed-batch-size`, or switch to `--device cpu` for smaller runs.

## Single-Item Utilities

After installation, these image utilities are available as console scripts:

- `flux-anonymize`: anonymize one image with local segmentation, fal.ai image editing, or OpenRouter vision.
- `flux-create-mask`: create a segmentation mask.
- `flux-inpaint`: inpaint an image using a mask.
- `flux-generate` and `flux-edit`: older FLUX generation/editing helpers.
- `compare-images`, `recognize-objects`, `concept-resolution`, `identify-obfuscation-concepts`, `reanalyze-image-resolution`, and `reanalyze-text-resolution`.

Text single-file utilities are run as modules:

```bash
python -m contrastive_privacy.scripts.text_anonymize \
  --input document.txt \
  --output anonymized.txt \
  --entities person organization \
  --mode blackout
```

Use each script's `--help` for the complete CLI:

```bash
resolution-analysis --help
python -m contrastive_privacy.scripts.text_resolution_analysis --help
similarity-analysis --help
```

## Outputs

Resolution runs write reproducible artifacts to the chosen `--output` directory:

- `obfuscated/`: one `obfuscated_<original-name>` file per successfully obfuscated item.
- `results.csv`: per-comparison resolution values.
- `params.json`: the effective parameters used by the run.
- `report.txt`: summary statistics, skipped items, and configuration.
- `comparisons/`: compound comparison images when comparison generation is enabled for image runs.
- `summary_originals.jpg` and `summary_obfuscated.jpg`: image-grid summaries for image runs.

Use `--continue` after a partial run to reuse already-created obfuscations. Use `--retry --output <previous-output>` to retry references listed as skipped in a previous `report.txt`.

## Troubleshooting

- `command not found`: activate the environment and run `pip install -e .` again.
- Missing `FAL_KEY`: required for `ai-gen`, `vlm-bounding-box`, and text `--approach concept`.
- `sam3` access errors: run `hf auth login` and ensure your account has access to `facebook/sam3`.
- Missing image data: the image commands in `experiments.sh` expect `data/dicaprio`, `data/mcdonalds_large`, and `data/mcdonalds_small`.
- CUDA out of memory: reduce `--embed-batch-size`, use `--embedder-quantization half`, use smaller samples, or run on CPU for smaller text experiments.
- Model download/auth errors: run `hf auth login` and accept any required model licenses on Hugging Face.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
