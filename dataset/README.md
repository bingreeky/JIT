<div align="center">

# `dataset/`

**The benchmark data**

[Main README](../README.md) · [`jit/`](../jit/) · [`scripts/`](../scripts/) · [`harness_factory/`](../harness_factory/) · [`benchmark/`](../benchmark/)

</div>

---

Every benchmark this project evaluates on, in one place. The runners read data
straight from here — the paths in `benchmark/config/*.yaml` are
`./dataset/<name>/`, relative to the repository root.

> [!NOTE]
> **Naming.** `dataset/` (this directory) is *data only*. `benchmark/` next to it
> holds the run configuration (`benchmark/config/`) and the code
> (`benchmark/adapter/`) for each of these datasets.

Small datasets ship in full. Anything large is a documented download instead of a
committed blob: the per-dataset `README.md` stays either way, so an absent dataset
tells you what it is and how to get it.

**Contents** — [What ships, what you download](#what-ships-in-this-repository-and-what-you-must-download) ·
[Downloading the large assets](#downloading-the-large-assets) ·
[Configuration](#configuration) · [Provenance and licensing](#provenance-and-licensing)

---

## What ships in this repository, and what you must download

Large assets are **not** committed. Anything over ~100 MB is a separate download
with a documented command; everything smaller is already here and works out of
the box.

| Key | Dataset | Directory | Cases | In repo | Download separately | Judge? |
|---|---|---|:--:|---|---|:--:|
| `xbench` | xbench DeepSearch-2505 | `xbench/` | 100 | 88 KB — complete | — | yes |
| `deepsearchqa` | DeepSearchQA (DSQA-full) | `deepsearchqa/` | 900 | 352 KB — complete | — | yes |
| `agentif` | AgentIF-OneDay | `agentif_oneday/` | 104 | 264 KB — questions | `Attachments/` (250 MB) | yes |
| `officebench` | OfficeBench | `officebench/` | 295 | 22 MB — complete | — | no |
| `odyssey` | OdysseyBench (plus) | `odysseybench/` | 300 | 55 MB — complete | — | no |
| `shopping` | DeepPlanning-Shopping | `deepplanning_shopping/` | 120 | 24 MB — complete | — | no |
| `travel` | DeepPlanning-Travel (zh) | `deepplanning_travel/` | 120 | 1.3 MB — tasks + evaluator | `database/` (**748 MB**) | yes |

Check the current state at any time:

```bash
python scripts/check_datasets.py
```

It prints one line per benchmark — present / missing / partial — and names the
exact command to run for anything missing.

---

## Downloading the large assets

All of it needs the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # only for gated datasets
```

`fetch_datasets.sh` wraps every command below:

```bash
bash scripts/fetch_datasets.sh travel      # one benchmark
bash scripts/fetch_datasets.sh all         # everything (≈1 GB)
```

Run it from the repository root. It is idempotent — re-running skips what is
already in place.

### `travel` — DeepPlanning-Travel database (748 MB)

The travel evaluator queries a local flight/hotel/restaurant database. Without
it, every case errors out.

```bash
hf download Qwen/DeepPlanning --repo-type dataset \
  --include "travelplanning/database/database_zh.zip" \
  --include "travelplanning/database/database_en.zip" \
  --local-dir .cache/datasets/deepplanning

mkdir -p dataset/deepplanning_travel/database
cp .cache/datasets/deepplanning/travelplanning/database/database_*.zip \
   dataset/deepplanning_travel/database/
unzip -o dataset/deepplanning_travel/database/database_zh.zip \
   -d dataset/deepplanning_travel/database
unzip -o dataset/deepplanning_travel/database/database_en.zip \
   -d dataset/deepplanning_travel/database
```

**Verify** — `ls dataset/deepplanning_travel/database/` shows `database_zh/` and
`database_en/`.

### `agentif` — AgentIF-OneDay attachments (250 MB)

The 104 questions ship in the repo; the attachments each one references (the
input files, and the reference answers the judge compares against) do not.

```bash
hf download xbench/AgentIF-OneDay --repo-type dataset \
  --local-dir dataset/agentif_oneday
```

**Verify** — `dataset/agentif_oneday/Attachments/` contains `Questions/` and
`Reference_answer/`.

> [!NOTE]
> Runs export `AGENTIF_JUDGE_TEXT_ONLY=1` (see `benchmark/registry.py`), which
> keeps HTML-render screenshots out of the judge payload — a text-only judge
> endpoint rejects them outright. Cases whose scoring criteria genuinely require
> seeing a produced image are therefore graded on their text criteria alone.

---

## Configuration

### Credentials

Copy `.env.example` to `.env` at the repository root and fill it in. The two
that most benchmarks need:

| Variable | Used by |
|---|---|
| `OPENAI_API_BASE` / `OPENAI_API_KEY` | the execution model and the judge |
| `SERPER_API_KEY` | `web_search` — `xbench`, `deepsearchqa`, `agentif` |
| `JINA_API_KEY` | `crawl_page` — same benchmarks |

> [!WARNING]
> A spent `SERPER_API_KEY` is the classic silent failure: `web_search` returns
> "not enough credits", the judge dutifully scores the resulting empty answers as
> wrong, and the run reads as a model-capability drop. Check your quota before
> blaming the model.

### Per-benchmark knobs

Each benchmark has one YAML under `benchmark/config/`, which owns the
execution model, the tools, the step budget and the judge. The registry in
[`benchmark/registry.py`](../benchmark/registry.py) owns everything else — which
field results are grouped by, the full-dataset case count, and any environment
variables the benchmark's own harness needs.

Adding a benchmark = one YAML plus one `BenchmarkSpec` entry.

### Extra Python dependencies

The office-style benchmarks manipulate real documents and need file-format
libraries beyond the base install:

```bash
# officebench, odyssey, agentif
pip install icalendar pytz PyMuPDF pdf2docx pdf2image pytesseract \
            openpyxl python-docx python-pptx PyPDF2
```

They are in the repo-root `requirements.txt` under their own heading.

---

## Provenance and licensing

Each dataset directory keeps its own upstream `README.md` with the original
source URL. The data here is redistributed for reproducibility and remains
under its **upstream license** — check the source repository before using any
of it beyond evaluating this project.

| Key | Upstream |
|---|---|
| `xbench` | https://huggingface.co/datasets/xbench/DeepSearch |
| `deepsearchqa` | https://huggingface.co/datasets/google/deepsearchqa |
| `agentif` | https://huggingface.co/datasets/xbench/AgentIF-OneDay |
| `officebench` | https://github.com/zlwang-cs/OfficeBench |
| `odyssey` | https://github.com/OdysseyBench/OdysseyBench |
| `shopping` / `travel` | https://huggingface.co/datasets/Qwen/DeepPlanning |
