<div align="center">

# PaperBrain

### A local-first workspace for deep paper reading, citation checks, and evidence-linked memory

**English** · [简体中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](pyproject.toml)

</div>

PaperBrain imports research papers locally, organizes them by section, assists with close reading and citation checks, and turns source-linked notes into searchable cross-paper memory. Remote language and vision models are optional; the core does not require a large local model.

> **Project status: under active development and not production-validated.** Model output, citation scoring, and automatic memory adjudication are not guaranteed to be correct. PaperBrain is a reading aid—not a substitute for checking the original paper, reproducing experiments, or conducting human peer review.

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Optional toolchain](#optional-toolchain)
- [Model configuration and consent](#model-configuration-and-consent)
- [Typical workflow](#typical-workflow)
- [Automatic learning and adjudication](#automatic-learning-and-adjudication)
- [Batch CLI](#batch-cli)
- [Outputs and local data](#outputs-and-local-data)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)
- [Development and validation](#development-and-validation)
- [Security and privacy](#security-and-privacy)
- [License](#license)

## What it does

| Capability | Current implementation and boundary |
| --- | --- |
| Document import | PDF, TXT, and Markdown through drag-and-drop, multi-select, or a local path. PDF and OCR depend on optional tools. |
| Section-aware reading | Organizes abstracts, introductions, methods, experiments, and related sections; low-confidence extraction is downgraded instead of silently trusted. |
| Reading tasks | Full reading, outline, method analysis, figure analysis, critical review, and memory extraction. Model-dependent work is limited when no model is available. |
| Outline and drafting | Generates an outline first and requires confirmation before drafting. Deep batch work does not bypass that confirmation step. |
| Citation checks | Checks source identifiers, figure/table references, and semantic support. Missing evidence or tools results in a review state, not a pass. |
| Learning memory | Stores notes, entities, and relations for cross-paper retrieval and answers annotated with `[M#]` memory sources. |
| Automatic adjudication | Re-examines candidate or contested notes with primary-paper evidence and optional Crossref abstract evidence. |
| Local-first storage | Uses local files and SQLite by default. Model, vision, embedding, and vector backends are opt-in. |

```text
Import → parse and segment → analyze and bind sources → check citations/claims
                                                     ├─ supported → active memory
                                                     ├─ insufficient → pending review
                                                     └─ refuted → rejected, audit retained
```

“Supported” means supported by the evidence available to the current checking pipeline. It does not prove that a scientific claim is objectively true.

## Quick start

PaperBrain requires **Python 3.9 or newer**. The commands below target macOS and Linux shells; Windows compatibility has not been fully validated.

```bash
git clone https://github.com/12echi/paperbrain.git
cd paperbrain
python3 -m venv .venv
source .venv/bin/activate
python3 -m paperbrain.server --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Keep the terminal running and press `Ctrl+C` to stop the server.

The basic offline TXT/Markdown path has no additional Python dependency. For a first run, import `demo/sample_paper.txt`. It is **synthetic test material**, not a real publication and not a valid research source.

For PDF parsing, install PyMuPDF inside the activated virtual environment:

```bash
python3 -m pip install pymupdf==1.26.5
```

Inspect detected tools at [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health). A working web page does not mean that PDF, OCR, formula, vector, or model capabilities are all available.

## Optional toolchain

Install only what your workflow needs. Python dependency declarations are in [pyproject.toml](pyproject.toml).

| Purpose | Dependency | Behavior when unavailable |
| --- | --- | --- |
| PDF text and image parsing | PyMuPDF | PDF processing is degraded and must not be treated as complete extraction. |
| OCR for scanned papers | Tesseract; relevant language data | Scanned pages may produce little or no usable text. |
| Formula checks | Node.js + KaTeX, and SymPy + ANTLR | Affected formulas remain unverified or require review. |
| Image processing | OpenCV | Some figure-analysis operations are unavailable. |
| Vector retrieval | Embedding endpoint + DuckDB VSS | Retrieval falls back to lexical search when the vector backend is unavailable. |

Example commands for macOS with Homebrew:

```bash
# OCR, including Chinese language data
brew install tesseract tesseract-lang

# Formula checking
brew install node
npm install -g katex@0.18.7
python3 -m pip install sympy==1.14.0 'antlr4-python3-runtime==4.11.*'

# Optional image, vector, and evaluation packages
python3 -m pip install opencv-python-headless duckdb==1.4.5 rapidfuzz==3.13.0
```

Installing `duckdb` does not prove that its VSS extension can be loaded, and it does not configure an embedding endpoint. KaTeX installed in an unusual global path may also go undetected. Use the health endpoint and task report as the source of truth.

## Model configuration and consent

### Configure in the web interface

1. Expand **Model settings** and enter the Base URL, API key, and text-model identifier supplied by your provider. Figure reading also requires a compatible vision model.
2. Separately enable cloud-transmission consent after confirming that the relevant paper text and selected images may be sent to that provider.
3. Save the configuration, test connectivity, and then start a task.

The opencode integration can reuse compatible service credentials already stored on the machine. It does not include free model usage and does not imply that every opencode model is compatible or available.

**“Configured but not authorized” is a permission state, not a connectivity error.** Without cloud consent, PaperBrain uses offline rules. Offline extraction is not equivalent to model-assisted deep reading or peer review.

### Configure with environment variables

Set variables in the same terminal that launches the server. Replace the placeholder endpoint and model name with values from your provider.

```bash
export PAPERBRAIN_BASE_URL='https://your-provider.example/v1'
export PAPERBRAIN_MODEL='your-text-model'
# Set PAPERBRAIN_API_KEY if injecting the credential through the environment.
# Web configuration is often safer than leaving a key in shell history.
export PAPERBRAIN_CLOUD_ALLOWED=1
python3 -m paperbrain.server --port 8000
```

PaperBrain does not automatically load `.env` files. Configuration generally resolves in this order: **process environment → local configuration file → code default**.

| Variable | Purpose / default behavior |
| --- | --- |
| `PAPERBRAIN_BASE_URL` | OpenAI-compatible API base URL. It must match the credential and model provider. |
| `PAPERBRAIN_API_KEY` | Text-model credential. Never commit it. |
| `PAPERBRAIN_MODEL` | Provider-supported text-model identifier. |
| `PAPERBRAIN_CLOUD_ALLOWED` | Consent for remote model transmission; disabled by default. |
| `PAPERBRAIN_WEB_VERIFY` | Enables Crossref evidence lookup; disabled by default. Claim text becomes the query. |
| `PAPERBRAIN_VL_MODEL` | Vision-model identifier; figure reading must also be enabled for the task. |
| `PAPERBRAIN_ALL_MODEL` | Lets the model participate in segmentation, graph extraction, and semantic checks; disabled by default. |
| `PAPERBRAIN_EMBED_BASE_URL` | Optional embedding endpoint; disabled when unset. |
| `PAPERBRAIN_EMBED_API_KEY` / `PAPERBRAIN_EMBED_MODEL` | Embedding credential and model identifier. |
| `PAPERBRAIN_REFLECT` | Automatic reflection toggle; enabled by default, but useful output still depends on model availability. |
| `PAPERBRAIN_CONF_FILE` / `PAPERBRAIN_MEMORY_DB` | Overrides the configuration-file or memory-database path. |

See [paperbrain/config.py](paperbrain/config.py) for the implementation. A loopback embedding URL does not require cloud consent. If that URL is forwarded through an SSH tunnel, data still leaves the machine; the operator must verify the destination and authorization.

## Typical workflow

1. **Import material.** Drop a PDF, TXT, or Markdown file onto the page, or use an advanced local path. Web uploads are limited to 20 MB per file; multiple files use the batch path.
2. **Choose the task.** Select full reading, method analysis, figures, review, outline, or memory extraction. Add a focus when useful.
3. **Choose the depth.** `fast`, `standard`, and `deep` control the amount of processing. Greater depth does not guarantee greater accuracy and usually increases latency and model cost.
4. **Inspect results.** Review the main reading result, complete analysis, verification report, and any additional artifacts generated by the selected task.
5. **Confirm the outline.** Draft-producing workflows require outline confirmation before generation.
6. **Query memory.** Inspect every `[M#]` source rather than trusting the synthesized answer alone.

`CLEAN` means that an artifact passed the checks available in that run. It does not mean that the paper is scientifically correct. `NEEDS_REVIEW`, `NO_CITATION`, and `NOT_RUN` mean review is required, no citation was found, or that check did not run; none is a pass.

## Automatic learning and adjudication

PaperBrain separates memory into states such as `active`, `candidate`, `contested`, and `rejected`. Trusted-memory answers exclude candidate, contested, rejected, and invalidated notes.

The pipeline in [paperbrain/adjudication.py](paperbrain/adjudication.py) works as follows:

1. On server startup, model-setting save, or memory finalization, a background review may start when cloud consent is enabled.
2. It takes a small batch of candidate or contested notes and retrieves original-paper chunks from the corresponding `state.json`.
3. The model must return `supported`, `refuted`, or `inconclusive` with verbatim evidence. The program verifies that evidence IDs and quotations occur in the supplied chunks.
4. If primary evidence is inconclusive and web verification is enabled, Crossref may add DOI-linked abstracts as supplementary evidence.
5. A supported or refuted result is checked by a second model call. Disagreement remains inconclusive.
6. Supported notes may become active. Refuted notes become rejected and leave trusted retrieval, while their audit records remain available.

### Current limitations

- A run handles at most **8 notes** by default. Notes attempted within the previous 24 hours are temporarily skipped. This is not a continuously draining background service.
- A note is not promoted from web evidence alone when its original paper cannot be bound. Primary-evidence lookup currently assumes the default `out/web/` layout; custom output paths may not be discovered.
- Crossref abstract search is not full-web or full-text fact checking. Some records lack abstracts, and returned records may be irrelevant.
- Two agreeing model calls are not two independent sources and do not establish an accuracy percentage.
- A verbatim quotation proves only that text occurs in the supplied evidence. It does not prove that the paper is correct or that the model interpreted it correctly.
- Missing evidence is not evidence of falsity. Inconclusive notes remain pending instead of being automatically deleted.
- Adjudication consumes extra model calls. Without consent, a functioning endpoint, and bindable evidence, automatic resolution remains incomplete.

## Batch CLI

Run from the repository root with the virtual environment activated:

```bash
# Two synthetic TXT samples
python3 tools/batch.py demo/sample_paper.txt demo/sample_paper2.txt \
  --out out/batch --task full_read --depth fast

# Replace this path with your paper directory
python3 tools/batch.py /path/to/papers \
  --pattern '*.pdf' --out out/batch --task full_read --depth standard

python3 tools/batch.py --help
```

Completed inputs can be skipped when their input, code, options, and artifact hashes match. `--force` bypasses resume checks and reprocesses inputs; verify the output directory and expected model cost first. Artifacts in a custom CLI output path may not be found by the default-path adjudicator.

## Outputs and local data

| Default path | Contents |
| --- | --- |
| `out/web/` | Web-task artifacts, uploaded inputs, and task state. |
| `out/batch/` | Default batch CLI output. |
| `out/memory.sqlite` | Cross-paper notes, relations, and adjudication records. |
| `out/memory.sqlite.vectors.duckdb` | Default vector database, created when needed. |
| `~/.config/paperbrain/env.json` | Model configuration and potentially credentials. |

Possible artifacts include `state.json`, `report.md`, `deepread.md`, `deepread_full.md`, `deepread_verify.json`, `outline_v1.json`, `draft.md`, `verify.json`, and `memory.json`. Exact output depends on the task and available tools.

Stop the service before making a consistent backup of `out/` and required configuration. Treat configuration backups as secrets. Keeping only the memory database while discarding source files and task states can remove evidence needed for adjudication. Git ignore rules are not encryption or access control.

## Docker

The repository includes a [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml):

```bash
docker compose up --build
# Stop the container
docker compose down
```

The default compose file binds to `127.0.0.1:8000`, persists `./out`, and limits the container to 3 GB RAM and 2 CPUs. Building downloads packages, system tools, KaTeX, and the DuckDB extension; network or extension failures are possible.

**The current documentation update did not validate the Docker build.** Host model configuration is not mounted automatically. Configuration saved inside the container may disappear after rebuilding unless separately persisted. A container cannot access arbitrary host paper paths without an explicit, controlled mount.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| The page does not open | Confirm that the server terminal is running. If port 8000 is busy, use `--port 8001`. |
| A configured model is not called | Check cloud consent, credential, Base URL, model identifier, provider quota, and environment-variable overrides. Never paste keys into an Issue. |
| Results look like simple excerpts | Check whether the run used offline rules, tool degradation, or low-confidence extraction. |
| A PDF has no body text | Check PyMuPDF. For scanned pages, check Tesseract and the required language data. |
| Formulas remain unverified | Check both Node/KaTeX and SymPy/ANTLR. Removing the warning does not validate a formula. |
| Trusted-memory count is zero | Imported papers may have generated only candidate notes, or evidence/model requirements may be unmet. |
| The quarantine never empties | Adjudication has a batch limit and retry interval; missing or conflicting evidence remains pending. |
| Vector retrieval is unavailable | Check both the embedding service and DuckDB VSS. Installing `duckdb` alone is insufficient. |

## Development and validation

```text
paperbrain/     application core, web server, parsing, models, and memory
tests/          unit and integration tests
tools/          batch, evaluation, calibration, and release-gate tools
demo/           synthetic fixtures—not scientific evidence
AGENTS.md       development constraints and module contracts
```

Local checks:

```bash
PAPERBRAIN_TEST=1 PAPERBRAIN_CLOUD_ALLOWED=0 \
  python3 -m unittest discover -s tests -v
python3 tools/release_gate.py
```

The initial public commit `4a77ba4` passed 294 tests on the maintainer's machine. This is not a CI badge and does not establish compatibility with every platform, provider, or Docker environment.

Production acceptance additionally requires labeled datasets, citation calibration, resource tests, and human review. Missing materials should cause the release gate to fail; weakening or skipping a check does not make the system production-ready. See [AGENTS.md](AGENTS.md) for current constraints and module contracts. Its metrics are targets, not verified performance claims.

When contributing, include a minimal reproduction, sanitized logs, Python and OS versions, and relevant test results. Do not submit paper collections, API keys, memory databases, or user configuration. Some helper scripts, including `tools/embed_tunnel.sh`, contain maintainer-specific assumptions and are not general setup commands.

## Security and privacy

- The server has no user authentication. Run it only in a trusted local environment and do not expose it directly to a LAN or the public internet.
- After model consent is enabled, relevant paper text or images are sent to the configured provider. With web verification enabled, claim text is sent to Crossref. Confirm that your material may leave the machine.
- Never place API keys in README files, screenshots, public logs, or Issues. `.gitignore` cannot protect a secret that has already been committed.
- Paper copyright and confidentiality obligations are unaffected by this project's license.
- Names, metrics, claims, and malformed formulas in the synthetic demo material must not be cited as scientific facts.

## License

PaperBrain is released under the [MIT License](LICENSE). Commercial use, modification, redistribution, and closed-source integration are allowed, provided that the copyright and license notice are retained in copies or substantial portions of the software. The software is provided without warranty.

Third-party dependencies, external model services, and user-imported papers remain subject to their own licenses and terms.
