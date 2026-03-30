# Credit Card Document Preprocessing Pipeline (Hybrid Rule + LLM)

A production-grade pipeline for classifying, renaming, and organising raw credit card PDF documents into a structured dataset ready for downstream Agentic RAG systems.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Role in the Agentic RAG System](#2-role-in-the-agentic-rag-system)
3. [Folder Structure](#3-folder-structure)
4. [Setup Instructions](#4-setup-instructions)
5. [How to Run](#5-how-to-run)
6. [How Classification Works](#6-how-classification-works)
7. [Metadata JSON Format](#7-metadata-json-format)
8. [Logs Explained](#8-logs-explained)
9. [Common Errors and Fixes](#9-common-errors-and-fixes)
10. [Troubleshooting Guide](#10-troubleshooting-guide)

---

## 1. Project Overview

This pipeline takes a folder of raw, unorganised credit card PDF files and:

- Extracts text from each PDF (with OCR fallback for scanned documents)
- Detects the **bank name**, **card name**, and **document type** (MITC / TNC / BR / LG)
- Handles **master/collective documents** that apply to all cards for a bank
- Applies a **hybrid classification approach**: fast rule-based detection first, LLM fallback only when needed
- Renames each file to a clean, consistent format: `BANK_CARD_DOCTYPE_YEAR.pdf`
- Organises files into labelled subfolders under `processed_docs/`
- Routes uncertain files to `needs_review/` for manual review
- Generates a **metadata JSON** alongside every processed PDF for use by the RAG system
- Validates dataset completeness and reports missing documents per card

---

## 2. Role in the Agentic RAG System

This pipeline is Stage 1 in the full system:

```
RAW PDFs
  ↓
[Stage 1: THIS PIPELINE]  ← You are here
  ↓
CLEAN DOCUMENT LIBRARY + METADATA JSON
  ↓
[Stage 2: RAG Indexing]
  ↓
VECTOR DATABASE (FAISS / Chroma)
  ↓
[Stage 3: Knowledge Extraction]
  ↓
STRUCTURED RULES (card_rules.json)
  ↓
[Stage 4: User Profiling + Simulation]
  ↓
[Stage 5: Agentic Recommendation]
  ↓
[Stage 6: Validation]
```

**Why preprocessing matters:** The RAG system's accuracy depends entirely on correctly classified, consistently named documents. If a MITC document is stored as TNC, the simulation engine will look in the wrong place for fee information and produce wrong recommendations. This pipeline ensures every document is correctly identified before it enters the knowledge base.

**The `data_quality` field** in metadata tells downstream agents whether a card's document set is complete enough to make a reliable recommendation:
- `"complete"` — both MITC and BR are present → agent can recommend
- `"partial"` — one required doc is missing → agent should flag uncertainty
- `"insufficient"` — no required docs → agent must refuse to recommend

---

## 3. Folder Structure

```
project_root/
│
├── data/
│   ├── raw_docs/                ← Place your input PDFs here (never modified)
│   │
│   ├── processed_docs/          ← Organised output (created automatically)
│   │   ├── HDFC_Millennia/
│   │   │   ├── HDFC_Millennia_MITC_2026.pdf
│   │   │   ├── HDFC_Millennia_MITC_2026.json   ← metadata for RAG
│   │   │   └── HDFC_Millennia_BR_2026.pdf
│   │   ├── SBI_Cashback/
│   │   └── HDFC_MASTER/         ← Bank-wide collective documents
│   │       └── HDFC_MASTER_MITC_2026.pdf
│   │
│   ├── needs_review/            ← Low-confidence files for manual check
│   │
│   └── logs/
│       ├── summary.csv                    ← One row per file
│       ├── preprocess_log.txt             ← Full per-file audit trail
│       ├── hybrid_classification_log.txt  ← LLM decision details
│       ├── missing_docs_report.csv        ← Coverage gaps per card
│       └── coverage_dashboard.xlsx        ← Visual Excel grid
│
└── data_pipeline/
    ├── preprocess.py              ← Rule-based core (do not modify)
    ├── preprocess_with_llm.py     ← Hybrid pipeline (entry point)
    └── llm_classifier.py          ← LLM classification module
```

> Raw files in `raw_docs/` are **never deleted or modified**. The pipeline only copies files to destinations.

---

## 4. Setup Instructions

### Step 1 — Python version

You need **Python 3.10 or higher**.

```bash
python --version
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install core Python dependencies

```bash
pip install pdfplumber pymupdf requests pandas openpyxl
```

| Package | Purpose |
|---|---|
| `pdfplumber` | Primary PDF text extraction |
| `pymupdf` | Fallback PDF text extraction + OCR page rendering |
| `requests` | HTTP calls to Ollama API |
| `pandas` | CSV and Excel report generation |
| `openpyxl` | Excel coverage dashboard |

### Step 4 — Install OCR dependencies (optional)

Only needed if you have scanned (image-only) PDFs that contain no embedded text.

```bash
pip install pytesseract Pillow
```

Then install the **Tesseract binary**:

**Windows:**
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the `.exe` and complete the wizard
3. The pipeline automatically looks for Tesseract at: `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. If you installed it somewhere else, open `preprocess_with_llm.py`, find the line `TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"` and update the path

**macOS:**
```bash
brew install tesseract
```

**Ubuntu / Debian:**
```bash
sudo apt install tesseract-ocr
```

### Step 5 — Install Ollama and pull Mistral

Ollama runs the LLM locally. No API key needed. No data leaves your machine.

**Install Ollama:**

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from: https://ollama.ai/download
# Run the installer (.exe) and follow the wizard
```

**Pull the Mistral model** (~4.1 GB, one-time download):

```bash
ollama pull mistral
```

**Start the Ollama server:**

```bash
ollama serve
```

Keep this terminal open while the pipeline runs. On most systems, Ollama starts automatically at boot after installation.

**Verify Mistral is working:**

```bash
ollama run mistral "say hello"
```

You should see a short text response within 30 seconds on CPU, under 5 seconds on GPU.

---

## 5. How to Run

All commands are run from the **project root folder** (the folder that contains `data/`).

### Normal run — full pipeline with LLM fallback

```bash
python data_pipeline/preprocess_with_llm.py
```

This is the recommended command. The LLM is called only for low-confidence documents.

### Rules only — no LLM, no Ollama needed

```bash
python data_pipeline/preprocess_with_llm.py --no-llm
```

Use this for fast testing or when Ollama is not available. Low-confidence documents go to `needs_review/` instead of being sent to the LLM.

### Dry run — simulate everything, move nothing

```bash
python data_pipeline/preprocess_with_llm.py --dry-run
```

Run this first to see what the pipeline *would* do without actually copying any files. Safe to run multiple times.

### Debug mode — print extracted text for each PDF

```bash
python data_pipeline/preprocess_with_llm.py --debug
```

Prints the first ~600 characters of extracted text from each PDF. Use this when a file goes to `needs_review/` and you want to see what text was actually extracted.

### Combine flags

```bash
python data_pipeline/preprocess_with_llm.py --dry-run --debug
python data_pipeline/preprocess_with_llm.py --no-llm --dry-run
```

### Test LLM in isolation

```bash
python data_pipeline/llm_classifier.py
```

Runs 4 built-in test cases and shows pass/fail. Use this to verify Ollama is configured correctly before running the full pipeline.

---

## 6. How Classification Works

### Overview

```
PDF text
  ↓
[STEP 1] Text extraction: pdfplumber → PyMuPDF → OCR (if both fail)
  ↓
[STEP 2] Rule-based detection (fast, deterministic)
         detect_bank() + detect_card() + detect_doc_type() + detect_master_doc()
         ↓
         Bank-specific card narrowing (FIX #2)
         If card doesn't belong to detected bank → re-scan for correct card
  ↓
[STEP 3] LLM trigger check
         Call LLM if: confidence<0.70 OR bank=UNKNOWN OR card=UNKNOWN OR doc_type=UNKNOWN
         ↓
         classify_with_llm() → Mistral via Ollama
         apply_llm_override() → LLM wins only if conf > rule_conf + 0.10
         Master doc protection → LLM can NEVER override is_master=True
  ↓
[STEP 4] Output: PDF copy + metadata JSON + logs
```

### Document Types

| Code | Full Name | Signals |
|---|---|---|
| `MITC` | Most Important Terms & Conditions | Fees, interest rate, annual fee, schedule of charges |
| `TNC` | Terms and Conditions | Cardmember agreement, governing law, exclusions |
| `BR` | Benefits & Rewards | Cashback, reward points, welcome bonus, earn rate |
| `LG` | Lounge Guide | Airport lounge, priority pass, domestic lounge |

**Priority order:** MITC → TNC → BR → LG. When signals from multiple types appear (e.g. a MITC that mentions "terms and conditions" in its body), MITC always wins.

### Master Documents

Some banks publish a single document that applies to ALL their credit cards (e.g. HDFC's common MITC). These are detected before card detection runs. When found:
- Card name is set to `"MASTER"` (never a specific card name)
- File is placed in `processed_docs/HDFC_MASTER/`
- The LLM can **never** override this — master classification is permanent once detected

### Confidence Scoring

Overall confidence is a weighted average:
- Doc type: 50%
- Bank: 30%
- Card: 20%

Files below 0.70 go to `needs_review/` (or get the LLM fallback).

---

## 7. Metadata JSON Format

Every processed PDF gets a `.json` file with the same base name:

```
HDFC_Millennia_MITC_2026.pdf
HDFC_Millennia_MITC_2026.json  ← this file
```

Example content:

```json
{
  "bank": "HDFC",
  "card": "Millennia",
  "doc_type": "MITC",
  "is_master": false,
  "confidence": 0.9125,
  "classification_source": "rule_based",
  "source_file": "hdfc_millennia_doc.pdf",
  "output_file": "HDFC_Millennia_MITC_2026.pdf",
  "llm_reason": null,
  "data_quality": "complete",
  "processing_timestamp": "2026-03-26T14:30:00"
}
```

**Field reference:**

| Field | Values | Meaning |
|---|---|---|
| `bank` | `HDFC`, `SBI`, etc. | Short bank code |
| `card` | `Millennia`, `MASTER`, etc. | Card product name |
| `doc_type` | `MITC`, `TNC`, `BR`, `LG` | Document category |
| `is_master` | `true` / `false` | Whether this is a bank-wide collective document |
| `confidence` | `0.0` – `1.0` | Classification confidence |
| `classification_source` | `rule_based` / `llm` | Which system provided the final result |
| `llm_reason` | string or `null` | LLM's one-sentence explanation (only when `classification_source=llm`) |
| `data_quality` | `complete`, `partial`, `insufficient` | Whether required docs (MITC + BR) are available for this card |
| `processing_timestamp` | ISO 8601 | When this file was processed |

The RAG system reads these JSON files during vector DB indexing to attach metadata to each text chunk.

---

## 8. Logs Explained

### `summary.csv`

One row per file. Columns: File Name, Bank, Card, Is_Master, DocType, Confidence, Reason, Status.

Open in Excel to filter by Status = `NEEDS_REVIEW` or `ERROR` to quickly find problem files.

### `preprocess_log.txt`

Full per-file audit trail. For each file shows:
- Bank detection result + confidence + reason
- Whether master doc detection fired and why
- Card detection result + confidence + reason
- Doc type detection result + confidence + reason
- Pages read, final status

### `hybrid_classification_log.txt`

Only files where the LLM was called. Shows the rule result, LLM result, and which one won. Use this to tune the `LLM_OVERRIDE_MARGIN` if LLM overrides are too aggressive or too conservative.

### `missing_docs_report.csv`

After all files are processed, this checks every expected card against a list of required documents. Statuses:
- `COMPLETE` — MITC and BR both found
- `PARTIAL` — required docs present, optional (TNC, LG) missing
- `CRITICAL` — one or more required docs (MITC or BR) missing
- `NOT_FOUND` — no documents found for this card at all

### `coverage_dashboard.xlsx`

Visual colour-coded grid: rows = cards, columns = doc types. Green = found, red = missing required, yellow = missing optional. Use this to quickly see which cards need more documents before the RAG system is ready.

### Console output tags

When the pipeline runs, look for these tags to understand what's happening:

```
[STEP 1] Extracting text from PDF
[OCR]    Running OCR on ...              ← only for scanned PDFs
[STEP 2] Rule result: bank=HDFC | card=Millennia | ...
[FIX #2] Card narrowed: 'X' → 'Y'       ← cross-bank correction
[STEP 3] LLM triggered — reasons: ...
[LLM REQUEST]  Attempt 1/2 — model=mistral
[LLM RESPONSE TIME]  12.4s
[LLM OUTPUT]   bank=HDFC | card=Millennia | ...
[LLM FAILURE]  Timeout after 120s        ← problem indicator
[DECISION]     LLM OVERRIDE applied / Rules kept
[STEP 4] Final result — ...
[OUTPUT] Saved: HDFC_Millennia_MITC_2026.pdf
```

---

## 9. Common Errors and Fixes

### LLM always times out

**Symptom:** `[LLM FAILURE] Timeout after 120s` on every file

**Causes and fixes:**

1. **Ollama is not running:**
   ```bash
   ollama serve
   ```
   Check that `http://localhost:11434` is reachable in your browser.

2. **Mistral model is not downloaded:**
   ```bash
   ollama pull mistral
   ```

3. **Running on CPU (slow machine):**
   Increase `REQUEST_TIMEOUT` in `llm_classifier.py`:
   ```python
   REQUEST_TIMEOUT = 180   # or 240 for very slow CPUs
   ```
   First-request latency on CPU can be 60–90 s due to model loading. Subsequent requests are faster.

4. **Another process is using too much RAM:**
   Mistral 7B requires ~5 GB of RAM. Close other heavy applications.

5. **Test with a direct call:**
   ```bash
   ollama run mistral "classify this: HDFC Millennia MITC"
   ```
   If this works, the issue is with timeout configuration. If it hangs, Ollama itself has a problem.

### OCR not working on Windows

**Symptom:** `[OCR] OCR failed` or `TesseractNotFoundError`

**Fix:**
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it and note the path (e.g. `C:\Program Files\Tesseract-OCR\`)
3. Open `preprocess_with_llm.py`, find `TESSERACT_CMD` inside `_extract_with_ocr()`, and update it:
   ```python
   TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
4. Re-run the pipeline

### Wrong doc_type detected (MITC vs TNC vs BR)

**Symptom:** A MITC document is classified as TNC

**Diagnosis:** Run with `--debug` and check what text was extracted from the first 500 characters. If the header says "Terms and Conditions" but the body contains fee schedules, the title phrase detection is picking up the header.

**Fix:** Add the specific header phrase from that PDF to `MITC_TITLE_PHRASES` in `preprocess.py`:
```python
MITC_TITLE_PHRASES = [
    "most important terms",
    "your new phrase here",   # ← add here
    ...
]
```

### Card detected from wrong bank (e.g. SBI_Millennia)

**Symptom:** A file ends up as `SBI_Millennia_BR_2026.pdf` but should be `SBI_Cashback_BR_2026.pdf`

**Cause:** The word "Millennia" appeared somewhere in the SBI document body (e.g. in a comparison table), and was matched before "Cashback".

**Fix (automatic):** The `_narrow_card_to_bank()` function in `preprocess_with_llm.py` should catch this. If it's still wrong, check that the card is in `BANK_CARDS["SBI"]` and not listed in another bank's entry.

**Fix (manual):** Add the card to the correct bank in `BANK_CARDS`:
```python
BANK_CARDS = {
    "SBI": ["Cashback", "Elite", "your_new_card", ...],
}
```

### Master document classified as specific card

**Symptom:** `HDFC_MASTER_MITC_2026.pdf` becomes `HDFC_Infinia_MITC_2026.pdf`

**Cause:** The master signal phrase from that specific PDF is not in `MASTER_DOC_SIGNALS`.

**Fix:** Run with `--debug`, find the header text, and add the collective signal phrase to `preprocess.py`:
```python
MASTER_DOC_SIGNALS = [
    "applicable to all credit cards",
    "the phrase from your pdf",   # ← add here
]
```

### Bank not detected

**Symptom:** `Bank: NOT FOUND` in `summary.csv`, file goes to `needs_review/`

**Fix:** Run with `--debug`. Find the bank name as it appears in the PDF text, then add it to `BANK_ALIASES` in `preprocess.py`:
```python
BANK_ALIASES = {
    "HDFC": ["hdfc bank ltd", "hdfc bank", "the phrase from your pdf", "hdfc"],
}
```

### No text extracted from PDF

**Symptom:** File goes to `needs_review/` with empty bank/card detection, OCR message appears

**Cause 1:** The PDF is a scanned image with no embedded text layer.
**Fix:** Install pytesseract and the Tesseract binary (see Setup Step 4). The OCR fallback will activate automatically.

**Cause 2:** PDF is encrypted or password-protected.
**Fix:** Remove the password protection before placing the file in `raw_docs/`.

---

## 10. Troubleshooting Guide

### Diagnostic checklist — when a file goes to `needs_review/`

1. Open `data/logs/summary.csv` and find the file row
2. Read the `Reason` column
3. Open `data/logs/preprocess_log.txt` and find the file section for full detail
4. Run with `--debug` to see the extracted text:
   ```bash
   python data_pipeline/preprocess_with_llm.py --debug --no-llm
   ```
5. Based on what you see:
   - If text is empty → OCR issue (see above)
   - If bank is wrong → add phrase to `BANK_ALIASES`
   - If card is wrong → check `BANK_CARDS` in `preprocess_with_llm.py`
   - If doc_type is wrong → add phrase to relevant `_TITLE_PHRASES` or `_KEYWORDS` in `preprocess.py`

### How to debug LLM decisions

1. Run the standalone LLM test:
   ```bash
   python data_pipeline/llm_classifier.py
   ```
2. Check `data/logs/hybrid_classification_log.txt` after a run
3. Look for `[LLM REQUEST]`, `[LLM RESPONSE TIME]`, and `[LLM FAILURE]` in console output
4. If LLM is overriding good rule results too often, increase `LLM_OVERRIDE_MARGIN` in `preprocess_with_llm.py`:
   ```python
   LLM_OVERRIDE_MARGIN = 0.15   # default is 0.10
   ```

### Performance tips

- Run `--no-llm` first to check rule-based results, then re-run with LLM for only the files in `needs_review/`
- LLM is only called for files that fail the multi-condition trigger (low confidence or missing fields)
- On GPU, Mistral inference takes 2–5 s per file. On CPU, 15–60 s per file (first call may be slower due to model loading)
- The pipeline processes files sequentially — parallel processing is not currently implemented

### File already processed but classification was wrong

1. Delete the wrongly classified file from `processed_docs/`
2. Fix the configuration (add missing phrase, update `BANK_CARDS`, etc.)
3. Re-run — the duplicate detection will skip files that already exist, so only the deleted file will be reprocessed

### Configuration reference

All user-editable settings are at the top of `preprocess.py`. Key settings:

| Setting | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.70` | Files below this go to `needs_review/` |
| `DEFAULT_YEAR` | `"2026"` | Year used when not found in PDF |
| `PAGE_TIERS` | `[3, 5, 10]` | Pages read per attempt (do not reduce) |
| `REQUIRED_DOCS` | `["MITC", "BR"]` | Docs needed for COMPLETE coverage status |
| `OPTIONAL_DOCS` | `["TNC", "LG"]` | Docs needed for PARTIAL vs COMPLETE |

LLM-specific settings in `llm_classifier.py`:

| Setting | Default | Effect |
|---|---|---|
| `REQUEST_TIMEOUT` | `120` | Seconds to wait for Ollama response |
| `MAX_RETRIES` | `2` | Number of retry attempts on failure |
| `LLM_TEXT_WINDOW` | `2000` | Characters sent to the LLM |
| `OLLAMA_MODEL` | `"mistral"` | Model name (must be pulled first) |

---

*Last updated: March 2026*