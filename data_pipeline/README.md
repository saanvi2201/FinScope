# Credit Card Document Preprocessing Pipeline

A rule-based, explainable Python pipeline for cleaning, classifying, renaming,
and organising raw credit card PDF documents into a structured dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Folder Structure](#3-folder-structure)
4. [Setup Instructions](#4-setup-instructions-windows--vs-code)
5. [How to Run](#5-how-to-run)
6. [How Files Are Named](#6-how-files-are-named)
7. [Classification Logic](#7-classification-logic)
8. [Master / Collective Document Handling](#8-master--collective-document-handling)
9. [Confidence & Reason System](#9-confidence--reason-system)
10. [Adaptive Page Reading](#10-adaptive-page-reading)
11. [Logs Explained](#11-logs-explained)
12. [Document Coverage Validation](#12-document-coverage-validation)
13. [The `needs_review` Folder](#13-the-needs_review-folder)
14. [How to Modify Rules](#14-how-to-modify-rules)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

This pipeline takes messy, unstructured PDF files from a raw documents folder and:

- Extracts text from the first few pages of each PDF
- Detects the document type (MITC, TNC, BR, LG), bank name, and card name
- Handles collective/master documents that apply to all cards for a bank
- Renames each file to a clean, consistent format using bank short codes
- Organises files into labelled subfolders
- Routes uncertain files to a separate review folder
- Validates that every expected card has its required documents after processing
- Generates detailed logs and CSV reports for auditing

The system is fully rule-based and deterministic — no machine learning is involved.
Every decision is logged with a human-readable explanation.

---

## 2. Features

| Feature | Description |
|---|---|
| Text extraction | Supports pdfplumber and PyMuPDF (auto-fallback) |
| Adaptive page reading | Reads 3 → 5 → 10 pages progressively, stops early when confident |
| Bank alias mapping | Maps full bank names to short codes for clean filenames |
| Document type detection | Title/header matching + keyword frequency scoring |
| Card detection | Text search with ordering rules for overlapping names |
| Master doc detection | Identifies collective bank-wide docs, routes to `BANK_MASTER/` folder |
| Confidence scoring | Weighted 0–1 score with explainable reasons per file |
| Coverage validation | Checks every expected card has MITC + BR after processing |
| Missing docs report | Generates `missing_docs_report.csv` with COMPLETE/PARTIAL/CRITICAL/NOT_FOUND status |
| Dry-run mode | Preview all actions without touching any files |
| Debug mode | Print extracted text samples for diagnosing detection failures |
| Duplicate detection | Skip files that already exist at the destination |
| CSV summary | One-row-per-file summary with `Is_Master` column for easy filtering |
| Detailed log | Full per-file audit trail in plain text including master doc reasoning |

---

## 3. Folder Structure

```
project_root/
│
├── data/
│   ├── raw_docs/                ← Place your input PDFs here (never modified)
│   │
│   ├── processed_docs/          ← Organised output (created automatically)
│   │   ├── HDFC_Millennia/      ← One subfolder per bank+card combination
│   │   │   ├── HDFC_Millennia_MITC_2024.pdf
│   │   │   └── HDFC_Millennia_BR_2024.pdf
│   │   ├── BOB_Eterna/
│   │   │   └── BOB_Eterna_MITC_2024.pdf
│   │   └── HDFC_MASTER/         ← Collective/bank-wide docs go here
│   │       ├── HDFC_MASTER_MITC_2024.pdf
│   │       └── HDFC_MASTER_TNC_2024.pdf
│   │
│   ├── needs_review/            ← Low-confidence or incomplete detections
│   │
│   └── logs/
│       ├── summary.csv              ← One-row-per-file processing summary
│       ├── preprocess_log.txt       ← Detailed per-file audit trail
│       └── missing_docs_report.csv  ← Coverage gaps per expected card
│
└── data_pipeline/
    └── preprocess.py            ← Main script (all config at the top)
```

> Raw files in `raw_docs/` are **never deleted or modified**.
> The script only copies files to destination folders.

---

## 4. Setup Instructions (Windows + VS Code)

### Prerequisites

- Python 3.10 or higher
- VS Code with the Python extension installed

### Step 1 — Place the project folder

Put the project anywhere on your machine, for example:

```
C:\Users\YourName\projects\credit_card_pipeline\
```

### Step 2 — Open a terminal in VS Code

Press `` Ctrl + ` `` (backtick) to open the integrated terminal.

### Step 3 — Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

This keeps project packages isolated from the rest of your Python installation.
Once activated, your terminal prompt will show `(venv)`.

### Step 4 — Install dependencies

```bash
pip install pdfplumber pandas
```

Or if you prefer PyMuPDF:

```bash
pip install pymupdf pandas
```

You only need **one** PDF library. The script tries pdfplumber first, then falls back to PyMuPDF automatically.

### Step 5 — Place your PDF files

Copy your raw PDF files into:

```
project_root\data\raw_docs\
```

---

## 5. How to Run

All commands are run from `project_root/`.

### Normal run

```bash
python data_pipeline\preprocess.py
```

### Dry-run mode — no files are copied, only logged

```bash
python data_pipeline\preprocess.py --dry-run
```

Use this first to preview everything the pipeline would do before committing.

### Debug mode — prints extracted text for each PDF

```bash
python data_pipeline\preprocess.py --debug
```

Prints the first ~600 characters of extracted text per file. Use this to diagnose
why a file is going to `needs_review/` or getting the wrong classification.

### Combined

```bash
python data_pipeline\preprocess.py --dry-run --debug
```

> **PowerShell users:** If running `python` opens the Microsoft Store instead of running the script, use `py` instead: `py data_pipeline\preprocess.py`

---

## 6. How Files Are Named

Every processed file follows this exact format:

```
[BANK_SHORT_CODE]_[CARD_NAME]_[DOCTYPE]_[YEAR].pdf
```

| Part | Source | Example |
|---|---|---|
| `BANK_SHORT_CODE` | Key from `BANK_ALIASES` in config | `BOB`, `SCB`, `HDFC` |
| `CARD_NAME` | Match from `CARDS` list (spaces → underscores) | `Eterna`, `Amazon_Pay` |
| `DOCTYPE` | Detected doc type | `MITC`, `TNC`, `BR`, `LG` |
| `YEAR` | Extracted from PDF text, or `DEFAULT_YEAR` | `2024` |

### Regular card examples

| Input PDF (messy name) | Output filename |
|---|---|
| `bank_of_baroda_doc.pdf` | `BOB_Eterna_MITC_2024.pdf` |
| `stanchart_smart.pdf` | `SCB_Smart_BR_2024.pdf` |
| `hdfc_tata.pdf` | `HDFC_Tata_Neu_Infinity_BR_2025.pdf` |
| `icici_amazon.pdf` | `ICICI_Amazon_Pay_TNC_2023.pdf` |

### Why short codes instead of full bank names?

The PDF text might say "Bank of Baroda" or "Standard Chartered Bank" in full.
`BANK_ALIASES` maps those full phrases to clean short codes for filenames.
This is entirely configurable — see [How to Modify Rules](#14-how-to-modify-rules).

### Master document examples

Collective/bank-wide documents get `MASTER` as the card name:

| Scenario | Output filename | Output folder |
|---|---|---|
| HDFC common MITC (all cards) | `HDFC_MASTER_MITC_2024.pdf` | `processed_docs/HDFC_MASTER/` |
| SBI collective TNC | `SBI_MASTER_TNC_2024.pdf` | `processed_docs/SBI_MASTER/` |

---

## 7. Classification Logic

### Document Types

| Code | Full Name | Description |
|---|---|---|
| `MITC` | Most Important Terms & Conditions | Fee schedules, interest rates, charges |
| `TNC` | Terms and Conditions | Legal agreement, dispute resolution |
| `BR` | Benefits & Rewards | Cashback, points, welcome bonuses |
| `LG` | Lounge Guide | Airport lounge access details |

### Layer 1 — Title / Header Detection

The script examines the **first 500 characters** of the extracted text, where the document title usually appears.

If it finds an exact phrase such as:
- `"Most Important Terms"` → classified as **MITC**
- `"Terms and Conditions"` → classified as **TNC**
- `"Benefits Guide"` → classified as **BR**
- `"Lounge Access Guide"` → classified as **LG**

A title match gives a strong confidence boost (+10 percentage points on top of keyword score).

### Layer 2 — Keyword Frequency Scoring

The script counts how many keywords from each document type's dictionary appear in the extracted text.

For example, a document containing `interest rate`, `annual fee`, `late payment fee`, and `minimum amount due` scores highly for **MITC**.

Each type has its own keyword list in the config (`MITC_KEYWORDS`, `TNC_KEYWORDS`, etc.). Scores are normalised 0–1. The highest-scoring type wins. If a title match and keyword scores agree, confidence is high. If they disagree, the title match takes priority.

### Bank Detection

The script searches for phrases listed in `BANK_ALIASES`. When a phrase matches, the **short code key** is used in the filename — not the full phrase. For example, finding "Bank of Baroda" in the text results in `BOB` in the filename.

Detection checks the **header zone** (first 500 characters) first, giving a confidence of 0.95. Matches deeper in the body give 0.80. If text search fails, the original filename is searched as a fallback (confidence 0.55).

### Card Detection

The `CARDS` list is scanned top-to-bottom. The first match found is used.

**Ordering matters:** if a card name contains another card name, the more specific one must be listed first. For example, `"Platinum Travel"` must appear before `"Platinum"` to avoid a Platinum Travel document being matched as just "Platinum".

---

## 8. Master / Collective Document Handling

Some banks publish a single document that legally applies to **all** their credit cards rather than issuing one per card variant. For example, HDFC publishes one common MITC covering Infinia, Millennia, Regalia, and every other variant.

### Why this matters

Without special handling, a collective HDFC MITC would either:
- Wrongly match the first card name it finds in the text (e.g. "Infinia"), OR
- Fail card detection and land in `needs_review/` with `UNKNOWN_CARD`

Both outcomes are wrong. The master doc system prevents this.

### How it works

1. After bank detection, the text is scanned for phrases in `MASTER_DOC_SIGNALS` **before** card detection runs
2. If any signal phrase is found, card name is set to `"MASTER"` and individual card detection is skipped entirely
3. The file is placed in a dedicated `BANK_MASTER/` folder

### Signal phrase examples

```
"applicable to all credit cards"
"all hdfc bank credit cards"
"irrespective of the card variant"
"common terms and conditions"
```

### Output

```
Detected signal: "all hdfc bank credit cards"
Filename:  HDFC_MASTER_MITC_2024.pdf
Folder:    processed_docs/HDFC_MASTER/
Status:    MASTER_DOC
```

### Coverage validation for master docs

Master docs are tracked in `EXPECTED_CARDS` just like individual cards:

```python
"HDFC MASTER",   # HDFC's collective doc
"SBI MASTER",    # SBI's collective doc
```

If a bank's master document hasn't been processed, the coverage report will flag it as `CRITICAL`.

### Adding new signal phrases

In `preprocess.py`, find `MASTER_DOC_SIGNALS` and add a new line:

```python
MASTER_DOC_SIGNALS = [
    "applicable to all credit cards",
    "your new phrase here",   # ← add here
]
```

---

## 9. Confidence & Reason System

Every detection (bank, card, doc type) produces three values:

```
value      → what was detected    e.g. "MITC", "HDFC", "Millennia"
confidence → how certain (0–1)    e.g. 0.91
reasons    → why this decision    e.g. ['Found "Most Important Terms" in header']
```

### Overall Confidence

The overall file confidence is a weighted average of the three individual scores:

| Component | Weight |
|---|---|
| Document type | 50% |
| Bank name | 30% |
| Card name | 20% |

If the overall confidence falls below **0.70**, the file is routed to `needs_review/`.

### Example log output for a regular file

```
BANK: HDFC (0.95)
REASONS:
  • Found 'hdfc bank' → mapped to short code 'HDFC' in header

MASTER / COLLECTIVE DOCUMENT: NO
  No collective signals found — proceeded with individual card detection.

CARD: Millennia (0.90)
REASONS:
  • Found 'Millennia' in document header

DOC TYPE: MITC (0.91)
REASONS:
  • Found title phrase "most important terms" in document header
  • Keyword matches for MITC: interest rate, annual fee, late payment fee
  • Low presence of BR-related keywords (score=0.05)
```

### Example log output for a master document

```
BANK: HDFC (0.95)
REASONS:
  • Found 'hdfc bank' → mapped to short code 'HDFC' in header

MASTER / COLLECTIVE DOCUMENT: YES
  Trigger signal : "all hdfc bank credit cards"
  Confidence     : 0.92
  Decision       : Individual card detection was SKIPPED.
                   Card name set to MASTER.
  Destination    : processed_docs/HDFC_MASTER/

CARD: MASTER (0.92)
REASONS:
  • MASTER DOCUMENT — applies to all cards for this bank
  • Trigger signal: "all hdfc bank credit cards"
  • Individual card detection was intentionally skipped
```

---

## 10. Adaptive Page Reading

Instead of reading the entire PDF (slow), the script reads pages progressively:

```
Tier 1: Read first 3 pages → run detection
         If confidence ≥ 0.70 → STOP (fast path)

Tier 2: Read first 5 pages → re-run detection
         If confidence ≥ 0.70 → STOP

Tier 3: Read first 10 pages → final detection
         Hard maximum — never reads the whole PDF
```

This also applies to master doc detection — if the collective signal is on page 4, Tier 2 will catch it.

### Log messages for fallback

```
[INFO] Extracting text (3 pages) from: doc1.pdf
[INFO] Confidence after 3 pages: 0.52 | HDFC | None | MITC
[INFO] Confidence 0.52 below threshold 0.70. Retrying with 5 pages...
[INFO] Extracting text (5 pages) from: doc1.pdf
[INFO] Confidence improved to 0.81 after reading 5 pages.
```

### Configuring the page tiers

In `preprocess.py`:

```python
PAGE_TIERS = [3, 5, 10]       # default
PAGE_TIERS = [2, 4, 8]        # read fewer pages (faster, less thorough)
PAGE_TIERS = [5, 10, 20]      # read more pages (slower, more thorough)
```

---

## 11. Logs Explained

### `summary.csv`

A spreadsheet-friendly one-row-per-file summary. Open in Excel or Google Sheets.

| Column | Description |
|---|---|
| File Name | Original filename from `raw_docs/` |
| Bank | Detected bank short code (e.g. `BOB`, `HDFC`) |
| Card | Detected card name, or `MASTER` for collective docs |
| Is_Master | `YES` if this is a collective/bank-wide document, `NO` otherwise |
| DocType | Detected document type (`MITC`, `TNC`, `BR`, `LG`) |
| Confidence | Overall confidence score (0–1) |
| Reason | Top reasons for the classification decision |
| Status | `SUCCESS`, `MASTER_DOC`, `NEEDS_REVIEW`, `ERROR`, `DUPLICATE_SKIPPED` |

**Useful filters in Excel:**
- Filter `Is_Master = YES` to see all collective documents
- Filter `Status = NEEDS_REVIEW` to find files needing manual attention
- Filter `Status = ERROR` to find files that crashed during processing

### `preprocess_log.txt`

A detailed plain-text audit trail. For every file it records:

- **Bank** — detected short code, confidence score, and exact phrase matched
- **Master doc check** — whether a collective signal was found, what phrase triggered it, and what decision was made (skip card detection or proceed normally)
- **Card** — detected card name, confidence, and where it was found
- **Doc type** — classification with keyword evidence
- **Pages read** — how many pages were needed (shows if fallback tiers were used)
- **Status** — final outcome

This log is the primary tool for understanding why any file was classified as it was.

### `missing_docs_report.csv`

Generated after all files are processed. One row per expected card.

| Column | Description |
|---|---|
| Card | Expected card (e.g. `HDFC Millennia`, `HDFC MASTER`) |
| Present_Docs | Doc types successfully processed (e.g. `BR, MITC`) |
| Missing_Docs | Doc types absent (e.g. `TNC, LG`) |
| Status | `COMPLETE`, `PARTIAL`, `CRITICAL`, or `NOT_FOUND` |

---

## 12. Document Coverage Validation

This step runs **automatically after all files are processed**. It is read-only — it does not move or modify any files.

### What it does

1. Scans all successfully processed files and builds a coverage map: `"HDFC Millennia" → {MITC, BR}`
2. Compares every entry in `EXPECTED_CARDS` against `REQUIRED_DOCS`
3. Assigns a status and logs warnings for gaps
4. Writes `missing_docs_report.csv`

### Status rules

| Status | Meaning |
|---|---|
| `COMPLETE` | All required (`MITC`, `BR`) and optional (`TNC`, `LG`) docs present |
| `PARTIAL` | Required docs present, but one or more optional docs missing |
| `CRITICAL` | One or more required docs (`MITC` or `BR`) missing |
| `NOT_FOUND` | No documents at all found for this card |

### Console output

```
[WARNING] Missing MITC for SBI Cashback
[WARNING] Missing BR for AXIS Magnus
[WARNING] Only [BR] found for KOTAK Flipkart — optional docs missing: TNC, LG
[WARNING] No documents found at all for: HDFC MASTER
```

### Configuring required vs optional docs

```python
REQUIRED_DOCS = ["MITC", "BR"]   # CRITICAL if missing
OPTIONAL_DOCS = ["TNC", "LG"]    # PARTIAL if missing
```

### Adding expected cards

In `preprocess.py`, find `EXPECTED_CARDS`. The format is `"BANK_SHORT_CODE CardName"`. The short code must exactly match a key in `BANK_ALIASES`, and the card name must exactly match an entry in `CARDS`.

```python
EXPECTED_CARDS = [
    "HDFC Millennia",
    "BOB Eterna",
    "HDFC MASTER",     # ← master doc entry
    # add more here
]
```

---

## 13. The `needs_review` Folder

A file is routed to `needs_review/` when **any** of the following are true:

- Overall confidence < 0.70
- Bank name not detected
- Card name not detected (and it is not a master document)

Files in `needs_review/` are still renamed using whatever was detected, with `UNKNOWN_BANK` or `UNKNOWN_CARD` as placeholders.

### What to do with these files

1. Open `summary.csv` and filter `Status = NEEDS_REVIEW`
2. Read the `Reason` column to understand why detection failed
3. Open `preprocess_log.txt` and find the file's entry for full detail
4. Fix the issue — either:
   - Add the missing bank/card/signal phrase to the config, then re-run, or
   - Manually rename and move the file to the correct `processed_docs/` subfolder

---

## 14. How to Modify Rules

All editable configuration is at the **top of `preprocess.py`** inside the section marked `USER EDITABLE CONFIGURATION`. Nothing below that marker needs to be changed for typical adjustments.

### Add a new bank

```python
BANK_ALIASES = {
    ...
    "MYNEWBANK": ["my new bank limited", "my new bank", "mnb"],
}
```

Then add it to `EXPECTED_CARDS`:
```python
"MYNEWBANK SomeCard",
```

### Add a new card

```python
CARDS = [
    "New Specific Card Name",   # ← add BEFORE any shorter name it contains
    "New Card",
    ...
]
```

### Add a master doc signal phrase

```python
MASTER_DOC_SIGNALS = [
    "applicable to all credit cards",
    "your new phrase here",   # ← add here, case-insensitive
]
```

### Add a bank to the collective docs list

If you discover a new bank publishes collective documents, add its short code to `MASTER_DOC_BANKS` (informational) and add its `EXPECTED_CARDS` entry:

```python
MASTER_DOC_BANKS = ["HDFC", "SBI", ..., "MYNEWBANK"]

# In EXPECTED_CARDS:
"MYNEWBANK MASTER",
```

### Add keywords for a document type

```python
MITC_KEYWORDS = [
    "interest rate", "annual fee", ...,
    "processing fee",   # ← add here
]
```

### Add a title/header phrase for a document type

```python
MITC_TITLE_PHRASES = [
    "most important terms",
    "mitc",
    "schedule of charges",   # ← add here
]
```

### Add a completely new document type

1. Add `FD_KEYWORDS = ["fee schedule", "tariff card", ...]`
2. Add `FD_TITLE_PHRASES = ["schedule of fees and charges", ...]`
3. Inside `detect_doc_type()`, add `"FD"` to the `title_hits` and `kw_data` dicts

### Change the confidence threshold

```python
CONFIDENCE_THRESHOLD = 0.70   # increase = stricter, decrease = more lenient
```

### Change the default year

```python
DEFAULT_YEAR = "2026"   # used when no year is found in the document text
```

### Change how many pages are read

```python
PAGE_TIERS = [3, 5, 10]   # default
PAGE_TIERS = [5, 10, 20]  # more thorough, slower
```

---

## 15. Troubleshooting

### No PDF library found

**Error:** `No PDF library found.`

**Fix:**
```bash
pip install pdfplumber
# or
pip install pymupdf
```

---

### Running `python` opens the Microsoft Store instead

**Symptom:** On PowerShell, `python` triggers the Store popup and nothing happens.

**Fix (Option 1 — permanent):** Open Windows Settings → search "Manage app execution aliases" → turn OFF both Python entries.

**Fix (Option 2 — immediate):** Use `py` instead of `python`:
```bash
py data_pipeline\preprocess.py
```

---

### No text extracted from PDF

**Symptom:** Files go to `needs_review/` with no bank/card detected.

**Cause:** The PDF is scanned (image-only, no embedded text layer).

**Fix:** pdfplumber and PyMuPDF cannot read scanned images. Use an OCR tool such as `pytesseract` or Adobe Acrobat to convert the PDF to a text-searchable version first, then re-run.

---

### Bank not detected

**Symptom:** `Bank: NOT FOUND` in `summary.csv`.

**Fix:** The bank name in the PDF text doesn't match any phrase in `BANK_ALIASES`. Open `preprocess_log.txt`, find the file, and check what text was extracted. Then add the phrase you see in the PDF to the correct bank's alias list:

```python
BANK_ALIASES = {
    "HDFC": ["hdfc bank ltd", "hdfc bank", "hdfc", "new phrase from pdf"],
}
```

---

### Card not detected

**Symptom:** `Card: NOT FOUND` or `UNKNOWN_CARD` in filename.

**Possible causes:**
1. The card name isn't in the `CARDS` list — add it
2. A shorter card name matched before the correct one — check ordering rules

---

### Collective document wrongly matched to a specific card

**Symptom:** An HDFC common MITC is being named `HDFC_Infinia_MITC_2024.pdf` instead of `HDFC_MASTER_MITC_2024.pdf`.

**Fix:** The master signal phrase from that specific PDF isn't in `MASTER_DOC_SIGNALS`. Run with `--debug` to see the extracted text, find the phrase that indicates it's a collective document, and add it:

```python
MASTER_DOC_SIGNALS = [
    "applicable to all credit cards",
    "the phrase you found in the pdf",   # ← add here
]
```

---

### Wrong document type detected

**Symptom:** A MITC document is classified as TNC.

**Fix:** Add stronger signals for MITC — either a title phrase or more keywords:

```python
MITC_TITLE_PHRASES = ["most important terms", "mitc", "your title here"]
MITC_KEYWORDS      = ["interest rate", "annual fee", ..., "your keyword"]
```

---

### Files not appearing in `processed_docs/`

**Check 1:** Are you running in `--dry-run` mode? Dry-run logs actions but copies nothing.

**Check 2:** Did the file go to `needs_review/` instead? Check `summary.csv`.

**Check 3:** Was it detected as a master doc? Look for a `BANK_MASTER/` subfolder.

---

### `pandas` not installed

The script falls back to Python's built-in `csv` module automatically. All CSV files will still be generated correctly. Install pandas only if you want faster CSV generation for very large datasets.

---

### Windows path errors

The script uses Python's `pathlib.Path` which handles Windows paths correctly. Always run from the project root:

```bash
cd C:\Users\YourName\projects\credit_card_pipeline
python data_pipeline\preprocess.py
```

---

*Last updated to reflect: BANK_ALIASES short code mapping, Master/Collective document detection, adaptive page reading, coverage validation with MASTER entries, and updated log formats.*