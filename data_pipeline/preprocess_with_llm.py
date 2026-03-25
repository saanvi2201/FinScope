# =============================================================================
# preprocess_with_llm.py — Hybrid (Rule + LLM) Preprocessing Pipeline
# =============================================================================
#
# ── CHANGES IN THIS VERSION ──────────────────────────────────────────────────
#
#  FIX #2 — Bank-specific card detection (CRITICAL)
#    _narrow_card_to_bank() is called after rule-based detection when the
#    detected card does not belong to the detected bank.
#    A BANK_CARDS mapping defines which cards each bank can issue.
#    If the rule result is cross-bank (e.g. card=Swiggy but bank=SBI),
#    the card is re-evaluated against only the detected bank's card list.
#    This eliminates the Millennia→Swiggy style mismatches.
#
#  FIX #4 — Repeated processing bug (CRITICAL)
#    The main loop is a plain `for pdf_path in pdf_files:` with a single
#    call to _process_one_file(). That helper NEVER calls itself or re-triggers
#    the outer loop. Retry logic inside classify_with_llm() is limited to
#    MAX_RETRIES=2 LLM HTTP retries only — it does NOT restart file processing.
#
#  FIX #5 — OCR fallback for scanned PDFs
#    If BOTH pdfplumber AND PyMuPDF return empty text, _extract_with_ocr()
#    is called (requires pytesseract + Pillow + pymupdf for rendering).
#    OCR is gated: triggered only when both standard extractors fail.
#    If pytesseract is not installed, OCR is skipped with a clear warning.
#
#  FIX #6 — Multi-condition LLM trigger (IMPORTANT)
#    Old: call LLM only if confidence < 0.70
#    New: call LLM if ANY of:
#         • confidence < CONFIDENCE_THRESHOLD (0.70)
#         • bank == None / "UNKNOWN"
#         • card == None / "UNKNOWN"
#         • doc_type is absent or "UNKNOWN"
#    This catches cases where confidence is high (e.g. bank is well-detected)
#    but a specific field is still missing.
#
#  FIX #8 — Logging improvements
#    [STEP 1] Extraction
#    [STEP 2] Rule classification → logs rule confidence
#    [STEP 3] LLM decision → logs rule conf, LLM conf, and override decision
#    [OUTPUT] Saved file
#    Each step logs its key numeric values explicitly.
#
#  UNCHANGED:
#    Folder structure — processed_docs/, needs_review/, logs/
#    All imports from preprocess.py — rule-based system untouched
#    Metadata JSON format — same schema
#    Coverage validation, write_detail_log, write_summary_csv — all intact
#    LLM_OVERRIDE_MARGIN = 0.10 — kept
#    Page reading — not reduced
#
# =============================================================================
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#
#  STEP 0 — Install Python dependencies (run once)
#    pip install pdfplumber pymupdf requests pandas openpyxl
#
#    For OCR support (optional — only needed for scanned/image-only PDFs):
#      pip install pytesseract Pillow
#      # Also install Tesseract binary:
#      # macOS:   brew install tesseract
#      # Ubuntu:  sudo apt install tesseract-ocr
#      # Windows: https://github.com/UB-Mannheim/tesseract/wiki
#
#  STEP 1 — Set up Ollama + Mistral (run once)
#    # Install Ollama
#    # macOS/Linux:
#    curl -fsSL https://ollama.ai/install.sh | sh
#    # Windows: download from https://ollama.ai/download
#
#    # Pull Mistral model (~4.1 GB, one-time download)
#    ollama pull mistral
#
#    # Start the Ollama server (keep this running in a separate terminal)
#    ollama serve
#
#  STEP 2 — Place input PDFs
#    Copy your raw PDF files into:
#      <project_root>/data/raw_docs/
#    Example:
#      project_root/
#        data/
#          raw_docs/
#            HDFC_Millennia_doc.pdf
#            SBI_Cashback.pdf
#            ...
#
#  STEP 3 — Run the pipeline (from project root)
#    Normal run (LLM fallback enabled):
#      python data_pipeline/preprocess_with_llm.py
#
#    Skip LLM (rule-based only, faster, no Ollama needed):
#      python data_pipeline/preprocess_with_llm.py --no-llm
#
#    Dry run (simulate only, no files moved or written):
#      python data_pipeline/preprocess_with_llm.py --dry-run
#
#    Debug mode (prints extracted text for each PDF):
#      python data_pipeline/preprocess_with_llm.py --debug
#
#    Combine flags:
#      python data_pipeline/preprocess_with_llm.py --debug --dry-run
#      python data_pipeline/preprocess_with_llm.py --no-llm --dry-run
#
#  STEP 4 — Check outputs
#    Processed PDFs  : data/processed_docs/<BANK>_<CARD>/
#    Metadata JSON   : alongside each PDF (same folder, same base name)
#    Low-confidence  : data/needs_review/
#    Summary table   : data/logs/summary.csv
#    Full audit log  : data/logs/preprocess_log.txt
#    LLM decisions   : data/logs/hybrid_classification_log.txt
#    Coverage report : data/logs/missing_docs_report.csv
#    Coverage grid   : data/logs/coverage_dashboard.xlsx
#
# =============================================================================

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP — add data_pipeline/ to sys.path so preprocess.py imports cleanly
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import (
    # Core detection — rule-based, called first; LLM is fallback only
    run_detection_with_fallback,
    compute_confidence,

    # Text extraction (used for OCR fallback check)
    extract_text,
    detect_year,

    # File I/O — unchanged from original
    generate_filename,
    get_output_folder,
    is_duplicate,
    move_file,

    # Logging + reporting — unchanged
    write_detail_log,
    write_summary_csv,
    write_missing_docs_csv,
    write_coverage_dashboard,
    print_validation_summary,

    # Coverage validation — unchanged
    build_coverage_map,
    validate_coverage,

    # Config constants
    CONFIDENCE_THRESHOLD,
    PAGE_TIERS,
    DEFAULT_YEAR,

    # Path constants — same dir structure as original pipeline
    RAW_DIR,
    PROCESSED_DIR,
    REVIEW_DIR,
    LOG_DIR,
    SUMMARY_CSV,
    DETAIL_LOG,
    MISSING_DOCS_CSV,
    COVERAGE_DASHBOARD,

    # Shared setup
    setup_logging,

    # Card and bank config lists (used for bank-specific card narrowing)
    CARDS,
    BANK_ALIASES,
)

from llm_classifier import classify_with_llm, check_ollama_available

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PIPELINE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# LLM must exceed rule confidence by this margin to override the rule result.
# Prevents marginal LLM wins from overriding well-tuned rules.
LLM_OVERRIDE_MARGIN = 0.10

METADATA_DIR = PROCESSED_DIR   # JSON files co-located with PDFs


# ─────────────────────────────────────────────────────────────────────────────
# FIX #2 — BANK-SPECIFIC CARD MAP
# Maps each bank short code to the card names it can legally issue.
# Used by _narrow_card_to_bank() to reject cross-bank card assignments.
# Only the card names that appear in preprocess.py's CARDS list are included.
# ─────────────────────────────────────────────────────────────────────────────

BANK_CARDS: dict[str, list[str]] = {
    "HDFC": [
        "Infinia", "Diners Club Black", "Diners", "Marriott Bonvoy",
        "Regalia Gold", "Regalia", "Millennia", "Tata Neu Infinity",
        "Tata Neu", "MoneyBack+", "MoneyBack", "Swiggy", "IndianOil",
        "Pixel", "HDFC Millennia",
    ],
    "SBI": [
        "Cashback", "Elite", "Aurum", "SimplyCLICK", "SimplySAVE",
        "BPCL Octane", "BPCL", "IRCTC", "Paytm", "Reliance", "Vistara",
    ],
    "AXIS": [
        "Magnus", "Atlas", "Reserve", "ACE", "Axis ACE", "Flipkart",
        "Airtel", "Select", "Vistara Infinite", "Vistara", "Coral",
    ],
    "ICICI": [
        "Emeralde", "Amazon Pay", "HPCL Super Saver", "Rubyx",
    ],
    "AMEX": [
        "Platinum Travel", "Blue Cash", "Platinum",
    ],
    "HSBC": [
        "Premier", "Live+",
    ],
    "KOTAK": [
        "League Platinum", "Myntra",
    ],
    "SCB": [
        "Smart", "Ultimate",
    ],
    "IDFC": [
        "Millennia",
    ],
    "AU": [
        "Altura",
    ],
    "BOB": [
        "Eterna",
    ],
    "RBL": [
        "World Safari",
    ],
    "INDUSIND": [
        "EazyDiner",
    ],
    "YES": [
        "Prosperity",
    ],
    "ONECARD": [
        "OneCard",
    ],
    "SCAPIA": [
        "Scapia",
    ],
    "JUPITER": [
        "Jupiter",
    ],
    "CITI": [
        "Signature", "Platinum",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# FIX #2 — BANK-SPECIFIC CARD NARROWING
# ─────────────────────────────────────────────────────────────────────────────

def _narrow_card_to_bank(
    bank: str | None,
    card: str | None,
    text_lower: str,
) -> str | None:
    """
    Validates that the detected card belongs to the detected bank.

    PROBLEM SOLVED:
      A document for "SBI Cashback" sometimes triggers "Millennia" detection
      (because HDFC Millennia is also a cashback-focused card and the word
      "Millennia" occasionally appears in body text comparisons).
      Without this check, the file becomes SBI_Millennia_BR_2026.pdf — wrong.

    HOW IT WORKS:
      1. Look up which cards this bank is known to issue (from BANK_CARDS).
      2. If the current card IS in that bank's list → no action, return as-is.
      3. If the current card is NOT in that list:
           a. Re-scan the text for the bank's own cards (exact match first,
              then header-zone match, then full-text match).
           b. If a bank-specific card is found → return it.
           c. If none found → return None ("UNKNOWN" card, send to review).
      4. If the bank has no entry in BANK_CARDS → return the original card
         (don't break classification for unconfigured banks).

    This function reads the text directly from the caller so it always works
    with the full extracted text — no page-count reduction.
    """
    if not bank or not card:
        return card  # can't narrow without both

    bank_upper = bank.upper()

    # No mapping for this bank → leave card unchanged
    if bank_upper not in BANK_CARDS:
        return card

    allowed_cards = BANK_CARDS[bank_upper]

    # Card is already valid for this bank → no action needed
    if card in allowed_cards:
        return card

    # Card is NOT valid for this bank — try to find the right card
    logger.warning(
        f"[FIX #2] Card '{card}' is not in {bank_upper}'s known card list. "
        f"Re-scanning text for a bank-specific card match."
    )

    header_zone = text_lower[:500]

    # Pass 1: header zone (most reliable)
    for candidate in allowed_cards:
        if candidate.lower() in header_zone:
            logger.info(
                f"[FIX #2] Found bank-specific card '{candidate}' in header "
                f"→ replacing '{card}'"
            )
            return candidate

    # Pass 2: full text
    for candidate in allowed_cards:
        if candidate.lower() in text_lower:
            logger.info(
                f"[FIX #2] Found bank-specific card '{candidate}' in body "
                f"→ replacing '{card}'"
            )
            return candidate

    # No bank-specific card found → mark as unknown
    logger.warning(
        f"[FIX #2] No card from {bank_upper}'s list found in text. "
        f"Setting card to None (will route to needs_review/)."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FIX #5 — OCR FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _extract_with_ocr(pdf_path: Path, max_pages: int) -> str:
    """
    OCR fallback for scanned (image-only) PDFs.

    Triggered ONLY when BOTH pdfplumber AND PyMuPDF return empty text.
    Uses PyMuPDF to render each page to an image, then pytesseract to OCR it.

    REQUIREMENTS:
      pip install pytesseract Pillow
      # Tesseract binary:
      # macOS:   brew install tesseract
      # Ubuntu:  sudo apt install tesseract-ocr
      # Windows: https://github.com/UB-Mannheim/tesseract/wiki

    Returns empty string if pytesseract or fitz is not installed (fails silently
    so the rest of the pipeline is not broken by missing OCR dependencies).
    """
    try:
        import fitz          # PyMuPDF
        import pytesseract
        from PIL import Image
        import io
    except ImportError as e:
        logger.warning(
            f"[OCR] Skipping OCR — missing dependency: {e}. "
            f"Install with: pip install pytesseract Pillow pymupdf"
        )
        return ""

    logger.info(f"[OCR] Running OCR on {pdf_path.name} (up to {max_pages} pages)...")

    text = ""
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            # Render at 300 DPI for good OCR accuracy
            mat  = fitz.Matrix(300 / 72, 300 / 72)
            pix  = page.get_pixmap(matrix=mat)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img, lang="eng")
            if page_text.strip():
                text += page_text + "\n"
        doc.close()

        if text.strip():
            logger.info(
                f"[OCR] Extracted {len(text)} characters from "
                f"{pdf_path.name} via OCR."
            )
        else:
            logger.warning(f"[OCR] OCR produced no text from {pdf_path.name}.")

    except Exception as e:
        logger.error(f"[OCR] OCR failed for {pdf_path.name}: {e}")

    return text


# ─────────────────────────────────────────────────────────────────────────────
# FIX #6 — MULTI-CONDITION LLM TRIGGER
# ─────────────────────────────────────────────────────────────────────────────

def _should_call_llm(
    rule_confidence: float,
    rule_bank:       str | None,
    rule_card:       str | None,
    rule_doc_type:   str,
) -> tuple[bool, str]:
    """
    Decides whether the LLM fallback should be triggered.

    FIX #6: Replaces the old single-condition check (confidence < threshold)
    with a multi-condition check. The LLM is called if ANY condition is true:
      1. rule_confidence < CONFIDENCE_THRESHOLD (0.70)
      2. Bank is None or UNKNOWN
      3. Card is None or UNKNOWN
      4. Doc type is empty or UNKNOWN

    Returns (should_call: bool, reason: str).
    The reason string is used in log messages.
    """
    reasons = []

    if rule_confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            f"confidence={rule_confidence:.2f} < threshold={CONFIDENCE_THRESHOLD}"
        )
    if not rule_bank or rule_bank.upper() == "UNKNOWN":
        reasons.append(f"bank='{rule_bank}' (UNKNOWN)")
    if not rule_card or rule_card.upper() == "UNKNOWN":
        reasons.append(f"card='{rule_card}' (UNKNOWN)")
    if not rule_doc_type or rule_doc_type.upper() == "UNKNOWN":
        reasons.append(f"doc_type='{rule_doc_type}' (UNKNOWN)")

    if reasons:
        return True, " | ".join(reasons)
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID DECISION LOGIC (unchanged core logic, improved logging)
# ─────────────────────────────────────────────────────────────────────────────

def apply_llm_override(
    rule_bank:      str | None,
    rule_card:      str | None,
    rule_doc_type:  str,
    rule_is_master: bool,
    rule_confidence:float,
    llm_result:     dict,
) -> tuple[str | None, str | None, str, bool, float, str]:
    """
    Decides whether to use LLM output or keep rule-based output.

    DECISION:
      LLM overrides rules when BOTH:
        1. LLM classification succeeded (llm_success=True)
        2. LLM confidence > rule confidence + LLM_OVERRIDE_MARGIN (0.10)

      If LLM returns UNKNOWN for bank/card but rules found a value,
      the rule value is preserved for those fields even when LLM wins overall.

    RETURNS:
      (bank, card, doc_type, is_master, final_confidence, source)
      source is "rule_based" or "llm"
    """
    if not llm_result.get("llm_success", False):
        logger.info(
            f"[DECISION] LLM failed → keeping rule result "
            f"(rule_conf={rule_confidence:.2f})"
        )
        return (
            rule_bank, rule_card, rule_doc_type, rule_is_master,
            rule_confidence, "rule_based",
        )

    llm_confidence = llm_result.get("confidence", 0.0)

    # FIX #8: log both confidence values and the required margin
    logger.info(
        f"[DECISION] rule_conf={rule_confidence:.2f} | "
        f"llm_conf={llm_confidence:.2f} | "
        f"required margin={LLM_OVERRIDE_MARGIN} | "
        f"override threshold={rule_confidence + LLM_OVERRIDE_MARGIN:.2f}"
    )

    if llm_confidence > rule_confidence + LLM_OVERRIDE_MARGIN:
        final_bank      = llm_result.get("bank",      "UNKNOWN")
        final_card      = llm_result.get("card_name", "UNKNOWN")
        final_doc_type  = llm_result.get("doc_type",   rule_doc_type)
        final_is_master = llm_result.get("is_master",  rule_is_master)

        # Preserve rule values for fields LLM couldn't identify
        if final_bank == "UNKNOWN" and rule_bank:
            final_bank = rule_bank
            logger.info(
                f"[DECISION] LLM bank=UNKNOWN → keeping rule bank={rule_bank}"
            )
        if final_card == "UNKNOWN" and rule_card:
            final_card = rule_card
            logger.info(
                f"[DECISION] LLM card=UNKNOWN → keeping rule card={rule_card}"
            )

        logger.info(
            f"[DECISION] ✓ LLM OVERRIDE applied "
            f"({llm_confidence:.2f} > {rule_confidence:.2f} + {LLM_OVERRIDE_MARGIN}) "
            f"→ {final_bank} | {final_card} | {final_doc_type}"
        )
        return (
            final_bank, final_card, final_doc_type, final_is_master,
            llm_confidence, "llm",
        )

    else:
        logger.info(
            f"[DECISION] Rules kept — LLM did not exceed margin "
            f"({llm_confidence:.2f} ≤ {rule_confidence:.2f} + {LLM_OVERRIDE_MARGIN})"
        )
        return (
            rule_bank, rule_card, rule_doc_type, rule_is_master,
            rule_confidence, "rule_based",
        )


# ─────────────────────────────────────────────────────────────────────────────
# METADATA GENERATION + SAVING (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def generate_metadata(
    bank:                  str | None,
    card:                  str | None,
    doc_type:              str,
    is_master:             bool,
    confidence:            float,
    classification_source: str,
    source_file:           str,
    llm_reason:            str | None = None,
    dest_path:             Path | None = None,
) -> dict:
    """
    Builds the structured metadata dict saved alongside each processed PDF.

    WHY METADATA IS CRITICAL FOR RAG:
      Each text chunk stored in the vector DB must carry bank/card/doc_type
      metadata so downstream agents can filter by document scope:
        "Only search HDFC Millennia MITC for fee-related questions"
      Without this, every query scans the entire corpus — slow and inaccurate.
    """
    return {
        "bank":                   bank or "UNKNOWN",
        "card":                   card or "UNKNOWN",
        "doc_type":               doc_type,
        "is_master":              is_master,
        "confidence":             round(confidence, 4),
        "classification_source":  classification_source,
        "source_file":            source_file,
        "output_file":            dest_path.name if dest_path else None,
        "llm_reason":             llm_reason,
        "processing_timestamp":   datetime.now().isoformat(timespec="seconds"),
    }


def save_metadata_json(metadata: dict, dest_path: Path, dry_run: bool) -> None:
    """
    Saves metadata dict as JSON alongside the processed PDF.
    e.g. HDFC_Millennia_MITC_2026.pdf → HDFC_Millennia_MITC_2026.json
    In dry-run mode: only logs, no file written.
    """
    json_path = dest_path.with_suffix(".json")
    if dry_run:
        logger.info(f"[DRY-RUN] Would write metadata JSON: {json_path.name}")
        return
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"[OUTPUT] Metadata JSON saved: {json_path}")
    except Exception as e:
        logger.error(f"[OUTPUT] Failed to save metadata JSON: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX #4 — SINGLE-FILE PROCESSOR
# Extracted into its own function so the main loop is a plain `for` loop
# with one call per file. This function never calls itself and never
# re-triggers the outer loop. LLM retries are contained inside
# classify_with_llm() and do NOT restart file processing.
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_file(
    pdf_path:      Path,
    debug:         bool,
    dry_run:       bool,
    use_llm:       bool,
    llm_available: bool,
) -> dict:
    """
    Processes a single PDF file through the full hybrid pipeline.

    Steps (FIX #4 — each step runs exactly once per call, no recursion):
      [STEP 1] Extract text via pdfplumber/PyMuPDF adaptive reader
               If both fail → OCR fallback (FIX #5)
      [STEP 2] Rule-based classification (detect_bank, detect_card,
               detect_doc_type, detect_master_doc, compute_confidence)
               + bank-specific card narrowing (FIX #2)
      [STEP 3] Multi-condition LLM trigger check (FIX #6)
               If triggered → classify_with_llm() called ONCE
               → apply_llm_override() decides winner
      [STEP 4] Write PDF + metadata JSON

    Returns a log entry dict compatible with write_detail_log().
    """
    pages_read         = PAGE_TIERS[0]
    llm_called         = False
    llm_result         = None
    classification_src = "rule_based"
    llm_reason         = None
    status             = "SUCCESS"

    # These are assigned inside the try block and read outside it for logging
    doc_type_result = {"value": "UNKNOWN", "confidence": 0.0, "reasons": []}
    bank_result     = {"value": None,      "confidence": 0.0, "reasons": []}
    card_result     = {"value": None,      "confidence": 0.0, "reasons": []}
    master_result   = {"is_master": False, "signal": None,   "confidence": 0.0}
    rule_confidence = 0.0
    final_confidence= 0.0
    final_bank      = None
    final_card      = None
    final_doc_type  = "UNKNOWN"
    final_is_master = False
    year            = DEFAULT_YEAR
    dest_path       = REVIEW_DIR / "UNKNOWN_UNKNOWN_UNKNOWN.pdf"  # safe default

    try:
        # ── STEP 1: Text extraction ────────────────────────────────────────────
        logger.info("[STEP 1] Extracting text from PDF")

        # ── STEP 2: Rule-based classification ─────────────────────────────────
        logger.info("[STEP 2] Running rule-based classification")

        (text, doc_type_result, bank_result,
         card_result, master_result, year, rule_confidence) = \
            run_detection_with_fallback(pdf_path, debug)

        # ── FIX #5: OCR fallback if no text was extracted ─────────────────────
        # run_detection_with_fallback() uses pdfplumber then PyMuPDF internally.
        # If the returned text is still empty, try OCR as a last resort.
        if not text.strip():
            logger.warning(
                f"[STEP 1] Standard extraction returned empty text. "
                f"Attempting OCR fallback..."
            )
            ocr_text = _extract_with_ocr(pdf_path, max_pages=PAGE_TIERS[-1])
            if ocr_text.strip():
                # Re-run rule classification on OCR text
                # We can't re-call run_detection_with_fallback with arbitrary text,
                # so we import the individual detectors and re-run them directly.
                from preprocess import (
                    detect_doc_type, detect_bank, detect_card,
                    detect_master_doc, detect_year as _detect_year,
                )
                text            = ocr_text
                doc_type_result = detect_doc_type(text)
                bank_result     = detect_bank(text, pdf_path.name)
                master_result   = detect_master_doc(text)
                if master_result["is_master"]:
                    card_result = {
                        "value": "MASTER",
                        "confidence": master_result["confidence"],
                        "reasons": ["OCR-detected master doc"],
                    }
                else:
                    card_result = detect_card(text, pdf_path.name)
                year            = _detect_year(text)
                rule_confidence = compute_confidence(
                    doc_type_result, bank_result, card_result, master_result
                )
                logger.info(
                    f"[STEP 1] OCR re-classification: "
                    f"bank={bank_result['value']} | "
                    f"card={card_result['value']} | "
                    f"doc_type={doc_type_result['value']} | "
                    f"conf={rule_confidence:.2f}"
                )
            else:
                logger.error(
                    f"[STEP 1] OCR also produced no text for {pdf_path.name}. "
                    f"File will be routed to needs_review/."
                )

        # FIX #8: explicit rule confidence in log
        logger.info(
            f"[STEP 2] Rule result: "
            f"bank={bank_result['value']} | "
            f"card={card_result['value']} | "
            f"doc_type={doc_type_result['value']} | "
            f"rule_confidence={rule_confidence:.2f}"
        )

        # Approximate pages_read from which tier the confidence passed
        for tier in PAGE_TIERS:
            pages_read = tier
            if rule_confidence >= CONFIDENCE_THRESHOLD:
                break

        # ── FIX #2: Bank-specific card narrowing ──────────────────────────────
        # Run AFTER rule classification, BEFORE LLM trigger check.
        # This corrects cross-bank card assignments in the rule result.
        raw_card    = card_result["value"]
        narrowed    = _narrow_card_to_bank(
            bank      = bank_result["value"],
            card      = raw_card,
            text_lower= text.lower() if text else "",
        )
        if narrowed != raw_card:
            logger.info(
                f"[FIX #2] Card narrowed: '{raw_card}' → '{narrowed}' "
                f"(bank={bank_result['value']})"
            )
            card_result = dict(card_result)  # don't mutate original
            card_result["value"] = narrowed
            # Re-compute confidence with the corrected card
            rule_confidence = compute_confidence(
                doc_type_result, bank_result, card_result, master_result
            )
            logger.info(
                f"[FIX #2] Confidence after card correction: {rule_confidence:.2f}"
            )

        rule_bank      = bank_result["value"]
        rule_card      = card_result["value"]
        rule_doc_type  = doc_type_result["value"]
        rule_is_master = master_result["is_master"]

        # Initialise final values to rule values (will be overridden if LLM wins)
        final_bank       = rule_bank
        final_card       = rule_card
        final_doc_type   = rule_doc_type
        final_is_master  = rule_is_master
        final_confidence = rule_confidence

        # ── STEP 3: Multi-condition LLM trigger check ─────────────────────────
        should_llm, trigger_reason = _should_call_llm(
            rule_confidence = rule_confidence,
            rule_bank       = rule_bank,
            rule_card       = rule_card,
            rule_doc_type   = rule_doc_type,
        )

        if should_llm and use_llm and llm_available:
            logger.info(
                f"[STEP 3] LLM triggered — reasons: {trigger_reason}"
            )
            llm_called = True
            # classify_with_llm() handles its own retries (MAX_RETRIES=2).
            # It NEVER restarts file processing — only retries the HTTP call.
            llm_result = classify_with_llm(text)

            (final_bank, final_card, final_doc_type, final_is_master,
             final_confidence, classification_src) = apply_llm_override(
                rule_bank       = rule_bank,
                rule_card       = rule_card,
                rule_doc_type   = rule_doc_type,
                rule_is_master  = rule_is_master,
                rule_confidence = rule_confidence,
                llm_result      = llm_result,
            )

            if classification_src == "llm":
                llm_reason = llm_result.get("reason")

        elif should_llm and use_llm and not llm_available:
            logger.info("[STEP 3] LLM would be triggered but Ollama is not available")
        elif should_llm and not use_llm:
            logger.info("[STEP 3] LLM skipped — disabled via --no-llm flag")
        else:
            logger.info(
                f"[STEP 3] LLM skipped — all conditions satisfied "
                f"(conf={rule_confidence:.2f}, bank={rule_bank}, "
                f"card={rule_card}, doc_type={rule_doc_type})"
            )

        # ── STEP 4: Build output path + copy file ─────────────────────────────
        logger.info(
            f"[STEP 4] Final result — "
            f"bank={final_bank} | card={final_card} | "
            f"doc_type={final_doc_type} | master={final_is_master} | "
            f"source={classification_src} | confidence={final_confidence:.2f}"
        )

        needs_review = (
            final_confidence < CONFIDENCE_THRESHOLD
            or final_bank is None
            or final_card is None
        )

        new_filename = generate_filename(final_bank, final_card, final_doc_type, year)

        if needs_review:
            dest_path = REVIEW_DIR / new_filename
            status    = "NEEDS_REVIEW"
        else:
            dest_dir  = get_output_folder(final_bank, final_card, PROCESSED_DIR)
            dest_path = dest_dir / new_filename
            status    = "MASTER_DOC" if final_is_master else "SUCCESS"

        if is_duplicate(dest_path):
            logger.info(f"Duplicate detected — skipping: {new_filename}")
            status = "DUPLICATE_SKIPPED"
        else:
            move_file(pdf_path, dest_path, dry_run)

        # ── Metadata JSON ─────────────────────────────────────────────────────
        if status not in ("DUPLICATE_SKIPPED",):
            metadata = generate_metadata(
                bank                  = final_bank,
                card                  = final_card,
                doc_type              = final_doc_type,
                is_master             = final_is_master,
                confidence            = final_confidence,
                classification_source = classification_src,
                source_file           = pdf_path.name,
                llm_reason            = llm_reason,
                dest_path             = dest_path,
            )
            save_metadata_json(metadata, dest_path, dry_run)
            logger.info(f"[OUTPUT] Saved: {dest_path.name}")

    except Exception as exc:
        logger.error(
            f"Error processing {pdf_path.name}: {exc}", exc_info=True
        )
        status          = "ERROR"
        doc_type_result = {"value": "ERROR",  "confidence": 0.0, "reasons": [str(exc)]}
        bank_result     = {"value": None,     "confidence": 0.0, "reasons": []}
        card_result     = {"value": None,     "confidence": 0.0, "reasons": []}
        master_result   = {"is_master": False, "signal": None,   "confidence": 0.0}
        final_confidence = 0.0
        final_is_master  = False
        classification_src = "rule_based"
        llm_reason       = None
        rule_confidence  = 0.0

    # ── Return log entry compatible with write_detail_log() ───────────────────
    return {
        # Original fields (format expected by write_detail_log + write_summary_csv)
        "filename":             pdf_path.name,
        "bank":                 bank_result["value"],
        "bank_conf":            bank_result["confidence"],
        "bank_reasons":         bank_result["reasons"],
        "card":                 card_result["value"],
        "card_conf":            card_result["confidence"],
        "card_reasons":         card_result["reasons"],
        "doc_type":             doc_type_result["value"],
        "doc_type_conf":        doc_type_result["confidence"],
        "doc_type_reasons":     doc_type_result["reasons"],
        "is_master":            master_result["is_master"],
        "master_signal":        master_result["signal"],
        "master_conf":          master_result["confidence"],
        "overall_conf":         rule_confidence,
        "pages_read":           pages_read,
        "status":               status,
        # Extended hybrid fields
        "llm_called":           llm_called,
        "llm_success":          llm_result.get("llm_success", False) if llm_result else False,
        "llm_confidence":       llm_result.get("confidence",  0.0)   if llm_result else 0.0,
        "llm_bank":             llm_result.get("bank")               if llm_result else None,
        "llm_card":             llm_result.get("card_name")          if llm_result else None,
        "llm_doc_type":         llm_result.get("doc_type")           if llm_result else None,
        "llm_reason":           llm_reason,
        "classification_source":classification_src,
        "final_confidence":     final_confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HYBRID PIPELINE
# FIX #4: Main loop is a plain `for` with one _process_one_file() call per PDF.
# ─────────────────────────────────────────────────────────────────────────────

def process_all_hybrid(
    dry_run: bool = False,
    debug:   bool = False,
    use_llm: bool = True,
) -> None:
    """
    Full hybrid preprocessing pipeline.

    Processing flow per file (FIX #4 — no recursion, no loop restart):
      for pdf_path in pdf_files:         ← single outer loop, one iteration per file
          entry = _process_one_file(...)  ← one call, never calls itself
          log_entries.append(entry)

    After the loop:
      write logs → coverage validation → print summary
    """
    for d in [RAW_DIR, PROCESSED_DIR, REVIEW_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    total     = len(pdf_files)

    if total == 0:
        logger.warning(f"No PDF files found in {RAW_DIR}. Exiting.")
        return

    logger.info(f"Found {total} PDF file(s) in {RAW_DIR}")

    # ── Preflight: LLM availability check ─────────────────────────────────────
    llm_available = False
    if use_llm:
        logger.info("[STEP 0] Checking LLM (Ollama/Mistral) availability...")
        llm_available = check_ollama_available()
        if not llm_available:
            logger.warning(
                "[LLM] Ollama not available — LLM fallback will be SKIPPED. "
                "Low-confidence documents will go to needs_review/ as usual. "
                "To enable: run 'ollama serve' and 'ollama pull mistral'"
            )
    else:
        logger.info("[LLM] LLM fallback disabled via --no-llm flag.")

    # ── Counters ───────────────────────────────────────────────────────────────
    count_processed = count_review = count_errors = count_skipped = 0
    count_llm_used  = 0
    count_llm_won   = 0
    log_entries: list[dict] = []

    # ── Main loop (FIX #4: plain for-loop, one call per file) ─────────────────
    for idx, pdf_path in enumerate(pdf_files, start=1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"[PROCESSING] {idx}/{total}: {pdf_path.name}")
        logger.info("=" * 60)

        entry = _process_one_file(
            pdf_path      = pdf_path,
            debug         = debug,
            dry_run       = dry_run,
            use_llm       = use_llm,
            llm_available = llm_available,
        )

        log_entries.append(entry)

        # Update counters from entry status
        s = entry["status"]
        if s == "ERROR":
            count_errors   += 1
        elif s == "DUPLICATE_SKIPPED":
            count_skipped  += 1
        elif s == "NEEDS_REVIEW":
            count_review   += 1
        else:
            count_processed += 1

        if entry["llm_called"]:
            count_llm_used += 1
        if entry.get("classification_source") == "llm":
            count_llm_won += 1

    # ── Write logs ─────────────────────────────────────────────────────────────
    write_detail_log(log_entries)
    write_summary_csv(log_entries)
    _write_hybrid_log(log_entries)

    # ── Coverage validation ────────────────────────────────────────────────────
    logger.info("")
    logger.info("Running document coverage validation...")
    coverage_map       = build_coverage_map(log_entries)
    validation_results = validate_coverage(coverage_map)
    write_missing_docs_csv(validation_results)
    write_coverage_dashboard(validation_results)
    print_validation_summary(validation_results)

    # ── Final summary ──────────────────────────────────────────────────────────
    master_count = sum(
        1 for e in log_entries
        if e.get("is_master") and e["status"] in ("MASTER_DOC", "SUCCESS")
    )
    logger.info("")
    logger.info("=" * 60)
    logger.info("HYBRID PIPELINE COMPLETE")
    logger.info(f"  Total files         : {total}")
    logger.info(f"  Processed           : {count_processed}")
    logger.info(f"  Master/Collective   : {master_count}")
    logger.info(f"  Needs review        : {count_review}")
    logger.info(f"  Errors              : {count_errors}")
    logger.info(f"  Duplicates skipped  : {count_skipped}")
    logger.info(f"  LLM triggered       : {count_llm_used} (of {total} files)")
    logger.info(f"  LLM overrides used  : {count_llm_won}")
    logger.info(f"  Dry-run mode        : {'ON' if dry_run else 'OFF'}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED HYBRID LOG
# Writes a supplementary log showing only LLM-triggered files
# with full before/after comparison. Separate from preprocess_log.txt.
# ─────────────────────────────────────────────────────────────────────────────

def _write_hybrid_log(entries: list[dict]) -> None:
    """
    Writes hybrid_classification_log.txt — files where LLM was called,
    showing rule result, LLM result, and the final decision.
    """
    hybrid_log_path = LOG_DIR / "hybrid_classification_log.txt"
    llm_entries     = [e for e in entries if e.get("llm_called")]

    with open(hybrid_log_path, "w", encoding="utf-8") as f:
        f.write(
            f"HYBRID CLASSIFICATION LOG — "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("=" * 70 + "\n")
        f.write(f"Files where LLM was invoked: {len(llm_entries)}\n\n")

        if not llm_entries:
            f.write(
                "No files required LLM fallback — "
                "all rule-based results met all trigger conditions.\n"
            )
            return

        for entry in llm_entries:
            f.write("=" * 40 + "\n")
            f.write(f"FILE: {entry['filename']}\n\n")

            f.write("  RULE RESULT:\n")
            f.write(f"    Bank       : {entry['bank']}\n")
            f.write(f"    Card       : {entry['card']}\n")
            f.write(f"    DocType    : {entry['doc_type']}\n")
            f.write(f"    Confidence : {entry['overall_conf']:.2f}\n\n")

            f.write("  LLM RESULT (Mistral):\n")
            f.write(f"    Bank       : {entry.get('llm_bank', 'N/A')}\n")
            f.write(f"    Card       : {entry.get('llm_card', 'N/A')}\n")
            f.write(f"    DocType    : {entry.get('llm_doc_type', 'N/A')}\n")
            f.write(f"    Confidence : {entry.get('llm_confidence', 0):.2f}\n")
            f.write(f"    Reason     : {entry.get('llm_reason', 'N/A')}\n\n")

            src = entry.get("classification_source", "rule_based")
            f.write(f"  DECISION: Used '{src.upper()}' result\n")
            f.write(f"  Final confidence: {entry.get('final_confidence', 0):.2f}\n")
            f.write(f"  Status: {entry['status']}\n\n")

    logger.info(f"Hybrid log written to: {hybrid_log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid (Rule + LLM) Credit Card PDF Preprocessing Pipeline. "
            "Extends preprocess.py with Mistral fallback via Ollama."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate processing without copying files or writing metadata.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print extracted text samples for each PDF.",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM fallback — run rule-based classification only.",
    )
    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 60)
    logger.info("HYBRID PREPROCESSING PIPELINE")
    logger.info(
        f"  LLM fallback : "
        f"{'DISABLED (--no-llm)' if args.no_llm else 'ENABLED (Mistral via Ollama)'}"
    )
    logger.info(f"  Dry-run      : {'ON' if args.dry_run else 'OFF'}")
    logger.info(f"  Debug        : {'ON' if args.debug else 'OFF'}")
    logger.info("=" * 60)

    process_all_hybrid(
        dry_run = args.dry_run,
        debug   = args.debug,
        use_llm = not args.no_llm,
    )


if __name__ == "__main__":
    main()