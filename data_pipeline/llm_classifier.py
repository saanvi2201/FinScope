# =============================================================================
# llm_classifier.py — LLM-Based Document Classification using Mistral (Ollama)
# =============================================================================
#
# ── CHANGES IN THIS VERSION ──────────────────────────────────────────────────
#
#  FIX #1 — doc_type priority logic (CRITICAL)
#    _apply_doc_type_priority() re-evaluates the LLM's self-reported doc_type
#    by scanning the actual document text in strict MITC > TNC > BR > LG order.
#    The LLM's value is only used as a fallback if no priority signals match.
#    This eliminates the most common confusion: MITC ↔ TNC, MITC ↔ BR.
#
#  FIX #3 — LLM output quality (CRITICAL)
#    a) STRICT BANK WHITELIST — only canonical short codes are accepted.
#       Long-form values ("State Bank of India", "HDFC Bank Ltd") and
#       hallucinated values ("SBIC", "HDFC Bank Credit") are normalised through
#       BANK_NORMALIZATION_MAP, then enforced against ALLOWED_BANKS.
#       Anything not in ALLOWED_BANKS → "UNKNOWN".
#    b) CARD NAME SANITATION — names longer than 5 words are rejected.
#       Sentence-like values containing indicator words ("is", "provides",
#       "document") are rejected. Both → "UNKNOWN".
#    c) PROMPT REWRITTEN — bank field shows the exact allowed list.
#       card_name rule says "1-3 words, short product name only" with
#       a wrong/right example. doc_type shows numbered priority rules.
#       "Return UNKNOWN if unsure" added explicitly.
#
#  FIX #7 — performance
#    REQUEST_TIMEOUT reduced from 180 s → 60 s.
#    Model confirmed as "mistral" (not llama3).
#    LLM_TEXT_WINDOW kept at 2000 chars (document text NOT reduced).
#    MAX_RETRIES kept at 2.
#
#  FIX #8 — logging
#    All logs use [STEP 3] / [LLM OUTPUT] / [LLM REASON] prefixes.
#    Confidence values logged explicitly at every decision point.
#
#  UNCHANGED:
#    4-strategy JSON parsing (_parse_llm_response) — kept intact.
#    check_ollama_available() — kept intact.
#    Retry loop structure — kept intact.
#    temperature=0 / num_predict=300 — kept intact.
#
# =============================================================================
#
# ── HOW TO SET UP OLLAMA + MISTRAL (one-time) ────────────────────────────────
#
#  Step 1 — Install Ollama
#    macOS / Linux:
#      curl -fsSL https://ollama.ai/install.sh | sh
#    Windows:
#      Download installer from https://ollama.ai/download
#      Run the .exe and follow the wizard.
#
#  Step 2 — Pull the Mistral model (~4.1 GB, one-time download)
#    ollama pull mistral
#
#  Step 3 — Start the Ollama server
#    ollama serve
#    (On most systems this starts automatically after install.
#     Keep this terminal open while running the pipeline.)
#
#  Step 4 — Verify the model is working
#    ollama run mistral "say hello"
#    You should see a short text response.
#
#  Step 5 — Run standalone test
#    python data_pipeline/llm_classifier.py
#
# =============================================================================

import json
import logging
import re
import time
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "mistral"

# Full document text window — NOT reduced.
# 2000 chars covers the title + first few sections, sufficient for classification.
LLM_TEXT_WINDOW = 2000

MAX_RETRIES     = 2       # kept at 2 per requirements
RETRY_DELAY_SEC = 1.0
REQUEST_TIMEOUT = 60      # FIX #7: reduced from 180 s → 60 s

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL BANK WHITELIST
# FIX #3a: Only these values are accepted from the LLM.
# Anything else (after normalisation) → UNKNOWN.
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_BANKS = {
    "HDFC", "SBI", "ICICI", "AXIS", "AMEX", "HSBC",
    "IDFC", "AU", "SCB", "KOTAK", "YES", "BOB",
    "RBL", "INDUSIND", "CITI", "ONECARD", "SCAPIA",
    "JUPITER", "UNKNOWN",
}

# Common LLM mis-spellings and long-form values → canonical short code.
# Applied BEFORE the whitelist check so recoverable values are rescued.
BANK_NORMALIZATION_MAP = {
    "STATE BANK OF INDIA":        "SBI",
    "SBIC":                       "SBI",   # very common LLM hallucination
    "SBI CARD":                   "SBI",
    "SBICARD":                    "SBI",
    "HDFC BANK":                  "HDFC",
    "HDFC BANK LTD":              "HDFC",
    "HDFC BANK LIMITED":          "HDFC",
    "ICICI BANK":                 "ICICI",
    "ICICI BANK LIMITED":         "ICICI",
    "AXIS BANK":                  "AXIS",
    "AXIS BANK LIMITED":          "AXIS",
    "AMERICAN EXPRESS":           "AMEX",
    "AMERICAN EXPRESS BANKING":   "AMEX",
    "STANDARD CHARTERED":         "SCB",
    "STANDARD CHARTERED BANK":    "SCB",
    "STANCHART":                  "SCB",
    "HSBC BANK":                  "HSBC",
    "KOTAK MAHINDRA":             "KOTAK",
    "KOTAK MAHINDRA BANK":        "KOTAK",
    "YES BANK":                   "YES",
    "IDFC FIRST":                 "IDFC",
    "IDFC FIRST BANK":            "IDFC",
    "AU SMALL FINANCE":           "AU",
    "AU SMALL FINANCE BANK":      "AU",
    "BANK OF BARODA":             "BOB",
    "RBL BANK":                   "RBL",
    "RATNAKAR BANK":              "RBL",
    "INDUSIND BANK":              "INDUSIND",
    "CITIBANK":                   "CITI",
    "ONE CARD":                   "ONECARD",
    "FTC FINTECH":                "ONECARD",
}


# ─────────────────────────────────────────────────────────────────────────────
# DOC TYPE PRIORITY SIGNALS
# FIX #1: MITC > TNC > BR > LG priority order.
# Each list contains the canonical keywords for that type.
# In _apply_doc_type_priority(), we check each list in order and return on
# the FIRST match — guaranteeing no overlap ambiguity.
# ─────────────────────────────────────────────────────────────────────────────

MITC_PRIORITY_SIGNALS = [
    "most important terms and conditions",  # most specific first
    "most important terms & conditions",
    "most important terms",
    "key fact statement cum most important",
    "key fact statement",
    "kfs",
    "mitc",
    "schedule of charges",
    "annual fee",
    "interest rate",
    "finance charge",
    "late payment fee",
    "cash advance fee",
]

TNC_PRIORITY_SIGNALS = [
    "cardmember agreement",
    "cardholder agreement",
    "client terms",
    "credit card terms",
    "terms of use",
    "terms and conditions",
    "terms & conditions",
]

BR_PRIORITY_SIGNALS = [
    "cashback proposition",
    "cashpoints proposition",
    "reward points",
    "welcome benefit",
    "welcome bonus",
    "welcome gift",
    "earn rate",
    "milestone benefit",
    "accelerated rewards",
    "reward redemption",
    "cashback",
    "reward",
    "benefit",
]

LG_PRIORITY_SIGNALS = [
    "airport lounge access program",
    "domestic airport lounge",
    "lounge access guide",
    "lounge access",
    "airport lounge",
    "priority pass",
    "domestic lounge",
    "international lounge",
]

# Words that indicate the LLM returned a sentence as card_name.
# Card names are short (1-3 words). Sentences contain these filler words.
_SENTENCE_INDICATORS = {
    "is", "are", "was", "provides", "document", "contains", "includes",
    "describes", "applicable", "pertaining", "related", "regarding",
    "stating", "outlining", "covering",
}

# Module-level store for the last text window passed to classify_with_llm.
# Allows _apply_doc_type_priority() to access the text without changing
# the entire call chain's function signatures.
_LAST_LLM_TEXT: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT ENGINEERING
# FIX #3c: Completely rewritten with strict bank whitelist, short card name
#          rule with example, numbered priority rules for doc_type, and
#          explicit "return UNKNOWN if unsure" instructions.
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(text: str) -> str:
    """
    Constructs the Mistral classification prompt.

    Key design decisions:
      1. Bank field shows exact allowed list — prevents free-text hallucination
      2. card_name rule: "1-3 words only, NOT a sentence" with wrong/right example
      3. doc_type: numbered priority order (MITC first) so model learns hierarchy
      4. Explicit "UNKNOWN if unsure" — prevents confident wrong answers
      5. No markdown in output — prevents code fences breaking JSON parsing
    """
    truncated_text = text[:LLM_TEXT_WINDOW]

    prompt = f"""You are an expert classifier for Indian bank credit card documents.
Read the document below carefully and extract metadata. Follow ALL rules strictly.

DOCUMENT TEXT:
{truncated_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — READ CAREFULLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — bank:
  Must be EXACTLY one value from this list (use the SHORT CODE, uppercase):
  HDFC | SBI | ICICI | AXIS | AMEX | HSBC | IDFC | AU | SCB | KOTAK | YES | BOB | RBL | INDUSIND | CITI | ONECARD | SCAPIA | JUPITER
  Do NOT return long names. Examples: "HDFC" not "HDFC Bank Ltd", "SBI" not "State Bank of India".
  If you are not sure → return "UNKNOWN"

RULE 2 — card_name:
  Must be a SHORT product name: 1 to 3 words. It is a credit card product name, not a sentence.
  WRONG: "HDFC Bank Millennia Credit Card Cashback Proposition Document"
  RIGHT: "Millennia"
  More examples: "Cashback", "ACE", "Amazon Pay", "Infinia", "Smart", "Regalia Gold"
  If you cannot identify a specific card name → return "UNKNOWN"

RULE 3 — doc_type:
  Use STRICT PRIORITY ORDER. Check from top to bottom. Use the FIRST type that matches:
  1. MITC  → document contains: fees, interest rate, annual fee, schedule of charges, "most important terms", KFS
  2. TNC   → document contains: terms and conditions, agreement, governing law, exclusions, cardmember agreement
  3. BR    → document contains: reward points, cashback, welcome benefit, milestone, earn rate, redemption
  4. LG    → document contains: lounge access, airport lounge, priority pass, domestic lounge

RULE 4 — is_master:
  true  → document applies to ALL cards of this bank (no specific card named)
  false → document is for one specific card

RULE 5 — confidence:
  A float 0.0 to 1.0. Be honest. Do not inflate.

RULE 6 — reason:
  One short sentence. Do not repeat the JSON fields in it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object below. No markdown. No code fences. No text before or after.
{{
  "bank": "HDFC",
  "card_name": "Millennia",
  "doc_type": "MITC",
  "is_master": false,
  "confidence": 0.88,
  "reason": "Header states MITC with fee schedule for HDFC Millennia card."
}}"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — 4-strategy fallback (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(raw_response: str) -> Optional[dict]:
    """
    Parses raw LLM text into a dict using 4 strategies in order:
      1. Direct JSON parse
      2. Strip markdown fences, then parse
      3. Regex-extract first {...} block, then parse
      4. Substring from first { to last }, then parse
    Returns None if all strategies fail.
    """
    if not raw_response or not raw_response.strip():
        return None

    raw = raw_response.strip()

    # Strategy 1: direct parse (cleanest output path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first {...} block via regex
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: substring from first { to last }
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.debug(f"[LLM PARSE] All 4 strategies failed. Raw: {raw[:300]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DOC TYPE PRIORITY EVALUATION
# FIX #1: Scans text in MITC > TNC > BR > LG order, returns on first match.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_doc_type_priority(llm_doc_type: str, text_lower: str) -> str:
    """
    Re-evaluates the LLM's self-reported doc_type against the actual document
    text using strict priority: MITC > TNC > BR > LG.

    WHY THIS IS NECESSARY:
      MITC documents commonly contain phrases like "terms and conditions" in
      their body text, which cause the LLM to output doc_type=TNC. By
      independently checking priority signals, we override wrong LLM values.

    Returns the canonical doc type string. Falls back to _map_doc_type_aliases()
    if no signal matches in the text.
    """
    if not text_lower:
        return _map_doc_type_aliases(llm_doc_type)

    # Check MITC first — it must win whenever its signals are present
    for signal in MITC_PRIORITY_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "MITC":
                logger.info(
                    f"[LLM VALIDATE] Priority override: MITC signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with MITC"
                )
            return "MITC"

    # Check TNC second
    for signal in TNC_PRIORITY_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "TNC":
                logger.info(
                    f"[LLM VALIDATE] Priority override: TNC signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with TNC"
                )
            return "TNC"

    # Check BR third
    for signal in BR_PRIORITY_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "BR":
                logger.info(
                    f"[LLM VALIDATE] Priority override: BR signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with BR"
                )
            return "BR"

    # Check LG last
    for signal in LG_PRIORITY_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "LG":
                logger.info(
                    f"[LLM VALIDATE] Priority override: LG signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with LG"
                )
            return "LG"

    # No priority signal matched — fall back to LLM's value
    mapped = _map_doc_type_aliases(llm_doc_type)
    logger.debug(
        f"[LLM VALIDATE] No priority signal in text; "
        f"using LLM doc_type '{llm_doc_type}' → mapped to '{mapped}'"
    )
    return mapped


def _map_doc_type_aliases(raw: str) -> str:
    """
    Maps common LLM doc_type aliases to canonical codes.
    Used as a last-resort fallback when priority signals produce no match.
    """
    if raw in {"MITC", "TNC", "BR", "LG"}:
        return raw
    alias_map = {
        "MOST IMPORTANT TERMS AND CONDITIONS":  "MITC",
        "MOST IMPORTANT TERMS & CONDITIONS":    "MITC",
        "MOST IMPORTANT TERMS":                 "MITC",
        "KEY FACT STATEMENT":                   "MITC",
        "KFS":                                  "MITC",
        "TERMS AND CONDITIONS":                 "TNC",
        "TERMS & CONDITIONS":                   "TNC",
        "CARDMEMBER AGREEMENT":                 "TNC",
        "CARDHOLDER AGREEMENT":                 "TNC",
        "AGREEMENT":                            "TNC",
        "BENEFITS AND REWARDS":                 "BR",
        "BENEFITS & REWARDS":                   "BR",
        "REWARDS":                              "BR",
        "CASHBACK":                             "BR",
        "BENEFITS":                             "BR",
        "LOUNGE":                               "LG",
        "LOUNGE GUIDE":                         "LG",
        "AIRPORT LOUNGE":                       "LG",
    }
    return alias_map.get(raw, "MITC")   # MITC is the safest default


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE VALIDATION
# FIX #3a (bank whitelist) + FIX #3b (card name sanitation) + FIX #1 (priority)
# ─────────────────────────────────────────────────────────────────────────────

VALID_DOC_TYPES = {"MITC", "TNC", "BR", "LG"}


def _validate_llm_output(parsed: dict) -> dict:
    """
    Validates and normalises all fields in the parsed LLM JSON.

    Changes from the previous version:
      1. Bank: normalise through BANK_NORMALIZATION_MAP → enforce ALLOWED_BANKS
      2. Card name: reject if > 5 words or contains sentence-indicator words
      3. Doc type: apply priority evaluation using the stored _LAST_LLM_TEXT
    """
    raw_bank     = str(parsed.get("bank",      "UNKNOWN")).strip().upper()
    raw_card     = str(parsed.get("card_name", "UNKNOWN")).strip()
    raw_doc_type = str(parsed.get("doc_type",  "")).strip().upper()
    is_master    = parsed.get("is_master", False)
    confidence   = parsed.get("confidence", 0.5)
    reason       = str(parsed.get("reason", "LLM classification")).strip()

    # ── 1. Bank: normalise → whitelist ────────────────────────────────────────
    # Step A: try the alias map (rescues long-form names and mis-spellings)
    normalised_bank = BANK_NORMALIZATION_MAP.get(raw_bank, raw_bank)
    # Step B: strip stray non-alphanumeric characters
    normalised_bank = re.sub(r"[^\w]", "", normalised_bank).strip()
    # Step C: enforce whitelist
    if normalised_bank not in ALLOWED_BANKS:
        logger.warning(
            f"[LLM VALIDATE] Bank '{raw_bank}' (normalised: '{normalised_bank}') "
            f"not in whitelist → UNKNOWN"
        )
        normalised_bank = "UNKNOWN"

    # ── 2. Card name: sanitation ──────────────────────────────────────────────
    card_words = raw_card.split()
    if len(card_words) > 5:
        # Too many words — almost certainly a description, not a product name
        logger.warning(
            f"[LLM VALIDATE] card_name '{raw_card[:60]}' exceeds 5 words "
            f"({len(card_words)} words) → UNKNOWN"
        )
        cleaned_card = "UNKNOWN"
    elif any(w.lower() in _SENTENCE_INDICATORS for w in card_words[1:]):
        # Contains filler words after the first word — it's a sentence
        logger.warning(
            f"[LLM VALIDATE] card_name '{raw_card[:60]}' looks like a sentence "
            f"→ UNKNOWN"
        )
        cleaned_card = "UNKNOWN"
    else:
        # Remove non-alphanumeric chars (keep +, -, spaces)
        cleaned_card = re.sub(r"[^\w\s\+\-]", "", raw_card).strip()
        if not cleaned_card:
            cleaned_card = "UNKNOWN"

    # ── 3. Doc type: apply priority evaluation ────────────────────────────────
    # Uses _LAST_LLM_TEXT set by classify_with_llm() before calling here.
    final_doc_type = _apply_doc_type_priority(raw_doc_type, _LAST_LLM_TEXT)

    # ── 4. Confidence: coerce to float in [0, 1] ─────────────────────────────
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5
        logger.debug("[LLM VALIDATE] Could not parse confidence; defaulting to 0.5")

    # ── 5. is_master: coerce to bool ─────────────────────────────────────────
    if isinstance(is_master, str):
        is_master = is_master.lower() in ("true", "yes", "1")
    else:
        is_master = bool(is_master)

    return {
        "bank":       normalised_bank,
        "card_name":  cleaned_card,
        "doc_type":   final_doc_type,
        "is_master":  is_master,
        "confidence": round(confidence, 3),
        "reason":     reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA AVAILABILITY CHECK (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_available() -> bool:
    """
    Preflight check: is Ollama running and is the model available?
    Returns True/False. Safe to call — never raises.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            logger.warning(
                f"[OLLAMA CHECK] Server responded {response.status_code}"
            )
            return False

        data             = response.json()
        available_models = [m.get("name", "") for m in data.get("models", [])]
        model_available  = any(OLLAMA_MODEL in m for m in available_models)

        if not model_available:
            logger.warning(
                f"[OLLAMA CHECK] '{OLLAMA_MODEL}' not found. "
                f"Available: {available_models}. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
            return False

        logger.info(
            f"[OLLAMA CHECK] ✓ Ollama running, '{OLLAMA_MODEL}' available."
        )
        return True

    except requests.exceptions.ConnectionError:
        logger.warning(
            f"[OLLAMA CHECK] Cannot connect to {OLLAMA_BASE_URL}. "
            f"Run: ollama serve"
        )
        return False
    except Exception as e:
        logger.warning(f"[OLLAMA CHECK] Unexpected error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFICATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def classify_with_llm(text: str) -> dict:
    """
    Classifies a credit card PDF document using Mistral via Ollama.

    Called by preprocess_with_llm.py when the LLM trigger condition fires.

    ARGS:
      text : str — Full extracted PDF text (truncated internally to 2000 chars)

    RETURNS:
      {bank, card_name, doc_type, is_master, confidence, reason, llm_success}
      llm_success=False on any error — the pipeline keeps the rule result.
    """
    global _LAST_LLM_TEXT  # used by _apply_doc_type_priority via _validate_llm_output

    logger.info("[STEP 3] LLM classification — calling Mistral via Ollama")

    failure_result = {
        "bank":        "UNKNOWN",
        "card_name":   "UNKNOWN",
        "doc_type":    "MITC",
        "is_master":   False,
        "confidence":  0.0,
        "reason":      "LLM classification failed",
        "llm_success": False,
    }

    if not text or not text.strip():
        logger.warning("[LLM] Empty text passed — skipping LLM call.")
        failure_result["reason"] = "Empty text — cannot classify"
        return failure_result

    # Store the lowercased text window so _apply_doc_type_priority can use it
    _LAST_LLM_TEXT = text[:LLM_TEXT_WINDOW].lower()

    prompt = _build_prompt(text)

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(
            f"[LLM] Attempt {attempt}/{MAX_RETRIES} — "
            f"model={OLLAMA_MODEL}, timeout={REQUEST_TIMEOUT}s"
        )

        try:
            response = requests.post(
                url     = f"{OLLAMA_BASE_URL}/api/generate",
                headers = {"Content-Type": "application/json"},
                json    = {
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # temperature=0: deterministic output — essential for
                        # classification tasks where we want the same result
                        # every time on the same input.
                        "temperature": 0,
                        # num_predict=300: a valid JSON response is ~100-200
                        # tokens; cap prevents runaway generation.
                        "num_predict": 300,
                    },
                },
                timeout = REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                logger.warning(
                    f"[LLM] HTTP {response.status_code} from Ollama. "
                    f"Response: {response.text[:200]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                return failure_result

            response_data = response.json()
            raw_text      = response_data.get("response", "")

            if not raw_text:
                logger.warning(f"[LLM] Empty response on attempt {attempt}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                return failure_result

            logger.debug(f"[LLM RAW] {raw_text[:500]}")

            parsed = _parse_llm_response(raw_text)
            if parsed is None:
                logger.warning(
                    f"[LLM] JSON parsing failed on attempt {attempt}. "
                    f"Raw: {raw_text[:200]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                return failure_result

            validated              = _validate_llm_output(parsed)
            validated["llm_success"] = True

            # FIX #8: explicit confidence value in structured log
            logger.info(
                f"[LLM OUTPUT] bank={validated['bank']} | "
                f"card={validated['card_name']} | "
                f"doc_type={validated['doc_type']} | "
                f"is_master={validated['is_master']} | "
                f"confidence={validated['confidence']:.2f}"
            )
            logger.info(f"[LLM REASON] {validated['reason']}")

            return validated

        except requests.exceptions.Timeout:
            logger.warning(
                f"[LLM] Timed out after {REQUEST_TIMEOUT}s "
                f"(attempt {attempt}/{MAX_RETRIES}). "
                f"Try GPU inference or increase REQUEST_TIMEOUT."
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except requests.exceptions.ConnectionError:
            logger.warning(
                f"[LLM] Cannot connect to {OLLAMA_BASE_URL} "
                f"(attempt {attempt}/{MAX_RETRIES}). Run: ollama serve"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except Exception as e:
            logger.error(f"[LLM] Unexpected error on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    logger.error(
        f"[LLM] Classification failed after {MAX_RETRIES} attempts. "
        f"Rule-based result will be used."
    )
    return failure_result


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# Run:  python data_pipeline/llm_classifier.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print(f"LLM Classifier Standalone Test — model: {OLLAMA_MODEL}")
    print("=" * 60)

    print("\n[TEST 0] Checking Ollama availability...")
    if not check_ollama_available():
        print(f"\n[FAIL] Ollama not available.")
        print(f"  1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
        print(f"  2. Pull:    ollama pull {OLLAMA_MODEL}")
        print(f"  3. Serve:   ollama serve")
        sys.exit(1)

    results = []

    # Test 1: MITC — must not be confused with TNC
    t1 = classify_with_llm("""
    HDFC Bank Millennia Credit Card
    Most Important Terms and Conditions (MITC)
    Annual Fee: Rs. 1,000 + GST
    Interest Rate: 3.6% per month (43.2% per annum)
    Minimum Amount Due: 5% of outstanding or Rs. 200
    Cash Advance Fee: 2.5% (minimum Rs. 500)
    Late Payment Fee: Rs. 100 to Rs. 750
    """)
    print(f"\n[TEST 1] HDFC Millennia MITC")
    print(json.dumps(t1, indent=2))
    p1 = t1["doc_type"] == "MITC" and t1["bank"] == "HDFC"
    print(f"  {'PASS ✓' if p1 else 'FAIL ✗'} — expected doc_type=MITC, bank=HDFC")
    results.append(p1)

    # Test 2: BR — must not be confused with MITC
    t2 = classify_with_llm("""
    SBI Cashback Credit Card
    Features and Benefits
    Welcome Benefit: Rs. 500 cashback on first transaction
    Earn Rate: 5% cashback on all online spends
    Milestone Benefit: Rs. 2,000 cashback on Rs. 2 lakh annual spend
    Reward Redemption: Cashback credited within 2 billing cycles
    """)
    print(f"\n[TEST 2] SBI Cashback BR")
    print(json.dumps(t2, indent=2))
    p2 = t2["doc_type"] == "BR" and t2["bank"] == "SBI"
    print(f"  {'PASS ✓' if p2 else 'FAIL ✗'} — expected doc_type=BR, bank=SBI")
    results.append(p2)

    # Test 3: Hallucinated bank guard (SBIC → should become SBI or UNKNOWN, never SBIC)
    t3 = classify_with_llm("""
    SBIC Credit Card
    Most Important Terms and Conditions
    Annual fee: Rs. 500. Interest rate: 3.5% per month.
    """)
    print(f"\n[TEST 3] Hallucinated bank 'SBIC' whitelist guard")
    print(json.dumps(t3, indent=2))
    p3 = t3["bank"] != "SBIC"
    print(f"  {'PASS ✓' if p3 else 'FAIL ✗'} — 'SBIC' must not reach output")
    results.append(p3)

    # Test 4: Long card name sanitation
    t4 = classify_with_llm("""
    AXIS Bank Credit Card
    ACE Credit Card Cashback Programme
    Benefits and Rewards
    Earn 2% cashback on all spends via ACE credit card
    """)
    print(f"\n[TEST 4] AXIS ACE — card name must not be a sentence")
    print(json.dumps(t4, indent=2))
    p4 = len(t4["card_name"].split()) <= 5
    print(f"  {'PASS ✓' if p4 else 'FAIL ✗'} — card_name must be ≤ 5 words")
    results.append(p4)

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*60}")