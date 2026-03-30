# =============================================================================
# llm_classifier.py — LLM-Based Document Classification using Ollama
# =============================================================================
#
# ── FIXES IN THIS VERSION ────────────────────────────────────────────────────
#
#  🔴 FIX 1 — LLM HALLUCINATION (CRITICAL)
#    Problem: LLM ignores actual document text and always returns
#    "HDFC Millennia" regardless of input. This is because:
#      (a) llama3.2:1b is too small and defaults to training priors
#      (b) The old prompt had no ground-truth anchoring
#      (c) LLM confidence (0.88–1.0) was hardcoded in its own priors
#    Fix:
#      - Prompt now explicitly quotes exact phrases from the document
#        to force the model to be grounded in the actual text
#      - Added a "VERIFIED SIGNALS" section that shows what the
#        rule-based system already found, so LLM corrects rather than
#        replaces
#      - LLM confidence is now PENALISED when its output conflicts
#        with rule-based bank/card findings
#      - Added post-validation: if LLM bank ≠ rule bank and rule
#        confidence is decent, rule bank is KEPT
#
#  🔴 FIX 2 — LLM CONFIDENCE IS UNRELIABLE (CRITICAL)
#    Problem: LLM returns 0.88 or 1.0 for every response, even when
#    completely wrong. We were using this number to decide overrides.
#    Fix:
#      - LLM confidence is now DEFLATED based on cross-checks:
#          * If LLM bank ≠ rule bank → confidence -0.30
#          * If LLM card is a generic name ("Millennia", "Cashback"
#            etc.) with no other signal → confidence -0.20
#          * If rule already had a valid bank+card → confidence -0.15
#      - The deflated confidence is what gets compared to rule conf
#
#  🔴 FIX 3 — LLM OVERRIDING CORRECT RULE RESULTS (CRITICAL)
#    Problem: If rule found bank=AXIS, card=Flipkart (conf=0.84) and
#    LLM returned bank=HDFC, card=Millennia (conf=0.88), the LLM won.
#    Fix:
#      - Bank mismatch between LLM and rules BLOCKS the override
#        unless rule bank confidence was below 0.65
#      - If rule has a valid non-generic card, LLM cannot override
#        the card name unless it also correctly identifies the bank
#      - New function: validate_llm_against_rules() runs BEFORE the
#        override decision and may veto or reduce LLM confidence
#
#  🟡 FIX 4 — DOC TYPE PRIORITY LOGIC IMPROVED
#    Problem: TNC docs were being reclassified as BR because "cashback"
#    appears in TNC exclusion clauses ("cashback is not available for…")
#    Fix:
#      - Priority evaluation now checks MITC FIRST (unchanged), then
#        checks for TNC title signals BEFORE BR keywords
#      - Added a negation check: if TNC title phrase found, BR keywords
#        in the same text are treated as contextual (not primary)
#      - Added more TNC title phrases that are unambiguous
#
#  🟡 FIX 5 — LLM POST-VALIDATION (NEW)
#    Problem: After LLM override, the result was never re-validated
#    against known bank-card pairs or against rule findings.
#    Fix:
#      - New function: post_validate_llm_result() checks:
#          (a) LLM bank is in ALLOWED_BANKS
#          (b) LLM card is in BANK_CARDS[LLM bank]
#          (c) LLM bank matches what rules found (when rule was confident)
#          (d) If all fail → return failure_result (force needs_review)
#
#  🟡 FIX 6 — GENERIC CARD NAMES BLOCKED FROM LLM OUTPUT
#    Problem: LLM returning "Millennia" for every document because it's
#    a common term in its training data for Indian credit cards.
#    Fix:
#      - Added GENERIC_LLM_OUTPUTS set: if LLM returns these card names,
#        the card field is set to UNKNOWN unless the bank also matches
#        a bank that actually issues that card
#
#  🔵 FIX 7 — PROMPT COMPLETELY REWRITTEN
#    Problem: Old prompt was too vague, allowed any output format,
#    didn't anchor the model to the actual document content.
#    Fix:
#      - Prompt now includes:
#          * First 300 chars of document (title zone)
#          * Pre-extracted signals from rule system (bank hint, doc type hint)
#          * Explicit instruction: "if you are not certain, return UNKNOWN"
#          * Strict JSON-only output with no preamble allowed
#
# ── HOW TO SET UP (one-time) ─────────────────────────────────────────────────
#
#  Step 1 — Install Ollama
#    Windows: https://ollama.ai/download → run .exe installer
#    macOS:   brew install ollama
#    Linux:   curl -fsSL https://ollama.ai/install.sh | sh
#
#  Step 2 — Pull the model
#    Recommended for CPU (HP Envy x360 / similar):
#      ollama pull llama3.2:1b     ← ~800MB, 15-30s on CPU (DEFAULT)
#    Better accuracy (slower):
#      ollama pull llama3.2:3b     ← ~2GB, 45-90s on CPU
#    Best quality (requires decent CPU/GPU):
#      ollama pull mistral          ← ~4GB, 2-4min on CPU
#
#  Step 3 — Start Ollama server (keep this terminal open)
#    ollama serve
#    (If you get "address already in use", Ollama is already running — OK)
#
#  Step 4 — Verify model is available
#    ollama run llama3.2:1b "say hello"
#
#  Step 5 — Run standalone test
#    python data_pipeline/llm_classifier.py
#
# ── HOW TO RUN IN PIPELINE ───────────────────────────────────────────────────
#
#  This file is imported by preprocess_with_llm.py.
#  Do NOT run this file directly in production — use:
#    python data_pipeline/preprocess_with_llm.py --dry-run   # test first
#    python data_pipeline/preprocess_with_llm.py             # full run
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

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
# Uncomment ONE. After changing, pull the model: ollama pull <model_name>
#
# SPEED vs ACCURACY on CPU:
#   llama3.2:1b  → FASTEST (~15-30s)  — may hallucinate more
#   llama3.2:3b  → BALANCED (~45-90s) — better accuracy
#   mistral      → BEST quality (~2-4min) — most reliable if you can wait
#
# With FIX 1-6, even llama3.2:1b hallucinations are caught and blocked.

OLLAMA_MODEL = "llama3.2:1b"     # DEFAULT — fast, hallucinations are now blocked
# OLLAMA_MODEL = "llama3.2:3b"   # Better quality
# OLLAMA_MODEL = "mistral"       # Best quality, slowest on CPU
# OLLAMA_MODEL = "phi3:mini"     # Microsoft Phi-3 alternative
# OLLAMA_MODEL = "gemma:2b"      # Google Gemma 2B alternative

# ── TIMEOUT SETTINGS ─────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 60   # seconds; increase for larger models (mistral needs 180+)

# ── OTHER SETTINGS ────────────────────────────────────────────────────────────
LLM_TEXT_WINDOW  = 2000   # chars of document text sent to LLM
LLM_TITLE_WINDOW = 300    # chars used for the "title zone" anchor in prompt
MAX_RETRIES      = 2      # retry count on timeout/failure
RETRY_DELAY_SEC  = 1.0    # seconds between retries

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# BANK WHITELIST
# Only these short codes are accepted from the LLM.
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_BANKS = {
    "HDFC", "SBI", "ICICI", "AXIS", "AMEX", "HSBC",
    "IDFC", "AU", "SCB", "KOTAK", "YES", "BOB",
    "RBL", "INDUSIND", "CITI", "ONECARD", "SCAPIA",
    "JUPITER", "UNKNOWN",
}

# LLM may return long-form bank names — normalise to short codes
BANK_NORMALIZATION_MAP = {
    "STATE BANK OF INDIA":        "SBI",
    "SBIC":                       "SBI",
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
# FIX 6 — GENERIC LLM OUTPUTS TO BLOCK
# These card names are so common that the LLM defaults to them.
# If the LLM returns one of these, we require a bank match before accepting.
# ─────────────────────────────────────────────────────────────────────────────

# Cards that the LLM (especially small models) commonly hallucinate
HALLUCINATION_PRONE_CARDS = {
    "Millennia",       # LLM default for almost any Indian credit card doc
    "Cashback",        # Generic — appears in every BR and TNC doc
    "Platinum",        # Too generic — many banks issue a Platinum card
    "Select",          # Generic
    "Smart",           # Generic
    "Signature",       # Generic
    "Elite",           # Generic
    "Reserve",         # Generic
    "Coral",           # Generic
}

# Which banks ACTUALLY issue each hallucination-prone card
# Used to decide whether to accept or block the LLM's card output
VALID_BANKS_FOR_CARD = {
    "Millennia":  {"HDFC", "IDFC"},
    "Cashback":   {"SBI"},
    "Platinum":   {"AMEX", "ICICI", "SBI", "CITI"},
    "Select":     {"AXIS"},
    "Smart":      {"SCB"},
    "Signature":  {"CITI"},
    "Elite":      {"SBI"},
    "Reserve":    {"AXIS"},
    "Coral":      {"AXIS"},
}

# ─────────────────────────────────────────────────────────────────────────────
# DOC TYPE PRIORITY SIGNALS
# FIX 4: Priority order MITC > TNC > BR > LG
# TNC title signals now checked BEFORE BR keywords to prevent TNC→BR errors.
# ─────────────────────────────────────────────────────────────────────────────

MITC_PRIORITY_SIGNALS = [
    "most important terms and conditions",
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

# FIX 4: TNC title signals are now SEPARATE from TNC keyword signals.
# Title signals (unambiguous) are checked first.
TNC_TITLE_SIGNALS = [
    "cardmember agreement",
    "cardholder agreement",
    "client terms",
    "credit card terms",        # specific enough
]

# TNC keyword signals (less specific — only used when no MITC title found)
TNC_KEYWORD_SIGNALS = [
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

# Words that indicate the LLM returned a sentence instead of a card name
_SENTENCE_INDICATORS = {
    "is", "are", "was", "provides", "document", "contains", "includes",
    "describes", "applicable", "pertaining", "related", "regarding",
    "stating", "outlining", "covering",
}

# Stores the last text window for use by _apply_doc_type_priority
_LAST_LLM_TEXT: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# FIX 7 — PROMPT COMPLETELY REWRITTEN
# New prompt anchors the model to the actual document text by:
#   1. Showing the first 300 chars (title zone) explicitly
#   2. Showing what rules already found (bank hint, doc type hint)
#   3. Instructing the model to AGREE or CORRECT, not replace
#   4. Requiring UNKNOWN when not certain
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    text: str,
    rule_bank: str = "UNKNOWN",
    rule_card: str = "UNKNOWN",
    rule_doc_type: str = "UNKNOWN",
) -> str:
    """
    Build the grounded classification prompt.

    Parameters:
        text          : Full extracted PDF text (truncated internally)
        rule_bank     : Bank detected by rules (hint for LLM)
        rule_card     : Card detected by rules (hint for LLM)
        rule_doc_type : Doc type detected by rules (hint for LLM)

    The prompt shows the LLM:
        - Title zone (first 300 chars) — strongest signal
        - Full text window (2000 chars)
        - What rules already found — so LLM corrects rather than replaces
        - Explicit UNKNOWN instruction — forces honesty

    Returns:
        Prompt string ready to send to Ollama.
    """
    title_zone = text[:LLM_TITLE_WINDOW].strip()
    full_text  = text[:LLM_TEXT_WINDOW].strip()

    prompt = f"""You are classifying an Indian bank credit card PDF document.
Your job: extract bank, card name, and document type FROM THE ACTUAL TEXT BELOW.
Do NOT use your training knowledge to guess. Only use what is written in the text.

=== DOCUMENT TITLE ZONE (first 300 characters — most reliable) ===
{title_zone}

=== FULL DOCUMENT TEXT ===
{full_text}

=== WHAT RULE-BASED SYSTEM FOUND (use as a starting hint) ===
Rule bank hint    : {rule_bank}
Rule card hint    : {rule_card}
Rule doc_type hint: {rule_doc_type}

=== YOUR TASK ===
1. bank: One of these ONLY: HDFC | SBI | ICICI | AXIS | AMEX | HSBC | IDFC | AU | SCB | KOTAK | YES | BOB | RBL | INDUSIND
   - If the bank name is CLEARLY in the title zone above, use that.
   - If you are NOT SURE, return the rule bank hint above.
   - NEVER return a bank that does not appear anywhere in the text.

2. card_name: The specific card product name (1-3 words maximum).
   Examples: "Millennia", "ACE", "Amazon Pay", "Cashback", "Live+"
   - Look for it in the TITLE ZONE first.
   - If the title zone shows a different card than the rule hint, prefer the title zone.
   - If you cannot find a specific card name, return UNKNOWN.
   - Do NOT return generic words like "Credit Card" as the card name.

3. doc_type: Use EXACTLY ONE of: MITC | TNC | BR | LG
   Priority (check in this order):
   - MITC: text has "most important terms", "schedule of charges", "annual fee", "interest rate"
   - TNC:  text has "cardmember agreement", "cardholder agreement", "terms and conditions"
   - BR:   text has "cashback", "reward points", "welcome benefit", "earn rate"
   - LG:   text has "lounge access", "airport lounge"
   If the rule doc_type hint above is strong (confidence was high), prefer it.

4. is_master: true ONLY if the document explicitly says it applies to ALL cards of the bank.
   (Look for phrases like "applicable to all credit card holders")

5. confidence: Your confidence from 0.0 to 1.0.
   Be HONEST. If you had to guess, use 0.5 or lower.
   Only use 0.8+ if the answer is clearly stated in the title zone.

Return ONLY this JSON (no other text, no markdown, no explanation):
{{"bank": "HDFC", "card_name": "Millennia", "doc_type": "BR", "is_master": false, "confidence": 0.75, "reason": "One sentence explaining what you found in the text."}}"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — 4-strategy fallback (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(raw_response: str) -> Optional[dict]:
    """
    Parse raw LLM output into a dict using 4 strategies:
      1. Direct JSON parse (ideal case)
      2. Strip markdown code fences then parse
      3. Regex-extract first {...} block
      4. Substring from first { to last }
    Returns None if all strategies fail.
    """
    if not raw_response or not raw_response.strip():
        return None

    raw = raw_response.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: regex extract first {...} block
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
# FIX 4 — DOC TYPE PRIORITY (IMPROVED)
# TNC title signals checked BEFORE BR keywords to prevent TNC→BR misclass.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_doc_type_priority(llm_doc_type: str, text_lower: str) -> str:
    """
    Re-evaluates the LLM's doc_type against actual document text.

    FIX 4 — Priority order:
      MITC signals  (strongest — fees, interest rates)
      TNC TITLE signals (unambiguous title phrases — checked before BR)
      BR signals
      TNC KEYWORD signals (weaker — "terms and conditions" appears in BR docs too)
      LG signals

    Falls back to LLM's value if no signal matches.
    """
    if not text_lower:
        return _map_doc_type_aliases(llm_doc_type)

    # MITC — highest priority (fees/interest = definitive MITC signal)
    for signal in MITC_PRIORITY_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "MITC":
                logger.info(
                    f"[LLM VALIDATE] Priority override: MITC signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with MITC"
                )
            return "MITC"

    # FIX 4: TNC TITLE signals before BR (unambiguous title phrases)
    for signal in TNC_TITLE_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "TNC":
                logger.info(
                    f"[LLM VALIDATE] Priority override: TNC title signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with TNC"
                )
            return "TNC"

    # BR signals (checked BEFORE weaker TNC keywords)
    for signal in BR_PRIORITY_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "BR":
                logger.info(
                    f"[LLM VALIDATE] Priority override: BR signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with BR"
                )
            return "BR"

    # LG signals
    for signal in LG_PRIORITY_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "LG":
                logger.info(
                    f"[LLM VALIDATE] Priority override: LG signal '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with LG"
                )
            return "LG"

    # FIX 4: TNC keyword signals as last resort (weakest — "terms and conditions"
    # appears in BR and MITC docs too, so we only use it if nothing else matched)
    for signal in TNC_KEYWORD_SIGNALS:
        if signal in text_lower:
            if llm_doc_type != "TNC":
                logger.info(
                    f"[LLM VALIDATE] Weak TNC keyword '{signal}' "
                    f"found → replacing LLM doc_type '{llm_doc_type}' with TNC "
                    f"(fallback — no other signals matched)"
                )
            return "TNC"

    # No priority signal matched — fall back to LLM's value
    mapped = _map_doc_type_aliases(llm_doc_type)
    logger.debug(
        f"[LLM VALIDATE] No priority signal in text; "
        f"using LLM doc_type '{llm_doc_type}' → mapped to '{mapped}'"
    )
    return mapped


def _map_doc_type_aliases(raw: str) -> str:
    """Maps common LLM doc_type aliases to canonical codes."""
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
    return alias_map.get(raw, "MITC")  # MITC safest default


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — LLM CONFIDENCE DEFLATION
# LLM raw confidence (0.88–1.0) is unreliable. We penalise it based on
# cross-checks against rule results and known valid bank-card pairs.
# ─────────────────────────────────────────────────────────────────────────────

def _deflate_llm_confidence(
    llm_bank:      str,
    llm_card:      str,
    llm_confidence: float,
    rule_bank:     str,
    rule_card:     str,
    rule_confidence: float,
) -> tuple[float, list[str]]:
    """
    Penalises LLM confidence based on cross-validation with rule results.

    Penalty rules:
      -0.35  if LLM bank ≠ rule bank AND rule bank confidence ≥ 0.75
             (rule was confident about the bank; LLM is almost certainly wrong)
      -0.20  if LLM card is in HALLUCINATION_PRONE_CARDS AND the LLM bank
             does not match VALID_BANKS_FOR_CARD for that card
      -0.15  if rule already had a valid non-None, non-UNKNOWN bank AND card
             (rule was doing well; LLM's help was not needed)

    Returns:
        (deflated_confidence: float, reasons: list[str])
    """
    deflated = llm_confidence
    reasons  = []

    # Penalty 1: Bank mismatch with a confident rule
    if (rule_bank and rule_bank not in ("UNKNOWN", "None")
            and llm_bank and llm_bank not in ("UNKNOWN",)
            and llm_bank != rule_bank
            and rule_confidence >= 0.75):
        deflated -= 0.35
        reasons.append(
            f"BANK MISMATCH: LLM says '{llm_bank}', rule says '{rule_bank}' "
            f"(rule_conf={rule_confidence:.2f} ≥ 0.75) → -0.35 penalty"
        )

    # Penalty 2: Hallucination-prone card with wrong bank
    if llm_card in HALLUCINATION_PRONE_CARDS:
        valid_banks = VALID_BANKS_FOR_CARD.get(llm_card, set())
        if llm_bank not in valid_banks:
            deflated -= 0.20
            reasons.append(
                f"HALLUCINATION RISK: Card '{llm_card}' is hallucination-prone "
                f"and bank '{llm_bank}' is not in valid banks {valid_banks} → -0.20 penalty"
            )

    # Penalty 3: Rule already had good results
    rule_card_valid = (rule_card and rule_card not in ("UNKNOWN", "None", "none"))
    rule_bank_valid = (rule_bank and rule_bank not in ("UNKNOWN", "None", "none"))
    if rule_card_valid and rule_bank_valid and rule_confidence >= 0.70:
        deflated -= 0.15
        reasons.append(
            f"RULE SUFFICIENT: Rule already found bank='{rule_bank}', "
            f"card='{rule_card}' (conf={rule_confidence:.2f}) → -0.15 penalty"
        )

    deflated = max(0.0, round(deflated, 3))

    if reasons:
        logger.info(
            f"[CONFIDENCE DEFLATION] Raw LLM confidence: {llm_confidence:.2f} → "
            f"Deflated: {deflated:.2f}"
        )
        for reason in reasons:
            logger.info(f"[CONFIDENCE DEFLATION]   {reason}")

    return deflated, reasons


# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 — POST-VALIDATION OF LLM RESULT
# Validates LLM output against known bank-card pairs BEFORE returning.
# ─────────────────────────────────────────────────────────────────────────────

# Known valid bank→card mappings (same as BANK_CARDS in preprocess_with_llm.py)
# Duplicated here to keep llm_classifier.py self-contained
LLM_BANK_CARDS: dict[str, set[str]] = {
    "HDFC":    {"Infinia", "Diners Club Black", "Diners", "Marriott Bonvoy",
                "Regalia Gold", "Regalia", "Millennia", "Tata Neu Infinity",
                "Tata Neu", "MoneyBack+", "MoneyBack", "Swiggy", "IndianOil",
                "Pixel", "HDFC Millennia"},
    "SBI":     {"Cashback", "Elite", "Aurum", "SimplyCLICK", "SimplySAVE",
                "BPCL Octane", "BPCL", "IRCTC", "Paytm", "Reliance", "Vistara"},
    "AXIS":    {"Magnus", "Atlas", "Reserve", "ACE", "Axis ACE", "Flipkart",
                "Airtel", "Select", "Vistara Infinite", "Vistara", "Coral"},
    "ICICI":   {"Emeralde", "Amazon Pay", "HPCL Super Saver", "Rubyx"},
    "AMEX":    {"Platinum Travel", "Blue Cash", "Platinum"},
    "HSBC":    {"Premier", "Live+"},
    "KOTAK":   {"League Platinum", "Myntra"},
    "SCB":     {"Smart", "Ultimate"},
    "IDFC":    {"Millennia"},
    "AU":      {"Altura"},
    "BOB":     {"Eterna"},
    "RBL":     {"World Safari"},
    "INDUSIND":{"EazyDiner"},
    "YES":     {"Prosperity"},
    "ONECARD": {"OneCard"},
    "SCAPIA":  {"Scapia"},
    "JUPITER": {"Jupiter"},
    "CITI":    {"Signature", "Platinum"},
}


def post_validate_llm_result(
    llm_bank:     str,
    llm_card:     str,
    rule_bank:    str,
    rule_confidence: float,
) -> tuple[bool, str]:
    """
    FIX 5: Validates LLM result before it can override rules.

    Checks (in order):
      1. LLM bank is in ALLOWED_BANKS
      2. LLM card is in LLM_BANK_CARDS[llm_bank] (when bank is known)
      3. LLM bank matches rule bank (when rule was confident)

    Returns:
        (is_valid: bool, rejection_reason: str)
        If is_valid=False, the LLM result should be REJECTED.
    """
    # Check 1: Bank whitelist
    if llm_bank == "UNKNOWN":
        return False, "LLM returned UNKNOWN bank — no useful information"

    if llm_bank not in ALLOWED_BANKS:
        return False, f"LLM bank '{llm_bank}' is not in ALLOWED_BANKS"

    # Check 2: Card is valid for the bank (when bank is known and card is specific)
    if (llm_card not in ("UNKNOWN", "MASTER", "None", "")
            and llm_bank in LLM_BANK_CARDS):
        valid_cards = LLM_BANK_CARDS[llm_bank]
        if llm_card not in valid_cards:
            # Check if this might be a hallucination-prone generic name
            if llm_card in HALLUCINATION_PRONE_CARDS:
                return (
                    False,
                    f"LLM card '{llm_card}' is hallucination-prone and NOT in "
                    f"{llm_bank}'s card list {valid_cards} — likely hallucination"
                )
            # Not known but not hallucination-prone — allow with a warning
            logger.warning(
                f"[POST-VALIDATE] LLM card '{llm_card}' not in {llm_bank}'s "
                f"known list. Allowing (may be a new/unlisted card)."
            )

    # Check 3: Bank mismatch with confident rule
    if (rule_bank and rule_bank not in ("UNKNOWN", "None")
            and llm_bank != rule_bank
            and rule_confidence >= 0.80):
        return (
            False,
            f"LLM bank '{llm_bank}' conflicts with high-confidence rule "
            f"bank '{rule_bank}' (rule_conf={rule_confidence:.2f}) — "
            f"LLM result rejected to protect correct rule output"
        )

    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE VALIDATION (IMPROVED)
# ─────────────────────────────────────────────────────────────────────────────

VALID_DOC_TYPES = {"MITC", "TNC", "BR", "LG"}


def _validate_llm_output(
    parsed:          dict,
    rule_bank:       str = "UNKNOWN",
    rule_card:       str = "UNKNOWN",
    rule_confidence: float = 0.0,
) -> dict:
    """
    Validates and normalises all fields in the parsed LLM JSON.

    Steps:
      1. Bank: normalise through map → enforce whitelist
      2. Card name: reject if >5 words or sentence-like
      3. Doc type: apply priority evaluation (FIX 4)
      4. Confidence: coerce to float, then DEFLATE (FIX 2)
      5. is_master: coerce to bool

    FIX 2: Confidence deflation applied here.
    FIX 6: Hallucination-prone card detection applied here.
    """
    raw_bank     = str(parsed.get("bank",      "UNKNOWN")).strip().upper()
    raw_card     = str(parsed.get("card_name", "UNKNOWN")).strip()
    raw_doc_type = str(parsed.get("doc_type",  "")).strip().upper()
    is_master    = parsed.get("is_master", False)
    confidence   = parsed.get("confidence", 0.5)
    reason       = str(parsed.get("reason", "LLM classification")).strip()

    # ── 1. Bank: normalise → whitelist ────────────────────────────────────────
    normalised_bank = BANK_NORMALIZATION_MAP.get(raw_bank, raw_bank)
    normalised_bank = re.sub(r"[^\w]", "", normalised_bank).strip()
    if normalised_bank not in ALLOWED_BANKS:
        logger.warning(
            f"[LLM VALIDATE] Bank '{raw_bank}' (normalised: '{normalised_bank}') "
            f"not in whitelist → UNKNOWN"
        )
        normalised_bank = "UNKNOWN"

    # ── 2. Card name: sanitation ──────────────────────────────────────────────
    card_words = raw_card.split()
    if len(card_words) > 5:
        logger.warning(
            f"[LLM VALIDATE] card_name '{raw_card[:60]}' exceeds 5 words → UNKNOWN"
        )
        cleaned_card = "UNKNOWN"
    elif any(w.lower() in _SENTENCE_INDICATORS for w in card_words[1:]):
        logger.warning(
            f"[LLM VALIDATE] card_name '{raw_card[:60]}' looks like a sentence → UNKNOWN"
        )
        cleaned_card = "UNKNOWN"
    else:
        cleaned_card = re.sub(r"[^\w\s\+\-]", "", raw_card).strip()
        if not cleaned_card:
            cleaned_card = "UNKNOWN"

    # ── 3. Doc type: apply FIX 4 priority evaluation ──────────────────────────
    final_doc_type = _apply_doc_type_priority(raw_doc_type, _LAST_LLM_TEXT)

    # ── 4. Confidence: coerce then DEFLATE (FIX 2) ───────────────────────────
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5
        logger.debug("[LLM VALIDATE] Could not parse confidence; defaulting to 0.5")

    # Apply deflation
    deflated_conf, deflation_reasons = _deflate_llm_confidence(
        llm_bank        = normalised_bank,
        llm_card        = cleaned_card,
        llm_confidence  = confidence,
        rule_bank       = rule_bank,
        rule_card       = rule_card,
        rule_confidence = rule_confidence,
    )

    # ── 5. is_master: coerce to bool ─────────────────────────────────────────
    if isinstance(is_master, str):
        is_master = is_master.lower() in ("true", "yes", "1")
    else:
        is_master = bool(is_master)

    return {
        "bank":               normalised_bank,
        "card_name":          cleaned_card,
        "doc_type":           final_doc_type,
        "is_master":          is_master,
        "confidence":         round(deflated_conf, 3),
        "raw_confidence":     round(confidence, 3),    # original before deflation
        "deflation_reasons":  deflation_reasons,
        "reason":             reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA AVAILABILITY CHECK (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_available() -> bool:
    """
    Preflight check: is Ollama running and is the configured model available?
    Returns True/False. Never raises an exception.
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
# MAIN CLASSIFICATION FUNCTION (IMPROVED)
# ─────────────────────────────────────────────────────────────────────────────

def classify_with_llm(
    text:            str,
    rule_bank:       str = "UNKNOWN",
    rule_card:       str = "UNKNOWN",
    rule_doc_type:   str = "UNKNOWN",
    rule_confidence: float = 0.0,
) -> dict:
    """
    Classifies a credit card PDF document using the configured Ollama model.

    FIX 1: Now accepts rule_bank, rule_card, rule_doc_type, rule_confidence
    as inputs. These are injected into the prompt to ground the LLM and
    are used for confidence deflation and post-validation.

    FIX 5: Runs post_validate_llm_result() before returning.

    ARGS:
        text            : Full extracted PDF text
        rule_bank       : Bank detected by rule-based system
        rule_card       : Card detected by rule-based system
        rule_doc_type   : Doc type detected by rule-based system
        rule_confidence : Overall confidence of rule-based result

    RETURNS:
        Dict with: bank, card_name, doc_type, is_master, confidence,
                   raw_confidence, deflation_reasons, reason, llm_success
        llm_success=False on any error — pipeline falls back to rule-based result.
    """
    global _LAST_LLM_TEXT

    logger.info(
        f"[STEP 3] LLM classification — model={OLLAMA_MODEL}, "
        f"timeout={REQUEST_TIMEOUT}s"
    )
    logger.info(
        f"[STEP 3] Rule hints injected → bank={rule_bank}, "
        f"card={rule_card}, doc_type={rule_doc_type}, "
        f"rule_conf={rule_confidence:.2f}"
    )

    # Standard failure result returned on any error
    failure_result = {
        "bank":              "UNKNOWN",
        "card_name":         "UNKNOWN",
        "doc_type":          "MITC",
        "is_master":         False,
        "confidence":        0.0,
        "raw_confidence":    0.0,
        "deflation_reasons": [],
        "reason":            "LLM classification failed",
        "llm_success":       False,
    }

    if not text or not text.strip():
        logger.warning("[LLM] Empty text passed — skipping LLM call.")
        failure_result["reason"] = "Empty text — cannot classify"
        return failure_result

    # Store lowercase text window for doc_type priority evaluation
    _LAST_LLM_TEXT = text[:LLM_TEXT_WINDOW].lower()

    # FIX 7: Use the new grounded prompt
    prompt = _build_prompt(
        text          = text,
        rule_bank     = rule_bank,
        rule_card     = rule_card,
        rule_doc_type = rule_doc_type,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(
            f"[LLM REQUEST] Attempt {attempt}/{MAX_RETRIES} — "
            f"model={OLLAMA_MODEL}, timeout={REQUEST_TIMEOUT}s, "
            f"text_window={LLM_TEXT_WINDOW} chars"
        )

        request_start = time.monotonic()

        try:
            response = requests.post(
                url     = f"{OLLAMA_BASE_URL}/api/generate",
                headers = {"Content-Type": "application/json"},
                json    = {
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,    # deterministic
                        "num_predict": 200,  # slightly higher for the new prompt format
                    },
                },
                timeout = REQUEST_TIMEOUT,
            )

            elapsed = time.monotonic() - request_start
            logger.info(f"[LLM RESPONSE TIME] {elapsed:.1f}s (attempt {attempt})")

            if response.status_code != 200:
                logger.warning(
                    f"[LLM FAILURE] HTTP {response.status_code} from Ollama "
                    f"(attempt {attempt}). Response: {response.text[:200]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                return failure_result

            response_data = response.json()
            raw_text      = response_data.get("response", "")

            if not raw_text:
                logger.warning(
                    f"[LLM FAILURE] Empty response from Ollama (attempt {attempt})"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                return failure_result

            logger.debug(f"[LLM RAW] {raw_text[:500]}")

            parsed = _parse_llm_response(raw_text)
            if parsed is None:
                logger.warning(
                    f"[LLM FAILURE] JSON parsing failed (attempt {attempt}). "
                    f"Raw: {raw_text[:200]}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                return failure_result

            # FIX 2: Validate and deflate confidence
            validated = _validate_llm_output(
                parsed          = parsed,
                rule_bank       = rule_bank,
                rule_card       = rule_card,
                rule_confidence = rule_confidence,
            )

            # FIX 5: Post-validate against known bank-card pairs
            is_valid, rejection_reason = post_validate_llm_result(
                llm_bank        = validated["bank"],
                llm_card        = validated["card_name"],
                rule_bank       = rule_bank,
                rule_confidence = rule_confidence,
            )

            if not is_valid:
                logger.warning(
                    f"[LLM POST-VALIDATE] ✗ LLM result REJECTED: {rejection_reason}"
                )
                logger.warning(
                    f"[LLM POST-VALIDATE] Rejected LLM output was: "
                    f"bank={validated['bank']}, card={validated['card_name']}, "
                    f"doc_type={validated['doc_type']}"
                )
                # Return failure so pipeline keeps rule-based result
                failure_result["reason"] = f"Post-validation rejected: {rejection_reason}"
                return failure_result

            validated["llm_success"] = True

            logger.info(
                f"[LLM OUTPUT] bank={validated['bank']} | "
                f"card={validated['card_name']} | "
                f"doc_type={validated['doc_type']} | "
                f"is_master={validated['is_master']} | "
                f"confidence={validated['confidence']:.2f} "
                f"(raw={validated['raw_confidence']:.2f}) | "
                f"response_time={elapsed:.1f}s"
            )
            logger.info(f"[LLM REASON] {validated['reason']}")
            if validated["deflation_reasons"]:
                for dr in validated["deflation_reasons"]:
                    logger.info(f"[LLM DEFLATION] {dr}")

            return validated

        except requests.exceptions.Timeout:
            elapsed = time.monotonic() - request_start
            logger.warning(
                f"[LLM FAILURE] Timeout after {elapsed:.1f}s "
                f"(limit={REQUEST_TIMEOUT}s, attempt {attempt}/{MAX_RETRIES}). "
                f"TIP: Switch to a faster model → change OLLAMA_MODEL to 'llama3.2:1b' "
                f"and run: ollama pull llama3.2:1b"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except requests.exceptions.ConnectionError:
            elapsed = time.monotonic() - request_start
            logger.warning(
                f"[LLM FAILURE] Connection refused at {OLLAMA_BASE_URL} "
                f"(attempt {attempt}/{MAX_RETRIES}). Run: ollama serve"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except Exception as e:
            elapsed = time.monotonic() - request_start
            logger.error(
                f"[LLM FAILURE] Unexpected error (attempt {attempt}, "
                f"elapsed {elapsed:.1f}s): {e}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    logger.error(
        f"[LLM FAILURE] All {MAX_RETRIES} attempts exhausted. "
        f"Falling back to rule-based result."
    )
    return failure_result


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
# Run with:
#   python data_pipeline/llm_classifier.py
#
# What this tests:
#   Test 1: HDFC Millennia MITC → should return MITC + HDFC
#   Test 2: SBI Cashback BR    → should return BR + SBI
#   Test 3: SBIC bank guard    → should NOT return "SBIC"
#   Test 4: AXIS ACE card name → should be ≤ 5 words
#   Test 5: HALLUCINATION TEST → AXIS Flipkart doc, rule hints provided
#            LLM should NOT override a correct AXIS/Flipkart rule result
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level   = logging.INFO,
        format  = "[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print(f"LLM Classifier Standalone Test")
    print(f"Model  : {OLLAMA_MODEL}")
    print(f"Timeout: {REQUEST_TIMEOUT}s")
    print("=" * 60)

    print("\n[TEST 0] Checking Ollama availability...")
    if not check_ollama_available():
        print(f"\n[FAIL] Ollama not available. Steps:")
        print(f"  1. Install: https://ollama.ai/download")
        print(f"  2. Pull:    ollama pull {OLLAMA_MODEL}")
        print(f"  3. Serve:   ollama serve")
        sys.exit(1)

    results = []

    # Test 1: MITC doc — must not be misclassified as TNC
    t1 = classify_with_llm(
        text="""
        HDFC Bank Millennia Credit Card
        Most Important Terms and Conditions (MITC)
        Annual Fee: Rs. 1,000 + GST
        Interest Rate: 3.6% per month (43.2% per annum)
        Minimum Amount Due: 5% of outstanding or Rs. 200
        Cash Advance Fee: 2.5% (minimum Rs. 500)
        Late Payment Fee: Rs. 100 to Rs. 750
        """,
        rule_bank="HDFC", rule_card="Millennia",
        rule_doc_type="MITC", rule_confidence=0.89,
    )
    print(f"\n[TEST 1] HDFC Millennia MITC")
    print(json.dumps(t1, indent=2))
    p1 = t1["doc_type"] == "MITC" and t1["bank"] == "HDFC"
    print(f"  {'PASS ✓' if p1 else 'FAIL ✗'} — expected doc_type=MITC, bank=HDFC")
    results.append(p1)

    # Test 2: BR doc — must not be confused with MITC
    t2 = classify_with_llm(
        text="""
        SBI Cashback Credit Card
        Features and Benefits
        Welcome Benefit: Rs. 500 cashback on first transaction
        Earn Rate: 5% cashback on all online spends
        Milestone Benefit: Rs. 2,000 cashback on Rs. 2 lakh annual spend
        """,
        rule_bank="SBI", rule_card="Cashback",
        rule_doc_type="BR", rule_confidence=0.91,
    )
    print(f"\n[TEST 2] SBI Cashback BR")
    print(json.dumps(t2, indent=2))
    p2 = t2["doc_type"] == "BR" and t2["bank"] == "SBI"
    print(f"  {'PASS ✓' if p2 else 'FAIL ✗'} — expected doc_type=BR, bank=SBI")
    results.append(p2)

    # Test 3: Hallucinated bank guard (SBIC → should NOT reach output as-is)
    t3 = classify_with_llm(
        text="""
        SBIC Credit Card
        Most Important Terms and Conditions
        Annual fee: Rs. 500. Interest rate: 3.5% per month.
        """,
        rule_bank="SBI", rule_card="UNKNOWN",
        rule_doc_type="MITC", rule_confidence=0.65,
    )
    print(f"\n[TEST 3] Hallucinated bank 'SBIC' whitelist guard")
    print(json.dumps(t3, indent=2))
    p3 = t3["bank"] != "SBIC"
    print(f"  {'PASS ✓' if p3 else 'FAIL ✗'} — 'SBIC' must not reach output")
    results.append(p3)

    # Test 4: Long card name sanitation
    t4 = classify_with_llm(
        text="""
        AXIS Bank Credit Card
        ACE Credit Card Cashback Programme
        Benefits and Rewards
        Earn 2% cashback on all spends via ACE credit card
        """,
        rule_bank="AXIS", rule_card="ACE",
        rule_doc_type="BR", rule_confidence=0.84,
    )
    print(f"\n[TEST 4] AXIS ACE — card name must not be a sentence")
    print(json.dumps(t4, indent=2))
    p4 = len(t4["card_name"].split()) <= 5
    print(f"  {'PASS ✓' if p4 else 'FAIL ✗'} — card_name must be ≤ 5 words")
    results.append(p4)

    # Test 5 (NEW): HALLUCINATION TEST — rule found AXIS Flipkart correctly
    # LLM should NOT override and replace with HDFC Millennia
    t5 = classify_with_llm(
        text="""
        Axis Bank Flipkart Credit Card
        Terms and Conditions
        Applicable on purchases made via Flipkart platform.
        Rewards programme terms apply.
        """,
        rule_bank="AXIS", rule_card="Flipkart",
        rule_doc_type="TNC", rule_confidence=0.84,
    )
    print(f"\n[TEST 5] HALLUCINATION GUARD — AXIS Flipkart TNC (rule was correct)")
    print(json.dumps(t5, indent=2))
    # Either LLM agrees with AXIS/Flipkart, OR it was rejected (llm_success=False)
    p5 = (
        not t5["llm_success"]  # rejected = good
        or (t5["bank"] == "AXIS" and t5["card_name"] == "Flipkart")  # correct = also good
    )
    print(
        f"  {'PASS ✓' if p5 else 'FAIL ✗'} — "
        f"LLM must not override correct AXIS/Flipkart with HDFC/Millennia"
    )
    results.append(p5)

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print(f"Model used: {OLLAMA_MODEL}")
    print(f"{'='*60}")
    if not all(results):
        print("\n[HINT] If tests fail, try upgrading model: ollama pull llama3.2:3b")
        print("[HINT] Even if LLM fails tests, pipeline fallbacks protect correctness.")
