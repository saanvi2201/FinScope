# =============================================================================
# parser.py — NLP Text Parser for Behavioral Agent
# =============================================================================
# Converts raw user text into:
#   - detected keywords
#   - (category → intensity_score) mapping
#
# FIXES APPLIED:
#   [Issue #1]  Weak/partial text parsing → extended phrase list in mapper.py
#   [Issue #2]  Intensity not per-category → now linked per-keyword/category
#   [Issue #3]  Weight distribution wrong → scores = frequency × intensity
#   [Issue #4]  Default category leakage → WEIGHT_FLOOR only on cold start
#   [Issue #6]  Invalid input handling → hard reset + safe fallback
#   [Issue #7]  State persistence bug → all state created fresh each call
#   [Issue #8]  Phrase-level understanding → phrases matched before single words
#   [Issue #10] Default mode triggers too easily → only on zero valid signals
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# =============================================================================

import re
import logging
from typing import Dict, List, Optional, Tuple

from defaults import (
    CATEGORIES,
    INTENSITY_MAP,
    DEFAULT_INTENSITY,
    WEIGHT_FLOOR,
    PROFESSION_ALIASES,
)
from mapper import extract_category_signals

logger = logging.getLogger(__name__)


# =============================================================================
# COLD START PERSONA PROFILES
# Used ONLY when zero signals are detected from user text.
# FIX [Issue #4]: These profiles are applied exclusively in cold-start paths.
# =============================================================================
COLD_START_PROFILES: Dict[str, Dict[str, float]] = {
    "student": {
        "dining":           0.30,
        "online_shopping":  0.25,
        "entertainment":    0.20,
        "groceries":        0.10,
        "travel":           0.10,
        "utilities":        0.03,
        "fuel":             0.01,
        "other":            0.01,
    },
    "traveler": {
        "travel":           0.40,
        "dining":           0.20,
        "online_shopping":  0.15,
        "groceries":        0.10,
        "entertainment":    0.05,
        "utilities":        0.05,
        "fuel":             0.03,
        "other":            0.02,
    },
    "salaried": {
        "groceries":        0.25,
        "dining":           0.20,
        "utilities":        0.15,
        "travel":           0.15,
        "online_shopping":  0.15,
        "entertainment":    0.05,
        "fuel":             0.03,
        "other":            0.02,
    },
    "homemaker": {
        "groceries":        0.35,
        "utilities":        0.25,
        "dining":           0.15,
        "online_shopping":  0.15,
        "entertainment":    0.05,
        "fuel":             0.02,
        "travel":           0.02,
        "other":            0.01,
    },
    "business": {
        "travel":           0.25,
        "dining":           0.20,
        "online_shopping":  0.20,
        "utilities":        0.15,
        "groceries":        0.10,
        "fuel":             0.05,
        "entertainment":    0.03,
        "other":            0.02,
    },
}

DEFAULT_COLD_START = {
    "dining":           0.20,
    "online_shopping":  0.20,
    "groceries":        0.20,
    "travel":           0.15,
    "utilities":        0.10,
    "entertainment":    0.05,
    "fuel":             0.05,
    "other":            0.05,
}


# =============================================================================
# is_valid_input()
# FIX [Issue #6]: Validate input early — catch gibberish/empty/numeric-only.
# =============================================================================
def is_valid_input(text: str) -> bool:
    """
    Returns False if input is:
    - Empty or whitespace only
    - Purely numeric / symbols (no real words)
    - Very short with no alphabetic content
    """
    if not text or not isinstance(text, str):
        return False

    stripped = text.strip()
    if not stripped:
        return False

    # Must contain at least 1 alphabetic character to be parseable
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars < 1:
        logger.warning(f"PARSER: Input has no alphabetic content: '{text}'")
        return False

    return True


# =============================================================================
# detect_intensity_from_context()
# FIX [Issue #2]: Intensity detected per keyword with local window search.
# =============================================================================
def detect_intensity_from_context(text: str, keyword: str) -> float:
    """
    Find intensity modifier in a CLAUSE-SCOPED window around the keyword.

    Strategy:
    1. Split text into clauses on connectors ("but", "and", "however", etc.)
    2. Find which clause contains the keyword
    3. Only search for intensity modifiers within THAT clause

    This prevents "I use Swiggy a lot but Uber sometimes" from assigning
    "sometimes" to Swiggy — each keyword is scoped to its own clause.

    Returns float from INTENSITY_MAP or DEFAULT_INTENSITY as fallback.
    """
    if not keyword:
        return DEFAULT_INTENSITY

    text_lower   = text.lower()
    keyword_lower = keyword.lower()

    if keyword_lower not in text_lower:
        return DEFAULT_INTENSITY

    # ── Step 1: Split into clauses on common connectors ───────────────────────
    # Reason: intensity words belong to the sub-clause that mentions the keyword,
    #         not to the entire sentence.
    clause_splitters = r'(?:\bbut\b|\band\b|\bhowever\b|\bwhile\b|\balso\b|,|;)'
    clauses = re.split(clause_splitters, text_lower)

    # ── Step 2: Find the clause(s) containing the keyword ─────────────────────
    matched_clauses = [c for c in clauses if keyword_lower in c]

    if not matched_clauses:
        # Fallback: use a tight character window (20 chars each side) as safety net
        kw_pos = text_lower.find(keyword_lower)
        ws = max(0, kw_pos - 20)
        we = min(len(text_lower), kw_pos + len(keyword_lower) + 20)
        matched_clauses = [text_lower[ws:we]]

    # ── Step 3: Search for intensity modifier within the matched clause ────────
    # FIX [Issue #2]: Use None sentinel so explicitly found low-intensity words
    #                 (e.g. "rarely" = 0.2) are not overridden by DEFAULT_INTENSITY (0.6).
    #                 max(0.6, 0.2) = 0.6 was silently swallowing "rarely" modifiers.
    found_intensity = None

    for clause in matched_clauses:
        logger.debug(f"  Intensity clause for '{keyword}': '{clause.strip()}'")
        # Longest modifier first to avoid partial matches
        for modifier, score in sorted(INTENSITY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if modifier in clause:
                logger.debug(f"  → Modifier '{modifier}' → intensity {score}")
                if found_intensity is None:
                    found_intensity = score
                else:
                    found_intensity = max(found_intensity, score)
                break  # One modifier per clause

    # Only fall back to DEFAULT_INTENSITY when no modifier was found at all
    return found_intensity if found_intensity is not None else DEFAULT_INTENSITY


# =============================================================================
# detect_profession()
# =============================================================================
def detect_profession(text: str, user_profile: Optional[dict]) -> Optional[str]:
    """
    Returns canonical profession key from PROFESSION_ALIASES or None.
    Priority: explicit user_profile field → text scan
    """
    # Priority 1: explicit user_profile
    if user_profile and "profession" in user_profile:
        raw = str(user_profile["profession"]).lower().strip()
        normalized = PROFESSION_ALIASES.get(raw)
        if normalized:
            logger.debug(f"  Profession from user_profile: '{raw}' → '{normalized}'")
            return normalized

    # Priority 2: text scan (longest alias first)
    text_lower = text.lower()
    for alias, canonical in sorted(PROFESSION_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
            logger.debug(f"  Profession from text: '{alias}' → '{canonical}'")
            return canonical

    return None


# =============================================================================
# parse_user_text()
# THE MAIN PARSER
#
# FIX [Issue #7]: ALL state is created fresh inside this function on every call.
#                 No global variables, no state reuse between calls.
# FIX [Issue #3]: score = frequency × intensity (not just count-based).
# FIX [Issue #4]: WEIGHT_FLOOR only applied on cold start / zero signal paths.
# FIX [Issue #6]: Invalid input detected upfront → hard reset + safe return.
# FIX [Issue #10]: Cold start ONLY when len(valid_categories) == 0.
# =============================================================================
def parse_user_text(
    user_text: str,
    user_profile: Optional[dict] = None
) -> Dict:
    """
    Master parsing function.

    Returns:
    {
        "raw_signals":        [(category, intensity), ...],
        "category_scores":    {cat: score, ...},   # intensity-weighted, NOT normalized
        "profession":         str | None,
        "keywords_found":     [str],
        "is_cold_start":      bool,
        "cold_start_persona": str | None,
        "input_valid":        bool,
    }
    """
    logger.info("=" * 60)
    logger.info("PARSER: Starting text parse")
    logger.info(f"Input: '{user_text}'")
    logger.info(f"Profile: {user_profile}")

    # ── FIX [Issue #7]: HARD RESET — fresh state every call ───────────────────
    # No global state. Everything below is local to this function call.
    category_scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    keywords_found:  List[str] = []
    raw_signals:     List[Tuple[str, float]] = []
    # ⚠️ IMPORTANT: Changing category names will break Simulation Agent

    # ── FIX [Issue #6]: Invalid input check ───────────────────────────────────
    if not is_valid_input(user_text):
        logger.warning(f"PARSER: Invalid/empty input detected: '{user_text}'")
        return {
            "raw_signals":        [],
            "category_scores":    {cat: 0.0 for cat in CATEGORIES},
            "profession":         None,
            "keywords_found":     [],
            "is_cold_start":      True,
            "cold_start_persona": None,
            "input_valid":        False,
        }

    # ── STEP 1: Extract all category signals from text ────────────────────────
    # FIX [Issue #8]: Phrase-level matching happens inside extract_category_signals
    #                 (mapper.py now matches longest phrases first).
    raw_hits = extract_category_signals(user_text)
    logger.info(f"PARSER Step 1 — Raw hits from mapper: {raw_hits}")

    # ── STEP 2: Per-keyword intensity detection ────────────────────────────────
    # FIX [Issue #2 & #3]: Intensity is now detected per detected keyword,
    #                       not globally for the whole sentence.
    #                       Score accumulates as: score += intensity (per hit)

    # Build a category → triggering keywords mapping for intensity lookup
    category_to_keywords: Dict[str, List[str]] = {cat: [] for cat in CATEGORIES}
    for (category, _) in raw_hits:
        kw = _find_triggering_keyword(user_text, category, already_found=category_to_keywords[category])
        if kw:
            category_to_keywords[category].append(kw)
            if kw not in keywords_found:
                keywords_found.append(kw)

    logger.info(f"PARSER Step 2 — Category→keywords map: {category_to_keywords}")

    # FIX [Issue #3]: Score = SUM of (intensity per keyword mention)
    # This correctly handles: "Swiggy a lot and Zomato daily" → dining gets 1.0 + 1.0 = 2.0
    for cat in CATEGORIES:
        kws = category_to_keywords[cat]
        if not kws:
            continue  # FIX [Issue #4]: No keywords → score stays 0.0 (no leakage)

        total_intensity = 0.0
        for kw in kws:
            intensity = detect_intensity_from_context(user_text, kw)
            total_intensity += intensity
            raw_signals.append((cat, intensity))
            logger.debug(f"  '{kw}' → {cat} → intensity {intensity:.2f}")

        category_scores[cat] = total_intensity

    logger.info(f"PARSER Step 3 — Accumulated scores (freq × intensity): {category_scores}")

    # ── STEP 3: Detect profession (for cold start + estimation) ───────────────
    profession = detect_profession(user_text, user_profile)
    logger.info(f"PARSER: Detected profession: '{profession}'")

    # ── STEP 4: Cold start detection ──────────────────────────────────────────
    # FIX [Issue #10]: Cold start ONLY if zero categories have any score.
    #                  Weak signals are still valid signals.
    active_categories = [cat for cat in CATEGORIES if category_scores[cat] > 0.0]
    is_cold_start = len(active_categories) == 0   # FIX [Issue #10]

    cold_start_persona = None
    if is_cold_start:
        logger.warning("PARSER: Cold start — zero signals detected. Using persona fallback.")
        cold_start_persona = profession
        persona_weights = _get_cold_start_weights(profession)
        for cat, weight in persona_weights.items():
            category_scores[cat] = weight
        # FIX [Issue #4]: WEIGHT_FLOOR only applied inside cold start path
        for cat in CATEGORIES:
            if category_scores[cat] < WEIGHT_FLOOR:
                category_scores[cat] = WEIGHT_FLOOR
        logger.info(f"PARSER: Cold start persona '{cold_start_persona}' applied.")
    else:
        # FIX [Issue #4]: For real detections, DO NOT apply floor to zero categories.
        #                 Zero means "not mentioned" — leave it as 0.
        logger.info(f"PARSER: {len(active_categories)} active categories detected — skipping floor.")

    logger.info(f"PARSER Final scores: {category_scores}")

    return {
        "raw_signals":        raw_signals,
        "category_scores":    category_scores,    # unnormalized; normalized in utils.py
        "profession":         profession,
        "keywords_found":     list(set(keywords_found)),
        "is_cold_start":      is_cold_start,
        "cold_start_persona": cold_start_persona,
        "input_valid":        True,
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _find_triggering_keyword(
    text: str,
    category: str,
    already_found: Optional[List[str]] = None
) -> Optional[str]:
    """
    Find which keyword in the text triggered a given category.
    FIX [Issue #2]: Returns ALL distinct keywords for a category
                    so intensity can be detected per-keyword.

    Uses word boundaries to avoid "air" matching inside "affair".
    """
    from mapper import BRAND_CATEGORY_MAP, LIFESTYLE_KEYWORD_MAP

    text_lower = text.lower()
    already_found = already_found or []

    # Check brands first (more specific), longest first
    for keyword, cat in sorted(BRAND_CATEGORY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if cat == category and keyword not in already_found:
            # Use word boundary where possible; fallback to substring for multi-word
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                return keyword
            elif ' ' in keyword and keyword in text_lower:
                # Multi-word phrase match (no word boundary needed at phrase level)
                return keyword

    # Then lifestyle phrases
    for keyword, cat in sorted(LIFESTYLE_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if cat == category and keyword not in already_found:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                return keyword
            elif ' ' in keyword and keyword in text_lower:
                return keyword

    return None


def _get_cold_start_weights(profession: Optional[str]) -> Dict[str, float]:
    """
    Returns persona-appropriate weights for cold start.
    Fallback → DEFAULT_COLD_START.
    """
    if profession and profession in COLD_START_PROFILES:
        return COLD_START_PROFILES[profession]

    # Map profession aliases to available profiles
    mapping = {
        "engineer":     "salaried",
        "doctor":       "salaried",
        "professional": "salaried",
        "freelancer":   "salaried",
        "retired":      "homemaker",
    }
    if profession and profession in mapping:
        return COLD_START_PROFILES[mapping[profession]]

    return DEFAULT_COLD_START
