# =============================================================================
# parser.py — NLP Text Parser for Behavioral Agent
# =============================================================================
# Converts raw user text into:
#   - detected keywords
#   - (category → intensity_score) mapping
#
# Pipeline: TEXT → KEYWORDS → CATEGORIES → WEIGHTS
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# =============================================================================

import re
import logging
from typing import Dict, List, Tuple

from defaults import (
    CATEGORIES,             # IMPORTANT: Changing category names will break Simulation Agent
    INTENSITY_MAP,
    DEFAULT_INTENSITY,
    WEIGHT_FLOOR,
    PROFESSION_ALIASES,
)
from mapper import extract_category_signals

logger = logging.getLogger(__name__)


# =============================================================================
# COLD START PERSONA PROFILES
# Reason: When user gives minimal info (e.g. "I'm a student"), we still need
#         a meaningful spend distribution. These profiles are archetypal
#         representations of Indian consumer segments.
# Type: heuristic assumption, extendable
# =============================================================================
COLD_START_PROFILES: Dict[str, Dict[str, float]] = {
    "student": {
        # Reason: students typically spend on food delivery, OTT, online shopping
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
        # Reason: travel-heavy persona → flights, hotels dominate
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
        # Reason: working professional: groceries, dining, utilities, some travel
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
        # Reason: household-centric: groceries and utilities dominate
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
        # Reason: business owners spend on travel, dining, varied categories
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

# Default cold start (no profession known)
DEFAULT_COLD_START = {
    # Reason: balanced urban consumer; no strong signals → uniform-ish distribution
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
# detect_intensity_from_context()
# Reason: Scans text NEAR the detected keyword (window of words) to find
#         intensity modifiers. Returns a float intensity score.
# =============================================================================
def detect_intensity_from_context(text: str, keyword: str) -> float:
    """
    Find intensity modifier near a keyword in the sentence.
    Example: "I use Swiggy a lot" → keyword="swiggy", modifier="a lot" → 1.0

    Returns float intensity from INTENSITY_MAP or DEFAULT_INTENSITY as fallback.
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()

    # Find position of keyword
    kw_pos = text_lower.find(keyword_lower)
    if kw_pos == -1:
        return DEFAULT_INTENSITY
        # Reason: keyword not in text; use moderate default (heuristic)

    # Extract a window of 30 chars before and after keyword
    # Reason: intensity words typically appear immediately before/after the brand
    window_start = max(0, kw_pos - 30)
    window_end   = min(len(text_lower), kw_pos + len(keyword_lower) + 30)
    window       = text_lower[window_start:window_end]

    logger.debug(f"  Intensity window for '{keyword}': '{window}'")

    # Check all intensity modifiers (longest first to avoid partial match issues)
    for modifier, score in sorted(INTENSITY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if modifier in window:
            logger.debug(f"  → Detected modifier '{modifier}' → intensity {score}")
            return score

    logger.debug(f"  → No modifier found → default intensity {DEFAULT_INTENSITY}")
    return DEFAULT_INTENSITY
    # Reason: no intensity word found; assume moderate usage


# =============================================================================
# detect_profession()
# Reason: Extracts profession from text or user_profile for spend estimation.
#         Checks user_profile dict first, then scans text for profession words.
# =============================================================================
def detect_profession(text: str, user_profile: dict | None) -> str | None:
    """
    Returns canonical profession key (from PROFESSION_ALIASES) or None.
    """
    # Priority 1: explicit user_profile field
    if user_profile and "profession" in user_profile:
        raw = str(user_profile["profession"]).lower().strip()
        normalized = PROFESSION_ALIASES.get(raw, None)
        if normalized:
            logger.debug(f"  Profession from user_profile: '{raw}' → '{normalized}'")
            return normalized

    # Priority 2: text scan
    text_lower = text.lower()
    for alias, canonical in sorted(PROFESSION_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in text_lower:
            logger.debug(f"  Profession from text: '{alias}' → '{canonical}'")
            return canonical

    return None  # Reason: no profession signal; handled by estimator fallback


# =============================================================================
# parse_user_text()
# THE MAIN PARSER — converts raw text into category intensity scores
# =============================================================================
def parse_user_text(
    user_text: str,
    user_profile: dict | None = None
) -> Dict:
    """
    Master parsing function.

    Returns:
    {
        "raw_signals": [(category, intensity), ...],
        "category_scores": {cat: score, ...},  # merged, intensity-weighted
        "profession": str | None,
        "keywords_found": [str],
        "is_cold_start": bool,
        "cold_start_persona": str | None,
    }
    """
    logger.info("=" * 60)
    logger.info("PARSER: Starting text parse")
    logger.info(f"Input text: '{user_text}'")
    logger.info(f"User profile: {user_profile}")

    # ── STEP 1: Extract all category signals from text ────────────────────────
    raw_signals = extract_category_signals(user_text)
    logger.info(f"PARSER Step 1 — Raw signals from mapper: {raw_signals}")

    # ── STEP 2: For each signal, detect intensity from context ────────────────
    intensity_weighted_signals: List[Tuple[str, float]] = []

    # Build keyword list for logging
    keywords_found = []

    for (category, _) in raw_signals:
        # Find which keyword triggered this category
        triggering_keyword = _find_triggering_keyword(user_text, category)
        if triggering_keyword:
            keywords_found.append(triggering_keyword)

        # Detect intensity for this category based on surrounding words
        intensity = detect_intensity_from_context(user_text, triggering_keyword or category)
        intensity_weighted_signals.append((category, intensity))
        logger.debug(f"  Category='{category}', keyword='{triggering_keyword}', intensity={intensity:.2f}")

    logger.info(f"PARSER Step 2 — Intensity-weighted signals: {intensity_weighted_signals}")

    # ── STEP 3: Merge signals per category (sum intensities) ──────────────────
    # Reason: same category may appear multiple times (e.g. "Swiggy and Zomato")
    #         We sum and will normalize later.
    category_scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    # IMPORTANT: Changing category names will break Simulation Agent

    for (category, intensity) in intensity_weighted_signals:
        category_scores[category] += intensity  # Reason: accumulate all signals for same category

    logger.info(f"PARSER Step 3 — Accumulated category scores: {category_scores}")

    # ── STEP 4: Cold start detection ──────────────────────────────────────────
    # Reason: if no signals found, we fall back to profession-based archetype
    total_signal = sum(category_scores.values())
    is_cold_start = total_signal < 0.1  # Reason: essentially zero signal detected

    profession = detect_profession(user_text, user_profile)

    cold_start_persona = None
    if is_cold_start:
        logger.warning("PARSER: Cold start detected — using persona fallback")
        cold_start_persona = profession
        # Load persona distribution
        persona_weights = _get_cold_start_weights(profession)
        for cat, weight in persona_weights.items():
            category_scores[cat] = weight
            # Reason: replace zeros with persona-based defaults
        logger.info(f"PARSER: Loaded cold start persona '{cold_start_persona}': {persona_weights}")

    # ── STEP 5: Apply WEIGHT_FLOOR to prevent zeros ───────────────────────────
    for cat in CATEGORIES:
        if category_scores[cat] < WEIGHT_FLOOR:
            category_scores[cat] = WEIGHT_FLOOR
            # Reason: avoids hard zero — every category should retain small residual

    logger.info(f"PARSER Step 5 — After floor: {category_scores}")

    return {
        "raw_signals":          raw_signals,
        "category_scores":      category_scores,  # raw (unnormalized)
        "profession":           profession,
        "keywords_found":       list(set(keywords_found)),  # deduplicated
        "is_cold_start":        is_cold_start,
        "cold_start_persona":   cold_start_persona,
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _find_triggering_keyword(text: str, category: str) -> str | None:
    """
    Reason: Needed to do intensity window lookup — we need to know WHICH
            word in the text triggered a category to look around it.
    """
    from mapper import BRAND_CATEGORY_MAP, LIFESTYLE_KEYWORD_MAP

    text_lower = text.lower()

    # Check brands first (more specific)
    for keyword, cat in sorted(BRAND_CATEGORY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if cat == category and keyword in text_lower:
            return keyword

    # Then lifestyle phrases
    for keyword, cat in sorted(LIFESTYLE_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if cat == category and keyword in text_lower:
            return keyword

    return None  # Reason: trigger not findable; intensity will use default


def _get_cold_start_weights(profession: str | None) -> Dict[str, float]:
    """
    Reason: Returns persona-appropriate spend weights for cold start.
            Falls back to DEFAULT_COLD_START if profession unknown.
    """
    if profession and profession in COLD_START_PROFILES:
        return COLD_START_PROFILES[profession]

    # Map profession aliases to profile keys
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
    # Reason: ultimate fallback; balanced distribution when no profession known
