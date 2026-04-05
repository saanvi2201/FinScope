# =============================================================================
# utils.py — Utility Functions for Behavioral Agent
# =============================================================================
# FIXES APPLIED:
#   [Issue #4]  Default leakage → normalize_weights only spreads across
#               categories that have actual scores (not zero ones).
#   [Issue #5]  Tag over-generation → stricter thresholds (0.40 / 0.20).
#   [Issue #9]  Confidence not accurate → tighter ladder with proper bands.
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# =============================================================================

import logging
from typing import Dict, List, Optional

from defaults import (
    CATEGORIES,
    WEIGHT_FLOOR,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    HIGH_CONFIDENCE_KEYWORD_THRESHOLD,
    MEDIUM_CONFIDENCE_KEYWORD_THRESHOLD,
    TAG_HEAVY_THRESHOLD,
    TAG_MODERATE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# normalize_weights()
# FIX [Issue #4]: ONLY categories with score > 0 participate in normalization.
#                 Zero-signal categories stay at 0.0 (no leakage).
#                 WEIGHT_FLOOR is only applied when ALL categories are zero
#                 (i.e. true cold-start — handled already in parser.py).
# =============================================================================
def normalize_weights(category_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalizes raw category scores to sum to 1.0.

    IMPORTANT BEHAVIOR CHANGE vs previous version:
    - Categories with score == 0.0 remain 0.0 after normalization.
    - Only non-zero categories are normalized.
    - This prevents unused categories from leaking weight.
    - If ALL scores are zero (shouldn't happen after parser), falls back
      to DEFAULT_WEIGHTS.

    Returns: normalized weight dict where non-zero values sum to ~1.0
    """
    logger.debug("UTILS: Normalizing weights...")

    total = sum(category_scores.values())

    if total == 0:
        # Absolute fallback — parser should have applied cold-start defaults
        from defaults import DEFAULT_WEIGHTS
        logger.warning("UTILS: All scores are zero — loading DEFAULT_WEIGHTS as fallback")
        return dict(DEFAULT_WEIGHTS)

    # FIX [Issue #4]: Divide by total, leave zeros as zeros
    normalized = {}
    for cat in CATEGORIES:
        # ⚠️ IMPORTANT: Changing category names will break Simulation Agent
        raw = category_scores.get(cat, 0.0)
        normalized[cat] = round(raw / total, 4)

    logger.debug(f"UTILS: Normalized weights: {normalized}")
    logger.debug(f"UTILS: Sum check = {sum(normalized.values()):.4f}")

    return normalized


# =============================================================================
# compute_spend_vector()
# Converts normalized weight × total_spend → actual ₹ per category.
# Unchanged from original — this is correct.
# =============================================================================
def compute_spend_vector(
    normalized_weights: Dict[str, float],
    total_spend: int
) -> Dict[str, int]:
    """
    Multiplies each weight by total_spend and rounds to nearest integer.
    Returns spend_vector with integer rupee values.
    Categories with 0.0 weight → ₹0 (correct — not mentioned = no spend).
    """
    logger.debug(f"UTILS: Computing spend vector with total_spend=₹{total_spend}")

    spend_vector = {}
    for cat in CATEGORIES:
        # ⚠️ IMPORTANT: Changing category names will break Simulation Agent
        weight = normalized_weights.get(cat, 0.0)
        spend_vector[cat] = int(round(weight * total_spend))

    logger.debug(f"UTILS: Spend vector: {spend_vector}")
    logger.debug(f"UTILS: Sum check = ₹{sum(spend_vector.values())}")
    return spend_vector


# =============================================================================
# calculate_confidence()
# FIX [Issue #9]: Confidence ladder is now tighter and more accurate.
#                 Also accounts for input validity flag.
# =============================================================================
def calculate_confidence(
    keywords_found: List[str],
    is_cold_start: bool,
    estimation_method: str,
    confidence_bonus: float,
    input_valid: bool = True,
) -> float:
    """
    Returns confidence score 0.0 → 1.0.

    Scoring:
    - Invalid input:            0.0  (hard zero)
    - Cold start (no signals):  0.35 + bonus
    - 1-2 keywords detected:    0.65 + bonus
    - 3+ keywords detected:     0.90 + bonus
    - Max capped at:            1.0

    FIX [Issue #9]: Previous version gave medium confidence even for complex
    inputs where it should be high. Now threshold-based.
    """
    if not input_valid:
        logger.warning("UTILS: Input invalid → confidence = 0.0")
        return 0.0

    n_keywords = len(keywords_found)
    logger.debug(f"UTILS: Confidence calc — keywords={n_keywords}, cold_start={is_cold_start}")

    if is_cold_start:
        base = CONFIDENCE_LOW    # 0.35 — no real signal
    elif n_keywords >= HIGH_CONFIDENCE_KEYWORD_THRESHOLD:
        base = CONFIDENCE_HIGH   # 0.90 — strong multi-signal
    elif n_keywords >= MEDIUM_CONFIDENCE_KEYWORD_THRESHOLD:
        base = CONFIDENCE_MEDIUM # 0.65 — partial signal
    else:
        base = CONFIDENCE_LOW    # 0.35 — parsed but minimal signal

    final = min(1.0, base + confidence_bonus)
    logger.info(f"UTILS: Confidence = {base:.2f} + {confidence_bonus:.2f} (bonus) = {final:.2f}")
    return round(final, 2)


# =============================================================================
# generate_tags()
# FIX [Issue #5]: Stricter thresholds — heavy ≥ 0.40, moderate ≥ 0.20.
#                 Previously: heavy ≥ 0.30, moderate ≥ 0.15 → too many tags.
#                 Goal tags also tightened.
# =============================================================================
def generate_tags(
    normalized_weights: Dict[str, float],
    profession: Optional[str],
    is_cold_start: bool,
    cold_start_persona: Optional[str],
    keywords_found: List[str],
    input_valid: bool = True,
) -> List[str]:
    """
    Generates descriptive string tags from the user profile.

    FIX [Issue #5]:
    - heavy_X_user:    weight ≥ 0.40 (was 0.30)
    - moderate_X_user: weight ≥ 0.20 (was 0.15)
    - Goal tags:       weight ≥ 0.30 (was 0.25)
    - Max 1 goal tag emitted (highest weight category wins)

    Returns list like: ["heavy_dining_user", "moderate_travel_user", "salaried"]
    """
    tags = []

    if not input_valid:
        return ["invalid_input"]

    # ── Spending pattern tags ─────────────────────────────────────────────────
    for cat in CATEGORIES:
        # ⚠️ IMPORTANT: Changing category names will break Simulation Agent
        weight = normalized_weights.get(cat, 0.0)

        if weight >= TAG_HEAVY_THRESHOLD:       # 0.40
            tags.append(f"heavy_{cat}_user")
        elif weight >= TAG_MODERATE_THRESHOLD:  # 0.20
            tags.append(f"moderate_{cat}_user")
        # Below 0.20 → not tagged (noise reduction)

    # ── Profession tag ────────────────────────────────────────────────────────
    if profession:
        tags.append(profession)

    # ── Cold start tag ────────────────────────────────────────────────────────
    if is_cold_start:
        tags.append("cold_start")
    if cold_start_persona:
        tags.append(f"persona:{cold_start_persona}")

    # ── Goal inference tags (ARAG-aligned) ────────────────────────────────────
    # FIX [Issue #5]: Only the single strongest goal is tagged (not all at once).
    #                 Threshold raised to 0.30 to reduce false goal assignments.
    w = normalized_weights
    goal_candidates = []

    if w.get("travel", 0) >= 0.30:
        goal_candidates.append(("goal:miles_or_lounge",    w["travel"]))
    if w.get("dining", 0) >= 0.30:
        goal_candidates.append(("goal:dining_rewards",      w["dining"]))
    if w.get("online_shopping", 0) >= 0.30:
        goal_candidates.append(("goal:cashback_ecommerce",  w["online_shopping"]))
    if w.get("fuel", 0) >= 0.20:
        goal_candidates.append(("goal:fuel_surcharge_waiver", w["fuel"]))
    if w.get("utilities", 0) >= 0.25:
        goal_candidates.append(("goal:bill_payment_rewards", w["utilities"]))

    if goal_candidates:
        # Emit only the strongest goal
        goal_candidates.sort(key=lambda x: x[1], reverse=True)
        tags.append(goal_candidates[0][0])

    # Dedup and sort
    tags = sorted(list(set(tags)))
    logger.info(f"UTILS: Generated tags: {tags}")
    return tags


# =============================================================================
# setup_logging()
# =============================================================================
def setup_logging(level: str = "INFO"):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s] %(name)s — %(message)s"
    )
    logger.info(f"Logging initialized at level: {level}")
