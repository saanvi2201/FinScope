# =============================================================================
# utils.py — Utility Functions for Behavioral Agent
# =============================================================================
# Provides:
#   - Weight normalization
#   - Confidence score calculation
#   - Tag generation
#   - Spend vector rounding
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# =============================================================================

import logging
from typing import Dict, List

from defaults import (
    CATEGORIES,                             # IMPORTANT: Changing category names will break Simulation Agent
    WEIGHT_FLOOR,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    HIGH_CONFIDENCE_KEYWORD_THRESHOLD,
    MEDIUM_CONFIDENCE_KEYWORD_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# normalize_weights()
# Reason: category_scores are raw accumulated values; they must sum to 1.0
#         before being used as proportional spend weights.
# =============================================================================
def normalize_weights(category_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalizes raw category scores to sum to 1.0.
    Applies WEIGHT_FLOOR first to ensure no category is zero.

    Returns: normalized weight dict where all values sum to ~1.0
    """
    logger.debug("UTILS: Normalizing weights...")

    # Apply floor
    floored = {
        cat: max(category_scores.get(cat, 0.0), WEIGHT_FLOOR)
        for cat in CATEGORIES
        # Reason: every category gets at least WEIGHT_FLOOR to stay alive
    }

    total = sum(floored.values())

    if total == 0:
        # Absolute fallback — should never happen after flooring, but just in case
        from defaults import DEFAULT_WEIGHTS
        logger.warning("UTILS: All weights zero after flooring — loading DEFAULT_WEIGHTS")
        return dict(DEFAULT_WEIGHTS)

    normalized = {
        cat: round(floored[cat] / total, 4)
        for cat in CATEGORIES
    }
    # Reason: divide each value by total → ensures sum = 1.0
    #         round to 4 decimal places for clean output

    logger.debug(f"UTILS: Normalized weights: {normalized}")
    logger.debug(f"UTILS: Sum check = {sum(normalized.values()):.4f}")

    return normalized


# =============================================================================
# compute_spend_vector()
# Reason: Converts normalized weight × total_spend → actual ₹ per category.
#         This is the direct input format for the Simulation Agent.
# =============================================================================
def compute_spend_vector(
    normalized_weights: Dict[str, float],
    total_spend: int
) -> Dict[str, int]:
    """
    Multiplies each weight by total_spend and rounds to nearest integer.
    Returns spend_vector with integer rupee values.
    """
    logger.debug(f"UTILS: Computing spend vector with total_spend=₹{total_spend}")

    spend_vector = {}
    for cat in CATEGORIES:
        # IMPORTANT: Changing category names will break Simulation Agent
        weight = normalized_weights.get(cat, 0.0)
        spend = int(round(weight * total_spend))
        # Reason: weight × total gives proportional rupee allocation;
        #         integer rounded because card rewards work on whole numbers
        spend_vector[cat] = spend

    logger.debug(f"UTILS: Spend vector: {spend_vector}")
    logger.debug(f"UTILS: Sum check = ₹{sum(spend_vector.values())}")

    return spend_vector


# =============================================================================
# calculate_confidence()
# Reason: Reflects quality of signal extracted from user text.
#         More keywords → more confident the profile is accurate.
# =============================================================================
def calculate_confidence(
    keywords_found: List[str],
    is_cold_start: bool,
    estimation_method: str,
    confidence_bonus: float
) -> float:
    """
    Returns a confidence score between 0.0 and 1.0.

    Factors:
    1. Number of distinct keywords found
    2. Whether cold start was used
    3. Whether income/profession data boosted confidence

    Scoring ladder:
    - HIGH   = 0.90  (≥3 keywords detected)
    - MEDIUM = 0.65  (1-2 keywords detected)
    - LOW    = 0.40  (cold start / no signals)
    + confidence_bonus from estimator (0.0 / 0.05 / 0.10)
    """
    n_keywords = len(keywords_found)
    logger.debug(f"UTILS: Confidence calc — keywords={n_keywords}, cold_start={is_cold_start}")

    # Base confidence from keyword count
    if is_cold_start:
        base = CONFIDENCE_LOW
        # Reason: cold start = no real signal, profile is mostly assumed
    elif n_keywords >= HIGH_CONFIDENCE_KEYWORD_THRESHOLD:
        base = CONFIDENCE_HIGH
        # Reason: 3+ keywords → strong signal from user → high trust
    elif n_keywords >= MEDIUM_CONFIDENCE_KEYWORD_THRESHOLD:
        base = CONFIDENCE_MEDIUM
        # Reason: 1-2 keywords → partial signal → moderate trust
    else:
        base = CONFIDENCE_LOW
        # Reason: no keywords despite not cold_start → something parsed but weakly

    # Add bonus from demographic data
    final = min(1.0, base + confidence_bonus)
    # Reason: cap at 1.0 — confidence cannot exceed 100%

    logger.info(f"UTILS: Confidence = {base:.2f} (base) + {confidence_bonus:.2f} (bonus) = {final:.2f}")
    return round(final, 2)


# =============================================================================
# generate_tags()
# Reason: Human-readable labels that let the Simulation/Recommendation Agent
#         understand the user archetype without re-parsing. Adds interpretability.
# =============================================================================
def generate_tags(
    normalized_weights: Dict[str, float],
    profession: str | None,
    is_cold_start: bool,
    cold_start_persona: str | None,
    keywords_found: List[str]
) -> List[str]:
    """
    Generates descriptive string tags from the user profile.

    Returns list like:
    ["heavy_diner", "online_shopper", "student", "low_fuel_user"]
    """
    tags = []

    # ── Spending pattern tags (based on weight thresholds) ────────────────────
    for cat in CATEGORIES:
        # IMPORTANT: Changing category names will break Simulation Agent
        weight = normalized_weights.get(cat, 0.0)

        if weight >= 0.30:
            tags.append(f"heavy_{cat}_user")
            # Reason: ≥30% of spend → dominant category → "heavy" label (heuristic)
        elif weight >= 0.15:
            tags.append(f"moderate_{cat}_user")
            # Reason: 15-30% → meaningful but not dominant (heuristic)
        # Below 15%: not worth tagging, noise

    # ── Profession tag ────────────────────────────────────────────────────────
    if profession:
        tags.append(profession)
        # Reason: profession is a key segmentation variable for card matching

    # ── Cold start tag ────────────────────────────────────────────────────────
    if is_cold_start:
        tags.append("cold_start")
        # Reason: signals to downstream agents that profile is assumed, not observed

    if cold_start_persona:
        tags.append(f"persona:{cold_start_persona}")
        # Reason: which persona profile was applied → traceability

    # ── Goal inference tags (ARAG-aligned) ───────────────────────────────────
    # Reason: infer likely card goals from spend pattern for agent reasoning
    w = normalized_weights
    if w.get("travel", 0) >= 0.25:
        tags.append("goal:miles_or_lounge")
        # Reason: high travel → likely wants miles/lounge benefits
    if w.get("dining", 0) >= 0.25:
        tags.append("goal:dining_rewards")
    if w.get("online_shopping", 0) >= 0.25:
        tags.append("goal:cashback_ecommerce")
    if w.get("fuel", 0) >= 0.15:
        tags.append("goal:fuel_surcharge_waiver")
    if w.get("utilities", 0) >= 0.20:
        tags.append("goal:bill_payment_rewards")

    # Dedup and sort
    tags = sorted(list(set(tags)))
    logger.info(f"UTILS: Generated tags: {tags}")
    return tags


# =============================================================================
# setup_logging()
# Reason: Centralizes logging config; called once from main.py
# =============================================================================
def setup_logging(level: str = "INFO"):
    """
    Sets up logging for the entire behavior_agent module.
    Level options: DEBUG, INFO, WARNING, ERROR
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s] %(name)s — %(message)s"
        # Reason: consistent format makes multi-module logs easy to parse
    )
    logger.info(f"Logging initialized at level: {level}")
