# =============================================================================
# estimator.py — Total Spend Estimator
# =============================================================================
# Converts user demographics (income / profession) into a total monthly
# discretionary spend figure used by the Simulation Agent.
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# =============================================================================

import logging
from typing import Optional

from defaults import (
    INCOME_SPEND_RATIO,     # 0.40 — % of income assumed as discretionary spend
    PROFESSION_SPEND_MAP,   # profession → default monthly spend
    DEFAULT_TOTAL_SPEND,    # neutral fallback
    MIN_SPEND,              # floor cap
    MAX_SPEND,              # ceiling cap
    PROFESSION_ALIASES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# estimate_total_spend()
# =============================================================================
def estimate_total_spend(
    user_profile: Optional[dict],
    detected_profession: Optional[str]
) -> dict:
    """
    Estimates total monthly discretionary credit card spend.

    Priority order (highest confidence first):
    1. Income from user_profile → income * INCOME_SPEND_RATIO
    2. Profession from user_profile or detected from text
    3. DEFAULT_TOTAL_SPEND fallback

    Returns:
    {
        "total_spend": int,
        "estimation_method": str,  # for logging/debug
        "confidence_bonus": float  # adds to overall confidence if income known
    }
    """
    logger.info("ESTIMATOR: Estimating total spend...")

    # ── METHOD 1: Income provided ─────────────────────────────────────────────
    if user_profile and "income" in user_profile:
        income = user_profile["income"]

        # Validate income is a reasonable number
        if isinstance(income, (int, float)) and income > 0:
            raw_spend = income * INCOME_SPEND_RATIO
            # Reason: Assumes ~40% of monthly income goes to discretionary spend
            #         (common financial heuristic; this excludes rent, EMI, savings)

            capped_spend = _apply_caps(raw_spend)
            logger.info(
                f"ESTIMATOR: Income={income} → raw_spend={raw_spend:.0f} → capped={capped_spend}"
            )
            return {
                "total_spend":        int(capped_spend),
                "estimation_method":  "income_based",
                "confidence_bonus":   0.10,
                # Reason: income provided → highest quality signal → boost confidence by 10%
            }
        else:
            logger.warning(f"ESTIMATOR: Invalid income value '{income}', ignoring.")

    # ── METHOD 2: Profession-based ────────────────────────────────────────────
    # Check user_profile profession first, then detected from text
    profession = None

    if user_profile and "profession" in user_profile:
        raw_prof = str(user_profile["profession"]).lower().strip()
        profession = PROFESSION_ALIASES.get(raw_prof, raw_prof)
        logger.info(f"ESTIMATOR: Profession from user_profile: '{profession}'")
    elif detected_profession:
        profession = detected_profession
        logger.info(f"ESTIMATOR: Profession from text detection: '{profession}'")

    if profession:
        # Normalize profession key for PROFESSION_SPEND_MAP
        canon = _normalize_profession(profession)
        if canon in PROFESSION_SPEND_MAP:
            raw_spend = PROFESSION_SPEND_MAP[canon]
            capped_spend = _apply_caps(raw_spend)
            logger.info(
                f"ESTIMATOR: Profession='{canon}' → spend={capped_spend}"
            )
            return {
                "total_spend":        int(capped_spend),
                "estimation_method":  f"profession_based:{canon}",
                "confidence_bonus":   0.05,
                # Reason: profession provides moderate signal → small confidence bonus
            }

    # ── METHOD 3: Default fallback ────────────────────────────────────────────
    logger.warning("ESTIMATOR: No income or profession signal — using default fallback")
    return {
        "total_spend":        DEFAULT_TOTAL_SPEND,
        "estimation_method":  "default_fallback",
        "confidence_bonus":   0.0,
        # Reason: no demographic info available; no confidence bonus
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _apply_caps(spend: float) -> float:
    """
    Apply MIN_SPEND and MAX_SPEND bounds.
    Reason: Prevents unrealistic outputs.
            Below MIN_SPEND → card rewards are negligible.
            Above MAX_SPEND → model accuracy degrades.
    """
    if spend < MIN_SPEND:
        logger.debug(f"ESTIMATOR: Spend {spend:.0f} below MIN={MIN_SPEND}, capping up.")
        return float(MIN_SPEND)

    if spend > MAX_SPEND:
        logger.debug(f"ESTIMATOR: Spend {spend:.0f} above MAX={MAX_SPEND}, capping down.")
        return float(MAX_SPEND)

    return spend


def _normalize_profession(profession: str) -> str:
    """
    Reason: Maps detected profession string to a key in PROFESSION_SPEND_MAP.
            Some professions are aliased (e.g. "engineer" → "salaried" if not found).
    """
    if profession in PROFESSION_SPEND_MAP:
        return profession

    # Fallback aliases for unmapped professions
    fallback_map = {
        "engineer":     "salaried",
        "doctor":       "doctor",
        "professional": "professional",
        "freelancer":   "freelancer",
        "retired":      "retired",
        "homemaker":    "homemaker",
    }
    # Reason: profession keys in PROFESSION_SPEND_MAP may differ from parsed strings
    return fallback_map.get(profession, "salaried")
    # Reason: unknown profession treated as salaried (most common category)
