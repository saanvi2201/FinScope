# =============================================================================
# estimator.py — Total Spend Estimator
# =============================================================================
# Converts user demographics (income / profession) into a total monthly
# discretionary spend figure used by the Simulation Agent.
#
# No changes required from original — this module was correct.
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# =============================================================================

import logging
from typing import Optional

from defaults import (
    INCOME_SPEND_RATIO,
    PROFESSION_SPEND_MAP,
    DEFAULT_TOTAL_SPEND,
    MIN_SPEND,
    MAX_SPEND,
    PROFESSION_ALIASES,
)

logger = logging.getLogger(__name__)


def estimate_total_spend(
    user_profile: Optional[dict],
    detected_profession: Optional[str]
) -> dict:
    """
    Estimates total monthly discretionary credit card spend.

    Priority order:
    1. Income from user_profile → income * INCOME_SPEND_RATIO
    2. Profession from user_profile or detected from text
    3. DEFAULT_TOTAL_SPEND fallback

    Returns:
    {
        "total_spend":        int,
        "estimation_method":  str,
        "confidence_bonus":   float
    }
    """
    logger.info("ESTIMATOR: Estimating total spend...")

    # ── METHOD 1: Income provided ─────────────────────────────────────────────
    if user_profile and "income" in user_profile:
        income = user_profile["income"]
        if isinstance(income, (int, float)) and income > 0:
            raw_spend = income * INCOME_SPEND_RATIO
            capped_spend = _apply_caps(raw_spend)
            logger.info(f"ESTIMATOR: Income={income} → raw={raw_spend:.0f} → capped={capped_spend}")
            return {
                "total_spend":       int(capped_spend),
                "estimation_method": "income_based",
                "confidence_bonus":  0.10,
            }
        else:
            logger.warning(f"ESTIMATOR: Invalid income value '{income}', ignoring.")

    # ── METHOD 2: Profession-based ────────────────────────────────────────────
    profession = None
    if user_profile and "profession" in user_profile:
        raw_prof = str(user_profile["profession"]).lower().strip()
        profession = PROFESSION_ALIASES.get(raw_prof, raw_prof)
        logger.info(f"ESTIMATOR: Profession from user_profile: '{profession}'")
    elif detected_profession:
        profession = detected_profession
        logger.info(f"ESTIMATOR: Profession from text detection: '{profession}'")

    if profession:
        canon = _normalize_profession(profession)
        if canon in PROFESSION_SPEND_MAP:
            raw_spend = PROFESSION_SPEND_MAP[canon]
            capped_spend = _apply_caps(raw_spend)
            logger.info(f"ESTIMATOR: Profession='{canon}' → spend={capped_spend}")
            return {
                "total_spend":       int(capped_spend),
                "estimation_method": f"profession_based:{canon}",
                "confidence_bonus":  0.05,
            }

    # ── METHOD 3: Default fallback ────────────────────────────────────────────
    logger.warning("ESTIMATOR: No income or profession signal — using default fallback")
    return {
        "total_spend":       DEFAULT_TOTAL_SPEND,
        "estimation_method": "default_fallback",
        "confidence_bonus":  0.0,
    }


def _apply_caps(spend: float) -> float:
    if spend < MIN_SPEND:
        return float(MIN_SPEND)
    if spend > MAX_SPEND:
        return float(MAX_SPEND)
    return spend


def _normalize_profession(profession: str) -> str:
    if profession in PROFESSION_SPEND_MAP:
        return profession
    fallback_map = {
        "engineer":     "salaried",
        "doctor":       "doctor",
        "professional": "professional",
        "freelancer":   "freelancer",
        "retired":      "retired",
        "homemaker":    "homemaker",
    }
    return fallback_map.get(profession, "salaried")
