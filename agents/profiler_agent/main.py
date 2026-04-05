# =============================================================================
# main.py — Behavioral Agent: Master Orchestrator
# =============================================================================
# Entry point for the Behavioral Agent.
# Exposes: generate_user_profile(user_text, user_profile) → structured dict
#
# FIXES APPLIED:
#   [Issue #6]  Invalid input handling → propagates input_valid flag
#   [Issue #7]  State persistence → all state is local, reset per call
#
# Full pipeline:
#   TEXT → PARSER → ESTIMATOR → WEIGHT CALC → SPEND VECTOR → OUTPUT
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# Output schema is FIXED — do not rename keys without updating Simulation Agent.
#
# Run:
#   python test_behavior.py          # CLI tests
#   streamlit run app.py             # Streamlit UI
# =============================================================================

import logging
from typing import Optional

from utils import (
    setup_logging,
    normalize_weights,
    compute_spend_vector,
    calculate_confidence,
    generate_tags,
)
from parser import parse_user_text
from estimator import estimate_total_spend
from defaults import CATEGORIES, DEFAULT_WEIGHTS, WEIGHT_FLOOR
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent

setup_logging("INFO")
logger = logging.getLogger(__name__)


# =============================================================================
# generate_user_profile()
# =============================================================================
def generate_user_profile(
    user_text: str,
    user_profile: Optional[dict] = None
) -> dict:
    """
    Converts natural language user input into a structured financial profile.

    Args:
        user_text    (str):  Raw user input. Required.
        user_profile (dict): Optional demographic context.
                             Keys: "age" (int), "income" (int), "profession" (str)

    Returns:
        {
            "spend_vector":      {category: int (₹/month)},
            "category_weights":  {category: float (0.0–1.0, sums to 1.0)},
            "tags":              [str],
            "confidence":        float (0.0–1.0),
        }

    Integration contract:
        All category keys in spend_vector and category_weights are FIXED.
        CATEGORIES = ["dining", "travel", "online_shopping", "fuel",
                      "groceries", "entertainment", "utilities", "other"]
        ⚠️ IMPORTANT: Changing category names will break Simulation Agent
    """
    logger.info("=" * 70)
    logger.info("BEHAVIORAL AGENT: generate_user_profile() called")
    logger.info(f"user_text    = '{user_text}'")
    logger.info(f"user_profile = {user_profile}")
    logger.info("=" * 70)

    # FIX [Issue #7]: All intermediate state is initialized fresh here.
    # There are no module-level globals that persist between calls.
    # Every variable below is scoped to this function invocation.

    # ── STAGE 1: PARSE TEXT ───────────────────────────────────────────────────
    logger.info("STAGE 1: Parsing user text...")
    parse_result = parse_user_text(user_text, user_profile)

    category_scores     = parse_result["category_scores"]
    profession          = parse_result["profession"]
    keywords_found      = parse_result["keywords_found"]
    is_cold_start       = parse_result["is_cold_start"]
    cold_start_persona  = parse_result["cold_start_persona"]
    input_valid         = parse_result["input_valid"]   # FIX [Issue #6]

    logger.info(f"STAGE 1 DONE: profession='{profession}', cold_start={is_cold_start}, valid={input_valid}")
    logger.info(f"  keywords_found:    {keywords_found}")
    logger.info(f"  category_scores:   {category_scores}")

    # FIX [Issue #6]: If input is completely invalid, return safe zero profile immediately.
    if not input_valid:
        logger.warning("BEHAVIORAL AGENT: Invalid input detected — returning empty profile.")
        return _empty_profile()

    # ── STAGE 2: ESTIMATE TOTAL SPEND ─────────────────────────────────────────
    logger.info("STAGE 2: Estimating total monthly spend...")
    spend_estimate      = estimate_total_spend(user_profile, profession)
    total_spend         = spend_estimate["total_spend"]
    estimation_method   = spend_estimate["estimation_method"]
    confidence_bonus    = spend_estimate["confidence_bonus"]

    logger.info(f"STAGE 2 DONE: total_spend=₹{total_spend}, method='{estimation_method}'")

    # ── STAGE 3: NORMALIZE WEIGHTS ────────────────────────────────────────────
    logger.info("STAGE 3: Normalizing category weights...")
    normalized_weights = normalize_weights(category_scores)
    logger.info(f"STAGE 3 DONE: weights = {normalized_weights}")

    # ── STAGE 4: COMPUTE SPEND VECTOR ─────────────────────────────────────────
    logger.info("STAGE 4: Computing spend vector (₹ per category)...")
    spend_vector = compute_spend_vector(normalized_weights, total_spend)
    logger.info(f"STAGE 4 DONE: spend_vector = {spend_vector}")

    # ── STAGE 5: CALCULATE CONFIDENCE ─────────────────────────────────────────
    logger.info("STAGE 5: Calculating confidence score...")
    confidence = calculate_confidence(
        keywords_found,
        is_cold_start,
        estimation_method,
        confidence_bonus,
        input_valid=input_valid,    # FIX [Issue #6, #9]
    )
    logger.info(f"STAGE 5 DONE: confidence = {confidence}")

    # ── STAGE 6: GENERATE TAGS ────────────────────────────────────────────────
    logger.info("STAGE 6: Generating tags...")
    tags = generate_tags(
        normalized_weights,
        profession,
        is_cold_start,
        cold_start_persona,
        keywords_found,
        input_valid=input_valid,    # FIX [Issue #6]
    )
    logger.info(f"STAGE 6 DONE: tags = {tags}")

    # ── FINAL OUTPUT ──────────────────────────────────────────────────────────
    # ⚠️ IMPORTANT: Changing category names will break Simulation Agent
    output = {
        "spend_vector":     spend_vector,
        "category_weights": normalized_weights,
        "tags":             tags,
        "confidence":       confidence,
    }

    logger.info("=" * 70)
    logger.info("BEHAVIORAL AGENT: Output generated successfully")
    logger.info(f"  spend_vector  : {output['spend_vector']}")
    logger.info(f"  weights       : {output['category_weights']}")
    logger.info(f"  tags          : {output['tags']}")
    logger.info(f"  confidence    : {output['confidence']}")
    logger.info("=" * 70)

    return output


# =============================================================================
# _empty_profile()
# FIX [Issue #6]: Guaranteed-safe zero-state return for invalid inputs.
#                 Fully resets — no contamination from any previous call.
# =============================================================================
def _empty_profile() -> dict:
    """
    Fallback output for invalid/empty input.
    Returns all-zero spend_vector and baseline default weights.
    confidence = 0.0 signals to downstream agents to skip/flag this profile.
    """
    return {
        # ⚠️ IMPORTANT: Changing category names will break Simulation Agent
        "spend_vector":     {cat: 0 for cat in CATEGORIES},
        "category_weights": {cat: DEFAULT_WEIGHTS.get(cat, WEIGHT_FLOOR) for cat in CATEGORIES},
        "tags":             ["invalid_input"],
        "confidence":       0.0,
    }
