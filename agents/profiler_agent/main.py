# =============================================================================
# main.py — Behavioral Agent: Master Orchestrator
# =============================================================================
# Entry point for the Behavioral Agent.
# Exposes: generate_user_profile(user_text, user_profile) → structured dict
#
# Full pipeline:
# TEXT → PARSER → ESTIMATOR → WEIGHT CALC → SPEND VECTOR → OUTPUT
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# Output schema is FIXED — do not rename keys without updating Simulation Agent.
#
# Run instructions:
#   pip install streamlit
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
from defaults import CATEGORIES  # IMPORTANT: Changing category names will break Simulation Agent

# Initialize logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


# =============================================================================
# generate_user_profile()
# THE MAIN FUNCTION — called by Agent 2 (Knowledge) and Agent 3 (Simulation)
# =============================================================================
def generate_user_profile(
    user_text: str,
    user_profile: Optional[dict] = None
) -> dict:
    """
    Converts natural language user input into a structured financial profile.

    Args:
        user_text    (str):  Raw user input. REQUIRED.
        user_profile (dict): Optional demographic context.
                             Keys: "age" (int), "income" (int), "profession" (str)

    Returns:
        Strictly typed dict with keys:
        - spend_vector      : {category: int (₹/month)}
        - category_weights  : {category: float (0.0–1.0, sums to 1.0)}
        - tags              : [str]
        - confidence        : float (0.0–1.0)

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

    # Validate input
    if not user_text or not isinstance(user_text, str):
        logger.error("BEHAVIORAL AGENT: user_text is empty or invalid!")
        return _empty_profile()

    # ── STAGE 1: PARSE TEXT ───────────────────────────────────────────────────
    logger.info("STAGE 1: Parsing user text...")
    parse_result = parse_user_text(user_text, user_profile)

    category_scores     = parse_result["category_scores"]
    profession          = parse_result["profession"]
    keywords_found      = parse_result["keywords_found"]
    is_cold_start       = parse_result["is_cold_start"]
    cold_start_persona  = parse_result["cold_start_persona"]

    logger.info(f"STAGE 1 DONE: profession='{profession}', cold_start={is_cold_start}")
    logger.info(f"  keywords_found: {keywords_found}")
    logger.info(f"  raw category_scores: {category_scores}")

    # ── STAGE 2: ESTIMATE TOTAL SPEND ─────────────────────────────────────────
    logger.info("STAGE 2: Estimating total monthly spend...")
    spend_estimate = estimate_total_spend(user_profile, profession)

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
        confidence_bonus
    )
    logger.info(f"STAGE 5 DONE: confidence = {confidence}")

    # ── STAGE 6: GENERATE TAGS ────────────────────────────────────────────────
    logger.info("STAGE 6: Generating tags...")
    tags = generate_tags(
        normalized_weights,
        profession,
        is_cold_start,
        cold_start_persona,
        keywords_found
    )
    logger.info(f"STAGE 6 DONE: tags = {tags}")

    # ── FINAL OUTPUT ──────────────────────────────────────────────────────────
    # IMPORTANT: Changing category names will break Simulation Agent
    output = {
        "spend_vector": spend_vector,
        # Format: {category: int} — monthly ₹ spend per category
        # Consumed directly by Simulation Agent for Net Value calculation

        "category_weights": normalized_weights,
        # Format: {category: float} — proportional weights summing to 1.0
        # Used for relative importance scoring

        "tags": tags,
        # Format: [str] — interpretable labels for reasoning
        # Used by Recommendation Agent for explanation generation

        "confidence": confidence,
        # Format: float 0.0–1.0
        # Used by Validation stage to flag low-confidence profiles
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
# Reason: Returns safe zero-value output on invalid input.
#         Prevents downstream agents from crashing on bad input.
# =============================================================================
def _empty_profile() -> dict:
    """
    Fallback output for empty/invalid input.
    All zeros with minimum WEIGHT_FLOOR applied.
    """
    from defaults import DEFAULT_WEIGHTS, WEIGHT_FLOOR

    return {
        "spend_vector":     {cat: 0 for cat in CATEGORIES},
        "category_weights": {cat: DEFAULT_WEIGHTS.get(cat, WEIGHT_FLOOR) for cat in CATEGORIES},
        "tags":             ["invalid_input"],
        "confidence":       0.0,
        # Reason: 0.0 confidence signals to downstream agents to skip/flag this profile
    }
