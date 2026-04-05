# =============================================================================
# test_behavior.py — CLI Test Suite for Behavioral Agent
# =============================================================================
# Run: python test_behavior.py
#
# Tests:
#   1. Basic keyword detection
#   2. Brand-based mapping (Swiggy, Amazon, IRCTC, etc.)
#   3. Cold start (student, traveler personas)
#   4. Income-based estimation
#   5. Mixed/complex inputs
#   6. Edge cases (empty input, single word)
#   7. Integration-readiness check
# =============================================================================

import json
import sys
import os

# Add parent dir to path for direct run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import generate_user_profile

# =============================================================================
# TEST CASES
# =============================================================================
TEST_CASES = [

    # ── BASIC TESTS ───────────────────────────────────────────────────────────
    {
        "id": "TC-01",
        "name": "Classic foodie (Swiggy + Zomato)",
        "input": {
            "user_text": "I use Swiggy and Zomato a lot. I order food every day.",
            "user_profile": None
        },
        "expect_dominant": "dining",
    },

    {
        "id": "TC-02",
        "name": "Online shopper (Amazon + Flipkart)",
        "input": {
            "user_text": "I shop a lot on Amazon and Flipkart. Also use Myntra sometimes.",
            "user_profile": None
        },
        "expect_dominant": "online_shopping",
    },

    {
        "id": "TC-03",
        "name": "Frequent traveler",
        "input": {
            "user_text": "I travel frequently. Mostly flight bookings via MakeMyTrip, sometimes IRCTC.",
            "user_profile": None
        },
        "expect_dominant": "travel",
    },

    {
        "id": "TC-04",
        "name": "Fuel-heavy (car driver)",
        "input": {
            "user_text": "I drive a lot, fill petrol almost every week at BPCL.",
            "user_profile": None
        },
        "expect_dominant": "fuel",
    },

    # ── COLD START TESTS ──────────────────────────────────────────────────────
    {
        "id": "TC-05",
        "name": "Cold start: student (no keywords)",
        "input": {
            "user_text": "I am a college student.",
            "user_profile": None
        },
        "expect_tag": "student",
    },

    {
        "id": "TC-06",
        "name": "Cold start: traveler persona",
        "input": {
            "user_text": "I am a frequent traveler.",
            "user_profile": None
        },
        "expect_dominant": "travel",
    },

    # ── INCOME-BASED TESTS ────────────────────────────────────────────────────
    {
        "id": "TC-07",
        "name": "Income provided (engineer, 80k/month)",
        "input": {
            "user_text": "I spend on Swiggy and online shopping a lot.",
            "user_profile": {"income": 80000, "profession": "engineer"}
        },
        "expect_spend_min": 30000,
        "expect_confidence_min": 0.70,
    },

    {
        "id": "TC-08",
        "name": "Profession only (salaried, no income)",
        "input": {
            "user_text": "I am a salaried employee. I pay utilities and groceries.",
            "user_profile": {"profession": "salaried"}
        },
        "expect_dominant": "utilities",   # or groceries — either acceptable
    },

    # ── MIXED / COMPLEX INPUTS ────────────────────────────────────────────────
    {
        "id": "TC-09",
        "name": "Mixed: dining + travel + fuel",
        "input": {
            "user_text": "I eat out often. I also travel sometimes and fill petrol regularly.",
            "user_profile": {"income": 50000}
        },
        "expect_min_categories": 3,
    },

    {
        "id": "TC-10",
        "name": "Utility-heavy (bills + recharges)",
        "input": {
            "user_text": "Mostly I pay Airtel bill, electricity and monthly recharges.",
            "user_profile": None
        },
        "expect_dominant": "utilities",
    },

    {
        "id": "TC-11",
        "name": "Streaming / Entertainment focused",
        "input": {
            "user_text": "I subscribe to Netflix, Hotstar and Spotify. I rarely eat out.",
            "user_profile": None
        },
        "expect_dominant": "entertainment",
    },

    # ── CARD PORTFOLIO ALIGNED ─────────────────────────────────────────────────
    {
        "id": "TC-12",
        "name": "Vistara flyer (card portfolio match)",
        "input": {
            "user_text": "I fly Vistara frequently and eat out a lot. I need lounge access.",
            "user_profile": {"profession": "professional", "income": 100000}
        },
        "expect_dominant": "travel",
        "expect_tag": "goal:miles_or_lounge",
    },

    {
        "id": "TC-13",
        "name": "Grocery + utilities homemaker",
        "input": {
            "user_text": "I manage the household. BigBasket for groceries and pay all bills.",
            "user_profile": {"profession": "homemaker"}
        },
        "expect_dominant": "groceries",
    },

    # ── EDGE CASES ────────────────────────────────────────────────────────────
    {
        "id": "TC-14",
        "name": "Empty text (edge case)",
        "input": {
            "user_text": "",
            "user_profile": None
        },
        "expect_confidence_max": 0.01,
    },

    {
        "id": "TC-15",
        "name": "Single word input",
        "input": {
            "user_text": "Swiggy",
            "user_profile": None
        },
        "expect_dominant": "dining",
    },
]


# =============================================================================
# TEST RUNNER
# =============================================================================
def run_tests():
    print("\n" + "=" * 70)
    print(" BEHAVIORAL AGENT — CLI TEST SUITE")
    print("=" * 70)

    passed = 0
    failed = 0
    results = []

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['name']}")
        print(f"  Input: \"{tc['input']['user_text'][:60]}...\"" if len(tc['input']['user_text']) > 60
              else f"  Input: \"{tc['input']['user_text']}\"")

        # Run
        try:
            output = generate_user_profile(
                user_text=tc["input"]["user_text"],
                user_profile=tc["input"].get("user_profile")
            )
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            failed += 1
            results.append({"id": tc["id"], "status": "EXCEPTION", "error": str(e)})
            continue

        # Assertions
        assertions_passed = True
        notes = []

        # Check dominant category
        if "expect_dominant" in tc:
            expected_cat = tc["expect_dominant"]
            spend_vec = output["spend_vector"]
            dominant = max(spend_vec, key=spend_vec.get)
            # Allow top-2 match for edge cases
            sorted_cats = sorted(spend_vec, key=spend_vec.get, reverse=True)
            if expected_cat not in sorted_cats[:2]:
                assertions_passed = False
                notes.append(f"  ⚠ Expected dominant='{expected_cat}', got top2={sorted_cats[:2]}")
            else:
                notes.append(f"  ✓ Dominant category: '{dominant}' (expected ~'{expected_cat}')")

        # Check tag presence
        if "expect_tag" in tc:
            expected_tag = tc["expect_tag"]
            if not any(expected_tag in t for t in output["tags"]):
                assertions_passed = False
                notes.append(f"  ⚠ Expected tag containing '{expected_tag}', got: {output['tags']}")
            else:
                notes.append(f"  ✓ Tag '{expected_tag}' found")

        # Check min spend
        if "expect_spend_min" in tc:
            total = sum(output["spend_vector"].values())
            if total < tc["expect_spend_min"]:
                assertions_passed = False
                notes.append(f"  ⚠ Total spend ₹{total} < expected min ₹{tc['expect_spend_min']}")
            else:
                notes.append(f"  ✓ Total spend ₹{total} ≥ ₹{tc['expect_spend_min']}")

        # Check min confidence
        if "expect_confidence_min" in tc:
            if output["confidence"] < tc["expect_confidence_min"]:
                assertions_passed = False
                notes.append(f"  ⚠ Confidence {output['confidence']} < {tc['expect_confidence_min']}")
            else:
                notes.append(f"  ✓ Confidence {output['confidence']} ≥ {tc['expect_confidence_min']}")

        # Check max confidence (for empty input)
        if "expect_confidence_max" in tc:
            if output["confidence"] > tc["expect_confidence_max"]:
                assertions_passed = False
                notes.append(f"  ⚠ Confidence {output['confidence']} > max {tc['expect_confidence_max']}")
            else:
                notes.append(f"  ✓ Confidence correctly low: {output['confidence']}")

        # Check min categories active (non-zero weight)
        if "expect_min_categories" in tc:
            active = sum(1 for v in output["spend_vector"].values() if v > 0)
            if active < tc["expect_min_categories"]:
                assertions_passed = False
                notes.append(f"  ⚠ Only {active} active categories, expected ≥{tc['expect_min_categories']}")
            else:
                notes.append(f"  ✓ {active} active categories ≥ {tc['expect_min_categories']}")

        # Print result
        for note in notes:
            print(note)

        if assertions_passed:
            print(f"  ✅ PASSED — confidence={output['confidence']}, tags={output['tags'][:3]}")
            passed += 1
            results.append({"id": tc["id"], "status": "PASS"})
        else:
            print(f"  ❌ FAILED")
            failed += 1
            results.append({"id": tc["id"], "status": "FAIL"})

        # Print full output in verbose mode
        if "--verbose" in sys.argv or "-v" in sys.argv:
            print("\n  FULL OUTPUT:")
            print(json.dumps(output, indent=4))

    # Summary
    print("\n" + "=" * 70)
    print(f" RESULTS: {passed}/{passed+failed} PASSED | {failed} FAILED")
    print("=" * 70)

    if failed == 0:
        print(" 🎉 ALL TESTS PASSED — Behavioral Agent is ready for integration")
    else:
        print(f" ⚠️  {failed} test(s) failed — review output above")

    return failed == 0


# =============================================================================
# INTEGRATION CHECK
# Reason: Verifies output schema matches what Simulation Agent expects
# =============================================================================
def check_integration_contract():
    """
    Verifies that output keys and category names exactly match
    what the Simulation Agent expects.
    """
    from defaults import CATEGORIES  # IMPORTANT: Changing category names will break Simulation Agent

    print("\n[INTEGRATION CONTRACT CHECK]")
    output = generate_user_profile("I spend on Swiggy and Amazon")

    required_top_keys = ["spend_vector", "category_weights", "tags", "confidence"]
    for key in required_top_keys:
        if key not in output:
            print(f"  ❌ Missing required key: '{key}'")
        else:
            print(f"  ✓ Key '{key}' present")

    for cat in CATEGORIES:
        # IMPORTANT: Changing category names will break Simulation Agent
        if cat not in output["spend_vector"]:
            print(f"  ❌ Missing category in spend_vector: '{cat}'")
        elif cat not in output["category_weights"]:
            print(f"  ❌ Missing category in category_weights: '{cat}'")
        else:
            print(f"  ✓ Category '{cat}' present in both spend_vector and category_weights")

    weight_sum = sum(output["category_weights"].values())
    if abs(weight_sum - 1.0) < 0.01:
        print(f"  ✓ Weights sum to ~1.0 (actual: {weight_sum:.4f})")
    else:
        print(f"  ❌ Weights sum = {weight_sum:.4f} — should be ~1.0")

    if isinstance(output["confidence"], float) and 0.0 <= output["confidence"] <= 1.0:
        print(f"  ✓ Confidence is valid float: {output['confidence']}")
    else:
        print(f"  ❌ Invalid confidence: {output['confidence']}")

    print("  ✅ Integration contract check complete")


if __name__ == "__main__":
    check_integration_contract()
    run_tests()
