# =============================================================================
# defaults.py — Central Configuration for Behavioral Agent
# =============================================================================
# ⚠️ INTEGRATION NOTE: These constants are tunable.
#    Changing CATEGORIES or CATEGORY keys WILL break the Simulation Agent.
#    Only change numeric values, not string keys.
# =============================================================================

# -----------------------------------------------------------------------------
# CATEGORIES (LOCKED — DO NOT RENAME)
# IMPORTANT: Changing category names will break Simulation Agent
# -----------------------------------------------------------------------------
CATEGORIES = [
    "dining",
    "travel",
    "online_shopping",
    "fuel",
    "groceries",
    "entertainment",
    "utilities",
    "other"
]

# -----------------------------------------------------------------------------
# DEFAULT SPEND DISTRIBUTION
# Reason: Balanced fallback distribution when user input is insufficient.
#         Derived from general Indian urban consumer spending patterns.
#         dining + online_shopping are heavier as digital spends dominate.
#         Type: heuristic assumption / empirically chosen approximation
# -----------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "dining":           0.20,   # Reason: Significant urban spend category (heuristic)
    "online_shopping":  0.20,   # Reason: Growing e-commerce usage in India (heuristic)
    "groceries":        0.20,   # Reason: Essential monthly spend, consistent (heuristic)
    "travel":           0.15,   # Reason: Moderate; not everyone travels frequently (heuristic)
    "utilities":        0.10,   # Reason: Recurring but lower reward-eligible spend (heuristic)
    "entertainment":    0.05,   # Reason: Discretionary, lower baseline (heuristic)
    "fuel":             0.05,   # Reason: Not universal; many use public transport (heuristic)
    "other":            0.05,   # Reason: Catch-all for uncategorized spend (fallback)
    # Total = 1.0 ✅
}

# -----------------------------------------------------------------------------
# DEFAULT INTENSITY SCORE
# Reason: Used as moderate usage baseline when user does NOT specify frequency.
#         Scale: 0.0 (never) → 1.0 (very frequently/daily).
#         0.6 represents "uses it, but not obsessively" (heuristic assumption).
# Type: heuristic assumption
# -----------------------------------------------------------------------------
DEFAULT_INTENSITY = 0.6

# -----------------------------------------------------------------------------
# INTENSITY MODIFIERS (keyword → intensity multiplier)
# Reason: Maps linguistic signals to quantitative usage intensity.
#         "a lot" / "heavy" → user is clearly a power user → 1.0
#         "sometimes" / "occasionally" → moderate use → 0.5
#         "rarely" / "almost never" → low priority → 0.2
# Type: heuristic mapping based on linguistic frequency cues
# -----------------------------------------------------------------------------
INTENSITY_MAP = {
    # HIGH usage signals
    "a lot":        1.0,    # Reason: explicit high-frequency signal
    "heavily":      1.0,    # Reason: strong usage signal
    "heavy":        1.0,
    "always":       1.0,
    "daily":        1.0,    # Reason: highest frequency possible
    "every day":    1.0,
    "mostly":       0.9,
    "mainly":       0.9,
    "primarily":    0.9,
    "frequently":   0.8,    # Reason: high but not absolute
    "often":        0.8,
    "regularly":    0.75,
    # MEDIUM usage signals
    "sometimes":    0.5,    # Reason: explicit moderate signal
    "occasionally": 0.4,
    "moderate":     0.5,
    "moderately":   0.5,
    # LOW usage signals
    "rarely":       0.2,    # Reason: explicit low signal
    "seldom":       0.2,
    "hardly":       0.15,
    "almost never": 0.1,
    "never":        0.0,    # Reason: no spend in this category
}

# -----------------------------------------------------------------------------
# TOTAL SPEND ESTIMATION BY PROFESSION
# Reason: Approximates monthly discretionary spend when income is not given.
# Type: heuristic assumption based on typical Indian urban spending
# -----------------------------------------------------------------------------
PROFESSION_SPEND_MAP = {
    "student":      20000,  # Reason: limited income, lower spend baseline (heuristic)
    "freelancer":   30000,  # Reason: variable income, moderate assumption (heuristic)
    "salaried":     40000,  # Reason: stable income → higher spend capacity (heuristic)
    "working":      40000,  # Reason: alias for salaried (heuristic)
    "professional": 50000,  # Reason: higher-paying professions → more spend (heuristic)
    "doctor":       60000,  # Reason: high earner (heuristic)
    "engineer":     45000,
    "business":     60000,  # Reason: business owners often have higher spend (heuristic)
    "retired":      25000,  # Reason: fixed income, moderate spend (heuristic)
    "homemaker":    25000,  # Reason: managed household budget (heuristic)
}

# Default when no profession given
DEFAULT_TOTAL_SPEND = 30000
# Reason: neutral middle-ground fallback for unknown user type (heuristic)

# -----------------------------------------------------------------------------
# INCOME-BASED SPEND RATIO
# Reason: Standard financial heuristic — roughly 40% of income is
#         discretionary (non-EMI, non-rent spend on credit cards).
#         Used when income is explicitly provided.
# Type: common financial heuristic (approx. 40% rule)
# -----------------------------------------------------------------------------
INCOME_SPEND_RATIO = 0.40

# -----------------------------------------------------------------------------
# SPEND CAPS (min / max sanity bounds)
# Reason: Prevents unrealistic outputs from extreme inputs.
#         Min 10,000 = minimum viable spend for card rewards to matter.
#         Max 100,000 = upper cap beyond which spend bucketing becomes inaccurate.
# Type: empirically chosen approximation
# -----------------------------------------------------------------------------
MIN_SPEND = 10_000   # Reason: Below this, no card gives meaningful rewards
MAX_SPEND = 100_000  # Reason: Above this, model accuracy degrades (heuristic ceiling)

# -----------------------------------------------------------------------------
# CONFIDENCE THRESHOLDS
# Reason: Reflects quality of signal extracted from user text.
# Type: heuristic scoring ladder
# -----------------------------------------------------------------------------
CONFIDENCE_HIGH   = 0.90  # Reason: ≥3 strong keyword signals detected
CONFIDENCE_MEDIUM = 0.65  # Reason: 1-2 keywords, some inference needed
CONFIDENCE_LOW    = 0.40  # Reason: No clear keywords, mostly defaults used

# Minimum keywords to achieve HIGH confidence
HIGH_CONFIDENCE_KEYWORD_THRESHOLD = 3
# Reason: 3+ distinct category signals → trustworthy profile (heuristic)

MEDIUM_CONFIDENCE_KEYWORD_THRESHOLD = 1
# Reason: at least 1 keyword → better than pure default (heuristic)

# -----------------------------------------------------------------------------
# WEIGHT SMOOTHING FACTOR (avoids 0.0 weights — keeps all categories alive)
# Reason: Prevents hard-zero categories; every category should have
#         a residual allocation in case user didn't mention it.
#         0.01 = 1% floor — small enough to not distort, big enough to exist.
# Type: empirically chosen smoothing constant
# -----------------------------------------------------------------------------
WEIGHT_FLOOR = 0.01
# IMPORTANT: Changing category names will break Simulation Agent

# -----------------------------------------------------------------------------
# PROFESSION ALIASES (normalize raw text → standard profession key)
# Reason: Users type many variations; we need a single canonical form.
# Type: configurable alias mapping
# -----------------------------------------------------------------------------
PROFESSION_ALIASES = {
    "student":      "student",
    "college":      "student",
    "uni":          "student",
    "university":   "student",
    "job":          "salaried",
    "employed":     "salaried",
    "salaried":     "salaried",
    "working":      "salaried",
    "it":           "engineer",
    "software":     "engineer",
    "developer":    "engineer",
    "engineer":     "engineer",
    "doctor":       "doctor",
    "physician":    "doctor",
    "consultant":   "professional",
    "professional": "professional",
    "business":     "business",
    "entrepreneur": "business",
    "self employed":"business",
    "self-employed":"business",
    "freelance":    "freelancer",
    "freelancer":   "freelancer",
    "retired":      "retired",
    "homemaker":    "homemaker",
    "housewife":    "homemaker",
}
