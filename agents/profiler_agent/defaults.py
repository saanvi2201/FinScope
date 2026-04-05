# =============================================================================
# defaults.py — Central Configuration for Behavioral Agent
# =============================================================================
# ⚠️ INTEGRATION NOTE: These constants are tunable.
#    Changing CATEGORIES or CATEGORY keys WILL break the Simulation Agent.
#    Only change numeric values, not string keys.
# =============================================================================

# -----------------------------------------------------------------------------
# CATEGORIES (LOCKED — DO NOT RENAME)
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
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
# Reason: Balanced fallback distribution used ONLY when user input is
#         completely empty. NOT used when any signals are detected.
#         FIX [Issue #4]: Default weights no longer leak into detected profiles.
# -----------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "dining":           0.20,
    "online_shopping":  0.20,
    "groceries":        0.20,
    "travel":           0.15,
    "utilities":        0.10,
    "entertainment":    0.05,
    "fuel":             0.05,
    "other":            0.05,
}

# -----------------------------------------------------------------------------
# DEFAULT INTENSITY SCORE
# Reason: Moderate baseline when user does NOT specify frequency.
#         Scale: 0.0 (never) → 1.0 (very frequently/daily).
# -----------------------------------------------------------------------------
DEFAULT_INTENSITY = 0.6

# -----------------------------------------------------------------------------
# INTENSITY MODIFIERS (keyword → intensity multiplier)
# FIX [Issue #2]: Intensity is detected PER CATEGORY using a local window.
# Reason: Maps linguistic signals to quantitative usage intensity.
# -----------------------------------------------------------------------------
INTENSITY_MAP = {
    # HIGH usage signals
    "a lot":        1.0,
    "heavily":      1.0,
    "heavy":        1.0,
    "always":       1.0,
    "daily":        1.0,
    "every day":    1.0,
    "mostly":       0.9,
    "mainly":       0.9,
    "primarily":    0.9,
    "frequently":   0.8,
    "frequent":     0.8,
    "often":        0.8,
    "regularly":    0.75,
    "every week":   0.75,
    "weekly":       0.75,
    "every month":  0.65,
    "monthly":      0.65,
    # MEDIUM usage signals
    "sometimes":    0.5,
    "occasional":   0.4,
    "occasionally": 0.4,
    "moderate":     0.5,
    "moderately":   0.5,
    "on weekends":  0.5,   # FIX [Issue #8]: phrase-level intensity
    "on weekend":   0.5,
    "during sales": 0.5,   # FIX [Issue #2]: was previously ignored
    # LOW usage signals
    "rarely":       0.2,
    "seldom":       0.2,
    "hardly":       0.15,
    "almost never": 0.1,
    "never":        0.0,
}

# -----------------------------------------------------------------------------
# TOTAL SPEND ESTIMATION BY PROFESSION
# -----------------------------------------------------------------------------
PROFESSION_SPEND_MAP = {
    "student":      20000,
    "freelancer":   30000,
    "salaried":     40000,
    "working":      40000,
    "professional": 50000,
    "doctor":       60000,
    "engineer":     45000,
    "business":     60000,
    "retired":      25000,
    "homemaker":    25000,
}

DEFAULT_TOTAL_SPEND = 30000

# -----------------------------------------------------------------------------
# INCOME-BASED SPEND RATIO
# -----------------------------------------------------------------------------
INCOME_SPEND_RATIO = 0.40

# -----------------------------------------------------------------------------
# SPEND CAPS
# -----------------------------------------------------------------------------
MIN_SPEND = 10_000
MAX_SPEND = 100_000

# -----------------------------------------------------------------------------
# CONFIDENCE THRESHOLDS
# FIX [Issue #9]: Tighter, more accurate confidence ladder.
# -----------------------------------------------------------------------------
CONFIDENCE_HIGH   = 0.90   # ≥ 3 strong keyword signals
CONFIDENCE_MEDIUM = 0.65   # 1-2 keyword signals
CONFIDENCE_LOW    = 0.35   # cold start / no valid signals
CONFIDENCE_ZERO   = 0.0    # invalid / empty input

HIGH_CONFIDENCE_KEYWORD_THRESHOLD   = 3
MEDIUM_CONFIDENCE_KEYWORD_THRESHOLD = 1

# -----------------------------------------------------------------------------
# TAG THRESHOLDS
# FIX [Issue #5]: Stricter tag thresholds to prevent over-generation.
# Previously: heavy ≥0.30, moderate ≥0.15 — too loose.
# Now:        heavy ≥0.40, moderate ≥0.20 — aligned with actual dominance.
# -----------------------------------------------------------------------------
TAG_HEAVY_THRESHOLD    = 0.40   # Reason: ≥40% → truly dominant category
TAG_MODERATE_THRESHOLD = 0.20   # Reason: 20-40% → meaningful but secondary

# -----------------------------------------------------------------------------
# WEIGHT SMOOTHING
# FIX [Issue #4]: WEIGHT_FLOOR is now ONLY applied during cold-start /
#                 zero-signal fallback paths, not to every detected profile.
#                 This prevents unused categories from leaking into results.
# -----------------------------------------------------------------------------
WEIGHT_FLOOR = 0.01   # Only used in cold-start & normalization guard

# -----------------------------------------------------------------------------
# PROFESSION ALIASES
# -----------------------------------------------------------------------------
PROFESSION_ALIASES = {
    "student":       "student",
    "college":       "student",
    "uni":           "student",
    "university":    "student",
    "job":           "salaried",
    "employed":      "salaried",
    "salaried":      "salaried",
    "working":       "salaried",
    "it":            "engineer",
    "software":      "engineer",
    "developer":     "engineer",
    "engineer":      "engineer",
    "doctor":        "doctor",
    "physician":     "doctor",
    "consultant":    "professional",
    "professional":  "professional",
    "business":      "business",
    "entrepreneur":  "business",
    "self employed": "business",
    "self-employed": "business",
    "freelance":     "freelancer",
    "freelancer":    "freelancer",
    "retired":       "retired",
    "homemaker":     "homemaker",
    "housewife":     "homemaker",
}
