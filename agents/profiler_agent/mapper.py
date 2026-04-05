# =============================================================================
# mapper.py — Brand/Keyword → Category Mapper
# =============================================================================
# Maps user-mentioned brands, platforms, and lifestyle keywords
# to the 8 standard spend categories consumed by the Simulation Agent.
#
# FIX [Issue #1]: Extended synonym and phrase coverage.
# FIX [Issue #8]: Added phrase-level understanding before keyword split.
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# All values MUST be one of:
# ["dining", "travel", "online_shopping", "fuel", "groceries",
#  "entertainment", "utilities", "other"]
# =============================================================================

from typing import Dict, List, Tuple
import re

# =============================================================================
# BRAND → CATEGORY MAP
# =============================================================================
BRAND_CATEGORY_MAP: Dict[str, str] = {

    # ── DINING ────────────────────────────────────────────────────────────────
    "swiggy":           "dining",
    "zomato":           "dining",
    "blinkit":          "groceries",
    "dominos":          "dining",
    "domino's":         "dining",
    "domino":           "dining",
    "mcdonalds":        "dining",
    "mcdonald's":       "dining",
    "kfc":              "dining",
    "starbucks":        "dining",
    "cafe":             "dining",
    "restaurant":       "dining",
    "eazydiner":        "dining",
    "dineout":          "dining",
    "pizza hut":        "dining",
    "burger king":      "dining",
    "subway":           "dining",
    "food delivery":    "dining",   # FIX [Issue #8]: phrase detection
    "eat out":          "dining",   # FIX [Issue #8]
    "eating out":       "dining",   # FIX [Issue #8]
    "order food":       "dining",   # FIX [Issue #8]
    "dine out":         "dining",   # FIX [Issue #8]

    # ── ONLINE SHOPPING ───────────────────────────────────────────────────────
    "amazon":           "online_shopping",
    "flipkart":         "online_shopping",
    "myntra":           "online_shopping",
    "meesho":           "online_shopping",
    "nykaa":            "online_shopping",
    "ajio":             "online_shopping",
    "tata cliq":        "online_shopping",
    "croma":            "online_shopping",
    "reliance digital": "online_shopping",
    "shop online":      "online_shopping",  # FIX [Issue #1 / #8]
    "online shopping":  "online_shopping",  # FIX [Issue #8]
    "online purchase":  "online_shopping",  # FIX [Issue #8]
    "order online":     "online_shopping",  # FIX [Issue #8]

    # ── TRAVEL ────────────────────────────────────────────────────────────────
    "uber":             "travel",
    "ola":              "travel",
    "rapido":           "travel",
    "irctc":            "travel",
    "railway":          "travel",
    "train":            "travel",
    "flight":           "travel",
    "flights":          "travel",
    "air":              "travel",
    "airline":          "travel",
    "makemytrip":       "travel",
    "make my trip":     "travel",
    "cleartrip":        "travel",
    "goibibo":          "travel",
    "yatra":            "travel",
    "hotel":            "travel",
    "marriott":         "travel",
    "vistara":          "travel",
    "indigo":           "travel",
    "air india":        "travel",
    "airindia":         "travel",
    "lounge":           "travel",
    "bus":              "travel",
    "metro":            "travel",
    "book flights":     "travel",   # FIX [Issue #8]
    "book flight":      "travel",   # FIX [Issue #8]
    "flight booking":   "travel",   # FIX [Issue #8]
    "train ticket":     "travel",   # FIX [Issue #8]
    "air ticket":       "travel",   # FIX [Issue #8]
    "cab":              "travel",   # FIX [Issue #1]
    "taxi":             "travel",

    # ── FUEL ──────────────────────────────────────────────────────────────────
    "petrol":           "fuel",
    "diesel":           "fuel",
    "bpcl":             "fuel",
    "hpcl":             "fuel",
    "indian oil":       "fuel",
    "indianoil":        "fuel",
    "iocl":             "fuel",
    "pump":             "fuel",
    "fuel":             "fuel",
    "gas station":      "fuel",
    "cng":              "fuel",
    "fill petrol":      "fuel",     # FIX [Issue #8]
    "fill diesel":      "fuel",     # FIX [Issue #8]
    "fill up":          "fuel",
    "refuel":           "fuel",

    # ── GROCERIES ─────────────────────────────────────────────────────────────
    "bigbasket":        "groceries",
    "big basket":       "groceries",
    "dmart":            "groceries",
    "d-mart":           "groceries",
    "grofers":          "groceries",
    "zepto":            "groceries",
    "instamart":        "groceries",
    "supermarket":      "groceries",
    "kirana":           "groceries",
    "grocery":          "groceries",
    "groceries":        "groceries",
    "vegetables":       "groceries",
    "fruits":           "groceries",
    "household":        "groceries",
    "provisions":       "groceries",

    # ── ENTERTAINMENT ─────────────────────────────────────────────────────────
    "netflix":          "entertainment",
    "amazon prime":     "entertainment",
    "hotstar":          "entertainment",
    "disney":           "entertainment",
    "jiocinema":        "entertainment",
    "jio cinema":       "entertainment",
    "prime video":      "entertainment",
    "spotify":          "entertainment",
    "jiosaavn":         "entertainment",
    "gaana":            "entertainment",
    "youtube":          "entertainment",
    "movies":           "entertainment",
    "cinema":           "entertainment",
    "pvr":              "entertainment",
    "inox":             "entertainment",
    "bookmyshow":       "entertainment",
    "book my show":     "entertainment",
    "gaming":           "entertainment",
    "steam":            "entertainment",
    "playstation":      "entertainment",
    "ott":              "entertainment",
    "streaming":        "entertainment",
    "binge":            "entertainment",
    "subscriptions":    "entertainment",
    "subscription":     "entertainment",

    # ── UTILITIES ─────────────────────────────────────────────────────────────
    "electricity":      "utilities",
    "water bill":       "utilities",
    "gas bill":         "utilities",
    "broadband":        "utilities",
    "wifi":             "utilities",
    "internet":         "utilities",
    "airtel":           "utilities",
    "jio":              "utilities",
    "vi ":              "utilities",     # trailing space avoids matching "vistara"
    "vodafone":         "utilities",
    "bsnl":             "utilities",
    "recharge":         "utilities",
    "mobile bill":      "utilities",
    "phone bill":       "utilities",
    "paytm":            "utilities",
    "insurance":        "utilities",
    "utility bills":    "utilities",    # FIX [Issue #8]
    "monthly bills":    "utilities",    # FIX [Issue #8]
    "bill payment":     "utilities",    # FIX [Issue #8]
    "pay bills":        "utilities",    # FIX [Issue #8]

    # ── OTHER ─────────────────────────────────────────────────────────────────
    "rent":             "other",
    "education":        "other",
    "medical":          "other",
    "hospital":         "other",
    "pharmacy":         "other",
    "emi":              "other",
    "google pay":       "other",
    "gpay":             "other",
    "phonepe":          "other",
    "upi":              "other",
}


# =============================================================================
# LIFESTYLE KEYWORD MAP
# FIX [Issue #1]: Expanded synonyms and phrasing variations.
# =============================================================================
LIFESTYLE_KEYWORD_MAP: Dict[str, str] = {

    # DINING
    "eat out":          "dining",
    "eating out":       "dining",
    "food delivery":    "dining",
    "foodie":           "dining",
    "dining":           "dining",
    "food":             "dining",
    "lunch":            "dining",
    "dinner":           "dining",
    "breakfast":        "dining",
    "dine":             "dining",
    "meal":             "dining",
    "snack":            "dining",
    "takeaway":         "dining",
    "takeout":          "dining",

    # TRAVEL
    "travel":           "travel",
    "trip":             "travel",
    "vacation":         "travel",
    "holiday":          "travel",
    "fly":              "travel",
    "flying":           "travel",
    "frequent flyer":   "travel",
    "miles":            "travel",
    "international":    "travel",
    "abroad":           "travel",
    "backpacking":      "travel",
    "commute":          "travel",
    "road trip":        "travel",
    "airport":          "travel",

    # ONLINE SHOPPING
    "online shopping":  "online_shopping",
    "shopping":         "online_shopping",
    "e-commerce":       "online_shopping",
    "ecommerce":        "online_shopping",
    "order online":     "online_shopping",
    "online purchase":  "online_shopping",
    "shop online":      "online_shopping",

    # FUEL
    "drive":            "fuel",
    "driving":          "fuel",
    "car":              "fuel",
    "vehicle":          "fuel",
    "bike":             "fuel",
    "two wheeler":      "fuel",
    "four wheeler":     "fuel",
    "scooter":          "fuel",

    # GROCERIES
    "grocery":          "groceries",
    "groceries":        "groceries",
    "household":        "groceries",
    "provisions":       "groceries",
    "vegetables":       "groceries",
    "fruits":           "groceries",
    "supermarket":      "groceries",

    # ENTERTAINMENT
    "entertainment":    "entertainment",
    "streaming":        "entertainment",
    "binge":            "entertainment",
    "subscriptions":    "entertainment",
    "ott":              "entertainment",
    "movies":           "entertainment",
    "gaming":           "entertainment",

    # UTILITIES
    "utility":          "utilities",
    "utilities":        "utilities",
    "bills":            "utilities",
    "bill payment":     "utilities",
    "recharges":        "utilities",
    "monthly bills":    "utilities",
    "pay bills":        "utilities",
}


# =============================================================================
# get_category_for_keyword()
# Priority: brand map > lifestyle map > None
# =============================================================================
def get_category_for_keyword(keyword: str) -> str | None:
    keyword_lower = keyword.lower().strip()

    if keyword_lower in BRAND_CATEGORY_MAP:
        return BRAND_CATEGORY_MAP[keyword_lower]

    if keyword_lower in LIFESTYLE_KEYWORD_MAP:
        return LIFESTYLE_KEYWORD_MAP[keyword_lower]

    # Partial substring match (longest first)
    for brand, category in sorted(BRAND_CATEGORY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if brand in keyword_lower or keyword_lower in brand:
            return category

    for phrase, category in sorted(LIFESTYLE_KEYWORD_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase in keyword_lower or keyword_lower in phrase:
            return category

    return None


# =============================================================================
# extract_category_signals()
# FIX [Issue #8]: Multi-word phrase detection runs BEFORE single-word scan.
# FIX [Issue #1]: Synonyms and phrase variants added above.
# =============================================================================
def extract_category_signals(text: str) -> List[Tuple[str, float]]:
    """
    Scan the full user text and extract (category, raw_weight) signals.
    Multi-word phrases are matched before single words to avoid partial conflicts.
    Returns list of (category_name, 1.0) hits — intensity applied in parser.
    """
    from defaults import CATEGORIES

    signals: List[Tuple[str, float]] = []
    text_lower = text.lower()

    # Sort all phrases longest-first: ensures "food delivery" matched before "food"
    all_phrases = sorted(
        list(BRAND_CATEGORY_MAP.keys()) + list(LIFESTYLE_KEYWORD_MAP.keys()),
        key=len, reverse=True
    )

    matched_spans = []

    for phrase in all_phrases:
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
        # FIX [Issue #1]: Added word boundaries (\b) to prevent partial matches
        # e.g. "air" should not fire inside "fair" or "affair"
        for match in pattern.finditer(text_lower):
            start, end = match.start(), match.end()

            # Skip overlapping matches (longest match wins)
            overlap = any(s <= start < e or s < end <= e for s, e in matched_spans)
            if overlap:
                continue

            matched_spans.append((start, end))
            category = get_category_for_keyword(phrase)
            if category and category in CATEGORIES:
                signals.append((category, 1.0))

    return signals
