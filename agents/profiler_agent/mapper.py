# =============================================================================
# mapper.py — Brand/Keyword → Category Mapper
# =============================================================================
# Maps user-mentioned brands, platforms, and lifestyle keywords
# to the 8 standard spend categories consumed by the Simulation Agent.
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
# Reason: Direct mapping from named platforms/brands to spend categories.
#         Built from the card list in the project (Swiggy HDFC, Amazon Pay, etc.)
#         and common Indian consumer apps.
# Type: configurable lookup table
# =============================================================================
BRAND_CATEGORY_MAP: Dict[str, str] = {

    # ── DINING ────────────────────────────────────────────────────────────────
    "swiggy":       "dining",   # Reason: food delivery → dining
    "zomato":       "dining",
    "blinkit":      "groceries",# Reason: quick commerce → groceries primarily
    "dominos":      "dining",
    "domino":       "dining",
    "mcdonalds":    "dining",
    "kfc":          "dining",
    "starbucks":    "dining",
    "cafe":         "dining",
    "restaurant":   "dining",
    "eazydiner":    "dining",   # Reason: card in list (EazyDiner IndusInd)
    "dineout":      "dining",

    # ── ONLINE SHOPPING ───────────────────────────────────────────────────────
    "amazon":       "online_shopping",  # Reason: ICICI Amazon Pay card in portfolio
    "flipkart":     "online_shopping",  # Reason: Axis Flipkart card in portfolio
    "myntra":       "online_shopping",  # Reason: Myntra Kotak card in portfolio
    "meesho":       "online_shopping",
    "nykaa":        "online_shopping",
    "ajio":         "online_shopping",
    "tata cliq":    "online_shopping",
    "croma":        "online_shopping",
    "reliance":     "online_shopping",  # Reason: Reliance SBI card in portfolio

    # ── TRAVEL ────────────────────────────────────────────────────────────────
    "uber":         "travel",   # Reason: cab rides → travel
    "ola":          "travel",
    "rapido":       "travel",
    "irctc":        "travel",   # Reason: IRCTC SBI card in portfolio
    "railway":      "travel",
    "train":        "travel",
    "flight":       "travel",
    "air":          "travel",   # catch "air travel", "air tickets"
    "airline":      "travel",
    "makemytrip":   "travel",
    "cleartrip":    "travel",
    "goibibo":      "travel",
    "yatra":        "travel",
    "hotel":        "travel",
    "marriott":     "travel",   # Reason: Marriott Bonvoy HDFC in portfolio
    "vistara":      "travel",   # Reason: Vistara IDFC/Axis in portfolio
    "indigo":       "travel",
    "airindia":     "travel",
    "lounge":       "travel",   # Reason: lounge access → travel benefit
    "bus":          "travel",
    "metro":        "travel",

    # ── FUEL ──────────────────────────────────────────────────────────────────
    "petrol":       "fuel",     # Reason: direct fuel purchase
    "diesel":       "fuel",
    "bpcl":         "fuel",     # Reason: BPCL SBI Octane in portfolio
    "hpcl":         "fuel",     # Reason: ICICI HPCL Super Saver in portfolio
    "indian oil":   "fuel",
    "indianoil":    "fuel",     # Reason: IndianOil HDFC/Axis in portfolio
    "iocl":         "fuel",
    "pump":         "fuel",
    "fuel":         "fuel",
    "gas station":  "fuel",
    "cng":          "fuel",

    # ── GROCERIES ─────────────────────────────────────────────────────────────
    "bigbasket":    "groceries",# Reason: online grocery → groceries
    "dmart":        "groceries",
    "grofers":      "groceries",
    "zepto":        "groceries",# Reason: quick commerce → groceries
    "instamart":    "groceries",# Swiggy Instamart
    "supermarket":  "groceries",
    "kirana":       "groceries",
    "grocery":      "groceries",
    "vegetables":   "groceries",
    "fruits":       "groceries",

    # ── ENTERTAINMENT ─────────────────────────────────────────────────────────
    "netflix":      "entertainment",
    "amazon prime": "entertainment",
    "hotstar":      "entertainment",
    "disney":       "entertainment",
    "jiocinema":    "entertainment",
    "prime video":  "entertainment",
    "spotify":      "entertainment",
    "jiosaavn":     "entertainment",
    "gaana":        "entertainment",
    "youtube":      "entertainment",
    "movies":       "entertainment",
    "cinema":       "entertainment",
    "pvr":          "entertainment",
    "inox":         "entertainment",
    "bookmyshow":   "entertainment",
    "gaming":       "entertainment",
    "steam":        "entertainment",
    "playstation":  "entertainment",

    # ── UTILITIES ─────────────────────────────────────────────────────────────
    "electricity":  "utilities",
    "water bill":   "utilities",
    "gas bill":     "utilities",
    "broadband":    "utilities",
    "wifi":         "utilities",
    "internet":     "utilities",
    "airtel":       "utilities",    # Reason: Airtel Axis card in portfolio
    "jio":          "utilities",
    "vi":           "utilities",
    "vodafone":     "utilities",
    "bsnl":         "utilities",
    "recharge":     "utilities",
    "mobile bill":  "utilities",
    "phone bill":   "utilities",
    "paytm":        "utilities",    # Reason: Paytm SBI card; mainly bill payments
    "insurance":    "utilities",    # Reason: recurring payment, often utility-like
    "emi":          "other",        # Reason: EMIs usually excluded from reward calc

    # ── OTHER ─────────────────────────────────────────────────────────────────
    "rent":         "other",        # Reason: rent is explicitly excluded in most cards
    "education":    "other",
    "medical":      "other",
    "hospital":     "other",
    "pharmacy":     "other",

    # FINTECH / UPI (treated as other unless specific context)
    "google pay":   "other",        # Reason: UPI payments vary in category
    "gpay":         "other",
    "phonepe":      "other",
    "upi":          "other",        # Reason: mapped to Tata Neu/Axis ACE if needed
}


# =============================================================================
# LIFESTYLE KEYWORD → CATEGORY MAP
# Reason: Users often describe spending style, not brand names.
#         These phrases signal dominant categories without naming platforms.
# Type: configurable, extendable mapping
# =============================================================================
LIFESTYLE_KEYWORD_MAP: Dict[str, str] = {

    # DINING signals
    "eat out":          "dining",
    "eating out":       "dining",
    "food delivery":    "dining",
    "foodie":           "dining",
    "dining":           "dining",
    "food":             "dining",
    "lunch":            "dining",
    "dinner":           "dining",
    "breakfast":        "dining",

    # TRAVEL signals
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

    # ONLINE SHOPPING signals
    "online shopping":  "online_shopping",
    "shopping":         "online_shopping",
    "e-commerce":       "online_shopping",
    "ecommerce":        "online_shopping",
    "order online":     "online_shopping",
    "online purchase":  "online_shopping",

    # FUEL signals
    "drive":            "fuel",
    "driving":          "fuel",
    "car":              "fuel",     # Reason: car owner → likely fuel spend
    "vehicle":          "fuel",
    "bike":             "fuel",

    # GROCERIES signals
    "groceries":        "groceries",
    "household":        "groceries",
    "provisions":       "groceries",

    # ENTERTAINMENT signals
    "entertainment":    "entertainment",
    "streaming":        "entertainment",
    "binge":            "entertainment",
    "subscriptions":    "entertainment",
    "ott":              "entertainment",    # Over-the-top platforms

    # UTILITIES signals
    "utility":          "utilities",
    "utilities":        "utilities",
    "bills":            "utilities",
    "bill payment":     "utilities",
    "recharges":        "utilities",
    "monthly bills":    "utilities",
}


# =============================================================================
# get_category_for_keyword()
# Reason: Unified lookup — checks brands first (specific), then lifestyle (general).
#         Returns None if no match (handled in parser).
# =============================================================================
def get_category_for_keyword(keyword: str) -> str | None:
    """
    Maps a single keyword to a spend category.
    Priority: brand map > lifestyle map > None
    """
    keyword_lower = keyword.lower().strip()

    # Step 1: exact brand match
    if keyword_lower in BRAND_CATEGORY_MAP:
        return BRAND_CATEGORY_MAP[keyword_lower]

    # Step 2: exact lifestyle match
    if keyword_lower in LIFESTYLE_KEYWORD_MAP:
        return LIFESTYLE_KEYWORD_MAP[keyword_lower]

    # Step 3: partial substring match (e.g. "amazon prime" found in "amazon prime video")
    for brand, category in BRAND_CATEGORY_MAP.items():
        if brand in keyword_lower or keyword_lower in brand:
            return category

    for phrase, category in LIFESTYLE_KEYWORD_MAP.items():
        if phrase in keyword_lower or keyword_lower in phrase:
            return category

    return None  # Reason: no match found, will be handled upstream


# =============================================================================
# extract_category_signals()
# Reason: Processes the FULL user text to extract all (category, intensity)
#         pairs. Handles multi-word phrases before single-word lookups.
#         Returns a list of (category, intensity_hint) tuples.
# =============================================================================
def extract_category_signals(text: str) -> List[Tuple[str, float]]:
    """
    Scan the full user text and extract (category, raw_weight) signals.
    Returns list of (category_name, 1.0) hits — intensity is applied in parser.
    """
    from defaults import CATEGORIES   # IMPORTANT: Changing category names will break Simulation Agent

    signals: List[Tuple[str, float]] = []
    text_lower = text.lower()

    # Check multi-word phrases first (longer match wins)
    all_phrases = sorted(
        list(BRAND_CATEGORY_MAP.keys()) + list(LIFESTYLE_KEYWORD_MAP.keys()),
        key=len, reverse=True  # Reason: longer phrases matched first to avoid partial conflicts
    )

    matched_spans = []  # track already-matched positions to avoid double-counting

    for phrase in all_phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            start, end = match.start(), match.end()

            # Reason: skip overlapping matches — first (longest) match wins
            overlap = any(s <= start < e or s < end <= e for s, e in matched_spans)
            if overlap:
                continue

            matched_spans.append((start, end))
            category = get_category_for_keyword(phrase)
            if category and category in CATEGORIES:
                signals.append((category, 1.0))  # raw hit; intensity scaled in parser.py

    return signals
