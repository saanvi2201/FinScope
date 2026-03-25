# =============================================================================
# preprocess.py — Credit Card Document Preprocessing Pipeline
# =============================================================================
# PURPOSE:
#   Reads raw PDF files from /data/raw_docs/, extracts text, classifies each
#   document (type, bank, card), renames and organises them into /data/processed_docs/,
#   and routes uncertain files to /data/needs_review/.
#   After processing, a coverage validation step checks that every expected
#   card has its required documents (MITC, BR) and reports gaps.
#
# USAGE:
#   python preprocess.py              # normal run
#   python preprocess.py --dry-run    # simulate only, no files moved
#   python preprocess.py --debug      # print extracted text for inspection
#
# OUTPUTS:
#   data/processed_docs/              organised PDFs
#   data/needs_review/                low-confidence files
#   data/logs/summary.csv             per-file processing summary
#   data/logs/preprocess_log.txt      detailed audit log
#   data/logs/missing_docs_report.csv coverage gaps per card
# =============================================================================

import os
import re
import csv
import sys
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pandas as pd
except ImportError:
    pd = None


# =============================================================================
# ██████████████████████  USER EDITABLE CONFIGURATION  ████████████████████████
# =============================================================================
# ALL project-specific settings live here.
# You should NEVER need to edit anything below the END OF CONFIGURATION marker.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# BANK ALIAS MAP  ← THIS IS WHERE SHORT CODES FOR FILENAMES ARE DEFINED
# ─────────────────────────────────────────────────────────────────────────────
#
# HOW THIS WORKS:
#   The script searches the PDF text for any of the PHRASES listed under
#   each bank. When it finds a match, it uses the SHORT CODE (the key)
#   in the output filename — NOT the full bank name.
#
# FORMAT:
#   "SHORT_CODE": ["search phrase 1", "search phrase 2", ...]
#
#   SHORT_CODE  → what appears in the renamed file
#                 e.g.  BOB_Eterna_MITC_2024.pdf   (not "Bank_of_Baroda_...")
#                 e.g.  SCB_Smart_BR_2024.pdf       (not "Standard_Chartered_...")
#
#   phrases     → what the detector looks for inside the PDF text
#                 add as many variations as the bank uses in its documents
#                 detection is case-insensitive ("hdfc bank" matches "HDFC Bank")
#
# ORDERING TIP:
#   List longer/more-specific phrases FIRST within each bank's list.
#   e.g. "hdfc bank ltd" before "hdfc bank" before "hdfc"
#   This ensures the most specific match is found first.
#
# TO ADD A NEW BANK:
#   1. Add a new entry:  "SHORTCODE": ["phrase in pdf", "another phrase"],
#   2. Use the same SHORTCODE in EXPECTED_CARDS entries for that bank
#
# REAL EXAMPLE:
#   "BOB": ["bank of baroda", "bob financial", "bob"]
#   → PDF contains "Bank of Baroda"
#   → matched phrase "bank of baroda" under key "BOB"
#   → output file is named  BOB_Eterna_MITC_2024.pdf  (not "Bank_of_Baroda_...")
# ─────────────────────────────────────────────────────────────────────────────

BANK_ALIASES = {
    # SHORT CODE   : [phrases to search for in PDF text — most specific first]
    "HDFC"         : ["hdfc bank ltd", "hdfc bank limited", "hdfc bank", "hdfc"],
    "SBI"          : ["state bank of india", "sbi cards and payment", "sbi card", "sbi"],
    "AXIS"         : ["axis bank limited", "axis bank", "axis"],
    "ICICI"        : ["icici bank limited", "icici bank", "icici"],
    "HSBC"         : ["the hongkong and shanghai banking", "hsbc bank", "hsbc"],
    "KOTAK"        : ["kotak mahindra bank limited", "kotak mahindra bank", "kotak mahindra", "kotak"],
    "YES"          : ["yes bank limited", "yes bank", "yes"],
    "IDFC"         : ["idfc first bank limited", "idfc first bank", "idfc first", "idfc"],
    "AU"           : ["au small finance bank limited", "au small finance bank", "au bank", "au small finance"],
    "BOB"          : ["bank of baroda", "bob financial solutions limited", "bob financial", "bob"],
    "RBL"          : ["rbl bank limited", "ratnakar bank limited", "rbl bank", "rbl"],
    "INDUSIND"     : ["indusind bank limited", "indusind bank", "indusind"],
    "CITI"         : ["citibank n.a", "citibank na", "citibank", "citi bank", "citi"],
    "AMEX"         : ["american express banking corp", "american express", "amex"],
    "SCB"          : [
        # FIX (Problem 10): SC_Smart_MITC_2026.pdf was detected as SBI because
        # the double-spaced text "IIMMPPOORRTTAANNTT IINNFFOORRMMAATTIIOONN" caused
        # "sbi" to appear as a substring somewhere in the garbled encoding.
        # Longer, more specific phrases for SCB are listed first so they match
        # before the short "sbi" string from the SBI alias list can fire.
        "standard chartered bank",
        "standard chartered credit card",
        "standard chartered",
        "stanchart",
        "sc.com/in",       # URL found in SC_General_TNC_Client_2022.pdf debug text
        "sc.com",
    ],
    "ONECARD"      : ["onecard", "one card", "ftc fintech private"],
    "SCAPIA"       : ["scapia technologies", "scapia", "federal bank scapia"],
    "JUPITER"      : ["jupiter money", "jupiter"],
}


# ─────────────────────────────────────────────────────────────────────────────
# MASTER / COLLECTIVE DOCUMENT DETECTION  ← USER EDITABLE
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS A MASTER DOCUMENT?
#   Some banks publish a single MITC or TNC that legally applies to ALL their
#   credit cards — rather than issuing one per card variant. For example:
#     • HDFC publishes one common MITC covering Infinia, Millennia, Regalia, etc.
#     • SBI issues a single TNC applicable across all SBI credit cards.
#
# HOW THE PIPELINE HANDLES IT:
#   1. Before card detection runs, the text is scanned for MASTER_DOC_SIGNALS.
#   2. If any signal phrase is found → card name is set to "MASTER" (skips
#      individual card detection entirely for this file).
#   3. The file is renamed and placed in a dedicated MASTER folder:
#        Filename : HDFC_MASTER_MITC_2024.pdf
#        Folder   : processed_docs/HDFC_MASTER/
#
# WHY THIS MATTERS:
#   Without this, a collective HDFC MITC would either:
#     (a) Wrongly match the first card name found in the text (e.g. "Infinia"),
#         making it look like a card-specific doc when it isn't, OR
#     (b) Fail card detection entirely and land in needs_review/ with UNKNOWN_CARD.
#   Both outcomes are wrong. MASTER detection prevents this.
#
# TO ADD A NEW SIGNAL PHRASE:
#   Just append it to MASTER_DOC_SIGNALS below. Matching is case-insensitive.
#
# BANKS KNOWN TO PUBLISH COLLECTIVE DOCS (informational — for your reference):
#   Documented during project scoping. Detection still relies on signal phrases,
#   not on this list. Add banks here as you discover more.
# ─────────────────────────────────────────────────────────────────────────────

MASTER_DOC_BANKS = [
    # Banks confirmed to publish collective/bank-wide documents.
    # This list is for documentation only — it is NOT used in detection logic.
    "HDFC", "SBI", "AXIS", "ICICI", "AMEX", "SCB",
]

MASTER_DOC_SIGNALS = [
    # ── Phrases confirmed from actual PDF debug output ────────────────────────
    # These were extracted by running --debug and reading the real document text.
    # Add new phrases here whenever you find a collective doc that isn't detected.

    # AXIS — confirmed from Axis_ACE_General_2024.pdf and Axis_Master_MITC_2026.pdf
    "applicable to all credit card holders / applicants of credit cards",
    "all the information herein is applicable to all credit card holders",
    "applicable to all credit card holders",

    # AXIS LOUNGE — confirmed from Axis_Master_Lounge_2024.pdf debug text
    # "Axis Bank Credit Cards Domestic Airport Lounge Access Program...
    #  list of Credit Card variants, their corresponding complimentary lounge access"
    "axis bank credit cards domestic airport lounge",
    "list of credit card variants",                # appears in Axis lounge doc header
    "credit card variants, their corresponding",   # more specific variant of above

    # IDFC — confirmed from IDFC_Master_MITC_2026.pdf
    "applicable to all credit card members / applicants of credit cards",
    "all the information herein is applicable to all credit card members",
    "applicable to all credit card members",

    # AMEX MITC — confirmed from AMEX_Master_MITC_2026.pdf
    "as a part of all credit card applications",
    "specific reference is given if any terms and conditions are applicable only to a particular",
    "this mitc is to be read along with",

    # AMEX TNC — confirmed from AMEX_Master_TNC_Agreement_2026.pdf debug text
    # "AMERICAN EXPRESS® CREDIT CARD CARDMEMBER AGREEMENT...
    #  Before you use the enclosed American Express Credit Card"
    # No specific card named → generic cardmember agreement = master doc
    "american express credit card cardmember agreement",
    "american express® credit card cardmember agreement",
    "before you use the enclosed american express credit card",

    # AU MASTER — confirmed from AU_Master_TNC_Agreement_2025.pdf debug text
    # "These Terms and Conditions apply to the AU Small Finance Bank Credit Card"
    # Generic — no specific card variant named
    "these terms and conditions apply to the au small finance bank credit card",
    "au small finance bank credit card",           # shorter fallback phrase

    # SCB — confirmed from SC_Smart_MITC_2026.pdf (already working)
    "applicable to all credit cards",

    # HDFC — confirmed from HDFC_General_MITC_2026.pdf
    "key fact statement cum most important terms & conditions for credit cards",
    "key fact statement cum most important terms and conditions for credit cards",
    "most important terms & conditions for credit cards",
    "most important terms and conditions for credit cards",

    # SBI — confirmed from SBI_General_MITC_2026.pdf (multi-card fee table)
    "sbi card miles",

    # ── Generic collective-doc phrases (bank-agnostic) ────────────────────────
    "applicable to all cards",
    "all credit cardholders",
    "common terms and conditions",
    "general terms and conditions for credit cards",
    "these terms apply to all",
    "irrespective of the card variant",
    "for all card variants",
    "master terms",
    "uniform terms",
    "applicable across all variants",
    "all hdfc bank credit cards",
    "all hdfc credit cardholders",
    "all sbi credit cards",
    "all state bank credit cards",
    "all axis bank credit cards",
    "all icici bank credit cards",
    "all american express cardmembers",
    "all standard chartered credit cards",
]


# ─────────────────────────────────────────────────────────────────────────────
# CARD NAMES
# ─────────────────────────────────────────────────────────────────────────────
#
# HOW THIS WORKS:
#   The detector searches the PDF text for these strings (case-insensitive).
#   The FIRST match found is used as the card name in the output filename.
#
# ORDERING RULE — critical:
#   The list is scanned TOP TO BOTTOM and stops at the FIRST match.
#   If Card A's name contains Card B's name, Card A MUST appear first.
#
#   Examples of correct ordering:
#     "Platinum Travel"   before  "Platinum"         (Travel is more specific)
#     "Tata Neu Infinity" before  "Tata Neu"          (Infinity is more specific)
#     "BPCL Octane"       before  "BPCL"              (Octane is more specific)
#     "Regalia Gold"      before  "Regalia"           (Gold is more specific)
#     "Diners Club Black" before  "Diners"            (Club Black is more specific)
#     "MoneyBack+"        before  "MoneyBack"         (+ distinguishes the newer card)
#     "Vistara Infinite"  before  "Vistara"
#     "HPCL Super Saver"  before  "HPCL"
#     "League Platinum"   before  "Platinum"
#
# TO ADD A NEW CARD:
#   Add it in the correct position — more specific names higher up in the list.
# ─────────────────────────────────────────────────────────────────────────────

CARDS = [
    # ── ORDERING RULE: more specific names MUST come before shorter ones ───────
    # Short generic names like "ACE", "Smart", "Cashback" appear in almost every
    # credit card document body. They are placed at the BOTTOM of the list.
    # Bank-qualified names (e.g. "Axis ACE") are preferred where possible.

    # ── Premium & Travel Cards — specific, unlikely to appear in other docs ────
    "Infinia",              # HDFC Infinia
    "Diners Club Black",    # BEFORE "Diners"
    "Diners",               # HDFC Diners variants
    "Magnus",               # Axis Magnus
    "Atlas",                # Axis Atlas

    # FIX (Problem 1): "Blue Cash" moved ABOVE "Platinum Travel" so that
    # AMEX_BlueCash docs match Blue Cash in the header before Platinum Travel
    # is found anywhere in the body text.
    "Blue Cash",            # Amex Blue Cash — MUST be before "Platinum Travel"
    "Platinum Travel",      # Amex Platinum Travel — BEFORE plain "Platinum"

    "Emeralde",             # ICICI Emeralde Private
    "Elite",                # SBI Elite
    "Marriott Bonvoy",      # Marriott Bonvoy HDFC
    "Reserve",              # Axis Reserve
    "Premier",              # HSBC Premier
    "Aurum",                # SBI Aurum
    "Vistara Infinite",     # BEFORE "Vistara"
    "Vistara",              # Vistara variants
    "Ultimate",             # StanC Ultimate
    "World Safari",         # RBL World Safari

    # ── Lifestyle & Niche Cards ───────────────────────────────────────────────
    "Tata Neu Infinity",    # BEFORE "Tata Neu"
    "Tata Neu",
    "Swiggy",
    "Airtel",
    "BPCL Octane",          # BEFORE "BPCL"
    "BPCL",
    "IndianOil",
    # FIX: "Flipkart" moved ABOVE "Myntra" — Axis Flipkart TNC doc header says
    # "FLIPKART AXIS BANK Credit Card... Flipkart, Myntra and Cleartrip".
    # "Myntra" was appearing in the same header line and matching first because
    # it was listed before Flipkart. Flipkart must come first.
    "Flipkart",
    "Myntra",
    "Reliance",
    "IRCTC",
    "Regalia Gold",         # BEFORE "Regalia"
    "Regalia",
    "HPCL Super Saver",     # BEFORE any plain "HPCL"
    "EazyDiner",
    "Paytm",
    "Pixel",
    "Coral",
    "Rubyx",
    "6E Rewards",
    "OneCard",
    "Scapia",
    "Jupiter",

    # ── Amazon Pay: plain name only ───────────────────────────────────────────
    # FIX: Removed "Amazon Pay ICICI" — the bank is already detected separately
    # by detect_bank(). Having "Amazon Pay ICICI" as the card name produced
    # folders like "ICICI_Amazon_Pay_ICICI/" and filenames with the bank name
    # appearing twice. Plain "Amazon Pay" is correct and sufficient.
    "Amazon Pay",

    # ── Cards safe to match by distinctive name ───────────────────────────────
    "Altura",               # AU Altura
    "Eterna",               # Bank of Baroda Eterna
    "League Platinum",      # BEFORE "Platinum"
    "Prosperity",           # YES Bank Prosperity
    "MoneyBack+",           # BEFORE "MoneyBack"
    "MoneyBack",
    "SimplyCLICK",
    "SimplySAVE",
    "Signature",
    "Live+",                # HSBC Live+
    "Millennia",            # HDFC / IDFC Millennia
    "Platinum",             # Amex Platinum (all Platinum X variants are above)
    "Select",               # Axis Select

    # ── Generic/short names — LAST to prevent false positives ─────────────────
    "Axis ACE",             # qualified form — BEFORE plain "ACE"
    "ACE",                  # word-boundary matched in detect_card()
    "HDFC Millennia",       # fully qualified fallback
    # FIX: Removed "Cashback SBI" — bank is already detected separately.
    # "Cashback SBI" as card name was producing "SBI_Cashback_SBI/" folders
    # (bank name twice). Plain "Cashback" is the correct card name.
    "Cashback",             # appears in almost every doc — last resort
    "Smart Credit Card",    # qualified — BEFORE plain "Smart"
    "Smart",                # plain form — last resort
]


# ─────────────────────────────────────────────────────────────────────────────
# HOW OUTPUT FILES ARE NAMED  ← READ THIS TO UNDERSTAND THE FINAL FILENAME
# ─────────────────────────────────────────────────────────────────────────────
#
# Final filename format:
#   [BANK_SHORT_CODE] _ [CARD_NAME] _ [DOCTYPE] _ [YEAR] .pdf
#
#   BANK_SHORT_CODE  → the KEY from BANK_ALIASES above (e.g. BOB, SCB, HDFC)
#   CARD_NAME        → matched value from CARDS list above (spaces → underscores)
#                      OR "MASTER" if this is a collective/bank-wide document
#   DOCTYPE          → one of: MITC | TNC | BR | LG  (defined in the next section)
#   YEAR             → extracted from PDF text, or DEFAULT_YEAR if not found
#
# Regular card examples:
#   Bank of Baroda  + Eterna            + MITC  + 2024  →  BOB_Eterna_MITC_2024.pdf
#   Standard Chrtd  + Smart             + BR    + 2024  →  SCB_Smart_BR_2024.pdf
#   HDFC            + Tata Neu Infinity + BR    + 2025  →  HDFC_Tata_Neu_Infinity_BR_2025.pdf
#   ICICI           + Amazon Pay        + TNC   + 2023  →  ICICI_Amazon_Pay_TNC_2023.pdf
#
# MASTER (collective) doc examples:
#   HDFC common MITC (all cards)  →  HDFC_MASTER_MITC_2024.pdf
#   SBI collective TNC (all cards) →  SBI_MASTER_TNC_2024.pdf
#
# Output folder structure:
#   processed_docs/
#   ├── BOB_Eterna/
#   │   ├── BOB_Eterna_MITC_2024.pdf
#   │   └── BOB_Eterna_BR_2024.pdf
#   └── HDFC_MASTER/                  ← dedicated folder for collective docs
#       ├── HDFC_MASTER_MITC_2024.pdf
#       └── HDFC_MASTER_TNC_2024.pdf
#
# Edge cases:
#   Spaces in card name   → replaced with underscores  (Amazon Pay → Amazon_Pay)
#   Year not in PDF       → DEFAULT_YEAR value is used (set below)
#   Bank not detected     → UNKNOWN_BANK in filename, file → needs_review/
#   Card not detected     → UNKNOWN_CARD in filename, file → needs_review/
#   Duplicate filename    → second file is skipped, logged as DUPLICATE_SKIPPED
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT TYPE DEFINITIONS  ← THESE BECOME THE "DOCTYPE" PART OF THE FILENAME
# ─────────────────────────────────────────────────────────────────────────────
#
# There are 4 predefined document types.
# Each has a short code used in the filename, title phrases (high-confidence
# signals from the document header), and keywords (scored by frequency).
#
# Short codes used in filenames:
#   MITC  →  e.g. HDFC_Infinia_MITC_2024.pdf
#   TNC   →  e.g. HDFC_Infinia_TNC_2024.pdf
#   BR    →  e.g. HDFC_Infinia_BR_2024.pdf
#   LG    →  e.g. HDFC_Infinia_LG_2024.pdf
# ─────────────────────────────────────────────────────────────────────────────

# ── MITC: Most Important Terms & Conditions ──────────────────────────────────
MITC_KEYWORDS = [
    "interest rate", "annual fee", "finance charge", "late payment fee",
    "minimum amount due", "credit limit", "cash advance fee",
    "overlimit fee", "foreign currency", "billing cycle",
    "statement date", "payment due date", "charges",
    "schedule of charges", "joining fee", "membership fee",
    "rate of interest", "overdue interest",
]
MITC_TITLE_PHRASES = [
    # Confirmed from real debug output — most specific first
    "key fact statement cum most important terms",   # HDFC exact header
    "key fact statement",                            # AMEX KFS docs
    "most important terms and conditions",           # standard MITC header
    "most important terms & conditions",             # ampersand variant
    "most important terms",                          # shorter variant
    "important terms and conditions",                # HSBC variant
    "important information document",                # SCB variant seen in debug
    "mitc",                                          # abbreviation
]

# ── TNC: Terms and Conditions ────────────────────────────────────────────────
TNC_KEYWORDS = [
    "terms and conditions", "cardmember agreement", "cardholder agreement",
    "governing law", "arbitration", "exclusions", "liability",
    "termination", "dispute resolution", "amendments", "jurisdiction",
    "binding", "indemnify", "force majeure",
]
TNC_TITLE_PHRASES = [
    # FIX (Problem 11): Removed plain "terms and conditions" — it appeared as the
    # title of HSBC Live+ welcome offer docs, overriding correct BR detection.
    # Real TNC documents have more specific headers like "Cardmember Agreement".
    # Specific phrases only — avoids false positives on offer/promo docs.
    "cardmember agreement",
    "cardholder agreement",
    "terms of use",
    "client terms",           # SCB SC_General_TNC_Client_2022.pdf
    "credit card terms",      # SCB SC_Smart_TNC_General_2026.pdf
    # NOTE: "terms and conditions" removed — too generic.
    # Docs titled "XYZ Terms and Conditions" often turn out to be BR offer docs.
    # Those will be classified via keyword scoring instead, which is more accurate.
]

# ── BR: Benefits & Rewards ───────────────────────────────────────────────────
# FIX: Added more specific reward-doc signals and removed generic phrases
# that also appear in MITC/TNC documents.
BR_KEYWORDS = [
    "cashback proposition",    # exact phrase in Axis ACE TNC and HDFC Millennia TNC
    "reward points", "welcome bonus", "milestone benefit",
    "accelerated rewards", "redemption", "gift voucher",
    "fuel surcharge waiver", "dining benefit", "earn rate",
    "cashpoints", "welcome benefit", "joining benefit",
    "spends milestone", "spend milestone",
    "welcome gift",            # confirmed from SBI SimplyCLICK welcome doc
    "bookmyshow voucher",      # confirmed from Axis Flipkart welcome doc
    "instant discount",        # confirmed from Axis Flipkart Swiggy promo doc
]
BR_TITLE_PHRASES = [
    "benefits guide",
    "features and benefits",
    "rewards programme",
    "reward catalogue",
    "cashback proposition",    # confirmed from Axis_ACE_TNC_2024.pdf debug
    "cashpoints proposition",  # HDFC Millennia reward doc
    "welcome benefit",         # Axis Flipkart welcome doc
    "welcome gift",            # SBI SimplyCLICK welcome doc
    # FIX (Problem 11): Added offer/promo phrases.
    # HSBC Live+ BR doc header: "Live+ - Terms and Conditions...This Offer is
    # brought to you by HSBC". The word "Offer" distinguishes it from real TNC.
    "offer terms and conditions",  # offer/promo docs titled "XYZ T&C"
    "offer terms & conditions",
    "this offer",              # appears in the header of HSBC welcome docs
    "spends increase",         # Axis ACE promo doc: "Spends Increase via Tap & Pay"
    "tap & pay offer",         # Axis ACE promo doc exact phrase
]

# ── LG: Lounge Guide ─────────────────────────────────────────────────────────
LG_KEYWORDS = [
    "lounge access", "airport lounge", "priority pass", "lounge program",
    "domestic lounge", "international lounge", "complimentary lounge",
    "meet and greet", "fast track immigration",
    "lounge eligibility",      # confirmed from Axis_Master_Lounge_2024.pdf debug
    "participating lounges",   # confirmed from Axis_Master_Lounge_2024.pdf
    "spend criteria",          # Axis lounge: "access based on a spend criteria"
]
LG_TITLE_PHRASES = [
    "lounge access guide",
    "airport lounge programme",
    "lounge benefit",
    "domestic airport lounge access program",   # confirmed from Axis debug text
    "lounge access program",
    "change in lounge access",                  # HDFC Millennia LG doc exact header
]

# ── TO ADD A NEW DOCTYPE (e.g. "FD" for Fee Document): ─────────────────────
#    1. Add  FD_KEYWORDS      = ["fee schedule", "tariff card", ...]
#    2. Add  FD_TITLE_PHRASES = ["schedule of fees and charges", ...]
#    3. Inside detect_doc_type(), add "FD" to both title_hits and kw_data dicts


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION & PIPELINE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # files scoring below this → routed to needs_review/
DEFAULT_YEAR         = "2026" # used in filename when no year is found in PDF
PAGE_TIERS           = [3, 5, 10]  # adaptive reading tiers (pages tried in order)


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT COVERAGE VALIDATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_DOCS = ["MITC", "BR"]  # card gets CRITICAL status if either is missing
OPTIONAL_DOCS = ["TNC", "LG"]   # card gets PARTIAL status if these are missing

# Every card listed here is checked after processing.
# Format: "BANK_SHORT_CODE CardName"
# BANK_SHORT_CODE must match a key in BANK_ALIASES exactly (case-sensitive).
# CardName must match an entry in CARDS exactly (case-sensitive).
EXPECTED_CARDS = [
    # ── Everyday Cashback (15) ────────────────────────────────────────────────
    "SBI Cashback",
    "HDFC Millennia",
    "AXIS ACE",
    "ICICI Amazon Pay",
    "AXIS Flipkart",
    "HSBC Live+",
    "SCB Smart",
    "HDFC MoneyBack+",
    "SBI SimplyCLICK",
    "AMEX Blue Cash",
    "IDFC Millennia",
    "AU Altura",
    "BOB Eterna",
    "KOTAK League Platinum",
    "YES Prosperity",

    # ── Premium & Travel (15) ─────────────────────────────────────────────────
    "HDFC Infinia",
    "HDFC Diners Club Black",
    "AXIS Magnus",
    "AXIS Atlas",
    "AMEX Platinum",
    "AMEX Platinum Travel",
    "ICICI Emeralde",
    "SBI Elite",
    "HDFC Marriott Bonvoy",
    "AXIS Reserve",
    "HSBC Premier",
    "SBI Aurum",
    "AXIS Vistara Infinite",
    "SCB Ultimate",
    "RBL World Safari",

    # ── Lifestyle & Niche (14) ────────────────────────────────────────────────
    "HDFC Tata Neu Infinity",
    "HDFC Swiggy",
    "AXIS Airtel",
    "SBI BPCL Octane",
    "HDFC IndianOil",
    "KOTAK Myntra",
    "SBI Reliance",
    "SBI IRCTC",
    "HDFC Regalia Gold",
    "ICICI HPCL Super Saver",
    "AXIS Select",
    "INDUSIND EazyDiner",
    "SBI Paytm",
    "HDFC Pixel",
    # "OneCard" intentionally excluded — fintech card, no fixed bank partner

    # ── Master / Collective Documents ─────────────────────────────────────────
    # These are bank-level documents that apply to ALL cards for that bank.
    # Each entry tracks whether the bank's collective doc has been processed.
    # Format is "BANK MASTER" — matches what the pipeline produces when
    # detect_master_doc() fires (card name is set to the string "MASTER").
    "HDFC MASTER",      # HDFC common MITC / TNC covering all variants
    "SBI MASTER",       # SBI collective T&C
    "AXIS MASTER",      # Axis Bank master document
    "ICICI MASTER",     # ICICI collective doc
    "AMEX MASTER",      # Amex uniform cardmember agreement
    "SCB MASTER",       # Standard Chartered master T&C
]

# =============================================================================
# END OF USER EDITABLE CONFIGURATION
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP (all relative — works on Windows and Mac without changes)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).resolve().parent   # .../data_pipeline/
PROJECT_ROOT      = SCRIPT_DIR.parent                 # .../project_root/
RAW_DIR           = PROJECT_ROOT / "data" / "raw_docs"
PROCESSED_DIR     = PROJECT_ROOT / "data" / "processed_docs"
REVIEW_DIR        = PROJECT_ROOT / "data" / "needs_review"
LOG_DIR           = PROJECT_ROOT / "data" / "logs"
SUMMARY_CSV           = LOG_DIR / "summary.csv"
DETAIL_LOG            = LOG_DIR / "preprocess_log.txt"
MISSING_DOCS_CSV      = LOG_DIR / "missing_docs_report.csv"
COVERAGE_DASHBOARD    = LOG_DIR / "coverage_dashboard.xlsx"  # visual grid report


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging():
    """Configure console logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_text(pdf_path: Path, max_pages: int) -> str:
    """
    Extract plain text from the first `max_pages` pages of a PDF.
    Tries pdfplumber first; falls back to PyMuPDF (fitz).
    Returns an empty string on failure.
    """
    text = ""

    if pdfplumber:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages[:max_pages]:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"pdfplumber failed on {pdf_path.name}: {e}")

    if fitz:
        try:
            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                text += page.get_text() + "\n"
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF failed on {pdf_path.name}: {e}")

    logger.error(f"No PDF library could extract text from {pdf_path.name}.")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for reliable matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _keyword_score(text_lower: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Count keyword matches; return (normalised score 0–1, matched list)."""
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    if not keywords:
        return 0.0, []
    score = min(len(matched) / max(len(keywords) * 0.3, 1), 1.0)
    return round(score, 3), matched


def _title_match(text_lower: str, phrases: list[str]) -> tuple[bool, str]:
    """Check for title/header phrase in first 500 characters of text."""
    header_zone = text_lower[:500]
    for phrase in phrases:
        if phrase.lower() in header_zone:
            return True, phrase
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DOCUMENT TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_doc_type(text: str) -> dict:
    """
    Two-layer classification:
      Layer 1: title/header phrase match → strong signal, boosts confidence
      Layer 2: keyword frequency scoring → used when no title match found
    Returns {value, confidence, reasons}.
    The value (MITC/TNC/BR/LG) becomes the DOCTYPE part of the filename.
    """
    text_lower = _normalise(text)
    reasons    = []

    # ── Layer 1: title/header match ───────────────────────────────────────────
    title_hits = {
        "MITC": _title_match(text_lower, MITC_TITLE_PHRASES),
        "TNC":  _title_match(text_lower, TNC_TITLE_PHRASES),
        "BR":   _title_match(text_lower, BR_TITLE_PHRASES),
        "LG":   _title_match(text_lower, LG_TITLE_PHRASES),
    }
    title_winner = None
    for doc_type, (found, phrase) in title_hits.items():
        if found:
            title_winner = doc_type
            reasons.append(f'Found title phrase "{phrase}" in document header')
            break

    # ── Layer 2: keyword scoring ──────────────────────────────────────────────
    kw_data = {
        "MITC": _keyword_score(text_lower, MITC_KEYWORDS),
        "TNC":  _keyword_score(text_lower, TNC_KEYWORDS),
        "BR":   _keyword_score(text_lower, BR_KEYWORDS),
        "LG":   _keyword_score(text_lower, LG_KEYWORDS),
    }
    ranked = sorted(kw_data.items(), key=lambda x: x[1][0], reverse=True)
    best_kw_type, (best_kw_score, best_kw_matches) = ranked[0]

    if best_kw_matches:
        reasons.append(
            f"Keyword matches for {best_kw_type}: {', '.join(best_kw_matches[:5])}"
        )
    for dtype, (score, _) in ranked[1:]:
        if score > 0.1:
            reasons.append(f"Low presence of {dtype}-related keywords (score={score})")

    # ── Combine layers ────────────────────────────────────────────────────────
    if title_winner:
        kw_score_for_winner = kw_data[title_winner][0]
        confidence = round(min(0.65 + kw_score_for_winner * 0.35 + 0.10, 1.0), 3)
        final_type = title_winner
    else:
        final_type = best_kw_type
        confidence = round(best_kw_score * 0.85, 3)

    if not reasons:
        reasons.append("No strong signals found; defaulting to best keyword score")

    return {"value": final_type, "confidence": confidence, "reasons": reasons}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BANK DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_bank(text: str, filename: str) -> dict:
    """
    Searches PDF text for any alias phrase defined in BANK_ALIASES.

    IMPORTANT — this is where the short code mapping happens:
      The PDF might say "Bank of Baroda" or "State Bank of India" in full.
      This function maps those full names to their short codes:
        "bank of baroda"      → "BOB"   (used in filename)
        "standard chartered"  → "SCB"   (used in filename)
        "american express"    → "AMEX"  (used in filename)

    The returned "value" is always the SHORT CODE (the dict key),
    never the full bank name — so filenames stay clean and consistent.

    Falls back to searching the filename if text search finds nothing.
    """
    text_lower = _normalise(text)
    fn_lower   = _normalise(filename)
    reasons    = []

    # ── Search PDF text using alias phrases ───────────────────────────────────
    # For each bank, we try all its aliases in order (most specific first).
    # The moment any phrase matches, we return the SHORT CODE for that bank.
    for short_code, aliases in BANK_ALIASES.items():
        for phrase in aliases:
            if phrase.lower() in text_lower:
                in_header  = phrase.lower() in text_lower[:500]
                confidence = 0.95 if in_header else 0.80
                reasons.append(
                    f"Found '{phrase}' → mapped to short code '{short_code}' "
                    f"in {'header' if in_header else 'body'}"
                )
                # Return the SHORT CODE — e.g. "BOB", not "Bank of Baroda"
                # This short code is what ends up in the filename
                return {"value": short_code, "confidence": confidence, "reasons": reasons}

    # ── Fallback: search the original filename ────────────────────────────────
    # Useful when the PDF is scanned (no extractable text) but was manually
    # named, e.g. "bank_of_baroda_eterna_mitc.pdf"
    for short_code, aliases in BANK_ALIASES.items():
        for phrase in aliases:
            if phrase.lower() in fn_lower:
                reasons.append(
                    f"Found '{phrase}' in filename → mapped to '{short_code}' "
                    f"(text extraction may have failed)"
                )
                return {"value": short_code, "confidence": 0.55, "reasons": reasons}

    reasons.append("No known bank name or alias found in text or filename")
    return {"value": None, "confidence": 0.0, "reasons": reasons}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CARD DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Cards that need word-boundary matching to prevent false positives.
# e.g. "ACE" must not match "places", "services", "faces".
# e.g. "Smart" must not match "smartphone", "smartly".
# Add any card name here whose letters appear inside common English words.
WORD_BOUNDARY_CARDS = {
    "ACE", "Smart", "Elite", "Select", "Reserve", "Coral",
    "Signature", "Platinum", "Atlas", "Airtel",
}


def _card_matches(card: str, text_lower: str) -> bool:
    """
    Check whether a card name appears in text.
    - Cards in WORD_BOUNDARY_CARDS use regex word boundaries (\\b) to prevent
      substring false positives (e.g. ACE inside "places").
    - All other cards use simple substring matching.
    """
    card_lower = card.lower()
    if card in WORD_BOUNDARY_CARDS:
        # \\b matches at word boundaries — ACE won't match "places" or "services"
        pattern = r"\b" + re.escape(card_lower) + r"\b"
        return bool(re.search(pattern, text_lower))
    return card_lower in text_lower


def detect_card(text: str, filename: str) -> dict:
    """
    Searches PDF text for card names from the CARDS list.

    MATCHING STRATEGY:
      Pass 1 — Header-first search (first 500 characters only):
        The card name in the document title/header is the most reliable signal.
        A match here gets confidence 0.92 and stops the search immediately.
        This prevents a card name mentioned later in the body from overriding
        the actual card stated in the title.

      Pass 2 — Full text search:
        Scans the entire extracted text. Confidence 0.75.
        Short/ambiguous card names (ACE, Smart, Elite, etc.) use word-boundary
        matching so they don't match substrings inside other words.

      Pass 3 — Filename fallback:
        Used when text extraction produced nothing (scanned PDFs).
        Confidence 0.50.

    WHY THIS ORDER MATTERS:
      The CARDS list has specific names first and generic ones last.
      A document about "Axis ACE Credit Card Cashback Proposition" will match
      "Axis ACE" in the header before the word "cashback" triggers "Cashback"
      in the body — because header pass runs before body pass.
    """
    text_lower = _normalise(text)
    fn_lower   = _normalise(filename)
    header_zone = text_lower[:500]   # first 500 chars = title + opening line
    reasons    = []

    # ── Pass 1: Header zone only (highest confidence) ─────────────────────────
    for card in CARDS:
        if _card_matches(card, header_zone):
            reasons.append(
                f"Found '{card}' in document header (first 500 chars) — "
                f"strong signal, header-zone match"
            )
            return {"value": card, "confidence": 0.92, "reasons": reasons}

    # ── Pass 2: Full text search ───────────────────────────────────────────────
    for card in CARDS:
        if _card_matches(card, text_lower):
            wb_note = " (word-boundary match)" if card in WORD_BOUNDARY_CARDS else ""
            reasons.append(
                f"Found '{card}' in document body{wb_note} — "
                f"not in header, lower confidence"
            )
            return {"value": card, "confidence": 0.75, "reasons": reasons}

    # ── Pass 3: Filename fallback ──────────────────────────────────────────────
    for card in CARDS:
        if _card_matches(card, fn_lower):
            reasons.append(
                f"Found '{card}' in filename — "
                f"text extraction may have failed (scanned PDF?)"
            )
            return {"value": card, "confidence": 0.50, "reasons": reasons}

    reasons.append(
        "No known card name found in header, body, or filename. "
        "Consider adding the card name to the CARDS list if this is unexpected."
    )
    return {"value": None, "confidence": 0.0, "reasons": reasons}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5B — MASTER / COLLECTIVE DOCUMENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_master_doc(text: str) -> dict:
    """
    Determines whether this PDF is a collective/master document that applies
    to ALL of a bank's credit cards, rather than being card-specific.

    SEARCH STRATEGY:
      Pass 1 — Header zone (first 500 chars): confidence 0.95
        The strongest signal. If the document title itself says "applicable to
        all credit card holders", it's unambiguously a master doc.

      Pass 2 — Full text: confidence 0.85
        Some collective signals appear in the opening paragraph rather than
        the exact title line. e.g. "as a part of all Credit Card applications"
        (AMEX MITC) appears in the second sentence, outside the 500-char zone.

    WHY THIS RUNS BEFORE detect_card():
      If we ran detect_card() first on an HDFC collective MITC, it might
      match "Infinia" or "Millennia" mentioned anywhere in the text, causing
      a wrong card-specific classification. Master detection short-circuits
      that mistake by checking for collective signals first.

    RETURN VALUE:
      { is_master: bool, signal: str|None, confidence: float, found_in: str|None }
    """
    text_lower  = _normalise(text)
    header_zone = text_lower[:500]

    # ── Pass 1: Header zone (strongest signal) ────────────────────────────────
    for signal in MASTER_DOC_SIGNALS:
        if signal.lower() in header_zone:
            return {
                "is_master":  True,
                "signal":     signal,
                "confidence": 0.95,
                "found_in":   "header",
            }

    # ── Pass 2: Full text (opening paragraph may contain the signal) ──────────
    for signal in MASTER_DOC_SIGNALS:
        if signal.lower() in text_lower:
            return {
                "is_master":  True,
                "signal":     signal,
                "confidence": 0.85,
                "found_in":   "body",
            }

    return {"is_master": False, "signal": None, "confidence": 0.0, "found_in": None}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — YEAR DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_year(text: str) -> str:
    """
    Extracts the document year from the PDF text.

    STRATEGY (three-pass, most reliable first):

    Pass 1 — Header zone (first 500 chars):
      The document date is almost always near the title or opening line.
      e.g. "Updated on 2nd Feb'26", "The MITC update as on 5 January 2026",
           "Published on 25th Apr 2025".
      If a year is found here, return it immediately — this is the most
      accurate signal and avoids confusion with historical dates in the body.

    Pass 2 — Most recent year in full text:
      Many documents reference old dates in their body text:
        - Effective dates of past policy changes ("w.e.f. April 2024")
        - Copyright footers ("© 2015")
        - Previous revision dates ("last updated 2011")
      The MOST RECENT year found is the best proxy for when the doc was issued,
      since old years are references and the current year is the actual date.

    Pass 3 — DEFAULT_YEAR fallback:
      Used when no year at all is found in the text.

    FIX NOTE:
      Previous version used the FIRST year found, which caused issues like
      HDFC_General_MITC_2026.pdf being named _2005 (a year in a fees table),
      and AMEX_Master_TNC picking up 2011 from a legal clause.
    """
    year_pattern = r"\b(20[0-2]\d)\b"

    # ── Pass 1: Header zone first ─────────────────────────────────────────────
    header = _normalise(text[:500])
    header_matches = re.findall(year_pattern, header)
    if header_matches:
        year = header_matches[0]  # first year in the header = document date
        return year

    # ── Pass 2: Most recent year in full text ─────────────────────────────────
    full_matches = re.findall(year_pattern, _normalise(text))
    if full_matches:
        year = max(full_matches)  # most recent year = best proxy for doc date
        return year

    # ── Pass 3: Default ───────────────────────────────────────────────────────
    return DEFAULT_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — OVERALL CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────
def compute_confidence(doc_type_result: dict, bank_result: dict,
                        card_result: dict,
                        master_result: dict | None = None) -> float:
    """
    Weighted average of the three individual detection confidence scores.
    Doc type carries the most weight (it's the most reliably detectable).

    MASTER DOC FLOOR (Fix for Problem 9):
      When a master signal is positively detected (is_master=True), the card
      confidence is already set to 0.85–0.95 from the master signal strength.
      However the DOC TYPE score can still be low if the text is garbled
      (e.g. SBI_General_MITC_2026.pdf — fee table text extracted as fragments).
      In that case the weighted average can drop below the 0.70 threshold even
      though we are confident it's a master doc.

      Fix: if master detection fired, enforce a minimum overall confidence of
      0.72 — just above the threshold — so a correctly identified master doc
      never gets routed to needs_review purely due to garbled body text.
      The bank must still be detected for the floor to apply (safety check).
    """
    score = round(
        doc_type_result["confidence"] * 0.50 +
        bank_result["confidence"]     * 0.30 +
        card_result["confidence"]     * 0.20,
        3
    )

    # Apply master doc confidence floor
    if (master_result and master_result.get("is_master")
            and bank_result.get("value") is not None):
        # Only apply floor if bank was also found — prevents a totally unknown
        # document from being force-passed just because it contained a phrase.
        score = max(score, 0.72)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — ADAPTIVE PAGE READING + DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def run_detection_with_fallback(pdf_path: Path, debug: bool):
    """
    Adaptive strategy — reads more pages only when needed:
      Tier 1: PAGE_TIERS[0] pages → detect → if confidence >= threshold, STOP
      Tier 2: PAGE_TIERS[1] pages → detect → if confidence >= threshold, STOP
      Tier 3: PAGE_TIERS[2] pages → final attempt, always stops here

    Detection order per tier:
      1. detect_doc_type()   — classify MITC / TNC / BR / LG
      2. detect_bank()       — identify bank, map to short code
      3. detect_master_doc() — check if this is a collective/bank-wide doc
         ↳ if MASTER → card = "MASTER", skip detect_card()
         ↳ if not    → detect_card() runs normally
      4. compute_confidence() — weighted average of all three scores

    Returns:
      (text, doc_type_result, bank_result, card_result, master_result, year, confidence)
      master_result is included so the main pipeline and logs can record
      whether this file was treated as a collective document.
    """
    filename           = pdf_path.name
    text               = ""
    doc_type_result = bank_result = card_result = master_result = None
    overall_confidence = 0.0

    for tier_idx, max_pages in enumerate(PAGE_TIERS):
        logger.info(f"Extracting text ({max_pages} pages) from: {filename}")
        text = extract_text(pdf_path, max_pages)

        if not text.strip():
            logger.warning(f"No text extracted from {filename} — skipping further tiers.")
            break

        if debug:
            logger.info(f"[DEBUG] Text sample ({max_pages} pages): "
                        f"{text[:600].replace(chr(10), ' ')}")

        doc_type_result = detect_doc_type(text)
        bank_result     = detect_bank(text, filename)
        year            = detect_year(text)

        # ── MASTER DOC CHECK — runs BEFORE individual card detection ──────────
        # Reason: a collective doc may mention many card names in its body
        # (e.g. "applies to Infinia, Millennia, Regalia..."). If we ran
        # detect_card() first, it would wrongly latch onto one of those names.
        # By checking for master signals first, we skip card detection entirely
        # for collective docs and assign the synthetic "MASTER" card name.
        master_result = detect_master_doc(text)

        if master_result["is_master"]:
            # ── Collective document path ──────────────────────────────────────
            # Override card detection with the synthetic "MASTER" value.
            # This is what produces HDFC_MASTER_MITC_2024.pdf in HDFC_MASTER/.
            card_result = {
                "value":      "MASTER",
                "confidence": master_result["confidence"],
                "reasons": [
                    "MASTER DOCUMENT — applies to all cards for this bank",
                    f'Trigger signal: "{master_result["signal"]}"',
                    f'Signal found in: {master_result.get("found_in", "document")} '
                    f'(conf={master_result["confidence"]})',
                    "Individual card detection was intentionally skipped",
                    f'Output will be routed to processed_docs/{bank_result["value"] or "UNKNOWN"}_MASTER/',
                ],
            }
            logger.info(
                f'[MASTER DOC] Collective document detected — '
                f'signal: "{master_result["signal"]}" | '
                f'card set to MASTER (conf={master_result["confidence"]})'
            )
        else:
            # ── Normal card-specific document path ────────────────────────────
            # Master check passed with no signal → run standard card detection.
            card_result = detect_card(text, filename)

        overall_confidence = compute_confidence(doc_type_result, bank_result, card_result, master_result)

        logger.info(
            f"Confidence after {max_pages} pages: {overall_confidence:.2f} "
            f"| {bank_result['value']} | {card_result['value']} | {doc_type_result['value']}"
        )

        if overall_confidence >= CONFIDENCE_THRESHOLD:
            if tier_idx > 0:
                logger.info(
                    f"Confidence improved to {overall_confidence:.2f} "
                    f"after reading {max_pages} pages."
                )
            break
        elif tier_idx < len(PAGE_TIERS) - 1:
            logger.info(
                f"Confidence {overall_confidence:.2f} below threshold "
                f"{CONFIDENCE_THRESHOLD}. Retrying with {PAGE_TIERS[tier_idx+1]} pages..."
            )
        else:
            logger.info(
                f"Reached maximum page limit ({max_pages}). "
                f"Final confidence: {overall_confidence:.2f}."
            )

    # Safe defaults if extraction failed entirely
    if doc_type_result is None:
        doc_type_result = {"value": "UNKNOWN", "confidence": 0.0,
                           "reasons": ["Text extraction failed"]}
    if bank_result is None:
        bank_result = {"value": None, "confidence": 0.0,
                       "reasons": ["Text extraction failed"]}
    if card_result is None:
        card_result = {"value": None, "confidence": 0.0,
                       "reasons": ["Text extraction failed"]}
    if master_result is None:
        master_result = {"is_master": False, "signal": None, "confidence": 0.0}

    year = detect_year(text) if text.strip() else DEFAULT_YEAR
    return text, doc_type_result, bank_result, card_result, master_result, year, overall_confidence


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — FILENAME + FOLDER GENERATION
# ─────────────────────────────────────────────────────────────────────────────
#
# This is where detection results are assembled into the actual output path.
#
# Data flow into the filename:
#   detect_bank()     → "BOB"              ← short code from BANK_ALIASES key
#   detect_card()     → "Eterna"           ← value from CARDS list
#   detect_doc_type() → "MITC"             ← MITC / TNC / BR / LG
#   detect_year()     → "2024"             ← from PDF text or DEFAULT_YEAR
#
#   generate_filename() assembles:
#     "BOB" + "Eterna" + "MITC" + "2024"  →  BOB_Eterna_MITC_2024.pdf
#
#   get_output_folder() builds the subfolder:
#     "BOB" + "Eterna"  →  processed_docs/BOB_Eterna/
# ─────────────────────────────────────────────────────────────────────────────

def _safe(value: str, fallback: str = "UNKNOWN") -> str:
    """
    Convert a string to a clean, filesystem-safe format for use in filenames
    and folder names.

    RULES:
      1. Replace "+" with "Plus" — preserves meaning of MoneyBack+, Live+, etc.
         Without this, "MoneyBack+" → "MoneyBack_" (trailing underscore, ugly).
      2. Replace any other non-alphanumeric character (space, @, /, etc.) with "_".
      3. Collapse multiple consecutive underscores into one.
      4. Strip leading/trailing underscores.

    EXAMPLES (before → after):
      "MoneyBack+"      →  "MoneyBackPlus"     (not "MoneyBack_")
      "Live+"           →  "LivePlus"          (not "Live_")
      "Amazon Pay ICICI"→  "Amazon_Pay_ICICI"
      "Cashback SBI"    →  "Cashback_SBI"
      None              →  fallback value
    """
    if not value:
        return fallback
    # Step 1: replace "+" with "Plus" before any other substitution
    result = str(value).replace("+", "Plus")
    # Step 2: replace all remaining non-alphanumeric/non-hyphen chars with "_"
    result = re.sub(r"[^\w\-]", "_", result)
    # Step 3: collapse multiple underscores (e.g. "__" → "_")
    result = re.sub(r"_+", "_", result)
    # Step 4: strip leading/trailing underscores
    result = result.strip("_")
    return result if result else fallback


def generate_filename(bank: str | None, card: str | None,
                       doc_type: str, year: str) -> str:
    """
    Assembles the four detected values into the final output filename.

    Format:  [BANK]_[CARD]_[DOCTYPE]_[YEAR].pdf

    bank     → short code from detect_bank()     e.g. "BOB"    (NOT "Bank of Baroda")
    card     → name from detect_card()           e.g. "Eterna"
    doc_type → code from detect_doc_type()       e.g. "MITC"
    year     → string from detect_year()         e.g. "2024"

    Result:  BOB_Eterna_MITC_2024.pdf
    """
    b = _safe(bank, "UNKNOWN_BANK")   # e.g. "BOB"
    c = _safe(card, "UNKNOWN_CARD")   # e.g. "Eterna" or "Amazon_Pay"
    d = _safe(doc_type, "UNKNOWN")    # e.g. "MITC"
    return f"{b}_{c}_{d}_{year}.pdf"  # → BOB_Eterna_MITC_2024.pdf


def get_output_folder(bank: str | None, card: str | None, base_dir: Path) -> Path:
    """
    Builds the subfolder path where the renamed file will be saved.

    Format:  processed_docs/[BANK]_[CARD]/

    Example:
      bank="BOB", card="Eterna"
      →  processed_docs/BOB_Eterna/

    Creates the folder automatically if it doesn't exist.
    """
    b      = _safe(bank, "UNKNOWN_BANK")   # e.g. "BOB"
    c      = _safe(card, "UNKNOWN_CARD")   # e.g. "Eterna"
    folder = base_dir / f"{b}_{c}"        # e.g. processed_docs/BOB_Eterna
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def is_duplicate(dest_path: Path) -> bool:
    """Return True if a file with the same output name already exists."""
    return dest_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — FILE COPY
# ─────────────────────────────────────────────────────────────────────────────
def move_file(src: Path, dest: Path, dry_run: bool) -> None:
    """
    Copies (does NOT delete) the source file to its destination.
    Raw files in raw_docs/ are NEVER modified or removed.
    In dry-run mode, only prints what would happen — no files are touched.
    """
    if dry_run:
        logger.info(f"[DRY-RUN] Would copy: {src.name} → {dest}")
        return
    try:
        shutil.copy2(str(src), str(dest))
        logger.info(f"Copied: {src.name} → {dest}")
    except Exception as e:
        logger.error(f"Failed to copy {src.name}: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — DETAILED LOG WRITER
# ─────────────────────────────────────────────────────────────────────────────
def write_detail_log(entries: list[dict]) -> None:
    """Write a human-readable per-file audit trail to preprocess_log.txt."""
    with open(DETAIL_LOG, "w", encoding="utf-8") as f:
        f.write(f"PREPROCESSING LOG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for entry in entries:
            f.write("=" * 30 + "\n")
            f.write(f"FILE: {entry['filename']}\n\n")

            # ── Bank ──────────────────────────────────────────────────────────
            f.write(f"BANK: {entry['bank']} ({entry['bank_conf']:.2f})\n")
            f.write("REASONS:\n")
            for r in entry["bank_reasons"]:
                f.write(f"  • {r}\n")
            f.write("\n")

            # ── Master doc decision — logged BEFORE card so the reader sees
            #    exactly why card detection was or wasn't skipped ──────────────
            if entry.get("is_master"):
                f.write("MASTER / COLLECTIVE DOCUMENT: YES\n")
                f.write(f"  Trigger signal : \"{entry['master_signal']}\"\n")
                f.write(f"  Confidence     : {entry['master_conf']:.2f}\n")
                f.write("  Decision       : Individual card detection was SKIPPED.\n")
                f.write("                   Card name set to MASTER.\n")
                f.write(f"  Destination    : processed_docs/{entry['bank']}_MASTER/\n")
            else:
                f.write("MASTER / COLLECTIVE DOCUMENT: NO\n")
                f.write("  No collective signals found — proceeded with individual card detection.\n")
            f.write("\n")

            # ── Card ──────────────────────────────────────────────────────────
            f.write(f"CARD: {entry['card']} ({entry['card_conf']:.2f})\n")
            f.write("REASONS:\n")
            for r in entry["card_reasons"]:
                f.write(f"  • {r}\n")
            f.write("\n")

            # ── Doc type ──────────────────────────────────────────────────────
            f.write(f"DOC TYPE: {entry['doc_type']} ({entry['doc_type_conf']:.2f})\n")
            f.write("REASONS:\n")
            for r in entry["doc_type_reasons"]:
                f.write(f"  • {r}\n")
            f.write("\n")

            f.write(f"PAGES READ: {entry.get('pages_read', 'N/A')}\n")
            f.write(f"STATUS: {entry['status']}\n\n")

    logger.info(f"Detailed log written to: {DETAIL_LOG}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — CSV SUMMARY WRITER
# ─────────────────────────────────────────────────────────────────────────────
def write_summary_csv(entries: list[dict]) -> None:
    """Write summary.csv. Falls back to built-in csv if pandas not installed."""
    fieldnames = ["File Name", "Bank", "Card", "Is_Master", "DocType",
                  "Confidence", "Reason", "Status"]
    rows = []
    for e in entries:
        combined_reason = " | ".join(
            e["doc_type_reasons"][:2] + e["bank_reasons"][:1] + e["card_reasons"][:1]
        )
        rows.append({
            "File Name":  e["filename"],
            "Bank":       e["bank"] or "NOT FOUND",
            "Card":       e["card"] or "NOT FOUND",
            # Is_Master column makes it easy to filter collective docs in Excel
            "Is_Master":  "YES" if e.get("is_master") else "NO",
            "DocType":    e["doc_type"],
            "Confidence": f"{e['overall_conf']:.2f}",
            "Reason":     combined_reason,
            "Status":     e["status"],
        })
    if pd:
        pd.DataFrame(rows, columns=fieldnames).to_csv(SUMMARY_CSV, index=False)
    else:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    logger.info(f"Summary CSV written to: {SUMMARY_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — DOCUMENT COVERAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def build_coverage_map(log_entries: list[dict]) -> dict[str, set[str]]:
    """
    Builds { "BANK Card" → {doc types found} } from successful entries only.
    The key format "BANK Card" must match entries in EXPECTED_CARDS.
    """
    coverage: dict[str, set[str]] = {}
    for entry in log_entries:
        if entry["status"] not in ("SUCCESS", "DUPLICATE_SKIPPED"):
            continue
        bank     = entry.get("bank")
        card     = entry.get("card")
        doc_type = entry.get("doc_type")
        if not bank or not card or not doc_type:
            continue
        key = f"{bank} {card}"
        coverage.setdefault(key, set()).add(doc_type)
    return coverage


def validate_coverage(coverage_map: dict[str, set[str]]) -> list[dict]:
    """
    Compares coverage_map against EXPECTED_CARDS and assigns status:
      COMPLETE   → all required + optional docs present
      PARTIAL    → required docs OK, some optional missing
      CRITICAL   → one or more required docs missing
      NOT_FOUND  → no documents found for this card at all
    """
    results = []
    for expected_card in EXPECTED_CARDS:
        present_docs     = coverage_map.get(expected_card, set())
        missing_required = [d for d in REQUIRED_DOCS if d not in present_docs]
        missing_optional = [d for d in OPTIONAL_DOCS if d not in present_docs]
        all_missing      = missing_required + missing_optional

        if not present_docs:
            status = "NOT_FOUND"
        elif missing_required:
            status = "CRITICAL"
        elif missing_optional:
            status = "PARTIAL"
        else:
            status = "COMPLETE"

        if status == "NOT_FOUND":
            logger.warning(f"No documents found at all for: {expected_card}")
        elif status == "CRITICAL":
            for m in missing_required:
                logger.warning(f"Missing {m} for {expected_card}")
        elif status == "PARTIAL":
            logger.warning(
                f"Only [{', '.join(sorted(present_docs))}] found for {expected_card} "
                f"— optional docs missing: {', '.join(missing_optional)}"
            )

        results.append({
            "card":         expected_card,
            "present_docs": sorted(present_docs),
            "missing_docs": all_missing,
            "status":       status,
        })
    return results


def write_missing_docs_csv(validation_results: list[dict]) -> None:
    """Write missing_docs_report.csv to /data/logs/."""
    fieldnames = ["Card", "Present_Docs", "Missing_Docs", "Status"]
    rows = [{
        "Card":         r["card"],
        "Present_Docs": ", ".join(r["present_docs"]) or "NONE",
        "Missing_Docs": ", ".join(r["missing_docs"]) or "NONE",
        "Status":       r["status"],
    } for r in validation_results]

    if pd:
        pd.DataFrame(rows, columns=fieldnames).to_csv(MISSING_DOCS_CSV, index=False)
    else:
        with open(MISSING_DOCS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    logger.info(f"Missing docs report written to: {MISSING_DOCS_CSV}")


def write_coverage_dashboard(validation_results: list[dict]) -> None:
    """
    Write a visual Excel dashboard to /data/logs/coverage_dashboard.xlsx.

    WHAT IT LOOKS LIKE:
      Each row = one expected card
      Columns  = Card | MITC | TNC | BR | LG | Overall Status

      Cell values:
        ✅  = document found and processed successfully
        ❌  = document MISSING (required doc type)
        ⚪  = document missing (optional doc type — less critical)
        (empty) = not applicable

      The Status column uses colour-coded text:
        COMPLETE   → green
        PARTIAL    → orange
        CRITICAL   → red
        NOT_FOUND  → dark red / bold

      A summary row at the bottom counts each status type.

    WHY THIS IS USEFUL:
      The missing_docs_report.csv tells you the same information, but you have
      to read each row individually. This dashboard lets you scan the whole
      matrix at once — at a glance you can see which cards have gaps,
      which doc types are most commonly missing, and which banks need attention.

    REQUIRES: pandas + openpyxl
      Install with:  pip install pandas openpyxl
      If openpyxl is not installed, falls back to writing a plain CSV with
      the same grid layout (still useful, just without colour formatting).
    """
    # All doc types in display order — required first, then optional
    all_doc_types = REQUIRED_DOCS + OPTIONAL_DOCS  # e.g. ["MITC", "BR", "TNC", "LG"]

    # ── Build the grid rows ───────────────────────────────────────────────────
    rows = []
    for r in validation_results:
        row = {"Card": r["card"]}

        for doc_type in all_doc_types:
            if doc_type in r["present_docs"]:
                row[doc_type] = "✅ Found"
            elif doc_type in REQUIRED_DOCS:
                row[doc_type] = "❌ MISSING"    # required — serious gap
            else:
                row[doc_type] = "⚪ Missing"    # optional — less critical

        row["Status"] = r["status"]
        rows.append(row)

    # Column order for the sheet
    columns = ["Card"] + all_doc_types + ["Status"]

    # ── Try to write a styled Excel file ─────────────────────────────────────
    try:
        import openpyxl
        from openpyxl.styles import (
            PatternFill, Font, Alignment, Border, Side
        )
        from openpyxl.utils import get_column_letter

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Coverage Dashboard"

        # ── Colour palette ────────────────────────────────────────────────────
        # Status row background colours
        STATUS_FILL = {
            "COMPLETE":  PatternFill("solid", fgColor="C6EFCE"),  # green
            "PARTIAL":   PatternFill("solid", fgColor="FFEB9C"),  # yellow
            "CRITICAL":  PatternFill("solid", fgColor="FFC7CE"),  # red
            "NOT_FOUND": PatternFill("solid", fgColor="E8B0B5"),  # dark red
        }
        STATUS_FONT = {
            "COMPLETE":  Font(bold=True, color="276221"),
            "PARTIAL":   Font(bold=True, color="9C6500"),
            "CRITICAL":  Font(bold=True, color="9C0006"),
            "NOT_FOUND": Font(bold=True, color="6B0000"),
        }

        # Cell fill colours for each doc type value
        CELL_FILL = {
            "✅ Found":   PatternFill("solid", fgColor="C6EFCE"),  # green
            "❌ MISSING": PatternFill("solid", fgColor="FFC7CE"),  # red
            "⚪ Missing": PatternFill("solid", fgColor="FFEB9C"),  # yellow
        }
        CELL_FONT = {
            "✅ Found":   Font(color="276221"),
            "❌ MISSING": Font(bold=True, color="9C0006"),
            "⚪ Missing": Font(color="9C6500"),
        }

        # ── Header row ────────────────────────────────────────────────────────
        header_fill = PatternFill("solid", fgColor="1F4E79")   # dark blue
        header_font = Font(bold=True, color="FFFFFF", size=11)
        thin_border = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin"),
        )

        for col_idx, col_name in enumerate(columns, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.fill    = header_fill
            cell.font    = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center",
                                       wrap_text=True)
            cell.border  = thin_border

        ws.row_dimensions[1].height = 28

        # ── Data rows ─────────────────────────────────────────────────────────
        for row_idx, row_data in enumerate(rows, start=2):
            status = row_data["Status"]

            for col_idx, col_name in enumerate(columns, start=1):
                value = row_data.get(col_name, "")
                cell  = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border    = thin_border

                if col_name == "Card":
                    # Card name: left-aligned, coloured by row status
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                    cell.fill = STATUS_FILL.get(status, PatternFill())
                    cell.font = STATUS_FONT.get(status, Font())

                elif col_name == "Status":
                    cell.fill = STATUS_FILL.get(status, PatternFill())
                    cell.font = STATUS_FONT.get(status, Font())

                elif col_name in all_doc_types:
                    # Doc type cell: colour by found/missing
                    cell.fill = CELL_FILL.get(value, PatternFill())
                    cell.font = CELL_FONT.get(value, Font())

            ws.row_dimensions[row_idx].height = 20

        # ── Summary row at the bottom ─────────────────────────────────────────
        summary_row = len(rows) + 3   # one blank row gap

        counts = {"COMPLETE": 0, "PARTIAL": 0, "CRITICAL": 0, "NOT_FOUND": 0}
        for r in validation_results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1

        summary_fill = PatternFill("solid", fgColor="D9E1F2")   # light blue
        summary_font = Font(bold=True, color="1F4E79")

        ws.cell(row=summary_row, column=1, value="SUMMARY").font = summary_font
        ws.cell(row=summary_row, column=1).fill = summary_fill

        summary_data = [
            ("✅ COMPLETE",   counts["COMPLETE"],  "C6EFCE", "276221"),
            ("⚠️ PARTIAL",    counts["PARTIAL"],   "FFEB9C", "9C6500"),
            ("❌ CRITICAL",   counts["CRITICAL"],  "FFC7CE", "9C0006"),
            ("🔍 NOT FOUND",  counts["NOT_FOUND"], "E8B0B5", "6B0000"),
        ]
        for col_offset, (label, count, bg, fg) in enumerate(summary_data):
            label_col = 2 + col_offset * 2
            count_col = label_col + 1
            lc = ws.cell(row=summary_row, column=label_col, value=label)
            cc = ws.cell(row=summary_row, column=count_col, value=count)
            for cell in (lc, cc):
                cell.fill      = PatternFill("solid", fgColor=bg)
                cell.font      = Font(bold=True, color=fg)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border    = thin_border

        # ── Column widths ─────────────────────────────────────────────────────
        ws.column_dimensions["A"].width = 30   # Card name
        for col_idx in range(2, len(columns)):
            ws.column_dimensions[get_column_letter(col_idx)].width = 14
        ws.column_dimensions[get_column_letter(len(columns))].width = 14  # Status

        # ── Freeze top row so header stays visible when scrolling ─────────────
        ws.freeze_panes = "A2"

        wb.save(str(COVERAGE_DASHBOARD))
        logger.info(f"Coverage dashboard (Excel) written to: {COVERAGE_DASHBOARD}")

    except ImportError:
        # ── Fallback: plain CSV grid (no colours, but same data layout) ───────
        logger.warning(
            "openpyxl not installed — writing plain CSV dashboard instead. "
            "Install with:  pip install openpyxl"
        )
        fallback_path = LOG_DIR / "coverage_dashboard.csv"
        if pd:
            pd.DataFrame(rows, columns=columns).to_csv(fallback_path, index=False)
        else:
            with open(fallback_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        logger.info(f"Coverage dashboard (CSV fallback) written to: {fallback_path}")


def print_validation_summary(validation_results: list[dict]) -> None:
    """Print validation summary to console. CRITICAL cards are shown first."""
    counts = {"COMPLETE": 0, "PARTIAL": 0, "CRITICAL": 0, "NOT_FOUND": 0}
    for r in validation_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    logger.info("")
    logger.info("=" * 50)
    logger.info("DOCUMENT COVERAGE VALIDATION REPORT")
    logger.info(f"  Total expected cards : {len(validation_results)}")
    logger.info(f"  COMPLETE             : {counts['COMPLETE']}")
    logger.info(f"  PARTIAL              : {counts['PARTIAL']}")
    logger.info(f"  CRITICAL             : {counts['CRITICAL']}")
    logger.info(f"  NOT FOUND            : {counts['NOT_FOUND']}")
    logger.info("=" * 50)

    critical = [r for r in validation_results if r["status"] in ("CRITICAL", "NOT_FOUND")]
    if critical:
        logger.info("CRITICAL ISSUES:")
        for r in critical:
            missing_str = ", ".join(r["missing_docs"]) or "all docs"
            logger.warning(f"  [{r['status']}] {r['card']} — missing: {missing_str}")

    partial = [r for r in validation_results if r["status"] == "PARTIAL"]
    if partial:
        logger.info("PARTIAL COVERAGE:")
        for r in partial:
            logger.info(
                f"  [PARTIAL] {r['card']} — present: {', '.join(r['present_docs'])} "
                f"| missing optional: {', '.join(r['missing_docs'])}"
            )
    logger.info("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_all(dry_run: bool = False, debug: bool = False) -> None:
    """
    Full pipeline:
      1. Find all PDFs in raw_docs/
      2. For each: adaptive text extraction → detect bank/card/type/year
      3. Generate renamed file path and copy to processed_docs/ or needs_review/
      4. Write logs, summary CSV, and coverage validation report
    """
    for d in [RAW_DIR, PROCESSED_DIR, REVIEW_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    total     = len(pdf_files)

    if total == 0:
        logger.warning(f"No PDF files found in {RAW_DIR}. Exiting.")
        return

    logger.info(f"Found {total} PDF file(s) in {RAW_DIR}")

    count_processed = count_review = count_errors = count_skipped = 0
    log_entries: list[dict] = []

    for idx, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"[PROCESSING] {idx}/{total}: {pdf_path.name}")
        pages_read = PAGE_TIERS[0]

        try:
            (text, doc_type_result, bank_result,
             card_result, master_result, year, overall_conf) = run_detection_with_fallback(pdf_path, debug)

            # Track which tier was ultimately needed
            for tier in PAGE_TIERS:
                pages_read = tier
                if overall_conf >= CONFIDENCE_THRESHOLD:
                    break

            bank     = bank_result["value"]     # short code, e.g. "BOB" or "HDFC"
            card     = card_result["value"]     # card name, e.g. "Eterna" or "MASTER"
            doc_type = doc_type_result["value"] # e.g. "MITC"
            is_master = master_result["is_master"]  # True if collective doc

            # ── Log a clear summary line ──────────────────────────────────────
            if is_master:
                logger.info(
                    f"Detected: {bank} | [MASTER/COLLECTIVE DOC] | {doc_type} "
                    f"(conf={overall_conf:.2f}) — will be placed in {bank}_MASTER/"
                )
            else:
                logger.info(
                    f"Detected: {bank} | {card} | {doc_type} (conf={overall_conf:.2f})"
                )

            needs_review = (
                overall_conf < CONFIDENCE_THRESHOLD or bank is None or card is None
            )

            # ── Build the output filename and folder ──────────────────────────
            # For regular docs:  BOB_Eterna_MITC_2024.pdf  → processed_docs/BOB_Eterna/
            # For master docs:   HDFC_MASTER_MITC_2024.pdf → processed_docs/HDFC_MASTER/
            new_filename = generate_filename(bank, card, doc_type, year)

            if needs_review:
                dest_path = REVIEW_DIR / new_filename
                status    = "NEEDS_REVIEW"
            else:
                dest_dir  = get_output_folder(bank, card, PROCESSED_DIR)
                dest_path = dest_dir / new_filename
                status    = "MASTER_DOC" if is_master else "SUCCESS"

            if is_duplicate(dest_path):
                logger.info(f"Duplicate detected — skipping: {new_filename}")
                status = "DUPLICATE_SKIPPED"
                count_skipped += 1
            else:
                move_file(pdf_path, dest_path, dry_run)
                count_review    += 1 if needs_review else 0
                count_processed += 0 if needs_review else 1

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            status          = "ERROR"
            doc_type_result = {"value": "ERROR", "confidence": 0.0, "reasons": [str(e)]}
            bank_result     = {"value": None,    "confidence": 0.0, "reasons": []}
            card_result     = {"value": None,    "confidence": 0.0, "reasons": []}
            master_result   = {"is_master": False, "signal": None, "confidence": 0.0}
            overall_conf    = 0.0
            is_master       = False
            count_errors   += 1

        log_entries.append({
            "filename":         pdf_path.name,
            "bank":             bank_result["value"],
            "bank_conf":        bank_result["confidence"],
            "bank_reasons":     bank_result["reasons"],
            "card":             card_result["value"],
            "card_conf":        card_result["confidence"],
            "card_reasons":     card_result["reasons"],
            "doc_type":         doc_type_result["value"],
            "doc_type_conf":    doc_type_result["confidence"],
            "doc_type_reasons": doc_type_result["reasons"],
            # ── Master doc fields — included so logs show the full decision trail
            "is_master":        master_result["is_master"],
            "master_signal":    master_result["signal"],
            "master_conf":      master_result["confidence"],
            # ── ─────────────────────────────────────────────────────────────────
            "overall_conf":     overall_conf,
            "pages_read":       pages_read,
            "status":           status,
        })

    write_detail_log(log_entries)
    write_summary_csv(log_entries)

    logger.info("")
    logger.info("Running document coverage validation...")
    coverage_map       = build_coverage_map(log_entries)
    validation_results = validate_coverage(coverage_map)
    write_missing_docs_csv(validation_results)
    write_coverage_dashboard(validation_results)   # ← visual Excel grid
    print_validation_summary(validation_results)

    logger.info("")
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total files      : {total}")
    logger.info(f"  Processed        : {count_processed}")
    logger.info(f"  Master/Collective: {sum(1 for e in log_entries if e.get('is_master') and e['status'] in ('MASTER_DOC', 'SUCCESS'))}")
    logger.info(f"  Needs review     : {count_review}")
    logger.info(f"  Errors           : {count_errors}")
    logger.info(f"  Duplicates       : {count_skipped}")
    logger.info(f"  Dry-run mode     : {'ON' if dry_run else 'OFF'}")
    logger.info("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Credit Card PDF Preprocessing Pipeline"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate processing without copying any files.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print sample extracted text for each PDF.",
    )
    args = parser.parse_args()

    setup_logging()

    if not (pdfplumber or fitz):
        logger.error(
            "No PDF library found. Install pdfplumber or PyMuPDF:\n"
            "  pip install pdfplumber\n"
            "  pip install pymupdf"
        )
        sys.exit(1)

    process_all(dry_run=args.dry_run, debug=args.debug)


if __name__ == "__main__":
    main()