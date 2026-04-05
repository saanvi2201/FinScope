# =============================================================================
# __init__.py — Behavioral Agent Module
# =============================================================================
# Exposes the public API of the behavior_agent module.
# Downstream agents should import ONLY from here.
#
# Usage:
#   from behavior_agent import generate_user_profile
#
# ⚠️ IMPORTANT: Changing category names will break Simulation Agent
# =============================================================================

from .main import generate_user_profile
from .defaults import CATEGORIES

__all__ = [
    "generate_user_profile",
    "CATEGORIES",
]

# Version tag for integration tracking
__version__ = "1.0.0"
