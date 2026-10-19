"""Generic shared helpers used across services."""
from typing import Any, Optional


def normalize_theme(value: Optional[Any]) -> Optional[str]:
    """Strip and collapse inner whitespace; None for a missing or blank value"""
    if value is None:
        return None
    # No case-folding: themes are stored and filtered exactly as sent,
    # like categories and organizations.
    collapsed = " ".join(str(value).split())
    return collapsed or None
