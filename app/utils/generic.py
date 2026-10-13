"""Generic shared helpers used across services."""
from typing import Any, Optional


def normalize_theme(value: Optional[Any]) -> Optional[str]:
    """Strip leading and trailing whitespace only; None for a missing or blank value"""
    if value is None:
        return None
    # A theme is a unique code from Commons, so inner spacing and case are kept
    # exactly as sent; upload and search filters must see the same string.
    stripped = str(value).strip()
    return stripped or None
