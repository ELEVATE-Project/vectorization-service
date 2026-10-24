"""Validation helpers shared across the service."""
import secrets
from typing import Optional

from app.config import settings
from app.constants import constants as const
from app.constants import messages as msg


def secret_matches(provided: Optional[str], configured: Optional[str]) -> bool:
    """True when both are set and equal; constant-time, and safe for non-ASCII bytes."""
    if not provided or not configured:
        return False
    return secrets.compare_digest(
        provided.encode("utf-8", "surrogateescape"),
        configured.encode("utf-8", "surrogateescape"),
    )


def check_required_env_variables() -> None:
    """Raise if a variable in const.REQUIRED_ENV_VARIABLES is empty."""
    missing = [
        f"{name} ({message})"
        for name, message in const.REQUIRED_ENV_VARIABLES.items()
        if not str(getattr(settings, name, "") or "").strip()
    ]
    if missing:
        raise RuntimeError(msg.ENV_VARIABLES_MISSING.format(missing="; ".join(missing)))
