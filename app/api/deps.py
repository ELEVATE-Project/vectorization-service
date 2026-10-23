from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings
from app.constants import messages as msg
from app.validations.generic import secret_matches


def verify_internal_token(
    internal_access_token: Optional[str] = Header(None, convert_underscores=False),
) -> None:
    """401 unless the internal_access_token header matches INTERNAL_API_TOKEN.

    A missing token gets the same 401 as a wrong one.
    """
    if not secret_matches(internal_access_token, settings.INTERNAL_API_TOKEN):
        raise HTTPException(status_code=401, detail=msg.INTERNAL_TOKEN_INVALID)
