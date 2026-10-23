"""Validation for the acronym bulk upload CSV: the header, then each row."""
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.constants import constants as const
from app.constants import messages as msg
from app.models.db_models import AcronymMapping

# Read off the model, so it cannot drift from the column length.
ACRONYM_MAX_LENGTH = AcronymMapping.__table__.c.acronym.type.length

# Letters only, up to ACRONYM_MAX_PHRASE_WORDS words: the only shape detection can
# find, so anything else ("WI-FI", "4G") is rejected at upload.
_ACRONYM_KEY_RE = re.compile(const.ACRONYM_KEY_TEMPLATE.format(
    word=const.ACRONYM_KEY_WORD_PATTERN, max_extra_words=settings.ACRONYM_MAX_PHRASE_WORDS - 1
))


class AcronymCsvError(ValueError):
    """A problem with the whole file; the message is the 400 response detail."""


# =============================================================================
# Header
# =============================================================================

def validate_csv_header(header: List[str]) -> Tuple[List[str], Set[str]]:
    """Clean the header names and check them; returns (names, unnamed placeholders).

    Names are trimmed and lowercased; an unnamed column gets a placeholder name.
    Raises AcronymCsvError for duplicate, unsupported or missing required columns.
    """
    names: List[str] = []
    unnamed: Set[str] = set()
    for position, name in enumerate(header, start=1):
        clean = (name or "").strip().lower()
        if not clean:
            clean = const.ACRONYM_CSV_UNNAMED_COLUMN.format(position=position)
            unnamed.add(clean)
        names.append(clean)
    named = [n for n in names if n not in unnamed]

    duplicates = sorted({n for n in named if named.count(n) > 1})
    if duplicates:
        raise AcronymCsvError(msg.ACRONYM_CSV_DUPLICATE_COLUMNS.format(columns=duplicates))
    unsupported = [n for n in named if n not in const.ACRONYM_CSV_COLUMNS]
    if unsupported:
        raise AcronymCsvError(msg.ACRONYM_CSV_UNSUPPORTED_COLUMNS.format(
            columns=unsupported, allowed=sorted(const.ACRONYM_CSV_COLUMNS)
        ))
    required_columns = {const.ACRONYM_CSV_COLUMN_ACRONYM, const.ACRONYM_CSV_COLUMN_EXPANSIONS}
    if not required_columns.issubset(named):
        raise AcronymCsvError(msg.ACRONYM_CSV_MISSING_COLUMNS.format(
            required_columns=sorted(required_columns), found_columns=header
        ))
    return names, unnamed


def validate_csv_rows_present(rows: List[dict], names: List[str], unnamed: Set[str]) -> None:
    """Raise AcronymCsvError when there are no rows, or an unnamed column holds values."""
    if not rows:
        raise AcronymCsvError(msg.ACRONYM_CSV_NO_DATA_ROWS)
    # An unnamed column is accepted only while it is empty (a trailing comma).
    for position, name in enumerate(names, start=1):
        if name in unnamed and any((row.get(name) or "").strip() for row in rows):
            raise AcronymCsvError(msg.ACRONYM_CSV_UNNAMED_COLUMN_HAS_VALUES.format(position=position))


# =============================================================================
# Rows
# =============================================================================

def validate_acronym_row(
    index: int, row: Dict[Optional[str], Any]
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Check one CSV row; returns (acronym, entry, None) or (acronym, None, error).

    The entry holds index, expansions, description and is_active. Checks that need
    the whole batch or the database (duplicates, unknown acronym) stay in bulk_upsert.
    """
    raw_acronym = (row.get(const.ACRONYM_CSV_COLUMN_ACRONYM) or "").strip()
    acronym = raw_acronym.upper()
    expansions = split_expansions(row.get(const.ACRONYM_CSV_COLUMN_EXPANSIONS) or "")
    description = (row.get(const.ACRONYM_CSV_COLUMN_DESCRIPTION) or "").strip() or None
    status = (row.get(const.ACRONYM_CSV_COLUMN_IS_ACTIVE) or "").strip().lower()

    def error(reason: str, key: Optional[str] = acronym) -> Tuple[str, None, Dict[str, Any]]:
        return acronym, None, {"index": index, "acronym": key or None, "reason": reason}

    # csv.DictReader puts extra fields under None and gives missing ones None.
    columns = [key for key in row if key is not None]
    found = sum(row[key] is not None for key in columns) + len(row.get(None) or [])
    if found != len(columns):
        return error(msg.ACRONYM_ROW_FIELD_COUNT.format(found=found, expected=len(columns)))

    if not acronym:
        return error(msg.ACRONYM_ROW_EMPTY, key=None)

    if not is_valid_acronym_key(acronym):
        return error(msg.ACRONYM_ROW_INVALID_KEY.format(max_words=settings.ACRONYM_MAX_PHRASE_WORDS))

    # Optional: missing means active; anything but true/false is a row error.
    if status not in ("", "true", "false"):
        return error(msg.ACRONYM_ROW_INVALID_IS_ACTIVE.format(status=status))
    is_active = status != "false"

    # A deactivation keeps the stored expansions, so it needs none.
    if is_active and not expansions:
        return error(msg.ACRONYM_ROW_EMPTY)

    if len(acronym) > ACRONYM_MAX_LENGTH:
        # Reject here: an over-length value would fail the whole multi-row INSERT.
        return error(msg.ACRONYM_ROW_TOO_LONG.format(max_length=ACRONYM_MAX_LENGTH))

    entry = {"index": index, "expansions": expansions, "description": description, "is_active": is_active}
    return acronym, entry, None


def is_valid_acronym_key(acronym: str) -> bool:
    if not _ACRONYM_KEY_RE.match(acronym):
        return False
    words = acronym.split(" ")
    # Detection skips single letters, so "A" alone can never match; a one-letter
    # word inside a phrase ("RBI GRADE B") is fine.
    if len(words) == 1 and len(words[0]) < 2:
        return False
    return True


def split_expansions(raw: str) -> List[str]:
    """Pipe-separated text to a deduped list, order kept.

    Duplicated in migration f13a664a31b6 on purpose: migrations stay frozen snapshots.
    """
    seen = set()
    result = []
    for part in raw.split("|"):
        expansion = part.strip()
        if expansion and expansion not in seen:
            seen.add(expansion)
            result.append(expansion)
    return result
