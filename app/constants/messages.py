"""API response messages for the vectorization service.

Error details and success messages are kept here, grouped by flow (upload now; search and
other flows can add their own section), so the wording can be changed in one place.
Messages with {placeholders} are filled at the call site with .format(...),
e.g. SOURCE_ID_TOO_LONG.format(max_length=255).

Usage: from app.constants import messages as msg
"""

# =============================================================================
# Internal and admin tokens (app/api/deps.py)
# =============================================================================

# A missing token and a wrong one get the same 401
INTERNAL_TOKEN_INVALID = "Invalid or missing internal token"
ADMIN_TOKEN_INVALID = "Invalid or missing admin token"


# =============================================================================
# Acronym bulk upload (POST /api/acronyms/bulk)
# =============================================================================

# File checks before any row is read
ACRONYM_CSV_TOO_LARGE = "File exceeds max size of {max_size_mb}MB"
ACRONYM_CSV_NOT_UTF8 = "File must be UTF-8 encoded"
ACRONYM_CSV_MISSING_COLUMNS = "CSV must have columns {required_columns}, found {found_columns}"
ACRONYM_CSV_NO_DATA_ROWS = "CSV file has no data rows"

# Per-row errors (errors[].reason); the other rows still commit
ACRONYM_ROW_EMPTY = "acronym and expansions must be non-empty after trimming whitespace"
ACRONYM_ROW_INVALID_KEY = (
    "acronym must be letters only (1-{max_words} space-separated words; "
    "a lone word needs at least 2 letters) — this can never be "
    "detected in a search query otherwise"
)
ACRONYM_ROW_INVALID_IS_ACTIVE = "is_active must be 'true' or 'false' if present, got {status!r}"
ACRONYM_ROW_TOO_LONG = "acronym exceeds max length of {max_length} characters"
ACRONYM_ROW_DUPLICATE = "duplicate acronym in batch, superseded by row {index}"

# The upsert failed; a connection lost during the commit can leave it saved
ACRONYM_UPLOAD_DB_ERROR = (
    "Database error: the upload could not be confirmed. Retrying the whole "
    "file is safe; check the saved acronyms if unsure."
)


# =============================================================================
# Acronym search (POST /documents/search, acronym_info.method_description)
# =============================================================================

ACRONYM_METHOD_MAX = (
    "The query and the query with each acronym written out are searched "
    "separately, for meaning and for keywords; each field keeps the higher score."
)
ACRONYM_METHOD_INLINE = (
    "One search text with each acronym's full form written next to it, e.g. "
    "\"PTM (Parent Teacher Meeting) documents\""
)
# Appended to ACRONYM_METHOD_INLINE, by ACRONYM_KEYWORD_OR
ACRONYM_METHOD_INLINE_KEYWORD_OR = (
    "; the keyword search runs once for the query and once more for each "
    "meaning, with every acronym replaced by that meaning (\"Parent Teacher "
    "Meeting documents\"), and each chunk keeps the higher keyword score."
)
ACRONYM_METHOD_INLINE_KEYWORD = "; one keyword search with the words of the query and of every full form."
