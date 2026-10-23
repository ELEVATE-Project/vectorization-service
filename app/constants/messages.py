"""API response messages for the vectorization service.

Error details and success messages are kept here, grouped by flow (upload now; search and
other flows can add their own section), so the wording can be changed in one place.
Messages with {placeholders} are filled at the call site with .format(...),
e.g. SOURCE_ID_TOO_LONG.format(max_length=255).

Usage: from app.constants import messages as msg
"""

# =============================================================================
# Internal token (app/api/deps.py)
# =============================================================================

# A missing token and a wrong one get the same 401
INTERNAL_TOKEN_INVALID = "Invalid or missing internal token"


# =============================================================================
# Startup (app/main.py)
# =============================================================================

ENV_VARIABLES_MISSING = (
    "Server could not start. Not all environment variables are provided: {missing}. "
    "Set them in .env (see release-doc/release-2.2.0.md)."
)


# =============================================================================
# Acronym bulk upload (POST /api/acronyms/bulkUpload)
# =============================================================================

# File checks before any row is read
ACRONYM_CSV_TOO_LARGE = "File exceeds max size of {max_size_mb}MB"
ACRONYM_CSV_NOT_UTF8 = "File must be UTF-8 encoded"
ACRONYM_CSV_MISSING_COLUMNS = "CSV must have columns {required_columns}, found {found_columns}"
ACRONYM_CSV_NO_DATA_ROWS = "CSV file has no data rows"
ACRONYM_CSV_DUPLICATE_COLUMNS = "CSV has duplicate columns: {columns}"
ACRONYM_CSV_UNSUPPORTED_COLUMNS = "CSV has unsupported columns: {columns}; allowed: {allowed}"
ACRONYM_CSV_UNNAMED_COLUMN_HAS_VALUES = (
    "CSV column {position} has no name but has values; name it or remove it"
)

# Per-row errors (errors[].reason); the other rows still commit
ACRONYM_ROW_FIELD_COUNT = (
    "row has {found} fields, expected {expected}; "
    "a value that contains a comma must be in double quotes"
)
ACRONYM_ROW_EMPTY = (
    "acronym and expansions must be non-empty after trimming whitespace "
    "(expansions may be empty when is_active is false)"
)
ACRONYM_ROW_NOT_FOUND = "acronym does not exist, so it cannot be deactivated"
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

# acronym_info.method_description has three keys: semantic, keyword, boosting.
ACRONYM_SEMANTIC_INLINE = (
    "One search text with each acronym's meaning written next to it, "
    "e.g. \"PTM (Parent Teacher Meeting) documents\"."
)
ACRONYM_SEMANTIC_DUAL_QUERY = (
    "The query and the query with each acronym written out "
    "(\"Parent Teacher Meeting documents\") are searched separately; each field keeps the higher score."
)
ACRONYM_KEYWORD_SINGLE = "One search with the words of the query and of every meaning."
ACRONYM_KEYWORD_PER_MEANING = (
    "One search for the query and one for each meaning; each chunk keeps the higher score."
)
ACRONYM_BOOSTING = (
    "A title that equals or contains the query with the acronym written out gets "
    "x{exact_title} or x{partial_title}, a summary x{exact_summary} or x{partial_summary}, "
    "on top of the normal title and summary boosts (hybrid mode only)."
)
