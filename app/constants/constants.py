"""Internal constant text for the vectorization service (never sent in API responses).

API error/success messages live in messages.py; this file holds everything else, such as
labels used only in log lines.

Usage: from app.constants import constants as const
"""

# Rollback log labels — shown in brackets on each delete/retry log line,
# so logs tell which flow removed the points.
ROLLBACK_REASON_PARTIAL_UPLOAD = "partial upload rollback"
ROLLBACK_REASON_OLD_VERSION_CLEANUP = "old version cleanup after replace"

# Payload key of the document theme. It is a top-level field, so a metadata update
# (PATCH /metadata) must not set a key with this name.
THEME_KEY = "theme"


# =============================================================================
# Acronym search
# =============================================================================

ACRONYM_CACHE_KEY_PREFIX = "acronym"

# ACRONYM_SEARCH_METHOD values: one text with the meanings written in, or one text per meaning.
ACRONYM_SEARCH_METHOD_INLINE = "inline"
ACRONYM_SEARCH_METHOD_DUAL_QUERY = "dual_query"
ACRONYM_SEARCH_METHODS = (ACRONYM_SEARCH_METHOD_INLINE, ACRONYM_SEARCH_METHOD_DUAL_QUERY)

# Bulk-upload CSV column names (spec §7)
ACRONYM_CSV_COLUMN_ACRONYM = "acronym"
ACRONYM_CSV_COLUMN_EXPANSIONS = "expansions"
ACRONYM_CSV_COLUMN_DESCRIPTION = "description"
ACRONYM_CSV_COLUMN_IS_ACTIVE = "is_active"
ACRONYM_CSV_COLUMNS = (
    ACRONYM_CSV_COLUMN_ACRONYM,
    ACRONYM_CSV_COLUMN_EXPANSIONS,
    ACRONYM_CSV_COLUMN_DESCRIPTION,
    ACRONYM_CSV_COLUMN_IS_ACTIVE,
)
# Largest ACRONYM_BULK_UPLOAD_BATCH_SIZE: 6 values per row, and Postgres allows at most
# 65,535 values per statement.
ACRONYM_BULK_UPLOAD_MAX_BATCH_SIZE = 10000

# Acronym matching patterns; changing one changes matching.
# Non-letters stripped from a query word: "D.I.E.T." -> "DIET", "PTM2024" -> "PTM".
ACRONYM_NON_LETTER_PATTERN = r"[^A-Za-z]"
# One word in a title, summary or expansion.
WORD_TOKEN_PATTERN = r"[A-Za-z0-9]+"
# One word of an uploaded acronym key (keys are stored uppercase).
ACRONYM_KEY_WORD_PATTERN = r"[A-Z]+"
# A whole uploaded key: 1 to N words of ACRONYM_KEY_WORD_PATTERN, one space between them.
ACRONYM_KEY_TEMPLATE = r"^{word}(?: {word}){{0,{max_extra_words}}}$"
# Finding an acronym in the query text, in every shape detection accepts:
# between the letters of one word ("D.I.E.T.", "d i e t"),
ACRONYM_LETTER_GAP_PATTERN = r"(?:[^A-Za-z0-9\s]*|\s+)"
# between the words of a multi-word acronym ("R.T.E. Act"),
ACRONYM_WORD_GAP_PATTERN = r"[^A-Za-z0-9\s]*\s+[^A-Za-z0-9\s]*"
# and around the acronym: not inside a word, optional plural "S", digits on either side kept.
ACRONYM_MATCH_TEMPLATE = r"(?<![A-Za-z])(?P<lead>\d*)(?:{alternatives})S?(?P<digits>\d*)(?![A-Za-z])"


# =============================================================================
# Required environment variables (checked at startup, as in samiksha-service's envVariables.js)
# =============================================================================

# Name -> message shown when it is empty. A new required variable needs only a line here.
REQUIRED_ENV_VARIABLES = {
    "INTERNAL_API_TOKEN": "Required internal access token",
}
