ACRONYM_CACHE_KEY_PREFIX = "acronym"

# Bulk-upload CSV column names (spec §7)
ACRONYM_CSV_COLUMN_ACRONYM = "acronym"
ACRONYM_CSV_COLUMN_EXPANSIONS = "expansions"
ACRONYM_CSV_COLUMN_DESCRIPTION = "description"
ACRONYM_CSV_COLUMN_IS_ACTIVE = "is_active"

# Acronym matching patterns (compiled where used). Changing one changes matching,
# so update the tests with it.
# Non-letters stripped from a query word: "D.I.E.T." -> "DIET", "PTM2024" -> "PTM".
ACRONYM_NON_LETTER_PATTERN = r"[^A-Za-z]"
# One word in a title, summary or expansion.
WORD_TOKEN_PATTERN = r"[A-Za-z0-9]+"
# One word of an uploaded acronym key (keys are stored uppercase).
ACRONYM_KEY_WORD_PATTERN = r"[A-Z]+"
