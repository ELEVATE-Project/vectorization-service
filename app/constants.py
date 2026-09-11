ACRONYM_CACHE_KEY_PREFIX = "acronym"

# Bulk-upload CSV column names (spec §7)
ACRONYM_CSV_COLUMN_ACRONYM = "acronym"
ACRONYM_CSV_COLUMN_EXPANSIONS = "expansions"
ACRONYM_CSV_COLUMN_DESCRIPTION = "description"
ACRONYM_CSV_COLUMN_IS_ACTIVE = "is_active"

# Expansion words match by prefix so inflections line up: institute /
# institutes, program / programme. But a prefix only means something once the
# shorter side is a word stem — below that, a single letter is a prefix of
# anything starting with it, so "s" stands in for "school" and "S M C Handbook"
# reads as "School Management Committee", "R.E.A.D" as "Right to Education".
# Four characters is where a prefix stops being an initial and starts being a
# stem.
ACRONYM_MIN_PREFIX_MATCH_LEN = 4
