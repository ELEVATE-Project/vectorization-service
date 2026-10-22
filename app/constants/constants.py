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
