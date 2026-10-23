"""API response messages for the vectorization service.

Error details and success messages are kept here, grouped by flow (upload now; search and
other flows can add their own section), so the wording can be changed in one place.
Messages with {placeholders} are filled at the call site with .format(...),
e.g. SOURCE_ID_TOO_LONG.format(max_length=255).

Usage: from app.constants import messages as msg
"""

# =============================================================================
# Document upload (POST /api/documents)
# =============================================================================

# Form parsing in the upload endpoint (metadata / tags sent as form strings)
METADATA_FORM_NOT_JSON_OBJECT = "Metadata must be a JSON object/dict"
METADATA_FORM_INVALID_JSON = "Invalid metadata JSON: {error}"
TAGS_FORM_NOT_JSON_ARRAY = "Tags must be a JSON array/list"
TAGS_FORM_BLANK_OR_NON_STRING_TAG = "Each tag must be a non-empty string"
TAGS_FORM_INVALID_JSON = "Invalid tags JSON: {error}"

# source_id and priority (also reused by update / upsert / delete / metadata PATCH)
SOURCE_ID_REQUIRED = "source_id is required and cannot be empty"
SOURCE_ID_TOO_LONG = "source_id must be at most {max_length} characters"
SOURCE_ID_INVALID_CHARACTERS = "source_id contains invalid characters (allowed pattern: {pattern})"
PRIORITY_INVALID_FORMAT = "Invalid priority format. Must be P1, P2, P3, etc."

# Descriptive fields: metadata, title / summary, tags, theme
METADATA_NOT_JSON_OBJECT = "metadata must be a JSON object"
METADATA_SOURCE_ID_MISMATCH = "metadata.source_id ({metadata_source_id}) does not match source_id ({source_id})"
METADATA_COMPANY_MISMATCH = "metadata.company ({metadata_company}) does not match company_id ({company_id})"
MARKDOWN_URL_NOT_HTTP = "metadata.markdown_url must be an http(s) URL"
FIELD_BLANK_WHEN_PROVIDED = "{field} cannot be empty when provided"
TAGS_NOT_A_LIST = "tags must be a list of strings"
TAGS_BLANK_OR_NON_STRING = "tags must be non-empty strings"
THEME_NOT_A_STRING = "theme must be a string"
METADATA_THEME_MISMATCH = "metadata.theme ({metadata_theme}) does not match theme ({theme})"

# Uploaded file checks (name, type, size) before any parsing
FILE_WITH_FILENAME_REQUIRED = "A file with a filename is required"
UNSUPPORTED_FILE_TYPE = "Unsupported file type: '{extension}'. Supported types: {supported_types}"
UPLOADED_FILE_EMPTY = "Uploaded file is empty"
FILE_TOO_LARGE = "File exceeds the maximum allowed size of {max_size_mb} MB"

# File processors: shared checks, then one block per file type
NO_CONTENT_EXTRACTED = "No content could be extracted from the file."
PROCESSOR_EMPTY_FILE_CONTENT = "Empty file content for {filename}"
PROCESSOR_FILE_TOO_LARGE = (
    "File {filename} is too large ({size_mb:.2f}MB). Maximum allowed size is {max_size_mb}MB"
)

PDF_NO_TEXT_EXTRACTED = (
    "Could not extract any text from PDF: {filename}. "
    "The document may be corrupted or contain no readable content."
)
PDF_OCR_LIBRARIES_MISSING = (
    "OCR libraries not available. Please install pytesseract, pdf2image, and Pillow. "
    "Also ensure tesseract-ocr is installed on your system."
)
PDF_PROCESSING_FAILED = "Error processing PDF: {error}"
DOCX_PROCESSING_FAILED = "Error processing DOCX: {error}"
XLSX_PROCESSING_FAILED = "Error processing XLSX: {error}"
CSV_MISSING_SL_NO_COLUMN = "Missing required column: SL NO"
CSV_PROCESSING_FAILED = "Error processing CSV: {error}"
TEXT_FILE_NO_READABLE_TEXT = "File {filename} is empty or contains no readable text"
TEXT_PROCESSING_FAILED = "Error processing text/markdown file: {error}"

# metadata.markdown_url: fetching the page and turning it into chunks
URL_EMPTY = "URL cannot be empty"
URL_INVALID_FORMAT = "Invalid URL format: {url}. URL must start with http:// or https://"
URL_NO_TEXT_EXTRACTED = "No text content could be extracted from URL: {url}"
URL_FETCH_TIMEOUT = "Request timeout while fetching URL: {url}"
URL_CONNECTION_FAILED = "Could not connect to URL: {url}"
URL_HTTP_ERROR = "HTTP error {status_code} while fetching URL: {url}"
URL_EXTRACTION_FAILED = "Failed to extract text from URL: {error}"
URL_TEXT_PROCESSING_FAILED = "Failed to process URL text: {error}"

# Storing the chunks in Qdrant (a partial upload is rolled back)
NO_VALID_CHUNKS_TO_UPLOAD = "No valid chunks could be built for upload; nothing was stored."
INVALID_CHUNKS = (
    "Could not build {invalid_count} of {chunk_count} chunks for upload "
    "(missing id, text or metadata); nothing was stored."
)
PARTIAL_UPLOAD_ROLLED_BACK = (
    "Upload to vector store failed for source_id {source_id}: "
    "{error_count} of {total_points} points failed. "
    "No partial document was kept; retry the request."
)
PARTIAL_UPLOAD_ROLLBACK_FAILED = (
    "Upload to vector store failed for source_id {source_id}: "
    "{error_count} of {total_points} points failed, and removing the "
    "{stored_count} stored points also failed. Cleanup is "
    "retrying in the background; they may be searchable until it succeeds."
)
UPLOAD_FAILED = "Upload failed: {error}"

# Success response
UPLOAD_SUCCEEDED = "Successfully processed {chunk_count} chunks from {filename}"


# =============================================================================
# Theme update (PATCH /api/documents/{source_id}/theme)
# =============================================================================

# Validation and lookup; a missing and a blank theme are the same 400
THEME_REQUIRED = "theme is required and cannot be empty"
DOCUMENTS_NOT_FOUND = "No documents found with source_id: {source_id}"
DOCUMENTS_NOT_FOUND_FOR_COMPANY = "No documents found with source_id: {source_id} and company_id: {company_id}"

# Storing the theme; a stale updated_at is reported but the theme write is kept
THEME_UPDATED_AT_NOT_REFRESHED = (
    "Theme was set on {count} chunks of source_id {source_id}, but metadata.updated_at "
    "could not be refreshed: {error}. Retrying this request is safe."
)
THEME_UPDATE_FAILED = "Theme update failed: {error}"

# Success response
THEME_UPDATE_SUCCEEDED = "Successfully updated theme for {count} documents"


# =============================================================================
# Metadata update (PATCH /api/documents/{source_id}/metadata)
# =============================================================================

# theme is a top-level field, so a metadata update cannot set the "theme" key
METADATA_UPDATE_THEME_NOT_ALLOWED = (
    "'theme' cannot be updated through metadata. Theme is a top-level field; "
    "use the theme update API instead."
)


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
ACRONYM_CSV_EMPTY_COLUMN_NAME = "CSV has a column with no name (often a trailing comma in the header); remove it"

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
