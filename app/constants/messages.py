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

# Descriptive fields: metadata, title / summary, tags
METADATA_NOT_JSON_OBJECT = "metadata must be a JSON object"
METADATA_SOURCE_ID_MISMATCH = "metadata.source_id ({metadata_source_id}) does not match source_id ({source_id})"
METADATA_COMPANY_MISMATCH = "metadata.company ({metadata_company}) does not match company_id ({company_id})"
MARKDOWN_URL_NOT_HTTP = "metadata.markdown_url must be an http(s) URL"
FIELD_BLANK_WHEN_PROVIDED = "{field} cannot be empty when provided"
TAGS_NOT_A_LIST = "tags must be a list of strings"
TAGS_BLANK_OR_NON_STRING = "tags must be non-empty strings"

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
