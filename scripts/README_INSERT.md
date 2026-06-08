# Document Insertion Scripts

This directory contains scripts to insert documents via the `/api/documents` endpoint.

## Available Scripts

### 1. Python Script (`insert_document.py`)

A Python script with helper functions for single and batch document uploads.

#### Prerequisites
```bash
pip install requests
```

#### Usage

**Single Document Upload:**
```python
from insert_document import insert_document

insert_document(
    file_path="/path/to/document.pdf",
    priority="P1",
    source_id="doc_001",
    company_id="company_123",
    title="Sample Document",
    summary="Document summary",
    metadata={"author": "John Doe", "department": "Engineering"},
    tags=["python", "api", "documentation"]
)
```

**Batch Upload:**
```python
from insert_document import insert_multiple_documents

documents = [
    {
        "file_path": "/path/to/doc1.pdf",
        "priority": "P1",
        "source_id": "doc_001",
        "title": "First Document",
        "tags": ["tag1", "tag2"]
    },
    {
        "file_path": "/path/to/doc2.csv",
        "priority": "P2",
        "source_id": "doc_002",
        "title": "Second Document"
    }
]

insert_multiple_documents(documents)
```

**Run directly:**
```bash
# Edit the script to set your file paths, then run:
python3 insert_document.py
```

---

### 2. Shell Script (`insert_document_curl.sh`)

A bash script using curl for quick document uploads.

#### Usage

**Basic Upload:**
```bash
./insert_document_curl.sh /path/to/document.pdf
```

**Upload with Parameters:**
```bash
# Modify the script examples or use the function directly:
insert_document \
    "/path/to/document.pdf" \
    "P1" \
    "doc_001" \
    "company_123" \
    "Document Title" \
    "Document Summary" \
    '["tag1", "tag2"]' \
    '{"author": "John Doe"}'
```

---

## API Endpoint Details

**Endpoint:** `POST /api/documents`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | File to upload (PDF, DOCX, CSV, XLSX, TXT) |
| `priority` | String | No | Priority level (P1, P2, P3, etc.) Default: P1 |
| `source_id` | String | No | Unique identifier for the source |
| `company_id` | String | No | Company identifier |
| `title` | String | No | Document title |
| `summary` | String | No | Document summary |
| `metadata` | JSON String | No | Additional metadata as JSON object |
| `tags` | JSON Array or CSV | No | Tags as JSON array or comma-separated |

**Supported File Types:**
- PDF (`.pdf`)
- Word Documents (`.doc`, `.docx`)
- Excel Files (`.xlsx`, `.xls`)
- CSV Files (`.csv`)
- Text Files (`.txt`)

---

## Examples

### Example 1: Simple Upload
```bash
curl -X POST "http://127.0.0.1:8000/api/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F "priority=P1"
```

### Example 2: Upload with Metadata
```bash
curl -X POST "http://127.0.0.1:8000/api/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F "priority=P1" \
  -F "source_id=doc_001" \
  -F "company_id=company_123" \
  -F "title=Sample Document" \
  -F "summary=This is a sample document" \
  -F 'metadata={"author": "John Doe", "department": "Engineering"}' \
  -F 'tags=["python", "api", "documentation"]'
```

### Example 3: Upload with Comma-Separated Tags
```bash
curl -X POST "http://127.0.0.1:8000/api/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.csv" \
  -F "priority=P2" \
  -F "source_id=doc_002" \
  -F "tags=sales, data, monthly"
```

---

## Response Format

**Success Response (HTTP 201):**
```json
{
  "status": "success",
  "message": "Successfully processed 10 chunks from document.pdf",
  "chunks_processed": 10,
  "points_uploaded": 10,
  "upload_failures": 0,
  "file_type": "pdf",
  "priority": "P1",
  "source_id": "doc_001",
  "company_id": "company_123",
  "title": "Sample Document",
  "summary": "Document summary",
  "tags": ["tag1", "tag2"],
  "supported_file_types": [".pdf", ".docx", ".csv", ".xlsx", ".txt"],
  "sample_chunk": {
    "text": "First chunk text...",
    "metadata": {...}
  }
}
```

**Error Response:**
```json
{
  "detail": "Error message"
}
```

---

## Configuration

Update the API URL in the scripts:

**Python Script:**
```python
API_BASE_URL = "http://127.0.0.1:8000"  # Change to your API URL
```

**Shell Script:**
```bash
API_URL="http://127.0.0.1:8000/api/documents"
```

---

## Notes

1. **Metadata Format:** Must be a valid JSON object (dict)
   - ✅ Valid: `{"author": "John", "dept": "IT"}`
   - ❌ Invalid: `["item1", "item2"]`

2. **Tags Format:** Can be either:
   - JSON array: `["tag1", "tag2", "tag3"]`
   - Comma-separated: `"tag1, tag2, tag3"`

3. **Priority Levels:** P1 (highest), P2, P3, etc.

4. **Source ID:** Used to group and manage related documents

5. **Company ID:** Used for multi-tenant filtering

---

## Troubleshooting

**File not found:**
- Ensure the file path is correct and the file exists
- Use absolute paths for reliability

**Invalid JSON:**
- Validate your JSON using a JSON validator
- Ensure proper escaping in shell commands

**Connection refused:**
- Verify the API server is running
- Check the API URL and port

**Unsupported file type:**
- Check the supported file types list
- Ensure the file extension is correct
