#!/bin/bash

# Simple script to insert documents via curl
# Usage: ./insert_document_curl.sh <file_path>

# Configuration
API_URL="http://127.0.0.1:8000/api/documents"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to insert a document
insert_document() {
    local file_path="$1"
    local priority="${2:-P1}"
    local source_id="$3"
    local company_id="$4"
    local title="$5"
    local summary="$6"
    local tags="$7"
    local metadata="$8"
    
    # Check if file exists
    if [ ! -f "$file_path" ]; then
        echo -e "${RED}❌ Error: File not found: $file_path${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}📤 Uploading: $(basename "$file_path")${NC}"
    
    # Build curl command
    curl_cmd=(
        curl -X POST "$API_URL"
        -H "accept: application/json"
        -H "Content-Type: multipart/form-data"
        -F "file=@$file_path"
        -F "priority=$priority"
    )
    
    # Add optional parameters
    [ -n "$source_id" ] && curl_cmd+=(-F "source_id=$source_id")
    [ -n "$company_id" ] && curl_cmd+=(-F "company_id=$company_id")
    [ -n "$title" ] && curl_cmd+=(-F "title=$title")
    [ -n "$summary" ] && curl_cmd+=(-F "summary=$summary")
    [ -n "$tags" ] && curl_cmd+=(-F "tags=$tags")
    [ -n "$metadata" ] && curl_cmd+=(-F "metadata=$metadata")
    
    # Execute curl command
    response=$("${curl_cmd[@]}" -w "\n%{http_code}" -s)
    
    # Extract status code and body
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    # Check response
    if [ "$http_code" -eq 201 ]; then
        echo -e "${GREEN}✅ Success! (HTTP $http_code)${NC}"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    else
        echo -e "${RED}❌ Error: HTTP $http_code${NC}"
        echo "$body"
    fi
    
    echo ""
}

# Example usage
echo "=========================================="
echo "Document Upload Script (curl)"
echo "=========================================="
echo ""

# Example 1: Basic upload
echo "Example 1: Basic Upload"
echo "------------------------------------------"
insert_document \
    "/path/to/your/document.pdf" \
    "P1"

# Example 2: Upload with all parameters
echo "Example 2: Upload with All Parameters"
echo "------------------------------------------"
insert_document \
    "/path/to/your/document.pdf" \
    "P1" \
    "doc_001" \
    "company_123" \
    "Sample Document Title" \
    "This is a sample document summary" \
    '["tag1", "tag2", "tag3"]' \
    '{"author": "John Doe", "department": "Engineering"}'

# Example 3: Upload CSV with tags as comma-separated
echo "Example 3: CSV Upload with Comma-Separated Tags"
echo "------------------------------------------"
insert_document \
    "/path/to/your/data.csv" \
    "P2" \
    "doc_002" \
    "company_123" \
    "Data File" \
    "Monthly sales data" \
    "sales, data, monthly"

echo "=========================================="
echo "Upload Complete"
echo "=========================================="
