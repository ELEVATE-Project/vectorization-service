#!/usr/bin/env python3
"""
Quick example script to insert a document
Modify the variables below and run: python3 quick_insert_example.py
"""
import requests
import json

# ============================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================

# API Configuration
API_URL = "http://127.0.0.1:8000/api/documents"

# File to upload
FILE_PATH = "/path/to/your/document.pdf"  # CHANGE THIS

# Document details
PRIORITY = "P1"  # P1, P2, P3, etc.
SOURCE_ID = "doc_001"  # Unique identifier
COMPANY_ID = "company_123"  # Optional company ID
TITLE = "Sample Document"  # Optional title
SUMMARY = "This is a sample document"  # Optional summary

# Tags (list of strings)
TAGS = ["python", "api", "example"]

# Metadata (dictionary)
METADATA = {
    "author": "John Doe",
    "department": "Engineering",
    "category": "Technical"
}

# ============================================
# SCRIPT - NO NEED TO MODIFY BELOW
# ============================================

def main():
    print("=" * 60)
    print("Document Upload Script")
    print("=" * 60)
    print(f"\n📄 File: {FILE_PATH}")
    print(f"🎯 Priority: {PRIORITY}")
    print(f"🔖 Source ID: {SOURCE_ID}")
    print(f"🏢 Company ID: {COMPANY_ID}")
    print(f"📝 Title: {TITLE}")
    print(f"🏷️  Tags: {TAGS}")
    print(f"📋 Metadata: {METADATA}")
    print()
    
    try:
        # Prepare the file
        with open(FILE_PATH, 'rb') as f:
            files = {'file': f}
            
            # Prepare form data
            data = {
                'priority': PRIORITY,
                'source_id': SOURCE_ID,
                'company_id': COMPANY_ID,
                'title': TITLE,
                'summary': SUMMARY,
                'tags': json.dumps(TAGS),
                'metadata': json.dumps(METADATA)
            }
            
            # Make the request
            print("📤 Uploading...")
            response = requests.post(
                API_URL,
                files=files,
                data=data,
                headers={'accept': 'application/json'}
            )
        
        # Check response
        if response.status_code == 201:
            result = response.json()
            print("\n✅ SUCCESS!")
            print("=" * 60)
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")
            print(f"Chunks Processed: {result.get('chunks_processed')}")
            print(f"Points Uploaded: {result.get('points_uploaded')}")
            print(f"Upload Failures: {result.get('upload_failures')}")
            print("=" * 60)
            
            # Print full response
            print("\n📊 Full Response:")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"\n❌ ERROR: HTTP {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found: {FILE_PATH}")
        print("Please update the FILE_PATH variable with a valid file path.")
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Could not connect to API at {API_URL}")
        print("Please ensure the API server is running.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    main()
