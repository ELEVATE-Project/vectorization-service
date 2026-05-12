#!/usr/bin/env python3
"""
Simple script to insert documents via the /api/documents endpoint
"""
import requests
import json
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Change to your API URL
API_ENDPOINT = f"{API_BASE_URL}/api/documents"

def insert_document(
    file_path: str,
    priority: str = "P1",
    source_id: str = None,
    company_id: str = None,
    title: str = None,
    summary: str = None,
    metadata: dict = None,
    tags: list = None
):
    """
    Insert a document via the API

    Args:
        file_path: Path to the file to upload
        priority: Priority level (P1, P2, P3, etc.)
        source_id: Unique identifier for the source
        company_id: Company identifier
        title: Document title
        summary: Document summary
        metadata: Additional metadata as dict
        tags: List of tags

    Returns:
        Response from the API
    """

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None

    # Prepare the file
    file_name = os.path.basename(file_path)
    files = {
        'file': (file_name, open(file_path, 'rb'))
    }

    # Prepare form data
    data = {
        'priority': priority
    }

    # Add optional parameters
    if source_id:
        data['source_id'] = source_id

    if company_id:
        data['company_id'] = company_id

    if title:
        data['title'] = title

    if summary:
        data['summary'] = summary

    # Convert metadata dict to JSON string
    if metadata:
        data['metadata'] = json.dumps(metadata)

    # Convert tags list to JSON string
    if tags:
        data['tags'] = json.dumps(tags)

    try:
        print(f"📤 Uploading: {file_name}")
        print(f"   Priority: {priority}")
        if source_id:
            print(f"   Source ID: {source_id}")
        if company_id:
            print(f"   Company ID: {company_id}")
        if title:
            print(f"   Title: {title}")
        if tags:
            print(f"   Tags: {tags}")

        # Make the API request
        response = requests.post(
            API_ENDPOINT,
            files=files,
            data=data,
            headers={'accept': 'application/json'}
        )

        # Close the file
        files['file'][1].close()

        # Check response
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Success! Processed {result.get('chunks_processed', 0)} chunks")
            print(f"   Points uploaded: {result.get('points_uploaded', 0)}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return None

    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        return None


def insert_multiple_documents(documents: list):
    """
    Insert multiple documents

    Args:
        documents: List of document configs, each containing:
            - file_path (required)
            - priority (optional)
            - source_id (optional)
            - company_id (optional)
            - title (optional)
            - summary (optional)
            - metadata (optional)
            - tags (optional)
    """
    print(f"\n{'='*60}")
    print(f"Starting batch upload of {len(documents)} documents")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0

    for i, doc_config in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Processing document...")

        result = insert_document(**doc_config)

        if result:
            success_count += 1
        else:
            fail_count += 1

        print("-" * 60)

    print(f"\n{'='*60}")
    print("Batch Upload Complete")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":

    # Example 1: Single document upload
    print("Example 1: Single Document Upload")
    print("=" * 60)

    insert_document(
        file_path="/Users/anujvaghani0/Downloads/COE ASSETS-SOLUTIONS/Involve/Assets/COE_AI1 Field_Visit_Observation_Tools.docx",  # Change this to your file path
        priority="P1",
        source_id="1",
        company_id="shikshalokan",
        title="Sample Document",
        summary="This is a sample document for testing",
        metadata={
            "TITLE": "John Doe",
            "DOCUMENT TYPE": "Engineering",
            "ASSET TITLE": "Technical",
            "SUBMITTING ORGANIZATION & CONTACT":"",
            "TYPE OF ASSET":"",
            "PURPOSE / USE CASE":"",
            "INTENDED USERS" :"",
            "APPLICABLE CONTEXTS":"",
            "INSTRUCTIONS FOR USE":"",
            "ALIGNMENT WITH SHIKSHAGRAHA MOVEMENT":"",
            "LIMITATIONS OR ASSUMPTIONS":"",
            "COMPLEMENTARY RESOURCES":"",
            "TAGS / CLASSIFICATION":"",

        },
        tags=["python", "api", "documentation"]
    )

    # # Example 2: Multiple documents upload
    # print("\n\nExample 2: Multiple Documents Upload")
    # print("=" * 60)


