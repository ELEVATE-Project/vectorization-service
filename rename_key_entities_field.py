"""
Script to rename 'DOCUMENT TYPE' field to 'DOCUMENT_TYPE' in all Qdrant documents
This will fix the filtering issue by removing the space from the field name.
"""
from qdrant_client import QdrantClient
from app.config import settings
import time

# Field name constants
FIELD_DOCUMENT_TYPE_OLD = "DOCUMENT TYPE"
FIELD_DOCUMENT_TYPE_NEW = "DOCUMENT_TYPE"


def rename_field_in_documents():
    """Rename DOCUMENT TYPE to DOCUMENT_TYPE in all documents"""
    
    client = QdrantClient(settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    collection_name = settings.COLLECTION_NAME
    
    print("="  * 80)
    print("Renaming 'DOCUMENT TYPE' to 'DOCUMENT_TYPE' in Qdrant collection")
    print(f"Collection: {collection_name}")
    print("=" * 80)
    
    # Get all documents
    print("\nStep 1: Fetching all documents...")
    all_points = []
    offset = None
    batch_count = 0
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = result
        
        if not points:
            break
        
        all_points.extend(points)
        batch_count += 1
        print(f"  Fetched batch {batch_count}: {len(points)} documents (Total: {len(all_points)})")
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    print(f"\n✓ Total documents fetched: {len(all_points)}")
    
    # Count documents with DOCUMENT TYPE field
    documents_to_update = []
    for point in all_points:
        if point.payload and 'metadata' in point.payload:
            metadata = point.payload['metadata']
            if FIELD_DOCUMENT_TYPE_OLD in metadata:
                documents_to_update.append(point)
    
    print(f"✓ Documents with 'DOCUMENT TYPE' field: {len(documents_to_update)}")
    
    if not documents_to_update:
        print("\n⚠️  No documents found with 'DOCUMENT TYPE' field. Nothing to update.")
        return
    
    # Ask for confirmation
    print(f"\n{'='*80}")
    print(f"⚠️  WARNING: This will modify {len(documents_to_update)} documents")
    print(f"{'='*80}")
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n❌ Operation cancelled by user")
        return
    
    # Update documents
    print("\nStep 2: Updating documents...")
    success_count = 0
    error_count = 0
    
    for i, point in enumerate(documents_to_update, 1):
        try:
            # Get the current payload
            payload = point.payload.copy()
            metadata = payload['metadata'].copy()
            
            # Rename the field
            if FIELD_DOCUMENT_TYPE_OLD in metadata:
                metadata[FIELD_DOCUMENT_TYPE_NEW] = metadata.pop(FIELD_DOCUMENT_TYPE_OLD)
                payload['metadata'] = metadata
                
                # Update the point in Qdrant
                client.set_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[point.id]
                )
                
                success_count += 1
                
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(documents_to_update)} documents updated")
            
        except Exception as e:
            error_count += 1
            print(f"  ❌ Error updating document {point.id}: {str(e)}")
    
    print(f"\n{'='*80}")
    print("Update completed!")
    print(f"  ✓ Successfully updated: {success_count} documents")
    print(f"  ❌ Errors: {error_count} documents")
    print(f"{'='*80}")
    
    # Verify the update
    print("\nStep 3: Verifying update...")
    result = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False
    )
    
    points, _ = result
    if points and points[0].payload:
        metadata = points[0].payload.get('metadata', {})
        if FIELD_DOCUMENT_TYPE_NEW in metadata:
            print("✅ Verification successful!")
            print(f"   Sample document now has 'DOCUMENT_TYPE': {metadata[FIELD_DOCUMENT_TYPE_NEW]}")
        elif FIELD_DOCUMENT_TYPE_OLD in metadata:
            print("⚠️  Sample document still has 'DOCUMENT TYPE' (with space)")
        else:
            print("⚠️  Sample document doesn't have DOCUMENT_TYPE field")
    
    print(f"\n{'='*80}")
    print("Done! You can now uncomment the filter code in prioritized_search_service.py")
    print("and change the field name to 'metadata.DOCUMENT_TYPE' (with underscore)")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        rename_field_in_documents()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
