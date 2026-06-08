import logging
from typing import List, Dict, Any
from app.core.clients.qdrant import qdrant_client
from app.core.clients.embedding import generate_single_embedding
from app.config import settings
from app.models.api_models import TextSearchRequest, TextSearchResponse, TextSearchResultItem

logger = logging.getLogger(__name__)


class TextEmbeddingSearchService:
    """Service for simple text embedding search on Qdrant"""
    
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME
    
    def search(self, request: TextSearchRequest) -> TextSearchResponse:
        """
        Search using text embeddings and return all matching chunks sorted by score.
        Multiple chunks from the same source_id can be returned if they match.
        
        Args:
            request: TextSearchRequest with query and top_k
            
        Returns:
            TextSearchResponse with all matching chunks sorted by score
        """
        try:
            # Generate embedding for the query
            query_embedding = generate_single_embedding(request.query)
            
            # Search in Qdrant using only the 'text' vector
            search_results = qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=("text", query_embedding.tolist()),
                limit=request.top_k
            )
            
            # Collect all matching chunks that meet the threshold
            matching_chunks: List[Dict[str, Any]] = []
            
            for result in search_results:
                source_id = result.payload.get("source_id")
                
                if not source_id:
                    continue
                
                # Filter by threshold
                if result.score < request.threshold:
                    continue
                
                # Add all matching chunks (including multiple from same source_id)
                matching_chunks.append({
                    "source_id": source_id,
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {})
                })
            
            # Results are already sorted by score from Qdrant (highest first)
            # Take top_k results
            results_list = matching_chunks[:request.top_k]
            
            # Create response items
            result_items = [
                TextSearchResultItem(
                    source_id=item["source_id"],
                    text=item["text"],
                    score=item["score"],
                    metadata=item["metadata"]
                )
                for item in results_list
            ]
            
            return TextSearchResponse(
                query=request.query,
                total_results=len(result_items),
                results=result_items
            )
            
        except Exception as e:
            logger.error(f"Error in text embedding search: {str(e)}")
            raise
