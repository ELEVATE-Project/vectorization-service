"""
Example script demonstrating the Prioritized Search API

This script shows various ways to use the new search endpoint with different configurations.
"""

import requests
import json
from typing import Dict, Any, List, Optional


class PrioritizedSearchClient:
    """Client for interacting with the Prioritized Search API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.search_endpoint = f"{base_url}/api/v1/documents/search"
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        company_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a prioritized search
        
        Args:
            query: Search query text
            top_k: Number of results to return
            company_id: Filter by company ID
            
        Returns:
            Search response with results
        """
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        if company_id:
            payload["company_id"] = company_id
        
        response = requests.post(self.search_endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    
    def print_results(self, results: Dict[str, Any], show_scores: bool = True):
        """Pretty print search results"""
        print(f"\n{'='*80}")
        print(f"Query: {results['query']}")
        print(f"Total Results: {results['total_results']} | Showing Top {results['top_k']}")
        print(f"{'='*80}\n")
        
        for i, item in enumerate(results['results'], 1):
            print(f"{i}. {item.get('title', 'No Title')} (Score: {item['score']:.3f})")
            print(f"   Source: {item['source_id']}")
            
            if item.get('tags'):
                print(f"   Tags: {', '.join(item['tags'])}")
            
            if item.get('summary'):
                summary = item['summary'][:100] + "..." if len(item['summary']) > 100 else item['summary']
                print(f"   Summary: {summary}")
            
            text_preview = item['text'][:150] + "..." if len(item['text']) > 150 else item['text']
            print(f"   Text: {text_preview}")
            
            if show_scores and item.get('field_scores'):
                print(f"   Field Scores: {json.dumps(item['field_scores'], indent=6)}")
            
            print()


def example_1_basic_search():
    """Example 1: Basic search with default settings"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Search")
    print("="*80)
    
    client = PrioritizedSearchClient()
    results = client.search(
        query="machine learning algorithms",
        top_k=5
    )
    client.print_results(results)


def example_2_more_results():
    """Example 2: Search with more results"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Search with More Results (Top 20)")
    print("="*80)
    
    client = PrioritizedSearchClient()
    results = client.search(
        query="API documentation",
        top_k=20
    )
    client.print_results(results)


def example_3_different_query():
    """Example 3: Different search query"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Security Best Practices Search")
    print("="*80)
    
    client = PrioritizedSearchClient()
    results = client.search(
        query="security best practices",
        top_k=10
    )
    client.print_results(results)


def example_4_company_filtered_search():
    """Example 4: Search filtered by company"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Company-Filtered Search")
    print("="*80)
    
    client = PrioritizedSearchClient()
    results = client.search(
        query="quarterly financial report",
        top_k=10,
        company_id="company_123"
    )
    client.print_results(results)


def example_5_comprehensive_search():
    """Example 5: Comprehensive search"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Cloud Infrastructure Search")
    print("="*80)
    
    client = PrioritizedSearchClient()
    results = client.search(
        query="deployment strategies cloud infrastructure",
        top_k=15
    )
    client.print_results(results)


def example_6_configuration_search():
    """Example 6: Configuration search"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Configuration Settings Search")
    print("="*80)
    
    client = PrioritizedSearchClient()
    results = client.search(
        query="configuration settings",
        top_k=5
    )
    client.print_results(results)


def example_7_authentication_search():
    """Example 7: Authentication implementation search"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Authentication Implementation Search")
    print("="*80)
    
    client = PrioritizedSearchClient()
    results = client.search(
        query="detailed implementation steps for authentication",
        top_k=5
    )
    client.print_results(results)


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_basic_search,
        example_2_more_results,
        example_3_different_query,
        example_4_company_filtered_search,
        example_5_comprehensive_search,
        example_6_configuration_search,
        example_7_authentication_search
    ]
    
    for example in examples:
        try:
            example()
        except requests.exceptions.RequestException as e:
            print(f"Error running example: {e}")
            print("Make sure the API server is running on http://localhost:8000")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PRIORITIZED SEARCH API - EXAMPLES")
    print("="*80)
    print("\nMake sure your API server is running before executing these examples.")
    print("Start server with: uvicorn app.main:app --reload")
    print("\n" + "="*80)
    
    # Run all examples
    run_all_examples()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
