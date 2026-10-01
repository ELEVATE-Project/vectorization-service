"""
Unified text matching classifier for both acronym and ordinary queries.

This module provides a single source of truth for text matching logic,
eliminating duplication between prioritized_search_service and acronym_ranking.
"""

import re
from typing import Optional


class TextMatchClassifier:
    """Unified text matching classifier for acronym and ordinary queries.
    
    Provides a consistent interface for classifying how a query matches a field,
    supporting both substring matching (for ordinary queries) and word-boundary
    matching (for acronym queries).
    """
    
    @staticmethod
    def classify_match(
        query_lower: str,
        field_lower: str,
        rule: str = "substring"
    ) -> Optional[str]:
        """Classify how query matches field.
        
        Determines whether a query (e.g., a search term or acronym) matches a
        field value (e.g., a title or summary) and returns the match type.
        
        Args:
            query_lower: Lowercase query string to search for
            field_lower: Lowercase field value to search in
            rule: Matching rule to apply:
                - "substring": query appears anywhere in field (prefix, mid, infix)
                - "word": query appears as a whole word (letter/digit lookaround)
        
        Returns:
            "exact": field value equals query exactly
            "partial": query found in field per the specified rule
            None: no match found
        
        Examples:
            >>> classify_match("diet", "diet handbook", rule="substring")
            'partial'
            >>> classify_match("diet", "diet", rule="substring")
            'exact'
            >>> classify_match("diet", "dietary guidelines", rule="substring")
            'partial'
            >>> classify_match("diet", "dietary guidelines", rule="word")
            None
            >>> classify_match("diet", "the diet faculty", rule="word")
            'partial'
        """
        if not field_lower or not query_lower:
            return None
        
        if rule == "word":
            # Word-boundary match: term must appear as a whole word
            if not TextMatchClassifier._term_in_text(query_lower, field_lower):
                return None
        else:  # rule == "substring" (default)
            # Substring match: term can appear anywhere
            if query_lower not in field_lower:
                return None
        
        # Exact match: field equals query exactly
        return "exact" if field_lower == query_lower else "partial"
    
    @staticmethod
    def _term_in_text(term: str, text: str) -> bool:
        """Check if term appears as a whole word in text.
        
        Uses letter/digit lookarounds instead of \\b (word boundary) because
        \\b treats underscore as a word character and misses "DIET" in
        filenames like "_DIET_Empowerment.pdf".
        
        Args:
            term: Term to search for (should be lowercase for consistency)
            text: Text to search in (should be lowercase for consistency)
        
        Returns:
            True if term is found as a whole word in text, False otherwise
        
        Examples:
            >>> _term_in_text("diet", "the diet faculty")
            True
            >>> _term_in_text("diet", "dietary guidelines")
            False
            >>> _term_in_text("diet", "_DIET_Empowerment")
            True
        """
        # Pattern explanation:
        # (?<![a-zA-Z0-9_])  - Not preceded by letter, digit, or underscore
        # {re.escape(term)}  - Literal term (escaped for safety)
        # (?![a-zA-Z0-9_])   - Not followed by letter, digit, or underscore
        pattern = rf"(?<![a-zA-Z0-9_]){re.escape(term)}(?![a-zA-Z0-9_])"
        return bool(re.search(pattern, text))
