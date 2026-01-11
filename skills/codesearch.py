from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class CodeSearchSkill:
    """Search code APIs and libraries."""
    
    def run(self, query: str) -> str:
        """
        Search for code examples and API usage.
        
        Args:
            query: The code search query
            
        Returns:
            Code search results as string
        """
        return f"[CodeSearch] {query}"


codesearch = CodeSearchSkill()
