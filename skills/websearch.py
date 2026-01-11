from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class WebSearchSkill:
    """Real-time web search."""
    
    def run(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: The search query
            
        Returns:
            Search results as string
        """
        return f"[WebSearch] {query}"


websearch = WebSearchSkill()
