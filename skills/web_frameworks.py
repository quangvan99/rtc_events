from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class WebFrameworksSkill:
    """Build full-stack web applications with Next.js, Turborepo."""
    
    def run(self, query: str) -> str:
        """
        Get guidance on web frameworks.
        
        Args:
            query: The web framework query
            
        Returns:
            Web framework guidance as string
        """
        return f"[WebFrameworks] {query}"


web_frameworks = WebFrameworksSkill()
