from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class PlanningSkill:
    """Plan technical solutions that are scalable, secure, and maintainable."""
    
    def run(self, query: str) -> str:
        """
        Generate a technical plan for the given query.
        
        Args:
            query: The planning request
            
        Returns:
            Technical plan as string
        """
        return f"[Planning] {query}"


planning = PlanningSkill()
