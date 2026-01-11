from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class DatabasesSkill:
    """Work with MongoDB and PostgreSQL."""
    
    def run(self, query: str) -> str:
        """
        Get guidance on databases.
        
        Args:
            query: The database query
            
        Returns:
            Database guidance as string
        """
        return f"[Databases] {query}"


databases = DatabasesSkill()
