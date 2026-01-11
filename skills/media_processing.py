from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class MediaProcessingSkill:
    """Process multimedia with FFmpeg, ImageMagick."""
    
    def run(self, query: str) -> str:
        """
        Get guidance on media processing.
        
        Args:
            query: The media processing query
            
        Returns:
            Media processing guidance as string
        """
        return f"[MediaProcessing] {query}"


media_processing = MediaProcessingSkill()
