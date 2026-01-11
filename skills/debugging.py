from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class DebuggingSkill:
    """Systematic debugging framework with root cause investigation."""
    
    def run(self, command: str, description: str = "Debug task") -> str:
        """
        Run debugging analysis.
        
        Args:
            command: The debugging command or issue
            description: Description of the debugging task
            
        Returns:
            Debugging results as string
        """
        return f"[Debugging] {description}: {command}"


Debugging = DebuggingSkill()
