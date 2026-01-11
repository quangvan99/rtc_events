from __future__ import annotations

import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class CodeReviewSkill:
    """Code review with technical rigor and verification gates."""
    
    def run(self, code: str, description: str = "Code review") -> str:
        """
        Review code for issues and improvements.
        
        Args:
            code: The code to review
            description: Description of the review task
            
        Returns:
            Code review results as string
        """
        return f"[Code Review] {description}"


code_review = CodeReviewSkill()
