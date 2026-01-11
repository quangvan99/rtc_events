#!/usr/bin/env python3
"""
Quick skill launcher for OpenCode
Usage: python3 bin/opencode_skill.py --skill planning --query "Design API"
"""

import argparse
import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')

from skills import (
    planning, research, Debugging, code_review,
    websearch, codesearch, web_frameworks,
    media_processing, databases, devops
)

SKILL_MAP = {
    'planning': planning,
    'research': research,
    'debugging': Debugging,
    'code-review': code_review,
    'websearch': websearch,
    'codesearch': codesearch,
    'web-frameworks': web_frameworks,
    'media-processing': media_processing,
    'databases': databases,
    'devops': devops,
}

def main():
    parser = argparse.ArgumentParser(description='OpenCode Quick Skill')
    parser.add_argument('--skill', choices=SKILL_MAP.keys(), required=True)
    parser.add_argument('--query', required=True)
    
    args = parser.parse_args()
    
    skill = SKILL_MAP[args.skill]
    result = skill.run(query=args.query)
    print(result)

if __name__ == '__main__':
    main()
