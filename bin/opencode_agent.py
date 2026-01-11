#!/usr/bin/env python3
"""
Quick agent launcher for OpenCode
Usage: python3 bin/opencode_agent.py --agent explore --prompt "Find face files"
"""

import argparse
import sys
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')

from agents import general, explore

def main():
    parser = argparse.ArgumentParser(description='OpenCode Quick Agent')
    parser.add_argument('--agent', choices=['general', 'explore'], required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--description', default='Quick task')
    
    args = parser.parse_args()
    
    if args.agent == 'general':
        result = general.run(prompt=args.prompt, description=args.description)
    else:
        result = explore.run(prompt=args.prompt, description=args.description)
    
    print(result)

if __name__ == '__main__':
    main()
