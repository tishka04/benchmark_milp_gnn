#!/usr/bin/env python
"""
Convenience script to run binary variable export.
Can be called from any directory.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.milp.export_binary_variables import main

if __name__ == "__main__":
    main()
