"""Root pytest configuration to fix import path conflicts.

This conftest.py ensures that the src/img2pose package is imported instead
of the img2pose.py file in the repo root.

IMPORTANT: The path manipulation MUST happen at module load time (before
any imports in test files), not in pytest hooks which run later.
"""

import sys
from pathlib import Path

# Get paths - must be done BEFORE any imports that might trigger img2pose
_repo_root = str(Path(__file__).parent)
_src_path = str(Path(__file__).parent / "src")

# Remove repo root from sys.path IMMEDIATELY to avoid img2pose.py shadowing
# the src/img2pose package. This must happen at conftest load time.
while _repo_root in sys.path:
    sys.path.remove(_repo_root)

# Ensure src is at the front of path
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
