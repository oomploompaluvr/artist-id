"""Make ``import artist_id`` work when a script is run directly (python scripts/xx.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
