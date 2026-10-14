import sys
from pathlib import Path

# Make the repository's top-level packages (``compression``) importable under plain ``pytest``.
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
