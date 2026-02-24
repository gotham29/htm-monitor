#tests/conftest.py

import os

# Force headless backend for matplotlib before pyplot is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")