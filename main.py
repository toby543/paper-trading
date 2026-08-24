#!/usr/bin/env python3
"""Entry point: `python main.py {run|once|portfolio|history}`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from papertrader.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
