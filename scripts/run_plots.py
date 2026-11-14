"""Build figures and the cost-matched table from the replay CSV files.

Usage:  python scripts\\run_plots.py
        python scripts\\run_plots.py --subdir replay_strict
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.experiments.plots import make_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="replay")
    args = ap.parse_args()
    make_all(load_config(), args.subdir)


if __name__ == "__main__":
    main()