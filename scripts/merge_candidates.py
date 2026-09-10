from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing source_*.csv.gz files")
    parser.add_argument("--output", default="outputs/main/all_candidates.csv.gz")
    args = parser.parse_args()

    paths = sorted(Path(args.directory).glob("source_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No candidate files found in {args.directory}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, compression="gzip")
    print(output)


if __name__ == "__main__":
    main()
