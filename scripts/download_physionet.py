from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PhysioNet EEGMMI files via MNE for future adapter development."
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--runs", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--out", default="data/raw/physionet")
    args = parser.parse_args()

    from mne.datasets import eegbci

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        files = eegbci.load_data(subject, args.runs, path=out)
        print(f"Subject {subject}:")
        for file in files:
            print(f"  {file}")


if __name__ == "__main__":
    main()

