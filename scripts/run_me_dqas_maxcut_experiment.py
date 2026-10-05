#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxcut_common import add_maxcut_args, json_default, run_single_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run measurement-efficient DQAS on MaxCut.")
    add_maxcut_args(parser)
    args = parser.parse_args()
    summary = run_single_experiment("ME-DQAS", args)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
