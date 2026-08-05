#!/usr/bin/env python3
"""Set the final workflow status on the current run's usage record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--status", choices=("success", "failure"), required=True)
    args = parser.parse_args()
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        raise RuntimeError("GITHUB_RUN_ID is not available")
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    records = payload.get("runs")
    if not isinstance(records, list):
        raise ValueError("usage file has no runs list")
    for record in records:
        if str(record.get("workflow_run_id")) == run_id:
            record["status"] = args.status
            break
    else:
        raise ValueError(f"usage record for workflow run {run_id} is missing")
    temporary = args.path.with_suffix(args.path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
