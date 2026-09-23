#!/usr/bin/env python3
"""Print a Markdown fact table from downloaded Modal Week 1 artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


COLUMNS = [
    "Cell",
    "Repeat",
    "GPU",
    "Success/Total",
    "QPS",
    "Audio s/s",
    "TTFC p50/p95",
    "ITL p95",
    "E2E p50/p95",
    "RTF p50",
    "C50",
    "Peak MiB",
]


def _value(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    return "" if value is None else str(value)


def _pair(mapping: dict[str, Any], first: str, second: str) -> str:
    left = _value(mapping, first)
    right = _value(mapping, second)
    return f"{left}/{right}" if left or right else ""


def _gpu_name(manifest: dict[str, Any]) -> str:
    raw = str(manifest.get("gpu_before_server", ""))
    return raw.split(",", 1)[0].strip()


def _rows(root: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cell") == "smoke":
            continue
        summary = manifest.get("speed_summary") or {}
        rows.append(
            [
                _value(manifest, "cell"),
                _value(manifest, "repeat"),
                _gpu_name(manifest),
                f"{_value(summary, 'completed_requests')}/{_value(summary, 'total_requests')}",
                _value(summary, "throughput_qps"),
                _value(summary, "audio_throughput_s_per_s"),
                _pair(summary, "audio_ttfp_median_s", "audio_ttfp_p95_s"),
                _value(summary, "inter_chunk_p95_s"),
                _pair(summary, "latency_median_s", "latency_p95_s"),
                _value(summary, "rtf_median"),
                _value(summary, "c50"),
                _value(manifest, "peak_gpu_memory_mib"),
            ]
        )
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} DOWNLOADED_RESULTS_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    rows = _rows(root)
    if not rows:
        print(f"No measured manifest.json files found under {root}", file=sys.stderr)
        return 1
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join(["---"] * len(COLUMNS)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
