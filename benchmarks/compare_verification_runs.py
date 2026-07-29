from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def _stage(metrics: dict[str, object], key: str) -> dict[str, object]:
    section = metrics.get(key, {})
    if not isinstance(section, dict):
        return {}
    return section


def _delta(metrics: dict[str, object], key: str) -> dict[str, object]:
    section = metrics.get(key, {})
    if not isinstance(section, dict):
        return {}
    return section


def _row(label: str, tp: int, fp: int, fn: int, precision: float, recall: float) -> str:
    return (
        f"{label:<22} "
        f"TP={tp:<3} FP={fp:<3} FN={fn:<3} "
        f"P={_fmt_float(precision):<7} R={_fmt_float(recall):<7}"
    )


def _extract_result(path: Path) -> tuple[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics_by_stage = payload.get("metrics_by_stage", {})
    verification_only = payload.get("verification_only_metrics", {})
    end_to_end = payload.get("end_to_end_metrics", {})

    candidate = _stage(metrics_by_stage, "candidate")
    verified = _stage(metrics_by_stage, "verified")
    published = _stage(metrics_by_stage, "published")

    delta_verified = _delta(verification_only or end_to_end, "delta_after_verification")
    delta_published = _delta(verification_only or end_to_end, "delta_after_publishing")

    model = str(payload.get("model", ""))
    mode = str(payload.get("benchmark_mode", ""))
    run_id = str(payload.get("run_id", ""))

    lines = [
        f"File: {path.as_posix()}",
        f"Mode: {mode}  Model: {model}  Run: {run_id}",
        _row(
            "Before verification",
            int(candidate.get("true_positives", 0)),
            int(candidate.get("false_positives", 0)),
            int(candidate.get("false_negatives", 0)),
            float(candidate.get("precision", 0.0)),
            float(candidate.get("recall", 0.0)),
        ),
        _row(
            "After verification",
            int(verified.get("true_positives", 0)),
            int(verified.get("false_positives", 0)),
            int(verified.get("false_negatives", 0)),
            float(verified.get("precision", 0.0)),
            float(verified.get("recall", 0.0)),
        ),
        _row(
            "After publishing",
            int(published.get("true_positives", 0)),
            int(published.get("false_positives", 0)),
            int(published.get("false_negatives", 0)),
            float(published.get("precision", 0.0)),
            float(published.get("recall", 0.0)),
        ),
        (
            "Delta (verification)   "
            f"TP-={int(delta_verified.get('tp_removed', 0)):<3} "
            f"FP-={int(delta_verified.get('fp_removed', 0)):<3} "
            f"newFN={int(delta_verified.get('new_fn', 0)):<3} "
            f"dP={_fmt_float(float(delta_verified.get('precision_change', 0.0))):<7} "
            f"dR={_fmt_float(float(delta_verified.get('recall_change', 0.0))):<7} "
            f"overhead={_fmt_float(float(delta_verified.get('runtime_overhead_seconds', 0.0)))}s"
        ),
        (
            "Delta (publishing)     "
            f"TP-={int(delta_published.get('tp_removed', 0)):<3} "
            f"FP-={int(delta_published.get('fp_removed', 0)):<3} "
            f"newFN={int(delta_published.get('new_fn', 0)):<3} "
            f"dP={_fmt_float(float(delta_published.get('precision_change', 0.0))):<7} "
            f"dR={_fmt_float(float(delta_published.get('recall_change', 0.0))):<7} "
            f"overhead={_fmt_float(float(delta_published.get('runtime_overhead_seconds', 0.0)))}s"
        ),
    ]
    return str(path), lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare verification metrics from benchmark result JSON files"
    )
    parser.add_argument("results", nargs="+", help="Result JSON files to compare")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for index, item in enumerate(args.results):
        path = Path(item)
        if not path.exists():
            print(f"Missing file: {path}")
            return 2
        _, lines = _extract_result(path)
        if index:
            print("-" * 90)
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
