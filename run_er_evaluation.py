from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    from pmf_benchmark.metrics.entropic_relevance.run_fast import (  # noqa: PLC0415
        ERFastConfig,
        run_er_evaluation_fast,
    )

    parser = argparse.ArgumentParser(
        description="Fast entropic relevance evaluation (refactored ER_v2).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., BPI2017, BPI2019_1, Hospital_Billing, sepsis)",
    )
    parser.add_argument(
        "--horizon",
        required=True,
        type=int,
        help="Rolling window length in days (e.g., 7 or 28)",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Start time for rolling windows (default: dataset-specific)",
    )
    parser.add_argument("--log-file", default=None, help="Path to log (.xes)")
    parser.add_argument(
        "--predictions-root",
        default=None,
        help=(
            "Root predictions dir (default: results/<dataset>/horizon_<h>/predictions)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/er_metrics_v2_fast",
        help="Output directory",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: results/<dataset>/horizon_<h>/er_cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache reads/writes",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Model keys to evaluate (e.g., statistical_ar2). "
            "Default: discover from predictions dir."
        ),
    )
    args = parser.parse_args()

    result = run_er_evaluation_fast(
        ERFastConfig(
            dataset=args.dataset,
            horizon=int(args.horizon),
            start_time=args.start_time,
            log_file=args.log_file,
            predictions_root=args.predictions_root,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            no_cache=bool(args.no_cache),
            models=args.models,
        ),
    )

    print("ER evaluation complete.")
    print(f"- Summary CSV: {result['summary_csv']}")
    print(f"- Window CSV: {result['window_csv']}")
    print(f"- Combined JSON: {result['combined_json']}")
    if not args.no_cache:
        print(f"- Cache: {result['cache_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
