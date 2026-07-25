from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))


@hydra.main(config_path="../configs/er", config_name="er_fast", version_base=None)
def main(cfg: DictConfig) -> None:
    _ensure_src_on_path()

    from pmf_benchmark.metrics.entropic_relevance.run_fast import (  # noqa: PLC0415
        ERFastConfig,
        run_er_evaluation_fast,
    )

    result = run_er_evaluation_fast(
        ERFastConfig(
            dataset=str(cfg.dataset),
            horizon=int(cfg.horizon),
            start_time=cfg.get("start_time"),
            log_file=cfg.get("log_file"),
            predictions_root=cfg.get("predictions_root"),
            output_dir=str(cfg.output_dir),
            cache_dir=cfg.get("cache_dir"),
            no_cache=bool(cfg.no_cache),
            models=list(cfg.models) if cfg.get("models") else None,
        ),
    )

    print("ER evaluation complete.")
    print(f"- Summary CSV: {result['summary_csv']}")
    print(f"- Window CSV: {result['window_csv']}")
    print(f"- Combined JSON: {result['combined_json']}")
    if not cfg.no_cache:
        print(f"- Cache: {result['cache_path']}")


if __name__ == "__main__":
    main()
