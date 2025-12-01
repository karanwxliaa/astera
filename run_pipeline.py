"""Entry point to run the full pipeline.

Usage (from project root):

    python run_pipeline.py --config config.yaml
"""
import argparse
from pathlib import Path

from atlas_cv_rebounding.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to pipeline config YAML (relative to project root)",
    )
    args = parser.parse_args()
    project_root = Path(".")
    run_full_pipeline(project_root, args.config)


if __name__ == "__main__":
    main()
