"""Command line entry point for the churn intelligence project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_dashboard() -> int:
    """Launch the Streamlit dashboard."""
    dashboard_path = PROJECT_ROOT / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def run_api() -> int:
    """Launch the API server with Uvicorn."""
    return subprocess.call([sys.executable, "-m", "uvicorn", "src.api.app:app", "--reload"])


def run_training() -> int:
    """Run the churn model training pipeline."""
    return subprocess.call([sys.executable, "-m", "src.training.train_model"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Customer Churn Intelligence System command runner."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("dashboard", help="Run the Streamlit dashboard.")
    subparsers.add_parser("api", help="Run the model-serving API.")
    subparsers.add_parser("train", help="Train and compare churn prediction models.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dashboard":
        return run_dashboard()
    if args.command == "api":
        return run_api()
    if args.command == "train":
        return run_training()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
