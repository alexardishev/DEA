"""Utility helpers for config loading, logging, and path handling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str | Path) -> None:
    """Create parent directories for file path if they do not exist."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def configure_logging(log_file: str = "outputs/logs/pipeline.log") -> None:
    """Configure logger for reproducible pipeline runs."""
    ensure_parent_dir(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Persist JSON payload with utf-8 and indentation."""
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
