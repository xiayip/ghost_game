from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from ament_index_python.packages import get_package_share_directory


def package_config_dir() -> Path:
    return Path(get_package_share_directory("img2doc")) / "config"


def load_api_key(api_key_file: str = "") -> Optional[str]:
    """Load the API key without exposing it as a ROS parameter."""
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value

    candidates = []
    if api_key_file.strip():
        candidates.append(Path(api_key_file).expanduser())
    candidates.append(package_config_dir() / "local_api.yaml")

    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            value = str((data.get("deepseek") or {}).get("api_key") or "").strip()
            if value:
                return value
        except (OSError, yaml.YAMLError, TypeError):
            continue
    return None
