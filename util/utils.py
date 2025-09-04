from pathlib import Path
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    """ルートの config.yaml を読み込み辞書を返す。"""
    project_root = Path(__file__).parents[1]
    with open(project_root / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
