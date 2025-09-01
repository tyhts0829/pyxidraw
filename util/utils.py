from pathlib import Path
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    this_dir = Path(__file__).parents[1]
    with open(this_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
