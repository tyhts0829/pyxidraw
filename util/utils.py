from pathlib import Path
from typing import Any, Dict

import yaml


def load_config() -> Dict[str, Any]:
    """ルートの config.yaml を読み込み辞書を返す。

    - ファイルが存在しない/読み込めない場合は空辞書を返す（フェイルソフト）。
    - YAML のトップレベルが辞書でない場合も空辞書を返す。
    """
    project_root = Path(__file__).parents[1]
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        # ログは呼び出し側に委ねる（ここでは静かにフォールバック）
        return {}
