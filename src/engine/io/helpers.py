from __future__ import annotations

# ボタン/シフトに相当するトグル系 CC 番号をモジュール定数として定義
TOGGLE_CC_NUMBERS = (25, 26, 27, 28, 29, 30, 35)


class DualKeyDict:
    """
    整数キー（CC 番号）と文字列キー（論理名）の両方で値にアクセスできる辞書。

    値は 0.0〜1.0 の浮動小数（正規化 CC 値）を前提とし、ボタン系は 0.0/1.0 を用いる。
    一部の呼び出しが整数（0/1 等）を渡してきても、内部では float へ安全に変換する。
    キー間で値は常に同期される。
    """

    def __init__(self):
        # CC 番号/論理名 → 正規化値（0.0〜1.0）
        self._int_to_value: dict[int, float] = {}
        self._str_to_value: dict[str, float] = {}
        # ボタン活性状態（将来的な UI 用途）
        self.is_active: dict[str, bool] | None = None

    def init_map(self, cc_map):
        self.cc_map = cc_map
        self._reverse_cc_map = {v: k for k, v in cc_map.items()}
        self.reset_activation()
        for cc, name in self.cc_map.items():
            self._int_to_value[cc] = 0.0
            self._str_to_value[name] = 0.0

    def __repr__(self):
        # _str_to_valueを1行ずつ表示
        texts = []
        for key, value in self._str_to_value.items():
            texts.append(f" {key}: {value}")
        return "\n".join(texts)

    def reset_activation(self):
        self.is_active = {name: False for name in self.cc_map.values()}

    def __getitem__(self, key: int | str) -> float:
        """
        キーに対応する値を取得。
        引数:
            key (int または str): キー。
        返り値:
            0.0〜1.0 の浮動小数。
        例外:
            KeyError: サポート外のキー型。
        """
        if isinstance(key, int):
            return self._int_to_value[key]
        elif isinstance(key, str):
            # 読み取りで副作用を起こさない（is_active の更新は setter/専用メソッドで行う）
            return self._str_to_value[key]
        else:
            raise KeyError(f"未対応のキー型です: {type(key)}")

    def __setitem__(self, key: int | str, value: float | int) -> None:
        """
        キーに値を設定し、対応するもう一方のキーにも反映。

        引数:
            key (int または str): キー。
            value (float): 設定する値（正規化 0.0〜1.0）。

        例外:
            KeyError: サポート外のキー型。
        """

        # ボタンキーの場合はトグルする
        # 互換性: 整数で渡された場合も float に変換
        if isinstance(value, int):
            value = float(value)

        if self._is_toggle_key(key):
            if value == 0 or value == 0.0:  # ボタンを離したときは何もしない
                return
            value = float(self._toggle_value(key))
            self._update_value(key, value)
        else:
            self._update_value(key, value)

    def _toggle_value(self, key: int | str) -> int:
        """
        ボタンキーの値をトグルした値を返す。
        引数:
            key (int or str): キー。
        """
        current_value = self[key]
        if current_value == 0:
            value = 1
        else:
            value = 0
        return value

    def _update_value(self, key: int | str, value: float) -> None:
        if isinstance(key, int):
            corresponding_str_key = self._get_str_key_from_int_key(key)
            self._int_to_value[key] = value
            if corresponding_str_key is not None:
                self._str_to_value[corresponding_str_key] = value
        elif isinstance(key, str):
            corresponding_int_key = self._get_int_key_from_str_key(key)
            self._str_to_value[key] = value
            if corresponding_int_key is not None:
                self._int_to_value[corresponding_int_key] = value
        else:
            raise KeyError(f"未対応のキー型です: {type(key)}")

    def _is_toggle_key(self, key: int | str) -> bool | None:
        """
        キーがボタンキーか確認。
        引数:
            key (int または str): キー。
        返り値:
            bool: ボタンキーなら True。
        """
        if isinstance(key, int):
            # 25〜30のbuttonキー、35のshiftキー
            return key in TOGGLE_CC_NUMBERS
        # 文字列キーの判定は現状未使用（必要なら名称規約で分岐）
        return None

    def keys(self):
        """整数キーの一覧を返す。"""
        return self._int_to_value.keys()

    def values(self):
        """値の一覧を返す。"""
        return self._int_to_value.values()

    def items(self):
        """整数キーと値のペアを返す。"""
        return self._int_to_value.items()

    def _get_str_key_from_int_key(self, int_key: int) -> str | None:
        """整数キーから対応する文字列キーを取得。"""
        return self.cc_map.get(int_key)

    def _get_int_key_from_str_key(self, str_key: str) -> int | None:
        """文字列キーから対応する整数キーを取得。"""
        return self._reverse_cc_map.get(str_key)

    def __contains__(self, key: int | str) -> bool:
        """
        キーが存在するか確認。
        引数:
            key (int または str): キー。
        返り値:
            bool: 存在すれば True。
        """
        if isinstance(key, int):
            return key in self._int_to_value
        elif isinstance(key, str):
            return key in self._str_to_value
        else:
            return False

    def get(self, key, default: float | None = None) -> float | None:
        """
        キーに対応する値を取得。なければデフォルトを返す。
        引数:
            key (int または str): キー。
            default (Optional[float]): デフォルト値。
        返り値:
            Optional[float]: 値またはデフォルト。
        """
        try:
            return self[key]
        except KeyError:
            return default
