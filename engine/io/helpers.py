from typing import Optional, Union

# ボタン/シフトに相当するトグル系 CC 番号をモジュール定数として定義
TOGGLE_CC_NUMBERS = (25, 26, 27, 28, 29, 30, 35)


class DualKeyDict:
    """
    整数キーと文字列キーの両方で値にアクセスできる辞書。
    キー間で値が同期される。
    """

    def __init__(self):
        self._int_to_value = {}
        self._str_to_value = {}
        self.is_active = None

    def init_map(self, cc_map):
        self.cc_map = cc_map
        self._reverse_cc_map = {v: k for k, v in cc_map.items()}
        self.reset_activation()
        for cc, name in self.cc_map.items():
            self._int_to_value[cc] = 0
            self._str_to_value[name] = 0

    def __repr__(self):
        # _str_to_valueを1行ずつ表示
        texts = []
        for key, value in self._str_to_value.items():
            texts.append(f" {key}: {value}")
        return "\n".join(texts)

    def reset_activation(self):
        self.is_active = {name: False for name in self.cc_map.values()}

    def __getitem__(self, key: Union[int, str]) -> int:
        """
        キーに対応する値を取得。
        Args:
            key (int or str): キー。
        Returns:
            1〜127の整数。
        Raises:
            KeyError: サポート外のキー型。
        """
        if isinstance(key, int):
            return self._int_to_value[key]
        elif isinstance(key, str):
            # 読み取りで副作用を起こさない（is_active の更新は setter/専用メソッドで行う）
            return self._str_to_value[key]
        else:
            raise KeyError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key: Union[int, str], value: int) -> None:
        """
        キーに値を設定し、対応するもう一方のキーにも反映。

        Args:
            key (int or str): キー。
            value (int): 設定する値。

        Raises:
            KeyError: サポート外のキー型。
        """

        # ボタンキーの場合はトグルする
        if self._is_toggle_key(key):
            if value == 0:  # ボタンを離したときは何もしない
                return
            value = self._toggle_value(key)
            self._update_value(key, value)
        else:
            self._update_value(key, value)

    def _toggle_value(self, key: Union[int, str]) -> int:
        """
        ボタンキーの値をトグルした値を返す。
        Args:
            key (int or str): キー。
        """
        current_value = self[key]
        if current_value == 0:
            value = 1
        else:
            value = 0
        return value

    def _update_value(self, key: Union[int, str], value: int) -> None:
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
            raise KeyError(f"Unsupported key type: {type(key)}")

    def _is_toggle_key(self, key: Union[int, str]) -> Optional[bool]:
        """
        キーがボタンキーか確認。
        Args:
            key (int or str): キー。
        Returns:
            bool: ボタンキーならTrue。
        """
        if isinstance(key, int):
            # 25〜30のbuttonキー、35のshiftキー
            return key in TOGGLE_CC_NUMBERS
        # if isinstance(key, str):
        #     # bから始まる文字列がボタンキー
        #     if key.startswith("b"):
        #         return True
        #     else:
        #         return False

    def keys(self):
        """整数キーの一覧を返す。"""
        return self._int_to_value.keys()

    def values(self):
        """値の一覧を返す。"""
        return self._int_to_value.values()

    def items(self):
        """整数キーと値のペアを返す。"""
        return self._int_to_value.items()

    def _get_str_key_from_int_key(self, int_key: int) -> Optional[str]:
        """整数キーから対応する文字列キーを取得。"""
        return self.cc_map.get(int_key)

    def _get_int_key_from_str_key(self, str_key: str) -> Optional[int]:
        """文字列キーから対応する整数キーを取得。"""
        return self._reverse_cc_map.get(str_key)

    def __contains__(self, key: Union[int, str]) -> bool:
        """
        キーが存在するか確認。
        Args:
            key (int or str): キー。
        Returns:
            bool: 存在すればTrue。
        """
        if isinstance(key, int):
            return key in self._int_to_value
        elif isinstance(key, str):
            return key in self._str_to_value
        else:
            return False

    def get(self, key, default=None):
        """
        キーに対応する値を取得。なければデフォルトを返す。
        Args:
            key (int or str): キー。
            default (Optional[int]): デフォルト値。
        Returns:
            Optional[int]: 値またはデフォルト。
        """
        try:
            return self[key]
        except KeyError:
            return default
