from pathlib import Path


def test_config_dirs_prepend(tmp_path, monkeypatch):
    # ダミーフォントファイルを作成
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    dummy = fonts_dir / "DummyFont-Regular.ttf"
    dummy.write_bytes(b"")

    # load_config をモンキーパッチして search_dirs を返す
    def _fake_load_config():  # noqa: N802 - テスト専用
        return {"fonts": {"search_dirs": [str(fonts_dir)]}}

    import util.utils as uu

    monkeypatch.setattr(uu, "load_config", _fake_load_config)

    # キャッシュをクリアしてからパスリストを取得
    from shapes.text import TextRenderer

    TextRenderer._font_paths = None  # type: ignore[attr-defined]
    paths = TextRenderer.get_font_path_list()

    # 先頭がダミーフォントであること（設定ディレクトリが優先される）
    assert isinstance(paths, list) and len(paths) >= 1
    assert Path(paths[0]) == dummy
