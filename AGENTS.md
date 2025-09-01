# リポジトリ ガイドライン

## プロジェクト構成とモジュールの整理
- `api/`: 高レベル API サーフェス — `E`（エフェクト）、`G`（シェイプ）、`runner.py`。
- `shapes/`: シェイプ生成（例: `sphere.py`, `polyhedron.py`, `torus.py`）。
- `effects/`: エフェクト実装（例: `noise.py`, `rotation.py`, `filling.py`）。
- `engine/`: コア幾何、I/O（MIDI）、レンダリング内部。
- `benchmarks/`: ベンチマーク CLI（`python -m benchmarks`）と可視化。
- `tests/`: モジュールに対応した Pytest スイート。`pytest.ini` と `conftest.py` を参照。
- ルートの例: `main.py`（フルデモ）、`simple.py`（最小）。設定: `config.yaml`。

## ビルド・テスト・開発コマンド
- デモ実行: `python main.py`（ヘッドレス実行は `--no-midi` を追加）。
- 簡易デモ: `python simple.py`。
- テスト（全体/一部）: `python -m pytest` / `python -m pytest tests/test_noise.py -q`。
- カバレッジ: `python -m pytest --cov=. --cov-report=term`。
- ベンチマーク: `python -m benchmarks run`（`benchmarks/__main__.py` を参照）。
- 依存関係: README と CLAUDE.md を参照。固定の `requirements.txt` はありません。

## コーディングスタイルと命名規則
- Python 3.x、インデントは 4 スペース。可能な限り関数は小さく純粋に。
- 命名: ファイル/モジュールおよび関数は `snake_case`、クラスは `CamelCase`、定数は `UPPER_SNAKE_CASE`。
- 公開 API: docstring を追加し、型ヒントを推奨。
- エフェクト: `effects/` に配置し、必要に応じて `@E.register()` で登録（関数名が登録名）。
- シェイプ: `shapes/` に追加し、`@shape` デコレータで登録（クラス名が登録名）。

## テストガイドライン
- フレームワーク: `pytest` は `pytest.ini` で構成（テストルートは `tests/`）。
- テストファイル: `tests/test_*.py`。テスト名は説明的に（例: `test_subdivision_handles_degenerate_faces`）。
- 新しいシェイプ/エフェクトや重要ユーティリティには単体テストを追加。可能ならモジュールパスをミラー。
- 再現性のため `--no-midi` 経路を使用。ハードウェア依存を避ける。

## コミットとプルリクエストのガイドライン
- コミット: 既存履歴の gitmoji プレフィックスを使用（例: ✨ feature, 🐛 fix, 🎨 style, ⚡️ perf, 🔥 remove, 📝 docs, 🚧 WIP, ⚰️ deprecate）＋短く命令形のサブジェクト。
- PR: 明確な要約、根拠、関連 Issue、テスト計画、レンダリング変更の前後スクリーンショットを含める。`config.yaml` の変更は明記。

## セキュリティと設定のヒント
- 秘密情報や個人デバイスのマッピングをコミットしない。ローカルの MIDI 設定は VCS 外に保つ。
- `config.yaml` の編集は検証し、安全なデフォルトを提供。破壊的変更は文書化。
