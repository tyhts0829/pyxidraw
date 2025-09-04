instructions = "Think in English, answer in Japanese."

# リポジトリ ガイドライン

## 破壊的変更（2025-09-03）

- **Geometry 統一:** `engine/core/geometry.py` の単一 `Geometry` 型に集約。変換は `translate/scale/rotate/concat` の純関数（新インスタンス返却）。
- **エフェクト関数化:** クラス継承は廃止し、`@effects.registry.effect` で登録する `Geometry -> Geometry` の関数のみ。
- **パイプライン一本化:** `E.pipeline ... .build()(g)` に統一（パイプライン単層キャッシュ）。
- **移行ガイド（主な置換）:** `size→scale`, `at→translate`, `spin→rotate(z=0..1→2π)`, `move→translate`, `grow→scale`。
- **使用例:**
  ```python
  from api import E, G
  g = G.sphere(subdivisions=0.5).scale(100,100,100).translate(100,100,0)
  result = (E.pipeline.displace(intensity=0.3).fill(density=0.5).build())(g)
  ```

### パイプライン仕様のシリアライズ/検証

- `to_spec(pipeline)`: `[{"name": str, "params": dict}]` に変換
- `from_spec(spec)`: 検証済み spec から `Pipeline` を生成
- `validate_spec(spec)`: 仕様を事前に検証（未登録名/不正パラメータを早期失敗）

```python
from api import E, to_spec, from_spec, validate_spec

pipeline = (E.pipeline.rotate(rotate=(0.25,0,0))
                      .displace(intensity=0.2)
                      .build())
spec = to_spec(pipeline)
validate_spec(spec)     # 例外が出なければOK
pipeline2 = from_spec(spec)
```

## プロジェクト構成とモジュールの整理

- `api/`: 高レベル API サーフェス — `E`（エフェクト）、`G`（シェイプ）、`runner.py`。
  - `pipeline.py`: `E.pipeline`（パイプライン構築と単層キャッシュ）
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
- 依存関係: README を参照（固定の `requirements.txt` はありません）。

### 開発向け Tips
- 形状生成の LRU キャッシュを無効化/調整:
  - `export PXD_CACHE_DISABLED=1`
  - `export PXD_CACHE_MAXSIZE=64`

## コーディングスタイルと命名規則

- Python 3.x、インデントは 4 スペース。可能な限り関数は小さく純粋に。
- 命名: ファイル/モジュールおよび関数は `snake_case`、クラスは `CamelCase`、定数は `UPPER_SNAKE_CASE`。
- 公開 API: docstring を追加し、型ヒントを推奨。
- エフェクト: `effects/` に配置し、`@effects.registry.effect` で登録（関数名が登録名）。
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
