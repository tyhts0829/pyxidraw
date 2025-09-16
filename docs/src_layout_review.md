# src ディレクトリ構成レビュー（2025-09-15）

本ドキュメントは `src/` 配下のディレクトリ構成と依存方向を俯瞰し、良い点と改善候補を短く整理したレビューです。対象リポジトリの現状観測に基づくもので、コード変更は一切行っていません。

## 対象と前提
- 対象: `src/` 以下のパッケージと主要モジュール（2025-09-15 時点）。
- 前提: `architecture.md` の L0–L3 レイヤリングと禁止エッジを規範とする。

## 構成の要約（トップレベル）
- `api/`: 公開面。`G`（形状ファクトリ）、`E`（エフェクト・パイプライン）、`shape/effect` デコレータ、スタブ `__init__.pyi` を提供。
- `engine/`: 実行系。`core/`（`Geometry` 等）、`render/`、`pipeline/`、`io/`、`ui/`、`monitor/` に細分。
- `effects/`: 加工（関数型 `Geometry -> Geometry`）。`registry.py` と各エフェクト関数群。
- `shapes/`: 生成（関数群）。`registry.py` と各シェイプ関数群。
- `common/`: 共有基盤。`BaseRegistry`、パラメータ正規化、型別名。
- `util/`: 低レベル汎用（数値処理・定数等）。
- `tools/`: 開発補助（スタブ生成ほか）。

簡易ツリー（抜粋）:
```
src/
  api/            # 公開API（G/E/shape/effect, pipeline builder）
  engine/
    core/         # Geometry, 変換, 基本クロック
    render/       # CPU→GPU 整形と描画
    pipeline/     # 実行用ワーカー/受信/バッファ等
    io/           # 入力（MIDI 等）
    ui/           # オーバーレイ表示
    monitor/      # 計測/サンプリング
  shapes/         # 形状（プラガブル）
  effects/        # エフェクト（プラガブル）
  common/         # 共有基盤（型・正規化・レジストリ）
  util/           # 汎用ユーティリティ
  tools/          # 開発補助スクリプト
```

## 依存方向の健全性（観測）
- `engine/* → api/*` の参照なし（禁止エッジ遵守）。
- `engine/* → (effects|shapes)/*` の参照なし（OK）。
- `common/` と `util/` は上位層へ依存せず（OK）。
- `api/*` から `engine.core/shapes/effects` の参照は意図通り。
- 循環依存は観測されず、レイヤリングは `architecture.md` と整合。

## 良い点
- 生成（shapes）/加工（effects）/表現（Geometry）/実行（engine）/公開（api）の責務が明確。
- レジストリ基盤（`common/base_registry.py`）により拡張点が統一。キー正規化やデコレータ API が対称。
- 公開 API の単一路線（`from api import G, E, shape, effect`）で利用導線がシンプル。
- `api/__init__.pyi` とスタブ生成スクリプトにより型・ドキュメント同期の素地がある。
- `engine` 配下の分割が適切で、並列処理/描画最適化の将来拡張に耐える。

## 気になる点 / 改善余地（構成観点）
- `tools/` の置き場: すでに `src/` 外配置。配布時はパッケージから除外する方針を README/architecture.md に明記したい。
- `helpers` vs `utils` の語感衝突: `engine/io/helpers.py` と `util/utils.py` は役割が近く見えやすい。用途が分かる命名へ寄せると衝突しにくい。
- 「パイプライン」の語の多義性: `engine/pipeline/`（実行）と `api.effects`（エフェクト適用）が読者には紛らわしい可能性。用語注記や README/architecture.md での語彙整流で解消可能。
- スケール時のディレクトリ増加: `shapes/` と `effects/` が増える場合、カテゴリ別サブパッケージ（例: 2D/3D/ノイズ系）導入の基準だけ決めておくと移行が円滑。

## 推奨アクション（軽微・非破壊）
- 命名整流（提案）
  - `engine/io/helpers.py` → 役割に即した名称（例: `cc_mapping.py`）へ。将来の `util/utils.py` との混同回避。
- 配布ポリシーの明記（docs）
  - 将来 PyPI 配布を想定するなら、`pyproject.toml` の `packages`/`exclude` 方針や `tools/` の扱いを README/architecture.md に追記。
- サブパッケージ化の基準を architecture.md に一行追記
  - 例）「shapes/effects が各20超 or 新規依存が必要になったカテゴリ出現時に導入」。
- 公開面の明確化（任意）
  - 内部用モジュールに `__all__` を置く/内部接頭辞を導入して露出面を安定化。

## architecture.md との同期状況（差分メモ）
- レイヤリング/禁止エッジは実装と整合。現時点で差分は観測されず（追補不要）。

---
更新履歴:
- 2025-09-15: 初版作成（観測のみ、変更なし）。
