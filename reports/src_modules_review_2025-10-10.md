# src/ 以下モジュール群レビュー（美しさ/簡潔さ/可読性）

- 日付: 2025-10-10
- 対象: `src/engine/*`, `src/api/*`, `src/common/*`, `src/util/*`, `src/shapes/*`, `src/effects/*`

## 総評

- 単一の `Geometry` 表現を中核に、生成（shapes）/変換（Geometry）/加工（effects）/描画（render）/実行（runtime）が綺麗に分離されている。
- パラメータ量子化（float のみ）を鍵にした決定的なキャッシュ設計（shape LRU・pipeline 単層 LRU）が一貫し、読みやすく拡張しやすい。
- オプショナル依存（moderngl/pyglet/mido）や環境差へのフェイルソフト設計が徹底され、導入コストが低い。
- ドキュストリングは「どこで/何を/なぜ」の形式で統一され、型注釈も適切。保守者が流れを追いやすい。

## 良い点（美しさ/簡潔さ/可読性）

- Geometry の不変条件と純関数 API が明快。`as_arrays(copy=False)` が読み取り専用ビューを返すのは安全設計として秀逸。
  - 参照: `src/engine/core/geometry.py`
- レジストリ層（shapes/effects）が薄く、`BaseRegistry` でキー正規化ポリシーを共有。拡張容易で対称性が保たれている。
  - 参照: `src/common/base_registry.py`, `src/effects/registry.py`, `src/shapes/registry.py`
- パラメータ量子化と署名生成が simple で統一的。float のみ量子化という仕様が実装と文書で同期。
  - 参照: `src/common/param_utils.py`
- API ファサード（`G`/`E`/`Pipeline`）が薄く、UI/ランタイム有無に関わらず「実値」で扱える。
  - 参照: `src/api/shapes.py`, `src/api/effects.py`
- ワーカ/受信/ダブルバッファの責務分離がシンプルで堅牢。例外ラップ・停止手順も明確。
  - 参照: `src/engine/runtime/worker.py`, `src/engine/runtime/receiver.py`, `src/engine/runtime/buffer.py`
- 描画層は遅延 import・可変リソースの寿命管理が明快。`LineRenderer` と `LineMesh` の分離も適切。
  - 参照: `src/engine/render/renderer.py`, `src/engine/render/line_mesh.py`

## 改善提案（重要度 高→低）

1) moderngl のオプショナル化を `line_mesh` にも反映
- 現状: 直接 `import moderngl as mgl` により未導入環境で import 失敗の可能性。
  - 参照: `src/engine/render/line_mesh.py`
- 提案: `renderer` と同様に try-import か、コンストラクタ内で遅延 import に変更（型は forward 参照）。

2) 用語統一: 「DoubleBuffer」→「SwapBuffer」
- 現状: ドキュメント/コメントに DoubleBuffer の表記が混在。
  - 参照例: `src/engine/render/renderer.py` の docstring/コメント
- 提案: クラス名に合わせ「SwapBuffer」で統一し、認知負荷を低減。

3) `_clamp01` の重複を解消
- 現状: 同等の小関数が複数箇所に重複。
  - 参照: `src/engine/render/renderer.py`（2 箇所）
- 提案: `util.color._clamp01` を流用するか、小ユーティリティへ集約。

4) 内部属性への直接アクセスを避ける（IO/MIDI）
- 現状: `DualKeyDict` の内部属性（`_str_to_value`/`_reverse_cc_map`）へ直接アクセスしてログ出力。
  - 参照: `src/engine/io/service.py`
- 提案: `DualKeyDict` 側に公開アクセサ（例: `items_by_name()`）を用意し隠蔽する。

5) shapes API の重複ヘルパ整理
- 現状: `_generate_shape_resolved` と `_generate_shape` が類似責務で併存、後者は未使用。
  - 参照: `src/api/shapes.py`
- 提案: 1 本へ集約し簡潔化。

6) ValueResolver の大きめ関数を小分割
- 現状: `resolve()` にマージ/型判定/ヒント構築/登録が集約され長め。
  - 参照: `src/engine/ui/parameters/value_resolver.py`
- 提案: ラベル生成・ヒント抽出を独立ヘルパへ切り出し、責務を明確化（50–80 行単位へ）。

7) 遅延 import 方針の明文化
- 現状: ファイルごとに try-import/遅延 import の方針が混在。
- 提案: architecture.md へ「オプショナル依存はモジュール import を破らない」方針と具体例を短く追記。

## 細かな観察/小改善

- shapes API のキャッシュ鍵（量子化後 `params_signature`）について、docstring に 1 行補足があると初見者に優しい。
  - 参照: `src/api/shapes.py`
- `ParameterStore.set_override()` は同値再設定でも通知が走る可能性。変更検出で通知を抑制してもよい（任意）。
  - 参照: `src/engine/ui/parameters/state.py`
- `engine/export/image.py` の NotImplementedError 方針は明確。引数検証（`include_overlay=False` で `draw`/`mgl_context` 必須）も読みやすい。
- `engine/export/gcode.py` は最小機能ながら、`y_down` の厳密反転（キャンバス高基準）やヘッダ/フッタが簡潔で良い。将来の `writer` 実装接続にも備えられている。

## まとめ

- 現状でも美しく読みやすいコードベース。上記の「import 遅延/用語統一/小さな重複除去/内部属性非公開化」を当てるだけで、さらに洗練される。
- 実装を進める場合の優先順は以下が推奨:
  1. `line_mesh` の import 改善
  2. 「DoubleBuffer」→「SwapBuffer」用語統一
  3. `_clamp01` の重複解消
  4. IO/MIDI の内部属性非公開化
  5. shapes API の未使用ヘルパ整理
  6. ValueResolver の小分割
  7. 遅延 import 方針の明文化

> 注: 本ファイルはレビュー結果の要約であり、修正は未実施です。必要ならチェックリスト化して段階的に対応します。

### 追記（2025-10-11）: SwapBuffer 識別子リネーム影響箇所（概要）

- 目的: 用語統一に合わせ、`double_buffer` 系識別子を `swap_buffer` へ改名。
- 影響箇所（主要）:
  - `src/engine/render/renderer.py:41, 46, 50, 84, 85`
  - `src/engine/runtime/receiver.py:21, 32`
  - `src/api/sketch.py:414`
- 除外（OpenGL 設定）:
  - `src/engine/core/render_window.py:38` の `Config(double_buffer=True)` は別概念。
- 詳細チェックリスト: `reports/plan_swapbuffer_identifier_rename.md`
