# 251106: Lazy/realize に起因する描画遅延の設計改善計画（チェックリスト）

- 背景: `docs/plans/deferred_geometry_metadata.md` に従い Lazy を導入後、描画が遅くなった。原因として、描画以外での暗黙 `realize()` と HUD サンプリング、ならびに実装上の真偽判定が `LazyGeometry.__len__` を通じて `realize()` を誘発している点を特定。
- 目的: 「描画のための 1 回だけ実体化」に収束させ、不要な実体化と計測由来の重複実体化を排除する。
- スコープ: 下記 A/B のみ（まずは最小修正）。
- 非スコープ（別提案）: 大量の `print` を `logging` へ移行、`Pipeline` 署名計算の抑制などは別PRで提案。

---

【A】Renderer の真偽判定修正（暗黙 `realize()` 回避）

- 現状と問題
  - 場所: `src/engine/render/renderer.py:126`
  - 現状: `if not geometry:` により `LazyGeometry.__len__()` → `as_arrays()` → `realize()` が発火。
  - 影響: 直後に `isinstance(geometry, LazyGeometry): geometry = geometry.realize()` もあるため、意図より前の地点で1回余計に実体化が走る（キャッシュは効いてもコスト発生）。

- 対応方針（最小）
  - [x] `if geometry is None:` に変更し、`LazyGeometry` の真偽判定経由を封じる。
  - [x] 可能なら順序を「Lazy 判定 → realize → 空判定」に変更し、空ジオメトリの早期 return を許容。

- 変更詳細（想定差分）
  - [x] `src/engine/render/renderer.py:126` を `if geometry is None:` に変更。
  - [x] 次行以降の順序を以下に整理:
    - `if isinstance(geometry, LazyGeometry): geometry = geometry.realize()`
    - （任意最適化）`if geometry.n_vertices == 0 or geometry.n_lines == 0: self.gpu.index_count = 0; return`

- 受け入れ基準（DoD）
  - [ ] HUD 無効時、実体化はアップロード直前の 1 回のみ。
  - [ ] 空ジオメトリ（0ライン）で早期 return し、`LineRenderer.draw()` がスキップされる（`index_count==0`）。

- リスク/ロールバック
  - 既存コードの「空を FBO へアップロードしても問題なし」という挙動が変わる可能性。問題が出たら早期判定は戻し、`is None` 変更のみ残す。

- 観測/計測
  - [ ] `LineRenderer` に簡易カウンタを一時追加（ローカル計測）し、1フレームあたりの `realize()` 回数が 1 回であることを確認。

---

【B】HUD サンプリングの方針変更（Renderer 公開値を参照）

- 現状と問題
  - 場所: `src/engine/ui/hud/sampler.py:159-177`（頂点数）, `:189-196`（ライン数）
  - 現状: HUD が `SwapBuffer.get_front()` の `geometry` から `n_vertices`/`n_lines` を直接読み、`LazyGeometry` では `as_arrays()` → `realize()` を誘発。描画用の実体化と重複。

- 対応方針
  - LazyGeometry に対する HUD 計測はスキップ（0 または N/A 表示）。
  - GPU アップロード後の実測値（頂点/ライン）を `LineRenderer` が公開し、HUD はそれを参照する。

- 変更詳細（段階導入）
  1) LineRenderer 側
     - [x] `LineRenderer` に直近アップロードの頂点/ライン数を保持するフィールドを追加（例: `_last_vertex_count: int`, `_last_line_count: int`）。
     - [x] `_upload_geometry()` 内で `verts`/`inds` から値を更新。ライン数は `num_lines = total_inds - total_verts` または `offsets` 起点で算出。
     - [x] 公開アクセサ（例: `get_last_counts() -> tuple[int,int]`）を追加。
  2) MetricSampler 側
     - [x] `MetricSampler` に `counts_provider: Callable[[], tuple[int,int]] | None` を追加。
     - [x] `counts_provider` があればそれを優先して `VERTEX`/`LINE` を更新。無ければ従来ロジック。ただし `isinstance(geometry, LazyGeometry)` の場合は実体化を禁止し、0 あるいは `N/A` を表示（表示テキストは `0` とし、将来 `N/A` 化も可）。
  3) 結線（sketch ランナー）
     - [x] `api.sketch` で `LineRenderer` 生成後に `counts_provider` を `MetricSampler` に注入する。

- 受け入れ基準（DoD）
  - [ ] HUD 有効時でも、描画以外の `realize()` が発生しない（`renderer` 側の公開値参照のみ）。
  - [ ] HUD 表示の頂点/ライン数が `LineRenderer` のアップロードに同期し、妥当な値になる。

- 表示仕様
  - 初期状態（アップロード前）は 0 表示。
  - 将来オプション: `HUDConfig` に `vertex_from_renderer: bool = True` のようなフラグ化（互換のため当面は内部実装で吸収）。

- リスク/ロールバック
  - `renderer` からの公開経路が未更新のまま HUD が参照すると 0 表示が続く。段階導入中は許容。

- 観測/計測
  - [ ] HUD 有効/無効で `realize()` 回数とフレーム時間を比較し、HUD 有効化によるオーバーヘッドが無視可能に低下したことを確認。

---

【関連ファイル（参照）】
- `src/engine/render/renderer.py:126` — 真偽判定の変更ポイント
- `src/engine/ui/hud/sampler.py:159-177` — 頂点数の算出（現状）
- `src/engine/ui/hud/sampler.py:189-196` — ライン数の算出（現状）
- `src/engine/runtime/buffer.py:1` — `SwapBuffer`（HUD がフロントを参照）

---

【追加の改善提案（別PRで検討）】
- ログ出力の最適化
  - [ ] `print` 多用箇所（`api/effects.py`, `api/lazy_signature.py`, `engine/runtime/receiver.py`, `api/sketch.py`, `api/sketch_runner/utils.py`）を `logging` へ移行し、既定では抑制。
- 署名計算の抑制
  - [ ] `Pipeline.__call__` で `_cache_maxsize != 0` のときのみ `lazy_signature_for(lg)` を計算（無効時はスキップ）。

---

【C】終端実体化の場所を Worker 側へオフロード（描画スレッドの負荷削減）

- 背景: Lazy 導入により実体化が Renderer 側に寄り、GPU アップロード前の重計算がメインスレッドに流入してフレーム落ちを誘発。
- 対応:
  - [x] `engine/runtime/worker.py` で `draw_callback` の戻り値が `LazyGeometry` の場合に `geometry.realize()` を実行してから `RenderPacket` に詰める（プロセス/インライン両経路）。
- 受け入れ基準:
  - [x] `[REALIZE] ... from=worker.py:...` としてワーカ側からの実体化ログが出る（Renderer からは出ない）。

---

【進捗トラッキング】
- A: Renderer 真偽判定修正 … [x] 完了
- B: HUD サンプリング変更 … [x] 完了（Renderer計数参照 + Lazy実体化回避）
- 追加提案（ログ/署名） … [ ] 未着手

---

【実施ルール（AGENTS.md 準拠）】
- 変更単位ごとに ruff/black/isort/mypy/pytest（対象限定）を実施。
- 変更が公開 API に波及する場合はスタブを再生成し、テストの整合を確認。
- 依存追加/破壊的変更/全体テスト実行は Ask-first。

---

承認可否と補足要望をご指示ください。承認後、本チェックリストに従って実装→各項目にチェックを付けて進捗を可視化します。
