どこで: `engine.ui.hud`（HUD サンプラ/オーバレイ）と `engine.render.renderer`（IBO/Indices キャッシュ統計）、`api.sketch`（HUD へのメトリクス配線）  
何を: HUD 上の `IBO` / `IDX` について、Effect/Shape の `E_CACHE` / `S_CACHE` と同様に Hit/Miss を検知し、バー表示でキャッシュ効率を可視化する実装改善計画  
なぜ: IBO/Indices キャッシュの効き具合を HUD から直感的に把握できるようにし、キャッシュ設定や描画パイプライン調整のフィードバックを取りやすくするため

# 背景 / 現状

- HUD メトリクス経路の概要
  - `MetricSampler`（`src/engine/ui/hud/sampler.py`）が頂点数/FPS/CPU/MEM などをサンプリングし、`data`（文字列）と `values`（数値）を保持。
  - `OverlayHUD`（`src/engine/ui/hud/overlay.py`）が `MetricSampler` の `data` をラベルとして表示し、`values` を `_normalized_ratio()` で 0..1 に正規化してメータ（バー）を描画。
  - メータは `HUDConfig.show_meters=True` のとき有効。
- Effect/Shape キャッシュ表示（既存）
  - Worker 側で `hud_metrics_snapshot()`（`src/api/sketch_runner/utils.py`）を `metrics_snapshot` として注入。
  - `engine.runtime.worker._execute_draw_to_packet()` が `before/after` の snapshot 差分から Effect/Shape の Hit/Miss を判定し、`cache_flags={"effect": "HIT"|"MISS", "shape": ...}` を `RenderPacket` に載せる。
  - `StreamReceiver`（`src/engine/runtime/receiver.py`）が `cache_flags` をメインスレッドへ渡し、`api.sketch.run_sketch()` 内の `_on_metrics()` が `sampler.data["E_CACHE"]` / `["S_CACHE"]` を `"HIT"` / `"MISS"` に更新。
  - `OverlayHUD._normalized_ratio()` は `key in ("S_CACHE", "E_CACHE")` のとき `status` 文字列を見て `MISS -> 1.0`, `HIT -> 0.0` としてバーを塗る（MISS 強調）。
- IBO / IDX 表示（既存）
  - `api.sketch.run_sketch()` が `line_renderer.get_ibo_stats()` と `engine.render.renderer.get_indices_cache_counters()` をラップした `_extra_metrics()` を定義し、`MetricSampler.set_extra_metrics_provider()` に登録。
  - `MetricSampler.tick()` は `extra_metrics_provider()` の戻り値から:
    - `IBO`: `reused` / `uploaded` / `indices_built` を `R:{r} U:{u} B:{b}` 形式の文字列として `self.data["IBO"]` に格納。
    - `IDX`: Indices LRU の `hits` / `misses` / `stores` / `evicts` / `size` を `H:{ih} M:{im} S:{is_} E:{ie} Z:{iz}` として `self.data["IDX"]` に格納。
  - IBO/IDX については `self.values[...]` には何も入れていないため、`OverlayHUD._normalized_ratio()` は常に `None` を返し、テキストのみが表示される（バー無し）。
- 課題
  - Effect/Shape は「直近フレームで MISS が発生したか」をバーで一目で確認できるが、IBO/Indices は累計カウンタの文字列のみで、Hit/Miss の有無や比率が視覚的に分かりづらい。
  - Indices LRU と IBO 固定化の効き具合をチューニングする際、MISS が多い状態を HUD 上で瞬時に認識しづらい。

# 目標（Goal）

- HUD 上の IBO / IDX についても「Hit/Miss の有無とバランス」を横棒メータで可視化する。
- Effect/Shape の `S_CACHE` / `E_CACHE` と同様、「MISS のときバーが大きくなる」直感的な表現を維持する。
- 実装は既存のメトリクス経路（`MetricSampler` / `OverlayHUD` / `HUDConfig`）を拡張する形に留め、キャッシュ本体の挙動や API 仕様は変えない。

# 非目標（Non‑Goals）

- Indices LRU や IBO 固定化ロジックそのもののアルゴリズム/ポリシー変更。
- キャッシュカウンタのリセット/永続化インターフェースの追加。
- HUD 全体のレイアウト/項目構成の大幅な見直し（必要最小限の項目追加に留める）。

# 設計方針

## 1. 判定粒度とデータソース

- 粒度
  - Effect/Shape と同様、「直近の一定期間で Hit/Miss がどれくらい発生したか」に基づいてバーを更新する。
  - 期間は HUD サンプル周期（`HUDConfig.sample_interval`、既定 0.5 秒）を単位とし、各サンプル間の差分カウンタから算出する。
- データソース
  - Indices LRU:
    - `renderer.get_indices_cache_counters()` が返す累計 `hits` / `misses`（+ `stores` / `evicts` / `size`）を利用。
  - IBO:
    - `LineRenderer.get_ibo_stats()` が返す累計 `reused` / `uploaded` / `indices_built` を利用。
  - いずれも既に `_extra_metrics()` → `MetricSampler.set_extra_metrics_provider()` 経由で HUD から参照できているため、新たな依存や API 追加は行わない。

## 2. IBO / IDX の Hit/Miss 判定ロジック

- Indices LRU（IDX）
  - 各 HUD サンプルで、前回サンプル時点の `hits` / `misses` を保持し、今回値との差分を取る:
    - `dh = hits_now - hits_prev`
    - `dm = misses_now - misses_prev`
  - 判定ルール（Effect/Shape と同じく MISS 優先）:
    - `dm > 0` かつ `dh + dm > 0` のとき:
      - 状態テキスト: 「MISS（dm ヒット / dh ミス）」等は HUD では直接は出さず、既存の `H:.. M:..` テキストを維持。
      - バー用の比率: `ratio_idx = dm / (dh + dm)`（0..1、1 に近いほど MISS 偏重）。
    - `dm == 0` かつ `dh > 0` のとき:
      - `ratio_idx = 0.0`（「このサンプル間は MISS なし、全て HIT」）。
    - `dh == 0` かつ `dm == 0` のとき:
      - 直近の `ratio_idx` を維持するか、`None`（バー非表示）とするかは後述のオープンクエスチョン。
- IBO
  - IBO 固定化は「既存 IBO を再利用できたか」が重要であり、以下のように解釈する:
    - Hit: `reused` の増分（`dr = reused_now - reused_prev`）。
    - Miss: 新規アップロード/構築の増分（`du = uploaded_now - uploaded_prev` と `db = indices_built_now - indices_built_prev` の合計）。
      - `dm_ibo = du + db`
  - 判定ルール:
    - `dm_ibo > 0` かつ `dr + dm_ibo > 0` のとき `ratio_ibo = dm_ibo / (dr + dm_ibo)`。
    - `dm_ibo == 0` かつ `dr > 0` のとき `ratio_ibo = 0.0`。
    - `dr == 0` かつ `dm_ibo == 0` のときは IDX 同様、前回値維持 or 非表示（後述）。

## 3. HUD メータへのマッピング

- `MetricSampler` 側
  - IBO/IDX の累計カウンタについて、前回値を保持するプライベートフィールドを追加:
    - 例: `_prev_idx_hits`, `_prev_idx_misses`, `_prev_ibo_reused`, `_prev_ibo_uploaded`, `_prev_indices_built` など。
  - `tick()` 内の「追加メトリクス（キャッシュ統計など、テキストのみ）」ブロックを拡張し:
    - 現在値と前回値から `ratio_idx` / `ratio_ibo` を計算。
    - 正常に計算できた場合のみ `self.values["IDX"]` / `self.values["IBO"]` に 0..1 の値として格納。
    - カウンタ取得に失敗・不正値・逆転（負の差分）の場合は、そのサイクルでは比率更新をスキップし、前回値を維持または削除。
  - 既存の `self.data["IBO"]` / `self.data["IDX"]` 文字列フォーマットはそのまま維持（互換性重視）。
- `OverlayHUD._normalized_ratio()` 側
  - 既存ロジック:
    - `S_CACHE` / `E_CACHE` の文字列を直接解釈して 0/1 に変換。
    - それ以外は `sampler.values[key]` を取得し、`CPU`/`MEM`/`FPS`/`VERTEX`/`LINE` について個別に 0..1 に正規化し、その他のキーは `None`（バー無し）。
  - 変更案:
    - `key in ("S_CACHE", "E_CACHE")` の分岐はそのまま維持。
    - それ以外のキーで `sampler.values.get(key)` が 0..1 の正規化済み値として格納されているケース（今回の `"IBO"` / `"IDX"`）については:
      - `v = self.sampler.values.get(key)` が `None` でなければ `return max(0.0, min(1.0, float(v)))` として汎用的に扱う。
      - 既存の `CPU`/`MEM` 等の個別分岐はこれまで通り（優先）。
    - これにより、`values` 側で 0..1 に正規化しておけば、今後も追加メトリクス（例: G-code バッファ使用率など）を同じ仕組みでバー表示できる。
- 視覚表現
  - `ratio_idx` / `ratio_ibo` は「MISS 割合」として解釈し、1.0 に近づくほどバーが右方向に伸びる（MISS 多い）。
  - Effect/Shape と同様に「MISS を強調する」方向性を維持しつつ、IBO/IDX では 0..1 の連続値で「どれくらい MISS が多いか」も読めるようにする。

## 4. 設定・表示仕様

- `HUDConfig.show_cache_status`
  - 現状どおり「Effect/Shape キャッシュステータス + IBO/IDX 統計」のまとめフラグとし、追加のフラグは導入しない（シンプルさ優先）。
  - IBO/IDX のバー表示も `show_cache_status=True` のときのみ有効。
- 項目順
  - `HUDConfig.resolved_order()` はこれまで通り `E_CACHE` → `S_CACHE` を既定順に含める。
  - IBO/IDX は `MetricSampler` の `data` にのみ現れるキーとして、現状通り「未知キーを末尾に追加」ロジックで表示順を決定（追加の order 設定は導入しない）。
- テキストフォーマット
  - `IBO` の `R: U: B:`、`IDX` の `H: M: S: E: Z:` 形式は維持し、バーはその右側に重ねて表示される。
  - 必要であれば、今後別計画として簡易ラベル（例: `IBO_CACHE` / `IDX_CACHE`）行を追加する可能性はあるが、本計画では scope 外。

# 実装タスク（チェックリスト）

- [x] 現状調査: IBO/IDX メトリクス経路（`get_ibo_stats` / `get_indices_cache_counters` → `_extra_metrics` → `MetricSampler` → `OverlayHUD`）の把握
- [x] Effect/Shape キャッシュ表示の Hit/Miss 判定経路（`hud_metrics_snapshot` / `metrics_snapshot` / `cache_flags` / `S_CACHE` / `E_CACHE`）の把握
- [ ] `MetricSampler` に IBO/IDX の前回カウンタ保持用フィールドを追加（初期値は取得時の現在値 or 0）
- [ ] `MetricSampler.tick()` の追加メトリクス処理を拡張し、IBO/IDX の差分から `ratio_ibo` / `ratio_idx` を計算して `self.values["IBO"]` / `["IDX"]` に格納
- [ ] 差分が取得できないケース（例外・型不整合・差分が負）でのフォールバック挙動を定義（比率更新スキップ + 直近比率維持 or None）
- [ ] `OverlayHUD._normalized_ratio()` を拡張し、「`S_CACHE` / `E_CACHE` 以外のキーでも `values` に 0..1 が入っていればそのままバーとして扱う」汎用ロジックを追加
- [ ] 必要に応じて `architecture.md`（HUD/キャッシュ可視化セクション）に IBO/IDX のバー表示仕様を追記
- [ ] 変更ファイル（`sampler.py` / `overlay.py` / `architecture.md` など）に対して `ruff` / `black` / `isort` / `mypy` を実行
- [ ] 動作確認: 簡単なスケッチで Indices LRU と IBO 固定化を有効にし、キャッシュヒット/ミスが発生しているケースで IBO/IDX のバーが期待通りに変化することを目視確認

# 検証（Acceptance Criteria）

- [ ] HUD 上で `IBO` / `IDX` 行に横棒メータが表示され、MISS が多いときほどバーが長くなる（MISS 割合が 0..1 で反映される）。
- [ ] キャッシュアクセスがほぼ完全にヒットする状況では、`IBO` / `IDX` のバーが 0 に近い状態を維持する。
- [ ] キャッシュが一切使われていない状況（Hits/Misses ともに増えない）でも HUD が例外を出さず、表示が破綻しない。
- [ ] 既存の `S_CACHE` / `E_CACHE` の Hit/Miss 表示とバー挙動に回 regress がない（従来通り MISS=塗りつぶし、HIT=空）。
- [ ] `HUDConfig.show_cache_status=False` のとき、IBO/IDX のテキストおよびバーが表示されない。

# オープンな確認事項（要オーナー確認）

- [ ] IBO/IDX のバーは「サンプル区間内の MISS 割合（0..1）」でよいか  
      - 代案: Effect/Shape と完全に揃えて「MISS が 1 件以上あれば 1.0、それ以外は 0.0」の二値バーとする案もあり。
- [ ] IBO の Hit/Miss 解釈として「`reused` を Hit、`uploaded + indices_built` を Miss とみなす」定義で問題ないか  
      - 代案: `indices_built` のみを Miss と見なし、`uploaded` は別指標として扱うなど。
- [ ] カウンタ差分が 0（活動なし）の場合、バーは「前回比率を保持」か「非表示（0 または None）」のどちらが好ましいか  
      - 現状案: 前回値維持（HUD 上で急にバーが消えるのを避ける）。
- [ ] IBO/IDX 用に専用ラベル行（例: `IBO_CACHE` / `IDX_CACHE`）を追加するか、それとも既存の `IBO` / `IDX` 行にバーを重ねるだけに留めるか（現計画では後者を想定）。

# 開発メモ / 実行コマンド（編集ファイル限定）

- Lint: `ruff check --fix src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py`
- Format: `black src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py && isort src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py`
- Type: `mypy src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py`
- Test: `pytest -q -k "hud or indices_cache or ibo"`（既存テストが無ければ、最小限の HUD 関連テスト追加を検討）

以上、この計画内容で問題なければ、次のステップとして実装に進みます。

