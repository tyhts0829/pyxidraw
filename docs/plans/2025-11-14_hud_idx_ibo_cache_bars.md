どこで: `engine.ui.hud`（HUD サンプラ/オーバレイ）と `engine.render.renderer`（IBO/Indices キャッシュ統計）、`api.sketch`（HUD へのメトリクス配線）  
何を: HUD に `IBO_CACHE` / `IDX_CACHE` という専用ラベル行を追加し、Effect/Shape の `E_CACHE` / `S_CACHE` と同様に IBO/Indices キャッシュの Hit/Miss を検知し、「MISS=1, HIT=0」の二値バーで可視化する実装改善計画  
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

- HUD 上に IBO/IDX キャッシュ専用の `IBO_CACHE` / `IDX_CACHE` 行を追加し、「Hit/Miss の有無」を横棒メータで可視化する。
- 既存の `IBO` / `IDX` 行における R/U/B や H/M/S/E/Z といった文字表示と、それに紐づく Hit/Miss 検知に関与しない古い累積カウンタ表示/ロジックを HUD から取り除き、IBO/IDX のキャッシュ状態は `IBO_CACHE` / `IDX_CACHE` 行のバーに一本化する。
- Effect/Shape の `S_CACHE` / `E_CACHE` と同様、「MISS が 1 回でもあったサンプルではバー=1.0、それ以外は 0.0」という二値表現を維持しつつ、実装は既存のメトリクス経路（`MetricSampler` / `OverlayHUD` / `HUDConfig`）を拡張する形に留め、キャッシュ本体の挙動や API 仕様は変えない。

# 非目標（Non‑Goals）

- Indices LRU や IBO 固定化ロジックそのもののアルゴリズム/ポリシー変更。
- キャッシュカウンタのリセット/永続化インターフェースの追加。
- HUD 全体のレイアウト/項目構成の大幅な見直し（`IBO_CACHE` / `IDX_CACHE` 行の追加と旧 `IBO` / `IDX` 行の整理に留める）。

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
  - いずれも既に `_extra_metrics()` → `MetricSampler.set_extra_metrics_provider()` 経由で HUD から参照できているため、新たな依存や API 追加は行わない。IBO/IDX の生の累計カウンタは HUD のラベル行としては用いず、`IBO_CACHE` / `IDX_CACHE` の Hit/Miss 判定用の内部入力としてのみ利用する。

## 2. IBO / IDX の Hit/Miss 判定ロジック（二値・MISS 優先）

- Indices LRU（IDX_CACHE）
  - HUD では「Indices LRU の MISS が発生したサンプルだけを MISS として強調したい」ため、`misses` の差分のみを見るシンプルな定義とする:
    - `dm = misses_now - misses_prev`
  - 判定ルール（二値、MISS 優先）:
    - `dm > 0` のとき `value_idx = 1.0`（このサンプル間に LRU ミスが 1 回以上発生した＝MISS 扱い）。
    - `dm == 0` のとき `value_idx = 0.0`（このサンプル間には LRU ミスが発生していない＝HIT 扱い）。
    - `misses` 差分の取得に失敗した場合は前回値維持などのフォールバック扱い。
- IBO（IBO_CACHE）
    - HUD では「IBO をアップロードしたフレームだけを MISS として強調したい」ため、`uploaded` の差分のみを見るシンプルな定義とする:
      - `du = uploaded_now - uploaded_prev`
    - 判定ルール（二値、MISS 優先）:
      - `du > 0` のとき `value_ibo = 1.0`（このサンプル間に IBO のアップロードが発生した＝コストを払ったフレームとして MISS 扱い）。
      - `du == 0` のとき `value_ibo = 0.0`（このサンプル間に IBO のアップロードは発生しておらず、少なくとも「IBO を張り替えるコスト」は払っていない＝HIT 扱い）。
      - `uploaded` 差分の取得に失敗した場合は IDX 同様、前回値維持などのフォールバック扱い。

## 3. HUD メータへのマッピング

- `MetricSampler` 側
  - IBO/IDX の累計カウンタについて、前回値を保持するプライベートフィールドを追加:
    - 例: `_prev_idx_misses`, `_prev_ibo_uploaded` など。
  - `tick()` 内の「追加メトリクス（キャッシュ統計など、テキストのみ）」ブロックを拡張し:
    - 現在値と前回値から `dm` / `du` を計算し、上記ルールに従って `value_idx` / `value_ibo` を 0.0 または 1.0 に決定。
    - 正常に計算できた場合のみ `self.values["IDX_CACHE"]` / `self.values["IBO_CACHE"]` に 0.0/1.0 の値として格納。
    - カウンタ取得に失敗・不正値・逆転（負の差分）の場合は、そのサイクルでは値更新をスキップし、前回値を維持または削除。
  - `self.data` 側には `self.data.setdefault("IBO_CACHE", "IBO_CACHE")`、`self.data.setdefault("IDX_CACHE", "IDX_CACHE")` のような専用ラベル行のみを用意し、従来の `self.data["IBO"]` / `self.data["IDX"]` における `R: U: B:` / `H: M: S: E: Z:` 形式の文字列は HUD から削除する。
- `OverlayHUD._normalized_ratio()` 側
  - 既存ロジック:
    - `S_CACHE` / `E_CACHE` の文字列を直接解釈して 0/1 に変換。
    - それ以外は `sampler.values[key]` を取得し、`CPU`/`MEM`/`FPS`/`VERTEX`/`LINE` について個別に 0..1 に正規化し、その他のキーは `None`（バー無し）。
  - 変更案:
    - `key in ("S_CACHE", "E_CACHE")` の分岐はそのまま維持。
    - それ以外のキーで `sampler.values.get(key)` が 0.0 または 1.0 の二値として格納されているケース（今回の `"IBO_CACHE"` / `"IDX_CACHE"`）については:
      - `v = self.sampler.values.get(key)` が `None` でなければ `return max(0.0, min(1.0, float(v)))` として汎用的に扱う（二値だが 0..1 の範囲という点では互換）。
      - 既存の `CPU`/`MEM` 等の個別分岐はこれまで通り（優先）。
    - これにより、`values` 側で 0.0/1.0 を用いる `IBO_CACHE` / `IDX_CACHE` だけでなく、将来的に 0..1 正規化済みの追加メトリクスをバー表示する場合にも同じ仕組みを使える。
- 視覚表現
  - `value_idx` / `value_ibo` は「このサンプル区間に MISS が発生したかどうか」を表す二値（1.0=MISS あり, 0.0=MISS なし）とする。
  - Effect/Shape と同様に「MISS を強調する」方向性を維持し、IBO/IDX でも「MISS が出たフレームでバーが一杯に塗られる」挙動に揃える。

## 4. 設定・表示仕様

- `HUDConfig.show_cache_status`
  - 現状どおり「Effect/Shape キャッシュステータス + IBO/IDX キャッシュステータス」のまとめフラグとし、追加のフラグは導入しない（シンプルさ優先）。
  - IBO/IDX のバー表示（`IBO_CACHE` / `IDX_CACHE` 行）も `show_cache_status=True` のときのみ有効。
- 項目順
  - `HUDConfig.resolved_order()` はこれまで通り `E_CACHE` → `S_CACHE` を既定順に含める。
  - `IBO_CACHE` / `IDX_CACHE` は `MetricSampler` の `data` に現れる追加キーとして、現状通り「未知キーを末尾に追加」ロジックで表示順を決定（追加の order 設定は導入しない）。
- テキストフォーマット
  - 既存の `IBO` の `R: U: B:`、`IDX` の `H: M: S: E: Z:` 形式のテキスト表示は HUD から削除する。
  - `IBO_CACHE` / `IDX_CACHE` 行はシンプルなラベル文字列（例: `"IBO_CACHE"` / `"IDX_CACHE"`）のみを表示し、Hit/Miss の状態はバー（0.0/1.0）のみで表現する。

# 実装タスク（チェックリスト）

- [x] 現状調査: IBO/IDX メトリクス経路（`get_ibo_stats` / `get_indices_cache_counters` → `_extra_metrics` → `MetricSampler` → `OverlayHUD`）の把握
- [x] Effect/Shape キャッシュ表示の Hit/Miss 判定経路（`hud_metrics_snapshot` / `metrics_snapshot` / `cache_flags` / `S_CACHE` / `E_CACHE`）の把握
- [x] `MetricSampler` に IBO/IDX の前回カウンタ保持用フィールドを追加（初期値は取得時の現在値 or 0）
- [x] `MetricSampler.tick()` の追加メトリクス処理を拡張し、IBO/IDX の差分から二値の `value_ibo` / `value_idx` を計算して `self.values["IBO_CACHE"]` / `["IDX_CACHE"]` に格納
- [x] 差分が取得できないケース（例外・型不整合・差分が負）でのフォールバック挙動を定義（値更新スキップ + 直近値維持 or None）
- [x] `OverlayHUD._normalized_ratio()` を拡張し、「`S_CACHE` / `E_CACHE` 以外のキーでも `values` に 0.0/1.0 が入っていればそのままバーとして扱う」汎用ロジックを追加
- [ ] 必要に応じて `architecture.md`（HUD/キャッシュ可視化セクション）に IBO/IDX のバー表示仕様を追記
- [x] 変更ファイル（`sampler.py` / `overlay.py` / `architecture.md` など）に対して `ruff` / `black` / `isort` / `mypy` を実行
- [ ] 動作確認: 簡単なスケッチで Indices LRU と IBO 固定化を有効にし、キャッシュヒット/ミスが発生しているケースで IBO/IDX のバーが期待通りに二値で変化することを目視確認

# 検証（Acceptance Criteria）

- [ ] HUD 上で `IBO_CACHE` / `IDX_CACHE` 行に横棒メータが表示され、サンプル区間内に MISS が 1 回でもあればバーがフル（1.0）、MISS がなく HIT のみのときは 0.0 になる（Effect/Shape の `S_CACHE` / `E_CACHE` と同じ二値挙動）。
- [ ] キャッシュアクセスがほぼ完全にヒットする状況では、`IBO_CACHE` / `IDX_CACHE` のバーが 0.0 の状態を維持する。
- [ ] キャッシュが一切使われていない状況（Hits/Misses ともに増えない）でも HUD が例外を出さず、表示が破綻しない。
- [ ] 既存の `S_CACHE` / `E_CACHE` の Hit/Miss 表示とバー挙動に回 regress がない（従来通り MISS=塗りつぶし、HIT=空）。
- [ ] `HUDConfig.show_cache_status=False` のとき、`IBO_CACHE` / `IDX_CACHE` のテキストおよびバーが表示されない。

# オープンな確認事項（要オーナー確認）

- [x] IBO/IDX のバーは Effect/Shape と同様「MISS が 1 件以上あれば 1.0、それ以外は 0.0」の二値バーとする  
      - 決定: 連続値の「MISS 割合」ではなく、`S_CACHE` / `E_CACHE` と同じ二値挙動に揃える。
- [x] IBO の Hit/Miss 解釈として「IBO を再利用したフレーム（`reused` 増加）を Hit、IBO を再構築したフレーム（`indices_built` 増加）を Miss」とみなす  
      - 補足: 実質的に「IBO を再利用したか／再計算してアップロードしたか」の 2 状態のみを MISS/HIT で表現する。`uploaded` は「IBO を GPU に送った回数」として別メトリクスで扱い、Hit/Miss 判定には含めない。
- [x] カウンタ差分が 0（活動なし）の場合、バーは「前回値を保持」か「非表示（0 または None）」のどちらが好ましいか  
      - 決定: `S_CACHE` / `E_CACHE` と同様に「前回値維持」とする（活動なし区間でバーが急に消えないようにする）。IBO/IDX も同じポリシーで扱う。
- [x] IBO/IDX 用に専用ラベル行（例: `IBO_CACHE` / `IDX_CACHE`）を追加するか、それとも既存の `IBO` / `IDX` 行にバーを重ねるだけに留めるか（現計画では後者を想定）。
      - 決定: HUD に `IBO_CACHE` / `IDX_CACHE` という専用ラベル行を追加し、そこで IBO/IDX の HIT/MISS バーを表示する。既存の `IBO` / `IDX` 行における R/U/B などの文字表示は廃止し、HIT/MISS 検知に関与しない古いカウンタや表示ロジックは実装から削除してクリーンな構成にする。

# 開発メモ / 実行コマンド（編集ファイル限定）

- Lint: `ruff check --fix src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py`
- Format: `black src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py && isort src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py`
- Type: `mypy src/engine/ui/hud/sampler.py src/engine/ui/hud/overlay.py`
- Test: `pytest -q -k "hud or indices_cache or ibo"`（既存テストが無ければ、最小限の HUD 関連テスト追加を検討）

以上、この計画内容で問題なければ、次のステップとして実装に進みます。
