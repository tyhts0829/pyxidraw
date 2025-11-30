# pyxidraw 実装改善計画（ドラフト）

このファイルは「src 配下の大規模な根本リファクタリング」のための作業計画とチェックリストです。
**現時点では方針ドラフトであり、あなたの承認後に実装へ着手します。**

---

## フェーズ0: 前提整理・安全網

- [ ] 現状アーキテクチャの再確認（`architecture.md`, `src/api/*`, `src/engine/*` を読み合わせ）
- [ ] 既存テストスイートとマーカーの把握（smoke / integration / e2e / perf / optional）
- [ ] 主要ユースケースの洗い出し（`main.py`, `demo/`, `sketch/` から典型パターンを列挙）
- [ ] 破壊的変更を許容する範囲の明示（API 互換 / 振る舞い / ファイル構成）
- [ ] リファクタリング単位ごとの検証コマンドを決める（対象モジュール単位で `ruff/mypy/pytest`）

---

## フェーズ1: `run_sketch` / ランナー周りの分割・単純化

ターゲット: `src/api/sketch.py` と、その配下の `sketch_runner/*`

- [ ] `run_sketch` の責務を列挙し、サブコンポーネントに分解（MIDI / WorkerPool / HUD / Export / 録画 / シグナル）
- [ ] 既存 `_RuntimeContext` の役割を見直し、ランナー用クラス（例: `SketchRuntime`）に統合して状態を集約
- [ ] 「状態を抱えるランナー」をクラス（例: `SketchRuntime`）または小さなモジュール群に切り出す
- [ ] pyglet / ModernGL / MIDI / Export の初期化と後始末を、それぞれ専用ヘルパに移動
- [ ] `init_only` 動作を明確化し、ヘッドレス検証用に最小の初期化経路を用意
- [ ] `signal.signal` / `atexit.register` / `sys.excepthook` 差し替えをオプトイン機構へ変更し、可能な限り CLI エントリポイント側（`main.py` など）へ移動
  - [ ] API としてフラグ（例: `install_signal_handlers: bool`）を導入するかどうか検討
- [ ] `run_sketch` を「薄いファサード（パラメータ整形 + ランナー起動）」に縮小
- [ ] 「とりあえず `except Exception: logger.debug`」で握りつぶしている箇所を洗い出し、原則として例外をそのまま送出するか、明示的なエラー通知に変更
- [ ] ランナー周辺のテスト用エントリ（ヘッドレス・短時間実行）を用意

---

## フェーズ2: グローバル状態とサービスロケータの整理

ターゲット: `src/util/cc_provider.py`, `src/util/palette_state.py`,
`src/api/effects.py`, `src/api/shapes.py`, `engine.ui.parameters.get_active_runtime` など

- [ ] 現在のグローバル状態（CC スナップショット / パレット / ParameterRuntime / HUD 設定）の一覧化
- [ ] 「誰がいつ設定し、誰が読むか」のライフサイクルをテキストで明示
- [ ] グローバル関数ベースのサービスロケータを、明示的なコンテキストオブジェクトに寄せる方針を設計
- [ ] ランタイムコンテキスト（例: `RuntimeContext`）の責務とフィールドを定義
- [ ] `util.cc_provider` / `util.palette_state` の getter/setter を統合し、「どこで設定されるか」が分かる小さな集約モジュールとドキュメントを用意
- [ ] `get_active_runtime()` 依存箇所を洗い出し、引数・コンテキスト注入への置き換え計画を作成
- [ ] CC / パレットの共有を、`run_sketch` 起動時に明示的に組み立てる形へ寄せる
- [ ] 並列テストや複数スケッチ同時実行でのグローバル衝突リスクを軽減
- [ ] `_PipelineRuntimeAdapter` / `_ShapeCallContext` などの小さなコンテキストクラスを runtime 層に集約し、「どこが Runtime の境界か」を明確にする
- [ ] `src/api/__init__.py` の公開 API セットを見直し、最小サブセットと推奨インポートスタイル（`from api import run, G, E, ...` など）を定義
- [ ] `src/api/shapes.py` / `src/api/effects.py` の `__getattr__` ベースの動的ディスパッチを再検討し、静的エクスポートまたは明示的なファクトリ API への簡素化案を検討

---

## フェーズ3: キャッシュ戦略の再設計・簡素化

ターゲット: `src/api/effects.py`, `src/api/shapes.py`, `src/common/param_utils.py`,
`src/engine/core/lazy_geometry.py`, 各 shape/effect 内の `@lru_cache`

- [ ] 現状のキャッシュポイントを全部列挙（Pipeline LRU / LazyGeometry / shape LRU / spec キャッシュなど）
- [ ] 「どの層でキャッシュするか」を 1 箇所に集約する方針を決める
- [ ] Pipeline のキャッシュキー（署名 + 設定）の仕様を整理し、重複実装を削減
- [ ] shape 生成の署名計算とキャッシュを `LazyGeometry` 側か `api.shapes` のどちらかに寄せる
- [ ] `PipelineBuilder.build()` と `Pipeline.__call__()` で重複している name 解決や `params_tuple`→`dict` 変換ロジックをどちらか一方に寄せる
- [ ] `@lru_cache` 依存を減らし、プロセス内 LRU を一元管理（サイズ・クリア API を統一）
- [ ] HUD 用メトリクス（ヒット数/ミス数）の収集をキャッシュ層から独立させる
- [ ] キャッシュ OFF / ON を環境変数や settings から制御するポリシーを明確化
- [ ] BaseRegistry / shapes.registry / effects.registry / LazyGeometry / runtime.cache / HUD 間のキャッシュ＆レジストリ層の責務境界を整理し、デバッグ時に追うべき層を一段に絞る

---

## フェーズ4: パラメータ量子化・設定ロジックの整理

ターゲット: `src/common/param_utils.py`, `src/common/settings.py`,
`src/api/shapes.py`, `src/api/effects.py`

- [ ] 量子化ロジック（`_env_quant_step`, `quantize_params`, `params_signature`）のフロー図を作成
- [ ] 「settings → 環境変数 → デフォルト」の優先順位と、使用箇所の一覧を確認
- [ ] 量子化対象（float / ベクトル）の範囲と、int / bool 非量子化ポリシーを仕様として明文化
- [ ] Shape / Effect それぞれで「署名だけ量子化」「実行値は生値」の違いを整理
- [ ] `__param_meta__` と GUI 仕様との整合を architecture.md に反映
- [ ] `_env_quant_step` の挙動を単純化（失敗時のフォールバックを減らす）案を検討
- [ ] パラメータ署名生成のユニットテストを強化（NaN / ベクトル / ndarray / dict ケース）

---

## フェーズ5: モジュール分割・読みやすさ向上

例: `src/api/sketch.py`, `src/shapes/sphere.py` などの巨大モジュール

- [ ] 1000 行級のモジュールを対象に「責務ごとのセクション」を洗い出し
- [ ] `api.sketch` を「ランナー本体」「エクスポート」「イベントハンドラ」「録画」などに分割
- [ ] `shapes/sphere.py` を内部ビルダー単位で整理し、コメントアウトされた古い subdiv clamp ロジックなどの残骸を削除（必要ならサブモジュール化）
- [ ] `common.param_utils` の責務を分割（0–1 正規化系 / 署名・量子化系）
- [ ] 「どこに何を追加するか」のガイドラインを architecture.md に反映
- [ ] docstring を AGENTS.md のルールに合わせてスタイル統一（終止形・過剰説明を抑制し、敬体表現を排除）
- [ ] `architecture.md` と各モジュール先頭の長大な docstring が同じ層の説明を重複している箇所を洗い出し、「architecture.md を真実のソース」として docstring 側を簡略化または参照リンクに縮小
- [ ] `util.utils._find_project_root()` の API と実装を整理し、`start` 引数の意味と返り値の挙動を一致させる（不正確な `cur.parent.parent` フォールバックを廃止）
- [ ] 「小さくきれいなコア + 周辺機能」という構造を意識し、HUD / MIDI / Export / 録画などの周辺機能を最小コアから明示的に分離する

---

## フェーズ6: テスト・検証フローの整備

- [ ] リファクタリング対象モジュールごとに「最小 smoke テスト」を追加または強化
- [ ] `run_sketch(..., init_only=True)` を使ったヘッドレス検証テストを追加
- [ ] キャッシュ関連の回帰テスト（ON/OFF・サイズ変更・クリア API）を作成
- [ ] 公開 API (`src/api/__init__.py`) に影響する変更に対してスタブ生成と同期テストを確認
- [ ] `architecture.md` と実装の差分を見直し、必要な箇所を更新
- [ ] `run_sketch` が触る外部環境（pyglet / ModernGL / MIDI / ファイルシステム / シグナル / ログ設定など）の境界をテスト観点から明文化し、「ライブラリ層の責務」と「エントリポイント側の責務」をドキュメント化

---

## 要事前確認事項（あなたに決めてほしい点）

以下は実装に入る前にポリシーを決めておきたい項目です。

- [ ] `run_sketch` でシグナル/atexit/sys.excepthook を扱う方針  
      - 候補: a) 現状維持, b) オプトイン（フラグで有効化）, c) 完全にエントリポイント側に移動
- [ ] API の互換性レベル  
      - 例: `api.run` / `api.run_sketch` のシグネチャをどこまで変更してよいか
- [ ] キャッシュのデフォルト挙動  
      - 例: shape / pipeline キャッシュを「明示有効化のみ」にするか、現状通り常に有効にするか
- [ ] グローバル状態（CC / パレット / runtime）の扱い  
      - 例: すべてコンテキストオブジェクト化するか、一部はグローバルのまま残すか
- [ ] モジュール分割の粒度  
      - 例: `api.sketch` を何ファイル程度に分けるか（2–3 個か、それ以上か）

---

## 今後の進め方（提案）

1. あなたに本計画ドラフトをレビューしてもらい、「やりすぎ」「足りない」「優先度」のフィードバックをもらう。
2. フェーズごとに小さめの PR/変更単位に区切って進める（まずはフェーズ1→2→3 を優先）。
3. 各フェーズ着手前に、このファイルのチェックリストを更新しつつ作業方針を再確認する。
4. 完了した項目から順に `[x]` にチェックを入れ、未完了部分を常に明示する。

---

### あなたへの質問

- この計画の粒度と範囲は、あなたのイメージする「大規模な根本リファクタリング」と合っていますか？
- 最初に着手してほしいフェーズ（1〜6）と、逆に「今はまだ触ってほしくない」領域があれば教えてください。
