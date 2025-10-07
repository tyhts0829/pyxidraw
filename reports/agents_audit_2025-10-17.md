# AGENTS.md 追記事項と簡素化候補（2025-10-17）

本ドキュメントは、現行の AGENTS.md に対し「追記すべき要点」と「記載不要/簡素化候補」を指摘するもの。根拠として関連実装のファイル参照を併記する。

## 追記すべき（必須）

- ランタイム環境変数（キャッシュ/挙動）
  - `PXD_PIPELINE_QUANT_STEP`（量子化既定 1e-6、環境で上書き）: src/common/param_utils.py:124
  - `PXD_PIPELINE_CACHE_MAXSIZE`（Pipeline 結果 LRU の既定上限）: src/api/effects.py:314
  - `PXD_COMPILED_CACHE_MAXSIZE`（compiled pipeline のグローバル再利用キャッシュ上限）: src/api/effects.py:88
  - `PXD_DISABLE_GEOMETRY_DIGEST`（Geometry の digest 計算を無効化）: src/engine/core/geometry.py:85
  - `PYXIDRAW_MIDI_STRICT`（MIDI 初期化の厳格モード切替）: src/api/sketch.py:195

- コンフィグ読み込みの優先順位とスコープ
  - 優先順は「configs/default.yaml → ルート config.yaml」でトップレベル浅い上書き（ディープマージ無し）であることを明記: src/util/utils.py:34

- Parameter GUI の保存/復元（永続化）仕様
  - 保存先の既定と上書きキー（`parameter_gui.state_dir`）: src/engine/ui/parameters/persistence.py:47
  - 保存対象（original と異なる current のみ）と量子化（RangeHint/VectorRangeHint の step 優先、未指定時は 1e-6）: src/engine/ui/parameters/persistence.py:15, 97
  - 復元は既知 Descriptor のみ適用（未知キー/型不一致は無視）: src/engine/ui/parameters/persistence.py:170

- Parameter GUI の表示対象/対応型の明確化
  - 「draw で未指定＝既定値採用」の引数のみ GUI に登録（provided は登録しない）: src/engine/ui/parameters/value_resolver.py:137, 184
  - 列挙は choices がある場合のみ GUI 対応（自由文字列は非対応）: src/engine/ui/parameters/value_resolver.py:236

- キャッシュの二層構造の明文化
  - 入力×定義で固定化された compiled pipeline の再利用キャッシュ（グローバル、上限は `PXD_COMPILED_CACHE_MAXSIZE`）: src/api/effects.py:236
  - 実行結果（Geometry）用の単層 LRU（インスタンスローカル、`.cache(maxsize=...)` と `PXD_PIPELINE_CACHE_MAXSIZE` で制御）: src/api/effects.py:350

- CC スナップショットの保存先
  - 既定は `data/cc` で、`io.cc_dir` で上書き可能である旨（運用ガイドとして明記）: configs/default.yaml:83

- スタブ生成の前提（軽量依存のダミー）
  - `tools/dummy_deps` により numba/fontTools/shapely が未導入でもスタブ生成/テストが動作すること（開発者向け注意）: tools/dummy_deps.py:1

## 記載不要/簡素化候補（提案）

- スタブ再生成の個別 Git 操作
  - Build セクションの「`git add src/api/__init__.pyi`」は運用手順として冗長。コマンド例は「生成コマンド」までに留め、VCS 操作は PR ルールに委ねるのが簡潔。

- 将来系メモの配置
  - 「モノレポ運用（将来拡張）」は方針メモとして価値はあるが、運用規約の必須事項からは外せる。`docs/` への移設や簡素化を検討。

- 既存の一般指示の重複
  - 「明確さ/シンプルさ優先」は有益だが、同趣旨の文言が複数あるため 1 箇所に要約すると読みやすい。

## 参考（要点の根拠）

- 量子化/署名: src/common/param_utils.py:124, src/common/param_utils.py:219
- GUI 表示条件/型サポート: src/engine/ui/parameters/value_resolver.py:137, src/engine/ui/parameters/value_resolver.py:184, src/engine/ui/parameters/value_resolver.py:236
- キャッシュ構造/上限: src/api/effects.py:88, src/api/effects.py:236, src/api/effects.py:314, src/api/effects.py:350
- Geometry ダイジェスト無効化: src/engine/core/geometry.py:85, src/engine/core/geometry.py:251
- MIDI 厳格モード: src/api/sketch.py:195, src/api/sketch.py:236
- コンフィグ優先順: src/util/utils.py:34
- CC 保存先設定: configs/default.yaml:83
- ダミー依存注入: tools/dummy_deps.py:1

以上。必要であれば AGENTS.md の該当セクション（Runtime/Env/Cache/GUI）へ追記パッチも作成可能。

