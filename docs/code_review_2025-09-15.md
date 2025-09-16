# PyXidraw6 コードレビュー（2025-09-15）

本ドキュメントは、リポジトリ全体（src/, tests/, docs/ ほか）を静的に読んだ結果のレビューです。実行や依存追加は行っていません（ネットワーク・依存導入なし）。Python 3.10 前提。

---

## サマリ

- 設計方針（Geometry の統一表現、純関数エフェクト、パイプラインの単層 LRU、API スタブ自動生成）が明確で、テストでもレイヤ規約・スタブ同期・スモークを押さえており、基盤は総じて良好。
- 一方で「効果（effects）の自動登録タイミング」「Renderer の描画モードとシェーダの不整合」「optional 依存の読み込み方の揺れ」「ドキュメント/メタデータの表記ずれ」が運用上の主なリスク。

---

## 強み（Good）

- 統一 Geometry 型と純関数主義：`Geometry` の不変条件・`as_arrays(copy=False)` の読み取りビュー・`digest` の設計が明快。
- パイプライン：`Pipeline` のキー設計（関数バイトコード近似 + 正規化 param）と LRU 風キャッシュ、`strict` と `validate_spec` による仕様検証が堅牢。
- レイヤ規約の自動検証：`tests/test_architecture.py` によるレイヤ/禁止エッジ/循環検出が良い安全網。
- スタブ自動生成：`tools/gen_g_stubs.py` と同期テストで公開 API の見える化と崩れ検出。
- 実行系の責務分離：`WorkerPool`/`StreamReceiver`/`SwapBuffer`/`LineRenderer`/`OverlayHUD` の分割が適切。`api.sketch` は遅延 import とフォールバックで軽量環境に配慮。

---

## 重要指摘（優先度 高→低）

1) Effects の自動登録タイミングの穴（実行時に未登録化し得る）
- 事象：`api.effects` は `effects` パッケージを import せず、`effects.registry` 参照のみ。したがって、どこかで `import effects`（副作用で各エフェクト登録）を行わない限り、`E.pipeline.rotate(...)` 等で `get_effect` が `KeyError` になり得る。
- 影響：実行パス（例: `main.py`）では `effects` が未 import のまま起動し得る。一方、テスト/スタブ生成は副作用 import を行うため“たまたま”緑になる可能性。
- 方針案：
  - A. `api.effects` のモジュール import 時に軽量シム導入（`tools.dummy_deps.install()`）→ `import effects` を実行して登録を確実化。
  - B. あるいは `effects.registry.get_effect` で未登録名アクセス時に一度だけ遅延 import を試みる（副作用 import）。

2) Renderer の描画モードと GS 入力の不整合
- 事象：`renderer.draw()` は `mgl.LINE_STRIP` + primitive restart。`shader.py` の Geometry Shader は `layout(lines) in`。GS 入力 primitive と描画モードは一致が必要で、現状の組み合わせは不整合。
- 選択肢：
  - A. 描画モードを `GL_LINES` に揃え、`_geometry_to_vertices_indices` を「各セグメント 2頂点インデックス」へ変更（primitive restart 不要）。
  - B. GS を撤廃し `LINE_STRIP` のまま描画（太線品質は環境依存）。
  - C. CPU 側でストロークの押し出し（帯メッシュ化）→ `TRIANGLE_STRIP` 描画（最良品質だがコスト高）。
- 併せて `architecture.md` の「1ドロー + primitive restart」記述を実装に同期すべき。

3) optional 依存（numba/shapely など）の import 戦略が混在
- 事象：`effects/displace.py`, `effects/affine.py`, `shapes/text.py` などで `numba`、`effects/offset.py` で `shapely` をトップレベル import。軽量環境や `main.py` 実行で ImportError を誘発し得る。一方、`api.sketch` は遅延 import とフォールバックを備えるなど方針が揺れている。
- 方針案：
  - 「遅延 import + フォールバック」を全 optional 依存に統一（関数内 import／`tools.dummy_deps` 併用／ユーザ向け警告）。
  - README に optional 依存の導入と失敗時の挙動を明記。

4) ドキュメント/メタデータの表記ゆれ
- 事象：`pyproject.toml` の name は `pyxidraw5`、README タイトルは `PyXidraw6`。README のリンクに存在しないドキュメント（例: `docs/effects.md`, `docs/dev-setup.md`）。
- 対応：名称の統一（v6 に合わせるか要決定）、リンク修正、描画方式に関する `architecture.md` の同期。

---

## 詳細レビュー

### アーキテクチャ/設計
- レイヤ規約：`effects/shapes → engine.{render,pipeline,ui,io,monitor}` 禁止や `engine → api` 禁止などの制約がテストで検証され、構造が保たれている。
- `Geometry`：不変条件・`digest` 設計・空集合/単一頂点の扱いが明瞭。`concat` もオフセットシフトの規則が適切。
- パイプライン：`strict` と `validate_spec` の二段構えでランタイム/ファイル入力の双方をカバー。
- 登録（registry）：キー正規化や API（`@shape`/`@effect`）は対称で良いが、effects 側の“初回 import タイミング”の一貫化が必要。

### 機能/正確性
- `api.sketch`：MIDI 厳格/非厳格や `init_only` 分岐でヘッドレス環境にも配慮。GL 初期化・投影行列・HUD 連携は明確。
- `effects/offset`：3D→XY 射影→Shapely→復元の流れ、曲線クローズ・縮尺補正・結合が実装されている。
- `effects/displace`/`affine`：Numba 前提最適化。フォールバック戦略の統一が望ましい。

### パフォーマンス
- `LineMesh`：バッファ再確保・`orphan` 書き込み・VAO 再構築の順序は適切。`primitive_restart_index` の設定も妥当。
- `WorkerPool`：プロセス間転送で `Geometry` を渡す設計は負荷次第で律速になり得るが、現状規模では合理的。
- パイプライン LRU：`maxsize=None/0/N` と環境変数上書きがあり運用性が高い。

### テスト/CI/型/スタイル
- 重要な安全網が揃っている（レイヤ規約、スタブ同期、スナップショット、スモーク、MIDI フォールバック）。
- mypy 対象が狭い（`util/utils.py` のみ）。段階的拡大の余地あり。
- ruff の `select` が最小（F401/F841/E9）。軽く広げると品質は上がるが、ノイズ増は要調整。

### ドキュメント/配布
- README のリンクや名称を現行実装に合わせて同期推奨（特に描画方式の記述）。
- `architecture.md` の「1ドロー + primitive restart」記述は現行 GS の厚み付け設計と整合を取る必要あり。

---

## 推奨アクション（要方針確認）

- Effects 登録の確実化
  - `api.effects` で `tools.dummy_deps.install()` 実行 → `import effects`（副作用登録）。
  - 代替/補助として `effects.registry.get_effect` に遅延 import のリトライを実装。
- Renderer の方式を決めて実装/Doc 同期
  - A: `GL_LINES` に統一（簡潔・予測可能）。B: GS 撤廃で `LINE_STRIP`。C: 三角形ストローク化。
- optional 依存の import ポリシー統一
  - トップレベル import を避け、実行点で遅延 import + 明示メッセージ。`dummy_deps` をツール/開発で活用。
- ドキュメント/メタデータ統一
  - パッケージ名（v6 へ統一）・README リンク修正・`architecture.md` 同期。
- 型/静的解析の段階拡大（任意）
  - `engine/core` と `api` 主要モジュールから段階的に mypy 対象拡大。ruff ルールの調整。

---

## 補足（確認対象の主要ファイル例）
- 設定/エントリ：`pyproject.toml`, `main.py`, `pytest.ini`, `README.md`
- API：`src/api/__init__.py`, `src/api/effects.py`, `src/api/shapes.py`, `src/api/sketch.py`
- コア：`src/engine/core/geometry.py`, `frame_clock.py`, `render_window.py`
- レンダ：`src/engine/render/{renderer.py, line_mesh.py, shader.py}`
- パイプライン：`src/engine/pipeline/{worker.py, receiver.py, buffer.py}`
- IO/MIDI：`src/engine/io/{manager.py, controller.py, service.py}`
- レジストリ：`src/common/base_registry.py`, `src/{effects,shapes}/registry.py`
- エフェクト/シェイプ：`src/effects/*`, `src/shapes/*`
- ユーティリティ：`src/util/{constants.py, utils.py, geom3d_ops.py}`
- スタブ生成：`tools/gen_g_stubs.py`
- 代表テスト：`tests/{test_architecture.py, stubs/*, api/*, effects/*, render/*}`

---

以上です。方針が決まり次第、改善ブランチ用のチェックリストを別途起こせます（本レビューは報告のみ、コード変更は含みません）。
