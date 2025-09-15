# 形状を「関数ベース」に全面統一する計画（破壊的変更）

日付: 2025-09-15
提案者: チーム
対象範囲: `shapes/*`, `src/shapes/registry.py`, `src/api/shape_factory.py`, `src/scripts/gen_g_stubs.py`, `src/api/__init__.py` 他

## 背景
- 現状の Shape は `BaseShape` 抽象と `@shape`（クラス登録）を前提としている。
- ユーザー拡張の直感性を高めるため、Effect と同様に「関数を登録して使う」モデルへ統一したい。
- 先行検討（docs/proposals/shape_decorator_import.md）では「関数も受理（ラッパ自動生成）」案と「全面関数化（Breaking）」案を比較。ここでは後者（よりシンプル）を具体化する。

## 目的（Goals）
- Shape を関数ベースに全面統一し、Effect と完全対称の API/実装にする。
- API 表面と実装/スタブ生成の複雑さを削減（保守性向上・学習コスト低減）。

## 非目的（Non-goals）
- 描画エンジン/パイプラインの仕様変更。
- 既存エフェクト API の変更。

## 結論（方針）
- 公開 API は「from api import shape」で関数登録のみを受け付ける。
- `BaseShape` は公開 API から撤廃し、内部にも保持しない（完全撤廃）。
- レジストリ（`shapes.registry`）は「関数のみ」受理し、`get_shape(name)` は関数を返す。
- `ShapeFactory (G)` はクラス解決をやめ、関数を直接呼び出して `Geometry` を受け取る。
- 返り値が `Geometry` 以外の場合は `Geometry.from_lines(...)` にフォールバック（現状踏襲）。
- 型スタブ生成（`gen_g_stubs`）は Shape 関数のシグネチャから直接生成する。

## 影響範囲
- API: `BaseShape` の削除、`api.shape` は関数のみ受理（ユーザーコードのクラス実装は要移行）。
- 実装: `shapes/*` の各 Shape を関数へ移行。
- スタブ/テスト/ドキュメント: 参照・説明・同期テストの更新が必要。

## 実装設計（要点）
- `src/shapes/registry.py`
  - `shape()` デコレータを「関数のみ」に変更。型: `Callable[..., Geometry | list[np.ndarray]]`。
  - 既存の `BaseShape` チェック削除、関数チェック（`inspect.isfunction`）へ変更。
  - `get_shape(name) -> Callable[..., Geometry]`。`list_shapes()`/`is_shape_registered()` は既存維持。
- `src/api/shape_factory.py`
  - `get_shape_generator(name)` の戻りを「関数」に変更（名称は `get_shape_fn` にリネーム可）。
  - `shape_cls = get_shape_generator(name)` → `fn = get_shape_fn(name)` に置換。
  - `instance.generate(**params)` は廃止し `fn(**params)` を直接呼ぶ。
  - 返り値が `Geometry` でなければ `Geometry.from_lines(...)` にフォールバック。
- `src/scripts/gen_g_stubs.py`
  - 署名抽出を「クラスの generate」ではなく「関数本体」から行う単純化へ変更。
  - `from api.shape_registry import get_shape_generator` 依存を廃止し、`shapes.registry.get_shape` か `api` 側の薄い関数へ移行。
  - 既存の Vec3/param meta などはそのまま適用（Effect 側と揃える）。
- `src/api/__init__.py`
  - `BaseShape` 関連の再エクスポートと説明を削除（既にユーザー文書では案内していないが明示）。
  - `__api_version__` を +1（例: 6.0）に上げ、`__breaking_changes__` に追記。
- `src/shapes/base.py`
  - 削除（または `docs/attic/` へ移動）。必要なら ADR にて撤廃理由を明記。
- `src/shapes/__init__.py`
  - クラスの再エクスポートを削除し、各モジュールの関数定義をインポートして登録副作用のみ残す（または static import を最小化）。

## マイグレーション（内製 Shape 移行）
- 代表例（Before → After）
  - Polygon:
    - Before: `@shape class Polygon(BaseShape): def generate(self, n_sides: int=3) -> Geometry: ...`
    - After:  `@shape def polygon(*, n_sides: int=3) -> Geometry: ...`
  - Sphere/Torus/Text なども同様に `generate` 本体を関数へ移す。必要ならモジュール内キャッシュや `functools.lru_cache` を使用。

## ロールバック戦略
- 移行中に問題が発覚した場合、短期的に「関数受理 + クラス受理（自動ラッパ）」の併存に切り戻せる実装に留める（Feature flag 化）。

---

## 実装アクションリスト（詳細）

### 0. 事前
- [ ] PR タイトル: `feat(shape): function-only API (breaking)`
- [ ] バージョニング方針の合意（`__api_version__ = '6.0'`）
- [ ] 影響ドメイン棚卸し（docs/tests/examples/stubs/CI）

### 1. レジストリの破壊的更新（関数のみ）
- [x] `src/shapes/registry.py`: `BaseShape` 型チェック削除、`inspect.isfunction` に変更
- [x] `get_shape()` の戻り型を `Callable[..., Geometry]` に変更（docstring/型注釈）
- [x] `__all__` と docstring 更新（「関数のみ」明記）
- [x] ユニットテスト追加: 関数登録/取得/エラー系（非関数）

### 2. ShapeFactory の簡素化
- [x] `src/api/shape_factory.py`: クラス解決→関数呼び出しへ全面置換
- [ ] `get_shape_generator` を `get_shape_fn` に改名（または既存名を関数返しに切替）
- [x] `G._cached_shape` 内の `instance.generate(**params)` を `fn(**params)` に変更
- [x] 返り値の `Geometry` 化フォールバックは現状踏襲
- [x] テスト更新: 既存の `get_shape_generator` 参照を置換（関数を返す想定に更新）

### 3. スタブ生成（gen_g_stubs）の単純化
- [x] 形状側の署名抽出を「関数」起点に変更
- [x] `install_dummy_deps()` の呼び出し位置を維持しつつ、重依存回避の安全性を検証
- [x] `api/__init__.pyi` の `G: _GShapes` 生成ロジックを関数シグネチャで再生成
- [x] スタブ同期テスト（`tests/stubs/*`）を更新して緑化

### 4. 内製 Shape 実装の移行
- [ ] `src/shapes/polygon.py`: 関数化
- [x] `src/shapes/sphere.py`: 関数化
- [ ] `src/shapes/torus.py`: 関数化
- [ ] `src/shapes/cylinder.py`: 関数化
- [ ] `src/shapes/cone.py`: 関数化
- [ ] `src/shapes/grid.py`: 関数化
- [ ] `src/shapes/lissajous.py`: 関数化
- [ ] `src/shapes/text.py`: 関数化（fontTools 依存の軽量化/キャッシュ見直し）
- [ ] `src/shapes/asemic_glyph.py`: 関数化（大きめ、段階移行）
- [ ] `src/shapes/polyhedron.py`: 関数化
- [ ] `src/shapes/capsule.py`: 関数化
- [ ] `src/shapes/attractor.py`: 関数化
- [ ] それぞれの docstring/型注釈更新（NumPy スタイル）

### 5. API/ドキュメント更新
- [ ] `src/api/__init__.py`: `__api_version__` を 6.0 に、`__breaking_changes__` に追記
- [ ] README/ガイド: 「継承不要、関数を @shape で登録」へ書き換え
- [x] `docs/user_extensions.md`: クラス実装例を関数実装例に差し替え
- [ ] `architecture.md`: BaseShape 撤廃・関数ベース統一を明記
- [ ] 破壊的変更の CHANGELOG 追加

### 6. 不要コードの撤廃/整理
- [ ] `src/shapes/base.py` の削除（または `docs/attic/` へ移動）
- [x] `src/shapes/__init__.py` の再エクスポート見直し（関数登録の副作用のみ残す）
- [ ] `src/api/shape_registry.py` の存否方針（完全撤廃 or 薄い互換レイヤで ImportError を案内）

### 7. テストの緑化（編集ファイル優先ループ）
- [x] 変更ファイルに限定して `ruff/black/isort/mypy` を実行
- [x] 影響テスト（`tests/shapes/*`, `tests/stubs/*`, `tests/api/*`）を段階的に緑化
- [x] `pytest -q -m smoke` → 影響箇所の個別テスト → stubs 同期テストの順で確認

### 8. CI/品質ゲート
- [ ] `.github/workflows/*` のジョブに変更が必要か確認（スタブ同期は必須）
- [ ] 生成スタブ差分がゼロであることを CI で検証

### 9. リリース準備
- [ ] 影響と移行手順をリリースノートに明記
- [ ] サンプル（`user_shape_demo.py`）の関数版へ更新確認

---

## 受け入れ基準（DoD）
- [ ] すべての Shape が関数実装である（クラス実装ゼロ）
- [x] `from api import shape` で関数登録が可能、`G.<name>(...)` が機能
- [ ] `BaseShape` に依存する公開 API/ドキュメントが残っていない
- [x] スタブ同期テスト（`tests/stubs/*`）および変更ファイルに対する `ruff/black/isort/mypy/pytest` が緑
- [ ] `__api_version__ = 6.0`、`__breaking_changes__` 追記済み

---

## ビルトイン Shape 関数化チェックリスト（進捗管理）
- [x] sphere（src/shapes/sphere.py）
- [x] polygon（src/shapes/polygon.py）
- [x] torus（src/shapes/torus.py）
- [x] cylinder（src/shapes/cylinder.py）
- [x] cone（src/shapes/cone.py）
- [x] grid（src/shapes/grid.py）
- [x] lissajous（src/shapes/lissajous.py）
- [x] text（src/shapes/text.py）
- [x] asemic_glyph（src/shapes/asemic_glyph.py）
- [x] polyhedron（src/shapes/polyhedron.py）
- [x] capsule（src/shapes/capsule.py）
- [x] attractor（src/shapes/attractor.py）

## リスクと対応
- Text など重量級 Shape の関数化で初期化コストが顕在化する恐れ → モジュールレベルキャッシュ/遅延ロードで緩和
- 外部ユーザーのクラス実装が壊れる → 移行ガイドとサンプル、暫定の「クラス→関数変換ガイド」を提供
- スタブ生成の一時不一致 → 先に gen_g_stubs 更新 → built-in shapes を関数化 → 再生成 → stubs テスト緑の順で進める

## スケジュール（目安）
- フェーズ1（レジストリ/Factory/スタブ生成の骨格）：0.5–1.0 日
- フェーズ2（内製 Shape の関数化 + テスト緑化）：1.5–2.0 日（Text/大物は別 PR 可）
- フェーズ3（ドキュメント/リリースノート/最終確認）：0.5 日

## 承認事項（Ask-first）
- [ ] BaseShape の完全撤廃（削除/attic 送り）
- [ ] `api/shape_registry.py` の撤廃（該当参照は既に最小）
- [ ] 破壊的変更として 6.0 をリリース（README/CHANGELOG 更新）

---

この計画で問題なければ、「フェーズ1」から着手します。必要に応じて PR を小さく分割（レジストリ→Factory→スタブ→各 Shape）し、段階的にレビュー可能な形で進めます。
