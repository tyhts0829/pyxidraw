# PyxiDraw6 src 配下 厳しめコードレビュー（2025-09-07）

このドキュメントは、`src/` 配下の実装を対象に、設計・型・可読性・可搬性・性能・安全性の観点で厳しめにレビューした結果をまとめたものです。主な対象は API 層（`api/`）、エンジン層（`engine/`）、エフェクト（`effects/`）、形状（`shapes/`）、共通基盤（`common/`、`util/`）です。

- 評価日時: 2025-09-07（ローカル環境で静的レビュー）
- Python: 3.10 前提（型ヒントあり）
- 重要度凡例: [HIGH] 重大バグ/破壊的、[MID] 実害あり、[LOW] 望ましい改善

## サマリー

- 設計方針（Geometry 統一、G/E の責務分離、Pipeline の strict 検証、単層キャッシュ）に一貫性があり、実装品質は総じて高い。
- 並行処理（Worker/Receiver/SwapBuffer）の責務分離は明確で、例外伝播・停止処理も概ね妥当。
- 主要 API スタブ（`api/__init__.pyi`）は充実しており、型とドキュメントの同期も概ね取れている。
- 一方で、実害が出うる不具合/互換性懸念が複数点存在（下記 [HIGH]/[MID]）。

## 重大・優先度高の指摘（要対応）

1) [HIGH] `RenderPacket.timestamp` がインポート時固定
- 場所: `src/engine/pipeline/packet.py`
- 問題: `timestamp: float = time.time()` は「定義時に一度だけ評価」され、全インスタンスが同一時刻になる。
- 影響: 時刻ベースの測定・デバッグが破綻。
- 対応案: `default_factory=time.time` を用いる。
  - 修正例:
    ```python
    from dataclasses import dataclass, field
    @dataclass(slots=True, frozen=True)
    class RenderPacket:
        geometry: Geometry
        frame_id: int
        timestamp: float = field(default_factory=time.time)
    ```

2) [HIGH] CC 値の型不一致（int/float）
- 場所: 
  - `src/engine/pipeline/task.py` → `cc_state: Mapping[int, int]`
  - `src/api/runner.py` → `user_draw: Callable[[float, Mapping[int, int]], Geometry]`
  - `src/engine/io/service.py` → `snapshot() -> Mapping[int, float]`
- 問題: 実際の CC は `0..1` 正規化 float。型が int で固定されており不整合。
- 影響: 型チェック破綻/利用側の誤解・バグ誘発。
- 対応案: すべて `Mapping[int, float]` へ揃える（`task.py` と `runner.py` のシグネチャ修正）。

3) [MID] Shapely 2.x 互換性リスク（`effects/offset.py`）
- 場所: `LineString.buffer(distance, join_style=..., resolution=...)` に文字列を渡し `# type: ignore`。
- 問題: Shapely 1.x/2.x で引数仕様が異なる。`join_style` には列挙/定数を使うのが安全。
- 影響: 環境により実行時エラー。
- 対応案: 版数検出し分岐、もしくは `shapely.geometry.JOIN_STYLE` 等の定数を利用。

4) [MID] フォント探索が macOS 前提（`shapes/text.py`）
- 場所: `TextRenderer.FONT_DIRS` が macOS の固定パス。
- 問題: Linux/Windows でフォント未検出 → デフォルト Helvetica も macOS 固有。
- 影響: 他 OS で文字描画不可/例外。
- 対応案: OS 判定で一般的パス（Linux: `/usr/share/fonts`, Windows: `%WINDIR%/Fonts`）を追加し、最終手段として DejaVu 等へフォールバック。

5) [MID] GPU 初期確保が過大（`engine/render/line_mesh.py`）
- 場所: `initial_reserve=200 * 1024 * 1024` を VBO/IBO 両方で確保。
- 問題: 起動直後から 400MB 近い GPU メモリ確保の可能性。
- 影響: メモリ圧迫/古い GPU で失敗。
- 対応案: 初期 4–16MB 程度 + 自動拡張へ（`_ensure_capacity` は既にある）。

6) [MID] 依存データが `src/` 直下に混在
- 場所: `src/engine/io/cc/*.pkl`, `*.json`
- 問題: ガイドラインでは大きな生成物は `data/` 推奨。
- 影響: パッケージ肥大、配布/ビルド汚染。
- 対応案: `data/` へ移動し参照パスを相対/設定化。

## 設計/アーキテクチャ

- Geometry 統一と G/E 責務分離は明瞭。`api/pipeline` の strict 検証・`validate_spec` の厳格化も一貫。
- `Pipeline` のキャッシュ鍵（`geometry_digest × pipeline_key`）は理に適う。`digest` 無効時のフォールバックも用意されている点が良い。
- 並行処理：`WorkerPool`（mp.Process）→`StreamReceiver`→`SwapBuffer` の流れは責務分離が適切。`WorkerTaskError` はシリアライズ考慮済みで◎。
- 終了処理：`WorkerPool.close()` は冪等/安全化されており、実運用での吊り対策ができている。

改善提案
- [LOW] `effects/translate.py` の「no-op 時に常にコピー」は方針（純関数/新インスタンス返却）として理解できるが、大規模データでメモリ負荷増。`Geometry` に軽量な `clone()`（配列共有・digest 再計算）を導入してもよい。
- [LOW] `engine/pipeline/receiver.py` の `max_packets_per_tick=2` はシーンによっては描画遅延を生む。設定化/動的調整を検討。

## 型・スタブ・一貫性

- API スタブ（`api/__init__.pyi`）は充実。`__param_meta__` による引数ドキュメント注入も良い。
- 指摘: `effects/registry.py` の `EffectFn = Callable[[Geometry], Geometry]` は実シグネチャ（`g, **kwargs`）と不一致。型だけ見ると誤解を生むため、`Callable[..., Geometry]` に揃えるか型別名を削除推奨。[LOW]
- `api/runner.py` / `engine/pipeline/task.py` の CC 型不一致は上記 [HIGH] 参照。

## 可読性・ドキュメンテーション

- ガイドライン「コメント/Docstring は日本語」に対し、英語 Docstring が散見（例: `shapes/__init__.py`, `common/logging.py`, `shapes/Lissajous` など）。統一を推奨。[LOW]
- 長大な説明コメントは有用だが、行長 100 を超える箇所が散見。`ruff/black` で整形基準を維持したい。[LOW]

## 互換性・環境依存

- OpenGL: ジオメトリシェーダ（GL 4.1）に依存（`engine/render/shader.py`）。一部環境で非対応/パフォーマンス懸念あり。将来的な太線描画の代替（スクリーンスペース拡張/テッセレーション非依存）を検討可。[LOW]
- Numba/FontTools/Shapely 非導入環境を dummy で回避する設計は CI 互換性に寄与しており良い。

## 性能

- `api/shape_factory` のパラメータ正規化（NumPy 配列指紋化）と LRU は堅実。
- `effects/*` は多くがベクトル化/Numba 対応で妥当。`fill` はスキャンラインの一括処理が入っており良い。
- `engine/render/_geometry_to_vertices_indices` はプリミティブリスタートを利用。末尾の余計なリスタートは実害小だが、削ってもよい。[LOW]

## セキュリティ/堅牢性

- MIDI 初期化の strict/非 strict 分岐、ログ出力、フォールバック（Null 実装）は良い。
- `util/utils.load_config()` はフェイルソフトで安全。

## 推奨修正パッチ（抜粋）

- `engine/pipeline/packet.py`
    ```diff
    -from dataclasses import dataclass
    +from dataclasses import dataclass, field
     @dataclass(slots=True, frozen=True)
     class RenderPacket:
         geometry: Geometry
         frame_id: int
    -    timestamp: float = time.time()
    +    timestamp: float = field(default_factory=time.time)
    ```

- `engine/pipeline/task.py` / `api/runner.py`
    ```diff
    -cc_state: Mapping[int, int]
    +cc_state: Mapping[int, float]
    ```
    ```diff
    -user_draw: Callable[[float, Mapping[int, int]], Geometry]
    +user_draw: Callable[[float, Mapping[int, float]], Geometry]
    ```

- `effects/registry.py`
    ```diff
    -EffectFn = Callable[[Geometry], Geometry]
    +EffectFn = Callable[..., Geometry]
    ```

- `engine/render/line_mesh.py`
    ```diff
    -    initial_reserve: int = 200 * 1024 * 1024,
    +    initial_reserve: int = 8 * 1024 * 1024,  # 実運用に合わせ小さく開始
    ```

- `effects/offset.py`（例: Shapely 2 対策・疑似コード）
    ```python
    try:
        from shapely.geometry import JOIN_STYLE
        js = {
            "mitre": JOIN_STYLE.mitre,
            "round": JOIN_STYLE.round,
            "bevel": JOIN_STYLE.bevel,
        }[join_style_str]
        buffered_line = line.buffer(actual_distance, join_style=js, resolution=resolution_int)
    except Exception:
        buffered_line = line.buffer(actual_distance, join_style=join_style_str, resolution=resolution_int)  # fallback
    ```

- `shapes/text.py`（OS ごとのフォント探索追加・疑似コード）
    ```python
    if sys.platform.startswith("linux"):
        FONT_DIRS += [Path("/usr/share/fonts"), Path.home()/".local/share/fonts"]
    elif sys.platform.startswith("win"):
        FONT_DIRS += [Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts"]
    ```

## 実施チェックリスト（今回）

- [x] `src/` 配下の主要モジュール構成を把握
- [x] API/Engine/Pipeline/Effects/Shapes の主要実装を精読
- [x] 重大不具合/型不整合/互換性リスクの抽出
- [x] 改善提案と最小修正案の提示
- [x] レビュー結果を Markdown として保存

## 次アクション提案

1. 上記 [HIGH]/[MID] の修正を小さな PR に分割して適用。
2. 変更後にガイドラインのチェック手順を実施：
   - `ruff check . && black . && isort .`
   - `mypy .`
   - `pytest -q`（存在する場合）
   - スタブ更新: `PYTHONPATH=src python -m scripts.gen_g_stubs && git add src/api/__init__.pyi`
3. Shapely の対応バージョンを `pyproject.toml` の `dependencies` に明記（1.x/2.x いずれか、もしくは両対応の分岐）。
4. `engine/io/cc/` の生成物を `data/` へ移動（参照コードの修正含む）。
5. 英語 Docstring の日本語化（方針に合わせて段階的に）。

---
以上です。必要であれば、上記修正のパッチ適用と最小再テストまで対応します。
