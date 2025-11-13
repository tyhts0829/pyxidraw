どこで: src/api/effects.py のグローバル初期化
何を: _GLOBAL_PIPELINES / _GLOBAL_COMPILED（および関連 maxsize）の初期化を一箇所へ集約
なぜ: 可読性と初期化順序の明確化、重複排除による保守性向上

# 背景 / 現状

- 重複している初期化が存在（WeakSet/OrderedDict を複数回生成）。
  - 参照箇所（抜粋）:
    - 前方宣言と関連 env 読み: src/api/effects.py:26, src/api/effects.py:30
    - 参照（get/put/Evict）: src/api/effects.py:186, src/api/effects.py:207
    - 重複初期化ブロック（再代入）: src/api/effects.py:265, src/api/effects.py:269
- 影響: 機能は保たれるが、読み手が初期化順序を追いづらい。型/前方参照の扱いも曖昧。

# 目標（Goal）

- グローバル初期化（WeakSet/OrderedDict/Maxsize）をモジュール末尾の単一ブロックに集約。
- 既存の try/except による安全化は維持（None ガード等の仕様変更は行わない）。
- 挙動（キャッシュ/LRU/統計）は変更しない。

# 非目標（Non‑Goals）

- キャッシュ仕様の変更（エビクション戦略、キー設計、署名計算）。
- 設定ソースの大規模整理（settings への全面移行）。今回は env 読みの集約に留める。

# 変更方針（Design）

- ファイル先頭:
  - 前方宣言のみ（型注釈）を残し、実体初期化は行わない。

- モジュール末尾（__all__ 直前または直後）に単一の初期化ブロックを配置:
  - settings（優先）→環境変数→既定の順で `_GLOBAL_COMPILED_MAXSIZE` を決定。
  - `_GLOBAL_PIPELINES = weakref.WeakSet()` / `_GLOBAL_COMPILED = OrderedDict()` を生成。
  - 例外時はフォールバック値を設定（現行ロジック踏襲）。

- 参照箇所:
  - 既存の try/except によるガードを維持（挙動変更を避ける）。

- 重複の初期化ブロックを削除（トップレベルの env 読みを末尾へ移設）。

# 作業手順（チェックリスト）

- [x] 現状の初期化箇所を特定し、末尾の単一ブロックへ集約（編集済み）
- [x] モジュール末尾へ単一の初期化ブロックを新設（settings→env→既定で maxsize 決定）
- [x] トップレベルの env 読みを削除し、重複初期化を解消
- [x] `PipelineBuilder.build()` の get/put 参照は現状維持（try/except 継続）
- [x] コメント: 初期化ブロックに目的とフォールバック方針を明記
- [ ] Lint/Format/Type（単一ファイル）: `ruff check --fix src/api/effects.py` / `black src/api/effects.py` / `isort src/api/effects.py` / `mypy src/api/effects.py`
- [ ] 影響低テスト（任意）: Pipeline ビルド/適用のスモーク（既存サンプルを用いて手動 or pytest 対象限定）

# 検証（Acceptance Criteria）

- [x] src/api/effects.py における `_GLOBAL_PIPELINES` / `_GLOBAL_COMPILED` の生成が 1 箇所のみであること（集約済み）
- [ ] mypy が型に関してエラーを出さないこと（前方宣言＋末尾初期化）
- [ ] 既存のパイプライン構築・キャッシュ挙動が変化しないこと（HIT/MISS カウンタに異常なし）

# 影響範囲・リスク

- 低: 初期化順序/重複の解消のみ。import 時の順序依存による例外リスクは低い。
- 緩和策: 既存の try/except ガードを維持。ファイル単体の mypy/ruff を実行。

# 事前確認事項（Open Questions）

- [ ] `_GLOBAL_COMPILED_MAXSIZE` の取得はこのまま settings→env→既定の順で問題ないですか？
- [ ] 初期化ブロックの配置は `__all__` 定義直後で問題ないですか？（モジュール import 時に確実に走る位置）

# ロールバック

- 単一ファイルの局所変更につき、問題発生時は差分を元に戻すだけで復旧可能。
