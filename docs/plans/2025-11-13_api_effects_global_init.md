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
- 前方参照は Optional 前提にし、利用箇所は None ガード or 既存の try/except で安全動作。
- 挙動（キャッシュ/LRU/統計）は変更しない。

# 非目標（Non‑Goals）

- キャッシュ仕様の変更（エビクション戦略、キー設計、署名計算）。
- 設定ソースの大規模整理（settings への全面移行）。今回は env 読みの集約に留める。

# 変更方針（Design）

- ファイル先頭:
  - 型注釈を Optional で定義して前方参照を明示。
    - `_GLOBAL_PIPELINES: weakref.WeakSet[Pipeline] | None = None`
    - `_GLOBAL_COMPILED: OrderedDict[tuple, Pipeline] | None = None`
  - `_GLOBAL_COMPILED_MAXSIZE: int | None` も値は未確定のまま宣言。

- モジュール末尾（__all__ 直前または直後）に単一の初期化ブロックを配置:
  - env（または settings）から maxsize を読み取り `_GLOBAL_COMPILED_MAXSIZE` へ設定。
  - `_GLOBAL_PIPELINES = weakref.WeakSet()` / `_GLOBAL_COMPILED = OrderedDict()` を生成。
  - 例外時はフォールバック値を設定（現行ロジック踏襲）。

- 参照箇所は None ガードを追加 or 既存の try/except 維持:
  - `Pipeline.__post_init__` は `if _GLOBAL_PIPELINES is not None: _GLOBAL_PIPELINES.add(self)` へ簡素化可（任意）。
  - `PipelineBuilder.build()` の get/put は現行の try/except 守りのままでも可。

- 重複の初期化ブロックを削除（再代入を全廃）。

# 作業手順（チェックリスト）

- [ ] 現状の初期化箇所を特定しコメントでマーキング（編集時の見落とし防止）
- [ ] ファイル先頭へ Optional 型の前方宣言を追加
- [ ] モジュール末尾へ単一の初期化ブロックを新設（env 読み + WeakSet/OrderedDict 生成）
- [ ] 既存の重複初期化（src/api/effects.py:265, src/api/effects.py:269 付近）を削除
- [ ] `Pipeline.__post_init__` の try/except を `if _GLOBAL_PIPELINES is not None` ガードへ置換（任意）
- [ ] `PipelineBuilder.build()` の get/put 参照は現状維持（初期化順序の明確化により None にはならない）
- [ ] コメント: 初期化ブロックに目的とフォールバック方針を明記
- [ ] Lint/Format/Type（単一ファイル）: `ruff check --fix src/api/effects.py` / `black src/api/effects.py` / `isort src/api/effects.py` / `mypy src/api/effects.py`
- [ ] 影響低テスト（任意）: Pipeline ビルド/適用のスモーク（既存サンプルを用いて手動 or pytest 対象限定）

# 検証（Acceptance Criteria）

- [ ] src/api/effects.py における `_GLOBAL_PIPELINES` / `_GLOBAL_COMPILED` の生成が 1 箇所のみであること（`rg` で確認）
- [ ] mypy が前方参照/Optional に関してエラーを出さないこと
- [ ] 既存のパイプライン構築・キャッシュ挙動が変化しないこと（HIT/MISS カウンタに異常なし）

# 影響範囲・リスク

- 低: 初期化順序/重複の解消のみ。None ガード漏れや import 時の順序依存があるとランタイム例外の可能性。
- 緩和策: Optional ガード + 末尾一括初期化で import 順序を安定化。ファイル単体の mypy/ruff を実行。

# 事前確認事項（Open Questions）

- [ ] `_GLOBAL_COMPILED_MAXSIZE` の取得を `common.settings.get().COMPILED_CACHE_MAXSIZE` へ統一しますか？
- [ ] 参照箇所の try/except を `if _GLOBAL_* is not None` に置換して明示条件にそろえてよいですか？（可読性優先）
- [ ] 初期化ブロックの配置は `__all__` 定義直後で問題ないですか？（モジュール import 時に確実に走る位置）

# ロールバック

- 単一ファイルの局所変更につき、問題発生時は差分を元に戻すだけで復旧可能。

