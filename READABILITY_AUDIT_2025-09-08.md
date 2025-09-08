# 可読性・単純化レビュー（2025-09-08）

本ドキュメントは、現行コードベース（src/ 配下）を「可読性が高く、素直でシンプル」な実装へ近づけるために走査した結果の要約です。実害のある不具合指摘は既存の CODE_REVIEW_2025-09-07.md に委ね、本書では読みやすさ／単純化にフォーカスします。

- 評価日時: 2025-09-08
- 前提: Python 3.10、行長 100、型ヒント必須（AGENTS.md）

## サマリー

- 全体として日本語ドキュメント化と責務分離は良好。特に `effects/` と `engine/render/` は読みやすい構成。
- 一方、以下の観点で「あと一歩」揃える余地あり。
  - 戻り値注釈の未記載（関数/メソッドの一部）
  - 英語 Docstring/コメントの混在
  - `type: ignore` の多用（主に外部依存）
  - 100 桁超の長行が点在
  - 名前やメタ情報の一貫性（プロジェクト名・型別名など）

## 改善提案（具体）

1) 戻り値型アノテーションの補完（簡単・効果大）
- 目的: 読み手に「副作用/戻り値有無」を即伝達し、mypy の誤検知も抑止。
- 観測: 戻り値注釈なしの定義（抜粋・件数上位）
  - src/effects/weave.py（13）
  - src/scripts/dummy_deps.py（11）
  - src/engine/io/controller.py（10）
  - src/engine/io/manager.py（8）、src/engine/io/helpers.py（7）
  - src/api/pipeline.py（6）、src/shapes/attractor.py（5） ほか
- 推奨: 
  - ライフサイクル系 `__init__`/更新系は `-> None` を明示。
  - 表示/文字列化は `__repr__ -> str`、イテレータは具体の `Iterator[...]`。
  - コールバック登録/解除は `-> None`、ファクトリは戻り型を具体化。

2) 英語 Docstring/コメントの日本語化（方針の統一）
- 目的: コメント言語の統一で認知負荷を下げる。
- 該当（例）:
  - src/engine/__init__.py: "Engine package ..."
  - src/common/logging.py: モジュール Docstring と関数 Docstring が英語
  - src/api/pipeline.py: 一部 Docstring（"Return a serializable spec" / "Create a Pipeline ..." / "Heuristic ..."）
  - src/effects/weave.py: 英語 Docstring/コメントが点在（アルゴリズム説明）
- 推奨: 既存の詳細さは維持しつつ、要約を先頭に日本語で。長文は箇条書きで簡潔に。

3) `type: ignore` の削減と包み込み型の導入
- 目的: 読み手が「なぜ無視するのか」を明確化し、将来の型改善余地を残す。
- 観測: 49 箇所（mido/shapely/ModernGL 由来が大半）。
- 推奨:
  - mido: 開発依存に `types-mido` を追加（可能なら）。難しければ軽い Protocol を `engine/io/types.py` で定義し、`mido.Message` の最小面だけ参照する。
  - shapely: 2.x の `JOIN_STYLE` を使う分岐で `ignore` 削減（互換は CODE_REVIEW 参照）。
  - OpenGL: `moderngl.Context` の属性アクセスでの `ignore` は、専用の薄いラッパプロパティで吸収。

4) 長行の折り返し（行長 100 に統一）
- 観測: 以下に 100 桁超が点在（例）
  - src/api/pipeline.py: 例外メッセージ f-string（258/353 行）
  - src/api/runner.py: OpenGL/ウィンドウ初期化引数列
  - src/engine/io/controller.py, src/effects/offset.py ほか
- 推奨: f-string の部分結合、引数の改行揃え、コメントの要約化で対応（意味単位で折る）。

5) 別名/型の一貫性
- 目的: API の読み替えコストを下げる。
- 観測: `effects/registry.py` の `EffectFn = Callable[[Geometry], Geometry]` は実態（`g: Geometry, **params`）と不一致。
- 推奨: `EffectFn = Callable[..., Geometry]` に修正。コード側で未使用なら削除も可。

6) `util/geometry.py` の死蔵コードの簡素化
- 観測: `if TYPE_CHECKING: pass` は未使用。
- 推奨: ブロック自体を削除（必要なら将来の型のみ import を記述）。

7) モジュール・メタ情報の整合
- 観測: `pyproject.toml` と `README.md` のプロジェクト名が `pyxidraw5`/`PyXidraw5`、リポ名は `pyxidraw6`。
- 推奨: 名称統一（読み手の混乱防止）。

8) 実行方法の表現を簡潔に
- 観測: `main.py` は `sys.path` を操作し `from api import ...` を可能に。
- 推奨: README に「開発時は `PYTHONPATH=src python main.py` 推奨」と併記し、`sys.path` 挿入は最小限のコメントで意図を説明（既に良好）。

9) Lint/型チェックのカバレッジを段階拡大
- 観測: mypy 対象が `src/util/utils.py` のみ。
- 推奨: `files` を段階的に拡張（例: `util/**` → `common/**` → `api/**`）。`ruff` も ANN ルールの一部（戻り値必須等）を限定的に有効化すると自動検出が進む。

---

## 最小変更サンプル（読みやすさ向上のための方針例）

- `effects/registry.py`
  ```python
  # 実態に合わせた型の簡素化
  EffectFn = Callable[..., Geometry]
  ```

- `engine/__init__.py`
  ```python
  """エンジン層（core/io/pipeline/render/ui/monitor）。"""
  ```

- `engine/io/controller.py`
  ```python
  def __repr__(self) -> str:
      return f"MidiController(port_name={self.port_name}, mode={self.mode})"
  ```

---

## 実施チェックリスト（今回）
- [x] リポジトリ走査（rg による欠落注釈/長行/英語コメントの抽出）
- [x] 可読性・単純化の観点で主要改善点を洗い出し
- [x] 本ドキュメントの作成・保存
- [ ] 具体修正の適用（小粒 PR に分割）
- [ ] `ruff`/`mypy` 設定の段階的拡大
- [ ] 必要に応じて `types-mido` 等の型スタブ追加（開発依存）

## 推奨ワークフロー（小粒に進める）
1. `effects/registry.py` の型別名修正 + 英語 Docstring 部分の日本語化（安全・衝突少）。
2. `engine/__init__.py` と `common/logging.py` の Docstring を日本語へ。
3. `controller.py`/`manager.py`/`helpers.py` に `-> None/str` 等の戻り値注釈を追加。
4. 長行の折返し（例外メッセージ、引数列）。
5. mypy 対象を `src/common/**` まで拡大し、`type: ignore` を段階的に解消。
6. 名前の整合（pyproject/README のプロジェクト名統一）。

以上。実装パッチの作成も対応可能です。必要な範囲を指定いただければ、最小差分で提案します。
