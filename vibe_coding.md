# Vibe Coding のための軽量ガードレール

vibe coding は推進力が出る反面、「構造の劣化」はテストで検知しづらい（挙動テストは通るのに、余剰コードや逆依存・矛盾が紛れ込む）。
対策の要点は「構造そのものを“テスト対象”にする」こと。下の最小セットだけで、勢いを殺さずに設計の一貫性を自動監視できる。

---

## まずは結論：入れるべき軽量ガードレール（コピペ可）

1. Architecture Canvas（1 枚）— 依存方向・不変条件を明文化
2. アーキテクチャテスト（pytest）— 依存の逆流・層違反を CI で落とす
3. Public API スナップショットテスト — “表の顔”の破壊的変更を検知
4. Health Check コマンド — ruff/mypy/pytest/vulture を一発実行
5. ミニ ADR（設計決定メモ）— vibe 中の判断を 10 行で残す

以下、最小構成の雛形と導入手順。Python を想定し、`src/yourpkg/` の一般的レイアウトを例にする（`yourpkg` はあなたのパッケージ名に置換）。

---

## 1. Architecture Canvas（1 枚）

対象ファイル: `docs/architecture-canvas.md`

```md
# Architecture Canvas (1-pager)

## モジュール境界（例：ヘキサゴナル寄りの薄い層）

- `yourpkg.domain/` : ビジネスルール（純粋、外界知らない）
- `yourpkg.usecases/` : アプリケーションサービス（オーケストレーション）
- `yourpkg.adapters/` : I/O 適合層（DB/HTTP/UI との接続）
- `yourpkg.infra/` : 具象インフラ（クライアント・ゲートウェイ等）
- `yourpkg.app/` : CLI/GUI/API エンドポイント

## 許可する依存方向（層番号：小さいほど内側）

domain(0) ← usecases(1) ← adapters(2) ← app(3)
infra(2) は adapters/usecases から参照されてもよいが、逆は不可。

## 不変条件（例）

- domain は標準ライブラリ以外に依存しない
- domain は usecases/adapters/infra/app を import しない
- usecases は infra を直接 import しない（port 経由）
- `print()` 禁止（ログは logger 経由）
- 循環依存を作らない
```

---

## 2. アーキテクチャテスト（pytest で静的解析）

役割: AST で import グラフをなぞり、層違反を落とす。`PKG` をあなたのパッケージ名に変更。

対象ファイル: `tests/test_architecture.py`

```python
# tests/test_architecture.py
from __future__ import annotations

import ast
import pathlib
from typing import Iterator, Tuple

# ここだけ自分のプロジェクトに合わせて編集
PKG = "yourpkg"  # ← パッケージ名

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
PKG_ROOT = SRC_DIR / PKG

LAYERS = {"domain": 0, "usecases": 1, "adapters": 2, "infra": 2, "app": 3}
CHECK_DIRS = set(LAYERS.keys())


def iter_py_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    for p in root.rglob("*.py"):
        if any(part in {"tests", "__pycache__"} for part in p.parts):
            continue
        yield p


def module_name_from_path(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC_DIR).with_suffix("")
    return ".".join(rel.parts)


def section_of(module: str) -> str | None:  # yourpkg.domain.x.y -> "domain"
    segs = module.split(".")
    if not segs or segs[0] != PKG:
        return None
    return segs[1] if len(segs) > 1 and segs[1] in CHECK_DIRS else None


def iter_import_edges(py_path: pathlib.Path) -> Iterator[Tuple[str, str]]:
    src_mod = module_name_from_path(py_path)
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield src_mod, alias.name  # e.g., "numpy"
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # 相対 import
                src_parts = src_mod.split(".")
                base = ".".join(src_parts[:-node.level])
                mod = node.module or ""
                tgt = f"{base}.{mod}" if mod else base
                yield src_mod, tgt
            else:
                yield src_mod, node.module or ""


def test_layering_rules():
    violations = []
    for py in iter_py_files(PKG_ROOT):
        for src, tgt in iter_import_edges(py):
            s_sec, t_sec = section_of(src), section_of(tgt or "")
            if s_sec is None or t_sec is None:
                continue  # パッケージ外 or 対象外
            s_layer, t_layer = LAYERS[s_sec], LAYERS[t_sec]
            # 依存は外側 → 内側のみ許可（同層は OK）
            if s_layer < t_layer:
                violations.append(
                    f"[{py}] {src} (layer {s_layer}, {s_sec}) -> {tgt} (layer {t_layer}, {t_sec})"
                )
    if violations:
        msg = "\n".join(violations)
        raise AssertionError(f"Layering violations detected:\n{msg}")


def test_no_cycles_within_pkg():
    # 簡易：同一ファイル内の循環は AST で検知しづらいので、層制約でほぼ防げる。
    # 必要なら importlib を使って強化しても OK。
    assert True
```

---

## 3. Public API スナップショットテスト

役割: 外向き API（domain/usecases の公開クラス・関数）の“不用意な増減”を検知。
振る舞いテストは通るが、同種の関数が重複して増える問題を抑える。

対象ファイル: `tests/test_public_api_snapshot.py`

```python
# tests/test_public_api_snapshot.py
from __future__ import annotations

import ast
import json
import os
import pathlib
from typing import Dict, List

PKG = "yourpkg"  # ← パッケージ名

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
PKG_ROOT = SRC_DIR / PKG

TARGET_SECTIONS = {"domain", "usecases"}
SNAP_PATH = pathlib.Path(__file__).parent / "_snapshots" / "public_api.json"
UPDATE = os.getenv("UPDATE_SNAPSHOT") == "1"


def iter_modules(section: str):
    base = PKG_ROOT / section
    for p in base.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        rel = p.relative_to(SRC_DIR).with_suffix("")
        mod = ".".join(rel.parts)
        yield mod, p


def public_defs(py_path: pathlib.Path) -> List[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
    return sorted(names)


def build_snapshot() -> Dict[str, List[str]]:
    snap: Dict[str, List[str]] = {}
    for sec in TARGET_SECTIONS:
        for mod, path in iter_modules(sec):
            defs = public_defs(path)
            if defs:
                snap[mod] = defs
    return dict(sorted(snap.items()))


def test_public_api_snapshot():
    current = build_snapshot()
    SNAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if UPDATE or not SNAP_PATH.exists():
        SNAP_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False))
        if UPDATE:
            return
        # 初回はスナップショットを作成して終了
        return
    saved = json.loads(SNAP_PATH.read_text(encoding="utf-8"))
    assert current == saved, (
        "Public API changed.\n"
        "意図的な変更なら `UPDATE_SNAPSHOT=1 pytest -q` でスナップショットを更新してください。"
    )
```

初回はスナップショットが生成される。意図した API 変更のときだけ更新する:

```bash
UPDATE_SNAPSHOT=1 pytest -q
```

---

## 4. Health Check コマンド

役割: 「vibe の後に 15 分だけ整備」ルーチンを作る。
ruff（Lint/Format + いくつかの設計匂いルール）, mypy（型整合）, pytest, vulture（死蔵コード）を一括実行。

`Makefile`:

```makefile
.PHONY: lint typecheck test deadcode arch health

SRC=src

lint:
	ruff check $(SRC) tests
	ruff format --check $(SRC) tests

typecheck:
	mypy --strict $(SRC)

test:
	pytest -q

deadcode:
	vulture $(SRC) --min-confidence 80

arch:
	pytest -q tests/test_architecture.py

health:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) deadcode
	$(MAKE) arch
```

`pyproject.toml`（最小）:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
extend-select = ["E", "F", "W", "B", "I", "SIM", "C4", "T20", "ERA"]
# - B: bugbear、I: import 順、SIM: simplify、C4: comprehensions
# - T20: print 系、ERA: コメントアウトされたコード
ignore = []

[tool.mypy]
python_version = "3.11"
strict = true
mypy_path = ["src"]
warn_unused_ignores = true
warn_redundant_casts = true
disallow_any_generics = true
no_implicit_optional = true
```

使うコマンド:

```bash
pip install pytest ruff mypy vulture
make health
```

---

## 5. ミニ ADR（設計決定メモ）

役割: vibe 中に生じた“判断”を 10 行で記録。後で矛盾の温床になりやすい箇所を可視化。

対象ファイル: `docs/adr/0001-<slug>.md`

```md
# タイトル（例：メッシュ生成器を usecases に置く）

- 日付: 2025-09-11
- コンテキスト: 何が痛かったか／制約（I/O 遅延、依存の都合など）
- 決定: 何を選んだか（具体的な配置・依存方向）
- 代替案: 少なくとも 1 つ書く
- 影響: 良い点と負債（将来どう見直すかのトリガー）
```

---

## 運用リズム（vibe を殺さない最短コース）

- 前（3 分）: 1 枚キャンバスの「許可する依存方向」「不変条件」を一瞥。今日触る層を 1 つに限定。
- 中: 1〜2 テスト（赤 → 緑）ごとにマイクロコミット（メッセージに `[vibe]`）。外部公開に触れたら `tests/_snapshots` が赤くなる → そのまま進んで OK。
- 後（15 分整備）: `make health` → 赤になった箇所だけ直す／ADR を 1 枚追記 → ここで初めて `UPDATE_SNAPSHOT=1` を実行。

---

## どこまで防げるか

検知できる:

- 逆依存・層違反（domain が adapters を import 等）
- Public API の無自覚な増減（重複 API や命名揺れの温床）
- 使われない関数・古いヘルパー（vulture）
- `print` やコメントアウト塊などの設計ノイズ

検知しづらい:

- 機能重複でも名前が完全に違うケース
- 隠れた性能退行（別途ベンチやプロパティテストを）

性能・数値特性が重要なコード（CAE/幾何/最適化など）は、プロパティテストを混ぜると「設計と数理の不変条件」を同時に守れる（例: `||normalize(v)|| ≈ 1`、メッシュ算出で 面積 `≥ 0` など）。

---

## 最後に（導入の指針）

- まず `test_architecture.py` と `Makefile` の 2 点だけ入れてみてください。
- 次に Public API スナップショットを足すと、“テストは通るのに API が増殖” 問題が目に見えて止まる。
- 迷ったら Canvas に“例外”を 1 行追記 → 同時に ADR を 1 枚残す。ルール変更も履歴化すれば、矛盾は設計ドキュメント上で解消される。

必要なら、あなたのリポ構成（`src` の有無、パッケージ名、既存の CI）に合わせてこの雛形を調整して出す。
