# clip: 平面性しきい値 `eps_abs`/`eps_rel` を定数化（引数削除）計画

目的
- `clip` の引数から `eps_abs`/`eps_rel` を削除し、モジュール内定数で管理する。
- API/GUIの簡素化と、分岐の一貫性を高める（モード選択は一定の基準で固定）。

背景
- `choose_coplanar_frame` は `max(eps_abs, eps_rel * 対角長)` をしきい値に共平面判定。
- しきい値の微妙な変更でキャッシュキーに含まれない差分が生じ得るため、外部公開パラメータとしての価値が低い。

変更内容（チェックリスト）
- [ ] API 署名の破壊的変更
  - [ ] `clip` シグネチャから `eps_abs: float` と `eps_rel: float` を削除
  - [ ] 関数 docstring から両引数の説明を削除/整理
  - [ ] `__param_meta__` から `eps_abs`/`eps_rel` を削除
- [ ] 内部定数の導入
  - [ ] `src/effects/clip.py` に `_PLANAR_EPS_ABS = 1e-5` と `_PLANAR_EPS_REL = 1e-4` を定義
  - [ ] `choose_coplanar_frame(..., eps_abs=_PLANAR_EPS_ABS, eps_rel=_PLANAR_EPS_REL)` で使用
- [ ] スタブ/メタの同期
  - [ ] `src/api/__init__.pyi` の `clip` から該当メタ行/引数を削除
- [ ] ドキュメント注記
  - [ ] モジュール/関数 docstring に「しきい値は定数運用（将来変更は内部実装）」を追記
- [ ] 検証（編集ファイル限定）
  - [ ] `ruff check --fix src/effects/clip.py`
  - [ ] `black src/effects/clip.py && isort src/effects/clip.py`
  - [ ] `mypy src/effects/clip.py`
  - [ ] スタブ整合（必要時）: `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`

互換性/移行
- 破壊的変更: 呼び出しで `eps_abs`/`eps_rel` を指定していた箇所は削除が必要。
- 既定値は従来デフォルト（1e-5 / 1e-4）と同一のため、実行結果の変化は基本的に無し。

備考（オプション）
- 将来必要になれば、環境変数/設定で内部定数を上書き可能にするが、今回は「定数」で導入。
