## displace エフェクト amplitude_mm ベクトル対応計画

### ゴール

- `src/effects/displace.py` の `amplitude_mm` を x, y, z の 3 成分で指定できるようにし、各軸ごとに変位量を制御可能にする。
- 既存のスカラー指定 (`float`) をそのまま受け入れ、指定がないコードの挙動は現状と同等（等方的な振幅）に保つ。

### 仕様案（ドラフト）

- 関数シグネチャ:
  - `amplitude_mm: float | Vec3 = (8.0, 8.0, 8.0)`
  - `Vec3` は `(ax, ay, az)` を mm 単位の変位量として解釈。
- スカラー指定時:
  - `amplitude_mm` が `float` または `int` の場合は `ax = ay = az = float(amplitude_mm)` として等方的な振幅とする。
- ベクトル指定時:
  - `amplitude_mm` がシーケンスの場合は `ensure_vec3` で `(ax, ay, az)` に正規化する。
  - 各成分は `noise_offset` の対応軸に対してスカラー倍される（`dx * ax`, `dy * ay`, `dz * az`）。
- メタデータ:
  - `displace.__param_meta__["amplitude_mm"]` は Vec3 ベースのレンジ指定に変更する案：
    - `{"type": "number", "min": (0.0, 0.0, 0.0), "max": (50.0, 50.0, 50.0)}` を想定。
  - step については、他パラメータとの整合を見つつ必要であれば追加（例: `step`: 0.5 or 1.0）。
- パイプライン署名:
  - `amplitude_mm` は `float | Vec3` として扱われるが、署名生成は既存の `__param_meta__` と `params_signature` のルール（Vec3 含む RangeHint, step による量子化）に従う。

### 実装タスクチェックリスト

- [x] API 仕様確定: `amplitude_mm` を `float | Vec3` とし、スカラーは等方的な振幅として扱うことを明文化する。
- [x] `src/effects/displace.py` の `_apply_noise_to_coords` を成分別振幅に対応させる（`intensity: float` → `amplitude: tuple[float, float, float]` 相当の形で扱う）。
- [x] `displace` 本体で `amplitude_mm` を `Vec3` に正規化し、`_apply_noise_to_coords` に渡す処理を実装する。
- [x] `spatial_freq` の既存挙動（`float | Vec3` → Vec3 正規化）は維持しつつ、`amplitude_mm` の処理と混同しないよう整理する。
- [x] `displace.__param_meta__["amplitude_mm"]` を Vec3 レンジ指定に更新する（`min/max` を 3 成分タプルに変更）。
- [ ] `architecture.md` 内の `displace` 説明を更新し、`amplitude_mm` がベクトル指定を受け付けることを反映する。
- [x] `docs/effects_arguments.md` の `displace` セクションを更新し、`amplitude_mm: float | Vec3` とする。
- [x] 公開 API スタブ (`src/api/__init__.pyi`) の `displace` シグネチャを更新する（`amplitude_mm: float | tuple[float, float, float]` を許容）。
- [x] スタブ再生成 (`python -m tools.gen_g_stubs`) と差分確認を行う。
- [x] `tests/effects/test_displace_minimal.py` にベクトル振幅のテストケースを追加する（例: `amplitude_mm=(10.0, 0.0, 0.0)` で主に x 軸に揺れが出ることを確認）。
- [x] `tests/ui/parameters` 配下で `displace` / `amplitude_mm` に依存するテストがあれば、Vec3 対応後も成立するように更新する（既存テストは dummy effect を使用しており、変更不要であることを確認）。
- [x] 変更ファイルに対して `ruff check --fix`, `black`, `isort`, `mypy`, `pytest -q`（対象ファイルまたはマーカー指定）を実行する。

### 要確認事項（ユーザーに確認したい点）

- [x] `amplitude_mm` の Vec3 デフォルト値:
  - 現状のデフォルト `8.0` を維持し、「未指定時は等方 8mm」とする運用でよいか？
  - それとも Vec3 デフォルト（例: `(8.0, 8.0, 8.0)`）に変更しても問題ないか？；はい → デフォルトを `(8.0, 8.0, 8.0)` に変更済み。
- [x] GUI での表示方法:
  - `amplitude_mm` を Vec3 とする場合、GUI では 1 つのスライダ（等方）を維持したいか、それとも x/y/z の 3 スライダを出したいか？；xyz の 3 スライダで → `min/max` を Vec3 にしたため 3 成分スライダとして扱われる。
  - 3 スライダとする場合、既定値は `(8.0, 8.0, 8.0)` とし、等方調整を簡単に行う UI（例: リンクボタン）を検討すべきか？（リンクボタンなど追加 UI は今回は未実装）
- [x] `__param_meta__` のレンジ:
  - 各成分の `min/max` を `(0.0, 0.0, 0.0)` / `(50.0, 50.0, 50.0)` とする案でよいか？;はい → `displace.__param_meta__` に反映済み。
  - 最大値 50mm が十分か、それともより広いレンジが必要か？（現状は 50mm のまま）
- [x] 後方互換性の許容範囲:
  - 既存コードは `amplitude_mm` を `float` で呼んでいる想定だが、`float | Vec3` への拡張のみであれば後方互換は保たれる。これで十分か？；はい → `float` 指定はそのまま等方振幅として解釈されることを確認。
  - もし API 名や意味を変える場合（例: `amplitude_mm` を等方専用のままにし、ベクトル用に別パラメータを追加する）案もあるが、その必要性はあるか？（名称変更・別パラメータ追加は行っていない）

### 追加の改善アイデア（任意）

- [ ] `displace` に「z 軸だけ揺らす」「xy 平面のみ揺らす」といったプリセットを持たせるかどうか（必要なら別エフェクトまたはヘルパとして検討）。
- [ ] `amplitude_mm` と `spatial_freq` を GUI 上でリンクさせる（強い揺れのときは周波数を下げる等）ような UX 改善のニーズがあるかどうか。

---

この計画内容で問題なければ、上記チェックリストに従って実装とテストを進めます。必要な修正・追加や希望する仕様があれば、このファイルへの追記内容として指示してください。
