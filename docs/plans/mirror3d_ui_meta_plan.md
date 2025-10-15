# effect: mirror3d 条件付き UI メタ導入計画

本ドキュメントは、mirror3d の mode 切替に伴う「不要パラメータが GUI に露出して UX が低下する」問題を、非破壊（API 互換）のまま `__param_meta__` の UI メタ拡張で解消する計画を示す。

## 目的（Goal）
- mode に応じて無関係なパラメータを GUI から隠す/無効化し、どの操作が効いているかを直観的にする。
- 公開 API（関数シグネチャ）は維持。mirror3d を 1 つのエフェクトとして使い続けられる。
- 実装はシンプルかつ他エフェクトにも再利用可能な仕組みにする。

## 背景 / 現状
- mirror3d のパラメータは mode によって有効・無効が分かれる。
  - 共通: `cx, cy, cz, axis, show_planes, mode`
  - azimuth 専用: `n_azimuth, phi0_deg, mirror_equator, source_side`
  - polyhedral 専用: `group, use_reflection`
- 現状 GUI は「draw 未指定の引数」をフラットに並べるため、モード外パラメータが混在し、操作しても結果が変わらない場面がある。
- `__param_meta__` に `mode` の `choices` はあるが、可視制御は未導入。`mirror_equator`/`source_side` は型情報の明記も未登録。

## 要件（Requirements）
- R1: mode の現在値に応じて表示/非表示（または Disable）を切り替える。
- R2: 非表示のパラメータはエフェクト呼び出し時に GUI から kwarg 注入しない（既定値のまま）。
- R3: 値は保持し、モードを戻した際に前回値を復元できる。
- R4: 実装は `__param_meta__` に閉じ、他エフェクトでも同じキーで使い回せる。
- R5: API・量子化・キャッシュ鍵生成の既存仕様を破らない（値の量子化は従来通り）。
- R6: 既定は「非表示」。設定で「Disable 表示」も選べる設計（後方互換オプション）。

## 提案（UI メタ拡張）
`__param_meta__` の各パラメータ項目に、以下の任意キーを追加する。

- `ui.visible_if: Mapping[str, list[Any]]`
  - 現在の「解決済みパラメータ値」（明示引数 > GUI 値 > 既定値）に対し、全条件 AND で一致する場合のみ表示。
  - 例: `{"mode": ["azimuth"]}` は mode が `"azimuth"` の時だけ表示。
  - 将来拡張: `ui.disable_if`（今回は導入見送り、仕様は同等で“Disable 表示”）。
- `group: str`
  - GUI 上のセクション名。例: `"Azimuth"`, `"Polyhedral"`。
- `help: str`
  - ツールチップ/補足文。例: 「mode='azimuth' で有効」等。

評価規則（UI 実装側）
- E1: 可視性の判定は毎フレーム/変更時に軽量評価（単純な等値のみ）。
- E2: 非表示項目は GUI からエフェクト呼び出し kwargs に含めない（値は内部ストアに保持）。
- E3: セクション（group）は、所属するいずれかの項目が可視のときに表示（空なら非表示）。

## 適用内容（mirror3d）
- 追加登録（型/説明）
  - `mirror_equator: {"type": "bool", "help": "mode='azimuth' で有効。赤道面での反転追加"}`
  - `source_side: {"type": "bool", "help": "mode='azimuth' かつ mirror_equator=True で有効。正側/負側の選択"}`
- 可視条件とグループ例（抜粋・擬似記法）
  - `n_azimuth`: `{..., ui.visible_if: {mode: ["azimuth"]}, group: "Azimuth", help: "mode='azimuth' で有効"}`
  - `phi0_deg`: `{..., ui.visible_if: {mode: ["azimuth"]}, group: "Azimuth"}`
  - `mirror_equator`: `{type: "bool", ui.visible_if: {mode: ["azimuth"]}, group: "Azimuth"}`
  - `source_side`: `{type: "bool", ui.visible_if: {mode: ["azimuth"], mirror_equator: [true]}, group: "Azimuth"}`
  - `group`: `{choices: ["T","O","I"], ui.visible_if: {mode: ["polyhedral"]}, group: "Polyhedral"}`
  - `use_reflection`: `{type: "bool", ui.visible_if: {mode: ["polyhedral"]}, group: "Polyhedral"}`
  - 共通（常時表示）: `cx, cy, cz, axis, show_planes, mode`

注記
- `source_side` は実装上 `bool | Sequence[bool]` だが、GUI は単一 `bool` とする（高度機能は将来の拡張項目へ）。
- 量子化/署名生成は従来通り（float のみ量子化）。UI が非表示にして注入しない値は鍵にも含まれないが、`mode` の違いが鍵に反映されるため実行キャッシュは自然に分離される。

## 実装ステップ（チェックリスト）
- [ ] S-1 仕様確定: 既定は「非表示」/設定で「Disable 表示」を選択可（要 GUI 設定の有無確認）。
- [ ] S-2 UI 実装: `ui.visible_if` の評価と項目の表示/注入制御を追加。
- [ ] S-3 UI 実装: `group` によるセクション分割（空セクションは非表示）。
- [ ] S-4 UI 実装: `help` ツールチップ表示。
- [ ] S-5 メタ更新: `src/effects/mirror3d.py` の `__param_meta__` に `mirror_equator`/`source_side` 登録と可視条件を追加。
- [ ] S-6 値保持: モード切替で各パラメータ値を保持/復元（内部ストア設計）。
- [ ] S-7 テスト: `tests/ui/parameters` に表示/非表示・注入有無の最小テストを追加。
- [ ] S-8 ドキュメント: `architecture.md` に UI メタ拡張（キーと評価規則）を追記。
- [ ] S-9 品質ゲート: 変更ファイル限定で `ruff/mypy/pytest` 緑化。必要に応じてスタブ同期確認。

## テスト計画（最小）
- T-1 mode=azimuth で `n_azimuth/phi0_deg/mirror_equator` が表示され、`group/use_reflection` は非表示。
- T-2 mode=polyhedral で `group/use_reflection` が表示され、`n_azimuth/phi0_deg` は非表示。
- T-3 `mirror_equator=False` のとき `source_side` は非表示、`True` にすると表示。
- T-4 非表示項目はエフェクト呼び出し kwargs に含まれない（モックで検査 or ログ出力で検査）。
- T-5 モードを往復すると、前回値が復元される。

## 互換性 / 移行
- API/シグネチャ変更なし。`__param_meta__` の追加キーは後方互換。
- UI 実装が導入されるまではメタは無害（単に未使用のフィールド）。

## リスクと対策
- Rk-1 UI 側の分岐評価不備 → 小さな等値評価に限定、単体テストで守る。
- Rk-2 隠した値の注入漏れ → kwargs 生成箇所を単一点に集約しテストで検知。
- Rk-3 既存セーブ/ロードとの整合 → 設定スキーマは値を保持し、可視性は表示時に判定。

## オープン事項（要確認）
- Q1 既定挙動: 非表示 vs Disable（推奨: 非表示）。
- Q2 セクション見出しの表記: `Azimuth` / `Polyhedral` で良いか（日本語表記にするか）。
- Q3 `source_side` の UI（単一 bool で十分か、将来拡張の余地）。

## DoD（完了条件）
- mirror3d の GUI でモード外パラメータが表示されない（または Disable で理由が示される）。
- 非表示時にスライダー操作が UX を阻害しない（効果なしの操作がなくなる）。
- 変更ファイルに対する `ruff/mypy/pytest` 緑。必要時 `architecture.md` 更新済み。

