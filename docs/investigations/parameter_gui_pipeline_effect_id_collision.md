調査メモ: Parameter GUI が複数パイプラインで同名エフェクトのパラメータを共有してしまう件

概要
- 対象: `sketch/251022.py` の `draw()` 内で `pipe = E.pipeline.affine().fill()...` と `pipe2 = E.pipeline.affine().fill()` を併用。
- 事象: Parameter GUI に `affine` や `fill` のパラメータが 1 セットしか出ず、その 1 セットが `pipe` と `pipe2` の両方に同時適用される。

結論（原因）
- Parameter GUI のパラメータ ID（Descriptor ID）が「エフェクト名 + ステップ番号（パイプライン内の位置）」のみで構成され、パイプライン個体の識別子を含まないため、複数のパイプラインで同じ位置に同じエフェクトが来ると ID が衝突する。
  - 例: どちらのパイプラインでも `affine` が 0 番目のステップ → 両方とも `effect.affine#0.*` という同一 ID を生成。
  - `ParameterStore` は ID をキーに単一エントリとして扱うため、GUI 上も 1 セットしか表示されず、変更が両者へ反映される。

根拠（コード参照）
- エフェクト呼び出し時の識別子形成
  - `ParameterRuntime.before_effect_call()` が `ParameterContext(scope="effect", name=effect_name, index=step_index)` を構築（パイプラインごとの step_index のみ）。
    - src/engine/ui/parameters/runtime.py:102
  - `ParameterContext.descriptor_prefix` は `f"{scope}.{name}#{index}"`。
    - src/engine/ui/parameters/value_resolver.py:50
  - 実際の Descriptor ID は `f"{context.descriptor_prefix}.{key}"`（= `effect.<name>#<step>.<param>`）。
    - src/engine/ui/parameters/value_resolver.py:77
- パイプライン側で `step_index` は各パイプライン内の `enumerate(self._steps)` で決定される（パイプライン個体は区別されない）。
  - src/api/effects.py:249
  - その `step_index` がそのまま `before_effect_call(step_index=idx, ...) `へ渡される。
    - src/api/effects.py:255
- 形状（shape）は呼び出し回数ごとに `ParameterRegistry.next_index()` で連番を振っており、同フレーム内の多重出現が区別されるのに対し、エフェクトはこのレジストリを使っていない。
  - shape 側: src/engine/ui/parameters/runtime.py:78-80
  - effect 側: src/engine/ui/parameters/runtime.py:96-103（`ParameterRegistry` 未使用）

再現手順（参考）
1. `sketch/251022.py` を実行（Parameter GUI 有効）
   - sketch/251022.py: `run(..., use_parameter_gui=True, ...)`
2. Parameter GUI を開くと `affine`/`fill` のパラメータが 1 セットのみ表示される。
3. そのスライダーを操作すると、`pipe(base)` と `pipe2(text)` の両方の見た目が同時に変化する。

設計上の含意
- 現状の ID 設計（`effect.<name>#<step>.<param>`）は「単一パイプライン内での安定なインデックス付け」を優先しているが、複数パイプライン併用時の一意性が担保されない。
- `ParameterRuntime` には `_effect_registry` が存在し frame ごとにリセットされているが、effect の index 生成には利用されていない（将来的な拡張点と思われる）。
  - src/engine/ui/parameters/runtime.py:51,69

補足メモ（対応案の方向性のみ。実装変更は未実施）
- ID の一意性強化例（採用方針はB）:
  - パイプライン呼び出し単位の識別子（例: `pipeline#<n>`）を `ParameterContext` に含め、`effect.<pipeline_id>.<name>#<step>.<param>` のように拡張する。
  - GUI 側の保存/復元（persistence）と整合性を考慮する必要がある。

対象スケッチ
- sketch/251022.py:1

以上。

---

修正案（実装は未実施・提案）

方針B（採用・推奨）: パイプライン識別子（pipeline_uid）を導入
- 概要: 各 `Pipeline` インスタンスに一意な `pipeline_uid` を付与し、Descriptor ID に組み込むことで、同一効果名・同一ステップ位置でもパイプラインごとに恒久的に区別する。
- 具体:
  1) `PipelineBuilder`/`Pipeline` に `pipeline_uid` を追加（デフォルトは起動中一意な連番。ユーザーが `.label("base")` などで明示設定可能）。
  2) `Pipeline.__call__` → `ParameterRuntime.before_effect_call()` へ `pipeline_uid` を引き渡す（新引数）。
  3) `ParameterContext` に `pipeline` フィールドを追加し、`descriptor_prefix` を `f"effect@{pipeline}.{name}#{index}"` へ拡張。
  4) `ParameterValueResolver` は拡張後の `descriptor_prefix` を用いて ID を生成。
- 変更箇所（想定）:
  - src/api/effects.py: `PipelineBuilder` に `_uid` と `.label()`、`Pipeline` に `pipeline_uid` の保持と伝播。
  - src/engine/ui/parameters/runtime.py: `before_effect_call(..., pipeline_uid: str, ...)` に拡張し `ParameterContext` へ渡す。
  - src/engine/ui/parameters/value_resolver.py: `ParameterContext` の拡張と `descriptor_prefix` の更新。
- Descriptor ID 例: `effect@base.affine#0.scale`, `effect@text.affine#0.scale`
- 長所: フレーム順序に依存せず、パイプライン単位で安定。GUI の保存/復元キーの明確性が高い。
- 短所: 破壊的変更（ID が変わる）。永続済みの旧キー（`effect.<name>#<idx>.*`）は自動移行されないため、必要なら移行ロジックを別途用意。

方針C（参考・非推奨）: パイプライン構造署名（効果名列のハッシュ）を ID に組み込む
- 概要: ステップ名列をハッシュ化して ID に含める。
- 問題: 構造が同一な 2 つのパイプラインは依然として同一 ID となり、今回の要件（同一構成でも別々に調整したい）を満たさない。

推奨
- 現状の要件（pipe と pipe2 を別々にコントロール）に対しては「方針B（pipeline_uid 導入）」を採用。

互換性・マイグレーション
- 旧キーとの互換は無し。`persistence.py` のロード時に旧フォーマット（`effect.<name>#<idx>.*`）を検出した場合、初回のみ `pipeline_uid` 未指定の最初のパイプラインへ移譲する簡易マイグレーションを追加する選択肢あり（要件次第）。

確認テスト案
- `tests/ui/parameters` に以下の観点を追加:
  - 2 つのパイプライン（同一構成）で `affine`/`fill` の Descriptor ID が衝突しない。
  - 片方の GUI override が他方へ影響しない。
  - 方針Bの場合、`.label("base")` と `.label("text")` の指定で ID に反映される。
