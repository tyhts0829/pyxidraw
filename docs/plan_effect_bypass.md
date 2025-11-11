エフェクト共通バイパス（parameter_gui 連携）改善計画

どこで・何を・なぜ
- どこで: `effects.registry.effect` デコレータ、`api.effects.PipelineBuilder`、`engine.ui.parameters.ParameterRuntime`
- 何を: すべてのエフェクトに共通の「バイパス」切り替えを追加し、Parameter GUI から制御可能にする。
- なぜ: 各エフェクトの有効/無効をランタイムに切り替えられるようにし、作業効率と試行錯誤性を高めるため。

目標（仕様）
- すべてのエフェクトに `bypass: bool` を共通導入（既定 False）。
- Parameter GUI に各パイプラインステップごとに「Bypass」を表示（カテゴリは当該ステップのパイプライン UID）。
- GUI 値が True のステップはパイプラインに追加せずスキップする（実行負荷ゼロ）。
- 署名生成（キャッシュ鍵）は `bypass` を含めない（drop）ため、バイパス切替は安全に再コンパイル/再構築される。
- 永続化: 既存の ParameterStore の override 保存/復元により、`bypass` 値はセッション間で保持される。

設計方針（単純・明確）
1) デコレータは実装本体を書き換えず、マーカー属性を付与する。
   - `__effect_impl__`: 既存どおりオリジナル関数参照（将来のラッパ導入に備え維持）。
   - `__effect_supports_bypass__ = True`、`__effect_bypass_param__ = "bypass"` を付与。
2) ParameterRuntime は effect 呼び出しのたびに、該当ステップ用の Bypass パラメータを登録する。
   - Descriptor ID: `effect@{pipeline_uid}.{effect_name}#{step_index}.bypass`
   - value_type: bool、default: False、param_order: -1（ステップ内の先頭に表示）
   - store.resolve() で現在値を取得し、`before_effect_call` の戻り値に `{"bypass": current}` を混入。
3) PipelineBuilder は `before_effect_call` からの `resolved` に `bypass` が含まれていれば `True` でステップ追加をスキップ、`False` で通常追加。
   - 追加時は `bypass` キーを pop してから `_params_signature(impl, resolved)` を計算（署名から除外）。

影響範囲と互換性
- 既存のエフェクト実装（関数シグネチャ/メタ）は変更不要。
- Parameter GUI が存在しない実行（ランタイム未有効）では、従来どおり全ステップが適用される。
- キャッシュ鍵は `impl` のシグネチャと既存パラメータのみで算出され、`bypass` による不整合は発生しない。

変更ファイル（予定）
- src/effects/registry.py
  - `effect()` デコレータで登録時にマーカー属性を付与。
- src/engine/ui/parameters/runtime.py
  - `before_effect_call()` で Bypass Descriptor 登録と現在値の解決、`resolved` への混入。
- src/api/effects.py
  - `PipelineBuilder.__getattr__()._adder` で `bypass` を見てステップをスキップ、`params_tuple` 算出前に `pop`。
- docs/architecture.md（実装後）
  - エフェクト/パイプライン/GUI の責務に「バイパス」項を追記。
- （必要に応じて）tests/ui/parameters/test_effect_bypass.py を新設。

実装チェックリスト（DoD）
1) Decorator
   [ ] デコレータが `__effect_supports_bypass__ = True` と `__effect_bypass_param__ = "bypass"` を付与する。
   [ ] 既存の `__effect_impl__` 参照は維持。
2) ParameterRuntime
   [ ] `before_effect_call()` 内で Descriptor を生成し `ParameterStore.register()` する。
   [ ] Descriptor: `source="effect"`、`category=pipeline_uid or "effect"`、`step_index` セット、`param_order=-1`。
   [ ] `store.resolve(id, default=False)` の結果を `resolved["bypass"]` として返す。
3) PipelineBuilder
   [ ] runtime が有効な経路で `resolved.pop("bypass", False)` を評価。
   [ ] True なら `self._steps` に追加しない（完全スキップ）。
   [ ] False なら従来どおり追加。
4) UI/永続化
   [ ] Parameter GUI 上に各ステップの Bypass が表示され、先頭に並ぶ。
   [ ] `save_overrides`/`load_overrides` で Bypass 状態が保存/復元される。
5) テスト（最小）
   [ ] Snapshot/Runtime 経由で Bypass の Descriptor 登録を検証。
   [ ] Bypass=True でステップ未追加、False で追加されることを `PipelineBuilder.build()` 経由で検証。
6) 品質ゲート
   [ ] 変更ファイルに対して `ruff/black/isort/mypy` を通過。
   [ ] 変更に関連するテストが緑。
   [ ] architecture.md を更新。

UI 表示と用語
- パラメータ名: `bypass`（ラベルは「Bypass」）。
- ステップ内の最上段に配置（`param_order=-1`）。
- 既定値 False（適用）、True で無効化（バイパス）。

オープン質問（要確認）
- ラベル表現: 「Bypass」/「Enabled」どちらにしますか？（現案: Bypass=True でスキップ）
- Bypass の既定値は常に False で良いですか？（再生時は常に適用）
- 将来的に Wet/Dry ミックス（0..1）も追加しますか？（今回は非対象、設計余地を残す）

リスク/注意点
- `PipelineBuilder` に小改修が入るため、パイプラインの compiled キャッシュキーに影響しないことを確認（`bypass` は `pop` 済み）。
- Descriptor ID はインデックスを含むため、同名エフェクトの複数回使用でも衝突しない。

作業手順（短期）
1) Decorator にマーカー属性追加（src/effects/registry.py）。
2) Runtime で Bypass Descriptor 登録 + 値解決を追加（src/engine/ui/parameters/runtime.py）。
3) PipelineBuilder でステップスキップを実装（src/api/effects.py）。
4) 最小のテストを追加し、変更ファイルに限定して lint/type/test を実行。
5) architecture.md を更新。

検証コマンド（変更ファイル優先）
- Lint: `ruff check --fix src/effects/registry.py src/engine/ui/parameters/runtime.py src/api/effects.py`
- Format: `black {files} && isort {files}`
- TypeCheck: `mypy {files}`
- Smoke: `pytest -q -k "effect and bypass"`（導入テスト追加後）

完了条件
- 変更ファイルに対する `ruff/mypy/pytest` が成功。
- Parameter GUI で Bypass が操作でき、True でステップがスキップされる。
- ドキュメント（architecture.md）が実装と合致。

