# api.effects と culling の境界整理計画（UX 維持 + 汚染回避）

目的

- 既存 UX `E.pipeline.culling(geos=[...])` は維持する。
- `src/api/effects.py` から `PipelineBuilder.culling(...)` という **culling 専用メソッド定義を削除** し、このモジュールを culling のためだけに汚染しない。
- culling 固有のロジック（Geometry 正規化、LazyGeometry 実体化、self-culling、face-aware occluder 等）はすべて `src/effects/culling.py` 側に閉じ込める。
- `api.effects` 側は「汎用的なエフェクト呼び出しフロント」としてのみ振る舞い、culling に限らない一般化された仕組みで `E.pipeline.culling(geos=[...])` を実現する。

現状整理

- 現在の `E.pipeline.culling(geos=[...])` は、`PipelineBuilder` 上の専用メソッドで実装されている:

  ```python
  class PipelineBuilder:
      ...
      def culling(..., geos: Sequence[Geometry | LazyGeometry], ...) -> list[Geometry]:
          from effects.registry import get_effect as _get_effect
          runtime = get_active_runtime()
          ...
          impl = getattr(fn, "__effect_impl__", fn)
          return impl(geos=norm_geos, thickness_mm=..., ...)
  ```

- これにより `src/api/effects.py` が culling 固有の責務を多く抱えている:
  - `LazyGeometry.realize()` を呼んで Geometry に実体化。
  - Parameter GUI ランタイム (`get_active_runtime` / `before_effect_call`) と直接対話。
  - `effects.registry.get_effect("culling")` を直接叩き、`__effect_impl__` まで解決。
  - `bypass` フラグを解釈し、戻り値として `list[Geometry]` を返す特別なユーティリティ。
- 一方 `PipelineBuilder.__getattr__` は「任意のエフェクト名に対するステップ追加」を汎用的に処理する薄い仕組みになっており、こちらはモジュールの意図（パイプライン宣言）と整合している。

設計方針

- `api.effects` の役割:
  - Pipeline/Builder の宣言（effect 名 + パラメータ）とキャッシュ管理に限定する。
  - 「エフェクトを今すぐ適用する」ような実行ロジックが必要な場合も、**エフェクト名に依存しない汎用的なパターン** でのみ提供する。
  - culling 固有のメソッドや特別分岐を持たない（文字列 `"culling"` を直接参照しない）。
- `effects.culling` の役割:
  - `Geometry | LazyGeometry | ObjectRef` を含む入力の正規化。
  - face-aware occluder / self-culling / 3D 対応を含む culling ロジックのすべて。
  - 「複数 Geometry 間の隠線処理」を担う公開エフェクト (`@effect`) と、その内部実装。
- UX の要件:
  - `E.pipeline.culling(geos=[p1(g1), p2(g2), ...])` が引き続き `list[Geometry]` を返す。
  - ただし、この UX を実現するコードは「汎用の『geos を受け取るエフェクトの即時実行モード』」として実装し、culling 専用の特別扱いを避ける。

新方針（案 E: geos キーをトリガとする汎用「即時エフェクト実行」モード）
----------------------------------------------------------------------

発想:

- `PipelineBuilder.__getattr__` で、名前付きエフェクト `name` の呼び出しを一括処理している点を活かす。
- ここに **「geos キーワード引数が渡された場合は、パイプラインのステップ追加ではなくエフェクトを即時実行する」** という汎用モードを追加する。
- これにより:
  - `E.pipeline.culling(geos=[...])` は「name='culling', geos=...」の即時実行モードとして処理される。
  - `E.pipeline.rotate(angle=...)` のような従来のステップ登録は、従来どおり geos を渡さないためパイプライン構築モードとして扱われる。
  - `api.effects` 側には「geos を持つ任意のエフェクトを即時実行する汎用ロジック」しか存在せず、culling 固有のコードは一切残らない。

イメージコード（概念）

```python
class PipelineBuilder:
    def __getattr__(self, name: str) -> Callable[..., PipelineBuilder | Any]:
        def _adder(**params: Any) -> PipelineBuilder | Any:
            # geos が含まれる場合は「即時実行モード」
            if "geos" in params:
                geos = params.pop("geos")
                runtime = get_active_runtime()
                from effects.registry import get_effect as _get_effect

                fn = _get_effect(name)
                resolved = dict(params)
                if runtime is not None:
                    resolved = runtime.before_effect_call(
                        step_index=0,
                        effect_name=name,
                        fn=fn,
                        params=resolved,
                        pipeline_uid=str(self._uid or ""),
                        pipeline_label=self._label_display,
                    )
                impl = getattr(fn, "__effect_impl__", fn)
                # geos はそのままエフェクト側へ委譲（正規化は effects 側の責務）
                return impl(geos=geos, **resolved)

            # geos が無い通常ケースは従来どおり「ステップ追加」
            ...
            self._steps.append((name, params_tuple))
            return self

        return _adder
```

- 重要なポイント:
  - `name` は汎用（"culling" に限らない）。
  - geos の中身（`Geometry` / `LazyGeometry` / ObjectRef）の正規化は、エフェクト側（`effects.culling`）の責務とし、`api` 層では一切触れない。
  - GUI ランタイムとのやり取り（`before_effect_call`）も「効果名に依存しない共通フロー」として扱う。

これにより:

- `E.pipeline.culling(geos=[...])` は「geos を受け取る culling エフェクトの即時実行」として実現される。
- `src/api/effects.py` から `PipelineBuilder.culling` メソッド定義は削除できる。
- culling の実装と入力正規化は `effects.culling` にだけ存在し、モジュール間の責務分離が明確になる。

具体的な改善ステップ（チェックリスト）

1. 現状 UX と依存関係の整理

- [ ] `rg "E.pipeline.culling" -n` でリポジトリ全体を検索し、`E.pipeline.culling(geos=...)` の実利用箇所（特に `sketch/251115.py` と docs 内のコード例）を洗い出す。
- [ ] `effects.culling.culling(...)` のシグネチャと挙動を再確認し、「`geos` 引数だけ渡せば自前で `_normalize_geos` してくれる」前提を整理する。
- [ ] `PipelineBuilder.__getattr__` の既存ロジック（Parameter GUI ランタイムとの連携、`bypass` 処理、ステップ登録）を把握し、そこに即時実行モードを安全に組み込めるか検討する。

2. `PipelineBuilder.__getattr__` に汎用「即時実行モード」を追加

- [ ] `__getattr__` 内で `params` を受け取った際、`"geos" in params` かどうかで分岐する設計をまとめる（上記イメージコードをベースに詰める）。
- [ ] 即時実行モードでは:
  - [ ] `geos = params.pop("geos")` で Geometry/LazyGeometry 群を取り出し、それ以外のパラメータは通常どおり runtime 経由で解決する。
  - [ ] `effects.registry.get_effect(name)` でエフェクト関数を取得し、`impl = getattr(fn, "__effect_impl__", fn)` により実装を取り出す。
  - [ ] `impl(geos=geos, **resolved)` をそのまま呼び出し、戻り値（たとえば `list[Geometry]`）をそのまま返す。
  - [ ] このモードでは `_steps` への追加は一切行わない（パイプライン構築には関与しない）。
- [ ] 通常モード（`"geos" not in params`）では、既存の挙動（GUI 連携 → `_params_signature` → `_steps.append((name, params_tuple))`）を維持する。

3. `PipelineBuilder.culling` 定義の削除

- [ ] `src/api/effects.py` から `PipelineBuilder.culling` メソッド定義を削除する。
- [ ] この削除によりコンパイルエラーが出ないことを確認する（`rg "PipelineBuilder.culling" -n` で他参照が無いことも含め確認）。
- [ ] `__all__` や docstring 等に culling 固有の記述が残っていないことを確認し、必要なら併せて削除する。

4. `effects.culling` 側との整合確認

- [ ] `effects.culling.culling` が `geos: Sequence[Any]` を引数名 `geos` で受け取っていることを確認する（引数名が変わると即時実行モードが壊れるため）。
- [ ] `_normalize_geos` が `LazyGeometry` や `ObjectRef` を含んだ入力を正規化していることを再確認し、`api` 側で正規化しなくても安全であることを保証する。
- [ ] self-culling や face-aware occluder のロジックが `culling` の中に完結しており、`E.pipeline.culling(geos=...)` 経由でも同じ結果になることをテストで確認する。

5. テスト/品質確認

- [ ] 既存の culling 関連テスト（`tests/effects/test_culling_hidden_line*.py`）は変更不要か確認し、必要であれば `effects.culling` を直接呼ぶ形に調整する。
- [ ] `E.pipeline.culling(geos=...)` の UX を検証する新規/更新テストを `tests/api/test_effects_culling_facade.py` のようなファイルに用意する:
  - `E.pipeline.culling(geos=[g1, g2])` の結果が `effects.culling.culling(geos=[g1, g2])` の結果と同一であること。
  - `E.pipeline.rotate(...).culling(geos=[...])` のような組み合わせで、`rotate` はパイプラインとして登録され、`culling` は即時実行として動く（想定どおりか）こと。
- [ ] `src/api/effects.py` に対して `ruff/black/isort/mypy` を実行し、警告/エラーが出ないことを確認する。

6. docs/スケッチの整合性更新

- [ ] `docs/plans/2025-11-15_effects_culling_hidden_line*.md` に記載されている `E.pipeline.culling(geos=[...])` の説明が、新挙動（geos 付き呼び出しで即時実行）と矛盾していないか確認し、必要に応じて「geos を指定した場合はパイプライン登録ではなく即時実行になる」旨を追記する。
- [ ] `sketch/251115.py` など、culling を使う代表的なスケッチで `E.pipeline.culling(geos=[...])` が引き続き動作することを確認し、必要に応じてコメントに新仕様を反映する。

補足/メモ

- この案では、`E.pipeline.<effect>(geos=[...])` という呼び出しパターンが「汎用の即時エフェクト実行モード」として確立されるため、将来的に別の `geos` ベースエフェクトを追加する場合にも同じ UX を再利用できる。
- 一方で、「geos を受け取るがパイプラインステップとして登録したい」ような特殊なニーズが出てきた場合は別途検討が必要（現時点では culling 専用 UX を主対象とする）。

この計画に従えば、`E.pipeline.culling(geos=[...])` という UX を維持したまま、`src/api/effects.py` から culling 専用メソッドを削除し、culling の実装と責務を `effects.culling` 側へ集約できる。***

