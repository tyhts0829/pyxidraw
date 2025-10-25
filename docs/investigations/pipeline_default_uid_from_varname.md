パイプラインの既定グループ名を「オブジェクト名（変数名）」にできるか検証（報告）

要望
- `.label()` 未指定時、現在はフレーム内出現順で `p0`, `p1`, ... を既定グループ名としている。
- これを、コード中のパイプライン変数名（例: `e_pipe1`, `e_pipe2`）を既定にできないか。

現状の仕組み（関連コード）
- 既定 UID はフレーム開始ごとに `ParameterRuntime.next_pipeline_uid()` が割り振る `p0`, `p1`, ...。
  - src/api/effects.py: `Pipeline.__call__` で `self._pipeline_uid` が空なら `runtime.next_pipeline_uid()` を取得して使用。
  - src/engine/ui/parameters/runtime.py: `begin_frame()` でカウンタを 0 に戻し、`next_pipeline_uid()` が `p{n}` を返す。
- GUI のグルーピングは `ParameterDescriptor.category` に `context.pipeline`（= 上記 UID）が入る仕様。
  - src/engine/ui/parameters/value_resolver.py: `category = context.pipeline or context.scope`

結論（簡潔）
- 変数名（例: `e_pipe1`）を実行時に自動で取得して既定グループ名にするのは、シンプルな変更では困難。

理由（技術的背景）
- Python のオブジェクト自体は「どの変数名に束縛されているか」を保持していない。
- 実行時に変数名を推定するには、以下のような脆い手法が必要になる:
  - `inspect` で呼び出し元フレームのソース行を取り出し、`pipe(base)` のような行をパースして左辺（または呼び出し名）から推測。
  - ただし複数行・インライン式・一時変数・演算混在・ラムダ/内包表記・同一オブジェクトの多重参照等で破綻しやすい。
  - spawn（マルチプロセス）環境ではソースが取得できないケースがあり、実装が不安定化する。
- 以上より、簡潔な変更で堅牢に実現するのは難しい。

代替案（シンプルに可能な選択肢）
1) 既定を「フレーム非依存の安定 UID（ビルド時採番）」に変更
   - `PipelineBuilder._ensure_pipeline()` で `.label()` 未指定時にプロセス内連番で `pl0`, `pl1`, ... を一度だけ付与。
   - `Pipeline.__call__` では `self._pipeline_uid` のみ使用（`runtime.next_pipeline_uid()` へのフォールバックを廃止）。
   - 効果: フレーム順依存が無くなり、`p0/p1` が毎フレーム入れ替わる問題を解消。実装は小変更で済む。

2) 明示ラベルの利用をガイド（現行仕様の活用）
   - `E.pipeline.label("e_pipe1") ...` のように開発者が明示指定。
   - コードの意図が最も明確で、GUI/永続ともに安定。

3)（参考）`id(self)` 等からハッシュ短縮名を生成
   - `pl-7f9a` のような見た目の識別子を自動生成。フレーム間で安定だが、再起動で再生成される。
   - 可読性は落ちるため推奨は 1) か 2)。

推奨
- 変数名の自動推定は行わず、1) の「ビルド時に安定 UID を自動採番」へ切り替えるのが最小・堅牢。
- さらに必要に応じて 2) の `.label()` を併用して人間可読の名前にする。

備考
- 本変更（1）を行う場合、`ParameterRuntime.next_pipeline_uid()` 経由の `p0/p1` は不要になり、カテゴリや ID は `effect@pl0.*` のように安定化する（互換性に注意）。
- 現在の仕様でも `.label("e_pipe1")` を付ければ即座に所望のグループ名にできる。

以上。

