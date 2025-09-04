# ADR 0002: パイプライン仕様（Spec）検証ポリシー

- ステータス: Accepted
- 日付: 2025-09-04

## 背景
パイプラインを `[{name, params}]` 形式で保存/復元するにあたり、無効な名前や不正なパラメータを早期に検出したい。実行フェーズでのエラーよりも、構築/ロード時点の失敗が望ましい。

## 決定
`api/pipeline.validate_spec(spec)` を設け、以下を検証する。
- 構造: list[dict] であり、各 dict は `{"name": str, "params": dict}`
- 名前: `effects.registry.get_effect(name)` が解決できること（未登録なら KeyError）
- 値の型: JSON様（数値/文字列/真偽/None、入れ子の list/dict）
- 余分キー: 対象関数が `**kwargs` を受けない場合、未知キーで TypeError（許可キーはシグネチャから抽出）

`from_spec()` は `validate_spec()` を内部で呼ぶ。

## 影響
構築フェーズで明確な例外を得られる。既存の関数に `**kwargs` があれば後方互換の拡張余地が残る。

## 代替案
実行時にのみ例外を投げる方式はデバッグコストが高く却下。

## 将来拡張
任意の `__param_meta__`（型/範囲/選択肢）を関数側に宣言できる仕組みを導入し、`validate_spec` が尊重する（数値域検証の拡張）。

