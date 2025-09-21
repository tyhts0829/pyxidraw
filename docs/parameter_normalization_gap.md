# パラメータ正規化ポリシー — 現状調査とギャップ整理（提案含む）

本メモは、次の理想/方針に対する現状実装・ドキュメントの差分を整理し、最小の変更案と確認事項をまとめる。なお、本方針は「0..1 外を実レンジ値として解釈する仕様は廃止（互換モードなし）」に確定。

- すべての公開パラメータは「0.0〜1.0 の正規化入力」を受け取り、`RangeHint.mapped_min/max/step` で宣言した実レンジへ線形変換される。
- 0.0〜1.0 を超える入力も、超過分を含めて同じ線形変換でスケールされる（オーバースケール許容）。
- CLI/パイプラインを含むすべての経路で、入力値は正規化値として扱う。0.0〜1.0 を超える値も正規化値としてそのまま扱う（オーバースケール）。

## 所感（提案への意見）
- 一貫して「0..1 正規化 → 実レンジ線形写像」を入口に置く方針は賛成。UI/非UI/パイプラインで単位系のねじれを排除できる。
- 「0..1 超をオーバースケール」とするのも賛成。モジュレーションや自動化で上限外へ意図的に振る用途が素直になる。
- 「0..1 外は“実レンジ値”として解釈」は廃止します。二義性（例: 実レンジ[0,10]で 2.0 を正規化2.0か実値2.0か）は解消され、2.0 は常に「正規化2.0（=200%）」として扱います。
- 推奨案（段階移行）
  - 既定は「正規化優先（オーバースケール許容）」に統一。GUI は表示/ドラッグのみクランプ、値は保持（>1 を許容）。

## 現状実装の要点（抜粋）
- 正規化/逆正規化
  - `clamp_normalized()` が正規化値を `min_value..max_value`（通常 0..1）へ強制的にクランプ: `src/engine/ui/parameters/normalization.py:22`
  - `normalize_scalar()` は実レンジ→正規化を行い、最後にクランプ: `src/engine/ui/parameters/normalization.py:35`
  - `denormalize_scalar()` は正規化→実レンジの前にクランプを行うため、オーバースケール不可: `src/engine/ui/parameters/normalization.py:71`
- 値解決（GUI/CLI/パイプライン共通の入口）
  - 現状は「0..1 内なら正規化値」「それ以外は実レンジ値として解釈」して正規化へ変換。ただし最終的にクランプされる: `src/engine/ui/parameters/value_resolver.py:355` 以降（特に `:364`, `:373`, `:375`）。本方針ではこの分岐を撤廃し、常に入力=正規化値として扱う必要がある。
- GUI 側の保持/入力
  - スライダーは常にクランプされた正規化値を使う: `src/engine/ui/parameters/panel.py:100`, `:129`
  - `ParameterStore.set_override()` が値を `RangeHint.min_value..max_value` にクランプして保存（0..1 外の保持不可）: `src/engine/ui/parameters/state.py:145`
- CLI/Spec の検証
  - `__param_meta__` の `min/max/choices` による実レンジ検証。上下限を超える実値は例外: `src/api/effects.py:500`

## ドキュメントの現状
- `architecture.md` には次が記載されている（方針の宣言自体は存在）。
  - 正規化→実レンジ変換の一元化: `architecture.md:63`
  - CLI/パイプラインでの解釈（0..1 は正規化、それ以外は実レンジ）: `architecture.md:64`（本方針では削除が必要）
- 一方で「0..1 超も線形でスケールする」旨は明示されておらず、現実装はクランプ挙動。
- `src/engine/ui/parameters/AGENTS.md` は clamp/normalize/denormalize を明記（クランプ前提）で、オーバースケールに触れていない。

## 不一致一覧（実装/ドキュメント vs 理想）
1) オーバースケール不可（クランプされる）
   - 影響箇所: `denormalize_scalar()` の事前クランプ（`src/engine/ui/parameters/normalization.py:71`）。
   - GUI 入力/値保存/逆変換の各所でクランプされ、正規化値 > 1.0 の保持も不可（`state.py:145`）。
2) CLI/パイプラインの「0..1 外は実レンジ値」分岐が実装されている（廃止対象）。
   - 影響箇所: `value_resolver._normalized_input_from_raw()` の分岐（`value_resolver.py:355` 付近）。
3) Spec 検証は `__param_meta__` の範囲で“受け付け”を制限する（上限外はエラー）ため、ドキュメントの「受け付ける」の文言と運用上の印象が異なる可能性。
   - 影響箇所: `src/api/effects.py:500` 付近。
4) ドキュメントの明確化不足
   - オーバースケールの扱い（許容/禁止）と、GUI 表示のクランプ（表示のみ/値も）を切り分けて記述していない。

## 影響範囲（変更時の考慮）
- 逆変換のクランプ解除は、負値や極端値がエフェクト実装へ到達し得ることを意味する。
  - 既存エフェクトが前提とするドメイン（非負/非ゼロなど）を明記し、必要に応じて早期ガードを追加するか、`RangeHint` に「ハード制限（hard_min/max）」の概念を導入する検討余地。
- GUI のスライダー表示はクランプのままでよいが、値自体は >1.0 を保持できるようにする必要がある（表現は 100% まで、実値は>100% を許容）。
- キャッシュ鍵（`pipeline_key`）は正規化後の値がインプット。>1.0 を許容しても同一性の扱いは変わらない。

## 実装計画（クランプ撤廃と正規化統一：チェックリスト）

決定事項（前提）
- [x] すべての入力は「正規化値」。0..1 外の実レンジ解釈は廃止（互換モードなし）。
- [x] 0..1 外もオーバースケールとして線形変換する。
- [x] GUI スライダーはストロークを 0..1 に固定。加速ドラッグ等は導入しない（シンプル優先）。
- [x] Spec/CLI のレンジ検証（`__param_meta__` の min/max に基づく範囲チェック）は削除し、type/choices のみを維持する。

Phase 1: ドキュメント整備（同期）
- [x] architecture.md を更新（オーバースケール明記、互換記述削除）。
- [x] `src/engine/ui/parameters/AGENTS.md` を更新（clamp は表示上の都合に限定する旨を明記）。
- [x] ルート AGENTS.md に方針を追記（正規化統一・オーバースケール許容）。

Phase 2: 変換プリミティブ（normalization.py）
- [x] `normalize_scalar()` の最終クランプを撤廃。実レンジ→正規化は線形変換のみ（bool=0/1、intはfloat正規化）。
- [x] `denormalize_scalar()` の事前クランプを撤廃。正規化→実レンジは線形変換＋`mapped_step` 量子化＋型変換のみ。
- [x] `clamp_normalized()` は残すが「UI 表示用ユーティリティ」として限定利用に位置付ける（docstring 明記）。
- [ ] 単体テスト: 正規化値 1.2/-0.3 の逆変換がオーバースケールで反映される（float/int/bool）。

Phase 3: 値解決（value_resolver.py）
- [x] `_normalized_input_from_raw()` の分岐撤廃。「常に入力=正規化値」。bool は 0/1 変換、数値は float 化のみ。
- [x] 既定値の正規化計算もクランプせずに線形変換。RangeHint がない場合の既定 0..1 は表示ヒントのみ。
- [x] 単体テスト: >1.0 の正規化入力がそのまま実レンジへ反映されること。

Phase 4: 値ストア（state.py）
- [x] `ParameterStore.set_override()` のクランプを撤廃。正規化値をそのまま保持。
- [x] dump_state などのデバッグ出力は変更不要。
- [x] 単体テスト: override に 1.5 を設定しても保持されること。

Phase 5: UI 層（panel.py 他）
- [x] スライダーのドラッグは 0..1 内に限定（視覚と操作の単純化）。内部保持値はクランプしない設計に合わせ、表示は `clamp_normalized()` で 0..1 に収める。
- [x] 値ラベルは `denormalize_scalar()` を用いて実レンジ値を表示（>実レンジ上限も表示）。
- [ ] 単体テスト: ストア値が 1.2 のとき、バー表示は最大だが実値ラベルは上限超を表示する。

Phase 6: Spec/CLI の検証（外部仕様の検証 API は廃止）
- [x] `min/max` による数値レンジ検証を削除（コード除去）。Spec は正規化値のみを受け取り、範囲は検証しない。
- [x] 実装: `src/api/effects.py` の検証ロジック（validate_spec）を削除。
  - 目印: `min_rule = rules.get("min")`, `max_rule = rules.get("max")`（付近の `zip_longest` を使う比較ループ一式）。
- [x] ドキュメント: 「Spec は正規化値を記述」「`__param_meta__` の `min/max` は UI の mapped レンジであり Spec 検証には用いない」を明記。
- [x] テスト調整: レンジ違反を期待するテストが存在する場合は削除/修正（type/choices は現状維持）。

Phase 7: 仕上げ
- [ ] 関連 docstring/コメント整備（clamp の用途を表示側に限定）。
- [ ] 変更対象の ruff/mypy/pytest を緑化（編集ファイル優先ルール）。

## オープン事項（判断メモ）
- 将来的に `normalized_min/max` をメタへ導入して Spec 側の形式検証を段階的に再導入するか（既定 0..1、オーバースケール前提）。
- GUI スライダー: ストローク固定（0..1）、加速ドラッグなしで確定。外部入力（MIDI/CLI）で >1.0 が入り得るが、UI は表示のみクランプし、内部値は保持する。

## 参考（該当実装箇所へのリンク）
- `src/engine/ui/parameters/normalization.py:22`（clamp）
- `src/engine/ui/parameters/normalization.py:35`（normalize + clamp）
- `src/engine/ui/parameters/normalization.py:61`（denormalize + clamp）
- `src/engine/ui/parameters/value_resolver.py:355`（CLI/GUI/Spec 入口の解釈）
- `src/engine/ui/parameters/panel.py:100`（GUI 側のクランプ）
- `src/engine/ui/parameters/state.py:145`（Store 側のクランプ）
- `src/api/effects.py:500`（Spec 範囲検証）
- `architecture.md:63`（正規化→実レンジの宣言）
- `architecture.md:64`（CLI/Spec の解釈）

---
このチェックリストで進めて問題ないかご確認ください。了承後、段階的に実装を進め、各項目の完了状況を本ファイルに反映します。
