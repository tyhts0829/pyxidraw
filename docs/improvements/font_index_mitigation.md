**目的**
- `.ttc` フォントに対して `font_index` が範囲外となり、`TTLibFileIsCollectionError: specify a font number between 0 and N (inclusive)` で描画が失敗する事象を解消する。

**背景**
- 現状の挙動:
  - `.ttc` で `TTFont(path, fontNumber=idx)` を直接開く実装。`idx` が収容フェイス数を超えると例外が発生する。
  - 参照: `src/shapes/text.py:151`, `src/shapes/text.py:161`, `src/shapes/text.py:178`
- GUI の `font_index` は静的に 0..32（ヒント）だが、実フォントのフェイス数と同期しないため、上限超過が起こり得る。

**要件**
- 範囲外の `font_index` 指定でもクラッシュさせない。
- 依存/環境に左右されずに安全に動作（fontTools が無い最小環境でも退避可能）。
- 既存の API 互換性を保ち、見た目が大きく変わらないようにする（警告は最小限）。

**対策候補**
- A. 事前に `.ttc` のフェイス数を取得してクランプ
  - `fontTools.ttLib.TTCollection` を用いてフェイス数を把握し、`idx = min(max(idx,0), faces-1)` に丸める。
  - 長所: 例外経路に入らない。範囲が明確。
  - 短所: `TTCollection` の import がダミー依存（`tools/dummy_deps.py`）に無い。import 失敗時のフォールバックが必要。
- B. 例外を捕捉してリトライ
  - `.ttc` を `TTFont(..., fontNumber=idx)` で開き、`TTLibFileIsCollectionError` を捕捉。メッセージから上限 N を推定してクランプ、または `idx=0` で再試行。
  - 長所: 追加依存なしでも機能。実フォントに依存しない。
  - 短所: メッセージ仕様に依存（将来変更リスク）。
- C. GUI 側の上限を動的化
  - 実際に選定されたフォントファイル（特に `.ttc`）に応じて `font_index` の最大値を GUI に反映。
  - 長所: ユーザー体験が良い。範囲外入力を未然に防ぐ。
  - 短所: 現状 `font` は GUI 非表示（`choices: []`）のため、選定フォントの確定タイミング/通知が必要。実装コストが高め。

**提案（実装方針）**
- 優先: A + B のハイブリッドで `TextRenderer.get_font()` 内を堅牢化。
  - まず `.ttc` 検出時に可能なら `TTCollection` でフェイス数を取得してクランプ（A）。
  - それでも失敗・未知環境では `TTLibFileIsCollectionError` を捕捉し、`idx=0`（または推定上限でクランプ）でリトライ（B）。
  - 例外を外に漏らさず、静かに安全側に寄せる（`logger.debug` に留める）。
- 次点（任意・後続タスク）: C の GUI 動的上限化を検討（別PR）。

**実装詳細（最小変更）**
- 対象: `src/shapes/text.py`
- 変更点:
  - `TextRenderer.get_font()` の `.ttc` 分岐で、以下を追加:
    - `try: from fontTools.ttLib import TTCollection`。成功時、`faces = len(TTCollection(path).fonts)` を取得し `idx = min(max(idx,0), faces-1)` にクランプ。
    - `except ImportError: pass`（ダミー環境ではそのまま）。
    - `TTFont(..., fontNumber=idx)` 呼び出しを `try/except` で保護し、`TTLibFileIsCollectionError` を捕捉して `idx=0`（または可能ならメッセージから上限を抽出しクランプ）で再試行。
  - 名前部分一致探索で見つけた `.ttc` に対しても同様の処理を適用。
  - フォールバックフォント（既定）は現状通り index=0 固定。
- ログ方針:
  - クランプや再試行は `logger.debug`。ユーザー向け警告は原則出さない（UI での操作を阻害しない）。

**テスト方針**
- 単体（モック中心）
  - `fontTools.ttLib.TTFont` をモックし、指定 index で `TTLibFileIsCollectionError` を投げるようにして、再試行で index=0 に丸めることを検証。
  - `.ttc` パス検出時に `TTCollection` が存在する場合、フェイス数にクランプされることを検証。
  - いずれも `tests` で monkeypatch（`pytest`）を用い、実フォント非依存にする。
- 既存動作の回帰確認
  - `.ttf/.otf` では `font_index` が無視され従来通り動作することを確認。

**移行影響**
- 既存の API は不変。範囲外指定時に落ちていたケースが自動的にクランプされるのみ。

**作業チェックリスト**
- [ ] `TextRenderer.get_font()` の `.ttc` ルートでフェイス数クランプ（`TTCollection` 利用、失敗時はスキップ）を追加
- [ ] `.ttc` オープン（パス指定/探索双方）を `try/except` で保護し、`TTLibFileIsCollectionError` 捕捉時に安全側に丸めて再試行
- [ ] `logger.debug` によるクランプ/再試行の記録を追加
- [ ] 単体テスト（モック）追加：範囲外指定 → クランプされること
- [ ] ドキュメント更新：`font_index` の意味とクランプ挙動（`docs/shapes.md` または `architecture.md` の追補）

**確認したい点（要回答）**
- クランプ戦略: 上限超過時は「最大値に丸める」で良いか、それとも「常に 0 に落とす」か。
- ログレベル: 範囲外 → クランプ時に `warning` を出すべきか、`debug` に留めるか。
- GUI: 当面は実装しない前提で良いか（将来の改善候補として別チケット化）。

**将来拡張（任意）**
- `font_face_name`（文字列）指定の追加で `.ttc` のフェイスをインデックスではなく名前で選択。
- Parameter GUI にフォント/フェイス選択コンボを追加し、`font_index` を非表示化。

