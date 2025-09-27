# NUMBA 環境変数の統一方針（提案）

目的: effects/shapes における numba トグルの環境変数を統一し、挙動・既定値・優先順位を明確化する。

## 現状の課題（要約）
- 変数名が不統一: `PXD_USE_NUMBA_DASH` と `PYX_USE_NUMBA` が混在。
- 解釈が不統一: 真偽値/文字列の扱い、許容値がモジュールごとに異なる。
- 参照箇所が分散: 各モジュールで個別に `os.environ` を解釈。
- shapes は numba を使うが環境トグルは持たないため、全体ポリシーが伝わらない。

## 設計目標
- 単一ポリシーで「グローバル」と「エフェクト別」の両方を制御。
- 既定は「自動(Auto)」で、導入済みなら有効・未導入ならフォールバック。
- 明確な優先順位と後方互換を用意。
- 実装は軽量・シンプル・副作用最小。

## 提案内容（仕様）
- グローバル変数: `PXD_NUMBA={auto|on|off}`（既定: `auto`）
  - `auto`: numba が導入済みなら有効、未導入なら無効（フォールバック）。
  - `on`: 強制有効（未導入時は警告ログを出しフォールバック）。
  - `off`: 明示無効。
- エフェクト別上書き: `PXD_NUMBA_<EFFECT>={auto|on|off}`（例: `PXD_NUMBA_DASH=off`）
  - 優先順位: エフェクト別 > グローバル > 既定(`auto`)。
  - `<EFFECT>` は大文字スネーク（`DASH`, `COLLAPSE`, …）。
- 受理値（後方互換）
  - 真: `1`, `true`, `on`, `yes`（大文字小文字不問）
  - 偽: `0`, `false`, `off`, `no`
  - 上記は `on/off` にマップ。`auto` も許容。
- 後方互換の別名（非推奨, deprecate）
  - `PXD_USE_NUMBA_DASH` → `PXD_NUMBA_DASH`
  - `PYX_USE_NUMBA` → `PXD_NUMBA`
  - 並存時は新名を優先。非推奨警告を一度だけログ（将来削除）。

## 実装指針（コード変更時の方針のみ。今回は実装しない）
- 共通ヘルパ（例: `util/numba_policy.py`）を追加し、環境値の解釈を一元化。
  - `parse_policy(str|None) -> Literal["auto","on","off"]`
  - `effective_policy(effect: str) -> Literal["auto","on","off"]`
  - `numba_enabled(effect: str, installed: bool) -> bool`（軽量計算）
- import パターンの統一
  - 可能なら「遅延 or try-import + 軽量スタブ」方式に揃える。
  - `numba` 未導入でも import エラーで落ちないよう、`njit` に no-op デコレータ代替を用意。
- ログ
  - `PXD_NUMBA=on` だが未導入 → 一度だけ警告。
  - 非推奨環境変数が使われた → 一度だけ警告。

## ロールアウト計画
1) 共通ヘルパ追加（ドキュメント/テスト含む）
2) `effects/dash.py`, `effects/collapse.py` をヘルパ経由に統一（旧名は互換）
3) shapes はグローバル方針（`PXD_NUMBA`）のみ尊重（個別トグルは無し）
4) ドキュメント更新（README, architecture.md, docs/numba_env_vars_in_effects_shapes.md）
5) 非推奨名の警告導入 → 1リリース後に削除告知 → 後日削除

## 影響とテスト
- 影響範囲: numba の有効/無効切替、性能経路の選択のみ（出力は同等を維持）。
- テスト（例）
  - `PXD_NUMBA=off` → 強制フォールバック経路が選択される。
  - `PXD_NUMBA=auto` で numba 有無を切替えた環境で期待経路になる。
  - `PXD_NUMBA_DASH=off` がグローバル `on` より優先される。
  - 旧名の変数で同等挙動＋非推奨警告が出る。

## チェックリスト（承認後に実施）
- [ ] 仕様確定（この提案の承認）
- [ ] 共通ヘルパ `util/numba_policy.py` を追加
- [ ] `effects/dash.py` をヘルパ準拠に変更（旧名互換）
- [ ] `effects/collapse.py` をヘルパ準拠に変更（旧名互換）
- [ ] shapes はグローバルのみ尊重するよう整理（必要なら）
- [ ] 旧名→新名の非推奨警告を実装（1回のみ）
- [ ] 単体テストを追加（env 行動, 優先順位, 旧名互換）
- [ ] ドキュメント更新（README/architecture.md/docs 一覧）
- [ ] リリースノートに移行手順と非推奨告知を記載

補足: 実装はシンプルさを最優先（環境値の評価は1回、キャッシュ可）。
