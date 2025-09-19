# src モジュール群コードレビュー（2025-03-07）

## 指摘事項

### 重大
- `src/engine/render/renderer.py:130`
  - `geometry.coords.astype(np.float32)` が毎フレーム新規配列を生成しており、dtype が既に `float32` の場合も無駄コピーが発生。
  - 大きな Geometry を連続処理するレンダリング経路でメモリ帯域とアロケーション負荷が増加し、描画スループットを落とす恐れ。
  - `np.asarray(..., dtype=np.float32)` や `astype(np.float32, copy=False)` など、不要コピーを避ける形へ改善が必要。

- `src/engine/ui/parameters/runtime.py:404`, `src/engine/ui/parameters/state.py:137`
  - パラメータメタ未設定時の既定レンジが `0.0〜0.1` に固定されており、GUI からの override を `set_override` が強制クランプ。
  - 100mm スケール等の一般的なパラメータが GUI 上ほぼ操作不能になり、ランタイムの責務（実用的な GUI 制御）が果たせない。
  - メタ必須化、またはパラメータのオーダーを推定するヒューリスティック導入など、既定レンジ決定ロジックの再設計が必要。

- `src/api/effects.py:122`
  - `Pipeline` の内部 LRU キャッシュが既定 `maxsize=None` のまま無制限成長。
  - フレームごとに異なるジオメトリを流すと Geometry が `OrderedDict` に蓄積し続け、長時間実行でメモリリーク的に増大。
  - 合理的な既定上限の導入、あるいは strict モード時に明示設定を要求するなどの対策が必要。

### 中程度
- `src/engine/ui/parameters/runtime.py` 全体
  - ParameterRuntime が「ランタイム活性管理」「シグネチャ解析」「メタキャッシュ」「ベクタ分解」「レンジ決定」まで一極集中。
  - 単一クラスに責務が集中し、変更時の影響範囲が過大で可読性・テスト性が低下。
  - Doc/メタ解析、値変換、ストア同期などに分割し、UI 層との疎結合を保てる構成への再検討を推奨。

## 補足・質問
- レンジ既定値を `0.1` に固定した背景が不明です。仕様化済みの理由や設計意図があれば教えてください。

## 推奨アクション
1. `_geometry_to_vertices_indices` での不要コピー除去とリグレッション確認。
2. ParameterRuntime のレンジ決定ロジック再設計（テスト整備を含む）。
3. Pipeline キャッシュの既定上限/設定フローを追加し、長時間実行シナリオでのメモリ挙動を検証。
