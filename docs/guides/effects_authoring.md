# エフェクト作成ガイド（関数API + param_meta）

本プロジェクトのエフェクトは「関数（純粋）→レジストリ登録→パイプライン組立」という流れで使用します。

- 署名: `def effect_name(g: Geometry, *, ...params) -> Geometry`
- 登録: `@effects.registry.effect()`（関数名が登録名）
- 原則: 入力/出力は Geometry、内部で破壊しない（新しい Geometry を返す）

## 命名・パラメータ指針（2025-09 完全切替）
- 動詞の命令形: `translate/scale/rotate/fill/repeat/offset/displace/...`
- 角度: `angles_rad`（ラジアンのみ）。`angles_deg`/`rotate(0..1)` は使用しない。
- 中心: `pivot` のみ（`center` は使用しない）。
- 平行移動: `delta: Vec3` のみ（`offset(_x/_y/_z)` は使用しない）。
- 正規化: 0..1 を取る値は「意味のある名前 + 範囲説明」を明記（例: `density: 0..1`）。
- 物理単位: mm 相当は `*_mm`（例: `amplitude_mm`）。

## 例: 基本的なエフェクト
```python
from effects.registry import effect
from engine.core.geometry import Geometry

@effect()
def my_fx(g: Geometry, *, strength: float = 0.5, mode: str = "soft") -> Geometry:
    coords, offsets = g.as_arrays(copy=False)
    # ... 処理して new_coords/new_offsets を作る ...
    return Geometry(new_coords, new_offsets)

# 仕様検証用の param_meta（任意）
my_fx.__param_meta__ = {
  "strength": {"type": "number", "min": 0.0, "max": 1.0},
  "mode": {"type": "string", "choices": ["soft", "hard"]},
}
```

## param_meta の書式
- キー: パラメータ名
- 値: 検証ルール辞書
  - `type`: "number" | "integer" | "string"
  - `min` / `max`: 数値範囲
  - `choices`: 候補列挙

`api.pipeline.validate_spec()` は以下を行います:
- JSON様式チェック（従来どおり）
- シグネチャ照合（未知キー検出、`**kwargs` なし関数に有効）
- param_meta があれば type/範囲/choices の軽量検証

エラー例:
- `spec[0]['params']['density']=1.1 exceeds max 1.0`
- `spec[2]['params']['join']='sharp' must be one of ['mitre','round','bevel']`

## よくあるパターン
- ベクトルを取る場合（Vec3）: param_meta は省略してシグネチャ検証を用いる
- 二系統の入力（旧/新）: 推奨名を優先し、旧名も受理。関数内で優先順に正規化
- 正規化入力: `norm_to_*` は `common.param_utils` を使用

## レジストリと公開名
- `@effect()` は関数名を `snake_case` で公開名として登録（ケバブ/キャメルは自動正規化）
- エイリアスは使用しません（明示名のみ）。

## テストの基本
- Geometry の点数・offsets の個数・恒等/境界（0/1）・極端値で検証
- pipeline からの往復: `to_spec`→`from_spec` で同一結果になるか

以上。テンプレに沿って関数と `__param_meta__` を用意すれば、spec 検証とパイプライン実行に自動で組み込まれます。

## ベストプラクティス（検証の厳格化）
- `**kwargs` は原則非推奨です。未知キー検出（`validate_spec()` による早期失敗）を有効化するため、公開パラメータはシグネチャに明示してください。
- 仕様の一貫性を高めるため、数値の型/範囲や列挙は `__param_meta__` に宣言してください。

## 旧名 → 新名の移行マップ（最終確定）
- 回転: `rotate(0..1)` → `angles_rad`（0..1→2π は呼び出し側で変換）
- 中心: `center` → `pivot`
- 平行移動: `offset/offset_x/offset_y/offset_z` → `delta`
- 塗り: `pattern/angle` → `mode/angle_rad`
- 複製: `n_duplicates/rotate/center` → `count/angles_rad_step/pivot`
- ノイズ: `intensity/frequency/time` → `amplitude_mm/spatial_freq/t_sec`
- バッファ: `join_style/resolution` → `join/segments_per_circle`
