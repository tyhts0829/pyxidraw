# PyxiDraw3 設定パラメータ分析レポート

## 概要

このレポートは、PyxiDraw3プロジェクト内のハードコードされた設定値を調査し、`config.yaml`に統合すべきパラメータを特定したものです。現在の設定ファイルは主にMIDIデバイスとUI要素に限定されており、多くの重要な設定値がソースコード内に散在しています。

## 現在のconfig.yaml構成

現在の`config.yaml`には以下のセクションが定義されています：

- **canvas**: 背景色設定
- **canvas_controller**: FPS設定
- **status_manager**: UI表示設定（フォント、色、位置）
- **midi_devices**: MIDIデバイス設定（5種類のコントローラー対応）

## 新規追加推奨パラメータ

### 🔴 高優先度（即座に設定化すべき）

#### 1. レンダリング設定 (`rendering`)

```yaml
rendering:
  default_render_scale: 4        # api/runner.py:28 - デフォルトレンダリング倍率
  default_canvas_size: "A5"      # api/runner.py:27 - デフォルトキャンバスサイズ
  default_fps: 60               # api/runner.py:29 - デフォルトFPS
  background_color: [1.0, 1.0, 1.0, 1.0]  # api/runner.py:30 - 背景色
  buffer_initial_size: 209715200  # engine/render/line_mesh.py:16 - 初期バッファサイズ(200MB)
  primitive_restart_index: 0xFFFFFFFF  # engine/render/renderer.py:44
```

**理由**: パフォーマンス調整とハードウェア依存の最適化に必須

#### 2. パフォーマンス設定 (`performance`)

```yaml
performance:
  default_workers: 4            # api/runner.py:31 - デフォルトワーカー数
  worker_timeout: 1.0           # engine/pipeline/worker.py:80 - ワーカー終了待機時間
  task_queue_multiplier: 2      # engine/pipeline/worker.py:53 - キューサイズ計算係数
```

**理由**: ハードウェア性能に応じた調整が必要

#### 3. 形状生成パラメータ (`shapes`)

```yaml
shapes:
  sphere:
    default_size: [80, 80, 80]    # main.py:37, simple.py:21
    subdivision_limits:
      min: 0                      # shapes/sphere.py:334
      max: 5                      # shapes/sphere.py:335
    segment_base_multiplier: 8    # shapes/sphere.py:25
    zigzag_point_multiplier: 200  # shapes/sphere.py:109
    golden_angle: 9.42477796076938  # shapes/sphere.py:113 (π * (3 - √5))
  
  polygon:
    default_size: [60, 60, 60]    # main.py:39
    sides_limits:
      min: 3                      # shapes/polygon.py:50
      max: 100                    # shapes/polygon.py:50
    nonlinear_mapping_coefficient: 100.0  # shapes/polygon.py:72
  
  grid:
    default_size: [40, 40, 40]    # main.py:41
    division_multiplier: 10       # main.py:41
    division_offset: 5           # main.py:41
  
  torus:
    default_size: [30, 30, 30]    # main.py:76
    rotation_speed: [30, 20]      # main.py:76 - x,y軸回転速度係数
```

**理由**: 形状品質とパフォーマンスのバランス調整

### 🟡 中優先度（段階的に設定化）

#### 4. エフェクト設定 (`effects`)

```yaml
effects:
  noise:
    default_frequency: [0.5, 0.5, 0.5]  # effects/noise.py:164
    intensity_multiplier: 10             # effects/noise.py:133
    time_offset_coefficient: 0.01        # effects/noise.py:134
    time_base_offset: 1000.0            # effects/noise.py:134
    component_separation_offsets: [100.0, 200.0]  # effects/noise.py:115-116
    frequency_adjustment: 0.1            # effects/noise.py:141
  
  wave:
    default_amplitude: 10.0              # main.py:25
    default_frequency: 0.1               # main.py:25
  
  swirl:
    angle_coefficient: 0.01              # main.py:16
```

**理由**: エフェクトの効果調整とクリエイティブな表現の幅を広げる

#### 5. UI拡張設定 (`ui`)

```yaml
ui:
  overlay:
    default_font_size: 8         # engine/ui/overlay.py:18
    line_spacing: 18             # engine/ui/overlay.py:38
    default_color: [0, 0, 0, 155]  # engine/ui/overlay.py:19
    default_font: "HackGenConsoleNF-Regular"  # engine/ui/overlay.py:26
```

**理由**: ユーザビリティ向上とアクセシビリティ対応

#### 6. ベンチマーク設定 (`benchmark`)

```yaml
benchmark:
  warmup_runs: 5               # benchmarks/core/config.py
  measurement_runs: 20         # benchmarks/core/config.py
  timeout_seconds: 30.0        # benchmarks/core/config.py
  chart_dpi: 150              # benchmarks/core/config.py
  output_directory: "benchmark_results"
```

**理由**: パフォーマンス測定の標準化

### 🟢 低優先度（必要に応じて設定化）

#### 7. 数学定数・アルゴリズム定数

```yaml
constants:
  perlin_noise:
    permutation_table: [151, 160, 137, ...]  # util/constants.py:37-291
    gradient_vectors:  # util/constants.py:295-308
      - [1, 1, 0]
      - [-1, 1, 0]
      # ... (12個の3次元ベクトル)
  
  paper_sizes:  # util/constants.py:15-30
    A4: [210, 297]
    A5: [148, 210]
    # ... 既存のサイズ定義
```

**理由**: アルゴリズム的に固定すべき値だが、研究・実験用途では変更可能性がある

## 設定統合の効果

### メリット

1. **パフォーマンス最適化**: ハードウェア性能に応じた調整が容易
2. **開発効率向上**: 設定変更時のコンパイル不要
3. **ユーザビリティ向上**: 非プログラマーでも設定変更可能
4. **保守性向上**: 設定値の一元管理により不整合を防止
5. **テストの容易性**: 異なる設定での動作確認が簡単

### 実装上の注意点

1. **後方互換性**: 既存のAPIを破壊しない形での実装
2. **バリデーション**: 設定値の妥当性チェック機能
3. **デフォルト値**: 設定ファイル未指定時の適切なフォールバック
4. **型安全性**: YAMLパースエラーへの適切な対処

## 実装優先順位

### Phase 1（即座に実装）
- レンダリング設定
- パフォーマンス設定
- 基本的な形状パラメータ

### Phase 2（次期リリース）
- エフェクト設定
- UI拡張設定
- ベンチマーク設定

### Phase 3（将来的に）
- 数学定数の設定化
- 高度なアルゴリズムパラメータ

## 現在の設定との統合案

現在の`config.yaml`を拡張する形で、新しいセクションを追加することを推奨します。これにより既存の機能を破壊することなく、段階的に設定の外部化を進められます。

```yaml
# 既存セクション（そのまま保持）
canvas:
  background_color: [1.0, 1.0, 1.0, 1.0]

# 新規追加セクション
rendering:
  default_render_scale: 4
  default_fps: 60
  # ...

performance:
  default_workers: 4
  # ...

shapes:
  sphere:
    default_size: [80, 80, 80]
    # ...
```

この分析により、PyxiDraw3の設定管理を大幅に改善し、より柔軟で保守しやすいシステムに発展させることができます。