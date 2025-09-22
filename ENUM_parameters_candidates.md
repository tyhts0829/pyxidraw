# enum 候補パラメータ一覧（effects/shapes）

判定方針
- enum として扱う条件は `__param_meta__` に `choices` があること（type の文字列指定有無は不問）
- `choices` が無い `type: "string"` は自由入力テキスト（非 enum）

## Effects（choices あり = enum）
- extrude.center_mode — ['origin', 'auto']（src/effects/extrude.py）
- offset.join — ['mitre', 'round', 'bevel']（src/effects/offset.py）
- twist.axis — ['x', 'y', 'z']（src/effects/twist.py）
- fill.mode — ['lines', 'cross', 'dots']（src/effects/fill.py）

## Shapes（choices あり = enum）
- text.align — ['left', 'center', 'right']（src/shapes/text.py）
- attractor.attractor_type — ['aizawa', 'lorenz', 'rossler', 'three_scroll', 'dejong']（src/shapes/attractor.py）
- polyhedron.polygon_type — ['tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron']（src/shapes/polyhedron.py）

## 参考（自由文字列 = 非 enum）
- text.font — 任意フォント名/パス（動的探索のため列挙不可）（src/shapes/text.py）

備考
- 上記は ripgrep による `choices` と `Literal[...]` 相当の走査結果に基づく現状の網羅です。
- 判定は「choices の有無」で統一するため、今後 enum 化したい引数には `choices` を追記してください。
