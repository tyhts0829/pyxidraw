ESC 閉時の GLFW 65537/セグフォ 原因確定チェックリスト

目的
- ESC でメインウィンドウを閉じた際に稀に発生する「GLFW Error 65537: The GLFW library is not initialized」およびセグメンテーションフォルトの根因を、コード変更なしで切り分け・確定する。
- 想定仮説: (A) Dear PyGui/GLFW の終了レース、(B) ModernGL/GL コンテキスト解放順、(C) multiprocessing の後処理影響（副作用）。

前提
- 仮想環境が有効化済みで `python main.py` が動作すること。
- すべてのコマンドはリポジトリルートで実行。
- 失敗時の詳細把握のため `faulthandler` を有効化する（`-X faulthandler`）。

ベースライン再現（ログ採取）
- [ ] `python -X faulthandler main.py` を起動し、ESC で終了。標準エラー出力を確認。
  - 期待: 問題が再現する場合、「GLFW Error 65537」→ セグフォ → `resource_tracker` 警告の順で出力されやすい。
  - 記録: 末尾 30 行程度を保存（例: `python -X faulthandler main.py 2> logs/baseline.txt`）。

仮説Aの切り分け: パラメータGUI（Dear PyGui）無効
- [ ] パラメータGUIを使わずに同じ描画関数で起動し ESC 終了。コード変更せずにワンライナーで実行:
  ```bash
  python -X faulthandler -c "import sys; sys.path.insert(0,'src'); from main import draw; from api import run; run(draw, canvas_size=(400,400), render_scale=3, use_midi=True, use_parameter_gui=False, workers=6, line_thickness=0.001)"
  ```
  - 観測: ここでクラッシュが消えるなら、DPG/GLFW の終了レースが主因と強く示唆される。
  - 記録: `2> logs/no_gui_workers6.txt`

仮説Cの補助切り分け: multiprocessing の影響有無
- [ ] 上記「GUI無効」構成でワーカーを無効（インライン実行）:
  ```bash
  python -X faulthandler -c "import sys; sys.path.insert(0,'src'); from main import draw; from api import run; run(draw, canvas_size=(400,400), render_scale=3, use_midi=True, use_parameter_gui=False, workers=0, line_thickness=0.001)"
  ```
  - 観測: セグフォ有無は GUI 無効の結果と同じになるはず。`resource_tracker` の警告は消える（mp 不使用のため）。
  - 結論: ここでセグフォが出ない場合、mp は主因でない（副作用として警告を増やしているだけ）。

仮説Aの確証取得: DPG 開いたまま vs 閉じてから
- [ ] DPG 有効・mp 無効で起動（DPG のみ残す）:
  ```bash
  python -X faulthandler -c "import sys; sys.path.insert(0,'src'); from main import draw; from api import run; run(draw, canvas_size=(400,400), render_scale=3, use_midi=True, use_parameter_gui=True, workers=0, line_thickness=0.001)"
  ```
- [ ] 終了フローA: 先に DPG ビューポート（パラメータウィンドウ）を×で閉じ、その後メインを ESC で閉じる。
  - 期待: 安定終了（エラーなし）。
- [ ] 終了フローB: DPG を開いたままメインを ESC で閉じる。
  - 期待: 再現率高く「GLFW 65537」→ セグフォ（＝ unschedule 直後の最後の DPG フレームが destroy 後に走るレース）。
- [ ] 再現性確認のため 5〜10 回繰り返して比率を記録。

仮説Bの切り分け: ModernGL/GL コンテキスト解放順の影響
- [ ] 「GUI無効」構成で、メインを ESC で閉じた場合とウィンドウの×で閉じた場合の両方を試す。
  - 観測: いずれも問題が出ない → GL 解放順は主因でない可能性が高い。
  - 観測: ESC だけで問題が出る → pyglet のイベント順序差異が影響している可能性（ただし DPG が無効なら再現しにくいはず）。

ログ・証跡の取得
- [ ] 各シナリオの標準エラー出力を分けて保存（例: `logs/` ディレクトリ）。
- [ ] 可能なら `PYTHONDEVMODE=1` を併用（`python -X dev -X faulthandler ...`）。
- [ ] macOS で詳細が必要な場合のみ、`ulimit -c unlimited` → 1 回再現 → `lldb -c /cores/core.*` でネイティブバックトレース採取（任意）。

確定条件（このチェックで「原因確定」とする基準）
- [ ] 「GUI無効」でセグフォが消え、「GUI有効かつ DPG 開いたまま ESC」で再現する。
  - 結論: DPG/GLFW の終了レースが主因。恒久対策は「pyglet の unschedule 完了を待ってから DPG destroy」または「_tick のガード強化（destroy フラグで早期 return）＋ destroy 遅延」の実装で解決可。
- [ ] 「GUI無効」でも再現する。
  - 結論: GL コンテキスト解放順 or pyglet 側の終了順序の問題。恒久対策は on_close の順序見直し（フレーム停止→GPU解放→ウィンドウ→app.exit）、および GL 呼び出しの存在チェックの強化。
- [ ] mp の有無でセグフォ有無は変わらず、mp 無効で `resource_tracker` 警告のみ消える。
  - 結論: mp は主因ではない（セグフォによりクリーンアップが途切れる副作用）。

補足
- 上記ワンライナーは `main.py` の `draw()` を再利用しつつフラグだけを切替えるため、コード変更不要で検証可能。
- 収集したログファイル一式を併せて見れば、終了順序とエラー発生箇所の相関が明確になる。

