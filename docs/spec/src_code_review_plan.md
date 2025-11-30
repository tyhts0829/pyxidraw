# src 配下コードレビュー（辛口）と実装改善計画

## スコープと前提

- 対象: `src/` 配下全体（API 層 / common / engine / effects / shapes / util / palette）。
- 方法: 代表的なファイルを精読し、構造・責務分割・エラーハンドリング・依存関係の観点からレビュー。
- 制約: すべての effects/shapes を一つずつ精読しているわけではなく、代表例（`displace` など）からパターンを推定している。
- ゴール: いきなりコードを書き換えない前提で、「どこが弱いか」と「どう直すか」を実装レベルのチェックリストとして整理する。

---

## 総評（辛口コメント）

- 全体として「設計意図や docstring はかなり丁寧」だが、その一方でフェイルソフト志向と例外握りつぶしが多く、バグ発見性・運用性の面でかなり弱い。
- API/engine/common は念入りに抽象化されているが、UI・runtime・effects 周辺は「守備範囲が広いクラス」と「肥大化したメソッド」が散見され、可読性を犠牲にしている箇所がある。
- レポジトリ規約では「過度に防御的な実装は避ける」としているのに、実際のコードは `try/except Exception: pass` 系が目立ち、規約と実態が噛み合っていない。
- Numba / ModernGL / multiprocessing など重めの依存に直接ベタ依存しており、「optional に落とす」「フォールバックを用意する」といった分離が中途半端な箇所がある。
- モジュール間の責務分割（`api` / `engine` / `effects` / `shapes` / `util`）は概ね良いが、UI・パラメータ周りは 1 モジュールに詰め込みすぎで「読みながら設計を追う」には辛いサイズになっている。
- docstring が非常に厚く、architecture.md と説明が二重化している部分がある割に、逆に「ここは非直感的だからコメントが欲しい」という箇所（例: cache フラグ計算や IBO freeze 周り）はコードだけ読んでも直感的でない。

---

## レイヤ別レビューと改善計画

### 1. API 層（`src/api`）

#### 辛口レビュー

- `api/__init__.py` はシンプルに再エクスポートしており構造は良いが、`__version__` / `__api_version__` がここに直書きされていて、`pyproject.toml` との同期戦略が明文化されていない。将来ズレる危険が高い。
- `api` パッケージ全体として「エイリアスの重複」（`run` / `run_sketch` 等）はあるが、どの名前を推奨するのか（deprecated なのか）がコードから読み取りづらい。
- docstring がやや「宣伝文」寄りで、実際の制約（例: `G` が実体としては `LazyGeometry` を返していることなど）が、使用者視点でどこまで保証されているかがやや曖昧。
- `api/sketch.py` は「何でも屋」状態で、設定解決・ウィンドウ生成・HUD 初期化・ワーカー結線・録画・エクスポート・シグナル処理などを 1 ファイル／1 関数（`run_sketch`）に詰め込んでおり、読み切るのにかなり根気が要る。
- `run_sketch` 内部でローカル関数 `_draw_main` / `_enter_quality_mode` / `_leave_quality_mode` / `_shutdown_parameter_gui` / `_capture_frame` などを多数定義しており、pyglet イベントとの結合も強いため、テストもしづらい構造。
- 例外処理は「とにかく落とさない」方向に倒れていて、`logger.debug` による握りつぶしが多く、「壊れているのに静かに動き続ける」リスクが高い（MIDI 保存失敗・HUD 更新失敗・parameter_manager.shutdown 失敗など）。
- SIGINT/SIGTERM に独自の handler を登録し、さらに `sys.excepthook` を差し替えて KeyboardInterrupt を黙殺するため、「ユーザが CTRL+C しても何も起きない」ような体験になり得る（対話環境では特に違和感が強い）。

#### 改善計画（チェックリスト）

- [ ] `__version__` / `__api_version__` の出自を整理し、`pyproject.toml` との同期方法を architecture/doc に明記する（もしくは単一路線に寄せる）。
- [ ] パブリック API の「推奨名 vs レガシー名」を洗い出し、`__all__` および docstring に整理（必要なら deprecated ポリシーを記述）。
- [ ] `G` / `Geometry` / `LazyGeometry` の関係を、API docstring と stub の双方で一貫したメンタルモデルとして説明し直す（`LazyGeometry` を表に出すか、あくまで実装詳細扱いにするかを決める）。
- [ ] `run_sketch` の責務を整理し、「設定解決＋依存初期化」「ウィンドウ生成＋イベント登録」「ワーカー／HUD／録画などのサブシステム構成」を別ヘルパに分解する（少なくともモジュール内 private 関数に切り出す）。
- [ ] pyglet イベントハンドラ用のローカル関数群（`on_key_press` / `on_close` など）からビジネスロジックを切り離し、テスト可能な小さな関数に寄せる。
- [ ] KeyboardInterrupt 処理ポリシーを見直し、「CTRL+C を黙殺する」のではなく、適切にクリーンアップした上で例外を表に出すか、設定可能にする。
- [ ] `run_sketch` 内部で行っているフェイルソフト系の `try/except` を棚卸しし、エラー種類ごとに「ログレベル／UI へのフィードバック／即時中止」の扱いを決める。

---

### 2. 共通ユーティリティ層（`src/common`）

#### 辛口レビュー

- `common/param_utils.py` はかなり頑張っているが、役割が多い（clamp / 正規化 / hashable 化 / 量子化）ため、一モジュールとしては責務が広すぎる印象。
- 型ヒント方針と実装がややズレている。リポジトリ規約では `dict[str, Any]` など組み込みジェネリック推奨だが、このファイルでは `Tuple` を `typing` から import している。
- `_env_quant_step` は `settings.get()` → env → デフォルトの三段 fallback を取っているが、例外を広く握りつぶす割にログも出さないため、「設定ミスに気付けない」設計になっている。
- `make_hashable_param` は柔軟だが、「hashable にできないオブジェクトの扱い」が `("obj", qualname, id(obj))` とかなり ad-hoc で、プロセスを跨ぐキャッシュで使うと破綻する設計（id 依存）になりやすい。

#### 改善計画（チェックリスト）

- [ ] `param_utils` の責務を「スカラー/ベクトルの 0–1 正規化」と「署名/キャッシュ鍵生成ユーティリティ」に緩やかに分割する（モジュール分割、もしくは内部ヘルパ分割）。
- [ ] `Tuple` などの `typing` ベース型を組み込みジェネリックへ置き換え、リポジトリの型ヒント規約と揃える。
- [ ] `_env_quant_step` で `common.settings` 読み込み失敗時に debug ログ程度は出すようにし、「なぜ 1e-6 になっているか」が追いやすい状態にする。
- [ ] `make_hashable_param` の id 依存フォールバックは、「プロセスを跨ぐキャッシュには使用しない」「あくまで in-proc LRU 向け」といった制約を docstring に明記するか、もしくは使用側で制約を課す。

---

### 3. Geometry コア（`src/engine/core/geometry.py`）

#### 辛口レビュー

- `Geometry` 自体の概念と実装はかなり綺麗だが、docstring が厚く、architecture.md と説明が二重化している印象。説明密度が高いため、新規読者にはむしろ「読むべき場所」が増えている。
- `translate` / `scale` / `rotate` で空ジオメトリのときにも毎回 `self.coords.copy()` / `self.offsets.copy()` を行っており、「常に新インスタンスを返す」を守るためとはいえ、ゼロサイズでの alloc がやや過剰に感じる。
- 変換系メソッドがすべて「新しい Geometry を返す」ポリシーなのに、`coords` / `offsets` の属性自体は公開されているため、「外から直接いじれるのか？」という疑問が残る（immutable のつもりか mutable のつもりかが曖昧）。
- `from_lines` は柔軟だが、「ありえる不正ケース」（空ライン・長さ 1 のライン・不揃いな 2D/3D 混在）の扱いがドキュメントから一望しづらい。実装を読めば分かるが、docstring からは読み取りにくい。

#### 改善計画（チェックリスト）

- [ ] architecture.md 側と `Geometry` の docstring を見直し、「データモデルの詳細説明」はどちらに寄せるかを決めて重複を減らす。
- [ ] 空ジオメトリに対する `translate/scale/rotate/concat` の挙動を整理し、本当に毎回新インスタンス＋コピーが必要か（もしくは `Geometry.empty()` のような共有インスタンスを使うか）方針を決める。
- [ ] `coords` / `offsets` の公開方針を明確化し、「原則として `as_arrays(copy=...)` 経由でアクセスすべき」といったルールを docstring または型レベルで示す。
- [ ] `from_lines` の受け入れ仕様とエラー条件を、docstring に網羅的に書き直し、「試してみないとわからない」状態を減らす。

---

### 4. ランタイム／ワーカ層（`src/engine/runtime/worker.py` 他）

#### 辛口レビュー

- `WorkerPool` は「ワーカープロセス管理」「タスク生成」「インライン実行」「CC/parameter snapshot 適用」「メトリクス差分算出」まで抱えており、クラス単位の責務がかなり広い。
- `_normalize_to_layers` は戻り値のバリエーションを頑張って扱っているが、`Sequence` 判定の使い方が微妙で、`np.ndarray` などが紛れ込んだ場合の挙動が読みづらい（将来の拡張に対して脆い）。
- `_apply_layer_overrides` は `overrides` が `Mapping` でない場合に黙って無視するが、ここでもエラーを握りつぶす方向に倒れており、「パラメータ GUI 側のバグに気付きにくい」。
- `_execute_draw_to_packet` は例外をキャッチして `WorkerTaskError` に包むのは良いが、例外種類をラップしてしまうことで、呼び出し側から本来の例外タイプが失われており、デバッグ体験がやや悪い。
- ログの多くが `logger.debug` に落ちており、「本番で発生したがユーザが気付きたいレベルのエラー」が `debug` のまま埋もれる可能性がある。

#### 改善計画（チェックリスト）

- [ ] `WorkerPool` の責務を整理し、少なくとも「ワーカー管理（プロセス/キュー）」と「draw 実行＋レイヤ正規化＋メトリクス」は別のヘルパ（モジュール or クラス）に分割する。
- [ ] `_normalize_to_layers` のインターフェースを固め、`Sequence[Layer | Geometry | LazyGeometry]` 以外を渡された場合は早めに明示エラーを出すようにして、暗黙挙動を減らす。
- [ ] `_apply_layer_overrides` の `overrides` 型チェックを明示化し、Mapping でない場合はログ＋早期 return するなど、「無言で無視する」ケースを減らす。
- [ ] `_execute_draw_to_packet` で `WorkerTaskError` に元の例外タイプや traceback 情報を含める手段（属性 or message）を整理し、ログだけではなく呼び出し側からも原因が追えるようにする。
- [ ] ログレベルを見直し、「パラメータ反映の失敗」「メトリクス取得の失敗」などは `info` or `warning` に引き上げるか、設定でレベルを切り替えられるようにする。

---

### 5. Parameter GUI / UI 層（`src/engine/ui/parameters/*`）

#### 辛口レビュー

- `ParameterManager.initialize` は「設定読み込み → layout 構成 → window config → theme config → runtime 起動 → palette/hud/layer descriptor 登録 → override 復元 → GUI 起動」まで 1 メソッドに詰め込んでおり、ほぼ「長大なスクリプト」になっている。
- `try/except Exception: pass` が目立ち、特に palette / HUD 関連でエラーが起きてもログなしで握りつぶされるため、「見た目が壊れても気付きづらい」。
- config のバリデーションが分散しており、`load_config()` で型を保証するのではなく、各所で `isinstance(dict)` チェックや `try/except` による補正を行っているため、読み手のコストが高い。
- `_register_layer_descriptors` と `_register_palette_descriptors` は「ParameterDescriptor を大量に組み立てるロジック」と「store への登録」が混ざっており、テストしづらく読みづらい。

#### 改善計画（チェックリスト）

- [ ] `ParameterManager.initialize` を段階的に分解し、「config 読み込みと整形」「Runtime 初期化」「Runner 用パラメータ登録」「Palette 用パラメータ登録」「HUD 用パラメータ登録」「オーバーライド復元と GUI 起動」に関する private メソッドへ切り出す。
- [ ] `try/except Exception: pass` を全面的に見直し、少なくとも debug ログは残す、もしくはエラーをユーザにフィードバックできるようにする。
- [ ] config の型・値バリデーションを `util.utils.load_config` 側、または専用の「設定正規化関数」に寄せて、UI 層のロジックを「正しい形になっている前提」で書けるようにする。
- [ ] ParameterDescriptor のインスタンス生成をヘルパ関数（例: `make_style_descriptor(...)`）に寄せ、GUI パラメータ追加ロジックの重複を減らす。
- [ ] HUD / palette パラメータと runner パラメータの境界（id 命名規則など）を簡潔に仕様化し、ドキュメントとコードを同期する。

---

### 6. レンダリング層（`src/engine/render/renderer.py` 他）

#### 辛口レビュー

- `LineRenderer` は「SwapBuffer の受信」「ModernGL の初期化」「IBO freeze の統計」「HUD 連携用のメトリクス」「レイヤ描画」「カラー・太さの粘着状態管理」など多機能で、1 クラスとしては情報量が多い。
- ModernGL 周りのエラー処理が `debug` ログに寄っており、GPU サイドの問題が起きたときにユーザ視点では単に「何も描画されない」状態になりかねない。
- GPU リソースライフサイクル（`release`）と SwapBuffer の関係がコードからは完全には読み取れず、「いつ何を解放して良いか」が明文化されていない。
- `tick` / `draw` の責務分離自体は良いが、`frame.has_layers` かどうかで挙動が大きく変わり、さらに `_last_layers_snapshot` / `_sticky_color` 等の状態フラグに依存しているため、状態マシンとしての把握が難しい。

#### 改善計画（チェックリスト）

- [ ] `LineRenderer` の責務を整理し、「SwapBuffer からのフレーム取得」と「実際の描画処理（layers or geometry）」を別クラス or ヘルパ関数へ分割する。
- [ ] ModernGL エラー時の扱いを決め、最低限 `warning` レベルでログしつつ、必要なら「レンダリング無効化」といった状態遷移を設計する。
- [ ] `_last_layers_snapshot` / `_sticky_color` などの状態変数について、どのタイミングでどう遷移するかをコメント or docstring で明文化し、状態マシンとして読みやすくする。
- [ ] GPU リソース解放ポリシー（アプリ終了時 / コンテキスト再生成時）を architecture.md or docs に追記し、`release` の呼び出し責務を明確化する。

---

### 7. Effects 層（`src/effects/*`）

#### 辛口レビュー

- `effects/__init__.py` がすべての effect モジュールを side-effect import しており、起動時の import コストが高くなりやすい。Numba や heavy な数値処理を含むモジュールまで一括でロードされている。
- 代表例 `displace.py` のように、Numba `@njit` 関数群がファイル先頭に密集しており、「パラメータ解決ロジック」と「数値カーネル」が一続きで並んでいて読みづらい。
- Numba の存在を前提にしており、「Numba が無い環境でどう振る舞うか」のポリシーが明示されていない（requirements を見ないとわからない）。
- docstring の粒度が effect ごとにまちまちで、Parameter GUI 用の `__param_meta__` と docstring の対応が 1:1 で説明されていない箇所がある。

#### 改善計画（チェックリスト）

- [ ] effect モジュールの import 方式を見直し、少なくとも heavy なエフェクト（Numba 依存など）は遅延ロード（registry 経由での import）に寄せる。
- [ ] 各 effect モジュール内で「純粋な数値カーネル（@njit）」と「Geometry / Parameter GUI 連携ロジック」を別セクションに分け、読みやすい構造にする（ヘッダコメント・関数順序など）。
- [ ] Numba が存在しない環境での挙動（ImportError 時にどうするか）を決め、最低限 graceful なエラー or フォールバックを設ける。
- [ ] `__param_meta__` に記載された Range/step と docstring の説明を同期し、「GUI 上のレンジ」と「実際に意味を持つ範囲」がズレないようにする。

---

### 8. Shapes 層（`src/shapes/*`）

#### 辛口レビュー

- `shapes/registry.py` 自体はシンプルで良いが、`ShapeFn` を `Callable[..., Any]` にしているため、戻り値が「Geometry」なのか「生ポリライン列」なのかが型レベルでは不透明。
- 各 shape モジュール（grid / sphere / text など）がどのくらい「純粋な幾何生成」に徹しているかはモジュールごとにバラつきがありそうで、テスト/再利用の観点から規約が欲しい。
- Parameter GUI との連携（RangeHint や `__param_meta__`）の整備レベルが effect 以上にまちまちで、GUI 側から見た「shape パラメータの質」は不均一な可能性が高い。

#### 改善計画（チェックリスト）

- [ ] `ShapeFn` の戻り値ポリシー（`Geometry` か `LineLike` か）を決め、必要に応じて型エイリアスと docstring で統一する。
- [ ] 代表的な shapes（grid/sphere/text 等）について、「入出力と不変条件」を短い docstring で明文化し、設計のばらつきを減らす。
- [ ] Parameter GUI と連携したい shape について、`__param_meta__` の付与・整理を行い、「GUI 上で調整できるもの / できないもの」を明確にする。

---

### 9. util / palette / その他

#### 辛口レビュー

- `util/utils.py` の `_find_project_root` は「.git, pyproject.toml, configs のどれかがあるディレクトリをルートと見なす」という heuristic だが、これが将来のレイアウト変更にどこまで耐えられるか不透明。失敗時の fallback も静かすぎる。
- `util/geom3d_ops.py` は numba 依存かつ `transform_to_xy_plane` のアルゴリズムがそこそこ重いわりに、使用側から「どのくらいのコストを許容する想定か」が見えづらい。
- palette 周辺は `DefaultColorEngine` などを UI ランタイム中から直接 import しており、依存方向が UI → palette → color engine と深くなっている。色変換エンジン自体は純粋ユーティリティとして切り離してもよさそう。

#### 改善計画（チェックリスト）

- [ ] `_find_project_root` のルールを architecture.md か docs に明文化し、「ソースツリーの前提構造」を固定化する（将来変更が必要になったときに検知しやすくする）。
- [ ] `geom3d_ops` の使用箇所を洗い出し、必要なら「高速パス / 低速パス」のように分けて numba 依存を緩和する。
- [ ] palette エンジンを UI から切り離し、「色変換ユーティリティ」として util/palette いずれかに薄く再配置する設計案を検討する。

---

## 横断的な改善テーマ

#### エラーハンドリングとログ

- [ ] `try/except Exception: pass` を全体から洗い出し、少なくとも debug ログ、できれば warning レベルでの記録に変える。
- [ ] 「フェイルソフトにしたい箇所」と「むしろ早めに落として気付きたい箇所」を整理し、ポリシーを architecture.md に追記する。

#### 責務分割とメソッド長

- [ ] 200 行クラスメソッド級の長いメソッド（特に `ParameterManager.initialize`）を優先的に分割する。
- [ ] クラス単位の責務が膨らんでいる場所（`WorkerPool` / `LineRenderer` / `ParameterManager` など）について、「どの層の concern なのか」を再整理し、小さなヘルパクラスやモジュールへ切り出す。

#### 依存関係の明確化

- [ ] Numba / ModernGL / multiprocess 等の重い依存の扱いを整理し、「必須か optional か」「フォールバックはあるか」を README or docs に明示する。
- [ ] `api` → `engine` → `util` の依存方向は概ね守られているが、UI → palette → util のような逆方向（高層 → 低層）依存がないか一度 dependency graph を洗い出す。

---

## この計画に関する相談ポイント

- [ ] バグ検出性を優先して「フェイルソフト志向」をかなり弱める（例: Parameter GUI 周りの例外をログ＋ユーザ通知に寄せる）方針に切り替えても良いか。
- [ ] Numba 依存を optional にして「性能よりも環境対応を優先する」モードを追加するかどうか。
- [ ] `Geometry` のイミュータビリティ方針（`coords/offsets` を外部から直接 mutate するのを許容しない）を強めるために、アクセス API を絞るかどうか。
- [ ] パラメータ/GUIまわりを次期リリースでどこまで壊してよいか（ID 命名や layout 設定などの破壊的変更許容度）。

このチェックリストで問題なければ、次のステップとして個別項目ごとにブランチを切り、対象モジュール単位で実装改善を進めつつ、完了した項目から順次チェックを付けていく運用を想定しています。
