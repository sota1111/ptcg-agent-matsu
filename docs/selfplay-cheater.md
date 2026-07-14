# チーターdeterminization自己対戦データ生成（SOT-1678）

Świechowski et al. (arXiv:1808.04794) §III-B1/§IV-C の再現。学習データ生成時
**のみ**真の隠れ情報で determinize した強 MCTS 同士を自己対戦させ、
(拡張盤面ベクトル, 最終勝敗) ペアの JSONL データセットを生成する。
実装: `train/gen_selfplay.py --cheater` + `train/cheater.py`（SOT-1674 の
生成器の拡張。既定の Greedy/Random mix モードは無変更）。

## 真の隠れ状態の取得元

ローカルエンジンの `cg.game.visualize_data()`。select ごとのスナップショット
履歴を返し、最終要素の `current` が現在の完全状態 — 両者の山札（順序付き・
カード ID）、手札、サイドの中身、裏向きバトル場の正体まで全部見える
（2026-07-14 実測: 各ゾーンの枚数が観測の `deckCount`/`handCount`/`prize` と
全時点で一致することを確認）。

決定ごとに `train/cheater.py:true_fills()` が手番プレイヤー視点の
`Fills`（my_deck / my_prize / opp_deck / opp_prize / opp_hand / opp_active）を
真状態から作り、`CheaterMctsAgent` 経由で `MctsPlanner` の determinization に
注入する（`fills_fn` コンストラクタ注入、SOT-1672 の `sample_fills` を置換）。
真状態が exact なので `n_worlds=1` が自然な設定（隠れ情報の分布が消えるため
複数ワールドの平均化は不要）。ゾーン枚数が観測と食い違う場合は fail-loud
（`ValueError`）で、planner は当該決定を greedy prior に degrade して継続する
（`degraded` として統計に出る。実測ランでは 0）。

## 公正性の分離（コンペ提出物はチート不可）

- チーター経路は `train/` のみに存在し、提出アーカイブに**入らない**
  （`scripts/build_submission.sh` は `main.py deck.csv agents cg` のみを梱包）。
- `agents/` は `visualize_data` を一切参照せず、`train` も import しない。
- 対戦用エージェントからフックに到達不能: `make_agent("mcts", ...,
  fills_fn=...)` は `TypeError`（PlannerConfig にそのフィールドが無い）。
  `MctsPlanner` の既定は従来どおり `sample_fills`（情報集合からのサンプル）。
- そもそも提出エージェントには不可能: `visualize_data()` はローカルの
  `battle_start()` ループが握る battle ポインタを読むもので、Kaggle ハーネスは
  エージェントにこれを渡さない。

以上すべて `tests/test_cheater.py`（16 テスト）で担保。

## データ形式とサンプリング

- エンジンターンごとの最初の決定局面を確率 `--sample-p`（論文の p、既定 0.5、
  シード注入済み Rng）でサンプリング。ターン 0（セットアップ）は除外。
- サンプルした 1 状態につき **両者分** の 2 レコードを出力:
  `{"m": 試合, "t": ターン, "who": 視点プレイヤー, "y": whoが勝てば1,
  "h": heuristic予測, "x": 特徴ベクトル}`。
- `x` は既定で SOT-1676 の v2 拡張盤面ベクトル（387 次元）。**手番プレイヤーの
  観測**から `featurize(obs, who)` で計算する: who=手番なら自手札あり
  （`my_turn=1`）、who=非手番なら自手札は隠れ（`my_turn=0`）。これは MCTS 内で
  value 評価が推論時に見る特徴分布（search 観測を root 視点で featurize）と
  一致する。**真状態は探索の determinize にだけ使い、特徴には使わない**
  （train/serve 分布のミスマッチを作らないため）。

## 再現性の範囲

シード（`--seed`）はエージェント乱数・状態サンプリングの全ストリームを固定
する。エンジン内部乱数（シャッフル・コイン）は注入不可（ASSUMPTIONS.md A-9）
なので、実エンジン上の再現性は SOT-1674 と同じ「生成手順の固定」。エンジン
応答を固定した条件では生成はシードの決定的関数であることを
`tests/test_cheater.py::test_same_seed_same_dataset`（scripted エンジン double、
バイト一致）で担保する。

## 実測ラン（2026-07-14）

コマンド 1 行（リポジトリルートで）:

```bash
venv/bin/python train/gen_selfplay.py --cheater --n 200 --seed 61678001 --out train/logs/cheater_shard_1.jsonl
```

| 項目 | 値 |
| --- | --- |
| 試合数 | 200（draws 0 / unfinished 0 / errors 0） |
| サンプル状態数 | 714（p=0.5、平均 ~7.1 状態/試合 ≒ 平均 ~14 ターン/試合） |
| 例数（JSONL 行数） | 1,428（両者分 = 状態 × 2） |
| 所要時間 | 9 分 06 秒（~2.7 秒/試合、シングルプロセス） |
| ファイルサイズ | 2,726,458 bytes（≈2.6 MB、`/train/logs/` は gitignore・再生成可能） |
| MCTS 設定 | SOT-1673 champion 構成、ただし `n_worlds=1`・`time_budget_s=0.1` |
| degraded / fallbacks | 0 / 0（全決定が真状態 determinization で探索できた） |

規模の根拠: 論文は 1 秒/手のチーター MCTS で 20k 試合 → 3.5M サンプル
（クラスタ計算）。ローカル 1 コアでは 1 秒/手 × 20k 試合 ≈ 数週間かかるため、
手あたり予算を 0.1 秒（`--mcts-config '{"time_budget_s": ...}'` で変更可）、
規模を 200 試合（`--n`）に縮小した。この設定の実測スループット（~2.7 秒/試合）
から、SOT-1679/1680 の学習で必要になればシャーディング
（`--seed`/`--out` を変えて並列実行、SOT-1674 と同じ流儀）で線形にスケール
できる（例: 8 シャード × 2,500 試合 ≈ 8 × 1.9 時間）。
