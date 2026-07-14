# 探索パラメータ ablation 結果表（SOT-1673）

SOT-1672 で採用した Determinized MCTS 構成（docs/mcts-design.md §7）を基準点に、
各設計パラメータを1要因ずつ動かして勝率への影響を定量化した。Strategy 部門
レポートへの転用を想定し、全行が「構成・試合数・勝率・95%CI」を持つ。

## 方法

- **基準構成（baseline）** — SOT-1672 採用値（docs/mcts-design.md §7）:
  `{"max_root_actions":6, "max_tree_depth":1, "rollout_turns":100,
  "rollout_depth":200, "n_worlds":4, "time_budget_s":0.8,
  "deviate_margin":0.1}`（未指定フィールドは PlannerConfig 既定値:
  `uct_c=1.4`, `rollout="greedy"`, 評価重みは evaluator.py の
  `DEFAULT_WEIGHTS`）。
- **ablation 軸**（one-factor-at-a-time）:
  - 世界サンプル数 N: 1（PIMC相当）/ 2 / **4** / 8
  - PUCT 探索定数 uct_c: 0.35 / 0.7 / **1.4** / 2.8
  - rollout policy: **greedy**（GreedyAgent両側）/ heuristic（型ティア）/ random
  - 評価関数重み: **default** / prize_only（プライズ以外の全特徴を0）/
    material2x（プライズ以外の重みを2倍）
  （太字が baseline の値）
- **対戦相手軸** — 単一相手への過適合（被搾取性）を避けるため複数軸で計測
  （ByteRL, arXiv:2404.16689 の指摘に倣う）: Random / Greedy /
  MCTS(baseline構成)。
- **計測** — 各セル 100 試合（10シャード×10試合、先後は1試合ごとに交替）、
  勝率は引き分け除外、95%CI は Wilson score 区間。シャードは
  `eval/bench.py` の独立プロセスとして並列実行し、
  `eval/aggregate_shards.py` でプール。
- **シード** — セルシードは `--seed 94000` + CRC32(構成名|相手) から決定的に
  導出。単独再実行でも全掃引でも同じシード列になる。再現性はエージェント側
  乱数のみ（エンジン内部乱数は非注入 = ASSUMPTIONS.md A-9。同一シードでも
  勝敗数は試行間で若干揺れる）。
- **健全性カウンタ** — 全セルで engine rejects / agent exceptions /
  budget violations / planner fallbacks / BaseAgent fallbacks の合計
  （表の最終列）が 0 であることを確認する。

## 再現コマンド

全掃引（リポジトリルートから; 24コアで約1時間）:

```bash
venv/bin/python eval/ablation.py --n 100 --shards 10 --seed 94000
```

各行（セル）は1行で単独再実行できる（例: 世界数N=1 vs Greedy）:

```bash
venv/bin/python eval/ablation.py --n 100 --shards 10 --seed 94000 --only-config n_worlds=1 --only-opponent greedy --force
```

表の生成: `venv/bin/python eval/ablation.py --report`
（各セルの再現コマンドは `eval/results/ablation/<セル>.json` の
`cell.repro` にも記録される。）

## 結果表

<!-- RESULTS_TABLE -->

## 考察

<!-- DISCUSSION -->

## 最良構成と選定根拠

<!-- BEST_CONFIG -->

## 相性の循環（Nash averaging / mElo）についての判断

<!-- NASH_JUDGMENT -->
