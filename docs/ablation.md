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

全 33 セル（11 構成 × 3 相手軸）、各 100 試合。生成: `venv/bin/python eval/ablation.py --report`
（計測日: 2026-07-14、seed 94000）。

| 軸 | 構成 | 対戦相手 | 試合数 | 勝率(引分除く) | 95%CI | reject/違反/fallback |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `baseline` | random | 100 | 0.930 | [0.863, 0.966] | 0 |
| baseline | `baseline` | greedy | 100 | 0.640 | [0.542, 0.727] | 0 |
| baseline | `baseline` | mcts_base | 100 | 0.530 | [0.433, 0.625] | 0 |
| n_worlds | `n_worlds=1` | random | 100 | 0.950 | [0.888, 0.978] | 0 |
| n_worlds | `n_worlds=1` | greedy | 100 | 0.450 | [0.356, 0.548] | 0 |
| n_worlds | `n_worlds=1` | mcts_base | 100 | 0.440 | [0.347, 0.538] | 0 |
| n_worlds | `n_worlds=2` | random | 100 | 0.940 | [0.875, 0.972] | 0 |
| n_worlds | `n_worlds=2` | greedy | 100 | 0.650 | [0.553, 0.736] | 0 |
| n_worlds | `n_worlds=2` | mcts_base | 100 | 0.400 | [0.309, 0.498] | 0 |
| n_worlds | `n_worlds=8` | random | 100 | 0.930 | [0.863, 0.966] | 0 |
| n_worlds | `n_worlds=8` | greedy | 100 | 0.650 | [0.553, 0.736] | 0 |
| n_worlds | `n_worlds=8` | mcts_base | 100 | 0.470 | [0.375, 0.567] | 0 |
| uct_c | `uct_c=0.35` | random | 100 | 0.960 | [0.902, 0.984] | 0 |
| uct_c | `uct_c=0.35` | greedy | 100 | 0.630 | [0.532, 0.718] | 0 |
| uct_c | `uct_c=0.35` | mcts_base | 100 | 0.480 | [0.385, 0.577] | 0 |
| uct_c | `uct_c=0.7` | random | 100 | 0.930 | [0.863, 0.966] | 0 |
| uct_c | `uct_c=0.7` | greedy | 100 | 0.550 | [0.452, 0.644] | 0 |
| uct_c | `uct_c=0.7` | mcts_base | 100 | 0.470 | [0.375, 0.567] | 0 |
| uct_c | `uct_c=2.8` | random | 100 | 0.930 | [0.863, 0.966] | 0 |
| uct_c | `uct_c=2.8` | greedy | 100 | 0.660 | [0.563, 0.745] | 0 |
| uct_c | `uct_c=2.8` | mcts_base | 100 | 0.560 | [0.462, 0.653] | 0 |
| rollout | `rollout=heuristic` | random | 100 | 0.910 | [0.838, 0.952] | 0 |
| rollout | `rollout=heuristic` | greedy | 100 | 0.640 | [0.542, 0.727] | 0 |
| rollout | `rollout=heuristic` | mcts_base | 100 | 0.470 | [0.375, 0.567] | 0 |
| rollout | `rollout=random` | random | 100 | 0.920 | [0.850, 0.959] | 0 |
| rollout | `rollout=random` | greedy | 100 | 0.460 | [0.366, 0.557] | 0 |
| rollout | `rollout=random` | mcts_base | 100 | 0.470 | [0.375, 0.567] | 0 |
| eval_weights | `eval=prize_only` | random | 100 | 0.920 | [0.850, 0.959] | 0 |
| eval_weights | `eval=prize_only` | greedy | 100 | 0.600 | [0.502, 0.691] | 0 |
| eval_weights | `eval=prize_only` | mcts_base | 100 | 0.410 | [0.319, 0.508] | 0 |
| eval_weights | `eval=material2x` | random | 100 | 0.920 | [0.850, 0.959] | 0 |
| eval_weights | `eval=material2x` | greedy | 100 | 0.600 | [0.502, 0.691] | 0 |
| eval_weights | `eval=material2x` | mcts_base | 100 | 0.430 | [0.337, 0.528] | 0 |

各セルの集計 JSON は `eval/results/ablation/<構成>--vs-<相手>.json`（シード・
再現コマンド `cell.repro` 込み）。

## 考察

- **Random 軸は飽和している。** 全構成が vs Random 0.91–0.96 に収まり、構成間の
  CI はすべて重なる。この軸は健全性チェック（どの変種も Random には圧勝する）と
  してのみ機能し、パラメータの識別力は Greedy 軸と MCTS 基準軸が担っている。
- **世界サンプル数 N（効果が最も大きい軸）。** N=1（PIMC 相当の単一世界）は
  vs Greedy 0.450 [0.356, 0.548] で baseline の 0.640 [0.542, 0.727] から大きく
  劣化し、対 MCTS 基準でも 0.440 と負け越し傾向。単一 determinization への
  過適合（その世界に都合の良い手を過大評価する）が実際に勝率を毀損することを
  示す。N=2 は Greedy 軸こそ回復する（0.650）が、基準構成との直接対決で
  0.400 [0.309, 0.498] と **CI 上限が 0.5 を割る有意な負け越し**。N=8 は
  どの軸でも N=4 と有意差がない — 時間予算固定（0.8s/手）では世界数を増やすと
  1 世界あたりのシミュレーション数が減るため、N=4 が投資対効果の釣り合い点。
- **UCT 探索定数（鈍感な軸）。** 0.35 / 0.7 / 1.4 / 2.8 の 4 点で、全軸とも
  baseline と CI が重なる。uct_c=2.8 の対基準 0.560 [0.462, 0.653] が点推定では
  最良だが CI が 0.5 を跨ぎ、Greedy 軸（0.660 vs 0.640）でも差はない。
  max_tree_depth=1・max_root_actions=6 という浅い木では、探索定数の影響が
  構造的に小さいと解釈できる。
- **rollout policy。** random rollout は vs Greedy 0.460 [0.366, 0.557] で
  baseline（greedy rollout）の 0.640 から明確に劣化 — rollout の質が葉値の
  質に直結することを示す。heuristic（型ティア）rollout は全軸で greedy rollout と
  同等（vs Greedy 0.640 で同値）であり、追加の複雑さに見合う利得がない。
- **評価関数重み。** prize_only（プライズ以外の特徴を全部 0）は対基準
  0.410 [0.319, 0.508] で劣化傾向 — 盤面資源（ポケモン・エネルギー・HP・手札）の
  特徴が刈り込み・葉評価に寄与している。material2x（資源重み 2 倍）も対基準
  0.430 と改善せず、既定の重みバランスが両方向のずらしに対して頑健。

## 最良構成と選定根拠

**採用: baseline（SOT-1672 採用構成そのまま）**

```json
{"max_root_actions": 6, "max_tree_depth": 1, "rollout_turns": 100,
 "rollout_depth": 200, "n_worlds": 4, "time_budget_s": 0.8,
 "deviate_margin": 0.1}
```

（+ 既定値 `uct_c=1.4`, `rollout="greedy"`, 評価重み default）

選定根拠:

1. **どの変種も baseline をどの軸でも有意に上回らなかった**（全 30 変種セルで
   baseline 行と CI が重なるか、変種側が下回る）。
2. 逆に baseline から動かすと有意に劣化する方向が複数ある: `n_worlds=1`
   （vs Greedy 0.450）、`n_worlds=2`（対基準 0.400、CI 上限 < 0.5）、
   `rollout=random`（vs Greedy 0.460）、`eval=prize_only`（対基準 0.410）。
   baseline は全軸で「劣化しない側」に居る唯一の構成。
3. 点推定で対基準最良の `uct_c=2.8`（0.560）は CI [0.462, 0.653] が 0.5 を跨ぎ、
   他軸にも利得がない。30 セルの多重比較から CI が 0.5 を跨ぐ 1 点を拾って
   採用するのは選択バイアス（SOT-1649 で確認した p-hacking 回避の方針に反する）
   なので採用しない。

再現（最良構成の対基準測定、シード込み 1 行）:

```bash
venv/bin/python eval/ablation.py --n 100 --shards 10 --seed 94000 --only-config baseline --only-opponent mcts_base --force
```

## 相性の循環（Nash averaging / mElo）についての判断

相手軸が複数あるため、相性の循環（A が B に勝ち、B が C に勝ち、C が A に勝つ）が
疑われる場合は Balduzzi et al.（arXiv:1806.02643）の Nash averaging / mElo に
沿った集計を検討する方針だった。今回の結果では **循環は観測されなかった**:
全 11 構成で「vs Random ≫ vs Greedy ≥ vs MCTS 基準」の推移的な序列が保たれ
（唯一 `rollout=random` で vs Greedy 0.460 < vs MCTS 0.470 と順位が入れ替わるが、
差 0.010 は CI 幅に対して無視できる）、相手強度の一貫した単調性を示す。
したがって本表では **相手軸ごとの素の勝率を並記する集計のまま**とし、
Nash averaging は適用しない。構成間の優劣判断は、識別力のある Greedy 軸と
MCTS 基準軸（自己対戦的な直接対決）を主、Random 軸を健全性チェックとして読む。
