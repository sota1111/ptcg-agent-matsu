# Value Network と rollout Early Cutoff — mctsV（SOT-1679）

Świechowski et al. (arXiv:1808.04794) §IV-B / §III-B4 の再現。チーターデータ
（SOT-1678、`docs/selfplay-cheater.md`）で学習した勝敗2値の MLP value network
を、leaf 差し替え（SOT-1674 方式）ではなく **rollout の早期打ち切り
（Early Cutoff）** として MCTS に統合した記録。

## 1. 構成

```
train/logs/cheater_shard_*.jsonl   チーター自己対戦ログ（SOT-1678、v2特徴387次元）
        ▼
train/train_value.py --arch 256-128-64 | 64-32   純Python MLP（tanh隠れ層+sigmoid出力、
        │                                        minibatch ADAM + BCE、シード決定的）
        ▼
train/value_net.json               採用ネット（64-32、下記 §3 の設計判断）
train/value_net_paper.json         論文構成 256-128-64（精度比較アーム、未コミット）
        ▼
agents/evaluator.py  LearnedEvaluator（"layers" キーで MLP、無ければ SOT-1674 線形）
agents/planner.py    PlannerConfig.rollout_cutoff = {"min_steps": k}（Early Cutoff）
        ▼
eval/bench.py --config-a '{"evaluator":"value_net","rollout_cutoff":{"min_steps":20},...}'
```

- **Early Cutoff（論文 §III-B4 の採用方式）**: 各 rollout を最低 `min_steps`
  ステップ進め、その後**最初の手番境界**（盤面が落ち着いた時点）で打ち切り、
  ネットの勝率予測を rollout 結果の代わりに返す。`rollout_cutoff=None`
  （デフォルト）は従来挙動と完全一致（`tests/test_value_net.py::
  test_default_config_has_no_cutoff` / `test_without_cutoff_rollout_runs_to_depth`）。
- **leaf 差し替え A/B**: 同じネットを `"evaluator":"value_net"` +
  `rollout_turns=2`（SOT-1674 の value 依存構成）でも使えるため、統合方式の
  A/B（early cutoff vs leaf 差し替え）が同一重みで比較できる。

## 2. 学習（シード固定・チェックポイント再開可）

- データ: SOT-1678 チーター determinization 自己対戦 41 shard、
  **match 単位 80/20 分割**（crc32(file:match-id) % 10、例単位分割は同一試合の
  相関でリークするため不採用）→ train 28,458 / holdout 6,980 例、v2 特徴 387 次元。
- 学習則: minibatch ADAM（lr 1e-3, batch 64, L2 1e-6）+ BCE、tanh 隠れ層。
  シャッフルは epoch ごとの `Rng` 子シードから決定的に生成され、
  `--checkpoint` 再開でも単発実行と同一モデルに到達する
  （`test_checkpoint_resume_reproduces_single_run` / `test_same_seed_same_model`）。
- 再現コマンド（リポジトリルートから）:

```bash
# 採用ネット（64-32）
venv/bin/python train/train_value.py train/logs/cheater_shard_*.jsonl \
  --features v2 --arch 64-32 --epochs 10 --seed 61679 \
  --checkpoint train/logs/vn_small.ckpt.json --out train/value_net.json
# 論文構成（256-128-64、精度比較アーム）
venv/bin/python train/train_value.py train/logs/cheater_shard_*.jsonl \
  --features v2 --arch 256-128-64 --epochs 10 --seed 61679 \
  --checkpoint train/logs/vn_paper.ckpt.json --out train/value_net_paper.json
```

### Holdout 精度（6,980 例、論文の val 精度は 0.76–0.79）

| 構成 | logloss | 精度 | 推論レイテンシ (1評価) |
| --- | --- | --- | --- |
| 64-32（採用） | 0.5578 | 0.7139 | 0.57 ms |
| 256-128-64（論文） | 0.5656 | 0.7192 | 2.99 ms |
| heuristic（同一状態、参考） | 0.8051 | 0.5834 | — |

- ターン帯別精度（64-32）: turn 0+ 0.722 / 5+ 0.726 / 10+ 0.669 / 15+ 0.621 /
  20+ 0.579 — 終盤ほど例数が減り精度が落ちる（SOT-1674 と同傾向）。

## 3. 設計判断

- **純 Python 維持（numpy 不採用）**: リポジトリは pip 依存ゼロ（提出環境の
  制約、`docs/engine-facts.md`）。numpy を許容すれば学習は数十倍速いが、
  推論側 `agents/` に依存を持ち込まないため学習側も純 Python とし、
  長時間学習は per-epoch チェックポイントで中断耐性を持たせた。
- **採用ネットは 64-32（論文の 256-128-64 ではなく）**: 推論が 0.57 ms/評価
  vs 2.99 ms/評価と約5倍速く、0.8 s/手の時間予算内で Early Cutoff が
  rollout 毎に1回呼ばれる（スモーク2試合で 2,315 回発火）ため、レイテンシが
  探索量に直結する。holdout 精度差は +0.53pp（256-128-64 が 0.7192 vs 64-32 の
  0.7139。logloss はむしろ 64-32 が良い: 0.5578 vs 0.5656）で、レイテンシ優位を
  取った。論文は val 精度 0.76–0.79 を報告しており、64-32 の 0.714 は
  データ量（3.5万例 vs 論文 10万例規模）を考えれば妥当な水準。

## 4. 勝率比較（mctsV vs mcts、N=500/ペアリング）

`eval/run_value_net.sh`（50 shard × 10 試合、先後交替、シードは shard ごとに
導出 — エンジン乱数は注入不可、`docs/engine-facts.md` A-9）。全アームの他
パラメータは SOT-1673 採用の基準構成
（`max_root_actions=6, max_tree_depth=1, n_worlds=4, time_budget_s=0.8,
deviate_margin=0.1`）。

| ペアリング | 勝率 (A) | Wilson 95% CI | 有意? |
| --- | --- | --- | --- |
| mctsV vs 基準mcts（本命、論文: 82.0%） | 0.506 (253-247) | [0.462, 0.550] | no |
| leafV vs 基準mcts（SOT-1674 方式 A/B） | 0.504 (252-248) | [0.460, 0.548] | no |
| mctsV vs leafV（統合方式直接 A/B） | 0.502 (251-249) | [0.458, 0.546] | no |

- 品質カウンタ: 全ペアリングで不正手 rejects 0 / agent exceptions 0 /
  degraded 0 / unfinished 0。時間超過 budget violations は mctsV の両アームで
  0（planner move max 647 ms < 予算 800 ms）。leafV vs 基準mcts アームのみ
  A=1 / B=2（基準mcts 側にも発生、最大 860.7 ms — 20 並列ジョブの CPU 競合下の
  ノイズ。エンジン乱数は注入不可のため選別再実行はせずそのまま報告）。
- Early Cutoff 発火: mctsV vs base で A=263,785 回 / 500 試合（rollout ほぼ
  毎回発火 — 統合は意図どおり動作している）。
- 結果 JSON: `eval/results/value_net/`（shard 単位 + `aggregate_shards.py` 集計）

## 5. 考察

**mctsV は改善しない（論文の 82% は再現されず）**。leaf 差し替え（SOT-1674
方式）との直接 A/B も 0.502 で、**統合方式の差ではなくネットの質が律速**という
解釈が一貫する。論文との差の候補（依頼書 §7）:

1. **ネット精度の差**: holdout 精度 0.714 vs 論文 0.76–0.79。学習データが
   3.5万例（チーター41 shard）と論文の10万例規模より小さく、特に終盤帯
   （turn 15+ で 0.62）の精度が低い。rollout の代替として返す値の質が
   足りない。
2. **基準 rollout が既に強い**: 本リポジトリの rollout は展開系>攻撃>END の
   序列付き heuristic 方策（SOT-1671）で終局まで到達し、terminal 勝敗という
   ノイズは大きいが偏りのない値を返す。精度 0.71 のネットで途中打ち切りする
   のと期待値的に大差がない可能性（SOT-1674 の leaf 差し替え結論と同じ構図）。
3. **探索構造の差**: 採用構成は max_tree_depth=1・n_worlds=4・0.8 s/手の
   root 支配的な浅い探索で、論文（Hearthstone、深い tree・長い持ち時間）ほど
   value の質が探索結果に効かない。
4. **一発学習**: 論文 §IV-C の反復学習（世代を重ねてデータの質を上げる）は
   SOT-1680 のスコープ。本 Issue は初代チーターデータの一発学習で、論文の
   82% は反復後の数字と読むべき。

**採用判断: champion は基準 mcts（heuristic evaluator）を維持**。
`evaluator="value_net"` + `rollout_cutoff` は config オプトインとして残し
（デフォルト不変）、SOT-1680 の反復学習・変種マトリクスの部品とする。

## 6. 再現手順

```bash
bash scripts/check.sh                 # lint + syntax + 160 tests
bash eval/run_value_net.sh            # 3ペアリング × N=500（欠損shardのみ再実行→集計）
```
