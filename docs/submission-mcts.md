# 提出エージェントの champion MCTS 切替（SOT-1693）

提出実体 `main.py` を GreedyAgent（SOT-1671 ベースライン）から、リポジトリ champion の
Determinized MCTS プランナー（SOT-1672 設計 / SOT-1673 ablation 確定構成）に切り替えた。
論文ベース（determinized MCTS）のスタイルはそのまま。

## 構成

- **プランナー**: `MctsAgent` champion 構成（`docs/mcts-design.md` §7）—
  `max_root_actions=6, max_tree_depth=1, rollout_turns=100, rollout_depth=200,
  n_worlds=4, time_budget_s=0.8, deviate_margin=0.1, rollout=greedy, uct_c=1.4`
- **残り時間アウェアの予算制御**: 自エージェントの累積 act() 実測時間に対する段階スケジュール
  （ASSUMPTIONS A-1: 1試合の持ち時間 ≈10分/プレイヤー を前提）
  - < 300s: 0.8s/手（champion 予算）
  - 300–420s: 0.4s/手
  - 420–510s: 0.2s/手
  - ≥ 510s: 探索停止・Greedy へハンドオフ
- **多段フォールバック**（Validation Episode Error 防止）: MctsAgent 内部の劣化
  （planner 例外→greedy prior→random-legal）に加え、act() 例外時は
  GreedyAgent → 生 observation からの合法手（`_last_resort`）の順で必ず合法手を返す。
  初手のデッキ提出（`select is None`）は常に 60 枚デッキを返す。

## ミラーデッキ仮定の determinization について

champion プランナーの determinization は「相手デッキ＝自デッキ（ミラー）」を仮定して
未知情報を補完する（SOT-1672）。25 デッキ mirror-random 環境（両者が同一デッキを使う）では
この仮定は**そのまま真**であり、追加の相手デッキ推定なしで有効。independent モード
（相手が別デッキ）での相手デッキ推定は本 Issue のスコープ外。

## 25 デッキ汎化ベンチ（新 main=MCTS vs 旧 main=Greedy）

`eval/bench_decks.py` — `decks/initial/` の 25 大会デッキ（SOT-1684）を mirror・先後入替
（match index 偶奇で座席交替）でローテーション。shard 実行＋再集計（中断・再開可能）。
エンジン乱数は注入不可（ASSUMPTIONS A-9）のため agent seed のみ `--seed` から導出。

再現手順:

```
venv/bin/python eval/bench_decks.py --match-index <s> --deck-offset <d> --deck-limit 5 \
    --json eval/results/submission/s<s>_d<d>.json    # s=0..15, d=0,5,10,15,20
venv/bin/python eval/bench_decks.py --aggregate 'eval/results/submission/s*.json' \
    --json eval/results/submission/final.json
```

### 結果（N=400 = 25 デッキ × 16 試合、seed 20260715）

| 指標 | 値 |
| --- | --- |
| 勝敗 | MCTS 283 勝 / Greedy 116 勝 / 引分 1 |
| 勝率（引分除外） | **0.709** |
| Wilson 95% CI | **[0.663, 0.752]**（下限 > 0.5 達成） |
| fault 合計 | **0**（reject / exception / unfinished / fallback / budget違反 / greedy handoff すべて 0） |
| 思考時間 /手 | mean 553ms / p95 641ms / max 643ms（予算 0.8s 内） |
| 思考時間 /試合 | mean 48.8s / p95 69.2s / max 390.4s（持ち時間 600s 内・時間切れ 0） |

per-deck 勝率は `eval/results/submission/final.json` の `per_deck` を参照。25 デッキ中 23 で
勝率 ≥ 0.5（例外: 04_dragapult_dusknoir 0.50, 23_slowking_naic_4th 0.44）、最高は
18_rocket_s_honchkrow / 20_cynthia_s_garchomp_ex の 0.94。

### 読み

- ablation（単一デッキ, vs Greedy 0.640）と整合する強さが 25 デッキ全体でも維持され、
  むしろ上回った（0.709）。determinized MCTS は mirror-random 環境で汎化する。
- 予算ガバナは 400 試合中 1 試合（09_slowking mirror、A の決定 1123 手・累積 390s）だけ
  300s 閾値を超え、設計通り予算縮小が発火した（超過後の平均思考 428ms→214ms）。Greedy
  ハンドオフ（510s）には一度も到達せず（greedy_handoffs=0）、残り 399 試合は champion
  予算のまま完走。時間切れ・fault は 0。
