# 松(matsu) 対戦KPI定義 (SOT-1708)

対戦性能の向上を継続観測するためのKPI。ベンチ/評価を走らせるたびに
`eval/kpi.py` がKPIレコード(1計測=1行)を **`eval/kpi_history.jsonl`**
(コミット対象。scratchな `eval/results/` とは分離)へ追記し、
`eval/kpi_report.py` が履歴と直近比較(改善/悪化)を表示する。

## KPI一覧

| # | KPI | 定義・測定方法 | 改善方向 |
|---|-----|---------------|---------|
| 1 | `mirror_winrate_vs_greedy` | 25大会デッキ(`decks/initial`, SOT-1684)のmirror対戦(両者同一デッキ・席交替)で、提出エージェント(`main.SubmissionAgent` = champion MCTS)が固定ベースライン **GreedyAgent** に勝つ率。draw除外、Wilson 95% CI付き。 | **高いほど良い**。CIが前回と重ならない上昇のみ有意な改善とみなす |
| 2 | `self_deck_out_loss_rate` | 松の敗戦のうち、終局観測から `deck_out`(自分の山札0・相手プライズ残あり)と分類される敗戦の割合(分類は `eval/loss_trace_matsu.py` と同一の機構分類: no_active / deck_out / prize_race / other)。SOT-1697で支配的敗因と判明。 | **低いほど良い** |
| 3 | `fault_total` | engine reject(違法手)+ agent exception + unfinished + 松側ヘルスカウンタ(fallback / budget violation=時間切れ / planner fallback / degraded / emergency fallback / greedy handoff)の総和。 | **0維持**(1でもあれば悪化=NG。トレンドではなくゲート) |
| 4 | `decision_time_mean_ms` | 松の1手あたり平均決定時間(ms)。参考値としてmaxと時間切れ数(budget_violations)も記録。 | **低いほど良い**(ただし時間予算内なら探索を厚くする方が優先。budget_violations=0 が前提) |

## 測定方法(標準の計測コマンド)

フルKPI(deck-out分類込み)は `eval/kpi.py --measure` で計測する。
shard分割で長時間実行を分割・再開できる(`bench_decks.py` と同じ流儀):

```bash
# shard s: 25デッキ×1試合(松の席は s%2)
venv/bin/python eval/kpi.py --measure --match-index 0 \
    --shard-json eval/results/kpi/shard_0.json
venv/bin/python eval/kpi.py --measure --match-index 1 \
    --shard-json eval/results/kpi/shard_1.json
# 集約して履歴へ1行追記
venv/bin/python eval/kpi.py --finalize 'eval/results/kpi/shard_*.json' \
    --issue SOT-XXXX
```

既存ベンチからも最小フックで記録できる(終局状態を捨てるため KPI 2 はnull):

```bash
venv/bin/python eval/bench.py --agent-a mcts --agent-b greedy --n 100 \
    --kpi SOT-XXXX                       # bench.py 直接
venv/bin/python eval/bench_decks.py --aggregate 'shard_*.json' --kpi SOT-XXXX
venv/bin/python eval/aggregate_shards.py --kpi=SOT-XXXX out.json shard*.json
```

## 記録スキーマ (`matsu-kpi-v1`)

1レコード=JSONL 1行。共通フィールド: `ts`(UTC) / `git_sha` / `issue` /
`source`(kpi-measure | bench | bench_decks) / `opponent` / `deck_pool` /
`n_decks` / `n_matches` / `seed`、および `kpis.{各KPI}`(値+CI+内訳)。

## トレンド確認

```bash
venv/bin/python eval/kpi_report.py
```

履歴テーブル(時系列)と、直近2計測の各KPIについて Δ と
改善/悪化/横ばい(`fault_total` は OK/NG)を表示する。微小変動は
`FLAT_EPS`(勝率±0.005 等)以内なら横ばい扱い。**採用判断はCI非重複を
基準にする**(点推定の上下だけで昇格/棄却しない — p-hacking回避)。

## 運用ルール

- champion(`main.SubmissionAgent`)へ変更を入れたら、マージ前後どちらかで
  KPI計測を1件記録する(小規模Nでも可 — Nはレコードに残るのでCI幅で解釈)。
- `eval/kpi_history.jsonl` は追記のみ(書き換え・削除しない)。
- ベースライン(GreedyAgent)は固定。ベースラインを変える場合は新KPI名を
  切る(履歴の連続性を壊さない)。
