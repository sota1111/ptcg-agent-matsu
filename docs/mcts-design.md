# Determinized MCTS プランナー設計判断（SOT-1672）

4層アーキテクチャの上2層（[3] Planner / [4] Evaluator）の設計判断と根拠の記録。
実装: `agents/planner.py`（planner）, `agents/evaluator.py`(evaluator),
`agents/mcts_agent.py`（agent契約への接続）。ベンチ: `eval/bench.py`。

参照論文: M. Świechowski, K. Godlewski, B. Sawicki, J. Mańdziuk,
"Monte Carlo Tree Search: a review of recent modifications and applications"
(arXiv:1808.04794)。以下「Świechowski et al.」。

## 1. 不完全情報の扱い — Determinization（PIMC系）を採用

**選択肢:** (a) Determinization / Perfect Information MC(PIMC系, 世界サンプリング
×各世界で完全情報探索), (b) Information Set MCTS（情報集合を直接ノード化）,
(c) 隠れ情報を無視した観測直上の探索。

**採用: (a) Determinization（root並列, 世界数 N は外部パラメータ）。**

- エンジンが determinization 前提の search API（`cg/api.py` `search_begin`:
  隠れゾーンの fill を引数で受け取り完全情報の探索状態を返す / `search_step`）を
  提供しており（docs/engine-facts.md §1.3）、(a) はエンジンをルールの単一情報源の
  まま使える。(b) は情報集合の同型判定を自前実装する必要があり、ルール再実装
  リスク（SOT-1670 の調査で回避と決定）を持ち込む。
- Świechowski et al. §4.1 の整理どおり、determinization には strategy fusion /
  non-locality の理論的弱点があるが、N を増やすことで隠れ情報の分布に対する
  平均化が効く。N は `PlannerConfig.n_worlds` として外部化し、N=1（PIMC 相当）
  から任意の N まで SOT-1673 の ablation で比較できる。
- 集約は root 並列: 各世界で独立に PUCT 探索し、root 行動の訪問数を世界間で
  合算、総訪問数最大の行動を返す（`MctsPlanner._best_action`）。

**サンプリング整合性** (`sample_fills`): トラッシュ・場・表向きサイド・
スタジアム等の可視カードを候補プールから差し引き、可視情報と矛盾しない
fill のみ生成する。相手デッキはミラー（自分と同一 60 枚）を仮定 — 自己対戦
ベンチでは厳密に正しい。非ミラー相手ではプール再サンプリングでサイズ契約
のみ満たす（相手デッキ推定は Phase 3 以降の課題）。

## 2. 確率的効果（コイン・シャッフル）— chance node 明示を採用

**選択肢:** (a) `manual_coin=True` でコイン選択を明示的な chance node として
受け取り、注入シード由来 Rng で 50/50 サンプルする, (b) エンジン内部 RNG に
任せて rollout サンプルとして扱う。

**採用: (a) chance node 明示 + 注入 Rng サンプリング。**

- エンジン内部 RNG は外部から seed 注入できない（ASSUMPTIONS.md A-9）。(b) では
  コイン結果までエージェント制御外になり、エージェント側乱数の再現性
  （SOT-1671 の Rng 規約）から chance node が完全に抜け落ちる。(a) なら
  コインという主要な確率的効果を注入シード由来 Rng の管理下に置ける。
- ただし **manual_coin で捕捉できるのはコインのみ**。search API はそれ以外
  （シャッフル効果等）に `std::random_device` シードの内部 RNG を使い続ける
  （docs/engine-facts.md §5, `Api.h:84-92`）。実測（2026-07-13）: 同一
  (観測, fills, 行動列) でも探索結果は分岐し、同一シードの着手が 12 局面中
  3 局面で変化した。→ 再現性の扱いは §9 参照。
- Świechowski et al. §3.2（確率ノードの扱い: expectimax 型の chance node を
  MCTS に組み込む系譜）に沿い、ツリー内では chance node を「エッジ通過時に
  1 outcome をサンプルして通過」(`_resolve_chance`)、rollout 中は毎回
  再サンプル (`_rollout_action` の COIN 分岐) とする。訪問回数が増えるほど
  コイン結果の分布が自然に平均化され、明示的な expectimax 展開（全分岐を
  子ノード化）よりノード数が抑えられる。
- シャッフルは fill 生成時（determinization）に注入 Rng で行うため、
  探索中の shuffle 系効果もエンジン側で決定的に処理される。

## 3. Anytime・時間予算アウェア探索

- `plan(view, rng, budget_s)` は手番ごとの時間予算を引数で受け取る
  （既定は `PlannerConfig.time_budget_s`）。**予算の 80% で探索を打ち切り、
  残り 20% は集約・後処理のマージン**（`budget_fraction=0.8`）。
- いつ打ち切られても最善手を返す: root 候補は Greedy prior 順に並び、
  反復 0 回でも index 0（greedy 最善）が返る。世界が 1 つも構築できない
  例外時も greedy prior に退化（`degraded_count` で計数、ベンチで 0 を確認）。
- `MctsAgent` は全決定の所要時間を計測し、予算超過を `budget_violations`
  として計数（受け入れ条件: 0 件）。`eval/bench.py` が集計・報告する。
- `max_iterations` を小さく設定すると壁時計より先に反復数で打ち切られ、
  決定経路から壁時計を排除できる（再現性テスト tests/test_mcts.py が
  決定的バックエンドと組み合わせて使用）。

## 4. Evaluator — 差し替え可能な leaf 価値インターフェース

- `Evaluator.evaluate(obs, root_player) -> [0,1]`（勝率推定）。
  `HeuristicEvaluator` はカード属性由来の特徴（取得サイド・場のポケモン数・
  付きエネルギー・HP 合計・手札枚数・山切れ）の線形和のロジスティック圧縮。
  重みは辞書で外部注入可能（SOT-1673 ablation 点）。
- カード ID / カード名をキーにしたハードコード表は禁止
  （`scripts/lint_hardcoded_cards.py` が強制）。SOT-1674 で learned value に
  同一インターフェースのまま差し替える。

## 5. Rollout policy — 軽量ヒューリスティック（A/B 可能）

`PlannerConfig.rollout` で切替: `"greedy"`（GreedyAgent を両側に適用、既定）/
`"heuristic"`（OptionType 段位表 + 攻撃ダメージ推定の軽量方策）/
`"random"`（一様合法手）。random rollout との A/B は
`--config-a '{"rollout": "random"}'` で取れる（チューニング結果 §7）。

## 6. 外部パラメータ（SOT-1673 ablation 準備）

`PlannerConfig`: `n_worlds`（世界数 N）, `uct_c`（UCT/PUCT 定数）,
`rollout` / `rollout_depth` / `rollout_turns`, `max_tree_depth`,
`max_root_actions` / `max_child_actions`, `prior_temperature`,
`time_budget_s` / `budget_fraction`, `max_iterations`,
`deviate_margin`（greedy prior から乖離するために挑戦手が上回るべき
平均価値マージン）。
すべて `eval/bench.py --config-a/--config-b` の JSON からコンストラクタに
注入できる。

## 7. チューニング経過（eval/results/tune/）

n=40〜80 の粗いスイープの要点（すべて vs Greedy・先後交替・reject/exception 0）:

- 深いツリー（max_tree_depth 4〜8）より **浅いツリー + ゲーム終端まで届く
  greedy rollout**（max_tree_depth=1, rollout_turns=100, rollout_depth=200）が
  安定して優位 — 中盤局面のヒューリスティック評価より終端結果の方が信頼できる。
- rollout は random < heuristic < greedy。
- 時間予算は 0.1s → 0.4s → 0.8s で単調に改善（それ以上 2.0s / 3.2s は
  改善が確認できず 0.8s を採用）。本番の持ち時間（ASSUMPTIONS.md A-1: 全体
  ≈10 分, 1 試合 ≈40 決定 → 0.8s/手で総計 ≈32s）に対して十分保守的。
- `deviate_margin`（低サンプル域での greedy prior からの誤乖離抑制,
  `_best_action`）: 0.8s 予算では反復数が少なく、visit-max 集約は
  ノイズで greedy 最善手から離れることがある。margin 0.1（挑戦手の
  プール平均価値が prior 比 +0.1 を超えたときのみ乖離）が
  0/0.05 より優位: ベース構成 0.577 (127/220) → margin 0.1 で
  0.625 (75/120, tune3)、独立シードでの検証 0.633 (114/180, tune3 v1)。
  `uct_c=0.7` との併用 (v2) は 0.606 で margin 単独に劣り不採用。
- **採用構成**:
  `{"max_root_actions": 6, "max_tree_depth": 1, "rollout_turns": 100,
  "rollout_depth": 200, "n_worlds": 4, "time_budget_s": 0.8,
  "deviate_margin": 0.1}`。
- **最終 500 戦ベンチ**（採用構成・チューニングと独立なシード
  93001–93020・先後交替, `eval/results/final/final_500.json`）:
  **勝率 0.618（309/500）, Wilson95 [0.575, 0.660]**, draw 0,
  reject（不正手）0, 時間切れ（budget violation）0, planner fallback 0,
  degraded 0, planner 決定時間最大 642ms（< 予算 800ms）。

## 8. 既知の限界（Phase 3+ への引き継ぎ）

- ミラーデッキ仮定: 対外戦では相手デッキ推定器に差し替える必要がある。
- strategy fusion / non-locality（Świechowski et al. §4.1）は root 並列
  determinization の原理的限界。ISMCTS 化は N の ablation（SOT-1673）で
  効果が頭打ちと分かった場合の選択肢。
- Greedy rollout はエンジン観測→raw dict 変換（`dataclasses.asdict`）を
  決定ごとに行う。プロファイル上ここが rollout コストの主因で、予算を
  増やす際の最初の最適化候補。

## 9. 再現性（同一シード+同一局面→同一着手）の適用範囲

受け入れ条件の再現性テストは、SOT-1670/1671 で確立した解釈
（ASSUMPTIONS.md A-9: エンジン内部乱数は固定不可 → 再現性要件は
**エージェント側乱数**への要件）に従って実装した:

- **エージェント側乱数**（determinization の fill サンプリング・コイン
  サンプリング・候補生成・タイブレーク jitter）はすべて注入シード由来の
  `Rng`（決定ごとに `rng.child("plan<n>")` で独立ストリーム化）。
  エンジン応答を固定した決定的バックエンド代替の下で、同一シード+同一
  局面→同一着手を tests/test_mcts.py が検証する（CI でも実行可能）。
- **実エンジンの search API 経由では**、シャッフル効果等が非注入の内部
  RNG を消費するため（§2 実測）、ビット再現は原理的に保証できない。
  Random/Greedy（エンジンを呼び返さない方策）の実エンジン再現性テスト
  （tests/test_engine_repro.py）はそのまま有効。
