# ターン内行動順ソルバー / mctsS（SOT-1677）

Świechowski et al.（arXiv:1808.04794）§III-B3 **Board Solver** の cabt 再現。
Hearthstone ではターン内の攻撃順の組合せが最大 ~10^10 に爆発するため、論文は
「攻撃 1 組ずつの gain-loss 評価（≤64 評価）」の貪欲ソルバーでターン内行動列を
一括生成し、MCTS のマクロ行動（1 手 = ソルバー実行）として使う **mctsS** 変種を
提案した。cabt の「ターン内行動列」（エネルギー付与 → 進化 → グッズ → にげる →
攻撃 …）に同じ構造を移植する。

## 実装

- `agents/turn_solver.py` — `TurnSolver`
  - **合法性はエンジンの select が唯一の真実**（Action Enumerator 契約,
    `agents/actions.py`）。ソルバーはエンジンが提示した option の並べ替え・
    選択だけを行い、インデックスを発明しないため不正手を生成できない。
  - **gain-loss 純関数スコア**（カード属性のみ・カード名/ID ハードコードなし）:
    - gain: 攻撃ダメージ推定（弱点 ×2 / 抵抗 −30、KO+プライズボーナス）、
      場に出るカードの属性価値（進化/プレイ/エネルギー付与）、回復量、有益カウント。
    - loss: コスト文脈で支払う資源（ディスカード等）、不利カウント、そして
      **ターン終了系（ATTACK/END）への機会費用** = 同じ select 内に残っている
      展開系 option 数 × 25。固定型ティア順序（SOT-1671 の教訓）を「その局面で
      何を諦めるか」から導出される量で置き換えたもので、リーサルは即座に
      機会費用を上回り、展開が尽きると攻撃 → END の順に自然に選ばれる。
  - **計算量上限**: option 評価回数 `max_evals`（既定 64、論文の ≤64 評価に対応）、
    ステップ上限 `max_steps=30`、`deadline`（planner の時間予算）で anytime に
    打ち切り。打ち切りはマクロが短くなるだけで、常に合法な状態で停止する。
  - 乱数はコイン（SelectContext.COIN_HEAD=46 のチャンスノード）のサンプリング
    のみで、注入 Rng からしか消費しない → 同一シード+同一エンジン応答で
    同一行動列（ASSUMPTIONS.md A-9 のエージェント側スコープ）。
- `agents/planner.py` — `PlannerConfig.solver: bool = False`（+
  `solver_max_evals`）。**on のとき木の展開 1 エッジ = マクロ行動**:
  エッジの候補手を step した後、TurnSolver がその手番プレイヤーのターン残りを
  補完し、子ノードはターン境界（または terminal / 上限打ち切り点）に置かれる。
  1 階層 ≒ 1 ターンになるため、`max_tree_depth` はターン数の意味になる。
  **既定 off は従来と完全一致**（1 エッジ = 1 select、既存テストがピン留め）。
- `eval/bench.py` — 変更不要。`--config-a '{"solver": true}'` が constructor
  kwargs → `PlannerConfig` に流れる既存経路で on/off できる。

```bash
# mctsS（基準構成 + solver）vs 基準 mcts
venv/bin/python eval/bench.py --agent-a mcts --agent-b mcts --n 16 --seed 981 \
  --config-a '{"max_root_actions": 6, "max_tree_depth": 1, "rollout_turns": 100,
               "rollout_depth": 200, "n_worlds": 4, "time_budget_s": 0.8,
               "deviate_margin": 0.1, "solver": true}' \
  --config-b '{"max_root_actions": 6, "max_tree_depth": 1, "rollout_turns": 100,
               "rollout_depth": 200, "n_worlds": 4, "time_budget_s": 0.8,
               "deviate_margin": 0.1}'
```

## 動作確認（実機プローブ）

実エンジン 2 試合のプローブで solve 呼び出し 521 回、マクロ長 mean 2.46 手 /
max 10 手、停止理由は turn_end / terminal のみ（eval_cap / deadline 打ち切りは
発生せず）— 評価上限 64 は実局面のターン長に対して十分な余裕がある。

## ベンチマーク結果（mctsS vs 基準 mcts）

N=320（20 シャード × 16 試合、シード 981–989 / 9810–9820、先後はシャード内で交替）。
生データ: `eval/results/solver/shard_*.json`、集計: `eval/results/solver/aggregate.json`。

| 指標 | 値 |
| --- | --- |
| mctsS 勝率（引き分け除外） | **0.475**（152 勝 168 敗、draw 0） |
| Wilson 95% CI | **[0.421, 0.530]** |
| engine rejects / exceptions | 0 / 0 |
| budget violations / planner fallbacks / unfinished | 0 / 0 / 0 |
| planner move max | 641 ms（予算 0.8 s 内） |

CI が 0.5 を跨ぐため **mctsS と基準 mcts に有意差なし**（点推定はわずかに劣後）。
不正手 0・時間切れ 0 は維持。→ **champion は基準 mcts（solver off 既定）を維持**し、
mctsS は `--config '{"solver": true}'` のオプトイン変種として残す。

### 改善しなかった理由の考察

1. **cabt のターン内分岐は Hearthstone より桁違いに小さい。** 論文の Board Solver が
   効いたのは攻撃順の組合せが ~10^10 に爆発し素の MCTS では木が浅くなる環境。実機
   プローブではマクロ長 mean 2.46 手 / max 10 手で、評価上限 64 にすら届かない —
   ソルバーで潰すべき組合せ爆発がそもそも存在しない。
2. **基準構成は既に 1 階層 ≒ 1 ターンに近い。** champion 構成（`max_tree_depth: 1` +
   rollout）はルートの 1 select だけ木で分岐し残りを rollout 方策で埋める。mctsS の
   「1 エッジ = 1 ターン」への置換で得られる深さの利得が小さい。
3. **ターン残りの補完が貪欲固定で分散が消える。** 基準 mcts は rollout 方策
   （deviate_margin=0.1 の確率的逸脱つき）でターン内も複数系列をサンプルするが、
   mctsS のマクロ内部は決定的な貪欲 1 系列で、ソルバーのスコアの癖（機会費用の
   重みなど）が全シミュレーションに一様にバイアスとして乗る。シャード別勝敗
   （4-12 〜 13-3）のばらつきもデッキ/初手条件でこのバイアスの当たり外れが出る
   ことと整合的。

論文（§IV-C）でも mctsS 単体の改善は環境・デッキ依存で、S は V（value network）との
併用（mctsVS）で真価が出る構成要素と位置づけられている。cabt では SOT-1679（mctsV）
との併用評価が SOT-1680 の変種マトリクスで行われる。
