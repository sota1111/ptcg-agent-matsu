# Learned Value 評価関数 — 設計・学習・勝率比較（SOT-1674）

自己対戦ログから盤面価値（勝率予測）を学習し、SOT-1672 で定義した
`Evaluator` interface（`agents/evaluator.py`）に差し替えて、ヒューリスティック
評価との勝率比較を行った記録。Strategy 部門レポート素材。

## 1. 構成

```
train/gen_selfplay.py   自己対戦ログ生成（シード固定・エージェント側再現可能）
        │  JSONL: 1ターンにつき1例 (特徴量 x, 最終勝敗 y, heuristic予測 h)
        ▼
train/train_value.py    純Pythonロジスティック回帰（依存追加ゼロ）
        │  train/value_model.json（重み・標準化・メタ情報）
        ▼
agents/evaluator.py     LearnedEvaluator（Evaluator interface 実装）
agents/features.py      特徴量定義（学習/推論で完全共有・dict/dataclass両対応）
        ▼
eval/bench.py --config-a '{"evaluator":"learned", ...}'   で差し替え
```

- **Evaluator interface 経由の差し替え**: `MctsAgent` は `--config` JSON の
  `"evaluator": "heuristic" | "learned"` 文字列で値関数を切り替える
  （`agents/evaluator.py make_evaluator`）。プランナ側 (`agents/planner.py`)
  は一切変更なし — interface が設計どおり機能した。
- **特徴量はカード属性由来のみ**（32次元）: 両陣営の 取得プライズ数 / 場の
  ポケモン数 / エネルギー総数 / HP・ダメージ総量 / 手札・山札枚数 / 山札切れ
  フラグ / バトル場のHP・エネルギー・最大攻撃力（CardIndex の属性参照）/
  場の与プライズ合計（ex・megaEx リスク）/ 状態異常、+ ターン数 + 手番フラグ。
  カード名・ID 直書きはリンタ（`scripts/lint_hardcoded_cards.py`、`train/` も
  走査対象に追加）で禁止を維持。未知カードは CardIndex の中立デフォルトに
  フォールバック。
- **方策学習には踏み込まない**（設計判断）: 本Issueは leaf value の差し替えのみ。
  方策（行動選択）を直接学習する場合は Huang & Ontañón (arXiv:2006.14171) の
  invalid action masking が必須になるが、行動空間が `obs.select` の可変
  インデックス集合であるため、まず value のみで効果を検証するのが低リスクと
  判断した。行動列挙・合法性は既存の Action Enumerator（合法手のみ列挙）が
  担保しており、不正手リスクは増えない。

## 2. 学習データと学習

- 生成: `train/gen_selfplay.py` ×8シャード（seed 61674001–61674008）、
  計 **32,000 試合・205,944 例**（draw/unfinished 0）。ミックスは
  Greedy自己対戦 60% / Greedy対Random 20%×2方向（MCTS の rollout 方策は
  greedy なので、rollout 到達分布に近い状態を主にしつつ、劣勢・優勢に振れた
  盤面もカバー）。ターン0（プライズ未配布のセットアップ）は除外。
- ラベル: その試合の最終勝敗（記録視点のプレイヤーが勝てば 1）。
- 学習: `train/train_value.py`（match 単位 80/20 分割 — 同一試合の状態は
  相関するため例単位分割はリークになる。標準化 + SGD、epochs 8, lr 0.02,
  L2 1e-6, seed 61674）。学習は数秒。
- **holdout 予測品質（41,500 例、同一状態での比較）**:

  | 予測器 | log-loss | 正解率 |
  | --- | --- | --- |
  | LearnedEvaluator | **0.5605** | **0.7124** |
  | HeuristicEvaluator | 0.6534 | 0.6688 |
  | ベースレート (0.552) | 0.688 | 0.552 |

  ターン帯別でも turn 0–4 (0.678 vs 0.630)、5–9 (0.758 vs 0.721)、
  10–14 (0.746 vs 0.705)、15–19 (0.782 vs 0.723) と全帯で learned が上回る
  （20+ のみ n=167 と少数で逆転）。**状態→勝敗の予測器としては learned が
  明確に優位。**
- 再現: `venv/bin/python train/gen_selfplay.py --n 4000 --seed 61674001 --out train/logs/shard_1.jsonl`（…以下 seed 61674002–8）→
  `venv/bin/python train/train_value.py train/logs/shard_*.jsonl --epochs 8 --lr 0.02`。
  エンジン内部乱数は注入不可（ASSUMPTIONS.md A-9）のためログのビット再現は
  不可で、再現可能性は生成手順（ミックス・シード・サンプリング規則）に対する
  もの（eval/bench.py と同じ規約）。

## 3. 勝率比較（差し替え前 vs 差し替え後）

**重要な前提**: SOT-1672 で採用した 基準構成 は `rollout_turns=100`（rollout が
ほぼ必ず終端到達）のため、**leaf evaluator がほとんど呼ばれない**。評価関数の
A/B を意味のある形で行うため、rollout をターン境界で打ち切って evaluator に
評価させる **value依存構成**（`rollout_turns=2, rollout_depth=60`、他は採用構成
と同一: `max_root_actions=6, max_tree_depth=1, n_worlds=4, time_budget_s=0.8,
deviate_margin=0.1`）を比較の主軸にした。

- H-V = MCTS + Heuristic（value依存構成） / L-V = MCTS + Learned（同）
- BASE = SOT-1672 採用 基準構成（heuristic; MCTS基準構成の相手軸）
- 各ペアリング N=200（10シャード×20、先後交替、シード固定コマンドは
  `eval/results/value_compare/commands.txt`）。全ペアリングで
  rejects / exceptions / budget violations / fallbacks = 0 を確認。

| 対戦 | N | 勝率 | Wilson 95% CI |
| --- | --- | --- | --- |
| H-V vs Random | 200 | 0.925 | [0.880, 0.954] |
| L-V vs Random | 200 | 0.910 | [0.862, 0.942] |
| H-V vs Greedy | 200 | 0.575 | [0.506, 0.641] |
| L-V vs Greedy | 200 | 0.550 | [0.481, 0.617] |
| **L-V vs H-V（直接対決）** | 200 | 0.545 | [0.476, 0.613] |
| H-V vs BASE（MCTS基準構成） | 200 | 0.515 | [0.446, 0.583] |
| **L-V vs BASE（MCTS基準構成）** | 200 | **0.385** | **[0.320, 0.454]** |
| L-BASE vs Greedy（採用構成に learned を入れた回帰確認） | 200 | 0.565 | [0.496, 0.632] |

参考: BASE vs Greedy は SOT-1672 の最終計測で 0.618 [0.575, 0.660]（N=500、
`eval/results/final/final_500.json`）。全ペアリングで planner の1手最大時間は
641–643ms（予算 800ms 内）、budget violations / degraded は 0。

再現: `bash eval/run_value_compare.sh`（シャード単位の1行再現コマンドは
`eval/results/value_compare/commands.txt`）。

## 4. 考察 — 予測は改善、対戦強さは改善せず（相手軸依存で有意に悪化）

**結論: learned value への差し替えは勝率を改善しない。** 提出・基準構成は
SOT-1672 の BASE（heuristic + 終端到達 rollout）を維持する。learned evaluator
は interface 経由でいつでも差し替え可能な状態で残す。

1. **予測品質と対戦強さの乖離。** holdout では learned が heuristic を全指標で
   明確に上回る（log-loss 0.5605 vs 0.6534、正解率 0.7124 vs 0.6688）が、
   直接対決（L-V vs H-V）は 0.545 [0.476, 0.613] で CI が 0.5 を跨ぎ有意差なし。
   「同一状態での勝敗予測の良さ」は「探索の leaf 評価としての手の順位付けの
   良さ」を保証しない — MCTS が必要とするのは校正された絶対値ではなく、
   候補手が導く盤面同士の相対順位であり、そこでは heuristic と実力差が
   出なかった。
2. **有意な悪化は「相手が MCTS 基準構成」の軸でのみ発生。** L-V vs BASE は
   0.385 [0.320, 0.454]（CI 上限 < 0.5 で有意に負け越し）。同じ打ち切り構成でも
   heuristic なら BASE と互角（H-V vs BASE 0.515）なので、打ち切り自体ではなく
   **learned value 固有の弱点**である。最も整合的な説明は**学習分布からの
   シフト**: 学習データは Greedy 自己対戦 60% + Greedy/Random 40% の到達分布で
   あり、Random/Greedy 相手（分布内）では互角、MCTS が誘導する盤面（分布外）
   でだけ価値推定が崩れる。平均的に良い予測器でも、系統的な誤評価は強い相手に
   一貫して突かれる（被搾取性）。
3. **単一相手軸では誤読するところだった。** 直接対決だけ見れば learned
   「やや優勢」(0.545) に見えるが、BASE 軸が弱点を露呈した。ByteRL
   (arXiv:2404.16689) に倣い相手軸を複数取った計測方針（SOT-1673 と同じ）が
   機能した実例であり、強さの非推移性 (Balduzzi et al., arXiv:1806.02643) が
   実データで観測された。
4. **この環境では leaf value の出番自体が少ない。** 平均 30 ターン程度で決着
   するため rollout が終端に届き（BASE 構成）、真の勝敗シグナルが得られる。
   value 依存構成（rollout_turns=2）にしても H-V vs BASE は 0.515 と改善せず、
   終端 rollout を学習値で置き換える動機がそもそも薄い。turn 20+ 帯で learned
   の予測精度が逆転する（n=167 の弱い証拠）ことも、長期戦になりやすい MCTS 戦
   での崩れと方向が一致する。
5. **回帰確認。** 採用構成に learned を入れた L-BASE vs Greedy は 0.565
   [0.496, 0.632] で、BASE の 0.618 [0.575, 0.660]（N=500）と CI が重なり有意な
   回帰ではない（採用構成では evaluator がほぼ呼ばれないため当然近い）。
   ただし改善もないため差し替える理由がない。
6. **改善するなら。** (a) MCTS 自己対戦ログを学習データに追加して分布シフトを
   詰める（iterated distillation）、(b) 特徴量を対戦相手の脅威（次ターン被
   ダメージ見込み等）まで拡張する、(c) 方策学習まで踏み込む場合は invalid
   action masking (Huang & Ontañón, arXiv:2006.14171) を必須とする — いずれも
   本Issue（任意フェーズ）のスコープ外として記録に留める。

## 5. 品質保証への回帰なし

- 不正手 0 / 時間予算違反 0 / fallback 0（上表の全ペアリングで確認）。
- `bash scripts/check.sh`（リンタ + compileall + unittest 全pass、`train/` を
  走査対象に追加した状態で確認）。
- 提出エージェント（main.py = Greedy）には触れていない。MCTS への切替判断は
  SOT-1673 の ablation 後（SOT-1672 の記録どおり）。
