# ベースライン計測結果（SOT-1671）

実行日: 2026-07-13 / 実行環境: DevContainer (Python 3.12.3, ローカル cabt エンジン) /
コマンド: `venv/bin/python eval/bench.py`（生 JSON は `eval/results/`）

エンジン内部乱数は外部注入不可（ASSUMPTIONS.md A-9）のため対戦内容は実行ごとに変わる。
エージェント側シードは `--seed` から試合ごとに派生（同一シード+同一観測列→同一着手は
`tests/test_engine_repro.py` で検証）。

## Random vs Random — N=1000（seed 20260713）

- 勝敗: A 489 / B 511 / 引き分け 0（勝率 A 0.489, Wilson95 [0.458, 0.520] — 対称性の確認）
- **エンジン reject（不正手）: 0 / エージェント例外: 0 / fallback: 0**
- 実行時間: 1試合 mean 4.53 ms / median 3.17 ms / max 20.89 ms（総 4.5 s）
- 1意思決定 mean 0.087 ms / 平均 51.2 意思決定/試合
- JSON: `eval/results/sot1671_random_vs_random_n1000.json`

## Greedy vs Random — N=1000（seed 20260713, 先後交替）

- 勝敗: **Greedy 938 / Random 62** / 引き分け 0
- **勝率 0.938, Wilson 95% CI [0.921, 0.951]**（引き分け除外; draws=0.5 でも 0.938）
- **エンジン reject（不正手）: 0 / エージェント例外: 0 / fallback: 0**
- 実行時間: 1試合 mean 2.61 ms / median 2.16 ms / max 23.55 ms（総 2.6 s）
- 1意思決定 mean 0.092 ms / max 1.31 ms / 平均 28.7 意思決定/試合
- JSON: `eval/results/sot1671_greedy_vs_random_n1000.json`

Random は A/B 両側で計 2000 試合、Greedy は 1000 試合を不正手 0・例外 0 で完走。

## 設計メモ

- Greedy は「攻撃はターンを終了させる」ため、攻撃を展開系アクション（PLAY/ATTACH/EVOLVE/
  ABILITY）より**下位**にスコアリングする（攻撃同士はダメージ推定で順位付け、KO ボーナスあり）。
  攻撃を最上位にした初期版は Random に勝率 0.35 と**負け越した**（展開ゼロで毎ターン即攻撃に
  なるため）。この序列変更だけで 0.938 まで改善。
- 評価はカード属性（HP・ワザダメージ・弱点/抵抗・にげるコスト・進化段階・ex/megaEx の
  サイド価値）由来の特徴量のみ。カード名/ID の即値参照は `scripts/lint_hardcoded_cards.py` が
  機械的に禁止（クリーンを CI で担保）。
- 未知カード ID・未知 enum 値・未知 context は デフォルト特徴量 / 最小コミット / 合法ランダム
  fallback に退避し、クラッシュも不正手も出さない（fallback 発生数はベンチで 0 を確認）。
