# matsu Kaggle エピソード敗因解析と改善A/B (SOT-1853)

対象: submission `54811671` / publicScore 557.2 / 生成: `python3 analysis/matsu_episode_analysis.py --replays-dir <dir>`

## 実戦エピソード集計

Kaggle EpisodeService から **76戦**を取得: 31勝 45敗、勝率 **40.8%** (Wilson95 [0.3044, 0.5202])。

| 相手とのrating差 | 戦数 | 勝 | 敗 | 勝率 |
| --- | ---: | ---: | ---: | ---: |
| much_weaker (<-100) | 4 | 2 | 2 | 50.0% |
| weaker (-100..-25) | 14 | 11 | 3 | 78.6% |
| peer (-25..+25) | 23 | 11 | 12 | 47.8% |
| stronger (+25..+100) | 24 | 5 | 19 | 20.8% |
| much_stronger (>+100) | 10 | 2 | 8 | 20.0% |

- 格下への upset 敗北: **12件**。
- 最新リプレイ 24件中の敗戦 11件を終局盤面で分類: `{'board_wipe': 10, 'other': 1}`。
- 上位(+25以上)への低勝率と、格下への取りこぼしの両方がスコア停滞要因。

## 改善仮説とA/B

1. rootで純ドロー手を抑える threshold=4（盤面全滅の主因となる展開不足を抑制）。
2. 同 threshold=6（より強いguard）。
3. 山札切れを抑える loss-aware deck_low 勾配（過去loss-traceの次点クラスタ対策）。

| 仮説 | 段階 | N | 勝敗 | Wilson95 | fault | 判断 |
| --- | --- | ---: | --- | --- | ---: | --- |
| deck_guard_threshold=4 | small-N screen | 50 | 22-28 | [0.3116, 0.5769] | 0 | 非昇格 |
| deck_guard_threshold=6 | small-N screen | 50 | 20-30 | [0.2761, 0.5382] | 0 | 非昇格 |
| loss-aware deck_low gradient | small-N screen | 140 | 87-53 | [0.5389, 0.6975] | 0 | 次段へ |
| loss-aware deck_low gradient | large-N confirm | 300 | 163-137 | [0.4868, 0.5988] | 0 | 非昇格 |

昇格条件は集約 Wilson 95% CI 下限 > 0.5、fault 0。deck_low はscreenを通過したため唯一large-Nへ進めたが、confirmではCIが0.5を跨ぎ棄却。guard 2案はscreenで点推定もchampion未満のためlarge-Nを実施しない。

## 結論

3案とも最終昇格条件を満たさないため **championは変更しない**。Kaggle再提出条件（champion更新時のみ）は非該当。既存championのfault 0と時間予算を維持した。
