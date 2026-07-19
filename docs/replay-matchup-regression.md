# Replay由来matchup回帰評価

Kaggle submission `54811671` の敗戦を、同じfixture versionとagent seedで比較できる
ローカルholdout suiteへ変換する。fixtureは入力episode、入力SHA-256、提出seat、勝敗、
status/failure mode、両デッキ、相手archetypeを保持する。結果JSONは各matchupをseat 0/1で
1回ずつ実行し、seat別W/L、archetype、fault、timeoutを記録する。

## Replay取得とfixture更新

```bash
mkdir -p /tmp/sot-1743-replays
kaggle competitions episodes 54811671 --format json
kaggle competitions replay <losing-episode-id> --path /tmp/sot-1743-replays

python3 eval/replay_matchups.py build '/tmp/sot-1743-replays/*.json' \
  --owner sota1111 --submission-ref 54811671 --version 54811671-v1 \
  --output eval/fixtures/kaggle-54811671-v1.json
```

fixtureを更新するのは、評価対象のKaggle submission、対戦環境schema、または固定holdout
方針を変更するときだけとする。versionを新しくし、元fixtureを上書きしない。生成時に
AlakazamとMega Lucarioの敗戦が1件以上ない場合は失敗する。生成物のdiffと
`source_episode_ids` / `source_sha256` をレビューしてcommitする。

## 先後交替の一括評価

`cg/` を `scripts/setup_engine.sh` で用意した環境で実行する。

```bash
venv/bin/python eval/replay_matchups.py run \
  --fixture eval/fixtures/kaggle-54811671-v1.json --seed 20260719 \
  --output eval/results/replay-matchups/54811671-v1-seed-20260719.json
```

同じfixture version、commit、seed、engine versionを比較単位にする。agent側乱数は再現するが、
engine内部乱数は外部seed不可（`ASSUMPTIONS.md` A-9）のため、戦績のbit-for-bit一致は保証せず、
条件と入力を再現する。出力の `fixture_sha256` が異なる結果同士は比較しない。
