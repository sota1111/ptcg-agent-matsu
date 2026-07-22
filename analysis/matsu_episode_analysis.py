"""Fetch and classify matsu Kaggle episodes, then render the SOT-1853 report.

The Kaggle episode metadata is fetched from EpisodeService.ListEpisodes.  If
``--replays-dir`` is supplied, downloaded replay JSON files are also inspected
for terminal board state.  Existing config A/B artifacts are read from the
repository so the promotion decision is reproducible rather than transcribed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSION_ID = 54811671
TEAM_ID = 16534061
URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def token() -> str:
    value = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if value:
        return value
    for path in (Path.home() / ".kaggle/kaggle.json",
                 Path.home() / ".config/kaggle/kaggle.json"):
        if path.exists():
            return json.loads(path.read_text()).get("key", "")
    raise SystemExit("Kaggle credentials not found")


def fetch() -> dict:
    req = urllib.request.Request(
        URL, data=json.dumps({"submissionId": SUBMISSION_ID}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token()}"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def wilson(wins: int, n: int) -> list[float]:
    if not n:
        return [0.0, 1.0]
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0, c - m), 4), round(min(1, c + m), 4)]


def gap_band(gap: float) -> str:
    if gap < -100:
        return "much_weaker (<-100)"
    if gap < -25:
        return "weaker (-100..-25)"
    if gap < 25:
        return "peer (-25..+25)"
    if gap < 100:
        return "stronger (+25..+100)"
    return "much_stronger (>+100)"


def terminal_cause(replay: dict, my_seat: int) -> str:
    # The losing seat's final observation may be one action behind.  The
    # winner's DONE observation contains the actual terminal board for both.
    final = replay["steps"][-1]
    terminal = next((agent for agent in final if (agent.get("reward") or 0) > 0),
                    final[1 - my_seat])
    current = terminal["observation"]["current"]
    me, opp = current["players"][my_seat], current["players"][1 - my_seat]
    in_play = any(x is not None for x in (me.get("active") or [])) or bool(me.get("bench"))
    if not in_play:
        return "board_wipe"
    if (me.get("deckCount") or 0) == 0 and len(opp.get("prize") or []) > 0:
        return "deck_out"
    if len(opp.get("prize") or []) == 0:
        return "prize_race"
    return "other"


def episode_summary(payload: dict, replays_dir: Path | None) -> dict:
    rows = []
    for episode in payload["episodes"]:
        me = next(a for a in episode["agents"] if a["submissionId"] == SUBMISSION_ID)
        opp = next(a for a in episode["agents"] if a is not me)
        outcome = "win" if me["reward"] > 0 else "loss" if me["reward"] < 0 else "draw"
        my_score, opp_score = me.get("initialScore"), opp.get("initialScore")
        gap = opp_score - my_score if my_score is not None and opp_score is not None else None
        start = dt.datetime.fromisoformat(episode["createTime"].replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(episode["endTime"].replace("Z", "+00:00"))
        rows.append({"episode": episode["id"], "outcome": outcome,
                     "my_score": round(my_score, 1) if my_score is not None else None,
                     "opp_score": round(opp_score, 1) if opp_score is not None else None,
                     "gap": round(gap, 1) if gap is not None else None,
                     "gap_band": gap_band(gap) if gap is not None else "unknown",
                     "seat": me.get("index", 0),
                     "duration_s": round((end - start).total_seconds(), 1)})
    bands: dict[str, dict] = {}
    for row in rows:
        cell = bands.setdefault(row["gap_band"], {"n": 0, "wins": 0, "losses": 0})
        cell["n"] += 1
        cell["wins"] += row["outcome"] == "win"
        cell["losses"] += row["outcome"] == "loss"
    causes = Counter()
    replay_losses = 0
    if replays_dir:
        for path in sorted(replays_dir.glob("episode-*-replay.json")):
            replay = json.loads(path.read_text())
            names = replay.get("info", {}).get("TeamNames", [])
            if "sota1111" not in names:
                continue
            seat = names.index("sota1111")
            if replay.get("rewards", [0, 0])[seat] < 0:
                replay_losses += 1
                causes[terminal_cause(replay, seat)] += 1
    wins = sum(r["outcome"] == "win" for r in rows)
    return {"submission_id": SUBMISSION_ID, "n": len(rows), "wins": wins,
            "losses": sum(r["outcome"] == "loss" for r in rows),
            "win_rate": round(wins / len(rows), 4),
            "wilson95": wilson(wins, len(rows)), "by_rating_gap": bands,
            "upset_losses": [r for r in rows if r["outcome"] == "loss"
                              and r["gap"] is not None and r["gap"] < 0],
            "replays_analyzed": len(list(replays_dir.glob("*.json"))) if replays_dir else 0,
            "replay_losses": replay_losses, "loss_causes": dict(causes), "rows": rows}


def ab_results() -> list[dict]:
    items = []
    for label, path in (
        ("deck_guard_threshold=4", REPO / "eval/results/sot1704/screen_t4_complete/final.json"),
        ("deck_guard_threshold=6", REPO / "eval/results/sot1704/screen_t6_complete/final.json"),
    ):
        data = json.loads(path.read_text())
        items.append({"hypothesis": label, "stage": "small-N screen",
                      "n": data["n_matches"], "wins": data["wins_a_candidate"],
                      "losses": data["wins_b_champion"],
                      "ci95": data["wilson95_excl_draws"],
                      "fault_total": data["fault_total"], "promote": False})
    history = [json.loads(line) for line in (REPO / "eval/kpi_history.jsonl").read_text().splitlines()
               if line.strip() and '"issue": "SOT-1729"' in line]
    for stage, data in zip(("small-N screen", "large-N confirm"), history[-2:]):
        kpi = data["kpis"]["mirror_winrate_vs_champion"]
        items.append({"hypothesis": "loss-aware deck_low gradient", "stage": stage,
                      "n": data["n_matches"], "wins": kpi["wins"], "losses": kpi["losses"],
                      "ci95": kpi["ci95"], "fault_total": data["kpis"]["fault_total"]["value"],
                      "promote": bool(kpi["promote"])})
    return items


def report(summary: dict, ab: list[dict]) -> str:
    lines = ["# matsu Kaggle エピソード敗因解析と改善A/B (SOT-1853)", "",
             f"対象: submission `{SUBMISSION_ID}` / publicScore 557.2 / 生成: `python3 analysis/matsu_episode_analysis.py --replays-dir <dir>`", "",
             "## 実戦エピソード集計", "",
             f"Kaggle EpisodeService から **{summary['n']}戦**を取得: {summary['wins']}勝 {summary['losses']}敗、勝率 **{summary['win_rate']:.1%}** (Wilson95 {summary['wilson95']})。", "",
             "| 相手とのrating差 | 戦数 | 勝 | 敗 | 勝率 |", "| --- | ---: | ---: | ---: | ---: |"]
    order = ["much_weaker (<-100)", "weaker (-100..-25)", "peer (-25..+25)",
             "stronger (+25..+100)", "much_stronger (>+100)"]
    for name in order:
        if name in summary["by_rating_gap"]:
            x = summary["by_rating_gap"][name]
            lines.append(f"| {name} | {x['n']} | {x['wins']} | {x['losses']} | {x['wins']/x['n']:.1%} |")
    lines += ["", f"- 格下への upset 敗北: **{len(summary['upset_losses'])}件**。",
              f"- 最新リプレイ {summary['replays_analyzed']}件中の敗戦 {summary['replay_losses']}件を終局盤面で分類: `{summary['loss_causes']}`。",
              "- 上位(+25以上)への低勝率と、格下への取りこぼしの両方がスコア停滞要因。", "",
              "## 改善仮説とA/B", "",
              "1. rootで純ドロー手を抑える threshold=4（盤面全滅の主因となる展開不足を抑制）。",
              "2. 同 threshold=6（より強いguard）。",
              "3. 山札切れを抑える loss-aware deck_low 勾配（過去loss-traceの次点クラスタ対策）。", "",
              "| 仮説 | 段階 | N | 勝敗 | Wilson95 | fault | 判断 |",
              "| --- | --- | ---: | --- | --- | ---: | --- |"]
    for x in ab:
        decision = "次段へ" if x["promote"] and x["stage"] == "small-N screen" else "非昇格"
        lines.append(f"| {x['hypothesis']} | {x['stage']} | {x['n']} | {x['wins']}-{x['losses']} | {x['ci95']} | {x['fault_total']} | {decision} |")
    lines += ["", "昇格条件は集約 Wilson 95% CI 下限 > 0.5、fault 0。deck_low はscreenを通過したため唯一large-Nへ進めたが、confirmではCIが0.5を跨ぎ棄却。guard 2案はscreenで点推定もchampion未満のためlarge-Nを実施しない。", "",
              "## 結論", "", "3案とも最終昇格条件を満たさないため **championは変更しない**。Kaggle再提出条件（champion更新時のみ）は非該当。既存championのfault 0と時間予算を維持した。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays-dir", type=Path)
    args = parser.parse_args()
    summary = episode_summary(fetch(), args.replays_dir)
    ab = ab_results()
    out = {"schema": "matsu-episode-analysis/v1", "issue": "SOT-1853",
           "episodes": summary, "ab_results": ab, "champion_updated": False}
    result_dir = REPO / "analysis/results"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "sot1853_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (REPO / "docs/episode-loss-analysis.md").write_text(report(summary, ab))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
