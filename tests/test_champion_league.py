import unittest

from eval.champion_league import aggregate, promotion_decision


def matches(wins, losses=0, opponent="champion", latency=10.0, fault=False):
    return ([{"candidate": "candidate", "opponent": opponent, "result": "win",
              "latency_ms": latency, "fault": fault}] * wins +
            [{"candidate": "candidate", "opponent": opponent, "result": "loss",
              "latency_ms": latency}] * losses)


class TestChampionLeague(unittest.TestCase):
    champion = {"latency_mean_ms": 10.0}

    def test_known_pass_fixture_promotes_against_champion_and_history(self):
        report = aggregate(matches(95, 5) + matches(19, 1, "history/v1"))
        decision = promotion_decision(report, self.champion,
                                      {"required_opponents": ["champion", "history/v1"]})
        self.assertTrue(decision["promote"], decision["reasons"])
        self.assertGreater(report["ci95"][0], 0.5)

    def test_known_fail_fixtures_reject_winrate_fault_and_latency(self):
        weak = aggregate(matches(5, 5))
        self.assertFalse(promotion_decision(weak, self.champion)["promote"])
        faulty = aggregate(matches(95, 5, fault=True))
        self.assertIn("candidate fault limit exceeded",
                      promotion_decision(faulty, self.champion)["reasons"])
        slow = aggregate(matches(95, 5, latency=12.0))
        self.assertIn("candidate latency ratio exceeded",
                      promotion_decision(slow, self.champion)["reasons"])

    def test_history_and_holdout_identity_is_explicit(self):
        report = aggregate(matches(20, opponent="champion") +
                           matches(20, opponent="history/holdout-deck-v2"))
        self.assertEqual(set(report["opponents"]),
                         {"champion", "history/holdout-deck-v2"})


if __name__ == "__main__":
    unittest.main()
