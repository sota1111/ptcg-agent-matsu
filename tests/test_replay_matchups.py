import json
import os
import tempfile
import unittest

from eval.replay_matchups import SCHEMA, build_fixture, parse_replay


def replay(episode, opponent_card, reward=-1, seat=0, status="DONE"):
    names = [{"Name": "sota1111"}, {"Name": "opponent"}]
    if seat == 1:
        names.reverse()
    decks = [[1] * 60, [2] * 60]
    players = [{"deck": [{"name": "Own Pokémon"}]},
               {"deck": [{"name": opponent_card}]}]
    if seat == 1:
        players.reverse()
    rewards = [reward, -reward] if seat == 0 else [-reward, reward]
    statuses = [status, "DONE"] if seat == 0 else ["DONE", status]
    return {"info": {"EpisodeId": episode, "Agents": names},
            "rewards": rewards, "statuses": statuses,
            "steps": [[{"visualize": [{"action": decks,
                "current": {"players": players}}]}]]}


class ReplayMatchupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_extracts_seat_result_archetype_and_failure(self):
        path = self.write("lucario.json", replay(10, "Mega Lucario ex", seat=1))
        got = parse_replay(path, "sota1111")
        self.assertEqual(got["submission_seat"], 1)
        self.assertEqual(got["opponent_archetype"], "Mega Lucario")
        self.assertEqual(got["failure_mode"], "loss")
        self.assertEqual(len(got["opponent_deck"]), 60)

    def test_fixture_is_deterministic_and_requires_holdouts(self):
        paths = [self.write("z.json", replay(12, "Mega Lucario ex")),
                 self.write("a.json", replay(11, "Alakazam"))]
        first = build_fixture(paths, "sota1111", "54811671", "v1")
        second = build_fixture(list(reversed(paths)), "sota1111", "54811671", "v1")
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], SCHEMA)
        self.assertEqual([m["opponent_archetype"] for m in first["matchups"]],
                         ["Alakazam", "Mega Lucario"])

    def test_missing_required_matchup_fails(self):
        path = self.write("a.json", replay(11, "Alakazam"))
        with self.assertRaisesRegex(ValueError, "Mega Lucario"):
            build_fixture([path], "sota1111", "54811671", "v1")


if __name__ == "__main__":
    unittest.main()
