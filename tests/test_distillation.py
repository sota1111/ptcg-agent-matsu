import json
import os
import tempfile
import unittest

from agents.features import FEATURE_NAMES_V2
from agents.search_encoding import ACTION_FEATURE_NAMES
from train.distill import (TeacherDatasetWriter, TeacherExample, load_dataset,
                           sha256_file, train)


class TestDistillation(unittest.TestCase):
    def example(self, trajectory="game-1", step=0, value=1.0):
        state = tuple([0.1] * len(FEATURE_NAMES_V2))
        actions = (tuple([0.0] * len(ACTION_FEATURE_NAMES)),
                   tuple([1.0] * len(ACTION_FEATURE_NAMES)))
        return TeacherExample(state, actions, (1, 7), value, trajectory, step)

    def test_dataset_keeps_visit_and_value_targets_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "teacher.jsonl")
            writer = TeacherDatasetWriter(path, {"teacher": "matsu-mcts", "seed": 7})
            self.assertEqual(writer.append_trajectory([self.example()]), 1)
            self.assertEqual(writer.append_trajectory([self.example()], valid=False), 0)
            manifest = writer.write_manifest(os.path.join(tmp, "manifest.json"))
            row = load_dataset(path)[0]
            self.assertEqual(row.visits, (1, 7))
            self.assertEqual(row.value, 1.0)
            self.assertEqual(manifest["dataset_sha256"], sha256_file(path))
            self.assertEqual(manifest["teacher"], "matsu-mcts")

    def test_end_to_end_training_and_checkpoint_resume_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "teacher.jsonl")
            writer = TeacherDatasetWriter(data, {"teacher": "fixture"})
            writer.append_trajectory([self.example(step=i, value=1.0) for i in range(4)])
            full = os.path.join(tmp, "full.json")
            resumed = os.path.join(tmp, "resumed.json")
            checkpoint = os.path.join(tmp, "checkpoint.json")
            manifest = train(data, full, None, epochs=3, hidden_size=4, seed=11)
            train(data, resumed, checkpoint, epochs=1, hidden_size=4, seed=11)
            train(data, resumed, checkpoint, epochs=3, hidden_size=4, seed=11)
            with open(full, encoding="utf-8") as left, open(resumed, encoding="utf-8") as right:
                self.assertEqual(json.load(left), json.load(right))
            with open(resumed + ".manifest.json", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["dataset_sha256"], sha256_file(data))
            self.assertIn("code_revision", manifest)
            self.assertEqual(manifest["epochs"], 3)
            self.assertLess(manifest["loss_after"]["total"],
                            manifest["loss_before"]["total"])

    def test_resume_rejects_changed_dataset_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "teacher.jsonl")
            writer = TeacherDatasetWriter(data, {})
            writer.append_trajectory([self.example()])
            checkpoint = os.path.join(tmp, "checkpoint.json")
            train(data, os.path.join(tmp, "model.json"), checkpoint,
                  epochs=1, hidden_size=4)
            writer.append_trajectory([self.example("game-2")])
            with self.assertRaisesRegex(ValueError, "provenance"):
                train(data, os.path.join(tmp, "model.json"), checkpoint,
                      epochs=2, hidden_size=4)


if __name__ == "__main__":
    unittest.main()
