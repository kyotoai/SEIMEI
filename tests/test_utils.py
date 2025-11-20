import json
import tempfile
import unittest
from pathlib import Path

from seimei.utils import load_run_messages


class LoadRunMessagesTests(unittest.TestCase):
    def _write_messages(self, base: Path, messages) -> Path:
        run_dir = base / "run-abc"
        run_dir.mkdir(parents=True, exist_ok=True)
        msg_path = run_dir / "messages.json"
        msg_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_dir

    def test_loads_full_history(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
            {"role": "agent", "name": "code_act", "content": "step-1"},
            {"role": "agent", "name": "code_act", "content": "step-2"},
            {"role": "assistant", "content": "done"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_messages(Path(tmp), messages)
            loaded = load_run_messages(run_dir)
            self.assertEqual(len(loaded), len(messages))
            self.assertEqual(loaded[-1]["content"], "done")

    def test_limits_by_agent_step(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
            {"role": "agent", "content": "step-1"},
            {"role": "agent", "content": "step-2"},
            {"role": "assistant", "content": "done"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_messages(Path(tmp), messages)
            first_step = load_run_messages(run_dir, step=1)
            self.assertEqual(len(first_step), 3)
            self.assertEqual(first_step[-1]["content"], "step-1")

            second_step = load_run_messages(run_dir, step=2)
            self.assertEqual(len(second_step), 4)
            self.assertEqual(second_step[-1]["content"], "step-2")

    def test_accepts_run_id_with_runs_dir(self) -> None:
        messages = [
            {"role": "user", "content": "Task"},
            {"role": "agent", "content": "step-1"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = self._write_messages(base, messages)
            run_id = run_dir.name
            loaded = load_run_messages(run_id, runs_dir=base)
            self.assertEqual(len(loaded), len(messages))

    def test_invalid_step_raises(self) -> None:
        messages = [
            {"role": "user", "content": "Task"},
            {"role": "agent", "content": "step-1"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_messages(Path(tmp), messages)
            with self.assertRaises(ValueError):
                load_run_messages(run_dir, step=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
