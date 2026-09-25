import importlib.util
from pathlib import Path
import shlex
import unittest


SCRIPT = Path(__file__).with_name("q6a-ask.py")
SPEC = importlib.util.spec_from_file_location("q6a_ask", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BroadcastCommandTest(unittest.TestCase):
    def test_prompt_survives_remote_shell_quoting(self):
        prompt = "I'm here; don't claim you saw a cat $(touch /tmp/not-run)"
        command = MODULE.broadcast_shell_command("request123", prompt)
        self.assertEqual(shlex.split(command)[-1], prompt)
        self.assertEqual(shlex.split(command)[-4], "request123")


if __name__ == "__main__":
    unittest.main()
