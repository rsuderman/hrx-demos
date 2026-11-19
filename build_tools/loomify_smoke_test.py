"""Generation recipes and runtime devices remain independent in local smoke."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "loomify_smoke", Path(__file__).with_name("loomify-smoke.py")
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class PlatformForwardingTest(unittest.TestCase):
    def run_smoke(self, arguments=(), export_status=0):
        commands = []

        def launch(command, **_kwargs):
            commands.append(command)
            process = mock.Mock(returncode=None)
            if Path(command[0]).name == "fixture-converter":
                package = Path(
                    next(
                        a.split("=", 1)[1]
                        for a in command
                        if a.startswith("--output-dir=")
                    )
                )
                package.mkdir()
                (package / "model.json").write_text('{"name":"fixture"}')
                process.returncode = export_status
            process.wait.return_value = process.returncode
            process.poll.side_effect = lambda: process.returncode
            return process

        def stop(process):
            process.returncode = 0

        def response(request, **_kwargs):
            if request.full_url.endswith("/v1/status"):
                data = {"data": [{"id": "fixture", "loom": {"registration": "ready"}}]}
            elif request.full_url.endswith("/input_tokens"):
                data = {"input_tokens": 3}
            else:
                data = {
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "length"}
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            return io.StringIO(json.dumps(data))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf, tokenizer = root / "fixture.gguf", root / "tokenizer.json"
            gguf.touch()
            tokenizer.write_text("{}")
            argv = [
                "loomify-smoke.py",
                f"--gguf={gguf}",
                f"--tokenizer={tokenizer}",
                "--converter=fixture-converter",
                "--serve=fixture-serve",
                *arguments,
            ]
            output = io.StringIO()
            opener = mock.Mock()
            opener.open.side_effect = response
            with (
                mock.patch.object(smoke.sys, "argv", argv),
                mock.patch.object(
                    smoke.shutil, "which", side_effect=lambda value: value
                ),
                mock.patch.object(smoke.subprocess, "Popen", side_effect=launch),
                mock.patch.object(smoke, "stop", side_effect=stop),
                mock.patch.object(
                    smoke.urllib.request, "build_opener", return_value=opener
                ),
                mock.patch.object(smoke.socket, "socket") as socket,
                redirect_stdout(output),
            ):
                socket.return_value.__enter__.return_value.getsockname.return_value = (
                    "127.0.0.1",
                    12345,
                )
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                status = smoke.main()
        return status, json.loads(output.getvalue()), commands

    def test_default_recipe_reaches_both_exports(self):
        status, result, commands = self.run_smoke()
        self.assertEqual(status, 0)
        self.assertTrue(result["success"])
        self.assertEqual(result["platform"], "amdgpu:gfx11-generic")
        self.assertEqual(result["device"], "amdgpu")
        self.assertEqual([case["format"] for case in result["cases"]], ["loom", "bc"])
        for command in commands[::2]:
            self.assertIn("--platform=amdgpu:gfx11-generic", command)
            self.assertFalse(any(a.startswith("--device=") for a in command))
        for command in commands[1::2]:
            self.assertIn("--device=amdgpu", command)
            self.assertFalse(any(a.startswith("--platform=") for a in command))

    def test_explicit_recipe_and_device_are_forwarded_independently(self):
        status, result, commands = self.run_smoke(
            ["--platform=fixture-recipe", "--device=fixture-device"]
        )
        self.assertEqual(status, 0)
        self.assertEqual(result["platform"], "fixture-recipe")
        self.assertEqual(result["device"], "fixture-device")
        for command in commands[::2]:
            self.assertIn("--platform=fixture-recipe", command)
        for command in commands[1::2]:
            self.assertIn("--device=fixture-device", command)

    def test_export_failure_retains_requested_recipe_and_device(self):
        status, result, commands = self.run_smoke(
            ["--platform=unsupported-fixture", "--device=fixture-device"],
            export_status=1,
        )
        self.assertEqual(status, 1)
        self.assertFalse(result["success"])
        self.assertEqual(result["platform"], "unsupported-fixture")
        self.assertEqual(result["device"], "fixture-device")
        self.assertEqual(result["error"], "loom export failed")
        self.assertEqual(len(commands), 1)
        self.assertIn("--platform=unsupported-fixture", commands[0])


if __name__ == "__main__":
    unittest.main()
