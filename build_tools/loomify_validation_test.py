# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Contract checks for the small CPU-only validation command."""

import contextlib
import importlib.util
import io
import json
import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location(
    "loomify_validation", Path(__file__).with_name("loomify-validation.py")
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
sys.path.pop(0)


class NativeValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.fixture = self.base / "fixture.gguf"
        self.fixture.touch()
        for name in ("MODULE.bazel", "MODULE.bazel.lock", ".bazelversion"):
            (self.root / name).write_text(name)
        self.wrapper = self.base / "bazel"
        self.wrapper.write_text("#!/bin/sh\nexit 0\n")
        self.wrapper.chmod(0o755)
        self.args = validation.parse_args(
            [
                "native",
                "--bazel",
                str(self.wrapper),
                "--result-dir",
                str(self.base / "result"),
            ]
        )
        self.environment = {
            **{name: str(self.fixture) for name in validation.FIXTURES},
            "LOOM_SERVE_DIR": str(self.base / "state"),
        }

    def events(
        self,
        xml="<testsuite tests='1'><testcase classname='Admission' name='source_and_bc'/></testsuite>",
        *,
        stale=False,
        status="PASSED",
        cached=False,
    ):
        events = self.base / "events.jsonl"
        records = []
        for index, label in enumerate(validation.NATIVE_TARGETS):
            path = self.base / f"{index}.xml"
            path.write_text(xml)
            if stale:
                os.utime(path, (1, 1))
            records.append(
                {
                    "id": {"testResult": {"label": label}},
                    "testResult": {
                        "status": status,
                        "cachedLocally": cached,
                        "testActionOutput": [
                            {"name": "test.xml", "uri": path.as_uri()}
                        ],
                    },
                }
            )
        events.write_text("\n".join(json.dumps(row) for row in records))
        return events

    def test_jobs_are_bounded_and_default_four(self):
        self.assertEqual(self.args.jobs, 4)
        for jobs in ("0", "5", "invalid"):
            with (
                self.subTest(jobs=jobs),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                validation.parse_args(
                    [
                        "native",
                        "--bazel",
                        str(self.wrapper),
                        "--result-dir",
                        str(self.base / "result"),
                        "--jobs",
                        jobs,
                    ]
                )

    def test_fresh_results_keep_complete_case_identity(self):
        start = time.time_ns()
        results = validation.read_results(self.events(), start)
        self.assertEqual(set(results), set(validation.NATIVE_TARGETS))
        self.assertTrue(
            all(
                row["cases"] == [{"class": "Admission", "name": "source_and_bc"}]
                for row in results.values()
            )
        )

    def test_bad_xml_and_cached_status_fail_closed(self):
        for options in (
            {"xml": "<testsuite/>"},
            {"xml": "<testsuite><testcase><skipped/></testcase></testsuite>"},
            {"xml": "<testsuite failures='1'><testcase/></testsuite>"},
            {"xml": "<testsuite><testcase><error/></testcase></testsuite>"},
            {"xml": "<testsuite><testcase status='notrun'/></testsuite>"},
            {"stale": True},
            {"status": "FAILED"},
            {"cached": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validation.read_results(
                    self.events(**options),
                    time.time_ns() if options.get("stale") else 0,
                )

    def test_missing_or_duplicate_results_fail_closed(self):
        events = self.events()
        rows = events.read_text().splitlines()
        events.write_text("\n".join(rows[:-1]))
        with self.assertRaisesRegex(ValueError, "Missing current"):
            validation.read_results(events, 0)
        events.write_text("\n".join(rows + [rows[0]]))
        with self.assertRaisesRegex(ValueError, "multiple test attempts"):
            validation.read_results(events, 0)
        events = self.events()
        (self.base / "0.xml").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or stale"):
            validation.read_results(events, 0)

    def test_missing_fixture_stops_before_any_command_and_records_failure(self):
        with (
            mock.patch.dict(
                os.environ, {"LOOM_SERVE_DIR": str(self.base / "state")}, clear=True
            ),
            mock.patch.object(validation.Path, "cwd", return_value=self.root),
            mock.patch.object(validation.subprocess, "Popen") as command,
        ):
            self.assertEqual(validation.run(self.args, self.root), 1)
        command.assert_not_called()
        result = json.loads((self.args.result_dir / "result.json").read_text())
        self.assertIn("Required fixture", result["error"])
        self.assertEqual(result["status"], "failed")

    def test_real_child_failure_cannot_advance_and_is_recorded(self):
        self.wrapper.write_text("#!/bin/sh\nexit 7\n")
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            mock.patch.object(validation.Path, "cwd", return_value=self.root),
            mock.patch.object(validation.source, "git", return_value=b"fixture-head"),
            mock.patch.object(validation.source, "inputs", return_value={}),
        ):
            self.assertEqual(validation.run(self.args, self.root), 1)
        result = json.loads((self.args.result_dir / "result.json").read_text())
        self.assertEqual(result["exit"], 7)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("tests", result)
        self.assertEqual(result["command"].count("--local_test_jobs=4"), 1)
        self.assertEqual(result["command"][-4:], list(validation.NATIVE_TARGETS))
        for name in validation.FIXTURES:
            self.assertIn(f"--test_env={name}", result["command"])

    def test_successful_child_and_source_drift(self):
        self.wrapper.write_text(
            "#!" + sys.executable + "\n"
            "import json, pathlib, sys\n"
            "event_path=pathlib.Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('--build_event_json_file=')))\n"
            "events=[]\n"
            "for index,label in enumerate(a for a in sys.argv if a.startswith('//')):\n"
            " path=event_path.parent/f'{index}.xml'\n"
            " path.write_text(\"<testsuite><testcase classname='Native' name='SOURCE_and_BC'/></testsuite>\")\n"
            " events.append({'id':{'testResult':{'label':label}},'testResult':{'status':'PASSED','testActionOutput':[{'name':'test.xml','uri':path.as_uri()}]}})\n"
            "event_path.write_text('\\n'.join(json.dumps(e) for e in events))\n"
        )
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            mock.patch.object(validation.Path, "cwd", return_value=self.root),
            mock.patch.object(validation.source, "git", return_value=b"fixture-head"),
            mock.patch.object(validation.source, "inputs", return_value={}),
        ):
            self.assertEqual(validation.run(self.args, self.root), 0)
        result = json.loads((self.args.result_dir / "result.json").read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(sum(t["count"] for t in result["tests"].values()), 4)
        self.args.result_dir = self.base / "drift"
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            mock.patch.object(validation.Path, "cwd", return_value=self.root),
            mock.patch.object(validation.source, "git", return_value=b"fixture-head"),
            mock.patch.object(
                validation.source, "inputs", side_effect=[{}, {"changed": {}}]
            ),
        ):
            self.assertEqual(validation.run(self.args, self.root), 1)
        self.assertIn(
            "Source changed",
            json.loads((self.args.result_dir / "result.json").read_text())["error"],
        )

    def cancellation_case(self, cleanup_exit):
        marker = self.base / "child.pid"
        daemon = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        self.wrapper.write_text(
            "#!" + sys.executable + "\n"
            "import os,pathlib,signal,sys,time\n"
            "if sys.argv[1]=='shutdown':\n"
            f" if {cleanup_exit}: sys.exit({cleanup_exit})\n"
            f" os.kill({daemon.pid},signal.SIGTERM)\n"
            " sys.exit(0)\n"
            "signal.signal(signal.SIGINT,lambda *_:sys.exit(130))\n"
            f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        environment = {
            **os.environ,
            **self.environment,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        command = [
            sys.executable,
            str(validation.ROOT / "build_tools/loomify-validation.py"),
            "native",
            "--bazel",
            str(self.wrapper),
            "--result-dir",
            str(self.args.result_dir),
        ]
        child = subprocess.Popen(command, cwd=validation.ROOT, env=environment)
        try:
            deadline = time.monotonic() + 10
            while (
                not marker.exists()
                and child.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertTrue(marker.exists())
            self.assertIsNone(daemon.poll())
            child.send_signal(signal.SIGTERM)
            self.assertEqual(child.wait(timeout=10), 1)
            result = json.loads((self.args.result_dir / "result.json").read_text())
            self.assertIn("Interrupted by signal", result["error"])
            self.assertEqual(
                result["cleanup"]["command"], [str(self.wrapper), "shutdown"]
            )
            self.assertEqual(result["cleanup"]["exit"], cleanup_exit)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)
            if cleanup_exit:
                self.assertEqual(result["cleanup"]["status"], "unresolved")
                self.assertIsNone(daemon.poll())
                self.assertIn("before reuse/removal", result["cleanup"]["error"])
            else:
                self.assertEqual(result["cleanup"]["status"], "complete")
                self.assertEqual(daemon.wait(timeout=3), -signal.SIGTERM)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            if daemon.poll() is None:
                daemon.terminate()
                daemon.wait()

    def test_cancellation_uses_public_cleanup_for_detached_work(self):
        self.cancellation_case(0)

    def test_failed_public_cleanup_is_unresolved_not_released(self):
        self.cancellation_case(7)

    def test_new_external_result_directory_required(self):
        for path in (self.root / "result", Path("relative"), self.root):
            self.args.result_dir = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                validation.run(self.args, self.root)

    def test_isolation_and_wrong_root_rejected(self):
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            mock.patch.object(validation.Path, "cwd", return_value=self.base),
        ):
            with self.assertRaisesRegex(ValueError, "checkout root"):
                validation.preflight(self.args, self.root)
        with (
            mock.patch.dict(
                os.environ, {**self.environment, "PYTHONPATH": "/untrusted"}, clear=True
            ),
            mock.patch.object(validation.Path, "cwd", return_value=self.root),
        ):
            with self.assertRaisesRegex(ValueError, "overrides"):
                validation.preflight(self.args, self.root)


if __name__ == "__main__":
    unittest.main()
