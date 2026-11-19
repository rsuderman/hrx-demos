# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Contract checks for the small CPU-only validation command."""

import contextlib
import copy
import fcntl
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
import xml.etree.ElementTree as ET

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


class HardwareValidationTest(unittest.TestCase):
    def setUp(self):
        NativeValidationTest.setUp(self)
        self.targets = [
            f"//lib/fake:{name}_test_execute_amd_gpu_hardware_test"
            for name in ("alpha_text", "alpha_bc")
        ]
        self.manifest = self.base / "targets.txt"
        self.manifest.write_text("\n".join(self.targets) + "\n")
        self.lock = self.base / "gpu.lock"
        self.execution_root = self.base / "external-execution-root"
        self.execution_root.mkdir()
        self.args = validation.parse_args(
            [
                "hardware",
                "--bazel",
                str(self.wrapper),
                "--result-dir",
                str(self.base / "result"),
                "--targets-file",
                str(self.manifest),
                "--gpu-lock",
                str(self.lock),
            ]
        )
        self.environment = {
            "LOOM_SERVE_DIR": str(self.base / "state"),
            "FAKE_BASE": str(self.base),
            "FAKE_TARGETS": json.dumps(self.targets),
            "FAKE_TAGS": json.dumps(sorted(validation.HARDWARE_TAGS)),
        }
        self.wrapper.write_text("#!" + sys.executable + "\n" + r"""
import fcntl,json,os,pathlib,signal,sys,time,xml.etree.ElementTree as ET
base=pathlib.Path(os.environ['FAKE_BASE'])
targets=json.loads(os.environ['FAKE_TARGETS'])
tags=json.loads(os.environ['FAKE_TAGS'])
lock=(base/'gpu.lock').open('a+')
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 held=False
 fcntl.flock(lock,fcntl.LOCK_UN)
except BlockingIOError: held=True
with (base/'calls.jsonl').open('a') as trace:
 trace.write(json.dumps({'argv':sys.argv[1:],'held':held})+'\n')
stage=sys.argv[1]
if stage=='shutdown':
 if os.environ.get('FAKE_RETRY_SHUTDOWN') and not (base/'allow-cleanup').exists(): sys.exit(7)
 if os.environ.get('FAKE_DAEMON_PID'): os.kill(int(os.environ['FAKE_DAEMON_PID']),signal.SIGTERM)
 sys.exit(0)
if stage=='query':
 doc=ET.Element('query')
 for label in targets:
  rule=ET.SubElement(doc,'rule',{'name':label,'class':'sh_test'})
  attr=ET.SubElement(rule,'list',{'name':'tags'})
  for tag in ([] if os.environ.get('FAKE_HOST') else tags): ET.SubElement(attr,'string',{'value':tag})
 print(ET.tostring(doc,encoding='unicode'));sys.exit(0)
if stage=='info':
 print(base/'external-execution-root');sys.exit(0)
if stage=='cquery':
 if '--output=label_kind' in sys.argv:
  for target in targets: print('sh_test rule '+target+' (abc1234)')
  if os.environ.get('FAKE_EXTRA'): print('cc_test rule //fake:host (abc1234)')
 else:
  relative=pathlib.Path('bazel-out/fake/bin/artifact')
  artifact=base/'external-execution-root'/relative
  artifact.parent.mkdir(parents=True,exist_ok=True);artifact.write_text('launcher')
  pathlib.Path(str(artifact)+'.sh').write_text('additional DefaultInfo file without runfiles')
  pathlib.Path(str(artifact)+'.runfiles_manifest').write_text('fixture '+str(base/'fixture.gguf')+'\n')
  if '--output=files' in sys.argv: print(str(relative)+'.sh')
  print(relative)
 sys.exit(0)
if stage=='build':
 if os.environ.get('FAKE_BUILD_FAIL'): sys.exit(9)
 if held and not os.environ.get('FAKE_ALLOW_BUSY_BUILD'): sys.exit(13)
 sys.exit(0)
if stage=='test':
 if not held: sys.exit(14)
 if os.environ.get('FAKE_WAIT'):
  signal.signal(signal.SIGINT,lambda *_:sys.exit(130))
  (base/'test-started').touch();time.sleep(30)
 events=pathlib.Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('--build_event_json_file=')))
 report={'format':'loom.test.v0','case_count':0,'scenario_count':1,'sample_count':0,'failed_sample_count':0,'trial_count':1,'failed_trial_count':0,'skipped_case_count':0,'skipped_scenario_count':0,'planning_issue_count':0,'xfail_count':0,'xpass_count':0,'xfail_mismatch_count':0,'samples':[],'trials':[{'scenario':'case.witness','action':'compare','entropy':'0x1234','configuration_ordinal':0,'trial_index':0,'trial_ordinal':0,'passed':True,'expectations':{'expectation_count':2,'passed_count':2,'failure_count':0,'failures':[]}}],'skipped_cases':[],'skipped_scenarios':[],'planning_issues':[],'xfails':[]}
 if os.environ.get('FAKE_BAD_REPORT'): report['planning_issue_count']=1
 rows=[]
 for index,label in enumerate(targets):
  xml=events.parent/f'{index}.xml';doc=ET.Element('testsuite');ET.SubElement(doc,'testcase',{'name':'hardware'})
  ET.SubElement(doc,'system-out').text=json.dumps(report)
  xml.write_text(ET.tostring(doc,encoding='unicode'))
  rows.append({'id':{'testResult':{'label':label}},'testResult':{'status':'PASSED','testActionOutput':[{'name':'test.xml','uri':xml.as_uri()}]}})
 events.write_text('\n'.join(json.dumps(row) for row in rows))
 profile=pathlib.Path(next(a.split('=',1)[1] for a in sys.argv if a.startswith('--profile=')))
 producer=os.environ.get('FAKE_BUILD_IN_TEST') and '--noslim_profile' in sys.argv
 profile.write_text(json.dumps({'traceEvents':[{'cat':'action processing','ph':'X','name':'Executing genrule //fake:build' if producer else 'Testing '+targets[0],'args':{'mnemonic':'Genrule' if producer else 'TestRunner'}},{'cat':'action processing','ph':'X','name':'BazelWorkspaceStatusAction stable-status.txt','args':{'mnemonic':'BazelWorkspaceStatusAction'}}]}))
""")

    def hardware_run(self, **extra):
        with (
            mock.patch.dict(os.environ, {**self.environment, **extra}, clear=True),
            mock.patch.object(validation.Path, "cwd", return_value=self.root),
            mock.patch.object(validation.source, "git", return_value=b"fixture-head"),
            mock.patch.object(validation.source, "inputs", return_value={}),
        ):
            return validation.run(self.args, self.root)

    def calls(self):
        return [
            json.loads(line)
            for line in (self.base / "calls.jsonl").read_text().splitlines()
        ]

    def test_hardware_prebuild_and_metadata_are_outside_lease(self):
        self.assertEqual(self.hardware_run(), 0)
        calls = self.calls()
        self.assertEqual(
            [row["argv"][0] for row in calls],
            ["query", "cquery", "build", "info", "cquery", "test"],
        )
        self.assertEqual(
            [row["held"] for row in calls], [False, False, False, False, False, True]
        )
        self.assertIn("--nocache_test_results", calls[-1]["argv"])
        self.assertIn("--local_test_jobs=1", calls[-1]["argv"])
        self.assertIn("--noslim_profile", calls[-1]["argv"])
        result = json.loads((self.args.result_dir / "result.json").read_text())
        self.assertEqual(result["status"], "passed")
        self.assertLessEqual(
            result["phases"]["build"]["end"], result["lock"]["requested"]
        )
        self.assertLessEqual(
            result["lock"]["acquired"], result["phases"]["test"]["start"]
        )
        self.assertLessEqual(
            result["phases"]["test"]["end"], result["lock"]["released"]
        )
        self.assertEqual(len(result["tests"]), 2)
        self.assertEqual(
            result["metadata_actions"][0]["args"]["mnemonic"],
            "BazelWorkspaceStatusAction",
        )
        self.assertEqual(
            next(iter(result["tests"].values()))["reports"][0]["trials"][0]["entropy"],
            "0x1234",
        )
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_execroot_relative_products_need_no_checkout_bazel_symlinks(self):
        self.assertFalse((self.root / "bazel-out").exists())
        self.assertEqual(self.hardware_run(), 0)
        result = json.loads((self.args.result_dir / "result.json").read_text())
        self.assertEqual(result["execution_root"], str(self.execution_root))
        artifact = self.execution_root / "bazel-out/fake/bin/artifact"
        self.assertIn(str(artifact), result["products"])
        self.assertIn(str(artifact) + ".runfiles_manifest", result["products"])
        self.assertTrue(Path(str(artifact) + ".sh").is_file())
        self.assertNotIn(str(artifact) + ".sh", result["products"])
        product_command = result["phases"]["products"]["command"]
        self.assertIn("--output=starlark", product_command)
        self.assertIn(
            '--starlark:expr=providers(target)["DefaultInfo"].files_to_run.executable.path',
            product_command,
        )
        self.assertFalse((self.root / "bazel-out").exists())

    def test_hardware_rejects_bad_manifests_before_any_command(self):
        for index, labels in enumerate(
            (
                [],
                [self.targets[0]] * 2,
                ["//fake:suite"],
                ["//fake:all"],
                ["//fake:*"],
                ["-//fake:test"],
                ["//fake:host_test"],
            )
        ):
            self.manifest.write_text("\n".join(labels))
            self.args.result_dir = self.base / f"bad-manifest-{index}"
            self.assertEqual(self.hardware_run(), 1)
        self.assertFalse((self.base / "calls.jsonl").exists())

    def test_hardware_invalid_metadata_and_analyzed_extra_stop_prebuild(self):
        for name, extra in [
            ("host", {"FAKE_HOST": "1"}),
            ("extra", {"FAKE_EXTRA": "1"}),
        ]:
            self.args.result_dir = self.base / name
            self.assertEqual(self.hardware_run(**extra), 1)
        self.assertNotIn("build", [row["argv"][0] for row in self.calls()])
        self.assertTrue(all(not row["held"] for row in self.calls()))

    def test_failed_prebuild_never_takes_lock_or_runs_hardware(self):
        self.assertEqual(self.hardware_run(FAKE_BUILD_FAIL="1"), 1)
        self.assertEqual(self.calls()[-1]["argv"][0], "build")
        self.assertTrue(all(not row["held"] for row in self.calls()))

    def test_locked_build_or_loom_planning_issue_fails(self):
        for name, extra in [
            ("build", {"FAKE_BUILD_IN_TEST": "1"}),
            ("report", {"FAKE_BAD_REPORT": "1"}),
        ]:
            self.args.result_dir = self.base / name
            self.assertEqual(self.hardware_run(**extra), 1)
            result = json.loads((self.args.result_dir / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("released", result["lock"])

    def test_workspace_metadata_is_retained_but_producers_are_rejected(self):
        profile = self.base / "profile.json"
        test = {
            "cat": "action processing",
            "ph": "X",
            "name": "Testing //fake:test",
            "args": {"mnemonic": "TestRunner"},
        }
        metadata = {
            "cat": "action processing",
            "ph": "X",
            "name": "BazelWorkspaceStatusAction stable-status.txt",
            "args": {"mnemonic": "BazelWorkspaceStatusAction"},
            "dur": 123,
        }
        profile.write_text(json.dumps({"traceEvents": [test, metadata]}))
        self.assertEqual(
            validation.no_locked_build_actions(profile),
            {"tests": [test["name"]], "metadata": [metadata]},
        )
        for producer in (
            {"name": "Executing genrule //fake:build", "args": {"mnemonic": "Genrule"}},
            {"name": "Compiling fake.cc", "args": {"mnemonic": "CppCompile"}},
            {**metadata, "args": {"mnemonic": "Unknown"}},
            {**metadata, "name": "BazelWorkspaceStatusAction unexpected-product"},
            {**test, "args": {"mnemonic": "Genrule"}},
            {**test, "args": {"mnemonic": "Unknown"}},
            {**test, "args": {}},
        ):
            with self.subTest(producer=producer):
                profile.write_text(
                    json.dumps({"traceEvents": [test, {**metadata, **producer}]})
                )
                with self.assertRaisesRegex(ValueError, "contains build actions"):
                    validation.no_locked_build_actions(profile)

    def test_runfile_content_is_in_frozen_product_identity(self):
        artifact = self.base / "product"
        artifact.write_text("launcher")
        manifest = Path(str(artifact) + ".runfiles_manifest")
        manifest.write_text("fixture " + str(self.fixture) + "\n")
        files = self.base / "files"
        files.write_text(str(artifact) + "\n")
        first = validation.hardware_products(files, self.root)
        self.fixture.write_text("changed")
        self.assertNotEqual(first, validation.hardware_products(files, self.root))
        self.fixture.unlink()
        with self.assertRaisesRegex(ValueError, "Missing physical"):
            validation.hardware_products(files, self.root)

    def test_hardware_xml_rejects_shell_only_and_inconsistent_expectations(self):
        document = ET.fromstring("<testsuite><testcase/></testsuite>")
        with self.assertRaisesRegex(ValueError, "no current Loom"):
            validation.hardware_reports(document)
        self.assertEqual(self.hardware_run(), 0)
        report = json.loads((self.args.result_dir / "result.json").read_text())[
            "tests"
        ][self.targets[0]]["reports"][0]
        for update in [
            dict(trial_count=0),
            dict(failed_trial_count=1),
            dict(skipped_case_count=1),
            dict(xfail_count=1),
        ]:
            row = {**report, **update}
            ET.SubElement(document, "system-out").text = json.dumps(row)
            with self.assertRaises(ValueError):
                validation.hardware_reports(document)
            document.remove(document.find("system-out"))
        row = copy.deepcopy(report)
        row["trials"][0]["expectations"]["passed_count"] = 1
        ET.SubElement(document, "system-out").text = json.dumps(row)
        with self.assertRaisesRegex(ValueError, "expectations"):
            validation.hardware_reports(document)

    def test_manifest_change_while_waiting_fails_before_test(self):
        real_flock = fcntl.flock

        def drift(lock, operation):
            if operation == fcntl.LOCK_EX:
                self.manifest.write_text(
                    self.manifest.read_text()
                    + "//lib/fake:new_test_execute_amd_gpu_hardware_test\n"
                )
            return real_flock(lock, operation)

        with mock.patch.object(validation.fcntl, "flock", side_effect=drift):
            self.assertEqual(self.hardware_run(), 1)
        self.assertNotIn("test", [row["argv"][0] for row in self.calls()])
        result = json.loads((self.args.result_dir / "result.json").read_text())
        self.assertIn("while waiting", result["error"])

    def start_helper(self, **extra):
        environment = {
            **os.environ,
            **self.environment,
            **extra,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        return subprocess.Popen(
            [
                sys.executable,
                str(validation.ROOT / "build_tools/loomify-validation.py"),
                "hardware",
                "--bazel",
                str(self.wrapper),
                "--result-dir",
                str(self.args.result_dir),
                "--targets-file",
                str(self.manifest),
                "--gpu-lock",
                str(self.lock),
            ],
            cwd=validation.ROOT,
            env=environment,
        )

    def await_condition(self, condition, child, timeout=10):
        deadline = time.monotonic() + timeout
        while not condition() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(condition())

    def test_hardware_cancellation_retains_lease_until_public_cleanup(self):
        daemon = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        child = self.start_helper(
            FAKE_WAIT="1", FAKE_RETRY_SHUTDOWN="1", FAKE_DAEMON_PID=str(daemon.pid)
        )
        try:
            self.await_condition(lambda: (self.base / "test-started").exists(), child)
            child.send_signal(signal.SIGTERM)
            result_path = self.args.result_dir / "result.json"
            self.await_condition(result_path.exists, child)
            result = json.loads(result_path.read_text())
            self.assertEqual(result["cleanup"]["status"], "unresolved")
            self.assertIsNone(child.poll())
            self.assertIsNone(daemon.poll())
            with self.lock.open("a+") as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (self.base / "allow-cleanup").touch()
            self.assertEqual(child.wait(timeout=8), 1)
            self.assertEqual(daemon.wait(timeout=3), -signal.SIGTERM)
            result = json.loads(result_path.read_text())
            self.assertEqual(result["cleanup"]["status"], "complete")
            self.assertIn("released", result["lock"])
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            if daemon.poll() is None:
                daemon.terminate()
                daemon.wait()

    def test_cancellation_while_waiting_does_not_steal_other_gpu_lease(self):
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            child = self.start_helper(FAKE_ALLOW_BUSY_BUILD="1")
            try:
                self.await_condition(
                    lambda: (
                        self.execution_root
                        / "bazel-out/fake/bin/artifact.runfiles_manifest"
                    ).exists(),
                    child,
                )
                child.send_signal(signal.SIGTERM)
                self.assertEqual(child.wait(timeout=10), 1)
                result = json.loads((self.args.result_dir / "result.json").read_text())
                self.assertEqual(result["cleanup"]["status"], "complete")
                self.assertNotIn("test", [row["argv"][0] for row in self.calls()])
                self.assertNotIn("released", result.get("lock", {}))
                with self.lock.open("a+") as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    unittest.main()
