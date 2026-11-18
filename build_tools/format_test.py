# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""Check the formatter runner's selection, failure, and write boundaries."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest


class FormatRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "checkout"
        self.root.mkdir()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.log = self.base / "calls.jsonl"
        source = Path(__file__).resolve().parents[1]
        (self.root / "build_tools").mkdir()
        for name in (
            "format.sh",
            "format-tools.env",
            "loom-format.sh",
            "pre-commit.sh",
        ):
            shutil.copyfile(
                source / "build_tools" / name, self.root / "build_tools" / name
            )
        self.pins = {}
        for line in (
            (self.root / "build_tools/format-tools.env").read_text().splitlines()
        ):
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                self.pins[key] = value
        self.env = dict(
            os.environ,
            PATH=str(self.bin),
            FORMAT_TEST_LOG=str(self.log),
            HRX_FORMAT_BAZEL_ROOT=str(self.base / "bazel"),
            HRX_FORMAT_HOOK_ROOT=str(self.base / "hooks"),
            HRX_FORMAT_TOOLS=str(self.base / "unused-tools"),
            BAZEL="bazel",
        )
        for tool in (
            "git",
            "dirname",
            "realpath",
            "mktemp",
            "rm",
            "mkdir",
            "bash",
            "flock",
            "sha256sum",
            "chmod",
        ):
            (self.bin / tool).symlink_to(shutil.which(tool))
        self.git("init", "-q")
        for tool in ("buildifier", "clang-format", "black", "bazel"):
            self.fake_tool(tool)

    def git(self, *arguments):
        return subprocess.run(
            [str(self.bin / "git"), *arguments],
            cwd=self.root,
            check=True,
            capture_output=True,
            env=self.env,
        )

    def fake_tool(self, name):
        version = {
            "bazel": "bazel fake",
            "buildifier": "buildifier version: "
            + self.pins["BUILDIFIER_VERSION"]
            + " ",
            "clang-format": "clang-format version " + self.pins["CLANG_FORMAT_VERSION"],
            "black": "black, " + self.pins["BLACK_VERSION"] + " (compiled: yes)",
        }[name]
        script = self.bin / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            f"name = {name!r}\n"
            "if sys.argv[1:] == ['--version']:\n"
            f"    print('wrong version' if os.environ.get('FORMAT_TEST_BAD') == name else {version!r})\n"
            "else:\n"
            "    with open(os.environ['FORMAT_TEST_LOG'], 'a') as stream:\n"
            "        stream.write(json.dumps([name, sys.argv[1:]]) + '\\n')\n"
            "    sys.exit(1 if os.environ.get('FORMAT_TEST_DRIFT') == name else 0)\n"
        )
        script.chmod(0o755)

    def track(self, name, content="unchanged\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.git("add", "--", name)
        return path

    def run_format(self, *arguments):
        return subprocess.run(
            ["/bin/bash", str(self.root / "build_tools/format.sh"), *arguments],
            cwd=self.base,
            env=self.env,
            text=True,
            capture_output=True,
        )

    def calls(self):
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def test_help_needs_no_tools_and_invalid_options_do_not_run_them(self):
        for name in ("black", "buildifier", "clang-format"):
            (self.bin / name).unlink()
        self.assertEqual(self.run_format("--help").returncode, 0)
        self.assertEqual(self.run_format("--wrong").returncode, 2)
        self.assertEqual(self.run_format("--check", "extra").returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_missing_or_wrong_version_fails_before_fix(self):
        source = self.track("test.py")
        self.env["FORMAT_TEST_BAD"] = "black"
        self.assertEqual(self.run_format("--fix").returncode, 2)
        self.assertEqual(self.calls(), [])
        self.assertEqual(source.read_text(), "unchanged\n")
        del self.env["FORMAT_TEST_BAD"]
        (self.bin / "black").unlink()
        self.assertEqual(self.run_format("--fix").returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_tracked_selection_preserves_argument_boundaries_and_exceptions(self):
        self.track("BUILD.bazel")
        self.track("odd name\npart.cc")
        self.track("normal.py")
        self.track("kernel.cxx")
        self.track("kernel.loom")
        self.track("odd name\npart.loom-test")
        self.track("vendor/skip.py")
        self.track("loom-pipeline/recognition/moonshine/preparer/prepare_assets.py")
        self.track("loom-pipeline/speech/pocket_tts/preparer/prepare_assets.py")
        (self.root / "untracked.py").write_text("untracked\n")
        (self.root / "link.py").symlink_to(self.base / "outside.py")
        self.git("add", "link.py")
        result = self.run_format()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = dict(self.calls())
        self.assertEqual(calls["buildifier"][-1], "BUILD.bazel")
        self.assertIn("-buildifier_disable=listsort", calls["buildifier"])
        self.assertIn("-buildifier_disable=label", calls["buildifier"])
        self.assertEqual(
            calls["clang-format"][-2:], ["kernel.cxx", "odd name\npart.cc"]
        )
        self.assertEqual(calls["black"][-1], "normal.py")
        self.assertIn("--check", calls["black"])
        self.assertIn("--dry-run", calls["clang-format"])
        self.assertEqual(
            calls["bazel"][-3:],
            [
                "--check",
                str(self.root / "kernel.loom"),
                str(self.root / "odd name\npart.loom-test"),
            ],
        )
        self.assertIn("@hrx_system//loom/src/loom/tools/loom-format", calls["bazel"])

    def test_check_reports_drift_without_source_or_index_writes(self):
        target = self.track("test.py")
        index = (self.root / ".git/index").read_bytes()
        self.env["FORMAT_TEST_DRIFT"] = "black"
        self.assertEqual(self.run_format("--check").returncode, 1)
        self.assertEqual(target.read_text(), "unchanged\n")
        self.assertEqual((self.root / ".git/index").read_bytes(), index)

    def test_fix_uses_each_tool_write_mode(self):
        for name in ("BUILD", "test.c", "test.py", "test.loom"):
            self.track(name)
        result = self.run_format("--fix")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = dict(self.calls())
        self.assertIn("-mode=fix", calls["buildifier"])
        self.assertIn("-buildifier_disable=listsort", calls["buildifier"])
        self.assertIn("-buildifier_disable=label", calls["buildifier"])
        self.assertIn("-i", calls["clang-format"])
        self.assertNotIn("--check", calls["black"])
        self.assertIn("--in-place", calls["bazel"])

    def run_hook(self, timeout=None):
        # Model the environment Git supplies, including its index override.
        env = dict(self.env, GIT_INDEX_FILE=str(self.root / ".git/index"))
        return subprocess.run(
            ["/bin/bash", str(self.root / "build_tools/pre-commit.sh")],
            cwd=self.root,
            env=env,
            timeout=timeout,
            text=True,
            capture_output=True,
        )

    def test_hook_checks_index_and_preserves_unstaged_content(self):
        self.track(
            "build_tools/format.sh", "#!/bin/bash\n! /bin/grep -q bad source.loom\n"
        )
        target = self.track("source.loom", "good\n")
        target.write_text("bad unstaged\n")
        index = (self.root / ".git/index").read_bytes()
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_text(), "bad unstaged\n")
        self.assertEqual((self.root / ".git/index").read_bytes(), index)
        self.assertEqual(list((self.base / "hooks").glob("*/staged")), [])

    def test_hook_blocks_bad_index_even_when_worktree_is_fixed(self):
        self.track(
            "build_tools/format.sh", "#!/bin/bash\n! /bin/grep -q bad source.loom\n"
        )
        target = self.track("source.loom", "bad\n")
        target.write_text("good unstaged\n")
        index = (self.root / ".git/index").read_bytes()
        result = self.run_hook()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Commit blocked by formatting", result.stderr)
        self.assertEqual(target.read_text(), "good unstaged\n")
        self.assertEqual((self.root / ".git/index").read_bytes(), index)
        self.assertEqual(list((self.base / "hooks").glob("*/staged")), [])

    def test_background_formatter_does_not_block_next_hook(self):
        pidfile = self.base / "background.pid"
        self.env["FORMAT_TEST_BACKGROUND_PID"] = str(pidfile)
        self.track(
            "build_tools/format.sh",
            "#!/bin/bash\n"
            'if [[ ! -f "$FORMAT_TEST_BACKGROUND_PID" ]]; then\n'
            "  /bin/sleep 60 >/dev/null 2>&1 &\n"
            '  echo "$!" > "$FORMAT_TEST_BACKGROUND_PID"\n'
            "fi\n",
        )
        try:
            first = self.run_hook(timeout=3)
            self.assertEqual(first.returncode, 0, first.stderr)
            try:
                second = self.run_hook(timeout=2)
            except subprocess.TimeoutExpired:
                self.fail(
                    "Background formatter retained hook lock and blocked next hook"
                )
            self.assertEqual(second.returncode, 0, second.stderr)
        finally:
            if pidfile.exists():
                os.kill(int(pidfile.read_text()), signal.SIGTERM)

    def test_hook_uses_raw_blobs_without_smudge_filters(self):
        self.git("config", "filter.replace.smudge", "/bin/sed s/good/bad/g")
        self.git("config", "filter.replace.clean", "/bin/cat")
        self.track(".gitattributes", "source.loom filter=replace\n")
        self.track(
            "build_tools/format.sh", "#!/bin/bash\n! /bin/grep -q bad source.loom\n"
        )
        self.track("source.loom", "good\n")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_bazel_fails_before_other_tools_write(self):
        self.track("source.loom")
        self.track("test.py")
        (self.bin / "bazel").unlink()
        result = self.run_format("--fix")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Missing Bazel", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_git_commit_invokes_tracked_hook(self):
        source = Path(__file__).resolve().parents[1]
        hook = self.track(
            ".githooks/pre-commit", (source / ".githooks/pre-commit").read_text()
        )
        hook.chmod(0o755)
        self.git("add", ".githooks/pre-commit")
        self.track(
            "build_tools/pre-commit.sh",
            (source / "build_tools/pre-commit.sh").read_text(),
        )
        self.track(
            "build_tools/format.sh", "#!/bin/bash\n! /bin/grep -q bad source.loom\n"
        )
        self.track("source.loom", "bad\n")
        self.git("config", "core.hooksPath", ".githooks")
        self.git("config", "user.name", "Formatter test")
        self.git("config", "user.email", "formatter@example.invalid")
        result = subprocess.run(
            [str(self.bin / "git"), "commit", "-qm", "fixture"],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Commit blocked by formatting", result.stderr)
        self.track("source.loom", "good\n")
        self.git("commit", "-qm", "fixture")

    def test_loom_failure_propagates(self):
        self.track("source.loom")
        self.env["FORMAT_TEST_DRIFT"] = "bazel"
        self.assertEqual(self.run_format("--check").returncode, 1)

    def test_git_selector_failure_rejects_and_cleans_list_before_formatting(self):
        target = self.track("test.py")
        temporary = self.base / "temporary"
        temporary.mkdir()
        self.env["TMPDIR"] = str(temporary)
        (self.root / ".git/index").write_bytes(b"invalid Git index")
        result = self.run_format("--fix")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Cannot list tracked formatter inputs", result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertEqual(target.read_text(), "unchanged\n")
        self.assertEqual(list(temporary.iterdir()), [])

    def test_external_parent_symlink_rejects_before_any_formatter_writes(self):
        self.track("first.py")
        path = self.track("nested/escape.py")
        path.unlink()
        path.parent.rmdir()
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "escape.py").write_text("outside\n")
        path.parent.symlink_to(outside, target_is_directory=True)
        result = self.run_format("--fix")
        self.assertEqual(result.returncode, 2)
        self.assertIn("outside this checkout", result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertEqual((outside / "escape.py").read_text(), "outside\n")


if __name__ == "__main__":
    unittest.main()
