# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""Exercise formatter reuse without building or borrowing an SDK."""

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from build_tools import loom_format_cache as cache


class FormatterCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.root = self.base / "cache"
        self.log = self.base / "calls"
        self.expected = {"pin": "a" * 40, "recipe": "recipe", "bazel_version": "9.3.2"}
        self.environment = mock.patch.dict(
            os.environ,
            HRX_FORMATTER_CACHE="on",
            HRX_FORMATTER_CACHE_ROOT=str(self.root),
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.identity = mock.patch.object(cache, "identity", return_value=self.expected)
        self.identity.start()
        self.addCleanup(self.identity.stop)
        self.launcher = mock.patch.object(
            cache, "canonical_launcher", return_value=Path("/canonical/bazel")
        )
        self.launcher.start()
        self.addCleanup(self.launcher.stop)
        self.population = mock.patch.object(
            cache, "populate", side_effect=self.populate
        )
        self.populate_mock = self.population.start()
        self.addCleanup(self.population.stop)

    def populate(self, source, destination, expected, launcher):
        with self.log.open("a") as stream:
            stream.write("populate\n")
        tool = destination / "loom-format"
        tool.write_text(
            f"#!{sys.executable}\n"
            "import json,sys\n"
            f"with open({str(self.log)!r}, 'a') as stream:\n"
            " stream.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "sys.exit(7 if '--fail' in sys.argv else 0)\n"
        )
        tool.chmod(0o755)
        (destination / "manifest.json").write_bytes(
            cache.encoded(
                {"identity": expected, "executable_sha256": cache.digest(tool)}
            )
        )

    def invoke(self, *args):
        return cache.main(list(args), self.source)

    def entry(self):
        return self.root / hashlib.sha256(cache.encoded(self.expected)).hexdigest()

    def test_hit_keeps_arguments_and_never_calls_operational_wrapper(self):
        with mock.patch.object(
            cache, "fallback", side_effect=AssertionError("wrapper")
        ):
            self.invoke("--check", "odd name\npart.loom")
            self.invoke("--check", "odd name\npart.loom")
        self.assertEqual(self.populate_mock.call_count, 1)
        self.assertEqual(
            json.loads(self.log.read_text().splitlines()[-1]),
            ["--check", "odd name\npart.loom"],
        )
        self.assertEqual(
            sorted(p.name for p in self.entry().iterdir()),
            ["loom-format", "manifest.json"],
        )

    def test_preflight_does_not_build(self):
        self.invoke("--cache-preflight")
        self.populate_mock.assert_not_called()
        self.assertFalse(self.entry().exists())

    def test_absent_setup_and_cache_off_use_uncached_fallback(self):
        with (
            mock.patch.object(cache, "canonical_launcher", return_value=None),
            mock.patch.object(cache, "fallback") as fallback,
        ):
            self.invoke("--check")
            fallback.assert_called_once_with(self.source, ["--check"], False)
        with (
            mock.patch.dict(os.environ, HRX_FORMATTER_CACHE="off"),
            mock.patch.object(cache, "fallback") as fallback,
        ):
            self.invoke("--check")
            fallback.assert_called_once()
        self.populate_mock.assert_not_called()

    def test_invalid_launcher_never_falls_back_or_publishes(self):
        with (
            mock.patch.object(
                cache, "canonical_launcher", side_effect=ValueError("bad launcher")
            ),
            mock.patch.object(cache, "fallback") as fallback,
        ):
            with self.assertRaisesRegex(ValueError, "bad launcher"):
                self.invoke("--check")
            fallback.assert_not_called()
        self.assertFalse(self.entry().exists())

    def test_corrupt_manifest_or_executable_and_symlinks_never_execute(self):
        self.invoke("--check")
        entry = self.entry()
        for name in ("manifest.json", "loom-format"):
            (entry / name).write_text("corrupt")
            self.invoke("--check")
        original = self.base / "saved"
        shutil.copyfile(entry / "loom-format", original)
        (entry / "loom-format").unlink()
        (entry / "loom-format").symlink_to(original)
        self.invoke("--check")
        self.assertFalse((entry / "loom-format").is_symlink())
        self.assertEqual(self.populate_mock.call_count, 4)

    def test_changed_identity_builds_new_key(self):
        self.invoke("--check")
        first = self.entry()
        self.expected["pin"] = "b" * 40
        self.invoke("--check")
        self.assertNotEqual(first, self.entry())
        self.assertTrue(first.exists())
        self.assertEqual(self.populate_mock.call_count, 2)

    def test_formatter_failure_is_not_rebuilt_or_retried(self):
        with self.assertRaises(subprocess.CalledProcessError) as failed:
            self.invoke("--fail")
        self.assertEqual(failed.exception.returncode, 7)
        self.assertEqual(self.populate_mock.call_count, 1)
        self.assertEqual(self.log.read_text().splitlines(), ["populate", '["--fail"]'])

    def test_interrupted_population_has_no_partial_entry(self):
        with mock.patch.object(cache, "populate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke("--check")
        self.assertFalse(self.entry().exists())
        self.assertFalse(list(self.root.glob("entry-*")))

    @unittest.skipUnless(
        sys.platform == "linux", "uses fork to model two independent callers"
    )
    def test_concurrent_misses_publish_once(self):
        context = multiprocessing.get_context("fork")
        start = context.Event()

        def worker():
            start.wait()
            self.invoke("--check")

        processes = [context.Process(target=worker) for _ in range(2)]
        for process in processes:
            process.start()
        start.set()
        try:
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
        self.assertEqual(self.log.read_text().splitlines().count("populate"), 1)

    def test_cache_inside_checkout_and_relative_path_rejected(self):
        for path in ("relative", str(self.source / "cache")):
            with (
                self.subTest(path=path),
                mock.patch.dict(os.environ, HRX_FORMATTER_CACHE_ROOT=path),
            ):
                with self.assertRaises(ValueError):
                    self.invoke("--check")

    def test_input_change_during_population_does_not_publish(self):
        self.identity.stop()
        with mock.patch.object(
            cache,
            "identity",
            side_effect=[self.expected, dict(self.expected, pin="changed")],
        ):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.invoke("--check")
        self.assertFalse(self.entry().exists())

    def test_warm_cache_rejects_explicit_invalid_launcher(self):
        self.invoke("--check")
        with mock.patch.object(
            cache, "canonical_launcher", side_effect=ValueError("invalid")
        ):
            with self.assertRaisesRegex(ValueError, "invalid"):
                self.invoke("--check")

    def test_population_uses_canonical_recipe_and_retires_output(self):
        self.population.stop()
        expected = dict(
            self.expected,
            remote="https://example.com/hrx.git",
            compilers=[{"path": "/usr/bin/clang"}, {"path": "/usr/bin/clang++"}],
        )
        staging = self.base / "entry" / "complete"
        staging.mkdir(parents=True)
        output = self.base / "execution" / "bazel-out" / "formatter"
        output.parent.mkdir(parents=True)
        output.write_text("standalone")
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            if "show_repo" in argv:
                return (expected["pin"] + " " + expected["remote"]).encode()
            if "cquery" in argv:
                return b"bazel-out/formatter\n"
            if "execution_root" in argv:
                return str(output.parents[1]).encode()
            return b""

        ldd = subprocess.CompletedProcess(
            [], 0, "libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6 (0x0)\n", ""
        )
        with (
            mock.patch.object(cache, "run", side_effect=runner),
            mock.patch.object(cache.subprocess, "run", return_value=ldd),
            mock.patch.dict(
                os.environ,
                BAZEL="/hostile/wrapper",
                CC="/hostile/compiler",
                PYTHONPATH="/hostile/site",
            ),
        ):
            cache.populate(self.source, staging, expected, Path("/canonical/bazel"))
        build, options = calls[0]
        self.assertEqual(build[0:2], ["/canonical/bazel", "--ignore_all_rc_files"])
        self.assertIn("--repo_env=CC=/usr/bin/clang", build)
        self.assertIn("--@hrx_system//loom/config:shared_tools=false", build)
        # Parsing the full maintained inventory needs the AMD compiler dialects,
        # even though formatter execution needs only a host task driver.
        self.assertIn("--@hrx_system//loom/config/arch:enable=amdgpu,vm,x86", build)
        self.assertIn("--@hrx_system//runtime/config/hal:drivers=task", build)
        self.assertIn("--@hrx_system//loom/config/execute:enable=iree_vm", build)
        self.assertEqual(options["env"]["PYTHONPATH"], "")
        self.assertNotIn("CC", options["env"])
        self.assertEqual(calls[-1][0][-1], "shutdown")
        self.assertEqual(list(self.base.glob("population-*")), [])
        self.assertTrue(cache.verified(staging, expected))

    def test_warm_reuse_prunes_obsolete_entry_after_reader_releases(self):
        self.invoke("--check")
        original = self.entry()
        with cache.locked(self.root / (original.name + ".lock"), exclusive=False):
            for pin in ("b", "c"):
                self.expected["pin"] = pin * 40
                self.invoke("--check")
            self.assertTrue(original.exists())
        self.assertEqual(self.populate_mock.call_count, 3)
        self.invoke("--check")
        self.assertEqual(self.populate_mock.call_count, 3)
        self.assertFalse(original.exists())
        self.assertEqual(len([p for p in self.root.iterdir() if p.is_dir()]), 2)


class FormatterIdentityTest(unittest.TestCase):
    def test_actual_identity_tracks_snapshot_inputs_recipe_and_host(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "MODULE.bazel").write_text(
                'git_override(module_name = "hrx", commit = "'
                + "a" * 40
                + '", remote = "https://example.com/hrx.git",)'
            )
            (source / ".bazelversion").write_text("9.3.2\n")
            (source / "MODULE.bazel.lock").write_text("lock")
            compiler = source / "compiler"
            compiler.write_text("fixture-compiler")
            platform_key = ("9.3.2", cache.platform.system(), cache.platform.machine())
            with (
                mock.patch.object(cache, "COMPILERS", [str(compiler), str(compiler)]),
                mock.patch.dict(cache.BAZEL_DIGESTS, {platform_key: "fixture-bazel"}),
            ):
                expected = cache.identity(source)
                self.assertIsNotNone(expected)
                (source / "MODULE.bazel.lock").write_text("changed")
                self.assertNotEqual(cache.identity(source), expected)
                (source / ".bazelrc.local").write_text(
                    "common --override_module=hrx=/local"
                )
                self.assertIsNone(cache.identity(source))

    def test_published_launcher_digest_checked_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary) / "bazel"
            launcher.write_text(f"#!{sys.executable}\nprint('bazel 9.3.2')\n")
            launcher.chmod(0o755)
            expected = {"bazel_sha256": "bad", "bazel_version": "9.3.2"}
            with mock.patch.dict(os.environ, HRX_FORMATTER_BAZEL=str(launcher)):
                with self.assertRaisesRegex(ValueError, "verified published"):
                    cache.canonical_launcher(expected)
                expected["bazel_sha256"] = cache.digest(launcher)
                self.assertEqual(cache.canonical_launcher(expected), launcher)
                expected["bazel_version"] = "wrong"
                with self.assertRaisesRegex(ValueError, "does not match"):
                    cache.canonical_launcher(expected)


if __name__ == "__main__":
    unittest.main()
