#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Run the reviewed early Loomify CPU batch in a caller-owned environment."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import site
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

from ci_tools import local_ci_source as source

# These tests materialize/inspect compiler IR; they do not create GPU devices.
NATIVE_TARGETS = (
    "//lib/programs/projection/testing:materialization_test",
    "//lib/programs/ffn/testing:capability_test",
    "//lib/kernels/embedding/testing:materialization_test",
    "//lib/architectures/llama:configured_model_test",
)
FIXTURES = (
    "LOOMIFY_TEST_MINICPM_GGUF",
    "LOOMIFY_TEST_GGUF",
    "LOOMIFY_TEST_HYPERCLOVAX_GGUF",
)
ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    native = commands.add_parser("native", help="reviewed CPU-only admission tests")
    native.add_argument("--bazel", type=Path, required=True)
    native.add_argument("--result-dir", type=Path, required=True)
    native.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    native.add_argument("--input-record", type=Path)
    native.add_argument("--rc", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def preflight(args, root):
    if Path.cwd().resolve() != root:
        raise ValueError("Run from this helper's checkout root")
    if (
        sys.prefix == sys.base_prefix
        or site.ENABLE_USER_SITE
        or "include-system-site-packages = false"
        not in (Path(sys.prefix) / "pyvenv.cfg").read_text()
    ):
        raise ValueError("Use an isolated task venv with user site disabled")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        raise ValueError("Remove inherited PYTHONPATH and PYTHONHOME overrides")
    state = Path(os.environ.get("LOOM_SERVE_DIR", "."))
    if (
        not state.is_absolute()
        or state == Path("/")
        or state.resolve().is_relative_to(root)
    ):
        raise ValueError("LOOM_SERVE_DIR must be an absolute external task-owned path")
    if (
        not args.bazel.is_absolute()
        or not args.bazel.is_file()
        or not os.access(args.bazel, os.X_OK)
    ):
        raise ValueError("--bazel must name an absolute executable wrapper")
    for name in FIXTURES:
        if not os.environ.get(name) or not Path(os.environ[name]).is_file():
            raise ValueError(f"Required fixture file is missing: {name}")
    for path in [*args.rc, *([args.input_record] if args.input_record else [])]:
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                f"Evidence input must be an absolute existing file: {path}"
            )


def read_results(events, started_ns):
    """Associate fresh XML with this invocation's uncached Bazel test events."""
    outputs = {}
    for line in events.read_text().splitlines():
        event = json.loads(line)
        test = event.get("testResult")
        if test is None:
            continue
        label = event.get("id", {}).get("testResult", {}).get("label")
        if label not in NATIVE_TARGETS:
            continue
        if label in outputs:
            raise ValueError(f"Unexpected multiple test attempts/shards: {label}")
        if (
            test.get("status") != "PASSED"
            or test.get("cachedLocally")
            or test.get("cachedRemotely")
        ):
            raise ValueError(f"Test was failed or cached: {label}")
        matches = [
            v for v in test.get("testActionOutput", []) if v.get("name") == "test.xml"
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing unique current XML event: {label}")
        uri = urlparse(matches[0].get("uri", ""))
        if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
            raise ValueError(f"XML must be a local file: {label}")
        path = Path(unquote(uri.path))
        if not path.is_file() or path.stat().st_mtime_ns < started_ns:
            raise ValueError(f"Missing or stale XML: {label}")
        document = ET.parse(path).getroot()
        cases = list(document.iter("testcase"))
        bad = any(
            int(node.get(k, "0"))
            for node in document.iter()
            for k in ("failures", "errors", "skipped", "disabled")
        )
        if (
            not cases
            or bad
            or any(
                c.find("failure") is not None
                or c.find("error") is not None
                or c.find("skipped") is not None
                or c.get("status") == "notrun"
                for c in cases
            )
        ):
            raise ValueError(f"Empty, failed, skipped or disabled XML: {label}")
        outputs[label] = {
            "xml": identity(path),
            "cases": [
                {"class": c.get("classname", ""), "name": c.get("name", "")}
                for c in cases
            ],
            "count": len(cases),
            "failures": 0,
            "errors": 0,
            "skips": 0,
        }
    if set(outputs) != set(NATIVE_TARGETS):
        raise ValueError(
            "Missing current results for: "
            + ", ".join(sorted(set(NATIVE_TARGETS) - set(outputs)))
        )
    return outputs


def run(args, root=ROOT):
    root = root.resolve()
    destination = args.result_dir
    if (
        not destination.is_absolute()
        or destination.resolve().is_relative_to(root)
        or destination.exists()
    ):
        raise ValueError(
            "--result-dir must be a new absolute directory outside the checkout"
        )
    destination.mkdir(parents=True)
    result = {
        "stage": args.stage,
        "status": "failed",
        "jobs": args.jobs,
        "targets": list(NATIVE_TARGETS),
    }
    child = None
    interrupted = None
    prior_handlers = {}

    def cancel(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        result["cleanup"] = {
            "status": "unresolved",
            "error": "Cancellation requires public wrapper shutdown before output reuse/removal",
        }
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)

    try:
        preflight(args, root)
        result["source"] = {
            "head": source.git(root, "rev-parse", "HEAD").decode().strip(),
            "digest": source.digest(source.inputs(root)),
        }
        result["pin_inputs"] = {
            name: identity(root / name)
            for name in ("MODULE.bazel", "MODULE.bazel.lock", ".bazelversion")
        }
        result["wrapper"] = identity(args.bazel)
        result["explicit_rc"] = [identity(path) for path in args.rc]
        result["input_record"] = (
            identity(args.input_record) if args.input_record else None
        )
        result["fixture_paths"] = {name: os.environ[name] for name in FIXTURES}
        result["resource_limits"] = {
            name: list(resource.getrlimit(getattr(resource, name)))
            for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS")
        }
        command = [
            str(args.bazel),
            "test",
            "--config=opt",
            f"--local_test_jobs={args.jobs}",
            "--nocache_test_results",
            f"--build_event_json_file={destination / 'events.jsonl'}",
            *[f"--test_env={name}" for name in FIXTURES],
            *NATIVE_TARGETS,
        ]
        result["command"] = command
        result["started"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result["started_ns"] = time.time_ns()
        result["monotonic_start"] = time.monotonic()
        for signum in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[signum] = signal.signal(signum, cancel)
        with (destination / "command.log").open("wb") as log:
            child = subprocess.Popen(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while child.poll() is None:
                try:
                    child.wait(timeout=0.2 if interrupted is None else 10)
                except subprocess.TimeoutExpired:
                    if interrupted is not None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        result["exit"] = child.returncode
        result["monotonic_end"] = time.monotonic()
        result["elapsed_seconds"] = result["monotonic_end"] - result["monotonic_start"]
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        result["children"] = {
            "peak_rss_kib": usage.ru_maxrss,
            "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime,
        }
        if interrupted is not None:
            # Bazel's daemon/executors may be detached from the client group.
            # Only this caller-owned wrapper may reconcile its output publicly.
            cleanup_command = [str(args.bazel), "shutdown"]
            result["cleanup"] = {"status": "unresolved", "command": cleanup_command}
            with (destination / "cleanup.log").open("wb") as cleanup_log:
                cleanup = subprocess.Popen(
                    cleanup_command,
                    cwd=root,
                    stdout=cleanup_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    cleanup.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(cleanup.pid, signal.SIGTERM)
                    try:
                        cleanup.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(cleanup.pid, signal.SIGKILL)
                        cleanup.wait()
                    result["cleanup"][
                        "error"
                    ] = "Public shutdown timed out; owner must reconcile output before reuse/removal"
                else:
                    result["cleanup"]["exit"] = cleanup.returncode
                    if cleanup.returncode == 0:
                        result["cleanup"]["status"] = "complete"
                    else:
                        result["cleanup"][
                            "error"
                        ] = "Public shutdown failed; owner must reconcile output before reuse/removal"
            raise ValueError(
                f"Interrupted by signal {interrupted}; public cleanup {result['cleanup']['status']}"
            )
        if child.returncode:
            raise ValueError(f"Native command failed: exit {child.returncode}")
        result["tests"] = read_results(
            destination / "events.jsonl", result["started_ns"]
        )
        if source.digest(source.inputs(root)) != result["source"]["digest"]:
            raise ValueError("Source changed during native validation")
        if (
            identity(args.bazel) != result["wrapper"]
            or [identity(path) for path in args.rc] != result["explicit_rc"]
        ):
            raise ValueError("Wrapper or explicit rc changed during validation")
        result["status"] = "passed"
    except (ValueError, OSError, subprocess.SubprocessError, ET.ParseError) as exc:
        result["error"] = str(exc)
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
        result["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "passed" else 1


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
