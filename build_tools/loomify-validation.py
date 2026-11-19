#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Run explicit Loomify validation batches in a caller-owned environment."""

import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
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
HARDWARE_TAGS = {
    "exclusive-if-local",
    "resource_group:gpu-device",
    "loom-executor=hardware",
    "loom-target-class=gpu",
    "loom-execution-profile=amd_gpu_hardware",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    native = commands.add_parser("native", help="reviewed CPU-only admission tests")
    native.add_argument("--bazel", type=Path, required=True)
    native.add_argument("--result-dir", type=Path, required=True)
    native.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    native.add_argument("--input-record", type=Path)
    native.add_argument("--rc", type=Path, action="append", default=[])
    hardware = commands.add_parser("hardware", help="prebuild explicit GPU tests")
    hardware.add_argument("--bazel", type=Path, required=True)
    hardware.add_argument("--result-dir", type=Path, required=True)
    hardware.add_argument("--targets-file", type=Path, required=True)
    hardware.add_argument("--gpu-lock", type=Path, required=True)
    hardware.add_argument("--input-record", type=Path)
    hardware.add_argument("--rc", type=Path, action="append", default=[])
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
    for name in FIXTURES if args.stage == "native" else ():
        if not os.environ.get(name) or not Path(os.environ[name]).is_file():
            raise ValueError(f"Required fixture file is missing: {name}")
    for path in [*args.rc, *([args.input_record] if args.input_record else [])]:
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                f"Evidence input must be an absolute existing file: {path}"
            )


def read_results(events, started_ns, targets=NATIVE_TARGETS, hardware=False):
    """Associate fresh XML with this invocation's uncached Bazel test events."""
    outputs = {}
    for line in events.read_text().splitlines():
        event = json.loads(line)
        test = event.get("testResult")
        if test is None:
            continue
        label = event.get("id", {}).get("testResult", {}).get("label")
        if label not in targets:
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
        if hardware:
            outputs[label]["reports"] = hardware_reports(document)
    if set(outputs) != set(targets):
        raise ValueError(
            "Missing current results for: "
            + ", ".join(sorted(set(targets) - set(outputs)))
        )
    return outputs


def hardware_targets(path):
    if not path.is_absolute() or not path.is_file():
        raise ValueError("--targets-file must be an absolute existing manifest")
    targets = path.read_text().splitlines()
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("Hardware manifest must be nonempty with no duplicates")
    for label in targets:
        if not re.fullmatch(r"//[\w./-]+:[\w.+=,@~-]+", label) or not label.endswith(
            "_execute_amd_gpu_hardware_test"
        ):
            raise ValueError(f"Expected a concrete supported hardware leaf: {label}")
    return targets


def hardware_metadata(path, targets):
    rules = ET.parse(path).getroot().findall("rule")
    if len(rules) != len(targets) or {r.get("name") for r in rules} != set(targets):
        raise ValueError("Hardware metadata does not match the explicit manifest")
    for rule in rules:
        tags = {
            node.get("value") for node in rule.findall("./list[@name='tags']/string")
        }
        if rule.get("class") != "sh_test" or not HARDWARE_TAGS <= tags:
            raise ValueError(f"Unsupported hardware metadata: {rule.get('name')}")


def hardware_reports(document):
    """The Bazel shell testcase alone does not prove Loom trials executed."""
    reports = []
    for node in document.iter("system-out"):
        for line in (node.text or "").splitlines():
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row.get("format") != "loom.test.v0":
                continue
            for key in (
                "failed_sample_count",
                "failed_trial_count",
                "skipped_case_count",
                "skipped_scenario_count",
                "planning_issue_count",
                "xfail_count",
                "xpass_count",
                "xfail_mismatch_count",
            ):
                if row.get(key) != 0:
                    raise ValueError(f"Loom report has a nonzero/missing {key}")
            trials, samples = row.get("trials", []), row.get("samples", [])
            if (
                any(
                    not isinstance(row.get(key), int) or row[key] < 0
                    for key in ("case_count", "scenario_count")
                )
                or row.get("case_count", 0) + row.get("scenario_count", 0) == 0
                or row.get("trial_count") != len(trials)
                or row.get("sample_count") != len(samples)
                or not trials
                and not samples
                or any(
                    row.get(key)
                    for key in (
                        "skipped_cases",
                        "skipped_scenarios",
                        "planning_issues",
                        "xfails",
                    )
                )
            ):
                raise ValueError("Empty or inconsistent Loom execution report")
            for trial in [*trials, *samples]:
                checks = trial.get("expectations", {})
                if (
                    trial.get("passed") is not True
                    or checks.get("failure_count") != 0
                    or checks.get("failures") != []
                    or not isinstance(checks.get("expectation_count"), int)
                    or checks["expectation_count"] <= 0
                    or checks.get("passed_count") != checks["expectation_count"]
                ):
                    raise ValueError("Failed or incomplete Loom trial expectations")
            reports.append(row)
    if not reports:
        raise ValueError("Hardware XML has no current Loom execution report")
    return reports


def hardware_products(files, execution_root):
    """Freeze selected launchers and every physical runfile once, outside the lease."""
    products = {}
    for name in files.read_text().splitlines():
        artifact = Path(name)
        if not artifact.is_absolute():
            artifact = execution_root / artifact
        manifest = Path(str(artifact) + ".runfiles_manifest")
        if not artifact.is_file() or not manifest.is_file():
            raise ValueError(f"Missing built launcher/runfiles manifest: {artifact}")
        for item in (artifact, manifest):
            products[str(item)] = identity(item)
        for line in manifest.read_text().splitlines():
            escaped = line.startswith(" ")
            fields = line[1:].split(" ", 1) if escaped else line.split(" ", 1)
            if len(fields) != 2 or not fields[1]:
                continue  # An empty runfile is represented in the manifest itself.
            physical = fields[1]
            if escaped:
                physical = re.sub(
                    r"\\([snb])",
                    lambda m: {"s": " ", "n": "\n", "b": "\\"}[m[1]],
                    physical,
                )
            path = Path(physical)
            if not path.is_absolute():
                path = execution_root / path
            resolved = str(path.resolve())
            if resolved not in products:
                if not path.is_file():
                    raise ValueError(f"Missing physical runfile: {path}")
                products[resolved] = identity(path)
    if not products:
        raise ValueError("The selected hardware build produced no launchers")
    return products


def no_locked_build_actions(path):
    """Reject product-building actions in Bazel's full public JSON trace."""
    opener = gzip.open if path.read_bytes()[:2] == b"\x1f\x8b" else open
    with opener(path, "rt") as stream:
        profile = json.load(stream)
    actions = [
        row
        for row in profile.get("traceEvents", [])
        if row.get("cat") == "action processing" and row.get("ph") == "X"
    ]
    tests = [
        row
        for row in actions
        if row.get("name", "").startswith("Testing ")
        and row.get("args", {}).get("mnemonic") == "TestRunner"
    ]
    metadata = [
        row
        for row in actions
        if row.get("name") == "BazelWorkspaceStatusAction stable-status.txt"
        and row.get("args", {}).get("mnemonic") == "BazelWorkspaceStatusAction"
    ]
    if not tests or len(tests) + len(metadata) != len(actions):
        raise ValueError(
            "Locked test profile is missing test actions or contains build actions"
        )
    return {"tests": [row["name"] for row in tests], "metadata": metadata}


def run_hardware(args, root=ROOT):
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
    result = {"stage": "hardware", "status": "failed", "phases": {}}
    child = None
    interrupted = None
    lock = None
    locked = False
    prior_handlers = {}

    def save():
        (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def cancel(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        else:
            raise InterruptedError(f"Interrupted by signal {signum}")

    def invoke(name, command):
        nonlocal child
        phase = {"command": command, "start": time.monotonic()}
        result["phases"][name] = phase
        with (
            (destination / f"{name}.stdout").open("wb") as output,
            (destination / f"{name}.log").open("wb") as log,
        ):
            child = subprocess.Popen(
                command, cwd=root, stdout=output, stderr=log, start_new_session=True
            )
            while child.poll() is None:
                try:
                    child.wait(timeout=0.2 if interrupted is None else 10)
                except subprocess.TimeoutExpired:
                    if interrupted is not None:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        phase.update({"end": time.monotonic(), "exit": child.returncode})
        child = None
        if interrupted is not None:
            raise InterruptedError(f"Interrupted by signal {interrupted}")
        if phase["exit"]:
            raise ValueError(f"{name} command failed: exit {phase['exit']}")

    def frozen():
        return {
            "source": source.digest(source.inputs(root)),
            "wrapper": identity(args.bazel),
            "rc": [identity(path) for path in args.rc],
            "manifest": identity(args.targets_file),
            "input_record": identity(args.input_record) if args.input_record else None,
        }

    try:
        preflight(args, root)
        targets = hardware_targets(args.targets_file)
        if (
            not args.gpu_lock.is_absolute()
            or args.gpu_lock.resolve().is_relative_to(root)
            or not args.gpu_lock.parent.is_dir()
        ):
            raise ValueError(
                "--gpu-lock must be an absolute external lock with an existing parent"
            )
        result["targets"] = targets
        result["source_head"] = source.git(root, "rev-parse", "HEAD").decode().strip()
        result["frozen"] = frozen()
        for signum in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[signum] = signal.signal(signum, cancel)
        expression = "set(" + " ".join(targets) + ")"
        invoke("metadata", [str(args.bazel), "query", "--output=xml", expression])
        hardware_metadata(destination / "metadata.stdout", targets)
        # cquery confirms the same explicit rules are available under the actual
        # analyzed configuration; query supplies their source-owned resource tags.
        invoke(
            "configured",
            [
                str(args.bazel),
                "cquery",
                "--config=opt",
                "--output=label_kind",
                expression,
            ],
        )
        configured = [
            re.fullmatch(r"sh_test rule (//\S+) \([^)]*\)", line)
            for line in (destination / "configured.stdout").read_text().splitlines()
            if line
        ]
        if (
            len(configured) != len(targets)
            or any(row is None for row in configured)
            or {row[1] for row in configured} != set(targets)
        ):
            raise ValueError("Analyzed hardware rule set does not match manifest")
        invoke("build", [str(args.bazel), "build", "--config=opt", *targets])
        invoke(
            "execution_root",
            [str(args.bazel), "info", "--config=opt", "execution_root"],
        )
        execution_root = Path(
            (destination / "execution_root.stdout").read_text().strip()
        )
        if not execution_root.is_absolute() or not execution_root.is_dir():
            raise ValueError(
                "Bazel execution_root must be an existing absolute directory"
            )
        result["execution_root"] = str(execution_root)
        invoke(
            "products",
            [
                str(args.bazel),
                "cquery",
                "--config=opt",
                "--output=starlark",
                '--starlark:expr=providers(target)["DefaultInfo"].files_to_run.executable.path',
                expression,
            ],
        )
        result["products"] = hardware_products(
            destination / "products.stdout", execution_root
        )
        if frozen() != result["frozen"]:
            raise ValueError(
                "Source, wrapper, config, inputs or manifest changed during prebuild"
            )
        lock = args.gpu_lock.open("a+")
        result["lock"] = {"path": str(args.gpu_lock), "requested": time.monotonic()}
        fcntl.flock(lock, fcntl.LOCK_EX)
        locked = True
        result["lock"]["acquired"] = time.monotonic()
        if frozen() != result["frozen"]:
            raise ValueError(
                "Source, wrapper, config, inputs or manifest changed while waiting"
            )
        profile = destination / "test.profile.gz"
        started_ns = time.time_ns()
        invoke(
            "test",
            [
                str(args.bazel),
                "test",
                "--config=opt",
                "--nocache_test_results",
                "--local_test_jobs=1",
                "--record_full_profiler_data",
                "--noslim_profile",
                f"--profile={profile}",
                f"--build_event_json_file={destination / 'events.jsonl'}",
                *targets,
            ],
        )
        result["tests"] = read_results(
            destination / "events.jsonl", started_ns, targets, hardware=True
        )
        actions = no_locked_build_actions(profile)
        result["test_actions"] = actions["tests"]
        result["metadata_actions"] = actions["metadata"]
        if frozen() != result["frozen"]:
            raise ValueError(
                "Source, wrapper, config, inputs or manifest changed during hardware execution"
            )
        result["status"] = "passed"
    except (ValueError, OSError, subprocess.SubprocessError, ET.ParseError) as exc:
        result["error"] = str(exc)
    finally:
        if interrupted is not None:
            # Detached Bazel work can outlive its client. Never deliberately
            # release a hardware lease until public shutdown confirms retirement.
            for signum in prior_handlers:
                signal.signal(signum, signal.SIG_IGN)
            result["cleanup"] = {
                "status": "unresolved",
                "command": [str(args.bazel), "shutdown"],
            }
            while True:
                with (destination / "cleanup.log").open("ab") as log:
                    cleanup = subprocess.Popen(
                        result["cleanup"]["command"],
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        cleanup.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(cleanup.pid, signal.SIGKILL)
                        cleanup.wait()
                    result["cleanup"]["exit"] = cleanup.returncode
                if cleanup.returncode == 0:
                    result["cleanup"]["status"] = "complete"
                    break
                result["cleanup"][
                    "error"
                ] = "Public shutdown failed/timed out; hardware lease retained pending successful owner cleanup"
                save()
                if not locked:
                    break
                time.sleep(5)
        if locked:
            result["lock"]["released"] = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_UN)
        if lock is not None:
            lock.close()
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
        result["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    return 0 if result["status"] == "passed" else 1


def run(args, root=ROOT):
    if args.stage == "hardware":
        return run_hardware(args, root)
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
