# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
"""Resolve a standalone pinned formatter without borrowing caller Bazel state."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile

TARGET = "@hrx_system//loom/src/loom/tools/loom-format"
# Published release binary, https://releases.bazel.build/9.3.2/release/.
BAZEL_DIGESTS = {
    (
        "9.3.2",
        "Linux",
        "x86_64",
    ): "2e14c55c298752d71e8a0bbd762665ffce5966c9ad00ec5d1a8b82a58e7bfa06",
}
COMPILERS = ("/usr/bin/clang", "/usr/bin/clang++")
FLAGS = [
    "--jobs=8",
    "--enable_bzlmod",
    "--check_direct_dependencies=error",
    "--lockfile_mode=error",
    "--incompatible_strict_action_env",
    "--compilation_mode=opt",
    "--symlink_prefix=/",
    "--@hrx_system//runtime/config/hal:drivers=task",
    "--@hrx_system//loom/config/arch:enable=amdgpu,vm,x86",
    "--@hrx_system//loom/config/emit:enable=elf,vm",
    "--@hrx_system//loom/config/execute:enable=iree_vm",
    "--@hrx_system//loom/config:shared_tools=false",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def run(argv, **kwargs):
    """Own and retire subprocesses before releasing any cache lock."""
    with subprocess.Popen(argv, start_new_session=True, **kwargs) as process:
        try:
            out, err = process.communicate()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, argv, out, err)
        return out


@contextlib.contextmanager
def locked(path, exclusive=True, blocking=True):
    with path.open("a+b") as stream:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(stream, mode | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def outside(path, source):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Formatter cache/state paths must be absolute and external")
    resolved = path.resolve()
    if resolved == source or source in resolved.parents:
        raise ValueError("Formatter cache/state must be outside the checkout")
    return resolved


def identity(source):
    module = (source / "MODULE.bazel").read_text()
    match = re.search(
        r'git_override\(\s*module_name\s*=\s*"hrx",\s*commit\s*=\s*"([0-9a-f]{40})",\s*remote\s*=\s*"(https://[^"\n]+)",?\s*\)',
        module,
    )
    if not match or re.search(r"\b(local_path_override|archive_override)\s*\(", module):
        return None
    for name in (".bazelrc.local", ".bazelrc.configured"):
        path = source / name
        if path.is_file() and "--override_module" in path.read_text():
            return None
    version = (source / ".bazelversion").read_text().strip()
    allowed = BAZEL_DIGESTS.get((version, platform.system(), platform.machine()))
    if not allowed:
        return None
    compilers = [Path(name).resolve() for name in COMPILERS]
    if not all(p.is_file() for p in compilers):
        return None
    return {
        "schema": 1,
        "pin": match[1],
        "remote": match[2],
        "inputs": {
            name: digest(source / name)
            for name in ("MODULE.bazel", "MODULE.bazel.lock", ".bazelversion")
        },
        "host": [platform.system(), platform.machine(), *platform.libc_ver()],
        "recipe": digest(Path(__file__)),
        "python": {
            "version": list(sys.version_info[:3]),
            "sha256": digest(sys.executable),
        },
        "target": TARGET,
        "flags": FLAGS,
        "bazel_version": version,
        "bazel_sha256": allowed,
        "compilers": [{"path": str(p), "sha256": digest(p)} for p in compilers],
    }


def verified(entry, expected):
    try:
        manifest = entry / "manifest.json"
        executable = entry / "loom-format"
        if entry.is_symlink() or manifest.is_symlink() or executable.is_symlink():
            return False
        data = json.loads(manifest.read_text())
        return (
            data["identity"] == expected
            and executable.is_file()
            and os.access(executable, os.X_OK)
            and digest(executable) == data["executable_sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def canonical_launcher(expected):
    configured = os.environ.get("HRX_FORMATTER_BAZEL")
    if not configured:
        return None
    launcher = Path(configured)
    if (
        not launcher.is_absolute()
        or not launcher.is_file()
        or digest(launcher) != expected["bazel_sha256"]
    ):
        raise ValueError(
            "HRX_FORMATTER_BAZEL must be the verified published Bazel binary"
        )
    observed = (
        run([str(launcher), "--version"], stdout=subprocess.PIPE).decode().strip()
    )
    if observed != "bazel " + expected["bazel_version"]:
        raise ValueError("Canonical Bazel version does not match the snapshot")
    return launcher


def fallback(source, arguments, preflight=False):
    bazel = shutil.which(os.environ.get("BAZEL", "bazel"))
    if not bazel:
        raise ValueError("Missing Bazel for the pinned Loom formatter")
    if preflight:
        return
    state = outside(
        os.environ.get(
            "HRX_FORMAT_BAZEL_ROOT", str(workspace() / "tools/formatting-bazel")
        ),
        source,
    )
    state.mkdir(parents=True, exist_ok=True)
    run(
        [
            bazel,
            "--output_user_root=" + str(state),
            "run",
            "--noshow_progress",
            "--ui_event_filters=-info",
            "--symlink_prefix=/",
            "--@hrx_system//runtime/config/hal:drivers=task",
            "--@hrx_system//loom/config/execute:enable=iree_vm",
            TARGET,
            "--",
            *arguments,
        ],
        cwd=source,
    )


def workspace():
    return (
        Path(
            os.environ.get(
                "CODEX_PROJECT_WORKSPACES",
                str(Path.home() / "codex/project-workspaces"),
            )
        )
        / "hrx-solutions"
    )


def populate(source, temporary, expected, launcher):
    environment = {
        k: v
        for k, v in os.environ.items()
        if k in ("HOME", "USER", "LOGNAME", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR")
        or k.startswith("XDG_")
    }
    # Child generators use this task interpreter, with no inherited Python overrides.
    environment.update(
        PATH=str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="",
        PYTHONHOME="",
    )
    scratch = tempfile.mkdtemp(prefix="population-", dir=temporary.parent.parent)
    command = [
        str(launcher),
        "--ignore_all_rc_files",
        "--output_user_root=" + scratch,
    ]
    cc, cxx = [c["path"] for c in expected["compilers"]]
    flags = FLAGS + [
        "--repo_env=IREE_DEPENDENCY_MODE=pinned",
        "--repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned",
        "--repo_env=CC=" + cc,
        "--repo_env=CXX=" + cxx,
        "--action_env=PATH=" + environment["PATH"],
        "--host_action_env=PATH=" + environment["PATH"],
        "--action_env=PYTHONNOUSERSITE=1",
        "--host_action_env=PYTHONNOUSERSITE=1",
        "--action_env=PYTHONDONTWRITEBYTECODE=1",
        "--host_action_env=PYTHONDONTWRITEBYTECODE=1",
        "--action_env=PYTHONPATH=",
        "--host_action_env=PYTHONPATH=",
        "--action_env=PYTHONHOME=",
        "--host_action_env=PYTHONHOME=",
    ]
    try:
        run(command + ["build", *flags, TARGET], cwd=source, env=environment)
        repository = run(
            command + ["mod", "show_repo", "@hrx_system"],
            cwd=source,
            env=environment,
            stdout=subprocess.PIPE,
        ).decode()
        if not all(
            value in repository for value in (expected["pin"], expected["remote"])
        ):
            raise ValueError("Resolved HRX repository does not match the published pin")
        outputs = (
            run(
                command + ["cquery", *flags, TARGET, "--output=files"],
                cwd=source,
                env=environment,
                stdout=subprocess.PIPE,
            )
            .decode()
            .splitlines()
        )
        if len(outputs) != 1:
            raise ValueError("Expected exactly one formatter executable")
        execution_root = Path(
            run(
                command + ["info", *flags, "execution_root"],
                cwd=source,
                env=environment,
                stdout=subprocess.PIPE,
            )
            .decode()
            .strip()
        )
        shutil.copyfile(execution_root / outputs[0], temporary / "loom-format")
        executable = temporary / "loom-format"
        executable.chmod(0o755)
        dependencies = subprocess.run(
            ["/usr/bin/ldd", str(executable)],
            text=True,
            capture_output=True,
            env=environment,
        )
        lines = dependencies.stdout + dependencies.stderr
        if dependencies.returncode and "not a dynamic executable" not in lines:
            raise ValueError(
                "Cannot verify standalone formatter dependencies: " + lines
            )
        if "not found" in lines:
            raise ValueError("Formatter has unresolved shared dependencies")
        for dependency in re.findall(r"(?:=>\s*)?(/\S+)", lines):
            resolved = Path(dependency).resolve()
            if not any(
                str(resolved).startswith(p) for p in ("/usr/lib/", "/lib/", "/lib64/")
            ):
                raise ValueError(
                    "Formatter depends on a private library: " + dependency
                )
        # Only the copied artifact is retained. No output tree/runfiles accompany it.
        run(
            [str(executable), "--help"],
            cwd=temporary,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        (temporary / "manifest.json").write_bytes(
            encoded(
                {
                    "identity": expected,
                    "executable_sha256": digest(executable),
                    "build_command": command[:2] + ["build", *flags, TARGET],
                }
            )
            + b"\n"
        )
    finally:
        try:
            run(command + ["shutdown"], cwd=source, env=environment)
        except BaseException:
            print(
                "Formatter shutdown unresolved; preserve owned output: " + scratch,
                file=sys.stderr,
            )
            raise
        else:
            shutil.rmtree(scratch)


def evict(cache, keep):
    """Retain the selected entry and newest other pin; never remove a live entry."""
    entries = sorted(
        (
            p
            for p in cache.iterdir()
            if re.fullmatch("[0-9a-f]{64}", p.name)
            and p.is_dir()
            and not p.is_symlink()
        ),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    previous = False
    selected_pin = json.loads((keep / "manifest.json").read_text())["identity"]["pin"]
    for entry in entries:
        if entry == keep:
            continue
        try:
            pin = json.loads((entry / "manifest.json").read_text())["identity"]["pin"]
            if pin != selected_pin and not previous:
                previous = True
                continue
            with locked(cache / (entry.name + ".lock"), blocking=False):
                shutil.rmtree(entry)
        except (BlockingIOError, OSError, ValueError, KeyError, TypeError):
            continue


def main(arguments=None, source=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    source = Path(source or Path(__file__).resolve().parents[1]).resolve()
    preflight = arguments == ["--cache-preflight"]
    if os.environ.get("HRX_FORMATTER_CACHE", "on") not in ("on", "off"):
        raise ValueError("HRX_FORMATTER_CACHE must be on or off")
    expected = (
        None if os.environ.get("HRX_FORMATTER_CACHE") == "off" else identity(source)
    )
    if expected is None:
        return fallback(source, arguments, preflight)
    cache = outside(
        os.environ.get(
            "HRX_FORMATTER_CACHE_ROOT", str(workspace() / "tools/loom-format")
        ),
        source,
    )
    cache.mkdir(parents=True, exist_ok=True)
    launcher = canonical_launcher(expected)
    key = hashlib.sha256(encoded(expected)).hexdigest()
    entry = cache / key
    lock = cache / (key + ".lock")
    with locked(lock, exclusive=False):
        if verified(entry, expected):
            if not preflight:
                run([str(entry / "loom-format"), *arguments], cwd=source)
                with locked(cache / "owner.lock"):
                    evict(cache, entry)
            return
    if launcher is None:
        return fallback(source, arguments, preflight)
    if preflight:
        return
    with locked(lock):
        if not verified(entry, expected):
            with tempfile.TemporaryDirectory(prefix="entry-", dir=cache) as staging:
                staged = Path(staging) / "complete"
                staged.mkdir()
                populate(source, staged, expected, launcher)
                if identity(source) != expected:
                    raise ValueError("Formatter inputs changed during population")
                if entry.exists() or entry.is_symlink():
                    if entry.is_symlink():
                        entry.unlink()
                    else:
                        shutil.rmtree(entry)
                staged.rename(entry)
    with locked(lock, exclusive=False):
        if not verified(entry, expected):
            raise ValueError("Formatter cache changed before execution; retry")
        run([str(entry / "loom-format"), *arguments], cwd=source)
        with locked(cache / "owner.lock"):
            evict(cache, entry)


def interrupted(signum, _frame):
    raise KeyboardInterrupt(signum)


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        main()
    except KeyboardInterrupt as error:
        sys.exit(128 + (error.args[0] if error.args else signal.SIGINT))
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
