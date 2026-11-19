#!/usr/bin/env python3
# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Local GGUF -> SOURCE/bytecode -> Serve smoke check. Never downloads models.

Supply --gguf/--tokenizer, or --recipe pointing to an existing recipe JSON.
The converter and server must already be built/installed. Temporary packages
and isolated server state are removed; the JSON result goes to stdout.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binding.python.models.model_download import _cache_path


def inputs(args):
    recipe = json.loads(args.recipe.read_text()) if args.recipe else None
    paths = {}
    for name, role in (("gguf", "weights"), ("tokenizer", "tokenizer")):
        path = getattr(args, name)
        if path is None and recipe is not None:
            source = recipe["files"][role]["source"]
            if source["kind"] != "huggingface":
                raise ValueError(f"Supply --{name} for this recipe's {role} asset")
            path = _cache_path(source)
        if path is None:
            raise ValueError(f"Supply --{name} or --recipe")
        # Preserve .gguf aliases and symlink/.. semantics; the exporter checks
        # the opened identity. Resolving here can lose the format extension.
        path = Path(path).expanduser().absolute()
        if not path.is_file():
            raise ValueError(f"Missing local {role}: {path} (no download attempted)")
        paths[name] = path
    return paths


def executable(value, name):
    found = shutil.which(value or name)
    if not found:
        option = "--serve" if name == "loom-serve" else "--converter"
        raise ValueError(f"Cannot find {name}; supply {option}")
    return str(Path(found).absolute())


def stop(process):
    if process.poll() is None:
        for sig, timeout in (
            (signal.SIGINT, 15),
            (signal.SIGTERM, 5),
            (signal.SIGKILL, 5),
        ):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            raise RuntimeError(f"Could not stop owned process {process.pid}")
        if sig != signal.SIGINT:
            raise RuntimeError(f"Owned process {process.pid} required forced shutdown")


def smoke(args, paths, converter, serve):
    root = Path(tempfile.mkdtemp(prefix="loomify-smoke-"))
    children = []
    result = {
        "success": False,
        "inputs": {k: str(v) for k, v in paths.items()},
        "converter": converter,
        "serve": serve,
        "platform": args.platform,
        "device": args.device,
        "cases": [],
    }
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(port, path, body=None, timeout=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with http.open(req, timeout=timeout or args.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(4000).decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    def launch(command, log, directory):
        child_env = dict(env, LOOM_SERVE_DIR=str(directory / "state"))
        process = subprocess.Popen(
            command,
            cwd=root,
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        children.append(process)
        return process

    try:
        for fmt in ("loom", "bc"):
            case = {"format": fmt, "stage": "export"}
            result["cases"].append(case)
            directory = root / fmt
            directory.mkdir()
            package = directory / "package"
            with (directory / "export.log").open("w") as log:
                process = launch(
                    [
                        converter,
                        f"--gguf={paths['gguf']}",
                        f"--tokenizer={paths['tokenizer']}",
                        f"--output-dir={package}",
                        f"--format={fmt}",
                        f"--platform={args.platform}",
                        f"--prefill-capacity={args.prefill_capacity}",
                        f"--context-capacity={args.context_capacity}",
                    ],
                    log,
                    directory,
                )
                try:
                    if process.wait(timeout=args.timeout) != 0:
                        raise RuntimeError(f"{fmt} export failed")
                finally:
                    stop(process)
            model = json.loads((package / "model.json").read_text())
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            case["stage"] = "readiness"
            with (directory / "server.log").open("w") as log:
                server = launch(
                    [
                        serve,
                        f"--device={args.device}",
                        "--pool_backing=fixed",
                        f"--listen=127.0.0.1:{port}",
                        f"--model={package / 'model.json'}",
                    ],
                    log,
                    directory,
                )
                try:
                    deadline = time.monotonic() + args.timeout
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(
                                f"{fmt} server exited with {server.returncode}"
                            )
                        try:
                            status = request(port, "/v1/status", timeout=1)
                            entry = next(
                                (
                                    m
                                    for m in status.get("data", [])
                                    if m.get("id") == model["name"]
                                ),
                                {},
                            )
                            registration = entry.get("loom", {})
                            if registration.get("registration") == "failed":
                                raise RuntimeError(
                                    f"Registration failed: {registration.get('error')}"
                                )
                            if registration.get("registration") == "ready":
                                break
                        except (urllib.error.HTTPError, json.JSONDecodeError):
                            raise
                        except (OSError, urllib.error.URLError):
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"{fmt} readiness timed out")
                        time.sleep(0.1)
                    case["devices"] = (
                        status.get("server", {}).get("resources", {}).get("devices", [])
                    )
                    case["stage"] = "request"
                    body = {
                        "model": model["name"],
                        "messages": [{"role": "user", "content": args.prompt}],
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                    }
                    counted = request(port, "/v1/chat/completions/input_tokens", body)[
                        "input_tokens"
                    ]
                    response = request(port, "/v1/chat/completions", body)
                    choice = response["choices"][0]
                    usage = response["usage"]
                    if usage["prompt_tokens"] != counted:
                        raise RuntimeError(
                            "Input count and completion accounting disagree"
                        )
                    generated = usage["completion_tokens"]
                    if (
                        type(generated) is not int
                        or not 0 < generated <= args.max_tokens
                    ):
                        raise RuntimeError(
                            "No generation progress or invalid completion count"
                        )
                    if usage["total_tokens"] != counted + generated:
                        raise RuntimeError("Invalid total token accounting")
                    if not isinstance(choice["message"], dict) or not choice.get(
                        "finish_reason"
                    ):
                        raise RuntimeError("Incomplete completion response")
                    case.update(
                        message=choice["message"],
                        usage=usage,
                        finish_reason=choice["finish_reason"],
                        decode_exercised=generated > 1,
                    )
                finally:
                    stop(server)
            case["stage"] = "shutdown"
            if server.returncode != 0:
                raise RuntimeError(f"{fmt} server exited with {server.returncode}")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"{fmt} server port remains open")
            case["stage"] = "passed"
        a, b = result["cases"]
        for field in ("message", "usage", "finish_reason"):
            if a[field] != b[field]:
                raise RuntimeError(
                    f"SOURCE/bytecode {field} differs under identical greedy settings"
                )
        result["success"] = True
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = str(error) or "Interrupted"
        result["logs"] = {
            str(p.relative_to(root)): p.read_text(errors="replace")[-4000:]
            for p in root.rglob("*.log")
        }
    finally:
        previous = {
            sig: signal.signal(sig, signal.SIG_IGN)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            for child in children:
                try:
                    stop(child)
                except RuntimeError as error:
                    result.update(success=False, cleanup_error=str(error))
            if all(child.poll() is not None for child in children):
                try:
                    shutil.rmtree(root)
                except OSError as error:
                    result.update(
                        success=False,
                        cleanup_error=str(error),
                        retained_work_dir=str(root),
                    )
            else:
                result.update(success=False, retained_work_dir=str(root))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--recipe",
        type=Path,
        help="existing recipe JSON; use pinned local HF cache only",
    )
    parser.add_argument(
        "--converter", help="existing loomify-gguf executable (otherwise PATH)"
    )
    parser.add_argument(
        "--serve", help="existing loom-serve executable (otherwise PATH)"
    )
    parser.add_argument(
        "--platform",
        default="amdgpu:gfx11-generic",
        help="maintained generation recipe, independent of the runtime device",
    )
    parser.add_argument("--device", default="amdgpu")
    parser.add_argument("--prompt", default="Write a short greeting.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefill-capacity", type=int, default=8)
    parser.add_argument("--context-capacity", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--report", type=Path, help="also save the JSON result here")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive")
    try:
        paths = inputs(args)
        converter = executable(args.converter, "loomify-gguf")
        serve = executable(args.serve, "loom-serve")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt()

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        result = smoke(args, paths, converter, serve)
    finally:
        signal.signal(signal.SIGTERM, previous)
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(text)
    print(text, end="")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
