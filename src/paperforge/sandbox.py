"""Run untrusted experiment code with a timeout and no network.

Backends:
- docker: `docker run --network none` with CPU/memory caps (strong isolation).
- subprocess: same interpreter, scrubbed env, network disabled via a sitecustomize
  guard, CPU-time/file-size rlimits, whole process group killed on timeout.
`auto` picks docker when the daemon and image are available.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

GUARD_DIR = Path(__file__).parent / "sandbox_guard"
DOCKER_IMAGE = "paper-forge-sandbox:latest"
MAX_OUTPUT = 20000


@dataclass
class RunResult:
    cmd: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    backend: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return asdict(self)


def _tail(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else "...[truncated]...\n" + s[-MAX_OUTPUT:]


def docker_available(image: str = DOCKER_IMAGE) -> bool:
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=15)
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "docker" if docker_available() else "subprocess"
    return backend


def _limits(cpu_s: int):
    def apply():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 2**20, 200 * 2**20))
        except (ImportError, ValueError, OSError):
            pass
    return apply


def _run_subprocess(cmd: list[str], cwd: Path, timeout_s: float) -> tuple[int | None, str, str, bool]:
    cmd = [sys.executable if c == "python" else c for c in cmd]
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
        "PYTHONPATH": os.pathsep.join([str(GUARD_DIR), str(cwd)]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NO_PROXY": "*",
    }
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True,
                            preexec_fn=_limits(int(timeout_s) + 5) if os.name == "posix" else None)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return proc.returncode, out, err, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            proc.kill()
        out, err = proc.communicate()
        return None, out, err, True


def _run_docker(cmd: list[str], cwd: Path, timeout_s: float, image: str) -> tuple[int | None, str, str, bool]:
    name = f"forge-{os.getpid()}-{int(time.time() * 1000)}"
    full = ["docker", "run", "--rm", "--name", name, "--network", "none", "--memory", "2g",
            "--cpus", "2", "--pids-limit", "256", "-e", "HOME=/work", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=2",
            "-v", f"{cwd.resolve()}:/work", "-w", "/work", image, *cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout_s)
        return r.returncode, r.stdout, r.stderr, False
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return None, "", f"killed after {timeout_s}s timeout", True


def run(cmd: list[str], cwd: str | Path, timeout_s: float = 300, backend: str = "auto",
        image: str = DOCKER_IMAGE) -> RunResult:
    cwd = Path(cwd)
    backend = resolve_backend(backend)
    t0 = time.monotonic()
    if backend == "docker":
        rc, out, err, to = _run_docker(cmd, cwd, timeout_s, image)
    elif backend == "subprocess":
        rc, out, err, to = _run_subprocess(cmd, cwd, timeout_s)
    else:
        raise ValueError(f"unknown sandbox backend {backend!r}")
    return RunResult(cmd, rc, _tail(out), _tail(err), round(time.monotonic() - t0, 3), to, backend)
