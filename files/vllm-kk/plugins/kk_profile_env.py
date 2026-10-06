"""Emit `export K=V` lines for the base image's lil profile environment.

The vllm-kk serve command runs vLLM directly because the image launcher
blocklists the OffloadingConnector flags (--kv-transfer-config,
--kv-offloading-size, --enable-cumem-allocator; runtime/launcher.py
parse_native guard list). The profile/hardware/common environment layers
(JIT cache namespace /cache/jit/<runtime-lock>/<profile-id>, b12x/NCCL
tuning, fastokens, CUDA arch) are still taken verbatim from the image's own
launcher via --print-config, so compose carries no per-build env drift.

Usage inside the container: eval "$(/opt/venv/bin/python -m kk_profile_env)"
"""

import json
import shlex
import subprocess
import sys

PROFILE = "qwen38-flash-next"
HARDWARE = "rtx-pro-6000-pcie"


def main() -> int:
    out = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "runtime.launcher",
            "--profile",
            PROFILE,
            "--hardware",
            HARDWARE,
            "--print-config",
            "--",
            "--cache-mode",
            "vram",
        ],
        text=True,
    )
    plan = json.loads(out)
    for key, spec in plan["environment"].items():
        print(f"export {key}={shlex.quote(spec['value'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
