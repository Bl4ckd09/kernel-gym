"""Run kernel-gym on a Modal GPU.

    modal run modal_run.py                          # A10G: CPU sanity + GPU test + grade
    KG_GPU=H100 modal run modal_run.py --cmd grade  # pick a GPU via env (read at import)
    modal run modal_run.py --cmd test               # just correctness
    modal run modal_run.py --solutions attempts/opus48   # eval-mode: grade attempt dir

Triton has no macOS backend, so the kernels can't run locally; Modal spins up a GPU,
installs torch (which bundles a working Triton on Linux), runs, and tears down.
"""

import os
import subprocess

import modal

GPU = os.environ.get("KG_GPU", "A10G")

REPO = "/root/kernel-gym"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "pytest", "tabulate", "numpy")
    .add_local_dir(
        "/Users/sunhwang/kernel-gym",
        remote_path=REPO,
        ignore=["**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/*.pyc",
                "**/.pytest_cache/**", "*.json"],
    )
)

app = modal.App("kernel-gym", image=image)


def _sh(cmd: str) -> int:
    print(f"\n\033[1m$ {cmd}\033[0m", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=REPO)
    return r.returncode


@app.function(gpu=GPU, timeout=2400)
def run(cmd: str = "all", solutions: str = ""):
    import torch
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    try:
        import triton
        print(f"triton {triton.__version__}", flush=True)
    except Exception as e:
        print(f"triton import failed: {e}", flush=True)

    sol = f" --solutions {solutions}" if solutions else ""
    if cmd in ("all", "cpu") and not solutions:
        _sh("python -m pytest tests/test_reference_cpu.py -q")
    if cmd in ("all", "test"):
        _sh(f"python -m gym test{sol}")
    if cmd in ("all", "grade"):
        _sh(f"python -m gym grade --json /root/card.json{sol}")

    try:
        with open("/root/card.json") as f:
            return f.read()
    except FileNotFoundError:
        return None


@app.local_entrypoint()
def main(cmd: str = "all", solutions: str = ""):
    card = run.remote(cmd=cmd, solutions=solutions)
    if card:
        with open("card.json", "w") as f:
            f.write(card)
        print("\nreport card written to ./card.json")
