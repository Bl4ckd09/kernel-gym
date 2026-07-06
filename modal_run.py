"""Run kernel-gym on a Modal GPU.

    modal run modal_run.py                 # A10G (default): CPU sanity + GPU test + grade
    modal run modal_run.py --gpu L4        # pick a GPU
    modal run modal_run.py --cmd test      # just correctness
    modal run modal_run.py --cmd grade     # just benchmarks/grades

Triton has no macOS backend, so the kernels can't run locally; Modal spins up a GPU,
installs torch (which bundles a working Triton on Linux), runs, and tears down.
"""

import subprocess

import modal

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


@app.function(gpu="A10G", timeout=2400)
def run(cmd: str = "all"):
    import torch
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    try:
        import triton
        print(f"triton {triton.__version__}", flush=True)
    except Exception as e:
        print(f"triton import failed: {e}", flush=True)

    if cmd in ("all", "cpu"):
        _sh("python -m pytest tests/test_reference_cpu.py -q")
    if cmd in ("all", "test"):
        _sh("python -m gym test")
    if cmd in ("all", "grade"):
        _sh("python -m gym grade --json /root/card.json")

    try:
        with open("/root/card.json") as f:
            return f.read()
    except FileNotFoundError:
        return None


@app.local_entrypoint()
def main(cmd: str = "all"):
    card = run.remote(cmd=cmd)
    if card:
        with open("card.json", "w") as f:
            f.write(card)
        print("\nreport card written to ./card.json")
