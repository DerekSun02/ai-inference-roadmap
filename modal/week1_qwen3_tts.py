"""Modal launcher for the Week 1 Qwen3-TTS minimum benchmark matrix.

The script deliberately automates infrastructure, not interpretation. Run one
cell and one repeat at a time, then inspect the produced speed_results.json.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "ai-inference-week1-qwen3-tts"
BASE_IMAGE = "hongccc/sglang-omni:dev"
SGLANG_OMNI_COMMIT = "442e559b40b5040965ec876650b32da05d31769f"
SOURCE_DIR = f"/opt/sglang-omni-{SGLANG_OMNI_COMMIT[:12]}"

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
DATASET_ID = "zhaochenyang20/seed-tts-eval-arrow"
CONFIG_PATH = f"{SOURCE_DIR}/examples/configs/qwen3_tts_1_7b_customvoice.yaml"

# H100! requests an actual H100 rather than Modal's free H100 -> H200 upgrade.
# Override before `modal run` only when you intentionally start a new matrix.
GPU_TYPE = os.environ.get("WEEK1_MODAL_GPU", "H100!")
HF_SECRET_NAME = os.environ.get("WEEK1_HF_SECRET")
SECRETS = [modal.Secret.from_name(HF_SECRET_NAME)] if HF_SECRET_NAME else []

CACHE_ROOT = Path("/cache")
HF_HOME = CACHE_ROOT / "huggingface"
RESULTS_ROOT = Path("/results/week-01")
PORT = 8000

CELL_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "concurrency": 1,
        "request_rate": None,
        "max_samples": 1,
        "warmup": 0,
        "purpose": "Validate the installation and request path; not a measured run.",
    },
    "c1": {
        "concurrency": 1,
        "request_rate": None,
        "max_samples": 32,
        "warmup": 1,
        "purpose": "Single-request latency and memory baseline.",
    },
    "c8": {
        "concurrency": 8,
        "request_rate": None,
        "max_samples": 32,
        "warmup": 8,
        "purpose": "Closed-loop continuous-batching baseline.",
    },
    "rps1": {
        # BenchmarkRunner interprets zero as no client-side concurrency cap.
        "concurrency": 0,
        "request_rate": 1.0,
        "max_samples": 60,
        "warmup": 1,
        "purpose": "Open-loop 1 RPS Poisson arrivals; expected horizon about 60 s.",
    },
}


image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install("curl", "ffmpeg", "git", "sox")
    .run_commands(
        f"git clone --filter=blob:none https://github.com/sgl-project/sglang-omni.git {SOURCE_DIR}",
        f"cd {SOURCE_DIR} && git switch --detach {SGLANG_OMNI_COMMIT}",
        "python -m pip install --upgrade pip uv",
        f"cd {SOURCE_DIR} && uv pip install --system --prerelease=allow -v -e .",
        "uv pip install --system --no-deps sox einops",
        "uv pip install --system --no-deps qwen-tts==0.1.1",
    )
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HOME / "hub"),
            "HF_DATASETS_CACHE": str(HF_HOME / "datasets"),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(
    "ai-inference-week1-cache", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "ai-inference-week1-results", create_if_missing=True
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_capture(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def _download_model() -> str:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(HF_HOME / "hub"),
    )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "week1-model.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "requested_revision": MODEL_REVISION,
                "snapshot_path": snapshot,
                "resolved_revision": Path(snapshot).name,
                "downloaded_at_utc": _utc_now(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    cache_volume.commit()
    return snapshot


@app.function(
    image=image,
    timeout=2 * 60 * 60,
    volumes={str(CACHE_ROOT): cache_volume},
    secrets=SECRETS,
)
def prepare() -> dict[str, str]:
    """Download the pinned model once without allocating a GPU."""

    snapshot = _download_model()
    result = {
        "model": MODEL_ID,
        "resolved_revision": Path(snapshot).name,
        "snapshot_path": snapshot,
    }
    print(json.dumps(result, indent=2))
    return result


def _wait_for_server(process: subprocess.Popen[Any], timeout_s: int = 1800) -> None:
    deadline = time.monotonic() + timeout_s
    next_update = time.monotonic()
    url = f"http://127.0.0.1:{PORT}/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"SGLang-Omni server exited early with code {return_code}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    print("Server is healthy; starting the selected benchmark cell.")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= next_update:
            print("Waiting for model load and CUDA Graph capture ...")
            next_update = time.monotonic() + 30
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout_s} seconds")


def _stop_process_group(process: subprocess.Popen[Any], grace_s: int = 30) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _start_gpu_sampler(csv_path: Path) -> tuple[subprocess.Popen[Any], Any]:
    output = csv_path.open("w", encoding="utf-8")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,uuid,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,nounits",
        "--loop-ms=500",
    ]
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True)
    return process, output


def _peak_memory_mib(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    memory_values: list[float] = []
    for row in rows:
        for key, value in row.items():
            if key and key.strip().startswith("memory.used") and value:
                try:
                    memory_values.append(float(value.strip()))
                except ValueError:
                    pass
    return max(memory_values) if memory_values else None


def _stream_command(command: list[str], log_path: Path, cwd: str) -> int:
    print("$ " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={
        str(CACHE_ROOT): cache_volume,
        "/results": results_volume,
    },
    secrets=SECRETS,
)
def run_cell(cell: str, repeat: int = 1) -> dict[str, Any]:
    """Run one matrix cell. The caller remains responsible for interpretation."""

    if cell not in CELL_CONFIGS:
        raise ValueError(f"Unknown cell {cell!r}; choose from {sorted(CELL_CONFIGS)}")
    if repeat < 0:
        raise ValueError("repeat must be >= 0")
    if cell != "smoke" and repeat not in {1, 2, 3}:
        raise ValueError("Measured cells use repeat 1, 2, or 3")

    cell_config = CELL_CONFIGS[cell]
    started_at = datetime.now(timezone.utc)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    result_dir = RESULTS_ROOT / cell / f"repeat-{repeat:02d}-{timestamp}"
    benchmark_dir = result_dir / "benchmark"
    result_dir.mkdir(parents=True, exist_ok=False)
    benchmark_dir.mkdir()

    snapshot = _download_model()
    server_command = [
        "sgl-omni",
        "serve",
        "--model-path",
        snapshot,
        "--config",
        CONFIG_PATH,
        "--port",
        str(PORT),
    ]
    benchmark_command = [
        "python",
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--generate-only",
        "--use-existing-server",
        "--stream",
        "--base-url",
        f"http://127.0.0.1:{PORT}",
        "--model",
        MODEL_ID,
        "--meta",
        DATASET_ID,
        "--no-ref-audio",
        "--voice",
        "Ryan",
        "--task-type",
        "CustomVoice",
        "--lang",
        "en",
        "--seed",
        "1234",
        "--max-new-tokens",
        "2048",
        "--warmup",
        str(cell_config["warmup"]),
        "--concurrency",
        str(cell_config["concurrency"]),
        "--max-samples",
        str(cell_config["max_samples"]),
        "--output-dir",
        str(benchmark_dir),
        "--disable-tqdm",
    ]
    if cell_config["request_rate"] is not None:
        benchmark_command.extend(["--request-rate", str(cell_config["request_rate"])])

    manifest: dict[str, Any] = {
        "status": "starting",
        "cell": cell,
        "repeat": repeat,
        "purpose": cell_config["purpose"],
        "started_at_utc": started_at.isoformat(),
        "requested_modal_gpu": GPU_TYPE,
        "base_image": BASE_IMAGE,
        "sglang_omni_commit": SGLANG_OMNI_COMMIT,
        "model_id": MODEL_ID,
        "requested_model_revision": MODEL_REVISION,
        "resolved_model_revision": Path(snapshot).name,
        "dataset": DATASET_ID,
        "controls": {
            "voice": "Ryan",
            "corpus_language": "English",
            "request_language_field": "auto (the pinned benchmark has no language CLI flag)",
            "instructions": None,
            "stream": True,
            "response_format": "pcm",
            "seed": 1234,
            "max_new_tokens": 2048,
        },
        "cell_config": cell_config,
        "server_command": server_command,
        "benchmark_command": benchmark_command,
        "gpu_before_server": _run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (result_dir / "pip-freeze.txt").write_text(
        _run_capture(["python", "-m", "pip", "freeze"]) + "\n",
        encoding="utf-8",
    )
    (result_dir / "nvidia-smi-q.txt").write_text(
        _run_capture(["nvidia-smi", "-q"]) + "\n",
        encoding="utf-8",
    )

    gpu_csv = result_dir / "gpu-samples.csv"
    sampler, sampler_output = _start_gpu_sampler(gpu_csv)
    server_log = (result_dir / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        server_command,
        cwd=SOURCE_DIR,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    benchmark_return_code: int | None = None
    error: str | None = None
    try:
        _wait_for_server(server)
        manifest["gpu_after_server_ready"] = _run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        )
        benchmark_return_code = _stream_command(
            benchmark_command,
            result_dir / "benchmark.log",
            SOURCE_DIR,
        )
        if benchmark_return_code != 0:
            raise RuntimeError(f"Benchmark exited with code {benchmark_return_code}")
        manifest["status"] = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        manifest["status"] = "failed"
        manifest["error"] = error
        raise
    finally:
        _stop_process_group(server)
        server_log.close()
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sampler.wait(timeout=5)
        sampler_output.close()
        manifest["benchmark_return_code"] = benchmark_return_code
        manifest["finished_at_utc"] = _utc_now()
        manifest["peak_gpu_memory_mib"] = _peak_memory_mib(gpu_csv)
        if error:
            manifest["error"] = error
        speed_results_path = benchmark_dir / "speed_results.json"
        if speed_results_path.exists():
            speed_results = json.loads(speed_results_path.read_text(encoding="utf-8"))
            manifest["speed_summary"] = speed_results.get("summary")
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        results_volume.commit()

    result = {
        "status": manifest["status"],
        "cell": cell,
        "repeat": repeat,
        "result_dir": str(result_dir),
        "speed_summary": manifest.get("speed_summary"),
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(action: str = "prepare", repeat: int = 1) -> None:
    """CLI: prepare, smoke, c1, c8, or rps1."""

    if action == "prepare":
        prepare.remote()
        return
    if action not in CELL_CONFIGS:
        raise ValueError("action must be prepare, smoke, c1, c8, or rps1")
    run_cell.remote(action, repeat)
