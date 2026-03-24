#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_DEFAULTS = {
    "MODEL_ID": "Qwen/Qwen3.5-9B",
    "BASE_NAME": "",
    "MODELS_DIR": "./models",
    "LLAMA_CPP_DIR": "./llama.cpp",
    "QUANTS": "q8_0 q6_K q5_K_M q4_K_M q4_0",
    "MODEL_FILES": "",
    "DATASET_PATH": "./eval/eval_dataset.json",
    "PROMPT": "",
    "PROMPTS": "Explain gradient descent in 3 concise bullet points.||What is overfitting and how do you mitigate it?||Compare precision and recall in one short paragraph.",
    "REPETITIONS": "2",
    "WARMUP_PROMPTS": "1",
    "TEMP": "0.1",
    "N_PREDICT": "128",
    "CTX_SIZE": "4096",
    "THREADS": "-1",
    "N_GPU_LAYERS": "auto",
    "SERVER_PORT_BASE": "8088",
    "SERVER_START_TIMEOUT_SEC": "120",
    "REQUEST_TIMEOUT_SEC": "300",
    "MEMORY_POLL_INTERVAL_SEC": "0.10",
    "OUTPUT_CSV": "./eval/bench_results.csv",
    "OUTPUT_JSON": "./eval/bench_results.json",
    "SERVER_LOG_DIR": "./logs/eval",
}


@dataclass
class RequestMetrics:
    ttft_ms: float | None
    tpot_ms_per_chunk: float | None
    tpot_ms_per_word: float | None
    e2e_latency_ms: float
    output_chars: int
    output_words: int
    chunks_seen: int
    peak_rss_gb: float | None
    peak_vram_gb: float | None
    gpu_indices: str | None
    gpu_names: str | None
    gpu_uuids: str | None
    peak_vram_by_gpu_json: str | None


class MemoryMonitor:
    def __init__(self, pid: int, poll_interval_sec: float):
        self.pid = pid
        self.poll_interval_sec = poll_interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_gb: float | None = None
        self.peak_vram_gb: float | None = None
        self.gpu_indices: set[str] = set()
        self.gpu_names: set[str] = set()
        self.gpu_uuids: set[str] = set()
        self.peak_vram_by_gpu: dict[str, float] = {}

    def _sample_once(self) -> None:
        rss_gb = self._get_rss_gb(self.pid)
        if rss_gb is not None:
            if self.peak_rss_gb is None or rss_gb > self.peak_rss_gb:
                self.peak_rss_gb = rss_gb

        usages = self._get_gpu_usage_for_pid(self.pid)
        if usages:
            total_vram_gb = sum(item["used_gb"] for item in usages)
            if self.peak_vram_gb is None or total_vram_gb > self.peak_vram_gb:
                self.peak_vram_gb = total_vram_gb

            for item in usages:
                gpu_index = item["index"]
                gpu_name = item["name"]
                gpu_uuid = item["uuid"]
                used_gb = item["used_gb"]

                self.gpu_indices.add(gpu_index)
                self.gpu_names.add(gpu_name)
                self.gpu_uuids.add(gpu_uuid)

                previous = self.peak_vram_by_gpu.get(gpu_index)
                if previous is None or used_gb > previous:
                    self.peak_vram_by_gpu[gpu_index] = used_gb

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            time.sleep(self.poll_interval_sec)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample_once()

    @staticmethod
    def _get_rss_gb(pid: int) -> float | None:
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        kb = float(parts[1])
                        # convert KB -> GB
                        return kb / 1024.0 / 1024.0
        except (FileNotFoundError, PermissionError, OSError):
            return None
        return None

    @staticmethod
    def _get_gpu_usage_for_pid(pid: int) -> list[dict[str, Any]]:
        nvidia_smi = shutil_which("nvidia-smi")
        if not nvidia_smi:
            return []

        gpu_inventory = MemoryMonitor._get_gpu_inventory(nvidia_smi)

        try:
            proc = subprocess.run(
                [
                    nvidia_smi,
                    "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                return []

            matches: list[dict[str, Any]] = []
            for raw in proc.stdout.strip().splitlines():
                if not raw.strip():
                    continue
                fields = [f.strip() for f in raw.split(",")]
                if len(fields) < 3:
                    continue
                try:
                    app_pid = int(fields[0])
                    gpu_uuid = fields[1]
                    # nvidia-smi returns MB; convert to GB
                    used_mb = float(fields[2])
                    used_gb = used_mb / 1024.0
                except ValueError:
                    continue
                if app_pid == pid:
                    gpu_meta = gpu_inventory.get(gpu_uuid, {})
                    matches.append(
                        {
                            "index": gpu_meta.get("index", "unknown"),
                            "name": gpu_meta.get("name", "unknown"),
                            "uuid": gpu_uuid,
                            "used_gb": used_gb,
                        }
                    )
            return matches
        except OSError:
            return []

    @staticmethod
    def _get_gpu_inventory(nvidia_smi: str) -> dict[str, dict[str, str]]:
        inventory: dict[str, dict[str, str]] = {}
        try:
            proc = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,uuid",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                return inventory

            for raw in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 3:
                    continue
                index = parts[0]
                name = parts[1]
                uuid = parts[2]
                inventory[uuid] = {"index": index, "name": name}
        except OSError:
            return inventory

        return inventory


def shutil_which(binary: str) -> str | None:
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path_dir) / binary
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def parse_config(path: Path) -> dict[str, str]:
    values = dict(CONFIG_DEFAULTS)
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        if value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]
        values[key] = value

    return values


def derive_base_name(model_id: str) -> str:
    short_name = model_id.split("/")[-1].lower()
    cleaned = []
    for ch in short_name:
        if ch.isalnum() or ch in "_-":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def to_int(config: dict[str, str], key: str) -> int:
    return int(config.get(key, CONFIG_DEFAULTS[key]))


def to_float(config: dict[str, str], key: str) -> float:
    return float(config.get(key, CONFIG_DEFAULTS[key]))


def resolve_model_paths(config: dict[str, str], repo_root: Path) -> list[Path]:
    models_dir = Path(config["MODELS_DIR"])
    if not models_dir.is_absolute():
        models_dir = (repo_root / models_dir).resolve()

    model_files_raw = config.get("MODEL_FILES", "").strip()
    if model_files_raw:
        paths: list[Path] = []
        for item in model_files_raw.split(","):
            candidate = item.strip()
            if not candidate:
                continue
            p = Path(candidate)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            paths.append(p)
        return paths

    model_id = config["MODEL_ID"]
    base_name = config.get("BASE_NAME", "").strip() or derive_base_name(model_id)
    quants = [q.strip() for q in config["QUANTS"].split() if q.strip()]

    paths = [models_dir / f"{base_name}-{quant}.gguf" for quant in quants]
    if not paths:
        raise ValueError("No model files resolved. Set QUANTS or MODEL_FILES.")
    return paths


def load_prompts(config: dict[str, str], repo_root: Path) -> list[str]:
    prompt = config.get("PROMPT", "").strip()
    if prompt:
        return [prompt]

    prompts: list[str] = []

    dataset_path = Path(config["DATASET_PATH"])
    if not dataset_path.is_absolute():
        dataset_path = (repo_root / dataset_path).resolve()

    if dataset_path.exists():
        try:
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        prompts.append(item.strip())
                    elif isinstance(item, dict):
                        for key in ("prompt", "question", "query"):
                            value = item.get(key)
                            if isinstance(value, str) and value.strip():
                                prompts.append(value.strip())
                                break
        except json.JSONDecodeError:
            pass

    if prompts:
        return prompts

    raw = config.get("PROMPTS", "")
    prompts = [p.strip() for p in raw.split("||") if p.strip()]
    if not prompts:
        raise ValueError("No prompts found. Set PROMPT, PROMPTS, or fill eval/eval_dataset.json")
    return prompts


def wait_for_server(port: int, timeout_sec: float) -> None:
    import requests

    url_candidates = [
        f"http://127.0.0.1:{port}/health",
        f"http://127.0.0.1:{port}/v1/models",
    ]

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        for url in url_candidates:
            try:
                r = requests.get(url, timeout=2)
                if r.status_code < 500:
                    return
            except Exception:
                pass
        time.sleep(0.5)

    raise TimeoutError(f"llama-server did not become ready within {timeout_sec}s on port {port}")


def start_server(
    llama_server_bin: Path,
    model_path: Path,
    port: int,
    config: dict[str, str],
    log_dir: Path,
) -> tuple[subprocess.Popen[str], Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"server_{model_path.stem}_{int(time.time())}.log"

    n_gpu_layers = config.get("N_GPU_LAYERS", "auto").strip()
    cmd = [
        str(llama_server_bin),
        "-m",
        str(model_path),
        "--port",
        str(port),
        "-c",
        str(to_int(config, "CTX_SIZE")),
        "-t",
        str(to_int(config, "THREADS")),
    ]

    if n_gpu_layers and n_gpu_layers.lower() != "auto":
        cmd.extend(["-ngl", n_gpu_layers])

    with open(log_file, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )

    return process, log_file


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_stream_request(
    port: int,
    prompt: str,
    config: dict[str, str],
    server_pid: int,
) -> RequestMetrics:
    import requests

    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": "local",
        "prompt": prompt,
        "temperature": to_float(config, "TEMP"),
        "max_tokens": to_int(config, "N_PREDICT"),
        "stream": True,
    }

    timeout_sec = to_float(config, "REQUEST_TIMEOUT_SEC")
    poll_interval = to_float(config, "MEMORY_POLL_INTERVAL_SEC")

    monitor = MemoryMonitor(server_pid, poll_interval)
    monitor.start()

    start_t = time.perf_counter()
    first_t: float | None = None
    last_t: float | None = None
    chunks: list[str] = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, timeout_sec)) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break

                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue

                choice = (obj.get("choices") or [{}])[0]
                text = choice.get("text") or ""
                if text:
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                    last_t = now
                    chunks.append(text)

    finally:
        monitor.stop()

    end_t = time.perf_counter()

    generated = "".join(chunks)
    output_words = len([w for w in generated.split() if w])

    ttft_ms = (first_t - start_t) * 1000.0 if first_t is not None else None

    tpot_chunk_ms: float | None = None
    if first_t is not None and last_t is not None and len(chunks) > 1:
        tpot_chunk_ms = (last_t - first_t) * 1000.0 / float(len(chunks) - 1)

    tpot_word_ms: float | None = None
    if first_t is not None and last_t is not None and output_words > 1:
        tpot_word_ms = (last_t - first_t) * 1000.0 / float(output_words - 1)

    return RequestMetrics(
        ttft_ms=ttft_ms,
        tpot_ms_per_chunk=tpot_chunk_ms,
        tpot_ms_per_word=tpot_word_ms,
        e2e_latency_ms=(end_t - start_t) * 1000.0,
        output_chars=len(generated),
        output_words=output_words,
        chunks_seen=len(chunks),
        peak_rss_gb=monitor.peak_rss_gb,
        peak_vram_gb=monitor.peak_vram_gb,
        gpu_indices=("|".join(sorted(monitor.gpu_indices)) if monitor.gpu_indices else None),
        gpu_names=("|".join(sorted(monitor.gpu_names)) if monitor.gpu_names else None),
        gpu_uuids=("|".join(sorted(monitor.gpu_uuids)) if monitor.gpu_uuids else None),
        peak_vram_by_gpu_json=(json.dumps(monitor.peak_vram_by_gpu, sort_keys=True) if monitor.peak_vram_by_gpu else None),
    )


def mean(values: list[float]) -> float | None:
    data = [v for v in values if v is not None]
    if not data:
        return None
    return sum(data) / len(data)


def p95(values: list[float]) -> float | None:
    data = sorted([v for v in values if v is not None])
    if not data:
        return None
    idx = max(0, int(round(0.95 * (len(data) - 1))))
    return data[idx]


def collect_union(values: list[Any]) -> str | None:
    union: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        for token in text.split("|"):
            token = token.strip()
            if token:
                union.add(token)
    if not union:
        return None
    return "|".join(sorted(union))


def as_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local GGUF models for memory, TTFT, and TPOT")
    parser.add_argument("--config-file", default="./eval/eval_config.txt", help="Path to eval config file")
    parser.add_argument("--model-id", default=None, help="Override MODEL_ID")
    parser.add_argument("--quants", default=None, help="Override QUANTS (space-separated)")
    parser.add_argument("--temp", type=float, default=None, help="Override TEMP")
    parser.add_argument("--n-predict", type=int, default=None, help="Override N_PREDICT")
    parser.add_argument("--repetitions", type=int, default=None, help="Override REPETITIONS")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    config = parse_config(config_path)

    if args.model_id:
        config["MODEL_ID"] = args.model_id
    if args.quants:
        config["QUANTS"] = args.quants
    if args.temp is not None:
        config["TEMP"] = str(args.temp)
    if args.n_predict is not None:
        config["N_PREDICT"] = str(args.n_predict)
    if args.repetitions is not None:
        config["REPETITIONS"] = str(args.repetitions)

    llama_cpp_dir = Path(config["LLAMA_CPP_DIR"])
    if not llama_cpp_dir.is_absolute():
        llama_cpp_dir = (repo_root / llama_cpp_dir).resolve()

    llama_server_bin = llama_cpp_dir / "build" / "bin" / "llama-server"
    if not llama_server_bin.exists():
        print(f"Error: llama-server not found at {llama_server_bin}", file=sys.stderr)
        print("Build llama.cpp first (cmake -S llama.cpp -B llama.cpp/build && cmake --build llama.cpp/build).", file=sys.stderr)
        return 1

    model_paths = resolve_model_paths(config, repo_root)
    prompts = load_prompts(config, repo_root)

    output_csv = Path(config["OUTPUT_CSV"])
    if not output_csv.is_absolute():
        output_csv = (repo_root / output_csv).resolve()
    output_json = Path(config["OUTPUT_JSON"])
    if not output_json.is_absolute():
        output_json = (repo_root / output_json).resolve()
    log_dir = Path(config["SERVER_LOG_DIR"])
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    repetitions = to_int(config, "REPETITIONS")
    warmup_prompts = to_int(config, "WARMUP_PROMPTS")
    port_base = to_int(config, "SERVER_PORT_BASE")
    startup_timeout = to_float(config, "SERVER_START_TIMEOUT_SEC")

    print(f"Config file: {as_rel(config_path, repo_root)}")
    print(f"Prompts loaded: {len(prompts)}")
    print(f"Repetitions: {repetitions}, Warmup prompts: {warmup_prompts}")

    for model_idx, model_path in enumerate(model_paths):
        if not model_path.exists():
            print(f"[skip] Missing model file: {as_rel(model_path, repo_root)}")
            continue

        rel_model_path = as_rel(model_path, repo_root)

        port = port_base + model_idx
        print(f"\n== Benchmarking: {rel_model_path} (port {port}) ==")

        process, log_file = start_server(llama_server_bin, model_path, port, config, log_dir)

        try:
            wait_for_server(port, startup_timeout)
            print(f"Server ready. Log: {as_rel(log_file, repo_root)}")

            for i in range(min(warmup_prompts, len(prompts))):
                _ = run_stream_request(port, prompts[i], config, process.pid)
                print(f"  warmup {i + 1}/{min(warmup_prompts, len(prompts))} complete")

            for rep in range(1, repetitions + 1):
                for prompt_idx, prompt in enumerate(prompts, start=1):
                    metrics = run_stream_request(port, prompt, config, process.pid)
                    row = {
                        "model_path": rel_model_path,
                        "model_file": model_path.name,
                        "repetition": rep,
                        "prompt_index": prompt_idx,
                        "prompt_preview": prompt[:120],
                        "ttft_ms": metrics.ttft_ms,
                        "tpot_ms_per_chunk": metrics.tpot_ms_per_chunk,
                        "tpot_ms_per_word": metrics.tpot_ms_per_word,
                        "e2e_latency_ms": metrics.e2e_latency_ms,
                        "output_chars": metrics.output_chars,
                        "output_words": metrics.output_words,
                        "chunks_seen": metrics.chunks_seen,
                            "peak_rss_gb": metrics.peak_rss_gb,
                            "peak_vram_gb": metrics.peak_vram_gb,
                        "gpu_indices": metrics.gpu_indices,
                        "gpu_names": metrics.gpu_names,
                        "gpu_uuids": metrics.gpu_uuids,
                        "peak_vram_by_gpu_json": metrics.peak_vram_by_gpu_json,
                        "temperature": to_float(config, "TEMP"),
                        "n_predict": to_int(config, "N_PREDICT"),
                        "ctx_size": to_int(config, "CTX_SIZE"),
                    }
                    rows.append(row)
                    print(
                        f"  rep={rep} prompt={prompt_idx}: "
                        f"TTFT={fmt(metrics.ttft_ms)} ms, "
                        f"TPOT(chunk)={fmt(metrics.tpot_ms_per_chunk)} ms, "
                        f"RSS_peak={fmt(metrics.peak_rss_gb)} GB, "
                        f"VRAM_peak={fmt(metrics.peak_vram_gb)} GB, "
                        f"GPU={metrics.gpu_indices or 'n/a'}"
                    )

        finally:
            stop_server(process)

        model_rows = [r for r in rows if r["model_path"] == rel_model_path]
        summary = {
            "model_path": rel_model_path,
            "samples": len(model_rows),
            "ttft_ms_mean": mean([r["ttft_ms"] for r in model_rows]),
            "ttft_ms_p95": p95([r["ttft_ms"] for r in model_rows]),
            "tpot_ms_per_chunk_mean": mean([r["tpot_ms_per_chunk"] for r in model_rows]),
            "tpot_ms_per_chunk_p95": p95([r["tpot_ms_per_chunk"] for r in model_rows]),
            "peak_rss_gb_max": max_or_none([r["peak_rss_gb"] for r in model_rows]),
            "peak_vram_gb_max": max_or_none([r["peak_vram_gb"] for r in model_rows]),
            "gpu_indices": collect_union([r.get("gpu_indices") for r in model_rows]),
            "gpu_names": collect_union([r.get("gpu_names") for r in model_rows]),
            "gpu_uuids": collect_union([r.get("gpu_uuids") for r in model_rows]),
        }
        summaries.append(summary)

    if not rows:
        print("No benchmark rows produced. Check model paths and config.", file=sys.stderr)
        return 2

    write_csv(output_csv, rows)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": str(config_path),
        "effective_config": config,
        "summaries": summaries,
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for item in summaries:
        print(
            f"{Path(item['model_path']).name}: "
            f"TTFT mean={fmt(item['ttft_ms_mean'])} ms, "
            f"TPOT mean={fmt(item['tpot_ms_per_chunk_mean'])} ms/chunk, "
            f"peak RSS={fmt(item['peak_rss_gb_max'])} GB, "
            f"peak VRAM={fmt(item['peak_vram_gb_max'])} GB, "
            f"GPU={item.get('gpu_indices') or 'n/a'}"
        )

    print(f"\nCSV:  {as_rel(output_csv, repo_root)}")
    print(f"JSON: {as_rel(output_json, repo_root)}")
    return 0


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def max_or_none(values: list[float | None]) -> float | None:
    data = [v for v in values if v is not None]
    if not data:
        return None
    return max(data)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
