#!/usr/bin/env python3
"""
Measure TTFT and TPOT with llama-cpp-python (streaming).

TTFT: ms from the streaming call until first non-empty text chunk.
TPOT: mean inter-chunk latency after first chunk: (t_last - t_first) / (N-1) ms/chunk;
      each stream chunk is typically one token.

CPU:  --n-gpu-layers 0
GPU:  --n-gpu-layers -1 (requires CUDA/Metal-enabled llama-cpp-python wheel)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator


def _maybe_cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _iter_stream(llm: Any, use_chat: bool, prompt: str, max_tokens: int, temperature: float) -> Iterator[Any]:
    if use_chat:
        return llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
    return llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )


def _chunk_text(chunk: Any, use_chat: bool) -> str:
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    c0 = choices[0]
    if not isinstance(c0, dict):
        return ""
    if use_chat:
        delta = c0.get("delta") or {}
        if isinstance(delta, dict):
            return delta.get("content") or ""
        return ""
    return c0.get("text") or ""


def run_one_streamed_completion(
    llm: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    use_chat: bool,
) -> dict[str, float | int | None]:
    _maybe_cuda_sync()
    t0 = time.perf_counter()
    stream = _iter_stream(llm, use_chat, prompt, max_tokens, temperature)

    first_text_time: float | None = None
    chunk_end_times: list[float] = []
    parts: list[str] = []

    for chunk in stream:
        text = _chunk_text(chunk, use_chat)
        if not text:
            continue
        now = time.perf_counter()
        if first_text_time is None:
            first_text_time = now
        chunk_end_times.append(now)
        parts.append(text)

    _maybe_cuda_sync()
    t_end = time.perf_counter()
    generated = "".join(parts)

    if first_text_time is None:
        return {
            "ttft_ms": None,
            "tpot_ms_per_token": None,
            "e2e_ms": (t_end - t0) * 1000.0,
            "n_chunks": 0,
            "output_chars": len(generated),
        }

    n = len(chunk_end_times)
    ttft_ms = (first_text_time - t0) * 1000.0
    e2e_ms = (t_end - t0) * 1000.0

    if n >= 2:
        tpot_ms = (chunk_end_times[-1] - first_text_time) * 1000.0 / float(n - 1)
    else:
        tpot_ms = None

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms_per_token": tpot_ms,
        "e2e_ms": e2e_ms,
        "n_chunks": n,
        "output_chars": len(generated),
    }


def warmup_stream(llm: Any, prompt: str, max_tokens: int, temperature: float, use_chat: bool) -> None:
    for chunk in _iter_stream(llm, use_chat, prompt, max_tokens, temperature):
        if _chunk_text(chunk, use_chat):
            pass


def bench_config(
    model_path: Path,
    n_gpu_layers: int,
    n_ctx: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    warmup: int,
    runs: int,
    verbose: bool,
    use_chat: bool,
) -> list[dict[str, float | int | None]]:
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )

    for _ in range(warmup):
        warmup_stream(llm, prompt, max_tokens, temperature, use_chat)

    results: list[dict[str, float | int | None]] = []
    for _ in range(runs):
        results.append(run_one_streamed_completion(llm, prompt, max_tokens, temperature, use_chat))

    del llm
    return results


def summarize(name: str, rows: list[dict[str, float | int | None]]) -> None:
    ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms_per_token"] for r in rows if r.get("tpot_ms_per_token") is not None]
    e2es = [r["e2e_ms"] for r in rows if r.get("e2e_ms") is not None]

    print(f"\n=== {name} ===")
    if ttfts:
        print(f"  TTFT (ms):     mean={statistics.mean(ttfts):.2f}  stdev={statistics.pstdev(ttfts) if len(ttfts) > 1 else 0:.2f}")
    else:
        print("  TTFT (ms):     (no tokens)")
    if tpots:
        print(f"  TPOT (ms/tok): mean={statistics.mean(tpots):.2f}  stdev={statistics.pstdev(tpots) if len(tpots) > 1 else 0:.2f}")
    else:
        print("  TPOT (ms/tok): (need >= 2 stream chunks; increase --max-tokens or prompt length)")
    if e2es:
        print(f"  E2E (ms):      mean={statistics.mean(e2es):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark TTFT and TPOT (llama-cpp-python); CPU vs GPU via n_gpu_layers.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to .gguf model file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain gradient descent briefly in about 150 words.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--verbose-llama", action="store_true", help="Enable llama.cpp verbose logging")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run CPU (n_gpu_layers=0) then GPU (n_gpu_layers=-1); ignores --n-gpu-layers.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="0=CPU only, -1=all layers on GPU if supported. Default: -1 if torch sees CUDA else 0.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Write per-run metrics JSON")
    parser.add_argument(
        "--completion",
        action="store_true",
        help="Use create_completion(raw prompt) instead of create_chat_completion(user message).",
    )
    args = parser.parse_args()
    use_chat = not args.completion

    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None  # type: ignore

    cuda_ok = False
    if torch is not None:
        try:
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False

    if args.both:
        configs = [("CPU (n_gpu_layers=0)", 0), ("GPU (n_gpu_layers=-1)", -1)]
    else:
        ngl = args.n_gpu_layers
        if ngl is None:
            ngl = -1 if cuda_ok else 0
        label = f"n_gpu_layers={ngl}"
        configs = [(label, ngl)]

    all_payload: dict[str, Any] = {
        "model": str(model_path),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n_ctx": args.n_ctx,
        "warmup": args.warmup,
        "runs": args.runs,
        "cuda_available": cuda_ok,
        "use_chat_api": use_chat,
        "modes": {},
    }

    for label, n_gpu_layers in configs:
        print(f"\n>> Loading model ({label}) ...", flush=True)
        try:
            rows = bench_config(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=args.n_ctx,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                warmup=args.warmup,
                runs=args.runs,
                verbose=args.verbose_llama,
                use_chat=use_chat,
            )
        except Exception as exc:
            print(f"  Failed: {exc}", file=sys.stderr)
            all_payload["modes"][label] = {"error": str(exc)}
            continue
        summarize(label, rows)
        all_payload["modes"][label] = rows

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(all_payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
