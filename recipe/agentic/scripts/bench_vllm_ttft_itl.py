#!/usr/bin/env python3
"""测 vLLM OpenAI 兼容接口的 TTFT / ITL（流式 chat/completions）。

TTFT：发出 HTTP 请求 → 收到第一个带 content 的 SSE chunk。
ITL：相邻两个 content chunk 的到达间隔（不含 TTFT）。vLLM 流式默认一 token 一块。

用法（服务已起）::

    python recipe/agentic/scripts/bench_vllm_ttft_itl.py \\
      --url http://127.0.0.1:18101/v1 \\
      --model /mnt/workspace/ckpt/Qwen3.5-9B

一键起 TP=2 再测：见同目录 run_bench_vllm_tp2.sh。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 1000.0:.2f}"


def _summarize(name: str, xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_ms": round(statistics.mean(xs) * 1000.0, 3),
        "p50_ms": round((_pct(xs, 50) or 0.0) * 1000.0, 3),
        "p90_ms": round((_pct(xs, 90) or 0.0) * 1000.0, 3),
        "p99_ms": round((_pct(xs, 99) or 0.0) * 1000.0, 3),
        "min_ms": round(min(xs) * 1000.0, 3),
        "max_ms": round(max(xs) * 1000.0, 3),
    }


def wait_ready(base: str, timeout_s: float) -> None:
    url = base.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[bench] server ready: {url}", flush=True)
                    return
        except (URLError, TimeoutError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise SystemExit(f"[bench] server not ready after {timeout_s:.0f}s ({url}): {last}")


def build_prompt(prompt_tokens: int, model: str) -> str:
    filler = (
        "Write a careful step-by-step explanation. "
        "Include numbered lists and short examples. "
    )
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        ids: List[int] = []
        while len(ids) < prompt_tokens:
            ids.extend(tok.encode(filler, add_special_tokens=False))
        text = tok.decode(ids[:prompt_tokens], skip_special_tokens=True)
        return text
    except Exception as exc:
        print(f"[bench] tokenizer unavailable ({exc}); pad by chars", flush=True)
        # ~4 chars / token 近似
        return (filler * ((prompt_tokens * 4) // max(len(filler), 1) + 1))[: prompt_tokens * 4]


def stream_one(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> Dict[str, Any]:
    import requests

    chat_url = url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.perf_counter()
    ttft: Optional[float] = None
    prev: Optional[float] = None
    itls: List[float] = []
    n_chunks = 0
    text_parts: List[str] = []
    usage: Dict[str, Any] = {}
    finish = ""

    resp = requests.post(chat_url, json=payload, stream=True, timeout=timeout_s)
    if resp.status_code == 400 and "stream_options" in (resp.text or ""):
        resp.close()
        payload.pop("stream_options", None)
        resp = requests.post(chat_url, json=payload, stream=True, timeout=timeout_s)
    with resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith(":"):
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = dict(obj["usage"])
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if choices[0].get("finish_reason"):
                finish = str(choices[0]["finish_reason"])
            if not piece:
                continue
            now = time.perf_counter()
            n_chunks += 1
            text_parts.append(piece)
            if ttft is None:
                ttft = now - t_send
            elif prev is not None:
                itls.append(now - prev)
            prev = now

    e2e = time.perf_counter() - t_send
    n_out = int(usage.get("completion_tokens") or n_chunks)
    decode_s = (e2e - ttft) if ttft is not None and n_out > 1 else None
    tpot = (decode_s / (n_out - 1)) if decode_s is not None and n_out > 1 else None
    return {
        "ok": ttft is not None and n_chunks > 0,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "itls_s": itls,
        "mean_itl_s": statistics.mean(itls) if itls else None,
        "tpot_s": tpot,
        "n_chunks": n_chunks,
        "completion_tokens": n_out,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "finish_reason": finish,
        "out_chars": sum(len(p) for p in text_parts),
    }


def _print_summary(title: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ttfts = [r["ttft_s"] for r in recs if r.get("ttft_s") is not None]
    e2es = [r["e2e_s"] for r in recs if r.get("e2e_s") is not None]
    itls: List[float] = []
    tpots = [r["tpot_s"] for r in recs if r.get("tpot_s") is not None]
    for r in recs:
        itls.extend(r.get("itls_s") or [])
    out_tok = [r["completion_tokens"] for r in recs]
    tok_per_s = []
    for r in recs:
        if r.get("tpot_s"):
            tok_per_s.append(1.0 / r["tpot_s"])
    summary = {
        "n_ok": sum(1 for r in recs if r.get("ok")),
        "n_fail": sum(1 for r in recs if not r.get("ok")),
        "ttft": _summarize("ttft", ttfts),
        "itl": _summarize("itl", itls),
        "tpot": _summarize("tpot", tpots),
        "e2e": _summarize("e2e", e2es),
        "completion_tokens_mean": round(statistics.mean(out_tok), 2) if out_tok else 0,
        "decode_tok_per_s_mean": round(statistics.mean(tok_per_s), 3) if tok_per_s else None,
    }
    print(f"\n===== {title} =====", flush=True)
    print(
        f"ok={summary['n_ok']} fail={summary['n_fail']}  "
        f"out_tokens_mean={summary['completion_tokens_mean']}  "
        f"decode_tok/s_mean={summary['decode_tok_per_s_mean']}",
        flush=True,
    )
    print(
        f"{'metric':<8} {'n':>6} {'mean':>10} {'p50':>10} {'p90':>10} {'p99':>10} {'min':>10} {'max':>10}  (ms)",
        flush=True,
    )
    for key in ("ttft", "itl", "tpot", "e2e"):
        s = summary[key]
        if not s.get("n"):
            print(f"{key:<8} {'0':>6}", flush=True)
            continue
        print(
            f"{key:<8} {s['n']:>6} "
            f"{s['mean_ms']:>10.2f} {s['p50_ms']:>10.2f} {s['p90_ms']:>10.2f} "
            f"{s['p99_ms']:>10.2f} {s['min_ms']:>10.2f} {s['max_ms']:>10.2f}",
            flush=True,
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM TTFT / ITL streaming bench")
    ap.add_argument("--url", default="http://127.0.0.1:18101/v1")
    ap.add_argument("--model", default="/mnt/workspace/ckpt/Qwen3.5-9B")
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--wait-ready-s", type=float, default=600.0)
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    if args.prompt_tokens < 8 or args.max_tokens < 8:
        raise SystemExit("--prompt-tokens / --max-tokens 至少 8，ITL 才有样本")

    if not args.skip_wait:
        wait_ready(args.url, args.wait_ready_s)

    print(
        f"[bench] url={args.url} model={args.model} "
        f"prompt_tokens≈{args.prompt_tokens} max_tokens={args.max_tokens} "
        f"warmup={args.warmup} runs={args.runs} conc={args.concurrency}",
        flush=True,
    )
    prompt = build_prompt(args.prompt_tokens, args.model)
    print(f"[bench] prompt_chars={len(prompt)}", flush=True)

    def _one() -> Dict[str, Any]:
        return stream_one(
            args.url, args.model, prompt, args.max_tokens, args.temperature, args.timeout
        )

    if args.warmup > 0:
        print(f"[bench] warmup ×{args.warmup}", flush=True)
        for i in range(args.warmup):
            rec = _one()
            print(
                f"  warmup[{i}] ok={rec['ok']} ttft={_fmt_ms(rec['ttft_s'])}ms "
                f"mean_itl={_fmt_ms(rec['mean_itl_s'])}ms tokens={rec['completion_tokens']}",
                flush=True,
            )

    recs: List[Dict[str, Any]] = []
    t0 = time.time()
    if args.concurrency <= 1:
        for i in range(args.runs):
            rec = _one()
            recs.append(rec)
            print(
                f"  run[{i}] ok={rec['ok']} ttft={_fmt_ms(rec['ttft_s'])}ms "
                f"mean_itl={_fmt_ms(rec['mean_itl_s'])}ms "
                f"tpot={_fmt_ms(rec['tpot_s'])}ms tokens={rec['completion_tokens']} "
                f"e2e={_fmt_ms(rec['e2e_s'])}ms",
                flush=True,
            )
    else:
        print(f"[bench] concurrent runs={args.runs} workers={args.concurrency}", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(_one) for _ in range(args.runs)]
            for i, fut in enumerate(as_completed(futs)):
                rec = fut.result()
                recs.append(rec)
                print(
                    f"  done[{i}] ok={rec['ok']} ttft={_fmt_ms(rec['ttft_s'])}ms "
                    f"mean_itl={_fmt_ms(rec['mean_itl_s'])}ms tokens={rec['completion_tokens']}",
                    flush=True,
                )
    wall = time.time() - t0
    summary = _print_summary(f"conc={args.concurrency}", recs)
    out = {
        "url": args.url,
        "model": args.model,
        "prompt_tokens_target": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "warmup": args.warmup,
        "runs": args.runs,
        "concurrency": args.concurrency,
        "wall_sec": round(wall, 3),
        "summary": summary,
        "per_request": [
            {k: v for k, v in r.items() if k != "itls_s"}
            | {"n_itl": len(r.get("itls_s") or [])}
            for r in recs
        ],
    }
    print(f"\n[bench] wall={wall:.1f}s", flush=True)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[bench] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
