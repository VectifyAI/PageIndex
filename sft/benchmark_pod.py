"""Benchmark SFT speed on an EXISTING pod (skips creation).

Usage:
    python benchmark_pod.py <pod_id> <jupyter_token> <label>
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time

import requests
import websockets

MAX_STEPS = 30
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip(), r.returncode


def wait_for_pod(pod_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        out, _ = run_cmd(f"runpodctl pod get {pod_id} -o json")
        rt = json.loads(out).get("runtime") or {}
        uptime = rt.get("uptimeInSeconds")
        if uptime and int(uptime) > 10:
            print(f"[{pod_id}] Pod ready ({uptime}s)", flush=True)
            return True
        elapsed = int(time.time() - start)
        print(f"[{pod_id}] Waiting... ({elapsed}s)", flush=True)
        time.sleep(20)
    return False


def wait_for_jupyter(pod_id, token, timeout=300):
    base = f"https://{pod_id}-8888.proxy.runpod.net"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base}/api/status?token={token}", timeout=10)
            if r.status_code == 200:
                print(f"[{pod_id}] Jupyter ready", flush=True)
                return base
        except requests.exceptions.RequestException:
            pass
        time.sleep(15)
    return None


def create_kernel(base, token):
    r = requests.post(
        f"{base}/api/kernels",
        headers={"Authorization": f"Token {token}"},
        json={"name": "python3"},
        timeout=15,
    )
    return r.json()["id"] if r.status_code == 201 else None


async def execute(base, token, kernel_id, code, timeout=600, label=""):
    ws_url = f"wss://{base.replace('https://', '')}/api/kernels/{kernel_id}/channels?token={token}"
    output = []
    async with websockets.connect(ws_url, ping_interval=30, max_size=None) as ws:
        msg = {
            "header": {
                "msg_id": f"e{time.time()}",
                "msg_type": "execute_request",
                "username": "",
                "session": "s",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {"code": code, "silent": False, "store_history": True},
            "channel": "shell",
            "buffers": [],
        }
        await ws.send(json.dumps(msg))
        try:
            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=timeout)
                data = json.loads(resp)
                mt = data.get("msg_type", "")
                if mt == "stream":
                    text = data["content"]["text"]
                    for line in text.split("\n"):
                        if line.strip():
                            print(f"[{label}] {line}", flush=True)
                    output.append(text)
                elif mt == "execute_result":
                    text = data["content"]["data"].get("text/plain", "")
                    print(f"[{label}] {text}", flush=True)
                    output.append(text)
                elif mt == "error":
                    err = data["content"]["evalue"]
                    print(f"[{label}] ERROR: {err}", flush=True)
                    output.append(f"ERROR: {err}")
                    break
                elif (
                    mt == "status"
                    and data["content"].get("execution_state") == "idle"
                    and output
                ):
                    break
        except asyncio.TimeoutError:
            print(f"[{label}] [Timeout {timeout}s]", flush=True)
    return "".join(output)


def upload_file(base, token, kernel_id, local, remote, label):
    with open(local, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    code = (
        f"import base64, os\n"
        f"os.makedirs(os.path.dirname('{remote}'), exist_ok=True)\n"
        f"with open('{remote}', 'wb') as f: f.write(base64.b64decode('{b64}'))\n"
        f"print(f'{{os.path.getsize(\"{remote}\")}} bytes')"
    )
    asyncio.run(execute(base, token, kernel_id, code, timeout=120, label=label))


async def benchmark_existing_pod(pod_id, token, label):
    print(f"\n[{label}] === BENCHMARK START ({pod_id}) ===", flush=True)
    start = time.time()

    # Get pod info
    out, _ = run_cmd(f"runpodctl pod get {pod_id} -o json")
    pod = json.loads(out)
    cost = pod.get("costPerHr", 0)
    print(f"[{label}] Cost: ${cost}/hr", flush=True)

    if not wait_for_pod(pod_id):
        return {"label": label, "error": "pod_not_ready"}

    base = wait_for_jupyter(pod_id, token)
    if not base:
        return {"label": label, "error": "jupyter_not_ready"}

    kernel_id = create_kernel(base, token)
    if not kernel_id:
        return {"label": label, "error": "kernel_failed"}

    # GPU info
    await execute(base, token, kernel_id,
        "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader",
        timeout=30, label=label)

    # Upload data + config
    print(f"[{label}] Uploading files...", flush=True)
    upload_file(base, token, kernel_id,
        f"{PROJECT_DIR}/steiner_3k_train.jsonl",
        "/workspace/data/steiner_3k_train.jsonl", label)
    upload_file(base, token, kernel_id,
        f"{PROJECT_DIR}/steiner_val.jsonl",
        "/workspace/data/steiner_val.jsonl", label)

    # Modified config with max_steps
    with open(f"{PROJECT_DIR}/sft/axolotl_config_3k.yaml") as f:
        cfg = f.read()
    cfg = cfg.replace("num_epochs: 1", f"max_steps: {MAX_STEPS}\nnum_epochs: 1")
    cfg = cfg.replace("/workspace/output/steiner-3k", "/workspace/output/bench")
    cfg_b64 = base64.b64encode(cfg.encode()).decode()
    await execute(base, token, kernel_id,
        f"import base64\nwith open('/workspace/bench_config.yaml','wb') as f:\n    f.write(base64.b64decode('{cfg_b64}'))\nprint('config written')",
        timeout=15, label=label)

    # Install axolotl
    print(f"[{label}] Installing axolotl...", flush=True)
    install_start = time.time()
    await execute(base, token, kernel_id,
        "!pip install -q axolotl[flash-attn] 2>&1 | tail -3",
        timeout=900, label=label)
    install_time = time.time() - install_start
    print(f"[{label}] Install: {install_time:.0f}s", flush=True)

    # Run training
    print(f"[{label}] Running {MAX_STEPS}-step training...", flush=True)
    train_start = time.time()
    train_output = await execute(base, token, kernel_id,
        "!cd /workspace && accelerate launch -m axolotl.cli.train bench_config.yaml 2>&1 | tail -50",
        timeout=3600, label=label)
    train_time = time.time() - train_start

    # Parse metrics
    result = {
        "label": label,
        "pod_id": pod_id,
        "cost_per_hr": cost,
        "install_time_s": install_time,
        "train_time_s": train_time,
        "total_time_s": time.time() - start,
        "estimated_cost": ((time.time() - start) / 3600) * cost,
    }

    for key, pat in [
        ("train_runtime_s", r"train_runtime['\"]?\s*[:=]\s*([\d.]+)"),
        ("samples_per_sec", r"train_samples_per_second['\"]?\s*[:=]\s*([\d.]+)"),
        ("steps_per_sec", r"train_steps_per_second['\"]?\s*[:=]\s*([\d.]+)"),
    ]:
        m = re.search(pat, train_output)
        if m:
            result[key] = float(m.group(1))

    print(f"\n[{label}] === RESULT ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


async def main():
    pods = [
        ("t1w29bt005w36j", "uosnsmge86qpnv7m78tp", "H100_SXM"),
    ]

    # Run all benchmarks in parallel
    tasks = [benchmark_existing_pod(pid, tok, lbl) for pid, tok, lbl in pods]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        if isinstance(r, Exception):
            print(f"ERROR: {r}")
        else:
            print(json.dumps(r, indent=2))

    with open(f"{PROJECT_DIR}/sft/benchmark_results.json", "w") as f:
        json.dump([r for r in results if not isinstance(r, Exception)], f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
