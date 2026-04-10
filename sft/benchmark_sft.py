"""SFT speed benchmark on a single GPU.

Creates a pod, runs ~30 training steps with the actual config, measures
steps/sec and tokens/sec, then deletes the pod.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time

import requests
import websockets

# Config
CLOUD_TYPE = "SECURE"
DISK_GB = 150
JUPYTER_TOKEN = "sft_bench_2026"
MAX_STEPS = 30  # benchmark steps
TEMPLATE_ID = "runpod-torch-v240"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode


def create_pod(gpu_id, name="sft-bench"):
    print(f"Creating pod with GPU: {gpu_id}")
    env_json = json.dumps({"JUPYTER_PASSWORD": JUPYTER_TOKEN})
    cmd = (
        f'runpodctl pod create --name "{name}" '
        f'--template-id {TEMPLATE_ID} '
        f'--gpu-id "{gpu_id}" '
        f'--cloud-type {CLOUD_TYPE} '
        f'--container-disk-in-gb {DISK_GB} '
        f"--env '{env_json}' "
        f'-o json'
    )
    out, rc = run_cmd(cmd)
    if rc != 0:
        print(f"FAILED: {out}")
        return None
    pod = json.loads(out)
    print(f"  Pod ID: {pod['id']}, Cost: ${pod.get('costPerHr')}/hr")
    return pod


def wait_for_pod(pod_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        out, _ = run_cmd(f"runpodctl pod get {pod_id} -o json")
        rt = json.loads(out).get("runtime") or {}
        uptime = rt.get("uptimeInSeconds")
        if uptime and int(uptime) > 10:
            print(f"  Pod ready ({uptime}s)")
            return True
        elapsed = int(time.time() - start)
        print(f"  Waiting... ({elapsed}s)")
        time.sleep(15)
    return False


def wait_for_jupyter(pod_id, timeout=180):
    base_url = f"https://{pod_id}-8888.proxy.runpod.net"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/api/status?token={JUPYTER_TOKEN}", timeout=10)
            if r.status_code == 200:
                print(f"  Jupyter ready")
                return base_url
        except requests.exceptions.RequestException:
            pass
        time.sleep(10)
    return None


def create_kernel(base_url):
    r = requests.post(
        f"{base_url}/api/kernels",
        headers={"Authorization": f"Token {JUPYTER_TOKEN}"},
        json={"name": "python3"},
        timeout=15,
    )
    if r.status_code != 201:
        return None
    return r.json()["id"]


async def execute(base_url, kernel_id, code, timeout=600, silent=False):
    ws_url = f"wss://{base_url.replace('https://', '')}/api/kernels/{kernel_id}/channels?token={JUPYTER_TOKEN}"
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
                    if not silent:
                        print(text, end="", flush=True)
                    output.append(text)
                elif mt == "execute_result":
                    text = data["content"]["data"].get("text/plain", "")
                    if not silent:
                        print(text)
                    output.append(text)
                elif mt == "error":
                    err = data["content"]["evalue"]
                    print(f"\nERROR: {err}")
                    output.append(f"ERROR: {err}")
                    break
                elif (
                    mt == "status"
                    and data["content"].get("execution_state") == "idle"
                    and output
                ):
                    break
        except asyncio.TimeoutError:
            print(f"\n[Timeout {timeout}s]")
    return "".join(output)


def run_on_pod(base_url, kernel_id, code, timeout=600, silent=False):
    return asyncio.run(execute(base_url, kernel_id, code, timeout, silent))


def upload_file(base_url, kernel_id, local_path, remote_path):
    import base64
    with open(local_path, "rb") as f:
        content = f.read()
    b64 = base64.b64encode(content).decode()
    code = f"""
import base64, os
os.makedirs(os.path.dirname('{remote_path}'), exist_ok=True)
with open('{remote_path}', 'wb') as f:
    f.write(base64.b64decode('{b64}'))
print(f'{{os.path.getsize("{remote_path}")}} bytes')
"""
    run_on_pod(base_url, kernel_id, code, timeout=60, silent=True)


def benchmark_gpu(gpu_id, label):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"{'='*60}")

    bench_start = time.time()
    pod = create_pod(gpu_id, name=f"bench-{label[:10]}")
    if not pod:
        return None
    pod_id = pod["id"]
    cost_per_hr = pod.get("costPerHr", 0)

    result = {"gpu": label, "pod_id": pod_id, "cost_per_hr": cost_per_hr}

    try:
        if not wait_for_pod(pod_id):
            return result

        base_url = wait_for_jupyter(pod_id)
        if not base_url:
            return result

        kernel_id = create_kernel(base_url)
        if not kernel_id:
            return result

        # Verify GPU
        print("\n--- GPU info ---")
        run_on_pod(base_url, kernel_id, "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")

        # Upload data files
        print("\n--- Uploading data ---")
        for f in ["steiner_3k_train.jsonl", "steiner_val.jsonl"]:
            upload_file(base_url, kernel_id, f"{PROJECT_DIR}/{f}", f"/workspace/data/{f}")
            print(f"  {f} uploaded")

        # Upload config (modified for benchmark: max_steps)
        print("\n--- Uploading benchmark config ---")
        with open(f"{PROJECT_DIR}/sft/axolotl_config_3k.yaml") as f:
            config = f.read()
        # Replace num_epochs with max_steps for benchmark
        config = config.replace("num_epochs: 1", f"max_steps: {MAX_STEPS}\nnum_epochs: 1")
        # Use unique output dir
        config = config.replace("/workspace/output/steiner-3k", "/workspace/output/bench")
        config_b64 = __import__("base64").b64encode(config.encode()).decode()
        run_on_pod(base_url, kernel_id,
            f"import base64\nwith open('/workspace/bench_config.yaml','wb') as f: f.write(base64.b64decode('{config_b64}'))\nprint('config written')",
            timeout=30, silent=True)

        # Install axolotl
        print("\n--- Installing axolotl (may take ~3-5min) ---")
        install_start = time.time()
        run_on_pod(base_url, kernel_id, "!pip install -q axolotl[flash-attn] 2>&1 | tail -5", timeout=600)
        result["install_time_s"] = time.time() - install_start
        print(f"\n  Install: {result['install_time_s']:.0f}s")

        # Run training (will download model + train MAX_STEPS steps)
        print(f"\n--- Running {MAX_STEPS}-step benchmark ---")
        train_start = time.time()
        train_output = run_on_pod(
            base_url, kernel_id,
            "!cd /workspace && accelerate launch -m axolotl.cli.train bench_config.yaml 2>&1 | tail -100",
            timeout=3600,
        )
        result["train_total_s"] = time.time() - train_start

        # Parse training metrics from output
        # Look for patterns like "{'loss': X, 'learning_rate': Y, 'epoch': Z}" or similar
        # Axolotl prints train/runtime, train/samples_per_second
        runtime_match = re.search(r"train_runtime['\"]?\s*[:=]\s*([\d.]+)", train_output)
        samples_per_sec_match = re.search(r"train_samples_per_second['\"]?\s*[:=]\s*([\d.]+)", train_output)
        steps_per_sec_match = re.search(r"train_steps_per_second['\"]?\s*[:=]\s*([\d.]+)", train_output)

        if runtime_match:
            result["train_runtime_s"] = float(runtime_match.group(1))
        if samples_per_sec_match:
            result["samples_per_sec"] = float(samples_per_sec_match.group(1))
        if steps_per_sec_match:
            result["steps_per_sec"] = float(steps_per_sec_match.group(1))

        print(f"\n--- Results ---")
        print(json.dumps(result, indent=2))

        result["total_pod_time_s"] = time.time() - bench_start
        result["estimated_cost"] = (result["total_pod_time_s"] / 3600) * cost_per_hr

    finally:
        print(f"\nDeleting pod {pod_id}...")
        run_cmd(f"runpodctl pod delete {pod_id}")

    return result


def main():
    gpus = [
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX_PRO_6000"),
    ]

    if len(sys.argv) > 1:
        gpu_filter = sys.argv[1]
        gpus = [g for g in gpus if gpu_filter.lower() in g[1].lower()]

    results = []
    for gpu_id, label in gpus:
        r = benchmark_gpu(gpu_id, label)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(json.dumps(r, indent=2))

    with open(f"{PROJECT_DIR}/sft/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
