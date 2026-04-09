"""Orchestrate SFT training on RunPod: create pod, train, infer, cleanup."""

import asyncio
import json
import os
import subprocess
import sys
import time

import requests
import websockets

# Config
GPU_ID = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
TEMPLATE_ID = "runpod-torch-v240"
CLOUD_TYPE = "SECURE"
DISK_GB = 150
JUPYTER_TOKEN = "sft_token_2026"
POLL_INTERVAL = 15
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files to upload to the pod
DATA_FILES = [
    ("steiner_3k_train.jsonl", "/workspace/data/steiner_3k_train.jsonl"),
    ("steiner_val.jsonl", "/workspace/data/steiner_val.jsonl"),
    ("eval_source.jsonl", "/workspace/data/eval_source.jsonl"),
    ("glossary.json", "/workspace/data/glossary.json"),
]
CONFIG_FILE = ("sft/axolotl_config_3k.yaml", "/workspace/axolotl_config_3k.yaml")


def run_cmd(cmd):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CMD FAILED: {cmd}")
        print(f"  STDERR: {result.stderr[:500]}")
    return result.stdout.strip(), result.returncode


def create_pod():
    """Create RunPod pod and return pod ID."""
    print("Creating RunPod pod...")
    env_json = json.dumps({"JUPYTER_PASSWORD": JUPYTER_TOKEN})
    cmd = (
        f'runpodctl pod create --name "steiner-sft" '
        f'--template-id {TEMPLATE_ID} '
        f'--gpu-id "{GPU_ID}" '
        f'--cloud-type {CLOUD_TYPE} '
        f'--container-disk-in-gb {DISK_GB} '
        f"--env '{env_json}' "
        f'-o json'
    )
    out, rc = run_cmd(cmd)
    if rc != 0:
        print(f"Failed to create pod")
        sys.exit(1)
    pod = json.loads(out)
    pod_id = pod["id"]
    print(f"  Pod ID: {pod_id}")
    print(f"  GPU: {pod.get('machine', {}).get('gpuDisplayName', 'N/A')}")
    print(f"  Cost: ${pod.get('costPerHr', '?')}/hr")
    return pod_id


def wait_for_pod(pod_id, timeout=300):
    """Wait for pod to be ready."""
    print("Waiting for pod to start...")
    start = time.time()
    while time.time() - start < timeout:
        out, _ = run_cmd(f"runpodctl pod get {pod_id} -o json")
        pod = json.loads(out)
        rt = pod.get("runtime") or {}
        uptime = rt.get("uptimeInSeconds")
        if uptime and int(uptime) > 10:
            print(f"  Pod ready (uptime: {uptime}s)")
            return True
        print(f"  Not ready yet... ({int(time.time() - start)}s elapsed)")
        time.sleep(POLL_INTERVAL)
    print("  TIMEOUT waiting for pod")
    return False


def wait_for_jupyter(pod_id, timeout=120):
    """Wait for Jupyter to be accessible."""
    print("Waiting for Jupyter...")
    base_url = f"https://{pod_id}-8888.proxy.runpod.net"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(
                f"{base_url}/api/status?token={JUPYTER_TOKEN}", timeout=10
            )
            if r.status_code == 200:
                print(f"  Jupyter ready at {base_url}")
                return base_url
        except requests.exceptions.RequestException:
            pass
        time.sleep(10)
    print("  TIMEOUT waiting for Jupyter")
    return None


def create_kernel(base_url):
    """Create a Jupyter kernel and return kernel ID."""
    r = requests.post(
        f"{base_url}/api/kernels",
        headers={"Authorization": f"Token {JUPYTER_TOKEN}"},
        json={"name": "python3"},
        timeout=15,
    )
    if r.status_code != 201:
        print(f"  Failed to create kernel: {r.status_code} {r.text[:200]}")
        return None
    kernel_id = r.json()["id"]
    print(f"  Kernel created: {kernel_id}")
    return kernel_id


async def execute_on_pod(base_url, kernel_id, code, timeout=600, silent=False):
    """Execute code on the pod via Jupyter kernel websocket."""
    ws_url = f"wss://{base_url.replace('https://', '')}/api/kernels/{kernel_id}/channels?token={JUPYTER_TOKEN}"
    output_lines = []

    async with websockets.connect(ws_url, ping_interval=30) as ws:
        msg = {
            "header": {
                "msg_id": f"exec-{time.time()}",
                "msg_type": "execute_request",
                "username": "",
                "session": "sft",
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
                msg_type = data.get("msg_type", "")
                if msg_type == "stream":
                    text = data["content"]["text"]
                    if not silent:
                        print(text, end="", flush=True)
                    output_lines.append(text)
                elif msg_type == "execute_result":
                    text = data["content"]["data"].get("text/plain", "")
                    if not silent:
                        print(text)
                    output_lines.append(text)
                elif msg_type == "error":
                    err = data["content"]["evalue"]
                    print(f"ERROR: {err}")
                    output_lines.append(f"ERROR: {err}")
                    break
                elif (
                    msg_type == "status"
                    and data["content"].get("execution_state") == "idle"
                    and output_lines
                ):
                    break
        except asyncio.TimeoutError:
            print(f"\n[Timeout after {timeout}s]")

    return "".join(output_lines)


def run_on_pod(base_url, kernel_id, code, timeout=600, silent=False):
    """Sync wrapper for execute_on_pod."""
    return asyncio.run(execute_on_pod(base_url, kernel_id, code, timeout, silent))


def delete_pod(pod_id):
    """Delete the pod."""
    print(f"Deleting pod {pod_id}...")
    out, rc = run_cmd(f"runpodctl pod delete {pod_id}")
    if rc == 0:
        print("  Pod deleted.")
    else:
        print(f"  Warning: delete may have failed: {out}")


def main():
    # Step 1: Create pod
    pod_id = create_pod()

    try:
        # Step 2: Wait for pod + Jupyter
        if not wait_for_pod(pod_id):
            delete_pod(pod_id)
            sys.exit(1)

        base_url = wait_for_jupyter(pod_id)
        if not base_url:
            delete_pod(pod_id)
            sys.exit(1)

        kernel_id = create_kernel(base_url)
        if not kernel_id:
            delete_pod(pod_id)
            sys.exit(1)

        # Step 3: Setup workspace
        print("\n=== Setting up workspace ===")
        run_on_pod(base_url, kernel_id, "!mkdir -p /workspace/data /workspace/output")

        # Step 4: Upload data files
        print("\n=== Uploading data files ===")
        for local_name, remote_path in DATA_FILES:
            local_path = os.path.join(PROJECT_DIR, local_name)
            print(f"  Uploading {local_name}...")
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Write file via Python on the pod
            escaped = content.replace("\\", "\\\\").replace("'", "\\'")
            # For large files, write in chunks
            if len(content) > 100000:
                # Use base64 for large files
                import base64
                b64 = base64.b64encode(content.encode()).decode()
                code = f"""
import base64
data = base64.b64decode('{b64}')
with open('{remote_path}', 'wb') as f:
    f.write(data)
print(f'Written {{len(data)}} bytes to {remote_path}')
"""
            else:
                code = f"""
with open('{remote_path}', 'w') as f:
    f.write('''{escaped}''')
import os
print(f'Written {{os.path.getsize("{remote_path}")}} bytes to {remote_path}')
"""
            run_on_pod(base_url, kernel_id, code, timeout=30, silent=True)
            print(f"    Done.")

        # Upload config
        print(f"  Uploading config...")
        config_local = os.path.join(PROJECT_DIR, CONFIG_FILE[0])
        with open(config_local, "r") as f:
            config_content = f.read()
        escaped_config = config_content.replace("\\", "\\\\").replace("'", "\\'")
        run_on_pod(
            base_url, kernel_id,
            f"with open('{CONFIG_FILE[1]}', 'w') as f:\n    f.write('''{escaped_config}''')\nprint('Config uploaded')",
            timeout=15, silent=True,
        )
        print("    Done.")

        # Step 5: Install axolotl
        print("\n=== Installing Axolotl ===")
        run_on_pod(
            base_url, kernel_id,
            "!pip install axolotl[flash-attn] 2>&1 | tail -5",
            timeout=300,
        )

        # Step 6: Verify GPU
        print("\n=== GPU Check ===")
        run_on_pod(base_url, kernel_id, "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")

        # Step 7: Run training
        print("\n=== Starting Training ===")
        print("This will take ~20-30 minutes...")
        run_on_pod(
            base_url, kernel_id,
            "!accelerate launch -m axolotl.cli.train /workspace/axolotl_config_3k.yaml 2>&1",
            timeout=7200,  # 2 hour max
        )

        # Step 8: Check output
        print("\n=== Training Complete ===")
        run_on_pod(base_url, kernel_id, "!ls -la /workspace/output/steiner-3k/")

        # Step 9: Run inference on eval set
        print("\n=== Running Inference ===")
        inference_code = '''
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(model, "/workspace/output/steiner-3k")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

print("Loading eval data...")
with open("/workspace/data/eval_source.jsonl") as f:
    eval_data = [json.loads(line) for line in f]

system_prompt = "Translate the following English text from Rudolf Steiner into Hebrew. Preserve philosophical terminology, anthroposophical terms, and the register of the original."

results = []
for i, item in enumerate(eval_data):
    if i % 20 == 0:
        print(f"  Processing {i}/{len(eval_data)}...")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["source"]},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=2048, do_sample=False)
    translation = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    results.append({"custom_id": item["custom_id"], "translation": translation})

with open("/workspace/gpt_oss_translations.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print(f"Saved {len(results)} translations to /workspace/gpt_oss_translations.jsonl")
'''
        run_on_pod(base_url, kernel_id, inference_code, timeout=3600)

        # Step 10: Download translations
        print("\n=== Downloading Results ===")
        output = run_on_pod(
            base_url, kernel_id,
            """
import json
with open('/workspace/gpt_oss_translations.jsonl') as f:
    content = f.read()
print(content)
""",
            timeout=30,
            silent=True,
        )
        output_path = os.path.join(PROJECT_DIR, "gpt_oss_3k_translations.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output.strip())
        lines = sum(1 for _ in open(output_path))
        print(f"  Saved {lines} translations to gpt_oss_3k_translations.jsonl")

    finally:
        # Step 11: Cleanup
        delete_pod(pod_id)

    print("\n=== Done! ===")
    print("Next: run evaluation with run_eval.py")


if __name__ == "__main__":
    main()
