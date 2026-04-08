---
name: flash
description: runpod-flash SDK and CLI for deploying AI workloads on Runpod serverless GPUs/CPUs.
user-invocable: true
---

# Runpod Flash

Write code locally, test with `flash run` (dev server at localhost:8888), and flash automatically provisions and deploys to remote GPUs/CPUs in the cloud. `Endpoint` handles everything.

## Setup

```bash
pip install runpod-flash # requires Python >=3.10

# auth option 1: browser-based login (saves token locally)
flash login

# auth option 2: API key via environment variable
export RUNPOD_API_KEY=your_key

flash init my-project # scaffold a new project in ./my-project
```

## CLI

```bash
flash run # start local dev server at localhost:8888
flash run --auto-provision # same, but pre-provision endpoints (no cold start)
flash build # package artifact for deployment (500MB limit)
flash build --exclude pkg1,pkg2 # exclude packages from build
flash deploy # build + deploy (auto-selects env if only one)
flash deploy --env staging # build + deploy to "staging" environment
flash deploy --app my-app --env prod # deploy a specific app to an environment
flash deploy --preview # build + launch local preview in Docker
flash env list # list deployment environments
flash env create staging # create "staging" environment
flash env get staging # show environment details + resources
flash env delete staging # delete environment + tear down resources
flash undeploy list # list all active endpoints
flash undeploy my-endpoint # remove a specific endpoint
```

## Endpoint: Three Modes

### Mode 1: Your Code (Queue-Based Decorator)

One function = one endpoint with its own workers.

```python
from runpod_flash import Endpoint, GpuGroup

@Endpoint(name="my-worker", gpu=GpuGroup.AMPERE_80, workers=5, dependencies=["torch"])
async def compute(data):
    import torch # MUST import inside function (cloudpickle)
    return {"sum": torch.tensor(data, device="cuda").sum().item()}

result = await compute([1, 2, 3])
```

### Mode 2: Your Code (Load-Balanced Routes)

Multiple HTTP routes share one pool of workers.

```python
from runpod_flash import Endpoint, GpuGroup

api = Endpoint(name="my-api", gpu=GpuGroup.ADA_24, workers=(1, 5), dependencies=["torch"])

@api.post("/predict")
async def predict(data: list[float]):
    import torch
    return {"result": torch.tensor(data, device="cuda").sum().item()}

@api.get("/health")
async def health():
    return {"status": "ok"}
```

### Mode 3: External Image (Client)

Deploy a pre-built Docker image and call it via HTTP.

```python
from runpod_flash import Endpoint, GpuGroup, PodTemplate

server = Endpoint(
    name="my-server",
    image="my-org/my-image:latest",
    gpu=GpuGroup.AMPERE_80,
    workers=1,
    env={"HF_TOKEN": "xxx"},
    template=PodTemplate(containerDiskInGb=100),
)

# LB-style
result = await server.post("/v1/completions", {"prompt": "hello"})
models = await server.get("/v1/models")

# QB-style
job = await server.run({"prompt": "hello"})
await job.wait()
print(job.output)
```

Connect to an existing endpoint by ID (no provisioning):

```python
ep = Endpoint(id="abc123")
job = await ep.runsync({"input": "hello"})
print(job.output)
```

## How Mode Is Determined

| Parameters | Mode |
|-----------|------|
| `name=` only | Decorator (your code) |
| `image=` set | Client (deploys image, then HTTP calls) |
| `id=` set | Client (connects to existing, no provisioning) |

## Endpoint Constructor

```python
Endpoint(
    name="endpoint-name",      # required (unless id= set)
    id=None,                   # connect to existing endpoint
    gpu=GpuGroup.AMPERE_80,    # single GPU type (default: ANY)
    gpu=[GpuGroup.ADA_24, GpuGroup.AMPERE_80],  # or list for auto-select by supply
    cpu=CpuInstanceType.CPU5C_4_8,  # CPU type (mutually exclusive with gpu)
    workers=5,                 # shorthand for (0, 5)
    workers=(1, 5),            # explicit (min, max)
    idle_timeout=60,           # seconds before scale-down (default: 60)
    dependencies=["torch"],    # pip packages for remote exec
    system_dependencies=["ffmpeg"],  # apt-get packages
    image="org/image:tag",     # pre-built Docker image (client mode)
    env={"KEY": "val"},        # environment variables
    volume=NetworkVolume(...), # persistent storage
    gpu_count=1,               # GPUs per worker
    template=PodTemplate(containerDiskInGb=100),
    flashboot=True,            # fast cold starts
    execution_timeout_ms=0,    # max execution time (0 = unlimited)
)
```

- `gpu=` and `cpu=` are mutually exclusive
- `workers=5` means `(0, 5)`. Default is `(0, 1)`
- `idle_timeout` default is **60 seconds**
- `flashboot=True` (default) -- enables fast cold starts via snapshot restore
- `gpu_count` -- GPUs per worker (default 1), use >1 for multi-GPU models

### NetworkVolume

```python
NetworkVolume(name="my-vol", size=100)  # size in GB, default 100
```

### PodTemplate

```python
PodTemplate(
    containerDiskInGb=64,  # container disk size (default 64)
    dockerArgs="",         # extra docker arguments
    ports="",              # exposed ports
    startScript="",        # script to run on start
)
```

## EndpointJob

Returned by `ep.run()` and `ep.runsync()` in client mode.

```python
job = await ep.run({"data": [1, 2, 3]})
await job.wait(timeout=120)  # poll until done
print(job.id, job.output, job.error, job.done)
await job.cancel()
```

## GPU Types (GpuGroup)

| Enum | GPU | VRAM |
|------|-----|------|
| `ANY` | any | varies |
| `AMPERE_16` | RTX A4000 | 16GB |
| `AMPERE_24` | RTX A5000/L4 | 24GB |
| `AMPERE_48` | A40/A6000 | 48GB |
| `AMPERE_80` | A100 | 80GB |
| `ADA_24` | RTX 4090 | 24GB |
| `ADA_32_PRO` | RTX 5090 | 32GB |
| `ADA_48_PRO` | RTX 6000 Ada | 48GB |
| `ADA_80_PRO` | H100 PCIe (80GB) / H100 HBM3 (80GB) / H100 NVL (94GB) | 80GB+ |
| `HOPPER_141` | H200 | 141GB |

## CPU Types (CpuInstanceType)

| Enum | vCPU | RAM | Max Disk | Type |
|------|------|-----|----------|------|
| `CPU3G_1_4` | 1 | 4GB | 10GB | General |
| `CPU3G_2_8` | 2 | 8GB | 20GB | General |
| `CPU3G_4_16` | 4 | 16GB | 40GB | General |
| `CPU3G_8_32` | 8 | 32GB | 80GB | General |
| `CPU3C_1_2` | 1 | 2GB | 10GB | Compute |
| `CPU3C_2_4` | 2 | 4GB | 20GB | Compute |
| `CPU3C_4_8` | 4 | 8GB | 40GB | Compute |
| `CPU3C_8_16` | 8 | 16GB | 80GB | Compute |
| `CPU5C_1_2` | 1 | 2GB | 15GB | Compute (5th gen) |
| `CPU5C_2_4` | 2 | 4GB | 30GB | Compute (5th gen) |
| `CPU5C_4_8` | 4 | 8GB | 60GB | Compute (5th gen) |
| `CPU5C_8_16` | 8 | 16GB | 120GB | Compute (5th gen) |

```python
from runpod_flash import Endpoint, CpuInstanceType

@Endpoint(name="cpu-work", cpu=CpuInstanceType.CPU5C_4_8, workers=5, dependencies=["pandas"])
async def process(data):
    import pandas as pd
    return pd.DataFrame(data).describe().to_dict()
```

## Common Patterns

### CPU + GPU Pipeline

```python
from runpod_flash import Endpoint, GpuGroup, CpuInstanceType

@Endpoint(name="preprocess", cpu=CpuInstanceType.CPU5C_4_8, workers=5, dependencies=["pandas"])
async def preprocess(raw):
    import pandas as pd
    return pd.DataFrame(raw).to_dict("records")

@Endpoint(name="infer", gpu=GpuGroup.AMPERE_80, workers=5, dependencies=["torch"])
async def infer(clean):
    import torch
    t = torch.tensor([[v for v in r.values()] for r in clean], device="cuda")
    return {"predictions": t.mean(dim=1).tolist()}

async def pipeline(data):
    return await infer(await preprocess(data))
```

### Parallel Execution

```python
import asyncio
results = await asyncio.gather(compute(a), compute(b), compute(c))
```

## Gotchas

1. **Imports outside function** -- most common error. Everything inside the decorated function.
2. **Forgetting await** -- all decorated functions and client methods need `await`.
3. **Missing dependencies** -- must list in `dependencies=[]`.
4. **gpu/cpu are exclusive** -- pick one per Endpoint.
5. **idle_timeout is seconds** -- default 60s, not minutes.
6. **10MB payload limit** -- pass URLs, not large objects.
7. **Client vs decorator** -- `image=`/`id=` = client. Otherwise = decorator.
8. **Auto GPU switching requires workers >= 5** -- pass a list of GPU types and set `workers=5` or higher.
9. **`runsync` timeout is 60s** -- cold starts can exceed 60s. Use `ep.runsync(data, timeout=120)` for first requests or use `ep.run()` + `job.wait()` instead.
