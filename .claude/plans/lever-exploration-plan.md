# SFT Fine-Tuning GPT-OSS-120B on RunPod (3k Data)

## Context

Fine-tune openai/gpt-oss-120b (native MXFP4, ~61GB) for English→Hebrew Steiner translation. Axolotl LoRA on RunPod Secure Cloud, 3k training set.

**Existing assets (already on disk)**:
- `sft/axolotl_config_3k.yaml` — config matching the spec below
- `sft/run_sft.py` — RunPod orchestrator (pod create → upload → train → infer → delete)
- `steiner_3k_train.jsonl`, `steiner_val.jsonl`, `steiner_20k_train.jsonl` at repo root
- `sft/benchmark_pod.py`, `sft/benchmark_sft.py` — pod/GPU benchmarking scripts

Phase 1 (this plan) = SFT only. Phase 2 (optional, later) = self-rejection DPO.

## Method Choice (per Axolotl docs)

**Method**: SFT (not DPO/GRPO/reward modeling) — we have English→Hebrew pairs, no preferences.

**Adapter**: LoRA with native MXFP4 (not QLoRA):
- Full FT impossible: needs ~960GB for 120B
- LoRA on MXFP4 model: ~65GB VRAM total, native precision
- QLoRA would add NF4 quantization on top of MXFP4 → quality loss for no memory benefit on our GPU

## Memory Model: Static vs Dynamic

Axolotl's rule: "Longer sequences and larger batches increase memory significantly due to activations."

| Component | VRAM | Scales with |
|---|---:|---|
| Model weights (MXFP4) | 61 GB | Fixed |
| LoRA adapters (r=8) + grads + optimizer | ~2 GB | Fixed |
| **Activations (with gradient checkpointing)** | **~2-4 GB** | **batch × seq_len** |
| **Total at seq_len=4096, batch=4** | **~65 GB** | |

**Gradient checkpointing is non-negotiable**: without it, activations would balloon to 60-80GB and OOM. With it, we pay a ~30% training slowdown but stay under 70GB.

**Sequence length choice**: `sequence_len: 4096`
- Avg example: ~600 tokens, 95th pct: ~1500, max: ~5000
- 4096 covers >99% without truncation
- With packing: ~6 examples per sequence (near-zero waste)
- Larger (8192) doubles activation memory for no real benefit on our data

## Context length vs memory headroom on 96GB GPU

| seq_len | batch | Activations (est) | Total VRAM | Headroom |
|---:|---:|---:|---:|---:|
| 2048 | 4 | ~1-2 GB | ~64 GB | 32 GB |
| 4096 | 4 | ~2-4 GB | ~65 GB | 31 GB |
| 4096 | 8 | ~4-8 GB | ~69 GB | 27 GB |
| 8192 | 4 | ~4-8 GB | ~69 GB | 27 GB |

All configurations fit comfortably on RTX PRO 6000 (96GB). Headroom suggests we could push `micro_batch` higher if needed.

## VRAM Budget (~78-83GB total)

| Component | VRAM |
|---|---:|
| Model weights (MXFP4) | ~61GB |
| LoRA adapters (r=8) | ~0.5GB |
| Optimizer (8-bit AdamW) | ~0.5GB |
| Gradients | ~0.5GB |
| Activations (grad ckpt, batch=4) | ~15-20GB |
| **Total** | **~78-83GB** |

## GPU Priority List (Secure Cloud, spot instances for cost)

| Priority | GPU | VRAM | Bandwidth | Stock | Fits? |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | H200 SXM | 141GB | 4.8 TB/s | Low | Yes, huge headroom |
| 2 | H100 SXM | 80GB | 3.35 TB/s | Medium | Tight, may need batch=1 |
| 3 | RTX PRO 6000 | 96GB | 1.79 TB/s | Medium | Yes, but 2x slower |
| 4 | MI300X | 192GB | 5.3 TB/s | Low | Yes, needs ROCm check |

Use **spot instances** — job is short (~30 min training), cheap to retry if preempted.

## Files (status)

### 1. `sft/axolotl_config_3k.yaml` — EXISTS, matches spec

Based on official `gpt-oss-20b-sft-lora-singlegpu.yaml`:

```yaml
base_model: openai/gpt-oss-120b
model_quantization_config: Mxfp4Config
model_quantization_config_kwargs:
  dequantize: true

plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin

experimental_skip_move_to_device: true

datasets:
  - path: /workspace/data/steiner_3k_train.jsonl
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content

chat_template: tokenizer_default

eval_dataset:
  - path: /workspace/data/steiner_val.jsonl
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content

dataset_prepared_path: /workspace/prepared
output_dir: /workspace/output/steiner-3k

sequence_len: 4096
sample_packing: true

adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target_linear: true

gradient_accumulation_steps: 4
micro_batch_size: 4
num_epochs: 1

optimizer: adamw_torch_8bit
lr_scheduler: constant_with_warmup
learning_rate: 2e-4
warmup_ratio: 0.1

bf16: true
tf32: true

flash_attention: true
gradient_checkpointing: true
activation_offloading: true

eval_steps: 50
logging_steps: 1
saves_per_epoch: 1
save_total_limit: 2

early_stopping_patience: 3
metric_for_best_model: eval_loss

special_tokens:
eot_tokens:
  - "<|end|>"
```

OOM fallback: micro_batch 4→2→1, then seq_len 4096→2048.

### Cost analysis: does the Gemini MQM rubric support batch API?

**Short answer: No, `EvalTask.evaluate()` is online only. But the actual cost is so low that batch mode doesn't matter.**

**Verification (via SDK introspection of `vertexai 1.146.0`)**:
- `EvalTask.evaluate()` signature has no batch parameter — only `evaluation_service_qps` (QPS rate limit for sync calls) and `retry_timeout`
- Line 613 of `_evaluation.py` logs "Multithreaded Batch Inference" — but "Batch" here means multithreaded parallelism over **online** sync calls, not the Vertex Batch Prediction API's 50% discount
- No `batch_evaluate`, no `BatchEval*` class exists anywhere in the installed SDK

**Pricing (Gemini 2.5 Flash, the default autorater for Gen AI Eval model-based metrics)**:

| Mode | Input | Output |
|---|---|---|
| Standard (online) | $0.30 / 1M tok | $2.50 / 1M tok |
| Batch (50% discount) | $0.15 / 1M tok | $1.25 / 1M tok |

**Per-sample cost estimate for the Steiner MQM rubric**:
- Input per sample: ~2,300 tokens (800 rubric template + 500 source + 500 reference + 500 response)
- Output per sample: ~400 tokens (rationale + 1-5 rating)

**Per 200-sample eval run**:

| Mode | Input cost | Output cost | Total |
|---|---|---|---|
| Online (`EvalTask.evaluate()`) | 460K × $0.30/1M = $0.14 | 80K × $2.50/1M = $0.20 | **~$0.34** |
| Batch (DIY) | 460K × $0.15/1M = $0.07 | 80K × $1.25/1M = $0.10 | **~$0.17** |

**Batch savings: ~$0.17 per run.** Negligible.

**Total composite cost per 200-sample run** (all online, Flash autorater):

| Metric | Estimated cost |
|---|---|
| `bleu` | $0 |
| `comet_22_src_ref` (managed neural) | ~$0 (preview pricing for typical volumes) |
| `metricx_24_src_ref` (managed neural) | ~$0 (preview pricing) |
| `metricx_24_src` (managed neural) | ~$0 (preview pricing) |
| `fluency` (Gemini Flash pointwise) | ~$0.10 |
| `steiner_mqm_rubric` (Gemini Flash custom rubric) | ~$0.34 |
| `terminology_recall` (custom Python) | $0 |
| **Total** | **~$0.44** |

**Conclusion**: at $0.44 per full eval run, there's no need to optimize for cost. You can run the eval 100 times for $44. Batch mode savings are pennies and not worth the engineering effort.

### Batch options (if volume ever grows to justify it)

Only revisit batch mode if any of these become true:
- Eval set grows to >5,000 samples
- >50 eval runs per week
- Evaluating hundreds of checkpoints simultaneously
- Gen AI Eval bill exceeds ~$50/month

If so, two paths exist:

**Path A — DIY via `google.genai` Batch Prediction API** (SDK v1.71.0 already installed):
1. Inline the rubric criteria + rating_rubric + evaluation_steps into a plain prompt string per sample
2. Write 200 prompts to JSONL in GCS
3. `client.batches.create(model="gemini-2.5-flash", source="gs://...")`
4. Poll for completion (typically minutes-hours, SLA up to 24h)
5. Parse output JSONL, merge scores into `run_eval.py` dataframe
- Cost: ~$0.17/run (50% of online)
- Eng effort: ~1 day
- Gives up: managed rubric parsing, automatic score normalization

**Path B — Newer unified `batch_evaluate()` API** (requires migrating off `vertexai.evaluation` to `google.genai` evals interface, not present in installed SDK):
1. Migrate `run_eval.py` from `vertexai.evaluation.EvalTask` to the newer `google.genai.Client().evals` API
2. Use `batch_evaluate()` which handles prompt construction + batch submission + parsing
- Cost: ~50% of online
- Eng effort: ~2-3 days (SDK migration + interface rewrite + retesting)
- Risk: newer API surface may still be evolving

**Recommendation: skip both. Stay online.** Online mode is fast (~2-5 min per run vs. minutes-to-hours for batch), supports rapid prompt iteration during rubric development, and costs less than $1 per full run.

## Evaluation Methodology — Final Approved Design (TL;DR)

After extensive iteration, the approved eval architecture is:

**1. Candidate scope**: comparing different post-training runs of GPT-OSS-120B against each other (not commercial baselines).

**2. Metrics suite (in `run_eval.py`)**:
- **Vertex AI EvalTask** (free preview): COMET_22, MetricX_24 (ref-based), MetricX_24 (ref-free QE). **BLEU dropped** — unreliable on Hebrew (morphology + free word order + single reference), contradicts earlier plan guidance "Skip BLEU".
- **OpenAI Batch API**: one merged LLM-judge metric — **Steiner-GEMBA-MQM** (GEMBA-MQM prompt structure + Steiner glossary injection + domain-specific severity calibration), prompt-cached
- **OpenAI Batch API (decision gates only)**: **Steiner-Pairwise** with position debiasing for A-vs-B checkpoint comparisons
- **Python local**: `terminology_recall` computed but **not in composite** — used as a deterministic correctness gate (see below)

**3. Judge model — hybrid**:
- Default / iteration: `gpt-5.4-mini` + `medium` reasoning effort for MQM, `low` for pairwise (~$0.25/run)
- Decision gates + final selection: `gpt-5.4` flagship, same effort levels (~$5.42/run)
- One-time calibration on 50 samples to verify mini tracks flagship (Spearman > 0.85 → trust mini)

**4. Composite v3 weights** (simpler — BLEU dropped, terminology_recall moved to gate):
```python
COMPOSITE_WEIGHTS_V3 = {
    "metricx_norm":           0.25,  # Vertex neural ref-based
    "metricx_src_norm":       0.20,  # Vertex neural ref-free QE (NEW)
    "comet":                  0.15,  # Vertex neural ref-based
    "steiner_gemba_mqm_norm": 0.40,  # Merged LLM-judge (NEW — primary signal)
}
# Sums to 1.00. Composite measures quality only.
```

**Terminology correctness gate** (separate from composite):
```python
# Rule: any checkpoint with terminology_recall < 0.85 is ineligible for shipping,
# regardless of composite score. Cheap, deterministic, zero-variance backstop
# on the most important failure mode. Not double-counted in the composite
# because the merged Steiner-GEMBA-MQM judge already penalizes terminology
# errors via the glossary injection.
TERMINOLOGY_GATE_THRESHOLD = 0.85
```

**Why BLEU is dropped**: Hebrew is morphologically rich with flexible word order and a single reference. BLEU's n-gram overlap penalizes valid lexical and word-order variation. MetricX and COMET catch everything BLEU catches and more. BLEU adds no independent signal, only noise. Confirmed by earlier plan guidance ("Skip BLEU — unreliable for Hebrew. Free word order, morphology").

**Why terminology_recall leaves the composite**: the merged Steiner-GEMBA-MQM prompt injects `glossary.json` and tells the judge that wrong anthroposophical terms = major terminology error. Keeping `terminology_recall` in the composite would double-count terminology errors. Instead, use it as a deterministic gate — it runs in milliseconds, costs $0, and provides a zero-variance backstop on the most critical error class (wrong Hebrew term for a Steiner concept) regardless of LLM judge variance.

**5. Project-level eval cost**: ~$35 across a full month of training iterations (30 daily mini runs + 5 flagship decision gates + 1 calibration). Negligible vs training costs.

**6. Key files to modify** (all exist):
- `run_eval.py` — add Vertex MetricX_SRC metric, add merged Steiner-GEMBA-MQM OpenAI batch path, add pairwise path, update `COMPOSITE_WEIGHTS`, wire `JUDGE_MODEL`/`JUDGE_EFFORT` env vars
- `eval_data/glossary.json` — consumed by the judge prompt prefix (for terminology calibration)
- `create_batch_gpt54mini.py` — keep as-is for generation (already uses reasoning effort correctly)

**7. What to build**:
- New module `eval/steiner_gemba_mqm.py` — renders merged prompt from `glossary.json`, serializes batch request JSONL, parses error-list JSON responses, computes deterministic score per GEMBA formula
- New module `eval/steiner_pairwise.py` — generates A/B comparison prompts with position debiasing, parses ternary judgment, aggregates win-rate
- Calibration script `eval/calibrate_judge.py` — runs both mini and flagship on 50 samples, reports Spearman, saves to `eval/calibration_report.md`

**Verification approach**:
1. Run calibration script once on 50 hand-picked samples (good/medium/bad mix)
2. Verify mini-vs-flagship Spearman > 0.85
3. Run full eval suite once with mini on existing baseline translations; verify outputs are well-formed
4. Run full eval suite once with flagship on same baselines; compare composite deltas
5. Proceed with mini as default for Phase 1 SFT iteration

---

### Clarified scope — candidate pool is GPT-OSS-120B post-training runs only

**User-confirmed scope**: this eval compares **different post-training runs of GPT-OSS-120B against each other**. It is not a benchmark against commercial systems. Candidates are GPT-OSS-120B variants with different hyperparameters, training data mixes, checkpoint steps, or SFT/DPO configurations.

**Implication 1 — sycophancy is not a concern**: `gpt-5.4-mini` is a different model family from GPT-OSS-120B (different vendor, different architecture, different training data). The LLM-as-judge self-preference failure mode only manifests when the judge has produced outputs in the candidate pool. Mini as judge is safe.

**Implication 2 — discrimination sensitivity becomes the dominant concern**: when candidates are all variants of the same 120B base model, quality differences between them are small (typically 2-10% on normalized metrics). A weak judge produces flat scores across all checkpoints and fails to discriminate. The judge must have high signal resolution.

### Judge model recommendation — hybrid: mini for iteration, flagship for gates

**Mini alone is risky** for this use case: small between-checkpoint differences may fall below mini's noise floor. **Flagship alone is wasteful** for rapid iteration during training (you'll run evals after every checkpoint — 10-30 times over a training project).

**Recommended strategy**:

| Stage | Judge | Effort | Cost/run |
|---|---|---|---|
| **Rapid iteration** (every checkpoint, rubric tuning) | `gpt-5.4-mini` | `medium` | ~$0.22 |
| **Decision gates** (phase transitions, final model selection, paper results) | `gpt-5.4` flagship | `medium` | ~$2.82 |
| **Optional calibration** (one-time, project start) | both, same 50 samples | `medium` | ~$0.75 |

**Why medium effort for mini specifically**: a weaker base model benefits more from extra reasoning budget than flagship does. Mini + medium effort is the sweet spot — compensates for the capability gap while still ~13× cheaper than flagship + medium. Low-effort mini would sacrifice too much resolution.

**Calibration protocol (one-time, ~$0.75)**:
1. Pick 50 representative samples spanning good/medium/bad translations
2. Run merged Steiner-GEMBA-MQM with mini + medium AND flagship + medium on the same 50 samples
3. Compute Spearman correlation between mini's and flagship's sample-level scores
4. If Spearman > 0.85 → trust mini for all iteration; only run flagship at decision gates
5. If Spearman 0.70-0.85 → use mini for ranking but verify absolute deltas with flagship at each phase boundary
6. If Spearman < 0.70 → mini is too noisy for this domain, upgrade iteration to flagship

**Fallback if mini discriminates poorly**: switch iteration to flagship. The cost delta across a whole training project is ~$60-100, which is still negligible vs training costs and much cheaper than shipping the wrong checkpoint.

**Config lever**: `run_eval.py` exposes `JUDGE_MODEL` and `JUDGE_EFFORT` as top-level constants (or CLI flags). Switching between mini and flagship is a one-line change. Default: mini for `python run_eval.py`, flagship explicitly via `python run_eval.py --judge=flagship`.

**Why NOT Claude Opus or Gemini as judge here**: both are strong options that also avoid sycophancy, but mini + flagship gives you a proven progression (mini is the exact same model family as flagship, so rankings should be directionally consistent). Cross-vendor judges would add unnecessary variance and a second API integration. Revisit this only if the calibration check shows mini and flagship disagree, in which case run both flagship and Opus on a tie-breaker sample.

### Reasoning effort level

GPT-5.4 reasoning effort parameter (`low`/`medium`/`high`) directly impacts judge quality and cost. Reasoning tokens are billed as output tokens but hidden from the visible response.

**Effort per task**:

| Task | Effort | Reason |
|---|---|---|
| Generation (current `create_batch_gpt54mini.py`) | `low` (keep) | Direct output, no deliberation needed |
| Steiner-GEMBA-MQM judgment | **`medium`** | Error identification + categorization + severity assignment all benefit from careful consideration |
| Pairwise comparison (A vs B) | **`low`** | Simpler ternary judgment; low effort is sufficient |

**Cost impact**: `medium` adds ~800 reasoning tokens per sample vs. ~200 at `low`. At $10/1M output tokens (GPT-5.4 batch), that's ~$1.20 extra per 200-sample metric run.

**Verification**: check `usage.completion_tokens_details.reasoning_tokens` in batch outputs to confirm actual reasoning token consumption matches estimates. Adjust effort level if costs drift.

### Merged metric: Steiner-GEMBA-MQM

**Decision**: bake Steiner-specific context into the GEMBA-MQM prompt rather than running GEMBA-MQM and a custom Steiner rubric as two separate metrics.

**Why merge**:
- Single LLM call per sample → half the cost of two metrics
- Unified judgment — generic MT errors and domain errors weighed coherently against each other in a single pass
- Keeps proven GEMBA-MQM structure: MQM category taxonomy, severity system (critical/major/minor), deterministic scoring from error list (`score = 0 - (25*crit + 5*maj + 1*min)`, clipped `[-25, 0]`, normalized to `[0, 1]`)
- Simpler pipeline: one JSONL, one column in `run_eval.py`

**What merging adds to vanilla GEMBA-MQM**:
1. **Terminology glossary injection**: paste `glossary.json` into the prompt prefix as authoritative reference terminology. Any alternative rendering is a *major terminology error*.
2. **Domain-specific severity calibration**: tell the judge explicitly that for Steiner translation, mistranslating anthroposophical terms = major (not minor); doctrinal distortion = critical; register drift = major; awkward word order = minor.
3. **Domain context paragraph**: who Steiner was, what register his lectures used, common MT failure modes (paraphrasing technical terms, modernizing register, materialist reinterpretation).

**What merging preserves**:
- MQM category taxonomy (accuracy/fluency/style/terminology/locale/non-translation)
- Severity system and deterministic scoring mechanics
- JSON error-list output format

**What you give up**: strict comparability to published WMT23 GEMBA-MQM scores (modified prompt breaks portability). Not important since this project isn't submitting to WMT.

**Validation check**: run vanilla GEMBA-MQM **once** at project start (on 50 calibration samples) to verify the merged metric tracks with vanilla. If divergence is small, trust the merged metric for all subsequent runs. ~$0.25 one-time cost.

**Prompt structure**:

```python
STEINER_GEMBA_MQM_INSTRUCTIONS = """\
You are an annotator for the quality of machine translation, specializing in
English-to-Hebrew translation of Rudolf Steiner's anthroposophical writings.

## Domain context
[Paragraph about Steiner, anthroposophy, register, common MT failure modes]

## Reference terminology (authoritative)
[Glossary injected here from glossary.json — formatted as table]
Using any alternative rendering for these terms is a MAJOR terminology error.

## Error categories and severity calibration for Steiner
[MQM categories + domain-specific severity rules:
  - CRITICAL: doctrinal distortion, non-translation, major omission
  - MAJOR: wrong anthroposophical term, register drift, missing key phrase
  - MINOR: grammar issues not impeding comprehension, word order]

## Output schema
[JSON array of {category, severity, span, explanation}]
"""
# Scoring: raw_score = -(25*crit + 5*maj + 1*min)
# Clipped to [-25, 0], normalized via (raw + 25) / 25 -> [0, 1]
```

### Revised cost table — per-judge, per-task

Batch pricing assumptions: GPT-5.4 flagship = $1.25/1M in, $10/1M out; gpt-5.4-mini (batch) = $0.075/1M in, $0.30/1M out; prompt caching gives ~90% discount on cached input (rubric/glossary prefix). Medium effort adds ~800 reasoning tokens/sample billed as output; low adds ~200.

**Per 200-sample run, merged Steiner-GEMBA-MQM:**

| Judge | Effort | Input cost | Output cost (vis + reasoning) | Total (with cache) |
|---|---|---|---|---|
| gpt-5.4-mini | medium | $0.04 | $0.07 | **~$0.11** |
| gpt-5.4 flagship | medium | $0.45 | $2.40 | **~$2.82** |

**Per 200-sample run, pairwise + position debias (2×):**

| Judge | Effort | Input cost | Output cost | Total (with cache) |
|---|---|---|---|---|
| gpt-5.4-mini | low | $0.08 | $0.07 | **~$0.14** |
| gpt-5.4 flagship | low | $1.40 | $1.20 | **~$2.60** |

**Full comprehensive eval run**:

| Path | Merged metric | Pairwise+debias | **Total / run** |
|---|---|---|---|
| **Iteration (mini, medium/low)** | $0.11 | $0.14 | **~$0.25** |
| **Decision gate (flagship, medium/low)** | $2.82 | $2.60 | **~$5.42** |
| Vertex neural (COMET + MetricX + BLEU) | — | — | $0 (preview) |

**Project-level cost estimate**: 30 days × daily mini run ($0.25) + 5 flagship decision gates ($5.42) + 1 calibration ($0.75) = **~$35 total** for a month of eval. Training costs dominate by orders of magnitude.

### Updated architecture

```
run_eval.py
  ├─ Vertex AI EvalTask (free preview pricing)
  │   ├─ COMET_22_SRC_REF
  │   ├─ MetricX_24_SRC_REF
  │   └─ MetricX_24_SRC (NEW — ref-free QE)
  │   # BLEU dropped — unreliable on Hebrew
  ├─ OpenAI Batch API (judge = mini or flagship, prompt-cached, medium/low effort)
  │   ├─ Steiner-GEMBA-MQM (NEW — merged metric, medium effort)
  │   └─ Steiner-Pairwise (NEW — position-debiased, low effort, A-vs-B only)
  └─ Python local
      └─ terminology_recall (gate only, not in composite)
```

Top-level config in `run_eval.py`:
```python
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.4-mini")  # default iteration; override for gates
JUDGE_EFFORT_MQM = "medium"
JUDGE_EFFORT_PAIRWISE = "low"
```

### Updated composite v3

```python
COMPOSITE_WEIGHTS_V3 = {
    "metricx_norm":            0.25,  # Vertex neural ref-based
    "metricx_src_norm":        0.20,  # Vertex neural ref-free QE (NEW)
    "comet":                   0.15,  # Vertex neural ref-based
    "steiner_gemba_mqm_norm":  0.40,  # Merged LLM-judge (NEW — primary signal)
}
# Sums to 1.00
```

**BLEU dropped** — unreliable on Hebrew (morphology + free word order + single reference). Confirmed by earlier plan section "Skip BLEU".

**terminology_recall NOT in composite** — the merged Steiner-GEMBA-MQM already penalizes terminology errors via glossary injection. Double-counting would over-weight the terminology signal. Instead, use `terminology_recall` as a **deterministic gate**: any checkpoint with `terminology_recall < 0.85` is ineligible for shipping regardless of composite score. Zero-variance backstop on the most important failure mode, computed in milliseconds at $0 cost.

**Pairwise NOT in composite** — it's a separate decision-gate tool run explicitly at Phase 1→2, Phase 2→4 transitions, and for A-vs-B checkpoint comparisons. Not a per-sample quality score.

---

### SUPERSEDED — original two-metric design (kept for historical context)
### GEMBA-MQM: add it alongside the custom Steiner rubric

**What it is**: GEMBA-MQM (Kocmi & Federmann, 2023) is the published state-of-the-art LLM-as-judge metric for MT. It's a **specific prompt template** asking the LLM to identify translation errors per the MQM taxonomy (accuracy / fluency / style / terminology / etc.) with severity tags (critical / major / minor). Score is computed deterministically from the error list: `score = 0 - (25*crit + 5*maj + 1*min)`, clipped to `[-25, 0]`, normalized to `[0, 1]`.

**Why it matters**:
- **WMT23 shared task finding**: GPT-4 with GEMBA-MQM prompt achieved **~0.89 Kendall's τ with expert human MQM** — highest of any automated metric
- **Published and reproducible**: exact prompt is in the paper, used in hundreds of follow-up evaluations
- **Vendor/project-independent baseline**: generic enough to compare against published numbers across MT projects
- **Complementary to custom rubric**: catches generic MT errors (omission, mistranslation, awkward fluency); custom Steiner rubric catches domain errors (wrong Hebrew term for etheric body, wrong Steiner register)

**Run both**: GEMBA-MQM as the trusted benchmark + custom Steiner rubric as the domain overlay. Neither fully replaces the other. If they agree on a translation's quality → high confidence. If they disagree → the custom rubric is catching domain-specific issues and you should investigate.

### Complete pricing — per 200-sample run, each metric × each judge

**Token assumptions**:
- GEMBA-MQM: 1,800 in + 300 out per sample (shorter, error-list output)
- Custom Steiner rubric: 2,300 in + 400 out per sample (larger rubric, structured JSON)
- Pairwise comparison: 2,800 in + 400 out per sample (both responses in prompt)
- Cacheable prefix: ~500-800 tokens of rubric instructions, cached at 90% discount after first call

| Metric | GPT-5.4 batch | **GPT-5.4 batch + cache** | Claude Opus 4.6 batch | GPT-5.4-mini (est.) |
|---|---|---|---|---|
| GEMBA-MQM (360K+60K) | $1.05 | **$0.95** | $1.65 | ~$0.15 |
| Custom Steiner rubric (460K+80K) | $1.38 | **$1.22** | $2.15 | ~$0.22 |
| Pairwise (560K+80K) | $1.50 | **$1.34** | $2.40 | ~$0.25 |
| Pairwise + position debias (2×) | $3.00 | **$2.68** | $4.80 | ~$0.50 |

### Full comprehensive suite per 200-sample run

| Judge | GEMBA-MQM | Custom Steiner | Pairwise+debias | **Total / run** | Plus Vertex neural |
|---|---|---|---|---|---|
| **GPT-5.4 batch + cache** (Recommended) | $0.95 | $1.22 | $2.68 | **$4.85** | $0 |
| Claude Opus 4.6 batch | $1.65 | $2.15 | $4.80 | **$8.60** | $0 |
| Ensemble (GPT-5.4 + Opus 4.6) | $2.60 | $3.37 | $7.48 | **$13.45** | $0 |
| GPT-5.4-mini (estimated) | $0.15 | $0.22 | $0.50 | **~$0.87** | $0 |

**All four metrics + Vertex COMET + MetricX variants** for ~$4.85/run on GPT-5.4. Full comprehensive eval costs less than $5. Daily runs for a month ≈ $150. Totally negligible vs training costs.

### Final composite with GEMBA-MQM added

```python
COMPOSITE_WEIGHTS_V3 = {
    "metricx_norm":         0.20,  # Vertex neural ref-based
    "metricx_src_norm":     0.10,  # Vertex neural ref-free QE (NEW)
    "comet":                0.10,  # Vertex neural
    "gemba_mqm_norm":       0.20,  # GPT-5.4 GEMBA-MQM (NEW — benchmark signal)
    "mqm_overall_norm":     0.20,  # GPT-5.4 custom Steiner rubric (NEW — domain signal)
    "mqm_terminology_norm": 0.05,  # custom rubric sub-score (NEW)
    "terminology_recall":   0.10,  # Python local
    "bleu":                 0.05,  # Vertex automatic
}
```

Dual LLM-judge design (generic GEMBA-MQM + domain Steiner rubric) means you get:
- **Benchmark correlation**: GEMBA-MQM ~0.89 with human per published work — trusted signal
- **Domain correlation**: custom rubric catches Steiner-specific errors
- **Disagreement detection**: if the two diverge, flag for manual review — free anomaly detection

### Implementation: single `run_judge.py` script

All four LLM-judge metrics share the same infrastructure. One script with a `--metric-type` argument:

```
run_judge.py --metric-type gemba_mqm    --batch-output ... --reference ... --source-batch ...
run_judge.py --metric-type steiner_mqm  --batch-output ... --reference ... --source-batch ...
run_judge.py --metric-type pairwise     --batch-output-a ... --batch-output-b ... --reference ... --source-batch ...
```

Same OpenAI batch upload/submit/poll/parse pattern as `create_batch_gpt54mini.py`. Different `instructions` string and `response_format` schema per metric type. Output: per-sample judge JSONL with structured scores. `run_eval.py` loads the judge JSONLs and merges by `custom_id` into the composite.

### Prompt caching setup

For the ~$0.27/run savings from prompt caching to actually materialize:
1. Put the rubric/instructions/schema in the `instructions` field of the OpenAI responses API (stable across samples)
2. OpenAI automatically caches prefixes ≥1024 tokens that are reused within ~5 minutes
3. Submit batch requests in chunks so cache is hit on subsequent requests in the same submission window
4. Verify cache hit rate via `usage.prompt_tokens_details.cached_tokens` in each response

If cache hit rate turns out to be low (e.g., if OpenAI batch doesn't preserve caching across batch items), fall back to uncached pricing — still cheap.

## LLM-as-Judge: OpenAI Batch API, not Vertex PointwiseMetric

**Decision reversal**: the earlier recommendation to implement the Steiner MQM rubric via Vertex `PointwiseMetric` is withdrawn. It should be done via **OpenAI Batch API with GPT-5.4 as judge**, mirroring the existing `create_batch_gpt54mini.py` pattern.

### Why the reversal

The project already has full OpenAI Batch API infrastructure:
- `create_batch_gpt54mini.py` — creates batch JSONL, uploads, submits to `/v1/responses` with 24h completion window
- `run_eval.py:load_batch_output()` — parses OpenAI batch output format

Adding a judge step is ~200 lines mirroring the existing pattern, not new infrastructure. The Vertex path was adding an unnecessary dependency on a different vendor + a different abstraction layer.

### Architectural distinction I should have made earlier

Two fundamentally different types of metric:

| Category | Examples | What it is | LLM-replaceable? |
|---|---|---|---|
| **Specialized neural metrics** | COMET, MetricX | Purpose-built neural nets (XLM-R / mT5) trained directly on human MQM ratings. Deterministic scalar output. | **NO** — these are regression models trained on human judgment data, not LLM prompts |
| **LLM-as-judge metrics** | GEMBA-MQM, custom rubrics, Vertex `PointwiseMetric`, Vertex `FLUENCY` | Just an LLM with a rubric prompt. No special model, no special training. | **YES** — completely replaceable by any capable LLM + good prompt |

**The correct architecture**:
- **Keep Vertex AI** for COMET + MetricX (specialized neural metrics, cannot replicate, preview pricing = free)
- **Drop Vertex AI** for MQM rubric + FLUENCY (these are just LLM prompts, do them better/cheaper with GPT-5.4 via Batch)

### Cost comparison (per 200-sample MQM rubric run, April 2026 pricing)

Assumptions: 2,300 input tok/sample (800 rubric + 500 src + 500 ref + 500 response), 400 output tok/sample.

| Path | Judge model | Mode | Input rate | Output rate | Total |
|---|---|---|---|---|---|
| Vertex `PointwiseMetric` | Gemini 2.5 Flash | Online only | $0.30/1M | $2.50/1M | **$0.34** |
| OpenAI Batch DIY | GPT-5.4 | Batch (50% off) | $1.25/1M | $10/1M | **$1.38** |
| OpenAI Batch DIY + caching | GPT-5.4 | Batch + 90% cache on rubric prefix | $1.25/1M non-cached, $0.25/1M cached | $10/1M | **$1.20** |
| Anthropic Batch DIY | Claude Opus 4.6 | Batch (50% off) | $2.50/1M | $12.50/1M | **$2.15** |

**Delta: ~$1/run.** Negligible in absolute terms — training runs cost orders of magnitude more.

### Quality comparison (the actual reason)

- **Gemini 2.5 Flash**: Google's cheap tier. Hebrew MQM judgment correlation not independently benchmarked — persistent uncertainty.
- **GPT-5.4**: current OpenAI flagship. Strong on Hebrew. Per WMT-style work, GPT-4-class models with MQM prompts correlate ~0.87-0.89 with expert humans.
- **Claude Opus 4.6**: known strong on low-resource languages including Hebrew. $5/$25 (67% cheaper than Opus 4.1).

For ~$1/run cost delta, you move from "cheap model with uncertain Hebrew MQM ability" to "flagship model with known strong Hebrew". Clear win.

### Recommended default: GPT-5.4 via OpenAI Batch

- **OPENAI_API_KEY already in place** (per CLAUDE.md)
- **Existing infrastructure pattern** (`create_batch_gpt54mini.py`) to mirror
- **No new vendor / new SDK**
- **Strong Hebrew** quality
- **Negligible cost** (~$1.20/run with prompt caching on the rubric prefix)
- **Structured output support** via `response_format: json_schema` → reliable parsing

Claude Opus 4.6 is an upgrade option if the user adds `ANTHROPIC_API_KEY`. Slightly better quality, slightly higher cost. Not required.

### Final proposed architecture

```
run_eval.py (orchestrator)
  ├─ Vertex AI EvalTask (specialized neural, free preview pricing)
  │   ├─ COMET_22_SRC_REF
  │   ├─ MetricX_24_SRC_REF
  │   └─ MetricX_24_SRC (NEW — ref-free QE)
  ├─ OpenAI Batch API (LLM-as-judge, GPT-5.4)
  │   ├─ MQM rubric (NEW — 5 dimensions + overall + error lists per dim)
  │   └─ Pairwise comparison (NEW — for head-to-head checkpoint selection)
  └─ Python local (deterministic)
      ├─ BLEU (via Vertex automatic metric)
      └─ terminology_recall (custom)
```

### Implementation sketch

New file `run_mqm_judge.py` mirroring `create_batch_gpt54mini.py`:

```python
"""Submit MQM rubric judge batch job to OpenAI."""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
MODEL = "gpt-5.4"  # stronger than the candidate

MQM_INSTRUCTIONS = """\
You are an expert evaluator of English-to-Hebrew translations of Rudolf Steiner's
anthroposophical writings.

Score on five criteria (1-5, 5=best):
1. terminology_fidelity: Steiner's key terms use conventional Hebrew equivalents
2. doctrinal_accuracy: spiritual-scientific claims preserved without distortion
3. register: formal, didactic, early-20th-century philosophical Hebrew
4. fluency: natural grammatical Hebrew, not translationese
5. semantic_accuracy: all propositional content accurately conveyed

Then assign overall_rating 1-5 (weight terminology and doctrinal most heavily).
Respond with structured JSON per the provided schema.
"""

MQM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mqm_rating",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "terminology_fidelity": {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}, "errors": {"type": "array", "items": {"type": "string"}}}, "required": ["score", "errors"]},
                "doctrinal_accuracy":   {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}, "errors": {"type": "array", "items": {"type": "string"}}}, "required": ["score", "errors"]},
                "register":             {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}, "errors": {"type": "array", "items": {"type": "string"}}}, "required": ["score", "errors"]},
                "fluency":              {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}, "errors": {"type": "array", "items": {"type": "string"}}}, "required": ["score", "errors"]},
                "semantic_accuracy":    {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}, "errors": {"type": "array", "items": {"type": "string"}}}, "required": ["score", "errors"]},
                "overall_rating":       {"type": "integer", "minimum": 1, "maximum": 5},
                "overall_justification": {"type": "string"},
            },
            "required": ["terminology_fidelity", "doctrinal_accuracy", "register", "fluency", "semantic_accuracy", "overall_rating", "overall_justification"],
            "additionalProperties": False,
        },
    },
}
```

Then: load source/reference/candidate by `custom_id` (reusing `run_eval.py` helpers), build one batch request per sample, upload + submit via the existing OpenAI batch pattern. Poll for completion. Parse each response's `output_text` as JSON, extract scalar scores, write per-sample judge JSONL.

Integration with `run_eval.py`: new `--mqm-judge-output` arg loads the judge JSONL, merges by `custom_id`, adds columns `mqm_overall`, `mqm_terminology`, `mqm_doctrinal`, `mqm_register`, `mqm_fluency`, `mqm_semantic`. Composite updated to include `mqm_overall_norm` (normalized 1-5 → 0-1).

### Updated composite

```python
COMPOSITE_WEIGHTS_V2 = {
    "metricx_norm":         0.25,  # Vertex neural ref-based (existing)
    "metricx_src_norm":     0.10,  # Vertex neural ref-free QE (NEW)
    "comet":                0.10,  # Vertex neural (existing)
    "mqm_overall_norm":     0.30,  # OpenAI GPT-5.4 MQM judge (NEW — biggest signal)
    "mqm_terminology_norm": 0.05,  # extracted sub-score (NEW)
    "terminology_recall":   0.15,  # Python local (existing)
    "bleu":                 0.05,  # Vertex automatic (existing)
}
```

Rationale: MQM overall gets 30% (highest weight) because it has the best human-correlation signal and gives dimension-level diagnostics. MetricX and COMET remain as independent neural signals. The MQM terminology sub-score is weighted 5% as a direct check on terminology fidelity that's orthogonal to the Python `terminology_recall` (which counts exact matches; MQM terminology_fidelity considers semantic equivalence of terms).

### Concrete benefits over Vertex PointwiseMetric

| Criterion | Vertex PointwiseMetric | OpenAI Batch DIY |
|---|---|---|
| Judge quality (Hebrew) | Gemini 2.5 Flash (uncertain) | GPT-5.4 (strong) or Opus 4.6 (strongest) |
| Cost per 200-sample run | $0.34 | $1.20-$2.15 |
| Prompt control | Constrained by `PointwiseMetricPromptTemplate` | Full — exact prompt + response schema |
| Structured output | Single integer 1-5 | Full JSON with 5 sub-scores + error lists + justification |
| Batch discount | No (online only in v1.146.0) | Yes (50% off) |
| Prompt caching discount | No | Yes (90% off cacheable prefix) |
| Uses existing infra | No (adds Vertex path) | Yes (mirrors `create_batch_gpt54mini.py`) |
| Dimension-level diagnostics | No | Yes (per-criterion scores + error lists) |
| Vendor consistency with generation | No (Vertex ≠ OpenAI) | Yes (same ecosystem) |

Only column where Vertex wins is raw cost, and that's a savings of ~$1/run — trivial.

### Pairwise comparison — same path, not Vertex

The pairwise comparison metric (for head-to-head checkpoint selection) should ALSO use OpenAI Batch with GPT-5.4, not Vertex `PairwiseMetric`. Same reasoning applies:
- Same infrastructure already in place
- Better judge quality on Hebrew
- Full prompt control (position-swapping for debiasing is trivial — submit each pair twice with A/B swapped)
- Structured output for reliable parsing
- Cost: ~$1.50/run including position debiasing (2x the pointwise cost because responses_A and _B are both in the prompt)

Implementation: second script `run_pairwise_judge.py` mirroring the MQM judge pattern. Called explicitly at decision gates (Phase 1→2, Phase 2→4, baseline comparisons).

### What to remove from earlier plan sections

The "Getting the Closest Metric to Human Judgment — Vertex AI Native" section above proposed `steiner_mqm_rubric` and `fluency` as Vertex `PointwiseMetric` entries in the `EvalTask` metrics list. **Those are superseded by the OpenAI Batch path described here.** The Vertex `EvalTask` call should contain only:
- `"bleu"` (automatic)
- `Comet(version="COMET_22_SRC_REF")` (existing)
- `MetricX(version="METRICX_24_SRC_REF")` (existing)
- `MetricX(version="METRICX_24_SRC")` (NEW — reference-free QE)

No LLM-as-judge metrics in the Vertex call. All LLM judging goes through OpenAI Batch.

## Getting the Closest Metric to Human Judgment — Vertex AI Native

**Constraint**: stay inside the Vertex AI Gen AI Evaluation ecosystem already used by `run_eval.py`. No new infrastructure, no external APIs, single `EvalTask.evaluate()` call.

### Complete inventory of what the installed SDK (`vertexai 1.146.0`) exposes

**Translation-specific neural metrics** (`pointwise_metric._TranslationMetric`):

| Metric | Class | Versions | Notes |
|---|---|---|---|
| COMET | `pointwise_metric.Comet` | `COMET_22_SRC_REF` (only version) | wmt22-comet-da |
| MetricX | `pointwise_metric.MetricX` | `METRICX_24_SRC_REF`, `METRICX_24_SRC`, `METRICX_24_REF` | `SRC` is reference-free QE mode |

**Automatic computation metrics** (string names in `EvalTask(metrics=[...])`): `bleu`, `exact_match`, `rouge_1`, `rouge_2`, `rouge_l`, `rouge_l_sum`.

**Model-based pointwise metrics** (Gemini as managed autorater, via `PointwiseMetric`):
- Pre-built templates in `MetricPromptTemplateExamples.Pointwise`: `FLUENCY`, `COHERENCE`, `SAFETY`, `GROUNDEDNESS`, `INSTRUCTION_FOLLOWING`, `VERBOSITY`, `TEXT_QUALITY`, `SUMMARIZATION_QUALITY`, `QUESTION_ANSWERING_QUALITY`, `MULTI_TURN_CHAT_QUALITY`, `MULTI_TURN_SAFETY`
- **Custom rubric**: `PointwiseMetric` + `PointwiseMetricPromptTemplate(metric_definition, criteria, rating_rubric, evaluation_steps, input_variables, few_shot_examples, instruction)`

**Model-based pairwise metrics** (Gemini autorater, via `PairwiseMetric`): same list + custom `PairwiseMetricPromptTemplate`.

**What is NOT in the managed service**: XCOMET / XCOMET-XL / XCOMET-XXL, MetricX Hybrid XXL, BLEURT-20, chrF / chrF++, COMET-KIWI, non-Gemini autoraters. Anything outside this list would require custom code calling external APIs, breaking the "stay inside Vertex AI" constraint.

### What `run_eval.py` already uses

```python
metrics = [
    "bleu",                                                       # automatic
    pointwise_metric.Comet(source_language="en", target_language="he"),          # COMET_22_SRC_REF
    pointwise_metric.MetricX(version="METRICX_24_SRC_REF",
                              source_language="en", target_language="he"),      # neural, ref-based
]
# + custom terminology_recall computed outside EvalTask
# Composite: 50% metricx_norm + 25% comet + 15% bleu + 10% terminology_recall
```

### Four Vertex-native additions, ranked by ROI

**Addition 1: `METRICX_24_SRC` (reference-free QE)** — 1 line of code
- `pointwise_metric.MetricX(version="METRICX_24_SRC", source_language="en", target_language="he")`
- Scores translation against source only, no reference needed
- Sidesteps single-reference penalty — if `SRC_REF` and `SRC` strongly disagree, the reference is idiosyncratic and your model may be fine despite scoring poorly on the ref-based metric
- Cost: ~$0 (same pricing as SRC_REF, doubles inference time), negligible engineering effort
- **Do this first** — trivial to add, immediately useful as sanity check

**Addition 2: Custom Gemini MQM rubric for Steiner translation** — ~1 day of prompt engineering, highest correlation win
- `PointwiseMetric` + `PointwiseMetricPromptTemplate` with Hebrew Steiner-specific criteria
- Criteria: terminology_fidelity, doctrinal_accuracy, register, fluency, semantic_accuracy
- Rating 1-5 with detailed rubric and evaluation_steps
- `input_variables=["source", "reference", "response"]` so Gemini sees all three
- This is the **Vertex-native equivalent of GEMBA-MQM** — closes most of the gap to ~0.88 correlation
- Cost: ~$2-10 per 200-sample run (Gemini autorater pricing)
- Caveat: Gemini's Hebrew translation judgment not independently benchmarked — need 20-30 sample manual validation before trusting
- **Biggest single quality win available within Vertex AI**

**Addition 3: Built-in `FLUENCY` pointwise metric** — 1 line of code
- `MetricPromptTemplateExamples.Pointwise.FLUENCY`
- Gemini scores Hebrew output for grammatical fluency 1-5
- Catches the "matches reference lexically but sounds awkward" failure that neural metrics miss
- Cost: negligible
- Bonus dimension in the composite

**Addition 4: Pairwise model comparison via `PairwiseMetric`** — ~half day
- Custom `PairwiseMetric` + `PairwiseMetricPromptTemplate` with Steiner rubric
- When comparing Model_v1 vs Model_v2, output is a **win rate** ("v2 wins 62%, ties 24%, loses 14%") instead of scalar composite difference
- More sensitive than comparing aggregate composites because Gemini sees both translations side-by-side on every sample
- Use for: checkpoint selection, Phase 1 vs Phase 2 comparison, deciding whether to commit to DPO
- Cost: ~$5-15 per 200-sample pairwise run

### Proposed updated composite (100% Vertex-native)

```python
COMPOSITE_WEIGHTS_V2 = {
    "metricx_24_src_ref":  0.25,  # existing, neural ref-based
    "metricx_24_src":      0.15,  # NEW: neural ref-free QE (Addition 1)
    "steiner_mqm_rubric":  0.25,  # NEW: Gemini LLM-as-judge custom rubric (Addition 2)
    "comet_22":            0.10,  # existing
    "fluency":             0.05,  # NEW: Gemini Hebrew fluency (Addition 3)
    "terminology_recall":  0.15,  # existing, weight up
    "bleu":                0.05,  # existing, weight down
}
```

All metrics live in a single `EvalTask.evaluate()` call. Results land in one `metrics_table` DataFrame. `run_eval.py` only needs a handful of edits.

**Estimated correlation with human judgment**: ~0.87-0.89 — within ~0.06 of the human-human ceiling. Approximately as close to human as automated metrics can currently get within the Vertex AI managed service.

### Validation step (mandatory before trusting the custom rubric)

Before relying on `steiner_mqm_rubric` for model-selection decisions:
1. Hand-pick 20-30 samples covering good / medium / bad translations
2. Manually score each on the same 1-5 rubric
3. Run the rubric metric on the same samples
4. Compute Spearman correlation between manual scores and Gemini rubric scores
5. If Spearman > 0.7 → Gemini judge is reliable on this domain, trust it
6. If Spearman 0.5-0.7 → use it as one signal but don't over-weight
7. If Spearman < 0.5 → Gemini judge doesn't understand Hebrew Steiner well enough, drop it and fall back to neural metrics + manual review

This validation is ~2-3 hours of your time and is **the difference between a metric you trust and one you're guessing at**. Do it once, reuse the validated rubric for all future runs.

### Implementation checklist

- [ ] Add `METRICX_24_SRC` metric to `run_eval.py` EvalTask (1 line)
- [ ] Add `FLUENCY` pointwise metric to EvalTask (1 line)
- [ ] Draft Steiner MQM rubric prompt template — iterate on 5-10 samples to tune criteria + rating_rubric + evaluation_steps
- [ ] Add `steiner_mqm_rubric` to EvalTask metrics list
- [ ] Update `COMPOSITE_WEIGHTS` dict and `normalize_metricx` logic for the new SRC variant
- [ ] Run validation step on 20-30 hand-scored samples; decide rubric weight
- [ ] (Optional) Add pairwise script for checkpoint comparisons
- [ ] Re-establish baselines (GPT-OSS-120B zero-shot, GPT-4.1, DeepL, Google Translate) under the new composite before Phase 1 training

## Terminology — two distinct "evals"

This project has **two completely separate evaluation mechanisms**. Be precise when referring to them:

1. **Validation loss** (a.k.a. `val_loss`, `eval_loss` in Axolotl config):
   - **When**: during training, every `eval_steps` steps
   - **What**: teacher-forced forward pass on `steiner_val.jsonl` (50 samples), computes cross-entropy on Hebrew tokens
   - **Cost**: ~15 sec per call
   - **Purpose**: drive early stopping, select best checkpoint
   - **Config keys**: `eval_dataset`, `eval_steps`, `metric_for_best_model: eval_loss`, `early_stopping_patience`, `evaluation_strategy`
   - **Tool**: built into Axolotl/HF Trainer

2. **Metric evaluation** (a.k.a. `metric eval`, `final eval`, `COMET eval`):
   - **When**: after training finishes (or between phases)
   - **What**: generate translations autoregressively on `eval_source.jsonl` (200 samples), then score with COMET / MetricX-24 / BLEU / terminology recall
   - **Cost**: ~20 min generation + 2-5 min scoring via Vertex AI
   - **Purpose**: measure real translation quality, decide ship / iterate / DPO
   - **Tool**: `run_eval.py` (Vertex AI `EvalTask`)

In Axolotl config, "eval" always means validation loss. Whenever the plan discusses COMET/MetricX, that's metric evaluation, run separately via `run_eval.py`.

### Known issue in current 3k config

`eval_steps: 50` is set, but the 3k run has only ~20 total training steps (2999 examples → ~330 packed sequences → ~20 steps at effective batch 16, 1 epoch). First eval would fire at step 50 which is never reached, so:
- **0 evals during training**
- `metric_for_best_model: eval_loss` has no data → falls back to final checkpoint
- `early_stopping_patience: 3` never triggers → dead config
- 1 save event at step ~20 (end of epoch) → **1 checkpoint on disk**

**Fix for 3k config**: change `eval_steps: 50` → `eval_steps: 10` (2 eval points: steps 10, 20) or `eval_steps: 5` (4 eval points). Either gives eval_loss signal for final checkpoint selection.

### 2. `sft/run_sft.py` — EXISTS, orchestrates end-to-end pipeline

Current flow:
1. `create_pod()` — Currently hard-coded to RTX PRO 6000 Blackwell, SECURE cloud, 150GB disk
2. `wait_for_pod()` + `wait_for_jupyter()` — polls for readiness
3. Upload `steiner_3k_train.jsonl`, `steiner_val.jsonl`, `eval_source.jsonl`, `glossary.json` via Jupyter kernel (base64 for large files)
4. Upload `axolotl_config_3k.yaml` to `/workspace/axolotl_config_3k.yaml`
5. `pip install axolotl[flash-attn]`
6. `accelerate launch -m axolotl.cli.train /workspace/axolotl_config_3k.yaml`
7. Inline inference via HF Transformers + PEFT (`PeftModel.from_pretrained`) — NOT vLLM
8. Download `gpt_oss_3k_translations.jsonl`
9. `delete_pod()` in finally block

**Gaps vs ideal**:
- Hard-codes GPU (no H200→H100→RTX fallback). Should iterate over priority list.
- Uses HF Transformers inference, not vLLM — slower but simpler. Acceptable for 200 eval examples.
- No spot-instance flag. Should add `--interruptible` for cost savings.
- Missing `eval_source.jsonl` check before upload (script will crash if absent).

### 3. Inference — inlined in run_sft.py (no separate file needed)

Uses `PeftModel.from_pretrained` to load LoRA adapter on top of base model, runs greedy decoding (`do_sample=False`) with 2048 max new tokens. Output saved to `/workspace/gpt_oss_translations.jsonl` then streamed back.

## Monitoring

- SSH/logs only (`logging_steps: 1` — every step printed)
- `runpodctl pod logs <id>` from outside

## Overfitting Strategy

1. 1 epoch — single pass, minimal overfit risk
2. Early stopping (patience=3) on val loss every 50 steps
3. Low LoRA rank (r=8) — constrains adapter capacity
4. lora_target_linear: true — all linear layers
5. No dropout (0.0) — unnecessary with 1 epoch + low rank
6. save_best_model on eval_loss

## Execution Steps (when plan approved)

1. **Pre-flight checks**:
   - Verify `.env` has RUNPOD_API_KEY and OPENAI_API_KEY
   - Verify `steiner_3k_train.jsonl`, `steiner_val.jsonl`, `eval_source.jsonl`, `glossary.json` exist at repo root
   - Verify `runpodctl config --apiKey` was set

2. **Fix `sft/run_sft.py` gaps** (minimal changes):
   - Add GPU priority list + fallback loop (H200 SXM → H100 SXM → RTX PRO 6000)
   - Add `--interruptible` flag for spot pricing
   - Add existence check for `eval_source.jsonl`, `glossary.json` before upload
   - (Optional) Switch inference to vLLM for 3-5× speedup on 200 eval samples

3. **Run training**: `python sft/run_sft.py`
   - ~20-30 min training + ~10-20 min inference on RTX PRO 6000
   - Pod auto-deleted after completion

4. **Verify output**: `gpt_oss_3k_translations.jsonl` has 200 entries matching `eval_source.jsonl` custom_ids

5. **Evaluate**: Run existing `run_eval.py` locally against Vertex AI COMET/MetricX

6. **Compare**: Side-by-side with GPT-5.4 mini baseline; decide if Phase 2 DPO is needed

## Verification

- `nvidia-smi` inside pod shows GPU + ~65GB VRAM used during training
- `tail` of training log shows `eval_loss` decreasing (expect ~2.0 → ~1.2 range)
- Final adapter files at `/workspace/output/steiner-3k/` (adapter_config.json, adapter_model.safetensors)
- `gpt_oss_3k_translations.jsonl` has 200 lines, each with `custom_id` + Hebrew `translation`
- Pod actually deleted: `runpodctl get pod` shows no `steiner-sft` pod after run

## Cost Estimate

- Model download (~65GB): depends on RunPod secure cloud bandwidth
- Training: 3k packed → ~500 sequences, 1 epoch → ~125 steps, ~20-30 min
- Inference: ~5-10 min via vLLM
- Spot pricing: cheaper than on-demand
- **Estimated: ~$2-5 total**

## Post-Training Eval

1. Serve model+adapter via vLLM on the pod
2. Run inference on eval_source.jsonl → gpt_oss_translations.jsonl
3. Download translations
4. Run `run_eval.py` locally (Vertex AI: COMET, MetricX, BLEU + terminology recall)
5. Compare with GPT-5.4 mini eval results

## Phase 2 (Optional — After SFT Evaluation)

If SFT Model_v1 has systematic errors vs. human reference, run **self-rejection DPO**:

1. **Generate**: Run Model_v1 on all 3k source texts → `model_v1_translations.jsonl`
2. **Build preference data**: Create triples `(english, chosen=human_hebrew, rejected=model_v1_hebrew)`
3. **DPO stage**: Axolotl config with `rl: dpo`, `beta: 0.1`, reference model = Model_v1
4. **Eval Model_v2**: Compare COMET/MetricX vs Model_v1

**Why this works**: rejected samples are real failure modes of our model, so DPO directly penalizes our actual mistakes rather than generic "bad" samples. Only run this if Phase 1 evaluation shows SFT leaves room for improvement.

**Decision gate**: Skip Phase 2 if Model_v1 COMET ≥ target or quality delta vs human is within noise.

## SFT Lever Exploration Methodology (autoresearch-style)

**Goal**: find the best SFT configuration (LoRA hyperparameters, optimizer, data-shape choices) through a fast iterative loop, inspired by Karpathy's autoresearch pattern — but adapted to the SFT case where each experiment is structurally more expensive and the loss surface is flatter than from-scratch training.

### Core principles

1. **Partial runs, not convergence runs.** Every lever experiment trains for a **fixed step budget** (60 steps on the 20k config, ≈43% of one epoch). Most lever effects are resolvable at this budget; running to completion wastes compute and inflates variance.
2. **Fixed step budget, not fixed wall-clock.** Different configurations have different step times (e.g., `lora_r=64` is slower per step than `lora_r=8`). Wall-clock budgeting would give faster-per-step variants more gradient updates, confounding the comparison. Use `max_steps: 60` in Axolotl.
3. **Learning-curve dominance kill.** Log eval_loss every 10 steps. If a challenger is uniformly worse than the champion at steps 10, 20, 30 by > 0.02, kill the run at step 30 — don't waste the remaining compute.
4. **Two-stage comparison.** Stage 1 is cheap eval_loss (screens clear winners/losers). Stage 2 is pairwise preference on a 20-sample probe set (resolves ties and validates Tier 4 data-shape levers where eval_loss is not comparable across variants).
5. **Persistent research pod.** Each adapter swap + probe generation requires the base 120B loaded in memory. Pod lifecycle overhead is paid once at session start, not per experiment.

### Fixed step budget: 60 steps on 20k, H200 GPU

| Budget | Steps | % of epoch 1 | RTX PRO 6000 | H200 SXM |
|---|---|---:|---:|---:|
| Short | 30 | 21% | ~30 min | ~13 min |
| **Standard (recommended)** | **60** | **43%** | **~60 min** | **~25 min** |
| Long (schedule-sensitive levers) | 140 | 100% | ~140 min | ~60 min |
| Full convergence | 280 | 200% | ~300 min | ~120 min |

**60 steps on H200 is the default experiment budget** — long enough to discriminate almost all Tier 1-3 levers, short enough to iterate 20 experiments in ~9 hours of GPU time.

### Probe set for pairwise comparison

**Definition**: 20 samples held fixed from `steiner_val.jsonl`, selected once at sweep start, **immutable for the entire lever exploration**. Save as `sft/probe_20.jsonl`.

Why fixed: probe-set rotation injects noise that can't be distinguished from lever effect. Same set, same samples, same ordering, every experiment.

Why 20 samples: balance between statistical power (≥ 15/20 wins = p < 0.05 one-tailed binomial) and generation cost (~3 min on H200, ~$0.20 judge cost). Upgrade to 40 samples if 20-sample ties become too frequent (roughly >30% of experiments inconclusive).

### Two-stage comparison protocol

**Stage 1 — learning-curve screening (free, automatic)**

After each 60-step training run, compare trajectories:

| Condition | Outcome |
|---|---|
| Challenger uniformly worse at steps 10, 20, 30 by > 0.02 (mid-run check) | **Kill early**, status = discard |
| Final eval_loss < champion − 0.010 | **Stage 1 advance**, commit, skip stage 2 |
| Final eval_loss > champion + 0.010 | **Stage 1 discard**, revert |
| Final eval_loss within ±0.010 of champion | **Promote to stage 2 pairwise** |

The 0.010 noise band is a starting estimate — calibrate empirically by running the baseline **twice** at the start of the sweep. If the two runs differ by 0.015 in final eval_loss, widen the band to 0.015.

**Stage 2 — pairwise confirmation (only for borderline cases)**

1. Load challenger adapter into the 120B base (already in memory on persistent pod)
2. Generate 20 translations on `sft/probe_20.jsonl` with greedy decoding, max_new_tokens=2048 — ~3 min on H200
3. Call Steiner-GEMBA-MQM judge in **pairwise mode** (mini + medium effort) on challenger-vs-champion
4. **Mitigate position bias**: run each sample A-vs-B and B-vs-A, count a sample as a "win" only if challenger is preferred in both orderings (strict) OR use average (lenient)
5. Advance rules (one-sided binomial, α=0.05):
   - ≥ 15/20 wins (75%) → **significant win**, advance
   - ≤ 7/20 wins (35%) → **significant loss**, discard
   - 8-14/20 → **inconclusive**, default to discard (simpler-wins rule)

Judge cost: ~40 API calls × mini pricing ≈ $0.20 per experiment. Negligible.

### Critical: which levers can use eval_loss, which REQUIRE pairwise

**Eval_loss is only comparable across variants when the loss definition is unchanged.** Some levers shift the loss target itself, making eval_loss values meaningless for cross-variant comparison:

| Lever category | eval_loss comparable? | Default comparison path |
|---|---|---|
| LR, warmup, scheduler (shape of optimization) | Yes | Stage 1 eval_loss |
| LoRA rank, alpha, dropout (capacity) | Yes | Stage 1 eval_loss |
| Optimizer, weight decay, grad clip | Yes | Stage 1 eval_loss |
| `lora_target_modules` (which layers get adapters) | Yes | Stage 1 eval_loss, often promote to stage 2 |
| `lora_use_dora`, `lora_use_rslora` (structural tricks) | Yes | Stage 1 eval_loss, often promote to stage 2 |
| Effective batch size | Yes | Stage 1 eval_loss |
| **`train_on_completion_only`** (masks prompt tokens from loss) | **NO** | **Stage 2 pairwise REQUIRED** |
| **Prompt template changes** (different conditioning) | **NO** | **Stage 2 pairwise REQUIRED** |
| **`sample_packing` on/off** (changes what "one step" means) | **NO** | **Stage 2 pairwise REQUIRED** |
| **`sequence_len` changes** (changes per-sequence loss normalization) | Partial | **Stage 2 pairwise REQUIRED** |
| **System prompt changes** | **NO** | **Stage 2 pairwise REQUIRED** |
| Label smoothing, NEFTune | Partial | Stage 1 eval_loss + stage 2 pairwise |

**Rule**: any lever that changes what tokens contribute to the loss, what the loss target is, or how sequences are packed, invalidates cross-variant eval_loss comparison. For those levers, skip stage 1 entirely and go straight to stage 2 pairwise.

### Lever tier list (priority order for exploration)

Starting from the current `sft/axolotl_config_3k.yaml` baseline (`lora_r=8`, `lora_alpha=16`, `adamw_torch_8bit`, `lr=2e-4`, `constant_with_warmup`, `warmup_ratio=0.1`, `sample_packing=true`, `sequence_len=4096`):

**Tier 0 — prerequisites (fixes, not experiments)**

| # | Fix | Why |
|---|---|---|
| 0a | `eval_steps: 50 → 5` | 20-step runs never reach step 50 → zero eval_loss signal → loop is blind |
| 0b | `saves_per_epoch: 1 → 2` | Need ≥ 2 save events for best-checkpoint selection |
| 0c | Refactor `run_sft.py` to session mode (persistent pod, skip-inference flag) | Per-run pod overhead makes the loop infeasible |
| 0d | Run baseline twice, calibrate noise floor | Sets threshold bands for advance/discard decisions |

**Tier 1 — hyperparameter sweeps (highest expected impact, eval_loss-driven)**

| # | Lever | Current | Candidates | Comparison path |
|---|---|---|---|---|
| 1 | `learning_rate` | 2e-4 | 5e-5, 1e-4, 5e-4, 1e-3 | Stage 1 eval_loss (4-way successive halving) |
| 2 | `lora_r` | 8 | 16, 32, 64 | Stage 1 eval_loss |
| 3 | `num_epochs` | 1 | 2, 3 | **Full run required** (exempt from 60-step budget) |
| 4 | `lr_scheduler` | constant+warmup | cosine, wsd | **Full run required** (late-stage decay matters) |
| 5 | `warmup_ratio` | 0.1 | 0.03, 0.05, 0.2 | Stage 1 eval_loss |

**Tier 2 — structural LoRA choices (stage 1 may be flat → promote to stage 2 often)**

| # | Lever | Current | Candidates | Comparison path |
|---|---|---|---|---|
| 6 | `lora_target_modules` | all linear | attention-only, attention+MLP | Stage 1 → stage 2 pairwise on tie |
| 7 | `lora_alpha` | 16 (=2r) | 8 (=r), 32 (=4r) | Stage 1 eval_loss |
| 8 | `lora_dropout` | 0.0 | 0.05, 0.1 | Stage 1 eval_loss (only if overfitting signal present) |
| 9 | `lora_use_rslora` | false | true | Stage 1 → stage 2 pairwise on tie |
| 10 | `lora_use_dora` | false | true | Stage 1 → stage 2 pairwise on tie |
| 11 | effective batch (grad_accum × micro_batch) | 16 | 8, 32 | Stage 1 eval_loss |

**Tier 3 — optimizer and regularization**

| # | Lever | Current | Candidates | Comparison path |
|---|---|---|---|---|
| 12 | `optimizer` | adamw_torch_8bit | adamw_bnb_8bit, paged_adamw_8bit, Lion | Stage 1 eval_loss |
| 13 | `weight_decay` | 0 (default) | 0.01, 0.1 | Stage 1 eval_loss |
| 14 | `max_grad_norm` | 1.0 (default) | 0.3, 0.5 | Stage 1 eval_loss |

**Tier 4 — data and loss shape (often biggest SFT wins; stage 2 pairwise REQUIRED)**

| # | Lever | Current | Candidates | Comparison path |
|---|---|---|---|---|
| 15 | **`train_on_completion_only`** | off | on | **Stage 2 pairwise only** (loss definition changes) |
| 16 | Prompt template | implicit via chat_template | minimal, detailed, glossary-injected | **Stage 2 pairwise only** |
| 17 | System prompt | absent | Steiner-specialist | **Stage 2 pairwise only** |
| 18 | `sample_packing` | true | false | **Stage 2 pairwise only** |
| 19 | `sequence_len` | 4096 | 2048, 8192 | **Stage 2 pairwise only** |

**Tier 5 — exotic tricks (lottery tickets, run late in the sweep)**

| # | Lever | Default | Candidates | Comparison path |
|---|---|---|---|---|
| 20 | `neftune_noise_alpha` | off | 5, 10 | Stage 1 → stage 2 on tie |
| 21 | Label smoothing | 0.0 | 0.05, 0.1 | Stage 1 eval_loss |
| 22 | PiSSA LoRA init | Kaiming (default) | PiSSA | Stage 1 eval_loss |

### 3k vs 20k lever placement (which experiments to run on which dataset)

Running experiments on the 3k subset is ~3× cheaper per run than 20k (20 steps/epoch vs 140 steps/epoch at effective batch 16), so we should push as many levers as possible down to 3k. But transferability depends on whether the lever's optimum is dataset-size-dependent. The rule:

**Transfer test**: does this lever's optimum depend on total training volume, late-stage training dynamics, or the overfitting regime? If yes → must test at 20k. If no (structural/diagnostic/data-shape) → test at 3k.

#### Group A — **BEST candidates for 3k testing** (Tier 4 data-shape levers)

| # | Lever | Why 3k transfers perfectly |
|---|---|---|
| 15 | `train_on_completion_only` | The loss-masking logic is independent of training volume. If masking helps at 3k (pairwise probe), it will help at 20k. |
| 16 | Prompt template | Template choice affects Day 1 outputs. Pairwise comparison at 3k shows the effect directly. |
| 17 | System prompt (Steiner-specialist) | Style-prompt effect is present from step 1, scales with any training volume. |
| 18 | `sample_packing` on/off | Structural change in data pipeline; effect is about sequence construction, not scale. |

These levers need **stage 2 pairwise only** (not eval_loss), so the 20-step budget is not a limitation. Pairwise generates ~20 outputs from each variant's final checkpoint, then head-to-head with a reference-free judge (COMET-Kiwi or GPT-4o rubric). Cost per 3k experiment: ~8 min training + ~3 min probe = **~11 min total on H200**.

**Run these on 3k**: Exp 11, 12 (and any prompt/template variants) — save ~30 min of H200 time per experiment vs running them on 20k.

#### Group B — **Good candidates for 3k testing** (Tier 3 optimizer/stability diagnostics)

| # | Lever | Why 3k is sufficient |
|---|---|---|
| 12 | `optimizer` (adamw_torch_8bit vs adamw_bnb_8bit vs paged_adamw_8bit) | Numerical stability and memory diagnostics: if a variant NaNs, diverges, or has pathologically bad first-20-step loss, it fails at 20k too. Use 3k as a cheap viability screen, then confirm the best 1-2 at 20k. |
| 14 | `max_grad_norm` | Gradient clipping effects are visible in the first 10 steps. Instability is loud; 3k catches it. |

**Run these on 3k as a viability screen, confirm winner on 20k**: cuts the optimizer-screening cost from 3 × 25 min at 20k to 3 × 8 min at 3k = save ~50 min.

#### Group C — **Partial transfer** (Tier 2 structural LoRA — depends on the lever)

| # | Lever | Transfer quality |
|---|---|---|
| 6 | `lora_target_modules` | **Good transfer**. Whether to adapt attention-only vs attention+MLP is a structural choice; the direction of effect is consistent across scales. Run at 3k, confirm at 20k. |
| 10 | `lora_use_dora` | **Poor transfer**. DoRA's benefit comes from late-stage training dynamics (magnitude update); at 20 steps on 3k you rarely see the advantage over vanilla LoRA. Keep at 20k. |
| 9 | `lora_use_rslora` | **Moderate transfer**. rsLoRA changes scaling law between rank and alpha, visible in convergence rate but small at 20 steps. Partial screen at 3k, definitive test at 20k. |
| 8 | `lora_dropout` | **Poor transfer**. Dropout only matters in the overfitting regime, which 3k × 20 steps never enters. Keep at 20k. |

#### Group D — **MUST run on 20k** (Tier 1 hyperparameter sweeps)

| # | Lever | Why 3k breaks the comparison |
|---|---|---|
| 1 | `learning_rate` | Optimal LR depends on total step count and gradient averaging regime. At 3k (20 steps), optimal LR is systematically higher than at 20k (140+ steps) because there's no risk of overshooting a minimum you never approach. A 3k LR sweep biases toward higher LRs than 20k would want. |
| 2 | `lora_r` (rank) | Optimal rank scales with data volume. At 3k, rank=4 or 8 saturates; at 20k, rank=16 or 32 can be utilized. Testing rank at 3k biases toward undersized adapters. |
| 3 | `num_epochs` | Already tagged "Full run required". 3k × 3 epochs = 60 steps is still in the same "undertrained" regime; the 20k overfitting dynamics never appear. |
| 4 | `lr_scheduler` (cosine vs wsd vs constant+warmup) | Late-stage decay behavior dominates. At 20 steps, cosine's decay hasn't started; at 140+ steps it's the whole story. Different regimes. |
| 5 | `warmup_ratio` | Absolute warmup step count is what matters. 10% × 20 = 2 steps (meaningless) vs 3% × 140 = 4 steps. You cannot calibrate at 3k. |
| 11 | effective batch size | Interacts with LR and total steps; regime differs across scales. |
| 13 | `weight_decay` | Regularization only bites in overfitting regime; 3k never enters it. |

**Do not run these on 3k.** The 3k run is too short (or too small) for the effect to appear or generalize.

#### Prerequisite: 3k config adjustment for lever runs

Before any 3k-based lever work, fix the 3k config beyond the Tier 0 items:

1. `eval_steps: 50 → 5` (Tier 0a, already noted)
2. `saves_per_epoch: 1 → 2` (Tier 0b, already noted)
3. **Optional**: `num_epochs: 1 → 3` to bring the 3k step budget up to ~60 steps, matching the 20k stage-1 budget. This makes eval_loss trajectories comparable across the two datasets. But it means we're training on 3 passes over a tiny dataset, which is itself non-representative — so prefer stage 2 pairwise comparison on 3k rather than eval_loss, and treat eval_loss on 3k as a coarse discriminator only.
4. Recalibrate noise floor on 3k specifically — the eval_loss noise band at 3k (20 steps) will differ from 20k (60 steps) noise band. Run `Exp 0` twice on 3k with different seeds, measure the difference, set the 3k-specific advance/discard thresholds.

#### Revised cost savings from 3k placement

If we push Group A (Tier 4 data-shape, 5 levers) and Group B (Tier 3 optimizer stability, 2-3 levers) down to 3k:
- **Savings**: ~8 experiments × ~17 min each (20k cost minus 3k cost) = ~2.2 hr of H200 time
- **New sequence cost**: ~19 hr → ~17 hr total sweep
- **Not huge, but**: the real value is faster iteration on data-shape levers, which are historically the biggest wins in SFT. Being able to try 3 prompt-template variants in ~35 min on 3k beats waiting 75 min on 20k.

### Proposed first-15-experiment sequence

This is a concrete starting point — the loop can deviate based on results. **Dataset column** indicates 3k or 20k per the placement rules above:

```
Exp 0:   [20k] baseline (60 steps on 20k with tier-0 fixes) — establishes 20k champion
Exp 0':  [20k] baseline rerun with different seed — calibrates 20k noise floor
Exp 0a:  [3k]  baseline on 3k — establishes 3k champion
Exp 0a': [3k]  baseline rerun — calibrates 3k noise floor

—— Group B: cheap viability screens on 3k ——
Exp A1:  [3k]  optimizer = adamw_bnb_8bit (cheap viability check)
Exp A2:  [3k]  optimizer = paged_adamw_8bit (cheap viability check)
                 → if a winner emerges, confirm on 20k as Exp A1'

—— Group A: data-shape levers on 3k (pairwise stage 2) ——
Exp A3:  [3k]  train_on_completion_only = true (pairwise vs 3k champion)
Exp A4:  [3k]  Steiner-specialist system prompt v1 (pairwise vs 3k champion)
Exp A5:  [3k]  Steiner-specialist system prompt v2 (pairwise vs v1 winner)
Exp A6:  [3k]  prompt template variant: glossary-injected (pairwise)
                 → promote Group A winners (up to 2) to 20k combined-config test

—— Group D: Tier 1 hyperparameter sweeps on 20k (cannot be done on 3k) ——
Exp 1-4: [20k] LR sweep {5e-5, 1e-4, 5e-4, 1e-3} via successive halving — winner becomes new 20k champion
Exp 5:   [20k] lora_r = 16 (best LR)
Exp 6:   [20k] lora_r = 32 (best LR)
Exp 7:   [20k] warmup_ratio = 0.03 (best LR, best rank)

—— Group C: structural LoRA on 20k (poor 3k transfer) ——
Exp 8:   [20k] lora_target_modules = attention-only (best combo so far)
Exp 9:   [20k] lora_use_dora = true (best combo)
Exp 10:  [20k] lora_use_rslora = true (best combo, if rank ≥ 16 won)

—— Combination + tricks ——
Exp 11:  [20k] combined champion: LR_best + rank_best + best Group A winners folded in
Exp 13:  [20k] effective batch 16 → 32 (best combo)
Exp 14:  [20k] neftune_noise_alpha = 5 (best combo)

—— stop point: run FULL eval (composite) on current 20k champion to validate ——
Exp 15+: [20k] schedule shootout — {1,2,3} epochs × {constant, cosine, wsd} on best short-run config, FULL runs
```

Expected totals:
- Exp 0, 0', 1-14 on 20k: ~13 × 25 min = ~5.4 hr on H200 (persistent pod)
- Exp 0a, 0a', A1-A6 on 3k: ~8 × 8 min = ~1.1 hr on H200 (persistent pod)
- Stage 2 pairwise for Group A variants: ~6 × 3 min = ~18 min
- Full composite validation check at exp 14: ~30 min generation + API calls
- Schedule shootout (exp 15-20): ~6 × 2 hr = ~12 hr

**Total sweep cost**: ~19 hours of H200 time ≈ $50-75 in compute. 3k placement saves ~2 hr vs running all levers on 20k. Fits comfortably in one weekend session.

**Iteration speed benefit** (the real win): Group A prompt/template variants now take ~11 min each on 3k instead of ~28 min on 20k, so we can try 3-4 template variants in the time it would take to try 1 on 20k. Data-shape levers are historically the biggest SFT wins, so fast iteration here is high-leverage.

### Levers that need FULL runs (exempt from 60-step budget)

A handful of levers specifically operate at late-training time scales and can't be evaluated at 60 steps:

- **`num_epochs`**: definitional — can only be evaluated by running the full epoch count
- **Cosine / WSD schedule tails**: the decay phase is late-training; a 60-step slice only sees the high-LR warmup phase
- **`early_stopping_patience` tuning**: requires running past the plateau point
- **Data scaling (3k vs 20k)**: comparison requires each to complete its own epoch(s)

**Handling**: after the short-run sweep has locked in the best "fast-lever" configuration, run 2-3 full-convergence runs on that best config with different `num_epochs` values and schedule choices. This is a separate "schedule shootout" batch of ~3-5 full runs (~2 hr each on H200, ~6-10 hr total).

### Successive halving for family sweeps

When testing a family of 4+ values together (LR sweep, rank sweep), use Hyperband-style halving:

```
Round 1:  launch all N candidates, 20 steps each        → kill bottom ½
Round 2:  N/2 survivors, +20 steps each (total 40)      → kill bottom ½
Round 3:  survivor(s), +20 steps (total 60)             → promote to stage 2 if borderline
```

For an 8-value LR sweep on H200:
- Round 1: 8 × 20 × 25 sec = ~67 min, kill 4
- Round 2: 4 × 20 × 25 sec = ~33 min, kill 2
- Round 3: 2 × 20 × 25 sec = ~17 min, kill 1
- Probe generation on winner: ~3 min
- **Total: ~2 hours for an 8-way LR sweep**, vs 8 × 25 = ~200 min for full 60-step runs on all.

Successive halving saves the most compute on wide sweeps (8+ candidates) and very early kills. For 2-way A/B tests, just run both at 60 steps.

### Experiment cost comparison (20 experiments)

| Method | Per-experiment time | 20-experiment sweep | Signal quality |
|---|---:|---:|---|
| Full 20k run on RTX 6000 (original plan) | ~5 hr | ~100 hr | Gold standard, infeasible |
| Full 20k run on H200 | ~2 hr | ~40 hr | Gold standard, still too slow |
| **60-step partial on H200 + Stage 1 only** | **~25 min** | **~8 hr** | **Fast screening, misses Tier 4 wins** |
| **60-step partial on H200 + Stage 2 when needed (~30%)** | **~28 min avg** | **~9 hr** | **Recommended — best quality/cost** |
| 60-step partial on H200 + Stage 2 on ALL runs | ~31 min | ~10 hr | Most rigorous, small overhead |
| 3k full sweep + validate top 3 on 20k H200 | varies | ~14 hr | Transfer risk from 3k → 20k |

**Winner**: 60-step partial on H200 with selective Stage 2 pairwise — ~9 hours for a 20-experiment sweep, no transfer risk, direct measurement on the real target dataset.

### Infrastructure changes needed in `sft/run_sft.py`

To support the autoresearch loop, `sft/run_sft.py` needs refactoring into **session mode**:

1. **`run_sft.py --session-start`** — create persistent pod, install axolotl, upload data + base model cache, leave pod alive, return pod_id
2. **`run_sft.py --session pod_id --experiment N --config path`** — run one experiment on an existing session pod: upload new config, run `max_steps: 60` training, extract eval_loss trajectory, run 20-sample probe generation, save adapter tagged by experiment N, return metrics
3. **`run_sft.py --session pod_id --pairwise challenger N --champion M`** — run pairwise judge on two experiments' probe outputs (calls out to existing Steiner-GEMBA-MQM judge in pairwise mode)
4. **`run_sft.py --session-end pod_id`** — delete pod

New helper module `sft/lever_loop.py` that wraps the above into an iteration loop:
- Reads the current best config from a state file (`sft/lever_state.json`)
- Applies a proposed edit
- Calls `run_sft.py --experiment`
- Reads the eval_loss trajectory, applies stage-1 decision rules
- If stage-1 inconclusive, calls `run_sft.py --pairwise`
- Applies stage-2 decision rules
- Advances state file and `results.tsv` on win, reverts on loss

Existing functions to reuse:
- `sft/run_sft.py:create_pod`, `wait_for_pod`, `wait_for_jupyter`, `create_kernel`, `execute_on_pod` — pod and jupyter setup, unchanged
- `sft/run_sft.py:upload_file` pattern — data/config upload, unchanged
- `eval/steiner_gemba_mqm.py` (to be built per earlier plan) — judge calls, extended with a pairwise mode

### Results tracking TSV

**`sft/lever_results.tsv`** (git-ignored, per autoresearch convention):

```
commit	stage	eval_loss_final	eval_loss_curve	probe_win_rate	peak_vram_gb	train_min	status	description
a1b2c3d	1	1.2345	[1.50,1.40,1.32,1.28,1.25,1.23]	-	62.3	24.8	keep	baseline (60 steps on 20k)
b2c3d4e	1	1.2189	[1.48,1.38,1.30,1.27,1.24,1.22]	-	62.5	24.9	keep	LR 2e-4 → 5e-4
c3d4e5f	1-kill	1.3100	[1.52,1.48,1.45,killed]	-	62.5	12.3	discard	LR 1e-3 (dominance kill @ step 30)
d4e5f6g	2	1.2180	[1.47,1.38,1.31,1.27,1.24,1.22]	15/20	64.1	28.1	keep	+ lora_r 8 → 16 (stage 2 confirmed)
e5f6g7h	2	1.2175	[...]	11/20	64.1	28.2	discard	+ DoRA (stage 2 inconclusive, simpler wins)
f6g7h8i	1-crash	0.0000	[]	-	0.0	0.0	crash	train_on_completion_only (config parse error)
```

Columns:
1. **commit**: short git hash
2. **stage**: `1` (eval_loss screening), `2` (pairwise confirmed), `1-kill` (dominance kill), `1-crash`, `2-crash`
3. **eval_loss_final**: at step 60, or last value before kill
4. **eval_loss_curve**: JSON array of eval_loss at steps 10, 20, 30, 40, 50, 60 (for post-hoc learning-curve inspection)
5. **probe_win_rate**: `X/20` if stage 2 ran, `-` if stage 1 was conclusive
6. **peak_vram_gb**: for VRAM regression tracking
7. **train_min**: training minutes only (not probe generation or pod setup)
8. **status**: `keep` / `discard` / `crash`
9. **description**: one-line summary of the change

### Decision thresholds (calibrate empirically on baseline)

Before the first real experiment, run the baseline **twice** (same config, different random seed for adapter init) to measure the noise floor:

- If two baseline runs differ in `eval_loss_final` by < 0.005 → use tight thresholds (0.010 band, 0.02 dominance)
- If they differ by 0.005-0.015 → widen bands proportionally
- If they differ by > 0.015 → noise is too high for 60-step budget, increase budget to 100 or 140 steps

This is a ~50-minute calibration cost that saves hours of false advances/reverts downstream.

### Open parameters (user-decided, not locked in this plan)

- **Start order**: Tier 0 infra fixes first (eval_steps bug, session mode refactor, baseline calibration) vs skip straight to levers (risk: loop is blind without eval_steps fix)
- **Budget stopping rule**: stop after N experiments, stop when Stage 2 pairwise plateaus for K rounds, or stop at fixed wall-clock
- **Probe set size**: start at 20, upgrade to 40 if > 30% of Stage 2 comparisons land in the 8-14/20 inconclusive band
- **Full-run schedule shootout**: run after the short-run sweep converges, or in parallel on a second pod

## Scaling Roadmap: 3k → 20k → (DPO)

### 20k SFT config delta (from 3k config)

When scaling to `steiner_20k_train.jsonl`, change these settings. **Several values are marked TBD** because they are treated as levers to be resolved by the lever-exploration methodology above, not by heuristic defaults.

```yaml
# Adapter capacity (more data → more capacity)
# TBD via Tier 1 rank sweep — candidates: 8, 16, 32, 64
lora_r: 16                  # starting point, subject to sweep
lora_alpha: 32              # keep 2×r convention

# Training schedule — resolved by lever exploration
# TBD via Tier 1 LR sweep — candidates: 5e-5, 1e-4, 2e-4, 5e-4, 1e-3
learning_rate: 1e-4         # starting point, subject to sweep

# TBD via schedule shootout (full-run phase) — see rationale below
# Candidates: 1, 2, 3 epochs × {constant+warmup, cosine, wsd}
num_epochs: TBD             # DO NOT HARDCODE — measured empirically
lr_scheduler: TBD           # DO NOT HARDCODE — measured empirically
warmup_ratio: 0.03          # starting point; Tier 1 lever

# Eval cadence (set after num_epochs is resolved)
# 140 steps per epoch (20k examples, effective batch 16)
# If num_epochs=1 → total 140 steps → eval_steps=14, saves_per_epoch=2
# If num_epochs=2 → total 280 steps → eval_steps=28, saves_per_epoch=2
# If num_epochs=3 → total 420 steps → eval_steps=35, saves_per_epoch=2
eval_steps: TBD             # derived from final num_epochs choice
saves_per_epoch: 2          # 2 saves per epoch regardless of count
save_total_limit: TBD       # = 2 × num_epochs after resolution
early_stopping_patience: 5  # don't stop on noise with longer run

# Unchanged: sequence_len, micro_batch, grad_accum, flash_attention, etc.
```

### Why `num_epochs` is not hardcoded (rationale)

The earlier version of this section hardcoded `num_epochs: 2` based on generic SFT best-practices guidance ("more data → more epochs"). That heuristic is not specifically justified for this project:

1. **Token volume at 1 epoch is already substantial**: 20k × ~2000 tok/sample = ~40M training tokens. For a frozen 120B base + LoRA adapter doing style transfer, this is in the range where one pass typically suffices.
2. **The base model is strong**: GPT-OSS-120B already speaks Hebrew and has anthroposophical priors from pretraining. SFT goal is **style activation**, not knowledge injection. Style shifts are fast.
3. **LoRA overfitting risk grows with epoch count**: with a small high-quality dataset, epoch 2+ often shows val_loss diverging from train_loss. `early_stopping_patience` exists precisely because this is expected.
4. **Style-transfer SFT is sensitive to over-training**: the model starts memorizing source-target pairs instead of learning the style mapping. Manifests as great train_loss, great val_loss, but brittle outputs on OOD samples — worst failure mode to debug post-hoc.
5. **The correct question is empirical**: run 1, 2, 3 epochs, look at val_loss at the best checkpoint, look at composite score on the test set. Pick the winner. Don't guess.

**Resolution path**: `num_epochs` is a Tier 1 lever (item #3 in the lever tier list above) marked "Full run required". After the short-run sweep (Exp 0-14) locks in the best `(LR, rank, target_modules, optimizer)` combination at fixed `num_epochs=1`, the schedule shootout phase (Exp 15+) runs full trainings at 1, 2, 3 epochs × {constant+warmup, cosine, wsd} and selects by composite score on the 200-sample test set. Cost: ~6 full runs × ~2 hr on H200 = ~12 hr = ~$20-30.

### Expected checkpoints, val_loss runs, and metric evals

| Run | Training steps | val_loss events (Eval #1) | Save events | Checkpoints on disk | Metric eval runs (Eval #2) |
|---|---:|---:|---:|---:|---:|
| 3k (Phase 1, after eval_steps fix) | ~20 | 2 (at steps 10, 20) | 1 (at step 20) | 1 | **1** at end via `run_eval.py` |
| 20k @ 1 epoch | ~140 | 10 (at steps 14, 28, …, 140) | 2 (at steps 70, 140) | 2 | 1 at end |
| 20k @ 2 epochs | ~280 | 10 (at steps 28, 56, …, 280) | 4 (at steps 70, 140, 210, 280) | 4 | 1 at end (optionally 1 per checkpoint) |
| 20k @ 3 epochs | ~420 | 12 (at steps 35, 70, …, 420) | 6 | 6 | 1 at end |

- **val_loss** = cheap teacher-forced forward pass, drives early stopping and best-checkpoint selection, happens inside Axolotl
- **Metric eval** = expensive generation + COMET/MetricX/BLEU via Vertex AI, happens after training via `run_eval.py`, measures real translation quality

**Val set for 20k run**: hold out 200 examples from `steiner_20k_train.jsonl` (leaving ~19,800 for training). The current `steiner_val.jsonl` (50 examples) is fine for eval_loss but too small for stable generation-based metrics. 200 examples gives stable ChrF/COMET-Kiwi at save points.

**Expected run time** (on H200 SXM, ~25 sec/step):
- 1 epoch: ~60 min per run
- 2 epochs: ~120 min per run
- 3 epochs: ~180 min per run

**Expected cost**: $1.50 (1 epoch) to $4.50 (3 epochs) per full run on H200.

**GPU recommendation for 20k**: Switch to H200 SXM if available. The 2x speedup usually pays for itself on a multi-hour run.

### Training-time eval strategy

| Run | Primary signal | Secondary signal |
|---|---|---|
| 3k (Phase 1) | `eval_loss` only | (end-of-run: full COMET/MetricX on 200 test set) |
| 20k (Phase 2) | `eval_loss` for early stopping | Custom callback generates 20 translations at each save point → ChrF++ + COMET-Kiwi (reference-free) logged alongside eval_loss |

**Why not COMET during every eval step?** Generation is ~15-25 tok/s on GPT-OSS-120B MXFP4 with HF Transformers. A full 50-sample generation eval takes ~25 min — would consume half the GPU budget if run at `eval_steps: 25`. Eval_loss stays fast (~15 sec forward pass) and tracks the real objective closely for style-transfer SFT.

**Why ChrF/COMET-Kiwi at save points?** COMET-Kiwi is reference-free (no need for reference translations) and ChrF++ is surface-level and fast. Both give a sanity check that eval_loss improvements translate to real quality improvements, without the cost of a full COMET-22 run.

The custom callback is ~40 lines in Axolotl (subclass `TrainerCallback`, implement `on_save`, generate on a sampled subset, log metrics to `state.log_history`). Not in the 3k run — only add it for the 20k run if you want the signal.

### Three-tier evaluation strategy

The real cost of eval comes from **generation**, not scoring. Different tiers for different contexts:

| Tier | When | What | Cost |
|---|---|---|---|
| **Tier 1: ultra-fast** | Every save during 20k training | ChrF++ on 20 generated samples via Axolotl callback | <1 sec scoring, ~2-3 min generation |
| **Tier 2: standard** | End of each run | Existing `run_eval.py` → Vertex AI: COMET + MetricX-24 + BLEU + terminology recall on 200 test set | 2-5 min API round-trip |
| **Tier 3: bulk scoring** | Phase 4 DPO filtering | Local `unbabel-comet` COMET-22 on GPU on all 20k training outputs | ~10-15 min scoring on GPU |

**Concrete metric speeds on 200 samples** (for reference):
- ChrF++ / BLEU (sacrebleu, pure Python): <1 sec
- COMET-22 local (0.5B, GPU): ~30 sec
- Vertex AI COMET + MetricX-24 + BLEU (managed): 2-5 min total
- XCOMET-XL local (3.5B, GPU): ~1-2 min
- MetricX-24-XXL local (13B, GPU): ~10-15 min

**The real bottleneck is generation, not scoring**: generating 200 translations from GPT-OSS-120B takes ~20-25 min on RTX PRO 6000. Scoring those 200 afterward takes 30 sec - 5 min depending on metric. For Phase 4 DPO filtering, generating 20k candidates takes ~30-40 hrs on RTX PRO 6000 or ~10-15 hrs on H200 — that dominates the cost, not the scoring step.

**Recommendation for Phase 4**: install `unbabel-comet` directly on the training pod after training completes, run bulk COMET-22 scoring before deleting the pod. Avoids Vertex AI rate limits at 20k scale and reuses the already-paid-for GPU.

### Sample size guidance

| Stage | Minimum | Good | Diminishing returns |
|---|---:|---:|---:|
| SFT (style transfer) | 500 | 2000-5000 | 10000-20000 |
| SFT (terminology + style) | 2000 | 20000 | 50000+ |
| Self-rejection DPO | 1000 | 3000-5000 | 10000+ |

### Recommended rollout sequence

1. **Phase 1**: SFT on 3k (this plan) — validates pipeline, establishes baseline
2. **Phase 2**: SFT on 20k with delta config above — likely the "good enough" model
3. **Phase 3 decision gate**: Evaluate Phase 2 Model_v2 against baselines (see Evaluation Methodology below)
4. **Phase 4 (optional)**: Self-rejection DPO
   - Generate Model_v2 outputs on all 20k inputs
   - Filter to bottom 15-20% by quality (~3k-4k worst outputs)
   - DPO config: `rl: dpo`, `beta: 0.1`, `learning_rate: 5e-7`, 1 epoch
5. **Phase 5 (rarely needed)**: Iterative DPO or additional data collection

**Most likely**: the 3k→20k jump gives the biggest gain. DPO is polish, not the main course.

## Model Comparison via Composite Score

**The target metric is the composite score from `run_eval.py`** — a single weighted number summarizing how close each model's translations are to the human reference. **Higher composite = closer to human = better model.**

Current weighting (from `run_eval.py`):
```python
COMPOSITE_WEIGHTS = {
    "metricx_norm":       0.50,  # best-calibrated metric
    "comet":              0.25,  # standard neural metric
    "bleu":               0.15,  # surface-level (weak for Hebrew)
    "terminology_recall": 0.10,  # anthroposophical term preservation
}
```

### Scoring table (build incrementally)

Run `run_eval.py` on each model's outputs against the frozen 200-sample test set, record the composite score, compare. Establish baselines **before Phase 1 training** starts.

| Model | COMET | MetricX_norm | BLEU | TermRecall | Composite | Role |
|---|---|---|---|---|---|---|
| GPT-OSS-120B zero-shot | | | | | | **Floor** — must beat this |
| Google Translate | | | | | | Commercial baseline |
| DeepL | | | | | | Commercial baseline |
| Claude Opus 4.6 | | | | | | Commercial baseline |
| GPT-4.1 zero-shot | | | | | | Commercial baseline |
| Model_v1 (3k SFT) | | | | | | Phase 1 output |
| Model_v2 (20k SFT) | | | | | | Phase 2 output |
| Model_v3 (DPO) | | | | | | Phase 4 output (optional) |
| Human-human (50 samples) | | | | | | **Ceiling** — noise floor |

### Decision procedure

After each training phase, run `run_eval.py` on the new model and look up the composite:

1. **`Composite_new ≥ Composite_best_commercial`** → ship it, beats all commercial systems
2. **`Composite_new ≥ Composite_human_human − 0.02`** → at the noise floor, further improvement is invisible to metrics, stop
3. **`Composite_new ≥ Composite_prev + 0.02`** → meaningful improvement, worth continuing to next phase
4. **`Composite_new < Composite_prev`** → regression, debug before proceeding
5. **Tie within ±0.01** → fall back to manual review of 20 samples

### Caveats

- **Learned metrics ≠ human judgment**: COMET/MetricX correlate ~0.7-0.8 with humans, not 1.0. Always layer manual review on top of composite scores.
- **Single reference punishes valid alternatives**: model may produce a perfectly good Hebrew translation that differs lexically from the reference; composite will penalize it. Human-human baseline tells you the ceiling this imposes.
- **Absolute composite values are meaningless** — always compare against the baselines above, never against a fixed threshold like "composite ≥ 0.80".
- **If metrics disagree** (COMET says v2 is better, MetricX says v3 is better): the weighted composite is the tie-breaker, and if composites are within ~0.01, trust manual review.

Absolute COMET numbers are meaningless in isolation — they depend on language pair, test set difficulty, domain, and metric version. **Targets are defined by relative comparisons to baselines**, not by fixed thresholds.

### Metrics to compute on the frozen 200-sample test set

| Metric | Type | Why |
|---|---|---|
| COMET-22 | Learned, ref-based | Industry standard, supports Hebrew |
| XCOMET-XL | Learned, ref-based | Better calibrated, catches hallucinations |
| MetricX-24 | Learned, ref-based | Independent signal from COMET (mT5-based) |
| ChrF++ | Surface, ref-based | Essential for morphologically rich Hebrew |
| COMET-Kiwi | Learned, ref-free | Monitor during training without references |

**Skip BLEU** — unreliable for Hebrew (free word order, morphology). Run all metrics together; if they disagree, investigate before trusting any one.

### Baselines to establish BEFORE training (these define "target")

Compute all of these on the 200-sample test set before Phase 1:

1. **GPT-OSS-120B zero-shot** (the floor) — run base model unchanged with a zero-shot translation prompt. Fine-tuned model must beat this.
2. **Commercial systems** (the ship threshold) — Google Translate, DeepL, Claude Opus 4.6, GPT-4.1 on the same 200 inputs. Target = match or beat the best.
3. **Human-human agreement** (the ceiling) — second human translator does 50 inputs from scratch without seeing the reference. COMET between the two humans = the metric's noise floor. If Model_v1 gets within 2 points of this, further improvement is invisible to metrics.
4. **Previous checkpoint** (progress tracking) — Phase 2 needs +2-3 COMET over Phase 1; DPO needs +1-2 COMET over Phase 2.

### Qualitative review (mandatory — numbers lie)

Set aside 20 samples from the test set for manual side-by-side review. Rate each on 1-5 scales for:
- **Adequacy** (meaning preserved)
- **Fluency** (natural Hebrew)
- **Style** (Steiner's voice)
- **Terminology** (anthroposophical terms correct)

Note systematic errors (e.g., "model always translates X as Y").

### Error categorization → treatment (drives the Phase 4 DPO decision)

| Error type | Fix with |
|---|---|
| Wrong meaning | More SFT data |
| Right meaning, wrong terminology | DPO or glossary prompting |
| Right meaning + terms, wrong style | DPO |
| Hallucinations / omissions | DPO (biggest wins) |
| Broken Hebrew grammar | More SFT data |

### Concrete decision criteria

```
SHIP Model_v2 if ALL of:
  1. COMET_v2 ≥ COMET_baseline + 5 points
  2. COMET_v2 ≥ best_commercial − 1 point
  3. XCOMET agrees with COMET on delta direction
  4. Manual review: ≥ 80% of 20 samples rated 4+ on adequacy
  5. Manual review: no systematic terminology errors

PROCEED TO DPO (Phase 4) if:
  1. COMET_v2 ≥ COMET_baseline + 3 points (improved but not enough)
  2. Manual review: 50-80% rated 4+
  3. Systematic errors concentrated in specific categories

ABANDON / RETHINK if:
  1. COMET_v2 ≤ COMET_v1 (more data hurt — bug somewhere)
  2. Manual review: < 50% rated 4+
  3. Fluency dropped below baseline (model broke Hebrew)
```

### Evaluation pipeline layout

```
eval/
├── test_200.jsonl              # frozen, never touch again
├── baselines/
│   ├── gpt_oss_zero_shot.jsonl
│   ├── google_translate.jsonl
│   ├── deepl.jsonl
│   ├── claude_opus.jsonl
│   ├── gpt_4_1.jsonl
│   └── human_second_translator.jsonl  (50 samples)
├── outputs/
│   ├── model_v1_3k.jsonl
│   ├── model_v2_20k.jsonl
│   └── model_v3_dpo.jsonl
├── metrics/
│   └── scores.csv              # all metrics × all models
└── manual_review/
    └── 20_samples.md
```

Build one `score_all.py` that takes an output jsonl and appends COMET, XCOMET, MetricX, ChrF++, Kiwi rows to `scores.csv`. Run it once per checkpoint. The decision gate becomes a pandas query.
