# A 7-Step Framework for Selecting an Open-Weight Model from the Llama Family

> **Scope:** A generalized, industry-standard evaluation and selection process for choosing a Llama-family model (or deciding against one) for a production application.
> **Audience:** AI/ML engineers, platform engineers, and the tech leads who sign off on the decision.
> **Companion to:** the closed-API selection process (leaderboard filter → custom eval). This document covers what changes when the model runs on **your** hardware.
>
> ⚠️ **Verify before you rely on it.** Model line-ups, license text, GPU prices and API prices all move fast. Every specific number in this document is illustrative and should be re-checked against primary sources (the model card, `LICENSE` file, and your cloud's current price list) at the time you run this process.

---

## Table of Contents

- [Why open-weight selection is a different problem](#why-open-weight-selection-is-a-different-problem)
- [The framework at a glance](#the-framework-at-a-glance)
- [Step 0 — The pre-flight gate: should you self-host at all?](#step-0--the-pre-flight-gate-should-you-self-host-at-all)
- [Step 1 — Define the task contract and non-negotiables](#step-1--define-the-task-contract-and-non-negotiables)
- [Step 2 — Apply hard gates: license, modality, language, sovereignty](#step-2--apply-hard-gates-license-modality-language-sovereignty)
- [Step 3 — Size the hardware envelope](#step-3--size-the-hardware-envelope)
- [Step 4 — Shortlist candidates (filtration, not selection)](#step-4--shortlist-candidates-filtration-not-selection)
- [Step 5 — Build the golden dataset and eval harness](#step-5--build-the-golden-dataset-and-eval-harness)
- [Step 6 — Run the eval: quality, cost, latency, safety](#step-6--run-the-eval-quality-cost-latency-safety)
- [Step 7 — Decide, document, deploy, and re-evaluate](#step-7--decide-document-deploy-and-re-evaluate)
- [Decision matrix template](#decision-matrix-template)
- [Worked example](#worked-example)
- [Appendix A — Llama family map](#appendix-a--llama-family-map)
- [Appendix B — VRAM and throughput reference](#appendix-b--vram-and-throughput-reference)
- [Appendix C — Common failure modes](#appendix-c--common-failure-modes)
- [Appendix D — Glossary](#appendix-d--glossary)

---

## Why open-weight selection is a different problem

When you evaluate a hosted API, the vendor absorbs everything except prompt design. When you evaluate an open-weight model, you inherit the whole stack — and four new decision axes appear:

| Axis | Hosted API | Open-weight (Llama) |
|---|---|---|
| **Cost model** | $ per million tokens, linear with usage | GPU-hours, fixed with capacity — **cost per token depends on your utilization** |
| **The unit of choice** | A model name | A **(model, size, quantization, serving stack, hardware)** tuple |
| **Legal exposure** | Vendor ToS | **You** hold the license obligations; some are structural (MAU caps, naming, jurisdiction) |
| **Quality ceiling** | Fixed by vendor | **Raisable** via fine-tuning, LoRA, distillation, better decoding |

**The most important structural consequence:** with an API you evaluate ~5 models. With open weights you evaluate ~5 models × 3 quantizations × 2 serving configs. Combinatorics will eat your schedule unless the framework prunes aggressively and early. Steps 2 and 3 exist to do that pruning **before** you spend a single GPU-hour on quality evaluation.

---

## The framework at a glance

```
STEP 0   PRE-FLIGHT GATE
         Should this be self-hosted at all? Honest TCO + team-capability check.
         │  ← most projects should stop here and use an API
         ▼
STEP 1   TASK CONTRACT
         Task taxonomy, quality bar, latency SLO, throughput, context,
         compliance posture. Written down, signed off.
         │
         ▼
STEP 2   HARD GATES               ← eliminates, never ranks
         License, modality, language coverage, data residency, jurisdiction.
         Output: a legally and functionally permissible candidate set.
         │
         ▼
STEP 3   HARDWARE ENVELOPE        ← eliminates on physics and budget
         VRAM math, KV-cache math, quantization, parallelism, batch size.
         Output: which (size, quant) combos actually fit your budget.
         │
         ▼
STEP 4   SHORTLIST                ← filtration via public evidence
         Leaderboards, benchmarks, model cards, community reports.
         Output: 3–6 deployable configurations to test.
         │
         ▼
STEP 5   GOLDEN DATASET + HARNESS ← the thing that makes the decision defensible
         Stratified, human-validated ground truth. Graders. Judge calibration.
         │
         ▼
STEP 6   RUN THE EVAL             ← quality AND cost AND latency AND safety
         At production quantization. Under production load. With statistics.
         │
         ▼
STEP 7   DECIDE + OPERATIONALIZE
         Weighted decision matrix, written rationale, rollout plan,
         production monitoring, scheduled re-evaluation.
```

**Governing principle, same as for APIs:** *public benchmarks filter; your own eval selects.* Nothing in Steps 2–4 chooses a model. It only decides what is worth testing.

---

## Step 0 — The pre-flight gate: should you self-host at all?

Run this before anything else. It is the cheapest step and it saves the most money.

### The five questions

**1. Is there a hard requirement that forces self-hosting?**

Legitimate forcing functions:
- Data cannot leave your network (regulated data, contractual obligation, air-gapped environment)
- Data residency in a jurisdiction the API vendors don't serve
- You need weight-level control: custom fine-tuning, logit access, custom decoding, embedding extraction
- Vendor-independence is a board-level requirement
- Per-token economics at your volume are genuinely prohibitive

If none apply, self-hosting is a preference, not a requirement — and should be justified on TCO alone.

**2. What is your sustained utilization?**

This is the single number that decides the economics.

```
self-hosted cost per 1M output tokens
    = (GPU_count × GPU_hourly_cost) / (throughput_tokens_per_sec × 3600) × 1,000,000
```

A GPU costs the same whether it is at 5% or 95% utilization. Self-hosting wins on cost **only under sustained, high, predictable load**. Spiky or low-volume traffic is API territory — you would be renting idle silicon.

**3. Do you have the team?**

Self-hosting is an ongoing operational commitment, not a one-time deployment: inference-server tuning, GPU driver and CUDA management, autoscaling, quantization pipelines, model upgrades, incident response at 3 AM. Budget **at least one engineer's sustained attention**. That salary belongs in the TCO.

**4. What is the total cost of ownership?**

```
TCO_monthly =
      GPU/instance cost
    + storage (weights, checkpoints, artifacts)
    + network egress
    + observability and logging stack
    + engineering time (loaded salary × FTE fraction)
    + a redundancy factor for failover capacity
```

Compare against the API baseline **including the API's zero ops cost**. Teams routinely compare a raw GPU hourly rate against an API bill and conclude self-hosting is 5× cheaper, having omitted everything except the GPU.

**5. What is the break-even volume?**

```
break_even_tokens_per_month
    = (GPU_count × GPU_hourly × 730 + ops_overhead) / api_blended_price_per_token
```

If your projected volume is below break-even, and no hard requirement from question 1 applies, **stop here and use an API**. Document the analysis — you will be asked again in six months.

### Gate exit criteria

> ✅ Proceed to Step 1 only if: a hard requirement forces self-hosting, **or** projected sustained volume clears break-even with margin, **and** the team capability exists.

---

## Step 1 — Define the task contract and non-negotiables

Requirements are written **before** any model is looked at. Otherwise you will find yourself reverse-engineering a justification for whatever the top of the leaderboard happened to be.

### 1.1 Classify the task

Different task classes have completely different model-size elasticity — how much quality you gain by scaling up:

| Task class | Size elasticity | Typical viable floor | Notes |
|---|---|---|---|
| Classification / routing / intent | **Very low** | 1B–3B | Small fine-tuned models often beat large zero-shot ones |
| Extraction / structured output | Low | 3B–8B | Constrained decoding matters more than size |
| Summarization | Moderate | 8B | Instruction-following quality dominates |
| RAG answer synthesis | Moderate | 8B–70B | Grounding discipline > raw knowledge |
| Code generation | **High** | 70B+ | Steep quality curve with scale |
| Multi-step reasoning / agentic | **Very high** | 70B+ | Largest gap between small and large |
| Tool calling / function calling | High | 8B+ | Depends heavily on training coverage — test explicitly |
| Long-context synthesis | High | Depends on context need | Advertised context ≠ effective context |

> **The most valuable output of this sub-step:** most production tasks are *not* frontier-reasoning tasks. If yours is classification or extraction, you are probably choosing between 1B and 8B, and the entire hardware conversation simplifies enormously.

### 1.2 Define the quality bar — as a number, not an adjective

Bad: *"the model should be accurate."*
Good: *"≥ 92% exact-match on the golden set, with ≤ 2% catastrophic-error rate, at 95% confidence."*

Specify:
- **Primary metric** and its target (exact match, F1, pass@1, rubric score, execution accuracy…)
- **Guardrail metrics** that must not regress (hallucination rate, refusal rate, format-compliance rate, toxicity)
- **Catastrophic error definition** — the failure mode that is disqualifying regardless of aggregate score. Every application has one; name it.

### 1.3 Define the latency SLO precisely

For streaming interfaces, one number is insufficient. Specify all four:

| Metric | Definition | Typically driven by |
|---|---|---|
| **TTFT** (time to first token) | Request → first token emitted | Prefill compute, prompt length, queueing |
| **TPOT / ITL** (time per output token) | Steady-state inter-token latency | Decode speed, memory bandwidth, batch size |
| **End-to-end latency** | Request → last token | TTFT + TPOT × output length |
| **Percentile** | p50 / **p95** / p99 | Always specify. p50 hides queueing pathologies |

> ⚠️ **Latency is not a model property.** It's a property of (model, quantization, hardware, batch size, concurrency, serving stack). The same model can be 10× apart on this axis. Measure it in your configuration; never take a leaderboard's number.

### 1.4 Define throughput and traffic shape

- **Peak** requests/sec and **average** requests/sec
- **Peak-to-average ratio** — this determines whether you provision for peak (expensive) or use a hybrid/burst strategy
- **Input and output token distributions** (mean *and* p95 — long-tail outputs blow up your latency budget)
- **Growth projection** over the depreciation horizon of the hardware

### 1.5 Define context requirements — honestly

Distinguish **advertised** context from **effective** context. A model advertising a very large window may degrade badly well before that limit on retrieval-in-the-middle tasks.

- What is the realistic p95 prompt length in production?
- Does the task need genuine long-context reasoning, or would retrieval + a smaller window serve better and cheaper?
- **KV cache scales linearly with context length × concurrency** — an over-specified context requirement is one of the most expensive mistakes available to you (see Step 3).

### 1.6 Define the compliance posture

- Data classification of inputs and outputs
- Residency and sovereignty requirements
- Auditability: do you need to reproduce a specific output months later? (This implies pinning weights, quantization, seeds, and serving version.)
- Retention and logging constraints
- Human-in-the-loop requirements for high-stakes outputs

### Step 1 deliverable

> **A signed-off requirements document** containing: task class, primary metric + numeric target, guardrail metrics, catastrophic error definition, TTFT/TPOT/e2e SLOs at a named percentile, peak/average throughput, token distributions, context p95, and compliance constraints.
>
> This document is the input to every subsequent step, and it is what you point at when someone asks why you didn't pick the model that was in the news that week.

---

## Step 2 — Apply hard gates: license, modality, language, sovereignty

Hard gates **eliminate** candidates. They never rank them. Apply them before spending engineering time, because a model that fails a hard gate cannot be rescued by a good eval score.

### 2.1 The license gate — read the actual license

**Llama models are released under the Llama Community License, which is *not* an OSI-approved open-source license.** They are **open-weight**, not open-source. This distinction has real consequences and is routinely misunderstood.

Categories of obligation to check in the license text for your specific model version:

| Obligation type | What to look for | Why it can be disqualifying |
|---|---|---|
| **Scale threshold** | A monthly-active-user ceiling above which you must request a separate license from Meta | If you are near or projected to cross it, this is a commercial negotiation, not a download |
| **Attribution** | Required display of a "Built with Llama" style notice | Affects product UI and marketing copy |
| **Derivative naming** | Requirement that derived/fine-tuned model names carry the family name | Affects your model registry and any model you distribute |
| **Acceptable use policy** | Prohibited use categories | Check against your actual and adjacent use cases |
| **Jurisdictional restrictions** | Some releases have carried region-specific restrictions, particularly for multimodal variants | Can silently disqualify an entire deployment region |
| **Distribution terms** | What you must pass along if you redistribute weights or a derivative | Relevant if you ship models to customers or to edge devices |
| **Output usage** | Terms governing use of model outputs, including for training other models | Relevant if you plan distillation |

**Process:**
1. Pull the `LICENSE` and acceptable-use documents **for the exact model version**, from the model repository. Terms differ between releases in the same family.
2. Route to legal **before** the technical evaluation, not after. A three-week eval invalidated by a license clause is a preventable waste.
3. Record the review outcome as an artifact with a date and a version hash.

> 💡 **If a license gate fails:** you are not stuck. Apache-2.0 and MIT-licensed families exist and are competitive — check current options at evaluation time. The framework in this document is family-agnostic; only Steps 2 and 3's specifics change.

### 2.2 The modality gate

- Text-only, vision-language, audio, or multimodal?
- If multimodal: are the modality-specific restrictions in the license acceptable in every region you operate in?
- Does the smallest model meeting your modality need also meet your quality need? (Multimodal variants are typically larger and slower for the same text quality.)

### 2.3 The language gate

- Which languages must be supported, and at what quality?
- Officially supported languages in the model card vs languages that merely appear in the training mix — the gap is large
- **Test non-English performance explicitly in Step 6.** Aggregate English benchmark scores tell you approximately nothing about performance in a lower-resource language.
- Check tokenizer efficiency per language: a language with poor tokenizer coverage inflates token counts, which inflates cost, latency and context consumption simultaneously.

### 2.4 The sovereignty and supply-chain gate

- Can you host in the required region with the required hardware?
- Weight provenance: are you pulling from the official repository, and do you verify checksums?
- Do you need to mirror weights internally to guarantee availability independent of an external host?
- Are the serving stack's licenses compatible with your distribution model?

### Step 2 deliverable

> **A permissible candidate set** — every model that clears all hard gates — plus a written record of what was eliminated and why. The eliminations matter: they are the answer to "why didn't you consider X?"

---

## Step 3 — Size the hardware envelope

This is the step with no analog in the API world, and it is where most open-weight projects go wrong. **Physics and budget eliminate more candidates here than quality ever will.**

### 3.1 Weight memory

```
weight_memory_GB ≈ parameter_count_B × bytes_per_parameter × overhead_factor
```

| Precision | Bytes/param | 8B model | 70B model | Notes |
|---|---:|---:|---:|---|
| FP32 | 4 | ~32 GB | ~280 GB | Never used for inference |
| **FP16 / BF16** | 2 | ~16 GB | ~140 GB | Reference quality baseline |
| **FP8** | 1 | ~8 GB | ~70 GB | Near-lossless on modern hardware |
| **INT8** | 1 | ~8 GB | ~70 GB | Typically minimal quality loss |
| **INT4** | ~0.5 | ~4.5 GB | ~40 GB | Measurable loss; **must be re-evaluated** |

Apply an **overhead factor of ~1.1–1.2** for activations, CUDA context, fragmentation and the runtime itself.

> ⚠️ **For Mixture-of-Experts (MoE) architectures the math splits in two.** You must hold *all* parameters in memory, but only the *active* parameters participate in each forward pass. So an MoE model has the **memory footprint of a large model** and the **compute cost of a small one**. This is excellent for throughput-per-dollar and terrible for anyone who sized their GPUs off the active-parameter count.

### 3.2 KV cache memory — the number teams forget

The KV cache is what actually determines how many concurrent users a GPU can serve, and it is routinely omitted from capacity planning.

```
kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
                     ↑
                     one each for K and V
```

(The `n_kv_heads` term is why Grouped-Query Attention matters so much — it shrinks this by the GQA ratio versus full multi-head attention.)

```
total_kv_GB = kv_bytes_per_token × avg_sequence_length × max_concurrent_requests / 1e9
```

**Worked example — a 70B-class model, FP16 KV cache:**

Assume 80 layers, 8 KV heads, head dimension 128:

```
kv_bytes_per_token = 2 × 80 × 8 × 128 × 2  =  327,680 bytes  ≈  0.31 MB/token

At 8,000-token sequences and 32 concurrent requests:
    0.31 MB × 8,000 × 32  ≈  80 GB of KV cache
```

**That is an entire 80 GB GPU consumed by cache alone, before the 140 GB of weights.**

**The three levers that fix this:**

| Lever | Effect | Cost |
|---|---|---|
| **KV cache quantization** (FP8/INT8) | 2× reduction | Small quality impact; test it |
| **Reduce max context** | Linear reduction | Requires honest Step 1.5 requirements |
| **Reduce max concurrency** | Linear reduction | Lower throughput per GPU → more GPUs |

> 📌 **The practical rule:** if you specified a very large context window in Step 1 "just to be safe," return to Step 1 and get a real number. Context is the most expensive requirement in the document.

### 3.3 Total memory and the GPU-count decision

```
total_VRAM_needed = weight_memory + kv_cache + activation_overhead + fragmentation headroom
```

Then map onto real hardware. Common accelerator capacities: 24 GB (consumer/entry datacenter), 48 GB, 80 GB, and 141 GB class parts.

**Parallelism strategies:**

| Strategy | When | Trade-off |
|---|---|---|
| **Single GPU** | Model + KV fits in one card | Simplest, no interconnect cost, best latency |
| **Tensor parallel (TP)** | Weights exceed one card | Requires fast interconnect; adds per-token communication latency |
| **Pipeline parallel (PP)** | Very large models across nodes | Higher throughput, worse latency; complex |
| **Replicas** | Model fits, need more throughput | Linear scaling, trivial ops — **always prefer this when possible** |

> 💡 **Strong heuristic:** a quantized model that fits on **one** GPU with adequate KV headroom will usually beat a higher-precision model split across **four** GPUs — on cost, on latency, and on operational simplicity. Test the quantized single-GPU configuration before assuming you need a multi-GPU deployment.

### 3.4 Choose the serving stack

The serving stack can move throughput by an order of magnitude. Evaluate on:

| Capability | Why it matters |
|---|---|
| **Continuous batching** | The single largest throughput win. Non-negotiable for production |
| **Paged KV cache** | Eliminates cache fragmentation; enables much higher concurrency |
| **Automatic prefix caching** | The open-weight analog of hosted prompt caching — reuses the KV cache for a shared prompt prefix. **Free**, rather than a paid surcharge. If your prompts share a large fixed prefix (a system prompt, a schema, a rubric), this is a very large win |
| **Quantization support** | Which formats, at what quality, on your hardware |
| **Speculative decoding** | Latency reduction via a small draft model |
| **Structured/constrained output** | Grammar- or schema-constrained decoding. For extraction tasks this often matters more than model size |
| **Observability** | Per-request token counts, queue depth, cache hit rates, batch occupancy |

Mainstream production options include high-throughput GPU inference servers (the vLLM/SGLang/TensorRT-LLM class) and, for local, edge or CPU deployment, the llama.cpp/GGUF ecosystem. Pick based on your hardware and required features — and note that **the stack is part of the configuration you evaluate**, not an implementation detail chosen later.

### 3.5 Compute the cost per token for each feasible configuration

```
cost_per_1M_output_tokens
    = (GPU_count × GPU_hourly_rate) / (measured_throughput_tok_per_sec × 3600) × 1,000,000

monthly_cost = GPU_count × GPU_hourly_rate × 730 × redundancy_factor
```

Note the asymmetry with APIs: your monthly cost is **fixed by capacity**, not variable with usage. Cost per token is therefore a function of your utilization — quote it at your *actual* expected utilization, not at theoretical peak throughput.

### Step 3 deliverable

> **A table of feasible deployment configurations**, each a full tuple:
> `(model size, quantization, GPU type, GPU count, parallelism, serving stack, max context, max concurrency)`
> with estimated monthly cost and estimated throughput for each.
>
> **Configurations that exceed the budget are eliminated now**, before any quality testing. This is the same cost-first filtration used in the API process, applied to infrastructure instead of price lists.

---

## Step 4 — Shortlist candidates (filtration, not selection)

Now, and only now, look at public evidence — to narrow the feasible set from Step 3 to something you can actually test.

### 4.1 Sources, ranked by usefulness

| Source type | Use it for | Treat with caution because |
|---|---|---|
| **Official model cards** | Architecture, context, supported languages, intended use, stated limitations | Vendor-reported; self-selected benchmarks |
| **Independent aggregators** | Cross-model comparison of quality, price and speed on a common basis | Blended metrics hide the input/output ratio — always find the ratio |
| **Open-model leaderboards** | Relative capability across the open ecosystem | Contamination; leaderboard overfitting; fine-tune spam |
| **Human-preference arenas** | General helpfulness and instruction-following | Measures preference, not task correctness; style bias |
| **Domain-specific benchmarks** | The closest available proxy for your task | Often stale; check the date and the model list |
| **Community reports** | Practical deployment gotchas, quantization quality, serving quirks | Anecdotal; unreproducible; but often the earliest signal |

### 4.2 The proxy-capability technique

If no benchmark exists for your exact task — the common case — **choose the closest capability proxy and say so explicitly.**

Examples:
- SQL generation → **coding benchmarks** (SQL generation is a coding task)
- Structured extraction → **instruction-following + JSON-mode benchmarks**
- RAG synthesis → **long-context + faithfulness/groundedness benchmarks**
- Agentic workflows → **tool-calling and multi-step reasoning benchmarks**

Document the proxy choice and its rationale. It is a defensible engineering decision when stated openly, and a hidden assumption when it isn't.

### 4.3 Benchmark hygiene

Public benchmarks are **weak evidence**. Discount them for:

- **Contamination** — benchmark data leaking into training sets, inflating scores
- **Overfitting** — models tuned to leaderboard distributions rather than to capability
- **Staleness** — leaderboards that stopped updating two model generations ago
- **Category confusion** — entries that rank an entire *system/harness* rather than a model. You cannot deploy a harness's score
- **Fine-tune inflation** — the open ecosystem produces enormous numbers of derivative fine-tunes; many top entries are narrowly tuned and generalize poorly
- **Precision mismatch** — leaderboard scores are typically at full precision. **Your INT4 deployment is a different model.**

### 4.4 Build the shortlist

Combine into a filter over the Step 3 feasible-configuration table:

1. **Eliminate** configurations that exceed the budget (already done in Step 3)
2. **Rank** the remainder on a weighted score of public-evidence signals, weighted according to your Step 1 requirements:

```
shortlist_score = w_q × normalized_quality_proxy
                + w_s × normalized_throughput
                + w_c × normalized_cost_efficiency
```

Normalize each column with min-max to [0,1] before weighting so the terms are comparable:

```
normalized = (value − column_min) / (column_max − column_min)
```

**Choose the weights from the application, and justify them.** A batch pipeline weights throughput low and quality high. An interactive assistant weights latency far higher. There is no universal weighting; there is only a documented one.

3. **Take 3–6 configurations forward.** Always include:
   - The **largest** viable model in budget (the quality ceiling)
   - The **smallest** plausible model (the cost floor — surprisingly often sufficient)
   - At least one **different quantization of the same model** (isolates the quantization effect)
   - A **hosted-API baseline** — even if you are committed to self-hosting. Without it you cannot quantify what self-hosting costs you in quality, and you will be asked

### Step 4 deliverable

> **3–6 fully-specified deployment configurations** to evaluate, plus one API baseline, plus a written rationale for the shortlist and the proxy-capability choice.

---

## Step 5 — Build the golden dataset and eval harness

This is where the decision actually gets made. Everything before this was narrowing; everything after is measurement. **The golden dataset is the single highest-leverage artifact in the entire process** — and unlike the model, it is yours permanently and outlives any model choice.

### 5.1 Construction principles

**Representative of the real distribution.** Sample from real production traffic where possible (logs, support tickets, historical queries). Synthetic-only datasets systematically miss the messy inputs that break models: typos, truncation, ambiguity, adversarial phrasing, mixed languages, empty fields.

**Stratified along the axes that matter for your task:**

```
Difficulty:     easy / medium / hard — in proportion to real traffic
Input length:   short / typical / p95-long
Feature coverage: whatever your task's structural dimensions are
                  (e.g. for text-to-SQL: joins, subqueries, window functions,
                   aggregations, date logic, ordering)
Edge cases:     ambiguous, unanswerable, adversarial, out-of-scope
Languages:      per your Step 2.3 language gate
```

> ⚠️ **An all-hard dataset is a common and instructive mistake.** It exaggerates the spread between models and tells you nothing about typical-case behavior. Stratify to match production, then *report per-stratum scores* — that's where the actionable signal is. A model at 95% on easy and 40% on hard is a completely different product decision from one at 75% flat.

**Human-validated ground truth.** For each item, a qualified human writes and verifies the expected output. Where the output is executable (SQL, code, API calls), **execute it and confirm it produces the right answer** — a query that runs is not the same as a query that is correct.

**Held-out and access-controlled.** Never let the eval set enter a fine-tuning corpus or a prompt. Keep a separate development split for prompt iteration and a locked test split touched only for final numbers. Contaminating your own eval set is the fastest way to ship a model that scores well and performs badly.

### 5.2 Sizing the dataset

Size is driven by the **statistical resolution you need** to distinguish candidates, not by convenience.

95% confidence interval half-width for an accuracy near 85%:

| n | CI half-width | Can reliably distinguish |
|---:|---:|---|
| 20 | ±16 pp | Almost nothing |
| 50 | ±10 pp | Only huge gaps |
| 100 | ±7 pp | Large gaps |
| **300** | **±4 pp** | **Practical minimum for a real decision** |
| 500 | ±3 pp | Comfortable |
| 1,000 | ±2 pp | Fine distinctions |

```
CI_half_width ≈ 1.96 × √( p(1−p) / n )
```

> **The n=20 trap:** with 20 items, each item is worth 5 percentage points. A model scoring 90% versus one scoring 85% differs by *one question*, which is noise. Demos at this size are useful for illustrating the mechanics; they cannot support a production decision.

**Cost discipline:** evaluation is itself an expense. `n_items × n_configs × n_repeats` API/GPU calls. Budget for it explicitly, and note that **paired testing** (below) buys you statistical power far more cheaply than raising `n`.

### 5.3 Choose graders, in order of preference

| Grader type | Use when | Strength | Weakness |
|---|---|---|---|
| **Exact / normalized match** | Output is canonical | Objective, free, instant | Brittle to valid variation |
| **Execution-based** | Output is executable (SQL, code) | **Gold standard** — checks semantics, not surface form | Needs a safe execution environment |
| **Programmatic checks** | Schema/format/constraint compliance | Objective, catches structural failures | Doesn't judge substance |
| **Reference-based metrics** | Free text with a reference | Cheap, automatic | Correlates weakly with human judgement |
| **LLM-as-judge** | Open-ended generation | Scales; correlates reasonably when calibrated | Biased, needs calibration, costs money |
| **Human evaluation** | High-stakes, subjective, or calibrating a judge | Ground truth | Slow, expensive, needs inter-rater agreement |

**Prefer objective graders wherever the task admits one.** For any task with an executable or checkable output, use execution-based grading — it is the difference between a defensible number and an argument.

> 💡 **Compare semantics, not strings.** For generated code, SQL, or structured output, two textually different outputs are frequently both correct. Execute both the candidate and reference output and compare *results*, with normalization for irrelevant differences (numeric type, float tolerance, whitespace, key order) and order-sensitivity handled explicitly — sort before comparing *unless* ordering is part of the specification.

### 5.4 Calibrating an LLM judge

If you must use LLM-as-judge, calibrate it or your results are decorative:

1. Human-label a subset (**≥100 items**) as the calibration set
2. Run the judge on it and measure agreement (Cohen's κ, or correlation for scored rubrics)
3. **Require κ ≥ 0.6 (substantial); prefer ≥ 0.8** before trusting the judge at scale
4. Mitigate known biases:
   - **Position bias** — randomize order in pairwise comparisons; run both orders and average
   - **Verbosity bias** — judges favor longer answers; control for length explicitly in the rubric
   - **Self-preference** — a judge favors its own family's outputs. **Use a judge from a different model family than any candidate**
5. Use a **detailed rubric with explicit scoring anchors**, not "rate 1–10"
6. Re-validate the judge whenever you change the judge model or the rubric

### 5.5 Build the harness

Requirements for the harness itself:

- **Deterministic where possible** — pin seeds, temperature (0 for graded correctness tasks), and all decoding parameters
- **Fully versioned** — dataset version, model version + weight hash, quantization, serving stack version, prompt template version. All of it, recorded with every run
- **Isolated failure handling** — distinguish and count separately: correct / incorrect / malformed-output / API-or-runtime error / timeout. A model that emits unparseable output is failing differently than one that emits a wrong answer, and the fix is different
- **Per-item logging** — store the full input, raw output, parsed output, grade, and latency for every item. Aggregate scores tell you *which* model; the per-item logs tell you *why*, and that is what drives the next iteration
- **Cheap to re-run** — you will run it many times. Every hour spent making it one command is repaid within the week

### Step 5 deliverable

> **A versioned golden dataset** (stratified, human-validated, held-out) and a **reproducible eval harness** that emits per-item results and aggregate metrics with confidence intervals.

---

## Step 6 — Run the eval: quality, cost, latency, safety

Four dimensions. A model that wins on quality and fails the latency SLO has not won.

### 6.1 The cardinal rule

> **Evaluate the exact configuration you will deploy.**

Not the FP16 weights if you will ship INT4. Not batch-size-1 latency if you will serve at concurrency 32. Not the default sampling parameters if you will use your own. **Quantization can change instruction-following and structured-output reliability well out of proportion to its effect on aggregate benchmark scores**, and it degrades unevenly across task types — reasoning and long-context tasks typically suffer first.

If you intend to consider multiple quantizations, they are **separate candidates** in the evaluation, not a footnote.

### 6.2 Quality evaluation

**Protocol:**
1. Fix the prompt template across all candidates. *(If you optimize the prompt per model — which is legitimate and often correct — you must budget equal optimization effort for each, and say so. Unequal prompt effort is the most common source of a biased comparison.)*
2. Run every candidate over the full golden set, at temperature 0 for graded tasks
3. **Repeat 3–5 times** even at temperature 0 — batching, hardware non-determinism, and any residual sampling produce run-to-run variance. Report mean and standard deviation
4. Compute per-stratum scores, not just the aggregate

**Statistical treatment:**
- Report **confidence intervals**, never bare point estimates
- Use **paired tests** (McNemar's test for binary outcomes) — all candidates saw identical items, and exploiting that pairing gives far more power than treating the samples as independent
- **Bootstrap** CIs for non-standard metrics
- State plainly when two candidates are **statistically indistinguishable**. That is a real and useful finding: it means you should decide on cost, latency, or operational grounds instead

**Error analysis — the part that pays for itself:**
Read the failures. Cluster them by type. Common clusters:
- Systematic misunderstanding of one task sub-type → may be fixable by prompting
- Format/parsing failures → fixable with constrained decoding, not a bigger model
- Knowledge gaps → fixable with retrieval, not a bigger model
- Reasoning-depth failures → **this is the one that actually requires a bigger model**

> This analysis frequently reveals that the gap between a small and a large model is concentrated in a fixable category — which changes the decision entirely.

### 6.3 Performance evaluation under load

Measured in your serving configuration, with production-shaped traffic:

| Measurement | How | Report |
|---|---|---|
| **TTFT** | Under realistic prompt-length distribution | p50, p95, p99 |
| **TPOT** | Steady-state generation | p50, p95 |
| **End-to-end latency** | Full request lifecycle | p50, p95, p99 |
| **Throughput** | Total tokens/sec at each concurrency level | Curve, not a point |
| **Saturation point** | Concurrency at which p95 latency breaches the SLO | **This is your capacity number** |
| **Cold start** | Weight loading and warm-up time | Matters for autoscaling |
| **Prefix cache hit rate** | If your prompts share a prefix | Directly reduces TTFT and cost |

**Sweep the concurrency curve.** Throughput and latency trade off against each other; the useful output is not a single number but the curve, and the point on it where you breach the SLO. That point, divided into your peak traffic, is your GPU count.

### 6.4 Cost evaluation

For each candidate, from *measured* throughput (never advertised throughput):

```
cost_per_1M_tokens = (GPU_count × GPU_hourly) / (measured_throughput × 3600) × 1e6

monthly_cost = GPU_count_at_peak × GPU_hourly × 730 × redundancy_factor
             + storage + egress + observability + amortized engineering time
```

Compare against the API baseline **including the API's zero operational overhead**. Report the break-even volume, and where current projected volume sits relative to it.

### 6.5 Safety, robustness and regression

Often skipped; often the reason a deployment gets rolled back.

- **Refusal calibration** — over-refusal on legitimate requests is a real product defect, not a safety win. Measure it
- **Prompt injection resistance** — mandatory if the model touches untrusted input or has tool access
- **Jailbreak resistance** — proportional to your risk surface
- **PII handling** — does the model leak or echo sensitive input?
- **Robustness** — degrade the inputs (typos, truncation, mixed language, adversarial phrasing) and measure the drop. A model that is excellent on clean inputs and brittle on real ones is a bad production model
- **Bias and fairness** — where the application affects people, per your governance requirements
- **Output stability** — run identical inputs repeatedly; high variance is a product problem independent of accuracy

### Step 6 deliverable

> **A results matrix** covering all four dimensions for every candidate, with confidence intervals, per-stratum breakdowns, latency percentiles under load, measured cost per token, and safety findings. Plus a written error analysis.

---

## Step 7 — Decide, document, deploy, and re-evaluate

### 7.1 Make the decision explicit

Score candidates against the Step 1 requirements using a weighted matrix (template below). Two rules make this honest:

1. **Apply hard constraints as pass/fail first.** A candidate that misses the latency SLO or the catastrophic-error ceiling is out, regardless of its weighted score. Do not let a high quality score paper over a constraint violation.
2. **Weights are set from the Step 1 requirements document, and are fixed before the scores are filled in.** Setting weights after seeing results is how you accidentally reverse-engineer a justification for a model you already liked.

**When candidates are statistically tied on quality** — which is common — decide on the tiebreakers in this order:

```
1. Operational simplicity   (fewer GPUs, single-node, mature stack)
2. Cost                     (including the ops overhead, not just GPU rate)
3. Latency headroom         (margin against the SLO, not just meeting it)
4. Ecosystem maturity       (tooling, quantization support, community depth)
5. Upgrade path             (does the family have a credible roadmap?)
```

> **Note the parallel to hosted-API selection:** there, teams often accept a small accuracy trade for a provider whose API is more reliable. The self-hosted analog is accepting a small accuracy trade for a configuration that fits on one GPU with a mature serving stack. Operational reliability is a real requirement, not a soft preference.

### 7.2 Write the decision record

An architecture decision record containing:

- The requirements it was decided against (link to the Step 1 document)
- The candidate set and **why each rejected candidate was rejected** — including license eliminations
- The full results matrix with confidence intervals
- The chosen configuration, **fully specified**: model + version + weight hash, quantization, serving stack + version, hardware, parallelism, max context, max concurrency, decoding parameters, prompt template version
- Known limitations and the error clusters you accepted
- The conditions that would trigger re-evaluation
- Sign-offs: engineering, legal (license), security, and product

> This document is the deliverable that makes the decision defensible six months later, when the person who made it has moved teams and someone asks why you aren't using whatever is currently in the news.

### 7.3 Deploy progressively

```
Shadow mode      → run alongside the incumbent, compare outputs, ship nothing
       ↓
Canary (1–5%)    → real traffic, tight monitoring, fast rollback
       ↓
Progressive      → 25% → 50% → 100%, with a gate at each step
       ↓
Full rollout     → incumbent retained as a fallback path
```

Keep a **fallback route** — to the previous model or to a hosted API — for at least one full incident cycle. Define rollback triggers numerically before you start, not during the incident.

### 7.4 Monitor in production

| Signal | Why | Alert on |
|---|---|---|
| Quality on a rotating sampled subset | Detect regression | Drop beyond CI |
| Latency percentiles | SLO compliance | p95 breach |
| GPU utilization + KV cache occupancy | Capacity planning | Sustained >85% |
| Malformed-output rate | Early failure signal | Any increase |
| Refusal rate | Over-refusal regression | Any increase |
| Cost per request | Economics drift | Trend break |
| Input distribution drift | Traffic changed | Statistical drift test |
| Prefix cache hit rate | Cost efficiency | Decline |

### 7.5 Close the loop

**Grow the golden dataset from production failures.** This is the highest-value ongoing activity in the entire framework. Every production failure becomes an eval item; the dataset compounds in value and becomes the durable asset that makes the *next* model selection dramatically cheaper.

**Schedule re-evaluation.** Quarterly, or triggered by:
- A new release in the family (or a competitive family)
- Traffic volume crossing a break-even threshold in either direction
- Sustained quality regression on the monitored subset
- A material change in GPU or API pricing
- A change in license terms

Because the harness and dataset already exist, a re-evaluation is days of work rather than weeks. **That is the compounding return on Step 5.**

---

## Decision matrix template

Fill hard constraints first — any FAIL eliminates the candidate outright.

| | Weight | Cand. A | Cand. B | Cand. C | API baseline |
|---|---:|---|---|---|---|
| **HARD CONSTRAINTS** | | | | | |
| License cleared | gate | PASS/FAIL | | | |
| Fits hardware budget | gate | PASS/FAIL | | | |
| p95 latency ≤ SLO | gate | PASS/FAIL | | | |
| Catastrophic error ≤ ceiling | gate | PASS/FAIL | | | |
| Data residency satisfied | gate | PASS/FAIL | | | |
| **SCORED CRITERIA** | | | | | |
| Quality (primary metric) | 0.40 | | | | |
| Cost efficiency | 0.20 | | | | |
| Latency headroom | 0.15 | | | | |
| Operational simplicity | 0.10 | | | | |
| Robustness / safety | 0.10 | | | | |
| Ecosystem + upgrade path | 0.05 | | | | |
| **WEIGHTED TOTAL** | 1.00 | | | | |

*Normalize each scored row to [0,1] across candidates before weighting. Weights shown are illustrative — set yours from the Step 1 requirements, and set them before you see the scores.*

---

## Worked example

*Illustrative, to show the shape of the reasoning. Numbers are made up.*

**Application:** internal document-extraction service. Structured JSON out of contracts. Regulated data — cannot leave the network.

| Step | Outcome |
|---|---|
| **0** | Hard requirement (data residency) forces self-hosting. Volume ~2M documents/month, steady load, high utilization → economics also favor it. **Proceed.** |
| **1** | Task: extraction (low size-elasticity). Bar: ≥94% field-level F1, ≤0.5% catastrophic (wrong party name). p95 e2e ≤ 4s. 40 req/s peak. p95 input 6k tokens, output ~800. English + one other language. |
| **2** | License reviewed by legal — MAU threshold not close, attribution and naming obligations accepted, jurisdiction clear. Text-only sufficient. Second language officially supported. **Gates cleared.** |
| **3** | KV math at 8k context × 24 concurrency rules out the largest family member on a single-node budget. Feasible set: **8B-class @ FP16 single GPU**, **8B-class @ INT8 single GPU**, **70B-class @ INT4 single GPU**, **70B-class @ FP8 across 2 GPUs**. Last option is 2.4× the cost of the others. |
| **4** | Extraction is low-elasticity → public evidence suggests the 8B class may suffice. Shortlist all four configs + a hosted-API baseline. Proxy capability: instruction-following + structured-output benchmarks. |
| **5** | 420-item golden set built from **real** contracts, stratified by document type, length, and language, with a deliberate edge-case stratum (scanned/OCR-noisy, missing fields, unusual clauses). Grading: **programmatic** — schema validation + field-level exact match with normalization. No LLM judge needed. |
| **6** | 8B FP16: 91.2% F1 (±2.7). 8B INT8: 90.8% (±2.7) — **statistically tied**, half the memory. 70B INT4: 95.1% (±2.1) — **clears the bar**. 70B FP8 ×2: 95.6% (±2.0) — tied with INT4, 2.4× cost. Latency: all within SLO; 8B has the most headroom. Error analysis: the 8B failures concentrate in one document type with unusual clause structure — a genuine reasoning-depth gap, not a format problem. **Constrained decoding eliminated all malformed-output failures across every candidate.** |
| **7** | **Chosen: 70B-class @ INT4, single GPU.** Only configurations clearing the 94% bar were the two 70B variants; they are statistically tied, so the tiebreaker is operational simplicity and cost — single-GPU INT4 wins on both. 8B retained as a documented fallback for a possible future cost-reduction effort, contingent on closing the identified document-type gap via fine-tuning. Re-evaluate on next family release. |

> **The instructive part:** Step 6's error analysis is what determined that the 8B gap was *not* fixable by prompting. Without reading the failures, the team would have known only that 8B scored lower — not whether that was worth spending a GPU tier to fix.

---

## Appendix A — Llama family map

> ⚠️ **This section dates quickly.** Verify the current line-up, sizes, context windows and license terms on the official model cards before relying on any of it. Treat this as a shape-of-the-family orientation, not a specification.

**Architectural generations, broadly:**

| Generation | Character | Typical use |
|---|---|---|
| **Small dense** (~1B–3B) | Edge, on-device, CPU-viable, very cheap | Classification, routing, simple extraction |
| **Mid dense** (~7B–8B) | The workhorse tier; single-GPU friendly | Extraction, summarization, RAG synthesis, fine-tuning base |
| **Large dense** (~70B) | Strong general capability; single-GPU at INT4 | Code, reasoning, complex RAG, agentic |
| **Very large dense** (~400B+) | Frontier-adjacent; multi-node | Research, distillation teacher, highest-stakes reasoning |
| **Vision-language variants** | Multimodal; larger for equivalent text quality | Document understanding, image Q&A |
| **MoE variants** | Large total params, small active params | High throughput-per-dollar; **memory-heavy, compute-light** |

**Selection heuristics within the family:**

- **Start smaller than you think.** Test the smallest plausible size first. Most production tasks are not frontier-reasoning tasks, and teams over-provision by default
- **A newer smaller model often beats an older larger one.** Compare across generations, not just within one
- **Instruction-tuned vs base:** use instruction-tuned unless you are doing your own alignment training
- **MoE changes your sizing math.** Budget memory for total parameters, budget compute for active parameters
- **Fine-tuning a smaller model** is frequently more cost-effective than serving a larger one zero-shot — especially for narrow, high-volume, well-specified tasks. Consider it before escalating a size tier
- **Long-context variants:** advertised context is a ceiling, not a promise. Test effective context on your task

---

## Appendix B — VRAM and throughput reference

### Quick VRAM estimates (weights only, before KV cache)

| Params | FP16 | FP8/INT8 | INT4 | Smallest single GPU (INT4, with KV headroom) |
|---:|---:|---:|---:|---|
| 1B | 2 GB | 1 GB | 0.6 GB | Consumer / CPU |
| 3B | 6 GB | 3 GB | 1.7 GB | 24 GB class, comfortably |
| 8B | 16 GB | 8 GB | 4.5 GB | 24 GB class |
| 70B | 140 GB | 70 GB | 40 GB | 80 GB class |
| 405B | 810 GB | 405 GB | 230 GB | Multi-node regardless |

**Always add:** ~10–20% runtime overhead, **plus the KV cache**, which at production concurrency is frequently *larger than the weights*.

### KV cache per token (FP16), by architecture shape

```
kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × 2
```

| Shape | Layers | KV heads | Head dim | Per token |
|---|---:|---:|---:|---:|
| 8B-class | 32 | 8 | 128 | ~0.13 MB |
| 70B-class | 80 | 8 | 128 | ~0.31 MB |
| 405B-class | 126 | 8 | 128 | ~0.49 MB |

**Multiply by sequence length × concurrency.** Then halve it with FP8 KV quantization if you need to.

### Sanity checks

- If KV cache > weights, your context or concurrency setting is the dominant cost driver — revisit Step 1.5
- If a configuration needs >1 GPU, price the quantized single-GPU alternative before committing
- If throughput is far below expectation, check: continuous batching enabled? batch actually filling? prefix caching on? Serving-stack misconfiguration accounts for more disappointing benchmarks than model choice does

---

## Appendix C — Common failure modes

| # | Failure | Consequence | Prevention |
|---:|---|---|---|
| 1 | Evaluating FP16, deploying INT4 | Production quality below eval | Step 6.1 — evaluate the deployed configuration |
| 2 | Ignoring KV cache in capacity planning | OOM at production concurrency | Step 3.2 |
| 3 | Golden set too small (n≈20) | Decision made on noise | Step 5.2 — n ≥ 300 |
| 4 | All-hard or all-easy golden set | Score doesn't predict production | Step 5.1 — stratify to match traffic |
| 5 | License reviewed after the eval | Weeks of work invalidated | Step 2.1 — legal review first |
| 6 | Comparing GPU rate to API bill | TCO understated, often by 2–3× | Step 0.4 — include ops and engineering |
| 7 | Uncalibrated LLM judge | Confident, meaningless scores | Step 5.4 — κ ≥ 0.6, cross-family judge |
| 8 | Batch-size-1 latency benchmarks | SLO breached under real load | Step 6.3 — sweep the concurrency curve |
| 9 | Unequal prompt-optimization effort | Systematically biased comparison | Step 6.2 — equal effort, disclosed |
| 10 | Leaderboard rank taken as task fit | Hyped model underperforms on your task | Step 4.3 — leaderboards filter only |
| 11 | Eval set leaks into fine-tuning data | Inflated scores, bad production model | Step 5.1 — held out and access-controlled |
| 12 | No production monitoring | Silent drift, discovered by users | Step 7.4 |
| 13 | Over-specified context window | Massive unnecessary GPU spend | Step 1.5 — get the real p95 |
| 14 | No API baseline in the comparison | Cannot quantify what self-hosting cost | Step 4.4 — always include it |
| 15 | Aggregate score only, no error analysis | Missed that the gap was prompt-fixable | Step 6.2 — read the failures |

---

## Appendix D — Glossary

| Term | Meaning |
|---|---|
| **Open-weight vs open-source** | Weights are downloadable, but the license may impose restrictions an OSI-approved license would not. Llama is open-weight |
| **Quantization** | Reducing numerical precision of weights/activations to shrink memory and increase speed, at some quality cost |
| **KV cache** | Cached attention key/value tensors for prior tokens; grows linearly with sequence length × concurrency |
| **GQA** (Grouped-Query Attention) | Multiple query heads share KV heads, shrinking the KV cache substantially |
| **MoE** (Mixture of Experts) | Only a subset of parameters activates per token: large memory footprint, small compute footprint |
| **TTFT** | Time to first token — dominated by prefill and queueing |
| **TPOT / ITL** | Time per output token — steady-state generation speed |
| **Continuous batching** | Adding and retiring requests from a running batch dynamically; the largest single throughput win in serving |
| **Prefix caching** | Reusing the KV cache for a shared prompt prefix across requests — the self-hosted analog of hosted prompt caching, and free |
| **Speculative decoding** | A small draft model proposes tokens that the large model verifies in parallel, reducing latency |
| **Constrained decoding** | Forcing output to conform to a grammar or schema; often more effective than scaling up for structured tasks |
| **Tensor / pipeline parallelism** | Splitting a model across GPUs by layer-internal shards / by layer groups |
| **Golden dataset** | Human-validated (input → expected output) pairs used as ground truth |
| **LLM-as-judge** | Using a model to grade another model's open-ended output; requires calibration against human labels |
| **Execution-based grading** | Running generated code/SQL and comparing *results* rather than text |
| **McNemar's test** | Paired significance test for binary outcomes on the same items — the right test for comparing models on a shared eval set |
| **Effective context** | The context length at which the model actually performs well, typically well below the advertised maximum |
| **Break-even volume** | The usage level at which fixed self-hosting cost equals variable API cost |

---

## Summary card

```
0. PRE-FLIGHT       Should you self-host? TCO + team + break-even. Most shouldn't.
1. REQUIREMENTS     Task class, numeric quality bar, latency SLO, throughput,
                    context p95, compliance. Signed off before looking at models.
2. HARD GATES       License (read it, legal first), modality, language,
                    sovereignty. Eliminates — never ranks.
3. HARDWARE         Weights + KV cache + overhead → feasible (size, quant, GPU)
                    tuples in budget. Physics prunes more than quality will.
4. SHORTLIST        Leaderboards filter. 3–6 configs + an API baseline.
                    Include largest-viable, smallest-plausible, and a quant variant.
5. GOLDEN SET       Stratified, human-validated, held out. n ≥ 300.
                    Objective graders where possible. Calibrate any judge.
6. RUN THE EVAL     At deployment quantization, under production load.
                    Quality + latency + cost + safety. CIs, paired tests,
                    per-stratum scores, and read the failures.
7. DECIDE + OPERATE Weighted matrix with hard constraints as gates. Write the ADR.
                    Shadow → canary → progressive. Monitor. Grow the golden set
                    from production failures. Re-evaluate on a schedule.
```

> **The one line worth keeping:** *the golden dataset outlives every model you will ever choose.* Models are replaced every few months; a well-built, production-derived eval set makes each replacement a days-long exercise instead of a quarter-long one. It is the compounding asset in this process — build it properly the first time.
