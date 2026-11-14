# Adaption feasibility (P2): research only, no credits spent yet

Researched 2026-11-14 from docs.adaptionlabs.ai. Treat anything not listed as verified as unknown.

## What Adaption can and cannot do for us

| Question | Answer (from docs) |
|---|---|
| Train our YOLO detector? | **No.** AutoScientist fine-tunes LLMs/VLMs (SFT, or SFT→DPO). Our detector stays on the GX10 (Kolyan). |
| Minimum data | 1,000 *effective* rows for small models (base rows + Adaption augmentation); 12,000 for DPO. Multimodal datasets cannot be augmented, so all 1,000 must be real rows. |
| Small text models | `Qwen/Qwen3.5-0.8B` (LoRA), `meta-llama/Llama-3.2-3B-Instruct` and `google/gemma-3-4b-it` (LoRA or full) |
| Small vision models | `google/gemma-3-4b-it-VLM`, `Qwen/Qwen3.5-4B` (LoRA) |
| Bring our own data | Yes: `datasets.upload_file(..., processing_mode="raw", column_mapping={prompt, completion})`, then `autoscientist.create(dataset_id=...)` |
| Generate data | `datasets.invent(dataset_prompt=..., rows=N, estimate=True)` gives a free credit estimate. It cannot be told to follow our JSON schema. |
| Get the model out | `autoscientist.download(run_id)` → tar with a LoRA adapter (`adapter_model.safetensors`) of the **best** iteration. The docs say nothing about hosting the fine-tuned model for inference. |
| Cost per run / per row | **Not documented.** Use `estimate=True` and `get_preprocessing_credits` before committing our 1,250 credits. |

## Options considered

1. **Task Engine distillation (recommended).** Fine-tune a small text model to do what Claude does in
   `compile_task`: request → `TaskSpecDraft` JSON.
   - Measurable function: a Task Engine that runs **offline on the GX10**, with no API key, subscription or
     network. The brief requires comparing it against the unchanged base model and against Claude.
   - Data is cheap to make well: our validator (`schema.consistency_errors`) filters bad completions.
2. **VLM label verifier**: fine-tune `gemma-3-4b-it-VLM` to answer "is this person wearing a vest?" on crops,
   to cut down manual review. Rejected for now: it needs about 1,000 human-verified crops (no augmentation for
   images) before it can be trained, and we don't have them.
3. **Adaption as YOLO trainer**: not supported (see table).

## Plan for option 1 (time-boxed, must not block P0/P1)

1. **Prompts (Adaption credits).** `datasets.invent(dataset_prompt=<"plain-English requests a business user
   types to set up a camera detector: PPE, occupancy, objects, plus vague and unsupported asks">, rows=1200,
   estimate=True)`. Check the credit estimate first, then generate. Use the prompt column only.
2. **Teacher labels.** Run every prompt through our `compile_task` (Claude) and keep only specs that pass
   `consistency_errors`. Cost options:
   - metered API with the Batches API (50% off): about 1,200 × ~4k tokens; estimate before running
   - or Kolyan's Claude-subscription CLI backend: no metered cost, but ~12 s per call (~4 h sequential)
3. **Split.** Hold out 150 prompts as the independent eval set; they never go to Adaption.
4. **Train.** Upload the rest with `processing_mode="raw"` (prompt → `TaskSpecDraft` JSON) and run
   `autoscientist.create(..., training_method="instruction", model="Qwen/Qwen3.5-0.8B")`, or let it pick.
   Download the LoRA adapter.
5. **Evaluate on the 150 held-out prompts**: base model vs adapted vs Claude:
   - valid JSON that passes `consistency_errors`
   - `status` agreement with Claude (ok / needs_clarification / unsupported)
   - same class set and target_event; attribute rules present where Claude has them; no negated detector prompts
   - latency on the GX10
6. **Integrate only if it wins**: an `llm` backend for `create_task_spec(prompt, llm=...)` (the hook already
   accepts one), with Claude as the fallback when the local spec fails validation.

## Decisions needed before spending anything

- Adaption API key and the `estimate=True` credit quote for 1,200 invented rows plus one AutoScientist run.
- Which teacher path to use for step 2 (metered API vs subscription CLI).
- Whether P0/P1 are far enough along to spend about 2 hours on this.
