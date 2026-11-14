# Adaption experiment: fine-tuned TaskSpec compiler (owner: Kolyan)

Separate module. Does not touch the vision pipeline (GX10/YOLO) or Kirill's Task Engine code.

**Question:** can a small open model fine-tuned with Adaption AutoScientist turn a camera-task prompt into a valid
ZALU `TaskSpec` (Kirill's schema) better than the same base model, measured on prompts it never saw?

Pipeline (each step is a script, outputs under `data/`):
1. `gen_prompts.py`: diverse natural-language camera-task prompts (Claude CLI on the owner's subscription), deduped,
   split by DOMAIN into train / held-out test (test domains never appear in training).
2. `gen_gold.py`: gold TaskSpec for every prompt = Kirill's `compile_task` + `claude_cli_backend`; only specs that pass
   Kirill's `consistency_errors` validator are kept. Output: prompt/completion JSONL (`completion` = TaskSpecDraft JSON).
3. `launch.py`: upload the train JSONL to Adaption (raw SFT), start AutoScientist on a small base model, record ids.
4. `poll.py`: status, best win rate, credits; download the LoRA checkpoint when done.
5. `serve_gx10.sh`: serve base + LoRA on the GX10 with vLLM (OpenAI-compatible).
6. `evaluate.py`: base vs fine-tuned on the held-out test prompts: JSON parse rate, schema-valid rate (Kirill's
   validator), class/target_event agreement with gold, latency. Results in `results.md`.
7. `client.py`: `adaption_backend()` = drop-in `LLMBackend` for `compile_task(prompt, llm=...)` (optional for Kirill).

No claim of improvement without the numbers from step 6.
