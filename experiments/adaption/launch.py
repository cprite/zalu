"""Step 3: upload train.jsonl to Adaption (raw SFT) and start AutoScientist. Needs ADAPTION_API_KEY."""
import sys, time
from adaption import Adaption
from common import BASE_MODEL, D, load_state, save_state

c = Adaption()
st = load_state()
if not st.get("dataset_id"):
    ds = c.datasets.upload_file(str(D / "train.jsonl"), processing_mode="raw",
                                column_mapping={"prompt": "prompt", "completion": "completion"})
    st = save_state(dataset_id=ds.dataset_id, uploaded_at=time.time())
    print("uploaded", ds.dataset_id)
while True:
    s = c.datasets.get_status(st["dataset_id"])
    if s.status == "failed":
        sys.exit(f"ingestion failed: {s.error_data}")
    if s.row_count is not None:
        print("rows", s.row_count); save_state(row_count=s.row_count); break
    time.sleep(5)
if "--dry" in sys.argv:
    print(c.autoscientist.recommend_hyperparams(dataset_id=st["dataset_id"], model=BASE_MODEL)); sys.exit()
if not st.get("run_id"):
    run = c.autoscientist.create(dataset_id=st["dataset_id"], model=BASE_MODEL, training_method="instruction",
                                 data_format="instruction", column_mapping={"prompt": "prompt", "completion": "completion"},
                                 idempotency_key="zalu-taskspec-v1")
    save_state(run_id=run.id, model=run.model, launched_at=time.time(), max_iterations=run.max_iterations,
               target_win_rate=run.target_win_rate)
    print("run", run.id, run.status, run.model, "max_iter", run.max_iterations, "target", run.target_win_rate)
