"""Step 4: status / best win rate; with --download, fetch the best LoRA checkpoint."""
import sys, tarfile
from adaption import Adaption
from common import D, load_state, save_state

c = Adaption()
st = load_state()
run = c.autoscientist.get(st["run_id"])
info = {"status": run.status, "model": run.model, "iterations": f"{run.iterations_completed}/{run.max_iterations}",
        "best_win_rate": run.best_win_rate, "target_win_rate": run.target_win_rate,
        "download_available": getattr(run, "download_available", None),
        "best_hyperparams": run.best_hyperparams.model_dump() if run.best_hyperparams else None,
        "error": getattr(run, "error", None)}
save_state(last_poll=info)
print(info)
if "--download" in sys.argv and run.status == "succeeded" and info["download_available"]:
    out = D / "best-checkpoint.tgz"
    with c.autoscientist.with_streaming_response.download(run.id) as r:
        r.stream_to_file(str(out))
    (D / "lora").mkdir(exist_ok=True)
    with tarfile.open(out) as t:
        t.extractall(D / "lora")
    print("downloaded", out, sorted(p.name for p in (D / "lora").rglob("*"))[:10])
