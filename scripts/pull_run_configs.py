import csv
import wandb

ENTITY = "kmaiello-worcester-polytechnic-institute"
PROJECTS = [
    "biofilm-cnn-pipeline-sweep-spinning-disk-trans-v4",       # old: none, fft_dct, mexican_hat, gabor
    "biofilm-cnn-pipeline-sweep-spinning-disk-threshold-v1",   # new: threshold methods
]

api = wandb.Api()
rows = []
for project in PROJECTS:
    for run in api.runs(f"{ENTITY}/{project}"):
        rows.append({
            "run_id": run.id,
            "project": project,
            "state": run.state,
            "transform_name": run.config.get("transform_name"),
            "threshold_method": run.config.get("threshold_method", "none"),
        })

with open("outputs/run_configs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run_id", "project", "state", "transform_name", "threshold_method"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} run configs to outputs/run_configs.csv")
