""" Dataset-wide test MAE/R2 per method, aggregated across repeats. Uses configsd made by pull_run_configs.py, so threshold_method attribution is exact"""
import csv
import pandas as pd
from scipy.stats import mannwhitneyu
import wandb

ENTITY = "kmaiello-worcester-polytechnic-institute"
RUN_CONFIG_FILE = "outputs/run_configs.csv"

# load run_id with project, state, transform_name, threshold_method
configs = {}
with open(RUN_CONFIG_FILE) as f:
    for row in csv.DictReader(f):
        configs[row["run_id"]] = row

# only keep finished mexican hat transform runs
target_run_ids = {
    run_id: cfg for run_id, cfg in configs.items()
    if cfg["state"] == "finished" and cfg["transform_name"] == "mexican_hat"
}
print(f"{len(target_run_ids)} finished mexican_hat runs found across both projects.")

# get test/mae, test/r2, val/mean_loss from each run's W&B summary & group by project, run_id
api = wandb.Api()
rows = []
projects = {cfg["project"] for cfg in target_run_ids.values()}
for project in projects:
    for run in api.runs(f"{ENTITY}/{project}"):
        cfg = target_run_ids.get(run.id)
        if cfg is None:
            continue  # not a target run
        rows.append({
            "run_id": run.id,
            "threshold_method": cfg["threshold_method"] or "none",
            "test_mae": run.summary.get("test/mae"),
            "test_r2": run.summary.get("test/r2"),
            "val_mean_loss": run.summary.get("val/mean_loss"),
        })

df = pd.DataFrame(rows).dropna(subset=["test_mae", "test_r2"])
print(f"{len(df)} runs with valid test/mae and test/r2 logged.\n")

# aggregate with mean, std, count per method
summary = (
    df.groupby("threshold_method")[["test_mae", "test_r2", "val_mean_loss"]]
    .agg(["mean", "std", "count"])
    .round(4)
)
summary = summary.sort_values(("test_mae", "mean"))
print(summary)

# flag any method with fewer than 10 finished runs
counts = df.groupby("threshold_method").size()
incomplete = counts[counts < 10]
if len(incomplete):
    print("\nWARNING -- these methods have fewer than 10 finished runs:")
    print(incomplete)

# rough significance v the "none" baseline with Mann-Whitney U
if "none" in df["threshold_method"].unique():
    base_mae = df[df.threshold_method == "none"]["test_mae"]
    base_r2 = df[df.threshold_method == "none"]["test_r2"]
    print("\nMann-Whitney U vs. 'none' baseline (descriptive, not corrected for multiple comparisons):")
    for method in sorted(set(df.threshold_method) - {"none"}):
        sub = df[df.threshold_method == method]
        p_mae = mannwhitneyu(sub["test_mae"], base_mae).pvalue
        p_r2 = mannwhitneyu(sub["test_r2"], base_r2).pvalue
        print(f"  {method:<12} test_mae p={p_mae:.4f}   test_r2 p={p_r2:.4f}")
else:
    print("\nNo 'none' baseline runs found in the filtered set -- check run_configs.csv.")

# save it
df.to_csv("outputs/threshold_aggregate_runs.csv", index=False)
summary.to_csv("outputs/threshold_aggregate_summary.csv")
print("\nSaved outputs/threshold_aggregate_runs.csv and outputs/threshold_aggregate_summary.csv")
