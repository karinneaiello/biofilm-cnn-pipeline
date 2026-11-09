"""Per-threshold-method test metrics from W&B"""
import pandas as pd
import wandb
from scipy.stats import mannwhitneyu

ENTITY = "kmaiello-worcester-polytechnic-institute"
BASELINE_PROJECT = "biofilm-cnn-pipeline-sweep-spinning-disk-trans-v4"
THRESHOLD_PROJECT = "biofilm-cnn-pipeline-sweep-spinning-disk-threshold-v1"

api = wandb.Api()
rows = []
for project in (BASELINE_PROJECT, THRESHOLD_PROJECT):
    for r in api.runs(f"{ENTITY}/{project}"):
        if r.state != "finished":
            continue
        if r.config.get("transform_name") != "mexican_hat":
            continue  # baseline proj also has none, fft_dct, gabor runs
        rows.append({
            # old runs have no threshold_method key, "none" baseline
            "threshold_method": r.config.get("threshold_method", "none"),
            "test_mae": r.summary.get("test/mae"),
            "test_r2": r.summary.get("test/r2"),
            "val_mean_loss": r.summary.get("val/mean_loss"),
        })

df = pd.DataFrame(rows).dropna()
summary = df.groupby("threshold_method").agg(["mean", "std", "median", "count"]).round(4)
print(summary.sort_values(("test_mae", "mean")))

# Rough significance check v baseline, 10 repeats each
base = df[df.threshold_method == "none"]
for m in sorted(set(df.threshold_method) - {"none"}):
    other = df[df.threshold_method == m]
    for col in ("test_mae", "test_r2"):
        p = mannwhitneyu(other[col], base[col]).pvalue
        print(f"{m:>18} vs none | {col}: p = {p:.3f}")

df.to_csv("outputs/threshold_runs.csv", index=False)
summary.to_csv("outputs/threshold_summary.csv")
