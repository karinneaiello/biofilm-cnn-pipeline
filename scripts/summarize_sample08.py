import csv
import re
from collections import defaultdict

PRED_FILE = "outputs/sample08_matches.txt"
CONFIG_FILE = "outputs/run_configs.csv"
ACTUAL = 10.624696  # from release_biomass_mapping.csv file

configs = {}
with open(CONFIG_FILE) as f:
    for row in csv.DictReader(f):
        configs[row["run_id"]] = row

line_pattern = re.compile(r"spinning_disk_([A-Za-z0-9]+)_predictions\.csv:.*,([-\d.eE]+),([-\d.eE]+)\s*$")

groups = defaultdict(list)
unmatched = []

with open(PRED_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = line_pattern.search(line)
        if not m:
            unmatched.append(("no regex match", line))
            continue
        run_id, predicted, actual = m.group(1), float(m.group(2)), float(m.group(3))
        cfg = configs.get(run_id)
        if cfg is None:
            unmatched.append(("run_id not found in W&B", line))
            continue
        if cfg["state"] != "finished":
            unmatched.append((f"run not finished ({cfg['state']})", line))
            continue
        if cfg["transform_name"] != "mexican_hat":
            continue  # drop any fft_dct, gabor, none transforms leftover from the old sweep
        method = cfg["threshold_method"] or "none"
        groups[method].append(predicted)

print(f"{'method':<15} {'n':>3} {'mean_pred':>10} {'mean_abs_err':>13} {'std_pred':>9}")
print("-" * 55)
for method in sorted(groups):
    preds = groups[method]
    n = len(preds)
    mean_pred = sum(preds) / n
    mean_err = sum(abs(p - ACTUAL) for p in preds) / n
    std = (sum((p - mean_pred) ** 2 for p in preds) / n) ** 0.5
    print(f"{method:<15} {n:>3} {mean_pred:>10.3f} {mean_err:>13.3f} {std:>9.3f}")

if unmatched:
    print(f"\n{len(unmatched)} lines excluded:")
    for reason, line in unmatched:
        print(f"  [{reason}] {line}")
