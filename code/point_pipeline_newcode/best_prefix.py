import csv
import os

import part_00_project_config as config


def get_best_prefix(env_name, default_prefix):
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value

    table_path = os.path.join(config.OUTPUT_DIR, "part_17_method_comparison_table.csv")
    if not os.path.exists(table_path):
        return default_prefix

    best_prefix = default_prefix
    best_rmse = None
    with open(table_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return default_prefix
        prefix_idx = -1
        rmse_idx = -1
        for i, name in enumerate(header):
            if name == "prefix":
                prefix_idx = i
            if name == "corrected_ate_rmse_m":
                rmse_idx = i
        if prefix_idx < 0 or rmse_idx < 0:
            return default_prefix
        for row in reader:
            if len(row) <= prefix_idx or len(row) <= rmse_idx:
                continue
            value = row[rmse_idx].strip()
            prefix = row[prefix_idx].strip()
            if not value or not prefix:
                continue
            text = value.replace(".", "", 1)
            if not text.isdigit():
                continue
            rmse = float(value)
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_prefix = prefix
    return best_prefix
