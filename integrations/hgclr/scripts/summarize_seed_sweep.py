import csv
import json
from pathlib import Path


DATASETS = ("WebOfScience", "nyt", "rcv1")
SEEDS = (1, 2, 3, 4, 5)
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_CSV = CHECKPOINTS_DIR / "seed_sweep_summary.csv"


def load_metrics(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def get(row, *keys):
    cur = row
    for key in keys:
        if cur is None:
            return None
        cur = cur.get(key)
    return cur


def main():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run_name = f"{dataset}-seed{seed}"
            metrics = load_metrics(CHECKPOINTS_DIR / run_name / "cost_metrics.json")
            if metrics is None:
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "run_name": run_name,
                    "status": "missing",
                })
                continue

            training = metrics.get("training", {})
            inference = metrics.get("inference", {})
            test_metrics = metrics.get("test_metrics", {})
            per_epoch = training.get("per_epoch", [])
            best_macro = max((epoch.get("macro_f1", 0.0) for epoch in per_epoch), default=None)
            best_micro = max((epoch.get("micro_f1", 0.0) for epoch in per_epoch), default=None)

            rows.append({
                "dataset": dataset,
                "seed": seed,
                "run_name": run_name,
                "status": "ok",
                "model_params_total": metrics.get("model_params_total"),
                "model_params_trainable": metrics.get("model_params_trainable"),
                "train_total_time_sec": get(metrics, "training", "total_time_sec"),
                "train_epochs_completed": get(metrics, "training", "epochs_completed"),
                "train_peak_gpu_memory_mb": get(metrics, "training", "peak_gpu_memory_mb"),
                "train_avg_gpu_memory_mb": get(metrics, "training", "avg_gpu_memory_mb"),
                "best_val_macro_f1": best_macro,
                "best_val_micro_f1": best_micro,
                "test_macro_f1": test_metrics.get("macro_f1"),
                "test_micro_f1": test_metrics.get("micro_f1"),
                "test_p_at_1": test_metrics.get("p@1"),
                "test_p_at_3": test_metrics.get("p@3"),
                "test_p_at_5": test_metrics.get("p@5"),
                "test_r_precision": test_metrics.get("r_precision"),
                "inference_checkpoint": inference.get("checkpoint"),
                "inference_total_time_sec": inference.get("total_time_sec"),
                "inference_num_samples": inference.get("num_samples"),
                "inference_throughput_samples_per_sec": inference.get("throughput_samples_per_sec"),
                "inference_peak_gpu_memory_mb": inference.get("peak_gpu_memory_mb"),
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed",
        "run_name",
        "status",
        "model_params_total",
        "model_params_trainable",
        "train_total_time_sec",
        "train_epochs_completed",
        "train_peak_gpu_memory_mb",
        "train_avg_gpu_memory_mb",
        "best_val_macro_f1",
        "best_val_micro_f1",
        "test_macro_f1",
        "test_micro_f1",
        "test_p_at_1",
        "test_p_at_3",
        "test_p_at_5",
        "test_r_precision",
        "inference_checkpoint",
        "inference_total_time_sec",
        "inference_num_samples",
        "inference_throughput_samples_per_sec",
        "inference_peak_gpu_memory_mb",
    ]
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
