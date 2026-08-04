import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_DATASETS = ("WebOfScience", "nyt", "rcv1")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
CHECKPOINTS_DIR = Path("checkpoints")
SUMMARY_STATS = ("mean", "std_sample", "std_population", "min", "max")


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def population_std(values):
    return statistics.pstdev(values) if values else 0.0


def collect_rows(dataset, seeds):
    rows = []
    for seed in seeds:
        metrics_path = CHECKPOINTS_DIR / f"{dataset}-seed{seed}" / "cost_metrics.json"
        with metrics_path.open() as f:
            metrics = json.load(f)

        test_metrics = metrics["test_metrics"]
        training = metrics["training"]
        inference = metrics["inference"]
        per_epoch = training.get("per_epoch", [])

        rows.append({
            "dataset": dataset,
            "seed": seed,
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "p_at_1": test_metrics["p@1"],
            "p_at_3": test_metrics["p@3"],
            "p_at_5": test_metrics["p@5"],
            "r_precision": test_metrics["r_precision"],
            "train_total_time_sec": training["total_time_sec"],
            "epochs_completed": training["epochs_completed"],
            "train_peak_gpu_memory_mb": training["peak_gpu_memory_mb"],
            "train_avg_gpu_memory_mb": training["avg_gpu_memory_mb"],
            "inference_total_time_sec": inference["total_time_sec"],
            "inference_throughput_samples_per_sec": inference["throughput_samples_per_sec"],
            "inference_peak_gpu_memory_mb": inference["peak_gpu_memory_mb"],
            "best_val_macro_f1": max(epoch.get("macro_f1", 0.0) for epoch in per_epoch),
            "best_val_micro_f1": max(epoch.get("micro_f1", 0.0) for epoch in per_epoch),
        })

    return rows


def build_aggregate(rows):
    metric_cols = [col for col in rows[0] if col not in {"dataset", "seed"}]
    aggregate = {}
    for col in metric_cols:
        values = [row[col] for row in rows]
        aggregate[col] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "std_sample": sample_std(values),
            "std_population": population_std(values),
            "min": min(values),
            "max": max(values),
        }
    return metric_cols, aggregate


def write_csv(path, dataset, rows, metric_cols, aggregate):
    with path.open("w", newline="") as f:
        fieldnames = ["row_type", *rows[0].keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_type": "seed", **row})
        for stat in SUMMARY_STATS:
            writer.writerow({
                "row_type": stat,
                "dataset": dataset,
                **{col: aggregate[col][stat] for col in metric_cols},
            })


def write_json(path, dataset, seeds, rows, aggregate):
    with path.open("w") as f:
        json.dump({
            "dataset": dataset,
            "seeds": list(seeds),
            "per_seed": rows,
            "aggregate": aggregate,
        }, f, indent=2)


def aggregate_dataset(dataset, seeds):
    rows = collect_rows(dataset, seeds)
    metric_cols, aggregate = build_aggregate(rows)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_csv_path = CHECKPOINTS_DIR / f"{dataset}_seed_aggregate.csv"
    checkpoint_json_path = CHECKPOINTS_DIR / f"{dataset}_seed_aggregate.json"
    root_csv_path = Path(f"{dataset}_seed_aggregate.csv")
    root_json_path = Path(f"{dataset}_seed_aggregate.json")

    for csv_path in (checkpoint_csv_path, root_csv_path):
        write_csv(csv_path, dataset, rows, metric_cols, aggregate)
    for json_path in (checkpoint_json_path, root_json_path):
        write_json(json_path, dataset, seeds, rows, aggregate)

    print(f"Wrote {checkpoint_csv_path}")
    print(f"Wrote {checkpoint_json_path}")
    print(f"Wrote {root_csv_path}")
    print(f"Wrote {root_json_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Datasets to aggregate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to include.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset in args.datasets:
        aggregate_dataset(dataset, args.seeds)


if __name__ == "__main__":
    main()
