import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

# ===== 設定 =====
DATASET_ROOT = Path("data/dataset")
OUTPUT_DIR = Path("outputs/timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===== 評価関数（仮）=====
def evaluate(text: str, task_type: str) -> Tuple[float, float, float]:
    """
    evaluate(text, taskA/B) -> precision, recall, f1
    ※ 実際の評価関数に置き換える
    """
    precision = 0.5
    recall = 0.5
    f1 = 0.5
    return precision, recall, f1


# ===== JSONスナップショット読み込み =====
def load_snapshots(task_dir: Path) -> List[dict]:
    snapshots = []

    for json_path in task_dir.glob("*.json"):
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
            data["parsed_time"] = datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            )
            snapshots.append(data)

    # 時刻順に並べる
    return sorted(snapshots, key=lambda x: x["parsed_time"])


# ===== タスク1件の処理 =====
def process_task(task_id: str, tool: str):
    task_dir = DATASET_ROOT / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_id}")

    task_type = "A" if task_id.startswith("taskA") else "B"
    snapshots = load_snapshots(task_dir)

    out_path = OUTPUT_DIR / f"{task_id}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ヘッダ
        writer.writerow([
            "task_type",
            "task_id",
            "tool",
            "timestamp",
            "precision",
            "recall",
            "f1"
        ])

        # 各スナップショット = 1行
        for snap in snapshots:
            precision, recall, f1 = evaluate(snap["text"], task_type)
            writer.writerow([
                task_type,
                task_id,
                tool,
                snap["parsed_time"].isoformat(),
                precision,
                recall,
                f1
            ])


# ===== 入力CSV処理 =====
def main(input_csv: Path):
    with input_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            process_task(row["taskID"], row["tool_condition"])


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]))
