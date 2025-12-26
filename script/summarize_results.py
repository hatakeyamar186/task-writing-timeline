print("=== summarize_results.py is running ===")

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TIMESERIES_DIR = BASE_DIR / "outputs" / "timeseries"
OUTPUT_CSV = BASE_DIR / "outputs" / "summary.csv"

def summarize_one_task(csv_path: Path):
    """
    timeseries CSV (1 task) から
    集約用の1行データを作る
    """
    rows = []

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 数値は float に直す
            row["time"] = int(row["time"])
            row["precision"] = float(row["precision"])
            row["recall"] = float(row["recall"])
            row["f1"] = float(row["f1"])
            rows.append(row)

    if not rows:
        return None

    # ---- 最終スナップショット ----
    final_row = rows[-1]
    final_f1 = final_row["f1"]

    # ---- 最終F1に到達した最初の time ----
    time_to_final_f1 = None
    for r in rows:
        if r["f1"] == final_f1:
            time_to_final_f1 = r["time"]
            break

    # ---- 最終テキストまでの時間 ----
    time_to_final_text = final_row["time"]

    return {
        "taskID": final_row["task_id"],
        "tool": final_row["tool"],
        "time_to_final_f1": time_to_final_f1,
        "time_to_final_text": time_to_final_text,
        "precision": final_row["precision"],
        "recall": final_row["recall"],
        "f1": final_row["f1"],
    }


def main():
    print("[DEBUG] TIMESERIES_DIR =", TIMESERIES_DIR)
    print("[DEBUG] CSV files =", list(TIMESERIES_DIR.glob("*.csv")))
    rows = []

    for csv_path in sorted(TIMESERIES_DIR.glob("*.csv")):
        result = summarize_one_task(csv_path)
        if result:
            rows.append(result)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "taskID",
            "tool",
            "time_to_final_f1",
            "time_to_final_text",
            "precision",
            "recall",
            "F1",
        ])

        for r in rows:
            writer.writerow([
                r["taskID"],
                r["tool"],
                r["time_to_final_f1"],
                r["time_to_final_text"],
                r["precision"],
                r["recall"],
                r["f1"],
            ])


if __name__ == "__main__":
    main()
