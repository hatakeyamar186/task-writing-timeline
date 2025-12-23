import csv
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Set

# ===== 設定 =====
DATASET_ROOT = Path("data/dataset")
OUTPUT_DIR = Path("outputs/timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===== 正解定義 =====
TASK_A_GOLD = {
    ("m1", "m7"),
    ("m2", "m3"),
    ("m2", "m5"),
    ("m6", "m7"),
    ("m6", "m8"),
    ("m8", "m7"),
    ("m9", "m1"),
    ("m10", "m9"),
}

TASK_B_GOLD = {
    ("m1", "m2"),
    ("m1", "m4"),
    ("m5", "m5"),
    ("m5", "m9"),
    ("m5", "m10"),
    ("m6", "m7"),
    ("m7", "m6"),
    ("m7", "m10"),
    ("m8", "m5"),
    ("m8", "m9"),
    ("m9", "m10"),
}


# ===== .textから回答を取り出す =====
def parse_text_relations(text: str) -> Set[Tuple[str, str]]:
    """
    text に書かれた関係を {(src, dst), ...} の集合として返す
    - 改行(\n)ごとに1関係
    - '#' で始まる行は無視
    - 各行の先頭2トークンのみ使用
    """

    relations: Set[Tuple[str, str]] = set()

    # \n はすでに Python 文字列では改行として解釈されている
    for raw_line in text.splitlines():
        line = raw_line.strip()

        # コメント・空行を無視
        if not line or line.startswith("#"):
            continue

        # 空白で分割（連続スペースもOK）
        parts = line.split()

        # 1行につき1関係 → 先頭2トークンが揃ったときだけ採用
        if len(parts) < 2:
            continue

        src = parts[0]
        dst = parts[1]

        relations.add((src, dst))

    return relations


# ===== precition / recall/ F1スコア の計算 =====
def compute_scores(
    predicted: Set[Tuple[str, str]],
    gold: Set[Tuple[str, str]]
):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


# ===== taskA / taskB を判定して評価する =====
def evaluate(text: str, task_type: str):
    predicted = parse_text_relations(text)

    if task_type == "A":
        gold = TASK_A_GOLD
    elif task_type == "B":
        gold = TASK_B_GOLD
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return compute_scores(predicted, gold)


# ===== JSONスナップショット読み込み =====
def load_snapshots(task_dir: Path) -> List[dict]:
    snapshots = []

    for txt_path in task_dir.glob("*.txt"):
        with txt_path.open(encoding="utf-8") as f:
            data = json.load(f)
            data["parsed_time"] = parse_timestamp(data["timestamp"])
            snapshots.append(data)

    # 時刻順に並べる
    return sorted(snapshots, key=lambda x: x["parsed_time"])


# ===== 小数秒を6桁に切り捨てる関数 =====
def parse_timestamp(ts: str) -> datetime:
    """
    ISO8601文字列を安全にdatetimeへ変換する
    - ナノ秒（7桁以上）は6桁に切り捨て
    - Z / +00:00 両対応
    """
    ts = ts.replace("Z", "+00:00")

    m = re.match(r"(.*)\.(\d+)([+-]\d\d:\d\d)", ts)
    if m:
        base = m.group(1)
        frac = m.group(2)[:6]   # ← 数字だけを6桁
        tz = m.group(3)
        ts = f"{base}.{frac}{tz}"

    return datetime.fromisoformat(ts)



# ===== タスク1件の処理 =====
def process_task(task_id: str, tool: str):
    task_dir = DATASET_ROOT / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_id}")

    task_type = "A" if task_id.startswith("taskA") else "B"
    snapshots = load_snapshots(task_dir)

    #　仮に入れたprint
    print(f"[DEBUG] {task_id}: snapshots={len(snapshots)}")

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
