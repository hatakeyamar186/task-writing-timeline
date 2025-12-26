import csv
import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import List, Set
from decimal import Decimal, ROUND_HALF_UP

# ===== 設定 =====
DATASET_ROOT = Path("data/dataset")
OUTPUT_DIR = Path("outputs/timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===== 正解定義 =====
TASK_A_GOLD_TEXT = """
m1 m7
m2 m3
m2 m5
m6 m7
m6 m8
m8 m7
m9 m1
m10 m9
"""

TASK_B_GOLD_TEXT = """
m1 m2
m1 m4
m5 m5
m5 m9
m5 m10
m6 m7
m7 m6
m7 m10
m8 m5
m8 m9
m9 m10
"""


def canon_line(line: str) -> str:
    if not line:
        return ""

    txt = unicodedata.normalize("NFKC", line).lower().strip()
    toks = re.findall(r"\bm\d+\b", txt)

    if len(toks) < 2:
        return ""

    return f"{toks[0]} {toks[1]}"



def answer_set(text: str) -> Set[str]:
    result: Set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ★ コメント行を無視
        if not line or line.startswith("#"):
            continue

        key = canon_line(line)
        if key:
            result.add(key)

    return result



def compute_scores_js_compatible(
    gold_text: str,
    pred_text: str
):
    G = answer_set(gold_text)
    P = answer_set(pred_text)

    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return precision, recall, f1


# ===== taskA / taskB を判定して評価する =====
def evaluate(text: str, task_type: str):
    if task_type == "A":
        gold_text = TASK_A_GOLD_TEXT
    elif task_type == "B":
        gold_text = TASK_B_GOLD_TEXT
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return compute_scores_js_compatible(
        gold_text=gold_text,
        pred_text=text
    )


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


# ==== 小数点第２位以下を四捨五入 ====
def round_half_up(x: float, ndigits: int = 2) -> float:
    """
    厳密な四捨五入（0.5は必ず切り上げ）
    """
    q = Decimal("1").scaleb(-ndigits)  # 例: ndigits=2 → Decimal("0.01")
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


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
            "time",
            "precision",
            "recall",
            "f1"
        ])


        # 各スナップショット = 1行
        for i, snap in enumerate(snapshots, start=1):
            precision, recall, f1 = evaluate(snap["text"], task_type)
            writer.writerow([
                task_type,
                task_id,
                tool,
                i,          # time = 1,2,3,...
                round_half_up(precision, 2),
                round_half_up(recall, 2),
                round_half_up(f1, 2)
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
