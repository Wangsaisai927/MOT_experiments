from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


try:
    if not hasattr(np, "asfarray"):
        np.asfarray = np.asarray
    import motmetrics as mm
except Exception:
    mm = None


def load_mot_txt(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.empty((0, 10), dtype=float)
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data.astype(float)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.astype(float).copy()
    if len(out) == 0:
        return out
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def box_area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_assignment(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    triples = [(float(cost[i, j]), i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    triples.sort(key=lambda x: x[0])
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for value, i, j in triples:
        if value > max_cost:
            break
        if i not in used_rows and j not in used_cols:
            used_rows.add(i)
            used_cols.add(j)
            matches.append((i, j))
    return matches


def evaluate_with_motmetrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt[:, 0].astype(int).tolist()) | set(pred[:, 0].astype(int).tolist()))
    for frame in frames:
        gt_rows = gt[gt[:, 0].astype(int) == frame]
        pr_rows = pred[pred[:, 0].astype(int) == frame]
        dists = mm.distances.iou_matrix(gt_rows[:, 2:6], pr_rows[:, 2:6], max_iou=0.5)
        acc.update(gt_rows[:, 1].astype(int), pr_rows[:, 1].astype(int), dists)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1", "idp", "idr", "num_switches"], name="seq")
    return {
        "MOTA": float(summary.loc["seq", "mota"]),
        "IDF1": float(summary.loc["seq", "idf1"]),
        "IDP": float(summary.loc["seq", "idp"]),
        "IDR": float(summary.loc["seq", "idr"]),
        "IDSwitch": int(summary.loc["seq", "num_switches"]),
    }


def evaluate_fallback(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt_count = len(gt)
    pred_count = len(pred)
    tp = fp = fn = idsw = 0
    last_match: dict[int, int] = {}

    frames = sorted(set(gt[:, 0].astype(int).tolist()) | set(pred[:, 0].astype(int).tolist()))
    for frame in frames:
        gt_rows = gt[gt[:, 0].astype(int) == frame]
        pr_rows = pred[pred[:, 0].astype(int) == frame]
        gt_ids = gt_rows[:, 1].astype(int)
        pr_ids = pr_rows[:, 1].astype(int)
        gt_boxes = xywh_to_xyxy(gt_rows[:, 2:6]) if len(gt_rows) else np.empty((0, 4))
        pr_boxes = xywh_to_xyxy(pr_rows[:, 2:6]) if len(pr_rows) else np.empty((0, 4))
        ious = pairwise_iou(gt_boxes, pr_boxes)
        matches = greedy_assignment(1.0 - ious, max_cost=0.5)
        matched_gt = {g for g, _ in matches}
        matched_pr = {p for _, p in matches}
        for g, p in matches:
            gt_id = int(gt_ids[g])
            pr_id = int(pr_ids[p])
            if gt_id in last_match and last_match[gt_id] != pr_id:
                idsw += 1
            last_match[gt_id] = pr_id
        tp += len(matches)
        fn += len(gt_rows) - len(matched_gt)
        fp += len(pr_rows) - len(matched_pr)

    mota = 1.0 - (fn + fp + idsw) / gt_count if gt_count else 0.0
    idp = tp / pred_count if pred_count else 0.0
    idr = tp / gt_count if gt_count else 0.0
    idf1 = 2 * idp * idr / (idp + idr) if idp + idr else 0.0
    return {"MOTA": mota, "IDF1": idf1, "IDP": idp, "IDR": idr, "IDSwitch": idsw}


def evaluate(gt_file: str | Path, result_file: str | Path) -> dict[str, float]:
    gt = load_mot_txt(gt_file)
    pred = load_mot_txt(result_file)
    if mm is not None:
        return evaluate_with_motmetrics(gt, pred)
    raise ImportError("缺少 motmetrics 依赖，请先运行：pip install motmetrics pandas scipy")


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MOT-format tracking result against gt.txt.")
    parser.add_argument("--gt", required=True, help="ground truth txt, e.g. gt.txt")
    parser.add_argument("--result", required=True, help="tracking result txt")
    args = parser.parse_args()

    metrics = evaluate(args.gt, args.result)
    print("=" * 60)
    print("Tracking Evaluation")
    print("=" * 60)
    print(f"{'MOTA':>8} {'IDF1':>8} {'IDP':>8} {'IDR':>8} {'IDSwitch':>10}")
    print(
        f"{format_percent(metrics['MOTA']):>8} "
        f"{format_percent(metrics['IDF1']):>8} "
        f"{format_percent(metrics['IDP']):>8} "
        f"{format_percent(metrics['IDR']):>8} "
        f"{int(metrics['IDSwitch']):>10}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
