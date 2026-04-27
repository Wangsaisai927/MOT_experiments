from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "缺少 opencv-python 依赖，请先运行：pip install opencv-python"
    ) from exc


SUPPORTED_MODELS = [
    "fairmot",
    "deepsort",
    "bytetrack",
    "botsort",
    "ocsort",
    "trackformer",
    "transcenter",
    "centertrack",
    "motr",
    "motrv2",
]


def xyxy_to_xywh(box: np.ndarray) -> tuple[float, float, float, float]:
    return float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])


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


def generalized_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    iou = pairwise_iou(a, b)
    if len(a) == 0 or len(b) == 0:
        return iou
    lt = np.minimum(a[:, None, :2], b[None, :, :2])
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    enc_area = wh[:, :, 0] * wh[:, :, 1]
    area_a = box_area(a)[:, None]
    area_b = box_area(b)[None, :]
    inter = iou * (area_a + area_b) / np.clip(1.0 + iou, 1e-12, None)
    union = area_a + area_b - inter
    return iou - np.where(enc_area > 0, (enc_area - union) / enc_area, 0.0)


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


@dataclass
class Detection:
    bbox: np.ndarray
    score: float


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    age: int = 0
    hits: int = 1
    confirmed: bool = True
    centers: list[np.ndarray] | None = None

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0], dtype=float)

    @property
    def size(self) -> float:
        return max(float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])), 0.0)

    def velocity(self) -> np.ndarray:
        if not self.centers or len(self.centers) < 2:
            return np.zeros(2, dtype=float)
        return self.centers[-1] - self.centers[-2]

    def warped_bbox(self) -> np.ndarray:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        c = self.center + self.velocity()
        return np.array([c[0] - 0.5 * w, c[1] - 0.5 * h, c[0] + 0.5 * w, c[1] + 0.5 * h], dtype=float)

    def update(self, bbox: np.ndarray, score: float) -> None:
        self.bbox = bbox.copy()
        self.score = float(score)
        self.age = 0
        self.hits += 1
        self.confirmed = True
        if self.centers is not None:
            self.centers.append(self.center)
            self.centers = self.centers[-3:]


class FrameDetector:
    """Simple frame detector for direct image-folder input.

    It uses OpenCV MOG2 foreground extraction. For publication-grade experiments,
    replace this detector with the same fish detector used by your method.
    """

    def __init__(self, min_area: int = 600, max_area: int = 80000, score: float = 0.9) -> None:
        self.bg = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=32, detectShadows=False)
        self.min_area = min_area
        self.max_area = max_area
        self.score = score

    def detect(self, image: np.ndarray) -> list[Detection]:
        mask = self.bg.apply(image)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: list[Detection] = []
        h_img, w_img = image.shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 2 or h <= 2:
                continue
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img - 1, x + w)
            y2 = min(h_img - 1, y + h)
            dets.append(Detection(np.array([x1, y1, x2, y2], dtype=float), self.score))
        return dets


class BaseTracker:
    def __init__(
        self,
        det_thresh: float = 0.4,
        new_thresh: float | None = None,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
        giou: bool = False,
    ) -> None:
        self.det_thresh = det_thresh
        self.new_thresh = det_thresh if new_thresh is None else new_thresh
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.giou = giou
        self.next_id = 1
        self.tracks: list[Track] = []

    def _new_track(self, det: Detection) -> Track:
        track = Track(self.next_id, det.bbox.copy(), float(det.score), 0, 1, self.min_hits <= 1)
        self.next_id += 1
        return track

    def _match(self, dets: list[Detection], tracks: list[Track], max_cost: float | None = None) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        unmatched_dets = set(range(len(dets)))
        unmatched_tracks = set(range(len(tracks)))
        matches: list[tuple[int, int]] = []
        if dets and tracks:
            det_boxes = np.stack([d.bbox for d in dets])
            track_boxes = np.stack([t.bbox for t in tracks])
            if self.giou:
                cost = 1.0 - generalized_iou(det_boxes, track_boxes)
                threshold = self.iou_threshold if max_cost is None else max_cost
            else:
                cost = 1.0 - pairwise_iou(det_boxes, track_boxes)
                threshold = 1.0 - self.iou_threshold if max_cost is None else max_cost
            for det_i, trk_i in greedy_assignment(cost, threshold):
                matches.append((det_i, trk_i))
                unmatched_dets.discard(det_i)
                unmatched_tracks.discard(trk_i)
        return matches, unmatched_dets, unmatched_tracks

    def update(self, detections: list[Detection]) -> list[Track]:
        dets = [d for d in detections if d.score >= self.det_thresh]
        matches, unmatched_dets, unmatched_tracks = self._match(dets, self.tracks)
        ret: list[Track] = []
        for det_i, trk_i in matches:
            track = self.tracks[trk_i]
            track.update(dets[det_i].bbox, dets[det_i].score)
            if track.hits >= self.min_hits:
                track.confirmed = True
            ret.append(track)
        for det_i in sorted(unmatched_dets):
            if dets[det_i].score >= self.new_thresh:
                ret.append(self._new_track(dets[det_i]))
        for trk_i in sorted(unmatched_tracks):
            track = self.tracks[trk_i]
            track.age += 1
            if track.age <= self.max_age:
                ret.append(track)
        self.tracks = ret
        return [t for t in ret if t.age == 0 and t.confirmed]


class ByteLikeTracker(BaseTracker):
    def __init__(self, high_thresh=0.5, low_thresh=0.1, new_thresh=0.6, first_iou=0.2, second_iou=0.5, max_age=60):
        super().__init__(high_thresh, new_thresh, first_iou, max_age, 1)
        self.low_thresh = low_thresh
        self.second_iou = second_iou

    def update(self, detections: list[Detection]) -> list[Track]:
        high = [d for d in detections if d.score >= self.det_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.det_thresh]
        matches, unmatched_high, unmatched_tracks = self._match(high, self.tracks)
        ret: list[Track] = []
        for det_i, trk_i in matches:
            track = self.tracks[trk_i]
            track.update(high[det_i].bbox, high[det_i].score)
            ret.append(track)

        remaining_tracks = [self.tracks[i] for i in sorted(unmatched_tracks)]
        low_matches, _, _ = self._match(low, remaining_tracks, 1.0 - self.second_iou)
        remaining_indices = sorted(unmatched_tracks)
        low_matched = set()
        for det_i, local_trk_i in low_matches:
            global_trk_i = remaining_indices[local_trk_i]
            track = self.tracks[global_trk_i]
            track.update(low[det_i].bbox, low[det_i].score)
            ret.append(track)
            low_matched.add(global_trk_i)

        for det_i in sorted(unmatched_high):
            if high[det_i].score >= self.new_thresh:
                ret.append(self._new_track(high[det_i]))
        for trk_i in sorted(unmatched_tracks):
            if trk_i in low_matched:
                continue
            track = self.tracks[trk_i]
            track.age += 1
            if track.age <= self.max_age:
                ret.append(track)
        self.tracks = ret
        return [t for t in ret if t.age == 0 and t.confirmed]


class CenterDistanceTracker(BaseTracker):
    def update(self, detections: list[Detection]) -> list[Track]:
        dets = [d for d in detections if d.score >= self.det_thresh]
        unmatched_dets = set(range(len(dets)))
        unmatched_tracks = set(range(len(self.tracks)))
        ret: list[Track] = []
        candidates: list[tuple[float, int, int]] = []
        for det_i, det in enumerate(dets):
            det_size = max(float((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])), 0.0)
            det_ct = np.array([(det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0])
            for trk_i, track in enumerate(self.tracks):
                dist = float(((det_ct - track.center) ** 2).sum())
                if dist <= det_size and dist <= track.size:
                    candidates.append((dist, det_i, trk_i))
        candidates.sort(key=lambda x: x[0])
        for _, det_i, trk_i in candidates:
            if det_i not in unmatched_dets or trk_i not in unmatched_tracks:
                continue
            track = self.tracks[trk_i]
            track.update(dets[det_i].bbox, dets[det_i].score)
            ret.append(track)
            unmatched_dets.remove(det_i)
            unmatched_tracks.remove(trk_i)
        for det_i in sorted(unmatched_dets):
            if dets[det_i].score >= self.new_thresh:
                ret.append(self._new_track(dets[det_i]))
        for trk_i in sorted(unmatched_tracks):
            track = self.tracks[trk_i]
            track.age += 1
            if track.age <= self.max_age:
                ret.append(track)
        self.tracks = ret
        return [t for t in ret if t.age == 0 and t.confirmed]


class MotionQueryTracker(BaseTracker):
    def __init__(self, det_thresh=0.4, new_thresh=0.5, max_age=15, match_cost=0.9, low_stage=False):
        super().__init__(det_thresh, new_thresh, 0.3, max_age, 1, giou=True)
        self.match_cost = match_cost
        self.low_stage = low_stage
        self.low_thresh = 0.1

    def _new_track(self, det: Detection) -> Track:
        track = Track(self.next_id, det.bbox.copy(), float(det.score), 0, 1, True, centers=[])
        track.centers.append(track.center)
        self.next_id += 1
        return track

    def update(self, detections: list[Detection]) -> list[Track]:
        high = [d for d in detections if d.score >= self.det_thresh]
        low = [d for d in detections if self.low_thresh <= d.score < self.det_thresh]
        ret: list[Track] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_high = set(range(len(high)))

        if high and self.tracks:
            track_boxes = np.stack([t.warped_bbox() if t.centers is not None else t.bbox for t in self.tracks])
            det_boxes = np.stack([d.bbox for d in high])
            cost = 1.0 - generalized_iou(track_boxes, det_boxes)
            for trk_i, det_i in greedy_assignment(cost, self.match_cost):
                track = self.tracks[trk_i]
                track.update(high[det_i].bbox, high[det_i].score)
                ret.append(track)
                unmatched_tracks.discard(trk_i)
                unmatched_high.discard(det_i)

        if self.low_stage and low:
            remaining = [self.tracks[i] for i in sorted(unmatched_tracks)]
            if remaining:
                track_boxes = np.stack([t.warped_bbox() if t.centers is not None else t.bbox for t in remaining])
                det_boxes = np.stack([d.bbox for d in low])
                cost = 1.0 - generalized_iou(track_boxes, det_boxes)
                remaining_ids = sorted(unmatched_tracks)
                for local_trk_i, det_i in greedy_assignment(cost, 0.7):
                    global_trk_i = remaining_ids[local_trk_i]
                    track = self.tracks[global_trk_i]
                    track.update(low[det_i].bbox, low[det_i].score)
                    ret.append(track)
                    unmatched_tracks.discard(global_trk_i)

        for det_i in sorted(unmatched_high):
            if high[det_i].score >= self.new_thresh:
                ret.append(self._new_track(high[det_i]))
        for trk_i in sorted(unmatched_tracks):
            track = self.tracks[trk_i]
            track.age += 1
            if track.age <= self.max_age:
                ret.append(track)
        self.tracks = ret
        return [t for t in ret if t.age == 0 and t.confirmed]


def create_tracker(model_name: str) -> BaseTracker:
    name = model_name.lower()
    if name == "fairmot":
        return CenterDistanceTracker(det_thresh=0.4, new_thresh=0.4, max_age=30)
    if name == "deepsort":
        return BaseTracker(det_thresh=0.0, new_thresh=0.0, iou_threshold=0.3, max_age=30, min_hits=3)
    if name == "bytetrack":
        return ByteLikeTracker(high_thresh=0.5, low_thresh=0.1, new_thresh=0.6, max_age=60)
    if name == "botsort":
        return ByteLikeTracker(high_thresh=0.5, low_thresh=0.1, new_thresh=0.6, max_age=30)
    if name == "ocsort":
        return BaseTracker(det_thresh=0.01, new_thresh=0.01, iou_threshold=0.3, max_age=30, min_hits=1)
    if name == "trackformer":
        return CenterDistanceTracker(det_thresh=0.4, new_thresh=0.4, max_age=32)
    if name == "transcenter":
        return MotionQueryTracker(det_thresh=0.3, new_thresh=0.3, max_age=60, match_cost=0.9, low_stage=True)
    if name == "centertrack":
        return CenterDistanceTracker(det_thresh=0.3, new_thresh=0.3, max_age=-1)
    if name == "motr":
        return MotionQueryTracker(det_thresh=0.45, new_thresh=0.5, max_age=3, match_cost=0.75, low_stage=False)
    if name == "motrv2":
        return MotionQueryTracker(det_thresh=0.6, new_thresh=0.6, max_age=15, match_cost=0.9, low_stage=True)
    raise ValueError(f"Unknown model '{model_name}'. Supported: {', '.join(SUPPORTED_MODELS)}")


def list_frames(frames_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])
    if not frames:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    return frames


def run_tracking(model_name: str, frames_dir: str | Path, output_txt: str | Path, min_area: int = 600) -> Path:
    frames_dir = Path(frames_dir)
    output_txt = Path(output_txt)
    detector = FrameDetector(min_area=min_area)
    tracker = create_tracker(model_name)
    rows: list[str] = []

    for frame_id, frame_path in enumerate(list_frames(frames_dir), start=1):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        detections = detector.detect(image)
        tracks = tracker.update(detections)
        for track in tracks:
            x, y, w, h = xyxy_to_xywh(track.bbox)
            rows.append(f"{frame_id},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{track.score:.4f},-1,-1,-1")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output_txt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline trackers on image frames and export MOT txt.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS, help="baseline model name")
    parser.add_argument("--frames", default=r"MOT\baseline\data\img1", help=r"image frame directory, default: MOT\baseline\data\img1")
    parser.add_argument("--output", default=None, help=r"output MOT-format txt path, default: MOT\baseline\outputs\<model>_result.txt")
    parser.add_argument("--min-area", type=int, default=600, help="minimum foreground area for the built-in detector")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = str(Path(r"MOT\baseline\outputs") / f"{args.model}_result.txt")

    out = run_tracking(args.model, args.frames, output, min_area=args.min_area)
    print(f"Tracking result saved to: {out}")


if __name__ == "__main__":
    main()
