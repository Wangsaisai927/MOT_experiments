# MOT Baseline Unified Running Interface

This directory is used to run multiple multi-object tracking baselines through a unified interface and provides a unified evaluation script.

It currently includes two core functions:

1. Given a video frame directory, directly output tracking results in MOT-format `txt`.
2. Given `gt.txt` and a tracking result `txt`, calculate `MOTA / IDF1 / IDP / IDR / IDSwitch`.

## Directory Structure

```text
Baseline/
├── baseline_track.py   # Run the specified baseline from video frames and output tracking result txt
├── evaluate_mot.py     # Calculate tracking metrics using gt.txt and result.txt
└── README.md           # Usage instructions
```

## Supported Models

The `--model` argument in `baseline_track.py` supports the following names:

```text
FairMOT
Deepsort
Bytetrack
Botsort
OCsort
Trackformer
TransCenter
CenterTrack
MOTR
MOTRv2
```

## Data Path Convention

The default video frame directory is uniformly written as:

```text
MOT\baseline\data\img1
```

That is, it is recommended to place your video frames in the following form:

```text
MOT/
└── baseline/
    └── data/
        ├── img1/
        │   ├── 000000.jpg
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ...
        └── gt/
            └── gt.txt
```

If your data is not in this location, you can also manually specify absolute paths in the command using `--frames` and `--gt`.

## Environment Installation

Python 3.8 or later is recommended.

```bash
pip install numpy opencv-python motmetrics pandas scipy
```

`evaluate_mot.py` uses `motmetrics` by default. If it is not installed, the script will directly prompt you to install it, avoiding the use of a non-standard IDF1 definition.

## 1. Output Tracking Result txt from Video Frames

Run ByteTrack:

```bash
python baseline_track.py ^
  --model bytetrack ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\bytetrack_result.txt
```

Run BoT-SORT:

```bash
python baseline_track.py ^
  --model botsort ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\botsort_result.txt
```

Run TrackFormer:

```bash
python baseline_track.py ^
  --model trackformer ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\trackformer_result.txt
```

Run MOTRv2:

```bash
python baseline_track.py ^
  --model motrv2 ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\motrv2_result.txt
```

If your video frames are placed in the default location `MOT\baseline\data\img1`, you can omit `--frames`:

```bash
python baseline_track.py ^
  --model bytetrack ^
  --output MOT\baseline\outputs\bytetrack_result.txt
```

If `--output` is not specified, the default output path is:

```text
MOT\baseline\outputs\<model>_result.txt
```

For example:

```bash
python baseline_track.py --model bytetrack
```

will generate:

```text
MOT\baseline\outputs\bytetrack_result.txt
```

The output txt uses the common MOTChallenge format:

```text
frame,id,x,y,w,h,score,-1,-1,-1
```

Example:

```text
1,1,877.92,577.95,136.88,132.32,0.9000,-1,-1,-1
```

## 2. Calculate Tracking Metrics

Assume the ground truth file is:

```text
MOT\baseline\data\gt\gt.txt
```

The tracking result is:

```text
MOT\baseline\outputs\bytetrack_result.txt
```

Run:

```bash
python evaluate_mot.py ^
  --gt MOT\baseline\data\gt\gt.txt ^
  --result MOT\baseline\outputs\bytetrack_result.txt
```

Example output:

```text
============================================================
Tracking Evaluation
============================================================
    MOTA     IDF1      IDP      IDR   IDSwitch
   87.8%    67.8%    68.4%    67.1%         85
============================================================
```

## Calling from Python Code

You can also call it directly from other Python files:

```python
from baseline_track import run_tracking
from evaluate_mot import evaluate

result_txt = run_tracking(
    model_name="bytetrack",
    frames_dir=r"MOT\baseline\data\img1",
    output_txt=r"MOT\baseline\outputs\bytetrack_result.txt",
)

metrics = evaluate(
    gt_file=r"MOT\baseline\data\gt\gt.txt",
    result_file=result_txt,
)

print(metrics)
```

## Notes

The focus of this directory is to provide a unified baseline calling interface and evaluation workflow.

`baseline_track.py` includes a simple detector based on OpenCV foreground segmentation, which allows the code to run directly from video frames. It does not represent the detection capability of the complete official models from the respective papers.

If you want to conduct formal comparison experiments for a paper, it is recommended to:

1. Use the same detector to generate detection boxes.
2. Feed the detection boxes into the tracking logic of different baselines.
3. Use the same `evaluate_mot.py` and the same IoU threshold to calculate metrics.

The results obtained this way are more suitable for inclusion in a paper or experiment table.
