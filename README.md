# MOT Baseline 统一运行接口

本目录用于统一运行多种多目标跟踪 baseline，并提供统一的评估脚本。

当前包含两个核心功能：

1. 给视频帧目录，直接输出 MOT 格式的跟踪结果 `txt`。
2. 给 `gt.txt` 和跟踪结果 `txt`，计算 `MOTA / IDF1 / IDP / IDR / IDSwitch`。

## 目录结构

```text
Baseline/
├── baseline_track.py   # 从视频帧运行指定 baseline，输出跟踪结果 txt
├── evaluate_mot.py     # 用 gt.txt 和 result.txt 计算跟踪指标
└── README.md           # 使用说明
```

## 支持的模型

`baseline_track.py` 的 `--model` 支持以下名称：

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

## 数据路径约定

默认视频帧目录统一写成：

```text
MOT\baseline\data\img1
```

也就是说，你的视频帧建议放成下面这种形式：

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

如果你的数据不在这个位置，也可以在命令里用 `--frames` 和 `--gt` 手动指定绝对路径。

## 环境安装

建议使用 Python 3.8 或以上版本。

```bash
pip install numpy opencv-python motmetrics pandas scipy
```

`evaluate_mot.py` 默认使用 `motmetrics`。如果没有安装，会直接提示安装，避免使用非标准 IDF1 口径。

## 1. 从视频帧输出跟踪结果 txt

运行 ByteTrack：

```bash
python baseline_track.py ^
  --model bytetrack ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\bytetrack_result.txt
```

运行 BoT-SORT：

```bash
python baseline_track.py ^
  --model botsort ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\botsort_result.txt
```

运行 TrackFormer：

```bash
python baseline_track.py ^
  --model trackformer ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\trackformer_result.txt
```

运行 MOTRv2：

```bash
python baseline_track.py ^
  --model motrv2 ^
  --frames MOT\baseline\data\img1 ^
  --output MOT\baseline\outputs\motrv2_result.txt
```

如果你的视频帧就放在默认位置 `MOT\baseline\data\img1`，可以省略 `--frames`：

```bash
python baseline_track.py ^
  --model bytetrack ^
  --output MOT\baseline\outputs\bytetrack_result.txt
```

如果不写 `--output`，默认输出到：

```text
MOT\baseline\outputs\<model>_result.txt
```

例如：

```bash
python baseline_track.py --model bytetrack
```

会生成：

```text
MOT\baseline\outputs\bytetrack_result.txt
```

输出的 txt 是 MOTChallenge 常用格式：

```text
frame,id,x,y,w,h,score,-1,-1,-1
```

示例：

```text
1,1,877.92,577.95,136.88,132.32,0.9000,-1,-1,-1
```

## 2. 计算跟踪指标

假设 ground truth 文件是：

```text
MOT\baseline\data\gt\gt.txt
```

跟踪结果是：

```text
MOT\baseline\outputs\bytetrack_result.txt
```

运行：

```bash
python evaluate_mot.py ^
  --gt MOT\baseline\data\gt\gt.txt ^
  --result MOT\baseline\outputs\bytetrack_result.txt
```

输出示例：

```text
============================================================
Tracking Evaluation
============================================================
    MOTA     IDF1      IDP      IDR   IDSwitch
   87.8%    67.8%    68.4%    67.1%         85
============================================================
```

## Python 代码中调用

你也可以在其它 Python 文件中直接调用：

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

## 注意事项

本目录的重点是统一 baseline 调用接口和评估流程。

`baseline_track.py` 内置了一个基于 OpenCV 前景分割的简单检测器，用于让代码可以直接从视频帧跑通。它不代表各论文官方完整模型的检测能力。

如果要做正式论文对比实验，建议：

1. 使用同一个检测器生成检测框。
2. 将检测框送入不同 baseline 的跟踪逻辑。
3. 使用同一个 `evaluate_mot.py` 和同一个 IoU 阈值计算指标。

这样得到的结果更适合写进论文或实验表格。
