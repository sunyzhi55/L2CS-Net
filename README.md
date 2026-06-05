# 1. FatigueGuard 预处理工具

## 1.0 说明

（1）本项目基于两个github项目进行合并：

1、https://github.com/Ahmednull/L2CS-Net/

2、https://github.com/FalchLucas/WebCamGazeEstimation


（2）预处理流程

```
# 输入：一段注视视频
# 第一步
使用l2cs项目中的gaze360模型提取每一时刻的(pitch,yaw)，对应文件：l2cs项目中的demo.py
提取注视视线，得到序列(pitch,yaw)

# 第二步
使用WebCamGazeEstimation项目把注视视线向量(pitch,yaw)映射到屏幕上的坐标点
链路：
(pitch,yaw) -> 3D注视向量gaze(x,y,z) -> 通过坐标变换投影到屏幕(mm)坐标
-> 融合滤波得到最终屏幕点Fgaze(mm) -> 转化为像素坐标 -> 得到点(x_px,y_px)
对应文件：gaze_to_point.py

# 第三步
训练校准网络，校准前面得到的估计点(x_px,y_px)与实际目标点的偏差，得到校准后的点(x_calib_px,y_calib_px)
```



---

## 1.1 介绍

本项目当前主要用于 FatigueGuard 数据预处理，核心目标是：

- 从视频中逐帧提取注视方向、屏幕映射点、人脸框、关键点和置信度。
- 将结果保存为 `jsonl`，便于后续训练、分析或标注对齐。
- 支持单视频处理和按被试目录批处理。

当前最重要的两个脚本是：

- [FatigueGuard_preprocess_single.py](/d:/code/SelfNet/ADFNet/FatigueGuard_preprocess_single.py:1)
- [FatigueGuard_preprocess_batch.py](/d:/code/SelfNet/ADFNet/FatigueGuard_preprocess_batch.py:1)

## 1.2 目录结构

```text
ADFNet/
├─ FatigueGuard_preprocess_single.py   # 单视频预处理：视频 -> 逐帧 JSONL
├─ FatigueGuard_preprocess_batch.py    # FatigueGuard 数据集批处理
├─ BATCH_PROCESSING_README.md          # 旧版批处理说明
├─ demo.py                             # L2CS 推理演示
├─ train.py                            # 原始 L2CS 训练脚本
├─ test.py                             # 原始 L2CS 测试脚本
├─ l2cs/                               # L2CS 模型、推理管线与可视化
├─ gaze_tracking/                      # OpenVINO 人脸/关键点/头姿/gaze 特征提取
├─ sfm/                                # 基于两帧关键点的 SFM/位姿估计
├─ utilities/                          # 工具函数
├─ camera_data/                        # 相机标定文件
├─ intel/                              # OpenVINO 模型文件
├─ models/                             # L2CS 权重目录，需要自行准备
└─ pyproject.toml                      # Python 项目依赖定义
```

## 1.3 核心脚本

### 1.3.1 `FatigueGuard_preprocess_single.py`

作用：

- 输入一个视频文件。
- 逐帧运行注视估计。
- 将每帧结果写入一个 `jsonl` 文件。

它内部主要完成三件事：

1. 使用 `l2cs.Pipeline` 估计 `pitch / yaw`。
2. 使用 `gaze_tracking.EyeModel` 补充人脸框、35 点关键点等特征。
3. 使用标定矩阵将 gaze 投影到屏幕坐标。

适用场景：

- 先验证某个视频是否能正常提特征。
- 单独处理某个样本。
- 为批处理流程排查问题。

### 1.3.2 `FatigueGuard_preprocess_batch.py`

作用：

- 扫描 FatigueGuard 数据集目录。
- 自动找到视频、目标点文件、标定矩阵。
- 调用 `FatigueGuard_preprocess_single.py` 中的 `GazeToPoint` 逐个视频处理。
- 将 gaze 特征与目标点标签合并，输出最终 `jsonl`。

适用场景：

- 整个 FatigueGuard 数据集批量预处理。
- 指定若干个被试批量处理。
- 将 easy/hard 两类任务统一整理成训练前特征文件。

## 1.4 单视频处理说明

单视频脚本入口是 [FatigueGuard_preprocess_single.py](/d:/code/SelfNet/ADFNet/FatigueGuard_preprocess_single.py:654)。

### 1.4.1 主要参数

- `--input`：输入视频路径。
- `--jsonl`：输出 `jsonl` 文件路径。
- `--directory`：工作目录，通常是项目根目录或当前样本目录。
- `--device`：推理设备，如 `cpu`、`cuda:0`。
- `--weights`：L2CS 模型权重，默认 `models/L2CSNet_gaze360.pkl`。
- `--arch`：L2CS 网络结构，默认 `ResNet50`。
- `--mode`：屏幕映射模式，`global` 或 `sfm`。
- `--stg_npy`：`STransG.npy` 路径。
- `--stw_npy`：`STransW.npy` 路径。
- `--scale_wtg`：`scaleWtG.npy` 路径。
- `--stg_aux_npy`：`StG.npy` 路径。
- `--stw_aux_npy`：`StW.npy` 路径。
- `--camera_data_dir`：相机标定目录路径。
- `--max_frames`：最多处理多少帧，`0` 表示全部。

### 1.4.2 使用示例

```bash
python FatigueGuard_preprocess_single.py --input path/to/training_video.mp4 --jsonl output/sample.jsonl --mode sfm --camera_data_dir ./camera_data
```

### 1.4.3 输出格式

输出文件为 `jsonl`，每一行对应一帧。当前脚本写出的主要字段如下：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "bbox": [412, 216, 871, 799],
  "landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998
}
```

字段说明：

- `timestamp`：当前帧时间戳，单位秒。
- `frame_idx`：帧号。
- `pitch_yaw_rad`：L2CS 输出的 `pitch`、`yaw`，单位弧度。
- `gaze_xyz`：由 `pitch / yaw` 转换得到的 3D gaze 向量。
- `gaze_screen_xy_mm`：映射到屏幕平面的毫米坐标。
- `gaze_screen_xy_px`：映射到屏幕平面的像素坐标。
- `bbox`：当前帧主脸框，格式为 `[x1, y1, x2, y2]`。
- `landmarks`：当前帧主脸关键点。
- `confidence`：检测置信度。

注意：

- 当前实现会跳过第 `0` 帧，不从第一帧开始写入。
- 如果某帧检测不到 gaze，相关字段可能是 `null` 或无法映射。

## 1.5 批处理说明

批处理脚本入口是 [FatigueGuard_preprocess_batch.py](/d:/code/SelfNet/ADFNet/FatigueGuard_preprocess_batch.py:144)。

它默认针对 FatigueGuard 数据集的被试目录结构工作。

### 1.5.1 期望的数据集结构

```text
DataRoot/
├─ camera_data/
│  ├─ calibration_data.txt
│  ├─ calibration_data_cm.txt
│  ├─ calibration_data_dm.txt
│  └─ ...
├─ 01/
│  ├─ alert/
│  │  ├─ easy/
│  │  │  ├─ training_video.mp4
│  │  │  └─ centers_easy.txt
│  │  └─ hard/
│  │     ├─ training_video.mp4
│  │     └─ Gaze_hard_centers.npy
│  ├─ sleepy/
│  │  ├─ easy/
│  │  └─ hard/
│  ├─ results/            # 可选
│  └─ STransG/            # 可选
├─ 02/
├─ 03/
└─ ...
```

### 1.5.2 批处理脚本做了什么

对每个被试，它会尝试处理：

- `alert/easy`
- `alert/hard`
- `sleepy/easy`
- `sleepy/hard`

对每个任务，它会：

1. 寻找 `training_video.mp4`。
2. 寻找对应的目标点文件。
3. 自动查找标定矩阵文件。
4. 调用单视频处理逻辑生成临时 gaze 特征 `jsonl`。
5. 将目标点按帧对齐后写入最终 `jsonl`。

### 1.5.3 标定矩阵查找规则

脚本会依次尝试以下位置：

1. `subject_dir/results` 和 `subject_dir/STransG`
2. `subject_dir/state/difficulty/results` 和 `subject_dir/state/difficulty/STransG`
3. `subject_dir/state/results` 和 `subject_dir/state/STransG`

并要求以下文件都存在：

- `STransG.npy`
- `StG.npy`
- `scaleWtG.npy`
- `STransW.npy`
- `StW.npy`

### 1.5.4 Easy 任务标签

文件名：`centers_easy.txt`

支持两种格式：

```text
1280 720
960 540
```

或

```text
1280,720
960,540
```

脚本逻辑：

- 每个目标点默认扩展为 `3` 秒。
- 实现里按 `fps * 3` 帧扩展。
- 若扩展后仍小于总帧数，使用最后一个目标点补齐。

最终会在每帧 JSON 中额外写入：

```json
"target_xy_px": [x, y]
```

### 1.5.5 Hard 任务标签

文件名：`Gaze_hard_centers.npy`

期望格式大致为：

```python
[
    {"image_index": 0, "centers": [(x1, y1), (x2, y2)]},
    {"image_index": 1, "centers": [(x1, y1), (x2, y2)]}
]
```

脚本逻辑：

- 每个 `image_index` 的 `centers` 默认扩展为 `180` 帧。
- 若扩展后仍小于总帧数，使用最后一组中心点补齐。

最终会在每帧 JSON 中额外写入：

```json
"target_centers_xy_px": [[x1, y1], [x2, y2]]
```

### 1.5.6 主要参数

- `--data_root`：数据集根目录。
- `--output_dir`：输出目录。
- `--device`：推理设备。
- `--weights`：L2CS 权重路径。
- `--arch`：网络结构，默认 `ResNet50`。
- `--mode`：`global` 或 `sfm`。
- `--sfm_openvino_device`：SFM 使用的 OpenVINO 设备。
- `--max_frames`：每个视频最多处理多少帧。
- `--subjects`：指定被试 ID，逗号分隔，如 `01,02,03`。
- `--camera_data_dir`：相机标定目录，默认 `./camera_data`。

### 1.5.7 使用示例

处理整个数据集：

```bash
python FatigueGuard_preprocess_batch.py ^
  --data_root D:/data/FatigueGuard ^
  --output_dir D:/data/FatigueGuard_jsonl ^
  --device cuda:0 ^
  --weights models/L2CSNet_gaze360.pkl
```

只处理指定被试：

```bash
python FatigueGuard_preprocess_batch.py ^
  --data_root D:/data/FatigueGuard ^
  --output_dir D:/data/FatigueGuard_jsonl ^
  --subjects 01,02,03 ^
  --device cuda:0
```

调试时限制帧数：

```bash
python FatigueGuard_preprocess_batch.py ^
  --data_root D:/data/FatigueGuard ^
  --output_dir D:/data/FatigueGuard_jsonl ^
  --subjects 01 ^
  --max_frames 300
```

### 1.5.8 输出文件命名

批处理输出文件命名格式为：

```text
[subject_id]_[task_type]_[label].jsonl
```

示例：

- `01_easy_alert.jsonl`
- `01_hard_alert.jsonl`
- `01_easy_sleepy.jsonl`
- `01_hard_sleepy.jsonl`

## 1.6 依赖与资源

### 1.6.1 Python 依赖

```bash
conda env create -f environment.yml -n <新环境名称>
```

### 1.6.2 模型文件

#### 1.6.2.1 L2CS 权重

默认使用：

```text
models/L2CSNet_gaze360.pkl
```

如果没有该文件，需要自行准备并通过 `--weights` 指定。

#### 1.6.2.2 OpenVINO 模型

当前默认依赖 `intel/` 下的模型，包括：

- `face-detection-adas-0001`
- `landmarks-regression-retail-0009`
- `facial-landmarks-35-adas-0002`
- `head-pose-estimation-adas-0001`
- `gaze-estimation-adas-0002`
- `open-closed-eye-0001`
- `PupilSegmentation`

### 1.6.3 标定文件

单视频或批处理都依赖：

- `camera_data/` 下的相机标定文件
- `STransG.npy`
- `StG.npy`
- `STransW.npy`
- `StW.npy`
- `scaleWtG.npy`

## 1.7 其他模块说明

- [gaze_tracking/model.py](/d:/code/SelfNet/ADFNet/gaze_tracking/model.py:1)：OpenVINO 推理封装，负责取人脸框、关键点、头姿和 gaze 特征。
- [sfm/sfm_module.py](/d:/code/SelfNet/ADFNet/sfm/sfm_module.py:1)：通过前后帧关键点估计相对位姿，用于 `sfm` 模式。
- [l2cs/pipeline.py](/d:/code/SelfNet/ADFNet/l2cs/pipeline.py:1)：L2CS 推理主管线。
- [demo.py](/d:/code/SelfNet/ADFNet/demo.py:1)：独立的 L2CS 演示脚本。
- [train.py](/d:/code/SelfNet/ADFNet/train.py:1) / [test.py](/d:/code/SelfNet/ADFNet/test.py:1)：原始训练与测试脚本，和 FatigueGuard 预处理主流程无直接耦合。

## 1.8 常见问题

### 1.8.1 找不到 `camera_data`

如果报错类似：

```text
No such file or directory: .../camera_data/calibration_data.txt
```

优先检查：

- `camera_data` 是否存在。
- `--camera_data_dir` 是否指向正确目录。

### 1.8.2 找不到标定矩阵

如果批处理时出现跳过任务，通常是缺少这些文件之一：

- `STransG.npy`
- `StG.npy`
- `scaleWtG.npy`
- `STransW.npy`
- `StW.npy`

### 1.8.3 视频无法打开

请检查：

- `training_video.mp4` 路径是否正确。
- 视频文件是否损坏。
- OpenCV 是否支持当前编码格式。

### 1.8.4 输出 `jsonl` 为空

优先排查：

- 模型权重是否存在。
- 当前帧是否检测到人脸。
- 标定文件和相机参数是否正确。
- 批处理生成的临时 `temp_*.jsonl` 是否成功写出。

## 1.9 处理建议

建议实际使用时按这个顺序来：

1. 先用 `FatigueGuard_preprocess_single.py` 跑一个视频，确认环境、模型和标定都正常。
2. 再用 `FatigueGuard_preprocess_batch.py` 跑一个被试目录。
3. 最后再放大到全量数据集。

如果你后面还会继续维护这个项目，README 可以继续往下补两类内容：

- 每个输出字段更精确的数学定义。
- 标定矩阵生成流程和来源说明。
