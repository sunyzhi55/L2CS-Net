# FatigueGuard Preprocess

FatigueGuard 数据集预处理工具，用于把原始视频转成逐帧 `jsonl` 特征文件，并在最后一步对 gaze 估计点做 TensorFlow 校准。

本项目基于两个上游工程整合而来：

1. [L2CS-Net](https://github.com/Ahmednull/L2CS-Net/)
2. [WebCamGazeEstimation](https://github.com/FalchLucas/WebCamGazeEstimation)

## 1. 项目目标

整个流程分为三步：

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



最终输出的结果是适合训练和分析的逐帧 `jsonl` 文件。

## 2. 目录结构

```text
FatigueGuard-Preprocess/
├── scripts/                                # ★ 主要处理脚本
│   ├── FatigueGuard_preprocess_single.py   # 单视频预处理（第一步+第二步）
│   ├── FatigueGuard_preprocess_batch.py    # 数据集批处理
│   ├── tf_calibrate_jsonl_batch.py         # 第三步：TensorFlow 校准
│   ├── ext_jsonl_to_screen.py              # 外部 JSONL 屏幕映射
│   ├── ext_jsonl_tf_calibrate.py           # 外部 JSONL TF 校准
│   └── draw_gaze_on_video.py              # 注视方向可视化
├── l2cs/                                   # L2CS gaze 推理
├── gaze_tracking/                          # OpenVINO 人脸/关键点/眼部特征
├── sfm/                                    # 基于两帧关键点的 SFM
├── utilities/                              # 几何与工具函数
├── camera_data/                            # 相机标定数据
├── intel/                                  # OpenVINO 模型
├── models/                                 # L2CS 权重
├── tf_calibrate_model/                     # 预训练校准模型
├── demo.py                                 # L2CS 推理演示脚本
├── train.py                                # L2CS 训练脚本
├── test.py                                 # L2CS 测试脚本
├── tf_easy_calibration_by_trained_model.py # 旧版 Easy 校准（独立脚本）
├── tf_hard_calibration_by_trained_model.py # 旧版 Hard 校准（独立脚本）
├── README.md
├── EXT_JSONL_README.md
└── BATCH_PROCESSING_README.md
```

## 3. 数据处理流程

### 3.1 第一步：单帧/逐帧 gaze 提取

脚本：[scripts/FatigueGuard_preprocess_single.py](./scripts/FatigueGuard_preprocess_single.py)

输入：

- 一个视频文件
- 相机标定文件
- L2CS 权重
- 屏幕映射矩阵

输出：

- 逐帧 `jsonl`

每一帧会记录：

- `pitch_yaw_rad`
- `gaze_xyz`
- `gaze_screen_xy_mm`
- `gaze_screen_xy_px`
- `bbox`
- `landmarks`
- `confidence`

### 3.2 第二步：批量处理数据集

脚本：[scripts/FatigueGuard_preprocess_batch.py](./scripts/FatigueGuard_preprocess_batch.py)

该脚本会自动扫描 FatigueGuard 数据集，逐个处理：

- `alert/easy`
- `alert/hard`
- `sleepy/easy`
- `sleepy/hard`

并把单视频输出合并为最终的训练前特征文件。

### 3.3 第三步：TensorFlow 校准

脚本：[scripts/tf_calibrate_jsonl_batch.py](./scripts/tf_calibrate_jsonl_batch.py)

这一步会读取第二步生成的 `jsonl`，并使用已经训练好的 TensorFlow 校准模型修正估计点与真实目标点之间的偏差。

校准后会输出新的 `jsonl`，其中额外加入字段：

- `gaze_screen_tf_calibrate_xy_px`
- `deviation_px_before_calibrate`
- `deviation_px_after_calibrate`

同时脚本会在终端打印：

- 校准前平均误差
- 校准后平均误差

每一帧的校准前/后误差也会写入对应的 `jsonl` 记录中。`deviation_px_before_calibrate` 和 `deviation_px_after_calibrate` 均为像素欧氏距离；hard 任务会取估计点到 `target_centers_xy_px` 中最近目标点的距离。

## 4 输出格式

### 4.1 Easy 任务输出

校准前的 `jsonl` 中，典型字段如下：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998,
  "target_xy_px": [1280, 720]
}
```

校准后的输出格式为：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "gaze_screen_tf_calibrate_xy_px": [1268.4, 713.2],
  "target_xy_px": [1280, 720],
  "deviation_px_before_calibrate": 65.35,
  "deviation_px_after_calibrate": 13.19,
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998
}
```

### 4.2 Hard 任务输出

校准前的 `jsonl` 中，典型字段如下：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998,
  "target_centers_xy_px": [[1280, 720], [960, 540]]
}
```

校准后的输出格式为：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "gaze_screen_tf_calibrate_xy_px": [1268.4, 713.2],
  "target_centers_xy_px": [[1280, 720], [960, 540]],
  "deviation_px_before_calibrate": 67.12,
  "deviation_px_after_calibrate": 13.70,
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], [541.0, 320.0]],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], [541.0, 320.0]],
  "confidence": 0.998
}
```

## 5 主要脚本说明

### 5.1 `scripts/FatigueGuard_preprocess_single.py`

功能：

- 输入一个视频文件
- 逐帧提取 gaze 和人脸特征
- 输出单个 `jsonl`

常用参数：

- `--input`：视频路径
- `--jsonl`：输出 `jsonl` 路径
- `--directory`：工作目录
- `--device`：推理设备，例如 `cpu` 或 `cuda:0`
- `--weights`：L2CS 权重路径
- `--arch`：网络结构，默认 `ResNet50`
- `--mode`：`global` 或 `sfm`
- `--camera_data_dir`：相机标定目录
- `--max_frames`：最大处理帧数，`0` 表示全部

示例：

```bash
python scripts/FatigueGuard_preprocess_single.py --input path/to/training_video.mp4 --jsonl output/sample.jsonl --mode sfm --camera_data_dir ./camera_data
```

### 5.2 `scripts/FatigueGuard_preprocess_batch.py`

功能：

- 扫描 FatigueGuard 数据集目录
- 自动查找视频、目标点和校准矩阵
- 调用单视频预处理逻辑
- 生成批量 `jsonl`

常用参数：

- `--data_root`：数据集根目录
- `--output_dir`：输出目录
- `--camera_data_dir`：相机标定目录
- `--device`：推理设备
- `--weights`：L2CS 权重路径
- `--arch`：网络结构
- `--mode`：`global` 或 `sfm`
- `--sfm_openvino_device`：SFM 使用的 OpenVINO 设备
- `--max_frames`：单视频最多处理帧数
- `--subjects`：指定受试者 ID，例如 `01,02,03`

示例：

```bash
python scripts/FatigueGuard_preprocess_batch.py --data_root D:/data/FatigueGuard --output_dir D:/data/FatigueGuard_json  --device cuda:0  --weights models/L2CSNet_gaze360.pkl
```

### 5.3 `scripts/tf_calibrate_jsonl_batch.py`

功能：

- 读取第二步的 `jsonl`
- 用 `tf_calibrate_model/gaze_calibration_model.ckpt` 做推理校准
- 生成新的校准后 `jsonl`
- 在每帧记录中写入校准前/后的像素误差
- 打印校准前后的平均误差

常用参数：

- `--input_path`：输入 `jsonl` 文件或目录
- `--output_dir`：校准后的输出目录
- `--model_ckpt`：TensorFlow 校准模型路径

示例：

```bash
python scripts/tf_calibrate_jsonl_batch.py --input_path D:/data/FatigueGuard_jsonl --output_dir D:/data/FatigueGuard_jsonl_calibrated
```

### 5.4 `scripts/ext_jsonl_to_screen.py`

功能：

- 将外部方法生成的 JSONL（含 `pitch_yaw_rad`）通过校准矩阵映射到屏幕坐标
- 自动查找对应受试者的校准矩阵和目标点文件
- 输出符合 FatigueGuard 格式的 JSONL

详细说明见 [EXT_JSONL_README.md](./EXT_JSONL_README.md)。

### 5.5 `scripts/ext_jsonl_tf_calibrate.py`

功能：

- 对屏幕映射后的 JSONL 施加 TensorFlow 校准网络
- 输出校准后的坐标及校准前后误差

详细说明见 [EXT_JSONL_README.md](./EXT_JSONL_README.md)。

### 5.6 `scripts/draw_gaze_on_video.py`

功能：

- 在原始视频上逐帧绘制注视视线方向
- 可视化人脸框、面部关键点、注视箭头和屏幕注视点
- 输出带标注的视频文件

## 6 数据集目录约定

批处理脚本默认按下面结构查找数据:

```text
DataRoot/
├── camera_data/
│   ├── calibration_data.txt
│   ├── calibration_data_cm.txt
│   └── calibration_data_dm.txt
├── 01/
│   ├── alert/
│   │   ├── easy/
│   │   │   ├── training_video.mp4
│   │   │   └── centers_easy.txt
│   │   └── hard/
│   │       ├── training_video.mp4
│   │       └── Gaze_hard_centers.npy
│   └── sleepy/
│       ├── easy/
│       └── hard/
├── 02/
└── ...
```

### 6.1 校准矩阵查找规则

脚本会按以下顺序查找校准矩阵目录：

1. `subject_dir/results` 和 `subject_dir/STransG`
2. `subject_dir/state/difficulty/results` 和 `subject_dir/state/difficulty/STransG`
3. `subject_dir/state/results` 和 `subject_dir/state/STransG`

需要存在的文件：

- `STransG.npy`
- `StG.npy`
- `scaleWtG.npy`
- `STransW.npy`
- `StW.npy`

## 7 Easy 与 Hard 目标点格式

### 7.1 Easy

目标点文件：`centers_easy.txt`

支持格式：

```text
1280 720
960 540
```

或者：

```text
1280,720
960,540
```

规则：

- 每个目标点默认扩展为 `3` 秒
- 以视频帧率计算扩展帧数
- 如果扩展后的长度仍不足总帧数，使用最后一个目标点补齐

最终在每帧中增加：

```json
"target_xy_px": [x, y]
```

### 7.2 Hard

目标点文件：`Gaze_hard_centers.npy`

期望格式大致为：

```python
[
    {"image_index": 0, "centers": [(x1, y1), (x2, y2)]},
    {"image_index": 1, "centers": [(x1, y1), (x2, y2)]}
]
```

规则：

- 每个 `image_index` 默认扩展为 `180` 帧
- 如果扩展后的长度仍不足总帧数，使用最后一组中心点补齐

最终在每帧中增加：

```json
"target_centers_xy_px": [[x1, y1], [x2, y2]]
```

## 8. 安装依赖

### 8.1 第一步和第二步

依赖 L2CS-Net 和 WebCamGazeEstimation 的环境。
推荐使用 `environment.yml` 创建环境：

```bash
conda create -n fatigueguard python=3.9
conda activate fatigueguard
pip install -r requirements.txt
```
### 8.2 第三步
依赖 TensorFlow 1.x 和相关库，推荐使用 `tf_calibrate_environment.yml` 创建环境：

```bash
conda env create -f environment_tf.yml
```

## 9 模型资源

### 9.1 L2CS 权重

默认路径：

```text
models/L2CSNet_gaze360.pkl
```

如果文件不在默认位置，可以通过 `--weights` 指定。

### 9.2 TensorFlow 校准模型

默认路径：

```text
tf_calibrate_model/gaze_calibration_model.ckpt
```

包含：

- `gaze_calibration_model.ckpt.index`
- `gaze_calibration_model.ckpt.data-00000-of-00001`
- `gaze_calibration_model.ckpt.meta`

### 9.3 OpenVINO 模型

默认使用 `intel/` 目录下的模型，包括：

- `face-detection-adas-0001`
- `landmarks-regression-retail-0009`
- `facial-landmarks-35-adas-0002`
- `head-pose-estimation-adas-0001`
- `gaze-estimation-adas-0002`
- `open-closed-eye-0001`
- `PupilSegmentation`

## 10. 常见问题

### 10.1 找不到 `camera_data`

报错通常类似：

```text
No such file or directory: .../camera_data/calibration_data.txt
```

解决方法：

- 确认 `camera_data` 在数据集根目录下
- 或使用 `--camera_data_dir` 手动指定路径

### 10.2 找不到校准文件

如果批处理跳过某些任务，请检查是否存在：

- `STransG.npy`
- `StG.npy`
- `scaleWtG.npy`
- `STransW.npy`
- `StW.npy`

### 10.3 视频无法打开

请确认：

- `training_video.mp4` 路径正确
- 视频文件未损坏
- OpenCV 支持当前编码格式

### 10.4 `jsonl` 为空

优先检查：

- 模型权重是否存在
- 当前帧是否检测到人脸
- 标定文件与相机参数是否匹配
- 临时文件是否成功写出

### 10.5. OpenCV `resize` 空输入报错

如果遇到：

```text
OpenCV ... error: (-215:Assertion failed) !ssize.empty() in function 'resize'
```

说明某一帧的人脸或眼部裁剪为空。当前版本已经加入空裁剪保护，正常情况下会自动跳过这类帧。

### 10.6 运行脚本时提示模块找不到

所有主要脚本已移至 `scripts/` 目录。运行时需在**项目根目录**下执行，例如：

```bash
# 正确：在项目根目录运行
python scripts/FatigueGuard_preprocess_batch.py --data_root ...

# 错误：进入 scripts 目录后运行
cd scripts && python FatigueGuard_preprocess_batch.py ...  # 可能导致路径错误
```

脚本内部已自动处理跨目录导入问题，但相对路径参数（如 `--camera_data_dir ./camera_data`、`--weights models/...`）仍然基于当前工作目录解析。

## 11 推荐使用顺序

建议按下面顺序跑完整流程：

1. 先运行 `scripts/FatigueGuard_preprocess_single.py`，确认单个视频能正常输出。
2. 再运行 `scripts/FatigueGuard_preprocess_batch.py`，批量生成基础 `jsonl`。
3. 最后运行 `scripts/tf_calibrate_jsonl_batch.py`，生成带校准点的新 `jsonl`。

如需对接外部方法生成的 JSONL，参考 [EXT_JSONL_README.md](./EXT_JSONL_README.md)。

## 12 说明

如果你后续还要继续维护这个项目，README 还可以继续补充两部分内容：

- 每个输出字段的严格数学定义
- 校准模型的训练过程和数据组织方式
