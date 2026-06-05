# FatigueGuard 数据集批处理脚本使用说明


## 文件说明

已创建 `batch_preprocess.py` 脚本，用于批量处理 FatigueGuard 数据集。

## 功能特性

### 1. 数据集结构支持

支持以下数据集结构：
```
Data20260604/
├─ camera_data/               # 相机标定数据（所有受试者共享）
│   ├─ calibration_data.txt
│   └─ ...
│
├─ 01/
│   ├─ alert/                  # 警觉状态
│   │   ├─ easy/               # 简单难度
│   │   │   ├─ centers_easy.txt
│   │   │   └─ training_video.mp4
│   │   └─ hard/               # 困难难度
│   │       ├─ Gaze_hard_centers.npy
│   │       └─ training_video.mp4
│   │
│   ├─ sleepy/                 # 困倦状态
│   │   ├─ easy/
│   │   └─ hard/
│   │
│   ├─ result/                 # 校准与结果文件
│   └─ STransG/                # gaze 校准矩阵文件
│
├─ 02/ ... 20/
```

**重要**: `camera_data` 目录在数据集根目录下，所有受试者共享相同的相机标定参数。

### 2. 校准文件查找

脚本会自动在以下位置查找校准矩阵文件（STransG.npy, StG.npy等）：
1. `受试者目录/result` 和 `受试者目录/STransG`
2. `受试者目录/状态/难度/result` 和 `受试者目录/状态/难度/STransG`
3. `受试者目录/状态/result` 和 `受试者目录/状态/STransG`

### 3. Easy 任务处理

- **目标点文件**: `centers_easy.txt` (每行一个坐标，格式: `x y` 或 `x,y`)
- **增广策略**: 每个目标点对应 3 秒 = 90 帧 (30fps)
- **输出格式**:
```json
{
    "timestamp": 4.16,
    "frame_idx": 100,
    "pitch_yaw_rad": [pitch, yaw],
    "gaze_xyz": [x, y, z],
    "gaze_screen_xy_mm": [x, y],
    "gaze_screen_xy_px": [x, y],
    "target_xy_px": [x, y],
    "bbox": [x1, y1, x2, y2],
    "landmarks": [[x0,y0], [x1,y1], ...],
    "confidence": ...
}
```

### 4. Hard 任务处理

- **目标点文件**: `Gaze_hard_centers.npy`
- **格式**: `[{'image_index': 0, 'centers': [(x1,y1), (x2,y2), ...]}, ...]`
- **增广策略**: 每个 image_index 对应 180 帧
- **输出格式**:
```json
{
    "timestamp": 4.16,
    "frame_idx": 100,
    "pitch_yaw_rad": [pitch, yaw],
    "gaze_xyz": [x, y, z],
    "gaze_screen_xy_mm": [x, y],
    "gaze_screen_xy_px": [x, y],
    "target_centers_xy_px": [[x1,y1], [x2,y2], ...],
    "bbox": [x1, y1, x2, y2],
    "landmarks": [[x0,y0], [x1,y1], ...],
    "confidence": ...
}
```

## 使用方法

### 基本用法

```bash
python batch_preprocess.py \
    --data_root /path/to/Data20260604 \
    --output_dir /path/to/output \
    --device cuda:0 \
    --weights models/L2CSNet_gaze360.pkl
```

**注意**: 脚本会自动从 `data_root/camera_data` 读取相机标定数据，所有受试者共享相同的相机标定参数。

### 指定 camera_data 路径

如果 `camera_data` 不在数据集根目录下，可以手动指定：

```bash
python batch_preprocess.py \
    --data_root /path/to/Data20260604 \
    --output_dir /path/to/output \
    --camera_data_dir /path/to/camera_data \
    --device cuda:0
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | 是 | - | 数据集根目录（包含 01-20 的文件夹） |
| `--output_dir` | 是 | - | 输出目录 |
| `--camera_data_dir` | 否 | data_root/camera_data | 相机标定数据目录（所有受试者共享） |
| `--device` | 否 | cpu | 计算设备 (cpu 或 cuda:0) |
| `--weights` | 否 | models/L2CSNet_gaze360.pkl | L2CS 模型权重文件 |
| `--arch` | 否 | ResNet50 | 模型架构 |
| `--mode` | 否 | sfm | 投影模式 (global 或 sfm) |
| `--sfm_openvino_device` | 否 | CPU | SFM 的 OpenVINO 设备 |
| `--max_frames` | 否 | 0 | 最多处理帧数，0 表示全部 |
| `--subjects` | 否 | None | 指定受试者 ID（逗号分隔），如 `01,02,03` |

### 处理指定受试者

```bash
# 只处理受试者 01, 02, 03
python batch_preprocess.py \
    --data_root /path/to/Data20260604 \
    --output_dir /path/to/output \
    --subjects 01,02,03 \
    --device cuda:0
```

### 测试单个视频

```bash
# 处理受试者 01 的所有数据
python batch_preprocess.py \
    --data_root /path/to/Data20260604 \
    --output_dir /path/to/output \
    --subjects 01 \
    --max_frames 300
```

## 输出文件命名

生成的文件命名格式：`[id]_[difficulty]_[label].jsonl`

示例：
- `01_easy_alert.jsonl` - 受试者 01 的 easy 任务，alert 状态
- `01_hard_sleepy.jsonl` - 受试者 01 的 hard 任务，sleepy 状态
- `02_easy_sleepy.jsonl` - 受试者 02 的 easy 任务，sleepy 状态

## 注意事项

### 1. 远程服务器使用

由于你在远程 Linux 服务器上运行，建议：

```bash
# 使用 nohup 在后台运行
nohup python batch_preprocess.py \
    --data_root /path/to/Data20260604 \
    --output_dir /path/to/output \
    --device cuda:0 \
    > batch_process.log 2>&1 &

# 查看进度
tail -f batch_process.log
```

### 2. 依赖项

确保已安装所有依赖：
```bash
pip install numpy opencv-python pandas matplotlib
```

### 3. 校准文件检查

如果某些受试者的某些任务被跳过，检查是否缺少以下文件：
- `STransG.npy`
- `StG.npy`
- `scaleWtG.npy`
- `STransW.npy`
- `StW.npy`

### 4. 目标点文件格式

**Easy 任务** (`centers_easy.txt`):
```
1280 720
960 540
640 360
...
```

**Hard 任务** (`Gaze_hard_centers.npy`):
```python
import numpy as np
data = np.load('Gaze_hard_centers.npy', allow_pickle=True)
# data[0] = {'image_index': 0, 'centers': [(x1,y1), (x2,y2), ...]}
```

## 处理流程

1. 扫描数据集根目录，获取所有受试者文件夹
2. 对每个受试者：
   - 处理 alert/easy
   - 处理 alert/hard
   - 处理 sleepy/easy
   - 处理 sleepy/hard
3. 对每个任务：
   - 查找视频文件和目标点文件
   - 查找校准矩阵文件
   - 运行 GazeToPoint 提取特征
   - 增广目标点到每一帧
   - 合并特征和目标点，保存为 JSONL

## 故障排除

### 问题：找不到 camera_data 目录

**错误信息**: `[Errno 2] No such file or directory: '.../camera_data/calibration_data.txt'`

**解决方法**:
1. 确认 `camera_data` 目录在数据集根目录下
2. 目录结构应该是：
```
Data2/
├─ camera_data/
│   ├─ calibration_data.txt
│   └─ ...
├─ 01/
├─ 02/
...
```

3. 如果 `camera_data` 在其他位置，使用 `--camera_data_dir` 参数指定：
```bash
python batch_preprocess.py \
    --data_root /path/to/Data2 \
    --output_dir /path/to/output \
    --camera_data_dir /path/to/camera_data
```

### 问题：找不到校准文件

**解决**: 检查校准文件是否在以下位置之一：
```
/01/result/ 和 /01/STransG/
/01/alert/easy/result/ 和 /01/alert/easy/STransG/
/01/alert/result/ 和 /01/alert/STransG/
```

### 问题：视频无法打开

**解决**: 确认视频文件存在且格式正确
```bash
ffmpeg -i training_video.mp4
```

### 问题：JSONL 文件为空

**解决**: 检查特征提取是否成功，查看临时文件是否生成

## 示例运行日志

```
共找到 20 个受试者

处理受试者: 01
  处理视频: training_video.mp4
  视频: 2700 帧, 30 fps
  完成: 01_easy_alert.jsonl
  处理视频: training_video.mp4
  视频: 9000 帧, 30 fps
  完成: 01_hard_alert.jsonl
...

完成!
```

## 联系方式

如有问题，请检查：
1. `FatigueGuard_preprocess.py` 是否正常工作
2. 依赖库是否完整安装
3. 数据集结构是否符合要求
