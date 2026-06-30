# 外部 JSONL 数据处理说明

本目录包含两个脚本，用于将**外部方法生成的注视预测 JSONL**通过本项目的校准矩阵和 TF 校准网络处理为最终结果。

两个脚本位于 `scripts/` 目录下：
- `scripts/ext_jsonl_to_screen.py` — 屏幕映射
- `scripts/ext_jsonl_tf_calibrate.py` — TF 校准

## 1. 背景

当使用其他方法（非 L2CS-Net）预测注视视线时，预测结果通常以 JSONL 格式输出。本工具将这些预测结果接入 FatigueGuard 的屏幕映射和校准流程，使其与原始管线的输出格式一致。

## 2. 外部 JSONL 输入格式

每个 JSONL 文件每行一条记录，必须包含以下字段：

```json
{
  "timestamps": 4.16,
  "frame_idx": 100,
  "pitch_yaw_deg": [6.87, -19.48],
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99]
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamps` | float | 时间戳（秒） |
| `frame_idx` | int | 帧序号 |
| `pitch_yaw_deg` | [float, float] | 俯仰角和偏航角（度） |
| `pitch_yaw_rad` | [float, float] | 俯仰角和偏航角（弧度） |
| `gaze_xyz` | [float, float, float] | 3D 注视单位向量 |

**注意**：脚本使用 `pitch_yaw_rad` 重新计算 gaze 向量（不用外部提供的 `gaze_xyz`），以确保坐标系与校准矩阵一致。

## 3. 文件命名约定

JSONL 文件名必须遵循以下格式：

```
{subject_id}_{difficulty}_{state}.jsonl
```

示例：`01_easy_alert.jsonl`、`02_hard_sleepy.jsonl`

其中：
- `subject_id`：受试者编号（数字，如 `01`、`02`）
- `difficulty`：任务难度（`easy` 或 `hard`）
- `state`：疲劳状态（`alert` 或 `sleepy`）

## 4. 数据目录结构

脚本需要访问原始数据目录以获取校准矩阵和目标点文件：

```
DataRoot/
├── 01/
│   ├── alert/
│   │   ├── easy/
│   │   │   ├── training_video.mp4    # 原始视频（SFM 模式需要）
│   │   │   └── centers_easy.txt      # Easy 目标点
│   │   └── hard/
│   │       ├── training_video.mp4
│   │       └── Gaze_hard_centers.npy  # Hard 目标点
│   ├── sleepy/
│   │   ├── easy/
│   │   └── hard/
│   ├── results/                       # 校准结果目录（候选）
│   ├── STransG/                       # 校准矩阵目录（候选）
│   └── alert/
│       ├── results/
│       └── STransG/
├── 02/
└── ...
```

### 4.1 校准矩阵查找规则

脚本按以下顺序查找校准矩阵目录：

1. `{subject_dir}/results` + `{subject_dir}/STransG`
2. `{subject_dir}/{state}/{difficulty}/results` + `{subject_dir}/{state}/{difficulty}/STransG`
3. `{subject_dir}/{state}/results` + `{subject_dir}/{state}/STransG`

需要的文件（在 `STransG` 目录下）：

- `STransG.npy` — 全局注视→屏幕变换矩阵
- `StG.npy` — 辅助标定点平移
- `scaleWtG.npy` — SFM 缩放因子
- `STransW.npy` — 屏幕→世界变换矩阵
- `StW.npy` — 世界坐标系辅助平移

## 5. 处理流程

```
外部 JSONL (pitch_yaw_rad)
        │
        ▼
┌──────────────────────────┐
│  脚本 A: 屏幕映射        │
│  pitch_yaw_rad → gaze    │
│  → 校准矩阵投影 → mm     │
│  → 像素坐标              │
│  + 合并目标点             │
└──────────────────────────┘
        │
        ▼ 输出 JSONL（含 gaze_screen_xy_px + target）
┌──────────────────────────┐
│  脚本 B: TF 校准          │
│  gaze_screen_xy_px       │
│  → TF 校准网络 → 修正坐标 │
│  + 计算校准前后误差       │
└──────────────────────────┘
        │
        ▼ 输出 JSONL（含 gaze_screen_tf_calibrate_xy_px + 误差）
```

## 6. 使用方法

**注意**：以下命令需要在**项目根目录**下运行，脚本位于 `scripts/` 目录中。

### 6.1 第一步：屏幕映射

```bash
python scripts/ext_jsonl_to_screen.py \
    --data_root /path/to/FatigueGuardData/Data \
    --jsonl_dir  /path/to/external_jsonl \
    --output_dir /path/to/screen_mapped \
    --mode global \
    --camera_data_dir ./camera_data
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_root` | （必填） | 原始数据根目录 |
| `--jsonl_dir` | （必填） | 外部 JSONL 文件目录 |
| `--output_dir` | （必填） | 输出目录 |
| `--mode` | `global` | 投影模式：`global` 或 `sfm` |
| `--camera_data_dir` | `./camera_data` | 相机标定数据目录 |

**global 模式**：直接使用 `STransG` 校准矩阵做投影，不需要视频文件。

**sfm 模式**：使用 `STransW @ WTransG` 做投影，需要读取视频帧通过 SFM 计算逐帧的 `WTransG`（头部运动补偿）。需要原始视频存在。

**输出示例：**

```bash
共找到 4 个 JSONL 文件
模式: global

[PROC] 01_easy_alert.jsonl
  [OK] 01_easy_alert.jsonl (5400 帧)
[PROC] 01_hard_alert.jsonl
  [OK] 01_hard_alert.jsonl (5400 帧)
...

完成！成功: 4, 跳过: 0
```

### 6.2 第二步：TF 校准

```bash
python scripts/ext_jsonl_tf_calibrate.py \
    --input_path /path/to/screen_mapped \
    --output_dir  /path/to/calibrated \
    --model_ckpt tf_calibrate_model/gaze_calibration_model.ckpt
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input_path` | （必填） | 第一步的输出目录或单个 JSONL 文件 |
| `--output_dir` | （必填） | 校准后的输出目录 |
| `--model_ckpt` | `tf_calibrate_model/gaze_calibration_model.ckpt` | TF 校准模型路径 |

**输出示例：**

```bash
[EASY] 01_easy_alert.jsonl | n=5400 | before=65.35 px | after=13.19 px
  -> /path/to/calibrated/01_easy_alert.jsonl
[HARD] 01_hard_alert.jsonl | n=5400 | before=67.12 px | after=13.70 px
  -> /path/to/calibrated/01_hard_alert.jsonl

[OVERALL] n=10800 | before=66.24 px | after=13.45 px
```

## 7. 输出 JSONL 格式

### 7.1 第一步输出（屏幕映射）

Easy 任务：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "target_xy_px": [1280, 720]
}
```

Hard 任务：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "target_centers_xy_px": [[1280, 720], [960, 540]]
}
```

### 7.2 第二步输出（TF 校准）

Easy 任务：

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
  "deviation_px_after_calibrate": 13.19
}
```

Hard 任务：

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
  "deviation_px_after_calibrate": 13.70
}
```

### 7.3 字段说明

| 字段 | 来源 | 说明 |
|---|---|---|
| `timestamp` | 外部 JSONL | 时间戳（秒） |
| `frame_idx` | 外部 JSONL | 帧序号 |
| `pitch_yaw_rad` | 外部 JSONL | 俯仰角和偏航角（弧度） |
| `gaze_xyz` | 脚本 A 重算 | 3D 注视单位向量（从 pitch_yaw_rad 计算） |
| `gaze_screen_xy_mm` | 脚本 A | 屏幕坐标（毫米） |
| `gaze_screen_xy_px` | 脚本 A | 屏幕坐标（像素） |
| `gaze_screen_tf_calibrate_xy_px` | 脚本 B | TF 校准后的屏幕坐标（像素） |
| `target_xy_px` | 脚本 A | Easy 任务目标点（像素） |
| `target_centers_xy_px` | 脚本 A | Hard 任务目标中心点（像素） |
| `deviation_px_before_calibrate` | 脚本 B | 校准前像素误差（欧氏距离） |
| `deviation_px_after_calibrate` | 脚本 B | 校准后像素误差（欧氏距离） |

## 8. 与原始管线的区别

| 对比项 | 原始管线 | 外部 JSONL 处理 |
|---|---|---|
| 视线提取 | L2CS-Net 逐帧推理 | 读取外部预测结果 |
| 第一帧 | 跳过不写入 JSONL | 跳过不处理 |
| gaze_xyz | 从 pitch_yaw 计算 | 从 pitch_yaw 重新计算 |
| SFM 模式 | 实时计算 WTransG | 需要视频，逐帧计算 WTransG |
| 人脸特征 | 输出 bbox/landmarks | 不输出（外部数据无此信息） |
| 校准矩阵 | 自动查找 | 同样自动查找 |
| 目标点合并 | 合并 | 同样合并 |

## 9. 常见问题

### 9.1 找不到校准矩阵

```
[SKIP] 找不到校准矩阵: 01/alert/easy
```

检查原始数据目录下是否存在 `STransG/` 目录及其中的 5 个 `.npy` 文件。

### 9.2 SFM 模式回退到 global

```
[WARN] SFM 模式需要视频但不存在: .../training_video.mp4，回退到 global 模式
```

SFM 模式需要原始视频来计算逐帧头部运动。如果视频不存在，自动回退到 global 模式。

### 9.3 校准前后误差为 NaN

说明该帧的 `gaze_screen_xy_px` 无效（人脸未检测到），校准网络无法处理。这在正常范围内。

### 9.4 依赖说明

- 脚本 A 依赖：`numpy`、`opencv-python`，以及本项目的 `FatigueGuard_preprocess_single`、`utilities`、`sfm`、`gaze_tracking` 模块
- 脚本 B 依赖：`numpy`、`tensorflow 1.x`

### 9.5 运行脚本时提示模块找不到

所有脚本位于 `scripts/` 目录。运行时需在**项目根目录**下执行，例如：

```bash
# 正确：在项目根目录运行
python scripts/ext_jsonl_to_screen.py --data_root ... --jsonl_dir ... --output_dir ...

# 错误：进入 scripts 目录后运行
cd scripts && python ext_jsonl_to_screen.py ...  # 可能导致路径和导入错误
```

脚本内部已自动将项目根目录加入 `sys.path`，确保可以正确导入 `gaze_tracking`、`sfm`、`utilities` 等模块。
