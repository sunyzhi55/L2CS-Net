# 切片版数据处理说明（clip_preprocess_batch.py）

当原始注视视频被预先切成若干 `clip_N.mp4` 切片时，使用 `scripts/clip_preprocess_batch.py` 完成第一步（L2CS 提取注视）+ 第二步（屏幕映射），输出与原管线字段一致的 `jsonl`。第三步 TensorFlow 校准仍用现有的 `scripts/tf_calibrate_jsonl_batch.py`，无需改动。

## 1. 数据组织

切片数据根目录 `clip_root`：

```text
clip_root/
├── 01_easy_alert/
│   ├── clip_1.mp4
│   ├── clip_2.mp4
│   └── ...                 # easy 每片 3 秒；hard 每片 6 秒
├── 01_easy_sleepy/
├── 01_hard_alert/
└── ...
```

子文件夹命名 `[id]_[difficulty]_[task]`：
- `id`：受试者编号（如 `01`）
- `difficulty`：`easy` / `hard`
- `task`：`alert` / `sleepy`

> 容忍 `task` 上的数字后缀（如 `09_hard_sleepy1`），输出文件名保留原始文件夹名（`09_hard_sleepy1.jsonl`）。

目标点文件与校准矩阵仍在**原始数据路径** `data_root` 下（与原批处理一致）：

```text
data_root/
├── 01/
│   ├── alert/
│   │   ├── easy/
│   │   │   ├── centers_easy.txt          # easy 目标点（每行一个）
│   │   │   └── training_video.mp4        # （切片版不再需要，SFM 模式也不用）
│   │   └── hard/
│   │       └── Gaze_hard_centers.npy      # hard 目标点（每个 entry 一组中心）
│   ├── sleepy/...
│   ├── results/                           # 校准结果目录（候选）
│   └── STransG/                           # 校准矩阵目录（候选）
└── 02/...
```

## 2. 切片 ↔ 目标点对应关系

**每个切片视频对应目标点文件中的一行/一个 entry**（不再做 3 秒/180 帧的增广）：

- easy：`centers_easy.txt` 第 i 行 → `clip_i.mp4`，该片所有（保留）帧的 `target_xy_px = [x, y]`
- hard：`Gaze_hard_centers.npy` 第 i 个 entry 的 `centers` → `clip_i.mp4`，该片所有（保留）帧的 `target_centers_xy_px = [[x,y], ...]`

每个切片的**第一帧**注视提取为负值，统一丢弃（与原管线丢弃整段视频第 0 帧一致）。

## 3. 处理流程

```
clip_N.mp4
   │  逐帧 L2CS 提取 (pitch, yaw)
   ▼
3D 注视向量 gaze_xyz
   │  GazeToPoint 投影（global: STransG / sfm: STransW@WTransG）
   ▼
屏幕点 gaze_screen_xy_mm → gaze_screen_xy_px
   │  + 该切片对应的目标点
   ▼
拼接同一子文件夹所有 clip → [id]_[difficulty]_[task].jsonl
```

- L2CS 推理管线与 OpenVINO EyeModel **全局只加载一次**，所有文件夹/切片复用。
- 每个子文件夹加载一份校准矩阵（`GazeToPoint`）。
- 每个切片独立重置中值滤波队列与 SFM `frame_prev`，避免跨切片污染。
- `frame_idx` / `timestamp` 跨切片连续累加，便于后续滑动窗口训练。

## 4. 用法

在**项目根目录**运行（需要 `intel/` 与 `models/` 在相对路径下）：

```bash
python scripts/clip_preprocess_batch.py \
    --clip_root  /path/to/20260821_clip \
    --data_root  /path/to/FatigueGuardData/Data \
    --output_dir /path/to/output \
    --device cuda:0 \
    --mode sfm \
    --camera_data_dir ./camera_data


```

后台运行：

```bash
nohup python scripts/clip_preprocess_batch.py \
    --clip_root ... --data_root ... --output_dir ... \
    --device cuda:0 --mode sfm > clip_process.log 2>&1 &

nohup python scripts/clip_preprocess_batch.py --mode sfm > result_clip_process.log &

tail -f clip_process.log
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `--clip_root` | 是 | - | 切片数据根目录 |
| `--data_root` | 是 | - | 原始数据根目录（目标点 + 校准矩阵） |
| `--output_dir` | 是 | - | 输出 jsonl 目录 |
| `--camera_data_dir` | 否 | `./camera_data` | 相机标定目录 |
| `--device` | 否 | `cpu` | `cpu` 或 `cuda:0` |
| `--weights` | 否 | `models/L2CSNet_gaze360.pkl` | L2CS 权重 |
| `--arch` | 否 | `ResNet50` | 网络结构 |
| `--mode` | 否 | `sfm` | `global` / `sfm` |
| `--sfm_openvino_device` | 否 | `CPU` | SFM 的 OpenVINO 设备 |
| `--max_clips` | 否 | `0` | 每文件夹最多切片数（测试用，0=全部） |
| `--subjects` | 否 | - | 只处理指定受试者，如 `01,02` |

## 5. 输出格式

每个子文件夹输出一个 `[id]_[difficulty]_[task].jsonl`（校准前），每帧一行。easy：

```json
{
  "timestamp": 0.033,
  "frame_idx": 0,
  "clip_idx": 0,
  "clip_frame_idx": 1,
  "pitch_yaw_rad": [0.12, -0.34],
  "gaze_xyz": [0.01, -0.03, 0.99],
  "gaze_screen_xy_mm": [315.2, 182.1],
  "gaze_screen_xy_px": [1345, 702],
  "face_detection_bbox": [412, 216, 871, 799],
  "facial_landmark_35": [[520.0, 311.0], ...],
  "RetinaFace_bbox": [412, 216, 871, 799],
  "RetinaFace_landmarks": [[520.0, 311.0], ...],
  "confidence": 0.998,
  "target_xy_px": [1280, 720]
}
```

hard 把 `target_xy_px` 换为 `target_centers_xy_px: [[x,y], ...]`。

> 相比原管线额外附带 `clip_idx` / `clip_frame_idx` 两个溯源字段；下游训练只读取
> `gaze_screen_tf_calibrate_xy_px` 与 `target_*` 等字段，不受影响。

## 6. 第三步：TensorFlow 校准

输出 jsonl 可直接用现有脚本校准（无需改动）：

```bash
python scripts/tf_calibrate_jsonl_batch.py \
    --input_path /path/to/output \
    --output_dir /path/to/output_tfCali \
    --model_ckpt tf_calibrate_model/gaze_calibration_model.ckpt
```

校准后每帧增加 `gaze_screen_tf_calibrate_xy_px`、`deviation_px_before_calibrate`、
`deviation_px_after_calibrate` 字段。

## 7. 常见问题

- **找不到校准矩阵**：检查 `data_root/{id}/` 下是否存在 `STransG/`（含 5 个 `.npy`）。
- **切片数 ≠ 目标点数**：脚本会告警，超出的切片复用最后一个目标点。
- **SFM 模式**：不再需要原始 `training_video.mp4`（头部运动改由切片自身相邻帧估计）；若某帧 SFM 失败，沿用上一帧变换。
- **远程无显示**：脚本不调用任何 GUI/imshow，可在无 X11 服务器上运行。
