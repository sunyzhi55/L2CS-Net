# 眼睑闭合指标增量管线（PERCLOS / 眨眼率 / 长闭眼时长）

本文档覆盖：方法、环境、运行、输出格式、本次实测结果、**已知局限**、以及下游任务如何接入。
面向"拿着这份文档直接做下游分析"的读者，因此第 7、8 节是重点，不要跳。

- 代码：`scripts/eye_metrics_core.py`（纯 numpy 计算核心）、`scripts/extract_eye_metrics_batch.py`（视频 IO + CLI）
- 测试：`tests/test_eye_metrics_core.py`（61 例，不需要 GPU/数据/mediapipe 即可运行）
- 本次结果：`hongmo_results/eye_metrics/`（80 个条件单元）
- 上游需求：研究方案 §12.2「状态操纵检验：独立客观通道」

---

## 1. 定位与边界

| | |
|---|---|
| 输入 | 与注视预处理**同一批** 30 fps 普通 RGB 摄像头视频（每人每条件一段完整 5 分钟，共 80 段） |
| 与三步主流程的关系 | **完全独立并行**。不读写逐帧 gaze `jsonl`，不经过 L2CS / OpenVINO / SFM / TF 校准 |
| 运行环境 | 独立 conda 环境（mediapipe 与仓库 `openvino==2022.3.0` / `numpy==1.23.4` 冲突） |
| 输出粒度 | **每个 `(id, difficulty, state)` 一行**（外加每个视频文件一行）；不是逐帧 |
| 与结果汇合的方式 | 在**分析层**按三元组 join，不在预处理层拼接 |

独立性是有意的：PERCLOS 等指标要作为"独立客观通道"去检验状态操纵，若与注视估计共享链路就不再独立。本管线不读 pitch/yaw、不经过注视管线里的中值滤波与校准网络。

---

## 2. 方法

### 2.1 主信号：虹膜遮蔽度（不是 EAR 幅度比）

对每只眼，在"眼坐标系"里计算：`u` 轴沿内外眼角连线，`v` 轴与之垂直，原点取虹膜 5 点均值。

```
遮蔽度 occ = 1 - 可见虹膜垂直高度 / 虹膜垂直直径
           = 1 - [ min(下睑_v, r) - max(上睑_v, -r) ]_+ / (2r)
```

`r`（虹膜半径）取**虹膜水平直径的一半**，并用该被试全部帧的中位数作常数。理由：上下眼睑只遮蔽虹膜的垂直方向，水平直径不受闭合程度影响，因此是一个稳定且**与条件标签无关**的尺度。

对虹膜 5 个点**不猜任何角色**：中心取 5 点均值，水平半径取投影绝对值最大者。所以只要这 5 个索引落在同一只眼的虹膜上就正确。

**为什么不用 EAR。** EAR 路线的闭合度要除以"该被试全开时的开合度"，而关键点模型在闭眼时不会把下睑点推到上睑点上。实测动态范围被压到 0–0.56，经典 PERCLOS 阈值够不到，全批 80 单元 PERCLOS 恒为 0。虹膜法把量程真正打开（实测遮蔽度 0.22–0.72，事件遮蔽度可达 1.00）。

EAR 仍作为副信号计算并写入 npz（列 `ear_r/ear_l`），用于与旧结果对照，不参与指标计算。

### 2.2 双眼共识与逐帧门控

- 左右眼各自算遮蔽度，**两眼的遮蔽度差 ≤ `sym_tol`(0.25)** 才保留该帧，取均值（Hering 定律：双眼同步闭合，大幅不对称即遮挡/关键点失败）。
- 眼裂宽度须在该被试中位数的 `[0.5, 2.0]` 倍内（剔除极端 yaw/遮挡）。
- `虹膜直径/眼裂宽度` 须在生理区间 `[0.22, 0.70]`（真人约 0.42）；越界判该帧无效——这是虹膜索引错位的主要征兆。
- 两眼独立入列，某一只眼坏掉不会废掉另一只（可用 `--eyes r` 走单眼）。
- **没有**任何"几何合理性门控"去拦截数据；索引错位一律靠报告列 + 叠加图暴露。

### 2.3 判据与事件分档

| 项 | 取值 | 说明 |
|---|---|---|
| PERCLOS 闭合判据 | **P50** 为主，另报 40% / 60% | 遮蔽虹膜垂直高度 ≥50% 即计为闭。经典 75%/80% 说的是"遮蔽瞳孔高度的比例"，与本量不同尺 |
| 比率分母 | **有效时长**，非总时长 | 否则遮挡不均会系统性压低 PERCLOS |
| 眨眼 vs 长闭眼 | 时长 <500 ms 为眨眼；≥500 ms 为长闭眼 | 两类**互斥**，不重复计数 |
| 眨眼率 | 同时报 ≥100 / ≥150 / ≥200 ms 三档 | 30 fps 下短眨眼被系统性漏计，不可只报一个数 |
| 长闭眼分档 | ≥500 / ≥1000 / ≥2000 ms | 方案中"微睡眠"的行为学代理 |
| 条件水平聚合 | 分子分母**各自求和**再相除 | 不是逐段求均值；避免比值均值偏倚 |
| 事件不跨无效帧合并 | 被无效帧隔断算两个事件 | |

事件时长取 `n_frames/fs`，30 fps 下至多低估 1 帧（期望 0.5 帧）；对 ≥500 ms 分档可忽略，对 100 ms 级眨眼不可忽略——这正是必须整档并报的原因。

### 2.4 定标完全不用条件标签

被试级常数只有 `虹膜直径中位数` 与 `眼裂宽度中位数`，跨该被试**全部**片段 pooled，与 alert/sleepy 标签无关。因此不存在"用 alert 段定尺子再声称 sleepy 更闭"的循环性（有测试 `test_scales_ignore_condition_labels` 锁住）。

---

## 3. 环境

```bash
conda create -n eyemetrics python=3.10 -y && conda activate eyemetrics
pip install mediapipe numpy opencv-python scipy
python -c "import mediapipe as mp, cv2; print(mp.__version__, cv2.__version__)"
```

不要装进现有环境。`mediapipe` 需要 numpy≥1.26，与仓库 `numpy==1.23.4` + `openvino==2022.3.0` 冲突。

---

## 4. 输入布局

两种布局自动识别，可混在同一棵目录树里；同一条件两者都命中时**优先连续视频**。

```
连续视频（主输入）  <root>/01/alert/easy/training_video.mp4
切片（兼容）        <root>/01_easy_alert/clip_1.mp4
```

**只收 `--video_pattern` 匹配的文件（默认 `training_video.mp4`）。** 数据树里混有校准视频，按 `*.mp4` 递归会把它们拼进同一条件单元，污染 PERCLOS 分母与定标。被排除项会报数量，`--list_excluded` 逐条查看——**不要静默过滤**。

---

## 5. 运行

```bash
# ① 可行性裁决（几分钟）。看遮蔽度 max 是否接近 1.0、虹膜比例是否在生理区间
python scripts/extract_eye_metrics_batch.py --root <完整视频根目录> --dry_run 6

# ② 索引目视复核（首次使用/换数据集/换 mediapipe 版本必做）
python scripts/extract_eye_metrics_batch.py --root <完整视频根目录> \
    --dump_overlay ./results/eye_overlay --dump_units 2 --limit_units 1
#    叠加图：绿=上睑 红=下睑 蓝/黄=内外眼角 白色十字=虹膜5点

# ③ 单人冒烟，确认收尾那行"越界 0/4 个单元"
python scripts/extract_eye_metrics_batch.py --root <完整视频根目录> \
    --subjects 01 --output_dir ./results/eye_metrics --save_signals

# ④ 全量：按被试分 5 路并行
for s in 01_02_03_04 05_06_07_08 09_10_11_12 13_14_15_16 17_18_19_20; do
  nohup python scripts/extract_eye_metrics_batch.py --root <完整视频根目录> \
    --output_dir ./results/eye_metrics_${s%%_*} --subjects ${s//_/,.} --save_signals \
    > logs/eye_${s}.log 2>&1 &
done
```

三条硬性注意：

1. **不要用 `--limit_units` 跑正式结果**，也不要把同一被试的 alert 与 sleepy 拆到不同次调用里——被试级定标需要一次看到该被试全部单元。按 `--subjects` 切分是安全的。
2. **每路必须用不同 `--output_dir`**，否则并行进程互相覆盖 `eye_metrics_by_condition.csv`。跑完合并：
   ```bash
   python - <<'PY'
   import glob, pandas as pd
   pd.concat([pd.read_csv(f, dtype={"id": str}) for f in
              glob.glob('results/eye_metrics_*/eye_metrics_by_condition.csv')]
             ).to_csv('results/eye_metrics/eye_metrics_by_condition.csv', index=False)
   PY
   ```
3. **务必加 `--save_signals`**。它存逐帧几何量，之后改判据/做敏感性分析只需离线重算，不必重跑 mediapipe（本次实测的基线锚定对比就是这么做的，零算力）。

实测吞吐约 **76 帧/秒/进程**（CPU，XNNPACK）。80 段 ≈ 72.6 万帧 → 单进程约 2.6 小时，5 路并行约 40–70 分钟。

主要参数：`--root` `--layout {auto,clip,video}` `--video_pattern` `--subjects` `--eyes {both,r,l}` `--sym_tol` `--iris_ratio_lo/hi` `--min_eye_width_px` `--dry_run` `--dump_overlay` `--index_map` `--save_signals` `--limit_units` `--list_excluded`。完整列表见 `--help`。

---

## 6. 输出

| 文件 | 粒度 | 本次规模 |
|---|---|---|
| `eye_metrics_by_condition.csv` | 每个 `(id, difficulty, state)` 一行 | 80 行 × 68 列 |
| `eye_metrics_by_segment.csv` | 每个视频文件一行 | 80 行 × 51 列 |
| `signals/<unit>__<seg>.npz` | 逐帧几何量 + 定标常数 | 80 个文件 |

> ⚠️ **本次 CSV 的一个表头缺陷（代码已修，数据无需重跑）**
> `SUMMARY_COLS` 里 `median_eye_width_px` 被误写两次，因此 `hongmo_results` 下两个 CSV 的表头该列出现两次；pandas 读回时第二列会被自动改名为 `median_eye_width_px.1`，若直接 `df["median_eye_width_px"]` 会拿到一个 DataFrame 而不是 Series。读取时去重即可：
> ```python
> df = pd.read_csv(path, dtype={"id": str})
> df = df.loc[:, ~df.columns.duplicated()]     # 保留第一份，丢弃 .1
> ```
> 两列内容完全相同，**不影响任何指标数值**，也不需要重新提取。

npz 键：`lo_r/hi_r/id_r/w_r/ear_r`、`lo_l/hi_l/id_l/w_l/ear_l`、`fps`、`iris_d_ref`、`eye_w_ref`。
**注意**：数组是"至少一眼可用的帧"的**子序列**，未存原始帧号，因此**不能**按位置与逐帧 gaze `jsonl` 对齐。需要帧级融合时须先给 npz 加 `frame_idx`。

### 列含义

**标识**：`id`（字符串，保留前导零）`difficulty`(easy/hard) `state`(alert/sleepy) `unit` `layout` `eyes` `n_videos` `n_segments` `fs`

**质量与定标**

| 列 | 含义 |
|---|---|
| `valid_s` / `valid_ratio` | 有效时长 / 占该段总时长比例（§12.1 要求 ≥80%） |
| `median_eye_width_px` | 被试级眼裂宽度中位数（像素） |
| `iris_d_r_px` / `iris_d_l_px` / `iris_d_mean_px` | 被试级虹膜直径中位数（定标分母） |
| **`iris_d_over_eye_w`** | 虹膜直径/眼裂宽度，**判断虹膜索引是否正确的首要仪表**（生理 ~0.42，代码区间 0.22–0.70） |
| **`iris_ratio_lr`** | 左右眼虹膜直径比值，应接近 1；>2 提示某侧索引错位 |
| `eye_w_r_px` / `eye_w_l_px` | 左右眼眼裂宽度中位数 |
| `iris_ratio_median` / `iris_ratio_bad_share` | 逐帧左右虹膜直径比的中位数 / 越界帧占比 |
| `eye_ratio_median` / `eye_ratio_bad_share` | 同上，眼裂宽度 |
| `n_frames_total` `n_face_total` `n_eye_total` `n_ear_ok` `n_iris_r` `n_iris_l` `n_width_gate_ok` `n_consensus_ok` | **逐级漏斗**：读帧→检到人脸→至少一眼→EAR 可用→左右眼几何→宽度门控→双眼共识。用来定位帧掉在哪一级 |

**指标**

| 列 | 含义 |
|---|---|
| `perclos_50`（主）/ `perclos_40` / `perclos_60` | 遮蔽度 ≥ 阈值的时长占有效时长比例 |
| `closed_s_50` / `long_closed_s_50` | PERCLOS 的分子（总闭合时长 / 其中长闭眼部分）——**合成 all 条件必需** |
| `perclos_50_excl_long_closure` | 剔除长闭眼后的 PERCLOS，用于判断差异来自眨眼还是长闭眼 |
| `n_blink_lt500ms` / `blink_rate_lt500ms_per_min` | 眨眼事件数 / 次每分钟（不加时长下限） |
| `n_blink_ge100ms` `n_blink_ge150ms` `n_blink_ge200ms` 及对应 `_per_min` | 三档下限的眨眼率 |
| `n_closure_ge500ms/1000ms/2000ms` | 长闭眼事件数 |
| `closure_dur_ge500ms/1000ms/2000ms_s` | 长闭眼总时长（秒） |
| `closure_rate_ge500ms_per_min` / `closure_dur_per_min_ge500ms_s` | 长闭眼频次 / 每分钟闭合秒数 |
| `max_dur_s` | 该单元内最长的单次闭合事件时长 |
| `mean_occ` / `p95_occ`(片段级) | 遮蔽度均值 / 95 分位 |
| `n_event` | 达主判据的闭合事件总数 |

---

## 7. 本次实测结果（`hongmo_results`，2026-09-15）

### 7.1 完成度与测量质量

80/80 单元齐全（20 被试 × easy/hard × alert/sleepy），全部 `video` 布局、`eyes=both`。

| 仪表 | 实测范围 | 判定 |
|---|---|---|
| `iris_d_over_eye_w` | 0.444 – 0.502（中位 0.462） | ✅ 紧贴真人理论值 ~0.42 且离散极小 → 虹膜索引正确 |
| `iris_ratio_lr` | 1.001 – 1.127（中位 1.022） | ✅ 左右眼一致 |
| `median_eye_width_px` | 22.3 – 37.9（中位 30.7） | ✅ 远高于 18px 不可行线 |
| `valid_ratio` | 0.958 – 1.000（中位 0.998） | ✅ 远高于 §12.1 的 80% |
| 漏斗 `n_consensus_ok` | 8597 – 9188（均值 9034），< `n_frames_total` 9083 | ✅ 门控有实际工作量（非上一轮那种恒等 1.0 的假象） |
| `mean_occ` | 0.220 – 0.722 | ✅ 量程已打开（EAR 版只有 0.05–0.29） |
| `perclos_50` | 0.002 – 0.988，**80/80 非零** | ✅ 不再恒为 0 |
| `blink_rate_ge100ms_per_min` | 0.6 – 78.3，中位 **23.8** | ✅ 中位落在成人正常区间（10–25/min） |
| `n_closure_ge500ms` | 53/80 单元非零 | |

### 7.2 状态操纵检验（§12.2）

配对 Wilcoxon 单尾（sleepy > alert），被试内：

| 指标 | 同向 | alert | sleepy | Δ中位 | dz | p |
|---|---|---|---|---|---|---|
| **all** | | | | | | |
| PERCLOS-50 | **18/20** | 0.142 | 0.291 | +0.101 | +0.74 | **5.2e-05** |
| 长闭眼时长/min | **17/20** | 4.27 s | 11.31 s | +2.39 | +0.52 | **6.4e-04** |
| 眨眼率 ≥100ms/min | 14/20 | 21.0 | 27.4 | +5.63 | +0.45 | 0.032 |
| mean_occ | **17/20** | 0.390 | 0.471 | +0.079 | **+1.11** | **3.1e-05** |
| **并集（至少一个同向）** | **19/20 = 95%** | | | | | |
| **easy** | | | | | | |
| PERCLOS-50 | 17/20 | 0.138 | 0.275 | +0.088 | +0.69 | 2.4e-04 |
| mean_occ | 16/20 | 0.401 | 0.461 | +0.047 | +0.74 | 1.4e-03 |
| **并集** | **19/20 = 95%** | | | | | |
| **hard** | | | | | | |
| PERCLOS-50 | **19/20** | 0.145 | 0.307 | +0.102 | +0.67 | 2.0e-04 |
| mean_occ | **18/20** | 0.380 | 0.480 | +0.095 | **+1.19** | **1.8e-05** |
| **并集** | **20/20 = 100%** | | | | | |

**结论：通过率 95%，落入方案"≥80% 按原计划主分析"档。状态操纵检验可以过关。**

只有 **13 号**不通过（四个指标全不同向）。

---

## 8. 已知局限 —— 下游必须知道，别踩

### L1（最重要）绝对 P50 判据受静息基线的跨被试差异影响

20 个被试的静息遮蔽度（p25）分布在 **0.26 – 0.51**。对基线已 ≥0.50 的人，P50 线画在他"正常睁开"的位置上。

受影响的具体证据：

- **12/80 单元 `perclos_50 > 0.5`**
- **07 号 hard-alert**：57.9% 的帧落在阈值 ±0.05 带内 → 344 个"事件"，其中 **41% 是 ≤2 帧的抖动**（纯阈值跳变，不是眨眼）
- **20 号 easy-sleepy**：遮蔽度 p5=0.50、p50=0.59，单次闭合事件长达 **171.7 秒**
- 对照干净单元：13 号带内帧仅 1.49%，01 号 3.80%

**后果与可用范围：**

| 用法 | 是否可用 |
|---|---|
| **被试内配对的方向检验 / 通过率**（§12.2 要的就是这个） | ✅ 可用。同一被试两条件共用一把尺子，基线偏置在配对相减中消掉。实测通过率 95% |
| **绝对 PERCLOS 数值跨被试比较**、"PERCLOS>0.3 即疲劳"这类阈值判断 | ❌ 不可用 |
| `blink_rate_*` 的绝对值（尤其 05/07/18/20） | ❌ 不可信，被阈值跳变污染 |
| `mean_occ` | ✅ 相对稳健（不依赖阈值），且效应量最大（dz 1.11） |

**已验证但尚未采纳的修正**：把闭合定义为相对该被试自身静息基线（全帧 pooled p25，无标签）的进一步遮蔽 `occ_rel = (occ-b)/(1-b)`。用本次 npz 离线重算的结果：

| | 绝对 P50（现状） | 基线锚定 |
|---|---|---|
| PERCLOS 同向 / p | 18/20，5.2e-05 | 18/20，**9.5e-06** |
| 眨眼率 p | 0.032 | **0.0014** |
| 长闭眼时长 p | 6.4e-04 | **1.5e-04** |
| alert 段 PERCLOS | 0.142 ❌ | **0.038** ✅ |
| alert 段眨眼率 | 21.0/min | **13.9/min** ✅ |
| alert 段长闭眼 | 4.27 s/min ❌ | **0.004 s/min** ✅ |
| 最长单事件 | 171.7 s ❌ | **11.7 s** |

方向检验结论不变（18/20），但绝对值合理得多、p 值更强。**若下游需要报绝对 PERCLOS 数值，应当先切到这个版本**——从 npz 离线重算即可，不需重跑 mediapipe。

### L2 超过 3 秒的"闭合"不是眨眼/微睡眠代理

`max_dur_s` 中位 1.33 s 但最大 171.7 s。>3 s 的持续闭眼通常是低头/看下方（眼睑下移但非困倦）或真睡，与 0.5–2 s 的微睡眠代理是不同现象。报 `n_closure_ge2000ms` / `max_dur_s` 时必须逐例核查，不能整体当作疲劳证据。

### L3 眨眼率只能作下限估计

30 fps 下 100 ms 眨眼只有 3 帧，短眨眼被系统性漏计。**必须整档并报（≥100/150/200 ms），不可只挑一个数**。

### L4 `mean_occ` 这个列名在三个版本里量纲不同

| 版本 | `mean_occ` 定义 | 位置 |
|---|---|---|
| EAR 版（已废弃） | `1 - EAR/EAR_ref(alert段)` | `results/`（旧） |
| **虹膜绝对版（当前）** | 虹膜被遮蔽比例均值 | `hongmo_results/` |
| 基线锚定版（未采纳） | 相对静息基线的进一步遮蔽比例 | — |

**不得把新旧 CSV 拼接或对比。** 旧目录应归档隔离（如 `results/eye_metrics_EAR_deprecated`）。

### L5 术语：没有 EEG 就不要写"微睡眠"

方案 §12.2 原文写"微睡眠/闭眼时长"，但微睡眠的定义性判据在 EEG。本数据集无 EEG，实测到的是 **长闭眼事件（≥500 ms）**。论文里应写 `prolonged eye closure (≥500 ms)`，并在 Limitations 注明"未同步 EEG，无法区分长闭眼与微睡眠"。

### L6 `valid_ratio` 高不等于测量正确

本次 `valid_ratio` 0.958–1.000 说明被试头部基本静止、门控几乎不触发，它**不能**作为数据质量证据。真正的质量证据是 `iris_d_over_eye_w`（0.462，紧贴解剖理论值）与 `iris_ratio_lr`（1.022）。论文里报质量要引后者。

### L7 虹膜索引 468–477 未经官方文档核实

MediaPipe 仓库对应文件路径 404，检索只返回博客。当前依据是间接的：`iris_d/eye_w` 落在生理区间且离散极小 + 叠加图目视。换 mediapipe 版本或换数据集后**必须重跑 `--index_map` 复核**。

---

## 9. 下游接入

### 9.1 join 契约

```python
import pandas as pd
eye  = pd.read_csv("hongmo_results/eye_metrics/eye_metrics_by_condition.csv", dtype={"id": str})
eye  = eye.loc[:, ~eye.columns.duplicated()]        # 见 §6 的表头重复缺陷
gaze = pd.read_csv("<你已有的注视条件级结果>", dtype={"id": str})

# 数据里任务名是 sleepy1/sleepy2，本管线已归一成 sleepy；你那边若用原始名必须同样归一
gaze["state"] = gaze["state"].str.replace(r"\d+$", "", regex=True)
gaze["id"]    = gaze["id"].astype(str).str.zfill(2)     # 别让它被读成整数 1

m = gaze.merge(eye, on=["id", "difficulty", "state"], validate="one_to_one")
assert len(m) == 80, f"只剩 {len(m)} 行，join 键没对上（不是数据少了）"
```

两个静默失败点：`id` 前导零、`sleepy1 → sleepy` 归一化。**inner merge 掉行不会报错，必须 assert 行数。**

### 9.2 合成 all 条件

`all` = easy + hard 合并。**比值不能相加**，必须分子分母各自求和：

```python
NUM = ["closed_s_50", "valid_s", "n_blink_ge100ms",
       "closure_dur_ge500ms_s", "n_closure_ge500ms", "long_closed_s_50"]
allc = m.groupby(["id", "state"])[NUM].sum()
allc["perclos_50"]  = allc.closed_s_50 / allc.valid_s
allc["blink_per_min"] = allc.n_blink_ge100ms / (allc.valid_s / 60)
allc["closure_dur_per_min"] = allc.closure_dur_ge500ms_s / (allc.valid_s / 60)
```

`mean_occ` 这类均值列按 `valid_s` 加权平均，不要直接取 mean。

### 9.3 §12.2 状态操纵检验

```python
from scipy.stats import wilcoxon

def delta(col):                       # allc 已 unstack 前先算好
    w = allc.reset_index().pivot_table(index="id", columns="state", values=col)
    return w["sleepy"] - w["alert"]

D = pd.DataFrame({c: delta(c) for c in
                  ["perclos_50", "blink_per_min", "closure_dur_per_min"]})
D["PASS"] = (D > 0).any(axis=1)
pass_rate = D.PASS.mean()            # 本次 0.95 -> >=80% 档，按原计划主分析
print(D.round(4).to_string())        # 未通过者进附录，不得静默剔除
```

判据映射（方案 §12.2 的候选指标 → 本管线列名）：

| 方案指标 | 用哪一列 | 命名注意 |
|---|---|---|
| ΔPERCLOS > 0 | `perclos_50`（主），40/60 做敏感性 | 报绝对值前先看 L1 |
| Δ眨眼率 > 0 | `blink_rate_ge100ms_per_min` 等三档 | 写成下限，不写单值 |
| 微睡眠/闭眼时长 > 0 | `closure_dur_ge500ms_s` 或 `closure_dur_per_min_ge500ms_s` | 改称"长闭眼事件"，见 L5 |
| Δ瞳孔直径 | 无 | 已按 §12.2 修订删除 |
| Δ漏检率 / Δ反应时 / Δ头部姿态方差 | 无 | 按 2026-09-14 决定不使用 |

### 9.4 论文里怎么陈述（可直接改用的措辞）

> 状态操纵检验使用视频眼动估计的独立客观通道。以 30 fps RGB 摄像头视频经 MediaPipe Face Mesh 提取虹膜遮蔽度（眼睑遮蔽虹膜垂直高度的比例），取该被试全部帧的虹膜水平直径中位数为尺度常数，不含任何条件标签信息。以遮蔽度 ≥50% 的时长占比（PERCLOS-P50）、≥500 ms 长闭眼事件累计时长、以及 ≥100/150/200 ms 三档眨眼率为指标，在 (被试, 难度, 状态) 条件水平上按分子分母求和聚合。20 名被试中 19 名（95%）至少一个指标呈同向变化（PERCLOS 18/20，配对 Wilcoxon 单尾 p=5.2×10⁻⁵，dz=0.74；长闭眼时长 17/20，p=6.4×10⁻⁴；眨眼率 14/20，p=0.032），达到 ≥80% 的主分析门槛。
>
> 局限：因未同步 EEG，≥500 ms 长闭眼仅作为微睡眠的行为学代理；30 fps 采样使短眨眼被系统性漏计，眨眼率以下界形式按三档并报；静息眼睑位置存在跨被试差异（遮蔽度基线 p25 分布于 0.26–0.51），故 PERCLOS 的绝对值不作跨被试比较，仅使用被试内配对的方向与效应量。

---

## 10. 离线重算与敏感性分析（不重跑 mediapipe）

`signals/*.npz` 存了逐帧几何量与定标常数，因此改阈值、改分档、换定标方式只需本地重算：

```python
import numpy as np, sys
sys.path.insert(0, "scripts")
from eye_metrics_core import default_config, iris_signals, combine_eyes, summarize_segment

cfg = default_config()                       # 在这里改 perclos_thresholds / blink_tier_ms / sym_tol
z = np.load("hongmo_results/eye_metrics/signals/01_easy_alert__training_video.npz")
ref_r = {"iris_d": float(z["iris_d_ref"][0]), "eye_w": float(z["eye_w_ref"][0])}
ref_l = {"iris_d": float(z["iris_d_ref"][1]), "eye_w": float(z["eye_w_ref"][1])}
o_r, k_r = iris_signals(z["lo_r"], z["hi_r"], z["id_r"], z["w_r"], ref_r, cfg)
o_l, k_l = iris_signals(z["lo_l"], z["hi_l"], z["id_l"], z["w_l"], ref_l, cfg)
occ, ok = combine_eyes(o_r, o_l, k_r, k_l, cfg)
print(summarize_segment(occ, ok, float(z["fps"][0]), cfg))
```

预先声明的敏感性维度（方案 §11 要求登记）：PERCLOS 阈值 40/50/60%、眨眼时长下限 100/150/200 ms、长闭眼分档 500/1000/2000 ms、`sym_tol` 0.25、绝对 P50 vs 基线锚定。

---

## 11. 代码与验证清单

| 文件 | 内容 |
|---|---|
| `scripts/eye_metrics_core.py` | 纯 numpy：虹膜几何、遮蔽度、双眼共识、事件化、条件聚合、可行性裁决。全部阈值集中在 `default_config()` 并随结果写出 |
| `scripts/extract_eye_metrics_batch.py` | 视频 IO、FaceMesh、两种目录布局发现与白名单、逐级漏斗报告、叠加图与 `--index_map` 标定、CSV/npz 落盘 |
| `tests/test_eye_metrics_core.py` | 61 例，本地 `python -m pytest tests/ -q` 直接可跑（不需 cv2/mediapipe/数据） |

已验证（本地合成信号，数值精确对齐注入真值）：遮蔽度 0/0.5/1 三档、虹膜水平直径不受眼睑遮蔽影响、缺失虹膜点返回 nan、轴朝向不变性、事件互斥与分档、跨层级一致性、分母用有效时长、`all` 条件可由分子分母合成、被试级定标与标签无关。

已验证（本次真实数据）：80/80 单元、`iris_d_over_eye_w` 0.444–0.502、`iris_ratio_lr` 1.00–1.13、四个指标方向与效应量（§7.2）、静息基线跨被试 0.26–0.51（L1）、基线锚定修正的离线对比（L1 表）。

**未验证 / 待办**：`EYE_IRIS_LANDMARKS` 未经官方文档核实（L7）；npz 无 `frame_idx`，不能与逐帧 gaze `jsonl` 做帧级对齐；>3 s 持续闭眼未单独分类（L2）；基线锚定版本未采纳为默认（L1）。
