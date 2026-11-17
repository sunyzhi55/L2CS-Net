# 补充实验：alert / fatigue 两次 session 的四点校准误差对比

回应审稿意见 **Session confounding**：

> Each video corresponds to a single state, so alert and fatigued samples come from
> different sessions. Session factors such as calibration, posture, lighting, and
> gaze-estimation error may therefore be confounded with the label. It is unclear
> whether calibration is repeated per session.

本实验回答两个问题：

1. **校准是否按 session 重复执行？** —— 脚本按与
   `FatigueGuard_preprocess_batch.find_calibration_files` 完全相同的三级规则
   （subject 级 → state/difficulty 级 → state 级）解析每个
   `(subject, state, difficulty)` 实际使用的校准目录，并在报告中列出命中层级。
   若 alert 与 sleepy 命中**同一**目录，说明校准未按 session 重复，脚本会如实标记
   `shared_calibration=True` 并把该被试剔除出配对统计——**这种情况必须如实写进回复**，
   不能声称做了逐 session 校准。
2. **两次 session 的校准精度是否可比？** —— 对每个 session 独立复算四点校准误差
   （mean±SD），做被试内配对统计检验。

## 1. 方法口径

误差复算完全复现原始校准管线（`gaze_tracking/homtransform.py`）的口径：

- 输入：该 session 的 `results/Calibration.csv`（原始校准记录，`set_x/y/z` 单位 mm）
  与 `STransG/STransG.npy`（该 session 实际用于预处理的屏幕变换矩阵）；
- 对每个校准点 `idx`（四点校准 = 屏幕四角，idx 0..3）：
  - 复现 `_RemoveOutliers` 的去离群掩码：`gaze_x`、`gaze_y` 各按 mean±1σ 过滤后取 AND；
  - 取去离群后样本的 **median gaze 向量**，按 `_getScale` 求与屏幕平面（z=0）的交点，
    经 `STransG` 映射为屏幕坐标（mm）；
  - 单点误差 = 映射点与目标点的 (x,y) 欧氏距离（mm），并按 `stats.txt` 中的
    屏幕物理尺寸换算为 px；
- 每个 session 报告 4 个单点误差的 **均值 ± 标准差**（`mean_err_mm ± sd_err_mm`）。

辅助口径（不作为主指标）：`mean_persample_err_mm` = 每点全部去离群样本逐帧映射误差
的均值再对 4 点平均，反映校准期间的注视抖动；两者结论应一致，若背离说明个别
session 抖动异常，需在回复中说明。

方法学参考（校准精度报告惯例，median-of-samples + 逐点误差）：
Holmqvist et al., *Eye Tracker Data Quality: What It Is and How to Measure It*,
ACM ETRA 2012。

## 2. 统计检验协议

配对设计：同一被试的 alert session vs fatigue（数据树中命名为 `sleepy`）session，
被试内配对；按 `easy`、`hard` 分别检验，另做 `combined`（每被试先对两场景取均值再配对）。

每个条件报告：

| 检验 | 用途 | 文献 |
|---|---|---|
| Shapiro-Wilk（配对差） | 正态性说明 | Shapiro & Wilk 1965, *Biometrika* |
| 配对 t 检验 + Cohen's dz + 差值 95%CI | 主检验"无显著差异" | 经典方法 |
| Wilcoxon 符号秩检验 | 非参数稳健性 | Wilcoxon 1945 |
| **TOST 等价检验** | "无显著差异"≠"等价"；须在预设边界内证明等价 | Schuirmann 1987, *J Pharmacokinet Biopharm*; Lakens 2017, *PeerJ* |

TOST 边界（跑之前定，不要看结果后改）：

- 绝对边界 `--tost_margin_mm`（默认 5 mm）：建议按"小于系统整体注视估计误差一个量级"
  或屏幕宽度的 ~1% 论证；
- 相对边界 `--tost_margin_rel`（默认 0.20，即 alert 组均值的 ±20%）。

**判读规则**（写回复前先看这个）：

- 配对 t 与 Wilcoxon 均 p ≥ 0.05 → 可写"no significant difference"；
- TOST p < 0.05 → 可进一步写"statistically equivalent within ±X mm"；
- 若 TOST 不成立而差异检验也不显著 → 只能写"未发现显著差异（underpowered for
  equivalence）"，不要写"等价"；
- 若差异**显著** → 不要隐瞒；改在回复中论证该差异远小于状态效应量，或补一个
  以校准误差为协变量的敏感性分析。

## 3. 运行方法（在数据服务器上）

依赖主预处理环境的 `numpy / pandas / scipy / matplotlib`，**不需要 TensorFlow、
不需要 GPU**，纯 CPU 秒级~分钟级（只读每 session 一份 Calibration.csv，约 240 行）。

```bash
cd /path/to/FatigueGuard-Preprocess
python scripts/calibration_session_error.py \
    --data_root /root/autodl-tmp/shenxy/Data/DATA \
    --output    /root/autodl-tmp/shenxy/Data/calib_session_check \
    --tost_margin_mm 5 --tost_margin_rel 0.20
```

常用参数：

- `--subjects 01,02,...`：只跑指定被试（先用 2~3 个被试试跑，确认列名/路径都能解析，再全量跑）；
- `--no_figure`：不出图。

## 4. 输出

| 文件 | 内容 |
|---|---|
| `calibration_session_error.csv` | 每行一个 `(subject, state, difficulty)`：命中层级、是否共享校准、四点逐点误差（mm/px）、mean±SD、辅助口径、有效帧数、失败原因 |
| `calibration_session_error_summary.md` | §1 校准来源表（直接回答"是否按 session 重复"）；§2 逐 session 误差表；§3 统计表 + 自动结论；数据质量备注 |
| `calibration_session_paired.png` | alert vs fatigue 配对点线图（每被试一条连线，横杠为组均值） |

回复审稿人时可直接引用 summary 中的：校准层级表（证明 per-session calibration）、
`alert M±SD vs fatigue M±SD`、Δ[95%CI]、配对 t / Wilcoxon p、TOST 结果。

## 5. 已知限制与降级路径

- **L1 共享校准**：若某被试只在 subject 级存在校准目录，则其 alert/sleepy 使用同一
  矩阵，四点误差恒等，不能用于"逐 session 校准精度可比"的论证。脚本会列出这些被试；
  回复中应改述为"该被试校准在两次 session 前统一执行"，并只用其余被试做统计。
  若**所有**被试都共享，本补充实验不成立，需与导师商量替代论证（例如改用逐 session
  jsonl 中 `deviation_px_before_calibrate` 在注视目标帧上的分布做 session 间对比——
  注意那是"任务期注视精度"，与"校准残差"是不同构念，引用时要说清）。
- **L2 缺 Calibration.csv**：老数据树可能只保留了 `STransG/*.npy` 而没有原始校准
  记录，此时无法复算误差；脚本会在 CSV 的 `reason` 列逐条列出。缺失过多时同样走 L1
  的替代路径。
- **L3 校准点数 ≠ 4**：若某 session 的 `StG.npy` 长度 > 4（含边中点/头部转动段），
  脚本只取前 4 个 idx（屏幕四角）保持"四点"口径，并在
  `n_calib_points_total` 列记录实际点数。
- **L4 范围**：本实验只覆盖审稿意见中的 calibration / gaze-estimation error 两项；
  posture、lighting 的 session 混淆不由本实验排除（可另行用 jsonl 中的
  `face_detection_bbox` / head pose 方差做 session 间对比）。
