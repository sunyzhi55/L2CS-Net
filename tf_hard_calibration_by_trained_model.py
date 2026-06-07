####上面处理的是屏幕上的目标点（hard/easy） 第三段处理的是注视视线
################################ hard ############################
import os
import tensorflow as tf
import numpy as np

########加载demo
def load_coordinates(file_path):
    """读取坐标文件，处理可能的格式问题"""
    coordinates = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                # 处理逗号分隔或空格分隔
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    coordinates.append([x, y])
            except ValueError:
                continue
    return np.array(coordinates)


# -------------------- 工具函数：计算欧氏距离 --------------------
def pixel_distance(est_coord: np.ndarray, true_coord: np.ndarray) -> float:
    """计算两点之间的欧几里得距离（像素/坐标距离）"""
    return np.sqrt((est_coord[0] - true_coord[0]) ** 2 + (est_coord[1] - true_coord[1]) ** 2)


# -------------------- 保存为txt格式的函数 --------------------
def save_as_txt(data, file_path, delimiter=','):
    """
    将数据保存为txt文件
    :param data: 要保存的数据（numpy数组）
    :param file_path: 保存路径
    :param delimiter: 分隔符，默认为逗号
    """
    np.savetxt(file_path, data, delimiter=delimiter, fmt='%.6f')
    print(f"✅ 数据已保存为txt格式：{file_path}")

def save_error_as_txt(error_value, file_path):
    """
    保存单个误差值为txt文件
    :param error_value: 误差值
    :param file_path: 保存路径
    """
    with open(file_path, 'w') as f:
        f.write(f"{error_value:.6f}")
    print(f"✅ 误差值已保存为txt格式：{file_path}")

# -------------------- 复用训练时的模型结构 --------------------
class ImprovedCalibrationNet:
    """改进的MLP校准网络（与训练时完全一致，确保加载预训练参数）"""

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 2])  # 输入：demo预测值（pitch, yaw）
        self.y_true = tf.placeholder(tf.float32, [None, 2])  # 标签占位符（推理喂dummy值）

        # 网络结构：2层隐藏层（10神经元/层）+ L2正则化
        self.hidden1 = tf.layers.dense(
            self.x, units=10, activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
        )
        self.hidden2 = tf.layers.dense(
            self.hidden1, units=10, activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
        )
        self.y_pred = tf.layers.dense(self.hidden2, units=2)  # 输出：校准后的（pitch, yaw）

        # 损失函数（推理不使用，但定义需与训练一致）
        self.loss = tf.reduce_mean(tf.square(self.y_pred - self.y_true)) + tf.losses.get_regularization_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)


# -------------------- 加载模型并校准 --------------------
def load_model_and_calibrate(model_save_path, new_demo_data):
    """加载预训练模型，校准新的demo预测值"""
    model = ImprovedCalibrationNet()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)
        print(f"✅ 成功加载预训练模型：{model_save_path}")

        # 推理（喂dummy标签满足placeholder要求）
        calibrated_pred = sess.run(
            model.y_pred,
            feed_dict={
                model.x: new_demo_data,
                model.y_true: np.zeros_like(new_demo_data)  # dummy值，无实际意义
            }
        )
    return calibrated_pred

# -------------------- 计算平均欧式误差 --------------------
def compute_euclidean_error(pred: np.ndarray, label: np.ndarray) -> float:
    """计算预测值与真实标签的平均欧式距离"""
    num_samples = pred.shape[0]
    all_error = np.linalg.norm(pred - label, axis=1)
    total_error = np.sum(np.linalg.norm(pred - label, axis=1))  # 逐样本计算距离
    return all_error,total_error / num_samples

# -------------------- 主程序：困难任务校准流程 --------------------
if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    MODEL_NAME = "gaze_calibration_model"
    MODEL_SAVE_PATH = rf"./tf_calibrate_model/{MODEL_NAME}.ckpt"  #预训练模型路径
    OUT_PUT_PATH = r"E:\ML1\tensorflow\pythonProject2\Cali_result(20)\hard\19\alert" #输出目录（区分困难任务）

    # 数据路径（替换为实际路径！）
    DATA_DIR1 = r"E:\ML1\tensorflow\pythonProject2\results(DATA2)\hard\19\alert"
    DATA_DIR2 = r"E:\ML1\tensorflow\pythonProject2\DATA2\19\alert\hard"
    DEMO_DATA_PATH = os.path.join(DATA_DIR1, "demo_gaze_directions.txt")  # demo预测值（逗号分隔）
    REAL_LABEL_PATH = os.path.join(DATA_DIR2, "Gaze_hard_centers.npy")  # 真实目标点（.npy，每个元素是图片的真实点列表）

    # 结果保存路径（改为txt格式）
    CALIBRATED_RES_PATH = os.path.join(OUT_PUT_PATH, "calibrated_results.txt")
    ERROR_BEFORE_PATH = os.path.join(OUT_PUT_PATH, "error_before.txt")  # 校准前误差（vs 固定真实点）
    ERROR_AFTER_PATH = os.path.join(OUT_PUT_PATH, "error_after.txt")  # 校准后误差（vs 固定真实点）

    ERROR_AFTER_PATH_ALL = os.path.join(OUT_PUT_PATH, "all_errors.txt")
    FIXED_TRUE_PATH = os.path.join(OUT_PUT_PATH, "fixed_true_points.txt")  # 保存固定真实点（可选）

    # -------------------- 1. 加载基础数据 --------------------
    # 1.1 加载demo预测值（逗号分隔的txt文件）
    try:
        new_demo_data = load_coordinates(DEMO_DATA_PATH)
        print(f"✅ 加载demo数据：{DEMO_DATA_PATH}，样本数：{new_demo_data.shape[0]}")
    except Exception as e:
        print(f"❌ 加载demo数据失败：{e}")
        exit()

    # 1.2 加载真实目标点列表（.npy文件，每个元素是图片的真实点数组）
    try:
        gaze_list = np.load(REAL_LABEL_PATH, allow_pickle=True)
        gaze_list = gaze_list[:-1]
        print(f"✅ 加载真实目标点：{REAL_LABEL_PATH}，图片数量：{len(gaze_list)}")
    except Exception as e:
        print(f"❌ 加载真实目标点失败：{e}")
        exit()

    # -------------------- 2. 为每个demo点匹配固定真实点 --------------------
    """核心逻辑：每个demo点找到其所属图片中距离最小的真实点，作为专属真实点"""
    num_images = len(gaze_list)  # 总图片数
    total_demo_points = new_demo_data.shape[0]  # demo点总数

    # 假设前49张图片每张180个demo点，最后一张图片有剩余点（根据实际情况调整）
    demo_per_image = 180
    total_expected = (num_images - 1) * demo_per_image
    print(f"前{num_images - 1}张图片期望点数：{total_expected}")
    last_image_rows = total_demo_points - total_expected
    print(f"最后一张图片点数：{last_image_rows}")

    if last_image_rows <= 0:
        print(f"❌ demo点数量不足，无法分割为{num_images}张图片")
        exit()

    avg_error_all_min = []
    fixed_true_points = np.zeros_like(new_demo_data)  # 存储每个估计点对应的最小误差真实点

    # 2.1 按图片分割demo数据
    for image_index in range(num_images):
        image_error = []  # 保存当前图片所有估计点对应的最小距离误差
        current_rows = demo_per_image if image_index < num_images - 1 else last_image_rows
        start_row = image_index * demo_per_image
        end_row = start_row + current_rows
        current_est_coords = new_demo_data[start_row:end_row]  # 当前图片的估计点坐标
        current_gaze_data = gaze_list[image_index]['centers']  # 当前图片的真实注视点

        # 逐点处理当前图片的估计值
        for idx, est_coord in enumerate(current_est_coords):
            min_error = float('inf')
            closest_true_point = None

            # 计算当前估计点与所有真实点的距离
            for true_coord in current_gaze_data:
                error = pixel_distance(est_coord, true_coord)
                if error < min_error:
                    min_error = error
                    closest_true_point = true_coord  # 更新最近真实点

            # 保存最小误差对应的真实点
            if closest_true_point is not None:
                # 计算在fixed_true_points中的全局位置
                global_row = start_row + idx
                fixed_true_points[global_row] = closest_true_point
                image_error.append(min_error)
        avg_error_all_min.append(image_error)  # 添加当前图片的误差列表

    # 保存固定真实点为txt格式
    save_as_txt(fixed_true_points, FIXED_TRUE_PATH)
    print(f"✅ 已保存固定真实点：{FIXED_TRUE_PATH}")

    # -------------------- 3. 加载模型并校准demo点 --------------------
    try:
        calibrated_results = load_model_and_calibrate(MODEL_SAVE_PATH, new_demo_data)
        print(f"✅ 校准完成，结果形状：{calibrated_results.shape}")
    except Exception as e:
        print(f"❌ 模型校准失败：{e}")
        exit()

    # -------------------- 4. 计算校准前后的误差（基于固定真实点） --------------------
    all_error1,error_before = compute_euclidean_error(new_demo_data, fixed_true_points)  # 校准前：demo点 vs 固定真实点
    all_error2,error_after = compute_euclidean_error(calibrated_results, fixed_true_points)  # 校准后：校准点 vs 固定真实点

    print(f"📊 校准前平均误差（vs 固定真实点）: {error_before:.6f}")
    print(f"📊 校准后平均误差（vs 固定真实点）: {error_after:.6f}")

    # -------------------- 5. 保存结果（txt格式） --------------------
    os.makedirs(OUT_PUT_PATH, exist_ok=True)

    # 保存校准结果为txt格式（逗号分隔）
    save_as_txt(calibrated_results, CALIBRATED_RES_PATH)

    save_as_txt(all_error2,ERROR_AFTER_PATH_ALL)

    # 保存误差值为txt格式
    save_error_as_txt(error_before, ERROR_BEFORE_PATH)
    save_error_as_txt(error_after, ERROR_AFTER_PATH)

    print(f"✅ 校准结果已保存：{CALIBRATED_RES_PATH}")
    print(f"✅ 误差已保存：{ERROR_BEFORE_PATH}、{ERROR_AFTER_PATH}")
    print("🎉 困难任务校准完成！")

