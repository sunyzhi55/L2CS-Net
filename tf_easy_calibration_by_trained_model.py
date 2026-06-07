## 环境 87 ： tf_cali
########################### easy #############################
import os
import tensorflow as tf
import numpy as np

# -------------------- 复用训练时的模型结构（关键！确保能加载预训练参数） --------------------
class ImprovedCalibrationNet:
    """改进的MLP校准网络（带L2正则化，与训练时完全一致）"""

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 2])  # 输入：demo预测值（pitch, yaw）
        self.y_true = tf.placeholder(tf.float32, [None, 2])  # 标签占位符（推理时喂dummy值）

        # 网络结构：2层隐藏层（10神经元/层）+ L2正则化（与训练时一致）
        self.hidden1 = tf.layers.dense(
            self.x, units=10, activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
        )
        self.hidden2 = tf.layers.dense(
            self.hidden1, units=10, activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
        )
        self.y_pred = tf.layers.dense(self.hidden2, units=2)  # 输出：校准后的（pitch, yaw）

        # 损失函数（推理时不使用，但定义需与训练一致）
        self.loss = tf.reduce_mean(tf.square(self.y_pred - self.y_true)) + tf.losses.get_regularization_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

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


# -------------------- 加载预训练模型并校准新数据 --------------------
def load_model_and_calibrate(model_save_path, new_demo_data):
    """
    加载预训练模型，对新的demo预测值进行校准
    :param model_save_path: 预训练模型路径（如"./model2/gaze_calibration_model.ckpt"）
    :param new_demo_data: 待校准的demo预测值（N×2 numpy数组，每行是[pitch, yaw]）
    :return: 校准后的结果（N×2 numpy数组）
    """
    # 1. 初始化模型（必须与训练时结构一致）
    model = ImprovedCalibrationNet()

    with tf.Session() as sess:
        # 2. 加载预训练参数
        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)
        print(f"✅ 成功加载预训练模型：{model_save_path}")

        # 3. 推理校准（喂dummy标签满足placeholder要求）
        calibrated_pred = sess.run(
            model.y_pred,
            feed_dict={
                model.x: new_demo_data,
                model.y_true: np.zeros_like(new_demo_data)  # dummy值，无实际意义
            }
        )
    return calibrated_pred


# -------------------- 计算平均欧式距离误差 --------------------
def compute_euclidean_error(pred, label):
    """
    计算预测值与真实标签的平均欧式距离
    :param pred: 预测值（N×2）
    :param label: 真实标签（N×2）
    :return: 平均欧式误差
    """
    num_samples = pred.shape[0]
    # 逐样本计算欧式距离（L2范数），再求平均
    all_error = np.linalg.norm(pred - label, axis=1)
    total_error = np.sum(np.linalg.norm(pred - label, axis=1))
    return all_error,total_error / num_samples


# -------------------- 主程序：加载数据→校准→计算误差→保存结果 --------------------
if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    MODEL_NAME = "gaze_calibration_model"  #训练时保存的模型名称
    MODEL_SAVE_PATH = r"./tf_calibrate_model/gaze_calibration_model.ckpt"  # 预训练模型路径（.ckpt文件）
    OUT_PUT_PATH = r"./cali_results"  # 输出结果路径

    # 加载数据路径（替换为你的实际路径！）
    DATA_DIR = r"./data"
    DEMO_DATA_PATH = os.path.join(DATA_DIR, "predict.txt")  # 待校准的demo预测值（.txt）
    REAL_LABEL_PATH = os.path.join(DATA_DIR, "gt.txt")  # 真实标签（.txt）

    # 结果保存路径（改为txt格式）
    CALIBRATED_RESULT_PATH = os.path.join(OUT_PUT_PATH, "calibrated_results.txt")  # 校准结果

    ERROR_AFTER_PATH_all = os.path.join(OUT_PUT_PATH, "all_errors.txt") ###距离误差序列
    ERROR_BEFORE_PATH = os.path.join(OUT_PUT_PATH, "error_before.txt")  # 校准前误差
    ERROR_AFTER_PATH = os.path.join(OUT_PUT_PATH, "error_after.txt")  # 校准后误差

    # -------------------- 1. 加载数据 --------------------
    # 加载待校准的demo预测值（.txt文件，需是N×2格式）一般预测值中间分隔符是 逗号
    new_demo_data = np.loadtxt(DEMO_DATA_PATH, delimiter=',')  # 新增delimiter=','
    print(f"✅ 加载待校准demo数据：{DEMO_DATA_PATH}，样本数：{new_demo_data.shape[0]}")

    # 加载真实标签（必须与demo数据同样本数、同维度） 真实标签中间分隔符是空格
    real_gaze_label = np.loadtxt(REAL_LABEL_PATH, delimiter=' ')  # 新增delimiter=','
    print(f"✅ 加载真实标签数据：{REAL_LABEL_PATH}，样本数：{real_gaze_label.shape[0]}")

    # -------------------- 2. 加载模型并校准 --------------------
    calibrated_results = load_model_and_calibrate(MODEL_SAVE_PATH, new_demo_data)
    print(f"✅ 校准完成，结果形状：{calibrated_results.shape}（每行是校准后的[x, y]）")

    # -------------------- 3. 计算校准前后的欧式误差 --------------------
    # 校准前：原始demo数据 vs 真实标签
    all_error1,error_before = compute_euclidean_error(new_demo_data, real_gaze_label)
    # 校准后：模型输出 vs 真实标签
    all_error2,error_after = compute_euclidean_error(calibrated_results, real_gaze_label)

    print(f"📊 校准前平均欧式误差: {error_before:.6f}")
    print(f"📊 校准后平均欧式误差: {error_after:.6f}")

    # -------------------- 4. 保存结果（txt格式） --------------------
    # 确保输出目录存在
    os.makedirs(OUT_PUT_PATH, exist_ok=True)

    # 保存校准结果为txt格式（逗号分隔）
    save_as_txt(calibrated_results, CALIBRATED_RESULT_PATH)
    save_as_txt(all_error2, ERROR_AFTER_PATH_all)

    # 保存误差值为txt格式
    save_error_as_txt(error_before, ERROR_BEFORE_PATH)
    save_error_as_txt(error_after, ERROR_AFTER_PATH)

    print(f"✅ 校准结果已保存到：{CALIBRATED_RESULT_PATH}")
    print(f"✅ 误差已保存到：{ERROR_BEFORE_PATH}、{ERROR_AFTER_PATH}")






# ##### 视线校准
# 用于校准l2cs的注视估计值 测试使用校准的预训练模型
#
# #####模型 MLP
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# #####输入弧度 转换成为3d值
# # gaze:以pitch（仰角）和yaw（方位角）表示的球坐标（pitch, yaw）
# def gazeto3d(gaze):
#     gaze_gt = np.zeros([3])
#     gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
#     gaze_gt[1] = -np.sin(gaze[1])
#     gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
#     return gaze_gt
#
# ####计算地面真实凝视和估计值之间的夹角
# #计算两个三维向量（gaze 和 label）之间的夹角，并以度数为单位返回这个角度
# def angular(gaze, label):
#     total = np.sum(gaze * label)
#     return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi
#
# ####计算误差 前后
# def compute_avg_error(pitch,yaw,pitch1,yaw1,label_p,label_y):
#     pitch_predicted = pitch##校准前
#     yaw_predicted = yaw
#     pitch_predicted1 = pitch1###校准后
#     yaw_predicted1 = yaw1
#     label_pitch = label_p###真实凝视
#     label_yaw = label_y
#     avg_error = .0
#     avg_error1 = .0
#     for p, y, pl, yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
#         avg_error += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))
#     avg_error /= num_samples
#     print('校准前的估计值与真实值之间的平均误差: {:.6f}'.format(avg_error))
#
#     for p, y, pl, yl in zip(pitch_predicted1, yaw_predicted1, label_pitch, label_yaw):
#         avg_error1 += angular(gazeto3d([p, y]), gazeto3d([pl, yl]))
#     avg_error1 /= num_samples
#     print('校准后的估计值与真实值之间的平均误差: {:.6f}'.format(avg_error1))
#     return avg_error,avg_error1
#
# # 定义改进的校准网络模型 多层感知机模型
# class ImprovedCalibrationNet:
#     def __init__(self, scope_name):
#         with tf.variable_scope(scope_name):
#             self.x = tf.placeholder(tf.float32, [None, 2])
#             self.y_true = tf.placeholder(tf.float32, [None, 2])
#
#             # 使用更深的网络结构 防止过拟合 添加L2正则化
#             self.hidden1 = tf.layers.dense(self.x, units=10, activation=tf.nn.relu,
#                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
#             self.hidden2 = tf.layers.dense(self.hidden1, units=10, activation=tf.nn.relu,
#                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
#             self.y_pred = tf.layers.dense(self.hidden2, units=2)
#
#             # 损失函数
#             self.loss = tf.reduce_mean(tf.square(self.y_pred - self.y_true)) + tf.losses.get_regularization_loss() # 均方误差损失 添加L2正则化
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
#
#
# def Train(load,estimated_value_Test,fold_index):
#     # 初始化校准网络模型
#     model = ImprovedCalibrationNet(scope_name='fold_{}'.format(fold_index))####tf.variable_scope 来创建独立的命名空间可以避免变量名称冲突的问题
#     loss_values = []  # 初始化损失值列表
#     # 开始 TensorFlow 会话
#     with tf.Session() as sess:
#         if (load == True):
#             # 恢复模型参数
#             saver = tf.train.Saver()
#             saver.restore(sess, 'E:/ML1/tensorflow/一阶段_实验数据/test_100_train_28/model2/calibration_model_{}.ckpt'.format(fold_index))
#             print("模型参数{}已加载！".format(fold_index+1))
#             # # 打印模型参数
#             # for var in tf.trainable_variables():  ############测试训练时使用的模型参数是否正常保存
#             #     print(var.name, sess.run(var))
#         else:
#             sess.run(tf.global_variables_initializer())
#
#         calibrated_value = sess.run(model.y_pred, feed_dict={model.x: estimated_value_Test})
#         return  loss_values,calibrated_value
#
# ################-main-###############
# load = True  # 设置为 True 以加载模型
# ii = 3  # 指定加载第四折的模型
# ###将txt文件数据转化成为.npy 文件
#
# ###测试集
# ##alert
# # new_estimated_value_Test = np.loadtxt(r'E:\ML1\tensorflow\pythonProject2\results(second)\hard\15\alert\demo_gaze_directions.txt')
#
# # new_true_value_Test = np.loadtxt(r'E:\ML1\tensorflow\pythonProject2\results(second)\easy\15\alert\real_gaze_directions.txt')
#
# ####sleepy
# new_estimated_value_Test = np.loadtxt(r'E:\ML1\tensorflow\pythonProject2\results(second)\hard\15\sleepy\demo_gaze_directions.txt')
#
#
# ####调用Train()
# loss_values, calibrated_value = Train(load=load,  estimated_value_Test=new_estimated_value_Test, fold_index=ii)
#
# ####计算角度误差进行评估#####
# ### 载入l2cs模型预估值（新测试集）
# pitch_predicted = new_estimated_value_Test[0]
# yaw_predicted = new_estimated_value_Test[1]
#
# ###载入地面真实凝视（新测试集）
# # label_pitch = new_true_value_Test[0]
# # label_yaw = new_true_value_Test[1]
#
# ###获取校准预估值
# pitch_predicted1 = calibrated_value[0]
# yaw_predicted1 = calibrated_value[1]
#
# num_samples = len(pitch_predicted)  #### 获取样本数
#
# ####计算校准前后的角度误差 compute_avg_error(pitch,yaw,pitch1,yaw1,label_p,label_y)
# # avg_error, avg_error1 = compute_avg_error(pitch=pitch_predicted, yaw=yaw_predicted, pitch1=pitch_predicted1, yaw1=yaw_predicted1,
# #                                           label_p=label_pitch, label_y=label_yaw)
#
# if load == True:  #####保存数据 注意文件夹gaze_model需要自己提前建好
#     # np.save(open('Random_gaze_model/loss_new%d.npy' % ii, 'wb'), loss_values)
#     #alert
#     # np.save(open(r'E:\ML1\tensorflow\pythonProject2\calibrated_value\hard\15\alert\calibrated_value.npy', 'wb'), calibrated_value)
#     #sleepy
#     np.save(open(r'E:\ML1\tensorflow\pythonProject2\calibrated_value\hard\15\sleepy\calibrated_value.npy', 'wb'),calibrated_value)
#
#
#     # np.save(open(r'E:\ML1\tensorflow\pythonProject2\calibrated_value\easy\15\sleepy\calibrated_value.npy', 'wb'),calibrated_value)
#     # np.save(open(r'E:\ML1\tensorflow\pythonProject2\calibrated_value\easy\15\sleepy\avg_error.npy', 'wb'), avg_error)
#     # np.save(open(r'E:\ML1\tensorflow\pythonProject2\calibrated_value\easy\15\sleepy\avg_error1.npy', 'wb'), avg_error1)