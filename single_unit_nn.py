import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from data import load_dataset
from sigmod import basic_sigmoid

train_x, train_y, test_x, test_y, classes = load_dataset()

# train_x的shape (209, 64, 64, 3)
# train_y的shape (1, 209)
# test_x的shape (50, 64, 64, 3)
# test_y的shape (1, 50)

# 转换形状
train_x = train_x.reshape((train_x.shape[0], -1)).T
test_x = test_x.reshape((test_x.shape[0], -1)).T


# train_x的shape (12288, 209)
# test_x的shape (12288, 50)

# 初始化
def initialize_with_zeros(shape):
    w = np.zeros((shape, 1))
    b = 0
    return w, b


# 前向和反向传播
def propagate(w, b, X, y):
    """
    w,b,x,y: 网路参数和数据
    :return: 损失cost、参数w的梯度dw、参数b的梯度db
    """
    m = X.shape[1]
    # 前向传播
    # w (n, 1) x(n, m) b
    A = basic_sigmoid(np.dot(w.T, X) + b)
    # 计算损失（逻辑回归的损失函数）
    cost = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

    # 反向传播
    dz = A - y
    dw = 1 / m * np.dot(X, dz.T)
    db = 1 / m * np.sum(dz)

    cost = np.squeeze(cost)

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


# 优化过程
def optimize(w, b, X, y, num_iterations, learning_rate):
    """
    :param w: 权重
    :param b: 偏置
    :param X: 特征
    :param y: 目标值
    :param num_iterations: 总迭代次数
    :param learning_rate: 学习率
    :return:
    params: 更新后的参数字典
    grads: 梯度
    costs: 损失结果
    """
    costs = []
    dw, db = None, None
    for i in range(num_iterations):
        # 梯度更新计算
        grads, cost = propagate(w, b, X, y)
        # 取出两个部分参数的梯度
        dw = grads["dw"]
        db = grads["db"]
        # 按照梯度下降的公式去计算
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            print("损失结果：%i: %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return params, grads, costs


# 预测函数
def predict(w, b, X):
    '''
    :param w:
    :param b:
    :param X:
    :return: 预测结果
    '''

    m = X.shape[1]
    y_pred = np.zeros((1, m))
    w = w.reshape((X.shape[0], m))

    # 计算结果
    A = basic_sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        if A[0, i] > 0.5:
            y_pred[0, i] = 1
        else:
            y_pred[0, i] = 0

    return y_pred


def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5):
    # 初始化参数
    w, b = initialize_with_zeros(x_train.shape[0])
    # 梯度下降
    params, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate)

    # 获取训练的参数
    w = params['w']
    b = params['b']

    y_predictions_train = predict(w, b, x_train)
    y_predictions_test = predict(w, b, x_test)

    # 打印准确率
    print("训练集准确率：{}".format(100 - np.mean(np.abs(y_predictions_train - y_train)) * 100))
    print("测试集准确率：{}".format(100 - np.mean(np.abs(y_predictions_test - y_test)) * 100))

    d = {
        "costs": costs,
        "y_predictions_train": y_predictions_train,
        "y_predictions_test": y_predictions_test,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    return d
