from numba import jit, prange
import numpy as np
import time


@jit(nopython=True, parallel=True)
def naive_bayes_predict(x_train, y_train, x_test):
    x_train = np.ascontiguousarray(x_train)
    y_train = np.ascontiguousarray(y_train)
    x_test = np.ascontiguousarray(x_test)

    n_samples_train = x_train.shape[0]
    n_samples_test = x_test.shape[0]
    n_features = x_train.shape[1]
    n_classes = len(np.unique(y_train))
    y_predict = np.zeros(n_samples_test)

    # 计算先验概率
    prior = np.zeros(n_classes)
    for i in prange(n_classes):
        prior[i] = np.sum(y_train == i) / n_samples_train

    # 计算条件概率
    likelihood = np.zeros((n_classes, n_features, 2))
    for i in prange(n_classes):
        for j in prange(n_features):
            for k in prange(2):
                likelihood[i, j, k] = np.sum((x_train[y_train == i, j] == k)) / np.sum(y_train == i)

    # 预测
    for i in prange(n_samples_test):
        prob = np.zeros(n_classes)
        for j in prange(n_classes):
            prod_likelihood = 1
            for feature_index in range(n_features):
                prod_likelihood *= likelihood[j, feature_index, x_test[i, feature_index]]
            prob[j] = prod_likelihood * prior[j]
        y_predict[i] = np.argmax(prob)
    return y_predict


@jit(nopython=True, parallel=True)
def generate_data(n_samples):
    X = np.random.rand(n_samples, 2) * 10
    Y = np.zeros(n_samples, dtype=np.int8) - 1
    for i in prange(n_samples):
        if 0 < X[i, 0] < 3 and 0 < X[i, 1] < 3:
            Y[i] = 0
        elif 0 < X[i, 0] < 3 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 1
        elif 0 < X[i, 0] < 3 and 7 < X[i, 1] < 10:
            Y[i] = 2
        elif 3.5 < X[i, 0] < 6.5 and 0 < X[i, 1] < 3:
            Y[i] = 3
        elif 3.5 < X[i, 0] < 6.5 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 4
        elif 3.5 < X[i, 0] < 6.5 and 7 < X[i, 1] < 10:
            Y[i] = 5
        elif 7 < X[i, 0] < 10 and 0 < X[i, 1] < 3:
            Y[i] = 6
        elif 7 < X[i, 0] < 10 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 7
        elif 7 < X[i, 0] < 10 and 7 < X[i, 1] < 10:
            Y[i] = 8
    valid = Y >= 0
    X = X[valid]
    Y = Y[valid]
    return X, Y


@jit(nopython=True, parallel=True)
def threshold(x):
    x_threshold = np.zeros((x.shape[0], 9), dtype=np.int8)
    for i in prange(x.shape[0]):
        if 0 < x[i, 0] < 3 and 0 < x[i, 1] < 3:
            x_threshold[i][0] = 1
        elif 0 < x[i, 0] < 3 and 3.5 < x[i, 1] < 6.5:
            x_threshold[i][1] = 1
        elif 0 < x[i, 0] < 3 and 7 < x[i, 1] < 10:
            x_threshold[i][2] = 1
        elif 3.5 < x[i, 0] < 6.5 and 0 < x[i, 1] < 3:
            x_threshold[i][3] = 1
        elif 3.5 < x[i, 0] < 6.5 and 3.5 < x[i, 1] < 6.5:
            x_threshold[i][4] = 1
        elif 3.5 < x[i, 0] < 6.5 and 7 < x[i, 1] < 10:
            x_threshold[i][5] = 1
        elif 7 < x[i, 0] < 10 and 0 < x[i, 1] < 3:
            x_threshold[i][6] = 1
        elif 7 < x[i, 0] < 10 and 3.5 < x[i, 1] < 6.5:
            x_threshold[i][7] = 1
        elif 7 < x[i, 0] < 10 and 7 < x[i, 1] < 10:
            x_threshold[i][8] = 1

    return x_threshold


@jit(nopython=True, parallel=True)
def add_noise(X, Y, n_noise):
    """
    向数据集添加噪声点。

    Parameters:
    - X: 原始数据的特征数组。
    - Y: 原始数据的标签数组。
    - n_noise: 要添加的噪声点数量。
    - n_classes: 类别总数，默认为9。

    Returns:
    - X_noise: 包含噪声点的新特征数组。
    - Y_noise: 包含噪声点的新标签数组。
    """
    X_noise = np.random.rand(n_noise, 2) * 10
    Y_noise = np.random.randint(0, 9, size=n_noise)

    # 将噪声点合并到原始数据集中
    X_combined = np.vstack((X, X_noise))
    Y_combined = np.concatenate((Y, Y_noise))

    return X_combined, Y_combined
