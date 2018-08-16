# -*- coding: utf-8 -*-
# @Time    : 8/16/2018 3:56 PM
# @Author  : DonaldY

"""示例数据
"""
scores=[[1],[1],[2],[2],[3],[3],[3],[4],[4],[5],[6],[6],[7],[7],[8],[8],[8],[9],[9],[10]]
passed= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]

"""示例数据绘图
"""
from matplotlib import pyplot as plt

plt.scatter(scores, passed, color='r')
plt.xlabel("scores")
plt.ylabel("passed")

plt.show();


"""线性回归拟合
"""
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(scores, passed)
model.coef_, model.intercept_


"""拟合后绘图
"""
import numpy as np

x = np.linspace(-2,12,100)

plt.plot(x, model.coef_[0] * x + model.intercept_)
plt.scatter(scores, passed, color='r')
plt.xlabel("scores")
plt.ylabel("passed")


"""Sigmoid 分布函数图像
"""
z = np.linspace(-12, 12, 100) # 生成等间距 x 值方便绘图
sigmoid = 1 / (1 + np.exp(-z))
plt.plot(z, sigmoid)
plt.xlabel("z")
plt.ylabel("y")


"""逻辑回归模型
"""
def sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

"""梯度计算
"""
def gradient(X, h, y):
    gradient = np.dot(X.T, (h - y)) / y.shape[0]
    return gradient

