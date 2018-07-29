# -*- coding: utf-8 -*-
# @Time    : 7/29/2018 10:08 PM
# @Author  : DonaldY

# 1. 定义数据集
x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]


# 2. 实现 n 次多项式拟合
import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot as plt

# 根据公式，定义 n 次多项式函数
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差函数（观测值与拟合值之间的差距）
def err_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# n 次多项式拟合
def n_poly(n):
    p_init = np.random.randn(n) # 生成 n 个随机数
    parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))
    return parameters[0]

n_poly(3)


# 3. 绘制出 4，5，6，7, 8, 9 次多项式的拟合图像
# 绘制拟合图像时需要的临时点
x_temp = np.linspace(0, 80, 10000)

# 绘制子图
fig, axes = plt.subplots(2, 3, figsize=(15,10))

axes[0,0].plot(x_temp, fit_func(n_poly(4), x_temp), 'r')
axes[0,0].scatter(x, y)
axes[0,0].set_title("m = 4")

axes[0,1].plot(x_temp, fit_func(n_poly(5), x_temp), 'r')
axes[0,1].scatter(x, y)
axes[0,1].set_title("m = 5")

axes[0,2].plot(x_temp, fit_func(n_poly(6), x_temp), 'r')
axes[0,2].scatter(x, y)
axes[0,2].set_title("m = 6")

axes[1,0].plot(x_temp, fit_func(n_poly(7), x_temp), 'r')
axes[1,0].scatter(x, y)
axes[1,0].set_title("m = 7")

axes[1,1].plot(x_temp, fit_func(n_poly(8), x_temp), 'r')
axes[1,1].scatter(x, y)
axes[1,1].set_title("m = 8")

axes[1,2].plot(x_temp, fit_func(n_poly(9), x_temp), 'r')
axes[1,2].scatter(x, y)
axes[1,2].set_title("m = 9")