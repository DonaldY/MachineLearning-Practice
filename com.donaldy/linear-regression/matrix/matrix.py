# -*- coding: utf-8 -*-
# @Time    : 7/28/2018 5:58 PM
# @Author  : DonaldY

import numpy as np

# 1. 定义数据集
# 针对原 x 数据添加截距项系数 1
x = np.matrix([[1, 56], [1, 72], [1, 69], [1, 88], [1, 102], [1, 86], [1, 76], [1, 79], [1, 94], [1, 74]])
y = np.matrix([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

# 2. 定义 求解函数
def w_matrix(x, y):
    w = (x.T * x).I * x.T * y
    return w


w_matrix(x, y.reshape(10, 1))
