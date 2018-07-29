# -*- coding: utf-8 -*-
# @Time    : 7/29/2018 10:00 AM
# @Author  : DonaldY

"""scikit-learn 线性回归拟合
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import scipy

# 1. 定义数据集
x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

# 2. 定义线性回归模型
model = LinearRegression()
# 训练, reshape 操作把数据处理成 fit 能接受的形状
model.fit(x.reshape(len(x), 1), y)

# 3.得到模型拟合参数
model.intercept_, model.coef_

# 4. 预测
model.predict([[150]])