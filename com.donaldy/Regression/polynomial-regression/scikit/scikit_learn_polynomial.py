# -*- coding: utf-8 -*-
# @Time    : 7/29/2018 10:29 PM
# @Author  : DonaldY

# 1. 使用 PolynomialFeatures 自动生成特征矩阵
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = [2, -1, 3]
X_reshape = np.array(X).reshape(len(X), 1) # 转换为列向量
PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_reshape)


# 2. 使用 sklearn 得到 2 次多项式回归特征矩阵
x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]
y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]

x = np.array(x).reshape(len(x), 1) # 转换为列向量
y = np.array(y).reshape(len(y), 1)


poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_x = poly_features.fit_transform(x)

poly_x


# 3. 转换为线性回归预测
from sklearn.linear_model import LinearRegression

# 定义线性回归模型
model = LinearRegression()
model.fit(poly_x, y) # 训练

# 得到模型拟合参数
model.intercept_, model.coef_


# 4. 绘制拟合图像
from matplotlib import pyplot as plt

# 绘制拟合图像时需要的临时点
x_temp = np.linspace(0, 80, 10000)

x_temp = np.array(x_temp).reshape(len(x_temp),1)
poly_x_temp = poly_features.fit_transform(x_temp)

plt.plot(x_temp, model.predict(poly_x_temp), 'r')
plt.scatter(x, y)
