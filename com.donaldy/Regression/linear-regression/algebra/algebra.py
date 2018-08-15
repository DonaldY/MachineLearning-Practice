# -*- coding: utf-8 -*-
# @Time    : 7/28/2018 5:23 PM
# @Author  : DonaldY

# 1. 导入数据集
import inline as inline
import matplotlib
import numpy as np

x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

# 2. 显示数据集在坐标上
from matplotlib import pyplot as plt
# %matplotlib inline

plt.scatter(x, y)
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# 3. 定义拟合直线
def f(x, b, w1):
    y = b + w1 * x
    return y

# 3.2 平方损失函数
def square_loss(x, y, w0, w1):
    loss = sum(np.square(y - (w0 + w1*x)))
    return loss

# 3.3 平方损失函数最小时对应的w参数值 ，b
def w_calculator(x, y):
    n = len(x)
    w1 = (n*sum(x*y) - sum(x)*sum(y))/(n*sum(x*x) - sum(x)*sum(x))
    w0 = (sum(x*x)*sum(y) - sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x))
    return w0, w1

# 3.4 代入计算
w_calculator(x, y)

b = w_calculator(x, y)[0]
w = w_calculator(x, y)[1]

square_loss(x, y, b, w)

# 4. 绘制图像
x_temp = np.linspace(50,120,100) # 绘制直线生成的临时点
plt.scatter(x, y)
plt.plot(x_temp, x_temp*w + b, 'r')
plt.show()

# 5. 如果手中有一套 150 平米的房产想售卖，获取预估报价：
f(150, b, w)
