# -*- coding: utf-8 -*-
# @Time    : 8/16/2018 5:53 PM
# @Author  : DonaldY

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
# !wget http://labfile.oss.aliyuncs.com/courses/1081/course-8-data.csv
df = pd.read_csv("course-8-data.csv", header=0) # 加载数据集
df.head()

x = df[['X0','X1']].values
y = df['Y'].values

model = LogisticRegression(tol=0.001, max_iter=10000) # 设置一样的学习率和迭代次数
model.fit(x, y)
model.coef_, model.intercept_

"""将上方得到的结果绘制成图
"""


plt.figure(figsize=(10, 6))
plt.scatter(df['X0'],df['X1'], c=df['Y'])

x1_min, x1_max = df['X0'].min(), df['X0'].max(),
x2_min, x2_max = df['X1'].min(), df['X1'].max(),

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]

probs = (np.dot(grid, model.coef_.T) + model.intercept_).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red');

model.score(x, y)