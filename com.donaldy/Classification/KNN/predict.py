# -*- coding: utf-8 -*-
# @Time    : 8/20/2018 10:46 AM
# @Author  : DonaldY

# 下载数据集
# !wget http://labfile.oss.aliyuncs.com/courses/1081/course-9-syringa.csv

"""加载数据集
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

lilac_data = pd.read_csv('course-9-syringa.csv')
lilac_data.head()  # 预览前 5 行

"""绘制丁香花特征子图
"""
fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # 构建生成 2*3 的画布，2 行 3 列
fig.subplots_adjust(hspace=0.3, wspace=0.2)  # 定义每个画布内的行间隔和高间隔
axes[0, 0].set_xlabel("sepal_length")  # 定义 x 轴坐标值
axes[0, 0].set_ylabel("sepal_width")  # 定义 y 轴坐标值
axes[0, 0].scatter(lilac_data.sepal_length[:50],
                   lilac_data.sepal_width[:50], c="b")
axes[0, 0].scatter(lilac_data.sepal_length[50:100],
                   lilac_data.sepal_width[50:100], c="g")
axes[0, 0].scatter(lilac_data.sepal_length[100:],
                   lilac_data.sepal_width[100:], c="r")
axes[0, 0].legend(["daphne", "syringa", "willow"], loc=2)  # 定义示例

axes[0, 1].set_xlabel("petal_length")
axes[0, 1].set_ylabel("petal_width")
axes[0, 1].scatter(lilac_data.petal_length[:50],
                   lilac_data.petal_width[:50], c="b")
axes[0, 1].scatter(lilac_data.petal_length[50:100],
                   lilac_data.petal_width[50:100], c="g")
axes[0, 1].scatter(lilac_data.petal_length[100:],
                   lilac_data.petal_width[100:], c="r")

axes[0, 2].set_xlabel("sepal_length")
axes[0, 2].set_ylabel("petal_length")
axes[0, 2].scatter(lilac_data.sepal_length[:50],
                   lilac_data.petal_length[:50], c="b")
axes[0, 2].scatter(lilac_data.sepal_length[50:100],
                   lilac_data.petal_length[50:100], c="g")
axes[0, 2].scatter(lilac_data.sepal_length[100:],
                   lilac_data.petal_length[100:], c="r")

axes[1, 0].set_xlabel("sepal_width")
axes[1, 0].set_ylabel("petal_width")
axes[1, 0].scatter(lilac_data.sepal_width[:50],
                   lilac_data.petal_width[:50], c="b")
axes[1, 0].scatter(lilac_data.sepal_width[50:100],
                   lilac_data.petal_width[50:100], c="g")
axes[1, 0].scatter(lilac_data.sepal_width[100:],
                   lilac_data.petal_width[100:], c="r")

axes[1, 1].set_xlabel("sepal_length")
axes[1, 1].set_ylabel("petal_width")
axes[1, 1].scatter(lilac_data.sepal_length[:50],
                   lilac_data.petal_width[:50], c="b")
axes[1, 1].scatter(lilac_data.sepal_length[50:100],
                   lilac_data.petal_width[50:100], c="g")
axes[1, 1].scatter(lilac_data.sepal_length[100:],
                   lilac_data.petal_width[100:], c="r")

axes[1, 2].set_xlabel("sepal_width")
axes[1, 2].set_ylabel("petal_length")
axes[1, 2].scatter(lilac_data.sepal_width[:50],
                   lilac_data.petal_length[:50], c="b")
axes[1, 2].scatter(lilac_data.sepal_width[50:100],
                   lilac_data.petal_length[50:100], c="g")
axes[1, 2].scatter(lilac_data.sepal_width[100:],
                   lilac_data.petal_length[100:], c="r")


# 数据划分
from sklearn.model_selection import train_test_split

# 得到 lilac 数据集中 feature 的全部序列: sepal_length,sepal_width,petal_length,petal_width
feature_data = lilac_data.iloc[:, :-1]
label_data = lilac_data["labels"]  # 得到 lilac 数据集中 label 的序列
x_train, x_test, y_train, y_test = train_test_split(
    feature_data, label_data, test_size=0.3, random_state=2)

x_test  # 输出 lilac_test 查看


# 训练模型
"""使用 sklearn 构建 KNN 预测模型
"""
from sklearn import neighbors
import sklearn


def sklearn_classify(train_data, label_data, test_data, k_num):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k_num)
    # 训练数据集
    knn.fit(train_data, label_data)
    # 预测
    predict_label = knn.predict(test_data)
    # 返回预测值
    return predict_label


# 模型预测
"""使用数据集进行预测
"""
y_predict = sklearn_classify(x_train, y_train, x_test, 3)
y_predict


# 精确计算
"""准确率计算
"""


def get_accuracy(test_labels, pred_labels):
    correct = np.sum(test_labels == pred_labels)  # 计算预测正确的数据个数
    n = len(test_labels)  # 总测试集数据个数
    accur = correct/n
    return accur

get_accuracy(y_test, y_predict)


# K 值选择
normal_accuracy = []  # 建立一个空的准确率列表
k_value = range(2, 11)
for k in k_value:
    y_predict = sklearn_classify(x_train, y_train, x_test, k)
    accuracy = get_accuracy(y_test, y_predict)
    normal_accuracy.append(accuracy)

plt.xlabel("k")
plt.ylabel("accuracy")
new_ticks = np.linspace(0.6, 0.9, 10)  # 设定 y 轴显示，从 0.6 到 0.9
plt.yticks(new_ticks)
plt.plot(k_value, normal_accuracy, c='r')
plt.grid(True)  # 给画布增加网格