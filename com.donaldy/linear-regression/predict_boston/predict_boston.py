# -*- coding: utf-8 -*-
# @Time    : 7/29/2018 10:36 AM
# @Author  : DonaldY

# Target: 加载一个真实数据集，并使用scikit-learn 构建预测模型，实现回归预测

# 运行并下载数据集
# !wget http://labfile.oss.aliyuncs.com/courses/1081/course-5-boston.csv

# 使用 Pandas 加载并预览数据集
import pandas as pd

df = pd.read_csv("course-5-boston.csv")

# 查看DataFrame 前5行数据
df.head()

# 选取 crim, rm, lstat 三个特征用于线性回归模型训练
features = df[['crim', 'rm', 'lstat']]
# 统计了每列数据的个数、最大值、最小值、平均数等信息
features.describe()

# 目标值数据
target = df['medv']

# 得到 70% 位置
split_num = int(len(features)*0.7)

# 训练集特征
train_x = features[:split_num]
# 训练集目标
train_y = target[:split_num]

# 测试集特征
test_x = features[split_num:]
# 测试集目标
test_y = target[split_num:]


# 使用scikit-learn 构建预测模型
from sklearn.linear_model import LinearRegression

# 建立模型
model = LinearRegression()
# 训练模型
model.fit(train_x, train_y)
# 输出训练后的模型参数和截距项
model.coef_, model.intercept_


# 输入测试集特征进行预测
preds = model.predict(test_x)
# 预测结果
preds


# 平均绝对误差、平均绝对百分比误差、均方差等多个指标进行评价

# 平均绝对误差（MAE）
def mae_value(y_true, y_pred):
    n = len(y_true)
    mae = sum(pd.np.abs(y_true - y_pred)) / n
    return mae


# 均方误差（MSE）
def mse_value(y_true, y_pred):
    n = len(y_true)
    mse = sum(pd.np.square(y_true - y_pred)) / n
    return mse


mae = mae_value(test_y.values, preds)
mse = mse_value(test_y.values, preds)

print("MAE: ", mae)
print("MSE: ", mse)
