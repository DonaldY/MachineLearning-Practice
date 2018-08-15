# -*- coding: utf-8 -*-
# @Time    : 7/31/2018 7:29 PM
# @Author  : DonaldY

# 生成 10 * 10的希尔伯特矩阵
from scipy.linalg import hilbert

x = hilbert(10)
x

# 希尔伯特转置矩阵于原矩阵相乘
import numpy as np

np.matrix(x).T * np.matrix(x)

