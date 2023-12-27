import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)
"""
np.set_printoptions(threshold=np.inf)是Numpy库中的一个函数调用，它用于设置打印数组时的选项。

默认情况下，当数组中的元素数量超过一个阈值时，Numpy会省略显示中间的元素，并以省略号 (...) 代替。通过设置threshold参数为np.inf，可以强制Numpy打印数组的所有元素，而不进行省略。

这在调试和查看数组内容时非常有用，特别是当数组非常大时。通过设置threshold=np.inf，可以确保所有元素都被显示，无论数组的大小。

以下是一个示例，演示了如何使用np.set_printoptions来设置打印数组时的选项
"""

"""
5.1 全连接网络回顾
√ 全连接 NN 特点：每个神经元与前后相邻层的每一个神经元都有连接关系。（可以实
现分类和预测）
全连接网络的参数个数为： ∑（前层×后层+后层）
如图 5-1 所示，针对一张分辨率仅为 28 * 28 的黑白图像（像素值个数为 28 * 28 * 1 = 
784），全连接网络的参数总量就有将近 40 万个
为了解决参数量过大而导致模型过拟合的问题，一般不会将原始图像直接输入，而是先
对图像进行特征提取，再将提取到的特征输入全连接网络，如图 5-3 所示，就是将汽车图片
经过多次特征提取后再喂入全连接网络
"""


cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 可视化训练集输入特征的第一个元素
plt.imshow(x_train[0])  # 绘制图片
plt.show()

# 打印出训练集输入特征的第一个元素
print("x_train[0]:\n", x_train[0])
# 打印出训练集标签的第一个元素
print("y_train[0]:\n", y_train[0])

# 打印出整个训练集输入特征形状
print("x_train.shape:\n", x_train.shape)
# 打印出整个训练集标签的形状
print("y_train.shape:\n", y_train.shape)
# 打印出整个测试集输入特征的形状
print("x_test.shape:\n", x_test.shape)
# 打印出整个测试集标签的形状
print("y_test.shape:\n", y_test.shape)
