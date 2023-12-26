import tensorflow as tf
from sklearn import datasets
import numpy as np


"""

tf.keras 搭建神经网络六部法 
第一步：import 相关模块，如 import tensorflow as tf。
第二步：指定输入网络的训练集和测试集，如指定训练集的输入 x_train 和标签
y_train，测试集的输入 x_test 和标签 y_test。
第三步：逐层搭建网络结构，model = tf.keras.models.Sequential()。
第四步：在 model.compile()中配置训练方法，选择训练时使用的优化器、损失
函数和最终评价指标。
第五步：在 model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、
每个 batch 的大小（batchsize）和数据集的迭代次数（epoch）。
第六步：使用 model.summary()打印网络结构，统计参数数目


"""

#用于对数组进行随机重排。
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

"""
序列模型是一种基本的神经网络模型，它由多个网络层按照顺序构成，数据会从第一个层顺序流过每一层，最终得到输出结果。

tf.keras.Sequential 可以通过将各个网络层按照顺序添加到模型中来构建序列模型。下面是一个示例：

python 蟒
Copy
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
在上述示例中，tf.keras.Sequential 创建了一个空的序列模型。然后，通过 layers.Dense 方法添加了两个全连接层，分别是输入层和输出层。第一个全连接层具有32个神经元和ReLU激活函数，输入形状为（784，），即输入数据的维度。第二个全连接层具有10个神经元和softmax激活函数，用于多分类问题的输出。

通过 tf.keras.Sequential 创建的序列模型，可以使用其它方法进行编译、训练和评估操作，例如 model.compile、model.fit 和 model.evaluate 等。
"""
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()
