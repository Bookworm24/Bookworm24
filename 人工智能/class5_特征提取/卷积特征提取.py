import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

"""
卷积层在神经网络模型中的重要作用:

1. 提取空间特征。卷积层通过滑动过滤器运算,可以有效提取输入信号在空间维度上的有用特征,如边缘、颜色变化等。这对图像、视频等结构化数据非常有效。

2. 强大的表达能力。多层次的卷积网络可以学习出高级抽象特征,有效实现复杂计算机视觉等任务。

3. 参数共享。卷积层通过参数共享,大大减少模型参数数量,缓解过拟合问题。

4. 模型可移植性好。卷积网络学习出的特征提取通常具有很强的模拟性,能应用于不同数据集和任务。

5. 对空间变化不敏感。卷积操作对输入信号位置信息不那么敏感,可以提取出空间不变的特征。

具体来说,卷积层在以下任务中效果显著:

- 图像分类:如MNIST、CIFAR、ImageNet等  
- 目标检测:图像中具体区域对象检测
- 图像分割:不同部分图像的分割
- 图像风格迁移:提取内容特征与风格特征
- 自然语言处理:嵌入层学习词汇特征
- 生物医学图像分析:CT、MRI的数据分类诊断

总之,通过有效提取输入空间结构特征,卷积层大大增强了神经网络学习复杂模式的能力,在各类计算机视觉和序列任务中都发挥重要作用。
"""
"""
tf.keras.layers.Conv2D 是 Keras 中 定义 2D 卷积层的类。

主要属性和方法:

- filters:整数,输出空间的维度(即卷积核数)。

- kernel_size:整数或元组长度为 2 的整数列表,指定卷积窗口的宽度和高度。 

- strides:整数或元组长度为 2 的整数列表,指定滑动步长的宽度和高度。

- padding:‘valid’ 或 ‘same’,指定补边方式。

- data_format:‘channels_first’ 或 ‘channels_last’。

- dilation_rate:整数或元组长度为 2 的整数列表,指定膨胀率。

- activation:激活函数,默认为 None。

- use_bias:布尔值,是否添加偏置向量。

- kernel_initializer:权重初始化方法。

- bias_initializer:偏置初始化方法。

- kernel_regularizer:正则化权重矩阵的方法。

- bias_regularizer:正则化偏置向量的方法。

- activity_regularizer:正则化卷积层输出的方法。

- kernel_constraint:权重值的约束。

- bias_constraint:偏置值的约束。

主要方法:

- call(inputs):执行前向传播,返回tensor.

用法:

```python
x = Conv2D(filters, kernel_size, strides=(1, 1), 
           padding='valid', activation='relu')(inputs)
```

它可以很灵活地定义常见的卷积层结构,在 CNN 中广泛应用。

批标准化（Batch Normalization， BN）
𝑯𝑯𝒊𝒊
′𝒌𝒌 =
𝑯𝑯𝒊𝒊
𝒌𝒌 − 𝝁𝝁batch
𝒌𝒌
𝝈𝝈batch
𝒌𝒌
𝑯𝑯𝒊𝒊
𝒌𝒌
： 批标准化前，第k个卷积核，输出特征图中第 i 个像素点
𝝁𝝁batch
𝒌𝒌
：批标准化前，第k个卷积核，batch张输出特征图中所有像素点平均值
𝝈𝝈batch
𝒌𝒌
：批标准化前，第k个卷积核，batch张输出特征图中所有像素点标准差
𝝁𝝁batch
𝒌𝒌 = 𝟏𝟏
𝒎𝒎�
i =𝟏𝟏
𝒎𝒎
𝑯𝑯𝒊𝒊
𝒌𝒌 𝝈𝝈batch
𝒌𝒌 = 𝜹𝜹 +
𝟏𝟏
𝒎𝒎�
i =𝟏𝟏
𝒎𝒎
(𝑯𝑯𝒊𝒊
𝒌𝒌−𝝁𝝁batch
𝒌𝒌 )𝟐𝟐
批标准化后，第 k个卷积核的输出特征图（feature map）中第 i 个像素点
标准化：使数据符合0均值，1为标准差的分布。
批标准化：对一小批数据（batch），做标准化处理 。

池化用于减少特征数据量。
最大值池化可提取图片纹理，均值池化可保留背景特征。
Baseline继承于tf.keras.Model,实现了__init__和call方法,定义了一个Keras函数模型。

__init__中使用了Sequential的风格,定义了一个标准的卷积神经网络前向结构,包含卷积层、BN层、池化层、 Dropout层等。

call方法定义了前向计算过程,按顺序将输入x传递给每一层,得到最终输出y。

每一层都使用对应的Keras层定义,如Conv2D、BatchNormalization、MaxPool2D等。

在最后flatten层后接两个Dense层组成全连接部分,进行分类。
"""
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y
"""
model = tf.keras.models.Sequential([
Conv2D(filters=6, kernel_size=(5, 5), padding='same'), # 卷积层
BatchNormalization(), # BN层
Activation('relu'), # 激活层
MaxPool2D(pool_size=(2, 2), strides=2, padding='same'), # 池化层
Dropout(0.2), # dropout层
"""

model = Baseline()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
