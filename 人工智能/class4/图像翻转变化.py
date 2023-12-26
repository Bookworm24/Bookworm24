# 显示原始图像和增强后的图像
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#常用增强方法：
# 缩放系数：rescale=所有数据将乘以提供的值
# 随机旋转：rotation_range=随机旋转角度数范围
# 宽度偏移：width_shift_range=随机宽度偏移量
# 高度偏移：height_shift_range=随机高度偏移量
# 水平翻转：horizontal_flip=是否水平随机翻转
# 随机缩放：zoom_range=随机缩放的范围 [1-n，1+n]：image_gen_train = ImageDataGenerator(
"""
这段代码展示了如何使用`ImageDataGenerator`类进行图像增强，并显示原始图像和增强后的图像。

首先，从MNIST数据集加载训练数据，并将其重新调整为适当的形状。然后，定义一个`ImageDataGenerator`对象，并设置一些常用的增强方法，例如缩放、随机旋转、宽度偏移、高度偏移、水平翻转和随机缩放。

接下来，调用`image_gen_train.fit(x_train)`方法来对训练数据进行增强。然后，从训练数据中选择一部分样本，分别命名为`x_train_subset1`和`x_train_subset2`。

然后，创建一个图形对象，用于显示原始图像。通过循环遍历`x_train_subset1`中的每个样本，并使用`imshow()`方法显示图像。

接下来，创建另一个图形对象，用于显示增强后的图像。通过调用`image_gen_train.flow()`方法，传入`x_train_subset2`和一些参数，生成一个增强后的图像批次。然后，通过循环遍历批次中的每个图像，并使用`imshow()`方法显示图像。

最后，使用`break`语句来终止增强图像的循环，只显示第一个增强后的图像批次。

通过运行这段代码，你将看到原始图像和增强后的图像，以便比较它们的差异。

"""
rescale=1./255, #原像素值 0
image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=False,
    zoom_range=0.5
)
image_gen_train.fit(x_train)
"""
具体而言，image_gen_train.fit(x_train)会根据ImageDataGenerator对象中设置的增强参数，对训练数据进行批量处理，以生成增强后的数据。这些增强操作可以包括缩放、旋转、平移、翻转、缩放等。在每个训练迭代期间，模型会随机应用这些增强操作，从而使模型在不同的图像变换下进行训练，增加模型对各种变化的适应能力。

通过这种方式，数据增强可以帮助模型更好地学习数据的不变性和泛化能力，减少过拟合的风险，提高模型的性能。

需要注意的是，image_gen_train.fit(x_train)只是对增强操作进行配置，并没有进行实际的数据增强处理。实际的数据增强会在训练过程中通过model.fit()方法和ImageDataGenerator.flow()方法来实现。
批量操作x_train的数据，把里面所有的图片执行一篇模型里面定义的
"""
print("xtrain",x_train.shape)
x_train_subset1 = np.squeeze(x_train[:12])
print("xtrain_subset1",x_train_subset1.shape)
print("xtrain",x_train.shape)
x_train_subset2 = x_train[:12]  # 一次显示12张图片
print("xtrain_subset2",x_train_subset2.shape)

fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')
# 显示原始图片
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i + 1)
    ax.imshow(x_train_subset1[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()

# 显示增强后的图片
fig = plt.figure(figsize=(20, 2))
for x_batch in image_gen_train.flow(x_train_subset2, 图像翻转变化=12, shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i + 1)
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break;
