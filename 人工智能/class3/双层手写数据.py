import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
"""
这一步是为了对输入数据进行归一化处理。

在神经网络中，对输入数据进行归一化可以帮助提高模型的训练效果和收敛速度。常见的归一化方式是将输入数据的取值范围缩放到 [0, 1] 或 [-1, 1] 之间。

在这段代码中，将训练集 x_train 和测试集 x_test 中的像素值除以 255.0，将其范围缩放到 [0, 1] 之间。因为 MNIST 数据集中的像素值是 0 到 255 的整数，除以 255.0 后得到的结果就是浮点数在 [0, 1] 之间的归一化值。

通过归一化处理，可以避免输入数据中的数值差异过大对模型训练造成的影响，使得模型能够更好地学习到数据的统计特征，提高模型的泛化能力。
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
"""
根据给出的模型概要信息，可以得到以下结论：

- 模型的名称是 "sequential"，表示这是一个序列模型。
- 模型共有三个层次：输入层、隐藏层和输出层。
- 输入层是一个 `Flatten` 层，将输入数据展平为一维数组，输出形状为 `(None, 784)`。其中，`None` 表示可以接受任意大小的输入样本，784 是输入数据的维度。
- 隐藏层是一个 `Dense` 层，具有128个神经元和ReLU激活函数，输出形状为 `(None, 128)`。其中，`None` 表示可以接受任意大小的输入样本。
- 输出层是一个 `Dense` 层，具有10个神经元和Softmax激活函数，输出形状为 `(None, 10)`。其中，`None` 表示可以接受任意大小的输入样本。
- 模型总共有101,770个参数。其中，隐藏层有100,480个参数，输出层有1,290个参数。
- 所有的参数都是可训练的，即可以在模型训练过程中进行更新和优化。
- 模型没有非训练参数，即没有需要学习的固定参数。

通过分析模型概要信息，我们可以了解到模型的结构、输入输出形状，以及模型中可训练参数的数量。这些信息对于模型的设计和调整非常有帮助。
"""