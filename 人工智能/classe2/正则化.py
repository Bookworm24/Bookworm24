# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#隐藏层担负的特征提取的功能

# 读入数据/标签 生成x_train y_train
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = x_data
y_train = y_data.reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
"""
我想知道关于输入层和隐藏层和输出层的解释

在神经网络中，输入层、隐藏层和输出层分别是网络的不同层次，具有不同的功能和特点。

1. 输入层（Input Layer）：输入层是神经网络的第一层，负责接收原始的输入数据。每个输入层神经元对应输入数据的一个特征。输入层的神经元数量通常与输入数据的特征数量相同。

2. 隐藏层（Hidden Layer）：隐藏层是位于输入层和输出层之间的一层或多层。隐藏层的神经元数量和层数可以根据问题的复杂度和数据的特点进行设计。隐藏层的作用是对输入数据进行非线性变换和特征提取，从而学习到更高级别的表示。每个隐藏层的神经元接收上一层的输出，经过激活函数的处理后，将结果传递给下一层。

3. 输出层（Output Layer）：输出层是神经网络的最后一层，负责产生网络的输出结果。输出层的神经元数量通常与问题的输出维度相同。输出层的激活函数根据问题的类型而定，例如对于二分类问题可以使用 sigmoid 函数，对于多分类问题可以使用 softmax 函数。

输入层、隐藏层和输出层之间的连接权重和偏置是神经网络的参数，通过训练过程来优化这些参数，使得神经网络能够学习到输入数据的特征和模式，从而得到准确的输出结果。

需要注意的是，神经网络的结构和层数以及每层的神经元数量是根据具体问题和数据特点进行设计的，在实际应用中需要根据问题的复杂度、数据的维度和数据量等因素进行调整。
https://blog.csdn.net/weixin_42426841/article/details/129569417
https://www.cnblogs.com/subconscious/p/5058741.html#second
"""
# 生成神经网络的参数，输入层为4个神经元，隐藏层为32个神经元，2层隐藏层，输出层为3个神经元
# 用tf.Variable()保证参数可训练矩阵相乘
#有四层，两个隐藏层
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
"""
w1 是输入层到第一个隐藏层的权重矩阵，形状为 [2, 11]。这表示输入层有 2 个神经元，第一个隐藏层有 11 个神经元。使用 tf.random.normal 函数以正态分布随机初始化权重矩阵。
"""
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))
"""https://blog.csdn.net/qq_34177812/article/details/104734520#:~:text=%E6%A0%B9%E6%8D%AE%E4%B8%8A%E8%BE%B9%E7%9A%84%E4%BB%8B%E7%BB%8D%E5%8F%AF%E7%9F%A5,%E6%A8%A1%E5%9E%8B%E6%9C%80%E7%BB%88%E8%AF%86%E5%88%AB%E7%9A%84%E7%BB%93%E6%9E%9C%E3%80%82"""
lr = 0.005  # 学习率为
epoch = 800  # 循环轮数

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            #在给定的代码中，h1 = tf.matmul(x_train, w1) + b1
            # 表示进行神经网络的乘加运算，将输入数据x_train与权重矩阵w1相乘，然后加上偏置向量b1。
            h1 = tf.nn.relu(h1)#激活函数，正则化
            y = tf.matmul(h1, w2) + b2
            """
            进行另一次神经网络的乘加运算，将ReLU激活后的输出h1与权重矩阵w2相乘，然后加上偏置向量b2。这样得到的输出y即为神经网络的最终输出。
            """

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_mse = tf.reduce_mean(tf.square(y_train - y))#定义损失函数之差，进行参数优化
            # 添加l2正则化
            loss_regularization = []
            # tf.nn.l2_loss(w)=sum(w ** 2) / 2
            """
            tf.nn.l2_loss()是TensorFlow中的一个函数，用于计算张量的L2范数的平方的一半。

函数的定义如下：
            """
            loss_regularization.append(tf.nn.l2_loss(w1))#添加正则化

            loss_regularization.append(tf.nn.l2_loss(w2))
            """
            在给出的代码中，`loss_regularization` 是一个列表，用于存储正则化项。正则化项用来惩罚模型的权重参数，以防止模型过拟合。具体来说，`loss_regularization` 列表中存储了 `w1` 和 `w2` 的 L2 正则化项。

`tf.nn.l2_loss(w1)` 是 TensorFlow 中的一个函数，用于计算张量 `w1` 的 L2 范数的平方的一半。L2 范数是指向量的平方和的平方根，用于衡量向量的大小。`tf.nn.l2_loss(w1)` 实际上计算了 `w1` 的平方和的一半，即 `sum(w1 ** 2) / 2`。这个值代表了 `w1` 的大小，通过添加到损失函数中，可以惩罚权重 `w1` 的大小，从而控制模型的复杂度。

同样地，`tf.nn.l2_loss(w2)` 计算了权重 `w2` 的 L2 范数的平方的一半，即 `sum(w2 ** 2) / 2`。这个值也代表了 `w2` 的大小，通过添加到损失函数中，可以惩罚权重 `w2` 的大小。

将正则化项添加到损失函数中，可以平衡模型的拟合能力和泛化能力。通过调整正则化项的权重系数，可以控制正则化的强度。较大的权重系数会更强烈地惩罚模型的复杂度，从而促使模型学习到更简单的权重参数。
            这行代码用于计算并添加权重矩阵w1的L2正则化项到loss_regularization列表中。

具体而言，tf.nn.l2_loss(w1)用于计算权重矩阵w1的L2范数的平方的一半。L2范数的定义是对向量的每个元素的平方求和后再开平方，即L2_norm(w) = sqrt(sum(w ** 2))。而tf.nn.l2_loss()函数计算的是L2范数的平方的一半，即l2_loss(w) = sum(w ** 2) / 2。

通过将tf.nn.l2_loss(w1)计算得到的L2正则化项添加到loss_regularization列表中，可以在总的损失函数中考虑模型复杂度的平衡。最终，通过tf.reduce_sum(loss_regularization)将loss_regularization列表中的元素求和，得到正则化项的值。
            """
            # 求和
            # 例：x=tf.constant(([1,1,1],[1,1,1]))
            #   tf.reduce_sum(x)
            # >>>6
            loss_regularization = tf.reduce_sum(loss_regularization)
            """
            最终，将正则化项的总和与均方误差损失函数相加，得到总的损失函数loss。这样，模型在训练过程中不仅会考虑预测值与真实值之间的差异（通过均方误差损失函数），
            还会考虑模型的复杂度（通过正则化项）。通过加权平衡预测精度和模型复杂度，可以提高模型的泛化能力。
            """
            loss = loss_mse + 0.03 * loss_regularization  # REGULARIZER = 0.03,把正则化的情况考虑

        # 计算loss对各个参数的梯度,
        variables = [w1, b1, w2, b2]
        """
        在给定的代码中，tape.gradient(loss, variables)的作用是使用梯度带对象tape来计算损失函数loss相对于模型参数variables的梯度。

具体而言，loss是模型的总损失函数，variables是模型的参数列表，包括w1, b1, w2, b2。tape.gradient(loss, variables)会计算损失函数loss对于模型参数variables的梯度，并将梯度值存储在grads变量中。

这样，grads变量就包含了损失函数相对于模型参数的梯度信息，可以被用于更新模型参数，例如通过梯度下降法进行参数优化。
        """
        #更新模型
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每200个epoch，打印loss信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_predict in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_predict], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
# 把坐标xx yy和对应的值probs放入contour函数，给probs值为0.5的所有点上色  plt.show()后 显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 读入红蓝点，画出分割线，包含正则化
# 不清楚的数据，建议print出来查看
