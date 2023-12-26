# -*- coding: UTF-8 -*-
# 利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线

# 导入所需模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data#输入特征，如花瓣，花色，花杆
y_data = datasets.load_iris().target#标签，如是什么花

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
"""
当我们使用随机数生成器生成一组随机数时，不同的种子会得到不同的随机数序列。让我们通过一个简单的例子来理解这个概念。

```python
import numpy as np

# 设置种子为 116
np.random.seed(116)

# 生成随机数
random_nums1 = np.random.rand(5)
print("随机数序列1:", random_nums1)

# 再次生成随机数
random_nums2 = np.random.rand(5)
print("随机数序列2:", random_nums2)
```

输出结果：

```
随机数序列1: [0.01987789 0.79649894 0.17162968 0.84700953 0.76449276]
随机数序列2: [0.02593392 0.54183424 0.76460504 0.40370008 0.50728847]
```

在这个例子中，我们首先设置种子为116，然后使用`np.random.rand(5)`生成了两个长度为5的随机数序列`random_nums1`和`random_nums2`。

由于我们使用相同的种子116，所以无论何时运行代码，都会得到相同的随机数序列。这意味着每次运行代码时，`random_nums1`和`random_nums2`的值都将保持不变。

如果我们注释掉`np.random.seed(116)`这一行，并再次运行代码，将会得到不同的随机数序列。

因此，设置种子可以确保随机数生成的可重复性，这对于调试代码和结果的复现非常有用。
"""
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)#np.random.shuffle 是 NumPy 中的一个函数，用于随机打乱数组中元素的顺序。,使特征值和标签仍然能够
#对应的上，因为使用了相同的随机种子
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
"""
我们使用了 tf.random.truncated_normal() 函数来生成一个具有截断正态分布的随机张量作为变量的初始值。这个随机张量的形状是 [4, 3]，意味着有 4 个输入特征和 3 个输出类别
"""
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
"""
偏置项（bias）是一种可学习的参数，用于调整神经元的激活值。偏置项的数量通常与输出层的神经元数量相同。

在上述代码中，我们创建了一个形状为 [3] 的偏置项变量 b1。这意味着我们的神经网络有 3 个输出类别，每个类别对应一个偏置项。
这样做的目的是为每个输出类别提供一个额外的参数，以调整神经元输出的偏移量。
"""

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

# 训练部分
for epoch in range(epoch):  #数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  #batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算matmul是相乘
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            """
            softmax 操作的输出可以被解释为对应每个类别的预测概率。例如，如果输出 y 的第一个元素是 0.8，表示模型对第一个类别的预测概率为 0.8，
            那么第一个类别在这个样本中的概率就是 0.8。同样地，可以通过 softmax 输出的其他元素得到对应类别的概率。
            """
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            """
            在上述代码中，我们使用 `tf.one_hot` 函数将训练标签 `y_train` 转换为 one-hot 编码形式的张量 `y_`。

具体而言，`y_train` 是一个包含了训练样本的标签的张量。通过设置 `depth=3` 参数，我们指定了 one-hot 编码的深度为 3。这意味着我们的问题是一个多分类问题，有 3 个输出类别。

在进行分类任务时，one-hot 编码是一种常见的标签编码方式。它通过将每个类别表示为一个独立的二进制特征向量，其中只有一个元素为 1，其余元素为 0。对于多分类问题，每个类别都会对应一个不同的 one-hot 编码。

通过使用 `tf.one_hot` 函数，并指定 `depth=3`，我们将 `y_train` 中的每个标签值转换为对应的 one-hot 编码形式。例如，如果某个训练样本的标签是 2，那么对应的 one-hot 编码就是 `[0, 0, 1],标签索引0,1,2`。

将标签转换为 one-hot 编码的好处是，它可以提供更多的信息给模型，帮助模型更好地理解不同的类别之间的关系。同时，它还可以使损失函数的计算更加方便和准确，以便进行模型的训练和评估。

总而言之，通过将训练标签转换为 one-hot 编码，我们可以更好地处理多分类问题，并为模型提供更多的信息。在这个例子中，由于有 3 个输出类别，所以指定 `depth=3` 来生成对应的 one-hot 编码。
罗 的 发，有的话就是1，没有就是0
1   0   0
            """
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        #在上述代码中，我们使用 TensorFlow 的函数 tf.equal 来比较模型的预测值 pred 和
        # 真实标签 y_test 是否相等。然后，使用 tf.cast 函数将比较结果转换为整数类型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
