import tensorflow as tf
import numpy as np

SEED = 23455

rdm = np.random.RandomState(seed=SEED)  # 生成[0,1)之间的随机数
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
"""
通过计算x1 + x2得到一个基础的标签，然后通过(rdm.rand() / 10.0 - 0.05)生成一个位于[-0.05, 0.05]范围内的随机噪声，并将这个噪声加到基础标签上。最终得到的结果作为带有噪声的标签y_。

这种生成带有噪声的标签的作用是模拟真实世界中的数据不完全准确性或随机性。在实际问题中，很少有完全准确的数据，通常会受到测量误差、噪声或其他随机因素的影响
"""

x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 15000
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss_mse = tf.reduce_mean(tf.square(y_ - y))

    grads = tape.gradient(loss_mse, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())

"""
使用优化器
一阶优化算法是指通过梯度计算来更新模型参数的优化算法。下面总结了一些常见的梯度下降方法：

1. 批量梯度下降（Batch Gradient Descent，BGD）：在每一次迭代中，使用所有训练样本的梯度来更新模型参数。

2. 随机梯度下降（Stochastic Gradient Descent，SGD）：在每一次迭代中，随机选择一个训练样本，使用该样本的梯度来更新模型参数。

3. 小批量梯度下降（Mini-Batch Gradient Descent）：在每一次迭代中，随机选择一小批训练样本，使用这些样本的梯度来更新模型参数。

4. 动量优化（Momentum）：引入动量项，根据之前的梯度更新方向和当前梯度的加权平均来更新模型参数，可以加速收敛。

5. AdaGrad（Adaptive Gradient）：根据梯度的历史累积信息来自适应地调整学习率，对稀疏特征有较好的效果。

6. RMSProp（Root Mean Square Propagation）：类似于 AdaGrad，但对梯度的历史累积信息进行指数加权平均，可以减缓学习率的下降速度。

7. Adam（Adaptive Moment Estimation）：结合了动量优化和 RMSProp，同时考虑梯度的一阶矩估计和二阶矩估计，适用于大多数问题。是目前最常用的优化算法之一。

这些梯度下降方法都有各自的特点和适用场景，选择合适的优化算法可以加快模型的收敛速度、提高准确性，并避免陷入局部最优解。在实际应用中，通常需要根据具体问题进行调参和选择合适的优化算法。
"""
