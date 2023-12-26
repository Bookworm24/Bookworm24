import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))#指定生成训练参数
print(w)
lr = 0.2
epoch = 40

for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环40次迭代。
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        """
        
        tf.GradientTape是TensorFlow中的一个上下文管理器，用于自动计算梯度
        。它提供了一种方便的方式来计算张量相对于某些可训练变量的导数。
        """
        loss = tf.square(w + 1)#loss = tf.square(w + 1)表示定义了一个损失函数loss，它是将参数w加上1后取平方
        #用来判断损失函数的与原值判断是否为0
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导,对5

    w.assign_sub(lr * grads)  #更新参数 .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    #初始学习率乘以学习衰减；率
    print("After %s epoch,w is %f,loss is %f" % (epoch, w.numpy(), loss))

# lr初始值：0.2   请自改学习率  0.001  0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数w
