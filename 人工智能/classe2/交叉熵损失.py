import tensorflow as tf
"""
交叉熵损失函数在机器学习中具有广泛的应用，特别是在分类问题中。它的主要作用包括以下几个方面：

1. 衡量预测结果与真实标签之间的差异：交叉熵损失函数可以度量模型的预测结果与真实标签之间的差异程度。通过最小化交叉熵损失，可以使模型更加准确地预测样本的类别。

2. 优化模型参数：交叉熵损失函数是一个可微分的函数，可以使用梯度下降等优化算法来最小化损失函数。通过不断迭代优化模型参数，可以使模型在训练数据上的损失逐渐减小，提高模型的性能。

3. 提供概率分布的输出：交叉熵损失函数可以将模型的输出转化为概率分布。通过对预测结果进行 softmax 操作，可以得到每个类别的概率值，表示模型对于每个类别的置信度。这对于多分类问题的结果解释和后续的决策非常重要。

4. 缓解类别不平衡问题：交叉熵损失函数可以在类别不平衡的情况下提供更好的结果。由于交叉熵损失函数对于错误分类的样本有较大的惩罚，因此模型更加关注少数类别样本，可以更好地处理类别不平衡问题。

总之，交叉熵损失函数在分类问题中起着重要的作用，能够衡量模型的预测结果和真实标签之间的差异，并通过优化模型
"""
loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
print("loss_ce1:", loss_ce1)
print("loss_ce2:", loss_ce2)

"""
上述代码中，使用了 TensorFlow 的 `tf.losses.categorical_crossentropy` 函数计算了交叉熵损失 `loss_ce1`。

交叉熵损失函数常用于多分类问题中，用于衡量模型的预测结果与真实标签之间的差异。在这个函数中，需要传入两个参数：预测值和真实标签。

`[1, 0]` 是真实标签，表示样本属于第一个类别的概率为1，属于第二个类别的概率为0。

`[0.6, 0.4]` 是模型的预测值，表示模型对于样本属于第一个类别和第二个类别的预测概率。

`tf.losses.categorical_crossentropy` 函数根据这两个参数计算了交叉熵损失。具体计算方式是对每个类别的预测概率和真实标签进行对数运算，然后相乘并取反，最后对所有类别的损失进行求和。

最终，得到的 `loss_ce1` 是一个标量，表示模型的交叉熵损失。
"""
# 交叉熵损失函数