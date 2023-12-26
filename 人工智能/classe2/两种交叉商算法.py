# softmax与交叉熵损失函数的结合
import tensorflow as tf
import numpy as np

y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pro = tf.nn.softmax(y)
#softmax 函数是一种常用的激活函数，可以将一个实数向量转化为概率分布。它的计算公式如下：


loss_ce1 = tf.losses.categorical_crossentropy(y_,y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)

print('分步计算的结果:\n', loss_ce1)
print('结合计算的结果:\n', loss_ce2)
"""
非常抱歉，我之前的回答有误。在 TensorFlow 中，`tf.losses.categorical_crossentropy` 函数和 `tf.nn.softmax_cross_entropy_with_logits` 函数都是用于计算交叉熵损失的函数，但它们的输入和计算方式不同。具体来说：

1. `tf.losses.categorical_crossentropy` 函数用于计算多分类问题中的交叉熵损失。它的输入是真实标签和模型的预测概率分布，其中真实标签是一个 one-hot 编码的向量，模型的预测概率分布是一个概率值的向量。这个函数会先对真实标签和预测概率分布进行对数运算，然后计算交叉熵损失。

2. `tf.nn.softmax_cross_entropy_with_logits` 函数也用于计算多分类问题中的交叉熵损失。它的输入是真实标签和模型的预测结果（logits），其中真实标签是一个 one-hot 编码的向量，模型的预测结果是一个未经过 softmax 操作的概率值的向量。这个函数会先对模型的预测结果进行 softmax 操作，然后计算交叉熵损失。

所以，正确的代码应该是：

```python
print('分步计算的结果:\n', loss_ce1)
print('结合计算的结果:\n', loss_ce2)
```

`loss_ce1` 是使用 `tf.losses.categorical_crossentropy` 函数计算的交叉熵损失。

`loss_ce2` 是使用 `tf.nn.softmax_cross_entropy_with_logits` 函数计算的交叉熵损失。

请注意，`loss_ce1` 和 `loss_ce2` 都是 TensorFlow 的张量对象，打印出来的结果可能是张量的信息，而不是具体的数值。如果需要获取数值结果，可以使用 `sess.run()` 或 `numpy()` 方法获取具体的数值。
"""

# 输出的结果相同