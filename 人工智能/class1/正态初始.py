import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)
"""
```python
import tensorflow as tf

# 使用正态分布随机初始化一个形状为2x2的张量，均值为0.5，标准差为1
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
# 输出:
# d: tf.Tensor(
# [[0.99960786 1.9074883 ]
#  [1.9417299  0.24629213]], shape=(2, 2), dtype=float32)

# 使用截断正态分布随机初始化一个形状为2x2的张量，均值为0.5，标准差为1
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)
# 输出:
# e: tf.Tensor(
# [[1.0884769  0.52135736]
#  [0.03011947 0.17342971]], shape=(2, 2), dtype=float32)
```

在上述代码中，我们使用`tf.random.normal`和`tf.random.truncated_normal`函数分别创建了形状为2x2的张量`d`和`e`。这些函数根据指定的均值和标准差参数，采用正态分布或截断正态分布生成随机数来填充张量。

`tf.random.normal`函数生成的随机数满足指定均值和标准差的正态分布。而`tf.random.truncated_normal`函数生成的随机数也满足指定均值和标准差的正态分布，但截断在均值两个标准差以外的数据，即不包含超过均值加减两倍标准差范围之外的值。

这些函数在神经网络中的权重初始化、生成噪声数据等方面非常有用。

更多关于`tf.random.normal`和`tf.random.truncated_normal`函数的用法和参数选项，请参考TensorFlow官方文档。
"""