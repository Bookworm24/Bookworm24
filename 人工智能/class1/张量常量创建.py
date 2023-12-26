import tensorflow as tf

a = tf.constant([1, 5], dtype=tf.int64)
print("a:", a)
print("a.dtype:", a.dtype)
print("a.shape:", a.shape)

# 本机默认 tf.int32  可去掉dtype试一下 查看默认值
"""
`tf.constant`是TensorFlow中的一个函数，用于创建一个张量常量。它接受一个值和一个可选的数据类型参数，并返回一个常量张量。

示例用法：
```python
import tensorflow as tf

# 创建一个整数常量
a = tf.constant(5)
print(a)  # 输出: tf.Tensor(5, shape=(), dtype=int32)

# 创建一个浮点数常量
b = tf.constant(3.14, dtype=tf.float32)
print(b)  # 输出: tf.Tensor(3.14, shape=(), dtype=float32)

# 创建一个字符串常量
c = tf.constant("Hello, TensorFlow!")
print(c)  # 输出: tf.Tensor(b'Hello, TensorFlow!', shape=(), dtype=string)
```

`tf.constant`函数可以用于创建具有不同维度和形状的常量张量，可以通过传递一个列表或多维数组作为值来实现。

更多关于`tf.constant`函数的用法和参数选项，请参考TensorFlow官方文档。

"""