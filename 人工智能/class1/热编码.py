import tensorflow as tf

classes = 3
labels = tf.constant([1, 2, 3,4,5,6,7,8])  # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels, depth=classes)
print("result of labels1:", output)
print("\n")
# 共有3行，意味的labes为3个元素。有3列因为classes为3..3列用0和1来排序有八种
"""
`tf.one_hot` 是 TensorFlow 中的一个函数，用于将标签转换为独热编码。它接受一个标签张量和一个指定深度的参数，并返回一个独热编码的张量。

在你的例子中，`output = tf.one_hot(labels, depth=classes)` 的意思是将标签张量 `labels` 转换为一个独热编码的张量，其中 `classes` 是指定的深度（或类别数）。

示例用法：
```python
import tensorflow as tf

# 假设有一个标签张量 labels 和类别数 classes
labels = tf.constant([0, 1, 2, 1])
classes = 3

# 使用 tf.one_hot 转换为独热编码
output = tf.one_hot(labels, depth=classes)

print(output)
# 输出:
# tf.Tensor(
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]], shape=(4, 3), dtype=float32)
```

在上述代码中，我们创建了一个标签张量 `labels`，并指定了类别数 `classes` 为 3。然后，我们使用 `tf.one_hot` 函数将标签张量转换为独热编码的张量 `output`。

`output` 的形状为 (4, 3)，其中每一行表示一个标签的独热编码。

`tf.one_hot` 函数在进行分类任务时非常有用，可以将离散的标签转换为神经网络模型可以处理的独热编码格式。

更多关于 `tf.one_hot` 函数的用法和参数选项，请参考 TensorFlow 官方文档。
性别:[“男”,”女”]

只有两个特征，所以N为2，下面同理。

男=>10

女=>01

 

班级:[“1班”,”2班”,”3班”]

1班=>100

2班=>010

3班=>001

 

年纪:[“一年级”,”二年级”,”三年级”,”四年级”]

一年级=>1000

二年级=>0100

三年级=>0010

四年级=>0001

 

所以如果一个样本为[“男”,”2班”,”四年级”]的时候，完整的特征数字化的结果为：

[1,0,0,1,0,0,0,0,1]
————————————————
版权声明：本文为CSDN博主「interesting233333」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/lipengfei0427/article/details/109393039
"""