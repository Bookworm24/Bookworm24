import tensorflow as tf

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print("a:", a)
print("b:", b)
print("c:", c)
# tf.fill是TensorFlow中的一个函数，用于创建一个指定形状并填充指定值的张量。它接受一个形状参数和一个值参数，并返回一个新的张量。
#
# 在你的例子中，c = tf.fill([2, 2], 9)的意思是创建一个形状为2x2的张量，并将其填充为值为9的元素。