import tensorflow as tf
# 使用均匀分布随机初始化一个形状为2x2的张量，取值范围为[0, 1)
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f:", f)
