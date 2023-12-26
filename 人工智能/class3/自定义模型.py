import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)
        return y

model = IrisModel()

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
model.summary()
"""
使用 Sequential 可以快速搭建网络结构，但是如果网络包含跳连等其他复
杂网络结构，Sequential 就无法表示了。这就需要使用 class 来声明网络结构。
class MyModel(Model):
def __init__(self):
super(MyModel, self).__init__()
 //初始化网络结构
def call(self, x):
y = self.d1(x)
return y
使用 class 类封装网络结构，如上所示是一个 class 模板，MyModel 表示声
明的神经网络的名字，括号中的 Model 表示创建的类需要继承 tensorflow 库中
的 Model 类。类中需要定义两个函数，__init__()函数为类的构造函数用于初
始化类的参数，spuer(MyModel,self).__init__()这行表示初始化父类的参
数。之后便可初始化网络结构,搭建出神经网络所需的各种网络结构块。call()
函数中调用__init__()函数中完成初始化的网络块，实现前向传播并返回推理
值。使用 class 方式搭建鸢尾花网络结构的代码如下所示。
class IrisModel(Model):
def __init__(self):
super(IrisModel, self).__init__()
self.d1 = Dense(3, activation='sigmoid', 
kernel_regularizer=tf.keras.regularizers.l2())
def call(self, x):
y = self.d1(x)
return y
搭建好网络结构后只需要使用 Model=MyModel()构建类的对象，就可以使用
该模型了
"""
