import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
"""
import tensorflow as tf

# 创建一个回调函数来保存模型的检查点
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,  # 指定检查点保存的路径
    save_weights_only=True,  # 仅保存模型的权重
    save_best_only=True,  # 只保存最佳模型
    monitor='val_loss',  # 监控的指标，例如验证集上的损失
    mode='min',  # 监控指标的模式，例如最小化损失
    verbose=1  # 显示保存模型的信息
)

# 创建并编译模型
model = tf.keras.models.Sequential(...)
model.compile(...)

# 进行模型训练，并使用回调函数保存模型的检查点
model.fit(x_train, y_train, ..., callbacks=[checkpoint_callback])

# 加载模型的检查点
model.load_weights(checkpoint_save_path)

# 继续训练模型
model.fit(x_train, y_train, ...)
"""