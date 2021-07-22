import os
os.environ['TF_CPP_MninN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np


# 模型构建
class VAECNN(keras.Model):
    def __init__(self, z_dim, L, opt, the_type=0):
        super(VAECNN, self).__init__()
        # 编码器
        self.en1 = layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1))
        self.en2 = layers.MaxPooling2D((2, 2))
        self.en3 = layers.Conv2D(16, (5, 5), activation='relu')
        self.en4 = layers.MaxPooling2D((2, 2))
        self.en5 = layers.Conv2D(120, (4, 4), activation='relu')
        self.en6 = layers.Flatten()
        self.en7 = layers.Dense(84, activation='relu')
        # 均值
        self.en8 = layers.Dense(z_dim)
        # 方差
        self.en9 = layers.Dense(z_dim)

        # 解码器
        self.de1 = layers.Dense(84)
        self.de2 = layers.Dense(120)
        self.de3 = layers.Conv2DTranspose(16, (4, 4), activation='relu')
        self.de4 = layers.UpSampling2D((2, 2))
        self.de5 = layers.Conv2DTranspose(6, (5, 5), activation='relu')
        self.de6 = layers.UpSampling2D((2, 2))
        self.de7 = layers.Conv2DTranspose(1, (5, 5))

        self.L = L
        self.opt = opt
        self.type = the_type

    def encoder(self, x):
        # 编码器
        x = self.en1(x)
        x = self.en2(x)
        x = self.en3(x)
        x = self.en4(x)
        x = self.en5(x)
        x = self.en6(x)
        h = self.en7(x)
        # 获取均值
        mu = self.en8(h)
        # 获取方差
        log_var = self.en9(h)

        return mu, log_var

    def decoder(self, z):
        # 解码器
        z = self.de1(z)
        z = self.de2(z)
        z = tf.reshape(z, (-1, 120))
        z = tf.expand_dims(z, 1)
        z = tf.expand_dims(z, 1)
        z = self.de3(z)
        z = self.de4(z)
        z = self.de5(z)
        z = self.de6(z)
        out = self.de7(z)
        return out

    def feedforward(self, inputs, training=None, mask=None):
        # 模型拼装
        mu, log_var = self.encoder(inputs)
        log_var_ex = tf.tile(tf.expand_dims(log_var, 0), (self.L, 1, 1))
        if self.type == 0:
            eps = tf.random.normal(log_var_ex.shape)
        else:
            eps = tf.cast(np.random.standard_cauchy(log_var_ex.shape), dtype=tf.float32)
        std = tf.exp(log_var_ex * 0.5)
        mu_ex = tf.tile(tf.expand_dims(mu, 0), (self.L, 1, 1))
        z = mu_ex + std * eps
        z = tf.reshape(z, (-1, z.shape[2]))
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# 模型构建
class VAEDNN(keras.Model):
    def __init__(self, z_dim, L, opt, the_type=0):
        super(VAEDNN, self).__init__()
        # 编码器
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)      # 获得均值
        self.fc3 = layers.Dense(z_dim)      # 获得均值

        # 解码器
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)
        self.L = L
        self.opt = opt
        self.type = the_type

    def encoder(self, x):
        # 编码器
        h = tf.nn.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def decoder(self, z):
        # 解码器
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        return out

    def feedforward(self, inputs, training=None, mask=None):
        # 模型拼装
        mu, log_var = self.encoder(inputs)
        log_var_ex = tf.tile(tf.expand_dims(log_var, 0), (self.L, 1, 1))
        if self.type == 0:
            eps = tf.random.normal(log_var_ex.shape)
        else:
            eps = tf.cast(np.random.standard_cauchy(log_var_ex.shape), dtype=tf.float32)
        std = tf.exp(log_var_ex * 0.5)
        mu_ex = tf.tile(tf.expand_dims(mu, 0), (self.L, 1, 1))
        z = mu_ex + std * eps
        z = tf.reshape(z, (-1, z.shape[2]))
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# 误差数据
class History():
    def __init__(self):
        self.kl_divs = []
        self.the_loss = []
        self.the_rec_loss = []
        self.kl_divs_tests = []
        self.the_loss_tests = []
        self.the_rec_loss_tests = []

    def update(self, kl_div, rec_loss, my_loss, kl_div_test, rec_loss_test, my_loss_test):
        self.kl_divs.append(kl_div.numpy())
        self.the_loss.append(my_loss.numpy())
        self.the_rec_loss.append(rec_loss.numpy())
        self.kl_divs_tests.append(kl_div_test.numpy())
        self.the_loss_tests.append(my_loss_test.numpy())
        self.the_rec_loss_tests.append(rec_loss_test.numpy())