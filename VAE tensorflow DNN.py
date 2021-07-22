from model import VAEDNN
from RunDNN import run_model
import numpy as np
from Auxiliary import GPU
import tensorflow as tf
from pathlib import Path


# 加载数据
def data(data):
    (x_train, _), (x_test, _) = data.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
    train_db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(batchsz * 5).batch(batchsz)
    return train_db, x_test


label = Path(__file__).name
path = './data/{}/6/'.format(label)
# 超参数
z_dim = 10
batchsz = 512
learn_rate = 1e-4
theta = 1
epochs = 500
L = 1

GPU()
mnist = tf.keras.datasets.mnist
train_db, x_test = data(mnist)
my_model = VAEDNN(z_dim, L, opt=tf.optimizers.Adam(learn_rate))
run_model(epochs, train_db, x_test, path, batchsz, z_dim, my_model, theta, learn_rate, label)
