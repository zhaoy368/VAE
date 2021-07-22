from Auxiliary import mkdir
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# 图像存储
def save_image(data, name, path):
    mkdir(path)
    save_img_path = '{}{}.jpg'.format(path, name)
    new_img = np.zeros((280, 280))
    for index, each_img in enumerate(data[:100]):
        row_start = int(index/10) * 28
        col_start = (index % 10)*28
        new_img[row_start:row_start+28, col_start:col_start+28] = each_img

    plt.imsave(save_img_path, new_img, cmap='Greys_r')


# 编码解码测试图像
def encode_decode_test(x, x_hat, epoch, path):
    save_image(x, '{}_label'.format(epoch), path)
    x_hat = tf.sigmoid(x_hat)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    save_image(x_hat, '{}_pre'.format(epoch), path)


# 添加噪音测试
def noise_test(x, epoch, path, my_model):
    x = np.array(x)
    x_0 = tf.reshape(x, [-1, 28, 28])
    save_image(x_0, '{}_origin'.format(epoch), path)
    the_noise = tf.cast(np.random.binomial(1, 0.1, x_0.shape), dtype=tf.float32)
    x_1 = tf.maximum(x_0, the_noise)
    save_image(x_1, '{}_noise'.format(epoch), path)
    x_hat, _, _ = my_model(x)
    x_hat = tf.sigmoid(x_hat)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    save_image(x_hat, '{}_recon'.format(epoch), path)


# 生成新的图像
def generate_picture(epoch, path, batchsz, z_dim, my_model, the_type):
    if the_type == 0:
        z = tf.random.normal((batchsz, z_dim))
    else:
        z = tf.cast(np.random.standard_cauchy((batchsz, z_dim)), dtype=tf.float32)
    logits = my_model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    save_image(x_hat, '{}_random'.format(epoch), path)


# 可视化
def graph(history, path, z_dim, batchsz, learn_rate, theta, label):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(history.kl_divs, label='train')
    plt.plot(history.kl_divs_tests, label='test')
    plt.ylabel('KL divergence')
    plt.legend()
    plt.subplot(132)
    plt.plot(history.the_rec_loss, label='train')
    plt.plot(history.the_rec_loss_tests, label='test')
    plt.ylabel('reconstruction loss')
    plt.legend()
    plt.subplot(133)
    plt.plot(history.the_loss, label='train')
    plt.plot(history.the_loss_tests, label='test')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('{}Z{} B{} L{} T{} {}.png'.format(path, z_dim, batchsz, learn_rate, theta, label))
