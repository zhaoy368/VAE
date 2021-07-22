from model import History
import tensorflow as tf
import pandas as pd
from CreatePicture import encode_decode_test, generate_picture, graph, noise_test
import random


# 损失函数
def process(x, L, my_model, theta):
    x_hat, mu, log_var = my_model(x)
    x_hat = tf.reshape(x_hat, (L, -1, x_hat.shape[1]))
    # 平方差
    #rec_loss = tf.square(tf.tile(tf.expand_dims(x, 0), (L, 1, 1))-x_hat)
    # 交叉熵
    rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.tile(tf.expand_dims(x, 0), (L, 1, 1)), logits=x_hat)
    rec_loss = tf.reduce_mean(rec_loss, 0)
    rec_loss = tf.reduce_sum(rec_loss)/x.shape[0]
    # KL散度
    if my_model.type == 0:
        kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
    else:
        kl_div = -0.5 * log_var - 2 * tf.math.log(2.) + tf.math.log((mu**2 - tf.exp(log_var) + 1)**2 + 4*tf.exp(log_var) * mu**2) - tf.math.log((tf.exp(0.5*log_var)-1)**2 + mu**2)
    kl_div = tf.reduce_sum(kl_div) / x.shape[0]

    # 两个误差结合
    my_loss = rec_loss + theta * kl_div
    return kl_div, rec_loss, my_loss, x_hat


# 运行模型
def run_model(epochs, train_db, x_test, path, batchsz, z_dim, my_model, theta, learn_rate, label):
    history = History()
    for epoch in range(epochs):
        epoch += 1
        train_db = train_db.shuffle(batchsz * 5)
        for x in train_db:
            x = tf.reshape(x, [-1, 784])
            with tf.GradientTape() as tape:
                kl_div, rec_loss, my_loss, _ = process(x, my_model.L, my_model, theta)
            grads = tape.gradient(my_loss, my_model.trainable_variables)
            my_model.opt.apply_gradients(zip(grads, my_model.trainable_variables))
        if epoch % 5 == 0:
            # 测试模型
            x_0 = random.sample(list(x_test), 100)
            x = tf.reshape(x_0, [-1, 784])
            kl_div_test, rec_loss_test, my_loss_test, x_hat_logits = process(x, 1, my_model, theta)
            print("epoch:{:<3d} (train) KL divergence:{:.2f} reconstruction loss:{:.2f} loss:{:.2f}".format(epoch, kl_div, rec_loss, my_loss))
            print("          (test)  KL divergence:{:.2f} reconstruction loss:{:.2f} loss:{:.2f}".format(kl_div_test, rec_loss_test, my_loss_test))
            history.update(kl_div, rec_loss, my_loss, kl_div_test, rec_loss_test, my_loss_test)
            if epoch % 100 == 0:
                # 编码解码图片
                encode_decode_test(x_0, x_hat_logits, epoch, path)
                # 噪音图片测试
                #noise_test(x, epoch, path, my_model)
                # 用解码器生成图片
                generate_picture(epoch, path, batchsz, z_dim, my_model, my_model.type)
        if epoch % 100 == 0:
            # 可视化
            graph(history, path, z_dim, batchsz, learn_rate, theta, label)
            # 存储数据
            frame = pd.DataFrame({'kl_divs': history.kl_divs, 'the_loss': history.the_loss, 'the_rec_loss': history.the_rec_loss, 'kl_divs_tests': history.kl_divs_tests, 'the_loss_tests': history.the_loss_tests, 'the_rec_loss_tests': history.the_rec_loss_tests})
            frame.to_csv('{}Z{} B{} L{} T{} DNN.csv'.format(path, z_dim, batchsz, learn_rate, theta), sep=',')
    return history
