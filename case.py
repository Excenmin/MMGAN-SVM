# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.svm import SVC
from process import divide_known_unknown_associations, constructHNet
from model import HAN, Loss


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(123)
np.random.seed(67)
MDA = np.loadtxt('./data/MD_mat.txt', dtype=float, delimiter=' ')
save_path = './result/'
M_num = MDA.shape[0]
D_num = MDA.shape[1]
# print(D_num)
node_num = M_num + D_num
Mi = pd.read_excel("./data/MiRNA.xlsx", header=None, names=["miRNA"])
Dis = pd.read_excel("./data/Disease.xlsx", header=None, names=["Disease"])
Mi = Mi['miRNA'].tolist()
Dis = Dis["Disease"].tolist()
mi_map = dict(enumerate(Mi))
dis_map = dict(enumerate(Dis))




def casestudy(case):
    known, unknown = divide_known_unknown_associations(MDA, exception=case)
    neg_index = [i for i in range(len(unknown))]
    random.shuffle(neg_index)
    neg_index = neg_index[: len(known)]
    unknown = unknown[neg_index]

    cls_train = known[:, :-1]
    cls_train_y = np.concatenate(
                (np.zeros((len(cls_train), 1)), np.ones((len(cls_train), 1))), axis=1)

    cls_train_neg = unknown[:, :-1]
    neg_train_y = np.concatenate(
                (np.ones((len(cls_train_neg), 1)), np.zeros((len(cls_train_neg), 1))), axis=1)


    cls_train_all = np.concatenate((cls_train, cls_train_neg), axis=0)
    cls_train_all[:, 1] = cls_train_all[:, 1] + MDA.shape[0]
    cls_train_y_all = np.concatenate((cls_train_y, neg_train_y), axis=0)
    y_train = np.array([cls_train_y_all[i][1] for i in range(len(cls_train_y_all))])



    mask_adj = np.ones(MDA.shape)
    adj1, adj2, adj3, adj_m = constructHNet(mask_adj)
    adj1 = tf.constant(adj1, dtype=tf.float32)
    adj2 = tf.constant(adj2, dtype=tf.float32)
    adj3 = tf.constant(adj3, dtype=tf.float32)
    adj_list = [adj1, adj2, adj3]

    feature = tf.constant(np.eye(adj1.shape[-1])[np.newaxis])
    feature_list = [feature, feature, feature]
    inputs_list = [
        keras.layers.Input(shape=feature.shape[1:], batch_size=1) for i in range(3)]

    model = HAN(inputs_list, adj_list)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    results = None
    embed = None
    for epoch in range(2000):
        with tf.GradientTape() as tape:
            # reconstruct, z_mean, z_log_std, latent, w1 = model(feature_list)
            reconstruct, w1, latent= model(feature_list)
            # A = tf.constant(adj_m, dtype=tf.float32)
            A = tf.constant(MDA, dtype=tf.float32)

            A = tf.reshape(A, [-1])
            # loss = Loss(reconstruct, A, z_mean, z_log_std, w1, node_num)
            loss = Loss(reconstruct, A, w1)
            results = reconstruct
            embed = latent

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if (epoch + 1) % 500 == 0:
            print('Epoch: {}   train_loss: {:.5f}'.format(epoch+1, loss))

    train_row = tf.gather(embed, cls_train_all[:, 0])
    train_col = tf.gather(embed, cls_train_all[:, 1])
    train_data = np.concatenate((train_row, train_col), axis=1)

    case_row = tf.gather(embed, range(MDA.shape[0]))
    mul = len(case_row)
    case_col = tf.reshape(tf.tile(tf.gather(embed, case + MDA.shape[0]), multiples=[mul]), (-1, 256))

    candidate = np.concatenate((case_row, case_col), axis=1)


    svm = SVC(kernel='rbf', C=50, gamma='auto', probability=True, cache_size=1000)

    svm.fit(train_data, y_train)

    pred = svm.predict_proba(candidate)
    scores = np.array([pred[i][1] for i in range(len(pred))])
    rank_idx = np.argsort(scores)
    index = rank_idx[-30:]
    print("for case {}".format(dis_map[case]))

    for idx in index:
        print("{}\t{:0.5f}".format(mi_map[idx], scores[idx]))

    print()


if __name__ =="__main__":

    for _ in range(10):
        casestudy(192)
        random.seed(_ * 13 + 2**_)

