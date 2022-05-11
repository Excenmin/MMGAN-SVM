# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC

from utils import plot_roc_curve, plot_pr_curve
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
node_num = M_num + D_num





def plot_learning_curve(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 2)
    plt.legend(loc=1)
    plt.show()



def cross_validation(epochs =2000):
    known, _ = divide_known_unknown_associations(MDA)
    known, unknown = divide_known_unknown_associations(MDA)

    # unknown = np.loadtxt('./data/negative1.txt', delimiter=' ', dtype=int)
    # unknown = np.loadtxt('./data/negative2.txt', delimiter=' ', dtype=int)
    # unknown = np.loadtxt('./data/negative3.txt', delimiter=' ', dtype=int)
    # unknown = np.loadtxt('./data/negative4.txt', delimiter=' ', dtype=int)
    # unknown = np.loadtxt('./data/negative5.txt', delimiter=' ', dtype=int)
    # unknown = np.loadtxt('./data/negative6.txt', delimiter=' ', dtype=int)
    # unknown = np.loadtxt('./data/negative7.txt', delimiter=' ', dtype=int)

    rs = np.random.randint(0, 1000, 1)[0]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    kf = kf.split(known[:, :-1], known[:, -1])
    kf = [item for item in kf]

    neg_index = [i for i in range(len(unknown))]
    random.shuffle(neg_index)
    neg_index = neg_index[: 10 * len(known)]

    unknown = unknown[neg_index]

    kf_n = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    kf_n = kf_n.split(unknown[:, :-1], unknown[:, -1])
    kf_n = [item for item in kf_n]



    plt_roc = []
    plt_pr = []

    tpr_fold = []
    cls_tpr_fold = []
    cls_sens_fold = []
    cls_spec_fold = []
    cls_mcc_fold = []
    cls_f1_fold = []
    cls_acc_fold = []
    cls_recall_fold = []
    precision_fold = []
    prec_fold = []
    cls_precision_fold = []
    aucs = []
    praucs = []

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)


    for i in range(5):

        train_index, test_index = kf[i]

        train = known[train_index]
        test = known[test_index]

        cls_test, cls_val = train_test_split(test[:, :-1], test_size=0.01, random_state=rs)

        cls_train = train[:, :-1]


        cls_train_y = np.concatenate(
            (np.zeros((len(cls_train), 1)), np.ones((len(cls_train), 1))), axis=1)
        cls_val_y = np.concatenate(
            (np.zeros((len(cls_val), 1)), np.ones((len(cls_val), 1))), axis=1)
        cls_test_y = np.concatenate(
            (np.zeros((len(cls_test), 1)), np.ones((len(cls_test), 1))), axis=1)


        neg_train_index, neg_test_index = kf_n[i]
        neg_train = unknown[neg_train_index]
        neg_test = unknown[neg_test_index]
        cls_test_neg, cls_val_neg = train_test_split(neg_test[:, :-1], test_size=0.01, random_state=rs)
        cls_train_neg = neg_train[:, :-1]




        neg_train_y = np.concatenate(
            (np.ones((len(cls_train_neg), 1)), np.zeros((len(cls_train_neg), 1))), axis=1)
        neg_val_y = np.concatenate(
            (np.ones((len(cls_val_neg), 1)), np.zeros((len(cls_val_neg), 1))), axis=1)
        neg_test_y = np.concatenate(
            (np.ones((len(cls_test_neg), 1)), np.zeros((len(cls_test_neg), 1))), axis=1)

        cls_train_all = np.concatenate((cls_train, cls_train_neg), axis=0)
        cls_train_all[:, 1] = cls_train_all[:, 1] + MDA.shape[0]
        cls_train_y_all = np.concatenate((cls_train_y, neg_train_y), axis=0)
        y_train = np.array([cls_train_y_all[i][1] for i in range(len(cls_train_y_all))])

        cls_val_all = np.concatenate((cls_val, cls_val_neg), axis=0)
        cls_val_all[:, 1] = cls_val_all[:, 1] + MDA.shape[0]
        cls_val_y_all = np.concatenate((cls_val_y, neg_val_y), axis=0)

        cls_test_all = np.concatenate((cls_test, cls_test_neg), axis=0)
        cls_test_all[:, 1] = cls_test_all[:, 1] + MDA.shape[0]
        cls_test_y_all = np.concatenate((cls_test_y, neg_test_y), axis=0)



        mask_adj = np.zeros(MDA.shape)
        for ele in train:
            mask_adj[ele[0], ele[1]] = ele[2]

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
        scores = []
        labels = []
        embed = None
        print("------------------------- Fold %d ------------------------- " % (i + 1))
        for epoch in range(epochs):
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
        val_row = tf.gather(embed, cls_val_all[:, 0])
        val_col = tf.gather(embed, cls_val_all[:, 1])
        test_row = tf.gather(embed, cls_test_all[:, 0])
        test_col = tf.gather(embed, cls_test_all[:, 1])


        train_data = np.concatenate((train_row, train_col), axis=1)
        cov_train_data = train_data.reshape(-1, 2, embed.shape[1], 1)
        val_data = np.concatenate((val_row, val_col), axis=1)
        cov_val_data = val_data.reshape(-1, 2, embed.shape[1], 1)
        test_data = np.concatenate((test_row, test_col), axis=1)
        cov_test_data = test_data.reshape(-1, 2, embed.shape[1], 1)


        # dnn = keras.models.Sequential()
        # dnn.add(keras.layers.Dense(256, input_shape=(2 * embed.shape[1],), kernel_regularizer='l2'))
        # dnn.add(keras.layers.BatchNormalization())
        # dnn.add(keras.layers.ReLU())
        #
        # dnn.add(keras.layers.Dense(64, kernel_regularizer='l2'))
        # dnn.add(keras.layers.BatchNormalization())
        # dnn.add(keras.layers.ReLU())
        #
        # dnn.add(keras.layers.Dense(2, activation='softmax', kernel_regularizer='l2'))
        #
        #
        # dnn.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        #     loss=keras.losses.BinaryCrossentropy(),
        #     metrics=[keras.metrics.binary_accuracy])
        #
        # dnn.fit(train_data, cls_train_y_all, batch_size=100, epochs=50, verbose=0, validation_data=(val_data, cls_val_y_all))



        # cnn = keras.models.Sequential()
        # cnn.add(keras.layers.Conv2D(filters=80, kernel_size=(1, 5), input_shape=(2, embed.shape[1], 1)))
        # cnn.add(keras.layers.MaxPool2D(pool_size=(1, 2)))
        # cnn.add(keras.layers.BatchNormalization())
        # cnn.add(keras.layers.ReLU())
        # cnn.add(keras.layers.Conv2D(filters=40, kernel_size=(1, 3), input_shape=(2, embed.shape[1], 1)))
        # cnn.add(keras.layers.MaxPool2D(pool_size=(1, 2)))
        # cnn.add(keras.layers.BatchNormalization())
        # cnn.add(keras.layers.ReLU())
        # # cnn.add(keras.layers.Conv2D(filters=48, kernel_size=(1, 2), input_shape=(2, embed.shape[1], 1)))
        # # cnn.add(keras.layers.MaxPool2D(pool_size=(1, 2)))
        # # cnn.add(keras.layers.BatchNormalization())
        # # cnn.add(keras.layers.ReLU())
        # cnn.add(keras.layers.Flatten())
        # cnn.add(keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'))
        # cnn.add(keras.layers.Dropout(0.5))
        # cnn.add(keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'))
        # cnn.add(keras.layers.Dropout(0.5))
        # # cnn.add(keras.layers.Dense(32, activation='relu', kernel_regularizer='l2'))
        # cnn.add(keras.layers.Dense(2, activation='softmax', kernel_regularizer='l2'))



        # cnn.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=0.001),
        #     loss=keras.losses.BinaryCrossentropy(),
        #     metrics=[keras.metrics.binary_accuracy])
        #
        # cnn.fit(cov_train_data, cls_tra in_y_all, batch_size=100, epochs=50, validation_data=(cov_val_data, cls_val_y_all))


        #
        svm = SVC(kernel='rbf', C=50, gamma='auto', probability=True, cache_size=1000)

        svm.fit(train_data, y_train)


        # clf = RandomForestClassifier(random_state=67, n_estimators=350, oob_score=False, n_jobs=-1)
        # clf.fit(train_data, y_train)
        # pred = clf.predict_proba(test_data)
        pred = svm.predict_proba(test_data)
        # predict_y_proba = np.array(predict_y_proba)

        # pred = dnn.predict(test_data)
        # pred = cnn.predict(cov_test_data)
        # pred_lab = np.argmax(pred, axis=1)

        y_test = np.array([cls_test_y_all[i][1] for i in range(len(cls_test_y_all))])
        pre_probability = np.array([pred[i][1] for i in range(len(pred))])
        # pre_probability = np.array(predict_y_proba)

        cls_fpr, cls_tpr, _ = roc_curve(y_test, pre_probability, drop_intermediate=False)
        roc_score = auc(cls_fpr, cls_tpr)
        aucs.append(roc_score)
        plt_roc.append((cls_fpr, cls_tpr, roc_score))

        cls_interp_tpr = np.interp(mean_fpr, cls_fpr, cls_tpr)
        cls_interp_tpr[0] = 0.0
        cls_tpr_fold.append(cls_interp_tpr)

        cls_precision, cls_recall, pr_thresholds = precision_recall_curve(y_test, pre_probability)
        rank_idx = np.argsort(cls_recall)
        cls_recall = cls_recall[rank_idx]
        cls_precision = cls_precision[rank_idx]
        aupr_score = auc(cls_recall, cls_precision)
        praucs.append(aupr_score)
        plt_pr.append((cls_recall, cls_precision, aupr_score))
        cls_interp_precision = np.interp(mean_recall, cls_recall, cls_precision)
        cls_interp_precision[0] = 1.0
        # plt_pr.append((cls_recall, cls_interp_precision, aupr_score))
        cls_precision_fold.append(cls_interp_precision)


        predicted_score = np.zeros(shape=(len(y_test), 1))
        # print('threshhold-----------', threshold)
        predicted_score[pre_probability >= 0.5] = 1
        predicted_score[pre_probability < 0.5] = 0

        # pre_probability[pre_probability >= 0.5] = 1
        # pre_probability[pre_probability < 0.5] = 0

        tn, fp, fn, tp = confusion_matrix(y_test, predicted_score).ravel()
        sens = tp / (tp + fn)
        cls_sens_fold.append(sens)
        spec = tn / (fp + tn)
        cls_spec_fold.append(spec)
        mcc = matthews_corrcoef(y_test, predicted_score)
        cls_mcc_fold.append(mcc)
        f1 = f1_score(y_test, predicted_score)
        cls_f1_fold.append(f1)
        accuracy = accuracy_score(y_test, predicted_score)
        cls_acc_fold.append(accuracy)
        recall = recall_score(y_test, predicted_score)
        cls_recall_fold.append(recall)
        prec = precision_score(y_test, predicted_score, zero_division=0)
        prec_fold.append(prec)






        print('Acc:{:.5f}   Roc:{:.5f}   Aupr:{:.5f}   Sens:{:.5f}   Spec:{:.5f}  Recall:{:.5f}   Precision:{:.5f}   Mcc:{:.5f}   f1:{:.5f}'.format(
            accuracy, roc_score, aupr_score, sens, spec, recall, prec, mcc, f1))





    cls_mean_tpr = np.mean(cls_tpr_fold, axis=0)
    cls_mean_tpr[-1] = 1.0
    cls_mean_auc = auc(mean_fpr, cls_mean_tpr)

    cls_mean_precision = np.mean(cls_precision_fold, axis=0)
    # cls_mean_precision[-1] = 0.0
    cls_mean_aupr = auc(mean_recall, cls_mean_precision)


    cls_mean_sens = np.mean(cls_sens_fold, axis=0)
    cls_mean_spec = np.mean(cls_spec_fold, axis=0)
    cls_mean_mcc = np.mean(cls_mcc_fold, axis=0)
    cls_mean_f1 = np.mean(cls_f1_fold, axis=0)
    cls_mean_recall = np.mean(cls_recall_fold, axis=0)
    cls_mean_acc = np.mean(cls_acc_fold, axis=0)
    cls_prec_fold = np.mean(prec_fold, axis=0)




    print("")
    print("")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")

    print('5-fold acc %0.6f    5-fold auc %0.6f    5-fold aupr %0.6f    5-fold sens %0.6f    5-fold spec %0.6f    5-fold recall %0.6f    5-fold precision %0.6f    5-fold f1 %0.6f    5-fold mcc %0.6f'
          %(cls_mean_acc, cls_mean_auc, cls_mean_aupr, cls_mean_sens, cls_mean_spec, cls_mean_recall, cls_prec_fold, cls_mean_f1, cls_mean_mcc))

    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")


    std_auc = np.std(aucs)
    std_pruc = np.std(praucs)
    plt.rcParams['figure.dpi'] = 400
    plt.figure(1)
    i = 1
    colr_list = ['#FF0000', '#f47920', '#1d953f', '#FF00FF', '#ffd400']

    for item in plt_roc:
        fpr, tpr, score = item
        plt.plot(fpr, tpr, lw=0.5, alpha=1, color=colr_list[i - 1], linestyle='--',
                 label='ROC fold %d(area=%0.4f)' % (i, score))
        i += 1

    plot_roc_curve(mean_fpr, cls_mean_tpr, cls_mean_auc, std_auc, 'ROC', False)

    plt.close()

    plt.figure(2)
    plt.rcParams['figure.dpi'] = 400
    j = 1
    for item in plt_pr:
        recall, precision, score = item
        plt.plot(recall, precision, lw=0.5, alpha=1, color=colr_list[j - 1], linestyle='--',
                 label='PR fold %d(area=%0.4f)' % (j, score))
        j += 1

    plot_pr_curve(mean_recall, cls_mean_precision, cls_mean_aupr, std_pruc, 'PR', False)


    return cls_mean_acc, cls_mean_auc, cls_mean_aupr, cls_mean_sens, cls_mean_spec, cls_mean_recall, cls_prec_fold, cls_mean_mcc, cls_mean_f1


Acc = []
Roc = []
Aupr = []
Sens = []
Spec = []
Recall = []
Prec = []
Mcc = []
F1 = []

for i in range(10):

    metric = cross_validation()
    Acc.append(metric[0])
    Roc.append(metric[1])
    Aupr.append(metric[2])
    Sens.append(metric[3])
    Spec.append(metric[4])
    Recall.append(metric[5])
    Prec.append(metric[6])
    Mcc.append(metric[7])
    F1.append(metric[8])


Acc_mean = np.mean(Acc)
Acc_std = np.std(Acc)
Roc_mean = np.mean(Roc)
Roc_std = np.std(Roc)
Aupr_mean = np.mean(Aupr)
Aupr_std = np.std(Aupr)
Sens_mean = np.mean(Sens)
Sens_std = np.std(Sens)
Spec_mean = np.mean(Spec)
Spec_std = np.std(Spec)
Recall_mean = np.mean(Recall)
Recall_std = np.std(Recall)
Prec_mean = np.mean(Prec)
Prec_std = np.std(Prec)
Mcc_mean = np.mean(Mcc)
Mcc_std = np.std(Mcc)
F1_mean = np.mean(F1)
F1_std = np.std(F1)



print("")
print("")
print("==========================================================================================================================================================================")

print("Acc: {:.6f} +- {:.6f}    Roc: {:.6f} +- {:.6f}    Aupr: {:.6f} +- {:.6f}    Sens: {:.6f} +- {:.6f}    Spec: {:.6f} +- {:.6f}    Recall: {:.6f} +- {:.6f}    Prec: {:.6f} +- {:.6f}    Mcc: {:.6f} +- {:.6f}    F1: {:.6f} +- {:.6f}".format(
    Acc_mean, Acc_std, Roc_mean, Roc_std, Aupr_mean, Aupr_std, Sens_mean, Sens_std, Spec_mean, Spec_std,
    Recall_mean, Recall_std, Prec_mean, Prec_std, Mcc_mean, Mcc_std, F1_mean, F1_std))

print("==========================================================================================================================================================================")











