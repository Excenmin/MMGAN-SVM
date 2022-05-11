# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


def plot_roc_curve(fpr, tpr, auc, std, title, flag):
    if flag == True:
        plt.figure(1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=0.5, color='black', label='Chance', alpha=.8)
    plt.plot(fpr, tpr, color='blue', label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (auc, std), lw=0.5, alpha=1)
    x_major_locator = plt.MultipleLocator(0.1)
    y_major_locator = plt.MultipleLocator(0.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def plot_pr_curve(recall, precision, aupr, std, title, flag):
    if flag == True:
        plt.figure(2)
    # plt.plot([0, 1], [1, 0], linestyle='--', lw=1.8, color='r', label='Chance', alpha=.8)
    plt.plot(recall, precision, color='blue', label=r'Mean PR (AUPR = %0.4f $\pm$ %0.4f)' % (aupr, std), lw=0.5, alpha=1)
    x_major_locator = plt.MultipleLocator(0.1)
    y_major_locator = plt.MultipleLocator(0.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()