# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops import array_ops
from tf_agents import keras_layers
import scipy.sparse as sp
import numpy as np

n_heads = [8, 1]
hid_units = [32]
MDA = np.loadtxt('./data/MD_mat.txt', dtype=float, delimiter=' ')

class attn_head(keras.Model):

    def __init__(self, output_size, adj, activation=tf.nn.elu, in_drop=0, coef_drop=0,
                 residual=False, return_coef=False):
        self.output_size = output_size
        self.adj = adj
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.return_coef = return_coef

        super(attn_head, self).__init__()
        self.conv1 = keras.layers.Conv1D(self.output_size, 1, use_bias=False)
        self.conv2 = keras.layers.Conv1D(1, 1)
        self.conv3 = keras.layers.Conv1D(1, 1)
        self.add_bias = keras_layers.BiasLayer()


    def call(self, inputs):
        with tf.name_scope('attn_head'):
            if self.in_drop != 0:
                inputs = tf.nn.dropout(inputs, self.in_drop)

            inputs_fts = self.conv1(inputs)
            # print('芜湖', inputs_fts.shape)

            f_1 = self.conv2(inputs_fts)
            # print('Nice!!!!!', f_1.shape)
            f_2 = self.conv3(inputs_fts)

            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            # print('ICU！！！！！', logits.shape)
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.adj)

            if self.coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, self.coef_drop)
            if self.in_drop != 0.0:
                inputs_fts = tf.nn.dropout(inputs_fts, self.in_drop)

            vals = tf.matmul(coefs, inputs_fts)
            ret = self.add_bias(vals)

            if self.residual:
                if inputs.shape[-1] != ret.shape[-1]:
                    pass
                    # ret = ret + keras.layers.Conv1D(ret.shape[-1], 1)
                else:
                    inputs_fts = ret + inputs

            if self.return_coef:
                return self.activation(ret), coefs
            else:
                return self.activation(ret)

class SimpleAttLayer(keras.Model):

    def __init__(self, inputs_size, attention_size, time_major=False, return_alphas=False):
        self.inputs_size = inputs_size
        self.attention_size = attention_size
        self.time_major = time_major
        self.return_alphas = return_alphas
        super(SimpleAttLayer, self).__init__()
        self.b = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
        self.w = tf.Variable(tf.random.normal([self.inputs_size, self.attention_size], stddev=0.1))
        self.u = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

    def call(self, inputs):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)

        if self.time_major:
            inputs = array_ops.transpose(inputs, [1, 0, 2])


        self.v = tf.nn.tanh(tf.tensordot(inputs, self.w, axes=1) + self.b)

        vu = tf.tensordot(self.v, self.u, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not self.return_alphas:
            return output
        else:
            return output, alphas


class InnerProductDecoder(keras.Model):

    def __init__(self, inputs_size, num_r, act=tf.nn.sigmoid):
        self.inputs_size = inputs_size
        self.num_r = num_r
        self.act = act
        super(InnerProductDecoder, self).__init__()
        self.w = tf.Variable(tf.random.normal([self.inputs_size, self.inputs_size], stddev=0.1), trainable=True)

    def call(self, inputs):
        R = inputs[0:self.num_r, :]
        D = inputs[self.num_r:, :]
        R = tf.matmul(R, self.w)
        D = tf.transpose(D)
        x = tf.matmul(R, D)
        # inputs_T = tf.transpose(inputs)
        # x = tf.matmul(inputs, self.w)
        # x = tf.matmul(x, inputs_T)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)

        return outputs, tf.keras.regularizers.L2()(self.w)


def Hete_attn(inputs, out, adj, n_heads, hid_units):
    attns1 = []
    for _ in range(n_heads[0]):
        attns1.append(attn_head(output_size=out, adj=adj)(inputs))

    concat1 = tf.keras.layers.concatenate(attns1)
    for i in range(1, len(hid_units)):
        hold = concat1
        attns = []
        for _ in range(n_heads[i]):
            attns.append(attn_head(output_size=out, adj=adj)(concat1))

        concat1 = tf.keras.layers.concatenate(attns)

    return concat1

def HAN(inputs_list, adj_list):
    embed_list = []
    for inputs, adj in zip(inputs_list, adj_list):
        concat = Hete_attn(inputs, 32, adj, n_heads=n_heads, hid_units=hid_units)
        embed_list.append(tf.expand_dims(tf.squeeze(concat), axis=1))

    concat_all = keras.layers.concatenate(embed_list, axis=1)

    simple = SimpleAttLayer(inputs_size=256, attention_size=512)(concat_all)

    # z_mean = SimpleAttLayer(inputs_size=64, attention_size=128)(concat_all)
    # z_log_std = SimpleAttLayer(inputs_size=64, attention_size=128)(concat_all)
    # z = z_mean + tf.random.normal(z_log_std.shape) * tf.exp(z_log_std)
    #
    # z_mean = keras.layers.Dense(units=64, activation=None)(simple)
    # z_log_std = keras.layers.Dense(units=64, activation=None)(simple)
    # z = z_mean + tf.random.normal(z_log_std.shape) * tf.exp(z_log_std)


    # res, w1 = InnerProductDecoder(inputs_size=z.shape[-1], num_r=MDA.shape[0], act=lambda x: x)(z)
    out = []
    for i in range(n_heads[-1]):
        out.append(keras.layers.Dense(units=256, activation=None)(simple))

    logits = tf.add_n(out) / n_heads[-1]
    # # logits = tf.expand_dims(logits, axis=0)

    res, w1 = InnerProductDecoder(inputs_size=logits.shape[-1], num_r=MDA.shape[0], act=lambda x: x)(logits)


    # model = keras.models.Model(inputs=[inputs_list], outputs=[res, z_mean, z_log_std, z, w1])

    model = keras.models.Model(inputs=[inputs_list], outputs=[res, w1, logits])

    return model




# def Loss(pred, label, z_mean, z_log_std, w1, node_num):
def Loss(pred, label, w1):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    rec_loss = 0.5 * tf.reduce_mean(loss(label, pred))
    # kl = (0.5 / node_num) * tf.reduce_mean(
    #                         tf.reduce_sum(
    #                             1 + 2 * z_log_std - tf.square(z_mean) - tf.square(tf.exp(z_log_std)), 1))
    #
    # kl = (0.5 / node_num) * tf.reduce_mean(
    #                         tf.reduce_sum(
    #                             1 + z_log_std - tf.square(z_mean) - tf.exp(z_log_std), 1))

    # latent_loss = 0.5 * tf.reduce_sum(
    #                     tf.exp(coder_gamma) + tf.square(coder_mean) - 1 - coder_gamma)
    # cost = rec_loss + kl + w1
    cost = rec_loss + w1

    return cost








