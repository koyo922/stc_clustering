# -*- coding: utf-8 -*-

import os
import subprocess

from rc_utils.misc.log_writer import init_log
from rc_utils.misc.time import timing
from time import time

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import Input, Dense

import metrics
from data_loader import load_data

logger = init_log(__name__)

def autoencoder(dims, act=tf.nn.leaky_relu, init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    x = tf.keras.layers.Input(shape=(dims[0],), name='input')
    h = x

    for i in range(n_stacks - 1):
        h = tf.keras.layers.Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
    h = tf.keras.layers.Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)

    y = h
    for i in range(n_stacks - 1, 0, -1):
        y = tf.keras.layers.Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    y = tf.keras.layers.Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return tf.keras.models.Model(inputs=x, outputs=y, name='AE'), tf.keras.models.Model(inputs=x, outputs=h,
                                                                                        name='encoder')


class ClusteringLayer(tf.keras.layers.Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1].value
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.keras.backend.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (tf.keras.backend.sum(
            tf.keras.backend.square(tf.keras.backend.expand_dims(inputs, axis=1) - self.clusters),
            axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.keras.backend.transpose(tf.keras.backend.transpose(q) / tf.keras.backend.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class STC:
    def __init__(self, dims, n_clusters, alpha=1.0, init='glorot_uniform'):
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = tf.keras.models.Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, epochs=200, batch_size=256, save_dir='results/temp'):
        self.autoencoder.compile(optimizer='adam', loss='mse')
        with timing('pretraining', log_fn=logger.info):
            self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        logger.debug('Pretrained weights are saved to %s/ae_weights.h5', save_dir)

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp', rand_seed=None):
        # Step 1: initialize cluster centers using k-means
        logger.info('Initializing cluster centers with k-means.')
        km = KMeans(n_clusters=self.n_clusters, n_init=100)
        y_pred = km.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)  # TODO
        self.model.get_layer(name='clustering').set_weights([km.cluster_centers_])

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)
                y_pred = q.argmax(1)
                acc = metrics.acc(y, y_pred)
                nmi = metrics.nmi(y, y_pred)
                logger.info('Iter %d: acc = %.5f, nmi = %.5f, loss = %.5f', ite, acc, nmi, loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        logger.info('saving model to: %s/STC_model_final.h5', save_dir)
        self.model.save_weights(save_dir + '/STC_model_final.h5')
        return y_pred


def auto_device(forced_gpus: str = None, n_gpus: int = 1):
    """ cuda不可用时，返回'cpu'; 否则，返回最空闲的那一张显卡，例如 'cuda:3' """
    if forced_gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = forced_gpus
        return 'cuda:{}'.format(forced_gpus)
    try:
        lines = subprocess.check_output('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free', shell=True).splitlines()
        best_gpus_id = ','.join(str(g) for g in np.argsort([int(x.split()[2]) for x in lines])[-n_gpus:])
        os.environ['CUDA_VISIBLE_DEVICES'] = best_gpus_id  # 注意启动的时候不要包括较满的卡 e.g. '4,5' 如果5是满的话
        if tf.__version__ >= '2.0':  # tf1.x 需要使用 SessionConfig才能开启memory_growth，比较麻烦
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        return 'cuda:{}'.format(best_gpus_id)
    except subprocess.CalledProcessError:
        return 'cpu'


def get_args():
    # args
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='stackoverflow',
                        choices=['stackoverflow', 'biomedical', 'search_snippets'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--maxiter', default=1000, type=int)
    parser.add_argument('--pretrain_epochs', default=15, type=int)
    parser.add_argument('--update_interval', default=30, type=int)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--ae_weights', default='/data/search_snippets/results/ae_weights.h5')
    parser.add_argument('--save_dir', default='/data/search_snippets/results')
    return parser.parse_args()


if __name__ == "__main__":
    auto_device(n_gpus=1)  # 只用一块显卡
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.dataset == 'search_snippets':
        args.update_interval = 100
        args.maxiter = 100
    elif args.dataset == 'stackoverflow':
        args.update_interval = 500
        args.maxiter = 1500
        args.pretrain_epochs = 12
    elif args.dataset == 'biomedical':
        args.update_interval = 300
    else:
        raise Exception("Dataset not found!")

    # load dataset
    X_trn, X_tst, y_trn, y_tst = train_test_split(*load_data(args.dataset),
                                                  test_size=0.1, random_state=0, shuffle=True)

    # create model
    dec = STC(dims=[X_trn.shape[1], 500, 500, 2000, 20], n_clusters=np.unique(y_trn).size)

    # pretrain AutoEncoder-part
    # if os.path.exists(args.ae_weights):
    #     dec.autoencoder.load_weights(args.ae_weights)
    # else:
    #     dec.pretrain(X_trn, epochs=args.pretrain_epochs, batch_size=args.batch_size, save_dir=args.save_dir)

    dec.autoencoder.load_weights(args.ae_weights)
    # dec.pretrain(X_trn, epochs=args.pretrain_epochs, batch_size=args.batch_size, save_dir=args.save_dir)

    # clustering
    dec.model.summary()
    dec.compile(optimizer=SGD(0.1, 0.9), loss='kld')
    y_pred = dec.fit(X_trn, y_trn, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=args.update_interval, save_dir=args.save_dir, rand_seed=0)
    logger.info('acc: %.3f  nmi: %.3f', metrics.acc(y_trn, y_pred), metrics.nmi(y_trn, y_pred))
