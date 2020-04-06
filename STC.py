# -*- coding: utf-8 -*-

import os
import subprocess

from rc_utils.misc.log_writer import init_log
from rc_utils.misc.time import timing

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import Input, Dense, InputSpec, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

import metrics
from data_loader import load_data

logger = init_log(__name__)


def autoencoder(dims, act=tf.nn.leaky_relu, init='glorot_uniform'):
    # 编码部分 shape = [48(略), 500, 500, 2000, 20] 从48维的词向量空间 编码到 20维的聚类空间
    h = x = Input(shape=(dims[0],), name='input')
    for i, d in enumerate(dims[1:]):  # 注意最后一层 activation=None
        y = h = Dense(d, kernel_initializer=init, name=f'encoder_{i}',
                      activation=None if i == len(dims) - 2 else act)(h)

    # 解码部分 shape = [20(略), 2000, 500, 500, 48] 从20维的聚类空间 解码到 48维的词向量空间
    for i, d in reversed(list(enumerate(dims[:-1]))):
        y = Dense(d, kernel_initializer=init, name=f'decoder_{i}',
                  activation=None if i == 0 else act)(y)

    auto_encoder = Model(inputs=x, outputs=y, name='AE')
    encoder_part = Model(inputs=x, outputs=h, name='encoder')
    return auto_encoder, encoder_part


def patch_tf_ops():
    """
    为TF Tensor增加一些常用运算符的语法糖; 减少嵌套层数, 写法更直观
    e.g. K.expand_dims(t, axis=1) --> t.exp_dim(axis=1)
    """
    from tensorflow.python.framework import ops

    def _exp_dim(self, **kwargs):
        return K.expand_dims(self, **kwargs)

    def _sum(self, **kwargs):
        return K.sum(self, **kwargs)

    def _square(self):
        return K.square(self)

    def _div(self, other):
        return self / other

    def _inv(self):
        return 1 / self

    @property
    def _T(self):
        return K.transpose(self)

    ops.Tensor.exp_dim = _exp_dim
    ops.Tensor.sum = _sum
    ops.Tensor.square = _square
    ops.Tensor.div = _div
    ops.Tensor.inv = _inv
    ops.Tensor.T = _T


patch_tf_ops()


class ClusteringLayer(Layer):
    """ 内部维护各簇心; 对于各输入点，根据L2距离，求其软聚类id """

    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)  # name
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_spec = None
        self.centroids = None

    def build(self, input_shape):  # 根据运行时输入数据确定具体的形状
        assert len(input_shape) == 2
        input_dim = input_shape[1].value
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.centroids = self.add_weight(shape=(self.n_clusters, input_dim),
                                         initializer='glorot_uniform', name='clusters')
        super().build(input_shape)  # self.built = True

    def call(self, inputs, **kwargs):
        # shape: inputs [B, 20] ---(expand)--> [B, 1, 20]
        # shape: centroids [20, 20]
        # broadcast: [B, 20, 20]
        # square error summed over axis=2: [B, 20] 表示 每个样本与每个簇心的L2距离
        q = 1 / (1 + (inputs.exp_dim(axis=1) - self.centroids).square().sum(axis=2).div(self.alpha))
        q **= (self.alpha + 1) / 2  # alpha 默认为1，暂不考虑

        # shape: q.T [20, B] 每个簇心与每个样本的L2距离
        # shape: q.sum(axis=1) [B] 每个样本到所有簇心距离之和
        # broadcast: [20, B]
        # shape: (q.T / q.sum()).T [B, 20] 每个样本的软聚类标签
        q = (q.T / q.sum(axis=1)).T
        return q


class STC:
    def __init__(self, dims, n_clusters, alpha=1.0, init='glorot_uniform'):
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.ae, self.encoder = autoencoder(self.dims, init=init)

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, epochs=200, batch_size=256, save_dir='results/temp'):
        self.ae.compile(optimizer='adam', loss='mse')
        with timing('pretraining', log_fn=logger.info):
            self.ae.fit(x, x, batch_size=batch_size, epochs=epochs)
        self.ae.save_weights(save_dir + '/ae_weights.h5')
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
    if os.path.exists(args.ae_weights):
        dec.ae.load_weights(args.ae_weights)
    else:
        dec.pretrain(X_trn, epochs=args.pretrain_epochs, batch_size=args.batch_size, save_dir=args.save_dir)

    # clustering
    dec.model.summary()
    dec.compile(optimizer=SGD(0.1, 0.9), loss='kld')
    y_pred = dec.fit(X_trn, y_trn, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=args.update_interval, save_dir=args.save_dir, rand_seed=0)
    logger.info('acc: %.3f  nmi: %.3f', metrics.acc(y_trn, y_pred), metrics.nmi(y_trn, y_pred))
