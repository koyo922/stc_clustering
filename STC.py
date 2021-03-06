# -*- coding: utf-8 -*-


import os
import subprocess
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

from rc_utils.misc.log_writer import init_log
from rc_utils.misc.time import timing

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import SGD, Adam
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
        self.n_clusters = n_clusters
        self.ae, self.encoder = autoencoder(dims, init=init)
        clustering_layer = ClusteringLayer(self.n_clusters, alpha, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, epochs=200, batch_size=256, save_dir='results/temp'):
        self.ae.compile(optimizer='adam', loss='mse')
        with timing('pretraining', log_fn=logger.info):
            self.ae.fit(x, x, batch_size=batch_size, epochs=epochs)
        self.ae.save_weights(save_dir + '/ae_weights.h5')
        logger.debug('Pretrained weights are saved to %s/ae_weights.h5', save_dir)

    def predict(self, x):
        return self.model.predict(x, verbose=0).argmax(1)

    @staticmethod
    def sharpen_distribution(q):
        """ 对软聚类结果q 进行锐化 """
        # shape: q [B, 20]
        # shape: q.sum(0) [20]
        # broadcast: weight [B, 20]
        weight = q ** 2 / q.sum(0)
        # shape: weight.T [20, B]
        # shape: weight.sum(1) [B]
        # broadcast: weight.T / weight.sum(1) result [20, B]
        # result: ?.T [B, 20]
        return (weight.T / weight.sum(1)).T

    def fit_predict(self, x, y=None, max_iter=10, batch_size=256, tol=1e-3, save_dir='./results/temp'):
        # Step 1: initialize cluster centers using k-means
        logger.info('Initializing cluster centers with k-means.')
        km = KMeans(n_clusters=self.n_clusters, n_init=100, n_jobs=7)  # 记得开多进程
        centroids = km.fit_predict(self.encoder.predict(x))  # 聚类id不直接使用，仅用作收敛判断条件
        self.model.get_layer(name='clustering').set_weights([km.cluster_centers_])  # 直接用的是簇心位置

        # Step 2: loop: [clustering, sharpening, fit] till convergence
        last_centroids = centroids
        for i in range(max_iter):
            q = self.model.predict(x, verbose=0)  # 软聚类结果
            p = self.sharpen_distribution(q)  # 轻度锐化(差距拉得更开)结果
            centroids = q.argmax(1)  # 重度锐化(硬聚类)结果

            # 如果硬聚类结果变动很小，就判定为收敛了，终止训练
            diff_centroids_frac = np.mean(centroids != last_centroids)
            if i > 0 and diff_centroids_frac < tol:
                logger.info('diff_centroids_frac(%.3f) < tol(%.3f), stop training', diff_centroids_frac, tol)
                break
            else:  # 否则继续训练
                last_centroids = centroids

            # 如果事先有人工标注的类别信息，可以用来打印精度；但是不参与模型训练
            if y is not None:
                acc = metrics.acc(y, centroids)
                nmi = metrics.nmi(y, centroids)
                loss = self.model.fit(x, p, batch_size=batch_size).history['loss'][0]
                logger.info('Iter %d: acc = %.3f, nmi = %.3f, diff_frac=%.3f, loss = %.3f',
                            i, acc, nmi, diff_centroids_frac, loss)

        logger.info('saving model to: %s/STC_model_final.h5', save_dir)
        self.model.save_weights(save_dir + '/STC_model_final.h5')
        return centroids

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
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='stackoverflow',
                        choices=['stackoverflow', 'biomedical', 'search_snippets'])
    parser.add_argument('--ae_weights', default='/data/search_snippets/results/ae_weights.h5')
    parser.add_argument('--save_dir', default='/data/search_snippets/results')
    return parser.parse_args()


if __name__ == "__main__":
    auto_device(n_gpus=1)  # 只用一块显卡
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # load dataset
    X_trn, X_tst, y_trn, y_tst = train_test_split(*load_data(args.dataset),
                                                  test_size=0.1, random_state=0, shuffle=True)

    # create model
    stc = STC(dims=[X_trn.shape[1], 500, 500, 2000, 20], n_clusters=np.unique(y_trn).size)

    # pretrain AutoEncoder-part
    if os.path.exists(args.ae_weights):
        stc.ae.load_weights(args.ae_weights)
    else:
        stc.pretrain(X_trn, epochs=15, batch_size=64, save_dir=args.save_dir)

    # clustering
    stc.model.summary()
    # dec.model.compile(optimizer=SGD(lr=0.1, momentum=0.9), loss='kld')
    stc.model.compile(optimizer=Adam(lr=0.001), loss='kld')  # 换adam可以再提高1个点, acc: 0.60, nmi: 0.59

    y_pred = stc.fit_predict(X_trn, y_trn, batch_size=128, save_dir=args.save_dir)  # batch_size太大会过早收敛
    logger.info('@train_set acc: %.3f  nmi: %.3f', metrics.acc(y_trn, y_pred), metrics.nmi(y_trn, y_pred))

    y_pred = stc.predict(X_tst)
    logger.info('@test_set acc: %.3f  nmi: %.3f', metrics.acc(y_tst, y_pred), metrics.nmi(y_tst, y_pred))
