# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score


def acc(y_true, y_pred):
    """
    将聚类标签做映射，然后算精度
    参考，匈牙利算法 http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html
    """
    assert y_pred.size == y_true.size
    y_true = y_true.astype(np.int64)
    max_cluster_id = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((max_cluster_id, max_cluster_id), dtype=np.int64)  # cost矩阵
    for yp, yt in zip(y_pred, y_true):
        w[yp, yt] += 1  # 为二分图上的每条边累积权重

    from scipy.optimize import linear_sum_assignment
    src, dst = linear_sum_assignment(w, maximize=True)  # 权重越多的边越容易被标注起来
    # 每条被标注的边 视为 一次成功的映射; 统计 成功映射次数 除以 总映射次数
    # 即最优映射的情况下，有多少比例的聚类簇是被正确分配的
    return sum([w[s, d] for s, d in zip(src, dst)]) / y_pred.size
