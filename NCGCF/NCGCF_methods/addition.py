'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''

__author__ = "Yancheng Chen"

import os
import re
import numpy as np
from sklearn.metrics import roc_auc_score

def ensureDir(path): # 判断路径是否存在，若不存在则进行创建
    p = os.path.dirname(path) # 去掉文件名，返回目录
    if not os.path.exists(p): # 判断括号里的文件是否存在
        os.makedirs(p) # 用于递归创建目录

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100): # 判断是否提前结束
    # assert用于判断一个表达式，在表达式条件为false的时候触发异常
    assert expected_order in ['acc', 'dec'] # 'acc'上升，'dec'下降

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step: # 运行次数大于限制次数
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def mean_k(r, k): # 前k个元素的均值
    assert k >= 1
    r = np.asarray(r)[:k] # 将结构数据转化为ndarray，不会copy该对象
    return np.mean(r)

def dcg_at_k(r, k, method=1): # 折现累积增益
    r = np.asfarray(r)[:k] # 将输入转换为浮点型数组
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))) # 相当于按照顺序乘上系数1/log2(n)
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2))) # 相当于按照顺序乘上系数1/log2(n+1)
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1): # 归一化折现累积增益
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max # 除以最大的折现累积增益

def recall_at_k(r, k, all_pos_num): # 召回率：所有正例中被正确预测出来的比例
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def hit_at_k(r, k): # 判断前k个元素是否有非零值，r中元素均为0或1
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def auc(ground_truth, prediction): # ROC曲线下的面积
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
        # AUC的取值范围在0.5和1之间，AUC越接近1.0，检测方法真实性越高
    except Exception:
        res = 0.
    return res