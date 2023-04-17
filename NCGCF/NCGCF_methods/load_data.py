'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        # 获取用户和项目的数量
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines(): # readlines()返回一个列表，其中包含文件中的每一行作为列表项
                if len(l) > 0:
                    l = l.strip('\n').split(' ') # split()通过指定分隔符对字符串进行切片
                    # strip()是行首尾处理函数，可以删掉指定的行首或行尾字符串，默认删掉行首和行尾的换行符'\n'
                    items = [int(i) for i in l] # 将文件每一行的元素转化为列表
                    uid = items[0]
                    self.exist_users.append(uid) # 用户ID
                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items)) # 最大值即为数量
                    self.n_train += len(items) # 训练集边总数

        self.exist_items = [i for i in range(self.n_items)]

        with open(test_file) as f:
            for l in f.readlines(): # readlines()返回一个列表，其中包含文件中的每一行作为列表项
                if len(l) > 0:
                    l = l.strip('\n').split(' ') # split()通过指定分隔符对字符串进行切片
                    # strip()是行首尾处理函数，可以删掉指定的行首或行尾字符串，默认删掉行首和行尾的换行符'\n'
                    try:
                        items = [int(i) for i in l] # 将文件每一行的元素转化为列表
                    except Exception: # 报错
                        continue
                    uid = items[0]
                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items)) # 最大值即为数量
                    self.n_test += len(items) # 测试集边总数
        self.n_items += 1
        self.n_users += 1

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        # 基于字典的稀疏矩阵存储方式，key由非零元素的的坐标值tuple(row, column)组成，value则代表数据值
        # dok matrix非常适合于增量构建稀疏矩阵，并一旦构建，就可以快速地转换为coo_matrix

        self.train_items, self.test_set = {}, {} # 创建字典
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:] # 第一列为uid

                    for i in train_items:
                        self.R[uid, i] = 1. # 构造图结构的矩阵形式
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items # 训练集项目集合与用户ID构建字典

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        self.R_csr = self.R.tocsr()

    def get_adj_mat(self): # 获取邻接矩阵
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz') # 使用npz格式从文件加载稀疏矩阵
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1) # 加载邻接矩阵

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat, SecondOrder_adj_mat = self.create_adj_mat() # 创建邻接矩阵
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat) # 使用npz格式保存稀疏矩阵文件
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self): # 创建邻接矩阵
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil() # tolil()用于将矩阵转为列表
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R # 用户对应项目             0   R
        adj_mat[self.n_users:, :self.n_users] = R.T # 项目对应用户(对称性)    R^T 0
        adj_mat = adj_mat.todok() # 将此矩阵转换为键字典格式
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj): # 归一化邻接矩阵
            rowsum = np.array(adj.sum(1)) # sum(1)求数组每一行的和

            d_inv = np.power(rowsum, -1).flatten() # 默认是按行的方向降为一维数组
            d_inv[np.isinf(d_inv)] = 0. # 去除无穷元素
            d_mat_inv = sp.diags(d_inv) # 构造对角矩阵

            norm_adj = d_mat_inv.dot(adj) # 矩阵乘法
            # norm_adj = adj.dot(d_mat_inv)

            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo() # 将此矩阵转换为坐标格式

        def check_adj_if_equal(adj): # 检查归一化邻接矩阵是否等于拉普拉斯矩阵
            dense_A = np.array(adj.todense()) # 将稀疏矩阵转为稠密矩阵
            degree = np.sum(dense_A, axis=1, keepdims=False) # 按行求和得到每个节点的度

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A) # D^(-1)*A
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0])) # 加单位矩阵
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
        # tocsr()将此矩阵转换为压缩稀疏行格式，重复的条目将被汇总在一起

    def negative_pool(self): # 创建负样本集
        t1 = time()
        for u in self.train_items.keys(): # 用户ID
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u])) # 集合差集，即该用户没有交互的项目
            pools = [rd.choice(neg_items) for _ in range(100)] # 随机抽取组成组成负样本集
            self.neg_pools[u] = pools # 为每个用户构造一个负样本集
        print('refresh negative pools', time() - t1)

    def sample(self): # 采样
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size) # 不重复随机采样
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)] # 随机抽取用户(有重复)

        if self.batch_size <= self.n_items:
            items = rd.sample(self.exist_items, self.batch_size) # 不重复随机采样
        else:
            items = [rd.choice(self.exist_items) for _ in range(self.batch_size)] # 随机抽取用户(有重复)


        def sample_pos_items_for_u(u, num): # 采样正样本集
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0] # 随机整数，high是上限
                pos_i_id = pos_items[pos_id] #

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id) # 随机添加(不重复)
            return pos_batch

        def sample_neg_items_for_u(u, num): # 采样负样本集
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num): # 从负样本池中随机不重复采样负样本
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1) # 每个用户随机取样一次
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items, items

    def get_num_users_items(self): # 用户项目数量
        return self.n_users, self.n_items

    def print_statistics(self): # 打印数据
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self): # 获得稀疏分解
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
            # enumerate()将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self): # 创建稀疏分解
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # 生成一个字典用来存储
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # 把用户集分成四个子集
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
