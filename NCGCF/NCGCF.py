'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (chenyancheng22@mails.ucas.ac.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''

import tensorflow.compat.v1 as tf

tf.get_logger().setLevel('ERROR') # 不显示ERROR

tf.disable_v2_behavior() # 将TensorFlow 1.x和2.x之间所有不同的全局行为切换为预定的1.x行为

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽INFO+WARNING，输出ERROR+FATAL

from NCGCF_methods.addition import *
from NCGCF_methods.batch_test import *
from SN_NCL import sn_ncl, sn_ncl_negative_sample
from SN_CCL import sn_ccl_negative_sample
from clustering import sn_ccl_k_means

class NCGCF(object):
    def __init__(self, data_config, pretrain_data):
        # 参数设置
        self.model_type = 'ncgcf' # 模型类型
        self.adj_type = args.adj_type # 邻接矩阵(拉普拉斯矩阵)的类型
        self.alg_type = args.alg_type # 图卷积层选择(图滤波)

        self.pretrain_data = pretrain_data # 预训练数据

        self.n_users = data_config['n_users'] # 用户数量
        self.n_items = data_config['n_items'] # 项目数量

        self.n_fold = 100 # 与dropout有关?

        self.norm_adj = data_config['norm_adj'] # 邻接矩阵(npz格式)
        self.n_nonzero_elems = self.norm_adj.count_nonzero() # 非零元素的数量

        self.lr = args.lr # 学习率

        self.emb_dim = args.embed_size # 嵌入向量大小
        self.batch_size = args.batch_size # 批量大小

        self.weight_size = eval(args.layer_size) # 每层输出的大小？(参数为一个列表)
        self.n_layers = len(self.weight_size) # 层数 2
        self.sn_ncl_k = args.sn_ncl_k # 结构对比所用偶数层 2

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers) # 模型类型

        self.regs = eval(args.regs) # 正则项系数
        self.decay = self.regs[0] # 衰减系数


        self.SN_NCL = args.SN_NCL # 是否使用SN-NCL损失
        self.weight_sn_ncl = args.weight_sn_ncl # SN-NCL损失系数
        self.sn_ncl_i = args.sn_ncl_i # 调节用户和项目损失比例的系数
        self.sn_ncl_beta = args.sn_ncl_beta # 调节负采样中正负例的权重
        self.sn_ncl_neg_num = args.sn_ncl_neg_num # 负采样中负例的数量

        self.SN_CCL = args.SN_CCL # 是否使用SN-CCL损失
        self.k_user = args.k_user # 用户的聚类类别数量
        self.k_item = args.k_item  # 项目的聚类类别数量
        self.weight_sn_ccl = args.weight_sn_ccl # SN-CCL损失系数
        self.sn_ccl_i = args.sn_ccl_i # 调节用户和项目损失比例的系数
        self.sn_ccl_beta = args.sn_ccl_beta  # 调节负采样中正负例的权重
        self.sn_ccl_neg_num = args.sn_ccl_neg_num  # 负采样中负例的数量

        self.verbose = args.verbose # 评估间隔

        tf.reset_default_graph() # 重置图保证能正常导入保存的参数
        '''
        *********************************************************
        为输入数据创建占位符 & Dropout设置。
        '''
        # 占位符定义
        self.users = tf.placeholder(tf.int32, shape=(None,)) # 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout
        self.node_dropout_flag = args.node_dropout_flag  # 用'node_dropout_flag'指示是否使用该技术
        self.node_dropout = tf.placeholder(tf.float32, shape=[None]) # 点的dropout系数
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None]) # 消息的dropout系数

        """
        *********************************************************
        设置模型参数 (i.e., 初始化权重)。
        """
        # 模型参数初始化
        self.weights = self._init_weights()

        """
        *********************************************************
        通过图神经网络的消息传递机制计算所有用户和项目的基于图的表示。
        图滤波器及来源:
            1. ncgcf: defined in '基于图神经网络的推荐系统——邻域对比图协同滤波'
            2. lightgcn: defined in 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation', SIGIR2020;
            3. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            4. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            5. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """

        if self.alg_type in ['ncgcf']:
            self.ua_embeddings, self.ia_embeddings, self.all_u_embeddings_k, self.all_i_embeddings_k = self._create_ncgcf_embed()

        elif self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        elif self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        批量建立用户-项目对的最终表示形式。
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users) # self.users为小批次训练采样标签
        self.i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.items)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        if self.SN_NCL:
            self.u_embeddings_k = tf.nn.embedding_lookup(self.all_u_embeddings_k, self.users)  # 对应于小批次随机抽样的k阶嵌入向量
            self.i_embeddings_k = tf.nn.embedding_lookup(self.all_i_embeddings_k, self.items)

        self.u_embeddings_0 = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users) # 对应于小批次随机抽样的初始嵌入向量
        self.i_embeddings_0 = tf.nn.embedding_lookup(self.weights['item_embedding'], self.items)
        self.pos_i_embeddings_0 = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_embeddings_0 = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        # tf.nn.embedding_lookup( params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
        # tf.nn.embedding_lookup( params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
        # 该函数作用是找到要寻找的embedding data中的对应的行下的vector
        # params：由一个tensor或者多个tensor组成的列表(多个tensor组成时，每个tensor除了第一个维度其他维度需相等)；
        # ids：一个类型为int32或int64的Tensor，包含要在params中查找的id；
        # 返回值是一个dense tensor，返回的shape为shape(ids)+shape(params)[1:]

        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True) # 用户-项目对的最终表示形式 y* = e_u^T * e_i

        """
        *********************************************************
        生成预测 & 通过BPR损失、SN-NCL损失和SN-CCL损失进行优化.
        """
        # BPR loss
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.BPR_loss = self.mf_loss + self.emb_loss + self.reg_loss # 推荐损失+嵌入损失+正则损失


        # SN-NCL loss
        self.sn_ncl_loss = tf.constant(0.0, tf.float32, [1])
        if self.SN_NCL:
            # self.sn_ncl_loss = sn_ncl(self.u_embeddings_k, self.i_embeddings_k,
            #                           self.u_embeddings_0, self.i_embeddings_0, self.sn_ncl_i)
            self.sn_ncl_loss = sn_ncl_negative_sample(self.u_embeddings_k, self.i_embeddings_k, self.u_embeddings_0,
                                                      self.i_embeddings_0, self.sn_ncl_i, self.sn_ncl_beta, self.sn_ncl_neg_num)
            self.sn_ncl_loss = self.weight_sn_ncl * self.sn_ncl_loss

        # SN-CCL loss
        self.sn_ccl_loss = tf.constant(0.0, tf.float32, [1])
        if self.SN_CCL:
            self.all_u_latent_prototype = sn_ccl_k_means(self.weights['user_embedding'], self.k_user)
            self.all_i_latent_prototype = sn_ccl_k_means(self.weights['item_embedding'], self.k_item)

            self.u_latent_prototype = tf.nn.embedding_lookup(self.all_u_latent_prototype, self.users)
            self.i_latent_prototype = tf.nn.embedding_lookup(self.all_i_latent_prototype, self.items)

            self.sn_ccl_loss = sn_ccl_negative_sample(self.u_g_embeddings, self.i_g_embeddings, self.u_latent_prototype,
                                                      self.i_latent_prototype, self.sn_ccl_i, self.sn_ccl_beta, self.sn_ccl_neg_num)
            self.sn_ccl_loss = self.weight_sn_ccl * self.sn_ccl_loss

        self.loss = self.BPR_loss + self.sn_ncl_loss + self.sn_ccl_loss


        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) # Adam优化器
        # tf.train.AdamOptimizer()除了利用反向传播算法对权重和偏置项进行修正外，也在运行中不断修正学习率。
        # 本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
        # Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。

    """
    *********************************************************
    初始化权重函数。
    """
    def _init_weights(self):
        all_weights = dict() # 建立空字典

        initializer = tf.keras.initializers.glorot_normal()
        # Glorot正态分布初始化器，也称为Xavier正态分布初始化器。
        # 它从以0为中心，标准差为stddev = sqrt(2 / (fan_in + fan_out))的截断正态分布中抽取样本，
        # 其中fan_in是权值张量中的输入单位的数量，fan_out是权值张量中的输出单位的数量。

        if self.pretrain_data is None: # 是否有预训练数据
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), # 初始器参数是形状
                                                        name='user_embedding') # 用户嵌入的参数
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding') # 项目嵌入的参数
            # tf.Variable()函数用于创建变量(Variable), 变量是一个特殊的张量，其可以是任意的形状和类型的张量。
            print('using xavier initialization') # 随机初始化
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization') # 预训练初始化

        self.weight_size_list = [self.emb_dim] + self.weight_size # 权重size的列表[64,64]/[输入,输出]

        for k in range(self.n_layers): # 消息聚合参数初始化 self.n_layers = 1，k = 0
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    """
    *********************************************************
    矩阵分割函数。
    """
    def _split_A_hat(self, X): # 输入X是矩阵(压缩稀疏行格式，csr_matrix)
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold # 分块数量
        for i_fold in range(self.n_fold): # 相当于按照n_fold大小进行切块
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end])) # 将切块后的矩阵(仍是csr_matrix)转化为稀疏张量加入列表
        return A_fold_hat # 返回列表

    def _split_A_hat_node_dropout(self, X): # 带有点dropout的矩阵分割函数
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero() # 计算非零元素个数
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    """
    *********************************************************
    图滤波器函数。
    """


    def _create_ncgcf_embed(self):
        # L = D^(-1/2)AD^(-1/2)
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        # E^(0) = [eu_0^(0),...,eu_M^(0), ei_0^(0),...ei_N^(0)]
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        # [E^(0),E^(1),...,E^(n_layers)]
        for k in range(0, self.n_layers):
            temp_embed = []
            # E^(k + 1) = L * E^(k)
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            ego_embeddings = tf.concat(temp_embed, 0)

            all_u_embeddings_k = None
            all_i_embeddings_k = None
            if self.SN_NCL == 1 and k == self.sn_ncl_k - 1: # 保存k阶邻居消息的聚合后的节点表示
                all_embeddings_k = ego_embeddings
                all_u_embeddings_k, all_i_embeddings_k = tf.split(all_embeddings_k, [self.n_users, self.n_items], 0)

            all_embeddings += [ego_embeddings]

        # [[eu_0^(0),eu_0^(1),...,eu_0^(n_layers)],...,[ei_N^(0),ei_N^(1),...,ei_N^(n_layers)]]
        all_embeddings = tf.stack(all_embeddings, 1)  # 张量堆叠，此时eu_0^(0),eu_0^(1)分别是独立的张量
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)  # 最终节点表示，可改为加权求和
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings, all_u_embeddings_k, all_i_embeddings_k


    def _create_lightgcn_embed(self):
        # L = D^(-1/2)AD^(-1/2)
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        # E^(0) = [eu_0^(0),...,eu_M^(0), ei_0^(0),...ei_N^(0)]
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        # [E^(0),E^(1),...,E^(n_layers)]
        for k in range(0, self.n_layers):
            temp_embed = []
            # E^(k + 1) = L * E^(k)
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            ego_embeddings = tf.concat(temp_embed, 0)
            all_embeddings += [ego_embeddings]

        # [[eu_0^(0),eu_0^(1),...,eu_0^(n_layers)],...,[ei_N^(0),ei_N^(1),...,ei_N^(n_layers)]]
        all_embeddings = tf.stack(all_embeddings, 1) # 张量堆叠，此时eu_0^(0),eu_0^(1)分别是独立的张量
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False) # 最终节点表示，可改为加权求和
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_ngcf_embed(self): # NGCF
        # L = D^(-1/2)AD^(-1/2)
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        # E^(0) = [eu_0^(0),...,eu_M^(0), ei_0^(0),...ei_N^(0)]
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        # [E^(0),E^(1),...,E^(n_layers)]
        for k in range(0, self.n_layers):
            temp_embed = []
            # L * E^(k)
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)

            # LeakyReLU(W_1 * (L * E^(k)) + b_1)
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # E^(k) ⊙ (L * E^(k))
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)

            # LeakyReLU(W_2 * (E^(k) ⊙ (L * E^(k))) + b_2)
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # E^(k + 1) = LeakyReLU(W_1 * (L * E^(k)) + b_1) + LeakyReLU(W_2 * (E^(k) ⊙ (L * E^(k))) + b_2)
            ego_embeddings = sum_embeddings + bi_embeddings
            # ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k]) # dropout
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1) # l2正则化
            all_embeddings += [norm_embeddings]

        # [[eu_0^(0)-eu_0^(1)-...-eu_0^(n_layers)],...,[ei_N^(0)-ei_N^(1)-...-ei_N^(n_layers)]]
        all_embeddings = tf.concat(all_embeddings, 1) # tf.concat()张量拼接，此时[eu_0^(0)-eu_0^(1)-...-eu_0^(n_layers)]是一个张量
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        # L = (D+I)^(-1/2)(A+I)(D+I)^(-1/2)
        A_fold_hat = self._split_A_hat(self.norm_adj)

        # E^(0) = [eu_0^(0),...,eu_M^(0), ei_0^(0),...ei_N^(0)]
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [embeddings]

        # [E^(0),E^(1),...,E^(n_layers)]
        for k in range(0, self.n_layers):
            temp_embed = []
            # L * E^(k)
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)

            # LeakyReLU(W * (L * E^(k)) + b)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k]) # dropout
            all_embeddings += [embeddings]

        # [[eu_0^(0)-eu_0^(1)-...-eu_0^(n_layers)],...,[ei_N^(0)-ei_N^(1)-...-ei_N^(n_layers)]]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        # L = D^(-1)A
        A_fold_hat = self._split_A_hat(self.norm_adj)

        # E^(0) = [eu_0^(0),...,eu_M^(0), ei_0^(0),...ei_N^(0)]
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = []

        # [E^(1),...,E^(n_layers)]
        for k in range(0, self.n_layers):
            # L * E^(k)
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # LeakyReLU(W * (L * E^(k)) + b)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # W_mlp * (LeakyReLU(W * (L * E^(k)) + b)) + b_mlp
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k]
            # mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k]) # dropout
            all_embeddings += [mlp_embeddings]

        # [[eu_0^(1)-eu_0^(2)-...-eu_0^(n_layers)],...,[ei_N^(1)-ei_N^(2)-...-ei_N^(n_layers)]]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) # tf.multiply()逐元素相乘
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) # axis=1按列求和

        # 节点嵌入表示的L2正则化，张量中的每一个元素进行平方，然后求和，最后乘一个1/2
        regularizer_emb = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer_emb = regularizer_emb / self.batch_size

        # -ln(σ(e_u^T * e_i - e_u^T * e_j))
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi)) # tf.reduce_mean函数用于计算张量tensor沿着指定的数轴(tensor的某一维度)上的的平均值
        # 另一种表示方法mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        # Softplus函数可以看作是ReLU函数的平滑，Softplus(x) = ln(1 + e^x)

        emb_loss = self.decay * regularizer_emb # 嵌入表示的正则化损失

        # reg_loss = tf.constant(0.0, tf.float32, [1]) # 参数的正则化损失，默认为常数0
        regularizer_reg = tf.concat([self.u_embeddings_0, self.pos_i_embeddings_0, self.neg_i_embeddings_0], axis=0) # 参数正则化(初始嵌入向量)
        reg_loss = (self.decay * tf.nn.l2_loss(regularizer_reg)) / self.batch_size

        return mf_loss, emb_loss, reg_loss

    """
    *********************************************************
    矩阵张量转化函数。
    """
    def _convert_sp_mat_to_sp_tensor(self, X): # 输入矩阵格式为csr_matrix
        coo = X.tocoo().astype(np.float32) # 将此矩阵转换为坐标格式，coo_matrix
        indices = np.mat([coo.row, coo.col]).transpose() # 用numpy将坐标转化成矩阵并转置
        return tf.SparseTensor(indices, coo.data, coo.shape) # indices:'numpy.matrix', data:'numpy.ndarray', shape:'tuple'
        # 稀疏张量tf.SparseTensor只记录非零元素的位置和数值，可以减少存储空间
        # 非零元素所在位置的索引，非零元素的数值，矩阵维度

    """
    *********************************************************
    稀疏张量dropout函数。
    """
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        noise_shape = [n_nonzero_elems] # 非零元素个数
        random_tensor = keep_prob # 1-dropout
        random_tensor += tf.random_uniform(noise_shape) # 从一个均匀分布[low,high)中随机采样，默认[0,1)，返回形状为(n_nonzero_elems,)的张量
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool) # tf.cast数据类型转换，tf.floor向下取整
        pre_out = tf.sparse_retain(X, dropout_mask)
        # tf.sparse_retain(sp_input, to_retain)
        # sp_input：输入的SparseTensor带有N个非空元素，to_retain：长度为N的具有M个真值的bool向量
        # 函数返回一个与输入具有相同形状并且有M个非空元素的SparseTensor，它对应于to_retain的真实位置

        return pre_out * tf.div(1., keep_prob) # tf.div点除运算


"""
*********************************************************
加载预训练数据函数。
"""
def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

"""
*********************************************************
NCGCF模型数据及调用。
"""
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict() # 创建字典
    config['n_users'] = data_generator.n_users # 用户数量
    config['n_items'] = data_generator.n_items # 项目数量

    data_generator.print_statistics() # 打印数据：用户数，项目数，交互数，稀疏性，训练集，测试集

    """
    *********************************************************
    生成拉普拉斯矩阵，其中每个位置的元素定义两个连接节点之间的衰减因子(例如，p_ui)。
    """
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat() # 返回的矩阵为压缩稀疏行格式csr_matrix，重复的条目将被汇总

    if args.adj_type == 'plain': # 常规
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm': # 加单位矩阵后归一化，默认
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'mean': # 归一化?
        config['norm_adj'] = mean_adj
        print('use the mean adjacency matrix')

    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0]) # 归一化后加单位矩阵(自连接)
        print('use the ngcf adjacency matrix')


    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data() # 加载预训练数据
    else:
        pretrain_data = None

    model = NCGCF(data_config=config, pretrain_data=pretrain_data) # NCGCF
    # 部分数据由上述定义，另一部分在参数包parser.py中定义

    """
    *********************************************************
    保存模型参数。
    """
    saver = tf.train.Saver() # tf.train.Saver()是一个类，提供了变量、模型(也称图Graph)的保存和恢复模型方法
    # TensorFlow是通过构造Graph的方式进行深度学习，任何操作(如卷积、池化等)都需要operator，保存和恢复操作也不例外
    # 在tf.train.Saver()类初始化时，用于保存和恢复的save和restore operator会被加入Graph
    # TensorFlow将变量保存在二进制checkpoint文件中，这类文件会将变量名称映射到张量值。所以，下列类初始化操作应在搭建Graph时完成

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)]) # 将每层输出的大小以—为间隔输出为字符串，eval()的输入也是字符串
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1) # max_to_keep表明保存的最大checkpoint文件数。
        # 当一个新文件创建的时候，旧文件就会被删掉，如果值为None或0，表示保存所有的checkpoint文件

    config = tf.ConfigProto() # tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
    # tf.ConfigProto()一般用在创建Session的时候，用来对Session进行参数配置
    config.gpu_options.allow_growth = True # 动态申请显存，GPU容量按需慢慢增加
    sess = tf.Session(config=config)
    # TensorFlow里变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行
    # session用于执行命令，对话控制。sess.run()用于执行某一个小图上的功能

    """
    *********************************************************
    重新加载预训练的模型参数。
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)])) # 路径同上

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        # tf.train.get_checkpoint_state：从“检查点”文件返回CheckpointState原型，有model_checkpoint_path和all_model_checkpoint_paths两个属性
        # 其中model_checkpoint_path保存了最新的tensorflow模型文件的路径+文件名，是个字符串。all_model_checkpoint_paths则有未被删除的所有tensorflow模型文件的路径+文件名，是个列表。

        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer()) # 初始化所有全局变量
            saver.restore(sess, ckpt.model_checkpoint_path) # saver.restore()会根据ckpt.model_checkpoint_path自动寻找参数名—值文件进行加载
            # 基于checkpoint文件(ckpt)加载参数时，实际上就是用Saver.restore取代了initializer的初始化
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # 呈现预训练模型的结果
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys()) # 测试集的用户
                ret = test(sess, model, users_to_test, drop_flag=True) # 测试函数，返回一个字典
                cur_best_pre_0 = ret['recall'][0] # 最佳召回率

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer()) # 初始化所有全局变量
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer()) # 初始化所有全局变量
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    获得不同稀疏级别的性能表现。
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split() # 获得稀疏分解，返回的是字符串构成的列表
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type) # 路径
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    训练模型。
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch): # 训练次数
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss, sn_ncl_loss, sn_ccl_loss= 0., 0., 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1 # 批次的数量

        for idx in range(n_batch): # 按批次进行训练
            users, pos_items, neg_items, items = data_generator.sample() # 采样
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_sn_ncl_loss, batch_sn_ccl_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss,
                 model.sn_ncl_loss, model.sn_ccl_loss], # 优化器，各种损失loss
                feed_dict={model.users: users, model.items: items, model.pos_items: pos_items,
                           model.node_dropout: eval(args.node_dropout),
                           model.mess_dropout: eval(args.mess_dropout),
                           model.neg_items: neg_items}) # feed_dict用来设置graph的输入值，先前已经定义了占位符，在这里输入数据即可

            loss += batch_loss # 损失要按批次进行叠加
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss
            sn_ncl_loss += batch_sn_ncl_loss
            sn_ccl_loss += batch_sn_ccl_loss



        if np.isnan(loss) == True: # 损失为NAN进行报错
            print('ERROR: loss is nan.')
            sys.exit()

        # 每args.cycle次训练打印测试评估指标
        if (epoch + 1) % args.cycle != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, sn_ncl_loss, sn_ccl_loss)
                print(perf_str) # 根据args.verbose打印运行时间及loss值
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys()) # 测试用户
        ret = test(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall']) # 召回率是数组形式，下面同理
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, sn_ncl_loss, sn_ccl_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str) # 打印评估指标，即用户在目标项目上的表现y* = e_u^T * e_i，在测试项目中排名最高和最低的数据

        # *********************************************************
        # 以排名最高的召回率为模型训练到最佳的标准，若cur_best_pre_0连续下降3次，则提前停止训练
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][-1], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=3)

        if should_stop == True:
            break

        # *********************************************************
        # 保存用户和项目嵌入进行预训练
        if ret['recall'][-1] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch) # 保存权重参数
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0]) # 找到最高的召回率
    idx = list(recs[:, 0]).index(best_rec_0) # 先把每个数组第一个元素提出组成列表，再返回最高召回率的索引

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf) # 以最高召回率为基准，打印在相同训练后的其他指标(不一定是最高或最佳)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path) # 创建结果文件
    f = open(save_path, 'a') # 打开文件

    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf)) # 写入模型参数与最终结果
    f.close() # 关闭文件
