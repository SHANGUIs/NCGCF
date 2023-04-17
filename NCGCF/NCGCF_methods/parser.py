'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NCGCF.") # 创建一个解析对象
    parser.add_argument('--weights_path', nargs='?', default='../NCGCF/', # 向该对象中添加关注的命令行参数和选项
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Datas/', # 数据路径
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='', # 项目路径
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='mini_gowalla', # 数据集
                        help='Choose a dataset from {mini_gowalla, mini_yelp2018, mini_amazon-book, gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0, # 预训练
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1, # 打印loss间隔
                        help='Interval of evaluation.')
    parser.add_argument('--cycle', type=int, default=3,  # 打印召回率、准确率等指标的间隔
                        help='Interval of printing recall rate, accuracy and other indicators')
    parser.add_argument('--epoch', type=int, default=15, # 训练次数
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=16, # 嵌入大小(64)
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[16, 16]', # 每层输出的大小(64)
                        help='Output sizes of every layer')
    parser.add_argument('--sn_ncl_k', nargs='?', default= 2,  # 结构对比所用偶数层
                        help='Even layers are used for structural contrast')
    parser.add_argument('--batch_size', type=int, default=200, # 批量大小 gowalla:1024
                        help='Batch size.')

    parser.add_argument('--lr', type=float, default=0.01, # 学习率
                        help='Learning rate.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-3,1e-1]', # 正则项系数
                        help='Regularizations.')

    parser.add_argument('--SN_NCL', nargs='?', default=1, # 是否使用SN-NCL损失
                        help='Whether to use SN-NCL loss.')
    parser.add_argument('--weight_sn_ncl', nargs='?', default=1e-5, # SN-NCL损失系数
                        help='Coefficient of SN-NCL loss.')
    parser.add_argument('--sn_ncl_i', nargs='?', default=1.0, # 调节用户和项目损失比例的系数
                        help='Coefficient of user and project loss ratio.')
    parser.add_argument('--sn_ncl_beta', nargs='?', default=1.0,  # 调节负采样中正负例的权重
                        help='Weight of positive and negative case losses in negative sampling.')
    parser.add_argument('--sn_ncl_neg_num', nargs='?', default=5,  # 负采样中负例的数量
                        help='Number of negative cases in a negative sample.')

    parser.add_argument('--SN_CCL', nargs='?', default=1, # 是否使用SN-CCL损失
                        help='Whether to use SN-CCL loss.')
    parser.add_argument('--k_user', nargs='?', default=100,  # 用户的聚类类别数量
                        help='Number of clustering categories of users.')
    parser.add_argument('--k_item', nargs='?', default=200,  # 项目的聚类类别数量
                        help='Number of clustering categories of items.')
    parser.add_argument('--weight_sn_ccl', nargs='?', default=1e-5, # SN-CCL损失系数
                        help='Coefficient of SN-CCL loss.')
    parser.add_argument('--sn_ccl_i', nargs='?', default=1.0, # 调节用户和项目损失比例的系数
                        help='Coefficient of user and project loss ratio.')
    parser.add_argument('--sn_ccl_beta', nargs='?', default=1.0,  # 调节负采样中正负例的权重
                        help='Weight of positive and negative case losses in negative sampling.')
    parser.add_argument('--sn_ccl_neg_num', nargs='?', default=5,  # 负采样中负例的数量
                        help='Number of negative cases in a negative sample.')

    parser.add_argument('--model_type', nargs='?', default='ncgcf', # 模型选择
                        help='Specify the name of model (ncgcf).')
    parser.add_argument('--adj_type', nargs='?', default='norm', # 邻接矩阵(拉普拉斯矩阵)的类型
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='ncgcf', # 图卷积层选择(图滤波)
                        help='Specify the type of the graph convolutional layer from {ncgcf, lightgcn, ngcf, gcn, gcmc}.')

    parser.add_argument('--node_dropout_flag', type=int, default=0, # 点的dropout是否激活
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]', # 点的dropout系数
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]', # 消息的dropout系数
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', # 测试时所用数据大小top-k
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0, # 模型参数储存
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part', # 部分(全部)测试
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0, # 稀疏级别的性能报告
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args() # 进行参数解析
