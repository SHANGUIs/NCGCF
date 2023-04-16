'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''

import addition as add
from NCGCF_methods.parser import parse_args
from NCGCF_methods.load_data import *
import multiprocessing # 多线程管理包
import heapq

cores = multiprocessing.cpu_count() // 2 # cpu线程数

args = parse_args() # 引入参数
Ks = eval(args.Ks) # eval()函数用来执行一个字符串表达式，并返回表达式的值

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size) # 载入数据
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items # 用户项目数量
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test # 训练集与测试集数量
BATCH_SIZE = args.batch_size # 批量大小

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks): # 测试项目评分排序(不含auc评分)
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i] # 测试集项目对应评估分数构建字典

    K_max = max(Ks) # 100
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get) # get返回指定键的值
    # 从item_score中寻找最大的K_max个元素，测试集评分最高的项目

    r = []
    for i in K_max_item_score:
        if i in user_pos_test: # 是否在用户的正测试集中
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test): # 获得auc评分
    item_score = sorted(item_score.items(), key=lambda kv: kv[1]) # 项目评分排序
    item_score.reverse() # 降序
    item_sort = [x[0] for x in item_score] # 项目
    posterior = [x[1] for x in item_score] # 评估分数

    r = [] # 真实类别
    for i in item_sort:
        if i in user_pos_test: # 是否在用户的正测试集中
            r.append(1)
        else:
            r.append(0)
    auc = add.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks): # # 测试项目评分排序(含auc评分)
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks): # 获得预测的表现
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(add.mean_k(r, K)) # 准确率
        recall.append(add.recall_at_k(r, K, len(user_pos_test))) # 召回率
        ndcg.append(add.ndcg_at_k(r, K)) # ndcg
        hit_ratio.append(add.hit_at_k(r, K)) # 是否完全正确(1 or 0)

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x): # 输入是二元的
    rating = x[0] # 用户评分(列表)
    u = x[1] # 用户ID(数字)

    try:
        training_items = data_generator.train_items[u] # 用户的项目训练集
    except Exception:
        training_items = []

    user_pos_test = data_generator.test_set[u] # 用户的项目测试集

    all_items = set(range(ITEM_NUM)) # 所有项目

    test_items = list(all_items - set(training_items)) # 训练集之外作为测试集

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks) # 不含auc评分
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks) # 含auc评分

    return get_performance(user_pos_test, r, auc, Ks)


def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False): # 测试
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)
    # Pool类代表了一个进程池的Worker来执行并行的任务，我们直接调用这个类的方法就可以将任务下发到Worker进程中去

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test # 进行测试的用户
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1 # 测试用户批量大小

    count = 0

    for u_batch_id in range(n_user_batchs): # 相当于对测试用户进行批次划分
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag: # 测试项目集分批次

            n_item_batchs = ITEM_NUM // i_batch_size + 1 # 批次数量
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM)) # 单个批次用户与测试项目对应的矩阵

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.]*len(eval(args.layer_size))})
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch,
                                                              model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                              model.mess_dropout: [0.] * len(eval(args.layer_size))})

        user_batch_rating_uid = zip(rate_batch, user_batch)
        # zip()函数用于将可迭代的对象作为参数，将两个对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        batch_result = pool.map(test_one_user, user_batch_rating_uid) # Pool.map()仅接受一个迭代器参数，输出为字典组成的列表
        count += len(batch_result) # 测试用户数量

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users # 将每个批次的测试结果进行累积，re是字典
            result['recall'] += re['recall']/n_test_users # 对应值为Ks长度的数组
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users # 检验测试用户是否全部测试完成
    pool.close() # 关闭进程池
    return result