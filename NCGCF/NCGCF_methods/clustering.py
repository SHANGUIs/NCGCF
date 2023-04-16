'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''
from random import shuffle
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() # 将TensorFlow 1.x和2.x之间所有不同的全局行为切换为预定的1.x行为

# k-means聚类算法
def sn_ccl_k_means(embeddings, k):
    '''
    :param embeddings: 用户或项目的初始节点嵌入，也是NCGCF模型的参数，属性为tf.Variable
    :param k: 聚类的类别数量，属性为int
    '''
    # 将输入的节点嵌入从tf.Variable类型转变为np.ndarray
    with tf.Session() as sess0:
        sess0.run(tf.global_variables_initializer())
        embeddings_ndarray = embeddings.eval()

    # 聚类的类别数量若超出输入的节点嵌入的数量则报错
    k = int(k)
    assert k < len(embeddings_ndarray)

    def kmeans(data, k, normalize=False, limit=500):
        # normalize 数据
        if normalize:
            stats = (data.mean(axis=0), data.std(axis=0))
            data = (data - stats[0]) / stats[1]

        # 随机选取k个嵌入向量作为聚类中心
        vector_indices = list(range(len(data)))
        shuffle(vector_indices)
        centers = np.array([data[i] for i in vector_indices[:k]])

        for i in range(limit):
            # 首先利用广播机制计算每个嵌入向量到聚类中心的距离，之后根据最小距离重新分类
            classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :]) ** 2).sum(axis=1), axis=1)
            # 对每个新的集群计算聚类中心
            new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])

            # 聚类中心不再移动的话，结束循环
            if (new_centers == centers).all():
                break
            else:
                centers = new_centers
        else:
            # 如果在for循环里正常结束，下面不会执行
            raise RuntimeError(f"聚类算法无法在{limit}次迭代内完成")

        # 如果对数据进行了normalize，聚类中心需要反向伸缩到原来的大小
        if normalize:
            centers = centers * stats[1] + stats[0]

        return classifications, centers

    # 运用kmeans算法求出聚类中心的向量表示和每个节点的分类结果
    assignments, centroids = kmeans(embeddings_ndarray, k)
    # 每个节点的隐藏语义(所属集群的聚类中心)
    latent_prototype = tf.convert_to_tensor([centroids[i].tolist() for i in assignments])

    return latent_prototype

'''
**************************************************************************
算法演示

# 随机生成大小为(2000,16)的嵌入向量矩阵，这与mini_gowalla数据集中的用户向量嵌入表示矩阵的大小相同
embeddings_test = tf.Variable(np.random.random(size=(2000,16)).astype(np.float32))
ans = sn_ccl_k_means(embeddings_test, 100)
print(ans)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(ans.eval()) # 输出结果(tensor类型)的数据

'''