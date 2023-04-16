'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() # 将TensorFlow 1.x和2.x之间所有不同的全局行为切换为预定的1.x行为

# softmax版本
def sn_ncl(u_embeddings_k, i_embeddings_k, u_embeddings_0, i_embeddings_0, alpha):
    # -∑ln(softmax(e_u^k, e_u^0)) 用户同质结构邻居对比学习
    norm_user_k = tf.nn.l2_normalize(u_embeddings_k) # 正则化
    norm_user_0 = tf.nn.l2_normalize(u_embeddings_0)
    pos_score_user = tf.reduce_sum(tf.multiply(norm_user_k, norm_user_0), axis=1)  # tf.multiply()逐元素相乘, axis=1按列求和
    all_score_user = tf.matmul(norm_user_k, norm_user_0, transpose_a=False, transpose_b=True)
    pos_score_user_exp = tf.math.exp(pos_score_user)
    all_score_user_exp = tf.reduce_sum(tf.math.exp(all_score_user), axis=1)
    sn_ncl_loss_user = tf.negative(tf.reduce_sum(tf.math.log(tf.math.divide(pos_score_user_exp, all_score_user_exp))))

    # -∑ln(softmax(e_i^k, e_i^0)) 项目同质结构邻居对比学习
    norm_item_k = tf.nn.l2_normalize(i_embeddings_k)  # 正则化
    norm_item_0 = tf.nn.l2_normalize(i_embeddings_0)
    pos_score_item = tf.reduce_sum(tf.multiply(norm_item_k, norm_item_0), axis=1)  # tf.multiply()逐元素相乘, axis=1按列求和
    all_score_item = tf.matmul(norm_item_k, norm_item_0, transpose_a=False, transpose_b=True)
    pos_score_item_exp = tf.math.exp(pos_score_item)
    all_score_item_exp = tf.reduce_sum(tf.math.exp(all_score_item), axis=1)
    sn_ncl_loss_item = tf.negative(tf.reduce_sum(tf.math.log(tf.math.divide(pos_score_item_exp, all_score_item_exp))))

    sn_ncl_loss = sn_ncl_loss_user + alpha * sn_ncl_loss_item
    return sn_ncl_loss

# 负采样版本
def sn_ncl_negative_sample(u_embeddings_k, i_embeddings_k, u_embeddings_0, i_embeddings_0, alpha, beta, neg_num):
    # 用户噪声对比估计
    # norm_user_k = tf.nn.l2_normalize(u_embeddings_k)  # 正则化
    # norm_user_0 = tf.nn.l2_normalize(u_embeddings_0)
    norm_user_k = u_embeddings_k
    norm_user_0 = u_embeddings_0
    pos_score_user = tf.math.log(tf.sigmoid(tf.reduce_sum(tf.multiply(norm_user_k, norm_user_0), axis=1)))  # tf.multiply()逐元素相乘, axis=1按列求和
    # all_score_user = tf.matmul(norm_user_k, norm_user_0, transpose_a=False, transpose_b=True)
    # neg_score_user = all_score_user - tf.matrix_diag(tf.diag_part(all_score_user))
    # neg_score_user = tf.math.log(tf.sigmoid(tf.negative(tf.reduce_sum(neg_score_user, axis=1))))
    # 以上三行代码是考虑所有其他向量作为负例并在log函数内部加和

    neg_sn_ncl_loss_user = 0
    for i in range(neg_num):
        norm_user_neg = tf.gather(norm_user_0, tf.random.shuffle(tf.range(tf.shape(norm_user_0)[0])))
        # norm_user_neg = tf.random_shuffle(norm_user_0) 这是错误的，随机混洗是没有梯度的
        neg_score_user = tf.math.log(tf.sigmoid(tf.negative(tf.reduce_sum(tf.multiply(norm_user_k, norm_user_neg), axis=1))))
        neg_sn_ncl_loss_user += tf.negative(tf.reduce_sum(neg_score_user))

    pos_sn_ncl_loss_user =  tf.negative(tf.reduce_sum(pos_score_user))
    # neg_sn_ncl_loss_user = tf.negative(tf.reduce_sum(neg_score_user))
    sn_ncl_loss_user = pos_sn_ncl_loss_user + beta * neg_sn_ncl_loss_user

    # 项目噪声对比估计
    norm_item_k = tf.nn.l2_normalize(i_embeddings_k)  # 正则化
    norm_item_0 = tf.nn.l2_normalize(i_embeddings_0)
    pos_score_item = tf.math.log(tf.sigmoid(tf.reduce_sum(tf.multiply(norm_item_k, norm_item_0), axis=1)))  # tf.multiply()逐元素相乘, axis=1按列求和
    # all_score_item = tf.matmul(norm_item_k, norm_item_0, transpose_a=False, transpose_b=True)
    # neg_score_item = all_score_item - tf.matrix_diag(tf.diag_part(all_score_item))
    # neg_score_item = tf.math.log(tf.sigmoid(tf.negative(tf.reduce_sum(neg_score_item, axis=1))))

    neg_sn_ncl_loss_item = 0
    for i in range(neg_num):
        norm_item_neg = tf.gather(norm_item_0, tf.random.shuffle(tf.range(tf.shape(norm_item_0)[0])))
        # norm_user_neg = tf.random_shuffle(norm_user_0) 这是错误的，随机混洗是没有梯度的
        neg_score_item = tf.math.log(tf.sigmoid(tf.negative(tf.reduce_sum(tf.multiply(norm_item_k, norm_item_neg), axis=1))))
        neg_sn_ncl_loss_item += tf.negative(tf.reduce_sum(neg_score_item))

    pos_sn_ncl_loss_item = tf.negative(tf.reduce_sum(pos_score_item))
    #neg_sn_ncl_loss_item = tf.negative(tf.reduce_sum(neg_score_item))
    sn_ncl_loss_item = pos_sn_ncl_loss_item + beta * neg_sn_ncl_loss_item

    # 总损失
    sn_ncl_loss = sn_ncl_loss_user + alpha * sn_ncl_loss_item

    return sn_ncl_loss