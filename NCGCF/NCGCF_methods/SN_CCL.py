'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() # 将TensorFlow 1.x和2.x之间所有不同的全局行为切换为预定的1.x行为

# 负采样版本
def sn_ccl_negative_sample(u_embeddings_final, i_embeddings_final, u_latent_prototype, i_latent_prototype, alpha, beta, neg_num):
    # 用户语义邻域聚类对比
    norm_user_f = tf.nn.l2_normalize(u_embeddings_final)  # 正则化
    norm_user_p = tf.nn.l2_normalize(u_latent_prototype)
    pos_score_user = tf.math.log(tf.sigmoid(tf.reduce_sum(tf.multiply(norm_user_f, norm_user_p), axis=1)))  # tf.multiply()逐元素相乘, axis=1按列求和

    # all_score_user = tf.matmul(norm_user_f, norm_user_p, transpose_a=False, transpose_b=True)
    # neg_score_user = all_score_user - tf.matrix_diag(tf.diag_part(all_score_user))
    # neg_score_user = tf.math.log(tf.sigmoid(tf.negative(tf.reduce_sum(neg_score_user, axis=1))))

    neg_sn_ccl_loss_user = 0
    for i in range(neg_num):
        norm_user_neg = tf.gather(norm_user_p, tf.random.shuffle(tf.range(tf.shape(norm_user_p)[0])))
        neg_score_user = tf.math.log(
            tf.sigmoid(tf.negative(tf.reduce_sum(tf.multiply(norm_user_f, norm_user_neg), axis=1))))
        neg_sn_ccl_loss_user += tf.negative(tf.reduce_sum(neg_score_user))

    pos_sn_ccl_loss_user = tf.negative(tf.reduce_sum(pos_score_user))
    # neg_sn_ccl_loss_user = tf.negative(tf.reduce_sum(neg_score_user))
    sn_ccl_loss_user = pos_sn_ccl_loss_user + beta * neg_sn_ccl_loss_user

    # 项目语义邻域聚类对比
    norm_item_f = tf.nn.l2_normalize(i_embeddings_final)  # 正则化
    norm_item_p = tf.nn.l2_normalize(i_latent_prototype)
    pos_score_item = tf.math.log(tf.sigmoid(tf.reduce_sum(tf.multiply(norm_item_f, norm_item_p), axis=1)))  # tf.multiply()逐元素相乘, axis=1按列求和

    # all_score_item = tf.matmul(norm_item_f, norm_item_p, transpose_a=False, transpose_b=True)
    # neg_score_item = all_score_item - tf.matrix_diag(tf.diag_part(all_score_item))
    # neg_score_item = tf.math.log(tf.sigmoid(tf.negative(tf.reduce_sum(neg_score_item, axis=1))))

    neg_sn_ccl_loss_item = 0
    for i in range(neg_num):
        norm_item_neg = tf.gather(norm_item_p, tf.random.shuffle(tf.range(tf.shape(norm_item_p)[0])))
        neg_score_item = tf.math.log(
            tf.sigmoid(tf.negative(tf.reduce_sum(tf.multiply(norm_item_f, norm_item_neg), axis=1))))
        neg_sn_ccl_loss_item += tf.negative(tf.reduce_sum(neg_score_item))

    pos_sn_ccl_loss_item = tf.negative(tf.reduce_sum(pos_score_item))
    # neg_sn_ccl_loss_item = tf.negative(tf.reduce_sum(neg_score_item))
    sn_ccl_loss_item = pos_sn_ccl_loss_item + beta * neg_sn_ccl_loss_item

    # 总损失
    sn_ccl_loss = sn_ccl_loss_user + alpha * sn_ccl_loss_item

    return sn_ccl_loss