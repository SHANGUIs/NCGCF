'''
邻域对比图协同滤波(NCGCF)的TensorFlow实现
    作者：山东大学 数学学院 陈彦丞 (201900090074@mail.sdu.edu.cn)
    指导教师：山东大学数据科学研究院 曲存全
    校外指导教师：中国科学院大学数学科学学院 姜志鹏
    时间：2023.3.28
'''

import random
import os

def miniDataSet(orgin_data_path, use_num, item_num):
    orgin_train_file = orgin_data_path+ '/train.txt'
    orgin_test_file = orgin_data_path + '/test.txt'

    use_list = [int(i) for i in range(0,int(use_num))]
    item_list = [int(i) for i in range(0,int(item_num))]

    file = open(orgin_train_file.replace('gowalla', 'mini_gowalla'), 'w').close()  # 提前清除数据
    file_1 = open(orgin_test_file.replace('gowalla', 'mini_gowalla'), 'w').close()  # 提前清除数据

    with open(orgin_train_file) as f: # 抽取训练集
        for l in f.readlines():  # readlines()返回一个列表，其中包含文件中的每一行作为列表项
            if len(l) > 0:
                l = l.strip('\n').split(' ')  # split()通过指定分隔符对字符串进行切片
                # strip()是行首尾处理函数，可以删掉指定的行首或行尾字符串，默认删掉行首和行尾的换行符'\n'
                if int(l[0]) in use_list:
                    items_train = [int(i) for i in l[1:] if int(i) in item_list]
                    if len(items_train) > 0:
                        with open(orgin_train_file.replace('gowalla','mini_gowalla'), 'a') as f_1:
                            f_1.write(l[0])
                            for k in items_train:
                                f_1.write(' ' + str(k))
                            f_1.write('\n')
                    else:
                        with open(orgin_train_file.replace('gowalla','mini_gowalla'), 'a') as f_1:
                            f_1.write(l[0] + ' ' + str(random.choice(item_list)) + '\n')

    with open(orgin_test_file) as f_2: # 抽取测试集
        for l in f_2.readlines():  # readlines()返回一个列表，其中包含文件中的每一行作为列表项
            if len(l) > 0:
                l = l.strip('\n').split(' ')  # split()通过指定分隔符对字符串进行切片
                # strip()是行首尾处理函数，可以删掉指定的行首或行尾字符串，默认删掉行首和行尾的换行符'\n'
                if int(l[0]) in use_list:
                    items_test = [int(i) for i in l[1:] if int(i) in item_list]
                    if len(items_test) > 0:
                        with open(orgin_test_file.replace('gowalla','mini_gowalla'), 'a') as f_3:
                            f_3.write(l[0])
                            for k in items_test:
                                f_3.write(' ' + str(k))
                            f_3.write('\n')
                    else:
                        with open(orgin_test_file.replace('gowalla','mini_gowalla'), 'a') as f_1:
                            f_1.write(l[0] + ' ' + str(random.choice(item_list)) + '\n')

    return 0

miniDataSet('Datas/gowalla', 2000, 8000)