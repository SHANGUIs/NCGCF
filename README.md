# NCGCF
这是邻域对比图协同滤波模型的Tensorflow实现。  

NCGCF模型由 陈彦丞 在本科毕业论文中提出。

>《基于图神经网络的推荐系统——邻域对比图协同滤波》

作者：陈彦丞 山东大学 数学学院 2019级 华罗庚基地班  
指导教师: 曲存全 山东大学 数据科学研究院  
校外指导教师：姜志鹏 中国科学院大学 数学科学学院


## 简介
我们提出了一个新的邻域对比图协同滤波(NCGCF)模型来更好地完成隐式交互信息的推荐任务。
我们借鉴了LightGCN的轻量化图卷积，用结构邻域和语义邻域两方面的对比学习来完善经典的贝叶斯个性化推荐(BPR)损失，
并通过负采样加速学习过程。

## 环境要求
代码在Python 3.9的运行下进行测试，所需软件包如下：
* tensorflow >= 2.8.0
* numpy >= 1.19.5
* scipy >= 1.8.0
* sklearn >= 0.19.1

## 数据集
我们提供三个经过处理的推荐系统的经典数据集： Gowalla，Yelp2018和Amazon-book。此外，我们在使用计算资源较少的个人笔记本进行初步的实验验证时，发现模型的训练速度过慢，
于是提供了一个miniDateSet文件可以从经典数据集中抽取任意合适大小的数据集，例如用户数量为2000、项目数量为8000的数据集mini_gowalla。
这种抽取是按照用户项目ID依次进行的，而不是随机的，这是为了避免引起模型内部参数的混乱，同时也消除了实验结果的随机性。
* `train.txt`
  * 训练集
  * 每一行都是一个用户与项目的交互信息。

* `test.txt`
  * 测试集(正例)
  * 每一行都是一个用户与项目的交互信息(正例)。
  * 当评估推荐效果时，我们将所有未观察到的交互看作负例。
  
* `user_list.txt`
  * 用户集
  * 每一行都是一个用户的二元组(org_id, remap_id)，其中org_id和remap_id分别代表原始数据集中和我们的数据集中用户的ID。
  
* `item_list.txt`
  * 项目集
  * 每一行都是一个用户的二元组(org_id, remap_id)，其中org_id和remap_id分别代表原始数据集中和我们的数据集中项目的ID。

## 模型的代码结构
整个NCGCF模型由以下8个文件构成。
* `NCGCF`
  * 模型的主文件
  * 在该文件中定义了模型的整体框架，包括图卷积算子的定义、占位符的创建、参数初始化、贝叶斯个性化损失(BPR)、
矩阵稀疏形式与稠密形式的各种转换、模型的训练等。

* `addition`
  * 附加函数文件
  * 主要在batch_test文件中调用，包含一些用于评估推荐效果的函数、判断路径是否存在的函数和判断是否提前终止训练的函数。

* `batch_test`
  * 批量测试文件
  * 载入了模型参数库的数据，定义了一些批量测试的推荐评分函数。

* `load_data`
  * 数据加载文件
  * 对数据集进行初步处理，创建邻接矩阵、负样本集，定义批量采样函数、矩阵的稀疏分解函数等。

* `parser`
  * NCGCF模型的参数库
  * 可以在该文件中更改模型的超参数、模型使用的数据集、图卷积算子、损失函数等。

* `clustering`
  * 聚类函数文件
  * 目前只有kmeans聚类函数，后续会尝试使用一些基于图的聚类算法

* `SN_NCL`
  * 结构邻域-噪声对比损失函数

* `SN_CCL`
  * 语义邻域-聚类对比损失函数

## 预训练
当开启保存模型参数的设置以及使用预训练数据的设置时，模型的参数会在训练的过程中保存到weights文件夹中，
当重新开始训练进程时，会直接应用先前保存的参数，否则将通过高斯分布随机生成。

## 注意
在运行NCGCF.py文件之前，请将文件夹NCGCF_methods设为源代码根目录，以便能够被NCGCF.py成功调用。

=======
