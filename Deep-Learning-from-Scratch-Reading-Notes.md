# Deep learning from scratch
# 深度学习入门 - 基于Python的理论与实现

author：斋藤康毅 translator：陆宇杰

## ch1 Python入门

pass

## ch2 感知机(perceptron)
1. XOR(x1,x2) = AND(NAND(x1,x2), OR(x1,x2))；
2. 3层神经元实际上经过的是2层运算，所以有的文献称为2层网络/感知机，有的称为3层网络/感知机；

## ch3 神经网络
1. 网络权重保存在字典中，程序写出来非常简洁易懂；
2. softmax函数：在网络训练时需要softmax函数，在网络推理时不需要softmax函数，因为根据softmax层前的网络输出就可以分类，softmax并不改变分类结果，只是在网络学习反向传播计算梯度的时候为方便设计出来的函数；回归问题一般用恒等函数，分类问题用softmax函数

$$y_k = \frac{exp(a_k)}{\sum_{i=1}^{n}exp(a_i)}$$

## ch4 神经网络的学习
1. 神经网络学习称为端到端的机器学习(end-to-end machine learning)，意思是从原始数据（输入）到目标结果（输出）；
2. 损失函数一般为均方误差(mean square error)或交叉熵误差(cross entropy error)；

$$E = \frac{1}{2} \sum_k(y_k-t_k)^2$$

$$E = -\sum_k t_k log y_k$$

3. batch: 一次学习使用多个数据；
4. mini-batch: 训练数据量太大，不能一次学完，从训练数据中选出一小批数据称为mini-batch，对每个mini-batch进行学习；
5. 网络学习过程：(a)选择出mini-batch, (b)计算梯度 (c)更新参数 (d)重复abc；
6. 随机梯度下降法(SGD, stochastic gradient descent)，随机指的是随机选择出的mini-batch的意思，即，对随机选择出的数据进行的梯度下降法；
7. mini-batch的一般做法是，先将所有训练数据随机打乱，然后按指定的批次大小，顺序生成mini-batch；这样每个mini-batch都有一个索引号，然后用索引号可以遍历所有mini-batch；遍历一次所有数据，称为一个epoch；每过一个或几个epoch，可以输出一些学习信息，显示学习进度；

## ch5 误差反向传播法
1. 计算图computation graph（参考Andrej Karpathy的博客），通过计算图的反向传播，可以方便的计算导数；
2. 网络的每个节点都用一个类实现，每个类中都有forward和backward函数，在实现网络的时候，就可以通过函数的叠加（事实上还有更简洁的方法）实现前向传播和后向传播；
3. softmax和cross entropy error层的反向传播系数为$(y_1-t_1, y_2-t_2, ... , y_n-t_n)$，得到这样漂亮的结果不是偶然的，而是特意设计了softmax函数和交叉熵误差一起，才得到这样的反向传播效果；回归问题中恒等函数与均方误差的配合，也是设计出来，得到这样的反向传播效果；
4. 通过OrderedDict存储神经网络各层的参数等信息，程序写出来特别简洁易懂；

## ch6 与学习相关的技巧
1. 参数更新方法：SGD, Momentum, AdaGrad, Adam; AdaGrad即adaptive grad, 对大幅更新过的参数，降低其后续更细幅度；Adam结合了Momentum和AdaGrad；
2. 权重初始值：Xavier初始值，标准差为$1/\sqrt{n}$，*n*为上一层节点数；ReLU激活函数配He初始值，标准差为$\sqrt{2/n}$；
3. Batch Normalization: 在学习时以mini-batch为单位，按照mini-batch进行归一化，即是数据称为零均值单位方差，Batch Norm操作可以放在激活函数前或后；
4. Regularization: 权值衰减weight decay, 即对大的权重进行惩罚，即在损失函数中加入权值的L2范数作为惩罚项；
5. Hyper parameter: 验证数据是为了优化超参数，选出一个好的超参数取值范围；

## ch7 卷积神经网络

