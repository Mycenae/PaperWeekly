# ArcFace: Additive Angular Margin Loss for Deep Face Recognition

Jiankang Deng et al. Imperial College London

## Abstract 摘要

One of the main challenges in feature learning using Deep Convolutional Neural Networks (DCNNs) for large-scale face recognition is the design of appropriate loss functions that enhance discriminative power. Centre loss penalises the distance between the deep features and their corresponding class centres in the Euclidean space to achieve intra-class compactness. SphereFace assumes that the linear transformation matrix in the last fully connected layer can be used as a representation of the class centres in an angular space and penalises the angles between the deep features and their corresponding weights in a multiplicative way. Recently, a popular line of research is to incorporate margins in well-established loss functions in order to maximise face class separability. In this paper, we propose an Additive Angular Margin Loss (ArcFace) to obtain highly discriminative features for face recognition. The proposed ArcFace has a clear geometric interpretation due to the exact correspondence to the geodesic distance on the hypersphere. We present arguably the most extensive experimental evaluation of all the recent state-of-the-art face recognition methods on over 10 face recognition benchmarks including a new large-scale image database with trillion level of pairs and a large-scale video dataset. We show that ArcFace consistently outperforms the state-of-the-art and can be easily implemented with negligible computational overhead. We release all refined training data, training codes, pre-trained models and training logs, which will help reproduce the results in this paper.

使用深度卷积神经网络(DCNNs)进行大规模人脸识别任务，在特征学习中的一个主要挑战是，设计合适的损失函数，增强区分能力。中间损失惩罚的是深度特征和其对应的类别中心之间在欧几里得空间中的距离，以得到类内的紧凑性。SphereFace假设，最后一个全连接层的线性变换矩阵，可以用于角度空间中的类别中心的表示，以乘法的方式惩罚深度特征之间的角度及其对应的权重。最近，一条流行的研究线是在已经确定的损失函数中加入空白边际，以最大化类别区分性。本文中，我们提出一种加性角度边际损失(ArcFace)，得到了高度可区分的特征，进行人脸识别。提出的ArcFace有清晰的几何解释性，因为精确的对应着超球上的测地距离。我们将所有最近最好的人脸识别方法，在超过10个人脸识别基准测试上进行了最广泛的试验，包括一个新的大规模人脸图像数据集（包含了trillion级的人脸对），和一个大规模视频数据集。试验结果一致说明，ArcFace超过了目前最好的算法，可以很容易的实现，计算量也非常小。我们开放了所有精炼的训练数据，训练代码，预训练模型和训练记录，这对重现本文的结果有所帮助。

## 1. Introduction 引言

Face representation using Deep Convolutional Neural Network (DCNN) embedding is the method of choice for face recognition [32, 33, 29, 24]. DCNNs map the face image, typically after a pose normalisation step [45], into a feature that has small intra-class and large inter-class distance.

使用深度卷积神经网络(DCNN)嵌入的人脸表示是人脸识别[32,33,29,24]等方法的选择。人脸图像通常在姿态规一化步骤[45]后，使用DCNNs映射到一个特征，这个特征有着小的类内距离，大的类间距离。

There are two main lines of research to train DCNNs for face recognition. Those that train a multi-class classifier which can separate different identities in the training set, such by using a softmax classifier [33, 24, 6], and those that learn directly an embedding, such as the triplet loss [29]. Based on the large-scale training data and the elaborate DCNN architectures, both the softmax-loss-based methods [6] and the triplet-loss-based methods [29] can obtain excellent performance on face recognition. However, both the softmax loss and the triplet loss have some drawbacks. For the softmax loss: (1) the size of the linear transformation matrix $W ∈ R^{d×n}$ increases linearly with the identities number n; (2) the learned features are separable for the closed-set classification problem but not discriminative enough for the open-set face recognition problem. For the triplet loss: (1) there is a combinatorial explosion in the number of face triplets especially for large-scale datasets, leading to a significant increase in the number of iteration steps; (2) semi-hard sample mining is a quite difficult problem for effective model training.

训练人脸识别的DCNNs有两条主要的研究线。一条线是训练一个多类别分类器，使用softmax分类器[33,24,6]，可以分离训练集中的不同个体；另一条线是直接学习一个嵌入，如triplet损失[29]。基于大规模训练数据和复杂的DCNN架构，这两类方法，即基于softmax-loss的方法[6]和基于triplet-loss的方法[29]都在人脸识别中取得了非常好的表现，但是都有一些缺陷。对于softmax loss来说，(1)线性变换矩阵$W ∈ R^{d×n}$的规模随着个体数量n线性变化；(2)学习到的特征对于闭集分类问题是可分的，但对于开放集合的人脸识别问题，可区分性不够好。对于triplet loss来说，(1)对于人脸三元组来说有组合爆炸的问题，尤其是对于大规模数据集，导致迭代次数会显著增加；(2)对模型训练来说，半难分样本挖掘是非常困难的问题。

Figure 1. Based on the centre [18] and feature [37] normalisation, all identities are distributed on a hypersphere. To enhance intraclass compactness and inter-class discrepancy, we consider four kinds of Geodesic Distance (GDis) constraint. (A) Margin-Loss: insert a geodesic distance margin between the sample and centres. (B) Intra-Loss: decrease the geodesic distance between the sample and the corresponding centre. (C) Inter-Loss: increase the geodesic distance between different centres. (D) Triplet-Loss: insert a geodesic distance margin between triplet samples. In this paper, we propose an Additive Angular Margin Loss (ArcFace), which is exactly corresponded to the geodesic distance (Arc) margin penalty in (A), to enhance the discriminative power of face recognition model. Extensive experimental results show that the strategy of (A) is most effective.

图1. 在进行中心归一化[18]和特征归一化[37]后，所有个体的人脸都分布在一个超球上。为增强类内的紧凑性和类间的差异性，我们考虑四种测地距离约束。(A)边际损失：在样本和中心之间插入一个测地距离边际；(B)类内损失：降低样本及对应中心的测地距离；(C)类间距离：增加不同中心间的测地距离；(D)triplet损失：在三元组样本间插入一个测地距离边际。本文中，我们提出一种加性角度边际损失(ArcFace)，正好对应着(A)中的测地距离(Arc)边际惩罚，可以增强识别模型的区分能力。大量的试验结果表明，(A)的策略是最有效的。

Several variants [38,9,46,18,37,35,7,34,27] have been proposed to enhance the discriminative power of the softmax loss. Wen et al. [38] pioneered the centre loss, the Euclidean distance between each feature vector and its class centre, to obtain intra-class compactness while the inter-class dispersion is guaranteed by the joint penalisation of the softmax loss. Nevertheless, updating the actual centres during training is extremely difficult as the number of face classes available for training has recently dramatically increased.

研究者们提出了一些变体[38,9,46,18,37,35,7,34,27]来增强softmax loss的区分能力。Wen等[38]首先研究了中心损失，即每个特征向量及其类心的欧几里得距离，以提高类内紧凑度，同时softmax loss的惩罚保证了类间距离。尽管如此，在训练中更新实际的中心是非常困难的，因为最近可用于训练的人脸类别数量显著的增加了。

By observing that the weights from the last fully connected layer of a classification DCNN trained on the softmax loss bear conceptual similarities with the centres of each face class, the works in [18,19] proposed a multiplicative angular margin penalty to enforce extra intra-class compactness and inter-class discrepancy simultaneously, leading to a better discriminative power of the trained model. Even though Sphereface [18] introduced the important idea of angular margin, their loss function required a series of approximations in order to be computed, which resulted in an unstable training of the network. In order to stabilise training, they proposed a hybrid loss function which includes the standard softmax loss. Empirically, the softmax loss dominates the training process, because the integer-based multiplicative angular margin makes the target logit curve very precipitous and thus hinders convergence. CosFace [37,35] directly adds cosine margin penalty to the target logit, which obtains better performance compared to SphereFace but admits much easier implementation and relieves the need for joint supervision from the softmax loss.

研究者观察到，用softmax损失训练的分类DCNN，其最后一个全连接层的权重与每个人脸类别的中心有概念上的类似性，于是[18,19]提出了一种乘性角度边际惩罚，强行同时加入了额外的类内紧凑度和类间的区分度，训练出的模型得到了更好的区分能力。即使SphereFace[18]提出了角度边际的重要概念，其损失函数需要一系列近似才能得到实际计算，其网络的训练过程也因此变得很不稳定。为稳定训练，他们提出了一种混合损失函数，包括了标准的softmax loss。从经验来看，softmax损失在训练过程中占主要地位，因为整数乘性角度边际使目标logit曲线非常多波折，所以阻碍了收敛。CosFace[37,35]直接向目标logit加入了cosine边际惩罚，与SphereFace相比得到了更好的性能，但承认实现起来非常简单，减少了softmax损失的联合监督的必要性。

In this paper, we propose an Additive Angular Margin Loss (ArcFace) to further improve the discriminative power of the face recognition model and to stabilise the training process. As illustrated in Figure 2, the dot product between the DCNN feature and the last fully connected layer is equal to the cosine distance after feature and weight normalisation. We utilise the arc-cosine function to calculate the angle between the current feature and the target weight. Afterwards, we add an additive angular margin to the target angle, and we get the target logit back again by the cosine function. Then, we re-scale all logits by a fixed feature norm, and the subsequent steps are exactly the same as in the softmax loss. The advantages of the proposed ArcFace can be summarised as follows:

本文中，我们提出了一种加性角度边际损失(ArcFace)，进一步改进人脸识别模型的区分能力，稳定训练过程。如图2所示，DCNN特征和最后一个全连接层之间的点乘，等于特征和权重归一化后的cosine距离。我们利用arc-cosine函数来计算目前的特征和目标权重间的角度。然后，我们向目标角度中加入了一个加性角度边际，通过cosine函数再次得到目标logit。然后，我们通过一个固定的特征范数改变所有logit的尺度，后续的步骤与softmax损失中的一样。我们提出的ArcFace的优点总结如下：

**Engaging**. ArcFace directly optimises the geodesic distance margin by virtue of the exact correspondence between the angle and arc in the normalised hypersphere. We intuitively illustrate what happens in the 512-D space via analysing the angle statistics between features and weights. ArcFace凭借归一化超球上角度和弧的精确对应来直接优化测地距离边际。我们通过分析特征与权重之间的角度统计信息，直观的描述了在512维空间中发生的情况。

**Effective**. ArcFace achieves state-of-the-art performance on ten face recognition benchmarks including large-scale image and video datasets. ArcFace在10个人脸识别基准测试中取得了目前最好的性能，包括大规模图像和视频数据集。

**Easy**. ArcFace only needs several lines of code as given in Algorithm 1 and is extremely easy to implement in the computational-graph-based deep learning frameworks, e.g. MxNet [8], Pytorch [25] and Tensorflow [4]. Furthermore, contrary to the works in [18, 19], ArcFace does not need to be combined with other loss functions in order to have stable performance, and can easily converge on any training datasets. ArcFace只需要很少几行代码，如算法1所示，在深度学习框架中非常容易实现，如MxNet[8],Pytorch[25]和TensorFlow[4]。而且，与[18,19]的工作不同，ArcFace不需要与其他损失函数一起使用，也可以得到稳定的表现，在任何训练数据集上都可以很容易的收敛。

**Efficient**. ArcFace only adds negligible computational complexity during training. Current GPUs can easily support millions of identities for training and the model parallel strategy can easily support many more identities. ArcFace在训练过程中只增加了很少计算量。目前的GPU可以很容易的进行数百万个体的训练，模型并行化策略可以很轻易的支持更多的个体。

Figure 2. Training a DCNN for face recognition supervised by the ArcFace loss. Based on the feature $x_i$ and weight W normalisation, we get the $cos θ_j$ (logit) for each class as $W_j^T x_i$. We calculate the $arccos(cosθ_{y_i})$ and get the angle between the feature $x_i$ and the ground truth weight $W_{y_i}$. In fact, $W_j$ provides a kind of centre for each class. Then, we add an angular margin penalty m on the target (ground truth) angle $θ_{y_i}$. After that, we calculate $cos(θ_{y_i} + m)$ and multiply all logits by the feature scale s. The logits then go through the softmax function and contribute to the cross entropy loss.

图2. 用ArcFace损失训练一个人脸识别的DCNN。在对特征$x_i$和权重W进行归一化后，我们得到了每个类别的$cos θ_j$(logit)，即$W_j^T x_i$。我们计算$arccos(cosθ_{y_i})$，得到了特征$x_i$和真值权重$W_{y_i}$之间的夹角。实际上，$W_j$某种程度上扮演的是每个类的中心的角色。然后，我们对目标（真值）角度$θ_{y_i}$上加入了一个角度边际惩罚m。之后，我们计算了$cos(θ_{y_i} + m)$，将所有的logits都乘以特征尺度s。这些logits然后经过softmax函数，然后计算出交叉熵损失。

## 2. Proposed Approach 提出的方法

### 2.1. ArcFace

The most widely used classification loss function, softmax loss, is presented as follows: 最常用的分类损失函数，softmax损失，如下所示：

$$L_1 = -\frac{1}{N} \sum_{i=1}^N log \frac{e^{W_{y_i}^T x_i + b_{y_i}}}{\sum_{j=1}^n e^{W_j^T x_i + b_j}}$$(1)

where $x_i ∈ R^d$ denotes the deep feature of the i-th sample, belonging to the $y_i$-th class. The embedding feature dimension d is set to 512 in this paper following [38,46,18,37]. $W_j ∈ R^d$ denotes the j-th column of the weight $W ∈ R^{d×n}$ and $b_j ∈ R^n$ is the bias term. The batch size and the class number are N and n, respectively. Traditional softmax loss is widely used in deep face recognition [24,6]. However, the softmax loss function does not explicitly optimise the feature embedding to enforce higher similarity for intra-class samples and diversity for inter-class samples, which results in a performance gap for deep face recognition under large intra-class appearance variations (e.g. pose variations [30,48] and age gaps [22,49]) and large-scale test scenarios (e.g. million [15,39,21] or trillion pairs [2]).

其中$x_i ∈ R^d$表示第i个样本的深度特征，属于第$y_i$类。嵌入特征维度d设为512，与文章[38,46,18,37]一样。$W_j ∈ R^d$表示权重$W ∈ R^{d×n}$的第j列，$b_j ∈ R^n$是偏置项。批规模为N，类别数为n。传统的softmax损失广泛的应用于深度人脸识别中[24,6]。但是，softmax并没有显式的优化特征嵌入，以增强类内样本相似性，和类间样本的区分性，所以在类内外形变化较大时（如姿态差异[30,48]和年龄差距[22,49]），在大规模测试场景时（如百万级[15,39,21]或万亿级人脸对[2]），性能上就有差距了。

For simplicity, we fix the bias $b_j = 0$ as in [18]. Then, we transform the logit [26] as $W_j^T x_i = ||W_j|| ||x_i|| cos θ_j$, where $θ_j$ is the angle between the weight $W_j$ and the feature $x_i$. Following [18,37,36], we fix the individual weight $||W_j|| = 1$ by $l_2$ normalisation. Following [28,37,36,35], we also fix the embedding feature $||x_i||$ by $l_2$ normalisation and re-scale it to s. The normalisation step on features and weights makes the predictions only depend on the angle between the feature and the weight. The learned embedding features are thus distributed on a hypersphere with a radius of s.

为简化，我们固定偏置$b_j = 0$，如同[18]一样。然后，我们将logit变换为[26]$W_j^T x_i = ||W_j|| ||x_i|| cos θ_j$，其中$θ_j$是权重$W_j$和特征$x_i$之间的夹角。遵循[18,37,36]，我们通过$l_2$归一化固定权重$||W_j|| = 1$。遵循[28,37,36,35]，我们还通过$l_2$归一化固定嵌入特征$||x_i||$并使其尺度为s。特征和权重的归一化步骤使预测只取决于特征和权重之间的角度。学习到的嵌入特征因此分布在一个超球上，半径为s。

$$L_2 = -\frac{1}{N} \sum_{i=1}^N log \frac {e^{scosθ_{y_i}}} {e^{scosθ_{y_i}}+ \sum_{j=1,j \neq y_i}^n e^{scosθ_j}}$$(2)

As the embedding features are distributed around each feature centre on the hypersphere, we add an additive angular margin penalty m between $x_i$ and $W_{y_i}$ to simultaneously enhance the intra-class compactness and inter-class discrepancy. Since the proposed additive angular margin penalty is equal to the geodesic distance margin penalty in the normalised hypersphere, we name our method as ArcFace.

由于嵌入特征分布在每个特征中心周围的超球上，我们在$x_i$和$W_{y_i}$之间增加一个加性的角度边际惩罚m，以同时增强类内紧凑度和类间的可分性。由于提出的加性角度边际惩罚等于归一化的超球上的测地距离边际惩罚，所以称这种方法为ArcFace。

$$L_3 = -\frac{1}{N} \sum_{i=1}^N log \frac {e^{s(cos(θ_{y_i}+m))}} {e^{s(cos(θ_{y_i}+m))}+ \sum_{j=1,j \neq y_i}^n e^{scosθ_j}}$$(3)

We select face images from 8 different identities containing enough samples (around 1,500 images/class) to train 2D feature embedding networks with the softmax and ArcFace loss, respectively. As illustrated in Figure 3, the softmax loss provides roughly separable feature embedding but produces noticeable ambiguity in decision boundaries, while the proposed ArcFace loss can obviously enforce a more evident gap between the nearest classes.

我们选择了8个不同个体的人脸图像，包含足够的样本（大约1500幅图像每类），来训练2D特征嵌入网络，分别使用softmax损失和ArcFace损失。如图3所示，softmax损失给出了大致可以分辨的特征嵌入，但在决策界面上有很大的模糊性，而提出的ArcFace损失则在最近的类别之间有非常明显的间隔。

Figure 3. Toy examples under the softmax and ArcFace loss on 8 identities with 2D features. Dots indicate samples and lines refer to the centre direction of each identity. Based on the feature normalisation, all face features are pushed to the arc space with a fixed radius. The geodesic distance gap between closest classes becomes evident as the additive angular margin penalty is incorporated.

图3. softmax和ArcFace损失在8个个体上训练2D特征的例子。点代表样本，线代表每个个体的中心方向。在特征归一化后，所有人脸特征都分布在圆弧空间上，半径固定。使用了加性角度边际惩罚后，相邻类之间的测地距离间隔变得非常明显。

### 2.2. Comparison with SphereFace and CosFace 与SphereFace和CosFace之间的比较

**Numerical Similarity**. In SphereFace [18, 19], ArcFace, and CosFace [37, 35], three different kinds of margin penalty are proposed, e.g. multiplicative angular margin $m_1$, additive angular margin $m_2$, and additive cosine margin $m_3$, respectively. From the view of numerical analysis, different margin penalties, no matter add on the angle [18] or cosine space [37], all enforce the intra-class compactness and inter-class diversity by penalising the target logit [26]. In Figure 4(b), we plot the target logit curves of SphereFace, ArcFace and CosFace under their best margin settings. We only show these target logit curves within [$20^◦, 100^◦$] because the angles between $W_{y_i}$ and $x_i$ start from around $90^◦$(random initialisation) and end at around $30^◦$ during ArcFace training as shown in Figure 4(a). Intuitively, there are three factors in the target logit curves that affect the performance, i.e. the starting point, the end point and the slope.

**计算上的相似性**。在SphereFace[18,19]、ArcFace和CosFace[37,35]中，提出了三种不同的边际惩罚，分别是乘性角度边际$m_1$，加性角度边际$m_2$和加性cosine边际$m_3$。从数值分析的角度来看，三种边际惩罚，不管是加在角度上[18]还是cosine空间[37]，都通过惩罚目标logit[26]，加强了类内紧凑性和类间可分性。在图4(b)中，我们画出了SphereFace、ArcFace和CosFace在其最佳边际设置上的目标logit曲线。我们只给出了角度范围[$20^◦, 100^◦$]内的目标logit曲线，因为$W_{y_i}$和$x_i$之间的角度在ArcFace训练期间始于约$90^◦$（随机初始化）终于约$30^◦$如图4(a)所示。直觉上来说，目标logit曲线中有3个因素影响性能，即开始点，终点和倾斜度。

By combining all of the margin penalties, we implement SphereFace, ArcFace and CosFace in an united framework with $m_1$, $m_2$ and $m_3$ as the hyper-parameters. 结合所有边际惩罚，我们以$m_1$, $m_2$和$m_3$为超参数，在统一的框架中实现了SphereFace, ArcFace和CosFace。

$$L_4 = -\frac{1}{N} \sum_{i=1}^N log \frac {e^{s(cos(m_1 θ_{y_i} + m_2)-m_3)}} {e^{s(m_1 θ_{y_i} + m_2)-m_3)}+ \sum_{j=1,j \neq y_i}^n e^{scosθ_j}}$$(4)

As shown in Figure 4(b), by combining all of the above-motioned margins $(cos(m_1 θ + m_2 ) − m_3)$, we can easily get some other target logit curves which also have high performance. 如图4(b)所示，结合上述所有边际$(cos(m_1 θ + m_2 ) − m_3)$，我们可以很容易的得到一些其他目标logit曲线，它们也可以得到很高的性能。

Figure 4. Target logit analysis. (a) $θ_j$ distributions from start to end during ArcFace training. (2) Target logit curves for softmax, SphereFace, ArcFace, CosFace and combined margin penalty $(cos(m_1 θ + m_2 ) − m_3)$.

图4. 目标logit分析。(a)$θ_j$分布在ArcFace训练期间从开始到结束的情况，(b)softmax, SphereFace, ArcFace, CosFace和结合边际惩罚$(cos(m_1 θ + m_2 ) − m_3)$的目标logit曲线。

**Geometric Difference**. Despite the numerical similarity between ArcFace and previous works, the proposed additive angular margin has a better geometric attribute as the angular margin has the exact correspondence to the geodesic distance. As illustrated in Figure 5, we compare the decision boundaries under the binary classification case. The proposed ArcFace has a constant linear angular margin throughout the whole interval. By contrast, SphereFace and CosFace only have a nonlinear angular margin.

**几何区别**。尽管ArcFace和之前的工作在数值计算上有很多相似性，但我们提出的加性角度边际还是有更好的几何属性的，因为角度边际精确对应着测地距离。如图5所示，我们比较了二值分类情况下的决策边界，提出的ArcFace在整个区间上有着常量线性角度边际。比较之下，SphereFace和CosFace的角度边际都是非线性的。

Figure 5. Decision margins of different loss functions under binary classification case. The dashed line represents the decision boundary, and the grey areas are the decision margins. 图5. 不同损失函数在二值分类情况下的决策边际。虚线表示决策边界，灰色区域为决策边际。

The minor difference in margin designs can have “butterfly effect” on the model training. For example, the original SphereFace [18] employs an annealing optimisation strategy. To avoid divergence at the beginning of training, joint supervision from softmax is used in SphereFace to weaken the multiplicative margin penalty. We implement a new version of SphereFace without the integer requirement on the margin by employing the arc-cosine function instead of using the complex double angle formula. In our implementation, we find that m = 1.35 can obtain similar performance compared to the original SphereFace without any convergence difficulty.

边际设计上的较小差异对于模型训练有着蝴蝶效应。比如，原版ShereFace[18]采用了退火优化策略。为防止在训练的开始就收敛，SphereFace使用了softmax进行联合训练，以弱化乘性边际惩罚。我们实现了SphereFace的新版本，边际没有整数的要求，因为使用了arc-cosine函数，而没有使用复杂的双倍角公式。在我们的实现中，我们发现m=1.35可以得到与原始SphereFace相似的效果，而且没有任何收敛困难。

### 2.3. Comparison with Other Losses 与其他损失的比较

Other loss functions can be designed based on the angular representation of features and weight-vectors. For examples, we can design a loss to enforce intra-class compactness and inter-class discrepancy on the hypersphere. As shown in Figure 1, we compare with three other losses in this paper.

基于特征和权重向量的角度表示，可以设计其他损失函数。比如，我们可以设计一个损失函数，在超球上增加类内紧凑性和类间可分性。如图1所示，我们在本文中与其他三种损失进行比较。

**Intra-Loss** is designed to improve the intra-class compactness by decreasing the angle/arc between the sample and the ground truth centre. 设计类内损失，降低样本和真值中心的角度/弧度，以改进类内紧凑性。

$$L_5 = L_2 + \frac {1}{πN} \sum_{i=1}^N θ_{y_i}$$(5)

**Inter-Loss** targets at enhancing inter-class discrepancy by increasing the angle/arc between different centres. 类间损失通过增加不同中心之间的角度/弧度，目标在于增强类间可分性。

$$L_6 = L_2 - \frac {1}{πN(n-1)} \sum_{i=1}^N \sum_{j=1,j \neq y_i}^n arccos(W_{y_i}^T W_j)$$(6)

The Inter-Loss here is a special case of the Minimum Hyper-spherical Energy (MHE) method [17]. In [17], both hidden layers and output layers are regularised by MHE. In the MHE paper, a special case of loss function was also proposed by combining the SphereFace loss with MHE loss on the last layer of the network. 这里的类间损失是最小超球能量法(Minimum Hyper-spherical Energy, MHE)[17]的一种特殊情况。在[17]中，隐藏层和输出层都被MHE正则化。在MHE文章中，网络最后一层也提出了一种特殊情况的损失函数，结合了SphereFace损失和MHE损失。

**Triplet-loss** aims at enlarging the angle/arc margin between triplet samples. In FaceNet [29], Euclidean margin is applied on the normalised features. Here, we employ the triplet-loss by the angular representation of our features as $arccos(x_i^{pos} x_i) + m ≤ arccos(x_i^{neg} x_i)$. 三元组损失目标在于增大三元组样本间的角度/弧度边际。在FaceNet[29]中，在归一化的特征中使用了欧几里得边际。这里，我们通过特征的角度表示来使用三元组损失，即$arccos(x_i^{pos} x_i) + m ≤ arccos(x_i^{neg} x_i)$。

## 3. Experiments 试验

### 3.1. Implementation Details 实现细节

**Datasets**. As given in Table 1, we separately employ CASIA [43], VGGFace2 [6], MS1MV2 and DeepGlint-Face (including MS1M-DeepGlint and Asian-DeepGlint) [2] as our training data in order to conduct fair comparison with other methods. Please note that the proposed MS1MV2 is a semi-automatic refined version of the MS-Celeb-1M dataset [10]. To best of our knowledge, we are the first to employ ethnicity-specific annotators for large-scale face image annotations, as the boundary cases (e.g. hard samples and noisy samples) are very hard to distinguish if the annotator is not familiar with the identity. During training, we explore efficient face verification datasets (e.g. LFW [13], CFP-FP[30], AgeDB-30 [22]) to check the improvement from different settings. Besides the most widely used LFW [13] and YTF [40] datasets, we also report the performance of ArcFace on the recent large-pose and large-age datasets(e.g. CPLFW [48] and CALFW [49]). We also extensively test the proposed ArcFace on large-scale image datasets (e.g. MegaFace [15], IJB-B [39], IJB-C [21] and Trillion-Pairs [2]) and video datasets (iQIYI-VID [20]).

**数据集**。如表1所示，我们分别采用了CASIA [43], VGGFace2 [6], MS1MV2和DeepGlint-Face（包括MS1M-DeepGlint和Asian-DeepGlint）[2]作为我们的训练数据，以与其他方法进行公平比较。请注意，提出的MS1MV2是MS-Celeb-1M数据集[10]的半自动优化版本。据我们所知，我们是第一个采用特定种族的标注者进行大规模人脸图像标注的，因为如果标注者不熟悉这些个体的话，边缘情况（如难分样本和含噪样本）非常难以辨识。在训练过程中，我们探索了一些人脸验证数据集（如LFW[13], CFP-FP[30], AgeDB-30[22]）以检查不同设置下的改进情况。除了最常用的LFW[13]和YTF[40]数据集，我们还给出了ArcFace在最近的多姿态和多年龄数据集（如CPLFW[48]和CALFW[49]）。我们还在很多大规模图像数据集（如MegaFace[15], IJB-B[39], IJB-C[21]和Trillion-Pairs[2]）和视频数据集（iQIYI-VID[20]）上对ArcFace上进行了测试。

**Experimental Settings**. For data prepossessing, we follow the recent papers [18,37] to generate the normalised face crops (112 × 112) by utilising five facial points. For the embedding network, we employ the widely used CNN architectures, ResNet50 and ResNet100 [12,11]. After the last convolutional layer, we explore the BN [14]-Dropout [31]-FC-BN structure to get the final 512-D embedding feature. In this paper, we use ([training dataset, network structure, loss]) to facilitate understanding of the experimental settings.

**试验设置**。对于数据预处理，我们采用最近的文章[18,37]的方法，使用5个人脸特征点生成归一化的人脸剪切块(112×112)。对于嵌入网络，我们采用的是广泛使用的CNN架构，ResNet50和ResNet100[12,11]。在最后一个卷积层后，我们尝试使用了BN[14]-Dropout[31]-FC-BN结构来得到最终的512维嵌入特征。在本文中，我们使用（[训练数据集，网络架构，损失函数]）来帮助理解试验设置。

We follow [37] to set the feature scale s to 64 and choose the angular margin m of ArcFace at 0.5. All experiments in this paper are implemented by MXNet [8]. We set the batch size to 512 and train models on four NVIDIA Tesla P40 (24GB) GPUs. On CASIA, the learning rate starts from 0.1 and is divided by 10 at 20K, 28K iterations. The training process is finished at 32K iterations. On MS1MV2, we divide the learning rate at 100K,160K iterations and finish at 180K iterations. We set momentum to 0.9 and weight decay to 5e-4. During testing, we only keep the feature embedding network without the fully connected layer (160MB for ResNet50 and 250MB for ResNet100) and extract the 512-D features (8.9 ms/face for ResNet50 and 15.4 ms/face for ResNet100) for each normalised face. To get the embedding features for templates (e.g. IJB-B and IJB-C) or videos (e.g. YTF and iQIYI-VID), we simply calculate the feature centre of all images from the template or all frames from the video. Note that, overlap identities between the training set and the test set are removed for strict evaluations, and we only use a single crop for all testing.

我们遵循[37]中的方案，设特征尺度s为64，选择ArcFace的角度边际m为0.5。本文中的所有试验都用MXNet[8]实现。我们设批大小为512，在4个NVidia Tesla P40 (24GB) GPUs上训练模型。在CASIA上，学习速率开始为0.1，在第20K、28K次迭代时除以10。训练过程结束于第32K次迭代。在MS1MV2上，在第100K、160K次迭代时学习率除以10，结束于第180K次迭代。我们设置动量为0.9，权重衰减为5e-4。在测试时，我们只保留特征嵌入网络，没有全连接层（ResNet50为160M，ResNet100为250M），对每个归一化的人脸提取出512D特征（ResNet50速度为每个人脸8.9ms，ResNet100为15.4ms每个人脸）。为对模板（如IJB-B和IJB-C）或视频（如YTF和iQIYI-VID）得到嵌入特征，我们只计算了模板所有图像的特征中心，或视频中的所有帧的。注意，训练集和测试集中重复的个体都被去除了，以进行严格的评估，而且我们只使用单个剪切块进行所有的测试。

Table 1. Face datasets for training and testing. “(P)” and “(G)” refer to the probe and gallery set, respectively.

Datasets | Identity | Image/Video
--- | --- | ---
CASIA [43] | 10K | 0.5M
VGGFace2 [6] | 9.1K | 3.3M
MS1MV2 | 85K | 5.8M
MS1M-DeepGlint [2] | 87K | 3.9M
Asian-DeepGlint [2] | 94K | 2.83M
LFW [13] | 5749 | 13233
CFP-FP [30] | 500 | 7000
AgeDB-30 [22] | 568 | 16488
CPLFW [48] | 5749 | 11652
CALFW [49] | 5749 | 12174
YTF [40] | 1595 | 3425
MegaFace [15] | 530(P) | 1M(G)
IJB-B [39] | 1845 | 76.8K
IJB-C [21] | 3531 | 148.8K
Trillion-Pairs [2] | 5749(P) | 1.58M(G)
iQIYI-VID [20] | 4934 | 172835

### 3.2. Ablation Study on Losses 损失函数的分离研究

In Table 2, we first explore the angular margin setting for ArcFace on the CASIA dataset with ResNet50. The best margin observed in our experiments was 0.5. Using the proposed combined margin framework in Eq. 4, it is easier to set the margin of SphereFace and CosFace which we found to have optimal performance when setting at 1.35 and 0.35, respectively. Our implementations for both SphereFace and CosFace can lead to excellent performance without observing any difficulty in convergence. The proposed ArcFace achieves the highest verification accuracy on all three test sets. In addition, we performed extensive experiments with the combined margin framework (some of the best performance was observed for CM1 (1, 0.3, 0.2) and CM2 (0.9, 0.4, 0.15)) guided by the target logit curves in Figure 4(b). The combined margin framework led to better performance than individual SphereFace and CosFace but upper-bounded by the performance of ArcFace.

在表2中，我们首先研究了在CASIA数据集上ResNet50的角度边际设置。试验观察到的最佳边际为0.5。使用式(4)的组合边际框架，很容易设置SphereFace和CosFace的边际，我们发现其最佳性能分别在1.35和0.35时达到。我们实现的SphereFace和CosFace都可以得到极佳的表现，没有任何收敛难度。提出的ArcFace在三个测试集上得到了最高的验证准确率。另外，我们用组合边际框架进行了广泛的试验，采用图4(b)中的目标logit曲线作为指引（观测到一些最佳性能，如CM1 (1, 0.3, 0.2)和CM2 (0.9, 0.4, 0.15)）。组合边际框架比单独的SphereFace和CosFace的性能都要好，但最佳表现是ArcFace。

Besides the comparison with margin-based methods, we conduct a further comparison between ArcFace and other losses which aim at enforcing intra-class compactness (Eq. 5) and inter-class discrepancy (Eq. 6). As the baseline we have chosen the softmax loss and we have observed performance drop on CFP-FP and AgeDB-30 after weight and feature normalisation. By combining the softmax with the intra-class loss, the performance improves on CFP-FP and AgeDB-30. However, combining the softmax with the inter-class loss only slightly improves the accuracy. The fact that Triplet-loss outperforms Norm-Softmax loss indicates the importance of margin in improving the performance. However, employing margin penalty within triplet samples is less effective than inserting margin between samples and centres as in ArcFace. Finally, we incorporate the Intra-loss, Inter-loss and Triplet-loss into ArcFace, but no improvement is observed, which leads us to believe that ArcFace is already enforcing intra-class compactness, inter-class discrepancy and classification margin.

除了与基于边际的方法进行了比较，我们还将ArcFace与其他损失函数进行了比较，即强调类内紧凑性的式(5)和类间可分性的式(6)。作为基准，我们选择了softmax损失，在权重和特征归一化后，在CFP-FP和AgeDB-30数据集上发现性能有下降。将softmax和类内损失组合起来，在CFP-FP和AgeDB-30上的性能得到了提升。但是，将softmax与类间损失组合到一起，只略微提升了准确度。Triplet损失超过了Norm-Softmax损失表明边际在改进性能上的重要性。但是，在三元组样本中使用边际惩罚，没有在样本和中心间插入边际有效，像ArcFace一样。最后，我们在ArcFace中使用了Intra-loss、Inter-loss和Triplet-loss，但没有观察到改进，这使我们相信，ArcFace已经增强了类内紧凑性、类间可分性和分类边际。

Table 2. Verification results (%) of different loss functions ([CASIA, ResNet50, loss*]).

Loss Functions | LFW | CFP-FP | AgeDB-30
--- | --- | --- | ---
ArcFace (0.4) | 99.53 | 95.41 | 94.98
ArcFace (0.45) | 99.46 | 95.47 | 94.93
ArcFace (0.5) | 99.53 | 95.56 | 95.15
ArcFace (0.55) | 99.41 | 95.32 | 95.05
SphereFace [18] | 99.42 | - | -
SphereFace (1.35) | 99.11 | 94.38 | 91.70
CosFace [37] | 99.33 | - | -
CosFace (0.35) | 99.51 | 95.44 | 94.56
CM1 (1, 0.3, 0.2) | 99.48 | 95.12 | 94.38
CM2 (0.9, 0.4, 0.15) | 99.50 | 95.24 | 94.86
Softmax | 99.08 | 94.39 | 92.33
Norm-Softmax (NS) | 98.56 | 89.79 | 88.72
NS+Intra | 98.75 | 93.81 | 90.92
NS+Inter | 98.68 | 90.67 | 89.50
NS+Intra+Inter | 98.73 | 94.00 | 91.41
Triplet (0.35) | 98.98 | 91.90 | 89.98
ArcFace+Intra | 99.45 | 95.37 | 94.73
ArcFace+Inter | 99.43 | 95.25 | 94.55
ArcFace+Intra+Inter | 99.43 | 95.42 | 95.10
ArcFace+Triplet | 99.50 | 95.51 | 94.40

To get a better understanding of ArcFace’s superiority, we give the detailed angle statistics on training data (CASIA) and test data (LFW) under different losses in Table 3. We find that (1) $W_j$ is nearly synchronised with embedding feature centre for ArcFace ($14.29^◦$), but there is an obvious deviation ($44.26^◦$) between $W_j$ and the embedding feature centre for Norm-Softmax. Therefore, the angles between $W_j$ cannot absolutely represent the inter-class discrepancy on training data. Alternatively, the embedding feature centres calculated by the trained network are more representative. (2) Intra-Loss can effectively compress intra-class variations but also brings in smaller inter-class angles. (3) Inter-Loss can slightly increase inter-class discrepancy on both W (directly) and the embedding network (indirectly), but also raises intra-class angles. (4) ArcFace already has very good intra-class compactness and inter-class discrepancy. (5) Triplet-Loss has similar intra-class compactness but inferior inter-class discrepancy compared to ArcFace. In addition, ArcFace has a more distinct margin than Triplet-Loss on the test set as illustrated in Figure 6.

为更好的理解ArcFace的优异性，我们在表3中给出了不同损失函数下，训练数据(CASIA)和测试数据(LFW)的详细角度统计。我们发现，(1)$W_j$几乎与ArcFace的嵌入特征中心同步($14.29^◦$)，但$W_j$与Norm-Softmax的嵌入特征中心却有很明显的偏差($44.26^◦$)。所以，$W_j$之间的角度不能绝对代表训练数据的类间可分性。或者，训练好的网络计算的嵌入特征中心更有代表性。(2)类内损失可以有效的压缩类内变化，但也带来了更小的类间角度。(3)类间损失会略微增加W（直接的）和嵌入网络（间接的）类间可分性，但也增加了类内的角度。(4)ArcFace已经有了非常好的类内紧凑性和类间可分性。(5)与ArcFace相比，三元组损失有着类似的类内紧凑性，但类间可分性更好一些。另外，ArcFace与Triplet-loss相比，在测试集上有着更可分的边际，如图6所示。

Table 3. The angle statistics under different losses ([CASIA, ResNet50, loss*]). Each column denotes one particular loss. “W-EC” refers to the mean of angles between W j and the corresponding embedding feature centre. “W-Inter” refers to the mean of minimum angles between W j ’s. “Intra1” and “Intra2” refer to the mean of angles between x i and the embedding feature centre on CASIA and LFW, respectively. “Inter1” and “Inter2” refer to the mean of minimum angles between embedding feature centres on CASIA and LFW, respectively.

| | NS | ArcFace | IntraL | InterL | TripletL
--- | --- | --- | --- | --- | ---
W-EC | 44.26 | 14.29 | 8.83 | 46.85 | -
W-Inter | 69.66 | 71.61 | 31.34 | 75.66 | -
Intra1 | 50.50 | 38.45 | 17.50 | 52.74 | 41.19
Inter1 | 59.23 | 65.83 | 24.07 | 62.40 | 50.23
Intra2 | 33.97 | 28.05 | 12.94 | 35.38 | 27.42
Inter2 | 65.60 | 66.55 | 26.28 | 67.90 | 55.94

Figure 6. Angle distributions of all positive pairs and random negative pairs (∼ 0.5M) from LFW. Red area indicates positive pairs while blue indicates negative pairs. All angles are represented in degree. ([CASIA, ResNet50, loss*]).

### 3.3. Evaluation Results 评估结果

**Results on LFW, YTF, CALFW and CPLFW**. LFW [13] and YTF [40] datasets are the most widely used benchmark for unconstrained face verification on images and videos. In this paper, we follow the unrestricted with labelled outside data protocol to report the performance. As reported in Table 4, ArcFace trained on MS1MV2 with ResNet100 beats the baselines (e.g. SphereFace [18] and CosFace [37]) by a significant margin on both LFW and YTF, which shows that the additive angular margin penalty can notably enhance the discriminative power of deeply learned features, demonstrating the effectiveness of ArcFace.

**在LFW, YTF, CALFW和CPLFW上的结果**。LFW[13]和YTF[40]数据集是在不受限的人脸验证中，最常使用的图像和视频基准测试。在本文中，我们遵循不受限的标注室外数据协议，给出性能结果。如表4所示，在MS1MV2上训练的ResNet100 ArcFace超过了基准方法很多（如SphereFace[18]和CosFace[37]），在LFW和YTF上都是，这说明加性角度边际惩罚可以显著增强深度学习特征的区分能力，表明了ArcFace的有效性。

Table 4. Verification performance (%) of different methods on LFW and YTF.

Method | Image | LFW | YTF
--- | --- | --- | ---
DeepID [32] | 0.2M | 99.47 | 93.20
Deep Face [33] | 4.4M | 97.35 | 91.4
VGG Face [24] | 2.6M | 98.95 | 97.30
FaceNet [29] | 200M | 99.63 | 95.10
Baidu [16] | 1.3M | 99.13 | -
Center Loss [38] | 0.7M | 99.28 | 94.9
Range Loss [46] | 5M | 99.52 | 93.70
Marginal Loss [9] | 3.8M | 99.48 | 95.98
SphereFace [18] | 0.5M | 99.47 | -
SphereFace+ [17] | 0.5M | 99.47 | -
CosFace [37] | 5M | 99.73 | 97.6
MS1MV2, R100, ArcFace | 5.8M | 99.83 | 98.02

Besides on LFW and YTF datasets, we also report the performance of ArcFace on the recently introduced datasets (e.g. CPLFW [48] and CALFW [49]) which show higher pose and age variations with same identities from LFW. Among all of the open-sourced face recognition models, the ArcFace model is evaluated as the top-ranked face recognition model as shown in Table 5, outperforming counterparts by an obvious margin. In Figure 7, we illustrate the angle distributions (predicted by ArcFace model trained on MS1MV2 with ResNet100) of both positive and negative pairs on LFW, CFP-FP, AgeDB-30, YTF, CPLFW and CALFW. We can clearly find that the intra-variance due to pose and age gaps significantly increases the angles between positive pairs thus making the best threshold for face verification increasing and generating more confusion regions on the histogram.

除了LFW和YTF数据集，我们还给出了ArcFace在最近提出的数据集上的表现（如CPLFW[48]和CALFW[49]），与LFW相比，这些数据集会在姿态和年龄上的变化更大一些。在所有开源的人脸识别模型中，ArcFace模型是评估性能最好的人脸识别模型，如表5所示，超过了其他算法很多。在图7中，我们给出了LFW, CFP-FP, AgeDB-30, YTF, CPLFW和CALFW数据集上阳性对和阴性对的角度分布，ArcFace模型在MS1MV2上采用ResNet100训练得到。我们可以很明显发现，姿态和年龄差显著的增加了阳性对的角度范围，这使得人脸验证的最佳阈值变大，在灰度图中产生了更多混淆区域。

Table 5. Verification performance (%) of open-sourced face recognition models on LFW, CALFW and CPLFW.

Method | LFW |CALFW | CPLFW
--- | --- | --- | ---
HUMAN-Individual | 97.27 | 82.32 | 81.21
HUMAN-Fusion | 99.85 | 86.50 | 85.24
Center Loss [38] | 98.75 | 85.48 | 77.48
SphereFace [18] | 99.27 | 90.30 | 81.40
VGGFace2 [6] | 99.43 | 90.57 | 84.00
MS1MV2, R100, ArcFace | 99.82 | 95.45 | 92.08

Figure 7. Angle distributions of both positive and negative pairs on LFW, CFP-FP, AgeDB-30, YTF, CPLFW and CALFW. Red area indicates positive pairs while blue indicates negative pairs. All angles are represented in degree. ([MS1MV2, ResNet100, ArcFace])

**Results on MegaFace**. The MegaFace dataset [15] includes 1M images of 690K different individuals as the gallery set and 100K photos of 530 unique individuals from FaceScrub [23] as the probe set. On MegaFace, there are two testing scenarios (identification and verification) under two protocols (large or small training set). The training set is defined as large if it contains more than 0.5M images. For the fair comparison, we train ArcFace on CAISA and MS1MV2 under the small protocol and large protocol, respectively. In Table 6, ArcFace trained on CASIA achieves the best single-model identification and verification performance, not only surpassing the strong baselines (e.g. SphereFace [18] and CosFace [37]) but also outperforming other published methods [38,17].

**MegaFace上的结果**。MegaFace数据集[15]包括690K不同个体的1M图像作为gallery集，从FaceScrub[23]上得到的530个体的100K幅图像作为probe集。在MegaFace上，有两种协议（大训练集或小训练集）下的两种测试场景（识别和验证）。训练集如果包含的图像超过0.5M，那么就定义为大。为公平比较，我们在CASIA和MS1MV2上分别用小的协议和大的协议训练ArcFace。在表6中，在CASIA上训练的ArcFace得到了最好的单模型识别和验证性能，不仅超过了强基准（如SphereFace[18]和CosFace[37]），也超过了其他发表的方法[38,17]。

Table 6. Face identification and verification evaluation of different methods on MegaFace Challenge1 using FaceScrub as the probe set. “Id” refers to the rank-1 face identification accuracy with 1M distractors, and “Ver” refers to the face verification TAR at $10^{−6}$ FAR. “R” refers to data refinement on both probe set and 1M distractors. ArcFace obtains state-of-the-art performance under both small and large protocols.

Methods | Id(%) | Ver(%)
--- | --- | ---
Softmax [18] | 54.85 | 65.92
Contrastive Loss[18, 32] | 65.21 | 78.86
Triplet [18, 29] | 64.79 | 78.32
Center Loss[38] | 65.49 | 80.14
SphereFace [18] | 72.729 | 85.561
CosFace [37] | 77.11 | 89.88
AM-Softmax [35] | 72.47 | 84.44
SphereFace+ [17] | 73.03 | -
CASIA, R50, ArcFace | 77350 | 92.34
CASIA, R50, ArcFace, R | 91.75 | 93.69
FaceNet [29] | 70.49 | 86.47
CosFace [37] | 82.72 | 93.69
MS1MV2, R100, ArcFace | 81.03 | 96.98
MS1MV2, R100, CosFace | 80.56 | 96.56
MS1MV2, R100, ArcFace, R | 98.35 | 98.48
MS1MV2, R100, CosFace, R | 97.91 | 97.91

As we observed an obvious performance gap between identification and verification, we performed a thorough manual check in the whole MegaFace dataset and found many face images with wrong labels, which significantly affects the performance. Therefore, we manually refined the whole MegaFace dataset and report the correct performance of ArcFace on MegaFace. On the refined MegaFace, ArcFace still clearly outperforms CosFace and achieves the best performance on both verification and identification.

我们发现识别和验证上性能有一定差距，所以在整个MegaFace数据集上进行了彻底的手工检查，发现很多人脸图像都有错误标注，显著的影响了算法性能。所以，我们手工提炼了整个MegaFace数据集，并给出了ArcFace在修正过的MegaFace上的表现。在提炼的MegaFace上，ArcFace仍然明显超过了CosFace，在验证和识别上都取得了最好的表现。

Under large protocol, ArcFace surpasses FaceNet [29] by a clear margin and obtains comparable results on identification and better results on verification compared to CosFace [37]. Since CosFace employs a private training data, we retrain CosFace on our MS1MV2 dataset with ResNet100. Under fair comparison, ArcFace shows superiority over CosFace and forms an upper envelope of CosFace under both identification and verification scenarios as shown in Figure 8.

在大的协议下，ArcFace超过了FaceNet[29]一大截，与CosFace[37]相比，在识别上取得了类似的结果，在验证上取得了更好的结果。由于CosFace采用了私有的训练数据，我们在MS1MV2数据集上用ResNet100重新训练了CosFace。在公平比较下，ArcFace超过了CosFace，如图8所示。

Figure 8. CMC and ROC curves of different models on MegaFace. Results are evaluated on both original and refined MegaFace dataset.

**Results on IJB-B and IJB-C**. The IJB-B dataset [39] contains 1,845 subjects with 21.8K still images and 55K frames from 7,011 videos. In total, there are 12, 115 templates with 10, 270 genuine matches and 8M impostor matches. The IJB-C dataset [39] is a further extension of IJB-B, having 3, 531 subjects with 31.3K still images and 117.5K frames from 11, 779 videos. In total, there are 23, 124 templates with 19, 557 genuine matches and 15, 639K impostor matches.

**在IJB-B和IJB-C上的结果**。IJB-B数据集[39]包括1845个主体的21.8K幅静止图像和7011个视频中的55K帧图像。总计有12115个模板，10270个真实匹配，8M个假的匹配。IJB-C数据集[39]是IJB-B的进一步扩展，有3531个主体的31.3K幅静止图像，和11779个视频中的117.5K帧图像。总共有23124个模板，19557个真实匹配，15.6M个假的匹配。

On the IJB-B and IJB-C datasets, we employ the VGG2 dataset as the training data and the ResNet50 as the embedding network to train ArcFace for the fair comparison with the most recent methods [6, 42, 41]. In Table 7, we compare the TAR (@FAR=1e-4) of ArcFace with the previous state-of-the-art models [6, 42, 41]. ArcFace can obviously boost the performance on both IJB-B and IJB-C (about 3 ∼ 5%, which is a significant reduction in the error). Drawing support from more training data (MS1MV2) and deeper neural network (ResNet100), ArcFace can further improve the TAR (@FAR=1e-4) to 94.2% and 95.6% on IJB-B and IJB-C, respectively. In Figure 9, we show the full ROC curves of the proposed ArcFace on IJB-B and IJB-C, and ArcFace achieves impressive performance even at FAR=1e-6 setting a new baseline.

在IJB-B和IJB-C数据集上，我们采用VGG2数据集作为训练数据，ResNet50作为嵌入网络，来训练ArcFace，以与最近的方法[6,42,41]进行公平比较。在表7中，我们将ArcFace与之前最好的模型[6,42,41]的TAR(@FAR=1e-4)进行了比较。ArcFace在两个数据集IJB-B和IJB-C上都可以明显提升性能（大约3-5%，这可以显著降低错误率）。若采用更多的训练数据(MS1MV2)和更深的伸进网络(ResNet100)，ArcFace可以将IJB-B和IJB-C上的TAR(@FAR=1e-4)分别提升到94.2%和95.6%。在图9中，我们给出ArcFace在IJB-B和IJB-C完整的ROC曲线，即使在FAR=1e-6上ArcFace也取得了令人印象深刻的性能，设定了一个新的基准。

Table 7. 1:1 verification TAR (@FAR=1e-4) on the IJB-B and IJB-C dataset.

Method | IJB-B | IJB-C
--- | --- | ---
ResNet50 [6] | 0.784 | 0.825
SENet50 [6] | 0.800 | 0.840
ResNet50+SENet50 [6] | 0.800 | 0.841
MN-v [42] | 0.818 | 0.852
MN-vc [42] | 0.831 | 0.862
ResNet50+DCN(Kpts) [41] | 0.850 | 0.867
ResNet50+DCN(Divs) [41] | 0.841 | 0.880
SENet50+DCN(Kpts) [41] | 0.846 | 0.874
SENet50+DCN(Divs) [41] | 0.849 | 0.885
VGG2, R50, ArcFace | 0.898 | 0.921
MS1MV2, R100, ArcFace | 0.942 | 0.956

Figure 9. ROC curves of 1:1 verification protocol on the IJB-B and IJB-C dataset.

**Results on Trillion-Pairs**. The Trillion-Pairs dataset [2] provides 1.58M images from Flickr as the gallery set and 274K images from 5.7k LFW [13] identities as the probe set. Every pair between gallery and probe set is used for evaluation (0.4 trillion pairs in total). In Table 8, we compare the performance of ArcFace trained on different datasets. The proposed MS1MV2 dataset obviously boosts the performance compared to CASIA and even slightly outperforms the DeepGlint-Face dataset, which has a double identity number. When combining all identities from MS1MV2 and Asian celebrities from DeepGlint, ArcFace achieves the best identification performance 84.840% (@FPR=1e-3) and comparable verification performance compared to the most recent submission (CIGIT IRSEC) from the lead-board.

**在Trillion-Pairs上的结果**。Trillion-Pairs数据集[2]包含1.58M Filickr上的图像作为gallery集，274K图像作为probe集，这是LFW[13]上5.7k个体的图像。Gallery和probe集上的每对图像都用于评估（总计有0.4万亿对）。在表8中，我们比较了在不同数据集上训练得到的ArcFace的性能。在MS1MV2数据集上训练的ArcFace明显比CASIA上训练的性能要好，比在DeepGlint-Face数据集上训练的也要好一些（其上有一个双重个体数）。当把MS1MV2和DeepGlint中的亚洲名人综合到一起，ArcFace得到了最好的识别性能84.840% (@FPR=1e-3)，与最新的CIGIT IRSEC排行榜上的验证性能类似。

Table 8. Identification and verification results (%) on the Trillion-Pairs dataset. ([Dataset*, ResNet100, ArcFace])

Method | Id(@FPR=1e-3) | Ver(@FPR=1e-9)
--- | --- | ---
CASIA | 26.643 | 21.452
MS1MV2 | 80.968 | 78.600
DeepGlint-Face | 80.331 | 78.586
MS1MV2+Asian | 84.840(1st) | 80.540
CIGIT IRSEC | 84.234(2st) | 81.558(1st)

**Results on iQIYI-VID**. The iQIYI-VID challenge [20] contains 565,372 video clips (training set 219,677, validation set 172,860, and test set 172,835) of 4934 identities from iQIYI variety shows, films and television dramas. The length of each video ranges from 1 to 30 seconds. This dataset supplies multi-modal cues, including face, cloth, voice, gait and subtitles, for character identification. The iQIYI-VID dataset employs MAP@100 as the evaluation indicator. MAP (Mean Average Precision) refers to the overall average accuracy rate, which is the mean of the average accuracy rate of the corresponding videos of person ID retrieved in the test set for each person ID (as the query) in the training set.

**在iQIYI-VID上的结果**。iQIYI-VID挑战赛[20]包含565372个视频片段（训练集219677，验证集172860，测试集172835），4934个个体，来源是iQIYI的各种电影电视剧和表演节目。每个视频片段长度都是1-30秒。这个数据集提供了多模的线索，包括人脸、着装、声音，步态和字幕，以进行角色辨认。iQIYI-VID数据集采用MAP@100作为其评估指标。MAP(Mean Aveage Precision)指的是总体的平均准确率。

As shown in Table 9, ArcFace trained on combined MS1MV2 and Asian datasets with ResNet100 sets a high baseline (MAP=(79.80%)). Based on the embedding feature for each training video, we train an additional three-layer fully connected network with a classification loss to get the customised feature descriptor on the iQIYI-VID dataset. The MLP learned on the iQIYI-VID training set significantly boosts the MAP by 6.60%. Drawing support from the model ensemble and context features from the off-the-shelf object and scene classifier [1], our final result surpasses the runner-up by a clear margin (0.99%).

如表9所示，在MS1MV2和亚洲数据集上共同训练的ResNet100-ArcFace设定了新的基准(MAP=79.80%)。基于每个训练视频的嵌入特征，我们训练一个额外的三层全连接网络，用的是分类损失函数，以得到在iQIYI-VID数据集上特有的特征描述子。在iQIYI-VID训练集上学到的MLP显著提升了MAP，多达6.60%。如果再采用模型集成，和现成的目标及场景分类器[1]的上下文特征，我们的最终结果超过了第二名很多(0.99%)。

Table 9. MAP of our method on the iQIYI-VID test set. “MLP” refers to a three-layer fully connected network trained on the iQIYI-VID training data.

Method | MAP(%)
--- | ---
MS1MV2+Asian, R100, ArcFace | 79.80
+MLP | 86.40
+Ensemble | 88.26
+Context | 88.65(1st)
Other Participant | 87.66(2nd)

## 4. Conclusions 结论

In this paper, we proposed an Additive Angular Margin Loss function, which can effectively enhance the discriminative power of feature embeddings learned via DCNNs for face recognition. In the most comprehensive experiments reported in the literature we demonstrate that our method consistently outperforms the state-of-the-art. Code and details have been released under the MIT license.

本文中，我们提出了一种加性角度边际损失函数，可以有效的增强人脸识别深度卷积网络的特征嵌入的区分能力。我们进行了非常广泛的试验，证明了我们的方法持续超过了目前最好的方法。

## 5. Appendix 附录

### 5.1 Parallel Acceleration 并行加速

Can we apply ArcFace on large-scale identities?

### 5.2 5.2. Feature Space Analysis 特征空间分析

Is the 512-d hypersphere space large enough to hold large-scale identities?