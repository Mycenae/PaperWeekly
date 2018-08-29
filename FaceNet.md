# FaceNet: A Unified Embedding for Face Recognition and Clustering
# FaceNet: 人脸识别和聚类的统一嵌套

Florian Schroff et. al  Google

## Abstract

Despite significant recent advances in the field of face recognition [10, 14, 15, 17], implementing face verification and recognition efficiently at scale presents serious challenges to current approaches. In this paper we present a system, called FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily  implemented using standard techniques with FaceNet embeddings as feature vectors.

尽管近年人脸识别领域有了明显进展[10,14,15,17]，但现有方法高效的实现大规模人脸验证和识别还是很有挑战的。在本文中我们提出了FaceNet系统，可以直接学习出从人脸图像到一个紧凑欧几里得空间，其距离测度直接对应一种人脸相似度的度量。得到这种空间后，用FaceNet将人脸嵌套为特征向量，人脸识别、验证和分类这些任务，可以很容易的用标准技术实现。

Our method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. To train, we use triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method. The benefit of our approach is much greater representational efficiency: we achieve state-of-the-art face recognition performance using only 128-bytes per face.

以前的深度学习方法是使用一个中间的bottleneck层，而我们的方法使用训练后的深度卷积网络来直接优化嵌套。我们使用大致对齐的(匹配好的或不匹配的)人脸块triplets来训练，这些triplets是由一个新的在线triplet挖掘方法产生的。我们方法的优势在于强大的多的表示效率，我们的方法每个人脸只使用128字节，就取得了目前最好的人脸识别效果。

On the widely used Labeled Faces in the Wild (LFW) dataset, our system achieves a new record accuracy of
99.63%. On YouTube Faces DB it achieves 95.12%. Our system cuts the error rate in comparison to the best published result [15] by 30% on both datasets.

在广泛使用的LFW(Labeled Faces in the Wild)数据集上，我们的系统得到了新的准确度记录，99.63%。在Youtube Faces DB数据集上结果是95.12%，我们的系统与发表出来的最好的结果[15]相比，错误率在两个数据集上都低了30%。

We also introduce the concept of harmonic embeddings, and a harmonic triplet loss, which describe different versions of face embeddings (produced by different networks) that are compatible to each other and allow for direct comparison between each other.

我们还提出了harmonic嵌套和harmonic triplet loss的概念，其描述的是互相兼容的不同版本的人脸嵌套（由不同网络产生），可以直接互相对比。

## 1. Introduction

In this paper we present a unified system for face verification (is this the same person), recognition (who is this person) and clustering (find common people among these faces). Our method is based on learning a Euclidean embedding per image using a deep convolutional network. The network is trained such that the squared L2 distances in the embedding space directly correspond to face similarity: faces of the same person have small distances and faces of distinct people have large distances.

本文我们提出了人脸验证（这是同一个人吗？），人脸识别（这个人是谁？）和人脸分类（在这些人脸中找出普通人）三种应用的统一系统，我们的方法使用深度卷积网络，对每幅图像学习嵌套进一个欧几里得空间。网络经过训练后，其嵌套空间内的L2距离平方直接对应着人脸相似度（同一个人的不同人脸图像距离小，不同人的人脸图像距离大）。

Once this embedding has been produced, then the aforementioned tasks become straight-forward: face verification simply involves thresholding the distance between the two embeddings; recognition becomes a k-NN classification problem; and clustering can be achieved using off-the-shelf techniques such as k-means or agglomerative clustering.

嵌套产生后，前面提到的任务变得非常直接：人脸验证问题成为了两种嵌套后距离度量的阈值问题，人脸识别成了一个kNN分类问题，人脸分类可以使用成熟的技术如k均值或聚类算法。

Previous face recognition approaches based on deep networks use a classification layer [15, 17] trained over a set of known face identities and then take an intermediate bottleneck layer as a representation used to generalize recognition beyond the set of identities used in training. The downsides of this approach are its indirectness and its inefficiency: one has to hope that the bottleneck representation generalizes well to new faces; and by using a bottleneck layer the representation size per face is usually very large (1000s of dimensions). Some recent work [15] has reduced this dimensionality using PCA, but this is a linear transformation that can be easily learnt in one layer of the network.

以前基于深度网络的人脸识别方法使用了一个分类层[15,17]，在已知的人脸集上进行训练，通过一个中间bottleneck层作为表示来把识别问题泛化到更大的人脸集上。这种方法的缺点是其间接性和低效性，必须指望bottleneck表示泛化到新的人脸时还能工作良好，而使用bottleneck层每个人脸的表示通常很大(1000s of dimensions)。一些最近的工作[15]通过PCA进行了降维，但这只是一种线性变换，可以很容易通过网络中的一层学习到。

In contrast to these approaches, FaceNet directly trains its output to be a compact 128-D embedding using a triplet-based loss function based on LMNN [19]. Our triplets consist of two matching face thumbnails and a non-matching face thumbnail and the loss aims to separate the positive pair from the negative by a distance margin. The thumbnails are tight crops of the face area, no 2D or 3D alignment, other than scale and translation is performed.

FaceNet则使用一种基于LMNN[19]的triplet-based损失函数，直接训练出一个紧凑的128-D的嵌套。我们的triplets由两个匹配的人脸缩略图和一个不匹配的人脸缩略图组成，损失函数的目的是将两个匹配的对与不匹配的通过一个距离差分离开来。这些缩略图是面部区域的紧凑剪切块，没有2D或3D对齐，但尺度和位移是处理过的。

Choosing which triplets to use turns out to be very important for achieving good performance and, inspired by curriculum learning [1], we present a novel online negative exemplar mining strategy which ensures consistently increasing difficulty of triplets as the network trains. To improve clustering accuracy, we also explore hard-positive mining techniques which encourage spherical clusters for the embeddings of a single person.

选择那个triplets来使用，对于是否可以取得好结果非常重要，受curriculum learning[1]的启发，我们提出了一种新的在线反面典型挖掘策略，这可以保证在网络进行训练时，持续增加triplets的难度。为改进分类准确率，我们还尝试了hard-positive挖掘技术，其对单独一个人的嵌套倾向于球形聚类。

As an illustration of the incredible variability that our method can handle see Figure 1. Shown are image pairs from PIE [13] that previously were considered to be very difficult for face verification systems.

图1展示了我们的方法所能处理问题的多样性，所展示的图像对是从PIE[13]中得到的，以前的工作认为这些图像对于人脸验证系统来说非常难。

An overview of the rest of the paper is as follows: in section 2 we review the literature in this area; section 3.1 defines the triplet loss and section 3.2 describes our novel triplet selection and training procedure; in section 3.3 we describe the model architecture used. Finally in section 4 and 5 we present some quantitative results of our embeddings and also qualitatively explore some clustering results.

文章组织如下：第二部分回顾了本领域的文献；第三部分第一节定义了triplet loss，第二节描述了我们新的triplet选择方法和训练过程，第三节描述了使用的模型结构；最后在第四和第五部分我们的嵌套的一些试验数值结果，并定性的探讨了一些聚类结果。

Figure 1. Illumination and Pose invariance. Pose and illumination have been a long standing problem in face recognition. This figure shows the output distances of FaceNet between pairs of faces of the same and a different person in different pose and illumination combinations. A distance of 0.0 means the faces are
identical, 4.0 corresponds to the opposite spectrum, two different identities. You can see that a threshold of 1.1 would classify every pair correctly.

图1. 光照和姿态不变性。姿态和光照在人脸识别问题中一直是个问题，本图展示了FaceNet在不同光照和姿态组合下，对相同的人和不同的人其输出之间的距离测度大小，距离0.0意味着人脸是相同的，4.0意味着对应的人不同。可以看出，取阈值为1.1可以使每对人脸都正确分类。

## 2. Related Work

Similarly to other recent works which employ deep networks [15, 17], our approach is a purely data driven method which learns its representation directly from the pixels of the face. Rather than using engineered features, we use a large dataset of labelled faces to attain the appropriate invariances to pose, illumination, and other variational conditions.

与其他采用深度网络的近期工作[15,17]类似，我们的方法是纯粹数据驱动的，直接从人脸像素中学习得到表示，而不使用加工得出的特征。我们使用有标签的人脸大型数据集，来得到对姿态、光照和其他可变条件的合适不变性。

In this paper we explore two different deep network architectures that have been recently used to great success in the computer vision community. Both are deep convolutional networks [8, 11]. The first architecture is based on the Zeiler&Fergus [22] model which consists of multiple interleaved layers of convolutions, non-linear activations, local response normalizations, and max pooling layers. We additionally add several 1×1×d convolution layers inspired by the work of [9]. The second architecture is based on the Inception model of Szegedy et al. which was recently used as the winning approach for ImageNet 2014 [16]. These networks use mixed layers that run several different convolutional and pooling layers in parallel and concatenate their responses. We have found that these models can reduce the number of parameters by up to 20 times and have the potential to reduce the number of FLOPS required for comparable performance.

本文中我们尝试了两种不同的深度网络框架，它们在最近的计算机视觉团体中都取得了很大成功，都是深度卷积网络[8,11]。第一种架构基于Zeiler&Fergus[22]模型，由多个交错的卷积层、非线性激活函数层、局部响应归一化层以及max pooling层组成，我们受[9]的启发另外加入了几个1×1×d的卷积层。第二种框架基于Szegedy等的Inception模型，其取得了2014年的ImageNet挑战赛。这些网络使用不同的卷积和pooling的混合层，将其响应拼接起来。我们发现这些模型最高可以将参数数量减少至1/20，并有降低FLOPS的可能。

There is a vast corpus of face verification and recognition works. Reviewing it is out of the scope of this paper so we will only briefly discuss the most relevant recent work.

有很多人脸验证和人脸识别的工作，本文不能全面回顾，只能简单讨论一下相关的近期工作。

The works of [15, 17, 23] all employ a complex system of multiple stages, that combines the output of a deep convolutional network with PCA for dimensionality reduction and an SVM for classification.

文章[15,17,23]都采用了多阶段的复杂系统，即，将深度卷积网络的输出与PCA降维、SVM分类进行组合。

Zhenyao et al. [23] employ a deep network to “warp” faces into a canonical frontal view and then learn CNN that classifies each face as belonging to a known identity. For face verification, PCA on the network output in conjunction with an ensemble of SVMs is used.

Zhenyao等[23]采用一种深度网络将人脸“弯曲”成典型前视图，然后采用CNN对人脸进行分类，对应到已知个体。对于人脸验证，采用的方法是，网络输出进行PCA，然后再进行SVM分类。

Taigman et al. [17] propose a multi-stage approach that aligns faces to a general 3D shape model. A multi-class network is trained to perform the face recognition task on over four thousand identities. The authors also experimented with a so called Siamese network where they directly optimize the L1-distance between two face features. Their best performance on LFW (97.35%) stems from an ensemble of three networks using different alignments and color channels. The predicted distances (non-linear SVM predictions based on the $χ^2$ kernel) of those networks are combined using a non-linear SVM.

Taigman等[17]提出了一种多阶段的方法，将人员与一般人脸3D形状模型对齐，训练了一个网络对超过4000人的库进行人脸识别任务，作者还用一个Siamese网络进行了试验，其中直接对两个人脸特征之间的L1距离进行优化。在LFW上的最优表现是97.35%，是通过三个网络，采用了不同的对齐和色彩通道，一起得到的。这些网络预测的距离（基于$χ^2$核的非线性SVM预测）一起通过一个非线性SVM组合到一起。

Sun et al. [14, 15] propose a compact and therefore relatively cheap to compute network. They use an ensemble of 25 of these network, each operating on a different face patch. For their final performance on LFW (99.47% [15]) the authors combine 50 responses (regular and flipped). Both PCA and a Joint Bayesian model [2] that effectively correspond to a linear transform in the embedding space are employed. Their method does not require explicit 2D/3D alignment. The networks are trained by using a combination of classification and verification loss. The verification loss is similar to the triplet loss we employ [12, 19], in that it minimizes the L2-distance between faces of the same identity and enforces a margin between the distance of faces of different identities. The main difference is that only pairs of images are compared, whereas the triplet loss encourages a relative distance constraint.

Sun等[14,15]提出了一种紧凑网络，其计算量相对较少，他们将25个这种网络组合到一起，每一个都对一个不同的人脸块进行处理。作者组合了50个响应（常规的和翻转的），得到了在LFW数据集上的最优结果99.47%[15]，还采用了PCA和联合贝叶斯模型[2]，其可以在嵌套空间中有效的对应上一种线性变换。他们的方法不需要显式的2D/3D对齐。训练网络采用人脸分类和人脸验证的损失函数的组合。验证的损失函数与我们采用的triplet损失[12,19]类似，这种损失函数中，最小化相同人的人脸之间的L2距离，并在不同人之间的人脸中强加一个差值。主要的区别在于只比较了图像对，而triplet loss更倾向于相对距离约束。

A similar loss to the one used here was explored in Wang et al. [18] for ranking images by semantic and visual similarity.

Wang等[18]试验采用了类似的损失函数，通过语义和视觉相似性对图像进行排序。

## 3. Method

FaceNet uses a deep convolutional network. We discuss two different core architectures: The Zeiler&Fergus [22] style networks and the recent Inception [16] type networks. The details of these networks are described in section 3.3.

FaceNet采用深度卷积网络。我们讨论了两个不同的核心架构，Zeiler&Fergus[22]式的网络，和最近的Inception[16]式网络。网络的细节在3.3节中讨论。

Given the model details, and treating it as a black box (see Figure 2), the most important part of our approach lies in the end-to-end learning of the whole system. To this end we employ the triplet loss that directly reflects what we want to achieve in face verification, recognition and clustering. Namely, we strive for an embedding *f(x)*, from an image *x* into a feature space $R^d$, such that the squared distance between all faces, independent of imaging conditions, of the same identity is small, whereas the squared distance between a pair of face images from different identities is large.

将模型视为黑盒，见图2，我们的方法中最重要的部分在于整个系统端到端的学习。在这一端是triplet loss，它直接反应了我们想要在人脸验证、识别和聚类中需要的量。即，我们努力得到一个嵌套*f(x)*，从图像*x*映射到特征空间$R^d$，这样其所有人脸之间的距离平方，与图像条件无关，同一个人的就小，不同人之间的就大。

Figure 2. Model structure. Our network consists of a batch input layer and a deep CNN followed by L2 normalization, which results in the face embedding. This is followed by the triplet loss during training.

图2 模型结构。我们的网络包括批量输入层，深度CNN层，然后是L2正则化，后面就是人脸嵌入，最后得到训练中用到的triplet loss

Although we did not directly compare to other losses, e.g. the one using pairs of positives and negatives, as used in [14] Eq. (2), we believe that the triplet loss is more suitable for face verification. The motivation is that the loss from [14] encourages all faces of one identity to be projected onto a single point in the embedding space. The triplet loss, however, tries to enforce a margin between each pair of faces from one person to all other faces. This allows the faces for one identity to live on a manifold, while still enforcing the distance and thus discriminability to other identities.

虽然我们没有直接与其他损失函数比较，比如，[14]中采用正负图像对的公式(2)，我们相信triplet loss更适合于人脸验证应用。动机是[14]的损失函数倾向于使同一个人的所有人脸图像映射到嵌套空间的同一个点，而triplet loss则是使一个人脸与其他人的人脸间有一个差值余量。这使得同一个人的人脸存在于一个流形中，而仍然保证与其他人人脸图像的距离也就是可鉴别性。

The following section describes this triplet loss and how it can be learned efficiently at scale.

下一节描述了triplet loss以及怎样有效的学习到正确的值。

### 3.1. Triplet Loss

The embedding is represented by $f(x) ∈ R^d$ . It embeds an image *x* into a *d*-dimensional Euclidean space. Additionally, we constrain this embedding to live on the *d*-dimensional hypersphere, $i.e. \vert\vert f(x) \vert\vert 2$ = 1. This loss is motivated in [19] in the context of nearest-neighbor classification. Here we want to ensure that an image $x_i^a$ (anchor) of a specific person is closer to all other images $x_i^p$ (positive) of the same person than it is to any image $x_i^n$ (negative) of any other person. This is visualized in Figure 3.

嵌套是由$f(x) ∈ R^d$表示的，它将图像*x*嵌套进一个*d*维欧几里得空间。另外，我们约束这个嵌套只分布在*d*维超球上，即，$\vert\vert f(x) \vert\vert ^2$ = 1，这个损失函数受[19]中的最近邻分类启发得到。这里我们需要确保一个人的人脸图像$x_i^a$ (anchor)与这个人的所有其他人脸图像$x_i^p$ (positive)之间的距离，比与其他人的任何人脸图像$x_i^n$ (negative)都要小，如图3所示。

Figure 3. The Triplet Loss minimizes the distance between an anchor and a positive, both of which have the same identity, and maximizes the distance between the anchor and a negative of a different identity.

图3 Triplet loss将anchor与positive的距离最小化，它们对应的是同一个人，将anchor与negative之间的距离最大化，它们对应不同的人

Thus we want 所以我们是要

$$\vert \vert f(x_i^a) - f(x_i^p) \vert \vert _2^2 + \alpha < \vert \vert f(x_i^a) - f(x_i^n) \vert \vert _2^2$$(1)

for any $(f(x_i^a), f(x_i^p, f(x_i^n) ∈  T$

where *α* is a margin that is enforced between positive and negative pairs. *T* is the set of all possible triplets in the training set and has cardinality *N*.

这里*α*是正负图像对之间强加的余量，*T*是训练集中所有可能的triplet组成的空间，其基为*N*。

The loss that is being minimized is then 进行最小化的损失函数为

$$L = \sum_i^N [\vert \vert f(x_i^a) - f(x_i^p) \vert \vert _2^2 - \vert \vert f(x_i^a) - f(x_i^n) \vert \vert _2^2 + \alpha]_+$$(3)

Generating all possible triplets would result in many triplets that are easily satisfied (i.e. fulfill the constraint in Eq. (1)). These triplets would not contribute to the training and result in slower convergence, as they would still be passed through the network. It is crucial to select hard triplets, that are active and can therefore contribute to improving the model. The following section talks about the different approaches we use for the triplet selection.

所有可能的triplet集中，有很多是很容易满足公式(1)的条件的，这些triplet不能对网络训练做出贡献，会导致收敛变慢，因为它们会一直满足网络条件。选择合适的hard triplets是非常关键的，要选择活跃的，能够为改善模型做出贡献的。下一节我们讨论所使用的不同的triplet选择方法。

### 3.2. Triplet Selection

In order to ensure fast convergence it is crucial to select triplets that violate the triplet constraint in Eq. (1). This means that, given $x_i^a$, we want to select an $x_i^p$ (hard positive) such that $argmax_{x_i^p} \vert \vert f(x_i^a) - f(x_i^p) \vert \vert _2^2$ and similarly $x_i^n$ (hard negative) such that $argmin_{x_i^n} \vert \vert f(x_i^a) - f(x_i^n) \vert \vert _2^2$.

为了确保快速收敛，选择违反公式(1)triplet约束的triplets是非常重要的，这个意思是，给定$x_i^a$，我们想要选择$x_i^p$ (hard positive)使得$argmax_{x_i^p} \vert \vert f(x_i^a) - f(x_i^p) \vert \vert _2^2$，类似的选择$x_i^n$ (hard negative)使得$argmin_{x_i^n} \vert \vert f(x_i^a) - f(x_i^n) \vert \vert _2^2$。

It is infeasible to compute the argmin and argmax across the whole training set. Additionally, it might lead to poor training, as mislabelled and poorly imaged faces would dominate the hard positives and negatives. There are two obvious choices that avoid this issue:

不可能在整个训练集上计算argmin和argmax，另外，由于标签错误的和拍照效果不好的人脸会充斥这些hard positives and negatives中，所以可能会导致较差的训练，为了不发生这样的情况，有两个很显然的选择：

- Generate triplets offline every n steps, using the most recent network checkpoint and computing the argmin and argmax on a subset of the data.
- Generate triplets online. This can be done by selecting the hard positive/negative exemplars from within a mini-batch.

- 每n步离线生成triplets，用网络最近的checkpoint，在数据集的子集上计算argmin和argmax。
- 在线生成triplets，通过在一个mini-batch图像集中选择hard positive/negative典型。

Here, we focus on the online generation and use large mini-batches in the order of a few thousand exemplars and only compute the argmin and argmax within a mini-batch.

这里，我们关注在线生成法，并使用较大规模的mini-batch，大约有几千个样本，只在一个mini-batch内部计算argmin和argmax。

To have a meaningful representation of the anchor-positive distances, it needs to be ensured that a minimal number of exemplars of any one identity is present in each mini-batch. In our experiments we sample the training data such that around 40 faces are selected per identity per mini-batch. Additionally, randomly sampled negative faces are added to each mini-batch.

为了对anchor-positive距离进行有意义的表示，需要确保每个mini-batch中都会包含那个人的最小数量的样本。在我们的实验中，我们对训练数据进行采样，在每个mini-batch中都有那个人的40张人脸图像。另外，会有随机数量的negative人脸包括在每个mini-batch中。

Instead of picking the hardest positive, we use all anchor-positive pairs in a mini-batch while still selecting the hard negatives. We don’t have a side-by-side comparison of hard anchor-positive pairs versus all anchor-positive pairs within a mini-batch, but we found in practice that the all anchor-positive method was more stable and converged slightly faster at the beginning of training.

我们不是选择hardest positive，而是用一个mini-batch中所有的anchor-positive对，同时仍然选择hard negatives。我们没有同时比较一个mini-batch中的hard anchor-positive对和所有的anchor-positive对，但我们在实践中发现采用所有anchor-positive的方法更加稳定，而且在训练初期收敛略快。

We also explored the offline generation of triplets in conjunction with the online generation and it may allow the use of smaller batch sizes, but the experiments were inconclusive.

我们也研究了离线生成triplet法与在线生成法共同作用，这可以使用更小的批次规模，但试验结论并不明确。

Selecting the hardest negatives can in practice lead to bad local minima early on in training, specifically it can result in a collapsed model ($i.e. f(x)$ = 0). In order to mitigate this, it helps to select $x^n_i$ such that

选择the hardest negatives在实践中可能导致训练初期的局部极值陷阱，有一个特例是，可以导致模型崩溃($i.e. f(x)$ = 0)。为了缓和这种情况，选择$x^n_i$满足

$$\vert \vert f(x_i^a) - f(x_i^p) \vert \vert _2^2 < \vert \vert f(x_i^a) - f(x_i^n) \vert \vert _2^2$$(4)

We call these negative exemplars semi-hard, as they are further away from the anchor than the positive exemplar, but still hard because the squared distance is close to the anchor-positive distance. Those negatives lie inside the margin *α*.

我们称这种negative样本semi-hard，因为它们距离anchor比positive要远，但仍然是hard因为其距离平方与anchor-positive距离是接近的，这些negative在余量*α*内部。

As mentioned before, correct triplet selection is crucial for fast convergence. On the one hand we would like to use small mini-batches as these tend to improve convergence during Stochastic Gradient Descent (SGD) [20]. On the other hand, implementation details make batches of tens to hundreds of exemplars more efficient. The main constraint with regards to the batch size, however, is the way we select hard relevant triplets from within the mini-batches. In most experiments we use a batch size of around 1,800 exemplars.

前面提到，正确的triplet选择对于快速收敛是很关键的。一方面我们希望用小的mini-batch，因为在随机梯度下降(SGD)[20]时会改善收敛情况。另一方面，算法实现细节使数十到数百个样本的batch更加高效。但在batch size方面的主要约束是，我们的mini-batch内部选择相关的hard triplets的方法。在多数试验中，我们使用的batch size约为1800个样本。

### 3.3. Deep Convolutional Networks

In all our experiments we train the CNN using Stochastic Gradient Descent (SGD) with standard backprop [8,11] and AdaGrad [5]. In most experiments we start with a learning rate of 0.05 which we lower to finalize the model. The models are initialized from random, similar to [16], and trained on a CPU cluster for 1,000 to 2,000 hours. The decrease in the loss (and increase in accuracy) slows down drastically after 500h of training, but additional training can still significantly improve performance. The margin *α* is set to 0.2.

我们的所有试验中，都用随机梯度下降(SGD)法、标准backprop[8,11]和AdaGrad[5]训练CNN。在大多数试验中我们开始使用的学习率为0.05，随着训练的进行慢慢减小学习率最后完成模型训练。模型随机初始化，与[16]类似，在一个CPU集群上训练1000到2000小时。损失函数的下降（准确率的增加）在500小时后迅速慢下来，但额外的训练仍然可以明显改善表现。余量*α*设为0.2。

We used two types of architectures and explore their trade-offs in more detail in the experimental section. Their practical differences lie in the difference of parameters and FLOPS. The best model may be different depending on the application. E.g. a model running in a datacenter can have many parameters and require a large number of FLOPS, whereas a model running on a mobile phone needs to have few parameters, so that it can fit into memory. All our models use rectified linear units as the non-linear activation function.

我们用两种架构的网络并在试验中细致探讨它们的优劣点。实践中它们的区别在于参数数量和FLOPS的区别。最好的模型可能根据应用不同而不一样，比如，运行在数据中心的模型可以有很多参数，进行很多FLOPS的运算，但运行在手机上的模型需要减少参数，这样才能满足存储需要。我们所有的模型都用ReLU作为非线性激活函数。

The first category, shown in Table 1, adds 1×1×d convolutional layers, as suggested in [9], between the standard convolutional layers of the Zeiler&Fergus [22] architecture and results in a model 22 layers deep. It has a total of 140 million parameters and requires around 1.6 billion FLOPS per image.

如表1所示，我们所用的第一类模型是Zeiler&Fergus [22]结构，并按照[9]中的建议在标准卷积层中加入了1×1×d卷积层，结果模型有22层深，共有1.4亿个参数，每幅图像需要进行16亿FLOPS的计算。

Table 1. NN1. This table show the structure of our Zeiler&Fergus [22] based model with 1×1 convolutions inspired by [9]. The input and output sizes are described in rows × cols × #filters. The kernel is specified as rows × cols,stride and the maxout [6] pooling size as p = 2.

表1 NN1. 这个表是我们采用的Zeiler&Fergus [22]架构与[9]中的1×1卷积层的混合，输入与输出大小为rows×cols×#filters，卷积核大小为rows×cols，卷积步长stride和maxout[6] pooling大小为p=2

layer | size-in | size-out | kernel | param | FLOPS
--- | --- | --- | --- | --- | ---
conv1 | 220×220×3 | 110×110×64 | 7×7×3,2 | 9K | 115M
pool1 | 110×110×64 | 55×55×64 | 3×3×64,2 | 0 |
rnorm1 | 55×55×64 | 55×55×64 | | 0 | 
conv2a | 55×55×64 | 55×55×64 | 1×1×64,1 | 4K | 13M
conv2 | 55×55×64 | 55×55×192 | 3×3×64,1 | 111K | 335M
rnorm2 | 55×55×192 | 55×55×192 | | 0 |
pool2 | 55×55×192 | 28×28×192 | 3×3×192,2 | 0
conv3a | 28×28×192 | 28×28×192 | 1×1×192,1 | 37K | 29M
conv3 | 28×28×192 | 28×28×384 | 3×3×192,1 | 664K | 521M
pool3 | 28×28×384 | 14×14×384 | 3×3×384,2 | 0 |
conv4a | 14×14×384 | 14×14×384 | 1×1×384,1 | 148K | 29M
conv4 | 14×14×384 | 14×14×256 | 3×3×384,1 | 885K | 173M
conv5a | 14×14×256 | 14×14×256 | 1×1×256,1 | 66K | 13M
conv5 | 14×14×256 | 14×14×256 | 3×3×256,1 | 590K | 116M
conv6a | 14×14×256 | 14×14×256 | 1×1×256,1 | 66K | 13M
conv6 | 14×14×256 | 14×14×256 | 3×3×256,1 | 590K | 116M
pool4 | 14×14×256 | 7×7×256 | 3×3×256,2 | 0 |
concat | 7×7×256 | 7×7×256 | | 0 |
fc1 | 7×7×256 | 1×32×128 | maxout p=2 | 103M | 103M
fc2 | 1×32×128 | 1×32×128 | maxout p=2 | 34M | 34M
fc7128 | 1×32×128 | 1×1×128 | | 524K | 0.5M
L2 | 1×1×128 | 1×1×128 | | 0 |
total | | | | 140M | 1.6B

The second category we use is based on GoogLeNet style Inception models [16]. These models have 20× fewer parameters(around6.6M-7.5M) and up to 5× fewer FLOPS (between 500M-1.6B). Some of these models are dramatically reduced in size (both depth and number of filters), so that they can be run on a mobile phone. One, NNS1, has 26M parameters and only requires 220M FLOPS per image. The other, NNS2, has 4.3M parameters and 20M FLOPS. Table 2 describes NN2 our largest network in detail. NN3 is identical in architecture but has a reduced input size of 160x160. NN4 has an input size of only 96x96, thereby drastically reducing the CPU requirements (285M FLOPS vs 1.6B for NN2). In addition to the reduced input size it does not use 5x5 convolutions in the higher layers as the receptive field is already too small by then. Generally we found that the 5x5 convolutions can be removed throughout with only a minor drop in accuracy. Figure 4 compares all our models.

我们用的第二类是基于GoogLeNet式的Inception模型[16]，这些模型参数只有1/20（约660万-750万），FLOPS只有1/5（约5亿-16亿），有些模型规模非常小（包括深度和滤波器数量），这样才可以在手机中运行。有一种MNS1有2600万个参数，每幅图只需要2.2亿FLOPS运算量，另外一种MNS2模型，430万参数，2000万FLOPS运算量。表2详细描述了NN2是我们最大的模型。NN3的结构是一样的，但输入尺寸只有160×160，NN4输入尺寸只有96×96，所以对CPU的要求下降很多（2.85亿FLOPS，NN2是16亿FLOPS）。除了输入尺寸很小，在更高的层中，不使用5×5的卷积，因为其感受野已经非常小了。一般来说，5×5卷积可以全部移除，造成的准确度下降非常小。图4对我们的所有模型做了对比。

Figure 4. FLOPS vs. Accuracy trade-off. Shown is the trade-off between FLOPS and accuracy for a wide range of different model sizes and architectures. Highlighted are the four models that we focus on in our experiments.

图4 FLOPS与Accuracy的折中，所示的是不同规模和架构的众多模型其计算量和准确度的对比，高亮的是我们关注的四种模型。

Table 2. NN2. Details of the NN2 Inception incarnation. This model is almost identical to the one described in [16]. The two major differences are the use of L2 pooling instead of max pooling (m), where specified. I.e. instead of taking the spatial max the L2 norm is computed. The pooling is always 3×3 (aside from the final average pooling) and in parallel to the convolutional modules inside each Inception module. If there is a dimensionality reduction after the pooling it is denoted with p. 1×1, 3×3, and 5×5 pooling are then concatenated to get the final output.

表2 NN2. Inception架构的NN2的细节。这个模型与[16]中的模型几乎一样，两个主要的区别是用了L2 pooling而不是max pooling，即不是取空域最大值，而是计算其元素的L2 norm。除了最后的average pooling，其他pooling都是3×3，而且是与其他卷积模块并行的在每个Inception模型里。如果在pooling后有降维，1×1,3×3,5×5会拼接起来得到最终输出。

type | output size | depth | #1×1 | #3×3 reduce | #3×3 | #5×5 reduce | #5×5 | pool proj (p) | params | FLOPS
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
conv1 (7×7×3,2) | 112×112×64 | 1 | | | | | | | 9K | 119M
max pool + norm | 56×56×64 | 0 | | | | | | m 3×3,2 | |
inception (2) | 56×56×192 | 2 | | 64 | 192 | | | | 115K | 360M
norm + max pool | 28×28×192 | 0 | | | | | | m 3×3,2 | |
inception (3a) | 28×28×256 | 2 | 64 | 96 | 128 | 16 | 32 | m, 32p | 164K | 128M
inception (3b) | 28×28×320 | 2 | 64 | 96 | 128 | 32 | 64 | L2 , 64p | 228K | 179M
inception (3c) | 14×14×640 | 2 | 0 | 128 | 256,2 | 32 | 64,2 | m 3×3,2 | 398K | 108M
inception (4a) | 14×14×640 | 2 | 256 | 96 | 192 | 32 | 64 | L2 , 128p | 545K | 107M
inception (4b) | 14×14×640 | 2 | 224 | 112 | 224 | 32 | 64 | L2 , 128p | 595K | 117M
inception (4c) | 14×14×640 | 2 | 192 | 128 | 256 | 32 | 64 | L2 , 128p | 654K | 128M
inception (4d) | 14×14×640 | 2 | 160 | 144 | 288 | 32 | 64 | L2 , 128p | 722K | 142M
inception (4e) | 7×7×1024 | 2 | 0 | 160 | 256,2 | 64 | 128,2 | m 3×3,2 | 717K | 56M
inception (5a) | 7×7×1024 | 2 | 384 | 192 | 384 | 48 | 128 | L2 , 128p | 1.6M | 78M
inception (5b) | 7×7×1024 | 2 | 384 | 192 | 384 | 48 | 128 | m, 128p | 1.6M | 78M
avg pool | 1×1×1024 | 0 | 
fully conn | 1×1×128 | 1 | | | | | | | 131K | 0.1M
L2 normalization | 1×1×128 | 0
total | | | | | | | | | 7.5M | 1.6B

## 4. Datasets and Evaluation

We evaluate our method on four datasets and with the exception of Labelled Faces in the Wild and YouTube Faces we evaluate our method on the face verification task. I.e. given a pair of two face images a squared $L_2$ distance threshold $D(x_i ,x_j)$ is used to determine the classification of same and different. All faces pairs (*i,j*) of the same identity are denoted with $P_{same}$, whereas all pairs of different identities are denoted with $P_{diff}$.

我们在四个数据集上评估我们的方法，在LFW和Youtube Faces上进行人脸验证工作来评估我们的方法。即，给定两张人脸图像，用$L_2$距离平方的阈值$D(x_i ,x_j)$来确定是不是同一个人。所有的相同人的人脸图像对(*i,j*)用$P_{same}$表示，不同人的则用$P_{diff}$表示。

We define the set of all *true accepts* as 我们定义全部*true accepts*的集合为

$$TA(d)=\{ (i,j)∈P_{same}, with D(x_i, x_j)≤d \}$$(5)

These are the face pairs (*i,j*) that were correctly classified as *same* at threshold *d*. Similarly

这是以阈值*d*正确人类为*相同*的人脸图像对(*i,j*)，类似的

$$FA(d)=\{ (i,j)∈P_{diff}, with D(x_i, x_j)≤d \}$$(6)

is the set of all pairs that was incorrectly classified as *same* (*false accept*).

是所有的对都被错误分类为*相同*的集合(*false accept*)。

The validation rate VAL(*d*) and the false accept rate FAR(*d*) for a given face distance *d* are then defined as

对于给定的人脸距离*d*，验证率VAL(*d*)和错误接受率FAR(*d*)定义为

$$VAL(d)=\frac{TA(d)}{P_{same}}, FAR(d)=\frac{FA(d)}{P_{diff}}$$(7)

### 4.1. Hold-out Test Set 秘密测试集

We keep a hold out set of around one million images, that has the same distribution as our training set, but disjoint identities. For evaluation we split it into five disjoint sets of 200k images each. The FAR and VAL rate are then computed on 100k × 100k image pairs. Standard error is reported across the five splits.

我们保有一个秘密集，有大约100万张图像，与训练集的分布一样，但其中没有同样的人。我们将其分为5个没有人员交集的集合，每个20万图像，作为评估用。FAR和VAL在10万×10万的图像对中计算，5个集合都记录了标准错误。

### 4.2. Personal Photos 个人图像

This is a test set with similar distribution to our training set, but has been manually verified to have very clean labels. It consists of three personal photo collections with a total of around 12k images. We compute the FAR and VAL rate across all 12k squared pairs of images.

这个测试集与我们的训练集分布类似，但已经人工核实过，其标签非常干净，集合由3个个人图像集组成，共有1.2万图片。我们在这1.2万平方个图像对上计算了FAR和VAL。

### 4.3. Academic Datasets 学术数据集

Labeled Faces in the Wild (LFW) is the de-facto academic test set for face verification [7]. We follow the standard protocol for unrestricted, labeled outside data and report the mean classification accuracy as well as the standard error of the mean.

LFW是人脸验证的de-facto学术测试集[7]，我们遵循其标准协议，采用不做限制的标记野外数据，得到了平均分类准确度和平均标准错误率。

Youtube Faces DB [21] is a new dataset that has gained popularity in the face recognition community [17, 15]. The setup is similar to LFW, but instead of verifying pairs of images, pairs of videos are used.

Youtube Face DB[21]是在人脸识别团体中越来越流行的新数据集[17,15]，其设置与LFW类似，但不是验证图像对，而是视频对。

## 5. Experiments 试验

If not mentioned otherwise we use between 100M-200M training face thumbnails consisting of about 8M different identities. A face detector is run on each image and a tight bounding box around each face is generated. These face thumbnails are resized to the input size of the respective
network. Input sizes range from 96x96 pixels to 224x224 pixels in our experiments.

没有另外说明的话，我们使用1亿-2亿训练用人脸缩略图包括大约8百万不同的人。每个图像都会用人脸检测器处理，每个人脸外都紧贴着的方框。这些人脸缩略图的大小都处理成网络输入的标准大小，输入尺寸从96×96到224×224几种。

## 5.1. Computation Accuracy Trade-off 计算量准确率的折中

Before diving into the details of more specific experiments we will discuss the trade-off of accuracy versus number of FLOPS that a particular model requires. Figure 4 shows the FLOPS on the x-axis and the accuracy at 0.001 false accept rate (FAR) on our user labelled test-data set
from section 4.2. It is interesting to see the strong correlation between the computation a model requires and the accuracy it achieves. The figure highlights the five models (NN1, NN2, NN3, NNS1, NNS2) that we discuss in more detail in our experiments.

在讨论特定试验的细节之前，我们先讨论一下准确率和特定应用所需的FLOPS量之间的折中。图4的x轴是FLOPS，y轴是以0.001位单位的FAR，用的是4.2中的用户标记的测试数据。可以看到计算量与准确度的相关性是非常强的，图里重点强调了我们讨论的最多的五种模型NN1, NN2, NN3, NNS1, NNS2。

We also looked into the accuracy trade-off with regards to the number of model parameters. However, the picture is not as clear in that case. For example, the Inception based model NN2 achieves a comparable performance to NN1, but only has a 20th of the parameters. The number of FLOPS is comparable, though. Obviously at some point the performance is expected to decrease, if the number of parameters is reduced further. Other model architectures may allow further reductions without loss of accuracy, just like Inception [16] did in this case.

我们还探讨了准确度与模型参数数量之间的关系。但是，不是很明确。比如，基于Inception的模型NN2与NN1性能类似，但只有1/20的参数，但其计算量FLOPS也是差不多的。很明显如果参数进一步减少，那么在某处性能就会下降。其他模型结构可能允许进一步减少，而准确率保持不变，就像Inception[16]在本例子中的情况一样。

## 5.2. Effect of CNN Model CNN模型的影响

We now discuss the performance of our four selected models in more detail. On the one hand we have our traditional Zeiler&Fergus based architecture with 1×1 convolutions[22, 9] (see Table 1). On the other hand we have Inception [16] based models that dramatically reduce the model size. Overall, in the final performance the top models of both architectures perform comparably. However, some of our Inception based models, such as NN3, still achieve good performance while significantly reducing both the FLOPS and the model size.

我们现在仔细讨论我们选定的四种模型的表现。一方面我们有基于Zeiler&Fergus的架构与1×1卷积混合的模型[22,9] (见表1)，另一方面我们有基于Inception[16]的模型，可以极大降低模型尺寸。总体来说，两种架构表现最好的模型差距不大。但一些基于Inception的模型，比如NN3，在大幅降低计算量和模型规模时，仍然可以取得很好的结果。

The detailed evaluation on our personal photos test set is shown in Figure 5. While the largest model achieves a dramatic improvement in accuracy compared to the tiny NNS2, the latter can be run 30ms / image on a mobile phone and is still accurate enough to be used in face clustering. The sharp drop in the ROC for FAR < $10^{−4}$ indicates noisy labels in the test data groundtruth. At extremely low false accept rates a single mislabeled image can have a significant impact on the curve.

图5展示了个人图像测试集的评估细节。最大的模型与极小模型NNS2相比，准确率改善是非常可观的，但后者可以以每幅图30ms的速度在手机上运行，而且用在图像分类中仍然足够准确。ROC曲线在FAR < $10^{−4}$时的剧烈下降，说明groundtruth数据有含噪的标签。在极低的FAR值时，一幅误标记的图会对曲线有很大的影响。

Figure 5. Network Architectures. This plot shows the complete ROC for the four different models on our personal photos test set from section 4.2. The sharp drop at 10 E -4 FAR can be explained by noise in the groundtruth labels. The models in order of performance are: NN2: 224×224 input Inception based model; NN1: Zeiler&Fergus based network with 1×1 convolutions; NNS1: small Inception style model with only 220M FLOPS; NNS2: tiny Inception model with only 20M FLOPS.

### 5.3. Sensitivity to Image Quality 图像质量敏感性

Table 4 shows the robustness of our model across a wide range of image sizes. The network is surprisingly robust with respect to JPEG compression and performs very well down to a JPEG quality of 20. The performance drop is very small for face thumbnails down to a size of 120x120 pixels and even at 80x80 pixels it shows acceptable performance. This is notable, because the network was trained on 220x220 input images. Training with lower resolution faces could improve this range further.

表4所示的是我们的模型对于图像尺寸变化的鲁棒性。网络对于JPEG压缩的鲁棒性是非常好的，在JPEG压缩质量为20的时候仍然表现很好。人脸缩略图的尺寸到了120×120的时候，准确度降低仍然非常小，即使在80×80的情况下，表现仍然可以接受。这是值得注意的，因为网络是在220×220的输入图像下训练出来的。如果训练时就采用低分辨率图像，那么会进一步改善性能。

Table 4. Image Quality. The table on the left shows the effect on the validation rate at 10 E -3 precision with varying JPEG quality. The one on the right shows how the image size in pixels effects the validation rate at 10 E -3 precision. This experiment was done with NN1 on the first split of our test hold-out dataset.

jpeg q | val-rate | | pixels | val-rate
--- | --- | --- | --- | ---
10 | 67.3% | | 1,600 | 37.8%
20 | 81.4% | | 6,400 | 79.5%
30 | 83.9% | | 14,400 | 84.5%
50 | 85.5% | | 25,600 | 85.7%
70 | 86.1% | | 65,536 | 86.4%
90 | 86.5% | | 

### 5.4. Embedding Dimensionality 嵌套维数

We explored various embedding dimensionalities and selected 128 for all experiments other than the comparison reported in Table 5. One would expect the larger embeddings to perform at least as good as the smaller ones, however, it is possible that they require more training to achieve the same accuracy. That said, the differences in the performance reported in Table 5 are statistically insignificant.

我们试验了多种嵌套维数，在表5中进行了对比，最后选择了128维。一般认为高维嵌套至少会和低维的表现一样好，但表中不是这样的，也可能经过更多的训练会取得相同的准确率。这就是说，表5中所示的准确率差异在统计意义上是不明显的。

Table 5. Embedding Dimensionality. This Table compares the effect of the embedding dimensionality of our model NN1 on our hold-out set from section 4.1. In addition to the VAL at 10E-3 we also show the standard error of the mean computed across five splits.

dims | VAL
--- | ---
64 | 86.8% ± 1.7
128 | 87.9% ± 1.9
256 | 87.7% ± 1.9

It should be noted, that during training a 128 dimensional float vector is used, but it can be quantized to 128-bytes without loss of accuracy. Thus each face is compactly represented by a 128 dimensional byte vector, which is ideal for large scale clustering and recognition. Smaller embeddings are possible at a minor loss of accuracy and could be employed on mobile devices.

应当指出，在训练的过程中使用的是128维浮点矢量，但可以量化为128字节而没有准确率损失。所以每个人脸可以紧凑的表示为一个128维的字节矢量，这对于大规模分类和识别是很理想的。更小的嵌套会损失很小的准确率，可以在移动设备上运行。

### 5.5. Amount of Training Data 训练数据集的规模

Table 6 shows the impact of large amounts of training data. Due to time constraints this evaluation was run on a smaller model; the effect may be even larger on larger models. It is clear that using tens of millions of exemplars results in a clear boost of accuracy on our personal photo test set from section 4.2. Compared to only millions of images the relative reduction in error is 60%. Using another order of magnitude more images (hundreds of millions) still gives a small boost, but the improvement tapers off.

表6所示的是大规模训练数据的影响。受时间约束，这项评估只在较小的模型上进行，在更大的模型上，其影响可能更大。在4.2节的个人图像测试数据集中，很明显使用上千万的样本训练会使准确度大幅提升。与只有百万级图像的结果相比，其错误率下降达60%。使用更高量级的图像训练只会带来很小的提升了，改进逐渐变小。

Table6.Training Data Size. This table compares the performance after 700h of training for a smaller model with 96x96 pixel inputs. The model architecture is similar to NN2, but without the 5x5 convolutions in the Inception modules.

training images | VAL
--- | ---
2,600,000 | 76.3%
26,000,000 | 85.1%
52,000,000 | 85.1%
260,000,000 | 86.2%

### 5.6. Performance on LFW 在LFW数据集上的表现

We evaluate our model on LFW using the standard protocol for unrestricted, labeled outside data. Nine training splits are used to select the L2-distance threshold. Classification (same or different) is then performed on the tenth test split. The selected optimal threshold is 1.242 for all test splits except split eighth (1.256).

我们在LFW数据集上用标准协议为不受限的标记室外数据进行了评估。用了9个训练分集来选择L2距离的阈值，然后对第10个分测试集进行分类（相同或不同）。除了第8个测试集选择的最优参数为1.256外，其余测试集选出的最优参数都是1.242。

Our model is evaluated in two modes: 我们的模型在两种模式下评估

1. Fixed center crop of the LFW provided thumbnail.
2. A proprietary face detector (similar to Picasa[3]) is run on the provided LFW thumbnails. If it fails to align the face(this happens for two images), the LFW alignment is used.

1. LFW提供的缩略图的固定中间剪切块；
2. 在提供的LFW缩略图上运行专用人脸检测器（与Picasa[3]类似），如果没能对齐人脸（在2个图像上发生了这种情况），则使用LFW对齐。

Figure 6 gives an overview of all failure cases. It shows false accepts on the top as well as false rejects at the bottom. We achieve a classification accuracy of 98.87%±0.15 when using the fixed center crop described in (1) and the record breaking 99.63%±0.09 standard error of the mean when using the extra face alignment (2). This reduces the error reported for DeepFace in [17] by more than a factor of 7 and the previous state-of-the-art reported for DeepId2+ in [15] by 30%. This is the performance of model NN1, but even the much smaller NN3 achieves performance that is not statistically significantly different.

图6给出了所有失败案例的概览。上部是false accepts，下部是false rejects。当使用(1)中的固定中间剪切块时，我们取得的分类正确率为98.87%±0.15，当使用额外的(2)人脸对齐时，达到了99.63%±0.09。这比DeepFace[17]中给出的错误率低了7个百分点，比[15]中的DeepID2+低了30%。这是模型NN1的表现，但即使小了很多的NN3其表现没有低很多。

Figure 6. LFW errors. This shows all pairs of images that were incorrectly classified on LFW. Only eight of the 13 false rejects shown here are actual errors the other five are mislabeled in LFW.

### 5.7. Performance on Youtube Faces DB 在Youtube Faces数据集上的表现

We use the average similarity of all pairs of the first one hundred frames that our face detector detects in each video. This gives us a classification accuracy of 95.12%±0.39. Using the first one thousand frames results in 95.18%. Compared to [17] 91.4% who also evaluate one hundred frames per video we reduce the error rate by almost half. DeepId2+ [15] achieved 93.2% and our method reduces this error by 30%, comparable to our improvement on LFW.

我们的人脸检测器在每个视频中检测前100帧图像，我们用所有图像对的平均相似度，这使我们的分类准确度达到了95.12%±0.39，使用前1000帧，达到95.18%。[17]也是使用的100帧，其正确率为91.4%，我们将错误率降低了接近一半。DeepId2+[15]取得了93.2%的结果，我们的将其错误率降低了30%，与在LFW上的改进类似。

### 5.8. Face Clustering 人脸聚类

Our compact embedding lends itself to be used in order to cluster a users personal photos into groups of people with the same identity. The constraints in assignment imposed by clustering faces, compared to the pure verification task,lead to truly amazing results. Figure 7 shows one cluster in a users personal photo collection, generated using agglomerative clustering. It is a clear showcase of the incredible invariance to occlusion, lighting, pose and even age.

我们的紧凑嵌套使模型可以对用户个人照片按照不同的人进行分类。The constraints in assignment imposed by clustering faces, compared to the pure verification task,lead to truly amazing results. 图7所示的是用户个人图像数据集中的一类，由聚类算法生成，其中展示了对遮挡、光照、姿态甚至年龄的不变性。

Figure 7. Face Clustering. Shown is an exemplar cluster for one user. All these images in the users personal photo collection were clustered together.

## 6. Summary

We provide a method to directly learn an embedding into an Euclidean space for face verification. This sets it apart from other methods [15, 17] who use the CNN bottleneck layer, or require additional post-processing such as concatenation of multiple models and PCA, as well as SVM classification. Our end-to-end training both simplifies the setup and shows that directly optimizing a loss relevant to the task at hand improves performance.

我们提出了一种方法可以直接学习一种嵌套到欧几里得空间，来进行人脸验证。这与[15,17]那些用CNN bottleneck层的，或需要另外的后续处理，比如与多个模型及PCA、SVM衔接。我们的端到端的训练方法简化了算法设置，直接优化与任务相关的损失函数还可以改进性能。

Another strength of our model is that it only requires minimal alignment (tight crop around the face area). [17], for example, performs a complex 3D alignment. We also experimented with a similarity transform alignment and notice that this can actually improve performance lightly. It is not clear if it is worth the extra complexity.

我们的模型的另一个优势是它只需要对图像最小的对齐（围绕人脸外缘剪切），比如[17]需要一种复杂的3D对齐。我们还进行了试验，如果进行一种相似性变换的对齐，会略微改善模型，但仍不清楚与增加的复杂性相比是否值得。

Future work will focus on better understanding of the error cases, further improving the model, and also reducing model size and reducing CPU requirements. We will also look into ways of improving the currently extremely long training times, e.g. variations of our curriculum learning with smaller batch sizes and offline as well as online positive and negative mining.

未来的工作将会对错误的案例进行更好的理解，进一步改进模型，以及缩小模型规模，降低CPU需求。我们还会寻找方法改善现在训练时间过长的问题，比如我们的curriculum学习，采用更小的batch size，离线及在线positive and negative挖掘。

## 7. Appendix: Harmonic Embedding 谐波嵌套

In this section we introduce the concept of harmonic embeddings. By this we denote a set of embeddings that are generated by different models v1 and v2 but are compatible in the sense that they can be compared to each other. This compatibility greatly simplifies upgrade paths. E.g. in an scenario where embedding v1 was computed across a large set of images and a new embedding model v2 is being rolled out, this compatibility ensures a smooth transition without the need to worry about version incompatibilities. Figure 8 shows results on our 3G dataset. It can be seen that the improved model NN2 significantly outperforms NN1, while the comparison of NN2 embeddings to NN1 embeddings performs at an intermediate level.

本节中我们引入了harmonic嵌套的概念，这是指由不同模型v1和v2产生的一系列嵌套，它们可以相互对比，所以是互相兼容的。这种兼容性极大的简化了升级路径，比如，在一个场景中，在一个很大的图像数据集上计算嵌套v1，而现在提出了一个新的嵌套模型v2，这种兼容性可以确保模型转换的平稳性，不需要担心版本不匹配问题。图8所示的是在3G数据集上的结果，可以看出改进的模型NN2明显比NN1要优秀，而NN2嵌套入NN1则处在中间水平。

Figure 8. Harmonic Embedding Compatibility. These ROCs show the compatibility of the harmonic embeddings of NN2 to the embeddings of NN1. NN2 is an improved model that performs much better than NN1. When comparing embeddings generated by NN1 to the harmonic ones generated by NN2 we can see the compatibility between the two. In fact, the mixed mode performance is still better than NN1 by itself.

### 7.1. Harmonic Triplet Loss

In order to learn the harmonic embedding we mix embeddings of v1 together with the embeddings v2, that are being learned. This is done inside the triplet loss and results in additionally generated triplets that encourage the compatibility between the different embedding versions. Figure 9 visualizes the different combinations of triplets that contribute to the triplet loss.

为了学习harmonic嵌套，我们将嵌套v1与嵌套v2混合在一起。这是在triplet loss内部进行完成的，结果会产生另外的triplets，在不同的嵌套版本中其兼容性会比较好。图9展示了对triplet loss有贡献的triplets的不同组合。

We initialized the v2 embedding from an independently trained NN2 and retrained the last layer (embedding layer) from random initialization with the compatibility encouraging triplet loss. First only the last layer is retrained, then we continue training the whole v2 network with the harmonic loss.

我们从一个独立训练出来的NN2初始化v2嵌套，然后从随机初始化重新训练最后一层（嵌套层），这还是与鼓励triplet loss兼容的。首先只有最后一层重新训练了，然后我们用harmonic loss继续训练所有的v2网络。

Figure 10 shows a possible interpretation of how this compatibility may work in practice. The vast majority of v2 embeddings may be embedded near the corresponding v1 embedding, however, incorrectly placed v1 embeddings can be perturbed slightly such that their new location in embedding space improves verification accuracy.

图10展示了这种兼容性在实践中如何起作用的一种可能解释。v2嵌套的大部分可能嵌入对应的v1嵌套，但放置错误的v1嵌套可能会略微变动，其在嵌套空间的新位置可能会改善验证准确率。

### 7.2. Summary

These are very interesting findings and it is somewhat surprising that it works so well. Future work can explore how far this idea can be extended. Presumably there is a limit as to how much the v2 embedding can improve over v1, while still being compatible. Additionally it would be interesting to train small networks that can run on a mobile phone and are compatible to a larger server side model.

这些发现非常有趣，某种程度上令人惊讶。未来的工作可以去探索这种观点可以走多远。v2嵌套改进v1嵌套是兼容的，但应当有一个极限。另外，训练可以在手机上运行的小型网络，同时又与服务器端的模型相兼容，这是非常有趣的。

## Acknowledgments

We would like to thank Johannes Steffens for his discussions and great insights on face recognition and Christian Szegedy for providing new network architectures like [16] and discussing network design choices. Also we are indebted to the DistBelief [4] team for their support especially to Rajat Monga for help in setting up efficient training schemes.

Also our work would not have been possible without the support of Chuck Rosenberg, Hartwig Adam, and Simon Han.

## References

- [1] Y. Bengio, J. Louradour, R. Collobert, and J. Weston. Curriculum learning. In Proc. of ICML, New York, NY, USA,2009. 2
- [2] D. Chen, X. Cao, L. Wang, F. Wen, and J. Sun. Bayesian face revisited: A joint formulation. In Proc. ECCV, 2012. 2
- [3] D. Chen, S. Ren, Y. Wei, X. Cao, and J. Sun. Joint cascade face detection and alignment. In Proc. ECCV, 2014. 7
- [4] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, Q. V. Le, and A. Y. Ng. Large scale distributed deep networks. In P. Bartlett, F. Pereira, C. Burges, L. Bottou, and K. Weinberger, editors, NIPS, pages 1232–1240. 2012. 10
- [5] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. J. Mach. Learn. Res., 12:2121–2159, July 2011. 4
- [6] I. J. Goodfellow, D. Warde-farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In In ICML, 2013. 4
- [7] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in  unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007. 5
- [8] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541–551, Dec. 1989. 2, 4
- [9] M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013. 2, 4, 6
- [10] C. Lu and X. Tang. Surpassing human-level face verification performance on LFW with gaussianface. CoRR, abs/1404.3840, 2014. 1
- [11] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 1986. 2, 4
- [12] M. Schultz and T. Joachims. Learning a distance metric from relative comparisons. In S. Thrun, L. Saul, and B. Schölkopf, editors, NIPS, pages 41–48. MIT Press, 2004. 2
- [13] T. Sim, S. Baker, and M. Bsat. The CMU pose, illumination, and expression (PIE) database. In In Proc. FG, 2002. 2
- [14] Y. Sun, X. Wang, and X. Tang. Deep learning face representation by joint identification-verification. CoRR, abs/1406.4773, 2014. 1, 2, 3
- [15] Y. Sun, X. Wang, and X. Tang. Deeply learned face representations are sparse, selective, and robust. CoRR, abs/1412.1265, 2014. 1, 2, 5, 8
- [16] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842, 2014. 2, 3, 4, 5, 6, 10
- [17] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. Deepface: Closing the gap to human-level performance in face verification. In IEEE Conf. on CVPR, 2014. 1, 2, 5, 7, 8, 9
- [18] J. Wang, Y. Song, T. Leung, C. Rosenberg, J. Wang, J. Philbin, B. Chen, and Y. Wu. Learning fine-grained image similarity with deep ranking. CoRR, abs/1404.4661, 2014. 2
- [19] K. Q. Weinberger, J. Blitzer, and L. K. Saul. Distance metric learning for large margin nearest neighbor classification. In NIPS. MIT Press, 2006. 2, 3
- [20] D. R. Wilson and T. R. Martinez. The general inefficiency of batch training for gradient descent learning. Neural Networks, 16(10):1429–1451, 2003. 4
- [21] L. Wolf, T. Hassner, and I. Maoz. Face recognition in unconstrained videos with matched background similarity. In IEEE Conf. on CVPR, 2011. 5
- [22] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. CoRR, abs/1311.2901, 2013. 2, 3, 4, 6
- [23] Z. Zhu, P. Luo, X. Wang, and X. Tang. Recover canonicalview faces in the wild with deep neural networks. CoRR, abs/1404.3543, 2014. 2