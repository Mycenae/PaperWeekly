# R-FCN: Object Detection via Region-based Fully Convolutional Networks

Jifeng Dai et al. Microsoft Research

## Abstract 摘要

We present region-based, fully convolutional networks for accurate and efficient object detection. In contrast to previous region-based detectors such as Fast/Faster R-CNN [6, 18] that apply a costly per-region subnetwork hundreds of times, our region-based detector is fully convolutional with almost all computation shared on the entire image. To achieve this goal, we propose position-sensitive score maps to address a dilemma between translation-invariance in image classification and translation-variance in object detection. Our method can thus naturally adopt fully convolutional image classifier backbones, such as the latest Residual Networks (ResNets) [9], for object detection. We show competitive results on the PASCAL VOC datasets (e.g., 83.6% mAP on the 2007 set) with the 101-layer ResNet. Meanwhile, our result is achieved at a test-time speed of 170ms per image, 2.5-20× faster than the Faster R-CNN counterpart. Code is made publicly available at: https://github.com/daijifeng001/r-fcn.

我们提出了一种基于区域的全卷积网络，进行准确高效的目标检测。之前基于区域的检测器，如Fast/Faster R-CNN[6,18]，它们在每个区域中都使用一个耗时的子网络，如此数百次，与其相比，我们的基于区域的检测器是全卷积的，所有计算都在整个图像中共享。为取得这个目标，我们提出一种对位置敏感的分数图，来解决图像分类中的平移不变性和目标检测中的平移变化的两难问题。我们的方法可以很自然的采用全卷积图像分类器骨干，比如最新的ResNets[9]，来进行目标检测。我们使用101层的ResNet在PASCAL VOC数据集上给出了很好的结果（如，在2007数据集上得到了83.6% mAP）。同时，我们的结果测试时达到了170ms每图像的速度，比Faster R-CNN的相应模型快了2.5-20x。代码已开源。

## 1 Introduction 引言

A prevalent family [8, 6, 18] of deep networks for object detection can be divided into two subnetworks by the Region-of-Interest (RoI) pooling layer [6]: (i) a shared, “fully convolutional” subnetwork independent of RoIs, and (ii) an RoI-wise subnetwork that does not share computation. This decomposition [8] was historically resulted from the pioneering classification architectures, such as AlexNet [10] and VGG Nets [23], that consist of two subnetworks by design — a convolutional subnetwork ending with a spatial pooling layer, followed by several fully-connected (fc) layers. Thus the (last) spatial pooling layer in image classification networks is naturally turned into the RoI pooling layer in object detection networks [8, 6, 18].

一族流行的DNN目标检测算法，可以由RoI池化层分成两个子网络：(i)共享的全卷积子网络，与RoIs无关；(ii)分RoI的子网络，计算不共享。这种分解是历史上分类架构的先驱架构的结果，如AlexNet和VGGNets，也在设计上包含两个子网络，卷积子网络以空间池化层结束，然后是几个全连接层。所以图像分类网络中，最后的空间池化层很自然的转化成目标检测网络中的RoI池化层。

But recent state-of-the-art image classification networks such as Residual Nets (ResNets) [9] and GoogLeNets [24, 26] are by design fully convolutional(Only the last layer is fully-connected, which is removed and replaced when fine-tuning for object detection.). By analogy, it appears natural to use all convolutional layers to construct the shared, convolutional subnetwork in the object detection architecture, leaving the RoI-wise subnetwork no hidden layer. However, as empirically investigated in this work, this naïve solution turns out to have considerably inferior detection accuracy that does not match the network’s superior classification accuracy. To remedy this issue, in the ResNet paper [9] the RoI pooling layer of the Faster R-CNN detector [18] is unnaturally inserted between two sets of convolutional layers — this creates a deeper RoI-wise subnetwork that improves accuracy, at the cost of lower speed due to the unshared per-RoI computation.

但是最近目前最好的图像分类网络，如ResNets和GoogLenets在设计上就是全卷积的（只有最后一层是全连接的，这在进行目标检测的时候就去除掉了并进行精调）。类比起来，使用全卷积网络来构建目标检测架构的共享的卷积子网络是很自然的，不给分RoI的子网络隐藏层。但是，根据本文中的经验研究，这种简单的解决方法会导致较差的检测准确率，与网络很高的分类准确率不符。为修复这个问题，在ResNet文章中，Faster R-CNN检测器中的RoI池化层，很不自然的插入到了两个卷积层集合之间，这产生了一个更深的分RoI的子网络，改进了准确率，其代价是更低的检测速度，因为每个RoI的计算都是不共享的。

We argue that the aforementioned unnatural design is caused by a dilemma of increasing translation invariance for image classification vs. respecting translation variance for object detection. On one hand, the image-level classification task favors translation invariance — shift of an object inside an image should be indiscriminative. Thus, deep (fully) convolutional architectures that are as translation-invariant as possible are preferable as evidenced by the leading results on ImageNet classification [9, 24, 26]. On the other hand, the object detection task needs localization representations that are translation-variant to an extent. For example, translation of an object inside a candidate box should produce meaningful responses for describing how good the candidate box overlaps the object. We hypothesize that deeper convolutional layers in an image classification network are less sensitive to translation. To address this dilemma, the ResNet paper’s detection pipeline [9] inserts the RoI pooling layer into convolutions — this region-specific operation breaks down translation invariance, and the post-RoI convolutional layers are no longer translation-invariant when evaluated across different regions. However, this design sacrifices training and testing efficiency since it introduces a considerable number of region-wise layers (Table 1).

我们认为，前面提到的不自然的设计是由于下面的两难问题导致的，即图像分类网络逐渐增加的平移不变性，和目标检测的平移可变性。一方面，图像级的分类任务倾向于平移不变性，在图像内移动一个目标，不应当影响分类结果。所以，深度（全）卷积架构要尽可能的平移不变，这样在图像分类问题中效果才会好，这在ImageNet分类上目前领先的模型中可以得到求证。另一方面，目标检测任务需要定位表示，这在一定程度上对平移是变化的。比如，目标在候选框中的平移，应当产生有意义的响应，描述候选框与目标的重叠程度。我们假设图像分类网络中，更深的卷积层对对平移没那么敏感。为解决这个两难困境，ResNet论文的检测过程在卷积中插入了RoI池化层，这个区域特定的操作破坏了平移不变性，RoI后的卷积层在对不同的区域进行评估时，不再是平移不变的。但是，这种设计牺牲了训练和测试效率，因为引入了相当多的分区域的层（表1）。

In this paper, we develop a framework called Region-based Fully Convolutional Network (R-FCN) for object detection. Our network consists of shared, fully convolutional architectures as is the case of FCN [15]. To incorporate translation variance into FCN, we construct a set of position-sensitive score maps by using a bank of specialized convolutional layers as the FCN output. Each of these score maps encodes the position information with respect to a relative spatial position (e.g., “to the left of an object”). On top of this FCN, we append a position-sensitive RoI pooling layer that shepherds information from these score maps, with no weight (convolutional/fc) layers following. The entire architecture is learned end-to-end. All learnable layers are convolutional and shared on the entire image, yet encode spatial information required for object detection. Figure 1 illustrates the key idea and Table 1 compares the methodologies among region-based detectors.

本文中，我们提出了一个目标检测的框架，称为基于区域的全卷积网络(R-FCN)。我们的网络包含的是共享的、全卷积架构，和FCN[15]的情况一样。为将平移变化的性质加入到FCN中，我们使用了一族特殊的卷积层作为FCN的输出，构建了为位置敏感的分数图的集合。每个分数图编码了相对空间位置的位置信息（如，在目标的左边）。在这些FCN之上，我们加上了对位置敏感的RoI池化层，利用了这些分数图的信息，后来就没有带有权重的层（卷积层、全连接层）了。整个架构是端到端学习的。所有可学习的层都是卷积层，在整个图像中是共享的，但是编码了目标检测所需的位置信息。图1描述了其中关键的思想，表1比较了基于区域的检测器的方法。

Using the 101-layer Residual Net (ResNet-101) [9] as the backbone, our R-FCN yields competitive results of 83.6% mAP on the PASCAL VOC 2007 set and 82.0% the 2012 set. Meanwhile, our results are achieved at a test-time speed of 170ms per image using ResNet-101, which is 2.5× to 20× faster than the Faster R-CNN + ResNet-101 counterpart in [9]. These experiments demonstrate that our method manages to address the dilemma between invariance/variance on translation, and fully convolutional image-level classifiers such as ResNets can be effectively converted to fully convolutional object detectors. Code is made publicly available at: https://github.com/daijifeng001/r-fcn.

使用ResNet-101作为骨干网络，我们的R-FCN在PASCAL VOC 2007集上得到了83.6%的很好结果，在2012集上得到了82.0%的结果。同时，我们使用ResNet-101的结果测试时的速度为每图像170ms，比相应的Faster R-CNN+ResNet-101[9]快了2.5x到20x。这些试验说明了，我们的方法可以处理平移不变性/平移变化性之间的两难问题，全卷积图像级的分类器，如ResNets，可以有效的转化成全卷积目标检测器。代码已开源。

Figure 1: Key idea of R-FCN for object detection. In this illustration, there are k × k = 3 × 3 position-sensitive score maps generated by a fully convolutional network. For each of the k × k bins in an RoI, pooling is only performed on one of the k 2 maps (marked by different colors).

Table 1: Methodologies of region-based detectors using ResNet-101 [9].

| | R-CNN [7] | Faster R-CNN [19, 9] | R-FCN [ours]
--- | --- | --- | ---
depth of shared convolutional subnetwork | 0 | 91 | 101
depth of RoI-wise subnetwork | 101 | 10 | 0

## 2 Our approach 我们的方法

**Overview**. Following R-CNN [7], we adopt the popular two-stage object detection strategy [7, 8, 6, 18, 1, 22] that consists of: (i) region proposal, and (ii) region classification. Although methods that do not rely on region proposal do exist (e.g., [17, 14]), region-based systems still possess leading accuracy on several benchmarks [5, 13, 20]. We extract candidate regions by the Region Proposal Network (RPN) [18], which is a fully convolutional architecture in itself. Following [18], we share the features between RPN and R-FCN. Figure 2 shows an overview of the system.

**概览**。我们按照R-CNN[7]的思路，采用流行的两阶段目标检测策略，包括：(i)区域建议，(ii)区域分类。虽然也有不需要进行区域建议的方法（如[17,14]），基于区域的系统仍然在几个基准测试中占据最高的准确率。我们使用RPN来提取区域建议[18]，这是一个全卷积架构。按照[18]的思路，我们在RPN与R-FCN中共享特征。图2是系统的概览。

Figure 2: Overall architecture of R-FCN. A Region Proposal Network (RPN) [18] proposes candidate RoIs, which are then applied on the score maps. All learnable weight layers are convolutional and are computed on the entire image; the per-RoI computational cost is negligible.

Given the proposal regions (RoIs), the R-FCN architecture is designed to classify the RoIs into object categories and background. In R-FCN, all learnable weight layers are convolutional and are computed on the entire image. The last convolutional layer produces a bank of $k^2$ position-sensitive score maps for each category, and thus has a $k^2 (C + 1)$-channel output layer with C object categories (+1 for background). The bank of $k^2$ score maps correspond to a k × k spatial grid describing relative positions. For example, with k × k = 3 × 3, the 9 score maps encode the cases of {top-left, top-center, top-right, ..., bottom-right} of an object category.

给定建议区域(RoIs)，R-FCN架构的设计是将RoIs分类成目标类别和背景。在R-FCN中，所有可学习的权重层都是卷积的，都是在整个图像上计算的。最后的卷积层为每个类别生成一族$k^2$个对位置敏感的分数图，所以有$k^2 (C+1)$个通道输出层（C个目标类别，1个背景类别）。这$k^2$个分数图对应着k×k个空间网格，描述的其相对位置。比如，在k × k = 3 × 3的情况下，这9个分数图编码了一个目标类别的{左上，中上，右上，...，右下}的情况。

R-FCN ends with a position-sensitive RoI pooling layer. This layer aggregates the outputs of the last convolutional layer and generates scores for each RoI. Unlike [8, 6], our position-sensitive RoI layer conducts selective pooling, and each of the k × k bin aggregates responses from only one score map out of the bank of k × k score maps. With end-to-end training, this RoI layer shepherds the last convolutional layer to learn specialized position-sensitive score maps. Figure 1 illustrates this idea. Figure 3 and 4 visualize an example. The details are introduced as follows.

R-FCN的结束是一个对位置敏感的RoI池化层。这个层聚积了最后卷积层的输出，并为每个RoI生成分数。与[8,6]不同的是，我们对位置敏感的RoI层进行的是选择性池化，每个k×k的格子聚积的是k×k分数图中的一个分数图的响应。在端到端的训练下，这个RoI层带着最后的卷积层学习专用的对位置敏感的分数图。图1描述的这种思想。图3和图4是一个可视化的例子。细节如下所述。

**Backbone architecture**. The incarnation of R-FCN in this paper is based on ResNet-101 [9], though other networks [10, 23] are applicable. ResNet-101 has 100 convolutional layers followed by global average pooling and a 1000-class fc layer. We remove the average pooling layer and the fc layer and only use the convolutional layers to compute feature maps. We use the ResNet-101 released by the authors of [9], pre-trained on ImageNet [20]. The last convolutional block in ResNet-101 is 2048-d, and we attach a randomly initialized 1024-d 1×1 convolutional layer for reducing dimension (to be precise, this increases the depth in Table 1 by 1). Then we apply the $k^2 (C + 1)$-channel convolutional layer to generate score maps, as introduced next.

**骨干架构**。本文中的R-FCN是基于ResNet-101的，其他的网络[10,23]也是可用的。ResNet-101有100个卷积层和1个全局平均池化层，最后是一个1000类的全连接层。我们去掉平均池化层和fc层，只使用卷积层来计算特征图。我们使用[9]中的ResNet-101，在ImageNet上进行预训练。ResNet-101中最后的卷积模块是2048维的，我们续接了一个随机初始化的1024-d 1×1卷积层，以进行降维（确切的说，这将表1中的深度增加了1）。然后我们使用$k^2 (C + 1)$通道的卷积层，以生成分数图，介绍如下。

**Position-sensitive score maps & Position-sensitive RoI pooling**. To explicitly encode position information into each RoI, we divide each RoI rectangle into k × k bins by a regular grid. For an RoI rectangle of a size w × h, a bin is of a size ≈ $w/k × h/k$ [8, 6]. In our method, the last convolutional layer is constructed to produce $k^2$ score maps for each category. Inside the (i, j)-th bin (0 ≤ i, j ≤ k − 1), we define a position-sensitive RoI pooling operation that pools only over the (i, j)-th score map:

**对位置敏感的分数图和对位置敏感的RoI池化**。为显式的在每个RoI中编码进位置信息，我们将每个RoI矩形用规则的网格分割成k×k bin。对于一个大小为w×h大小的RoI矩形，一个bin大小为约$w/k × h/k$。在我们的方法中，最后一个卷积层可以对每个类别生成$k^2$个分数图。在第(i,j)个bin中(0 ≤ i, j ≤ k − 1)，我们定义一个位置敏感的RoI池化运算，运算对象就是第(i,j)个分数图：

$$r_c(i,j | Θ) = \sum_{(x,y)∈bin(i,j)} z_{i,j,c} (x+x_0, y+y_0 | Θ)/n$$(1)

Here $r_c (i, j)$ is the pooled response in the (i, j)-th bin for the c-th category, $z_{i,j,c}$ is one score map out of the $k^2 (C + 1)$ score maps, ($x_0, y_0$) denotes the top-left corner of an RoI, n is the number of pixels in the bin, and Θ denotes all learnable parameters of the network. The (i, j)-th bin spans $\lfloor i \frac{w}{k} \rfloor ≤ x < \lceil (i+1) \frac {w}{k} \rceil$ and $\lfloor j \frac{h}{k} \rfloor ≤ y < \lceil (j+1) \frac {h}{k} \rceil$. The operation of Eqn.(1) is illustrated in Figure 1, where a color represents a pair of (i, j). Eqn.(1) performs average pooling (as we use throughout this paper), but max pooling can be conducted as well.

这里$r_c (i, j)$是在第(i,j)个bin中对第c类的池化特征，$z_{i,j,c}$是$k^2 (C + 1)$个特征图的其中的一个，($x_0, y_0$)表示RoI的左上角坐标，n是这个bin中的像素数量，Θ表示网络可以学习的参数。第(i,j)个bin的范围为$\lfloor i \frac{w}{k} \rfloor ≤ x < \lceil (i+1) \frac {w}{k} \rceil$ and $\lfloor j \frac{h}{k} \rfloor ≤ y < \lceil (j+1) \frac {h}{k} \rceil$。式(1)的运算如图1所示，其中一种色彩表示一对(i,j)。式(1)进行的是平均池化（我们在整篇文章中都使用），但使用最大池化也是可以的。

The $k^2$ position-sensitive scores then vote on the RoI. In this paper we simply vote by averaging the scores, producing a (C + 1)-dimensional vector for each RoI: $r_c (Θ) = \sum_{i,j} r_c (i,j | Θ)$. Then we compute the softmax responses across categories: $s_c (Θ) = e^{r_c (Θ)} / \sum_{c'=0}^C e^{r_{c'} (Θ)}$. They are used for evaluating the cross-entropy loss during training and for ranking the RoIs during inference.

这$k^2$个位置敏感的分数对RoI进行投票。本文中，我们使用简单的分数平均，对每个RoI生成一个(C+1)维的向量：$r_c (Θ) = \sum_{i,j} r_c (i,j | Θ)$。然后计算每个类别的softmax响应：$s_c (Θ) = e^{r_c (Θ)} / \sum_{c'=0}^C e^{r_{c'} (Θ)}$。这用于在训练时评估交叉熵损失，以及在推理时对RoIs进行排序。

We further address bounding box regression [7, 6] in a similar way. Aside from the above $k^2 (C +1)$-d convolutional layer, we append a sibling $4k^2$-d convolutional layer for bounding box regression. The position-sensitive RoI pooling is performed on this bank of $4k^2$ maps, producing a $4k^2$-d vector for each RoI. Then it is aggregated into a 4-d vector by average voting. This 4-d vector parameterizes a bounding box as $t = (t_x, t_y, t_w, t_h)$ following the parameterization in [6]. We note that we perform class-agnostic bounding box regression for simplicity, but the class-specific counterpart (i.e., with a $4k^2 C$-d output layer) is applicable.

我们进一步以类似的方式解决边界框回归的问题。除了上面的$k^2 (C +1)$维卷积层，我们还接入了一个并行的$4k^2$维的卷积层，以进行边界框回归。对位置敏感的RoI池化是在这$4k^2$的图上进行的，对每个RoI生成了一个$4k^2$维向量。然后这个向量通过平均投票集聚成一个4维向量。这个四维向量将边界框参数化为$t = (t_x, t_y, t_w, t_h)$，这采用的是[6]中的参数化方式。我们要说明的是，我们进行的是类别无关的边界框回归，这样更简单一些，但也可以使用类别相关的方式（即，输出维度为$4k^2 C$）。

The concept of position-sensitive score maps is partially inspired by [3] that develops FCNs for instance-level semantic segmentation. We further introduce the position-sensitive RoI pooling layer that shepherds learning of the score maps for object detection. There is no learnable layer after the RoI layer, enabling nearly cost-free region-wise computation and speeding up both training and inference.

对位置敏感的分数图的概念，是部分受到[3]的启发，他们提出了FCN进行实例级的语义分割。我们进一步提出了位置敏感的RoI池化层，进行目标识别的分数图学习。在RoI层之后就没有可以学习的层了，所以分层的计算几乎是没有什么计算量的，这加速了训练和推理过程。

**Training**. With pre-computed region proposals, it is easy to end-to-end train the R-FCN architecture. Following [6], our loss function defined on each RoI is the summation of the cross-entropy loss and the box regression loss: $L(s, t_{x,y,w,h}) = L_{cls} (s_{c^∗}) + λ[c^∗ > 0]L_{reg} (t, t^∗)$. Here $c^∗$ is the RoI’s ground-truth label ($c^∗$ = 0 means background). $L_{cls} (s_{c^∗}) = − log(s_{c^∗})$ is the cross-entropy loss for classification, $L_{reg}$ is the bounding box regression loss as defined in [6], and $t^∗$ represents the ground truth box. [$c^∗$ > 0] is an indicator which equals to 1 if the argument is true and 0 otherwise. We set the balance weight λ = 1 as in [6]. We define positive examples as the RoIs that have intersection-over-union (IoU) overlap with a ground-truth box of at least 0.5, and negative otherwise.

**训练**。在预先计算好的建议区域上，很容易进行R-FCN架构的端到端训练。采用[6]的思想，我们的损失函数在每个RoI上都定义为交叉熵损失和边界框回归损失的和：$L(s, t_{x,y,w,h}) = L_{cls} (s_{c^∗}) + λ[c^∗ > 0]L_{reg} (t, t^∗)$。这里$c^∗$是RoI的真值标签（$c^∗$ = 0意思是背景）。$L_{cls} (s_{c^∗}) = − log(s_{c^∗})$是分类的交叉熵损失，$L_{reg}$是边界框回归损失，和[6]中的定义一样，$t^∗$代表真值框。[$c^∗$ > 0]是指示器函数，如果参数为真则等于1，否则等于0。我们设置平衡权重λ = 1，这和[6]中一样。我们定义RoI的正样本为，与真值框的RoI重叠大于0.5，否则为负样本。

It is easy for our method to adopt online hard example mining (OHEM) [22] during training. Our negligible per-RoI computation enables nearly cost-free example mining. Assuming N proposals per image, in the forward pass, we evaluate the loss of all N proposals. Then we sort all RoIs (positive and negative) by loss and select B RoIs that have the highest loss. Backpropagation [11] is performed based on the selected examples. Because our per-RoI computation is negligible, the forward time is nearly not affected by N , in contrast to OHEM Fast R-CNN in [22] that may double training time. We provide comprehensive timing statistics in Table 3 in the next section.

我们的方法在训练过程中很容易使用在线难分样本挖掘(OHEM)。分RoI的计算量是可以忽略的，所以样本挖掘的计算量也很少。假设每幅图像有N个建议区域，在前向过程中，我们计算所有N个建议区域的损失。然后我们对所有RoIs（正样本和负样本）根据损失函数排序，选择B个最高损失的RoIs。反向传播[11]是基于选择的样本来进行的。因为我们的分RoI的计算是可以忽略的，前向时间几乎不受N的影响，而Fast R-CNN中的OHEM则可能会使训练时间加倍。我们在下一节中在表3给出了综合的时间统计结果。

We use a weight decay of 0.0005 and a momentum of 0.9. By default we use single-scale training: images are resized such that the scale (shorter side of image) is 600 pixels [6, 18]. Each GPU holds 1 image and selects B = 128 RoIs for backprop. We train the model with 8 GPUs (so the effective mini-batch size is 8×). We fine-tune R-FCN using a learning rate of 0.001 for 20k mini-batches and 0.0001 for 10k mini-batches on VOC. To have R-FCN share features with RPN (Figure 2), we adopt the 4-step alternating training in [18], alternating between training RPN and training R-FCN(Although joint training [18] is applicable, it is not straightforward to perform example mining jointly).

我们使用的权重衰减为0.0005，动量0.9。我们默认使用单尺度训练：图像要进行大小变换，这样其大小（图像短边）为600像素。每个GPU计算1幅图像，并选择B=128个RoIs进行反向传播。我们使用8 GPUs进行模型训练（这样有效的mini-batch size就是8倍的）。我们在VOC上精调R-FCN使用的学习速率为0.001，经过20K个mini-batch，然后用0.0001进行10K mini-batches的训练。为使R-FCN与RPN共享特征（图2），我们采用[18]中的4步轮流训练法，轮流训练RPN和R-FCN（虽然联合训练也是可用的，但就不能直接进行联合的样本挖掘了）。

**Inference**. As illustrated in Figure 2, the feature maps shared between RPN and R-FCN are computed (on an image with a single scale of 600). Then the RPN part proposes RoIs, on which the R-FCN part evaluates category-wise scores and regresses bounding boxes. During inference we evaluate 300 RoIs as in [18] for fair comparisons. The results are post-processed by non-maximum suppression (NMS) using a threshold of 0.3 IoU [7], as standard practice.

**推理**。如图2所示，计算RPN和R-FCN共享的特征图（图像尺度为600）。然后RPN部分提出建议区域RoIs，然后R-FCN计算分类别的分数并回归边界框。在推理过程中，我们计算300个RoIs，和[18]一样，这样可以进行公平比较。结果通过NMS进行后处理，使用阈值为0.3 IoU[7]，这都是标准操作。

**Àtrous and stride**. Our fully convolutional architecture enjoys the benefits of the network modifications that are widely used by FCNs for semantic segmentation [15, 2]. Particularly, we reduce ResNet-101’s effective stride from 32 pixels to 16 pixels, increasing the score map resolution. All layers before and on the conv4 stage [9] (stride=16) are unchanged; the stride=2 operations in the first conv5 block is modified to have stride=1, and all convolutional filters on the conv5 stage are modified by the “hole algorithm” [15, 2] (“Algorithme à trous” [16]) to compensate for the reduced stride. For fair comparisons, the RPN is computed on top of the conv4 stage (that are shared with R-FCN), as is the case in [9] with Faster R-CNN, so the RPN is not affected by the à trous trick. The following table shows the ablation results of R-FCN (k × k = 7 × 7, no hard example mining). The àtrous trick improves mAP by 2.6 points.

**孔洞和步长**。我们的全卷积架构可以享受到FCN网络修改的好处，这在语义分割中广泛使用。尤其是，我们将ResNet-101的有效步长从32降到了16，增加了分数图分辨率。conv4阶段及之前的所有层(stride=16)都是不变的；第一个conv5模块中步长=2的运算修改为步长=1，所有conv5阶段卷积滤波器都用孔洞算法进行修改，以步长降低的步长。为公平比较，RPN是在conv4阶段之上进行计算的（这部分是与R-FCN共享的），这与Faster R-CNN中的情况是一样的，所以RPN不受孔洞算法的影响。下表给出了R-FCN的分离试验结果(k × k = 7 × 7, no hard example mining)。孔洞的技巧改进了2.6点AP。

R-FCN with ResNet-101 on: | conv4, stride=16 | conv5, stride=32 | conv5, àtrous, stride=16
--- | --- | --- | ---
mAP (%) on VOC 07 test | 72.5 | 74.0 | 76.6

**Visualization**. In Figure 3 and 4 we visualize the position-sensitive score maps learned by R-FCN when k × k = 3 × 3. These specialized maps are expected to be strongly activated at a specific relative position of an object. For example, the “top-center-sensitive” score map exhibits high scores roughly near the top-center position of an object. If a candidate box precisely overlaps with a true object (Figure 3), most of the $k^2$ bins in the RoI are strongly activated, and their voting leads to a high score. On the contrary, if a candidate box does not correctly overlaps with a true object (Figure 4), some of the $k^2$ bins in the RoI are not activated, and the voting score is low.

**可视化**。在图3和图4中，我们对R-FCN学习的k × k = 3 × 3的对位置敏感的分数图进行了可视化。这些专用的图在目标的特殊相对位置应当有很强的激活值。比如，对中上位置敏感的分数图在目标的中上位置上给出很高的分数。如果候选框精确的与真实目标重叠（图3），RoI中$k^2$ bins的大多数激活值都会很大，其投票会带来很高的分数。相反，如果候选框与真实目标的重叠度没那么高（图4），RoI的$k^2$ bins中的一些的激活值就会很低，那么投票值就会很低。

Figure 3: Visualization of R-FCN (k × k = 3 × 3) for the person category.

Figure 4: Visualization when an RoI does not correctly overlap the object.

## 3 Related Work 相关工作

R-CNN [7] has demonstrated the effectiveness of using region proposals [27, 28] with deep networks. R-CNN evaluates convolutional networks on cropped and warped regions, and computation is not shared among regions (Table 1). SPPnet [8], Fast R-CNN [6], and Faster R-CNN [18] are “semi-convolutional”, in which a convolutional subnetwork performs shared computation on the entire image and another subnetwork evaluates individual regions.

R-CNN[7]证明了在深度网络中使用建议区域的有效性。R-CNN在剪切和变形的区域中计算卷积网络，这些计算在不同的区域中不是共享的（表1）。SPPnet, Fast R-CNN和Faster R-CNN是半卷积的，其中一个卷积子网络在整个图像中计算共享部分，另一个子网络计算每个区域的部分。

There have been object detectors that can be thought of as “fully convolutional” models. OverFeat [21] detects objects by sliding multi-scale windows on the shared convolutional feature maps; similarly, in Fast R-CNN [6] and [12], sliding windows that replace region proposals are investigated. In these cases, one can recast a sliding window of a single scale as a single convolutional layer. The RPN component in Faster R-CNN [18] is a fully convolutional detector that predicts bounding boxes with respect to reference boxes (anchors) of multiple sizes. The original RPN is class-agnostic in [18], but its class-specific counterpart is applicable (see also [14]) as we evaluate in the following.

有一些目标检测器可以认为是全卷积模型。OverFeat检测目标是通过在共享的卷积特征图中滑动多尺度窗口；类似的，在Fast R-CNN [6]和[12]中，也研究了替换建议区域的滑动窗口。在这些情况下，可以将单尺度下的滑动窗口重新cast为一个卷积层。Faster R-CNN[18]中的RPN是一个全卷积检测器，对多种大小的参考框（锚框）预测边界框。[18]中的原始RPN是类别无关的，但与类别相关的RPN也是可以采用的，我们在下面进行了试验。

Another family of object detectors resort to fully-connected (fc) layers for generating holistic object detection results on an entire image, such as [25, 4, 17]. 另一族目标检测器使用全连接层来生成在整个图像中进行全面目标检测。

## 4 Experiments 试验

### 4.1 Experiments on PASCAL VOC

We perform experiments on PASCAL VOC [5] that has 20 object categories. We train the models on the union set of VOC 2007 trainval and VOC 2012 trainval (“07+12”) following [6], and evaluate on VOC 2007 test set. Object detection accuracy is measured by mean Average Precision (mAP).

我们在20类的PASCAL VOC上进行试验。我们在VOC 2007和VOC 2012的trainval并集上进行训练，在VOC 2007测试集上进行评估。目标检测准确率用mAP衡量。

**Comparisons with Other Fully Convolutional Strategies**

Though fully convolutional detectors are available, experiments show that it is nontrivial for them to achieve good accuracy. We investigate the following fully convolutional strategies (or “almost” fully convolutional strategies that have only one classifier fc layer per RoI), using ResNet-101: 虽然已经有一些全卷积检测器，但试验表明，要取得好的准确率并不是很容易的。我们研究了下列全卷积策略（或几乎是全卷积的策略，每个RoI上只有一个分类器的fc层），使用的都是ResNet-101：

**Naïve Faster R-CNN**. As discussed in the introduction, one may use all convolutional layers in ResNet-101 to compute the shared feature maps, and adopt RoI pooling after the last convolutional layer (after conv5). An inexpensive 21-class fc layer is evaluated on each RoI (so this variant is “almost” fully convolutional). The à trous trick is used for fair comparisons.

**简单的Faster R-CNN**：就像在引言中介绍的，可以使用ResNet-101中所有的卷积层来计算共享的特征，在最后的卷积层(conv5)之后采用RoI池化。对每个RoI使用一个计算量不大的21类全连接层（这样其变体几乎是全连接的）。为公平比较，也使用了孔洞的技巧。

**Class-specific RPN**. This RPN is trained following [18], except that the 2-class (object or not) convolutional classifier layer is replaced with a 21-class convolutional classifier layer. For fair comparisons, for this class-specific RPN we use ResNet-101’s conv5 layers with the à trous trick.

**与类别有关的RPN**。这个RPN使用[18]的方法进行训练，除了两类的卷积分类层（目标/非目标）替换为21类卷积分类层。为公平比较，对于这个与类别有关的RPN，我们对ResNet-101的conv5层使用了孔洞技巧。

**R-FCN without position-sensitivity**. By setting k = 1 we remove the position-sensitivity of the R-FCN. This is equivalent to global pooling within each RoI. **没有位置敏感性的R-FCN**。设k=1，我们就去掉了R-FCN的位置敏感性。这等价于对每个RoI进行全局池化。

**Analysis**. Table 2 shows the results. We note that the standard (not naïve) Faster R-CNN in the ResNet paper [9] achieves 76.4% mAP with ResNet-101 (see also Table 3), which inserts the RoI pooling layer between conv4 and conv5 [9]. As a comparison, the naïve Faster R-CNN (that applies RoI pooling after conv5) has a drastically lower mAP of 68.9% (Table 2). This comparison empirically justifies the importance of respecting spatial information by inserting RoI pooling between layers for the Faster R-CNN system. Similar observations are reported in [19].

**分析**。表2给出了结果。我们要说明，标准（非简单）Faster R-CNN使用ResNet-101骨干得到了76.4% mAP（也见表3），其在conv4和conv5之间插入了RoI池化层。作为比较，简单的Faster R-CNN（在conv5之后进行RoI池化）的mAP则非常低68.9%（表2）。这个比较也从经验上说明了空间信息的重要性。类似的观察结果在[19]中也有。

The class-specific RPN has an mAP of 67.6% (Table 2), about 9 points lower than the standard Faster R-CNN’s 76.4%. This comparison is in line with the observations in [6, 12] — in fact, the class-specific RPN is similar to a special form of Fast R-CNN [6] that uses dense sliding windows as proposals, which shows inferior results as reported in [6, 12].

类别相关的RPN的mAP有67.6%（表2），比标准的Faster R-CNN 76.4%低了9个百分点。这种比较与[6,12]的观察是一致的，实际上，与类别相关的RPN类似于Fast R-CNN的特殊形式，使用密集滑窗作为建议窗口，如[6,12]中所述，结果较差。

On the other hand, our R-FCN system has significantly better accuracy (Table 2). Its mAP (76.6%) is on par with the standard Faster R-CNN’s (76.4%, Table 3). These results indicate that our position-sensitive strategy manages to encode useful spatial information for locating objects, without using any learnable layer after RoI pooling.

另一方面，我们的R-FCN系统则有非常高的准确率（表2）。其mAP为76.6%与标准Faster R-CNN类似（76.4%，表3）。这些结果说明，我们的对位置敏感的策略所使用的空间信息对目标定位非常有用，在RoI池化后就没有可学习的层了。

The importance of position-sensitivity is further demonstrated by setting k = 1, for which R-FCN is unable to converge. In this degraded case, no spatial information can be explicitly captured within an RoI. Moreover, we report that naïve Faster R-CNN is able to converge if its RoI pooling output resolution is 1 × 1, but the mAP further drops by a large margin to 61.7% (Table 2).

我们设k=1，就可以进一步看到对位置的敏感度的重要性，这样R-FCN就无法收敛。在这个降质的情况下，无法显式捕获到RoI中的空间信息。而且，简单Faster R-CNN在其RoI池化输出分辨率为1×1的情况下仍然可以收敛，但mAP会进一步下降很多，到61.7%（表2）。

Table 2: Comparisons among fully convolutional (or “almost” fully convolutional) strategies using ResNet-101. All competitors in this table use the à trous trick. Hard example mining is not conducted.

method | RoI output size(k×k) | mAP on VOC 07(%)
--- | --- | ---
naïve Faster R-CNN | 1×1 | 61.7
naïve Faster R-CNN | 7×7 | 68.9
class-specific RPN | - | 67.6
R-FCN(w/o position-sensitivity) | 1×1 | fail
R-FCN | 3×3 | 75.5
R-FCN | 7×7 | 76.6

**Comparisons with Faster R-CNN Using ResNet-101**

Next we compare with standard “Faster R-CNN + ResNet-101” [9] which is the strongest competitor and the top-performer on the PASCAL VOC, MS COCO, and ImageNet benchmarks. We use k × k = 7 × 7 in the following. Table 3 shows the comparisons. Faster R-CNN evaluates a 10-layer subnetwork for each region to achieve good accuracy, but R-FCN has negligible per-region cost. With 300 RoIs at test time, Faster R-CNN takes 0.42s per image, 2.5× slower than our R-FCN that takes 0.17s per image (on a K40 GPU; this number is 0.11s on a Titan X GPU). R-FCN also trains faster than Faster R-CNN. Moreover, hard example mining [22] adds no cost to R-FCN training (Table 3). It is feasible to train R-FCN when mining from 2000 RoIs, in which case Faster R-CNN is 6× slower (2.9s vs. 0.46s). But experiments show that mining from a larger set of candidates (e.g., 2000) has no benefit (Table 3). So we use 300 RoIs for both training and inference in other parts of this paper.

下一步，我们与标准的Faster R-CNN + ResNet-101进行比较，这是最强的竞争者，是PASCAL VOC, MSCOCO和ImageNet基准测试性能最佳的模型。我们下面使用k × k = 7 × 7的设置。表3给出了比较结果。Faster R-CNN对每个区域要计算一个10层的子网络，以得到好的准确率，但R-FCN的分区域计算量则可以忽略。在测试时，使用300 RoIs，Faster R-CNN每幅图像使用0.42s，我们的模型则是每幅图像0.17秒，快了2.5x（在K40 GPU上；在Titan X GPU上则是0.11s）。R-FCN的训练时间也少于Faster R-CNN。而且，难分样本挖掘[22]对于R-FCN的训练，几乎没有增加计算量（表3）。从2000 RoIs中进行挖掘然后训练也是可行的，而Faster R-CNN则要慢6x（2.9s vs 0.46s）。但试验表明，从更大的候选集（如2000）中进行挖掘没有多少好处（表3）。所以本文的其他部分，我们在训练和推理中都使用300 RoIs。

Table 3: Comparisons between Faster R-CNN and R-FCN using ResNet-101. Timing is evaluated on a single Nvidia K40 GPU. With OHEM, N RoIs per image are computed in the forward pass, and 128 samples are selected for backpropagation. 300 RoIs are used for testing following [18].

Table 4 shows more comparisons. Following the multi-scale training in [8], we resize the image in each training iteration such that the scale is randomly sampled from {400,500,600,700,800} pixels. We still test a single scale of 600 pixels, so add no test-time cost. The mAP is 80.5%. In addition, we train our model on the MS COCO [13] trainval set and then fine-tune it on the PASCAL VOC set. R-FCN achieves 83.6% mAP (Table 4), close to the “Faster R-CNN +++” system in [9] that uses ResNet-101 as well. We note that our competitive result is obtained at a test speed of 0.17 seconds per image, 20× faster than Faster R-CNN +++ that takes 3.36 seconds as it further incorporates iterative box regression, context, and multi-scale testing [9]. These comparisons are also observed on the PASCAL VOC 2012 test set (Table 5).

表4给出了更多比较结果。采用[8]中多尺度训练的思想，我们在每次训练迭代时都改变图像的大小，使其尺度是{400,500,600,700,800}随机选择的像素数。测试时图像仍然是一个尺度600像素，所以测试时不会增加代价。mAP是80.5%。另外，我们在MS COCO trainval集上训练模型，然后在PASCAL VOC上精调。R-FCN得到了83.6% mAP（表4），与[9]中的Faster R-CNN+++系统接近。我们要说明的是，我们这些结果的取得，运行速度是每图像0.17s，比Faster R-CNN+++系统快了20x，因为它进一步整合了迭代框回归、上下文和多尺度测试[9]，速度为3.36s每图像。在PASCAL VOC 2012测试集也可以看到类似的比较（表5）。

Table 4: Comparisons on PASCAL VOC 2007 test set using ResNet-101. “Faster R-CNN +++” [9] uses iterative box regression, context, and multi-scale testing.

Table 5: Comparisons on PASCAL VOC 2012 test set using ResNet-101. “07++12” [6] denotes the union set of 07 trainval+test and 12 trainval.

**On the Impact of Depth**

The following table shows the R-FCN results using ResNets of different depth [9]. Our detection accuracy increases when the depth is increased from 50 to 101, but gets saturated with a depth of 152. 下表展示了R-FCN使用不同深度的ResNets的结果。我们的检测准确率在深度从50增加到101时随之增加，但在152层时达到饱和。

**On the Impact of Region Proposals**

R-FCN can be easily applied with other region proposal methods, such as Selective Search (SS) [27] and Edge Boxes (EB) [28]. The following table shows the results (using ResNet-101) with different proposals. R-FCN performs competitively using SS or EB, showing the generality of our method. 

R-FCN可以很容易的使用其他区域建议方法，如Selective Search[27]和Edge Boxes[28]。下表给出使用不同候选方法的结果。R-FCN使用SS或EB也得到了不错的结果，说明我们方法的通用性不错。

### 4.2 Experiments on MS COCO

Next we evaluate on the MS COCO dataset [13] that has 80 object categories. Our experiments involve the 80k train set, 40k val set, and 20k test-dev set. We set the learning rate as 0.001 for 90k iterations and 0.0001 for next 30k iterations, with an effective mini-batch size of 8. We extend the alternating training [18] from 4-step to 5-step (i.e., stopping after one more RPN training step), which slightly improves accuracy on this dataset when the features are shared; we also report that 2-step training is sufficient to achieve comparably good accuracy but the features are not shared.

下面我们在MS COCO数据集上进行评估，有80类。我们的试验涉及到80k的训练集，40k的验证集和20k的测试开发集。我们设学习率为0.001，进行90k次迭代，然后再以0.0001进行下面30k次迭代，使用的有效mini-batch size为8。我们将轮流训练从4步拓展到5步（即，多一个RPN训练步骤），在特征共享时，这略微改进了准确率；我们还发现，2步训练足可以得到很好的准确率，但特征没有共享。

The results are in Table 6. Our single-scale trained R-FCN baseline has a val result of 48.9%/27.6%. This is comparable to the Faster R-CNN baseline (48.4%/27.2%), but ours is 2.5× faster testing. It is noteworthy that our method performs better on objects of small sizes (defined by [13]). Our multi-scale trained (yet single-scale tested) R-FCN has a result of 49.1%/27.8% on the val set and 51.5%/29.2% on the test-dev set. Considering COCO’s wide range of object scales, we further evaluate a multi-scale testing variant following [9], and use testing scales of {200,400,600,800,1000}. The mAP is 53.2%/31.5%. This result is close to the 1st-place result (Faster R-CNN +++ with ResNet-101, 55.7%/34.9%) in the MS COCO 2015 competition. Nevertheless, our method is simpler and adds no bells and whistles such as context or iterative box regression that were used by [9], and is faster for both training and testing.

结果如表6所示。我们单尺度训练的R-FCN基准验证集结果为48.9%/27.6%。这与Faster R-CNN结果类似(48.4%/27.2%)，但我们快了2.5x。值得注意的是，我们的方法在小目标上表现更好。我们的多尺度训练（单尺度测试）R-FCN结果为验证集49.1%/27.8%，测试集51.5%/29.2%。考虑到COCO中目标尺度非常广，我们进一步评估了多尺度测试的变体，采用[9]中的方法，使用的测试尺度包括{200,400,600,800,1000}。mAP结果为53.2%/31.5%。这个结果与MS COCO 2015比赛第一名的结果接近(Faster R-CNN +++ with ResNet-101, 55.7%/34.9%)。尽管如此，我们的方法更简单一些，没有增加任何花样，如上下文，或迭代框回归，这在[9]中都得到了使用，而且我们的方法在训练和测试中都更快。

Table 6: Comparisons on MS COCO dataset using ResNet-101. The COCO-style AP is evaluated @ IoU ∈ [0.5, 0.95]. AP@0.5 is the PASCAL-style AP evaluated @ IoU = 0.5.

## 5 Conclusion and Future Work

We presented Region-based Fully Convolutional Networks, a simple but accurate and efficient framework for object detection. Our system naturally adopts the state-of-the-art image classification backbones, such as ResNets, that are by design fully convolutional. Our method achieves accuracy competitive with the Faster R-CNN counterpart, but is much faster during both training and inference.

我们提出了基于区域的全卷积网络(R-FCN)，这是一种简单但准确高效的目标检测框架。我们的系统很自然的采用目前最好的图像分类骨干网络，如ResNets，从设计上就是全卷积的。我们的方法得到的准确率与相应的Faster R-CNN模型类似，但在训练和测试时都更简单。

We intentionally keep the R-FCN system presented in the paper simple. There have been a series of orthogonal extensions of FCNs that were developed for semantic segmentation (e.g., see [2]), as well as extensions of region-based methods for object detection (e.g., see [9, 1, 22]). We expect our system will easily enjoy the benefits of the progress in the field.

我们可以保持本文提出的R-FCN框架简单。已经有一系列FCN扩展可以进行语义分割，以及基于区域的方法进行目标检测。我们期待我们的系统可以很容易的享受这些好处。

Figure 5: Curated examples of R-FCN results on the PASCAL VOC 2007 test set (83.6% mAP). The network is ResNet-101, and the training data is 07+12+COCO. A score threshold of 0.6 is used for displaying. The running time per image is 170ms on one Nvidia K40 GPU.

Figure 6: Curated examples of R-FCN results on the MS COCO test-dev set (31.5% AP). The network is ResNet-101, and the training data is MS COCO trainval. A score threshold of 0.6 is used for displaying.

Table 7: Detailed detection results on the PASCAL VOC 2007 test set.

Table 8: Detailed detection results on the PASCAL VOC 2012 test set.