# Deformable Convolutional Networks

Jifeng Dai et. al. Microsoft Research Asia

## 0. Abstract

Convolutional neural networks (CNNs) are inherently limited to model geometric transformations due to the fixed geometric structures in their building modules. In this work, we introduce two new modules to enhance the transformation modeling capability of CNNs, namely, deformable convolution and deformable RoI pooling. Both are based on the idea of augmenting the spatial sampling locations in the modules with additional offsets and learning the offsets from the target tasks, without additional supervision. The new modules can readily replace their plain counterparts in existing CNNs and can be easily trained end-to-end by standard back-propagation, giving rise to deformable convolutional networks. Extensive experiments validate the performance of our approach. For the first time, we show that learning dense spatial transformation in deep CNNs is effective for sophisticated vision tasks such as object detection and semantic segmentation. The code is released at https://github.com/msracver/Deformable-ConvNets.

CNNs固有的局限于对几何形变进行建模，因为在其构建模块中有固定的几何结构。本文中，我们提出了新的模块，增强CNNs的变换建模能力，即，形变卷积和形变RoI池化。这两者都是基于，用额外的偏移来增加模块中空间采样位置的数量，从目标任务中学习这些偏移，而不需要额外的监督。新模块可以随时替代现有CNNs中的对应普通部分，可以很容易的用标准反向传播进行端到端的训练，得到形变CNNs。很多实验验证了我们方法的性能。第一次，我们证明了，在深度CNNs中学习密集空间变换对于复杂的视觉任务是有效的，比如目标检测和语义分割。代码已经开源。

## 1. Introduction

A key challenge in visual recognition is how to accommodate geometric variations or model geometric transformations in object scale, pose, viewpoint, and part deformation. In general, there are two ways. The first is to build the training datasets with sufficient desired variations. This is usually realized by augmenting the existing data samples, e.g., by affine transformation. Robust representations can be learned from the data, but usually at the cost of expensive training and complex model parameters. The second is to use transformation-invariant features and algorithms. This category subsumes many well known techniques, such as SIFT (scale invariant feature transform) [42] and sliding window based object detection paradigm.

在视觉识别中，一个关键的挑战是，怎样容纳目标尺度，姿态，视角和部分形变中的几何变换或模型几何变换。一般来说，有两种方法。第一是构建的训练数据集具有足够的期望的变换。这通常是将现有数据样本进行扩充实现的，比如，通过仿射变换。可以从数据学习到稳健的表示，但通常的代价是昂贵的训练和复杂的模型参数。第二种是使用对变换不变的特征和算法。这个类别中包含了很多著名的技术，比如SIFT和基于滑窗的目标检测方案。

There are two drawbacks in above ways. First, the geometric transformations are assumed fixed and known. Such prior knowledge is used to augment the data, and design the features and algorithms. This assumption prevents generalization to new tasks possessing unknown geometric transformations, which are not properly modeled. Second, handcrafted design of invariant features and algorithms could be difficult or infeasible for overly complex transformations, even when they are known.

上述方法中有两个缺陷。第一，几何变换假设是固定和已知的。用这样的先验知识来扩增数据，并设计特征和算法。这个假设阻碍了向包含未知几何变换的新任务的泛化，得不到合适的建模。第二，即使对于已知的情况，手工设计的不变特征和算法可能会很难，或对过于复杂的变换来说，是不可行的。

Recently, convolutional neural networks (CNNs) [35] have achieved significant success for visual recognition tasks, such as image classification [31], semantic segmentation [41], and object detection [16]. Nevertheless, they still share the above two drawbacks. Their capability of modeling geometric transformations mostly comes from the extensive data augmentation, the large model capacity, and some simple hand-crafted modules (e.g., max-pooling [1] for small translation-invariance).

最近，CNNs在视觉识别任务中获得了显著的成功，比如图像分类，语义分割和目标检测。尽管如此，仍然存在上述的缺点。几何形变建模的能力，主要来自于大量的数据扩增，大型模型的能力，和一些简单的手工设计模块（如，对小的平移不变性的最大池化）。

In short, CNNs are inherently limited to model large, unknown transformations. The limitation originates from the fixed geometric structures of CNN modules: a convolution unit samples the input feature map at fixed locations; a pooling layer reduces the spatial resolution at a fixed ratio; a RoI (region-of-interest) pooling layer separates a RoI into fixed spatial bins, etc. There lacks internal mechanisms to handle the geometric transformations. This causes noticeable problems. For one example, the receptive field sizes of all activation units in the same CNN layer are the same. This is undesirable for high level CNN layers that encode the semantics over spatial locations. Because different locations may correspond to objects with different scales or deformation, adaptive determination of scales or receptive field sizes is desirable for visual recognition with fine localization, e.g., semantic segmentation using fully convolutional networks [41]. For another example, while object detection has seen significant and rapid progress [16, 52, 15, 47, 46, 40, 7] recently, all approaches still rely on the primitive bounding box based feature extraction. This is clearly sub-optimal, especially for non-rigid objects.

简短来说，CNNs固有的局限在对大型未知变换的建模上。这个局限起源于CNNs模块的固定几何结构：卷积单元对输入特征图在固定的位置上进行采样；池化层以固定的比率降低空间分辨率；RoI池化层将RoI分离到固定的空间bins中，等。这缺少内部机制来处理几何变换。这导致了不可忽视的问题。比如，在同样的CNN层中的所有激活单元的感受野大小是一样的。对高层CNNs层，在空间位置中包含很多语义信息，这不是最理想的。因为不同的位置，会对应着不同尺度或变形的目标，如果可以自适应的确定尺度或感受野大小，对精细定位的视觉识别，是非常理想的，如使用FCN的语义分割。另外一个例子，虽然最近目标检测有了显著的快速进展，但所有的方法仍然依赖于原始的基于边界框的特征提取。这不是最优的，尤其是对于非刚性目标。

In this work, we introduce two new modules that greatly enhance CNNs’ capability of modeling geometric transformations. The first is deformable convolution. It adds 2D offsets to the regular grid sampling locations in the standard convolution. It enables free form deformation of the sampling grid. It is illustrated in Figure 1. The offsets are learned from the preceding feature maps, via additional convolutional layers. Thus, the deformation is conditioned on the input features in a local, dense, and adaptive manner.

本文中，我们提出了两个模块，极大的增强了CNNs对几何变换建模的能力。第一是形变卷积，对标准卷积中的规则网格采样位置，加入了2D偏移，这使得可以对采样网格进行自由形式的形变，如图1所示。偏移是从前面的特征图学习得到的，通过额外的卷积层。因此，在输入特征上的形变是局部、密集和自适应的。

The second is deformable RoI pooling. It adds an offset to each bin position in the regular bin partition of the previous RoI pooling [15, 7]. Similarly, the offsets are learned from the preceding feature maps and the RoIs, enabling adaptive part localization for objects with different shapes.

第二个是可形变的RoI池化，对之前的RoI池化的规则bin分割上，对每个bin位置增加了偏移。类似的，偏移是从之前的特征图和RoI上学习得到的，可以对有不同形状的目标进行自适应的部位定位。

Both modules are light weight. They add small amount of parameters and computation for the offset learning. They can readily replace their plain counterparts in deep CNNs and can be easily trained end-to-end with standard back-propagation. The resulting CNNs are called deformable convolutional networks, or deformable ConvNets.

这两个模块都是轻量的。偏移学习增加了一些参数量和计算量。它们可以随时在深度CNN中替换掉其他部分，可以用标准反向传播进行简单的端到端训练。得到的CNNs称为deformable-CNNs。

Our approach shares similar high level spirit with spatial transform networks [26] and deformable part models [11]. They all have internal transformation parameters and learn such parameters purely from data. A key difference in deformable ConvNets is that they deal with dense spatial transformations in a simple, efficient, deep and end-to-end manner. In Section 3.1, we discuss in details the relation of our work to previous works and analyze the superiority of deformable ConvNets.

我们的方法与spatial transform networks[26]和deformable part models [11]有一样的高层思想。它们都有内部的变换参数，从数据中学习这样的参数。我们的方法的一个关键差异是，处理密集空间变换的方式非常简单、高效、深度、端到端。在3.1节中，我们详细讨论了我们的工作与之前的工作之间的关系，分析了我们方法的优势。

## 2. Deformable Convolutional Networks

The feature maps and convolution in CNNs are 3D. Both deformable convolution and RoI pooling modules operate on the 2D spatial domain. The operation remains the same across the channel dimension. Without loss of generality, the modules are described in 2D here for notation clarity. Extension to 3D is straightforward.

CNNs中的特征图和卷积是3D的。形变卷积和RoI池化模块都是在2D空间域上运算的，这些运算在通道维度上是一样的。不失一般性，这些模块在这里以2D进行描述，以保持表示的简洁性。拓展到3D是很直观的。

### 2.1. Deformable Convolution

The 2D convolution consists of two steps: 1) sampling using a regular grid R over the input feature map x; 2) summation of sampled values weighted by w. The grid R defines the receptive field size and dilation. For example,

2D卷积包含两步：1)在输入特征图x上使用规则网格R进行采样；2)对采样的值用w进行加权然后相加。网格R定义了感受野大小和膨胀度。比如

$$R = \{ (−1, −1), (−1, 0), . . . , (0, 1), (1, 1) \}$$

defines a 3 × 3 kernel with dilation 1. 定义了膨胀度为1的3x3核。

For each location p_0 on the output feature map y, we have 对输出特征图y上的每个位置p_0，我们有

$$y(p_0) = \sum_{p_n \in R} w(p_n) ⋅ x(p_0 + p_n)$$(1)

where p_n enumerates the locations in R. 其中p_n枚举了R中的位置。

In deformable convolution, the regular grid R is augmented with offsets {$∆p_n$|n = 1, ..., N}, where N = |R|. Eq. (1) becomes 在形变卷积中，规则网格用偏移{$∆p_n$|n = 1, ..., N}进行扩充，其中N = |R|。式(1)变成了

$$y(p_0) = \sum_{p_n \in R} w(p_n) ⋅ x(p_0 + p_n + ∆p_n)$$(2)

Now, the sampling is on the irregular and offset locations $p_n+∆p_n$. As the offset $∆p_n$ is typically fractional, Eq. (2) is implemented via bilinear interpolation as 现在，采样是在不规则和偏移的位置$p_n+∆p_n$上进行的。因为偏移$∆p_n$一般是很小的，式(2)用双线性差值进行实现

$$x(p) = \sum_q G(q,p) ⋅ x(q)$$(3)

where p denotes an arbitrary (fractional) location ($p = p_0 + p_n + ∆p_n$ for Eq. (2)), q enumerates all integral spatial locations in the feature map x, and G(·, ·) is the bilinear interpolation kernel. Note that G is two dimensional. It is separated into two one dimensional kernels as

其中p表示任意的位置，q枚举了在特征图x中的所有整数位置，G(·, ·)是双线性差值核。注意G是二维的，可以分解成两个一维核，如

$$G(q,p) = g(q_x,p_x)⋅g(q_y,p_y)$$(4)

where g(a, b) = max(0, 1 − |a − b|). Eq. (3) is fast to compute as G(q, p) is non-zero only for a few qs.

As illustrated in Figure 2, the offsets are obtained by applying a convolutional layer over the same input feature map. The convolution kernel is of the same spatial resolution and dilation as those of the current convolutional layer (e.g., also 3 × 3 with dilation 1 in Figure 2). The output offset fields have the same spatial resolution with the input feature map. The channel dimension 2N corresponds to N 2D offsets. During training, both the convolutional kernels for generating the output features and the offsets are learned simultaneously. To learn the offsets, the gradients are back-propagated through the bilinear operations in Eq. (3) and Eq. (4). It is detailed in appendix A.

如图2所示，偏移是通过将一个卷积核应用到相同的输入特征图得到的。卷积核与目前卷积层中的右相同的空间分辨率和膨胀度（如，图2中的膨胀度为1的3x3的核）。输出的偏移场与输入特征图有相同的空间分辨率。通道维度2N对应着N个2D偏移。在训练中，生成输出特征和偏移的卷积核，是同时学习的。为学习偏移，梯度是通过式3和式4的双线性运算进行反向传播的。详见附录A。

### 2.2. Deformable RoI Pooling

RoI pooling is used in all region proposal based object detection methods [16, 15, 47, 7]. It converts an input rectangular region of arbitrary size into fixed size features.

RoI池化在所有基于区域推荐的目标检测方法中都有使用，将输入的任意大小的矩形区域转化成固定大小的特征。

**RoI Pooling [15]**. Given the input feature map x and a RoI of size w×h and top-left corner $p_0$, RoI pooling divides the RoI into k × k (k is a free parameter) bins and outputs a k × k feature map y. For (i, j)-th bin (0 ≤ i, j < k), we have

**RoI池化**。给定输入的特征图x，和一个大小为w×h的RoI，top-left角点为$p_0$，RoI池化将RoI分成k × k个bins，输出k × k的特征图y。对于第(i, j)个bin(0 ≤ i, j < k)，我们有

$$y(i,j) = \sum_{p \in bin(i,j)} x(p_0+p)/n_{ij}$$(5)

where $n_{ij}$ is the number of pixels in the bin. The (i,j)-th bin spans $\lfloor iw/k \rfloor ≤ p_x < \lceil (i+1)w/k \rceil$ and $\lfloor jh/k \rfloor ≤ p_y < \lceil (j+1)h/k \rceil$.

Similarly as in Eq. (2), in deformable RoI pooling, offsets {$∆p_{ij}$|0 ≤ i, j < k} are added to the spatial binning positions. Eq.(5) becomes

与式(2)类似，在形变RoI池化中，在空间binning位置上，加上了偏移{$∆p_{ij}$|0 ≤ i, j < k}。式5变成了

$$y(i,j) = \sum_{p \in bin(i,j)} x(p_0 + p + ∆p_{ij})/n_{ij}$$(6)

Typically, $∆p_{ij}$ is fractional. Eq. (6) is implemented by bilinear interpolation via Eq. (3) and (4). 一般来说，$∆p_{ij}$是很小的。式6通过式3和式4用双线性差值来实现。

Figure 3 illustrates how to obtain the offsets. Firstly, RoI pooling (Eq. (5)) generates the pooled feature maps. From the maps, a fc layer generates the normalized offsets $∆\hat p_{ij}$, which are then transformed to the offsets $∆p_{ij}$ in Eq. (6) by element-wise product with the RoI’s width and height, as $∆p_{ij} = γ · ∆\hat p_{ij} ◦ (w, h)$. Here γ is a pre-defined scalar to modulate the magnitude of the offsets. It is empirically set to γ = 0.1. The offset normalization is necessary to make the offset learning invariant to RoI size. The fc layer is learned by back-propagation, as detailed in appendix A.

图3展示了怎样得到偏移的。首先，RoI池化生成池化的特征图。从图中，fc层生成了归一化的偏移$∆\hat p_{ij}$，然后通过与RoI的宽度和高度进行逐元素的乘积，变换到式6中的偏移$∆p_{ij}$，如$∆p_{ij} = γ · ∆\hat p_{ij} ◦ (w, h)$。这里γ是一个预定义的标量，对偏移的幅度进行调制。经验上设置γ = 0.1。偏移的归一化是必须的，以使偏移学习对RoI的大小不变。fc层是用反向传播学习的，如附录A中详述。

**Position-Sensitive (PS) RoI Pooling [7]**. It is fully convolutional and different from RoI pooling. Through a conv layer, all the input feature maps are firstly converted to k^2 score maps for each object class (totally C + 1 for C object classes), as illustrated in the bottom branch in Figure 4. Without need to distinguish between classes, such score maps are denoted as {$x_{i,j}$} where (i, j) enumerates all bins. Pooling is performed on these score maps. The output value for (i, j)-th bin is obtained by summation from one score map $x_{i,j}$ corresponding to that bin. In short, the difference from RoI pooling in Eq.(5) is that a general feature map x is replaced by a specific positive-sensitive score map $x_{i,j}$.

**对位置敏感的RoI池化**。这是全卷积的，与RoI池化是不同的。通过一个卷积层，所有输入特征图首先对每个目标类别转换成k^2个分数图，如图4中底部分支所描述。不需要在类别间进行区分，这样的分数图表示为{$x_{i,j}$}，其中(i, j)枚举了所有的bins。池化在这些分数图上进行。对第(i, j)个bin，输出值是通过对应这个bin的分数图$x_{i,j}$相加得到的。简短来说，与式5中RoI池化的区别是，通用特征图x替换为专用的位置敏感的分数图$x_{i,j}$。

In deformable PS RoI pooling, the only change in Eq. (6) is that x is also modified to $x_{i,j}$. However, the offset learning is different. It follows the “fully convolutional” spirit in [7], as illustrated in Figure 4. In the top branch, a conv layer generates the full spatial resolution offset fields. For each RoI (also for each class), PS RoI pooling is applied on such fields to obtain normalized offsets $∆\hat p_{ij}$, which are then transformed to the real offsets $∆p_{ij}$ in the same way as in deformable RoI pooling described above.

在形变PS RoI池化中，式6中唯一的变化是，x也修改成了$x_{i,j}$。但是，偏移学习是不一样的。其遵循[7]中全卷积的思想，如图4所示。在上面分支中，卷积层生成了全空间分辨率偏移场。对每个RoI（也是对每个类别），PS RoI池化应用到这种场中，以得到归一化的偏移$∆\hat p_{ij}$，然后变换到真实的偏移$∆p_{ij}$，方式与上述的形变RoI池化一样。

### 2.3. Deformable ConvNets

Both deformable convolution and RoI pooling modules have the same input and output as their plain versions. Hence, they can readily replace their plain counterparts in existing CNNs. In the training, these added conv and fc layers for offset learning are initialized with zero weights. Their learning rates are set to β times (β = 1 by default, and β = 0.01 for the fc layer in Faster R-CNN) of the learning rate for the existing layers. They are trained via back propagation through the bilinear interpolation operations in Eq. (3) and Eq. (4). The resulting CNNs are called deformable ConvNets.

形变卷积和RoI池化模块与其普通版有着相同的输入和输出。因此，它们可以在现有的CNNs中随时替换掉其普通对应部分。在训练中，这些加入的用于偏移学习的conv和fc层，用零权重进行初始化。其学习速率设置为现有层的学习速率的β倍（在Faster R-CNN中，默认β=1，对fc层，β=0.01）。它们通过反向传播，用式3和式4的双线性差值运算进行训练。得到的CNNs称为形变CNNs。

To integrate deformable ConvNets with the state-of-the-art CNN architectures, we note that these architectures consist of two stages. First, a deep fully convolutional network generates feature maps over the whole input image. Second, a shallow task specific network generates results from the feature maps. We elaborate the two steps below.

为将形变CNNs与目前最好的CNN架构结合起来，我们要说明，这些架构时由两个阶段构成。第一，深度FCN在整幅输入图像中生成特征图。第二，浅层任务专用网络从特征图中生成结果。我们下面详述这两个步骤。

**Deformable Convolution for Feature Extraction**. We adopt two state-of-the-art architectures for feature extraction: ResNet-101 [22] and a modifed version of Inception-ResNet [51]. Both are pre-trained on ImageNet [8] classification dataset. 我们采用目前最好的两种架构进行特征提取：ResNet-101和修改版的Inception-ResNet。两者都是在ImageNet分类数据集中进行的预训练。

The original Inception-ResNet is designed for image recognition. It has a feature misalignment issue and problematic for dense prediction tasks. It is modified to fix the alignment problem [20]. The modified version is dubbed as “Aligned-Inception-ResNet” and is detailed in appendix B.

原始的Inception-ResNet是设计用于图像识别的，它有特征不对齐的问题，对于密集预测任务来说是有问题的。我们对其修正，以解决对齐问题。修正版称为Aligned-Inception-ResNet，在附录B中详述。

Both models consist of several convolutional blocks, an average pooling and a 1000-way fc layer for ImageNet classification. The average pooling and the fc layers are removed. A randomly initialized 1 × 1 convolution is added at last to reduce the channel dimension to 1024. As in common practice [4, 7], the effective stride in the last convolutional block is reduced from 32 pixels to 16 pixels to increase the feature map resolution. Specifically, at the beginning of the last block, stride is changed from 2 to 1 (“conv5” for both ResNet-101 and Aligned-Inception-ResNet). To compensate, the dilation of all the convolution filters in this block (with kernel size > 1) is changed from 1 to 2.

两个模型都包含几个卷积层，一个平均池化层，和一个1000路的fc层，以进行ImageNet分类。去掉了平均池化和fc层。最后加上了随机初始化的1x1卷积，以将通道维度降低到1024。常见的做法还有，最后一个卷积层的有效步长，从32像素降低到16像素，以增加特征图分辨率。具体的，在最后一个模块的开始，步长从2变为1（ResNet-101和Aligned-Inception-ResNet的conv5）。为补偿，这个模块中所有卷积滤波器（核大小>1）的膨胀从1改变为2。

Optionally, deformable convolution is applied to the last few convolutional layers (with kernel size > 1). We experimented with different numbers of such layers and found 3 as a good trade-off for different tasks, as reported in Table 1.

可选的是，形变卷积应用到最后几个卷积层（核大小>1）上。我们用不同数量的这种层进行试验，发现对于不同的任务来说，3是一个很好的折中选择，如表1所示。

**Segmentation and Detection Networks**. A task specific network is built upon the output feature maps from the feature extraction network mentioned above. 任务专用网络是在上述特征提取网络的输出特征图上构建起来的。

In the below, C denotes the number of object classes. 下面，C表示目标类别的数量。

DeepLab [5] is a state-of-the-art method for semantic segmentation. It adds a 1 × 1 convolutional layer over the feature maps to generates (C + 1) maps that represent the per-pixel classification scores. A following softmax layer then outputs the per-pixel probabilities. DeepLab是一种目前最好的语义分割方法。其在特征图上加入了1×1卷积层，以生成(C+1)个图，表示逐像素分类的分数。后随着softmax层输出逐像素概率。

Category-Aware RPN is almost the same as the region proposal network in [47], except that the 2-class (object or not) convolutional classifier is replaced by a (C + 1)-class convolutional classifier. It can be considered as a simplified version of SSD [40].

对类别敏感的RPN与[47]中的区域候选网络几乎相同，除了2类卷积分类器替换为了C+1类的卷积分类器，可以认为是SSD的简化版本。

Faster R-CNN [47] is the state-of-the-art detector. In our implementation, the RPN branch is added on the top of the conv4 block, following [47]. In the previous practice [22,24], the RoI pooling layer is inserted between the conv4 and the conv5 blocks in ResNet-101, leaving 10 layers for each RoI. This design achieves good accuracy but has high per-RoI computation. Instead, we adopt a simplified design as in [38]. The RoI pooling layer is added at last. On top of the pooled RoI features, two fc layers of dimension 1024 are added, followed by the bounding box regression and the classification branches. Although such simplification (from 10 layer conv5 block to 2 fc layers) would slightly decrease the accuracy, it still makes a strong enough baseline and is not a concern in this work.

Faster R-CNN是目前最好的检测器。在我们的实现中，RPN分支是在conv4模块之上添加的，与[47]一样。在之前的实践中，RoI池化层是在ResNet-101的conv4和conv5之间插入的，对每个RoI中留下10层。这个设计获得了很好的准确率，但每个RoI计算量较高。我们采用了与[38]一样的一个简化版，其中RoI池化是在最后加入的。在最后池化的RoI特征之上，加入了2个fc层，维度1024，然后是边界框回归和分类分支。虽然这样的简化（从10层conv5模块，到2个fc层）会略微降低准确率，但这仍然是一个不错的基准，本文中不进行考虑。

Optionally, the RoI pooling layer can be changed to deformable RoI pooling. RoI池化层可以变为形变RoI池化层。

R-FCN [7] is another state-of-the-art detector. It has negligible per-RoI computation cost. We follow the original implementation. Optionally, its RoI pooling layer can be changed to deformable position-sensitive RoI pooling. R-FCN是另一个目前最好的检测器，其每个RoI的计算代价还是可忽略的。我们按照原始实现进行。其RoI池化层可以改变为形变PS RoI池化层。

## 3. Understanding Deformable ConvNets

This work is built on the idea of augmenting the spatial sampling locations in convolution and RoI pooling with additional offsets and learning the offsets from target tasks. 本文是建立在下述思想之上的，对卷积和RoI池化中的空间采样位置用额外的偏移进行扩增，从目标任务中学习偏移。

When the deformable convolution are stacked, the effect of composited deformation is profound. This is exemplified in Figure 5. The receptive field and the sampling locations in the standard convolution are fixed all over the top feature map (left). They are adaptively adjusted according to the objects’ scale and shape in deformable convolution (right). More examples are shown in Figure 6. Table 2 provides quantitative evidence of such adaptive deformation.

当形变卷积堆叠时，复合形变的效果是巨大的，这如图5所示。标准卷积中的感受野和采样位置，在整个特征图中都是固定的。在形变卷积中，它们可以根据目标的尺度和形状进行自适应调整。图6中给出了更多的例子。表2给出了这种自适应形变的量化证据。

The effect of deformable RoI pooling is similar, as illustrated in Figure 7. The regularity of the grid structure in standard RoI pooling no longer holds. Instead, parts deviate from the RoI bins and move onto the nearby object foreground regions. The localization capability is enhanced, especially for non-rigid objects.

形变RoI池化的效果类似，如图7所示。标准RoI池化中的规则化网格结构没有了，部位从RoI bins中偏移，移到目标前景区域的附近。定位能力得到了增强，尤其是对于非刚性目标。

### 3.1. In Context of Related Works

Our work is related to previous works in different aspects. We discuss the relations and differences in details. 我们的工作与之前的工作在几个方面是不一样的。下面详细讨论关系和差异。

**Spatial Transform Networks (STN) [26]**. It is the first work to learn spatial transformation from data in a deep learning framework. It warps the feature map via a global parametric transformation such as affine transformation. Such warping is expensive and learning the transformation parameters is known difficult. STN has shown successes in small scale image classification problems. The inverse STN method [37] replaces the expensive feature warping by efficient transformation parameter propagation.

这是第一个从数据中学习空间变换的深度学习框架，它将特征图通过一个全局参数化变换进行变形，如仿射变换。这种变形的计算量是很大的，学习这种变换参数是很困难的。STN在小规模图像分类问题中比较成功。逆STN方法将昂贵的特征变形，替换为高效的变换参数传播。

The offset learning in deformable convolution can be considered as an extremely light-weight spatial transformer in STN [26]. However, deformable convolution does not adopt a global parametric transformation and feature warping. Instead, it samples the feature map in a local and dense manner. To generate new feature maps, it has a weighted summation step, which is absent in STN.

形变卷积中的偏移学习，可以认为是STN中一个极度轻量的空间变换器。但是，形变卷积并没有采用一个全局的参数变换和特征变形，而是对特征图以局部和密集的方式进行采样。为生成新的特征图，还有一个加权求和步骤，在STN中是没有的。

Deformable convolution is easy to integrate into any CNN architectures. Its training is easy. It is shown effective for complex vision tasks that require dense (e.g., semantic segmentation) or semi-dense (e.g., object detection) predictions. These tasks are difficult (if not infeasible) for STN [26, 37].

形变卷积是很容易整合进任何CNN架构的。其训练是很简单的，对需要密集（如，语义分割）或半密集（如，目标检测）预测的复杂视觉任务来说，是有效的。这些任务对于STN来说，是非常困难的。

**Active Convolution [27]**. This work is contemporary. It also augments the sampling locations in the convolution with offsets and learns the offsets via back-propagation end-to-end. It is shown effective on image classification tasks.

这个工作是同时的，它也用偏移扩增了卷积中的采样位置，通过反向传播端到端的学习了偏移，在图像分类任务中证明是有效的。

Two crucial differences from deformable convolution make this work less general and adaptive. First, it shares the offsets all over the different spatial locations. Second, the offsets are static model parameters that are learnt per task or per training. In contrast, the offsets in deformable convolution are dynamic model outputs that vary per image location. They model the dense spatial transformations in the images and are effective for (semi-)dense prediction tasks such as object detection and semantic segmentation.

与形变卷积有两个关键的差异，这使得这个工作没有那么通用和自适应。第一，在不同的空间位置上共享偏移。第二，偏移是静态的模型参数，在每个任务或每次训练中进行学习。比较之下，形变卷积中的偏移是动态模型输出，在每个图像位置上都不一样。它们对图像中的密集空间变换进行建模，对（半）密集的预测任务，如目标检测和语义分割，是有效的。

**Effective Receptive Field [43]**. It finds that not all pixels in a receptive field contribute equally to an output response. The pixels near the center have much larger impact. The effective receptive field only occupies a small fraction of the theoretical receptive field and has a Gaussian distribution. Although the theoretical receptive field size increases linearly with the number of convolutional layers, a surprising result is that, the effective receptive field size increases linearly with the square root of the number, therefore, at a much slower rate than what we would expect.

感受野中，并不是所有像素都会输出响应有相同的贡献。中心附近的像素的影响要大的多。有效感受野只占理论感受野的很小一部分，有高斯分布。虽然理论感受野大小随着卷积层数量的增加而线性增加，一个令人惊讶的结果是，有效感受野是随着数量的平方根线性增加的，因此，比我们预期的速度要慢的多。

This finding indicates that even the top layer’s unit in deep CNNs may not have large enough receptive field. This partially explains why atrous convolution [23] is widely used in vision tasks (see below). It indicates the needs of adaptive receptive field learning.

这个发现说明，即使是CNNs中的顶层，也不一定有足够大的感受野。这部分解释了，为什么atrous卷积广泛应用于视觉任务中。这说明，自适应感受野学习有很强的需求。

Deformable convolution is capable of learning receptive fields adaptively, as shown in Figure 5, 6 and Table 2. 形变卷积是能够自适应的学习感受野的，如图5，6和表2所示。

**Atrous convolution [23]**. It increases a normal filter’s stride to be larger than 1 and keeps the original weights at sparsified sampling locations. This increases the receptive field size and retains the same complexity in parameters and computation. It has been widely used for semantic segmentation [41, 5, 54] (also called dilated convolution in [54]), object detection [7], and image classification [55].

将正常滤波器的步长增加，超过1，在稀疏的采样位置保持原始权重。这增加了感受野大小，参数和计算量上保持了相同的复杂度。在语义分割，目标检测和图像分类中都有广泛的应用。

Deformable convolution is a generalization of atrous convolution, as easily seen in Figure 1 (c). Extensive comparison to atrous convolution is presented in Table 3. 形变卷积是atrous卷积的推广，如图1c所示。与atrous卷积的广泛比较如表3所示。

**Deformable Part Models (DPM) [11]**. Deformable RoI pooling is similar to DPM because both methods learn the spatial deformation of object parts to maximize the classification score. Deformable RoI pooling is simpler since no spatial relations between the parts are considered.

形变RoI卷积与DPM类似，因为两种方法都学习了目标部分的空间形变，以最大化分类分数。形变RoI池化更简单，因为部分之间的空间关系并没有考虑。

DPM is a shallow model and has limited capability of modeling deformation. While its inference algorithm can be converted to CNNs [17] by treating the distance transform as a special pooling operation, its training is not end-to-end and involves heuristic choices such as selection of components and part sizes. In contrast, deformable ConvNets are deep and perform end-to-end training. When multiple deformable modules are stacked, the capability of modeling deformation becomes stronger.

DPM是一个浅层模型，建模形变的能力有限。其推理算法可以通过将距离变换当作一种特殊的池化运算，从而转化到CNNs，其训练并不是端到端的，涉及到直觉上的选择，比如组成部分数量和部位大小的选择。比较之下，形变CNNs是端到端的。当多个形变模块堆积时，建模形变的能力变强了。

**DeepID-Net [44]**.It introduces a deformation constrained pooling layer which also considers part deformation for object detection. It therefore shares a similar spirit with deformable RoI pooling, but is much more complex. This work is highly engineered and based on RCNN [16]. It is unclear how to adapt it to the recent state-of-the-art object detection methods [47, 7] in an end-to-end manner.

此文提出了形变约束的池化层，对目标检测考虑了部位形变。因此与形变RoI池化思想类似，但要复杂的多。此文是基于RCNN的，工程工作很多。怎样将其改变以端到端的进行目标检测，尚不清楚。

**Spatial manipulation in RoI pooling**. Spatial pyramid pooling [34] uses hand crafted pooling regions over scales. It is the predominant approach in computer vision and also used in deep learning based object detection [21, 15].

空间金字塔池化[34]使用了不同尺度之间的手工设计的池化区域。这是计算机视觉中的主要方法，在基于深度学习的目标检测中也进行了使用。

Learning the spatial layout of pooling regions has received little study. The work in [28] learns a sparse subset of pooling regions from a large over-complete set. The large set is hand engineered and the learning is not end-to-end.

学习池化区域的空间分布，最近研究很少。[28]中的工作从一个大的过完备集中学习了一个池化区域稀疏子集。这个大型集合是手工设计的，学习并不是端到端的。

Deformable RoI pooling is the first to learn pooling regions end-to-end in CNNs. While the regions are of the same size currently, extension to multiple sizes as in spatial pyramid pooling [34] is straightforward.

形变RoI池化是第一个端到端的学习池化区域的CNNs。目前区域是同样大小的，拓展到多种大小是很直接的。

**Transformation invariant features and their learning**. There have been tremendous efforts on designing transformation invariant features. Notable examples include scale invariant feature transform (SIFT) [42] and ORB [49] (O for orientation). There is a large body of such works in the context of CNNs. The invariance and equivalence of CNN representations to image transformations are studied in [36]. Some works learn invariant CNN representations with respect to different types of transformations such as [50], scattering networks [3], convolutional jungles [32], and TI-pooling [33]. Some works are devoted for specific transformations such as symmetry [13, 9], scale [29], and rotation [53].

设计变换不变的特征，有很多努力。值得注意的例子包括，SIFT和ORB。在CNNs中，这种工作非常多。CNN对图像变换的不变性和等价性在[36]中进行了研究。一些工作学习了对不同类型的变换的不变的CNN表示，如[50,3,32,33]。一些工作致力于具体的变换，如对称性，尺度和旋转。

As analyzed in Section 1, in these works the transformations are known a priori. The knowledge (such as parameterization) is used to hand craft the structure of feature extraction algorithm, either fixed in such as SIFT, or with learnable parameters such as those based on CNNs. They cannot handle unknown transformations in the new tasks.

如第1部分分析，在这些工作中，这些变换都是已知的。这些知识用于手工设计特征提取算法的结构，要么是固定的，如SIFT，或有可学习的参数，比如基于CNNs的。它们不能处理新任务中的未知变换。

In contrast, our deformable modules generalize various transformations (see Figure 1). The transformation invariance is learned from the target task. 比较之下，我们的形变模块泛化到各种变换（见图1）。变换不变性是从目标任务学习的。

**Dynamic Filter [2]**. Similar to deformable convolution, the dynamic filters are also conditioned on the input features and change over samples. Differently, only the filter weights are learned, not the sampling locations like ours. This work is applied for video and stereo prediction. 与形变卷积类似，动态滤波器也是以输入特征为条件的，随着样本进行变化。不同的是，只学习了滤波器权重，而不像我们一样包括采样位置。此工作应用于视频和立体预测中。

**Combination of low level filters**. Gaussian filters and its smooth derivatives [30] are widely used to extract low level image structures such as corners, edges, T-junctions, etc. Under certain conditions, such filters form a set of basis and their linear combination forms new filters within the same group of geometric transformations, such as multiple orientations in Steerable Filters [12] and multiple scales in [45]. We note that although the term deformable kernels is used in [45], its meaning is different from ours in this work.

高斯滤波器和其平滑导数广泛用于提取低层图像结构，如角点，边缘，T形节，等。在特定条件下，这样的滤波器形成了基集，其线性组合形成了相同几何变换组中新的滤波器，比如Steerable Filters中的多方向，和[45]中的多尺度。我们注意到，虽然[45]中使用了形变核的属于，其意义与本文是不一样的。

Most CNNs learn all their convolution filters from scratch. The recent work [25] shows that it could be unnecessary. It replaces the free form filters by weighted combination of low level filters (Gaussian derivatives up to 4-th order) and learns the weight coefficients. The regularization over the filter function space is shown to improve the generalization ability when training data are small.

多数CNNs都从头学习其卷积滤波器。最近的工作[25]表明，这是不必要的，其将低层滤波器的加权组合替换掉自由形式的滤波器，学习权重系数。在滤波器函数空间的正则化，可以在训练数据规模很小的情况下，改进泛化能力。

Above works are related to ours in that, when multiple filters, especially with different scales, are combined, the resulting filter could have complex weights and resemble our deformable convolution filter. However, deformable convolution learns sampling locations instead of filter weights.

上面的工作与我们的工作的关系在于，当组合了多个滤波器时，尤其是不同尺度的，得到的滤波器可以有复杂的权重，与我们的形变卷积滤波器很像。但是，形变卷积学习的采样位置而不是滤波器权重。

## 4. Experiments

### 4.1. Experiment Setup and Implementation

**Semantic Segmentation**. We use PASCAL VOC [10] and CityScapes [6]. For PASCAL VOC, there are 20 semantic categories. Following the protocols in [19, 41, 4], we use VOC 2012 dataset and the additional mask annotations in [18]. The training set includes 10, 582 images. Evaluation is performed on 1, 449 images in the validation set. For CityScapes, following the protocols in [5], training and evaluation are performed on 2, 975 images in the train set and 500 images in the validation set, respectively. There are 19 semantic categories plus a background category.

**语义分割**。我们使用PASCAL VOC和CityScapes。对于PASCAL VOC，与20个语义类别。按照[19,41,4]中的协议，我们使用VOC 2012数据集和[18]中额外的掩模标注。训练数据集包括10582幅图像。评估在验证集中的1449幅图像中进行。对于CityScapes，按照[5]中的协议，训练集包含2975幅图像，在500幅图像的验证集中进行评估。有19个语义类别，加上1个背景类别。

For evaluation, we use the mean intersection-over-union (mIoU) metric defined over image pixels, following the standard protocols [10, 6]. We use mIoU@V and mIoU@C for PASCAl VOC and Cityscapes, respectively.

对评估，按照标准协议，我们使用在图像像素上定义的mIoU。我们对PASCAL VOC和CityScapes分别使用mIoU@V和mIoU@C。

In training and inference, the images are resized to have a shorter side of 360 pixels for PASCAL VOC and 1, 024 pixels for Cityscapes. In SGD training, one image is randomly sampled in each mini-batch. A total of 30k and 45k iterations are performed for PASCAL VOC and Cityscapes, respectively, with 8 GPUs and one mini-batch on each. The learning rates are 10^−3 and 10^−4 in the first 2/3 and the last 1/3 iterations, respectively.

在训练和推理中，我们改变图像大小，使PASCAL VOC中的图像短边有360个像素，CityScapes中图像的短边有1024个像素。在SGD训练中，一个mini-batch中有一幅随机选择的图像。对PASCAL VOC和CityScapes分别有30k和45k次迭代，用8个GPU进行训练，每个上有一个mini-batch。学习速率在前2/3和后1/3分别为10^−3和10^−4。

**Object Detection**. We use PASCAL VOC and COCO [39] datasets. For PASCAL VOC, following the protocol in [15], training is performed on the union of VOC 2007 trainval and VOC 2012 trainval. Evaluation is on VOC 2007 test. For COCO, following the standard protocol [39], training and evaluation are performed on the 120k images in the trainval and the 20k images in the test-dev, respectively.

我们使用PASCAL VOC和COCO数据集。对于PASCAL VOC，按照[15]中的协议，训练在VOC 2007的trainval和VOC 2012 trainval的并集中进行。评估在VOC 2007 test上进行。对于COCO，按照标准协议[39]，训练集trainval包含120k图像，测试集test-dev包含20k图像。

For evaluation, we use the standard mean average precision (mAP) scores [10, 39]. For PASCAL VOC, we report mAP scores using IoU thresholds at 0.5 and 0.7. For COCO, we use the standard COCO metric of mAP@[0.5:0.95], as well as mAP@0.5.

对于评估，我们使用标准的mAP分数。对于PASCAL VOC，我们使用IoU阈值0.5和0.7给出mAP分数。对于COCO，我们使用标准COCO度量mAP@[0.5:0.95]，以及mAP@0.5。

In training and inference, the images are resized to have a shorter side of 600 pixels. In SGD training, one image is randomly sampled in each mini-batch. For class-aware RPN, 256 RoIs are sampled from the image. For Faster R-CNN and R-FCN, 256 and 128 RoIs are sampled for the region proposal and the object detection networks, respectively. 7 × 7 bins are adopted in RoI pooling. To facilitate the ablation experiments on VOC, we follow [38] and utilize pre-trained and fixed RPN proposals for the training of Faster R-CNN and R-FCN, without feature sharing between the region proposal and the object detection networks. The RPN network is trained separately as in the first stage of the procedure in [47]. For COCO, joint training as in [48] is performed and feature sharing is enabled for training. A total of 30k and 240k iterations are performed for PASCAL VOC and COCO, respectively, on 8 GPUs. The learning rates are set as 10−3 and 10−4 in the first 2/3 and the last 1/3 iterations, respectively.

在训练和推理中，我们改变了图像的大小，其短边有600个像素。在SGD训练中，每个mini-batch中是一个随机选择的样本。对于类别感知的RPN，每幅图像采样了256个RoIs。对于Faster R-CNN和R-FCN，对区域候选和目标检测网络分别采样了256和128个RoIs。对RoI池化采用7 × 7 bins。为促进VOC上的分离试验，我们按照[38]利用预训练和固定的RPN候选，训练Faster R-CNN和R-FCN，区域候选和目标检测网络之间并没有特征共享。RPN网络是在第一阶段进行单独训练的。对于COCO，进行了[48]中的联合训练，在训练中开启了特征共享。对PASCAL VOC和COCO，在8个GPUs上分别进行30k和240k次迭代。在前2/3和后1/3的迭代上的学习速率分别是10^−3和10^−4。

### 4.2. Ablation Study

Extensive ablation studies are performed to validate the efficacy and efficiency of our approach. 进行了广泛的分离试验，以验证我们方法的效能和效率。

**Deformable Convolution**. Table 1 evaluates the effect of deformable convolution using ResNet-101 feature extraction network. Accuracy steadily improves when more deformable convolution layers are used, especially for DeepLab and class-aware RPN. The improvement saturates when using 3 deformable layers for DeepLab, and 6 for others. In the remaining experiments, we use 3 in the feature extraction networks.

表1评估了形变卷积的效果，使用ResNet-101特征提取网络。当使用越来越多的形变卷积层时，准确率稳步改进，尤其是对于DeepLab和类别感知的RPN。对于DeepLab，在使用3个形变卷积层时，就没有改进了，对于其他网络则是6个。在剩余的试验中，我们在特征提取网络中使用3个形变卷积层。

We empirically observed that the learned offsets in the deformable convolution layers are highly adaptive to the image content, as illustrated in Figure 5 and Figure 6. To better understand the mechanism of deformable convolution, we define a metric called effective dilation for a deformable convolution filter. It is the mean of the distances between all adjacent pairs of sampling locations in the filter. It is a rough measure of the receptive field size of the filter.

我们通过经验观察到，在形变卷积层中学习到的偏移对图像内容是高度自适应的，如图5和图6所示。为更好的理解形变卷积的机制，我们定义了一个度量，称为形变卷积滤波器的有效膨胀。这是滤波器中所有临近的采样位置点对的平均距离。这是滤波器的感受野大小的粗糙度量。

We apply the R-FCN network with 3 deformable layers (as in Table 1) on VOC 2007 test images. We categorize the deformable convolution filters into four classes: small, medium, large, and background, according to the ground truth bounding box annotation and where the filter center is. Table 2 reports the statistics (mean and std) of the effective dilation values. It clearly shows that: 1) the receptive field sizes of deformable filters are correlated with object sizes, indicating that the deformation is effectively learned from image content; 2) the filter sizes on the background region are between those on medium and large objects, indicating that a relatively large receptive field is necessary for recognizing the background regions. These observations are consistent in different layers.

我们在R-FCN中使用3个形变层，在VOC 2007测试集上进行测试。我们将形变卷积滤波器分成四类：小，中，大和背景，根据真值边界框标准和滤波器中心的位置。表2给出了有效膨胀值的统计（均值和标准差）。这明显表明：1)形变滤波器的感受野大小，是与目标大小相关的，表明从图像内容中有效的学习到了形变；2)在背景上的滤波器大小是在中和大型目标之间的，说明相对较大的感受野是识别背景区域所必须的。这些观察在不同的层中是连续的。

The default ResNet-101 model uses atrous convolution with dilation 2 for the last three 3 × 3 convolutional layers (see Section 2.3). We further tried dilation values 4, 6, and 8 and reported the results in Table 3. It shows that: 1) accuracy increases for all tasks when using larger dilation values, indicating that the default networks have too small receptive fields; 2) the optimal dilation values vary for different tasks, e.g., 6 for DeepLab but 4 for Faster R-CNN; 3) deformable convolution has the best accuracy. These observations verify that adaptive learning of filter deformation is effective and necessary.

默认的ResNet-101模型在最后3个3 × 3的卷积层中使用膨胀为2的atrous卷积。我们进一步尝试了膨胀值4，6和8，在表3中给出了结果。这表明：1)当使用更大的膨胀值时，准确率增加了，说明默认网络的感受野太小；2)最优膨胀值在不同的任务中不同，如，DeepLab最优值是6，Faster R-CNN最优值为4；3)。形变卷积有最佳的准确率。这些观察验证了，滤波器形变的自适应学习是有效和必须的。

**Deformable RoI Pooling**. It is applicable to Faster R-CNN and R-FCN. As shown in Table 3, using it alone already produces noticeable performance gains, especially at the strict mAP@0.7 metric. When both deformable convolution and RoI Pooling are used, significant accuracy improvements are obtained.

这对Faster R-CNN和R-FCN是可应用的。如表3所示，单独使用已经得到了可观察到的性能改进，尤其是在很严格的mAP@0.7度量下。当同时使用了形变卷积和RoI池化时，可以得到显著的准确率改进。

**Model Complexity and Runtime**. Table 4 reports the model complexity and runtime of the proposed deformable ConvNets and their plain versions. Deformable ConvNets only add small overhead over model parameters and computation. This indicates that the significant performance improvement is from the capability of modeling geometric transformations, other than increasing model parameters.

表4给出了提出的形变CNNs和普通版的模型复杂度和运行时间。形变CNNs在模型参数和计算量上只增加了很小的开销。这表明，显著的性能改进是来自于对几何变换的建模能力上，而不是增加模型参数。

### 4.3. Object Detection on COCO

In Table 5, we perform extensive comparison between the deformable ConvNets and the plain ConvNets for object detection on COCO test-dev set. We first experiment using ResNet-101 model. The deformable versions of class-aware RPN, Faster R-CNN and R-FCN achieve mAP@[0.5:0.95] scores of 25.8%, 33.1%, and 34.5% respectively, which are 11%, 13%, and 12% relatively higher than their plain-ConvNets counterparts respectively. By replacing ResNet-101 by Aligned-Inception-ResNet in Faster R-CNN and R-FCN, their plain-ConvNet baselines both improve thanks to the more powerful feature representations. And the effective performance gains brought by deformable ConvNets also hold. By further testing on multiple image scales (the image shorter side is in [480, 576, 688, 864, 1200, 1400]) and performing iterative bounding box average [14], the mAP@[0.5:0.95] scores are increased to 37.5% for the deformable version of R-FCN. Note that the performance gain of deformable ConvNets is complementary to these bells and whistles.

在表5中，我们在COCO test-dev集上对目标检测，在形变CNNs和普通CNNs之间，进行了广泛的比较。我们首先使用ResNet-101模型进行了试验。class-aware RPN, Faster R-CNN和R-FCN的形变版分别获得了25.8%, 33.1%, 34.5%的mAP@[0.5:0.95]，分别比其普通版相对改进了11%, 13%, 12%。在Faster R-CNN和R-FCN中将ResNet-101替换成Aligned-Inception-ResNet，其普通的ConvNet基准都得到了改进。进一步在多个图像尺度上进行测试（图像短边长度为[480, 576, 688, 864, 1200, 1400]），进行迭代的边界框平均，R-FCN的形变版，其mAP@[0.5:0.95]分数增加到37.5%。注意，形变CNNs的性能改进，与其他模块是可补充的。

## 5. Conclusion

This paper presents deformable ConvNets, which is a simple, efficient, deep, and end-to-end solution to model dense spatial transformations. For the first time, we show that it is feasible and effective to learn dense spatial transformation in CNNs for sophisticated vision tasks, such as object detection and semantic segmentation.

本文提出了形变CNNs，这是一种简单，高效，深度，端到端的建模密集空间变换的方法。我们第一次证明了，在CNN中对复杂视觉任务学习密集空间变换是可行的，高效的，在目标检测和语义分割中都进行了应用。