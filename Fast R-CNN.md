# Fast R-CNN

Ross Girshick / Microsoft Research

## Abstract 摘要

This paper proposes a Fast Region-based Convolutional Network method (Fast R-CNN) for object detection. Fast R-CNN builds on previous work to efficiently classify object proposals using deep convolutional networks. Compared to previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN trains the very deep VGG16 network 9× faster than R-CNN, is 213× faster at test-time, and achieves a higher mAP on PASCAL VOC 2012. Compared to SPPnet, Fast R-CNN trains VGG16 3× faster, tests 10× faster, and is more accurate. Fast R-CNN is implemented in Python and C++ (using Caffe) and is available under the open-source MIT License at https://github.com/rbgirshick/fast-rcnn.

本文提出一种快速的基于区域的卷积神经网络目标检测方法(Fast R-CNN)。Fast R-CNN是在之前R-CNN工作（使用深度卷积网络来高效分类候选目标）基础上的。与前面的工作相比，Fast R-CNN提出了几个创新点来改进训练和测试速度，同时也提高了检测准确率。Fast R-CNN训练非常深的VGG16网络，比R-CNN快9倍，在测试时快了213倍，在PASCAL VOC 2012上得到了更高的mAP。与SPPnet比，Fast R-CNN训练VGG16快了3倍，测试快10倍，准确度更高。Fast R-CNN用Python和C++ (using Caffe)实现，并在Github上开源。

## 1. Introduction 引言

Recently, deep ConvNets [14, 16] have significantly improved image classification [14] and object detection [9, 19] accuracy. Compared to image classification, object detection is a more challenging task that requires more complex methods to solve. Due to this complexity, current approaches (e.g., [9, 11, 19, 25]) train models in multi-stage pipelines that are slow and inelegant.

近年来，深度卷积网络[14,16]显著改进了图像分类[14]和目标检测[9,19]准确率。与图像分类相比，目标检测是更有挑战的工作，需要更复杂的模型来解决。因其复杂性，现有的方法，如[9,11,19,25]，以多阶段管线训练模型，既慢且不优美。

Complexity arises because detection requires the accurate localization of objects, creating two primary challenges. First, numerous candidate object locations (often called “proposals”) must be processed. Second, these candidates provide only rough localization that must be refined to achieve precise localization. Solutions to these problems often compromise speed, accuracy, or simplicity.

复杂性是因为检测需要目标的准确位置，这产生了两个基本挑战。第一，必须处理很多目标位置候选（经常称为“提议”）。第二，这些候选只提供了大致位置，必须提炼得到精确位置。这些问题的解决方案经常在速度、准确度或简洁性之间折中。

In this paper, we streamline the training process for state-of-the-art ConvNet-based object detectors [9, 11]. We propose a single-stage training algorithm that jointly learns to classify object proposals and refine their spatial locations.

本文中，我们使用最新的基于卷积网络的训练过程来构建流线型目标检测器[9,11]。我们提取单阶段训练算法，同时学习候选目标分类并提炼其空间位置。

The resulting method can train a very deep detection network (VGG16 [20]) 9× faster than R-CNN [9] and 3× faster than SPPnet [11]. At runtime, the detection network processes images in 0.3s (excluding object proposal time) while achieving top accuracy on PASCAL VOC 2012 [7] with a mAP of 66% (vs. 62% for R-CNN).(All timings use one Nvidia K40 GPU overclocked to 875 MHz.)

得到的方法可以训练非常深的检测网络(VGG16[20])，这比R-CNN[9]快了9倍，比SPPnet[11]快了3倍。运行时，检测网络处理图像用0.3秒（去除目标候选的时间）就得到了PASCAL VOC 2012[7]的最高准确率，mAP高达66%（R-CNN为62%）（所有时间都是使用超频到875MHz 的Nvidia K40 GPU得到的）。

### 1.1. R-CNN and SPPnet

The Region-based Convolutional Network method (R-CNN) [9] achieves excellent object detection accuracy by using a deep ConvNet to classify object proposals. R-CNN, however, has notable drawbacks:

R-CNN[9]通过使用深度卷积网络来对候选目标进行分类，取得了极好的目标检测准确率。但是R-CNN有严重的缺陷：

1. Training is a multi-stage pipeline. R-CNN first finetunes a ConvNet on object proposals using log loss. Then, it fits SVMs to ConvNet features. These SVMs act as object detectors, replacing the softmax classifier learnt by fine-tuning. In the third training stage, bounding-box regressors are learned.

- 训练是多阶段的结构。首先在候选目标上精调ConvNet，使用log loss。然后使用SVM对ConvNet特征分类。这些SVM实际上是取代了softmax分类器的目标检测器。在第三训练阶段，通过学习得到边界框回归器。

2. Training is expensive in space and time. For SVM and bounding-box regressor training, features are extracted from each object proposal in each image and written to disk. With very deep networks, such as VGG16, this process takes 2.5 GPU-days for the 5k images of the VOC07 trainval set. These features require hundreds of gigabytes of storage.

- 训练耗费极大的存储空间和时间。对于SVM和边界框回归器训练，每幅图像中每个候选目标都要提取特征，然后写入磁盘。对于非常深的网络，如VGG16来说，这种处理在VOC2007 trainval集上的5000幅图像上要耗费2.5 GPU-days。这些特征需要几百G存储空间。

3. Object detection is slow. At test-time, features are extracted from each object proposal in each test image. Detection with VGG16 takes 47s / image (on a GPU).

- 目标检测过程很慢。在测试时，对每幅图像的每个候选目标都计算特征，用VGG16进行检测每幅图像要47秒（在GPU上）。

R-CNN is slow because it performs a ConvNet forward pass for each object proposal, without sharing computation. Spatial pyramid pooling networks (SPPnets) [11] were proposed to speed up R-CNN by sharing computation. The SPPnet method computes a convolutional feature map for the entire input image and then classifies each object proposal using a feature vector extracted from the shared feature map. Features are extracted for a proposal by max-pooling the portion of the feature map inside the proposal into a fixed-size output (e.g., 6 × 6). Multiple output sizes are pooled and then concatenated as in spatial pyramid pooling [15]. SPPnet accelerates R-CNN by 10 to 100× at test time. Training time is also reduced by 3× due to faster proposal feature extraction.

R-CNN非常慢，是因为对每个候选目标都用ConvNet执行一次前向过程，而没有共享计算。空域金字塔pooling网络(SPPnet)[11]提出通过共享计算来加速R-CNN。SPPnet方法计算整幅输入图像的卷积特征图，然后从共享特征图中提取出特征向量，对候选目标进行分类。提取一个候选目标的特征的方法是，将候选目标对应部分的特征图进行max-pooling，得到固定大小的输出（如6×6）。pool出多个输出大小，然后按照[15]空域金字塔pooling的方式将其拼接。SPPnet在测试时加速了R-CNN 10倍到100倍。由于候选目标特征提取速度加快，训练时间也缩短了3倍。

SPPnet also has notable drawbacks. Like R-CNN, training is a multi-stage pipeline that involves extracting features, fine-tuning a network with log loss, training SVMs, and finally fitting bounding-box regressors. Features are also written to disk. But unlike R-CNN, the fine-tuning algorithm proposed in [11] cannot update the convolutional layers that precede the spatial pyramid pooling. Unsurprisingly, this limitation (fixed convolutional layers) limits the accuracy of very deep networks.

SPPnet也有一些缺陷。和R-CNN一样，训练是多阶段管道式的，涉及到特征提取，精调log loss的网络，训练SVM，最后学习边界框回归器。特征也写入磁盘。但与R-CNN不同的是，[11]中提出的精调算法不能更新空域金字塔pooling之前的卷积层。这个限制（固定卷积层）限制了非常深网络的准确度。

### 1.2. Contributions 贡献

We propose a new training algorithm that fixes the disadvantages of R-CNN and SPPnet, while improving on their speed and accuracy. We call this method Fast R-CNN because it’s comparatively fast to train and test. The Fast R-CNN method has several advantages:

我们提出一种新的训练算法，修复了R-CNN和SPPnet的上述缺点，改进了其速度和准确性。我们称这种方法为Fast R-CNN因为训练和测试都相对很快。Fast R-CNN方法有以下几处优点：

1. Higher detection quality (mAP) than R-CNN, SPPnet 更高的检测质量mAP
2. Training is single-stage, using a multi-task loss 单阶段训练，多任务损失函数
3. Training can update all network layers 训练可以更新所有网络层
4. No disk storage is required for feature caching 不需要在磁盘上缓存特征

Fast R-CNN is written in Python and C++ (Caffe [13]) and is available under the open-source MIT License at https://github.com/rbgirshick/fast-rcnn.

## 2. Fast R-CNN architecture and training 架构和训练

Fig. 1 illustrates the Fast R-CNN architecture. A Fast R-CNN network takes as input an entire image and a set of object proposals. The network first processes the whole image with several convolutional (conv) and max pooling layers to produce a conv feature map. Then, for each object proposal a region of interest (RoI) pooling layer extracts a fixed-length feature vector from the feature map. Each feature vector is fed into a sequence of fully connected (fc) layers that finally branch into two sibling output layers: one that produces softmax probability estimates over K object classes plus a catch-all “background” class and another layer that outputs four real-valued numbers for each of the K object classes. Each set of 4 values encodes refined bounding-box positions for one of the K classes.

图1所示的就是Fast R-CNN的架构。Fast R-CNN网络以一整幅图像和候选目标集为输入。网络首先对整幅图像用几个卷积层conv和max pooling层来产生conv特征图。然后，对于每个候选目标都有一个感兴趣区域(ROI) pooling层从特征图中提取一个固定长度的特征矢量。每个特征矢量送入一系列全连接层fc中，最后进入两个分支输出层：一个产生K个目标类加上一个背景类的softmax概率估计，另一个输出4个实数值，也就是K类目标中其中一类的边界框。

Figure 1. Fast R-CNN architecture. An input image and multiple regions of interest (RoIs) are input into a fully convolutional network. Each RoI is pooled into a fixed-size feature map and then mapped to a feature vector by fully connected layers (FCs). The network has two output vectors per RoI: softmax probabilities and per-class bounding-box regression offsets. The architecture is trained end-to-end with a multi-task loss.

图1 Fast R-CNN架构。一幅图像和多个感兴趣区域RoI输入一个全卷积网络。每个RoI经过pool操作得到固定尺寸的特征图，然后通过全连接层映射到一个特征向量。网络对每个RoI有两个输出矢量：softmax概率和每类的边界框回归偏移。架构是端到端训练的，损失函数则是多任务的。

### 2.1. The RoI pooling layer

The RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of H × W (e.g., 7×7), where H and W are layer hyper-parameters that are independent of any particular RoI. In this paper, an RoI is a rectangular window into a conv feature map. Each RoI is defined by a four-tuple (r,c,h,w) that specifies its top-left corner (r,c) and its height and width (h,w).

RoI pooling层使用max pooling来转换任何RoI内部的特征成为一个小的特征图，固定空域尺寸为H × W (如7×7)，其中H和W分别为超参数，与任何特定的RoI无关。在本文中，RoI是一个conv特征图中的矩形窗口。每个RoI定义为元组(r,c,h,w)，其中(r,c)为做上角位置，(h,w)为高和宽。

RoI max pooling works by dividing the h × w RoI window into an H × W grid of sub-windows of approximate size h/H × w/W and then max-pooling the values in each sub-window into the corresponding output grid cell. Pooling is applied independently to each feature map channel, as in standard max pooling. The RoI layer is simply the special-case of the spatial pyramid pooling layer used in SPPnets [11] in which there is only one pyramid level. We use the pooling sub-window calculation given in [11].

RoI max pooling通过将h×w的RoI窗口分成H×W的网格，每个网格都是大小约为h/W × w/W的子窗口，然后对每个子窗口进行max-pooling得到输出。pooling独立的对每个特性图通道进行，就像标准的max pooling一样。RoI层仅仅是SPPnets[11]中使用的空域金字塔pooling层的特殊情况，只不过只有一个金字塔层。我们用[11]中的子窗口pooling计算方法。

### 2.2. Initializing from pre-trained networks 用预训练网络初始化

We experiment with three pre-trained ImageNet [4] networks, each with five max pooling layers and between five and thirteen conv layers (see Section 4.1 for network details). When a pre-trained network initializes a Fast R-CNN network, it undergoes three transformations.

我们用三种预训练的ImageNet[4]网络进行试验，每种都在5到13个卷积层中包含5个max pooling层（网络细节详见4.1节）。当预训练网络初始化R-CNN网络时，需要经过三种变换。

First, the last max pooling layer is replaced by a RoI pooling layer that is configured by setting H and W to be compatible with the net’s first fully connected layer (e.g., H = W = 7 for VGG16).

首先，最后的max pooling层替换为RoI pooling层，配置为设定H和W使网络与第一个全连接层兼容（如，对于VGG16来说，H=W=7）。

Second, the network’s last fully connected layer and softmax (which were trained for 1000-way ImageNet classification) are replaced with the two sibling layers described earlier (a fully connected layer and softmax over K+1 categories and category-specific bounding-box regressors).

第二，网络最后的全连接层和softmax层（它们是为ImageNet的1000路分类训练的）替换为前述的两个输出层（一个全连接层和softmax层，类别数为K+1，和一个类别相关的边界框回归层）。

Third, the network is modified to take two data inputs: a list of images and a list of RoIs in those images.

最后，网络修改为接收两种数据输入：图像列表和图像中的RoI列表。

### 2.3. Fine-tuning for detection 为检查进行精调

Training all network weights with back-propagation is an important capability of Fast R-CNN. First, let’s elucidate why SPPnet is unable to update weights below the spatial pyramid pooling layer.

用反向传播训练所有网络权重是Fast R-CNN的重要能力。首先，我们阐明为什么SPPnet不能更新空域金字塔pooling层之下的网络权重。

The root cause is that back-propagation through the SPP layer is highly inefficient when each training sample (i.e. RoI) comes from a different image, which is exactly how R-CNN and SPPnet networks are trained. The inefficiency stems from the fact that each RoI may have a very large receptive field, often spanning the entire input image. Since the forward pass must process the entire receptive field, the training inputs are large (often the entire image).

根本原因是当每个训练样本（即RoI）来自不同图像时，SPP中的反向传播是效率很低的，这也正是R-CNN和SPPnet网络训练的方法。这种低效源自每个RoI可能有很大的感受野，经常张满整个输入图像。因为前向过程必须处理整个图像，所以训练输入是大的（经常是整个图像）。

We propose a more efficient training method that takes advantage of feature sharing during training. In Fast R-CNN training, stochastic gradient descent (SGD) mini-batches are sampled hierarchically, first by sampling N images and then by sampling R/N RoIs from each image. Critically, RoIs from the same image share computation and memory in the forward and backward passes. Making N small decreases mini-batch computation. For example, when using N = 2 and R = 128, the proposed training scheme is roughly 64× faster than sampling one RoI from 128 different images (i.e., the R-CNN and SPPnet strategy).

我们提出一种更加高效的训练方法，利用了训练过程中的特征分享。在Fast R-CNN训练时，mini-batch的随机梯度下降层次化的进行取样，首先取样N个图像，然后每幅图像取样R/N个RoI。同一幅图像中的RoI在前向和后向过程中分享计算和内存。取小一些的N值，可以降低mini-batch的计算量。例如，当N=2, R=128时，利用提出的训练方案进行计算，比N=128的情况下要快64倍（这也是R-CNN和SPPnet的策略）。

One concern over this strategy is it may cause slow training convergence because RoIs from the same image are correlated. This concern does not appear to be a practical issue and we achieve good results with N = 2 and R = 128 using fewer SGD iterations than R-CNN.

这种策略的一种担心是可能导致训练收敛变慢，因为同一幅图像中的RoI是相关的。这种担心并没有成为实际问题，我们用N=2, R=128得到了很好的结果，而且比R-CNN的用到的SGD迭代次数要少。

In addition to hierarchical sampling, Fast R-CNN uses a streamlined training process with one fine-tuning stage that jointly optimizes a softmax classifier and bounding-box regressors, rather than training a softmax classifier, SVMs, and regressors in three separate stages [9, 11]. The components of this procedure (the loss, mini-batch sampling strategy, back-propagation through RoI pooling layers, and SGD hyper-parameters) are described below.

除了层次化取样，Fast R-CNN使用流线型的训练过程，还有一个精调的阶段，可以同时优化softmax分类器和边界框回归器，而不是在三个不同阶段训练softmax分类、SVM和回归器[9,11]。这个过程的组成部分（损失函数、mini-batch取样策略、通过RoI pooling层的反向传播和SGD超参数）在下面详述。

**Multi-task loss**. A Fast R-CNN network has two sibling output layers. The first outputs a discrete probability distribution (per RoI), $p = (p_0 ,...,p_K)$, over K + 1 categories. As usual, p is computed by a softmax over the K+1 outputs of a fully connected layer. The second sibling layer outputs bounding-box regression offsets, $t^k = (t^k_x, t^k_y, t^k_w, t^k_h)$, for each of the K object classes, indexed by k. We use the parameterization for $t^k$ given in [9], in which $t^k$ specifies a scale-invariant translation and log-space height/width shift relative to an object proposal.

**多任务损失函数**。Fast R-CNN网络有两个输出层。第一个输出K+1类上的离散概率分布（每个RoI），$p = (p_0 ,...,p_K)$。和普通的一样，p的计算是将全连接层的K+1维输出通过softmax层。第二个层输出的是K个目标类别的边界框回归偏移，$t^k = (t^k_x, t^k_y, t^k_w, t^k_h)$，以k为索引。我们使用[9]中给定的参数化的$t^k$，其中$t^k$指定了一个尺度不变的平移和相对于候选目标的log-space高/宽偏移。

Each training RoI is labeled with a ground-truth class u and a ground-truth bounding-box regression target v. We use a multi-task loss L on each labeled RoI to jointly train for classification and bounding-box regression:

每个训练RoI都有ground-truth类别u的标签，和ground-truth边界框回归目标v。我们在每个标记的RoI上使用多任务损失函数L，来同时训练分类和边界框回归：

$$L(p,u,t^u ,v) = L_{cls}(p,u) + λ[u ≥ 1] L_{loc} (t^u ,v)$$(1)

in which $L_{cls} (p,u) = −logp_u$ is log loss for true class u. 其中$L_{cls} (p,u) = −logp_u$是真实类别u的log损失。

The second task loss, $L_{loc}$, is defined over a tuple of true bounding-box regression targets for class u, $v =(v_x ,v_y ,v_w ,v_h )$, and a predicted tuple $t^u = (t^u_x, t^u_y, t^u_w, t^u_h)$, again for class u. The Iverson bracket indicator function [u ≥ 1] evaluates to 1 when u ≥ 1 and 0 otherwise. By convention the catch-all background class is labeled u = 0. For background RoIs there is no notion of a ground-truth bounding box and hence $L_{loc}$ is ignored. For bounding-box regression, we use the loss

第二个损失$L_{loc}$，定义于一个类别u的真实边界框回归的目标，元组$v =(v_x ,v_y ,v_w ,v_h )$，和一个预测的类别u的元组$t^u = (t^u_x, t^u_y, t^u_w, t^u_h)$之间。Iverson括号指示函数[u ≥ 1]当u ≥ 1时为1，其他为0。按照惯例，背景类别的标签是u=0。对于背景RoI，没有ground-truth边界框的概念，所以忽略其$L_{loc}$。对于边界框回归，我们使用的损失函数为

$$L_{loc} (t^u ,v) = \sum_{i∈(x,y,w,h)} smooth_{L_1} (t^u_i − v_i )$$(2)

in which 其中

$smooth_{L_1} (x) = 0.5x^2$ $if |x| < 1$ or $= |x| − 0.5$ $otherwise$              (3)

is a robust $L_1$ loss that is less sensitive to outliers than the $L_2$ loss used in R-CNN and SPPnet. When the regression targets are unbounded, training with $L_2$ loss can require careful tuning of learning rates in order to prevent exploding gradients. Eq. 3 eliminates this sensitivity.

这是一个鲁棒的$L_1$损失，对离群点没$L_2$损失那么敏感，$L_2$损失是用在R-CNN和SPPnet中的。当回归目标无界时，用$L_2$损失来训练需要小心调节学习速率，以防止梯度爆炸。式(3)消除了这种敏感性。

The hyper-parameter λ in Eq. 1 controls the balance between the two task losses. We normalize the ground-truth regression targets $v_i$ to have zero mean and unit variance. All experiments use λ = 1.

式(1)中的超参数λ控制了两个任务损失函数间的平衡。我们将ground-truth回归目标$v_i$归一化为零均值单位方差。所有的试验中使用λ = 1。

We note that [6] uses a related loss to train a class-agnostic object proposal network. Different from our approach, [6] advocates for a two-network system that separates localization and classification. OverFeat [19], R-CNN [9], and SPPnet [11] also train classifiers and bounding-box localizers, however these methods use stage-wise training, which we show is suboptimal for Fast R-CNN (Section 5.1).

我们注意到[6]使用了一个相关的损失函数来训练类别无关的候选目标网络。与我们的方法不同，[6]主张双网络系统将定位与分类分开。OverFeat [19], R-CNN [9]和SPPnet [11]也训练分类器和边界框定位器，但这些方法都使用了分阶段的训练，这相对于Fast R-CNN来说都是次优的（见5.1节）。

**Mini-batch sampling**. During fine-tuning, each SGD mini-batch is constructed from N = 2 images, chosen uniformly at random (as is common practice, we actually iterate over permutations of the dataset). We use mini-batches of size R = 128, sampling 64 RoIs from each image. As in [9], we take 25% of the RoIs from object proposals that have intersection over union (IoU) overlap with a ground-truth bounding box of at least 0.5. These RoIs comprise the examples labeled with a foreground object class, i.e. u ≥ 1. The remaining RoIs are sampled from object proposals that have a maximum IoU with ground-truth in the interval [0.1,0.5), following [11]. These are the background examples and are labeled with u = 0. The lower threshold of 0.1 appears to act as a heuristic for hard example mining [8]. During training, images are horizontally flipped with probability 0.5. No other data augmentation is used.

**Mini-batch取样**。在精调的过程中，每个SGD mini-batch由2幅图像构成，随机选取（实际上我们在整个数据集的排列上迭代，这很普通）。我们使用mini-batch大小R=128，从每个图像中取样64个RoI。如同[9]中一样，我们所取的候选目标中的RoI，要与ground-truth边界框至少有0.5的交集(IoU)，这样的RoI取出25%。这些RoI组成的样本标签是前景目标类别，即u ≥ 1。根据[11]，剩下的从候选目标中选取的RoI是这样进行的，其与ground-truth的IoU在[0.1,0.5)范围内的最大值。这些是背景的样本，标签是u=0。更低的阈值0.1可以作为难分样本挖掘的heuristic。在训练过程中，图像以0.5的概率水平翻转。没有使用数据扩充。

**Back-propagation through RoI pooling layers**. Back-propagation routes derivatives through the RoI pooling layer. For clarity, we assume only one image per mini-batch (N = 1), though the extension to N > 1 is straightforward because the forward pass treats all images independently.

**反向传播通过RoI pooling层**。反向传播经过RoI pooling层传递导数。为清晰起见，我们假设每mini-batch只有一幅图像(N=1)，但扩展到N>1是直接的，因为前向过程对所有图像都是单独处理的。

Let $x_i ∈ R$ be the i-th activation input into the RoI pooling layer and let $y_{rj}$ be the layer’s j-th output from the r-th RoI. The RoI pooling layer computes $y_{rj} = x_{i^∗ (r,j)}$, in which $i^∗ (r,j) = argmax_{i'∈R(r,j)}  x_{i'}$. $R(r,j)$ is the index set of inputs in the sub-window over which the output unit $y_{rj}$ max pools. A single $x_i$ may be assigned to several different outputs $y_{rj}$.

令$x_i ∈ R$为输入到RoI pooling层的第i个激活值，令$y_{rj}$为本层从第r个RoI的第j个输出。RoI pooling层计算$y_{rj} = x_{i^∗ (r,j)}$，其中$i^∗ (r,j) = argmax_{i'∈R(r,j)}  x_{i'}$。$R(r,j)$是输入子窗口的索引集，输出单元$y_{rj}$在这上面进行max pool。一个$x_i$可能会被赋值给几个不同的输出$y_{rj}$。

The RoI pooling layer’s `backwards` function computes partial derivative of the loss function with respect to each input variable $x_i$ by following the argmax switches:

RoI pooling层的`backwards`函数计算损失函数对每个输入变量$x_i$的偏导数，

$$\frac {∂L} {∂x_i} = \sum_r \sum_j [i = i^*(r,j)] \frac {∂L} {∂y_{rj}}$$(4)

In words, for each mini-batch RoI r and for each pooling output unit $y_{rj}$, the partial derivative $∂L/∂y_{rj}$ is accumulated if i is the argmax selected for $y_{rj}$ by max pooling. In back-propagation, the partial derivatives $∂L/∂y_{rj}$ are already computed by the `backwards` function of the layer on top of the RoI pooling layer.

用语言叙述一下，对于每个mini-batch RoI r，对每个pooling输出单元$y_{rj}$，如果i是max pooling选出的$y_{rj}$的索引，那么偏导数$∂L/∂y_{rj}$就累加。在反向传播中，偏导数$∂L/∂y_{rj}$已经被RoI pooling层的上一层的`backwards`函数计算了。

**SGD hyper-parameters**. The fully connected layers used for softmax classification and bounding-box regression are initialized from zero-mean Gaussian distributions with standard deviations 0.01 and 0.001, respectively. Biases are initialized to 0. All layers use a per-layer learning rate of 1 for weights and 2 for biases and a global learning rate of 0.001. When training on VOC07 or VOC12 trainval we run SGD for 30k mini-batch iterations, and then lower the learning rate to 0.0001 and train for another 10k iterations. When we train on larger datasets, we run SGD for more iterations, as described later. A momentum of 0.9 and parameter decay of 0.0005 (on weights and biases) are used.

**SGD超参数**。用于softmax分类和边界框回归的全连接层分别从零均值标准差为0.01和0.001的Gaussian分布中初始化。偏置初始化为0。所有的层使用权值学习速率1，偏置学习速率2，全局学习速率0.001。当在VOC07或VOC12 trainval集上训练时，我们运行SGD 30k个mini-batch迭代，然后降低学习速率到0.0001，然后继续训练10k个迭代。当我们在更大的数据集上训练时，我们运行的SGD迭代更多，后面会详述。动量取0.9，（权值和偏重的）参数衰减取0.0005。

## 3. Fast R-CNN detection

Once a Fast R-CNN network is fine-tuned, detection amounts to little more than running a forward pass (assuming object proposals are pre-computed). The network takes as input an image (or an image pyramid, encoded as a list of images) and a list of R object proposals to score. At test-time, R is typically around 2000, although we will consider cases in which it is larger (≈ 45k). When using an image pyramid, each RoI is assigned to the scale such that the scaled RoI is closest to $224^2$ pixels in area [11].

Fast R-CNN网络精调好之后，检测相当于比运行一次正向过程多一点点（假设候选目标是计算好的）。网络以一幅图像（或一个图像金字塔，编码为图像列表）和R个候选目标的列表作为输入进行评分。在测试时，R通常为2000，我们也会考虑更大的情况(≈ 45k)。当使用图像金字塔时，每个RoI都指定一个尺度，这个尺度的RoI与$224^2$像素最接近[11]。

For each test RoI r, the forward pass outputs a class posterior probability distribution p and a set of predicted bounding-box offsets relative to r (each of the K classes gets its own refined bounding-box prediction). We assign a detection confidence to r for each object class k using the estimated probability $Pr(class = k | r) = p_k$. We then perform non-maximum suppression independently for each class using the algorithm and settings from R-CNN [9].

对于每个测试RoI r，前向过程输出一个类别的后验概率分布p，和有关r的预测边界框偏移集（K类的每个都得到自己的提炼过的边界框预测）。我们给r指定一个每个目标类别k的检测信心，使用的是预测概率$Pr(class = k | r) = p_k$。我们然后对每个类别独立进行非最大抑制，使用的算法和设置都与R-CNN[9]相同。

### 3.1. Truncated SVD for faster detection 截断SVD加速检测

For whole-image classification, the time spent computing the fully connected layers is small compared to the conv layers. On the contrary, for detection the number of RoIs to process is large and nearly half of the forward pass time is spent computing the fully connected layers (see Fig. 2). Large fully connected layers are easily accelerated by compressing them with truncated SVD [5, 23].

对于整图分类，在全连接层的计算时间是小于在卷积层的时间的。对于检测来说则相反，要处理的RoI数量很多，几乎一半的前向过程的时间是在计算全连接层（见图2）。大型全连接层很容易用截断SVD[5,23]压缩来加速。

Figure 2. Timing for VGG16 before and after truncated SVD. Before SVD, fully connected layers fc6 and fc7 take 45% of the time.

In this technique, a layer parameterized by the u × v weight matrix W is approximately factorized as 在这种技术中，一层的权值矩阵W，大小u×v，用SVD近似分解为：

$$W ≈ UΣ_t V^T$$(5)

using SVD. In this factorization, U is a u × t matrix comprising the first t left-singular vectors of W, $Σ_t$ is a t × t diagonal matrix containing the top t singular values of W, and V is v × t matrix comprising the first t right-singular vectors of W. Truncated SVD reduces the parameter count from uv to t(u + v), which can be significant if t is much smaller than min(u,v). To compress a network, the single fully connected layer corresponding to W is replaced by two fully connected layers, without a non-linearity between them. The first of these layers uses the weight matrix $Σ_t V^T$ (and no biases) and the second uses U (with the original biases associated with W). This simple compression method gives good speedups when the number of RoIs is large.

在这个分解中，U是一个u×t的矩阵由W的前t个左奇异矢量构成，$Σ_t$是t×t对角矩阵，包括W的最大t个奇异值，V是v×t矩阵，由W的前t个右奇异矢量构成。截断SVD将参数数量由uv个减少到t(u+v)个，如果t远小于min(u,v)，那么减少量是很大的。为压缩网络，对应W的单个全连接层替换为两个全连接层，中间没有非线性处理。第一层使用权值矩阵$Σ_t V^T$（没有偏置），第二个使用U作为权值矩阵 （使用之前W相关的原偏置）。这种简单压缩方法在RoI数量巨大时，加速效果明显。

## 4. Main results 主要结果

Three main results support this paper’s contributions: 三个主要结果支撑本文的贡献：

- State-of-the-art mAP on VOC07, 2010, and 2012 在VOC07,2010和2012上的最好mAP
- Fast training and testing compared to R-CNN, SPPnet 与R-CNN、SPPnet相比的快速训练和测试
- Fine-tuning conv layers in VGG16 improves mAP 精调VGG16的卷积层改进mAP

### 4.1. Experimental setup 试验设置

Our experiments use three pre-trained ImageNet models that are available online. The first is the CaffeNet (essentially AlexNet [14]) from R-CNN[9]. We alternatively refer to this CaffeNet as model S, for “small.” The second network is VGG_CNN_M_1024 from [3], which has the same depth as S, but is wider. We call this network model M, for “medium.” The final network is the very deep VGG16 model from [20]. Since this model is the largest, we call it model L. In this section, all experiments use single-scale training and testing (s = 600; see Section 5.2 for details).

我们的试验使用三种在线可用的预训练ImageNet模型。第一个是R-CNN[9]中的CaffeNet（本质上就是AlexNet[14]）。我们称CaffeNet为模型S，为small的缩写。第二种网络是[3]中的VGG_CNN_M_1024，其深度与S相同，但更宽一些。我们称这个网络为M，为medium的缩写。最后一个网络是非常深的[20]中的VGG16模型。由于这个模型是最大的，我们称之为模型L。在本节中，所有试验使用单尺度训练和测试（s=600，详见5.2节）。

### 4.2. VOC 2010 and 2012 results

On these datasets, we compare Fast R-CNN (FRCN, for short) against the top methods on the comp4 (outside data) track from the public leaderboard (Table 2, Table 3). For the NUS_NIN_c2000 and BabyLearning methods, there are
no associated publications at this time and we could not find exact information on the ConvNet architectures used; they are variants of the Network-in-Network design [17]. All other methods are initialized from the same pre-trained VGG16 network.

在这个数据集上，我们将Fast R-CNN与公开排行榜上的最好方法相比。对于NUS_NIN_c2000和BabyLearning方法，目前还没有相关的论文，也不知道其使用的卷积网络架构的确切信息；它们是Network-in-Network[17]的变体。所有其他方法都从相同的预训练VGG16网络初始化。

Fast R-CNN achieves the top result on VOC12 with a mAP of 65.7% (and 68.4% with extra data). It is also two orders of magnitude faster than the other methods, which are all based on the “slow” R-CNN pipeline. On VOC10, SegDeepM [25] achieves a higher mAP than Fast R-CNN (67.2% vs. 66.1%). SegDeepM is trained on VOC12 trainval plus segmentation annotations; it is designed to boost R-CNN accuracy by using a Markov random field to reason over R-CNN detections and segmentations from the O2P[1] semantic-segmentation method. Fast R-CNN can be swapped into SegDeepM in place of R-CNN, which may lead to better results. When using the enlarged 07++12 training set (see Table 2 caption), Fast R-CNN’s mAP increases to 68.8%, surpassing SegDeepM.

Fast R-CNN在VOC12上以65.7%的mAP得到了最高结果（有额外数据时达到了68.4%），也比其他方法快了两个数量级，其他方法都是基于慢速的R-CNN流程。在VOC10上，SegDeepM[25]比Fast R-CNN得到了更高的mAP值(67.2% vs. 66.1%)。SegDeepM在VOC12 trainval上训练，外加分割标注；这是设计用于加速R-CNN准确度的。Fast R-CNN可以替换SegDeepM中的R-CNN，可能得到更好结果。当使用扩大的07++12训练集时（见表2说明），Fast R-CNN的mAP增加到了68.8%，超过了SegDeepM。

Table 2. VOC 2010 test detection average precision (%). BabyLearning uses a network based on [17]. All other methods use VGG16. Training set key: 12: VOC12 trainval, Prop.: proprietary dataset, 12+seg: 12 with segmentation annotations, 07++12: union of VOC07 trainval, VOC07 test, and VOC12 trainval.

Table 3. VOC 2012 test detection average precision (%). BabyLearning and NUS NIN c2000 use networks based on [17]. All other methods use VGG16. Training set key: see Table 2, Unk.: unknown.

### 4.3. VOC 2007 results

On VOC07, we compare Fast R-CNN to R-CNN and SPPnet. All methods start from the same pre-trained VGG16 network and use bounding-box regression. The VGG16 SPPnet results were computed by the authors of [11]. SPPnet uses five scales during both training and testing. The improvement of Fast R-CNN over SPPnet illustrates that even though Fast R-CNN uses single-scale training and testing, fine-tuning the conv layers provides a large improvement in mAP (from 63.1% to 66.9%). R-CNN achieves a mAP of 66.0%. As a minor point, SPPnet was trained without examples marked as “difficult” in PASCAL. Removing these examples improves Fast R-CNN mAP to 68.1%. All other experiments use “difficult” examples.

在VOC07上，我们将Fast R-CNN与R-CNN和SPPnet作对比。所有方法都从相同的预训练VGG16网络开始，使用边界框回归。VGG16 SPPnet的结果由[11]的作者计算得到。SPPnet在训练和测试时使用5种尺度。Fast R-CNN相对于SPPnet的改进结果说明，即使Fast R-CNN使用了单尺度的训练和测试，精调卷积层提供了很大的mAP改进（从63.1%到66.9%）。R-CNN的mAP为66.0%。SPPnet没有使用PASCAL中标记为难的样本进行训练。去除这些样本Fast R-CNN将mAP提升到68.1%。所有其他试验都使用了难样本。

### 4.4. Training and testing time 训练及测试时间

Fast training and testing times are our second main result. Table 4 compares training time (hours), testing rate (seconds per image), and mAP on VOC07 between Fast R-CNN, R-CNN, and SPPnet. For VGG16, Fast R-CNN processes images 146× faster than R-CNN without truncated SVD and 213× faster with it. Training time is reduced by 9×, from 84 hours to 9.5. Compared to SPPnet, Fast R-CNN trains VGG16 2.7× faster (in 9.5 vs. 25.5 hours) and tests 7× faster without truncated SVD or 10× faster with it. Fast R-CNN also eliminates hundreds of gigabytes of disk
storage, because it does not cache features.

快速训练和测试时我们的第二个主要结果。表4对比了Fast R-CNN，R-CNN和SPPnet在VOC07上的训练时间（小时）、测试速度（每幅图像的秒数）和mAP。对于VGG16，Fast R-CNN处理图像的速度在没有使用截断SVD时比R-CNN快了146倍，使用后快了213倍。训练时间减少了9倍，从84小时减少到9.5小时。与SPPnet相比，Fast R-CNN训练VGG16网络快了2.7倍（9.5小时对比25.5小时），测试在没用截断SVD时快了7倍，使用后快了10倍。Fast R-CNN还节省了几百G的磁盘存储空间，因为不需要缓存特征。

Table 4. Runtime comparison between the same models in Fast R-CNN, R-CNN, and SPPnet. Fast R-CNN uses single-scale mode. SPPnet uses the five scales specified in [11]. † Timing provided by the authors of [11]. Times were measured on an Nvidia K40 GPU.

**Truncated SVD**. Truncated SVD can reduce detection time by more than 30% with only a small (0.3 percentage point) drop in mAP and without needing to perform additional fine-tuning after model compression. Fig. 2 illustrates how using the top 1024 singular values from the 25088×4096 matrix in VGG16’s fc6 layer and the top 256 singular values from the 4096×4096 fc7 layer reduces runtime with little loss in mAP. Further speed-ups are possible with smaller drops in mAP if one fine-tunes again after compression.

**截断SVD**。截断SVD可以减少多达30%的检测时间，mAP只减少很少(0.3%)，而且在模型压缩后不需要进行额外的精调。图2所示的是使用VGG16的fc6层的25088×4096矩阵的最大的1024个奇异值，以及fc7层的4096×4096矩阵的最大的256个奇异值，减少了运算时间，而且mAP降低很小。如果压缩后再精调，还可以进一步加速网络。

## 4.5. Which layers to fine-tune? 应当精调哪些层？

For the less deep networks considered in the SPPnet paper [11], fine-tuning only the fully connected layers appeared to be sufficient for good accuracy. We hypothesized that this result would not hold for very deep networks. To validate that fine-tuning the conv layers is important for VGG16, we use Fast R-CNN to fine-tune, but freeze the thirteen conv layers so that only the fully connected layers learn. This ablation emulates single-scale SPPnet training and decreases mAP from 66.9% to 61.4% (Table 5). This experiment verifies our hypothesis: training through the RoI pooling layer is important for very deep nets.

对于SPPnet[11]中使用的没那么深的的网络，只对全连接层进行精调似乎足够了。我们假设这个结果对于非常深的网络是不成立的。为验证精调卷积层对于VGG16是重要的，我们使用Fast R-CNN来精调，但冻结前13个卷积层，所以只有全连接层得到学习。这种简化试验模仿单尺度SPPnet训练，降低mAP从66.9%到61.4%（表5）。这个试验验证了我们的假设：对于非常深网络来说，经过RoI pooling层来训练是非常重要的。

Table 5. Effect of restricting which layers are fine-tuned for VGG16. Fine-tuning ≥ fc6 emulates the SPPnet training algorithm [11], but using a single scale. SPPnet L results were obtained using five scales, at a significant (7×) speed cost.

Does this mean that all conv layers should be fine-tuned? In short, no. In the smaller networks (S and M) we find that conv1 is generic and task independent (a well-known fact [14]). Allowing conv1 to learn, or not, has no meaningful effect on mAP. For VGG16, we found it only necessary to update layers from conv3_1 and up (9 of the 13 conv layers). This observation is pragmatic: (1) updating from conv2_1 slows training by 1.3× (12.5 vs. 9.5 hours) compared to learning from conv3_1; and (2) updating from conv1_1 over-runs GPU memory. The difference in mAP when learning from conv2_1 up was only +0.3 points (Table 5, last column). All Fast R-CNN results in this paper using VGG16 fine-tune layers conv3_1 and up; all experiments with models S and M fine-tune layers conv2 and up.

这是否意味着所有卷积层都需要精调呢？简单来说，不是。在较小的网络(S和M)中，我们发现conv1层是通用的，也是与任务无关的（一个广为人知的事实[14]）。是否精调conv1对mAP没有实质性影响。对于VGG16来说，我们发现只需要更新conv3_1以上的卷积层（13个卷积层中的9个）。这种观察是实用的：(1)从conv2_1更新与从conv3_1相比，使训练减慢了1.3倍（12.5小时 vs. 9.5小时）；(2)从conv1_1开始更新GPU内存不够用。从conv2_1更新，只提升了0.3%的mAP（表5，最后一列）。本文所有Fast R-CNN结果都使用VGG16精调conv3_1以上的层；所有模型S和M的试验都精调conv2以上的层。

## 5. Design evaluation 设计评估

We conducted experiments to understand how Fast R-CNN compares to R-CNN and SPPnet, as well as to evaluate design decisions. Following best practices, we performed these experiments on the PASCAL VOC07 dataset.

我们进行了多个试验来理解Fast R-CNN和R-CNN、SPPnet的对比，以及评估设计决定。我们在PASCAL VOC数据集上进行这些试验。

### 5.1. Does multi-task training help? 多任务训练有用吗？

Multi-task training is convenient because it avoids managing a pipeline of sequentially-trained tasks. But it also has the potential to improve results because the tasks influence each other through a shared representation (the ConvNet)[2]. Does multi-task training improve object detection accuracy in Fast R-CNN?

多任务训练是很方便的，因为避免了连续训练的任务。但也有改进结果的潜能，因为任务通过共享的表示(ConvNet[2])互相影响。多任务训练在Fast R-CNN中改进了目标检测的训练吗？

To test this question, we train baseline networks that use only the classification loss, $L_{cls}$, in Eq. 1 (i.e., setting λ = 0). These baselines are printed for models S, M, and L in the first column of each group in Table 6. Note that these models do not have bounding-box regressors. Next (second column per group), we take networks that were trained with the multi-task loss (Eq. 1, λ = 1), but we disable bounding-box regression at test time. This isolates the networks’ classification accuracy and allows an apples-to-apples comparison with the baseline networks.

为检验这个问题，我们只用式(1)分类损失$L_{cls}$训练一个基准网络。这个基准在表6中模型S、M和L的每组第一列给出。注意这些模型没有边界框回归器。每组的第二列，我们使用多任务损失训练的网络，但测试时没有用边界框回归。这分离出了网络的分类准确率，可以与基准网络逐项进行比较。

Across all three networks we observe that multi-task training improves pure classification accuracy relative to training for classification alone. The improvement ranges from +0.8 to +1.1 mAP points, showing a consistent positive effect from multi-task learning.

三个网络之间我们观察到多任务训练改进了分类准确率，这是相对于只进行分类的网络。改进幅度为0.8%到1.1% mAP，显示了多任务学习确实有正面作用。

Finally, we take the baseline models (trained with only the classification loss), tack on the bounding-box regression layer, and train them with $L_{loc}$ while keeping all other network parameters frozen. The third column in each group shows the results of this stage-wise training scheme: mAP improves over column one, but stage-wise training underperforms multi-task training (forth column per group).

最后，我们采用基准模型，外加边界框回归层，然后用$L_{loc}$进行训练，同时保持网络其他参数冻结。每组第三列显示了这种分阶段训练方案的结果：mAP相比于第一列有改进，但分阶段训练性能没有超过多任务训练（每组第4列）。

### 5.2. Scale invariance: to brute force or finesse? 尺度不变性：暴力解决还是巧妙解决？

We compare two strategies for achieving scale-invariant object detection: brute-force learning (single scale) and image pyramids (multi-scale). In either case, we define the scale s of an image to be the length of its shortest side.

我们对比了得到目标识别尺度不变性的两种策略：暴力学习（单尺度）和图像金字塔（多尺度）。在任一情况中，我们都定义图像的尺度s为其短边的长度。

All single-scale experiments use s = 600 pixels; s may be less than 600 for some images as we cap the longest image side at 1000 pixels and maintain the image’s aspect ratio. These values were selected so that VGG16 fits in GPU memory during fine-tuning. The smaller models are not memory bound and can benefit from larger values of s; however, optimizing s for each model is not our main concern. We note that PASCAL images are 384 × 473 pixels on average and thus the single-scale setting typically upsamples images by a factor of 1.6. The average effective stride at the RoI pooling layer is thus ≈ 10 pixels.

所有单尺度试验使用s=600像素；对一些图像s可能小于600，因为我们使图像长边最长为1000像素，而且维持图像纵横比不变。选择了这些值，VGG16才能在GPU内存精调（太大了内存不够用）。更小的模型不受内存限制，s值大一些会更好；但是，对每个模型优化s不是我们的主要考虑。我们注意PASCAL图像平均大小为384×473像素，所以单尺度设置一般会将图像进行1.6倍的上采样。RoI pooling层的平均有效步长因此是大约10像素。

In the multi-scale setting, we use the same five scales specified in [11] (s ∈ {480,576,688,864,1200}) to facilitate comparison with SPPnet. However, we cap the longest side at 2000 pixels to avoid exceeding GPU memory.

在多尺度设置中，我们使用与[11]中相同的5种尺度，即s ∈ {480,576,688,864,1200}，以与SPPnet进行比较。但是，我们设定长边最长为2000像素，以免超出GPU内存。

Table 7 shows models S and M when trained and tested with either one or five scales. Perhaps the most surprising result in [11] was that single-scale detection performs almost as well as multi-scale detection. Our findings confirm their result: deep ConvNets are adept at directly learning scale invariance. The multi-scale approach offers only a small increase in mAP at a large cost in compute time (Table 7). In the case of VGG16 (model L), we are limited to using a single scale by implementation details. Yet it achieves a mAP of 66.9%, which is slightly higher than the 66.0% reported for R-CNN [10], even though R-CNN uses “infinite” scales in the sense that each proposal is warped to a canonical size.

表7所示的是模型S和M，在使用单尺度或5尺度时进行训练和测试时的结果。可能[11]中最令人惊奇的结果是单尺度检测与多尺度检测的结果机会一样好。我们的发现确认了他们的结果：深度卷积网络可以很熟练的直接学习尺度不变性。多尺度方法对提升mAP的作用很小，但计算时间的代价比较大（表7）。在VGG16的L模型的情况下，我们受限于实现细节，只使用了单尺度。但是得到了66.9% mAP的结果，这比R-CNN[10]中的结果66.0%略高一点，即使R-CNN使用的无限尺度，因为每个候选区域变形为统一大小。

Table 7. Multi-scale vs. single scale. SPPnet ZF (similar to model S) results are from [11]. Larger networks with a single-scale offer the best speed / accuracy tradeoff. (L cannot use multi-scale in our implementation due to GPU memory constraints.)

Since single-scale processing offers the best tradeoff between speed and accuracy, especially for very deep models, all experiments outside of this sub-section use single-scale training and testing with s = 600 pixels.

由于单尺度处理在速度和准确度间的折中最好，尤其是非常深的模型，本小结以外的所有试验都使用单尺度训练和测试，参数s=600像素。

### 5.3. Do we need more training data? 我们需要更多训练数据吗？

A good object detector should improve when supplied with more training data. Zhu et al. [24] found that DPM [8] mAP saturates after only a few hundred to thousand training examples. Here we augment the VOC07 trainval set with the VOC12 trainval set, roughly tripling the number of images to 16.5k, to evaluate Fast R-CNN. Enlarging the training set improves mAP on VOC07 test from 66.9% to 70.0% (Table 1). When training on this dataset we use 60k mini-batch iterations instead of 40k.

好的目标检测器应当在提供更多数据时，性能有所改进。Zhu et al. [24]发现DPM[8]在几百至几千幅训练图像后mAP就饱和了。这里我们使用VOC12 trainval集扩充VOC07 trainval集，使图像数量增加到大约3倍，16.5k幅，来评估Fast R-CNN。增大训练集使得在VOC07上的测试结果从66.9%提升至70.0%（表1）。当在这个数据集上训练时，我们使用60k次mini-batch迭代，而不是40k次。

We perform similar experiments for VOC10 and 2012, for which we construct a dataset of 21.5k images from the union of VOC07 trainval, test, and VOC12 trainval. When training on this dataset, we use 100k SGD iterations and lower the learning rate by 0.1× each 40k iterations (instead of each 30k). For VOC10 and 2012, mAP improves from 66.1% to 68.8% and from 65.7% to 68.4%, respectively.

我们对VOC10和2012进行类似的试验，从VOC07 trainval,test和VOC12 trainval集的合并构建了一个数据集包含了21.5k幅图像。当使用这个数据集训练时，我们使用100k次SGD迭代，每40k次迭代将学习速率降低10倍。对于VOC10和VOC2012，mAP分别从66.1%改进至68.8%，从65.7%改进至68.4%。

### 5.4. Do SVMs outperform softmax? SVM比softmax要好吗？

Fast R-CNN uses the softmax classifier learnt during fine-tuning instead of training one-vs-rest linear SVMs post-hoc, as was done in R-CNN and SPPnet. To understand the impact of this choice, we implemented post-hoc SVM training with hard negative mining in Fast R-CNN. We use the same training algorithm and hyper-parameters as in R-CNN.

Fast R-CNN在精调时使用softmax分类器，而没有训练线性SVM分类器，在R-CNN和SPPnet中都是这样做的。为理解这个选择的影响，我们在Fast R-CNN中实现了用难分负样本挖掘训练的SVM。我们使用了与R-CNN一样的训练算法和超参数。

Table 8 shows softmax slightly outperforming SVM for all three networks, by +0.1 to +0.8 mAP points. This effect is small, but it demonstrates that “one-shot” fine-tuning is sufficient compared to previous multi-stage training approaches. We note that softmax, unlike one-vs-rest SVMs, introduces competition between classes when scoring a RoI.

表8所示的是在三种网络上softmax比SVM略好一些的结果，大约0.1%到0.8% mAP。这个影响是很小的，但说明了one-shot精调与之前的多阶段训练方法比较，是充分好用的。我们注意到softmax不像SVM一样，在给RoI评分时，不同类之间会有竞争。

Table 8. Fast R-CNN with softmax vs. SVM (VOC07 mAP).

method | classifier | S | M | L
--- | --- | --- | --- | ---
R-CNN [9, 10] | SVM | 58.5 | 60.2 | 66.0
FRCN [ours] | SVM | 56.3 | 58.7 | 66.8
FRCN [ours] | softmax | 57.1 | 59.2 | 66.9

### 5.5. Are more proposals always better? 我们的候选一直都更好吗？

There are (broadly) two types of object detectors: those that use a sparse set of object proposals (e.g., selective search [21]) and those that use a dense set (e.g., DPM [8]). Classifying sparse proposals is a type of cascade [22] in which the proposal mechanism first rejects a vast number of candidates leaving the classifier with a small set to evaluate. This cascade improves detection accuracy when applied to DPM detections [21]. We find evidence that the proposal-classifier cascade also improves Fast R-CNN accuracy.

大致上有两类目标检测器：使用稀疏候选目标集（如selective search[21]），和使用稠密集的（如DPM[8]）。稀疏候选分类是级联类型的[22]，其中推荐机制首先拒绝了大量候选，给分类器剩下很小一个候选集来评估。当应用到DPM检测[21]的时候，这种级联改进检测准确率。我们发现这种推荐-分类器的级联也改进Fast R-CNN的准确率。

Using selective search’s quality mode, we sweep from 1k to 10k proposals per image, each time re-training and re-testing model M. If proposals serve a purely computational role, increasing the number of proposals per image should not harm mAP.

使用selective search的质量模式，我们从每幅图像中得到1k到10k幅图像，每次重新训练并重新测试图像M。如果推荐只是作为计算的角色，增加每幅图像候选的数量不会降低mAP。

We find that mAP rises and then falls slightly as the proposal count increases (Fig. 3, solid blue line). This experiment shows that swamping the deep classifier with more proposals does not help, and even slightly hurts, 
accuracy.

我们发现当候选数量增加时，mAP首先上升，然后略微下降（图3中的蓝色实线）。这个实验说明，给深度分类器提供更多的候选没有多少帮助，还使准确率略微下降。

Figure 3. VOC07 test mAP and AR for various proposal schemes.

This result is difficult to predict without actually running the experiment. The state-of-the-art for measuring object proposal quality is Average Recall (AR) [12]. AR correlates well with mAP for several proposal methods using R-CNN, when using a fixed number of proposals per image. Fig. 3 shows that AR (solid red line) does not correlate well with mAP as the number of proposals per image is varied. AR must be used with care; higher AR due to more proposals does not imply that mAP will increase. Fortunately, training and testing with model M takes less than 2.5 hours. Fast R-CNN thus enables efficient, direct evaluation of object proposal mAP, which is preferable to proxy metrics.

如果没有实际运行实验，这个结果是很难预测到的。衡量候选目标质量的最新指标是平均召回率(Average Recall, AR)[12]。当每幅图像的候选目标数量固定时，AR与mAP很有关联，这在使用R-CNN的几种候选方法中都有体现。AR必须小心使用；由于候选更多带来的AR越高并不一定保证mAP会改进。幸运的是，模型M的训练和测试耗时少于2.5小时。Fast R-CNN使得对候选目标mAP的评估更加直接高效，相对代理指标来说，这是很好的。

We also investigate Fast R-CNN when using densely generated boxes (over scale, position, and aspect ratio), at a rate of about 45k boxes / image. This dense set is rich enough that when each selective search box is replaced by its closest (in IoU) dense box, mAP drops only 1 point (to 57.7%, Fig. 3, blue triangle).

我们还研究了使用密集生成边界框情况下的Fast R-CNN（不同尺度，位置和纵横比），大概每幅图像45k个边界框。这种稠密集内容丰富，当每个selective search边界框替换成最接近(IoU)的密集框时，mAP下降了大约1个百分点（到57.7%，图3中的蓝色三角形）。

The statistics of the dense boxes differ from those of selective search boxes. Starting with 2k selective search boxes, we test mAP when adding a random sample of 1000 × {2,4,6,8,10,32,45} dense boxes. For each experiment we re-train and re-test model M. When these dense boxes are added, mAP falls more strongly than when adding more selective search boxes, eventually reaching 53.0%.

稠密边界框的统计与那些selective search边界框的不同。从2k个selective search边界框开始，我们加入1000 × {2,4,6,8,10,32,45}个密集边界框的随机样本，然后我们测试mAP。对于每个试验，我们重新训练、重新测试模型M。当这些密集框加入时，mAP下降的很多，最终到达53.0%。

We also train and test Fast R-CNN using only dense boxes (45k / image). This setting yields a mAP of 52.9% (blue diamond). Finally, we check if SVMs with hard negative mining are needed to cope with the dense box distribution. SVMs do even worse: 49.3% (blue circle).

我们还只使用密集边界框训练和测试了Fast R-CNN（每幅图像45k个）。这种设置得到了52.9%的mAP（蓝色星型）。最终，我们检查是否带有难分负样本挖掘的SVM可以和密集边界框分布合拍，SVM的结果甚至更差：49.3%（蓝色圆形）。

### 5.6. Preliminary MS COCO results 在MS COCO上的初步结果

We applied Fast R-CNN (with VGG16) to the MS COCO dataset [18] to establish a preliminary baseline. We trained on the 80k image training set for 240k iterations and evaluated on the “test-dev” set using the evaluation server. The PASCAL-style mAP is 35.9%; the new COCO-style AP, which also averages over IoU thresholds, is 19.7%.

我们将Fast R-CNN(VGG16)应用在MS COCO数据集[18]上，来确定初步基准。我们训练了80k幅图像进行了240k次迭代，使用评估服务器在test-dev集上进行评估。PASCAL-style mAP为35.9%；而新的COCO-style mAP为19.7%，在IoU阈值上进行了平均。

## 6. Conclusion 结论

This paper proposes Fast R-CNN, a clean and fast update to R-CNN and SPPnet. In addition to reporting state-of-the-art detection results, we present detailed experiments that we hope provide new insights. Of particular note, sparse object proposals appear to improve detector quality. This issue was too costly (in time) to probe in the past, but becomes practical with Fast R-CNN. Of course, there may exist yet undiscovered techniques that allow dense boxes to perform as well as sparse proposals. Such methods, if developed, may help further accelerate object detection.

本文提出了Fast R-CNN，是R-CNN和SPPnet的一种干净快速的更新。除了报告了最新监测结果，我们还给出了详细的试验过程，希望能提供新的思想。特意强调的是，稀疏候选目标似乎可以改进检测器质量。这个问题在以前很难调查得到，因为试验时间太长，但在Fast R-CNN下可以进行试验了。当然，可能还存在尚未发现的技术，可以使密集边界框与稀疏候选得到一样的效果。这种方法如果发现，可能进一步加速目标检测。