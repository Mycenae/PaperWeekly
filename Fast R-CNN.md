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

### 3.1. Truncated SVD for faster detection

For whole-image classification, the time spent computing the fully connected layers is small compared to the conv layers. On the contrary, for detection the number of RoIs to process is large and nearly half of the forward pass time is spent computing the fully connected layers (see Fig. 2). Large fully connected layers are easily accelerated by compressing them with truncated SVD [5, 23].

In this technique, a layer parameterized by the u × v weight matrix W is approximately factorized as

$$W ≈ UΣ_t V^T$$(5)

using SVD. In this factorization, U is a u × t matrix comprising the first t left-singular vectors of W, $Σ_t$ is a t × t diagonal matrix containing the top t singular values of W, and V is v × t matrix comprising the first t right-singular vectors of W. Truncated SVD reduces the parameter count from uv to t(u + v), which can be significant if t is much smaller than min(u,v). To compress a network, the single fully connected layer corresponding to W is replaced by two fully connected layers, without a non-linearity between them. The first of these layers uses the weight matrix $Σ_t V^T$ (and no biases) and the second uses U (with the original biases associated with W). This simple compression method gives good speedups when the number of RoIs is large.

## 4. Main results

Three main results support this paper’s contributions:

- State-of-the-art mAP on VOC07, 2010, and 2012
- Fast training and testing compared to R-CNN, SPPnet
- Fine-tuning conv layers in VGG16 improves mAP

### 4.1. Experimental setup

Our experiments use three pre-trained ImageNet models that are available online. The first is the CaffeNet (essentially AlexNet [14]) from R-CNN[9]. We alternatively refer 