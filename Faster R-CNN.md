# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

Shaoqing Ren  Kaiming He Ross Girshick Jian Sun Microsoft Research

## Abstract  摘要

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet [7] and Fast R-CNN [5] have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully-convolutional network that simultaneously predicts object bounds and objectness scores at each position. RPNs are trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. With a simple alternating optimization, RPN and Fast R-CNN can be trained to share convolutional features. For the very deep VGG-16 model [19], our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007 (73.2% mAP) and 2012 (70.4% mAP) using 300 proposals per image. Code is available at https://github.com/ShaoqingRen/faster_rcnn.

目前最好的目标识别网络依靠候选区域算法来假设目标位置。SPPnet[7]和Fast R-CNN[5]这样的进展减少了这些检测网络的运行时间，使候选区域计算成为算法瓶颈。在本文中，我们引入了一种候选区域网络(RPN)，与检测网络共享整图卷积特征，所以区域候选算法就几乎没有代价了。RPN是一个全卷积网络，可以在每个位置同时预测目标边界和目标评分。RPN的训练是端到端的，生成高质量的候选区域，供Fast R-CNN检测使用。通过简单的交替优化，RPN和Fast R-CNN可以共享卷积特征。对于非常深的VGG-16模型[19]，我们的检测系统在GPU上可以达到5FPS的速度，也取得了目前最佳的目标检测准确率，在PASCAL VOC 2007上为73.2% mAP，2012上为70.4% mAP，每个图像的候选区域为300个。代码已经公开。

## 1 Introduction 引言

Recent advances in object detection are driven by the success of region proposal methods (e.g., [22]) and region-based convolutional neural networks (R-CNNs) [6]. Although region-based CNNs were computationally expensive as originally developed in [6], their cost has been drastically reduced thanks to sharing convolutions across proposals [7, 5]. The latest incarnation, Fast R-CNN [5], achieves near real-time rates using very deep networks [19], when ignoring the time spent on region proposals. Now, proposals are the computational bottleneck in state-of-the-art detection systems.

目标检测的最新进展是由区域候选方法（如[22]）和基于区域的卷积神经网络(R-CNN)[6]的成功驱动的。虽然基于区域的CNN在刚提出的时候[6]计算量很大，但通过在候选之间共享卷积[7,5]得到了急剧减少。其最新的成果Fast R-CNN[5]，如果忽略其区域候选算法的时间，使用非常深的网络[19]取得了接近实时的速度。现在，在最好的检测系统中，区域候选算法成为了计算了瓶颈。

Region proposal methods typically rely on inexpensive features and economical inference schemes. Selective Search (SS) [22], one of the most popular methods, greedily merges superpixels based on engineered low-level features. Yet when compared to efficient detection networks [5], Selective Search is an order of magnitude slower, at 2s per image in a CPU implementation. EdgeBoxes [24] currently provides the best tradeoff between proposal quality and speed, at 0.2s per image. Nevertheless, the region proposal step still consumes as much running time as the detection network.

区域候选方法一般依靠廉价特征和经济的推理方案。Selctive Search (SS) [22]是一种最流行的方法，通过合并基于底层特征设计得到的超像素实现候选算法。但与高效的检测网络[5]比，Selective Search慢了一个数量级，在CPU实现时约2秒每幅图像。EdgeBoxes[24]最近在候选质量和速度上取得了最佳折中，达到了每幅图像0.2秒的速度。但是，区域候选步骤仍然与检测网络消耗的时间类似。

One may note that fast region-based CNNs take advantage of GPUs, while the region proposal methods used in research are implemented on the CPU, making such runtime comparisons inequitable. An obvious way to accelerate proposal computation is to re-implement it for the GPU. This may be an effective engineering solution, but re-implementation ignores the down-stream detection network and therefore misses important opportunities for sharing computation.

应当注意到，Fast R-CNN利用GPU计算，但区域候选方法仍然是用CPU计算的，这种运算时间的对比是不公平的。加速候选计算的一种明显方法是，在GPU上重新实现。这应当是有效的工程解决方法，但重新实现忽略了检测网络的顺流结构，所以不能进行有效的共享计算。

In this paper, we show that an algorithmic change—computing proposals with a deep net—leads to an elegant and effective solution, where proposal computation is nearly cost-free given the detection network’s computation. To this end, we introduce novel Region Proposal Networks (RPNs) that share convolutional layers with state-of-the-art object detection networks [7, 5]. By sharing convolutions at test-time, the marginal cost for computing proposals is small (e.g., 10ms per image).

在本文中，我们采用深度网络计算区域候选，这种算法的改变带来了一种优雅高效的解决方案，给定了检测网络的计算后，其中候选区域计算基本是没有代价的。为此，我们提出了新的区域候选网络(RPN)，与目前最好的目标检测网络[7,5]共享卷积层。通过在测试时共享卷积，计算候选的边缘代价非常小（如，每幅图像10毫秒）。

Our observation is that the convolutional (conv) feature maps used by region-based detectors, like Fast R-CNN, can also be used for generating region proposals. On top of these conv features, we construct RPNs by adding two additional conv layers: one that encodes each conv map position into a short (e.g., 256-d) feature vector and a second that, at each conv map position, outputs an objectness score and regressed bounds for k region proposals relative to various scales and aspect ratios at that location (k = 9 is a typical value).

我们观察到，像Fast R-CNN这样基于区域的检测器使用的卷积特征图，也可以用作生成候选区域。在这些卷积特征之上，我们增加了另外两层卷积层来构建RPN：一层在每个卷积特征图位置上编码为形成一个短的特征向量（如256维），第二层在每个卷积特征图的位置上，对k个不同尺度和纵横比的候选区域，每个都输出目标评分和回归得到的边界（k=9是典型值）。

Our RPNs are thus a kind of fully-convolutional network (FCN) [14] and they can be trained end-to-end specifically for the task for generating detection proposals. To unify RPNs with Fast R-CNN [5] object detection networks, we propose a simple training scheme that alternates between fine-tuning for the region proposal task and then fine-tuning for object detection, while keeping the proposals fixed. This scheme converges quickly and produces a unified network with conv features that are shared between both tasks.

所以我们的RPN是一种全卷积网络(FCN)[14]，可以进行端到端的训练，然后专门进行候选区域的生成。为将RPN和Fast R-CNN[5]目标检测网络统一起来，我们提出一种简单的训练方案，交替进行精调区域候选任务和在固定候选的情况下精调目标检测网络。这种方案收敛快，得到的网络其卷积特征在两种任务之间是共享的。

We evaluate our method on the PASCAL VOC detection benchmarks [4], where RPNs with Fast R-CNNs produce detection accuracy better than the strong baseline of Selective Search with Fast R-CNNs. Meanwhile, our method waives nearly all computational burdens of SS at test-time—the effective running time for proposals is just 10 milliseconds. Using the expensive very deep models of [19], our detection method still has a frame rate of 5fps (including all steps) on a GPU, and thus is a practical object detection system in terms of both speed and accuracy (73.2% mAP on PASCAL VOC 2007 and 70.4% mAP on 2012). Code is available at https://github.com/ShaoqingRen/faster_rcnn.

我们在PASCAL VOC标准测试[4]上评估我们的方法，其中RPN与Fast R-CNN得到的检测准确度优于Selective Search和Fast R-CNN的强基准。同时，我们的方法在测试时基本没有SS的计算负担，候选计算的有效运行时间只有10毫秒。使用了计算量很大的非常深的模型[19]，我们的检测方法在GPU上仍然可以达到5FPS（包括所有步骤），所以在速度和准确率上都是一种实用的目标检测系统（在PASCAL VOC 2007上73.2% mAP，在2012上70.4% mAP）。代码已经开源。

## 2 Related Work 相关工作

Several recent papers have proposed ways of using deep networks for locating class-specific or class-agnostic bounding boxes [21, 18, 3, 20]. In the OverFeat method [18], a fully-connected (fc) layer is trained to predict the box coordinates for the localization task that assumes a single object. The fc layer is then turned into a conv layer for detecting multiple class-specific objects. The MultiBox methods [3, 20] generate region proposals from a network whose last fc layer simultaneously predicts multiple (e.g., 800) boxes, which are used for R-CNN [6] object detection. Their proposal network is applied on a single image or multiple large image crops (e.g., 224×224) [20]. We discuss OverFeat and MultiBox in more depth later in context with our method.

几篇最近的文章都提出使用深度网络来定位类别相关或无关的边界框[21,18,3,20]。在OverFeat方法[18]中，训练了一个全连接层(fc)来预测单个目标的边界框的坐标。fc层然后就变成了卷积层，进行检测多类别的目标。MultiBox[3,20]方法用网络来生成候选区域，网络最后的fc层同时预测多个（如800个）边界框，用于R-CNN[6]目标检测。他们的区域候选网络应用在单个图像或多个大的图像块（如224×224）中[20]。我们在后面与我们的方法一起再深度讨论OverFeat和MultiBox。

Shared computation of convolutions [18, 7, 2, 5] has been attracting increasing attention for efficient, yet accurate, visual recognition. The OverFeat paper [18] computes conv features from an image pyramid for classification, localization, and detection. Adaptively-sized pooling (SPP) [7] on shared conv feature maps is proposed for efficient region-based object detection [7, 16] and semantic segmentation [2]. Fast R-CNN [5] enables end-to-end detector training on shared conv features and shows compelling accuracy and speed.

共享卷积计算[18,7,2,5]可以得到高效又准确的视觉识别任务，吸引了越来越多的注意力。论文OverFeat[18]从图像金字塔中计算卷积特征来分类、定位和检测。共享卷积特征图的自适应大小pooling (SPP) [7]的提出，可以进行高效的基于区域的目标检测[7,16]和语义分割[2]。Fast R-CNN[5]可以在卷积特征上进行端到端的检测器训练，得到很好的准确度和速度。

## 3 Region Proposal Networks 区域候选网络

A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of rectangular object proposals, each with an objectness score(“Region” is a generic term and in this paper we only consider rectangular regions, as is common for many methods (e.g., [20, 22, 24]). “Objectness” measures membership to a set of object classes vs. background.). We model this process with a fully-convolutional network [14], which we describe in this section. Because our ultimate goal is to share computation with a Fast R-CNN object detection network [5], we assume that both nets share a common set of conv layers. In our experiments, we investigate the Zeiler and Fergus model [23] (ZF), which has 5 shareable conv layers and the Simonyan and Zisserman model [19] (VGG), which has 13 shareable conv layers.

区域候选网络(RPN)以任意大小的图像为输入，输出矩形目标候选集，每个都包含一个目标评分（区域是一个一般性词语，本文中我们只考虑矩形区域，这在很多方法中都很常见，如[20,22,24]。目标评分衡量的是属于目标类别或背景的程度。）我们用全卷积网络[14]来对这个过程进行建模，在本节中进行详述。因为我们的最终目标是与Fast R-CNN目标检测网络[5]共享计算，我们假设两个网络存在共享卷积层。在我们的实验中，我们研究了Zeiler and Fergus模型[23](ZF)，有5个可共享的卷积层，还研究了VGG[19]模型，有13个可共享的卷积层。

To generate region proposals, we slide a small network over the conv feature map output by the last shared conv layer. This network is fully connected to an n × n spatial window of the input conv feature map. Each sliding window is mapped to a lower-dimensional vector (256-d for ZF and 512-d for VGG). This vector is fed into two sibling fully-connected layers—a box-regression layer (reg) and a box-classification layer (cls). We use n = 3 in this paper, noting that the effective receptive field on the input image is large (171 and 228 pixels for ZF and VGG, respectively). This mini-network is illustrated at a single position in Fig. 1 (left). Note that because the mini-network operates in a sliding-window fashion, the fully-connected layers are shared across all spatial locations. This architecture is naturally implemented with an n × n conv layer followed by two sibling 1 × 1 conv layers (for reg and cls, respectively). ReLUs [15] are applied to the output of the n × n conv layer.

为生成候选区域，我们在最后一个共享卷积层输出的卷积特征图上用一个小网络滑过。这个网络与一个输入卷积特征图的n × n空间窗口全连接。每个滑窗都映射到一个低维矢量中（对于ZF是256维，对于VGG是512维）。这个矢量然后送入两个全连接层中，一个进行边界框回归(reg)，一个进行分类(cls)。本文中我们使用n=3，注意输入图像的有效感受野是很大的（对于ZF和VGG分别是171和228）。这个迷你网络在图1（左）进行了单独描述。注意因为这个迷你网络要进行滑窗操作，全连接层在所有的空间位置上都是共享的。这种架构很自然的可以用n × n卷积层后跟着两个1 × 1的卷积层（分别对应reg和cls）实现。n × n卷积层的输出进行了ReLU[15]激活。

Figure 1: Left: Region Proposal Network (RPN). Right: Example detections using RPN proposals on PASCAL VOC 2007 test. Our method detects objects in a wide range of scales and aspect ratios.

图1：左：RPN；右：在PASCAL VOC 2007测试集上使用RPN候选进行的检测例子，我们的方法检测各种尺度和纵横比的对象。

**Translation-Invariant Anchors** **平移不变的锚窗**

At each sliding-window location, we simultaneously predict k region proposals, so the reg layer has 4k outputs encoding the coordinates of k boxes. The cls layer outputs 2k scores that estimate probability of object / not-object for each proposal (For simplicity we implement the cls layer as a two-class softmax layer. Alternatively, one may use logistic regression to produce k scores). The k proposals are parameterized relative to k reference boxes, called anchors. Each anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio. We use 3 scales and 3 aspect ratios, yielding k = 9 anchors at each sliding position. For a conv feature map of a size W × H (typically ∼2,400), there are WHk anchors in total. An important property of our approach is that it is translation invariant, both in terms of the anchors and the functions that compute proposals relative to the anchors.

在每个滑窗位置，我们同时预测k个候选区域，所以reg层有k个边界框坐标的4k个输出编码。cls层输出2k个评分，估计每个候选区域是目标或不是目标的概率（为简化，我们将cls层实现为2类softmax层，也可以使用logistic回归来产生k个评分）。这k个候选区域是相对于k个参考窗口的，称为锚窗。每个锚窗都处于滑窗的中心，包含一个尺度参数和纵横比参数。我们使用3个尺度，3种纵横比，也就是在每个滑窗位置产生k=9个锚窗。对于一个大小为W × H的卷积特征图（一般大约为2400），总计就有WHk个锚窗。我们方法的一个重要性质就是具有平移不变性，对于锚窗是，对于相对于锚窗计算候选窗口的函数也是。

As a comparison, the MultiBox method [20] uses k-means to generate 800 anchors, which are not translation invariant. If one translates an object in an image, the proposal should translate and the same function should be able to predict the proposal in either location. Moreover, because the MultiBox anchors are not translation invariant, it requires a (4+1)×800-dimensional output layer, whereas our method requires a (4+2)×9-dimensional output layer. Our proposal layers have an order of magnitude fewer parameters (27 million for MultiBox using GoogLeNet [20] vs. 2.4 million for RPN using VGG-16), and thus have less risk of overfitting on small datasets, like PASCAL VOC.

作为一个对比，MultiBox方法[20]使用了k均值法来生成800个锚窗，且不是平移不变的。如果在图像中平移了一个目标，那么候选窗也要平移，同样的函数应当可以能够在相应位置对候选窗进行预测。而且，由于MultiBox锚窗不是平移不变的，需要(4+1)×800维输出层，而我们的方法只需要(4+2)×9维输出层。我们提出的层参数少了一个数量级（对于使用GoogLeNet[20]的MultiBox为27m，使用VGG-16的RPN为2.4m个），所以在较小的数据集如PASCAL VOC上不太会过拟合。

**A Loss Function for Learning Region Proposals** **学习候选区域的损失函数**

For training RPNs, we assign a binary class label (of being an object or not) to each anchor. We assign a positive label to two kinds of anchors: (i) the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box. Note that a single ground-truth box may assign positive labels to multiple anchors. We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes. Anchors that are neither positive nor negative do not contribute to the training objective.

为训练RPN，我们为每个锚窗指定了一个二值类别标签（是否是一个目标）。我们为两种锚窗指定一个正标签：(i)与ground-truth边界框重叠IoU最高的锚窗；(ii)与ground-truth边界框的IoU重叠度高于0.7的锚窗。注意单个ground-truth边界框可能指定多个正标签的锚窗。对于非正的锚窗，如果对于所有ground-truth边界框其IoU低于0.3，就指定为负标签。非正非负的锚窗对训练目标函数没有贡献。

With these definitions, we minimize an objective function following the multi-task loss in Fast R-CNN [5]. Our loss function for an image is defined as: 在这些定义下，我们按照Fast R-CNN[5]中的多任务损失的形式最小化目标函数。对于一幅图像，我们的损失函数定义为：

$$L(\{p_i\},\{t_i\})=\sum_i L_{cls}(p_i, p_i^*) /N_{cls} + λ\sum_i p_i^* L_{reg} (t_i, t_i^*)/N_{reg}$$(1)

Here, i is the index of an anchor in a mini-batch and $p_i$ is the predicted probability of anchor i being an object. The ground-truth label $p^∗_i$ is 1 if the anchor is positive, and is 0 if the anchor is negative. $t_i$ is a vector representing the 4 parameterized coordinates of the predicted bounding box, and $t^∗_i$ is that of the ground-truth box associated with a positive anchor. The classification loss $L_{cls}$ is log loss over two classes (object vs. not object). For the regression loss, we use $L_{reg} (t_i , t^∗_i) = R(t_i − t^∗_i)$ where R is the robust loss function (smooth $L_1$) defined in [5]. The term $p^∗_i L_{reg}$ means the regression loss is activated only for positive anchors ($p^∗_i = 1$) and is disabled otherwise ($p^∗_i = 0$). The outputs of the cls and reg layers consist of {$p_i$} and {$t_i$} respectively. The two terms are normalized with $N_{cls}$ and $N_{reg}$, and a balancing weight λ(In our early implementation (as also in the released code), λ was set as 10, and the cls term in Eqn.(1) was normalized by the mini-batch size (i.e., $N_{cls}$ = 256) and the reg term was normalized by the number of anchor locations (i.e., $N_{reg}$ ∼ 2, 400). Both cls and reg terms are roughly equally weighted in this way).

这里，i是一个mini-batch中一个锚窗的索引，$p_i$是预测锚窗是一个目标的概率。如果锚窗为正，ground-truth标签$p^∗_i$就是1，锚窗为负则为0。$t_i$是预测的边界框的4个坐标参数构成的矢量，$t^∗_i$是正锚窗的ground-truth边界框的坐标构成的矢量。分类损失$L_{cls}$是两类的log损失（目标或非目标）。对于回归损失，我们使用$L_{reg} (t_i , t^∗_i) = R(t_i − t^∗_i)$，其中R是鲁棒的损失函数[5]（平滑$L_1$范数）。$p^∗_i L_{reg}$项意思是，回归损失只在正锚窗($p^∗_i = 1$)时存在，否则($p^∗_i = 0$)为零。cls层和reg层的输出分别由{$p_i$}和{$t_i$}组成。这两项用$N_{cls}$和$N_{reg}$归一化，平衡权重为λ（在我们前期的实验中，λ设为10，式1中的cls项由mini-batch的大小来归一化，即$N_{cls}$ = 256，reg项由锚窗位置个数归一化，即$N_{reg}$ ∼ 2, 400；这样cls和reg项都大致平均加权；这在放出的代码中也有说明）。

For regression, we adopt the parameterizations of the 4 coordinates following [6]: 对于回归，我们采用[6]中4坐标参数：

$$t_x=(x-x_a)/w_a, t_y=(y-y_a)/h_a,t_w=log(w/w_a),t_h=log(h/h_a)$$
$$t^*_x=(x^*-x_a)/w_a, t^*_y=(y^*-y_a)/h_a,t^*_w=log(w^*/w_a), t^*_h=log(h^*/h_a)$$

where x, y, w, and h denote the two coordinates of the box center, width, and height. Variables x, $x_a$, and $x^∗$ are for the predicted box, anchor box, and ground-truth box respectively (likewise for y, w, h). This can be thought of as bounding-box regression from an anchor box to a nearby ground-truth box.

这里x, y, w和h表示边界框的中心坐标、宽度和高度。变量x, $x_a$和$x^∗$分别表示预测边界框、锚窗和ground-truth边界框的数值（对y, w, h是类似的）。这可以认为是锚窗对附近ground-truth边界框的回归。

Nevertheless, our method achieves bounding-box regression by a different manner from previous feature-map-based methods [7, 5]. In [7, 5], bounding-box regression is performed on features pooled from arbitrarily sized regions, and the regression weights are shared by all region sizes. In our formulation, the features used for regression are of the same spatial size (n × n) on the feature maps. To account for varying sizes, a set of k bounding-box regressors are learned. Each regressor is responsible for one scale and one aspect ratio, and the k regressors do not share weights. As such, it is still possible to predict boxes of various sizes even though the features are of a fixed size/scale.

不过，我们的方法进行边界框回归的方式，与之前基于特征图的方法[7,5]不一样。在[7,5]中，边界框回归是在任意大小的区域上pool之后得到的特征上进行的，回归权重为所有区域大小所共享。在我们的方法中，用于回归的特征是特征图中相同空域大小的(n×n)。为顾及多种大小，学习了k个边界框回归器。每个回归器都对应一种尺度和一种纵横比，k个回归器不共享权重。这样，即使特征固定大小/尺度，也可能预测得到不同大小的边界框。

**Optimization 优化**

The RPN, which is naturally implemented as a fully-convolutional network [14], can be trained end-to-end by back-propagation and stochastic gradient descent (SGD) [12]. We follow the “image-centric” sampling strategy from [5] to train this network. Each mini-batch arises from a single image that contains many positive and negative anchors. It is possible to optimize for the loss functions of all anchors, but this will bias towards negative samples as they are dominate. Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1. If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones.

RPN很自然的以全卷积网络[14]的形式进行实现，可以用反向传播和随机梯度下降(SGD)[12]进行端到端的训练。我们采用[5]中的图像中心取样策略训练这个网络。每个mini-batch都是一幅图像中包含的很多正锚窗和负锚窗组成。对所有锚窗的损失函数进行优化是可能的，但这会偏向负样本，因为负样本数量占优势。所以，我们在一幅图像中随机取样256个锚窗，来计算一个mini-batch的损失函数，其中取样的正锚窗和负锚窗比例为1:1。如果一幅图像中的正样本数量少于128，那么就用负样本来补充这个mini-batch。

We randomly initialize all new layers by drawing weights from a zero-mean Gaussian distribution with standard deviation 0.01. All other layers (i.e., the shared conv layers) are initialized by pre-training a model for ImageNet classification [17], as is standard practice [6]. We tune all layers of the ZF net, and conv3_1 and up for the VGG net to conserve memory [5]. We use a learning rate of 0.001 for 60k mini-batches, and 0.0001 for the next 20k mini-batches on the PASCAL dataset. We also use a momentum of 0.9 and a weight decay of 0.0005 [11]. Our implementation uses Caffe [10].

我们用零均值标准差0.01的高斯分布的随机数初始化所有新层。所有其他层（即共享的卷积层）由ImageNet分类[17]预训练的网络进行初始化，这也是标准操作[6]。我们调整ZF net的所有层，VGG中conv3_1及以上的层以保留内存[5]。在PASCAL VOC数据集上，我们使用的学习率在前60k个mini-batch为0.001，随后20k个mini-batch为0.0001。我们使用的动量为0.9，权重衰减为0.0005[11]。我们使用Caffe[10]进行实现。

**Sharing Convolutional Features for Region Proposal and Object Detection**
**共享卷积特征进行区域候选与目标检测**

Thus far we have described how to train a network for region proposal generation, without considering the region-based object detection CNN that will utilize these proposals. For the detection network, we adopt Fast R-CNN [5] and now describe an algorithm that learns conv layers that are shared between the RPN and Fast R-CNN.

迄今，我们描述了怎样训练候选区域生成网络，而没有考虑使用这些候选的基于区域的目标检测CNN。对于检测网络，我们采用Fast R-CNN[5]，现在我们描述学习RPN和Fast R-CNN共享的卷积层的算法。

Both RPN and Fast R-CNN, trained independently, will modify their conv layers in different ways. We therefore need to develop a technique that allows for sharing conv layers between the two networks, rather than learning two separate networks. Note that this is not as easy as simply defining a single network that includes both RPN and Fast R-CNN, and then optimizing it jointly with back-propagation. The reason is that Fast R-CNN training depends on fixed object proposals and it is not clear a priori if learning Fast R-CNN while simultaneously changing the proposal mechanism will converge. While this joint optimizing is an interesting question for future work, we develop a pragmatic 4-step training algorithm to learn shared features via alternating optimization.

独立训练的RPN和Fast R-CNN，会以不同的方式修改卷积层。因此我们需要一种能在两种网络中共享卷积层的技术，而不是学习两个分离的网络。注意这不是像定义一个包括RPN和Fast R-CNN的网络，然后用反向传播共同训练优化那样简单。原因是Fast R-CNN的训练依赖固定的候选目标，如果学习Fast R-CNN的同时改变候选机制，不知道会不会收敛。这种共同优化的技术留待以后研究，我们提出了实用的4步训练算法，通过交替优化学习共享特征。

In the first step, we train the RPN as described above. This network is initialized with an ImageNet-pre-trained model and fine-tuned end-to-end for the region proposal task. In the second step, we train a separate detection network by Fast R-CNN using the proposals generated by the step-1 RPN. This detection network is also initialized by the ImageNet-pre-trained model. At this point the two networks do not share conv layers. In the third step, we use the detector network to initialize RPN training, but we fix the shared conv layers and only fine-tune the layers unique to RPN. Now the two networks share conv layers. Finally, keeping the shared conv layers fixed, we fine-tune the fc layers of the Fast R-CNN. As such, both networks share the same conv layers and form a unified network.

第一步，我们如上述训练RPN。网络用ImageNet预训练模型初始化，然后端到端的精调进行区域候选任务。第二步，我们使用第一步的RPN生成的候选区域，训练一个分离的Fast R-CNN检测网络。检测网络也由ImaegNet预训练模型初始化。到这时两个网络还没有共享卷积层。第三步，我们用检测网络来初始化RPN训练，但我们固定共享卷积层，只对RPN独有的层进行精调。现在两个网络共享卷积层了。最后，保持共享卷积层固定，我们精调Fast R-CNN的fc层。这样，两个网络共享同样的卷积层，形成了统一的网络。

**Implementation Details 实现细节**

We train and test both region proposal and object detection networks on single-scale images [7, 5]. We re-scale the images such that their shorter side is s = 600 pixels [5]. Multi-scale feature extraction may improve accuracy but does not exhibit a good speed-accuracy trade-off [5]. We also note that for ZF and VGG nets, the total stride on the last conv layer is 16 pixels on the re-scaled image, and thus is ∼10 pixels on a typical PASCAL image (∼500×375). Even such a large stride provides good results, though accuracy may be further improved with a smaller stride.

我们训练、测试区域候选网络和目标检测网络都是在单尺度图像[7,5]上的。我们改变了图像的大小，使其短边长度为s=600像素[5]。多尺度特征提取可能改进准确度，但速度-准确度折中上不是很好[5]。我们还注意到，对于ZF和VGG网络，最后一个卷积层在重新调整了大小的图像上的总步长为16像素，所以在一般的PASCAL图像（约500×375）上就是大约10个像素。即使这样大的步长也会得到好的结果，说明可以用较小的步长进一步改进准确率。

For anchors, we use 3 scales with box areas of $128^2$, $256^2$, and $512^2$ pixels, and 3 aspect ratios of 1:1, 1:2, and 2:1. We note that our algorithm allows the use of anchor boxes that are larger than the underlying receptive field when predicting large proposals. Such predictions are not impossible—one may still roughly infer the extent of an object if only the middle of the object is visible. With this design, our solution does not need multi-scale features or multi-scale sliding windows to predict large regions, saving considerable running time. Fig. 1 (right) shows the capability of our method for a wide range of scales and aspect ratios. The table below shows the learned average proposal size for each anchor using the ZF net (numbers for s = 600).

对于锚窗，我们使用了3种尺度，即128×128，256×256，512×512，和3种纵横比，即1:1，1:2，2:1。我们注意到，当预测大的候选区域时，我们的算法允许锚窗大于潜在的感受野。这种预测是不可能的，如果只有目标的中间是可见的，那只能大致推测目标的程度。这种设计使得我们的方法不需要多尺度特征或多尺度滑窗来预测大型区域，节省了相当的运行时间。图1右所是的是我们的方法可以检测多种尺度和多种纵横比的目标。下表是用ZF网络对每个锚窗学习到的平均候选大小（s=600时的数量）。

anchor | 128,2:1 | 128,1:1 | 128,1:2 | 256,2:1 | 256,1:1 | 256,1:2 | 512,2:1 | 512,1:1 | 512,1:2
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
proposal | 188×111 | 113×114 | 70×92 | 416×229 | 261×284 | 174×332 | 768×437 | 499×501 | 355×715

The anchor boxes that cross image boundaries need to be handled with care. During training, we ignore all cross-boundary anchors so they do not contribute to the loss. For a typical 1000 × 600 image, there will be roughly 20k (≈ 60 × 40 × 9) anchors in total. With the cross-boundary anchors ignored, there are about 6k anchors per image for training. If the boundary-crossing outliers are not ignored in training, they introduce large, difficult to correct error terms in the objective, and training does not converge. During testing, however, we still apply the fully-convolutional RPN to the entire image. This may generate cross-boundary proposal boxes, which we clip to the image boundary.

超出了图像边界的锚窗要小心处理。在训练时，我们忽略了所有的跨边界锚窗，所以对损失函数没有有贡献。对于典型的1000 × 600图像，共计约有20k(≈ 60 × 40 × 9)个锚窗。忽略了跨边界锚窗，每幅图像大约有6k个锚窗可用于训练。如果训练时没有忽略这些跨边界的异常情况，它们将会在目标函数中引入大的难以修正的误差项，训练不会收敛。但在测试时，我们还是对整个图像进行全卷积RPN。这可能会产生跨边界的候选框，我们剪切掉图像边界之外的部分。

Some RPN proposals highly overlap with each other. To reduce redundancy, we adopt non-maximum suppression (NMS) on the proposal regions based on their cls scores. We fix the IoU threshold for NMS at 0.7, which leaves us about 2k proposal regions per image. As we will show, NMS does not harm the ultimate detection accuracy, but substantially reduces the number of proposals. After NMS, we use the top-N ranked proposal regions for detection. In the following, we train Fast R-CNN using 2k RPN proposals, but evaluate different numbers of proposals at test-time.

一些RPN候选互相之间高度重叠。为减少冗余，我们对候选区域进行非最大抑制(NMS)，基于其cls分数。我们固定NMS的IoU阈值为0.7，这样每幅图像中还剩2k候选区域。我们将会看到，NMS不会降低最终的检测准确率，但极大的减少了候选区域数量。NMS后，我们采用top-N排名候选区域进行检测。下面我们使用2k RPN候选训练Fast R-CNN，但在测试时评估不同数量的候选。

## 4 Experiments 实验

We comprehensively evaluate our method on the PASCAL VOC 2007 detection benchmark [4]. This dataset consists of about 5k trainval images and 5k test images over 20 object categories. We also provide results in the PASCAL VOC 2012 benchmark for a few models. For the ImageNet pre-trained network, we use the “fast” version of ZF net [23] that has 5 conv layers and 3 fc layers, and the public VGG-16 model [19] that has 13 conv layers and 3 fc layers. We primarily evaluate detection mean Average Precision (mAP), because this is the actual metric for object detection (rather than focusing on object proposal proxy metrics).

我们在PASCAL VOC 2007检测基准测试[4]中综合评估了我们的方法。这个数据集包括大约5k trainval图像，5k测试图像，共20个类别。我们还给出了在PASCAL VOC 2012基准测试中一些模型的结果。对于ImageNet预训练网络，我们使用了快速版的ZF网络[23]，包括5个卷积层和3个全连接层，公开的VGG-16模型[19]有13个卷积层和3个全连接层。我们主要评估了检测mAP，因为这是目标检测的实际度量标准（而不是聚焦在候选目标的衡量上）。

Table 1 (top) shows Fast R-CNN results when trained and tested using various region proposal methods. These results use the ZF net. For Selective Search (SS) [22], we generate about 2k SS proposals by the “fast” mode. For EdgeBoxes (EB) [24], we generate the proposals by the default EB setting tuned for 0.7 IoU. SS has an mAP of 58.7% and EB has an mAP of 58.6%. RPN with Fast R-CNN achieves competitive results, with an mAP of 59.9% while using up to 300 proposals (For RPN, the number of proposals (e.g., 300) is the maximum number for an image. RPN may produce fewer proposals after NMS, and thus the average number of proposals is smaller). Using RPN yields a much faster detection system than using either SS or EB because of shared conv computations; the fewer proposals also reduce the region-wise fc cost. Next, we consider several ablations of RPN and then show that proposal quality improves when using the very deep network.

表1（上）给出了Fast R-CNN使用不同的区域候选方法进行训练和测试的结果，这些结果使用的是ZF网络。对Selective Search(SS)[22]，我们用快速模式生成2k个SS候选；对于EdgeBoxes(EB)[24]，我们用默认的EB设置即0.7 IoU生成候选。SS得到了58.7% mAP，EB得到了58.6% mAP。RPN和Fast R-CNN取得了很好的结果，使用最多300个候选得到了59.9% mAP。（对RPN，候选数量是一幅图像中的最大数量，如300；RPN在NMS后可能生成没那么多候选，所以候选平均数量会略少一点）。使用RPN比使用SS或EB进行检测速度快很多，原因是共享卷积计算；候选数量更少，也减少了与区域相关的全连接层计算代价。下一步，我们进行一些RPN的分离对比实验，说明使用非常深网络的时候候选质量确实会改善。

Table 1: Detection results on PASCAL VOC 2007 test set (trained on VOC 2007 trainval). The detectors are Fast R-CNN with ZF, but using various proposal methods for training and testing.

**Ablation Experiments**. To investigate the behavior of RPNs as a proposal method, we conducted several ablation studies. First, we show the effect of sharing conv layers between the RPN and Fast R-CNN detection network. To do this, we stop after the second step in the 4-step training process. Using separate networks reduces the result slightly to 58.7% (RPN+ZF, unshared, Table 1). We observe that this is because in the third step when the detector-tuned features are used to fine-tune the RPN, the proposal quality is improved.

**分离对比实验**。为研究RPN作为候选方法的表现，我们进行了几个分离对比实验。首先，我们展示了RPN和Fast R-CNN检测网络共享卷积层的效果。我们在4步训练法中第2步进行完后就停止，以进行对比。使用分离网络使结果略微降低到58.7%（表1中的PRN+ZF，非共享）。我们观察到，这是因为在第3步中，检测器调整的特征用于精调RPN，这改进了候选质量。

Next, we disentangle the RPN’s influence on training the Fast R-CNN detection network. For this purpose, we train a Fast R-CNN model by using the 2k SS proposals and ZF net. We fix this detector and evaluate the detection mAP by changing the proposal regions used at test-time. In these ablation experiments, the RPN does not share features with the detector.

然后，我们理顺RPN在训练Fast R-CNN检测网络中的作用。为达到目的，我们用2k个SS候选和ZF网络训练了一个Fast R-CNN模型。我们固定这个检测器，在测试时改变候选区域来评估检测mAP。在这些分离对比实验中，RPN不和检测器共享特征。

Replacing SS with 300 RPN proposals at test-time leads to an mAP of 56.8%. The loss in mAP is because of the inconsistency between the training/testing proposals. This result serves as the baseline for the following comparisons.

测试时将SS替换为300个RPN候选，得到了56.8%的mAP。这个mAP的下降是因为训练/测试候选区域的不一致性。这个结果作为后续对比的基准。

Somewhat surprisingly, the RPN still leads to a competitive result (55.1%) when using the top-ranked 100 proposals at test-time, indicating that the top-ranked RPN proposals are accurate. On the other extreme, using the top-ranked 6k RPN proposals (without NMS) has a comparable mAP (55.2%), suggesting NMS does not harm the detection mAP and may reduce false alarms.

当在测试时使用排名最高的100个候选，RPN仍然得到了有竞争力的结果(55.1%)，这有些令人惊讶，说明排名最前的RPN候选都是准确的。在另外一个极端情况时，使用排名最前的6k个候选（没有经过NMS）得到了不错的mAP(55.2%)，说明NMS并没有损害mAP，可以减少虚警。

Next, we separately investigate the roles of RPN’s cls and reg outputs by turning off either of them at test-time. When the cls layer is removed at test-time (thus no NMS/ranking is used), we randomly sample N proposals from the unscored regions. The mAP is nearly unchanged with N = 1k (55.8%), but degrades considerably to 44.6% when N = 100. This shows that the cls scores account for the accuracy of the highest ranked proposals.

然后，我们分别研究了RPN的cls和reg输出的作用，方法是在测试时关闭任意一个。当测试时去除cls层时（所以没有使用NMS/排名），我们从未评分的区域中随机取样了N个候选。在N=1k时，mAP基本不变(55.8%)，当N=100时，有很大的下降，到了44.6%。这说明cls对排名最高的那些候选的准确率有很大影响。

On the other hand, when the reg layer is removed at test-time (so the proposals become anchor boxes), the mAP drops to 52.1%. This suggests that the high-quality proposals are mainly due to regressed positions. The anchor boxes alone are not sufficient for accurate detection.

另一方面，当测试时去除reg层时（这样候选成为了锚窗），mAP下降到了52.1%。这说明高质量的候选主要是由于对位置的回归。仅仅是锚窗不足以得到准确的检测。

We also evaluate the effects of more powerful networks on the proposal quality of RPN alone. We use VGG-16 to train the RPN, and still use the above detector of SS+ZF. The mAP improves from 56.8% (using RPN+ZF) to 59.2% (using RPN+VGG). This is a promising result, because it suggests that the proposal quality of RPN+VGG is better than that of RPN+ZF. Because proposals of RPN+ZF are competitive with SS (both are 58.7% when consistently used for training and testing), we may expect RPN+VGG to be better than SS. The following experiments justify this hypothesis.

我们还只用RPN候选评估了更强大的网络的效果。我们使用VGG-16训练RPN，仍然使用SS+ZF的检测器。mAP从56.8%(RPN+ZF)提升到了59.2%(RPN+VGG)。这是个很有希望的结果，因为这说明RPN+VGG的候选质量要高于RPN+ZF。因为RPN+ZF的候选与SS的候选类似（当一致用于训练和测试时，两者都是58.7%），我们期待RPN+VGG要比SS好。后续的实验证实了这个假设。

**Detection Accuracy and Running Time of VGG-16**. Table 2 shows the results of VGG-16 for both proposal and detection. Using RPN+VGG, the Fast R-CNN result is 68.5% for unshared features, slightly higher than the SS baseline. As shown above, this is because the proposals generated by RPN+VGG are more accurate than SS. Unlike SS that is pre-defined, the RPN is actively trained and benefits from better networks. For the feature-shared variant, the result is 69.9%—better than the strong SS baseline, yet with nearly cost-free proposals. We further train the RPN and detection network on the union set of PASCAL VOC 2007 trainval and 2012 trainval, following [5]. The mAP is 73.2%. On the PASCAL VOC 2012 test set (Table 3), our method has an mAP of 70.4% trained on the union set of VOC 2007 trainval+test and VOC 2012 trainval, following [5].

**VGG-16的检测准确率和运行时间**。表2给出了VGG-16进行候选和检测的结果。使用RPN+VGG，Fast R-CNN在未共享特征时得到了68.5%的结果，略高于SS基准。这是因为RPN+VGG生成的候选比SS要更精确。SS是预定义好的，而RPN则是在动态训练的，还从更好的网络中得益。对于共享特征的版本，结果是69.9%，比SS基准要好，但候选计算几乎没有代价。我们进一步在07+12 trainval上训练RPN和检测网络，得到了73.2%的mAP。在PASCAL VOC 2012测试集上（表3），我们的方法得到了70.4% mAP，训练则是在07+12 trainval上，与[5]一样。

Table 2: Detection results on PASCAL VOC 2007 test set. The detector is Fast R-CNN and VGG-16. Training data: “07”: VOC 2007 trainval, “07+12”: union set of VOC 2007 trainval and VOC 2012 trainval. For RPN, the train-time proposals for Fast R-CNN are 2k. † : this was reported in [5]; using the repository provided by this paper, this number is higher (68.0±0.3 in six runs).

Table 3: Detection results on PASCAL VOC 2012 test set. The detector is Fast R-CNN and VGG-16. Training data: “07”: VOC 2007 trainval, “07++12”: union set of VOC 2007 trainval+test and VOC 2012 trainval. For RPN, the train-time proposals for Fast R-CNN are 2k.

In Table 4 we summarize the running time of the entire object detection system. SS takes 1-2 seconds depending on content (on average 1.51s), and Fast R-CNN with VGG-16 takes 320ms on 2k SS proposals (or 223ms if using SVD on fc layers [5]). Our system with VGG-16 takes in total 198ms for both proposal and detection. With the conv features shared, the RPN alone only takes 10ms computing the additional layers. Our region-wise computation is also low, thanks to fewer proposals (300). Our system has a frame-rate of 17 fps with the ZF net.

在表4中，我们总结了整个目标检测系统的运行时间。SS对不同的内容花费1-2秒钟（平均1.51秒），采用VGG16的Fast R-CNN对2k个SS候选处理时间为320ms（如果对fc层使用SVD[5]则为223ms）。我们采用VGG-16的系统候选和检测共计为198ms。在共享卷积层特征的情况下，RPN只耗费10ms计算额外的层。分区域的计算耗费也很低，因为候选区域很少(300)。我们系统使用ZF网络是帧率为17fps。

Table 4: Timing (ms) on a K40 GPU, except SS proposal is evaluated in a CPU. “Region-wise” includes NMS, pooling, fc, and softmax. See our released code for the profiling of running time.

**Analysis of Recall-to-IoU**. Next we compute the recall of proposals at different IoU ratios with ground-truth boxes. It is noteworthy that the Recall-to-IoU metric is just loosely [9, 8, 1] related to the ultimate detection accuracy. It is more appropriate to use this metric to diagnose the proposal method than to evaluate it.

**Recall-to-IoU分析**。下一步，我们计算在不同IoU下候选区域的召回率。值得注意，终极检测结果相关的Recall-to-IoU度量仅是[9, 8, 1]。这种度量更适合于分析候选方法，而不是评估它。

In Fig. 2, we show the results of using 300, 1k, and 2k proposals. We compare with SS and EB, and the N proposals are the top-N ranked ones based on the confidence generated by these methods. The plots show that the RPN method behaves gracefully when the number of proposals drops from 2k to 300. This explains why the RPN has a good ultimate detection mAP when using as few as 300 proposals. As we analyzed before, this property is mainly attributed to the cls term of the RPN. The recall of SS and EB drops more quickly than RPN when the proposals are fewer.

在图2中，我们给出了使用300、1k和2k个候选的结果。我们与SS和EB方法进行了比较，N个候选是根据置信度排前N名的那些。图2说明，RPN方法在候选从2k减少到300时，表现一直很不错。这解释了为什么RPN在只使用了300个候选时还能得到很好的终极检测mAP。我们前面分析过，这种性质主要来自RPN的cls项。当候选减少时，SS和EB的召回率下降很快。

Figure 2: Recall vs. IoU overlap ratio on the PASCAL VOC 2007 test set

**One-Stage Detection vs. Two-Stage Proposal + Detection**. The OverFeat paper [18] proposes a detection method that uses regressors and classifiers on sliding windows over conv feature maps. OverFeat is a one-stage, class-specific detection pipeline, and ours is a two-stage cascade consisting of class-agnostic proposals and class-specific detections. In OverFeat, the region-wise features come from a sliding window of one aspect ratio over a scale pyramid. These features are used to simultaneously determine the location and category of objects. In RPN, the features are from square (3×3) sliding windows and predict proposals relative to anchors with different scales and aspect ratios. Though both methods use sliding windows, the region proposal task is only the first stage of RPN + Fast R-CNN—the detector attends to the proposals to refine them. In the second stage of our cascade, the region-wise features are adaptively pooled [7, 5] from proposal boxes that more faithfully cover the features of the regions. We believe these features lead to more accurate detections.

**单阶段检测与两阶段候选+检测的对比**。OverFeat[18]提出了一种在卷积特征图上的滑窗使用回归器和分类器的检测方法。OverFeat是单阶段指定类别的检测流程，我们的是两阶段级联过程，包括与类别无关的候选，和类别相关的检测。在OverFeat中，区域,分区域的特征来自于一种纵横比几种尺度的滑窗。这些特征用于同时确定目标的位置和类别。在RPN中，特征来自于方形滑窗(3×3)和预测与不同尺度不同纵横比的锚窗关联的候选。虽然两种方法都使用滑窗，区域候选任务只是RPN+Fast R-CNN的第一阶段，检测器会处理候选并进行提炼。在我们级联的第二阶段，分区域特征从候选框中自适应的pool出来[7,5]，更能忠实的反映区域的特征。

To compare the one-stage and two-stage systems, we emulate the OverFeat system (and thus also circumvent other differences of implementation details) by one-stage Fast R-CNN. In this system, the “proposals” are dense sliding windows of 3 scales (128, 256, 512) and 3 aspect ratios (1:1, 1:2, 2:1). Fast R-CNN is trained to predict class-specific scores and regress box locations from these sliding windows. Because the OverFeat system uses an image pyramid, we also evaluate using conv features extracted from 5 scales. We use those 5 scales as in [7, 5].

为比较单阶段和两阶段系统，我们用单阶段Fast R-CNN模拟OverFeat系统（也防止实现细节的其他不同）。在这个系统中，候选是三种尺度(128,256,512)和三种纵横比(1:1,1:2,2:1)的稠密滑窗。Fast R-CNN训练进行预测类别相关的评分，并回归得到框的位置。由于OverFeat系统使用了图像金字塔，我们也使用5个尺度的卷积特征进行评估。我们使用[7,5]中的5个尺度。

Table 5 compares the two-stage system and two variants of the one-stage system. Using the ZF model, the one-stage system has an mAP of 53.9%. This is lower than the two-stage system (58.7%) by 4.8%. This experiment justifies the effectiveness of cascaded region proposals and object detection. Similar observations are reported in [5, 13], where replacing SS region proposals with sliding windows leads to ∼6% degradation in both papers. We also note that the one-stage system is slower as it has considerably more proposals to process.

表5对比了两阶段系统和单阶段系统的两种变体。使用ZF模型，单阶段系统得到了53.9% mAP。这低于两阶段系统58.7%。这个实验验证了区域候选和目标检测级联系统的有效性。[5,13]也给出了类似的观察结果，两篇文章中将SS区域候选法替换为滑窗法得到了大约6%的下降。我们还注意到，单阶段系统更慢，因为要处理的候选多出不少。

Table 5: One-Stage Detection vs. Two-Stage Proposal + Detection. Detection results are on the PASCAL VOC 2007 test set using the ZF model and Fast R-CNN. RPN uses unshared features.

## 5 Conclusion 结论

We have presented Region Proposal Networks (RPNs) for efficient and accurate region proposal generation. By sharing convolutional features with the down-stream detection network, the region proposal step is nearly cost-free. Our method enables a unified, deep-learning-based object detection system to run at 5-17 fps. The learned RPN also improves region proposal quality and thus the overall object detection accuracy.

我们给出了区域候选网络(RPN)来进行高效准确的候选区域生成。通过与检测网络共享卷积特征，区域候选步骤几乎是不耗费计算时间的。我们的方法使得统一的基于深度学习的目标检测系统在5-17fps的速度运行。学习到的RPN也改进了区域候选质量，所以也改进了总体的目标检测准确率。