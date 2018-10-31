# SSD: Single Shot MultiBox Detector

Wei Liu et al. / UNC Chapel Hill

## Abstract 摘要

We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. SSD is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stages and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, COCO, and ILSVRC datasets confirm that SSD has competitive accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. For 300×300 input, SSD achieves 74.3% mAP on VOC2007 test at 59 FPS on a Nvidia Titan X and for 512 × 512 input, SSD achieves 76.9% mAP, outperforming a comparable state-of-the-art Faster R-CNN model (We achieved even better results using an improved data augmentation scheme in follow-on experiments: 77.2% mAP for 300×300 input and 79.8% mAP for 512×512 input on VOC2007. Please see Sec. 3.6 for details). Compared to other single stage methods, SSD has much better accuracy even with a smaller input image size. Code is available at: https://github.com/weiliu89/caffe/tree/ssd.

我们提出一种图像目标检测的方法，使用单个深度神经网络。我们的方法名称为SSD，将边界框的输出空间离散化为不同纵横比的默认框集合，然后衡量每个特征图的位置。在预测时，网络在每个默认框中生成每类目标的存在分数，并对框产生调整以更好的匹配目标形状。另外，网络将从多个不同分辨率的特征图得到的预测结合起来，从而很自然的处理各种大小的目标。SSD与需要进行目标候选的方法比是相对简单的，因为它完全不需要生成候选，以及随后的像素或特征重新排布阶段，将所有运算封装在一个网络中。这使SSD容易训练，可以直接继承进需要检测模块的系统中。在PASCAL VOC，COCO和ILSVRC数据集上的试验结果确认了SSD与其他采用候选目标的方法的相比，可以得到有竞争力的准确率结果，而且速度快很多，还给出了训练和推理的统一框架。对于300×300的输入，SSD在VOC2007上得到74.3%的mAP，检测速度59FPS，使用NVidia Titan X GPU；对于512×512输入，SSD得到了76.9% mAP，超过了目前最好的Faster R-CNN模型（我们在后续试验中，使用了一种数据扩充方案后，得到了更好的结果：对于VOC2007数据集，300×300输入得到了77.2% mAP，512×512输入得到了79.8% mAP，详见3.6节）。与其他单阶段方法相比，即使输入图像尺寸较小，SSD准确率也远高于它们。

**Keywords**: Real-time Object Detection; Convolutional Neural Network

## 1 Introduction 引言

Current state-of-the-art object detection systems are variants of the following approach: hypothesize bounding boxes, resample pixels or features for each box, and apply a high-quality classifier. This pipeline has prevailed on detection benchmarks since the Selective Search work [1] through the current leading results on PASCAL VOC, COCO, and ILSVRC detection all based on Faster R-CNN[2] albeit with deeper features such as [3]. While accurate, these approaches have been too computationally intensive for embedded systems and, even with high-end hardware, too slow for real-time applications. Often detection speed for these approaches is measured in seconds per frame (SPF), and even the fastest high-accuracy detector, Faster R-CNN, operates at only 7 frames per second (FPS). There have been many attempts to build faster detectors by attacking each stage of the detection pipeline (see related work in Sec. 4), but so far, significantly increased speed comes only at the cost of significantly decreased detection accuracy.

目前最好的目标检测系统是下面这些方法的变体：假设边界框，对每个框进行像素重取样，或特征重取样，然后送入高质量的分类器。这个流程自从selective search[1]提出以来，一直占据检测标准，在PASCAL VOC、COCO、ILSVRC检测都取得了领先的结果，它们都是基于Faster R-CNN[2]的，有的采用了[3]这样的深度特征。虽然非常精确，但这些方法计算量非常大，不适用于嵌入式系统，即使用高端硬件，对于实时应用来说也太慢。通常这些方法的检测速度都是用SPF(seconds per frame)来衡量，即使是最快的高精度检测器，Faster R-CNN，也只有7FPS(frames per second)。有很多尝试要构建更快的检测器，试图改进检测流程的每个阶段（见第4节中的相关工作），但是迄今为止，速度的明显提升代价是检测准确率的明显下降。

This paper presents the first deep network based object detector that does not re-sample pixels or features for bounding box hypotheses and and is as accurate as approaches that do. This results in a significant improvement in speed for high-accuracy detection (59 FPS with mAP 74.3% on VOC2007 test, vs. Faster R-CNN 7 FPS with mAP 73.2% or YOLO 45 FPS with mAP 63.4%). The fundamental improvement in speed comes from eliminating bounding box proposals and the subsequent pixel or feature resampling stage. We are not the first to do this (cf [4,5]), but by adding a series of improvements, we manage to increase the accuracy significantly over previous attempts. Our improvements include using a small convolutional filter to predict object categories and offsets in bounding box locations, using separate predictors (filters) for different aspect ratio detections, and applying these filters to multiple feature maps from the later stages of a network in order to perform detection at multiple scales. With these modifications—especially using multiple layers for prediction at different scales—we can achieve high-accuracy using relatively low resolution input, further increasing detection speed. While these contributions may seem small independently, we note that the resulting system improves accuracy on real-time detection for PASCAL VOC from 63.4% mAP for YOLO to 74.3% mAP for our SSD. This is a larger relative improvement in detection accuracy than that from the recent, very high-profile work on residual networks [3]. Furthermore, significantly improving the speed of high-quality detection can broaden the range of settings where computer vision is useful.

本文提出了第一个基于深度网络的目标检测器，并不对假设的边界框进行重采样像素或特征，且与这样的方法一样精确。这个结果明显改进了高精确度检测方法的速度（在VOC2007测试集上，以59FPS的速度得到了74.3%的结果，对比Faster R-CNN速度7 FPS 73.2% mAP，或YOLO 45 FPS 63.4% mAP）。速度的根本性提升来自于取消边界框候选和后续的像素或特征重采样阶段。我们不是第一个这样做的（参考[4,5]），但通过增加了一系列改进，我们最终明显改进了以前这些尝试的准确率。我们的改进包括，使用小卷积核来预测目标类别和边界框位置的偏移，对于不同的纵横比的检测使用不同的预测器（滤波器），将这些滤波器应用于网络后期阶段得到的不同特征图来在不同尺度上进行检测。通过这些修正，尤其是使用了在不同尺度上使用多层预测，我们可以得到高检测准确率使用相对较低分辨率的输入，进一步增加了检测速度。虽然这些贡献各自看起来都很小，我们注意到得到的系统在PASCAL VOC上将实时检测的mAP从YOLO的63.4%提升到我们SSD的74.3%。这比最近很高调的残差网络[3]在检测准确率上有了更大的改进。而且，高质量检测的明显速度提升可以扩大到计算机视觉有用的范围设定中。

We summarize our contributions as follows: 我们总结贡献如下：

- We introduce SSD, a single-shot detector for multiple categories that is faster than the previous state-of-the-art for single shot detectors (YOLO), and significantly more accurate, in fact as accurate as slower techniques that perform explicit region proposals and pooling (including Faster R-CNN).
- 我们提出了SSD，一个多类别单发检测器，比之前最好的单发检测器(YOLO)还要快，而且显著提升准确率，实际上与那些更慢的技术一样准确，如Faster R-CNN这样的先进行显式的区域候选然后pooling。
- The core of SSD is predicting category scores and box offsets for a fixed set of default bounding boxes using small convolutional filters applied to feature maps.
- SSD的核心是对固定的默认边界框预测类别分数和框偏移，方法是将小卷积核滤波器应用于特征图。
- To achieve high detection accuracy we produce predictions of different scales from feature maps of different scales, and explicitly separate predictions by aspect ratio.
- 为得到高检测准确率，我们在特征图的不同尺度上产生预测，显式的根据纵横比分离预测。
- These design features lead to simple end-to-end training and high accuracy, even on low resolution input images, further improving the speed vs accuracy trade-off.
- 这些设计特征带来的是简单的端到端的训练和高准确率，即使在低分辨率输入上也是，进一步改进了速度与准确率的折中。
- Experiments include timing and accuracy analysis on models with varying input size evaluated on PASCAL VOC, COCO, and ILSVRC and are compared to a range of recent state-of-the-art approaches.
- 在PASCAL VOC、COCO和ILSVRC上进行了不同输入大小的试验，试验包括计时与准确率分析，并与一系列目前最好的方法进行了比较。

## 2 The Single Shot Detector (SSD)

This section describes our proposed SSD framework for detection (Sec. 2.1) and the associated training methodology (Sec. 2.2). Afterwards, Sec. 3 presents dataset-specific model details and experimental results.

本节叙述的是我们提出的SSD检测框架（2.1节），和相关的训练方法（2.2节）。然后第3部分给出了与特定数据集相关的模型细节和试验结果。

### 2.1 Model 模型

The SSD approach is based on a feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification (truncated before any classification layers), which we will call the base network (We use the VGG-16 network as a base, but other networks should also produce good results). We then add auxiliary structure to the network to produce detections with the following key features:

SSD方法是基于前馈卷积网络的，它计算得到含有固定数量的边界框的集合，然后在这些框中为目标类别实例的存在打分，然后进行非最大抑制，得到最终检测结果。网络前面的层是基于高质量图像分类的标准框架（任何分类层之前的截取），我们称之为基础网络（我们使用VGG16作为基础，但是其他网络也可以得到很好的结果）。我们然后加入辅助结构来产生检测，检测具有以下关键特征：

**Multi-scale feature maps for detection**. We add convolutional feature layers to the end of the truncated base network. These layers decrease in size progressively and allow predictions of detections at multiple scales. The convolutional model for predicting detections is different for each feature layer (cf Overfeat[4] and YOLO[5] that operate on a single scale feature map).

**用于检测的多尺度特征图**。我们在截取的基础网络上增加卷积特征层。这些层尺寸逐渐减小，允许在多个尺度上进行检测预测。用于预测检测的卷积模型对于每个特征层都不一样（参考OverFeat[4]和YOLO[5]，它们都在单尺度特征图上进行操作）。

**Convolutional predictors for detection**. Each added feature layer (or optionally an existing feature layer from the base network) can produce a fixed set of detection predictions using a set of convolutional filters. These are indicated on top of the SSD network architecture in Fig. 2. For a feature layer of size m × n with p channels, the basic element for predicting parameters of a potential detection is a 3 × 3 × p small kernel that produces either a score for a category, or a shape offset relative to the default box coordinates. At each of the m × n locations where the kernel is applied, it produces an output value. The bounding box offset output values are measured relative to a default box position relative to each feature map location (cf the architecture of YOLO[5] that uses an intermediate fully connected layer instead of a convolutional filter for this step).

**用于检测的卷积预测器**。每个增加的特征层（或基础网络中已经存在的特征层）都能用卷积滤波器集生成固定数量的检测预测集。图2中SSD网络架构的顶部就是这些层。对于一个p通道大小m×n的特征层，潜在检测的预测参数的基础元素是一个3×3×p的小卷积核，产生的或是一个类别的评分，或相对于默认框坐标的形状偏移。卷积核作用于所有的m×n位置，得到一个输出值。边界框偏移输出值是相对于一个默认的边界框位置的，而这个默认框是相对于每个特征图位置的（参考YOLO[5]的框架中这一步使用了中间全连接层，而不是使用的卷积滤波器）。

**Default boxes and aspect ratios**. We associate a set of default bounding boxes with each feature map cell, for multiple feature maps at the top of the network. The default boxes tile the feature map in a convolutional manner, so that the position of each box relative to its corresponding cell is fixed. At each feature map cell, we predict the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. Specifically, for each box out of k at a given location, we compute c class scores and the 4 offsets relative to the original default box shape. This results in a total of (c + 4)k filters that are applied around each location in the feature map, yielding (c + 4)kmn outputs for a m × n feature map. For an illustration of default boxes, please refer to Fig. 1. Our default boxes are similar to the anchor boxes used in Faster R-CNN [2], however we apply them to several feature maps of different resolutions. Allowing different default box shapes in several feature maps let us efficiently discretize the space of possible output box shapes.

**默认边界框和纵横比**。我们将默认边界框集合与每个特征图单元关联起来，在网络上层有多个特征图。默认框在特征图上以卷积的方式摆放，每个框的位置相对于对应的单元是固定的。在每个特征图单元中，我们预测单元中相对于默认框的偏移，以及每类的评分，这个评分表明再每个框中一个类别实例的存在。特别的，对于给定位置上k个框中的每个来说，我们计算c类评分和相对于原始默认框形状的4个偏移量。这需要总共(c+4)k个滤波器，每个滤波器都作用于特征图的每个点，对于m×n大小的特征产生(c + 4)kmn个输出。参考图1的默认框描述。我们的默认框与Faster R-CNN[2]中使用的锚框类似，但是我们将其应用于多个分辨率的多个特征图上。多个特征图中的多个默认框，使我们可以高效的将可能输出框的形状空间离散化。

Fig.1: SSD framework. (a) SSD only needs an input image and ground truth boxes for each object during training. In a convolutional fashion, we evaluate a small set (e.g. 4) of default boxes of different aspect ratios at each location in several feature maps with different scales (e.g. 8 × 8 and 4 × 4 in (b) and (c)). For each default box, we predict both the shape offsets and the confidences for all object categories (($c_1 ,c_2 ,··· ,c_p$)). At training time, we first match these default boxes to the ground truth boxes. For example, we have matched two default boxes with the cat and one with the dog, which are treated as positives and the rest as negatives. The model loss is a weighted sum between localization loss (e.g. Smooth L1 [6]) and confidence loss (e.g. Softmax).

图1：SSD框架。(a)在训练过程中，对于每个目标，SSD只需要一个输入图像和ground-truth框。我们以卷积的样式来评估默认框的小集合（如4个），默认框是不同尺度下几个特征图中每个位置上的，拥有不同的纵横比（如b中的8×8，和c中的4×4）。对每个默认框来说，我们预测形状偏移和所有目标类别的置信度($c_1 ,c_2 ,··· ,c_p$)。在训练时，我们首先将这些默认框与ground-truth边界框匹配。比如我们对猫匹配了两个默认框，对狗匹配了一个，这些认为是正样本，剩下的为负样本。模型损失函数为定位损失函数（如平滑L1[6]）和置信度损失函数（如softmax）的加权和。

Fig.2: A comparison between two single shot detection models: SSD and YOLO [5]. Our SSD model adds several feature layers to the end of a base network, which predict the offsets to default boxes of different scales and aspect ratios and their associated confidences. SSD with a 300 × 300 input size significantly outperforms its 448 × 448 YOLO counterpart in accuracy on VOC2007 test while also improving the speed.

图2 两种单发检测模型的对比：SSD与YOLO[5]。我们的SSD模型在基础网络的后面增加了几个特征层，以预测默认框的偏移，默认框是多尺度的、多纵横比的，还预测与默认框关联的置信度。SSD输入为300×300像素，其准确度明显超过了对应的YOLO算法，其输入为448×448，两个算法都在VOC2007测试集上评估，SSD还在运算速度上有改进。

### 2.2 Training 训练

The key difference between training SSD and training a typical detector that uses region proposals, is that ground truth information needs to be assigned to specific outputs in the fixed set of detector outputs. Some version of this is also required for training in YOLO[5] and for the region proposal stage of Faster R-CNN[2] and MultiBox[7]. Once this assignment is determined, the loss function and back propagation are applied end-to-end. Training also involves choosing the set of default boxes and scales for detection as well as the hard negative mining and data augmentation strategies.

训练SSD与训练其他使用候选区域的典型预测器的关键差别在于，ground-truth信息需要指定给特定输出，这个输出是固定检测器输出集中的。YOLO[5]中的训练也有一部分类似的要求，还有Faster R-CNN[2]中的候选区域阶段和MultiBox[7]也有类似的要求。一旦确定了这种指定，损失函数和反向传播都是端到端的。训练还要选择默认框集合和检测尺度，还有难分负样本挖掘，和数据扩充策略。

**Matching strategy**. During training we need to determine which default boxes correspond to a ground truth detection and train the network accordingly. For each ground truth box we are selecting from default boxes that vary over location, aspect ratio, and scale. We begin by matching each ground truth box to the default box with the best jaccard overlap (as in MultiBox [7]). Unlike MultiBox, we then match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5). This simplifies the learning problem, allowing the network to predict high scores for multiple overlapping default boxes rather than requiring it to pick only the one with maximum overlap.

**匹配策略**。在训练时，我们需要确定ground-truth检测对应哪些默认框，对应的训练网络。对于每个ground-truth框，我们选择不同位置、不同纵横比和尺度的默认框。我们将每个ground-truth框匹配具有最佳重叠度的默认框，以此开始（和MultiBox[7]中一样）。但与MultiBox不同的是，我们然后将默认框与任何重叠度大于阈值0.5的ground-truth框进行匹配。这简化了学习问题，使网络可以对多个重叠的默认框预测得到高分，而不是要求它只能选择一个最大重叠的。

**Training objective**. The SSD training objective is derived from the MultiBox objective [7,8] but is extended to handle multiple object categories. Let $x^p_{ij}= {1,0}$ be an indicator for matching the i-th default box to the j-th ground truth box of category p. In the matching strategy above, we can have $\sum_i x^p_{ij}≥ 1$. The overall objective loss function is a weighted sum of the localization loss (loc) and the confidence loss (conf):

**训练目标函数**。SSD训练目标函数是从MultiBox[7,8]的目标函数推导出来的，但扩展到处理多个目标类别。令$x^p_{ij}= {1,0}$为第i个默认框与第j个类别p的ground-truth框匹配的指示值。在上面的匹配策略中，我们有$\sum_i x^p_{ij}≥ 1$。总体的目标损失函数为定位损失(loc)与置信度损失(conf)的加权和：

$$L(x,c,l,g) =(L_{conf} (x,c) + αL_{loc} (x,l,g))/N$$(1)

where N is the number of matched default boxes. If N = 0, we set the loss to 0. The localization loss is a Smooth L1 loss [6] between the predicted box (l) and the ground truth box (g) parameters. Similar to Faster R-CNN [2], we regress to offsets for the center (cx,cy) of the default bounding box (d) and for its width (w) and height (h).

其中N是匹配到的默认框的数目。如果N=0，我们设置损失为0。定位损失函数为预测框(l)和和ground-truth框(g)的平滑L1损失[6]。与Faster R-CNN[2]类似，我们回归得到默认边界框(d)的中心(cx,cy)和它的宽度(w)和高度(h)。

$$L_{loc}(x,l,g) = \sum_{i∈pos} \sum_{m∈(cx,cy,w,h)} x^k_{ij} smooth_{L1} (l_i^m - \hat g_j^m)$$(2)
$$\hat g_j^{cx} = (g_j^{cx} - d_i^{cx})/d_i^w$$(2)
$$\hat g_j^{cy} = (g_j^{cy} - d_i^{cy})/d_i^h$$(2)
$$\hat g_j^w = log(g_j^w/d_i^w)$$(2)
$$\hat g_j^h = log(g_j^h/d_i^h)$$(2)

The confidence loss is the softmax loss over multiple classes confidences (c). 置信度损失是在多个类别的置信度上的softmax损失。

$$L_{conf}(x,c)=-sum_{i∈pos}^N x_{ij}^p log(\hat c_i^p) - \sum_{i∈neg} log(\hat c_i^0)$$(3)

where 其中 $\hat c_i^p = \frac {exp(c_i^p)}{\sum_p exp(c_i^p)}$

and the weight term α is set to 1 by cross validation. 通过交叉验证，将权重α设置为1。

**Choosing scales and aspect ratios for default boxes**. To handle different object scales, some methods [4,9] suggest processing the image at different sizes and combining the results afterwards. However, by utilizing feature maps from several different layers in a single network for prediction we can mimic the same effect, while also sharing parameters across all object scales. Previous works [10,11] have shown that using feature maps from the lower layers can improve semantic segmentation quality because the lower layers capture more fine details of the input objects. Similarly, [12] showed that adding global context pooled from a feature map can help smooth the segmentation results. Motivated by these methods, we use both the lower and upper feature maps for detection. Figure 1 shows two exemplar feature maps (8×8 and 4×4) which are used in the framework. In practice, we can use many more with small computational overhead.

**选择默认框的尺度和纵横比**。为处理不同的目标尺度，一些方法[4,9]建议在不同大小上处理图像，然后将结果合并。但是，通过在单个网络中利用几个不同层的特征图来预测我们可以模仿这种效果，同时还在所有不同目标尺度间分享参数。之前的工作[10,11]已经显示，使用较低的层特征图可以改进语义分割质量，因为较低的层捕获更多的是输入目标的细节。类似的，[12]显示，增加从特征图pool得到的全局上下文可以帮助平滑分割结果。受这些方法激发，我们既使用低层特征图，也使用高层特征图进行检测。图1展示的是在框架中使用的两个特征图例子（8×8和4×4）。在实践中，我们使用更多计算量很小的特征图。

Feature maps from different levels within a network are known to have different (empirical) receptive field sizes [13]. Fortunately, within the SSD framework, the default boxes do not necessary need to correspond to the actual receptive fields of each layer. We design the tiling of default boxes so that specific feature maps learn to be responsive to particular scales of the objects. Suppose we want to use m feature maps for prediction. The scale of the default boxes for each feature map is computed as:

网络中不同层的特征图有着不同的（经验）感受野大小[13]。幸运的是，在SSD框架中，默认框不需要与每层的实际感受野相对应。我们设计的默认框的摆放，其效果是特定特征图学习对特定尺度的目标有响应。假设我们想使用m个特征图进行预测，对每个特征图，默认框的尺度计算如下：

$$s_k = s_{min} + \frac {s_{max} − s_{min}} {m − 1} (k − 1), k ∈ [1,m]$$(4)

where $s_{min}$ is 0.2 and $s_{max}$ is 0.9, meaning the lowest layer has a scale of 0.2 and the highest layer has a scale of 0.9, and all layers in between are regularly spaced. We impose different aspect ratios for the default boxes, and denote them as $a_r ∈{1,2,3,1/2,1/3}$. We can compute the width ($w^a_k= s_k \sqrt a_r$) and height ($h^a_k= s_k /\sqrt a_r$) for each default box. For the aspect ratio of 1, we also add a default box whose scale is $s'_k= \sqrt {s_k s_{k+1}}$, resulting in 6 default boxes per feature map location. We set the center of each default box to ($\frac {i+0.5}{|f_k|}, \frac {j+0.5} {|f_k|}$), where $|f_k|$ is the size of the k-th square feature map, $i,j ∈ [0,|f_k|)$. In practice, one can also design a distribution of default boxes to best fit a specific dataset. How to design the optimal tiling is an open question as well.

这里$s_{min}$为0.2，$s_{max}$为0.9，意味着最低层的尺度为0.2，最高层的尺度为0.9，所有中间层的尺度均匀间隔。我们为默认框加上不同的纵横比，表示为$a_r ∈{1,2,3,1/2,1/3}$。我们现在对每个默认框计算宽度($w^a_k= s_k \sqrt a_r$)和高度($h^a_k= s_k /\sqrt a_r$)。当纵横比为1时，我们还增加一个默认框尺度为$s'_k= \sqrt {s_k s_{k+1}}$，这样在每个特征图位置上得到6个默认框尺度。我们将默认框的中心设为($\frac {i+0.5}{|f_k|}, \frac {j+0.5} {|f_k|}$)，其中$|f_k|$是第k个方形特征图的大小，$i,j ∈ [0,|f_k|)$。在实践中，还可以设计默认框的分布来适应特定的数据集。怎样设计最佳摆放仍然是一个开放的问题。

By combining predictions for all default boxes with different scales and aspect ratios from all locations of many feature maps, we have a diverse set of predictions, covering various input object sizes and shapes. For example, in Fig. 1, the dog is matched to a default box in the 4 × 4 feature map, but not to any default boxes in the 8 × 8 feature map. This is because those boxes have different scales and do not match the dog box, and therefore are considered as negatives during training.

通过结合很多特征图中所有位置上的不同尺度不同纵横比的所有默认框预测，我们得到了多种预测集，覆盖了不同的输入目标尺寸和形状。例如在图1中，狗在4×4特征图中匹配到了一个默认框，但在8×8特征图中没有匹配到默认框。这是因为这些框都不同的尺度，不能匹配到狗框，所以在训练时被认为是负样本。

**Hard negative mining**. After the matching step, most of the default boxes are negatives, especially when the number of possible default boxes is large. This introduces a significant imbalance between the positive and negative training examples. Instead of using all the negative examples, we sort them using the highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1. We found that this leads to faster optimization and a more stable training.

**难分负样本挖掘**。在匹配步骤后，大多数默认框都是负样本，尤其是可能的默认框数目巨大的时候。这带来了训练正样本和负样本的明显的不平衡。我们没有使用全部的负样本，而是根据每个默认框的最高置信度损失将其排序，挑选最高的那些，这样达到负样本与正样本的比例为最多3:1。我们发现这带来了更快的优化和更稳定的训练。

**Data augmentation**. To make the model more robust to various input object sizes and shapes, each training image is randomly sampled by one of the following options:

**数据扩充**。为使模型对各种输入目标大小和形状更加稳健，每个训练图像都根据下面的选项进行随机采样：

- Use the entire original input image. 使用整个原始输入图像
- Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3, 0.5, 0.7, or 0.9. 取其中一个图像块，使与目标的最小重叠度为0.1, 0.3, 0.5, 0.7, or 0.9。
- Randomly sample a patch. 随机取一图像块

The size of each sampled patch is [0.1, 1] of the original image size, and the aspect ratio is between 1/2 and 2. We keep the overlapped part of the ground truth box if the center of it is in the sampled patch. After the aforementioned sampling step, each sampled patch is resized to fixed size and is horizontally flipped with probability of 0.5, in addition to applying some photo-metric distortions similar to those described in [14].

每个取样的图像块的大小为原始图像大小的[0.1,1]，纵横比在1/2和2之间。如果ground-truth框的中心在取样块内，那么我们就保持重叠部分。在前述的取样步骤后，每个取样的块都将大小改为固定尺寸，并以0.5的概率进行水平翻转，另外还应用一些与[14]类似的光学变形处理。

## 3 Experimental Results 试验结果

**Base network**. Our experiments are all based on VGG16 [15], which is pre-trained on the ILSVRC CLS-LOC dataset [16]. Similar to DeepLab-LargeFOV [17], we convert fc6 and fc7 to convolutional layers, subsample parameters from fc6 and fc7, change pool5 from 2 × 2 − s2 to 3 × 3 − s1, and use the $à trous$ algorithm [18] to fill the ”holes”. We remove all the dropout layers and the fc8 layer. We fine-tune the resulting model using SGD with initial learning rate 0.001, 0.9 momentum, 0.0005 weight decay, and batch size 32. The learning rate decay policy is slightly different for each dataset, and we will describe details later. The full training and testing code is built on Caffe [19] and is open source at: https://github.com/weiliu89/caffe/tree/ssd.

**基础网络**。我们的试验都是基于VGG16[15]的，并在ILSVRC CLS-LOC[16]数据集上进行了预训练。与DeepLab-LargeFOV[17]类似，我们将fc6和fc7转换为卷积层，从fc6和fc7中对参数进行降采样，将pool5层从2×2 -s2变换为3×3 -s1，使用$à trous$算法[18]来填充孔洞。我们去除了所有dropout层和fc8层。我们对得到的模型进行了精调，使用SGD算法，初始学习速率为0.001，动量0.9，权值衰减0.005，批次规模32,。学习速率衰减策略在每个数据集上略微不一样，我们将会在后面详述。全部的训练代码和测试代码都是基于Caffe[19]构建的。

### 3.1 PASCAL VOC2007

On this dataset, we compare against Fast R-CNN [6] and Faster R-CNN [2] on VOC2007 test (4952 images). All methods fine-tune on the same pre-trained VGG16 network.

在这个数据集上，我们与Fast R-CNN[6]和Faster R-CNN[2]在VOC2007 test集上进行比较(4952图像)。所有方法都在一样的预训练VGG16网络上精调。

Figure 2 shows the architecture details of the SSD300 model. We use conv4_3, conv7 (fc7), conv8_2, conv9_2, conv10_2, and conv11_2 to predict both location and confidences. We set default box with scale 0.1 on conv4_3(For SSD512 model, we add extra conv12_2 for prediction, set $s_min$ to 0.15, and 0.07 on conv4_3). We initialize the parameters for all the newly added convolutional layers with the "xavier" method[20]. For conv4_3, conv10_2 and conv11_2, we only associate 4 default boxes at each feature map location – omitting aspect ratios of 1/3 and 3. For all other layers, we put 6 default boxes as described in Sec. 2.2. Since, as pointed out in [12], conv4_3 has a different feature scale compared to the other layers, we use the L2 normalization technique introduced in [12] to scale the feature norm at each location in the feature map to 20 and learn the scale during back propagation. We use the 0.001 learning rate for 40k iterations, then continue training for 10k iterations with 0.0001 and 0.00001. When training on VOC2007 trainval, Table 1 shows that our low resolution SSD300 model is already more accurate than Fast R-CNN. When we train SSD on a larger 512×512 input image, it is even more accurate, surpassing Faster R-CNN by 1.7% mAP. If we train SSD with more (i.e. 07+12) data, we see that SSD300 is already better than Faster R-CNN by 1.1% and that SSD512 is 3.6% better. If we take models trained on COCO trainval35k as described in Sec. 3.4 and fine-tuning them on the 07+12 dataset with SSD512, we achieve the best results: 81.6% mAP.

图2所示的是SSD300模型的架构细节。我们使用conv4_3, conv7 (fc7), conv8_2, conv9_2, conv10_2和conv11_2来预测位置和置信度。我们设置conv4_3层上默认框尺度0.1（对于SSD512模型，我们增加了conv12_2进行预测，设$s_min$为0.15，conv4_3层的尺度为0.07）。我们用"xavier"方法初始化参数所有新加入的卷积层[20]。对于conv4_3, conv10_2 and conv11_2，我们在每个特征图的每个位置关联4个默认框 - 忽略纵横比为1/3和3的框。对于其他层，我们关联6个默认框，就像2.2节的一样。就像[12]中所指出的那样，与其他层相比，conv4_3层有不同的特征尺度，我们使用[12]中提出的L2归一化技术来将特征图中每个位置的特征范数缩放到20，在反向传播的过程中学习这个大小。前40k次迭代使用学习速率0.001，然后继续训练10k次迭代，学习速率0.0001，然后10k次0.00001。当在VOC2007 trainval上训练时，表1所示的是我们的低分辨率SSD300模型已经比Fast R-CNN更加准确了。当我们用更大的512×512输入图像训练SSD时，甚至更加准确，超过了Faster R-CNN 1.7% mAP。如果我们使用更多数据（如07+12）训练SSD时，我们看到SSD300已经比Faster R-CNN准确度高了1.1%，SSD512高了3.6%。如果我们使用在COCO trainval35k数据集上训练的模型，像3.4节所述的，然后在07+12数据集上用SSD512精调，那么得到了最好结果：81.6% mAP。

Table 1: PASCAL VOC2007 test detection results. Both Fast and Faster R-CNN use input images whose minimum dimension is 600. The two SSD models have exactly the same settings except that they have different input sizes (300×300 vs. 512×512). It is obvious that larger input size leads to better results, and more data always helps. Data: ”07”: VOC2007 trainval, ”07+12”: union of VOC2007 and VOC2012 trainval. ”07+12+COCO”: first train on COCO trainval35k then fine-tune on 07+12.

Method | data | mAP | person plant sheep sofa train tv et al. 20 classes
--- | --- | --- | --- 
Fast [6] | 07 | 66.9 | 69.0 30.1 65.4 70.2 75.8 65.8
Fast [6] | 07+12 | 70.0 | 69.9 31.8 70.1 74.8 80.4 70.4
Faster [2] | 07 | 69.9 | 76.3 39.1 68.3 67.3 81.1 67.6
Faster [2] | 07+12 | 73.2 | 76.7 38.8 73.6 73.9 83.0 72.6
Faster [2] | 07+12+COCO | 78.8 | 82.3 53.6 80.4 75.8 86.6 78.9
SSD300 | 07 | 68.0 | 72.5 41.2 64.2 69.1 78.0 68.5
SSD300 | 07+12 | 74.3 | 76.2 48.6 73.9 76.0 83.4 74.0
SSD300 | 07+12+COCO | 79.6 | 81.4 55.0 81.9 81.5 85.9 78.9
SSD512 | 07 | 71.6 | 76.6 44.9 69.9 69.1 78.1 71.8
SSD512 | 07+12 | 76.8 | 79.7 50.3 77.9 73.9 82.5 75.3
SSD512 | 07+12+COCO | 81.6 | 84.6 59.1 85.0 80.4 87.4 81.2

To understand the performance of our two SSD models in more details, we used the detection analysis tool from [21]. Figure 3 shows that SSD can detect various object categories with high quality (large white area). The majority of its confident detections are correct. The recall is around 85-90%, and is much higher with “weak” (0.1 jaccard overlap) criteria. Compared to R-CNN [22], SSD has less localization error, indicating that SSD can localize objects better because it directly learns to regress the object shape and classify object categories instead of using two decoupled steps. However, SSD has more confusions with similar object categories (especially for animals), partly because we share locations for multiple categories. Figure 4 shows that SSD is very sensitive to the bounding box size. In other words, it has much worse performance on smaller objects than bigger objects. This is not surprising because those small objects may not even have any information at the very top layers. Increasing the input size (e.g. from 300×300 to 512×512) can help improve detecting small objects, but there is still a lot of room to improve. On the positive side, we can clearly see that SSD performs really well on large objects. And it is very robust to different object aspect ratios because we use default boxes of various aspect ratios per feature map location.

为更详细的理解两个SSD模型的表现，我们使用[21]中的检测分析工具。图3所示的是SSD可以高质量的检测多种目标（大量白色区域）。大部分置信度很高的检测都是正确的。回召率大约为85%到90%，远高于弱准则（0.1重叠度）。与R-CNN[22]相比，SSD的定位错误少，说明SSD可以更好的定位目标，因为直接学习回归得到目标形状并归类目标类别，而不是使用两个分离的步骤。但是，SSD与类似目标类别混淆的较多（尤其对于动物类别来说），部分可能是因为为多个类别共享位置。图4所示的是SSD对边界框的大小非常敏感。换句话说，在小目标上的表现比大目标差的多。这不令人惊讶，因为这些小目标可能在很高层中基本不会有什么信息。增加输入大小，如300×300到512×512，对小目标检测有帮助，但仍然还有很大的改进空间。从好的方面来说，我们可以明显看到SSD在大目标上表现非常好。而且也对不同纵横比的目标也很鲁棒，因为我们在特征图每个位置上都使用了不同纵横比的默认框。

Fig.3: Visualization of performance for SSD512 on animals, vehicles, and furniture from VOC2007 test. The top row shows the cumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories (Sim), with others (Oth), or with background (BG). The solid red line reflects the change of recall with strong criteria (0.5 jaccard overlap) as the number of detections increases. The dashed red line is using the weak criteria (0.1 jaccard overlap). The bottom row shows the distribution of top-ranked false positive types.

图3 SSD512对VOC2007测试集中动物、交通工具和家具类别的可视化表现。上一行展示的是正确的检测累积部分(Cor)或false positive中定位错误的(Loc)，与相似类别混淆的(Sim)，与其他类别混淆的(Oth)，或与背景混淆(BG)的。红色实线反应的是强规则（重叠率0.5）的情况下随着检测数目的增加，回召率的变化。红色虚线是使用弱规则（0.1重叠率）的变化。下一行展示的是排名最高的false positive类型的分布。

Fig.4: Sensitivity and impact of different object characteristics on VOC2007 test set using [21]. The plot on the left shows the effects of BBox Area per category, and the right plot shows the effect of Aspect Ratio. Key: BBox Area: XS=extra-small; S=small; M=medium; L=large; XL =extra-large. Aspect Ratio: XT=extra-tall/narrow; T=tall; M=medium; W=wide; XW =extra-wide.

图4 使用[21]绘制的VOC207测试集上不同目标特性的敏感度和影响。左边的图展示的是每个类别的边界框大小的影响，右边展示的是纵横比的影响。边界框：XS=非常小 S=小 M=中 L=大 XL=非常大；纵横比：XT=非常高（窄） T=高 M=中等 W=宽 XW=非常宽。

### 3.2 Model analysis 模型分析

To understand SSD better, we carried out controlled experiments to examine how each component affects performance. For all the experiments, we use the same settings and input size (300 × 300), except for specified changes to the settings or component(s).

为更好的理解SSD，我们进行了一些受控试验，来检查每个组件是怎样影响表现的。对于所有的试验，我们都使用同样的设置，输入都是300 × 300，除了另外指出的一些设置或部件改变。

**Data augmentation is crucial**. Fast and Faster R-CNN use the original image and the horizontal flip to train. We use a more extensive sampling strategy, similar to YOLO [5]. Table 2 shows that we can improve 8.8% mAP with this sampling strategy. We do not know how much our sampling strategy will benefit Fast and Faster R-CNN, but they are likely to benefit less because they use a feature pooling step during classification that is relatively robust to object translation by design.

**数据扩充是重要的**。Fast和Faster R-CNN使用原始图像和水平翻转图像来进行训练。我们使用了更广泛的取样策略，与YOLO[5]类似。表2所示的是，我们可以用这种取样策略提升8.8%的mAP。我们不知道我们的取样策略会在多大程度上使Fast和Faster R-CNN受益，但很可能受益较少，因为他们在分类阶段使用了特征pooling步骤，根据设计，这是对目标平移是相对鲁棒的。

**More default box shapes is better**. As described in Sec. 2.2, by default we use 6 default boxes per location. If we remove the boxes with 1/3 and 3 aspect ratios, the performance drops by 0.6%. By further removing the boxes with 1/2 and 2 aspect ratios, the performance drops another 2.1%. Using a variety of default box shapes seems to make the task of predicting boxes easier for the network.

**更多的默认边界框形状效果更好**。如2.2节所叙述的，默认我们每个位置使用6个默认框。如果我们移除纵横比为1/3和3的框，mAP会下降0.6%。如果进一步去除纵横比1/2和2的框，会再下降2.1%。使用多样化的默认框形状会使预测边界框更容易。

**Atrous is faster**. As described in Sec. 3, we used the atrous version of a subsampled VGG16, following DeepLab-LargeFOV [17]. If we use the full VGG16, keeping pool5 with 2×2−s2 and not subsampling parameters from fc6 and fc7, and add conv5_3 for prediction, the result is about the same while the speed is about 20% slower.

**使用Atrous更快一些**。如第3节所述，我们学习DeepLab-LargeFov[17]使用atrous版本的降采样VGG16。如果使用full VGG16，保持pool5层的2×2-s2，并不对fc6和fc7的参数进行下采样，增加conv5_3层进行预测，其结果是相同的，但是速度慢了20%。

Table 2: Effects of various design choices and components on SSD performance.

| | | | SSD300 | | | 
--- | --- | --- | --- | --- | ---
more data augmentation | | y | y | y | y
include { 1/2 ,2} box | y | | y | y | y
include { 1/3 ,3} box | y | | | y | y
use atrous | y | y | y | | y
VOC2007 test mAP | 65.5 | 71.6 | 73.7 | 74.2 | 74.3

**Multiple output layers at different resolutions is better**. A major contribution of SSD is using default boxes of different scales on different output layers. To measure the advantage gained, we progressively remove layers and compare results. For a fair comparison, every time we remove a layer, we adjust the default box tiling to keep the total number of boxes similar to the original (8732). This is done by stacking more scales of boxes on remaining layers and adjusting scales of boxes if needed. We do not exhaustively optimize the tiling for each setting. Table 3 shows a decrease in accuracy with fewer layers, dropping monotonically from 74.3 to 62.4. When we stack boxes of multiple scales on a layer, many are on the image boundary and need to be handled carefully. We tried the strategy used in Faster R-CNN [2], ignoring boxes which are on the boundary. We observe some interesting trends. For example, it hurts the performance by a large margin if we use very coarse feature maps (e.g. conv11_2 (1 × 1) or conv10_2 (3 × 3)). The reason might be that we do not have enough large boxes to cover large objects after the pruning. When we use primarily finer resolution maps, the performance starts increasing again because even after pruning a sufficient number of large boxes remains. If we only use conv7 for prediction, the performance is the worst, reinforcing the message that it is critical to spread boxes of different scales over different layers. Besides, since our predictions do not rely on ROI pooling as in [6], we do not have the collapsing bins problem in low-resolution feature maps [23]. The SSD architecture combines predictions from feature maps of various resolutions to achieve comparable accuracy to Faster R-CNN, while using lower resolution input images.

**对不同的分辨率多输出层要更好**。SSD的一个主要贡献是在不同的输出层使用了不同尺度的默认框。为衡量得到的益处，我们逐渐移除了各层，并比较结果。对于一个公平的比较来说，每次我们移除一层，我们调整默认框的摆放来保持框的总量与原来类似(8732)。这就要在剩下的层中堆积更多尺度的框，如果需要就调整框的尺度。我们没有对每个设置穷尽式优化摆放。表3显示，层数少了，准确率也下降了，从74.3单调下降至62.4。当我们在一层中堆积了多尺度框时，很多是在图像边缘的，需要小心处理。我们使用了Faster R-CNN[2]中使用的策略，忽略了在边缘处的框。我们观察到一些有趣的趋势。比如，如果我们用非常粗糙的特征图（如conv11_2(1×1)或conv10_2(3×3)）的时候，会使表现下降很多。原因可能是，在修剪过后，我们没有足够大的框来覆盖大目标。当我们主要使用更精细分辨率的特征图时，性能再次开始上升，因为即使在修剪过后，足够大数量的大框仍然留存下来。如果我们只使用conv7来进行预测，性能是最差的，这验证了必须要在不同的层之间使用不同尺度的框。另外，既然我们的预测不依赖于[6]中的RoI pooling，在低分辨率特征图中我们就没有collapsing bins的问题[23]。SSD框架结合了不同分辨率的特征图的预测，取得了与Faster R-CNN差不多的准确度，但是使用的是低分辨率输入图像。

## 3.3 PASCAL VOC2012

We use the same settings as those used for our basic VOC2007 experiments above, except that we use VOC2012 trainval and VOC2007 trainval and test (21503images) for training, and test on VOC2012 test (10991 images). We train the models with 10e−3 learning rate for 60k iterations, then 10e−4 for 20k iterations. Table 4 shows the results of our SSD300 and SSD512 model. We see the same performance trend as we observed on VOC2007 test. Our SSD300 improves accuracy over Fast/Faster R-CNN. By increasing the training and testing image size to 512×512, we are 4.5% more accurate than Faster R-CNN. Compared to YOLO, SSD is significantly more accurate, likely due to the use of convolutional default boxes from multiple feature maps and our matching strategy during training. When fine-tuned from models trained on COCO, our SSD512 achieves 80.0% mAP, which is 4.1% higher than Faster R-CNN.

我们与上述基本的VOC2007试验使用相同的设置，除了我们使用VOC2012 trainval和VOC2007 trainval和test集(21503图像)进行训练，然后在VOC2012测试集(10991图像)上进行测试。我们训练模型的学习速率为前60k次迭代为10e-3，然后20k次迭代为10e-4。表4所示的是SSD300和SSD512模型的结果。我们看到，其中与在VOC2007测试集上一样的性能趋势。我们的SSD300比Fast/Faster R-CNN性能更好。通过增加训练和测试图像大小至512×512，我们比Faster R-CNN的准确度高了4.5%。与YOLO相比，SSD准确度明显更高，可能是在训练过程中使用了多特征图中的卷积型默认框和我们的匹配策略。当精调在COCO上训练的模型时，我们的SSD512得到了80.0%的mAP，比Faster R-CNN高了4.1%。

Table 4: PASCAL VOC2012 test detection results. Fast and Faster R-CNN use images with minimum dimension 600, while the image size for YOLO is 448 × 448. data: ”07++12”: union of VOC2007 trainval and test and VOC2012 trainval. ”07++12+COCO”: first train on COCO trainval35k then fine-tune on 07++12.

Method | data | mAP | person plant sheep sofa train tv et al. 20 classed
--- | --- | --- | ---
Fast[6] | 07++12 | 68.4 | 72.0 35.1 68.3 65.7 80.4 64.2
Faster[2] | 07++12 | 70.4 | 79.6 40.1 72.6 60.9 81.2 61.5
Faster[2] | 07++12+COCO | 75.9 | 84.1 52.2 78.9 65.5 85.4 70.2
YOLO[5] | 07++12 | 57.9 | 63.5 28.9 52.2 54.8 73.9 50.8
SSD300 | 07++12 | 72.4 | 79.4 45.9 75.9 69.5 81.9 67.5
SSD300 | 07++12+COCO | 77.5 | 84.3 52.6 82.5 74.1 88.4 74.2
SSD512 | 07++12 | 74.9 | 83.3 50.2 78.0 66.3 86.3 72.0
SSD512 | 07++12+COCO | 80.0 | 86.8 57.2 85.1 72.8 88.4 75.9

### 3.4 COCO

To further validate the SSD framework, we trained our SSD300 and SSD512 architectures on the COCO dataset. Since objects in COCO tend to be smaller than PASCAL VOC, we use smaller default boxes for all layers. We follow the strategy mentioned in Sec. 2.2, but now our smallest default box has a scale of 0.15 instead of 0.2, and the scale of the default box on conv4_3 is 0.07 (e.g. 21 pixels for a 300 × 300 image)(For SSD512 model, we add extra conv12_2 for prediction, set $s_{min}$ to 0.1, and 0.04 on conv4_3).

为进一步验证SSD框架，我们在COCO数据集上训练我们的SSD300和SSD512框架。由于COCO中的目标一般比PASCAL VOC中的要小，我们在所有层中使用更小的默认框。我们采取2.2节提到的策略，但是现在我们最小的默认框尺度是0.15，而不是0.2，conv4_3层上的默认框的尺度为0.07（如对于300×300的图像来说，就是21个像素）（对于SSD512模型，我们增加了conv12_2层进行预测，设$s_{min}$为0.1，在conv4_3上为0.04）。

We use the trainval35k [24] for training. We first train the model with 10e−3 learning rate for 160k iterations, and then continue training for 40k iterations with 10e−4 and 40k iterations with 10e−5. Table 5 shows the results on test-dev2015. Similar to what we observed on the PASCAL VOC dataset, SSD300 is better than Fast R-CNN in both mAP@0.5 and mAP@[0.5:0.95]. SSD300 has a similar mAP@0.75 as ION [24] and Faster R-CNN [25], but is worse in mAP@0.5. By increasing the image size to 512 × 512, our SSD512 is better than Faster R-CNN [25] in both criteria. Interestingly, we observe that SSD512 is 5.3% better in mAP@0.75, but is only 1.2% better in mAP@0.5. We also observe that it has much better AP (4.8%) and AR (4.6%) for large objects, but has relatively less improvement in AP (1.3%) and AR (2.0%) for small objects. Compared to ION, the improvement in AR for large and small objects is more similar (5.4% vs. 3.9%). We conjecture that Faster R-CNN is more competitive on smaller objects with SSD because it performs two box refinement steps, in both the RPN part and in the Fast R-CNN part. In Fig. 5, we show some detection examples on COCO test-dev with the SSD512 model.

我们使用trainval35k[24]进行训练。我们首先以学习速率10e-3训练模型，迭代160k次，然后继续以10e-4的速率迭代40k次，然后40k次10e-5。表5所示的是在test-dev2015集上的结果。与我们在PASCAL VOC数据集上观察到的类似，SSD300比Fast R-CNN要好，mAP@0.5和mAP@[0.5:0.95]的都是这样。SSD300的mAP@0.75与ION[24]和Faster R-CNN[25]类似，但在mAP@0.5时更差。通过把图像增大到512×512，我们的SSD512比Faster R-CNN[25]在两种准则下都要好。有趣的是，我们观察到SSD512在mAP@0.75时要高5.3%，但在mAP@0.5时只高了1.2%。我们还观察到，对于大型目标结果要好的多，AP高了4.8%，AR高了4.6%，但对于小目标来说改进较少，AP 1.3%，AR 2.0%。与ION相比，AR的改进对于大目标和小目标类似5.4% vs. 3.9%。我们推测与SSD相比，Faster R-CNN在较小的目标上更有竞争力，因为它包含2次框的提炼步骤，在RPN部分和Fast R-CNN部分。在图5中，我们展示了SSD512模型检测的COCO test-dev一些例子。

### 3.5 Preliminary ILSVRC results ILSVRC的初步结果

We applied the same network architecture we used for COCO to the ILSVRC DET dataset [16]. We train a SSD300 model using the ILSVRC2014 DET train and val1 as used in [22]. We first train the model with 10e−3 learning rate for 320k iterations, and then continue training for 80k iterations with 10e−4 and 40k iterations with 10e−5. We can achieve 43.4% mAP on the val2 set [22]. Again, it validates that SSD is a general framework for high quality real-time detection.

我们将同样的COCO网络架构应用于ILSVRC DET数据集[16]。我们用ILSVRC2014 DET train and val1集训练了一个SSD300模型，这与[22]中类似。我们首先以10e-3的学习速率训练模型迭代320k次，然后继续训练80k次10e-4，然后40k次10e-5。我们在val2集上[22]得到了43.4% mAP。这又一次验证了SSD是一个高质量实时的通用检测框架。

### 3.6 Data Augmentation for Small Object Accuracy 为提高小目标准确率进行数据扩充

Without a follow-up feature resampling step as in Faster R-CNN, the classification task for small objects is relatively hard for SSD, as demonstrated in our analysis (see Fig. 4). The data augmentation strategy described in Sec. 2.2 helps to improve the performance dramatically, especially on small datasets such as PASCAL VOC. The random crops generated by the strategy can be thought of as a ”zoom in” operation and can generate many larger training examples. To implement a ”zoom out” operation that creates more small training examples, we first randomly place an image on a canvas of 16× of the original image size filled with mean values before we do any random crop operation. Because we have more training images by introducing this new ”expansion” data augmentation trick, we have to double the training iterations. We have seen a consistent increase of 2%-3% mAP across multiple datasets, as shown in Table 6. In specific, Figure 6 shows that the new augmentation trick significantly improves the performance on small objects. This result underscores the importance of the data augmentation strategy for the final model accuracy.

没有Faster R-CNN中后续的特征重采样步骤，小目标的分类任务对于SSD来说相对较难，图4的分析说明了这一点。2.2节叙述的数据扩充策略帮助显著改善了性能，尤其在PASCAL VOC这样的小数据集上。这个策略产生的随机裁剪可以看作是放大操作，可以生成很多更大的训练样本。为实现缩小的操作，这样可以产生更多小训练样本，我们首先随机将图像放置在16倍大原图像的大小上，然后进行随机裁剪。由于通过这个心的扩展数据的技巧，我们有了更多的训练图像，我们需要将训练迭代次数加倍。我们在多个数据集上都看到了一致的2%-3%的mAP提升，如表6所示。特别的，图6展示了新的扩充技巧明显提升了在小目标上的表现。这个结果强调了数据扩充策略对于最终模型准确性的重要性。

Table 6: Results on multiple datasets when we add the image expansion data augmentation trick. SSD300* and SSD512* are the models that are trained with the new data augmentation.

Fig.6: Sensitivity and impact of object size with new data augmentation on VOC2007 test set using [21]. The top row shows the effects of BBox Area per category for the original SSD300 and SSD512 model, and the bottom row corresponds to the SSD300* and SSD512* model trained with the new data augmentation trick. It is obvious that the new data augmentation trick helps detecting small objects significantly.

An alternative way of improving SSD is to design a better tiling of default boxes so that its position and scale are better aligned with the receptive field of each position on a feature map. We leave this for future work.

改进SSD的一种替代方法是设计更好的默认框摆放，这样其位置和尺度可以更好的与特征图上每个位置的感受野对齐。我们未来做这个工作。

### 3.7 Inference time 推理时间

Considering the large number of boxes generated from our method, it is essential to perform non-maximum suppression (nms) efficiently during inference. By using a confidence threshold of 0.01, we can filter out most boxes. We then apply nms with jaccard overlap of 0.45 per class and keep the top 200 detections per image. This step costs about 1.7 msec per image for SSD300 and 20 VOC classes, which is close to the total time (2.4 msec) spent on all newly added layers. We measure the speed with batch size 8 using Titan X and cuDNN v4 with Intel Xeon E5-2667v3@3.20GHz.

考虑我们方法产生了大量框，很必要在推理的时候进行非最大抑制(NMS)。我们使用置信度阈值0.01，滤除掉大部分框。然后我们对每类中重叠度大于0.45的框应用NMS，维持每幅图像中的最高200个检测。在VOC的20类上SSD300模型的这个步骤每幅图像耗费1.7毫秒，这与在所有新加的层上所耗费的时间2.4毫秒比较接近了。我们在batch size 8的情况下衡量这个速度，使用的是Titan X和cuDNN v4的Intel Xeon E5-2667V3@3.20GHz。

Table 7 shows the comparison between SSD, Faster R-CNN[2], and YOLO[5]. Both our SSD300 and SSD512 method outperforms Faster R-CNN in both speed and accuracy. Although Fast YOLO[5] can run at 155 FPS, it has lower accuracy by almost 22% mAP. To the best of our knowledge, SSD300 is the first real-time method to achieve above 70% mAP. Note that about 80% of the forward time is spent on the base network (VGG16 in our case). Therefore, using a faster base network could even further improve the speed, which can possibly make the SSD512 model real-time as well.

表7所示的是SSD、Faster R-CNN[2]和YOLO[5]的对比。我们的SSD300和SSD512方法在速度和准确率上都比Faster R-CNN要好。虽然快速YOLO[5]可以以155FPS的速度运行，但是其准确率低了几乎22%mAP。据我们所知，SSD300是第一个实时检测方法达到70% mAP的。注意大约80%的前向时间是耗费在基础网络上的（我们的情况中是VGG16）。所以，使用更快的基础网络可以进一步改进速度，这很可能使SSD512模型也可以实时运行。

Table 7: Results on Pascal VOC2007 test. SSD300 is the only real-time detection method that can achieve above 70% mAP. By using a larger input image, SSD512 outperforms all methods on accuracy while maintaining a close to real-time speed.

Method | mAP | FPS | batch size | # Boxes | Input resolution
--- | --- | --- | --- | --- | ---
Faster R-CNN (VGG16) | 73.2 | 7 | 1 | ∼ 6000 | ∼ 1000 × 600
Fast YOLO | 52.7 | 155 | 1 | 98 | 448 × 448
YOLO (VGG16) | 66.4 | 21 | 1 | 98 | 448 × 448
SSD300 | 74.3 | 46 | 1 | 8732 | 300 × 300
SSD512 | 76.8 | 19 | 1 | 24564 | 512 × 512
SSD300 | 74.3 | 59 | 8 | 8732 | 300 × 300
SSD512 | 76.8 | 22 | 8 | 24564 | 512 × 512

## 4 Related Work 相关工作

There are two established classes of methods for object detection in images, one based on sliding windows and the other based on region proposal classification. Before the advent of convolutional neural networks, the state of the art for those two approaches – Deformable Part Model (DPM) [26] and Selective Search [1] – had comparable performance. However, after the dramatic improvement brought on by R-CNN [22], which combines selective search region proposals and convolutional network based post-classification, region proposal object detection methods became prevalent.

有两类确定的图像目标检测方法，一种基于滑窗方法，另一种基于候选区域分类法。在卷积神经网络到来之前，两类对应的最好的方法是DPM[26]和Selective Search[1]，它们性能接近。但是，R-CNN[22]结合了selective search候选区域和基于卷积网络的后续分类，带来了激动人心的改进，其后候选区域目标检测法逐渐流行起来。

The original R-CNN approach has been improved in a variety of ways. The first set of approaches improve the quality and speed of post-classification, since it requires the classification of thousands of image crops, which is expensive and time-consuming. SPPnet [9] speeds up the original R-CNN approach significantly. It introduces a spatial pyramid pooling layer that is more robust to region size and scale and allows the classification layers to reuse features computed over feature maps generated at several image resolutions. Fast R-CNN [6] extends SPPnet so that it can fine-tune all layers end-to-end by minimizing a loss for both confidences and bounding box regression, which was first introduced in MultiBox [7] for learning objectness.

原始R-CNN方法已经进行了多种改进。第一类方法集合改进后分类的质量和速度，因为需要对数千个图像块进行分类，计算量大，耗时长。SPPnet[9]显著提升了原始R-CNN的速度，它引入了一种空间金字塔pooling层，对区域大小和尺度都很鲁棒，允许分类层重新使用特征，这些特征是在几种图像分辨率上生成的特征图中计算出来的。Fast R-CNN[6]扩展了SPPnet，可以端到端的精调所有的层，方法是最小化置信度和边界框回归的联合损失，这是MultiBox[7]为学习objectness首次引入的。

The second set of approaches improve the quality of proposal generation using deep neural networks. In the most recent works like MultiBox [7,8], the Selective Search region proposals, which are based on low-level image features, are replaced by proposals generated directly from a separate deep neural network. This further improves the detection accuracy but results in a somewhat complex setup, requiring the training of two neural networks with a dependency between them. Faster R-CNN [2] replaces selective search proposals by ones learned from a region proposal network (RPN), and introduces a method to integrate the RPN with Fast R-CNN by alternating between fine-tuning shared convolutional layers and prediction layers for these two networks. This way region proposals are used to pool mid-level features and the final classification step is less expensive. Our SSD is very similar to the region proposal network (RPN) in Faster R-CNN in that we also use a fixed set of (default) boxes for prediction, similar to the anchor boxes in the RPN. But instead of using these to pool features and evaluate another classifier, we simultaneously produce a score for each object category in each box. Thus, our approach avoids the complication of merging RPN with Fast R-CNN and is easier to train, faster, and straightforward to integrate in other tasks.

第二种方法集合用深度神经网络改进生成的候选区域的质量。最近的工作如MultiBox[7,8]中，基于低层图像特征的selective search区域候选法，被替换为另一个深度神经网络直接生成的候选。这进一步改进了检测效率，但得到了相对复杂的设置，需要训练两个神经网络，而且之间还有依赖关系。Faster R-CNN[2]将selective search区域候选替换为区域候选网络RPN，引入了整合RPN和Fast R-CNN的方法。这种方法的候选区域被用来pool中层特征，最后的分类步骤没那么费时。我们的SSD与Faster R-CNN中的RPN比较类似，在于我们也使用了默认框的固定集合来预测，与PRN中的锚框类似。但是我们没有用这个来pool特征然后评估另一个分类器，我们同时在每个框中产生了每个类别的分数。所以，我们的方法避免了将RPN和Fast R-CNN合并的复杂性，更容易进行训练，可以更快更直接的整合到其他任务中。

Another set of methods, which are directly related to our approach, skip the proposal step altogether and predict bounding boxes and confidences for multiple categories directly. OverFeat [4], a deep version of the sliding window method, predicts a bounding box directly from each location of the topmost feature map after knowing the confidences of the underlying object categories. YOLO [5] uses the whole topmost feature map to predict both confidences for multiple categories and bounding boxes (which are shared for these categories). Our SSD method falls in this category because we do not have the proposal step but use the default boxes. However, our approach is more flexible than the existing methods because we can use default boxes of different aspect ratios on each feature location from multiple feature maps at different scales. If we only use one default box per location from the topmost feature map, our SSD would have similar architecture to OverFeat [4]; if we use the whole topmost feature map and add a fully connected layer for predictions instead of our convolutional predictors, and do not explicitly consider multiple aspect ratios, we can approximately reproduce YOLO [5].

令一类方法直接与我们的方法相关，跳过了候选步骤，直接预测多类别的边界框和置信度。OverFeat[4]，滑窗方法的深度网络版本，在知道了潜在的目标类别的置信度后，直接从最高层特征图的每个位置预测边界框。YOLO[5]使用整个最高层特征图来预测多类别置信度和边界框。我们的SSD方法也属于这个类别，因为我们没有候选区域步骤，而是使用了默认框。但是，我们的方法比现有方法更加灵活，因为我们可以使用不同纵横比的默认框，这是不同尺度上的多个特征图的每个位置上都有的。如果我们只在最高层特征图的每个位置使用一个默认框，我们的SSD将和OverFeat[4]架构类似；如果我们使用整个最高层特征图，然后增加一个全连接层进行预测，而不是我们的卷积预测器，而且不显式的考虑多个纵横比，我们可以近似的再现YOLO[5]。

## 5 Conclusions 结论

This paper introduces SSD, a fast single-shot object detector for multiple categories. A key feature of our model is the use of multi-scale convolutional bounding box outputs attached to multiple feature maps at the top of the network. This representation allows us to efficiently model the space of possible box shapes. We experimentally validate that given appropriate training strategies, a larger number of carefully chosen default bounding boxes results in improved performance. We build SSD models with at least an order of magnitude more box predictions sampling location, scale, and aspect ratio, than existing methods [5,7]. We demonstrate that given the same VGG-16 base architecture, SSD compares favorably to its state-of-the-art object detector counterparts in terms of both accuracy and speed. Our SSD512 model significantly outperforms the state-of-the-art Faster R-CNN [2] in terms of accuracy on PASCAL VOC and COCO, while being 3× faster. Our real time SSD300 model runs at 59 FPS, which is faster than the current real time YOLO [5] alternative, while producing markedly superior detection accuracy.

本文提出了SSD算法，一种快速的单发多类目标检测器。我们模型的一个关键特征是使用了网络高层的多特征图关联的多尺度卷积边界框输出。这种表示使我们可以有效的对可能的边界框空间进行建模。我们的试验验证了，给定合适的训练策略，仔细选择的默认框越多，越能改进性能。我们构建的SSD模型比现有方法[5,7]，在边界框位置、尺度和纵横比上都多了一个量级。我们证明了，给定同样的VGG16基础网络框架，SSD在准确度和速度上都超过了现在最好的目标检测器。我们的SSD512模型显著超过了现在最好的Faster R-CNN[2]，在PASCAL VOC和COCO上准确率上得到了超越，而且速度快了3倍。我们的实时SSD300模型以59FPS运行，比现在的实时YOLO[5]更快，而且检测准确率高的多。

Apart from its standalone utility, we believe that our monolithic and relatively simple SSD model provides a useful building block for larger systems that employ an object detection component. A promising future direction is to explore its use as part of a system using recurrent neural networks to detect and track objects in video simultaneously.

除了单机版，我们相信这个整体的相对简单的SSD模型可以作为更大的系统的有用部件。未来的一个方向是探索作为系统一部分的应用，该系统会使用循环神经网络来对视频进行检测和跟踪目标。

## 6 Acknowledgment

This work was started as an internship project at Google and continued at UNC. We would like to thank Alex Toshev for helpful discussions and are indebted to the Image Understanding and DistBelief teams at Google. We also thank Philip Ammirato and Patrick Poirson for helpful comments. We thank NVIDIA for providing GPUs and acknowledge support from NSF 1452851, 1446631, 1526367, 1533771.