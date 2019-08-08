# Objects as Points

Xingyi Zhou et al. UT Austin

## Abstract

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point — the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy tradeoff on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

检测将图像中的目标识别为与坐标轴对齐的框。多数成功的目标检测器，都几乎穷尽枚举了潜在的目标位置，并对其分类。这很浪费，低效，需要额外的后处理。本文中，我们采取了一种不同的方法。我们将目标建模为一个点，边界框的中心点。我们的检测器使用关键点估计来找到中心点，并回归出所有其他目标性质，如大小，3D位置，方向，甚至是姿态。我们这种基于中心点的位置，称为CenterNet，是端到端可微分的，更简单，更快速，比对应的基于边界框的检测器更准确。CenterNet在MS COCO数据集上取得来最好的准确率-速度折中，AP 28.1%时速度142 FPS，AP 37.4%时速度52 FPS，AP 45.1%时采用来多尺度测试，速度为1.4 FPS。我们使用相同的方法在KITTI数据集中来估计3D边界框，在COCO关键点数据集中估计人体姿态。我们的方法与复杂的多阶段方法性能类似，并可以以实时速度运行。

## 1. Introduction 引言

Object detection powers many vision tasks like instance segmentation [7, 21, 32], pose estimation [3, 15, 39], tracking [24, 27], and action recognition [5]. It has down-stream applications in surveillance [57], autonomous driving [53], and visual question answering [1]. Current object detectors represent each object through an axis-aligned bounding box that tightly encompasses the object [18, 19, 33, 43, 46]. They then reduce object detection to image classification of an extensive number of potential object bounding boxes. For each bounding box, the classifier determines if the image content is a specific object or background. One- stage detectors [33, 43] slide a complex arrangement of possible bounding boxes, called anchors, over the image and classify them directly without specifying the box content. Two-stage detectors [18, 19, 46] recompute image features for each potential box, then classify those features. Post-processing, namely non-maxima suppression, then removes duplicated detections for the same instance by computing bounding box IoU. This post-processing is hard to differentiate and train [23], hence most current detectors are not end-to-end trainable. Nonetheless, over the past five years [19], this idea has achieved good empirical success [12,21,25,26,31,35,47,48,56,62,63]. Sliding window based object detectors are however a bit wasteful, as they need to enumerate all possible object locations and dimensions.

目标检测为很多视觉任务赋能，如实例分割，姿态估计，跟踪和动作识别。在监控、自动驾驶和视觉问题回答中有很多下游应用。目前的目标检测器将每个目标表示为与坐标轴对齐的边界框，紧紧的框住目标。然后将目标检测的问题，转化为图像分类，并在数量巨大的潜在目标边界框中进行。对于每个边界框，分类器确定图像内容是特定的目标或者是背景。单阶段检测器包含整幅图像上的复杂的可能边界框排列，称为锚框，然后直接进行分类，不指定框的内容。两阶段检测器重新计算每个可能的框的图像特征，然后对这些特征进行分类。后处理，即非极大抑制，通过计算边界框IoU去掉同一实例的重复检测。这种后处理很难微分和训练，所以目前多数检测器都不是端到端可训练的。尽管如此，在过去5年中，这种思想取得很好的成功。基于滑窗的目标检测器有一点浪费，因为需要枚举所有可能的目标位置和维度。

In this paper, we provide a much simpler and more efficient alternative. We represent objects by a single point at their bounding box center (see Figure 2). Other properties, such as object size, dimension, 3D extent, orientation, and pose are then regressed directly from image features at the center location. Object detection is then a standard keypoint estimation problem [3, 39, 60]. We simply feed the input image to a fully convolutional network [37, 40] that generates a heatmap. Peaks in this heatmap correspond to object centers. Image features at each peak predict the objects bounding box height and weight. The model trains using standard dense supervised learning [39,60]. Inference is a single network forward-pass, without non-maximal suppression for post-processing.

本文中，我们提出了一种更简单更高效的选择。我们用一个单点来表示目标，在其边界框中心，如图2所示。其他性质，如目标大小，维度，3D大小，方向和姿态，都直接从中心位置的图像特征中直接回归得到。目标检测然后就称为来一个标准的关键估计问题。我们只需要将图像送入一个全卷积网络，生成一个热力图。热力图中的峰值就对应着目标中心。每个峰值处的图像特征可以预测目标边界框的宽度和高度。模型训练使用标准的密集监督学习。推理的过程是网络的一次前向过程，不需要非极大抑制进行后处理。

Our method is general and can be extended to other tasks with minor effort. We provide experiments on 3D object detection [17] and multi-person human pose estimation [4], by predicting additional outputs at each center point (see Figure 4). For 3D bounding box estimation, we regress to the object absolute depth, 3D bounding box dimensions, and object orientation [38]. For human pose estimation, we consider the 2D joint locations as offsets from the center and directly regress to them at the center point location.

我们的方法是一般性的，可以很容易拓展到其他任务中。我们给出了3D目标检测的实验，和多人人体姿态估计的实验，在每个中心点预测额外的输出（见图4）。对于3D边界框估计，我们回归目标的绝对深度，3D边界框的维度，和目标方向。对于人体姿态估计，我们将2D关节位置表示为中心点的偏移，直接在中心点位置对其回归。

The simplicity of our method, CenterNet, allows it to run at a very high speed (Figure 1). With a simple Resnet-18 and up-convolutional layers [55], our network runs at 142 FPS with 28.1% COCO bounding box AP. With a carefully designed keypoint detection network, DLA-34 [58], our network achieves 37.4% COCO AP at 52 FPS. Equipped with the state-of-the-art keypoint estimation network, Hourglass-104 [30, 40], and multi-scale testing, our network achieves 45.1% COCO AP at 1.4 FPS. On 3D bounding box estimation and human pose estimation, we perform competitively with state-of-the-art at a higher inference speed. Code is available at https://github.com/xingyizhou/CenterNet.

我们的方法CenterNet的简洁性，使其可以非常高速运行（图1）。使用简单的ResNet-18和上卷积层，我们的网络运行速度可达142 FPS，COCO边界框AP达28.1%。使用仔细设计的关键点检测网络，DLA-34，我们的网络可以达到37.4% AP，速度52 FPS。使用目前最好的关键点估计网络，Hourglass-104，并使用多尺度测试，我们的网络可以达到45.1% COCO AP，速度1.4 FPS。在3D边界框估计和人体姿态估计上，我们与目前最好的模型性能相当，但推理速度更快。代码已开源。

Figure 1: Speed-accuracy trade-off on COCO validation for real-time detectors. The proposed CenterNet outperforms a range of state-of-the-art algorithms.

Figure 2: We model an object as the center point of its bounding box. The bounding box size and other object properties are inferred from the keypoint feature at the center. Best viewed in color.

## 2. Related work

**Object detection by region classification**. One of the first successful deep object detectors, RCNN [19], enumerates object location from a large set of region candidates [52], crops them, and classifies each using a deep network. Fast-RCNN [18] crops image features instead, to save computation. However, both methods rely on slow low-level region proposal methods.

**通过区域分类目标检测**。第一个成功的深度目标检测器，RCNN，从区域候选的大型集合中枚举目标位置，对其进行剪切，并使用深度网络对其进行分类。Fast-RCNN转而剪切图像特征，以节省计算量。但是，这两类方法都依赖于很慢的低层区域建议算法。

**Object detection with implicit anchors**. Faster RCNN [46] generates region proposal within the detection network. It samples fixed-shape bounding boxes (anchors) around a low-resolution image grid and classifies each into “foreground or not”. An anchor is labeled foreground with a >0.7 overlap with any ground truth object, background with a <0.3 overlap, or ignored otherwise. Each generated region proposal is again classified [18]. Changing the proposal classifier to a multi-class classification forms the basis of one-stage detectors. Several improvements to one-stage detectors include anchor shape priors [44, 45], different feature resolution [36], and loss re-weighting among different samples [33].

**采用隐式锚框的目标检测**。Faster RCNN使用检测网络生成区域建议。在低分辨率图像网格上对固定形状的边界框（锚框）进行取样，将每一个分类为前景或背景。一个锚框如果与任何真值目标的IoU大于0.7，就标记为前景类别，如果IoU小于0.3，就是背景。每个生成的区域建议被再次分类。将建议分类器变为多类别分类，形成了单阶段检测器的基础。几种单阶段检测器的改进包括，锚框形状的先验，不同的特征分辨率，和在不同样本中进行损失重新赋权。

Our approach is closely related to anchor-based one-stage approaches [33, 36, 43]. A center point can be seen as a single shape-agnostic anchor (see Figure 3). However, there are a few important differences. First, our CenterNet assigns the “anchor” based solely on location, not box overlap [18]. We have no manual thresholds [18] for foreground and background classification. Second, we only have one positive “anchor” per object, and hence do not need Non-Maximum Suppression (NMS) [2]. We simply extract local peaks in the keypoint heatmap [4, 39]. Third, CenterNet uses a larger output resolution (output stride of 4) compared to traditional object detectors [21, 22] (output stride of 16). This eliminates the need for multiple anchors [47].

我们的方法与基于锚框的单阶段方法紧密相关。一个中心点可以视为与形状无关的单个锚框。但是，有几个重要的不同之处。第一，我们的CenterNet只基于位置来指定“锚框”，而不基于重叠度。对于前景和背景的分类，我们没有手工设置的阈值。第二，每个目标我们只有一个正“锚”，所以不需要非极大值抑制(NMS)。我们只需要在关键点热力图中提取局部极值。第三，CenterNet使用的输出分辨率更大一些（输出步长为4），而传统目标检测器则一般为16。这样就不需要多个锚框了。

**Object detection by keypoint estimation**. We are not the first to use keypoint estimation for object detection. CornerNet [30] detects two bounding box corners as keypoints, while ExtremeNet [61] detects the top-, left-, bottom-, right- most, and center points of all objects. Both these methods build on the same robust keypoint estimation network as our CenterNet. However, they require a combinatorial grouping stage after keypoint detection, which significantly slows down each algorithm. Our CenterNet, on the other hand, simply extracts a single center point per object without the need for grouping or post-processing.

**使用关键点估计进行目标检测**。我们不是第一个采用关键点估计来进行目标检测的。CornerNet检测边界框的两个角点作为关键点，而ExtremeNet检测所有目标的上、左、下、右和中心点。这些方法和我们的CeneterNet一样，都是基于相同的稳健关键点估计网络。但是，他们在关键点检测后，需要一个组合分组的阶段，这使得算法速度明显下降。我们的CenterNet，每个目标只提取一个中心点，不需要分组或后处理。

**Monocular 3D object detection**. 3D bounding box estimation powers autonomous driving [17]. Deep3Dbox [38] uses a slow-RCNN [19] style framework, by first detecting 2D objects [46] and then feeding each object into a 3D estimation network. 3D RCNN [29] adds an additional head to Faster-RCNN [46] followed by a 3D projection. Deep Manta [6] uses a coarse-to-fine Faster-RCNN [46] trained on many tasks. Our method is similar to a one-stage version of Deep3Dbox [38] or 3DRCNN [29]. As such, CenterNet is much simpler and faster than competing methods.

**单目3D目标检测**。3D边界框估计为自动驾驶赋能。Deep3Dbox使用slow-RCNN类型的框架，首先检测2D目标，然后将每个目标送入一个3D估计网络。3D RCNN给Faster R-CNN增加了一个额外的头，即一个3D投影。Deep Manta使用了一个由粗糙到精细的Faster R-CNN，在很多任务上进行的训练。我们的方法类似于单阶段版的Deep3Dbox或3D RCNN。所以，CenterNet比这些方法要更简单，更快速。

Figure 3: Different between anchor-based detectors (a) and our center point detector (b). Best viewed on screen.

(a) Standard anchor based detection. Anchors count as positive with an overlap IoU > 0.7 to any object, negative with an overlap IoU < 0.3, or are ignored otherwise.

(b) Center point based detection. The center pixel is assigned to the object. Nearby points have a reduced negative loss. Object size is regressed.

## 3. Preliminary

Let I ∈ $R^{W×H×3}$ be an input image of width W and height H. Our aim is to produce a keypoint heatmap $\hat Y ∈[0,1]^{W/R × H/R × C}$, where R is the output stride and C is the number of keypoint types. Keypoint types include C = 17 human joints in human pose estimation [4,55], or C = 80 object categories in object detection [30,61]. We use the default output stride of R = 4 in literature [4,40,42]. The output stride downsamples the output prediction by a factor R. A prediction $\hat Y_{x,y,c} = 1$ corresponds to a detected keypoint, while $\hat Y_{x,y,c} = 0$ is background. We use several different fully-convolutional encoder-decoder networks to predict $\hat Y$ from an image I: A stacked hourglass network [30,40], up-convolutional residual networks (ResNet) [22,55], and deep layer aggregation (DLA) [58].

令I ∈ $R^{W×H×3}$表示输入图像，宽W高H。我们的目标是生成关键点热力图$\hat Y ∈[0,1]^{W/R × H/R × C}$，其中R是输出步长，C是关键点类型的数量。关键点类型包括，在人体姿态估计中，C=17人体关节点，在目标检测中C=80目标类别。我们使用文献[4,40,42]中默认的输出步长R=4。输出步长是将输出预测降采样的系数R。预测$\hat Y_{x,y,c} = 1$对应着检测到的关键点，而$\hat Y_{x,y,c} = 0$则为背景。我们使用几种不同的全卷积编码器-解码器网络，来从图像I中预测$\hat Y$：堆叠沙漏网络，上卷积残差网络，和深度层聚集DLA。

We train the keypoint prediction network following Law and Deng [30]. For each ground truth keypoint p ∈ R^2 of class c, we compute a low-resolution equivalent $\tilde p  = ⌊ \frac{p}{R} ⌋$. We then splat all ground truth keypoints onto a heatmap $Y ∈ [0, 1]^{W/R × H/R × C}$ using a Gaussian kernel $Y_{xyc} = exp(-\frac {(x-\tilde p_x)^2+(y-\tilde p_y)^2} {2σ_p^2})$, where σ_p is an object size-adaptive standard deviation [30]. If two Gaussians of the same class overlap, we take the element-wise maximum [4]. The training objective is a penalty-reduced pixelwise logistic regression with focal loss [33]:

我们使用Law等[30]等人的方法训练关键点预测网络。对于每个类别为c的真值关键点p ∈ R^2，我们计算一个低分辨率的等价点$\tilde p  = ⌊ \frac{p}{R} ⌋$。我们然后计算所有真值关键点的热力图$Y ∈ [0, 1]^{W/R × H/R × C}$，使用一个高斯核$Y_{xyc} = exp(-\frac {(x-\tilde p_x)^2+(y-\tilde p_y)^2} {2σ_p^2})$进行卷积，其中σ_p是随目标大小变化的标准偏差。如果两个高斯核有相同的类别重叠，那么我们就取其逐元素的最大值。训练目标是penalty-reduced逐元素的logistic回归，有聚焦损失（focal loss）：

$$L_k = -\frac {1} {N} \sum_{xyc} (1-\hat Y_{xyc})^α log(\hat Y_{xyc}), if Y_{xyc}=1; -\frac {1} {N} \sum_{xyc} (1-Y_{xyc})^β (\hat Y_{xyc})^α log(1-\hat Y_{xyc}), otherwise$$(1)

where α and β are hyper-parameters of the focal loss [33], and N is the number of keypoints in image I. The normalization by N is chosen as to normalize all positive focal loss instances to 1. We use α = 2 and β = 4 in all our experiments, following Law and Deng [30].

其中α和β是focal loss的超参数，N是图像I中的关键点数量。除以N归一化是要将所有正的focal loss实例归一化到1。我们在所有试验中都使用α = 2和β = 4，这与Law等[30]一样。

To recover the discretization error caused by the output stride, we additionally predict a local offset $\hat O ∈ R^{W/R × H/R × 2}$ for each center point. All classes c share the same offset prediction. The offset is trained with an L1 loss

为将输出步长导致的离散化误差进行恢复，我们对每个中心点额外预测了一个局部偏移$\hat O ∈ R^{W/R × H/R × 2}$。所有类别c共享相同的偏移预测。偏移的训练用的是L1损失

$$L_{off} = \frac {1} {N} \sum_p |\hat O_{\tilde p} - (\frac {p} {R} - \tilde p)|$$(2)

The supervision acts only at keypoints location $\tilde p$, all other locations are ignored. 这个监督只在关键点位置起作用，其他位置上都忽略这个监督。

In the next section, we will show how to extend this keypoint estimator to a general purpose object detector. 下一节，我们会展示，怎样将这个关键点估计器拓展为一个通用目标的目标检测器。

## 4. Objects as Points

Let $(x_1^{(k)}, y_1^{(k)}, x_2^{(k)}, y_2^{(k)})$ be the bounding box of object k with category $c_k$. Its center point lies at $p_k = (\frac {x_1^{(k)}+x_2^{(k)}} {2}, \frac {y_1^{(k)}+y_2^{(k)}} {2})$. We use our keypoint estimator $\hat Y$ to predict all center points. In addition, we regress to the object size $s_k = (x_2^{(k)} − x_1^{(k)}, y_2^{(k)} − y_1^{(k)})$ for each object k. To limit the computational burden, we use a single size prediction $\hat S ∈ R^{W/R × H/R ×2}$ for all object categories. We use an L1 loss at the center point similar to Objective 2:

令$(x_1^{(k)}, y_1^{(k)}, x_2^{(k)}, y_2^{(k)})$为类别$c_k$的目标k的边界框。其中心点在$p_k = (\frac {x_1^{(k)}+x_2^{(k)}} {2}, \frac {y_1^{(k)}+y_2^{(k)}} {2})$。我们使用关键点估计器$\hat Y$来预测所有的中心点。另外，我们对每个目标k回归得到目标大小$s_k = (x_2^{(k)} − x_1^{(k)}, y_2^{(k)} − y_1^{(k)})$。为降低计算负担，我们对所有目标类别都使用单尺度的预测$\hat S ∈ R^{W/R × H/R ×2}$。我们在中心点使用L1损失，与目标函数2类似：

$$L_{size} = \frac {1} {N} \sum_{k=1}^N |\hat S_{p_k} - s_k|$$(3)

We do not normalize the scale and directly use the raw pixel coordinates. We instead scale the loss by a constant $λ_{size}$. The overall training objective is 我们不对尺度进行归一化，而直接使用原始像素坐标。我们对损失函数进行加权，使用常数$λ_{size}$。总体的训练目标为

$$L_{det} = L_k + λ_{size} L_{size} +λ_{off} L_{off}$$(4)

We set $λ_{size}$ = 0.1 and $λ_{off}$ = 1 in all our experiments unless specified otherwise. We use a single network to predict the keypoints $\hat Y$, offset $\hat O$, and size $\hat S$. The network predicts a total of C + 4 outputs at each location. All outputs share a common fully-convolutional backbone network. For each modality, the features of the backbone are then passed through a separate 3 × 3 convolution, ReLU and another 1 × 1 convolution. Figure 4 shows an overview of the network output. Section 5 and supplementary material contain additional architectural details.

我们设$λ_{size}$ = 0.1，$λ_{off}$ = 1，在所有试验中都使用这些值，除非另外指定。我们使用单个网络来预测关键点$\hat Y$, 偏移$\hat O$, 和大小$\hat S$。网络对每个位置共预测C+4个输出。所有输出共享一个常用的全卷积骨干网络。对每个模式，骨干的特征都送入一个3✖3卷积、ReLU和1✖1卷积中。图4给出来网络输出的概览。第5部分和附加资料包含了另外的架构细节。

Figure 4: Outputs of our network for different tasks: top for object detection, middle for 3D object detection, bottom: for pose estimation. All modalities are produced from a common backbone, with a different 3 × 3 and 1 × 1 output convolutions separated by a ReLU. The number in brackets indicates the output channels. See section 4 for details. keypoint heatmap [C] local offset [2] object size [2] 3D size [3] depth [1] orientation [8] joint locations [k × 2] joint heatmap [k] joint offset [2]

**From points to bounding boxes**. At inference time, we first extract the peaks in the heatmap for each category independently. We detect all responses whose value is greater or equal to its 8-connected neighbors and keep the top 100 peaks. Let $\hat P_c$ be the set of n detected center points $\hat P = \{(\hat x_i, \hat y_i)\}^n_{i=1}$ of class c. Each keypoint location is given by an integer coordinates ($x_i,y_i$). We use the keypoint values $\hat Y_{x_iy_ic}$ as a measure of its detection confidence, and produce a bounding box at location

**从点到边界框**。在推理时，我们首先对每个类别独立从热力图中提取出峰值。我们检测所有值大于或等于其8邻域的响应，保留最高的100个峰值。令$\hat P_c$为n个检测到的类别c的n个中心点的集合$\hat P = \{(\hat x_i, \hat y_i)\}^n_{i=1}$。每个关键点位置由于整数坐标给出($x_i,y_i$)。我们使用关键点的值$\hat Y_{x_iy_ic}$作为其检测置信度的度量，在下面的位置生成一个边界框

$$(\hat x_i + δ\hat x_i − \hat w_i/2, \hat y_i + δ\hat y_i − \hat h_i/2, \hat x_i + δ\hat x_i + \hat w_i/2, \hat y_i + δ\hat y_i + \hat h_i/2)$$

where $(δ\hat x_i,δ\hat y_i) = \hat O_{\hat x_i,\hat y_i}$ is the offset prediction and $(\hat w_i,\hat h_i) = \hat S_{\hat x_i, \hat y_i}$ is the size prediction. All outputs are produced directly from the keypoint estimation without the need for IoU-based non-maxima suppression (NMS) or other post-processing. The peak keypoint extraction serves as a sufficient NMS alternative and can be implemented efficiently on device using a 3 × 3 max pooling operation.

其中$(δ\hat x_i,δ\hat y_i) = \hat O_{\hat x_i,\hat y_i}$是偏移预测，$(\hat w_i,\hat h_i) = \hat S_{\hat x_i, \hat y_i}$是大小预测。所有输出都直接从关键点预测中生成，不需要基于IoU的非极大抑制(NMS)或其他后处理。峰值关键点提取是NMS的充分替代，可以在设备上进行高效的实现，使用3✖3最大池化运算。

### 4.1. 3D detection

3D detection estimates a three-dimensional bounding box per objects and requires three additional attributes percenter point: depth, 3D dimension, and orientation. We add a separate head for each of them. The depth d is a single scalar per center point. However, depth is difficult to regress to directly. We instead use the output transformation of Eigen et al. [13] and d = 1/σ($\hat d$) − 1, where σ is the sigmoid function. We compute the depth as an additional output channel $\hat D ∈ [0, 1]^{W/R × H/R}$ of our keypoint estimator. It again uses two convolutional layers separated by a ReLU. Unlike previous modalities, it uses the inverse sigmoidal transformation at the output layer. We train the depth estimator using an L1 loss in the original depth domain, after the sigmoidal transformation.

3D检测对每个目标估计三维边界框，每个中心点需要三个额外的属性：深度、3D维度和方向。我们对每一个，增加一个单独的头。对每个中心点来说，深度d是单个标量。但是，深度很难直接回归得到。我们使用了Eigen等的输出变换，和d = 1/σ($\hat d$) − 1，其中σ是sigmoid函数。我们将深度计算为我们的关键点估计器的一个额外的输出通道$\hat D ∈ [0, 1]^{W/R × H/R}$，再次使用了ReLU分隔的两个卷积层。与之前的模式不一样的是，在输出层中这次使用了逆sigmoid变换。在sigmoid变换后，我们使用原始深度域的L1损失训练深度估计器。

The 3D dimensions of an object are three scalars. We directly regress to their absolute values in meters using a separate head $\hat Γ ∈ R^{W/R × H/R ×3}$ and an L1 loss.

一个目标的3D维度是三个标量。我们直接回归其以米为单位的绝对值，使用了一个单独的头$\hat Γ ∈ R^{W/R × H/R ×3}$，使用的是L1损失。

Orientation is a single scalar by default. However, it can be hard to regress to. We follow Mousavian et al. [38] and represent the orientation as two bins with in-bin regression. Specifally, the orientation is encoded using 8 scalars, with 4 scalars for each bin. For one bin, two scalars are used for softmax classification and the rest two scalar regress to an angle within each bin. Please see the supplementary for details about these losses.

方向默认是一个标量。但是，要回归却很难。我们按照Mousavian等[38]的思路，将方向表示为两个bins，使用bin内回归。具体的，方向使用8个标量编码，每个bin有4个标量。每个bin中，2个标量用于softmax分类，剩余的2个标量每个bin回归到一个角度。这些损失函数详见附加材料。

### 4.2. Human pose estimation

Human pose estimation aims to estimate k 2D human joint locations for every human instance in the image (k = 17 for COCO). We considered the pose as a k × 2-dimensional property of the center point, and parametrize each keypoint by an offset to the center point. We directly regress to the joint offsets (in pixels) $\hat J ∈ R^{W/R × H/R × k × 2}$ with an L1 loss. We ignore the invisible keypoints by masking the loss. This results in a regression-based one-stage multiperson human pose estimator similar to the slow-RCNN version counterparts Toshev et al. [51] and Sun et al. [49].

人体姿态估计的目标是，对图像中的每个人，估计k个2D人体关节位置（对于COCO来说，k=17）。我们将姿态考虑为k个中心点的2维属性，将每个关键点参数化为到中心点的偏移。我们直接用L1损失回归关节点的偏移（以像素计）$\hat J ∈ R^{W/R × H/R × k × 2}$。我们忽略了不可见的关键点，将其损失掩膜掉。这个可以得到一种基于回归的单阶段多人人体姿态估计器，与Toshev等[51]和Sun等[49]的slow-RCNN版类似。

To refine the keypoints, we further estimate k human joint heatmaps $\hat Φ ∈ R^{W/R × H/R × k}$ using standard bottom-up multi-human pose estimation [4,39,41]. We train the human joint heatmap with focal loss and local pixel offset analogous to the center detection discussed in Section. 3.

为提炼关键点，我们进一步估计k个人体关节点的热力图$\hat Φ ∈ R^{W/R × H/R × k}$，使用的是标准的自下而上的多人姿态估计器。我们使用focal loss和局部像素偏移训练人体关节点热力图，类似于第3部分讨论的中心点检测。

We then snap our initial predictions to the closest detected keypoint on this heatmap. Here, our center offset acts as a grouping cue, to assign individual keypoint detections to their closest person instance. Specifically, let ($\hat x, \hat y$) be a detected center point. We first regress to all joint locations $l_j = (\hat x, \hat y) + \hat J_{\hat x \hat y j}$ for $j ∈ 1...k$. We also extract all keypoint locations $L_j = \{\tilde l_{ji} \}_{i=1}^{nj}$ with a confidence > 0.1 for each joint type j from the corresponding heatmap $\hat Φ_{··j}$. We then assign each regressed location $l_j$ to its closest detected keypoint $argmin_{l∈L_j} (l − l_j)^2$ considering only joint detections within the bounding box of the detected object.

我们然后将初始预测与这个热力图上最接近的检测关键点对齐。这里，我们的中心偏移的作用是分组线索，将单个的关键点检测指定到最接近的人体实例中去。令($\hat x, \hat y$)为检测到的中心点。我们首先对$j ∈ 1...k$回归所有的关节点位置$l_j = (\hat x, \hat y) + \hat J_{\hat x \hat y j}$。我们还以>0.1的置信度对每种关节点j从对应的热力图$\hat Φ_{··j}$中提取所有的关节点位置。然后我们将每个回归到的位置$l_j$指定给其最接近的检测到的关键点$argmin_{l∈L_j} (l − l_j)^2$，只考虑检测到的目标的边界框内的关节点检测。

## 5. Implementation details

We experiment with 4 architectures: ResNet-18, ResNet-101 [55], DLA-34 [58], and Hourglass-104 [30]. We modify both ResNets and DLA-34 using deformable convolution layers [12] and use the Hourglass network as is. 我们采用4种架构进行试验：ResNet-18，ResNet-101，DLA-34和Hourglass-104。我们使用可变形卷积修正ResNets和DLA-34，Hourglass则没有改变进行使用。

**Hourglass**. The stacked Hourglass Network [30, 40] downsamples the input by 4×, followed by two sequential hourglass modules. Each hourglass module is a symmetric 5-layer down- and up-convolutional network with skip connections. This network is quite large, but generally yields the best keypoint estimation performance.

**Hourglass**。堆叠Hourglass网络将输入进行4x下采样，然后是2个顺序排列的hourglass模块。每个hourglass模块都是对称的5层下卷积和上卷积网络，并带有跳跃链接。这个网络非常大，但一般都会得到最好的关键点估计结果。

**ResNet**. Xiao et al. [55] augment a standard residual network [22] with three up-convolutional networks to allow for a higher-resolution output (output stride 4). We first change the channels of the three upsampling layers to 256, 128, 64, respectively, to save computation. We then add one 3 × 3 deformable convolutional layer before each up-convolution with channel 256, 128, 64, respectively. The up-convolutional kernels are initialized as bilinear interpolation. See supplement for a detailed architecture diagram.

**ResNet**。Xiao等[55]将标准的残差网络用三个上卷积网络进行了扩充，以得到更高分辨率的输出（输出步长为4）。我们首先将三个上采样的层的通道数分别变为256，128和64，以节省计算量。然后我们在每个上卷积前增加一个3✖3的可变形卷积层，通道数分别为256，128和64。上卷积核初始化为双线性插值。详见附录的架构图。

**DLA**. Deep Layer Aggregation (DLA) [58] is an image classification network with hierarchical skip connections. We utilize the fully convolutional upsampling version of DLA for dense prediction, which uses iterative deep aggregation to increase feature map resolution symmetrically. We augment the skip connections with deformable convolution [63] from lower layers to the output. Specifically, we replace the original convolution with 3 × 3 deformable convolution at every upsampling layer. See supplement for a detailed architecture diagram.

**DLA**。DLA是一种图像分类网络，有层次化的跳跃链接。我们利用DLA的全卷积上采样版进行密集预测，这使用了迭代的深度聚集来对称的增加特征图分辨率。我们使用可变形卷积来扩充跳跃连接，从较低的层到输出。具体的，我们在每个上采样层中将原始的卷积替代为3✖3的可变形卷积。详见附录的架构图。

We add one 3 × 3 convolutional layer with 256 channel before each output head. A final 1 × 1 convolution then produces the desired output. We provide more details in the supplementary material.

我们在每个输出头之前加上一个256通道的3✖3的卷积层。最终的1✖1卷积生成想要的输出。我们在附加材料中给出详细信息。

**Training**. We train on an input resolution of 512 × 512. This yields an output resolution of 128×128 for all the models. We use random flip, random scaling (between 0.6 to 1.3), cropping, and color jittering as data augmentation, and use Adam [28] to optimize the overall objective. We use no augmentation to train the 3D estimation branch, as cropping or scaling changes the 3D measurements. For the residual networks and DLA-34, we train with a batch-size of 128 (on 8 GPUs) and learning rate 5e-4 for 140 epochs, with learning rate dropped 10× at 90 and 120 epochs, respectively (following [55]). For Hourglass-104, we follow ExtremeNet [61] and use batch-size 29 (on 5 GPUs, with master GPU batch-size 4) and learning rate 2.5e-4 for 50 epochs with 10× learning rate dropped at the 40 epoch. For detection, we finetune the Hourglass-104 from ExtremeNet [61] to save computation. The downsampling layers of Resnet-101 and DLA-34 are initialized with ImageNet pretrain and the upsampling layers are randomly initialized. Resnet-101 and DLA-34 train in 2.5 days on 8 TITAN-V GPUs, while Hourglass-104 requires 5 days.

**训练**。我们训练的输入分辨率为512✖512。对所有模型来说，这生成的输出分辨率为128✖128。我们使用随机翻转、随机尺度变化（0.6到1.3之间）、剪切和色彩抖动作为数据扩充方式，使用Adam来优化总体目标函数。我们对3D估计的分支不使用数据扩充，因为剪切或尺度变化都会影响3D度量。对于残差网络和DLA-34，我们使用批大小128训练（在8个GPU上），学习速率5e-4学习140轮，在第90轮和120轮时学习率分别降低10x（与[55]类似）。对于Hourglass-104，我们按照ExtremeNet[61]的方法，使用的批规模为29（在5个GPUs上，主GPU的批大小为4），以2.5e-4的学习率学习50轮，在第40轮时学习率下降10x。对检测来说，我们精调ExtremeNet中的Hourglass-104以节约计算量。ResNet-101和DLA-34的下采样层使用ImageNet预训练的进行初始化，上采样层是随机初始化的。ResNet-101和DLA-34在8块TITAN-V GPUs上训练了2.5天，而Hourglass需要5天训练。

**Inference**. We use three levels of test augmentations: no augmentation, flip augmentation, and flip and multi-scale (0.5, 0.75, 1, 1.25, 1.5). For flip, we average the network outputs before decoding bounding boxes. For multi-scale, we use NMS to merge results. These augmentations yield different speed-accuracy trade-off, as is shown in the next section.

**推理**。我们使用三级的测试扩充：无扩充，翻转扩充，和翻转及多尺度(0.5, 0.75, 1, 1.25, 1.5)。对于翻转，我们将网络进行平均之后，然后再对边界框进行解码。对于多尺度，我们使用NMS对结果进行融合。这些扩充可以得到不同的速度-准确率折中，如下节所示。

## 6. Experiments 试验

We evaluate our object detection performance on the MS COCO dataset [34], which contains 118k training images (train2017), 5k validation images (val2017) and 20k hold-out testing images (test-dev). We report average precision over all IOU thresholds (AP), AP at IOU thresholds 0.5(AP50) and 0.75 (AP75). The supplement contains additional experiments on PascalVOC [14].

我们在MS COCO数据集上评估我们的目标检测性能，这个数据集包括118k训练图像（train2017），5k验证图像（val2017）和20k保留的测试图像（test-dev）。我们在三种IOU上给出AP，即在所有IOU阈值（AP），在IOU阈值0.5的情况下（AP50），在IOU阈值0.75的情况下（AP75）。附加材料中给出了在PASCAL VOC上的额外试验。

### 6.1. Object detection

Table 1 shows our results on COCO validation with different backbones and testing options, while Figure 1 compares CenterNet with other real-time detectors. The running time is tested on our local machine, with Intel Core i7-8086K CPU, Titan Xp GPU, Pytorch 0.4.1, CUDA 9.0, and CUDNN 7.1. We download code and pretrained models to test run time for each model on the same machine.

表1给出了使用不同骨干网络和测试条件在COCO验证集上的结果，图1比较了CenterNet和其他实时检测器。运行时间是在本机上测试的，运行环境为Intel Core i7-8086K CPU，Titan Xp GPU，PyTorch 0.4.1，CUDA 9.0和CUDNN 7.1。我们在相同的机器上测试每个模型的运行时间。

Table 1: Speed / accuracy trade off for different networks on COCO validation set. We show results without test augmentation (N.A.), flip testing (F), and multi-scale augmentation (MS).

Hourglass-104 achieves the best accuracy at a relatively good speed, with a 42.2% AP in 7.8 FPS. On this backbone, CenterNet outperforms CornerNet [30] (40.6% AP in 4.1 FPS) and ExtremeNet [61](40.3% AP in 3.1 FPS) in both speed and accuracy. The run time improvement comes from fewer output heads and a simpler box decoding scheme. Better accuracy indicates that center points are easier to detect than corners or extreme points.

Hourglass-104以相对较好的速度取得了最佳的准确率，即42.2% AP和7.8 FPS。以这个网络为骨干，CenterNet在速度和准确率上都超过了CornerNet (40.6% AP in 4.1 FPS)和ExtremeNet (40.3% AP in 3.1 FPS)。运行时间的改进是因为输出头更少，边界框解码方案也更简单。更好的准确率说明，中心点比角点或极限点更容易检测。

Using ResNet-101, we outperform RetinaNet [33] with the same network backbone. We only use deformable convolutions in the upsampling layers, which does not affect RetinaNet. We are more than twice as fast at the same accuracy (CenterNet 34.8%AP in 45 FPS (input 512 × 512) vs. RetinaNet 34.4%AP in 18 FPS (input 500 × 800)). Our fastest ResNet-18 model also achieves a respectable performance of 28.1% COCO AP at 142 FPS.

使用ResNet-101，我们超过了使用同样骨干网络的RetinaNet。我们只在上采样层中使用可变形卷积，这不影响RetinaNet。在相同的准确度下，我们的速度比RetinaNet快2倍(CenterNet 34.8%AP in 45 FPS (input 512 × 512) vs. RetinaNet 34.4%AP in 18 FPS (input 500 × 800))。我们最快的ResNet-18模型也取得了不错的性能，28.1% COCO AP at 142 FPS。

DLA-34 gives the best speed/accuracy trade-off. It runs at 52FPS with 37.4%AP. This is more than twice as fast as YOLOv3 [45] and 4.4%AP more accurate. With flip testing, our model is still faster than YOLOv3 [45] and achieves accuracy levels of Faster-RCNN-FPN [46] (CenterNet 39.2% AP in 28 FPS vs Faster-RCNN 39.8% AP in 11 FPS).

DLA-34得到了最好的速度/准确率折中。速度为52 FPS时，AP为37.4%。这是YOLOv3的两倍速度，AP高了4.4%。在翻转测试下，我们的模型仍然比YOLOv3要快，得到的准确率则与Faster-RCNN-FPN类似(CenterNet 39.2% AP in 28 FPS vs Faster-RCNN 39.8% AP in 11 FPS)。

**State-of-the-art comparison**. We compare with other state-of-the-art detectors in COCO test-dev in Table 2. With multi-scale evaluation, CenterNet with Hourglass-104 achieves an AP of 45.1%, outperforming all existing one-stage detectors. Sophisticated two-stage detectors [31,35,48,63] are more accurate, but also slower. There is no significant difference between CenterNet and sliding window detectors for different object sizes or IoU thresholds. CenterNet behaves like a regular detector, just faster.

**与目前最好的比较**。我们与其他目前最好的检测器在COCO test-dev上进行比较，见表2。在多尺度评估下，使用Hourglass-104的CenterNet取得了45.1%的AP，超过了所有现有的单阶段检测器。复杂的两阶段检测器更准确，但也更慢。CenterNet和滑窗法检测器对不同的目标大小，在不同的IOU阈值下，并没有什么明显不同。CenterNet与常规检测器性能类似，但速度更快。

Table 2: State-of-the-art comparison on COCO test-dev. Top: two-stage detectors; bottom: one-stage detectors. We show single-scale / multi-scale testing for most one-stage detectors. Frame-per-second (FPS) were measured on the same machine whenever possible. Italic FPS highlight the cases, where the performance measure was copied from the original publication. A dash indicates methods for which neither code and models, nor public timings were available.

#### 6.1.1 Additional experiments

In unlucky circumstances, two different objects might share the same center, if they perfectly align. In this scenario, CenterNet would only detect one of them. We start by studying how often this happens in practice and put it in relation to missing detections of competing methods.

在一些异常情况下，两个不同的目标可能中心重叠，即它们完全对齐的情况。在这种情况下，CenterNet只能检测到其中一个。我们研究一下实际中这种情况的发生频度，与其他方法的检测结果进行比较。

**Center point collision**. In the COCO training set, there are 614 pairs of objects that collide onto the same center point at stride 4. There are 860001 objects in total, hence CenterNet is unable to predict < 0.1% of objects due to collisions in center points. This is much less than slow- or fast- RCNN miss due to imperfect region proposals [52] (∼ 2%), and fewer than anchor-based methods miss due to insufficient anchor placement [46] (20.0% for Faster-RCNN with 15 anchors at 0.5 IOU threshold). In addition, 715 pairs of objects have bounding box IoU > 0.7 and would be assigned to two anchors, hence a center-based assignment causes fewer collisions.

**中心点冲突**。在COCO训练集中，有614对目标在步长为4时，中心点重叠到了一起。总计有86001个目标，所以由于中心点的冲突，CenterNet无法预测<0.1%的目标。这比slow-或fast-RCNN的错误率要低的多，因为区域建议的瑕疵（～2%），比基于锚框的方法也更少，因为不充足的锚框放置（对Faster-RCNN在0.5 IOU阈值下15锚框下有20%的丢失率）。另外，715对目标其IOU>0.7，被指定给两个锚框，所以基于中心的指定会导致更少的冲突。

**NMS**. To verify that IoU based NMS is not needed for CenterNet, we ran it as a post-processing step on our predictions. For DLA-34 (flip-test), the AP improves from 39.2% to 39.7%. For Hourglass-104, the AP stays at 42.2%. Given the minor impact, we do not use it.

**NMS**。为验证CenterNet不需要基于IoU的NMS，我们在我们预测的基础上进行后处理。对于DLA-34（翻转测试），AP从39.2%提升到39.7%。对于Hourglass-104，AP维持在42.2%不变。影响很小，所以我们不使用后处理。

Next, we ablate the new hyperparameters of our model. All the experiments are done on DLA-34. 下面，我们对模型中新的超参数进行分离试验。所有的试验都是在DLA-34上进行的。

**Training and Testing resolution**. During training, we fix the input resolution to 512 × 512. During testing, we follow CornerNet [30] to keep the original image resolution and zero-pad the input to the maximum stride of the network. For ResNet and DLA, we pad the image with up to 32 pixels, for HourglassNet, we use 128 pixels. As is shown in Table. 3a, keeping the original resolution is slightly better than fixing test resolution. Training and testing in a lower resolution (384 × 384) runs 1.7 times faster but drops 3AP.

**训练和测试分辨率**。在训练时，我们固定输入分辨率为512✖512。在测试时，我们按照CornerNet的思路，保持原有图像分辨率，对输入进行补零，达到网络的最大步长。对于ResNet和DLA，我们对图像补零最多32像素，对Hourglass，我们使用128像素。如图3a所示，保持原有分辨率比固定测试分辨率略好一些。以更低的分辨率（384✖384）训练和测试运行速度快了1.7倍，但AP降低了3%。

**Regression loss**. We compare a vanilla L1 loss to a Smooth L1 [18] for size regression. Our experiments in Table 3c show that L1 is considerably better than Smooth L1. It yields a better accuracy at fine-scale, which the COCO evaluation metric is sensitive to. This is independently observed in keypoint regression [49, 50].

**回归损失**。我们比较了传统的L1损失和Smooth L1损失，进行尺寸回归。我们的试验结果如表3c所示，这显示了L1损失一直要比Smooth L1损失要好。在精细尺度上可以得到更好的准确率，而COCO评估度量标准对这个很敏感。这在关键点回归中也独立的观测到了。

**Bounding box size weight**. We analyze the sensitivity of our approach to the loss weight λsize. Table 3b shows 0.1 gives a good result. For larger values, the AP degrades significantly, due to the scale of the loss ranging from 0 to output size w/R or h/R, instead of 0 to 1. However, the value does not degrade significantly for lower weights.

**边界框大小权重**。我们分析我们的方法对损失权重λsize的敏感性。表3b说明0.1的取值可以得到不错的效果。更大的值情况下，AP会明显下降，因为损失的尺度范围是从0到输出大小w/R或h/R，而不是0到1。但是，对于更低的权重，这个值不会下降的很明显。

**Training schedule**. By default, we train the keypoint estimation network for 140 epochs with a learning rate drop at 90 epochs. If we double the training epochs before dropping the learning rate, the performance further increases by 1.1 AP (Table 3d), at the cost of a much longer training schedule. To save computational resources (and polar bears), we use 140 epochs in ablation experiments, but stick with 230 epochs for DLA when comparing to other methods.

**训练方案**。默认情况下，我们训练关键点估计网络140轮，学习速率在第90轮时候衰减。如果我们在学习速率降低前训练轮数加倍，那么性能会进一步增加1.1 AP（表3d），而代价则是训练时间要长很多。为节省计算资源，我们在分离对比试验中使用140轮的训练，但在与其他方法比较时，DLA还是使用230轮的训练。

Finally, we tried a multiple “anchor” version of CenterNet by regressing to more than one object size. The experiments did not yield any success. See supplement. 最后，我们尝试轮一个多“锚”版本的CenterNet，回归的目标大小多于一个。试验没有得到成功的效果。见补充材料。

### 6.2. 3D detection

We perform 3D bounding box estimation experiments on KITTI dataset [17], which contains carefully annotated 3D bounding box for vehicles in a driving scenario. KITTI contains 7841 training images and we follow standard training and validation splits in literature [10, 54]. The evaluation metric is the average precision for cars at 11 recalls (0.0 to 1.0 with 0.1 increment) at IOU threshold 0.5, as in object detection [14]. We evaluate IOUs based on 2D bounding box (AP), orientation (AOP), and Bird-eye-view bounding box (BEV AP). We keep the original image resolution and pad to 1280 × 384 for both training and testing. The training converges in 70 epochs, with learning rate dropped at the 45 and 60 epoch, respectively. We use the DLA-34 backbone and set the loss weight for depth, orientation, and dimension to 1. All other hyper-parameters are the same as the detection experiments.

我们在KITTI数据集上进行3D边界框估计试验，其中包含了车辆驾驶场景中仔细标注的3D边界框。KITTI包含7841幅训练图像，我们采用文献中标准的训练和验证分隔。评估标准是11种召回下的车辆的AP（0.0到1.0，步长0.1），IOU阈值为0.5，和目标检测中一样。我们评估基于IOU的2D边界框（AP），方向（AOP），以及鸟视边界框（BEV AP）。我们保持原始图像分辨率，并补零到1280✖384以进行训练和测试。训练在70轮训练后收敛，学习率分别在45轮和60轮时下降。我们使用DLA-34骨干网络，设置深度、方向和维度的损失权重为1。所有其他超参数都与检测试验一样。

Since the number of recall thresholds is quite small, the validation AP fluctuates by up to 10% AP. We thus train 5 models and report the average with standard deviation.

由于召回阈值的数量非常小，验证AP向上浮动轮10%的AP。所以我们训练了5个模型，然后给出其在标准偏差下的平均值。

We compare with slow-RCNN based Deep3DBox [38] and Faster-RCNN based method Mono3D [9], on their specific validation split. As is shown in Table 4, our method performs on-par with its counterparts in AP and AOS and does slightly better in BEV. Our CenterNet is two orders of magnitude faster than both methods.

我们与基于slow-RCNN的Deep3DBox、基于Faster-RCNN的Mono3D，在其特定验证集分隔上进行了比较。如表4所示，我们的方法与其对手在AP和AOS上表现类似，在BEV上表现略好。我们的CenterNet比这两种方法都快了2个数量级。

Table 4: KITTI evaluation. We show 2D bounding box AP, average orientation score (AOS), and bird eye view (BEV) AP on different validation splits. Higher is better.

### 6.3. Pose estimation

Finally, we evaluate CenterNet on human pose estimation in the MS COCO dataset [34]. We evaluate keypoint AP, which is similar to bounding box AP but replaces the bounding box IoU with object keypoint similarity. We test and compare with other methods on COCO test-dev.

最后，我们在MS COCO数据集上评估了CenterNet在人体姿态估计上的效果。我们评估了关键点AP，这与边界框AP类似，但将边界框IOU替换为了目标关键点相似度。我们测试并与其他方法在COCO test-dev上进行了比较。

We experiment with DLA-34 and Hourglass-104, both fine-tuned from center point detection. DLA-34 converges in 320 epochs (about 3 days on 8GPUs) and Hourglass-104 converges in 150 epochs (8 days on 5 GPUs). All additional loss weights are set to 1. All other hyper-parameters are the same as object detection.

我们用DLA-34和Hourglass-104进行了试验，都是从中心点检测的基础上精调的。DLA-34在320轮训练后收敛（在8个GPU上耗时3天），Hourglass-104在150轮训练后收敛（在5个GPU上训练8天）。所有额外的损失权重都设为1。所有其他超参数都与目标检测相似。

The results are shown in Table 5. Direct regression to keypoints performs reasonably, but not at state-of-the-art. It struggles particularly in high IoU regimes. Projecting our output to the closest joint detection improves the results throughout, and performs competitively with state-of-the- art multi-person pose estimators [4, 21, 39, 41]. This verifies that CenterNet is general, easy to adapt to a new task.

结果如表5所示。直接回归到关键点效果不错，但并不是目前最好的。在高IOU区域中尤其不太好。将我们的输出投影到最接近的关节点检测中，会改进结果，与目前最好的多人姿态估计器表现类似。这验证轮CenterNet是通用的，容易在新任务中适应。

Figure 5 shows qualitative examples on all tasks. 图5给出轮在所有任务中的定性例子。

Figure 5: Qualitative results. All images were picked thematically without considering our algorithms performance. First row: object detection on COCO validation. Second and third row: Human pose estimation on COCO validation. For each pair, we show the results of center offset regression (left) and heatmap matching (right). fourth and fifth row: 3D bounding box estimation on KITTI validation. We show projected bounding box (left) and bird eye view map (right). The ground truth detections are shown in solid red solid box. The center heatmap and 3D boxes are shown overlaid on the original image.

Table 5: Keypoint detection on COCO test-dev. -reg/ -jd are for direct center-out offset regression and matching regres- sion to the closest joint detection, respectively. The results are shown in COCO keypoint AP. Higher is better.

## 7. Conclusion

In summary, we present a new representation for objects: as points. Our CenterNet object detector builds on successful keypoint estimation networks, finds object centers, and regresses to their size. The algorithm is simple, fast, accurate, and end-to-end differentiable without any NMS post-processing. The idea is general and has broad applications beyond simple two-dimensional detection. CenterNet can estimate a range of additional object properties, such as pose, 3D orientation, depth and extent, in one single forward pass. Our initial experiments are encouraging and open up a new direction for real-time object recognition and related tasks.

总结起来，我们提出了一种新的目标表示方法：表示为点。我们的CenterNet目标检测器在关键点估计网络上构建起来，找到目标中心点，并回归到其大小。算法简单快速准确，端到端可微分，不需要NMS后处理。这种思想是一般性的，在简单的二维检测之外还有很多应用。CenterNet可以估计很多其他额外目标属性，如姿态，3D方向，深度和广度，并在一个前向过程中估计得到。我们初始的试验是鼓舞人心的，开启了实时目标识别和相关任务的新方向。