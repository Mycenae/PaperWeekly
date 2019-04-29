# Feature Pyramid Networks for Object Detection

Tsung-Yi Lin Kaiming He et al. Facebook AI Research (FAIR)

## Abstract 摘要

Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications. Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art single-model results on the COCO detection benchmark without bells and whistles, surpassing all existing single-model entries including those from the COCO 2016 challenge winners. In addition, our method can run at 6 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection. Code will be made publicly available.

特征金字塔是检测不同尺度上目标的识别系统中的基本组件。但最近的深度学习目标检测器避免使用金字塔表示，部分是因为计算量和存储都很大。在本文中，我们探索了利用深度卷积网络内在的多尺度性和金字塔层级结构来构建特征金字塔，使用极少的额外开销。我们提出了一种带有横向连接的自上而下架构，构建了所有尺度上的高层次语义特征图。这种架构，我们称为特征金字塔网络(FPN)，在几种应用中都表现出了作为通用特征提取器的显著改进。在基本的Faster R-CNN系统使用FPN，这种方法在COCO检测基准测试中得到了目前最好的单模型结果，超过了所有现有的单模型结果，包括COCO 2016挑战赛的胜利者。另外，我们的方法在GPU上的速度为6 FPS，所以是多尺度目标检测器的实用准确的解决方案。代码将会开源。

## 1. Introduction 引言

Recognizing objects at vastly different scales is a fundamental challenge in computer vision. Feature pyramids built upon image pyramids (for short we call these featurized image pyramids) form the basis of a standard solution [1] (Fig. 1(a)). These pyramids are scale-invariant in the sense that an object’s scale change is offset by shifting its level in the pyramid. Intuitively, this property enables a model to detect objects across a large range of scales by scanning the model over both positions and pyramid levels.

计算机视觉中的一个基础挑战是，在非常多不同的尺度上识别出物体。在图像金字塔上构建出的特征金字塔（简化起见，我们称之为特征化的图像金字塔）形成了标准解决方案的基础[1]（图1(a)）。这些金字塔是对尺度不变的，因为目标的尺度变化，不过是在金字塔中改变了层级。直觉上，这种性质使一个模型可以检测很多尺度上的目标，只要在不同金字塔层级和空间位置上进行扫描。

Featurized image pyramids were heavily used in the era of hand-engineered features [5, 25]. They were so critical that object detectors like DPM [7] required dense scale sampling to achieve good results (e.g., 10 scales per octave). For recognition tasks, engineered features have largely been replaced with features computed by deep convolutional networks (ConvNets) [19, 20]. Aside from being capable of representing higher-level semantics, ConvNets are also more robust to variance in scale and thus facilitate recognition from features computed on a single input scale [15, 11, 29] (Fig. 1(b)). But even with this robustness, pyramids are still needed to get the most accurate results. All recent top entries in the ImageNet [33] and COCO [21] detection challenges use multi-scale testing on featurized image pyramids (e.g., [16, 35]). The principle advantage of featurizing each level of an image pyramid is that it produces a multi-scale feature representation in which all levels are semantically strong, including the high-resolution levels.

特征化的图像金字塔在手工设计特征的时代用的非常多[5,25]。特征金字塔非常关键，像DPM[7]这样的目标检测器需要密集尺度取样，才可以得到好的结果（如，每两倍分辨率需要10个尺度）。对于识别任务，设计的特征基本都被深度卷积网络(ConvNets)计算出的特征替代了[19,20]。除了可以表示更高层的语义，ConvNets也对尺度变化更稳健，所以可以识别在单个尺度输入上计算出来的特征[15,11,29]（图1(b)）。即使有这种稳健性，仍然需要金字塔来得到最准确的结果。所有最近在ImageNet[33]和COCO[22]检测挑战上的最好成绩，都使用了特征化图像金字塔上的多尺度测试（如[16,35]）。图像金字塔每个层次上的特征化的主要好处是，生成了多尺度特征表示，其中所有的层次都是语义强壮的，包括高分辨率层次。

Figure 1. (a) Using an image pyramid to build a feature pyramid. Features are computed on each of the image scales independently, which is slow. (b) Recent detection systems have opted to use only single scale features for faster detection. (c) An alternative is to reuse the pyramidal feature hierarchy computed by a ConvNet as if it were a featurized image pyramid. (d) Our proposed Feature Pyramid Network (FPN) is fast like (b) and (c), but more accurate. In this figure, feature maps are indicate by blue outlines and thicker outlines denote semantically stronger features.

图1. (a)使用图像金字塔来构建特征金字塔。特征是在每个图像尺度上单独计算的，这样速度很慢；(b)最近的检测系统选择采用单尺度特征进行更快的检测；(c)另一种选择是重复使用ConvNet计算出的金字塔特征层次，就像是特征化的图像金字塔；(d)我们提出的特征金字塔网络(FPN)与(b)和(c)一样快速，但更准确。本图中，特征图用蓝色外框表示，更粗的框表示语义更强的特征。

Nevertheless, featurizing each level of an image pyramid has obvious limitations. Inference time increases considerably (e.g., by four times [11]), making this approach impractical for real applications. Moreover, training deep networks end-to-end on an image pyramid is infeasible in terms of memory, and so, if exploited, image pyramids are used only at test time [15, 11, 16, 35], which creates an inconsistency between train/test-time inference. For these reasons, Fast and Faster R-CNN [11, 29] opt to not use featurized image pyramids under default settings.

尽管如此，将图像金字塔的每一层都特征化有着明显的限制。推理时间增加很多（如，4倍[11]），使这种方法对实际应用来说不实用。而且，在图像金字塔上进行端到端的深度网络训练，在内存上是不可行的；图像金字塔是只在测试时[15,11,16,35]使用的，这在训练/测试时推理上产生了不一致性。因为这些原因，Fast R-CNN和Faster R-CNN[11,29]选择在默认情况下不使用特征化的图像金字塔。

However, image pyramids are not the only way to compute a multi-scale feature representation. A deep ConvNet computes a feature hierarchy layer by layer, and with subsampling layers the feature hierarchy has an inherent multiscale, pyramidal shape. This in-network feature hierarchy produces feature maps of different spatial resolutions, but introduces large semantic gaps caused by different depths. The high-resolution maps have low-level features that harm their representational capacity for object recognition.

但是，图像金字塔不是计算多尺度特征表示的唯一途径。深度ConvNet逐层计算特征层次，而因为有下采样层，所以特征层次有内在的多尺度、金字塔形状。这种网络中的特征层次生成了不同空间分辨率的特征图，但不同的深度造成很大的空白。高分辨率特征图有底层特征，这对目标检测的表示能力是有害的。

The Single Shot Detector (SSD) [22] is one of the first attempts at using a ConvNet’s pyramidal feature hierarchy as if it were a featurized image pyramid (Fig. 1(c)). Ideally, the SSD-style pyramid would reuse the multi-scale feature maps from different layers computed in the forward pass and thus come free of cost. But to avoid using low-level features SSD foregoes reusing already computed layers and instead builds the pyramid starting from high up in the network (e.g., conv4_3 of VGG nets [36]) and then by adding several new layers. Thus it misses the opportunity to reuse the higher-resolution maps of the feature hierarchy. We show that these are important for detecting small objects.

SSD[22]是第一个使用ConvNet的金字塔特征层次的，将其当作特征化的图像金字塔（图1(c)）。理想情况下，SSD类的金字塔会重复使用前向过程中不同层上的多尺度特征图，所以是没有计算代价的。但为了避免使用低层特征，SSD并没有重复使用已经计算的层，而是从网络中很高的层开始构建这个金字塔（如VGG网络的conv4_3），然后增加几个新层，所以其错过了重复使用特征层次中较高分辨率的特征图的机会。我们会证明，这对于检测小目标是非常重要的。

The goal of this paper is to naturally leverage the pyramidal shape of a ConvNet’s feature hierarchy while creating a feature pyramid that has strong semantics at all scales. To achieve this goal, we rely on an architecture that combines low-resolution, semantically strong features with high-resolution, semantically weak features via a top-down pathway and lateral connections (Fig. 1(d)). The result is a feature pyramid that has rich semantics at all levels and is built quickly from a single input image scale. In other words, we show how to create in-network feature pyramids that can be used to replace featurized image pyramids without sacrificing representational power, speed, or memory.

本文的目标是很自然的利用ConvNet的特征层次的金字塔形状，同时生成一个在所有层次上都有很强语义的特征金字塔。为达到这个目标，我们使用的这个架构，通过自上而下的通道和横向连接（图1(d)），综合了低分辨率、语义强的特征，和高分辨率，语义上弱的特征。这个结果是一个在所有层次上都有丰富语义的特征金字塔，而且是从输入的单幅图像尺度上快速构建的。换句话说，我们给出的是如何构建网络中的特征金字塔，这个金字塔可以用于替换特征化的图像金字塔，而且不用牺牲表示能力、速度或内存。

Similar architectures adopting top-down and skip connections are popular in recent research [28, 17, 8, 26]. Their goals are to produce a single high-level feature map of a fine resolution on which the predictions are to be made (Fig. 2 top). On the contrary, our method leverages the architecture as a feature pyramid where predictions (e.g., object detections) are independently made on each level (Fig. 2 bottom). Our model echoes a featurized image pyramid, which has not been explored in these works.

类似的采用自上而下和跳跃连接的架构在最近的研究中很流行[28,17,8,26]。其目标是生成单个高层次高分辨率特征图，然后在其上进行预测（图2上）。相反，我们的方法将架构用作特征金字塔，在每个层次上独自进行预测（如，目标检测）（图2下）。我们的模型与特征化的图像金字塔对应，这在这些文献中还没有进行探索。

We evaluate our method, called a Feature Pyramid Network (FPN), in various systems for detection and segmentation [11, 29, 27]. Without bells and whistles, we report a state-of-the-art single-model result on the challenging COCO detection benchmark [21] simply based on FPN and a basic Faster R-CNN detector [29], surpassing all existing heavily-engineered single-model entries of competition winners. In ablation experiments, we find that for bounding box proposals, FPN significantly increases the Average Recall (AR) by 8.0 points; for object detection, it improves the COCO-style Average Precision (AP) by 2.3 points and PASCAL-style AP by 3.8 points, over a strong single-scale baseline of Faster R-CNN on ResNets [16]. Our method is also easily extended to mask proposals and improves both instance segmentation AR and speed over state-of-the-art methods that heavily depend on image pyramids.

我们的方法称为特征金字塔网络(FPN)，我们在各种检测和分割系统中进行了评价[11,29,27]。我们仅仅基于FPN和基础Faster R-CNN检测器[29]，就在COCO检测基准测试[21]上给出了目前最好的单模型结果，超过所有现有的重设计的单模型结果。在分离实验中，我们发现在边界框建议上，FPN将平均召回(AR)显著提升了8.0个百分点；对于目标检测，以COCO类的AP作为衡量，改进了2.3个百分点，PASCAL类的AP来衡量，改进了3.8个百分点，这是以一个ResNet Faster R-CNN[16]的强基准进行比较的。我们的方法也很容易拓展到掩膜建议上，改进了实例分割的AR和速度，现有的最好方法都非常依赖于图像金字塔。

In addition, our pyramid structure can be trained end-to-end with all scales and is used consistently at train/test time, which would be memory-infeasible using image pyramids. As a result, FPNs are able to achieve higher accuracy than all existing state-of-the-art methods. Moreover, this improvement is achieved without increasing testing time over the single-scale baseline. We believe these advances will facilitate future research and applications. Our code will be made publicly available.

另外，我们的金字塔结构可以在所有尺度上进行端到端的训练，在训练/测试时一致的使用，如果是使用图像金字塔，则在内存上是不可行的。结果是，FPN可以比所有现有的最好方法都得到更高的准确率。而且，这个改进不会比单尺度基准提高测试时间。我们相信，这些进展会使将来的研究和应用更方便。代码将会开源。

Figure 2. Top: a top-down architecture with skip connections, where predictions are made on the finest level (e.g., [28]). Bottom: our model that has a similar structure but leverages it as a feature pyramid, with predictions made independently at all levels.

图2. 上：有跳跃连接的自上而下的架构，预测是在最高分辨率的层进行（如[28]）。下：我们的模型有着类似的结构，但将其用作特征金字塔，在所有层次上独立进行预测。

## 2. Related Work 相关的工作

**Hand-engineered features and early neural networks**. SIFT features [25] were originally extracted at scale-space extrema and used for feature point matching. HOG features [5], and later SIFT features as well, were computed densely over entire image pyramids. These HOG and SIFT pyramids have been used in numerous works for image classification, object detection, human pose estimation, and more. There has also been significant interest in computing featurized image pyramids quickly. Dollár et al. [6] demonstrated fast pyramid computation by first computing a sparsely sampled (in scale) pyramid and then interpolating missing levels. Before HOG and SIFT, early work on face detection with ConvNets [38, 32] computed shallow networks over image pyramids to detect faces across scales.

**手工设计的特征和早期神经网络**。SIFT特征[25]最早是用于提取尺度空间的极值，并用于特征点匹配。HOG特征[5]，以及后来的SIFT特征，都是在整个图像金字塔上密集计算得到的。这些HOG和SIFT金字塔在非常多的图像分类、目标检测、人体姿态估计等等工作中得到了使用。计算特征化的图像金字塔也有非常多的工作。Dollár等[6]给出了快速金字塔计算，首先计算一个稀疏采样（在尺度上的）金字塔，然后对丢失的层次进行插值。在HOG和SIFT之前，早期的使用ConvNets[38,32]的人脸检测工作在图像金字塔上计算浅层网络来在各个尺度上检测人脸。

**Deep ConvNet object detectors**. With the development of modern deep ConvNets [19], object detectors like OverFeat [34] and R-CNN [12] showed dramatic improvements in accuracy. OverFeat adopted a strategy similar to early neural network face detectors by applying a ConvNet as a sliding window detector on an image pyramid. R-CNN adopted a region proposal-based strategy [37] in which each proposal was scale-normalized before classifying with a ConvNet. SPPnet [15] demonstrated that such region-based detectors could be applied much more efficiently on feature maps extracted on a single image scale. Recent and more accurate detection methods like Fast R-CNN [11] and Faster R-CNN [29] advocate using features computed from a single scale, because it offers a good trade-off between accuracy and speed. Multi-scale detection, however, still performs better, especially for small objects.

**深度ConvNet目标检测器**。随着现代深度ConvNets[19]的发展，目标检测器像OverFeat[34]和R-CNN[12]显著的改进了准确率。OverFeat采用了与早期神经网络人脸检测器的类似的策略，将ConvNet在图像金字塔上作为滑窗检测器使用。R-CNN采用了一个基于区域建议的策略[37]，其中每个建议都是尺度归一化的，然后用ConvNet进行分类。SPPnet[15]证明了，这种基于区域的检测器可以在单图像尺度上提取出来的特征图上更高效的运行。最近更准确的检测方法，像Fast R-CNN[11]和Faster R-CNN[29]提倡使用单尺度上计算得到的特征，因为可以在准确率和速度上得到很好的折中。但多尺度检测仍然表现的更好，尤其对于小目标来说。

**Methods using multiple layers**. A number of recent approaches improve detection and segmentation by using different layers in a ConvNet. FCN [24] sums partial scores for each category over multiple scales to compute semantic segmentations. Hypercolumns [13] uses a similar method for object instance segmentation. Several other approaches (HyperNet [18], ParseNet [23], and ION [2]) concatenate features of multiple layers before computing predictions, which is equivalent to summing transformed features. SSD [22] and MS-CNN [3] predict objects at multiple layers of the feature hierarchy without combining features or scores.

**使用多层的方法**。一些最近的方法，使用ConvNet中的不同层，来改进检测和分割结果。FCN[24]在多个尺度上对每个类别的部分分数相加，计算得到语义分割结果。Hypercolumns[13]使用类似的方法进行目标实例分割。几种其他方法(HpyerNet[18], ParseNet[23], ION[2])将不同层的特征拼接起来，然后计算预测，这与变换特征求和是等价的。SSD[22]和MS-CNN[3]在多个尺度的特征层次上预测目标，而没有将特征或分数组合起来。

There are recent methods exploiting lateral/skip connections that associate low-level feature maps across resolutions and semantic levels, including U-Net [31] and Sharp-Mask [28] for segmentation, Recombinator networks [17] for face detection, and Stacked Hourglass networks [26] for keypoint estimation. Ghiasi et al. [8] present a Laplacian pyramid presentation for FCNs to progressively refine segmentation. Although these methods adopt architectures with pyramidal shapes, they are unlike featurized image pyramids [5, 7, 34] where predictions are made independently at all levels, see Fig. 2. In fact, for the pyramidal architecture in Fig. 2 (top), image pyramids are still needed to recognize objects across multiple scales [28].

也有最近的方法探索横向连接和跳跃连接的作用，将不同分辨率和语义层次的低层特征图关联起来，包括分割模型的U-Net[31]和Sharp-Mask[28]，人脸检测的Recominator networks[17]，和关键点估计的Stacked Hourglass networks[26]。Ghiasi等[8]提出了在FCN中使用Laplacian金字塔表示，以渐近的提炼分割结果。虽然这些方法采用了金字塔形状的架构，但与特征化的图像金字塔[5,7,34]还不一样，那些文章中预测是在所有的层中独立进行的，见图2。实际上，对于图2（上）中的金字塔架构，仍然需要图像金字塔来在不同尺度上识别目标[28]。

## 3. Feature Pyramid Networks

Our goal is to leverage a ConvNet’s pyramidal feature hierarchy, which has semantics from low to high levels, and build a feature pyramid with high-level semantics throughout. The resulting Feature Pyramid Network is general-purpose and in this paper we focus on sliding window proposers (Region Proposal Network, RPN for short) [29] and region-based detectors (Fast R-CNN) [11]. We also generalize FPNs to instance segmentation proposals in Sec. 6.

我们的目标是利用ConvNet的金字塔特征层次，包括从低层到高层的语义，来构建一个特征金字塔，各个层次上都有高水平的语义。得到的FPN是通用目标的，本文中，我们关注的是滑窗建议(Region Proposal Network, RPN)[29]和基于区域的检测器(Fast R-CNN)[11]。我们还在第6部分中将FPN推广用于实例分割建议。

Our method takes a single-scale image of an arbitrary size as input, and outputs proportionally sized feature maps at multiple levels, in a fully convolutional fashion. This process is independent of the backbone convolutional architectures (e.g., [19, 36, 16]), and in this paper we present results using ResNets [16]. The construction of our pyramid involves a bottom-up pathway, a top-down pathway, and lateral connections, as introduced in the following.

我们的方法以任意大小的单尺度图像作为输入，在多个层次上，以全卷积的形式，输出对应大小的特征图。这个过程与骨干卷积架构独立（如[19,36,16]），本文中，我们给出使用ResNets的结果[16]。我们的金字塔的构建包括一个自下而上的通道，一个自上而下的通道，横向连接，下面会逐一介绍。

**Bottom-up pathway**. The bottom-up pathway is the feed-forward computation of the backbone ConvNet, which computes a feature hierarchy consisting of feature maps at several scales with a scaling step of 2. There are often many layers producing output maps of the same size and we say these layers are in the same network stage. For our feature pyramid, we define one pyramid level for each stage. We choose the output of the last layer of each stage as our reference set of feature maps, which we will enrich to create our pyramid. This choice is natural since the deepest layer of each stage should have the strongest features.

**自下而上的通道**。自下而上的通道是骨干ConvNet的前向计算，计算的特征层次，包括多个尺度上的特征图，尺度步长为2。通常有很多层生成同样大小的输出图，我们称这些层是在同样的网络阶段中。对于我们的特征金字塔上，我们对每个阶段都定义一个金字塔层次。我们选择每个阶段最后一层的输出作为特征图的参考集，我们会充实以形成我们的金字塔。这个选择是很自然的，因为每个阶段最深的层应当具有最强的特征。

Specifically, for ResNets [16] we use the feature activations output by each stage’s last residual block. We denote the output of these last residual blocks as {$C_2, C_3, C_4, C_5$} for conv2, conv3, conv4, and conv5 outputs, and note that they have strides of {4, 8, 16, 32} pixels with respect to the input image. We do not include conv1 into the pyramid due to its large memory footprint.

特别的，对于ResNets[16]我们使用每个阶段最后的残差模块的特征激活的输出。我们将这些最后的残差模块表示为{$C_2, C_3, C_4, C_5$}，表示的是conv2, conv3, conv4, and conv5的输出，对于输入图像来说，它们的步长为{4, 8, 16, 32}像素。我们在金字塔中没有包含conv1，因为占用的内存很大。

**Top-down pathway and lateral connections**. The top-down pathway hallucinates higher resolution features by upsampling spatially coarser, but semantically stronger, feature maps from higher pyramid levels. These features are then enhanced with features from the bottom-up pathway via lateral connections. Each lateral connection merges feature maps of the same spatial size from the bottom-up path-way and the top-down pathway. The bottom-up feature map is of lower-level semantics, but its activations are more accurately localized as it was subsampled fewer times.

**自上而下的通道和横向连接**。更高层次的金字塔中的特征图空间分辨率更粗糙，但是语义上更强，自上而下的通道对这些特征图进行上采样，可以得到更高分辨率的特征。这些特征通过自下而上通道中的横向连接可以得到增强。每个横向连接，将自下而上通道中和自上而下通道中，同样空间大小的特征图进行合并。自下而上的特征图的语义特征没那么强，但其激活定位更准确，因为其经过的下采样的次数更少。

Fig. 3 shows the building block that constructs our top-down feature maps. With a coarser-resolution feature map, we upsample the spatial resolution by a factor of 2 (using nearest neighbor upsampling for simplicity). The upsampled map is then merged with the corresponding bottom-up map (which undergoes a 1×1 convolutional layer to reduce channel dimensions) by element-wise addition. This process is iterated until the finest resolution map is generated. To start the iteration, we simply attach a 1×1 convolutional layer on $C_5$ to produce the coarsest resolution map. Finally, we append a 3×3 convolution on each merged map to generate the final feature map, which is to reduce the aliasing effect of upsampling. This final set of feature maps is called {$P_2, P_3, P_4, P_5$}, corresponding to {$C_2, C_3, C_4, C 5$} that are respectively of the same spatial sizes.

图3给出了构建我们自上而下的特征图的基本模块。有了低分辨率的特征图之后，我们对空间分辨率进行因子为2的上采样（简化起见，使用最近邻上采样）。上采样的特征图与对应的自下而上的特征图（经过了1×1卷积层以减少通道维数），通过逐元素相加合并到一起。这个过程迭代进行，直到生成最精细分辨率的特征图。这个迭代的开始，我们简单的在$C_5$上附加上1×1卷积层，以生成最粗糙分辨率的特征图。最后，我们在每个合并的特征图上接上一个3×3卷积，以生成最后的特征图，这是为了减少上采样的混叠效应。

Because all levels of the pyramid use shared classifiers/regressors as in a traditional featurized image pyramid, we fix the feature dimension (numbers of channels, denoted as d) in all the feature maps. We set d = 256 in this paper and thus all extra convolutional layers have 256-channel outputs. There are no non-linearities in these extra layers, which we have empirically found to have minor impacts.

在传统的特征化图像金字塔中，所有层使用共享的分类器/回归器，我们的金字塔中也使用一样的方案，将特征维度（通道数量，表示为d）在所有特征图中固定。本文中我们设d=256，所以所有额外的卷积层都有256维的输出。在这些额外的层中没有非线性处理，这是我们通过试验经验发现的，影响很小。

Simplicity is central to our design and we have found that our model is robust to many design choices. We have experimented with more sophisticated blocks (e.g., using multi-layer residual blocks [16] as the connections) and observed marginally better results. Designing better connection modules is not the focus of this paper, so we opt for the simple design described above.

我们设计的中心思想是简洁，我们发现模型对很多设计选择都很稳健。我们用更复杂的模块也进行过试验（如，使用多层残差模块[16]作为连接），观察到的是略微更好的结果。设计更好的连接模块不是本文的注意点，所以我们选择上述的简洁设计。

## 4. Applications 应用

Our method is a generic solution for building feature pyramids inside deep ConvNets. In the following we adopt our method in RPN [29] for bounding box proposal generation and in Fast R-CNN [11] for object detection. To demonstrate the simplicity and effectiveness of our method, we make minimal modifications to the original systems of [29, 11] when adapting them to our feature pyramid.

我们的方法是在深度卷积网络中构建特征金字塔的一个通用解决方案。下面我们在RPN[29]中采用我们的方法，进行边界框建议生成，并在Fast R-CNN[11]中采用我们的方法，进行目标检测。为展现我们方法的简洁性和有效性，我们对原系统[11,29]进行最小的修正，以适应我们的特征金字塔。

### 4.1. Feature Pyramid Networks for RPN

RPN [29] is a sliding-window class-agnostic object detector. In the original RPN design, a small subnetwork is evaluated on dense 3×3 sliding windows, on top of a single-scale convolutional feature map, performing object/non-object binary classification and bounding box regression. This is realized by a 3×3 convolutional layer followed by two sibling 1×1 convolutions for classification and regression, which we refer to as a network head. The object/non-object criterion and bounding box regression target are defined with respect to a set of reference boxes called anchors [29]. The anchors are of multiple pre-defined scales and aspect ratios in order to cover objects of different shapes.

RPN[29]是一个滑窗的类别无关的目标检测器。在RPN的原始设计中，在密集的3×3滑动窗口中运行一个小的子网络，在一个单尺度卷积特征图上，进行目标/非目标的二值分类和边界框回归。这是由3×3卷积层和2个并行分别用于分类和回归的1×1卷积实现的，我们称之为网络头。目标/非目标规则和边界框回归目标是关于参考框集即锚框[29]定义的。锚框是有多个预定义的尺度和纵横比的，以覆盖不同形状的目标。

We adapt RPN by replacing the single-scale feature map with our FPN. We attach a head of the same design (3×3 conv and two sibling 1×1 convs) to each level on our feature pyramid. Because the head slides densely over all locations in all pyramid levels, it is not necessary to have multi-scale anchors on a specific level. Instead, we assign anchors of a single scale to each level. Formally, we define the anchors to have areas of {$32^2, 64^2, 128^2, 256^2, 512^2$} pixels on {$P_2, P_3, P_4, P_5, P_6$} respectively (Here we introduce $P_6$ only for covering a larger anchor scale of $512^2$. $P_6$ is simply a stride two subsampling of $P_5$. $P_6$ is not used by the Fast R-CNN detector in the next section). As in [29] we also use anchors of multiple aspect ratios {1:2, 1:1, 2:1} at each level. So in total there are 15 anchors over the pyramid.

我们调整了RPN，因为要将单尺度特征图替换为我们的FPN。我们将同样设计的网络头（3×3卷积和2个并行的1×1卷积）接入我们的特征金字塔的每一层。由于头部在所有金字塔层级上密集滑过所有位置，所以特定层上的多尺度锚框就没必要了，我们给每个层指定单尺度锚框。正式的，我们定义锚框在{$P_2, P_3, P_4, P_5, P_6$}特征图上分别有{$32^2, 64^2, 128^2, 256^2, 512^2$}的像素面积（这里我们引入了$P_6$，只是为了覆盖更大的锚框尺度$512^2$，$P_6$是$P_5$的步长为2的下采样。$P_6$在下一节的Fast R-CNN中没有使用）。和[29]中一样，我们也在每个层次上都使用多个纵横比的锚框{1:2, 1:1, 2:1}。所以在这个金字塔上，我们有总计15种锚框。

We assign training labels to the anchors based on their Intersection-over-Union (IoU) ratios with ground-truth bounding boxes as in [29]. Formally, an anchor is assigned a positive label if it has the highest IoU for a given ground-truth box or an IoU over 0.7 with any ground-truth box, and a negative label if it has IoU lower than 0.3 for all ground-truth boxes. Note that scales of ground-truth boxes are not explicitly used to assign them to the levels of the pyramid; instead, ground-truth boxes are associated with anchors, which have been assigned to pyramid levels. As such, we introduce no extra rules in addition to those in [29].

基于与真值边界框的IoU比率，我们为锚框指定训练标签，和[29]中一样。如果与任何真值边界框的IoU最高，或大于0.7，那么一个锚框就指定一个正标签；如果与所有的真值框的IoU都小于0.3，就指定一个负标签。注意，真值边界框的尺度没有显式的用于指定给金字塔的层次；而是，真值边界框与锚框关联在一起，锚框指定给特定的金字塔层次。这样，与[29]相比，我们就没有引入了更多额外的规则。

We note that the parameters of the heads are shared across all feature pyramid levels; we have also evaluated the alternative without sharing parameters and observed similar accuracy. The good performance of sharing parameters indicates that all levels of our pyramid share similar semantic levels. This advantage is analogous to that of using a featurized image pyramid, where a common head classifier can be applied to features computed at any image scale.

我们指出，网络头的参数在所有特征金字塔层之间共享；我们还评估了不共享参数的选项，得到了类似的准确率。共享参数的好的表现说明，我们金字塔的所有层级共享类似的语义水平。这种优点与使用特征化的图像金字塔类似，即使用通用的头分类器可用于任何图像尺度上计算得到的特征。

With the above adaptations, RPN can be naturally trained and tested with our FPN, in the same fashion as in [29]. We elaborate on the implementation details in the experiments. 在上述改变下，RPN可以用于很自然的用我们的FPN进行训练和测试，与[29]中的风格一样。我们在试验中详述其实现细节。

### 4.2. Feature Pyramid Networks for Fast R-CNN

Fast R-CNN [11] is a region-based object detector in which Region-of-Interest (RoI) pooling is used to extract features. Fast R-CNN is most commonly performed on a single-scale feature map. To use it with our FPN, we need to assign RoIs of different scales to the pyramid levels.

Fast R-CNN[11]是一个基于区域的目标检测器，其中使用了RoI pooling来提取特征。Fast R-CNN通常在一个单尺度的特征图上进行。为使用我们的FPN，我们需要给金字塔层指定不同尺度的RoIs。

We view our feature pyramid as if it were produced from an image pyramid. Thus we can adapt the assignment strategy of region-based detectors [15, 11] in the case when they are run on image pyramids. Formally, we assign an RoI of width w and height h (on the input image to the network) to the level $P_k$ of our feature pyramid by:

我们视特征金字塔仿佛是从图像金字塔中生成的。所以我们可以修改基于区域的检测器的指定策略[15,11]，它们是在图像金字塔上运行的。正式的，我们通过下式在特征金字塔的层$P_k$上指定宽w高h的RoI：

$$k = k_0 + log_2 (\sqrt {wh}/224)$$(1)

Here 224 is the canonical ImageNet pre-training size, and $k_0$ is the target level on which an RoI with w × h = $224^2$ should be mapped into. Analogous to the ResNet-based Faster R-CNN system [16] that uses $C_4$ as the single-scale feature map, we set $k_0$ to 4. Intuitively, Eqn. (1) means that if the RoI’s scale becomes smaller (say, 1/2 of 224), it should be mapped into a finer-resolution level (say, k = 3).

这里224是经典的ImageNet预训练的大小，$k_0$是目标层，要把w × h = $224^2$的RoI映射到这个层上。与基于ResNet的Faster R-CNN系统[16]类似，使用了$C_4$作为单尺度特征图，我们设$k_0$为4。直觉上来说，式(1)意味着，如果RoI的尺度变得更小（如，224的一半），那么就应当映射到更精细分辨率的层（如，k=3）。

We attach predictor heads (in Fast R-CNN the heads are class-specific classifiers and bounding box regressors) to all RoIs of all levels. Again, the heads all share parameters, regardless of their levels. In [16], a ResNet’s conv5 layers (a 9-layer deep subnetwork) are adopted as the head on top of the conv4 features, but our method has already harnessed conv5 to construct the feature pyramid. So unlike [16], we simply adopt RoI pooling to extract 7×7 features, and attach two hidden 1,024-d fully-connected (fc) layers (each followed by ReLU) before the final classification and bounding box regression layers. These layers are randomly initialized, as there are no pre-trained fc layers available in ResNets. Note that compared to the standard conv5 head, our 2-fc MLP head is lighter weight and faster.

我们将预测器的头（在Fast R-CNN中，头部是类别相关的分类器和边界框回归器）连接到所有层的所有RoIs上。这些头也共享参数，不管哪个层次。在[16]中，一个ResNet的conv5层（一个9层的深度子网络）用在了conv4的特征的上面，但我们的方法已经利用了conv5来构建特征金字塔。所以与[16]不一样，我们只利用RoI pooling提取7×7的特征，并将2个隐藏的1024维全连接层（每个后面都有ReLU）接在了最后的分类和边界框回归层之前。这些层进行随机初始化，因为在ResNet中没有可用的预训练的fc层。注意，与标准的conv5头相比，我们的2-fc MLP头更为轻量更快。

Based on these adaptations, we can train and test Fast R-CNN on top of the feature pyramid. Implementation details are given in the experimental section. 基于这些修改，我们可以在特征金字塔上训练并测试Fast R-CNN。实现细节在试验部分给出。

## 5. Experiments on Object Detection 目标检测上的试验

We perform experiments on the 80 category COCO detection dataset [21]. We train using the union of 80k train images and a 35k subset of val images (trainval35k [2]), and report ablations on a 5k subset of val images (minival). We also report final results on the standard test set (test-std) [21] which has no disclosed labels. 我们在80类的COCO检测数据集[21]上进行试验。我们使用80k训练集和35k的验证子集的并集(trainval35k[2])进行训练，并在5k验证子集(minival)给出分离试验结果。我们还在标准测试集(test-std)[21]上给出最后结果，这个集合上没有放出的标签。

As is common practice [12], all network backbones are pre-trained on the ImageNet1k classification set [33] and then fine-tuned on the detection dataset. We use the pre-trained ResNet-50 and ResNet-101 models that are publicly available. Our code is a reimplementation of py-faster-rcnn using Caffe2. 我们采用[12]中的常规做法，所有网络骨架都是在ImageNet1k分类集[33]上进行预训练，然后在检测数据集上精调。我们使用公开可用的预训练的ResNet-50和ResNet-101模型。我们的代码是使用caffe2重新实现的py-faster-rcnn。

### 5.1. Region Proposal with RPN

We evaluate the COCO-style Average Recall (AR) and AR on small, medium, and large objects ($AR_s$, $AR_m$, and $AR_l$) following the definitions in [21]. We report results for 100 and 1000 proposals per images ($AR^{100}$ and $AR^{1k}$).

我们用[21]中的定义评估COCO式的AR，和在小、中、大型目标上的AR($AR_s$, $AR_m$ 和 $AR_l$)。我们给出每幅图像100个建议和1000个建议的结果($AR^{100}$ and $AR^{1k}$)。

**Implementation details**. All architectures in Table 1 are trained end-to-end. The input image is resized such that its shorter side has 800 pixels. We adopt synchronized SGD training on 8 GPUs. A mini-batch involves 2 images per GPU and 256 anchors per image. We use a weight decay of 0.0001 and a momentum of 0.9. The learning rate is 0.02 for the first 30k mini-batches and 0.002 for the next 10k. For all RPN experiments (including baselines), we include the anchor boxes that are outside the image for training, which is unlike [29] where these anchor boxes are ignored. Other implementation details are as in [29]. Training RPN with FPN on 8 GPUs takes about 8 hours on COCO.

**试验细节**。表1中的所有架构都是端到端训练出来的。输入图像改变其大小，使其短边为800像素。我们采用8个GPU上的同步SGD训练。Minibatch大小16，每GPU 2幅图像，每幅图像256个锚框。我们使用的权重衰减为0.0001，动量为0.9。学习速率为0.02，到30k次mini-batch迭代后学习速率降为0.002，然后再训练10k次迭代。对所有的RPN试验（包括基准），我们在训练中也包括了图像之外的锚框，这与[29]不同，其中这样的锚框都被忽略了。其他实现细节与[29]中一样。在8个GPU上用FPN训练RPN在COCO上大约耗时8小时。

#### 5.1.1.1 Ablation Experiments 分离试验

**Comparisons with baselines**. For fair comparisons with original RPNs [29], we run two baselines (Table 1(a, b)) using the single-scale map of $C_4$ (the same as [16]) or $C_5$, both using the same hyper-parameters as ours, including using 5 scale anchors of {$32^2, 64^2, 128^2, 256^2, 512^2$}. Table 1(b) shows no advantage over (a), indicating that a single higher-level feature map is not enough because there is a trade-off between coarser resolutions and stronger semantics.

**与基准测试的比较**。为与原始RPNs[29]进行公平比较，我们运行两个基准（表1(a,b)），使用的是单尺度特征图$C_4$（与[16]中一样）或$C_5$，两个都与之前使用相同的超参数，包括使用5个尺度的锚框，即{$32^2, 64^2, 128^2, 256^2, 512^2$}。表1(b)与(a)相比，没有表现出优势，这说明单个的较高层特征图是不够的，因为在较低的分辨率与较高的语义之间有一个折中关系。

Placing FPN in RPN improves $AR^{1k}$ to 56.3 (Table 1(c)), which is 8.0 points increase over the single-scale RPN baseline (Table 1 (a)). In addition, the performance on small objects ($AR^{1k}_s$) is boosted by a large margin of 12.9 points. Our pyramid representation greatly improves RPN’s robustness to object scale variation.

用FPN实现的RPN将$AR^{1k}$改进到56.3（表1(c)），这比单尺度RPN基准（表1(a)）高了8.0点。另外，在小目标上的表现($AR^{1k}_s$)提升了一大截，12.9个点。我们的金字塔表示极大的改进了RPN对目标尺度变化的稳健性。

**How important is top-down enrichment**? Table 1(d) shows the results of our feature pyramid without the top-down pathway. With this modification, the 1×1 lateral connections followed by 3×3 convolutions are attached to the bottom-up pyramid. This architecture simulates the effect of reusing the pyramidal feature hierarchy (Fig. 1(b)).

The results in Table 1(d) are just on par with the RPN baseline and lag far behind ours. We conjecture that this is because there are large semantic gaps between different levels on the bottom-up pyramid (Fig. 1(b)), especially for very deep ResNets. We have also evaluated a variant of Table 1(d) without sharing the parameters of the heads, but observed similarly degraded performance. This issue cannot be simply remedied by level-specific heads.

**自上而下的特征充实有多重要**？表1(d)给出了没有自上而下的通道下我们的特征金字塔的结果。有了这个修正，1×1的横向连接后的3×3卷积接到了自下而上的金字塔。这个架构模拟了重复使用金字塔特征层次的效果（图1(b)）。

表1(d)中的结果刚刚与RPN基准的类似，远在我们的结果之后。我们推测这是因为在自下而上的金字塔上不同层之间有很大的语义空白（图1(b)），尤其是对于很深的ResNet而言。我们还评估了表1(d)的一个变体，网络头之间不共享参数，观察到的也是类似的不太好的性能。这个问题不能简单的由每层不同的网络头所修正。

**How important are lateral connections**? Table 1(e) shows the ablation results of a top-down feature pyramid without the 1×1 lateral connections. This top-down pyramid has strong semantic features and fine resolutions. But we argue that the locations of these features are not precise, because these maps have been downsampled and upsampled several times. More precise locations of features can be directly passed from the finer levels of the bottom-up maps via the lateral connections to the top-down maps. As a results, FPN has an $AR^{1k}$ score 10 points higher than Table 1(e).

**横向连接的作用有多重要**？表1(e)展示了自上而下的特征金字塔没有1×1的横向连接时的分离试验。这种自上而下的金字塔有着很强的语义特征和很精细的分辨率，但我们认为，这些特征的位置不那么准确，因为这些特征图已经被下采样、上采样几次了。更精确的特征位置可以从自下而上特征图的精细层中，通过横向连接直接传递到自上而下的特征图中。结果是，FPN比表1(e)的$AR^{1k}$分数高了10个点。

**How important are pyramid representations**? Instead of resorting to pyramid representations, one can attach the head to the highest-resolution, strongly semantic feature maps of $P_2$ (i.e., the finest level in our pyramids). Similar to the single-scale baselines, we assign all anchors to the $P_2$ feature map. This variant (Table 1(f)) is better than the baseline but inferior to our approach. RPN is a sliding window detector with a fixed window size, so scanning over pyramid levels can increase its robustness to scale variance.

In addition, we note that using $P_2$ alone leads to more anchors (750k, Table 1(f)) caused by its large spatial resolution. This result suggests that a larger number of anchors is not sufficient in itself to improve accuracy.

**金字塔表示有多重要**？如果不用金字塔表示，可以将网络头接到最高分辨率、语义最强的特征图$P_2$上（即我们金字塔的最精细层）。与单尺度基准类似，我们给$P_2$特征图指定所有的锚框。这个变体（表1(f)）比基准要好一些，但比我们的要差。RPN是一个滑窗检测器，窗口大小固定，所以在金字塔各层上扫描可以增加对尺度变化的稳健性。

另外，我们指出，只使用$P_2$会带来更多的锚框（表1(f)，750k），这是因为其空间分辨率很大。这个结果说明，很多锚框并不足以改进准确率。

Table 1. Bounding box proposal results using RPN [29], evaluated on the COCO minival set. All models are trained on trainval35k. The columns “lateral” and “top-down” denote the presence of lateral and top-down connections, respectively. The column “feature” denotes the feature maps on which the heads are attached. All results are based on ResNet-50 and share the same hyper-parameters.

RPN | feature | anchors | lateral? | top-down? | $AR^{100}$ | $AR^{1k}$ | $AR^{1k}_s$ | $AR^{1k}_m$ | $AR^{1k}_l$
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
(a) baseline on conv4 | C4 | 47k | n | n | 36.1 | 48.3 | 32.0 | 58.7 | 62.2
(b) baseline on conv5 | C5 | 12k | n | n | 36.3 | 44.9 | 25.3 | 55.5 | 64.2
(c) FPN | {Pk} | 200k | y | y | 44.0 | 56.3 | 44.9 | 63.4 | 66.2
Ablation experiments follow: |
(d) bottom-up pyramid | {Pk} | 200k | y | n | 37.4 | 49.5 | 30.5 | 59.9 | 68.0
(e) top-down pyramid, w/o lateral | {Pk} | 200k | n | y | 34.5 | 46.1 | 26.5 | 57.4 | 64.7
(f) only finest level | P2 | 750k | y | y | 38.4 | 51.3 | 35.1 | 59.7 | 67.6

### 5.2. Object Detection with Fast/Faster R-CNN

Next we investigate FPN for region-based (non-sliding window) detectors. We evaluate object detection by the COCO-style Average Precision (AP) and PASCAL-style AP (at a single IoU threshold of 0.5). We also report COCO AP on objects of small, medium, and large sizes (namely, $AP_s$, $AP_m$, and $AP_l$) following the definitions in [21].

下面我们研究一下使用FPN进行基于区域的（非滑窗式的）目标检测。我们用COCO式的AP和PASCAL式的AP（在单IoU阈值0.5上）评估目标检测。我们还给出小、中、大目标上的COCO AP（即$AP_s$, $AP_m$和$AP_l$），与[21]中相同。

**Implementation details**. The input image is resized such that its shorter side has 800 pixels. Synchronized SGD is used to train the model on 8 GPUs. Each mini-batch involves 2 image per GPU and 512 RoIs per image. We use a weight decay of 0.0001 and a momentum of 0.9. The learning rate is 0.02 for the first 60k mini-batches and 0.002 for the next 20k. We use 2000 RoIs per image for training and 1000 for testing. Training Fast R-CNN with FPN takes about 10 hours on the COCO dataset.

**实现细节**。输入图像改变大小，使其短边为800像素。在8个GPU上用同步SGD进行训练。每个mini-batch包括2图像/GPU，每幅图像512 RoI。我们使用的权重衰减为0.0001，动量0.9。学习速率在前60k个mini-batch为0.02，后面20k次迭代为0.002。我们在训练时每幅图像使用2000 RoI，测试时使用1000个。在COCO数据集上训练FPN版的Fast R-CNN耗时约10小时。

#### 5.2.1 Fast R-CNN (on fixed proposals)

To better investigate FPN’s effects on the region-based detector alone, we conduct ablations of Fast R-CNN on a fixed set of proposals. We choose to freeze the proposals as computed by RPN on FPN (Table 1(c)), because it has good performance on small objects that are to be recognized by the detector. For simplicity we do not share features between Fast R-CNN and RPN, except when specified.

为更好的研究FPN对基于区域的检测器的影响，我们在固定建议区域集合上进行Fast R-CNN的分离试验。我们选择冻结FPN-RPN计算得到的建议区域（表1(c)），因为在小目标上有很好的表现。简化起见，在Fast R-CNN和RPN之间不共享参数，除非另有说明。

As a ResNet-based Fast R-CNN baseline, following [16], we adopt RoI pooling with an output size of 14×14 and attach all conv5 layers as the hidden layers of the head. This gives an AP of 31.9 in Table 2(a). Table 2(b) is a baseline exploiting an MLP head with 2 hidden fc layers, similar to the head in our architecture. It gets an AP of 28.8, indicating that the 2-fc head does not give us any orthogonal advantage over the baseline in Table 2(a).

作为一个基于ResNet的R-CNN基准，遵循[16]，我们采用输出大小14×14的RoI pooling，将所有的conv5层作为网络头的隐藏层接上去。这可以得到31.9 AP，如表2(a)所示。表2(b)也是一个基准，采用一个MLP头，有2个隐藏的全连接层，与我们架构中的网络头类似，得到了28.8 AP，说明2-fc头并不比表2(a)有什么优势。

Table 2(c) shows the results of our FPN in Fast R-CNN. Comparing with the baseline in Table 2(a), our method improves AP by 2.0 points and small object AP by 2.1 points. Comparing with the baseline that also adopts a 2fc head (Table 2(b)), our method improves AP by 5.1 points. (We expect a stronger architecture of the head [30] will improve upon our results, which is beyond the focus of this paper.) These comparisons indicate that our feature pyramid is superior to single-scale features for a region-based object detector.

表2(c)是我们的FPN版Fast R-CNN结果。与表2(a)的基准相比较，我们的方法改进了AP 2.0点，在小目标上改进2.1点。与也采用了2fc头的基准表2(b)相比，我们的方法改进了AP 5.1点。（我们希望[30]中更强的架构的头会改进我们的结果，但这不在本文的讨论中。）这些比较说明，我们的特征金字塔比单尺度特征基于区域的目标检测器要更好。

Table 2(d) and (e) show that removing top-down connections or removing lateral connections leads to inferior results, similar to what we have observed in the above subsection for RPN. It is noteworthy that removing top-down connections (Table 2(d)) significantly degrades the accuracy, suggesting that Fast R-CNN suffers from using the low-level features at the high-resolution maps.

表2(d)和(e)所展示的是，去掉自上而下的连接，或去掉横向连接，会得到更差的结果，与我们的上面RPN部分观察到的结果类似。值得注意的是，去掉自上而下的连接（表2(d)）显著的降低了准确率，说明Fast R-CNN在使用高分辨率特征图中的低层特征时，效果并不好。

In Table 2(f), we adopt Fast R-CNN on the single finest scale feature map of $P_2$. Its result (33.4 AP) is marginally worse than that of using all pyramid levels (33.9 AP, Table 2(c)). We argue that this is because RoI pooling is a warping-like operation, which is less sensitive to the region’s scales. Despite the good accuracy of this variant, it is based on the RPN proposals of {$P_k$} and has thus already benefited from the pyramid representation.

在表2(f)中，我们在单个精细尺度特征图层$P_2$上采用Fast R-CNN。其结果33.4 AP比使用所有的金字塔层级的结果（33.9 AP，表2(c)）略差。我们认为，这是因为RoI pooling是一种类似变形的操作，对区域的尺度并不敏感。尽管这种变体有很好的准确率，但是基于RPN建议区域{$P_k$}的，因此已经使用了金字塔表示。

Table 2. Object detection results using Fast R-CNN [11] on a fixed set of proposals (RPN, {P k }, Table 1(c)), evaluated on the COCO minival set. Models are trained on the trainval35k set. All results are based on ResNet-50 and share the same hyper-parameters.

Fast R-CNN | proposals | feature | head | lateral? | top-down? | AP@0.5 | AP | $AP_s$ | $AP_m$ | $AP_l$
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
(a) baseline on conv4 | RPN,{Pk} | C4 | conv5 | n | n | 54.7 | 31.9 | 15.7 | 36.5 | 45.5
(b) baseline on conv5 | RPN,{Pk} | C5 | 2fc | n | n | 52.9 | 28.8 | 11.9 | 32.4 | 43.4
(c) FPN | RPN,{Pk} | {Pk} | 2fc | y | y | 56.9 | 33.9 | 17.8 | 37.7 | 45.8
Ablation experiments follow: |
(d) bottom-up pyramid | RPN,{Pk} | {Pk} | 2fc | y | n | 44.9 | 24.9 | 10.9 | 24.4 | 38.5
(e) top-down pyramid, w/o lateral | RPN,{Pk} | {Pk} | 2fc | n | y | 54.0 | 31.3 | 13.3 | 35.2 | 45.3
(f) only finest level | RPN,{Pk} | P2 | 2fc | y | y | 56.3 | 33.4 | 17.3 | 37.3 | 45.6

#### 5.2.2 Faster R-CNN (on consistent proposals)

In the above we used a fixed set of proposals to investigate the detectors. But in a Faster R-CNN system [29], the RPN and Fast R-CNN must use the same network backbone in order to make feature sharing possible. Table 3 shows the comparisons between our method and two baselines, all using consistent backbone architectures for RPN and Fast R-CNN. Table 3(a) shows our reproduction of the baseline Faster R-CNN system as described in [16]. Under controlled settings, our FPN (Table 3(c)) is better than this strong baseline by 2.3 points AP and 3.8 points AP@0.5.

在上面，我们使用固定集合的建议区域来研究检测器。但在Faster R-CNN系统中[29]，RPN和Fast R-CNN必须使用相同的骨干网络，以使特征共享成为可能。表3所示的是我们的方法和两个基准的比较，RPN和Fast R-CNN使用的都是一样的骨干架构。表3(a)是我们复现的[16]中的Faster R-CNN基准系统。在受控的设置中，我们的FPN（表3(c)）比这个强基准高了2.3 AP及3.8 AP@0.5。

Note that Table 3(a) and (b) are baselines that are much stronger than the baseline provided by He et al. [16] in Table 3(* ). We find the following implementations contribute to the gap: (i) We use an image scale of 800 pixels instead of 600 in [11, 16]; (ii) We train with 512 RoIs per image which accelerate convergence, in contrast to 64 RoIs in [11, 16]; (iii) We use 5 scale anchors instead of 4 in [16] (adding $32^2$); (iv) At test time we use 1000 proposals per image instead of 300 in [16]. So comparing with He et al.’s ResNet50 Faster R-CNN baseline in Table 3( *), our method improves AP by 7.6 points and AP@0.5 by 9.6 points.

注意表3(a)和(b)比表3( *)中的He等[16]的基准要强的多。我们发现下面的实现增大了这个差距：(i)我们使用的图像尺度为800像素，而不是[11,16]中的600；(ii)我们使用每幅图像512 RoI进行训练，这加速了收敛，比较之下，[11,16]中使用的64 RoIs；(iii)我们使用5个尺度的锚框，而[16]中使用的4个尺度（增加了$32^2$这个尺度）；(iv)在测试时，我们使用每幅图像1000个建议的设置，而[16]中使用的是300。所以与He等的ResNet50 Faster R-CNN基准表3( *)相比，我们的方法改进了7.6 AP和9.6 AP@0.5。

**Sharing features**. In the above, for simplicity we do not share the features between RPN and Fast R-CNN. In Table 5, we evaluate sharing features following the 4-step training described in [29]. Similar to [29], we find that sharing features improves accuracy by a small margin. Feature sharing also reduces the testing time.

**共享特征**。上面，为简化我们在RPN和Fast R-CNN之间没有共享特征。表5中，我们评估共享特征时的表现，类似[29]中的4步训练。与[29]类似，我们发现共享特征对准确率有一些改进。特征共享也减少了测试时间。

**Running time**. With feature sharing, our FPN-based Faster R-CNN system has inference time of 0.148 seconds per image on a single NVIDIA M40 GPU for ResNet-50, and 0.172 seconds for ResNet-101. As a comparison, the single-scale ResNet-50 baseline in Table 3(a) runs at 0.32 seconds. Our method introduces small extra cost by the extra layers in the FPN, but has a lighter weight head. Overall our system is faster than the ResNet-based Faster R-CNN counterpart. We believe the efficiency and simplicity of our method will benefit future research and applications.

**运行时间**。在特征共享下，我们基于FPN的ResNet-50 Faster R-CNN的推理时间为每幅图像0.148秒，在单块NVidia M40 GPU上，ResNet-101则为0.172秒。比较之下，表3(a)中的单尺度ResNet-50基准运行时间为0.32秒。我们的方法引入的FPN中的额外层，带来了很小的额外代价，权重头很轻量。总体上我们的系统比基于ResNet的Faster R-CNN更快。我们相信我们方法的效率和简单性会使将来的研究和应用受益。

Table 3. Object detection results using Faster R-CNN [29] evaluated on the COCO minival set. The backbone network for RPN are consistent with Fast R-CNN. Models are trained on the trainval35k set and use ResNet-50. † Provided by authors of [16].

Faster R-CNN | proposal | feature | head | lateral? | top-down? | AP@0.5 | AP | $AP_s$ | $AP_m$ | $AP_l$
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
(*) baseline from He et al. [16] † | RPN,C4 | C4 | conv5 | n | n | 47.3 | 26.3 | - | - | -
(a) baseline on conv4 | RPN,C4 | C4 | conv5 | n | n | 53.1 | 31.6 | 13.2 | 35.6 | 47.1
(b) baseline on conv5 | RPN,C5 | C5 | 2fc | n | n | 51.7 | 28.0 | 9.6 | 31.9 | 43.1
(c) FPN | RPN,{Pk} | {Pk} | 2fc | y | y | 56.9 | 33.9 | 17.8 | 37.7 | 45.8

Table 5. More object detection results using Faster R-CNN and our FPNs, evaluated on minival. Sharing features increases train time by 1.5× (using 4-step training [29]), but reduces test time.

share features? | ResNet-50 AP@0.5 | ResNet-50 AP | ResNet-101 AP@0.5 | ResNet-101 AP
--- | --- | --- | --- | ---
no | 56.9 | 33.9 | 58.0 | 35.0
yes | 57.2 | 34.3 | 58.2 | 35.2

#### 5.2.3 Comparing with COCO Competition Winners

We find that our ResNet-101 model in Table 5 is not sufficiently trained with the default learning rate schedule. So we increase the number of mini-batches by 2× at each learning rate when training the Fast R-CNN step. This increases AP on minival to 35.6, without sharing features. This model is the one we submitted to the COCO detection leaderboard, shown in Table 4. We have not evaluated its feature-sharing version due to limited time, which should be slightly better as implied by Table 5.

我们发现，表5中我们的ResNet-101模型在默认的学习速率方案下没有得到充分的训练。所以我们在训练Fast R-CNN时，在每个学习速率下增加了一倍的mini-batch数量。这将在minival下的AP增加到了35.6，没有共享特征。这个模型我们提交到了COCO检测排行榜上，如表4所示。由于时间有限，还没有评估其共享特征的版本，根据表5，应当效果更好一些。

Table 4. Comparisons of single-model results on the COCO detection benchmark. Some results were not available on the test-std set, so we also include the test-dev results (and for Multipath [40] on minival).

Table 4 compares our method with the single-model results of the COCO competition winners, including the 2016 winner G-RMI and the 2015 winner Faster R-CNN+++. Without adding bells and whistles, our single-model entry has surpassed these strong, heavily engineered competitors. On the test-dev set, our method increases over the existing best results by 0.5 points of AP (36.2 vs. 35.7) and 3.4 points of AP@0.5 (59.1 vs. 55.7). It is worth noting that our method does not rely on image pyramids and only uses a single input image scale, but still has outstanding AP on small-scale objects. This could only be achieved by high-resolution image inputs with previous methods.

表4将我们的方法与COCO竞赛胜利者的单模型结果进行了比较，包括2016赢家G-RMI和2015赢家Faster R-CNN+++。我们的单模型超过了这些竞争者。在test-dev集上，我们的方法比现有最好的结果提高了AP 0.5点(36.2 vs 36.7)，AP@0.5增加了3.4 (59.1 vs 55.7)。值得注意的是，我们的方法没有使用图像金字塔，只使用了单个输入图像尺度，但仍然对小尺度目标表现非常好。这在之前的方法中，只能对高分辨率图像输入才能取得这样的结果。

Moreover, our method does not exploit many popular improvements, such as iterative regression [9], hard negative mining [35], context modeling [16], stronger data augmentation [22], etc. These improvements are complementary to FPNs and should boost accuracy further.

而且，我们的方法没有使用很多流行的改进，比如迭代回归[9]，难分负样本挖掘[35]，上下文建模[16]，更强的数据扩充[22]等。这些改进对FPN都是补充，应当可以进一步改进准确率。

Recently, FPN has enabled new top results in all tracks of the COCO competition, including detection, instance segmentation, and keypoint estimation. See [14] for details. 最近，FPN已经在所有COCO比赛上得到了更好的最佳结果，包括检测、实例分割和关键点估计。

## 6. Extensions: Segmentation Proposals 拓展：分割建议

Our method is a generic pyramid representation and can be used in applications other than object detection. In this section we use FPNs to generate segmentation proposals, following the DeepMask/SharpMask framework [27, 28].

我们的方法是一个通用的金字塔表示，可以用于目标检测之外的应用。本节中，我们使用FPN生成分割建议，采用的是DeepMask/SharpMask框架[27,28]。

DeepMask/SharpMask were trained on image crops for predicting instance segments and object/non-object scores. At inference time, these models are run convolutionally to generate dense proposals in an image. To generate segments at multiple scales, image pyramids are necessary [27, 28].

DeepMask/SharpMask在图像剪切块上进行训练，进行实例分割和目标/非目标分数的预测。在推理时，这些模型都是在一幅图像上进行卷积运算，以生成密集建议区域。为在多个尺度上生成实例分割，图像金字塔就成为了必须[27,28]。

It is easy to adapt FPN to generate mask proposals. We use a fully convolutional setup for both training and inference. We construct our feature pyramid as in Sec. 5.1 and set d = 128. On top of each level of the feature pyramid, we apply a small 5×5 MLP to predict 14×14 masks and object scores in a fully convolutional fashion, see Fig. 4. Additionally, motivated by the use of 2 scales per octave in the image pyramid of [27, 28], we use a second MLP of input size 7×7 to handle half octaves. The two MLPs play a similar role as anchors in RPN. The architecture is trained end-to-end; full implementation details are given in the appendix.

很容易调整FPN以生成掩膜建议。我们对训练和推理使用同一个全卷积设置。我们像5.1节中一样构建特征金字塔，设d=128。在特征金字塔的每层之上，我们应用一个小的5×5 MLP来以卷积的形式预测14×14的掩膜和目标分数，见图4。另外，受到[27,28]中图像金字塔中的每个两倍尺度间隔使用2个尺度的启发，我们使用第二个MLP，输入大小为7×7，来处理半octave。这两个MLP与RPN中的锚的作用类似。架构是进行的端到端的训练；完整的实现细节在附录中给出。

Figure 4. FPN for object segment proposals. The feature pyramid is constructed with identical structure as for object detection. We apply a small MLP on 5×5 windows to generate dense object segments with output dimension of 14×14. Shown in orange are the size of the image regions the mask corresponds to for each pyramid level (levels P 3−5 are shown here). Both the corresponding image region size (light orange) and canonical object size (dark orange) are shown. Half octaves are handled by an MLP on 7x7 windows ($7 ≈ 5 \sqrt 2$), not shown here. Details are in the appendix.

### 6.1. Segmentation Proposal Results

Results are shown in Table 6. We report segment AR and segment AR on small, medium, and large objects, always for 1000 proposals. Our baseline FPN model with a single 5×5 MLP achieves an AR of 43.4. Switching to a slightly larger 7×7 MLP leaves accuracy largely unchanged. Using both MLPs together increases accuracy to 45.7 AR. Increasing mask output size from 14×14 to 28×28 increases AR another point (larger sizes begin to degrade accuracy). Finally, doubling the training iterations increases AR to 48.1.

结果如表6所示。我们给出分割AR和在小、中、大目标上的分割AR结果，一直使用的1000建议的配置。我们的基准FPN模型带有单5×5 MLP得到了43.4 AR。切换到略大的7×7 MLP，准确率基本上没有变化。一起使用2个MLP，准确率提升到45.7 AR。将掩膜输出大小从14×14增加到28×28，AR进一步提升1个点（更大的大小会降低准确率）。最后，训练迭代次数加倍，AR增加到48.1。

We also report comparisons to DeepMask [27], SharpMask [28], and InstanceFCN [4], the previous state of the art methods in mask proposal generation. We outperform the accuracy of these approaches by over 8.3 points AR. In particular, we nearly double the accuracy on small objects.

我们还与DeepMask [27], SharpMask [28], and InstanceFCN [4]进行了比较，这都是之前最好的方法。我们超过了这些方法的准确率，AR提高了8.3个点。特别的，我们在小目标上的准确率提高了近一倍。

Existing mask proposal methods [27, 28, 4] are based on densely sampled image pyramids (e.g., scaled by $2^{\{−2:0.5:1\}}$ in [27, 28]), making them computationally expensive. Our approach, based on FPNs, is substantially faster (our models run at 6 to 7 FPS). These results demonstrate that our model is a generic feature extractor and can replace image pyramids for other multi-scale detection problems.

现有的掩膜建议方法[27,28,4]都是基于密集采样的图像金字塔（如，[27,28]中的尺度为$2^{\{−2:0.5:1\}}$），这样计算量非常大。我们的方法是基于FPN的，则快的多（我们的模型运行速度有6-7 FPS）。这些结果表明了，我们的模型是一个通用的特征提取器，可以将其他多尺度检测问题替换为图像金字塔。

## 7. Conclusion 结论

We have presented a clean and simple framework for building feature pyramids inside ConvNets. Our method shows significant improvements over several strong baselines and competition winners. Thus, it provides a practical solution for research and applications of feature pyramids, without the need of computing image pyramids. Finally, our study suggests that despite the strong representational power of deep ConvNets and their implicit robustness to scale variation, it is still critical to explicitly address multiscale problems using pyramid representations.

我们提出了一种简洁的框架，在ConvNets中构建特征金字塔。我们的方法与几种很强的基准和竞赛获胜者相比较都得到了显著的改进。所以，这为特征金字塔的研究和应用给出了一种实际的解决方案，而且不需要计算图像金字塔。最后，我们的研究表明，尽管深度ConvNets的很强的表示能力，及其隐式的对尺度变化的稳健性，使用金字塔表示，显式的处理多尺度问题，还是非常关键的。

## A. Implementation of Segmentation Proposals