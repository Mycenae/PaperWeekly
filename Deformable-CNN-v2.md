# Deformable ConvNets v2: More Deformable, Better Results

Xizhou Zhu, Jifeng Dai, et. al. Microsoft Research Asia

## 0. Abstract

The superior performance of Deformable Convolutional Networks arises from its ability to adapt to the geometric variations of objects. Through an examination of its adaptive behavior, we observe that while the spatial support for its neural features conforms more closely than regular ConvNets to object structure, this support may nevertheless extend well beyond the region of interest, causing features to be influenced by irrelevant image content. To address this problem, we present a reformulation of Deformable ConvNets that improves its ability to focus on pertinent image regions, through increased modeling power and stronger training. The modeling power is enhanced through a more comprehensive integration of deformable convolution within the network, and by introducing a modulation mechanism that expands the scope of deformation modeling. To effectively harness this enriched modeling capability, we guide network training via a proposed feature mimicking scheme that helps the network to learn features that reflect the object focus and classification power of R-CNN features. With the proposed contributions, this new version of Deformable ConvNets yields significant performance gains over the original model and produces leading results on the COCO benchmark for object detection and instance segmentation.

形变网络的优异性能是源于对目标的几何变化的适应能力。通过对其适应行为的检查，我们观察到，与常规CNN相比，其神经特征的空间支撑更加符合目标结构，但是这种支撑仍然会拓展到ROI以外很远的地方，导致特征被不相关的图像内容影响。为解决这个问题，我们对形变CNN进行了重新表述，通过增强建模能力，更强的训练，改进其能力，聚焦在合适的图像区域中。建模能力是通过形变卷积与网络更加全面的结合来得到增强的，并提出了一种调节机制，拓展了形变建模的范围。为有效的利用这种增强的建模能力，我们通过一种提出的特征模仿方案，来指导网络训练，帮助网络学习反应目标焦点的特征和R-CNN分类能力的特征。提出的新版形变CNN，比以前的模型得到了显著的性能提升，在COCO基准测试的目标检测和实例分割中得到了非常好的效果。

## 1. 1. Introduction

Geometric variations due to scale, pose, viewpoint and part deformation present a major challenge in object recognition and detection. The current state-of-the-art method for addressing this issue is Deformable Convolutional Networks (DCNv1) [8], which introduces two modules that aid CNNs in modeling such variations. One of these modules is deformable convolution, in which the grid sampling locations of standard convolution are each offset by displacements learned with respect to the preceding feature maps. The other is deformable RoIpooling, where offsets are learned for the bin positions in RoIpooling [16]. The incorporation of these modules into a neural network gives it the ability to adapt its feature representation to the configuration of an object, specifically by deforming its sampling and pooling patterns to fit the object’s structure. With this approach, large improvements in object detection accuracy are obtained.

由于尺度、姿态、视角和部位形变导致的几何变化，在目标识别和检测中提出了主要的挑战。解决这个问题目前最好的方法是形变CNN，它提出了两个模块，帮助CNN对这种变化进行建模。一个是形变卷积，其中的标准卷积网格采样位置是学到的对前面的特征图的偏移。另一个是形变ROI池化，对ROI池化中的bin位置学习了偏移。将这两个模块结合到神经网络中，可以将其特征表示改变适应目标的配置，特别是改变其采样和池化模式，以适应目标的结构。以这种方法，可以在目标检测中得到改进的性能。

Towards understanding Deformable ConvNets, the authors visualized the induced changes in receptive field, via the arrangement of offset sampling positions in PASCAL VOC images [11]. It is found that samples for an activation unit tend to cluster around the object on which it lies. However, the coverage over an object is inexact, exhibiting a spread of samples beyond the area of interest. In a deeper analysis of spatial support using images from the more challenging COCO dataset [29], we observe that such behavior becomes more pronounced. These findings suggest that greater potential exists for learning deformable convolutions.

通过理解形变CNN，作者通过对PASCAL VOC中的图像进行偏移采样位置的安排，对感受野中带来的变化进行了可视化，发现对一个激活单元的样本，会倾向于聚积在其上的目标附近。但是，对目标的覆盖是不准确的，会扩散到样本感兴趣区域之外。使用COCO数据集中的图像对空间支撑的更深入分析，我们发现这种行为变得更加显著。这种发现说明，学习形变卷积存在更大的潜能。

In this paper, we present a new version of Deformable ConvNets, called Deformable ConvNets v2 (DCNv2), with enhanced modeling power for learning deformable convolutions. This increase in modeling capability comes in two complementary forms. The first is the expanded use of deformable convolution layers within the network. Equipping more convolutional layers with offset learning capacity allows DCNv2 to control sampling over a broader range of feature levels. The second is a modulation mechanism in the deformable convolution modules, where each sample not only undergoes a learned offset, but is also modulated by a learned feature amplitude. The network module is thus given the ability to vary both the spatial distribution and the relative influence of its samples.

本文中，我们提出了一种新版的形变卷积网络，称为Deformable-CNN v2，学习形变卷积的建模能力得到了增强。建模能力的增强来自于两种互补的形式。第一种是网络中形变卷积层使用的更广泛。装备了更多卷积层，可以学习更多的偏移，使DCNv2可以控制更广范围内的特征层次的采样。第二个是在形变卷积模块中的调制机制，其中每个样本不仅经历了学习的偏移，而且还通过学习的特征幅度进行调节。这个网络模块因此就有了下面的能力，根据样本的空间分布和相对影响来变化。

To fully exploit the increased modeling capacity of DCNv2, effective training is needed. Inspired by work on knowledge distillation in neural networks [2, 22], we make use of a teacher network for this purpose, where the teacher provides guidance during training. We specifically utilize R-CNN [17] as the teacher. Since it is a network trained for classification on cropped image content, R-CNN learns features unaffected by irrelevant information outside the region of interest. To emulate this property, DCNv2 incorporates a feature mimicking loss into its training, which favors learning of features consistent to those of R-CNN. In this way, DCNv2 is given a strong training signal for its enhanced deformable sampling.

为完全利用DCNv2的增强的建模能力，需要进行有效的训练。受到神经网络的知识蒸馏工作的启发，我们利用了一个teacher网络，teacher在训练中给出指导。我们具体利用了R-CNN作为teacher。由于这是在图像内容剪切块上对分类训练的网络，R-CNN学习的特征不会受到ROI外的不相关信息影响。为模仿这种性质，DCNv2在训练中利用了一个特征模拟损失，倾向于学习与R-CNN一致的特征。这样，DCNv2在训练中有一个很强的增强形变采样的信号。

With the proposed changes, the deformable modules remain lightweight and can easily be incorporated into existing network architectures. Specifically, we incorporate DCNv2 into the Faster R-CNN [33] and Mask R-CNN [20] systems, with a variety of backbone networks. Extensive experiments on the COCO benchmark demonstrate the significant improvement of DCNv2 over DCNv1 for object detection and instance segmentation. The code for DCNv2 will be released.

提出了这些变化，形变模块仍然保持轻量的，可以很容易的与现有的网络架构整合到一起。具体的，我们将DCNv2整合到Faster R-CNN和Mask R-CNN中，有多种骨干网络结构。在COCO基准测试中进行了广泛的试验，证明了DCNv2比DCNv1在目标检测和实例分割中有显著的性能改进。代码已开源。

## 2. Analysis of Deformable ConvNet Behavior

### 2.1. Spatial Support Visualization

To better understand the behavior of Deformable ConvNets, we visualize the spatial support of network nodes by their effective receptive fields [31], effective sampling locations, and error-bounded saliency regions. These three modalities provide different and complementary perspectives on the underlying image regions that contribute to a node’s response.

为更好的理解形变CNN的行为，我们对网络节点的空间支撑通过其有效感受野、有效采样位置和误差边界显著性区域进行了可视化。这三种模态对潜在的对节点响应有贡献的图像区域给出了不同的互补的视角。

**Effective receptive fields**. Not all pixels within the receptive field of a network node contribute equally to its response. The differences in these contributions are represented by an effective receptive field, whose values are calculated as the gradient of the node response with respect to intensity perturbations of each image pixel [31]. We utilize the effective receptive field to examine the relative influence of individual pixels on a network node, but note that this measure does not reflect the structured influence of full image regions.

**有效感受野**。一个网络节点的感受野中，并不是所有的像素都对其响应贡献相等。其贡献的差异是通过有效感受野表示的，其值为节点响应对每个像素的扰动的梯度。我们利用有效感受野来检查一个网络节点上单个像素的相对影响，但注意，这个度量并不反应完整图像区域的结构化影响。

**Effective sampling / bin locations**. In [8], the sampling locations of (stacked) convolutional layers and the sampling bins in RoIpooling layers are visualized for understanding the behavior of Deformable ConvNets. However, the relative contributions of these sampling locations to the network node are not revealed. We instead visualize effective sampling locations that incorporate this information, computed as the gradient of the network node with respect to the sampling / bin locations, so as to understand their contribution strength.

**有效采样/bin位置**。为理解形变CNN中的行为，在[8]中，对堆叠的卷积层的采样位置，和ROI池化层中的采样bins进行了可视化。但是，这些采样位置对网络节点的相对贡献并没有揭示。我们对有效采样位置进行了可视化，包含了这个信息，计算为网络节点对采样/bin位置的梯度，以理解其贡献强度。

**Error-bounded saliency regions**. The response of a network node will not change if we remove image regions that do not influence it, as demonstrated in recent research on image saliency [41, 44, 13, 7]. Based on this property, we can determine a node’s support region as the smallest image region giving the same response as the full image, within a small error bound. We refer to this as the error-bounded saliency region, which can be found by progressively masking parts of the image and computing the resulting node response, as described in more detail in the Appendix. The error-bounded saliency region facilitates comparison of support regions from different networks.

**误差边界显著性区域**。如果我们移除了对其没有影响的图像区域，一个网络节点的响应不会变化，最近在图像显著性上的研究证明了这一点。基于这个性质，我们可以将一个节点的支持区域，确定为给出与完整图像相同响应的最小图像区域，有很小的响应误差界限。我们称之为误差界限显著性区域，可以通过逐渐将图像的一部分掩模掉，然后计算结果节点响应来得到，这在附录中有详述。误差界限显著性区域，促进了不同网络的支持区域的比较。

### 2.2. Spatial Support of Deformable ConvNets

We analyze the visual support regions of Deformable ConvNets in object detection. The regular ConvNet we employ as a baseline consists of a Faster R-CNN + ResNet-50 [21] object detector with aligned RoIpooling [20]. All the convolutional layers in ResNet-50 are applied on the whole input image. The effective stride in the conv5 stage is reduced from 32 to 16 pixels to increase feature map resolution. The RPN [33] head is added on top of the conv4 features of ResNet-101. On top of the conv5 features we add the Fast R-CNN head [16], which is composed of aligned RoIpooling and two fully-connected (fc) layers, followed by the classification and bounding box regression branches. We follow the procedure in [8] to turn the object detector into its deformable counterpart. The three layers of 3 × 3 convolutions in the conv5 stage are replaced by deformable convolution layers. Also, the aligned RoIpooling layer is replaced by deformable RoIPooling. Both networks are trained and visualized on the COCO benchmark. It is worth mentioning that when the offset learning rate is set to zero, the Deformable Faster R-CNN detector degenerates to regular Faster R-CNN with aligned RoIpooling.

我们分析了形变CNN在目标检测中的可视化支持区域。我们采用作为基准的常规CNN，包含了Faster R-CNN + ResNet-50目标检测器，带有对齐的ROIpooling。ResNet-50中的所有卷积层应用到整个输入图像中。conv5阶段的有效步长从32降低到了16，以增加特征图分辨率。在ResNet-101的conv4特征上，加上了RPN头。在conv5特征上，我们加上了Fast R-CNN头，由对齐ROI池化和两个fc层组成，随后是分类和边界框回归分支。我们按照[8]中的过程，将目标检测器变成了其形变版。在conv5阶段中的三层3x3卷积，替换为形变卷积层。同时，对齐的ROI池化层替换成了形变ROI池化层。两个网络都是在COCO基准测试中进行训练和可视化。值得提到的是，当偏移学习率设为0时，形变Faster R-CNN就退化成了常规的带有对齐ROI池化的Faster R-CNN。

Using the three visualization modalities, we examine the spatial support of nodes in the last layer of the conv5 stage in Figure 1 (a)∼(b). The sampling locations analyzed in [8] are also shown. From these visualizations, we make the following observations:

使用三种可视化模态，我们在conv5阶段的最后一层检查了节点的空间支撑，如图1a-b所示。[8]中的采样位置也进行了展示。从这些可视化中，我们得到了下面的观察结果：

1. Regular ConvNets can model geometric variations to some extent, as evidenced by the changes in spatial support with respect to image content. Thanks to the strong representation power of deep ConvNets, the network weights are learned to accommodate some degree of geometric transformation. 常规CNN可以在一定程度上对几何变化进行建模，图中可以看到空间支撑对不同的图像内容是有变化的。多亏了DCNN有很强的表示能力，网络权重通过学习适应了一定的几何变换。

2. By introducing deformable convolution, the network’s ability to model geometric transformation is considerably enhanced, even on the challenging COCO benchmark. The spatial support adapts much more to image content, with nodes on the foreground having support that covers the whole object, while nodes on the background have expanded support that encompasses greater context. However, the range of spatial support may be inexact, with the effective receptive field and error-bounded saliency region of a foreground node including background areas irrelevant for detection. 通过引入形变卷积，网络建模几何变换的能力得到相当的强化，甚至是在COCO基准测试中。空间支撑对图像内容的变化适应更多，在前景上的节点，其支撑覆盖了整个目标，在背景上的节点，其支撑包含了更多的上下文。但是，空间支撑的范围可能是不精确的，一个前景节点的有效感受野和误差界限显著性区域，包含了检测无关的背景区域。

3. The three presented types of spatial support visualizations are more informative than the sampling locations used in [8]. This can be seen, for example, with regular ConvNets, which have fixed sampling locations along a grid, but actually adapt its effective spatial support via network weights. The same is true for Deformable ConvNets, whose predictions are jointly affected by learned offsets and network weights. Examining sampling locations alone, as done in [8], can result in misleading conclusions about Deformable ConvNets. 三种给出的空间支撑可视化类型，比[8]中使用的采样位置更加有信息量。比如，在常规CNN中，其采样位置是在网格上固定的，但实际上通过网络权重调整了其有效空间支撑。对于形变CNN也是一样的，其预测受到学习的偏移和网络权重的共同影响。单独检查采样位置，如[8]中所做的，得到的关于形变CNN的结论可能是有误导性质的。

Figure 2 (a)∼(b) display the spatial support of the 2fc node in the per-RoI detection head, which is directly followed by the classification and the bounding box regression branches. The visualization of effective bin locations suggests that bins on the object foreground generally receive larger gradients from the classification branch, and thus exert greater influence on prediction. This observation holds for both aligned RoIpooling and Deformable RoIpooling. In Deformable RoIpooling, a much larger proportion of bins cover the object foreground than in aligned RoIpooling, thanks to the introduction of learnable bin offsets. Thus, more information from relevant bins is available for the downstream Fast R-CNN head. Meanwhile, the error-bounded saliency regions in both aligned RoIpooling and Deformable RoIpooling are not fully focused on the object foreground, which suggests that image content outside of the RoI affects the prediction result. According to a recent study [6], such feature interference could be harmful for detection.

图2a-b展示了在per-ROI检测头中的2fc节点的空间支撑，其后就是分类和边界框回归分支。有效bin位置的可视化说明，在前景目标上的bins，一般都从分类分支上得到更大的梯度，因此对预测施加了更大的影响。这个观察对于对齐ROI池化和形变ROI池化都是对的。在形变ROI池化中，与对齐ROI池化相比，更大比例的bins覆盖了前景目标，这是由于可学习的bin偏移的引入。因此，从相关bins中的更多信息对于下游的Fast R-CNN头是可用的。同时，在对齐ROI池化和形变ROI池化中，误差界限显著性区域并没有完全聚焦在目标前景中，这说明，在RoI之外的图像内容影响了预测结果。根据[6]的最近研究，这种特征影响可能对检测是不利的。

While it is evident that Deformable ConvNets have markedly improved ability to adapt to geometric variation in comparison to regular ConvNets, it can also be seen that their spatial support may extend beyond the region of interest. We thus seek to upgrade Deformable ConvNets so that they can better focus on pertinent image content and deliver greater detection accuracy.

与常规CNN相比，形变CNN显著改进了对几何变化的适应能力，但也可以看到，其空间支撑超出了感兴趣区域。我们因此寻求来对形变CNN进行升级，使其更好的聚焦在合适的图像内容中，给出更好的检测准确率。

## 3. More Deformable ConvNets

To improve the network’s ability to adapt to geometric variations, we present changes to boost its modeling power and to help it take advantage of this increased capability.

为改进网络对几何变化的适应能力，我们提出变化来提升其建模能力，帮助其利用这种强化的能力。

### 3.1. Stacking More Deformable Conv Layers

Encouraged by the observation that Deformable ConvNets can effectively model geometric transformation on challenging benchmarks, we boldly replace more regular conv layers by their deformable counterparts. We expect that by stacking more deformable conv layers, the geometric transformation modeling capability of the entire network can be further strengthened.

形变CNN可以在很有挑战的基准测试中有效的对几何变换进行建模，受此鼓励，我们大胆了将更多的常规卷积层替换成其形变版本。我们期望，通过堆叠更多的形变卷积层，整个网络的几何变换建模能力可以进一步得到加强。

In this paper, deformable convolutions are applied in all the 3 × 3 conv layers in stages conv3, conv4, and conv5 in ResNet-50. Thus, there are 12 layers of deformable convolution in the network. In contrast, just three layers of deformable convolution are used in [8], all in the conv5 stage. It is observed in [8] that performance saturates when stacking more than three layers for the relatively simple and small-scale PASCAL VOC benchmark. Also, misleading offset visualizations on COCO may have hindered further exploration on more challenging benchmarks. In experiments, we observe that utilizing deformable layers in the conv3-conv5 stages achieves the best tradeoff between accuracy and efficiency for object detection on COCO. See Section 5.2 for details.

本文中，ResNet-50中的conv3, conv4和conv5阶段中所有的3x3卷积层都替换成了形变卷积。因此，网络中有12个形变卷积层。比较起来，[3]中只使用了3个形变卷积层，都是在conv5阶段中。[8]中观察到，当堆叠更多的层时，在相对简单和小规模的PASCAL VOC基准测试中，性能就达到了瓶颈。而且，在COCO上的有误导的偏移可视化，阻碍了在更有挑战的基准测试中的进一步探索。在试验中，我们观察到，在conv3-conv5阶段利用形变层，在COCO目标检测上可以得到准确率和效率的最佳折中。详见5.2节。

### 3.2. Modulated Deformable Modules

To further strengthen the capability of Deformable ConvNets in manipulating spatial support regions, a modulation mechanism is introduced. With it, the Deformable ConvNets modules can not only adjust offsets in perceiving input features, but also modulate the input feature amplitudes from different spatial locations / bins. In the extreme case, a module can decide not to perceive signals from a particular location / bin by setting its feature amplitude to zero. Consequently, image content from the corresponding spatial location will have considerably reduced or no impact on the module output. Thus, the modulation mechanism provides the network module another dimension of freedom to adjust its spatial support regions.

为进一步加强形变卷积在控制空间支撑区域的能力，提出了一种调节机制。有了这个，形变CNN模块不仅可以从感知输入特征中调整偏移，而且可以从不同的空间位置/bins中调节输入特征幅度。在极端情况下，一个模块可以通过将其特征幅度设为0，从而决定不从特定的位置/bin中感知信号。结果是，对应的空间位置上的图像内容对模块输出的影响将极大降低，甚至消失。因此，调节机制给网络模块提供了另一种自由度，来调整其空间支持区域。

Given a convolutional kernel of K sampling locations, let w_k and p_k denote the weight and pre-specified offset for the k-th location, respectively. For example, K = 9 and p_k ∈ {(−1, −1), (−1, 0), . . . , (1, 1)} defines a 3 × 3 convolutional kernel of dilation 1. Let x(p) and y(p) denote the features at location p from the input feature maps x and output feature maps y, respectively. The modulated deformable convolution can then be expressed as

给定一个卷积核，有K个采样位置，令w_k和p_k分别表示第k个位置的权重和预先指定的偏移。比如，K=9，p_k ∈ {(−1, −1), (−1, 0), . . . , (1, 1)}定义了一个3x3的卷积核，膨胀系数为1。令x(p)和y(p)分别表示输入特征图x和输出特征图y在位置p上的特征。调制形变卷积可以表示为

$$y(p) = \sum_{k=1}^K w_k ⋅ x(p+p_k+Δp_k) ⋅ Δm_k$$(1)

where ∆p_k and ∆m_k are the learnable offset and modulation scalar for the k-th location, respectively. The modulation scalar ∆m_k lies in the range [0, 1], while ∆p_k is a real number with unconstrained range. As p + p_k + ∆p_k is fractional, bilinear interpolation is applied as in [8] in computing x(p + p_k + ∆p_k). Both ∆p_k and ∆m_k are obtained via a separate convolution layer applied over the same input feature maps x. This convolutional layer is of the same spatial resolution and dilation as the current convolutional layer. The output is of 3K channels, where the first 2K channels correspond to the learned offsets {∆p_k}_k=1^K, and the remaining K channels are further fed to a sigmoid layer to obtain the modulation scalars {∆m_k}_k=1^K. The kernel weights in this separate convolution layer are initialized to zero. Thus, the initial values of ∆p_k and ∆m_k are 0 and 0.5, respectively. The learning rates of the added conv layers for offset and modulation learning are set to 0.1 times those of the existing layers.

其中∆p_k和∆m_k分别是第k个位置的可学习的偏移和调节量。调节标量∆m_k在[0, 1]范围内，而∆p_k是一个实数，没有数值范围。p + p_k + ∆p_k不是整数，在计算x(p + p_k + ∆p_k)时，和[8]一样采用了双线性差值。∆p_k和∆m_k是通过分离的卷积层应用到同样的输入特征图x得到的。这个卷积层与目前的卷积层有着相同的空间分辨率和膨胀系数。输出有3K个通道，前2K个通道对应着学习到的偏移{∆p_k}_k=1^K，剩余的K个通道进一步送入sigmoid层，以得到调制标量{∆m_k}_k=1^K。在这个分离的卷积层中的核权重初始化为0。因此，∆p_k和∆m_k的初始值分别为0和0.5。增加的偏移和调制卷积层的学习率数值为已有层的0.1倍。

The design of modulated deformable RoIpooling is similar. Given an input RoI, RoIpooling divides it into K spatial bins (e.g. 7 × 7). Within each bin, sampling grids of even spatial intervals are applied (e.g. 2 × 2). The sampled values on the grids are averaged to compute the bin output. Let ∆p_k and ∆m_k be the learnable offset and modulation scalar for the k-th bin. The output binning feature y(k) is computed as

调制形变ROI池化的设计是类似的。给定输入的ROI，ROI池化将其分成K个空间bins（如，7 × 7）。在每个bin中，使用了平均空间间隔的采样网格（如2 × 2）。在网格上的采样值进行了平均，以计算bin的输出。令∆p_k和∆m_k为第k个bin的可学习的偏移和调节标量。输出binning特征y(k)计算为

$$y(k) = \sum_{j=1}^{n_k} x(p_{kj} + Δp_k) ⋅ Δm_k/n_k$$(2)

where p_kj is the sampling location for the j-th grid cell in the k-th bin, and n_k denotes the number of sampled grid cells. Bilinear interpolation is applied to obtain features x(p_kj + ∆p_k). The values of ∆p_k and ∆m_k are produced by a sibling branch on the input feature maps. In this branch, RoIpooling generates features on the RoI, followed by two fc layers of 1024-D (initialized with Gaussian distribution of standard derivation of 0.01). On top of that, an additional fc layer produces output of 3K channels (weights initialized to be zero). The first 2K channels are the normalized learnable offsets, where element-wise multiplications with the RoI’s width and height are computed to obtain {∆p_k}_k=1^K. The remaining K channels are normalized by a sigmoid layer to produce {∆m_k}_k=1^K. The learning rates of the added fc layers for offset learning are the same as those of the existing layers.

其中p_kj是在第k个bin中第j个网格单元的采样位置，n_k表示采样的网格单元的数量。应用双线性差值来得到特征x(p_kj + ∆p_k)。∆p_k和∆m_k的值，通过在输入特征图的一个sibling分支得到。在这个分支中，ROI池化生成ROI上的特征，然后是2个1024-D的fc层（初始化为标准差为0.01的高斯分布值）。在此之上，一个额外的fc层生成了3K通道的输出（权重初始化为0）。前2K个通道是归一化的可学习偏移，其中计算与ROI的宽度和高度的逐元素乘积，以得到{∆p_k}_k=1^K。剩余的K个通道通过一个sigmoid层来归一化，以生成{∆m_k}_k=1^K。增加进行偏移学习的fc层的学习速率与现有的层相同。

### 3.3. R-CNN Feature Mimicking

As observed in Figure 2, the error-bounded saliency region of a per-RoI classification node can stretch beyond the RoI for both regular ConvNets and Deformable ConvNets. Image content outside of the RoI may thus affect the extracted features and consequently degrade the final results of object detection.

如图2所示，一个per-ROI分类节点的误差界限显著性区域会超出ROI之外，常规CNN和形变CNN都是如此。ROI之外的图像内容因此会影响提取的特征，结果使目标检测的最终结果变差。

In [6], the authors find redundant context to be a plausible source of detection error for Faster R-CNN. Together with other motivations (e.g., to share fewer features between the classification and bounding box regression branches), the authors propose to combine the classification scores of Faster R-CNN and R-CNN to obtain the final detection score. Since R-CNN classification scores are focused on cropped image content from the input RoI, incorporating them would help to alleviate the redundant context problem and improve detection accuracy. However, the combined system is slow because both the Faster-RCNN and R-CNN branches need to be applied in both training and inference.

在[6]中，作者发现，对于Faster R-CNN来说，冗余的上下文，是检测错误的一个可能的根源。与其他动机一起（如，在分类和边界框回归分支中共享更少的特征），作者提出将Faster R-CNN和R-CNN的分类分数结合到一起，以得到最终的检测分数。由于R-CNN分类分数是聚焦在输入ROI的剪切图像内容中的，将其结合进来，会帮助减缓冗余的上下文问题，改进检测准确率。但是，结合的系统很慢，因为Faster R-CNN和R-CNN分支都要应用在训练和推理分支中。

Meanwhile, Deformable ConvNets are powerful in adjusting spatial support regions. For Deformable ConvNets v2 in particular, the modulated deformable RoIpooling module could simply set the modulation scalars of bins in a way that excludes redundant context. However, our experiments in Section 5.3 show that even with modulated deformable modules, such representations cannot be learned well through the standard Faster R-CNN training procedure. We suspect that this is because the conventional Faster R-CNN training loss cannot effectively drive the learning of such representations. Additional guidance is needed to steer the training.

同时，形变CNN在调节空间支撑区域中很强。特别的，对于形变CNN v2，调制形变ROI池化模块可以很简单的设置bins的调制常数，排除掉冗余的上下文。但是，我们在5.3节中的试验表明，即使使用了调制形变模块，这样的表示也不可能通过标准Faster R-CNN训练过程学习的很好。我们推测这是因为，传统Faster R-CNN的训练损失不能有效的推动这种表示的学习。需要额外的引导，来推动训练。

Motivated by recent work on feature mimicking [2, 22, 28], we incorporate a feature mimic loss on the per-RoI features of Deformable Faster R-CNN to force them to be similar to R-CNN features extracted from cropped images. This auxiliary training objective is intended to drive Deformable Faster R-CNN to learn more “focused” feature representations like R-CNN. We note that, based on the visualized spatial support regions in Figure 2, a focused feature representation may well not be optimal for negative RoIs on the image background. For background areas, more context information may need to be considered so as not to produce false positive detections. Thus, the feature mimic loss is enforced only on positive RoIs that sufficiently overlap with ground-truth objects.

受到最近特征模仿工作的启发，我们在形变Faster R-CNN上per-ROI特征上加入了特征模仿损失，以迫使其与从图像剪切块中提取的R-CNN特征类似。这个辅助的训练目标函数的目标是，推动形变Faster R-CNN学习更聚焦的特征表示，更像R-CNN。我们注意到，基于图2中的可视化空间支撑区域，聚焦的特征表示对于图像背景上的负ROIs并不是最优的。对于背景区域，需要更多的上下文信息来考虑，以不产生假阳性检测。因此，特征模仿损失只在正RoI上施加，与真值目标有足够的重叠。

The network architecture for training Deformable Faster R-CNN is presented in Figure 3. In addition to the Faster R-CNN network, an additional R-CNN branch is added for feature mimicking. Given an RoI b for feature mimicking, the image patch corresponding to it is cropped and resized to 224 × 224 pixels. In the R-CNN branch, the backbone network operates on the resized image patch and produces feature maps of 14 × 14 spatial resolution. A (modulated) deformable RoIpooling layer is applied on top of the feature maps, where the input RoI covers the whole resized image patch (top-left corner at (0, 0), and height and width are 224 pixels). After that, 2 fc layers of 1024-D are applied, producing an R-CNN feature representation for the input image patch, denoted by f_RCNN(b). A (C+1)-way Softmax classifier follows for classification, where C denotes the number of foreground categories, plus one for background. The feature mimic loss is enforced between the R-CNN feature representation f_RCNN(b) and the counterpart in Faster R-CNN, f_FRCNN(b), which is also 1024-D and is produced by the 2 fc layers in the Fast R-CNN head. The feature mimic loss is defined on the cosine similarity between f_RCNN(b) and f_FRCNN(b), computed as

训练形变Faster R-CNN的网络架构如图3所示。在Faster R-CNN网络的基础之上，加上了R-CNN分支，以进行特征模仿。给定一个RoI b进行特征模仿，对应的图像块剪切出来，改变到224x224大小。在R-CNN分支中，骨干网络在改变大小的图像块上进行运算，产生14x14大小的特征图。一个（调制的）形变ROI池化层应用到这些特征图上，其中输入RoI覆盖了整个改变大小的图像块。在这之后，应用了2个1024-D的fc层，对输入图像块生成了一个R-CNN特征表示，表示为f_RCNN(b)。一个C+1路的softmax分类器随后进行分类，其中C表示前景类别的数量，加上一个背景类别。特征模仿损失加入到R-CNN特征表示f_RCNN(b)和Faster R-CNN特征表示f_FRCNN(b)之间，这也是1024维的，是由Fast R-CNN头的2个fc层生成的。特征模仿损失定义为f_RCNN(b)和f_FRCNN(b)的cos相似性上，计算为

$$L_{mimic} = \sum_{b∈Ω} [1-cos(f_{RCNN}(b), f_{FRCNN}(b))]$$(3)

where Ω denotes the set of RoIs sampled for feature mimic training. In the SGD training, given an input image, 32 positive region proposals generated by RPN are randomly sampled into Ω. A cross-entropy classification loss is enforced on the R-CNN classification head, also computed on the RoIs in Ω. Network training is driven by the feature mimic loss and the R-CNN classification loss, together with the original loss terms in Faster R-CNN. The loss weights of the two newly introduced loss terms are 0.1 times those of the original Faster R-CNN loss terms. The network parameters between the corresponding modules in the R-CNN and the Faster R-CNN branches are shared, including the backbone network, (modulated) deformable RoIpooling, and the 2 fc heads (the classification heads in the two branches are unshared). In inference, only the Faster R-CNN network is applied on the test images, without the auxiliary R-CNN branch. Thus, no additional computation is introduced by R-CNN feature mimicking in inference.

其中Ω表示采样用于特征模仿训练的RoIs的集合。在SGD训练中，给定输入图像，RPN生成32个正区域建议，随机采样得到Ω。在R-CNN分类头中采用交叉熵分类损失，在Ω中的RoIs中计算。网络训练是受特征模仿损失和R-CNN分类损失，和Faster R-CNN的原始损失驱动的。新引入的两个损失项的损失权重，是原始Faster R-CNN损失项的0.1倍。在R-CNN和Faster R-CNN分支中对应模块的网络参数是共享的，包括骨干网络，调制形变ROI池化，和2个fc头（两个分支中的分类头是不共享的）。在推理中，只有Faster R-CNN网络应用到测试图像中，不需要辅助R-CNN分支。因此，在推理阶段，R-CNN特征模仿没有引入额外的计算。

## 4. Related Work

**Deformation Modeling** is a long-standing problem in computer vision, and there has been tremendous effort in designing translation-invariant features. Prior to the deep learning era, notable works include scale-invariant feature transform (SIFT) [30], oriented FAST and rotated BRIEF (ORB) [34], and deformable part-based models (DPM) [12]. Such works are limited by the inferior representation power of handcrafted features and the constrained family of geometric transformations they address (e.g., affine transformations). Spatial transformer networks (STN) [25] is the first work on learning translation-invariant features for deep CNNs. It learns to apply global affine transformations to warp feature maps, but such transformations inadequately model the more complex geometric variations encountered in many vision tasks. Instead of performing global parametric transformations and feature warping, Deformable ConvNets sample feature maps in a local and dense manner, via learnable offsets in the proposed deformable convolution and deformable RoIpooling modules. Deformable ConvNets is the first work to effectively model geometric transformations in complex vision tasks (e.g., object detection and semantic segmentation) on challenging benchmarks.

形变建模是计算机视觉中的一个长久问题，设计变换不变的特征有非常多的工作。在深度学习时代之前，值得注意的工作包括SIFT，ORB和DPM。这些工作受到手工特征的低表示能力，和他们处理的有限的几何变换族所局限。STN是第一个学习变换不变特征的DCNN工作，学习应用全局仿射变换以对特征图变形，但这种变换不足以对很多视觉任务中遇到的复杂几何变化进行建模。形变CNN并没有进行全局参数变换和特征变形，而是以局部和密集的方式，通过形变卷积和形变ROI池化模块中的可学习偏移，对特征图进行采样。形变CNNs是在很有挑战的基准测试中，在复杂视觉任务中有效建模几何变换的第一个工作（如，目标检测和语义分割）。

Our work extends Deformable ConvNets by enhancing its modeling power and facilitating network training. This new version of Deformable ConvNets yields significant performance gains over the original model.

我们的工作拓展了形变CNN，增强了其建模能力，促进了网络训练。这个新版本的形变CNN比原始模型有了显著的性能提升。

**Relation Networks and Attention Modules** are first proposed in natural language processing [14, 15, 4, 36] and physical system modeling [3, 38, 23, 35, 10, 32]. An attention / relation module effects an individual element (e.g., a word in a sentence) by aggregating features from a set of elements (e.g., all the words in the sentence), where the aggregation weights are usually defined on feature similarities among the elements. They are powerful in capturing long-range dependencies and contextual information in these tasks. Recently, the concurrent works of [24] and [37] successfully extend relation networks and attention modules to the image domain, for modeling long-range object-object and pixel-pixel relations, respectively. In [19], a learnable region feature extractor is proposed, unifying the previous region feature extraction modules from the pixel-object relation perspective. A common issue with such approaches is that the aggregation weights and the aggregation operation need to be computed on the elements in a pairwise fashion, incurring heavy computation that is quadratic to the number of elements (e.g., all the pixels in an image). Our developed approach can be perceived as a special attention mechanism where only a sparse set of elements have non-zero aggregation weights (e.g., 3 × 3 pixels from among all the image pixels). The attended elements are specified by the learnable offsets, and the aggregation weights are controlled by the modulation mechanism. The computational overhead is just linear to the number of elements, which is negligible compared to that of the entire network (See Table 1).

关系网络和注意力模块首先在NLP和物理系统建模中提出。一个注意力/关系模块，通过聚积元素集合的特征，来影响单个元素，聚积权重通常是基于元素间的特征相似度定义起来的。在这些任务中，在捕获长程依赖关系和上下文信息中非常强大。最近，[24, 37]同时成功的将关系网络和注意力模块拓展到了图像领域，分别对长程的目标-目标关系和像素-像素关系进行建模。在[19]中，提出了一个可学习的区域特征提取器，从像素-目标的关系的角度，统一了之前的区域特征提取模块。这种方法的共同问题是，聚积权重和聚积运算需要以成对的方式在元素之间计算，带来了很大的计算量，是元素数量的平方量级。我们提出的方法可以视为一种特殊的注意力机制，只有元素的稀疏集上有非零聚积权重。参与的元素是由可学习的偏移指定的，聚积权重是由调节机制控制的。计算代价与元素数量是线性关系，与整个网络的相比，是可以忽略的。

**Spatial Support Manipulation**. For atrous convolution, the spatial support of convolutional layers has been enlarged by padding zeros in the convolutional kernels [5]. The padding parameters are handpicked and predetermined. In active convolution [26], which is contemporary with Deformable ConvNets, convolutional kernel offsets are learned via back-propagation. But the offsets are static model parameters fixed after training and shared over different spatial locations. In a multi-path network for object detection [40], multiple RoIpooling layers are employed for each input RoI to better exploit multi-scale and context information. The multiple RoIpooling layers are centered at the input RoI, and are of different spatial scales. A common issue with these approaches is that the spatial support is controlled by static parameters and does not adapt to image content.

空间支撑的操作。对于atrous卷积，卷积层的空间支撑增大了，卷积核中填充了很多0。这种填充的参数是手工选择的，预先确定的。在active卷积[26]中，与形变卷积是同时的，卷积核偏移是通过反向传播学习得到的。但偏移是训练后的固定的模型参数，在不同的空间位置中共享。在目标检测的多块网络[40]中，对每个输入的RoI采用了多个RoI池化层，以更好的利用多尺度和上下文信息。多RoI池化层是以输入RoI为中心的，具有不同的空间尺度。这些方法的共同问题是，空间支撑是由静态参数控制的，不是对图像内容自适应的。

**Effective Receptive Field and Salient Region**. Towards better interpreting how a deep network functions, significant progress has been made in understanding which image regions contribute most to network prediction. Recent works on effective receptive fields [31] and salient regions [41, 44, 13, 7] reveal that only a small proportion of pixels in the theoretical receptive field contribute significantly to the final network prediction. The effective support region is controlled by the joint effect of network weights and sampling locations. Here we exploit the developed techniques to better understand the network behavior of Deformable ConvNets. The resulting observations guide and motivate us to improve over the original model.

有效感受野和显著性区域。为更好的揭示DCNN的功能，哪个图像区域对网络预测贡献最大，有了显著的进展。最近在有效感受野[31]和显著性区域上的工作揭示了，理论感受野中只有一小部分像素，对最终网络预测的贡献很大。有效支撑区域是由网络权重和采样位置的共同效果控制的。这里我们采用提出的技术来更好的理解形变CNN的网络行为。得到的观察结果，指引并推动我们来对原始模型进行改进。

**Network Mimicking and Distillation** are recently introduced techniques for model acceleration and compression. Given a large teacher model, a compact student model is trained by mimicking the teacher model output or feature responses on training images [2, 22, 28]. The hope is that the compact model can be better trained by distilling knowledge from the large model.

网络模拟和蒸馏是最近提出的模型加速和压缩技术。给定一个大型teacher模型，通过模拟teacher模型输出或在训练图像上的特征响应，训练一个紧凑的student模型。希望紧凑模型可以通过从大型模型中蒸馏知识，来更好的训练。

Here we employ a feature mimic loss to help the network learn features that reflect the object focus and classification power of R-CNN features. Improved accuracy is obtained and the visualized spatial supports corroborate this approach.

这里我们采用了一个特征模仿损失，来帮助网络学习特征，反应目标焦点和R-CNN的分类能力。得到了改进的准确率，可视化的空间支撑确认了这种方法的有效性。

## 5. Experiments

### 5.1. Experiment Settings

Our models are trained on the 118k images of the COCO 2017 train set. In ablation, evaluation is done on the 5k images of the COCO 2017 validation set. We also evaluate performance on the 20k images of the COCO 2017 test-dev set. The standard mean average-precision scores at different box and mask IoUs are used for measuring object detection and instance segmentation accuracy, respectively.

我们的模型是在COCO 2017训练集的118k图像上进行训练的。评估是在COCO 2017验证集的5k图像上进行的。我们还在COCO 2017 test-dev集的20k图像上评估了性能。在不同box上的标准mAP和mask IoUs分别用于度量目标检测和实例分割的准确率。

Faster R-CNN and Mask R-CNN are chosen as the baseline systems. ImageNet [9] pre-trained ResNet-50 is utilized as the backbone. The implementation of Faster R-CNN is the same as in Section 3.3. For Mask R-CNN, we follow the implementation in [20]. To turn the networks into their deformable counterparts, the last set of 3 × 3 regular conv layers (close to the output in the bottom-up computation) are replaced by (modulated) deformable conv layers. Aligned RoIpooling is replaced by (modulated) deformable RoIpooling. Specially for Mask R-CNN, the two aligned RoIpooling layers with 7 × 7 and 14 × 14 bins are replaced by two (modulated) deformable RoIpooling layers with the same bin numbers. In R-CNN feature mimicking, the feature mimic loss is enforced on the RoI head for classification only (excluding that for mask estimation). For both systems, the choice of hyper-parameters follows the latest Detectron [18] code base except for the image resolution, which is briefly presented here. In both training and inference, images are resized so that the shorter side is 1,000 pixels. Anchors of 5 scales and 3 aspect ratios are utilized. 2k and 1k region proposals are generated at a non-maximum suppression threshold of 0.7 at training and inference respectively. In SGD training, 256 anchor boxes (of positive-negative ratio 1:1) and 512 region proposals (of positive-negative ratio 1:3) are sampled for backpropagating their gradients. In our experiments, the networks are trained on 8 GPUs with 2 images per GPU for 16 epochs. The learning rate is initialized to 0.02 and is divided by 10 at the 10-th and the 14-th epochs. The weight decay and the momentum parameters are set to 10^−4 and 0.9, respectively.

Faster R-CNN和Mask R-CNN选作基准系统。ImageNet预训练的ResNet-50用做骨干。Faster R-CNN的实现与3.3节一样。对于Mask R-CNN，我们按照[20]的实现。为将网络变为形变版，3x3常规卷积层的最后集合替换为（调制的）形变卷积层。对齐ROI池化替换为（调制的）形变ROI池化。对Mask R-CNN，两个对齐ROI池化层，大小为7x7和14x14的bins，替换为两个（调制的）形变ROI池化层，有着相同的bins数量。在R-CNN特征模仿中，在RoI头上加入了特征模仿损失，只用于分类。对两个系统，超参数的选择按照最新的Detectron代码库进行，但是除了分辨率，这里简单的进行介绍。在训练和推理中，图像大小进行改变，其短边成为1000像素。利用了5个尺度3种纵横比的锚框。在训练和推理中，分别生成了2k和1k个区域建议，nms的阈值为0.7。在SGD训练中，采样得到256个锚框（正负比率1:1），512个区域候选（正负比率1：3），以将其梯度进行反向传播。在我们的试验中，网络在8个GPUs上进行训练，每个GPU 2幅图像，训练16个epochs。学习速率初始为0.02，在第10和第14个epoch时除以10。权重衰减和动量参数分别设为10^−4和0.9。

### 5.2. Enriched Deformation Modeling

The effects of enriched deformation modeling are examined from ablations shown in Table 1. The baseline with regular CNN modules obtains an AP^bbox score of 34.7% for
Faster R-CNN, and AP^bbox and AP^mask scores of 36.6% and 32.2% respectively for Mask R-CNN. To obtain a DCNv1 baseline, we follow the original Deformable ConvNets paper by replacing the last three layers of 3 × 3 convolution in the conv5 stage and the aligned RoIpooling layer by their deformable counterparts. This DCNv1 baseline achieves an AP^bbox score of 38.0% for Faster R-CNN, and AP^bbox and AP^mask scores of 40.4% and 35.3% respectively for Mask R-CNN. The deformable modules considerably improve accuracy as observed in [8].

增强的形变建模的效果，通过分离试验进行检查，如表1所示。常规CNN模块的基准，Faster R-CNN得到了34.7%的AP^bbox，Mask R-CNN得到了36.6%和32.2%的AP^bbox和AP^mask。为得到DCNv1的基准，我们按照原始DCNv1文章，将conv5阶段的最后三层的3x3卷积替换为形变conv，将对齐ROI池化层替换为形变版。DCNv1基准在Faster R-CNN上得到了38.0%的AP^bbox，在Mask R-CNN上分别得到了40.4%和35.3%的AP^bbox和AP^mask。形变模块显著改进了准确率。

By replacing more 3 × 3 regular conv layers by their deformable counterparts, the accuracy of both Faster R-CNN and Mask R-CNN steadily improve, with gains between 2.0% and 3.0% for APbbox and APmask scores when the conv layers in conv3-conv5 are replaced. No additional improvement is observed on the COCO benchmark by further replacing the regular conv layers in the conv2 stage. By upgrading the deformable modules to modulated deformable modules, we obtain further gains between 0.3% and 0.7% in APbbox and APmask scores. In total, enriching the deformation modeling capability yields a 41.7% APbbox score on Faster R-CNN, which is 3.7% higher than that of the DCNv1 baseline. On Mask R-CNN, 43.1% APbbox and 37.3% APmask scores are obtained with the enriched deformation modeling, which are respectively 2.7% and 2.0% higher than those of the DCNv1 baseline. Note that the added parameters and FLOPs for enriching the deformation modeling are minor compared to those of the overall networks.

将更多的3x3常规卷积层替换为形变卷积，Faster R-CNN和Mask R-CNN的准确率持续稳步提升，当conv3-conv5的卷积层替换掉时，APbbox和APmask分数提升了2.0%和3.0%。将conv2阶段中的卷积层也替换掉，在COCO上没有观察到更多的改进。将形变模块升级到调制形变模块，APbbox和APmask进一步提升了0.3%和0.7%。总计上，增强形变建模能力，Faster R-CNN得到了41.7%的APbbox，比DCNv1提升了3.7%。在Mask R-CNN上，增强形变建模能力得到了43.1%的APbbox和37.3%的APmask，比DCNv1基准分别提升了2.7%和2.0%。增强形变建模能力所增加的参数和FLOPs，与整个网络的相比，是非常小的。

Shown in Figure 1 (b)∼(c), the spatial support of the enriched deformable modeling exhibits better adaptation to image content compared to that of DCNv1. 如图1b-c所示，与DCNv1相比，强化形变建模能力的空间支撑对图像内容得到了更好的自适应能力。

Table 2 presents the results at input image resolution of 800 pixels, which follows the default setting in the Detectron code base. The same conclusion holds. 表2给出了输入图像分辨率800的结果，这是按照Detectron中的默认设置来的。可以得到相同的结论。

### 5.3. R-CNN Feature Mimicking

Ablations of the design choices in R-CNN feature mimicking are shown in Table 3. With the enriched deformation modeling, R-CNN feature mimicking further improves the APbbox and APmask scores by about 1% to 1.4% in both the Faster R-CNN and Mask R-CNN systems. Mimicking features of positive boxes on the object foreground is found to be particularly effective, and the results when mimicking all the boxes or just negative boxes are much lower. As shown in Figure 2 (c)∼(d), feature mimicking can help the network features better focus on the object foreground, which is beneficial for positive boxes. For the negative boxes, the network tends to exploit more context information (see Figure 2), where feature mimicking would not be helpful.

设计选项中，R-CNN特征模仿的分离试验如表3所示。带有增强的形变建模能力时，R-CNN特征模仿进一步将APbbox和APmask分数改进了大约1%和1.4%。在目标前景中模仿正框的特征，发现特别有效，而模仿所有框或只有负框的结果则要低很多。如图2c-d所示，特征模仿可以帮助网络特征更好的聚焦在目标前景中，这对于正框是有好处的。对于负框，网络倾向于更多的利用上下文信息（见图2），而特征模仿不会有所帮助。

We also apply R-CNN feature mimicking to regular ConvNets without any deformable layers. Almost no accuracy gains are observed. The visualized spatial support regions are shown in Figure 2 (e), which are not focused on the object foreground even with the auxiliary mimic loss. This is likely because it is beyond the representation capability of regular ConvNets to focus features on the object foreground, and thus this cannot be learned.

我们还应用R-CNN特征模仿到常规CNN中，没有任何形变层，几乎没有观察到性能提升。可视化的空间支撑区域如图2e所示，即使有辅助的模仿损失，也没有聚焦到目标前景中。这是很有可能的，因为要让常规CNN将特征聚焦在前景中，这超出了其能力，因为没有学习得到。

### 5.4. Application on Stronger Backbones

Results on stronger backbones, by replacing ResNet-50 with ResNet-101 and ResNext-101 [39], are presented in Table 4. For the entries of DCNv1, the regular 3 × 3 conv layers in the conv5 stage are replaced by the deformable counterpart, and aligned RoIpooling is replaced by deformable RoIpooling. For the DCNv2 entries, all the 3 × 3 conv layers in the conv3-conv5 stages are of modulated deformable convolution, and modulated deformable RoIpooling is used instead, with supervision from the R-CNN feature mimic loss. DCNv2 is found to outperform regular ConvNet and DCNv1 considerably on all the network backbones.

将ResNet-50替换成ResNet-101和ResNext-101，在更强的骨干上的结果，如表4所示。对于DCNv1的入口，在conv5阶段的常规3x3卷积层，替换成形变部分，对齐ROI池化替换成了形变ROI池化。对于DCNv2入口，conv3-conv5阶段的所有3x3卷积层，都替换成了调制形变卷积，而且使用了调制形变ROI池化，而且还有R-CNN特征模仿损失的监督。DCNv2在所有网络骨干上，都明显超过了常规CNN和DCNv1。

## 6. Conclusion

Despite the superior performance of Deformable ConvNets in modeling geometric variations, its spatial support extends well beyond the region of interest, causing features to be influenced by irrelevant image content. In this paper, we present a reformulation of Deformable ConvNets which improves its ability to focus on pertinent image regions, through increased modeling power and stronger training. Significant performance gains are obtained on the COCO benchmark for object detection and instance segmentation.

尽管形变CNN在建模几何形变上有优异的性能，其空间支撑超出了ROI，导致特征受到无关图像内容的影响。本文中，我们提出了形变CNN的重新表述，改进了其能力，聚焦在合适的图像区域中。在COCO基准测试中，在目标检测和实例分割中，得到了明显的性能改进。