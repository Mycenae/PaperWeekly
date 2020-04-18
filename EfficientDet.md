# EfficientDet: Scalable and Efficient Object Detection

Mingxing Tan Ruoming Pang Quoc V. Le Google Research, Brain Team {tanmingxing, rpang, qvl}@google.com

## 0. Abstract

Model efficiency has become increasingly important in computer vision. In this paper, we systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency. First, we propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multi-scale feature fusion; Second, we propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. Based on these optimizations and EfficientNet backbones, we have developed a new family of object detectors, called EfficientDet, which consistently achieve much better efficiency than prior art across a wide spectrum of resource constraints. In particular, with single-model and single-scale, our EfficientDet-D7 achieves state-of-the-art 52.2 AP on COCO test-dev with 52M parameters and 325B FLOPs1, being 4x – 9x smaller and using 13x – 42x fewer FLOPs than previous detectors. Code is available at https://github.com/google/automl/tree/master/efficientdet.

模型效率在计算机视觉中正变得越来越重要。本文中，我们系统的研究了目标检测的神经网络架构涉及的选择，提出了几种关键的优化来改进效率。首先，我们提出了一种加权的双向特征金字塔网络(Bi-FPN)，这可以进行简单和快速的多尺度特征融合；第二，我们提出了一种复合缩放方法，统一的对所有骨干、特征网络和框/类别预测网络，的分辨率、深度和宽度，进行同时缩放。基于这些优化和EfficientNet骨干网络，我们提出了一族新的目标检测器，称为EfficientDet，可以比之前最好的模型在很多的资源限制中一直得到好的多的效率。特别是，在单模型和单尺度下，我们的EfficientDet-D7在COCO test-dev中，用52M个参数和325B FLOPs，得到了52.2 AP，比之前的检测器小4-9倍，FLOPs少13x-42x。代码已经开源。

## 1. Introduction

Tremendous progresses have been made in recent years towards more accurate object detection; meanwhile, state-of-the-art object detectors also become increasingly more expensive. For example, the latest AmoebaNet-based NAS-FPN detector [42] requires 167M parameters and 3045B FLOPs (30x more than RetinaNet [21]) to achieve state-of-the-art accuracy. The large model sizes and expensive computation costs deter their deployment in many real-world applications such as robotics and self-driving cars where model size and latency are highly constrained. Given these real-world resource constraints, model efficiency becomes increasingly important for object detection.

最近几年，在更准确的目标检测上，得到了非常多的进展；同时，目前最好的目标检测器正变得越来越代价昂贵。比如，最新的基于AmoebaNet的NAS-FPN检测器，需要167M参数和3405B FLOPs（比RetinaNet多30x），才能得到目前最好的准确率。模型非常大，计算量非常大，阻碍了其在很多真实世界的应用中的部署，如机器人和自动驾驶汽车，其中模型大小和延迟是高度受限的。给定这些真实世界的资源限制，目标检测的模型效率正变得越来越重要。

There have been many previous works aiming to develop more efficient detector architectures, such as one-stage [24, 30, 31, 21] and anchor-free detectors [18, 41, 37], or compress existing models [25, 26]. Although these methods tend to achieve better efficiency, they usually sacrifice accuracy. Moreover, most previous works only focus on a specific or a small range of resource requirements, but the variety of real-world applications, from mobile devices to datacenters, often demand different resource constraints.

有很多之前的工作，其目标是提出更高效的检测器架构，如单阶段的，无锚框的检测器，或压缩现有的模型。虽然这些方法会得到更高的效率，但他们通常牺牲的准确率。而且，多数之前的工作只关注在一个特定或更小范围内的资源需求中，但多数真实世界的应用，从移动设备到数据中心，通常要求不同的资源限制。

A natural question is: Is it possible to build a scalable detection architecture with both higher accuracy and better efficiency across a wide spectrum of resource constraints (e.g., from 3B to 300B FLOPs)? This paper aims to tackle this problem by systematically studying various design choices of detector architectures. Based on the one-stage detector paradigm, we examine the design choices for backbone, feature fusion, and class/box network, and identify two main challenges:

一个更自然的问题是：是否可能构建一个可缩放的检测架构，在很宽的资源限制范围内（如，从3B到300B FLOPs），同时拥有更高的效率和更好的准确性？本文的目标是，通过系统的研究各种检测器架构的设计选择，处理这个问题。基于单阶段检测器的范式，我们研究了骨干网络、特征融合网络、类别/框网络的网络的设计选择，发现了两个主要的挑战：

Challenge 1: efficient multi-scale feature fusion – Since introduced in [20], FPN has been widely used for multi-scale feature fusion. Recently, PANet [23], NAS-FPN [8], and other studies [17, 15, 39] have developed more network structures for cross-scale feature fusion. While fusing different input features, most previous works simply sum them up without distinction; however, since these different input features are at different resolutions, we observe they usually contribute to the fused output feature unequally. To address this issue, we propose a simple yet highly effective weighted bi-directional feature pyramid network (BiFPN), which introduces learnable weights to learn the importance of different input features, while repeatedly applying top-down and bottom-up multi-scale feature fusion.

挑战1:高效的多尺度特征融合 - 自从[20]中提出了FPN，就在多尺度特征融合中得到了广泛的使用。最近，PANet，NAS-FPN和其他研究已经提出了更多的网络架构，进行跨尺度的特征融合。多数之前的工作融合了不同的输入特征，但只是不加区别的对其进行了叠加；但是因为这些不同的输入特征是不同分辨率的，我们观察到他们通常对融合的输出特征的贡献并不均衡。为解决这个问题，我们提出了一种简单但高度有效的加权双向特征金字塔网络(BiFPN)，提出可学习的权重，以学习不同的输入特征的重要性，同时重复的应用自上而下和自下而上的多尺度特征融合。

Challenge 2: model scaling – While previous works mainly rely on bigger backbone networks [21, 32, 31, 8] or larger input image sizes [11, 42] for higher accuracy, we observe that scaling up feature network and box/class prediction network is also critical when taking into account both accuracy and efficiency. Inspired by recent works [36], we propose a compound scaling method for object detectors, which jointly scales up the resolution/depth/width for all backbone, feature network, box/class prediction network.

挑战2:，模型缩放 - 之前的工作主要依赖于更大的骨干网络，或更大的输入图像大小，以得到更高的准确率，但我们观察到，特征网络和框/类别预测网络的放大，在考虑到准确率和效率的同时，也非常关键。受到最近的工作的启发，我们提出了一种符合缩放方法，进行目标检测，这对所有骨干、特征网络、框/类别预测网络的分辨率/深度/宽度进行联合缩放。

Finally, we also observe that the recently introduced EfficientNets [36] achieve better efficiency than previous commonly used backbones. Combining EfficientNet backbones with our propose BiFPN and compound scaling, we have developed a new family of object detectors, named EfficientDet, which consistently achieve better accuracy with much fewer parameters and FLOPs than previous object detectors. Figure 1 and Figure 4 show the performance comparison on COCO dataset [22]. Under similar accuracy constraint, our EfficientDet uses 28x fewer FLOPs than YOLOv3 [31], 30x fewer FLOPs than RetinaNet [21], and 19x fewer FLOPs than the recent ResNet based NAS-FPN [8]. In particular, with single-model and single test-time scale, our EfficientDet-D7 achieves state-of-the-art 52.2 AP with 52M parameters and 325B FLOPs, outperforming previous best detector [42] with 1.5 AP while being 4x smaller and using 13x fewer FLOPs. Our EfficientDet is also up to 3x to 8x faster on GPU/CPU than previous detectors.

最后，我们还观察到，最近提出的EfficientNets比之前常用的骨干取得了更高的效率。将EfficientNet骨干与我们提出的BiFPN和复合缩放结合到一起，我们提出了一族新的目标检测器，名为EfficientDet，比之前的目标检测器使用的参数量和FLOPs量少的多的情况下，可以一直得到更好的准确率。图1和图4给出了在COCO数据集上的性能比较。在类似的准确率约束下，我们的EfficientDet使用的FLOPs比YOLOv3少28倍，比RetinaNet少30倍，比最近的基于ResNet的NAS-FPN的FLOPs少19x。特别是，对于单模型和单测试时尺度，我们的EfficientDet-D7用52M的参数和325B的FLOPs，在COCO数据集上取得了目前最好的52.2 AP的结果，超过了之前最好的检测器1.5 AP，而模型小了4倍，使用的FLOPs小了13x。我们的EfficientDet比之前的检测器，在GPU/CPU上快了3x到8x。

With simple modifications, we also demonstrate that our single-model single-scale EfficientDet achieves 81.74% mIOU accuracy with 18B FLOPs on Pascal VOC 2012 semantic segmentation, outperforming DeepLabV3+ [4] by 1.7% better accuracy with 9.8x fewer FLOPs.

只需要很简单的修改，我们还证明了，我们的单模型单尺度EfficientDet，在Pascal VOC 2012语义分割上，使用18B FLOPs上取得了81.74% mIOU的准确率，超过了DeepLabV3+ 1.7%。FLOPs只有9.8x。

## 2. Related Work

**One-Stage Detectors**: Existing object detectors are mostly categorized by whether they have a region-of-interest proposal step (two-stage [9, 32, 3, 11]) or not (one-stage [33, 24, 30, 21]). While two-stage detectors tend to be more flexible and more accurate, one-stage detectors are often considered to be simpler and more efficient by leveraging predefined anchors [14]. Recently, one-stage detectors have attracted substantial attention due to their efficiency and simplicity [18, 39, 41]. In this paper, we mainly follow the one-stage detector design, and we show it is possible to achieve both better efficiency and higher accuracy with optimized network architectures.

**单阶段检测器**：现有的目标检测器的主要分类，是基于其是否有一个感兴趣区域建议的步骤，分为两阶段或单阶段的。两阶段检测器一般会更灵活，更准确，单阶段检测器一般认为更简单，更有效率，因为利用了预定义的锚框。最近，单阶段的检测器吸引了很多注意力，因为其高效和简单性。在本文中，我们主要遵循单阶段检测器的设计，我们展示了，使用优化的网络架构，是可能同时得到更好的效率和更高的准确率的。

**Multi-Scale Feature Representations**: One of the main difficulties in object detection is to effectively represent and process multi-scale features. Earlier detectors often directly perform predictions based on the pyramidal feature hierarchy extracted from backbone networks [2, 24, 33]. As one of the pioneering works, feature pyramid network (FPN) [20] proposes a top-down pathway to combine multi-scale features. Following this idea, PANet [23] adds an extra bottom-up path aggregation network on top of FPN; STDL [40] proposes a scale-transfer module to exploit cross-scale features; M2det [39] proposes a U-shape module to fuse multi-scale features, and G-FRNet [1] introduces gate units for controlling information flow across features. More recently, NAS-FPN [8] leverages neural architecture search to automatically design feature network topology. Although it achieves better performance, NAS-FPN requires thousands of GPU hours during search, and the resulting feature network is irregular and thus difficult to interpret. In this paper, we aim to optimize multi-scale feature fusion with a more intuitive and principled way.

**多尺度特征表示**：目标检测的一个主要困难，是高效的表示和处理多尺度特征。早期的检测器，通常基于从骨干网络中提出的特征金字塔的层次结构，直接进行预测。一个先驱性的工作就是特征金字塔网络FPN，提出了一种自上而下的通道，来将多尺度特征综合到一起。按照这种思想，PANet对FPN之上加入了一个额外的自下而上的通道聚积网络；STDL提出了一个尺度迁移模块，来利用跨尺度特征；M2det提出了一个U形的模块，来融合多尺度特征，G-FRNet提出了门单元，控制特征间的信息流动。最近，NAS-FPN利用神经架构搜索来自动设计特征网络拓扑。虽然其得到了更好的性能，但NAS-FPN在搜索时需要数千GPU小时，得到的特征网络是不规则的，因此很难解释。本文中，我们的目标是，以一种更直观更有规则的方式，优化多尺度特征融合。

**Model Scaling**: In order to obtain better accuracy, it is common to scale up a baseline detector by employing bigger backbone networks (e.g., from mobile-size models [35, 13] and ResNet [12], to ResNeXt [38] and AmoebaNet [29]), or increasing input image size (e.g., from 512x512 [21] to 1536x1536 [42]). Some recent works [8, 42] show that increasing the channel size and repeating feature networks can also lead to higher accuracy. These scaling methods mostly focus on single or limited scaling dimensions. Recently, [36] demonstrates remarkable model efficiency for image classification by jointly scaling up network width, depth, and resolution. Our proposed compound scaling method for object detection is mostly inspired by [36].

**模型缩放**：为得到更高的准确率，采用更大的骨干网络，对基准检测器进行缩放，这是很常见的（如，从移动大小的模型，ResNet，到ResNeXt和AmoebaNet），或增加输入图像大小（如，从512x512到1536x1536）。一些最近的工作表明，增加通道数量，重复特征网络，也可以带来更高的准确率。这些缩放方法主要聚焦在单缩放维度，或有限的缩放维度。最近，[36]证明了，通过对网络宽度、深度和分辨率进行联合缩放，可以在图像分类中得到更高的模型效率。我们提出的目标检测的复合缩放方法，主要是受到[36]的启发。

## 3. BiFPN

In this section, we first formulate the multi-scale feature fusion problem, and then introduce the main ideas for our proposed BiFPN: efficient bidirectional cross-scale connections and weighted feature fusion.

本节中，我们首先描述了多尺度特征融合问题，然后我们介绍了我们提出的BiFPN的主要思想：高效的双向跨尺度链接和加权特征融合。

### 3.1. Problem Formulation

Multi-scale feature fusion aims to aggregate features at different resolutions. Formally, given a list of multi-scale features $\overrightarrow P_{in}=(P_{l_1}^{in}, P_{l_2}^{in}, ...)$ where $P_{l_i}^{in}$ represents the features at level $l_i$, our goal is to find a transformation f that can effectively aggregate different features and output a list of new features: $\overrightarrow P_{out} = f(\overrightarrow P_{in})$. As a concrete example, Figure 2(a) shows the conventional top-down FPN [20]. It takes level 3-7 input features $\overrightarrow P_{in}=(P_{l_3}^{in}, ..., P_{l_7}^{in})$, where $P_i^{in}$ represents a feature level with resolution of $1/2^i$ of the input images. For instance, if input resolution is 640x640, then $P_3^in$ represents feature level 3 (640/2^3 = 80) with resolution 80x80, while $P_7^{in}$ represents feature level 7 with resolution 5x5. The conventional FPN aggregates multi-scale features in a top-down manner:

多尺度特征融合的目标是聚积不同分辨率的特征。正式的，给定多尺度特征列表$\overrightarrow P_{in}=(P_{l_1}^{in}, P_{l_2}^{in}, ...)$，其中$P_{l_i}^{in}$表示在层次$l_i$上的特征，我们的目标是，找到一个变换f，可以有效的聚积不同的特征，输出新的特征列表$\overrightarrow P_{out} = f(\overrightarrow P_{in})$。作为一个具体例子，图2(a)展示的是传统的自上而下的FPN。其以层次3-7的特征为输入$\overrightarrow P_{in}=(P_{l_3}^{in}, ..., P_{l_7}^{in})$，其中$P_i^{in}$表示特征层次的分辨率为输入图像的$1/2^i$。比如，如果输入分辨率为640x640, 那么$P_3^in$表示层次3的特征，分辨率为80x80 (640/2^3 = 80)，而$P_7^{in}$表示层次7的特征，分辨率为5x5。传统的FPN以自上而下的方式聚积多尺度特征：

$$P_7^{out} = conv(P_7^{in}$$
$$P_6^{out} = conv(P_6^{in}+resize(P_7^{out}))$$
...
$$P_3^{out} = conv(P_3^{in}+resize(P_4^{out}))$$

where Resize is usually a upsampling or downsampling op for resolution matching, and Conv is usually a convolutional op for feature processing.

其中resize通常是一个上采样或下采样操作，以进行分辨率匹配，conv通常是一个卷积操作，进行特征处理。

Figure 2: Feature network design – (a) FPN [20] introduces a top-down pathway to fuse multi-scale features from level 3 to 7 ($P_3 - P_7$); (b) PANet [23] adds an additional bottom-up pathway on top of FPN; (c) NAS-FPN [8] use neural architecture search to find an irregular feature network topology and then repeatedly apply the same block; (d) is our BiFPN with better accuracy and efficiency trade-offs.

### 3.2. Cross-Scale Connections 跨尺度的连接

Conventional top-down FPN is inherently limited by the one-way information flow. To address this issue, PANet [23] adds an extra bottom-up path aggregation network, as shown in Figure 2(b). Cross-scale connections are further studied in [17, 15, 39]. Recently, NAS-FPN [8] employs neural architecture search to search for better cross-scale feature network topology, but it requires thousands of GPU hours during search and the found network is irregular and difficult to interpret or modify, as shown in Figure 2(c).

传统的自上而下的FPN，受到其内在的单向信息流限制。为解决这个问题，PANet增加了一个额外的自下而上的通道聚积网络，如图2(b)所示。跨尺度的连接在[17,15,39]中有进一步的研究。最近，NAS-FPN采用神经架构搜索来搜索更好的跨尺度特征网络拓扑，但需要在搜索过程耗费数千GPU小时，找到的网络是不规则的，很难解释或修改，如图2(c)所示。

By studying the performance and efficiency of these three networks (Table 5), we observe that PANet achieves better accuracy than FPN and NAS-FPN, but with the cost of more parameters and computations. To improve model efficiency, this paper proposes several optimizations for cross-scale connections: First, we remove those nodes that only have one input edge. Our intuition is simple: if a node has only one input edge with no feature fusion, then it will have less contribution to feature network that aims at fusing different features. This leads to a simplified bi-directional network; Second, we add an extra edge from the original input to output node if they are at the same level, in order to fuse more features without adding much cost; Third, unlike PANet [23] that only has one top-down and one bottom-up path, we treat each bidirectional (top-down & bottom-up) path as one feature network layer, and repeat the same layer multiple times to enable more high-level feature fusion. Section 4.2 will discuss how to determine the number of layers for different resource constraints using a compound scaling method. With these optimizations, we name the new feature network as bidirectional feature pyramid network (BiFPN), as shown in Figure 2 and 3.

通过研究这三个网络的性能和效率（表5），我们观察到，PANet比FPN和NAS-FPN有更好的效率，但其代价是更多的参数和计算量。为改进模型效率，本文提出了跨尺度连接的几种优化：首先，我们去除了哪些只有一个输入边的节点。我们的直觉是很简单的：如果一个节点只有一个输入边，没有特征融合，那么其对特征网络就只有很少的贡献，而特征网络的目标是融合不同的特征。这带来了一个简化的双向网络；第二，我们从原始的输入到输出节点，如果其在相同的层次，那么就增加了额外的边，以融合更多的特征，而不增加太多的代价；第三，与PANet不同，只使用了一个自上而下和一个自下而上的通道，我们将每个双向（自上而下和自下而上）的通道都作为一个特征网络曾，将相同的层重复多次，以使更高层次的特征融合得到实现。4.2节会讨论，怎样使用一种复合缩放方法，对不同的资源约束，确定层数。使用这种优化，我们将新的特征网络命名为双向特征金字塔网络(BiFPN)，如图2和3所示。

### 3.3. Weighted Feature Fusion 加权特征融合

When fusing features with different resolutions, a common way is to first resize them to the same resolution and then sum them up. Pyramid attention network [19] introduces global self-attention upsampling to recover pixel localization, which is further studied in [8]. All previous methods treat all input features equally without distinction. However, we observe that since different input features are at different resolutions, they usually contribute to the output feature unequally. To address this issue, we propose to add an additional weight for each input, and let the network to learn the importance of each input feature. Based on this idea, we consider three weighted fusion approaches:

当在不同分辨率上融合特征时，一种常用的方法是，将其改变到相同的分辨率大小，然后将其相加。金字塔注意力网络[19]提出了全局自注意力上采样，以恢复像素的位置，这在[8]中进行了进一步的研究。所有之前的方法，都将所有输入特征进行相同的对待，不加区分。但是，我们观察到，由于不同的输入特征是在不同的分辨率上的，他们对输出特征的贡献通常不一样。为解决这个问题，我们提出对每个输入增加一个额外的权重，令网络学习到每个输入特征的重要性。基于这种思想，我们考虑三种加权融合的方法：

**Unbounded fusion**: $O = \sum_i w_i · I_i$, where $w_i$ is a learnable weight that can be a scalar (per-feature), a vector (per-channel), or a multi-dimensional tensor (per-pixel). We find a scalar can achieve comparable accuracy to other approaches with minimal computational costs. However, since the scalar weight is unbounded, it could potentially cause training instability. Therefore, we resort to weight normalization to bound the value range of each weight.

**无界限融合**：$O = \sum_i w_i · I_i$，其中$w_i$是一种可学习的权重，可以是标量（每个特征），一个矢量（每个通道），或一个多维度的张量（逐像素的）。我们发现标量与其他方法相比，可以用最小的计算代价，获得相似的准确率。但是，由于标量权重是没有边界的，这可能会导致训练不稳定。因此，我们寻求权重归一化，来对每个权重的取值范围进行限制。

**softmax-based fusion**: $O = \sum_i \frac {e^{w_i}}{\sum_j e^{w_j}}$. An intuitive idea is to apply softmax to each weight, such that all weights are normalized to be a probability with value range from 0 to 1, representing the importance of each input. However, as shown in our ablation study in section 6.3, the extra softmax leads to significant slowdown on GPU hardware. To minimize the extra latency cost, we further propose a fast fusion approach.

**基于softmax的融合**：$O = \sum_i \frac {e^{w_i}}{\sum_j e^{w_j}}$。一个直观的想法是，对每个权重进行softmax，这样所有的权重都归一化到一个0到1范围内的概率值上，表示每个输入的重要性。但是，如6.3节我们的分离实验所展示，额外的softmax对GPU硬件带来显著的降速。为最小化额外的延迟代价，我们进一步提出一种快速融合方法。

**Fast normalized fusion**: $O = \sum_i \frac {w_i}{ε+\sum_j w_j} · I_i$, where $w_i ≥ 0$ is ensured by applying a Relu after each $w_i$, and ε = 0.0001 is a small value to avoid numerical instability. Similarly, the value of each normalized weight also falls between 0 and 1, but since there is no softmax operation here, it is much more efficient. Our ablation study shows this fast fusion approach has very similar learning behavior and accuracy as the softmax-based fusion, but runs up to 30% faster on GPUs (Table 6).

**快速归一化融合**：$O = \sum_i \frac {w_i}{ε+\sum_j w_j} · I_i$，其中$w_i ≥ 0$通过对每个$w_i$进行Relu运算来确保，ε = 0.0001是一个很小的值，以防止数值不稳定。类似的，每个归一化的权重的值也在0到1之间，但由于这里没有softmax运算，所以会更加高效。我们的分离实验说明，这种快速融合方法，与基于softmax的融合，有很类似的学习行为和准确率，但在GPU上运行速度快了30%（表6）。

Our final BiFPN integrates both the bidirectional cross-scale connections and the fast normalized fusion. As a concrete example, here we describe the two fused features at level 6 for BiFPN shown in Figure 2(d): 我们的BiFPN将双向跨尺度连接与快速归一化融合整合到了一起。一个具体的例子是，这里我们描述了两种融合的特征，如图2(d)中的BiFPN在层次6上：

$$P_6^{td} = conv(\frac {w_1·P_6^{in}+w_2·resize(P_7^{in})}{w_1+w_2+ε})$$
$$P_6^{out} = conv(\frac {w'_1·P_6^{in}+w'_2·P_6^{td}+w'_3·resize(p_5^{out})} {w'_1+w'_2+w'_3+ε})$$

where $P_6^{td}$ is the intermediate feature at level 6 on the top-down pathway, and $P_6^{out}$ is the output feature at level 6 on the bottom-up pathway. All other features are constructed in a similar manner. Notably, to further improve the efficiency, we use depthwise separable convolution [5, 34] for feature fusion, and add batch normalization and activation after each convolution.

其中$P_6^{td}$是在层次6上自上而下通道的的中间特征，而$P_6^{out}$是在层次6上自下而上的通道的输出特征。所有其他特征都是用类似的方式构建起来的。值得注意的是，为进一步改进效率，我们使用逐深度可分离卷积以进行特征融合，并对每个卷积增加批归一化和激活。

## 4. EfficientDet

Based on our BiFPN, we have developed a new family of detection models named EfficientDet. In this section, we will discuss the network architecture and a new compound scaling method for EfficientDet.

基于我们的BiFPN，我们提出了一族新的检测模型，称为EfficientDet。本节中，我们会讨论EfficientDet的网络架构和新的复合缩放方法。

### 4.1. EfficientDet Architecture

Figure 3 shows the overall architecture of EfficientDet, which largely follows the one-stage detectors paradigm [24, 30, 20, 21]. We employ ImageNet-pretrained EfficientNets as the backbone network. Our proposed BiFPN serves as the feature network, which takes level 3-7 features {P3, P4, P5, P6, P7} from the backbone network and repeatedly applies top-down and bottom-up bidirectional feature fusion. These fused features are fed to a class and box network to produce object class and bounding box predictions respectively. Similar to [21], the class and box network weights are shared across all levels of features.

图3给出了EfficientDet的总体架构，主要遵循的是单阶段检测器的范式。我们采用了ImageNet预训练的EfficientNets作为骨干网络。我们提出的BiFPN是特征网络，以骨干网络3-7层的特征{P3, P4, P5, P6, P7}作为输入，反复进行自上而下和自下而上的双向特征融合。这些融合的特征，送入分类网络和框网络，以分别生成目标类别和进行边界框预测。与[21]类似，分类和框网络的权重，在所有特征层次中都是共享的。

Figure 3: EfficientDet architecture – It employs EfficientNet [36] as the backbone network, BiFPN as the feature network, and shared class/box prediction network. Both BiFPN layers and class/box net layers are repeated multiple times based on different resource constraints as shown in Table 1.

### 4.2. Compound Scaling

Aiming at optimizing both accuracy and efficiency, we would like to develop a family of models that can meet a wide spectrum of resource constraints. A key challenge here is how to scale up a baseline EfficientDet model.

我们的目标是对准确率和效率进行同时优化，我们提出一族模型，会满足很宽范围内的资源约束。一个关键的挑战是，对基准EfficientDet模型进行扩大。

Previous works mostly scale up a baseline detector by employing bigger backbone networks (e.g., ResNeXt [38] or AmoebaNet [29]), using larger input images, or stacking more FPN layers [8]. These methods are usually ineffective since they only focus on a single or limited scaling dimensions. Recent work [36] shows remarkable performance on image classification by jointly scaling up all dimensions of network width, depth, and input resolution. Inspired by these works [8, 36], we propose a new compound scaling method for object detection, which uses a simple compound coefficient φ to jointly scale up all dimensions of backbone network, BiFPN network, class/box network, and resolution. Unlike [36], object detectors have much more scaling dimensions than image classification models, so grid search for all dimensions is prohibitive expensive. Therefore, we use a heuristic-based scaling approach, but still follow the main idea of jointly scaling up all dimensions.

之前的工作对一个基准检测器进行放大，主要采用的是更大的骨干网络（如，ResNeXt或AmoebaNet），使用更大的输入图像，或堆叠更多的FPN层。这些方法通常效率不高，因为他们只聚焦在单个或有限的缩放维度上。最近的工作[36]通过网络宽度、深度和输入分辨率的联合缩放，在图像分类中给出了非常好的性能。受这些工作启发[8,36]，我们提出一种目标检测的新复合缩放方法，使用了一种简单的复合系数φ，来对所有维度进行联合缩放，包括骨干网络，BiFPN网络，分类/框网络，和分辨率。与[36]不同的是，目标检测器比分类模型有更多的缩放维度，所以对所有维度的网格搜索，其代价太高，无法进行。因此，我们使用了一种基于启发式的缩放方法，但仍然是遵循的联合缩放所有维度的主要思想。

**Backbone network** – we reuse the same width/depth scaling coefficients of EfficientNet-B0 to B6 [36] such that we can easily reuse their ImageNet-pretrained checkpoints.

**骨干网络** 我们重用EfficientNet-B0到B6的所有宽度/深度缩放系数，这样我们可以很轻松的重用其ImageNet预训练的checkpoints。

**BiFPN network** – we linearly increase BiFPN depth $D_{bifpn}$ (#layers) since depth needs to be rounded to small integers. For BiFPN width $W_{bifpn}$ (#channels), exponentially grow BiFPN width $W_{bifpn}$ (#channels) as similar to [36]. Specifically, we perform a grid search on a list of values {1.2, 1.25, 1.3, 1.35, 1.4, 1.45}, and pick the best value 1.35 as the BiFPN width scaling factor. Formally, BiFPN width and depth are scaled with the following equation:

**BiFPN网络** - 我们线性的增加BiFPN的深度$D_{bifpn}$（#层数），因为深度需要四舍五入到小的整数。对于BiFPN的宽度$W_{bifpn}$（#通道数），按照指数增加BiFPN的宽度$W_{bifpn}$（#通道数），与[36]类似。特别的，我们进行一些值的列表上进行网格搜索{1.2, 1.25, 1.3, 1.35, 1.4, 1.45}，并选择最佳值1.35作为BiFPN的宽度缩放系数。正式的，BiFPN的宽度和深度用下面的公式进行缩放：

$$W_{bifpn} =64·(1.35^φ), D_{bifpn} = 3+φ$$(1)

**Box/class prediction network** – we fix their width to be always the same as BiFPN (i.e., $W_{pred} = W_{bifpn}$), but linearly increase the depth (#layers) using equation:

**框/分类预测网络** - 我们固定其宽度，使其永远与BiFPN相同（即，$W_{pred} = W_{bifpn}$），但线性的增加其深度（#层数），使用下式：

$$D_{box} = D_{class} = 3 + ⌊φ/3⌋$$(2)

**Input image resolution** – Since feature level 3-7 are used in BiFPN, the input resolution must be dividable by 2^7 = 128, so we linearly increase resolutions using equation:

**输入图像分辨率** - 因为特征层次3-7用于BiFPN中，输入图像分辨率必须可以为128整除，所以我们使用下式线性的增加分辨率

$$R_{input} = 512+φ·128$$(3)

Following Equations 1,2,3 with different φ, we have developed EfficientDet-D0 (φ = 0) to D7 (φ = 7) as shown in Table 1, where D7 is the same as D6 except higher resolution. Notably, our scaling is heuristic-based and might not be optimal, but we will show that this simple scaling method can significantly improve efficiency than other single-dimension scaling method in Figure 6.

按照式(1,2,3)，使用不同的φ值，我们提出了EfficientDet-D0 (φ = 0)到D7 (φ = 7)，如表1所示，其中D7与D6一样，除了有更高的分辨率。值得注意的是，我们的缩放是基于启发式的，可能不是最优的，但我们会展示，这种简单的缩放方法，与单维度的缩放方法，可以显著的改进效率，如图6所示。

Table 1: Scaling configs for EfficientDet D0-D6 – φ is the compound coefficient that controls all other scaling dimensions; BiFPN, box/class net, and input size are scaled up using equation 1, 2, 3 respectively.

| | Input size $R_{input}$ | Backbone network | BiFPN #channels $W_{bifpn}$ | BiFPN #layers $D_{bifpn}$ | Box/class #layers $D_{class}$
--- | --- | --- | --- | --- | ---
D0(φ=0) | 512 | B0 | 64 | 3 | 3
D1(φ=1) | 640 | B1 | 88 | 4 | 3
D2(φ=2) | 768 | B2 | 112 | 5 | 3
D3(φ=3) | 896 | B3 | 160 | 6 | 4
D4(φ=4) | 1024 | B4 | 224 | 7 | 4
D5(φ=5) | 1280 | B5 | 288 | 7 | 4
D6(φ=6) | 1280 | B6 | 384 | 8 | 5
D6(φ=7) | 1536 | B6 | 384 | 8 | 5

## 5. Experiments

### 5.1. EfficientDet for Object Detection

We evaluate EfficientDet on COCO 2017 detection datasets [22] with 118K training images. Each model is trained using SGD optimizer with momentum 0.9 and weight decay 4e-5. Learning rate is linearly increased from 0 to 0.16 in the first training epoch and then annealed down using cosine decay rule. Synchronized batch normalization is added after every convolution with batch norm decay 0.99 and epsilon 1e-3. Same as the [36], we use swish activation [28, 6] and exponential moving average with decay 0.9998. We also employ commonly-used focal loss [21] with α = 0.25 and γ = 1.5, and aspect ratio {1/2, 1, 2}. Each model is trained 300 epochs with batch total size 128 on 32 TPUv3 cores. We use RetinaNet [21] preprocessing with training-time multi-resolution cropping/scaling and flipping augmentation. Notably, we do not use auto-augmentation [42] for any of our models.

我们在COCO 2017检测数据集上评估了EfficientDet，使用了118K幅训练图像。每个模型都使用了SGD优化器，动量0.9，权重衰减4e-5。学习速率线性增长，在第一个训练epoch中，从0到0.16，然后使用cosine衰减规则退火下降。同步批归一化在每次卷积后都有，批归一化衰减0.99，epsilon为1e-3。与[36]一样，我们使用swish激活和指数级滑动平均，衰减为0.9998。我们还采用了常用的focal loss，α = 0.25，γ = 1.5，aspect ratio为{1/2, 1, 2}。每个模型都训练了300轮，batch的总大小为128，在32个TPUv3核上进行的训练。我们使用RetinaNet[21]预处理，训练时的多分辨率剪切/缩放和翻转的数据扩充。值得注意的额是，我们没有对任何我们的模型使用自动扩充。

Table 2 compares EfficientDet with other object detectors, under the single-model single-scale settings with no test-time augmentation. We report accuracy for both test-dev (20K test images with no public ground-truth) and val (5K validation images with ground-truth). Our EfficientDet achieves better efficiency than previous detectors, being 4x – 9x smaller and using 13x - 42x less FLOPs across a wide range of accuracy or resource constraints. On relatively low-accuracy regime, our EfficientDet-D0 achieves similar accuracy as YOLOv3 with 28x fewer FLOPs. Compared to RetinaNet [21] and Mask-RCNN [11], our EfficientDet-D1 achieves similar accuracy with up to 8x fewer parameters and 21x fewer FLOPs. On high-accuracy regime, our EfficientDet also consistently outperforms recent NAS-FPN [8] and its enhanced versions in [42] with much fewer parameters and FLOPs. In particular, our EfficientDet-D7 achieves a new state-of-the-art 52.2 AP on test-dev and 51.8 AP on val for single-model single-scale. Notably, unlike the large AmoebaNet + NAS-FPN + AutoAugment models [42] that require special settings (e.g., change anchors from 3x3 to 9x9, train with model parallelism, and rely on expensive auto-augmentation), all EfficientDet models use the same 3x3 anchors and trained without model parallelism or auto-augmentation.

表2比较了EfficientDet与其他目标检测器，在单模型单尺度的设置下，而且没有测试时的数据扩充。我们在test-dev（20K测试图像，没有公开的真值）和val（5K验证图像，有真值）上都给出了准确率。我们的EfficientDet比之前的检测器有着更好的效率，模型小了4x-9x，使用的FLOPs小了13x-42x，准确率或资源限制的范围都很宽。在相对较低准确率的范围内，我们的EfficientDet-D0与YOLOv3取得了类似的准确率，但FLOPs少了28x。与RetinaNet和Mask-RCNN相比，我们的Efficient-D1取得了类似的准确率，但参数少了8x，FLOPs少了21x。在高准确率的范围内，我们的EfficientDet也一直超过了最近的NAS-FPN，及其在[42]中的增强版，参数量和FLOPs也少了好多。特别是，我们的EfficientDet-D7在test-dev和val上分别取得了52.2 AP和51.8 AP的目前最佳成绩，单模型单尺度。值得注意的是，与大型的AmoebaNet + NAS-FPN + AutoAugment模型[42]不同，他们需要特殊的设置（如，改变锚框从3x3到9x9，使用模型并行训练，并依赖很昂贵的自动扩充），所有的EfficientDet模型使用相同的3x3锚框，没有进行模型并行化的训练，也没有auto-augmentation。

Table 2: EfficientDet performance on COCO [22] – Results are for single-model single-scale. test-dev is the COCO test set and val is the validation set. Params and FLOPs denote the number of parameters and multiply-adds. Latency denotes inference latency with batch size 1. AA denotes auto-augmentation [42]. We group models together if they have similar accuracy, and compare their model size, FLOPs, and latency in each group.

In addition to parameter size and FLOPs, we have also compared the real-world latency on Titan-V GPU and single-thread Xeon CPU. We run each model 10 times with batch size 1 and report the mean and standard deviation. Figure 4 illustrates the comparison on model size, GPU latency, and single-thread CPU latency. For fair comparison, these figures only include results that are measured on the same machine with the same settings. Compared to previous detectors, EfficientDet models are up to 4.1x faster on GPU and 10.8x faster on CPU, suggesting they are also efficient on real-world hardware.

除了参数大小和FLOPs，我们比较了在Titan-V GPU和单线程Xeon CPU上的真实世界延迟。我们运行每个模型10次，批大小1，并给出平均和标准偏差。图4描述了对模型大小、GPU延迟和单线程CPU延迟的比较。为公平比较，这些图只包含了在同样的机器上用同样的设置中也能测量到的结果。与之前的检测器相比，EfficientDet模型在GPU上快了4.1倍，在CPU上快了10.8倍，说明他们在真实世界的硬件中也同样高效。

Figure 4: Model size and inference latency comparison – Latency is measured with batch size 1 on the same machine equipped with a Titan V GPU and Xeon CPU. AN denotes AmoebaNet + NAS-FPN trained with auto-augmentation [42]. Our EfficientDet models are 4x - 9x smaller, 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors.

### 5.2. EfficientDet for Semantic Segmentation

While our EfficientDet models are mainly designed for object detection, we are also interested in their performance on other tasks such as semantic segmentation. Following [16], we modify our EfficientDet model to keep feature level {P2,P3,...,P7} in BiFPN, but only use P2 for the final per-pixel classification. For simplicity, here we only evaluate a EfficientDet-D4 based model, which uses a ImageNet pretrained EfficientNet-B4 backbone (similar size to ResNet-50). We set the channel size to 128 for BiFPN and 256 for classification head. Both BiFPN and classification head are repeated by 3 times.

我们的EfficientDet模型主要设计用于目标检测，我们同时也对其在其他任务上的性能感兴趣，比如语义分割。根据[16]，我们更改EfficientDet模型，维持BiFPN中的特征层次{P2,P3,...,P7}，但只使用P2进行最后的逐像素的分类。为简化起见，这里我们只评估一个基于EfficientDet-D4的模型，使用了一个ImageNet预训练的EfficientNet-B4的骨干网络（与ResNet-50大小相似）。我们设置到BiFPN通道大小为128，对于分类头的有256。BiFPN和分类头都重复了3次。

Table 3 shows the comparison between our models and previous DeepLabV3+ [4] on Pascal VOC 2012 [7]. Notably, we exclude those results with ensemble, test-time augmentation, or COCO pretraining. Under the same single-model single-scale settings, our model achieves 1.7% better accuracy with 9.8x fewer FLOPs than the prior art of DeepLabV3+ [4]. These results suggest that EfficientDet is also quite promising for semantic segmentation.

表3给出了我们模型和之前的DeepLabV3+在Pascal VOC 2012上的比较。值得注意的是，我们排除了这些结果中使用集成学习，测试时的数据扩充，或COCO预训练。在相同的单模型单尺度设置下，我们的模型比之前的DeepLabV3+使用的FLOPs少了9.8x，准确率提高了1.7%。这些结果说明，EfficientDet对于语义分割也非常有希望。

Table 3: Performance comparison on Pascal VOC semantic segmentation.

Model | mIOU | Params | FLOPs
--- | --- | --- | ---
DeepLabV3+ (ResNet-101) [4] | 79.35% | - | 298B
DeepLabV3+ (Xception) [4] | 80.02% | - | 177B
Our EfficientDet† | 81.74% | 17M | 18B

## 6. Ablation Study

In this section, we ablate various design choices for our proposed EfficientDet. For simplicity, all accuracy results here are for COCO validation set. 本节中，我们对提出的EfficientDet的各种设计选择进行了分离实验。简化起见，所有的准确率结果这里都是在COCO验证集上的。

### 6.1. Disentangling Backbone and BiFPN 将骨干网络与BiFPN分离

Since EfficientDet uses both a powerful backbone and a new BiFPN, we want to understand how much each of them contributes to the accuracy and efficiency improvements. Table 4 compares the impact of backbone and BiFPN. Starting from a RetinaNet detector [21] with ResNet-50 [12] backbone and top-down FPN [20], we first replace the backbone with EfficientNet-B3, which improves accuracy by about 3 AP with slightly less parameters and FLOPs. By further replacing FPN with our proposed BiFPN, we achieve additional 4 AP gain with much fewer parameters and FLOPs. These results suggest that EfficientNet backbones and BiFPN are both crucial for our final models.

因为EfficientDet使用了一个很强的骨干网络，和一个新的BiFPN，我们希望理解它们各自都对准确率和效率的改进有多少贡献。表4比较了骨干网络和BiFPN的影响。我们从RetinaNet检测器开始，其骨干网络为ResNet-50，使用了自上而下的FPN，我们首先替换骨干网络为EfficientNet-B3，这提升了准确率大约3 AP，参数和FLOPs略少。进一步将FPN替换为我们提出的BiFPN，我们进一步提升了4 AP，参数和FLOPs都更少一些。这些结果说明，EfficientNet骨干和BiFPN对于我们最后的模型都非常关键。

Table 4: Disentangling backbone and BiFPN – Starting from the standard RetinaNet (ResNet50+FPN), we first replace the backbone with EfficientNet-B3, and then replace the baseline FPN with our proposed BiFPN.

| | AP | Parameters | FLOPs
--- | --- | --- | ---
ResNet50 + FPN | 37.0 | 34M | 97B
EfficientNet-B3 + FPN | 40.3 | 21M | 75B
EfficientNet-B3 + BiFPN | 44.4 | 12M | 24B

### 6.2. BiFPN Cross-Scale Connections

Table 5 shows the accuracy and model complexity for feature networks with different cross-scale connections listed in Figure 2. Notably, the original FPN [20] and PANet [23] only have one top-down or bottom-up flow, but for fair comparison, here we repeat each of them multiple times and replace all convs with depthwise separable convs, which is the same as BiFPN. We use the same backbone and class/box prediction network, and the same training settings for all experiments. As we can see, the conventional top-down FPN is inherently limited by the one-way information flow and thus has the lowest accuracy. While repeated FPN+PANet achieves slightly better accuracy than NAS-FPN [8], it also requires more parameters and FLOPs. Our BiFPN achieves similar accuracy as repeated FPN+PANet, but uses much less parameters and FLOPs. With the additional weighted feature fusion, our BiFPN further achieves the best accuracy with fewer parameters and FLOPs.

表5展示了，在图2中所列车的不同的跨尺度连接下，特征网络的准确率和模型复杂度。值得注意的是，原始的FPN和PANet只有一个自上而下和自下而上的流向，但为公平比较，这里我们对每个都重复了多次，并将所有的convs替换成逐层可分离convs，这就与BiFPN一样了。我们对所有实验都使用相同的骨干网络和分类/框预测网络，相同的训练设置。我们可以看到，传统的自上而下的FPN受限于其内在的单向信息流，因此准确率最低。而重复的FPN+PANet比NAS-FPN取得了略好的准确率，但仍然需要更多的参数和FLOPs。我们的BiFPN与重复的FPN+PANet获得了类似的准确率，但使用了少的多的参数和FLOPs。在额外的加权特征融合下，我们的BiFPN使用了更少的参数和FLOPs，获得了最高的准确率。

Table 5: Comparison of different feature networks – Our weighted BiFPN achieves the best accuracy with fewer parameters and FLOPs.

| | AP | #Params ratio | #FLOPs ratio
--- | --- | --- | ---
Repeated top-down FPN | 42.29 | 1.0x | 1.0x
Repeated FPN+PANet | 44.08 | 1.0x | 1.0x
NAS-FPN | 43.16 | 0.71x | 0.72x
Fully-Connected FPN | 43.06 | 1.24x | 1.21x
BiFPN (w/o weighted) | 43.94 | 0.88x | 0.67x
BiFPN (w/ weighted) | 44.39 | 0.88x | 0.68x

### 6.3. Softmax vs Fast Normalized Fusion

As discussed in Section 3.3, we propose a fast normalized feature fusion approach to get rid of the expensive softmax while retaining the benefits of normalized weights. Table 6 compares the softmax and fast normalized fusion approaches in three detectors with different model sizes. As shown in the results, our fast normalized fusion approach achieves similar accuracy as the softmax-based fusion, but runs 1.26x - 1.31x faster on GPUs.

如3.3节所述，我们提出了一种快速归一化的特征融合方法，以替换掉昂贵的softmax运算，同时保持归一化加权的好处。表6比较了softmax和快速归一化的融合方法在三个检测器中的表现，分别采用了不同的模型大小。如结果所示，我们的快速归一化的融合方法，取得了与基于softmax融合的类似准确率，但运行速度比在GPUs上更快1.26x-1.31x。

Table 6: Comparison of different feature fusion – Our fast fusion achieves similar accuracy as softmax-based fusion, but runs 28% - 31% faster.

Model | Softmax Fusion AP | Fast Fusion AP (delta) | Speedup
--- | --- | --- | ---
Model1 | 33.96 | 33.85(-0.11) | 1.28x
Model2 | 43.78 | 43.77(-0.11) | 1.26x
Model3 | 48.79 | 48.74(-0.05) | 1.31x

In order to further understand the behavior of softmax-based and fast normalized fusion, Figure 5 illustrates the learned weights for three feature fusion nodes randomly selected from the BiFPN layers in EfficientDet-D3. Notably, the normalized weights (e.g., $e^{w_i} / \sum_j e^{w_j}$ for softmax-based fusion, and $w_i/(ε + \sum_j w_j)$ for fast normalized fusion) always sum up to 1 for all inputs. Interestingly, the normalized weights change rapidly during training, suggesting different features contribute to the feature fusion unequally. Despite the rapid change, our fast normalized fusion approach always shows very similar learning behavior to the softmax-based fusion for all three nodes.

为进一步理解基于softmax的和快速归一化融合的行为，图5给出了EfficientDet-D3中的BiFPN层中随机选取的三个特征融合节点学习到的权重。值得注意的是，归一化的权重（如，基于softmax融合的$e^{w_i} / \sum_j e^{w_j}$，和快速归一化融合的$w_i/(ε + \sum_j w_j)$）对于输入其和永远是1。有趣的是，归一化的权重在训练时变化迅速，说明不同的特征对特征融合的贡献并不一样。虽然变化迅速，我们的快速归一化融合的方法，与基于softmax的融合学习行为相比，在三个节点上一直都表现出了非常类似的学习行为。

Figure 5: Softmax vs. fast normalized feature fusion – (a) - (c) shows normalized weights (i.e., importance) during training for three representative nodes; each node has two inputs (input1 & input2) and their normalized weights always sum up to 1.

### 6.4. Compound Scaling 复合缩放

As discussed in section 4.2, we employ a compound scaling method to jointly scale up all dimensions of depth/width/resolution for backbone, BiFPN, and box/class prediction networks. Figure 6 compares our compound scaling with other alternative methods that scale up a single dimension of resolution/depth/width. Although starting from the same baseline detector, our compound scaling method achieves better efficiency than other methods, suggesting the benefits of jointly scaling by better balancing difference architecture dimensions.

如4.2节讨论，我们采用了一种复合缩放的方法，来对骨干网络，BiFPN，框/分类的预测网络的深度/宽度/分辨率的所有维度进行缩放。图6比较了我们的复合缩放与其他的替换方法，对单维度的分辨率/深度/宽度进行缩放。虽然是从相同的基准检测器开始的，但我们的缩放方法比其他的方法取得了更好的效率，说明联合缩放可以更好的平衡不同的架构维度。

Figure 6: Comparison of different scaling methods – compound scaling achieves better accuracy and efficiency.

## 7. Conclusion

In this paper, we systematically study network architecture design choices for efficient object detection, and propose a weighted bidirectional feature network and a customized compound scaling method, in order to improve accuracy and efficiency. Based on these optimizations, we develop a new family of detectors, named EfficientDet, which consistently achieve better accuracy and efficiency than the prior art across a wide spectrum of resource constraints. In particular, our scaled EfficientDet achieves state-of-the-art accuracy with much fewer parameters and FLOPs than previous object detection and semantic segmentation models.

本文中，我们系统的研究了，高效目标检测中网络架构的设计选择，提出了一种加权双向特征网络，和一种定制的复合缩放方法，以提升准确率和效率。基于这些优化，我们提出了一族新检测器，名为EfficientDet，与之前的最好的方法相比，在很大范围的资源约束前，取得了更好的准确率和效率。特别是，与之前的目标检测方法和语义分割模型相比，我们的缩放的EfficientDet取得了目前最好的准确率，但使用了少的多的参数和FLOPs。