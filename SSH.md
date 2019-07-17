# SSH: Single Stage Headless Face Detector

Mahyar Najibi et al. University of Maryland

## Abstract 

We introduce the Single Stage Headless (SSH) face detector. Unlike two stage proposal-classification detectors, SSH detects faces in a single stage directly from the early convolutional layers in a classification network. SSH is headless. That is, it is able to achieve state-of-the-art results while removing the “head” of its underlying classification network – i.e. all fully connected layers in the VGG-16 which contains a large number of parameters. Additionally, instead of relying on an image pyramid to detect faces with various scales, SSH is scale-invariant by design. We simultaneously detect faces with different scales in a single forward pass of the network, but from different layers. These properties make SSH fast and light-weight. Surprisingly, with a headless VGG-16, SSH beats the ResNet-101-based state-of-the-art on the WIDER dataset. Even though, unlike the current state-of-the-art, SSH does not use an image pyramid and is 5X faster. Moreover, if an image pyramid is deployed, our light-weight network achieves state-of-the-art on all subsets of the WIDER dataset, improving the AP by 2.5%. SSH also reaches state-of-the-art results on the FDDB and Pascal-Faces datasets while using a small input size, leading to a runtime of 50 ms/image on a GPU. The code is available at https://github.com/mahyarnajibi/SSH.

我们提出了单阶段无头人脸检测器(SSH)。与两阶段的建议-分类检测器不同，SSH检测人脸是一阶段的，直接从分类网络前期的卷积层从检测。SSH是无头的，即，取得目前最好的效果，而且去掉了分类网络的头，即VGG-16中所有的全连接层，这些层中包含了大量参数。另外，SSH没有依赖图像金字塔来检测不同尺度下的人脸，SSH在设计上就是尺度不变的。我们在网络的一次前向过程中，在不同的层中，同时检测不同尺度下的人脸。这些性质使SSH快速而且轻量级。令人惊讶的是，使用无头的VGG-16，SSH在WIDER数据集击败了目前最好的基于ResNet-101的方法。即使这样，与目前最好的方法不同，SSH没有使用图像金字塔，速度快了5倍。而且，如果使用了图像金字塔，我们的轻量级网络在WIDER的所有子集上取得了目前最好的结果，AP改进达2.5%。SSH在FDDB和PASCAL-Faces数据集上也取得了目前最好的效果，使用的输入很小，在GPU上运行速度为50ms每图像。代码已开源。

## 1. Introduction

Face detection is a crucial step in various problems involving verification, identification, expression analysis, etc. From the Viola-Jones [29] detector to recent work by Hu et al. [7], the performance of face detectors has been improved dramatically. However, detecting small faces is still considered a challenging task. The recent introduction of the WIDER face dataset [35], containing a large number of small faces, exposed the performance gap between humans and current face detectors. The problem becomes more challenging when the speed and memory efficiency of the detectors are taken into account. The best performing face detectors are usually slow and have high memory foot-prints (e.g. [7] takes more than 1 second to process an image, see Section 4.5) partly due to the huge number of parameters as well as the way robustness to scale or incorporation of context are addressed.

人脸检测是很多问题的关键步骤，包括验证、鉴定、表情分析等。从Viola-Jones检测器，到最近Hu等[7]的工作，人脸检测器的性能已经得到了极大的改进。但是，检测小型人脸仍然是一个很有挑战的工作。最近提出的WIDER Face数据集[35]，包含了大量小的人脸，暴露了目前的人脸检测器与人的性能差距。当考虑到检测器的速度和内存性能时，这个问题就变得更加严峻了。最好的人脸检测器通常很慢，而且内存占用很多（如[7]处理一幅图像耗时超过1秒，见4.5节），部分因为参数量大，以及对尺度的稳健性或包含了上下文信息。

State-of-the-art CNN-based detectors convert image classification networks into two-stage detection systems [4, 24]. In the first stage, early convolutional feature maps are used to propose a set of candidate object boxes. In the second stage, the remaining layers of the classification networks (e.g. fc6~8 in VGG-16 [26]), which we refer to as the network “head”, are deployed to extract local features for these candidates and classify them. The head in the classification networks can be computationally expensive (e.g. the network head contains ∼ 120M parameters in VGG-16 and ∼ 12M parameters in ResNet-101). Moreover, in the two stage detectors, the computation must be performed for all proposed candidate boxes.

目前最好的基础CNN的检测器是将图像分类网络转化成两阶段检测系统。在第一阶段，早期的卷积特征图用于提出候选目标框的集合；在第二阶段，分类网络剩下的层（如，VGG-16中的fc6-8），我们称之为网络“头”，要提取这些候选的局部特征并对其分类。分类网络中的头计算量非常大（如，VGG-16的网络头包含大约120M参数，ResNet-101包含大约12M参数）。而且，在两阶段检测器中，这个计算要在所有提出的候选框中进行。

Very recently, Hu et al. [7] showed state-of-the-art results on the WIDER face detection benchmark by using a similar approach to the Region Proposal Networks (RPN) [24] to directly detect faces. Robustness to input scale is achieved by introducing an image pyramid as an integral part of the method. However, it involves processing an input pyramid with an up-sampling scale up to 5000 pixels per side and passing each level to a very deep network which increased inference time.

最近，Hu等[7]在WIDER Face检测基准测试中得到了目前最好的效果，使用的是与RPN类似的方法直接检测人脸。尺度稳健性是通过将图像金字塔作为这个方法的一部分而得到的。但是，这涉及到了处理一个输入金字塔，上采样尺度达到了5000像素每边，将每一级都送入非常深的网络，这增加了推理时间。

In this paper, we introduce the Single Stage Headless (SSH) face detector. SSH performs detection in a single stage. Like RPN [24], the early feature maps in a classification network are used to regress a set of predefined anchors towards faces. However, unlike two-stage detectors, the final classification takes place together with regressing the anchors. SSH is headless. It is able to achieve state-of-the-art results while removing the head of its underlying network (i.e. all fully connected layers in VGG-16), leading to a light-weight detector. Finally, SSH is scale-invariant by design. Instead of relying on an external multi-scale pyramid as input, inspired by [14], SSH detects faces from various depths of the underlying network. This is achieved by placing an efficient convolutional detection module on top of the layers with different strides, each of which is trained for an appropriate range of face scales. Surprisingly, SSH based on a headless VGG-16, not only outperforms the best-reported VGG-16 by a large margin but also beats the current ResNet-101-based state-of-the-art method on the WIDER face detection dataset. Unlike the current state-of-the-art, SSH does not deploy an input pyramid and is 5 times faster. If an input pyramid is used with SSH as well, our light-weight VGG-16-based detector outperforms the best reported ResNet-101 [7] on all three subsets of the WIDER dataset and improves the mean average precision by 4% and 2.5% on the validation and the test set respectively. SSH also achieves state-of-the-art results on the FDDB and Pascal-Faces datasets with a relatively small input size, leading to a runtime of 50 ms/image.

本文中，我们提出了SSH人脸检测器。SSH在一阶段中进行检测。与RPN类似，分类网络中的早期特征图，用于对预定义的锚框进行回归到人脸。但是，与两阶段检测器不同，最终的分类与回归这些锚框同时发生。SSH是无头的。在去掉网络头（如，VGG-16中的所有全连接层）时还能取得目前最好的结果，得到一个轻量级检测器。最后，SSH在设计上就是尺度不变的。SSH没有依赖外部的多尺度金字塔作为输入，而是受[14]启发，从网络的不同深度中检测人脸。在不同步长的层之上，放置高效的卷积检测模块，每个都训练用来检测一定尺度范围的人脸，就可以得到这样的效果。令人惊奇的是，基于无头的VGG-16的SSH，不仅大大超过了基于VGG-16的最好模型，还击败了在WIDER Face上基于ResNet-101的目前最好的模型。与目前好的模型不同，SSH没有使用输入金字塔，所以快了5倍。如果使用了图像金字塔作为SSH的输入，我们的轻量级基于VGG-16的检测器，在WIDER数据集的所有三个子集上，都超过了最好的基于ResNet-101的模型，对验证集和测试集上的mAP分别改进了4%和2.5%。SSH在FDDB和PASCAL Faces数据集上也取得了目前最好的结果，输入大小相对较小，运行时间50ms每图像。

The rest of the paper is organized as follows. Section 2 provides an overview of the related works. Section 3 introduces the proposed method. Section 4 presents the experiments and Section 5 concludes the paper.

本文组织如下。第2部分概览了相关工作。第3部分介绍了提出的方法。第4部分给出了试验，第5部分总结了本文。

Figure 1: SSH detects various face sizes in a single CNN forward pass and without employing an image pyramid in ∼ 0.1 second for an image with size 800 × 1200 on a GPU.

## 2. Related Works 相关工作

### 2.1. Face Detection 人脸检测

Prior to the re-emergence of convolutional neural networks (CNN), different machine learning algorithms were developed to improve face detection performance [29, 39, 10, 11, 18, 2, 31]. However, following the success of these networks in classification tasks [9], they were applied to detection as well [6]. Face detectors based on CNNs significantly closed the performance gap between human and artificial detectors [12, 33, 32, 38, 7]. However, the introduction of the challenging WIDER dataset [35], containing a large number of small faces, re-highlighted this gap. To improve performance, CMS-RCNN [38] changed the Faster R-CNN object detector [24] to incorporate context information. Very recently, Hu et al. proposed a face detection method based on proposal networks which achieves state-of-the-art results on this dataset [7]. However, in addition to skip connections, an input pyramid is processed by rescaling the image to different sizes, leading to slow detection speeds. In contrast, SSH is able to process multiple face scales simultaneously in a single forward pass of the network, which reduces inference time noticeably.

在CNN的出现之前，提出了不同的机器学习算法用于改进人脸检测性能。但是，在这些网络在分类任务中取得成功后，它们也用于检测任务了。基于CNNs的人脸检测器明显将人工检测器与人之间的差距缩小了。但是，WIDER数据的提出非常有挑战性，包含了大量很小的人脸，再次强调了这个差距。为改进性能，CMS-RCNN改变了Faster R-CNN目标检测器，加入了上下文信息。最近，Hu等基于建议网络提出了一种人脸检测方法，在这个数据集上得到了目前最好的效果。但是，除了使用跳跃连接，还使用了输入图像金字塔，这导致检测速度很慢。比较之下，SSH可以在网络一个前向过程中同时处理多个人脸尺度，极大降低了推理时间。

### 2.2. Single Stage Detectors and Proposal Networks

The idea of detecting and localizing objects in a single stage has been previously studied for general object detection. SSD [16] and YOLO [23] perform detection and classification simultaneously by classifying a fixed grid of boxes and regressing them towards objects. G-CNN [19] models detection as a piece-wise regression problem and iteratively pushes an initial multi-scale grid of boxes towards objects while classifying them. However, current state-of-the-art methods on the challenging MS-COCO object detection benchmark are based on two-stage detectors[15]. SSH is a single stage detector; it detects faces directly from the early convolutional layers without requiring a proposal stage.

一阶段检测并定位目标的思想，之前曾在通用目标检测中得到研究。SSD和YOLO同时进行检测和分类，主要是通过对固定网络的边界框进行分类，并将其向目标回归。G-CNN将检测作为一个分片回归问题，将初始的多尺度网格边界框迭代推向目标，同时进行分类。但是，在MS-COCO目标检测基准测试中，目前最好的方法是基于两阶段的检测器。SSH是一阶段检测器；它直接从前期卷积层中检测人脸，不需要建议阶段。

Although SSH is a detector, it is more similar to the object proposal algorithms which are used as the first stage in detection pipelines. These algorithms generally regress a fixed set of anchors towards objects and assign an objectness score to each of them. MultiBox [28] deploys clustering to define anchors. RPN [24], on the other hand, defines anchors as a dense grid of boxes with various scales and aspect ratios, centered at every location in the input feature map. SSH uses similar strategies, but to localize and at the same time detect, faces.

虽然SSH是一个检测器，它与目标建议算法更相似，即用在检测流程中的第一阶段的算法。这些算法一般对固定锚框集进行回归，得到目标，并对每个目标指定一个objectness分数。MultiBox用聚类来定义锚框。RPN则将锚框定义为密集边界框网格，包含各种尺度和纵横比，中心位于输入特征图的每个位置。SSH采用类似的策略，但定位和检测人脸同时进行。

### 2.3. Scale Invariance and Context Modeling

Being scale invariant is important for detecting faces in unconstrained settings. For generic object detection, [1, 36] deploy feature maps of earlier convolutional layers to detect small objects. Recently, [14] used skip connections in the same way as [17] and employed multiple shared RPN and classifier heads from different convolutional layers. For face detection, CMS-RCNN [38] used the same idea as [1, 36] and added skip connections to the Faster RCNN [24]. [7] creates a pyramid of images and processes each separately to detect faces of different sizes. In contrast, SSH is capable of detecting faces at different scales in a single forward pass of the network without creating an image pyramid. We employ skip connections in a similar fashion as [17, 14], and train three detection modules jointly from the convolutional layers with different strides to detect small, medium, and large faces.

在不受限的设置中检测人脸，尺度不变性是非常重要的。对于通用目标检测，[1,36]将前期卷积层的特征图用于检测小目标。最近，[14]采用[17]的方式，使用跳跃连接，在不同卷积层中采用多个共享的RPN和分类头。对于人脸检测，CMS-RCNN使用了与[1,36]相同的思想，对Faster RCNN增加了跳跃连接。[7]生成了图像金字塔，对每一个单独进行处理，以检测不同大小的人脸。对比起来，SSH可以在网络的一个前向过程中检测不同尺度的人脸，不用生成图像金字塔。我们与[17,14]类似，也采用了跳跃连接，训练了卷积层中的三个检测模块，各有不同的步长，以检测小型、中型和大型人脸。

In two stage object detectors, context is usually modeled by enlarging the window around proposals [36]. [1] models context by deploying a recurrent neural network. For face detection, CMS-RCNN [38] utilizes a larger window with the cost of duplicating the classification head. This increases the memory requirement as well as detection time. SSH uses simple convolutional layers to achieve the same larger window effect, leading to more efficient context modeling.

在两阶段目标检测器中，通常通过增大建议附近的窗口来对上下文建模。[1]通过一个RNN来对上下文建模。对于人脸检测，CMS-RCNN使用更大的窗口来对上下文建模，代价是多了一个分类头。这增加了内存需求和检测时间。SSH使用简单的卷积层，以得到同样的更大窗口效果，得到了更高效的上下文建模。

## 3. Proposed Method

SSH is designed to decrease inference time, have a low memory foot-print, and be scale-invariant. SSH is a single-stage detector; i.e. instead of dividing the detection task into bounding box proposal and classification, it performs classification together with localization from the global information extracted from the convolutional layers. We empirically show that in this way, SSH can remove the “head” of its underlying network while achieving state-of-the-art face detection accuracy. Moreover, SSH is scale-invariant by design and can incorporate context efficiently.

SSH在设计上就降低了推理时间，内存占用少，具有尺度不变性。SSH是一阶段检测器；即，没有将检测任务分为边界框建议和分类，而是将分类与定位一起进行，利用卷积层提取出的全局信息。我们通过经验展示了，SSH可以去掉网络的头，同时得到目前最好的人脸检测准确率。而且，SSH在设计上就是对尺度不变的，还可以高效的包含上下文信息。

### 3.1. General Architecture

Figure 2 shows the general architecture of SSH. It is a fully convolutional network which localizes and classifies faces early on by adding a detection module on top of feature maps with strides of 8, 16, and 32, depicted as $M_1$, $M_2$, and $M_3$ respectively. The detection module consists of a convolutional binary classifier and a regressor for detecting faces and localizing them respectively.

图2给出了SSH的一般架构，这是一个全卷积网络，在步长为8、16、32的特征图上增加了检测模块，进行同时人脸定位和分类，这三个特征图分别表示为$M_1$, $M_2$和$M_3$。检测模块包含一个卷积二值分类器和一个回归器，分别进行人脸检测和定位。

To solve the localization sub-problem, as in [28, 24, 19], SSH regresses a set of predefined bounding boxes called anchors, to the ground-truth faces. We employ a similar strategy to the RPN [24] to form the anchor set. We define the anchors in a dense overlapping sliding window fashion. At each sliding window location, K anchors are defined which have the same center as that window and different scales. However, unlike RPN, we only consider anchors with aspect ratio of one to reduce the number of anchor boxes. We noticed in our experiments that having various aspect ratios does not have a noticeable impact on face detection precision. More formally, if the feature map connected to the detection module $M_i$ has a size of $W_i × H_i$, there would be $W_i × H_i × K_i$ anchors with aspect ratio one and scales {$S_i^1, S_i^2, . . . S_i^{K_i}$}.

为解决定位的子问题，就像在[28,24,19]中一样，SSH对预定义的边界框集，称为锚框集，回归到真值人脸上。我们采用与RPN类似的策略形成锚框集。我们以密集重叠的滑窗方式定义锚框。在每个滑窗位置都定义了K个锚框，中心位置相同，但尺度不一样。但是，与RPN不同的是，我们只考虑一种纵横比的锚框，以降低锚框数量。我们在试验中注意到，多个纵横比的锚框对人脸检测准确度的影响可以忽略不计。正式的，如果与特征图相连的检测模块$M_i$大小为$W_i × H_i$，那么就有$W_i × H_i × K_i$个锚框，纵横比为1，尺度为{$S_i^1, S_i^2, . . . S_i^{K_i}$}。

For the detection module, a set of convolutional layers are deployed to extract features for face detection and localization as depicted in Figure 3. This includes a simple context module to increase the effective receptive field as discussed in section 3.3. The number of output channels of the context module, (i.e. “X” in Figures 3 and 4) is set to 128 for detection module $M_1$ and 256 for modules $M_2$ and $M_3$. Finally, two convolutional layers perform bounding box regression and classification. At each convolution location in $M_i$, the classifier decides whether the windows at the filter’s center and corresponding to each of the scales $\{ S_i^k \}^K_{k=1}$ contains a face. A 1 × 1 convolutional layer with 2 × K output channels is used as the classifier. For the regressor branch, another 1×1 convolutional layer with 4×K output channels is deployed. At each location during the convolution, the regressor predicts the required change in scale and translation to match each of the positive anchors to faces.

对于检测模块，采用了卷积层的集合来提取特征，进行人脸检测和定位，如图3所示。这包括了一个简单的上下文模块，以增大有效感受野，如3.3节所述。上下文模块的输出通道数量（即图3和4中的X），对于检测模块$M_1$为128，对于$M_2$和$M_3$为256。最后，两个卷积层进行边界框回归和分类。在$M_i$的每个卷积位置，由分类器决定，位于滤波器中心的窗口，和对应的每个尺度上$\{ S_i^k \}^K_{k=1}$，是否包含人脸。使用了一个2×K输出通道的1×1卷积层来作为分类器。对于回归分支，另一个4×K输出通道的1×1卷积层用于精确定位。在卷积过程中的每个位置上，回归器预测在尺度和平移上需要的变化，以将每个正锚框与人脸相匹配。

### 3.2. Scale-Invariance Design 尺度不变的设计

In unconstrained settings, faces in images have varying scales. Although forming a multi-scale input pyramid and performing several forward passes during inference, as in [7], makes it possible to detect faces with different scales, it is slow. In contrast, SSH detects large and small faces simultaneously in a single forward pass of the network. Inspired by [14], we detect faces from three different convolutional layers of our network using detection modules M1, M2, and M3. These modules have strides of 8, 16, and 32 and are designed to detect small, medium, and large faces respectively.

在不受限的设置中，图像中的人脸有不同的尺度。在[7]中，将输入形成一个图像金字塔，在推理过程中进行多次前向过程，使其可以检测不同尺度的人脸，但速度非常慢。比较之下，SSH在网络的一次前向过程中，同时检测大的人脸和小的人脸。受[14]启发，我们从网路中三个不同的卷积层上检测人脸，使用了检测模块M1, M2和M3。这些模块的步长为8,16和32，在设计上分别用于检测小型、中型和大型人脸。

More precisely, the detection module M2 performs detection from the conv5-3 layer in VGG-16. Although it is possible to place the detection module M1 directly on top of conv4-3, we use the feature map fusion which was previously deployed for semantic segmentation [17], and generic object detection [14]. However, to decrease the memory consumption of the model, the number of channels in the feature map is reduced from 512 to 128 using 1 × 1 convolutions. The conv5-3 feature maps are up-sampled and summed up with the conv4-3 features, followed by a 3 × 3 convolutional layer. We used bilinear up-sampling in the fusion process. For detecting larger faces, a max-pooling layer with stride of 2 is added on top of the conv5-3 layer to increase its stride to 32. The detection module M3 is placed on top of this newly added layer.

确切的说，检测模块M2在VGG-16的conv5-3上进行检测。虽然可以将检测模块M1直接放于conv4-3之上，但我们使用的是特征图融合，之前用于语义分割[17]和通用目标检测[14]。但是，为降低模型的内存消耗，特征图的通道数量需要从512降到128，这是使用1×1卷积实现的。conv5-3特征图进行了上采样，并与conv4-3特征图求和，然后进行3×3卷积。我们在融合过程中使用双线性插值的上采样。为检测更大的人脸，对conv5-3进行了步长为2的最大池化，以将其尺度增加到32。检测模块M3置于新加入的层之上。

During the training phase, each detection module Mi is trained to detect faces from a target scale range as discussed in 3.4. During inference, the predicted boxes from the different scales are joined together followed by Non-Maximum Suppression (NMS) to form the final detections.

在训练阶段，每个检测模块Mi都训练用于检测一定尺度范围的人脸，如3.4节所述。在推理过程中，不同尺度上预测的人脸框联合到一起，然后进行NMS，形成最后的检测结果。

### 3.3. Context Module 上下文模块

In two-stage detectors, it is common to incorporate context by enlarging the window around the candidate proposals. SSH mimics this strategy by means of simple convolutional layers. Figure 4 shows the context layers which are integrated into the detection modules. Since anchors are classified and regressed in a convolutional manner, applying a larger filter resembles increasing the window size around proposals in a two-stage detector. To this end, we use 5 × 5 and 7 × 7 filters in our context module. Modeling the context in this way increases the receptive field proportional to the stride of the corresponding layer and as a result the target scale of each detection module. To reduce the number of parameters, we use a similar approach as [27] and deploy sequential 3×3 filters instead of larger convolutional filters. The number of output channels of the detection module (i.e. “X” in Figure 4) is set to 128 for M1 and 256 for modules M2 and M3. It should be noted that our detection module together with its context filters uses fewer of parameters compared to the module deployed for proposal generation in [24]. Although, more efficient, we empirically found that the context module improves the mean average precision on the WIDER validation dataset by more than half a percent.

在两阶段检测器中，通常通过增加候选建议附近的框，来将上下文信息纳入进来。SSH以简单的卷积层模仿这个策略。图4给出了与检测模块整合到一起的上下文层。由于锚框是以卷积的形式进行分类和回归，使用一个更大的滤波器，类似于在两阶段检测器中增加建议区域附近的窗口大小。为此，我们在上下文模块中使用5×5和7×7滤波器。这种方式的上下文建模增大了感受野，与对应层的步长成比例，也就是与每个检测模块的目标尺度成比例。为降低参数数量，我们使用与[27]类似的方式，使用3×3滤波器的序列，而不是使用更大的卷积滤波器。检测模块的输出通道数量（即图4中的X），M1中为128，M2和M3为256。应当说明的是，我们的检测模块，与上下文滤波器一起，与[24]中的建议生成模块相比，参数数量要更少。虽然效率更高了，我们通过经验还是发现，上下文模块改进了在WIDER验证集上的mAP超过0.5%。

### 3.4. Training 训练

We use stochastic gradient descent with momentum and weight decay for training the network. As discussed in section 3.2, we place three detection modules on layers with different strides to detect faces with different scales. Consequently, our network has three multi-task losses for the classification and regression branches in each of these modules as discussed in Section 3.4.1. To specialize each of the three detection modules for a specific range of scales, we only back-propagate the loss for the anchors which are assigned to faces in the corresponding range. This is implemented by distributing the anchors based on their size to these three modules (i.e. smaller anchors are assigned to M1 compared to M2 , and M3). An anchor is assigned to a ground-truth face if and only if it has a higher IoU than 0.5. This is in contrast to the methods based on Faster R-CNN which assign to each ground-truth at least one anchor with the highest IoU. Thus, we do not back-propagate the loss through the network for ground-truth faces inconsistent with the anchor sizes of a module.

我们使用带有动量的随机梯度下降和权重衰减进行网络训练。如3.2节所述，我们放置了三个步长不同的检测模块，以检测不同大小的人脸。结果是，我们的网络有三个多任务损失的分支，在每个模块中都进行分类和回归，如3.4.1节所述。为使每个检测模块专注于某一范围的尺度，我们只将特定损失反向传播回来，即指定对应范围的人脸的锚框。这个的实现，是通过将锚框根据其大小分配给三个模块实现的，即指定给M1，M2，M3的锚框逐渐变大。一个锚框指定给一个真值人脸，当且仅当其IoU大于0.5时。这与Faster R-CNN方法形成对比，给每个真值指定至少一个锚框，原则是IoU最大。因此，对于与真值人脸不一致的锚框大小，我们不将其损失反向传播回对应的模块。

#### 3.4.1 Loss function

SSH has a multi-task loss. This loss can be formulated as follows: 多损失函数的公式如下：

$$\sum_k \frac {1}{N_k^c} \sum_{i∈A_k} l_c(p_i, g_i) + λ \sum_k \frac {1}{N_k^r} \sum_{i∈A_k} I(g_i = 1) l_r(b_i, t_i)$$(1)

where $l_c$ is the face classification loss. We use standard multinomial logistic loss as $l_c$. The index k goes over the SSH detection modules $M = \{ M_k \}_1^K$ and $A_k$ represents the set of anchors defined in $M_k$. The predicted category for the i-th anchor in $M_k$ and its assigned ground-truth label are denoted as $p_i$ and $g_i$ respectively. As discussed in Section 3.2, an anchor is assigned to a ground-truth bounding box if and only if it has an IoU greater than a threshold (i.e. 0.5). As in [24], negative labels are assigned to anchors with IoU less than a predefined threshold (i.e. 0.3) with any ground-truth bounding box. $N_k^c$ is the number of anchors in module $M_k$ which participate in the classification loss computation.

其中$l_c$是人脸分类损失。我们的$l_c$使用标准多项式logistic损失。索引k代表所有SSH检测模块$M = \{ M_k \}_1^K$，$A_k$表示$M_k$中定义的锚框集。$M_k$中第i个锚框预测的类别表示为$p_i$，其指定的真值标签表示为$g_i$。如3.2节讨论，锚框指定给一个真值框，当且仅当其IoU大于一定阈值（即0.5）。和[24]中一样，对于锚框与真值框IoU小于预定义阈值（如0.3）的，就给这个锚框指定一个负标签。$N_k^c$是$M_k$模块中对分类损失有贡献的锚框数量。

$l_r$ represents the bounding box regression loss. Following [6, 5, 24], we parameterize the regression space with a log-space shift in the box dimensions and a scale-invariant translation and use smooth $l_1$ loss as $l_r$. In this parametrized space, $p_i$ represents the predicted four dimensional translation and scale shift and $t_i$ is its assigned ground-truth regression target for the i-th anchor in module $M_k$. I(.) is the indicator function that limits the regression loss only to the positively assigned anchors, and $N_k^r = \sum_{i∈A_k} I(g_i = 1)$.

$l_r$代表边界框回归损失。按照[6,5,24]中的思想，我们将回归空间参数化为一个边界框维度的log空间的偏移和尺度不变的平移，使用平滑$l_1$损失。在这个参数化的空间中，$p_i$表示预测的四个维度的平移和尺度变化，$t_i$表示模块$M_k$中第i个锚框的指定的真值回归目标。I(.)是指示器函数，限制了回归损失只在指定的正锚框上。

### 3.5. Online hard negative and positive mining

We use online negative and positive mining (OHEM) for training SSH as described in [25]. However, OHEM is applied to each of the detection modules ($M_k$) separately. That is, for each module $M_k$, we select the negative anchors with the highest scores and the positive anchors with the lowest scores with respect to the weights of the network at that iteration to form our mini-batch. Also, since the number of negative anchors is more than the positives, following [4], 25% of the mini-batch is reserved for the positive anchors. As empirically shown in Section 4.8, OHEM has an important role in the success of SSH which removes the fully connected layers out of the VGG-16 network.

我们使用在线负样本和正样本挖掘(OHEM)来训练SSH，和[25]中一样。但是，OHEM分别对每个检测模块($M_k$)单独使用。也就是说，对于每个模块$M_k$，我们在那次迭代中，选择对于网络权重最高评分的负锚框和最低评分的正锚框，以形成mini-batch。同时，既然负锚框的数量比正的更多，按照[4]中的方法，每个mini-batch的25%为正锚框保留。根据4.8中的经验显示，OHEM在SSH的成功中占重要作用，足以将VGG-16中的全连接层移除掉。

## 4. Experiments

### 4.1. Experimental Setup

All models are trained on 4 GPUs in parallel using stochastic gradient descent. We use a mini-batch of 4 images. Our networks are fine-tuned for 21K iterations starting from a pre-trained ImageNet classification network. Following [4], we fix the initial convolutions up to conv3-1. The learning rate is initially set to 0.004 and drops by a factor of 10 after 18K iterations. We set momentum to 0.9, and weight decay to 5e^−4. Anchors with IoU> 0.5 are assigned to positive class and anchors which have an IoU< 0.3 with all ground-truth faces are assigned to the background class. For anchor generation, we use scales {1, 2} in M1, {4, 8} in M2, and {16, 32} in M3 with a base anchor size of 16 pixels. All anchors have aspect ratio of one. During training, 256 detections per module is selected for each image. During inference, each module outputs 1000 best scoring anchors as detections and NMS with a threshold of 0.3 is performed on the outputs of all modules together.

所有模型都在4个GPUs上使用SGD进行并行训练。我们使用mini-batch为4幅图像。我们的网络从ImageNet分类网络预训练开始，精调21K次迭代。按照[4]中的思想，我们固定初始卷积到conv3-1。学习速率初始化为0.004， 18K次迭代后除以10。我们设动量为0.9，权重衰减为5e^−4。IoU大于0.5的锚框指定为正锚框，与所有真值框IoU小于0.3的锚框为背景类。锚框的生成，我们使用的基准锚框大小为16像素，M1中尺度为{1, 2}，M2中为{4, 8}，M3中为{16, 32}。所有锚框的纵横比都是1。在训练过程中，每幅图像中，选择每个模块的256个检测结果。在推理中，每个模块输出1000个最好分数的锚框作为检测，在所有模块结果的合集上使用阈值0.3的NMS。

### 4.2. Datasets

**WIDER dataset[35]**: This dataset contains 32, 203 images with 393, 703 annotated faces, 158, 989 of which are in the train set, 39, 496 in the validation set and the rest are in the test set. The validation and test set are divided into “easy”, “medium”, and “hard” subsets cumulatively (i.e. the “hard” set contains all images). This is one of the most challenging public face datasets mainly due to the wide variety of face scales and occlusion. We train all models on the train set of the WIDER dataset and evaluate on the validation and test sets. Ablation studies are performed on the the validation set (i.e. “hard” subset).

**WIDER数据集**：这个数据集包含32203幅图像，标注了393703个人脸，其中158989个为训练集，39496为验证集，剩下的是测试集。验证和测试集都分为简单、普通和困难累积型子集（困难集包含所有图像）。这是最有挑战性的公开人脸数据集，主要是因为人脸尺度和遮挡情况非常多。我们在WIDER数据集的训练集上训练所有模型，在验证测试集上评估模型。分离试验在验证集上进行（即，困难子集）。

**FDDB[8]**: FDDB contains 2845 images and 5171 annotated faces. We use this dataset only for testing. FDDB包含2845图向，5171个标注的人脸。这个数据集我们只用于测试。

**Pascal Faces[30]**: Pascal Faces is a subset of the Pascal VOC dataset [3] and contains 851 images annotated for face detection. We use this dataset only to evaluate our method. PASCAL Faces是PASCAL VOC数据集的一个子集，包含851幅图像，用于人脸检测。这个数据集只用于评估。

### 4.3. WIDER Dataset Result

We compare SSH with HR [7], CMS-RCNN [38], Multitask Cascade CNN [37], LDCF [20], Faceness [34], and Multiscale Cascade CNN [35]. When reporting SSH without an image pyramid, we rescale the shortest side of the image up to 1200 pixels while keeping the largest side below 1600 pixels without changing the aspect ratio. SSH+Pyramid is our method when we apply SSH to a pyramid of input images. Like HR, a four level image pyramid is deployed. To form the pyramid, the image is first scaled to have a shortest side of up to 800 pixels and the longest side less than 1200 pixels. Then, we scale the image to have min sizes of 500, 800, 1200, and 1600 pixels in the pyramid. All modules detect faces on all pyramid levels, except M3 which is not applied to the largest level.

我们将SSH与HR，CMS-RCNN，Multitash Cascade CNN，LDCF，Faceness，Multiscale Cascade CNN进行了比较。SSH不用图像金字塔作输入时，我们将输入图像的短边变到最多1200像素，长边保持小于1600像素，不改变纵横比。SSH+Pyramid是以图像金字塔作为SSH的输入的方法。与HR一样，使用了四级图像金字塔。为创建图像金字塔，首先将图像短边变为最大800像素，长边小于1200像素。然后，我们将图像变到短边500,800,1200和1600像素，形成金字塔。所有模块都在所有金字塔级别上检测人脸，除了M3不检测最大级的人脸。

Table 1 compares SSH with best performing methods on the WIDER validation set. SSH without using an image pyramid and based on the VGG-16 network outperforms the VGG-16 version of HR by 5.7%, 6.3%, and 6.5% in “easy”, “medium”, and “hard” subsets respectively. Surprisingly, SSH also outperforms HR based on ResNet-101 on the whole dataset (i.e. “hard” subset) by 0.8. In contrast HR deploys an image pyramid. Using an image pyramid, SSH based on a light VGG-16 model, outperforms the ResNet-101 version of HR by a large margin, increasing the state-of-the-art on this dataset by ∼ 4%.

表1将SSH与目前最好的方法在WIDER验证集上进行了比较。SSH在没有使用图像金字塔时，基于VGG-16网络，超过了VGG-16版的HR，在简单、普通和困难集上分别超出了5.7%, 6.3%和6.5%。令人惊讶的是，SSH还在整个数据集上（即困难子集）超过了基于ResNet-101的HR 0.8，而HR使用了图像金字塔作为输入。SSH在使用图像金字塔的情况下，基于轻量VGG-16模型，超过了基于ResNet-101的HR非常多，将这个数据集上目前最好的结果提升了～4%。

Table 1: Comparison of SSH with top performing methods on the validation set of the WIDER dataset

Method | easy | medium | hard
--- | --- | --- | ---
CMS-RCNN [38] | 89.9 | 87.4 | 62.9
HR(VGG-16)+Pyramid [7] | 86.2 | 84.4 | 74.9
HR(ResNet-101)+Pyramid [7] | 92.5 | 91.0 | 80.6
SSH(VGG-16) | 91.9 | 90.7 | 81.4
SSH(VGG-16)+Pyramid | 93.1 | 92.1 | 84.5

The precision-recall curves on the test set is presented in Figure 5. We submitted the detections of SSH with an image pyramid only once for evaluation. As can be seen, SSH based on a headless VGG-16, outperforms the prior methods on all subsets, increasing the state-of-the-art by 2.5%.

图5中给出了在测试集上的精度-召回曲线。我们将SSH使用图像金字塔的检测结果提交进行评估。可以看到，基于无头VGG-16的SSH，在所有子集上都超过了之前的方法，将目前最好的结果提升了2.5%。

### 4.4. FDDB and Pascal Faces Results

In these datasets, we resize the shortest side of the input to 400 pixels while keeping the larger side less than 800 pixels, leading to an inference time of less than 50 ms/image. We compare SSH with HR[7], HR-ER[7], Conv3D[13], Faceness[34], Faster R-CNN(VGG-16)[24], MTCNN[37], DP2MFD[21], and Headhunter[18]. Figures 6a and 6b show the ROC curves with respect to the discrete and continuous measures on the FDDB dataset respectively.

在这些数据集上，我们将输入图像短边变换到400像素，保持长边小于800像素。推理时间每幅图像小于50ms。我们将SSH与HR[7], HR-ER[7], Conv3D[13], Faceness[34], Faster R-CNN(VGG-16)[24], MTCNN[37], DP2MFD[21], and Headhunter[18]进行了比较。图6(a)和(b)为在FDDB数据集上离散度量和连续度量下的ROC曲线。

It should be noted that HR-ER also uses FDDB as a training data in a 10-fold cross validation fashion. Moreover, HR-ER and Conv3D both generate ellipses to decrease the localization error. In contrast, SSH does not use FDDB for training, and is evaluated on this dataset out-of-the-box by generating bounding boxes. However, as can be seen, SSH outperforms all other methods with respect to the discrete score. Compare to HR, SSH improved the results by 5.6% and 1.1% with respect to the continuous and discrete scores.

应当指出，HR-ER还使用FDDB作为训练数据，而且进行了10倍交叉验证。而且，HR-ER和Conv3D都生成了椭圆，以降低定位错误。比较之下，SSH没有将FDDB作为训练集，在这个数据集上生成边界框进行评估。但是，可以看到，SSH在离散分数上超过了所有其他方法。与HR相比，SSH在连续分数和离散分数上改进了结果5.6%和1.1%。

We also compare SSH with Faster R-CNN(VGG-16)[24], HyperFace[22], Headhunter[18], and Faceness[34] on the Pascal-Faces dataset. As shown in Figure 6c, SSH achieves state-of-the-art results on this dataset.

我们还在PASCAL Faces数据集上将SSH与Faster R-CNN(VGG-16)[24], HyperFace[22], Headhunter[18], and Faceness[34]进行了比较。如图6c所示，SSH在这个数据集上取得了目前最好的结果。

### 4.5. Timing

SSH performs face detection in a single stage while removing all fully-connected layers from the VGG-16 network. This makes SSH an efficient detection algorithm. Table 2 shows the inference time with respect to different input sizes. We report average time on the WIDER validation set. Timing are performed on a NVIDIA Quadro P6000 GPU. In column with max size m × M , the shortest side of the images are resized to “m” pixels while keeping the longest side less than “M ” pixels. As shown in section 4.3, and 4.4, SSH outperforms HR on all datasets without an image pyramid. On WIDER we resize the image to the last column and as a result detection takes 182 ms/image. In contrast, HR has a runtime of 1010 ms/image, more than 5X slower. As mentioned in Section 4.4, a maximum input size of 400 × 800 is enough for SSH to achieve state-of-the-art performance on FDDB and Pascal-Faces, with a detection time of 48 ms/image. If an image pyramid is used, the runtime would be dominated by the largest scale.

SSH在一阶段中进行人脸检测，去掉了VGG-16中的所有全连接层。这使SSH算法非常高效。表2给出了不同输入大小下的推理时间。我们在WIDER验证集上给出平均时间，计时是在NVidia Quadro P6000 GPU上进行。短边变换到m像素大小，长边小于M像素。如4.3、4.4节所示，SSH在没有图像金字塔的情况下在所有数据集上都超过了HR算法。在WIDER上，我们将图像变换到最后一列大小，检测结果耗时182ms每图像。比较之下，HR的耗时为1010ms每图像，慢了5倍多。如4.4节所述，输入大小最大为400×800，在FDDB和PASCAL-Faces上SSH就可以得到目前最好的结果，检测时间为48ms每图像。如果使用图像金字塔，最大尺度的输入会成为主要运行时间部分。

Table 2: SSH inference time with respect to different input sizes.

Max Size| 400 × 800 | 600 × 1000 | 800 × 1200 | 1200 × 1600
--- | --- | --- | --- | ---
Time | 48 ms | 74 ms | 107 ms | 182 ms

### 4.6. Ablation study: Scale-invariant design

As discussed in Section 3.2, SSH uses each of its detections modules, $\{ M_i \}_{i=1}^3$, to detect faces in a certain range of scales from layers with different strides. To better understand the impact of these design choices, we compare the results of SSH with and without multiple detection modules. That is, we remove {M1, M3} and only detect faces with M2 from conv5-3 in VGG-16. However, for fair comparison, all anchor scales in {M1, M3} are moved to M2 (i.e. we use $∪^3_{i=1} S_i$ in M2). Other parameters remain the same. We refer to this simpler method as ”SSH-Only M2 ”. As shown in Figure 7a, by removing the multiple detection modules from SSH, the AP significantly drops by ∼ 12.8% on the hard subset which contains smaller faces. Although SSH does not deploy the expensive head of its underlying network, results suggest that having independent simple detection modules from different layers of the network is an effective strategy for scale-invariance.

如3.2节所讨论，SSH每个检测模块$\{ M_i \}_{i=1}^3$，从不同步长的层中检测一定尺度范围内的人脸。为更好的理解这些设计选择的影响，我们将SSH有与没有多个检测模块的结果进行了比较。即，我们去掉了{M1, M3}，只用M2从VGG-16的conv5-3中检测人脸。但是，为公平比较，所有{M1, M3}中的锚框都移到了M2中，即在M2中使用$∪^3_{i=1} S_i$。其他参数保持相同。我们称这种更简单的方法为SSH-Only M2。如图7a所示，从SSH去掉了多个检测模块后，AP在困难子集上显著下降了约～12.8%。虽然SSH没有使用计算量大的头，结果说明，从网络中的不同层进行独立的简单检测，是有效的尺度不变的策略。

### 4.7. Ablation study: The effect of input size

The input size can affect face detection precision, especially for small faces. Table 3 shows the AP of SSH on the WIDER validation set when it is trained and evaluated with different input sizes. Even at a maximum input size of 800 × 1200, SSH outperforms HR-VGG16, which up-scales images up to 5000 pixels, by 3.5%, showing the effectiveness of our scale-invariant design for detecting small faces.

输入大小会影响检测准确率，尤其是对小的人脸。表3所示的是，SSH在验证集上使用不同的输入大小进行训练和评估时的AP比较。即使最大输入大小为800×1200，SSH也超过了HR-VGG16 3.5%，其最大图像大小可达5000像素，这表明我们的尺度不变设计对检测较小人脸是非常有效的。

Table 3: The effect of input size on average precision.

Max Size | 600 × 1000 | 800 × 1200 | 1200 × 1600 | 1400 × 1800
--- | --- | --- | --- | ---
AP | 68.6 | 78.4 | 81.4 | 81.0

### 4.8. Ablation study: The effect of OHEM

As discussed in Section 3.5, we apply hard negative and positive mining (OHEM) to select anchors for each of our detection modules. To show its role, we train SSH, with and without OHEM. All other factors are the same. Figure 7b shows the results. Clearly, OHEM is important for the success of our light-weight detection method which does not use the pre-trained head of the VGG-16 network.

如3.5节所述，我们使用OHEM以为每个检测模块选择锚框。为展示其作用，我们在使用与不使用OHEM的情况下对SSH进行训练，所有其他参数保持不变。图7b展示了其结果。很明显，OHEM对于我们的轻量检测模型的成功非常重要，模型没有使用预训练的VGG网络。

### 4.9. Ablation study: The effect of feature fusion

In SSH, to form the input features for detection module M1, the outputs of conv4-3 and conv5-3 are fused together. Figure 7c, shows the effectiveness of this design choice. Although it does not have a noticeable computational overhead, as illustrated, it improves the AP on the WIDER validation set.

在SSH中，为形成检测模块M1的输入特征，conv4-3和conv5-3的特征融合到了一起。图7c展示了这种设计选择的有效性。虽然没有很大的计算消耗，但确实改进了在WIDER验证集上的AP。

### 4.10. Ablation study: Selection of anchor scales

As mentioned in Section 4.1, SSH uses S1 = {1, 2}, S2 = {4, 8}, S3 = {16, 32} as anchor scale sets. Figure 7d compares SSH with its slight variant which uses S1 = {0.25, 0.5, 1, 2, 3}, S2 = {4, 6, 8, 10, 12}, S3 = {16, 20, 24, 28, 32}. Although using a finer scale set leads to a slower inference, it also reduces the AP due to the increase in the number of False Positives.

如4.1节所述，SSH使用S1 = {1, 2}, S2 = {4, 8}, S3 = {16, 32}作为锚框尺度集。图7d比较了SSH的另一个变体，使用了S1 = {0.25, 0.5, 1, 2, 3}, S2 = {4, 6, 8, 10, 12}, S3 = {16, 20, 24, 28, 32}。虽然使用了更精细的尺度集，导致推理更慢了一些，但AP反而下降了，因为False Positive数量增加了。

### 4.11. Qualitative Results

Figure 8 shows some qualitative results on the Wider validation set. The colors encode the score of the classifier. Green and blue represent score 1.0 and 0.5 respectively. 图8给出了WIDER验证集上的可视化结果。颜色表示了分类器的分数。绿色和蓝色分别表示1.0和0.5。

## 5. Conclusion

We introduced the SSH detector, a fast and lightweight face detector that, unlike two-stage proposal/classification approaches, detects faces in a single stage. SSH localizes and detects faces simultaneously from the early convolutional layers in a classification network. SSH is able to achieve state-of-the-art results without using the “head” of its underlying classification network (i.e. fc layers in VGG-16). Moreover, instead of processing an input pyramid, SSH is designed to be scale-invariant while detecting different face scales in a single forward pass of the network. SSH achieves state-of-the-art performance on the challenging WIDER dataset as well as FDDB and Pascal-Faces while reducing the detection time considerably.

我们提出了SSH人脸检测器，一种快速轻量的人脸检测器，与两阶段的建议/分类方法不同的是，SSH在一阶段中检测人脸。SSH从分类网络的早期卷积层中，同时进行人脸定位和检测。SSH没有使用分类网络的头（即，VGG-16中的fc层），取得了目前最好的结果。而且，SSH不需要图像金字塔输入，在设计上就是尺度不变的，在网络的一次前向过程中就可以检测不同尺度的人脸。SSH在WIDER数据集上，以及FDDB和PASCAL-Faces数据集上取得了目前最好的结果，同时极大的降低了检测时间。