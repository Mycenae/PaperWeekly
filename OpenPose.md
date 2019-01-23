# OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

Zhe Cao et al. The Robotics Institute, Carnegie Mellon University

## Abstract 摘要

Realtime multi-person 2D pose estimation is a key component in enabling machines to have an understanding of people in images and videos. In this work, we present a realtime approach to detect the 2D pose of multiple people in an image. The proposed method uses a nonparametric representation, which we refer to as Part Affinity Fields (PAFs), to learn to associate body parts with individuals in the image. This bottom-up system achieves high accuracy and realtime performance, regardless of the number of people in the image. In previous work, PAFs and body part location estimation were refined simultaneously across training stages. We demonstrate that a PAF-only refinement rather than both PAF and body part location refinement results in a substantial increase in both runtime performance and accuracy. We also present the first combined body and foot keypoint detector, based on an internal annotated foot dataset that we have publicly released. We show that the combined detector not only reduces the inference time compared to running them sequentially, but also maintains the accuracy of each component individually. This work has culminated in the release of OpenPose, the first open-source realtime system for multi-person 2D pose detection, including body, foot, hand, and facial keypoints.

实时多人2D姿态估计是使机器能理解图像和视频中的人的关键技术组件。本文中，我们提出一种检测图像中多人的2D姿态的实时方法。提出的方法使用了非参数表示，我们称之为部位亲和向量场(PAFs)，用于将身体部位关联至图像中的个人。这种自下而上的系统可以得到很高的准确率和实时的性能，与图像中的人数无关。在之前的工作中，PAFs和身体部位位置估计通过各个训练阶段不断进行优化提炼。我们展示了一种只对PAF进行提炼的系统，而不是同时优化PAF和身体部位位置的系统，这样可以同时大幅度提升运行时表现和准确度。我们还给出了第一个身体和脚部关键点联合检测器，这是基于一个内部标注的脚部数据集，已经公开。我们证明了，这种联合检测器比按顺序检测它们减少了推理时间，而且保持了每个部件各自的准确率。本文的工作产生了OpenPose，这是第一个多人2D姿态检测的开源实时系统，包括身体、脚部、手部和脸部关键点。

**Index Terms** — 2D human pose estimation, 2D foot keypoint estimation, real-time, multiple person, part affinity fields.

## 1 Introduction 引言

In this paper, we consider a core component in obtaining a detailed understanding of people in images and videos: human 2D pose estimation—or the problem of localizing anatomical keypoints or “parts”. Human estimation has largely focused on finding body parts of individuals. Inferring the pose of multiple people in images presents a unique set of challenges. First, each image may contain an unknown number of people that can appear at any position or scale. Second, interactions between people induce complex spatial interference, due to contact, occlusion, or limb articulations, making association of parts difficult. Third, runtime complexity tends to grow with the number of people in the image, making realtime performance a challenge.

如何详细理解图像和视频中的人？本文中我们思考这个问题的一个核心技术组件：人体2D姿态估计，或定位结构关键点/部位。人体检测主要聚焦在找到个体的身体部位上。图像中的多人姿态推理提出了独特的挑战。第一，每幅图像中的人数是未知的，可能出现在任何位置和尺度上。第二，人与人之间的互动会引入复杂的空间干扰，由于接触、遮挡或肢体铰接，使得部位的关联变难。第三，运行复杂度一般随着图像中人数的增加而增长，使得实时性成为挑战。

A common approach is to employ a person detector and perform single-person pose estimation for each detection. These top-down approaches directly leverage existing techniques for single-person pose estimation, but suffer from early commitment: if the person detector fails–as it is prone to do when people are in close proximity–there is no recourse to recovery. Furthermore, their runtime is proportional to the number of people in the image, for each person detection, a single-person pose estimator is run. In contrast, bottom-up approaches are attractive as they offer robustness to early commitment and have the potential to decouple runtime complexity from the number of people in the image. Yet, bottom-up approaches do not directly use global contextual cues from other body parts and other people. Initial bottom-up methods ([1], [2]) did not retain the gains in efficiency as the final parse required costly global inference, taking several minutes per image.

常用的方法是使用人体检测器和对每个检测结果使用单人姿态估计。这种自上而下的方法直接使用了现有的单人姿态估计技术，但存在过早承诺的问题：如果人体检测器结果有错误，这在人们亲密接近的时候经常会出现错误，就会得到无法恢复的错误。而且，其运行时间是与图像中的人数成正比的，对于每个检测到的人体，都会运行一个单人姿态估计器。与之形成对比的是，自下而上的方法就很有吸引力了，因为对过早承诺的问题有稳健性，而且可能使得运行时间与图像中的人数无关。但是，自下而上的方法不会直接使用其他身体部位和其他人的全局上下文线索。初始的自下而上的方法[1,2]没有得到性能上的提升，因为最终的解析需要大量的全局推理，每幅图像要花费好几分钟。

In this paper, we present an efficient method for multi-person pose estimation with competitive performance on multiple public benchmarks. We present the first bottom-up representation of association scores via Part Affinity Fields (PAFs), a set of 2D vector fields that encode the location and orientation of limbs over the image domain. We demonstrate that simultaneously inferring these bottom-up representations of detection and association encodes sufficient global context for a greedy parse to achieve high-quality results, at a fraction of the computational cost.

本文中，我们提出一种高效的多人姿态估计方法，在多个公开基准测试中也得到了有竞争力的结果。我们提出部位亲和向量场(PAFs)，这是第一种自下而上的关联分数表示，是一个2D向量场集合，编码了肢体在图像中的位置和方向。我们证明了，同时推理这些自下而上的检测和关联表示，编码了足够的全局上下文，使得贪婪解析可以得到高质量的结果，而计算量则非常小。

An earlier version of this manuscript appeared in [3]. This version makes several new contributions. First, we prove that PAF refinement is crucial for maximizing accuracy, while body part prediction refinement is not that important. We increase the network depth but remove the body part refinement stages (Sections 3.1 and 3.2). This refined network increases both speed and accuracy by approximately 45% and 7%, respectively (detailed analysis in Sections 5.2 and 5.3). Second, we present an annotated foot dataset with 15K human foot instances that has been publicly released (Section 4.2), and we show that a combined model with body and foot keypoints can be trained preserving the speed of the body-only model while maintaining its accuracy. (described in Section 5.4). Third, we demonstrate the generality of our method by applying it to the task of vehicle keypoint estimation (Section 5.5). Finally, this work documents the release of OpenPose [4]. This open-source library is the first available realtime system for multi-person 2D pose detection, including body, foot, hand, and facial keypoints (described in Section 4). We also include a runtime comparison to Mask R-CNN [5] and Alpha-Pose [6], showing the computational advantage of our bottom-up approach (Section 5.3).

本文有一个更早的版本[3]。本文的版本主要有几个新贡献。第一，我们证明了PAF提炼对于准确率最大化来说非常关键，而身体部位预测的提炼则没那么重要。我们增加了网络的深度，但是去掉了身体部位提炼的阶段（见3.1和3.2节）。这种提炼网络使速度提升了大约45%，准确率提升了7%（详细分析见5.2和5.3节）。第二，我们提出了一种标注的脚部数据集，包括1.5万个脚部实例，并已公开（4.2节），我们展示了同时检测身体和脚部关键点的模型，在训练后可以保持只检测身体模型的速度，而同时还能维持其准确率（见5.4节）。第三，我们证明了我们模型的泛化能力，将其应用于车辆关键点估计（5.5节）。最后，本文记录了OpenPoe的发布[4]。这个开源库是第一个可用的实时多人2D姿态检测系统，包括身体、脚部、手部和面部关键点（第4部分）。我们还与Mask R-CNN[5]和Alpha-Pose[6]进行了比较，展示了我们的自下而上方法的计算量优势（5.3节）。

Fig. 1: Top: Multi-person pose estimation. Body parts belonging to the same person are linked, including foot keypoints (big toes, small toes, and heels). Bottom left: Part Affinity Fields (PAFs) corresponding to the limb connecting right elbow and wrist. The color encodes orientation. Bottom right: A 2D vector in each pixel of every PAF encodes the position and orientation of the limbs.

## 2 Related work 相关工作

**Single Person Pose Estimation**. The traditional approach to articulated human pose estimation is to perform inference over a combination of local observations on body parts and the spatial dependencies between them. The spatial model for articulated pose is either based on tree-structured graphical models [7], [8], [9], [10], [11], [12], [13], which parametrically encode the spatial relationship between adjacent parts following a kinematic chain, or non-tree models [14], [15], [16], [17], [18] that augment the tree structure with additional edges to capture occlusion, symmetry, and long-range relationships. To obtain reliable local observations of body parts, Convolutional Neural Networks (CNNs) have been widely used, and have significantly boosted the accuracy on body pose estimation [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32]. Tompson et al. [23] used a deep architecture with a graphical model whose parameters are learned jointly with the network. Pfister et al. [33] further used CNNs to implicitly capture global spatial dependencies by designing networks with large receptive fields. The convolutional pose machines architecture proposed by Wei et al. [20] used a multi-stage architecture based on a sequential prediction framework [34]; iteratively incorporating global context to refine part confidence maps and preserving multimodal uncertainty from previous iterations. Intermediate supervisions are enforced at the end of each stage to address the problem of vanishing gradients [35], [36], [37] during training. Newell et al. [19] also showed intermediate supervisions are beneficial in a stacked hourglass architecture. However, all of these methods assume a single person, where the location and scale of the person of interest is given.

**单人姿态估计**。铰接人体姿态估计的传统方法是对身体部位的局部信息和其之间的空间依赖关系进行推理。铰接姿态的空间模型有两类，一类是基于树状的图形模型的[7,8,9,10,11,12,13]，使用了参数化的方式对邻接的部位编码成一个运动链；还有一类非树模型[14,15,16,17,18]，用额外的边扩展了树状模型，以捕捉遮挡、对称和长距离关系。为得到身体部位的可靠局部观察，广泛的使用了卷积神经网络(CNNs)，这显著提升了身体姿态估计的准确性[19,20,21,22,23,24,25,26,27,28,29,30,31,32]。Tompson等[23]使用了带有图形模型的深度架构，两部分的参数在网络训练的过程中同时得到学习。Pfister等[33]设计了大感受野的网络，以隐含的捕捉全局空间依赖关系。Wei等人提出的卷积姿态机架构[20]使用了基于序贯预测框架的多阶段架构[34]，迭代着不断加入全局上下文以提炼部位置信度图，并从之前的迭代中保存多模不确定性；训练过程中每一阶段结束的时候都加入中间监督以解决梯度消失的问题[35,36,37]。Newell等[19]也展示了在堆叠的沙漏架构中加入中间监督是有益的。但是，所有这些方法都假设只有一个人，而且人的位置和尺度都已经给定。

**Multi-Person Pose Estimation**. For multi-person pose estimation, most approaches [5], [6], [38], [39], [40], [41], [42], [43], [44] have used a top-down strategy that first detects people and then have estimated the pose of each person independently on each detected region. Although this strategy makes the techniques developed for the single person case directly applicable, it not only suffers from early commitment on person detection, but also fails to capture the spatial dependencies across different people that require global inference. Some approaches have started to consider inter-person dependencies. Eichner et al. [45] extended pictorial structures to take a set of interacting people and depth ordering into account, but still required a person detector to initialize detection hypotheses. Pishchulin et al. [1] proposed a bottom-up approach that jointly labels part detection candidates and associated them to individual people, with pairwise scores regressed from spatial offsets of detected parts. This approach does not rely on person detections, however, solving the proposed integer linear programming over the fully connected graph is an NP-hard problem and thus the average processing time for a single image is on the order of hours. Insafutdinov et al. [2] built on [1] with a stronger part detectors based on ResNet [46] and image-dependent pairwise scores, and vastly improved the runtime with an incremental optimization approach, but the method still takes several minutes per image, with a limit of at most 150 part proposals. The pairwise representations used in [2], which are offset vectors between every pair of body parts, are difficult to regress precisely and thus a separate logistic regression is required to convert the pairwise features into a probability score.

**多人姿态估计**。对于多人姿态估计，多数方法都使用了自上而下的策略[5,6,38,39,40,41,42,43,44]，首先检测人体，然后独立的对每个人在每个检测到的区域进行姿态估计。虽然这个策略可以直接使用已有的单人姿态估计技术，但存在过早承诺的问题，而且不能捕捉不同人之间的空间依赖关系，这需要进行全局推理。一些方法开始考虑人与人之间的依赖关系。Eichner等[45]扩展了pictorial结构，考虑了多人交互和深度排序的情况，但仍然需要人体检测器来初始化检测假设。Pishchulin等[1]提出了一种自下而上的方法，同时标准部位检测候补，并与个体关联，并从检测到的身体部位的空间偏移回归得到成对的分数。这种方法不依赖于人体检测，但是，对在全连接图上提出的整数线性规划问题进行求解，是一个NP难题，所以每幅图像的平均处理时间要以小时计。Insafutdinov等[2]在[1]的基础上，基于ResNet[46]构建了一个更强的部位检测器，和与图像有关的成对分数，使用渐进优化方法极大的改进了运行时间，但这种方法每幅图像处理时间仍需要几分钟，最多支持150个部位候选。[2]中使用的成对表示，是每对身体部位之间的偏移向量，很难准确回归，所以需要一个分离的logistic回归来将成对的特征转化成概率分数。

In earlier work [3], we present part affinity fields (PAFs), a representation consisting of a set of flow fields that encodes unstructured pairwise relationships between body parts of a variable number of people. In contrast to [1] and [2], we can efficiently obtain pairwise scores from PAFs without an additional training step. These scores are sufficient for a greedy parse to obtain high-quality results with realtime performance for multi-person estimation. Concurrent to this work, Insafutdinov et al. [47] further simplified their body-part relationship graph for faster inference in single-frame model and formulated articulated human tracking as spatio-temporal grouping of part proposals. Recenetly, Newell et al. [48] proposed associative embeddings which can be thought as tags representing each keypoint’s group. They group keypoints with similar tags into individual people. Papandreou et al. [49] proposed to detect individual keypoints and predict their relative displacements, allowing a greedy decoding process to group keypoints into person instances. Kocabas et al. [50] proposed a Pose Residual Network which receives keypoint and person detections, and then assigns keypoints to detected person bounding boxes. Nie et al. [51] proposed to partition all keypoint detections using dense regressions from keypoint candidates to centroids of persons in the image.

在前面的工作[3]中，我们提出了部位亲和向量场(PAFs)，这种表示是一个向量场集合，编码了未结构化的未知人数的身体部位之间的成对关系。与[1,2]对比，我们可以高效的从PAFs中得到成对的分数，不需要额外的训练步骤。这些分数足够进行贪婪解析，以对多人估计任务实时的得到高质量结果。与本文同时，Insafutdinov等[47]进一步简化了其身体部位关系图以更快的在单帧模型中推理，将铰接人体跟踪的问题作为部位候选的时空群聚。最近，Newell等[48]提出了关联性嵌套，可以认为是表示每个关键点的组的标签。他们对关键点分组，将类似标签的分组进个体。Papandreou等[49]提出检测个体的关键点并预测其相对位移，使贪婪解码过程将关键点分组成为人体实例。Kocabas等[50]提出了姿态残差网络，接受关键点和人体检测输入，然后将关键点指定给检测到的人的边界框。Nie等[51]提出了对关键点候选使用密集回归来分割所有关键点检测。

In this work, we make several extensions to our earlier work [3]. We prove that PAF refinement is critical and sufficient for high accuracy, removing the body part confidence map refinement while increasing the network depth. This leads to a computationally faster and more accurate model. We also present the first combined body and foot keypoint detector, created from an annotated foot dataset that will be publicly released. We prove that combining both detection approaches not only reduces the inference time compared to running them independently, but also maintains their individual accuracy. Finally, we present OpenPose, the first open-source library for real time body, foot, hand, and facial keypoint detection.

本文中，我们对早期的工作[3]做了几个拓展。我们证明了PAF提炼对于高准确率是关键的也是充分的，去除了身体部位置信度图提炼过程，同时增加了网络深度。这使得运算速度更快，模型更加准确。我们还提出第一个综合身体和脚部关键点检测器，提出了一个标注的脚部数据集，将来会公开放出。我们证明了综合两种检测的方法不仅降低了推理时间（与单独运行两种检测相比），还保持了其单独的准确率。最后，我们提出了OpenPose，可以实时进行身体、脚部、手部和面部的关键点检测，这是第一个这种功能的开源库。

## 3 Method 方法

Fig. 2 illustrates the overall pipeline of our method. The system takes, as input, a color image of size w × h (Fig. 2a) and produces the 2D locations of anatomical keypoints for each person in the image (Fig. 2e). First, a feedforward network predicts a set of 2D confidence maps S of body part locations (Fig. 2b) and a set of 2D vector fields L of part affinities, which encode the degree of association between parts (Fig. 2c). The set $S = (S_1, S_2, ..., S_J)$ has J confidence maps, one per part, where $S_j ∈ R^{w×h}, j ∈ \{1...J\}$. The set $L = (L_1, L_2, ..., L_C)$ has C vector fields, one per limb( We refer to part pairs as limbs for clarity, despite the fact that some pairs are not human limbs (e.g., the face).), where $L_c ∈ R^{w×h×2}, c ∈ \{1...C\}$. Each image location in $L_c$ encodes a 2D vector (as shown in Fig. 1). Finally, the confidence maps and the affinity fields are parsed by greedy inference (Fig. 2d) to output the 2D keypoints for all people in the image.

图2给出了我们方法的总体流程。系统输入为大小为w×h的彩色图像（图2a），生成图中每个人解剖关键点的2D位置（图2e）。首先，一个前向网络预测出身体部位位置的2D置信度图S的集合（图2b），和部位亲和度的2D向量场L的集合，其中编码的是部位之间的关联程度（图2c）。集合$S = (S_1, S_2, ..., S_J)$有J个置信度图，每个部位一个图，其中$S_j ∈ R^{w×h}, j ∈ \{1...J\}$。集合$L = (L_1, L_2, ..., L_C)$有C个向量场，每个肢体一个（我们称成对的部位为肢体，虽然一些成对的不是人体肢体，如脸部），其中$L_c ∈ R^{w×h×2}, c ∈ \{1...C\}$。$L_c$中的每个图像位置都编码了一个2D向量（如图1所示）。最后，置信度图和亲和向量场由贪婪推理进行解析（图2d），输出图像中所有人的2D关键点。

Fig. 2: Overall pipeline. (a) Our method takes the entire image as the input for a CNN to jointly predict (b) confidence maps for body part detection and (c) PAFs for part association. (d) The parsing step performs a set of bipartite matchings to associate body part candidates. (e) We finally assemble them into full body poses for all people in the image.

图2：总体流程。(a)我们的方法输入为整个图像，图像经过CNN处理，同时预测(b)身体部位检测的置信度图，和(c)部位关联的PAFs，(d)解析步骤进行一系列双向匹配，关联身体部位候选，(e)最后，我们将这些身体部位组装成图像中所有人的整个人体姿态。

### 3.1 Network Architecture 网络架构

Our architecture, shown in Fig. 3, iteratively predicts affinity fields that encode part-to-part association, shown in blue, and detection confidence maps, shown in beige. The iterative prediction architecture, following [20], refines the predictions over successive stages, t ∈ {1,...,T}, with intermediate supervision at each stage. The network depth is increased with respect to [3]. In the original approach, the network architecture included several 7x7 convolutional layers. In our current model, the receptive field is preserved while the computation is reduced, by replacing each 7x7 convolutional kernel by 3 consecutive 3x3 kernels. While the number of operations for the former is $2 × 7^2 − 1 = 97$, it is only 51 for the latter. Additionally, the output of each one of the 3 convolutional kernels is concatenated, following an approach similar to DenseNet [52]. The number of non-linearity layers is tripled, and the network can keep both lower level and higher level features. Sections 5.2 and 5.3 analyze the accuracy and runtime speed improvements, respectively.

我们的架构如图3所示，不断迭代着预测亲和向量场，其中编码了部位和部位间的关联，以蓝色显示，和检测置信度图，以米黄色显示。这种迭代预测架构与[20]中类似，在几个连续的阶段中提炼出好的预测，t ∈ {1,...,T}，每个阶段中都有中间监督。网络深度比[3]中更深。在原来的方法中，网络架构中包括了几个7×7的卷积层。在我们现在的模型中，感受野的大小得到了保持，同时计算量减小了，原因是我们将每个7×7的卷积核替换成了3个连续的3×3卷积核。前者的运算数量为$2 × 7^2 − 1 = 97$，后者只有51。另外，这3个卷积核的三个输出进行了拼接，学习的是DenseNet[52]的类似的方法。非线性层的数量增加到三倍，网络可以同时保持底层特征和高层特征。5.2节和5.3节分别分析了准确率和运行时间的改进。

Fig. 3: Architecture of the multi-stage CNN. First 4 stages predict PAFs $L^t$, while the last 2 stages predict confidence maps $S^t$. The predictions of each stage and their corresponding image features are concatenated for each subsequent stage. Convolutions of kernel size 7 from the original approach [3] are replaced with 3 layers of convolutions of kernel 3 which are concatenated at their end.

图3：多阶段CNN的架构。前面4个阶段预测PAFs $L^t$，最后2个阶段预测的是置信度图$S^t$。每个阶段的预测及其对应的图像特征拼接起来送入后续的阶段。原始方法[3]中的核心大小为7的卷积替换为3层核心大小为3的卷积，最后拼接到一起。

### 3.2 Simultaneous Detection and Association 同步检测并关联

The image is analyzed by a convolutional network (initialized by the first 10 layers of VGG-19 [53] and fine-tuned), generating a set of feature maps F that is input to the first stage. At this stage, the network produces a set of part affinity fields (PAFs) $L^1 = φ^1 (F)$, where $φ^1$ refers to the CNNs for inference at Stage 1. In each subsequent stage, the predictions from the previous stage and the original image features F are concatenated and used to produce refined predictions,

网络送入CNN中分析（由VGG-19[53]的前10层初始化并精调），生成特征图集F，送入第1阶段。在这一阶段，网络生成部位亲和向量场(PAFs)集合$L^1 = φ^1 (F)$，其中$φ^1$指阶段1的CNN的推理。在每个后续的阶段，前一阶段的预测和图像原始特征F拼接起来用于生成提炼后的预测。

$$L^t = φ^t (F,L^{t−1}), ∀2 ≤ t ≤ T_P$$(1)

where $φ^t$ refers to the CNNs for inference at Stage t, and $T_P$ to the number of total PAF stages. After $T_P$ iterations, the process is repeated for the confidence maps detection, starting in the most updated PAF prediction,

其中$φ^t$指的是阶段t的CNN推理，$T_P$指的是总计的PAF阶段。在$T_P$次迭代后，重复进行的是置信度图的检测，从最新的PAF预测处开始，

$$S^{T_P} = φ^t (F, L^{T_P}), ∀t = T_P$$(2)
$$S^t = φ^t (F, L^{T_P}, S^{t−1}), ∀T_P < t ≤ T_P + T_C$$(3)

where $φ^t$ refers to the CNNs for inference at Stage t , and $T_C$ to the number of total confidence map stages. 其中$φ^t$指在阶段t的CNN推理，$T_C$是总计的置信度图阶段数量。

This approach differs from [3], where both the affinity field and confidence map branches were refined at each stage. Hence, the amount of computation per stage is reduced by half. We empirically observe in Section 5.2 that refined affinity field predictions improve the confidence map results, while the opposite does not hold. Intuitively, if we look at the PAF channel output, the body part locations can be guessed. However, if we see a bunch of body parts with no other information, we cannot parse them into different people.

这种方法与[3]不同，[3]中每个阶段都进行亲和向量场和置信度图两个分支的精炼。所以现在每个阶段的计算量大致减半了。在5.2节中，我们经验性的发现精炼后的亲和向量场预测可以改进置信度图的结果，而反过来则不行。根据直观感觉，如果我们观察PAF通道的输出，是可以猜测出来身体部位的位置的。但是，如果我们观察很多身体部位而没有其他信息，则不能将其解析成不同的人。

Fig. 4 shows the refinement of the affinity fields across stages. The confidence map results are predicted on top of the latest and most refined PAF predictions, resulting in a barely noticeable difference across confidence map stages. To guide the network to iteratively predict PAFs of body parts in the first branch and confidence maps in the second branch, we apply a loss function at the end of each stage. We use an $L_2$ loss between the estimated predictions and the groundtruth maps and fields. Here, we weight the loss functions spatially to address a practical issue that some datasets do not completely label all people. Specifically, the loss function of the PAF branch at stage $t_i$ and loss function of the confidence map branch at stage $t_k$ are:

图4所示的是不同阶段中亲和向量场的精炼过程。置信度图结果是在最新的也是最准确的PAF预测之上进行预测的，所以在置信度图预测的阶段中，各阶段区别很小。为引导网络在第一分支中迭代预测身体部位的PAFs，在第二分支中迭代预测置信度图，我们在每个阶段的最后都设置了一个损失函数。我们使用估计的预测和真值图/场之间的$L_2$损失。这里，我们对损失函数按空间位置进行加权，以解决一个实际问题，就是一些数据集没有完全标记所有人。特别的，阶段$t_i$的PAF分支的损失函数，以及阶段$t_k$的置信度图分支的损失函数，分别为：

$$f_L^{t_i} = \sum_{c=1}^C \sum_p W(p) ·||L_c^{t_i}(p) - L_c^*(p)||_2^2$$(4)
$$f_S^{t_k} = \sum_{j=1}^J \sum_p W(p) ·||S_j^{t_k}(p) - S_j^*(p)||_2^2$$(5)

where $L^∗_c$ is the groundtruth part affinity vector field, $S^∗_j$ is the groundtruth part confidence map, and W is a binary mask with W(p) = 0 when the annotation is missing at an image location p. The mask is used to avoid penalizing the true positive predictions during training. The intermediate supervision at each stage addresses the vanishing gradient problem by replenishing the gradient periodically [20]. The overall objective is

其中$L^∗_c$是真值部位亲和向量场，$S^∗_j$是真值部位置信度图，W是二值掩模，如果在图像位置p上的标注缺失，则W(p) = 0。掩模用于避免在训练过程中惩罚了真阳性预测。每个阶段的中间监督解决了梯度消失的问题，周期性的补充梯度。总体的目标函数为

$$f = \sum_{t=1}^{T_P} f_L^t + \sum_{t=T_P+1}^{T_P+T_C} f_S^t$$(6)

Fig. 4: PAFs of right forearm across stages. Although there is confusion between left and right body parts and limbs in early stages, the estimates are increasingly refined through global inference in later stages.

图4. 不同阶段右前臂的PAFs。虽然在早期阶段身体部位和肢体的左边和右边有混淆，但在后续阶段中，通过全局推理，这些估计得到了很好的提炼。

### 3.3 Confidence Maps for Part Detection 部位检测的置信度图

To evaluate $f_S$ in Eq. (6) during training, we generate the groundtruth confidence maps $S^∗$ from the annotated 2D keypoints. Each confidence map is a 2D representation of the belief that a particular body part can be located in any given pixel. Ideally, if a single person appears in the image, a single peak should exist in each confidence map if the corresponding part is visible; if multiple people are in the image, there should be a peak corresponding to each visible part j for each person k.

为在训练过程中求式(6)中的$f_S$，我们从标注的2D关键点中生成了真值置信度图$S^∗$。每个置信度图都是一个特定身体部位在给定像素位置上的信心的2D表示。理想情况下，如果单人出现在图像中，如果一个部位是可见的，那么在置信度图的对应位置上应当存在一个单峰；如果图像中有多人，那么每个人k的每个可见部位j都应当有一个对应的峰值。

We first generate individual confidence maps $S^∗_{j,k}$ for each person k. Let $x_{j,k} ∈ R^2$ be the groundtruth position of body part j for person k in the image. The value at location $p ∈ R^2$ in $S^∗_{j,k}$ is defined as,

我们首先为每个人k生成单个置信度图$S^∗_{j,k}$。令$x_{j,k} ∈ R^2$是图像中人k的身体部位j的真值位置，$S^∗_{j,k}$在位置$p ∈ R^2$上的值定义为，

$$S^*_{j,k} (p) = exp(-\frac{||p-x_{j,k}||^2_2}{σ^2})$$(7)

where σ controls the spread of the peak. The groundtruth confidence map predicted by the network is an aggregation of the individual confidence maps via a max operator,

这里σ控制着峰值的扩散。网络预测的真值置信度图是单个置信度图的max操作符聚集

$$S^*_j (p) = max_k S^*_{j,k} (p)$$(8)

We take the maximum of the confidence maps instead of the average so that the precision of nearby peaks remains distinct, as illustrated in the right figure. At test time, we predict confidence maps, and obtain body part candidates by performing non-maximum suppression.

我们取置信度图的最大值，而不是均值，这样相邻的峰值可以区分开来，如右图所示。在测试时，我们预测置信度图，并通过非最大抑制来得到身体部位候选。

### 3.4 Part Affinity Fields for Part Association 部位关联的部位亲和向量场

Given a set of detected body parts (shown as the red and blue points in Fig. 5a), how do we assemble them to form the full-body poses of an unknown number of people? We need a confidence measure of the association for each pair of body part detections, i.e., that they belong to the same person. One possible way to measure the association is to detect an additional midpoint between each pair of parts on a limb and check for its incidence between candidate part detections, as shown in Fig. 5b. However, when people crowd together—as they are prone to do—these midpoints are likely to support false associations (shown as green lines in Fig. 5b). Such false associations arise due to two limitations in the representation: (1) it encodes only the position, and not the orientation, of each limb; (2) it reduces the region of support of a limb to a single point.

给定检测到的身体部位集合（如图5a中的红点和蓝点），我们怎样将其组装并形成未知人数的全身姿态呢？我们需要每对检测到的身体部位的关联置信度度量，即，它们属于同一个人的度量。一个可能的途径来衡量这种关联是，检测每个肢体的部位对的额外中间点，检查其在身体部位候选间出现的概率，如图5b所示。但是，当人们群聚在一起（人们通常倾向于这样），其中间点通常会形成误关联（如图5b中的绿线所示）。这样的误关联的出现是因为这种表示的两个局限：(1)这只编码了每个肢体的未知，而没有方向信息；(2)这将每个肢体的支持区域缩减到了一个单点。

Part Affinity Fields (PAFs) address these limitations. They preserve both location and orientation information across the region of support of the limb (as shown in Fig. 5c). Each PAF is a 2D vector field for each limb, also shown in Fig. 1d. For each pixel in the area belonging to a particular limb, a 2D vector encodes the direction that points from one part of the limb to the other. Each type of limb has a corresponding PAF joining its two associated body parts.

PAFs解决了这些局限，不仅保存了肢体支持区域的位置和方向信息（如图5c所示）。每个PAF是一个肢体的2D向量场，如图1d所示。对于一个特定肢体的所属区域中的每个像素，都有一个2D向量编码了一个身体部位到另一个身体部位的方向。每个肢体类型都有一个对应的PAF连接其关联的身体部位。

Fig. 5: Part association strategies. (a) The body part detection candidates (red and blue dots) for two body part types and all connection candidates (grey lines). (b) The connection results using the midpoint (yellow dots) representation: correct connections (black lines) and incorrect connections (green lines) that also satisfy the incidence constraint. (c) The results using PAFs (yellow arrows). By encoding position and orientation over the support of the limb, PAFs eliminate false associations.

图5.部位关联策略。(a)两种身体部位类似的检测候选（红点和蓝点）及其连接候选（灰线）；(b)使用中间点（灰点）表示的连接结果：正确的连接为黑线，同样满足概率约束的错误连接为绿线；(c)使用PAFs（黄色箭头）得到的结果。通过对肢体支持区域的位置和方向进行编码，PAFs消除了错误关联。

Consider a single limb shown in the figure below. Let $x_{j_1 ,k}$ and $x_{j_2 ,k}$ be the groundtruth positions of body parts $j_1$ and $j_2$ from the limb c for person k in the image. If a point p lies on the limb, the value at $L^∗_{c,k}(p)$ is a unit vector that points from $j_1$ to $j_2$; for all other points, the vector is zero-valued.

考虑下图中所示的一个肢体。令$x_{j_1 ,k}$和$x_{j_2 ,k}$为图像中某人k肢体c的身体部位$j_1$和$j_2$的真值位置，如果p点在肢体上，那么$L^∗_{c,k}(p)$值是从$j_1$指向$j_2$的单位向量；其他点上的向量为零。

To evaluate $f_L$ in Eq. 6 during training, we define the groundtruth part affinity vector field, $L^∗_{c,k}$ at an image point p as 为在训练过程中求式(6)中的$f_L$的值，我们定义真值PAF在图像点p上的值$L^∗_{c,k}$为

$$L^∗_{c,k}(p) = v, \space if \space p \space on \space limb \space c,k; 0, \space otherwise.$$(9)

Here, $v = (x_{j_2 ,k} − x_{j_1 ,k})/||x_{j_2 ,k} − x_{j_1 ,k} ||_2$ is the unit vector in the direction of the limb. The set of points on the limb is defined as those within a distance threshold of the line segment, i.e., those points p for which 这里v是肢体方向上的单位矢量。肢体上的点集定义为距肢体线段一定距离内的点，即满足下式条件的点

$$0 ≤ v · (p − x_{j_1 ,k} ) ≤ l_{c,k} \space and \space |v_⊥ · (p − x_{j_1 ,k})| ≤ σ_l$$

where the limb width $σ_l$ is a distance in pixels, the limb length is $l_{c,k} = ||x_{j_2 ,k} − x_{j_1 ,k}||_2$, and $v_⊥$ is a vector perpendicular to v. 其中肢体宽度$σ_l$是以像素为单位的距离，肢体长度为$l_{c,k}$，$v_⊥$是垂直于v的矢量。

The groundtruth part affinity field averages the affinity fields of all people in the image, 真值PAFs是图像中所有人的亲和向量场的平均值

$$L^∗_c (p) = \frac{1}{n_c (p)} \sum_k L^∗_{c,k} (p),$$(10)

where $n_c (p)$ is the number of non-zero vectors at point p across all k people. 其中$n_c (p)$是k个人在点p上非零向量的数量。

During testing, we measure association between candidate part detections by computing the line integral over the corresponding PAF along the line segment connecting the candidate part locations. In other words, we measure the alignment of the predicted PAF with the candidate limb that would be formed by connecting the detected body parts. Specifically, for two candidate part locations $d_{j_1}$ and $d_{j_2}$, we sample the predicted part affinity field, $L_c$ along the line segment to measure the confidence in their association:

在测试时，我们度量检测的身体部位候选间的关联的方法是，计算对应的PAF在连接候选部位位置的线段上的线积分。换句话说，我们度量预测到的PAF与连接两个检测到的身体部位所形成的候选肢体的对齐程度。特别的，对于两个候选身体部位位置$d_{j_1}$和$d_{j_2}$，我们沿着线段对预测的PAF $L_c$进行采样，来度量其关联的置信度：

$$E = \int_{u=0}^{u=1} L_c(p(u)) · \frac{d_{j_2} - d_{j_1}}{||d_{j_2} - d_{j_1}||_2} du$$(11)

where p(u) interpolates the position of the two body parts $d_{j_1}$ and $d_{j_2}$ 其中p(u)是两个身体部位$d_{j_1}$和$d_{j_2}$的位置插值

$$p(u) = (1 − u)d_{j_1} + ud_{j_2}$$(12)

In practice, we approximate the integral by sampling and summing uniformly-spaced values of u. 在实践中，我们将u值均匀间隔取样并求和，以近似积分值。

### 3.5 Multi-Person Parsing using PAFs 使用PAFs的多人解析

We perform non-maximum suppression on the detection confidence maps to obtain a discrete set of part candidate locations. For each part, we may have several candidates, due to multiple people in the image or false positives (shown in Fig. 6b). These part candidates define a large set of possible limbs. We score each candidate limb using the line integral computation on the PAF, defined in Eq. 11. The problem of finding the optimal parse corresponds to a K-dimensional matching problem that is known to be NP-Hard [54] (shown in Fig. 6c). In this paper, we present a greedy relaxation that consistently produces high-quality matches. We speculate the reason is that the pair-wise association scores implicitly encode global context, due to the large receptive field of the PAF network.

我们在检测置信度图上进行非最大抑制，以得到部位候选位置的离散集。对于每个部位，我们可能有几个候选，这是因为图像中可能有多个人，或者误检（如图6b）。这些部位候选定义了一个大型可能肢体集。我们使用式(11)中定义的PAF上的线积分，来为每个候选肢体打分。寻找最优解析的问题，对应的是K维匹配问题，这是一个NP难题[54]，如图6c所示。在本文中，我们提出了一个贪婪松弛，一直可以得到高质量的匹配。我们推测原因是，由于PAF网络的大型感受野，成对关联评分隐式的编码了全局上下文。

Formally, we first obtain a set of body part detection candidates $D_J$ for multiple people, where $D_J = \{ d^m_j :$ for $j ∈ \{1...J\}, m ∈ \{1...N_j \}\}$, where $N_j$ is the number of candidates of part j, and $d^m_j ∈ R^2$ is the location of the m-th detection candidate of body part j. These part detection candidates still need to be associated with other parts from the same person—in other words, we need to find the pairs of part detections that are in fact connected limbs. We define a variable $z^{mn}_{j_1 j_2} ∈ \{0,1\}$ to indicate whether two detection candidates $d^m_{j_1}$ and $d^n_{j_2}$ are connected, and the goal is to find the optimal assignment for the set of all possible connections, $Z = \{z^{mn}_{j_1 j_2}:$ for $j_1 ,j_2 ∈ \{1...J\}, m ∈ \{1...N_{j_1} \},n ∈ \{1...N_{j_2}\}\}$.

正式的，我们首先得到多人的身体部位检测候选集$D_J$，其中$D_J = \{ d^m_j :$ for $j ∈ \{1...J\}, m ∈ \{1...N_j \}\}$，其中$N_j$是身体部位候选j的数量，$d^m_j ∈ R^2$是身体部位j的第m个检测候选位置。这些部位检测候选仍然需要与同一个人的其他部位关联，换句话说，我们需要找到检测部位配对，使其是实际上相连的肢体。我们定义变量$z^{mn}_{j_1 j_2} ∈ \{0,1\}$来指代两个检测候选$d^m_{j_1}$和$d^n_{j_2}$是否是相连的，其目标是找到所有可能连接的集合的最佳指定，$Z = \{z^{mn}_{j_1 j_2}:$ for $j_1 ,j_2 ∈ \{1...J\}, m ∈ \{1...N_{j_1} \},n ∈ \{1...N_{j_2}\}\}$.

If we consider a single pair of parts $j_1$ and $j_2$ (e.g., neck and right hip) for the c-th limb, finding the optimal association reduces to a maximum weight bipartite graph matching problem [54]. This case is shown in Fig. 5b. In this graph matching problem, nodes of the graph are the body part detection candidates $D_{j_1}$ and $D_{j_2}$, and the edges are all possible connections between pairs of detection candidates. Additionally, each edge is weighted by Eq. 11—the part affinity aggregate. A matching in a bipartite graph is a subset of the edges chosen in such a way that no two edges share a node. Our goal is to find a matching with maximum weight for the chosen edges,

如果我们考虑部位$j_1$和$j_2$对（如脖子和右臀）的第c个肢体，找到最佳关联的问题，就成为最大加权双向图匹配问题[54]，如图5b所示。在这个图匹配问题中，图的节点是身体部位检测候选$D_{j_1}$和$D_{j_2}$，其边是检测候选对之间的所有可能连接。另外，每条边由式(11)加权，即部位亲和度聚集。双向图中的一个匹配是边的一个子集，在这个子集中，没有两条边共有一个节点。我们的目标是找到一个匹配，对于选择的边来说权值最大，

$$max_{Z_c} E_c = max_{Z_c} \sum_{m ∈ D_{j_1}} \sum_{n ∈ D_{j_2}} E_{mn}·z^{mn}_{j_1 j_2}$$(13)
$$s.t. ∀ m∈D_{j_1}, \sum_{n∈D_{j_2}} z^{mn}_{j_1 j_2}≤1$$(14)
$$∀ n∈D_{j_2}, \sum_{m∈D_{j_1}} z^{mn}_{j_1 j_2}≤1$$(15)

where $E_c$ is the overall weight of the matching from limb type c, $Z_c$ is the subset of Z for limb type c, and $E_{mn}$ is the part affinity between parts $d^m_{j_1}$ and $d^n_{j_2}$ defined in Eq. 11. Eqs. 14 and 15 enforce that no two edges share a node, i.e., no two limbs of the same type (e.g., left forearm) share a part. We can use the Hungarian algorithm [55] to obtain the optimal matching.

其中$E_c$是肢体c的匹配的总体权重，$Z_c$是肢体c在集合Z中的子集，$E_{mn}$是部位$d^m_{j_1}$和$d^n_{j_2}$根据式(11)定义的部位亲和度。式(14)(15)增加了限制，任意两条边不能共享同一个节点，即，同一种类型的两个肢体（如左前臂）不能共享同一部位。我们可以使用Hungarian算法[55]来得到最佳匹配。

When it comes to finding the full body pose of multiple people, determining Z is a K-dimensional matching problem. This problem is NP-Hard [54] and many relaxations exist. In this work, we add two relaxations to the optimization, specialized to our domain. First, we choose a minimal number of edges to obtain a spanning tree skeleton of human pose rather than using the complete graph, as shown in Fig. 6c. Second, we further decompose the matching problem into a set of bipartite matching subproblems and determine the matching in adjacent tree nodes independently, as shown in Fig. 6d. We show detailed comparison results in Section 5.1, which demonstrate that minimal greedy inference well-approximates the global solution at a fraction of the computational cost. The reason is that the relationship between adjacent tree nodes is modeled explicitly by PAFs, but internally, the relationship between nonadjacent tree nodes is implicitly modeled by the CNN. This property emerges because the CNN is trained with a large receptive field, and PAFs from non-adjacent tree nodes also influence the predicted PAF.

当要寻找多人的全身姿态时，确定Z是一个K维匹配问题，这个问题是一个NP难题[54]，存在很多松弛解。在本文中，我们为优化问题增加两个松弛条件，是我们这个领域专有的。首先，我们选择最小数量的边来得到人体姿态的支撑树骨架，而不是使用完全图，如图6c所示。第二，我们进一步将匹配问题分解为若干双向匹配子问题，在临近的树节点间独立的确定匹配，如图6d所示。我们在5.1节中展示了详细的结果比较，结果显示，最小贪婪推理以很小的计算量很好的近似了全局解。其原因是，临近的树节点之间的关系由PAFs显式的进行了建模，但在内部，非临近树节点的关系由CNN隐式的建模了。这个性质的出现，是因为CNN训练时的感受野就很大，非临近树节点的PAFs也影响到了预测的PAF。

With these two relaxations, the optimization is decomposed simply as: 在这个松弛条件下，优化问题直接分解为：

$$max_Z E = \sum_{c=1}^C max_{Z_c} E_c$$(16)

We therefore obtain the limb connection candidates for each limb type independently using Eqns. 13-15. With all limb connection candidates, we can assemble the connections that share the same part detection candidates into full-body poses of multiple people. Our optimization scheme over the tree structure is orders of magnitude faster than the optimization over the fully connected graph [1], [2].

所以我们使用式13-15对每种肢体单独得到肢体连接候选。有了所有的肢体连接候选，我们可以组装这些连接，将共享相同的部位的候选拼接起来，成为多人全身姿态。我们的树结构优化方案比全连接图优化[1,2]快了几个数量级。

Our current model also incorporates redundant PAF connections (e.g., between ears and shoulders, wrists and shoulders, ankles and hips). This redundancy particularly improves the accuracy in crowded images. Fig. 7 shows a crowded example. To handle these redundant connections, we slightly modify the multi-person parsing algorithm. While the original approach started from a root component, our algorithm sorts all pairwise possible connections by their PAF score. If a connection tries to connect 2 body parts which have already been assigned to different people, the algorithm recognizes that this would contradict a PAF connection with a higher confidence, and the current connection is subsequently ignored.

我们现在的模型还采用了冗余PAF连接（如，耳朵和肩膀之间，手腕和肩膀之间，膝盖和臀部之间）。这种冗余性尤其的改进了群聚图像的准确率。图7所示的是一幅群聚图像示例。为处理这些冗余连接，我们略微修改了多人解析算法。原方法是从根节点开始的，我们的算法根据其PAF分数对所有可能的成对连接进行排序。如果一个连接涉及到的两个身体部位已经指定给了不同人，算法会辨认出这会与一个PAF连接矛盾非常大，从而将现有的连接忽略。

Fig. 6: Graph matching. (a) Original image with part detections. (b) K -partite graph. (c) Tree structure. (d) A set of bipartite graphs.

Fig. 7: Importance of redundant PAF connections. (a) Two different people are wrongly merged due to a wrong neck-nose connection. (b) The higher confidence of the right ear-shoulder connection avoids the wrong nose-neck link.

图7. 冗余PAF连接的重要性。(a)由于错误的脖子-鼻子连接，两个人被错误的混合到了一起；(b)右耳朵-肩膀的高置信度连接，避免了错误的鼻子-脖子连接。

## 4 OpenPose

A growing number of computer vision and machine learning applications require 2D human pose estimation as an input for their systems [56], [57], [58], [59], [60], [61], [62], [63]. To help the research community boost their work, we have publicly released OpenPose [4], the first real-time multi-person system to jointly detect human body, foot, hand, and facial keypoints (in total 135 keypoints) on single images. See Fig. 8 for an example of the whole system.

越来越多的计算机视觉和机器学习应用需要2D人体姿态估计作为其系统输入[56], [57], [58], [59], [60], [61], [62], [63]。为促进研究，我们公开发布了OpenPose[4]，第一个在单幅图像上实时多人同时检测人体、脚部、手部和面部关键点系统（共计135个关键点）。见图8的整体系统示例。

Fig. 8: Output of OpenPose, detecting body, foot, hand, and facial keypoints in real-time. OpenPose is robust against occlusions including during human-object interaction.

图8.OpenPose的输出，实时检测人体、脚部、手部和面部关键点。OpenPose对遮挡很稳健，包括人与物体互动时的遮挡。

### 4.1 System

Available 2D body pose estimation libraries, such as Mask R-CNN [5] or Alpha-Pose [6], require their users to implement most of the pipeline, their own frame reader (e.g., video, images, or camera streaming), a display to visualize the results, output file generation with the results (e.g., JSON or XML files), etc. In addition, existing facial and body keypoint detectors are not combined, requiring a different library for each purpose. OpenPose overcome all of these problems. It can run on different platforms, including Ubuntu, Windows, Mac OSX, and embedded systems (e.g., Nvidia Tegra TX2). It also provides support for different hardware, such as CUDA GPUs, OpenCL GPUs, and CPU-only devices. The user can select an input between images, video, webcam, and IP camera streaming. He can also select whether to display the results or save them on disk, enable or disable each detector (body, foot, face, and hand), enable pixel coordinate normalization, control how many GPUs to use, skip frames for a faster processing, etc.

可用的2D身体姿态估计库，如Mask R-CNN[5]或Alpha-Pose[6]，需要其用户实现大部分流程，自己的帧读取器（如视频、图像或摄像头流媒体），对结果进行可视化的显示，生成输出结果的文件（如JSON或XML文件）等等。另外，现有的面部和身体关键点检测器并没有结合到一起，每个目的需要不同的库。OpenPose克服了所有这些问题，可以运行在不同的平台上，包括Ubuntu、Windows、Max OSX和嵌入式系统（如Nvidia Tegra TX2）。还支持不同的硬件平台，比如CUDA GPUs、OpenCL GPUs和只有CPU的设备。用户可以选择输入方式，如图像、视频、摄像头或IP摄像头流媒体，还可以选择是显示结果或存储到磁盘上，是否启用每个检测器（身体、脚部、脸部、手部），启用像素坐标归一化，控制使用多少个GPU，跳过一些帧以实现快速处理，等等。

OpenPose consists of three different blocks: (a) body+foot detection, (b) hand detection [64], and (c) face detection. The core block is the combined body+foot keypoint detector (detailed in Section 4.2). It can alternatively use the original body-only detectors [3] trained on COCO and MPII datasets. Based on the output of the body detector, facial bounding box proposals can roughly be estimated from some body part locations, in particular ears, eyes, nose, and neck. Analogously, the hand bounding box proposals are generated with the arm keypoints. This methodology inherits the problems of top-down approaches discussed in Section 1. The hand keypoint detector algorithm is explained in further detail in [64], while the facial keypoint detector has been trained in the same fashion as that of the hand keypoint detector. The library also includes 3D realtime single-person keypoint detection, able to predict 3D pose estimation out of multiple synchronized camera views. It performs 3D triangulation with non-linear Levenberg-Marquardt refinement [65].

OpenPose包括三种不同的模块：(a)身体+脚部检测，(b)手部检测[64]，(c)脸部检测。核心模块是身体+脚部关键点联合检测器（详见4.2节），也可以或者使用原始的在COCO和MPII数据集上训练的身体检测器。基于身体检测器的输出，脸部边界框候选可以粗略从一些身体部位位置中估计出来，特别是耳朵，眼睛，鼻子和脖子。类似的，手部边界框候选可以从胳膊关键点生成。这种方法继承了第一节中自上而下方法的问题。手部关键点检测器算法详见[64]，脸部关键点检测器的训练方式与手部关键点检测器一样。这个库还包括了3D实时单人关键点检测，可以从多个同步过的摄像头视角预测3D姿态估计。还可以用非线性Levenberg-Marquardt精炼[65]来进行3D三角测量。

The inference time of OpenPose outperforms all state-of-the-art methods, while preserving high-quality results. It is able to run at about 22 FPS in a machine with a Nvidia GTX 1080 Ti while preserving high accuracy (more details in Section 5.3). OpenPose has already been used by the research community for many vision and robotics topics, such as person re-identification [56], GAN-based video retargeting of human faces [57] and bodies [58], Human-Computer Interaction [59], 3D human pose estimation [60], and 3D human mesh model generation [61]. In addition, the OpenCV library [66] has included OpenPose and our PAF-based network architecture within its Deep Neural Network (DNN) module.

OpenPose的推理时间在所有目前最好的方法中是最优的，结果也是高质量的。在Nvidia GTX 1080 Ti上运行速度约为22FPS，同时保持很高的准确率（详见5.3节）。研究团体已经将OpenPose用于很多视觉和机器人领域，如行人重识别[56]，基于GAN的视频人脸重新确定目标[57]和身体[58]，人机互动[59]，3D人体姿态估计[60]，3D人体网格模型生成[61]。另外，OpenCV库[66]已经包括了OpenPose和我们基于PAF的网络架构，在其DNN模块中。

### 4.2 Extended Foot Keypoint Detection 拓展脚部关键点检测

Existing human pose datasets ([67], [68]) contain limited body part types. The MPII dataset [67] annotates ankles, knees, hips, shoulders, elbows, wrists, necks, torsos, and head tops, while COCO [68] also includes some facial keypoints. For both of these datasets, foot annotations are limited to ankle position only. However, graphics applications such as avatar retargeting or 3D human shape reconstruction ([61], [69]) require foot keypoints such as big toe and heel. Without foot keypoint information, these approaches suffer from problems such as the candy wrapper effect, floor penetration, and foot skate. To address these issues, a small subset of about 15K human foot instances has been labeled using the Clickworker platform [70]. The dataset is obtained out of the over 100K person annotation instances available in the COCO dataset. It is split up with 14K annotations from the COCO training set and 545 from the validation set. A total of 6 foot keypoints have been labeled (see Fig. 9a). We consider the 3D coordinate of the foot keypoints rather than the surface position. For instance, for the exact toe positions, we label the area between the connection of the nail and skin, and also take depth into consideration by labeling the center of the toe rather than the surface.

现有的人体姿态数据集[67,68]包括身体部位类型有限。MPII数据集[67]标注了脚踝、膝盖、臀部、肩膀、肘部、手腕、脖子、躯干和头部，而COCO[68]还包括了一些脸部关键点。这两个数据集中，脚部的标注就只有脚踝的位置。但是，图形应用比如化身重新确定目标或3D人体形状重建([61,69])需要脚部关键点比如大脚趾和脚后跟。没有脚部关键点信息，这些方法存在一些问题，如candy wrapper effect, floor penetration, and foot skate. 为解决这些问题，我们使用Clickworker平台[70]标注了一个小型脚部数据集，包括大约15k个脚部实例。这个数据集不在COCO数据集的100k个人体标注之中，需要另外获得。数据集从COCO训练集中分割出14K个标注，验证集中分割出545个。标注了共6个脚部关键点（见图9a）。我们考虑脚部关键点的3D坐标，而不是其表面位置。比如，对于精确的脚趾位置，我们标注了脚趾甲和皮肤之间的连接的区域，还考虑了深度信息，标注的是脚趾的中间，而不是表面。

Fig. 9: Foot keypoint analysis. (a) Foot keypoint annotations, consisting of big toes, small toes, and heels. (b) Body-only model example at which right ankle is not properly estimated. (c) Analogous body+foot model example, the foot information helped predict the right ankle location.

Using our dataset, we train a foot keypoint detection algorithm. A näive foot keypoint detector could have been built by using a body keypoint detector to generate foot bounding box proposals, and then training a foot detector on top of it. However, this method suffers from the top-down problems stated in Section 1. Instead, the same architecture previously described for body estimation is trained to predict both the body and foot locations. Fig. 10 shows the keypoint distribution for the three datasets (COCO, MPII, and COCO+foot). The body+foot model also incorporates an interpolated point between the hips to allow the connection of both legs even when the upper torso is occluded or out of the image. We find evidence that foot keypoint detection implicitly helps the network to more accurately predict some body keypoints, in particular leg keypoints, such as ankle locations. Fig. 9b shows an example where the body-only network was not able to predict ankle location. By including foot keypoints during training, while maintaining the same body annotations, the algorithm can properly predict the ankle location in Fig. 9c. We quantitatively analyze the accuracy difference in Section 5.4.

使用我们的数据集，我们训练了一个脚部关键点检测算法。一个简单的脚部关键点检测器可以使用身体关键点检测器来生成脚部边界框候选，然后在这之上训练一个脚部检测器。但是，这种方法存在第1部分所说的自上而下的问题。我们采用了与身体估计相同的架构来训练预测身体+脚部位置。图10所示的是三种数据集(COCO, MPII, COCO+foot)的关键点分布。身体+脚部模型还包含了两个臀部之间插值的点，这样当上部躯干被遮挡，或不在图像中时，可以保持两条腿之间的连接平衡。我们发现证据，脚部关键点检测隐式的帮助网络更准确的预测一些身体关键点，尤其是腿部的关键点，比如脚踝的位置。图9b就是一个例子，只有身体检测的网络没能预测到脚踝的位置；通过在训练时候包含脚部关键点，保持同样的身体标注，算法可以正确的预测脚踝的位置，如图9c所示。我们在5.4节中量化分析准确率的差异。

Fig. 10: Keypoint annotation configuration for the 3 datasets. (a) MPII (b) COCO (c) COCO+Foot

## 5 Datasets and evaluations 数据集和评估

We evaluate our method on three benchmarks for multi-person pose estimation: (1) MPII human multi-person dataset [67], which consists of 3844 training and 1758 testing groups of multiple interacting individuals in highly articulated poses with 14 body parts; (2) COCO keypoint challenge dataset [68], which requires simultaneously detecting people and localizing 17 keypoints (body parts) in each person (including 12 human body parts and 5 facial keypoints); (3) our foot dataset, which is a subset of 15K annotations out of the COCO keypoint dataset. These datasets collect images in diverse scenarios that contain many real-world challenges such as crowding, scale variation, occlusion, and contact. Our approach placed first at the inaugural COCO 2016 keypoints challenge [71], and significantly exceeded the previous state-of-the-art results on the MPII multi-person benchmark. We also provide runtime analysis comparison against Mask R-CNN and Alpha-Pose to quantify the efficiency of the system and analyze the main failure cases.

我们在三个多人姿态估计的基准测试中评估我们的方法：(1)MPII人体多人数据集[67]，由3844个训练群以及1758个测试群组成，每个群都是多个交互的个体，每个个体是由14个身体部位形成的高度铰接的姿态；(2)COCO关键点挑战数据集[68]，需要同时检测人体并定位每个人的17个关键点（身体部位，包括12个人体身体部位和5个脸部关键点）；(3)我们的脚部数据集，是COCO关键点数据集的子集，包含15K个标注。这些数据集收集了各种场景的图像，包含了众多真实世界的挑战，比如群聚、尺度变化、遮挡以及接触。我们的方法在第一次COCO 2016关键点挑战[71]中排名第一，明显超出之前在MPII多人基准测试中最好的结果。我们还给出了运行时间分析，与Mask R-CNN和Alpha-Pose进行比较，以量化系统的效率，分析主要的错误情景。

### 5.1 Results on the MPII Multi-Person Dataset

For comparison on the MPII dataset, we use the toolkit [1] to measure mean Average Precision (mAP) of all body parts following the “PCKh” metric from [67]. Table 1 compares mAP performance between our method and other approaches on the official MPII testing sets. We also compare the average inference/optimization time per image in seconds. For the 288 images subset, our method outperforms previous state-of-the-art bottom-up methods [2] by 8.5% mAP. Remarkably, our inference time is 6 orders of magnitude less. We report a more detailed runtime analysis in Section 5.3. For the entire MPII testing set, our method without scale search already outperforms previous state-of-the-art methods by a large margin, i.e., 13% absolute increase on mAP. Using a 3 scale search (×0.7 , ×1 and ×1.3) further increases the performance to 75.6% mAP. The mAP comparison with previous bottom-up approaches indicate the effectiveness of our novel feature representation, PAFs, to associate body parts. Based on the tree structure, our greedy parsing method achieves better accuracy than a graphcut optimization formula based on a fully connected graph structure [1], [2].

为在MPII数据集上进行比较，我们使用了[1]中的工具箱来衡量所有身体部位的mAP，遵循的是[67]中的PCKh(Percentage of Correct Keypoints w.r.t. head)度量。表1比较了我们的方法与其他方法在官方MPII测试集上的mAP性能。我们还比较了每幅图像的平均推理/优化时间（以秒为单位）。对于288图像的子集，我们的方法超过了之前最好的自下而上方法[2] 8.5% mAP。令人印象深刻的是，我们的推理时间低了6个数量级。我们在5.3节中详细分析了运行时间。对于整个MPII测试集，我们的方法在没有尺度搜索时已经超过了之前最好的方法一大截，即，mAP上13%的绝对幅度提升。使用了3尺度搜索(×0.7 , ×1 and ×1.3)进一步提升性能至75.6% mAP。与之前的自下而上方法的mAP比较，说明了我们新的特征表示PAFs可以有效的关联身体部位。基于树结构，我们的贪婪解析方法，比基于全连接图结构[1,2]的图割优化，取得了更好的准确率。

TABLE 1: Results on the MPII dataset. Top: Comparison results on the testing subset defined in [1]. Middle: Comparison results on the whole testing set. Testing without scale search is denoted as “(one scale)”.

Method | Hea | Sho | Elb | Wri | Hip | Kne | Ank | mAP | s/image
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Deepcut [1] | 73.4 | 71.8 | 57.9 | 39.9 | 56.7 | 44.0 | 32.0 | 54.1 | 57995
Iqbal et al. [41] | 70.0 | 65.2 | 56.4 | 46.1 | 52.7 | 47.9 | 44.5 | 54.7 | 10
DeeperCut [2] | 87.9 | 84.0 | 71.9 | 63.9 | 68.8 | 63.8 | 58.1 | 71.2 | 230
Newell et al. [48] | 91.5 | 87.2 | 75.9 | 65.4 | 72.2 | 67.0 | 62.1 | 74.5 | -
ArtTrack [47] | 92.2 | 91.3 | 80.8 | 71.4 | 79.1 | 72.6 | 67.8 | 79.3 | 0.005
Fang et al. [6] | 89.3 | 88.1 | 80.7 | 75.5 | 73.7 | 76.7 | 70.0 | 79.1 | -
Ours | 92.9 | 91.3 | 82.3 | 72.6 | 76.0 | 70.9 | 66.8 | 79.0 | 0.005
DeeperCut [2] | 78.4 | 72.5 | 60.2 | 51.0 | 57.2 | 52.0 | 45.4 | 59.5 | 485
Iqbal et al. [41] | 58.4 | 53.9 | 44.5 | 35.0 | 42.2 | 36.7 | 31.1 | 43.1 | 10
Levinko et al. [72] | 89.8 | 85.2 | 71.8 | 59.6 | 71.1 | 63.0 | 53.5 | 70.6 | -
ArtTrack [47] | 88.8 | 87.0 | 75.9 | 64.9 | 74.2 | 68.8 | 60.5 | 74.3 | 0.005
Fang et al. [6] | 88.4 | 86.5 | 78.6 | 70.4 | 74.4 | 73.0 | 65.8 | 76.7 | -
Newell et al. [48] | 92.1 | 89.3 | 78.9 | 69.8 | 76.2 | 71.6 | 64.7 | 77.5 | -
Fieraru et al. [73] | 91.8 | 89.5 | 80.4 | 69.6 | 77.3 | 71.7 | 65.5 | 78.0 | -
Ours (one scale) | 89.0 | 84.9 | 74.9 | 64.2 | 71.0 | 65.6 | 58.1 | 72.5 | 0.005
Ours | 91.2 | 87.6 | 77.7 | 66.8 | 75.4 | 68.9 | 61.7 | 75.6 | 0.005

In Table 2, we show comparison results for the different skeleton structures shown in Fig. 6. We created a custom validation set consisting of 343 images from the original MPII training set. We train our model based on a fully connected graph, and compare results by selecting all edges (Fig. 6b, approximately solved by Integer Linear Programming), and minimal tree edges (Fig. 6c, approximately solved by Integer Linear Programming, and Fig. 6d, solved by the greedy algorithm presented in this paper). Both methods yield similar results, demonstrating that it is sufficient to use minimal edges. We trained our final model to only learn the minimal edges to fully utilize the network capacity, denoted as Fig. 6d (sep). This approach outperforms Fig. 6c and even Fig. 6b, while maintaining efficiency. The fewer number of part association channels (13 edges of a tree vs 91 edges of a graph) needed facilitates the training convergence.

表2中，我们展示了图6中使用不同梗概结构得到的结果对比。我们用原始MPII训练集中的343幅图像创建了一个定制的验证集。我们基于全连接图训练了模型，比较了几种不同的结果，第一个是选择所有的边（如图6b，使用整数线性规划近似求解），第二个是最小树状边（如图6c，使用整数线性规划近似求解，和图6d，使用本文的贪婪算法求解）。所有方法都得到了近似的结果，证明了使用最小边是足够的。我们训练最终模型时，只学习了最小边以充分利用网络容量，用图6d(sep)表示。这种方法超过了图6c甚至是图6d，同时保持了效率。需要的部位关联通道较少（树状结构的13边 vs 图结构的91边），有利于训练快速收敛。

TABLE 2: Comparison of different structures on our custom validation set.

Method | Hea | Sho | Elb | Wri | Hip | Kne | Ank | mAP | s/image
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Fig. 6b | 91.8 | 90.8 | 80.6 | 69.5 | 78.9 | 71.4 | 63.8 | 78.3 | 362
Fig. 6c | 92.2 | 90.8 | 80.2 | 69.2 | 78.5 | 70.7 | 62.6 | 77.6 | 43
Fig. 6d | 92.0 | 90.7 | 80.0 | 69.4 | 78.4 | 70.1 | 62.3 | 77.4 | 0.005
Fig. 6d (sep) | 92.4 | 90.4 | 80.9 | 70.8 | 79.5 | 73.1 | 66.5 | 79.1 | 0.005

Fig. 11a shows an ablation analysis on our validation set. For the threshold of PCKh-0.5 [67], the accuracy of our PAF method is 2.9% higher than one-midpoint and 2.3% higher than two intermediate points, generally outperforming the method of midpoint representation. The PAFs, which encode both position and orientation information of human limbs, are better able to distinguish the common cross-over cases, e.g., overlapping arms. Training with masks of unlabeled persons further improves the performance by 2.3% because it avoids penalizing the true positive prediction in the loss during training. If we use the ground-truth keypoint location with our parsing algorithm, we can obtain a mAP of 88.3%. In Fig. 11a, the mAP obtained using our parsing with GT detection is constant across different PCKh thresholds due to no localization error. Using GT connection with our keypoint detection achieves a mAP of 81.6%. It is notable that our parsing algorithm based on PAFs achieves a similar mAP as when based on GT connections ( 79.4% vs 81.6% ). This indicates parsing based on PAFs is quite robust in associating correct part detections. Fig. 11b shows a comparison of performance across stages. The mAP increases monotonically with the iterative refinement framework. Fig. 4 shows the qualitative improvement of the predictions over stages.

图11是在我们验证集上的分离试验分析。对于PCKh-0.5的阈值[67]，我们PAF方法的准确度比一个中间点的准确度高2.9%，比两个中间点的高2.3%，总体上比中间点表示的方法要好。PAFs编码了人体肢体的位置和方向信息，可以更好的区分普通的交叉情况，如重叠的手臂。使用未标注的人的掩模进行训练，可以进一步改进准确率2.3%，因为这避免了在训练中损失函数上惩罚真阳性预测。如果我们使用真值关键点位置和我们的解析算法，我们可以得到88.3% mAP。如图11a，使用我们的解析和真值检测得到的mAP，在不同的PCKh阈值下是常数，因为没有定位错误。使用真值连接和我们的关键点检测，得到的mAP为81.6%。值得注意的是，我们基于PAFs的解析算法在基于真值连接时得到了类似的mAP( 79.4% vs 81.6% )。这说明，基于PAFs的解析在关联正确的部位检测时非常稳健。图11b展示的是不同阶段的性能比较，mAP随着迭代优化框架单调增加。图4是不同阶段下预测的定性改进。

Fig. 11: mAP curves over different PCKh thresholds on MPII validation set. (a) mAP curves of self-comparison experiments. (b) mAP curves of PAFs across stages.

### 5.2 Results on the COCO Keypoints Challenge

The COCO training set consists of over 100K person instances labeled with over 1 million keypoints. The testing set contains “test-challenge” and “test-dev”subsets, which have roughly 20K images each. The COCO evaluation defines the object keypoint similarity (OKS) and uses the mean average precision (AP) over 10 OKS thresholds as the main competition metric [71]. The OKS plays the same role as the IoU in object detection. It is calculated from the scale of the person and the distance between predicted and GT points. Table 3 shows results from top teams in the challenge. It is noteworthy that our method has a higher drop in accuracy when considering only people of higher scales ($AP^L$).

COCO训练集包括超过100K个人体实例，共标记了超过1M关键点。测试集包括了test-challenge和test-dev两个子集，每个子集约有20K幅图像。COCO评估标准定义了目标关键点相似性(OKS)并使用10个OKS阈值上的mAP作为主要竞争度量标准[71]。OKS扮演的角色与目标识别中的IoU一样，是由人体的尺度和预测与真值点间的距离计算得到的。表3所示的是挑战中最好的队伍的结果。值得注意的是，我们的方法在只考虑较高尺度人的时候$AP^L$准确率下降较高。

TABLE 3: Results on the COCO test-dev dataset. Top: top-down results. Bottom: bottom-up results (top methods only). $AP^{50}$ is for OKS = 0.5 , $AP^L$ is for large scale persons.

Team | AP | $AP^{50}$ | $AP^{75}$ | $AP^M$ | $AP^L$
--- | --- | --- | --- | --- | ---
Megvii [43] | 78.1 | 94.1 | 85.9 | 74.5 | 83.3
MRSA [44] | 76.5 | 92.4 | 84.0 | 73.0 | 82.7
The Sea Monsters | 75.9 | 92.1 | 83.0 | 71.7 | 82.1
ByteDance-SEU | 74.2 | 91.8 | 81.9 | 70.6 | 80.2
fadivugibs | 74.0 | 91.3 | 81.5 | 70.6 | 80.1
G-RMI [42] | 71.0 | 87.9 | 77.7 | 69.0 | 75.2
Mask R-CNN [5] | 69.2 | 90.4 | 76.0 | 64.9 | 76.3

Team | AP | $AP^{50}$ | $AP^{75}$ | $AP^M$ | $AP^L$
--- | --- | --- | --- | --- | ---
METU [50] | 70.5 | 87.7 | 77.2 | 66.1 | 77.3
TFMAN | 70.2 | 89.2 | 77.0 | 65.6 | 76.3
PersonLab [49] | 68.7 | 89.0 | 75.4 | 64.1 | 75.5
Associative Emb. [48] | 65.5 | 86.8 | 72.3 | 60.6 | 72.6
Ours | 64.2 | 86.2 | 70.1 | 61.0 | 68.8
Ours [3] | 61.8 | 84.9 | 67.5 | 57.1 | 68.2

In Table 4, we report self-comparisons on the COCO validation set. If we use the GT bounding box and a single person CPM [20], we can achieve an upper-bound for the top-down approach using CPM, which is 62.7% AP. If we use the state-of-the-art object detector, Single Shot MultiBox Detector (SSD) [74], the performance drops 10%. This comparison indicates the performance of top-down approaches rely heavily on the person detector. In contrast, our original bottom-up method achieves 58.4% AP. If we refine the results by applying a single person CPM on each rescaled region of the estimated persons parsed by our method, we gain a 2.6% overall AP increase. We only update estimations on predictions in which both methods roughly agree, resulting in improved precision and recall. The new architecture without CPM refinement is approximately 7% more accurate than the original approach, while increasing the speed by 45%.

在表4中，我们给出了在COCO验证集上的自对比结果。如果我们使用真值边界框和单人CPM[20]，我们可以得到使用CPM的自上而下方法的上限，即62.7%。如果我们使用目前最好的目标检测器SSD[74]，性能下降10%。这种比较说明自上而下的方法严重依赖人体检测器。对比起来，我们初始的自下而上的方法得到了58.4% AP，如果我们对结果进行精炼，在我们的方法解析得到的人体估计区域中，使用单人CPM，我们得到的总体AP提升为2.6%。两种方法大致都一致的预测，我们才更新其估计，得到了改进的精确率和召回率。没有CPM优化的新架构，比原始方法提升了约7%的精度，而速度提升了45%。

TABLE 4: Self-comparison experiments on the COCO validation set. Our new body+foot model outperforms the original work in [3] by 6.8%.

Method | AP | $AP^{50}$ | $AP^{75}$ | $AP^M$ | $AP^L$
--- | --- | --- | --- | --- | ---
GT Bbox + CPM [2] | 62.7 | 86.0 | 69.3 | 58.5 | 70.6
SSD [74] + CPM [2] | 52.7 | 71.1 | 57.2 | 47.0 | 64.2
Ours [3] | 58.4 | 81.5 | 62.6 | 54.4 | 65.1
+CPM refinement | 61.0 | 84.9 | 67.5 | 56.3 | 69.3
Ours | 65.1 | 85.0 | 71.2 | 62.0 | 70.1

We analyze the effect of PAF refinement over confidence map estimation in Table 5. We fix the computation to a maximum of 6 stages, distributed differently across the PAF and confidence map branches. We can extract 3 conclusions from this experiment. First, PAF requires a higher number of stages to converge and benefits more from refinement stages. Second, increasing the number of PAF channels mainly improves the number of true positives, even though they might not be too accurate (higher $AP^{50}$). However, increasing the number of confidence map channels further improves the localization accuracy (higher $AP^{75}$). Third, we prove that the accuracy of the part confidence maps highly increases when using PAF as a prior, while the opposite results in a 4% absolute accuracy decrease. Even the model with only 4 stages (3 PAF - 1 CM) is more accurate than the computationally more expensive 6-stage model that first predicts confidence maps (3 CM - 3 PAF). Some other additions that further increased the accuracy of the new models with respect to the original work are PReLU over ReLU layers and Adam optimization instead of SGD with momentum. Differently to [3], we do not refine the current approach with CPM [20] to avoid harming the speed.

我们分析了PAF精炼对于置信度图估计的效果，如表5。我们固定计算量为最多6个阶段，分布在PAF和置信度图中。我们从试验中可以得到3个结论。第一，PAF需要较多的阶段数收敛，从精炼阶段受益较多。第二，增加PAF通道数量主要改进了真阳性预测的数量，但是还是不会太精确（$AP^{50}$更高）。但是，增加置信度图的通道数改进了定位准确率（$AP^{75}$更高）。第三，我们证明了部位置信度图的准确率在使用PAF作为先验时增加很多，而反过来会使准确率降低4%绝对值。即使模型只使用4个阶段(3PAF-1CM)，也比使用6阶段但首先预测置信度图(3CM-3PAF)的方式更准确。还有一些措施可以提升模型准确率，包括使用PReLU层而不使用ReLU层，使用Adam优化而不使用带有动量的SGD。与[3]不同的是，我们没有对现在的方法使用CPM[20]进行精炼，以免影响速度。

TABLE 5: Self-comparison experiments on the COCO validation set. CM refers to confidence map, while the numbers express the number of estimation stages for PAF and CM. Stages refers to the number of PAF and CM stages. Reducing the number of stages increases the runtime performance.

Method | AP | $AP^{50}$ | $AP^{75}$ | $AP^M$ | $AP^L$ | Stages
--- | --- | --- | --- | --- | --- | ---
5 PAF - 1 CM | 65.1 | 85.0 | 71.2 | 62.0 | 70.1 | 6
4 PAF - 2 CM | 65.2 | 85.3 | 71.4 | 62.3 | 70.1 | 6
3 PAF - 3 CM | 65.0 | 85.1 | 71.2 | 62.4 | 69.4 | 6
4 PAF - 1 CM | 64.8 | 85.3 | 70.9 | 61.9 | 69.6 | 5
3 PAF - 1 CM | 64.6 | 84.8 | 70.6 | 61.8 | 69.5 | 4
3 CM - 3 PAF | 61.0 | 83.9 | 65.7 | 58.5 | 65.3 | 6

### 5.3 Inference Runtime Analysis 推理运行时间分析

There are only 3 state-of-the-art, well-maintained, and widely-used multi-person pose estimation libraries, OpenPose [4], based on this work, Mask R-CNN [5], and Alpha-Pose [6]. We analyze the inference runtime performance of the 3 methods in Fig. 12. Megvii (Face++) [43] and MSRA [44] GitHub repositories do not include the person detector they use and only provide pose estimation results given a cropped person. Thus, we cannot know their exact runtime performance and have been excluded from this analysis. Mask R-CNN is only compatible with Nvidia graphics cards, so we perform the analysis on a Nvidia-powered system. As top-down approaches, the inference times of Mask R-CNN, Alpha-Pose, Megvii, and MSRA are roughly proportional to the number of people in the image. To be more precise, they are proportional to the number of proposals that their person detectors extract. In contrast, the inference time of our bottom-up approach is invariant to the number of people in the image. The runtime of OpenPose consists of two major parts: (1) CNN processing time whose complexity is O(1), constant with varying number of people; (2) multi-person parsing time, whose complexity is O($n^2$), where n represents the number of people. However, the parsing time is two orders of magnitude less than the CNN processing time. For instance, the parsing takes 0.58 ms for 9 people while the CNN takes 41 ms.

现在只有3种目前最好的，维护良好的，普遍使用的多人姿态估计库，基于本文工作的OpenPose[4]，Mask R-CNN[5]和Alpha-Pose[6]。我们在图12中分析了这三种方法的推理运行时间表现。Megvii(Face++)[43]和MSRA[44]的Github repo没有包括其使用的人体检测器，只给出了剪切出来的人体的姿态估计结果。所以我们不能得知其精确的运行时间性能，在此次分析中就被排除在外。Mask R-CNN只与Nvidia显卡兼容，所以我们的分析在Nvidia系统上进行。作为自上而下的系统，Mask R-CNN, Alpha-Pose, Megvii和MSRA的推理时间大致与图像中的人数成正比。确切的说，与使用的人体检测器提取出来的候选数量成正比。对比起来，我们的自下而上方法的推理时间与图像中的人数无关。OpenPose的运行时间主要由两块组成：(1)CNN的处理时间，与人数无关；(2)多人解析的时间，其复杂度为O($n^2$)，其中n为人数。但是，解析时间比CNN处理时间少了2个数量级。例如，CNN耗时41ms，而9人的解析时间只耗费0.58ms。

Fig. 12: Inference time comparison between OpenPose, Mask R-CNN, and Alpha-Pose (fast Pytorch version). While OpenPose inference time is invariant, Mask R-CNN and Alpha-Pose runtimes grow linearly with the number of people. Testing with and without scale search is denoted as “max accuracy” and “1 scale”, respectively. This analysis was performed using the same images for each algorithm and a batch size of 1. Each analysis was repeated 1000 times and then averaged. This was all performed on a system with a Nvidia 1080 Ti and CUDA 8.

In Table 6, we analyze the difference in inference time between the models released in OpenPose, i.e., the MPII and COCO models from [3] and the new body+foot model. Our new combined model is not only more accurate, but is also 45% faster than the original model when using the GPU version. Interestingly, the runtime for the CPU version is 5x slower compared to that of the original model. The new architecture consists of many more layers, which requires a higher amount of memory, while the number of operations is significantly fewer. Graphic cards seem to benefit more from the reduction in number of operations, while the CPU version seems to be significantly slower due to the higher memory requirements. OpenCL and CUDA performance cannot be directly compared to each other, as they require different hardware, in particular, different GPU brands.

在表6中，我们分析了OpenPose中发布的模型的推理时间的区别，即[3]中的MPII和COCO模型，和新的身体+脚部模型。我们新的组合模型不仅更准确，而且在使用GPU时比原始模型快了45%。有趣的是，CPU版的运行时间比原始模型慢了5倍。新的架构包括了更多的层，这需要大量存储空间，而运算量则明显更少。显卡似乎从运算量减少中受益更多，而CPU版则由于更高的内存需求变得更慢。OpenCL和CUDA的性能不能直接相互对比，因为需要不同的硬件，尤其是不同的GPU品牌。

TABLE 6: Runtime difference between the 3 models released in OpenPose with CUDA and CPU-only versions, running in a NVIDIA GeForce GTX-1080 Ti GPU and a i7-6850K CPU. MPII and COCO models refer to our work in [3].

Method | CUDA | CPU-only
--- | --- | ---
Original MPII model | 75 ms | 2309 ms
Original COCO model | 77 ms | 2407 ms
Body+foot model | 42 ms | 10396 ms

### 5.4 Results on the Foot Keypoint Dataset

To evaluate the foot keypoint detection results obtained using our foot keypoint dataset, we calculate the mean average precision and recall over 10 OKS, as done in the COCO evaluation metric. There are only minor differences between the combined and body-only approaches. In the combined training scheme, there exist two separate and completely independent datasets. The larger of the two datasets consists of the body annotations while the smaller set contains both body and foot annotations. The same batch size used for the body-only training is used for the combined training. Nevertheless, it contains only annotations from one dataset at a time. A probability ratio is defined to select the dataset from which to pick each batch. A higher probability is assigned to select a batch from the larger dataset, as the number of annotations and diversity is much higher. Foot keypoints are masked out during the back-propagation pass of the body-only dataset to avoid harming the net with non-labeled data. In addition, body annotations are also masked out from the foot dataset. Keeping these annotations yields a small drop in accuracy, probably due to overfitting, as those samples are repeated in both datasets.

为评估使用我们的脚部关键点数据集得到的脚部关键点检测结果，我们计算在10个OKS上的mAP和召回率，这和COCO评估标准一样。组合检测方法和只有身体的检测方法只有细微的差别。在组合训练方案中，存在两种分离的和完全独立的数据集。两个数据集中较大的那个，包括身体标注，而较小的那个包含身体和脚部的标注。同样的两种训练的batch size相同。然而，同时只包括一个数据集的标注。定义了一个概率比，来确定从哪个数据集中选择每个批次。较大数据集对应的较高的概率，因为其标注数量和多样性更高。只有身体的数据集训练时，反向传播过程将脚部关键点遮盖掉了，以免损害没有标注的数据。另外。身体标注也从脚部数据集中遮盖掉了。保持这些标注会得到准确率的小幅度降低，可能是因为过拟合，因为这些样本在两个数据集中有重复。

Table 7 shows the foot keypoint accuracy for the validation set. Here, the network tries to implicitly predict occluded foot keypoints, while the ground-truth data only annotates foot keypoints if they are not occluded, leading to a high recall but low precision. While foot keypoint annotations are only labeled when they are visible, COCO body annotations include occluded parts that can be easily guessed. We intend to follow COCO format and also label occluded foot keypoints that can be estimated by humans, balancing the recall-precision difference. Qualitatively, we find a higher amount of jitter and number of detection errors compared to body keypoint prediction. We believe 14K training annotations are not a sufficient number to train a robust foot detector, considering that over 100K instances are used for the body keypoint dataset. Rather than using the whole batch with either only foot or body annotations, we also tried using a mixed batch where samples from both datasets (either COCO or COCO+foot) could be fed to the same batch, maintaining the same probability ratio. However, the network accuracy was slightly reduced. By mixing the datasets with an unbalanced ratio, we effectively assign a very small batch size for foot, hindering foot convergence.

表7所示的是验证集上的脚部关键点准确率。这里，网络试图隐式的预测遮挡的脚部关键点，而真值数据只标注了未遮挡的脚部关键点，这导致了高召回率低准确率。脚部关键点只有在可见的情况下进行了标注，而COCO身体则包括了容易猜测的被遮挡的部位。我们倾向于遵循COCO的格式，同时标注遮挡的但可以由人体估计的脚部关键点，以平衡召回率-精确度的区别。定性的说，与身体关键点预测相比，我们发现了更多数量的抖动和一些检测错误。我们相信14K训练标注不足以训练一个稳健的脚部检测器，因为考虑到身体关键点数据集中有超过100K个实例。我们还尝试了同一批次中混合使用两种数据集的样本（COCO或COCO+foot），保持同样的概率比。但是，网络的准确率稍有降低。

TABLE 7: Foot keypoint analysis on the foot validation set.

Method | AP | AR | $AP^{75}$ | $AR^{75}$
--- | --- | --- | --- | ---
Body+foot model (5 PAF - 1 CM) | 25.7 | 86.0 | 26.8 | 89.2

In Table 8, we show that there is almost no accuracy difference in the COCO test-dev set with respect to the same network architecture trained only with body annotations. We compared the model consisting of 5 PAF and 1 confidence map stages, with a 95% probability of picking a batch from the COCO body-only dataset, and 5% of choosing from the body+foot dataset. There is no architecture difference compared to the body-only model other than the increase in the number of outputs to include the foot CM and PAFs.

表8中，我们说明，同样的网络架构进行只有身体标注的训练，在COCO test-dev集中几乎没有准确率上的差别。我们比较的模型包括5个PAF阶段1个置信度图阶段，95%的概率从COCO身体数据集中去批次，5%的概率从身体+脚部数据集中取批次。与只有身体的模型相比，没有任何架构上的差异，除了输出数量上的增加，以包括脚部的CM和PAFs。

TABLE 8: Self-comparison experiments for body on the COCO test-dev validation set. Foot keypoints are predicted but ignored for the evaluation.

Method | AP | $AP^{50}$ | $AP^{75}$ | $AP^M$ | $AP^L$
--- | --- | --- | --- | --- | ---
Body-only (5 PAF - 1 CM) | 65.0 | 84.8 | 70.8 | 61.9 | 69.9
Body+foot (5 PAF - 1 CM) | 65.1 | 85.0 | 71.2 | 62.0 | 70.1

### 5.5 Vehicle Pose Estimation 车辆姿态估计

Our approach is not limited to human body or foot keypoints, but can be generalized to any keypoint annotation task. To demonstrate this, we have run the same network architecture for the task of vehicle keypoint detection [75]. Once again, we use mean average precision over 10 OKS for the evaluation. The results are shown in Table 9. Both the average precision and recall are higher than in the body keypoint task, mainly because we are using a smaller and simpler dataset. This initial dataset consists of image annotations from 19 different cameras. We have used the first 18 camera frames as a training set, and the camera frames from the last camera as a validation set. No variations in the model architecture or training parameters have been made. We show qualitative results in Fig. 13.

我们的方法不止可以用于身体或脚部关键点，也可以泛化到任何关键点标注任务。为证明这个结论，我们对车辆关键点检测[75]的任务运行了相同的网络架构。我们仍然使用10 OKS下的mAP进行评估，结果如表9所示。平均精确率和召回率都比身体关键点任务的结果要高，这主要是因为我们使用了较小较简单的数据集。这个原始数据集由19个不同的相机的图像标注组成。我们使用了前18个相机的帧作为训练集，最后一个相机的帧作为验证集。模型架构和训练参数都没有变化。定性的结果如图13所示。

TABLE 9: Vehicle keypoint validation set.

Method | AP | AR | $AP^{75}$ | $AR^{75}$
--- | --- | --- | --- | ---
Vehicle keypoint detector | 70.1 | 77.4 | 73.0 | 79.7

Fig. 13: Vehicle keypoint detection examples from the validation set. The keypoint locations are successfully estimated under challenging scenarios, including overlapping between cars, cropped vehicles, and different scales.

### 5.6 Failure Case Analysis

We have analyzed the main cases where the current approach fails in the MPII, COCO, and COCO+foot validation sets. Fig. 14 shows an overview of the main body failure cases, while Fig. 15 shows the main foot failure cases. Fig. 14a refers to non typical poses and upside-down examples, where the predictions usually fail. Increasing the rotation augmentation visually seems to partially solve these issues, but the global accuracy on the COCO validation set is reduced by about 5%. A different alternative is to run the network using different rotations and keep the poses with the higher confidence. Body occlusion can also lead to false negatives and high localization error. This problem is inherited from the dataset annotations, in which occluded keypoints are not included. In highly crowded images where people are overlapping, the approach tends to merge annotations from different people, while missing others, due to the overlapping PAFs that make the greedy multi-person parsing fail. Animals and statues also frequently lead to false positive errors. This issue could be mitigated by adding more negative examples during training to help the network distinguish between humans and other humanoid figures.

我们分析了模型在MPII、COCO和COCO+foot数据集上的主要错误案例。图14所示的是主要的身体错误案例总览，图15所示的是主要的脚部错误案例。图14a是非典型姿态和上下颠倒的样本，预测经常会失败。增加旋转扩充可以部分解决这个问题，但在COCO验证集上的总体准确率下降了大约5%。另一种方法是使用不同的旋转运行网络，保持姿态的高置信度。身体遮挡也会导致误报和定位错误。这个问题是由数据集标注引起的，其中没有包括被遮挡的关键点。在高度拥挤的图像中，人们互相重叠，结果通常会合并不同人的标注，漏掉其他人的，因为重叠的PAFs会使贪婪多人解析失败。动物和雕像也会导致误报。这个问题可以通过在训练的过程中增加更多负样本弥补，帮助网络区分人和其他人形物体。

Fig. 14: Common failure cases: (a) rare pose or appearance, (b) missing or false parts detection, (c) overlapping parts, i.e., part detections shared by two persons, (d) wrong connection associating parts from two persons, (e-f): false positives on statues or animals.

Fig. 15: Common foot failure cases: (a) foot or leg occluded by the body, (b) foot or leg occluded by another object, (c) foot visible but leg occluded, (d) shoe and foot not aligned, (e): false negatives when foot visible but rest of the body occluded, (f): soles of their feet are usually not detected (rare in training), (g): swap between right and left body parts.

## 6 Conclusion

Realtime multi-person 2D pose estimation is a critical component in enabling machines to visually understand and interpret humans and their interactions. In this paper, we present an explicit nonparametric representation of the keypoint association that encodes both position and orientation of human limbs. Second, we design an architecture that jointly learns part detection and association. Third, we demonstrate that a greedy parsing algorithm is sufficient to produce high-quality parses of body poses, and preserves efficiency regardless of the number of people. Fourth, we prove that PAF refinement is far more important than combined PAF and body part location refinement, leading to a substantial increase in both runtime performance and accuracy. Fifth, we show that combining body and foot estimation into a single model boosts the accuracy of each component individually and reduces the inference time of running them sequentially. We have created a foot keypoint dataset consisting of 15K foot keypoint instances, which we will publicly release. Finally, we have open-sourced this work as OpenPose [4], the first realtime system for body, foot, hand, and facial keypoint detection. The library is being widely used today for many research topics involving human analysis, such as human re-identification, retargeting, and Human-Computer Interaction. In addition, OpenPose has been included in the OpenCV library [66].

实时多人2D姿态估计是让机器看懂并翻译人类及其交互的关键部件。本文中，我们提出了一种关键点关联的显式非参数表示，编码了人体肢体的位置和方向。第二，我们设计了一种架构，可以同时学习部位检测和关联。第三，我们证明了贪婪解析算法足以生成高质量的人体姿态解析，不论有多少人都不影响效率。第四，我们证明了PAF精炼比同时精炼PAF和身体部位位置要重要的多，可以得到运行时间和准确度的实质性提升。第五，我们展示了将身体和脚部估计综合到一个模型中，提升了每个单独部件的准确率，减少了单独运行它们的运行时间。我们创建了一个脚部关键点数据集包括15K个脚部关键点实例，将会公开发布。最后，我们将这些工作开源为OpenPose[4]，这是第一个实时检测身体、脚部、手部和脸部关键点的系统。这个库已经在很多人体分析的研究领域广泛使用，包括行人重识别、重新确定目标、人机交互。另外，OpenPose已经集成在OpenCV库中[66]。