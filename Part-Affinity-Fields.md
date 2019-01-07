# Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

Zhe Cao et al. The Robotics Institute, Carnegie Mellon University

## Abstract 摘要

We present an approach to efficiently detect the 2D pose of multiple people in an image. The approach uses a non-parametric representation, which we refer to as Part Affinity Fields (PAFs), to learn to associate body parts with individuals in the image. The architecture encodes global context, allowing a greedy bottom-up parsing step that maintains high accuracy while achieving realtime performance, irrespective of the number of people in the image. The architecture is designed to jointly learn part locations and their association via two branches of the same sequential prediction process. Our method placed first in the inaugural COCO 2016 keypoints challenge, and significantly exceeds the previous state-of-the-art result on the MPII MultiPerson benchmark, both in performance and efficiency.

我们提出一种方法可以有效的检测一幅图像中多个人的2D姿态。这种方法使用一种非参数的表示，我们称之为部位亲和向量场(PAF)，来学习在图像中将身体部位与个体联系起来。这种架构对全局上下文进行编码，使用一种贪婪自下而上的解析步骤，在取得高准确率的同时还能达到实时的性能，而且与图像中人的数量无关。这种架构通过相同的序贯预测过程的两个分支，可以对部位位置及其关系进行联合学习。我们的方法在第一次COCO 2016关键点挑战中取得了第一名，而且明显超过之前在MPII多人数据集上的最好结果，包括性能和效率。

## 1. Introduction 引言

Human 2D pose estimation—the problem of localizing anatomical keypoints or “parts”—has largely focused on finding body parts of individuals [8, 4, 3, 21, 33, 13, 25, 31, 6, 24]. Inferring the pose of multiple people in images, especially socially engaged individuals, presents a unique set of challenges. First, each image may contain an unknown number of people that can occur at any position or scale. Second, interactions between people induce complex spatial interference, due to contact, occlusion, and limb articulations, making association of parts difficult. Third, runtime complexity tends to grow with the number of people in the image, making realtime performance a challenge.

人体2D姿态估计，也就是定位解剖学上的关键点或称部位，一般关注在找到个体的身体部位[8, 4, 3, 21, 33, 13, 25, 31, 6, 24]。对图像中的多人姿态进行推理，尤其是社交上关联的个体，是一种很独特的挑战。首先，每幅图像可能包含的人数是未知的，出现的位置和尺度也是未知的。第二，人们之间的互动会带来复杂的空间干扰，比如接触、遮挡、肢体铰接，使得很难将各部位对上号。第三，很可能会随着图像中人数的增多而导致运行时复杂度增加，从而不能实时运行。

A common approach [23, 9, 27, 12, 19] is to employ a person detector and perform single-person pose estimation for each detection. These top-down approaches directly leverage existing techniques for single-person pose estimation [17, 31, 18, 28, 29, 7, 30, 5, 6, 20], but suffer from early commitment: if the person detector fails–as it is prone to do when people are in close proximity–there is no recourse to recovery. Furthermore, the runtime of these top-down approaches is proportional to the number of people: for each detection, a single-person pose estimator is run, and the more people there are, the greater the computational cost. In contrast, bottom-up approaches are attractive as they offer robustness to early commitment and have the potential to decouple runtime complexity from the number of people in the image. Yet, bottom-up approaches do not directly use global contextual cues from other body parts and other people. In practice, previous bottom-up methods [22, 11] do not retain the gains in efficiency as the final parse requires costly global inference. For example, the seminal work of Pishchulin et al. [22] proposed a bottom-up approach that jointly labeled part detection candidates and associated them to individual people. However, solving the integer linear programming problem over a fully connected graph is an NP-hard problem and the average processing time is on the order of hours. Insafutdinov et al. [11] built on [22] with stronger part detectors based on ResNet [10] and image-dependent pairwise scores, and vastly improved the runtime, but the method still takes several minutes per image, with a limit on the number of part proposals. The pairwise representations used in [11], are difficult to regress precisely and thus a separate logistic regression is required.

一种通常的方法[23, 9, 27, 12, 19]是采用人形检测器并对每个检测结果进行单人的姿态估计。这种自上而下的方法直接利用了现有的激素进行单人姿态估计[17, 31, 18, 28, 29, 7, 30, 5, 6, 20]，但存在提前承诺的问题：如果人形检测器失败了，就没有补救的办法了，而在人与人之间非常接近时，这很容易发生。而且，这些自上而下的方法在运行时的复杂度是与人的数量成正比的：对于每个检测结果，都要运行单人姿态估计器，人数越多，计算复杂度越高。这样自下而上的方法就很有吸引力了，因为可以解决提前承诺的问题，而且运行时的复杂度可能与人数无关。但是，自下而上的方法没有直接使用其他身体部位和其他人的全局上下文线索。在实践中，之前的自下而上的方法[22,11]效率不是很高，因为最后的解析需要非常耗时的全局推理。比如，Pishchulin等[22]的初始工作提出了一种自下而上的方法，可以联合标记部位检测候选，并将之与个体关联起来。但是，对整数线性程序问题在全连接图上进行求解是一个NP难题，平均处理时间是以小时计的。Insafutdinov等[11]在[22]的基础上基于ResNet[10]构建了更强的部位检测器和依赖于图像的成对分数，极大的改进了运行时性能，但这种方法运算速度仍然是每幅图像几分钟，在部位候选的数量上还有限制。[11]中用到的成对表示，难于精确回归，所以需要一个单独的logitstic回归。

In this paper, we present an efficient method for multi-person pose estimation with state-of-the-art accuracy on multiple public benchmarks. We present the first bottom-up representation of association scores via Part Affinity Fields (PAFs), a set of 2D vector fields that encode the location and orientation of limbs over the image domain. We demonstrate that simultaneously inferring these bottom-up representations of detection and association encode global context sufficiently well to allow a greedy parse to achieve high-quality results, at a fraction of the computational cost. We have publically released the code for full reproducibility, presenting the first realtime system for multi-person 2D pose detection.

在本文中，我们提出一种多人姿态估计的高效方法，在多个公开基准测试中取得了目前最好的效果。我们通过部位亲和向量场(PAF)提出了相关分数的第一种自下而上的表示，PAF是2D向量场的集合，对图像中四肢的位置和方向进行了编码。我们证明了，同时推理这些自下而上的检测和关联表示，对全局上下文的编码非常好，可以进行贪婪解析以得到高质量的结果，计算量也只需要很小一部分。我们已经公开放出了代码，以完全重现，给出了第一个多人2D姿态检测的实时系统。

Figure 1. Top: Multi-person pose estimation. Body parts belonging to the same person are linked. Bottomleft: Part Affinity Fields (PAFs) corresponding to the limb connecting right elbow and right wrist. The color encodes orientation. Bottom right: A zoomed in view of the predicted PAFs. At each pixel in the field, a 2D vector encodes the position and orientation of the limbs.

图1：上：多人姿态估计。同一个人的身体部位是相连的。下左：部位亲和向量场(PAF)，这里是右肘和右手腕连接的肢体的，颜色是对方向的编码。下右：预测PAF的放大视图，2D向量编码了肢体的位置和方向。

## 2. Method 方法

Fig. 2 illustrates the overall pipeline of our method. The system takes, as input, a color image of size w × h (Fig. 2a) and produces, as output, the 2D locations of anatomical keypoints for each person in the image (Fig. 2e). First, a feed-forward network simultaneously predicts a set of 2D confidence maps S of body part locations (Fig. 2b) and a set of 2D vector fields L of part affinities, which encode the degree of association between parts (Fig. 2c). The set $S = (S_1, S_2, ..., S_J)$ has J confidence maps, one per part, where $S_j ∈ R^{w×h}, j ∈ {1...J}$. The set $L = (L_1, L_2, ..., L_C)$ has C vector fields, one per limb(We refer to part pairs as limbs for clarity, despite the fact that some pairs are not human limbs (e.g., the face)), where $L_c ∈ R^{w×h×2}, c ∈ {1...C}$, each image location in $L_c$ encodes a 2D vector (as shown in Fig. 1). Finally, the confidence maps and the affinity fields are parsed by greedy inference (Fig. 2d) to output the 2D keypoints for all people in the image.

图2给出了我们方法的总体流程。系统的输入为一幅彩色图像，大小w×h（图2a），输出为图像中每个人的关键点的2D位置（图2e）。首先，一个前向网络同时预测一组身体部位位置的2D置信度图S（图2b），和一组部位亲和度的2D向量场L，其中编码的是部位间的关联程度（图2c）。$S = (S_1, S_2, ..., S_J)$有J个置信度图，每个部位一个图，其中$S_j ∈ R^{w×h}, j ∈ {1...J}$。$L = (L_1, L_2, ..., L_C)$有C个向量场，每个肢体一个（为清楚起见，我们称成对的部位为肢体，虽然一些对不是人体肢体，如人脸），其中$L_c ∈ R^{w×h×2}, c ∈ {1...C}$，每个图像位置的$L_c$都是一个2D向量（如图1所示）。最后，置信度图和亲和度他由贪婪推理解析（图2d），最后输出图像中所有人的2D关键点。

Figure 2. Overall pipeline. Our method takes the entire image as the input for a two-branch CNN to jointly predict confidence maps for body part detection, shown in (b), and part affinity fields for parts association, shown in (c). The parsing step performs a set of bipartite matchings to associate body parts candidates (d). We finally assemble them into full body poses for all people in the image (e).

图2. 总体流程。我们的方法输入为整个图像，送入两个分支的CNN，同时预测身体部位检测的置信度图，如(b)所示，和部位关联的部位亲和向量场，如(c)所示。解析步骤进行各种双向匹配以与身体部位候选进行关联(d)。最后我们将其组装成图像中所有人的整体身体姿态。

### 2.1. Simultaneous Detection and Association 同时检测与关联

Our architecture, shown in Fig. 3, simultaneously predicts detection confidence maps and affinity fields that encode part-to-part association. The network is split into two branches: the top branch, shown in beige, predicts the confidence maps, and the bottom branch, shown in blue, predicts the affinity fields. Each branch is an iterative prediction architecture, following Wei et al. [31], which refines the predictions over successive stages, t ∈ {1,...,T}, with intermediate supervision at each stage.

我们的架构如图3所示，同时预测检测的置信度图和亲和向量场，其中编码了部位和部位的关联。网络分成两个分支：上面这个分支，用米黄色表示，预测的是置信度图，下面这个分支，用蓝色表示，预测的是亲和向量场。每个分支都是一个迭代预测架构，与Wei等[31]类似，其中通过连续的阶段提炼预测结果，t ∈ {1,...,T}，在每个阶段都有中间监督。

Figure 3. Architecture of the two-branch multi-stage CNN. Each stage in the first branch predicts confidence maps $S^t$, and each stage in the second branch predicts PAFs $L^t$. After each stage, the predictions from the two branches, along with the image features, are concatenated for next stage.

图3. 两分支多阶段的CNN架构。每个阶段的第一分支预测的是置信度图$S^t$，每个阶段的第二分支预测的是PAFs $L^t$。在每个阶段后，两个分支的预测结果与图像特征拼接在一起，送入下一阶段。

The image is first analyzed by a convolutional network (initialized by the first 10 layers of VGG-19 [26] and fine-tuned), generating a set of feature maps F that is input to the first stage of each branch. At the first stage, the network produces a set of detection confidence maps $S^1 = ρ^1 (F)$ and a set of part affinity fields $L^1 = φ^1 (F)$, where $ρ^1$ and $φ^1$ are the CNNs for inference at Stage 1. In each subsequent stage, the predictions from both branches in the previous stage, along with the original image features F, are concatenated and used to produce refined predictions,

图像首先送入一个卷积网络（由VGG-19[26]的前10层初始化，并精调），生成特征图集合F，送入第一阶段的两个分支。在第一阶段，网络生成检测置信度图$S^1 = ρ^1 (F)$集合和部位亲和向量场$L^1 = φ^1 (F)$集合，这里$ρ^1$和$φ^1$是第一阶段推理的CNN。在每个接下来的阶段，前一阶段两个分支的预测和原始图像特征F，拼接在一起用于生成精炼的预测，

$$S^t = ρ^t (F, S^{t−1}, L^{t−1}), ∀t ≥ 2$$(1)
$$L^t = φ^t (F, S^{t−1}, L^{t−1}), ∀t ≥ 2$$(2)

where $ρ^t$ and $φ^t$ are the CNNs for inference at Stage t. 其中$ρ^t$和$φ^t$是阶段t推理的CNN。

Fig. 4 shows the refinement of the confidence maps and affinity fields across stages. To guide the network to iteratively predict confidence maps of body parts in the first branch and PAFs in the second branch, we apply two loss functions at the end of each stage, one at each branch respectively. We use an $L_2$ loss between the estimated predictions and the groundtruth maps and fields. Here, we weight the loss functions spatially to address a practical issue that some datasets do not completely label all people. Specifically, the loss functions at both branches at stage t are:

图4展示了不同阶段间置信度图和亲和向量场的提炼过程。为引导网络迭代预测第一分支的身体部位置信度图，和第二分支的PAFs，我们在每个阶段的最后使用了两个损失函数，每个分支分别一个。我们使用估计的预测和真值图/场间的$L_2$损失。这里，我们对损失函数进行空间加权，以解决一个实际问题，即一些数据集没有完全标记所有人。特别的，阶段t两个分支的损失函数为：

$$f^t_S = \sum_{j=1}^J \sum_p W(p)·||S_j^t(p) - S_j^*(p)||^2_2$$(3)
$$f^t_L = \sum_{c=1}^C \sum_p W(p)·||L_c^t(p) - L_c^*(p)||^2_2$$(4)

where $S^∗_j$ is the groundtruth part confidence map, $L^∗_c$ is the groundtruth part affinity vector field, W is a binary mask with W(p) = 0 when the annotation is missing at an image location p. The mask is used to avoid penalizing the true positive predictions during training. The intermediate supervision at each stage addresses the vanishing gradient problem by replenishing the gradient periodically [31]. The overall objective is

其中$S^∗_j$是真值部位置信度图，$L^∗_c$是真值部位亲和向量场，W是一个二值掩模，当在位置缺少标注时，W(p)=0。掩模用于避免在训练时惩罚了真阳性预测。每个阶段的中间监督通过周期性的补充梯度[31]来解决梯度消失问题。总共的目标函数为：

$$f = \sum_{t=1}^T (f_S^t + f_L^t)$$(5)

Figure 4. Confidence maps of the right wrist (first row) and PAFs (second row)of right forearm across stages. Although there is confusion between left and right body parts and limbs in early stages, the estimates are increasingly refined through global inference in later stages, as shown in the highlighted areas.

图4. 不同阶段的右手腕置信度图（第一行）和右前臂PAFs（第二行）。虽然在前期阶段存在左右身体部分和肢体的混淆，但这些估计在后续的阶段中通过全局推理不断提炼，如高亮区域所示。

### 2.2. Confidence Maps for Part Detection 部位检测中的置信度图

To evaluate $f_S$ in Eq. (5) during training, we generate the groundtruth confidence maps $S^∗$ from the annotated 2D keypoints. Each confidence map is a 2D representation of the belief that a particular body part occurs at each pixel location. Ideally, if a single person occurs in the image, a single peak should exist in each confidence map if the corresponding part is visible; if multiple people occur, there should be a peak corresponding to each visible part j for each person k.

为在训练中评估式(5)中的$f_S$，我们从标注的2D关键点中生成真值置信度图$S^∗$。每个置信度图是都是每个像素属于一个特定的身体部分的信念值的2D表示。理想的，如果图像中出现单个人，如果一个身体部位可见，那么在对应的置信度图中应当出现一个峰值；如果存在多个人，那么对应每个人k的可见部位j都应当存在一个峰值。

We first generate individual confidence maps $S^∗_{j,k}$ for each person k. Let $x{j,k} ∈ R^2$ be the groundtruth position of body part j for person k in the image. The value at location $p ∈ R^2$ in $S^∗_{j,k}$ is defined as,

我们首先对每个人k生成单个的置信度图$S^∗_{j,k}$。令$x{j,k} ∈ R^2$是个体k的身体部位j在图像中的真值位置，那么在位置$p ∈ R^2$上的$S^∗_{j,k}$值定义为

$$S_{j,k}^* (p) = exp(- \frac {||p-x_{j,k}||^2_2}{σ^2})$$(6)

where σ controls the spread of the peak. The groundtruth confidence map to be predicted by the network is an aggregation of the individual confidence maps via a max operator,

其中σ控制峰值扩散的情况。网络要预测的置信度图真值就是单个置信度图的聚合，通过一个max运算符

$$S_j^*(p) = max_k S_{j,k}^*(p)$$(7)

We take the maximum of the confidence maps instead of the average so that the precision of close by peaks remains distinct, as illustrated in the right figure. At test time, we predict confidence maps (as shown in the first row of Fig. 4), and obtain body part candidates by performing non-maximum suppression.

我们取置信度图的最大值，而不是平均值，这样峰值接近的仍然可以保持清晰可辨，如右图所示。在测试时，我们预测置信度图（如图4第一行所示），然后采用非最大抑制得到身体部位候选。

### 2.3. Part Affinity Fields for Part Association 利用部位亲和向量场进行部位关联

Given a set of detected body parts (shown as the red and blue points in Fig. 5a), how do we assemble them to form the full-body poses of an unknown number of people? We need a confidence measure of the association for each pair of body part detections, i.e., that they belong to the same person. One possible way to measure the association is to detect an additional midpoint between each pair of parts on a limb, and check for its incidence between candidate part detections, as shown in Fig. 5b. However, when people crowd together—as they are prone to do—these midpoints are likely to support false associations (shown as green lines in Fig. 5b). Such false associations arise due to two limitations in the representation: (1) it encodes only the position, and not the orientation, of each limb; (2) it reduces the region of support of a limb to a single point.

给定一组检测好的身体部位（如图5a中的红点和蓝点所示），我们怎样将其组合，形成未知人数的全身姿态呢？我们需要每对检测的身体部位间的关联置信度度量，即，它们是属于同一个人的。度量这种关联的一个可能方法是检测肢体上每对部位间的额外中间点，并在部位候选间检测其可能性，如图5b所示。但是，当人们群聚在一起的时候（人们经常这样做），这些中间点很可能导致错误的关联（如图5b中绿线所示）。这种错误关联式由于表示中的两个限制引起的：(1)只对每个肢体的位置进行了编码，而没有方向信息；(2)将一个肢体的支持区域缩小到了一个单点。

To address these limitations, we present a novel feature representation called part affinity fields that preserves both location and orientation information across the region of support of the limb (as shown in Fig. 5c). The part affinity is a 2D vector field for each limb, also shown in Fig. 1d: for each pixel in the area belonging to a particular limb, a 2D vector encodes the direction that points from one part of the limb to the other. Each type of limb has a corresponding affinity field joining its two associated body parts.

为解决这个限制，我们提出了一种新的特征表示方法，称为部位亲和向量场，保存了肢体支持范围内的位置和方向信息（如图5c所示）。部位亲和是每个肢体的2D向量场，如图1d所示：对于属于特定肢体的区域中的像素，一个2D向量编码了肢体的一个部位指向另一个部位的方向。每种肢体都有相应的亲和向量场连接其两个相关的身体部位。

Figure 5. Part association strategies. (a) The body part detection candidates (red and blue dots) for two body part types and all connection candidates (grey lines). (b) The connection results using the midpoint (yellow dots) representation: correct connections (black lines) and incorrect connections (green lines) that also satisfy the incidence constraint. (c) The results using PAFs (yellow arrows). By encoding position and orientation over the support of the limb, PAFs eliminate false associations.

图5. 部位关联策略。(a)两个身体部位类型的身体部位检测候选（红点和蓝点）以及其连接候选（灰线）；(b)使用中间点（黄点）表示的连接结果：正确的连接（黑线）和不正确的连接（绿线）；(c)使用PAFs的结果（黄色箭头）。通过对肢体支持区域的位置和方向进行编码，PAFs消除了错误关联。

Consider a single limb shown in the figure below. Let $x_{j_1, k}$ and $x_{j_2, k}$ be the groundtruth positions of body parts $j_1$ and $j_2$ from the limb c for person k in the image. If a point p lies on the limb, the value at $L^∗_{c,k} (p)$ is a unit vector that points from $j_1$ to $j_2$; for all other points, the vector is zero-valued.

考虑下图中的单个肢体。令$x_{j_1, k}$和$x_{j_2, k}$为身体部位$j_1$和$j_2$真值位置，对应图中人体k的肢体c。如果点p在这个肢体上，那么在点p上的值$L^∗_{c,k} (p)$是一个从$j_1$指向$j_2$的单位向量；对于其他点，向量值为0。

To evaluate $f_L$ in Eq. 5 during training, we define the groundtruth part affinity vector field, $L^∗_{c,k}$, at an image point p as

为了在训练中求的$f_L$的值，我们定义真值部位亲和向量场$L^∗_{c,k}$，在图像位置p上的值为：

$$L^*_{c,k} (p) = v, if\space p\space on\space limb\space c,k; = 0, otherwise$$(8)

Here, $v = (x_{j_2, k} − x_{j_1, k})/||x_{j_2, k} −x_{j_1, k}||_2$ is the unit vector in the direction of the limb. The set of points on the limb is defined as those within a distance threshold of the line segment, i.e., those points p for which

这里$v = (x_{j_2, k} − x_{j_1, k})/||x_{j_2, k} −x_{j_1, k}||_2$，是肢体方向的单位向量。肢体上的点的集合定义为线段距离阈值范围内的点，即满足下式的点：

$0 ≤ v · (p − x_{j_1, k})≤l_{c,k}$ and $|v_⊥ · (p − x_{j_1, k})| ≤ σ_l$ 

其中肢体的宽度$σ_l$是一个距离，单位为像素，肢体的长度为$l_{c,k} = ||x_{j_2, k} −x_{j_1, k}||_2$，向量$v_⊥$是与v垂直的向量。

The groundtruth part affinity field averages the affinity fields of all people in the image, 真值部位亲和向量场将图像中所有人的亲和向量场进行了平均

$$L_c^*(p) = \frac {\sum_k L^*_{c,k}(p)}{n_c(P)}$$(9)

where $n_c (p)$ is the number of non-zero vectors at point p across all k people (i.e., the average at pixels where limbs of different people overlap). 其中$n_c (p)$是k个人在p点的非零向量的数目（即，不同人的肢体重叠处像素的平均）。

During testing, we measure association between candidate part detections by computing the line integral over the corresponding PAF, along the line segment connecting the candidate part locations. In other words, we measure the alignment of the predicted PAF with the candidate limb that would be formed by connecting the detected body parts. Specifically, for two candidate part locations $d_{j_1}$ and $d_{j_2}$, we sample the predicted part affinity field, $L_c$ along the line segment to measure the confidence in their association:

在测试时，我们衡量部位检测候选间的关联的方法是，沿着连接部位候选位置的线段，计算对应PAF的线积分。换句话说，我们衡量预测到的PAF，与连接检测到的身体部位形成的候选肢体，之间的对齐程度。特别的，对于两个部位候选位置$d_{j_1}$和$d_{j_2}$，我们沿着线段对预测的部位亲和向量场进行取样，衡量其关联的置信度：

$$E = \int_{u=0}^{u=1} L_c(p(u)) · \frac{d_{j_2}-d_{j_1}}{||d_{j_2}-d_{j_1}||_2}$$(10)

where p(u) interpolates the position of the two body parts $d_{j_1}$ and $d_{j_2}$, 其中p(u)是两个身体部位位置$d_{j_1}$和$d_{j_2}$之间的差值：

$$p(u) = (1-u)d_{j_1} + ud_{j_2}$$(11)

In practice, we approximate the integral by sampling and summing uniformly-spaced values of u. 实际中，我们将积分进行近似计算，即将u进行均匀取样并对值进行相加。

### 2.4. Multi-Person Parsing using PAFs 解析多人PAF

We perform non-maximum suppression on the detection confidence maps to obtain a discrete set of part candidate locations. For each part, we may have several candidates, due to multiple people in the image or false positives (shown in Fig. 6b). These part candidates define a large set of possible limbs. We score each candidate limb using the line integral computation on the PAF, defined in Eq. 10. The problem of finding the optimal parse corresponds to a K-dimensional matching problem that is known to be NP-Hard [32] (shown in Fig. 6c). In this paper, we present a greedy relaxation that consistently produces high-quality matches. We speculate the reason is that the pair-wise association scores implicitly encode global context, due to the large receptive field of the PAF network.

我们对检测置信度图进行非最大抑制，以得到部位候选位置的离散集合。对于每个部位，由于图像中可能有多人，或者有误检，所以我们可能有几个候选（如图6b）。这些部位候选定义了可能肢体的集合。我们对每个候选肢体进行评分，使用式(10)的PAF线积分。找到最佳解析的问题，对应的是一个K维匹配问题，这是一个NP难题[32]（如图6c所示）。本文中，我们提出一种贪婪relaxation，一直可以得到高质量的匹配。我们推测原因是，由于PAF网络的大感受野，成对的关联分数隐含的编码了全局上下文信息。

Figure 6. Graph matching. (a) Original image with part detections (b) K-partite graph (c) Tree structure (d) A set of bipartite graphs

Formally, we first obtain a set of body part detection candidates $D_J$ for multiple people, where $D_J = \{d^m_j:$ for $j ∈ \{1...J\},m ∈ \{1...N_j \}\}$, with $N_j$ the number of candidates of part j, and $d^m_j ∈ R^2$ is the location of the m-th detection candidate of body part j. These part detection candidates still need to be associated with other parts from the same person—in other words, we need to find the pairs of part detections that are in fact connected limbs. We define a variable $z_{j_1 j_2}^{mn}$ to indicate whether two detection candidates $d^m_{j_1}$ and $d^m_{j_2}$ are connected, and the goal is to find the optimal assignment for the set of all possible connections, $Z = \{ z_{j_1 j_2}^{mn}$ : for $j_1, j_2 ∈ \{1...J\}, m ∈ \{1...N_{j_1}\}, n ∈ \{1...N_{j_2}\}\}$.

正式的，我们先得到多人的身体部位检测候选集$D_J$，其中$D_J = \{d^m_j:$ for $j ∈ \{1...J\},m ∈ \{1...N j \}\}$，这里$N_j$是部位j候选的数量，$d^m_j ∈ R^2$是身体部位j的第m个检测候选的位置。这些部位检测候选仍然需要与同一个人的其他部位进行关联，换句话说，我们需要找到是实际连在一起的肢体的部位检测对。我们定义一个变量$z_{j_1 j_2}^{mn}$来指代两个检测候选$d^m_{j_1}$和$d^m_{j_2}$是连在一起的，目标是找到所有可能连接的最佳者，$Z = \{ z_{j_1 j_2}^{mn}$ : for $j_1, j_2 ∈ \{1...J\}, m ∈ \{1...N_{j_1}\}, n ∈ \{1...N_{j_2}\}\}$。

If we consider a single pair of parts $j_1$ and $j_2$ (e.g., neck and right hip) for the c-th limb, finding the optimal association reduces to a maximum weight bipartite graph matching problem [32]. This case is shown in Fig. 5b. In this graph matching problem, nodes of the graph are the body part detection candidates $D_{j_1}$ and $D_{j_2}$, and the edges are all possible connections between pairs of detection candidates. Additionally, each edge is weighted by Eq. 10—the part affinity aggregate. A matching in a bipartite graph is a subset of the edges chosen in such a way that no two edges share a node. Our goal is to find a matching with maximum weight for the chosen edges,

如果我们考虑第c个肢体的部位$j_1$和$j_2$的组合（如脖子和右髋部），找到最佳关联就成为一个最大加权双向图匹配问题[32]。这种情况如图5b所示。在这个图匹配问题中，图的节点是身体部位检测候选$D_{j_1}$和$D_{j_2}$，边是检测候选之间所有可能的连接。另外，每个边由式(10)加权，即部位亲和聚集。双向图中的匹配问题是要选出边的子集，其中任何两条边都不能共享一个节点。我们的目标是找到的匹配对于选定的边来说权值最大

$$max_{Z_c} E_c = max_{Z_c} \sum_{m∈D_{j_1}} \sum_{n∈D_{j_2}} E_{mn}·Z^{mn}_{j_1 j_2}$$(12)
$$s.t. \space ∀m∈D_{j_1}, \sum_{n∈D_{j_2}} Z^{mn}_{j_1 j_2} ≤ 1$$(13)
$$∀n∈D_{j_2}, \sum_{m∈D_{j_1}} Z^{mn}_{j_1 j_2} ≤ 1$$(14)

where $E_c$ is the overall weight of the matching from limb type c, $Z_c$ is the subset of Z for limb type c, $E_{mn}$ is the part affinity between parts $d^m_{j_1}$ and $d^n_{j_2}$ defined in Eq. 10. Eqs. 13 and 14 enforce no two edges share a node, i.e., no two limbs of the same type (e.g., left forearm) share a part. We can use the Hungarian algorithm [14] to obtain the optimal matching.

其中$E_c$是肢体类别c匹配的总体权重，$Z_c$是肢体类别cZ的子集，$E_{mn}$是部位$d^m_{j_1}$和 $d^n_{j_2}$之间的部位亲和，如式(10)定义。式(13)(14)是任意两个边都不能共享一个节点的约束，即任意两个相同类别的肢体（如左前臂）不能共享一个部位。我们可以使用Hungarian算法[14]来得到最佳匹配。

When it comes to finding the full body pose of multiple people, determining Z is a K-dimensional matching problem. This problem is NP Hard [32] and many relaxations exist. In this work, we add two relaxations to the optimization, specialized to our domain. First, we choose a minimal number of edges to obtain a spanning tree skeleton of human pose rather than using the complete graph, as shown in Fig. 6c. Second, we further decompose the matching problem into a set of bipartite matching subproblems and determine the matching in adjacent tree nodes independently, as shown in Fig. 6d. We show detailed comparison results in Section 3.1, which demonstrate that minimal greedy inference well-approximate the global solution at a fraction of the computational cost. The reason is that the relationship between adjacent tree nodes is modeled explicitly by PAFs, but internally, the relationship between nonadjacent tree nodes is implicitly modeled by the CNN. This property emerges because the CNN is trained with a large receptive field, and PAFs from non-adjacent tree nodes also influence the predicted PAF.

要解决的多人整体身体姿态估计，其中确定Z是一个K维匹配问题。这是一个NP难题[32]，有很多relaxation。在本文中，我们为这个优化问题增加2个ralaxation，都是我们领域特有的。首先，我们选择最小数量的边得到人体姿态的支撑树骨架，而不使用完全的图，如图6c所示。第二，我们进一步将匹配问题分解成一系列双向匹配子问题，独立确定临近树节点的匹配，如图6d所示。我们在3.1节中给出详细的对比，说明最小贪婪推理的很好，以很小的计算代价近似得到了全局解。其原因是，临近树节点的关系是由PAF显式的建模的，但在内部，非临近树节点的关系由CNN隐式建模。这种性质是由于CNN的训练感受野很大，非近邻树节点的PAF也影响的预测的PAF。


With these two relaxations, the optimization is decomposed simply as: 有了这两个relaxation， 优化问题被分解为如下形式：

$$max_Z E = \sum_{c=1}^C max_{Z_c} E_c$$(15)

We therefore obtain the limb connection candidates for each limb type independently using Eqns. 12- 14. With all limb connection candidates, we can assemble the connections that share the same part detection candidates into full-body poses of multiple people. Our optimization scheme over the tree structure is orders of magnitude faster than the optimization over the fully connected graph [22, 11].

所以我们使用式(12-14)为每种肢体类别独立的得到了肢体连接候选。有了这些肢体连接候选，我们可以组合这些连接，使其共享相同的部位预测候选，得到多人的完整身体姿态。我们在树结构上的优化方案比在全连接图上的优化方案快了好几个数量级[22,11]。

## 3. Results 结果

We evaluate our method on two benchmarks for multi-person pose estimation: (1) the MPII human multi-person dataset [2] and (2) the COCO 2016 keypoints challenge dataset [15]. These two datasets collect images in diverse scenarios that contain many real-world challenges such as crowding, scale variation, occlusion, and contact. Our approach set the state-of-the-art on the inaugural COCO 2016 keypoints challenge [1], and significantly exceeds the previous state-of-the-art result on the MPII multi-person benchmark. We also provide runtime analysis to quantify the efficiency of the system. Fig. 10 shows some qualitative results from our algorithm.

我们在两个多人姿态估计的基准测试中评估了我们的方法：(1)MPII多人姿态数据集[2]；(2)COCO 2016关键点挑战数据集[15]。这两个数据集从各种场景收集了图片，包括很多真实世界的挑战，如群聚、多种尺度、遮挡和接触。我们的方法在刚刚兴起的COCO 2016关键点挑战[1]中树立了最好的结果，明显超过了之前在MPII多人基准测试上的最好结果。我们还给出了运行时分析，量化了系统的效率。图10是我们算法的一些定性结果。

Figure 10. Results containing viewpoint and appearance variation, occlusion, crowding, contact, and other common imaging artifacts.

### 3.1. Results on the MPII Multi-Person Dataset 在MPII多人数据集上的结果

For comparison on the MPII dataset, we use the toolkit [22] to measure mean Average Precision (mAP) of all body parts based on the PCKh threshold. Table 1 compares mAP performance between our method and other approaches on the same subset of 288 testing images as in [22], and the entire MPI testing set, and self-comparison on our own validation set. Besides these measures, we compare the average inference/optimization time per image in seconds. For the 288 images subset, our method outperforms previous state-of-the-art bottom-up methods [11] by 8.5% mAP. Remarkably, our inference time is 6 orders of magnitude less. We report a more detailed runtime analysis in Section 3.3. For the entire MPII testing set, our method without scale search already outperforms previous state-of-the-art methods by a large margin, i.e., 13% absolute increase on mAP. Using a 3 scale search (×0.7, ×1 and ×1.3) further increases the performance to 75.6% mAP. The mAP comparison with previous bottom-up approaches indicate the effectiveness of our novel feature representation, PAFs, to associate body parts. Based on the tree structure, our greedy parsing method achieves better accuracy than a graphcut optimization formula based on a fully connected graph structure [22, 11].

为在MPII数据集上进行对比，我们使用了[22]中的工具集，基于PCKh阈值来衡量所有身体部位的mAP。表1比较了我们的方法和其他方法在288幅测试图像上的mAP性能，这些图像与[22]中所用的样，以及完整的MPII测试集，以及在我们子集的验证集上的自我对比。除了这些衡量，我们还比较了每幅图像的平均推理/优化时间。对于288幅图像的子集，我们的方法超过了之前最好的自下而上的方法[11] 8.5% mAP。我们的推理时间小了6个数量级。我们在3.3节中更详细的分析运行时。对于完整的MPII测试集，我们的方法在没有尺度搜索的情况下，已经超过了之前最好的方法很多，即mAP的13%绝对增长。使用3尺度搜索(×0.7, ×1 and ×1.3)，可以进一步将性能提升到75.6% mAP。与之前的自下而上的方法的mAP对比表明我们新颖的特征表示PAFs进行关联身体部位的有效性。基于树结构，我们的贪婪解析方法比graphcut优化公式得到了更好的准确率，graphcut是基于全连接图结构的[22,11]。

Table 1. Results on the MPII dataset. Top: Comparison result on the 288 images testing subset. Middle: Comparison results on the whole testing set. Testing without scale search is denoted as “(one scale)”.

Method | Hea | Sho | Elb | Wri | Hip | Kne | Ank | mAP | s/image
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Deepcut [22] | 73.4 | 71.8 | 57.9 | 39.9 | 56.7 | 44.0 | 32.0 | 54.1 | 57995
Iqbal et al. [12] | 70.0 | 65.2 | 56.4 | 46.1 | 52.7 | 47.9 | 44.5 | 54.7 | 10
DeeperCut [11] | 87.9 | 84.0 | 71.9 | 63.9 | 68.8 | 63.8 | 58.1 | 71.2 | 230
Ours | 93.7 | 91.4 | 81.4 | 72.5 | 77.7 | 73.0 | 68.1 | 79.7 | 0.005
DeeperCut [11] | 78.4 | 72.5 | 60.2 | 51.0 | 57.2 | 52.0 | 45.4 | 59.5 | 485
Iqbal et al. [12] | 58.4 | 53.9 | 44.5 | 35.0 | 42.2 | 36.7 | 31.1 | 43.1 | 10
Ours (one scale) | 89.0 | 84.9 | 74.9 | 64.2 | 71.0 | 65.6 | 58.1 | 72.5 | 0.005
Ours | 91.2 | 87.6 | 77.7 | 66.8 | 75.4 | 68.9 | 61.7 | 75.6 | 0.005

In Table 2, we show comparison results on different skeleton structures as shown in Fig. 6 on our validation set, i.e., 343 images excluded from the MPII training set. We train our model based on a fully connected graph, and compare results by selecting all edges (Fig. 6b, approximately solved by Integer Linear Programming), and minimal tree edges (Fig. 6c, approximately solved by Integer Linear Programming, and Fig. 6d, solved by the greedy algorithm presented in this paper). Their similar performance shows that it suffices to use minimal edges. We trained another model that only learns the minimal edges to fully utilize the network capacity—the method presented in this paper—that is denoted as Fig. 6d (sep). This approach outperforms Fig. 6c and even Fig. 6b, while maintaining efficiency. The reason is that the much smaller number of part association channels (13 edges of a tree vs 91 edges of a graph) makes it easier for training convergence.

在表2中，我们在我们的验证集（从MPII训练集上剔除出的343幅图像）上，对比了图6中不同骨架架构的结果对比。我们基于全连接图训练我们的模型，比较了几种结果，包括选择所有边的（图6b，近似用整数线性编程求解的），最小树边数（图6c，近似用整数线性编程求解的，和图6d，用本文给出的贪婪算法求解）。其类似的性能说明，使用最小边就足够了。我们训练了另外一个模型，只学习了最小边数以充分利用网络容量（本文提出的方法），用图6d(sep)表示。这种方法超过了图6c甚至图6b，而且保持了效率。原因是部位关联通道少的多（树的13边，对比图的91边），使得训练容易收敛。

Table 2. Comparison of different structures on our validation set.

Method | Hea | Sho | Elb | Wri | Hip | Kne | Ank | mAP | s/image
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Fig. 6b | 91.8 | 90.8 | 80.6 | 69.5 | 78.9 | 71.4 | 63.8 | 78.3 | 362
Fig. 6c | 92.2 | 90.8 | 80.2 | 69.2 | 78.5 | 70.7 | 62.6 | 77.6 | 43
Fig. 6d | 92.0 | 90.7 | 80.0 | 69.4 | 78.4 | 70.1 | 62.3 | 77.4 | 0.005
Fig. 6d (sep) | 92.4 | 90.4 | 80.9 | 70.8 | 79.5 | 73.1 | 66.5 | 79.1 | 0.005

Fig. 7a shows an ablation analysis on our validation set. For the threshold of PCKh-0.5, the result using PAFs outperforms the results using the midpoint representation, specifically, it is 2.9% higher than one-midpoint and 2.3% higher than two intermediate points. The PAFs, which encodes both position and orientation information of human limbs, is better able to distinguish the common cross-over cases, e.g., overlapping arms. Training with masks of unlabeled persons further improves the performance by 2.3% because it avoids penalizing the true positive prediction in the loss during training. If we use the ground-truth keypoint location with our parsing algorithm, we can obtain a mAP of 88.3%. In Fig.7a, the mAP of our parsing with GT detection is constant across different PCKh thresholds due to no localization error. Using GT connection with our keypoint detection achieves a mAP of 81.6%. It is notable that our parsing algorithm based on PAFs achieves a similar mAP as using GT connections (79.4% vs 81.6%). This indicates parsing based on PAFs is quite robust in associating correct part detections. Fig. 7b shows a comparison of performance across stages. The mAP increases monotonically with the iterative refinement framework. Fig. 4 shows the qualitative improvement of the predictions over stages.

图7a是在我们的验证集上的分离对比分析。对于阈值PCKh-0.5，采用PAFs的结果超过了使用中间点表示的结果，特别的，比一个中间的结果高2.9%，比两个中间点的结果高2.3%。PAFs编码了人体肢体的位置信息和方向信息，能够更好的区分常见的交叉情况，如重叠的胳膊。使用未标记人体的掩模训练进一步改进了性能2.3%，因为避免了惩罚损失函数中的真阳性预测。如果我们使用真值关键点位置和我们的解析算法，我们可以得到88.3%的mAP。如图7a，我们用GT检测的解析得到的mAP在不同的PCKh阈值下是常数，因为没有定位错误。使用GT连接和我们的关键点检测得到的mAP为81.6%。值得注意，我们基于PAFs的解析算法与使用GT连接得到了类似的mAP(79.4% vs 81.6%)。这说明基于PAFs的解析在关联正确的部位检测时非常稳健。图7b展示了不同阶段的性能对比。mAP随着迭代优化框架不断增加。图4展示了随着阶段不断改进的预测结果。

Figure 7. mAP curves over different PCKh threshold on MPII validation set. (a) mAP curves of self-comparison experiments. (b) mAP curves of PAFs across stages.

### 3.2. Results on the COCO Keypoints Challenge 在COCO关键点挑战中的结果

The COCO training set consists of over 100K person instances labeled with over 1 million total keypoints (i.e. body parts). The testing set contains “test-challenge”, “test-dev” and “test-standard” subsets, which have roughly 20K images each. The COCO evaluation defines the object keypoint similarity (OKS) and uses the mean average precision (AP) over 10 OKS thresholds as main competition metric [1]. The OKS plays the same role as the IoU in object detection. It is calculated from scale of the person and the distance between predicted points and GT points. Table 3 shows results from top teams in the challenge. It is noteworthy that our method has lower accuracy than the top-down methods on people of smaller scales ($AP^M$). The reason is that our method has to deal with a much larger scale range spanned by all people in the image in one shot. In contrast, top-down methods can rescale the patch of each detected area to a larger size and thus suffer less degradation at smaller scales.

COCO训练集包括超过10万个人体实例，标记了总计超过100万个关键点（即，身体部位）。测试集包括“test-challenge”, “test-dev” and “test-standard”三个子集，每个子集大约有2万幅图像。COCO评估定义了目标目标关键点类似度(OKS)，使用了10个OKS阈值上的mAP值作为主要的竞赛度量[1]。OKS在关键点检测中的作用，与IOU在目标检测中的作用类似。OKS是由人体的尺度和预测点与GT点的距离计算出来的。表3给出了挑战中最好的团队结果。值得注意的是，我们的方法在较小的尺度上比自上而下的方法准确率低一些($AP^M$)。原因是我们的方法需要处理的尺度范围更大，一次就处理图像中所有人。作为对比，自上而下的方法可以将检测到的区域块重新改变大小，成为一个更大的尺寸，所以在更小的尺度上降质少一些。

Table 3. Results on the COCO 2016 keypoint challenge. Top: results on test-challenge. Bottom: results on test-dev (top methods only). $AP^{50}$ is for OKS = 0.5, $AP^L$ is for large scale persons.

Team | AP | $AP^{50}$ | $AP^{75}$ | $AP^M$ | $AP^L$
--- | --- | --- | --- | --- | ---
Ours | 60.5 | 83.4 | 66.4 | 55.1 | 68.1
G-RMI [19] | 59.8 | 81.0 | 65.1 | 56.7 | 66.7
DL-61 | 53.3 | 75.1 | 48.5 | 55.5 | 54.8
R4D | 49.7 | 74.3 | 54.5 | 45.6 | 55.6
Ours | 61.8 | 84.9 | 67.5 | 57.1 | 68.2
G-RMI [19] | 60.5 | 82.2 | 66.2 | 57.6 | 66.6
DL-61 | 54.4 | 75.3 | 50.9 | 58.3 | 54.3
R4D | 51.4 | 75.0 | 55.9 | 47.4 | 56.7

In Table 4, we report self-comparisons on a subset of the COCO validation set, i.e., 1160 images that are randomly selected. If we use the GT bounding box and a single person CPM [31], we can achieve a upper-bound for the top-down approach using CPM, which is 62.7% AP. If we use the state-of-the-art object detector, Single Shot MultiBox Detector (SSD)[16], the performance drops 10%. This comparison indicates the performance of top-down approaches rely heavily on the person detector. In contrast, our bottom-up method achieves 58.4% AP. If we refine the results of our method by applying a single person CPM on each rescaled region of the estimated persons parsed by our method, we gain an 2.6% overall AP increase. Note that we only update estimations on predictions that both methods agree well enough, resulting in improved precision and recall. We expect a larger scale search can further improve the performance of our bottom-up method. Fig. 8 shows a breakdown of errors of our method on the COCO validation set. Most of the false positives come from imprecise localization, other than background confusion. This indicates there is more improvement space in capturing spatial dependencies than in recognizing body parts appearances.

表4中，我们给出了COCO验证集子集上的自我对比试验，子集是随机选择的1160幅图像。如果我们使用GT边界框和单人CPM[31]，我们可以得到使用CPM的自上而下方法的最好成绩，62.7%。如果我们使用最好的目标检测器SSD[16]，性能下降了10%。这种对比说明自上而下的方法的表现严重依赖人体检测器。作为对比，我们的自下而上的方法得到了58.4% AP。如果将我们方法的结果进行提炼，将我们的方法解析得到的人重新改变区域大小，送入单人CPM，我们会得到总体AP 2.6%的改进。注意我们只更新对两种方法结果一致的预测的估计，得到的结果精度和召回率都有改善。我们觉得扩大尺度搜索范围可以进一步改进我们的自下而上方法的性能。图8所示的是我们的方法在COCO验证集上错误率的breakdown。多数误报是因为不准确的定位，而不是与背景混淆。这说明在捕捉空间依赖关系方面有更多改进的空间，而不是识别身体部位的外观。

Table 4. Self-comparison experiments on the COCO validation set.

Method | AP | $AP^{50}$ | $AP^{75}$ | $AP^M$ | $AP^L$
--- | --- | --- | --- | --- | --- | --- | ---
GT Bbox + CPM [11] | 62.7 | 86.0 | 69.3 | 58.5 | 70.6
SSD [16] + CPM [11] | 52.7 | 71.1 | 57.2 | 47.0 | 64.2
Ours - 6 stages | 58.4 | 81.5 | 62.6 | 54.4 | 65.1
+CPM refinement | 61.0 | 84.9 | 67.5 | 56.3 | 69.3

### 3.3. Runtime Analysis 运行时间分析

To analyze the runtime performance of our method, we collect videos with a varying number of people. The original frame size is 1080×1920, which we resize to 368×654 during testing to fit in GPU memory. The runtime analysis is performed on a laptop with one NVIDIA GeForce GTX-1080 GPU. In Fig. 8d, we use person detection and single-person CPM as a top-down comparison, where the runtime is roughly proportional to the number of people in the image. In contrast, the runtime of our bottom-up approach increases relatively slowly with the increasing number of people. The runtime consists of two major parts: (1) CNN processing time whose runtime complexity is O(1), constant with varying number of people; (2) Multi-person parsing time whose runtime complexity is $O(n^2)$, where n represents the number of people. However, the parsing time does not significantly influence the overall runtime because it is two orders of magnitude less than the CNN processing time, e.g., for 9 people, the parsing takes 0.58 ms while CNN takes 99.6 ms. Our method has achieved the speed of 8.8 fps for a video with 19 people.

为分析我们方法的运行时间性能，我们收集包含不同人数的视频。原始帧大小为1080×1920，在测试时大小改变为368×654，以适应GPU内存。运行时间分析是在一台笔记本上进行的，带有一个NVidia GeForce GTX 1080 GPU。如图8d所示，我们我们使用人体检测加单人CPM作为自上而下的对比，其运行时间大概与图像中的人数成正比。而随着人数的增加，我们的自下而上的方法增长就缓慢的多。运行时间主要包括两个主要部分：(1)CNN处理时间，其时间复杂度为O(1)，与人数变化无关；(2)多人解析时间，其运行时间复杂度为$O(n^2)$，n表示人数。但是，解析时间没有明显影响总共的运行时间，因为比CNN处理的时间少了两个数量级，如，在9人的情况下，解析时间为0.58ms，而CNN时间为99.6ms。我们的方法在19人的视频中运行速度为8.8fps。

Figure 8. AP performance on COCO validation set in (a), (b), and (c) for Section 3.2, and runtime analysis in (d) for Section 3.3.

## 4. Discussion 讨论

Moments of social significance, more than anything else, compel people to produce photographs and videos. Our photo collections tend to capture moments of personal significance: birthdays, weddings, vacations, pilgrimages, sports events, graduations, family portraits, and so on. To enable machines to interpret the significance of such photographs, they need to have an understanding of people in images. Machines, endowed with such perception in realtime, would be able to react to and even participate in the individual and social behavior of people.

人们拍照摄影的场合一般是社交场合。我们的图像集一般会捕捉对个人重要的时刻：生日，婚礼，假期，朝圣，体育事项，毕业，家庭合影等等。为使机器可以解释这些图像的重要性，它们需要理解图像中的人。能够实时处理这种场景的机器，可以对人的社交行为做出反应，甚至参加进去。

In this paper, we consider a critical component of such perception: realtime algorithms to detect the 2D pose of multiple people in images. We present an explicit nonparametric representation of the keypoints association that encodes both position and orientation of human limbs. Second, we design an architecture for jointly learning parts detection and parts association. Third, we demonstrate that a greedy parsing algorithm is sufficient to produce high-quality parses of body poses, that maintains efficiency even as the number of people in the image increases. We show representative failure cases in Fig. 9. We have publicly released our code (including the trained models) to ensure full reproducibility and to encourage future research in the area.

本文中，我们考虑这种感知的一个关键部件：检测图像中多人的2D姿态的实时算法。我们提出一种关键点关联的显式非参数表示，编码了人体肢体的位置和方向。第二，我们设计了一种架构来对部位检测和部位关联共同学习。第三，我们证明了贪婪解析算法可以生成高质量的身体姿态解析，即使图像中人数增多，也会保持性能。我们在图9中给出代表性的错误案例。我们已经公开了我们的代码（和训练好的模型），以确保完全可复现性，鼓励这个领域的未来研究。

**Acknowledgements** We acknowledge the effort from the authors of the MPII and COCO human pose datasets. These datasets make 2D human pose estimation in the wild possible. This research was supported in part by ONR Grants N00014-15-1-2358 and N00014-14-1-0595.