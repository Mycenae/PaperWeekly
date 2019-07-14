# RetinaFace: Single-stage Dense Face Localisation in the Wild

## Abstract 摘要

Though tremendous strides have been made in uncontrolled face detection, accurate and efficient face localisation in the wild remains an open challenge. This paper presents a robust single-stage face detector, named RetinaFace, which performs pixel-wise face localisation on various scales of faces by taking advantages of joint extra-supervised and self-supervised multi-task learning. Specifically, We make contributions in the following five aspects: (1) We manually annotate five facial landmarks on the WIDER FACE dataset and observe significant improvement in hard face detection with the assistance of this extra supervision signal. (2) We further add a self-supervised mesh decoder branch for predicting a pixel-wise 3D shape face information in parallel with the existing supervised branches. (3) On the WIDER FACE hard test set, RetinaFace outperforms the state of the art average precision (AP) by 1.1% (achieving AP equal to 91.4%). (4) On the IJB-C test set, RetinaFace enables state of the art methods (ArcFace) to improve their results in face verification (TAR=89.59% for FAR=1e-6). (5) By employing light-weight backbone networks, RetinaFace can run real-time on a single CPU core for a VGA-resolution image. Extra annotations and code have been made available at: https://github.com/deepinsight/insightface/tree/master/RetinaFace.

虽然不受控人脸检测已经有了非常多的工作，但自然环境中准确和高效的人脸定位仍然是一个未完成的挑战。本文提出了一种稳健的单阶段人脸检测器，名为RetinaFace，进行各个尺度上的像素级人脸定位，使用的是多任务学习，同时进行了额外的监督和自监督。具体来说，我们在以下五个方面有所贡献：(1)我们手工在WIDER FACE数据集上标注了5个人脸关键点，在这些额外监督信号的帮助下，在困难人脸检测中有了非常好的改进；(2)我们进一步增加了一个自监督的网格解码器分支，预测像素级的3D形状人脸信息，与现有的监督分支并行；(3)在WIDER FACE困难测试集上，RetinaFace超过了目前最好的AP 1.1%（达到了91.4%）；(4)在IJB-C测试集下，RetinaFace使目前最好的人脸识别方法ArcFace改进了其人脸验证率(TAR=89.59% for FAR=1e-6)；(5)通过使用轻量级骨干网络，RetinaFace在单CPU上VGA分辨率可以实时运行。额外的标注和代码已经开源。

## 1. Introduction 引言

Automatic face localisation is the prerequisite step of facial image analysis for many applications such as facial attribute (e.g. expression [64] and age [38]) and facial identity recognition [45, 31, 55, 11]. A narrow definition of face localisation may refer to traditional face detection [53, 62], which aims at estimating the face bounding boxes without any scale and position prior. Nevertheless, in this paper we refer to a broader definition of face localisation which includes face detection [39], face alignment [13], pixelwise face parsing [48] and 3D dense correspondence regression [2, 12]. That kind of dense face localisation provides accurate facial position information for all different scales.

自动人脸定位是很多应用中人脸图像分析的先决步骤，如人脸属性分析（如表情，和年龄），和人脸身份识别。狭义上的人脸定位是指传统的人脸检测，其目标是，在没有任何尺度和位置先验信息下，估计人脸边界框。但在本文中，我们使用广义的人脸定位，包括人脸检测、人脸对齐、像素级人脸解析和3D密集对应性回归。这种密集人脸定位给出的是所有尺度下的准确人脸位置信息。

Inspired by generic object detection methods [16, 43, 30, 41, 42, 28, 29], which embraced all the recent advances in deep learning, face detection has recently achieved remarkable progress [23, 36, 68, 8, 49]. Different from generic object detection, face detection features smaller ratio variations (from 1:1 to 1:1.5) but much larger scale variations (from several pixels to thousand pixels). The most recent state-of-the-art methods [36, 68, 49] focus on single-stage [30, 29] design which densely samples face locations and scales on feature pyramids [28], demonstrating promising performance and yielding faster speed compared to two-stage methods [43, 63, 8]. Following this route, we improve the single-stage face detection framework and propose a state-of-the-art dense face localisation method by exploiting multi-task losses coming from strongly supervised and self-supervised signals. Our idea is examplified in Fig. 1.

通用目标检测方法采用了深度学习最近的所有进展，受此启发，人脸检测也取得了令人瞩目的成绩。与通用目标检测不同的是，人脸检测的纵横比变化较小（从1:1到1:1.5），但尺度变化要大很多（从几个像素到几千像素）。最近最好的方法聚焦在单阶段设计上，在特征金字塔上进行人脸位置和尺度的密集采样，得到了很好的性能，比两阶段方法取得了更快的速度。按照这个路径，我们改进了单阶段人脸检测框架，提出了目前最好的密集人脸定位方法，研究了来自于外部监督和自监督信号的多任务损失函数。我们的思想如图1所示。

Figure 1. The proposed single-stage pixel-wise face localisation method employs extra-supervised and self-supervised multi-task learning in parallel with the existing box classification and regression branches. Each positive anchor outputs (1) a face score, (2) a face box, (3) five facial landmarks, and (4) dense 3D face vertices projected on the image plane.

图1. 提出的单阶段像素级人脸定位方法，采用了额外监督和自监督的多任务学习，并现有的框分类和回归分支并行。每个正锚框输出的是：(1)人脸分数，(2)人脸框，(3)5个人脸关键点，(4)在图像平面投影的密集3D人脸顶点。

Typically, face detection training process contains both classification and box regression losses [16]. Chen et al. [6] proposed to combine face detection and alignment in a joint cascade framework based on the observation that aligned face shapes provide better features for face classification. Inspired by [6], MTCNN [66] and STN [5] simultaneously detected faces and five facial landmarks. Due to training data limitation, JDA [6], MTCNN [66] and STN [5] have not verified whether tiny face detection can benefit from the extra supervision of five facial landmarks. One of the questions we aim at answering in this paper is whether we can push forward the current best performance (90.3% [67]) on the WIDER FACE hard test set [60] by using extra supervision signal built of five facial landmarks.

典型的人脸检测训练过程，包含分类和边界框回归损失。Chen等观察到，对齐的人脸形状可以给出更好的人脸分类特征，因此提出将人脸检测与对齐结合到一个联合级联框架中。受[6]启发，MTCNN[66]和STN[5]同时检测人脸和五个人脸关键点。由于训练数据的限制，JDA, MTCNN和STN没有验证微小人脸检测是否可以从五个人脸关键点的额外监督中受益。我们在本文中要回答的一个问题是，是否我们可以通过五点人脸关键点的额外监督信号，将WIDER FACE困难测试集上的目前最好性能(90.3%[67])向前继续推进。

In Mask R-CNN [20], the detection performance is significantly improved by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition and regression. That confirms that dense pixel-wise annotations are also beneficial to improve detection. Unfortunately, for the challenging faces of WIDER FACE it is not possible to conduct dense face annotation (either in the form of more landmarks or semantic segments). Since supervised signals cannot be easily obtained, the question is whether we can apply unsupervised methods to further improve face detection.

在Mask R-CNN中，在现有的边界框分类和回归的基础上，并行增加了一个预测目标掩膜的分支，检测性能得到了显著改进。这确认了，密集像素级标注对改进检测也是有好处的。不幸的是，对于WIDER FACE中的人脸，不太可能去进行密集的人脸标注（更多的关键点，或语义分割）。因为监督信号不可能很容易得到，问题就变成了，是否我们可以使用非监督方法以进一步改进人脸检测。

In FAN [56], an anchor-level attention map is proposed to improve the occluded face detection. Nevertheless, the proposed attention map is quite coarse and does not contain semantic information. Recently, self-supervised 3D morphable models [14, 51, 52, 70] have achieved promising 3D face modelling in-the-wild. Especially, Mesh Decoder [70] achieves over real-time speed by exploiting graph convolutions [10, 40] on joint shape and texture. However, the main challenges of applying mesh decoder [70] into the single-stage detector are: (1) camera parameters are hard to estimate accurately, and (2) the joint latent shape and texture representation is predicted from a single feature vector (1 × 1 Conv on feature pyramid) instead of the RoI pooled feature, which indicates the risk of feature shift. In this paper, we employ a mesh decoder [70] branch through self-supervision learning for predicting a pixel-wise 3D face shape in parallel with the existing supervised branches.

在FAN[56]中，提出了一种锚框级的注意力图，来改进遮挡的人脸检测。尽管如此，提出的注意力图非常粗糙，不包含语义信息。最近，自监督的3D可变性模型在自然环境下的3D人脸建模上得到了不错的结果。尤其是，Mesh Decoder[70]通过在形状和纹理上联合利用图卷积，得到了实时速度。但是，在单阶段检测器中应用网格解码器的主要挑战在于：(1)摄像机参数很难准确估计，(2)形状和纹理的联合表示，是由单个特征向量（在特征金字塔上的1×1卷积）预测而来的，而不是RoI池化的特征，这说明有特征偏移的危险。本文中，我们通过自监督学习提出了一个mesh decoder分支，来预测像素级的3D人脸形状，与现有的监督分支是并行的。

To summarise, our key contributions are: 总结起来，我们的关键贡献在于：

- Based on a single-stage design, we propose a novel pixel-wise face localisation method named RetinaFace, which employs a multi-task learning strategy to simultaneously predict face score, face box, five facial landmarks, and 3D position and correspondence of each facial pixel. 基于单阶段设计，我们提出了一种新的像素级人脸定位方法，名为RetinaFace，采用多任务学习策略，以同时预测人脸分数、人脸边界框、五个人脸关键点，和3D位置与每个人脸像素的对应关系。

- On the WIDER FACE hard subset, RetinaFace outperforms the AP of the state of the art two-stage method (ISRN [67]) by 1.1% (AP equal to 91.4%). 在WIDER FACE困难子集，RetinaFace超过现有最好的两阶段方法(ISRN[67]) 1.1% AP，达到了91.4%。

- On the IJB-C dataset, RetinaFace helps to improve ArcFace’s [11] verification accuracy (with TAR equal to 89.59% when FAR=1e-6). This indicates that better face localisation can significantly improve face recognition. 在IJB-C集上，RetinaFace帮助改进了ArcFace的验证准确率(with TAR=89.59% when FAR=1e-6)。这说明，更好的人脸定位可以显著改进人脸识别率。

- By employing light-weight backbone networks, RetinaFace can run real-time on a single CPU core for a VGA-resolution image. 采用了轻量级骨干网络，RetinaFace在单CPU核上对于VGA分辨率的图像可以达到实时运行的效果。

- Extra annotations and code have been released to facilitate future research. 公开了额外的标注和代码。

Figure 2. An overview of the proposed single-stage dense face localisation approach. RetinaFace is designed based on the feature pyramids with independent context modules. Following the context modules, we calculate a multi-task loss for each anchor.

## 2. Related Work 相关工作

**Image pyramid v.s. feature pyramid**: The sliding-window paradigm, in which a classifier is applied on a dense image grid, can be traced back to past decades. The milestone work of Viola-Jones [53] explored cascade chain to reject false face regions from an image pyramid with real-time efficiency, leading to the widespread adoption of such scale-invariant face detection framework [66, 5]. Even though the sliding-window on image pyramid was the leading detection paradigm [19, 32], with the emergence of feature pyramid [28], sliding-anchor [43] on multi-scale feature maps [68, 49], quickly dominated face detection.

**图像金字塔vs特征金字塔**：滑窗的方式，即将分类器应用于密集图像网格，可以回溯到过去几十年。Viola-Jones[53]的里程碑式的工作，探索了采用级联结构，从图像金字塔中拒绝假人脸区域，达到了实时的运行效率，这种对尺度不变的人脸检测框架，得到了广泛的采用。虽然在图像金字塔上的滑窗是很先进的检测方式，但随着特征金字塔的出现，在多尺度特征图上采用滑动锚框，迅速的成为了人脸检测的主流。

**Two-stage v.s. single-stage**: Current face detection methods have inherited some achievements from generic object detection approaches and can be divided into two categories: two-stage methods (e.g. Faster R-CNN [43, 63, 72]) and single-stage methods (e.g. SSD [30, 68] and RetinaNet [29, 49]). Two-stage methods employed a “proposal and refinement” mechanism featuring high localisation accuracy. By contrast, single-stage methods densely sampled face locations and scales, which resulted in extremely unbalanced positive and negative samples during training. To handle this imbalance, sampling [47] and re-weighting [29] methods were widely adopted. Compared to two-stage methods, single-stage methods are more efficient and have higher recall rate but at the risk of achieving a higher false positive rate and compromising the localisation accuracy.

**两阶段vs单阶段**：目前的人脸检测方法继承于通用目标检测方法，可以分成两类：两阶段方法和单阶段方法。两阶段方法采用“建议和优化”的机制，特点是定位准确率高。形成对比的是单阶段方法，对人脸位置和尺度进行密集采样，这在训练时会形成极不平衡的正样本和负样本数量。为处理这种不平衡，采样[47]和重新赋权[29]方法得到了广泛的采用。与两阶段方法相比，单阶段方法效率更高，召回率更高，但风险是false positive率很高，定位准确率也需要折中。

**Context Modelling**: To enhance the model’s contextual reasoning power for capturing tiny faces [23], SSH [36] and PyramidBox [49] applied context modules on feature pyramids to enlarge the receptive field from Euclidean grids. To enhance the non-rigid transformation modelling capacity of CNNs, deformable convolution network (DCN) [9, 74] employed a novel deformable layer to model geometric transformations. The champion solution of the WIDER Face Challenge 2018 [33] indicates that rigid (expansion) and non-rigid (deformation) context modelling are complementary and orthogonal to improve the performance of face detection.

**上下文建模**：为增强模型的上下文推理能力，以捕获微小人脸，SSH和PyramidBox在特征金字塔上使用了上下文模块，以增大感受野。为增强CNNs的非刚性变换建模能力，可变性卷积网络(DCN)采用了新的可变性层，来对几何变换进行建模。WIDER Face 2018挑战赛的冠军方案说明，刚性和非刚性上下文建模是互补的，都可以改进人脸检测的性能。

**Multi-task Learning**: Joint face detection and alignment is widely used [6, 66, 5] as aligned face shapes provide better features for face classification. In Mask R-CNN [20], the detection performance was significantly improved by adding a branch for predicting an object mask in parallel with the existing branches. Densepose [1] adopted the architecture of Mask-RCNN to obtain dense part labels and coordinates within each of the selected regions. Nevertheless, the dense regression branch in [20, 1] was trained by supervised learning. In addition, the dense branch was a small FCN applied to each RoI to predict a pixel-to-pixel dense mapping.

**多任务学习**：同时人脸检测和对齐得到了广泛的采用，因为对齐的人脸会给出更好的人脸分类特征。在Mask R-CNN中，与现有的分支一起，并行增加了一个目标掩膜预测分支，检测性能得到了显著的改善。Densepose采用了Mask R-CNN的架构，在每个选定的区域内，得到了密集部位标签和坐标。尽管如此，[20,1]中的密集回归分支是通过监督学习的方式训练的。另外，密集分支是应用于每个RoI的小型FCN，以预测像素到像素的密集映射。

## 3. RetinaFace

### 3.1. Multi-task Loss

For any training anchor i, we minimise the following multi-task loss: 对任意训练锚框i，我们最小化下列多任务损失：

$$L = L_{cls} (p_i, p^∗_i) + λ_1 p^∗_i L_{box} (t_i, t^∗_i) + λ_2 p^∗_i L_{pts} (l_i, l_i^∗) + λ_3 p^∗_i L_{pixel}$$(1)

(1) Face classification loss $L_{cls} (p_i, p^∗_i)$, where $p_i$ is the predicted probability of anchor i being a face and $p^∗_i$ is 1 for the positive anchor and 0 for the negative anchor. The classification loss $L_{cls}$ is the softmax loss for binary classes (face/not face). 人脸分类损失$L_{cls}$，其中$p_i$是锚框i是人脸的预测概率，$p^∗_i$对于正锚框是1，对于负锚框是0。分类损失$L_{cls}$对于二值分类（人脸/非人脸）是softmax损失。

(2) Face box regression loss $L_{box} (t_i, t^∗_i)$, where $t_i = \{t_x, t_y, t_w, t_h\}_i$ and $t^∗_i = \{t^∗_x, t^∗_y, t^∗_w, t^∗_h\}_i$ represent the coordinates of the predicted box and ground-truth box associated with the positive anchor. We follow [16] to normalise the box regression targets (i.e. centre location, width and height) and use $L_{box} (t_i, t^∗_i) = R(t_i − t^∗_i)$, where R is the robust loss function (smooth-L1) defined in [16]. 人脸框回归损失$L_{box}$，其中$t_i$和$t^∗_i$表示与正锚框相关的预测框和真值框的坐标。我们采用[16]中的方法对框回归的目标进行归一化（即，中心位置，宽度和高度），使用$L_{box} (t_i, t^∗_i) = R(t_i − t^∗_i)$，其中R是[16]中定义的稳健损失函数(smooth-L1)。

(3) Facial landmark regression loss $L_{pts} (l_i, l_i^∗)$, where $l_i = \{l_{x1}, l_{y1}, . . . , l_{x5}, l_{y5} \}_i$ and $l_i^∗ = \{l_{x1}^∗, l_{y1}^∗, . . . , l_{x5}^∗, l_{y5}^∗\}_i$ represent the predicted five facial landmarks and groundtruth associated with the positive anchor. Similar to the box centre regression, the five facial landmark regression also employs the target normalisation based on the anchor centre. 人脸关键点回归损失$L_{pts}$，其中$l_i$和$l_i^*$代表正锚框相关的五个人脸关键点的预测值和真值。与边界框中心回归一样，五个人脸关键点的回归也采用了基于锚框中心的目标归一化。

(4) Dense regression loss $L_{pixel}$ (refer to Eq. 3).

The loss-balancing parameters $λ_1 - λ_3$ are set to 0.25, 0.1 and 0.01, which means that we increase the significance of better box and landmark locations from supervision signals. 损失平衡参数$λ_1 - λ_3$设为0.25, 0.1和0.01，意味着，我们增加边界框和关键点位置的重要性。

### 3.2. Dense Regression Branch 密集回归分支

**Mesh Decoder**. We directly employ the mesh decoder (mesh convolution and mesh up-sampling) from [70, 40], which is a graph convolution method based on fast localised spectral filtering [10]. In order to achieve further acceleration, we also use a joint shape and texture decoder similarly to the method in [70], contrary to [40] which only decoded shape. 我们直接采用[70,40]中的mesh decoder（mesh卷积和mesh上采样），这是一个图卷积方法，基于快速局部谱滤波[10]。为取得进一步的加速，我们还使用了形状和纹理的联合decoder，与[70]中的方法类似，而[40]中的只对形状进行解码。

Below we will briefly explain the concept of graph convolutions and outline why they can be used for fast decoding. As illustrated in Fig. 3(a), a 2D convolutional operation is a “kernel-weighted neighbour sum” within the Euclidean grid receptive field. Similarly, graph convolution also employs the same concept as shown in Fig. 3(b). However, the neighbour distance is calculated on the graph by counting the minimum number of edges connecting two vertices. We follow [70] to define a coloured face mesh G = (V, E), where $V ∈ R^{n×6}$ is a set of face vertices containing the joint shape and texture information, and $E ∈ \{0, 1\}^{n×n}$ is a sparse adjacency matrix encoding the connection status between vertices. The graph Laplacian is defined as L = D − E ∈ $R^{n×n}$ where $D ∈ R^{n×n}$ is a diagonal matrix with $D_{ii} = \sum_j E_{ij}$.

下面我们会简要的解释一下图卷积的概念，说明为什么可以用于快速解码。如图3(a)所示，2D卷积运算，是在欧几里得网格感受野上的，核加权邻域和。类似的，图卷积也采用了相同的概念，如图3(b)所示。但是，邻域距离是在图上计算的，是连接两个顶点的边的最小数量。我们按照[70]的思想，定义一个彩色人脸网格G = (V, E)，其中$V ∈ R^{n×6}$是人脸顶点的集合，包含形状和纹理的联合信息，$E ∈ \{0, 1\}^{n×n}$是稀疏邻接矩阵，编码了两个顶点间的连接状态。图的Laplacian定义为L = D − E ∈ $R^{n×n}$，其中$D ∈ R^{n×n}$是对角矩阵，$D_{ii} = \sum_j E_{ij}$。

Figure 3. (a) 2D Convolution is kernel-weighted neighbour sum within the Euclidean grid receptive field. Each convolutional layer has $Kernel_H × Kernel_W × Channel_{in} × Channel_{out}$ parameters. (b) Graph convolution is also in the form of kernel-weighted neighbour sum, but the neighbour distance is calculated on the graph by counting the minimum number of edges connecting two vertices. Each convolutional layer has $K × Channel_{in} × Channel_{out}$ parameters and the Chebyshev coefficients $θ_{i,j} ∈ R^K$ are truncated at order K.

Following [10, 40, 70], the graph convolution with kernel $g_θ$ can be formulated as a recursive Chebyshev polynomial truncated at order K, 按照[10,40,70]，卷积核为$g_θ$的图卷积，可以写成如下形式，迭代Chebyshev多项式，在K阶时截断：

$$y = g_θ(L)x = \sum_{k=0}^{K=1} θ_k T_k(\tilde L)x$$(2)

where $θ ∈ R^K$ is a vector of Chebyshev coefficients and $T_k (\tilde L) ∈ R^{n×n}$ is the Chebyshev polynomial of order k evaluated at the scaled Laplacian $\tilde L$. Denoting $x̄_k = T_k (\tilde L) x ∈ R^n$, we can recurrently compute $x̄_k = 2 \tilde Lx̄_{k−1} − x̄_{k−2}$ with $x̄_0 = x$ and $x̄ 1 = \tilde Lx$. The whole filtering operation is extremely efficient including K sparse matrix-vector multiplications and one dense matrix-vector multiplication $y = g_θ (L)x = [x̄_0, . . ., x̄_{K−1}]θ$.

其中$θ ∈ R^K$一个Chebyshev系数向量，$T_k (\tilde L) ∈ R^{n×n}$是k阶Chebyshev多项式，在尺度Laplacian$\tilde L$上的值。令$x̄_k = T_k (\tilde L) x ∈ R^n$，我们可以循环计算$x̄_k = 2 \tilde Lx̄_{k−1} − x̄_{k−2}$，初始值$x̄_0 = x$，$x̄ 1 = \tilde Lx$。整个滤波运算可以非常高效的计算出来，包括K个稀疏矩阵-向量乘法，和一个密集矩阵-向量乘法$y = g_θ (L)x = [x̄_0, . . ., x̄_{K−1}]θ$。

**Differentiable Renderer**. After we predict the shape and texture parameters $P_{ST} ∈ R^128$, we employ an efficient differentiable 3D mesh renderer [14] to project a coloured-mesh $D_{P_{ST}}$ onto a 2D image plane with camera parameters $P_{cam} = [x_c, y_c, z_c, x'_c, y'_c, z'_c, f_c]$ (i.e. camera location, camera pose and focal length) and illumination parameters $P_{ill} = [x_l, y_l, z_l, r_l, g_l, b_l, r_a, g_a, b_a]$ (i.e. location of point light source, colour values and colour of ambient lighting).

**可微分渲染器**。我们预测了形状和纹理参数$P_{ST} ∈ R^128$后，我们采用了一个高效的可微分3D网格渲染器[14]，来将一个彩色网格$D_{P_{ST}}$投影到2D图像平面上，摄像机参数为$P_{cam} = [x_c, y_c, z_c, x'_c, y'_c, z'_c, f_c]$（即，摄像机位置，摄像机姿态，和焦距），以及光照参数$P_{ill} = [x_l, y_l, z_l, r_l, g_l, b_l, r_a, g_a, b_a]$（即，点光源位置，色彩值，和环境光的色彩）。

**Dense Regression Loss**. Once we get the rendered 2D face $R(D_{P_{ST}}, P_{cam}, P_{ill})$, we compare the pixel-wise difference of the rendered and the original 2D face using the following function:

**密集回归损失**。我们得到渲染的2D人脸后$R(D_{P_{ST}}, P_{cam}, P_{ill})$，我们计算渲染的人脸和原始2D人脸的逐像素差别，使用下面的函数：

$$L_{pixel} = \frac {1} {W*H} \sum_i^W \sum_j^H ||R(D_{P_{ST}}, P_{cam}, P_{ill})_{i,j} - I^*_{i,j}||_1$$(3)

where W and H are the width and height of the anchor crop $I^*_{i,j}$, respectively. 其中W和H分别是锚框剪切块$I^*_{i,j}$的宽度和高度。

## 4. Experiments 试验

### 4.1. Dataset 数据集

The WIDER FACE dataset [60] consists of 32, 203 images and 393, 703 face bounding boxes with a high degree of variability in scale, pose, expression, occlusion and illumination. The WIDER FACE dataset is split into training (40%), validation (10%) and testing (50%) subsets by randomly sampling from 61 scene categories. Based on the detection rate of EdgeBox [76], three levels of difficulty (i.e. Easy, Medium and Hard) are defined by incrementally incorporating hard samples.

WIDER Face数据集包含32203图像和393703人脸边界框，其中人脸的尺度、姿态、表情、遮挡和光照都变化非常大。数据集分为训练集(40%)、验证集(10%)和测试集(50%)，都是从61个场景类别中随机选取的。基于EdgeBox的检测率，定义了三个难度水平（容易、中等和困难），困难样本逐步增加。

**Extra Annotations**. As illustrated in Fig. 4 and Tab. 1, we define five levels of face image quality (according to how difficult it is to annotate landmarks on the face) and annotate five facial landmarks (i.e. eye centres, nose tip and mouth corners) on faces that can be annotated from the WIDER FACE training and validation subsets. In total, we have annotated 84.6k faces on the training set and 18.5k faces on the validation set.

**额外的标注**。如图4和表1所示，我们定义了五级人脸图像质量（根据在人脸上标注关键点的难度），并对WIDER FACE训练和验证子集的可标注人脸，标注了五个关键点（即，眼睛中心，鼻尖和嘴角）。共计，我们在训练集上标注了84.6k人脸，在验证集上标注了18.5k人脸。

Figure 4. We add extra annotations of five facial landmarks on faces that can be annotated (we call them “annotatable”) from the WIDER FACE training and validation sets.

Table 1. Five levels of face image quality. In the indisputable category a human can, without a lot of effort, locale the landmarks. In the annotatable category finding an approximate location requires some effort.

Level | Face Number | Criterion
--- | --- | ---
1 | 4127 | indisputable 68 landmarks [44]
2 | 12636 | annotatable 68 landmarks [44]
3 | 38140 | indisputable 5 landmarks
4 | 50024 | annotatable 5 landmarks
5 | 94095 | distinguish by context

### 4.2. Implementation details

**Feature Pyramid**. RetinaFace employs feature pyramid levels from $P_2$ to $P_6$, where $P_2$ to $P_5$ are computed from the output of the corresponding ResNet residual stage ($C_2$ through $C_5$) using top-down and lateral connections as in [28, 29]. $P_6$ is calculated through a 3×3 convolution with stride=2 on $C_5$. $C_1$ to $C_5$ is a pre-trained ResNet-152 [21] classification network on the ImageNet-11k dataset while $P_6$ are randomly initialised with the “Xavier” method [17].

**特征金字塔**。RetinaFace使用$P_2$到$P_6$级的特征金字塔，其中$P_2$到$P_5$是对应的ResNet残差阶段的输出($C_2$到$C_5$)中计算出来的，使用[28,29]中的自上而下的和横向连接。$P_6$是对$C_5$使用步长为2的3×3卷积计算得到的。$C_1$到$C_5$是在ImageNet-11k数据集上的预训练ResNet-152分类网络，$P_6$是用Xavier方法随机初始化得到的。

**Context Module**. Inspired by SSH [36] and Pyramid-Box [49], we also apply independent context modules on five feature pyramid levels to increase the receptive field and enhance the rigid context modelling power. Drawing lessons from the champion of the WIDER Face Challenge 2018 [33], we also replace all 3 × 3 convolution layers within the lateral connections and context modules by the deformable convolution network (DCN) [9, 74], which further strengthens the non-rigid context modelling capacity.

**上下文模块**。受SSH和Pyramid-Box启发，我们在五个特征金字塔层级上使用了独立的上下文模块，以增大感受野，并增强刚性上下文建模能力。从WIDER Face 2018挑战赛冠军那吸取了经验，我们还将所有横向连接和上下文模块中的3×3卷积层替换成可变卷积网络(Deformable Convolution Network, DCN)，这进一步增强了非刚性上下文建模能力。

**Loss Head**. For negative anchors, only classification loss is applied. For positive anchors, the proposed multi-task loss is calculated. We employ a shared loss head (1 × 1 conv) across different feature maps $H_n × W_n × 256$, n ∈ {2, . . . , 6}. For the mesh decoder, we apply the pre-trained model [70], which is a small computational overhead that allows for efficient inference.

**损失头**。对于负锚框，只使用了分类损失。对于正锚框，则计算了提出的多任务损失。我们在不同的特征图$H_n × W_n × 256$, n ∈ {2, . . . , 6}中使用了一个共享的损失头(1×1卷积)。对于mesh decoder，我们使用了预训练模型[70]，其计算消耗比较小，可以进行高效的推理。

**Anchor Settings**. As illustrated in Tab. 2, we employ scale-specific anchors on the feature pyramid levels from $P_2$ to $P_6$ like [56]. Here, $P_2$ is designed to capture tiny faces by tiling small anchors at the cost of more computational time and at the risk of more false positives. We set the scale step at $2^{1/3}$ and the aspect ratio at 1:1. With the input image size at 640 × 640, the anchors can cover scales from 16 × 16 to 406 × 406 on the feature pyramid levels. In total, there are 102,300 anchors, and 75% of these anchors are from $P_2$.

**锚框设置**。如表2所示，我们在特征金字塔级$P_2$到$P_6$上使用了与尺度相关的锚框，如[56]一样。这里，$P_2$的设计是用于捕获微小人脸，将小型锚框平铺在一起，代价是更多的计算时间，风险是更多的假阳性判断。我们设置尺度步长为$2^{1/3}$，纵横比为1:1。输入图像大小在640×640时，锚框的大小在特征金字塔上可以覆盖16×16到406到406大小。总计有102300个锚框，锚框中的75%是在$P_2$上。

Table 2. The details of feature pyramid, stride size, anchor in RetinaFace. For a 640 × 640 input image, there are 102,300 anchors in total, and 75% of these anchors are tiled on $P_2$.

Feature Pyramid | Stride | Anchor
--- | --- | ---
P2 (160 × 160 × 256) | 4 | 16,20.16,25.40
P3 (80 × 80 × 256) | 8 | 32,40.32,50.80
P4 (40 × 40 × 256) | 16 | 64,80.63,101.59
P5 (20 × 20 × 256) | 32 | 128,161.26,203.19
P6 (10 × 10 × 256) | 64 | 256,322.54,406.37

During training, anchors are matched to a ground-truth box when IoU is larger than 0.5, and to the background when IoU is less than 0.3. Unmatched anchors are ignored during training. Since most of the anchors (> 99%) are negative after the matching step, we employ standard OHEM [47, 68] to alleviate significant imbalance between the positive and negative training examples. More specifically, we sort negative anchors by the loss values and select the top ones so that the ratio between the negative and positive samples is at least 3:1.

在训练时，当锚框与真值框IoU大于0.5时，则匹配为真值框，当小于0.3时，则匹配为背景。未匹配的锚框在训练时被忽略。因为大多数锚框(>99%)经过匹配后都是负的，所以我们采用标准的OHEM以缓解训练正样本和负样本的明显不均衡。具体的，我们将负锚框根据损失值进行排序，选择排名靠前的，这样负样本和正样本的比率为至少3:1。

**Data Augmentation**. Since there are around 20% tiny faces in the WIDER FACE training set, we follow [68, 49] and randomly crop square patches from the original images and resize these patches into 640 × 640 to generate larger training faces. More specifically, square patches are cropped from the original image with a random size between [0.3, 1] of the short edge of the original image. For the faces on the crop boundary, we keep the overlapped part of the face box if its centre is within the crop patch. Besides random crop, we also augment training data by random horizontal flip with the probability of 0.5 and photo-metric colour distortion [68].

**数据扩充**。因为在WIDER Face训练集中有大约20%的微小人脸，我们按照[68,49]的方法，从原始图像中随机剪切正方形块，将这些块改变大小到640×640，以生成更大的训练人脸。具体的，方形块是从原始图像中剪切出来的，大小为原始图像短边长度的[0.3,1]中的随机值。对于剪切块边缘的人脸，如果人脸中心是在剪切块中的话，那我们就保留其重叠部分。除了随机剪切，我们还用概率0.5的随机水平翻转和色彩扭曲来扩充训练数据。

**Training Details**. We train the RetinaFace using SGD optimiser (momentum at 0.9, weight decay at 0.0005, batch size of 8 × 4) on four NVIDIA Tesla P40 (24GB) GPUs. The learning rate starts from 10^−3, rising to 10^−2 after 5 epochs, then divided by 10 at 55 and 68 epochs. The training process terminates at 80 epochs.

**训练细节**。我们使用SGD优化器（动量0.9，权重衰减0.0005，批大小为8×4）在4台NVidia Tesla P40(24GB)上来训练RetinaFace。学习速率初始为0.001，在5轮之后增加到0.01，在第55轮和第68轮时分别除以10。训练过程在第80轮结束。

**Testing Details**. For testing on WIDER FACE, we follow the standard practices of [36, 68] and employ flip as well as multi-scale (the short edge of image at [500, 800, 1100, 1400, 1700]) strategies. Box voting [15] is applied on the union set of predicted face boxes using an IoU threshold at 0.4.

**测试细节**。对于在WIDER Face上的测试，我们按照[36,68]中的标准操作，采用翻转和多尺度策略（图像短边为[500, 800, 1100, 1400, 1700]）。框投票[15]用于预测人脸框的并集，使用的IoU阈值为0.4。

### 4.3. Ablation Study 分离试验

To achieve a better understanding of the proposed RetinaFace, we conduct extensive ablation experiments to examine how the annotated five facial landmarks and the proposed dense regression branch quantitatively affect the performance of face detection. Besides the standard evaluation metric of average precision (AP) when IoU=0.5 on the Easy, Medium and Hard subsets, we also make use of the development server (Hard validation subset) of the WIDER Face Challenge 2018 [33], which employs a more strict evaluation metric of mean AP (mAP) for IoU=0.5:0.05:0.95, rewarding more accurate face detectors.

为更好的理解提出的RetinaFace，我们进行了广泛的分离试验，以检验标注的5点人脸关键和提出的密集回归分支，是怎样定量影响人脸检测的性能的。除了标准评估标准，即与Easy, Medium和Hard子集在IoU=0.5时的AP，我们还利用了WIDER Face 2018挑战赛的困难验证子集的开发服务器，采用了一个更严格的评估标准，即在IoU=0.5:0.05:0.95时的mAP，以奖励更准确的人脸检测器。

As illustrated in Tab. 3, we evaluate the performance of several different settings on the WIDER FACE validation set and focus on the observations of AP and mAP on the Hard subset. By applying the practices of state-of-the-art techniques (i.e. FPN, context module, and deformable convolution), we set up a strong baseline (91.286%), which is slightly better than ISRN [67] (90.9%). Adding the branch of five facial landmark regression significantly improves the face box AP (0.408%) and mAP (0.775%) on the Hard subset, suggesting that landmark localisation is crucial for improving the accuracy of face detection. By contrast, adding the dense regression branch increases the face box AP on Easy and Medium subsets but slightly deteriorates the results on the Hard subset, indicating the difficulty of dense regression under challenging scenarios. Nevertheless, learning landmark and dense regression jointly enables a further improvement compared to adding landmark regression only. This demonstrates that landmark regression does help dense regression, which in turn boosts face detection performance even further.

如表3所述，我们在WIDER Face验证集上评估几种不同设置的性能，聚焦观察困难子集上的AP和mAP。通过使用目前最好的技术（如，FPN，上下文模块，和变形卷积），我们设定了一个很强的基准(91.286%)，比ISRN[67]的90.9%略好一些。增加5个人脸关键点回归的分支，显著提升了困难子集的人脸框AP和mAP(0.408%, 0.775%)，说明关键点定位对于改进人脸检测准确率是非常关键的。比较之下，增加密集回归分支，改进了在容易和一般子集上的人脸框AP，但在困难子集上的结果略有恶化，说明密集回归在更有挑战的场景是非常困难的。尽管如此，同时学习关键点和密集回归，与只增加关键点回归相比，还是得到了进一步的改进。这说明，关键点回归确实帮助了密集回归，也继而提升了人脸检测的性能。

Table 3. Ablation experiments of the proposed methods on the WIDER FACE validation subset.

Method | Easy | Medium | Hard | mAP[33]
--- | --- | --- | --- | ---
FPN+Context | 95.532 | 95.134 | 90.714 | 50.842
+DCN | 96.349 | 95.833 | 91.286 | 51.522
+L_pts | 96.467 | 96.075 | 91.694 | 52.297
+L_pixel | 96.413 | 95.864 | 91.276 | 51.492
+L_pts + L_pixel | 96.942 | 96.175 | 91.857 | 52.318

### 4.4. Face box Accuracy

Following the stander evaluation protocol of the WIDER FACE dataset, we only train the model on the training set and test on both the validation and test sets. To obtain the evaluation results on the test set, we submit the detection results to the organisers for evaluation. As shown in Fig. 5, we compare the proposed RetinaFace with other 24 state-of-the-art face detection algorithms (i.e. Multi-scale Cascade CNN [60], Two-stage CNN [60], ACF-WIDER [58], Faceness-WIDER [59], Multitask Cascade CNN [66], CMS-RCNN [72], LDCF+ [37], HR [23], Face R-CNN [54], ScaleFace [61], SSH [36], SFD [68], Face R-FCN [57], MSCNN [4], FAN [56], Zhu et al. [71], PyramidBox [49], FDNet [63], SRN [8], FANet [65], DSFD [27], DFS [50], VIM-FD [69], ISRN [67]). Our approach outperforms these state-of-the-art methods in terms of AP. More specifically, RetinaFace produces the best AP in all subsets of both validation and test sets, i.e., 96.9% (Easy), 96.1% (Medium) and 91.8% (Hard) for validation set, and 96.3% (Easy), 95.6% (Medium) and 91.4% (Hard) for test set. Compared to the recent best performed method [67], RetinaFace sets up a new impressive record (91.4% v.s. 90.3%) on the Hard subset which contains a large number of tiny faces.

按照WIDER Face数据集上的标准评估方法，我们只在训练集上训练模型，在验证和测试集上进行测试。为得到测试集的评估结果，我们将检测结果提交给组织者以进行评估。如图5所示，我们将提出的RetinaFace与其他24种目前最好的人脸检测算法进行比较。我们的方法在AP上超过了这些算法。更具体的，RetinaFace在验证集和测试集上都得到了最好的AP结果，即验证集96.9% (Easy), 96.1% (Medium)和91.8% (Hard), 测试集96.3% (Easy), 95.6% (Medium)和91.4%。与最近最好的方法进行比较，RetinaFace在困难子集上竖立了一个新的记录，这个子集中包含了大量微小人脸。

In Fig. 6, we illustrate qualitative results on a selfie with dense faces. RetinaFace successfully finds about 900 faces (threshold at 0.5) out of the reported 1, 151 faces. Besides accurate bounding boxes, the five facial landmarks predicted by RetinaFace are also very robust under the variations of pose, occlusion and resolution. Even though there are some failure cases of dense face localisation under heavy occlusion, the dense regression results on some clear and large faces are good and even show expression variations.

在图6中，我们在一个密集人脸的自拍中给出了定性结果。RetinaFace在1151个人脸中，成功的发现了大约900人脸（阈值为0.5）。除了准确的边界框，RetinaFace预测的5个人脸关键点也非常稳健，这是在姿态、遮挡和分辨率的变化下得到的。虽然在严重遮挡的情况下，密集人脸定位有一些失败情况，在一些清晰大型人脸上的密集回归结果是很好的，甚至展示出了表情的变化。

Figure 5. Precision-recall curves on the WIDER FACE validation and test subsets.

Figure 6. RetinaFace can find around 900 faces (threshold at 0.5) out of the reported 1151 people, by taking advantages of the proposed joint extra-supervised and self-supervised multi-task learning. Detector confidence is given by the colour bar on the right. Dense localisation masks are drawn in blue. Please zoom in to check the detailed detection, alignment and dense regression results on tiny faces.

### 4.5. Five Facial Landmark Accuracy

To evaluate the accuracy of five facial landmark localisation, we compare RetinaFace with MTCNN [66] on the AFLW dataset [26] (24,386 faces) as well as the WIDER FACE validation set (18.5k faces). Here, we employ the face box size ($\sqrt{W × H}$) as the normalisation distance. As shown in Fig. 7(a), we give the mean error of each facial landmark on the AFLW dataset [73]. RetinaFace significantly decreases the normalised mean errors (NME) from 2.72% to 2.21% when compared to MTCNN. In Fig. 7(b), we show the cumulative error distribution (CED) curves on the WIDER FACE validation set. Compared to MTCNN, RetinaFace significantly decreases the failure rate from 26.31% to 9.37% (the NME threshold at 10%).

为评估五点人脸关键点的预测准确率，我们将RetinaFace与MTCNN在AFLW数据集(24386人脸)和WIDER Face验证集(18.5k人脸)上进行了比较。这里，我们采用人脸框的大小($\sqrt{W × H}$)作为归一化距离。如图7(a)所示，我们给出每个人脸关键点在AFLW数据集上的平均误差。RetinaFace与MTCNN相比，将归一化平均误差(NME)从2.72%显著降低到2.21%。在图7(b)中，我们给出了在WIDER Face验证集上的累积误差分布(CED)曲线。与MTCNN相比，RetinaFace明显降低了错误率，从26.31%到9.37%（NME阈值为10%）。

Figure 7. Qualitative comparison between MTCNN and RetinaFace on five facial landmark localisation. (a) AFLW (b) WIDER FACE validation set.

### 4.6. Dense Facial Landmark Accuracy

Besides box and five facial landmarks, RetinaFace also outputs dense face correspondence, but the dense regression branch is trained by self-supervised learning only. Following [12, 70], we evaluate the accuracy of dense facial landmark localisation on the AFLW2000-3D dataset [75] considering (1) 68 landmarks with the 2D projection coordinates and (2) all landmarks with 3D coordinates. Here, the mean error is still normalised by the bounding box size [75]. In Fig. 8(a) and 8(b), we give the CED curves of state-of-the-art methods [12, 70, 75, 25, 3] as well as RetinaFace. Even though the performance gap exists between supervised and self-supervised methods, the dense regression results of RetinaFace are comparable with these state-of-the-art methods. More specifically, we observe that (1) five facial landmarks regression can alleviate the training difficulty of dense regression branch and significantly improve the dense regression results. (2) using single-stage features (as in RetinaFace) to predict dense correspondence parameters is much harder than employing (Region of Interest) RoI features (as in Mesh Decoder [70]). As illustrated in Fig. 8(c), RetinaFace can easily handle faces with pose variations but has difficulty under complex scenarios. This indicates that mis-aligned and over-compacted feature representation (1 × 1 × 256 in RetinaFace) impedes the single-stage framework achieving high accurate dense regression outputs. Nevertheless, the projected face regions in the dense regression branch still have the effect of attention [56] which can help to improve face detection as confirmed in the section of ablation study.

除了边界框和五个人脸关键点，RetinaFace还输出密集人脸对应性，但密集回归分支是只用自监督学习训练的。按照[12,70]的思想，我们在AFLW2000-3D数据集上评估密集人脸关键点定位的准确率，考虑以下两种情况：(1)68个关键点和2D投影坐标，(2)所有关键点和3D坐标。这里，平均误差仍然用边界框大小归一化。在图8(a)和8(b)中，我们给出了目前最好方法和RetinaFace的CED曲线。即使监督方法和自监督方法之间存在性能差距，RetinaFace密集回归的结果与目前最好的方法是相似的。具体的，我们观察到：(1)五点关键点回归可以缓解密集回归分支的训练难度，显著改进密集回归的结果；(2)使用单阶段特征（就像RetinaFace一样）来预测密集对应性参数，比使用RoI特征（如Mesh Decoder[70]）要难的多。如图8(c)中所示，RetinaFace可以很容易处理姿态变化的人脸，但很难处理复杂场景的情况。这说明，错误对齐和过度紧凑的特征表示(1 × 1 × 256 in RetinaFace)阻碍了单阶段框架取得更准确的密集回归结果。尽管如此，密集回归分支的投影人脸区域仍然注意力的效果，这可以帮助改进人脸检测的结果，这也在分离试验的部分得到了验证。

Figure 8. CED curves on AFLW2000-3D. Evaluation is performed on (a) 68 landmarks with the 2D coordinates and (b) all landmarks with 3D coordinates. In (c), we compare the dense regression results from RetinaFace and Mesh Decoder [70]. RetinaFace can easily handle faces with pose variations but has difficulty to predict accurate dense correspondence under complex s cenarios.

### 4.7. Face Recognition Accuracy 人脸识别准确率

Face detection plays a crucial role in robust face recognition but its effect is rarely explicitly measured. In this paper, we demonstrate how our face detection method can boost the performance of a state-of-the-art publicly available face recognition method, i.e. ArcFace [11]. ArcFace [11] studied how different aspects in the training process of a deep convolutional neural network (i.e., choice of the training set, the network and the loss function) affect large scale face recognition performance. However, ArcFace paper did not study the effect of face detection by applying only the MTCNN [66] for detection and alignment. In this paper, we replace MTCNN by RetinaFace to detect and align all of the training data (i.e. MS1M [18]) and test data (i.e. LFW [24], CFP-FP [46], AgeDB-30 [35] and IJBC [34]), and keep the embedding network (i.e. ResNet100 [21]) and the loss function (i.e. additive angular margin) exactly the same as ArcFace.

人脸检测在人脸识别中扮演了关键的角色，但其效果很少有直接的衡量。本文中，我们证明了，我们的人脸检测方法，可以提升目前最好的人脸识别方法的性能，即ArcFace。ArcFace研究的是训练过程中DCNN的不同方面（即，训练集，网络和损失函数的选择），是怎样影响大规模人脸识别的性能的。但是，ArcFace文章并没有研究人脸检测效果的影响，只采用了MTCNN进行检测和对齐。本文中，我们将MTCNN替代成RetinaFace，进行所有训练数据(MS1M[18])和测试数据(LFW [24], CFP-FP [46], AgeDB-30 [35] and IJBC [34])的检测和对齐，维持骨干网络(ResNet100[21])和损失函数(additive angular margin)与ArcFace一样。

In Tab. 4, we show the influence of face detection and alignment on deep face recognition (i.e. ArcFace) by comparing the widely used MTCNN [66] and the proposed RetinaFace. The results on CFP-FP, demonstrate that RetinaFace can boost ArcFace’s verification accuracy from 98.37% to 99.49%. This result shows that the performance of frontal-profile face verification is now approaching that of frontal-frontal face verification (e.g. 99.86% on LFW).

在表4中，我们展示了人脸检测和对齐对人脸检测(ArcFace)的影响，比较了MTCNN和RetinaFace。在CFP-FP上的结果证明了，RetinaFace可以提升ArcFace的验证准确率，从98.37%到99.49%。这个结果说明，正面-侧面人脸验证的性能，现在已经接近正面-正面人脸验证的性能(99.86% on LFW)。

Table 4. Verification performance (%) of different methods on LFW, CFP-FP and AgeDB-30.

Methods | LFW | CFP-FP | AgeDB-30
--- | --- | --- | ---
MTCNN+ArcFace [11] | 99.83 | 98.37 | 98.15
RetinaFace+ArcFace | 99.86 | 99.49 | 98.60

In Fig. 9, we show the ROC curves on the IJB-C dataset as well as the TAR for FAR= 1e−6 at the end of each legend. We employ two tricks (i.e. flip test and face detection score to weigh samples within templates) to progressively improve the face verification accuracy. Under fair comparison, TAR (at FAR= 1e−6) significantly improves from 88.29% to 89.59% simply by replacing MTCNN with RetinaFace. This indicates that (1) face detection and alignment significantly affect face recognition performance and (2) RetinaFace is a much stronger baseline than MTCNN for face recognition applications.

在图9中，我们给出了IJB-C数据集上的ROC曲线，图中末尾给出了FAR=1e-6下的TAR。我们采用了两个技巧（即，翻转测试和人脸验证分数，来在模板中评估样本），来渐进的改进人脸验证准确率。在公平比较下，将MTCNN替换成RetinaFace，TAR (at FAR= 1e−6)从88.26%提高到89.59%。这说明，(1)人脸检测和对齐可以显著影响人脸识别的性能，(2)RetinaFace是一个比MTCNN更强的基准。

Figure 9. ROC curves of 1:1 verification protocol on the IJB-C dataset. “+F” refers to flip test during feature embedding and “+S” denotes face detection score used to weigh samples within templates. We also give TAR for FAR= 1e − 6 at the end of the each legend.

### 4.8. Inference Efficiency

During testing, RetinaFace performs face localisation in a single stage, which is flexible and efficient. Besides the above-explored heavy-weight model (ResNet-152, size of 262MB, and AP 91.8% on the WIDER FACE hard set), we also resort to a light-weight model (MobileNet-0.25 [22], size of 1MB, and AP 78.2% on the WIDER FACE hard set) to accelerate the inference.

在测试时，RetinaFace以单阶段进行人脸定位，灵活又高效。除了上述重型模型(ResNet-152, 262M, 在WIDER Face困难集的AP为91.8%)，我们还用轻量级模型(MobileNet-0.25, 1M, 在WIDER Face困难集上AP为78.2%)加速推理。

For the light-weight model, we can quickly reduce the data size by using a 7 × 7 convolution with stride=4 on the input image, tile dense anchors on $P_3$, $P_4$ and $P_5$ as in [36], and remove deformable layers. In addition, the first two convolutional layers initialised by the ImageNet pre-trained model are fixed to achieve higher accuracy.

对于轻量级模型，我们可以很快降低数据规模，对输入图像使用步长为4的7×7卷积，像[36]一样在$P_3$, $P_4$和$P_5$中平铺密集锚框，去除可变形层。另外，前两个以ImageNet预训练模型初始化的卷积层固定下来，以得到更高的准确率。

Tab. 5 gives the inference time of two models with respect to different input sizes. We omit the time cost on the dense regression branch, thus the time statistics are irrelevant to the face density of the input image. We take advantage of TVM [7] to accelerate the model inference and timing is performed on the NVIDIA Tesla P40 GPU, Intel i7-6700K CPU and ARM-RK3399, respectively. RetinaFace-ResNet-152 is designed for highly accurate face localisation, running at 13 FPS for VGA images (640 × 480). By contrast, RetinaFace-MobileNet-0.25 is designed for highly efficient face localisation which demonstrates considerable real-time speed of 40 FPS at GPU for 4K images (4096 × 2160), 20 FPS at multi-thread CPU for HD images (1920 × 1080), and 60 FPS at single-thread CPU for VGA images (640 × 480). Even more impressively, 16 FPS at ARM for VGA images (640 × 480) allows for a fast system on mobile devices.

表5给出了两个模型在不同输入大小下的推理时间。我们忽略了密集回归分支上的时间耗费，所以时间统计值与输入图像中的人脸密度是无关的。我们利用TVM来加速模型推理，计时平台分别为NVIDIA Tesla P40 GPU, Intel i7-6700K CPU and ARM-RK3399。RetinaFace-ResNet-152的设计是为了高准确率人脸定位，对于VGA图像(640 × 480)运行速度为13FPS。比较之下，RetinaFace-MobileNet-0.25是高效人脸定位模型，在GPU上的4K图像(4096 × 2160)达到了40FPS的实时速度，在CPU上对高清图像(1920 × 1080)达到了20FPS的速度，在单核CPU上对VGA图像(640×480)达到了60FPS的速度。更令人印象深刻的是，在ARM上对VGA图像也达到了16FPS的速度，所以在移动设备上也可以实现一个快速系统。

Table 5. Inference time (ms) of RetinaFace with different backbones (ResNet-152 and MobileNet-0.25) on different input sizes (VGA@640x480, HD@1920x1080 and 4K@4096x2160). “CPU-1” and “CPU-m” denote single-thread and multi-thread test on the Intel i7-6700K CPU, respectively. “GPU” refers to the NVIDIA Tesla P40 GPU and “ARM” platform is RK3399(A72x2).

Backbones | VGA | HD | 4K
--- | --- | --- | ---
ResNet-152 (GPU) | 75.1 | 443.2 | 1742
MobileNet-0.25 (GPU) | 1.4 | 6.1 | 25.6
MobileNet-0.25 (CPU-m) | 5.5 | 50.3 | -
MobileNet-0.25 (CPU-1) | 17.2 | 130.4 | -
MobileNet-0.25 (ARM) | 61.2 | 434.3 | -

## 5. Conclusions

We studied the challenging problem of simultaneous dense localisation and alignment of faces of arbitrary scales in images and we proposed the first, to the best of our knowledge, one-stage solution (RetinaFace). Our solution outperforms state of the art methods in the current most challenging benchmarks for face detection. Furthermore, when RetinaFace is combined with state-of-the-art practices for face recognition it obviously improves the accuracy. The data and models have been provided publicly available to facilitate further research on the topic.

我们研究同时进行图像中任意尺度人脸密集定位和对齐的挑战性问题，提出了第一个单阶段的方法(RetinaFace)。我们的方法，在目前最有挑战性的基准测试中，超过了目前最好的人脸检测方法。而且，当RetinaFace与目前最好的人脸识别方法结合使用时，明显改进了其准确率。数据和模型已经开源，以方便这个课题的未来研究。