# Convolutional Pose Machines

Shih-En Wei et al. The Robotics Institute Carnegie Mellon University

## Abstract

Pose Machines provide a sequential prediction framework for learning rich implicit spatial models. In this work we show a systematic design for how convolutional networks can be incorporated into the pose machine framework for learning image features and image-dependent spatial models for the task of pose estimation. The contribution of this paper is to implicitly model long-range dependencies between variables in structured prediction tasks such as articulated pose estimation. We achieve this by designing a sequential architecture composed of convolutional networks that directly operate on belief maps from previous stages, producing increasingly refined estimates for part locations, without the need for explicit graphical model-style inference. Our approach addresses the characteristic difficulty of vanishing gradients during training by providing a natural learning objective function that enforces intermediate supervision, thereby replenishing back-propagated gradients and conditioning the learning procedure. We demonstrate state-of-the-art performance and outperform competing methods on standard benchmarks including the MPII, LSP, and FLIC datasets.

Pose Machines是一种可以学习富内在空间建模的序贯预测框架。本文中我们展示了一种用于姿态估计的系统性设计，其中卷积网络在pose machine框架中用于学习图像特征和与依赖图像的空域模型。本文的贡献是在结构化的预测任务如铰接姿态预测中对变量进行长距离依赖关系建模。我们设计了一种序贯架构，由卷积网络构成，网络直接对前一阶段的信念图运算，得到部位位置高度精炼的估计，不需要显式的模型风格的图形推理。我们的方法在处理训练过程中的消失梯度问题时，是给出了一种自然的学习目标函数，函数中强行加入了中间的监督，所以会补充反向传播的梯度，自动调节学习过程。我们给出了目前最好的结果，在标准的基准测试中超过了之前的方法，包括MPII, LSP和FLIC数据集。

## 1. Introduction 引言

We introduce Convolutional Pose Machines (CPMs) for the task of articulated pose estimation. CPMs inherit the benefits of the pose machine [29] architecture—the implicit learning of long-range dependencies between image and multi-part cues, tight integration between learning and inference, a modular sequential design—and combine them with the advantages afforded by convolutional architectures: the ability to learn feature representations for both image and spatial context directly from data; a differentiable architecture that allows for globally joint training with backpropagation; and the ability to efficiently handle large training datasets.

我们提出了卷积姿态机(CPM)进行铰接姿态估计的任务。CPMs继承了姿态机[29]架构的优势，即隐式的学习图像和多部件提示间的长距离依赖关系，学习与推理间的紧密整合，模块化的序贯化的设计；还与卷积架构所提供的优势结合了起来，即对于图像和直接从数据得到的空域上下文都可以学习特征表示的能力，一种可以用反向传播进行全局联合训练的可微分架构，以及可以高效的处理大型训练数据集的能力。

CPMs consist of a sequence of convolutional networks that repeatedly produce 2D belief maps (We use the term belief in a slightly loose sense, however the belief maps described are closely related to beliefs produced in message passing inference in graphical models. The overall architecture can be viewed as an unrolled mean-field message passing inference algorithm [31] that is learned end-to-end using backpropagation.) for the location of each part. At each stage in a CPM, image features and the belief maps produced by the previous stage are used as input. The belief maps provide the subsequent stage an expressive non-parametric encoding of the spatial uncertainty of location for each part, allowing the CPM to learn rich image-dependent spatial models of the relationships between parts. Instead of explicitly parsing such belief maps either using graphical models [28, 38, 39] or specialized post-processing steps [38, 40], we learn convolutional networks that directly operate on intermediate belief maps and learn implicit image-dependent spatial models of the relationships between parts. The overall proposed multi-stage architecture is fully differentiable and therefore can be trained in an end-to-end fashion using backpropagation.

CPMs包括一系列卷积网络，网络对于每个部位的位置都不断的生成2D信念图。在一个CPM的每个阶段，图像特征和从前一阶段生成的信念图用作输入。信念图为后续的阶段提供了一个每个部位位置的空域不确定性的有表现力的非参数编码，使CPM可以学习丰富的依赖于图像的空域模型，以处理部位间的关系。我们没有使用图形工具[28,38,39]或专门的后处理步骤来解析这些信念图，而是学习卷积网络以直接在中间的信念图中运算，学习内在依赖图像的空间模型以处理部位间的关系。提出的多阶段架构是可以全微分的，所以可以使用反向传播进行端到端的训练。

At a particular stage in the CPM, the spatial context of part beliefs provide strong disambiguating cues to a subsequent stage. As a result, each stage of a CPM produces belief maps with increasingly refined estimates for the locations of each part (see Figure 1). In order to capture long-range interactions between parts, the design of the network in each stage of our sequential prediction framework is motivated by the goal of achieving a large receptive field on both the image and the belief maps. We find, through experiments, that large receptive fields on the belief maps are crucial for learning long range spatial relationships and result in improved accuracy.

在CPM的一个特定阶段，部位信念值的空域上下文会将毫无歧义的提示送入下一阶段。结果是，CPM的每个阶段都会生成信念图，这是每个部位位置的估计，而且越来越精炼（见图1）。为捕捉部位间的长距离互动关系，序贯预测框架中每个阶段网络的设计，都是要尽量得到图像和信念图更大的感受野。我们通过试验发现，信念图中大的感受野对于学习长距离空间关系是非常关键的，可以得到更好的结果。

Figure 1: A Convolutional Pose Machine consists of a sequence of predictors trained to make dense predictions at each image location. Here we show the increasingly refined estimates for the location of the right elbow in each stage of the sequence. (a) Predicting from local evidence often causes confusion. (b) Multi-part context helps resolve ambiguity. (c) Additional iterations help converge to a certain solution. Input image -> (a) Stage 1 -> (b) Stage 2 -> (c) Stage 3 

图1：CPM由一系列预测器组成，每个预测器都训练在每个图像位置进行密集预测。这里我们给出序列的每个阶段对右肘位置的越来越精确的估计。(a)由局部线索给出的预测经常导致混淆；(b)多部位上下文帮助解决疑义；(c)额外的迭代帮助收敛到一个确定的解。

Composing multiple convolutional networks in a CPM results in an overall network with many layers that is at risk of the problem of vanishing gradients [4, 5, 10, 12] during learning. This problem can occur because back-propagated gradients diminish in strength as they are propagated through the many layers of the network. While there exists recent work (New results have shown that using skip connections with identity mappings [11] in so-called residual units also aids in addressing vanishing gradients in “very deep” networks. We view this method as complementary and it can be noted that our modular architecture easily allows us to replace each stage with the appropriate residual network equivalent.) which shows that supervising very deep networks at intermediate layers aids in learning [20, 36], they have mostly been restricted to classification problems. In this work, we show how for a structured prediction problem such as pose estimation, CPMs naturally suggest a systematic framework that replenishes gradients and guides the network to produce increasingly accurate belief maps by enforcing intermediate supervision periodically through the network. We also discuss different training schemes of such a sequential prediction architecture.

将CPM中的多个卷积网络组成一个整体很多层的网络，这在训练的时候有梯度消失的问题[4,5,10,12]。这个问题的出现是因为反向传播的梯度在传播通过很多层的时候，强度逐渐消失。最近有工作表明在中间层监督非常深的网络有助于学习[20,36]（在残差单元中使用恒等映射的跳跃连接[11]也有助于在非常深的网络中解决消失梯度问题，我们视这种方法为补充，因为我们的模块化结构可以轻松的将每个阶段替换为适当的残差网络），这大多是应用于分类问题。在本文中，我们证明了对于一个结构化的预测问题，如姿态估计，CPMs很自然的引出一个系统性的框架可以补充梯度，通过在网络中周期性的强加入中间监督，来引导网络生成越来越精确的信念图。我们还讨论了这样的序贯预测架构的不同训练方案。

Our main contributions are (a) learning implicit spatial models via a sequential composition of convolutional architectures and (b) a systematic approach to designing and training such an architecture to learn both image features and image-dependent spatial models for structured prediction tasks, without the need for any graphical model style inference. We achieve state-of-the-art results on standard benchmarks including the MPII, LSP, and FLIC datasets, and analyze the effects of jointly training a multi-staged architecture with repeated intermediate supervision.

我们的主要贡献如下：(a)通过卷积架构的序列化组合学习出了一个内在空间模型，(b)一个系统性的方法来设计并训练这样一种架构，以学习图像特征和依赖于图像的空间模型，进行结构化预测任务，不需要任何图形模型类型的推理。我们在标准基准测试中得到了目前最好的结果，包括MPII，LSP和FLIC数据集，还分析了使用重复的中间监督来联合训练一个多阶段架构的效果。

## 2. Related Work 相关工作

The classical approach to articulated pose estimation is the pictorial structures model [2, 3, 9, 14, 26, 27, 30, 43] in which spatial correlations between parts of the body are expressed as a tree-structured graphical model with kinematic priors that couple connected limbs. These methods have been successful on images where all the limbs of the person are visible, but are prone to characteristic errors such as double-counting image evidence, which occur because of correlations between variables that are not captured by a tree-structured model. The work of Kiefel et al. [17] is based on the pictorial structures model but differs in the underlying graph representation. Hierarchical models [35, 37] represent the relationships between parts at different scales and sizes in a hierarchical tree structure. The underlying assumption of these models is that larger parts (that correspond to full limbs instead of joints) can often have discriminative image structure that can be easier to detect and consequently help reason about the location of smaller, harder-to-detect parts. Non-tree models [8, 16, 19, 33, 42] incorporate interactions that introduce loops to augment the tree structure with additional edges that capture symmetry, occlusion and long-range relationships. These methods usually have to rely on approximate inference during both learning and at test time, and therefore have to trade off accurate modeling of spatial relationships with models that allow efficient inference, often with a simple parametric form to allow for fast inference. In contrast, methods based on a sequential prediction framework [29] learn an implicit spatial model with potentially complex interactions between variables by directly training an inference procedure, as in [22, 25, 31, 41].

铰接姿态估计的经典方法是pictorial structures模型[2, 3, 9, 14, 26, 27, 30, 43]，其中身体部位的空间关联表示为树状的图形模型，根据的是运动学的先验知识配对连着的四肢。这些方法在人的所有肢体都可见时效果很好，但是容易犯一些典型的错误，如重复计算图像证据，这是因为变量间的关联没有纳入到树形模型中。Kiefel等[17]的工作是基于pictorial structures模型的，但底层的图表示不同。层次化的模型[35,37]将不同尺度不同大小部位间的关系在一个层次化的树形结构中表示。这些模型的潜在假设是更大的部位（这对应着整个肢体而不是关节）经常有着可以区分的图像结构，更容易检测，结果可以帮助推理那些更小更难检测的部位的位置。非树模型[8, 16, 19, 33, 42]包含了引入环形的元素，以额外的边来增强树形结构，可以捕捉到对称性、遮挡或者长距关系。这些方法通常在学习和测试时必须依赖近似推理，所以必须在精确的空间关系建模和能高效推理的模型之间做取舍，经常有一个简单的参数形式来能够快速推理。而基于序贯预测框架[29]的方法学习了一个隐式的空间模型，在变量之间有潜在的复杂互动关系，这是通过直接训练一个推理过程得到的，见[22, 25, 31, 41]。

There has been a recent surge of interest in models that employ convolutional architectures for the task of articulated pose estimation [6, 7, 23, 24, 28, 38, 39]. Toshev et al. [40] take the approach of directly regressing the Cartesian coordinates using a standard convolutional architecture [18]. Recent work regresses image to confidence maps, and resort to graphical models, which require hand-designed energy functions or heuristic initialization of spatial probability priors, to remove outliers on the regressed confidence maps. Some of them also utilize a dedicated network module for precision refinement [28, 38]. In this work, we show the regressed confidence maps are suitable to be inputted to further convolutional networks with large receptive fields to learn implicit spatial dependencies without the use of hand designed priors, and achieve state-of-the-art performance over all precision region without careful initialization and dedicated precision refinement. Pfister et al. [24] also used a network module with large receptive field to capture implicit spatial models. Due to the differentiable nature of convolutions, our model can be globally trained, where Tompson et al. [39] and Steward et al. [34] also discussed the benefit of joint training.

近年来采用卷积架构来进行铰接姿态估计任务的研究突然增多[6, 7, 23, 24, 28, 38, 39]。[40]用一个标准的卷积架构[18]来直接回归笛卡尔坐标。最近的工作用图像回归置信度图，并使用图形模型，这需要手工设计能量函数或根据直觉来初始化先验空间概率，来去除回归得到的置信度图中的离群值。一些人也使用专用网络模块来精炼精度[28,38]。在本文中，我们证明了回归的置信度图适合输入到更多的大感受野卷积网络中，以学习隐式的空间依赖关系，而不用使用手工设计的先验，还能在所有精度区域中得到目前最好的结果，也不用小心的初始化，也不用可以提炼精度。Pfister等[24]也使用了一个大感受野的网络模块来捕捉隐式空间模型。由于卷积的可微分性质，我们的模型可以进行全局训练，其中Topmson等[39]和Steward等[34]也讨论了联合训练的优点。

Carreira et al. [6] train a deep network that iteratively improves part detections using error feedback but use a cartesian representation as in [40] which does not preserve spatial uncertainty and results in lower accuracy in the high-precision regime. In this work, we show how the sequential prediction framework takes advantage of the preserved uncertainty in the confidence maps to encode the rich spatial context, with enforcing the intermediate local supervisions to address the problem of vanishing gradients.

Carreira等[6]训练了一个深度网络使用误差反馈反复改进部位检测，但是像[40]一样使用了笛卡尔表示，这没有保存空间不确定性，结果准确率在高精度区域降低。本文中，我们展示了序贯预测框架是怎样利用置信度图中保存的不确定性来对富空间上下文进行编码，而且通过强制中间局部监督来解决消失梯度的问题。

## 3. Method 方法

### 3.1. Pose Machines 姿态机

We denote the pixel location of the p-th anatomical landmark (which we refer to as a part), $Y_p ∈ Z ⊂ R^2$, where Z is the set of all (u,v) locations in an image. Our goal is to predict the image locations $Y = (Y_1, ..., Y_P)$ for
all P parts. A pose machine [29] (see Figure 2a and 2b) consists of a sequence of multi-class predictors, $g_t (·)$, that are trained to predict the location of each part in each level of the hierarchy. In each stage t ∈ {1...T}, the classifiers $g_t$ predict beliefs for assigning a location to each part $Y_p = z, ∀z ∈ Z$, based on features extracted from the image at the location z denoted by $x_z ∈ R^d$ and contextual information from the preceding classifier in the neighborhood around each $Y_p$ in stage t. A classifier in the first stage t = 1, therefore produces the following belief values:

我们将第p个部位的像素位置表示为$Y_p ∈ Z ⊂ R^2$，其中Z是图像的所有位置(u,v)的集合。我们的目的是预测所有P个部位的图像位置$Y = (Y_1, ..., Y_P)$。姿态机[29]（见图2a和图2b）包括一系列多类别预测器$g_t (·)$，经过训练预测每个部位的位置。在每个阶段t ∈ {1...T}，分类器$g_t$将位置指定给每个部位$Y_p = z, ∀z ∈ Z$来预测信念值，预测是基于从图像位置z提取出的特征$x_z ∈ R^d$，和从上一个分类器来的在每个$Y_p$周围在阶段t的上下文信息。在第一阶段t=1的一个分类器，会产生如下的信念值：

$$g_1(x_z) → \{b_1^p (Y_p = z)\}_{p∈\{0...P\}}$$(1)

where $b_1^p (Y_p = z)$ is the score predicted by the classifier $g_1$ for assigning the $p^{th}$ part in the first stage at image location z. We represent all the beliefs of part p evaluated at every location $z = (u,v)^T$ in the image as $b^p_t ∈ R^{w×h}$, where w and h are the width and height of the image, respectively. That is,

其中$b_1^p (Y_p = z)$是分类器$g_1$，在第一阶段将图像位置z指定给第p个部位，预测得到的分数。我们将部位p在图像每个位置$z = (u,v)^T$评估得到的所有信念值表示为$b^p_t ∈ R^{w×h}$，其中w和h分别是图像的宽和高，这就是：

$$b_t^p [u,v] = b_t^p(Y_p = z)$$(2)

For convenience, we denote the collection of belief maps for all the parts as $b_t ∈ R^{w×h×(P+1)}$ (P parts plus one for background).

方便起见，我们将所有部位信念图的集合表示为$b_t ∈ R^{w×h×(P+1)}$（P个部位加上1表示背景）。

In subsequent stages, the classifier predicts a belief for assigning a location to each part $Y_p = z, ∀z ∈ Z$, based on (1) features of the image data $x^t_z ∈ R^d$ again, and (2) contextual information from the preceeding classifier in the neighborhood around each $Y_p$:

在接下来的阶段里，分类器将一个位置指定给每个部位$Y_p = z, ∀z ∈ Z$来预测信念值，这是基于 (1)再次依据图像特征数据$x^t_z ∈ R^d$，(2)从前面的分类在每个$Y_p$点的上下文信息：

$$g_t(x'_z, ψ_t(z, b_{t-1})) → \{ b_t^p(Y_p = z) \}_{p∈\{0...P+1\}}$$(3)

where ψ t>1 (·) is a mapping from the beliefs $b_{t−1}$ to context features. In each stage, the computed beliefs provide an increasingly refined estimate for the location of each part. Note that we allow image features $x'_z$ for subsequent stage to be different from the image feature used in the first stage x. The pose machine proposed in [29] used boosted random forests for prediction ({$g_t$}), fixed hand-crafted image features across all stages (x' = x), and fixed hand-crafted context feature maps ($ψ_t(·)$) to capture spatial context across all stages.

其中$ψ_{t>1}(·)$是从信念值$b_{t−1}$到上下文特征的映射。在每个阶段，计算的信念值给出了一个不断优化的每个部位位置的估计。注意我们允许后续阶段的图像特征$x'_z$与第一阶段使用的图像特征不一样。[29]提出的姿态机使用boosted随机森林来预测({$g_t$})，在所有阶段使用固定手工设计的图像特征(x' = x)，固定手工设计的上下文特征图($ψ_t(·)$)来在所有阶段捕捉空间上下文。

Figure 2: Architecture and receptive fields of CPMs. We show a convolutional architecture and receptive fields across layers for a CPM with any T stages. The pose machine [29] is shown in insets (a) and (b), and the corresponding convolutional networks are shown in insets (c) and (d). Insets (a) and (c) show the architecture that operates only on image evidence in the first stage. Insets (b) and (d) shows the architecture for subsequent stages, which operate both on image evidence as well as belief maps from preceding stages. The architectures in (b) and (d) are repeated for all subsequent stages (2 to T). The network is locally supervised after each stage using an intermediate loss layer that prevents vanishing gradients during training. Below in inset (e) we show the effective receptive field on an image (centered at left knee) of the architecture, where the large receptive field enables the model to capture long-range spatial dependencies such as those between head and knees. (Best viewed in color.)

图2：CPM的架构和感受野。我们给出了CPM在任意T阶段的跨层卷积架构和感受野。姿态机[29]如(a)和(b)所示，相应的卷积网络如(c)(d)所示。(a)(c)所示的是第一阶段的架构。(b)(d)是后续阶段的架构，对图像证据和前一阶段的信念图进行运算。(b)(d)的架构在后续所有的阶段（第2到T阶段）不断重复。网络是在每个阶段之后都是用一个中间损失层局部监督的，这会防止训练时的梯度消失问题。(e)中所示的是这个架构在一幅图像中的有效感受野（中心在左膝盖上），其中大感受野可以使模型捕捉到长距离空间依赖关系，比如在头和膝盖之间的关系。

### 3.2. Convolutional Pose Machines 卷积姿态机

We show how the prediction and image feature computation modules of a pose machine can be replaced by a deep convolutional architecture allowing for both image and contextual feature representations to be learned directly from data. Convolutional architectures also have the advantage of being completely differentiable, thereby enabling end-to-end joint training of all stages of a CPM. We describe our design for a CPM that combines the advantages of deep convolutional architectures with the implicit spatial modeling afforded by the pose machine framework.

我们说明了，姿态机的预测和图像特征计算模块，可以替换为深度卷积架构，从数据中直接学习图像和上下文特征表示。卷积架构完全可微分，这个优势使得所有阶段端到端的联合训练可以进行。我们描述一下我们的CPM设计，这结合了深度卷积架构的优势和姿态机架构的内在空间建模的优势。

#### 3.2.1 Keypoint Localization Using Local Image Evidence 使用图像局部证据的关键点定位

The first stage of a convolutional pose machine predicts part beliefs from only local image evidence. Figure 2c shows the network structure used for part detection from local image evidence using a deep convolutional network. The evidence is local because the receptive field of the first stage of the network is constrained to a small patch around the output pixel location. We use a network structure composed of five convolutional layers followed by two 1 × 1 convolutional layers which results in a fully convolutional architecture [21]. In practice, to achieve certain precision, we normalize input cropped images to size 368×368 (see Section 4.2 for details), and the receptive field of the network shown above is 160 × 160 pixels. The network can effectively be viewed as sliding a deep network across an image and regressing from the local image evidence in each 160 × 160 image patch to a P + 1 sized output vector that represents a score for each part at that image location.

卷积姿态机第一阶段只从图像局部证据中预测部位信念值。图2c展示了用于从局部图像证据进行部位检测的深度卷积网络的架构。图像证据是局部的，因为第一阶段网络的感受野限制到了输出图像位置的小块附近。我们的网络由5个卷积层和2个1×1的卷积层构成，这是一个全卷积架构[21]。在实践中，未得到相应的精度，我们将输入图像大小统一为368×368（详见4.2节），上面这个网络的感受野大小为160×160。这个网络实际上可以看作是，将深度网络滑过一幅图像，从每个160×160大小图像块的图像局部证据中回归得到一个P+1大小的输出向量，代表在这个图像位置每个部位的分数。

#### 3.2.2 Sequential Prediction with Learned Spatial Context Features 用已学到的空间上下文特征进行序贯预测

While the detection rate on landmarks with consistent appearance, such as the head and shoulders, can be favorable, the accuracies are often much lower for landmarks lower down the kinematic chain of the human skeleton due to their large variance in configuration and appearance. The landscape of the belief maps around a part location, albeit noisy, can, however, be very informative. Illustrated in Figure 3, when detecting challenging parts such as right elbow, the belief map for right shoulder with a sharp peak can be used as a strong cue. A predictor in subsequent stages ($g_t>1$) can use the spatial context ($ψ_{t>1}(·)$) of the noisy belief maps in
a region around the image location z and improve its predictions by leveraging the fact that parts occur in consistent geometric configurations. In the second stage of a pose machine, the classifier $g_2$ accepts as input the image features $x^2_z$ and features computed on the beliefs via the feature function ψ for each of the parts in the previous stage. The feature function ψ serves to encode the landscape of the belief maps from the previous stage in a spatial region around the location z of the different parts. For a convolutional pose machine, we do not have an explicit function that computes context features. Instead, we define ψ as being the receptive field of the predictor on the beliefs from the previous stage.

虽然外表一致的部位标志检测率还不错，比如头部和肩部，但对其他一些部位来说，由于其配置和外观变化多样，所以准确率要低得多。一个部位附近的信念图的样子，虽然包含噪声，但是可能信息量很大。如图3所示，当检测有挑战性的部位，如右肘时，右肩的信念图会有很尖锐的峰值，可以用作很强的线索。后续阶段的预测器($g_t>1$)可以使用图像位置z附近区域的含噪信念图的空间上下文($ψ_{t>1}(·)$)，利用各个部位的相对几何配置关系是不变的这个事实，来改进预测。在姿态机的第二阶段，分类器$g_2$的输入为图像特征$x^2_z$，和从信念图中计算得到的特征，信念图主要体现在前一阶段每个部位的特征函数ψ中。特征函数ψ的作用是前一阶段信念图的编码，包括不同部位在不同位置z附近的空间区域的信念图。对于卷积姿态机，我们没有一个显式的函数来计算上下文特征，我们将ψ定义为前一阶段信念图预测器的感受野。

Figure 3: Spatial context from belief maps of easier-to-detect parts can provide strong cues for localizing difficult-to-detect parts. The spatial contexts from shoulder, neck and head can help eliminate wrong (red) and strengthen correct (green) estimations on the belief map of right elbow in the subsequent stages.

图3：易于检测部位的信念图中的空间上下文可以为难于检测部位的定位提供非常有用的线索。从肩部、脖子和头部得到的空间上下文可以在后续阶段右肘部位的检测中，帮助消除信念图中的错误估计（红色）并加强正确（绿色）估计。

The design of the network is guided by achieving a receptive field at the output layer of the second stage network that is large enough to allow the learning of potentially complex and long-range correlations between parts. By simply supplying features on the outputs of the previous stage (as opposed to specifying potential functions in a graphical model), the convolutional layers in the subsequent stage allow the classifier to freely combine contextual information by picking the most predictive features. The belief maps from the first stage are generated from a network that examined the image locally with a small receptive field. In the second stage, we design a network that drastically increases the equivalent receptive field. Large receptive fields can be achieved either by pooling at the expense of precision, increasing the kernel size of the convolutional filters at the expense of increasing the number of parameters, or by increasing the number of convolutional layers at the risk of encountering vanishing gradients during training. Our network design and corresponding receptive field for the subsequent stages (t ≥ 2) is shown in Figure 2d. We choose to use multiple convolutional layers to achieve large receptive field on the 8× downscaled heatmaps, as it allows us to be parsimonious with respect to the number of parameters of the model. We found that our stride-8 network performs as well as a stride-4 one even at high precision region, while it makes us easier to achieve larger receptive fields. We also repeat similar structure for image feature maps to make the spatial context be image-dependent and allow error correction, following the structure of pose machine.

网络的设计要满足以下要求，网络第二阶段的输出层的感受野，要大到可以学习到部位之间的潜在复杂和长距离的相关性。只需要在前一阶段输出时提供特征（与指定潜在的图形模型函数相对），后续阶段的卷积层使分类器可以通过选择最有预测能力的特征，来自由的组合上下文信息。产生第一阶段信念图的网络，其感受野很小，处理的是图像局部的信息。在第二阶段，我们设计的网络急剧增加了相应的感受野。大的感受野可以通过pooling取得，其代价是精度降低，也可以增加卷积核的大小，大家是增加参数数量，或者增加卷积层的数量，代价是在训练时可能遇到梯度消失的问题。后续阶段(t ≥ 2)，我们的网络设计，和对应的感受野大小，如图2d所示。我们选择在8倍下采样的热力图中，使用多个卷积层来得到更大的感受野，这样可以控制模型参数数量。我们发现8步长网络与4步长网络即使在高精度区域也一样出色，而且这使得我们可以得到更大的感受野。我们对图像特征图重复类似的结构，使空间上下文是依赖于图像的，并允许修正误差，遵循姿态机的结构。

We find that accuracy improves with the size of the receptive field. In Figure 4 we show the improvement in accuracy on the FLIC dataset [32] as the size of the receptive field on the original image is varied by varying the architecture without significantly changing the number of parameters, through a series of experimental trials on input images normalized to a size of 304 × 304. We see that the accuracy improves as the effective receptive field increases, and starts to saturate around 250 pixels, which also happens to be roughly the size of the normalized object. This improvement in accuracy with receptive field size suggests that the network does indeed encode long range interactions between parts and that doing so is beneficial. In our best performing setting in Figure 2, we normalize cropped images into a larger size of 368 × 368 pixels for better precision, and the receptive field of the second stage output on the belief maps of the first stage is set to 31 × 31, which is equivalently 400 × 400 pixels on the original image, where the radius can usually cover any pair of the parts. With more stages, the effective receptive field is even larger. In the following section we show our results from up to 6 stages.

我们发现准确率随着感受野的增大而改进。如图4，我们给出了FLIC数据集中准确率的改进与感受野的大小的关系，在原始图像中感受野的变化主要是通过改变网络架构，但参数数量并没有明显改变，输入图像的大小统一为304×304大小，进行了一系列试验。我们看到随着感受野的增大，准确率也不断改进，在大约250像素的时候达到饱和，这也大约是目标的大小。这种随着感受野大小增加的准确率改进说明，网络确实对长距离部位间的关系进行了编码，而且这样是有帮助的。在图2中的最有试验设置中，我们将输入图像大小统一为368×368，以得到更好的精度，第二阶段输出在第一阶段的信念图中的感受野设为31×31，这相当于原始图像中的400×400大小，这样的半径基本可以覆盖任何部位的组合。阶段数更多的话，有效感受野更大。下文中我们会最多给出6个阶段的结果。

Figure 4: Large receptive fields for spatial context. We show that networks with large receptive fields are effective at modeling long-range spatial interactions between parts. Note that these experiments are operated with smaller normalized images than our best setting.

图4：空间上下文的大感受野。我们展示了大感受野的网络可以有效对部位间的长距离空间关系进行建模。注意这些试验与我们的最有设置比，图像分辨率更小。

### 3.3. Learning in Convolutional Pose Machines 卷积姿态机的学习

The design described above for a pose machine results in a deep architecture that can have a large number of layers. Training such a network with many layers can be prone to the problem of vanishing gradients [4, 5, 10] where, as observed by Bradley [5] and Bengio et al. [10], the magnitude of back-propagated gradients decreases in strength with the number of intermediate layers between the output layer and the input layer.

上述姿态机的设计是一个深度架构，层数很多。训练很多层的网络容易陷入梯度消失的问题[4,5,10]，据Bradley[5]和Bengio等[10]人观察，其中反向传播的梯度的幅度随着网络深度的加深逐渐减小甚至消失。

Fortunately, the sequential prediction framework of the pose machine provides a natural approach to training our deep architecture that addresses this problem. Each stage of the pose machine is trained to repeatedly produce the belief maps for the locations of each of the parts. We encourage the network to repeatedly arrive at such a representation by defining a loss function at the output of each stage t that minimizes the $l_2$ distance between the predicted and ideal belief maps for each part. The ideal belief map for a part p is written as $b^p_∗ (Y_p = z)$, which are created by putting Gaussian peaks at ground truth locations of each body part p. The cost function we aim to minimize at the output of each stage at each level is therefore given by:

幸运的是，姿态机的序贯预测框架在训练深度框架的同时给出了自然解决这个问题的方法。姿态机的每个阶段经过训练都重复生成每个部位对应位置的信念图。我们在每个阶段t的输出都定义了损失函数，最小化预测的每个部位和理想的每个部位的信念图的$l_2$距离，这样就鼓励网络重复达到这种表示。部位p的理想信念图用$b^p_∗ (Y_p = z)$表示，是每个部位p的真值位置的高斯峰值。我们在每个层每个阶段的输出要最小化的代价函数由下式给出：

$$f_t = \sum_{p=1}^{P+1} \sum_{z∈Z} ||b_t^p(z) - b_*^p(z)||^2_2$$(4)

The overall objective for the full architecture is obtained by adding the losses at each stage and is given by:

整个框架的总计目标函数就是各个阶段损失函数相加，即：

$$F = \sum_{t=1}^T f_t$$(5)

We use standard stochastic gradient descend to jointly train all the T stages in the network. To share the image feature x' across all subsequent stages, we share the weights of corresponding convolutional layers (see Figure 2) across stages t ≥ 2.

我们使用标准的随机梯度下降方法来联合训练网络中所有的T阶段。为在接下来的所有阶段分享图像特征x'，我们在t≥2的层中分享相应的卷积层（见图2）。

## 4. Evaluation 评估

### 4.1. Analysis 分析

**Addressing vanishing gradients**. The objective in Equation 5 describes a decomposable loss function that operates on different parts of the network(see Figure2). Specifically, each term in the summation is applied to the network after each stage t effectively enforcing supervision in intermediate stages through the network. Intermediate supervision has the advantage that, even though the full architecture can have many layers, it does not fall prey to the vanishing gradient problem as the intermediate loss functions replenish the gradients at each stage.

**解决消失的梯度问题**。式(5)的目标函数是一个可分解的损失函数，是网络不同部分的组合（见图2）。特别的，求和式中的每个项都是在每个阶段中对中间阶段增加了监督后，才作用于网络的，这在网络整个过程中都是如此。中间的监督的优点是，虽然网络整体架构层数非常多，但是不会出现消失梯度的问题，因为中间的损失函数在每个阶段中都补充了梯度。

We verify this claim by observing histograms of gradient magnitude (see Figure 5) at different depths in the architecture across training epochs for models with and without intermediate supervision. In early epochs, as we move from the output layer to the input layer, we observe on the model without intermediate supervision, the gradient distribution is tightly peaked around zero because of vanishing gradients. The model with intermediate supervision has a much larger variance across all layers, suggesting that learning is indeed occurring in all the layers thanks to intermediate supervision. We also notice that as training progresses, the variance in the gradient magnitude distributions decreases pointing to model convergence.

我们观察了架构中不同深度的梯度幅度，在训练过程中有中间监督和没有中间监督两种情况的幅度变化，来验证我们的结论（见图5）。在早期的epoch中，当我们从输入层看到输出层时，我们观察到，在没有中间监督的模型中，由于消失梯度现象的存在，梯度几乎都在零值附近。含有中间监督的模型在所有层中方差都更大一些，意味着所有的层确实都在学习。我们还注意到在训练过程中，梯度幅值方差的分布随着模型收敛而减小。

Figure 5: Intermediate supervision addresses vanishing gradients. We track the change in magnitude of gradients in layers at different depths in the network, across training epochs, for models with and without intermediate supervision. We observe that for layers closer to the output, the distribution has a large variance for both with and without intermediate supervision; however as we move from the output layer towards the input, the gradient magnitude distribution peaks tightly around zero with low variance (the gradients vanish) for the model without intermediate supervision. For the model with intermediate supervision the distribution has a moderately large variance throughout the network. At later training epochs, the variances decrease for all layers for the model with intermediate supervision and remain tightly peaked around zero for the model without intermediate supervision. (Best viewed in color)

图5：中间监督解决消失梯度的问题。我们追踪了梯度幅度的变化，包括网络中不同深度的梯度，在不同训练epoch时的梯度，有中间监督和没有中间监督的模型的梯度。我们观察到对于靠近输出的层，不论有没有中间监督，分布的方差都较大；但是当我们观察输入端时，发现对于没有中间监督的模型，梯度幅度分布基本都分布在0值附近，方差很小（梯度消失了）。对于有中间监督的模型，分布的方差在整个网络中都相对较大。在后面的训练epoch中，对于有中间监督的模型，所有层中的方差都下降了，对于没有中间监督的模型，梯度幅值分布都仅仅围绕在0附近。

**Benefit of end-to-end learning**. We see in Figure 6a that replacing the modules of a pose machine with the appropriately designed convolutional architecture provides a large boost of 42.4 percentage points over the previous approach of [29] in the high precision regime (PCK@0.1) and 30.9 percentage points in the low precision regime (PCK@0.2).

**端到端训练的优点**。我们在图6a中看到，将姿态机算法的模块替换成我们设计的卷积架构，比之前的方法[29]性能有了大幅提升：在高精度区域(PCK@0.1)提升了42.4%，在低精度区域(PCK@0.2)提升了30.9%。

**Comparison on training schemes**. We compare different variants of training the network in Figure 6b on the LSP dataset with person-centric (PC) annotations. To demonstrate the benefit of intermediate supervision with joint training across stages, we train the model in four ways: (i) training from scratch using a global loss function that enforces intermediate supervision (ii) stage-wise; where each stage is trained in a feed-forward fashion and stacked (iii) as same as (i) but initialized with weights from (ii), and (iv) as same as (i) but with no intermediate supervision. We find that network (i) outperforms all other training methods, showing that intermediate supervision and joint training across stage is indeed crucial in achieving good performance. The stagewise training in (ii) saturate at sub-optimal, and the jointly fine-tuning in (iii) improves from this sub-optimal to the accuracy level closed to (i), however with effectively longer training iterations.

**不同训练方案的比较**。我们在图6b中比较了对网络不同训练方案的结果，数据集为LSP，以人为中心的标注。为表现中间监督和各阶段联合训练的优势，我们以4种方式训练模型：(i)带有中间监督的全局损失函数的从头训练；(ii)分阶段训练，每个阶段都用前向的方式训练，然后叠加在一起；(iii)与(i)相同，但是用(ii)中的权重初始化；(iv)与(i)相同，但没有中间监督。我们发现(i)的结果超过了其他的训练方法，表示中间监督和各阶段联合训练确实可以取得最好的结果。(ii)中的分阶段训练在次优解上饱和了，而从这个次优解出发，在(iii)中精调，得到了接近(i)的结果，但是训练过程长了很多。

**Performance across stages**. We show a comparison of performance across each stage on the LSP dataset (PC) in Figure 6c. We show that the performance increases monotonically until 5 stages, as the predictors in subsequent stages make use of contextual information in a large receptive field on the previous stage beliefs maps to resolve confusions between parts and background. We see diminishing returns at the 6th stage, which is the number we choose for reporting our best results in this paper for LSP and MPII datasets.

**分阶段的性能**。我们在图6c中展示了各阶段在LSP数据集(PC)上的性能对比。我们证明了性能单调增长直到5阶段，因为后续阶段的预测器使用了前阶段信念图的大感受野的上下文信息，解决了各部位和背景的混淆问题。在第6阶段我们看到性能达到了饱和，我们在本文中也选择了6阶段方法在LSP和MPII数据集上进行试验，得到最好结果。

Figure 6: Comparisons on 3-stage architectures on the LSP dataset (PC): (a) Improvements over Pose Machine. (b) Comparisons between the different training methods. (c) Comparisons across each number of stages using joint training from scratch with intermediate supervision.

图6：我们的模型在LSP数据集(PC)上的性能比较：(a)相对于姿态机的改进；(b)不同训练方法见的比较；(c)不同阶段数的模型的比较，使用中间监督和各阶段联合训练。

### 4.2. Datasets and Quantitative Analysis 数据集和定量研究

In this section we present our numerical results in various standard benchmarks including the MPII, LSP, and FLIC datasets. To have normalized input samples of 368 × 368 for training, we first resize the images to roughly make the samples into the same scale, and then crop or pad the image according to the center positions and rough scale estimations provided in the datasets if available. In datasets such as LSP without these information, we estimate them according to joint positions or image sizes. For testing, we perform similar resizing and cropping (or padding), but estimate center position and scale only from image sizes when necessary. In addition, we merge the belief maps from different scales (perturbed around the given one) for final predictions, to handle the inaccuracy of the given scale estimation.

本节中，我们在几种标准基准测试中给出数值结果，包括MPII、LSP和FLIC数据集。为使输入样本统一大小为368×368进行训练，我们首先改变图像大小，大致将样本变成相同的尺度，然后根据数据集提供的中心位置和大致尺度估计（如果可用的话），剪切或补齐图像。对于测试，我们进行类似的改变图像大小和剪切（补齐），但是只在需要的时候从图像大小中估计中心位置和尺度。另外，我们从几个不同的尺度合并信念图（给定的尺度的前后）进行最后预测，这是为应付给定尺度估计的不准确性。

We define and implement our model using the Caffe [13] libraries for deep learning. We publicly release the source code and details on the architecture, learning parameters, design decisions and data augmentation to ensure full reproducibility. 

我们使用Caffe[13]定义并实现我们的模型。我们公开放出了我们的源代码和框架细节、学习参数、设计决策和数据扩充来确保完全的可重现性。

**MPII Human Pose Dataset**. We show in Figure 8 our results on the MPII Human Pose dataset [1] which consists more than 28000 training samples. We choose to randomly augment the data with rotation degrees in [−40◦, 40◦], scaling with factors in [0.7,1.3], and horizonal flipping. The evaluation is based on PCKh metric [1] where the error tolerance is normalized with respect to head size of the target. Because there often are multiple people in the proximity of the interested person (rough center position is given in the dataset), we made two sets of ideal belief maps for training: one includes all the peaks for every person appearing in the proximity of the primary subject and the second type where we only place peaks for the primary subject. We supply the first set of belief maps to the loss layers in the first stage as the initial stage only relies on local image evidence to make predictions. We supply the second type of belief maps to the loss layers of all subsequent stages. We also find that supplying to all subsequent stages an additional heat-map with a Gaussian peak indicating center of the primary subject is beneficial.

**MPII人体姿态数据集**。我们在图8中给出在MPII人体姿态数据集[1]上的结果，数据集包含超过28000幅训练样本。我们选择随机扩充数据，旋转角度范围[-40◦, 40◦]，尺度变换因子[0.7,1.3]，以及水平翻转。评估标准是基于PCKh度量[1]，其中误差容忍度根据目标头部大小归一化。因为在感兴趣的人（数据集给出了大致的图像中央位置）周围经常有多个人，我们制作了两种理想的信念图集来进行训练：一种包括主要目标周围的每个人的所有峰值，第二种是我们只包括主要目标的峰值。我们将第一种信念图用于第一阶段的损失层，因为初始阶段只依赖于图像局部证据来进行预测。我们将第二种信念图用于所有接下来阶段的损失层。我们还发现，给随后的所有阶段提供一个额外的热力图，用高斯峰值表明主要目标的中心位置，是有益处的。

Figure 8: Quantitative results on the MPII dataset using the PCKh metric. We achieve state of the art performance and outperform significantly on difficult parts such as the ankle.

Our total PCKh-0.5 score achieves state of the art at 87.95% (88.52% when adding LSP training data), which is 6.11% higher than the closest competitor, and it is noteworthy that on the ankle (the most challenging part), our PCKh-0.5 score is 78.28% (79.41% when adding LSP training data), which is 10.76% higher than the closest competitor. This result shows the capability of our model to capture long distance context given ankles are the farthest parts from head and other more recognizable parts. Figure 11 shows our accuracy is also consistently significantly higher than other methods across various view angles defined in [1], especially in those challenging non-frontal views. In summary, our method improves the accuracy in all parts, over all precisions, across all view angles, and is the first one achieving such high accuracy without any pre-training from other data, or post-inference parsing with hand-design priors or initialization of such a structured prediction task as in [28, 39]. Our methods also does not need another module dedicated to location refinement as in [38] to achieve great high-precision accuracy with a stride-8 network.

我们的PCKh-0.5总分取得最好成绩87.95%（如果加入LSP训练数据，是88.52%），比第二名高出6.11%，值得注意的是在脚踝部位（最有挑战性的部位），我们的PCKh-0.5分数是78.28%（如果加入LSP训练数据是79.41%），比第二名高出10.76%。这种结果表明了我们模型在捕捉长距离上下文的能力，因为脚踝是距离头部（和其他更容易识别的部位）最远的部位。图11给出我们的准确率也是明显一致的超出其他方法，包括定义在[1]种的各种不同视角，尤其是那些很有挑战性的非正面视角。总结一下，我们的方法在所有部位都改进了准确率，包括所有精度、所有视角，也是第一个在没有从其他数据预训练、没有用手工设计的先验知识进行后推理解析、没有像[28,39]种对这样一个结构化预测任务进行初始化的情况下，得到这么高的准确率的。我们的方法也不需要另一个模块进行位置优化[38]来得到高精度的高准确度。

**Leeds Sports Pose (LSP) Dataset**. We evaluate our method on the Extended Leeds Sports Dataset [15] that consists of 11000 images for training and 1000 images for testing. We trained on person-centric (PC) annotations and evaluate our method using the Percentage Correct Keypoints (PCK) metric [44]. Using the same augmentation scheme as for the MPI dataset, our model again achieves state of the art at 84.32% (90.5% when adding MPII training data). Note that adding MPII data here significantly boosts our performance, due to its labeling quality being much better than LSP. Because of the noisy label in the LSP dataset, Pishchulin et al. [28] reproduced the dataset with original high resolution images and better labeling quality.

**Leeds运动姿态数据集**。我们在扩展LSP数据集[15]上评估了我们的方法，这个数据集包括11000幅训练图像和1000幅测试图像。我们在人物中心(PC)标注上进行训练，用PCK度量[44]来评估我们的方法。数据扩充方案与MPII数据集相同，我们的模型再一次得到了最好的成绩84.32%（如果加入MPII训练数据的话就是90.5%）。注意加入了MPII数据后明显提升了我们的性能，这是因为其标注质量明显比LSP要好。由于LSP数据集种的含噪标签，Pishchulin等[28]用原始高清图和更好的标记质量重制了这个数据集。

Figure 9: Quantitative results on the LSP dataset using the PCK metric. Our method again achieves state of the art performance and has a significant advantage on challenging parts.

**FLIC Dataset**. We evaluate our method on the FLIC Dataset [32] which consists of 3987 images for training and 1016 images for testing. We report accuracy as per the metric introduced in Sapp et al. [32] for the elbow and wrist joints in Figure 12. Again, we outperform all prior art at PCK@0.2 with 97.59% on elbows and 95.03% on wrists. In higher precision region our advantage is even more significant: 14.8 percentage points on wrists and 12.7 percentage points on elbows at PCK@0.05, and 8.9 percentage points on wrists and 9.3 percentage points on elbows at PCK@0.1.

**FLIC数据集**。我们在FLIC数据集上评估了我们的方法，包括3987幅训练图像和1016幅测试图像。我们在图12中按照Sapp等[32]提出的度量给出肘部和手腕关节的准确率。我们再一次超过了之前的工作，肘部PCK@0.2准确率为97.59%，手腕为95.0%。在高精度区域我们的优势更加明显：PCK@0.05时手腕14.8%，肘部12.7%，PCK@0.1时手腕8.9%，肘部9.3%。

Figure 10: Qualitative results of our method on the MPII, LSP and FLIC datasets respectively. We see that the method is able to handle non-standard poses and resolve ambiguities between symmetric parts for a variety of different relative camera views.

Figure 11: Comparing PCKh-0.5 across various viewpoints in the MPII dataset. Our method is significantly better in all the viewpoints.

Figure 12: Quantitative results on the FLIC dataset for the elbow and wrist joints with a 4-stage CPM. We outperform all competing methods.

## 5. Discussion 讨论

Convolutional pose machines provide an end-to-end architecture for tackling structured prediction problems in computer vision without the need for graphical-model style inference. We showed that a sequential architecture composed of convolutional networks is capable of implicitly learning a spatial models for pose by communicating increasingly refined uncertainty-preserving beliefs between stages. Problems with spatial dependencies between variables arise in multiple domains of computer vision such as semantic image labeling, single image depth prediction and object detection and future work will involve extending our architecture to these problems. Our approach achieves state of the art accuracy on all primary benchmarks, however we do observe failure cases mainly when multiple people are in close proximity. Handling multiple people in a single end-to-end architecture is also a challenging problem and an interesting avenue for future work.

卷积姿态机为计算机视觉中的结构化预测问题给出了端到端的架构，而不需要图形模型风格的推理。我们给出了由卷积网络组成的序贯架构，能通过在不同阶段间不断优化保持不确定的信念值，隐式的学习姿态的空间模型。变量间空间依赖关系的问题在计算机视觉的多个领域中都存在，比如语义图像标记，单图像深度预测和目标检测，未来的工作会把我们的工作扩展到这些问题中。我们的方法在所有主要基准测试中取得了目前最好的准确率，但是我们确实观察到一些失败的案例，主要是在多人接近的情况下。在单个端到端框架中处理多人问题是一个很有挑战的问题，也是未来工作的一个方向。