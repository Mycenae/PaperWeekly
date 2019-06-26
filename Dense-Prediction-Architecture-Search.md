# Searching for Efficient Multi-Scale Architectures for Dense Image Prediction 搜索密集图像预测的高效多尺度架构

Liang-Chieh Chen et al. Google Inc.

## Abstract 摘要

The design of neural network architectures is an important component for achieving state-of-the-art performance with machine learning systems across a broad array of tasks. Much work has endeavored to design and build architectures automatically through clever construction of a search space paired with simple learning algorithms. Recent progress has demonstrated that such meta-learning methods may exceed scalable human-invented architectures on image classification tasks. An open question is the degree to which such methods may generalize to new domains. In this work we explore the construction of meta-learning techniques for dense image prediction focused on the tasks of scene parsing, person-part segmentation, and semantic image segmentation. Constructing viable search spaces in this domain is challenging because of the multi-scale representation of visual information and the necessity to operate on high resolution imagery. Based on a survey of techniques in dense image prediction, we construct a recursive search space and demonstrate that even with efficient random search, we can identify architectures that outperform human-invented architectures and achieve state-of-the-art performance on three dense prediction tasks including 82.7% on Cityscapes (street scene parsing), 71.3% on PASCAL-Person-Part (person-part segmentation), and 87.9% on PASCAL VOC 2012 (semantic image segmentation). Additionally, the resulting architecture is more computationally efficient, requiring half the parameters and half the computational cost as previous state of the art systems.

为取得最佳性能，神经网络架构的设计，在各种机器学习系统任务中都是最重要的要素之一。设计一个搜索空间，并配以简单的学习算法，自动设计并构建架构，这方面有了很多努力。最新的进展证明了，这种元学习方法在图像分类任务中可以超过人类发明的可扩展架构。这种方法可以泛化到怎样的新领域，还是一个开放的问题。本文中，我们研究了对密集图像预测任务构建元学习技术，聚焦在场景解析、人体部位分割和语义分割任务上。在这个领域中构造可行的搜索空间是很有挑战性的，因为需要视觉信息的多尺度表示，和需要在高分辨率图像上进行运算。基于密集图像预测的技术回顾，我们构建了一个递归搜索空间，证明了即使使用有效的随机搜索，我们也可以得到超过人类设计性能的架构，在三个密集预测任务中得到目前最佳的表现，包括在Cityscapes上的82.7%（街道场景解析），在PASCAL-Person-Part上的71.3%（人体部位分割），在PASCAL VOC 2012上的87.9%（语义分割）。另外，得到的架构计算量也很少，与之前最好的系统相比，只需要一般的参数量和计算量。

## 1 Introduction 引言

The resurgence of neural networks in machine learning has shifted the emphasis for building state-of-the-art systems in such tasks as image recognition [44, 84, 83, 34], speech recognition [36, 8], and machine translation [88, 82] towards the design of neural network architectures. Recent work has demonstrated successes in automatically designing network architectures, largely focused on single-label image classification tasks [100, 101, 52] (but see [100, 65] for language tasks). Importantly, in just the last year such meta-learning techniques have identified architectures that exceed the performance of human-invented architectures for large-scale image classification problems [101, 52, 68].

神经网络在机器学习中的复兴，将构建最好系统的重点变成了设计神经网络架构，其应用包括图像识别、语音识别和机器翻译。最近很多工作成功的进行了网络结构设计的自动化，主要是在单标签图像分类任务中（也有语言任务的例子）。重要的是，就是在去年，这种元学习技术发现架构的性能，在大规模图像分类任务中，超过了人类发明架构的性能。

Image classification has provided a great starting point because much research effort has identified successful network motifs and operators that may be employed to construct search spaces for architectures [52, 68, 101]. Additionally, image classification is inherently multi-resolution whereby fully convolutional architectures [77, 58] may be trained on low resolution images (with minimal computational demand) and be transferred to high resolution images [101].

图像分类是一个重要的起始点，因为很多研究工作在此成功发现了网络基元和算子，可用于构建架构搜索空间。另外，图像分类本质上就是多分辨率的，可以在低分辨率上训练全卷积架构（使用最小的计算代价），然后迁移到高分辨率图像中去。

Although these results suggest opportunity, the real promise depends on the degree to which meta-learning may extend into domains beyond image classification. In particular, in the image domain, many important tasks such as semantic image segmentation [58, 11, 97], object detection [71, 21], and instance segmentation [20, 33, 9] rely on high resolution image inputs and multi-scale image representations. Naïvely porting ideas from image classification would not suffice because (1) the space of network motifs and operators differ notably from systems that perform classification and (2) architecture search must inherently operate on high resolution imagery. This final point makes previous approaches computationally intractable where transfer learning from low to high image resolutions was critical [101].

虽然这些结果意味着机会，但真正的希望要看元学习可以拓展到图像分类之外的领域的程度。特别是，在图像领域，很多重要任务，如语义分割，目标检测，和实例分割，都需要高分辨率图像输入和多尺度图像表示。简单将图像分类的想法移植可能不够，因为：(1)网络基元和算子的空间与图像分类的系统非常不同；(2)架构搜索必须真正在高分辨率图像上进行。最后一点使之前的方法在计算量上很难办，因为之前在从低分辨率到高分辨率的迁移学习很关键。

In this work, we present the first effort towards applying meta-learning to dense image prediction (Fig. 1) – largely focused on the heavily-studied problem of scene labeling. Scene labeling refers to the problem of assigning semantic labels such as "person" or "bicycle" to every pixel in an image. State-of-the-art systems in scene labeling are elaborations of convolutional neural networks (CNNs) largely structured as an encoder-decoder in which various forms of pooling, spatial pyramid structures [97] and atrous convolutions [11] have been explored. The goal of these operations is to build a multi-scale representation of a high resolution image to densely predict pixel values (e.g., stuff label, object label, etc.). We leverage off this literature in order to construct a search space over network motifs for dense prediction. Additionally, we perform an array of experiments to demonstrate how to construct a computationally tractable and simple proxy task that may provide predictive information on multi-scale architectures for high resolution imagery.

本文中，我们第一次将元学习应用到密集图像预测中（图1），主要关注研究很多的场景标记。场景标记是指为图像中的每个像素指定一个语义标签的问题，如人体或自行车。目前最好的场景标记系统是基于CNNs的，主要采用的编码器-解码器结构，使用了各种形式的池化、空间金字塔结构和孔洞卷积。这些算子的目的是为了构建高分辨率图像的多尺度表示，以进行密集像素值预测（如，stuff标签，目标标签，等）。我们在网络基元上构建了搜索空间，以进行密集预测。另外，我们进行了很多试验，展示了怎样构建计算量可行的、简单的代理任务，在多尺度架构上为高分辨率图像给出可预测的信息。

Figure 1: Schematic diagram of architecture search for dense image prediction. Example tasks explored in this paper include scene parsing [18], semantic image segmentation [24] and person-part segmentation [16].

We find that an effective random search policy provides a strong baseline [5, 30] and identify several candidate network architectures for scene labeling. In experiments on the Cityscapes dataset [18], we find architectures that achieve 82.7% mIOU accuracy, exceeding the performance of human-invented architectures by 0.7% [6]. For reference, note that achieving gains on the Cityscapes dataset is challenging as the previous academic competition elicited gains of 0.8% in mIOU from [97] to [6] over more than one year. Additionally, this same network applied to other dense prediction tasks such as person-part segmentation [16] and semantic image segmentation [24] surpasses state-of-the-art results [25, 93] by 3.7% and 1.7% in absolute percentage, respectively (and comparable to concurrent works [14, 96, 48] on VOC 2012). This is the first time to our knowledge that a meta-learning algorithm has matched state-of-the-art performance using architecture search techniques on dense image prediction problems. Notably, the identified architecture operates with half the number of trainable parameters and roughly half the computational demand (in Multiply-Adds) as previous state-of-the-art systems [14], when employing the powerful Xception [17, 67, 14] as network backbone.

我们发现了一种有效的随机搜索策略，给出了很强的基准[5,30]，发现了几种场景标记的网络架构候选。在Cityscapes数据集上的试验中，我们发现的架构取得了82.7%的mIOU准确率，超过了人类发明的架构0.7%[6]。注意在Cityscapes数据集上得到改进是很有挑战性的，因为之前的学术比赛从[97]到[6]得到了0.8%的改进用了一年多，可以作为参考。另外，同样的网络用于其他密集预测任务，如人体部位分割和语义分割，分别超过了目前最好的结果3.7%和1.7%（在VOC 2012上与目前的工作结果类似）。据我们所知，这是第一次元学习算法在密集预测问题上使用架构搜索得到目前最好的性能。值得注意的是，在采用Xception作为网络骨干时，发现的架构与之前最好的系统[14]相比，只有大约一半的可训练参数和计算量。

## 2 Related Work 相关工作

### 2.1 Architecture search 架构搜索

Our work is motivated by the neural architecture search (NAS) method [100, 101], which trains a controller network to generate neural architectures. In particular, [101] transfers architectures learned on a proxy dataset [43] to more challenging datasets [73] and demonstrates superior performance over many human-invented architectures. Many parallel efforts have employed reinforcement learning [3, 99], evolutionary algorithms [81, 69, 59, 90, 53, 68] and sequential model-based optimization [61, 52] to learn network structures. Additionally, other works focus on successively increasing model size [7, 15], sharing model weights to accelerate model search [65], or a continuous relaxation of the architecture representation [54]. Note that our work is complimentary and may leverage all of these advances in search techniques to accelerate the search and decrease computational demand.

我们的工作受到神经架构搜索(NAS)方法启发，[100,101]训练了一个控制器网络来生成神经架构。特别是，[101]将在代理数据集[43]上学到的架构迁移到更有挑战性的数据集[73]上，得到了比很多人类设计的架构还要好的性能。很多同时的工作使用了强化学习、演化算法和基于序列模型的优化来学习网络结构。另外，其他工作聚焦在连续增加模型大小，共享模型权重以加速模型搜索，或架构表示的连续松弛。注意，我们的工作赞同这些工作，可能会利用所有这些搜索技术的进展，以加速搜索并降低计算需求。

Critically, all approaches are predicated on constructing powerful but tractable architecture search spaces. Indeed, [52, 101, 68] find that sophisticated learning algorithms achieve superior results; however, even random search may achieve strong results if the search space is not overly expansive. Motivated by this last point, we focus our efforts on developing a tractable and powerful search space for dense image prediction paired with efficient random search [5, 30].

关键的是，所有方法都在努力构建强力但容易处理的架构搜索空间。确实，[52,101,68]发现复杂的学习算法可以得到更好的结果；但是，如果搜索空间不是太大，即使随机搜索也可以得到很好的结果。受最后一点启发，我们将努力为密集图像预测构建一个容易处理而且强力的搜索空间，辅以高效的随机搜索。

Recently, [75, 27] proposed methods for embedding an exponentially large number of architectures in a grid arrangement for semantic segmentation tasks. In this work, we instead propose a novel recursive search space and simple yet predictive proxy tasks aimed at finding effective architectures for dense image prediction.

最近，[75,27]提出将数量为指数级大的架构嵌入到一个网格排列中，以进行语义分割任务。本文中，我们提出了一种新的递归搜索空间，和一种简单却可预测的代理任务，目标是发现密集预测的有效架构。

### 2.2 Multi-scale representation for dense image prediction 密集图像预测中的多尺度表示

State-of-the-art solutions for dense image predictions derive largely from convolutional neural networks [46]. A critical element of building such systems is supplying global features and context information to perform pixel-level classification [35, 78, 41, 45, 31, 92, 60, 19, 63]. Several approaches exist for how to efficiently encode the multi-scale context information in a network architecture: (1) designing models that take as input an image pyramid so that large scale objects are captured by the downsampled image [26, 66, 23, 50, 13, 11], (2) designing models that contain encoder-decoder structures [2, 72, 49, 28, 64, 93, 96], or (3) designing models that employ a multi-scale context module, e.g., DenseCRF module [42, 4, 10, 98, 50, 76], global context [56, 95], or atrous convolutions deployed in cascade [57, 94, 12] or in parallel [11, 12]. In particular, PSPNet [97] and DeepLab [12, 14] perform spatial pyramid pooling at several hand-designed grid scales.

目前最好的密集预测方法主要是从卷积神经网络中推导而来。构建这种系统的一个关键元素是，提供全局特征和上下文信息，以进行像素级的分类。有几种方法可以有效的在一个网络架构中编码多尺度上下文信息：(1)设计的模型以图像金字塔为输入，大尺度的目标由下采样的图像捕获；(2)设计的模型包含编码器-解码器结构；(3)设计的模型包含一个多尺度上下文模块，如DenseCRF模块，全局上下文，或级联/并行形式部署的孔洞卷积。特别是，PSPNet和DeepLab在几种手工设计的网格尺度上进行空间金字塔池化。

A common theme in the dense prediction literature is how to best tune an architecture to extract context information. Several works have focused on sampling rates in atrous convolution to encode multi-scale context [37, 29, 77, 62, 10, 94, 11]. DeepLab-v1 [10] is the first model that enlarges the sampling rate to capture long range information for segmentation. The authors of [94] build a context module by gradually increasing the rate on top of belief maps, the final CNN feature maps that contain output channels equal to the number of predicted classes. The work in [87] employs a hybrid of rates within the last two blocks of ResNet [34], while Deformable ConvNets [22] proposes the deformable convolution which generalizes atrous convolution by learning the rates. DeepLab-v2 [11] and DeepLab-v3 [12] employ a module, called ASPP (atrous spatial pyramid pooling module), which consists of several parallel atrous convolutions with different rates, aiming to capture different scale information. Dense-ASPP [91] proposes to build the ASPP module in a densely connected manner. We discuss below how to construct a search space that captures all of these features.

密集预测文献中的一个常见主题是，怎样调节一个架构以提取上下文信息。一些工作聚焦在孔洞卷积中的采样率，以编码多尺度上下文。DeepLab-v1是第一种增大采样率以捕获长程信息进行分割的模型。[94]通过逐渐增加belief maps的比率以构建一个上下文模块，最后的CNN特征图包含的输出通道数量与预测类别的数量相等。[87]在ResNet的最后两个模块中采用了复合比率，而Deformable ConvNets [22] 提出了deformable卷积，通过学习这些比率将孔洞卷积进行了推广。DeepLab-v2[11]和DeepLab-v3[12]采用了ASPP模块，包含了几个不同比率的并行孔洞卷积，目标是捕获不同尺度的信息。Dense-ASPP[91]提出以密集连接的方式构建ASPP模块。我们下面讨论如何构建一个搜索空间，可以捕获所有这些特征。

## 3 Methods 方法

Two key components for building a successful architecture search method are the design of the search space and the design of the proxy task [100, 101]. Most of the human expertise shifts from architecture design to the construction of a search space that is both expressive and tractable. Likewise, identifying a proxy task that is both predictive of the large-scale task but is extremely quick to run is critical for searching this space efficiently.

构建成功的架构搜索方法，两个关键的组件是，设计搜索空间，和设计代理任务。人类专业技能从架构设计，转移到构建搜索空间，这个空间既要有表达力，还要容易处理。类似的，确定一种可以预测大规模任务的代理任务，同时运行起来非常快速，对于高效的搜索此空间，也非常关键。

### 3.1 Architecture search space 架构搜索空间

The goal of architecture search space is to design a space that may express a wide range of architectures, but also be tractable enough for identifying good models. We start with the premise of building a search space that may express all of the state-of-the-art dense prediction and segmentation models previously discussed (e.g. [12, 97] and see Sec. 2 for more details).

架构搜索空间的目标是设计一种空间，可以表示大量架构，但又容易处理，以确定好的模型。我们从一个前提开始，即构建的搜索空间可以表示所有目前最好的之前讨论过的密集预测和分割模型（如[12,97]，详见第2部分）。

We build a recursive search space to encode multi-scale context information for dense prediction tasks that we term a Dense Prediction Cell (DPC). The cell is represented by a directed acyclic graph (DAG) which consists of B branches and each branch maps one input tensor to another output tensor. In preliminary experiments we found that B = 5 provides a good trade-off between flexibility and computational tractability (see Sec. 5 for more discussion).

我们构建了一个递归的搜索空间，以编码多尺度上下文信息，进行密集预测，我们称之为密集预测单元(Dense Prediction Cell, DPC)。这个单元由一个有向无环图(Directed Acyclic Graph, DAG)表示，包含B个分支，每个分支将一个输入张量映射到另一个输出张量。通过初步的试验，我们发现B=5在灵活性和计算可行性上得到很好的平衡（详见第5部分）。

We specify a branch $b_i$ in a DPC as a 3-tuple, ($X_i, OP_i, Y_i$), where $X_i ∈ \bold X_i$ specifies the input tensor, $OP_i ∈ OP$ specifies the operation to apply to input $X_i$, and $Y_i$ denotes the output tensor. The final output, Y, of the DPC is the concatenation of all branch outputs, i.e., Y = concat($Y_1, Y_2, . . ., Y_B$), allowing us to exploit all the learned information from each branch. For branch $b_i$, the set of possible inputs, $\bold X_i$, is equal to the last network backbone feature maps, F, plus all outputs obtained by previous branches, $Y_1, . . ., Y_{i−1}$, i.e., $\bold X_i = \{F, Y_1, . . ., Y_{i−1}\}$. Note that $\bold X_1$ = {F}, i.e., the first branch can only take F as input.

我们将DPC的一个分支$b_i$指定为一个三元组，($X_i, OP_i, Y_i$)，其中$X_i ∈ \bold X_i$指定了输入张量，$OP_i ∈ OP$指定了对输入$X_i$运算的算子，$Y_i$表示输出张量。DPC最终的输出Y，是所有分支输出的拼接，即Y = concat($Y_1, Y_2, . . ., Y_B$)，这样可以利用所有分支学习到的信息。对于分支$b_i$，可能输入的集合$\bold X_i$，就是骨干网络最后的特征图F，加上之前分支的所有输出$Y_1, . . ., Y_{i−1}$，即，$\bold X_i = \{F, Y_1, . . ., Y_{i−1}\}$。注意，$\bold X_1$ = {F}，即，第一个分支只能以F作为输入。

The operator space, OP, is defined as the following set of functions: 算子空间OP，定义为下列函数的集合：

- Convolution with a 1 × 1 kernel.
- 3×3 atrous separable convolution with rate $r_h × r_w$, where $r_h$ and $r_w$ ∈ {1, 3, 6, 9, . . . , 21}.
- Average spatial pyramid pooling with grid size $g_h × g_w$, where $g_h$ and $g_w$ ∈ {1, 2, 4, 8}.

For the spatial pyramid pooling operation, we perform average pooling in each grid. After the average pooling, we apply another 1 × 1 convolution followed by bilinear upsampling to resize back to the same spatial resolution as input tensor. For example, when the pooling grid size $g_h × g_w$ is equal to 1 × 1, we perform image-level average pooling followed by another 1 × 1 convolution, and then resize back (i.e., tile) the features to have the same spatial resolution as input tensor.

对于空间金字塔池化运算，我们在每个网格中进行平均池化。在平均池化后，我们使用1×1卷积，随后是双线性插值上采样，以将分辨率变换到输入张量的空间分辨率。比如，当池化网格大小$g_h × g_w$等于1×1时，我们进行图像级的平均池化，然后是另一个1×1卷积，然后将特征分辨率变换为输入张量的相同空间分辨率。

We employ separable convolution [79, 85, 86, 17, 38] with 256 filters for all the convolutions, and decouple sampling rates in the 3 × 3 atrous separable convolution to be $r_h × r_w$ which allows us to capture object scales with different aspect ratios. See Fig. 2 for an example.

我们对所有卷积都采用256滤波器的可分离卷积，并将3×3孔洞可分离卷积的采样率解耦合为$r_h × r_w$，这使我们可以捕获不同纵横比的目标。图2是一个例子。

Figure 2: Diagram of the search space for atrous convolutions. 3 × 3 atrous convolutions with sampling rates $r_h × r_w$ to capture contexts with different aspect ratios. From left to right: standard convolution (1 × 1), equal expansion (6 × 6), short and fat (6 × 24) and tall and skinny (24 × 6).

The resulting search space may encode all leading architectures but is more diverse as each branch of the cell may build contextual information through parallel or cascaded representations. The potential diversity of the search space may be expressed in terms of the total number of potential architectures. For i-th branch, there are i possible inputs, including the last feature maps produced by the network backbone (i.e., F) as well as all the previous branch outputs (i.e., $Y_1, . . ., Y_{i−1}$), and 1 + 8 × 8 + 4 × 4 = 81 functions in the operator space, resulting in i × 81 possible options. Therefore, for B = 5 branches, the search space contains $B! × 81^B ≈ 4.2 × 10^{11}$ configurations.

得到的搜索空间可能包括了所有领先的架构，但会更多样化，因为单元的每个分支可能会通过并行或级联表示构建上下文表示。搜索空间的多样化潜力，可以用可能的架构总数量表示。对于第i个分支，可能有i个输入，包括骨干网络生成的最后的特征图即F，以及之前的分支的输出（即，$Y_1, . . ., Y_{i−1}$），算子空间有1 + 8 × 8 + 4 × 4 = 81个函数，最后得到i×81个可能的选项。所以，对于B=5的分支，搜索空间包含$B! × 81^B ≈ 4.2 × 10^{11}$种配置。

### 3.2 Architecture search 架构搜索

The model search framework builds on top of an efficient optimization service [30]. It may be thought of as a black-box optimization tool whose task is to optimize an objective function f : b → R with a limited evaluation budget, where in our case b = {$b_1, b_2, . . . , b_B$} is the architecture of DPC and f(b) is the pixel-wise mean intersection-over-union (mIOU) [24] evaluated on the dense prediction dataset. The black-box optimization refers to the process of generating a sequence of b that approaches the global optimum (if any) as fast as possible. Our search space size is on the order of $10^{11}$ and we adopt the random search algorithm implemented by Vizier [30], which basically employs the strategy of sampling points b uniformly at random as well as sampling some points b near the currently best observed architectures. We refer the interested readers to [30] for more details. Note that the random search algorithm is a simple yet powerful method. As highlighted in [101], random search is competitive with reinforcement learning and other learning techniques [52].

模型搜索框架是在一个高效的优化服务[30]上构建起来的。这可以认为是一种黑盒优化工具，其任务是在有限的评估预算中优化一个目标函数f:b→R，在我们这种情况下，b = {$b_1, b_2, . . . , b_B$}，是DPC的架构，f(b)是在密集预测数据集上计算的像素mIOU[24]。黑盒优化是指，生成b序列的过程，尽快达到全局最优点。我们的搜索空间大小是$10^{11}$数量级的，我们采用的是Vizier[30]实现的随机搜索算法，算法使用的是基本策略，即对b进行均匀随机采样，也对目前最好的架构附近进行一些采样。有兴趣的读者请参考[30]。注意随机搜索算法是一种简单但强力的方法。[101]中强调了，随机搜索与强化学习以及其他学习技术一起非常有竞争力[52]。

### 3.3 Design of the proxy task

Naïvely applying architecture search to a dense prediction task requires an inordinate amount of computation and time, as the search space is large and training a candidate architecture is time-consuming. For example, if one fine-tunes the entire model with a single dense prediction cell (DPC) on the Cityscapes dataset, then training a candidate architecture with 90K iterations requires 1+ week with a single P100 GPU. Therefore, we focus on designing a proxy task that is (1) fast to compute and (2) may predict the performance in a large-scale training setting.

直接对密集预测任务使用架构搜索，需要极大量的计算和时间，因为搜索空间很大，训练一个候选架构也非常耗时。比如，如果用单个密集预测单元(DPC)在Cityscapes数据集上精调整个模型，然后训练一个候选架构90K次迭代，在单个P100 GPU上，需要1个星期多的时间。所以，我们设计的代理任务需要：(1)计算快速；(2)可以预测在大规模训练设置下的性能。

Image classification employs low resolution images [43] as a fast proxy task for high-resolution [73]. This proxy task does not work for dense image prediction where high resolution imagery is critical for conveying multi-scale context information. Therefore, we propose to design the proxy dataset by (1) employing a smaller network backbone and (2) caching the feature maps produced by the network backbone on the training set and directly building a single DPC on top of it. Note that the latter point is equivalent to not back-propagating gradients to the network backbone in the real setting. In addition, we elect for early stopping by not training candidate architectures to convergence. In our experiments, we only train each candidate architecture with 30K iterations. In summary, these two design choices result in a proxy task that runs in 90 minutes on a GPU cutting down the computation time by 100+-fold but is predictive of larger tasks (ρ ≥ 0.4).

图像分类采用低分辨率图像[43]作为高分辨率图像[73]的快速代理任务。这种代理任务对于密集图像预测不适用，因为高分辨率对传递多尺度上下文信息是非常关键的。所以，我们提出通过以下方法设计代理数据集，(1)采用更小的骨干网络；(2)将骨干网络在训练集上计算的特征图进行缓存，直接在其上构建一个DPC。注意，后面一点等价于，在真实设置中，梯度不会反向传播到骨干网络。另外，我们不会将候选架构训练到收敛，选择早停。在我们的试验中，我们只训练每个候选架构30K次迭代。总结一下，这两种设计选择得到的代理任务，在一个GPU上运行时间为90分钟，将计算时间缩减了100+倍，但还可以预测更大的任务(ρ ≥ 0.4)。

After performing architecture search, we run a reranking experiment to more precisely measure the efficacy of each architecture in the large-scale setting [100, 101, 68]. In the reranking experiments, the network backbone is fine-tuned and trained to full convergence. The new top architectures returned by this experiment are presented in this work as the best DPC architectures.

在进行架构搜索后，我们进行一个重新排序的试验，以更精确的衡量每种架构在大规模设置下的效率。在重新排序的试验里，骨干网络得到精调，也训练到完全收敛。试验得到的新的最高排名的架构，作为最佳DPC架构。

## 4 Results 结果

We demonstrate the effectiveness of our proposed method on three dense prediction tasks that are well studied in the literature: scene parsing (Cityscapes [18]), person part segmentation (PASCAL-Person-Part [16]), and semantic image segmentation (PASCAL VOC 2012 [24]). Training and evaluation protocols follow [12, 14]. In brief, the network backbone is pre-trained on the COCO dataset [51]. The training protocol employs a polynomial learning rate [56] with an initial learning rate of 0.01, large crop sizes (e.g., 769 × 769 on Cityscapes and 513 × 513 on PASCAL images), fine-tuned batch normalization parameters [40] and small batch training (batch size = 8, 16 for proxy and real tasks, respectively). For evaluation and architecture search, we employ a single image scale. For the final results in which we compare against other state-of-the-art systems (Tab. 2, Tab. 3 and Tab. 4), we perform evaluation by averaging over multiple scalings of a given image.

我们在三个密集预测任务中证明了提出方法的有效性：场景解析(Cityscapes[18])，人体部位分割(PASCAL-Person_Part[16])，和语义图像分割(PASCAL VOC 2012[24])。训练和评估方法与[12,14]中类似。简要来说，骨干网络在COCO数据集上进行预训练。训练方法采用多项式学习速率[56]，初始学习速率为0.01，大剪切块大小（如，在Cityscapes上769×769，在PASCAL图像上为513×513），精调BN参数[40]，小批次训练（对于代理和真实任务，分别为batch size = 8, 16）。在评估和架构搜索时，我们采用单图像尺度。与目前最好的系统相比的最后结果（表2,3,4），我们对给定的图像在多尺度上进行预测，然后进行平均。

### 4.1 Designing a proxy task for dense prediction 为密集预测设计一个代理任务

The goal of a proxy task is to identify a problem that is quick to evaluate but provides a predictive signal about the large-scale task. In the image classification work, the proxy task was classification on low resolution (e.g. 32 × 32) images [100, 101]. Dense prediction tasks innately require high resolution images as training data. Because the computational demand of convolutional operations scale as the number of pixels, another meaningful proxy task must be identified.

代理任务的目标是确定一个可以快速评估的问题，但可以给出大规模任务的预测信息。在图像分类任务中，代理任务是在低分辨率图像上进行分类（如32×32）[100,101]。密集预测任务天生需要高分辨率图像作为训练数据。由于卷积运算的计算量随像素数量变化而变化，所以必须找到另一个有意义的代理任务。

We approach the problem of proxy task design by focusing on speed and predictive ability. As discussed in Sec. 3, we employ several strategies for devising a fast and predictive proxy task to speed up the evaluation of a model from 1+ week to 90 minutes on a single GPU. In these preliminary experiments, we demonstrate that these strategies provide an instructive signal for predicting the efficacy of a given architecture.

我们聚焦代理任务设计的速度和预测能力上。如第3部分所述，我们采用几种策略设计一个快速、预测能力强的代理任务，以加速模型评估，在单个GPU上从1周多到90分钟。在这些初步试验中，我们证明了这些策略在预测给定框架的有效性时，可以给出有意义的信号。

To minimize stochastic variation due to sampling architectures, we first construct an extremely small search space containing only 31 architectures in which we may exhaustively explore performance. We perform the experiments and subsequent architecture search on Cityscapes [18], which features large variations in object scale across 19 semantic labels.

为最小化架构采样带来的随机变化，我们首先构建了一个极小的搜索空间，只包含31个架构，我们会在其中穷尽探索性能。我们在Cityscapes[18]上进行试验和后续的架构搜索，在这个数据集上有19个语义标签，目标尺度变化很大。

Following previous state-of-the-art segmentation models, we employ the Xception architecture [17, 67, 14] for the large-scale setting. We first asked whether a smaller network backbone, MobileNet-v2 [74] provides a strong signal of the performance of the large network backbone (Fig. 3a). MobileNet-v2 consists of roughly 1/20 the computational cost and cuts down the backbone feature channels from 2048 to 320 dimensions. We indeed find a rank correlation (ρ = 0.36) comparable to learned predictors [52], suggesting that this may provide a reasonable substitute for the proxy task. We next asked whether employing a fixed and cached set of activations correlates well with training end-to-end. Fig. 3b shows that a higher rank correlation between cached activations and training end-to-end for COCO pretrained MobileNet-v2 backbone (ρ = 0.47). The fact that these rank correlations are significantly above chance rate (ρ = 0) indicates that these design choices provide a useful signal for large-scale experiments (i.e., more expensive network backbone) comparable to learned predictors [52, 101] (for reference, ρ ∈ [0.41, 0.47] in the last stage of [52]) as well as a fast proxy task.

借鉴之前最好的分割模型，我们在大规模设置下采用Xception架构。我们首先尝试一个更小的骨干网络，MobileNet-v2，是否可以作为大型骨干网络的替代（图3a）。MobileNet-v2的计算量大约是Xception的1/20，将骨干网络通道数从2018降到320。我们的确找到了排名相关性(ρ = 0.36)，说明这可能是代理任务的一个合理替代。下一步，我们试验，采用固定的、缓存的激活集与进行端到端训练是否相关性很大。图3b表明，在缓存激活和端到端训练（COCO预训练的MobileNet-v2骨干）间存在更强的关联(ρ = 0.47)。这些排名相关性明显高于chance rate(ρ = 0)，这说明这些设计选择可以为大规模试验（即，更大的骨干网络）给出一个有用的指示信号（可供参考的是，[52]的最后阶段ρ ∈ [0.41, 0.47]），以及快速代理任务。

Figure 3: Measuring the fidelity of proxy tasks for a dense prediction cell (DPC) in a reduced search space. In preliminary search spaces, a comparison of (a) small to large network backbones, and (b) proxy versus large-scale training with MobileNet-v2 backbone. ρ is Spearman’s rank correlation coefficient.

### 4.2 Architecture search for dense prediction cells 密集预测单元的架构搜索

We deploy the resulting proxy task, with our proposed architecture search space, on Cityscapes to explore 28K DPC architectures across 370 GPUs over one week. We employ a simple and efficient random search [5, 30] and select the top 50 architectures (w.r.t. validation set performance) for re-ranking based on fine-tuning the entire model using MobileNet-v2 network backbone. Fig. 4a highlights the distribution of performance scores on the proxy dataset, showing that the architecture search algorithm is able to explore a diversity of architectures. Fig. 4b demonstrates the correlation of the found top-50 DPCs between the original proxy task and the re-ranked scores. Notably, the top model identified with re-ranking was the 12-th best model as measured by the proxy score.

我们用我们提出的架构搜索空间，在Cityscapes上部署得到的代理任务，在370个GPUs上用超过一个星期探索了28K个DPC架构。我们采用了一种简单但有效的随机搜索方法[5,30]，选择了最好的50个架构（在验证集上的表现）进行重新排序，排序时使用MobileNet-v2作为骨干网络精调了整个模型。图4a强调了在代理数据集上的性能分数分布，表明架构搜索算法可以探索很多架构。图4b证明了找到的top-50 DPCs在原始的代理任务和重新排序的分数中的相关性。尤其是，重新排序后的最高表现模型是在代理分数排名12的模型。

Figure 4: Measuring the fidelity of the proxy tasks for a dense prediction cell (DPC) in the full search space. (a) Score distribution on the proxy task. The search algorithm is able to explore a diversity of architectures. (b) Correlation of the found top-50 architectures between the proxy dataset and large-scale training with MobileNet-v2 backbone. ρ is Spearman’s rank correlation coefficient.

Fig. 5a provides a schematic diagram of the top DPC architecture identified (see Fig. 6 for the next best performing ones). Following [39] we examine the L1 norm of the weights connecting each branch (via a 1 × 1 convolution) to the output of the top performing DPC in Fig. 5b. We observe that the branch with the 3×3 convolution (rate = 1×6) contributes most, whereas the branches with large rates (i.e., longer context) contribute less. In other words, information from image features in closer proximity (i.e. final spatial scale) contribute more to the final outputs of the network. In contrast, the worst-performing DPC (Fig. 6c) does not preserve fine spatial information as it cascades four branches after the global image pooling operation.

图5a给出了发现的最好DPC架构的示意图（图6是下几个表现最好的模型）。与[39]中一样，我们检查连接每个分支（通过1×1卷积）到输出的权重的L1范数，图5b中给出表现最好的DPC的结果。我们观察到，采用3×3卷积(rate=1×6)的分支贡献最大，而比率很大的分支（即更长的上下文）贡献反而更小。换句话说，图像特征更紧密的信息（即，最后的空间尺度）对网络最后输出贡献更多。比较起来，表现最差的DPC（图6c）没有保存精细的空间信息，因为在全局图像池化运算后级联了四个分支。

Figure 5: Schematic diagram of top ranked DPC (left) and average absolute filter weights (L1 norm) for each operation (right).

Figure 6: Diversity of DPCs explored in architecture search. (b-d) Top-2, Top-3 and worst DPCs.

### 4.3 Performance on scene parsing 在场景解析上的表现

We train the best learned DPC with MobileNet-v2 [74] and modified Xception [17, 67, 14] as network backbones on Cityscapes training set [18] and evaluate on the validation set. The network backbone is pretrained on the COCO dataset [51] for this and all subsequent experiments. Fig. 7 in the appendix shows qualitative results of the predictions from the resulting architecture. Quantitative results in Tab. 1 highlight that the learned DPC provides 1.4% improvement on the validation set when using MobileNet-v2 network backbone and a 0.6% improvement when using the larger modified Xception network backbone. Furthermore, the best DPC only requires half of the parameters and 38% of the FLOPS of the previous state-of-the-art dense prediction network [14] when using Xception as network backbone. We note the computation saving results from the cascaded structure in our top-1 DPC, since the feature channels of Xception backbone is 2048 and thus it is expensive to directly build parallel operations on top of it (like ASPP).

我们使用MobileNet-v2和调整后的Xception作为网络骨干，在Cityscapes训练集上训练学习到最好的DPC，并在验证集上评估。网络骨干在COCO数据集上预训练，在本试验和后续的试验中都是这样。图7所示的是得到的架构的一些定性预测结果。表1给出了量化的结果，强调了学习到的DPC在使用MobileNet-v2骨干网络时，有1.4%的性能改进；在使用更大的调整Xception作为骨干网络时，有0.6%的性能改进。而且，与之前最好的密集预测网络[14]相比，在使用Xception作为网络骨干时，搜索得到的最好的DPC只需要大约一半的参数量，和38%的FLOPS。我们注意到，节省的计算量是我们搜索到的top-1 DPC级联起来的结果，因为Xception骨干的特征通道数为2048，所以在其上直接构建并行的运算（像ASPP）非常耗时。

Table 1: Cityscapes validation set performance (labeling IOU) across different network backbones (output stride = 16). ASPP is the previous state-of-the-art system [12] and DPC indicates this work. Params and MAdds indicate the number of parameters and number of multiply-add operations in each multi-scale context module.

Network Backbone | Module | Params | MAdds | mIOU (%)
--- | --- | --- | --- | ---
MobileNet-v2 | ASPP[12] | 0.25M | 2.82B | 73.97
MobileNet-v2 | DPC | 0.36M | 3.00B | 75.98
Modified Xception | ASPP[12] | 1.59M | 18.12B | 80.25
Modified Xception | DPC | 0.81M | 6.84B | 80.85

Figure 7: Visualization of predictions on the Cityscapes validation set.

We next evaluate the performance on the test set (Tab. 2). DPC sets a new state-of-the-art performance of 82.7% mIOU – an 0.7% improvement over the state-of-the-art model [6]. This model outperforms other state-of-the-art models across 11 of the 19 categories. We emphasize that achieving gains on Cityscapes dataset is challenging because this is a heavily researched benchmark. The previous academic competition elicited gains of 0.8% in mIOU from [97] to [6] over the span of one year.

我们然后在测试集上评估算法性能（表2）。DPC竖立了新的最好成绩，82.7% mIOU，比之前最好的模型[6]改进了0.7%。这个模型在19类中的11类中都超过了其他最好的模型。我们强调在Cityscapes上取得改进是很有挑战的，因为这是一个研究很多的基准测试。之前的学术比赛，从[97]到[6]取得了0.8%的mIOU改进，花了一年多的时间。

Table 2: Cityscapes test set performance across leading competitive models.

Method | mIOU
--- | ---
PSPNet [97] | 81.2
Mapillary Research [6] | 82.0
DeepLabv3+ [14] | 82.1
DPC | 82.7

### 4.4 Performance on person part segmentation

PASCAL-Person-Part dataset [16] contains large variation in object scale and human pose annotating six person part classes as well as the background class. We train a model on this dataset employing the same DPC identified during architecture search using the modified Xception network backbone.

PASCAL-Person-Part数据集包含了六个人体部位类别和背景类别的标注，目标尺度和人体姿态的变化都很大。我们在这个数据集上训练一个模型，采用的是架构搜索找到的相同的DPC，使用修正Xception作为网络骨干。

Fig. 8 in the appendix shows a qualitative visualization of these results and Tab. 3 quantifies the model performance. The DPC architecture achieves state-of-the-art performance of 71.34%, representing a 3.74% improvement over the best state-of-the-art model [25], consistently outperforming other models w.r.t. all categories except the background class. Additionally, note that the DPC model does not require extra MPII training data [1], as required in [89, 25].

图8给出了可视化结果，表3给出模型性能的量化结果。DPC架构得到了目前最好的性能，71.34%，比之前最好的模型[25]改进了3.74%，在所有类别中都超过了其他模型（除了背景类别）。另外，注意DPC模型不需要在MPII训练集上进行额外训练，而[89,25]则需要。

Table 3: PASCAL-Person-Part validation set performance.

Method | mIOU
--- | ---
Liang et al. [47] | 63.57
Xia et al. [89] | 64.39
Fang et al. [25] | 67.60
DPC | 71.34

Figure 8: Visualization of predictions on PASCAL-Person-Part validation set.

### 4.5 Performance on semantic image segmentation

The PASCAL VOC 2012 benchmark [24] (augmented by [32]) involves segmenting 20 foreground object classes and one background class. We train a model on this dataset employing the same DPC identified during architecture search using the modified Xception network backbone.

PASCAL VOC 2012基准测试[24]（由[32]扩充）分割的是20个前景类别和一个背景类别。我们使用架构搜索得到的相同DPC，使用修正的Xception作为网络骨干，在这个数据集上训练了一个模型。

Fig. 9 in the appendix provides a qualitative visualization of the results and Tab. 4 quantifies the model performance on the test set. The DPC architecture outperforms previous state-of-the-art models [95, 93] by more than 1.7%, and is comparable to concurrent works [14, 96, 48]. Across semantic categories, DPC achieves state-of-the-art performance in 6 categories of the 20 categories.

图9给出了可视化结果，表4给出了模型性能在测试集上的量化结果。DPC架构超过了之前最好的模型[95,93] 1.7%，与正在进行的工作[14,96,48]性能相当。DPC在20个类别中的6个都取得了最好结果。

Table 4: PASCAL VOC 2012 test set performance.

Method | mIOU
--- | ---
EncNet [95] | 85.9
DFN [93] | 96.2
DeepLabv3+ [14] | 87.8
ExFuse [96] | 87.9
MSCI [48] | 88.0
DPC | 87.9

Figure 9: Visualization of predictions on PASCAL VOC 2012 validation set.

## 5 Conclusion 结论

This work demonstrates how architecture search techniques may be employed for problems beyond image classification – in particular, problems of dense image prediction where multi-scale processing is critical for achieving state-of-the-art performance. The application of architecture search to dense image prediction was achieved through (1) the construction of a recursive search space leveraging innovations in the dense prediction literature and (2) the construction of a fast proxy predictive of the large-scale task. The resulting learned architecture surpasses human-invented architectures across three dense image prediction tasks: scene parsing [18], person-part segmentation [16] and semantic segmentation [24]. In the first task, the resulting architecture achieved performance gains comparable to the gains witnessed in last year’s academic competition [18]. In addition, the resulting architecture is more efficient than state-of-the-art systems, requiring half of the parameters and 38% of the computational demand when using deeper Xception [17, 67, 14] as network backbone.

本文证明了，架构搜索技术可以用于图像分类之外的问题，特别是密集预测问题，其中多尺度处理对于取得最好性能非常关键。架构搜索在密集图像预测中的应用是通过如下方法得到的：(1)构建一个递归搜索空间，利用了密集预测文献中的创新；(2)构建了一个快速代理任务，可以预测大规模任务。得到的学习好的架构，在三个密集预测任务中，超过了人类发明的架构：场景解析[18]，人体部位分割[16]和语义分割[24]。在第一个任务中，得到的架构取得的进展，与去年学术比赛的进展[18]接近。另外，得到的架构比目前最好的系统效率更高，在使用Xception作为骨干网络时，只需要一半的参数量，和38%的计算量。

Several opportunities exist for improving the quality of these results. Previous work identified the design of a large and flexible search space as a critical element for achieving strong results [101, 52, 100, 65]. Expanding the search space further by increasing the number of branches B in the dense prediction cell may yield further gains. Preliminary experiments with B > 5 on the scene parsing data suggest some opportunity, although random search in an exponentially growing space becomes more challenging. The use of intelligent search algorithms such as reinforcement learning [3, 99], sequential model-based optimization [61, 52] and evolutionary methods [81, 69, 59, 90, 53, 68] may be leveraged to further improve search efficiency particularly as the space grows in size. We hope that these ideas may be ported into other domains such as depth prediction [80] and object detection [70, 55] to achieve similar gains over human-invented designs.

还有几种方法可以改进这些结果的质量。之前的工作表明，设计一个大型灵活的搜索空间，是取得很强结果的一个关键元素。进一步扩展搜索空间，增加DPC的分支数量B可能得到更好的改进。B>5时，在场景解析数据集上初步的试验表明，有一些改进的机会，但随着搜索空间增大，随机搜索计算量增长巨大，这很有挑战。智能搜索算法的使用，如强化学习，基于序列模型的优化，和演化方法，都可以用于进一步改进搜索效率，尤其是在搜索空间变大的时候。我们希望这些思想可以移植到其他领域，如深度预测和目标检测中，取得类似的进展。