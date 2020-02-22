# Convolutional neural network architecture for geometric matching

Ignacio Rocco et al. DI ENS

## 0. Abstract

We address the problem of determining correspondences between two images in agreement with a geometric model such as an affine or thin-plate spline transformation, and estimating its parameters. The contributions of this work are three-fold. First, we propose a convolutional neural network architecture for geometric matching. The architecture is based on three main components that mimic the standard steps of feature extraction, matching and simultaneous in-lier detection and model parameter estimation, while being trainable end-to-end. Second, we demonstrate that the network parameters can be trained from synthetically generated imagery without the need for manual annotation and that our matching layer significantly increases generalization capabilities to never seen before images. Finally, we show that the same model can perform both instance-level and category-level matching giving state-of-the-art results on the challenging Proposal Flow dataset.

我们处理的问题是，确定两幅图像在一个几何模型下具有对应性，如仿射变换，或薄板样条变换，并估计出其参数。本文的贡献有三点。第一，我们提出了一种几何配准的CNN架构。这个架构主要包括三个部分，模仿了特征提取，匹配并同时检测偏移、估计模型参数这些标准步骤，并可以进行端到端的训练。第二，我们证明了，网络参数可以用合成图像训练得到，而且不需要手动标注，我们的匹配层明显增加了网络的泛化性能，可以在之前从未见过的图像中得到较好效果。最后，我们展示了，同样的模型可以进行实例级和类别级的匹配，在很有挑战的Proposal Flow数据集上得到了目前最好的结果。

## 1. Introduction

Estimating correspondences between images is one of the fundamental problems in computer vision [20, 26] with applications ranging from large-scale 3D reconstruction [2] to image manipulation [22] and semantic segmentation [44]. Traditionally, correspondences consistent with a geometric model such as epipolar geometry or planar affine transformation, are computed by detecting and matching local features (such as SIFT [40] or HOG [12, 23]), followed by pruning incorrect matches using local geometric constraints [45, 49] and robust estimation of a global geometric transformation using algorithms such as RANSAC [19] or Hough transform [34, 36, 40]. This approach works well in many cases but fails in situations that exhibit (i) large changes of depicted appearance due to e.g. intra-class variation [23], or (ii) large changes of scene layout or non-rigid deformations that require complex geometric models with many parameters which are hard to estimate in a manner robust to outliers.

在计算机视觉中，估计图像中的对应性，是一个很基础的问题，其应用有大规模3D重建[2]，图像manipulation[22]，和语义分割[44]。传统上，在对极几何或平面仿射变换的几何模型中一致对应的图像，其计算是通过检测和匹配局部特征（如SIFT和HOG），然后使用局部几何约束将不正确的匹配剪掉，使用RANSAC或Hough变换这样的算法稳健的估计全局几何变换。这种方法在很多情况下都挺好，但在一些情况下会失败，如(i)表现出的外表变化很大，如类别间的变化[23]，或(ii)场景布局变化很大，或有非刚性形变，需要很多参数的复杂几何模型来模拟，要想对于异常情况很稳健的话，这是很难估计的。

In this work we build on the traditional approach and develop a convolutional neural network (CNN) architecture that mimics the standard matching process. First, we replace the standard local features with powerful trainable convolutional neural network features [33, 48], which allows us to handle large changes of appearance between the matched images. Second, we develop trainable matching and transformation estimation layers that can cope with noisy and incorrect matches in a robust way, mimicking the good practices in feature matching such as the second nearest neighbor test [40], neighborhood consensus [45, 49] and Hough transform-like estimation [34, 36, 40].

在本文中，我们基于传统方法构建提出了一种CNN架构，模仿标准的匹配过程。首先，我们将标准的局部特征替换为强力的可训练的CNN特征，使我们可以处理待匹配图像间较大的外观变化。第二，我们提出了可训练的匹配和形变估计层，可以稳健的处理含噪的不正确的匹配，模仿特征匹配中的一些实践，如第二最近邻测试[40]，近邻审查[45,49]和Hough变换类的估计[34,36,40]。

The outcome is a convolutional neural network architecture trainable for the end task of geometric matching, which can handle large appearance changes, and is therefore suitable for both instance-level and category-level matching problems.

本文的成果是一个可训练的用于几何匹配的CNN架构，可以处理很大的外观变化，因此可以进行实例级的和类别级的匹配问题。

## 2. Related work

The classical approach for finding correspondences involves identifying interest points and computing local descriptors around these points [9, 10, 25, 39, 40, 41, 45]. While this approach performs relatively well for instance-level matching, the feature detectors and descriptors lack the generalization ability for category-level matching.

找到对应性的经典方法是，识别出感兴趣的点，并在这些点中计算局部描述子[9,10,25,39,40,41,45]。虽然这种方法对于实例级的匹配表现较好，但这些特征提取器和描述子缺少对于类别级匹配的泛化能力。

Recently, convolutional neural networks have been used to learn powerful feature descriptors which are more robust to appearance changes than the classical descriptors [8, 24, 29, 47, 54]. However, these works still divide the image into a set of local patches and extract a descriptor individually from each patch. Extracted descriptors are then compared with an appropriate distance measure [8, 29, 47], by directly outputting a similarity score [24, 54], or even by directly outputting a binary matching/non-matching decision [3].

最近，CNN经常用于学习强力的特征描述子，与经典描述子相比，对于外观变化更为稳健。但是，这些工作仍然将图像分割成一些局部剪切块，从每个图像块提取描述子。提取出的描述子然后用合适的距离度量进行比较，直接输出一个相似度分数，或甚至直接输出一个二值匹配/非匹配决策[3]。

In this work, we take a different approach, treating the image as a whole, instead of a set of patches. Our approach has the advantage of capturing the interaction of the different parts of the image in a greater extent, which is not possible when the image is divided into a set of local regions.

本文中，我们采用了不同的方法，将整幅图像作为一个整体，而不是很多图像块。我们方法的优点是，在更大的程度上捕获图像不同部分的交互，这在图像分割成很多局部区域时，是不可能的。

Related are also network architectures for estimating inter-frame motion in video [18, 50, 52] or instance-level homography estimation [14], however their goal is very different from ours, targeting high-precision correspondence with very limited appearance variation and background clutter. Closer to us is the network architecture of [30] which, however, tackles a different problem of fine-grained category-level matching (different species of birds) with limited background clutter and small translations and scale changes, as their objects are largely centered in the image. In addition, their architecture is based on a different matching layer, which we show not to perform as well as the matching layer used in our work.

有关的还有，估计视频中帧间运动或实例级单应性估计的网络架构，但是其目标与我们的非常不同，是在外观变化很有限、背景杂乱程度也有限的情况下，进行高精度的对应性研究。与我们更接近的是，[30]的网络架构，但其处理的是不同的问题，细粒度类别级的匹配（不同的鸟类），其背景杂乱程度有限，平移和尺度变化也较小，因为其目标主要在图像中央。另外，其架构是基于不同的匹配层的，我们会证明，我们的匹配层其效果要更好。

Some works, such as [10, 15, 23, 31, 37, 38], have addressed the hard problem of category-level matching, but rely on traditional non-trainable optimization for matching [10, 15, 31, 37, 38], or guide the matching using object proposals [23]. On the contrary, our approach is fully trainable in an end-to-end manner and does not require any optimization procedure at evaluation time, or guidance by object proposals.

一些工作，如[10,15,23,31,37,38]，处理的是类别级匹配的困难问题，但依赖的是传统的不能训练的优化方法进行匹配[10,15,31,37,38]，或用目标建议来引导匹配[23]。相反的，我们的方法是端到端完全可训练的，在评估时不需要优化过程，或者目标建议的引导。

Others [35, 46, 55] have addressed the problems of instance and category-level correspondence by performing joint image alignment. However, these methods differ from ours as they: (i) require class labels; (ii) don’t use CNN features; (iii) jointly align a large set of images, while we align image pairs; and (iv) don’t use a trainable CNN architecture for alignment as we do.

其他的[35,46,55]通过进行联合图像对齐，来处理实例级和类别级的对应性。但是，这些方法与我们的不同，(i)需要类别标签，(ii)没有使用CNN特征，(iii)联合对齐大型图像集，而我们是对图像对进行对齐，(iv)没有使用可训练的CNN架构进行对齐，我们则是这样进行的。

## 3. Architecture for geometric matching

In this section, we introduce a new convolutional neural network architecture for estimating parameters of a geometric transformation between two input images. The architecture is designed to mimic the classical computer vision pipeline (e.g. [42]), while using differentiable modules so that it is trainable end-to-end for the geometry estimation task. The classical approach consists of the following stages: (i) local descriptors (e.g. SIFT) are extracted from both input images, (ii) the descriptors are matched across images to form a set of tentative correspondences, which are then used to (iii) robustly estimate the parameters of the geometric model using RANSAC or Hough voting.

本节中，我们提出一种新的CNN架构，估计两幅输入图像间的几何变形的参数。架构的设计是用来模拟经典计算机视觉流程的（如[42]），同时使用的是可微分模块，这样对于几何估计的任务就可以进行端到端的训练。经典方法包含如下阶段：(i)从输入图像中提取出局部描述子（如SIFT），(ii)在不同图像间匹配描述子，形成临时的对应性，然后用于下一步骤，(iii)使用RANSAC或Hough投票，稳健的估计几何形变的参数。

Our architecture, illustrated in Fig. 2, mimics this process by: (i) passing input images I_A and I_B through a siamese architecture consisting of convolutional layers, thus extracting feature maps f_A and f_B which are analogous to dense local descriptors, (ii) matching the feature maps (“descriptors”) across images into a tentative correspondence map f_AB , followed by a (iii) regression network which directly outputs the parameters of the geometric model, $\hat θ$, in a robust manner. The inputs to the network are the two images, and the outputs are the parameters of the chosen geometric model, e.g. a 6-D vector for an affine transformation.

我们的架构，如图2所示，通过如下过程进行模拟：(i)将输入图像I_A和I_B送入siamese架构，这个架构由卷积层组成，然后提取特征图f_A和f_B，这可以类比于密集局部描述子，(ii)匹配不同图像之间的特征图（描述子），形成临时性的对应图f_AB，然后，(iii)回归网络很稳健的直接输出几何模型的参数$\hat θ$。网络的输入是两幅图像，输出是选定的几何模型的参数，如仿射变换的6D向量。

In the following, we describe each of the three stages in detail. 下面，我们对三个阶段分别进行详述。

Figure 2: Diagram of the proposed architecture. Images IA and IB are passed through feature extraction networks which have tied parameters W, followed by a matching network which matches the descriptors. The output of the matching network is passed through a regression network which outputs the parameters of the geometric transformation.

### 3.1. Feature extraction

The first stage of the pipeline is feature extraction, for which we use a standard CNN architecture. A CNN without fully connected layers takes an input image and produces a feature map $f ∈ R^{h×w×d}$, which can be interpreted as a h × w dense spatial grid of d-dimensional local descriptors. A similar interpretation has been used previously in instance retrieval [4, 6, 7, 21] demonstrating high discriminative power of CNN-based descriptors. Thus, for feature extraction we use the VGG-16 network [48], cropped at the pool4 layer (before the ReLU unit), followed by per-feature L2-normalization. We use a pre-trained model, originally trained on ImageNet [13] for the task of image classification. As shown in Fig. 2, the feature extraction network is duplicated and arranged in a siamese configuration such that the two input images are passed through two identical networks which share parameters.

流程的第一阶段是特征提取，这里我们使用标准的CNN架构。没有全连接层的CNN，以图像为输入，输出一个特征图$f ∈ R^{h×w×d}$，这可以解释为一个h × w的密集空间网格的d维局部描述子。类似的解释之前曾经在实例检索[4,6,7,21]中使用，展示了基于CNN的描述子的极高的区分力。因此，我们使用VGG-16网络进行特征提取，在pool4层进行剪切（在ReLU单元之前），然后进行逐个特征的L2归一化。我们使用了一个预训练模型，是在ImageNet上训练进行图像分类的。如图2所示，特征提取网络进行了复制，安排为siamese配置，这样两幅输入图像都通过相同的网络，这两个网络是共享参数的。

### 3.2. Matching network

The image features produced by the feature extraction networks should be combined into a single tensor as input to the regressor network to estimate the geometric transformation. We first describe the classical approach for generating tentative correspondences, and then present our matching layer which mimics this process.

特征提取网络生成的图像特征，应当合并成为一个张量，输入到回归网络中，估计几何变换。我们首先描述经典的生成临时对应性的方法，然后给出我们模仿这个过程的匹配层。

**Tentative matches in classical geometry estimation**. Classical methods start by computing similarities between all pairs of descriptors across the two images. From this point on, the original descriptors are discarded as all the necessary information for geometry estimation is contained in the pairwise descriptor similarities and their spatial locations. Secondly, the pairs are pruned by either thresholding the similarity values, or, more commonly, only keeping the matches which involve the nearest (most similar) neighbors. Furthermore, the second nearest neighbor test [40] prunes the matches further by requiring that the match strength is significantly stronger than the second best match involving the same descriptor, which is very effective at discarding ambiguous matches.

**经典几何估计中的临时对应**。经典方法首先在两幅图像中的所有描述子对中计算相似性。从这一点开始，原始的描述子会被抛弃，因为所有几何估计必须的信息都包含在成对的描述子相似性及其空间位置之中了。第二，这些相似性对通过设置阈值，或更常见的只保留最近邻的样本（最相似的样本），来进行删减。而且，第二近邻测试[40]进一步对这些匹配对进行了删减，要求匹配强度要明显比同样描述子的第二最佳匹配要强很多，这在抛弃模糊匹配的过程中非常有效。

**Matching layer**. Our matching layer applies a similar procedure. Analogously to the classical approach, only descriptor similarities and their spatial locations should be considered for geometry estimation, and not the original descriptors themselves.

**匹配层**。我们的匹配层，应用的是类似的过程。从传统方法类比过来，对于几何估计问题，只需要考虑描述子相似度及其空间位置，而不需要考虑原始的描述子本身。

To achieve this, we propose to use a correlation layer followed by normalization. Firstly, all pairs of similarities between descriptors are computed in the correlation layer. Secondly, similarity scores are processed and normalized such that ambiguous matches are strongly down-weighted.

为取得这个目的，我们提出来使用一个相关层，然后进行归一化。首先，所有描述子之间的相似度对都在相关层中进行计算。第二，相似度分数进行处理和归一化，这样有歧义的匹配其权重得到极大的降低。

In more detail, given L2-normalized dense feature maps $f_A,f_B ∈ R^{h×w×d}$, the correlation map $c_{AB} ∈ R^{h×w×(h×w)}$ outputted by the correlation layer contains at each position the scalar product of a pair of individual descriptors $f_A ∈ f_A$ and $f_B ∈ f_B$, as detailed in Eq. (1).

更细节上来说，给定L2归一化的密集特征图$f_A,f_B ∈ R^{h×w×d}$，相关层输出的相关图$c_{AB} ∈ R^{h×w×(h×w)}$，就是在每个点位置上，计算一对描述子$f_A ∈ f_A$和$f_B ∈ f_B$的点积，如式(1)所示。

$$c_{AB}(i,j,k)=f_B(i,j)^Tf_A(i_k,j_k)$$(1)

where (i, j) and ($i_k, j_k$) indicate the individual feature positions in the h×w dense feature maps, and $k = h(j_k−1)+i_k$ is an auxiliary indexing variable for ($i_k, j_k$).

其中(i, j)和($i_k, j_k$)是在h×w的密集特征图上的单个特征位置，$k = h(j_k−1)+i_k$是对位置($i_k, j_k$)上的一个辅助索引变量。

A diagram of the correlation layer is presented in Fig. 3. Note that at a particular position (i, j), the correlation map $c_{AB}$ contains the similarities between $f_B$ at that position and all the features of $f_A$.

关联层的一个示意图如图3所示。注意，在某个特征位置(i, j)，关联图$c_{AB}$包含的是，$f_B$在这个位置上与$f_A$所有特征的相似度。

As is done in the classical methods for tentative correspondence estimation, it is important to postprocess the pairwise similarity scores to remove ambiguous matches. To this end, we apply a channel-wise normalization of the correlation map at each spatial location to produce the final tentative correspondence map $f_{AB}$. The normalization is performed by ReLU, to zero out negative correlations, followed by L2-normalization, which has two desirable effects. First, let us consider the case when descriptor $f_B$ correlates well with only a single feature in $f_A$. In this case, the normalization will amplify the score of the match, akin to the nearest neighbor matching in classical geometry estimation. Second, in the case of the descriptor $f_B$ matching multiple features in $f_A$ due to the existence of clutter or repetitive patterns, matching scores will be down-weighted similarly to the second nearest neighbor test [40]. However, note that both the correlation and the normalization operations are differentiable with respect to the input descriptors, which facilitates backpropagation thus enabling end-to-end learning.

如同在经典方法中的临时对应性估计中所做的一样，对成对相似性分数进行后处理非常重要，这样可以删掉模棱两可的匹配。为此，我们对关联图在每个空间位置上进行逐通道的归一化，以生成最终的临时对应性图$f_{AB}$。归一化过程首先进行ReLU，以将负的关联值去掉，然后进行L2归一化，这有两个较好的效果。首先，我们考虑当描述子$f_B$只与$f_A$中的一个特征关联的很好的情况。这种情况下，归一化会将匹配值放大，与经典几何估计中的最近邻匹配类似。第二，描述子$f_B$与$f_A$中的多个特征都匹配的情况，这可能是由于群聚或重复模式的存在，那么匹配分数会类似的会被减少权重给第二个最近邻测试[40]。但是，注意关联运算和归一化运算对于输入描述子都是可微分的，这就可以很顺利的进行反向传播，因此可以进行端到端的学习。

**Discussion**. The first step of our matching layer, namely the correlation layer, is somewhat similar to layers used in DeepMatching [52] and FlowNet [18]. However, Deep-Matching [52] only uses deep RGB patches and no part of their architecture is trainable. FlowNet [18] uses a spatially constrained correlation layer such that similarities are only computed in a restricted spatial neighborhood thus limiting the range of geometric transformations that can be captured. This is acceptable for their task of learning to estimate optical flow, but is inappropriate for larger transformations that we consider in this work. Furthermore, neither of these methods performs score normalization, which we find to be crucial in dealing with cluttered scenes.

**讨论**。我们匹配层的第一步，即关联层，与在DeepMatching[52]和FlowNet[18]中使用的类似。但是，Deep-Matching[52]只使用深度RGB块，其架构是不可训练的。FlowNet[18]使用的是一个空间约束的关联层，这样相似度只在一个受限的空间邻域进行计算，因此限制了可以捕获到的几何变换的范围。这对于他们的应用是可以接受的，即学习估计光流，但对于更大的变换来说就不太合适了，而我们考虑的就是更大的变换的情况。而且，这些方法都没有进行分数归一化，我们发现这对于处理群聚的场景是非常关键的。

Previous works have used other matching layers to combine descriptors across images, namely simple concatenation of descriptors along the channel dimension [14] or subtraction [30]. However, these approaches suffer from two problems. First, as following layers are typically convolutional, these methods also struggle to handle large transformations as they are unable to detect long-range matches. Second, when concatenating or subtracting descriptors, instead of computing pairwise descriptor similarities as is commonly done in classical geometry estimation and mimicked by the correlation layer, image content information is directly outputted. To further illustrate why this can be problematic, consider two pairs of images that are related with the same geometric transformation – the concatenation and subtraction strategies will produce different outputs for the two cases, making it hard for the regressor to deduce the geometric transformation. In contrast, the correlation layer output is likely to produce similar correlation maps for the two cases, regardless of the image content, thus simplifying the problem for the regressor. In line with this intuition, in Sec. 5.5 we show that the concatenation and subtraction methods indeed have difficulties generalizing beyond the training set, while our correlation layer achieves generalization yielding superior results.

之前的工作使用过其他的匹配层，来在不同图像之间合并描述子，即简单的沿着通道维度拼接描述子[14]或进行相减。但是，这些方法存在两个问题。第一，因为后续的层一般是卷积层，这些方法一般很难处理大的变换，因为不能检测到长程的匹配。第二，当描述子进行拼接或相减，而不是计算成对的描述子相似度（这在经典几何估计方法中是这样的，并由关联层进行模仿），图像内容信息是直接输出的。为进一步描述这为什么会是一个问题，考虑两对图像，都用同样的几何变换关联起来，拼接和相减的策略，会对这两种情况产生不同的输出，使回归器很难推断出几何形变。比较之下，关联层的输出对于这两种情况很可能输出类似的关联图，与图像内容无关，因此简化了回归器要解决的问题。与这种直觉相符合，在5.5节中，我们展示了，拼接和相减的方法泛化到训练集之外确实是有困难的，而我们的关联层得到了很好的泛化效果。

### 3.3. Regression network

The normalized correlation map is passed through a regression network which directly estimates parameters of the geometric transformation relating the two input images. In classical geometry estimation, this step consists of robustly estimating the transformation from the list of tentative correspondences. Local geometric constraints are often used to further prune the list of tentative matches [45, 49] by only retaining matches which are consistent with other matches in their spatial neighborhood. Final geometry estimation is done by RANSAC [19] or Hough voting [34, 36, 40].

归一化的关联图送入回归网络，直接估计与两幅输入图像相关的几何形变参数。在经典的几何估计中，这一步的主要工作是，稳健的从临时对应性列表中估计变形。局部几何约束通常用于进一步剪掉临时性匹配[45,49]，其方法是只保留与其他匹配在其空间邻域中一致的匹配。最后的几何估计由RANSAC或Hough投票来进行。

We again mimic the classical approach using a neural network, where we stack two blocks of convolutional layers, followed by batch normalization [27] and the ReLU non-linearity, and add a final fully connected layer which regresses to the parameters of the transformation, as shown in Fig. 4. The intuition behind this architecture is that the estimation is performed in a bottom-up manner somewhat like Hough voting, where early convolutional layers vote for candidate transformations, and these are then processed by the later layers to aggregate the votes. The first convolutional layers can also enforce local neighborhood consensus [45, 49] by learning filters which only fire if nearby descriptors in image A are matched to nearby descriptors in image B, and we show qualitative evidence in Sec. 5.5 that this indeed does happen.

我们再一次使用神经网络来模仿经典方法，我们将两个模块堆积起来，模块中是卷积层，后面接上批归一化和ReLU非线性，两个模块后加上一个最后的全卷积层，最终回归出变换的参数，如图4所示。这个架构背后的直觉是，这个估计是以一种自上而下的方式进行的，有些像Hough投票，其中前面的卷积层对候选变换进行投票，然后这些送入后续的层以积聚投票。第一个卷积层也可以加上局部邻域的共识，学习到的滤波器只会在下面的情况fire，即图像A中附近的描述子与图像B中附近的描述子是相互匹配的，我们在5.5节中给出量化的证据，这个确实是这样发生的。

Figure 4: Architecture of the regression network. It is composed of two convolutional layers without padding and stride equal to 1, followed by batch normalization and ReLU, and a final fully connected layer which regresses to the P transformation parameters.

**Discussion**. A potential alternative to a convolutional regression network is to use fully connected layers. However, as the input correlation map size is quadratic in the number of image features, such a network would be hard to train due to a large number of parameters that would need to be learned, and it would not be scalable due to occupying too much memory and being too slow to use. It should be noted that even though the layers in our architecture are convolutional, the regressor can learn to estimate large transformations. This is because one spatial location in the correlation map contains similarity scores between the corresponding feature in image B and all the features in image A (c.f. equation (1)), and not just the local neighborhood as in [18].

**讨论**。卷积回归网络的一个潜在替代是，使用全卷积网络。但是，由于输入的关联图的大小与图像特征的数量是成平方的关系，这样一个网络是很难训练的，因为需要去学习大量参数，而由于耗用的内存太多，所以无法进行扩展，而且使用起来也太慢。应当指出，即使我们架构中的层都是卷积的，回归器也可以学习来对大的形变进行估计。这是因为，关联图中的一个空间位置，包含的相似度分数，是图像B中的对应特征与图像A中的所有特征之间的相似性分数（参考式1），而并不是像[18]中的局部邻域。

### 3.4. Hierarchy of transformations

Another commonly used approach when estimating image to image transformations is to start by estimating a simple transformation and then progressively increase the model complexity, refining the estimates along the way [10, 39, 42]. The motivation behind this method is that estimating a very complex transformation could be hard and computationally inefficient in the presence of clutter, so a robust and fast rough estimate of a simpler transformation can be used as a starting point, also regularizing the subsequent estimation of the more complex transformation.

当估计图像到图像的变换时，另一种常见的使用方法是，开始先估计一个简单的变换，然后逐渐的增加模型复杂度，改进估计。这种方法背后的动机是，估计一个非常复杂的变换可能是很难的，在存在群聚的情况时，计算效率比较低，所以稳健、快速的大致估计一个简单一些的变换，可以用作一个初始值，同时对后续的更复杂的变换进行正则化。

We follow the same good practice and start by estimating an affine transformation, which is a 6 degree of freedom linear transformation capable of modeling translation, rotation, non-isotropic scaling and shear. The estimated affine transformation is then used to align image B to image A using an image resampling layer [28]. The aligned images are then passed through a second geometry estimation network which estimates 18 parameters of a thin-plate spline transformation. The final estimate of the geometric transformation is then obtained by composing the two transformations, which is also a thin-plate spline. The process is illustrated in Fig. 5.

我们按照相同的方式进行，先估计一个仿射变换，这是一个6自由度的自由线性变换，可以对平移、旋转、非各向同性的缩放和剪切进行建模。估计得到的仿射变换，然后使用一个图像重采样层，将图像B对齐到图像A[28]。对齐的图像然后送入第二个几何估计网络，估计18个参数的薄板样条变换。最终的几何变形估计，是通过将两个变换组合到一起得到的，也是一个薄板样条。这个过程如图5所示。

Figure 5: Estimating progressively more complex geometric transformations. Images A and B are passed through a network which estimates an affine transformation with parameters $\hat θ_{Aff}$ (see Fig. 2). Image A is then warped using this transformation to roughly align with B, and passed along with B through a second network which estimates a thin-plate spline (TPS) transformation that refines the alignment.

## 4. Training

In order to train the parameters of our geometric matching CNN, it is necessary to design the appropriate loss function, and to use suitable training data. We address these two important points next.

为训练几何配准CNN的参数，需要设计合适的损失函数，使用合适的训练数据。下面我们处理这两个重要的点。

### 4.1. Loss function

We assume a fully supervised setting, where the training data consists of pairs of images and the desired outputs in the form of the parameters $θ_{GT}$ of the ground-truth geometric transformation. The loss function L is designed to compare the estimated transformation $\hat θ$ with the ground-truth transformation $θ_{GT}$ and, more importantly, compute the gradient of the loss function with respect to the estimates $∂L/∂\hat θ$. This gradient is then used in a standard manner
to learn the network parameters which minimize the loss function by using backpropagation and Stochastic Gradient Descent.

我们假设是完全监督的设置，其中训练数据是由成对图像组成的，期望的输出是真值几何变换的参数$θ_{GT}$。损失函数L的设计是用于比较估计的变换参数$\hat θ$与真值变换参数$θ_{GT}$，更重要的是，计算损失函数对于估计参数的梯度$∂L/∂\hat θ$。这个梯度然后用于标准的学习网络参数的方式，即通过反向传播和SGD来最小化损失函数。

It is desired for the loss to be general and not specific to a particular type of geometric model, so that it can be used for estimating affine, homography, thin-plate spline or any other geometric transformation. Furthermore, the loss should be independent of the parametrization of the transformation and thus should not directly operate on the parameter values themselves. We address all these design constraints by measuring loss on an imaginary grid of points which is being deformed by the transformation. Namely, we construct a grid of points in image A, transform it using the ground truth and neural network estimated transformations $T_{θ_{GT}}$ and $T_{\hat θ}$ with parameters $θ_{GT}$ and θ, respectively, and measure the discrepancy between the two transformed grids by summing the squared distances between the corresponding grid points:

如果损失函数是一般性的，而不是对特定几何模型专用的，则就非常理想，这样就可以用于估计仿射、单对应性、薄板样条或其他任何几何变换。而且，损失函数应当独立于变换的参数化，所以不应当直接对参数值本身进行运算。我们处理这些设计约束的方法是，在一个假想的被变换变形的网格的点之上度量损失。即，我们在图像A中构建一个点的网格，用真值变换进行变换，神经网络分别用参数$θ_{GT}$和θ估计变换$T_{θ_{GT}}$和$T_{\hat θ}$，在对应的网格点上求其距离平方之和，在这两个变换过的网格上度量其差异性：

$$L(\hat θ, θ_{GT}) = \frac {1}{N} \sum_{i=1}^N d(T_{\hat θ} (g_i), T_{θ_{GT}} (g_i))^2$$(2)

where G = {$g_i$} = {$(x_i,y_i)$} is the uniform grid used, and N = |G|. We define the grid as having $x_i,y_i$ ∈ {s : s = −1 + 0.1 × n,n ∈ {0,1,...,20}}, that is to say, each coordinate belongs to a partition of [−1, 1] in equally spaced subintervals of steps 0.1. Note that we construct the coordinate system such that the center of the image is at (0, 0) and that the width and height of the image are equal to 2, i.e. the bottom left and top right corners have coordinates (−1, −1) and (1, 1), respectively.

其中G = {$g_i$} = {$(x_i,y_i)$}是使用的统一网格，N = |G|。我们将网格定义为$x_i,y_i$ ∈ {s : s = −1 + 0.1 × n,n ∈ {0,1,...,20}}，也就是说，每个坐标值都属于[-1, 1]的一个分割中，以0.1为间隔进行等距分割。注意我们构建的坐标系系统，使得图像中央是在(0,0)点上的，图像的宽度和高度都等于2，即，左下和右上的坐标分别为(−1, −1)和(1, 1)。

The gradient of the loss function with respect to the transformation parameters, needed to perform backpropagation in order to learn network weights, can be computed easily if the location of the transformed grid points $T_{\hat θ} (g_i)$ is differentiable with respect to $\hat θ$. This is commonly the case, for example, when T is an affine transformation, $T_{\hat θ} (g_i)$ is linear in parameters $\hat θ$ and therefore the loss can be differentiated in a straightforward manner.

损失函数对变换参数的梯度，需要进行反向传播，才能学到网络参数，如果被变换的网格点的位置$T_{\hat θ} (g_i)$对于$\hat θ$是可微分的话，那么计算起来就很容易了。通常来说都是这个情况，比如，当T是仿射变换时，$T_{\hat θ} (g_i)$对$\hat θ$是线性的，因此损失函数可以很直接的进行求取微分。

### 4.2. Training from synthetic transformations

Our training procedure requires fully supervised training data consisting of image pairs and a known geometric relation. Training CNNs usually requires a lot of data, and no public datasets exist that contain many image pairs annotated with their geometric transformation. Therefore, we opt for training from synthetically generated data, which gives us the flexibility to gather as many training examples as needed, for any 2-D geometric transformation of interest. We generate each training pair (I_A, I_B), by sampling I_A from a public image dataset, and generating I_B by applying a random transformation T_{θ_{GT}} to I_A. More precisely, I_A is created from the central crop of the original image, while I_B is created by transforming the original image with added symmetrical padding in order to avoid border artifacts; the procedure is shown in Fig. 6.

我们的训练过程，需要完全监督的训练数据，要由已知几何关系的图像对组成。训练CNNs通常需要很多数据，并没有这样的数据集，有很多图像对，并标注了其几何关系。因此，我们选择用合成生成的数据来进行训练，这使得我们可以对任意感兴趣的2-D几何变换，任意收集所需的训练样本。我们从一个公开图像数据集中选择样本I_A，对I_A应用一个随机变换T_{θ_{GT}}，生成I_B，生成每个训练图像对(I_A, I_B)。更精确的，I_A是从原始图像的中间剪切块创建的，而I_B是将原始图像增加对称的padding生成的，以避免形成边界伪影；这个过程如图6所示。

## 5. Experimental results

In this section we describe our datasets, give implementation details, and compare our method to baselines and the state-of-the-art. We also provide further insights into the components of our architecture.

本节中，我们描述数据集，给出实现细节，将我们的方法与基准、目前最好的结果进行比较。我们还对我们架构的组件给出更多的洞见。

### 5.1. Evaluation dataset and performance measure

Quantitative evaluation of our method is performed on the Proposal Flow dataset of Ham et al. [23]. The dataset contains 900 image pairs depicting different instances of the same class, such as ducks and cars, but with large intraclass variations, e.g. the cars are often of different make, or the ducks can be of different subspecies. Furthermore, the images contain significant background clutter, as can be seen in Fig. 8. The task is to predict the locations of predefined keypoints from image A in image B. We do so by estimating a geometric transformation that warps image A into image B, and applying the same transformation to the keypoint locations. We follow the standard evaluation metric used for this benchmark, i.e. the average probability of correct keypoint (PCK) [53], being the proportion of keypoints that are correctly matched. A keypoint is considered to be matched correctly if its predicted location is within a distance of α · max(h, w) of the target keypoint position, where α = 0.1 and h and w are the height and width of the object bounding box, respectively.

我们方法的量化评估，在Ham等[23]的Proposal Flow数据集上进行。数据集包含900对图像，是同类目标的不同实例，如鸭子和车，但有很大的类间变化，如，车通常有很多不同的品牌，或者鸭子可以有很多不同的子属。而且，图像包含显著的背景杂乱，如图8所示。我们的任务是从图像A中预定义的关键点预测出图像B中这些点。我们估计一个几何形变，对图像A进行这种形变，形成图像B，并对这些关键点位置进行同样的变换。我们采用在这个基准测试中的标准评估度量，即，正确关键点概率的平均(PCK, probability of correct keypoint)[53]。一个关键点的正确匹配，是预测的位置在目标关键点位置的一定距离之内，距离为α · max(h, w)，其中α = 0.1，h/w分别是目标边界框的宽和高。

### 5.2. Training data 训练数据

Two different training datasets for the affine and thin-plate spline stages, dubbed StreetView-synth-aff and StreetView-synth-tps respectively, were generated by applying synthetic transformations to images from the Tokyo Time Machine dataset [4] which contains Google Street View images of Tokyo.

对仿射变换和薄板样条变换，有两个不同的训练数据集，分别是StreetView-synth-aff和StreetView-synth-tps，是通过对Tokyo Time Machine数据集进行合成的变换得到的，这个数据集是由东京市的Google街景构成的。

Each synthetically generated dataset contains 40k images, divided into 20k for training and 20k for validation. The ground truth transformation parameters were sampled independently from reasonable ranges, e.g. for the affine transformation we sample the relative scale change of up to 2×, while for thin-plate spline we randomly jitter a 3 × 3 grid of control points by independently translating each point by up to one quarter of the image size in all directions.

每个合成生成的数据集，包含40k图像，20k用于训练，20k用于validation。真值变换参数在一定范围内进行独立的采样，如，对于仿射变换，我们对相对尺度变换取样为最多2x，而对于薄板样条，我们随机对3 × 3网格的控制点进行抖动，对每个点在各个方向上进行独立的平移，最多是图像大小的1/4。

In addition, a second training dataset for the affine stage was generated, created from the training set of Pascal VOC 2011 [16] which we dubbed Pascal-synth-aff. In Sec. 5.5, we compare the performance of networks trained with StreetView-synth-aff and Pascal-synth-aff and demonstrate the generalization capabilities of our approach.

另外，对仿射阶段还生成了另一个训练集，是从Pascal VOC 2011的训练集中创建的，我们称之为Pascal-synth-aff。在5.5节，我们用StreetView-synth-aff和Pascal-synth-aff进行了训练，然后比较了其网络性能，证明了我们方法的泛化能力。

### 5.3. Implementation details

We use the MatConvNet library [51] and train the networks with stochastic gradient descent, with learning rate 10^−3, momentum 0.9, no weight decay and batch size of 16. There is no need for jittering as instead of data augmentation we can simply generate more synthetic training data. Input images are resized to 227 × 227 producing 15×15 feature maps that are passed into the matching layer. The affine and thin-plate spline stages are trained independently with the StreetView-synth-aff and StreetView-synth-tps datasets, respectively. Both stages are trained until convergence which typically occurs after 10 epochs, and takes 12 hours on a single GPU. Our final method for estimating affine transformations uses an ensemble of two networks that independently regress the parameters, which are then averaged to produce the final affine estimate. The two networks were trained on different ranges of affine transformations. As in Fig. 5, the estimated affine transformation is used to warp image A and pass it together with image B to a second network which estimates the thin-plate spline transformation. All training and evaluation code, as well as our trained networks, are online at [1].

我们使用MatConvNet库，用SGD训练网络，学习速率10^−3，动量0.9，权重衰减没有，批规模为16。这里数据扩增时就不需要jitter了，因为我们可以直接生成更多的合成训练数据。输入图像大小变换到227 × 227，生成15×15特征图，送入匹配层。仿射和薄板样条阶段独立进行训练，分别使用StreetView-synth-aff和StreetView-synth-tps数据集。两个阶段都训练到收敛，通常在10轮训练后收敛，在单GPU上耗费12小时。我们估计仿射参数的最终方法，是使用了两个网络的集成，它们都独立的对参数进行回归，然后两组参数进行平均，以得到最终的仿射估计。两个网络是在不同的仿射变换范围内进行训练的。如图5所示，估计得到的仿射变换用于对图像A进行变形，然后和图像B一起送入第二个网络，来估计薄板样条变换。所有的训练和评估代码，以及训练好的网络，都已经开源。

### 5.4. Comparison to state-of-the-art

We compare our method against SIFT Flow [37], Graph-matching kernels (GMK) [15], Deformable spatial pyramid matching (DSP) [31], DeepFlow [43], and all three variants of Proposal Flow (NAM, PHM, LOM) [23]. As shown in Tab. 1, our method outperforms all others and sets the new state-of-the-art on this data. The best competing methods are based on Proposal Flow and make use of object proposals, which enables them to guide the matching towards regions of images that contain objects. Their performance varies significantly with the choice of the object proposal method, illustrating the importance of this guided matching. On the contrary, our method does not use any guiding, but it still manages to outperform even the best Proposal Flow and object proposal combination.

我们将我们的方法与SIFT Flow，Graph-matching核(GMK)，Deformable spatial pyramid matching (DSP)，DeepFlow，以及Proposal Flow的三个变体(NAM, PHM, LOM)进行了比较。如表1所示，我们的方法超过了所有其他方法，在这些数据中是目前最好的结果。最有竞争力的方法是基于Proposal Flow的，使用了目标建议，这使其可以将匹配导引到包含目标的图像区域中。选择不同的目标建议方法，其算法性能变化很大，说明这种引导配准算法的重要性。相反的是，我们的方法没有使用任何导引，但仍然超过了表现最佳的Proposal Flow和目标建议组合。

Furthermore, we also compare to affine transformations estimated with RANSAC using the same descriptors as our method (VGG-16 pool4). The parameters of this baseline have been tuned extensively to obtain the best result by adjusting the thresholds for the second nearest neighbor test and by pruning proposal transformations which are outside of the range of likely transformations. Our affine estimator outperforms the RANSAC baseline on this task with 49% (ours) compared to 47% (RANSAC).

而且，我们还比较了使用RANSAC估计得到的仿射变换，使用的描述子与我们的方法是一样的(VGG-16 pool4)。这个基准的参数已经进行了广泛的调整，以得到最佳结果，主要是为第二最近邻测试调整了阈值，对可能变换范围之外的建议变换进行剪枝。我们的仿射估计器在这个任务中超过了RANSAC基准，即49% (我们的) 和47% (RANSAC)。

Table 1: Comparison to state-of-the-art and baselines. Match- ing quality on the Proposal Flow dataset measured in terms of PCK. The Proposal Flow methods have four different PCK values, one for each of the four employed region proposal methods. All the numbers apart from ours and RANSAC are taken from [23].

Methods | PCK(%)
--- | ---
DeepFlow [43] | 20
GMK [15] | 27
SIFT Flow [37] | 38
DSP [31] | 29
Proposal Flow NAM [23] | 53
Proposal Flow PHM [23] | 55
Proposal Flow LOM [23] | 56
RANSAC with our features (affine) | 47
Ours (affine) | 49
Ours (affine + thin-plate spline) | 56
Ours (affine ensemble + thin-plate spline) | 57

### 5.5. Discussions and ablation studies

In this section we examine the importance of various components of our architecture. Apart from training on the StreetView-synth-aff dataset, we also train on Pascal-synth-aff which contains images that are more similar in nature to the images in the Proposal Flow benchmark. The results of these ablation studies are summarized in Tab. 2.

本节中，我们对架构的各种组件进行重要性检查。除了在StreetView-synth-aff数据集上进行训练，我们还在Pascal-synth-aff上进行了训练，这个数据集所含的图像与Proposal Flow基准测试中很相似。这些分离试验的结果如表2所示。

**Correlation versus concatenation and subtraction**. Replacing our correlation-based matching layer with feature concatenation or subtraction, as proposed in [14] and [30], respectively, incurs a large performance drop. The behavior is expected as we designed the matching layer to only keep information on pairwise descriptor similarities rather than the descriptors themselves, as is good practice in classical geometry estimation methods, while concatenation and subtraction do not follow this principle.

**相关vs拼接与相减**。将我们基于相关的匹配层替换成特征拼接，或相减，分别就像[14,30]中那样，会带来性能的明显下降。这种表现是符合预期的，因为我们设计匹配层，就只保有了成对描述子的相似性，而不是描述子本身，这在经典估计方法中表现很好，而拼接和相减则没有这样的表现。

**Generalization**. As seen in Tab. 2, our method is relatively unaffected by the choice of training data as its performance is similar regardless whether it was trained with StreetView or Pascal images. We also attribute this to the design choice of operating on pairwise descriptor similarities rather than the raw descriptors.

**泛化**。如表2所示，我们的方法与训练数据的选择相对无关，不论是在StreetView上训练，还是在Pascal上训练，其性能是类似的。我们认为，这是由于我们的设计，就是在成对描述子相似性上运算，而不是在原始描述子上运算。

**Normalization**. Tab. 2 also shows the importance of the correlation map normalization step, where the normalization improves results from 44% to 49%. The step mimics the second nearest neighbor test used in classical feature matching [40], as discussed in Sec. 3.2. Note that [18] also uses a correlation layer, but they do not normalize the map in any way, which is clearly suboptimal.

**归一化**。表2还展示了相关图归一化步骤的重要性，归一化将性能从44%提高到了49%。这一步骤模仿的是经典特征匹配[40]中的第二最近邻域测试，这在3.2节中进行了讨论。注意[18]也使用了相关层，但并没有以任何形式对图进行归一化，只得到了次优的结果。

**What is being learned?** We examine filters from the first convolutional layer of the regressor, which operate directly on the output of the matching layer, i.e. the tentative correspondence map. Recall that each spatial location in the correspondence map (see Fig. 3, in green) contains all similarity scores between that feature in image B and all features in image A. Thus, each single 1-D slice through the weights of one convolutional filter at a particular spatial location can be visualized as an image, showing filter’s preferences to features in image B that match to specific locations in image A. For example, if the central slice of a filter contains all zeros apart from a peak at the top-left corner, this filter responds positively to features in image B that match to the top-left of image A. Similarly, if many spatial locations of the filter produce similar visualizations, then this filter is highly sensitive to spatially co-located features in image B that all match to the top-left of image A. For visualization, we pick the peaks from all slices of filter weights and average them together to produce a single image. Several filters shown in Fig. 7 confirm our hypothesis that this layer has learned to mimic local neighborhood consensus as some filters respond strongly to spatially co-located features in image B that match to spatially consistent locations in image A. Furthermore, it can be observed that the size of the preferred spatial neighborhood varies across filters, thus showing that the filters are discriminative of the scale change.

**学到了什么？**。我们对回归器的第一卷积层的滤波器进行了检查，这是直接在匹配层的输出上进行运算的，即在临时对应性图上进行运算的。想想对应性图中的每个空间位置（图3中的绿色）包含了图像B与图像A中所有特征的相似性分数。因此，每个单个的1-D切片，通过一个卷积滤波器的权重，在一个特定空间位置上，可以可视化为一幅图像，展示滤波器对图像B与图像A中具体位置匹配上的特征的偏好。比如，如果一个滤波器在中间切片，在左上角存在极值，其他位置都是0，那么这个滤波器对于图像B与图像A的左上角可以配准的特征有正面的响应。类似的，如果滤波器的很多空间位置生成了类似的可视化效果，那么这个滤波器就对图像B中对图像A左上角的所有匹配都是高度敏感的。对于可视化，我们选择所有切片的滤波器权重的峰值，将其平均，得到单幅图像。几个滤波器如图7所示，这确认了我们的假设，即这个层学习来模仿局部邻域的一致性，因为一些滤波器在一些位置响应强烈，在这个位置上图像B与图像A的位置匹配上。而且，可以观察到，偏好的空间邻域在不同的滤波器中都不一样，因此滤波器对尺度变化是有区分性的。

### 5.6. Qualitative results

Fig.8 illustrates the effectiveness of our method in category-level matching, where challenging pairs of images from the Proposal Flow dataset [23], containing large intra-class variations, are aligned correctly. The method is able to robustly, in the presence of clutter, estimate large translations, rotations, scale changes, as well as non-rigid transformations and some perspective changes. Further examples are shown in appendix A.

图8展示了我们的方法在类别级匹配中的有效性，这是Proposal Flow数据集中很有挑战的图像对，包含很大的类别内差异，但是很准确的对齐了。这种方法在存在群聚的情况下，可以稳健的估计大的平移、旋转、尺度变化，以及非刚性形变和一些视角变化。更多的例子详见附录A。

Fig.9 shows the quality of instance-level matching, where different images of the same scene are aligned correctly. The images are taken from the Tokyo Time Machine dataset [4] and are captured at different points in time which are months or years apart. Note that, by automatically high-lighting the differences (in the feature space) between the aligned images, it is possible to detect changes in the scene, such as occlusions, changes in vegetation, or structural differences e.g. new buildings being built.

图9所示的是实例级匹配的质量，同样场景的不同图像也得到了正确的对齐。这些图像是Tokyo Time Machine数据集中的，这是在不同的时间点拍摄的，时间差异达到数月或数年。注意，通过自动高亮对齐图像间的差异（在特征空间），是可能检测到场景之间的变化的，比如遮挡，植被的变化，或结构性差异，如新建了一些建筑。

## 6. Conclusions

We have described a network architecture for geometric matching fully trainable from synthetic imagery without the need for manual annotations. Thanks to our matching layer, the network generalizes well to never seen before imagery, reaching state-of-the-art results on the challenging Proposal Flow dataset for category-level matching. This opens-up the possibility of applying our architecture to other difficult correspondence problems such as matching across large changes in illumination (day/night) [4] or depiction style [5].

我们描述了一种进行几何匹配的网络架构，从合成的图像进行训练，不需要手工标注。由于我们的匹配层，网络在之前从未见到的图像中可以很好的泛化，在何由挑战的Proposal Flow数据集中，在类别级的配准达到了目前最好的效果。这开启了新的可能性，可以将我们的架构应用到其他更难的对应性问题中，如光照变化很大的匹配中，或表现形式差别很大的情况中。