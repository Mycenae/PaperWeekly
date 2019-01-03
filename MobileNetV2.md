MobileNetV2: Inverted Residuals and Linear Bottlenecks 逆残差和线性瓶颈

Mark Sandler et al. Google Inc.

Abstract 摘要

In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call MobileDeepLabv3.

本文我们描述了一种新的移动框架，MobileNetV2，改进了在多种任务和测试基准上的移动模型的最好表现，也包括多种不同的模型规模。我们还描述了将这种移动模型高效率的应用在目标检测的新框架中，我们称这种框架为SSDLite。另外，我们展示了如果基于DeepLabv3的简化形式（我们称之为MobileDeepLabv3）来构建移动语义分割模型。

MobileNetV2 is based on an inverted residual structure where the shortcut connections are between the thin bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design.

MobileNetV2是基于逆残差结构的，其中捷径连接是在那些细细的瓶颈之间的。中间扩展层使用轻量的depthwise卷积来对特征滤波，作为一种非线性的来源。另外，我们发现在窄层中去除非线性是很重要的，可以保持表示的强度。我们证明了这可以改进性，并给出了产生这种设计的直觉想法。

Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on ImageNet [1] classification, COCO object detection [2], VOC image segmentation [3]. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as actual latency, and the number of parameters.

最后，我们的方法可以使输入/输出之间的耦合取消，这为以后的分析提供了方便的框架。我们在ImageNet[1]分类任务、COCO目标识别[2]、VOC图像分割任务[3]上测试了模型的表现。我们评估了准确度、计算量（以乘法加法运算数量MAdd计），以及实际延迟，和参数数量。

## 1. Introduction 引言

Neural networks have revolutionized many areas of machine intelligence, enabling superhuman accuracy for challenging image recognition tasks. However, the drive to improve accuracy often comes at a cost: modern state of the art networks require high computational resources beyond the capabilities of many mobile and embedded applications.

神经网络已经使很多机器智能很多领域得到革命，使图像识别任务中出现了超过人类的准确率。但是，提升效率通常有一个代价：现代最好的网络需要很高的计算资源，这超出了很多移动和嵌入式应用的计算能力。

This paper introduces a new neural network architecture that is specifically tailored for mobile and resource constrained environments. Our network pushes the state of the art for mobile tailored computer vision models, by significantly decreasing the number of operations and memory needed while retaining the same accuracy.

本文提出了一种新的神经网络架构，这是为移动环境和资源有限的环境所定制的架构。我们的网络将最优秀的移动定制计算机视觉模型向前进行了推进，显著降低了需要的运算数量和存储空间，同时保持了同样的准确度。

Our main contribution is a novel layer module: the inverted residual with linear bottleneck. This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution. The official implementation is available as part of TensorFlow-Slim model library in [4].

我们的主要贡献是一种新的层的模块：带有线性瓶颈的逆残差结构。这个模块输入为低维度压缩表示，首先延展为高维，然后用轻量的depthwise卷积滤波。随后特征用线性卷积投影回低维表示。官方实现在TensorFlow Slim模型库的一部分已经可用[4]。

This module can be efficiently implemented using standard operations in any modern framework and allows our models to beat state of the art along multiple performance points using standard benchmarks. Furthermore, this convolutional module is particularly suitable for mobile designs, because it allows to significantly reduce the memory footprint needed during inference by never fully materializing large intermediate tensors. This reduces the need for main memory access in many embedded hardware designs, that provide small amounts of very fast software controlled cache memory.

这种模块在任何现代框架中采用标准运算都可以高效率的实现，使我们的模型在标准测试中击败了目前最好的模型。而且，这种卷积模块尤其适合于移动设计，因为显著降低了推理时的内存需求，这是因为从未将大型中间张量完全实例化。这在很多嵌入式硬件设计上降低了主要的内存需求，带来了非常快速的软件实现，使用少量可控的缓存内存。

## 2. Related Work 相关工作

Tuning deep neural architectures to strike an optimal balance between accuracy and performance has been an area of active research for the last several years. Both manual architecture search and improvements in training algorithms, carried out by numerous teams has lead to dramatic improvements over early designs such as AlexNet [5], VGGNet [6], GoogLeNet [7], and ResNet [8]. Recently there has been lots of progress in algorithmic architecture exploration included hyperparameter optimization [9, 10, 11] as well as various methods of network pruning [12, 13, 14, 15, 16, 17] and connectivity learning [18, 19]. A substantial amount of work has also been dedicated to changing the connectivity structure of the internal convolutional blocks such as in ShuffleNet [20] or introducing sparsity [21] and others [22].

过去的几年里，调节深度神经网络架构，以达到准确度与性能的最佳平衡，一直是一个活跃的研究领域。手工搜索架构，改进训练算法，众多数量的团队的实验，带来了网络设计的迅速进步，如AlexNet[5], VGGNet[6], GoogLeNet[7]和ResNet[8]。最近有很多用算法探索架构的进展，也包括超参数优化[9,10,11]和各种方法的网络修剪[12,13,14,15,16,17]和连接性学习[18,19]。也有很多研究关注内部卷积模块的连接性结构改变如ShuffleNet[20]，或引入稀疏性[21]，或其他改进。

Recently, [23, 24, 25, 26], opened up a new direction of bringing optimization methods including genetic algorithms and reinforcement learning to architectural search. However one drawback is that the resulting networks end up very complex. In this paper, we pursue the goal of developing better intuition about how neural networks operate and use that to guide the simplest possible network design. Our approach should be seen as complimentary to the one described in [23] and related work. In this vein our approach is similar to those taken by [20, 22] and allows to further improve the performance, while providing a glimpse on its internal operation. Our network design is based on MobileNetV1 [27]. It retains its simplicity and does not require any special operators while significantly improves its accuracy, achieving state of the art on multiple image classification and detection tasks for mobile applications.

最近，[23,24,25,26]开启了新的研究方向，将遗传算法和强化学习这样的优化算法引入到结构搜索中。但是，一个缺陷是，得到的网络都非常复杂。本文中，我们追求关于神经网络怎样工作的更好的直觉原理，用于指导我们得到最简单的网络设计。我们的方法应当被视为[23]及相关工作中方法的补充。在这方面，我们的方法与[20,22]中的方法类似，允许进一步改进性能，同时一窥其内部运算。我们的网络设计是基于MobileNetV1[27]的，其保持了简洁性，不需要任何特殊的运算，同时明显改进了准确度，在多个移动应用的分类和检测任务中得到了目前最好的结果。

## 3. Preliminaries, discussion and intuition 预备知识，讨论和直觉

### 3.1. Depthwise Separable Convolutions

Depthwise Separable Convolutions are a key building block for many efficient neural network architectures [27, 28, 20] and we use them in the present work as well. The basic idea is to replace a full convolutional operator with a factorized version that splits convolution into two separate layers. The first layer is called a depthwise convolution, it performs lightweight filtering by applying a single convolutional filter per input channel. The second layer is a 1 × 1 convolution, called a pointwise convolution, which is responsible for building new features through computing linear combinations of the input channels.

Depthwise Separable卷积是很多高效神经网络架构的关键模块[27,28,30]，我们也在这个工作中使用了这个模块。其基本思想是将完整的卷积操作替换为一个分解的版本，将卷积分解为两个单独的层。第一层称为depthwise卷积，对每个输入通道都应用一个卷积滤波器，这是一种轻量的滤波操作。第二层是1×1卷积，称为pointwise卷积，通过计算输入通道的线性组合来负责构建新的特征。

Standard convolution takes an $h_i × w_i × d_i$ input tensor $L_i$, and applies convolutional kernel $K ∈ R^{k×k×d_i ×d_j}$ to produce an $h_i × w_i × d_j$ output tensor $L_j$. Standard convolutional layers have the computational cost of $h_i · w_i · d_i · d_j · k · k$.

标准卷积有输入张量$L_i$，维度$h_i × w_i × d_i$，卷积核$K ∈ R^{k×k×d_i ×d_j}$，得到的输出张量为$L_j$，维度$h_i × w_i × d_j$。标准卷积层的计算量为$h_i · w_i · d_i · d_j · k · k$。

Depthwise separable convolutions are a drop-in replacement for standard convolutional layers. Empirically they work almost as well as regular convolutions but only cost:

Depthwise separable卷积是标准卷积层的替代品，经验上来说，其作用与常规卷积一样好，但是计算量只有：

$$h_i · w_i · d_i (k^2 + d_j )$$(1)

which is the sum of the depthwise and 1 × 1 pointwise convolutions. Effectively depthwise separable convolution reduces computation compared to traditional layers by almost a factor of $k^2$(more precisely, by a factor $k^2 d_j /(k^2 + d_j)$). MobileNetV2 uses k = 3 (3 × 3 depthwise separable convolutions) so the computational cost is 8 to 9 times smaller than that of standard convolutions at only a small reduction in accuracy [27].

这是depthwise和1 × 1 pointwise的计算量之和。Depthwise separable卷积有效的降低了计算量，与传统卷积层相比，计算量下降了$k^2$倍（精确来说，是$k^2 d_j /(k^2 + d_j)$）。MobileNetV2中k=3（3×3 depthwise separable卷积），所以计算量下降了8至9倍，准确率只是略有降低[27]。

### 3.2. Linear Bottlenecks 线性瓶颈

Consider a deep neural network consisting of n layers $L_i$ each of which has an activation tensor of dimensions $h_i × w_i × d_i$. Throughout this section we will be discussing the basic properties of these activation tensors, which we will treat as containers of $h_i × w_i$ “pixels” with $d_i$ dimensions. Informally, for an input set of real images, we say that the set of layer activations (for any layer $L_i$) forms a “manifold of interest”. It has been long assumed that manifolds of interest in neural networks could be embedded in low-dimensional subspaces. In other words, when we look at all individual d-channel pixels of a deep convolutional layer, the information encoded in those values actually lie in some manifold, which in turn is embeddable into a low-dimensional subspace(Note that dimensionality of the manifold differs from the dimensionality of a subspace that could be embedded via a linear transformation).

考虑一个深度神经网络，包括n层$L_i$，每层都有一个激活张量，维度为$h_i × w_i × d_i$。贯穿本节，我们将会讨论这些激活张量的基本性质，我们将这些张量看作是$h_i × w_i$像素的容器，有$d_i$维。非正式的，对于输入真实图像集，我们称（对于任意层$L_i$）层激活为一个“兴趣流形”。一直以来，都假设神经网络中的兴趣流形可以嵌入低维子空间。换句话说，当我们观察一个深度卷积层的所有d-通道像素个体，这些值中所编码的信息实际上在某个流形之中，这是嵌入在一个低维子空间中的（注意，流形的维数与子空间的维数是不一样的，向这个子空间的嵌入是通过一个线性变换完成的）。

At a first glance, such a fact could then be captured and exploited by simply reducing the dimensionality of a layer thus reducing the dimensionality of the operating space. This has been successfully exploited by MobileNetV1 [27] to effectively trade off between computation and accuracy via a width multiplier parameter, and has been incorporated into efficient model designs of other networks as well [20]. Following that intuition, the width multiplier approach allows one to reduce the dimensionality of the activation space until the manifold of interest spans this entire space. However, this intuition breaks down when we recall that deep convolutional neural networks actually have non-linear per coordinate transformations, such as ReLU. For example, ReLU applied to a line in 1D space produces a ’ray’, where as in $R^n$ space, it generally results in a piece-wise linear curve with n-joints.

第一眼看上去，可以利用这样一个事实，对一层进行降维，以降低运算空间的维度。这已经在MobileNetV1[27]中成功应用了，通过一个width multiplier参数，可以有效的在计算量和准确性之间均衡，这也其他网络模型设计所采用[20]。跟随这个直觉，这个width multiplier方法使激活空间维度降低，直到兴趣流形撑起了整个空间。但是，我们记得，深度卷积神经网络在每个坐标变换时都有非线性成分，比如ReLU，所以这个直觉不成立。比如，在1D空间中，对一条线使用ReLU函数，产生的是一条曲线，在$R^n$空间中，一般来说会得到一个分段线性的曲线，其中有n个转接点。

It is easy to see that in general if a result of a layer transformation ReLU(Bx) has a non-zero volume S, the points mapped to interior S are obtained via a linear transformation B of the input, thus indicating that the part of the input space corresponding to the full dimensional output, is limited to a linear transformation. In other words, deep networks only have the power of a linear classifier on the non-zero volume part of the output domain. We refer to supplemental material for a more formal statement.

很容易看出来，一般来说，如果一个层的变换ReLU(Bx)的结果容量S非零，映射入S内部的点是通过输入的线性变换B得到的，所以这说明对应着输出的全维度那部分输入空间，是受到一个线性变换限制的。换句话说，深度网络对于输出领域的非零容量部分只有一个线性分类器的作用。更正式的阐述请参考补充材料。

On the other hand, when ReLU collapses the channel, it inevitably loses information in that channel. However if we have lots of channels, and there is a structure in the activation manifold that information might still be preserved in the other channels. In supplemental materials, we show that if the input manifold can be embedded into a significantly lower-dimensional subspace of the activation space then the ReLU transformation preserves the information while introducing the needed complexity into the set of expressible functions.

另一方面，当ReLU瓦解了这个通道时，这个通道的信息不可逆转的有损失。但是，如果我们有很多通道，激活流形中是有结构的，信息可能仍然保存在其他通道中。在补充材料中，我们证明了，如果输入流形可以嵌入到激活空间足够低维的子空间中，那么ReLU变换会保留这些信息，同时引入需要的复杂性到可表达的函数集中。

Figure 1: Examples of ReLU transformations of low-dimensional manifolds embedded in higher-dimensional spaces. In these examples the initial spiral is embedded into an n-dimensional space using random matrix T followed by ReLU, and then projected back to the 2D space using $T^{−1}$. In examples above n = 2, 3 result in information loss where certain points of the manifold collapse into each other, while for n = 15 to 30 the transformation is highly non-convex.

图1：低维流形经过ReLU变换嵌入高维空间的例子。在这些例子中，初始的螺旋通过随机矩阵T和ReLU变换，嵌入到n维空间，然后用$T^{−1}$映射回2D空间。当n=2,3时得到信息有损失，其中流形中的一些点坍缩成一点，而n=15或30的时候变换是高度非凸的。

To summarize, we have highlighted two properties that are indicative of the requirement that the manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space:

总结一下，我们强调了两个性质，表示了下面的需求，兴趣流形应当在较高维的激活空间中的低维子空间中：

- If the manifold of interest remains non-zero volume after ReLU transformation, it corresponds to a linear transformation. 如果兴趣流形在ReLU变换后，仍然是非零容量，那么就对应着一个线性变换。
- ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensional subspace of the input space. ReLU可以保存输入流形的完整信息，但这只在输入流形是输入空间的低维子空间时才成立。

These two insights provide us with an empirical hint for optimizing existing neural architectures: assuming the manifold of interest is low-dimensional we can capture this by inserting linear bottleneck layers into the convolutional blocks. Experimental evidence suggests that using linear layers is crucial as it prevents non-linearities from destroying too much information. In Section 6, we show empirically that using non-linear layers in bottlenecks indeed hurts the performance by several percent, further validating our hypothesis(We note that in the presence of shortcuts the information loss is actually less strong). We note that similar reports where non-linearity was helped were reported in [29] where non-linearity was removed from the input of the traditional residual block and that lead to improved performance on CIFAR dataset.

这两点洞见为我们优化现有的神经网络架构提供了经验性的提示：假设兴趣流形是低维的，我们可以通过在卷积模块中插入线性瓶颈层来验证。实验结果表明，使用线性层是关键的，因为这防止了非线性变换摧毁太多信息。在第6节中，我们经验性的证明，在瓶颈中使用非线性层确实使性能下降了几个百分点，进一步验证了我们的假设（我们注意到，在捷径连接存在的情况下，信息损失没有那么多）。我们注意到在[29]中也报告了类似的结论，其中从传统残差模块的输入中去掉了非线性成分，这在CIFAR数据集中改进了性能。

For the remainder of this paper we will be utilizing bottleneck convolutions. We will refer to the ratio between the size of the input bottleneck and the inner size as the expansion ratio. 本文剩下的篇幅中，我们会使用瓶颈卷积。我们称输入瓶颈的大小与内部的大小的比值为扩展率。

Figure 2: Evolution of separable convolution blocks. The diagonally hatched texture indicates layers that do not contain non-linearities. The last (lightly colored) layer indicates the beginning of the next block. Note: 2d and 2c are equivalent blocks when stacked. Best viewed in color. (a)Regular, (b)Separable, (c)Separable with linear bottleneck, (d)Bottleneck with expansion layer.

### 3.3. Inverted residuals 逆残差

The bottleneck blocks appear similar to residual block where each block contains an input followed by several bottlenecks then followed by expansion [8]. However, inspired by the intuition that the bottlenecks actually contain all the necessary information, while an expansion layer acts merely as an implementation detail that accompanies a non-linear transformation of the tensor, we use shortcuts directly between the bottlenecks. Figure 3 provides a schematic visualization of the difference in the designs. The motivation for inserting shortcuts is similar to that of classical residual connections: we want to improve the ability of a gradient to propagate across multiplier layers. However, the inverted design is considerably more memory efficient (see Section 5 for details), as well as works slightly better in our experiments.

瓶颈模块与残差模块类似，两个模块都包含输入，接着是几个瓶颈，然后是扩展模块[8]。但是，瓶颈实际上包含了所有必要的信息，同时扩展模块只是张量非线性变换伴随的一个实现细节，受此启发，我们直接在瓶颈之间使用捷径。图3给出了设计上的不同之处的示意图。加入捷径的动机与经典残差连接类似：我们希望改善梯度在跨乘积（多？）层中传播的能力。但是，其逆设计在内存使用上更有效率（详见第5节），在我们的实验中，效果还更好一些。

Figure 3: The difference between residual block [8, 30] and inverted residual. Diagonally hatched layers do not use non-linearities. We use thickness of each block to indicate its relative number of channels. Note how classical residuals connects the layers with high number of channels, whereas the inverted residuals connect the bottlenecks. Best viewed in color.

图3 残差模块[8,30]与逆残差模块的区别。有对角阴影线的没有使用非线性变换。我们用每个模块的厚度来表示其通道的相对数量。注意经典残差结构连接的通道数多的层，而逆残差结构连接的是瓶颈层。

**Running time and parameter count for bottleneck convolution**. The basic implementation structure is illustrated in Table 1. For a block of size h × w, expansion factor t and kernel size k with d' input channels and d'' output channels, the total number of multiply add required is $h · w · d' · t(d' + k^2 + d'')$. Compared with (1) this expression has an extra term, as indeed we have an extra 1 × 1 convolution, however the nature of our networks allows us to utilize much smaller input and output dimensions. In Table 3 we compare the needed sizes for each resolution between MobileNetV1, MobileNetV2 and ShuffleNet.

**瓶颈卷积的运行时和参数数量**。基本的实现结构如表1所示。对于大小为h×w的块，扩展率为t，滤波核大小为k，输入通道数为d'，输出通道数为d''，乘法加法运算的总共数量是$h · w · d' · t(d' + k^2 + d'')$。与(1)比较，这个式子有个额外的项，确实我们有额外的1×1卷积，但是我们网络的本质使我们可以利用小的多的输入和输出维度。在表3中，我们比较了MobileNetV1, MobileNetV2和ShuffleNet每种分辨率所需的大小。

Table 1: Bottleneck residual block transforming from k to k' channels, with stride s, and expansion factor t. 瓶颈残差模块，将k通道输入变换为k'通道的输出，步长s，扩展率t。

Input | Operator | Output
--- | --- | ---
h × w × k | 1x1 conv2d , ReLU6 | h × w × (tk)
h × w × tk | 3x3 dwise s=s, ReLU6 | h/s × w/s × (tk)
h/s × w/s × tk | linear 1x1 conv2d | h/s × w/s × k'

Table 3: The max number of channels/memory (in Kb) that needs to be materialized at each spatial resolution for different architectures. We assume 16-bit floats for activations. For ShuffleNet, we use 2x, g = 3 that matches the performance of MobileNetV1 and MobileNetV2. For the first layer of MobileNetV2 and ShuffleNet we can employ the trick described in Section 5 to reduce memory requirement. Even though ShuffleNet employs bottlenecks elsewhere, the non-bottleneck tensors still need to be materialized due to the presence of shortcuts between non-bottleneck tensors.

表3：不同模型在不同分辨率下，需要实现的通道/内存(Kb)最大数量。我们假设对于激活来说是16-bit的浮点数。对于ShuffleNet，我们使用2x, g=3，这种配置下性能与MobileNetV1和MobileNetV2最匹配。对于MobileNetV2的第一层和ShuffleNet，我们可以使用第5节中所说的技巧来减少内存需求。即使ShuffleNet在其他的地方也使用了瓶颈结构，非瓶颈的张量仍然需要实例化，因为非瓶颈张量之间也有捷径连接。

Size | MobileNetV1 | MobileNetV2 | ShuffleNet(2x,g=3)
--- | --- | --- | ---
112x112 | 1/O(1) | 1/O(1) | 1/O(1)
56x56 | 128/800 | 32/200 | 48/300
28x28 | 256/400 | 64/100 | 400/600K
14x14 | 512/200 | 160/62 | 800/310
7x7 | 1024/199 | 320/32 | 1600/156
1x1 | 1024/2 | 1280/2 | 1600/3
max | 800K | 200K | 600K

### 3.4. Information flow interpretation 信息流解释

One interesting property of our architecture is that it provides a natural separation between the input/output domains of the building blocks (bottleneck layers), and the layer transformation – that is a nonlinear function that converts input to the output. The former can be seen as the capacity of the network at each layer, whereas the latter as the expressiveness. This is in contrast with traditional convolutional blocks, both regular and separable, where both expressiveness and capacity are tangled together and are functions of the output layer depth.

我们架构的一个有趣的性质是，它给出了输入/输出域的自然分离的构建模块（瓶颈层），以及层的变换，即将输入变换到输出的非线性函数。前者可以看作是网络对于彼此的容量，而后者看作是表示能力。这是与传统卷积模块的对比，包括常规卷积和separable卷积，其中表示能力和容量是缠在一起的，是输出层的深度的函数。

In particular, in our case, when inner layer depth is 0 the underlying convolution is the identity function thanks to the shortcut connection. When the expansion ratio is smaller than 1, this is a classical residual convolutional block [8, 30]. However, for our purposes we show that expansion ratio greater than 1 is the most useful.

特别的，在我们的情况中，当内部层的深度为0时，由于捷径连接的存在，所以潜在的卷积是恒等函数。当扩展率小于1时，这是经典的残差卷积模块[8,30].但是，在我们的情况中，我们证实，扩展率大于1是最有用的。

This interpretation allows us to study the expressiveness of the network separately from its capacity and we believe that further exploration of this separation is warranted to provide a better understanding of the network properties.

这种解释使我们可以撇开网络容量研究其表示能力，我们相信这种分离的进一步研究肯定 可以更好的理解网络的性质。

## 4. Model Architecture 模型架构

Now we describe our architecture in detail. As discussed in the previous section the basic building block is a bottleneck depth-separable convolution with residuals. The detailed structure of this block is shown in Table 1. The architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers described in the Table 2. We use ReLU6 as the non-linearity because of its robustness when used with low-precision computation [27]. We always use kernel size 3 × 3 as is standard for modern networks, and utilize dropout and batch normalization during training.

现在我们给出框架的细节。如前节所述，基本构成模块是带残差的瓶颈depth-separable卷积。这个模块的结构细节如表1所示。MobileNetV2包含初始的全卷积层，有32个滤波器，随后是19个残差瓶颈层，如表2所示。我们使用ReLU6作为非线性变换，因为在使用低精度计算的时候稳健性高[27]。我们一直使用3×3大小的卷积核，这是现代网络的标配，在训练的时候使用了dropout和批归一化。

Table 2: MobileNetV2 : Each line describes a sequence of 1 or more identical (modulo stride) layers, repeated n times. All layers in the same sequence have the same number c of output channels. The first layer of each sequence has a stride s and all others use stride 1. All spatial convolutions use 3 × 3 kernels. The expansion factor t is always applied to the input size as described in Table 1.

表2：MobileNetV2：每一行都是相同层的序列，重复n次。同一序列中的所有层输出通道数都相同。每个序列中的第一层的步长为s，剩下的使用步长1。所有空间卷积使用3×3核。如表1所示，对输入大小应用的扩展率为t。

Input | Operator | t | c | n | s
--- | --- | --- | --- | --- | ---
$224^2×3$ | conv2d | - | 32 | 1 | 2
$112^2×32$ | bottleneck | 1 | 16 | 1 | 1
$112^2×16$ | bottleneck | 6 | 24 | 2 | 2
$56^2×24$ | bottleneck | 6 | 32 | 3 | 2
$28^2×32$ | bottleneck | 6 | 64 | 4 | 2
$14^2×64$ | bottleneck | 6 | 96 | 3 | 1
$14^2×96$ | bottleneck | 6 | 160 | 3 | 2
$7^2×160$ | bottleneck | 6 | 320 | 1 | 1
$7^2×320$ | conv2d 1×1 | - | 1280 | 1 | 1
$7^2×1280$ | avgpool 7×7 | - | - | 1 | -
$1^2×1280$ | conv2d 1×1 | - | k | - | -

With the exception of the first layer, we use constant expansion rate throughout the network. In our experiments we find that expansion rates between 5 and 10 result in nearly identical performance curves, with smaller networks being better off with slightly smaller expansion rates and larger networks having slightly better performance with larger expansion rates.

除了第一层，我们在整个网络中使用常数扩展率。在我们的实验中，我们发现扩展率在5-10之间的结果是几乎一样的，扩展率较小的时候，网络更小一些，而扩展率大的时候，网络更大，性能也略微好一点。

For all our main experiments we use expansion factor of 6 applied to the size of the input tensor. For example, for a bottleneck layer that takes 64-channel input tensor and produces a tensor with 128 channels, the intermediate expansion layer is then 64 · 6 = 384 channels.

我们所有的主要实验中，都使用扩展率为6。比如，对于一个瓶颈层，输入张量为64通道，结果为128通道，那么中间的扩展层为64 · 6 = 384层。

**Trade-off hyper parameters**. As in [27] we tailor our architecture to different performance points, by using the input image resolution and width multiplier as tunable hyper parameters, that can be adjusted depending on desired accuracy/performance trade-offs. Our primary network (width multiplier 1, 224 × 224), has a computational cost of 300 million multiply-adds and uses 3.4 million parameters. We explore the performance trade offs, for input resolutions from 96 to 224, and width multipliers of 0.35 to 1.4. The network computational cost ranges from 7 multiply adds to 585M MAdds, while the model size vary between 1.7M and 6.9M parameters.

**超参数的折衷**。和[27]一样，我们为不同的性能定制我们网络的架构，我们使用输入图像分辨率和width multiplier作为可调节的超参数，这主要是根据期望的准确率/性能的折衷来调节。我们的基本网络(width multiplier 1, 224 × 224)，计算量为3亿乘法加法运算，参数数量为340万。我们研究了性能的折衷，对于不同的输入分辨率，从96到224，和不同的width multiplier，从0.35到1.4。对应的网络计算量为7百万乘法加法运算到5.85亿次乘法加法运算，模型规模变化范围为170万到690万参数。

One minor implementation difference, with [27] is that for multipliers less than one, we apply width multiplier to all layers except the very last convolutional layer. This improves performance for smaller models.

有一个很小的实现上的差异，在[27]中，对于小于1的乘子，我们对所有层都使用width multiplier，除了最后一个卷积层。这改进了较小的模型的性能。

Figure 4: Comparison of convolutional blocks for different architectures. ShuffleNet uses Group Convolutions [20] and shuffling, it also uses conventional residual approach where inner blocks are narrower than output. ShuffleNet and NasNet illustrations are from respective papers.

图4：不同架构中卷积模块的比较。ShuffleNet使用Group卷积[20]和shuffling，也使用传统残差方法，其中内部模块比输出更窄。ShuffleNet和NasNet的描述都是从各自的论文中摘出来的。

## 5. Implementation Notes 实现笔记

### 5.1 Memory efficient inference 内存使用率高的推理

The inverted residual bottleneck layers allow a particularly memory efficient implementation which is very important for mobile applications. A standard efficient implementation of inference that uses for instance TensorFlow[31] or Caffe [32], builds a directed acyclic compute hypergraph G, consisting of edges representing the operations and nodes representing tensors of intermediate computation. The computation is scheduled in order to minimize the total number of tensors that needs to be stored in memory. In the most general case, it searches over all plausible computation orders Σ(G) and picks the one that minimizes

逆残差瓶颈层的实现内存使用效率非常高，这对于移动应用来说非常重要。使用TensorFlow[31]或Caffe[32]实现标准推理过程时，会构建一个有向无环的计算图G，边代表运算，节点代表中间计算的张量结果。计算中要最小化需要存储在内存中的张量数目。在最一般的情况下，要搜索所有可行的计算Σ(G)，选择可以最小化下式的那个：

$$M(G) = min_{π ∈ Σ(G)} max_{i∈1...n} [\sum_{A∈R(i,π,G)} |A|] + size(π_i)$$

where R(i, π, G) is the list of intermediate tensors that are connected to any of $π_i . . . π_n$ nodes, |A| represents the size of the tensor A and size(i) is the total amount of memory needed for internal storage during operation i.

其中R(i, π, G)是中间张量列表，这些张量与任一$π_i . . . π_n$节点有连接，|A|表示张量A的大小，size(i)是运算i的过程中总计需要的内部存储。

For graphs that have only trivial parallel structure (such as residual connection), there is only one non-trivial feasible computation order, and thus the total amount and a bound on the memory needed for inference on compute graph G can be simplified:

对于只有简单并行结构（如残差连接）的图来说，只有一个有意义的可行计算，所以计算图G推理总计需要的内存以及其上限可以简化为：

$$M(G) = max_{op∈G} [\sum_{A∈op_{in}} |A| + \sum_{B∈op_{out}} |B| + |op|]$$(2)

Or to restate, the amount of memory is simply the maximum total size of combined inputs and outputs across all operations. In what follows we show that if we treat a bottleneck residual block as a single operation (and treat inner convolution as a disposable tensor), the total amount of memory would be dominated by the size of bottleneck tensors, rather than the size of tensors that are internal to bottleneck (and much larger).

即，内存总需求为所有操作的输入和输出的大小的总和。下面我们证明，如果我们将瓶颈残差模块看作是一个单独运算（将其内部卷积视为一次性使用的张量），那么总计需要的内存中大部分都是瓶颈的张量，而不是瓶颈内部的张量。

**Bottleneck Residual Block**. A bottleneck block operator F(x) shown in Figure 3b can be expressed as a composition of three operators $F(x) = [A ◦ N ◦ B]x$, where A is a linear transformation $A : R^{s×s×k} → R^{s×s×n}$, N is a non-linear per-channel transformation: $N : R^{s×s×n}→ R^{s'×s'×n}$, and B is again a linear transformation to the output domain: $B : R^{s'×s'×n} → R^{s'×s'×k'}$.

**瓶颈残差模块**。图3b中的瓶颈模块运算符F(x)可以表示为三个运算符的组合$F(x) = [A ◦ N ◦ B]x$，其中A是一个线性变换$A : R^{s×s×k} → R^{s×s×n}$，N是一个非线性的逐通道变换：$N : R^{s×s×n}→ R^{s'×s'×n}$，B也是一个线性变换，变换到输出域：$B : R^{s'×s'×n} → R^{s'×s'×k'}$。

For our networks N = ReLU6 ◦ dwise ◦ ReLU6, but the results apply to any per-channel transformation. Suppose the size of the input domain is |x| and the size of the output domain is |y|, then the memory required to compute F(X) can be as low as $|s^2 k| + |s'^2 k'| + O(max(s^2 , s'^2))$.

对我们的网络来说，N = ReLU6 ◦ dwise ◦ ReLU6，但结果可以应用于任一逐通道变换。假设输入域的大小为|x|，输出域的大小为|y|，那么计算F(x)所需的内存可以低至$|s^2 k| + |s'^2 k'| + O(max(s^2 , s'^2))$。

The algorithm is based on the fact that the inner tensor I can be represented as concatenation of t tensors, of size n/t each and our function can then be represented as

算法是基于这样的事实，内部张量I可以表示为t个张量的拼接，每个的大小是n/t，那么我们的函数就可以表示为：

$$F(x) = \sum_{i=1}^t (A_i ◦ N ◦ B_i)(x)$$

by accumulating the sum, we only require one intermediate block of size n/t to be kept in memory at all times. Using n = t we end up having to keep only a single channel of the intermediate representation at all times. The two constraints that enabled us to use this trick is (a) the fact that the inner transformation (which includes non-linearity and depthwise) is per-channel, and (b) the consecutive non-per-channel operators have significant ratio of the input size to the output. For most of the traditional neural networks, such trick would not produce a significant improvement.

通过这种累积和，我们只需要维持一个大小为n/t的中间层始终在内存中就可以了。n=t时，我们只需要保持内部表示的一个通道始终在内存中。我们能使用这种技巧是因为以下两个限制：(a)内部变换（包括非线性变换和depthwise变换）是逐通道的，(b)连续的non-per-channel运算符有明显的输入大小对输出大小的比率。对于多数传统神经网络，这种技巧不会得到明显的改善。

We note that, the number of multiply-adds operators needed to compute F (X) using t-way split is independent of t, however in existing implementations we find that replacing one matrix multiplication with several smaller ones hurts runtime performance due to increased cache misses. We find that this approach is the most helpful to be used with t being a small constant between 2 and 5. It significantly reduces the memory requirement, but still allows one to utilize most of the efficiencies gained by using highly optimized matrix multiplication and convolution operators provided by deep learning frameworks. It remains to be seen if special framework level optimization may lead to further runtime improvements.

我们注意到，使用t路分割来计算F(x)所需的乘法加法运算的数量与t是无关的，但是在现有的实现中，我们发现将矩阵相乘替换为几个更小的矩阵相乘会使性能下降，这是因为缓存不命中的增加造成的。我们发现，当t为2-5之间小的常数时，性能最好。可以明显降低内存需求，同时也可以充分利用深度学习框架中高度优化的矩阵相乘和卷积算子带来的效率。如果有特殊的框架级的优化，还可以进一步带来运行时的改进，这是有可能的。

## 6. Experiments 试验

### 6.1. ImageNet Classification 图像分类

**Training setup**. We train our models using TensorFlow[31]. We use the standard RMSPropOptimizer with both decay and momentum set to 0.9. We use batch normalization after every layer, and the standard weight decay is set to 0.00004. Following MobileNetV1[27] setup we use initial learning rate of 0.045, and learning rate decay rate of 0.98 per epoch. We use 16 GPU asynchronous workers, and a batch size of 96.

**训练设置**。我们使用TensorFlow[31]训练模型。我们使用标准的RMSPropOptimizer，decay和momentum都设为0.9。我们每层之后都使用批归一化，标准权重衰减设置为0.00004。遵循MobileNetV1[27]的设置，我们使用初始学习率为0.045，每轮训练学习率衰减0.98。我们使用16个GPU训练，批大小为96。

**Results**. We compare our networks against MobileNetV1, ShuffleNet and NASNet-A models. The statistics of a few selected models is shown in Table 4 with the full performance graph shown in Figure 5.

**结果**。我们将本文的网络与MobileNetV1, ShuffleNet和NASNet-A模型进行了比较。表4给出了选出的一些模型的统计数值，完整的性能图见图5。

Table 4: Performance on ImageNet, comparison for different networks. As is common practice for ops, we count the total number of Multiply-Adds. In the last column we report running time in milliseconds (ms) for a single large core of the Google Pixel 1 phone (using TF-Lite). We do not report ShuffleNet numbers as efficient group convolutions and shuffling are not yet supported.

表4：不同模型在ImageNet上的性能对比。我们使用乘法加法的运算数量来对比计算复杂度。在最后一列，我们给出了在Google Pixel 1 phone上使用TF-Lite在单核上运行的时间，单位浩渺。我们没有给出ShuffleNet的结果，因为group卷积和shuffling还没有得到很好的支持。

Network | Top 1 | Param | MAdds | CPU
--- | --- | --- | --- | ---
MobileNetV1 | 70.6 | 4.2M | 575M | 113ms
ShuffleNet(1.5) | 71.5 | 3.4M | 292M | -
ShuffleNet(x2) | 73.7 | 5.4M | 524M | -
NasNet-A | 74.0 | 5.3M | 564M | 183ms
MobileNetV2 | 72.0 | 3.4M | 300M | 75ms
MobileNetV2(1.4) | 74.7 | 6.9M | 585M | 143ms

Figure 5: Performance curve of MobileNetV2 vs MobileNetV1, ShuffleNet, NAS. For our networks we use multipliers 0.35, 0.5, 0.75, 1.0 for all resolutions, and additional 1.4 for for 224. Best viewed in color.

### 6.2. Object Detection 目标检测

We evaluate and compare the performance of MobileNetV2 and MobileNetV1 as feature extractors [33] for object detection with a modified version of the Single Shot Detector (SSD) [34] on COCO dataset [2]. We also compare to YOLOv2 [35] and original SSD (with VGG-16 [6] as base network) as baselines. We do not compare performance with other architectures such as Faster-RCNN [36] and RFCN [37] since our focus is on mobile/real-time models.

我们将MobileNetV2和MobileNetV1作为特征提取器[33]，并修改了SSD[34]作为目标检测系统，在COCO数据集[2]上评估和比较模型性能。我们还将YOLOv2[35]和原版SSD（使用VGG-16[6]作为基础网络）作为基准进行比较。我们没有和其他架构比较，如Faster-RCNN[36]和RFCN[37]，因为我们主要聚焦在移动/实时模型。

SSDLite: In this paper, we introduce a mobile friendly variant of regular SSD. We replace all the regular convolutions with separable convolutions (depthwise followed by 1 × 1 projection) in SSD prediction layers. This design is in line with the overall design of MobileNets and is seen to be much more computationally efficient. We call this modified version SSDLite. Compared to regular SSD, SSDLite dramatically reduces both parameter count and computational cost as shown in Table 5.

**SSDLite**：在本文中，我们提出了一种常规SSD的变体，对移动应用更加友好。我们将SSD预测层中所有常规卷积替换为separable卷积（depthwise卷积和1×1投影）。这种设计与MobileNets的整体设计是一致的，而且计算上更有效率。我们称这种模型为SSDLite。与常规SSD相比，SSDLite在参数数量和计算代价上都有显著下降，见表5。

For MobileNetV1, we follow the setup in [33]. For MobileNetV2, the first layer of SSDLite is attached to the expansion of layer 15 (with output stride of 16). The second and the rest of SSDLite layers are attached on top of the last layer (with output stride of 32). This setup is consistent with MobileNetV1 as all layers are attached to the feature map of the same output strides.

对于MobileNetV1，我们采用[33]的设置。对MobileNetV2，SSDLite的第一层与15层的扩展相接（输出步长为16）。SSDLite第二层和剩余的层与剩下的层相连（输出步长为32）。这种设置与MobileNetV1一致，因为所有层相接的特征图的输出步长都一样。

Both MobileNet models are trained and evaluated with Open Source TensorFlow Object Detection API [38]. The input resolution of both models is 320 × 320. We benchmark and compare both mAP (COCO challenge metrics), number of parameters and number of Multiply-Adds. The results are shown in Table 6. MobileNetV2 SSDLite is not only the most efficient model, but also the most accurate of the three. Notably, MobileNetV2 SSDLite is 20× more efficient and 10× smaller while still outperforms YOLOv2 on COCO dataset.

两种MobileNet模型都使用开源TensorFlow目标检测API进行训练和评估[38]。两种模型的输入分辨率都是320×320。我们用基准检测并比较mAP（COCO挑战的衡量标准），参数数量和乘法加法运算数量。结果如表6所示。MobileNetV2 SSDLite不仅是三种模型中最有效率的，同时还是最准确的。值得注意的是，MobileNetV2 SSDLite在COCO数据集上比YOLOv2效率高了20倍，同时小了10倍。

Table 5: Comparison of the size and the computational cost between SSD and SSDLite configured with MobileNetV2 and making predictions for 80 classes.

| | Params | MAdds
--- | --- | ---
SSD[34] | 14.8M | 1.25B
SSDLite | 2.1M | 0.35B

Table 6: Performance comparison of MobileNetV2 + SSDLite and other realtime detectors on the COCO dataset object detection task. MobileNetV2 + SSDLite achieves competitive accuracy with significantly fewer parameters and smaller computational complexity. All models are trained on trainval35k and evaluated on test-dev. SSD/YOLOv2 numbers are from [35]. The running time is reported for the large core of the Google Pixel 1 phone, using an internal version of the TF-Lite engine.

Network | mAP | Params | MAdd | CPU
--- | --- | --- | --- | --- 
SSD300[34] | 23.2 | 36.1M | 35.2B | -
SSD512[34] | 26.8 | 36.1M | 35.2B | -
YOLOv2[35] | 21.6 | 50.7M | 17.5B | -
MNetV1+SSDLite | 22.2 | 5.1M | 1.3B | 270ms
MNetV2+SSDLite | 22.1 | 4.3M | 0.8B | 200ms

### 6.3. Semantic Segmentation 语义分割

In this section, we compare MobileNetV1 and MobileNetV2 models used as feature extractors with DeepLabv3 [39] for the task of mobile semantic segmentation. DeepLabv3 adopts atrous convolution [40, 41, 42], a powerful tool to explicitly control the resolution of computed feature maps, and builds five parallel heads including (a) Atrous Spatial Pyramid Pooling module (ASPP) [43] containing three 3 × 3 convolutions with different atrous rates, (b) 1 × 1 convolution head, and (c) Image-level features [44]. We denote by output_stride the ratio of input image spatial resolution to final output resolution, which is controlled by applying the atrous convolution properly. For semantic segmentation, we usually employ output_stride = 16 or 8 for denser feature maps. We conduct the experiments on the PASCAL VOC 2012 dataset [3], with extra annotated images from [45] and evaluation metric mIOU.

在本节中，我们将MobileNetV1和MobileNetV2作为特征提取器，与DeepLabv3[39]一起进行移动语义分割，并进行比较。DeepLabv3采用atrous卷积[40,41,42]，这是一种有力的工具，可以显式控制计算的特征图的分辨率，并构建5条并行的heads，包括(a)Atrous空域金字塔pooling模块(ASSP)[43]，包括三个3×3卷积，atrous率不同；(b)1×1卷积head；(c)图像级别的特征[44]。我们用output_stride表示输入图像分辨率与最后的输出分辨率间的比率，这是由atrous卷积控制的。对于语义分割，我们通常采用output_stride=16，对于更密集的特征图为8。我们在PASCAL VOC 2012数据集[3]上进行实验，还有[45]中的另外标注的图像，评估标准为mIOU。

To build a mobile model, we experimented with three design variations: (1) different feature extractors, (2) simplifying the DeepLabv3 heads for faster computation, and (3) different inference strategies for boosting the performance. Our results are summarized in Table 7. We have observed that: (a) the inference strategies, including multi-scale inputs and adding left-right flipped images, significantly increase the MAdds and thus are not suitable for on-device applications, (b) using output_stride = 16 is more efficient than output_stride = 8, (c) MobileNetV1 is already a powerful feature extractor and only requires about 4.9 − 5.7 times fewer MAdds than ResNet-101 [8] (e.g., mIOU: 78.56 vs 82.70, and MAdds: 941.9B vs 4870.6B), (d) it is more efficient to build DeepLabv3 heads on top of the second last feature map of MobileNetV2 than on the original last-layer feature map, since the second to last feature map contains 320 channels instead of 1280, and by doing so, we attain similar performance, but require about 2.5 times fewer operations than the MobileNetV1 counterparts, and (e) DeepLabv3 heads are computationally expensive and removing the ASPP module significantly reduces the MAdds with only a slight performance degradation. In the end of the Table 7, we identify a potential candidate for on-device applications (in bold face), which attains 75.32% mIOU and only requires 2.75B MAdds.

为构建一个移动模型，我们对三种设计变化进行实验：(1)不同的特征提取器，(2)为更加快速的计算简化DeepLabv3 heads，(3)不同的推理策略，以提升性能。我们的结果总结在表7当中。我们观察到：(a)推理策略，包括多尺度输入和增加左右反转图像，明显增加MAdds运算量，所以不适用于移动设备上的应用，(b)使用output_stride=16比output_stride=8要更有效率，(c)MobileNetV1已经是一个强力的特征提取器了，比ResNet-101[8]的MAdds计算量少4.9-5.7倍（如mIOU：78.56 vs 82.70，MAdds：941.9B vs 4870.6B），(d)在MobileNetV2的倒数第二个特征图上构建DeepLabv3 heads，比在原始的最后一层特征图上更有效率，因为倒数第二个特征图包括320个通道，而不是1280，这样做，我们得到类似的表现，但需要的运算量比MobileNetV1的相应操作少了2.5倍，(e)DeepLabv3 heads计算复杂度很高，移除ASSP模块明显降低运算量MAdds，但性能降低很少。在表7的最后，我们标出了一种移动应用的潜在候选（粗体），只需要2.75B MAdds就得到了75.32%的mIOU。

Table 7: MobileNet + DeepLabv3 inference strategy on the PASCAL VOC 2012 validation set. MNetV2*: Second last feature map is used for DeepLabv3 heads, which includes (1) Atrous Spatial Pyramid Pooling (ASPP) module, and (2) 1 × 1 convolution as well as image-pooling feature. OS: output_stride that controls the output resolution of the segmentation map. MF: Multi-scale and left-right flipped inputs during test. All of the models have been pretrained on COCO. The potential candidate for on-device applications is shown in bold face. PASCAL images have dimension 512 × 512 and atrous convolution allows us to control output feature resolution without increasing the number of parameters.

表7：MobileNet+DeepLabv3在PASCAL VOC 2012验证集上的推理策略。MNetV2*：使用倒数第二个特征图构建DeepLabv3 heads，这包括：(1)Atrous空域金字塔pooling(ASSP)模块，(2)1×1卷积和image-pooling特征。OS：控制分割图的输出分辨率的output_stride。MF：测试时的多尺度和左右翻转的输入。所有模型都在COCO上预训练。移动应用的潜在候选用粗体表示出来了。PASCAL图像的维度为512×512，atrous卷积使我们可以控制输出特征的分辨率，而不增加参数数量。

Network | OS | ASPP | MF | mIOU | Params | Madds
--- | --- | --- | --- | --- | --- | ---
MNet V1 | 16 | Y | | 75.29 | 11.15M | 14.25B
MNet V1 | 8 | Y | Y | 78.56 | 11.15M | 941.9B
MNet V2* | 16 | Y | | 75.70 | 4.52M | 5.8B
MNet V2* | 8 | Y | Y | 78.42 | 4.52B | 387B
MNet V2* | 16 | | | 75.32 | 2.11M | 2.75B
MNet V2* | 8 | | Y | 77.33 | 2.11M | 152.6B
ResNet-101 | 16 | Y | | 80.49 | 58.16M | 81.0B
ResNet-101 | 8 | Y | Y | 82.70 | 58.16M | 4870.6B

### 6.4. Ablation study 分离试验

**Inverted residual connections**. The importance of residual connection has been studied extensively [8, 30, 46]. The new result reported in this paper is that the shortcut connecting bottleneck perform better than shortcuts connecting the expanded layers (see Figure 6b for comparison).

**逆残差连接**。残差连接的重要性已经得到广泛研究[8,30,46]。本文得到的新结果是，连接瓶颈的捷径，比连接扩展层的捷径表现要好。（见图6b的比较）。

Figure 6: The impact of non-linearities and various types of shortcut (residual) connections.(a) Impact of non-linearity in the bottleneck layer.(b) Impact of variations in residual blocks.

**Importance of linear bottlenecks**. The linear bottleneck models are strictly less powerful than models with non-linearities, because the activations can always operate in linear regime with appropriate changes to biases and scaling. However our experiments shown in Figure 6a indicate that linear bottlenecks improve performance, providing support that non-linearity destroys information in low-dimensional space.

**线性瓶颈的重要性**。线性瓶颈模型比带有非线性变换的模型要差很多，因为激活可以永远对线性区域运算，只要适当的调整偏移和尺度。但是如图6a所示，我们的实验说明线性瓶颈改进性能，这支持了非线性变换在低维空间中摧毁信息的说法。

## 7. Conclusions and future work 结论和今后工作

We described a very simple network architecture that allowed us to build a family of highly efficient mobile models. Our basic building unit, has several properties that make it particularly suitable for mobile applications. It allows very memory-efficient inference and relies utilize standard operations present in all neural frameworks.

我们提出了一种非常简单的网络架构，使我们能够构建出一族高效的移动模型。我们的基本构建单元有几个性质，这使其特别适合于移动应用。这使推理内存利用很少，而且都是利用的标准运算，在所有的神经网络框架中都存在。

For the ImageNet dataset, our architecture improves the state of the art for wide range of performance points.

对于ImageNet数据集，我们的架构极大改进了最好的性能。

For object detection task, our network outperforms state-of-art realtime detectors on COCO dataset both in terms of accuracy and model complexity. Notably, our architecture combined with the SSDLite detection module is 20× less computation and 10× less parameters than YOLOv2.

对于目标检测任务，我们的网络超过了目前COCO数据集上最好的实时检测器，而且在准确度和模型复杂性上双双超越。值得注意，我们的架构和SSDLite检测模块一起，比YOLOv2的计算量减少了20倍，参数数量减少了10倍。

On the theoretical side: the proposed convolutional block has a unique property that allows to separate the network expressiviness (encoded by expansion layers) from its capacity (encoded by bottleneck inputs). Exploring this is an important direction for future research.

在理论层面：提出的卷积模块有一个唯一的性质，使网络的表达能力（编码在扩展层中）和容量（在瓶颈输入中）分开。未来研究的一个重要方向就是探索这个。

**Acknowledgments** We would like to thank Matt Streeter and Sergey Ioffe for their helpful feedback and discussion.

## Appendix

### A. Bottleneck transformation

### B. Semantic segmentation visualization results