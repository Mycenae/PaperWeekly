# Densely Connected Convolutional Networks

Gao Huang et al. Cornell University Tsinghua University

## Abstract 摘要

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections—one between each layer and its subsequent layer—our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. Code and pre-trained models are available at https://github.com/liuzhuang13/DenseNet.

最近的工作显示，卷积网络的深度可以更深，更准确，如果层与输入或输出之间有更短的连接，那么训练起来也会更高效。本文中，我们利用这种观察结果，提出了密集卷积网络(DenseNet)，每层与其他层之间的连接方式是前向的方式。尽管L层的传统卷积网络有L个连接（每层及其后续的层有一个连接），我们的网络有L(L+1)/2个直接连接。对于每个层，所有前面层的特征图都用于输入，而这一层的特征图则用于所有后续层的输入。DenseNets有几个令人信服的优势：减轻了梯度消失的问题，加强了特征传播，鼓励特征重复使用，极大的降低了参数数量。我们在四种高度有竞争性的目标识别基准检测任务(CIFAR-10, CIFAR-100, SVHN, ImageNet)中评估我们提出的架构。DenseNets比之前所有最好的模型都有明显的改进，同时需要更少的计算，得到来了非常好的性能。代码和预训练模型已经开源。

## 1. Introduction 引言

Convolutional neural networks (CNNs) have become the dominant machine learning approach for visual object recognition. Although they were originally introduced over 20 years ago [18], improvements in computer hardware and network structure have enabled the training of truly deep CNNs only recently. The original LeNet5 [19] consisted of 5 layers, VGG featured 19 [29], and only last year Highway Networks [34] and Residual Networks (ResNets) [11] have surpassed the 100-layer barrier.

CNNs已经成为了视觉目标识别的主流机器学习方法。虽然早在20年前首次提出[18]，但计算机硬件和网络结构的进步，使最近才可能训练真正很深的CNNs。最开始的LeNet5[19]包括5层，VGG有19层[29]，去年高速公路网络[34]和残差网络(ResNets)[11]才超过了100层的门槛。

As CNNs become increasingly deep, a new research problem emerges: as information about the input or gradient passes through many layers, it can vanish and “wash out” by the time it reaches the end (or beginning) of the network. Many recent publications address this or related problems. ResNets [11] and Highway Networks [34] by-pass signal from one layer to the next via identity connections. Stochastic depth [13] shortens ResNets by randomly dropping layers during training to allow better information and gradient flow. FractalNets [17] repeatedly combine several parallel layer sequences with different number of convolutional blocks to obtain a large nominal depth, while maintaining many short paths in the network. Although these different approaches vary in network topology and training procedure, they all share a key characteristic: they create short paths from early layers to later layers.

由于CNNs变的越来越深，就出现了一个新的研究问题：随着输入信息或梯度信息经过了非常的层，会随着时间在最后端（或最前面）消失。很多最近的研究都在处理这个或有关的问题。ResNets[11]和高速公路网络[34]在一层到下一层之间通过恒等映射开辟出一条支路。Stochastic depth[13]通过在训练时随机丢掉一些层，缩短了ResNets，使信息和梯度可以更好的流动。FractalNets[17]重复的将几个并行的层序列合并到一起，合并的层有不同数量的卷积模块，这样虽然网络名义上深度很深，但在网络中有很多很短的路径。虽然这些不同的方法在网络拓扑和训练过程上不一样，但他们都有一个关键的特点：他们从前面的层到后面的层中间创建了更短的路径。

In this paper, we propose an architecture that distills this insight into a simple connectivity pattern: to ensure maximum information flow between layers in the network, we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. Figure 1 illustrates this layout schematically. Crucially, in contrast to ResNets, we never combine features through summation before they are passed into a layer; instead, we combine features by concatenating them. Hence, the l-th layer has l inputs, consisting of the feature-maps of all preceding convolutional blocks. Its own feature-maps are passed on to all L−l subsequent layers. This introduces L(L+1)/2 connections in an L-layer network, instead of just L, as in traditional architectures. Because of its dense connectivity pattern, we refer to our approach as Dense Convolutional Network (DenseNet).

本文中，我们提出一种架构，将这种思想提取出来并形成一种简单的连接模式：为保证网络中各层间最大的信息流，我们讲所有层互相直接链接起来（连接起来的层特征图大小要匹配）。为保持前向的本质，每一层从所有之前的层中得到额外的输入，将其本身的特征图传入所有后续的层。图1所示的就是这种布局。关键是，与ResNets比较起来，我们不会在特征图传入一个层之前用加法将特征图合并；我们是通过拼接来合并特征。所以，第l层有l个输入，包括所有之前的卷积层的特征图。其本身的特征图传入到后续的L-l个后续的层。在L层的网络中，引入了L(L+1)个连接，而不是传统架构的L个连接。由于其密集连接的模式，我们将这种方法称之为DenseNet。

A possibly counter-intuitive effect of this dense connectivity pattern is that it requires fewer parameters than traditional convolutional networks, as there is no need to re-learn redundant feature-maps. Traditional feed-forward architectures can be viewed as algorithms with a state, which is passed on from layer to layer. Each layer reads the state from its preceding layer and writes to the subsequent layer. It changes the state but also passes on information that needs to be preserved. ResNets [11] make this information preservation explicit through additive identity transformations. Recent variations of ResNets [13] show that many layers contribute very little and can in fact be randomly dropped during training. This makes the state of ResNets similar to (unrolled) recurrent neural networks [21], but the number of parameters of ResNets is substantially larger because each layer has its own weights. Our proposed DenseNet architecture explicitly differentiates between information that is added to the network and information that is preserved. DenseNet layers are very narrow (e.g., 12 filters per layer), adding only a small set of feature-maps to the “collective knowledge” of the network and keep the remaining feature-maps unchanged—and the final classifier makes a decision based on all feature-maps in the network.

这种密集连接模式可能的反直觉效果是，比传统卷积网络需要的参数更少，因为不需要重新学习冗余的特征图。传统的前向架构可以视为带有状态的算法，状态从一层传递到另一层。每一层从前一层读取状态，向后续的层写入状态。网络改变状态，但也传递了需要保存的信息。ResNets[11]通过加性恒等变换，使这种信息保存显式表现出来。ResNets[13]最近的变化表明，很多层的贡献很小，实际上可以在训练中随机的丢弃。这使ResNets的状态与（展开的）RNN[21]类似，但ResNets的参数数量却大的多，因为每一层都有其权重。我们提出的DenseNet架构显式的对加入到网络中的信息和保存的信息之间取其差异。DenseNet层非常窄（如，每层12个滤波器），只在网络的共同知识上增加了很少几个特征图，保持剩下的特征图不变，最后的分类器基于网络中所有的特征图做出决策。

Besides better parameter efficiency, one big advantage of DenseNets is their improved flow of information and gradients throughout the network, which makes them easy to train. Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supervision [20]. This helps training of deeper network architectures. Further, we also observe that dense connections have a regularizing effect, which reduces overfitting on tasks with smaller training set sizes.

除了更好的参数效率，DenseNets的一个很大的优点是，网络中的信息流和梯度流得到了改进，这样训练起来更加容易。每一层都可以访问损失函数的梯度和原始输入信号，这带来了隐式的深度监督[20]。这对训练更深的网络架构有好处。而且，我们还观察到，密集连接有正则化的作用，在训练集规模很小的任务中，降低了过拟合效果。

We evaluate DenseNets on four highly competitive benchmark datasets (CIFAR-10, CIFAR-100, SVHN, and ImageNet). Our models tend to require much fewer parameters than existing algorithms with comparable accuracy. Further, we significantly outperform the current state-of-the-art results on most of the benchmark tasks.

我们在四个基准测试数据集(CIFAR-10, CIFAR-100, SVHN, ImageNet)上评估DenseNet。我们的模型在准确率差不多的时候，需要的参数比现有的算法更少。而且，我们在多数基准测试任务中都明显超过了目前最好方法的结果。

## 2. Related Work 相关工作

The exploration of network architectures has been a part of neural network research since their initial discovery. The recent resurgence in popularity of neural networks has also revived this research domain. The increasing number of layers in modern networks amplifies the differences between architectures and motivates the exploration of different connectivity patterns and the revisiting of old research ideas.

网络架构的探索自从最初发现时，就是神经网络研究的一部分。最近神经网络的复苏也使这个研究领域重新充满活力。现代网络中越来越多的层放大了不同架构间的差异，推动了不同连接模式的探索和老的研究思想的再次讨论。

A cascade structure similar to our proposed dense network layout has already been studied in the neural networks literature in the 1980s [3]. Their pioneering work focuses on fully connected multi-layer perceptrons trained in a layer-by-layer fashion. More recently, fully connected cascade networks to be trained with batch gradient descent were proposed [40]. Although effective on small datasets, this approach only scales to networks with a few hundred parameters. In [9, 23, 31, 41], utilizing multi-level features in CNNs through skip-connnections has been found to be effective for various vision tasks. Parallel to our work, [1] derived a purely theoretical framework for networks with cross-layer connections similar to ours.

与我们提出的密集连接网络类似的级联结构，已经在1980年代的神经网络文献中得到过研究[3]。他们的先驱工作聚焦在全连接多层感知机上，训练时采用逐层训练的方式。再近一些，[40]提出了全连接级联网络的批梯度下降训练方法。虽然在小数据集上很有效，这种方法只对于有几百个参数的网络有效。在[9,23,31,41]中，通过跳跃连接利用CNNs中的多层特征，证明了在多种视觉任务中都很有效。与我们的工作同时，[1]推导出了跨层连接网络纯理论上的框架，与我们的类似。

Highway Networks [34] were amongst the first architectures that provided a means to effectively train end-to-end networks with more than 100 layers. Using bypassing paths along with gating units, Highway Networks with hundreds of layers can be optimized without difficulty. The bypassing paths are presumed to be the key factor that eases the training of these very deep networks. This point is further supported by ResNets [11], in which pure identity mappings are used as bypassing paths. ResNets have achieved impressive, record-breaking performance on many challenging image recognition, localization, and detection tasks, such as ImageNet and COCO object detection [11]. Recently, stochastic depth was proposed as a way to successfully train a 1202-layer ResNet [13]. Stochastic depth improves the training of deep residual networks by dropping layers randomly during training. This shows that not all layers may be needed and highlights that there is a great amount of redundancy in deep (residual) networks. Our paper was partly inspired by that observation. ResNets with pre-activation also facilitate the training of state-of-the-art networks with > 1000 layers [12].

高速公路网络[34]是第一批可以有效的对超过100层的网络进行端到端的训练架构之一。使用旁路通道和门单元，高速公路网络有数百层，可以很轻松的进行最优化。旁路通道被认为是训练这些非常深的网络的关键因素。这一点进一步由ResNets[11]得到证明，其中的旁路通道使用了纯恒等映射。ResNets在很多很有挑战的图像识别、定位和检测任务中取得了令人印象深刻的破纪录的性能，如ImageNet和COCO目标检测[11]。最近，[13]提出了随机深度方法，成功的训练了1202层的ResNet。随机深度通过在训练过程中随机的丢弃一些层，改进了深度残差网络的训练。这说明，并不是所有的层都是必要的，深度（残差）网络中有很多冗余性。我们的文章部分受此观察结果启发。预激活的ResNets[12]也帮助训练了目前最好的超过1000层的网络。

An orthogonal approach to making networks deeper (e.g., with the help of skip connections) is to increase the network width. The GoogLeNet [36, 37] uses an “Inception module” which concatenates feature-maps produced by filters of different sizes. In [38], a variant of ResNets with wide generalized residual blocks was proposed. In fact, simply increasing the number of filters in each layer of ResNets can improve its performance provided the depth is sufficient [42]. FractalNets also achieve competitive results on several datasets using a wide network structure [17].

另一种使网络更深的方法是增加网络宽度。GoogLeNet[36,37]使用一种Inception模块，将不同大小的滤波器生成的特征图拼接起来。[38]中，ResNets的一种变体，提出了一种加宽的扩展残差模块。事实上，只是增加ResNets中每层滤波器的数量，就可以改进其性能，只要深度足够[42]。FractalNets[17]使用了一种很宽的网络架构，也在几个数据集上取得了很不错的结果。

Instead of drawing representational power from extremely deep or wide architectures, DenseNets exploit the potential of the network through feature reuse, yielding condensed models that are easy to train and highly parameter-efficient. Concatenating feature-maps learned by different layers increases variation in the input of subsequent layers and improves efficiency. This constitutes a major difference between DenseNets and ResNets. Compared to Inception networks [36, 37], which also concatenate features from different layers, DenseNets are simpler and more efficient.

DenseNet没有利用非常深或非常宽的架构的表示能力，而是挖掘网络特征重复使用的潜力，得到了容易训练、参数利用率高的压缩模型。拼接不同层学习到的特征图，增加了后续层输入的变化，改进了效率。这是DenseNets和ResNets的主要不同。与Inception网络[36,37]相比（这种网络也拼接不同层的特征），DenseNet更简单也更有效率。

There are other notable network architecture innovations which have yielded competitive results. The Network in Network (NIN) [22] structure includes micro multi-layer perceptrons into the filters of convolutional layers to extract more complicated features. In Deeply Supervised Network (DSN) [20], internal layers are directly supervised by auxiliary classifiers, which can strengthen the gradients received by earlier layers. Ladder Networks [27, 25] introduce lateral connections into autoencoders, producing impressive accuracies on semi-supervised learning tasks. In [39], Deeply-Fused Nets (DFNs) were proposed to improve information flow by combining intermediate layers of different base networks. The augmentation of networks with pathways that minimize reconstruction losses was also shown to improve image classification models [43].

还有其他网络架构的创新，也得到了很好的结果。Network in Network(NIN)[22]结构中卷积层的滤波器包括了微型多层感知机，可以提取更复杂的特征。在Deeply Supervised Network(DSN)[20]中，内部层直接由辅助分类器监督，可以加强之前的层收到的梯度。Ladder Network[27,25]在自动编码机中引入了横向连接，在半监督学习任务中得到了很好的准确率。在[39]中，Densely-Fused Nets(DFNs)通过将不同基础网络的中间层综合到一起，以改进信息流。网络使用旁路进行扩充，最小化了重建损失，可以改进图像分类模型[43]。

## 3. DenseNets

Consider a single image $x_0$ that is passed through a convolutional network. The network comprises L layers, each of which implements a non-linear transformation $H_l$(·), where l indexes the layer. $H_l$(·) can be a composite function of operations such as Batch Normalization (BN) [14], rectified linear units (ReLU) [6], Pooling [19], or Convolution (Conv). We denote the output of the l-th layer as $x_l$.

考虑单幅图像$x_0$流经一个卷积网络。网络由L层组成，每一层实现的是一个非线性变换$H_l$(·)，其中l是层的索引。$H_l$(·)可以是一些运算的复合函数，如BN[14]，ReLU[6]，Pooling[19]，或卷积(Conv)。我们将第l层的输出表示为$x_l$。

**ResNets**. Traditional convolutional feed-forward networks connect the output of the l-th layer as input to the (l+1)-th layer [16], which gives rise to the following layer transition: $x_l = H_l (x_l−1)$. ResNets [11] add a skip-connection that bypasses the non-linear transformations with an identity function: 传统卷积前向网络将第l层的输出作为第l+1层的输入[16]，于是有下面的层变换公式：$x_l = H_l (x_l−1)$。ResNets[11]增加了一个跳跃连接，其恒等连接是非线性变换的旁路：

$$x_l = H_l(x_{l-1}) + x_{l-1}$$(1)

An advantage of ResNets is that the gradient can flow directly through the identity function from later layers to the earlier layers. However, the identity function and the output of $H_l$ are combined by summation, which may impede the information flow in the network. ResNets的优势是梯度可以直接经由恒等函数从后面的层流向前面的层。但是，恒等函数和$H_l$的输出由加法结合起来，可能会阻碍网络中的信息流。

**Dense connectivity**. To further improve the information flow between layers we propose a different connectivity pattern: we introduce direct connections from any layer to all subsequent layers. Figure 1 illustrates the layout of the resulting DenseNet schematically. Consequently, the l-th layer receives the feature-maps of all preceding layers, $x_0, . . . , x_{l−1}$, as input: 为进一步改进层之间的信息流，我们提出一种不同的连接模式：我们从任意层往其后的所有层引入直接连接。图1给出了DenseNet的图示。结果是，第l层的输入由之前所有层的特征图构成，$x_0, . . . , x_{l−1}$：

$$x_l = H_l ([x_0, x_1, . . . , x_{l−1}])$$(2)

where $[x_0, x_1, . . . , x_{l−1}]$ refers to the concatenation of the feature-maps produced in layers 0, . . . , l−1. Because of its dense connectivity we refer to this network architecture as Dense Convolutional Network (DenseNet). For ease of implementation, we concatenate the multiple inputs of $H_l$(·) in eq. (2) into a single tensor. 其中$[x_0, x_1, . . . , x_{l−1}]$表示层0, . . . , l−1生成的特征图的拼接。由于其连接的密集性，我们称这种网络架构为DenseNet。为容易实现，我们将(2)式中$H_l$(·)的多个输入拼接为单个张量。

**Composite function**. Motivated by [12], we define $H_l$(·) as a composite function of three consecutive operations: batch normalization (BN) [14], followed by a rectified linear unit (ReLU) [6] and a 3×3 convolution (Conv). 符合函数。受[12]启发，我们将$H_l$(·)定义为三个算子的符合函数：BN[14]，ReLU[6]和3×3卷积(Conv)。

**Pooling layers**. The concatenation operation used in Eq. (2) is not viable when the size of feature-maps changes. However, an essential part of convolutional networks is down-sampling layers that change the size of feature-maps. To facilitate down-sampling in our architecture we divide the network into multiple densely connected dense blocks; see Figure 2. We refer to layers between blocks as transition layers, which do convolution and pooling. The transition layers used in our experiments consist of a batch normalization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer. 池化层。式(2)中用到的拼接算子在特征图的大小变化时是不可用的。但是，卷积网络的一个基本组成部分是下采样层，会改变特征图的大小。为使我们的架构也可以使用下采样层，我们将网络分解称多个密集连接块，如图2所示。我们将模块之间的层为过渡层，其任务就是卷积和池化。我们试验中使用的过渡层包括BN层和1×1卷积层、2×2平均池化层。

Figure 2: A deep DenseNet with three dense blocks. The layers between two adjacent blocks are referred to as transition layers and change feature-map sizes via convolution and pooling.

**Growth rate**. If each function $H_l$ produces k feature-maps, it follows that the l-th layer has $k_0 + k × (l − 1)$ input feature-maps, where $k_0$ is the number of channels in the input layer. An important difference between DenseNet and existing network architectures is that DenseNet can have very narrow layers, e.g., k = 12. We refer to the hyper-parameter k as the growth rate of the network. We show in Section 4 that a relatively small growth rate is sufficient to obtain state-of-the-art results on the datasets that we tested on. One explanation for this is that each layer has access to all the preceding feature-maps in its block and, therefore, to the network’s “collective knowledge”. One can view the feature-maps as the global state of the network. Each layer adds k feature-maps of its own to this state. The growth rate regulates how much new information each layer contributes to the global state. The global state, once written, can be accessed from everywhere within the network and, unlike in traditional network architectures, there is no need to replicate it from layer to layer.

**增长速率**。如果每个函数$H_l$产生l个特征图，那么第l层有$k_0 + k × (l − 1)$个输入特征图，其中$k_0$是输入层的通道数量。DenseNet和现有网络架构的一个重要区别是，DenseNet的层可以很窄，如k=12。我们称超参数k为网络的增长速率。我们在第4部分说明，相对很小的增长速率k就足够在我们测试的数据集上得到目前最好的结果。一个解释是，每一层都可以获得之前所有特征图的信息，所以，就是可以访问到网络的集体知识。我们可以将特征图视为网络的全局状态。每一层将自身的k个特征图添加到这个状态上。增长速率规定了每一层给全局状态贡献多少新信息。全局状态可以在网络中任意位置访问，与传统的网络架构不同，没有必要将之从一层复制到另一层。

**Bottleneck layers**. Although each layer only produces k output feature-maps, it typically has many more inputs. It has been noted in [37, 11] that a 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency. We find this design especially effective for DenseNet and we refer to our network with such a bottleneck layer, i.e., to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of $H_l$, as DenseNet-B. In our experiments, we let each 1×1 convolution produce 4k feature-maps.

**瓶颈层**。虽然每一层只产生k个输出特征图，但其输入会多很多。[37,11]中已经说明，1×1卷积可以用于瓶颈层，以减少输入特征图的数量，然后再进行3×3卷积，所以可以改进计算效率。我们发现这种设计对于DenseNet非常有效，我们称带有这种结构的DenseNet为DenseNet-B，即BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)版的$H_l$。在我们的试验中，我们让每个1×1卷积生成4k特征图。

**Compression**. To further improve model compactness, we can reduce the number of feature-maps at transition layers. If a dense block contains m feature-maps, we let the following transition layer generate [θm] output feature-maps, where 0 < θ ≤ 1 is referred to as the compression factor. When θ = 1, the number of feature-maps across transition layers remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5 in our experiment. When both the bottleneck and transition layers with θ < 1 are used, we refer to our model as DenseNet-BC.

**压缩**。为进一步提高模型的紧凑型，我们可以降低过渡层中的特征图数量。如果一个密集模块包含m个特征图，我们令下列过渡层生成[θm]个输出特征图，其中0 < θ ≤ 1称为压缩因子。如果θ = 1，经过过渡层处理后特征图的数量不变。我们称θ < 1的DenseNet为DenseNet-C，在我们的试验中我们设θ = 0.5。如果使用了瓶颈层，也使用了θ < 1的过渡层，我们称模型为DenseNet-BC。

**Implementation Details**. On all datasets except ImageNet, the DenseNet used in our experiments has three dense blocks that each has an equal number of layers. Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC) output channels is performed on the input images. For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel to keep the feature-map size fixed. We use 1×1 convolution followed by 2×2 average pooling as transition layers between two contiguous dense blocks. At the end of the last dense block, a global average pooling is performed and then a softmax classifier is attached. The feature-map sizes in the three dense blocks are 32× 32, 16×16, and 8×8, respectively. We experiment with the basic DenseNet structure with configurations {L = 40, k = 12}, {L = 100, k = 12} and {L = 100, k = 24}. For DenseNet-BC, the networks with configurations {L = 100, k = 12}, {L = 250, k = 24} and {L = 190, k = 40} are evaluated.

**实现细节**。在除了ImageNet的其他所有数据集中，试验中使用的DenseNet有三个密集块，每一个块的层数都相等。在进入第一个密集块之前，输入图像先进行一次卷积，输出16个通道（或DenseNet-BC的增长速率的2倍）。对3×3卷积层，输入的四边都用0值补充一个像素，保证特征图大小固定。我们使用1×1卷积和2×2平均池化的组合作为过渡层，即两个密集模块之间的层。在最后一个密集模块之后，进行全局平均池化然后是softmax分类器。三个密集模块中的特征图大小分别是32× 32, 16×16和8×8。我们用基本的DenseNet结构进行试验，配置为{L = 40, k = 12}, {L = 100, k = 12}和{L = 100, k = 24}。对于DenseNet-BC，评估的网络配置为{L = 100, k = 12}, {L = 250, k = 24}和{L = 190, k = 40}。

In our experiments on ImageNet, we use a DenseNet-BC structure with 4 dense blocks on 224×224 input images. The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2; the number of feature-maps in all other layers also follow from setting k. The exact network configurations we used on ImageNet are shown in Table 1.

我们在ImageNet上的试验中，使用的DenseNet-BC结构有4个密集模块，输入图像大小224×224。初始卷积层由2k个卷积组成，大小7×7，步长为2；其他所有层的特征图数量也使用设置k的数量。我们在ImageNet上使用的网络配置如表1所示。

Table 1: DenseNet architectures for ImageNet. The growth rate for all the networks is k = 32. Note that each “conv” layer shown in the table corresponds the sequence BN-ReLU-Conv.

## 4. Experiments 试验

We empirically demonstrate DenseNet’s effectiveness on several benchmark datasets and compare with state-of-the-art architectures, especially with ResNet and its variants. 我们根据试验表明DenseNet在几个标准测试数据集的有效性，与目前最好的架构进行比较，尤其是与ResNet及其变体。

### 4.1. Datasets 数据集

**CIFAR**. The two CIFAR datasets [15] consist of colored natural images with 32×32 pixels. CIFAR-10 (C10) consists of images drawn from 10 and CIFAR-100 (C100) from 100 classes. The training and test sets contain 50,000 and 10,000 images respectively, and we hold out 5,000 training images as a validation set. We adopt a standard data augmentation scheme (mirroring/shifting) that is widely used for these two datasets [11, 13, 17, 22, 28, 20, 32, 34]. We denote this data augmentation scheme by a “+” mark at the end of the dataset name (e.g., C10+). For preprocessing, we normalize the data using the channel means and standard deviations. For the final run we use all 50,000 training images and report the final test error at the end of training.

**CIFAR**。两个CIFAR数据集[15]是由32×32像素的彩色自然图像组成。CIFAR-10包含10类图像，CIFAR-100包含100类图像。训练集和测试集分别包含50000和10000幅图像，我们保留5000训练集图像作为验证集。我们采用标准的数据扩充方法（镜像/平移），这在这两个数据集中有着广泛的使用[11, 13, 17, 22, 28, 20, 32, 34]。我们将这种数据扩充方法表示为数据集名字后面的“+”号（如C10+）。对于预处理，我们使用通道平均值和标准偏差来对数据进行正则化。对于最后一轮，我们使用全部50000幅图像，并在训练结束后给出最终的测试错误率。

**SVHN**. The Street View House Numbers (SVHN) dataset [24] contains 32×32 colored digit images. There are 73,257 images in the training set, 26,032 images in the test set, and 531,131 images for additional training. Following common practice [7, 13, 20, 22, 30] we use all the training data without any data augmentation, and a validation set with 6,000 images is split from the training set. We select the model with the lowest validation error during training and report the test error. We follow [42] and divide the pixel values by 255 so they are in the [0, 1] range.

**SVHN**。街景门牌号(SVHN)数据集[24]包含32×32大小的彩色数字字符图像。训练集中有73257幅图像，测试集26032，还有531131幅可供额外训练。我们采用一般的方法[7, 13, 20, 22, 30]，即使用所有训练数据，不采用数据扩充技术，从训练集中分离出6000幅图像进行验证。我们在训练过程中选择最低验证误差的模型，给出测试误差。我们采用[42]中的方法，将像素值除以255以归一化，数值范围为[0,1]。

**ImageNet**. The ILSVRC 2012 classification dataset [2] consists 1.2 million images for training, and 50,000 for validation, from 1, 000 classes. We adopt the same data augmentation scheme for training images as in [8, 11, 12], and apply a single-crop or 10-crop with size 224×224 at test time. Following [11, 12, 13], we report classification errors on the validation set.

**ImageNet**。ILSVRC2012分类数据集[2]包含1.2 million训练图像，50000幅验证图像，共1000类。我们采用[8, 11, 12]相同的数据扩充技术进行训练，在测试时采用单剪切块或10个剪切块224×224大小的图像。采用[11,12,13]的方法，我们在验证集上给出分类错误率。

### 4.2. Training 训练

All the networks are trained using stochastic gradient descent (SGD). On CIFAR and SVHN we train using batch size 64 for 300 and 40 epochs, respectively. The initial learning rate is set to 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs. On ImageNet, we train models for 90 epochs with a batch size of 256. The learning rate is set to 0.1 initially, and is lowered by 10 times at epoch 30 and 60. Note that a naive implementation of DenseNet may contain memory inefficiencies. To reduce the memory consumption on GPUs, please refer to our technical report on the memory-efficient implementation of DenseNets [26].

所有的网络都使用随机梯度下降(SGD)进行训练。在CIFAR和SVHN上，我们训练使用batch size 64,分别训练300轮和40轮。初始学习率设为0.1，在总训练轮数的50%和75%处除以10。在ImageNet上，我们训练90轮，batch size 256。初始学习速率设为0.1，在第30轮和第60轮的时候除以10。注意，DenseNet的简单实现可能会有内存利用率不高的问题。为降低GPU内存使用量，请参考我们DenseNet实现内存使用率的技术报告[26]。

Following [8], we use a weight decay of $10^{−4}$ and a Nesterov momentum [35] of 0.9 without dampening. We adopt the weight initialization introduced by [10]. For the three datasets without data augmentation, i.e., C10, C100 and SVHN, we add a dropout layer [33] after each convolutional layer (except the first one) and set the dropout rate to 0.2. The test errors were only evaluated once for each task and model setting.

采用[8]中的方案，我们使用的权重衰减为$10^{−4}$，Nesterov动量[35]为0.9。我们使用的权重初始化方案与[10]中的一样。对于没有使用数据扩充的三个数据集，即C10，C100和SVHN，我们在每个卷积层（除了第一层）后，增加一个dropout层，设dropout率为0.2。对于每个任务的每种模型设置，只评估一次测试错误率。

### 4.3. Classification Results on CIFAR and SVHN 分类结果

We train DenseNets with different depths, L, and growth rates, k. The main results on CIFAR and SVHN are shown in Table 2. To highlight general trends, we mark all results that outperform the existing state-of-the-art in boldface and the overall best result in blue.

我们用不同深度L和增长速率k来训练DenseNets。在CIFAR和SVHN上的主要结果如表2所示。为强调总体趋势，我们对超过现有最好方法的结果进行加粗，总体最好的结果以蓝色显示。

Table 2: Error rates (%) on CIFAR and SVHN datasets. k denotes network’s growth rate. Results that surpass all competing methods are bold and the overall best results are blue. “+” indicates standard data augmentation (translation and/or mirroring). ∗ indicates results run by ourselves. All the results of DenseNets without data augmentation (C10, C100, SVHN) are obtained using Dropout. DenseNets achieve lower error rates while using fewer parameters than ResNet. Without data augmentation, DenseNet performs better by a large margin.

**Accuracy**. Possibly the most noticeable trend may originate from the bottom row of Table 2, which shows that DenseNet-BC with L = 190 and k = 40 outperforms the existing state-of-the-art consistently on all the CIFAR datasets. Its error rates of 3.46% on C10+ and 17.18% on C100+ are significantly lower than the error rates achieved by wide ResNet architecture [42]. Our best results on C10 and C100 (without data augmentation) are even more encouraging: both are close to 30% lower than FractalNet with drop-path regularization [17]. On SVHN, with dropout, the DenseNet with L = 100 and k = 24 also surpasses the current best result achieved by wide ResNet. However, the 250-layer DenseNet-BC doesn’t further improve the performance over its shorter counterpart. This may be explained by that SVHN is a relatively easy task, and extremely deep models may overfit to the training set.

**准确率**。可能最引人注意的趋势能从表2的下面的行中看到，说明了DenseNet-BC在L=190，k=40的情况下超过了现有最好的算法在CIFAR数据集上的表现。其在C10+上的错误率3.46%和C100+上的错误率17.18%比宽ResNet架构[42]要明显好很多。我们在C10和C100上取得的最好结果（没有数据扩充）更鼓舞人心：比使用了drop-path正则化[17]的FractalNet低了30%。在SVHN上，使用了dropout的情况下，L=100、k=24的DenseNet超过了目前宽型ResNet最好的结果。但是，250层的DenseNet-BC并没有比浅一些的模型有所改进。这可能是因为，SVHN是一个相对简单的任务，非常深的模型可能在这个训练集上过拟合。

**Capacity**. Without compression or bottleneck layers, there is a general trend that DenseNets perform better as L and k increase. We attribute this primarily to the corresponding growth in model capacity. This is best demonstrated by the column of C10+ and C100+. On C10+, the error drops from 5.24% to 4.10% and finally to 3.74% as the number of parameters increases from 1.0M, over 7.0M to 27.2M. On C100+, we observe a similar trend. This suggests that DenseNets can utilize the increased representational power of bigger and deeper models. It also indicates that they do not suffer from overfitting or the optimization difficulties of residual networks [11].

**能力**。在没有压缩或瓶颈层的情况下，随着L和k的增加，DenseNet的表现会越来越好。这主要是因为，模型的容量也相应的增加了。最好的表现可以参考C10+和C100+的列。在C10+上，随着参数数量由1.0M增加到7.0M，最后到27.2M，错误率由5.24%降到了4.10%，最后到了3.74%。在C100+上，我们观察到类似的趋势。这说明了DenseNets可以利用更大更深的模型的提升的表示能力。这也说明，DenseNet没有残差网络[11]中的过拟合问题或优化困难问题。

**Parameter Efficiency**. The results in Table 2 indicate that DenseNets utilize parameters more efficiently than alternative architectures (in particular, ResNets). The DenseNet-BC with bottleneck structure and dimension reduction at transition layers is particularly parameter-efficient. For example, our 250-layer model only has 15.3M parameters, but it consistently outperforms other models such as FractalNet and Wide ResNets that have more than 30M parameters. We also highlight that DenseNet-BC with L = 100 and k = 12 achieves comparable performance (e.g., 4.51% vs 4.62% error on C10+, 22.27% vs 22.71% error on C100+) as the 1001-layer pre-activation ResNet using 90% fewer parameters. Figure 4 (right panel) shows the training loss and test errors of these two networks on C10+. The 1001-layer deep ResNet converges to a lower training loss value but a similar test error. We analyze this effect in more detail below.

**参数效率**。表2的结果说明，DenseNets利用参数的效率比其他模型更高（尤其是ResNets）。DenseNet-BC利用了瓶颈结构，和过渡层的维度缩减，所以参数利用效率尤其高。比如，我们的250层模型只有15.3M参数，但在各种情况下都超过了其他模型，如拥有超过30M参数的FractalNet和Wide ResNets。我们还强调了，L=100、k=12情况下的DenseNet-BC取得了与1001层预激活的ResNet可以比较的性能，使用的参数少了90%（如，在C10+上错误率4.51% vs 4.62%，在C100+上22.27% vs 22.71%）。图4（右）给出了这两种网络在C10+上的训练损失和测试错误率。1001层的深度ResNet训练收敛的损失值更低，但是测试错误率类似。下面我们更详细的分析这个现象。

**Overfitting**. One positive side-effect of the more efficient use of parameters is a tendency of DenseNets to be less prone to overfitting. We observe that on the datasets without data augmentation, the improvements of DenseNet architectures over prior work are particularly pronounced. On C10, the improvement denotes a 29% relative reduction in error from 7.33% to 5.19%. On C100, the reduction is about 30% from 28.20% to 19.64%. In our experiments, we observed potential overfitting in a single setting: on C10, a 4× growth of parameters produced by increasing k = 12 to k = 24 lead to a modest increase in error from 5.77% to 5.83%. The DenseNet-BC bottleneck and compression layers appear to be an effective way to counter this trend.

**过拟合**。参数利用效率更高，其一个正面的副作用是，DenseNet不容易过拟合。我们观察到，在没有进行数据扩充的数据集上，DenseNet比之前的工作的改进特别明显。在C10上，这种改进是错误率从7.33%到5.19%，大约29%的相对减少。在C100上，这种改进是从28.20%到19.64%，降低大约30%。在我们的试验中，我们只在一个设置中观察到可能的过拟合：在C10上，k=12到k=24时，参数增长了4倍，错误率从5.77%增加到了5.83%。DenseNet-BC的瓶颈层和压缩层是应对这种趋势的很好方法。

### 4.4. Classification Results on ImageNet 在ImageNet上的分类结果

We evaluate DenseNet-BC with different depths and growth rates on the ImageNet classification task, and compare it with state-of-the-art ResNet architectures. To ensure a fair comparison between the two architectures, we eliminate all other factors such as differences in data preprocessing and optimization settings by adopting the publicly available Torch implementation for ResNet by [8]. We simply replace the ResNet model with the DenseNet-BC network, and keep all the experiment settings exactly the same as those used for ResNet.

我们对不同深度和增长速率的DenseNet-BC在ImageNet分类任务上进行评估，并与目前最好的ResNet架构进行比较。为确保两个框架之间的公平比较，我们去除了其他因素，如数据预处理和优化设置上的差异，采用公开可用的Torch实现的ResNet [8]。我们只是将ResNet模型替换为DenseNet-BC网络，保持其他试验设置不变。

We report the single-crop and 10-crop validation errors of DenseNets on ImageNet in Table 3. Figure 3 shows the single-crop top-1 validation errors of DenseNets and ResNets as a function of the number of parameters (left) and FLOPs (right). The results presented in the figure reveal that DenseNets perform on par with the state-of-the-art ResNets, whilst requiring significantly fewer parameters and computation to achieve comparable performance. For example, a DenseNet-201 with 20M parameters model yields similar validation error as a 101-layer ResNet with more than 40M parameters. Similar trends can be observed from the right panel, which plots the validation error as a function of the number of FLOPs: a DenseNet that requires as much computation as a ResNet-50 performs on par with a ResNet-101, which requires twice as much computation.

我们在表3中给出DenseNets在ImageNet上的单剪切块和10剪切块的验证错误率。图3给出了单剪切块的top-1验证错误率，包括DenseNets和ResNets，作为参数数量（左）和FLOPs（右）的函数。图中的结果说明，DenseNets与目前最好的ResNets结果类似，但需要的参数和计算量要少的多。例如，DenseNet-201参数量20M，得到的结果与ResNet-101类似，其参数量大于40M。类似的趋势可以从右边的面板看出来，其验证错误率是FLOPs的函数：与ResNet50计算量相同的DenseNet，可以达到ResNet-101的结果，而这需要两倍的计算量。

Table 3: The top-1 and top-5 error rates on the ImageNet validation set, with single-crop/10-crop testing.

Model | top-1 | top-5
--- | --- | ---
DenseNet-121 | 25.02 / 23.61 | 7.71 / 6.66
DenseNet-169 | 23.80 / 22.08 | 6.85 / 5.92
DenseNet-201 | 22.58 / 21.46 | 6.34 / 5.54
DenseNet-264 | 22.15 / 20.80 | 6.12 / 5.29

Figure 3: Comparison of the DenseNets and ResNets top-1 error rates (single-crop testing) on the ImageNet validation dataset as a function of learned parameters (left) and FLOPs during test-time (right).

It is worth noting that our experimental setup implies that we use hyperparameter settings that are optimized for ResNets but not for DenseNets. It is conceivable that more extensive hyper-parameter searches may further improve the performance of DenseNet on ImageNet. 需要指出，我们的试验设置说明，我们使用的超参数是为ResNets优化过的，而不是DenseNets。可以想象，如果进行超参数搜索，则可以进一步改进DenseNet在ImageNet上的分类效果。

## 5. Discussion 讨论

Superficially, DenseNets are quite similar to ResNets:　Eq. (2) differs from Eq. (1) only in that the inputs to $H_l$(·) are concatenated instead of summed. However, the implications of this seemingly small modification lead to substantially different behaviors of the two network architectures.

表面来看，DenseNets与ResNets很像：式(2)与式(1)的区别只在于$H_l$(·)的输入是拼接在一起的，而不是求和的。但是，这看似很小的改变会导致两种网络架构间完全不同的的行为。

**Model compactness**. As a direct consequence of the input concatenation, the feature-maps learned by any of the DenseNet layers can be accessed by all subsequent layers. This encourages feature reuse throughout the network, and leads to more compact models.

**模型紧凑型**。作为输入拼接的直接结果，DenseNet任意层学习到的特征图都可以被后续的所有层访问到。这鼓励了网络中的特征重复使用，带来了更紧凑的模型。

The left two plots in Figure 4 show the result of an experiment that aims to compare the parameter efficiency of all variants of DenseNets (left) and also a comparable ResNet architecture (middle). We train multiple small networks with varying depths on C10+ and plot their test accuracies as a function of network parameters. In comparison with other popular network architectures, such as AlexNet [16] or VGG-net [29], ResNets with pre-activation use fewer parameters while typically achieving better results [12]. Hence, we compare DenseNet (k = 12) against this architecture. The training setting for DenseNet is kept the same as in the previous section.

图4的左边两个图，给出的是所有DenseNets的变体（左）和一个可比的ResNet架构（中）的参数利用效率。我们在C10+上训练多个不同深度的小网络，画出了测试准确率对网络参数数量的函数。与其他流行的网络架构比较起来，如AlexNet[16]或VGGNet[16]，带有预激活的ResNets一般都会取得更好的结果[12]。所以，我们将DenseNet(k=12)与这种架构进行比较。DenseNet的训练设置与前面一节相同。

Figure 4: Left: Comparison of the parameter efficiency on C10+ between DenseNet variations. Middle: Comparison of the parameter efficiency between DenseNet-BC and (pre-activation) ResNets. DenseNet-BC requires about 1/3 of the parameters as ResNet to achieve comparable accuracy. Right: Training and testing curves of the 1001-layer pre-activation ResNet [12] with more than 10M parameters and a 100-layer DenseNet with only 0.8M parameters.

The graph shows that DenseNet-BC is consistently the most parameter efficient variant of DenseNet. Further, to achieve the same level of accuracy, DenseNet-BC only requires around 1/3 of the parameters of ResNets (middle plot). This result is in line with the results on ImageNet we presented in Figure 3. The right plot in Figure 4 shows that a DenseNet-BC with only 0.8M trainable parameters is able to achieve comparable accuracy as the 1001-layer (pre-activation) ResNet [12] with 10.2M parameters.

从图中可以看出，DenseNet-BC一直是DenseNet变体中参数利用效率最高的。而且，为达到相同水平的准确率，DenseNet-BC只需要ResNets参数的1/3（中间图）。这个结果符合图3中在ImageNet上的结果。图4右图所示的DenseNet-BC只使用了0.8M可训练的参数，得到的准确率与1001层（预激活）的ResNet[12]类似，这个模型参数量达到了10.2M。

**Implicit Deep Supervision**. One explanation for the improved accuracy of dense convolutional networks may be that individual layers receive additional supervision from the loss function through the shorter connections. One can interpret DenseNets to perform a kind of “deep supervision”. The benefits of deep supervision have previously been shown in deeply-supervised nets (DSN; [20]), which have classifiers attached to every hidden layer, enforcing the intermediate layers to learn discriminative features.

**隐式的深度监督**。DenseNet改进准确率的一种解释是，单个的层从更短的连接中受到损失函数的额外监督，可以将DenseNets解释为进行了一种深度监督。深度监督的好处之前在DSN[20](deeply-supervised nets)中得到过验证，其分类器连接到每个隐含层之上，确保中间层学习到有分辨力的特征。

DenseNets perform a similar deep supervision in an implicit fashion: a single classifier on top of the network provides direct supervision to all layers through at most two or three transition layers. However, the loss function and gradient of DenseNets are substantially less complicated, as the same loss function is shared between all layers.

DenseNets以隐式的方式进行了类似的深度监督：网络最后的分类器，通过两到三个过渡层，对所有层提供了直接监督。但是，DenseNet的损失函数和梯度没那么复杂，因为在所有层中都共享相同的损失函数。

**Stochastic vs. deterministic connection**. There is an interesting connection between dense convolutional networks and stochastic depth regularization of residual networks [13]. In stochastic depth, layers in residual networks are randomly dropped, which creates direct connections between the surrounding layers. As the pooling layers are never dropped, the network results in a similar connectivity pattern as DenseNet: there is a small probability for any two layers, between the same pooling layers, to be directly connected—if all intermediate layers are randomly dropped. Although the methods are ultimately quite different, the DenseNet interpretation of stochastic depth may provide insights into the success of this regularizer.

**随机连接和确定性连接**。DenseNet和随机深度正则化的残差网络[13]有着很有趣的联系。在随机深度中，残差网络中的层随机丢弃，这在周围的层当中生成了直接连接。由于池化层一直没有丢弃，网络得到了与DenseNet类似的连接模式：在相同的池化层之间，任意两层都有很小的概率进行直接连接，因为中间层是随机抛弃的。虽然方法截然不同，随机深度的DenseNet解释也可以看到这种正则化器的思想。

**Feature Reuse**. By design, DenseNets allow layers access to feature-maps from all of its preceding layers (although sometimes through transition layers). We conduct an experiment to investigate if a trained network takes advantage of this opportunity. We first train a DenseNet on C10+ with L = 40 and k = 12. For each convolutional layer l within a block, we compute the average (absolute) weight assigned to connections with layer s. Figure 5 shows a heat-map for all three dense blocks. The average absolute weight serves as a surrogate for the dependency of a convolutional layer on its preceding layers. A red dot in position (l, s) indicates that the layer l makes, on average, strong use of feature-maps produced s-layers before. Several observations can be made from the plot:

**特征重用**。从设计上来说，DenseNets允许各层访问所有之前的层的特征图（有时候是通过过渡层）。我们设计一个试验，研究一下训练好的网络是否从这种机制中受益。我们首先在C10+上训练一个DenseNet，L=40，k=12。对一个模块中的每个卷积层l，我们计算层s的连接上的平均（绝对）权重。图5给出了所有三个密集块的热力图。平均绝对权重可以代表一个卷积层对其之前的层的依赖性。在(l,s)位置上的红点说明层l利用了很多前s层的特征图。可以从这个图中观察到如下结果：

1. All layers spread their weights over many inputs within the same block. This indicates that features extracted by very early layers are, indeed, directly used by deep layers throughout the same dense block. 所有层的权重其值都分布于同一模块的很多输入中。这说明了非常早期的层提取到的特征确实直接用于同一密集模块中的更深层。

2. The weights of the transition layers also spread their weight across all layers within the preceding dense block, indicating information flow from the first to the last layers of the DenseNet through few indirections. 过渡层的权重也将其权重散布于之前的密集模块的所有层中，说明信息从DenseNet的第一层到最后一层的流动经过的迂回很少。

3. The layers within the second and third dense block consistently assign the least weight to the outputs of the transition layer (the top row of the triangles), indicating that the transition layer outputs many redundant features (with low weight on average). This is in keeping with the strong results of DenseNet-BC where exactly these outputs are compressed. 第2和第3个密集块中的层对过渡层的权重输出一直是最少的（三角形的最顶层的一行），说明过渡层输出了很多冗余特征（平均下来低权重）。这与DenseNet-BC的强结果一致，其中压缩的是同样的输出。

4. Although the final classification layer, shown on the very right, also uses weights across the entire dense block, there seems to be a concentration towards final feature-maps, suggesting that there may be some more high-level features produced late in the network. 虽然在最右边所示的最终分类层也使用了整个密集块的所有权重，似乎向最终的特征图上有一种聚集的效果，说明在网络后期可能产生了一些更高层的特征。

Figure 5: The average absolute filter weights of convolutional layers in a trained DenseNet. The color of pixel (s, l) encodes the average L1 norm (normalized by number of input feature-maps) of the weights connecting convolutional layer s to ` within a dense block. Three columns highlighted by black rectangles correspond to two transition layers and the classification layer. The first row encodes weights connected to the input layer of the dense block.

## 6. Conclusion 结论

We proposed a new convolutional network architecture, which we refer to as Dense Convolutional Network (DenseNet). It introduces direct connections between any two layers with the same feature-map size. We showed that DenseNets scale naturally to hundreds of layers, while exhibiting no optimization difficulties. In our experiments, DenseNets tend to yield consistent improvement in accuracy with growing number of parameters, without any signs of performance degradation or overfitting. Under multiple settings, it achieved state-of-the-art results across several highly competitive datasets. Moreover, DenseNets require substantially fewer parameters and less computation to achieve state-of-the-art performances. Because we adopted hyperparameter settings optimized for residual networks in our study, we believe that further gains in accuracy of DenseNets may be obtained by more detailed tuning of hyperparameters and learning rate schedules.

我们提出了一种新的卷积网络架构，称为密集卷积网络(DenseNet)。新架构在任意两层（同样特征图大小）之间都引入连接。我们证明了，DenseNets可以很自然的扩充到数百层，而没有任何优化困难。在我们的试验中，DenseNets随着参数的增多，会得到更高的准确率，而没有性能降低或过拟合的现象。在多种设置下，可以在几个数据集上都获得目前最好的结果。而且，DenseNets需要的参数数量和计算量要少很多，就可以取得目前最好的结果。因为在本文的研究中采用的超参数设置是为残差网络优化的，我们相信如果对超参数进行更细致的调节，设计新的学习速度方案，DenseNet会进一步改进准确率。

Whilst following a simple connectivity rule, DenseNets naturally integrate the properties of identity mappings, deep supervision, and diversified depth. They allow feature reuse throughout the networks and can consequently learn more compact and, according to our experiments, more accurate models. Because of their compact internal representations and reduced feature redundancy, DenseNets may be good feature extractors for various computer vision tasks that build on convolutional features, e.g., [4, 5]. We plan to study such feature transfer with DenseNets in future work.

DenseNets使用了简单的连接原则，所以很自然的有恒等连接、深度监督和多样化深度的性质。这在整个网络中都可以进行特征重用，最终可以学到更紧凑更准确的模型，我们的试验证明了这一点。由于其紧凑的内部表示，特征冗余的缩减，在很多基于卷积特征计算机视觉任务[4,5]中，DenseNets可以是很好的特征提取器。我们计划在将来研究这样的特征转移。