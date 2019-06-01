# Aggregated Residual Transformations for Deep Neural Networks

Saining Xie Kaiming He et al. UC San Diego Facebook AI Research

## Abstract 摘要

We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online.

我们提出一种简单的、高度模块化的图像分类网络架构。我们网络的构建是一种模块的重复，是用同样一种拓扑结构聚积一种变换的集合。我们的简单设计得到的是一种同种类的、多分支的架构，只有几个超参数需要设置。这种策略下，出现了一个新的维度，我们称为cardinality（变换集合的大小），除了网络的深度和宽度，这也是一种关键的因子。在ImageNet-1K数据集上，我们通过经验，表明即使是在受限的条件下，保持复杂度不变，通过增加cardinality也可以改进分类准确率。而且，在增加网络能力时，增加cardinality比增加网络深度或宽度更有效。我们的模型，称为ResNeXt，是我们提交给ILSVRC 2016分类任务的基础，我们赢得了第二名的成绩。我们在ImageNet-5K集和COCO检测集上进一步研究ResNeXt，也表明了比相应的ResNet网络结果更好。代码和模型已经开源。

## 1. Introduction 简介

Research on visual recognition is undergoing a transition from “feature engineering” to “network engineering” [25, 24, 44, 34, 36, 38, 14]. In contrast to traditional hand-designed features (e.g., SIFT [29] and HOG [5]), features learned by neural networks from large-scale data [33] require minimal human involvement during training, and can be transferred to a variety of recognition tasks [7, 10, 28]. Nevertheless, human effort has been shifted to designing better network architectures for learning representations.

视觉识别上的研究正在经历从特征工程到网络工程的变化[25,24,44,34,36,38,14]。与传统的手工设计特征相比（如SIFT[29]和HOG[5]），神经网络在训练过程中从大规模数据[33]中学到的特征需要人的参与很少，而且可以迁移到很多识别任务中[7,10,28]。尽管如此，人的努力是转移到了设计更好的网络架构，以学习表示。

Designing architectures becomes increasingly difficult with the growing number of hyper-parameters (width (Width refers to the number of channels in a layer) , filter sizes, strides, etc.), especially when there are many layers. The VGG-nets [36] exhibit a simple yet effective strategy of constructing very deep networks: stacking building blocks of the same shape. This strategy is inherited by ResNets [14] which stack modules of the same topology. This simple rule reduces the free choices of hyperparameters, and depth is exposed as an essential dimension in neural networks. Moreover, we argue that the simplicity of this rule may reduce the risk of over-adapting the hyperparameters to a specific dataset. The robustness of VGGnets and ResNets has been proven by various visual recognition tasks [7, 10, 9, 28, 31, 14] and by non-visual tasks involving speech [42, 30] and language [4, 41, 20].

随着超参数数量越来越多（宽度，指一层中的通道数量，滤波器大小，步长等），设计架构也变得越来越难，尤其是层数很多的时候。VGGNet[36]是一种构造非常深网络的简单但有效的策略：堆叠相同形状的模块。这种策略由ResNet[14]继承了，ResNet也是堆叠相同拓扑的模块。这种简单的规则降低了超参数的自由选择，深度变成了神经网络的一个关键维度。而且，我们认为，这条规则的简单性可能降低超参数对一个特定数据集的过适应性。VGGNets和ResNets的稳健性已经得到很多视觉识别任务[7,10,9,28,31,14]和非视觉任务包括语音[42,30]和语言[4,41,20]证明。

Unlike VGG-nets, the family of Inception models [38, 17, 39, 37] have demonstrated that carefully designed topologies are able to achieve compelling accuracy with low theoretical complexity. The Inception models have evolved over time [38, 39], but an important common property is a split-transform-merge strategy. In an Inception module, the input is split into a few lower-dimensional embeddings (by 1×1 convolutions), transformed by a set of specialized filters (3×3, 5×5, etc.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer (e.g., 5×5) operating on a high-dimensional embedding. The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity.

与VGGNets不同，Inception模型[38,17,39,37]已经证明了仔细设计的拓扑结构可以得到很好的准确率，而且理论上的复杂度很低。Inception模型随着时间已经演化[38,39]，但一个重要的通用属性是一个分离-变换-合并策略。在一个Inception模型中，输入分割成几个低维嵌入（通过1×1卷积），通过特定滤波器集(3×3,5×5等)变换，通过拼接进行合并。可以证明，这种架构的解空间，是单个大型层（如5×5）在一个高纬度嵌入运算解空间的严格子空间。Inception模块的分离-变换-合并行为，可以在相当低的计算复杂度上，达到大型密集层的表示能力。

Despite good accuracy, the realization of Inception models has been accompanied with a series of complicating factors — the filter numbers and sizes are tailored for each individual transformation, and the modules are customized stage-by-stage. Although careful combinations of these components yield excellent neural network recipes, it is in general unclear how to adapt the Inception architectures to new datasets/tasks, especially when there are many factors and hyper-parameters to be designed.

尽管准确率高，Inception模型的实现一直有一系列复杂的问题，滤波器数量和大小对每个变换都是定制的，每个阶段的每个模块都是定制的。虽然这些组件的精细组合得到了非常好的神经网络架构，总体上来说，Inception架构怎样改造以适应新的数据集/任务还是不清楚的，尤其是涉及到的因素很多，需要涉及的超参数很多的时候。

In this paper, we present a simple architecture which adopts VGG/ResNets’ strategy of repeating layers, while exploiting the split-transform-merge strategy in an easy, extensible way. A module in our network performs a set of transformations, each on a low-dimensional embedding, whose outputs are aggregated by summation. We pursuit a simple realization of this idea — the transformations to be aggregated are all of the same topology (e.g., Fig. 1 (right)). This design allows us to extend to any large number of transformations without specialized designs.

本文中，我们提出了一种简单的架构，采用了VGG/ResNet的重复层的策略，而且以一种简单、可拓展的方式利用了分离-变换-合并策略。我们网络中的一个模块进行一类变换，形成一个集合，每个都是在低维嵌入上进行运算，其输出通过加法聚积起来。我们追求这种想法的简单实现，要聚积的变换都是同样的拓扑结构（如，图1右）。这种设计使我们可以扩展到任意数量的变换，而不需要专门的设计。

Interestingly, under this simplified situation we show that our model has two other equivalent forms (Fig. 3). The reformulation in Fig. 3(b) appears similar to the Inception-ResNet module [37] in that it concatenates multiple paths; but our module differs from all existing Inception modules in that all our paths share the same topology and thus the number of paths can be easily isolated as a factor to be investigated. In a more succinct reformulation, our module can be reshaped by Krizhevsky et al.’s grouped convolutions [24] (Fig. 3(c)), which, however, had been developed as an engineering compromise.

有趣的是，在这种简化的情况下，我们证明了我们的模型有两个其他等价的形式（图3）。图3(b)的重组似乎与Inception-ResNet模块[37]类似，在于其拼接了多个路径；但我们的模块与所有现有的Inception模块都不一样，即我们的所有路径共享了同样的拓扑结构，所以路径数量可以很容易分离出来，进行研究。在一个足够简洁的结构重组中，我们的模块可以由Krizhevsky等的分组卷积[24]变换图3(c)的形状，虽然这只是由于工程折中设计出来的架构。

We empirically demonstrate that our aggregated transformations outperform the original ResNet module, even under the restricted condition of maintaining computational complexity and model size — e.g., Fig. 1(right) is designed to keep the FLOPs complexity and number of parameters of Fig. 1(left). We emphasize that while it is relatively easy to increase accuracy by increasing capacity (going deeper or wider), methods that increase accuracy while maintaining (or reducing) complexity are rare in the literature.

我们利用经验证明了，我们聚积的变换超过了原始的ResNet模块，即使是在受限的条件下，即保持计算复杂度和模型大小的情况下，如图1右的模型架构设计是与图1左的FLOPs复杂度和参数数量相同的。我们要强调，通过增加模型容量（更深或更宽）来提高准确率是相对容易的，而文献中在保持复杂度的情况下提升准确率是较少的。

Our method indicates that cardinality (the size of the set of transformations) is a concrete, measurable dimension that is of central importance, in addition to the dimensions of width and depth. Experiments demonstrate that increasing cardinality is a more effective way of gaining accuracy than going deeper or wider, especially when depth and width starts to give diminishing returns for existing models.

我们的方法说明，cardinality（变换集合的大小）是一种非常重要的、坚实的，可度量的维度，是宽度和深度的维度的补充。试验表明，增加cardinality是提高准确率的一种更有效的方式，比增加网络深度和宽度都要好一些，尤其是在现有的模型中，增加深度和宽度带来的受益越来越小的时候。

Our neural networks, named ResNeXt (suggesting the next dimension), outperform ResNet-101/152 [14], ResNet-200 [15], Inception-v3 [39], and Inception-ResNet-v2 [37] on the ImageNet classification dataset. In particular, a 101-layer ResNeXt is able to achieve better accuracy than ResNet-200 [15] but has only 50% complexity. Moreover, ResNeXt exhibits considerably simpler designs than all Inception models. ResNeXt was the foundation of our submission to the ILSVRC 2016 classification task, in which we secured second place. This paper further evaluates ResNeXt on a larger ImageNet-5K set and the COCO object detection dataset [27], showing consistently better accuracy than its ResNet counterparts. We expect that ResNeXt will also generalize well to other visual (and non-visual) recognition tasks.

我们的神经网络，命名为ResNeXt（表明下一个维度），在ImageNet分类数据集上超过了ResNet-101/152 [14], ResNet-200 [15], Inception-v3 [39], 和Inception-ResNet-v2 [37]。特别是，101层的ResNeXt比ResNet-200[15]准确率还要高，但复杂度只有50%.而且，ResNeXt比所有的Inception模型设计上都要更简单。ResNeXt是我们提交给ILSVRC 2016分类任务的基础，我们取得了第二名的成绩。本文进一步在更大的ImageNet-5K集和COCO目标检测数据集[27]上评估ResNeXt，表明可以一直取得比相应的ResNet模型更好的成绩。我们期待ResNeXt在其他视觉（和非视觉）识别任务中也会得到很好的泛化。

## 2. Related Work 相关的工作

**Multi-branch convolutional networks**. The Inception models [38, 17, 39, 37] are successful multi-branch architectures where each branch is carefully customized. ResNets [14] can be thought of as two-branch networks where one branch is the identity mapping. Deep neural decision forests [22] are tree-patterned multi-branch networks with learned splitting functions.

**多分支卷积网络**。Inception模型[38,17,39,37]是成功的多分支架构，其中每个分支都是仔细设计的。ResNet[14]也可以认为是两分支网络，其中一个分支为恒等映射。深度神经决策森林[22]是树型的多分支网络，有学习好的分割函数。

**Grouped convolutions**. The use of grouped convolutions dates back to the AlexNet paper [24], if not earlier. The motivation given by Krizhevsky et al. [24] is for distributing the model over two GPUs. Grouped convolutions are supported by Caffe [19], Torch [3], and other libraries, mainly for compatibility of AlexNet. To the best of our knowledge, there has been little evidence on exploiting grouped convolutions to improve accuracy. A special case of grouped convolutions is channel-wise convolutions in which the number of groups is equal to the number of channels. Channel-wise convolutions are part of the separable convolutions in [35].

**分组卷积**。分组卷积的使用可以回溯到AlexNet[24]。其动机是将模型部署在两个GPU上。Caffe[19], Torch[3]和其他库都支持分组卷积，主要是为了兼容AlexNet。据我们所知，还没有多少工作利用分组卷积来提高准确率。分组卷积的一个特例是逐通道卷积，其组数与通道数相同。逐通道的卷积是separable卷积[35]的一部分。

**Compressing convolutional networks**. Decomposition (at spatial [6, 18] and/or channel [6, 21, 16] level) is a widely adopted technique to reduce redundancy of deep convolutional networks and accelerate/compress them. Ioannou et al. [16] present a “root”-patterned network for reducing computation, and branches in the root are realized by grouped convolutions. These methods [6, 18, 21, 16] have shown elegant compromise of accuracy with lower complexity and smaller model sizes. Instead of compression, our method is an architecture that empirically shows stronger representational power.

**压缩卷积网络**。在空域分解[6,18]或通道层次分解[6,21,16]是广泛采用的技术，以降低深度卷积网络的冗余度，进行加速或压缩。Ioannou等[16]提出了一种根型网络以降低计算量，在根处的分支是由分组卷积实现的。这些方法[6,18,21,16]都在准确率与更低的复杂度和更小的模型规模之前有很好的折中。我们的方法没有压缩，但通过试验表明了有更好的表示能力。

**Ensembling**. Averaging a set of independently trained networks is an effective solution to improving accuracy [24], widely adopted in recognition competitions [33]. Veit et al. [40] interpret a single ResNet as an ensemble of shallower networks, which results from ResNet’s additive behaviors [15]. Our method harnesses additions to aggregate a set of transformations. But we argue that it is imprecise to view our method as ensembling, because the members to be aggregated are trained jointly, not independently.

**集成模型**。独立训练的若干网络经过平均，是一种有效的提高准确率的方法[24]，这在识别竞赛[33]中得到了广泛采用。Veit等[40]将ResNet解释为更浅的网络的集成，这是ResNet的加法行为的结果[15]。我们的方法利用相加来聚积一系列变换。但我们认为，将我们的方法看做集成是不准确的，因为聚积的成员是同时训练的，不是独立训练的。

## 3. Method 方法

### 3.1. Template 模板

We adopt a highly modularized design following VGG/ResNets. Our network consists of a stack of residual blocks. These blocks have the same topology, and are subject to two simple rules inspired by VGG/ResNets: (i) if producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes), and (ii) each time when the spatial map is downsampled by a factor of 2, the width of the blocks is multiplied by a factor of 2. The second rule ensures that the computational complexity, in terms of FLOPs (floating-point operations, in # of multiply-adds), is roughly the same for all blocks.

我们和VGG/ResNets类似，采用高度模块化的设计。我们的网络由若干残差模块堆叠组成。这些模块的拓扑结构都一样，而且遵守两条受VGG/ResNets启发得到的简单规则：(i)如果生成的空间图大小一样，这些模块的超参数就都一样（宽度和滤波器大小）；(ii)每次空间图进行步长为2的下采样，模块的宽度就乘以2。第二个规则确保了所有模块的计算复杂度都大致相同，以FLOPs计。

With these two rules, we only need to design a template module, and all modules in a network can be determined accordingly. So these two rules greatly narrow down the design space and allow us to focus on a few key factors. The networks constructed by these rules are in Table 1.

有了这两条规则，我们只需要设计一个模块模板，那么网络中的所有模块都可以相应确定。所以这两条规则极大的缩小了设计空间，使我们关注于一些关键因素。由这些规则构建出的网络如表1所示。

Table 1. (Left) ResNet-50. (Right) ResNeXt-50 with a 32×4d template (using the reformulation in Fig. 3(c)). Inside the brackets are the shape of a residual block, and outside the brackets is the number of stacked blocks on a stage. “C=32” suggests grouped convolutions [24] with 32 groups. The numbers of parameters and FLOPs are similar between these two models.

### 3.2. Revisiting Simple Neurons 重温简单的神经元

The simplest neurons in artificial neural networks perform inner product (weighted sum), which is the elementary transformation done by fully-connected and convolutional layers. Inner product can be thought of as a form of aggregating transformation: 人工神经网络中最简单的神经元，进行的是内积计算（加权求和），这是全连接层和卷积层进行的基础变换。内积可以认为是一种聚积变换的形式：

$$\sum_{i=1}^D w_i x_i$$(1)

where $x = [x_1, x_2, ..., x_D]$ is a D-channel input vector to the neuron and $w_i$ is a filter’s weight for the i-th channel. This operation (usually including some output nonlinearity) is referred to as a “neuron”. See Fig. 2. 其中$x = [x_1, x_2, ..., x_D]$是神经元的D维输入向量，$w_i$是滤波器第i个通道的权重。这种运算（通常包括一些输出非线性）被称为神经元。见图2。

Figure 2. A simple neuron that performs inner product.

The above operation can be recast as a combination of splitting, transforming, and aggregating. (i) Splitting: the vector x is sliced as a low-dimensional embedding, and in the above, it is a single-dimension subspace $x_i$. (ii) Transforming: the low-dimensional representation is transformed, and in the above, it is simply scaled: $w_i x_i$. (iii) Aggregating: the transformations in all embeddings are aggregated by $\sum_{i=1}^D$.

上述运算可以重新表示为分裂-变换-聚积的组合。(i)分裂：向量x被切成低维嵌入，在上面是单维度子空间$x_i$；(ii)变换：低维表示经过变换，在上面，就是简单的变化尺度$w_i x_i$；(iii)聚积：在所有嵌入中的变换通过$\sum_{i=1}^D$聚积。

### 3.3. Aggregated Transformations 聚积变换

Given the above analysis of a simple neuron, we consider replacing the elementary transformation ($w_i x_i$) with a more generic function, which in itself can also be a network. In contrast to “Network-in-Network” [26] that turns out to increase the dimension of depth, we show that our “Network-in-Neuron” expands along a new dimension.

有了上面关于简单神经元的分析，我们考虑将初级变换($w_i x_i$)替换成更一般的函数，其本身也可以是一个网络。Network-in-Network[26]增加了深度这个维度，与之形成对比，我们的Network-in-Network扩张了一个新的维度。

Formally, we present aggregated transformations as: 正式的，我们在下式中给出聚积变换：

$$F(x) = \sum_{i=1}^C τ_i (x)$$(2)

where $τ_i (x)$ can be an arbitrary function. Analogous to a simple neuron, $τ_i$ should project x into an (optionally low-dimensional) embedding and then transform it. 其中$τ_i (x)$可以是任意函数。与简单的神经元类比，$τ_i$应当将x投影到（可能是低维）嵌入上，然后对其进行变换。

In Eqn.(2), C is the size of the set of transformations to be aggregated. We refer to C as cardinality [2]. In Eqn.(2) C is in a position similar to D in Eqn.(1), but C need not equal D and can be an arbitrary number. While the dimension of width is related to the number of simple transformations (inner product), we argue that the dimension of cardinality controls the number of more complex transformations. We show by experiments that cardinality is an essential dimension and can be more effective than the dimensions of width and depth.

在式(2)中，C是需要聚积的变换集的大小。我们称C为cardinality[2]。在式(2)中，C是与式(1)中的D类似的作用，但C不需要等于D，可以是任意数值。宽度的维度与简单变换的数量相关（内积），我们认为cardinality的维度控制的是更复杂变换的数量。我们通过试验表明，cardinality是一种非常重要的维度，比宽度和深度更有效率。

In this paper, we consider a simple way of designing the transformation functions: all $τ_i$’s have the same topology. This extends the VGG-style strategy of repeating layers of the same shape, which is helpful for isolating a few factors and extending to any large number of transformations. We set the individual transformation $τ_i$ to be the bottleneck-shaped architecture [14], as illustrated in Fig. 1 (right). In this case, the first 1×1 layer in each $τ_i$ produces the low-dimensional embedding.

本文中，我们考虑设计变换函数的一种简单方法：所有$τ_i$都有相同的拓扑结构。这拓展了VGG类型的策略，即重复相同形状的层，这有助于分离一些因子，拓展到任意大小的变换。我们设单个变换$τ_i$为瓶颈形状的架构[14]，如图1右所示。在这种情况下，在每个$τ_i$中的第一个1×1卷积层生成低维嵌入。

The aggregated transformation in Eqn.(2) serves as the residual function [14] (Fig. 1 right): 式(2)中的聚积变换即残差函数[14]（图1右）：

$$y = x + \sum_{i=1}^C τ_i (x)$$(3)

where y is the output. 其中y是输出。

**Relation to Inception-ResNet**. Some tensor manipulations show that the module in Fig. 1(right) (also shown in Fig. 3(a)) is equivalent to Fig. 3(b). 3 Fig. 3(b) appears similar to the Inception-ResNet [37] block in that it involves branching and concatenating in the residual function. But unlike all Inception or Inception-ResNet modules, we share the same topology among the multiple paths. Our module requires minimal extra effort designing each path.

**与Inception-ResNet的关系**。一些张量处理表明，图1右（即图3a）与图3(b)是等价的。图3(b)似乎与Inception-ResNet[37]类似，因为都有残差函数的分支和拼接。但与所有Inception或Inception-ResNet模块不同，我们在所有路径上拓扑都一样。我们的模块在每条路径上的额外设计努力是最小的。

**Relation to Grouped Convolutions**. The above module becomes more succinct using the notation of grouped convolutions [24]. This reformulation is illustrated in Fig. 3(c). All the low-dimensional embeddings (the first 1×1 layers) can be replaced by a single, wider layer (e.g., 1×1, 128-d in Fig 3(c)). Splitting is essentially done by the grouped convolutional layer when it divides its input channels into groups. The grouped convolutional layer in Fig. 3(c) performs 32 groups of convolutions whose input and output channels are 4-dimensional. The grouped convolutional layer concatenates them as the outputs of the layer. The block in Fig. 3(c) looks like the original bottleneck residual block in Fig. 1(left), except that Fig. 3(c) is a wider but sparsely connected module.

**与分组卷积的关系**。上述的模块在使用分组卷积[24]的表示后更为简明。这种重新阐述如图3(c)所示。所有低维嵌入（第一个1×1层）可以替换成一整个更宽的层（如图3c中的1×1,128-d）。分割是有分组卷积层完成的，将其输入通道分成若干组。图3(c)中的分组卷积层进行32组卷积，其输入和输出通道都是4维的。分组卷积层将其拼接起来，作为本层的输出。图3(c)中的模块看起来很像图1左中的原始瓶颈残差模块，不过图3(c)是更宽而且稀疏连接的模块。

Figure 3. Equivalent building blocks of ResNeXt. (a): Aggregated residual transformations, the same as Fig. 1 right. (b): A block equivalent to (a), implemented as early concatenation. (c): A block equivalent to (a,b), implemented as grouped convolutions [24]. Notations in bold text highlight the reformulation changes. A layer is denoted as (# input channels, filter size, # output channels).

We note that the reformulations produce nontrivial topologies only when the block has depth ≥3. If the block has depth = 2 (e.g., the basic block in [14]), the reformulations lead to trivially a wide, dense module. See the illustration in Fig. 4.

我们注意到，只有在模块深度≥3的时候，重组才会生成不一样的拓扑。如果模块深度=2（如[14]中的基础模块），那么重组带来的只是宽的、密集的模块。如图4中描述所示。

Figure 4. (Left): Aggregating transformations of depth = 2. (Right): An equivalent block, which is trivially wider.

**Discussion**. We note that although we present reformulations that exhibit concatenation (Fig. 3(b)) or grouped convolutions (Fig. 3(c)), such reformulations are not always applicable for the general form of Eqn.(3), e.g., if the transformation $τ_i$ takes arbitrary forms and are heterogenous. We choose to use homogenous forms in this paper because they are simpler and extensible. Under this simplified case, grouped convolutions in the form of Fig. 3(c) are helpful for easing implementation.

**讨论**。我们注意到，虽然我们提出的重组，表现出来的是拼接（图3b）或分组卷积（图3c），这种重组对于式(3)的一般形式并不可用，如，若变换$τ_i$形式任意，并且各不相同。我们在本文中选择同质的形式，因为更简单而且可扩展。在这种简化的情况下，图3(c)式的分组卷积实现起来更为容易。

### 3.4. Model Capacity 模型容纳能力

Our experiments in the next section will show that our models improve accuracy when maintaining the model complexity and number of parameters. This is not only interesting in practice, but more importantly, the complexity and number of parameters represent inherent capacity of models and thus are often investigated as fundamental properties of deep networks [8].

我们下一节的试验将会表明，我们的模型可以在保持模型复杂度和参数数量的情况下，提高准确率。模型复杂度和参数数量代表着模型的内在容纳能力，所以是深度网络的最基础的性质[8]。

When we evaluate different cardinalities C while preserving complexity, we want to minimize the modification of other hyper-parameters. We choose to adjust the width of the bottleneck (e.g., 4-d in Fig 1(right)), because it can be isolated from the input and output of the block. This strategy introduces no change to other hyper-parameters (depth or input/output width of blocks), so is helpful for us to focus on the impact of cardinality.

当我们在保持复杂度的同时，研究不同的cardinalities C，我们希望对其他超参数的改动尽量的小。我们选择调整瓶颈结构的宽度（如，图1右的4d），因为可以孤立在模块的输入和输出之外。这种策略没有引入其他超参数的变化（模块输入/输出宽度或深度），所以有助于我们聚焦于cardinality的影响。

In Fig. 1(left), the original ResNet bottleneck block [14] has 256 · 64 + 3 · 3 · 64 · 64 + 64 · 256 ≈ 70k parameters and proportional FLOPs (on the same feature map size). With bottleneck width d, our template in Fig. 1(right) has: 在图1左中，原始ResNet瓶颈模块[14]有约70k参数，和成比例的FLOPs（在同样的特征图大小下）。在瓶颈宽度为d时，我们图1右的模板的参数数量为：

$$C · (256 · d + 3 · 3 · d · d + d · 256)$$(4)

parameters and proportional FLOPs. When C = 32 and d = 4, Eqn.(4) ≈ 70k. Table 2 shows the relationship between cardinality C and bottleneck width d. 以及成比例的FLOPs。当C=32、d=4时，式(4)约等于70k。表2给出了cardinality C和瓶颈宽度d的关系。

Table 2. Relations between cardinality and width (for the template of conv2), with roughly preserved complexity on a residual block. The number of parameters is ∼70k for the template of conv2. The number of FLOPs is ∼0.22 billion (# params×56×56 for conv2). 宽度和cardinality之间的关系（对于模板中的conv2），在大致相同的残差模块复杂度下。对于conv2的模板来说，参数数量大约是70k。FLOPs数量大约是0.22 billion（对于conv2来说是，参数数量×56×56）。

cardinality C | 1 | 2 | 4 | 8 | 32
--- | --- | --- | --- | --- | ---
width of bottleneck d | 64 | 40 | 24 | 14 | 4
width of group conv | 64 | 80 | 96 | 112 | 128

Because we adopt the two rules in Sec. 3.1, the above approximate equality is valid between a ResNet bottleneck block and our ResNeXt on all stages (except for the subsampling layers where the feature maps size changes). Table 1 compares the original ResNet-50 and our ResNeXt-50 that is of similar capacity. We note that the complexity can only be preserved approximately, but the difference of the complexity is minor and does not bias our results.

因为我们采用3.1节中的两个原则，上述的近似等式在ResNet瓶颈模块和我们的ResNeXt所有阶段中都是有效的（除了下采样层，其中特征图大小有了变化）。表1比较了原始ResNet-50和我们的ResNeXt-50，其容纳能力类似。我们注意到，复杂度只能近似保持一致，但复杂度的差异很小，并不能改变我们的结果。

## 4. Implementation details 实现细节

Our implementation follows [14] and the publicly available code of fb.resnet.torch [11]. On the ImageNet dataset, the input image is 224×224 randomly cropped from a resized image using the scale and aspect ratio augmentation of [38] implemented by [11]. The shortcuts are identity connections except for those increasing dimensions which are projections (type B in [14]). Downsampling of conv3, 4, and 5 is done by stride-2 convolutions in the 3×3 layer of the first block in each stage, as suggested in [11]. We use SGD with a mini-batch size of 256 on 8 GPUs (32 per GPU). The weight decay is 0.0001 and the momentum is 0.9. We start from a learning rate of 0.1, and divide it by 10 for three times using the schedule in [11]. We adopt the weight initialization of [13]. In all ablation comparisons, we evaluate the error on the single 224×224 center crop from an image whose shorter side is 256.

我们的实现与[14]和开源的fb.resnet.torch [11]类似。在ImageNet数据集中，原始图像使用[11]中实现的尺度和纵横比扩充[38]改变大小，随机剪切为224×224作为输入图像。捷径是恒等映射，除了那些增加维度的投影运算（[14]中的类型B）。conv3,4,5的下采样是每个阶段中第一个模块中的3×3卷积层步长为2，在[11]中也是这样。我们使用SGD，mini-batch大小为256，在8个GPU上运算（每个GPU上32幅图像）。权重衰减为0.0001，动量为0.9。初始学习速率为0.1，三次除以10，方案为[11]。我们采用[13]中的权重初始化方案。在所有的分离试验比较中，我们在单个224×224中间剪切块中评估错误率，图像的短边为256像素。

Our models are realized by the form of Fig. 3(c). We perform batch normalization (BN) [17] right after the convolutions in Fig. 3(c). ReLU is performed right after each BN, expect for the output of the block where ReLU is performed after the adding to the shortcut, following [14].

我们的模型以图3(c)的形式实现。我们在图3(c)的卷积层后进行BN[17]。每次BN后都进行ReLU，除了模块的输出与捷径相加后进行ReLU的，这与[14]中一致。

We note that the three forms in Fig. 3 are strictly equivalent, when BN and ReLU are appropriately addressed as mentioned above. We have trained all three forms and obtained the same results. We choose to implement by Fig. 3(c) because it is more succinct and faster than the other two forms.

我们注意，当BN和ReLU如上述进行合理的放置后，图3中的三种形式是严格等价的。我们训练了所有三种形式，得到了相同的结果。我们选择实现图3(c)，因为其形式更简洁，实现起来更快。

## 5. Experiments 试验

### 5.1. Experiments on ImageNet-1K

We conduct ablation experiments on the 1000-class ImageNet classification task [33]. We follow [14] to construct 50-layer and 101-layer residual networks. We simply replace all blocks in ResNet-50/101 with our blocks. 我们在1000类的ImageNet分类任务[33]中进行分离试验。我们和[14]中一样，构建50层和101层的残差网络。我们只是将ResNet-50/101中的所有模块替换成我们的模块。

**Notations**. Because we adopt the two rules in Sec. 3.1, it is sufficient for us to refer to an architecture by the template. For example, Table 1 shows a ResNeXt-50 constructed by a template with cardinality = 32 and bottleneck width = 4d (Fig. 3). This network is denoted as ResNeXt-50 (32×4d) for simplicity. We note that the input/output width of the template is fixed as 256-d (Fig. 3), and all widths are doubled each time when the feature map is subsampled (see Table 1).

****。因为我们采用了3.1中的两种规则，用模板来指代架构已经足够了。比如，表1所示的ResNeXt-50是由cardinality=32和瓶颈宽度=4d的模板构建的（图3）。网络简化表示为ResNeXt-50(32×4d)。我们注意到，模板的输入/输出宽度固定为256-d（图3），每当特征图下采样时，网络宽度就加倍（见表1）。

**Cardinality vs. Width**. We first evaluate the trade-off between cardinality C and bottleneck width, under preserved complexity as listed in Table 2. Table 3 shows the results and Fig. 5 shows the curves of error vs. epochs. Comparing with ResNet-50 (Table 3 top and Fig. 5 left), the 32×4d ResNeXt-50 has a validation error of 22.2%, which is 1.7% lower than the ResNet baseline’s 23.9%. With cardinality C increasing from 1 to 32 while keeping complexity, the error rate keeps reducing. Furthermore, the 32×4d ResNeXt also has a much lower training error than the ResNet counterpart, suggesting that the gains are not from regularization but from stronger representations.

**Cardinality与宽度的对比**。我们首先评估，在表2列出的保持复杂度的情况下，cardinality C和瓶颈宽度的折中。表3给出了结果，图5给出了错误率vs轮数的曲线。与ResNet-50相比（表3和图5左），32×4d ResNeXt-50的验证错误率为22.2%，比ResNet基准的23.9%，低了1.7%。随着cardinality C从1增加到32，复杂度维持不变，错误率持续降低。而且，32×4d ResNeXt的训练误差也比相应的ResNet模型低的多，说明这个提升不是来自于正则化，而是更强的表示能力。

Table 3. Ablation experiments on ImageNet-1K. (Top): ResNet-50 with preserved complexity (∼4.1 billion FLOPs); (Bottom): ResNet-101 with preserved complexity (∼7.8 billion FLOPs). The error rate is evaluated on the single crop of 224×224 pixels.

 | | setting | top-1 error(%)
--- | --- | ---
ResNet-50 | 1×64d | 23.9
ResNeXt-50 | 2×40d | 23.0
ResNeXt-50 | 4×24d | 22.6
ResNeXt-50 | 8×14d | 22.3
ResNeXt-50 | 32×4d | 22.2
ResNet-101 | 1×64d | 22.0
ResNeXt-101 | 2×40d | 21.7
ResNeXt-101 | 4×24d | 21.4
ResNeXt-101 | 8×14d | 21.3
ResNeXt-101 | 32×4d | 21.2

Figure 5. Training curves on ImageNet-1K. (Left): ResNet/ResNeXt-50 with preserved complexity (∼4.1 billion FLOPs, ∼25 million parameters); (Right): ResNet/ResNeXt-101 with preserved complexity (∼7.8 billion FLOPs, ∼44 million parameters).

Similar trends are observed in the case of ResNet-101 (Fig. 5 right, Table 3 bottom), where the 32×4d ResNeXt-101 outperforms the ResNet-101 counterpart by 0.8%. Although this improvement of validation error is smaller than that of the 50-layer case, the improvement of training error is still big (20% for ResNet-101 and 16% for 32×4d ResNeXt-101, Fig. 5 right). In fact, more training data will enlarge the gap of validation error, as we show on an ImageNet-5K set in the next subsection.

在ResNet-101中也可以观察到类似的趋势（图5右，表3下面），其中32×4d ResNeXt-101超过了对应的ResNet-101 0.8%。虽然验证错误率的这个改进比50层的情况更小一些，但训练误差的改进仍然很大（图5右，ResNet-101为20%，32×4d ResNeXt-101为16%）。实际上，更多的训练数据会增大验证数据的差距，我们在下一节的ImageNet-5K集上会看到这个结论。

Table 3 also suggests that with complexity preserved, increasing cardinality at the price of reducing width starts to show saturating accuracy when the bottleneck width is small. We argue that it is not worthwhile to keep reducing width in such a trade-off. So we adopt a bottleneck width no smaller than 4d in the following.

表3还说明，在维持复杂度的情况下，增加cardinality、降低宽度，在瓶颈宽度很小时会呈现准确率饱和的情况。我们认为在这种情况下，不应当一直降低宽度，所以我们在下面都采用瓶颈宽度不小于4d。

**Increasing Cardinality vs. Deeper/Wider**. Next we investigate increasing complexity by increasing cardinality C or increasing depth or width. The following comparison can also be viewed as with reference to 2× FLOPs of the ResNet-101 baseline. We compare the following variants that have ∼15 billion FLOPs. (i) Going deeper to 200 layers. We adopt the ResNet-200 [15] implemented in [11]. (ii) Going wider by increasing the bottleneck width. (iii) Increasing cardinality by doubling C.

**增加cardinality与更深/更宽的对比**。下一步我们研究增加复杂度的方法，即增加cardinality C或增加深度或宽度。下面的比较也可以看作是ResNet-101基准的2倍FLOPs。我们比较下面三种变体，都有大约15 billion FLOPs。(i)深度增加到200层，我们采用[15]中的ResNet-200的[11]实现形式；(ii)增加瓶颈宽度，从而变得更宽；(iii)增加cardinality，使C加倍。

Table 4 shows that increasing complexity by 2× consistently reduces error vs. the ResNet-101 baseline (22.0%). But the improvement is small when going deeper (ResNet-200, by 0.3%) or wider (wider ResNet-101, by 0.7%).

表4所示的是，复杂度增加2倍，与ResNet-101基准(22.0%)相比，肯定会降低错误率。但通过增加深度(ResNet-200, 0.3%)或变得更宽(wider ResNet-101, 0.7%)其改进更小。

On the contrary, increasing cardinality C shows much better results than going deeper or wider. The 2×64d ResNeXt-101 (i.e., doubling C on 1×64d ResNet-101 baseline and keeping the width) reduces the top-1 error by 1.3% to 20.7%. The 64×4d ResNeXt-101 (i.e., doubling C on 32×4d ResNeXt-101 and keeping the width) reduces the top-1 error to 20.4%.

增加cardinality C可以得到更好的结果。2×64d ResNeXt-101（即在基准1×64d ResNet-101上对C进行加倍，保持其宽度），其top-1错误率降低了1.3%，到了20.7%。64×4d ResNeXt-101（即将32×4d ResNeXt-101的C加倍，保持其宽度）将top-1错误率降低到了20.4%。

We also note that 32×4d ResNeXt-101 (21.2%) performs better than the deeper ResNet-200 and the wider ResNet-101, even though it has only ∼50% complexity. This again shows that cardinality is a more effective dimension than the dimensions of depth and width.

我们还注意到，32×4d ResNeXt-101 (21.2%)比更深的ResNet-200和更宽的ResNet-101效果更好，即使其复杂度只有大约后两者的50%。这再一次说明了，cardinality比深度和宽度的维度更加有效。

Table 4. Comparisons on ImageNet-1K when the number of FLOPs is increased to 2× of ResNet-101’s. The error rate is evaluated on the single crop of 224×224 pixels. The highlighted factors are the factors that increase complexity.

| | setting | top-1 err(%) | top-5 err(%)
--- | --- | --- | ---
1×complexity reference: |
ResNet-101 | 1×64d | 22.0 | 6.0
ResNeXt-101 | 32×4d | 21.2 | 5.6
2×complexity models follow: |
ResNet-200 [15] | 1×64d | 21.7 | 5.8
ResNet-101, wider | 1×100d | 21.3 | 5.7
ResNeXt-101 | 2×64d | 20.7 | 5.5
ResNeXt-101 | 64×4d | 20.4 | 5.3

**Residual connections**. The following table shows the effects of the residual (shortcut) connections: 残差连接，下表给出了残差（捷径）连接的效果：

 | | setting | w/residual | w/o residual
--- | --- | --- | ---
ResNet-50 | 1×64d | 23.9 | 31.2
ResNeXt-50 | 32×4d | 22.2 | 26.1

Removing shortcuts from the ResNeXt-50 increases the error by 3.9 points to 26.1%. Removing shortcuts from its ResNet-50 counterpart is much worse (31.2%). These comparisons suggest that the residual connections are helpful for optimization, whereas aggregated transformations are stronger representations, as shown by the fact that they perform consistently better than their counterparts with or without residual connections.

从ResNeXt-50中去掉了捷径连接，错误率提高了3.9%，到了26.1%。从ResNet-50中去掉捷径连接则更差(31.2%)。这些比较说明，残差连接对于优化是很有帮组的，而聚积变换则是更强的表示，因为其表现比对应的模型一直要好，不管有没有残差连接。

**Performance**. For simplicity we use Torch’s built-in grouped convolution implementation, without special optimization. We note that this implementation was brute-force and not parallelization-friendly. On 8 GPUs of NVIDIA M40, training 32×4d ResNeXt-101 in Table 3 takes 0.95s per mini-batch, vs. 0.70s of ResNet-101 baseline that has similar FLOPs. We argue that this is a reasonable overhead. We expect carefully engineered lower-level implementation (e.g., in CUDA) will reduce this overhead. We also expect that the inference time on CPUs will present less overhead. Training the 2×complexity model (64×4d ResNeXt-101) takes 1.7s per mini-batch and 10 days total on 8 GPUs.

**性能**。简单起见，我们使用Torch的内建分组卷积实现，没有特别的优化。我们注意到，这种实现是暴力实现的，对并行计算很不友好。在8块NVidia M40 GPU上，训练表3中的32×4d ResNeXt-101每个mini-batch耗时0.95s，基准ResNet-101有类似的FLOPs，耗时0.70s。我们认为这是合理的时间开销。我们期待，仔细工程设计的底层实现（如，在CUDA中）会降低这个时间开销。我们还期待在CPU上的推理时间耗时会少一些。训练两倍复杂度的模型(64×4d ResNeXt-101)耗时1.7s每mini-batch，在8个GPU上耗时10天。

**Comparisons with state-of-the-art results**. Table 5 shows more results of single-crop testing on the ImageNet validation set. In addition to testing a 224×224 crop, we also evaluate a 320×320 crop following [15]. Our results compare favorably with ResNet, Inception-v3/v4, and Inception-ResNet-v2, achieving a single-crop top-5 error rate of 4.4%. In addition, our architecture design is much simpler than all Inception models, and requires considerably fewer hyper-parameters to be set by hand.

**与目前最好的结果的比较**。表5给出了在ImageNet验证集上的更多单剪切块测试结果。除了测试224×224剪切块，我们还像[15]一样评估了320×320剪切块。我们的结果比ResNet, Inception-v3/v4, 和Inception-ResNet-v2都更好一些，得到的单剪切块top-5错误率达到了4.4%。另外，我们的架构设计比所有的Inception模型都简单的多，需要手工设置的超参数也少的多。

ResNeXt is the foundation of our entries to the ILSVRC 2016 classification task, in which we achieved 2nd place. We note that many models (including ours) start to get saturated on this dataset after using multi-scale and/or multicrop testing. We had a single-model top-1/top-5 error rates of 17.7%/3.7% using the multi-scale dense testing in [14], on par with Inception-ResNet-v2’s single-model results of 17.8%/3.7% that adopts multi-scale, multi-crop testing. We had an ensemble result of 3.03% top-5 error on the test set, on par with the winner’s 2.99% and Inception-v4/Inception-ResNet-v2’s 3.08% [37].

ResNeXt是我们提交给ILSVRC 2016分类比赛任务的基础，我们赢得了第二名的位置。我们注意到，很多模型（包括我们的）在使用了多尺度或多剪切块测试时会在这个数据集上饱和。我们的单模型top-1/top-5错误率分别为17.7%/3.7%，使用的是[14]中的多尺度密集测试，与Inception-ResNet-v2的单模型结果17.8%/3.7%类似，它采用了多尺度、多剪切块测试。我们的集成模型在测试集上的结果为3.03% top-5错误率，与获胜者的2.99%和Inception-v4/Inception-ResNet-v2的3.08%接近。

Table 5. State-of-the-art models on the ImageNet-1K validation set (single-crop testing). The test size of ResNet/ResNeXt is 224×224 and 320×320 as in [15] and of the Inception models is 299×299.

### 5.2. Experiments on ImageNet-5K

The performance on ImageNet-1K appears to saturate. But we argue that this is not because of the capability of the models but because of the complexity of the dataset. Next we evaluate our models on a larger ImageNet subset that has 5000 categories.

在ImageNet-1K上的性能看起来饱和了。但我们认为，这不是因为模型的能力原因，而是数据集的复杂度的原因。下面我们在更大的ImageNet子集（包含5000个类别）中评估我们的模型。

Our 5K dataset is a subset of the full ImageNet-22K set [33]. The 5000 categories consist of the original ImageNet-1K categories and additional 4000 categories that have the largest number of images in the full ImageNet set. The 5K set has 6.8 million images, about 5× of the 1K set. There is no official train/val split available, so we opt to evaluate on the original ImageNet-1K validation set. On this 1K-class val set, the models can be evaluated as a 5K-way classification task (all labels predicted to be the other 4K classes are automatically erroneous) or as a 1K-way classification task (softmax is applied only on the 1K classes) at test time.

我们的5K数据集是完整的ImageNet-22K集[33]的子集。这5K类别包含了ImageNet-1K的原始类别，和另外的4000类，这4000类在完整的ImageNet集中所含图像最多。这个5K集有6.8 million图像，是1K集的5倍多。没有可用的官方train/val分割，所以我们选择在原始的ImageNet-1K验证集上进行评估。在这个1K类的验证集上，模型作为5K路的分类任务进行评估（预测所有标签属于其他4K类的自动为错误），或在测试时作为1K路分类任务（只在1K类上进行softmax）。

The implementation details are the same as in Sec. 4. The 5K-training models are all trained from scratch, and are trained for the same number of mini-batches as the 1K-training models (so 1/5× epochs). Table 6 and Fig. 6 show the comparisons under preserved complexity. ResNeXt-50 reduces the 5K-way top-1 error by 3.2% comparing with ResNet-50, and ResNetXt-101 reduces the 5K-way top-1 error by 2.3% comparing with ResNet-101. Similar gaps are observed on the 1K-way error. These demonstrate the stronger representational power of ResNeXt.

实现细节与第4节中相同。5K训练模型都是从头开始训练，都使用与1K训练模型相同的mini-batch数来进行训练（轮数为1/5）。表6和图6是在相同复杂度下的对比。ResNeXt-50的5K路的top-1错误率与ResNet相比降低了3.2%，ResNeXt-101与ResNet-101相比，5K路top-1错误率降低了2.3%。在1K路错误率上也观察到了类似的结果。这证明了ResNeXt更强的表示能力。

Moreover, we find that the models trained on the 5K set (with 1K-way error 22.2%/5.7% in Table 6) perform competitively comparing with those trained on the 1K set (21.2%/5.6% in Table 3), evaluated on the same 1K-way classification task on the validation set. This result is achieved without increasing the training time (due to the same number of mini-batches) and without fine-tuning. We argue that this is a promising result, given that the training task of classifying 5K categories is a more challenging one.

而且，我们发现在5K集上训练出来的模型（1K路错误率22.2%/5.7%，表6），与在1K集上训练出来的结果相比很不错（21.2%/5.6%，表3），这是在同样的1K路分类任务在验证集上评估的结果。这个结果的取得，没有增加训练时间（因为使用的mini-batch数量一样），没有精调。我们认为，这是一个很有希望的结果，因为训练分类5K类的任务更有挑战性。

Figure 6. ImageNet-5K experiments. Models are trained on the 5K set and evaluated on the original 1K validation set, plotted as a 1K-way classification task. ResNeXt and its ResNet counterpart have similar complexity.

Table 6. Error (%) on ImageNet-5K. The models are trained on ImageNet-5K and tested on the ImageNet-1K val set, treated as a 5K-way classification task or a 1K-way classification task at test time. ResNeXt and its ResNet counterpart have similar complexity. The error is evaluated on the single crop of 224×224 pixels.

### 5.3. Experiments on CIFAR

We conduct more experiments on CIFAR-10 and 100 datasets [23]. We use the architectures as in [14] and replace the basic residual block by the bottleneck template of [1×1,64;3×3,64;1×1,256]. Our networks start with a single 3×3 conv layer, followed by 3 stages each having 3 residual blocks, and end with average pooling and a fully-connected classifier (total 29-layer deep), following [14]. We adopt the same translation and flipping data augmentation as [14]. Implementation details are in the appendix.

我们在CIFAR-10和CIFAR-100数据集[23]上进行更多的试验。我们使用[14]中的架构，将基本残差模块替换为瓶颈模板[1×1,64;3×3,64;1×1,256]。我们的网络以单个3×3卷积层开始，然后是3个阶段，每个阶段有3个残差模块，最后以平均池化和全连接分类器结束（共29层深），和[14]中一样。我们采用[14]中相同的平移和翻转数据扩充方案。实现细节如附录所示。

We compare two cases of increasing complexity based on the above baseline: (i) increase cardinality and fix all widths, or (ii) increase width of the bottleneck and fix cardinality = 1. We train and evaluate a series of networks under these changes. Fig. 7 shows the comparisons of test error rates vs. model sizes. We find that increasing cardinality is more effective than increasing width, consistent to what we have observed on ImageNet-1K. Table 7 shows the results and model sizes, comparing with the Wide ResNet [43] which is the best published record. Our model with a similar model size (34.4M) shows results better than Wide ResNet. Our larger method achieves 3.58% test error (average of 10 runs) on CIFAR-10 and 17.31% on CIFAR-100. To the best of our knowledge, these are the state-of-the-art results (with similar data augmentation) in the literature including unpublished technical reports.

我们基于上面的基准，比较了两种增加复杂度的情况：(i)固定所有宽度，增加cardinality，或(ii)固定cardinality=1，增加瓶颈的宽度。我们训练并评估一系列网络和这些变化。图7是测试错误率与模型大小的比较。我们发现，增加cardinality比增加宽度的效率要高的多，这与我们在ImageNet-1K上观察到的结论一致。表7给出了结果和模型大小，与Wide ResNet[43]的比较，这是已经发表的最好结果。我们的模型在类似的大小下(34.4M)得到了比Wide ResNet更好的结果。我们更大的模型在CIFAR-10上得到了3.58%测试错误率（10次运行的平均），在CIFAR-100上17.31%。据我们所知，这是目前最好的结果（在类似的数据扩充下）。

Figure 7. Test error vs. model size on CIFAR-10. The results are computed with 10 runs, shown with standard error bars. The labels show the settings of the templates.

Table 7. Test error (%) and model size on CIFAR. Our results are the average of 10 runs.

| | #params | CIFAR-10 | CIFAR-100
--- | --- | --- | ---
Wide ResNet[43] | 36.5M | 4.17 | 20.50
ResNeXt-29, 8×64d | 34.4M | 3.65 | 17.77
ResNeXt-29, 16×64d | 68.1M | 3.58 | 17.31

### 5.4. Experiments on COCO object detection 在COCO目标检测上的试验

Next we evaluate the generalizability on the COCO object detection set [27]. We train the models on the 80k training set plus a 35k val subset and evaluate on a 5k val subset (called minival), following [1]. We evaluate the COCO-style Average Precision (AP) as well as AP@IoU=0.5 [27]. We adopt the basic Faster R-CNN [32] and follow [14] to plug ResNet/ResNeXt into it. The models are pre-trained on ImageNet-1K and fine-tuned on the detection set. Implementation details are in the appendix.

下面我们评估在COCO目标检测集[27]上的泛化性能。我们在80k训练集和35k验证子集上训练模型，在5k验证子集上进行评估（称为minival），和[1]一样。我们评估COCO式的AP，以及AP@IoU=0.5 [27]。我们采用基本Faster R-CNN[32]，和[14]一样，将ResNet/ResNeXt插入到其中。模型在ImageNet-1K上进行预训练，并在检测集上进行精调。实现细节见附录。

Table 8 shows the comparisons. On the 50-layer baseline, ResNeXt improves AP@0.5 by 2.1% and AP by 1.0%, without increasing complexity. ResNeXt shows smaller improvements on the 101-layer baseline. We conjecture that more training data will lead to a larger gap, as observed on the ImageNet-5K set.

表8给出了比较结果。在50层的基准上，ResNeXt将AP@0.5改进了2.1%，AP改进了1.0%，没有增加复杂度。ResNeXt在101层的基准上的改进更小一些。我们推测，更多的训练数据可能可以得到更大的改进，就像在ImageNet-5K上的结果一样。

It is also worth noting that recently ResNeXt has been adopted in Mask R-CNN [12] that achieves state-of-the-art results on COCO instance segmentation and object detection tasks. 值得一提的是，最近ResNeXt已经在Mask R-CNN[12]得到采用，在COCO实例分割和目标检测任务中得到了目前最好的结果。

### A. Implementation Details: CIFAR

### B. Implementation Details: Object Detection