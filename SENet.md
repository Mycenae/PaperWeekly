# Squeeze-and-Excitation Networks

Jie Hu et al. Chinese Academy of Sciences, Momenta, VGG

## Abstract 摘要

The central building block of convolutional neural networks (CNNs) is the convolution operator, which enables networks to construct informative features by fusing both spatial and channel-wise information within local receptive fields at each layer. A broad range of prior research has investigated the spatial component of this relationship, seeking to strengthen the representational power of a CNN by enhancing the quality of spatial encodings throughout its feature hierarchy. In this work, we focus instead on the channel relationship and propose a novel architectural unit, which we term the “Squeeze-and-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. We show that these blocks can be stacked together to form SENet architectures that generalise extremely effectively across different datasets. We further demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at minimal additional computational cost. Squeeze-and-Excitation Networks formed the foundation of our ILSVRC 2017 classification submission which won first place and reduced the top-5 error to 2.251%, surpassing the winning entry of 2016 by a relative improvement of ∼25%. Models and code are available at https://github.com/hujie-frank/SENet.

卷积神经网络(CNNs)的中心模块是卷积算子，这使网络可以通过在每层中的局部感受野融合空间和逐通道的信息，构建信息丰富的特征。之前的很多研究调查的是这种关系的空间组件，通过在整个特征层级关系中增强空间编码质量，以增强表示能力。在本文中，我们关注的是通道间的关系，提出了一种新的框架单元，称为挤压激励(SE)单元，对通道间的依赖关系进行建模，以自适应的重新校准逐通道的特征响应。我们证明了，这种模块可以堆叠在一起形成SENet架构，在不同的数据集之间可以极其高效的泛化。我们进一步证明了，SE模块可以显著的提升现有的最好的CNN的性能，而增加的计算量是非常小的。SENet是我们提交到ILSVRC 2017分类比赛的工作的基础，我们得到了第一名，将top-5错误降低到了2.251%，比2016年的获胜者的性能相对提高了大约25%。模型和代码都已开源。

**Index Terms**—Squeeze-and-Excitation, Image classification, Convolutional Neural Network.

## 1 Introduction 引言

Convolutional neural networks (CNNs) have proven to be useful models for tackling a wide range of visual tasks [1]–[4]. At each convolutional layer in the network, a collection of filters expresses neighbourhood spatial connectivity patterns along input channels—fusing spatial and channel-wise information together within local receptive fields. By interleaving a series of convolutional layers with non-linear activation functions and downsampling operators, CNNs are able to produce robust representations that capture hierarchical patterns and attain global theoretical receptive fields. Recent research has shown that these representations can be strengthened by integrating learning mechanisms into the network that help capture spatial correlations between features. One such approach, popularised by the Inception family of architectures [5], [6], incorporates multi-scale processes into network modules to achieve improved performance. Further work has sought to better model spatial dependencies [7], [8] and incorporate spatial attention into the structure of the network [9].

卷积神经网络(CNNs)已经证明了在处理很多视觉任务中都很有用[1-4]。在网络中的每个卷积层，滤波器集合沿着输入通道表示邻域空间连接性模式，即将空间和通道信息在局部感受野中融合到一起。通过一系列卷积层与非线性激活函数、下采样算子交叉到一起，CNNs可以生成稳健的表示，捕捉到层次化的模型，得到全局的理论感受野。最近的研究已经表明，这些表示可以通过将学习机制整合到网络中得到增强，这可以帮助捕捉特征间的空间关系。一种这样的方法把Inception架构族[5,6]与多尺度过程整合到一起，得到了改进的性能。进一步的工作寻求对空间依赖关系进行更好的建模[7,8]，将空间注意力整合进网络结构中[9]。

In this paper, we investigate a different aspect of network design - the relationship between channels. We introduce a new architectural unit, which we term the Squeeze-and-Excitation (SE) block, with the goal of improving the quality of representations produced by a network by explicitly modelling the interdependencies between the channels of its convolutional features. To this end, we propose a mechanism that allows the network to perform feature recalibration, through which it can learn to use global information to selectively emphasise informative features and suppress less useful ones.

在本文中，我们研究了网络设计的不同方面-通道间的关系。我们引入了一种新的架构单元，称为SE模块，目标是通过对卷积特征通道间的依赖关系进行建模，来改进网络生成表示的质量。为此，我们提出一种机制，使网络可以进行特征重新校准，这样可以学习使用全局信息有选择性的强调信息多的特征，抑制不太有用的。

The structure of the SE building block is depicted in Fig. 1. For any given transformation $F_{tr}: X → U, X ∈ R^{H'×W'×C'}, U ∈ R^{H×W×C}$, (e.g. a convolution), we can construct a corresponding SE block to perform feature recalibration. The features U are first passed through a squeeze operation, which produces a channel descriptor by aggregating feature maps across their spatial dimensions (H × W). The function of this descriptor is to produce an embedding of the global distribution of channel-wise feature responses, allowing information from the global receptive field of the network to be used by all its layers. The aggregation is followed by an excitation operation, which takes the form of a simple self-gating mechanism that takes the embedding as input and produces a collection of per-channel modulation weights. These weights are applied to the feature maps U to generate the output of the SE block which can be fed directly into subsequent layers of the network.

SE模块的结构如图1所示。对任意给定的变换$F_{tr}: X → U, X ∈ R^{H'×W'×C'}, U ∈ R^{H×W×C}$，（如一个卷积），我们可以构建一个对应的SE模块，进行特征重新校准。特征U首先经过一个挤压操作，通过把特征图在整个空间维度上累加(H × W)，生成一个通道描述子。这个描述子的作用是产生逐通道特征响应的全局分布的嵌入，允许所有层都可以使用全局感受野的信息。这个累加之后是一个激励操作，其形式是一个简单的自我门机制，以前面的嵌入为输入，生成每个通道的模块权重集合。这些权重应用于特征图U，生成SE模块的输出，然后可以直接送入网络的后续层。

It is possible to construct an SE network (SENet) by simply stacking a collection of SE blocks. Moreover, these SE blocks can also be used as a drop-in replacement for the original block at a range of depths in the network architecture (Sec. 6.4). While the template for the building block is generic, the role it performs at different depths differs throughout the network. In earlier layers, it excites informative features in a class-agnostic manner, strengthening the shared low-level representations. In later layers, the SE blocks become increasingly specialised, and respond to different inputs in a highly class-specific manner (Sec. 7.2). As a consequence, the benefits of the feature recalibration performed by SE blocks can be accumulated through the network.

简单的将一些SE模块堆叠在一起，就可能构建一个SE网络(SENet)。而且，这些SE模块可以在网络架构一定深度内随时替代原始模块（见6.4节）。模块的模板是通用性的，但在网络中不同深度所扮演的角色是不一样的。在早期网络层中，其作用是以类别无关的方式对信息量大的特征进行激励，增强共享的低级表示。在后期的网络层中，SE模块变得越来越专用，对不同输入的响应跟类别高度相关（见7.2节）。结果是，SE模块带来的特征重校准的好处可以在网络中得到累积。

The design and development of new CNN architectures is a difficult engineering task, typically requiring the selection of many new hyperparameters and layer configurations. By contrast, the structure of the SE block is simple and can be used directly in existing state-of-the-art architectures by replacing components with their SE counterparts, where the performance can be effectively enhanced. SE blocks are also computationally lightweight and impose only a slight increase in model complexity and computational burden.

新CNN架构的设计和发展是一项非常困难的工程任务，典型的是需要选择很多新的超参数和层的配置。比较起来，SE模块的结构是简单的，可以直接用于现有的最好架构中，将部件替换为相应的SE替代品，其性能可以得到有效的改善。SE模块在计算上消耗很少，在模型复杂度和计算负担上只有略微的增加。

To provide evidence for these claims, in Sec. 4 we develop several SENets and conduct an extensive evaluation on the ImageNet 2012 dataset [10]. We also present results beyond ImageNet that indicate that the benefits of our approach are not restricted to a specific dataset or task. By making use of SENets, we ranked first in the ILSVRC 2017 classification competition. Our best model ensemble achieves a 2.251% top-5 error on the test set. This represents roughly a 25% relative improvement when compared to the winner entry of the previous year (top-5 error of 2.991%).

在第4部分中我们提出了几个SENets，在ImageNet 2012数据集上[10]进行了广泛的试验评估，以证明我们的结论。我们还给出了ImageNet之外的结果，表明我们方法的好处没有局限于特定的任务或数据集。使用了SENets，我们在ILSVRC 2017分类竞赛上得到了第一名的成绩。我们最好的模型集成在测试集上取得了2.251%的top-5错误率。与前一年的获胜者相比(top-5 error 2.991%)，我们的相对改进幅度有大约25%。

## 2 Related Work 相关工作

**Deeper architectures**. VGGNets [11] and Inception models [5] showed that increasing the depth of a network could significantly increase the quality of representations that it was capable of learning. By regulating the distribution of the inputs to each layer, Batch Normalization (BN) [6] added stability to the learning process in deep networks and produced smoother optimisation surfaces [12]. Building on these works, ResNets demonstrated that it was possible to learn considerably deeper and stronger networks through the use of identity-based skip connections [13], [14]. Highway networks [15] introduced a gating mechanism to regulate the flow of information along shortcut connections. Following these works, there have been further reformulations of the connections between network layers [16], [17], which show promising improvements to the learning and representational properties of deep networks.

**更深的架构**。VGGNets[11]和Inception模型[5]表明，增加网络深度可以显著提升能够学习到的表示的质量。通过对输入的分布进行规范化，批归一化(BN)[6]为深度网络的学习过程增加了稳定性，得到了更平滑的优化表面[12]。在这些工作的基础上，ResNet证明了，使用恒等跳跃连接[13,14]，可以学习到深很多也强很多的网络。Highway网络[15]引入了门机制，来沿着捷径连接规范信息的流动。在这些工作之后，[16,17]进一步重组了网络层之间的连接，对深度网络的学习和表示性质都有了不少改进。

An alternative, but closely related line of research has focused on methods to improve the functional form of the computational elements contained within a network. Grouped convolutions have proven to be a popular approach for increasing the cardinality of learned transformations [18], [19]. More flexible compositions of operators can be achieved with multi-branch convolutions [5], [6], [20], [21], which can be viewed as a natural extension of the grouping operator. In prior work, cross-channel correlations are typically mapped as new combinations of features, either independently of spatial structure [22], [23] or jointly by using standard convolutional filters [24] with 1 × 1 convolutions. Much of this research has concentrated on the objective of reducing model and computational complexity, reflecting an assumption that channel relationships can be formulated as a composition of instance-agnostic functions with local receptive fields. In contrast, we claim that providing the unit with a mechanism to explicitly model dynamic, non-linear dependencies between channels using global information can ease the learning process, and significantly enhance the representational power of the network.

另一条紧密相关的研究线聚焦在改进网络中的计算元素的函数形式。分组卷积已经被证明是可以增加学习到的变换的基的很好的方法[18,19]。更灵活的算子组合可以通过多分支卷积[5,6,20,21]得到，这可以视作分组算子的自然延伸。在之前的工作中，跨通道的相关性一般映射为特征的新组合，要么与空间结构无关[22,23]，或通过使用1 × 1的标准卷积滤波器[24]来一起实现。这个研究很多都集中在降低模型和计算的复杂度的目标上，反应了一个假设，即通道间的关系可以表示为与实例无关的、局部感受野的函数组合。对比起来，我们则认为，设计一种机制来使用全局信息，显式的对通道间的动态、非线性依赖关系进行建模，并给出这种功能的单元模块，可以使学习过程变得更容易，显著增强网络的表示能力。

**Algorithmic Architecture Search**. Alongside the works described above, there is also a rich history of research that aims to forgo manual architecture design and instead seeks to learn the structure of the network automatically. Much of the early work in this domain was conducted in the neuro-evolution community, which established methods for searching across network topologies with evolutionary methods [25], [26]. While often computationally demanding, evolutionary search has had notable successes which include finding good memory cells for sequence models [27], [28] and learning sophisticated architectures for large-scale image classification [29]–[31]. With the goal of reducing the computational burden of these methods, efficient alternatives to this approach have been proposed based on Lamarckian inheritance [32] and differentiable architecture search [33].

**算法架构搜索**。与上述的工作同时，还有很多研究的目标是，放弃手工设计架构，而是寻求自动学习网络结构。这个领域的很多早期工作是在神经演化团体中进行的，确定了使用演化方法[25,26]来搜索网络拓扑的方法。虽然计算量非常大，演化搜索在为序列模型寻找好的记忆单元上非常成功[27,28]，还有为大规模图像分类任务学习复杂架构[29-31]上也非常成功。一些研究者在降低这些方法的计算负担上也有一些工作，如[32]提出了基于Lamarchian inheritance的替代方法，[33]提出了基于可微分架构搜索的方法。

By formulating architecture search as hyperparameter optimisation, random search [34] and other more sophisticated model-based optimisation techniques [35], [36] can also be used to tackle the problem. Topology selection as a path through a fabric of possible designs [37] and direct architecture prediction [38], [39] have been proposed as additional viable architecture search tools. Particularly strong results have been achieved with techniques from reinforcement learning [40]–[44]. SE blocks can be used as atomic building blocks for these search algorithms, and were demonstrated to be highly effective in this capacity in concurrent work [45].

随机搜索[34]和其他更复杂的基于模型的优化技术[35,36]将架构搜索表示为超参数优化，也可以用于解决这个问题。[37]将拓扑选择作为可能的设计的结构中的路径，[38,39]将其作为直接架构预测问题，这也都是另外的可行架构搜索工具。使用强化学习[40-44]技术取得了非常好的结果。SE模块可以用于这些搜索算法的原子模块，也证明了在现在的工作中非常有效[45]。

**Attention and gating mechanisms**. Attention can be interpreted as a means of biasing the allocation of available computational resources towards the most informative components of a signal [46]–[51]. Attention mechanisms have demonstrated their utility across many tasks including sequence learning [52], [53], localisation and understanding in images [9], [54], image captioning [55], [56] and lip reading [57]. In these applications, it can be incorporated as an operator following one or more layers representing higher-level abstractions for adaptation between modalities. Concurrent work provides an interesting study into the combined use of spatial and channel attention around convolutional [58] and bottleneck units [59]. Wang et al. [60] introduced a powerful trunk-and-mask attention mechanism based on hourglass modules [8] that is inserted between the intermediate stages of deep residual networks. By contrast, our proposed SE block comprises a lightweight gating mechanism which focuses on enhancing the representational power of the network by modelling channel-wise relationships in a computationally efficient manner.

**注意力和门机制**。注意力可以解释为将可用的计算资源向最有信息量的信号部分进行倾斜的一种方法[46-51]。注意力机制证明了在很多任务中都非常有用，包括序列学习[52,53]，图像中的定位和理解[9,54]，添加图像说明文字[55,56]和唇语阅读[57]。在这些应用中，可以作为算子置于表示较高层的抽象的一层或多层后面，作为模式之间的适应。同时有工作进行了很有趣的研究，在卷积模块[58]和瓶颈模块[59]周围把空间注意力和通道注意力结合了起来。Wang等[60]引入了一种很强的基于hourglass模块[8]的trunk-and-mask注意力机制，插入在深度残差网络的中间阶段之间。对比起来，我们提出的SE模块由一种轻量级的门机制组成，通过对通道之间的关系建模，聚焦在增强网络的表示能力，计算量上的增加也非常少。

## 3 Squeeze-and-Excitation Blocks

The Squeeze-and-Excitation block is a computational unit which can be constructed for any given transformation $U = F_{tr}(X), X ∈ R^{H'×W'×C'}, U ∈ R^{H×W×C}$。For simplicity, in the notation that follows we take $F_{tr}$ to be a convolutional operator. Let $V = [v_1, v_2, . . . , v_C]$ denote the learned set of filter kernels, where $v_c$ refers to the parameters of the c-th filter. We can then write the outputs of $F_{tr}$ as $U = [u_1, u_2, . . ., u_C]$, where

SE模块是对任何给定的变换都可以构造的计算模块，$U = F_{tr}(X), X ∈ R^{H'×W'×C'}, U ∈ R^{H×W×C}$。为简化起见，在后面的表示中，我们用$F_{tr}$表示卷积算子。令$V = [v_1, v_2, . . . , v_C]$表示学习到的滤波器核的集合，其中$v_c$表示第c个滤波器的参数。我们然后可以将$F_{tr}$的输出记为$U = [u_1, u_2, . . ., u_C]$，其中

$$u_c = v_c * X = \sum_{s=1}^{C'} v_c^s * x^s$$(1)

Here ∗ denotes convolution, $v_c = [v_c^1, v_c^2, . . ., v_c^{C'}]$, $X =  [x_1, x_2, . . ., x_C]$ (to simplify the notation, bias terms are omitted) and $v_c^s$ is a 2D spatial kernel representing a single channel of $v_c$ that acts on the corresponding channel of X. Since the output is produced by a summation through all channels, channel dependencies are implicitly embedded in $v_c$, but are entangled with the local spatial correlation captured by the filters. As a consequence, the channel relationships modelled by convolution are inherently local. Since our goal is to ensure that the network is able to increase its sensitivity to informative features so that they can be exploited by subsequent transformations most effectively, we would like to provide it with access to global information. We propose to achieve this by explicitly modelling channel interdependencies to recalibrate filter responses in two steps, squeeze and excitation, before they are fed into the next transformation, described next. A diagram illustrating the structure of an SE block is shown in Fig. 1.

这里*表示卷积，$v_c = [v_c^1, v_c^2, . . ., v_c^{C'}]$, $X =  [x_1, x_2, . . ., x_C]$（为简化表示，忽略了偏置项），$v_c^s$是一个2D空域核，表示$v_c$的一个通道，作用于X对应的通道上。由于输出是所有通道之和，通道依赖关系隐式的嵌入在$v_c$中，但与滤波器捕捉到的局部空间相关纠缠在了一起。结果是，卷积建模的通道间关系天生就是局部的。由于我们的目标是确保网络提高对信息丰富的特征的敏感性，这样才能由后续变换更高效的利用，我们要使其可以访问利用全局信息。我们提出，通过显式的对通道相互关系进行建模，以重新标定滤波器的相应，达到这个目标；分为两步，squeeze和excitation，然后送入后续的变换中，详见下面。图1给出了SE模块结构的图示。

### 3.1 Squeeze: Global Information Embedding 全局信息嵌入

In order to tackle the issue of exploiting channel dependencies, we first consider the signal to each channel in the output features. Each of the learned filters operates with a local receptive field and consequently each unit of the transformation output U is unable to exploit contextual information outside of this region.

为处理利用通道依赖关系的问题，我们首先考虑输出特征中每个通道的信号。每个学习到的滤波器都是在局部感受野上进行计算，结果是每个变换的单元的输出U，是不能利用这个区域之外的上下文信息。

To mitigate this problem, we propose to squeeze global spatial information into a channel descriptor. This is achieved by using global average pooling to generate channel-wise statistics. Formally, a statistic $z ∈ R^C$ is generated by shrinking U through its spatial dimensions H × W, such that the c-th element of z is calculated by: 为减缓这个问题，我们提出将全局空域信息挤压进一个通道描述子中。这是通过全局平均pooling来实现的，生成的是逐通道的统计。正式来说，统计量$z ∈ R^C$是通过将U在空域维度H × W上收缩得到的，这样z的第c个元素计算如下：

$$z_c = F_{sq} (u_c) = \frac {1}{H × W} \sum_{i=1}^H \sum_{j=1}^W u_c(i,j)$$(2)

Discussion. The output of the transformation U can be interpreted as a collection of the local descriptors whose statistics are expressive for the whole image. Exploiting such information is prevalent in prior feature engineering work [61]–[63]. We opt for the simplest aggregation technique, global average pooling, noting that more sophisticated strategies could be employed here as well.

讨论。变换U的输出可以解释为局部描述子的集合，其统计信息可以表达整个图像。利用这些信息在之前的特征工程工作[61-63]中非常流行。我们选择用最简单的累加技术，全局平均pooling，注意这里也可以采用更复杂的策略。

### 3.2 Excitation: Adaptive Recalibration 自适应重校准

To make use of the information aggregated in the squeeze operation, we follow it with a second operation which aims to fully capture channel-wise dependencies. To fulfil this objective, the function must meet two criteria: first, it must be flexible (in particular, it must be capable of learning a nonlinear interaction between channels) and second, it must learn a non-mutually-exclusive relationship since we would like to ensure that multiple channels are allowed to be emphasised (rather than enforcing a one-hot activation). To meet these criteria, we opt to employ a simple gating mechanism with a sigmoid activation:

为使用squeeze操作中累加得到的信息，我们之后进行另一个计算，其目标是完全捕捉逐通道的依赖关系。为满足这个目标，这个函数必须满足两条准则：首先，必须是灵活的（特别的，必须能学习通道间的非线性交互）；第二，必须能学习一个非互相排斥的关系，因为我们想要确保同时强调多个通道（而不是进行one-hot激活）。为满足这些准则，我们选择采用一个简单的带有sigmoid激活的门机制：

$$s = F_{ex} (z, W) = σ(g(z, W)) = σ(W_2 δ(W_1 z))$$(3)

where δ refers to the ReLU [64] function, $W_1 ∈ R^{\frac{C}{r}×C}$ and $W_2 ∈ R^{C× \frac{C}{r}}$. To limit model complexity and aid generalisation, we parameterise the gating mechanism by forming a bottleneck with two fully connected (FC) layers around the non-linearity, i.e. a dimensionality-reduction layer with parameters $W_1$ and reduction ratio r (this parameter choice is discussed in Sec. 6.1), a ReLU and then a dimensionality-increasing layer with parameters $W_2$. The final output of the block is obtained by rescaling the transformation output U with the activations:

其中δ为ReLU[64]函数，$W_1 ∈ R^{\frac{C}{r}×C}$ and $W_2 ∈ R^{C× \frac{C}{r}}$。为限制模型复杂度并协助泛化，我们将门机制参数化，形成非线性附近的两个全连接层(FC)组成的瓶颈结构，即，一个维度压缩层，参数为$W_1$压缩率为r（这个参数选择在6.1中讨论），一个ReLU层，然后是一个维度增加层，参数为$W_2$。模块最后的输出是用下面的激活来改变变换输出U的幅度：

$$\tilde x_c = F_{scale} (u_c, s_c) = s_c · u_c$$(4)

where $\tilde X = [\tilde x_1 , \tilde x_2 , . . . , \tilde x_C]$ and $F_{scale} (u_c, s_c)$ refers to where X channel-wise multiplication between the scalar $s_c$ and the feature map $u_c ∈ R^{H×W}$. 其中$\tilde X = [\tilde x_1 , \tilde x_2 , . . . , \tilde x_C]$和$F_{scale} (u_c, s_c)$指的是X逐通道与标量$s_c$和特征图$u_c ∈ R^{H×W}$相乘的结果。

Discussion. The excitation operator maps the input-specific descriptor z to a set of channel specific weights. In this regard, SE blocks intrinsically introduce dynamics conditioned on the input, helping to boost feature discriminability.

讨论。激励算子将特定输入的描述子z映射成通道特定的权重集合。在这一点上，SE模块从本质上讲引入了依赖于输入的动态，帮助提升特征区分性。

### 3.3 Instantiations 实例化

The SE block can be integrated into standard architectures such as VGGNet [11] by insertion after the non-linearity following each convolution. Moreover, the flexibility of the SE block means that it can be directly applied to transformations beyond standard convolutions. To illustrate this point, we develop SENets by incorporating SE blocks into several examples of more complex architectures, described next.

SE模块可以整合进标准架构中，如VGGNet[11]，方法是在每个卷积层后的非线性后插入。而且，SE模块的灵活性意味着，在标准卷积之外的变换上，也可以直接应用。为描述这一点，我们将SE模块整合进几个更复杂的的架构例子，下面详述。

We first consider the construction of SE blocks for Inception networks [5]. Here, we simply take the transformation $F_{tr}$ to be an entire Inception module (see Fig. 2) and by making this change for each such module in the architecture, we obtain an SE-Inception network. SE blocks can also be used directly with residual networks (Fig. 3 depicts the schema of an SE-ResNet module). Here, the SE block transformation $F_{tr}$ is taken to be the non-identity branch of a residual module. Squeeze and Excitation both act before summation with the identity branch. Further variants that integrate SE blocks with ResNeXt [19], Inception-ResNet [21], MobileNet [65] and ShuffleNet [66] can be constructed by following similar schemes (Sec. 5.1). For concrete examples of SENet architectures, a detailed description of SE-ResNet-50 and SE-ResNeXt-50 is given in Table 1.

我们首先考虑将SE模块与Inception网络[5]整合到一起。这里，我们简单的将$F_{tr}$认为是整个Inception模块（见图2），将架构中的每个模块都进行这样的改变，我们得到了SE-Inception网络。SE模块也可以直接用于残差网络（图3就是SE-ResNet模块的示意图）。这里，SE模块变换$F_{tr}$被当作是残差模块的非恒等分支。Squeeze和Excitation都在与恒等分支相加之前进行。更多的变体，如将SE模块整合进ResNeXt[19]，Inception-ResNet[21]，MobileNet[65]和ShuffleNet[66]可以遵循类似的方法（见5.1节）。对于SENet架构的具体例子，表1给出了SE-ResNet-50和SE-ResNeXt-50的详细描述。

One consequence of the flexible nature of the SE block is that there are several viable ways in which it could be integrated into these architectures. Therefore, to assess sensitivity to the integration strategy used to incorporate SE blocks into a network architecture, we also provide ablation experiments exploring different designs for block inclusion in Sec. 6.5.

SE模块灵活本质的结果是，有几种可行的方法将其整合进这些架构中。因此，为得到各种不同的将SE模块整合进网络架构的策略的敏感性，我们在6.5节中进行了分离试验，探索各种不同的包含模块的设计。

Table 1 (Left) ResNet-50. (Middle) SE-ResNet-50. (Right) SE-ResNeXt-50 with a 32 × 4d template. The shapes and operations with specific parameter settings of a residual building block are listed inside the brackets and the number of stacked blocks in a stage is presented outside. The inner brackets following by fc indicates the output dimension of the two fully connected layers in an SE module.

## 4 Model and computational complexity 模型复杂度和计算复杂度

For the proposed SE block design to be of practical use, it must offer a good trade-off between improved performance and increased model complexity. We set the reduction ratio r (introduced in Sec. 3.2) to 16 in all experiments, except where stated otherwise (an ablation study of this design decision is provided in Sec. 6.1). To illustrate the computational burden associated with the module, we consider a comparison between ResNet-50 and SE-ResNet-50 as an example. ResNet-50 requires ∼3.86 GFLOPs in a single forward pass for a 224 × 224 pixel input image. Each SE block makes use of a global average pooling operation in the squeeze phase and two small fully connected layers in the excitation phase, followed by an inexpensive channel-wise scaling operation. In aggregate, SE-ResNet-50 requires ∼3.87 GFLOPs, corresponding to a 0.26% relative increase over the original ResNet-50. In exchange for this slight additional computational burden, the accuracy of SE-ResNet-50 surpasses that of ResNet-50 and indeed, approaches that of a deeper ResNet-101 network requiring ∼7.58 GFLOPs (Table 2).

对于提出的SE模块设计，要具有实际用处，就必须在改进的性能和增加的模型复杂度间给出一个均衡。我们设定3.2节中提出的缩减率r为16，除了另有叙述，所有试验都使用这个设置（在6.1节中有这种设计决定的分离研究）。为描述这个模块相关的计算量，我们考虑比较一下ResNet-50和SE-ResNet-50，作为例子。ResNet-50的一个前向过程，对于224 × 224输入图像，需要大约3.86 GFLOPs。每个SE模块都在squeeze阶段使用一个全局平均pooling运算，在excitation阶段使用两个全连接层，随后是两个计算量不大的逐通道变化幅度的运算。累加起来，SE-ResNet-50需要大约3.87 GFLOPs，与原始ResNet-50比较起来，增加了大约0.26%的计算量。SE-ResNet-50的准确率超过了ResNet-50，达到了ResNet-101的水平，而其计算量为大约7.58 GFLOPs（见表2）。

In practical terms, a single pass forwards and backwards through ResNet-50 takes 190 ms, compared to 209 ms for SE-ResNet-50 with a training minibatch of 256 images (both timings are performed on a server with 8 NVIDIA Titan X GPUs). We suggest that this represents a reasonable runtime overhead, which may be further reduced as global pooling and small inner-product operations receive further optimisation in popular GPU libraries. Due to its importance for embedded device applications, we further benchmark CPU inference time for each model: for a 224 × 224 pixel input image, ResNet-50 takes 164 ms in comparison to 167 ms for SE-ResNet-50. We believe that the small additional computational cost incurred by the SE block is justified by its contribution to model performance.

在实际中，ResNet-50的一次前向过程和反向过程耗时190ms，SE-ResNet-50则为209ms，训练minibatch大小为256（两个计时都是在8块NVidia Titan X GPUs的服务器上进行的）。我们认为，这是合理的运行耗费，由于全局pooling和小内积核运算在常用的GPU库中有进一步的优化，所以可以进一步降低运行时间。我们还进一步测试了每个模型的CPU推理时间，因为这对于嵌入式设备应用非常重要：对于224 × 224的输入图像，ResNet-50推理时间为164ms，SE-ResNet-50则为167ms。我们相信SE模块带来的很小的计算量增加是值得的，因为模型的性能也有增加。

We next consider the additional parameters introduced by the proposed SE block. These additional parameters result solely from the two fully-connected layers of the gating mechanism and therefore constitute a small fraction of the total network capacity. Concretely, the total number of additional parameters introduced by the proposed approach is given by:

下面我们考虑SE模块带来的额外参数。这些额外的参数完全是由门机制的两个全连接层带来的，是整个网络容量的一小部分。具体的，我们提出的方法引入的额外参数总量为：

$$\frac {2}{r} \sum_{s=1}^S N_s · C_s^2$$(5)

where r denotes the reduction ratio, S refers to the number of stages (a stage refers to the collection of blocks operating on feature maps of a common spatial dimension), $C_s$ denotes the dimension of the output channels and $N_s$ denotes the number of repeated blocks for stage s. SE-ResNet-50 introduces ∼2.5 million additional parameters beyond the ∼25 million parameters required by ResNet-50, corresponding to a ∼10% increase. In practice, the majority of these parameters come from the final stage of the network, where the excitation operation is performed across the greatest number of channels. However, we found that this comparatively costly final stage of SE blocks could be removed at only a small cost in performance ( <0.1% top- 5 error on ImageNet) reducing the relative parameter increase to ∼4%, which may prove useful in cases where parameter usage is a key consideration (see Sec. 7.2 for further discussion).

其中r表示缩减率，S表示阶段数量（一个阶段是指在共同的空域维度的特征图上操作的模块集合），$C_s$表示输出通道的维度，$N_s$表示在第s阶段上重复模块的数量。SE-ResNet-50引入了大约2.5 million额外参数，ResNet本文需要大约25 million参数，这对应着增加了大约10%。实际中，这些参数多数来自网络的最终阶段，其中excitation操作在最多数量的通道上进行操作的。但是，我们发现这种相对耗费更多的最终阶段的SE模块是可以被移除的，移除后性能降低非常小(<0.1% top-5 error on ImageNet)，使得增加的参数降低到大约4%，这在参数数量非常关键的情况中可能会很有用（见7.2节的进一步讨论）。

Table 2 Single-crop error rates (%) on the ImageNet validation set and complexity comparisons. The original column refers to the results reported in the original papers. To enable a fair comparison, we re-train the baseline models and report the scores in the re-implementation column. The SENet column refers to the corresponding architectures in which SE blocks have been added. The numbers in brackets denote the performance improvement over the re-implemented baselines. † indicates that the model has been evaluated on the non-blacklisted subset of the validation set (this is discussed in more detail in [21]), which may slightly improve results. VGG-16 and SE-VGG-16 are trained with batch normalization.

## 5 Experiments 试验

In this section, we conduct experiments to investigate the effectiveness of SE blocks across a range of tasks, datasets and model architectures. 本节中，我们进行试验研究SE模块在一系列任务、数据集和模型架构上的有效性。

### 5.1 Image Classification 图像分类

To evaluate the influence of SE blocks, we first perform experiments on the ImageNet 2012 dataset [10] which comprises 1.28 million training images and 50K validation images from 1000 different classes. We train networks on the training set and report the top- 1 and top- 5 error on the validation set.

为评估SE模块的影响，我们首先在ImageNet 2012数据集[10]上进行试验，数据集包含1.28 million训练图像，50k验证图像，1000个不同的类别。我们在训练集上训练网络，在验证集上给出top-1和top-5错误率。

Each original network architecture and its corresponding SE counterpart are trained with identical optimisation schemes. We follow standard practices and perform data augmentation with random cropping [5] to a size of 224 × 224 pixels (or 299 × 299 for Inception-ResNet-v2 [21] and SE-Inception-ResNet-v2) and perform random horizontal flipping. Each input image is normalised through mean RGB-channel subtraction. We adopt the data balancing strategy described in [67] for minibatch sampling. All models are trained on our distributed learning system ROCS which is designed to handle efficient parallel training of large networks. Optimisation is performed using synchronous SGD with momentum 0.9 and a minibatch size of 1024. The initial learning rate is set to 0.6 and decreased by a factor of 10 every 30 epochs. All models are trained for 100 epochs from scratch, using the weight initialisation strategy described in [68].

每个原始网络架构及其对应的SE架构都用相同的优化方案进行训练。我们也进行标准步骤，对数据进行扩充，包括随机裁减[5]为224 × 224像素的图像块（对于Inception-ResNet-v2 [21]和SE-Inception-ResNet-v2则为299 × 299），及随机水平翻转。每个输入图像都通过减去RGB通道均值进行归一化。我们采用[67]中描述的数据均衡策略进行minibatch取样。所有的模型都在分布式学习系统中进行训练，可以进行大型网络的高效并行训练。优化都是用同步SGD进行的，动量0.9，minibatch大小为1024。初始学习率为0.6，每30轮学习率除以10。所有模型都是从头训练100轮，使用的权重初始化策略与[68]中一样。

When evaluating the models we apply centre-cropping so that 224 × 224 pixels are cropped from each image, after its shorter edge is first resized to 256 (299 × 299 from each image whose shorter edge is first resized to 352 for Inception-ResNet-v2 and SE-Inception-ResNet-v2).

评估模型时，我们使用的是中心裁减，首先将图像的短边长度改变为256，这样从每幅图像中裁减出来的是224 × 224像素（对于Inception-ResNet-v2和SE-Inception-ResNet-v2，其短边首先改变为352像素，然后剪切出299 × 299大小的图像）。

**Network depth**. We begin by comparing SE-ResNet against ResNet architectures with different depths and report the results in Table 2. We observe that SE blocks consistently improve performance across different depths with an extremely small increase in computational complexity. Remarkably, SE-ResNet-50 achieves a single-crop top-5 validation error of 6.62%, exceeding ResNet-50 (7.48%) by 0.86% and approaching the performance achieved by the much deeper ResNet-101 network (6.52% top-5 error) with only half of the total computational burden (3.87 GFLOPs vs. 7.58 GFLOPs). This pattern is repeated at greater depth, where SE-ResNet-101 (6.07% top-5 error) not only matches, but outperforms the deeper ResNet-152 network (6.34% top-5 error) by 0.27% . While it should be noted that the SE blocks themselves add depth, they do so in an extremely computationally efficient manner and yield good returns even at the point at which extending the depth of the base architecture achieves diminishing returns. Moreover, we see that the gains are consistent across a range of different network depths, suggesting that the improvements induced by SE blocks may be complementary to those obtained by simply increasing the depth of the base architecture.

**网络深度**。我们将不同深度的ResNet架构与SE-ResNet进行比较，并将结果放在表2中。我们观察到，SE模块在不同深度上一直都有改进效果，计算量的增加则非常小。令人印象深刻的是，SE-ResNet-50得到的单剪切块top-5验证错误率为6.62%，超过了ResNet-50 (7.48%)很多，达到了深的多的ResNet-101网络的结果(6.52% top-5 error)，而其计算量则只有其一半(3.87 GFLOPs vs. 7.58 GFLOPs)。这种模式在更深的情况下也有，如SE-ResNet-101 (top-5 error 6.07%)超过了更深的ResNet-152网络(top-5 error 6.34%)。应该注意到，SE模块本文就增加了深度，但增加的计算量则极其的小，而且得到了非常好的回报，在这种情况下其他的增加深度的方法都已经几乎得不到回报了。而且，我们看到，在不同的网络深度中都会得到类似的改进，说明SE模块带来的改进，与增加网络深度带来的改进，是互补的。

**Integration with modern architectures**. We next study the effect of integrating SE blocks with two further state-of-the-art architectures, Inception-ResNet-v2 [21] and ResNeXt (using the setting of 32 × 4d) [19], both of which introduce additional computational building blocks into the base network. We construct SENet equivalents of these networks, SE-Inception-ResNet-v2 and SE-ResNeXt (the configuration of SE-ResNeXt-50 is given in Table 1) and report results in Table 2. As with the previous experiments, we observe significant performance improvements induced by the introduction of SE blocks into both architectures. In particular, SE-ResNeXt-50 has a top-5 error of 5.49% which is superior to both its direct counterpart ResNeXt-50 (5.90% top-5 error) as well as the deeper ResNeXt-101 (5.57% top-5 error), a model which has almost twice the total number of parameters and computational overhead. We note a slight difference in performance between our re-implementation of Inception-ResNet-v2 and the result reported in [21]. However, we observe a similar trend with regard to the effect of SE blocks, finding that SE-Inception-ResNet-v2 (4.79% top-5 error) outperforms our reimplemented Inception-ResNet-v2 baseline (5.21% top-5 error) by 0.42% (a relative improvement of 8.1%) as well as the reported result in [21]. The training curves for the baseline architectures ResNet-50, ResNet-152, ResNeXt-50 and BN-Inception, and their respective SE counterparts are depicted in Fig. 4. We observe that SE blocks yield a steady improvement throughout the optimisation procedure. Moreover, this trend is consistent across each of the families of state-of-the-art architectures considered as baselines.

**与现代架构的整合**。下面我们研究将SE模块与两种目前最好的架构的整合，即Inception-ResNet-v2[21]和ResNeXt（使用32 × 4d的设置）[19]，两种架构都在基础网络中引入了额外的计算模块。我们构建这些网络对应的SENet，即SE-Inception-ResNet-v2和SE-ResNeXt（SE-ResNeXt-50的配置如表1所示），并将结果在表2中给出。与之前的试验一样，我们观察到，SE模块的引入，在两种架构中都带来了明显的性能改进。特别是，SE-ResNeXt-50的top-5 error为5.49%，超过了对应的ResNeXt-50(5.90% top-5 error)及其更深的ResNeXt-101(5.57% top-5 error)，ResNeXt的模型参数数量和计算复杂度都几乎是两倍。我们注意到，我们重新实现的Inception-ResNet-v2与[21]中的结果略有不一样。但是，SE模块的影响是类似的，SE-Inception-ResNet-v2 (4.79% top-5 error)超过了我们重新实现的Inception-ResNet-v2基准(5.21% top-5 error)，相对改进达到8.1%，也超过了[21]中给出的结果。ResNet-50, ResNet-152, ResNeXt-50 和 BN-Inception的训练曲线，及其对应的SE模型如图4所示。我们观察到SE模块在整个优化过程都有稳定的改进。而且，这种趋势在目前最好的架构中也是一样的。

Fig. 4. Training curve comparisons for different baseline architectures with their SENet counterparts on ImageNet. SENets exhibit improved optimisation characteristics and produce consistent gains in performance which are sustained throughout the training process. Note that (d) uses a different y-axis scaling (to allow visualisation of early training accuracies).

We also assess the effect of SE blocks when operating on non-residual networks by conducting experiments with the VGG-16 [11] and BN-Inception architecture [6]. To facilitate the training of VGG-16 from scratch, we add Batch Normalization layers after each convolution. As with the previous models, we use identical training schemes for both VGG-16 and SE-VGG-16. The results of the comparison are shown in Table 2. Similarly to the results reported for the residual baseline architectures, we observe that SE blocks bring improvements in performance.

我们还评估了SE模块在非残差网络中的效果，在VGG-16[11]和BN-Inception架构[6]上进行试验。为帮助VGG-16的训练，我们在每个卷积层之后都增加了BN层。与之前的模型一样，我们对VGG-16和SE-VGG-16使用一样的训练方案。比较结果如表2所示。与残差网络架构类似，我们也看到SE模块带来了性能的改进。

**Mobile setting**. Finally, we consider two representative architectures from the class of mobile-optimised networks, MobileNet [65] and ShuffleNet [66]. For these experiments, we used a minibatch size of 256 and a weight decay of $4 × 10^{-5}$. We trained the models across 8 GPUs using SGD with momentum (set to 0.9) and an initial learning rate of 0.1 which was reduced by a factor of 10 each time the validation loss plateaued (rather than using a fixed-length schedule). The total training process required ∼ 400 epochs (we found that this approach enabled us to reproduce the baseline performance of [66]). The results reported in Table 3 show that SE blocks consistently improve the accuracy by a large margin at a minimal increase in computational cost.

**移动设置**。最后，我们考虑一下移动优化类别的网络中两个有代表性的架构，MobileNet[65]和ShuffleNet[66]。对于这些试验，我们使用的minibatch size为256，权重衰减为$4 × 10^{-5}$。我们在8 GPUs上进行模型训练，SGD的动量为0.9，初始学习率为0.1，每当验证损失下降停止的时候就学习率除以10（而没有使用固定时刻减小的方案）。全部训练过程需要大约400轮（我们发现这种方法可以复现[66]的基准性能）。表3给出的结果说明，SE模块能够持续显著改进性能，而且计算量增加很小。

Table 3 Single-crop error rates (%) on the ImageNet validation set and complexity comparisons. MobileNet refers to “1.0 MobileNet-224” in [65] and ShuffleNet refers to “ShuffleNet 1 × (g = 3) ” in [66]. The numbers in brackets denote the performance improvement over the re-implementation.

**Additional datasets**. We next investigate whether the benefits of SE blocks generalise to datasets beyond ImageNet. We perform experiments with several popular baseline architectures and techniques (ResNet-110 [14], ResNet-164 [14], WideResNet-16-8 [69], Shake-Shake [70] and Cutout [71]) on the CIFAR-10 and CIFAR-100 datasets [73]. These comprise a collection of 50k training and 10k test 32 × 32 pixel RGB images, labelled with 10 and 100 classes respectively. The integration of SE blocks into these networks follows the same approach that was described in Sec. 3.3. Each baseline and its SENet counterpart are trained with a standard data augmentation strategy [24], [74]. During training, images are randomly horizontally flipped and zero-padded on each side with four pixels before taking a random 32 × 32 crop. Mean and standard deviation normalisation is also applied. The setting of the training strategy and other hyperparameters (e.g. minibatch size, initial learning rate, number of epochs, weight decay) match those suggested by the authors of each model. We report the performance of each baseline and its SENet counterpart on CIFAR-10 in Table 4 and performance on CIFAR-100 in Table 5. We observe that in every comparison SENets outperform the baseline architectures, suggesting that the benefits of SE blocks are not confined to the ImageNet dataset.

**另外的数据集**。下面我们研究SE模块的优点是否能泛化到其他数据集上。我们用几种流行的基准架构和方法(ResNet-110 [14], ResNet-164 [14], WideResNet-16-8 [69], Shake-Shake [70] and Cutout [71])在CIFAR-10和CIFAR-100[73]上进行试验。这包括了50K训练图像，10K测试图像，图像为32 × 32像素的RGB图像，分别标注了10类和100类。SE模块整合到这些网络中的方法与3.3节中一样。每个基准及其对应的SENet训练时都用了标准数据扩充策略[24,74]。在训练时，图像进行随机水平翻转，在进行随机32 × 32剪切时，在每个边都补齐4个像素。均值和标准差归一化也都进行了使用。训练策略的设置以及其他超参数（如minibatch size，初始学习率，训练轮数，权重衰减）与每个模型作者建议的一样。我们每个基准模型及其对应的SENet的性能，表4是在CIFAR-10数据集上的结果，表5是在CIFAR-100上的结果。我们观察到在每个比较中，SENet都超过了其对应的基准架构，说明SE模块不止在ImageNet数据集上可以使用。

### 5.2 Scene Classification 场景分类

We next conduct experiments on the Places365-Challenge dataset [75] for scene classification. This dataset comprises 8 million training images and 36, 500 validation images across 365 categories. Relative to classification, the task of scene understanding offers an alternative assessment of a model’s ability to generalise well and handle abstraction. This is because it often requires the model to handle more complex data associations and to be robust to a greater level of appearance variation.

下一步我们在Places365-Challenge数据集[75]上进行场景分类的试验。这个数据集包括8 million训练图像，36500验证图像，365个类别。与分类相关，场景理解的任务也可以评估模型泛化和处理抽象情况的能力。这是因为它经常需要模型处理更复杂的数据关联，对更大的外貌变化更加稳健。

We opted to use ResNet-152 as a strong baseline to assess the effectiveness of SE blocks and carefully follow the training and evaluation protocols described in [72]. In these experiments, all models are trained from scratch. We report the results in Table 6, comparing also with prior work. We observe that SE-ResNet-152 (11.01% top-5 error) achieves a lower validation error than ResNet-152 (11.61% top-5 error), providing evidence that SE blocks can also yield improvements for scene classification. This SENet surpasses the previous state-of-the-art model Places-365-CNN [72] which has a top-5 error of 11.48% on this task.

我们选择使用ResNet-152作为强基准来评估SE模块的有效性，小心的遵循[72]中的训练和评估方案。在这些试验中，所有模型都从头训练。我们在表6中给出结果，与之前的工作进行比较。我们观察到SE-ResNet-152(11.01% top-5 error)取得了更低的验证错误率，ResNet-152的top-5 error则为11.61%，说明SE模块在场景分类任务中也可以取得改进。这个SENet模型超过了之前最好的模型Places-365-CNN [72]，其在这个任务上的top-5 error为11.48%。

### 5.3 Object Detection on COCO 目标检测

We further assess the generalisation of SE blocks on the task of object detection using the COCO dataset [76] which comprises 80k training images and 40k validation images, following the splits used in [13]. We use the Faster R-CNN [4] detection framework as the basis for evaluating our models and follow the basic implementation described in [13]. Our goal is to evaluate the effect of replacing the trunk architecture (ResNet) in the object detector with SE-ResNet, so that any changes in performance can be attributed to better representations. Table 7 reports the validation set performance of the object detector using ResNet-50, ResNet-101 and their SE counterparts as trunk architectures. SE-ResNet-50 outperforms ResNet-50 by 1.3% (a relative 5.2% improvement) on COCO’s standard AP metric and by 1.6% on AP@IoU=0.5. SE blocks also bring improvements on the deeper ResNet-101 architecture achieving a 0.7% improvement (or 2.6% relative improvement) on the AP metric.

我们进一步评估SE模块在目标检测上的泛化能力，使用COCO数据集[76]，包括80k训练图像，40k验证图像，使用[13]中的分割方法。我们使用Faster R-CNN[4]检测框架作为评估我们方法的基础，使用[13]中的基础实现。我们的目标是评估将目标检测器中的骨干架构ResNet替换为SE-ResNet的效果，这样性能上的改变都可以认为是更好的表示带来的结果。表7给出了目标检测器在验证集上的性能，分别使用的是ResNet-50, ResNet-101及其对应的SE部分作为骨干架构。SE-ResNet-50超过了ResNet-50 1.3%（相对改进5.2%），使用的是COCO的标准AP度量标准，在AP@IoU=0.5时候是1.6%。SE模块在更深的ResNet-101架构上也带来了改进，在AP度量标准上取得了0.7%的改进（或2.6%的相对改进）。

In summary, this set of experiments demonstrate that the improvements induced by SE blocks can be realised across a broad range of architectures, tasks and datasets. 总结起来，这一系列试验说明了，SO模块带来的改进可以在广泛的架构、任务和数据集中得到实现。

### 5.4 ILSVRC 2017 Classification Competition

SENets formed the foundation of our submission to the ILSVRC competition where we achieved first place. Our winning entry comprised a small ensemble of SENets that employed a standard multi-scale and multi-crop fusion strategy to obtain a top-5 error of 2.251% on the test set. As part of this submission, we constructed an additional model, SENet-154, by integrating SE blocks with a modified ResNeXt [19] (the details of the architecture are provided in Appendix). We compare this model with prior work on the ImageNet validation set in Table 8 using standard crop sizes (224 × 224 and 320 × 320). We observe that SENet-154 achieves a top-1 error of 18.68% and a top-5 error of 4.47% using a 224 × 224 centre crop evaluation, which represents the strongest reported result.

SENets是我们提交给ILSVRC竞赛的基础，我们在比赛中取得了第一名的位置。我们获胜的模型是小型的SENets的集成，使用了标准的多尺度和多剪切块融合策略，在测试集上得到的top-5错误率为2.251%。我们还构建了一个另外的模型，SENet-154，是通过将SE模块集成到了修正的ResNeXt[19]中（架构细节如附录所示）。我们将这个模型与之前的工作在ImageNet验证集上进行比较，结果如表8所示，使用的是标准剪切块大小(224 × 224 and 320 × 320)。我们观察到，SENet-154取得的top-1 error为18.68%，top-5 error为4.47%，使用的是224 × 224中间剪切块评估，这代表了最强的报导结果。

Following the challenge there has been a great deal of further progress on the ImageNet benchmark. For comparison, we include the strongest results that we are currently aware of among the both published and unpublished literature in Table 9. The best performance using only ImageNet data was recently reported by [79]. This method uses reinforcement learning to develop new polices for data augmentation during training to improve the performance of the architecture proposed by [31]. The best overall performance was reported by [80] using a ResNeXt-101 32×48d architecture. This was achieved by pretraining their model on approximately one billion weakly labelled images and finetuning on ImageNet. The improvements yielded by more sophisticated data augmentation [79] and extensive pretraining [80] may be complementary to our proposed changes to the network architecture.

在这次挑战上，ImageNet基准测试有了很多进一步的进展。为比较，我们在表9中包括了我们所知的已发表和未发表的最强结果。只使用ImageNet数据的最佳性能是最近发表的[79]。这种方法使用强化学习发展出了新的训练过程中数据扩充的方法，改进了[31]提出的网络架构的性能。最佳的总体性能由[80]给出，使用的是一个ResNeXt-101 32×48d架构。这是在大约1 billion弱标注图像中预训练模型并在ImageNet上精调的结果。更复杂的数据扩充[79]和广泛的预训练[80]带来的改进，与我们对网络架构的改变，是互补的。

## 6 Ablation Study 分离试验

In this section we conduct ablation experiments to gain a better understanding of the relative importance of components in the SE block design. All ablation experiments are performed on the ImageNet dataset on a single machine (with 8 GPUs). ResNet-50 is used as the backbone architecture. The data augmentation strategy follows the approach described in Sec. 5.1. To allow us to study the upper limit of performance for each variant, the learning rate is initialised to 0.1 and training continues until the validation loss plateaus (rather than continuing for a fixed number of epochs). The learning rate is then reduced by a factor of 10 and then this process is repeated (three times in total).

在这一部分中，我们进行分离试验，以更好的理解SE模块设计中的组建的相对重要性。所有的分离试验都在ImageNet数据集及同一台机器(8 GPSs)上进行。ResNet-50用作骨架架构。数据扩充策略采用5.1中描述的方法。为研究每种变体的上限表现，学习率初始化为0.1，直到所有的验证损失停止下降，才停止训练（而不是固定轮数的训练），然后学习率除以10，这个过程重复下去（共重复3次）。

### 6.1 Reduction ratio 缩减率

The reduction ratio r introduced in Eqn. 5 is a hyperparameter which allows us to vary the capacity and computational cost of the SE blocks in the network. To investigate the trade-off between performance and computational cost mediated by this hyperparameter, we conduct experiments with SE-ResNet-50 for a range of different r values. The comparison in Table 10 shows that performance does not improve monotonically with increased capacity, suggesting that with enough weights the SE block is able to overfit to the channel interdependencies of the training set. We found that setting r = 16 achieved a good balance between accuracy and complexity and consequently, we used this value for all experiments reported in this work.

式(5)中引入的缩减率是一个超参数，使我们改变网络中SE模块的容量和计算量。为研究这个超参数带来的性能和计算量的折中，我们用SE-ResNet-50对不同的r值进行了试验。比较结果如表10所示，表明性能并不是随着容量的提升而同时提升，说明权重足够多时，SE模块对于训练集上的通道之间的依赖关系会过拟合。我们发现，设r=16会在准确率和复杂度之间得到较好的均衡，后面我们在所有试验中都使用这个值。

TABLE 10 Single-crop error rates (%) on ImageNet validation set and parameter sizes for SE-ResNet-50 at different reduction ratios r. Here, original refers to ResNet-50.

Ratio r | top-1 err. | top-5 err. | Params
--- | --- | --- | ---
4 | 22.25 | 6.09 | 35.7M
8 | 22.26 | 5.99 | 30.7M
16 | 22.28 | 6.03 | 28.1M
32 | 22.72 | 6.20 | 26.9M
original | 23.30 | 6.55 | 25.6M

### 6.2 Squeeze Operator

We examine the significance of using global average pooling as opposed to global max pooling as our choice of squeeze operator (since this worked well, we did not consider more sophisticated alternatives). The results are reported in Table 11. While both max and average pooling are effective, average pooling achieves slightly better performance, justifying its selection as the basis of the squeeze operation. However, we note that the performance of SE blocks is fairly robust to the choice of specific aggregation operator.

我们检验了使用全局平均pooling与全局max pooling作为squeeze算子的比较（没有考虑更复杂的选择）。结果如表11所示。Max和平均pooling都是有效的，但是平均pooling得到了略好的结果，所以我们选择其作为squeeze算子的基础。但是，我们也注意到，SE模块的性能对于特定的聚集算子的选择还是很稳健的。

TABLE 11 Effect of using different squeeze operators in SE-ResNet-50 on ImageNet (%).

Squeeze | top-1 err. | top-5 err.
--- | --- | ---
Max | 22.57 | 6.09
Avg | 22.28 | 6.03

### 6.3 Excitation Operator

We next assess the choice of non-linearity for the excitation mechanism. We consider two further options: ReLU and tanh, and experiment with replacing the sigmoid with these alternative non-linearities. The results are reported in Table 12. We see that exchanging the sigmoid for tanh slightly worsens performance, while using ReLU is dramatically worse and in fact causes the performance of SE-ResNet-50 to drop below that of the ResNet-50 baseline. This suggests that for the SE block to be effective, careful construction of the excitation operator is important.

下一步我们评估excitation机制的非线性的选择。我们考虑两种更多的选择：ReLU和tanh，将其替换掉sigmoid进行试验。其结果如表12所示。我们可以看到，将sigmoid替换为tanh带来的略差的结果，而使用了ReLU则效果差的多，使SE-ResNet-50的性能降到了ResNet-50之下。这说明为使SE模块有效，需要仔细的选择excitation算子。

TABLE 12 Effect of using different non-linearities for the excitation operator in SE-ResNet-50 on ImageNet (%).

Excitation | top-1 err. | top-5 err.
--- | --- | ---
ReLU | 23.47 | 6.98
Tanh | 23.00 | 6.38
Sigmoid | 22.28 | 6.03

### 6.4 Different stages

We explore the influence of SE blocks at different stages by integrating SE blocks into ResNet-50, one stage at a time. Specifically, we add SE blocks to the intermediate stages: stage 2, stage 3 and stage 4, and report the results in Table 13. We observe that SE blocks bring performance benefits when introduced at each of these stages of the architecture. Moreover, the gains induced by SE blocks at different stages are complementary, in the sense that they can be combined effectively to further bolster network performance.

我们将SE模块与ResNet-50整合到一起（每次整合进一个阶段），探讨SE模块在不同阶段的影响。特别的，我们将SE模块放入中间阶段中：阶段2，3和4，结果如表13所示。我们观察SE模块在每个阶段中对性能的影响。而且，SE模块在不同阶段的收益是互补的，因为它们可以有效的组合到一起以提升网络性能。

TABLE 13 Effect of integrating SE blocks with ResNet-50 at different stages on ImageNet (%).

Stage | top-1 err. | top-5 err. | GFLOPs | Params
--- | --- | --- | --- | ---
ResNet-50 | 23.30 | 6.55 | 3.86 | 25.6M
SE Stage 2 | 23.03 | 6.48 | 3.86 | 25.6M
SE Stage 3 | 23.04 | 6.32 | 3.86 | 25.7M
SE Stage 4 | 22.68 | 6.22 | 3.86 | 26.4M
SE All | 22.28 | 6.03 | 3.87 | 28.1M

### 6.5 Integration strategy 组合策略

Finally, we perform an ablation study to assess the influence of the location of the SE block when integrating it into existing architectures. In addition to the proposed SE design, we consider three variants: (1) SE-PRE block, in which the SE block is moved before the residual unit; (2) SE-POST block, in which the SE unit is moved after the summation with the identity branch and (3) SE-Identity block, in which the SE unit is placed on the identity connection in parallel to the residual unit. These variants are illustrated in Figure 5 and the performance of each variant is reported in Table 14. We observe that the SE-PRE, SE-Identity and proposed SE block each perform similarly well, while usage of the SE-POST block leads to a drop in performance. This experiment suggests that the performance improvements produced by SE units are fairly robust to their location, provided that they are applied prior to branch aggregation.

最后，我们进行一个分离试验，来评估SE模块插入现有模块中时候，其位置的影响。除了已经提出的SE设计，我们考虑以下三种变体：(1)SE-PRE模块，其中SE模块移到了残差模块的前面；(2)SE-POST模块，其中SE模块放在了与恒等分支相加之后；(3)SE-Identity模块，其中SE模块放在与恒等连接平行的位置。这些变体如图5所示，每个变体的性能如表14所示。我们观察到SE-PRE，SE-Identity和提出的标准SE模块性能类似，而SE-POST模块则会导致性能下降。这个试验说明，SE单元带来的性能改进对其位置是相对稳健的，如果是放在之前，或平行的位置。

TABLE 14 Effect of different SE block integration strategies with ResNet-50 on ImageNet (%).

Design | top-1 err. | top-5 err.
--- | --- | ---
SE | 22.28 | 6.03
SE-PRE | 22.23 | 6.00
SE-POST | 22.78 | 6.35
SE-Identity | 22.20 | 6.15

In the experiments above, each SE block was placed outside the structure of a residual unit. We also construct a variant of the design which moves the SE block inside the residual unit, placing it directly after the 3 × 3 convolutional layer. Since the 3 × 3 convolutional layer possesses fewer channels, the number of parameters introduced by the corresponding SE block is also reduced. The comparison in Table 15 shows that the SE 3 × 3 variant achieves comparable classification accuracy with fewer parameters than the standard SE block. Although it is beyond the scope of this work, we anticipate that further efficiency gains will be achievable by tailoring SE block usage for specific architectures.

在上面的试验中，每个SE模块都在残差模块之外。我们还构建了一种设计的变体，将SE模块放入了残差模块之内，将其直接放在3 × 3卷积层的后面。由于3 × 3卷积通道数不多，相应的SE层引入的参数数量也相应减少了。表15的比较结果说明，SE的3 × 3变体得到还可以的分类准确率，参数数量也比标准SE模块要少。我们期待将SE模块与其他特定架构的组合也能得到进一步的效率提升，虽然这不再我们的工作研究之内。

TABLE 15 Effect of integrating SE blocks at the 3x3 convolutional layer of each residual branch in ResNet-50 on ImageNet (%).

Design | top-1 err. | top-5 err. | GFLOPs | Params
--- | --- | --- | --- | --- 
SE | 22.28 | 6.03 | 3.87 | 28.1M
SE_3×3 | 22.48 | 6.02 | 3.86 | 25.8M

## 7 Role of SE Blocks

Although the proposed SE block has been shown to improve network performance on multiple visual tasks, we would also like to understand the role of the squeeze operation and how the excitation mechanism operates in practice. Unfortunately, a rigorous theoretical analysis of the representations learned by deep neural networks remains challenging. We therefore take an empirical approach to examining the role played by the SE block with the goal of attaining at least a primitive understanding of its practical function.

虽然提出的SE模块在多个视觉任务中都改进了网络性能，但我们还是想理解squeeze操作的角色和excitation机制在实践中是怎么运行的。不幸的是，深度神经网络学习到的表示的严格理论分析仍然是一个挑战。所以我们用经验的方法来检验SE模块扮演的角色，目标是得到其实际函数的原始理解。

### 7.1 Effect of Squeeze

To assess whether the global embedding produced by the squeeze operation plays an important role in performance, we experiment with a variant of the SE block that adds an equal number of parameters, but does not perform global average pooling. Specifically, we remove the pooling operation and replace the two FC layers with corresponding 1 × 1 convolutions with identical channel dimensions in the excitation operator, namely NoSqueeze, where the excitation output maintains the spatial dimensions as input. In contrast to the SE block, these point-wise convolutions can only remap the channels as a function of the output of a local operator. While in practice, the later layers of a deep network will typically possess a (theoretical) global receptive field, global embeddings are no longer directly accessible throughout the network in the NoSqueeze variant. The accuracy and computational complexity of the both models are compared to a standard ResNet-50 model in Table 16. We observe that the use of global information has a significant influence on the model performance, underlining the importance of the squeeze operation. Moreover, in comparison to the NoSqueeze design, the SE block allows this global information to be used in a computationally parsimonious manner.

为评估squeeze操作得到的全局嵌入是否在性能中非常重要，我们设计了一种SE模块的变体进行试验，即插入了相同数量的参数，但并没有进行全局平均pooling。特别的，我们去除了pooling操作，将两个全连接层替换成对应的1 × 1卷积，通道维数与excitation操作相同，名称为NoSqueeze，其中excitation输出保持与输入空间维度相同。与SE模块比较起来，这些逐点的卷积只能重新映射通道，作为输出的局部描述子的函数。实践中，深度网络后面的层一般会有一个（理论上的）全局感受野，在NoSqueeze变体中，全局嵌入不再是直接可用的。两个模型的准确度和计算复杂度在表16中与标准ResNet-50模型进行了比较。我们观察到，使用了全局信息对模型性能有着显著影响，说明squeeze操作是有着很重要的作用的。而且，与NoSqueeze设计比较起来，SE模块对全局信息的利用，消耗的计算量是非常少的。

### 7.2 Role of Excitation

To provide a clearer picture of the function of the excitation operator in SE blocks, in this section we study example activations from the SE-ResNet-50 model and examine their distribution with respect to different classes and different input images at various depths in the network. In particular, we would like to understand how excitations vary across images of different classes, and across images within a class.

为认识SE模块中excitation算子的作用，本节中我们研究了SE-ResNet-50模型的样本激活，检查相对于不同类别和不同输入图像的在网络不同深度的分布。特别的，我们希望理解excitation在不同类别中的图像中怎样变化，以及在一个类别的图像中怎样变化。

We first consider the distribution of excitations for different classes. Specifically, we sample four classes from the ImageNet dataset that exhibit semantic and appearance diversity, namely goldfish, pug, plane and cliff (example images from these classes are shown in Figure 6). We then draw fifty samples for each class from the validation set and compute the average activations for fifty uniformly sampled channels in the last SE block of each stage (immediately prior to downsampling) and plot their distribution in Fig. 7. For reference, we also plot the distribution of the mean activations across all of the 1000 classes.

我们首先考虑不同类别的激活的分布。特别的，我们从ImageNet数据集中取样4类，有不同的语义和外表，即金鱼，哈巴狗，飞机和悬崖（图6中给出了这些类别的图像样本）。我们然后从验证集的每个类别中取出50幅样本，计算50个均匀取样的通道，在每个阶段的最后一个SE模块的其平均激活（下采样之前的一层），将其分布画于图7中。我们还给出了1000个类别的平均激活的分布，作为参考。

We make the following three observations about the role of the excitation operation. First, the distribution across different classes is very similar at the earlier layers of the network, e.g. SE_2_3. This suggests that the importance of feature channels is likely to be shared by different classes in the early stages. The second observation is that at greater depth, the value of each channel becomes much more class-specific as different classes exhibit different preferences to the discriminative value of features, e.g. SE_4_6 and SE_5_1. These observations are consistent with findings in previous work [81], [82], namely that earlier layer features are typically more general (e.g. class agnostic in the context of the classification task) while later layer features exhibit greater levels of specificity [83].

我们对excitation算子的操作得到了下面三个观察结果。第一，在网络前期的层中不同类别的分布非常类似，如SE_2_3。这说明在早期阶段特征通道的重要性是不同类别共享的。第二个观察是，在更深的层中，每个通道的值变成与类别相关性更大，因为不同类别对于特征区分性值的表现是不同的，如SE_4_6和SE_5_1。这些观察与之前的工作[81,82]是一致的，即早期的层更一般性（即，在分类工作中表现出类别无关性），而更晚的层表现出了更多的特别性[83]。

Next, we observe a somewhat different phenomena in the last stage of the network. SE_5_2 exhibits an interesting tendency towards a saturated state in which most of the activations are close to one and the remainder is close to zero. At the point at which all activations take the value one, an SE block reduces to the identity operator. At the end of the network in the SE_5_3 (which is immediately followed by global pooling prior before classifiers), a similar pattern emerges over different classes, up to a slight change in scale (which could be tuned by the classifiers). This suggests that SE_5_2 and SE_5_3 are less important than previous blocks in providing recalibration to the network. This finding is consistent with the result of the empirical investigation in Sec. 4 which demonstrated that the additional parameter count could be significantly reduced by removing the SE blocks for the last stage with only a marginal loss of performance.

然后，我们在网络的最后阶段得到了一定程度的不同现象。SE_5_2表现出了一定的饱和状态，这是一个有趣的倾向，大多数激活接近1，剩下的接近0。在等于1的点上，SE模块退化成恒等算子。在网络最后的SE_5_3（其后是一个全局pooling，然后就是分类器），不同类别表现出了相似的模式，在尺度上有一定的变化（这可以由分类器调节）。这说明SE_5_2和SE_5_3在向网络提供重新校准方面没有之前的模块重要。这个发现与第4部分的经验是一致的，证明了去除最后阶段的SE模块可以显著减少额外的参数数量，性能则只有很小的降低。

Finally, we show the mean and standard deviations of the activations for image instances within the same class for two sample classes (goldfish and plane) in Fig. 8. We observe a trend consistent with the inter-class visualisation, indicating that the dynamic behaviour of SE blocks varies over both classes and instances within a class. Particularly in the later layers of the network where there is considerable diversity of representation within a single class, the network learns to take advantage of feature recalibration to improve its discriminative performance. In summary, SE blocks produce instance-specific responses which nevertheless function to support the increasingly class-specific needs of the model at different layers in the architecture.

最后，我们给出了两个类别（金鱼和飞机）中，同样类别里的图像实例的激活的均值和标准差，如图8所示。我们观察到与类内可视化相同的趋势，说明SE模块的动态行为在类别之间是不一样的，在同一类别内的实例之间也是不一样的。特别是在网络后面的层中，同一类别表示中的变化还是很大的，网络利用特征重校准来改进其区分能力。总结来说，SE模块会生成实例相关的响应，在网络架构中的不同层上起到越来越多的类别相关的需求的作用。

## 8 Conclusion 结论

In this paper we proposed the SE block, an architectural unit designed to improve the representational power of a network by enabling it to perform dynamic channel-wise feature recalibration. A wide range of experiments show the effectiveness of SENets, which achieve state-of-the-art performance across multiple datasets and tasks. In addition, SE blocks shed some light on the inability of previous architectures to adequately model channel-wise feature dependencies. We hope this insight may prove useful for other tasks requiring strong discriminative features. Finally, the feature importance values produced by SE blocks may be of use for other tasks such as network pruning for model compression.

在本文中，我们提出了SE模块，这种架构单元设计是用于改进网络的表示能力，其方法是进行动态的逐通道特征重校准。进行了一系列试验，展现了SENets的有效性，在多个数据集和任务中都得到了目前最好的性能。另外，SE模块也反应了之前的架构不能充分的对通道间的特征依赖关系进行建模。我们希望这种思想能在其他需要区分性特征的任务中也证明有用。最后，SE模块产生的特征重要性值可能对其他任务有用，比如模型压缩中的网络剪枝。

## Appendix: Details of SENet-154

SENet-154 is constructed by incorporating SE blocks into a modified version of the 64 × 4d ResNeXt-152 which extends the original ResNeXt-101 [19] by adopting the block stacking strategy of ResNet-152 [13]. Further differences to the design and training of this model (beyond the use of SE blocks) are as follows: (a) The number of the first 1 × 1 convolutional channels for each bottleneck building block was halved to reduce the computational cost of the model with a minimal decrease in performance. (b) The first 7 × 7 convolutional layer was replaced with three consecutive 3 × 3 convolutional layers. (c) The 1 × 1 down-sampling projection with stride-2 convolution was replaced with a 3 × 3 stride-2 convolution to preserve information. (d) A dropout layer (with a dropout ratio of 0.2) was inserted before the classification layer to reduce overfitting. (e) Label-smoothing regularisation (as introduced in [20]) was used during training. (f) The parameters of all BN layers were frozen for the last few training epochs to ensure consistency between training and testing. (g) Training was performed with 8 servers (64 GPUs) in parallel to enable large batch sizes (2048). The initial learning rate was set to 1.0.

SENet-154是将SE模块与修正版的64 × 4d ResNeXt-152整合到一起形成的，修正版采用了ResNet-152 [13]的模块堆叠策略，拓展了原始的ResNeXt-101 [19]。模型设计和训练的更多区别（SE模块之外的）如下：(a)每个瓶颈模块的第一个1 × 1卷积层的通道数量都减半了，以降低模型计算量，且对性能影响最小；(b)第一个7 × 7卷积层替换成3个连续的3 × 3卷积层；(c)步长为2的1 × 1卷积下采样投影替换为步长为2的3 × 3卷积以保留信息；(d)分类层中插入了一个dropout层（dropout率为0.2），以减少过拟合；(e)[20]中引入的标签平滑正则化在训练中得到了使用；(f)所有的BN层的参数在最后几轮的训练中都冻结了，以确保训练和测试时的一致性；(g)训练使用了8个服务器（64个GPUs）并行进行，以使用大的batch size (2048)。初始学习率设为1.0。