# Deep Residual Learning for Image Recognition
# 图像识别中的深度残差学习

Kaiming He et al. Microsoft Research

## Abstract

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers — 8× deeper than VGG nets [41] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet testset. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.

更深的神经网络更难训练。我们提出了一个残差学习框架使网络更加容易训练，这些网络比以前使用的要深很多。我们将各层重新组合成对残差函数进行学习，函数参数就是层的输入。我们有充足的经验证据说明，这些残差网络优化起来更加容易，并从更深的网络中得到更高的准确率。在ImageNet数据集中，我们用152层的深度网络来评估残差网络，这比VGG[41]要深8倍，但复杂度仍然低一些。这些残差网络的集成在ImageNet测试集上得到了3.75%错误率的成绩，这赢得了ILSVRC-2015分类任务的第一名。我们还在CIFAR-10数据集上提出了100层和1000层的网络，并进行分析。

The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

表示的深度对于很多视觉识别任务来说是最重要的。仅仅由于深度非常深，我们在COCO目标识别数据集上得到了28%的相对进步。深度残差网络是我们参加ILSVRC-2015和COCO-2015的基础，我们在ImageNet检测和定位、COCO检测和分割都取得了第一名的成绩。

## 1. Introduction

Deep convolutional neural networks [22, 21] have led to a series of breakthroughs for image classification [21, 50, 40]. Deep networks naturally integrate low/mid/high-level features [50] and classifiers in an end-to-end multi-layer fashion, and the “levels” of features can be enriched by the number of stacked layers (depth). Recent evidence [41, 44] reveals that network depth is of crucial importance, and the leading results [41, 44, 13, 16] on the challenging ImageNet dataset [36] all exploit “very deep” [41] models, with a depth of sixteen [41] to thirty [16]. Many other non-trivial visual recognition tasks [8, 12, 7, 32, 27] have also greatly benefited from very deep models.

深度卷积神经网络[22,21]在图像分类中取得了一系列突破[21,50,40]。深度网络很自然的将低层/中层/高层特征[50]集成起来，并且是端到端的多层模式，特征的层次可以通过增加层数来增加。最近的研究[41,44]表明，网络深度非常重要，目前领先的模型[41,44,13,16]都在挖掘深度的潜力，达到了16层[41]到30层[16]的深度。其他有意义的视觉识别任务[8,12,7,32,27]都从深度模型中获益良多。

Driven by the significance of depth, a question arises: Is learning better networks as easy as stacking more layers? An obstacle to answering this question was the notorious problem of vanishing/exploding gradients [1, 9], which hamper convergence from the beginning. This problem, however, has been largely addressed by normalized initialization [23, 9, 37, 13] and intermediate normalization layers [16], which enable networks with tens of layers to start converging for stochastic gradient descent (SGD) with backpropagation [22].

深度的重要性带来了一个问题：学习到更好的网络就像叠加更多的层那么容易吗？回答这个问题的一个障碍就是著名的梯度消失和梯度爆炸问题[1,9]，这从一开始就阻碍收敛。这个问题已经很大程度上用初始化归一化[23,9,37,13]和中间归一化层[16]解决了，这使得几十层的网络用随机梯度下降法(SGD)和反向传播[22]逐渐收敛。

When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by our experiments. Fig. 1 shows a typical example.

当更深的网络可以收敛时，出现了一个问题：随着网络深度的增加，准确度慢慢饱和，然后迅速恶化。这不让人感到意外，这种问题不是由于过拟合，向一个合理的深度网络增加更多层时，会导致训练错误率更高，这在[11,42]中有表现，我们的试验也完全证实了这一点。图1给出了一个典型例子。

Figure 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4.

图1. 20层和56层网络在CIFAR-10上的训练错误率（左边）和测试错误率（右）。较深的网络反而训练错误率更高，测试错误率也一样。ImageNet中也有类似的现象，如图4所示。

The degradation (of training accuracy) indicates that not all systems are similarly easy to optimize. Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time).

这说明网络优化的难易程度是不一的。我们考虑一个更浅的架构，其加深的架构就是通过增加更多的层。构建这个更深层的模型，有一个解决方案：增加的层为恒等映射，其他层从学习好的浅层模型拷贝而来。这种构建解决方案说明，深层模型的训练错误率不会比浅层模型更高。但试验表明，现有的解决方案没有达到前述的方案的效果。

In this paper, we address the degradation problem by introducing a deep residual learning framework. Instead of hoping each few stacked layers directly fit a desired underlying mapping, we explicitly let these layers fit a residual mapping. Formally, denoting the desired underlying mapping as *H(x)*, we let the stacked nonlinear layers fit another mapping of *F(x) := H(x)−x*. The original mapping is recast into *F(x)+x*. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

本文中，我们引入了深度残差学习框架来解决这个问题。我们显式的让这些层成为残差映射，而不是试图让几个叠加的层学习成希望的映射。令希望的映射为*H(x)*，我们使叠加的非线性层逼近另一个映射*F(x) := H(x)−x*，那么原来的映射也就成为了*F(x)+x*。我们假设优化残差映射更为容易。一个极端情况是，如果恒等映射是最佳的，那么非线性层的叠加逼近残差0更容易，而不用逼近恒等映射。

The formulation of *F(x)+x* can be realized by feedforward neural networks with “shortcut connections” (Fig. 2). Shortcut connections [2, 34, 49] are those skipping one or more layers. In our case, the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers (Fig. 2). Identity shortcut connections add neither extra parameter nor computational complexity. The entire network can still be trained end-to-end by SGD with backpropagation, and can be easily implemented using common libraries (e.g., Caffe [19]) without modifying the solvers.

*F(x)+x*可以用带有“捷径连接”的前向神经网络来实现（图2）。捷径连接[2,34,49]跳过一层或几层。在我们的情况中，捷径连接就是恒等映射，其输出与叠加的层的输出相加（图2）。恒等捷径连接不增加额外的参数量，也不增加计算复杂度。整个网络仍然可以用SGD和反向传播进行端到端的训练，可以很容易用常见库（如Caffe[19]）实现，而不用修改。

We present comprehensive experiments on ImageNet [36] to show the degradation problem and evaluate our method. We show that: 1) Our extremely deep residual nets are easy to optimize, but the counterpart “plain” nets (that simply stack layers) exhibit higher training error when the depth increases; 2) Our deep residual nets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks.

我们在ImageNet上进行了很多实验，结果显示：(1)我们的极深残差网络易于优化，但如果没有捷径连接，那么随着网络深度增加，训练错误率也增加；(2)我们的深度残差网络随着深度增加，准确度也得到提升，得到了比以前的架构好的多的效果。

Similar phenomena are also shown on the CIFAR-10 set [20], suggesting that the optimization difficulties and the effects of our method are not just a kin to a particular dataset. We present successfully trained models on this dataset with over 100 layers, and explore models with over 1000 layers.

在CIFAR-10数据集上的试验中也有类似的现象，说明我们方法的优化难度和效果并不是在某一特定数据集的特例。我们在此数据集上训练的模型深度超过100，还试验了超过1000层的网络。

On the ImageNet classification dataset [36], we obtain excellent results by extremely deep residual nets. Our 152-layer residual net is the deepest network ever presented on ImageNet, while still having lower complexity than VGG nets [41]. Our ensemble has 3.57% top-5 error on the ImageNet test set, and won the 1st place in the ILSVRC 2015 classification competition. The extremely deep representations also have excellent generalization performance on other recognition tasks, and lead us to further win the 1st places on: ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation in ILSVRC & COCO 2015 competitions. This strong evidence shows that the residual learning principle is generic, and we expect that it is applicable in other vision and non-vision problems.

在ImageNet分类数据集[36]上，我们得到非常好的效果。我们的152层残差网络是在ImageNet上提出的最深网络，但比VGG网络运算复杂度还要低一些。我们的集成方法在ImageNet测试集上得到了3.75%的top-5错误率，在ILSVRC-2015分类比赛上得到了第一的位置。这个极深的表示在其他识别任务中也有很好的泛化表现，我们在这些比赛中也得到了第一的成绩：2015年的ImageNet检测，ImageNet定位，COCO检测，COCO分割。这表明残差学习的原则是一般性的，我们期待在其他视觉任务和非视觉任务中也得到应用。

## 2. Related Work

**Residual Representations**. In image recognition, VLAD [18] is a representation that encodes by the residual vectors with respect to a dictionary, and Fisher Vector [30] can be formulated as a probabilistic version [18] of VLAD. Both of them are powerful shallow representations for image retrieval and classification [4, 48]. For vector quantization, encoding residual vectors [17] is shown to be more effective than encoding original vectors.

**残差表示**。在图像识别中，VLAD[18]是残差矢量对一个字典的表示编码，Fisher矢量[30]可以看做是VLAD的概率版。它们都是图像检索和图像分类应用的强大浅层表示[4,48]。对于矢量量化，对残差矢量进行编码[17]比对原始矢量进行编码更有效率。

In low-level vision and computer graphics, for solving Partial Differential Equations (PDEs), the widely used Multigrid method [3] reformulates the system as subproblems at multiple scales, where each subproblem is responsible for the residual solution between a coarser and a finer scale. An alternative to Multigrid is hierarchical basis preconditioning [45, 46], which relies on variables that represent residual vectors between two scales. It has been shown [3,45,46] that these solvers converge much faster than standard solvers that are unaware of the residual nature of the solutions. These methods suggest that a good reformulation or preconditioning can simplify the optimization.

在低层视觉和计算机图形学中，为求解偏微分方程(PDEs)，广泛使用的Multigrid法[3]将问题分解为多尺度上的子问题，每个子问题负责两个相邻尺度之间的残差解。Multigrid的一个替代品是hierarchical basis preconditioning [45, 46]，依赖代表两个尺度间的残差矢量的变量。这些解决方法比没有使用残差的标准方法收敛速度快很多[3,45,46]。这些方法说明残差可以简化优化问题。

**Shortcut Connections**. Practices and theories that lead to shortcut connections [2, 34, 49] have been studied for a long time. An early practice of training multi-layer perceptrons (MLPs) is to add a linear layer connected from the network input to the output [34, 49]. In [44, 24], a few intermediate layers are directly connected to auxiliary classifiers for addressing vanishing/exploding gradients. The papers of [39, 38, 31, 47] propose methods for centering layer responses, gradients, and propagated errors, implemented by shortcut connections. In [44], an “inception” layer is composed of a shortcut branch and a few deeper branches.

**捷径连接**。捷径连接的实践和理论[2,34,49]已经有了很长时间的研究。早期训练多层感知机(MLP)时，增加了从输入到输出的线性连接层[34,49]。在[44,24]中，一些中间层直接与辅助分类器相连，以解决梯度消失/爆炸问题。文章[39, 38, 31, 47]提出方法令层响应，梯度和传播的错误中心化，这也是通过捷径连接实现的。在[44]中，Inception层是由一个捷径分支和一些深一些的分支组成的。

Concurrent with our work, “highway networks” [42, 43] present shortcut connections with gating functions [15]. These gates are data-dependent and have parameters, in contrast to our identity shortcuts that are parameter-free. When a gated shortcut is “closed” (approaching zero), the layers in highway networks represent non-residual functions. On the contrary, our formulation always learns residual functions; our identity shortcuts are never closed, and all information is always passed through, with additional residual functions to be learned. In addition, highway networks have not demonstrated accuracy gains with extremely increased depth (e.g., over 100 layers).

与我们的工作同时出现的是“highway networks” [42, 43]，其中提出了带有门函数的捷径连接[15]。这些门与数据有关，带有参数，而我们的恒等捷径是无参数的。当一个有门的捷径关闭（接近零）时，highway networks中的层代表无残差函数。相反，我们永远学习残差函数，我们的恒等捷径从不关闭，所有的信息都可以通过，这样才可以学习残差函数。另外，highway networks在深度增加（比如，超过100层）时，没有表现出准确度的增加。

## 3. Deep Residual Learning 深度残差学习
### 3.1. Residual Learning 残差学习

Let us consider *H(x)* as an underlying mapping to be fit by a few stacked layers (not necessarily the entire net), with *x* denoting the inputs to the first of these layers. If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions (This hypothesis, however, is still an open question) , then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, i.e., *H(x) − x* (assuming that the input and output are of the same dimensions). So rather than expect stacked layers to approximate *H(x)*, we explicitly let these layers approximate a residual function *F(x) := H(x) − x*. The original function thus becomes *F(x)+x*. Although both forms should be able to asymptotically approximate the desired functions (as hypothesized), the ease of learning might be different.

令*H(x)*为数层网络（不一定是整个网络）要逼近的映射函数，其中*x*为第一层的输入。我们假设多个非线性层可以渐进近似复杂函数（这个假设仍然有待证明），这也就相当于假设，可以渐进近似残差函数即*H(x) − x*（假设输入输出的维数相同）。我们让这些堆叠层近似残差函数*F(x) := H(x) − x*，而不去近似*H(x)*。那么原始函数*H(x)*就可以表示为*F(x)+x*。虽然两种形成都可以渐进近似到理想的函数，但其学习难度可能不一样。

This reformulation is motivated by the counterintuitive phenomena about the degradation problem (Fig. 1, left). As we discussed in the introduction, if the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart. The degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers. With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

这种改变是受到图1的降质问题的反直觉现象所启发得到的。如我们在引言中讨论的一样，如果增加的层可以构建成恒等映射，那么深层模型不会比相应的浅层模型错误率更高。降质问题说明，多个非线性层来近似恒等映射会有困难。通过改变成残差学习，如果恒等映射是最佳的话，网络学习可能会将权重驱赶到接近零的方向。

In real cases, it is unlikely that identity mappings are optimal, but our reformulation may help to precondition the problem. If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping, than to learn the function as a new one. We show by experiments (Fig.7) that the learned residual functions in general have small responses, suggesting that identity mappings provide reasonable preconditioning.

在真实案例中，恒等函数不可能使最佳函数，但我们的改变可能可以帮助问题的预处理。如果最佳函数与恒等函数接近，而不是与零映射接近，那么残差学习会更容易一些。我们通过实验表明（图7）学习到的残差函数一般响应很小，表明恒等映射是一个合理的预处理。

### 3.2. Identity Mapping by Shortcuts 通过捷径实现恒等映射

We adopt residual learning to every few stacked layers. A building block is shown in Fig. 2. Formally, in this paper we consider a building block defined as:

我们每隔几层就会进行一次残差学习，构建模块如图2所示。本文中我们将一个构建模块定义为

$$y = F(x,\{ W_i \}) + x.$$(1)

Here *x* and *y* are the input and output vectors of the layers considered. The function $F(x,\{ W_i \})$ represents the residual mapping to be learned. For the example in Fig. 2 that has two layers, $F = W_2 σ(W_1 x)$ in which σ denotes ReLU [29] and the biases are omitted for simplifying notations. The operation *F + x* is performed by a shortcut connection and element-wise addition. We adopt the second nonlinearity after the addition (i.e., σ(y), see Fig. 2).

这里*x*和*y*为输入和输出矢量。函数$F(x,\{ W_i \})$代表要学习的残差映射。如图2，共有2层，$F = W_2 σ(W_1 x)$，这里σ代表ReLU[29]，为简化表达省略了偏置。操作*F + x*通过捷径连接实现，即逐元素相加。我们在相加之后再进行非线性处理（即σ(y)，见图2）。

The shortcut connections in Eqn.(1) introduce neither extra parameter nor computation complexity. This is not only attractive in practice but also important in our comparisons between plain and residual networks. We can fairly compare plain/residual networks that simultaneously have the same number of parameters, depth, width, and computational cost(except for the negligible element-wise addition).

式(1)中的捷径连接既没有引入额外的参数，也没有引入更多的计算复杂度。这在实践中不仅具有吸引力，而且在我们的比较试验中发现非常重要。普通网络和残差网络含有同样的参数、深度、宽度和计算复杂度（逐元素相加的运算量可以忽略），这样我们就可以进行公平的比较。

The dimensions of *x* and *F* must be equal in Eqn.(1). If this is not the case (e.g., when changing the input/output channels), we can perform a linear projection $W_s$ by the shortcut connections to match the dimensions:

*x*和*F*的维数在式(1)中相等。如果不相等（也就是输入输出的通道数不相等），我们就对输入进行一个线性变换$W_s$，使维度匹配：

$$y = F(x,\{ W_i \}) + W_s x.$$(2)

We can also use a square matrix $W_s$ in Eqn.(1). But we will show by experiments that the identity mapping is sufficient for addressing the degradation problem and is economical, and thus $W_s$ is only used when matching dimensions.

我们也可以在式(1)中使用一个方阵$W_s$，但试验说明恒等变换已经可以解决降质问题，而且节约计算量，所以$W_s$只在维度匹配的时候用。

The form of the residual function *F* is flexible. Experiments in this paper involve a function *F* that has two or three layers (Fig. 5), while more layers are possible. But if *F* has only a single layer, Eqn.(1) is similar to a linear layer: $y = W_1 x+x$, for which we have not observed advantages.

残差函数*F*的形式是灵活的。本文试验中的*F*都对应着两层或三层的映射，但更多的层也是可能的。但如果*F*只有一层，式(1)就和线性层很类似了$y = W_1 x+x$，我们也进行了试验但没有发现有改进。

We also note that although the above notations are about fully-connected layers for simplicity, they are applicable to convolutional layers. The function $F(x,\{ W_i \})$ can represent multiple convolutional layers. The element-wise addition is performed on two feature maps, channel by channel.

虽然上述表达式都是关于全连接层的，但也可以应用于卷积层。函数$F(x,\{ W_i \})$可以代表多卷积层，对应元素相加也可以是两个特征图的对应通道相加。

### 3.3. Network Architectures 网络架构

We have tested various plain/residual nets, and have observed consistent phenomena. To provide instances for discussion, we describe two models for ImageNet as follows.

我们测试了很多普通/残差网络，观察到的现象是一致的。下面我们给出两种模型(ImageNet)供讨论。

**Plain Network**. Our plain baselines (Fig. 3, middle) are mainly inspired by the philosophy of VGG nets [41] (Fig. 3, left). The convolutional layers mostly have 3×3 filters and follow two simple design rules: (i) for the same output feature map size, the layers have the same number of filters; and (ii) if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer. We perform downsampling directly by convolutional layers that have a stride of 2. The network ends with a global average pooling layer and a 1000-way fully-connected layer with softmax. The total number of weighted layers is 34 in Fig. 3 (middle).

**普通网络**。我们研究的普通网络（图3中间）主要受VGG网络的哲学所启发（图3左）。卷积层主要是3×3滤波器，设计原则主要是2个：(i)对于同样的输出特征图尺寸，卷积层的滤波器数量一样；(ii)如果特征图尺寸减半，其滤波器数量就加倍，以保持每层的时间复杂度。有的卷积层卷积步长为2，起到了下采样的作用。网络最后是一个全局平均pooling层和一个1000路全连接层和softmax处理。图3中间带权值的层总数为34。

It is worth noticing that our model has fewer filters and lower complexity than VGG nets [41] (Fig. 3, left). Our 34-layer baseline has 3.6 billion FLOPs (multiply-adds), which is only 18% of VGG-19 (19.6 billion FLOPs).

值得注意的是这个网络比VGG网络（图3左）滤波器少，复杂度低。我们的34层基准网络计算复杂度为36亿FLOPs（乘加运算），只有VGG-19（196亿FLOPs）的18%。

**Residual Network**. Based on the above plain network, we insert shortcut connections (Fig. 3, right) which turn the network into its counterpart residual version. The identity shortcuts (Eqn.(1)) can be directly used when the input and output are of the same dimensions (solid line shortcuts in Fig.3). When the dimensions increase(dotted line shortcuts in Fig. 3), we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

**残差网络**。基于以上的普通网络，我们插入了捷径连接（图3右），这就将网络变成了残差版。输入和输出维度相同时（图3中的实线捷径），恒等捷径（式1）可以直接使用。当维度增加（图3中的虚线捷径），我们考虑两种选项：(A)捷径仍然是恒等变换，增加的维度补零，这种选项不引入额外的参数；(B)使用式2的投影捷径来匹配维度（用1×1卷积）。这两种选项，当捷径跨越不同的特征图时，卷积步长为2。

Figure 3. Example network architectures for ImageNet. Left: the VGG-19 model [41] (19.6 billion FLOPs) as a reference. Middle: a plain network with 34 parameter layers (3.6 billion FLOPs). Right: a residual network with 34 parameter layers (3.6 billion FLOPs). The dotted shortcuts increase dimensions. Table 1 shows more details and other variants.

图3. 网络架构例子。左：VGG-19模型，196亿FLOPs，供参考。中：34参数层的普通网络，36亿FLOPs。右：34层的残差网络，36亿FLOPs。虚线捷径增加维度。表1为细节和其他变化。

### 3.4. Implementation

Our implementation for ImageNet follows the practice in [21, 41]. The image is resized with its shorter side randomly sampled in [256,480] for scale augmentation [41]. A 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted [21]. The standard color augmentation in [21] is used. We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16]. We initialize the weights as in [13] and train all plain/residual nets from scratch. We use SGD with a mini-batch size of 256. The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained for up to 60×$10^4$ iterations. We use a weight decay of 0.0001 and a momentum of 0.9. We do not use dropout [14], following the practice in [16].

我们的实现遵循[21, 41]的经验。图像尺寸变化的方法是其短边范围为[256,480]，这样来进行尺度扩充[41]。从图像或其翻转中随机剪切出224×224剪切块，减去每个像素的通道均值[21]。还采用了[21]中标准的色彩扩充方法。我们在每次卷积后激活前使用批归一化(batch normalization, BN)[16]。权值初始化方法见[13]，从头开始训练所有的普通/残差网络。我们用SGD优化，mini-batch size为256。学习速率初始化为0.1，错误率不再下降时，就除以10，模型训练迭代次数达60×$10^4$。我们使用权值衰减系数为0.0001，动量0.9。我们不使用dropout[14]，这一点与[16]相同。

In testing, for comparison studies we adopt the standard 10-crop testing [21]. For best results, we adopt the fully-convolutional form as in [41, 13], and average the scores at multiple scales (images are resized such that the shorter side is in {224,256,384,480,640}).

测试时，为了比较研究，我们采用了标准的10-crop测试[21]。为了得到最好结果，我们采用[41,13]中的全卷积形式，将不同尺度的得分进行平均（图像改变尺寸，其短边为{224,256,384,480,640}中的一个）。

## 4. Experiments
### 4.1. ImageNet Classification

We evaluate our method on the ImageNet 2012 classification dataset[36] that consists of 1000 classes. The models are trained on the 1.28 million training images, and evaluated on the 50k validation images. We also obtain a final result on the 100k test images, reported by the test server. We evaluate both top-1 and top-5 error rates.

我们在ImageNet-2012分类数据集上评估我们的方法，数据集包含1000类图像。模型在128万训练图像上进行训练，在5万验证图像上进行评估。我们还在10万测试图像上得到了最后结果，由测试服务器给出了top-1和top-5错误率。

**Plain Networks**. We first evaluate 18-layer and 34-layer plain nets. The 34-layer plain net is in Fig. 3 (middle). The 18-layer plain net is of a similar form. See Table 1 for detailed architectures.

**普通网络**。我们首先评估18层和34层的普通网络。34层网络的结构见图3中间，18层普通网络与之结构类似。表1给出了详细结构。

The results in Table 2 show that the deeper 34-layer plain net has higher validation error than the shallower 18-layer plain net. To reveal the reasons, in Fig. 4 (left) we compare their training/validation errors during the training procedure. We have observed the degradation problem - the 34-layer plain net has higher training error throughout the whole training procedure, even though the solution space of the 18-layer plain network is a subspace of that of the 34-layer one.

表2中较深的34层普通网络比较浅的18层普通网络验证错误率要高。为了找出原因，在图4左比较了它们在训练过程中的训练/验证错误率。我们观察了降质的问题，34层普通网络在整个训练过程中的训练错误率都更高一些，而18层普通的网络的解空间只是34层的一个子空间。

We argue that this optimization difficulty is unlikely to be caused by vanishing gradients. These plain networks are trained with BN [16], which ensures forward propagated signals to have non-zero variances. We also verify that the backward propagated gradients exhibit healthy norms with BN. So neither forward nor backward signals vanish. In fact, the 34-layer plain net is still able to achieve competitive accuracy (Table 3), suggesting that the solver works to some extent. We conjecture that the deep plain nets may have exponentially low convergence rates, which impact the reducing of the training error (We have experimented with more training iterations (3×) and still observed the degradation problem, suggesting that this problem cannot be feasibly addressed by simply using more iterations.). The reason for such optimization difficulties will be studied in the future.

我们认为其优化难度不是由于梯度消失造成的。普通网络训练时使用了BN[16]，这确保了前向传播的信号方差非零。我们还确认了后向传播的梯度在BN的处理下范数也非常健康。所以前向与后向的信号都没有消失。实际上，34层普通网络的正确率仍然是不错的（见表3），这说明在一定程度上仍然可以工作。我们推测，深度普通网络的收敛速度以指数级降低，这影响训练错误率的降低（我们试验了更多的训练迭代次数，3倍，还是能观察到降质问题，说明这个问题不能仅靠增加迭代次数解决）。这种优化难度的原因将来还要进行研究。

**Residual Networks**. Next we evaluate 18-layer and 34-layer residual nets (ResNets). The baseline architectures are the same as the above plain nets, except that a shortcut connection is added to each pair of 3×3 filters as in Fig. 3 (right). In the first comparison (Table 2 and Fig. 4 right), we use identity mapping for all shortcuts and zero-padding for increasing dimensions (option A). So they have no extra parameter compared to the plain counterparts.

**残差网络**。下面我们评价18层和34层残差网络（ResNets）。基准架构与上面的普通网络相同，但每两个3×3滤波器组后都增加了一个捷径连接，见图3右。在第一个比较中，就是表2和图4右，所有的捷径连接都采用恒等映射，对增加的维度都使用0补充（选项A），这样与普通网络比没有增加任何参数。

We have three major observations from Table 2 and Fig. 4. First, the situation is reversed with residual learning – the 34-layer ResNet is better than the 18-layer ResNet (by 2.8%). More importantly, the 34-layer ResNet exhibits considerably lower training error and is generalizable to the validation data. This indicates that the degradation problem is well addressed in this setting and we manage to obtain accuracy gains from increased depth.

通过表2和图4我们有三个主要的观察结果。首先，在残差网络下，情况得到了变化，34层ResNet比18层ResNet效果要好（错误率低了2.8%）。更重要的是，34层ResNet的训练错误率低了很多，还可以泛化到验证集。这说明降质的问题得到了很好的解决，通过增加网络深度我们得到了准确度的提升。

Table 2. Top-1 error (%, 10-crop testing) on ImageNet validation. Here the ResNets have no extra parameter compared to their plain counterparts. Fig. 4 shows the training procedures.

| | plain | ResNet
--- | --- | ---
18 layers | 27.94 | 27.88
34 layers | 28.54 | 25.03

Second, compared to its plain counterpart, the 34-layer ResNet reduces the top-1 error by 3.5% (Table 2), resulting from the successfully reduced training error (Fig. 4 right vs. left). This comparison verifies the effectiveness of residual learning on extremely deep systems.

第二，与普通网络相比，34层的ResNet其top-1错误率降低了3.5%（见表2），这是训练错误率降低的结果（图4左右对比）。这个比较验证了残差学习在极深系统的有效性。

Last, we also note that the 18-layer plain/residual nets are comparably accurate (Table 2), but the 18-layer ResNet converges faster (Fig. 4 right vs. left). When the net is “not overly deep” (18 layers here), the current SGD solver is still able to find good solutions to the plain net. In this case, the ResNet eases the optimization by providing faster convergence at the early stage.

最后，我们还注意到18层普通/残差网络的准确度是差距不大的（表2），但18层ResNet收敛更快一些（图4左右对比）。当网络不是极其深时（比如18层），现有的SGD优化器仍然是可以找到普通网络的较优解的。在这种情况下，ResNet在初期就收敛较快，从而使优化过程更容易。

**Identity vs. Projection Shortcuts**. We have shown that parameter-free, identity shortcuts help with training. Next we investigate projection shortcuts (Eqn.(2)). In Table 3 we compare three options: (A) zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameter-free (the same as Table 2 and Fig. 4 right); (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity; and (C) all shortcuts are projections.

**恒等映射与投影捷径**。我们已经知道无参数的恒等捷径对训练有益。下一步我们研究一下投影捷径（式2）。在表3中我们比较了3中选择：(A)增加维数时采用0填充捷径，所有的捷径都没有参数（与表2和图4右相同）；(B)增加维度时使用投影捷径，其他捷径为恒等；(C)所有捷径都是投影捷径。

Table 3 shows that all three options are considerably better than the plain counterpart. B is slightly better than A. We argue that this is because the zero-padded dimensions in A indeed have no residual learning. C is marginally better than B, and we attribute this to the extra parameters introduced by many (thirteen) projection shortcuts. But the small differences among A/B/C indicate that projection shortcuts are not essential for addressing the degradation problem. So we do not use option C in the rest of this paper, to reduce memory/time complexity and model sizes. Identity shortcuts are particularly important for not increasing the complexity of the bottleneck architectures that are introduced below.

表3的结果说明，所有三种选项都比普通网络要好很多。B略微比A好一些。我们认为是因为A中的0填充维度没有学习到残差。C比B也好一点，我们认为这是因为很多(13)投影捷径带来了额外的参数。但A/B/C之间细微的区别说明投影捷径对于解决降质问题不是最重要的。所以在本文剩下的试验中没有采用C选项，以减少内存/实践复杂度和模型尺寸。对于下面介绍的瓶颈结构来说，恒等捷径就非常重要，因为没有提升复杂度。

Table 3. Error rates (%, 10-crop testing) on ImageNet validation. VGG-16 is based on our test. ResNet-50/101/152 are of option B that only uses projections for increasing dimensions.

model | top-1 err. | top-5 err.
--- | --- | ---
VGG-16 [41] | 28.07 | 9.33
GoogLeNet [44] | - | 9.15
PReLU-net [13] | 24.27 | 7.38
plain-34 | 28.54 | 10.02
ResNet-34 A | 25.03 | 7.76
ResNet-34 B | 24.52 | 7.46
ResNet-34 C | 24.19 | 7.40
ResNet-50 | 22.85 | 6.71
ResNet-101 | 21.75 | 6.05
ResNet-152 | 21.43 | 5.71

**Deeper Bottleneck Architectures**. Next we describe our deeper nets for ImageNet. Because of concerns on the training time that we can afford, we modify the building block as a bottleneck design (Deeper non-bottleneck ResNets (e.g., Fig. 5 left) also gain accuracy from increased depth (as shown on CIFAR-10), but are not as economical as the bottleneck ResNets. So the usage of bottleneck designs is mainly due to practical considerations. We further note that the degradation problem of plain nets is also witnessed for the bottleneck designs). For each residual function *F*, we use a stack of 3 layers instead of 2 (Fig. 5). The three layers are 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions, leaving the 3×3 layer a bottleneck with smaller input/output dimensions. Fig. 5 shows an example, where both designs have similar time complexity.

**更深的瓶颈结构**。下面我们叙述一下更深的网络模型。因为考虑可以承受的训练时间，我们将组成结构修改成为瓶颈设计（更深的非瓶颈ResNet，比如图5左，在增加深度时也可以得到准确度的提升，这在CIFAR-10数据集上有体现，但是瓶颈ResNet更加节约计算量；所以瓶颈结构的设计主要是为了实践考虑；还要注意到，普通网络存在的降质问题，在瓶颈设计上同样存在）。对于每个残差函数*F*，我们使用3层堆叠而不是2层（如图5）。这三层分别是1×1,3×3和1×1的卷积，1×1层的主要作用是减少和恢复维数，3×3卷积层的输入/输出维数都较小，成为瓶颈。图5给出示例，这两种设计其时间复杂度类似。

Figure 5. A deeper residual function F for ImageNet. Left: a building block (on 56×56 feature maps) as in Fig. 3 for ResNet-34. Right: a “bottleneck” building block for ResNet-50/101/152.

The parameter-free identity shortcuts are particularly important for the bottleneck architectures. If the identity shortcut in Fig. 5 (right) is replaced with projection, one can show that the time complexity and model size are doubled, as the shortcut is connected to the two high-dimensional ends. So identity shortcuts lead to more efficient models for the bottleneck designs.

无参数的恒等捷径对于瓶颈架构非常重要，如果图5右恒等捷径换成投影捷径，可以看出其时间复杂度和模型大小都加倍，因为捷径与两个高维度端是连着的。所以对于瓶颈设计来说，恒等捷径可以使模型更有效率。

**50-layer ResNet**: We replace each 2-layer block in the 34-layer net with this 3-layer bottleneck block, resulting in a 50-layer ResNet (Table 1). We use option B for increasing dimensions. This model has 3.8 billion FLOPs.

**50层ResNet**。我们将34层模型的每个2层模块都换成3层的瓶颈模块，得到了50层的ResNet（见表1）。对于增加的维度，我们使用选项B。这个模型运算量为38亿FLOPs。

**101-layer and 152-layer ResNets**: We construct 101-layer and 152-layer ResNets by using more 3-layer blocks (Table 1). Remarkably, although the depth is significantly increased, the 152-layer ResNet (11.3 billion FLOPs) still has lower complexity than VGG-16/19 nets (15.3/19.6 billion FLOPs).

**101层和152层ResNet**：我们用更多的3层瓶颈结构构建了101层和152层ResNet（见表1）。虽然深度增加了很多，但152层ResNet的计算量（113亿FLOPs）比VGG-16/19网络的复杂度仍然要低（153/196亿FLOPs）。

The 50/101/152-layer ResNets are more accurate than the 34-layer ones by considerable margins (Table 3 and 4). We do not observe the degradation problem and thus enjoy significant accuracy gains from considerably increased depth. The benefits of depth are witnessed for all evaluation metrics (Table 3 and 4).

50/101/152层ResNet比34层ResNet准确度高出不少（见表3和表4）。我们没有观察到降质问题。深度增加带来的准确度增加，在所有评估标准上都存在（见表3和表4）。

Table 4. Error rates (%) of single-model results on the ImageNet validation set (except† reported on the test set).

Method | top-1 err. | top-5 err.
--- | --- | ---
VGG [41] (ILSVRC’14) | - | 8.43 †
GoogLeNet [44] (ILSVRC’14) | - | 7.89
VGG [41] (v5) | 24.4 | 7.1
PReLU-net [13] | 21.59 | 5.71
BN-inception [16] | 21.99 | 5.81
ResNet-34 B | 21.84 | 5.71
ResNet-34 C | 21.53 | 5.60
ResNet-50 | 20.74 | 5.25
ResNet-101 | 19.87 | 4.60
ResNet-152 | 19.38 | 4.49

**Comparisons with State-of-the-art Methods**. In Table 4 we compare with the previous best single-model results. Our baseline 34-layer ResNets have achieved very competitive accuracy. Our 152-layer ResNet has a single-model top-5 validation error of 4.49%. This single-model result outperforms all previous ensemble results (Table 5). We combine six models of different depth to form an ensemble (only with two 152-layer ones at the time of submitting). This leads to 3.57% top-5 error on the test set (Table 5). This entry won the 1st place in ILSVRC 2015.

**与目前最好方法的对比**。在表4中我们与前面最好的单个模型进行了对比。我们基准的34层ResNet已经有了非常好的准确率表现，152层ResNet单模型的top-5验证错误率为4.49%。这个单模型结果超过了所有以前的集成算法结果（见表5）。我们综合了6个不同深度的模型形成集成模型（在提交时只有2个152层模型）。这在测试集上得到了top-5错误率3.57%的结果（表5），赢得了ILSVRC-2015第一名的结果。

Table 5. Error rates (%) of ensembles. The top-5 error is on the test set of ImageNet and reported by the test server.

Method | top-5 err. (test)
--- | ---
VGG [41] (ILSVRC’14) | 7.32
GoogLeNet [44] (ILSVRC’14) | 6.66
VGG [41] (v5) | 6.8
PReLU-net [13] | 4.94
BN-inception [16] | 4.82
ResNet (ILSVRC’15) | 3.57

### 4.2. CIFAR-10 and Analysis

We conducted more studies on the CIFAR-10 dataset [20], which consists of 50k training images and 10k testing images in 10 classes. We present experiments trained on the training set and evaluated on the test set. Our focus is on the behaviors of extremely deep networks, but not on pushing the state-of-the-art results, so we intentionally use simple architectures as follows.

我们在CIFAR-10数据集上进行了更多的研究，数据集包含5万训练图像，1万测试图像，共10类。我们的试验在训练集上训练网络，并在测试集上评价模型。我们关注的是极深网络的行为，而不是取得更好的准确率效果，所以我们故意使用了一些很简单的架构。

The plain/residual architectures follow the form in Fig. 3 (middle/right). The network inputs are 32×32 images, with the per-pixel mean subtracted. The first layer is 3×3 convolutions. Then we use a stack of 6n layers with 3×3 convolutions on the feature maps of sizes {32,16,8} respectively, with 2n layers for each feature map size. The numbers of filters are {16,32,64} respectively. The subsampling is performed by convolutions with a stride of 2. The network ends with a global average pooling, a 10-way fully-connected
layer, and softmax. There are totally 6n+2 stacked weighted layers. The following table summarizes the architecture:

普通/残差架构如图3中、右所示，网络输入32×32大小图像，减去每个像素的均值。第一层是3×3卷积层，然后使用6n层叠加，卷积核3×3，特征图大小分别为{32,16,8}，每个特征图大小对应2n层。滤波器数量分别是{16,32,64}。降采样通过步长为2的卷积实现。网络最后是一个全局平均pooling层，一个10路全连接层和softmax层。共有6n+2个带权值层。下表总结了网络架构：

output map size | 32×32 | 16×16 | 8×8
--- | --- | --- | ---
layers | 1+2n | 2n | 2n
filters | 16 | 32 | 64

When shortcut connections are used, they are connected to the pairs of 3×3 layers (totally 3n shortcuts). On this dataset we use identity shortcuts in all cases (i.e., option A),so our residual models have exactly the same depth, width, and number of parameters as the plain counterparts.

当使用捷径连接时，与3×3卷积层连接（共3n个捷径）。在这个数据集上我们全部使用恒等捷径（即，选项A），所以我们的残差模型与普通模型的深度、宽度、参数数量严格相等。

We use a weight decay of 0.0001 and momentum of 0.9, and adopt the weight initialization in [13] and BN [16] but with no dropout. These models are trained with a mini-batch size of 128 on two GPUs. We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations, and terminate training at 64k iterations, which is determined on a 45k/5k train/val split. We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side, and a 32×32 crop is randomly sampled from the padded image or its horizontal flip. For testing, we only evaluate the single view of the original 32×32 image.

我们使用权值衰减，系数0.0001，动量0.9，采用[13]里的权值初始化方案，[16]中的BN方案，但没有使用dropout。模型训练的mini-batch size为128，训练在2个GPU上进行。学习速率开始是0.1, 32k和48k个迭代后除以10, 64k迭代后结束训练，这是在45k/5k的训练/验证分割集上确定的参数。我们采用[24]中的简单数据扩充方案进行训练：每个边上填充4个像素，填充后的图像（或其翻转）随机进行32×32剪切。在测试中，我们只评估原始32×32图像的单视图。

We compare n = {3,5,7,9}, leading to 20, 32, 44, and 56-layer networks. Fig. 6 (left) shows the behaviors of the plain nets. The deep plain nets suffer from increased depth, and exhibit higher training error when going deeper. This phenomenon is similar to that on ImageNet (Fig. 4, left) and on MNIST (see [42]), suggesting that such an optimization difficulty is a fundamental problem.

我们比较了n = {3,5,7,9}的情况，对应着20、32、44、56层网络。图6左是普通网络的表现。深度普通网络增加深度没有得益，反而得到更高的训练错误率。这个现象与ImageNet（图4左）和MNIST（见[42]）上类似，说明优化难度是一个基本问题。

Fig. 6 (middle) shows the behaviors of ResNets. Also similar to the ImageNet cases (Fig. 4, right), our ResNets manage to overcome the optimization difficulty and demonstrate accuracy gains when the depth increases.

图6中间所示的ResNet的表现。与ImageNet的情况类似（图4右），我们的ResNet克服了优化难度问题，深度增加时提高了准确度。

We further explore n = 18 that leads to a 110-layer ResNet. In this case, we find that the initial learning rate of 0.1 is slightly too large to start converging (With an initial learning rate of 0.1, it starts converging (<90% error) after several epochs, but still reaches similar accuracy). So we use 0.01 to warm up the training until the training error is below 80% (about 400 iterations), and then go back to 0.1 and continue training. The rest of the learning schedule is as done previously. This 110-layer network converges well (Fig. 6, middle). It has fewer parameters than other deep and thin networks such as FitNet [35] and Highway [42] (Table 6), yet is among the state-of-the-art results (6.43%, Table 6).

我们还研究了n=18，即110层ResNet的情况。在这种情况下，我们发现初始学习速率0.1有些略大，而不能收敛（如果初始学习速率为0.1，几个epoch之后开始收敛到小于90%的错误率，但最后仍然得到了类似的准确率）。所以我们用0.01来进行训练，直到训练错误率低于80%（大约400次迭代后），然后回到0.1的学习速率，继续训练。剩下的学习按照以前的来进行。这个110层的网络最后收敛的很好（图6中）。它比其他网络如FitNet[35]和Highway[42]参数更少（见表6），但是结果却是最好的（见表6,6.43%）。

Figure 6. Training on CIFAR-10. Dashed lines denote training error, and bold lines denote testing error. Left: plain networks. The error of plain-110 is higher than 60% and not displayed. Middle: ResNets. Right: ResNets with 110 and 1202 layers.

Table 6. Classification error on the CIFAR-10 test set. All methods are with data augmentation. For ResNet-110, we run it 5 times and show “best (mean±std)” as in [43].

**Analysis of Layer Responses**. Fig. 7 shows the standard deviations (std) of the layer responses. The responses are the outputs of each 3×3 layer, after BN and before other nonlinearity (ReLU/addition). For ResNets, this analysis reveals the response strength of the residual functions. Fig. 7 shows that ResNets have generally smaller responses than their plain counterparts. These results support our basic motivation (Sec.3.1) that the residual functions might be generally closer to zero than the non-residual functions. We also notice that the deeper ResNet has smaller magnitudes of responses, as evidenced by the comparisons among ResNet-20, 56, and 110 in Fig. 7. When there are more layers, an individual layer of ResNets tends to modify the signal less.

**层响应分析**。图7所示的是层响应的标准差(std)。响应是3×3滤波器层的输出，经过BN，在其他非线性(ReLU/addition)处理前。对于ResNet，这项分析是残差函数的响应强度。图7说明，ResNet的响应比普通网络对应的层要小。这个结果支持了我们的基本动机（见3.1节），即残差函数比非残差函数一般更接近0。我们还注意到，更深的ResNet其响应强度更小，图7中ResNet-20,56,110的比较都得到了验证。当层数更多时，每个单独的ResNet层倾向于更少改变信号。

**Exploring Over 1000 layers**. We explore an aggressively deep model of over 1000 layers. We set n = 200 that leads to a 1202-layer network, which is trained as described above. Our method shows no optimization difficulty, and this $10^3$-layer network is able to achieve training error <0.1% (Fig. 6, right). Its test error is still fairly good (7.93%, Table 6).

**探索超过1000层**。我们研究了一个非常激进的模型，超过了1000层。我们令n=200，这就形成了1202层的网络，训练过程和上面叙述的一样。我们的方法没有遇到优化难题，这个超过1000层的网络可以得到小于0.1%的训练错误率（图6右），测试错误率还好（7.93%，表6）。

But there are still open problems on such aggressively deep models. The testing result of this 1202-layer network is worse than that of our 110-layer network, although both have similar training error. We argue that this is because of overfitting. The 1202-layer network may be unnecessarily large (19.4M) for this small dataset. Strong regularization such as maxout [10] or dropout [14] is applied to obtain the best results ([10, 25, 24, 35]) on this dataset. In this paper, we use no maxout/dropout and just simply impose regularization via deep and thin architectures by design, without distracting from the focus on the difficulties of optimization. But combining with stronger regularization may improve results, which we will study in the future.

但这样激进的深度模型还是有一些未解的问题的。这个1202层网络的测试结果比110层网络要差，但训练错误率很接近。我们认为这是因为过拟合的原因。1202层网络对于这个小数据集可能太大了（参数1940万）。在这个数据集上需要使用强正则化比如maxout[10]和dropout[14]来得到最好结果([10, 25, 24, 35])。在本文中，我们没有使用maxout/dropout，只用了深度瘦架构设计中包含的简单正则化，而主要关注在优化难度这个问题中。更强的正则化应该会改进结果，这个我们在将来进行研究。

### 4.3. Object Detection on PASCAL and MS COCO

Our method has good generalization performance on other recognition tasks. Table 7 and 8 show the object detection baseline results on PASCAL VOC 2007 and 2012 [5] and COCO [26]. We adopt Faster R-CNN [32] as the detection method. Here we are interested in the improvements of replacing VGG-16 [41] with ResNet-101. The detection implementation (see appendix) of using both models is the same, so the gains can only be attributed to better networks. Most remarkably, on the challenging COCO dataset we obtain a 6.0% increase in COCO’s standard metric (mAP@[.5, .95]), which is a 28% relative improvement. This gain is solely due to the learned representations.

我们的方法在其他识别任务中的泛化性能也非常好。表7和表8所示的是在PASCAL VOC 2007和2012[5]和COCO[26]上的目标检测基准结果。我们采用了Faster R-CNN [32]作为检测方法。这里我们感兴趣的是，将VGG-16[41]改为ResNet-101带来的改进。使用两种模型的检测的实现（见附录）是一样的，所以改进肯定是由更好的网络带来的。值得关注的是，在COCO数据集上我们得到了6.0%的性能提升，相比COCO的标准度量(mAP@[.5, .95])，这是28%的相对改进。这个提升仅仅是由于学习到的表示。

Table 7. Object detection mAP (%) on the PASCAL VOC 2007/2012 test sets using baseline Faster R-CNN. See also Table 10 and 11 for better results.

training data | 07+12 | 07++12
--- | --- | ---
test data VOC | 07 test | VOC 12 test
VGG-16 | 73.2 | 70.4
ResNet-101 | 76.4 | 73.8

Table 8. Object detection mAP (%) on the COCO validation set using baseline Faster R-CNN. See also Table 9 for better results.

metric | mAP@.5 | mAP@[.5, .95]
--- | --- | ---
VGG-16 | 41.5 | 21.2
ResNet-101 | 48.4 | 27.2

Based on deep residual nets, we won the 1st places in several tracks in ILSVRC & COCO 2015 competitions: ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. The details are in the appendix.

基于深度残差网络，我们在ILSVRC & COCO 2015数个比赛中赢得了第一的位置：ImageNet检测，ImageNet定位，COCO检测，COCO分割。详见附录。

## References

fifty papers

## A. Object Detection Baselines
## B. Object Detection Improvements
## C. ImageNet Localization