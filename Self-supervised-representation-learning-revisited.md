# Revisiting Self-Supervised Visual Representation Learning

Alexander Kolesnikov et al. Google Brain

## 0. Abstract

Unsupervised visual representation learning remains a largely unsolved problem in computer vision research. Among a big body of recently proposed approaches for unsupervised learning of visual representations, a class of self-supervised techniques achieves superior performance on many challenging benchmarks. A large number of the pretext tasks for self-supervised learning have been studied, but other important aspects, such as the choice of convolutional neural networks (CNN), has not received equal attention. Therefore, we revisit numerous previously proposed self-supervised models, conduct a thorough large scale study and, as a result, uncover multiple crucial insights. We challenge a number of common practices in self-supervised visual representation learning and observe that standard recipes for CNN design do not always translate to self-supervised representation learning. As part of our study, we drastically boost the performance of previously proposed techniques and outperform previously published state-of-the-art results by a large margin.

无监督视觉表示学习在计算机视觉研究中基本上仍然是一个未解决的问题。最近提出的视觉表示无监督学习很多文章中，一类自监督技术在很多有挑战的基准测试中得到了很好的性能。大量自监督学习的pretext任务得到了研究，但其他重要的方面，如CNN的选择，没有得到同样的关注。因此，我们重新回顾了很多之前提出的自监督模型，进行了彻底的大规模研究，揭开了很多关键的洞见。我们在自监督的视觉表示学习中挑战了一些常见的方法，观察到CNN设计的标准方法，不一定能够用到自监督表示学习中。作为我们研究的一部分，我们极大了提高了之前提出的方法的性能，超过了之前发表的最好的结果很多。

## 1. Introduction

Automated computer vision systems have recently made drastic progress. Many models for tackling challenging tasks such as object recognition, semantic segmentation or object detection can now compete with humans on complex visual benchmarks [15, 48, 14]. However, the success of such systems hinges on a large amount of labeled data, which is not always available and often prohibitively expensive to acquire. Moreover, these systems are tailored to specific scenarios, e.g. a model trained on the ImageNet (ILSVRC-2012) dataset [41] can only recognize 1000 semantic categories or a model that was trained to perceive road traffic at daylight may not work in darkness [5, 4].

自动计算机视觉系统最近有了很大进展。处理挑战性任务的很多模型，如目标识别，语义分割，或目标检测，现在可以在复杂的视觉基准检测中与人类相媲美。但是，这种系统的成功依赖于大量标注的数据，这并不总是可用的，要获得这样的标注通常代价很大。而且，这些系统对特定的场景定制的，如，在ImageNet上训练的模型，只能识别1000种类别，一个模型训练用于在白天感知道路叫停，在晚上就可能无法工作。

As a result, a large research effort is currently focused on systems that can adapt to new conditions without leveraging a large amount of expensive supervision. This effort includes recent advances on transfer learning, domain adaptation, semi-supervised, weakly-supervised and unsupervised learning. In this paper, we concentrate on self-supervised visual representation learning, which is a promising sub-class of unsupervised learning. Self-supervised learning techniques produce state-of-the-art unsupervised representations on standard computer vision benchmarks [11, 37, 3].

结果是，很多的研究努力目前关注的都是，在无需利用大量昂贵的监督的情况下，系统可以适用于新的环境。这种努力包括最近在迁移学习，领域适应，半监督，弱监督和无监督领域的进展。本文中，我们聚焦在自监督的视觉表示学习，这是无监督学习中很有希望的一个类别。自监督学习技术在标准计算机视觉基准测试中，可以得到目前最好的无监督表示。

The self-supervised learning framework requires only unlabeled data in order to formulate a pretext learning task such as predicting context [7] or image rotation [11], for which a target objective can be computed without supervision. These pretext tasks must be designed in such a way that high-level image understanding is useful for solving them. As a result, the intermediate layers of convolutional neural networks (CNNs) trained for solving these pretext tasks encode high-level semantic visual representations that are useful for solving downstream tasks of interest, such as image recognition.

自监督学习框架，只需要未标注的数据，以对一个pretext学习任务进行表述，如预测上下文[7]，或图像旋转[11]，其中目标函数可以无需监督的进行计算。这些pretext任务，需要这样进行设计，即高层图像理解可用于解决这个任务。结果是，训练用于解决这些pretext任务的CNN的中间层，编码了高层语义视觉表示，可用于解决下游的感兴趣任务，如图像识别。

Most of the prior work, which aims at improving performance of self-supervised techniques, does so by proposing novel pretext tasks and showing that they result in improved representations. Instead, we propose to have a closer look at CNN architectures. We revisit a prominent subset of the previously proposed pretext tasks and perform a large-scale empirical study using various architectures as base models. As a result of this study, we uncover numerous crucial insights. The most important are summarized as follows:

很多之前的工作，其目的是改进自监督技术的性能，通过提出新的pretext任务，表明他们可以得到改进的表示，达成这样的目的。而我们则提出近距离的观察一下CNN架构。我们重新研究了之前提出的很多pretext任务，使用这种架构作为基准模型，进行了大规模经验性的研究。本文研究的结果，我们揭示了几个关键的洞见。最重要的我们总结如下：

- Standard architecture design recipes do not necessarily translate from the fully-supervised to the self-supervised setting. Architecture choices which negligibly affect performance in the fully labeled setting, may significantly affect performance in the self-supervised setting. 标准的架构设计，不一定可以从全监督的设置，泛化到自监督的设置。在全标注的设置中，可以忽略的影响性能的架构设计，在自监督的设置中回显著影响性能。

- In contrast to previous observations with the AlexNet architecture [11, 51, 34], the quality of learned representations in CNN architectures with skip-connections does not degrade towards the end of the model. 与之前的AlexNet相比，带有跳跃连接的CNN架构学习到的表示的质量，在趋向于模型最后的时候，不会下降。

- Increasing the number of filters in a CNN model and, consequently, the size of the representation significantly and consistently increases the quality of the learned visual representations. 增加CNN模型中的滤波器的数量的结果是，表示的大小对学习到的视觉表示的质量有显著的一致的提升。

- The evaluation procedure, where a linear model is trained on a fixed visual representation using stochastic gradient descent, is sensitive to the learning rate schedule and may take many epochs to converge. 评估过程，即在一个固定的视觉表示上使用SGD训练得到的线性模型，对于学习速率计划是敏感的，需要很多轮的训练才能收敛。

In Section 4 we present experimental results supporting the above observations and offer additional in-depth insights into the self-supervised learning setting. We make the code for reproducing our core experimental results publicly available. 在第4部分中，我们给出试验性的结果，支持上述观察结果，并在自监督学习设置中给出额外的深度洞见。我们核心试验结果的复现代码已经开源。

In our study we obtain new state-of-the-art results for visual representations learned without labeled data. Interestingly, the context prediction [7] technique that sparked the interest in self-supervised visual representation learning and that serves as the baseline for follow-up research, outperforms all currently published results (among papers on self-supervised learning) if the appropriate CNN architecture is used.

在我们的研究中，我们在不需要标注数据学习到的视觉表示方面，得到了新的目前最好的结果。有趣的是，上下文预测[7]技术点燃了自监督视觉表示学习的兴趣，是后续研究的基准，如果使用合适的CNN架构，其结果反而超过了目前发表的所有结果（在自监督学习方面的文章中）。

## 2. Related Work

Self-supervision is a learning framework in which a supervised signal for a pretext task is created automatically, in an effort to learn representations that are useful for solving real-world downstream tasks. Being a generic framework, self-supervision enjoys a wide number of applications, ranging from robotics to image understanding.

自监督是一个学习框架，对一个pretext任务的监督信号是自动创建的，学习到的表示，对于解决真实世界的下游任务是有帮助的。作为一个通用框架，自监督有很多应用，从机器人到图像理解。

In robotics, both the result of interacting with the world, and the fact that multiple perception modalities simultaneously get sensory inputs are strong signals which can be exploited to create self-supervised tasks [22, 44, 29, 10]. 在机器人中，与世界交互的结果，和多种感知模态同时得到感知输入的事实，都是很强的信号，可以用于创建自监督的任务。

Similarly, when learning representation from videos, one can either make use of the synchronized cross-modality stream of audio, video, and potentially subtitles [38, 42, 26, 47], or of the consistency in the temporal dimension [44]. 类似的，当从视频中学习表示时，可以利用同步的跨模态的音频、视频信号，和可能的字幕信息，或时间维度中的信号一致性。

In this paper we focus on self-supervised techniques that learn from image databases. These techniques have demonstrated impressive results for learning high-level image representations. Inspired by unsupervised methods from the natural language processing domain which rely on predicting words from their context [31], Doersch et al. [7] proposed a practically successful pretext task of predicting the relative location of image patches. This work spawned a line of work in patch-based self-supervised visual representation learning methods. These include a model from [34] that predicts the permutation of a “jigsaw puzzle” created from the full image and recent follow-ups [32, 36].

本文中，我们关注的是自监督技术，从图像数据库中进行学习。这种技术已经在学习高层图像表示中证明了有让人印象深刻的结果。NLP领域的无监督学习方法依靠上下文预测单词，受其启发，Doersch等[7]提出了一种在实践中很成功的pretext任务，预测图像块的相对位置。这篇文章产生了一系列工作，都是基于图像块的自监督视觉表示学习方法。这包括从[34]中的模型，预测一个拼图游戏的排列，拼图是从完整图像创建的，最近也有一些后续研究。

In contrast to patch-based methods, some methods generate cleverly designed image-level classification tasks. For instance, in [11] Gidaris et al. propose to randomly rotate an image by one of four possible angles and let the model predict that rotation. Another way to create class labels is to use clustering of the images [3]. Yet another class of pretext tasks contains tasks with dense spatial outputs. Some prominent examples are image inpainting [40], image colorization [50], its improved variant split-brain [51] and motion segmentation prediction [39]. Other methods instead enforce structural constraints on the representation space. Noroozi et al. propose an equivariance relation to match the sum of multiple tiled representations to a single scaled representation [35]. Authors of [37] propose to predict future patches in representation space via autoregressive predictive coding.

与基于图像块的方法相比，一些方法产生了设计很巧妙的图像层次的分类任务。比如，在[11]中，Gidaris等提出在四个角度中随机旋转一幅图像，让模型预测其旋转。另一种创建类别标签的方法，是使用图像的聚类[3]。而另一类pretext任务，则有密集的空间输出。一些重要的例子包括，图像修补[40]，图像上色[50]，其改进的大脑分裂变体[51]，和运动分割预测[39]。其他的方法则对表示空间加入了结构约束。Noroozi等提出一种等价关系，将多个排列好的表示的求和，与单个缩放的表示相匹配[35]。[37]的作者提出通过自回归预测编码在表示空间预测未来的图像块。

Our work is complimentary to the previously discussed methods, which introduce new pretext tasks, since we show how existing self-supervision methods can significantly benefit from our insights. 我们的工作与之前讨论的方法是互补的，提出了新的pretext任务，因为我们证明了，现有的自监督方法可以从我们的洞见中显著获益。

Finally, many works have tried to combine multiple pretext tasks in one way or another. For instance, Kim et al. extend the “jigsaw puzzle” task by combining it with colorization and inpainting in [25]. Combining the jigsaw puzzle task with clustering-based pseudo labels as in [3] leads to the method called Jigsaw++ [36]. Doersch and Zisserman [8] implement four different self-supervision methods and make one single neural network learn all of them in a multi-task setting.

最后，很多工作尝试了，将多个pretext任务结合成一种方法，或另外一种。比如，Kim等拓展了拼图游戏，将其与上色和修补任务结合到一起[25]。将拼图游戏任务与基于聚类的伪标签结合到一起[3]，得到了称为Jigsaw++的方法[36]。Doersch和Zisserman实现了四种不同的自监督方法，并用一个神经网络在一个多任务设置中学习所有方法。

The latter work is similar to ours since it contains a comparison of different self-supervision methods using a unified neural network architecture, but with the goal of combining all these tasks into a single self-supervision task. The authors use a modified ResNet101 architecture [16] without further investigation and explore the combination of multiple tasks, whereas our focus lies on investigating the influence of architecture design on the representation quality.

后者的工作与我们的类似，因为包含了不同的自监督方法的对比，使用了统一的神经网络架构，但其目标是将所有这些任务结合成一个自监督的任务。作者使用了ResNet101架构的变体，没有进一步的调查并探索多种任务的结合，而我们的关注点则在研究架构设计对表示质量的影像。

## 3. Self-supervised study setup

In this section we describe the setup of our study and motivate our key choices. We begin by introducing six CNN models in Section 3.1 and proceed by describing the four self-supervised learning approaches used in our study in Section 3.2. Subsequently, we define our evaluation metrics and datasets in Sections 3.3 and 3.4. Further implementation details can be found in Supplementary Material.

本节中，我们描述了我们研究的设置，并推动我们的核心选择。我们开始在3.1节中先介绍了6种CNN模型，然后在3.2节中介绍了在我们研究中使用的四种自监督学习方法。后来，我们在3.3节中定义了我们的评估度量标准，3.4节中定义了数据集。在附加材料中有更进一步的实现细节。

### 3.1. Architectures of CNN models

A large part of the self-supervised techniques for visual representation approaches uses AlexNet [27] architecture. In our study, we investigate whether the landscape of self-supervision techniques changes when using modern network architectures. Thus, we employ variants of ResNet and a batch-normalized VGG architecture, all of which achieve high performance in the fully-supervised training setup. VGG is structurally close to AlexNet as it does not have skip-connections and uses fully-connected layers.

大部分视觉表示方法的自监督技术，使用的都是AlexNet架构。在我们的研究中，我们研究了，当我们使用现代网络架构时，自监督技术的性能会有什么变化。因此，我们采用ResNet的变体和一个批归一化的VGG架构，这些架构在全监督的训练设置中得到了很高的性能。VGG在结构上与AlexNet相似，因为其没有跳跃连接，使用全连接层。

In our preliminary experiments, we observed an intriguing property of ResNet models: the quality of the representations they learn does not degrade towards the end of the network (see Section 4.5). We hypothesize that this is a result of skip-connections making residual units invertible under certain circumstances [2], hence facilitating the preservation of information across the depth even when it is irrelevant for the pretext task. Based on this hypothesis, we include RevNets [12] into our study, which come with stronger invertibility guarantees while being structurally similar to ResNets.

在我们的初步试验中，我们观察到了ResNet模型令人好奇的一个性质：学习到的表示的质量，在趋向于网络最后的时候，不会降低（见4.5节）。我们假设，这是跳跃连接的结果，使残差单元在某种条件下可逆[2]，因此促进了网络不同深度中的信息的保持，即使这与pretext任务是无关的。基于这种假设，我们将RevNets纳入到研究之中，其架构与ResNets类似，而且有更强的可逆性保证。

**ResNet** was introduced by He et al. [16], and we use the width-parametrization proposed in [49]: the first 7 × 7 convolutional layer outputs 16 × k channels, where k is the widening factor, defaulting to 4. This is followed by a series of residual units of the form y := x + F(x), where F is a residual function consisting of multiple convolutions, ReLU non-linearities [33] and batch normalization layers [20]. The variant we use, ResNet50, consists of four blocks with 3, 4, 6, and 3 such units respectively, and we refer to the output of each block as block1, block2, etc. The network ends with a global spatial average pooling producing a vector of size 512 × k, which we call pre-logits as it is followed only by the final, task-specific logits layer. More details on this architecture are provided in [16].

**ResNet**由He等人提出，我们使用[49]中的宽度参数化版：第一个7×7的卷积层输出16×k个通道，其中k是加宽因子，默认是4。然后接着是一系列残差单元，形式是y := x + F(x)，其中F是残差函数，包含多个卷积、ReLU非线性函数和批归一化层。我们使用的变体是ResNet50，包含4个模块，分别有3、4、6、3个这样的单元，我们称每个模块的输出为block1，block2等。网络以全局空间平均池化层结束，生成了一个512×k维的向量，我们称之为pre-logits，因为后面就是最后的、与任务特定的logits层。这种架构的更多细节见[16]。

In our experiments we explore k ∈ {4, 8, 12, 16}, resulting in pre-logits of size 2048, 4096, 6144 and 8192 respectively. For some self-supervised techniques we skip configurations that do not fit into memory. 在我们的试验中，我们对k ∈ {4, 8, 12, 16}进行了试验，得到的pre-logits的大小分别为2048, 4096, 6144和8192。对于一些自监督的技术，我们对于内存占用的模型就跳过这种配置。

Moreover, we analyze the sensitivity of the self-supervised setting to underlying architectural details by using two variants of ordering operations known as ResNet v1 [16] and ResNet v2 [17] as well as a variant without ReLU preceding the global average pooling, which we mark by a “(-)”. Notably, these variants perform similarly on the pretext task.

而且，我们分析了自监督设置对潜在的架构细节的敏感性，使用了两种变体，我们称之为ResNet v1[16]和ResNet v2[17]，以及一种在全局平均池化层之前没有ReLU的变体，我们以“(-)”进行标记。值得注意的是，这些变体在pretext任务上表现类似。

**RevNet** slightly modifies the design of the residual unit such that it becomes analytically invertible [12]. We note that the residual unit used in [12] is equivalent to double application of the residual unit from [21] or [6]. Thus, for conceptual simplicity, we employ the latter type of unit, which can be defined as follows. The input x is split channel-wise into two equal parts x1 and x2. The output y is then the concatenation of y2 := x2 and y1 := x1 + F(x2).

**RevNet**对残差单元的设计进行了一些修改，这样其变得在解析上是可逆的[12]。我们注意到，[12]中使用的残差单元与[21]或[6]中的残差单元的双重应用是等价的。因此，为了概念上的简洁性，我们采用后者这个类型的单元，可以定义如下。输入x按照通道进行分割成两个相等的部分x1和x2，输出y是y2 := x2和y1 := x1 + F(x2)的拼接。

It easy to see that this residual unit is invertible, because its inverse can be computed in closed form as x2 = y2 and x1 = y1 − F(x2). 很容易看出，这种残差单元是可逆的，因为其逆可以计算为下面的封闭形式x2 = y2和x1 = y1 − F(x2)。

Apart from this slightly different residual unit, RevNet is structurally identical to ResNet and thus we use the same overall architecture and nomenclature for both. In our experiments we use RevNet50 network, that has the same depth and number of channels as the original Resnet50 model. In the fully labelled setting, RevNet performs only marginally worse than its architecturally equivalent ResNet.

除了这个些许不同的残差单元，RevNet与ResNet在架构上是完全相同的，因此我们使用相同的总体架构和命名系统。在我们的试验中，我们使用RevNet50网络，与原始的ResNet50模型有着相同的深度和通道数。在全标注的设置中，RevNet比架构上等价的ResNet的表现要稍差。

**VGG** as proposed in [45] consists of a series of 3 × 3 convolutions followed by ReLU non-linearities, arranged into blocks separated by max-pooling operations. The VGG19 variant we use has 5 such blocks of 2, 2, 4, 4, and 4 convolutions respectively. We follow the common practice of adding batch normalization between the convolutions and non-linearities.

**VGG**在[45]中提出，包含一系列3×3卷积层和ReLU非线性函数，其单元由max池化运算进行分离。我们使用的VGG19变体有5个这样的模块，分别有2,2,4,4,4个卷积层。我们按照通常的实践，在卷积和非线性之间增加批归一化。

In an effort to unify the nomenclature with ResNets, we introduce the widening factor k such that k = 8 corresponds to the architecture in [45], i.e. the initial convolution produces 8 × k channels and the fully-connected layers have 512 × k channels. Furthermore, we call the inputs to the second, third, fourth, and fifth max-pooling operations block1 to block4, respectively, and the input to the last fully-connected layer pre-logits.

为与ResNets统一命名系统，我们提出加宽系数k，这样k=8的时候，对应着[45]中的架构，即初始卷积生成8×k个通道，全连接层有512×k个通道。而且，我们称第2，3，4，5个最大池化运算的输入为block1到block4，最后的全连接层的输入为pre-logits。

### 3.2. Self-supervised techniques

In this section we describe the self-supervised techniques that are used in our study. 本节中，我们描述了在本研究中使用的自监督技术。

**Rotation** [11]: Gidaris et al. propose to produce 4 copies of a single image by rotating it by {0°, 90°, 180°, 270°} and let a single network predict the rotation which was applied — a 4-class classification task. Intuitively, a good model should learn to recognize canonical orientations of objects in natural images. Gidaris等人提出将图像旋转{0°, 90°, 180°, 270°}四个角度，产生图像的4幅拷贝，让一个网络来预测图像的旋转，这是一个4类的分类任务。从直觉上来说，一个好的模型应当可以识别自然图像中目标的标准方向。

**Exemplar** [9]: In this technique, every individual image corresponds to its own class, and multiple examples of it are generated by heavy random data augmentation such as translation, scaling, rotation, and contrast and color shifts. We use data augmentation mechanism from [46]. [8] proposes to use the triplet loss [43, 18] in order to scale this pretext task to a large number of images (hence, classes) present in the ImageNet dataset. The triplet loss avoids explicit class labels and, instead, encourages examples of the same image to have representations that are close in the Euclidean space while also being far from the representations of different images. Example representations are given by a 1000-dimensional logits layer. 在这种技术中，每个单独的图像对应着其自己的类别，通过很多随机图像扩增方法，可以生成很多个样本，如平移，缩放，旋转和对比度变化，色彩变化。我们使用[46]中的数据扩增机制。[8]提出使用三元组损失以对pretext任务进行缩放，到很多数量的图像（因此也是很多类别），也即ImageNet数据集中的图像数量。三元组损失可以不使用显式的类别标签，而是鼓励同样的图像的样本其表示在Euclidean空间中很接近，而与不同图像的表示很远。样本表示是由一个1000维的logits层给出的。

**Jigsaw** [34]: the task is to recover relative spatial position of 9 randomly sampled image patches after a random permutation of these patches was performed. All of these patches are sent through the same network, then their representations from the pre-logits layer are concatenated and passed through a two hidden layer fully-connected multi-layer perceptron (MLP), which needs to predict a permutation that was used. In practice, the fixed set of 100 permutations from [34] is used. 这个任务是恢复9个随机采样的图像块的相对位置，这些图像块是经过了随机组合的操作的。所有这些图像块都送入同样的网络，然后其pre-logits层的表示拼接到一起，送入一个两层的隐含层的全连接MLP中，预测其使用的排列组合。实践中，使用了[34]中的100个组合的固定集。

In order to avoid shortcuts relying on low-level image statistics such as chromatic aberration [34] or edge alignment, patches are sampled with a random gap between them. Each patch is then independently converted to grayscale with probability 2⁄3 and normalized to zero mean and unit standard deviation. More details on the preprocessing are provided in Supplementary Material. After training, we extract representations by averaging the representations of nine uniformly sampled, colorful, and normalized patches of an image. 为防止依赖于低层图像统计的捷径，如色彩反常[34]或边缘对齐，图像块的采样，各个块之间是有随机间隙的。每个图像块都以2/3的概率独立的转化到灰度上，并归一化到0均值，和单位标准方差。预处理上的更多细节见附加材料。在训练之后，我们通过对9个均匀采样的图像块，经过色彩变化、归一化的表示的平均，提取出表示。

**Relative Patch Location** [7]: The pretext task consists of predicting the relative location of two given patches of an image. The model is similar to the Jigsaw one, but in this case the 8 possible relative spatial relations between two patches need to be predicted, e.g. “below” or “on the right and above”. We use the same patch prepossessing as in the Jigsaw model and also extract final image representations by averaging representations of 9 cropped patches. 这个pretext任务包含预测给定的图像块的相对位置。这个模型与拼图模型类似，但在这个问题中，要预测的是两个图像块的8种相对空间位置关系，如，“在下面”，或“在右上”。我们使用拼图模型中相同的图像块预处理，并通过对9个图像块的表示的平均来提取最终的图像表示。

### 3.3. Evaluation of Learned Visual Representations

We follow common practice and evaluate the learned visual representations by using them for training a linear logistic regression model to solve multiclass image classification tasks requiring high-level scene understanding. These tasks are called downstream tasks. We extract the representation from the (frozen) network at the pre-logits level, but investigate other possibilities in Section 4.5.

我们按照常规做法，使用学习到的视觉表示，训练一个线性逻辑回归模型，以解决多类别图像分类问题，这需要高层场景理解的能力，这样对学习到的表示进行评估。这些任务称为下游任务。我们从冻结的网络中在pre-logits层提取表示，但在4.5节中研究其他可能性。

In order to enable fast evaluation, we use an efficient convex optimization technique for training the logistic regression model unless specified otherwise. Specifically, we precompute the visual representation for all training images and train the logistic regression using L-BFGS [30]. 为进行快速评估，我们使用一种高效的凸集优化技术来训练逻辑回归模型，除非另外指定。具体的，我们预先计算所有训练图像的视觉表示，使用L-BFGS来训练逻辑回归。

For consistency and fair evaluation, when comparing to the prior literature in Table 1, we opt for using stochastic gradient descent (SGD) with momentum and use data augmentation during training. 为一致和公平评估，我们在表1中与之前的文献进行了比较，我们选择使用带有动量的SGD，并在训练时使用了数据扩增。

We further investigate this common evaluation scheme in Section 4.3, where we use a more expressive model, which is an MLP with a single hidden layer with 1000 channels and the ReLU non-linearity after it. More details are given in Supplementary material. 我们进一步在4.3中研究了这种常见的评估方案，那里我们使用了一个更具有表现力的模型，那是一个MLP只有一个隐含层，有1000个通道，其后有ReLU非线性函数。更多细节详见附加材料。

### 3.4. Datasets

In our experiments, we consider two widely used image classification datasets: ImageNet and Places205. 在我们的试验中，我们使用了两个广泛使用的图像分类数据集：ImageNet和Places205。

ImageNet contains roughly 1.3million natural images that represent 1000 various semantic classes. There are 50 000 images in the official validation and test sets, but since the official test set is held private, results in the literature are reported on the validation set. In order to avoid overfitting to the official validation split, we report numbers on our own validation split (50 000 random images from the training split) for all our studies except in Table 2, where for a fair comparison with the literature we evaluate on the official validation set.

ImageNet大约包含130万自然图像，表示了1000种各种语义类别。在官方的验证和测试集中，有5万幅图像，但由于官方测试集是私有的，我们在验证集上给出文献的结果。为防止在官方的验证集上过拟合，我们给出我们自己的验证集分割的数量（从训练集中分出了5万幅随机图像），这在本文的所有研究中都是这样的，除了表2中，为了进行公平比较，我们在官方验证集上进行的评估。

The Places205 dataset consists of roughly 2.5million images depicting 205 different scene types such as airfield, kitchen, coast, etc. This dataset is qualitatively different from ImageNet and, thus, a good candidate for evaluating how well the learned representations generalize to new unseen data of different nature. We follow the same procedure as for ImageNet regarding validation splits for the same reasons.

Places205数据集包含大约250万图像，表示205种不同的场景类别，如机场，厨房，海岸等。这个数据集性质上与ImageNet不同，因此，要评估学习到的表示泛化到未曾见过的不同本质的数据上的好坏，这是一个很好的候选。我们按照在ImageNet上一样的过程，取相应的验证集。

## 4. Experiments and Results

In this section we present and interpret results of our large-scale study. All self-supervised models are trained on ImageNet (without labels) and consequently evaluated on our own hold-out validation splits of ImageNet and Places205. Only in Table 2, when we compare to the results from the prior literature, we use the official ImageNet and Places205 validation splits. 本节中，我们提出并解释我们的大规模研究的结果。所有的自监督模型都是在ImageNet上进行训练的（不需要标签），然后在我们自己的ImageNet和Places205保留验证集上进行评估。只有在表2中，当我们与之前文献的结果进行比较时，我们使用了官方的ImageNet和Places205验证集。

### 4.1. Evaluation on ImageNet and Places205

In Table 1 we highlight our main evaluation results: we measure the representation quality produced by six different CNN architectures with various widening factors (Section 3.1), trained using four self-supervised learning techniques (Section 3.2). We use the pre-logits of the trained self-supervised networks as representation. We follow the standard evaluation protocol (Section 3.3) which measures representation quality as the accuracy of a linear regression model trained and evaluated on the ImageNet dataset.

在表1中，我们强调了我们的主要评估结果：我们度量了6种不同的CNN架构在各种不同的加宽因子时的表示质量（3.1节），使用四种自监督学习技术进行的训练（3.2节）。我们使用训练得到的自监督网络的pre-logits作为表示。我们按照标准的评估协议（3.3节），将表示质量度量为线性回归模型的准确率，模型是在ImageNet数据集上进行的训练和评估。

Now we discuss key insights that can be learned from the table and motivate our further in-depth analysis. First, we observe that similar models often result in visual representations that have significantly different performance. Importantly, neither is the ranking of architectures consistent across different methods, nor is the ranking of methods consistent across architectures. For instance, the RevNet50 v2 model excels under Rotation self-supervision, but is not the best model in other scenarios. Similarly, relative patch location seems to be the best method when basing the comparison on the ResNet50 v1 architecture, but not otherwise. Notably, VGG19-BN consistently demonstrates the worst performance, even though it achieves performance similar to ResNet50 models on standard vision benchmarks [45]. Note that VGG19-BN performs better when using representations from layers earlier than the pre-logit layer are used, though still falls short. We investigate this in Section 4.5. We depict the performance of the models with the largest widening factor in Figure 2 (left), which displays these ranking inconsistencies.

现在我们讨论一下，从表格中可以学到的关键的洞见，以推动我们进一步的深度分析。首先，我们观察到类似的模型通常可以得到的视觉表示，会有明显不同的性能。重要的是，架构的排名在不同的模型也不是一致的，方法的排名与架构也不是一致的。比如，RevNet50 v2模型在Rotation的自监督中胜出，但在其他场景中并不是最好的。类似的，在基于ResNet50 v1架构时，相对图像块位置似乎是最好的方法，但其他的不是这样。值得注意的是，VGG19-BN在使用pre-logit层之前的层的表示时，有更好的效果，但仍然落后于其他模型。我们在4.5节中研究了这个。我们在图2左中给出了有最大宽度因子的模型性能，这显示了排名的不一致性。

Our second observation is that increasing the number of channels in CNN models improves performance of self-supervised models. While this finding is in line with the fully-supervised setting [49], we note that the benefit is more pronounced in the context of self-supervised representation learning, a fact not yet acknowledged in the literature. 我们的第二个观察是，增加CNN模型的通道数量，可以改进自监督模型的性能。这个发现与全监督的设置是一致的[49]，但我们要说明，这种获益在自监督表示学习下更明显，这在其他文献中尚未有说明。

We further evaluate how visual representations trained in a self-supervised manner on ImageNet generalize to other datasets. Specifically, we evaluate all our models on the Places205 dataset using the same evaluation protocol. The performance of models with the largest widening factor are reported in Figure 2 (right) and the full result table is provided in Supplementary Material. We observe the following pattern: ranking of models evaluated on Places205 is consistent with that of models evaluated on ImageNet, indicating that our findings generalize to new datasets.

我们进一步评估了，在ImageNet上以自监督的方式训练的视觉表示，泛化到其他数据集上的性能。具体的，我们评估我们的模型在Places205数据集上的性能，使用相同的评估协议。最大加宽因子模型的性能在表2右给出了，完整的结果表格在附录资料中给出。我们观察到下面的模式：在Place205上评估的模型的排名，与在ImageNet上评估的模型的顺序一致，说明我们的发现可以泛化到新的数据集上。

### 4.2. Comparison to prior work

In order to put our findings in context, we select the best model for each self-supervision from Table 1 and compare them to the numbers reported in the literature. For this experiment only, we precisely follow standard protocol by training the linear model with stochastic gradient descent (SGD) on the full ImageNet training split and evaluating it on the public validation set of both ImageNet and Places205. We note that in this case the learning rate schedule of the evaluation plays an important role, which we elaborate in Section 4.7.

为将我们的发现与上下文放到一起，我们从表1中选择了几个最好的自监督模型，将其与文献中给出的结果进行比较。只在这个试验中，我们精确的按照标准协议，使用SGD在完整的ImageNet训练集上训练线性模型，并在ImageNet和Places205的公开验证集上评估模型。我们要说明的是，在这种情况下，评估的学习率安排起到很大的作用，这在4.7中进行详细说明。

Table 2 summarizes our results. Surprisingly, as a result of selecting the right architecture for each self-supervision and increasing the widening factor, our models significantly outperform previously reported results. Notably, context prediction [7], one of the earliest published methods, achieves 51.4% top-1 accuracy on ImageNet. Our strongest model, using Rotation, attains unprecedentedly high accuracy of 55.4%. Similar observations hold when evaluating on Places205.

表2总结了我们的结果。令人惊讶的是，对每个自监督任务选择了正确的架构，增大了宽度因子，其结果是，我们的模型显著超过了之前给出的结果。值得注意的是，上下文预测[7]，这是最早发表的一种方法，在ImageNet上获得了51.4%的top-1准确率。我们最强的模型，使用Rotation，获得了最高的55.4%的准确率。在Places205上评估的时候，有类似的观测结果。

Importantly, our design choices result in almost halving the gap between previously published self-supervised result and fully-supervised results on two standard benchmarks. Overall, these results reinforce our main insight that in self-supervised learning architecture choice matters as much as choice of a pretext task.

重要的是，我们的设计选择，使得之前发表的在两个标准基准测试上的自监督结果，和全监督结果，差距减半了。总体上，这些结果加强了我们的主要洞见，即自监督学习架构的架构选择，与pretext任务的选择一样重要。

### 4.3. A linear model is adequate for evaluation.

Using a linear model for evaluating the quality of a representation requires that the information relevant to the evaluation task is linearly separable in representation space. This is not necessarily a prerequisite for a “useful” representation. Furthermore, using a more powerful model in the evaluation procedure might make the architecture choice for a self-supervised task less important. Hence, we consider an alternative evaluation scenario where we use a multi-layer perceptron (MLP) for solving the evaluation task, details of which are provided in Supplementary Material.

使用一个线性模型来评估表示的质量，需要相关的信息保证，即评估任务在表示空间是线性可分的。这对于一个有用的表示来说，并不是必要的。而且，在评估过程中，使用一种更加强力的模型，会使得自监督任务的架构选择没那么重要。因此，我们考虑一种替代评估场景，其中我们使用一个多层感知机(MLP)，求解评估任务，细节在附录资料中。

Figure 3 clearly shows that the MLP provides only marginal improvement over the linear evaluation and the relative performance of various settings is mostly unchanged. We thus conclude that the linear model is adequate for evaluation purposes. 图3清楚的表明了，MLP与线性评估模型相比，只有较小的改进，多数设置中的相对性能基本未变。因此我们得出结论，线性模型是足以进行评估的目的的。

### 4.4. Better performance on the pretext task does not always translate to better representations.

In many potential applications of self-supervised methods, we do not have access to downstream labels for evaluation. In that case, how can a practitioner decide which model to use? Is performance on the pretext task a good proxy? 在很多自监督方法的潜在应用，我们都没有下游应用的标签，以进行评估。在这种情况下，一个参与者怎么决定使用什么模型呢？在pretext任务上的性能是一个好的代理吗？

In Figure 4 we plot the performance on the pretext task against the evaluation on ImageNet. It turns out that performance on the pretext task is a good proxy only once the model architecture is fixed, but it can unfortunately not be used to reliably select the model architecture. Other label-free mechanisms for model-selection need to be devised, which we believe is an important and underexplored area for future work.

在图4中，我们画出了在pretext任务上的性能，与在ImageNet上评估的结果的比较。结果是，在pretext任务上的性能，只在模型架构固定的时候，是一个好的代理，但不能用于选择模型架构。需要设计出其他无标签的机制来选择模型，我们相信这是一个重要的、尚未探索的领域，是未来的工作。

### 4.5. Skip-connections prevent degradation of representation quality towards the end of CNNs.

We are interested in how representation quality depends on the layer choice and how skip-connections affect this dependency. Thus, we evaluate representations from five intermediate layers in three models: Resnet v2, RevNet and VGG19-BN. The results are summarized in Figure 5. 我们对表示质量与网络层的选择感兴趣，以及跳跃连接怎样影像这种依赖性。因此，我们对三个模型中的5个中间层中评估表示：Resnet v2, RevNet和VGG19-BN。

Similar to prior observations [11, 51, 34] for AlexNet [28], the quality of representations in VGG19-BN deteriorates towards the end of the network. We believe that this happens because the models specialize to the pretext task in the later layers and, consequently, discard more general semantic features present in the middle layers. 与[11,51,34]对AlexNet的观察类似，在VGG19-BN网络中的表示质量，随着向网络后端的推移，质量逐渐下降。我们相信，这是因为，模型专精于后面层的pretext任务，结果是，抛弃了一些中间层的通用语义特征。

In contrast, we observe that this is not the case for models with skip-connections: representation quality in ResNet consistently increases up to the final pre-logits layer. We hypothesize that this is a result of ResNet’s residual units being invertible under some conditions [2]. Invertible units preserve all information learned in intermediate layers, and, thus, prevent deterioration of representation quality.

比较起来，我们观察到，对于带有跳跃连接的模型，则没有这种情况：在ResNet中的表示质量一直增加，直到最终的pre-logits层。我们推测，这是ResNet的残差单元的结果，在某种条件下这是可逆的[2]。可逆单元保存了在中间层学习到的所有信息，因此，防止了表示质量的降低。

We further test this hypothesis by using the RevNet model that has stronger invertibility guarantees. Indeed, it boosts performance by more than 5% on the Rotation task, albeit it does not result in improvements across other tasks. We leave identifying further scenarios where Revnet models result in significant boost of performance for the future research.

我们通过使用有更强可逆性保证的ResNet模型，进一步测试了这种假设。确实，这会在Rotation任务中提升高达5%的性能，尽管在其他任务中并没有得到性能的改进。RevNet模型对哪些任务还有明显的性能提升呢，这个任务我们留在将来完成。

### 4.6. Model width and representation size strongly influence the representation quality.

Table 1 shows that using a wider network architecture consistently leads to better representation quality. It should be noted that increasing the network’s width has the side effect of also increasing the dimensionality of the final representation (Section 3.1). Hence, it is unclear whether the increase in performance is due to increased network capacity or to the use of higher-dimensional representations, or to the interplay of both.

表1给出了，使用更宽的网络架构，会持续带来更好的表示质量。应当说明，增加网络宽度有一些副作用，即增加了最终的表示的维度。因此，现在仍然不明确的是，性能的增加是因为网络容量增加了，还是因为使用了更高维度的表示，还是两者的相互作用导致的。

In order to answer this question, we take the best rotation model (RevNet50) and disentangle the network width from the representation size by adding an additional linear layer to control the size of the pre-logits layer. We then vary the widening factor and the representation size independently of each other, training each model from scratch on ImageNet with the Rotation pretext task. The results, evaluated on the ImageNet classification task, are shown in Figure 6. In essence, it is possible to increase performance by increasing either model capacity, or representation size, but increasing both jointly helps most. Notably, one can significantly boost performance of a very thin model from 31 % to 43 % by increasing representation size.

为回答这个问题，我们以最好的rotation模型（RevNet50），将网络宽度从表示大小中分离开来，增加了一个线性层，以控制pre-logits层的大小。我们然后使得宽度系数和表示大小独立变化，每个都从头在ImageNet上训练Rotation pretext任务。在ImageNet分类任务上评估得到的结果，如图6所示。基本上，通过增加网络容量，或表示的大小，都是可能增加性能的，但增加两者都增加，性能提升的幅度最大。值得注意的是，通过增加表示大小，可以将一个非常瘦小的模型的性能，从31%显著提升到43%。

**Low-data regime**. In principle, the effectiveness of increasing model capacity and representation size might only work on relatively large datasets for downstream evaluation, and might hurt representation usefulness in the low-data regime. In Figure 7, we depict how the number of channels affects the evaluation using both full and heavily subsampled (10 % and 5 %) ImageNet and Places205 datasets.

**数据较少的情况**。原则上，增加模型容量个表示大小的有效性，可能只在相对较大的数据集上有效，在很少的数据的情况时，可能会有损表示的有用性。在图7中，我们对ImageNet和Places205数据集进行了采样，两种采样中(10 % and 5 %)，我们对通道数量对评估的影响，放在图7中所示。

We observe that increasing the widening factor consistently boosts performance in both the full- and low-data regimes. We present more low-data evaluation experiments in Supplementary Material. This suggests that self-supervised learning techniques are likely to benefit from using CNNs with increased number of channels across wide range of scenarios.

我们观察到，增加宽度因子会持续提升性能，在全数据和少量数据的情况都是这样。我们在附加资料中给出了更多的少量数据评估试验结果。这说明，自监督的学习技术，如果使用的CNN的通道数量增加，则会在很多场景中可能得到性能提升。

### 4.7. SGD for training linear model takes long time to converge

In this section we investigate the importance of the SGD optimization schedule for training logistic regression in downstream tasks. We illustrate our findings for linear evaluation of the Rotation task, others behave the same and are provided in Supplementary Material.

本节中，我们研究了SGD优化方案对在下游任务中训练逻辑回归的重要性。我们在Rotation任务中的线性评估中描述了我们的发现，其他的表现类似，在附加材料中给出。

We train the linear evaluation models with a mini-batch size of 2048 and an initial learning rate of 0.1, which we decay twice by a factor of 10. Our initial experiments suggest that when the first decay is made has a large influence on the final accuracy. Thus, we vary the moment of first decay, applying it after 30, 120 or 480 epochs. After this first decay, we train for an extra 40 extra epochs, with a second decay after the first 20.

我们训练线性评估模型的mini-batch大小为2048，初始学习速率为0.1，两次进行衰减，系数为10。我们的初始试验表明，当第一次衰减时，对最终准确率有很大影响。因此，我们对第一次衰减的时机进行了变化，在30、120或480轮训练后进行。在这第一次衰减后，我们再训练另外40轮，在第一次20轮训练再一次进行衰减。

Figure 8 depicts how accuracy on our validation split progresses depending on when the learning rate is first decayed. Surprisingly, we observe that very long training (≈ 500 epochs) results in higher accuracy. Thus, we conclude that SGD optimization hyperparameters play an important role and need to be reported.

图8给出了验证过程的准确率，与第一次学习率衰减之间的关系。令人惊讶的是，我们观察到，非常长时间的训练(≈ 500 epochs)会得到更高的准确率。因此，我们得出结论说，SGD优化的超参数有很大的作用，需要具体给出。

## 5. Conclusion

In this work, we have investigated self-supervised visual representation learning from the previously unexplored angles. Doing so, we uncovered multiple important insights, namely that (1) lessons from architecture design in the fully-supervised setting do not necessarily translate to the self-supervised setting; (2) contrary to previously popular architectures like AlexNet, in residual architectures, the final pre-logits layer consistently results in the best performance; (3) the widening factor of CNNs has a drastic effect on performance of self-supervised techniques and (4) SGD training of linear logistic regression may require very long time to converge. In our study we demonstrated that performance of existing self-supervision techniques can be consistently boosted and that this leads to halving the gap between self-supervision and fully labeled supervision.

本文中，我们研究了自监督视觉表示学习，研究角度未在之前的文献中出现。这样，我们揭示了多个重要的洞见，即，(1)全监督设置中架构设计不一定会在自监督的设置中带来一样的结果，(2)与之前流行的架构如AlexNet相反，在残差架构中，最终的pre-logits层一直是性能最好的，(3)CNN的宽度因子在自监督技术中有很神奇的效果，(4)线性逻辑回归的SGD训练可能会需要很长时间才收敛。在我们的研究中，我们证明了，现有的自监督技术可以持续得到提升，会使得自监督和全标注监督的性能差异减半。

Most importantly, though, we reveal that neither is the ranking of architectures consistent across different methods, nor is the ranking of methods consistent across architectures. This implies that pretext tasks for self-supervised learning should not be considered in isolation, but in conjunction with underlying architectures.

最重要的是，我们揭示了，架构在不同方法中排序不是一致不变的，方法使用不同的架构的排序也不是不变的。这说明，自监督的pretext任务不应当单独考虑，而要与潜在的架构一起考虑。