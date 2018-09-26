# Xception: Deep Learning with Depthwise Separable Convolutions

Francois Chollet, Google, Inc.

## Abstract 摘要

We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.

我们提出了卷积神经网络中Inception模块的一个解释，是正常卷积和depthwise separable卷积（depthwise卷积后跟着一个pointwise卷积）操作之间的中间步骤。在这种情况下，depthwise separable卷积可以理解为很多数量塔层结构的Inception模块。这种观察使我们提出了一种受到Inception结构启发得到的新的深度卷积神经网络架构，其中Inception模块替换成了depthwise separable卷积。我们称这种框架为Xception，在ImageNet数据集上略微超过了Inception V3（而Inception V3就是为这个数据集设计的），在更大的一个图像分类数据集上，包括1.7万类的3.5亿张图像，大大超过了Inception V3的表现。由于Xception架构与Inception V3的参数数量一样，所以性能的提升不是由于增大的容量，而是由于模型参数使用效率更高。

## 1. Introduction 简介

Convolutional neural networks have emerged as the master algorithm in computer vision in recent years, and developing recipes for designing them has been a subject of considerable attention. The history of convolutional neural network design started with LeNet-style models [10], which were simple stacks of convolutions for feature extraction and max-pooling operations for spatial sub-sampling. In 2012, these ideas were refined into the AlexNet architecture [9], where convolution operations were being repeated multiple times in-between max-pooling operations, allowing the network to learn richer features at every spatial scale. What followed was a trend to make this style of network increasingly deeper, mostly driven by the yearly ILSVRC competition; first with Zeiler and Fergus in 2013 [25] and then with the VGG architecture in 2014 [18].

卷积神经网络已经成为近几年计算机视觉的终极算法，网络设计方法已经得到了广泛关注。卷积神经网络的设计历史开始于LeNet类模型[10]，也就是卷积层和最大池化层的叠加，卷积层用于特征提取，最大池化层用于空间降采样。在2012年，这些思想提炼出了AlexNet架构[9]，其中叠加了几个卷积层，中间间隔着最大池化层，使得网络可以在每个空间尺度上都学习更丰富的特征。接下来的趋势就是使这种风格的网络更深，主要是受每年的ILSVRC竞赛驱动；首先是2013年的Zeiler and Fergus[25]，然后是2014年的VGG架构[18]。

At this point a new style of network emerged, the Inception architecture, introduced by Szegedy et al. in 2014 [20] as GoogLeNet (Inception V1), later refined as Inception V2 [7], Inception V3 [21], and most recently Inception-ResNet [19]. Inception itself was inspired by the earlier Network-In-Network architecture [11]. Since its first introduction, Inception has been one of the best performing family of models on the ImageNet dataset [14], as well as internal datasets in use at Google, in particular JFT [5].

然后2014年提出了一种新式网络，也就是Szegedy et al.提出的Inception架构，首先是GoogLeNet，也就是Inception V1，后来提炼出了Inception V2 [7], Inception V3 [21], 还有最新的Inception-ResNet [19]。Inception是由更早一点的Network-In-Network架构[11]启发得到的。自从第一次提出，Inception就是ImageNet数据集上性能最佳的网络之一，在Google内部数据集也是这样，特别是JFT[5]。

The fundamental building block of Inception-style models is the Inception module, of which several different versions exist. In figure 1 we show the canonical form of an Inception module, as found in the Inception V3 architecture. An Inception model can be understood as a stack of such modules. This is a departure from earlier VGG-style networks which were stacks of simple convolution layers. 

Inception风格的模型的基础模块就是Inception模块，也有几种不同的版本。图1是Inception模块的经典形式，在Inception V3架构中使用。一个Inception模型可以理解成这种模块的堆叠。这与较早的VGG类型网络不同，那只是简单的卷积层的叠加。

While Inception modules are conceptually similar to convolutions (they are convolutional feature extractors), they empirically appear to be capable of learning richer representations with less parameters. How do they work, and how do they differ from regular convolutions? What design strategies come after Inception?

虽然Inception模块概念上与卷积层类似（它们是卷积特征提取器），经验上来说，它们可以用更少的参数学到更丰富的表示。它们是怎么工作的呢，与传统卷积层有什么不同呢？Inception遵循什么样的设计策略呢？

### 1.1. The Inception hypothesis Inception假设

A convolution layer attempts to learn filters in a 3D space, with 2 spatial dimensions (width and height) and a channel dimension; thus a single convolution kernel is tasked with simultaneously mapping cross-channel correlations and spatial correlations.

一个卷积层要在3D空间中学习滤波器，其中2维是空间（宽和高）和一个通道维；所以单个卷积核的任务就是同时映射通道间的相关和空间相关。

This idea behind the Inception module is to make this process easier and more efficient by explicitly factoring it into a series of operations that would independently look at cross-channel correlations and at spatial correlations. More precisely, the typical Inception module first looks at cross-channel correlations via a set of 1x1 convolutions, mapping the input data into 3 or 4 separate spaces that are smaller than the original input space, and then maps all correlations in these smaller 3D spaces, via regular 3x3 or 5x5 convolutions. This is illustrated in figure 1. In effect, the fundamental hypothesis behind Inception is that cross-channel correlations and spatial correlations are sufficiently decoupled that it is preferable not to map them jointly ( A variant of the process is to independently look at width-wise correlations and height-wise correlations. This is implemented by some of the modules found in Inception V3, which alternate 7x1 and 1x7 convolutions. The use of such spatially separable convolutions has a long history in image processing and has been used in some convolutional neural network implementations since at least 2012 (possibly earlier)).

Inception模块后面的思想是，使这个过程更容易更有效率，其做法是显式的将其分解成一系列操作，可以独立的进行通道间的相关和空间相关操作。更精确的说，典型的Inception模块首先用1×1卷积集合进行通道间相关操作，将输入数据映射到3或4个分隔的比原始输入空间更小的空间，然后用常规的3×3或5×5卷积将这些空间中进行相关性的映射，如图1所示。实际上，Inception后的基本假设是，通道间相关和空间相关是充分去耦合的，最好不要将它们进行联合映射（这个过程的一个变体是，独立的进行宽度相关和高度相关，Inception V3的一些模块就是这样实现的，也就是那些7×1和1×7的卷积；这种空间分离的卷积在图像处理中有很长的历史，其在卷积神经网络中的应用至少可以追溯到2012年，或者更早）。

Consider a simplified version of an Inception module that only uses one size of convolution (e.g. 3x3) and does not include an average pooling tower (figure 2). This Inception module can be reformulated as a large 1x1 convolution followed by spatial convolutions that would operate on non-overlapping segments of the output channels (figure 3). This observation naturally raises the question: what is the effect of the number of segments in the partition (and their size)? Would it be reasonable to make a much stronger hypothesis than the Inception hypothesis, and assume that cross-channel correlations and spatial correlations can be mapped completely separately?

考虑一个Inception模块的简化版本，其中只使用了一种卷积尺寸（即3×3），没有包含平均池化层（如图2）。这种Inception模块可以重新表示为一个大的1×1卷积，后面跟着的空间卷积对输出通道的（不重叠的）不同段进行操作（图3）。这种观察自然提出下面的问题：分割的段数的多少（以及尺寸）有什么影响？可以进行一个更强的假设吗，即假设通道间相关和空间相关可以完全分离的进行映射？

### 1.2. The continuum between convolutions and separable convolutions 卷积和分离卷积的统一

An “extreme” version of an Inception module, based on this stronger hypothesis, would first use a 1x1 convolution to map cross-channel correlations, and would then separately map the spatial correlations of every output channel. This is shown in figure 4. We remark that this extreme form of an Inception module is almost identical to a depthwise separable convolution, an operation that has been used in neural network design as early as 2014 [15] and has become more popular since its inclusion in the TensorFlow framework [1] in 2016.

基于这个更强的假设，Inception模块的一个极限版本，首先使用1×1卷积进行通道间相关的映射，然后对每个输出通道分别进行空间相关的映射，如图4所示。我们注意，Inception模块的这种极限形式几乎与depthwise separable卷积是一样的，这种操作早在2014年[15]就用于神经网络设计，由于2016年TensorFlow将其实现为库函数数更为流行起来。

A depthwise separable convolution, commonly called “separable convolution” in deep learning frameworks such as TensorFlow and Keras, consists in a depthwise convolution, i.e. a spatial convolution performed independently over each channel of an input, followed by a pointwise convolution, i.e. a 1x1 convolution, projecting the channels output by the depthwise convolution onto a new channel space. This is not to be confused with a spatially separable convolution, which is also commonly called “separable convolution” in
the image processing community.

一个depthwise separable卷积，在TensorFlow和Keras这样的深度学习框架中常被称为“可分离卷积”，包含了一个depthwise卷积，即在输入的每个通道上独立的进行空间卷积，跟随的是一个pointwise卷积，即1×1卷积，将depthwise卷积的通道输出投影到新的通道空间中。不要将这个与空间可分离卷积进行混淆，这种卷积在图像处理团体中也常被称为可分离卷积。

Two minor differences between and “extreme” version of an Inception module and a depthwise separable convolution would be: Inception模块的极限版本与depthwise separable卷积的两个细微的区别是：

- The order of the operations: depthwise separable convolutions as usually implemented (e.g. in TensorFlow) perform first channel-wise spatial convolution and then perform 1x1 convolution, whereas Inception performs the 1x1 convolution first.

- 操作的顺序：depthwise separable卷积通常的实现方法（如TensorFlow中）首先进行channel-wise空间卷积，然后进行1×1卷积，而Inception首先进行1×1卷积。

- The presence or absence of a non-linearity after the first operation. In Inception, both operations are followed by a ReLU non-linearity, however depthwise separable convolutions are usually implemented without non-linearities.

- 在第一个操作后是否存在非线性操作。在Inception中，两个操作之后都有ReLU非线性操作，而depthwise separable卷积通常没有非线性操作。

We argue that the first difference is unimportant, in particular because these operations are meant to be used in a stacked setting. The second difference might matter, and we investigate it in the experimental section (in particular see figure 10).

我们认为第一个区别不重要，尤其是这些操作是叠加起来进行的。第二个区别可能更要紧一点，我们在试验部分进行研究（见图10）。

We also note that other intermediate formulations of Inception modules that lie in between regular Inception modules and depthwise separable convolutions are also possible: in effect, there is a discrete spectrum between regular convolutions and depthwise separable convolutions, parametrized by the number of independent channel-space segments used for performing spatial convolutions. A regular convolution (preceded by a 1x1 convolution), at one extreme of this spectrum, corresponds to the single-segment case; a depthwise separable convolution corresponds to the other extreme where there is one segment per channel; Inception modules lie in between, dividing a few hundreds of channels into 3 or 4 segments. The properties of such intermediate modules appear not to have been explored yet.

我们还注意到，常规Inception模块和depthwise separable卷积之间的其他Inception模块的中间形式也是有可能的：实际上，常规卷积与depthwise separable卷积之间有一个离散的范围，其参数是独立的channel-space段的数目，用作空间卷积。常规卷积（前面是1×1卷积），是这个范围的一个极端，对应着一段的情况；depthwise separable卷积对应着另一个极端，即每个通道都对应着一段；中间形式的Inception模块，将几百个通道分成3段或4段。这些中间模块的性质尚未进行研究。

Having made these observations, we suggest that it may be possible to improve upon the Inception family of architectures by replacing Inception modules with depthwise separable convolutions, i.e. by building models that would be stacks of depthwise separable convolutions. This is made practical by the efficient depthwise convolution implementation available in TensorFlow. In what follows, we present a convolutional neural network architecture based on this idea, with a similar number of parameters as Inception V3, and we evaluate its performance against Inception V3 on two large-scale image classification task.

得到这样的观察后，我们认为，可以将Inception模块替换成depthwise separable卷积，可能会改进Inception架构，也就是说，模型是depthwise separable卷积的叠加。由于TensorFlow中有depthwise卷积的实现，其验证就很可行。下面我们提出基于这种思想的卷积神经网络架构，其参数数量与Inception V3类似，我们在两个大规模图像分类任务中将两种模型进行比较。

## 2. Prior work 前面的工作

The present work relies heavily on prior efforts in the following areas: 目前的工作与下面领域的先前努力是分不开的：

- Convolutional neural networks [10 , 9 , 25], in particular the VGG-16 architecture [18], which is schematically similar to our proposed architecture in a few respects.

- 卷积神经网络[10,9,25]，特别是VGG-16架构[18]，它与我们提出的架构在一些方面有类似的框架。

- The Inception architecture family of convolutional neural networks [20 , 7 , 21 , 19], which first demonstrated the advantages of factoring convolutions into multiple branches operating successively on channels and then on space.

- Inception架构的卷积神经网络[20,7,21,19]，将卷积分解成了多个分支，先对通道进行操作，然后对空间进行操作，第一个展示了卷积分解的优势；

- Depthwise separable convolutions, which our proposed architecture is entirely based upon. While the use of spatially separable convolutions in neural networks has a long history, going back to at least 2012 [12] (but likely even earlier), the depthwise version is more recent. Laurent Sifre developed depthwise separable convolutions during an internship at Google Brain in 2013, and used them in AlexNet to obtain small gains in accuracy and large gains in convergence speed, as well as a significant reduction in model size. An overview of his work was first made public in a presentation at ICLR 2014 [23]. Detailed experimental results are reported in Sifre’s thesis, section 6.2 [15]. This initial work on depthwise separable convolutions was inspired by prior research from Sifre and Mallat on transformation-invariant scattering [16 , 15]. Later, a depthwise separable convolution was used as the first layer of Inception V1 and Inception V2 [20 , 7]. Within Google, Andrew Howard [6] has introduced efficient mobile models called MobileNets using depthwise separable convolutions. Jin et al. in 2014 [8] and Wang et al. in 2016 [24] also did related work aiming at reducing the size and computational cost of convolutional neural networks using separable convolutions. Additionally, our work is only possible due to the inclusion of an efficient implementation of depthwise separable convolutions in the TensorFlow framework [1].

- Depthwise separable convolutions，我们提出的架构完全基于这种卷积。神经网络中空间可分离卷积的使用历史很长，至少可以追溯到2012年[12]（甚至可能更早），而depthwise版则是更近来的事。Laurent Sifre 2013年在Google Brain实习的时候提出了depthwise separable卷积，将其应用在了AlexNet中，得到的准确率提升较小，收敛速度的提升很大，模型规模的减小也非常明显。其工作概览见ICLR 2014[23]的介绍。详细的试验结果见6.2节Sifre的学位论文[15]。Depthwise separable卷积的最初工作是受到Sifre和Mallat在transformation-invariant scattering[16,15]的先期研究工作的启发。后来，在Inception V1和Inception V2 [20,7]的第一层使用了depthwise separable卷积。在Google内部，Andrew Howard[6]提出了高效的移动模型称为MobileNets，也使用了depthwise separable卷积。Jin et al. in 2014 [8]和Wang et al. in 2016 [24]也做了相关工作，目的是使用可分离卷积降低卷积神经网络的规模和计算量。另外，由于TensorFlow框架中包含了depthwise separable卷积的高效实现，我们的工作才成为可能。

- Residual connections, introduced by He et al. in [4], which our proposed architecture uses extensively.

- 残差连接，由He et al.提出[4]，我们提出的框架进行了广泛的使用。

## 3. The Xception architecture

We propose a convolutional neural network architecture based entirely on depthwise separable convolution layers. In effect, we make the following hypothesis: that the mapping of cross-channels correlations and spatial correlations in the feature maps of convolutional neural networks can be entirely decoupled. Because this hypothesis is a stronger version of the hypothesis underlying the Inception architecture, we name our proposed architecture Xception, which stands for “Extreme Inception”.

我们完全基于depthwise separable卷积层提出了一个卷积网络架构。实际上，我们进行了以下假设：卷积神经网络特征图的通道间相关和空间相关的映射可以完全解耦合。由于这个假设是Inception架构的假设的更强的版本，我们将我们提出的架构命名为Xception，代表"Extreme Inception"。

A complete description of the specifications of the network is given in figure 5. The Xception architecture has 36 convolutional layers forming the feature extraction base of the network. In our experimental evaluation we will exclusively investigate image classification and therefore our convolutional base will be followed by a logistic regression layer. Optionally one may insert fully-connected layers before the logistic regression layer, which is explored in the experimental evaluation section (in particular, see figures 7 and 8). The 36 convolutional layers are structured into 14 modules, all of which have linear residual connections around them, except for the first and last modules.

网络指标的完全描述见图5。Xception架构有36个卷积层，形成了网络特征提取的基础。在我们的试验评估中，我们只对图像分类进行了研究，所以我们的卷积后跟着一个logistic回归层。可以选择在logistic回归层之前加入全连接层，这在试验评估节中有研究（见图7和图8）。36个卷积层形成了14个模块，除了第一个和最后一个模块，它们都有线性残差连接。

In short, the Xception architecture is a linear stack of depthwise separable convolution layers with residual connections. This makes the architecture very easy to define and modify; it takes only 30 to 40 lines of code using a high-level library such as Keras [ 2 ] or TensorFlow-Slim [ 17 ], not unlike an architecture such as VGG-16 [ 18 ], but rather unlike architectures such as Inception V2 or V3 which are far more complex to define. An open-source implementation of Xception using Keras and TensorFlow is provided as part of the Keras Applications module, under the MIT license.

简而言之，Xception架构就是带有残差连接的depthwise separable卷积层的叠加。这使模型的定义和修改非常容易；如果用高层库，比如Keras或TensorFlow-Slim，那么只需要30到40行代码，与VGG-16很像，反而与Inception V2或V3很不像，定义这些模型非常复杂。使用Keras和TensorFlow的Xception开源实现是Keras应用的一个模块。

## 4. Experimental evaluation 试验评估

We choose to compare Xception to the Inception V3 architecture, due to their similarity of scale: Xception and Inception V3 have nearly the same number of parameters (table 3), and thus any performance gap could not be attributed to a difference in network capacity. We conduct our comparison on two image classification tasks: one is the well-known 1000-class single-label classification task on the ImageNet dataset [14], and the other is a 17,000-class multi-label classification task on the large-scale JFT dataset.

我们将Xception与Inception V3架构进行比较，这是因为它们规模相当：Xception与Inception V3的参数数量几乎一样（见表3），所以模型性能上的差异肯定不是由网络能力引起的。我们在两个图像分类任务中进行比较：第一个是有名的ImageNet数据集，共1000类的单标签分类任务，另一个是在大规模JFT数据集上的17000类的多标签分类任务。

### 4.1. The JFT dataset JFT数据集

JFT is an internal Google dataset for large-scale image classification dataset, first introduced by Hinton et al. in [5], which comprises over 350 million high-resolution images annotated with labels from a set of 17,000 classes. To evaluate the performance of a model trained on JFT, we use an auxiliary dataset, FastEval14k.

JFT是一个Google的内部数据集，进行大规模图像分类的数据集，首先由Hinton et al.在[5]中提出，包括了3.5亿张标注过的高分辨率图像，共17000类。为评估模型在JFT上的其性能，我们使用一个辅助数据集，FastEval14k。

FastEval14k is a dataset of 14,000 images with dense annotations from about 6,000 classes (36.5 labels per image on average). On this dataset we evaluate performance using Mean Average Precision for top 100 predictions (MAP@100), and we weight the contribution of each class to MAP@100 with a score estimating how common (and therefore important) the class is among social media images. This evaluation procedure is meant to capture performance on frequently occurring labels from social media, which is crucial for production models at Google.

FastEval14k数据集包含14000张图，带有6000类的稠密标注（平均每幅图36.5个标签）。在这个数据集上，我们用最高100个预测的Mean Average Precision(MAP@100)进行性能评估，我们用一个分数来衡量每个类对MAP@100的贡献，这个分数是对这个类在社交媒体图片中的常见程度（也就是重要程度）。这个评估过程是要得到社交媒体中经常出现的标签的表现，这对于Google中的生产模型非常重要。

### 4.2. Optimization configuration 优化配置

A different optimization configuration was used for ImageNet and JFT: 对于ImageNet和JFT使用了两种不同的优化配置：

1. On ImageNet:
- Optimizer: SGD
- Momentum: 0.9
- Initial learning rate: 0.045 初始学习速率
- Learning rate decay: decay of rate 0.94 every 2 epochs 学习速率衰减：每2轮衰减0.94
2. On JFT:
- Optimizer: RMSprop [22]
- Momentum: 0.9
- Initial learning rate: 0.001 初始学习速率
- Learning rate decay: decay of rate 0.9 every 3,000,000 samples 学习速率衰减：每3M样本衰减率0.9

For both datasets, the same exact same optimization configuration was used for both Xception and Inception V3. Note that this configuration was tuned for best performance with Inception V3; we did not attempt to tune optimization hyperparameters for Xception. Since the networks have different training profiles (figure 6), this may be suboptimal, especially on the ImageNet dataset, on which the optimization configuration used had been carefully tuned for Inception V3.

对于每个数据集，Xception和Inception V3网络都使用了相同的优化配置。注意这个配置是针对Inception V3进行的优化配置；我们不希望对Xception的超参数进行调整优化。由于网络有不同的训练情况（见图6），这可能不是最优的，尤其是对于ImageNet数据集。

Additionally, all models were evaluated using Polyak averaging [13] at inference time. 另外，所有模型在推理时都使用Polyak averaging[13]进行评估。

### 4.3. Regularization configuration 正则化配置

- Weight decay: The Inception V3 model uses a weight decay (L2 regularization) rate of 4e−5, which has been carefully tuned for performance on ImageNet. We found this rate to be quite suboptimal for Xception and instead settled for 1e−5. We did not perform an extensive search for the optimal weight decay rate. The same weight decay rates were used both for the ImageNet experiments and the JFT experiments.

- 权重衰减：Inception V3模型使用的权重衰减（L2正则化）系数为4e-5，这是在ImageNet上仔细调整过的。我们发现这个速率对于Xception来说效果很不好，并设定为1e-5。我们没有就寻找最优权重衰减系数进行广泛搜索。在ImageNet试验和JFT试验中使用了相同的权重衰减系数。

- Dropout: For the ImageNet experiments, both models include a dropout layer of rate 0.5 before the logistic regression layer. For the JFT experiments, no dropout was included due to the large size of the dataset which made overfitting unlikely in any reasonable amount of time.

- Dropout：对于ImageNet试验，两个模型都使用了0.5的dropout层，位置是在logistic回归层前。对于JFT试验，没有使用dropout层，因为数据集规模很大，要达到过拟合基本不太可能。

- Auxiliary loss tower: The Inception V3 architecture may optionally include an auxiliary tower which back-propagates the classification loss earlier in the network, serving as an additional regularization mechanism. For simplicity, we choose not to include this auxiliary tower in any of our models.

- 辅助损失tower：Inception V3架构可能可能会有一个可选的辅助tower，将分类损失函数梯度反向传播到网络的早期（底层），作为一种另外的正则化机制。为简单期间，我们的模型中都没有使用辅助tower。

### 4.4. Training infrastructure 训练基础设施

All networks were implemented using the TensorFlow framework [1] and trained on 60 NVIDIA K80 GPUs each. For the ImageNet experiments, we used data parallelism with synchronous gradient descent to achieve the best classification performance, while for JFT we used asynchronous gradient descent so as to speed up training. The ImageNet experiments took approximately 3 days each, while the JFT experiments took over one month each. The JFT models were not trained to full convergence, which would have taken over three month per experiment.

所有的网络都用TensorFlow实现，在60个NVidia K80 GPU上训练。在ImageNet试验中，我们使用了同步梯度下降的数据并行来得到最佳分类性能；在JFT试验中，我们使用了异步梯度下降来加速训练。ImageNet试验用了大约3天训练，而JFT试验用了超过一个月。JFT模型的训练没有充分收敛，如果充分收敛每个试验会耗时超过三个月。

### 4.5. Comparison with Inception V3 与Inception V3模型的比较

#### 4.5.1 Classification performance 分类性能

All evaluations were run with a single crop of the inputs images and a single model. ImageNet results are reported on the validation set rather than the test set (i.e. on the non-blacklisted images from the validation set of ILSVRC 2012). JFT results are reported after 30 million iterations (one month of training) rather than after full convergence. Results are provided in table 1 and table 2, as well as figure 6, figure 7, figure 8. On JFT, we tested both versions of our networks that did not include any fully-connected layers, and versions that included two fully-connected layers of 4096 units each before the logistic regression layer.

所有的评估都用输入图像的一个剪切块在单个模型上运行。ImageNet结果是在验证集上进行的，而不是测试集（即ILSVRC-2012的验证集中的非黑名单图像）。JFT结果运行了3000万次迭代（一个月的训练），但仍然没有充分收敛。结果如表和表2所示，还有图6,7,8。在JFT上，我们测试了网络的两个版本，没有全连接层，包含2个4096单元的全连接层的后面都是logistic回归层。

On ImageNet, Xception shows marginally better results than Inception V3. On JFT, Xception shows a 4.3% relative improvement on the FastEval14k MAP@100 metric. We also note that Xception outperforms ImageNet results reported by He et al. for ResNet-50, ResNet-101 and ResNet-152 [4].

在ImageNet上，Xception比Inception V3上效果略好一点点。在JFT上，Xception在FastEval14k上MAP@100衡量标准上有4.3%的相对改进。我们还注意到，Xception在ImageNet上的结果比He et al. for ResNet-50, ResNet-101 and ResNet-152 [4]都要好。

The Xception architecture shows a much larger performance improvement on the JFT dataset compared to the ImageNet dataset. We believe this may be due to the fact that Inception V3 was developed with a focus on ImageNet and may thus be by design over-fit to this specific task. On the other hand, neither architecture was tuned for JFT. It is likely that a search for better hyperparameters for Xception on ImageNet (in particular optimization parameters and regularization parameters) would yield significant additional improvement.

Xception架构在JFT数据集上的改进比ImageNet数据集上要大。我们相信这是因为Inception V3就是在ImageNet数据集上设计的，可能在这个特殊任务上有一定的过拟合现象。另一方面，两个架构都没有在JFT上精调过。如果在ImageNet上优化Xception的超参数（尤其是最优化参数和正则化参数），将会得到额外的改进。

#### 4.5.2 Size and speed 规模和速度

Table 3. Size and training speed comparison. 规模和训练速度的比较

| | Parameter count | Steps/second
--- | --- | ---
Inception V3 | 23,626,728 | 31
Xception | 22,855,952 | 28

In table 3 we compare the size and speed of Inception V3 and Xception. Parameter count is reported on ImageNet (1000 classes, no fully-connected layers) and the number of training steps (gradient updates) per second is reported on ImageNet with 60 K80 GPUs running synchronous gradient descent. Both architectures have approximately the same size (within 3.5%), and Xception is marginally slower. We expect that engineering optimizations at the level of the depthwise convolution operations can make Xception faster than Inception V3 in the near future. The fact that both architectures have almost the same number of parameters indicates that the improvement seen on ImageNet and JFT does not come from added capacity but rather from a more efficient use of the model parameters.

在表3中我们比较了Inception V3和Xception的规模和速度。参数数量是在ImageNet（1000类，没有全连接层）上训练的，每秒训练步数（梯度更新次数）是用60个K80 GPUs对ImageNet进行同步梯度下降。两个架构规模很接近（3.5%以内），Xception略慢一点。我们希望depthwise卷积操作层次的工程优化将来可以使Xception比Inception V3更快。两种架构参数数量几乎一样，这个事实说明在ImageNet和JFT上的改进并不是来自模型复杂度的增加而是更模型参数更有效的使用。

### 4.6. Effect of the residual connections 残差连接的影响

To quantify the benefits of residual connections in the Xception architecture, we benchmarked on ImageNet a modified version of Xception that does not include any residual connections. Results are shown in figure 9. Residual connections are clearly essential in helping with convergence, both in terms of speed and final classification performance. However we will note that benchmarking the non-residual model with the same optimization configuration as the residual model may be uncharitable and that better optimization configurations might yield more competitive results.

为量化残差连接在Xception架构中的益处，我们在ImageNet上基准测试了一个修改的Xception架构，其中不包含任何残差连接。结果如图9所示。残差连接很清楚的在改进收敛方面是必不可少的，在速度和最终分类配置方面都是。但是我们应当说明，无残差模型和残差模型用同样的优化配置进行基准测试是不太公平的，更好的优化配置可能会得到更有竞争力的结果。

Additionally, let us note that this result merely shows the importance of residual connections for this specific architecture, and that residual connections are in no way required in order to build models that are stacks of depthwise separable convolutions. We also obtained excellent results with non-residual VGG-style models where all convolution layers were replaced with depthwise separable convolutions (with a depth multiplier of 1), superior to Inception V3 on JFT at equal parameter count.

另外，我们注意到，这个结果只是说明了残差连接对于这个特定结构的重要性，建立堆叠的depthwise separable卷积模型，并不需要残差连接。我们将无残差VGG类模型中的卷积层都替换成了depthwise separable卷积（depth乘数为1），也得到了优秀的结果，比Inception V3在JFT数据集上的成绩要好，而参数数量是一样的。

### 4.7.Effect of an intermediate activation after pointwise convolutions

We mentioned earlier that the analogy between depthwise separable convolutions and Inception modules suggests that depthwise separable convolutions should potentially include a non-linearity between the depthwise and pointwise operations. In the experiments reported so far, no such non-linearity was included. However we also experimentally tested the inclusion of either ReLU or ELU [3] as intermediate non-linearity. Results are reported on ImageNet in figure 10, and show that the absence of any non-linearity leads to both faster convergence and better final performance.

前面我们提到了depthwise separable卷积和Inception模块的类比，这说明depthwise separable卷积应当在depthwise和pointwise操作之间包含一个非线性处理。在目前报告的试验中，没有这种非线性处理。但是我们实验性的测试了将ReLU或ELU作为中间非线性处理层。在ImageNet上得到的结果如图10所示，说明没有任何非线性处理会导致收敛更快，最后性能也更好。

This is a remarkable observation, since Szegedy et al. report the opposite result in [ 21 ] for Inception modules. It may be that the depth of the intermediate feature spaces on which spatial convolutions are applied is critical to the usefulness of the non-linearity: for deep feature spaces (e.g. those found in Inception modules) the non-linearity is helpful, but for shallow ones (e.g. the 1-channel deep feature spaces of depthwise separable convolutions) it becomes harmful, possibly due to a loss of information.

这是一个非凡的观察结果，因为Szegedy et al.在[21]中对Inception模块报出了相反的结果。可能是空间卷积作用的中间特征空间的深度对于非线性处理的用处是关键的：对于深度特征空间（如在Inception模块中发现的），非线性处理是有帮助的，但对于浅层特征空间（如1通道depthwise separable卷积的深度特征空间）是有害的，可能因为有信息损失。

## 5. Future directions 未来方向

We noted earlier the existence of a discrete spectrum between regular convolutions and depthwise separable convolutions, parametrized by the number of independent channel-space segments used for performing spatial convolutions. Inception modules are one point on this spectrum. We showed in our empirical evaluation that the extreme formulation of an Inception module, the depthwise separable convolution, may have advantages over a regular Inception module. However, there is no reason to believe that depthwise separable convolutions are optimal. It may be that intermediate points on the spectrum, lying between regular Inception modules and depthwise separable convolutions, hold further advantages. This question is left for future investigation.

我们注意到，在常规卷积与depthwise separable卷积之间离散谱的存在，其参数为独立的channel-space段的数量，可以用于进行空间卷积。Inception模块是这个谱上的一个点。我们用我们的经验估计，展示了Inception模块的极限表示，也就是depthwise separable卷积，与传统Inception模块相比会有优势。但是，没有理由相信depthwise separable卷积是最优的。可能是谱中的某些中间点有更大的优势，在普通Inception模块与depthwise separable卷积之间。这个问题有待未来研究。

## 6. Conclusions 结论

We showed how convolutions and depthwise separable convolutions lie at both extremes of a discrete spectrum, with Inception modules being an intermediate point in between. This observation has led to us to propose replacing Inception modules with depthwise separable convolutions in neural computer vision architectures. We presented a novel architecture based on this idea, named Xception, which has a similar parameter count as Inception V3. Compared to Inception V3, Xception shows small gains in classification performance on the ImageNet dataset and large gains on the JFT dataset. We expect depthwise separable convolutions to become a cornerstone of convolutional neural network architecture design in the future, since they offer similar properties as Inception modules, yet are as easy to use as regular convolution layers.

我们展示了普通卷积与depthwise separable卷积为什么是在一个离散谱的两个极端，而Inception模块是在其中的中间点。这个观察使我们提出，将Inception模块替换成depthwise separable卷积，形成新的神经计算机视觉架构。基于这个思想，我们提出了新的架构，名为Xception，与Inception V3有类似的参数数量。与Inception V3相比，Xception在ImageNet数据集上的分类任务有较小的改进，在JFT数据集上有很大的改进。我们希望将来depthwise separable卷积成为卷积神经网络架构设计的基石，因为与Inception模块性质类似，但却与普通卷积层一样易于使用。

## References