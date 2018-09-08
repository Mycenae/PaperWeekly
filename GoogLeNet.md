# Going deeper with convolutions 更深的卷积

Christian Szegedy et al. Google Inc.

## Abstract

We propose a deep convolutional neural network architecture codenamed Inception, which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC14 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.

我们提出了一种深度卷积神经网络架构，代号Inception，在ILSVRC-14中获得了分类和检测的最好成绩。这个架构的主要特点是改进了网络内部计算资源的利用，在保持计算能力的预算不增加的同时，仔细设计出了更深更宽的网络。为了使质量得到最优化，架构设计是基于Hebbian原则和多尺度处理的直觉的。我们向ILSVRC-14提交的一个特殊版本称为GoogLeNet，是一个22层的深度网络，在分类和检测的任务中评估了其质量。

## 1 Introduction

In the last three years, mainly due to the advances of deep learning, more concretely convolutional networks [10], the quality of image recognition and object detection has been progressing at a dramatic pace. One encouraging news is that most of this progress is not just the result of more powerful hardware, larger datasets and bigger models, but mainly a consequence of new ideas, algorithms and improved network architectures. No new data sources were used, for example, by the top entries in the ILSVRC 2014 competition besides the classification dataset of the same competition for detection purposes. Our GoogLeNet submission to ILSVRC 2014 actually uses 12× fewer parameters than the winning architecture of Krizhevsky et al [9] from two years ago, while being significantly more accurate. The biggest gains in object-detection have not come from the utilization of deep networks alone or bigger models, but from the synergy of deep architectures and classical computer vision, like the R-CNN algorithm by Girshick et al[6].

在过去的三年中，由于深度学习（更具体来说是卷积网络[10]）的进展，图像识别和目标检测的效果得到了很大提升。其主要进步不是因为硬件更强大、数据集更大或模型更大，而主要是因为新的思想、算法和改进的网络架构，这很鼓舞人心。比如在ILSVRC-14中，除了在分类的数据集上的检测目的任务中，获奖的团队没有利用新的数据源。我们向ILSVRC-14提交的GoogLeNet的参数数量只是两年前获胜的Krizhevsky et al [9]架构的1/12，而准确率却得到了显著提高。目标检测方面最大的收获不是单单通过利用深度网络或更大的模型，而是深度模型和经典计算机视觉的协同作用，像Girshick et al[6]的R-CNN算法。

Another notable factor is that with the ongoing traction of mobile and embedded computing, the efficiency of our algorithms – especially their power and memory use – gains importance. It is noteworthy that the considerations leading to the design of the deep architecture presented in this paper included this factor rather than having a sheer fixation on accuracy numbers. For most of the experiments, the models were designed to keep a computational budget of 1.5 billion multiply-adds at inference time, so that the they do not end up to be a purely academic curiosity, but could be put to real world use, even on large datasets, at a reasonable cost.

另一个值得注意的因素是随着移动和嵌入式计算的不断牵引，我们算法的效率（尤其是能耗和内存使用）越来越重要，本文模型的深度框架的设计考虑包括了这个因素，而不是仅仅关注准确率的提升。对于大多数试验来说，设计的模型保持了推理时15亿次乘/加法的计算量，所以这最后没有成为纯粹的学术好奇试验，而可能放在真实世界的任务中使用，即使是在大型数据集，代价比较合理。

In this paper, we will focus on an efficient deep neural network architecture for computer vision, codenamed Inception, which derives its name from the Network in network paper by Lin et al [12] in conjunction with the famous “we need to go deeper” internet meme [1]. In our case, the word “deep” is used in two different meanings: first of all, in the sense that we introduce a new level of organization in the form of the “Inception module” and also in the more direct sense of increased network depth. In general, one can view the Inception model as a logical culmination of [12] while taking inspiration and guidance from the theoretical work by Arora et al [2]. The benefits of the architecture are experimentally verified on the ILSVRC 2014 classification and detection challenges, on which it significantly outperforms the current state of the art.

本文中我们聚焦在一种高效计算机视觉深度神经网络架构，代号Inception，名称来自Lin et al [12]的论文Network in network和著名的网络meme盗梦空间[1]。这里深度有两个不同的意思：首先我们新提出的Inception架构，而且更直接的意思是网络的深度增加了。总体来说，可以将Inception模型看做[12]的逻辑极限，而我们的理论指导是Arora et al [2]。这个架构的好处由ILSVRC-14的分类和检测挑战试验验证了，在那里我们比现有的最好成绩要好很多。

## 2 Related Work 相关工作

Starting with LeNet-5 [10], convolutional neural networks (CNN) have typically had a standard structure – stacked convolutional layers (optionally followed by contrast normalization and max-pooling) are followed by one or more fully-connected layers. Variants of this basic design are prevalent in the image classification literature and have yielded the best results to-date on MNIST, CIFAR and most notably on the ImageNet classification challenge [9, 21]. For larger datasets such as Imagenet, the recent trend has been to increase the number of layers [12] and layer size [21, 14], while using dropout [7] to address the problem of overfitting.

从LeNet-5[10]开始，卷积神经网络(CNN)一般都有一个标准结构，即堆栈式的卷积层（有的带有对比度归一化和max-pooling）后面跟着一个或几个全连接层。这种基本设计的变体在图像分类的文献中相当普遍，并在几个数据集上得到了目前最好的结果，包括MNIST, CIFAR和最著名的ImageNet分类挑战赛[9, 21]。对于大一些的数据集比如ImageNet，最近的趋势是增加层数[12]和层规模[21, 14]，同时用dropout[7]来解决过拟合的问题。

Despite concerns that max-pooling layers result in loss of accurate spatial information, the same convolutional network architecture as [9] has also been successfully employed for localization [9, 14], object detection [6, 14, 18, 5] and human pose estimation [19]. Inspired by a neuroscience model of the primate visual cortex, Serre et al.[15] use a series of fixed Gabor filters of different sizes in order to handle multiple scales, similarly to the Inception model. However, contrary to the fixed 2-layer deep model of [15], all filters in the Inception model are learned. Furthermore, Inception layers are repeated many times, leading to a 22-layer deep model in the case of the GoogLeNet model.

尽管担心max-pooling层会损失准确的空间信息，和[9]一样的卷积神经网络架构也在定位[9, 14]，目标检测[6, 14, 18, 5]和人体姿态估计[9]中成功得到了应用。由神经科学中的灵长目视觉皮层模型所启发，Serre et al.[15]使用了一系列不同尺寸的固定Garbor滤波器来处理多尺度问题，这与Inception模型类似。但与固定的2层深度模型[15]形成对比的是，Inception模型中的所有滤波器都是学习得到的。更进一步，Inception层重复了很多次，在GoogLeNet模型中模型深度达22。

Network-in-Network is an approach proposed by Lin et al. [12] in order to increase the representational power of neural networks. When applied to convolutional layers, the method could be viewed as additional 1×1 convolutional layers followed typically by the rectified linear activation [9]. This enables it to be easily integrated in the current CNN pipelines. We use this approach heavily in our architecture. However, in our setting, 1 × 1 convolutions have dual purpose: most critically, they are used mainly as dimension reduction modules to remove computational bottlenecks, that would otherwise limit the size of our networks. This allows for not just increasing the depth, but also the width of our networks without significant performance penalty.

Lin et al. [12]提出Network-in-Network模型来增强神经网络的表示能力。当应用在卷积层中时，这个方法可以看做是，额外的1×1卷积层后跟着ReLU激活函数[9]。这使其很容易整合到目前的CNN架构流水线中。我们的架构中很多地方都用了这种方法。但是在我们的设置中，1×1卷积有双重目的，最主要用作降维模块，来去除计算瓶颈，如果不这样，就会限制我们模型的规模。这不仅可以增加模型深度，还可以网络宽度，而且不会显著增加计算量。

The current leading approach for object detection is the Regions with Convolutional Neural Networks (R-CNN) proposed by Girshick et al. [6]. R-CNN decomposes the overall detection problem into two subproblems: to first utilize low-level cues such as color and superpixel consistency for potential object proposals in a category-agnostic fashion, and to then use CNN classifiers to identify object categories at those locations. Such a two stage approach leverages the accuracy of bounding box segmentation with low-level cues, as well as the highly powerful classification power of state-of-the-art CNNs. We adopted a similar pipeline in our detection submissions, but have explored enhancements in both stages, such as multi-box [5] prediction for higher object bounding box recall, and ensemble approaches for better categorization of bounding box proposals.

目前领先的目标检测方法是Girshick et al. [6]提出的R-CNN(Regions with Convolutional Neural Networks)。R-CNN将检测问题分解成两个子问题：首先使用低级线索如色彩和超像素连贯性进行潜在的目标建议，在不知道类别的情况下进行，然后使用CNN分类器在这些区域识别目标类别。这种两阶段方法将边界框分割的准确性与底层线索和最先进的CNN的强大的分类能力结合起来，提高了模型了性能。我们在检测任务中采用了类似的流水线结构，但在两个阶段都进行了改进，比如对于更高的边界框联想使用多边界框[5]预测，组合各种方法得到更好的边界框建议分类。

## 3 Motivation and High Level Considerations 动机和高层考虑

The most straightforward way of improving the performance of deep neural networks is by increasing their  size. This includes both increasing the depth – the number of levels – of the network and its width: the number of units at each level. This is as an easy and safe way of training higher quality models, especially given the availability of a large amount of labeled training data. However this simple solution comes with two major drawbacks.

深度神经网络改进性能最直接的方法就是增加规模，这包括增加网络深度（层数）和宽度（每层的单元数目）。这是训练更高质量模型的简单安全的方法，尤其是有大量标记训练数据可用时。但是这种简单的方法却又两个主要的缺陷。

Bigger size typically means a larger number of parameters, which makes the enlarged network more prone to overfitting, especially if the number of labeled examples in the training set is limited. This can become a major bottleneck, since the creation of high quality training sets can be tricky and expensive, especially if expert human raters are necessary to distinguish between fine-grained visual categories like those in ImageNet (even in the 1000-class ILSVRC subset) as demonstrated by Figure 1.

更大的规模通常都意味着参数更多，这使得这个增大的网络更加容易过拟合，尤其是当训练集中的标记样本数量有限时。这可能成为主要瓶颈，因为高质量训练集的获得通常很难，代价很大，尤其是如果需要专家打分人员来区分细粒度视觉类别图像，比如图1所示的ImageNet图像（1000类的ILSVRC子集也很难）。

Figure 1: Two distinct classes from the 1000 classes of the ILSVRC 2014 classification challenge. (a) Siberian husky (b) Eskimo dog

图1. ILSVRC-14分类挑战赛1000类中的两个不同类别。(a)西伯利亚哈士奇；(b)爱斯基摩狗

Another drawback of uniformly increased network size is the dramatically increased use of computational resources. For example, in a deep vision network, if two convolutional layers are chained, any uniform increase in the number of their filters results in a quadratic increase of computation. If the added capacity is used inefficiently (for example, if most weights end up to be close to zero), then a lot of computation is wasted. Since in practice the computational budget is always finite, an efficient distribution of computing resources is preferred to an indiscriminate increase of size, even when the main objective is to increase the quality of results.

模型增大的另一个问题是急剧增加的计算需求。比如，在深度视觉网络中，如果两个卷积层连在一起，任何滤波器大小的统一增加都会导致运算量平方式的增加。如果增加的容量没有充分使用（比如，如果多数权重都很接近0），那么就会浪费很多计算量。由于在实践中计算量预算永远是有限的，即使我们的主要目标是增加分类结果质量，那么我们也倾向于计算资源的有效分配，而不是随意增加模型规模。

The fundamental way of solving both issues would be by ultimately moving from fully connected to sparsely connected architectures, even inside the convolutions. Besides mimicking biological systems, this would also have the advantage of firmer theoretical underpinnings due to the ground-breaking work of Arora et al. [2]. Their main result states that if the probability distribution of the data-set is representable by a large, very sparse deep neural network, then the optimal network topology can be constructed layer by layer by analyzing the correlation statistics of the activations of the last layer and clustering neurons with highly correlated outputs. Although the strict mathematical proof requires very strong conditions, the fact that this statement resonates with the well known Hebbian principle – neurons that fire together, wire together – suggests that the underlying idea is applicable even under less strict conditions, in practice.

解决这两个问题的根本方法是，彻底从全连接的架构，转向稀疏连接的架构，即使在卷积层内也要如此。除了是要模仿生物系统，这还有更坚实的理论支持，就是Arora et al. [2]的工作，其主要结论是，如果数据集的概率分布可以由一个大的很稀疏的深度神经网络表示的话，那么最佳网络拓扑可以以如下的方式构建，分析当前层激活的统计相关性，对具有高度相关性输出的神经元进行聚类，并确定下一层的结构。虽然严格的数学证明需要很强的假设条件，但这个结论与著名的Hebbian原则不谋而合，即一起放电的神经元，其连接也会增强，这说明其基本思想即使在没那么严格的条件下，也是可以在实践中应用的。

On the downside, todays computing infrastructures are very inefficient when it comes to numerical calculation on non-uniform sparse data structures. Even if the number of arithmetic operations is reduced by 100×, the overhead of lookups and cache misses is so dominant that switching to sparse matrices would not pay off. The gap is widened even further by the use of steadily improving, highly tuned, numerical libraries that allow for extremely fast dense matrix multiplication, exploiting the minute details of the underlying CPU or GPU hardware [16, 9]. Also, non-uniform sparse models require more sophisticated engineering and computing infrastructure. Most current vision oriented machine learning systems utilize sparsity in the spatial domain just by the virtue of employing convolutions. However, convolutions are implemented as collections of dense connections to the patches in the earlier layer. ConvNets have traditionally used random and sparse connection tables in the feature dimensions since [11] in order to break the symmetry and improve learning, the trend changed back to full connections with [9] in order to better optimize parallel computing. The uniformity of the structure and a large number of filters and greater batch size allow for utilizing efficient dense computation.

不幸的是，今天的计算基础设施对于非一致稀疏数据结构的数值计算是非常低效的。即使算数操作数量减少到1/100，用于查找和缓存未命中的开销仍然非常大，转换到稀疏矩阵仍然不划算。当使用持续改进的、高度调谐的数值计算库，而这个库允许极快的稠密矩阵乘法，压榨潜在的CPU和GPU硬件的计算资源的微小细节时，这个不划算的差距会进一步拉大[16, 9]。同时，非一致稀疏模型需要更复杂的工程和计算基础设施。多数现在的视觉相关的机器学习系统在空间域使用稀疏性只是因为要进行卷积。但是，前一层图像块的稠密连接的集合才是卷积运算。ConvNets传统上从[11]才在特征维度开始使用随机稀疏的连接，为的是要打破对称性，改善学习，而这个趋势又从[9]改回了全连接，因为要更好的优化并行计算。结构的一致性，大量滤波器，更大的batch size都需要采用有效的稠密计算。

This raises the question whether there is any hope for a next, intermediate step: an architecture that makes use of the extra sparsity, even at filter level, as suggested by the theory, but exploits our current hardware by utilizing computations on dense matrices. The vast literature on sparse matrix computations (e.g. [3]) suggests that clustering sparse matrices into relatively dense submatrices tends to give state of the art practical performance for sparse matrix multiplication. It does not seem far-fetched to think that similar methods would be utilized for the automated construction of non-uniform deep-learning architectures in the near future.

这提出了如下的问题，是否可能有下一个间接的步骤：这个架构利用了额外的稀疏性，即使在滤波器层，像理论建议的那样，但利用现有硬件的方式是稠密矩阵的计算。关于稀疏矩阵计算有很多文献如[3]，建议将稀疏矩阵分成相对稠密的子矩阵，有助于在稀疏矩阵乘法运算中得到最佳工程性能。在不远的将来，类似的方法可以用于非一致深度学习框架的自动构建，这个想法好像不是那么异想天开。

The Inception architecture started out as a case study of the first author for assessing the hypothetical output of a sophisticated network topology construction algorithm that tries to approximate a sparse structure implied by [2] for vision networks and covering the hypothesized outcome by dense, readily available components. Despite being a highly speculative undertaking, only after two iterations on the exact choice of topology, we could already see modest gains against the reference architecture based on [12]. After further tuning of learning rate, hyperparameters and improved training methodology, we established that the resulting Inception architecture was especially useful in the context of localization and object detection as the base network for [6] and [5]. Interestingly, while most of the original architectural choices have been questioned and tested thoroughly, they turned out to be at least locally optimal.

Inception架构第一个开始进行这样的研究，尝试构建一个复杂网络拓扑，得到的网络试图近似一个[2]中的视觉网络稀疏矩阵，其近似结果是稠密的可用组件，就像上面阐述的那样。尽管这一切都是假设，但在拓扑结构的选择上只经过2次迭代，我们已经看到了一定的收获，相对于[12]中的参考架构。在进一步调整了学习速率、超参数，改进了训练方法后，我们确定得到的Inception架构对于定位、目标检测都非常有用，可以作为[6]和[5]的基础网络架构。有意思的是，大多数原有的架构选择已经经过彻底的拷问和测试，它们只能算是局部最优的。

One must be cautious though: although the proposed architecture has become a success for computer vision, it is still questionable whether its quality can be attributed to the guiding principles that have lead to its construction. Making sure would require much more thorough analysis and verification: for example, if automated tools based on the principles described below would find similar, but better topology for the vision networks. The most convincing proof would be if an automated system would create network topologies resulting in similar gains in other domains using the same algorithm but with very differently looking global architecture. At very least, the initial success of the Inception architecture yields firm motivation for exciting future work in this direction.

人必须要谨慎：虽然提出的架构已经在计算机视觉方面很成功，但仍然可以质疑，其质量是否达到了构建原则中的理想效果？做这样的确认需要非常多的彻底分析和验证：比如，如果基于这种原则的自动工具（如下述）可以找到视觉网络的类似但更好的拓扑结构怎么办？最有说服力的证据是，如果自动系统可以生成的网络拓扑，在其他领域结果类似，使用的也是一样的算法，但全局架构很不一样。至少，Inception架构最初的成功一定会激励出更多这个方向的工作。

## 4 Architectural Details 架构细节

The main idea of the Inception architecture is based on finding out how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components. Note that assuming translation invariance means that our network will be built from convolutional building blocks. All we need is to find the optimal local construction and to repeat it spatially. Arora et al. [2] suggests a layer-by-layer construction in which one should analyze the correlation statistics of the last layer and cluster them into groups of units with high correlation. These clusters form the units of the next layer and are connected to the units in the previous layer. We assume that each unit from the earlier layer corresponds to some region of the input image and these units are grouped into filter banks. In the lower layers (the ones close to the input) correlated units would concentrate in local regions. This means, we would end up with a lot of clusters concentrated in a single region and they can be covered by a layer of 1×1 convolutions in the next layer, as suggested in [12]. However, one can also expect that there will be a smaller number of more spatially spread out clusters that can be covered by convolutions over larger patches, and there will be a decreasing number of patches over larger and larger regions. In order to avoid patch-alignment issues, current incarnations of the Inception architecture are restricted to filter sizes 1×1, 3×3 and 5×5, however this decision was based more on convenience rather than necessity. It also means that the suggested architecture is a combination of all those layers with their output filter banks concatenated into a single output vector forming the input of the next stage. Additionally, since pooling operations have been essential for the success in current state of the art convolutional networks, it suggests that adding an alternative parallel pooling path in each such stage should have additional beneficial effect, too (see Figure 2(a)).

Inception架构的主要思想是，卷积视觉网络中的一个最佳局部稀疏结构怎样用已有的稠密结构近似。如果假设模型有平移不变性，那么我们的模型就必须用卷积模块构建起来。我们需要的就是找到局部最佳结构然后不断在空间域重复之。Arora et al. [2]建议逐层构建，并先分析上一层的统计相关性，将其分成高相关的几个单元组。这些组就是下一层的单元结构，并与上一层的单元相连。我们假设上一层的每个单元都与输入图像的某区域对应，这些单元分进不同的滤波器组。在较低的层中（与输入接近的层），相关的单元将集中在局部区域，这意思是，我们将会得到单个区域中集中了很多簇，[12]中建议，在下一层中可以用1×1的卷积覆盖。但是，也可以假设有较少的空域分布更广的簇，需要用更大块的卷积才能覆盖，这样块的数量将会减少，区域将会越来越大。为了避免块对齐的问题，现有的Inception模型限制滤波器大小为1×1,3×3,5×5，但这样的规定只是因为便利，而不是必须。这还意味着，组成这个结构的这些层，其输出滤波器组串成一个单个的输出矢量，形成下一层的输入。另外，由于pooling操作对于目前卷积神经网络的成功是必须的，说明在每一层增加可替代的并行pooling通道应当有额外的好处（见图2a）。

Figure 2: Inception module (a) Inception module, naive version (b) Inception module with dimension reductions

图2 Inception模型 (a)朴素版 (b)降维版

As these “Inception modules” are stacked on top of each other, their output correlation statistics are bound to vary: as features of higher abstraction are captured by higher layers, their spatial concentration is expected to decrease suggesting that the ratio of 3×3 and 5×5 convolutions should increase as we move to higher layers.

当这些Inception模块一层一层堆积起来时，它们的输出的统计相关肯定会不一样：更高级抽象的特征为更高的层所捕获，其集中的空域应当缩减，说明3×3和5×5的卷积当层级更高时应当增加。

One big problem with the above modules, at least in this naive form, is that even a modest number of 5×5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters. This problem becomes even more pronounced once pooling units are added to the mix: their number of output filters equals to the number of filters in the previous stage. The merging of the output of the pooling layer with the outputs of convolutional layers would lead to an inevitable increase in the number of outputs from stage to stage. Even while this architecture might cover the optimal sparse structure, it would do it very inefficiently, leading to a computational blow up within a few stages.

上述模块的一个大问题是（至少是在naive版里），即使5×5卷积的数量不是很多，在一个有很多滤波器的卷积层上一层，也会计算代价非常夸张。在加入pooling单元后，问题更加明显：其输出滤波器数量与前一层的滤波器数量相等。pooling层的输出与卷积层的输出的合并，会导致输出的数量逐阶段增加。即使这个架构能够覆盖最佳稀疏结构，效率也会很低，在几个阶段之内就会导致计算量剧增到爆炸。

This leads to the second idea of the proposed architecture: judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise. This is based on the success of embeddings: even low dimensional embeddings might contain a lot of information about a relatively large image patch. However, embeddings represent information in a dense, compressed form and compressed information is harder to model. We would like to keep our representation sparse at most places (as required by the conditions of [2]) and compress the signals only whenever they have to be aggregated en masse. That is, 1×1 convolutions are used to compute reductions before the expensive 3×3 and 5×5 convolutions. Besides being used as reductions, they also include the use of rectified linear activation which makes them dual-purpose. The final result is depicted in Figure 2(b).

这就要介绍我们提出的架构的第二个主意了：只要哪里计算量增加太多，那么就进行降维和投影。这是基于嵌套(embedding)的成功的：即使是低维嵌套，也可能包含一个相对较大的图像块的很多信息。但是，嵌套表示信息以密集、压缩的形式存储，压缩的信息是很难建模的。我们希望我们的表示在多数位置都是稀疏的（也是文献[2]要求的条件），只在必须的地方进行压缩。也就是说，先用1×1的卷积来降维，然后再进行昂贵的3×3和5×5的卷积。除了用于降维，还用了ReLU，如图2(b)所示。

In general, an Inception network is a network consisting of modules of the above type stacked upon each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid. For technical reasons (memory efficiency during training), it seemed beneficial to start using Inception modules only at higher layers while keeping the lower layers in traditional convolutional fashion. This is not strictly necessary, simply reflecting some infrastructural inefficiencies in our current implementation.

总体来说，Inception网络就是上述模块堆叠在一起组成的，偶尔有步长为2的max-pooling层来使数据分辨率减半。由于技术原因（训练时的内存使用率），看起来在高层使用Inception结构更有利一些，低层仍然使用传统的卷积层。这不是严格必要的，只是反映了我们目前的实现中一些结构的效率没那么高而已。

One of the main beneficial aspects of this architecture is that it allows for increasing the number of units at each stage significantly without an uncontrolled blow-up in computational complexity. The ubiquitous use of dimension reduction allows for shielding the large number of input filters of the last stage to the next layer, first reducing their dimension before convolving over them with a large patch size. Another practically useful aspect of this design is that it aligns with the intuition that visual information should be processed at various scales and then aggregated so that the next stage can abstract features from different scales simultaneously.

这种架构的一个主要好处是，在每一阶段的单元数都可以非常多，而计算量还可以得到控制。降维的普遍使用，使得前一层大量的滤波器不影响后一层，这主要是因为在卷积前有一个降维处理。这种设计的另一种好处是，直觉上我们都认为视觉信息应当多尺度处理，然后再进行汇聚信息，这样下一层可以同时对不同尺度的特征进行抽象。

The improved use of computational resources allows for increasing both the width of each stage as well as the number of stages without getting into computational difficulties. Another way to utilize the inception architecture is to create slightly inferior, but computationally cheaper versions of it. We have found that all the included the knobs and levers allow for a controlled balancing of computational resources that can result in networks that are 2−3× faster than similarly performing networks with non-Inception architecture, however this requires careful manual design at this point.

计算资源的增多，使得模型的宽度和深度都可以增加。另一种利用Inception架构的方法是，生成略微低级但计算量更少的模型版本。我们发现，控制计算量的使用，我们可以生成的模型比其他非Inception模型速度快2-3倍，但表现仍然差不多，但这需要小心的手动设计模型。

## 5 GoogLeNet

We chose GoogLeNet as our team-name in the ILSVRC14 competition. This name is an homage to Yann LeCuns pioneering LeNet 5 network [10]. We also use GoogLeNet to refer to the particular incarnation of the Inception architecture used in our submission for the competition. We have also used a deeper and wider Inception network, the quality of which was slightly inferior, but adding it to the ensemble seemed to improve the results marginally. We omit the details of that network, since our experiments have shown that the influence of the exact architectural parameters is relatively minor. Here, the most successful particular instance (named GoogLeNet) is described in Table 1 for demonstrational purposes. The exact same topology (trained with different sampling methods) was used for 6 out of the 7 models in our ensemble.

我们在ILSVRC比赛时，选择了GoogLeNet作为团队名称。这是对Yann LeCuns先驱性的LeNet-5网络[10]致敬，也用来指我们向比赛提交的基于Inception架构的具体模型。我们用过一个更宽更深的Inception网络，其性能略差一些，但将其加入集成模型可以将结果改进一些。我们忽略那个网络的细节，因为我们的试验证明，具体架构的参数的影响相对较小。这里，最成功的网络（名为GoogLeNet）如表1所示，我们的集成模型中包含7个模型，其中6个的拓扑结构都是一样的（用不同的采样方法进行的训练）。

Table 1: GoogLeNet incarnation of the Inception architecture Inception架构的实例GoogLeNet

type           | patch size/stride | output size | depth | #1×1 | #3×3 reduce | #3×3 | #5×5 reduce | #5×5 | pool proj | params | ops
---            | ---               | ---         | ---   | ---  | ---         | ---  | ---         | ---  | ---       | ---    | ---
convolution    | 7×7/2             | 112×112×64  | 1     | | | | | | | 2.7K | 34M
max pool       | 3×3/2             | 56×56×64    | 0
convolution    | 3×3/1             | 56×56×192   | 2     | | 64 | 192 | | | | 112K | 360M
max pool       | 3×3/2             | 28×28×192   | 0
inception (3a) |                   | 28×28×256   | 2     | 64   | 96          | 128  | 16          | 32   | 32        | 159K   | 128M
inception (3b) |                   | 28×28×480   | 2     | 128  | 128         | 192  | 32          | 96   | 64        | 380K   | 304M
max pool       | 3×3/2             | 14×14×480   | 0
inception (4a) |                   | 14×14×512   | 2     | 192  | 96          | 208  | 16          | 48   | 64        | 364K   | 73M
inception (4b) |                   | 14×14×512   | 2     | 160  | 112         | 224  | 24          | 64   | 64        | 437K   | 88M
inception (4c) |                   | 14×14×512   | 2     | 128  | 128         | 256  | 24          | 64   | 64        | 463K   | 100M
inception (4d) |                   | 14×14×528   | 2     | 112  | 144         | 288  | 32          | 64   | 64        | 580K   | 119M
inception (4e) |                   | 14×14×832   | 2     | 256  | 160         | 320  | 32          | 128  | 128       | 840K   | 170M
max pool       | 3×3/2             | 7×7×832     | 0
inception (5a) |                   | 7×7×832     | 2     | 256  | 160         | 320  | 32          | 128  | 128       | 1072K  | 54M
inception (5b) |                   | 7×7×1024    | 2     | 384  | 192         | 384  | 48          | 128  | 128       | 1388K  | 71M
avg pool       | 7×7/1             | 1×1×1024    | 0
dropout (40%)  |                   | 1×1×1024    | 0
linear         |                   | 1×1×1000    | 1 | | | | | | | 1000K | 1M
softmax        |                   | 1×1×1000    | 0

All the convolutions, including those inside the Inception modules, use rectified linear activation. The size of the receptive field in our network is 224×224 taking RGB color channels with mean subtraction. “#3×3 reduce” and “#5×5 reduce” stands for the number of 1×1 filters in the reduction layer used before the 3×3 and 5×5 convolutions. One can see the number of 1×1 filters in the projection layer after the built-in max-pooling in the pool proj column. All these reduction/projection layers use rectified linear activation as well.

所有的卷积，包括Inception内部的那些，都用ReLU激活函数。输入是224×224大小，RGB颜色3通道，减去了均值。“#3×3 reduce”和“#5×5 reduce”代表3×3和5×5卷积前有1×1卷积的降维处理。1×1的滤波器数量可以在内建的max-pooling层后的投影层看到，在pool proj栏。所有这些降维/投影层也都使用ReLU激活函数。

The network was designed with computational efficiency and practicality in mind, so that inference can be run on individual devices including even those with limited computational resources, especially with low-memory footprint. The network is 22 layers deep when counting only layers with parameters (or 27 layers if we also count pooling). The overall number of layers (independent building blocks) used for the construction of the network is about 100. However this number depends on the machine learning infrastructure system used. The use of average pooling before the classifier is based on [12], although our implementation differs in that we use an extra linear layer. This enables adapting and fine-tuning our networks for other label sets easily, but it is mostly convenience and we do not expect it to have a major effect. It was found that a move from fully connected layers to average pooling improved the top-1 accuracy by about 0.6%, however the use of dropout remained essential even after removing the fully connected layers.

网络设计实用，计算效率高，其网络推理过程可以在计算能力很有限的设备上运行，尤其是内存有限的设备。如果只数有参数的层，网络深度为22，如果加上pooling层，那么是27层。构建网络的所有单元数大约是100，但这个数目要看机器学习的基础设施系统用了多少。分类器前使用平均pooling层是基于[12]，我们的模型略有不同，多使用了一个线性层。这可以使网络在其他标记数据集上很方便的进行精调，但it is mostly convenience and we do not expect it to have a major effect. 试验发现，从全连接层改用平均pooling层，使top-1准确率增加了0.6%，但即使去掉了全连接层，dropout仍然不可或缺。

Given the relatively large depth of the network, the ability to propagate gradients back through all the layers in an effective manner was a concern. One interesting insight is that the strong performance of relatively shallower networks on this task suggests that the features produced by the layers in the middle of the network should be very discriminative. By adding auxiliary classifiers connected to these intermediate layers, we would expect to encourage discrimination in the lower stages in the classifier, increase the gradient signal that gets propagated back, and provide additional regularization. These classifiers take the form of smaller convolutional networks put on top of the output of the Inception (4a) and (4d) modules. During training, their loss gets added to the total loss of the network with a discount weight (the losses of the auxiliary classifiers were weighted by 0.3). At inference time, these auxiliary networks are discarded.

我们的网络很深，这就使有效的经过各层反向传播梯度需要关注。相对较浅的网络效果也很强，这说明网络中间各层产生的特征应当很有区分能力。我们在中间层添加了辅助分类器，希望能在分类器的较低层就增加分类能力，增强传回的梯度信号，并提供额外的正则化。这两个分类器在Inception 4(a)和4(d)模块后，形成小一些的卷积网络。训练过程中，其损失函数加权（辅助分类器0.3）加到网络的总损失中。推理时，这些辅助分类器就不需要了。

The exact structure of the extra network on the side, including the auxiliary classifier, is as follows: 网络具体结构如下：

- An average pooling layer with 5×5 filter size and stride 3, resulting in an 4×4×512 output for the (4a), and 4×4×528 for the (4d) stage.
- A 1×1 convolution with 128 filters for dimension reduction and rectified linear activation.
- A fully connected layer with 1024 units and rectified linear activation.
- A dropout layer with 70% ratio of dropped outputs.
- A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the main classifier, but removed at inference time).

- 平均pooling层滤波器大小5×5，步长3，在4(a)处的输出为4×4×512,4(d)处大小4×4×528；
- 128个1×1卷积的滤波器进行降维和ReLU激活；
- 全连接层1024个单元，有ReLU激活；
- dropout率0.7；
- 线性单元层与softmax loss分类器共用（预测1000类作为主分类器，在推理时去除）；

A schematic view of the resulting network is depicted in Figure 3. 得到的网络如图3所示。

## 6 Training Methodology 训练方法

Our networks were trained using the DistBelief [4] distributed machine learning system using modest amount of model and data-parallelism. Although we used CPU based implementation only, a rough estimate suggests that the GoogLeNet network could be trained to convergence using few high-end GPUs within a week, the main limitation being the memory usage. Our training used asynchronous stochastic gradient descent with 0.9 momentum [17], fixed learning rate schedule (decreasing the learning rate by 4% every 8 epochs). Polyak averaging [13] was used to create the final model used at inference time.

我们的网络使用DistBelief[4]分布式机器学习系统进行训练，这个系统中模型和数据都有一定的并行性。虽然我们主要使用基于CPU的实现，据粗略估计，GoogLeNet网络使用少数几个高端GPU在一个星期内就可以训练结束，主要的限制在于内存使用。我们的训练使用异步随机梯度下降法，动量0.9[17]，固定学习速率（每8个epoch降低学习速率4%）。使用Polyak平均[13]生成推理时使用的模型。

Our image sampling methods have changed substantially over the months leading to the competition, and already converged models were trained on with other options, sometimes in conjunction with changed hyperparameters, like dropout and learning rate, so it is hard to give a definitive guidance to the most effective single way to train these networks. To complicate matters further, some of the models were mainly trained on smaller relative crops, others on larger ones, inspired by [8]. Still, one prescription that was verified to work very well after the competition includes sampling of various sized patches of the image whose size is distributed evenly between 8% and 100% of the image area and whose aspect ratio is chosen randomly between 3/4 and 4/3. Also, we found that the photometric distortions by Andrew Howard [8] were useful to combat overfitting to some extent. In addition, we started to use random interpolation methods (bilinear, area, nearest neighbor and cubic, with equal probability) for resizing relatively late and in conjunction with other hyperparameter changes, so we could not tell definitely whether the final results were affected positively by their use.

我们的图像采样方法在进入比赛前几个月变化很大，已经收敛的模型训练时有很多条件，有时候改变了超参数，比如dropout率和学习速率，所以很难给出一个确定的训练网络的有效指南。更复杂的是，一些模型主要在相对较小的剪切上训练的，其他的在较大的剪切上，这是受[8]启发得到的。但有一个技巧是在竞赛后得到验证确实有效的，在从图像中采样不同尺寸的图像块，其大小在8%到100%间均匀分布，纵横比在3/4与4/3之间随机选择。同时，我们发现Andrew Howard [8]的光度变形在一定程度上有助于对抗过拟合。另外，我们使用了随机插值法（双线性，区域，最近邻，三次，使用概率相等）来改变图像尺寸，同时还改变了一些其他超参数，所以我们不能确定的说最后的结果哪个因素对结果是确定的正面影响。

## 7 ILSVRC 2014 Classification Challenge Setup and Results

The ILSVRC 2014 classification challenge involves the task of classifying the image into one of 1000 leaf-node categories in the Imagenet hierarchy. There are about 1.2 million images for training, 50,000 for validation and 100,000 images for testing. Each image is associated with one ground truth category, and performance is measured based on the highest scoring classifier predictions. Two numbers are usually reported: the top-1 accuracy rate, which compares the ground truth against the first predicted class, and the top-5 error rate, which compares the ground truth against the first 5 predicted classes: an image is deemed correctly classified if the ground truth is among the top-5, regardless of its rank in them. The challenge uses the top-5 error rate for ranking purposes.

ILSVRC 2014分类挑战赛的任务是将图像分为1000类，训练图像有120万幅，验证图像5万幅，测试图像10万。每幅图都有真值类别，性能衡量是根据最高分类器预测得分。要得到两个结果，top-1准确率，将真值与预测的第一个类别进行比较，top-5错误率，将真值与前5个预测的类别比较，如果在这5个预测里就认为是正确，不论其排序如何。挑战赛用top-5错误率来进行排序。

We participated in the challenge with no external data used for training. In addition to the training techniques aforementioned in this paper, we adopted a set of techniques during testing to obtain a higher performance, which we elaborate below.

我们参加了这次挑战赛，没有用外部数据进行训练。除了本文前面提到的训练方法，我们还采用了一些技术来得到更好的表现，如下所述：

1. We independently trained 7 versions of the same GoogLeNet model (including one wider version), and performed ensemble prediction with them. These models were trained with the same initialization (even with the same initial weights, mainly because of an oversight) and learning rate policies, and they only differ in sampling methodologies and the random order in which they see input images.

- 我们独立的训练了GoogLeNet模型的7个版本（包括一个更宽的模型），然后用这些版本集成起来预测。这些模型训练用的一样的初始值策略（连初始权值都是一样的，这主要是因为疏忽）和学习速率策略，只是在采样方法和输入图像的顺序上不同。

2. During testing, we adopted a more aggressive cropping approach than that of Krizhevsky et al. [9]. Specifically, we resize the image to 4 scales where the shorter dimension (height or width) is 256, 288, 320 and 352 respectively, take the left, center and right square of these resized images (in the case of portrait images, we take the top, center and bottom squares). For each square, we then take the 4 corners and the center 224×224 crop as well as the square resized to 224×224, and their mirrored versions. This results in 4×3×6×2 = 144 crops per image. A similar approach was used by Andrew Howard [8] in the previous year’s entry, which we empirically verified to perform slightly worse than the proposed scheme. We note that such aggressive cropping may not be necessary in real applications, as the benefit of more crops becomes marginal after a reasonable number of crops are present (as we will show later on).

- 在测试中，我们一种比Krizhevsky et al. [9]更激进的剪切方法。即，我们将图像尺寸改变成4类，其较小的一维分别是256,288,320和352，取这些图像的左部、中部和右部正方形（如果是竖长图，就是上中下部正方形）。对于每个方形图像，都有4个角和中间的224×224大小剪切块，还将图像尺寸改变为224×224，以及这些图像的镜像图。所以每幅图可以得到4×3×6×2=144个剪切块。Andrew Howard [8]在去年用了类似的方法，我们验证了一下，发现比我们提出的方案效果略差了一些。这种激进的剪切方法在真实应用中不一定是必要的，因为建切块的收益在一定倍数之后会越来越小（后面我们会给出例子）。

3. The softmax probabilities are averaged over multiple crops and over all the individual classifiers to obtain the final prediction. In our experiments we analyzed alternative approaches on the validation data, such as max pooling over crops and averaging over classifiers, but they lead to inferior performance than the simple averaging.

- 多个crops以及所有分类器的Softmax概率进行了平均，以得到最后预测。在我们的试验中，我们分析了验证数据上的替代方法，比如crops上的max pooling和分类器的平均，但都比简单的平均效果要差。

In the remainder of this paper, we analyze the multiple factors that contribute to the overall performance of the final submission.

论文的余下部分，我们分析对于最后提交版本的总效果做出贡献的多个因素。

Our final submission in the challenge obtains a top-5 error of 6.67% on both the validation and testing data, ranking the first among other participants. This is a 56.5% relative reduction compared to the SuperVision approach in 2012, and about 40% relative reduction compared to the previous year’s best approach (Clarifai), both of which used external data for training the classifiers. The following table shows the statistics of some of the top-performing approaches.

最后提交给挑战赛的版本在验证集合测试集上都得到了top-5错误率6.67%的成绩，排名第一。比2012年的SuperVision方法下降了56.5%，比去年最好的方法(Clarifai)下降了大约40%，而这两个模型都用了外部数据进行了训练。下表是几个表现最好的方法的统计数据。

Table 2 Classification Performance

Team | Year | Place | Error (top-5) | Uses external data
--- | --- | --- | --- | ---
SuperVision | 2012 | 1st | 16.4% | no
SuperVision | 2012 | 1st | 15.3% | Imagenet 22k
Clarifai | 2013 | 1st | 11.7% | no
Clarifai | 2013 | 1st | 11.2% | Imagenet 22k
MSRA | 2014 | 3rd | 7.35% | no
VGG | 2014 | 2nd | 7.32% | no
GoogLeNet | 2014 | 1st | 6.67% | no

We also analyze and report the performance of multiple testing choices, by varying the number of models and the number of crops used when predicting an image in the following table. When we use one model, we chose the one with the lowest top-1 error rate on the validation data. All numbers are reported on the validation dataset in order to not overfit to the testing data statistics.

我们还分析了多个测试选项的性能对比，改变了模型数量和crop数量，如下表。当我们用一个模型，我们选择了在验证数据集上top-1错误率最低的那个。所有的数据都是在验证数据集上得到的，为了不要在测试集上产生过拟合的效果。

Table 3: GoogLeNet classification performance break down

Number of models | Number of Crops | Cost | Top-5 error | compared to base
--- | --- | --- | --- | ---
1 | 1 | 1 | 10.07% | base
1 | 10 | 10 | 9.15% | -0.92%
1 | 144 | 144 | 7.89% | -2.18%
7 | 1 | 7 | 8.09% | -1.98%
7 | 10 | 70 | 7.62% | -2.45%
7 | 144 | 1008 | 6.67% | -3.45%

## 8 ILSVRC 2014 Detection Challenge Setup and Results

The ILSVRC detection task is to produce bounding boxes around objects in images among 200 possible classes. Detected objects count as correct if they match the class of the groundtruth and their bounding boxes overlap by at least 50% (using the Jaccard index). Extraneous detections count as false positives and are penalized. Contrary to the classification task, each image may contain many objects or none, and their scale may vary from large to tiny. Results are reported using the mean average precision (mAP).

ILSVRC检测任务是要在图像中生成围绕目标的包围框，有大约200类目标。检测到的目标如果与真值匹配，且包围框与真值框重叠超过50%（用Jaccard index计算），就算作正确。无关的检测结果算作错误，会有惩罚。与分类任务不同，每个图像可能包含多个目标，或没有目标，尺度也或大或小。结果以mAP，即平均AP值。

The approach taken by GoogLeNet for detection is similar to the R-CNN by [6], but is augmented with the Inception model as the region classifier. Additionally, the region proposal step is improved by combining the Selective Search [20] approach with multi-box [5] predictions for higher object bounding box recall. In order to cut down the number of false positives, the superpixel size was increased by 2×. This halves the proposals coming from the selective search algorithm. We added back 200 region proposals coming from multi-box [5] resulting, in total, in about 60% of the proposals used by [6], while increasing the coverage from 92% to 93%. The overall effect of cutting the number of proposals with increased coverage is a 1% improvement of the mean average precision for the single model case. Finally, we use an ensemble of 6 ConvNets when classifying each region which improves results from 40% to 43.9% accuracy. Note that contrary to R-CNN, we did not use bounding box regression due to lack of time.

GoogLeNet采用的检测方法是与[6]中的R-CNN类似，但是用Inception模型作为区域分类器作为增强。另外，区域建议步骤中，将Selective Search [20]方法和multi-box [5]方法结合。为了减少false positive，超像素尺寸增大2倍。这使selective search算法得到的建议区域减半，我们从multi-box [5]增加了200个建议区域，大约是[6]中60%的建议区域，将覆盖率从92%增加到了93%。减少建议区域数量，增加覆盖区域，对单模型的情况共计有1%的mAP提升效果。最后，我们在对区域分类时使用了集成了6个convnets的方法，结果从40%提升到43.9%。注意与R-CNN比，由于缺少时间，没有使用包围框回归。

We first report the top detection results and show the progress since the first edition of the detection task. Compared to the 2013 result, the accuracy has almost doubled. The top performing teams all use Convolutional Networks. We report the official scores in Table 4 and common strategies for each team: the use of external data, ensemble models or contextual models. The external data is typically the ILSVRC12 classification data for pre-training a model that is later refined on the detection data. Some teams also mention the use of the localization data. Since a good portion of the localization task bounding boxes are not included in the detection dataset, one can pre-train a general bounding box regressor with this data the same way classification is used for pre-training. The GoogLeNet entry did not use the localization data for pretraining.

我们首先给出检测结果，还有从第一版检测以来的进步。与2013年的结果相比，准确率几乎加倍。性能最好的组都用了卷积网络。下表4中是官方得分，以及每个队伍的常用策略：外部数据使用，集成模型或contextual模型。外部数据通常是ILSVRC-12分类数据，在此之上预训练模型，然后在检测数据上精调。一些团队提到使用了定位数据。因为相当一部分定位任务的包围框没有在检测数据集中，可以用这个数据预训练一个一般的包围框回归器，这跟在分类任务中用到的预训练一样。GoogLeNet没有用定位数据进行预训练。

Table 4: Detection performance

Team | Year | Place | mAP | external data | ensemble | approach
--- | --- | --- | --- | --- | --- | --- 
UvA-Euvision | 2013 | 1st | 22.6% | none | ? | Fisher vectors
Deep Insight | 2014 | 3rd | 40.5% | ImageNet 1k | 3 | CNN
CUHK DeepID-Net | 2014 | 2nd | 40.7% | ImageNet 1k | ? | CNN
GoogLeNet | 2014 | 1st | 43.9% | ImageNet 1k | 6 | CNN

In Table 5, we compare results using a single model only. The top performing model is by Deep Insight and surprisingly only improves by 0.3 points with an ensemble of 3 models while the GoogLeNet obtains significantly stronger results with the ensemble.

在表5中，我们比较了只用一个模型的结果。最高表现是Deep Insight的，用了3个模型的集成只提高了0.3个百分点，而GoogLeNet用集成模型显著提高了结果。

Table 5: Single model performance for detection

Team | mAP | Contextual model | Bounding box regression
--- | --- | --- | ---
Trimps-Soushen | 31.6% | no | ?
Berkeley Vision | 34.5% | no | yes
UvA-Euvision | 35.4% | ? | ?
CUHK DeepID-Net2 | 37.7% | no | ?
GoogLeNet | 38.02% | no | no
Deep Insight | 40.2% | yes | yes

## 9 Conclusions

Our results seem to yield a solid evidence that approximating the expected optimal sparse structure by readily available dense building blocks is a viable method for improving neural networks for computer vision. The main advantage of this method is a significant quality gain at a modest increase of computational requirements compared to shallower and less wide networks. Also note that our detection work was competitive despite of neither utilizing context nor performing bounding box regression and this fact provides further evidence of the strength of the Inception architecture. Although it is expected that similar quality of result can be achieved by much more expensive networks of similar depth and width, our approach yields solid evidence that moving to sparser architectures is feasible and useful idea in general. This suggest promising future work towards creating sparser and more refined structures in automated ways on the basis of [2].

我们的结果证明了用现有的稠密模块来近似最佳稀疏结构是可行的，可以改进计算机视觉中的神经网络。其主要优势是可以明显改善质量，而计算量与浅层或更窄的模型比，增加不算多。还要注意到，我们的检测工作的结果也是很有竞争力的，而我们既没用上下文，也没有进行包围框回归，这说明Inception模型在这方面的潜力。

## 10 Acknowledgements

We would like to thank Sanjeev Arora and Aditya Bhaskara for fruitful discussions on [2]. Also we are indebted to the DistBelief [4] team for their support especially to Rajat Monga, Jon Shlens, Alex Krizhevsky, Jeff Dean, Ilya Sutskever and Andrea Frome. We would also like to thank to Tom Duerig and Ning Ye for their help on photometric distortions. Also our work would not have been possible without the support of Chuck Rosenberg and Hartwig Adam.

## References
- [1] Know your meme: We need to go deeper. http://knowyourmeme.com/memes/we-need-to-go-deeper. Accessed: 2014-09-15.
- [2] Sanjeev Arora, Aditya Bhaskara, Rong Ge, and Tengyu Ma. Provable bounds for learning some deep representations. CoRR, abs/1310.6343, 2013.
- [3] Umit V. C¸atalyürek, Cevdet Aykanat, and Bora Uc ¸ar. On two-dimensional sparse matrix partitioning: Models, methods, and a recipe. SIAM J. Sci. Comput., 32(2):656–683, February 2010.
- [4] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc’aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Quoc V. Le, and Andrew Y. Ng. Large scale distributed deep networks. In P. Bartlett, F.c.n. Pereira, C.j.c. Burges, L. Bottou, and K.q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1232–1240. 2012.
- [5] Dumitru Erhan, Christian Szegedy, Alexander Toshev, and Dragomir Anguelov. Scalable object detection using deep neural networks. In Computer Vision and Pattern Recognition, 2014. CVPR 2014. IEEE Conference on, 2014.
- [6] Ross B. Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Computer Vision and Pattern Recognition, 2014. CVPR 2014. IEEE Conference on, 2014.
- [7] Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. CoRR, abs/1207.0580, 2012.
- [8] Andrew G. Howard. Some improvements on deep convolutional neural network based image classification. CoRR, abs/1312.5402, 2013.
- [9] Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pages 1106–1114, 2012.
- [10] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Comput., 1(4):541–551,December 1989.
- [11] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
- [12] Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. CoRR, abs/1312.4400, 2013.
- [13] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM J. Control Optim., 30(4):838–855, July 1992.
- [14] Pierre Sermanet, David Eigen, Xiang Zhang, Michaël Mathieu, Rob Fergus, and Yann Le-Cun. Overfeat: Integrated recognition, localization and detection using convolutional networks. CoRR, abs/1312.6229, 2013.
- [15] Thomas Serre, Lior Wolf, Stanley M. Bileschi, Maximilian Riesenhuber, and Tomaso Poggio. Robust object recognition with cortex-like mechanisms. IEEE Trans. Pattern Anal. Mach. Intell., 29(3):411–426, 2007.
- [16] Fengguang Song and Jack Dongarra. Scaling up matrix computations on shared-memory manycore systems with 1000 cpu cores. In Proceedings of the 28th ACM International Conference on Supercomputing, ICS ’14, pages 333–342, New York, NY, USA, 2014. ACM.
- [17] Ilya Sutskever, James Martens, George E. Dahl, and Geoffrey E. Hinton. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on Machine Learning, ICML 2013, Atlanta, GA, USA, 16-21 June 2013, volume 28 of JMLR Proceedings, pages 1139–1147. JMLR.org, 2013.
- [18] Christian Szegedy, Alexander Toshev, and Dumitru Erhan. Deep neural networks for object detection. In Christopher J. C. Burges, Léon Bottou, Zoubin Ghahramani, and Kilian Q. Weinberger, editors, Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States., pages 2553–2561, 2013.
- [19] Alexander Toshev and Christian Szegedy. Deeppose: Human pose estimation via deep neural networks. CoRR, abs/1312.4659, 2013.
- [20] Koen E. A. van de Sande, Jasper R. R. Uijlings, Theo Gevers, and Arnold W. M. Smeulders. Segmentation as selective search for object recognition. In Proceedings of the 2011 International Conference on Computer Vision, ICCV ’11, pages 1879–1886, Washington, DC, USA, 2011. IEEE Computer Society.
- [21] Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In David J. Fleet, Tomás Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I, volume 8689 of Lecture Notes in Computer Science, pages 818–833. Springer, 2014.