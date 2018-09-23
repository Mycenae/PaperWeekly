# Rethinking the Inception Architecture for Computer Vision 重新思考计算机视觉中的Inception架构

Christian Szegedy et al. Google Inc.

## Abstract

Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we are exploring ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error and 17.3% top-1 error.

卷积网络是现在很多任务的计算机视觉最佳解决方案的算法核心。2014年以来，非常深的卷积网络成为主流，在很多基准测试中得到了很大性能提升。虽然模型规模增加和计算代价增加一般马上会带来性能提升（只要有标记数据进行训练），计算效率和低参数数量仍然是很多应用情景的使能因素，如移动视觉和大数据情景。这里我们探索方法在加大网络的规模的同时，尽量高效的利用计算资源，主要通过合理的分解卷积和激进的正则化。我们在ILSVRC-2012分类验证集上验证了我们的方法，与现有最佳方法相比取得了明显的提升：21.2%的top-1错误率，5.6%的top-5错误率，采用单模型评估，网络计算量为每次推理50亿乘加运算，参数数量少于2500万。经过4个模型的集成，和多剪切块评估，我们取得了3.5%的top-5错误率和17.3%的top-1错误率。

## 1. Introduction

Since the 2012 ImageNet competition [16] winning entry by Krizhevsky et al [9], their network “AlexNet” has been successfully applied to a larger variety of computer vision tasks, for example to object-detection [5], segmentation [12], human pose estimation [22], video classification [8], object tracking [23], and superresolution [3]. These successes spurred a new line of research that focused on finding higher performing convolutional neural networks. Starting in 2014, the quality of network architectures significantly improved by utilizing deeper and wider networks. VGGNet [18] and GoogLeNet [20] yielded similarly high performance in the 2014 ILSVRC [16] classification challenge. One interesting observation was that gains in the classification performance tend to transfer to significant quality gains in a wide variety of application domains. This means that architectural improvements in deep convolutional architecture can be utilized for improving performance for most other computer vision tasks that are increasingly reliant on high quality, learned visual features. Also, improvements in the network quality resulted in new application domains for convolutional networks in cases where AlexNet features could not compete with hand engineered, crafted solutions, e.g. proposal generation in detection[4].

自从2012年ImageNet竞赛[16]Krizhevsky et al [9]成为获胜者，他们的网络AlexNet成功的应用在了大量计算机视觉任务中，如目标检测，分割，人体姿态估计，视频分类，目标跟踪，超分辨率等。这些成功刺激了新的研究线，聚焦在寻找更高性能的卷积神经网络。2014年开始，网络架构的质量得到了明显改进，网络变得更宽更深。VGGNet和GoogLeNet在ILSVRC-2014分类挑战中都得到了很高的性能。一个有趣的观察是，分类性能上的提升，会转移到很大范围内的应用都得到提升。这意味着卷积神经网络的架构改进可以应用于改进其他多数计算机视觉任务，其任务要依赖于高质量、学习到的视觉特征。同时，网络质量的改进使卷积网络有了新的应用领域，如检测任务中的建议产生，而当时AlexNet特征不能与手工设计的解决方案相竞争。

Although VGGNet [18] has the compelling feature of architectural simplicity, this comes at a high cost: evaluating the network requires a lot of computation. On the other hand, the Inception architecture of GoogLeNet [20] was also designed to perform well even under strict constraints on memory and computational budget. For example, GoogleNet employed only 5 million parameters, which represented a 12× reduction with respect to its predecessor AlexNet, which used 60 million parameters. Furthermore, VGGNet employed about 3x more parameters than AlexNet.

虽然VGGNet[18]的特征结构简洁引人注目，但是代价较大：评估这个网络需要大量计算。另一方面，GoogLeNet的Inception架构设计时就要在内存和计算能力的严格限制下也可以表现很好。比如，GoogLeNet只使用了500万参数，是AlexNet参数的1/12，AlexNet使用了6000万参数。更进一步，VGGNet使用了AlexNet3倍的参数。

The computational cost of Inception is also much lower than VGGNet or its higher performing successors [6]. This has made it feasible to utilize Inception networks in big-data scenarios[17], [13], where huge amount of data needed to be processed at reasonable cost or scenarios where memory or computational capacity is inherently limited, for example in mobile vision settings. It is certainly possible to mitigate parts of these issues by applying specialized solutions to target memory use [2], [15] or by optimizing the execution of certain operations via computational tricks [10]. However, these methods add extra complexity. Furthermore, these methods could be applied to optimize the Inception architecture as well, widening the efficiency gap again.

Inception的计算代价比VGGNet要低的多。这使得在大数据场景下可以使用Inception网络，这种场景下有大量的数据需要以很低的代价进行处理，内存和计算能力都是受限的，比如移动视觉的情况。可以通过一些方法来缓解这些问题，比如应用特定的减少内存使用的解决方法[2,15]或用计算技巧来优化操作执行[10]。但这些方法额外增加了复杂度。进一步，这些方法也可以对Inception架构进行优化，这又加大了效率鸿沟。

Still, the complexity of the Inception architecture makes it more difficult to make changes to the network. If the architecture is scaled up naively, large parts of the computational gains can be immediately lost. Also, [20] does not provide a clear description about the contributing factors that lead to the various design decisions of the GoogLeNet architecture. This makes it much harder to adapt it to new use-cases while maintaining its efficiency. For example, if it is deemed necessary to increase the capacity of some Inception-style model, the simple transformation of just doubling the number of all filter bank sizes will lead to a 4x increase in both computational cost and number of parameters. This might prove prohibitive or unreasonable in a lot of practical scenarios, especially if the associated gains are modest. In this paper, we start with describing a few general principles and optimization ideas that that proved to be useful for scaling up convolution networks in efficient ways. Although our principles are not limited to Inception-type networks, they are easier to observe in that context as the generic structure of the Inception style building blocks is flexible enough to incorporate those constraints naturally. This is enabled by the generous use of dimensional reduction and parallel structures of the Inception modules which allows for mitigating the impact of structural changes on nearby components. Still, one needs to be cautious about doing so, as some guiding principles should be observed to maintain high quality of the models.

而Inception架构的复杂度也使改变网络更加困难。如果架构规模简单增加，计算量上的优势很快就没有了。而且，[20]没有明确阐述GoogLeNet架构的设计决定因素。这使应用新场景的同时保持效率更加困难。比如，如果确实需要增加Inception类模型的规模，简单的方法是所有滤波器组大小加倍，这将带来4倍的计算量和参数数量的增加。这在很多实际应用中是不可行的，尤其是如果性能没有响应的提高的情况下。本文中，我们首先描述了一些简单原则和优化思想，在增加卷积网络规模中很有用。我们的原则没有限制在Inception类的网络情况下，但Inception类模块的一般结构是很灵活的，可以包含这些限制，很容易可以看出来。降维的使用，以及Inception模块的并行结构，使得结构变化对周边组件的影响得到缓和，这都是有利因素。但仍然需要小心的进行，确保模型的高性能。

## 2. General Design Principles 一般设计原则

Here we will describe a few design principles based on large-scale experimentation with various architectural choices with convolutional networks. At this point, the utility of the principles below are speculative and additional future experimental evidence will be necessary to assess their accuracy and domain of validity. Still, grave deviations from these principles tended to result in deterioration in the quality of the networks and fixing situations where those deviations were detected resulted in improved architectures in general.

这里我们描述几个基于大规模试验的设计原则，这些大规模试验的卷积网络的架构是不同的。在这一点上，这些原则具有推测性，未来需要更多的试验来验证其准确性和有效性。但是，如果偏离这些原则，那么网络的质量会恶化，不偏离的话，性能一般就会得到改善。

1. Avoid representational bottlenecks, especially early in the network. Feed-forward networks can be represented by an acyclic graph from the input layer(s) to the classifier or regressor. This defines a clear direction for the information flow. For any cut separating the inputs from the outputs, one can access the amount of information passing though the cut. One should avoid bottlenecks with extreme compression. In general the representation size should gently decrease from the inputs to the outputs before reaching the final representation used for the task at hand. Theoretically, information content can not be assessed merely by the dimensionality of the representation as it discards important factors like correlation structure; the dimensionality merely provides a rough estimate of information content.

- 避免表示瓶颈，尤其在网络的初期。前向网络，从输入层到分类器或回归器，可以表示为无环图。这定义了清晰的信息流向。在任意处分割输入和输出，可以访问经过这个分割的信息。需要避免极度压缩的瓶颈。一般情况下，表示的大小应当从输入到输出缓缓减少，最后达到任务用到的表示。理论上，信息内容不能仅靠表示的维度来评估，因为丢弃了重要的因素如相关性结构；维数只是提供了信息内容粗略的估计。

2. Higher dimensional representations are easier to process locally within a network. Increasing the activations per tile in a convolutional network allows for more disentangled features. The resulting networks will train faster.

- 网络中的高层维度表示比较容易处理。卷积网络中每层增大激活，可以得到更加明晰的特征。得到的网络训练起来更快。

3. Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. For example, before performing a more spread out (e.g. 3 × 3) convolution, one can reduce the dimension of the input representation before the spatial aggregation without expecting serious adverse effects. We hypothesize that the reason for that is the strong correlation between adjacent unit results in much less loss of information during dimension reduction, if the outputs are used in a spatial aggregation context. Given that these signals should be easily compressible, the dimension reduction even promotes faster learning.

- 空间聚合可以通过低维度嵌套得到，而不损失表示能力。比如，在进行卷积（如3×3）前，可以在空间聚合前降低输入表示的维度，不会有很大的不良作用。我们假设其原因是，如果输出用在空间聚合的上下文中，相邻单元的强相关性在降维过程中得到的信息损失很少。如果这些信号可以很轻易压缩，降维甚至可以使训练加快。

4. Balance the width and depth of the network. Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network. Increasing both the width and the depth of the network can contribute to higher quality networks. However, the optimal improvement for a constant amount of computation can be reached if both are increased in parallel. The computational budget should therefore be distributed in a balanced way between the depth and width of the network.

- 网络的宽度和深度要均衡。每层的滤波器数量和网络的深度的均衡，可以达到网络的最佳性能。宽度和深度同时增加，可能使网络质量增加。但，在计算量保持不变的情况下，要得到最佳改进，就要两个同时增加。所以，计算量预算应当在网络的宽度和深度中均衡分布。

Although these principles might make sense, it is not straightforward to use them to improve the quality of networks out of box. The idea is to use them judiciously in ambiguous situations only.

虽然这些原则可能有意义，但直接进行使用以改进网络质量仍然不是简单的事。要在模糊的环境中明智的使用这些原则。

## 3. Factorizing Convolutions with Large Filter Size 分解大滤波器卷积层

Much of the original gains of the GoogLeNet network [20] arise from a very generous use of dimension reduction. This can be viewed as a special case of factorizing convolutions in a computationally efficient manner. Consider for example the case of a 1 × 1 convolutional layer followed by a 3 × 3 convolutional layer. In a vision network, it is expected that the outputs of near-by activations are highly correlated. Therefore, we can expect that their activations can be reduced before aggregation and that this should result in similarly expressive local representations.

GoogLeNet的性能提升多是由于降维的使用。这可以看做是分解卷积的特殊情况，其运算效率也提高了。比如，考虑这种情况，1×1的卷积层后面是3×3的卷积层。在视觉网络中，一般相邻的输出激活是高度相关的。所以，我们可以在聚集之前就减少激活，这在局部表示上的结果应该是类似的。

Here we explore other ways of factorizing convolutions in various settings, especially in order to increase the computational efficiency of the solution. Since Inception networks are fully convolutional, each weight corresponds to one multiplication per activation. Therefore, any reduction in computational cost results in reduced number of parameters. This means that with suitable factorization, we can end up with more disentangled parameters and therefore with faster training. Also, we can use the computational and memory savings to increase the filter-bank sizes of our network while maintaining our ability to train each model replica on a single computer.

这里我们探索在各种设置下，其他分解卷积的方法，尤其是增加计算效率的解决方案。由于Inception网络是全卷积的，每个权值对应每个激活的一次乘法。所以，任何降低计算代价的方法都减少参数数量。这意味着有适当的分解方法的话，我们可以得到更少纠缠的参数，所以更快的训练过程。同时，我们可以使用节约的计算量和内存来增加网络滤波器组的规模，这样还可以保持在一台计算机上训练每个模型的能力。

### 3.1. Factorization into smaller convolutions 分解成小型卷积

Convolutions with larger spatial filters (e.g. 5 × 5 or 7 × 7) tend to be disproportionally expensive in terms of computation. For example, a 5 × 5 convolution with n filters over a grid with m filters is 25/9 = 2.78 times more computationally expensive than a 3 × 3 convolution with the same number of filters. Of course, a 5×5 filter can capture dependencies between signals between activations of units further away in the earlier layers, so a reduction of the geometric size of the filters comes at a large cost of expressiveness. However, we can ask whether a 5×5 convolution could be replaced by a multi-layer network with less parameters with the same input size and output depth. If we zoom into the computation graph of the 5 × 5 convolution, we see that each output looks like a small fully-connected network sliding over 5×5 tiles over its input (see Figure 1). Since we are constructing a vision network, it seems natural to exploit translation invariance again and replace the fully connected component by a two layer convolutional architecture: the first layer is a 3×3 convolution, the second is a fully connected layer on top of the 3 × 3 output grid of the first layer (see Figure 1). Sliding this small network over the input activation grid boils down to replacing the 5 × 5 convolution with two layers of 3 × 3 convolution (compare Figure 4 with 5).

大型空域滤波器（如5×5,7×7）的卷积在计算上非常耗时。比如，m个5×5×n的卷积滤波器与相应数目的3×3滤波器比，计算量是其25/9=2.78倍。当然，5×5的滤波器捕捉到更多的激活信号单元间的依赖性，所以几何大小的减小带来的是表示力上的降低。但是，5×5的卷积是不是可以被较少参数的多层网络代替呢，其输入大小和输出深度都是一样的。如果我们放大观看5×5卷积的计算图，我们可以看到每个输出都像一样小型全卷积网络，滑过输入的5×5块（见图1）。由于我们要构建一个视觉网络，很自然的我们要再次利用平移不变性，将全卷积模块替换成2层卷积结构：第一层是3×3卷积，第二层是3×3输出上的全卷积层（见图1）。将这个较小的网络滑过输入，也就是将5×5卷积替换为2个3×3卷积（与图4、5相比）。

This setup clearly reduces the parameter count by sharing the weights between adjacent tiles. To analyze the expected computational cost savings, we will make a few simplifying assumptions that apply for the typical situations: We can assume that n = αm, that is that we want to change the number of activations/unit by a constant alpha factor. Since the 5 × 5 convolution is aggregating, α is typically slightly larger than one (around 1.5 in the case of GoogLeNet). Having a two layer replacement for the 5 × 5 layer, it seems reasonable to reach this expansion in two steps: increasing the number of filters by $\sqrt{α}$ in both steps. In order to simplify our estimate by choosing α = 1 (no expansion), If we would naivly slide a network without reusing the computation between neighboring grid tiles, we would increase the computational cost. sliding this network can be represented by two 3×3 convolutional layers which reuses the activations between adjacent tiles. This way, we end up with a net $\frac{9+9}{25}$× reduction of computation, resulting in a relative gain of 28% by this factorization. The exact same saving holds for the parameter count as each parameter is used exactly once in the computation of the activation of each unit. Still, this setup raises two general questions: Does this replacement result in any loss of expressiveness? If our main goal is to factorize the linear part of the computation, would it not suggest to keep linear activations in the first layer? We have ran several control experiments (for example see figure 2) and using linear activation was always inferior to using rectified linear units in all stages of the factorization. We attribute this gain to the enhanced space of variations that the network can learn especially if we batch-normalize [7] the output activations. One can see similar effects when using linear activations for the dimension reduction components.

这种设定通过在相邻块之间共享权值，明显减少了参数数量。为分析节约的计算代价，我们对典型情况进行几个简化的假设：假设n = αm，即我们希望将激活/单元数减少常数因子α倍。由于5×5卷积是聚集的，α一般略比1大，在GoogLeNet里约为1.5。有了5×5卷积的两层替代之后，可以通过两步达到这个效果：在两步中将滤波器数目增加到$\sqrt{α}$倍。为简化估计，我们取α=1，如果我们滑动网络的时候没有重复使用相邻块之间的计算，那么计算代价可能增长。滑动网络可以表示为两个3×3卷积层，也就重用了相邻块之间的激活。通过这种方法，我们得到的网络计算量减少到$\frac{9+9}{25}$，也就是通过分解有了28%的性能提升，参数数量也有了响应的减少。但这种设定还有两个一般性问题：这种替换有没有表达力方面的损失？如果我们的主要目标是分解计算的线性部分，为什么我们不保持在第一层的线性激活？我们已经进行了几次控制实验（如图2），在分解的所有过程中，使用线性激活的情况都不如使用ReLU激活。我们认为这种提升应当是由于网络可以学习到的空间变化大了，尤其是如果我们对输出激活进行批归一化。在降维模块中使用线性激活时，可以看到类似的情况。

Figure 1. Mini-network replacing the 5 × 5 convolutions.

Figure 2. One of several control experiments between two Inception models, one of them uses factorization into linear + ReLU layers, the other uses two ReLU layers. After 3.86 million operations, the former settles at 76.2%, while the latter reaches 77.2% top-1 Accuracy on the validation set.

### 3.2. Spatial Factorization into Asymmetric Convolutions 空间分解为非对称卷积

The above results suggest that convolutions with filters larger 3 × 3 a might not be generally useful as they can always be reduced into a sequence of 3 × 3 convolutional layers. Still we can ask the question whether one should factorize them into smaller, for example 2×2 convolutions. However, it turns out that one can do even better than 2 × 2 by using asymmetric convolutions, e.g. n×1. For example using a 3 × 1 convolution followed by a 1 × 3 convolution is equivalent to sliding a two layer network with the same receptive field as in a 3 × 3 convolution (see figure 3). Still the two-layer solution is 33% cheaper for the same number of output filters, if the number of input and output filters is equal. By comparison, factorizing a 3 × 3 convolution into a two 2 × 2 convolution represents only a 11% saving of computation.

上述结果说明，大于3×3的卷积一般不如将其分解为3×3卷积层系列更有用。我们还可以提问，是不是应该分解成更小的，如2×2的卷积呢？但是，我们还可以将其分解成非对称卷积，这样效果更好，如n×1卷积。比如，使用一个3×1卷积，后续接着1×3卷积，这样的两层卷积滑过图像的感受野与3×3卷积是一样的（见图3）。如果一层网络的输入输出滤波器数量相等，在同样数目的输出滤波器情况下，这样的2层网络比一层计算量少了33%。而将3×3卷积分解成2个2×2卷积只减少了11%的运算量。

Figure 3. Mini-network replacing the 3 × 3 convolutions. The lower layer of this network consists of a 3 × 1 convolution with 3 output units.

In theory, we could go even further and argue that one can replace any n × n convolution by a 1 × n convolution followed by a n×1 convolution and the computational cost saving increases dramatically as n grows (see figure 6). In practice, we have found that employing this factorization does not work well on early layers, but it gives very good results on medium grid-sizes (On m×m feature maps, where m ranges between 12 and 20). On that level, very good results can be achieved by using 1 × 7 convolutions followed by 7 × 1 convolutions.

理论上，我们可以更进一步，将n×n卷积替换成1×n卷积和n×1卷积，当n增大时，计算量的减少比例越来越多（见图6）。在实践中，我们发现采用分解的方法在开始的层中效果不是很好，但在中型网格规模上效果较好（在m×m的特征图上，m介于12和20之间）。在这个水平上，用1×7和7×1的卷积一起可以取得很好的效果。

## 4. Utility of Auxiliary Classifiers 辅助分类器工具

[20] has introduced the notion of auxiliary classifiers to improve the convergence of very deep networks. The original motivation was to push useful gradients to the lower layers to make them immediately useful and improve the convergence during training by combating the vanishing gradient problem in very deep networks. Also Lee et al[11] argues that auxiliary classifiers promote more stable learning and better convergence. Interestingly, we found that auxiliary classifiers did not result in improved convergence early in the training: the training progression of network with and without side head looks virtually identical before both models reach high accuracy. Near the end of training, the network with the auxiliary branches starts to overtake the accuracy of the network without any auxiliary branch and reaches a slightly higher plateau.

为改进非常深度网络的收敛效果，[20]引入了辅助分类器的概念。最初的目的是想将梯度推送到较低的层中，使梯度立刻在这些层中产生作用，改进训练过程中的收敛效果，防止非常深度网络中的梯度消失现象。Lee et al[11]也提出理由说，辅助分类器可以推动更稳定的学习和更好的收敛结果。有趣的是，我们发现辅助分类器在训练初期的时候，没有改进收敛效果：有没有辅助分类器的网络训练过程看起来几乎是一样的，一直达到很高的准确率。在训练就要结束的时候，有辅助分类器的网络超过了没有的网络的准确率，达到了一个略高的水平。

Also [20] used two side-heads at different stages in the network. The removal of the lower auxiliary branch did not have any adverse effect on the final quality of the network. Together with the earlier observation in the previous paragraph, this means that original the hypothesis of [20] that these branches help evolving the low-level features is most likely misplaced. Instead, we argue that the auxiliary classifiers act as regularizer. This is supported by the fact that the main classifier of the network performs better if the side branch is batch-normalized [7] or has a dropout layer. This also gives a weak supporting evidence for the conjecture that batch normalization acts as a regularizer.

同时[20]还在网络的不同阶段使用了辅助分类器。低层辅助分类器的去除，不影响网络的最后结果。和我们在前节的结论一起，这说明[20]的原始假设，即辅助分类器会帮助低层特征进化，这可能是错误的。而我们认为辅助分类器可以作为正则化器。这是因为，如果辅助分类器经过批归一化，或有一个dropout层，那么网络的主分类器效果会更好一些。这也间接的说明，批归一化可以作为正则化器使用。

## 5. Efficient Grid Size Reduction 有效的网格规模缩减

Traditionally, convolutional networks used some pooling operation to decrease the grid size of the feature maps. In order to avoid a representational bottleneck, before applying maximum or average pooling the activation dimension of the network filters is expanded. For example, starting a d×d grid with k filters, if we would like to arrive at a d/2 × d/2 grid with 2k filters, we first need to compute a stride-1 convolution with 2k filters and then apply an additional pooling step. This means that the overall computational cost is dominated by the expensive convolution on the larger grid using $2d^2 k^2$ operations. One possibility would be to switch to pooling with convolution and therefore resulting in $2(d/2)^2 k^2$ reducing the computational cost by a quarter. However, this creates a representational bottlenecks as the overall dimensionality of the representation drops to $(d/2)^2 k^2$ resulting in less expressive networks (see Figure 9). Instead of doing so, we suggest another variant the reduces the computational cost even further while removing the representational bottleneck. (see Figure 10). We can use two parallel stride 2 blocks: P and C. P is a pooling layer (either average or maximum pooling) the activation, both of them are stride 2 the filter banks of which are concatenated as in figure 10.

传统上，卷积网络使用一些池化操作来减小特征图的网格大小。为避免表示瓶颈，在进行最大池化或平均池化之前，增大网络滤波器激活的维数。比如，开始时是d×d的大小，k个滤波器，如果想要达到d/2 × d/2大小，2k个滤波器，首先我们需要计算2k个步长为1的卷积，然后进行一次池化操作。这就是说总共的计算量主要是卷积操作的，共$2d^2 k^2$个操作。一种可能性是将卷积与池化的顺序颠倒过来，这样计算量缩减到1/4，即$2(d/2)^2 k^2$个计算。但是，这会产生表示上的瓶颈，因为总计的表示维数跌落到$(d/2)^2 k^2$，得到的网络表示力衰减（见图9）。我们提出一种方法，可以去除网络表示瓶颈，同时大幅缩减计算量，见图10。我们使用两路步长为2的块：P和C，C表示池化层，然后如图10一样进行拼接。

## 6. Inception-v2

Here we are connecting the dots from above and propose a new architecture with improved performance on the ILSVRC 2012 classification benchmark. The layout of our network is given in table 1. Note that we have factorized the traditional 7 × 7 convolution into three 3 × 3 convolutions based on the same ideas as described in section 3.1. For the Inception part of the network, we have 3 traditional inception modules at the 35×35 with 288 filters each. This is reduced to a 17 × 17 grid with 768 filters using the grid reduction technique described in section 5. This is is followed by 5 instances of the factorized inception modules as depicted in figure 5. This is reduced to a 8 × 8 × 1280 grid with the grid reduction technique depicted in figure 10. At the coarsest 8 × 8 level, we have two Inception modules as depicted in figure 6, with a concatenated output filter bank size of 2048 for each tile. The detailed structure of the network, including the sizes of filter banks inside the Inception modules, is given in the supplementary material, given in the model.txt that is in the tar-file of this submission. However, we have observed that the quality of the network is relatively stable to variations as long as the principles from Section 2 are observed. Although our network is 42 layers deep, our computation cost is only about 2.5 higher than that of GoogLeNet and it is still much more efficient than VGGNet.

这里我们将上述的各点综合起来，提出一个新的框架，在ILSVRC-2012分类基准测试中得到了改进的结果。网络的结构如表1。注意我们将传统的7×7卷积分解成了3层3×3卷积，这是基于3.1节的思想。对于网络的Inception结构部分，在35×35大小时，有3个传统的Inception模块，每个288滤波器；然后网格缩小到17×17大小，有768个滤波器，网格缩减用的是第5节中描述的方法，然后是5个分解Inception模块，如第5节中描述；然后缩减到8×8×1280网格大小，网格缩减技术为图10中描述的，在最粗糙的8×8水平，我们有2个Inception模块，结构如图6所示，每块拼接滤波器输出，滤波器组大小为2048。网络的详细结构，包括Inception模块滤波器组大小，在补充材料中给出。但是，我们观察到，只要第2节中的原则得到遵守，网络的质量对变化表现相对稳定。虽然网络有42层深，但计算量只比GoogLeNet高出2.5倍，比VGG网络效率要高出不少。

Table 1. The outline of the proposed network architecture. The output size of each module is the input size of the next one. We are using variations of reduction technique depicted Figure 10 to reduce the grid sizes between the Inception blocks whenever applicable. We have marked the convolution with 0-padding, which is used to maintain the grid size. 0-padding is also used inside those Inception modules that do not reduce the grid size. All other layers do not use padding. The various filter bank sizes are chosen to observe principle 4 from Section 2.

type | patch size/stride | input size
--- | --- | ---
conv | 3×3/2 | 299×299×3
conv | 3×3/1 | 149×149×32
conv padded | 3×3/1 | 147×147×32
pool | 3×3/2 | 147×147×64
conv | 3×3/1 | 73×73×64
conv | 3×3/2 | 71×71×80
conv | 3×3/1 | 35×35×192
3×Inception | As in figure 5 | 35×35×288
5×Inception | As in figure 6 | 17×17×768
2×Inception | As in figure 7 | 8×8×1280
pool | 8 × 8 | 8 × 8 × 2048
linear | logits | 1 × 1 × 2048
softmax | classifier | 1 × 1 × 1000

## 7. Model Regularization via Label Smoothing 通过标签平滑来进行模型正则化

Here we propose a mechanism to regularize the classifier layer by estimating the marginalized effect of label-dropout during training.

这里我们提出一种方法来对分类器层进行正则化，就是通过在训练过程中估计标签dropout的边际化效果。

For each training example x, our model computes the probability of each label k ∈ {1...K}: $p(k|x) = \frac{exp(z_k)}{\sum_{i=1}^K exp(z_i)}$. Here, $z_i$ are the logits or unnormalized log-probabilities. Consider the ground-truth distribution over labels q(k|x) for this training example, normalized so that $\sum_k q(k|x) = 1$. For brevity, let us omit the dependence of p and q on example x. We define the loss for the example as the cross entropy: $l = −\sum^K_{k=1} log(p(k))q(k)$. Minimizing this is equivalent to maximizing the expected log-likelihood of a label, where the label is selected according to its ground-truth distribution q(k). Cross-entropy loss is differentiable with respect to the logits $z_k$ and thus can be used for gradient training of deep models. The gradient has a rather simple form: $\frac {∂l} {∂z_k} = p(k)−q(k)$, which is bounded between −1 and 1.

对于每个训练样本x，我们的模型计算每个标签的概率k ∈ {1...K}: $p(k|x) = \frac{exp(z_k)}{\sum_{i=1}^K exp(z_i)}$。这里，$z_i$是logits或未归一化的log概率。考虑这个训练样本标签q(k|x)的真值分布，归一化后有$\sum_k q(k|x) = 1$。为简化表达，我们忽略p和q对x的标记，我们定义样本损失为交叉熵$l = −\sum^K_{k=1} log(p(k))q(k)$。最小化交叉熵与最大化标签的期望log概率是等价的，这里标签是根据其真值分布q(k)选择的。交叉熵损失对logits$z_k$可微分，所以可以用于深度模型的梯度训练。其梯度的形式很简单，$\frac {∂l} {∂z_k} = p(k)−q(k)$，其值在-1与1之间。

Consider the case of a single ground-truth label y, so that q(y) = 1 and q(k) = 0 for all $k \neq y$. In this case, minimizing the cross entropy is equivalent to maximizing the log-likelihood of the correct label. For a particular example x with label y, the log-likelihood is maximized for $q(k) = δ_{k,y}$, where $δ_{k,y}$ is Dirac delta, which equals 1 for k = y and 0 otherwise. This maximum is not achievable for finite $z_k$ but is approached if $z_y >> z_k$ for all $k \neq y$ – that is, if the logit corresponding to the ground-truth label is much great than all other logits. This, however, can cause two problems. First, it may result in over-fitting: if the model learns to assign full probability to the ground-truth label for each training example, it is not guaranteed to generalize. Second, it encourages the differences between the largest logit and all others to become large, and this, combined with the bounded gradient $\frac{∂l}{∂z_k}$, reduces the ability of the model to adapt. Intuitively, this happens because the model becomes too confident about its predictions.

考虑单真值标签y的情况，那么对于所有的$k \neq y$，有q(y) = 1和q(k) = 0。在这种情况下，最小化交叉熵与最大化正确标签的log概率是等价的。对于任一带有标签y的样本x，其log概率在$q(k) = δ_{k,y}$的时达到最大化，这里$δ_{k,y}$为狄拉克函数，在k=y时等于1，在其他情况等于0。对于有限的$z_k$来说，其最大值是不能达到的，但如果对于所有的$k \neq y$都有$z_y >> z_k$的话，那么就可以逼近了，也就是说，如果真值对应的logit比其他所有的logit都大的多。但这可能导致两个问题。第一，可能导致过拟合：如果模型学会了对每个训练样本的真值标签都赋予充分最高概率，那么泛化能力就不能保证了。第二，它鼓励最大的logit与其他之间的差距越大越好，而梯度$\frac{∂l}{∂z_k}$是有界的，这减少了模型适应的能力。直觉上来说，这种情况是因为模型对于自己的预测过于自信了。

We propose a mechanism for encouraging the model to be less confident. While this may not be desired if the goal is to maximize the log-likelihood of training labels, it does regularize the model and makes it more adaptable. The method is very simple. Consider a distribution over labels u(k), independent of the training example x, and a smoothing parameter $\epsilon$. For a training example with ground-truth label y, we replace the label distribution $q(k|x) = δ_{k,y}$ with

我们提出一种方法，使模型不那么自信。如果目标是最大化训练标签的log概率，这可能不是很理想的，但它确实使模型正则化了，适应性更好。这个方法非常简单。考虑标签u(k)的分布，与训练样本x独立，平滑参数$\epsilon$，对于一个真值标签y的训练样本，我们将标签分布$q(k|x) = δ_{k,y}$替换为

$$q'(k|x) = (1 − \epsilon)δ_{k,y} + \epsilon u(k)$$

which is a mixture of the original ground-truth distribution q(k|x) and the fixed distribution u(k), with weights 1 − $\epsilon$ and $\epsilon$, respectively. This can be seen as the distribution of the label k obtained as follows: first, set it to the ground-truth label k = y; then, with probability $\epsilon$, replace k with a sample drawn from the distribution u(k). We propose to use the prior distribution over labels as u(k). In our experiments, we used the uniform distribution u(k) = 1/K, so that

这是原来的真值分布q(k|x)与固定分布u(k)的混合，其权值分别为1 − $\epsilon$和$\epsilon$。这可以视为标签k的分布是这样得到的：首先，将其设为真值标签k=y；然后，以$\epsilon$概率将k替换为概率为u(k)的样本。我们提出标签的先验概率为u(k)。在我们的试验中，我们使用均匀分布u(k) = 1/K，这样

$$q'(k|x) = (1 − \epsilon)δ_{k,y} + \epsilon /K$$

We refer to this change in ground-truth label distribution as label-smoothing regularization, or LSR. 我们称真值标签的这个改变为标签平滑正则化，或LSR。

Note that LSR achieves the desired goal of preventing the largest logit from becoming much larger than all others. Indeed, if this were to happen, then a single q(k) would approach 1 while all others would approach 0. This would result in a large cross-entropy with q'(k) because, unlike $q(k) = δ_{k,y}$ , all q'(k) have a positive lower bound.

注意LRS具有防止最大的logit变得比其他的大得多的理想特性。确实，如果发生了这个，那么单个q(k)会趋向1，其他都趋向0。这将使交叉熵q'(k)很大，因为，与$q(k) = δ_{k,y}$不同，所有q'(k)都有一个正的下界。

Another interpretation of LSR can be obtained by considering the cross entropy: LSR的另一个解释可以从下面的交叉熵得到：

$$H(q',p) = −\sum_{k=1}^K logp(k)q'(k) = (1−\epsilon)H(q,p)+\epsilon H(u,p)$$

Thus, LSR is equivalent to replacing a single cross-entropy loss H(q,p) with a pair of such losses H(q,p) and H(u,p).The second loss penalizes the deviation of predicted label distribution p from the prior u, with the relative weight $\frac{\epsilon}{1−\epsilon}$. Note that this deviation could be equivalently captured by the KL divergence, since $H(u,p) = D_{KL} (u||p) + H(u)$ and H(u) is fixed. When u is the uniform distribution, H(u,p) is a measure of how dissimilar the predicted distribution p is to uniform, which could also be measured (but not equivalently) by negative entropy −H(p); we have not experimented with this approach.

所以，LSR相当于将单交叉熵损失H(q,p)替换为两个损失H(q,p)和H(u,p)。第二个损失函数惩罚的是预测标签分布p与先验u的偏差，相对权重为 $\frac{\epsilon}{1−\epsilon}$。注意这种偏差可以等价的为KL散度捕捉到，因为$H(u,p) = D_{KL} (u||p) + H(u)$，而且H(u)是固定的。当u是均匀分布时，H(u,p)是预测分布p与均分分布不相似的测度，这也可以由负熵-H(p)测量，但不等价；我们没有对这种方法进行试验。

In our ImageNet experiments with K = 1000 classes, we used u(k) = 1/1000 and $/epsilon$ = 0.1. For ILSVRC 2012, we have found a consistent improvement of about 0.2% absolute both for top-1 error and the top-5 error (cf. Table 3).

在我们的ImageNet试验中，K=1000，我们使用u(k)=1/1000，$\epsilon$=0.1。对于ILSVRC-2012，我们发现一个top-1错误率和top-5错误率都稳定提升了0.2%（如表3）。

## 8. Training Methodology 训练方法

We have trained our networks with stochastic gradient utilizing the TensorFlow [1] distributed machine learning system using 50 replicas running each on a NVidia Kepler GPU with batch size 32 for 100 epochs. Our earlier experiments used momentum [19] with a decay of 0.9, while our best models were achieved using RMSProp [21] with decay of 0.9 and $\epsilon$ = 1.0. We used a learning rate of 0.045, decayed every two epoch using an exponential rate of 0.94. In addition, gradient clipping [14] with threshold 2.0 was found to be useful to stabilize the training. Model evaluations are performed using a running average of the parameters computed over time.

我们用TensorFlow训练我们的网络，优化方法为随机梯度下降法，采用分布式机器学习系统，50个副本运行在NVidia Kepler GPU上，批大小32，运行100个epoch。我们早期的试验用了动量，衰减0.9。我们最好的模型是用RMSProp得到的，衰减0.9，$\epsilon$=1.0。我们使用学习速率0.045，每2个epoch乘以0.94进行衰减。另外，阈值2.0的梯度剪切[14]对于稳定训练非常有用。模型评估用的是参数量平均。

## 9. Performance on Lower Resolution Input 低分辨率输入的性能

A typical use-case of vision networks is for the the post-classification of detection, for example in the Multibox [4] context. This includes the analysis of a relative small patch of the image containing a single object with some context. The tasks is to decide whether the center part of the patch corresponds to some object and determine the class of the object if it does. The challenge is that objects tend to be relatively small and low-resolution. This raises the question of how to properly deal with lower resolution input.

视觉网络的一个典型应用情景是检测后分类，比如MultiBox[4]的情况。这包括，分析较小的一个图像块是不是包括某个上下文的单个物体。这个任务是要确定图像中央部分是不是对应着某个物体，如果有，确定目标的类别。其中的挑战是其中的目标通常较小，分辨率较低。这提出一个问题，怎样合理的处理较低分辨率的输入。

The common wisdom is that models employing higher resolution receptive fields tend to result in significantly improved recognition performance. However it is important to distinguish between the effect of the increased resolution of the first layer receptive field and the effects of larger model capacitance and computation. If we just change the resolution of the input without further adjustment to the model, then we end up using computationally much cheaper models to solve more difficult tasks. Of course, it is natural, that these solutions loose out already because of the reduced computational effort. In order to make an accurate assessment, the model needs to analyze vague hints in order to be able to “hallucinate” the fine details. This is computationally costly. The question remains therefore: how much does higher input resolution helps if the computational effort is kept constant. One simple way to ensure constant effort is to reduce the strides of the first two layer in the case of lower resolution input, or by simply removing the first pooling layer of the network.

模型使用高分辨率感受野常常识别性能会很好。但是要分辨第一层感受野增大的分辨率的效果，和更大的模型的容量和计算量的效果。如果我们仅仅改变输入的分辨率，而没有进一步调整模型，那么我们就会用计算量很简单的模型去解决更复杂的任务。当然，这些解决方案由于减少的计算量会已经松动，这很自然。要做出准确的评估，模型需要分析模糊的线索来幻想精细的细节。这使得计算代价很高。那么剩下的问题是：如果计算量保持不变，那么较高的输入分辨率会有多大帮助？一个简单的方法是确保计算量恒定是在较低的输入分辨率时，减小前2层的步长，或者把第一个池化层去掉。

For this purpose we have performed the following three experiments: 为了这个目的，我们进行了以下三个实验：

1. 299 × 299 receptive field with stride 2 and maximum pooling after the first layer. 
2. 151 × 151 receptive field with stride 1 and maximum pooling after the first layer.
3. 79×79 receptive field with stride 1 and without pooling after the first layer.

- 299×299感受野，步长2，第一层后进行最大池化
- 151×151感受野，步长1，第一层后进行最大池化
- 79×79感受野，步长1，第一层后没有池化

All three networks have almost identical computational cost. Although the third network is slightly cheaper, the cost of the pooling layer is marginal and (within 1% of the total cost of the)network. In each case, the networks were trained until convergence and their quality was measured on the validation set of the ImageNet ILSVRC 2012 classification benchmark. The results can be seen in table 2. Although the lower-resolution networks take longer to train, the quality of the final result is quite close to that of their higher resolution counterparts.

这三个网络的计算量基本是一样的。虽然第三个网络计算量略低，但池化层的计算量非常小，只占到整个网络计算量的1%。在这种情况下，网络进行训练直到收敛，其性能用ImageNet ILSVRC-2012分类基准测试。其结果可以见表2。虽然低分辨率训练时间长一些，最终结果的质量与高分辨率的相比非常接近。

However, if one would just naively reduce the network size according to the input resolution, then network would perform much more poorly. However this would an unfair comparison as we would are comparing a 16 times cheaper model on a more difficult task.

但是，如果仅仅因为输入分辨率就简单的减小网络大小，那么网络的表现就会非常差。但这也是一个很不公平的比较，因为是将一个1/16计算量的模型与一个更难的模型之间的比较。

Also these results of table 2 suggest, one might consider using dedicated high-cost low resolution networks for smaller objects in the R-CNN [5] context.

表2的结果说明，可以考虑使用在R-CNN[5]的情况下，较小目标可以使用昂贵的低分辨率网络。

## 10. Experimental Results and Comparisons 试验结果和比较

Table 3 shows the experimental results about the recognition performance of our proposed architecture (Inception-v2) as described in Section 6. Each Inception-v2 line shows the result of the cumulative changes including the highlighted new modification plus all the earlier ones. Label Smoothing refers to method described in Section 7. Factorized 7 × 7 includes a change that factorizes the first 7 × 7 convolutional layer into a sequence of 3 × 3 convolutional layers. BN-auxiliary refers to the version in which the fully connected layer of the auxiliary classifier is also batch-normalized, not just the convolutions. We are referring to the model in last row of Table 3 as Inception-v3 and evaluate its performance in the multi-crop and ensemble settings.

表3所示的是第6节我们提出的Inception-v2的识别性能的试验结果。每个Inception-v2行显示的是包括高亮的新修正和所有以前的修正的累积变化的结果。标签平滑对应的是第7节的方法。分解7×7包括将7×7的第一个卷积层分解成为一系列3×3的卷积层。BN辅助意思是全连接层的辅助分类器也经过了批归一化，不只是卷积。表3的最后一行我们称为Inception-v3，其性能在多剪切块和集成设置下进行评估。

All our evaluations are done on the 48238 non-blacklisted examples on the ILSVRC-2012 validation set, as suggested by [16]. We have evaluated all the 50000 examples as well and the results were roughly 0.1% worse in top-5 error and around 0.2% in top-1 error. In the upcoming version of this paper, we will verify our ensemble result on the test set, but at the time of our last evaluation of BN-Inception in spring [7] indicates that the test and validation set error tends to correlate very well.

我们所有的评估都在ILSVRC-2012验证集上的48238个样本进行，这是[16]建议的。我们还评估了50000个样本，结果top-5错误率略差了0.1%，top-1错误率差了0.2%。在本文未来的版本中，我们将在测试集上验证集成方法，但我们最后一次评估BN-Inception表明，测试集和验证集上的错误率差别非常小。

## 11. Conclusions 结论

We have provided several design principles to scale up convolutional networks and studied them in the context of the Inception architecture. This guidance can lead to high performance vision networks that have a relatively modest computation cost compared to simpler, more monolithic architectures. Our highest quality version of Inception-v3 reaches 21.2%, top-1 and 5.6% top-5 error for single crop evaluation on the ILSVR 2012 classification, setting a new state of the art. This is achieved with relatively modest (2.5×) increase in computational cost compared to the network described in Ioffe et al [7]. Still our solution uses much less computation than the best published results based on denser networks: our model outperforms the results of He et al [6] – cutting the top-5 (top-1) error by 25% (14%) relative, respectively – while being six times cheaper computationally and using at least five times less parameters (estimated). Our ensemble of four Inception-v3 models reaches 3.5% with multi-crop evaluation reaches 3.5% top-5 error which represents an over 25% reduction to the best published results and is almost half of the error of ILSVRC 2014 winining GoogLeNet ensemble.

我们提出了几种设计原则来增大卷积网络，并在Inception架构的环境下进行了研究。这个指南可以得到高性能的视觉网络，与更简单的模型相比，计算量增加并不是非常多。我们最高质量的Inception-v3达到了top-1错误率21.2%，top-5错误率5.6%，这是在单剪切块上的ILSVRC-2012分类试验上的结果，是目前最好的结果。与Ioffe[7]相比，计算量增加了2.5倍。但我们的解决方案还是比稠密网络的最好方法要简单的多：比He et al[6]的结果要好，top-5(top-1)错误率分别降低了25%(14%)，但计算量只是其1/6，参数数量也只是1/5（估计）。我们集成了4中Inception-v3模型，达到了3.5%的多剪切块评估top-5错误率，比最好的模型降低了25%，几乎是ILSVRC-2014获胜者GoogLeNet集成方法错误率的的一半。

We have also demonstrated that high quality results can be reached with receptive field resolution as low as 79×79. This might prove to be helpful in systems for detecting relatively small objects. We have studied how factorizing convolutions and aggressive dimension reductions inside neural network can result in networks with relatively low computational cost while maintaining high quality. The combination of lower parameter count and additional regularization with batch-normalized auxiliary classifiers and label-smoothing allows for training high quality networks on relatively modest sized training sets.

我们还证明了，即使感受野分辨率低达79×79，也可以得到高质量的结果。这在检测相对较小的物体的系统非常有用。我们研究了神经网络中的分解卷积和激进的降维，可以在相对较小的计算量下取得很好的结果。较少的参数数量，额外的正则化，BN辅助分类器和标签平滑，这些的组合使得可以在相对规模不大的训练集上得到高质量的网络。

## References