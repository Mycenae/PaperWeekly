# MnasNet: Platform-Aware Neural Architecture Search for Mobile

Mingxing Tan et al. Google Brain, Google Inc

## Abstract 摘要

Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this paper, we propose an automated mobile neural architecture search (MNAS) approach, which explicitly incorporate model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. Unlike previous work, where latency is considered via another, often inaccurate proxy (e.g., FLOPS), our approach directly measures real-world inference latency by executing the model on mobile phones. To further strike the right balance between flexibility and search space size, we propose a novel factorized hierarchical search space that encourages layer diversity throughout the network. Experimental results show that our approach consistently outperforms state-of-the-art mobile CNN models across multiple vision tasks. On the ImageNet classification task, our MnasNet achieves 75.2% top-1 accuracy with 78ms latency on a Pixel phone, which is 1.8× faster than MobileNetV2 [29] with 0.5% higher accuracy and 2.3× faster than NASNet [36] with 1.2% higher accuracy. Our MnasNet also achieves better mAP quality than MobileNets for COCO object detection. Code is at https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet.

为移动设备设计CNN是很有挑战的，因为移动模型需要小型快速，还需要准确。虽然有很多工作在各个维度上设计改进移动CNNs，但在有非常多架构可能需要考虑的情况下，很难手工对这些折中进行平衡。本文中，我们提出了一种自动移动平台NAS的方法，显式的将模型延迟纳入到主要目标中，这样搜索得到的目标可以取得准确率与延迟的很好折中。之前的工作中，延迟是通过另一个不太准确的代理考虑的（如FLOPs），我们的方法通过在移动手机上运行这个模型，从而直接度量真实世界的推理延迟。为进一步达到灵活性和搜索空间大小的正确平衡，我们提出了一种新的分解的层次式搜索空间，通过网络鼓励层的多样性。试验结果表明，我们的方法在多种视觉任务中，一直超过目前最好的移动CNN模型。在ImageNet分类任务中，我们的MnasNet取得了75.2%的top-1准确率，在Pixel手机上的延迟为78ms，比MobileNetv2快了1.8倍，准确率高了0.5%，比NASNet快了2.3x，准确率高了1.2%。我们的MnasNet在COCO目标检测上也比MobileNets取得了更好的mAP成绩。代码已开源。

## 1. Introduction 引言

Convolutional neural networks (CNN) have made significant progress in image classification, object detection, and many other applications. As modern CNN models become increasingly deeper and larger [31, 13, 36, 26], they also become slower, and require more computation. Such increases in computational demands make it difficult to deploy state-of-the-art CNN models on resource-constrained platforms such as mobile or embedded devices.

CNN在图像分类，目标检测和其他很多应用中取得了很大进展。随着现代CNN模型变得越来越深，越来越大，它们也变得更慢了，需要更多计算量。这种计算量的增加使在资源有限的平台，如移动或嵌入式平台上，很难部署目前最好的CNN模型。

Given restricted computational resources available on mobile devices, much recent research has focused on designing and improving mobile CNN models by reducing the depth of the network and utilizing less expensive operations, such as depthwise convolution [11] and group convolution [33]. However, designing a resource-constrained mobile model is challenging: one has to carefully balance accuracy and resource-efficiency, resulting in a significantly large design space.

移动设备上可用资源有限，很多最近的工作聚焦在设计和改进移动CNN模型，减少网络的深度，利用计算量没那么大的算子，如分层卷积和分组卷积。但是，设计一个资源有限的移动模型是很有挑战的：必须仔细的在准确率和资源效率间取得均衡，这会有很大的设计空间。

In this paper, we propose an automated neural architecture search approach for designing mobile CNN models. Figure 1 shows an overview of our approach, where the main differences from previous approaches are the latency aware multi-objective reward and the novel search space. Our approach is based on two main ideas. First, we formulate the design problem as a multi-objective optimization problem that considers both accuracy and inference latency of CNN models. Unlike in previous work [36, 26, 21] that use FLOPS to approximate inference latency, we directly measure the real-world latency by executing the model on real mobile devices. Our idea is inspired by the observation that FLOPS is often an inaccurate proxy: for example, MobileNet [11] and NASNet [36] have similar FLOPS (575M vs. 564M), but their latencies are significantly different (113ms vs. 183ms, details in Table 1). Secondly, we observe that previous automated approaches mainly search for a few types of cells and then repeatedly stack the same cells through the network. This simplifies the search process, but also precludes layer diversity that is important for computational efficiency. To address this issue, we propose a novel factorized hierarchical search space, which allows layers to be architecturally different yet still strikes the right balance between flexibility and search space size.

本文中，我们提出一种自动NAS方法，设计移动CNN模型。图1给出了我们方法的概览，与之前方法的主要区别是，可以感知延迟的多目标奖励，和新的搜索空间。我们的方法基于两个主要思想。第一，我们将设计问题建模，形成一个多目标优化问题，考虑CNN模型的准确率和推理延迟。与之前的工作[36,26,21]不一样的是，它们使用FLOPs来近似推理延迟，我们是通过在真实的移动设备上运行模型，来直接衡量真实世界的延迟。我们的思想是受到下面的观察结果的启发，即FLOPs通常只是个不精确的代理：比如，MobileNet和NASNet的FLOPs类似(575M vs. 564M)，但其延迟却有非常大差别(113ms vs 183ms，详见表1)。第二，我们观察到，之前的自动方法主要搜索几种单元类型，然后将其重复堆叠起来，形成网络。这简化了搜索过程，但也排除了层的多样性，这是对计算效率非常重要的。为解决这个问题，我们提出一种新的分解式层次式搜索空间，这使得层可以在架构上不同，但仍然可以得到灵活性和搜索空间大小之间的均衡。

Figure 1: An Overview of Platform-Aware Neural Architecture Search for Mobile.

We apply our proposed approach to ImageNet classification [28] and COCO object detection [18]. Figure 2 summarizes a comparison between our MnasNet models and other state-of-the-art mobile models. Compared to the MobileNetV2 [29], our model improves the ImageNet accuracy by 3.0% with similar latency on the Google Pixel phone. On the other hand, if we constrain the target accuracy, then our MnasNet models are 1.8× faster than MobileNetV2 and 2.3× faster thans NASNet [36] with better accuracy. Compared to the widely used ResNet-50 [9], our MnasNet model achieves slightly higher (76.7%) accuracy with 4.8× fewer parameters and 10× fewer multiply-add operations. By plugging our model as a feature extractor into the SSD object detection framework, our model improves both the inference latency and the mAP quality on COCO dataset over MobileNetsV1 and MobileNetV2, and achieves comparable mAP quality (23.0 vs 23.2) as SSD300 [22] with 42× less multiply-add operations.

我们将提出的方法应用于ImageNet分类任务和COCO目标检测任务。图2总结了我们的MnasNet模型和其他目前最好的移动模型之间的比较。与MobileNetV2相比，我们的模型改进了ImageNet准确率3.0%，在Pixel手机上的延迟相似。另一方面，如果我们限制目标准确率，那么我们的MnasNet模型比MobileNetV2快1.8x，比NASNet快2.3x，准确率还更高。与广泛使用的ResNet-50相比，我们的MnasNet模型准确率略微高一些，参数数量少了4.8x，MultiAdd运算少了10x。将我们的模型作为特征提取器，使用在SSD目标检测框架中，我们的模型在COCO数据集上，比MobileNetV1和MobileNetV2的推理延迟和mAP都要高，与SSD300相比，mAP类似(23.0 vs. 23.2)，MultiAdd运算少了42x。

Figure 2: Accuracy vs. Latency Comparison – Our MnasNet models significantly outperforms other mobile models [29, 36, 26] on ImageNet. Details can be found in Table 1.

To summarize, our main contributions are as follows: 总结起来，我们的主要贡献如下：

- We introduce a multi-objective neural architecture search approach that optimizes both accuracy and realworld latency on mobile devices. 我们提出了多目标NAS方法，优化准确率的同时，还优化在移动设备上的真实延迟。

- We propose a novel factorized hierarchical search space to enable layer diversity yet still strike the right balance between flexibility and search space size. 我们提出了一种新型分解层次式搜索空间，可以使层更多样化，而且仍然得到灵活性与搜索空间大小之间的平衡。

- We demonstrate new state-of-the-art accuracy on both ImageNet classification and COCO object detection under typical mobile latency constraints. 我们在典型的移动延迟限制下，在ImageNet分类任务和COCO目标检测任务上，都得到了目前最好的结果。

## 2. Related Work

Improving the resource efficiency of CNN models has been an active research topic during the last several years. Some commonly-used approaches include 1) quantizing the weights and/or activations of a baseline CNN model into lower-bit representations [8, 16], or 2) pruning less important filters according to FLOPs [6, 10], or to platform-aware metrics such as latency introduced in [32]. However, these methods are tied to a baseline model and do not focus on learning novel compositions of CNN operations.

改进CNN模型的资源效率，在过去几年中一直是活跃的研究课题。一些常用的方法包括：1)将基准CNN模型的权重和/或激活进行量化，成低bit表示[8,16]；2)根据FLOPs剪枝不太重要的滤波器[6,10]，或根据平台相关的度量，如延迟[32]。但是，这些方法都是基于固定的基准模型，没有关注学习CNN运算的新的组合。

Another common approach is to directly hand-craft more efficient mobile architectures: SqueezeNet [15] reduces the number of parameters and computation by using lower-cost 1x1 convolutions and reducing filter sizes; MobileNet [11] extensively employs depthwise separable convolution to minimize computation density; ShuffleNets [33, 24] utilize low-cost group convolution and channel shuffle; Condensenet [14] learns to connect group convolutions across layers; Recently, MobileNetV2 [29] achieved state-of-the-art results among mobile-size models by using resource-efficient inverted residuals and linear bottlenecks. Unfortunately, given the potentially huge design space, these handcrafted models usually take significant human efforts.

另一种常见的方法是，直接手工设计更高效的移动架构：SqueezeNet通过使用低成本的1×1卷积，以及降低滤波器大小，来减少参数数量和计算量；MobileNet使用了很多分层可分离卷积，来最小化计算复杂度；ShuffleNet使用低成本的分组卷积和通道混洗；CondenseNet学习连接层之间的分组卷积；最近，MobileNetV2通过使用资源效率高的逆残差和线性瓶颈，在移动大小的模型中得到了目前最好的效果。不幸的是，由于潜在的设计空间非常大，这些人工设计的模型通常需要耗费大量人力。

Recently, there has been growing interest in automating the model design process using neural architecture search. These approaches are mainly based on reinforcement learning [35, 36, 1, 19, 25], evolutionary search [26], differentiable search [21], or other learning algorithms [19, 17, 23]. Although these methods can generate mobile-size models by repeatedly stacking a few searched cells, they do not incorporate mobile platform constraints into the search process or search space. Closely related to our work is MONAS [12], DPP-Net [3], RNAS [34] and Pareto-NASH [4] which attempt to optimize multiple objectives, such as model size and accuracy, while searching for CNNs, but their search process optimizes on small tasks like CIFAR. In contrast, this paper targets real-world mobile latency constraints and focuses on larger tasks like ImageNet classification and COCO object detection.

最近，越来越的工作关注使用NAS的自动模型设计。这些方法主要基于强化学习，演化搜索，可微分搜索，或其他学习算法。虽然这些方法可以生成移动大小的模型，重复堆叠一些搜索到的单元，但它们没有考虑到移动平台的限制，将其纳入到搜索过程或搜过空间中。与我们的工作紧密相关的有MONAS [12], DPP-Net [3], RNAS [34]和Pareto-NASH [4]，都在尝试搜过过程中优化多个目标，如模型大小和准确率，但其搜索过程在小型任务如CIFAR中进行优化。比较之下，本文的目标是真实世界的移动延迟约束，聚焦在更大的任务中，如ImageNet分类和COCO目标检测。

## 3. Problem Formulation 问题描述

We formulate the design problem as a multi-objective search, aiming at finding CNN models with both high-accuracy and low inference latency. Unlike previous architecture search approaches that often optimize for indirect metrics, such as FLOPS, we consider direct real-world inference latency, by running CNN models on real mobile devices, and then incorporating the real-world inference latency into our objective. Doing so directly measures what is achievable in practice: our early experiments show it is challenging to approximate real-world latency due to the variety of mobile hardware/software idiosyncrasies.

我们将设计问题建模为一个多目标搜索，要找到高准确率和低推理延迟的CNN模型。之前的架构搜索方法通常优化的是间接度量，如FLOPs，我们考虑直接优化真实世界的推理延迟，在真实移动设备上运行CNN模型，然后将真实世界推理延迟纳入到目标函数中。这样做直接得到了实践中的目标：我们的早期试验表明，对真实世界的延迟进行近似非常有挑战性，因为移动硬件/软件的特性的多样性很大。

Given a model m, let ACC(m) denote its accuracy on the target task, LAT(m) denotes the inference latency on the target mobile platform, and T is the target latency. A common method is to treat T as a hard constraint and maximize accuracy under this constraint:

给定模型m，令ACC(m)表示其在目标任务上的准确率，LAT(m)表示其在目标移动平台上的推理延迟，T是目标延迟。通常的方法是将T作为硬约束，在这个约束下最大化准确率：

$$maximize_m \space ACC(m) \space subject \space to \space LAT(m) ≤ T$$(1)

However, this approach only maximizes a single metric and does not provide multiple Pareto optimal solutions. Informally, a model is called Pareto optimal [2] if either it has the highest accuracy without increasing latency or it has the lowest latency without decreasing accuracy. Given the computational cost of performing architecture search, we are more interested in finding multiple Pareto-optimal solutions in a single architecture search.

但是，这种方法只最大化了一个度量标准，没有给出多Pareto最佳解。一个模型，如果在不增加延迟下是最高准确率的，或在不降低准确率的情况下延迟最低，就称为是Pareto最佳模型。给定进行架构搜索的计算代价，我们更感兴趣的是，在一个架构搜索中找到多Pareto最佳的解。

While there are many methods in the literature [2], we use a customized weighted product method to approximate Pareto optimal solutions, with optimization goal defined as(We pick the weighted product method because it is easy to customize, but we expect methods like weighted sum should be also fine):

文献中有很多方法，但我们使用了定制权重的方法来近似Pareto最佳解，优化目标定义为（我们选择了加权乘积方法，因为这容易定制，但我们期待加权求和的方法应当也应当很好）：

$$maximize_m \space ACC(m) × [\frac {LAT (m)}{T}]^w$$(2)

where w is the weight factor defined as: 其中权重因子w定义为：

$$w = α, if LAT (m) ≤ T; = β, otherwise$$(3)

where α and β are application-specific constants. An empirical rule for picking α and β is to ensure Pareto-optimal solutions have similar reward under different accuracy-latency trade-offs. For instance, we empirically observed doubling the latency usually brings about 5% relative accuracy gain. Given two models: (1) M1 has latency l and accuracy a; (2) M2 has latency 2l and 5% higher accuracy a · (1 + 5%), they should have similar reward: Reward(M2) = a · (1 + 5%) · (2l/T)^β ≈ Reward(M1) = a · (l/T)^β. Solving this gives β ≈ −0.07. Therefore, we use α = β = −0.07 in our experiments unless explicitly stated.

其中α和β是应用专属的常量。选择α和β的一个经验规则是，确保Pareto最佳的解，在不同的准确率-延迟折中下，有类似的奖励。比如，我们从经验观察到，延迟加倍，通常会来带5%的相对准确率收益。给定两个模型：(1)M1的延迟为l，准确率为a，(2)M2延迟为2l，准确率高5%，即a(1+5%)，它们应当有类似的奖励：Reward(M2) = a · (1 + 5%) · (2l/T)^β ≈ Reward(M1) = a · (l/T)^β。解此可以得到β ≈ −0.07。因此，除非另外声明，我们在试验中都使用α = β = −0.07。

Figure 3 shows the objective function with two typical values of (α, β). In the top figure with (α = 0, β = −1), we simply use accuracy as the objective value if measured latency is less than the target latency T; otherwise, we sharply penalize the objective value to discourage models from violating latency constraints. The bottom figure (α = β = −0.07) treats the target latency T as a soft constraint, and smoothly adjusts the objective value based on the measured latency.

图3所示的是两种(α, β)典型值下的目标函数。在上图中(α = 0, β = −1)，即，如果延迟小于T，就使用简单的准确率作为目标函数值；否则，延迟大于T的情况下，就对延迟进行很重的惩罚。下图中(α = β = −0.07)将延迟目标T作为一个软约束，基于延迟对目标函数进行平缓的调整。

Figure 3: Objective Function Defined by Equation 2, assuming accuracy ACC(m)=0.5 and target latency T=80ms: (top) show the object values with latency as a hard constraint; (bottom) shows the objective values with latency as a soft constraint.

## 4. Mobile Neural Architecture Search

In this section, we will first discuss our proposed novel factorized hierarchical search space, and then summarize our reinforcement-learning based search algorithm. 本节中，我们首先讨论提出新型分解层次式搜索空间，然后总结我们的基于强化学习的搜索算法。

### 4.1. Factorized Hierarchical Search Space

As shown in recent studies [36, 20], a well-defined search space is extremely important for neural architecture search. However, most previous approaches [35, 19, 26] only search for a few complex cells and then repeatedly stack the same cells. These approaches don’t permit layer diversity, which we show is critical for achieving both high accuracy and lower latency.

最近的研究[36,20]显示，定义良好的搜索空间对于神经网络搜索是非常重要的。但是，多数之前的方法[25,19,26]只搜索了几个复杂的单元，然后就重复堆叠相同的单元。这种方法欠缺层的多样性，我们会证明，这对网络取得高准确率和低延迟是非常关键的。

In contrast to previous approaches, we introduce a novel factorized hierarchical search space that factorizes a CNN model into unique blocks and then searches for the operations and connections per block separately, thus allowing different layer architectures in different blocks. Our intuition is that we need to search for the best operations based on the input and output shapes to obtain better accurate-latency trade-offs. For example, earlier stages of CNNs usually process larger amounts of data and thus have much higher impact on inference latency than later stages. Formally, consider a widely-used depthwise separable convolution [11] kernel denoted as the four-tuple (K, K, M, N) that transforms an input of size (H, W, M) (We omit batch size dimension for simplicity) to an output of size (H, W, N), where (H, W) is the input resolution and M, N are the input/output filter sizes. The total number of multiply-adds can be described as:

与之前的方法形成对比，我们提出了一种新颖的分解层次式搜索空间，将一个CNN模型分解成唯一的模块，然后逐个模块单独搜索其算子和连接，这样不同模块中就可以有不同的层架构。我们的直觉是，我们需要基于输入和输出的形状，搜索最佳的算子，以得到更好的准确率-延迟折中。比如，CNN早期的层通常处理更多的数据，所以与后期的阶段相比，对推理延迟有更大的影响。正式的，我们考虑一种广泛使用的分层可分离卷积[11]核，表示为四元组(K,K,M,N)，将大小为(H,W,M)的输入转换为大小为(H,W,N)的输出（简化起见，我们忽略了批大小的维度），其中(H,W)是输入分辨率，M,N是输入/输出滤波器大小。MultiAdd计算量可以表示为：

$$H ∗ W ∗ M ∗ (K ∗ K + N)$$(4)

Here we need to carefully balance the kernel size K and filter size N if the total computation is constrained. For instance, increasing the receptive field with larger kernel size K of a layer must be balanced with reducing either the filter size N at the same layer, or compute from other layers.

这里，如果总计算量是固定的话，我们需要仔细的平衡核大小K和滤波器的数量N。比如，用更大的核大小K来增加一层的感受野的大小，要得到均衡，必须降低这一层的滤波器数量N，或降低其他层的计算量。

Figure 4 shows the baseline structure of our search space. We partition a CNN model into a sequence of pre-defined blocks, gradually reducing input resolutions and increasing filter sizes as is common in many CNN models. Each block has a list of identical layers, whose operations and connections are determined by a per-block sub search space. Specifically, a sub search space for a block i consists of the following choices:

图4是我们搜索空间的基准结构。我们将一个CNN模型分解成一系列预定义的模块，逐渐降低输入分辨率，增加滤波器数量，CNN模型通常都是这样的。每个模块中都是相同的层，其运算和连接是由每个模块的子搜索空间确定的。具体的，模块i的子搜索空间包括以下选择：

- Convolutional ops ConvOp: regular conv (conv), depthwise conv (dconv), and mobile inverted bottleneck conv [29]. 卷积算子ConvOp：常规卷积(conv)，分层卷积(dconv)，和移动逆瓶颈卷积[29]。
- Convolutional kernel size KernelSize: 3x3, 5x5. 卷积核大小KernelSize：3×3,5×5。
- Squeeze-and-excitation [13] ratio SERatio: 0, 0.25. SE率：0,0.25。
- Skip ops SkipOp: pooling, identity residual, or no skip. 跳跃连接SkipOp：池化，恒等残差，或没有跳跃连接。
- Output filter size $F_i$. 输出滤波器数量$F_i$。
- Number of layers per block $N_i$. 每一层的模块数$N_i$。

ConvOp, KernelSize, SERatio, SkipOp, $F_i$ determines the architecture of a layer, while $N_i$ determines how many times the layer will be repeated for the block. For example, each layer of block 4 in Figure 4 has an inverted bottleneck 5x5 convolution and an identity residual skip path, and the same layer is repeated $N_4$ times. We discretize all search options using MobileNetV2 as a reference: For #layers in each block, we search for {0, +1, -1} based on MobileNetV2; for filter size per layer, we search for its relative size in {0.75, 1.0, 1.25} to MobileNetV2 [29].

ConvOp, KernelSize, SERatio, SkipOp, $F_i$确定了一层的架构，而$N_i$确定了每个模块中一层会重复多少次。比如，图4中的模块4中的每层，都有一个逆瓶颈5×5卷积和一个恒等残差跳跃路径，同样的层要重复$N_4$次。我们离散化所有的搜索选项，使用MobileNetV2作为一个参考：对于每个模块中的#layers，我们搜索基于MobileNetV2搜索{0, +1, -1}；对于每一层的滤波器数量，我们搜索MobileNetV2中的相对数量的{0.75, 1.0, 1.25}倍数。

Our factorized hierarchical search space has a distinct advantage of balancing the diversity of layers and the size of total search space. Suppose we partition the network into B blocks, and each block has a sub search space of size S with average N layers per block, then our total search space size would be $S^B$, versing the flat per-layer search space with size $S^{B∗N}$. A typical case is S = 432, B = 5, N = 3, where our search space size is about $10^{13}$, versing the perlayer approach with search space size $10^{39}$.

我们的分解层次化搜索空间，在层的多样性和和整个搜索空间的规模上，可以达到更好的均衡。假设我们将网络分成B个模块，每个模块有一个子搜索空间，大小为S，平均每个模块有N层，那么我们的整个搜索空间大小将是$S^B$，比较起来，普通的逐层搜索空间的大小为$S^{B∗N}$。典型的情况是，S = 432, B = 5, N = 3, 其中我们的搜索空间大小大约为$10^{13}$，逐层的方法其搜索空间大小为$10^{39}$。

Figure 4: Factorized Hierarchical Search Space. Network layers are grouped into a number of predefined skeletons, called blocks, based on their input resolutions and filter sizes. Each block contains a variable number of repeated identical layers where only the first layer has stride 2 if input/output resolutions are different but all other layers have stride 1. For each block, we search for the operations and connections for a single layer and the number of layers N , then the same layer is repeated N times (e.g., Layer 4-1 to 4-$N_4$ are the same). Layers from different blocks (e.g., Layer 2-1 and 4-1) can be different.

### 4.2. Search Algorithm 搜索算法

Inspired by recent work [35, 36, 25, 20], we use a reinforcement learning approach to find Pareto optimal solutions for our multi-objective search problem. We choose reinforcement learning because it is convenient and the reward is easy to customize, but we expect other methods like evolution [26] should also work.

受最近的工作[35,36,25,20]启发，我们使用强化学习的方法，来对我们的多目标搜索问题，寻找Pareto最优解。我们选择强化学习，是因为方便，其奖励函数很容易定制，但我们希望其他方法如演化算法[26]也应当起到作用。

Concretely, we follow the same idea as [36] and map each CNN model in the search space to a list of tokens. These tokens are determined by a sequence of actions $a_{1:T}$ from the reinforcement learning agent based on its parameters θ. Our goal is to maximize the expected reward:

具体的，我们采用的是与[36]中一样的算法，将每个CNN模型映射到符号列表的搜索空间。这些符号由强化学习代理的动作序列$a_{1:T}$确定，这些动作序列是基于其参数θ的。我们的目标是最大化期望奖励：

$$J = E_{P(a_{1:T};θ)} [R(m)]$$(5)

where m is a sampled model determined by action $a_{1:T}$, and R(m) is the objective value defined by equation 2. 其中m是由动作序列$a_{1:T}$确定的一个采样模型，R(m)是式(2)定义的目标值。

As shown in Figure 1, the search framework consists of three components: a recurrent neural network (RNN) based controller, a trainer to obtain the model accuracy, and a mobile phone based inference engine for measuring the latency. We follow the well known sample-eval-update loop to train the controller. At each step, the controller first samples a batch of models using its current parameters θ, by predicting a sequence of tokens based on the softmax logits from its RNN. For each sampled model m, we train it on the target task to get its accuracy ACC(m), and run it on real phones to get its inference latency LAT (m). We then calculate the reward value R(m) using equation 2. At the end of each step, the parameters θ of the controller are updated by maximizing the expected reward defined by equation 5 using Proximal Policy Optimization [30]. The sample-eval-update loop is repeated until it reaches the maximum number of steps or the parameters θ converge.

如图1所示，搜索框架包含三个部分：一个基于RNN的控制器，一个训练器来得到模型准确率，一个基于移动手机的推理引擎来衡量延迟。我们按照著名的“采样-评估-更新”的循环，来训练控制器。在每一步中，控制器首先使用现有参数θ来采样一批模型，主要是通过预测一个符号序列，基于RNN的softmax logits。对每个采样的模型m，我们在目标任务中对其训练，以得到其准确率ACC(m)，并在真实手机上运行，以得到其推理延迟LAT(m)。我们然后使用公式2计算其奖励值R(m)。在每一步骤的最后，通过使用Proximal Policy Optimization，最大化式5定义的期望奖励，来更新控制器的参数θ。这个“采样—评估—更新”的训练重复进行，直到达到最大步骤数量，或参数θ收敛。

## 5. Experimental Setup 试验设置

Directly searching for CNN models on large tasks like ImageNet or COCO is expensive, as each model takes days to converge. While previous approaches mainly perform architecture search on smaller tasks such as CIFAR-10 [36, 26], we find those small proxy tasks don’t work when model latency is taken into account, because one typically needs to scale up the model when applying to larger problems. In this paper, we directly perform our architecture search on the ImageNet training set but with fewer training steps (5 epochs). As a common practice, we reserve randomly selected 50K images from the training set as the fixed validation set. To ensure the accuracy improvements are from our search space, we use the same RNN controller as NASNet [36] even though it is not efficient: each architecture search takes 4.5 days on 64 TPUv2 devices. During training, we measure the real-world latency of each sampled model by running it on the single-thread big CPU core of Pixel 1 phones. In total, our controller samples about 8K models during architecture search, but only 15 top-performing models are transferred to the full ImageNet and only 1 model is transferred to COCO.

在大型任务中直接搜索CNN模型是非常耗时的，如ImageNet或COCO，因为每个模型都需要几天才能收敛。之前的方法主要在更小的任务中，如CIFAR-10中搜索架构，而我们则发现，在考虑到模型延迟的情况下，这些小型代理任务不好用，因为在更大的问题中应用时，一般需要放大这个模型。在本文中，我们在ImageNet训练集上直接进行架构搜索，但使用更少的训练步骤（5轮迭代）。与通常做法一样，我们从训练集中随机选择50K图像，作为固定的验证集。为确保准确率的改进是因为我们的搜索空间的原因，我们使用与NASNet[36]相同的控制器RNN，虽然其效率不太高：每次架构搜索在TPUv2上耗时4.5天。在训练时，我们测量每个采样模型的真实世界延迟，在Pixel 1手机上单线程用大CPU核运行模型。总计，我们的控制器在架构搜索中，采样了大约8K个模型，但只有15个表现最好的模型迁移到了完整的ImageNet上，只有一个模型迁移到了COCO上。

For full ImageNet training, we use RMSProp optimizer with decay 0.9 and momentum 0.9. Batch norm is added after every convolution layer with momentum 0.99, and weight decay is 1e-5. Dropout rate 0.2 is applied to the last layer. Following [7], learning rate is increased from 0 to 0.256 in the first 5 epochs, and then decayed by 0.97 every 2.4 epochs. We use batch size 4K and Inception preprocessing with image size 224×224. For COCO training, we plug our learned model into SSD detector [22] and use the same settings as [29], including input size 320 × 320.

对于完整的ImageNet训练，我们使用RMSProp优化器，衰减为0.9，动量为0.9。在买个卷积层后，都加入了批归一化，动量0.99，权重衰减1e-5。对于最后一层，Dropout率为0.2。按照[7]的方法，学习率在前5轮中从0增加到0.256，然后每2.4轮衰减0.97。我们使用批规模为4K，图像大小224×224，Inception预处理。对于COCO训练，我们将学习到的模型插入到SSD检测器中，使用[29]相同的设置，包括输入大小为320×320。

## 6. Results

In this section, we study the performance of our models on ImageNet classification and COCO object detection, and compare them with other state-of-the-art mobile models. 本节中，我们在ImageNet分类和COCO目标检测中研究我们模型的性能，与其他目标最好的移动模型进行比较。

### 6.1. ImageNet Classification Performance

Table 1 shows the performance of our models on ImageNet [28]. We set our target latency as T = 75ms, similar to MobileNetV2 [29], and use Equation 2 with α=β=-0.07 as our reward function during architecture search. Afterwards, we pick three top-performing MnasNet models, with different latency-accuracy trade-offs from the same search experiment and compare them with existing mobile models.

表1给出了我们的模型在ImageNet上的结果。我们设目标延迟为T = 75ms，与MobileNetV2类似，式2中α=β=-0.07，是我们架构搜索过程中的奖励函数。然后，我们选择了表现最好的三个MnasNet模型，在相同的搜索试验中有着不同的延迟-准确率折中，将其与现有的移动模型进行比较。

As shown in the table, our MnasNet A1 model achieves 75.2% top-1 / 92.5% top-5 accuracy with 78ms latency and 3.9M parameters / 312M multiply-adds, achieving a new state-of-the-art accuracy for this typical mobile latency constraint. In particular, MnasNet runs 1.8× faster than MobileNetV2 (1.4) [29] on the same Pixel phone with 0.5% higher accuracy. Compared with automatically searched CNN models, our MnasNet runs 2.3× faster than the mobile-size NASNet-A [36] with 1.2% higher top-1 accuracy. Notably, our slightly larger MnasNet-A3 model achieves better accuracy than ResNet-50 [9], but with 4.8× fewer parameters and 10× fewer multiply-add cost.

如表中所示，我们的MnasNet A1模型得到了75.2% top-1 / 92.5% top-5准确率，延迟为78ms，参数量为3.9M，MultiAdds为312M，对于这种典型的移动延迟的约束来说，取得了目前最好的准确率。特别的，MnasNet比MobileNetV2(1.4)在同样的Pixel手机上快了1.8x，准确率高了0.5%。与自动搜索到的CNN模型相比，我们的MnasNet比移动大小的NASNet-A快了2.3x，top-1准确率高了1.2%。需要说明的是，我们略大的MnasNet-A3模型比ResNet-50模型准确率更高，但参数量少了4.8x，MultiAdds少了10x。

Given that squeeze-and-excitation (SE [13]) is relatively new and many existing mobile models don’t have this extra optimization, we also show the search results without SE in the search space in Table 2; our automated approach still significantly outperforms both MobileNetV2 and NASNet.

因为SE模块相对较新，很多现有的移动模型没有这个额外的优化，我们也给出了搜索空间中没有SE模块的结果，见表2；我们的自动方法仍然比MobileNetV2和NASNet要好很多。

Table 1: Performance Results on ImageNet Classification [28]. We compare our MnasNet models with both manually-designed mobile models and other automated approaches – MnasNet-A1 is our baseline model;MnasNet-A2 and MnasNet-A3 are two models (for comparison) with different latency from the same architecture search experiment; #Params: number of trainable parameters; #Mult-Adds: number of multiply-add operations per image; Top-1/5 Acc.: the top-1 or top-5 accuracy on ImageNet validation set; Inference Latency is measured on the big CPU core of a Pixel 1 Phone with batch size 1.

Table 2: Performance Study for Squeeze-and-Excitation SE [13] – MnasNet-A denote the default MnasNet with SE in search space; MnasNet-B denote MnasNet with no SE in search space.

### 6.2. Model Scaling Performance 模型缩放的性能

Given the myriad application requirements and device heterogeneity present in the real world, developers often scale a model up or down to trade accuracy for latency or model size. One common scaling technique is to modify the filter size using a depth multiplier [11]. For example, a depth multiplier of 0.5 halves the number of channels in each layer, thus reducing the latency and model size. Another common scaling technique is to reduce the input image size without changing the network.

在真实世界中存在多样化的应用需求和设备异质性，开发者通常会把模型放大或缩小，用模型准确率换取延迟或模型大小。一个通常的缩放技术是用深度乘子来改变滤波器数量。比如，深度乘子为0.5时，将每层中的通道数量减半，因此降低了延迟和模型大小。另一种常用的缩放技术是降低输入图像分辨率，而不改变网络。

Figure 5 compares the model scaling performance of MnasNet and MobileNetV2 by varying the depth multipliers and input image sizes. As we change the depth multiplier from 0.35 to 1.4, the inference latency also varies from 20ms to 160ms. As shown in Figure 5a, our MnasNet model consistently achieves better accuracy than MobileNetV2 for each depth multiplier. Similarly, our model is also robust to input size changes and consistently outperforms MobileNetV2 (increaseing accuracy by up to 4.1%) across all input image sizes from 96 to 224, as shown in Figure 5b.

图5比较了MnasNet和MobileNetV2模型缩放的性能，使用了不同的深度乘子和输入图像大小。我们将深度乘子从0.35变化到1.4，推理延迟也从20ms变化到160ms。如图5a所示，我们的MnasNet模型在每个深度乘子下，一直都比MobileNetV2效果要好。类似的，我们的模型对于输入图像大小也非常稳健，在所有的输入图像大小下（从96到224）一直都比MobileNetV2要好（准确率最多超了4.1%），如图5b所示。

Figure 5: Performance Comparison with Different Model Scaling Techniques. MnasNet is our baseline model shown in Table 1. We scale it with the same depth multipliers and input sizes as MobileNetV2. (a) Depth multiplier = 0.35, 0.5, 0.75, 1.0, 1.4, corresponding to points from left to right. (b) Input size = 96, 128, 160, 192, 224, corresponding to points from left to right.

In addition to model scaling, our approach also allows searching for a new architecture for any latency target. For example, some video applications may require latency as low as 25ms. We can either scale down a baseline model, or search for new models specifically targeted to this latency constraint. Table 4 compares these two approaches. For fair comparison, we use the same 224x224 image sizes for all models. Although our MnasNet already outperforms MobileNetV2 with the same scaling parameters, we can further improve the accuracy with a new architecture search targeting a 22ms latency constraint.

除了模型缩放，我们的方法还可以对任意的目标延迟，搜索一个新的架构。比如，一些视频应用可能需要的延迟在25ms以下。我们可以将基准模型缩小，或具体针对这个延迟约束搜索新的模型。表4比较了这两种方法。为公平比较，我们对所有模型使用相同的输入图像分辨率224×224。虽然我们的MnasNet在相同的缩放参数下已经超过了MobileNetV2，我们可以针对22ms的延迟约束，搜索一个新的架构，进一步改进准确率。

Table 4: Model Scaling vs. Model Search – MobileNetV2 (0.35x) and MnasNet-A1 (0.35x) denote scaling the baseline models with depth multiplier 0.35; MnasNet-search1/2 denotes models from a new architecture search that targets 22ms latency constraint.

| | Params | MAdds | Latency | Top-1 Acc.
--- | --- | --- | --- | ---
MobileNetV2 (0.35x) | 1.66M | 59M | 21.4ms | 60.3%
MnasNet-A1 (0.35x) | 1.7M | 63M | 22.8ms | 64.1%
MnasNet-search1 | 1.9M | 65M | 22.0ms | 64.9%
MnasNet-search2 | 2.0M | 68M | 23.2ms | 66.0%

### 6.3. COCO Object Detection Performance

For COCO object detection [18], we pick the MnasNet models in Table 2 and use them as the feature extractor for SSDLite, a modified resource-efficient version of SSD [29]. Similar to [29], we compare our models with other mobilesize SSD or YOLO models.

对于COCO目标检测[18]，我们选择了表2中的MnasNet模型，使用其作为SSDLite的特征提取器，SSDLite是SSD的修正的节省资源的版本。与[29]类似，我们将模型与其他移动规模的SSD或YOLO模型进行比较。

Table 3 shows the performance of our MnasNet models on COCO. Results for YOLO and SSD are from [27], while results for MobileNets are from [29]. We train our models on COCO trainval35k and evaluate them on testdev2017 by submitting the results to COCO server. As shown in the table, our approach significantly improve the accuracy over MobileNet V1 and V2. Compare to the standard SSD300 detector [22], our MnasNet model achieves comparable mAP quality (23.0 vs 23.2) as SSD300 with 7.4× fewer parameters and 42× fewer multiply-adds.

表3给出了我们的MnasNet模型在COCO上的表现。YOLO和SSD的结果是[27]中的，MobileNets的结果是[29]中的。我们在COCO trainval35k中训练我们的模型，并在testdev2017上评估，将结果提交给COCO服务器。如表中所示，我们的方法比MobileNetV1和V2的效果要好很多。与标准的SSD300检测器相比，我们的MnasNet模型与SSD300的mAP类似(23.0 vs 23.2)，但参数少了7.4x，MultiAdds少了42x。

Table 3: Performance Results on COCO Object Detection – #Params: number of trainable parameters; #Mult-Adds: number of multiply-additions per image; mAP : standard mean average precision on test-dev2017; mAP_S, mAP_M, mAP_L: mean average precision on small, medium, large objects; Inference Latency: the inference latency on Pixel 1 Phone.

Network | Params | Mult-Adds | mAP | mAP_S | mAP_M | mAP_L | Inference Latency
--- | --- | --- | --- | --- | --- | --- | ---
YOLOv2 [27] | 50.7M | 17.5B | 21.6 | 5.0 | 22.4 | 35.5 | -
SSD300 [22] | 36.1M | 35.2B | 23.2 | 5.3 | 23.2 | 39.6 | -
SSD512 [22] | 36.1M | 99.5B | 26.8 | 9.0 | 28.9 | 41.9 | -
MobileNetV1 + SSDLite [11] | 5.1M | 1.3B | 22.2 | - | - | - | 270ms
MobileNetV2 + SSDLite [29] | 4.3M | 0.8B | 22.1 | - | - | - | 200ms
MnasNet-A1 + SSDLite | 4.9M | 0.8B | 23.0 | 3.8 | 21.7 | 42.0 | 203ms

## 7. Ablation Study and Discussion

In this section, we study the impact of latency constraint and search space, and discuss MnasNet architecture details and the importance of layer diversity. 本节中，我们研究了延迟约束和搜索空间的影响，讨论了MnasNet的架构细节，和层多样性的重要性。

### 7.1. Soft vs. Hard Latency Constraint

Our multi-objective search method allows us to deal with both hard and soft latency constraints by setting α and β to different values in the reward equation 2. Figure 6 shows the multi-objective search results for typical α and β. When α = 0, β = −1, the latency is treated as a hard constraint, so the controller tends to focus more on faster models to avoid the latency penalty. On the other hand, by setting α = β = −0.07, the controller treats the target latency as a soft constraint and tries to search for models across a wider latency range. It samples more models around the target latency value at 75ms, but also explores models with latency smaller than 40ms or greater than 110ms. This allows us to pick multiple models from the Pareto curve in a single architecture search as shown in Table 1.

我们的多目标搜索方法，通过设置奖励函数式2中α和β为不同的值，可以处理延迟硬约束和软约束的情况。图6给出了典型α和β值的多目标搜索结果。当α=0，β=-1时，延迟是作为硬约束的，所以控制器倾向于关注更快的模型，以避免延迟惩罚。另一方面，设α=β=-0.07，控制器将目标延迟作为软约束，在更多的延迟范围内搜索模型。在目标延迟值75ms附近采样更多的模型，但也会探索延迟小于40ms或大于110ms的模型。这允许我们在一次架构搜索中从Pareto曲线中选择模型，如表1所示。

Figure 6: Multi-Objective Search Results based on equation 2 with (a) α=0, β=-1; and (b) α=β=−0.07. Target latency is T =75ms. Top figure shows the Pareto curve (blue line) for the 3000 sampled models (green dots); bottom figure shows the histogram of model latency.

### 7.2. Disentangling Search Space and Reward

To disentangle the impact of our two key contributions: multi-objective reward and new search space, Figure 5 compares their performance. Starting from NASNet [36], we first employ the same cell-base search space [36] and simply add the latency constraint using our proposed multiple-object reward. Results show it generates a much faster model by trading the accuracy to latency. Then, we apply both our multi-objective reward and our new factorized search space, and achieve both higher accuracy and lower latency, suggesting the effectiveness of our search space.

为将两个关键贡献的影响分开，即多目标奖励函数和新的搜索空间。图5比较了其性能。从NASNet开始，我们首先采用相同的基于单元的搜索空间，并简单的加入了延迟约束，使用提出的多目标奖励。结果表明，可以产生一个快的多的模型，用准确率换取了延迟。然后，我们使用了多目标奖励和我们新的架构搜索空间，得到了更高的准确率和更低的延迟，说明我们搜索空间的有效性。

Table 5: Comparison of Decoupled Search Space and Reward Design – Multi-obj denotes our multi-objective reward; Single-obj denotes only optimizing accuracy.

Reward | Search Space | Latency | Top-1 Acc.
--- | --- | --- | ---
Single-obj [36] | Cell-based[36] | 183ms | 74.0%
Multi-obj | Cell-based[36] | 100ms | 72.0%
Multi-obj | MnasNet | 78ms | 75.2%

### 7.3. MnasNet Architecture and Layer Diversity

Figure 7(a) illustrates our MnasNet-A1 model found by our automated approach. As expected, it consists of a variety of layer architectures throughout the network. One interesting observation is that our MnasNet uses both 3x3 and 5x5 convolutions, which is different from previous mobile models that all only use 3x3 convolutions.

图7(a)给出了自动方法搜索到的MnasNet-A1模型，在网络中包含了多种层的架构。一个有趣的观察是，我们的MnasNet使用了3×3和5×5两种卷积，与之前的移动模型不太一样，只使用3×3卷积。

Figure 7: MnasNet-A1 Architecture – (a) is a representative model selected from Table 1; (b) - (d) are a few corresponding layer structures. MBConv denotes mobile inverted bottleneck conv, DWConv denotes depthwise conv, k3x3/k5x5 denotes kernel size, BN is batch norm, HxWxF denotes tensor shape (height, width, depth), and ×1/2/3/4 denotes the number of repeated layers within the block.

images(224×224×3)) -> conv3×3 -> SepConv(k3×3) /×1 -> MBConv6(k3×3) /×2 -> MBConv3(k5×5), SE /×3 -> MBConv6(k3×3) /×4 -> MBConv6(k3×3), SE /×2 -> MBConv6(k5×5), SE /×3 -> MBConv6(k3×3) /×1 -> Pooling, FC -> logits

In order to study the impact of layer diversity, Table 6 compares MnasNet with its variants that only repeat a single type of layer (fixed kernel size and expansion ratio). Our MnasNet model has much better accuracy-latency trade-offs than those variants, highlighting the importance of layer diversity in resource-constrained CNN models.

为研究层多样性的影响，表6比较了MnasNet的一个变体，只重复同一类型的层（固定的核大小和扩展率）。我们的MnasNet模型比这些变体有着好的多的准确率-延迟折中，强调了层多样性在资源限制的CNN模型中的重要性。

Table 6: Performance Comparison of MnasNet and Its Variants – MnasNet-A1 denotes the model shown in Figure 7(a); others are variants that repeat a single type of layer throughout the network. All models have the same number of layers and same filter size at each layer.

| | Top-1 Acc. | Inference Latency
--- | --- | ---
MnasNet-A1 | 75.2% | 78ms
MBConv3 (k3x3) only | 71.8% | 63ms
MBConv3 (k5x5) only | 72.5% | 78ms
MBConv6 (k3x3) only | 74.9% | 116ms
MBConv6 (k5x5) only | 75.6% | 146ms

## 8. Conclusion

This paper presents an automated neural architecture search approach for designing resource-efficient mobile CNN models using reinforcement learning. Our main ideas are incorporating platform-aware real-world latency information into the search process and utilizing a novel factorized hierarchical search space to search for mobile models with the best trade-offs between accuracy and latency. We demonstrate that our approach can automatically find significantly better mobile models than existing approaches, and achieve new state-of-the-art results on both ImageNet classification and COCO object detection under typical mobile inference latency constraints. The resulting MnasNet architecture also provides interesting findings on the importance of layer diversity, which will guide us in designing and improving future mobile CNN models.

本文提出了一种自动神经架构搜索方法，采用强化学习设计资源利用高效的移动CNN模型。我们的主要思想是将与平台相关的真实世界的延迟信息纳入到搜索过程中来，利用了一种新型分解的层次式搜索空间，来搜索拥有最好准确率-延迟折中的移动模型。我们证明了，我们的方法可以比现有的方法自动找到明显更好的移动模型，在典型的移动推理延迟限制下，在ImageNet分类和COCO目标检测中得到了新的目前更好的结果。在得到的MnasNet架构中，还可以观察到层多样性也是非常重要的，这在设计和改进将来的移动CNN模型中，有指导作用。