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

In contrast to previous approaches, we introduce a novel factorized hierarchical search space that factorizes a CNN model into unique blocks and then searches for the operations and connections per block separately, thus allowing different layer architectures in different blocks. Our intuition is that we need to search for the best operations based on the input and output shapes to obtain better accurate-latency trade-offs. For example, earlier stages of CNNs usually process larger amounts of data and thus have much higher impact on inference latency than later stages. Formally, consider a widely-used depthwise separable convolution [11] kernel denoted as the four-tuple (K, K, M, N) that transforms an input of size (H, W, M) (We omit batch size dimension for simplicity) to an output of size (H, W, N), where (H, W) is the input resolution and M, N are the input/output filter sizes. The total number of multiply-adds can be described as: