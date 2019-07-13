# ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

Ningning Ma et al. Megvii Inc (Face++) Tsinghua University

## Abstract 摘要

Currently, the neural network architecture design is mostly guided by the indirect metric of computation complexity, i.e., FLOPs. However, the direct metric, e.g., speed, also depends on the other factors such as memory access cost and platform characterics. Thus, this work proposes to evaluate the direct metric on the target platform, beyond only considering FLOPs. Based on a series of controlled experiments, this work derives several practical guidelines for efficient network design. Accordingly, a new architecture is presented, called ShuffleNet V2. Comprehensive ablation experiments verify that our model is the state-of-the-art in terms of speed and accuracy tradeoff.

目前，神经网络架构设计的一个主要原则是计算复杂度的间接标准，即FLOPs。但是，速度这样的直接标准也依赖于其他因素，如内存访问代价和平台特性。所以，本文提出在目标平台上评估其直接标准，超越了只考虑FLOPs的方法。在一系列试验的基础上，本文推导出高效网络设计的使用准则。也相应的提出一种新框架，称为ShuffleNet V2。综合的分离对比试验表明，我们的模型在速度与准确率的折中上是目前最好的。

**Keywords**: CNN architecture design, efficiency, practical

## 1 Introduction

The architecture of deep convolutional neutral networks (CNNs) has evolved for years, becoming more accurate and faster. Since the milestone work of AlexNet [1], the ImageNet classification accuracy has been significantly improved by novel structures, including VGG [2], GoogLeNet [3], ResNet [4,5], DenseNet [6], ResNeXt [7], SE-Net [8], and automatic neutral architecture search [9,10,11], to name a few.

DCNNs的架构已经演进了很多年，变得越来越准确越来越快速。自从AlexNet里程碑式的工作后，通过新框架的提出，ImageNet分类准确率提高了很多，举例来说，包括VGG [2], GoogLeNet [3], ResNet [4,5], DenseNet [6], ResNeXt [7], SE-Net [8]和自动神经架构搜索。

Besides accuracy, computation complexity is another important consideration. Real world tasks often aim at obtaining best accuracy under a limited computational budget, given by target platform (e.g., hardware) and application scenarios (e.g., auto driving requires low latency). This motivates a series of works towards light-weight architecture design and better speed-accuracy tradeoff, including Xception [12], MobileNet [13], MobileNet V2 [14], ShuffleNet [15], and CondenseNet [16], to name a few. Group convolution and depth-wise convolution are crucial in these works.

除了准确率，计算复杂度是另一个重要的考虑。真实世界的任务通常目标是，在一定的计算资源限制下，在给定的目标平台（如硬件）和应用场景（如自动驾驶需要低延迟）下，得到最好的准确率。这促使人们设计轻量架构，得到更好的速度-准确率的折中，举例来说，这包括Xception[12]，MobileNet[13]，MobileNet V2[14]，ShuffleNet [15]和CondenseNet [16]。分组卷积和分层卷积在这些工作中非常关键。

To measure the computation complexity, a widely used metric is the number of float-point operations, or FLOPs. However, FLOPs is an indirect metric. It is an approximation of, but usually not equivalent to the direct metric that we really care about, such as speed or latency. Such discrepancy has been noticed in previous works [17,18,14,19]. For example, MobileNet v2 [14] is much faster than NASNET-A [9] but they have comparable FLOPs. This phenomenon is further exmplified in Figure 1(c)(d), which show that networks with similar FLOPs have different speeds. Therefore, using FLOPs as the only metric for computation complexity is insufficient and could lead to sub-optimal design.

为衡量计算复杂度，一个广泛使用的度量是浮点计算的数量，即FLOPs。但是，FLOPs是一个间接度量。我们真正关心的是像速度或延迟这样的直接度量，FLOPs只是一个近似，通常不是其等价。这种差异之前的工作中就有提到。比如，MobileNetv2比NASNet-A快了很多，但它们的FLOPs数差不多。这种现象在图1(c)(d)中可以进一步看到，其中说明了，类似FLOPs的网络有不同的运行速度。因此，使用FLOPs作为计算复杂度的唯一度量是不够的，可能不会得到最有设计。

The discrepancy between the indirect (FLOPs) and direct (speed) metrics can be attributed to two main reasons. First, several important factors that have considerable affection on speed are not taken into account by FLOPs. One such factor is memory access cost (MAC). Such cost constitutes a large portion of runtime in certain operations like group convolution. It could be bottleneck on devices with strong computing power, e.g., GPUs. This cost should not be simply ignored during network architecture design. Another one is degree of parallelism. A model with high degree of parallelism could be much faster than another one with low degree of parallelism, under the same FLOPs.

间接度量(FLOPs)和直接度量(Speed)之间的差异可以归结为两个原因。第一，影响速度的几个重要因素FLOPs没有考虑到。一个是内存访问耗时(MAC)。在一些运算中，这个耗时是运行时间的一大部分，如分组卷积。在算力很高的设备上，这可能就是瓶颈，如GPUs。在网络架构设计时，这种代价不应当被简单忽略掉。另一个是并行程度。在相同FLOPs的情况下，一个可以高度并行计算的模型可以比并行程度很低的模型快很多。

Second, operations with the same FLOPs could have different running time, depending on the platform. For example, tensor decomposition is widely used in early works [20,21,22] to accelerate the matrix multiplication. However, the recent work [19] finds that the decomposition in [22] is even slower on GPU although it reduces FLOPs by 75%. We investigated this issue and found that this is because the latest CUDNN [23] library is specially optimized for 3 × 3 conv. We cannot certainly think that 3 × 3 conv is 9 times slower than 1 × 1 conv.

第二，相同FLOPs的运算，在不同的平台上，可以有不同的运行时间。比如，张量分解在早期的工作中有广泛的使用，以加速矩阵乘法运算。但是，最近的工作发现，[22]中的分解降低了75%的FLOPs，但是在GPU上运行甚至更慢。我们研究了这个问题，发现这是因为最新的CUDNN库为3×3卷积进行了专门的优化。我们不能想当然的认为3×3卷积比1×1卷积慢了9倍。

With these observations, we propose that two principles should be considered for effective network architecture design. First, the direct metric (e.g., speed) should be used instead of the indirect ones (e.g., FLOPs). Second, such metric should be evaluated on the target platform.

在这些观察结果上，我们提出在高效网络架构设计上应当考虑两个因素。第一，应当使用直接度量标准（如速度），而不是间接标准（如FLOPs）。第二，度量标准应当在目标平台上进行评估计算。

In this work, we follow the two principles and propose a more effective network architecture. In Section 2, we firstly analyze the runtime performance of two representative state-of-the-art networks [15,14]. Then, we derive four guidelines for efficient network design, which are beyond only considering FLOPs. While these guidelines are platform independent, we perform a series of controlled experiments to validate them on two different platforms (GPU and ARM) with dedicated code optimization, ensuring that our conclusions are state-of-the-art.

本文中，我们按照这两条原则，提出了一种更高效的网络架构。在第2节中，我们首先分析了两种有代表性的目前最好网络的运行时间性能。然后，我们推导出了高效网络设计的指南，考虑的因素不止是FLOPs。这些指南是与平台无关的，我们进行了一系列试验，在两个不同的平台(GPU和ARM)上进行了验证，代码进行了细致的优化，确保我们的结论是目前最好的。

In Section 3, according to the guidelines, we design a new network structure. As it is inspired by ShuffleNet [15], it is called ShuffleNet V2. It is demonstrated much faster and more accurate than the previous networks on both platforms, via comprehensive validation experiments in Section 4. Figure 1(a)(b) gives an overview of comparison. For example, given the computation complexity budget of 40M FLOPs, ShuffleNet v2 is 3.5% and 3.7% more accurate than ShuffleNet v1 and MobileNet v2, respectively.

在第3节中，根据指南，我们设计了一个新的网络架构。因为受到ShuffleNet启发，我们称之为ShuffleNet V2。在两个平台上，这个模型都比之前的网络更快更准确，在第4部分给出了综合的验证试验。图1(a)(b)给出了总体对比。比如，给定计算复杂度为40MFLOPs，ShuffleNetv2比ShuffleNetv1、MobileNetv2的准确率分别高了3.5%和3.7。

Fig. 1: Measurement of accuracy (ImageNet classification on validation set), speed and FLOPs of four network architectures on two hardware platforms with four different level of computation complexities (see text for details). (a, c) GPU results, batchsize = 8. (b, d) ARM results, batchsize = 1. The best performing algorithm, our proposed ShuffleNet v2, is on the top right region, under all cases.

## 2 Practical Guidelines for Efficient Network Design

Our study is performed on two widely adopted hardwares with industry-level optimization of CNN library. We note that our CNN library is more efficient than most open source libraries. Thus, we ensure that our observations and conclusions are solid and of significance for practice in industry.

我们的研究在两个广泛采用的硬件平台上进行，都有工业级的CNN优化库。我们要指出，我们的CNN库要比多数开源库要更高效。因此，我们确保我们的观察结果和结论是坚实的，在工业实践中非常有意义。

- GPU. A single NVIDIA GeForce GTX 1080Ti is used. The convolution library is CUDNN 7.0 [23]. We also activate the benchmarking function of CUDNN to select the fastest algorithms for different convolutions respectively. 使用了单块NVidia GeForce GTX 1080Ti。卷积库是CUDNN 7.0。我们还激活了CUDNN的基准测试函数，以选择最快的算法分别计算不同的卷积。

- ARM. A Qualcomm Snapdragon 810. We use a highly-optimized Neon-based implementation. A single thread is used for evaluation. 使用了高通Snapdragon 810。我们使用了一个高度优化的基于Neon的实现。评估时使用了单线程。

Other settings include: full optimization options (e.g. tensor fusion, which is used to reduce the overhead of small operations) are switched on. The input image size is 224 × 224. Each network is randomly initialized and evaluated for 100 times. The average runtime is used.

其他设置包括：开启了全优化选项（如，张量融合，用于降低小型运算的开销）。输入图像大小为224×224。每个网络都是随机初始化的，并评估了100次。使用了平均时间。

To initiate our study, we analyze the runtime performance of two state-of-the-art networks, ShuffleNet v1 [15] and MobileNet v2 [14]. They are both highly efficient and accurate on ImageNet classification task. They are both widely used on low end devices such as mobiles. Although we only analyze these two networks, we note that they are representative for the current trend. At their core are group convolution and depth-wise convolution, which are also crucial components for other state-of-the-art networks, such as ResNeXt [7], Xception [12], MobileNet [13], and CondenseNet [16].

为给研究做好准备，我们分析了两种目前最好网络的运行时间性能，ShuffleNet v1和MobileNet v2。它们在ImageNet分类任务上都非常高效准确，在低端设备如移动设备上都广泛使用。虽然我们只分析了这两种网络，我们注意到它们是目前趋势的代表。其内核是分组卷积和分层卷积，也是目前最好的网络的关键部件，如ResNeXt, Xception, MobileNet和CondenseNet。

The overall runtime is decomposed for different operations, as shown in Figure 2. We note that the FLOPs metric only account for the convolution part. Although this part consumes most time, the other operations including data I/O, data shuffle and element-wise operations (AddTensor, ReLU, etc) also occupy considerable amount of time. Therefore, FLOPs is not an accurate enough estimation of actual runtime.

总体运行时间分解成不同的运算，如图2所示。我们要说明，FLOPs标准只对卷积部分有效。虽然这一部分占据了大多数时间，但其他运算也占据了相当部分的时间，包括数据I/O，数据混洗和逐元素运算(AddTensor, ReLU等)。因此，FLOPs估计实际运行时间是不够的。

Fig. 2: Run time decomposition on two representative state-of-the-art network architectures, ShuffeNet v1 [15] (1×, g = 3) and MobileNet v2 [14] (1×).

Based on this observation, we perform a detailed analysis of runtime (or speed) from several different aspects and derive several practical guidelines for efficient network architecture design.

基于以上观察结果，我们从几个不同方面详细分析了运行时间（或速度），推导出了几种实际的高效网络架构设计指南。

**G1) Equal channel width minimizes memory access cost (MAC).**

The modern networks usually adopt depthwise separable convolutions [12,13,15,14], where the pointwise convolution (i.e., 1 × 1 convolution) accounts for most of the complexity [15]. We study the kernel shape of the 1 × 1 convolution. The shape is specified by two parameters: the number of input channels $c_1$ and output channels $c_2$. Let h and w be the spatial size of the feature map, the FLOPs of the 1 × 1 convolution is $B = hwc_1 c_2$.

现代网络通常采用分层可分离卷积，其中点卷积（即1×1卷积）占了大部分计算复杂度。我们研究1×1卷积的核心形状。这个形状由两个参数决定：输入通道数量$c_1$，和输出通道数量$c_2$。令h和w是特征图的空间大小，1×1卷积的FLOPs为$B = hwc_1 c_2$。

For simplicity, we assume the cache in the computing device is large enough to store the entire feature maps and parameters. Thus, the memory access cost (MAC), or the number of memory access operations, is MAC = $hw(c_1+c_2)+c_1 c_2$. Note that the two terms correspond to the memory access for input/output feature maps and kernel weights, respectively.

简化起见，我们假设计算设备上的缓存足以存储整个特征图和参数。因此，内存访问代价(MAC)，或内存访问操作的数量，是MAC = $hw(c_1+c_2)+c_1 c_2$。注意，这两项分别对应着对输入、输出特征图和核心权重的访问。

From mean value inequality, we have 由均值不等式，我们有

$$MAC ≥ 2 \sqrt{hwB} + \frac{B}{hw}$$(1)

Therefore, MAC has a lower bound given by FLOPs. It reaches the lower bound when the numbers of input and output channels are equal. 所以，给定FLOPs后，MAC有一个下限。当输入通道和输出通道数量相等时，达到下限。

The conclusion is theoretical. In practice, the cache on many devices is not large enough. Also, modern computation libraries usually adopt complex blocking strategies to make full use of the cache mechanism [24]. Therefore, the real MAC may deviate from the theoretical one. To validate the above conclusion, an experiment is performed as follows. A benchmark network is built by stacking 10 building blocks repeatedly. Each block contains two convolution layers. The first contains $c_1$ input channels and $c_2$ output channels, and the second otherwise.

这个结论是理论上的。实际中，很多设备上的缓存是不够大的。同时，现代计算库通常采用复杂的blocking策略已充分利用缓存机制。因此，实际MAC与理论值可能有偏差。为验证上述结论，下面进行了一个试验。构建了一个基准测试网络，重复堆叠了一个模块10次，每个模块包含2个卷积层，第一个包含$c_1$个输入通道，$c_2$个输出通道，第二个反之。

Table 1 reports the running speed by varying the ratio $c_1 : c_2$ while fixing the total FLOPs. It is clear that when $c_1 : c_2$ is approaching 1 : 1, the MAC becomes smaller and the network evaluation speed is faster.

表1给出了不同的$c_1 : c_2$比率下的运行速度，但总体FLOPs固定。很清楚可以看到，当$c_1 : c_2$接近1:1时，MAC变得较小，网络计算速度更快一些。

Table 1: Validation experiment for Guideline 1. Four different ratios of number of input/output channels (c1 and c2) are tested, while the total FLOPs under the four ratios is fixed by varying the number of channels. Input image size is 56 × 56.

c1:c2 | (c1,c2) for ×1 | GPU ×1 | ×2 | ×4 | (c1,c2) for ×1 | ARM ×1 | ×2 | ×4
--- | --- | --- | --- | --- | --- | --- | --- | ---
1:1 | (128,128) | 1480 | 723 | 232 | (32,32) | 76.2 | 21.7 | 5.3
1:2 | (90,180) | 1296 | 586 | 206 | (22,44) | 72.9 | 20.5 | 5.1
1:6 | (52,312) | 876 | 489 | 189 | (13,78) | 69.1 | 17.9 | 4.6
1:12 | (36,432) | 748 | 392 | 163 | (9,108) | 57.6 | 15.1 | 4.4

**G2) Excessive group convolution increases MAC.**

Group convolution is at the core of modern network architectures [7,15,25,26,27,28]. It reduces the computational complexity (FLOPs) by changing the dense convolution between all channels to be sparse (only within groups of channels). On one hand, it allows usage of more channels given a fixed FLOPs and increases the network capacity (thus better accuracy). On the other hand, however, the increased number of channels results in more MAC.

分组卷积处在现代网络架构的核心。分组卷积可以降低计算复杂度(FLOPs)，将所有通道间的密集卷积变成稀疏的（只在分组内的通道间进行卷积）。另一方面，在给定FLOPs下可以使用更多的通道数，增加了网络的能力（即可以得到更好的准确率）。但是另一方面，通道数量的增加会需要更多的MAC。

Formally, following the notations in G1 and Eq. 1, the relation between MAC and FLOPs for 1 × 1 group convolution is 利用G1和式1中的符号，1×1卷积中MAC和FLOPs的关系为：

$$MAC = hw(c_1 + c_2) + \frac {c_1 c_2} {g} = hwc_1 + \frac {Bg}{c_1} + \frac {B} {hw}$$(2)

where g is the number of groups and $B = hwc_1 c_2 /g$ is the FLOPs. It is easy to see that, given the fixed input shape $c_1 × h × w$ and the computational cost B, MAC increases with the growth of g.

其中g是分组数量，$B = hwc_1 c_2 /g$是FLOPs。很容易看出，给定固定的输入形状$c_1 × h × w$和计算代价B，MAC随着g的增加而增加。

To study the affection in practice, a benchmark network is built by stacking 10 pointwise group convolution layers. Table 2 reports the running speed of using different group numbers while fixing the total FLOPs. It is clear that using a large group number decreases running speed significantly. For example, using 8 groups is more than two times slower than using 1 group (standard dense convolution) on GPU and up to 30% slower on ARM. This is mostly due to increased MAC. We note that our implementation has been specially optimized and is much faster than trivially computing convolutions group by group.

为研究实际的影响，我们构建了一个基准测试网络，将10个点分组卷积层堆叠到一起。表2给出了不同分组数量下的运行速度，同时FLOPs数量固定。很清楚可以看到，使用更多的分组数量会明显降低运行速度。比如，使用8个分组比使用1个分组（标准密集卷积）在GPU上会慢2倍，在ARM上慢了30%。主要是因为MAC的增加。我们要说明的是，我们的实现已经进行了专门的优化，比普通的分组卷积计算要快很多。

Table 2: Validation experiment for Guideline 2. Four values of group number g are tested, while the total FLOPs under the four values is fixed by varying the total channel number c. Input image size is 56 × 56.

g | c for ×1 | GPU ×1 | ×2 | ×4 | c for ×1 | CPU ×1 | ×2 | ×4
--- | --- | --- | --- | --- | --- | --- | --- | ---
1 | 128 | 2451 | 1289 | 437 | 64 | 40.0 | 10.2 | 2.3
2 | 180 | 1725 | 873 | 341 | 90 | 35.0 | 9.5 | 2.2
4 | 256 | 1026 | 644 | 338 | 128 | 32.9 | 8.7 | 2.1
8 | 360 | 634 | 445 | 230 | 180 | 27.8 | 7.5 | 1.8

Therefore, we suggest that the group number should be carefully chosen based on the target platform and task. It is unwise to use a large group number simply because this may enable using more channels, because the benefit of accuracy increase can easily be outweighed by the rapidly increasing computational cost.

所以，我们建议分组数量应当基于目标平台和任务进行仔细选择。因为可以使用更多的通道数，而使用大量分组，这是不明智的，因为准确率的增加，会导致计算代价的迅速增加。

**G3) Network fragmentation reduces degree of parallelism.**

In the GoogLeNet series [29,30,3,31] and auto-generated architectures [9,11,10]), a “multi-path” structure is widely adopted in each network block. A lot of small operators (called “fragmented operators” here) are used instead of a few large ones. For example, in NASNET-A [9] the number of fragmented operators (i.e. the number of individual convolution or pooling operations in one building block) is 13. In contrast, in regular structures like ResNet [4], this number is 2 or 3.

在GoogLeNet系列和自动生成的架构系列中，在每个网络模块中都广泛使用了多路径结构。很多小型算子（这里称为“碎片化算子”）得到了使用，而没有使用几个大型算子。比如，在NASNet-A中，碎片化算子为13（即，在一个模块中的卷积或池化运算的数量）。比较起来，像ResNet这样的常规结构中，这个数量为2或3。

Though such fragmented structure has been shown beneficial for accuracy, it could decrease efficiency because it is unfriendly for devices with strong parallel computing powers like GPU. It also introduces extra overheads such as kernel launching and synchronization.

虽然这样的碎片化结构对准确率是有好处的，但可能降低效率，因为对于有很强并行计算能力的设备如GPU来说，这是很不友好的。这还会带来其他开销，如核心启动和同步。

To quantify how network fragmentation affects efficiency, we evaluate a series of network blocks with different degrees of fragmentation. Specifically, each building block consists of from 1 to 4 1 × 1 convolutions, which are arranged in sequence or in parallel. The block structures are illustrated in appendix. Each block is repeatedly stacked for 10 times. Results in Table 3 show that fragmentation reduces the speed significantly on GPU, e.g. 4-fragment structure is 3× slower than 1-fragment. On ARM, the speed reduction is relatively small.

为衡量网络碎片化对效率的影响，我们评估一系列不同碎片化程度的网络模块。具体来说，每个模块由1到4个1×1卷积组成，以并行或串行排列。模块结构见附录。每个模块重复堆叠10次。表3中的结果说明，碎片化使得在GPU上的运行速度明显下降，如4碎片的结构比1碎片的结构慢3×。在ARM上，速度的下降相对更慢一些。

Table 3: Validation experiment for Guideline 3. c denotes the number of channels for 1-fragment. The channel number in other fragmented structures is adjusted so that the FLOPs is the same as 1-fragment. Input image size is 56 × 56.

| | GPU c=128 | c=256 | c=512 | CPU c=64 | c=128 | c=256
--- | --- | --- | --- | --- | --- | ---
1-fragment | 2446 | 1274 | 434 | 40.2 | 10.1 | 2.3
2-fragment-series | 1790 | 909 | 336 | 38.6 | 10.1 | 2.2
4-fragment-series | 752 | 745 | 349 | 38.4 | 10.1 | 2.3
2-fragment-parallel | 1537 | 803 | 320 | 33.4 | 9.1 | 2.2
4-fragment-parallel | 691 | 572 | 292 | 35.0 | 8.4 | 2.1

**G4) Element-wise operations are non-negligible.**

As shown in Figure 2, in light-weight models like [15,14], element-wise operations occupy considerable amount of time, especially on GPU. Here, the element-wise operators include ReLU, AddTensor, AddBias, etc. They have small FLOPs but relatively heavy MAC. Specially, we also consider depthwise convolution [12,13,14,15] as an element-wise operator as it also has a high MAC/FLOPs ratio.

如图2所示，在轻量模型如[15,14]中，逐元素的运算占据了相当的计算时间，尤其是在GPU上。这里，逐元素算子，包括ReLU, AddTensor, AddBias等。它们的FLOPs很小，但MAC很严重。尤其是，我们还将分层卷积认为是逐元素的算子，因为其MAC/FLOPs比也很高。

For validation, we experimented with the “bottleneck” unit (1 × 1 conv followed by 3 × 3 conv followed by 1 × 1 conv, with ReLU and shortcut connection) in ResNet [4]. The ReLU and shortcut operations are removed, separately. Runtime of different variants is reported in Table 4. We observe around 20% speedup is obtained on both GPU and ARM, after ReLU and shortcut are removed.

为验证结论，我们使用ResNet的瓶颈结构进行试验（1×1卷积、3×3卷积、1×1卷积的组合，包含ReLU和捷径连接）。ReLU和捷径连接分开去掉。表4给出了不同变体的运行时间。我们观察到，在ReLU和捷径连接去掉后，在GPU和ARM上有大约20%的加速效果。

Table 4: Validation experiment for Guideline 4. The ReLU and shortcut operations are removed from the “bottleneck” unit [4], separately. c is the number of channels in unit. The unit is stacked repeatedly for 10 times to benchmark the speed.

ReLU | short-cut | GPU c=32 | c=64 | c=128 | CPU c=32 | c=64 | c=128
--- | --- | --- | --- | --- | --- | --- | ---
yes | yes | 2427 | 2066 | 1436 | 56.7 | 16.9 | 5.0
yes | no | 2647 | 2256 | 1735 | 61.9 | 18.8 | 5.2
no | yes | 2672 | 2121 | 1458 | 57.3 | 18.2 | 5.1
no | no | 2842 | 2376 | 1782 | 66.3 | 20.2 | 5.4

**Conclusion and Discussions**

Based on the above guidelines and empirical studies, we conclude that an efficient network architecture should 1) use ”balanced“ convolutions (equal channel width); 2) be aware of the cost of using group convolution; 3) reduce the degree of fragmentation; and 4) reduce element-wise operations. These desirable properties depend on platform characterics (such as memory manipulation and code optimization) that are beyond theoretical FLOPs. They should be taken into accout for practical network design.

基于上述准则和经验研究，我们得出结论，高效的网络架构应当：1)使用均衡的卷积（相同的通道宽度）；2)小心使用分组卷积的代价；3)降低碎片化的程度；4)减少逐元素的运算。这些理想的性质，还依赖于平台特性（如内存操作和代码优化），这都是在理论FLOPs之外的。这在实际的网络设计中，都应当考虑进去。

Recent advances in light-weight neural network architectures [15,13,14,9,11,10,12] are mostly based on the metric of FLOPs and do not consider these properties above. For example, ShuffleNet v1 [15] heavily depends group convolutions (against G2) and bottleneck-like building blocks (against G1). MobileNet v2 [14] uses an inverted bottleneck structure that violates G1. It uses depthwise convolutions and ReLUs on “thick” feature maps. This violates G4. The auto-generated structures [9,11,10] are highly fragmented and violate G3.

最近的轻量网络架构设计的进展中，主要都是基于FLOPs的标准，而没有考虑上述的性质。比如，ShuffleNet v1过度依赖于分组卷积（有违G2）和类瓶颈模块（有违G1）。MobileNet v2使用了逆瓶颈结构，有违G1，在很厚的特征图上使用了分层卷积和ReLU，这有违G4。自动生成的结构是高度碎片化的，有违G3。

## 3 ShuffleNet V2: an Efficient Architecture

**Review of ShuffleNet v1 [15].** ShuffleNet is a state-of-the-art network architecture. It is widely adopted in low end devices such as mobiles. It inspires our work. Thus, it is reviewed and analyzed at first.

ShuffleNet是目前最好的网络架构，在低端设备如移动设备中广泛采用，启发了我们的工作，所以，首先对其进行回顾和分析。

According to [15], the main challenge for light-weight networks is that only a limited number of feature channels is affordable under a given computation budget (FLOPs). To increase the number of channels without significantly increasing FLOPs, two techniques are adopted in [15]: pointwise group convolutions and bottleneck-like structures. A “channel shuffle” operation is then introduced to enable information communication between different groups of channels and improve accuracy. The building blocks are illustrated in Figure 3(a)(b).

根据[15]，轻量级网络的主要挑战是，在给定的计算资源下，只能承担一定数量的特征通道。为在保持FLOPs的情况下，增加通道数量，[15]中采用了两个技术：点分组卷积和类瓶颈结构。提出了通道混洗运算，使不同分组间的通道信息可以沟通，以提高准确率。基本模块如图3(a)(b)所示。

Fig. 3: Building blocks of ShuffleNet v1 [15] and this work. (a): the basic ShuffleNet unit; (b) the ShuffleNet unit for spatial down sampling (2×); (c) our basic unit; (d) our unit for spatial down sampling (2×). DWConv: depthwise convolution. GConv: group convolution.

As discussed in Section 2, both pointwise group convolutions and bottleneck structures increase MAC (G1 and G2). This cost is non-negligible, especially for light-weight models. Also, using too many groups violates G3. The element-wise “Add” operation in the shortcut connection is also undesirable (G4). Therefore, in order to achieve high model capacity and efficiency, the key issue is how to maintain a large number and equally wide channels with neither dense convolution nor too many groups.

如第2小节所述，点分组卷积和瓶颈结构会增加MAC (G1 and G2)。这部分是不可忽视的，尤其是对于轻量级模型。同时，使用太多分组会违反G3。捷径连接中的逐元素相加运算也是不理想的(G4)。因此，为使模型容量高、效率高，关键问题是如何保持大量等宽的通道，同时分组数量也不要过多。

**Channel Split and ShuffleNet V2**. Towards above purpose, we introduce a simple operator called channel split. It is illustrated in Figure 3(c). At the beginning of each unit, the input of c feature channels are split into two branches with c − c' and c' channels, respectively. Following G3, one branch remains as identity. The other branch consists of three convolutions with the same input and output channels to satisfy G1. The two 1 × 1 convolutions are no longer group-wise, unlike [15]. This is partially to follow G2, and partially because the split operation already produces two groups.

**通道分离和ShuffleNet V2**。有了上面的目标，我们提出了一种简单的算子，叫通道分离。如图3(c)所示。在每个单元的开始，输入的c个特征通道划分成两个分支，分别有c − c'和c'个通道。按照G3，一个分支仍然是恒等映射。另一个分支包括3个卷积，输入输出通道数量相同，以满足G1。两个1×1卷积不再是分组的，这与[15]不一样。这部分是根据G2，部分是因为分离操作已经产生了两个分组。

After convolution, the two branches are concatenated. So, the number of channels keeps the same (G1). The same “channel shuffle” operation as in [15] is then used to enable information communication between the two branches.

在卷积之后，两个分支拼接起来。所以，通道数量保持不变(G1)。然后使用[15]中的通道混洗操作，以进行两个分支间的信息沟通。

After the shuffling, the next unit begins. Note that the “Add” operation in ShuffleNet v1 [15] no longer exists. Element-wise operations like ReLU and depthwise convolutions exist only in one branch. Also, the three successive elementwise operations, “Concat”, “Channel Shuffle” and “Channel Split”, are merged into a single element-wise operation. These changes are beneficial according to G4.

在混洗后，开始下一个单元。注意，ShuffleNetv1中的加法运算不在存在。逐元素运算如ReLU和分层卷积只在一个分支存在。同时，三个连续的逐元素运算，拼接，通道混洗和通道分离，可以融合成一个逐元素运算。根据G4，这些变化是有好处的。

For spatial down sampling, the unit is slightly modified and illustrated in Figure 3(d). The channel split operator is removed. Thus, the number of output channels is doubled. 对于空间下采样，这个单元进行了略微修改，如图3(d)所示。去掉了通道分离算子。所以，输出通道数加倍了。

The proposed building blocks (c)(d), as well as the resulting networks, are called ShuffleNet V2. Based the above analysis, we conclude that this architecture design is highly efficient as it follows all the guidelines.

提出的模块(c)(d)和得到的网络，称之为ShuffleNetv2。基于上述分析，我们可以得出结论，这种架构设计是非常高效的，因为遵循了所有设计指导原则。

The building blocks are repeatedly stacked to construct the whole network. For simplicity, we set c' = c/2. The overall network structure is similar to ShuffleNet v1 [15] and summarized in Table 5. There is only one difference: an additional 1 × 1 convolution layer is added right before global averaged pooling to mix up features, which is absent in ShuffleNet v1. Similar to [15], the number of channels in each block is scaled to generate networks of different complexities, marked as 0.5×, 1×, etc.

这些单元重复堆叠在一起，以构成整个网络。为简化起见，我们设c'=c/2。整体网络结构与ShuffleNetv1类似，总结在表5中。只有一点区别：在全局平均池化层之前，增加了一个额外的1×1卷积层，以将特征混合在一起，这在ShuffleNetv1中也存在。与[15]类似的是，每个模块中的通道数量可以缩放，以生成不同复杂度的网络，称为0.5×, 1×等。

Table 5: Overall architecture of ShuffleNet v2, for four different levels of complexities.

Layer | Output size | KSize | Stride | Repeat | Output channels 0.5× | 1× | 1.5× | 2×
--- | --- | --- | --- | --- | --- | --- | --- | ---
Image | 224×224 | | | | 3 | 3 | 3 | 3 
Conv1 | 112×112 | 3×3 | 2 | 1 | 24 | 24 | 24 | 24
MaxPool | 56×56 | 3×3 | 2 | 1 | 24 | 24 | 24 | 24
Stage2 | 28×28 | | 2 | 1 | 48 | 116 | 176 | 244
Stage2 | 28×28 | | 1 | 3 | 48 | 116 | 176 | 244
Stage3 | 14×14 | | 2 | 1 | 96 | 232 | 352 | 488
Stage3 | 14×14 | | 1 | 7 | 96 | 232 | 352 | 488
Stage4 | 7×7 | | 2 | 1 | 192 | 464 | 704 | 976
Stage4 | 7×7 | | 1 | 3 | 192 | 464 | 704 | 976
Conv5 | 7×7 | 1×1 | 1 | 1 | 1024 | 1024 | 1024 | 2048
GlobalPool | 1×1 | 7×7 | 
FC | | | | | 1000 | 1000 | 1000 | 1000
FLOPs | | | | | 41M | 146M | 299M | 591M
Num of Weights | | | | | 1.4M | 2.3M | 3.5M | 7.4M

**Analysis of Network Accuracy**. ShuffleNet v2 is not only efficient, but also accurate. There are two main reasons. First, the high efficiency in each building block enables using more feature channels and larger network capacity.

**网络准确率分析**。ShuffleNetv2不仅高效，而且准确率高。主要有两个原因。第一，每个模块的效率都很高，所以可以使用更多特征通道，有更大的网络容量。

Second, in each block, half of feature channels (when c’ = c/2) directly go through the block and join the next block. This can be regarded as a kind of feature reuse, in a similar spirit as in DenseNet [6] and CondenseNet [16].

第二，在每个模块中，一半的特征通道（当c'=c/2时）直接通道模块，加入下一个模块。这可以认为是一种特征复用，与DenseNet和CondenseNet的思想一样。

In DenseNet[6], to analyze the feature reuse pattern, the l1-norm of the weights between layers are plotted, as in Figure 4(a). It is clear that the connections between the adjacent layers are stronger than the others. This implies that the dense connection between all layers could introduce redundancy. The recent CondenseNet [16] also supports the viewpoint.

在DenseNet中，为分析特征复用模式，图4(a)画出了不同层之间的权重的l1-范数。很清楚可以看出，相邻层之间的联系比其他层之间的联系要强。这说明，所有层之间的密集连接会带来冗余。最近的CondenseNet也支持这个观点。

In ShuffleNet V2, it is easy to prove that the number of “directly-connected” channels between i-th and (i+j)-th building block is $r^j c$, where r = (1−c')/c. In other words, the amount of feature reuse decays exponentially with the distance between two blocks. Between distant blocks, the feature reuse becomes much weaker. Figure 4(b) plots the similar visualization as in (a), for r = 0.5. Note that the pattern in (b) is similar to (a).

在ShuffleNetv2中，很容易证明，第i个和第i+j个模块之间，直接相连的通道数量为$r^j c$，其中r = (1−c')/c。换句话说，特征复用随着模块之间的距离，以指数级递减。在很远的模块之间，特征复用变得很弱。图4(b)给出了与(a)类似的可视化图示，其中r=0.5。注意(b)中的模式与(a)类似。

Thus, the structure of ShuffleNet V2 realizes this type of feature re-use pattern by design. It shares the similar benefit of feature re-use for high accuracy as in DenseNet [6], but it is much more efficient as analyzed earlier. This is verified in experiments, Table 8.

所以，ShuffleNetv2的结构在设计上实现了这种类型的特征复用。和DenseNet中一样，特征复用带来了高准确率，但同时还非常高效，后面会进行分析。这在表8中的试验得到了验证。

Fig. 4: Illustration of the patterns in feature reuse for DenseNet [6] and ShuffleNet V2. (a) (courtesy of [6]) the average absolute filter weight of convolutional layers in a model. The color of pixel (s, l) encodes the average l1-norm of weights connecting layer s to l. (b) The color of pixel (s, l) means the number of channels directly connecting block s to block l in ShuffleNet v2. All pixel values are normalized to [0, 1].

## 4 Experiment 试验

Our ablation experiments are performed on ImageNet 2012 classification dataset [32,33]. Following the common practice [15,13,14], all networks in comparison have four levels of computational complexity, i.e. about 40, 140, 300 and 500+ MFLOPs. Such complexity is typical for mobile scenarios. Other hyper-parameters and protocols are exactly the same as ShuffleNet v1 [15].

我们的分离试验在ImageNet 2012分类数据集上进行。与通常实践[15,13,14]一样，所有网络的比较有四种不同的计算复杂度，即，约40,140,300和500+ MFLOPs。这种复杂度在移动场景下是很典型的。其他超参数和方案与ShuffleNetv1中一样。

We compare with following network architectures [12,14,6,15]: 我们比较了下面的网络架构：

- ShuffleNet v1 [15]. In [15], a series of group numbers g is compared. It is suggested that the g = 3 has better trade-off between accuracy and speed. This also agrees with our observation. In this work we mainly use g = 3. 在[15]中，比较了一系列分组数量，得到结论g=3有更好的准确率与速度的折中。这也与我们的观察复合。

- MobileNet v2 [14]. It is better than MobileNet v1 [13]. For comprehensive comparison, we report accuracy in both original paper [14] and our reimplemention, as some results in [14] are not available. MobileNetv2比MobileNetv1更好一些。为进行综合比较，我们给出原始论文和我们复现的准确率，因为一些结果不可用。

- Xception [12]. The original Xception model [12] is very large (FLOPs > 2G), which is out of our range of comparison. The recent work [34] proposes a modified light weight Xception structure that shows better trade-offs between accuracy and efficiency. So, we compare with this variant. 原始Xception模型非常大(FLOPs>2G)，超出了我们比较的范围。最近的工作提出了一种改进的轻量级Xception结构，有着更好的准确率和效率折中。所以，我们与这个变体进行比较。

- DenseNet [6]. The original work [6] only reports results of large models (FLOPs >2G). For direct comparison, we reimplement it following the architecture settings in Table 5, where the building blocks in Stage 2-4 consist of DenseNet blocks. We adjust the number of channels to meet different target complexities. 原始DenseNet工作[6]只给出了大型模型的结果(FLOPs>2G)。对进行直接比较，我们根据表5中的架构设置进行了复现，其中阶段2-4的模块是DenseNet模块。我们调整了通道数量，以符合不同的目标复杂度。

Table 8 summarizes all the results. We analyze these results from different aspects. 表8总结了所有结果。我们从不同角度分析了这些结果。

**Accuracy vs. FLOPs**. It is clear that the proposed ShuffleNet v2 models outperform all other networks by a large margin 2 , especially under smaller computational budgets. Also, we note that MobileNet v2 performs pooly at 40 MFLOPs level with 224 × 224 image size. This is probably caused by too few channels. In contrast, our model do not suffer from this drawback as our efficient design allows using more channels. Also, while both of our model and DenseNet [6] reuse features, our model is much more efficient, as discussed in Sec. 3.

**准确率vs FLOPs**。很清楚可以看到，提出的ShuffleNetv2模型超出了其他所有网络很多，尤其是在更小的计算复杂度下。同时，我们要说明，MobileNetv2在40MFLOPs下224×224的输入中表现很差。这可能是因为通道数量过少导致的。作为比较，我们的模型则没有这个缺点，因为我们的高效设计可以使用更多的模型。同时，我们的模型和DenseNet都复用了特征，但我们的模型效率更高，这在第3节中也进行了讨论。

As reported in [14], MobileNet v2 of 500+ MFLOPs has comparable accuracy with the counterpart ShuffleNet v2 (25.3% vs. 25.1% top-1 error); however, our reimplemented version is not as good (26.7% error, see Table 8).

[14]中给出，MobileNetv2在500+ MFLOPs下，与相应的ShuffleNetv2有类似的准确率(25.3% vs. 25.1% top-1 error)；但是，我们重现的版本则效果没有那么好（26.7%错误率，见表8）。

Table 8 also compares our model with other state-of-the-art networks including CondenseNet [16], IGCV2 [27], and IGCV3 [28] where appropriate. Our model performs better consistently at various complexity levels.

表8还将我们的模型，与其他目前最好的网络进行了比较，包括CondenseNet，IGCV2，IGCV3。我们的模型在各种不同复杂度下一直表现更好。

**Inference Speed vs. FLOPs/Accuracy**. For four architectures with good accuracy, ShuffleNet v2, MobileNet v2, ShuffleNet v1 and Xception, we compare their actual speed vs. FLOPs, as shown in Figure 1(c)(d). More results on different resolutions are provided in Appendix Table 1.

**推理速度与FLOPs/准确率的对比**。对于四种准确率很好的架构，ShuffleNet v2, MobileNet v2, ShuffleNet v1和Xception，我们比较其实际运行速度与FLOPs的对比，如图1(c)(d)所示。在不同分辨率下的结果如附录表1所示。

ShuffleNet v2 is clearly faster than the other three networks, especially on GPU. For example, at 500MFLOPs ShuffleNet v2 is 58% faster than MobileNet v2, 63% faster than ShuffleNet v1 and 25% faster than Xception. On ARM, the speeds of ShuffleNet v1, Xception and ShuffleNet v2 are comparable; however, MobileNet v2 is much slower, especially on smaller FLOPs. We believe this is because MobileNet v2 has higher MAC (see G1 and G4 in Sec. 2), which is significant on mobile devices.

ShuffleNetv2明显比其他方法要快，尤其是在GPU上。比如，在500MFLOPs下，ShuffleNetv2比MobileNetv2快了58%，比ShuffleNetv1快了63%，比Xception快了25%。在ARM上，ShuffleNetv1，Xception和ShuffleNetv2的速度是类似的；但是，MobileNetv2则要慢的多，尤其是在更小的FLOPs下。我们相信这是因为MobileNetv2的MAC更高（见第2部分的G1和G4），这在移动设备上更加明显。

Compared with MobileNet v1 [13], IGCV2 [27], and IGCV3 [28], we have two observations. First, although the accuracy of MobileNet v1 is not as good, its speed on GPU is faster than all the counterparts, including ShuffleNet v2. We believe this is because its structure satisfies most of proposed guidelines (e.g. for G3, the fragments of MobileNet v1 are even fewer than ShuffleNet v2). Second, IGCV2 and IGCV3 are slow. This is due to usage of too many convolution groups (4 or 8 in [27,28]). Both observations are consistent with our proposed guidelines.

与MobileNetv1，IGCV2和IGCV3相比，我们有两个观察结果。第一，虽然MobileNetv1的准确率不怎么高，但在GPU上其速度比所有模型都要快，包括ShuffleNetv2。我们相信这是因为，其结构复合我们提出的多数原则（如，对于G3来说，MobileNetv1的碎片甚至比ShuffleNetv2要少）。第二，IGCV2和IGCV3是很慢的。这是因为使用了过多的分组卷积（4或8）。这两个观察结果都和我们提出的准则是相符的。

Recently, automatic model search [9,10,11,35,36,37] has become a promising trend for CNN architecture design. The bottom section in Table 8 evaluates some auto-generated models. We find that their speeds are relatively slow. We believe this is mainly due to the usage of too many fragments (see G3). Nevertheless, this research direction is still promising. Better models may be obtained, for example, if model search algorithms are combined with our proposed guidelines, and the direct metric (speed) is evaluated on the target platform.

最近，NAS变成了一个CNN架构设计的趋势。表8的底部评估了一些自动生成的模型。我们发现其速度相对较低。我们相信这主要是因为使用了过多的碎片（见G3）。即使如此，这个研究方向仍然是非常有希望的。比如，如果NAS与我们提出的原则结合起来，在目标平台上使用直接度量标准（速度）进行衡量，则可能得到更好的模型。

Finally, Figure 1(a)(b) summarizes the results of accuracy vs. speed, the direct metric. We conclude that ShuffeNet v2 is best on both GPU and ARM. 最后，图1(a)(b)总结了准确率vs速度的结果。我们得出结论，ShuffleNet v2在GPU和ARM上都是最好的。

**Compatibility with other methods**. ShuffeNet v2 can be combined with other techniques to further advance the performance. When equipped with Squeeze-and-excitation (SE) module [8], the classification accuracy of ShuffleNet v2 is improved by 0.5% at the cost of certain loss in speed. The block structure is illustrated in Appendix Figure 2(b). Results are shown in Table 8 (bottom section).

**与其他方法的兼容性**。ShuffleNetv2可以与其他技术一起使用，以进一步提高性能。当配备了SE模块时，ShuffleNetv2的分类准确率改进了0.5%，但速度有一定降低。模块结构如附录图2(b)所示。结果在表8的底部所示。

**Generalization to Large Models**. Although our main ablation is performed for light weight scenarios, ShuffleNet v2 can be used for large models (e.g, FLOPs ≥ 2G). Table 6 compares a 50-layer ShuffleNet v2 (details in Appendix) with the counterpart of ShuffleNet v1 [15] and ResNet-50 [4]. ShuffleNet v2 still outperforms ShuffleNet v1 at 2.3GFLOPs and surpasses ResNet-50 with 40% fewer FLOPs.

**推广到到大型模型**。虽然我们主要的分离试验都是在轻量级场景进行的，ShuffleNetv2也可以用于大型模型（如，FLOPs≥ 2G）。表6比较了50层的ShuffleNetv2（详见附录）与50层的ShuffleNetv1和ResNet-50。ShuffleNetv2在2.3GFLOPs下仍然超过了ShuffleNetv1，也在少了40% FLOPs的情况下超过了ResNet-50。

For very deep ShuffleNet v2 (e.g. over 100 layers), for the training to converge faster, we slightly modify the basic ShuffleNet v2 unit by adding a residual path (details in Appendix). Table 6 presents a ShuffleNet v2 model of 164 layers equipped with SE [8] components (details in Appendix). It obtains superior accuracy over the previous state-of-the-art models [8] with much fewer FLOPs.

对于更深的ShuffleNetv2（如，超过100层），为使训练更快的收敛，我们略微修改了基本ShuffleNetv2单元，增加了一个残差路径（详见附录）。表6给出了一个164层的ShuffleNetv2模型，使用了SE模块。比之前最好的模型准确率更好，但FLOPs则要少的多。

Table 6: Results of large models. See text for details.

Model | FLOPs | Top-1 err. (%)
--- | --- | ---
ShuffleNet v2-50 (ours) | 2.3G | 22.8
ShuffleNet v1-50 [15] (our impl.) | 2.3G | 25.2
ResNet-50 [4] | 3.8G | 24.0
SE-ShuffleNet v2-164 (ours, with residual) | 12.7G | 18.56
SENet [8] | 20.7G | 18.68

**Object Detection**. To evaluate the generalization ability, we also tested COCO object detection [38] task. We use the state-of-the-art light-weight detector – Light-Head RCNN [34] – as our framework and follow the same training and test protocols. Only backbone networks are replaced with ours. Models are pretrained on ImageNet and then finetuned on detection task. For training we use train+val set in COCO except for 5000 images from minival set, and use the minival set to test. The accuracy metric is COCO standard mmAP, i.e. the averaged mAPs at the box IoU thresholds from 0.5 to 0.95.

**目标检测**。为评估泛化能力，我们还测试了COCO目标检测任务。我们使用目前最好的轻量级检测器Light-Head RCNN，作为我们的框架，使用相同的训练和测试方案，只有骨干网络替换成我们的。模型在ImageNet上预训练，然后在目标检测任务上精调。对于训练，我们使用COCO trainval集，使用minival进行测试。准确率度量标准是COCO标准的mAP，即，框IoU阈值从0.5到0.95下的平均mAP。

ShuffleNet v2 is compared with other three light-weight models: Xception [12,34], ShuffleNet v1 [15] and MobileNet v2 [14] on four levels of complexities. Results in Table 7 show that ShuffleNet v2 performs the best.

ShuffleNetv2与另外三种轻量级模型进行了比较：Xception，ShuffleNet v1和MobileNet v2，复杂度分为四种。表7中的结果表明ShuffleNetv2结果最好。

Compared the detection result (Table 7) with classification result (Table 8), it is interesting that, on classification the accuracy rank is ShuffleNet v2 ≥ MobileNet v2 > ShuffeNet v1 > Xception, while on detection the rank becomes ShuffleNet v2 > Xception ≥ ShuffleNet v1 ≥ MobileNet v2. This reveals that Xception is good on detection task. This is probably due to the larger receptive field of Xception building blocks than the other counterparts (7 vs. 3). Inspired by this, we also enlarge the receptive field of ShuffleNet v2 by introducing an additional 3 × 3 depthwise convolution before the first pointwise convolution in each building block. This variant is denoted as ShuffleNet v2*. With only a few additional FLOPs, it further improves accuracy.

将表7中的检测结果，与表8中的分类结果比较起来，很有趣的是，在分类准确率上的排序是ShuffleNet v2 ≥ MobileNet v2 > ShuffeNet v1 > Xception，而在检测上的排序变成了ShuffleNet v2 > Xception ≥ ShuffleNet v1 ≥ MobileNet v2。这说明，Xception在检测任务上是很好的。这可能是因为，Xception的基础模块的感受野比其他模型要大（7 vs 3）。受此启发，我们也将ShuffleNetv2的感受野变大，在每个模块中，在第一个点卷积之前，增加一个额外的3×3分层卷积。这种变体表示为ShuffleNetv2*。增加的FLOPs不多，进一步提高了准确率。

We also benchmark the runtime time on GPU. For fair comparison the batch size is set to 4 to ensure full GPU utilization. Due to the overheads of data copying (the resolution is as high as 800 × 1200) and other detection-specific operations (like PSRoI Pooling [34]), the speed gap between different models is smaller than that of classification. Still, ShuffleNet v2 outperforms others, e.g. around 40% faster than ShuffleNet v1 and 16% faster than MobileNet v2.

我们还在GPU上测试了运行时间。为公平比较，batch size设为4，以确保充分利用GPU。因为数据复制的开销（分辨率很高，800×1200），和其他检测特有的运算（如PSRoI pooling[34]），不同模型之间的速度差距比分类更小一些。但ShuffleNetv2仍然超过了其他模型，如比ShuffleNetv1快了40%，比MobileNetv2快了16%。

Furthermore, the variant ShuffleNet v2* has best accuracy and is still faster than other methods. This motivates a practical question: how to increase the size of receptive field? This is critical for object detection in high-resolution images [39]. We will study the topic in the future.

而且，ShuffleNetv2*变体得到了最高的准确率，还是比其他方法要快。这激发了一个实际问题：怎样增加感受野的大小？这对于高分辨率图像的目标检测非常关键。我们在将来会研究这个课题。

Table 7: Performance on COCO object detection. The input image size is 800 × 1200. FLOPs row lists the complexity levels at 224 × 224 input size. For GPU speed evaluation, the batch size is 4. We do not test ARM because the PSRoI Pooling operation needed in [34] is unavailable on ARM currently.

Methods/FLOPs | mmAP@40M | 140M | 300M | 500M | GPU Speed@40M | 140M | 300M | 500M
--- | --- | --- | --- | --- | --- | --- | --- | ---
Xception | 21.9 | 29.0 | 31.3 | 32.9 | 178 | 131 | 101 | 83
ShuffleNet v1 | 20.9 | 27.0 | 29.9 | 32.9 | 152 | 85 | 76 | 60
MobileNet v2 | 20.7 | 24.4 | 30.0 | 30.6 | 146 | 111 | 94 | 72
ShuffleNet v2 (ours) | 22.5 | 29.0 | 31.8 | 33.3 | 188 | 146 | 109 | 87
ShuffleNet v2* (ours) | 23.7 | 29.6 | 32.2 | 34.2 | 183 | 138 | 105 | 83

## 5 Conclusion 结论

We propose that network architecture design should consider the direct metric such as speed, instead of the indirect metric like FLOPs. We present practical guidelines and a novel architecture, ShuffleNet v2. Comprehensive experiments verify the effectiveness of our new model. We hope this work could inspire future work of network architecture design that is platform aware and more practical.

我们提出了网络架构设计应当遵循的直接度量标准，如速度，而不是间接标准如FLOPs。我们提出了实际的准则，和新的架构，ShuffleNetv2。广泛的试验验证了我们新模型的有效性。我们希望这个工作会启发未来网络架构设计的思路，可以更针对平台，更加实用。

Table 8: Comparison of several network architectures over classification error (on validation set, single center crop) and speed, on two platforms and four levels of computation complexity. Results are grouped by complexity levels for better comparison. The batch size is 8 for GPU and 1 for ARM. The image size is 224 × 224 except: [*] 160 × 160 and [**] 192 × 192. We do not provide speed measurements for CondenseNets [16] due to lack of efficient implementation currently.

## Appendix

Appendix Fig. 1: Building blocks used in experiments for guideline 3. (a) 1-fragment. (b) 2-fragment-series. (c) 4-fragment-series. (d) 2-fragment-parallel. (e) 4-fragment-parallel.

Appendix Fig. 2: Building blocks of ShuffleNet v2 with SE/residual. (a) ShuffleNet v2 with residual. (b) ShuffleNet v2 with SE. (c) ShuffleNet v2 with SE and residual.