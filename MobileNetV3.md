# Searching for MobileNetV3

Andrew Howard et al. Google Research, Google Brain

## Abstract 摘要

We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2% more accurate on ImageNet classification while reducing latency by 15% compared to MobileNetV2. MobileNetV3-Small is 4.6% more accurate while reducing latency by 5% compared to MobileNetV2. MobileNetV3-Large detection is 25% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.

基于新的架构设计和补充搜索技巧的结合，我们提出下一代MobileNets。通过针对硬件的NAS和NetAdapt算法，然后通过新的架构进展的改进，MobileNetV3调节用于移动手机CPUs。本文研究的是，自动搜索算法和网络设计可以怎样一起工作，利用互补的方法推动最新的发展。通过这个过程，我们创建了两个新的MobileNet模型：MobileNetV3-Large和MobileNetV3-Small，其目标分别是计算资源高和低的使用情况。这些模型然后经过修改用于目标检测和语义分割任务。对于语义分割的任务（或任何密集像素预测），我们提出了一种新的分割解码器Lite Reduced Atrous Spatial Pyramid Pooling(LR-ASPP)。我们在移动分类、检测和分割上得到了新的目前最好的结果。MobileNetV3-Large与MobileNetV2相比，在ImageNet分类上准确率高了3.2%，耗时少了15%。MobileNetV3-Small与MobileNetV2相比，准确度高了4.6%，耗时少了5%。MobileNetV3-Large与MobileNetV2在COCO检测上有大致相同的准确率，但是快了25%。MobileNetV3-Large LR-ASPP与MobileNetV2 R-ASPP在Cityscapes上的分割结果比较起来，准确率类似，但速度快了30%。

## 1. Introduction 引言

Efficient neural networks are becoming ubiquitous in mobile applications enabling entirely new on-device experiences. They are also a key enabler of personal privacy allowing a user to gain the benefits of neural networks without needing to send their data to the server to be evaluated. Advances in neural network efficiency not only improve user experience via higher accuracy and lower latency, but also help preserve battery life through reduced power consumption.

高效的神经网络正在移动应用中变得很常见，使得在设备上全新的体验成为可能。这也是保护个人隐私的关键因素之一，使得用户不需要将数据传送到服务器上进行计算，就可以得到神经网络计算的好处。神经网络效率上的进展不仅通过更高的准确率和更低的耗时改进用户体验，也通过降低能量消耗帮助节省电池寿命。

This paper describes the approach we took to develop MobileNetV3 Large and Small models in order to deliver the next generation of high accuracy efficient neural network models to power on-device computer vision. The new networks push the state of the art forward and demonstrate how to blend automated search with novel architecture advances to build effective models.

这篇文章描述的是，我们开发MobileNetV3 Large和Small模型的方法，以发布下一代高准确度高效神经模型，为设备上的计算机视觉赋能。新的网络将最好的前沿往前推进，展示了怎样将自动搜索与新型架构进展融合到一起，以构建高效的模型。

The goal of this paper is to develop the best possible mobile computer vision architectures optimizing the accuracy-latency trade off on mobile devices. To accomplish this we introduce (1) complementary search techniques, (2) new efficient versions of nonlinearities practical for the mobile setting, (3) new efficient network design, (4) a new efficient segmentation decoder. We present thorough experiments demonstrating the efficacy and value of each technique evaluated on a wide range of use cases and mobile phones.

本文的目标是开发出尽可能好的移动计算机视觉架构，在移动设备上优化准确度-耗时的折中。为达此目标，我们介绍(1)补充的搜索技术，(2)新的高效版的非线性函数，对于移动应用的设置非常实用，(3)新的高效的网络设计，(4)新的高效的分割解码器。我们进行了深入的试验，在很多使用情况和移动手机上证明了每种技术的功效和价值。

The paper is organized as follows. We start with a discussion of related work in Section 2. Section 3 reviews the efficient building blocks used for mobile models. Section 4 reviews architecture search and the complementary nature of MnasNet and NetAdapt algorithms. Section 5 describes novel architecture design improving on the efficiency of the models found through the joint search. Section 6 presents extensive experiments for classification, detection and segmentation in order do demonstrate efficacy and understand the contributions of different elements. Section 7 contains conclusions and future work.

本文组织如下。我们在第2部分讨论了相关的工作。第3部分回顾了用于移动模型使用的高效模块。第4部分回顾了架构搜索和MnasNet和NetAdapt算法的互补本质。第5部分描述了新的架构设计，改进了通过联合搜索找到的模型的效率。第6部分给出了广泛的试验，包括分类、检测和分割，证明并理解了模型不同元素的功效和贡献。第7部分为总结和展望。

## 2. Related Work 相关工作

Designing deep neural network architecture for the optimal trade-off between accuracy and efficiency has been an active research area in recent years. Both novel handcrafted structures and algorithmic neural architecture search have played important roles in advancing this field.

设计深度神经网络架构，得到准确率和效率之间最优的平衡，是最近几年的活跃研究领域。在这个领域的进展中，手工设计的新架构，和算法神经架构搜索都起了重要作用。

SqueezeNet[20] extensively uses 1x1 convolutions with squeeze and expand modules primarily focusing on reducing the number of parameters. More recent works shifts the focus from reducing parameters to reducing the number of operations (MAdds) and the actual measured latency. MobileNetV1[17] employs depthwise separable convolution to substantially improve computation efficiency. MobileNetV2[37] expands on this by introducing a resource-efficient block with inverted residuals and linear bottlenecks. ShuffleNet[47] utilizes group convolution and channel shuffle operations to further reduce the MAdds. CondenseNet[19] learns group convolutions at the training stage to keep useful dense connections between layers for feature re-use. ShiftNet[44] proposes the shift operation interleaved with point-wise convolutions to replace expensive spatial convolutions.

SqueezeNet[20]使用了很多1x1卷积，对模块进行压缩和扩张，主要着眼于降低参数数量。最近的工作将焦点从降低参数数量转移到了降低运算的数量(MAdds)和实际测量的延迟（耗时）上。MobileNetV1[17]采用了depthwise separable卷积，极大改进了计算效率。MobileNetV2[37]对此进一步扩展，引入了一个资源高效的逆残差和线性瓶颈模块。ShuffleNet[47]利用分组卷积和通道变换次序操作，来进一步降低MAdds。CondenseNet[19]在训练阶段学习分组卷积，以保持层间有用的密集连接，进行特征复用。ShiftNet[44]提出shift操作与点卷积交错，以替代运算量大的空间卷积。

To automate the architecture design process, reinforcement learning (RL) was first introduced to search efficient architectures with competitive accuracy [51, 52, 3, 25, 33]. A fully configurable search space can grow exponentially large and intractable. So early works of architecture search focus on the cell level structure search, and the same cell is reused in all layers. Recently, [41] explored a block-level hierarchical search space allowing different layer structures at different resolution blocks of a network. To reduce the computational cost of search, differentiable architecture search framework is used in [26, 5, 43] with gradient-based optimization. Focusing on adapting existing networks to constrained mobile platforms, [46, 14, 12] proposed more efficient automated network simplification algorithms.

为使架构设计过程自动化，首先引入了强化学习(RL)进行有效的架构搜索，得到了不错的准确率[51,52,3,25,33]。完全可配置的搜索空间可能指数级增长变大，很难处理。所以早期的架构搜索工作聚焦在单元层次的架构搜索，相同的单元在所有层重复使用。最近，[41]探索了一种块层次的层级搜索空间，使得网络不同分辨率模块的层的结构可以不同。为降低搜索计算代价，[26,5,43]使用了基于梯度的可微分的架构搜索框架优化。[46,14,12]聚焦于在受限的移动平台修改已有的网络，提出更高效率的自动网络简化算法。

Quantization [21, 23, 45, 39, 49, 50, 35] is another important complementary effort to improve the network efficiency through reduced precision arithmetic. Finally, knowledge distillation [4, 15] offers an additional complementary method to generate small accurate ”student” networks with the guidance of a large ”teacher” network.

量化[21,23,45,39,49,50,35]是改进网络效率的另一种重要的补充努力，主要通过降低算术精确度。最后，知识蒸馏[4,15]提供了通过大的“老师”网络的指导生成小的准确的“学生”网络的另一种补充方法。

## 3. Efficient Mobile Building Blocks 高效的移动基础模块

Mobile models have been built on increasingly more efficient building blocks. MobileNetV1 [17] introduced depth-wise separable convolutions as an efficient replacement for traditional convolution layers. Depthwise separable convolutions effectively factorize traditional convolution by separating spatial filtering from the feature generation mechanism. Depthwise separable convolutions are defined by two separate layers: light weight depthwise convolution for spatial filtering and heavier 1x1 pointwise convolutions for feature generation.

移动模型在越来越高效的模块上构建起来。MobileNetV1 [17]引入了depth-wise separable卷积，是传统卷积层的高效替代。Depthwise separable卷积高效的分解了传统卷积，将空间滤波和特征生成机制分离开。Depthwise separable卷积由两个分离的层定义：轻量级的卷积层进行空间滤波，较重的1x1点卷积进行特征生成。

MobileNetV2 [37] introduced the linear bottleneck and inverted residual structure in order to make even more efficient layer structures by leveraging the low rank nature of the problem. This structure is shown on Figure 3 and is defined by a 1x1 expansion convolution followed by depthwise convolutions and a 1x1 projection layer. The input and output are connected with a residual connection if and only if they have the same number of channels. This structure maintains a compact representation at the input and the output while expanding to a higher-dimensional feature space internally to increase the expressiveness of non-linear per-channel transformations.

MobileNetV2 [37]引入了线性瓶颈和逆残差结构，利用了问题的低秩本质，创造出了更高效的层结构。这种结构如图3所示，由1x1扩展卷积后接depth-wise卷积和1x1投影层构成。输入和输出只有在通道数相同的情况下通过残差连接。这种结构在输入输出层维持了一种紧凑的表示，同时扩展到了一种更高维的内在特征空间，以增加非线性逐通道变换的表现力。

Figure 3. MobileNetV2 [37] layer (Inverted Residual and Linear Bottleneck). Each block consists of narrow input and output (bottleneck), which don’t have non-linearity, followed by expansion to a much higher-dimensional space and projection to the output. The residual connects bottleneck (rather than expansion).

MnasNet [41] built upon the MobileNetV2 structure by introducing lightweight attention modules based on squeeze and excitation into the bottleneck structure. Note that the squeeze and excitation module are integrated in a different location than ResNet based modules proposed in [18]. The module is placed after the depthwise filters in the expansion in order for attention to be applied on the largest representation as shown on Figure 4.

MnasNet[41]是在MobileNetV2架构的基础上，向瓶颈结构中引入了基于squeeze-and-excitation的轻量的注意力模块。注意squeeze-and-excitation模块的整合位置不同于[18]中基于ResNet的模块。这个模块置于depthwise滤波器之后的扩展部分中，这样注意力模块可以应用于最大的表示中，如图4所示。

Figure 4. MobileNetV2 + Squeeze-and-Excite [18]. In contrast with [18] we apply the squeeze and excite in the residual layer. We use different non-linearity depending on the layer, see section 5.2 for details.

For MobileNetV3, we use a combination of these layers as building blocks in order to build the most effective models. Layers are also upgraded with modified swish non-linearities [34]. Both squeeze and excitation as well as the swish nonlinearity use the sigmoid which can be inefficient to compute as well challenging to maintain accuracy in fixed point arithmetic so we replace this with the hard sigmoid [2, 11] as discussed in section 5.2.

对于MobileNetV3，我们使用这些层的组合作为基础模块，以构建最高效的模型。这些层用修正的swish非线性[34]进行升级。Squeeze-and-Excitation和swish非线性都使用sigmoid，这在计算上效率较低，也很难在定点计算中保持准确率，所以我们将这个替换为hard sigmoid[2,11]，见5.2节所述。

## 4. Network Search 网络搜索

Network search has shown itself to be a very powerful tool for discovering and optimizing network architectures [51, 41, 5, 46]. For MobileNetV3 we use platform-aware NAS to search for the global network structures by optimizing each network block. We then use the NetAdapt algorithm to search per layer for the number of filters. These techniques are complementary and can be combined to effectively find optimized models for a given hardware platform.

网络搜索已经证明了是一种发现和优化网络架构的有力工具[51,41,5,46]。对于MobileNetV3，我们使用感知平台的NAS来通过对每个网络模块进行优化来搜索全局网络架构。我们然后使用NetAdapt算法来搜索每层的滤波器数量。这些技术是互补的，可以结合起来针对给定的硬件平台有效的找到优化的模型。

### 4.1. Platform-Aware NAS for Block-wise Search

Similar to [41], we employ a platform-aware neural architecture approach to find the global network structures. Since we use the same RNN-based controller and the same factorized hierarchical search space, we find similar results as [41] for Large mobile models with target latency around 80ms. Therefore, we simply reuse the same MnasNet-A1[41] as our initial Large mobile model, and then apply NetAdapt [46] and other optimizations on top of it.

与[41]类似，我们采用感知平台的神经架构搜索方法寻找全局网络结构。由于我们使用相同的基于RNN的控制器，以及相同的分解层次搜索空间，我们找到的Large移动模型与[41]类似，目标耗时大约80ms。所以，我们只是重复使用相同的MnasNet-A1[41]作为我们的初始Large移动模型，然后在其上应用NetAdapt[46]和其他优化方法。

However, we observe the original reward design is not optimized for small mobile models. Specifically, it uses a multi-objective reward $ACC(m) × [LAT(m)/TAR]^w$ to approximate Pareto-optimal solutions, by balancing model accuracy ACC(m) and latency LAT (m) for each model m based on the target latency TAR. We observe that accuracy changes much more dramatically with latency for small models; therefore, we need a smaller weight factor w = −0.15 (vs the original w = −0.07 in [41]) to compensate for the larger accuracy change for different latencies. Enhanced with this new weight factor w, we start a new architecture search from scratch to find the initial seed model and then apply NetAdapt and other optimizations to obtain the final MobilenetV3-Small model.

但是，我们观察到原始的reward设计对于小型移动模型不是最优的。特别的，它使用了多目标reward $ACC(m) × [LAT(m)/TAR]^w$，基于目标耗时TAR，通过在每个模型m的准确率ACC(m)和耗时LAT(m)平衡，来近似Pareto最优解。我们观察到，对于小型模型，准确率随着耗时的变化变化更大；所以，我们需要更小的权重因子w = −0.15（[41]中原始的为w = −0.07），以补偿不同耗时下更大的准确率变化。采用这个新的权重因子w，我们从头开始进行新的架构搜索，来发现初始的种子模型，然后使用NetAdapt和其他优化策略来得到最后的MobileNetV3-Small模型。

### 4.2. NetAdapt for Layer-wise Search

The second technique that we employ in our architecture search is NetAdapt [46]. This approach is complimentary to platform-aware NAS: it allows fine-tuning of individual layers in a sequential manner, rather than trying to infer coarse but global architecture. We refer to the original paper for the full details. In short the technique proceeds as follows:

我们在架构搜索方面采用的第二种技术是NetAdapt[46]。这种方法与感知平台的NAS是互补的：这可以顺序形式的精调单层，而不是试图推理粗糙的全局架构。详见原始论文。简而言之，这种技术按如下方式进行：

1. Starts with a seed network architecture found by platform-aware NAS. 从感知平台的NAS发现的种子架构开始。
2. For each step: 每个步骤进行如下工作：

(a) Generate a set of new proposals. Each proposal represents a modification of an architecture that generates at least δ reduction in latency compared to the previous step.

(a)生成新建议的集合。每个建议代表架构的一种变化，与之前的步骤比，生成的网络耗时至少减少δ。

(b) For each proposal we use the pre-trained model from the previous step and populate the new proposed architecture, truncating and randomly initializing missing weights as appropriate. Fine-tune each proposal for T steps to get a coarse estimate of the accuracy.

(b)对每个建议，我们使用前一步骤预训练的模型，移植到新提出的架构中，截断对应的部分，或对遗失的权重进行随机初始化。对每个建议进行T步精调，得到准确率的粗略估计。

(c) Selected best proposal according to some metric. 根据一些度量选择最好的建议。

3. Iterate previous step until target latency is reached. 前面的步骤进行迭代，直到达到目标耗时。

In [46] the metric was to minimize the accuracy change. We modify this algorithm and minimize the ratio between latency change and accuracy change. That is for all proposals generated during each NetAdapt step, we pick one that maximizes: $\frac{∆Acc}{|∆latency|}$, with ∆latency satisfying the constraint in 2(a). The intuition is that because our proposals are discrete, we prefer proposals that maximize the slope of the trade-off curve.

[46]中的度量是最小化准确率变化。我们对算法进行修改，最小化耗时变化与准确率变化之比率。在每个NetAdapt步骤中生成的所有建议，我们选择一个最大化$\frac{∆Acc}{|∆latency|}$，其中∆latency满足2(a)中的约束。直觉上来说，因为我们的建议是离散的，我们倾向于最大化折中曲线斜率的建议。

This process is repeated until the latency reaches its target, and then we re-train the new architecture from scratch. We use the same proposal generator as was used in [46] for MobilenetV2. Specifically, we allow the following two types of proposals:

这个过程不断重复，直到耗时达到目标，然后我们从头重新训练新的架构。我们使用与[46]相同的建议生成方法，应用在MobileNetV2上。特别的，我们允许下面两类建议：

1. Reduce the size of any expansion layer; 降低任何扩展层的大小的；
2. Reduce bottleneck in all blocks that share the same bottleneck size - to maintain residual connections. 在所有共享相同瓶颈大小的模块中减少瓶颈的，以保持残差连接。

For our experiments we used T = 10000 and find that while it increases the accuracy of the initial fine-tuning of the proposal. However, it does not generally change the final accuracy, when trained from scratch. We set δ = 0.01|L|, where L is the latency of the seed model. 对于我们的试验，我们使用T=10000，发现可以增加初始精调建议的准确率。但是，当从头开始训练时，一般并不改变最终准确率。我们设δ = 0.01|L|，其中L是种子模型的耗时。

## 5. Network Improvements 网络改进

In addition to network search, we also introduce several new components to the model to further improve the final model. We redesign the computionally-expensive layers at the beginning and the end of the network. We also introduce a new nonlinearity, h-swish, a modified version of the recent swish nonlinearity, which is faster to compute and more quantization-friendly.

除了网络搜索，我们还为模型引入了几种新的部件，以进一步改进最终模型。我们重新设计了网络开始和结束时计算量很大的层。我们还引入了一种新的非线性，h-swish，最近提出的swish非线性的修正版，计算更快速，量化更友好。

### 5.1. Redesigning Expensive Layers 重新设计计算量大的层

Once models are found through architecture search, we observe that some of the last layers as well as some of the earlier layers are more expensive than others. We propose some modifications to the architecture to reduce the latency of these slow layers while maintaining the accuracy. These modifications are outside of the scope of the current search space.

一旦模型通过架构搜索找到了模型，我们观察到，最后的一些层和最开始的一些层计算量比其他的大很多。我们提出对架构进行一些修正，以降低这些慢速层的耗时，同时保持准确率。这些修正是在目前的搜索空间之外的。

The first modification reworks how the last few layers of the network interact in order to produce the final features more efficiently. Current models based on MobileNetV2’s inverted bottleneck structure and variants use 1x1 convolution as a final layer in order to expand to a higher-dimensional feature space. This layer is critically important in order to have rich features for prediction. However, this comes at a cost of extra latency.

第一个修正对网络最后几层怎样互动进行了再加工，以更高效的生成最终的特征。目前的模型是基于MobileNetV2的逆瓶颈(?)结构的，其变体使用1x1卷积作为最终层以扩展到更高维的特征图。这个层非常重要，可以得到很丰富的特征进行预测。但是，其代价是额外的计算量即耗时。

To reduce latency and preserve the high dimensional features, we move this layer past the final average pooling. This final set of features is now computed at 1x1 spatial resolution instead of 7x7 spatial resolution. The outcome of this design choice is that the computation of the features becomes nearly free in terms of computation and latency.

为减少耗时，保护高维度特征，我们将这个层移到最后的平均池化层之后。最终的特征集就是在1x1分辨率上进行计算的，而不是在7x7空间分辨率上计算的。这个设计选择的结果是，特征计算几乎没有计算量和耗时。

Once the cost of this feature generation layer has been mitigated, the previous bottleneck projection layer is no longer needed to reduce computation. This observation allows us to remove the projection and filtering layers in the previous bottleneck layer, further reducing computational complexity. The original and optimized last stages can be seen in figure 5. The efficient last stage reduces the latency by 10 milliseconds which is 15% of the running time and reduces the number of operations by 30 millions MAdds with almost no loss of accuracy. Section 6 contains detailed results.

一旦这个特征生成的层的计算量降低了，也就不需要之前的瓶颈投影层来降低计算量了。这个观察结果使我们可以去掉前面瓶颈层的投影和滤波层，进一步降低计算复杂度。原始的最后几层和优化的最后几层如图5所示。优化过的最后几层降低了10ms的耗时，这是15%的运行时间，减少了30 millions MAdds的计算量，而且几乎没有降低准确率。

Figure 5. Comparison of original last stage and efficient last stage. This more efficient last stage is able to drop three expensive layers at the end of the network at no loss of accuracy.

Another expensive layer is the initial set of filters. Current mobile models tend to use 32 filters in a full 3x3 convolution to build initial filter banks for edge detection. Often these filters are mirror images of each other. We experimented with reducing the number of filters and using different nonlinearities to try and reduce redundancy. We settled on using the hard swish nonlinearity for this layer as it performed as well as other nonlinearities tested. We were able to reduce the number of filters to 16 while maintaining the same accuracy as 32 filters using either ReLU or swish. This saves an additional 3 milliseconds and 10 million MAdds.

另一个计算量很大的层是初始滤波器集。目前的移动模型倾向于使用32个完整的3x3滤波器，构建初始滤波器组，进行边缘检测。这些滤波器经常互相是镜像。我们尝试减少滤波器的数量，使用不同的非线性以降低冗余。最后确定在这一层使用hard swish非线性，因为与测试的其他非线性表现一样好。我们将滤波器数量降低到了16，同时保持与32滤波器使用ReLU或swish时同样的准确率。这节省了额外的3ms和10 million MAdds。

### 5.2. Nonlinearities 非线性

In [34] a non-linearity called swish was introduced that when used as a drop-in replacement for ReLU, that significantly improves the accuracy of neural networks. The non-linearity is defined as [34]中引入了swish非线性，可以随时替代ReLU，显著改进了神经网络的准确率。这种非线性定义如下

$$swish(x) = x · σ(x)$$

While this nonlinearity improves accuracy, it comes with non-zero cost in embedded environments as the sigmoid function is much more expensive to compute on mobile devices. We deal with this problem in two ways. 虽然这种非线性改进了准确率，在嵌入式环境中是有一定代价的，因为sigmoid函数在移动设备上计算量相对很大。我们用两种方法处理这个问题。

1. We replace sigmoid function with its piece-wise linear hard analog: ReLU6(x+3)/6 similar to [11, 42]. The minor difference is we use ReLU6 rather than a custom clipping constant. Similarly, the hard version of swish becomes： 我们将sigmoid函数替换为其分段线性的硬模拟：与[11,42]类似的ReLU6(x+3)/6。很小的差别是，我们使用ReLU6而不是定制的截断常数。类似的，硬版swish变成了：

$$h-swish[x] = x·ReLU6(x + 3)/6$$

A similar version of hard-swish was also recently proposed in [2]. The comparison of the soft and hard version of sigmoid and swish non-linearities is shown in figure 6. Our choice of constants was motivated by simplicity and being a good match to the original smooth version. In our experiments, we found hard-version of all these functions to have no discernible difference in accuracy, but multiple advantages from a deployment perspective. First, optimized implementations of ReLU6 are available on virtually all software and hardware frameworks. Second, in quantized mode, it eliminates potential numerical precision loss caused by different implementations of the approximate sigmoid. Finally, even optimized implementations of quantized sigmoid tend to be far slower than their ReLU counterparts. In our experiments, replacing h-swish with swish in quantized mode increased inference latency by 15% (In floating point mode, memory access dominates the latency cost.).

[2]中提出了硬swish的类似版本。Sigmoid、swish非线性的soft版和hard版的比较如图6所示。我们选择的常数主要考虑简单性，同时也很好的匹配原始平滑版本。在我们的试验中，我们发现所有硬版函数的应用都没有准确率上的明显区别，但从部署角度来讲有多个好处。首先，ReLU6的优化实现在所有软件和硬件框架中都是可用的。第二，在量化模式下，sigmoid函数有很多不同的近似实现版本，我们则消除了潜在的数值精度损失。最后，即使是优化实现的量化sigmoid，也比相应的ReLU函数慢很多。在我们的试验中，用量化模式的swish替换h-swish增加耗时15% （在浮点模式下，内存访问占耗时代价的主要部分）。

2. The cost of applying nonlinearity decreases as we go deeper into the network, since each layer activation memory typically halves every time the resolution drops. Incidentally, we find that most of the benefits swish are realized by using them only in the deeper layers. Thus in our architectures we only use h-swish at the second half of the model. We refer to the tables 1 and 2 for the precise layout. Even with these optimization h-swish still introduces some latency cost. However as we demonstrate in section 6 the net effect on accuracy and latency is positive, and it provides a venue for further software optimization: once smooth sigmoid is replaced by piece-wise linear function, most of the overhead is in memory accesses, which could be eliminated by fusing the nonlinearities with the previous layers.

网络深度加深，使用非线性的代价就下降，因为每层的激活内存一般都随着分辨率下降而减半。我们还附带发现，使用swish的多数好处是在更深的层中体现出来的。所以在我们的架构中，我们只在模型的后半部使用h-swish。精确的网络布局如表1、2所示。即使在这些优化下，h-swish仍然带来了一些耗时代价。但是就像我们在第6部分证明的那样，网络在准确率和耗时上的效果是正面的，而且这为将来软件的优化提供了一个点：一旦sigmoid替换成分段线性函数，多数时间消耗都在内存访问上，这可以通过将非线性与之前的层进行融合。

#### 5.2.1 Large squeeze-and-excite

In [41], the size of the squeeze-and-excite bottleneck was relative the size of the convolutional bottleneck. Instead, we replace them all to fixed to be 1/4 of the number of channels in expansion layer. We find that doing so increases the accuracy, at the modest increase of number of parameters, and no discernible latency cost. 在[41]中，squeeze-and-excite瓶颈的大小大约是卷积瓶颈的大小。我们则全部将其固定为扩展层通道数的1/4。我们发现，这样做提高了准确率，参数数量的增加也很适量，没有增加耗时。

### 5.3. MobileNetV3 Definitions

MobileNetV3 is defined as two models: MobileNetV3-Large and MobileNetV3-Small. These models are targeted at high and low resource use cases respectively. The models are created through applying platform-aware NAS and NetAdapt for network search and incorporating the network improvements defined in this section. See table 1 and 2 for full specification of our networks.

MobileNetV3定义为如下两个模型：MobileNetV3-Large和MobileNetV3-Small。这两个模型目标分别是计算资源多与少的使用情况。模型是通过感知平台的NAS和网络搜索的NetAdapt创造出来的，并使用了这一节定义的网络改进方法。详见表1和表2。

Table 1. Specification for MobileNetV3-Large. SE denotes whether there is a Squeeze-And-Excite in that block. NL denotes the type of non-linearity used. Here, HS denotes h-swish and RE denotes ReLU. NBN denotes no batch normalization. s denotes stride.

Input | Operator | exp size | #out | SE | NL | s
--- | --- | --- | --- | --- | --- | ---
224^2 × 3 | conv2d | - | 16 | - | HS | 2
112^2 × 16 | bneck,3x3 | 16 | 16 | - | RE | 1
112^2 × 16 | bneck,3x3 | 64 | 24 | - | RE | 2
56^2 × 24 | bneck,3x3 | 72 | 24 | - | RE | 1
56^2 × 24 | bneck,5x5 | 72 | 40 | y | RE | 2
28^2 × 40 | bneck,5x5 | 120 | 40 | y | RE | 1
28^2 × 40 | bneck,5x5 | 120 | 40 | y | RE | 1
28^2 × 40 | bneck,3x3 | 240 | 80 | - | HS | 2
14^2 × 80 | bneck,3x3 | 200 | 80 | - | HS | 1
14^2 × 80 | bneck,3x3 | 184 | 80 | - | HS | 1
14^2 × 80 | bneck,3x3 | 184 | 80 | - | HS | 1
14^2 × 80 | bneck,3x3 | 480 | 112 | y | HS | 1
14^2 × 112 | bneck,3x3 | 672 | 112 | y | HS | 1
14^2 × 112 | bneck,5x5 | 672 | 160 | y | HS | 1
14^2 × 112 | bneck,5x5 | 672 | 160 | y | HS | 2
7^2 × 160 | bneck,5x5 | 960 | 160 | y | HS | 1
7^2 × 160 | conv2d,1x1 | - | 960 | - | HS | 1
7^2 × 960 | Pool,7x7 | - | - | - | HS | -
1^2 × 960 | conv2d 1x1,NBN | - | 1280 | - | HS | 1
1^2 × 1280 | conv2d 1x1,NBN | - | k | - | | -

Table 2. Specification for MobileNetV3-Small. See table 1 for notation.

## 6. Experiments 试验

We present experimental results to demonstrate the effectiveness of the new MobileNetV3 models. We report results on classification, detection and segmentation. We also report various ablation studies to shed light on the effects of various design decisions. 我们给出试验结果，证明新的MobileNetV3模型的有效性，我们在分类、检测和分割任务上给出结果，我们还给出各种分离试验，解释清楚各种设计决定的效果。

### 6.1. Classification 分类

As has become standard, we use ImageNet[36] for all our classification experiments and compare accuracy versus various measures of resource usage such as latency and multiply adds (MAdds). 我们使用标准的ImageNet[36]进行所有分类试验，对准确率和各种资源使用度量，如耗时和MAdds进行比较。

#### 6.1.1 Training Setup 训练设置

We train our models using synchronous training setup on 4x4 TPU Pod [22] using standard tensorflow RMSPropOp-timizer with 0.9 momentum. We use the initial learning rate of 0.1, with batch size 4096 (128 images per chip), with learning rate decay rate of 0.01 every 3 epochs. We use dropout of 0.8, and l2 weight decay 1e-5. For the same image preprocessing as Inception [40]. Finally we use exponential moving average with decay 0.9999. All our convolutional layers use batch-normalization layers with average decay of 0.99.

我们训练模型使用在4x4 TPU Pod[22]上，使用同步的训练设置，标准的tensorflow RMSPropOp-timizer，动量0.9。我们使用的初始学习速率为0.1，batch size 4096（每个chip 128图像），学习速度每3个epochs衰减为0.01。我们使用dropout 0.8，l2权重衰减1e-5。图像预处理与Inception [40]一样。最后，我们使用指数滑动平均，衰减0.9999。我们所有卷积层都使用批归一化层，平均衰减0.99。

#### 6.1.2 Measurement setup 度量设置

To measure latencies we use standard Google Pixel phones and run all networks through the standard TFLite Benchmark Tool. We use single-threaded large core in all our measurements. We don’t report multi-core inference time, since we find this setup not very practical for mobile applications. 为衡量耗时，我们使用标准的Google Pixel phones，通过标准的TFLite基准测试工具运行所有网络。我们在所有度量中使用单线程大核。我们没有给出多核的推理时间，因为我们发现这个设置对移动应用不太实用。

### 6.2. Results 结果

As can be seen on figure 1 our models outperform the current state of the art such as MnasNet [41], ProxylessNas [5] and MobileNetV2 [37]. We report the floating point performance on different Pixel phones in the table 3. We include quantization results in table 4.

如图1所示，我们的模型超过了目前最好的模型，如MnasNet[41]，ProxylessNas[5]和MobileNetV2[37]。如表3所示，我们给出了不同Pixel手机上的浮点性能。我们在表4中给出了量化后的性能。

Figure 1. The trade-off between Pixel 1 latency and top-1 ImageNet accuracy. All models use the input resolution 224. V3 large and V3 small use multipliers 0.75, 1 and 1.25 to show optimal frontier. All latencies were measured on a single large core of the same device using TFLite[1]. MobileNetV3-Small and Large are our proposed next-generation mobile models.

Table 3. Floating point performance on Pixel family of phones (“P-n“ denotes a Pixel-n phone). All latencies are in ms. The inference latency is measured using a single large core with a batch size of 1.

Network | Top-1 | MAdds | Params | P-1 | P-2 | P-3
--- | --- | --- | --- | --- | --- | ---
V3-Large 1.0 | 75.2 | 219 | 5.4M | 66 | 77 | 52.6
V3-Large 0.75 | 73.3 | 155 | 4M | 50 | 61 | 40.1
Mnasnet-A1 | 75.2 | 315 | 3.9 | 84 | 93 | 64
Proxyless[5] | 74.6 | 320 | 4M | 88 | 98 | 65
V2 1.0 | 72.0 | 300 | 3.4M | 78 | 90 | 61.3
V3-Small 1.0 | 67.4 | 66 | 2.9M | 21.3 | 26 | 17.7
V3-Small 0.75 | 65.4 | 44 | 2.4M | 18 | 21.06 | 15.1
Mnasnet [41] | 64.9 | 65.1 | 1.9M | 24.9 | 28.1 | 18.3
V2 0.35 | 60.8 | 59.2 | 1.6M | 19 | 22 | 14.5

Table 4. Quantized performance. All latencies are in ms. The inference latency is measured using single large core on the respective Pixel 1/2/3 device.

Network | Top-1 | P-1 | P-2 | P-3
--- | --- | --- | --- | ---
V3-Large 1.0 | 73.8 | 57 | 55.9 | 38.9
V2 1.0 | 70.9 | 62 | 53.6 | 38.4
V3-Small | 64.9 | 21 | 20.5 | 13.9
V2 0.35 | 57.2 | 20.6 | 18.6 | 13.1

In figure 7 we show the MobileNetV3 performance trade-offs as a function of multiplier and resolution. Note how MobileNetV3-Small outperforms the MobileNetV3-Large with multiplier scaled to match the performance by nearly 3%. On the other hand, resolution provides an even better trade-offs than multiplier. However, it should be noted that resolution is often determined by the problem (e.g. segmentation and detection problem generally require higher resolution), and thus can’t always be used as a tunable parameter.

图7中，我们给出了MobileNetV3 乘子和分辨率的折中关系函数。注意MobileNetV3-Small在一定的乘子下超过了MobileNetV3-Large的性能达到3%。另一方面，分辨率给出了比乘子更好的折中关系。但是，应当指出，分辨率经常是由问题决定的（如，分割和检测问题一般需要更高的分辨率），所以不能一直作为可调的参数。

Figure 7. Performance of MobilenetV3 as a function of different multipliers and resolutions. In our experiments we have used multipliers 0.35, 0.5, 0.75, 1.0 and 1.25, with a fixed resolution of 224, and resolutions 96, 128, 160, 192, 224 and 256 with a fixed depth multiplier of 1.0. Best viewed in color.

#### 6.2.1 Ablation study 分离研究

**Impact of non-linearities**. In table 5 and figure 8 we show how the decision where to insert h-swish affects the latency. Of particular importance we note that using h-swish on the
entire network results in slight increase of accuracy (0.2), while adding nearly 20% in latency, and again ending up under the efficient frontier.

**非线性的影响**。表5和图8中，我们给出了在不同位置插入h-swish非线性对耗时的影响。我们要指出，在整个网络中使用h-swish非线性会得到略微提高的准确率(0.2)，但会增加接近20%的耗时，所以会在效率前沿线之下。

Figure 8. Impact of h-swish vs swish vs ReLU on latency. The curve shows a frontier of using depth multiplier. Note that placing h-swish at all layer with 112 channels or more moves along the optimal frontier.

Table 5. Effect of non-linearities on MobileNetV3-Large. In h-swish @N , N denotes the number of channels, in the first layer that has h-swish enabled.

| | Top 1 | Latency P-1
--- | --- | ---
V3 | 75.2 | 66
0.85 V3 | 74.3 | 55
ReLU | 74.5(-.7%) | 59(-12%)
h-swish @16 | 75.4(+.2%) | 78(+20%)
h-swish @112 | 75.0(-.3%) | 64(-3%)

On the other hand using h-swish moves the efficient frontier up compared to ReLU despite still being about 12% more expensive. Finally, we note, that as h-swish gets optimized by fusing it into the convolutional operator, we expect the latency gap between h-swish and ReLU to drop significantly if not disappear. However, such improvement can’t be expected between h-swish and swish, since computing sigmoid is inherently more expensive.

另外，使用h-swish，与使用ReLU相比，将效率前沿线上移了，但是运算量增加了约12%。最后，我们要说明，由于h-swish可以通过融合进卷集算子中得到优化，我们希望使用h-swish和使用ReLU的耗时差距会显著下降，甚至不存在差距。但是，不能期望在h-swish和swish之间有这样的改进，因为计算sigmoid非常耗时。

**Impact of other components**. In figure 9 we show how introduction of different components moved along the latency/accuracy curve. 其他单元的影响。在图9中，我们展示了引入不同的单元怎样在耗时/准确率曲线中变化。

Figure 9. Impact of adding individual components to the network architecture.

### 6.3. Detection 检测

We use MobileNetV3 as a drop-in replacement for the backbone feature extractor in SSDLite [37] and compare with other backbone networks on COCO dataset [24]. 我们使用MobileNetV3替代SSDLite [37]的骨干特征提取器，在COCO数据集[24]上与其他骨干网络进行比较。

Following MobileNetV2 [37], we attach the first layer of SSDLite to the last feature extractor layer that has an output stride of 16, and attach the second layer of SSDLite to the last feature extractor layer that has an output stride of 32. Following the detection literature, we refer to these two feature extractor layers as C4 and C5, respectively. For MobileNetV3-Large, C4 is the expansion layer of the 13-th bottleneck block. For MobileNetV3-Small, C4 is the expansion layer of the 9-th bottleneck block. For both networks, C5 is the layer immediately before pooling.

和MobileNetV2 [37]一样，我们将SSDLite的第一层与特征提取器输出步长为16的最后一层连接到一起，将SSDLite的第二层与特征提取器输出步长为32的最后一层连接到一起。与现有的检测工作一样，我们分别称这两个特征提取层为C4和C5。对于MobileNetV3-Large，C4是第13个瓶颈模块的扩展层；对于MobileNetV3-Small，C4是第9个瓶颈模块的扩展层。对于两个网络，C5都是池化层前的那一层。

We additionally reduce the channel counts of all feature layers between C4 and C5 by 2. This is because the last few layers of MobileNetV3 are tuned to output 1000 classes, which may be redundant when transferred to COCO with 90 classes.

我们另外将C4和C5之间的所有特征层的通道数减少一半。这是因为，MobileNetV3的最后几层调节为输出1000类，这在迁移到90类的COCO时，可能是冗余的。

The results on COCO test set are given in Tab. 6. With the channel reduction, MobileNetV3-Large is 25% faster than MobileNetV2 with near identical mAP. MobileNetV3-Small with channel reduction is also 2.4 and 0.5 mAP higher than MobileNetV2 and MnasNet at similar latency. For both MobileNetV3 models the channel reduction trick contributes to approximately 15% latency reduction with no mAP loss, suggesting that Imagenet classification and COCO object detection may prefer different feature extractor shapes.

在COCO测试集上的结果如表6所示。通过减少通道数量，MobileNetV3-Large比MobileNetV2快了25%，mAP基本一样。通道数减少的MobileNetV3-Small也比MobileNetV2高了2.4 mAP，比MnasNet高了0.5 mAP，耗时大约相同。两个MobileNetV3模型的减少通道的技巧贡献了大约15%的耗时减少，而且没有降低mAP，说明ImageNet分类和COCO目标检测可能适用于不同的特征提取器形状。

Table 6. Object detection results of SSDLite with different backbones on COCO test set. † : Channels in the blocks between C4 and C5 are reduced by a factor of 2.

Backbone | mAP | Latency (ms) | Params (M) | MAdd (B)
--- | --- | --- | --- | ---
V1 | 22.2 | 270 | 5.1 | 1.3
V2 | 22.1 | 200 | 4.3 | 0.80
Mnasnet | 23.0 | 215 | 4.88 | 0.84
V3 | 22.0 | 173 | 4.97 | 0.62
V3 † | 22.0 | 150 | 3.22 | 0.51
V2 0.35 | 13.7 | 54.8 | 0.93 | 0.16
V2 0.5 | 16.6 | 80.4 | 1.54 | 0.27
Mnasnet 0.35 | 15.6 | 58.2 | 1.02 | 0.18
Mnasnet 0.5 | 18.5 | 86.2 | 1.68 | 0.29
V3-Small | 16.0 | 67.2 | 2.49 | 0.21
V3-Small † | 16.1 | 56.2 | 1.77 | 0.16

### 6.4. Semantic Segmentation 语义分割

In this subsection, we employ MobileNetV2 [37] and the proposed MobileNetV3 as network backbones for the task of mobile semantic segmentation. Additionally, we compare two segmentation heads. The first one, referred to as R-ASPP, was proposed in [37]. R-ASPP is a reduced design of the Atrous Spatial Pyramid Pooling module [7, 8, 9], which adopts only two branches consisting of a 1 × 1 convolution and a global-average pooling operation [27, 48]. In this work, we propose another light-weight segmentation head, referred to as Lite R-ASPP (or LR-ASPP), as shown in Fig. 10. Lite R-ASPP, improving over R-ASPP, deploys the global-average pooling in a fashion similar to the Squeeze-and-Excitation module [18], in which we employ a large pooling kernel with a large stride (to save some computation) and only one 1 × 1 convolution in the module. We apply atrous convolution [16, 38, 31, 6] to the last block of MobileNetV3 to extract denser features, and further add a skip connection [28] from low-level features to capture more detailed information.

在这个小节中，我们采用MobileNetV2 [37]和提出的MobileNetV3作为骨干网络，进行移动语义分割任务。另外，我们比较了两种分割头。第一个称之为R-ASPP，在[37]中提出的。R-ASPP是Atrous Spatial Pyramid Pooling模块[7,8,9]的蜕化设计，只采用了两个包括1×1卷积的分支和一个全局平均池化层操作[27,48]。在本文中，我们提出另一个轻量的分割头，称之为Lite R-ASPP(LR-ASPP)，如图10所示。LR-ASPP是R-ASPP的改进版，部署全局平均池化层的方式与SE模块[18]类似，其中使用的是一个大步长的大pooling核（为节省一些计算量），在这个模块中只有一个1×1卷积。我们在MobileNetV3的最后一个模块中使用atrous卷积，以提取更密集的特征，进一步从低层特征中增加一个跳跃连接[28]，以捕捉更详细的信息。

Figure 10. Building on MobileNetV3, the proposed segmentation head, Lite R-ASPP, delivers fast semantic segmentation results.

We conduct the experiments on the Cityscapes dataset [10] with metric mIOU [13], and only exploit the ‘fine’ annotations. We employ the same training protocol as [8, 37]. All our models are trained from scratch without pretraining on ImageNet [36], and are evaluated with a single-scale input. Similar to object detection, we observe that we could reduce the channels in the last block of network backbone by a factor of 2 without degrading the performance significantly. We think it is because the backbone is designed for 1000 classes ImageNet image classification [36] while there are only 19 classes on Cityscapes, implying there is some channel redundancy in the backbone.

我们在Cityscapes数据集[10]上用mIOU的度量[13]进行试验，只利用'fine'标注的。我们使用的训练方案与[8,37]一样。我们的所有模型都是从头训练的，没有使用ImageNet[36]预训练模型，都是用的单尺度输入进行评估。与目标检测类似，我们观察到，我们可以将骨干网络最后一个模块的通道数量减半，而不明显降低性能，我们认为，这是因为骨干网络是设计用于1000类的ImageNet图像分类[36]的，而在Cityscapes里只有19类，说明骨干网络中有一些通道冗余。

We report our Cityscapes validation set results in Tab. 7. As shown in the table, we observe that (1) reducing the channels in the last block of network backbone by a factor of 2 significantly improves the speed while maintaining similar performances (row 1 vs. row 2, and row 5 vs. row 6), (2) the proposed segmentation head LR-ASPP is slightly faster than R-ASPP [37] while performance is improved (row 2 vs. row 3, and row 6 vs. row 7), (3) reducing the filters in the segmentation head from 256 to 128 improves the speed at the cost of slightly worse performance (row 3 vs. row 4, and row 7 vs. row 8), (4) when employing the same setting, MobileNetV3 model variants attain similar performance while being slightly faster than MobileNetV2 counterparts (row 1 vs. row 5, row 2 vs. row 6, row 3 vs. row 7, and row 4 vs. row 8), (5) MobileNetV3-Small attains similar performance as MobileNetV2-0.5 while being faster, and (6) MobileNetV3-Small is significantly better than MobileNetV2-0.35 while yielding similar speed.

我们在表7中给出在Cityscapes验证集上的结果。如表中所示，我们观察到(1)把骨干网络最后一个模块中的通道数量减半，可以显著提高速度，同时保持类似的性能（行1行2对比，行5行6对比）；(2)提出的分割头LR-ASPP比R-ASPP[37]略快，同时性能也得到了改进（行2行3对比，行6行7对比）；(3)将分割头中的滤波器数量从256减少到128，改进了速度，代价是性能略微降低（行3行4对比，行7行8对比）；(4)当使用相同的设置时，MobileNetV3模型的变体取得了类似的性能，而比相应的MobileNetV2模型略快（行1行5对比，行2行6对比，行3行7对比，行4行8对比）；(5)MobileNetV3-Small取得了与MobileNetV2-0.5类似的性能，而速度更快；(6)MobileNetV3-Small比MobileNetV2-0.35性能明显更好，而速度类似。

Table 7. Semantic segmentation results on Cityscapes val set. RF2: Reduce the Filters in the last block by a factor of 2. V2 0.5 and V2 0.35 are MobileNetV2 with depth multiplier = 0.5 and 0.35, respectively. SH: Segmentation Head, where × employs the R-ASPP while X employs the proposed LR-ASPP. F: Number of Filters used in the Segmentation Head. CPU (f): CPU time measured on a single large core of Pixel 3 (floating point) w.r.t. a full-resolution input (i.e., 1024 × 2048). CPU (h): CPU time measured w.r.t. a half-resolution input (i.e., 512 × 1024). Row 8, and 11 are our MobileNetV3 segmentation candidates.

N | Backbone | RF2 | SH | F | mIOU | Params | Madds | CPU (f) | CPU (h)
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
1 | V2 | n | n | 256 | 72.84 | 2.11M | 21.29B | 4.20s | 1.07s
2 | V2 | y | n | 256 | 72.56 | 1.15M | 13.68B | 3.23s | 819ms
3 | V2 | y | y | 256 | 72.97 | 1.02M | 12.83B | 3.16s | 808ms
4 | V2 | y | y | 128 | 72.74 | 0.98M | 12.57B | 3.12s | 797ms
5 | V3 | n | n | 256 | 72.64 | 3.60M | 18.43B | 4.17s | 1.07s
6 | V3 | y | n | 256 | 71.91 | 1.76M | 11.24B | 3.01s | 765ms
7 | V3 | y | y | 256 | 72.37 | 1.63M | 10.33B | 2.96s | 750ms
8 | V3 | y | y | 128 | 72.36 | 1.51M | 9.74B | 2.87s | 730ms
9 | V2 0.5 | y | y | 128 | 68.57 | 0.28M | 4.00B | 1.61s | 402ms
10 | V2 0.35 | y | y | 128 | 66.83 | 0.16M | 2.54B | 1.31s | 323ms
11 | V3-Small | y | y | 128 | 68.38 | 0.47M | 2.90B | 1.38s | 349ms

Tab. 8 shows our Cityscapes test set results. Our segmentation models with MobileNetV3 as network backbone significantly outperforms ESPNetv2 [30], CCC2 [32], and ESPNetv1 [30] by 10.5%, 10.6%, 12.3%, respectively while being faster in terms of Madds. The performance drops slightly by 0.6% when not employing the atrous convolution to extract dense feature maps in the last block of MobileNetV3, but the speed is improved to 1.98B (for half-resolution input), which is 1.7, 1.59, and 2.24 times faster than ESPNetv2, CCC2, and ESPNetv1, respectively. Furthermore, our models with MobileNetV3-Small as network backbone still outperforms all of them by at least a healthy margin of 6.2%. Our fastest model variant is 13.6% better than ESPNetv2-small with a slightly faster inference speed.

表8给出了在Cityscapes测试集上的结果。我们使用MobileNetV3作为骨干网络的分割模型明显超过了ESPNetv2[30], CCC2[32]和ESPNetv1[30]达10.5%、10.6%、12.3%之多，而同时速度更快。如果在MobileNetV3的最后模块中不使用atrous卷积来提取密集特征图，性能略微下降0.6%，但速度降低到1.98B（对于半分辨率输入），这比ESPNetv2, CCC2和ESPNetv1分别快了1.7、1.59和2.24倍。而且，我们使用MobileNetV3-Small作为骨干网络的模型仍然超过所有这些模型，至少6.2%。我们最快的模型变体比ESPNetv2-small性能高了13.6%，而且推理速度也略高。

Table 8. Semantic segmentation results on Cityscapes test set. OS: Output Stride, the ratio of input image spatial resolution to backbone output resolution. When OS = 16, atrous convolution is applied in the last block of backbone. When OS = 32, no atrous convolution is used. Madds (f): Multiply-Adds measured w.r.t. a full-resolution input (i.e., 1024 × 2048). Madds (h): Multiply-Adds measured w.r.t. a half-resolution input (i.e., 512 × 1024). CPU (f): CPU time measured on a single large core of Pixel 3 (floating point) w.r.t. a full-resolution input (i.e., 1024 × 2048). CPU (h): CPU time measured w.r.t. a half-resolution input (i.e., 512 × 1024). † : The Madds are estimated from [30] which only provides Madds for input size 224 × 224 in their Fig. 7.

Backbone | OS | mIOU | Madds (f) | Madds (h) | CPU (f) | CPU (h)
--- | --- | --- | --- | --- | --- | ---
V3 | 16 | 72.6 | 9.74B | 2.48B | 2.87s | 730ms
V3 | 32 | 72.0 | 7.74B | 1.98B | 2.39s | 607ms
V3-Small | 16 | 69.4 | 2.90B | 0.74B | 1.38s | 349ms
V3-Small | 32 | 68.3 | 2.06B | 0.53B | 1.16s | 290ms
ESPNetv2[30]† | - | 62.1 | 13.46B | 3.36B | - | -
CCC2[32] | - | 62.0 | - | 3.15B | - | -
ESPNetv1[29]† | - | 60.3 | 17.72B | 4.43B | - | -
ESPNetv2 small[30]† | - | 54.7 | 2.26B | 0.56B | - | -

## 7. Conclusions and future work 结论和未来工作

In this paper we introduced MobileNetV3 Large and Small models demonstrating new state of the art in mobile classification, detection and segmentation. We have described our efforts to harness multiple types of network architecture search as well as advances in network design to deliver the next generation of mobile models. We have also shown how to adapt non-linearities like swish and apply squeeze and excite in a quantization friendly and efficient manner introducing them into the mobile model domain as effective tools. We also introduced a new form of lightweight segmentation decoders called LR-ASPP. While it remains an open question of how best to blend automatic search techniques with human intuition, we are pleased to present these first positive results and will continue to refine methods as future work.

本文中，我们提出了MobileNetV3 Large和Small模型，给出了移动分类、检测和分割的最先进方法。我们描述了使用的多种网络框架搜索技术，以及网络设计的进展，以给出下一代移动模型。我们也展示了如何修改像swish这样的非线性，并将squeeze-and-excite以量化友好且高效的方式，引入到移动模型领域中，作为有效的工具。我们还提出了新形式的轻量分割解码器，称为LR-ASPP。虽然怎样将自动搜索技术和人类直觉结合起来仍然是一个问题，我们很乐意给出的这些最开始的正面结果，并在将来继续改进这些方法。

## A. Performance table for different resolutions and multipliers

We give detailed table containing multiply-adds, accuracy, parameter count and latency in Table 9.

Table 9. Floating point performance for Large and Small V3 models. P-1 corresponds to large single core performance on Pixel 1.

Network | Top-1(%) | MAdds(M) | Params(M) | P-1(ms)
--- | --- | --- | --- | ---
large 224/1.25 | 76.6 | 356 | 7.5 | 104.0
large 224/1.0 | 75.2 | 217 | 5.4 | 66.0
large 224/0.75 | 73.3 | 155 | 4.0 | 50.0
large 224/0.5 | 68.8 | 69 | 2.6 | 28.7
large 224/0.35 | 64.2 | 40 | 2.2 | 20.4
large 256/1.0 | 76.0 | 282 | 5.4 | 87.0
large 192/1.0 | 73.7 | 160 | 5.4 | 49.0
large 160/1.0 | 71.7 | 112 | 5.4 | 36.0
large 128/1.0 | 68.4 | 73 | 5.4 | 24.2
large 96/1.0 | 63.3 | 43 | 5.4 | 15.9
small 224/1.25 | 70.4 | 91 | 3.6 | 32.0
small 224/1.0 | 67.5 | 66 | 2.9 | 21.3
small 224/0.75 | 65.4 | 44 | 2.4 | 18.0
small 224/0.5 | 58.0 | 23 | 1.9 | 10.7
small 224/0.35 | 49.8 | 13 | 1.7 | 8.2
small 256/1.0 | 68.5 | 74 | 2.9 | 27.8
small 192/1.0 | 65.4 | 42 | 2.9 | 16.3
small 160/1.0 | 62.8 | 30 | 2.9 | 12.0
small 128/1.0 | 57.3 | 20 | 2.9 | 8.3
small 96/1.0 | 51.7 | 12 | 2.9 | 5.8