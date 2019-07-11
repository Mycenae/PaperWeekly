# ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

Xiangyu Zhang et al. Megvii Inc (Face++)

## Abstract 摘要

We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet [12] on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ∼13× actual speedup over AlexNet while maintaining comparable accuracy.

我们提出了一种极其高效的CNN架构，称为ShuffleNet，这是特意为计算资源有限的移动设备设计的（如，10-150MFLOPs）。这种新的架构利用了两种新的运算，点分组卷积和通道混洗，极大的降低了计算量，同时可以保持准确率。在ImageNet上的分类和MS COCO上的目标检测，证明了ShuffleNet的优异性能，比其他架构要好很多，如比MobileNet在ImageNet分类任务中的top-1错误率更低(7.8%)，而计算资源只需要40MFLOPs。在一个基于ARM的移动设备上，ShuffleNet比AlexNet快了13x，而且可以得到类似的准确率。

## 1. Introduction 引言

Building deeper and larger convolutional neural networks (CNNs) is a primary trend for solving major visual recognition tasks [21, 9, 33, 5, 28, 24]. The most accurate CNNs usually have hundreds of layers and thousands of channels [9, 34, 32, 40], thus requiring computation at billions of FLOPs. This report examines the opposite extreme: pursuing the best accuracy in very limited computational budgets at tens or hundreds of MFLOPs, focusing on common mobile platforms such as drones, robots, and smartphones. Note that many existing works [16, 22, 43, 42, 38, 27] focus on pruning, compressing, or low-bit representing a “basic” network architecture. Here we aim to explore a highly efficient basic architecture specially designed for our desired computing ranges.

构建更深更大的CNNs是解决主要视觉识别任务的一个基本趋势。最准确的CNNs通常有几百层，几千个通道，所以需要的计算资源在是billions of FLOPs。本文研究相反的极限：在非常有限的计算资源中，几十或几百MFLOPs，追求最好的准确率，聚焦在普通的移动平台，如无人机，机器人和智能手机。注意很多现有的工作关注了剪枝、压缩，或低bit，表示基本的网络架构。这里我们的目标是，探索了非常高效的基本架构，特别为目标计算资源平台而设计的。

We notice that state-of-the-art basic architectures such as Xception [3] and ResNeXt [40] become less efficient in extremely small networks because of the costly dense 1 × 1 convolutions. We propose using pointwise group convolutions to reduce computation complexity of 1 × 1 convolutions. To overcome the side effects brought by group convolutions, we come up with a novel channel shuffle operation to help the information flowing across feature channels. Based on the two techniques, we build a highly efficient architecture called ShuffleNet. Compared with popular structures like [30, 9, 40], for a given computation complexity budget, our ShuffleNet allows more feature map channels, which helps to encode more information and is especially critical to the performance of very small networks.

我们注意到，目前最好的基本架构，如Xception和ResNet，在很小的网络下变得效率很低，因为密集的1×1卷积计算代价很大。我们提出使用点分组卷积来降低1×1卷积的计算复杂度。为克服分组卷积带来的副作用，我们提出了一种新的通道混洗的运算，帮助信息流过特征通道。基于这两种技术，我们构建了一个高效的架构，称为ShuffleNet。与流行的架构[30,9,40]相比，对于给定的计算复杂度预算，我们的ShuffleNet可以计算更多的特征图通道，这可以包含更多信息，对于很小的网络来说尤其重要。

We evaluate our models on the challenging ImageNet classification [4, 29] and MS COCO object detection [23] tasks. A series of controlled experiments shows the effectiveness of our design principles and the better performance over other structures. Compared with the state-of-the-art architecture MobileNet [12], ShuffleNet achieves superior performance by a significant margin, e.g. absolute 7.8% lower ImageNet top-1 error at level of 40 MFLOPs.

我们在ImageNet分类任务和MS COCO目标检测任务中评估我们的模型。一系列试验表明了我们的设计原则的有效性，以及超过其他模型的性能。比目前最好的MobileNet相比，ShuffleNet明显超过了其性能，如，以40MFLOPs的计算量得到了7.8%的top-1错误率。

We also examine the speedup on real hardware, i.e. an off-the-shelf ARM-based computing core. The ShuffleNet model achieves ∼13× actual speedup (theoretical speedup is 18×) over AlexNet [21] while maintaining comparable accuracy.

我们还在真实硬件上检验了加速效果，即，一个立即可用的基于ARM的计算核心。ShuffleNet比AlexNet有13x的加速效果（理论效果有18x加速），同时保持了类似的准确率。

## 2. Related Work

**Efficient Model Designs**. The last few years have seen the success of deep neural networks in computer vision tasks [21, 36, 28], in which model designs play an important role. The increasing needs of running high quality deep neural networks on embedded devices encourage the study on efficient model designs [8]. For example, GoogLeNet [33] increases the depth of networks with much lower complexity compared to simply stacking convolution layers. SqueezeNet [14] reduces parameters and computation significantly while maintaining accuracy. ResNet [9, 10] utilizes the efficient bottleneck structure to achieve impressive performance. SENet [13] introduces an architectural unit that boosts performance at slight computation cost. Concurrent with us, a very recent work [46] employs reinforcement learning and model search to explore efficient model designs. The proposed mobile NASNet model achieves comparable performance with our counterpart ShuffleNet model (26.0% @ 564 MFLOPs vs. 26.3% @ 524 MFLOPs for ImageNet classification error). But [46] do not report results on extremely tiny models (e.g. complexity less than 150 MFLOPs), nor evaluate the actual inference time on mobile devices.

**高效模型设计**。过去几年见证了DNN在计算机视觉任务中的成功，其中模型设计起了重要的作用。在嵌入式设备上运行高质量DNN的需求越来越多，这促进了高效模型的设计。比如，GoogLeNet增加了网络的深度，但与简单的堆叠卷积层相比，计算复杂度的增加却很少。SqueezeNet显著减少了参数和计算复杂度，同时保持了准确率。ResNet利用了高效的瓶颈结构得到了非常好的性能。SENet提出了一个架构单元，在略微增加计算代价的同时可以提升性能。与我们的工作同时，[46]使用强化学习和模型修锁来探索高效的模型设计。提出的NASNet模型的性能与我们的ShuffleNet类似(26.0% @ 564 MFLOPs vs. 26.3% @ 524 MFLOPs for ImageNet classification error)。但[46]没有给出极小模型的结果（如，小于150MFLOPs），也没有在移动设备上实际推理测试。

**Group Convolution**. The concept of group convolution, which was first introduced in AlexNet [21] for distributing the model over two GPUs, has been well demonstrated its effectiveness in ResNeXt [40]. Depthwise separable convolution proposed in Xception [3] generalizes the ideas of separable convolutions in Inception series [34, 32]. Recently, MobileNet [12] utilizes the depthwise separable convolutions and gains state-of-the-art results among lightweight models. Our work generalizes group convolution and depthwise separable convolution in a novel form.

**分组卷积**。分组卷积的概念，首先在AlexNet中提出，是为了将模型在两个GPU上运行，也在ResNeXt中证明了其有效性。Xception中提出的分层可分离卷积，推广了Inception系列中可分离卷积的思想。最近，MobileNet利用分层可分离卷积在轻量模型中得到了目前最好的结果。我们的工作以一种新形式推广了分组卷积和分层可分离卷积。

**Channel Shuffle Operation**. To the best of our knowledge, the idea of channel shuffle operation is rarely mentioned in previous work on efficient model design, although CNN library cuda-convnet [20] supports “random sparse convolution” layer, which is equivalent to random channel shuffle followed by a group convolutional layer. Such “random shuffle” operation has different purpose and been seldom exploited later. Very recently, another concurrent work [41] also adopt this idea for a two-stage convolution. However, [41] did not specially investigate the effectiveness of channel shuffle itself and its usage in tiny model design.

**通道混洗运算**。据我们所知，通道混洗的运算在之前的高效模型设计中极少被提到，但CNN库cuda-convnet支持“随机稀疏卷积”层，这等价于随机通道混洗和分组卷积层的组合。这样的“随机混洗”运算有不同的目的，之后很少被研究过。最近，另一个同时的工作[41]也采用了这种思想进行两阶段卷积。但是，[41]没有专门研究通道混洗本身的有效性，和其在微型模型设计中的使用。

**Model Acceleration**. This direction aims to accelerate inference while preserving accuracy of a pre-trained model. Pruning network connections [6, 7] or channels [38] reduces redundant connections in a pre-trained model while maintaining performance. Quantization [31, 27, 39, 45, 44] and factorization [22, 16, 18, 37] are proposed in literature to reduce redundancy in calculations to speed up inference. Without modifying the parameters, optimized convolution algorithms implemented by FFT [25, 35] and other methods [2] decrease time consumption in practice. Distilling [11] transfers knowledge from large models into small ones, which makes training small models easier.

**模型加速**。这个方向的目标是加速推理过程，同时保持预训练模型的准确度。网络连接或通道的剪枝，降低了预训练模型中连接的冗余度，同时保持了性能不降低。文献中还提出了量化和分解以降低计算中的冗余，加速推理。在没有修改参数的情况下，用FFT和其他方法实现的优化卷积算法可以降低耗时。蒸馏将知识从大模型迁移到小模型，使训练小模型更加容易。

## 3. Approach 方法

### 3.1. Channel Shuffle for Group Convolutions 用通道混洗进行分组卷积

Modern convolutional neural networks [30, 33, 34, 32, 9, 10] usually consist of repeated building blocks with the same structure. Among them, state-of-the-art networks such as Xception [3] and ResNeXt [40] introduce efficient depthwise separable convolutions or group convolutions into the building blocks to strike an excellent trade-off between representation capability and computational cost. However, we notice that both designs do not fully take the 1 × 1 convolutions (also called pointwise convolutions in [12]) into account, which require considerable complexity. For example, in ResNeXt [40] only 3 × 3 layers are equipped with group convolutions. As a result, for each residual unit in ResNeXt the pointwise convolutions occupy 93.4% multiplication-adds (cardinality = 32 as suggested in [40]). In tiny networks, expensive pointwise convolutions result in limited number of channels to meet the complexity constraint, which might significantly damage the accuracy.

现代卷积神经网络通常包含重复的同样结构的模块。其中，目前最好的网络，如Xception和ResNeXt在模块中引入了高效的分层可分离卷积或分组卷积，达到了表示能力和计算代价的极好折中。但是，我们注意到，这两种设计都没有考虑到1×1卷积（[12]中也称为点卷积），这需要相当的复杂度。比如，在ResNeXt中，只将3×3的卷积层进行了分组卷积。结果是，对ResNeXt中的每个残差单元，点卷积占整体计算量Mul-Add的93.4%（cardinality=32，如[40]中建议）。在微型网络中，复杂度很高的点卷积导致只能使用有限的通道数，才能不超出复杂度限制，这可能会显著损害准确率。

To address the issue, a straightforward solution is to apply channel sparse connections, for example group convolutions, also on 1 × 1 layers. By ensuring that each convolution operates only on the corresponding input channel group, group convolution significantly reduces computation cost. However, if multiple group convolutions stack together, there is one side effect: outputs from a certain channel are only derived from a small fraction of input channels. Fig 1 (a) illustrates a situation of two stacked group convolution layers. It is clear that outputs from a certain group only relate to the inputs within the group. This property blocks information flow between channel groups and weakens representation.

为解决这个问题，一个直接解决方案是使用在1×1层中也使用通道稀疏连接，比如分组卷积。分组卷积会确保每个卷积只在对应的输入通道组中进行计算，会显著的降低计算代价。但是，如果多个分组卷积堆叠在一起，会有一个副作用：从特定通道的输出只是从一小部分输入通道中推导出来的。图1(a)描述的是两个分组卷积层堆叠在一起的情况。可以很清楚的看到，从某个组的输出只与这个组的输入有关。这种性质会阻碍信息在通道组之间的流动，弱化表示的效果。

If we allow group convolution to obtain input data from different groups (as shown in Fig 1 (b)), the input and output channels will be fully related. Specifically, for the feature map generated from the previous group layer, we can first divide the channels in each group into several subgroups, then feed each group in the next layer with different subgroups. This can be efficiently and elegantly implemented by a channel shuffle operation (Fig 1 (c)): suppose a convolutional layer with g groups whose output has g × n channels; we first reshape the output channel dimension into (g, n), transposing and then flattening it back as the input of next layer. Note that the operation still takes effect even if the two convolutions have different numbers of groups. Moreover, channel shuffle is also differentiable, which means it can be embedded into network structures for end-to-end training.

如果我们让分组卷积从不同分组中得到输入数据（如图1(b)所示），那么输入和输出通道会充分相关。具体的，对于从之前的分组层生成的特征图，我们首先将每个组中的通道分成几个小组，然后向下一层的每个分组送入不同的小组数据。这可以通过一个通道混洗运算高效简洁的实现（图1(c)）：假设一个卷积层有g个分组，输出有g×n个通道；我们首先将输出通道的维度变换为(g,n)，将其转置，并拉平，作为下一层的输入。注意，即使在两个卷积层分组数量不同的情况下，这个运算也是有效的。而且，通道混洗还是可微分的，说明其可以嵌入网络结构中，进行端到端的训练。

Channel shuffle operation makes it possible to build more powerful structures with multiple group convolutional layers. In the next subsection we will introduce an efficient network unit with channel shuffle and group convolution.

通道混洗运算可以使用多个分组卷积层构建更强大的结构。下一小节中，我们提出一种使用通道混洗和分组卷积的高效网络单元。

Figure 1. Channel shuffle with two stacked group convolutions. GConv stands for group convolution. a) two stacked convolution layers with the same number of groups. Each output channel only relates to the input channels within the group. No cross talk; b) input and output channels are fully related when GConv2 takes data from different groups after GConv1; c) an equivalent implementation to b) using channel shuffle.

### 3.2. ShuffleNet Unit

Taking advantage of the channel shuffle operation, we propose a novel ShuffleNet unit specially designed for small networks. We start from the design principle of bottleneck unit [9] in Fig 2(a). It is a residual block. In its residual branch, for the 3 × 3 layer, we apply a computational economical 3 × 3 depthwise convolution [3] on the bottleneck feature map. Then, we replace the first 1 × 1 layer with pointwise group convolution followed by a channel shuffle operation, to form a ShuffleNet unit, as shown in Fig 2 (b). The purpose of the second pointwise group convolution is to recover the channel dimension to match the shortcut path. For simplicity, we do not apply an extra channel shuffle operation after the second pointwise layer as it results in comparable scores. The usage of batch normalization (BN) [15] and nonlinearity is similar to [9, 40], except that we do not use ReLU after depthwise convolution as suggested by [3]. As for the case where ShuffleNet is applied with stride, we simply make two modifications (see Fig 2 (c)): (i) add a 3 × 3 average pooling on the shortcut path; (ii) replace the element-wise addition with channel concatenation, which makes it easy to enlarge channel dimension with little extra computation cost.

利用通道混洗运算，我们提出一种新的ShuffleNet单元，专门为小型网络设计。我们从瓶颈单元的设计原则开始，如图2(a)。这是一个残差模块。在其残差分支，对于3×3的层，我们在瓶颈特征图上使用一个节约计算量的3×3分层卷积[3]。然后，我们将第一个1×1层替换为点分组卷积和通道混洗运算的组合，以形成一个ShuffleNet单元，如图2(b)所示。第二个点分组卷积的目的是恢复通道维度，与捷径连接通道数匹配。为简化起见，我们在第二个点卷积层后就没有加上通道混洗运算，因为得到的分数类似。BN和非线性的使用与[9,40]类似，但我们没有在分层卷积后使用ReLU，这是[3]中建议的。对于ShuffleNet带有步长的情况，我们只进行两个修正（见图2(c)）：(i)在捷径连接上加入一个3×3平均池化层；(ii)将逐元素相加替换成通道拼接，这使得扩大通道维度很容易，增加的计算量很小。

Thanks to pointwise group convolution with channel shuffle, all components in ShuffleNet unit can be computed efficiently. Compared with ResNet [9] (bottleneck design) and ResNeXt [40], our structure has less complexity under the same settings. For example, given the input size c × h × w and the bottleneck channels m, ResNet unit requires $hw(2cm + 9m^2)$ FLOPs and ResNeXt has $hw(2cm + 9m^2/g)$ FLOPs, while our ShuffleNet unit requires only hw(2cm/g + 9m) FLOPs, where g means the number of groups for convolutions. In other words, given a computational budget, ShuffleNet can use wider feature maps. We find this is critical for small networks, as tiny networks usually have an insufficient number of channels to process the information.

有了点分组卷积和通道混洗，ShuffleNet单元中的所有部分都可以得到高效的计算。与ResNet（瓶颈设计）和ResNeXt相比，我们的结构在相同的设置下复杂度更低。比如，给定输入大小c × h × w，瓶颈通道数量m，ResNet单元需要的计算量为$hw(2cm + 9m^2)$ FLOPs，ResNeXt需要$hw(2cm + 9m^2/g)$ FLOPs，而我们的ShuffleNet只需要hw(2cm/g + 9m) FLOPs，其中g是卷积分组的数量。换句话说，给定计算量预算，ShuffleNet可以使用更宽的特征图。我们发现这对于小型网络非常关键，因为小型网络的通道数量通常不足以处理信息。

In addition, in ShuffleNet depthwise convolution only performs on bottleneck feature maps. Even though depthwise convolution usually has very low theoretical complexity, we find it difficult to efficiently implement on low-power mobile devices, which may result from a worse computation/memory access ratio compared with other dense operations. Such drawback is also referred in [3], which has a runtime library based on TensorFlow [1]. In ShuffleNet units, we intentionally use depthwise convolution only on bottleneck in order to prevent overhead as much as possible.

另外，在ShuffleNet中，分层卷积只在瓶颈特征图上进行计算。即使分层卷积理论上计算复杂度很低，但我们发现在低能耗移动设备上很难高效实现，与其他密集运算相比，这可能是计算/内存访问比例很差的原因导致的。这样的缺点在[3]中也提到了，其中有个基于Tensorflow的运行库。在ShuffleNet单元中，我们刻意的只在瓶颈结构中使用分层卷积，以尽可能节省开销。

Figure 2. ShuffleNet Units. a) bottleneck unit [9] with depthwise convolution (DWConv) [3, 12]; b) ShuffleNet unit with pointwise group convolution (GConv) and channel shuffle; c) ShuffleNet unit with stride = 2.

### 3.3. Network Architecture 网络架构

Built on ShuffleNet units, we present the overall ShuffleNet architecture in Table 1. The proposed network is mainly composed of a stack of ShuffleNet units grouped into three stages. The first building block in each stage is applied with stride = 2. Other hyper-parameters within a stage stay the same, and for the next stage the output channels are doubled. Similar to [9], we set the number of bottleneck channels to 1/4 of the output channels for each ShuffleNet unit. Our intent is to provide a reference design as simple as possible, although we find that further hyper-parameter tunning might generate better results.

在ShuffleNet单元的基础上，我们提出了表1中的总体ShuffleNet架构。提出的网络主要是由ShuffleNet单元的堆叠构成的，分成三个阶段。每个阶段的第一个模块都是步长为2的。一个阶段中的其他超参数都是一样的，对于下一阶段，其输出通道数加倍。与[9]类似，我们设ShuffleNet单元中，瓶颈通道的数量为输出通道的1/4。我们是要给出一个尽可能简单的参考设计，但我们也发现，进一步的超参数调节可能会生成更好的结果。

In ShuffleNet units, group number g controls the connection sparsity of pointwise convolutions. Table 1 explores different group numbers and we adapt the output channels to ensure overall computation cost roughly unchanged (∼140 MFLOPs). Obviously, larger group numbers result in more output channels (thus more convolutional filters) for a given complexity constraint, which helps to encode more information, though it might also lead to degradation for an individual convolutional filter due to limited corresponding input channels. In Sec 4.1.1 we will study the impact of this number subject to different computational constrains.

在ShuffleNet单元中，分组数量g控制着点卷积的连接稀疏度。表1列出了不同分组数量，为保证总体计算复杂度大致不变(~140 MFLOPs)，我们调节了输出通道数量。显然，在给定的复杂度限制下，更大的分组数量会得到更多的输出通道数量（所以需要更多的卷积滤波器），这有助于编码更多信息，但也会由于对应的输入通道数量有限，导致单个卷积滤波器的降质。在4.1.1节中，我们研究了在不同的计算量限制下，这个数量的影响。

To customize the network to a desired complexity, we can simply apply a scale factor s on the number of channels. For example, we denote the networks in Table 1 as ”ShuffleNet 1×”, then ”ShuffleNet s×” means scaling the number of filters in ShuffleNet 1× by s times thus overall complexity will be roughly $s^2$ times of ShuffleNet 1×.

为定制理想复杂度下的网络，我们可以简单的在通道数量上增加一个尺度因子s。比如，我们将表1中的网络表示为ShuffleNet 1×，那么ShuffleNet s×意思就是将ShuffleNet 1×中的滤波器数量扩大s倍，所以整体复杂度大约是ShuffleNet 1×的$s^2$倍。

Table 1. ShuffleNet architecture. The complexity is evaluated with FLOPs, i.e. the number of floating-point multiplication-adds. Note that for Stage 2, we do not apply group convolution on the first pointwise layer because the number of input channels is relatively small.

Layer | Output Size | KSize | Stride | Repeat | g=1 | g=2 | g=3 | g=4 | g=8
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Image | 224×224 | - | - | - | 3 | 3 | 3 | 3 | 3
Conv1 | 112×112 | 3×3 | 2 | 1 | 24 | 24 | 24 | 24 | 24
MaxPool | 56×56 | 3×3 | 2 | - | - | - | - | - | -
Stage2 | 28×28 | - | 2 | 1 | 144 | 200 | 240 | 272 | 384
Stage2 | 28×28 | - | 1 | 3 | 144 | 200 | 240 | 272 | 384
Stage3 | 14×14 | - | 2 | 1 | 288 | 400 | 480 | 544 | 768
Stage3 | 14×14 | - | 1 | 7 | 288 | 400 | 480 | 544 | 768
Stage4 | 7×7 | - | 2 | 1 | 576 | 800 | 960 | 1088 | 1536
Stage4 | 7×7 | - | 1 | 3 | 576 | 800 | 960 | 1088 | 1536
GlobalPool | 1×1 | 7×7 | 
FC | - | - | - | - | 1000 | 1000 | 1000 | 1000 | 1000
Complexity | - | - | - | - | 143M | 140M | 137M | 133M | 137M

## 4. Experiments 试验

We mainly evaluate our models on the ImageNet 2012 classification dataset [29, 4]. We follow most of the training settings and hyper-parameters used in [40], with two exceptions: (i) we set the weight decay to 4e-5 instead of 1e-4 and use linear-decay learning rate policy (decreased from 0.5 to 0); (ii) we use slightly less aggressive scale augmentation for data preprocessing. Similar modifications are also referenced in [12] because such small networks usually suffer from underfitting rather than overfitting. It takes 1 or 2 days to train a model for 3×10^5 iterations on 4 GPUs, whose batch size is set to 1024. To benchmark, we compare single crop top-1 performance on ImageNet validation set, i.e. cropping 224 × 224 center view from 256× input image and evaluating classification accuracy. We use exactly the same settings for all models to ensure fair comparisons.

我们主要在ImageNet 2012分类数据集上评估我们的模型。我们主要采用[40]中的训练设置和超参数，但有两点不同：(i)我们设置权重衰减为4e-5，而不是1e-4，并使用线性衰减的学习速率策略（从0.5降低到0）；(ii)我们使用数据预处理的尺度扩充没那么激进。[12]中也采用了类似的变化，因为小型模型通常有欠拟合的问题，而不是过拟合。在4个GPUs上训练一个模型需要1到2天，3×10^5次迭代，其batch size大小设为1024。为进行基准测试，我们在ImageNet验证集上比较单剪切块的top-1性能，即256×的输入图像中剪切出224×224的中间部分，计算分类准确率。我们对所有模型使用相同的设置，以确保公平比较。

### 4.1 Ablation Study 分离对比试验

The core idea of ShuffleNet lies in pointwise group convolution and channel shuffle operation. In this subsection we evaluate them respectively. ShuffleNet的核心思想是点分组卷积和通道混洗运算。本小节中，我们分别进行评估。

#### 4.1.1 Pointwise Group Convolutions 点分组卷积

To evaluate the importance of pointwise group convolutions, we compare ShuffleNet models of the same complexity whose numbers of groups range from 1 to 8. If the group number equals 1, no pointwise group convolution is involved and then the ShuffleNet unit becomes an ”Xception-like” [3] structure. For better understanding, we also scale the width of the networks to 3 different complexities and compare their classification performance respectively. Results are shown in Table 2.

为评估点分组卷积的重要性，我们比较相同复杂度的ShuffleNet模型，分组数量从1到8。如果分组数量为1，那么就没有点分组卷积了，ShuffleNet单元变成一个类似Xception的结构。为更好的理解，我们对网络的宽度也进行了三种缩放，并分别比较其分类性能，结果如表2所示。

Table 2. Classification error vs. number of groups g (smaller number represents better performance)

Model | MFLOPs | g=1 | g=2 | g=3 | g=4 | g=8
--- | --- | --- | --- | --- | --- | ---
ShuffleNet 1× | 140 | 33.6 | 32.7 | 32.6 | 32.8 | 32.4
ShuffleNet 0.5× | 38 | 45.1 | 44.4 | 43.2 | 41.6 | 42.3
ShuffleNet 0.25× | 13 | 57.1 | 56.8 | 55.0 | 54.2 | 52.7

From the results, we see that models with group convolutions (g > 1) consistently perform better than the counterparts without pointwise group convolutions (g = 1). Smaller models tend to benefit more from groups. For example, for ShuffleNet 1× the best entry (g = 8) is 1.2% better than the counterpart, while for ShuffleNet 0.5× and 0.25× the gaps become 3.5% and 4.4% respectively. Note that group convolution allows more feature map channels for a given complexity constraint, so we hypothesize that the performance gain comes from wider feature maps which help to encode more information. In addition, a smaller network involves thinner feature maps, meaning it benefits more from enlarged feature maps.

从结果中我们可以看到，进行分组卷积的模型(g > 1)一直都可以取得更好的效果。更小的模型从分组中受益更多。比如，ShuffleNet 1×的最好结果(g=8)比(g=1)改进了1.2%，而ShuffleNet 0.5×和0.25×的差距分别为3.5%和4.4%。注意分组卷积时，在相同的复杂度限制下可以有更多的特征图通道，所以我们假设性能改进是从更多的特征图中得到的，这可以包含更多的信息。另外，更小的网络其特征图更细窄，意味着从更多的特征图中受益更多。

Table 2 also shows that for some models (e.g. ShuffleNet 0.5×) when group numbers become relatively large (e.g. g = 8), the classification score saturates or even drops. With an increase in group number (thus wider feature maps), input channels for each convolutional filter become fewer, which may harm representation capability. Interestingly, we also notice that for smaller models such as ShuffleNet 0.25× larger group numbers tend to better results consistently, which suggests wider feature maps bring more benefits for smaller models.

表2还显示，对于一些模型（如ShuffleNet 0.5x），当分组数量相对较大时（如g=8），分类分数会饱和，甚至下降。随着分组数量增加（特征图变宽），每个卷积滤波器的输入通道也变少了，这可能会使表示能力受损。有趣的是，我们还注意到，对于更小的模型，如ShuffleNet 0.25×，更大的分组数量会一直得到更好的结果，这说明，对于更小的模型来说，更宽的特征图会带来更多好处。

#### 4.1.2 Channel Shuffle vs. No Shuffle

The purpose of shuffle operation is to enable cross-group information flow for multiple group convolution layers. Table 3 compares the performance of ShuffleNet structures (group number is set to 3 or 8 for instance) with/without channel shuffle. The evaluations are performed under three different scales of complexity. It is clear that channel shuffle consistently boosts classification scores for different settings. Especially, when group number is relatively large (e.g. g = 8), models with channel shuffle outperform the counterparts by a significant margin, which shows the importance of cross-group information interchange.

混洗运算的目的是，对于多个分组卷积层，使跨分组的信息流动起来。表3比较了在有/没有通道混洗情况下的ShuffleNet结构的性能（分组数量为3或8）。在三种不同复杂度下比计算了其性能。很清楚可以看到，通道混洗可以持续改进不同设置下的分类分数。尤其是，当分组数量更大的时候（如g=8），使用的通道混洗的模型超过没有混洗的更多，这表明跨分组的信息交换是非常重要。

Table 3. ShuffleNet with/without channel shuffle (smaller number represents better performance)

Model | cls err.(%, no shuffle) | cls err.(%, shuffle) | ∆ err. (%)
--- | --- | --- | ---
ShuffleNet 1x (g = 3) | 34.5 | 32.6 | 1.9
ShuffleNet 1x (g = 8) | 37.6 | 32.4 | 5.2
ShuffleNet 0.5x (g = 3) | 45.7 | 43.2 | 2.5
ShuffleNet 0.5x (g = 8) | 48.1 | 42.3 | 5.8
ShuffleNet 0.25x (g = 3) | 56.3 | 55.0 | 1.3
ShuffleNet 0.25x (g = 8) | 56.5 | 52.7 | 3.8

### 4.2. Comparison with Other Structure Units 与其他结构单元的对比

Recent leading convolutional units in VGG [30], ResNet [9], GoogleNet [33], ResNeXt [40] and Xception [3] have pursued state-of-the-art results with large models (e.g. ≥ 1GFLOPs), but do not fully explore low-complexity conditions. In this section we survey a variety of building blocks and make comparisons with ShuffleNet under the same complexity constraint.

最近领先的卷积单元，包括VGG [30], ResNet [9], GoogleNet [33], ResNeXt [40] and Xception [3]，都用大型模型追求目前最好的结果（如≥ 1GFLOPs），但没有充分探索低复杂度的条件。本节中，我们调查了很多模块，与ShuffleNet在相同复杂度限制下进行比较。

For fair comparison, we use the overall network architecture as shown in Table 1. We replace the ShuffleNet units in Stage 2-4 with other structures, then adapt the number of channels to ensure the complexity remains unchanged. The structures we explored include:

为公平比较，我们使用的总体网络架构如表1所示。我们将第2到第4阶段的ShuffleNet单元替换成其他结构，然后修改通道数量以确保复杂度保持不变。我们探索的结构包括：

- VGG-like. Following the design principle of VGG net [30], we use a two-layer 3×3 convolutions as the basic building block. Different from [30], we add a Batch Normalization layer [15] after each of the convolutions to make end-to-end training easier. 按照VGGNet[30]的设计原则，我们使用2层的3×3卷积作为基本模块。与[30]不同的是，我们在每个卷积后加入了一个BN层[15]，以确保端到端的训练更容易。

- ResNet. We adopt the ”bottleneck” design in our experiment, which has been demonstrated more efficient in [9]. Same as [9], the bottleneck ratio is also 1 : 4(In the bottleneck-like units (like ResNet, ResNeXt or ShuffleNet) bottleneck ratio implies the ratio of bottleneck channels to output channels. For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times the width of the bottleneck feature map). 我们在试验中采用瓶颈设计，这在[9]中也证明了更有效。与[9]一样，瓶颈率也是1:4（在瓶颈类的单元中，如ResNet, ResNeXt或ShuffleNet，瓶颈率为瓶颈通道与输出通道数量之比。例如，瓶颈率=1:4表示，输出特征图数量是瓶颈特征图宽度的4倍）。

- Xception-like. The original structure proposed in [3] involves fancy designs or hyper-parameters for different stages, which we find difficult for fair comparison on small models. Instead, we remove the pointwise group convolutions and channel shuffle operation from ShuffleNet (also equivalent to ShuffleNet with g = 1). The derived structure shares the same idea of “depthwise separable convolution” as in [3], which is called an Xception-like structure here. [3]中提出的原始结构包括花哨的设计或不同阶段的超参数，这在小型模型的限制下很难进行公平比较。相反，我们从ShuffleNet中去掉了点分组卷积和通道混洗运算（即设g=1）。得到的结构与[3]中的分层可分离卷积思想一样，这里称之为类Xception结构。

- ResNeXt. We use the settings of cardinality = 16 and bottleneck ratio = 1 : 2 as suggested in [40]. We also explore other settings, e.g. bottleneck ratio = 1 : 4, and get similar results. 我们使用的设置为cardinality=16和瓶颈率=1:2。我们还探索了其他设置，如，瓶颈率为1:4，得到了类似的结果。

We use exactly the same settings to train these models. Results are shown in Table 4. Our ShuffleNet models outperform most others by a significant margin under different complexities. Interestingly, we find an empirical relationship between feature map channels and classification accuracy. For example, under the complexity of 38 MFLOPs, output channels of Stage 4 (see Table 1) for VGG-like, ResNet, ResNeXt, Xception-like, ShuffleNet models are 50, 192, 192, 288, 576 respectively, which is consistent with the increase of accuracy. Since the efficient design of ShuffleNet, we can use more channels for a given computation budget, thus usually resulting in better performance.

我们使用相同的设置来训练这些模型。结果如表4所示。我们的ShuffleNet模型，在相同复杂度下，超过了其他大多数模型很多。有趣的是，我们发现了特征图通道数量和分类准确率之间的一个经验关系。比如，在38MFLOPs的复杂度下，第4阶段的输出通道数量（见表1），对于VGG-like, ResNet, ResNeXt, Xception-like, ShuffleNet模型分别为50,192,192,288和576，与准确率的增加关系是一致的。由于ShuffleNet设计的高效性，我们可以在给定的计算复杂度下使用更多通道，所以通常会得到更好的性能。

Table 4. Classification error vs. various structures (%, smaller number represents better performance). We do not report VGG-like structure on smaller networks because the accuracy is significantly worse.

Complexity (MFLOPs) | VGG-like | ResNet | Xception-like | ResNeXt | ShuffleNet(ours)
--- | --- | --- | --- | --- | ---
140 | 50.7 | 37.3 | 33.6 | 33.3 | 32.4(1×,g=8)
38 | - | 48.8 | 45.1 | 46.0 | 41.6(0.5×,g=4)
13 | - | 63.7 | 57.1 | 65.2 | 52.7(0.25×,g=8)

Note that the above comparisons do not include GoogleNet or Inception series [33, 34, 32]. We find it non-trivial to generate such Inception structures to small networks because the original design of Inception module involves too many hyper-parameters. As a reference, the first GoogleNet version [33] has 31.3% top-1 error at the cost of 1.5 GFLOPs (See Table 6). More sophisticated Inception versions [34, 32] are more accurate, however, involve significantly increased complexity. Recently, Kim et al. propose a lightweight network structure named PVANET [19] which adopts Inception units. Our reimplemented PVANET (with 224×224 input size) has 29.7% classification error with a computation complexity of 557 MFLOPs, while our ShuffleNet 2x model (g = 3) gets 26.3% with 524 MFLOPs (see Table 6).

注意上述比较没有包括GoogLeNet或Inception系列。我们发现将这些Inception结构网络生成小规模网络比较复杂，因为Inception模块的原始设计涉及到太多超参数。作为参考，GoogLeNet第一版在1.5GFLOPs下有31.3%的top-1错误率（见表6）。更复杂的Inception版本准确率更高，但其复杂度也大大增加。最近，Kim等人提出一种轻量网络名为PVANET，采用了Inception单元。我们重新实现的PVANET（输入大小224×224）分类错误率为29.7%，计算复杂度为557MFLOPs，而我们的ShuffleNet 2×模型(g=3)在524MFLOPs下得到了26.3%的错误率。

### 4.3. Comparison with MobileNets and Other Frameworks

Recently Howard et al. have proposed MobileNets [12] which mainly focus on efficient network architecture for mobile devices. MobileNet takes the idea of depthwise separable convolution from [3] and achieves state-of-the-art results on small models.

最近Howard等提出了MobileNets，也是为移动设备设计的高效网络架构。MobileNet采用了分层可分离卷积的思想，在小型模型中取得了目前最好的结果。

Table 5 compares classification scores under a variety of complexity levels. It is clear that our ShuffleNet models are superior to MobileNet for all the complexities. Though our ShuffleNet network is specially designed for small models (< 150 MFLOPs), we find it is still better than MobileNet for higher computation cost, e.g. 3.1% more accurate than MobileNet 1× at the cost of 500 MFLOPs. For smaller networks (∼40 MFLOPs) ShuffleNet surpasses MobileNet by 7.8%. Note that our ShuffleNet architecture contains 50 layers while MobileNet only has 28 layers. For better understanding, we also try ShuffleNet on a 26-layer architecture by removing half of the blocks in Stage 2-4 (see ”ShuffleNet 0.5× shallow (g = 3)” in Table 5). Results show that the shallower model is still significantly better than the corresponding MobileNet, which implies that the effectiveness of ShuffleNet mainly results from its efficient structure, not the depth.

表5在不同复杂度下比较了分类效果。很明显，我们的ShuffleNet模型在各种复杂度下都比MobileNet要好。虽然我们的ShuffleNet是专门为小型模型设计的(< 150 MFLOPs)，我们发现在更高的计算代价下还是比MobileNet要好，如在500MFLOPs情况下，比MobileNet 1×的准确率高3.1%。对于更小的网络(∼40 MFLOPs)，ShuffleNet超过了MobileNet 7.8%。注意，我们的ShuffleNet架构包含50层，而MobileNet只有28层。为更好的理解，我们也尝试了26层架构的ShuffleNet，移除了第2-4阶段的一半模块（见表5中的ShuffleNet 0.5× shallow）。结果表明，我们更浅的模型仍然比对应的MobileNet要好很多，这说明ShuffleNet的有效性主要源自于其高效的架构，而不是深度。

Table 5. ShuffleNet vs. MobileNet [12] on ImageNet Classification

Model | MFLOPs | cls err.(%) | ∆ err. (%)
--- | --- | --- | ---
1.0 MobileNet-224 | 569 | 29.4 | -
ShuffleNet 2× (g = 3) | 524 | 26.3 | 3.1
ShuffleNet 2× (with SE[13], g = 3) | 527 | 24.7 | 4.7
0.75 MobileNet-224 | 325 | 31.6 | -
ShuffleNet 1.5× (g = 3) | 292 | 28.5 | 3.1
0.5 MobileNet-224 | 149 | 36.3 | -
ShuffleNet 1× (g = 8) | 140 | 32.4 | 3.9
0.25 MobileNet-224 | 41 | 49.4 | -
ShuffleNet 0.5× (g = 4) | 38 | 41.6 | 7.8
ShuffleNet 0.5× (shallow, g = 3) | 40 | 42.8 | 6.6

Table 6 compares our ShuffleNet with a few popular models. Results show that with similar accuracy ShuffleNet is much more efficient than others. For example, ShuffleNet 0.5× is theoretically 18× faster than AlexNet [21] with comparable classification score. We will evaluate the actual running time in Sec 4.5.

表6将ShuffleNet与一些流行模型进行了比较。结果表明，在类似的准确率下，ShuffleNet比其他模型要高效很多。例如，ShuffleNet 0.5×在类似的准确率下，理论上比AlexNet快18×。我们还在4.5节中给出了实际运行时间。

Table 6. Complexity comparison. *Implemented by BVLC (https://github.com/BVLC/caffe/tree/master/models/bvlc googlenet)

Model | cls err.(%) | MFLOPs
--- | --- | ---
VGG-16 [30] | 28.5 | 15300
ShuffleNet 2× (g = 3) | 26.3 | 524
GoogleNet [33]* | 31.3 | 1500
ShuffleNet 1× (g = 8) | 32.4 | 140
AlexNet [21] | 42.8 | 720
SqueezeNet [14] | 42.5 | 833
ShuffleNet 0.5× (g = 4) | 41.6 | 38

It is also worth noting that the simple architecture design makes it easy to equip ShuffeNets with the latest advances such as [13, 26]. For example, in [13] the authors propose Squeeze-and-Excitation (SE) blocks which achieve state-of-the-art results on large ImageNet models. We find SE modules also take effect in combination with the backbone ShuffleNets, for instance, boosting the top-1 error of ShuffleNet 2× to 24.7% (shown in Table 5). Interestingly, though negligible increase of theoretical complexity, we find ShuffleNets with SE modules are usually 25 ∼ 40% slower than the “raw” ShuffleNets on mobile devices, which implies that actual speedup evaluation is critical on low-cost architecture design. In Sec 4.5 we will make further discussion.

值得注意的是，ShuffleNet架构设计非常简单，使其可以利用最新的进展，如[13,26]。比如，[13]中提出了SE模块，在大规模ImageNet模型中取得了目前最好的结果。我们发现SE模块与ShuffleNet骨干组合在一起也起作用，比如，将ShuffleNet 2×的top-1错误率降低至24.7%（见表5）。有趣的是，虽然理论上复杂度的增加很小，我们发现ShuffleNet与SE模块一起通常比原始ShuffleNet在移动设备上慢了25~40%，这说明在低代价架构设计中，实际的加速效果非常关键。在4.5节中，我们会进一步讨论。

### 4.4. Generalization Ability 泛化能力

To evaluate the generalization ability for transfer learning, we test our ShuffleNet model on the task of MS COCO object detection [23]. We adopt Faster-RCNN [28] as the detection framework and use the publicly released Caffe code [28, 17] for training with default settings. Similar to [12], the models are trained on the COCO train+val dataset excluding 5000 minival images and we conduct testing on the minival set. Table 7 shows the comparison of results trained and evaluated on two input resolutions. Comparing ShuffleNet 2× with MobileNet whose complexity are comparable (524 vs. 569 MFLOPs), our ShuffleNet 2× surpasses MobileNet by a significant margin on both resolutions; our ShuffleNet 1× also achieves comparable results with MobileNet on 600× resolution, but has ∼4× complexity reduction. We conjecture that this significant gain is partly due to ShuffleNet’s simple design of architecture without bells and whistles.

为评估迁移学习的泛化能力，我们在MS COCO目标检测任务中测试我们的ShuffleNet模型。我们采用Faster R-CNN作为检测框架，使用开源的Caffe代码进行默认设置的训练。与[12]类似，模型在COCO trainval数据集（去除了5000幅minival图像）上进行训练，在minival集上进行测试。表7给出了两种输入分辨率下的训练和评估结果比较。将ShuffleNet 2×与复杂度相似的MobileNet进行比较(524 vs. 569 MFLOPs)，我们的ShuffleNet 2×在两种分辨率上都超过了MobileNet很多；我们的ShuffleNet 1×也在600×分辨率上与MobileNet取得了类似的结果，但复杂度降低了4×。我们推测这种显著改进部分是因为ShuffleNet的简单设计。

Table 7. Object detection results on MS COCO (larger numbers represents better performance). For MobileNets we compare two results: 1) COCO detection scores reported by [12]; 2) finetuning from our reimplemented MobileNets, whose training and finetuning settings are exactly the same as that for ShuffleNets.

Model | mAP [.5, .95] (300× image) | mAP [.5, .95] (600× image)
--- | --- | ---
ShuffleNet 2× (g = 3) | 18.7% | 25.0%
ShuffleNet 1× (g = 3) | 14.5% | 19.8%
1.0 MobileNet-224 [12] | 16.4% | 19.8%
1.0 MobileNet-224 (our impl.) | 14.9% | 19.3%

### 4.5. Actual Speedup Evaluation 实际加速效果

Finally, we evaluate the actual inference speed of ShuffleNet models on a mobile device with an ARM platform. Though ShuffleNets with larger group numbers (e.g. g = 4 or g = 8) usually have better performance, we find it less efficient in our current implementation. Empirically g = 3 usually has a proper trade-off between accuracy and actual inference time. As shown in Table 8, three input resolutions are exploited for the test. Due to memory access and other overheads, we find every 4× theoretical complexity reduction usually results in ∼2.6× actual speedup in our implementation. Nevertheless, compared with AlexNet [21] our ShuffleNet 0.5× model still achieves ∼13× actual speedup under comparable classification accuracy (the theoretical speedup is 18×), which is much faster than previous AlexNet-level models or speedup approaches such as [14, 16, 22, 42, 43, 38].

最后，我们评估了在ARM平台的移动设备上，ShuffleNet的实际推理加速效果。虽然更大分组数的ShuffleNet (e.g. g = 4 or g = 8)通常有更好的性能，我们发现在目前的实现中效率没那么高。经验上来说g=3可以取得很好的准去率和实际推理时间的折中。如表8所示，在测试集上进行了三种分辨率的试验。因为内存访问和其他开销的原因，我们发现计算复杂度每降低4×，在我们的实现中会得到～2.6×的实际加速效果。尽管如此，与AlexNet相比，我们的ShuffleNet 0.5×在类似的分类准确率下，仍然取得了～13×的实际加速效果（理论加速效果为18×），比之前的AlexNet级模型要快了很多。

Table 8. Actual inference time on mobile device (smaller number represents better performance). The platform is based on a single Qualcomm Snapdragon 820 processor. All results are evaluated with single thread.

Model | Cls err. (%) | FLOPs | 224×224 | 480×640 | 720×1280
--- | --- | --- | --- | --- | ---
ShuffleNet 0.5× (g = 3) | 43.2 | 38M | 15.2ms | 87.4ms | 260.1ms
ShuffleNet 1× (g = 3) | 32.6 | 140M | 37.8ms | 222.2ms | 684.5ms
ShuffleNet 2× (g = 3) | 26.3 | 524M | 108.8ms | 617.0ms | 1857.6ms
AlexNet [21] | 42.8 | 720M | 184ms | 1156.7ms | 3633.9ms
1.0 MobileNet-224 [12] | 29.4 | 569M | 110.0ms | 612.0ms | 1879.2ms