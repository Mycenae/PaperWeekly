# Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution

Yunpeng Chen et al. Facebook AI

## Abstract 摘要

In natural images, information is conveyed at different frequencies where higher frequencies are usually encoded with fine details and lower frequencies are usually encoded with global structures. Similarly, the output feature maps of a convolution layer can also be seen as a mixture of information at different frequencies. In this work, we propose to factorize the mixed feature maps by their frequencies, and design a novel Octave Convolution (OctConv) operation to store and process feature maps that vary spatially “slower” at a lower spatial resolution reducing both memory and computation cost. Unlike existing multi-scale methods, OctConv is formulated as a single, generic, plug-and-play convolutional unit that can be used as a direct replacement of (vanilla) convolutions without any adjustments in the network architecture. It is also orthogonal and complementary to methods that suggest better topologies or reduce channel-wise redundancy like group or depth-wise convolutions. We experimentally show that by simply replacing convolutions with OctConv, we can consistently boost accuracy for both image and video recognition tasks, while reducing memory and computational cost. An OctConv-equipped ResNet-152 can achieve 82.9% top-1 classification accuracy on ImageNet with merely 22.2 GFLOPs.

在自然图像中，信息是通过不同频率传递的，其中高频通常都是精细的细节信息，低频通常是全局结构。类似的，卷积层的输出特征图也可以看作是不同频率的信息混合。本文中，我们提出对混合的特征图进行频率分解，并设计了一种新的Octave Convolution (OctConv)运算，以在更低的分辨率上存储和处理空间变化缓慢的特征图，减少了内存和计算代价。与现有的多尺度方法不同，OctConv的形式为单个的、通用的、即插即用的卷积单元，可以直接用作常规卷积的替代品，不需要网络架构的任何调整。这对于group convolution或depth-wise convolution这样拓扑结构更好或减少通道的冗余性的方法也是一种补充。通过试验，我们表明，只要将卷积替换为OctConv，我们可以提高图像识别或视频识别任务的准确率，同时减少内存和计算代价。采用了OctConv的ResNet-152可以在ImageNet上取得82.9%的top-1分类准确率，计算量仅仅为22.2 GFLOPs。

## 1. Introduction 引言

Convolutional Neural Networks (CNNs) have achieved remarkable success in many computer vision tasks [18, 17, 42] and their efficiency keeps increasing with recent efforts to reduce the inherent redundancy in dense model parameters [16, 32, 40] and in the channel dimension of feature maps [45, 20, 7, 10]. However, substantial redundancy also exists in the spatial dimension of the feature maps produced by CNNs, where each location stores its own feature descriptor independently, while ignoring common information between adjacent locations that could be stored and processed together.

卷积神经网络(CNN)已经在很多计算机任务中取得了非凡的成功[18,17,42]，近年来很多努力都在减少这些密集模型参数的内在冗余性[16,32,40]，和特征图通道维数的内在冗余性[45,20,7,10]，因此效率不断提高。但是，CNN生成的特征图中，空间维度仍然存在非常多的冗余，每个位置独立存储其特征描述子，忽略了邻近位置之间的共同信息其实可以共同存储并处理。

Figure 1: (a) Motivation. The spatial frequency model for vision [1, 12] shows that natural image can be decomposed into a low and a high spatial frequency part. (b) The output maps of a convolution layer can also be factorized and grouped by their spatial frequency. (c) The proposed multi-frequency feature representation stores the smoothly changing, low-frequency maps in a low-resolution tensor to reduce spatial redundancy. (d) The proposed Octave Convolution operates directly on this representation. It updates the information for each group and further enables information exchange between groups.

图1：(a)动机。视觉的空域频率模型[1,12]表明，自然图像可以分解成低频和高频部分。(b)卷积层的输出特征图也可以通过空间频率分解并分组。(c)提出的多频率特征表示将平稳变化的低频特征图存储在低分辨率张量中，以降低空间冗余性。(d)提出的Octave Convolution直接在这个表示上进行运算，对每个分组进行信息更新，进一步使分组间的信息进行交换。

As shown in Figure 1(a), a natural image can be decomposed into a low spatial frequency component that describes the smoothly changing structure and a high spatial frequency component that describes the rapidly changing fine details [1, 12]. Similarly, we argue that the output feature maps of a convolution layer can also be decomposed into features of different spatial frequencies and propose a novel multi-frequency feature representation which stores high- and low-frequency feature maps into different groups as shown in Figure 1(b). Thus, the spatial resolution of the low-frequency group can be safely reduced by sharing information between neighboring locations to reduce spatial redundancy as shown in Figure 1(c). To accommodate the novel feature representation, we generalize the vanilla convolution, and propose Octave Convolution (OctConv) which takes in feature maps containing tensors of two frequencies one octave apart, and extracts information directly from the low-frequency maps without the need of decoding it back to the high-frequency as shown in Figure 1(d). As a replacement of vanilla convolution, OctConv consumes substantially less memory and computational resources. In addition, OctConv processes low-frequency information with corresponding (low-frequency) convolutions and effectively enlarges the receptive field in the original pixel space and thus can improve recognition performance.

如图1(a)所示，一幅自然图像可以分解为低空间频率部分，即描述平稳变化的结构，和高空间分辨率部分，描述迅速变化的精细细节[1,12]。类似的，我们认为卷积层的输出特征图也可以分解为不同空间频率的特征，提出一种新的多频率特征表示，将高频特征图和低频特征图存储进不同的组，如图1(b)所示。所以，低频组的空间分辨率可以很安全的降低，在相邻位置间共享信息，以降低空间冗余性，如图1(c)所示。我们将传统卷积进行了泛化，以容纳新的特征表示，提出了Octave Convolution (OctConv)，以两种频率张量的特征图为输入，直接从低频特征图中提取信息，不需要将其解码回高频域，如图1(d)所示。作为传统卷积的替代，OctConv消耗的存储和计算资源少很多。另外，OctConv以相应的（低频）卷积处理低频信息，有效的在原始像素空间增大了感受野，所以可以改进识别性能。

We design the OctConv in a generic way, making it a plug-and-play replacement for the vanilla convolution. Since OctConv mainly focuses on processing feature maps at multiple spatial frequencies and reducing their spatial redundancy, it is orthogonal and complementary to existing methods that focus on building better CNN topology [24, 38, 36, 34, 30], reducing channel-wise redundancy in convolutional feature maps [45, 10, 35, 33, 23] and reducing redundancy in dense model parameters [40, 16, 32]. We further discuss the integration of OctConv into the group, depth-wise, and 3D convolution cases. Moreover, different from methods that exploit multi-scale information [4, 41, 14], OctConv can be easily deployed as a plug-and-play unit to replace convolution, without the need of changing network architectures or requiring hyper-parameters tuning.

我们设计的OctConv是通用的，使其可以以即插即用的方式替代传统卷积。由于OctConv主要关注处理多个空间频率特征图，降低其空间冗余性，而现有的方法主要关注构建更好的CNN拓扑[24,38,36,34,30]，降低卷积特征图的通道冗余性[45,10,35,33,23]，降低密集模型参数的冗余性[40,16,32]，所以OctConv可以作为一种补充。我们进一步讨论将OctConv整合进分组卷积，depth-wise卷积和3D卷积的情形。而且，与利用多尺度信息的方法[4,41,14]不同，OctConv可以轻易的部署为即插即用的卷积替代，不需要改变网络架构，或超参数调节。

Our experiments demonstrate that by simply replacing the vanilla convolution with OctConv, we can consistently improve the performance of popular 2D CNN backbones including ResNet [18, 19], ResNeXt [45], DenseNet [24], MobileNet [20, 35] and SE-Net [21] on 2D image recognition on ImageNet [13], as well as 3D CNN backbones C2D [42] and I3D [42] on video action recognition on Kinetics [26, 3, 2]. The OctConv-equipped Oct-ResNet-152 can match or outperform state-of-the-art manually designed networks [33, 21] at lower memory and computational cost. Our contributions can be summarized as follows:

我们的试验表明，只要将传统卷积替换为OctConv，我们就可以改进流行的2D CNN骨干网络在ImageNet[13] 2D图像识别的性能，包括ResNet [18,19], ResNeXt [45], DenseNet [24], MobileNet [20, 35] 和 SE-Net [21]，以及3D CNN骨干网络C2D[42]和I3D[42]在Kinetics[26,3,2]上的视频动作识别上的性能。使用了OctConv的Oct-ResNet-152可以达到或超过目前最好的手工设计的网络[33,21]的性能，并且内存和计算代价都要小。我们的贡献可以总结如下：

- We propose to factorize convolutional feature maps into two groups at different spatial frequencies and process them with different convolutions at their corresponding frequency, one octave apart. As the resolution for low frequency maps can be reduced, this saves both storage and computation. This also helps each layer gain a larger receptive field to capture more contextual information.

- 我们提出分解卷积特征图为两组不同的空间频率，并用对应频率的不同卷积进行处理，one octave apart。由于低频率特征图的分辨率可以降低，这节省了存储空间和计算量。这也使每一层获得了更大的感受野，可以捕获更多上下文信息。

- We design a plug-and-play operation named OctConv to replace the vanilla convolution for operating on the new feature representation directly and reducing spatial redundancy. Importantly, OctConv is fast in practice and achieves a speedup close to the theoretical limit.

- 我们设计了一种即插即用的运算，名为OctConv，以替换传统卷积，在这种新的特征表示上进行直接运算，降低空间冗余性。重要的是，OctConv在运行时非常快，取得了接近理论极限的加速。

- We extensively study the properties of the proposed OctConv on a variety of backbone CNNs for image and video tasks and achieve significant performance gain even comparable to the best AutoML networks.

- 我们在很多CNN骨干结构上，在图像和视频任务上，广泛的研究了提出的OctConv的性质，得到了显著的性能提升，甚至与最好的AutoML工作可比。

## 2. Related Work 相关工作

**Improving the efficiency of CNNs**. Ever since the pioneering work on AlexNet [27] and VGG [36] which achieve astonishing results by stacking a set of convolution layers, researchers have made substantial efforts to improve the efficiency of CNNs. ResNet [18, 19] and DenseNet [24] improve the network topology by adding shortcut connections to early layers to enhance the feature reusing mechanism and alleviate optimization difficulties. ResNeXt [45] and ShuffleNet [47] use sparsely connected group convolutions to reduce redundancy in inter-channel connectivity, making it feasible to adopt deeper or wider networks under the same computational budget. Xception [10] and MobileNet [20, 35] adopt depth-wise convolutions that further reduce the connection density. Besides these manually designed networks, researchers also tried to atomically find the best network topology for a given task. NAS [49], PNAS [30] and AmoebaNet [34] successfully discovered typologies that perform better than manually designed networks. Another stream of work focuses on reducing the redundancy in the model parameters. DSD [16] reduces the redundancy in model connections by pruning connections of low weights. ThiNet [32] prunes convolutional filters based on statistics computed from its next layer. However, all of these methods ignore the redundancy on the spatial dimension of feature maps, which is addressed by the proposed OctConv, making OctConv orthogonal and complimentary to the previous methods.

**改进CNN的效率**。AlexNet[27]和VGG[36]通过将一系列卷积层堆叠到一起，取得了令人惊诧的结果，自从其先驱工作后，研究者进行了非常多的努力以改进CNN的效率。ResNet[18,19]和DenseNet[24]改进了网络拓扑结构，为前面的层增加了捷径连接，以改进特征复用机制，减轻优化困难。ResNeXt[45]和ShuffleNet[47]使用稀疏连接的分组卷积以降低通道间连接的冗余，在相同的计算代价下可以使用更深或更宽的网络。Xception[10]和MobileNet[20,35]采用depth-wise卷积，进一步降低了连接密度。除了这些手工设计的网络，研究者还试图对给定的任务自动寻找最佳的网络拓扑。NAS[49]，PNAS[30]和AmoebaNet[34]成功的发现了比手工设计的网络更好性能的网络拓扑结构。另一类工作聚焦在降低模型参数冗余性上。DSD[16]通过对较低权重的连接进行剪枝，从而降低了模型连接的冗余性。ThiNet[32]对卷积滤波器基于下一层的统计进行剪枝。但是，所有这些方法忽略了特征图空间维度中的冗余性，我们提出的OctConv则可以处理这个问题，使OctConv与之前的方法形成互补关系。

**Multi-scale Representation Learning**. The proposed OctConv performs convolution on feature maps at different spatial resolutions, resulting in a multi-scale feature representation with an enlarged receptive field. Scale-spaces have long been applied for local feature extraction, such as the once popular SIFT features [31]. In the deep learning era, existing methods focus on merging multi-scale features [28, 48, 14, 22] and better capturing long range information [42, 6, 9]. Such approaches however, aggregate information only at a small number of depths (usually in the middle or close to the end) of the network by inserting newly proposed blocks. The bL-Net [4] and ELASTIC-Net [41] frequently down- and up-sample the feature maps throughout the network to automatically learn multi-scale features. However, both methods are designed as a replacement of residual block which requires extra expertise and hyper-parameter tuning, especially when applied to different network architectures like MobileNetV1 [20], DenseNet [24]. Besides, these methods only synchronize multi-scale information at the end of each building block and synchronize all information on high resolution maps. In [22], Huang et al. keep multi-scale features throughout the network and have inter-scale connections at each layer of a DenseNet. Aiming at a reduced computational cost, they use only the coarse features as inputs to multiple classifiers at different depths. In contrast, OctConv is designed as a replacement of vanilla convolution and can be applied to existing CNNs directly without network architecture adjustment. For OctConv, multi-scale information is synchronized at every layer in each group, delivering higher learning capacity and efficiency. We extensively compare OctConv to all closely related approaches in the experiments section and show that OctConv CNNs give the top results on a number of challenging benchmarks.

**多尺度表示学习**。提出的OctConv在特征图的不同空间分辨率上进行卷积，得到了多尺度特征表示，并扩大了感受野。尺度空间应用于局部特征提取已经很久了，比如曾经流行的SIFT特征[31]。在深度学习时代，现有的方法聚焦于多尺度特征[28,48,14,22]和更好的捕捉长距离信息[42,6,9]的融合。但是，这些方法只对网络的少数几层深度进行聚合信息（通常是在网络的中间或接近结束的位置），方法是插入新提出的模块。bL-Net[4]和ELASTIC-Net[41]在整个网络中对特征图进行频繁的下采样和上采样，以自动的学习多尺度特征。但是，两种方法的设计都是用于取代残差模块，因为残差模块需要额外的专业知识和超参数调节，尤其是应用于不同的网络架构中的时候，如MobileNetV1[20], DenseNet[24]。另外，这些方法只在每个模块最后同步多尺度信息，并且在高分辨率特征图上同步所有信息。在[22]中，Huang等人在整个网络中保持多尺度特征，并且在一个DenseNet中的每一层都有尺度间连接。为了降低计算代价，他们只在不同的深度使用粗糙的特征作为多分类器的输入。比较之下，OctConv设计上就是用来替代传统卷积的，可以直接用于现有网络，无需进行网络架构调整。对于OctConv，多尺度信息在每个分组的每层都进行同步，说明有更高的学习容量和效率。我们将OctConv与其他所有紧密相关的方法都在试验中进行了广泛的比较，结果表明OctConv CNNs在几个很有挑战的基准测试中都给出了最好的结果。

## 3. Method 方法

In this section, we first introduce the octave feature representation for reducing the spatial redundancy in feature maps and then describe the Octave Convolution that operates directly on it. We also discuss implementation details and show how to integrate OctConv into group and depth-wise convolution architectures.

本节中，我们首先介绍了octave特征表示，可以降低特征图的空间冗余度，然后描述了直接在octave特征表示上进行的Octave Convolution。我们还讨论了实现细节，展示了怎样将OctConv整合进分组卷积架构和depth-wise卷积架构。

### 3.1. Octave Feature Representation

For the vanilla convolution, all input and output feature maps have the same spatial resolution. However, the spatial-frequency model [1, 12] argues that a natural image can be factorized into a low-frequency signal that captures the global layout and coarse structure, and a high-frequency part that captures fine details, as shown in Figure 1(a). In an analogous way, we argue that there is a subset of the feature maps that capture spatially low-frequency changes and contain spatially redundant information.

对于传统卷积，所有输入和输出特征图都有相同的空间分辨率。但是，空间-频率模型[1,12]认为，自然图像可以分解为全局布局和粗糙结构的低频信号，和精细细节的高频部分，如图1(a)所示。类似的，我们也认为，特征图也有一个子集可以捕捉空间上低频变化的部分，包含着空间冗余的信息。

To reduce such spatial redundancy, we introduce the octave feature representation that explicitly factorizes the feature map tensors into groups corresponding to low and high frequencies. The scale-space theory [29] provides us with a principled way of creating scale-spaces of spatial resolutions, and defines an octave as a division of the spatial dimensions by a power of 2 (we only explore $2^1$ in this work). We define the low- and high-frequency spaces in this fashion, i.e. by reducing the spatial resolution of the low-frequency feature maps by an octave.

为减少这种空间冗余性，我们提出了octave特征表示，显式的将特征图张量分解为两组，对应低频部分和高频部分。尺度空间理论[29]给了我们为尺度空间指定分辨率的原则方法，定义了一个octave为空间维度除以2的整数次幂（我们在本文中只研究$2^1$）。我们以如下方式定义低频和高频空间，即将低频特征图的空间分辨率降低一个octave。

Formally, let $X ∈ R^{c×h×w}$ denote the input feature tensor of a convolutional layer, where h and w denote the spatial dimensions and c the number of feature maps or channels. We explicitly factorize X along the channel dimension into X = {$X^H, X^L$}, where the high-frequency feature maps $X^H ∈ R^{(1−α)c×h×w}$ capture fine details and the low-frequency maps $X^L ∈ R^{αc× h/2 × w/2}$ vary slower in the spatial dimensions (w.r.t. the image locations). Here α ∈ [0, 1] denotes the ratio of channels allocated to the low-frequency part and the low-frequency feature maps are defined an octave lower than the high frequency ones, i.e. at half of the spatial resolution as shown in Figure 1(c).

正式的，令$X ∈ R^{c×h×w}$表示一个卷积层的输入特征张量，其中h和w表示空间维度，c是特征图或通道的数量。我们显式的将X沿着通道维度分解为X = {$X^H, X^L$}，其中高频特征图$X^H ∈ R^{(1−α)c×h×w}$，捕捉的精细细节，低频特征图$X^L ∈ R^{αc× h/2 × w/2}$在空间维度上变化更缓慢（相对于图像位置）。这里α ∈ [0, 1]，表示分配给低频部分的通道比例，低频特征图定义为比高频的部分低一个octave，即，空间分辨率的一半，如图1(c)所示。

In the next subsection, we introduce a convolution operator that operates directly on this multi-frequency feature representation and name it Octave Convolution, or OctConv for short. 在下一小节中，我们介绍直接在这种多频率特征表示上进行的卷积运算，并将其命名为Octave Convolution，简称为OctConv。

### 3.2. Octave Convolution

The octave feature representation presented in Section 3.1 reduces the spatial redundancy and is more compact than the original representation. However, the vanilla convolution cannot directly operate on such a representation, due to differences in spatial resolution in the input features. A naive way of circumventing this is to up-sample the low-frequency part $X^L$ to the original spatial resolution, concatenate it with $X^H$ and then convolve, which would lead to extra costs in computation and memory and diminish all the savings from the compression. In order to fully exploit our compact multi-frequency feature representation, we introduce the Octave Convolution that can directly operate on factorized tensors X = {$X^H, X^L$} without requiring any extra computational or memory overhead.

3.1节提出的octave特征表示降低了空间冗余性，比原始表示更加紧凑。但是，传统卷积不能直接在这样一种表示上进行，因为输入特征在空间分辨率上不同。一种简单的方法是将低频部分$X^L$进行上采样，成为原始空间分辨率，然后与高频部分$X^H$拼接，然后卷积，这会带来额外的计算和存储代价，使压缩带来的各种节省都消失掉。为完全利用我们的紧凑多频率特征表示，我们提出了Octave Convolution，可以直接在分解的张量X = {$X^H, X^L$}上进行运算，需要额外的计算或内存消耗。

**Vanilla Convolution**. Let $W ∈ R^{c×k×k}$ denote a k × k convolution kernel and $X, Y ∈ R^{c×h×w}$ denote the input and output tensors, respectively. Each feature map in $Y_{p,q} ∈ R^c$ can be computed by

**常规卷积**。令$W ∈ R^{c×k×k}$表示k × k的卷积核，$X, Y ∈ R^{c×h×w}$分别表示输入和输出张量。特征图在每个空间上的$Y_{p,q} ∈ R^c$可以计算为：

$$Y_{p,q} = \sum_{i,j∈N_k} W_{i+ \frac{k−1}{2}, j+ \frac{k−1}{2}} ⊤ X_{p+i,q+j}$$(1)

where (p, q) denotes the location coordinate and $N_k = \{(i, j) : i = \{−\frac{k−1}{2}, . . . , \frac{k−1}{2} \}, j = \{−\frac{k−1}{2}, . . . , \frac{k−1}{2} \}\}$ defines a local neighborhood. For simplicity, in all equations we omit the padding, we assume k is an odd number and that the input and output data have the same dimensionality, i.e. $c_{in} = c_{out} = c$.

其中(p,q)表示位置坐标，$N_k = \{(i, j) : i = \{−\frac{k−1}{2}, . . . , \frac{k−1}{2} \}, j = \{−\frac{k−1}{2}, . . . , \frac{k−1}{2} \}\}$定义了一个局部邻域。为简化起见，在所有的等式中，我们都忽略了补齐，我们假设k是一个奇数，输入和输出数据维度相同，即$c_{in} = c_{out} = c$。

**Octave Convolution**. The goal of our design is to effectively process the low and high frequency in their corresponding frequency tensor but also enable efficient communication between the high and low frequency component of our Octave feature representation. Let X, Y be the factorized input and output tensors. Then the high- and low-frequency feature maps of the output Y = {$Y^H, Y^L$} will be given by $Y^H = Y^{H→H} + Y^{L→H}$ and $Y^L = Y^{L→L} + Y^{H→L}$, respectively, where $Y^{A→B}$ denotes the convolutional update from feature map group A to group B. Specifically, $Y^{H→H}$, $Y^{L→L}$ denote intra-frequency information update, while $Y^{H→L}$, $Y^{L→H}$ denote inter-frequency communication.

**Octave Convolution**。我们设计的目标是有效的处理频率张量中的高频和低频分量，同时还能够在octave特征表示中高频和低频部分进行有效的沟通。令X，Y为分解的输入和输出张量，然后输出高频和低频特征图Y = {$Y^H, Y^L$}，分别由$Y^H = Y^{H→H} + Y^{L→H}$和$Y^L = Y^{L→L} + Y^{H→L}$给出，其中$Y^{A→B}$表示组A的特征图到组B特征图的卷积更新。特别的，$Y^{H→H}$, $Y^{L→L}$表示频率内信息更新，而$Y^{H→L}$, $Y^{L→H}$表示频率间通信。

To compute these terms, we split the convolutional kernel W into two components W = [$W^H, W^L$] responsible for convolving with $X^H$ and $X^L$ respectively. Each component can be further divided into intra- and inter-frequency part: $W^H = [W^{H→H}, W^{L→H}]$ and $W^L = [W^{L→L}, W^{H→L}]$ with the parameter tensor shape shown in Figure 2(b). Specifically for high-frequency feature map, we compute it at location (p, q) by using a regular convolution for the intra-frequency update, and for the inter-frequency communication we can fold the up-sampling over the feature tensor $X^L$ into the convolution, removing the need of explicitly computing and storing the up-sampled feature maps as follows:

为计算这些项，我们将卷积核W分成两项W = [$W^H, W^L$]，分别负责与$X^H$和$X^L$卷积。每个项可以进一步分为频率内部分和频率间部分：$W^H = [W^{H→H}, W^{L→H}]$，和$W^L = [W^{L→L}, W^{H→L}]$，参数张量形状如图2(b)所示。特别的，对于特征图的高频部分，我们在位置(p, q)计算如下，在频率内更新部分使用的是常规卷积，在频率间通信部分，我们可以将特征张量$X^L$的上采样叠加到卷积上，不需要显式的计算并存储上采样的特征图，如下所示：

$$Y^H_{p,q} = Y^{H→H}_{p,q} + Y^{L→H}_{p,q} = \sum_{i,j∈N_k} W^{H→H}_{i+\frac{k-1}{2}, j+\frac{k-1}{2}} ⊤ X^H_{p+i,q+j}$$
$$+ \sum_{i,j∈N_k} W^{L→H}_{i+\frac{k-1}{2}, j+\frac{k-1}{2}} ⊤ X^L_{(⌊\frac{p}{2}⌋+i), (⌊\frac{q}{2}⌋+j)}$$(2)

where ⌊·⌋ denotes the floor operation. Similarly, for the low-frequency feature map, we compute the intra-frequency update using a regular convolution. Note that, as the map is in one octave lower, the convolution is also low-frequency w.r.t. the high-frequency coordinate space. For the inter-frequency communication we can again fold the down-sampling of the feature tensor $X^H$ into the convolution as follows:

其中⌊·⌋表示floor算子。类似的，对于低频特征图，我们用常规卷积计算频率内更新。注意，由于特征图低了一个octave，所以相对于高频坐标空间来说，卷积也是低频的。对于频率间的通信，我们可以再将特征张量$X^H$的下采样部分叠加到卷积上，如下所示：

$$Y^L_{p,q} = Y^{L→L}_{p,q} + Y^{H→L}_{p,q} = \sum_{i,j∈N_k} W^{L→L}_{i+\frac{k-1}{2}, j+\frac{k-1}{2}} ⊤ X^L_{p+i,q+j}$$
$$+ \sum_{i,j∈N_k} W^{H→L}_{i+\frac{k-1}{2}, j+\frac{k-1}{2}} ⊤ X^H_{(2∗p+0.5+i),(2∗q+0.5+j)}$$(3)

where multiplying a factor 2 to the locations (p, q) performs down-sampling, and further shifting the location by half step is to ensure the down-sampled maps well aligned with the input. However, since the index of $X^H$ can only be an integer, we could either round the index to (2∗p+i, 2∗q+j) or approximate the value at (2∗p+0.5+i, 2∗q+0.5+j) by averaging all 4 adjacent locations. The first one is also known as strided convolution and the second one as average pooling. As we discuss in Section 3.3 and Fig. 3, strided convolution leads to misalignment; we therefore use average pooling to approximate this value for the rest of the paper.

其中将位置(p, q)乘以2进行的是下采样，而且将位置偏移一半步长，是确保下采样的特征图与输入对齐。但是，由于$X^H$的索引只能是整数，我们要么把索引四舍五入到(2∗p+i, 2∗q+j)，要么通过平均4邻域的位置近似为(2∗p+0.5+i, 2∗q+0.5+j)。第一种就是有步长的卷积，而第二种就是平均池化。就像我们在3.3节和图3中讨论的，有步长的卷积带来的没有对齐；所以在本文下面中我们使用平均池化来近似这个值。

An interesting and useful property of the Octave Convolution is the larger receptive field for the low-frequency feature maps. Convolving the low-frequency part $X^L$ with k × k convolution kernels, results in an effective enlargement of the receptive field by a factor of 2 compared to vanilla convolutions. This further helps each OctConv layer capture more contextual information from distant locations and can potentially improve recognition performance.

Octave Convolution的一个有趣而有用的性质是，对于低频特征图，有着更大的感受野。将低频部分$X^L$与k × k大小的卷积核进行卷积，得到的有效感受野大小，与传统卷积相比，实际上乘以了2。这进一步帮助每个Octave层使用更多的更远位置的上下文信息，可能改进识别性能。

Figure 2: Octave Convolution. We set $α_{in} = α_{out} = α$ throughout the network, apart from the first and last OctConv of the network where $α_{in}$ = 0, $α_{out}$ = α and $α_{in}$ = α, $α_{out}$ = 0, respectively.

(a) Detailed design of the Octave Convolution. Green arrows correspond to information updates while red arrows facilitate information exchange between the two frequencies.

(b) The Octave Convolution kernel. The k × k Octave Convolution kernel $W ∈ R^{c_{in}×c_{out}×k×k}$ is equivalent to the vanilla convolution kernel in the sense that the two have the exact same number of parameters.

### 3.3. Implementation Details

As discussed in the previous subsection, the index {(2∗p+0.5+i), (2∗q+0.5+j)} has to be an integer for Eq. 3. Instead of rounding it to {(2∗p+i), (2∗q+j)}, i.e. conduct convolution with stride 2 for down-sampling, we adopt average pooling to get more accurate approximation. This helps alleviate misalignments that appear when aggregating information from different scales [11], as shown in Figure 3 and validated in Table 3. We can now rewrite the output Y = {$Y^H, Y^L$} of the Octave Convolution using average pooling for down-sampling as:

就像前面小节讨论的那样，索引{(2∗p+0.5+i), (2∗q+0.5+j)}对于式(3)来说必须是一个整数。我们没有将其近似为{(2∗p+i), (2∗q+j)}，即进行步长为2的卷积下采样，而是采取了平均池化的方法，得到更精确的近似。这帮助缓解了没有对齐的问题，这个问题在聚集不同尺度的信息时会出现[11]，如图3和表3所示。我们可以将Octave Convolution的输出Y = {$Y^H, Y^L$}用平均池化的下采样重写为：

$$Y^H =f (X^H; W^{H→H}) + upsample(f (X^L; W^{L→H}), 2)$$
$$Y^L =f (X^L; W^{L→L}) + f (pool(X^H, 2); W^{H→L}))$$(4)

where f(X; W) denotes a convolution with parameters W, pool(X, k) is an average pooling operation with kernel size k × k and stride k. upsample(X, k) is an up-sampling operation by a factor of k via nearest interpolation. 其中f(X; W)表示与参数W进行卷积，pool(X, k)是平均池化运算，核心大小为k × k，步长为k。upsample(X, k)是一个最近邻插值的k因子上采样运算。

Figure 3: Strided convolution causes misaligned feature maps after up-sampling. As the example shows, up-sampling after the strided convolution will cause the entire feature map to move to the lower right, which is problematic when we add the shifted map with the unshifted map.

图3. 有步长的卷积在上采样后导致特征图对不齐。如图中例子所示，有步长的卷积后进行上采样，会导致整个特征图向右下移动，这在将偏移特征图与未偏移特征图相加时会导致问题。

The details of the OctConv operator implementation are shown in Figure 2. It consists of four computation paths that correspond to the four terms in Eq. (4): two green paths correspond to information updating for the high- and low-frequency feature maps, and two red paths facilitate information exchange between the two octaves.

OctConv算子的实现细节，如图2所示，包括4种计算路径，对应着式(4)中四项：两条绿色路径对应着高频和低频特征图的信息更新，两条红色路径则对应着两个octave的信息交换。

**Group and Depth-wise convolutions**. The Octave Convolution can also be adopted to other popular variants of the vanilla convolution such as group [45] or depth-wise [20] convolutions. For the group convolution case, we simply set all four convolution operations that appear inside the design of the OctConv to group convolutions. Similarly, for the depth-wise convolution case, the convolution operations are depth-wise and therefore the information exchange paths are eliminated, leaving only two depth-wise convolution operations. We note that both group OctConv and depth-wise OctConv reduce to their respective vanilla versions if we do not compress the low-frequency part.

**分组卷积和depth-wise卷积**。Octave卷积也可以用于传统卷积的流行变体，如分组卷积或[45]depth-wise卷积[20]。我们只需要将OctConv设计中的所有四种卷积操作设为分组卷积。类似的，对于depth-wise卷积的情况，卷积操作是depth-wise的，所以信息交换通道就不需要了，只有两个depth-wise卷积操作。我们注意到，分组OctConv和depth-wise OctConv在不压缩低频部分的时候，会退化到其相应的传统卷积版本。

**Efficiency analysis**. Table 1 shows the theoretical computational cost and memory consumption of OctConv over the vanilla convolution and vanilla feature map representation. More information on deriving the theoretical gains presented in Table 1 can be found in the supplementary material. We note the theoretical gains are calculated per convolutional layer. In Section 4 we present the corresponding practical gains on real scenarios and show that our OctConv implementation can sufficiently approximate the theoretical numbers.

**效率分析**。表1给出了OctConv和传统卷积及传统特征图表示的理论计算量和内存消耗对比。补充材料中给出了怎样从理论上推导出表1的结果的。我们要说明的是，理论收益是在每个卷积层上计算得到的。在第4部分，我们给出在实际场景中的相应实际收益，表明我们的OctConv实现可以充分的近似理论数字。

Table 1: Relative theoretical gains for the proposed multi-frequency feature representation over vanilla feature maps for varying choices of the ratio α of channels used by the low-frequency feature. When α = 0, no low-frequency feature is used which is the case of vanilla convolution. Note the number of parameters in OctConv operator is constant regardless of the choice of ratio.

ratio (α) | .0 | .125 | .25 | .50  | .75 | .875 | 1.0
--- | --- | --- | --- | --- | --- | --- | ---
FLOPs Cost | 100% | 82% | 67% | 44% | 30% | 26% | 25%
Memory Cost | 100% | 91% | 81% | 63% | 44% | 35% | 25%

**Integrating OctConv into backbone networks**. OctConv is backwards compatible with vanilla convolution and can be inserted to regular convolution networks without special adjustment. To convert a vanilla feature representation to a multi-frequency feature representation, i.e. at the first OctConv layer, we set $α_{in}$ = 0 and $α_{out}$ = α. In this case, OctConv paths related to the low-frequency input is disabled, resulting in a simplified version which only has two paths. To convert the multi-frequency feature representation back to vanilla feature representation, i.e. at the last OctConv layer, we set $α_{out}$ = 0. In this case, OctConv paths related to the low-frequency output is disabled, resulting in a single full resolution output.

**将OctConv整合到骨干网络中**。OctConv与传统卷积是反向兼容的，不需要特殊的调整就可以插入到常规卷积网络中。为将传统特征表示转换到多频率特征表示上，即在第一个OctConv层上，我们设$α_{in}$ = 0，$α_{out}$ = α。在这种情况下，与低频输入相关的OctConv路径是不可用的，得到的简化版只有两条路径。为将多频率特征表示转换为常规特征表示，即在最后一个OctConv层，我们设置$α_{out}$ = 0。在这种情况下，与低频输出相关的OctConv路径是不可用的，得到了单个完整分辨率的输出。

## 4. Experimental Evaluation 试验评估

In this section, we validate the effectiveness and efficiency of the proposed Octave Convolution for both 2D and 3D networks. We first present ablation studies for image classification on ImageNet [13] and then compare it with the state-of-the-art. Then, we show the proposed OctConv also works in 3D CNNs using Kinetics-400 [26, 3] and Kinetics-600 [2] datasets. The best results per category/block are highlighted in bold font throughout the paper.

在本节中，我们对2D网络和3D网络验证提出的OctConv卷积的有效性和效率。我们首先在ImageNet[13]上进行图像分类任务的分离研究，然后与目前最好的成绩进行比较。然后，我们证明了，提出的OctConv在3D CNNs中也可以应用，包括Kinetics-400 [26,3]和Kinetics-600[2]数据集。本文中，每一类/模块中最好的结果都用粗体进行高亮显示。

### 4.1. Experimental Setups

**Image classification**. We examine OctConv on a set of most popular CNNs [20, 35, 18, 19, 24, 45, 21] by replacing the regular convolutions with OctConv (except the first convolutional layer before the max pooling). The resulting networks only have one global hyper-parameter α, which denotes the ratio of low frequency part. We do apple-to-apple comparison and reproduce all baseline methods by ourselves under the same training/testing setting for internal ablation studies. All networks are trained with naı̈ve softmax cross entropy loss except that the MobileNetV2 also adopts the label smoothing [37], and the best ResNet-152 adopts both label smoothing and mixup [46] to prevent overfitting. Same as [4], all networks are trained from scratch and optimized by SGD with cosine learning rate [15]. Standard accuracy of single centeral crop [18, 19, 45, 4, 41] on validation set is reported.

**图像分类**。我们在最流行的CNN集合上[20,35,18,19,24,45,21]将常规卷积替换为OctConv（除了max pooling之前的第一个卷积层），以检验OctConv。得到的网络只有一个全局超参数α，代表低频部分的比率。我们逐个进行比较，并在相同的训练/测试设置中独立复现了基准方法，以进行内部的分离研究。所有网络都用简单的softmax交叉熵训练，除了MobileNetV2采用了标签平滑[37]，最好的ResNet-152采用了标签平滑和mixup[46]以防止过拟合。与[4]相同，所有网络都从头训练并用SGD和cosine学习速率[15]进行优化。给出的结果是在验证集上的单个中间剪切块的标准准确率[18,19,45,4,41]。

**Video action recognition**. We use both Kinetics-400 [26, 3] and Kinetics-600 [2] for human action recognition. We choose standard baseline backbones from Inflated 3D ConvNet [42] and compare them with the OctConv counterparts. We follow the setting from [43] using frame length of 8 as standard input size and training 300k iterations in total. To make fair comparison, we report the performance of the baseline and OctConv under precisely the same settings. For the inference time, we average the predictions over 30 crops ( each of (left, center, right) × 10 crops along temporal dimension), again following prior work [42].

**视频动作识别**。我们使用Kinetics-400[26,3]和Kinetics-600[2]进行人类动作识别。我们从Inflated 3D ConvNet[42]中选择标准基准的骨干网络，将其与对应的OctConv网络进行比较。我们使用[43]中的设置，使用帧长度8作为标准的输入大小，总共训练300k次迭代。为进行公平的比较，基准和OctConv的性能在相同的设置中给出结果。对于推理时间，我们对30个剪切块的预测进行平均（每个都是沿着时间维度的（左，中，右）×10剪切块），使用的是[42]中的方案。

### 4.2. Ablation Study on ImageNet

We conduct a series of ablation studies aiming to answer the following questions: 1) Does OctConv have better FLOPs-Accuracy trade-off than vanilla convolution? 2) In which situation does the OctConv work the best? 我们进行了一系列分离试验，以回答下列问题：1)OctConv比传统卷积是不是有更好的FLOPs-准确率折中？ 2)OctConv在哪种情况下表现更好？

**Results on ResNet-50**. We begin with using the popular ResNet-50 [19] as the baseline CNN and replacing the regular convolution with our proposed OctConv to examine the flops-accuracy trade-off. In particular, we vary the global ratio α ∈ {0.125, 0.25, 0.5, 0.75} to compare the image classification accuracy versus computational cost (i.e. FLOPs) [18, 19, 45, 8] with the baseline. The results are shown in Figure 4 in pink.

**在ResNet-50上的结果**。我们使用流行的ResNet-50[19]作为基准CNN，将常规的卷积替换为我们提出的OctConv，以检查flops-accuracy折中。特别的，我们使用不同的全局比率α ∈ {0.125, 0.25, 0.5, 0.75}，比较图像分类准确率与计算量(FLOPs)的对比[18,19,45,8]。结果如图4中的粉色所示。

We make following observations. 1) The flops-accuracy trade-off curve is a concave curve, where the accuracy first rises up and then slowly goes down. 2) We can see two sweet spots: The first at α = 0.5, where the network gets similar or better results even when the FLOPs are reduced by about half; the second at α = 0.125, where the network reaches its best accuracy, 1.2% higher than baseline (black circle). We attribute the increase in accuracy to OctConv’s effective design of multi-frequency processing and the corresponding enlarged receptive field which provides more contextual information to the network. While reaching the accuracy peak at 0.125, the accuracy does not suddenly drop but decreases slowly for higher ratios α, indicating reducing the resolution of the low frequency part does not lead to significant information loss. Interestingly, 75% of the feature maps can be compressed to half the resolution with only 0.4% accuracy drop, which demonstrates effectiveness of grouping and compressing the smoothly changed feature maps for reducing the spatial redundancy in CNNs. In Table 2 we demonstrate the theoretical FLOPs saving of OctConv is also reflected in the actual CPU inference time in practice. For ResNet-50, we are close to obtaining theoretical FLOPs speed up. These results indicate OctConv is able to deliver important practical benefits, rather than only saving FLOPs in theory.

我们得到以下几个观察结果。1)flops-accuracy折中曲线是一个凹曲线，准确率首先上升，然后缓慢下降；2)我们看到两个最佳点：第一个是α = 0.5，这里网络的性能与基准的性能相比类似或更好，但是运算量降低了大约一半；第二个是在α = 0.125的时候，网络得到最好的准确率结果，比基准（黑色圆圈）高了1.2%。我们认为这是因为OctConv的有效设计的原因，可以进行多频率处理，以及可以增大感受野，利用更多上下文信息。在0.125的时候达到了准确率的最高点，准确率并没有忽然下降，而是随着α的增大慢慢下降，表明降低低频部分的分辨率并没有带来明显的信息损失。有趣的是，75%的特征图可以压缩为一半的分辨率，而准确率只下降了0.4%，这说明分组和压缩平滑变化的特征图、降低CNN中的空间冗余性的有效性。在表2中，我们证明了理论上OctConv节省的FLOPs也反应在实际的CPU推理时间上。对于ResNet-50，我们接近于达到了理论上的FLOPs加速效果。这些结果表明，OctConv可以得到非常重要的实际收益，而不是只在理论中，实际没有验证。

Table 2: Results of ResNet-50. Inference time is measured on Intel Skylake CPU at 2.0 GHz (single thread). We report Intel(R) Math Kernel Library for Deep Neural Networks v0.18.1 (MKLDNN) [25] inference time for vanila ResNet-50. Because vanilla ResNet-50 is well optimized by Intel, we also show MKLDNN results as additional performance baseline. OctConv networks are compiled by TVM [5] v0.5.

ratio (α) | Top-1 (%) | #FLOPs (G) | Inference Time (ms) | Backend
--- | --- | --- | --- | ---
N/A | 77.0 | 4.1 | 119 | MKLDNN
N/A | 77.0 | 4.1 | 115 | TVM
.125 | 78.2 | 3.6 | 116 | TVM
.25 | 78.0 | 3.1 | 99 | TVM
.5 | 77.3 | 2.4 | 74 | TVM
.75 | 76.6 | 1.9 | 61 | TVM

**Results on more CNNs**. To further examine if the proposed OctConv works for other networks with different depth/wide/topology, we select the currently most popular networks as baselines and repeat the same ablation study. These networks are ResNet-(26;50;101;200) [19], ResNeXt-(50,32×4d;101,32×4d) [45], DenseNet-121 [24] and SE-ResNet-50 [21]. The ResNeXt is chosen for assessing the OctConv on group convolution, while the SENet [21] is used to check if the gain of SE block found on vanilla convolution based networks can also be seen on OctConv. As shown in Figure 4, OctConv equipped networks for different architecture behave similarly to the Oct-ResNet-50, where the FLOPs-Accuracy trade-off is in a concave curve and the performance peak also appears at ratio α = 0.125 or α = 0.25. The consistent performance gain on a variety of backbone CNNs confirms that OctConv is a good replacement of vanilla convolution.

**在更多CNNs上的结果**。为进一步检查提出的OctConv是否在其他不同的深度/宽度/拓扑网络上也可以工作，我们选择了目前最流行的网络作为基准，重复相同的分离研究。这些网络是ResNet-(26;50;101;200) [19], ResNeXt-(50,32×4d;101,32×4d) [45], DenseNet-121 [24] and SE-ResNet-50 [21]。选择ResNeXt是评估OctConv在分组卷积上表现，而SENet[21]则用于检查SE模块在传统卷积上的收益，是否在OctConv上也有类似的收益。如图4所示，使用了OctConv的不同架构的网络与Oct-ResNet-50的表现类似，其中FLOPs-Accuracy折中是一个凹曲线，性能顶点也是在α = 0.125或α = 0.25点上。在不同的CNNs骨架上的一致表现收益，确认了OctConv是传统卷积很好的替代品。

Besides, we also have some intriguing findings: 1) OctConv can help CNNs improve the accuracy while decreasing the FLOPs, deviating from other methods that reduce the FLOPs with a cost of lower accuracy. 2) At test time, the gain of OctConv over baseline models increases as the test image resolution grows because OctConv can detect large objects better due to its larger receptive field, as shown in Table 4. 3) Both the information exchanging paths are important, since removing any of them can lead to accuracy drop as shown in Table 3. 4) Shallow networks, e.g. ResNet-26, have a rather limited receptive field, and can especially benefit from OctConv, which greatly enlarges their receptive field.

另外，我们还有一些有趣的发现：1)OctConv可以帮助CNNs改进准确率，同时降低FLOPs，其他的方法则是降低FLOPs的同时降低准确率；2)在测试时，OctConv相对于基准模型的收益，在测试图像分辨率增加的时候也随之增加，因为OctConv因为有更大的感受野，对大目标的检测效果更好，如表4所示；3)两个信息交换的通道都很重要，因为去掉任何一个都会导致准确率下降，如表3所示；4)浅层网络如ResNet-26，其感受野很有限，从OctConv中受益非常多，因为极大增大了感受野。

Table 3: Ablation on down-sampling and inter-octave connectivity on ImageNet.

Method | Down-sampling | Low-High | High-Low | Top-1(%)
--- | --- | --- | --- | ---
Oct-ResNet-50 ratio:0.5 | strided conv. | y | y | 76.3
Oct-ResNet-50 ratio:0.5 | avg. pooling | n | n | 76.0
Oct-ResNet-50 ratio:0.5 | avg. pooling | y | n | 76.4
Oct-ResNet-50 ratio:0.5 | avg. pooling | n | y | 76.4
Oct-ResNet-50 ratio:0.5 | avg. pooling | y | y | 77.3

Table 4: ImageNet classification accuracy. The short length of input images are resized to the target crop size while keeping the aspect ratio unchanged. A centre crop is adopted if the input image size is not square. ResNet-50 backbone trained with crops size of 256 × 256 pixels.

ratio(α) | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 740
--- | --- | --- | --- | --- | --- | --- | --- | ---
N/A | 77.2 | 78.6 | 78.7 | 78.7 | 78.3 | 77.6 | 76.7 | 75.8
.5 | +0.7 | +0.7 | +0.9 | +0.9 | +0.8 | +1.0 | + 1.1 | +1.2

### 4.3. Comparing with SOTAs on ImageNet

**Small models**. We adopt the most popular light weight networks as baselines and examine if OctConv works well on these compact networks with depth-wise convolution. In particular, we use the “0.75 MobileNet (v1)” [20] and “1.0 MobileNet (v2)” [35] as baseline and replace the regular convolution with our proposed OctConv. The results are shown in Table 5. We find that OctConv can reduce the FLOPs of MobileNetV1 by 34%, and provide better accuracy and faster speed in practice; it is able to reduce the FLOPs of MobileNetV2 by 15%, achieving the same accuracy with faster speed. When the computation budget is fixed, one can adopt wider models to increase the learning capacity because OctConv can compensate the extra computation cost. In particular, our OctConv equipped networks achieve 2% improvement on MobileNetV1 under the same FLOPs and 1% improvement on MobileNetV2.

**小型模型**。我们使用最流行的轻量级网络作为基准，检查OctConv在这些紧凑网络中与depth-wise卷积一起是否可以很好的工作。特别的，我们使用“0.75 MobileNet (v1)” [20]和“1.0 MobileNet (v2)” [35]作为基准，将常规卷积替换为提出的OctConv。结果如表5所示。我们发现OctConv可以将MobileNetV1的FLOPs降低34%，在实际中给出更好的准确率、更快的速度；可以将MobileNetV2的FLOPs降低15%，得到同样的准确率，但速度更快。如果固定计算量，可以采用更宽的模型，以提升学习能力，因为OctConv可以补偿这些额外的计算代价。特别的，使用了OctConv的MobileNetV1在同样的FLOPs下有2%的改进，MobileNetV2则有1%的改进。

Table 5: ImageNet classification results for Small models. * indicates it is better than original reproduced by MXNet GluonCV v0.4. The inference speed is tested using TVM on Intel Skylake processor (2.0GHz, single thread).

Method | ratio (α) | #Params (M) | #FLOPs (M) | CPU (ms) | Top-1 (%)
--- | --- | --- | --- | --- | ---
0.75 MobileNet (v1) [20] | - | 2.6 | 325 | 13.4 | 70.3*
0.75 Oct-MobileNet (v1) (ours) | .375 | 2.6 | 213 | 11.9 | 70.6
1.0 Oct-MobileNet (v1) (ours) | .5 | 4.2 | 321 | 18.4 | 72.4
1.0 MobileNet (v2) [35] | - | 3.5 | 300 | 24.5 | 72.0
1.0 Oct-MobileNet (v2) (ours) | .375 | 3.5 | 256 | 17.1 | 72.0
1.125 Oct-MobileNet (v2) (ours) | .5 | 4.2 | 295 | 26.3 | 73.0

**Medium models**. In the above experiment, we have compared and shown that OctConv is complementary with a set of state-of-the-art CNNs [18, 19, 45, 24, 20, 35, 21]. In this part, we compare OctConv with Elastic [41] and bL-Net [4] which share a similar idea as our method. Five groups of results are shown in Table 6. In group 1, our Oct-ResNeXt-50 achieves better accuracy than the Elastic [41] based method (78.7% v.s. 78.4%) while reducing the computational cost by 31%. In group 2, the Oct-ResNeXt-101 also achieves higher accuracy than the Elastic based method (79.5% v.s. 79.2%) while costing 38% less computation. When compared to the bL-Net [4], OctConv equipped methods achieve better FLOPs-Accuracy trade-off without bells and tricks. When adopting the tricks used in the baseline bL-Net [4], our Oct-ResNet-50 achieves 0.8% higher accuracy than bL-ResNet-50 under the same computational budget (group 3), and Oct-ResNeXt-50 (group 4) and Oct-ResNeXt-101 (group 5) get better accuracy under comparable or even lower computational budget. This is because both the Elastic-Net [41] and bL-Net [4] are designed to exploit multi-scale features instead of reducing the spatial redundancy. In contrast, OctConv uses a more compact feature representation to store and process the information throughout the network, and can thus achieve better efficiency and performance.

**中型模型**。在上面的试验中，我们比较并表明了，OctConv是现有最好的CNNs的补充[18,19,45,24,20,35,21]。在这一部分，我们将OctConv与Elastic[41]及bL-Net[4]进行比较，因为这些方法的思想都是类似的。表6给出了5组结果。第1组中，我们的Oct-ResNeXt-50比基于Elastic [41]的方法得到了更好的结果(78.7% vs. 78.4%)，而计算量则降低了31%。在第2组中，Oct-ResNeXt-101也比基于Elastic的方法得到了更高的准确率(79.5% v.s. 79.2%)，计算量则降低了38%。与bL-Net[4]比较时，使用了OctConv的方法取得了更好的FLOPs-Accuracy折中。如果使用基准bL-Net[4]中的技巧，我们的Oct-ResNet-50比bL-ResNet-50取得了高了0.8%的准确率，计算量则相同（组3）。Oct-ResNeXt-50 (组4)和Oct-ResNeXt-101 (组5)在类似或更少的计算量下，得到了更好的准确率。这是因为Elastic-Net [41]和bL-Net [4]设计上就是利用多尺度特征的，而不是降低其空间冗余性的。比较起来，OctConv使用更紧凑的特征表示，以在网络中存储并处理信息，所以可以得到更好的准确率及性能。

Table 6: ImageNet Classification results for Middle sized models. ‡ refers to method that replaces “Max Pooling” by extra convolution layer(s) [4]. § refers to method that uses balanced residual block distribution [4].

Method | ratio (α) | Depth | #Params (M) | #FLOPs (G) | Top-1 (%)
--- | --- | --- | --- | --- | ---
ResNeXt-50 + Elastic [41] | - | 50 | 25.2 | 4.2 | 78.4
Oct-ResNeXt-50 (32×4d) (ours) | .25 | 50 | 25.0 | 3.2 | 78.7
ResNeXt-101 + Elastic [41] | - | 101 | 44.3 | 7.9 | 79.2
Oct-ResNeXt-101 (32×4d) (ours) | .25 | 101 | 44.2 | 5.7 | 79.5
bL-ResNet-50 ‡ (α = 4, β = 4) [4] | - | 50 (+3) | 26.2 | 2.5 | 76.9
Oct-ResNet-50 ‡ (ours) | .5 | 50 (+3) | 25.6 | 2.5 | 77.7
Oct-ResNet-50 (ours) | .5 | 50 | 25.6 | 2.4 | 77.3
bL-ResNeXt-50 ‡ (32×4d) [4] | - | 50 (+3) | 26.2 | 3.0 | 78.4
Oct-ResNeXt-50 ‡ (32×4d) (ours) | .5 | 50 (+3) | 25.1 | 2.7 | 78.6
Oct-ResNeXt-50 (32×4d) (ours) | .5 | 50 | 25.0 | 2.4 | 78.3
bL-ResNeXt-101 ‡ § (32×4d) [4] | - | 101 (+1) | 43.4 | 4.1 | 78.9
Oct-ResNeXt-101 ‡ § (32×4d) (ours) | .5 | 101 (+1) | 40.1 | 4.2 | 79.3
Oct-ResNeXt-101 ‡ (32×4d) (ours) | .5 | 101 (+1) | 44.2 | 4.2 | 79.1
Oct-ResNeXt-101 (32×4d) (ours) | .5 | 101 | 44.2 |  4.0 | 78.9

**Large models**. Table 7 shows the results of OctConv in large models. Here, we choose the ResNet-152 as the backbone CNN, replacing the first 7 × 7 convolution by three 3 × 3 convolution layers and removing the max pooling by a lightweight residual block [4]. We report results for Oct-ResNet-152 with and without the SE-block [21]. As can be seen, our Oct-ResNet-152 achieves accuracy comparable to the best manually designed networks with less FLOPs (10.9G v.s. 12.7G). Since our model does not use group or depth-wise convolutions, it also requires significantly less GPU memory, and runs faster in practice compared to the SE-ShuffleNet v2-164 and AmoebaNet-A (N=6, F=190) which have low FLOPs in theory but run slow in practice due to the use of group and depth-wise convolutions. Our proposed method is also complementary to Squeeze-and-excitation [21], where the accuracy can be further boosted when the SE-Block is added (last row).

**大型模型**。表7所示的是OctConv在大型模型上的结果。这里，我们选择ResNet-152作为骨干CNN，将第一个7×7卷积替换为3个3×3卷积层，去掉了max pooling层成为了轻量的残差模块[4]。我们给出了带有SE模块和不含有SE模块[21]的Oct-ResNet-152。可以看出，我们的Oct-ResNet-152得到的准确率可以与最好的手工设计的网络类似，而且计算量FLOPs更少(10.9G v.s. 17.2G)。由于我们的模型没有使用分组卷积或depth-wise卷积，所以也需要少的多的GPU内存，与SE-ShuffleNet v2-164和AmoebaNet-A (N=6, F=190)相比，在实际中运行速度更快，而它们虽然FLOPs理论上很低，但在实际中运行更慢，因为使用了分组卷积或depth-wise卷积。我们提出的方法也是squeeze-and-excitation[21]的补充，如果增加了SE模块，其准确率可以进一步提升。

Table 7: ImageNet Classification results for Large models. The names of OctConv-equiped models are in bold font and performance numbers for related works are copied from the corresponding papers. Networks are evaluated using CuDNN v10.0 in flop16 on a single Nvidia Titan V100 (32GB) for their training memory cost and speed. Works that employ neural architecture search are denoted by ( 3 ). We set batch size to 128 in most cases, but had to adjust it to 64 (noted by † ), 32 (noted by ‡ ) or 8 (noted by § ) for networks that are too large to fit into GPU memory.

### 4.4. Experiments of Video Recognition on Kinetics

In this subsection, we evaluate the effectiveness of OctConv for action recognition in videos and demonstrate that our spatial OctConv is sufficiently generic to be integrated into 3D convolution to decrease #FLOPs and increase accuracy at the same time. As shown in Table 8, OctConv consistently decreases FLOPs and meanwhile improves the accuracy when added to C2D and I3D [42, 43], and is also complimentary to the Nonlocal building block [42]. This is observed for models pre-trained on ImageNet [13] as well as models trained from scratch on Kinetics.

在这个小节中，我们对视频中的动作识别任务评估OctConv的有效性，证明了我们的空域OctConv足够通用，可以整合进3D卷积，以降低FLOPs并同时提升准确率。如表8所示，OctConv整合进C2D和I3D[42,43]时，一直降低FLOPs，同时改进了准确率，也是Nonlocal模块[42]的补充。在ImageNet[13]上预训练的模型上和从头在Kinetcis训练的模型上都观察到了这个结果。

Specifically, we first investigate the behavior of training OctConv equipped I3D models from scratch on Kinetics. We use a learning rate 10× larger than the standard and train it 16 times longer than finetuning setting for a better convergence. Compared to the vanilla I3D model, Oct-I3D achieves 1.0% higher accuracy with 91% of the FLOPs.

特别的，我们首先研究了从头在Kinetics上训练使用了OctConv的I3D模型。与精调的设置相比，我们使用学习率比标准的大10倍，训练的时长长了16倍，收敛的也更好。与传统I3D模型相比，Oct-I3D的计算量只有91%，准确率高了1.0%。

We then explore the behavior of finetuning a OctConv on ImageNet pre-trained model with step-wise learning schedule. For this, we train an OctConv ResNet-50 model [18] on ImageNet [13] and then inflate it into a network with 3D convolutions [39] (over space and time) using the I3D technique [3]. After the inflation, we finetune the inflated OctConv following the schedule described in [43] on Kinetics-400. Compared to the 71.9% Top-1 accuracy of the C2D baseline on the Kinetics-400 validation set, the OctConv counterpart achieves 73.8% accuracy, using 90% of the FLOPs. For I3D, adding OctConv improves accuracy from 73.3% to 74.6% accuracy, while using only 91% of the FLOPs. We also demonstrate that the gap is consistent when adding Non-local [42]. Finally, we repeat the I3D experiment on Kinetics-600 [2] dataset and have a consistent finding, which further confirms the effectiveness of our method.

我们然后研究了，在ImageNet预训练的模型上精调OctConv，学习速率为分步方案。我们在ImageNet[13]上训练了一个OctConv ResNet-50模型[18]，然后使用I3D技术[3]将其膨胀为3D卷积[39]（时间和空间）。这次膨胀后，我们使用[43]中描述的方案在Kinetics-400中精调膨胀的OctConv。基准模型C2D在Kinetics-400验证集上的Top-1准确率为71.9%，而对应的OctConv版则取得了73.8%准确率，计算量FLOPs则为90%。对于I3D，使用OctConv将准确率从73.3%提升到了74.6%，而只使用了91%的运算量。我们还证明了，增加Non-local[42]之后，这个变化还是这样。最后，我们还在Kinetics-600[2]上重复了I3D试验，观察到了一致的现象，进一步确认了我们方法的有效性。

Table 8: Action Recognition in videos, ablation study, all models with ResNet50 [18].

Method | ImageNet Pretrain | FLOPs | Top-1(%)
--- | --- | --- | ---
(a) Kinetics-400 [3] |
I3D | n | 28.1 | 72.6
Oct-I3D, α=0.1, (ours) | n | 25.6 | 73.6(+1.0)
Oct-I3D, α=0.2, (ours) | n | 22.1 | 73.1(+0.5)
Oct-I3D, α=0.3, (ours) | n | 15.3 | 72.1(-0.5)
C2D | y | 19.3 | 71.9
Oct-C2D, α=0.1, (ours) | y | 17.4 | 73.8(+1.9)
I3D | y | 28.1 | 73.3
Oct-I3D, α=0.1, (ours) | y | 25.6 | 74.6(+1.3)
I3D+Non-local | y | 33.3 | 74.7
Oct-I3D+Non-local, α=0.1, (ours) | y | 28.9 | 75.7(+1.0)
(b) Kinetics-600 [2] |
I3D | y | 28.1 | 74.3
Oct-I3D, α=0.1, (ours) | y | 25.6 | 76.0(+1.7)

## 5. Conclusion

In this work, we address the problem of reducing spatial redundancy that widely exists in vanilla CNN models, and propose a novel Octave Convolution operation to store and process low- and high-frequency features separately to improve the model efficiency. Octave Convolution is sufficiently generic to replace the regular convolution operation in-place, and can be used in most 2D and 3D CNNs without model architecture adjustment. Beyond saving a substantial amount of computation and memory, Octave Convolution can also improve the recognition performance by effective communication between the low- and high-frequency and by enlarging the receptive field size which contributes to capturing more global information. Our extensive experiments on image classification and video action recognition confirm the superiority of our method for striking a much better trade-off between recognition performance and model efficiency, not only in FLOPs, but also in practice.

在本文中，我们处理的是降低空间冗余性的问题，这在传统CNN模型中广泛存在；我们提出一种新的Octave Convolution运算，分别存储并处理低频和高频特征，以改进模型效率。Octave Convolution非常通用，可以将常规卷积运算直接替换，可以在多数2D和3D CNNs中使用，不用调整模型架构。除了节省了相当的计算量和内存，Octave Convolution也可以改进识别性能，因为在低频和高频分量中可以有效的通信，并增大了感受野，可以捕捉到更多全局信息。我们进行了广泛的试验，包括图像分类和视频行为识别，确认了我们的方法的优势性能，得到了更好的识别准确率和计算效率折中，不止在理论上，实际计算结果也是。