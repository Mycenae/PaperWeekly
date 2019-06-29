# Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation

Chenxi Liu, Liang-Chieh Chen, Li Fei-Fei et al. Johns Hopkins University, Google, Stanford University

## Abstract 摘要

Recently, Neural Architecture Search (NAS) has successfully identified neural network architectures that exceed human designed ones on large-scale image classification. In this paper, we study NAS for semantic image segmentation. Existing works often focus on searching the repeatable cell structure, while hand-designing the outer network structure that controls the spatial resolution changes. This choice simplifies the search space, but becomes increasingly problematic for dense image prediction which exhibits a lot more network level architectural variations. Therefore, we propose to search the network level structure in addition to the cell level structure, which forms a hierarchical architecture search space. We present a network level search space that includes many popular designs, and develop a formulation that allows efficient gradient-based architecture search (3 P100 GPU days on Cityscapes images). We demonstrate the effectiveness of the proposed method on the challenging Cityscapes, PASCAL VOC 2012, and ADE20K datasets. Auto-DeepLab, our architecture searched specifically for semantic image segmentation, attains state-of-the-art performance without any ImageNet pretraining.

最近，在大规模图像分类任务中，NAS已经成功的搜索到了超过人类设计性能的网络架构。本文中，我们研究NAS在语义图像分割中的应用。现有的工作经常关注搜索重复的单元结构，而手工设计的外部网络结构则控制着空间分辨率变化。这个选项简化了搜索空间，但对于密集图像预测问题，却是问题很多，密集预测中有更多的网络层的架构变化。因此，我们提出除了搜索单元层的结构，还要搜索网络层的结构，这构成了层次化的架构搜索空间。我们提出了一个网络层的搜索空间，包括了很多流行的设计，并发展出了一个公式，可以进行高效的基于梯度的架构搜索（在Cityscapes图像中花费 3 P100 GPU天）。我们在Cityscapes，PASCAL VOC 2012，和ADE20K数据集上证明了提出方法的有效性。我们的架构搜索方法名为Auto-DeepLab，只为语义分割应用搜索，在没有ImageNet预训练的情况下得到了目前最好的性能。

## 1. Introduction 引言

Deep neural networks have been proved successful across a large variety of artificial intelligence tasks, including image recognition [38, 25], speech recognition [27], machine translation [73, 81] etc. While better optimizers [36] and better normalization techniques [32, 80] certainly played an important role, a lot of the progress comes from the design of neural network architectures. In computer vision, this holds true for both image classification [38, 72, 75, 76, 74, 25, 85, 31, 30] and dense image prediction [16, 51, 7, 64, 56, 55].

DNN在很多人工智能任务中都被证明有用，包括图像识别、语音识别、机器翻译等。更好的优化器和更好的归一化技术起了很重要的作用，还有很多进展是神经网络架构设计方面的。在计算机视觉中，对于图像分类和密集图像预测，这都是正确的。

More recently, in the spirit of AutoML and democratizing AI, there has been significant interest in designing neural network architectures automatically, instead of relying heavily on expert experience and knowledge. Importantly, in the past year, Neural Architecture Search (NAS) has successfully identified architectures that exceed human-designed architectures on large-scale image classification problems [93, 47, 62].

最近，在自动设计神经网络架构上有很多兴趣，而不是严重依赖专家经验和知识。重要的是，在去年，NAS发现的架构，在大规模图像分类问题上，已经超过了人类设计的架构的性能。

Image classification is a good starting point for NAS, because it is the most fundamental and well-studied high-level recognition task. In addition, there exists benchmark datasets (e.g., CIFAR-10) with relatively small images, resulting in less computation and faster training. However, image classification should not be the end point for NAS, and the current success shows promise to extend into more demanding domains. In this paper, we study Neural Architecture Search for semantic image segmentation, an important computer vision task that assigns a label like “person” or “bicycle” to each pixel in the input image.

图像分类对NAS来说是很好的起点，因为这是最基础和研究最多的高层识别任务。另外，有相对较小图像的基准测试数据集（如，CIFAR-10），计算量较小，训练更快。但是，图像分类不应当是NAS的终点，目前的成功说明，拓展到更多领域是很有希望的。本文中，我们研究语义图像分割的NAS，这是一个很重要的计算机视觉任务，是为输入图像中的每个像素指定一个标签，如人或自行车。

Naively porting ideas from image classification would not suffice for semantic segmentation. In image classification, NAS typically applies transfer learning from low resolution images to high resolution images [93], whereas optimal architectures for semantic segmentation must inherently operate on high resolution imagery. This suggests the need for: (1) a more relaxed and general search space to capture the architectural variations brought by the higher resolution, and (2) a more efficient architecture search technique as higher resolution requires heavier computation.

简单的从图像分类中将NAS的思想移植到语义分割中是不够的。在图像分类中，NAS一般使用迁移学习，从低分辨率图像迁移到高分辨率图像，而语义分割的最佳架构必须就在高分辨率图像上运算。这说明需要以下两点：(1)更一般更松弛的搜索空间，以捕获高分辨率带来的架构变化；(2)更高效的架构搜索技巧，因为更高的分辨率需要更多的计算量。

We notice that modern CNN designs [25, 85, 31] usually follow a two-level hierarchy, where the outer network level controls the spatial resolution changes, and the inner cell level governs the specific layer-wise computations. The vast majority of current works on NAS [93, 47, 62, 59, 49] follow this two-level hierarchical design, but only automatically search the inner cell level while hand-designing the outer network level. This limited search space becomes problematic for dense image prediction, which is sensitive to the spatial resolution changes. Therefore in our work, we propose a trellis-like network level search space that augments the commonly-used cell level search space first proposed in [93] to form a hierarchical architecture search space. Our goal is to jointly learn a good combination of repeatable cell structure and network structure specifically for semantic image segmentation.

我们注意到，现代CNN设计通常都是两级的层次，其中外围网络层控制着空间分辨率变化，内部单元层控制着特定的逐层计算。目前NAS工作的大部分都采用了这种两级的层次设计，但只是自动搜索内部单元层，同时手工设计外围网络层。这种有限的搜索空间对于密集图像预测来说是有很大问题的，因为对空间分辨率变化非常敏感。所以在我们的工作中，我们提出了一种trellis-like网络级搜索空间，扩展了通常使用的单元级搜索空间（在[93]中首次提出），以形成一个层次化的架构搜索空间。我们的目标是同时学习重复单元结构和网络结构的很好组合，专门进行语义图像分割。

In terms of the architecture search method, reinforcement learning [92, 93] and evolutionary algorithms [63, 62] tend to be computationally intensive even on the low resolution CIFAR-10 dataset, therefore probably not suitable for semantic image segmentation. We draw inspiration from the differentiable formulation of NAS [69, 49], and develop a continuous relaxation of the discrete architectures that exactly matches the hierarchical architecture search space. The hierarchical architecture search is conducted via stochastic gradient descent. When the search terminates, the best cell architecture is decoded greedily, and the best network architecture is decoded efficiently using the Viterbi algorithm. We directly search architecture on 321×321 image crops from Cityscapes [13]. The search is very efficient and only takes about 3 days on one P100 GPU.

在架构搜索方法上，强化学习和演化算法即使在低分辨率的CIFAR-10数据集上，计算量也非常大，所以可能不太适合语义图像分割。我们从NAS的可微分形式中得到灵感，提出了严格满足层次化架构搜索空间的离散架构的连续松弛条件。层次化架构搜索是通过随机梯度下降进行的。当搜索停止时，最好的单元架构是贪婪编码的，最好的网络架构是使用Viterbi算法有效的编码的。我们直接从Cityscapes上的321×321图像剪切块上搜索架构。搜索效率非常高，在一个P100 GPU上只用了三天。

We report experimental results on multiple semantic segmentation benchmarks, including Cityscapes [13], PASCAL VOC 2012 [15], and ADE20K [90]. Without ImageNet [65] pretraining, our best model significantly outperforms FRRN-B [60] by 8.6% and GridNet [17] by 10.9% on Cityscapes test set, and performs comparably with other ImageNet-pretrained state-of-the-art models [82, 88, 4, 11, 6] when also exploiting the coarse annotations on Cityscapes. Notably, our best model (without pretraining) attains the same performance as DeepLabv3+ [11] (with pretraining) while being 2.23 times faster in MultiAdds. Additionally, our light-weight model attains the performance only 1.2% lower than DeepLabv3+ [11], while requiring 76.7% fewer parameters and being 4.65 times faster in Multi-Adds. Finally, on PASCAL VOC 2012 and ADE20K, our best model outperforms several state-of-the-art models [90, 44, 82, 88, 83] while using strictly less data for pretraining.

我们在多个语义分割基准测试上给出了试验结果，包括Cityscapes，PASCAL VOC 2012和ADE20K。在没有ImageNet预训练的情况下，我们的最佳模型在Cityscapes测试集上显著超过了FRRN-B[60]达8.6%，超过了GridNet[17]达10.9%，与其他在ImageNet预训练的目前最好模型[82,88,4,11,6]表现类似，还利用了Cityscapes上的粗糙标注。尤其是，我们的最好模型（在没有预训练的情况下）得到了DeepLabV3+在预训练下的性能，而MultiAdds少了2.23倍。另外，我们的轻量模型得到的性能比DeepLabV3+低了1.2%，但参数少了76.7%，MultiAdds少了4.65倍。最后，在PASCAL VOC 2012和ADE20K上，我们的最佳模型超过了几个目前最好的模型，而只使用了很少的数据进行预训练。

To summarize, the contribution of our paper is four-fold: 总结一下，我们的文章有以下四个贡献：

- Ours is one of the first attempts to extend NAS beyond image classification to dense image prediction. 第一次将NAS从图像分类拓展到密集图像预测。
- We propose a network level architecture search space that augments and complements the much-studied cell level one, and consider the more challenging joint search of network level and cell level architectures. 我们提出了一种网络级结构搜索空间，与研究很多的单元级结构是扩充和互补的关系，并进行了更有挑战的网络级和单元级联合搜索。
- We develop a differentiable, continuous formulation that conducts the two-level hierarchical architecture search efficiently in 3 GPU days. 我们提出了一种可微分的，连续公式，进行高效的两级层次化架构搜索，只需3 GPU天。
- Without ImageNet pretraining, our model significantly outperforms FRRN-B and GridNet, and attains comparable performance with other ImageNet-pretrained state-of-the-art models on Cityscapes. On PASCAL VOC 2012 and ADE20K, our best model also outperforms several state-of-the-art models. 在没有ImageNet预训练的情况下，我们的模型显著超过了FRRN-B和GridNet，在Cityscapes数据集上，得到了与其他ImageNet预训练的目前最好模型可比的性能。在PASCAL VOC 2012和ADE20K上，我们最好的模型也超过了几个目前最好的模型。

## 2. Related Work 相关工作

**Semantic Image Segmentation**: Convolutional neural networks [42] deployed in a fully convolutional manner (FCNs [68, 51]) have achieved remarkable performance on several semantic segmentation benchmarks. Within the state-of-the-art systems, there are two essential components: multi-scale context module and neural network design. It has been known that context information is crucial for pixel labeling tasks [26, 70, 37, 39, 16, 54, 14, 10]. Therefore, PSPNet [88] performs spatial pyramid pooling [21, 41, 24] at several grid scales (including image-level pooling [50]), while DeepLab [8, 9] applies several parallel atrous convolution [28, 20, 68, 57, 7] with different rates. On the other hand, the improvement of neural network design has significantly driven the performance from AlexNet [38], VGG [72], Inception [32, 76, 74], ResNet [25] to more recent architectures, such as Wide ResNet [86], ResNeXt [85], DenseNet [31] and Xception [12, 61]. In addition to adopting those networks as backbones for semantic segmentation, one could employ the encoder-decoder structures [64, 2, 55, 44, 60, 58, 33, 79, 18, 11, 87, 83] which efficiently captures the long-range context information while keeping the detailed object boundaries. Nevertheless, most of the models require initialization from the ImageNet [65] pretrained checkpoints except FRRN [60] and GridNet [17] for the task of semantic segmentation. Specifically, FRRN [60] employs a two-stream system, where full-resolution information is carried in one stream and context information in the other pooling stream. GridNet, building on top of a similar idea, contains multiple streams with different resolutions. In this work, we apply neural architecture search for network backbones specific for semantic segmentation. We further show state-of-the-art performance without ImageNet pretraining, and significantly outperforms FRRN [60] and GridNet [17] on Cityscapes [13].

**语义图像分割**：CNNs以全卷积的形式部署，在几个语义分割基准测试中得到了非常好的结果。在目前最好的系统中，有两个基本的组件：多尺度上下文模块，和神经网络设计。众所周知，上下文信息对于像素标记任务是非常关键的。所以，PSPNet在几个网格尺度下进行空间金字塔池化（包括图像级的池化），而DeepLab使用了几个并行的不同比率的孔洞卷积。另一方面，神经网络设计的改进，显著推进了网络性能的提升，从AlexNet, VGG, Inception, ResNet到最近的架构，如Wide ResNet，ResNeXt，DenseNet和Xception。除了采用这些网络作为骨干，以进行语义分割，还可以采用编码器-解码器结构，可以高效的捕获长程上下文信息，而保持目标的细节边缘。尽管如此，多数模型需要进行ImageNet预训练初始化，以进行语义分割，除了FRRN和GridNet模型。具体的，FRRN采用了一种双流结构系统，其中全分辨率信息是在一个流上，上下文信息在另一个池化流上。GridNet，是构建在类似的思想上，包含多个不同分辨率的流。本文中，我们对网络骨干使用神经架构搜索，专门进行语义分割。我们进一步给出目前最好的性能，没有经过ImageNet预训练，在Cityscapes上显著超过了FRRN和GridNet。

**Neural Architecture Search Method**: Neural Architecture Search aims at automatically designing neural network architectures, hence minimizing human hours and efforts. While some works [22, 34, 92, 49] search RNN cells for language tasks, more works search good CNN architectures for image classification. 神经架构搜索目标是自动设计神经网络架构，减少人的参与。一些工作对语言任务搜索RNN单元，更多的工作为图像分类搜索好的CNN架构。

Several papers used reinforcement learning (either policy gradients [92, 93, 5, 77] or Q-learning [3, 89]) to train a recurrent neural network that represents a policy to generate a sequence of symbols specifying the CNN architecture. An alternative to RL is to use evolutionary algorithms (EA), that “evolves” architectures by mutating the best architectures found so far [63, 84, 53, 48, 62]. However, these RL and EA methods tend to require massive computation during the search, usually thousands of GPU days. PNAS [47] proposed a progressive search strategy that markedly reduced the search cost while maintaining the quality of the searched architecture. NAO [52] embedded architectures into a latent space and performed optimization before decoding. Additionally, several works [59, 49, 1] utilized architectural sharing among sampled models instead of training each of them individually, thereby further reduced the search cost. Our work follows the differentiable NAS formulation [69, 49] and extends it into the more general hierarchical setting.

几篇文章使用强化学习(Reinforcement Learning, RL)来训练一个RNN，代表一种策略来生成符号序列，指定CNN架构。还可以使用演化算法(evolutionary algorithms, EA)，通过对目前找到的最佳架构进行变异，来演化架构。但是，这些RL和EA方法一般需要在搜索过程中进行大量计算，通过需要几千GPU天。PNAS[47]提出一种渐进搜索策略，极大降低了搜索耗时，而且维持了搜索得到的架构的质量。NAO[52]将架构嵌入到一种潜伏空间中，在编码前进行优化。另外，几篇文章使用在采样的模型中架构共享，而不是单独训练每一种架构，所以进一步降低了搜索代价。我们的工作遵循可微分NAS公式，并将其拓展到更一般的层次化设置中。

**Neural Architecture Search Space**: Earlier papers, e.g., [92, 63], tried to directly construct the entire network. However, more recent papers [93, 47, 62, 59, 49] have shifted to searching the repeatable cell structure, while keeping the outer network level structure fixed by hand. First proposed in [93], this strategy is likely inspired by the two-level hierarchy commonly used in modern CNNs.

**神经架构搜索空间**：早期的文章，如[92,63]，试图直接构建整个网络。但是，最近的文章转移到了搜索可重复的单元结构，同时保持外部网络级的架构由手工设计。这种策略首先在[93]中提出，很可能是受现代CNNs常用的两级层次结构启发得到的。

Our work still uses this cell level search space to keep consistent with previous works. Yet one of our contributions is to propose a new, general-purpose network level search space, since we wish to jointly search across this two-level hierarchy. Our network level search space shares a similar outlook as [67], but the important difference is that [67] kept the entire “fabrics” with no intention to alter the architecture, whereas we associate an explicit weight for each connection and focus on decoding a single discrete structure. In addition, [67] was evaluated on segmenting face images into 3 classes [35], whereas our models are evaluated on large-scale segmentation datasets such as Cityscapes [13], PASCAL VOC 2012 [15], and ADE20K [90].

我们的工作仍然使用了这种单元级的搜索空间，以与之前的工作保持连续性。但我们的一个贡献是，提出了新的通用的网络级搜索空间，因为我们希望同时搜索这种两级层次架构。我们的网络级搜索空间与[67]外表相似，但重要的区别是[67]保持了整个fabrics，不想改变架构，而我们为每个连接关联了一个显式的权重，聚焦在解码单个离散结构。另外，[67]的评估是将人脸图像分割成3个类别，而我们的模型是在大规模分割数据集上评估的，如Cityscapes，PASCAL VOC 2012，和ADE20K。

The most similar work to ours is [6], which also studied NAS for semantic image segmentation. However, [6] focused on searching the much smaller Atrous Spatial Pyramid Pooling (ASPP) module using random search, whereas we focus on searching the much more fundamental network backbone architecture using more advanced and more efficient search methods.

与我们最相似的工作是[6]，也研究NAS在语义分割上的应用。但是，[6]关注的是使用随机搜索方法，搜索小的多的ASPP模块，而我们关注的是搜索更基础的网络骨干架构，使用的是更先进更有效率的搜索方法。

## 3. Architecture Search Space 架构搜索空间

This section describes our two-level hierarchical architecture search space. For the inner cell level (Sec. 3.1), we reuse the one adopted in [93, 47, 62, 49] to keep consistent with previous works. For the outer network level (Sec. 3.2), we propose a novel search space based on observation and summarization of many popular designs.

这一节描述的是我们两级层次化架构搜索空间。对于内部单元级（3.1节），我们重用了[93,47,62,49]的方法以与之前的工作保持连续。对于外部网络级（3.2节），我们提出了新的搜索空间，基于对很多流行设计的观察和总结。

### 3.1. Cell Level Search Space 单元级搜索空间

We define a cell to be a small fully convolutional module, typically repeated multiple times to form the entire neural network. More specifically, a cell is a directed acyclic graph consisting of B blocks.

我们定义单元为小型全卷积模块，一般重复很多次，以形成整个神经网络。特别的，一个单元是一个有向无环图，包含B个模块。

Each block is a two-branch structure, mapping from 2 input tensors to 1 output tensor. Block i in cell l may be specified using a 5-tuple ($I_1, I_2, O_1, O_2, C$), where $I_1, I_2 ∈ I_i^l$ are selections of input tensors, $O_1, O_2 ∈ O$ are selections of layer types applied to the corresponding input tensor, and C ∈ C is the method used to combine the individual outputs of the two branches to form this block’s output tensor, $H_i^l$. The cell’s output tensor $H^l$ is simply the concatenation of the blocks’ output tensors $H_1^l, . . ., H_B^l$ in this order.

每个模块是一个双分支的结构，两个输入张量映射到一个输出张量。单元l中的模块i可以使用一个5元组指定($I_1, I_2, O_1, O_2, C$)，其中$I_1, I_2 ∈ I_i^l$是输入张量的选项，$O_1, O_2 ∈ O$是应用于对应输入张量的层的类型的选项，C ∈ C是将两个分支的输出结合到一起形成这个模块的输出张量$H_i^l$的方法。单元的输出张量$H^l$是每个模块输出张量的简单拼接$H_1^l, . . ., H_B^l$。

The set of possible input tensors, $I_i^l$, consists of the output of the previous cell $H^{l−1}$, the output of the previous-previous cell $H^{l−2}$, and previous blocks’ output in the current cell {$H_1^l, . . . , H_i^l$}. Therefore, as we add more blocks in the cell, the next block has more choices as potential source of input.

可能的输入张量集合$I_i^l$，包括之前单元的输出$H^{l−1}$，之前两个单元的输出$H^{l−2}$，当前单元前面模块的输出{$H_1^l, . . . , H_i^l$}。因此，当我们在单元中加入更多模块时，下一个模块有更多的可能输入选项。

The set of possible layer types, O, consists of the following 8 operators, all prevalent in modern CNNs: 可能的层的类型O，包含下面8个算子，在现代CNNs中很常用：

- 3 × 3 depthwise-separable conv
- 5 × 5 depthwise-separable conv
- 3 × 3 atrous conv with rate 2
- 5 × 5 atrous conv with rate 2
- 3 × 3 average pooling
- 3 × 3 max pooling
- skip connection
- no connection (zero)

For the set of possible combination operators C, we simply let element-wise addition to be the only choice. 对于可能的组合算子C，我们只进行逐元素的相加。

### 3.2. Network Level Search Space 网络级搜索空间

In the image classification NAS framework pioneered by [93], once a cell structure is found, the entire network is constructed using a pre-defined pattern. Therefore the network level was not part of the architecture search, hence its search space has never been proposed nor designed.

在图像分类NAS的先驱工作[93]中，一旦发现一个单元结构，就使用一个预定义的模式来构建整个网络。因此网络级不是架构搜索的一部分，所以这种搜索空间从来没有提出来或设计过。

This pre-defined pattern is simple and straightforward: a number of “normal cells” (cells that keep the spatial resolution of the feature tensor) are separated equally by inserting “reduction cells” (cells that divide the spatial resolution by 2 and multiply the number of filters by 2). This keep-downsampling strategy is reasonable in the image classification case, but in dense image prediction it is also important to keep high spatial resolution, and as a result there are more network level variations [9, 56, 55].

这种预定义的模式是简单直接的：几个正常单元（保持特征张量的空间分辨率的单元）之间由降维单元（空间分辨率减少一半滤波器数量增加两倍的单元）均匀分隔开。这种持续下采样的策略在图像分类的情况下是正常的，但在密集预测任务中，保持高分辨率也是非常重要的，结果是，有更多网络级的变化。

Among the various network architectures for dense image prediction, we notice two principles that are consistent: 在各种密集预测的网络架构中，我们注意到一直有两类原则：

- The spatial resolution of the next layer is either twice as large, or twice as small, or remains the same.下一层空间分辨率，要么是大了两倍，要么是小了两倍；
- The smallest spatial resolution is downsampled by 32. 最小的空间分辨率是下采样了32倍的。

Following these common practices, we propose the following network level search space. The beginning of the network is a two-layer “stem” structure that each reduces the spatial resolution by a factor of 2. After that, there are a total of L layers with unknown spatial resolutions, with the maximum being downsampled by 4 and the minimum being downsampled by 32. Since each layer may differ in spatial resolution by at most 2, the first layer after the stem could only be either downsampled by 4 or 8. We illustrate our network level search space in Fig. 1. Our goal is then to find a good path in this L-layer trellis.

按照这些通用的实践，我们提出下面的网络级搜索空间。网络的开始是两层stem结构，每层将空间分辨率降低两倍。之后，共有L层未知空间分辨率的层，最大可以进行四倍下采样，最小可以进行32倍下采样。由于每一层的空间分辨率最多差两倍，stem后的第一层应当是下采样了4倍或8倍。我们的网络级搜索如图1所示。我们的目标是在这个L层的结构中找到一个好的路径。

Figure 1: Left: Our network level search space with L = 12. Gray nodes represent the fixed “stem” layers, and a path along the blue nodes represents a candidate network level architecture. Right: During the search, each cell is a densely connected structure as described in Sec. 4.1.1. Every yellow arrow is associated with the set of values $α_{j→i}$. The three arrows after concat are associated with $β^l_{s/2→s}, β^l_{s→s}, β^l_{2s→s}$ respectively, as described in Sec. 4.1.2. Best viewed in color.

In Fig. 2 we show that our search space is general enough to cover many popular designs. In the future, we have plans to relax this search space even further to include U-net architectures [64, 45, 71], where layer l may receive input from one more layer preceding l in addition to l − 1.

图2中，我们展示了，我们的搜索空间是足够通用的，可以覆盖很多流行设计。在将来，我们计划进一步对这个搜索空间松弛化，以包括U-net架构，其中层l的输入可以除了是l-1层，还可以是l-1的前一层。

We reiterate that our work searches the network level architecture in addition to the cell level architecture. Therefore our search space is strictly more challenging and general-purpose than previous works.

我们重申，我们的工作搜索网络级的架构，和单元级的架构。所以我们的搜索空间比之前的工作更有挑战性，更通用。

Figure 2: Our network level search space is general and includes various existing designs. (a) Network level architecture used in DeepLabv3 [9]. (b)Conv-Deconv [56]. (c)Stacked Hourglass [55].

## 4. Methods 方法

We begin by introducing a continuous relaxation of the (exponentially many) discrete architectures that exactly matches the hierarchical architecture search described above. We then discuss how to perform architecture search via optimization, and how to decode back a discrete architecture after the search terminates.

我们先介绍一个（指数级数量的）离散架构的连续松弛，与上面所述的层次式架构搜索严格匹配；然后讨论如何通过优化进行架构搜索，以及在搜索停止时怎样将离散结构解码回来。

### 4.1. Continuous Relaxation of Architectures 架构的连续松弛

#### 4.1.1 Cell Architecture 单元架构

We reuse the continuous relaxation described in [49]. Every block’s output tensor $H_i^l$ is connected to all hidden states in $I_i^l$: 我们重用[49]中的连续松弛。每个模块的输出张量$H_i^l$都与$I_i^l$中的所有隐藏状态相连接：

$$H_i^l = \sum_{H_j^l ∈ I_i^l} O_{j→i} (H_j^l)$$(1)

In addition, we approximate each $O_{j→i}$ with its continuous relaxation $Ō_{j→i}$, defined as: 另外，我们用每个$O_{j→i}$的连续松弛来进行近似$Ō_{j→i}$，定义为：

$$Ō_{j→i} (H_j^l) = \sum_{O_k ∈ O} α_{j→i}^k O^k (H_j^l)$$(2)

where 其中

$$\sum_{k=1}^|O| α_{j→i}^k = 1, ∀i, j$$(3)
$$α_{j→i}^k ≥ 0, ∀i, j, k$$(4)

In other words, $α_{j→i}^k$ are normalized scalars associated with each operator $O^k ∈ O$, easily implemented as softmax. 换句话说，$α_{j→i}^k$是与每个算子$O^k ∈ O$相关联的归一化标量，可以很容易的用softmax实现。

Recall from Sec. 3.1 that $H^{l−1}$ and $H^{l−2}$ are always included in $I_i^l$, and that $H^l$ is the concatenation of $H_1^l, . . ., H_B^l$. Together with Eq. (1) and Eq. (2), the cell level update may be summarized as: 回想一下3.1节，$H^{l−1}$和$H^{l−2}$一直包含在$I_i^l$中，$H^l$是of $H_1^l, . . ., H_B^l$的拼接。与式(1)和式(2)一起，单元级的更新可以总结为：

$$H^l = Cell(H^{l−1}, H^{l−2}; α)$$(5)

#### 4.1.2 Network Architecture 网络架构

Within a cell, all tensors are of the same spatial size, which enables the (weighted) sum in Eq. (1) and Eq. (2). However, as clearly illustrated in Fig. 1, tensors may take different sizes in the network level. Therefore in order to set up the continuous relaxation, each layer l will have at most 4 hidden states {$^4H^l, ^8H^l, ^{16} H^l, ^{32}H^l$}, with the upper left superscript indicating the spatial resolution.

在一个单元中，所有张量都是相同的空间分辨率大小，所以才可以在式(1)(2)中进行加权求和。但是，如图1中清晰所示，张量在网络级的空间大小可能是不一样的。因此为进行连续松弛，每一层l可以有最多4个隐藏层{$^4H^l, ^8H^l, ^{16} H^l, ^{32}H^l$}，左上角的上标表示空间分辨率。

We design the network level continuous relaxation to exactly match the search space described in Sec. 3.2. We associated a scalar with each gray arrow in Fig. 1, and the network level update is: 我们设计网络级连续松弛，与3.2节叙述的搜索空间严格匹配。我们将图1中的每个灰色箭头都关联一个标量，网络级的更新为：

$$^sH^l = β_{s/2→s}^l Cell(^{s/2} H^{l-1}, ^s H^{l-2};α) + β_{s→s}^l Cell(^sH^{l-1}, ^sH^{l-2};α) + β_{2s→s}^l Cell(^{2s} H^{l-1}, ^sH^{l-2};α)$$(6)

where s = 4, 8, 16, 32 and l = 1, 2, . . . , L. The scalars β are normalized such that 其中s = 4, 8, 16, 32，l = 1, 2, . . . , L。标量β是归一化的，这样

$$β_{s→s/2}^l + β_{s→s}^l + β_{s→2s}^l = 1, ∀s, l$$(7)
$$β_{s→s/2}^l>0, β_{s→s}^l>0, β_{s→2s}^l>0, ∀s, l$$(8)

also implemented as softmax.也是用softmax实现的。

Eq. (6) shows how the continuous relaxations of the two-level hierarchy are weaved together. In particular, β controls the outer network level, hence depends on the spatial size and layer index. Each scalar in β governs an entire set of α, yet α specifies the same architecture that depends on neither spatial size nor layer index.

式(6)展示了两级层次结构是怎样一起进行连续松弛的。特别的，β控制着外部网络级，所以依赖于空间分辨率大小和层的索引。β中的每个标量控制着α的整个集合，但α指定了同样的架构，既不依赖于空间分辨率大小，也不依赖于层索引。

As illustrated in Fig. 1, Atrous Spatial Pyramid Pooling (ASPP) modules are attached to each spatial resolution at the L-th layer (atrous rates are adjusted accordingly). Their outputs are bilinear upsampled to the original resolution before summed to produce the prediction.

如图1所示，ASPP接在第L层的每个分辨率上（孔洞率也随之调整）。其输出进行双线性插值回原始分辨率大小，然后求和得到预测。

### 4.2. Optimization 优化

The advantage of introducing this continuous relaxation is that the scalars controlling the connection strength between different hidden states are now part of the differentiable computation graph. Therefore they can be optimized efficiently using gradient descent. We adopt the first-order approximation in [49], and partition the training data into two disjoint sets trainA and trainB. The optimization alternates between:

引入连续松弛的优势是，控制不同隐藏状态间连接强度的标量，现在是可微分计算图的一部分。因此，可以使用梯度下降进行高效的优化。我们采用[49]中的一阶近似，将训练数据分成两个不相交的集合trainA和trainB。优化在下面两个之间交替进行：

1. Update network weights w by $∇_w L_{trainA} (w, α, β)$;
2. Update architecture α, β by $∇_{α,β} L_{trainB} (w, α, β)$;

where the loss function L is the cross entropy calculated on the semantic segmentation mini-batch. The disjoint set partition is to prevent the architecture from overfitting the training data. 损失函数L是在语义分割mini-batch计算得到的交叉熵。不想交的集合分割是为了防止架构从训练数据中过拟合。

### 4.3. Decoding Discrete Architectures 解码离散架构

**Cell Architecture**. Following [49], we decode the discrete cell architecture by first retaining the 2 strongest predecessors for each block (with the strength from hidden state j to hidden state i being $max_{k, O^k != zero} α_{j→i}^k$; recall from Sec. 3.1 that “zero” means “no connection”), and then choose the most likely operator by taking the argmax.

**单元架构**：按照[49]，我们通过首先为每个模块得到2个最强的前任，解码离散单元架构，（从隐藏状态j到隐藏状态i的强度为$max_{k, O^k != zero} α_{j→i}^k$，回想3.1节，zero意味着没有连接），然后通过采用argmax来选择最可能的算子。

**Network Architecture**. Eq. (7) essentially states that the “outgoing probability” at each of the blue nodes in Fig. 1 sums to 1. In fact, the β values can be interpreted as the “transition probability” between different “states” (spatial resolution) across different “time steps” (layer number). Quite intuitively, our goal is to find the path with the “maximum probability” from start to end. This path can be decoded efficiently using the classic Viterbi algorithm, as in our implementation.

**网络架构**。式(7)说明，图1中每个蓝色节点的向外的概率总结为1。实际上，β值可以解释为不同“时间步骤”（层数）不同状态（空间分辨率）间的迁移概率。我们的目标是找到从开始到结束的最大概率的路径，这很自然。这个路径可以使用经典的Viterbi算法进行高效的解码，我们的实现就是这样的。

## 5. Experimental Results 试验结果

Herein, we report our architecture search implementation details as well as the search results. We then report semantic segmentation results on benchmark datasets with our best found architecture. 这里，我们给出架构搜索算法的细节，以及搜索结果。我们然后在基准测试数据集上，给出找到的最好模型的语义分割结果。

### 5.1. Architecture Search Implementation Details 架构搜索实现的细节

We consider a total of L = 12 layers in the network, and B = 5 blocks in a cell. The network level search space has 2.9 × $10^4$ unique paths, and the number of cell structures is 5.6 × $10^{14}$. So the size of the joint, hierarchical search space is in the order of $10^{19}$.

我们考虑网络中总计有12层的情况，一个单元中有B=5个模块。网络级的搜索空间有2.9 × $10^4$条路径，单元结构的数量为5.6 × $10^{14}$。所以联合、层次式的搜索空间的大小为$10^{19}$量级。

We follow the common practice of doubling the number of filters when halving the height and width of feature tensor. Every blue node in Fig. 1 with downsample rate s has B × F × s/4 output filters, where F is the filter multiplier controlling the model capacity. We set F = 8 during the architecture search. A stride 2 convolution is used for all s/2 → s connections, both to reduce spatial size and double the number of filters. Bilinear upsampling followed by 1 × 1 convolution is used for all 2s → s connections, both to increase spatial size and halve the number of filters.

我们采用通行做法，特征张量的宽度和高度减半时，滤波器数量加倍。图1中的每个蓝色节点，下采样率为s，有B × F × s/4个输出滤波器，其中F是滤波器乘子，控制着模型的容量。我们在架构搜索的过程中，设F=8。步长为2的卷积用在所有的s/2 → s连接上，以进行降低空间分辨率大小，滤波器数量加倍。双线性插值上采样，随后是1×1卷积，用于2s → s连接，以增加空间分辨率大小和将滤波器数量减半。

The Atrous Spatial Pyramid Pooling module used in [9] has 5 branches: one 1 × 1 convolution, three 3 × 3 convolution with various atrous rates, and pooled image feature. During the search, we simplify ASPP to have 3 branches instead of 5 by only using one 3 × 3 convolution with atrous rate 96/s. The number of filters produced by each ASPP branch is still B × F × s/4.

[9]中使用的ASPP模块有5个分支：一个1×1卷积，3个不同孔洞率的3×3卷积，和池化的图像特征。在搜索中，我们简化了ASPP，由5个分支变成3个分支，只使用了一个3×3卷积，孔洞率为96/s。每个ASPP分支产生的滤波器数量仍然是B × F × s/4。

We conduct architecture search on the Cityscapes dataset [13] for semantic image segmentation. More specifically, we use 321 × 321 random image crops from half-resolution (512 × 1024) images in the train_fine set. We randomly select half of the images in train_fine as trainA, and the other half as trainB (see Sec. 4.2).

我们在Cityscapes数据集上进行语义分割的架构搜索。更具体的，我们使用train_fine集中的半分辨率512×1024图像的321×321随机图像剪切块。我们从train_fine集中随机选择一半图像作为trainA，另外一半作为trainB（见4.2节）。

The architecture search optimization is conducted for a total of 40 epochs. The batch size is 2 due to GPU memory constraint. When learning network weights w, we use SGD optimizer with momentum 0.9, cosine learning rate that decays from 0.025 to 0.001, and weight decay 0.0003. The initial values of α, β before softmax are sampled from a standard Gaussian times 0.001. They are optimized using Adam optimizer [36] with learning rate 0.003 and weight decay 0.001. We empirically found that if α, β are optimized from the beginning when w are not well trained, the architecture tends to fall into bad local optima. Therefore we start optimizing α, β after 20 epochs. The entire architecture search optimization takes about 3 days on one P100 GPU. Fig. 4 shows that the validation accuracy steadily improves throughout this process. We also tried searching for longer epochs (60, 80, 100), but did not observe benefit.

架构搜索优化供进行了40轮。由于GPU内存限制，batch size为2。在学习网络权重w时，我们使用SGD优化器，动量为0.9，cosine学习速率，从0.025衰减到0.001，权重衰减为0.0003。在softmax前的α, β初始值是服从标准Gaussian分布的随机量，乘以0.001。优化使用Adam优化器，学习速率0.003，权重衰减0.001。我们通过经验发现，如果α, β在w尚未训练好的时候，从开始就进行优化，那么架构倾向于落入不好的局部极值。所以我们在20轮之后，才开始优化α, β。整个架构搜索优化过程在一个P100 GPU上耗时三天。图4展示了，在这个过程中，验证准确率在稳定的改进。我们也尝试搜索更多轮(60,80,100)，但没有发现更多的好处。

Figure 4: Validation accuracy during 40 epochs of architecture search optimization across 10 random trials.

Fig. 3 visualizes the best architecture found. In terms of network level architecture, higher resolution is preferred at both beginning (stays at downsample by 4 for longer) and end (ends at downsample by 8). We also show the strongest outgoing connection at each node using gray dashed arrows. We observe a general tendency to downsample in the first 3/4 layers and upsample in the last 1/4 layers. In terms of cell level architecture, the conjunction of atrous convolution and depthwise-separable convolution is often used, suggesting that the importance of context has been learned. Note that atrous convolution is rarely found to be useful in cells for image classification(Among NASNet-{A, B, C}, PNASNet-{1, 2, 3, 4, 5}, AmoebaNet-{A, B, C}, ENAS, DARTS, atrous convolution was used only once in AmoebaNet-B reduction cell).

图3是找到的最好框架。在网络级架构上，开始时和最后时更高的分辨率都倾向于更高的分辨率，开始时保持在下采样率为4很长时间，最后以下采样率为8结束。我们还用灰色虚线箭头给出了每个节点上的最强外连节点。我们观察到在前面3/4的层中都倾向于下采样，最后1/4层中倾向于上采样。在单元级架构上，孔洞卷积和深度可分离卷积的组合经常使用，说明学习到了上下文的重要性。注意，孔洞卷积在单元中一般很少对图像分类有用（在NASNet-{A, B, C}, PNASNet-{1, 2, 3, 4, 5}, AmoebaNet-{A, B, C}, ENAS, DARTS中，孔洞卷积只在AmoebaNet-B降维单元中使用了一次）。

Figure 3: The Auto-DeepLab architecture found by our Hierarchical Neural Architecture Search on Cityscapes. Gray dashed arrows show the connection with maximum β at each node. atr: atrous convolution. sep: depthwise-separable convolution.

### 5.2. Semantic Segmentation Results 语义分割结果

We evaluate the performance of our found best architecture (Fig. 3) on Cityscapes [13], PASCAL VOC 2012 [15], and ADE20K [90] datasets. 我们在三个语义分割数据集上评估我们找到的最佳架构（图3）。

We follow the same training protocol in [9, 11]. In brief, during training we adopt a polynomial learning rate schedule [50] with initial learning rate 0.05, and large crop size (e.g., 769 × 769 on Cityscapes, and 513 × 513 on PASCAL VOC 2012 and resized ADE20K images). Batch normalization parameters [32] are fine-tuned during training. The models are trained from scratch with 1.5M iterations on Cityscapes, 1.5M iterations on PASCAL VOC 2012, and 4M iterations on ADE20K, respectively.

我们采用[9,11]中的训练方案。简要来说，在训练时，我们采用多项式学习速率策略[50]，初始学习速率为0.05，采用大剪切块（如在Cityscapes上769×769，在PASCAL VOC 2012和ADE20K上513×513）。BN参数在训练时进行精调。模型从头进行训练，在Cityscapes上1.5M迭代次数，在PASCAL VOC 2012上1.5M迭代次数，在ADE20K上4M迭代次数。

We adopt the simple encoder-decoder structure similar to DeepLabv3+ [11]. Specifically, our encoder consists of our found best network architecture augmented with the ASPP module [8, 9], and our decoder is the same as the one in DeepLabv3+ which recovers the boundary information by exploiting the low-level features that have downsample rate 4. Additionally, we redesign the “stem” structure with three 3 × 3 convolutions (with stride 2 in the first and third convolutions). The first two convolutions have 64 filters while the third convolution has 128 filters. This “stem” has been shown to be effective for segmentation in [88, 78].

我们采用简单的编码器-解码器结构，与DeepLabV3+类似。具体的，我们的编码器由我们找到的最佳网络结构组成，使用ASPP模块进行增强，我们的解码器与DeepLabV3+中使用的一样，通过利用底层特征（下采样率为4）来恢复图像边缘。另外，我们重新设计了stem结构，采用3个3×3卷积（第1个和第3个步长为2）。前2个卷积有64个滤波器，第3个有128个滤波器。这种stem在[88,78]中已经显示非常有效。

#### 5.2.1 Cityscapes

Cityscapes [13] contains high quality pixel-level annotations of 5000 images with size 1024 × 2048 (2975, 500, and 1525 for the training, validation, and test sets respectively) and about 20000 coarsely annotated training images. Following the evaluation protocol [13], 19 semantic labels are used for evaluation without considering the void label.

Cityscapes包含高质量像素级标注的5000幅图像，大小1024×2048，（训练、验证和测试集分别有2975,500和1525幅），还有约20000幅粗糙标注的训练图像。按照[13]的评估方案，对19个语义标签进行评估，没有考虑void标签。

In Tab. 2, we report the Cityscapes validation set results. Similar to MobileNets [29, 66], we adjust the model capacity by changing the filter multiplier F. As shown in the table, higher model capacity leads to better performance at the cost of slower speed (indicated by larger Multi-Adds).

在表2中，我们给出了Cityscapes验证集的结果。与MobileNets类似[29,66]，我们通过改变滤波器乘子F来调整模型容量。如表所示，更高的模型容量带来更好的性能，代价是更低的速度（即更多的Multi-Adds）。

Table 2: Cityscapes validation set results with different Auto-DeepLab model variants. F : the filter multiplier controlling the model capacity. All our models are trained from scratch and with single-scale input during inference.

Method | ImageNet | F | Multi-Adds | Params | mIOU(%)
--- | --- | --- | --- | --- | ---
Auto-DeepLab-S | n | 20 | 333.25B | 10.15M | 79.74
Auto-DeepLab-M | n | 32 | 460.93B | 21.62M | 80.04
Auto-DeepLab-L | n | 48 | 695.03B | 44.42M | 80.33
FRRN-A [60] | n | - | - | 17.76M | 65.7
FRRN-B [60] | n | - | - | 24.78M | -
DeepLabv3+ [11] | y | - | 1551.05B | 43.48M | 79.55

In Tab. 3, we show that increasing the training iterations from 500K to 1.5M iterations improves the performance by 2.8%, when employing our light-weight model variant, Auto-DeepLab-S. Additionally, adopting the Scheduled Drop Path [40, 93] further improves the performance by 1.74%, reaching 79.74% on Cityscapes validation set.

在表3中，我们展示了，当使用轻量模型变体Auto-DeepLab-S时，将训练迭代次数从500K增加到1.5M，性能可以改进2.8%。另外，采用Scheduled Drop Path[40,93]，可以进一步将性能改进1.74%，在Cityscapes验证集上达到79.74%。

Table 3: Cityscapes validation set results. We experiment with the effect of adopting different training iterations (500K, 1M, and 1.5M iterations) and the Scheduled Drop Path method (SDP). All models are trained from scratch.

Method | iter-500K | iter-1M | iter-1.5M | SDP | mIOU(%)
--- | --- | --- | --- | --- | ---
Auto-DeepLab-S | y | n | n | n | 75.20
Auto-DeepLab-S | n | y | n | n | 77.09
Auto-DeepLab-S | n | n | y | n | 78.00
Auto-DeepLab-S | n | n | y | y | 79.74

We then report the test set results in Tab. 4. Without any pretraining, our best model (Auto-DeepLab-L) significantly outperforms FRNN-B [60] by 8.6% and GridNet [17] by 10.9%. With extra coarse annotations, our model Auto-DeepLab-L, without pretraining on ImageNet [65], achieves the test set performance of 82.1%, outperforming PSPNet [88] and Mapillary [4], and attains the same performance as DeepLabv3+ [11] while requiring 55.2% fewer Mutli-Adds computations. Notably, our light-weight model variant, Auto-DeepLab-S, attains 80.9% on the test set, comparable to PSPNet, while using merely 10.15M parameters and 333.25B Multi-Adds.

我们在表4中给出在测试集上的结果。没有任何预训练，我们的最佳模型(Auto-DeepLab-L)显著超过了FRNN-B[60] 8.6%，超过了GridNet[17] 10.9%。有额外的粗糙标注，我们的模型Auto-DeepLab-L，在没有ImageNet预训练的情况下，达到测试集性能82.1%，超过了PSPNet[88]和Mapillary[4]，与DeepLabV3+[11]性能类似，但需要的计算量少了55.2%。尤其是，我们的轻量模型变体，Auto-DeepLab-S，在测试集上达到了80.9%，与PSPNet性能类似，但只有10.15M参数，Multi-Adds只有333.25B。

Table 4: Cityscapes test set results with multi-scale inputs during inference. ImageNet: Models pretrained on ImageNet. Coarse: Models exploit coarse annotations.

Method | ImageNet | Coarse | mIOU (%)
--- | --- | --- | ---
FRRN-A [60] | - | - | 63.0
GridNet [17] | - | - | 69.5
FRRN-B [60] | - | - | 71.8
Auto-DeepLab-S | - | - | 79.9
Auto-DeepLab-L | - | - | 80.4
Auto-DeepLab-S | - | y | 80.9
Auto-DeepLab-L | - | y | 82.1
ResNet-38 [82] | y | y | 80.6
PSPNet [88] | y | y | 81.2
Mapillary [4] | y | y | 82.0
DeepLabv3+ [11] | y | y | 82.1
DPC [6] | y | y | 82.7
DRN CRL Coarse [91] | y | y | 82.8

#### 5.2.2 PASCAL VOC 2012

PASCAL VOC 2012 [15] contains 20 foreground object classes and one background class. We augment the original dataset with the extra annotations provided by [23], resulting in 10582 (train aug) training images.

PASCAL VOC 2012 [15]包含20个前景目标类别和1个背景类别。我们对原始数据集使用[23]提供的额外标注进行扩充，得到10582(trainaug)训练图像。

In Tab. 5, we report our validation set results. Our best model, Auto-DeepLab-L, with single scale inference significantly outperforms [19] by 20.36%. Additionally, for all our model variants, adopting multi-scale inference improves the performance by about 1%. Further pretraining our models on COCO [46] for 4M iterations improves the performance significantly.

在表5中，我们给出在验证集上的结果。我们的最佳模型，Auto-DeepLab-L，在单尺度推理上显著超过了[19] 20.36%。另外，对于我们的所有模型变体，采用多尺度推理可以改进性能大约1%。进一步在COCO[46]上预训练4M迭代，可以显著改进模型性能。

Table 5: PASCAL VOC 2012 validation set results. We experiment with the effect of adopting multi-scale inference (MS) and COCO-pretrained checkpoints (COCO). Without any pretraining, our best model (Auto-DeepLab-L) outperforms DropBlock by 20.36%. All our models are not pretrained with ImageNet images.

Method | MS | COCO | mIOU(%)
--- | --- | --- | ---
DropBlock[19] | - | - | 53.4
Auto-DeepLab-S | - | - | 71.68
Auto-DeepLab-S | y | - | 72.54
Auto-DeepLab-M | - | - | 72.78
Auto-DeepLab-M | y | - | 73.69
Auto-DeepLab-L | - | - | 73.76
Auto-DeepLab-L | y | - | 75.26
Auto-DeepLab-S | - | y | 78.31
Auto-DeepLab-S | y | y | 80.27
Auto-DeepLab-M | - | y | 79.78
Auto-DeepLab-M | y | y | 80.73
Auto-DeepLab-L | - | y | 80.75
Auto-DeepLab-L | y | y | 82.04

Finally, we report the PASCAL VOC 2012 test set result with our COCO-pretrained model variants in Tab. 6. As shown in the table, our best model attains the performance of 85.6% on the test set, outperforming RefineNet [44] and PSPNet [88]. Our model is lagged behind the top-performing DeepLabv3+ [11] with Xception-65 as network backbone by 2.2%. We think that PASCAL VOC 2012 dataset is too small to train models from scratch and pretraining on ImageNet is still beneficial in this case.

最后，我们在表6中给出COCO预训练的模型变体在PASCAL VOC 2012测试集的结果。如表所示，我们的最佳模型在测试集上得到最佳性能85.6%，超过了RefineNet[44]和PSPNet[88]。我们的模型低于最佳表现的DeepLabV3+使用Xception-65作为骨干网络的性能2.2%。我们认为PASCAL VOC 2012太小，在ImageNet上进行预训练仍然很有好处。

Table 6: PASCAL VOC 2012 test set results. Our Auto-DeepLab-L attains comparable performance with many state-of-the-art models which are pretrained on both ImageNet and COCO datasets. We refer readers to the official leader-board for other state-of-the-art models.

Method | ImageNet | COCO | mIOU(%)
--- | --- | --- | ---
Auto-DeepLab-S | - | y | 82.5
Auto-DeepLab-M | - | y | 84.1
Auto-DeepLab-L | - | y | 85.6
RefineNet [44] | y | y | 84.2
ResNet-38 [82] | y | y | 84.9
PSPNet [88] | y | y | 85.4
DeepLabv3+ [11] | y | y | 87.8
MSCI [43] | y | y | 88.0

#### 5.2.3 ADE20K

ADE20K [90] has 150 semantic classes and high quality annotations of 20000 training images and 2000 validation images. In our experiments, the images are all resized so that the longer side is 513 during training.

ADE20K[90]有150个语义类别，包含高质量标注的20000幅训练图像和2000幅验证图像。在我们的试验中，训练的时候，图像大小都经过了变换，长边为513像素。

In Tab. 7, we report our validation set results. Our models outperform some state-of-the-art models, including RefineNet [44], UPerNet [83], and PSPNet (ResNet-152) [88]; however, without any ImageNet [65] pretraining, our performance is lagged behind the latest work of [11].

在表7中，我们给出了在验证集上的结果。我们的模型超过了目前最好的模型，包括RefineNet [44], UPerNet [83], and PSPNet (ResNet-152) [88]；但在没有ImageNet预训练的情况下，我们的性能落后于最新的工作[11]。

Table 7: ADE20K validation set results. We employ multi-scale inputs during inference. †: Results are obtained from their up-to-date model zoo websites respectively. ImageNet: Models pretrained on ImageNet. Avg: Average of mIOU and Pixel-Accuracy.

Method | ImageNet | mIOU(%) | Pixel-Acc(%) | Avg(%)
--- | --- | --- | --- | ---
Auto-DeepLab-S | n | 40.69 | 80.60 | 60.65
Auto-DeepLab-M | n | 42.19 | 81.09 | 61.64
Auto-DeepLab-L | n | 43.98 | 81.72 | 62.85
CascadeNet (VGG-16) [90] | y | 34.90 | 74.52 | 54.71
RefineNet (ResNet-152) [44] | y | 40.70 | - | -
UPerNet (ResNet-101) [83] † | y | 42.66 | 81.01 | 61.84
PSPNet (ResNet-152) [88] | y | 43.51 | 81.38 | 62.45
PSPNet (ResNet-269) [88] | y | 44.94 | 81.69 | 63.32
DeepLabv3+ (Xception-65) [11] † | y | 45.65 | 82.52 | 64.09

## 6. Conclusion 结论

In this paper, we present one of the first attempts to extend Neural Architecture Search beyond image classification to dense image prediction problems. Instead of fixating on the cell level, we acknowledge the importance of spatial resolution changes, and embrace the architectural variations by incorporating the network level into the search space. We also develop a differentiable formulation that allows efficient (about 1000× faster than DPC [6]) architecture search over our two-level hierarchical search space. The result of the search, Auto-DeepLab, is evaluated by training on benchmark semantic segmentation datasets from scratch. On Cityscapes, Auto-DeepLab significantly outperforms the previous state-of-the-art by 8.6%, and performs comparably with ImageNet-pretrained top models when exploiting the coarse annotations. On PASCAL VOC 2012 and ADE20K, Auto-DeepLab also outperforms several ImageNet-pretrained state-of-the-art models.

本文中，我们第一次尝试了将NAS从图像分类拓展到密集预测问题。我们没有停留在单元层，我们承认了空间分辨率变化的重要性，在搜索空间中包括了网络级空间，以适应架构变化。我们还提出了一种可微分的公式，可以在我们的两级层次化搜索空间中进行高效的架构搜索（大约比DPC快1000倍）。搜索得到的结果，Auto-DeepLab，在语义分割基准测试中从头开始训练，以进行评估。在Cityscapes中，Auto-DeepLab明显超过了之前最好的模型8.6%，当利用粗糙标注时，与在ImageNet预训练的最好模型表现类似。在PASCAL VOC 2012和ADE20K上，Auto-DeepLab也超过了几个在ImageNet上预训练的目前最好模型。

For future work, within the current framework, related applications such as object detection should be plausible; we could also try untying the cell architecture α across different layers (cf . [77]) with little computation overhead. Beyond the current framework, a more general network level search space should be beneficial (cf . Sec. 3.2).

未来，在目前的框架中，相关的应用如目标检测应当是可行的；我们还应当尝试，使用很小的计算代价，在不同的层中将解开单元架构α。在我们目前的框架之外，更一般的网络级搜索空间应当更有好处。