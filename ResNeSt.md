# ResNeSt: Split-Attention Networks

Hang Zhang et al. Amazon

## 0. Abstract

While image classification models have recently continued to advance, most downstream applications such as object detection and semantic segmentation still employ ResNet variants as the backbone network due to their simple and modular structure. We present a modular Split-Attention block that enables attention across feature-map groups. By stacking these Split-Attention blocks ResNet-style, we obtain a new ResNet variant which we call ResNeSt. Our network preserves the overall ResNet structure to be used in downstream tasks straightforwardly without introducing additional computational costs.

虽然图像分类模型最近持续推进，但多数下游应用如目标检测和语义分割，仍然使用ResNet的变体作为骨干网络，因为简单模块化。我们提出了一种模块化的主意力分裂模块，可以在特征图组中采用注意力机制。通过将这种注意力分裂模块做成ResNet类型，我们得到了一种新的ResNet变体，我们称之为ResNeSt。我们的网络总体上保持了ResNet的结构，可以直接用于下游应用，不需要引入额外的计算代价。

ResNeSt models outperform other networks with similar model complexities. For example, ResNeSt-50 achieves 81.13% top-1 accuracy on ImageNet using a single crop-size of 224 × 224, outperforming previous best ResNet variant by more than 1% accuracy. This improvement also helps downstream tasks including object detection, instance segmentation and semantic segmentation. For example, by simply replace the ResNet-50 backbone with ResNeSt-50, we improve the mAP of Faster-RCNN on MS-COCO from 39.3% to 42.3% and the mIoU for DeeplabV3 on ADE20K from 42.1% to 45.1%.

ResNeSt模型超过了类似复杂度的其他网络。比如，ResNeSt-50在ImageNet上，只使用224 × 224的单个crop大小，就得到了81.13%的top-1准确率，超过了之前最好的ResNet变体1%。这种改进也改进了下游应用，包括目标检测，实例分割和语义分割。比如，只要将ResNet-50替换为ResNeSt-50，我们就改进了Faster-RCNN在MS-COCO上的mAP，从39.3%到42.3%，将DeepLabV3在ADE20K上的mIoU从42.1%改进到45.1%。

**Keywords**: ResNeSt, Image Classification, Transfer Learning, Object Detection, Semantic Segmentation, Instance Segmentation

## 1 Introduction

Image classification is a fundamental task in computer vision research. Networks trained for image classification often serve as the backbone of the neural networks designed for other applications, such as object detection [22,46], semantic segmentation [6,43,73] and pose estimation [14,58]. Recent work has significantly boosted image classification accuracy through large scale neural architecture search (NAS) [45,55]. Despite their state-of-the-art performance, these NAS-derived models are usually not optimized for training efficiency or memory usage on general/commercial processing hardware (CPU/GPU) [36]. Due to excessive memory consumption, some of the larger versions of these models are not even trainable on a GPU with an appropriate per-device batch-size [55]. This has limited the adoption of NAS-derived models for other applications, especially tasks involving dense predictions such as segmentation.

图像分类是计算机视觉研究的一个基础任务。训练用于图像分类的网络，通常是其他应用的网络的骨干，比如，目标检测，语义分割和姿态估计。最近的工作通过NAS显著提升了图像分类的准确率。尽管性能最优，这些NAS推导出来的模型，通常在通用/商用处理硬件(CPU/GPU)上时，在训练效率或内存使用方面并不是最优的。由于内存消耗过多，一些更大版本的这些模型，在合理的每个设备的batch大小时，在GPU上根本不能进行训练。这限制了其他应用采用NAS推导得出的模型的，尤其是对于密集预测的任务，比如分割。

Most recent work on downstream applications still uses the ResNet [23] or one of its variants as the backbone CNN. Its simple and modular design can be easily adapted to various tasks. However, since ResNet models are originally designed for image classification, they may not be suitable for various downstream applications because of the limited receptive-field size and lack of cross-channel interaction. This means that boosting performance on a given computer vision task requires “network surgery” to modify the ResNet to be more effective for that particular task. For example, some methods add a pyramid module [8,69] or introduce long-range connections [56] or use cross-channel feature-map attention [15,65]. While these approaches do improve the transfer learning performance for certain tasks, they raise the question: Can we create a versatile backbone with universally improved feature representations, thereby improving performance across multiple tasks at the same time? Cross-channel information has demonstrated success in downstream applications [56,64,65], while recent image classification networks have focused more on group or depth-wise convolution [27,28,54,60]. Despite their superior computation and accuracy tradeoff in classification tasks, these models do not transfer well to other tasks as their isolated representations cannot capture cross-channel relationships [27,28]. Therefore, a network with cross-channel representations is desirable.

下游应用的很多最近工作仍然使用ResNet，或其变体，作为CNN的骨干。其简单模块化的设备可以在多种任务中很容易的得到采用。但是，由于ResNet模型的设计是用于图像分类的，它们可能并不适合各种下游应用，因为其感受野有限，缺少通道间的交互。这意味着，在给定的计算机视觉任务中，要进行性能提升，就要对网络进行修改，使得改进的ResNet更加适合特定任务。比如，一些方法增加了一个金字塔模块，或引入了长程连接，或使用通道间特征图注意力。这些方法对特定任务确实改进了迁移学习的性能，但也提出一个问题：我们能不能创造一个多样化的骨干网络，对所有特征表示性能都有提升，因此在多个任务中可以同时改进性能呢？跨通道的信息在下游应用中已经证明是成功的，而最近的图像分类网络主要更关注分组卷积或分层卷积。尽管在分类任务中有很好的计算效率和准确率之间的折中，这些模型在其他任务中迁移效果并不好，因为其孤立的表示不能捕获到跨通道的关系。因此，一种带有跨通道表示的网络会是更加理想的。

As the first contribution of this paper, we explore a simple architectural modification of the ResNet [23], incorporating feature-map split attention within the individual network blocks. More specifically, each of our blocks divides the feature-map into several groups (along the channel dimension) and finer-grained subgroups or splits, where the feature representation of each group is determined via a weighted combination of the representations of its splits (with weights chosen based on global contextual information). We refer to the resulting unit as a Split-Attention block, which remains simple and modular. By stacking several Split-Attention blocks, we create a ResNet-like network called ResNeSt (S stands for “split”). Our architecture requires no more computation than existing ResNet-variants, and is easy to be adopted as a backbone for other vision tasks.

本文的第一个贡献是，我们对ResNet进行了简单的架构修正，将特征图分裂的注意力与单个网络模块结合了起来。更具体的，每个我们提出的模块将特征图分成几组（沿着通道维度），和细粒度的子组或分裂，其中每个组的特征表示是通过其分裂的表示的加权组合确定的（权重的选择是基于全局上下文信息的）。我们称得到的单元为注意力分裂模块，简单而又模块化。通过将几个注意力分裂模块堆叠到一起，我们创建了一种和ResNet很像的网络，称为ResNeSt（S表示分裂）。我们的架构的计算量比现有的ResNet变体不会更多，很容易在其他视觉任务中当作骨干网络。

The second contributions of this paper are large scale benchmarks on image classification and transfer learning applications. We find that models utilizing a ResNeSt backbone are able to achieve state of the art performance on several tasks, namely: image classification, object detection, instance segmentation and semantic segmentation. The proposed ResNeSt outperforms all existing ResNet variants and has the same computational efficiency and even achieves better speed-accuracy trade-offs than state-of-the-art CNN models produced via neural architecture search [55] as shown in Table 1. Our single Cascade-RCNN [3] model using a ResNeSt-101 backbone achieves 48.3% box mAP and 41.56% mask mAP on MS-COCO instance segmentation. Our single DeepLabV3 [7] model, again using a ResNeSt-101 backbone, achieves mIoU of 46.9% on the ADE20K scene parsing validation set, which surpasses the previous best result by more than 1% mIoU. Additional results can be found in Sections 5 and 6.

本文的第二个贡献是在图像分类和迁移学习应用中进行了大规模基准测试。我们发现利用ResNeSt骨干的模型可以在几个任务中获得目前最好的性能，即：图像分类，目标检测，实例分割和语义分割。提出的ResNeSt超过了已有的ResNet变体，计算效率类似，其速度-准确率折中比通过NAS搜索的目前最好的CNN模型还要好，如表1所示。我们使用ResNeSt-101骨干的单Cascade-RCNN模型在MS-COCO实例分割应用中获得了48.3%的框mAP和41.56%的掩膜mAP。我们的单DeepLabV3模型，也使用了ResNeSt-101骨干，在ADE20K场景解析验证中，获得了46.9%的mIoU，超过了之前最好的结果1%。更多结果见第5和第6部分。

Table 1: (Left) Accuracy and latency trade-off on GPU using official code implementation (details in Section 5). (Right-Top) Top-1 accuracy on ImageNet using ResNeSt. (Right-Bottom) Transfer learning results: object detection mAP on MS-COCO [42] and semantic segmentation mIoU on ADE20K [71].

## 2 Related Work

**Modern CNN Architectures**. Since AlexNet [34], deep convolutional neural networks [35] have dominated image classification. With this trend, research has shifted from engineering handcrafted features to engineering network architectures. NIN [40] first uses a global average pooling layer to replace the heavy fully connected layers, and adopts 1 × 1 convolutional layers to learn non-linear combination of the featuremap channels, which is the first kind of featuremap attention mechanism. VGG-Net [47] proposes a modular network design strategy, stacking the same type of network blocks repeatedly, which simplifies the workflow of network design and transfer learning for downstream applications. Highway network [50] introduces highway connections which makes the information flow across several layers without attenuation and helps the network convergence. Built on the success of the pioneering work, ResNet [23] introduces an identity skip connection which alleviates the difficulty of vanishing gradient in deep neural network and allows network learning deeper feature representations. ResNet has become one of the most successful CNN architectures which has been adopted in various computer vision applications.

**现代CNN架构**。自从AlexNet，DCNN已经主宰了图像分类。在这个趋势之下，研究已经从手工设计的特征转向到手工设计网络架构。NIN首先使用全局平均池化层来替换掉很重的全连接层，采用1×1的卷积层来学习特征图通道的非线性组合，这是第一种特征图注意力机制。VGG-Net提出了一种模块化的网络设计策略，将同类型的网络模块重复累加，这简化了下游应用的网络设计和迁移学习的工作流。Highway网络提出了highway连接，使得信息可以在几层之间没有衰减的流动，帮助网络收敛。在这些先驱工作之上，ResNet提出了一个恒等的skip连接，缓解了DNN中的消失梯度问题，使得网络学习更深的特征表示。ResNet已经成为了一种最成功的CNN架构，在各种计算机视觉应用都有采用。

**Multi-path and Feature-map Attention**. Multi-path representation has shown success in GoogleNet [52], in which each network block consists of different convolutional kernels. ResNeXt [61] adopts group convolution [34] in the ResNet bottle block, which converts the multi-path structure into a unified operation. SE-Net [29] introduces a channel-attention mechanism by adaptively recalibrating the channel feature responses. SK-Net [38] brings the feature-map attention across two network branches. Inspired by the previous methods, our network generalizes the channel-wise attention into feature-map group representation, which can be modularized and accelerated using unified CNN operators.

**多路径和特征图注意力**。多路径表示在GoogLeNet中得到了成功，其中每个网络模块都由不同的卷积核构成。ResNeXt在resNet瓶颈模块中采用了分组卷积，将多路结构转换成了一种统一的运算。SE-Net引入了一种通道注意力机制，自适应的对通道特征响应进行重新校准。SK-Net将特征图注意力带到了两个网络分支中。受之前的方法启发，我们的网络对逐通道的注意力机制进行泛化，到了特征图分组表示，可以使用统一的CNN算子进行模块化和加速。

**Neural Architecture Search**. With increasing computational power, interest has begun shifting from manually designed architectures to systematically searched architectures which are adaptively tailored to a particular task. Recent neural architecture search algorithms have adaptively produced CNN architectures that achieved state-of-the-art classification performance, such as: AmoebaNet [45], MNASNet [54], and EfficientNet [55]. Despite their great success in image classification, the meta network structures are distinct from each other, which makes it hard for downstream models to build upon. Instead, our model preserves ResNet meta structure, which can be directly applied on many existing downstream models [22,41,46,69]. Our approach can also augment the search spaces for neural architecture search and potentially improve the overall performance, which can be studied in the future work.

**神经架构搜索**。随着计算能力越来越强，兴趣已经从手工设计架构转移到系统的搜索架构，这对于特定的任务是自适应定制的。最近的NAS算法自适应的产生CNN架构，可以获得目前最好的分类性能，比如：AmoebaNet[45], MNASNet[54]和EfficientNet[55]。尽管在图像分类中比较成功，其元网络架构各不相同，这使得下游模型很难进行利用。而我们的模型保持了ResNet的元架构，可以直接应用到很多现有的下游模型中。我们的方法还可以扩充NAS的搜索空间，可能改进整体的性能，这可以在未来的工作中进行研究。

## 3 Split-Attention Networks

We now introduce the Split-Attention block, which enables feature-map attention across different feature-map groups. Later, we describe our network instantiation and how to accelerate this architecture via standard CNN operators. 我们现在提出分裂注意力模块，这可以在不同的特征图分组中进行特征图注意力。后来，我们描述了我们的网络实例化，以及怎样通过标准的CNN算子加速这个架构。

Our Split-Attention block is a computational unit, consisting feature-map group and split attention operations. Figure 1 (Right) depicts an overview of a Split-Attention Block. 我们的分裂注意力模块是一个计算单元，由特征图分组和分裂注意力运算组成。图1右是我们的分裂注意力模块的概览图。

**Feature-map Group**. As in ResNeXt blocks [61], the feature can be divided into several groups, and the number of feature-map groups is given by a cardinality hyperparameter K. We refer to the resulting feature-map groups as cardinal groups. We introduce a new radix hyperparameter R that indicates the number of splits within a cardinal group, so the total number of feature groups is G = KR. We may apply a series of transformations {$F_1,F_2,...F_G$} to each individual group, then the intermediate representation of each group is $U_i = F_i(X)$, for i ∈ {1, 2, ...G}.

**特征图分组**。就像在ResNeXt模块中一样，特征可以分成几组，特征图分组数量由一个基数超参数K给出。我们称得到的特征图分组为基数组。我们引入一个基数超参数R，表示一个基数组中的分裂数量，这样特征组的总数为G=KR。我们可以将一系列变换{$F_1,F_2,...F_G$}应用到每个单独的组中，然后每个组的中间表示是$U_i = F_i(X)$，i ∈ {1, 2, ...G}。

**Split Attention in Cardinal Groups**. Following [30,38], a combined representation for each cardinal group can be obtained by fusing via an element-wise summation across multiple splits. The representation for k-th cardinal group is $\hat U^k = \sum_{j=R(k-1)+1}^{Rk} U_j$, where $\hat U^k∈R^{H×W×C/K}$ for k∈1,2,...K, and H, W and C are the block output feature-map sizes. Global contextual information with embedded channel-wise statistics can be gathered with global average pooling across spatial dimensions $s^k∈R^{C/K}$ [29,38]. Here the c-th component is calculated as:

**在基数组中的分裂注意力**。按[30,38]，通过在多个分裂中进行逐元素的相加进行融合，可以得到每个基数组的综合表示。第k个基数组的表示为$\hat U^k = \sum_{j=R(k-1)+1}^{Rk} U_j$，其中$\hat U^k∈R^{H×W×C/K}$中k∈1,2,...K，H, W和C是模块输出的特征图大小。带有嵌入逐通道的统计量的全局上下文信息可以使用全局平均池化得到。这里第c个组件计算如下：

$$s_c^k = \frac {1} {H×W} \sum_{i=1}^H \sum_{j=1}^W \hat U_c^k(i,j)$$(1)

A weighted fusion of the cardinal group representation $V^k∈R^{H×W×C/K}$ is aggregated using channel-wise soft attention, where each feature-map channel is produced using a weighted combination over splits. The c-th channel is calculated as: 基数组表示的加权融合$V^k∈R^{H×W×C/K}$，使用逐通道的软注意力积聚得到，其中每个特征图通道使用每个分裂上的加权组合得到。第c个通道计算如下：

$$V_c^k = \sum_{i=1}^R a_i^k(c) U_{R(k-1)+i}$$(2)

where $a^k_i (c)$ denotes a (soft) assignment weight given by: 其中$a^k_i (c)$表示一个权重赋值，由下式给出：

$$a^k_i (c) = \left \{ \begin{matrix} \frac{exp(g_i^c(s^k))}{\sum_{j=0}^R exp(g_j^c(s^k))} & if R>1 \\ \frac{1}{1+exp(g_j^c(s^k))} & if R=1 \end{matrix} \right.$$(3)

and mapping $g_i^c$ determines the weight of each split for the c-th channel based on the global context representation $s^k$. 映射$g_i^c$根据全局上下文表示$s^k$来确定第c通道的每个分裂的权重。

**ResNeSt Block**. The cardinal group representations are then concatenated along the channel dimension: V = Concat{$V^1,V^2,...V^K$}. As in standard residual blocks, the final output Y of our Split-Attention block is produced using a shortcut connection: Y = V + X, if the input and output feature-map share the same shape. For blocks with a stride, an appropriate transformation T is applied to the shortcut connection to align the output shapes: Y = V + T(X). For example, T can be strided convolution or combined convolution-with-pooling.

**ResNeSt模块**。基数组表示然后沿着通道维进行拼接：V = Concat{$V^1,V^2,...V^K$}。就像在标准残差模块中一样，分裂注意力模块的最终输出Y，使用捷径连接生成：Y = V + X，当然这要求输入和输出特征图共享同样的形状。对于有步长的模块，会对捷径连接应用合适的变换T，以使输出形状对齐：Y = V + T(X)。比如，T可以是带步长的卷积或卷积与池化的合并。

**Instantiation, Acceleration, and Computational Costs**. Figure 1 (right) shows an instantiation of our Split-Attention block, in which the group transformation $F_i$ is a 1×1 convolution followed by a 3×3 convolution, and the attention weight function G is parameterized using two fully connected layers with ReLU activation. We draw this figure in a cardinality-major view (the featuremap groups with same cardinality index reside next to each other) for easily describing the overall logic. By switching the layout to a radix-major view, this block can be easily accelerated using standard CNN layers (such as group convolution, group fully connected layer and softmax operation), which we will describe in details in the supplementary material. The number of parameters and FLOPS of a Split-Attention block are roughly the same as a residual block [23,60] with the same cardinality and number of channels.

**实例化，加速和计算代价**。图1右是分裂注意力模块的一个实例，其中分组变换$F_i$是一个1×1的卷积和3×3卷积的合并，衰减加权函数G使用两个带有ReLU激活的全连接层进行参数化。我们在这幅图中主要表现了基准的视图（带有相同基数的特征图组，并排在一起），以很容易的描述整个逻辑。将这个排布变换到radix为主，这个模块可以很容易的使用标准CNN层进行加速（如分组卷积，分组全连接层和softmax运算），我们在附录资料中会详述。分裂注意力模块的参数数量和FLOPs，与基数和通道数相同的残差模块大致一样。

**Relation to Existing Attention Methods**. First introduced in SE-Net [29], the idea of squeeze-and-attention (called excitation in the original paper) is to employ a global context to predict channel-wise attention factors. With radix = 1, our Split-Attention block is applying a squeeze-and-attention operation to each cardinal group, while the SE-Net operates on top of the entire block regardless of multiple groups. Previous models like SK-Net [38] introduced feature attention between two network branches, but their operation is not optimized for training efficiency and scaling to large neural networks. Our method generalizes prior work on feature-map attention [29,38] within a cardinal group setting [60], and its implementation remains computationally efficient. Figure 1 shows an overall comparison with SE-Net and SK-Net blocks.

**与现有的注意力方法的关系**。压缩-注意力（原文中称为激励）的思想首先在SE-Net中提出，使用全局上下文来预测逐通道的注意力因素。在radix=1的情况下，分裂注意力模块是对每个基数组应用了一个SE运算，而SE-Net是在整个模块上运算的，不论有多少个分组。之前的模型如SK-Net，在两个网络分支之间引入了特征注意力，但其运算不是为了训练效率和缩放到大型神经网络而优化的。我们的方法，将以前在单基数组设置的特征图注意力的工作，进行了推广，其实现的计算效率仍然非常高。图1给出了与SE-Net和SK-Net模块的总体比较。

## 4 Network and Training

We now describe the network design and training strategies used in our experiments. First, we detail a couple of tweaks that further improve performance, some of which have been empirically validated in [25].

我们现在描述一下试验中用到的网络设计和训练策略。第一，我们详述了几个小调整，进一步改进了性能，其中一些调整已经通过经验在[25]中得到了验证。

### 4.1 Network Tweaks 网络调整

**Average Downsampling**. When downstream applications of transfer learning are dense prediction tasks such as detection or segmentation, it becomes essential to preserve spatial information. Recent ResNet implementations usually apply the strided convolution at the 3 × 3 layer instead of the 1 × 1 layer to better preserve such information [26,30]. Convolutional layers require handling feature-map boundaries with zero-padding strategies, which is often suboptimal when transferring to other dense prediction tasks. Instead of using strided convolution at the transitioning block (in which the spatial resolution is downsampled), we use an average pooling layer with a kernel size of 3 × 3.

**平均下采样**。当下游迁移学习应用是密集预测任务时，如检测或分割，保持空间信息就非常关键了。最近的ResNet实现，通常将有步长的卷积应用到3 × 3层，而不会应用到1 × 1层，以更好的保留这种信息。卷积层需要使用补零处理特征图边缘，当迁移到其他密集预测任务中时，通常不是最优的选择。除了在过度模块使用带步长的卷积（其中空间分辨率进行了降采样），我们使用平均池化层，核大小为3 × 3。

**Tweaks from ResNet-D**. We also adopt two simple yet effective ResNet modifications introduced by [26]: (1) The first 7 × 7 convolutional layer is replaced with three consecutive 3 × 3 convolutional layers, which have the same receptive field size with a similar computation cost as the original design. (2) A 2 × 2 average pooling layer is added to the shortcut connection prior to the 1 × 1 convolutional layer for the transitioning blocks with stride of two.

**从ResNet-D的调整**。我们还采用两种简单但有效的ResNet修正，在[26]中引入：(1)第一个7 × 7的卷积层，替换为3个连续的3 × 3卷积层，与这与原始设计有类似的计算代价，也有相同的感受野。(2)对于步长为2的过度模块，我们在1 × 1卷积层之前，对捷径连接加入了2 × 2的平均池化层。

### 4.2 Training Strategy 训练策略

**Large Mini-batch Distributed Training**. Following prior work [19,37], we train our models using 8 servers (64 GPUs in total) in parallel. Our learning rates are adjusted according to a cosine schedule [26,31]. We follow the common practice using linearly scaling-up the initial learning rate based on the mini-batch size. The initial learning rate is given by $η = \frac{B}{256} η_{base}$, where B is the mini-batch size and we use $η_{base} = 0.1$ as the base learning rate. This warm-up strategy is applied over the first 5 epochs, gradually increasing the learning rate linearly from 0 to the initial value for the cosine schedule [19, 39]. The batch normalization (BN) parameter γ is initialized to zero in the final BN operation of each block, as has been suggested for large batch training [19].

**大mini-batch分布式训练**。遵循以前的工作[19,37]，我们使用8个服务器（共64个GPUs）并行训练我们的模型。我们的学习速率是根据一个cosine方案进行调整的。我们按照常见的实践，基于mini-batch大小，使用线性缩放的初始学习速率。初始学习速率由$η = \frac{B}{256} η_{base}$给定，其中B是mini-batch大小，我们使用$η_{base} = 0.1$作为基准学习速率。这种热身的策略应用在前5轮的训练，逐步增加学习速率，对于cosine方案从0增加到初始值。BN参数γ在每个模块的最终的BN运算中初始化为0，这对于大batch训练是推荐的方案。

**Label Smoothing**. Label smoothing was first used to improve the training of Inception-V2 [53]. Recall the cross entropy loss incurred by our network’s predicted class probabilities q is computed against ground-truth p as:

**标签平滑**。标签平滑第一次是用于改进Inception-V2的训练。回忆一下交叉熵的损失，这是从我们网络预测的类别概率q中得到的，与真值p一起计算得到：

$$l(p,q) = \sum_{i=1}^K p_i logq_i$$(4)

where K is total number of classes, $p_i$ is the ground truth probability of the i-th class, and $q_i$ is the network’s predicted probability for the i-th class. As in standard image classification, we define: $q_i = \frac {exp(z_i)}{\sum_{j=1}^K exp(z_j)}$ where $z_i$ are the logits produced by our network’s ouput layer. When the provided labels are classes rather than class-probabilities (hard labels), $p_i = 1$ if i equals the ground truth class c, and is otherwise = 0. Thus in this setting: $l_{hard}(p, q) = − log q_c = −zc + log(\sum_{j=1}^K exp(z_j))$. During the final phase of training, the logits $z_j$ tend to be very small for $j\neq c$, while $z_c$ is being pushed to its optimal value ∞, and this can induce overfitting [26,53]. Rather than assigning hard labels as targets, label smoothing uses a smoothed ground truth probability:

其中K是总共类别数量，$p_i$是第i个类别的真值概率，$q_i$是第i个类别的网络的预测概率。就像在标准的图像分类中一样，我们定义：$q_i = \frac {exp(z_i)}{\sum_{j=1}^K exp(z_j)}$，其中$z_i$是我们网络输出层生成的logits。当给定的标签是类别，而不是类别概率时（硬标签），如果i等于真值标签c，那么$p_i = 1$，否则=0。因此，在这种设置中，$l_{hard}(p, q) = − log q_c = −zc + log(\sum_{j=1}^K exp(z_j))$。在训练的最终阶段，logits $z_j$对于$j\neq c$的情况会非常小，而$z_c$则被推向其最优值∞，这会带来过拟合。我们不赋值硬标签作为目标，标签平滑使用平滑的真值概率：

$$p_i = \left \{ \begin{matrix} 1-ε & if i=c \\ ε/(K-1) & otherwise \end{matrix} \right.$$(5)

with small constant ε > 0. This mitigates network overconfidence and overfitting. 其中小常数ε > 0。这缓解了网络的overconfidence和过拟合。

**Auto Augmentation**. Auto-Augment [11] is a strategy that augments the training data with transformed images, where the transformations are learned adaptively. 16 different types of image jittering transformations are introduced, and from these, one augments the data based on 24 different combinations of two consecutive transformations such as shift, rotation, and color jittering. The magnitude of each transformation can be controlled with a relative parameter (e.g. rotation angle), and transformations may be probabilistically skipped. A search which tries various candidate augmentation policies returns the best 24 best combinations. One of these 24 policies is then randomly chosen and applied to each sample image during training. The original Auto-Augment implementation uses reinforcement learning to search over these hyperparameters, treating them as categorical values in a discrete search space. For continuous search spaces, it first discretizes the possible values before searching for the best.

**自动数据扩充**。自动扩增是一种策略，用变换的图像对训练数据进行扩增，其中的变换是自适应学习得到的。引入了16种图像微调变换，从这些变换中，可以用两种连续变换得到的24种组合，对数据进行扩增，变换如平移，旋转和色彩抖动。每种变换的幅度可以用一个相对参数来控制（如旋转角度），变换可以以一定概率跳过。对各种候选扩增策略的搜索，可以得到最好的24种组合。随机选择这24种策略的一种，在训练时应用到每个样本中。原始的Auto-Augment实现使用了强化学习来搜索这些超参数，在一个离散的搜索空间将其认为是类别值。对于连续的搜索空间，在搜索最佳值之前首先将可能的值进行离散化。

**Mixup Training**. Mixup is another data augmentation strategy that generates a weighted combinations of random image pairs from the training data [67]. Given two images and their ground truth labels: $(x^{(i)},y^{(i)}),(x^{(j)},y^{(j)})$, a synthetic training example ($\hat x, \hat y$) is generated as:

**混合训练**。混合是另一种数据扩增策略，从训练图像中生成随机图像对的一种加权组合。给定两幅图像及其真值标签$(x^{(i)},y^{(i)}),(x^{(j)},y^{(j)})$，合成训练样本($\hat x, \hat y$)按下式生成：

$$\hat x = λx^i + (1 − λ)x^j$$(6)
$$\hat y = λy^i + (1 − λ)y^j$$(6)

where λ ∼ Beta(α = 0.2) is independently sampled for each augmented example. 其中λ ∼ Beta(α = 0.2)从每个扩增的样本中独立采样。

**Large Crop Size**. Image classification research typically compares the performance of different networks operating on images that share the same crop size. ResNet variants [23,26,29,60] usually use a fixed training crop size of 224, while the Inception-Net family [51–53] uses a training crop size of 299. Recently, the EfficientNet method [55] has demonstrated that increasing the input image size for a deeper and wider network may better trade off accuracy vs. FLOPS. For fair comparison, we use a crop size of 224 when comparing our ResNeSt with ResNet variants, and a crop size of 256 when comparing with other approaches.

**大的剪切大小**。图像分类研究一般比较不同的网络，但其剪切块大小都是一样的。ResNet变体通常使用固定的训练剪切块大小224，而Inception-Net族使用的训练剪切块大小为299。最近，EfficientNet方法证明了，输入图像大小增加，网络变深变宽，可能会得到更好的准确率，但FLOPs也会变大。为公平比较，我们在与ResNet变体比较时，使用的剪切块大小为224，与其他方法比较时，剪切块大小为256。

**Regularization**. Very deep neural networks tend to overfit even for large datasets [68]. To prevent this, dropout regularization randomly masks out some neurons during training (but not during inference) to form an implicit network ensemble [29, 49, 68]. A dropout layer with the dropout probability of 0.2 is applied before the final fully-connected layer to the networks with more than 200 layers. We also apply DropBlock layers to the convolutional layers at the last two stages of the network. As a structured variant of dropout, DropBlock [18] randomly masks out local block regions, and is more effective than dropout for specifically regularizing convolutional layers.

**正则化**。DNN对于大型数据集也容易过拟合。为防止这种现象，会使用dropout正则化，在训练时随机使一些神经元不起作用（但不是推理时），以形成一种隐式的网络集成。对于超过200层的网络，在最终的全连接层之前，会使用一个dropout概率0.2的层。我们也对网络的最后两个阶段的卷积层使用DropBlock层。作为dropout的结构化变体，DropBlock随机掩膜掉局部模块区域，对特别的正则化卷积层比dropout更有效率。

Finally, we also apply weight decay (i.e. L2 regularization) which additionally helps stabilize training. Prior work on large mini-batch training suggests weight decay should only be applied to the weights of convolutional and fully connected layers [19,26]. We do not subject any of the other network parameters to weight decay, including bias units, γ and β in the batch normalization layers.

最后，我们还使用了权重衰减（即，L2正则化），这进一步帮助训练更加稳定。在大的mini-batch训练上的之前工作建议，在卷积层和全连接层上都应当进行权重衰减。我们认为其他网络参数都不应当进行权重衰减，包括偏置单元，批归一化层的γ和β。

## 5 Image Classification Results

### 5.1 Implementation Details

### 5.2 Ablation Study

### 5.3 Comparing against the State-of-the-Art

## 6 Transfer Learning Results

### 6.1 Object Detection

### 6.2 Instance Segmentation

### 6.3 Semantic Segmentation

## 7 Conclusion

This work proposed the ResNeSt architecture with a novel Split-Attention block that universally improves the learned feature representations to boost performance across image classification, object detection, instance segmentation and semantic segmentation. In the latter downstream tasks, the empirical improvement produced by simply switching the backbone network to our ResNeSt is substantially better than task-specific modifications applied to a standard backbone such as ResNet. Our Split-Attention block is easy to work with and computationally efficient, and thus should be broadly applicable across vision tasks.

本文提出了ResNeSt架构，包含一种新的分裂注意力模块，改进了学习到的特征表示，在图像分类，目标检测，实例分割和语义分割中改进了性能。在之后的下游应用中，只要将骨干网络替换成ResNeSt就可以显著改进性能。我们的分裂注意力模块很容易使用，计算效率高，因此可以在很多视觉应用中都可以广泛的使用。