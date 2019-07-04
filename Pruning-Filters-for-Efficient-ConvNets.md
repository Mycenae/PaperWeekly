# Pruning Filters for Efficient ConvNets

Hao Li et al. University of Maryland / NEC Labs America

## Abstract 摘要

The success of CNNs in various applications is accompanied by a significant increase in the computation and parameter storage costs. Recent efforts toward reducing these overheads involve pruning and compressing the weights of various layers without hurting original accuracy. However, magnitude-based pruning of weights reduces a significant number of parameters from the fully connected layers and may not adequately reduce the computation costs in the convolutional layers due to irregular sparsity in the pruned networks. We present an acceleration method for CNNs, where we prune filters from CNNs that are identified as having a small effect on the output accuracy. By removing whole filters in the network together with their connecting feature maps, the computation costs are reduced significantly. In contrast to pruning weights, this approach does not result in sparse connectivity patterns. Hence, it does not need the support of sparse convolution libraries and can work with existing efficient BLAS libraries for dense matrix multiplications. We show that even simple filter pruning techniques can reduce inference costs for VGG-16 by up to 34% and ResNet-110 by up to 38% on CIFAR10 while regaining close to the original accuracy by retraining the networks.

CNN在各种应用中的成功，都伴随着计算量和参数存储代价的显著增加。最近降低开销的努力包括对各种层的权重剪枝和压缩，而不降低原始准确率。基于幅度的权重剪枝可以显著降低全连接层的参数数量，但不能很好的降低卷积层的计算量，因为在剪枝的网络中稀疏性非常不规则。我们提出一种CNN的加速方法，即对那些对输出准确率影响很小的滤波器进行剪枝。通过从网络中去除整个滤波器，以及与其相连的特征图，计算量下降非常明显。与对权重的剪枝相比，这种方法得到的不是稀疏连接的模式。所以，这不需要稀疏卷积库的支持，可以使用现有的BLAS库进行密集矩阵乘积。我们证明了，即使是简单的滤波器剪枝技术，也可以降低在CIFAR1-上，VGG-16的推理耗时34%，降低ResNet-100的38%，通过重新训练网络，可以得到接近原始准确率的结果。

## 1 Introduction 引言

The ImageNet challenge has led to significant advancements in exploring various architectural choices in CNNs (Russakovsky et al. (2015); Krizhevsky et al. (2012); Simonyan & Zisserman (2015); Szegedy et al. (2015a); He et al. (2016)). The general trend since the past few years has been that the networks have grown deeper, with an overall increase in the number of parameters and convolution operations. These high capacity networks have significant inference costs especially when used with embedded sensors or mobile devices where computational and power resources may be limited. For these applications, in addition to accuracy, computational efficiency and small network sizes are crucial enabling factors (Szegedy et al. (2015b)). In addition, for web services that provide image search and image classification APIs that operate on a time budget often serving hundreds of thousands of images per second, benefit significantly from lower inference times.

ImageNet挑战赛带来了CNNs架构探索的大发展。过去几年的一般趋势是，网络变得越来越深，参数总体数量和卷积运算越来越多。这些高容量网络推理时消耗也很多，尤其是在嵌入式设备或移动设备时，其计算能力和能耗是有限的。这些应用中，除了准确率，计算效率和网络小型化也是非常关键的。另外，对于进行图像搜索和图像分类的API来说，其耗时也非常关键，经常需要每秒跑成百上千幅图像，如果推理时间更短，这类应用会受益很多。

There has been a significant amount of work on reducing the storage and computation costs by model compression (Le Cun et al. (1989); Hassibi & Stork (1993); Srinivas & Babu (2015); Han et al. (2015); Mariet & Sra (2016)). Recently Han et al. (2015; 2016b) report impressive compression rates on AlexNet (Krizhevsky et al. (2012)) and VGGNet (Simonyan & Zisserman (2015)) by pruning weights with small magnitudes and then retraining without hurting the overall accuracy. However, pruning parameters does not necessarily reduce the computation time since the majority of the parameters removed are from the fully connected layers where the computation cost is low, e.g., the fully connected layers of VGG-16 occupy 90% of the total parameters but only contribute less than 1% of the overall floating point operations (FLOP). They also demonstrate that the convolutional layers can be compressed and accelerated (Iandola et al. (2016)), but additionally require sparse BLAS libraries or even specialized hardware (Han et al. (2016a)). Modern libraries that provide speedup using sparse operations over CNNs are often limited (Szegedy et al. (2015a); Liu et al. (2015)) and maintaining sparse data structures also creates an additional storage overhead which can be significant for low-precision weights.

通过模型压缩，降低存储和计算资源，有很多工作。最近Han等在AlexNet和VGGNet上通过修剪小幅值权重，然后重新训练，得到了很好的压缩效果，而且还不损害总体准确率。但是，修剪参数并不一定带来计算时间的减少，因为去除的大量参数是在全连接层中，其计算时间不是很高，如VGG-16的全连接层占总参数量的90%，但只占总体FLOPs的不到1%。他们还证明了，卷积层也可以得到压缩和加速，但需要稀疏BLAS库，甚至专门的硬件。在CNNs上使用稀疏运算的现代库很少，维护稀疏数据结构也会带来额外的存储消耗，这对于低精度权重也不可忽视。

Recent work on CNNs have yielded deep architectures with more efficient design (Szegedy et al. (2015a;b); He & Sun (2015); He et al. (2016)), in which the fully connected layers are replaced with average pooling layers (Lin et al. (2013); He et al. (2016)), which reduces the number of parameters significantly. The computation cost is also reduced by downsampling the image at an early stage to reduce the size of feature maps (He & Sun (2015)). Nevertheless, as the networks continue to become deeper, the computation costs of convolutional layers continue to dominate.

最近的工作得到了更高效的CNNs架构设计，其中全连接层替换成了平均池化层，这大大降低了参数数量。在网络较早阶段就对图像进行下采样，降低了特征图的大小，从而计算代价也得到了降低。但是，由于网络还是持续变深，卷积层的计算代价一直占主要部分。

CNNs with large capacity usually have significant redundancy among different filters and feature channels. In this work, we focus on reducing the computation cost of well-trained CNNs by pruning filters. Compared to pruning weights across the network, filter pruning is a naturally structured way of pruning without introducing sparsity and therefore does not require using sparse libraries or any specialized hardware. The number of pruned filters correlates directly with acceleration by reducing the number of matrix multiplications, which is easy to tune for a target speedup. In addition, instead of layer-wise iterative fine-tuning (retraining), we adopt a one-shot pruning and retraining strategy to save retraining time for pruning filters across multiple layers, which is critical for pruning very deep networks. Finally, we observe that even for ResNets, which have significantly fewer parameters and inference costs than AlexNet or VGGNet, still have about 30% of FLOP reduction without sacrificing too much accuracy. We conduct sensitivity analysis for convolutional layers in ResNets that improves the understanding of ResNets.

大容量CNNs的滤波器和特征通道通常冗余都很多。在本文中，我们关注的是，通过对滤波器剪枝，降低训练好的CNNs的计算量。与在整个网络中修剪权重相比，滤波器剪枝是对自然结构的剪枝，不会引入稀疏性，因此不需要使用稀疏库或定制硬件。修剪滤波器的数量与加速直接相关，因为降低了矩阵乘积运算的数量，这很容易调节到目标的加速水平。另外，我们不是进行逐层的迭代精调（重新训练），而是采用的一步剪枝和重新训练策略，以降低多层滤波器剪枝的重新训练时间，这对于极深网络的剪枝是很关键的。最后，我们观察到，即使是对于ResNets，与AlexNet或VGGNet相比，其参数数量和推理代价少了很多，也可以降低30%的FLOPs，而准确率不会下降很多。我们对ResNets中的卷积层进行敏感度分析，也对ResNet得到了更深的理解。

## 2 Related Work 相关工作

The early work by LeCun et al. (1989) introduces Optimal Brain Damage, which prunes weights with a theoretically justified saliency measure. Later, Hassibi & Stork (1993) propose Optimal Brain Surgeon to remove unimportant weights determined by the second-order derivative information. Mariet & Sra (2016) reduce the network redundancy by identifying a subset of diverse neurons that does not require retraining. However, this method only operates on the fully-connected layers and introduce sparse connections.

LeCun等的早期提出了Optimal Brain Damage，使用一种理论上证明的显著性度量进行权重修剪。后来，Hassibi等提出Optimal Brain Surgeon，利用二阶导数信息来去除不重要的权重。Mariet等通过辨认出不需要重新训练的神经元子集，来去除网络的冗余性。但是，这种方法只在全连接层上进行，并且回带来稀疏连接。

To reduce the computation costs of the convolutional layers, past work have proposed to approximate convolutional operations by representing the weight matrix as a low rank product of two smaller matrices without changing the original number of filters (Denil et al. (2013); Jaderberg et al. (2014); Zhang et al. (2015b;a); Tai et al. (2016); Ioannou et al. (2016)). Other approaches to reduce the convolutional overheads include using FFT based convolutions (Mathieu et al. (2013)) and fast convolution using the Winograd algorithm (Lavin & Gray (2016)). Additionally, quantization (Han et al. (2016b)) and binarization (Rastegari et al. (2016); Courbariaux & Bengio (2016)) can be used to reduce the model size and lower the computation overheads. Our method can be used in addition to these techniques to reduce computation costs without incurring additional overheads.

为降低卷积层的计算消耗，过去的工作提出，通过将权重矩阵表示为两个更小矩阵的低秩乘积，不改变原始滤波器的数量，来近似卷积运算。其他降低卷积运算量的方法包括，使用基于FFT的卷积，使用Winograd算法的快速卷积。另外，量化和二值化也可以用于降低模型大小，降低计算消耗。我们的方法可以与这些方法一起使用，来降低计算代价，而不会带来额外的开销。

Several work have studied removing redundant feature maps from a well trained network (Anwar et al. (2015); Polyak & Wolf (2015)). Anwar et al. (2015) introduce a three-level pruning of the weights and locate the pruning candidates using particle filtering, which selects the best combination from a number of random generated masks. Polyak & Wolf (2015) detect the less frequently activated feature maps with sample input data for face detection applications. We choose to analyze the filter weights and prune filters with their corresponding feature maps using a simple magnitude based measure, without examining possible combinations. We also introduce network-wide holistic approaches to prune filters for simple and complex convolutional network architectures.

有几个工作研究了从训练好的网络移除冗余特征图。Anwar等提出了一种三级权重修剪过程，使用粒子滤波来定位候选剪枝，从几个随机生成的掩膜中选择最好的组合。Polyak等使用人脸检测用的简单输入数据，来检测变动不多的特征图。我们选择去分析滤波器权重，使用一种简单的基于幅度的度量，利用其对应的特征图来对滤波器剪枝，这样就不用去检查可能的组合。我们还提出，对简单和复杂的卷积网络架构，考虑网络整体的方法来对滤波器剪枝。

Concurrently with our work, there is a growing interest in training compact CNNs with sparse constraints (Lebedev & Lempitsky (2016); Zhou et al. (2016); Wen et al. (2016)). Lebedev & Lempitsky (2016) leverage group-sparsity on the convolutional filters to achieve structured brain damage, i.e., prune the entries of the convolution kernel in a group-wise fashion. Zhou et al. (2016) add group-sparse regularization on neurons during training to learn compact CNNs with reduced filters. Wen et al. (2016) add structured sparsity regularizer on each layer to reduce trivial filters, channels or even layers. In the filter-level pruning, all above work use $l_{2,1}$-norm as a regularizer.

与我们的工作同时，有很多工作在使用稀疏约束来训练紧凑的CNNs。Lebedev等利用卷积滤波器的分组稀疏性来得到有结构的大脑损伤，即以分组的形式来对卷积核的索引进行剪枝。Zhou等在训练过程中，对神经元增加分组稀疏性正则化约束，使用减少的滤波器来学习紧凑的CNNs。Wen等在每一层上增加结构化的稀疏性正则，来减少不重要的滤波器、通道，甚至是层。在滤波器级的剪枝中，所有上述工作都使用$l_{2,1}$范数作为正则化器。

Similar to the above work, we use $l_1$-norm to select unimportant filters and physically prune them. Our fine-tuning process is the same as the conventional training procedure, without introducing additional regularization. Our approach does not introduce extra layer-wise meta-parameters for the regularizer except for the percentage of filters to be pruned, which is directly related to the desired speedup. By employing stage-wise pruning, we can set a single pruning rate for all layers in one stage.

与上述工作类似，我们使用$l_1$范数来选择不重要的滤波器，并将其剪枝。我们的精调过程与传统训练过程相同，没有引入额外的正则化。我们的方法，没有为正则化器引入额外的分层的元参数，除了要剪枝的滤波器的百分比，这与期望的加速直接相关。通过采用分阶段的剪枝，我们对一个阶段中的所有层，设置一个单独的剪枝率。

## 3 Pruning Filters and Feature Maps 滤波器和特征图剪枝

Let $n_i$ denote the number of input channels for the i-th convolutional layer and $h_i/w_i$ be the height/width of the input feature maps. The convolutional layer transforms the input feature maps $x_i ∈ R^{n_i×h_i×w_i}$ into the output feature maps $x_{i+1} ∈ R^{n_{i+1}×h_{i+1}×w_{i+1}}$, which are used as input feature maps for the next convolutional layer. This is achieved by applying $n_{i+1}$ 3D filters $F_{i,j} ∈ R^{n_i×k×k}$ on the $n_i$ input channel, in which one filter generates one feature map. Each filter is composed by $n_i$ 2D kernels $K ∈ R^{k×k}$ (eg, 3×3). All the filters, together, constitute the kernel matrix $F_i ∈ R^{n_i×n_{i+1}×k×k}$. The number of operations of the convolutional layer is $n_{i+1} n_i k^2 h_{i+1} w_{i+1}$. As shown in Figure 1, when a filter F_{i,j} is pruned, its corresponding feature map $x_{i+1,j}$ is removed, which reduces $n_i k^2 h_{i+1} w_{i+1}$ operations. The kernels that apply on the removed feature maps from the filters of the next convolutional layer are also removed, which saves an additional $n_{i+2} k^2 h_{i+2} w_{i+2}$ operations. Pruning m filters of layer i will reduce $m/n_{i+1}$ of the computation cost for both layers i and i+1.

令$n_i$表示第i卷积层的输入通道数量，$h_i/w_i$是输入特征图的高度/宽度。卷积层将输入特征图$x_i ∈ R^{n_i×h_i×w_i}$，变换成输出特征图$x_{i+1} ∈ R^{n_{i+1}×h_{i+1}×w_{i+1}}$，然后又用作下一个卷积层的输入特征图。这是通过对$n_i$个输入通道，应用$n_{i+1}$个3D滤波器$F_{i,j} ∈ R^{n_i×k×k}$，一个滤波器生成一个特征图。每个滤波器是由$n_i$个2D滤波核$K ∈ R^{k×k}$组成（如3×3）。所有滤波器一起构成了核矩阵$F_i ∈ R^{n_i×n_{i+1}×k×k}$。卷积层的运算量为$n_{i+1} n_i k^2 h_{i+1} w_{i+1}$。如图1所示，当滤波器F_{i,j}被剪枝时，其对应的特征图$x_{i+1,j}$也被移除了，这就减少了$n_i k^2 h_{i+1} w_{i+1}$次运算。下一个卷积层的滤波器中，应用在这个移除的特征图上的滤波器，也被去除了，这就节省了$n_{i+2} k^2 h_{i+2} w_{i+2}$次运算。在第i层剪枝m个滤波器，对第i层和第i+1层来说，会减少$m/n_{i+1}$的计算消耗。

Figure 1: Pruning a filter results in removal of its corresponding feature map and related kernels in the next layer.

### 3.1 Determining Which Filters to Prune within a Single Layer

Our method prunes the less useful filters from a well-trained model for computational efficiency while minimizing the accuracy drop. We measure the relative importance of a filter in each layer by calculating the sum of its absolute weights $\sum|F_{i,j}|$, i.e., its $l_1$-norm $||F_{i,j}||_1$. Since the number of input channels, $n_i$, is the same across filters, $\sum|F_{i,j}|$ also represents the average magnitude of its kernel weights. This value gives an expectation of the magnitude of the output feature map. Filters with smaller kernel weights tend to produce feature maps with weak activations as compared to the other filters in that layer. Figure 2(a) illustrates the distribution of filters’ absolute weights sum for each convolutional layer in a VGG-16 network trained on the CIFAR-10 dataset, where the distribution varies significantly across layers. We find that pruning the smallest filters works better in comparison with pruning the same number of random or largest filters (Section 4.4). Compared to other criteria for activation-based feature map pruning (Section 4.5), we find $l_1$-norm is a good criterion for data-free filter selection.

我们的方法从训练好的模型中对用处不大的滤波器进行剪枝，以提高计算效率，同时使准确率下降幅度最小化。我们计算权重的绝对值之和$\sum|F_{i,j}|$，即其$l_1$-范数$||F_{i,j}||_1$，来衡量每一层中滤波器的相对重要性。由于输入通道数量$n_i$对不同的滤波器都是一样的，所以$\sum|F_{i,j}|$也代表了滤波器核权重的平均幅度。这个值给出了输出特征图的幅度期望。滤波器核的权重值越小，比本层其他滤波器相比，其生成的特征图激活值就越弱。图2(a)描述了，在CIFAR-10数据集上训练的VGG-16网络，其每一个卷积层的滤波器的权重绝对值和的分布，可以发现，不同的层分布变化很明显。我们发现，对最小的滤波器进行剪枝，比对随机值或最大的滤波器剪枝要更好（见4.4节）。与其他基于激活的特征图剪枝规则相比（见4.5节），我们发现$l_1$范数对于不用数据的滤波器选择是一个好的原则。

The procedure of pruning m filters from the ith convolutional layer is as follows: 从第i个卷积层剪枝m个滤波器的过程如下：

- For each filter $F_{i,j}$, calculate the sum of its absolute kernel weights $s_j = \sum_{l=1}^{n_i} \sum|K_l|$. 对每个滤波器，计算其核权重的绝对值和。
- Sort the filters by $s_j$. 根据其和对滤波器进行排序。
- Prune m filters with the smallest sum values and their corresponding feature maps. The kernels in the next convolutional layer corresponding to the pruned feature maps are also removed. 对和值最小的m个滤波器进行剪枝，以及其对应的特征图。下一个卷积层中，对应剪枝的特征图的核，也移除掉。
- A new kernel matrix is created for both the i-th and (i+1)-th layers, and the remaining kernel weights are copied to the new model. 对第i层和第i+1层生成新的核矩阵，剩余的核权重复制到新的模型中。

**Relationship to pruning weights**. Pruning filters with low absolute weights sum is similar to pruning low magnitude weights (Han et al. (2015)). Magnitude-based weight pruning may prune away whole filters when all the kernel weights of a filter are lower than a given threshold. However, it requires a careful tuning of the threshold and it is difficult to predict the exact number of filters that will eventually be pruned. Furthermore, it generates sparse convolutional kernels which can be hard to accelerate given the lack of efficient sparse libraries, especially for the case of low-sparsity.

**剪枝权重的关系**。对权重的绝对值和小的滤波器进行剪枝，与剪枝低幅度的权重类似。基于幅度的权重剪枝，当一个滤波器的所有核权重都低于给定阈值时，会将整个滤波器整个滤波器。但是，这需要仔细调节阈值，很难预测最终剪枝的滤波器的精确数量。而且，这会生成稀疏卷积核，由于缺少有效的稀疏库，这很难加速，尤其对于低稀疏度的情况。

**Relationship to group-sparse regularization on filters**. Recent work (Zhou et al. (2016); Wen et al. (2016)) apply group-sparse regularization ($sum\_{j=1}^{n_i}||F_{i,j}||_2$ or $l_{2,1}$-norm) on convolutional filters, which also favor to zero-out filters with small $l_2$-norms, i.e. $F_{i,j}$= 0. In practice, we do not observe noticeable difference between the $l_2$-norm and the $l_1$-norm for filter selection, as the important filters tend to have large values for both measures (Appendix 6.1). Zeroing out weights of multiple filters during training has a similar effect to pruning filters with the strategy of iterative pruning and retraining as introduced in Section 3.4.

**与对滤波器的分组稀疏正则化的关系**。最近的工作对卷积滤波器使用分组稀疏正则化($sum\_{j=1}^{n_i}||F_{i,j}||_2$ or $l_{2,1}$-norm)，也会倾向于将$l_2$范数很小的滤波器赋零去除，即$F_{i,j}$= 0。在实践中，我们没有观察到，使用$l_2$范数和$l_1$范数在选择滤波器时，有什么值得注意的差异，因为重要的滤波器在两种度量下都会有很大的值（见附录6.1）。在训练时对多个滤波器的权重赋零，与3.4节介绍的滤波器剪枝再重新训练的迭代策略是类似的。

### 3.2 Determining Single Layer's Sensitivity to Pruning

To understand the sensitivity of each layer, we prune each layer independently and evaluate the resulting pruned network’s accuracy on the validation set. Figure 2(b) shows that layers that maintain their accuracy as filters are pruned away correspond to layers with larger slopes in Figure 2(a). On the contrary, layers with relatively flat slopes are more sensitive to pruning. We empirically determine the number of filters to prune for each layer based on their sensitivity to pruning. For deep networks such as VGG-16 or ResNets, we observe that layers in the same stage (with the same feature map size) have a similar sensitivity to pruning. To avoid introducing layer-wise meta-parameters, we use the same pruning ratio for all layers in the same stage. For layers that are sensitive to pruning, we prune a smaller percentage of these layers or completely skip pruning them.

为理解每一层的敏感度，我们对每一层进行独立剪枝，评估得到的剪枝网络在验证集上的准确率。图2(b)说明，当剪枝掉滤波器后，准确率保持不变的层，对应着那些在图2(a)中斜率更大的层。相反的，较平缓的层则对于剪枝更敏感。我们通过经验确定每一层要剪枝的滤波器，主要基于其对剪枝的敏感度。对于VGG-16或ResNet这样的深度网络，我们观察到同一阶段的层（特征图大小相同的层），对于剪枝的敏感度是类似的。为防止引入分层的元参数，我们对同一阶段的所有层使用相同的剪枝率。对于那些剪枝敏感的层，我们剪枝的百分比更小一些，或者干脆不剪枝。

Figure 2: (a) Sorting filters by absolute weights sum for each layer of VGG-16 on CIFAR-10. The x-axis is the filter index divided by the total number of filters. The y-axis is the filter weight sum divided by the max sum value among filters in that layer. (b) Pruning filters with the lowest absolute weights sum and their corresponding test accuracies on CIFAR-10. (c) Prune and retrain for each single layer of VGG-16 on CIFAR-10. Some layers are sensitive and it can be harder to recover accuracy after pruning them.

### 3.3 Pruning Filters across Multiple Layers

We now discuss how to prune filters across the network. Previous work prunes the weights on a layer by layer basis, followed by iteratively retraining and compensating for any loss of accuracy (Han et al. (2015)). However, understanding how to prune filters of multiple layers at once can be useful: 1) For deep networks, pruning and retraining on a layer by layer basis can be extremely time-consuming 2) Pruning layers across the network gives a holistic view of the robustness of the network resulting in a smaller network 3) For complex networks, a holistic approach may be necessary. For example, for the ResNet, pruning the identity feature maps or the second layer of each residual block results in additional pruning of other layers.

我们现在讨论怎样在整个网络中剪枝滤波器。之前的工作在剪枝权重时是一层一层的，然后迭代进行重新训练，以补偿准确率损失。但是，理解怎样在多层中一次性剪枝滤波器是非常有用的：1)对于深度网络，一层一层剪枝并重新训练非常耗时；2)在整个网络中剪枝，可以得到网络变成小型网络后的稳健性的整体视角；3)对于复杂的网络，需要一个整体的方法。比如，对于ResNet，剪枝掉恒等特征图，或每个残差单元的第二层，会得到额外剪枝其他层的结果。

To prune filters across multiple layers, we consider two strategies for layer-wise filter selection: 对于在多层中剪枝滤波器，我们考虑分层滤波器选择的两种策略：

- Independent pruning determines which filters should be pruned at each layer independent of other layers. 独立剪枝，在每一层中独立确定哪些滤波器需要剪枝，与其他层无关。
- Greedy pruning accounts for the filters that have been removed in the previous layers. This strategy does not consider the kernels for the previously pruned feature maps while
calculating the sum of absolute weights. 贪婪剪枝，考虑到前面的层中已经移除的滤波器。这种策略在计算权重绝对值之和时，不会计算之前剪枝掉的特征图对应的核。

Figure 3 illustrates the difference between two approaches in calculating the sum of absolute weights. The greedy approach, though not globally optimal, is holistic and results in pruned networks with higher accuracy especially when many filters are pruned.

图3描述了两种方法在计算权重绝对值和时的差异。贪婪方法，虽然不是全局最有的，是整体的方法，可以得到更高准确率的剪枝网络，尤其在剪枝很多滤波器的时候。

Figure 3: Pruning filters across consecutive layers. The independent pruning strategy calculates the filter sum (columns marked in green) without considering feature maps removed in previous layer (shown in blue), so the kernel weights marked in yellow are still included. The greedy pruning strategy does not count kernels for the already pruned feature maps. Both approaches result in a $(n_{i+1} − 1) × (n_{i+2} − 1)$ kernel matrix.

For simpler CNNs like VGGNet or AlexNet, we can easily prune any of the filters in any convolutional layer. However, for complex network architectures such as Residual networks (He et al. (2016)), pruning filters may not be straightforward. The architecture of ResNet imposes restrictions and the filters need to be pruned carefully. We show the filter pruning for residual blocks with projection mapping in Figure 4. Here, the filters of the first layer in the residual block can be arbitrarily pruned, as it does not change the number of output feature maps of the block. However, the correspondence between the output feature maps of the second convolutional layer and the identity feature maps makes it difficult to prune. Hence, to prune the second convolutional layer of the residual block, the corresponding projected feature maps must also be pruned. Since the identical feature maps are more important than the added residual maps, the feature maps to be pruned should be determined by the pruning results of the shortcut layer. To determine which identity feature maps are to be pruned, we use the same selection criterion based on the filters of the shortcut convolutional layers (with 1 × 1 kernels). The second layer of the residual block is pruned with the same filter index as selected by the pruning of the shortcut layer.

对于更简单的CNNs，如VGGNet或AlexNet，我们可以容易的剪枝掉任何卷积层的任何滤波器。但是，对于像ResNet这样的复杂网络结构，滤波器剪枝可能没那么直接。ResNet的架构有很多限制，滤波器剪枝需要很小心。我们在图4中给出，残差模块的滤波器剪枝，在有投影映射下的情况。这里，残差模块第一层的滤波器，可以任意剪枝，因为不会改变模块输出特征图的数量。但是，第二个卷积层的输出特征图和恒等特征图的对应关系，使其很难剪枝。所以，为对残差模块的第二个卷积层进行剪枝，对应的投影特征图也必须剪枝。由于恒等特征图比相加的残差图更重要，所以需要剪枝的特征图，应当由捷径层的剪枝结果来确定。为确定哪个恒等特征图要剪枝，我们使用相同的选择原则，基于捷径卷积层的滤波器（1×1核）。残差模块的第二层剪枝的滤波器索引，与捷径层的剪枝索引是一样的。

Figure 4: Pruning residual blocks with the projection shortcut. The filters to be pruned for the second layer of the residual block (marked as green) are determined by the pruning result of the shortcut projection. The first layer of the residual block can be pruned without restrictions.

### 3.4 Retraining Pruned Networks to Regain Accuracy

After pruning the filters, the performance degradation should be compensated by retraining the network. There are two strategies to prune the filters across multiple layers: 在剪枝滤波器后，性能的下降由重新训练网络解决。有两种策略在多层中进行滤波器剪枝：

- Prune once and retrain: Prune filters of multiple layers at once and retrain them until the original accuracy is restored. 剪枝一次然后重新训练：一次性对多层进行剪枝并进行重新训练，直到恢复到原始准确率。
- Prune and retrain iteratively: Prune filters layer by layer or filter by filter and then retrain iteratively. The model is retrained before pruning the next layer for the weights to adapt to the changes from the pruning process. 剪枝和重新训练迭代进行：一层一层进行滤波器剪枝，或一个滤波器一个滤波器进行剪枝，然后重新训练，迭代进行。模型在对下一层进行剪枝前重新训练，使权重适应剪枝过程的变化。

We find that for the layers that are resilient to pruning, the prune and retrain once strategy can be used to prune away significant portions of the network and any loss in accuracy can be regained by retraining for a short period of time (less than the original training time). However, when some filters from the sensitive layers are pruned away or large portions of the networks are pruned away, it may not be possible to recover the original accuracy. Iterative pruning and retraining may yield better results, but the iterative process requires many more epochs especially for very deep networks.

我们发现，对于可以适应剪枝的层，剪枝和重新训练一次的策略可以用于剪枝掉相当一部分网络，任何分辨率损失都可以通过重新训练一小段时间恢复回来（少于原始训练时间）。但是，当一些敏感的层中，一些滤波器剪枝掉后，或者大部分网络都被剪枝掉后，可能就恢复不到原始准确率了。迭代剪枝和重新训练可能会得到更好的结果，但迭代过程通常需要非常多的时间，尤其是对于非常深的网络。

## 4 Experiments

We prune two types of networks: simple CNNs (VGG-16 on CIFAR-10) and Residual networks (ResNet-56/110 on CIFAR-10 and ResNet-34 on ImageNet). Unlike AlexNet or VGG (on ImageNet) that are often used to demonstrate model compression, both VGG (on CIFAR-10) and Residual networks have fewer parameters in the fully connected layers. Hence, pruning a large percentage of parameters from these networks is challenging. We implement our filter pruning method in Torch7 (Collobert et al. (2011)). When filters are pruned, a new model with fewer filters is created and the remaining parameters of the modified layers as well as the unaffected layers are copied into the new model. Furthermore, if a convolutional layer is pruned, the weights of the subsequent batch normalization layer are also removed. To get the baseline accuracies for each network, we train each model from scratch and follow the same pre-processing and hyper-parameters as ResNet (He et al. (2016)). For retraining, we use a constant learning rate 0.001 and retrain 40 epochs for CIFAR-10 and 20 epochs for ImageNet, which represents one-fourth of the original training epochs. Past work has reported up to 3× original training times to retrain pruned networks (Han et al. (2015)).

我们对两种类型的网络进行剪枝：简单CNNs(VGG-16 on CIFAR-10)和残差网络(ResNet-56/110 on CIFAR-10 and ResNet-34 on ImageNet)。AlexNet或VGG (on ImageNet)通常用于模型压缩的例子，与之不同的是，VGG (on CIFAR-10)和残差网络在全连接层的参数更少。所以，从这些网络中剪枝掉大部分参数是非常有挑战性的。我们在Torch7上实现我们的滤波器剪枝方法。当滤波器剪枝掉后，就产生了一个滤波器更少的新模型，剩下的参数和不受影响的层都赋值到新的模型中。而且，如果一个卷积层剪枝掉了，随后的BN层的参数也一样去除掉了。为对每个网络得到基准准确率，我们从头训练每个模型，使用与ResNet相同的预处理方法和超参数设置。对于重新训练，我们使用常数学习速率0.001，并在CIFAR-10上进行40轮的重新训练，在ImageNet上进行20轮的重新训练，这是原始训练轮数的1/4。过去的工作有的采用3x的原始训练时间来重新训练剪枝的网络。

### 4.1 VGG-16 on CIFAR-10

VGG-16 is a high-capacity network originally designed for the ImageNet dataset (Simonyan & Zisserman (2015)). Recently, Zagoruyko (2015) applies a slightly modified version of the model on CIFAR-10 and achieves state of the art results. As shown in Table 2, VGG-16 on CIFAR-10 consists of 13 convolutional layers and 2 fully connected layers, in which the fully connected layers do not occupy large portions of parameters due to the small input size and less hidden units. We use the model described in Zagoruyko (2015) but add Batch Normalization (Ioffe & Szegedy (2015)) layer after each convolutional layer and the first linear layer, without using Dropout (Srivastava et al. (2014)). Note that when the last convolutional layer is pruned, the input to the linear layer is changed and the connections are also removed.

VGG-16是高容量网络，开始设计时是用于ImageNet数据集的。最近，Zagoruyko等将一些修改模型用于CIFAR-10数据集，得到了目前最好的结果。如表2所示，VGG-16在CIFAR-10上包含13个卷积层和2个全连接层，其中全连接层不是参数的大部分，因为输入尺寸很小，隐藏单元也少。我们使用Zagoruyko描述的模型，但在每个卷积层和第一个线性层后加入了BN层，没有使用dropout。注意，当最后一个卷积层剪枝后，到线性层的输入有所变化，其连接也进行了移除。

Table 2: VGG-16 on CIFAR-10 and the pruned model. The last two columns show the number of feature maps and the reduced percentage of FLOP from the pruned model.

layer type | $w_i×h_i$ | Maps | FLOP | Params | Maps | FLOP%
--- | --- | --- | --- | --- | --- | ---
Conv 1 | 32×32 | 64 | 1.8E+06 | 1.7E+03 | 32 | 50%
Conv 2 | 32×32 | 64 | 3.8E+07 | 3.7E+04 | 64 | 50%
Conv 3 | 16×16 | 128 | 1.9E+07 | 7.4E+04 | 128 | 0%
Conv 4 | 16×16 | 128 | 3.8E+07 | 1.5E+05 | 128 | 0%
Conv 5 | 8×8 | 256 | 1.9E+07 | 2.9E+05 | 256 | 0%
Conv 6 | 8×8 | 256 | 3.8E+07 | 5.9E+05 | 256 | 0%
Conv 7 | 8×8 | 256 | 3.8E+07 | 5.9E+05 | 256 | 0%
Conv 8 | 4×4 | 512 | 1.9E+07 | 1.2E+06 | 256 | 50%
Conv 9 | 4×4 | 512 | 3.8E+07 | 2.4E+06 | 256 | 75%
Conv 10 | 4×4 | 512 | 3.8E+07 | 2.4E+06 | 256 | 75%
Conv 11 | 2×2 | 512 | 9.4E+06 | 2.4E+06 | 256 | 75%
Conv 12 | 2×2 | 512 | 9.4E+06 | 2.4E+06 | 256 | 75%
Conv 13 | 2×2 | 512 | 9.4E+06 | 2.4E+06 | 256 | 75%
Linear | 1 | 512 | 2.6E+05 | 2.6E+05 | 512 | 50%
Linear | 1 | 10 | 5.1E+03 | 5.1E+03 | 10 | 0%
Total | | | 3.1E+08 | 1.5E+07 | | 34%

As shown in Figure 2(b), each of the convolutional layers with 512 feature maps can drop at least 60% of filters without affecting the accuracy. Figure 2(c) shows that with retraining, almost 90% of the filters of these layers can be safely removed. One possible explanation is that these filters operate on 4 × 4 or 2 × 2 feature maps, which may have no meaningful spatial connections in such small dimensions. For instance, ResNets for CIFAR-10 do not perform any convolutions for feature maps below 8 × 8 dimensions. Unlike previous work (Zeiler & Fergus (2014); Han et al. (2015)), we observe that the first layer is robust to pruning as compared to the next few layers. This is possible for a simple dataset like CIFAR-10, on which the model does not learn as much useful filters as on ImageNet (as shown in Figure. 5). Even when 80% of the filters from the first layer are pruned, the number of remaining filters (12) is still larger than the number of raw input channels. However, when removing 80% filters from the second layer, the layer corresponds to a 64 to 12 mapping, which may lose significant information from previous layers, thereby hurting the accuracy. With 50% of the filters being pruned in layer 1 and from 8 to 13, we achieve 34% FLOP reduction for the same accuracy.

如图2(b)所示，每个卷积层有512个特征图的，可以至少丢弃60%的滤波器，而不损失准确率。图2(c)展示了，进行了重新训练后，这些层几乎90%的滤波器可以安全的被移除掉。一个可能的解释是，这些滤波器运算的对象是4×4或2×2的特征图，在这么小的维度上，不会有什么有意义的空间连接。比如，在CIFAR-10上的ResNet，对于分辨率低于8×8的特征图，进行任何卷积运算。与之前的工作不同，我们观察到，与下面几层相比，第一层对于剪枝是很稳健的。这对于简单数据集如CIFAR-10来说是可能的，其中模型可能不会学到很多有用的滤波器，但在ImageNet中就不一样（如图5所示）。即使第一层中80%的滤波器都剪枝掉，剩下的滤波器(12)也还是比输入通道数要多。但是，当剪枝掉80%的第二层滤波器时，这个层就对应着64与12的映射，可能会丢失掉上一层的很多信息，所以准确率会损失掉。第1层，和第8到13层中，50%的滤波器剪枝掉后，我们在相同的准确率下，得到了34%的FLOP降低。

Figure 5: Visualization of filters in the first convolutional layer of VGG-16 trained on CIFAR-10. Filters are ranked by $l_1$-norm.

### 4.2 ResNet-56/100 on CIFAR-10

ResNets for CIFAR-10 have three stages of residual blocks for feature maps with sizes of 32 × 32, 16 × 16 and 8 × 8. Each stage has the same number of residual blocks. When the number of feature maps increases, the shortcut layer provides an identity mapping with an additional zero padding for the increased dimensions. Since there is no projection mapping for choosing the identity feature maps, we only consider pruning the first layer of the residual block. As shown in Figure 6, most of the layers are robust to pruning. For ResNet-110, pruning some single layers without retraining even improves the performance. In addition, we find that layers that are sensitive to pruning (layers 20, 38 and 54 for ResNet-56, layer 36, 38 and 74 for ResNet-110) lie at the residual blocks close to the layers where the number of feature maps changes, e.g., the first and the last residual blocks for each stage. We believe this happens because the precise residual errors are necessary for the newly added empty feature maps.

在CIFAR-10上的ResNets有三个阶段的残差模块，特征图大小分别为32×32, 16×16和8×8。每个阶段有相同数量的残差模块。当特征图数量增加时，捷径层的恒等映射会对增加的维度有额外的补零。由于没有投影映射以选择恒等特征图，我们只考虑对残差模块的第一层进行剪枝。如图6所示，多数层对剪枝是稳健的。对于ResNet-110，对一些层进行剪枝，在没有重新训练的情况下，甚至性能有所改进。另外，我们发现，对剪枝敏感的层（ResNet-56的层20,38和54，ResNet-110的层36,38和74），在残差模块中的位置，与特征图数量变化的层非常接近，如每个阶段中的第一和最后一个残差模块。我们相信，这是因为，对于新增加的空特征图，需要精确的残差误差。

Figure 6: Sensitivity to pruning for the first layer of each residual block of ResNet-56/110.

The retraining performance can be improved by skipping these sensitive layers. As shown in Table 1, ResNet-56-pruned-A improves the performance by pruning 10% filters while skipping the sensitive layers 16, 20, 38 and 54. In addition, we find that deeper layers are more sensitive to pruning than layers in the earlier stages of the network. Hence, we use a different pruning rate for each stage. We use $p_i$ to denote the pruning rate for layers in the i-th stage. ResNet-56-pruned-B skips more layers (16, 18, 20, 34, 38, 54) and prunes layers with $p_1$ =60%, $p_2$ =30% and $p_3$ =10%. For ResNet-110, the first pruned model gets a slightly better result with $p_1$ =50% and layer 36 skipped. ResNet-110-pruned-B skips layers 36, 38, 74 and prunes with $p_1$ =50%, $p_2$ =40% and $p_3$ =30%. When there are more than two residual blocks at each stage, the middle residual blocks may be redundant and can be easily pruned. This might explain why ResNet-110 is easier to prune than ResNet-56.

重新训练的性能可以通过跳过敏感的层得到赶紧。如表1所示，ResNet-56-pruned-A通过剪枝10%的滤波器，同时跳过敏感的层16,20,38和54，从而改进了性能。另外，我们发现，更深的层对剪枝更敏感，网络早期阶段的层对剪枝反而不是很敏感。所以，我们对每个阶段使用不同的剪枝率。我们使用$p_i$来表示第i阶段的剪枝率。ResNet-56-pruned-B跳过了更多的层(16, 18, 20, 34, 38, 54)，剪枝率为$p_1$ =60%, $p_2$ =30% and $p_3$ =10%。对于ResNet-110，第一个剪枝的模型使用$p_1$ =50%并跳过了36层，得到了更好的结果。ResNet-110-pruned-B跳过了36,38,74层，并使用剪枝率$p_1$ =50%, $p_2$ =40% and $p_3$ =30%。当在每个阶段有多余2个残差模块时，中间的残差模块可能是冗余的，可以轻松的剪枝掉。这也解释了为什么ResNet-110比ResNet-56更容易剪枝。

### 4.3 ResNEt-34 on ILSVRC2012

ResNets for ImageNet have four stages of residual blocks for feature maps with sizes of 56 × 56, 28 × 28, 14 × 14 and 7 × 7. ResNet-34 uses the projection shortcut when the feature maps are down-sampled. We first prune the first layer of each residual block. Figure 7 shows the sensitivity of the first layer of each residual block. Similar to ResNet-56/110, the first and the last residual blocks of each stage are more sensitive to pruning than the intermediate blocks (i.e., layers 2, 8, 14, 16, 26, 28, 30, 32). We skip those layers and prune the remaining layers at each stage equally. In Table 1 we compare two configurations of pruning percentages for the first three stages: (A) $p_1$ =30%, $p_2$ =30%, $p_3$ =30%; (B) $p_1$ =50%, $p_2$ =60%, $p_3$ =40%. Option-B provides 24% FLOP reduction with about 1% loss in accuracy. As seen in the pruning results for ResNet-50/110, we can predict that ResNet-34 is relatively more difficult to prune as compared to deeper ResNets.

在ImageNet上的ResNet有四个阶段的残差模块，特征图大小分别为56 × 56, 28 × 28, 14 × 14和7 × 7。ResNet-34在特征图进行了下采样后，使用的是投影捷径。我们首先对每个残差模块的第一层进行剪枝。图7展示了每个残差模块第一层的敏感度。与ResNet-56/110类似的是，每个阶段第一个和最后一个残差模块，对剪枝更敏感，中间模块对剪枝更不敏感一些（即层2,8,14,16,26,28,30,32）。在每个阶段中，我们跳过这些层，对剩余的层进行剪枝。在表1中，我们比较了前三个阶段进行如下比重的剪枝的性能：(A)$p_1$ =30%, $p_2$ =30%, $p_3$ =30%;(B)$p_1$ =50%, $p_2$ =60%, $p_3$ =40%。选项B减低了24%的FLOPs，准确率下降不超过1%。观察ResNet-50/110的剪枝结果，我们可以预测，与更深的ResNets相比，ResNet-34更难剪枝一些。

We also prune the identity shortcuts and the second convolutional layer of the residual blocks. As these layers have the same number of filters, they are pruned equally. As shown in Figure 7(b), these layers are more sensitive to pruning than the first layers. With retraining, ResNet-34-pruned-C prunes the third stage with $p_3$ =20% and results in 7.5% FLOP reduction with 0.75% loss in accuracy. Therefore, pruning the first layer of the residual block is more effective at reducing the overall FLOP than pruning the second layer. This finding also correlates with the bottleneck block design for deeper ResNets, which first reduces the dimension of input feature maps for the residual layer and then increases the dimension to match the identity mapping.

我们还对恒等捷径和残差模块的第二卷积层进行剪枝。由于这些层的滤波器数量相同，所以它们的剪枝是一样的。如图7(b)所示，这些层比第一层更难剪枝。在重新训练的情况下，ResNet-34-pruned-C对第三个阶段进行$p_3$ =20%的剪枝，降低了7.5%的FLOPs，准确率损失0.75%。因此，与对第二层进行剪枝相比，对残差模块的第一层进行剪枝，降低FLOPs效率更高。这个发现也与更深的ResNets的瓶颈模块设计相关，在瓶颈模块中，首先降低输入特征图的维度，然后再提升维度，以和恒等映射匹配。

Figure 7: Sensitivity to pruning for the residual blocks of ResNet-34. (a) Pruning the first layer of residual blocks; (b) Pruning the second layer of residual blocks.

### 4.4 Comparison with Pruning Random Filters and Largest Filters

We compare our approach with pruning random filters and largest filters. As shown in Figure 8, pruning the smallest filters outperforms pruning random filters for most of the layers at different pruning ratios. For example, smallest filter pruning has better accuracy than random filter pruning for all layers with the pruning ratio of 90%. The accuracy of pruning filters with the largest $l_1$-norms drops quickly as the pruning ratio increases, which indicates the importance of filters with larger $l_1$-norms.

我们与随机滤波器剪枝和最大滤波器剪枝进行对比。如图8所示，对最小的滤波器进行剪枝，在不同剪枝率下，在大多数层中超过了随机滤波器剪枝。比如，在所有层的剪枝率在90%的情况下，最小滤波器剪枝比随机滤波器剪枝的准确率更高。最大$l_1$范数的滤波器剪枝，其准确率在剪枝率上升的时候迅速下降，这说明更大的$l_1$范数的滤波器重要性更高。

Figure 8: Comparison of three pruning methods for VGG-16 on CIFAR-10: pruning the smallest filters, pruning random filters and pruning the largest filters. In random filter pruning, the order of filters to be pruned is randomly permuted.

### 4.5 Comparison with Activation-Based Feature Map Pruning

The activation-based feature map pruning method removes the feature maps with weak activation patterns and their corresponding filters and kernels (Polyak & Wolf (2015)), which needs sample data as input to determine which feature maps to prune. A feature map $x_{i+1,j} ∈ R^{w_{i+1}×h_{i+1}}$ is generated by applying filter $F_{i,j} ∈ R^{n_i×k×k}$ to feature maps of previous layer $x_i ∈ R^{n_i×w_i×h_i}$, i.e., $x_{i+1,j} = F_{i,j} ∗ x_i$. Given N randomly selected images $\{x^n_1\}^N_{n=1}$ from the training set, the statistics of each feature map can be estimated with one epoch forward pass of the N sampled data. Note that we calculate statistics on the feature maps generated from the convolution operations before batch normalization or non-linear activation. We compare our $l_1$-norm based filter pruning with feature map pruning using the following criteria: $σ_{mean-mean} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N mean(x^n_{i,j})$, $σ_{mean-std} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N std(x^n_{i,j})$, $σ_{mean-l_1} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N ||x^n_{i,j}||_1$, $σ_{mean-l_2} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N ||x^n_{i,j}||_2$, and $σ_{var-l_2} (x_{i,j}) = var(\{||x^n_{i,j}||_2\}_{n=1}^N)$, where mean, std and var are standard statistics (average, standard deviation and variance) of the input. Here, $σ_{var-l_2}$ is the contribution variance of channel criterion proposed in Polyak & Wolf (2015), which is motivated by the intuition that an unimportant feature map has almost similar outputs for the whole training data and acts like an additional bias.

基于激活的特征图剪枝方法，移除弱激活模式的特征图，及其对应的滤波器和核，这需要样本数据作为输入，以确定哪个特征图进行剪枝。一个特征图$x_{i+1,j} ∈ R^{w_{i+1}×h_{i+1}}$是通过将滤波器$F_{i,j} ∈ R^{n_i×k×k}$应用到之前层的特征图$x_i ∈ R^{n_i×w_i×h_i}$得到的，即$x_{i+1,j} = F_{i,j} ∗ x_i$。从训练集随机选择N幅图像$\{x^n_1\}^N_{n=1}$，每个特征图的统计量可以通过N个采样数据的一轮前向过程估计得到。注意，我们计算统计量的特征图，是在卷积运算后，在BN或非线性激活前的。我们将我们的基于$l_1$范数的滤波器剪枝，与基于以下原则的特征图剪枝进行对比：$σ_{mean-mean} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N mean(x^n_{i,j})$, $σ_{mean-std} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N std(x^n_{i,j})$, $σ_{mean-l_1} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N ||x^n_{i,j}||_1$, $σ_{mean-l_2} (x_{i,j}) = \frac{1}{N} \sum_{n=1}^N ||x^n_{i,j}||_2$, and $σ_{var-l_2} (x_{i,j}) = var(\{||x^n_{i,j}||_2\}_{n=1}^N)$, 其中mean, std和var是输入数据的标准统计量（平均值，标准差和方差）。这里，$σ_{var-l_2}$是Polyak等提出的通道贡献方差准则，这是受到了下面的直觉启发，即，不重要的特征图对整个输入数据有类似的输出，其行为就像一个额外的偏置。

The estimation of the criteria becomes more accurate when more sample data is used. Here we use the whole training set (N = 50, 000 for CIFAR-10) to compute the statistics. The performance of feature map pruning with above criteria for each layer is shown in Figure 9. Smallest filter pruning outperforms feature map pruning with the criteria $σ_{mean-mean}$, $σ_{mean-l_1}$, $σ_{mean-l_2$ and $σ_{var-l_2}$. The $σ_{mean-std}$ criterion has better or similar performance to $l_1$-norm up to pruning ratio of 60%. However, its performance drops quickly after that especially for layers of conv_1, conv_2 and conv_3. We find $l_1$-norm is a good heuristic for filter selection considering that it is data free.

当使用更多样本数据的时候，准则的估计变得更加准确。这里我们使用整个训练集（对于CIFAR-10来说，N=50000）来计算统计量。使用上述准则对每一层进行特征图剪枝的性能，如图9所示。最小滤波器剪枝超过了使用准则$σ_{mean-mean}$, $σ_{mean-l_1}$, $σ_{mean-l_2$和$σ_{var-l_2}$的特征图剪枝。$σ_{mean-std}$准则的特征图剪枝，与$l_1$范数的最小滤波器剪枝在剪枝率60%下的性能相当或更好。但是，在对conv_1, conv_2和conv_3进行剪枝后，其性能迅速下降。我们发现$l_1$范数在选择滤波器上是很好的，考虑到没有使用任何数据。

Figure 9: Comparison of activation-based feature map pruning for VGG-16 on CIFAR-10.

## 5 Conclusions

Modern CNNs often have high capacity with large training and inference costs. In this paper we present a method to prune filters with relatively low weight magnitudes to produce CNNs with reduced computation costs without introducing irregular sparsity. It achieves about 30% reduction in FLOP for VGGNet (on CIFAR-10) and deep ResNets without significant loss in the original accuracy. Instead of pruning with specific layer-wise hayperparameters and time-consuming iterative retraining, we use the one-shot pruning and retraining strategy for simplicity and ease of implementation. By performing lesion studies on very deep CNNs, we identify layers that are robust or sensitive to pruning, which can be useful for further understanding and improving the architectures.

现代CNNs通常容量很高，训练和推理代价都很大。在本文中，我们提出一种剪枝滤波器的方法，即对低权重幅度的滤波器进行剪枝，生成的CNNs计算量降低，而且没有引入不规则的稀疏性。在VGGNet (on CIFAR-10)和深度ResNets上降低了30%的FLOPs，原始准确率没有明显下降。我们没有采用分层的含有超参数的剪枝和耗时的迭代重新训练，而是使用了一次性的剪枝和重新训练策略，更简洁也更易于实现。在非常深CNNs上进行的损伤研究，我们识别出了对于剪枝稳健或敏感的层，这对未来的研究和架构改进是有用的。