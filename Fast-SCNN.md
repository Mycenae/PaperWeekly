# Fast-SCNN: Fast Semantic Segmentation Network

Rudra PK Poudel et. al. Toshiba Research

## 0. Abstract

The encoder-decoder framework is state-of-the-art for offline semantic image segmentation. Since the rise in autonomous systems, real-time computation is increasingly desirable. In this paper, we introduce fast segmentation convolutional neural network (Fast-SCNN), an above real-time semantic segmentation model on high resolution image data (1024 × 2048px) suited to efficient computation on embedded devices with low memory. Building on existing two-branch methods for fast segmentation, we introduce our 'learning to downsample' module which computes low-level features for multiple resolution branches simultaneously. Our network combines spatial detail at high resolution with deep features extracted at lower resolution, yielding an accuracy of 68.0% mean intersection over union at 123.5 frames per second on Cityscapes. We also show that large scale pre-training is unnecessary. We thoroughly validate our metric in experiments with ImageNet pre-training and the coarse labeled data of Cityscapes. Finally, we show even faster computation with competitive results on subsampled inputs, without any network modifications.

编码器-解码器框架是目前最好的离线图像语义分割模型。由于自动驾驶的兴起，实时计算越来越受到欢迎。本文中，我们引入了快速分割CNN (Fast-SCNN)，一种在高分辨率图像数据(1024 × 2048px)上高于实时的语义分割模型，适用于在小内存的嵌入式设备上进行高效计算。在现有的双分支快速分割方法之上，我们提出了“学习降采样”模块，同时计算多分辨率分支的底层特征。我们的网络结合了高分辨率的空间细节，和提取出的低分辨率深度特征，在CityScapes上以123.5fps的速度得到了68.0%的mIoU。我们还展示了，大规模预训练是没必要的。我们以实验在ImageNet预训练和CityScapes的粗糙标记数据上彻底验证了我们的度量。最后，我们展示了在更低分辨率的输入上，更快的计算结果，性能也很有竞争力，网络结构没有任何改变。

## 1. Introduction

Fast semantic segmentation is particular important in real-time applications, where input is to be parsed quickly to facilitate responsive interactivity with the environment. Due to the increasing interest in autonomous systems and robotics, it is therefore evident that the research into real-time semantic segmentation has recently enjoyed significant gain in popularity [21, 34, 17, 25, 36, 20]. We emphasize, faster than real-time performance is in fact often necessary, since semantic labeling is usually employed only as preprocessing step of other time-critical tasks. Furthermore, real-time semantic segmentation on embedded devices (without access to powerful GPUs) may enable many additional applications, such as augmented reality for wearables.

快速语义分割在实时应用中是非常重要的，其中输入需要被快速解析，以与环境进行响应性的交互。由于对自动系统和机器人的兴趣越来越大，因此很明显，研究实时语义分割最近越来越多。我们估计，比实时性能更快通常是必要的，因此语义标记通常是一个预处理步骤，还有很多在时间上非常关键的任务。而且，在嵌入式设备上的实时语义分割（不用GPUs）会使很多其他应用成为可能，比如可穿戴设备上的增强现实。

We observe, in literature semantic segmentation is typically addressed by a deep convolutional neural network (DCNN) with an encoder-decoder framework [29, 2], while many runtime efficient implementations employ a two- or multi-branch architecture [21, 34, 17]. It is often the case that

我们观察到，在文献中，语义分割通常是用编码器-解码器框架的DCNN处理的，而运行时间高效的实现，则采用了双分支，或多分支架构。通常情况如下：

- a larger receptive field is important to learn complex correlations among object classes (i.e. global context), 学习目标类别间的复杂关联（即，全局上下文），更大的感受野是非常重要的；
- spatial detail in images is necessary to preserve object boundaries, and 图像中的空间细节，对保存目标边缘，是很必要的；
- specific designs are needed to balance speed and accuracy (rather than re-targeting classification DCNNs). 要平衡速度和准确率，需要专有的设计。

Specifically in the two-branch networks, a deeper branch is employed at low resolution to capture global context, while a shallow branch is setup to learn spatial details at full input resolution. The final semantic segmentation result is then provided by merging the two. Importantly since the computational cost of deeper networks is overcome with a small input size, and execution on full resolution is only employed for few layers, real-time performance is possible on modern GPUs. In contrast to the encoder-decoder framework, the initial convolutions at different resolutions are not shared in the two-branch approach. Here it is worth noting, the guided upsampling network (GUN) [17] and the image cascade network (ICNet) [36] only share the weights among the first few layers, but not the computation.

在双分支网络中，采用了一个更深的分支，得到低分辨率，以捕获全局上下文，而浅的一支，则用于在完整输入分辨率上，学习空间细节。最终的语义分割结果，是融合两个结果得到的。由于更深的网络的输入更小，在完整分辨率上的计算只在几层中，所以计算量得到了降低，在现代GPUs上的实时性能是可能的。与编码器-解码器架构形成对比的是，在不同分辨率上的初始卷积在双分支方法中是不共享的。这里值得说的是，导向上采样网络GUN和图像级联网络ICNet只在前几层共享权重，并不是在所有的计算中共享。

In this work we propose fast segmentation convolutional neural network Fast-SCNN, an above real-time semantic segmentation algorithm merging the two-branch setup of prior art [21, 34, 17, 36], with the classical encoder-decoder framework [29, 2] (Figure 1). Building on the observation that initial DCNN layers extract low-level features [35, 19], we share the computations of the initial layers in the two-branch approach. We call this technique learning to downsample. The effect is similar to a skip connection in the encoder-decoder model, but the skip is only employed once to retain runtime efficiency, and the module is kept shallow to ensure validity of feature sharing. Finally, our Fast-SCNN adopts efficient depthwise separable convolutions [30, 10], and inverse residual blocks [28].

在本文中，我们提出快速分割CNN (Fast-SCNN)，这是一个超越实时的语义分割算法，将之前工作的双分支设置，与经典的编码器-解码器框架结合起来了。我们观察到，初始的DCNN层提取的是底层特征，我们在双分支方法中共享初始层的计算。我们称这个技术为“学习下采样”。这个效果与编码器-解码器中的跳跃连接效果类似，但跳跃只采用了一次，来得到运行时间上的效率，这个模块保持很浅，以确保特征共享的有效性。最后，我们的Fast-SCNN采用了高效的逐层可分离卷积和逆残差模块。

Applied on Cityscapes [6], Fast-SCNN yields a mean intersection over union (mIoU) of 68.0% at 123.5 frames per second (fps) on a modern GPU (Nvidia Titan Xp (Pascal)) using full (1024×2048px) resolution, which is twice as fast as prior art i.e. BiSeNet (71.4% mIoU)[34].

Fast-SCNN在CityScapes上得到了68.0%的mIoU，速度123.5fps，平台为Nvidia Titan Xp (Pascal)，分辨率1024×2048px，比之前的BiSeNet快了2倍。

While we use 1.11 million parameters, most offline segmentation methods (e.g. DeepLab [4] and PSPNet [37]), and some real-time algorithms (e.g. GUN [17] and ICNet [36]) require much more than this. The model capacity of Fast-SCNN is kept specifically low. The reason is two-fold: (i) lower memory enables execution on embedded devices, and (ii) better generalisation is expected. In particular, pretraining on ImageNet [27] is frequently advised to boost accuracy and generality [37]. In our work, we study the effect of pre-training on the low capacity Fast-SCNN. Contradicting the trend of high-capacity networks, we find that results only insignificantly improve with pre-training or additional coarsely labeled training data (+0.5% mIoU on Cityscapes [6]). In summary our contributions are:

我们使用了1.11 million参数，多数离线分割方法（如DeepLab和PSPNet），和一些实时算法（如GUN和ICNet）的参数都比这个多很多。Fast-SCNN的模型容量非常低。原因有两个：(i)更低的内存可以在嵌入式设备上运行，(ii)可能会有更好的泛化能力。特别是，大家频繁推荐使用在ImageNet上的预训练来提升准确率和泛化能力。在我们的工作中，我们研究了预训练在低容量Fast-SCNN上的效果。与高容量网络的趋势相反，我们发现用预训练或额外粗糙标记的训练数据只对性能有很少的改进（在CityScapes上+0.5% mIoU）。总结起来，我们的贡献如下：

1. We propose Fast-SCNN, a competitive (68.0%) and above real-time semantic segmentation algorithm (123.5 fps) for high resolution images (1024 × 2048px). 我们提出了Fast-SCNN，这对于高分辨率图像来说是一个超过实时的语义分割算法，效果非常好。
   
2. We adapt the skip connection, popular in offline DCNNs, and propose a shallow learning to downsample module for fast and efficient multi-branch low-level feature extraction. 我们调整了离线DCNNs中的跳跃连接，提出了一个浅层学习来降采样的模块，可以快速高效的得到多分支的底层特征提取。
   
3. We specifically design Fast-SCNN to be of low capacity, and we empirically validate that running training for more epochs is equivalently successful to pretraining with ImageNet or training with additional coarse data in our small capacity network. 我们专门设计Fast-SCNN为低容量的，经验上我们验证了训练更多epochs，与用在ImageNet上的预训练，或在额外的粗糙标注数据上进行训练，对我们的小容量网络是等效的。

Moreover, we employ Fast-SCNN to subsampled input data, achieving state-of-the-art performance without the need for redesigning our network. 而且，我们对降采样的输入数据采用Fast-SCNN，得到了目前最好的性能，不需要重新设计网络。

## 2. Related Work

We discuss and compare semantic image segmentation frameworks with a particular focus on real-time execution with low energy and memory requirements [2, 20, 21, 36, 34, 17, 25, 18]. 我们讨论比较一下图像语义分割框架，特别关注低能耗低内存需求的实时模型。

### 2.1. Foundation of Semantic Segmentation

State-of-the-art semantic segmentation DCNNs combine two separate modules: the encoder and the decoder. The encoder module uses a combination of convolution and pooling operations to extract DCNN features. The decoder module recovers the spatial details from the sub-resolution features, and predicts the object labels (i.e. the semantic segmentation) [29, 2]. Most commonly, the encoder is adapted from a simple classification DCNN method, such as VGG [31] or ResNet [9]. In semantic segmentation, the fully connected layers are removed.

目前最好的语义分割DCNNs结合了两个分离的模块：编码器与解码器。编码器模块使用了卷积和池化的组合来提取DCNN特征。解码器模块从低分辨率的特征中恢复出空间细节，并预测目标标签（即，语义分割）。最常见的是，编码器是从简单的分类DCNN方法中改造过来了，如VGG或ResNet。在语义分割中，全连接层就移除掉了。

The seminal fully convolution network (FCN) [29] laid the foundation for most modern segmentation architectures. Specifically, FCN employs VGG [31] as encoder, and bilinear upsampling in combination with skip-connection from lower layers to recover spatial detail. U-Net [26] further exploited the spatial details using dense skip connections.

FCN为多数现代分割框架打下了基础。具体的，FCN采用VGG作为编码器，双线性上采样与从更低的层的跳跃连接一起，来恢复空间细节。U-Net用密集的跳跃连接来进一步利用了空间细节。

Later, inspired by global image-level context prior to DCNNs [13, 16], the pyramid pooling module of PSPNet [37] and atrous spatial pyramid pooling (ASPP) of DeepLab [4] are employed to encode and utilize global context.

后来，受到全局图像级的上下文的启发[13,16]，PSPNet采用金字塔池化模块，DeepLab采用了ASPP，来对全局上下文进行编码和利用。

Other competitive fundamental segmentation architectures use conditional random fields (CRF) [38, 3] or recurrent neural networks [32, 38]. However, none of them run in real-time.

其他有竞争力的基础分割框架，使用了CRF或RNN。但是，这些都没有达到实时运行的速度。

Similar to the object detection [23, 24, 15], speed became one important factor in image segmentation system design [21, 34, 17, 25, 36, 20]. Building on FCN, SegNet [2] introduced a joint encoder-decoder model and became one of the earliest efficient segmentation models. Following SegNet, ENet [20] also design an encoder-decoder with few layers to reduce the computational cost.

与目标检测类似，在图像分割系统设计中，速度越来越成为一个重要的因素。在FCN之上，SegNet提出了一个联合编码器-解码器模型，成为了最早的高效分割模型。随着SegNet，ENet也用几层构建了编码器-解码器，来降低计算代价。

More recently, two-branch and multi-branch systems were introduced. ICNet [36], ContextNet [21], BiSeNet [34] and GUN [17] learned global context with reduced-resolution input in a deep branch, while boundaries are learned in a shallow branch at full resolution.

最近提出了双分支和多分支系统。ICNet，ContextNet，BiSeNet和GUN用降低分辨率的输入在深分支学习了全局上下文，而在浅分支上在全分辨率下学习边缘。

However, state-of-the-art real-time semantic segmentation remains challenging, and typically requires high-end GPUs. Inspired by two-branch methods, Fast-SCNN incorporates a shared shallow network path to encode detail, while context is efficiently learned at low resolution (Figure 2).

但是，目前最好的实时语义分割仍然非常很有挑战，一般都需要高端GPUs。受双分支方法启发，Fast-SCNN利用了共享的浅网络路径来编码细节，而上下文在低分辨率上进行高效的学习（图2）。

### 2.2. Efficiency in DCNNs

The common techniques of efficient DCNNs can be divided into four categories: 高效DCNNs的常用技术可以分成以下四类：

**Depthwise Separable Convolutions**: MobileNet [10] decomposes a standard convolution into a depthwise convolution and a 1×1 pointwise convolution, together known as depthwise separable convolution. Such a factorization reduces the floating point operations and convolutional parameters, hence the computational cost and memory requirement of the model is reduced. MobileNet将标准卷积分解成分层卷积和1x1逐点卷积，一起称为逐层可分离卷积。这样一个分解降低了浮点数运算和卷积参数，因此模型的计算代价和内存要求得到了降低。

**Efficient Redesign of DCNNs**: Chollet [5] designed the Xception network using efficient depthwise separable convolution. MobileNet-V2 proposed inverted bottleneck residual blocks [28] to build an efficient DCNN for the classification task. ContextNet [21] used inverted bottleneck residual blocks to design a two-branch network for efficient real-time semantic segmentation. Similarly, [34, 17, 36] propose multi-branch segmentation networks to achieve real-time performance. Chollet[5]使用高效的逐层可分离卷积设计了Xception网络。MobileNet-V2提出了逆瓶颈残差模块，来对分类任务构建高效DCNN。ContextNet[21]使用逆瓶颈残差模块，设计了双分支网络，得到高效的实时语义分割。类似的，[34,17,36]提出了多分支分割网络，得到实时性能。

**Network Quantization**: Since floating point multiplications are costly compared to integer or binary operations, runtime can be further reduced using quantization techniques for DCNN filters and activation values [11, 22, 33]. 与整数或二值相乘相比，浮点相乘计算很耗时，对DCNN滤波器和激活值采用量化技术，可以进一步降低运行时间。

**Network Compression**: Pruning is applied to reduce the size of a pre-trained network, resulting in faster runtime, a smaller parameter set, and smaller memory footprint [21, 8, 14]. 对预训练网络使用剪枝，可以降低网络大小，得到更快的运行时间，更小的参数集，和更小的内存消耗。

Fast-SCNN relies heavily on depthwise separable convolutions and residual bottleneck blocks [28]. Furthermore we introduce a two-branch model that incorporates our learning to downsample module, allowing for shared feature extraction at multiple resolution levels (Figure 2). Note, even though the initial layers of the multiple branches extract similar features [35, 19], common two-branch approaches do not leverage this. Network quantization and network compression can be applied orthogonally, and is left to future work.

Fast-SCNN严重依赖于逐层可分离卷积和残差瓶颈模块。而且我们提出了一个双分支模型，将我们的“学习下采样”模块纳入到了一起，可以在多个分辨率层次共享特征提取（图2）。注意，即使是多分支的初始层提取的都是类似的特征，常见的双分支方法并没有利用这个。网络量化和网络压缩可以独立的应用，留在未来的工作进行研究。

### 2.3. Pre-training on Auxiliary Tasks

It is a common belief that pre-training on auxiliary tasks boosts system accuracy. Earlier works on object detection [7] and semantic segmentation [4, 37] have shown this with pre-training on ImageNet [27]. Following this trend, other real-time efficient semantic segmentation methods are also pre-trained on ImageNet [36, 34, 17]. However, it is not known whether pre-training is necessary on low-capacity networks. Fast-SCNN is specifically designed with low capacity. In our experiments we show that small networks do not get significant benefit from pre-training. Instead, aggressive data augmentation and more number of epochs provide similar results.

在辅助任务上进行预训练，通常认为会提升系统准确率。在目标检测和语义分割上的早期工作表明在ImageNet上的预训练是有效的。随着这个趋势，其他实时高效的语义分割方法也在ImageNet上进行了预训练。但是，在低容量网络上，预训练是否有必要，并不是很确定。Fast-SCNN是特意设计成低容量的。在我们的实验中，我们证明了，小型网络从预训练中受益其实非常小。而激进的数据扩增，和更多的训练epochs数量，会得到类似的结果。

## 3. Proposed Fast-SCNN

Fast-SCNN is inspired by the two-branch architectures [21, 34, 17] and encoder-decoder networks with skip connections [29, 26]. Noting that early layers commonly extract low-level features. We reinterpret skip connections as a learning to downsample module, enabling us to merge the key ideas of both frameworks, and allowing us to build a fast semantic segmentation model. Figure 1 and Table 1 present the layout of Fast-SCNN. In the following we discuss our motivation and describe our building blocks in more detail.

Fast-SCNN是受到双分支架构和带有跳跃连接的编码器-解码器网络启发得到的。我们将跳跃连接重新解释为一种“学习降采样”模块，使我们可以将两种框架的关键思想融合起来，使我们可以构建快速语义分析模型。图1和表1给出了Fast-SCNN的布局。下面我们讨论设计动机，更加详细的描述我们的构建模块。

### 3.1. Motivation

Current state-of-the-art semantic segmentation methods that run in real-time are based on networks with two branches, each operating on a different resolution level [21, 34, 17]. They learn global information from low-resolution versions of the input image, and shallow networks at full input resolution are employed to refine the precision of the segmentation results. Since input resolution and network depth are main factors for runtime, these two-branch approaches allow for real-time computation.

目前最好的实时语义分割方法，是基于双分支网络的，每个分支在不同的分辨率层次上运算。它们从输入图像的低分辨率版本中学习全局信息，采用了浅层网络利用完整分辨率图像来提炼分割结果的精度。由于输入分辨率和网络深度是运行时间的主要要素，这种双分支方法得以进行实时计算。

It is well known that the first few layers of DCNNs extract the low-level features, such as edges and corners [35, 19]. Therefore, rather than employing a two-branch approach with separate computation, we introduce learning to downsample, which shares feature computation between the low and high-level branch in a shallow network block.

大家都知道，DCNNs的前几层学习的是低层特征，比如边缘和角点。因此，我们没有采用双分支方法进行分别计算，而是引入了学习下采样模块，在低分辨率分支和浅层网络模块的高层分支之间共享特征计算。

### 3.2. Network Architecture

Our Fast-SCNN uses a learning to downsample module, a coarse global feature extractor, a feature fusion module and a standard classifier. All modules are built using depthwise separable convolution, which has become a key building block for many efficient DCNN architectures [5, 10, 21].

我们的Fast-SCNN使用了学习降采样模块，粗糙全局特征提取器，特征融合模块和标准的分类器。所有模块都是使用逐层可分离卷积构建的，这是很多高效DCNN架构的关键构建模块。

#### 3.2.1 Learning to Downsample

In our learning to downsample module, we employ three layers. Only three layers are employed to ensure low-level feature sharing is valid, and efficiently implemented. The first layer is a standard convolutional layer (Conv2D) and the remaining two layers are depthwise separable convolutional layers (DSConv). Here we emphasize, although DSConv is computationally more efficient, we employ Conv2D since the input image only has three channels, making DSConv’s computational benefit insignificant at this stage.

在我们的学习降采样模块中，我们用了三层。只利用了三层来确保低层特征共享是有效的，而且进行了高效的实现。第一层是标准卷积层(Conv2D)，剩下的两层是逐层可分离卷积层(DSConv)。这里我们强调，虽然DSConv计算上很高效，我们还是采用了Conv2D，因为输入图像只有三通道，使DSConv在计算优势在这个阶段并不是很明显。

All three layers in our learning to downsample module use stride 2, followed by batch normalization [12] and ReLU. The spatial kernel size of the convolutional and depthwise layers is 3 × 3. Following [5, 28, 21], we omit the nonlinearity between depthwise and pointwise convolutions.

在我们的学习下采样模块中，所有三层都使用步长2，然后是批归一化和ReLU。逐层和卷积的空间核大小为3x3。按照[5,28,21]，我们忽略了逐层和逐点卷积的非线性性。

#### 3.2.2 Global Feature Extractor

The global feature extractor module is aimed at capturing the global context for image segmentation. In contrast to common two-branch methods which operate on low-resolution versions of the input image, our module directly takes the output of the learning to downsample module (which is at 1/8-resolution of the original input). The detailed structure of the module is shown in Table 1. We use efficient bottleneck residual block introduced by MobileNet-V2 [28] (Table 2). In particular, we employ residual connection for the bottleneck residual blocks when the input and output are of the same size. Our bottleneck block uses an efficient depthwise separable convolution, resulting in less number of parameters and floating point operations. Also, a pyramid pooling module (PPM) [37] is added at the end to aggregate the different-region-based context information.

全局特征提取器的目标是，捕获全局上下文进行图像分割。常见的双分支方法是在输入图像的低分辨率版本上计算，我们的模块则直接用学习下采样模块的输出（是原始输入的1/8分辨率）。模块的详细结构如表1所示。我们使用MobileNet-V2提出的高效瓶颈残差模块（表2）。特别是，当输入和输出大小相同时，我们对瓶颈残差模块采用了残差连接。我们的瓶颈模块使用了高效的逐层可分离卷积，得到了很少的参数和浮点运算。同时，尾端加上了金字塔池化模块PPM，以聚积基于不同区域的上下文信息。

#### 3.2.3 Feature Fusion Module

Similar to ICNet [36] and ContextNet [21] we prefer simple addition of the features to ensure efficiency. Alternatively, more sophisticated feature fusion modules (e.g. [34]) could be employed at the cost of runtime performance, to reach better accuracy. The detail of the feature fusion module is shown in Table 3.

与ICNet和ContextNet类似，我们倾向于特征的简单相加，以确保高效。另外，可以采用更复杂的特征融合模块以得到更好的准确率，但代价是运行性能可能会下降。特征融合模块的细节如表3所示。

#### 3.2.4 Classifier

In the classifier we employ two depthwise separable convolutions (DSConv) and one pointwise convolution (Conv2D). We found that adding few layers after the feature fusion module boosts the accuracy. The details of the classifier module is shown in the Table 1.

在分类器中，我们采用了两个逐层可分离卷积(DSConv)和一个逐点卷积(Conv2D)。我们发现在特征融合模块后加上几层，会提升准确率。分类器模块的细节如表1所示。

Softmax is used during training, since gradient decent is employed. During inference we may substitute costly softmax computations with argmax, since both functions are monotonically increasing. We denote this option as Fast-SCNN cls (classification). On the other hand, if a standard DCNN based probabilistic model is desired, softmax is used, denoted as Fast-SCNN prob (probability).

训练时会使用softmax，因为采用了梯度下降。在推理时，我们将昂贵的softmax计算替换成argmax，因为两个函数都是单调递增的。我们将这个选项表示为Fast-SCNN cls。另外，如果期望用标准的基于概率模型的DCNN，那么就使用softmax，表示为Fast-SCNN prob。

### 3.3. Comparison with Prior Art

Our model is inspired by the two-branch framework, and incorporates ideas of encoder-decorder methods (Figure 2). 我们的模型是受到双分支框架的启发，并结合了编码器-解码器方法的思想。

#### 3.3.1 Relation with Two-branch Models

The state-of-the-art real-time models (ContextNet [21], BiSeNet [34] and GUN [17]) use two-branch networks. Our learning to downsample module is equivalent to their spatial path, as it is shallow, learns from full resolution, and is used in the feature fusion module (Figure 1).

目前最好的实时模型(ContextNet [21], BiSeNet [34] and GUN [17])使用的都是双分支网络。我们的学习下采样模块，与其空间路径是等价的，因为是很浅的，从完整分辨率学习得到的，在特征融合模块进行了使用（图1）。

Our global feature extractor module is equivalent to the deeper low-resolution branch of such approaches. In contrast, our global feature extractor shares its computation of the first few layers with the learning to downsample module. By sharing the layers we not only reduce computational complexity of feature extraction, but we also reduce the required input size as Fast-SCNN uses 1/8-resolution instead of 1/4-resolution for global feature extraction.

我们的全局特征提取器模块，等价于这些方法中的更深的低分辨率分支。比较起来，我们的全局特征提取器，与学习降采样模块，共享前几层的计算。通过共享层，我们不仅降低了特征提取的计算复杂度，而且我们还降低了输入的大小，Fast-SCNN使用的是1/8分辨率，而不是1/4分辨率进行的全局特征提取。

#### 3.3.2 Relation with Encoder-Decoder Models

Proposed Fast-SCNN can be viewed as a special case of an encoder-decoder framework, such as FCN [29] or U-Net [26]. However, unlike the multiple skip connections in FCN and the dense skip connections in U-Net, Fast-SCNN only employs a single skip connection to reduce computations as well as memory.

提出的Fast-SCNN可以视为，编码器-解码器框架（如FCN或U-Net）的一种特殊情况。但是，与FCN中的多个跳跃连接，和U-Net中的密集跳跃连接不同，Fast-SCNN只用了一个跳跃连接，来降低计算量和内存消耗。

In correspondence with [35], who advocate that features are shared only at early layers in DCNNs, we position our skip connection early in our network. In contrast, prior art typically employ deeper modules at each resolution, before skip connections are applied.

[35]支持在DCNNs中只共享前几层的特征，与之对应，我们将跳跃连接放在网络的很早期的层中。与之相比，之前的工作在每个分辨率上会采用更深的模块，然后才使用跳跃连接。

## 4. Experiments

We evaluated our proposed fast segmentation convolutional neural network (Fast-SCNN) on the validation set of the Cityscapes dataset [6], and report its performance on the Cityscapes test set, i.e. the Cityscapes benchmark server.

我们在CityScapes数据集上评估了提出的Fast-SCNN，在CityScapes测试集上给出其性能，即，CityScapes基准测试服务器。

### 4.1. Implementation Details

Implementation detail is as important as theory when it comes to efficient DCNNs. Hence, we carefully describe our setup here. We conduct experiments on the TensorFlow machine learning platform using Python. Our experiments are executed on a workstation with either Nvidia Titan X (Maxwell) or Nvidia Titan Xp (Pascal) GPU, with CUDA 9.0 and CuDNN v7. Runtime evaluation is performed in a single CPU thread and one GPU to measure the forward inference time. We use 100 frames for burn-in and report average of 100 frames for the frames per second (fps) measurement.

对于高效DCNNs，实现细节是理论一样重要。因此，我们仔细描述了我们的设置。我们用Python+TensorFlow进行试验。我们的实验在Nvidia Titan X (Maxwell) 或 Nvidia Titan Xp (Pascal) GPU上进行，CUDA 9.0和CuDNN v7。运行时间评估是在单CPU线程和一个GPU上进行的，衡量的是前向推理时间。我们使用100帧进行测试，fps是100帧速度的平均。

We use stochastic gradient decent (SGD) with momentum 0.9 and batch-size 12. Inspired by [4, 37, 10] we use 'poly' learning rate with the base one as 0.045 and power as 0.9. Similar to MobileNet-V2 we found that ℓ2 regularization is not necessary on depthwise convolutions, for other layers ℓ2 is 0.00004. Since training data for semantic segmentation is limited, we apply various data augmentation techniques: random resizing between 0.5 to 2, translation/crop, horizontal flip, color channels noise and brightness. Our model is trained with cross-entropy loss. We found that auxiliary losses at the end of learning to downsample and the global feature extraction modules with 0.4 weights are beneficial.

我们使用SGD动量0.9，批大小12。受[4,37,10]启发，我们使用poly学习速率，基础为0.045，幂次为0.9。与MobileNet-V2类似，我们发现ℓ2正则化在逐层卷积上并不必须，对于其他层ℓ2是0.00004。由于语义分割的训练数据是有限的，我们使用各种数据扩增技术：在0.5和2之间随机改变大小，平移、剪切，水平翻转，色彩通道噪声和亮度。我们的模型使用交叉熵损失进行训练。我们发现，在学习降采样尾处和全局特征提取处的辅助损失，辅以0.4的权重，是有好处的。

Batch normalization [12] is used before every non-linear function. Dropout is used only on the last layer, just before the softmax layer. Contrary to MobileNet [10] and ContextNet [21], we found that Fast-SCNN trains faster with ReLU and achieves slightly better accuracy than ReLU6, even with the depthwise separable convolutions that we use throughout our model.

批归一化[12]在每个非线性函数之前都使用。Dropout只在最后一层使用，即softmax之前。与MobileNet和ContextNet形成对比的是，我们发现Fast-SCNN用ReLU训练的更快，比ReLU6得到略微更好的准确率，即使我们在整个模型中一直使用逐层可分离卷积。

We found that the performance of DCNNs can be improved by training for higher number of iterations, hence we train our model for 1,000 epochs unless otherwise stated, using the Cityescapes dataset [6]. It is worth noting here, Fast-SCNN's capacity is deliberately very low, as we employ 1.11 million parameters. Later we show that aggressive data augmentation techniques make overfitting unlikely.

我们发现，DCNNs的性能可以通过训练更多次迭代得到改进，因此我们在Cityscapes数据集上训练模型1000 epochs，除非另外说明。值得说明的是，Fast-SCNN的容量是故意设计的很低，因为我们只用了1.11 million参数。后来我们证明了，激进的数据扩增会使过拟合不太可能。

### 4.2. Evaluation on Cityscapes

We evaluate our proposed Fast-SCNN on Cityscapes, the largest publicly available dataset on urban roads [6]. This dataset contains a diverse set of high resolution images (1024×2048px) captured from 50 different cities in Europe. It has 5,000 images with high label quality: a training set of 2,975, validation set of 500 and test set of 1,525 images. The label for the training set and validation set are available and test results can be evaluated on the evaluation server. Additionally, 20,000 weakly annotated images (coarse labels) are available for training. We report results with both, fine only and fine with coarse labeled data. Cityscapes provides 30 class labels, while only 19 classes are used for evaluation. The mean of intersection over union (mIoU), and network inference time are reported in the following.

我们在Cityscapes上评估我们提出的Fast-SCNN，这是最大的公开可用的城市道路数据集。这个数据集大量高分辨率图像(1024×2048px)，是在欧洲50个不同的城市采集的。有5000幅很高标注质量的图像：训练集2975，验证集500，测试集1525。训练集和验证集的标签是可用的，测试结果是在评估服务器上评估得到的。另外，20000弱标记的图像（粗糙标签）可用于训练。我们给出只用精细标注的图像，和精细和粗糙标注的图像一起的两种结果。Cityscapes给出了30个类别标签，但只有19个类别用于评估。后面给出mIoU的均值，和网络推理时间。

We evaluate overall performance on the withheld test set of Cityscapes [6]. The comparison between the proposed Fast-SCNN and other state-of-the-art real-time semantic segmentation methods (ContextNet [21], BiSeNet [34], GUN [17], ENet [20] and ICNet [36]) and offline methods (PSPNet [37] and DeepLab-V2 [4]) is shown in Table 4. Fast-SCNN achieves 68.0% mIoU, which is slightly lower than BiSeNet (71.5%) and GUN (70.4%). ContextNet only achieves 66.1% here.

我们在Cityscapes的保留测试集上评估总体性能。提出的Fast-SCNN和其他目前最好的实时语义分割方法(ContextNet [21], BiSeNet [34], GUN [17], ENet [20] and ICNet [36])和离线方法(PSPNet [37] and DeepLab-V2 [4])进行了比较，如表4所示。Fast-SCNN得到了68.0%的mIoU，比BiSeNet (71.5%) and GUN (70.4%)略低。ContextNet只得到了66.1%的准确率。

Table 5 compares runtime at different resolutions. Here, BiSeNet (57.3 fps) and GUN (33.3 fps) are significantly slower than Fast-SCNN (123.5 fps). Compared to ContextNet (41.9 fps), Fast-SCNN is also significantly faster on Nvidia Titan X (Maxwell). Therefore we conclude, Fast-SCNN significantly improves upon state-of-the-art runtime with minor loss in accuracy. At this point we emphasize, our model is designed for low memory embedded devices. Fast-SCNN uses 1.11 million parameters, that is five times less than the competing BiSeNet at 5.8 million.

表5比较了不同分辨率的运行时间。这里，BiSeNet (57.3 fps) 和 GUN (33.3 fps)比Fast-SCNN (123.5 fps)要慢的多。与ContextNet (41.9 fps)相比，Fast-SCNN也快了很多。因此我们得出结论，Fast-SCNN明显改进了目前最好的运行时间，准确率的下降很小。在这一点上我们强调，我们的模型是设计用于低内存嵌入式设备的。Fast-SCNN只有1.11 million参数，比SiSeNet的5.8 million少了5倍。

Finally, we zero-out the contribution of the skip connection and measure Fast-SCNN's performance. The mIoU reduced from 69.22% to 64.30% on the validation set. The qualitative results are compared in Figure 3. As expected, Fast-SCNN benefits from the skip connection, especially around boundaries and objects of small size.

最后，我们将跳跃连接的贡献清零，衡量Fast-SCNN的性能。在验证集上mIoU从69.22%下降到了64.30%。图3比较了定性的结果。就像期望一样，Fast-SCNN从跳跃连接中受益，尤其是在很小的目标处和边缘处。

### 4.3. Pre-training and Weakly Labeled Data

High capacity DCNNs, such as R-CNN [7] and PSPNet [37], have shown that performance can be boosted with pretraining through different auxiliary tasks. As we specifically design Fast-SCNN to have low capacity, we now want to test performance with and without pre-training, and in connection with and without additional weakly labeled data. To the best of our knowledge, the significance of pre-training and additional weakly labeled data on low capacity DCNNs has not been studied before. Table 6 shows the results.

高容量DCNNs，比如R-CNN和PSPNet，可以通过不同的辅助任务的预训练，来提高性能。我们专门讲Fast-SCNN设计为低容量的，所以我们想测试一下有没有预训练的性能差异，以及有没有弱标注图像的关系。据我们所知，预训练和额外的弱标签的在低容量DCNNs上显著性，还没有被研究过。表6给出了结果。

We pre-train Fast-SCNN on ImageNet [27] by replacing the feature fusion module with average pooling and the classification module now has a softmax layer only. Fast-SCNN achieves 60.71% top-1 and 83.0% top-5 accuracies on the ImageNet validation set. This result indicates that Fast-SCNN has insufficient capacity to reach comparable performance to most standard DCNNs on ImageNet (>70% top-1) [10, 28]. The accuracy of Fast-SCNN with ImageNet pre-training yields 69.15% mIoU on the validation set of Cityscapes, only 0.53% improvement over Fast-SCNN without pre-training. Therefore we conclude, no significant boost can be achieved with ImageNet pre-training in Fast-SCNN.

我们在ImageNet上预训练Fast-SCNN，将特征融合模块替换为平均池化，分类模块现在只有一个softmax层。Fast-SCNN在ImageNet验证集上获得了60.71%的top-1和83.0%的top-5准确率。这个结果表明，Fast-SCNN的能力不足以达到ImageNet上的多数标准DCNNs的性能(>70% top-1)。在ImageNet预训练下的Fast-SCNN的准确率，在Cityscapes验证集上达到了69.15% mIoU，比没有预训练的只改进了0.53%。因此我们得出结论，Fast-SCNN在ImageNet预训练上没有得到显著的性能提升。

Since the overlap between Cityscapes' urban roads and ImageNet's classification task is limited, it is reasonable to assume that Fast-SCNN may not benefit due to limited capacity for both domains. Therefore, we now incorporate the 20,000 coarsely labeled additional images provided by Cityscapes, as these are from a similar domain. Nevertheless, Fast-SCNN trained with coarse training data (with or without ImageNet) perform similar to each other, and only slightly improve upon the original Fast-SCNN without pretraining. Please note, small variations are insignificant and due to random initializations of the DCNNs.

由于Cityscapes的城市道路，和ImageNet的分类任务的重叠很有限，所以假设Fast-SCNN不会受益很多，因为在两个领域能力都有限，这是合理的。因此，我们现在纳入了Cityscapes的20000幅弱标注的图像，因为这是类似领域的。尽管这样，用粗糙训练数据训练得到的Fast-SCNN（有或没有ImageNet）性能也是类似的，只比原始的没有预训练的Fast-SCNN改进了一点点。要注意，一些很小的变化是不明显的，因为DCNNs的随机初始化。

It is worth noting here that working with auxiliary tasks is non-trivial as it requires architectural modifications in the network. Furthermore, licence restrictions and lack of resources further limit such setups. These costs can be saved, since we show that neither ImageNet pre-training nor weakly labeled data are significantly beneficial for our low capacity DCNN. Figure 4 shows the training curves. Fast-SCNN with coarse data trains slow in terms of iterations because of the weak label quality. Both ImageNet pre-trained versions perform better for early epochs (up to 400 epochs for training set alone, and 100 epochs when trained with the additional coarse labeled data). This means, we only need to train our model for longer to reach similar accuracy when we train our model from scratch.

值得注意的是，使用辅助任务并不是无意义的，因为这需要网络架构进行修改。而且，许可的限制和缺少资源进一步限制了这样的设置。这些代价可以节约出来，因为我们展示了，ImageNet预训练和弱标记数据都不会使我们的低容量DCNN受益很多。图4展示了训练曲线。带有粗糙数据训练的Fast-SCNN收敛变慢了，因为标记质量很弱。两个ImageNet预训练的版本在早期epochs里表现更好一些。这意味着，当我们从头训练模型时，我们只需要多训练一段时间模型，就可以得到类似的准确率。

### 4.4. Lower Input Resolution

Since we are interested in embedded devices that may not have full resolution input, or access to powerful GPUs, we conclude our evaluation with the study of performance at half, and quarter input resolutions (Table 7).

由于我们对嵌入式设备很感兴趣，可能不会进行完整分辨率输入，或不会有很强的GPUs，我们用半分辨率和1/4分辨率评估一下性能。

At quarter resolution, Fast-SCNN achieves 51.9% accuracy at 485.4 fps, which significantly improves on (anonymous) MiniNet with 40.7% mIoU at 250 fps [6]. At half resolution, a competitive 62.8% mIoU at 285.8 fps is reached. We emphasize, without modification, Fast-SCNN is directly applicable to lower input resolution, making it highly suitable for embedded devices.

在1/4分辨率下，Fast-SCNN以485.4 fps获得了51.9%的准确率，而MiniNet以250 fps只获得了40.7% mIoU，有了显著改进。在半分辨率下，以285.8 fps的速度达到了62.8% mIoU。我们推测，在没有变化的情况下，Fast-SCNN可以直接应用到更低的输入分辨率上，使其非常适合嵌入式设备。

## 5. Conclusions

We propose a fast segmentation network for above real-time scene understanding. Sharing the computational cost of the multi-branch network yields run-time efficiency. In experiments our skip connection is shown beneficial for recovering the spatial details. We also demonstrate that if trained for long enough, large-scale pre-training of the model on an additional auxiliary task is not necessary for the low capacity network.

我们提出了一种快速分割网络，得到了超过实时的场景理解模型。多分支网络的共享计算代价，得到了高效的运行时间。在实验中，我们的跳跃连接证明是有好处的，可以恢复空间细节。我们还证明了，如果训练的时间足够长，在辅助任务上的大规模预训练，对于低容量网络并不是必须的。