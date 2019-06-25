# Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

Liang-Chieh Chen et al. Google Inc.

## Abstract 摘要

Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasets, achieving the test set performance of 89.0% and 82.1% without any post-processing. Our paper is accompanied with a publicly available reference implementation of the proposed models in Tensorflow at https://github.com/tensorflow/models/tree/master/research/deeplab.

空间金字塔池化模块或编码器-解码器结构在DNN中用于语义分割任务。前者的网络捕获多尺度上下文信息的方法是，用多孔洞率和多种有效视野的滤波器或池化运算检测特征；后者的网络可以捕获更清晰的目标边缘，方法是逐渐恢复空域信息。在本文中，我们提出结合这两种方法的优势。具体的，我们提出的模型，称为DeepLabV3+，对DeepLabV3进行了拓展，增加了一个简单却有效的解码器模块，来提炼分割结果，尤其是沿着目标边缘方向。我们进一步探索了Xception模型，将depthwise separable卷积应用于ASPP和解码器模块，得到了一种更快更强的编码器-解码器网络。我们在PASCAL VOC 2012和Cityscape数据集上证明了提出模型的有效性，在测试集上分别得到了89.0%和82.1%的性能，没有任何后处理。文章提出的模型已经开源。

**Keywords**: Semantic image segmentation, spatial pyramid pooling, encoder-decoder, depthwise separable convolution.

## 1 Introduction 引言

Semantic segmentation with the goal to assign semantic labels to every pixel in an image [1,2,3,4,5] is one of the fundamental topics in computer vision. Deep convolutional neural networks [6,7,8,9,10] based on the Fully Convolutional Neural Network [8,11] show striking improvement over systems relying on hand-crafted features [12,13,14,15,16,17] on benchmark tasks. In this work, we consider two types of neural networks that use spatial pyramid pooling module [18,19,20] or encoder-decoder structure [21,22] for semantic segmentation, where the former one captures rich contextual information by pooling features at different resolution while the latter one is able to obtain sharp object boundaries.

语义分割的目的是为图像中的每个像素指定语义标签，这是计算机视觉中的一个基本课题。基于全卷积网络的DCNNs，比依赖手工设计特征的系统的改进非常显著。在本文中，我们考虑两种神经网络用于语义分割，即使用空间金字塔池化模块的，或编码器-解码器结构的模块，其中前者通过池化不同分辨率的特征来捕获丰富的上下文信息，后者可以得到清晰的目标边缘。

In order to capture the contextual information at multiple scales, DeepLabv3 [23] applies several parallel atrous convolution with different rates (called Atrous Spatial Pyramid Pooling, or ASPP), while PSPNet [24] performs pooling operations at different grid scales. Even though rich semantic information is encoded in the last feature map, detailed information related to object boundaries is missing due to the pooling or convolutions with striding operations within the network backbone. This could be alleviated by applying the atrous convolution to extract denser feature maps. However, given the design of state-of-art neural networks [7,9,10,25,26] and limited GPU memory, it is computationally prohibitive to extract output feature maps that are 8, or even 4 times smaller than the input resolution. Taking ResNet-101 [25] for example, when applying atrous convolution to extract output features that are 16 times smaller than input resolution, features within the last 3 residual blocks (9 layers) have to be dilated. Even worse, 26 residual blocks (78 layers!) will be affected if output features that are 8 times smaller than input are desired. Thus, it is computationally intensive if denser output features are extracted for this type of models. On the other hand, encoder-decoder models [21,22] lend themselves to faster computation (since no features are dilated) in the encoder path and gradually recover sharp object boundaries in the decoder path. Attempting to combine the advantages from both methods, we propose to enrich the encoder module in the encoder-decoder networks by incorporating the multi-scale contextual information.

为在多尺度捕获上下文信息，DeepLabV3使用了并行的多孔洞率的孔洞卷积(ASPP)，而PSPNet[24]在不同的网格尺度上进行池化操作。即使在最后的特征图中包含了丰富的语义信息，但与目标边缘的详细的信息仍然缺失，主要是由于骨干网络中的池化运算或带步长的卷积运算。使用孔洞卷积来提取更密集的特征图，可以减轻这个问题。但是，在现有最好的神经网络下，以及有限的GPU内存下，要提取输出特征图分辨率在输入分辨率的1/8甚至是1/4，这在计算上是不可能的。以ResNet-101为例，当使用孔洞卷积来提取的输出特征，其分辨率为输入的1/16时，最后三个残差单元（9层）的特征必须膨胀。如果是输入分辨率的1/8，那么26个残差模块(78层)都会受到影响。所以，如果使用这种模型来提取更密集的输出特征，那么计算量非常之大。另一方面，编码器-解码器模型[21,22]在编码器路径上计算量相对较低（因为特征不需要膨胀），在解码器路径上逐渐恢复出目标边缘。我们试图综合两种方法的优点，提出增强编码器-解码器网络中的编码器模块，加入多尺度上下文信息。

In particular, our proposed model, called DeepLabv3+, extends DeepLabv3 [23] by adding a simple yet effective decoder module to recover the object boundaries, as illustrated in Fig. 1. The rich semantic information is encoded in the output of DeepLabv3, with atrous convolution allowing one to control the density of the encoder features, depending on the budget of computation resources. Furthermore, the decoder module allows detailed object boundary recovery.

特别的，我们提出的模型，称为DeepLabV3+，拓展了DeepLabV3，加入了一个简单却有效的解码器模块，以恢复目标边缘，如图1所示。DeepLabV3的输出中包含了丰富的语义信息，而孔洞卷积可以控制编码器特征的密度，视计算资源的预算而定。进一步，解码器模块可以恢复目标边缘的细节。

Fig. 1. We improve DeepLabv3, which employs the spatial pyramid pooling module (a), with the encoder-decoder structure (b). The proposed model, DeepLabv3+, contains rich semantic information from the encoder module, while the detailed object boundaries are recovered by the simple yet effective decoder module. The encoder module allows us to extract features at an arbitrary resolution by applying atrous convolution. (a)Spatial Pyramid Pooling; (b)Encoder-Decoder; (c)Encoder-Decoder with Atrous Conv.

Motivated by the recent success of depthwise separable convolution [27,28,26,29,30], we also explore this operation and show improvement in terms of both speed and accuracy by adapting the Xception model [26], similar to [31], for the task of semantic segmentation, and applying the atrous separable convolution to both the ASPP and decoder modules. Finally, we demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasts and attain the test set performance of 89.0% and 82.1% without any post-processing, setting a new state-of-the-art.

受最近关于depthwise separable卷积的推动，我们也研究了这种运算，改造了Xception模型[26]进行语义分割的任务，与[31]类似，并对ASPP和解码器模块都使用了孔洞可分离卷积，这在准确率和速度上都有改进。最后，我们证明了提出的模型在PASCAL VOC 2012和Cityscape数据集上的有效性，在没有后处理的情况下分别得到了89.0%和82.1%的性能，竖立了新的最佳成绩。

In summary, our contributions are: 总结起来，我们的贡献在于：

- We propose a novel encoder-decoder structure which employs DeepLabv3 as a powerful encoder module and a simple yet effective decoder module. 我们提出了一种新的编码器-解码器结构，采用DeepLabV3作为编码器模块，解码器模块简单却有效。

- In our structure, one can arbitrarily control the resolution of extracted encoder features by atrous convolution to trade-off precision and runtime, which is not possible with existing encoder-decoder models. 在我们的结构中，可以通过孔洞卷积任意控制提取出的编码器特征的分辨率，在精度与运行时间中取得平衡，这在现有的编码器-解码器模型中是不可能的。

- We adapt the Xception model for the segmentation task and apply depthwise separable convolution to both ASPP module and decoder module, resulting in a faster and stronger encoder-decoder network. 我们改造了Xception模型进行语义分割任务，将depthwise separable卷积应用于ASPP和解码器模块，得到了一种更快更强的编码器-解码器网络。

- Our proposed model attains a new state-of-art performance on PASCAL VOC 2012 and Cityscapes datasets. We also provide detailed analysis of design choices and model variants. 我们提出的模型在PASCAL VOC 2012和Cityscapes数据集上得到了新的最佳性能。我们详细分析了设计选项和模型变体。

- We make our Tensorflow-based implementation of the proposed model publicly available at https://github.com/tensorflow/models/tree/master/research/deeplab. 代码已开源。

## 2 Related Work 相关工作

Models based on Fully Convolutional Networks (FCNs) [8,11] have demonstrated significant improvement on several segmentation benchmarks [1,2,3,4,5]. There are several model variants proposed to exploit the contextual information for segmentation [12,13,14,15,16,17,32,33], including those that employ multi-scale inputs (i.e., image pyramid) [34,35,36,37,38,39] or those that adopt probabilistic graphical models (such as DenseCRF [40] with efficient inference algorithm [41]) [42,43,44,37,45,46,47,48,49,50,51,39]. In this work, we mainly discuss about the models that use spatial pyramid pooling and encoder-decoder structure.

基于全卷积网络的模型在几个语义分割基准测试中展示出了显著的改进效果。提出了几种模型变体，以利用上下文信息进行分割，包括采用多尺度输入的（即，图像金字塔），或采用概率图模型的（如DenseCRF）。本文中，我们主要讨论使用空间金字塔池化和编码器-解码器结构的模型。

**Spatial pyramid pooling**: Models, such as PSPNet [24] or DeepLab [39,23], perform spatial pyramid pooling [18,19] at several grid scales (including image-level pooling [52]) or apply several parallel atrous convolution with different rates (called Atrous Spatial Pyramid Pooling, or ASPP). These models have shown promising results on several segmentation benchmarks by exploiting the multi-scale information.

**空间金字塔池化**：PSPNet或DeepLab这样的模型，在多个网格尺度上进行空间金字塔池化（包括图像层的池化），或使用多孔洞率的并行孔洞卷积（称为ASPP）。这些模型，通过利用多尺度信息，在几个语义分割基准测试中得到了很有希望的结果。

**Encoder-decoder**: The encoder-decoder networks have been successfully applied to many computer vision tasks, including human pose estimation [53], object detection [54,55,56], and semantic segmentation [11,57,21,22,58,59,60,61,62,63,64]. Typically, the encoder-decoder networks contain (1) an encoder module that gradually reduces the feature maps and captures higher semantic information, and (2) a decoder module that gradually recovers the spatial information. Building on top of this idea, we propose to use DeepLabv3 [23] as the encoder module and add a simple yet effective decoder module to obtain sharper segmentations.

**编码器-解码器**：编码器-解码器网络已经成功用于多种计算机视觉任务中，包括人体姿态估计，目标检测和语义分割。典型的编码器-解码器网络包括：(1)编码器模块，逐渐降低特征图分辨率，捕获更高的语义信息；(2)解码器模块，逐渐恢复空间信息。在这种思想的基础上，我们提出使用DeepLabV3作为编码器模块，增加了一个简单却有效的解码器模块，以得到更清晰的分割。

**Depthwise separable convolution**: Depthwise separable convolution [27,28] or group convolution [7,65], a powerful operation to reduce the computation cost and number of parameters while maintaining similar (or slightly better) performance. This operation has been adopted in many recent neural network designs [66,67,26,29,30,31,68]. In particular, we explore the Xception model [26], similar to [31] for their COCO 2017 detection challenge submission, and show improvement in terms of both accuracy and speed for the task of semantic segmentation. 这是一种强有力的算子，可以降低计算代价和参数数量，同时可保持类似的性能（或略微更好）。这种算子在最近很多神经网络设计中得到了采用。特别的，我们探索了Xception模型，与[31]提交给COCO 2017目标检测挑战的类似，在语义分割任务中展现了准确率与速度的改进。

## 3 Methods 方法

In this section, we briefly introduce atrous convolution [69,70,8,71,42] and depthwise separable convolution [27,28,67,26,29]. We then review DeepLabv3 [23] which is used as our encoder module before discussing the proposed decoder module appended to the encoder output. We also present a modified Xception model [26,31] which further improves the performance with faster computation.

本节中，我们简要介绍了孔洞卷积和depthwise separable卷积，然后回顾了DeepLabV3，将其用于我们的解码器模块，然后讨论提出的解码器模块，置于编码器输出之后。我们还给出了一个修正的Xception模型，既改进了性能，计算速度也更快。

### 3.1 Encoder-Decoder with Atrous Convolution

**Atrous convolution**: Atrous convolution, a powerful tool that allows us to explicitly control the resolution of features computed by deep convolutional neural networks and adjust filter’s field-of-view in order to capture multi-scale information, generalizes standard convolution operation. In the case of two-dimensional signals, for each location i on the output feature map y and a convolution filter w, atrous convolution is applied over the input feature map x as follows:

**孔洞卷积**：孔洞卷积是一种强有力的工具，可以显式的控制DCNNs计算的特征图的分辨率，调整滤波器的视野，以捕获多尺度信息，对标准卷积进行了推广。在二维信号的情况下，对输出特征图y的每个位置i，和卷积滤波器w，孔洞卷积按照如下的方式应用于输入特征图：

$$y[i] = \sum_k x[i+r·k] · w[k]$$(1)

where the atrous rate r determines the stride with which we sample the input signal. We refer interested readers to [39] for more details. Note that standard convolution is a special case in which rate r = 1. The filter’s field-of-view is adaptively modified by changing the rate value. 其中孔洞率r为对输入信号进行采样的步长。读者可以参考[39]。注意标准卷积是孔洞卷积的一种特殊情况，即r=1。滤波器的视野通过改变孔洞率可以自适应的变化。

**Depthwise separable convolution**: Depthwise separable convolution, factorizing a standard convolution into a depthwise convolution followed by a pointwise convolution (i.e., 1 × 1 convolution), drastically reduces computation complexity. Specifically, the depthwise convolution performs a spatial convolution independently for each input channel, while the pointwise convolution is employed to combine the output from the depthwise convolution. In the TensorFlow [72] implementation of depthwise separable convolution, atrous convolution has been supported in the depthwise convolution (i.e., the spatial convolution), as illustrated in Fig. 3. In this work, we refer the resulting convolution as atrous separable convolution, and found that atrous separable convolution significantly reduces the computation complexity of proposed model while maintaining similar (or better) performance.

**Depthwise separable convolution**：这种卷积将标准卷积分解成depthwise convolution和pointwise convolution（即1×1卷积）的级联，极大的降低了计算复杂度。具体的，depthwise卷积进行的空间卷积在每个输入通道中都是独立的，而pointwise卷积用于将depthwise卷积的输出综合起来。在depthwise separable卷积的TensorFlow实现中，孔洞卷积也有depthwise卷积（即空间卷积）支持了，如图3所示。本文中，我们称得到的卷积为孔洞可分离卷积，发现孔洞可分离卷积显著降低了提出模型的计算复杂度，而保持其性能没有下降（甚至更好）。

Fig. 3. 3 × 3 Depthwise separable convolution decomposes a standard convolution into (a) a depthwise convolution (applying a single filter for each input channel) and (b) a pointwise convolution (combining the outputs from depthwise convolution across channels). In this work, we explore atrous separable convolution where atrous convolution is adopted in the depthwise convolution, as shown in (c) with rate = 2. (a) Depthwise conv. (b) Pointwise conv. (c) Atrous depthwise conv.

**DeepLabv3 as encoder**: DeepLabv3 [23] employs atrous convolution [69,70,8,71] to extract the features computed by deep convolutional neural networks at an arbitrary resolution. Here, we denote output_stride as the ratio of input image spatial resolution to the final output resolution (before global pooling or fully-connected layer). For the task of image classification, the spatial resolution of the final feature maps is usually 32 times smaller than the input image resolution and thus output_stride = 32. For the task of semantic segmentation, one can adopt output_stride = 16 (or 8) for denser feature extraction by removing the striding in the last one (or two) block(s) and applying the atrous convolution correspondingly (e.g., we apply rate = 2 and rate = 4 to the last two blocks respectively for output_stride = 8). Additionally, DeepLabv3 augments the Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales by applying atrous convolution with different rates, with the image-level features [52]. We use the last feature map before logits in the original DeepLabv3 as the encoder output in our proposed encoder-decoder structure. Note the encoder output feature map contains 256 channels and rich semantic information. Besides, one could extract features at an arbitrary resolution by applying the atrous convolution, depending on the computation budget.

**将DeepLabV3作为编码器**：DeepLabV3[23]采用孔洞卷积来提取特征，分辨率可变。这里，我们将输入图像分辨率与最终输出分辨率（在全局池化或全连接层之前）的比率表示为output_stride。对于图像分类的任务，最后特征图的空间分辨率通常是输入图像分辨率的1/32，所以output_stride = 32。对于语义分割任务来说，可以采用output_stride = 16 (or 8)进行更密集的特征提取，即去掉最后一（或二）层的步长，对应的使用孔洞卷积（如，对于output_stride = 8的情况，我们对最后两层分别使用孔洞率为2和4的卷积）。另外，DeepLabV3增强了ASPP模块，采用不同孔洞率的孔洞卷积在多尺度提取卷积特征，并用图像级的特征进行增强。我们在提出的编码器-解码器结构中，使用原始DeepLabV3中logits前的最后一个特征图作为编码器的输出。注意编码器输出的特征图包含256个通道和丰富的语义信息。另外，可以根据计算资源预算，采用孔洞卷积以任意分辨率提取特征。

**Proposed decoder**: The encoder features from DeepLabv3 are usually computed with output_stride = 16. In the work of [23], the features are bilinearly upsampled by a factor of 16, which could be considered a naive decoder module. However, this naive decoder module may not successfully recover object segmentation details. We thus propose a simple yet effective decoder module, as illustrated in Fig. 2. The encoder features are first bilinearly upsampled by a factor of 4 and then concatenated with the corresponding low-level features [73] from the network backbone that have the same spatial resolution (e.g., Conv2 before striding in ResNet-101 [25]). We apply another 1 × 1 convolution on the low-level features to reduce the number of channels, since the corresponding low-level features usually contain a large number of channels (e.g., 256 or 512) which may outweigh the importance of the rich encoder features (only 256 channels in our model) and make the training harder. After the concatenation, we apply a few 3 × 3 convolutions to refine the features followed by another simple bilinear upsampling by a factor of 4. We show in Sec. 4 that using output_stride = 16 for the encoder module strikes the best trade-off between speed and accuracy. The performance is marginally improved when using output_stride = 8 for the encoder module at the cost of extra computation complexity.

**提出的解码器**：DeepLabV3计算的编码器特征通常output_stride = 16。在[23]中，特征通过双线性插值，上采样了16倍，这可以认为是一种简单的解码器模块。但是，这种简单的解码器模块可能不会成功的恢复目标分割的细节。所以我们提出了一种简单但有效的解码器模块，如图2所示。编码器特征首先双线性插值，上采样到4倍分辨率，然后与对应的低层特征[73]（来自骨干网络中有相同空间分辨率的部分，如在ResNet-101中，步长之前的conv2）拼接到一起。我们将另一个1×1卷积用在低层特征上，来降低通道数量，因为对应的低层特征通常包含大量通道数（如256或512），这可能会超过丰富的编码器特征的重要性（在我们的模型中，只有256个通道），这会使训练更困难。在拼接后，我们将几个3×3卷积用于提炼特征，然后再用另一个简单的双线性插值，进行4倍上采样。我们在第4节中证明，对于编码器模块使用output_stride = 16，在速度与准确率上会得到最好的平衡。使用output_stride = 8会略微改进编码器模块的性能，但代价是额外的计算复杂度。

Fig. 2. Our proposed DeepLabv3+ extends DeepLabv3 by employing a encoder-decoder structure. The encoder module encodes multi-scale contextual information by applying atrous convolution at multiple scales, while the simple yet effective decoder module refines the segmentation results along object boundaries.

### 3.2 Modified Aligned Xception 修正的对齐Xception

The Xception model [26] has shown promising image classification results on ImageNet [74] with fast computation. More recently, the MSRA team [31] modifies the Xception model (called Aligned Xception) and further pushes the performance in the task of object detection. Motivated by these findings, we work in the same direction to adapt the Xception model for the task of semantic image segmentation. In particular, we make a few more changes on top of MSRA’s modifications, namely (1) deeper Xception same as in [31] except that we do not modify the entry flow network structure for fast computation and memory efficiency, (2) all max pooling operations are replaced by depthwise separable convolution with striding, which enables us to apply atrous separable convolution to extract feature maps at an arbitrary resolution (another option is to extend the atrous algorithm to max pooling operations), and (3) extra batch normalization [75] and ReLU activation are added after each 3 × 3 depthwise convolution, similar to MobileNet design [29]. See Fig. 4 for details.

Xception在ImageNet分类任务上表现不错，而且计算量小。最近，MSRA组对Xception模型进行了修正（称为Aligned Xception），在目标检测中进一步推进了性能提升。受其发现启发，我们在相同的方向进行努力，修改Xception模型进行语义分割。特别的，我们在MSRA的修改上进行更多的变化，即：(1)Xception更深，和[31]一样，除了我们没有修改entry flow网络结构，以进行快速计算，并节约内存；(2)所有的最大池化运算都替换成有步长的depthwise separable卷积，这使我们可以使用孔洞可分离卷积来提取任意空间分辨率的特征图（另一个选项是将孔洞算法拓展为最大池化运算）；(3)在每个3×3 depthwise卷积后，都增加了额外的BN和ReLU激活，这与MobileNet的设计类似。详见图4。

Fig. 4. We modify the Xception as follows: (1) more layers (same as MSRA’s modification except the changes in Entry flow), (2) all the max pooling operations are replaced by depthwise separable convolutions with striding, and (3) extra batch normalization and ReLU are added after each 3 × 3 depthwise convolution, similar to MobileNet.

## 4 Experimental Evaluation 试验评估

We employ ImageNet-1k [74] pretrained ResNet-101 [25] or modified aligned Xception [26,31] to extract dense feature maps by atrous convolution. Our implementation is built on TensorFlow [72] and is made publicly available. 我们采用在ImageNet-1k上预训练的ResNet-101，或修正的aligned Xception，采用孔洞卷积来提取密集特征。我们的实现是基于TensorFlow的，已经开源。

The proposed models are evaluated on the PASCAL VOC 2012 semantic segmentation benchmark [1] which contains 20 foreground object classes and one background class. The original dataset contains 1, 464 (train), 1, 449 (val ), and 1, 456 (test) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10, 582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

提出的模型在PASCAL VOC 2012语义分割基准测试[1]中进行评估，包含了20个前景目标类别，和一个背景类。原始数据集包含1464（训练）、1449（验证）和1456（测试）像素级标注的图像。我们通过[76]提供的额外标注对数据集进行扩充，得到了10582(trainaug)训练图像。算法性能用像素IOU在21类上的平均来进行度量。

We follow the same training protocol as in [23] and refer the interested readers to [23] for details. In short, we employ the same learning rate schedule (i.e., “poly” policy [52] and same initial learning rate 0.007), crop size 513 × 513, fine-tuning batch normalization parameters [75] when output_stride = 16, and random scale data augmentation during training. Note that we also include batch normalization parameters in the proposed decoder module. Our proposed model is trained end-to-end without piecewise pretraining of each component.

我们采用[23]中一样的训练方法。简短来说，我们采用相同的学习速率方案（即，poly策略，和相同的初始学习速率0.007），剪切块大小513×513，精调BN参数，output_stride = 16，训练时采用随机尺度变化数据扩充。注意我们在提出的解码器模块中也包含了BN参数。我们提出的模型是端到端训练的，没有对每个部分进行分段的预训练。

### 4.1 Decoder Design Choices 解码器设计选项

We define “DeepLabv3 feature map” as the last feature map computed by DeepLabv3 (i.e., the features containing ASPP features and image-level features), and [k × k, f] as a convolution operation with kernel k × k and f filters. 我们将DeepLabV3特征图定义为DeepLabV3计算得到的最后特征图（即，包含ASPP特征和图像级特征的总特征），定义卷积运算[k × k, f]为卷积核为k×k大小，滤波器数量为f。

When employing output_stride = 16, ResNet-101 based DeepLabv3 [23] bilinearly upsamples the logits by 16 during both training and evaluation. This simple bilinear upsampling could be considered as a naive decoder design, attaining the performance of 77.21% [23] on PASCAL VOC 2012 val set and is 1.2% better than not using this naive decoder during training (i.e., downsampling groundtruth during training). To improve over this naive baseline, our proposed model “DeepLabv3+” adds the decoder module on top of the encoder output, as shown in Fig. 2. In the decoder module, we consider three places for different design choices, namely (1) the 1 × 1 convolution used to reduce the channels of the low-level feature map from the encoder module, (2) the 3 × 3 convolution used to obtain sharper segmentation results, and (3) what encoder low-level features should be used.

采用output_stride = 16，基于ResNet-101的DeepLabV3[23]时，在训练和测试时，将logits双线性插值，上采样16倍。这种简单的双线性插值上采样也可以认为是一种简单的解码器设计，在PASCAL VOC 2012验证集上得到了77.21%[23]的性能，比在训练时不用这种简单解码器的情况要好1.2%（即，在训练时对真值进行下采样）。为在这种简单基准上改进，我们提出的模型DeepLabV3+，在编码器输出上增加了解码器模块，如图2所示。在解码器模块中，我们考虑在三个地方进行不同的设计，即：(1)使用1×1卷积降低编码器模块中底层特征图的通道数；(2)使用3×3卷积得到更清晰的分割结果；(3)要使用编码器中什么样的底层特征。

To evaluate the effect of the 1 × 1 convolution in the decoder module, we employ [3 × 3, 256] and the Conv2 features from ResNet-101 network backbone, i.e., the last feature map in res2x residual block (to be concrete, we use the feature map before striding). As shown in Tab. 1, reducing the channels of the low-level feature map from the encoder module to either 48 or 32 leads to better performance. We thus adopt [1 × 1, 48] for channel reduction.

为评估解码器模块中的1×1卷积的效果，我们采用[3×3, 256]和ResNet-101骨干网络中的Conv2特征，即res2x残差模块的最后一个特征图（具体来说，我们使用步长前的特征图）。如表1所示，减少编码器模块的底层特征图数量到48或32，会得到更好的性能。所以我们采用[1×1, 48]减少通道数。

Table 1. PASCAL VOC 2012 val set. Effect of decoder 1 × 1 convolution used to reduce the channels of low-level feature map from the encoder module. We fix the other components in the decoder structure as using [3 × 3, 256] and Conv2.

Channels | 8 | 16 | 32 | 48 | 64
--- | --- | --- | --- | --- | ---
mIOU | 77.61% | 77.92% | 78.16% | 78.21% | 77.94%

We then design the 3 × 3 convolution structure for the decoder module and report the findings in Tab. 2. We find that after concatenating the Conv2 feature map (before striding) with DeepLabv3 feature map, it is more effective to employ two 3×3 convolution with 256 filters than using simply one or three convolutions. Changing the number of filters from 256 to 128 or the kernel size from 3 × 3 to 1×1 degrades performance. We also experiment with the case where both Conv2 and Conv3 feature maps are exploited in the decoder module. In this case, the decoder feature map are gradually upsampled by 2, concatenated with Conv3 first and then Conv2, and each will be refined by the [3 × 3, 256] operation. The whole decoding procedure is then similar to the U-Net/SegNet design [21,22]. However, we have not observed significant improvement. Thus, in the end, we adopt the very simple yet effective decoder module: the concatenation of the DeepLabv3 feature map and the channel-reduced Conv2 feature map are refined by two [3 × 3, 256] operations. Note that our proposed DeepLabv3+ model has output_stride = 4. We do not pursue further denser output feature map (i.e., output_stride < 4) given the limited GPU resources.

我们然后为解码器模块设计3×3卷积结构，结果如表2所示。我们发现，将Conv2特征图（步长运算前）与DeepLabV3特征图进行拼接后，使用2个3×3卷积256个滤波器，比使用1个或3个卷积效率更高。滤波器数量从256变到128，或将滤波核大小从3×3降到1×1，都会使性能降低。我们还试验了在解码器模块中使用Conv2和Conv3特征图的情况。在这种情况下，解码器特征图逐渐上采样2倍，先与Conv3拼接到一起，然后与Conv2拼接到一起，每个都进行[3×3, 256]运算的精炼。整个解码过程与U-Net/SegNet设计类似[21,22]。但是，我们没有观察到明显的改进。所以，最后，我们还是采用了简单却有效的解码器模块：将DeepLabV3特征图与与降低通道数的Conv2特征图拼接到一起，然后通过2个[3×3, 256]运算。注意，我们提出的DeepLabV3+模型的output_stride = 4。由于GPU资源有限，我们没有追求更密集的输出特征图（即，output_stride < 4）。

Table 2. Effect of decoder structure when fixing [1 × 1, 48] to reduce the encoder feature channels. We found that it is most effective to use the Conv2 (before striding) feature map and two extra [3 × 3, 256] operations. Performance on VOC 2012 val set.

Conv2 | Conv3 | 3×3 Conv Structure | mIOU
--- | --- | --- | ---
y | n | [3×3, 256] | 78.21%
y | n | [3×3, 256]×2 | 78.85%
y | n | [3×3, 256]×3 | 78.02%
y | n | [3×3, 128] | 77.25%
y | n | [1×1, 256] | 78.07%
y | y | [3×3, 256] | 78.61%

### 4.2 ResNet-101 as Network Backbone

To compare the model variants in terms of both accuracy and speed, we report mIOU and Multiply-Adds in Tab. 3 when using ResNet-101 [25] as network backbone in the proposed DeepLabv3+ model. Thanks to atrous convolution, we are able to obtain features at different resolutions during training and evaluation using a single model.

为在准确率和速度方面比较模型变体，我们在表3中给出，我们的DeepLabV3+模型使用ResNet-101作为骨干网络时的mIOU和乘法-加法运算量。使用了孔洞卷积，我们才可以使用一个模型，在训练和评估时得到不同分辨率的特征。

Table 3. Inference strategy on the PASCAL VOC 2012 val set using ResNet-101. train OS: The output stride used during training. eval OS: The output stride used during evaluation. Decoder: Employing the proposed decoder structure. MS: Multi-scale inputs during evaluation. Flip: Adding left-right flipped inputs.

train OS | eval OS | Decoder | MS | Flip | mIOU | Multiply-Adds
--- | --- | --- | --- | --- | --- | ---
16 | 16 | n | n | n | 77.21% | 81.02B
16 | 8 | n | n | n | 78.51% | 276.18B
16 | 8 | n | y | n | 78.45% | 2435.37B
16 | 8 | n | y | y | 79.77% | 4870.59B
16 | 16 | y | n | n | 78.85% | 101.28B
16 | 16 | y | y | n | 80.09% | 898.69B
16 | 16 | y | y | y | 80.22% | 1797.23B
16 | 8 | y | n | n | 79.35% | 297.92B
16 | 8 | y | y | n | 80.43% | 2623.61B
16 | 8 | y | y | y | 80.57% | 5247.07B
32 | 32 | n | n | n | 75.43% | 52.43B
32 | 32 | y | n | n | 77.37% | 74.20B
32 | 16 | y | n | n | 77.80% | 101.28B
32 | 8 | y | n | n | 77.92% | 297.92B

**Baseline**: The first row block in Tab. 3 contains the results from [23] showing that extracting denser feature maps during evaluation (i.e., eval output_stride = 8) and adopting multi-scale inputs increases performance. Besides, adding left-right flipped inputs doubles the computation complexity with only marginal performance improvement.

**基准**：表3中的第一行包含了[23]中的结果，说明评估时提取更密集的特征图（即，评估时output_stride = 8），采用多尺度输入，会提高性能。另外，增加左右翻转输入会使计算量加倍，但性能改进只有很少。

**Adding decoder**: The second row block in Tab. 3 contains the results when adopting the proposed decoder structure. The performance is improved from 77.21% to 78.85% or 78.51% to 79.35% when using eval output_stride = 16 or 8, respectively, at the cost of about 20B extra computation overhead. The performance is further improved when using multi-scale and left-right flipped inputs.

**增加解码器**：表3中第二行是采用了提出的解码器结构时的结果。在使用output_stride = 16 or 8时，性能分别从77.21%提升到了78.85%，或从78.51%提升到79.35%，代价是增加了20B额外计算量。当使用了多尺度输入和左右翻转输入时，性能可以得到进一步提升。

**Coarser feature maps**: We also experiment with the case when using train output_stride = 32 (i.e., no atrous convolution at all during training) for fast computation. As shown in the third row block in Tab. 3, adding the decoder brings about 2% improvement while only 74.20B Multiply-Adds are required. However, the performance is always about 1% to 1.5% below the case in which we employ train output_stride = 16 and different eval output_stride values. We thus prefer using output_stride = 16 or 8 during training or evaluation depending on the complexity budget.

**更粗糙的特征图**：我们还尝试使用output_stride = 32（即，在训练时不采用孔洞卷积）进行试验，比进行快速计算。如表3的第三行所示，增加了解码器会带来2%的改进，而且只需要74.20B乘法-加法计算。但是，与output_stride = 16和不同的评估output_stride值相比，性能一直低了1%到1.5%。所以我们在现有的计算资源预算下，倾向于在训练或评估时使用output_stride = 16或8。

### 4.3 Xception as Network Backbone 

We further employ the more powerful Xception [26] as network backbone. Following [31], we make a few more changes, as described in Sec. 3.2. 我们进一步改用更强大的Xception作为网络骨干。与[31]一样，我们进行了一些改变，如3.2节所示。

**ImageNet pretraining**: The proposed Xception network is pretrained on ImageNet-1k dataset [74] with similar training protocol in [26]. Specifically, we adopt Nesterov momentum optimizer with momentum = 0.9, initial learning rate = 0.05, rate decay = 0.94 every 2 epochs, and weight decay 4e-5. We use asynchronous training with 50 GPUs and each GPU has batch size 32 with image size 299×299. We did not tune the hyper-parameters very hard as the goal is to pretrain the model on ImageNet for semantic segmentation. We report the single-model error rates on the validation set in Tab. 4 along with the baseline reproduced ResNet-101 [25] under the same training protocol. We have observed 0.75% and 0.29% performance degradation for Top1 and Top5 accuracy when not adding the extra batch normalization and ReLU after each 3 × 3 depthwise convolution in the modified Xception.

**在ImageNet上预训练**：提出的Xception网络在ImageNet-1k数据集上采用与[26]类似的训练方案进行了预训练。具体的，我们采用了Nesterov动量优化器，动量为0.9，初始学习速率0.05，每两轮训练衰减率0.94，权重衰减4e-5。我们使用50 GPUs异步训练，每个GPU批大小为50，图像大小299×299。我们没有进行很多超参数调整，因为目标是在ImageNet上预训练模型，以进行语义分割。我们在表4中给出验证集上的单模型错误率，以及采用相同的训练方法复现ResNet-101[25]的基准。在修正Xception中，如果在每个3×3 depthwise卷积后不加入额外的BN和ReLU，其top-1和top-5准确率会分别下降0.75%和0.29%。

Table 4. Single-model error rates on ImageNet-1K validation set.

Model | Top-1 Error | Top-5 Error
--- | --- | ---
Reproduced ResNet-101 | 22.40% | 6.02%
Modified Xception | 20.19% | 5.17%

The results of using the proposed Xception as network backbone for semantic segmentation are reported in Tab. 5. 使用提出的Xception作为骨干网络进行语义分割的结果，如表5所示。

**Baseline**: We first report the results without using the proposed decoder in the first row block in Tab. 5, which shows that employing Xception as network backbone improves the performance by about 2% when train output stride = eval output stride = 16 over the case where ResNet-101 is used. Further improvement can also be obtained by using eval output stride = 8, multi-scale inputs during inference and adding left-right flipped inputs. Note that we do not employ the multi-grid method [77,78,23], which we found does not improve the performance.

**基准**：我们首先在表5中第1行给出不使用提出的解码器的结果，这表明使用Xception作为骨干网络，训练时output stride与评估时output stride都为16，比使用ResNet-101的情况，性能会提升大约2%。使用output stride = 8，推理时使用多尺度输入，增加左右翻转输入，也可以进一步改进性能。注意我们没有采用多网格方法[77,78,23]，我们发现这不会改进性能。

**Adding decoder**: As shown in the second row block in Tab. 5, adding decoder brings about 0.8% improvement when using eval output stride = 16 for all the different inference strategies. The improvement becomes less when using eval output stride = 8.

**增加解码器**：如表5中的第二行所示，增加解码器会带来0.8%的性能改进，使用的评估output stride = 16，在所有不同的推理策略下都是这样。在使用的评估output stride = 8时，性能改进会更少一些。

**Using depthwise separable convolution**: Motivated by the efficient computation of depthwise separable convolution, we further adopt it in the ASPP and the decoder modules. As shown in the third row block in Tab. 5, the computation complexity in terms of Multiply-Adds is significantly reduced by 33% to 41%, while similar mIOU performance is obtained.

**使用depthwise separable卷积**：受depthwise separable卷积的高效计算启发，我们进一步在ASPP和解码器模块中也采用。如表5的第三行所示，计算复杂度显著降低了33%到41%，而且保持了类似的mIOU。

**Pretraining on COCO**: For comparison with other state-of-art models, we further pretrain our proposed DeepLabv3+ model on MS-COCO dataset [79], which yields about extra 2% improvement for all different inference strategies. 为与其他目前最好的模型对比，我们进一步在MS COCO数据集上对提出的DeepLabV3+进行预训练，这对所有不同的推理策略都可以得到大约2%的改进。

**Pretraining on JFT**: Similar to [23], we also employ the proposed Xception model that has been pretrained on both ImageNet-1k [74] and JFT-300M dataset [80,26,81], which brings extra 0.8% to 1% improvement. 与[23]类似，我们采用的Xception模型，在ImageNet-1k和JFT-300M数据集上都进行过预训练，这会带来0.8%到1%的额外改进。

**Test set results**: Since the computation complexity is not considered in the benchmark evaluation, we thus opt for the best performance model and train it with output stride = 8 and frozen batch normalization parameters. In the end, our ‘DeepLabv3+’ achieves the performance of 87.8% and 89.0% without and with JFT dataset pretraining. 由于在基准测试评估中没有考虑计算复杂度，我们因为选择最佳性能的模型，并使用output stride = 8进行训练，冻结BN参数。最后，我们的DeepLabV3+在有/没有JFT数据集预训练的情况下分别得到了87.8%和89.0%的性能。

**Qualitative results**: We provide visual results of our best model in Fig. 6. As shown in the figure, our model is able to segment objects very well without any post-processing. 我们在图6中给出可视化结果。如图所示，我们的模型可以在没有后处理的情况下，得到很好的分割结果。

**Failure mode**: As shown in the last row of Fig. 6, our model has difficulty in segmenting (a) sofa vs. chair, (b) heavily occluded objects, and (c) objects with rare view. 如图6最后一行所示，我们的模型在分割以下目标时有困难：(a)沙发和椅子；(b)遮挡严重的目标；(c)罕见视角的目标。

Table 5. Inference strategy on the PASCAL VOC 2012 val set when using modified Xception. train OS: The output stride used during training. eval OS: The output stride used during evaluation. Decoder: Employing the proposed decoder structure. MS: Multi-scale inputs during evaluation. Flip: Adding left-right flipped inputs. SC: Adopting depthwise separable convolution for both ASPP and decoder modules. COCO: Models pretrained on MS-COCO. JFT: Models pretrained on JFT.

train OS | eval OS | Decoder | MS | Flip | SC | COCO | JFT | mIOU | Mul-Adds
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
16 | 16 | n | n | n | n | n | n | 79.17% | 68.00B
16 | 16 | n | y | n | n | n | n | 80.57% | 601.74B
16 | 16 | n | y | y | n | n | n | 80.79% | 1203.34B
16 | 8 | n | n | n | n | n | n | 79.64% | 240.85B
16 | 8 | n | y | n | n | n | n | 81.15% | 2149.91B
16 | 8 | n | y | y | n | n | n | 81.34% | 4299.68B
16 | 16 | y | n | n | n | n | n | 79.93% | 89.76B
16 | 16 | y | y | n | n | n | n | 81.38% | 790.12B
16 | 16 | y | y | y | n | n | n | 81.44% | 1580.10B
16 | 8 | y | n | n | n | n | n | 80.22% | 262.59B
16 | 8 | y | y | n | n | n | n | 81.60% | 2338.15B
16 | 8 | y | y | y | n | n | n | 81.63% | 4676.16B
16 | 16 | y | n | n | y | n | n | 79.79% | 54.17B
16 | 16 | y | y | y | y | n | n | 81.21% | 928.81B
16 | 8 | y | n | n | y | n | n | 80.02% | 177.10B
16 | 8 | y | y | y | y | n | n | 81.39% | 3055.35B
16 | 16 | y | n | n | y | y | n | 82.20% | 54.17B
16 | 16 | y | y | y | y | y | n | 83.34% | 928.81B
16 | 8 | y | n | n | y | y | n | 82.45% | 177.10B
16 | 8 | y | y | y | y | y | n | 83.58% | 3055.35B
16 | 16 | y | n | n | y | y | y | 83.03% | 54.17B
16 | 16 | y | y | y | y | y | y | 84.22% | 928.81B
16 | 8 | y | n | n | y | y | y | 83.39% | 177.10B
16 | 8 | y | y | y | y | y | y | 84.56% | 3055.35B

Fig. 6. Visualization results on val set. The last row shows a failure mode.

### 4.4 Improvement along Object Boundaries

In this subsection, we evaluate the segmentation accuracy with the trimap experiment [14,40,39] to quantify the accuracy of the proposed decoder module near object boundaries. Specifically, we apply the morphological dilation on ‘void’ label annotations on val set, which typically occurs around object boundaries. We then compute the mean IOU for those pixels that are within the dilated band (called trimap) of ‘void’ labels. As shown in Fig. 5 (a), employing the proposed decoder for both ResNet-101 [25] and Xception [26] network backbones improves the performance compared to the naive bilinear upsampling. The improvement is more significant when the dilated band is narrow. We have observed 4.8% and 5.4% mIOU improvement for ResNet-101 and Xception respectively at the smallest trimap width as shown in the figure. We also visualize the effect of employing the proposed decoder in Fig. 5 (b).

这一小节中，我们评估用triamp试验来评估分割准确率，以量化提出的解码器模块在目标边缘附近的准确率。具体的，我们在验证集上的void标签标注上使用形态学膨胀运算，void标签一般在目标边缘处。然后我们计算这些像素在void标签膨胀带（称为trimap）的mIOU。如图5(a)所示，对ResNet-101[25]和Xception[26]骨干网络采用提出的解码器，比使用简单的双线性插值上采样，可以改进性能。当膨胀带很窄的时候，改进更明显。如图所示，在最窄的trimap宽度时，对ResNet-101和Xception的mIOU改进分别有4.8%和5.4%。我们还在图5(b)中给出采用提出的解码器的可视化结果。

Fig. 5. (a) mIOU as a function of trimap band width around the object boundaries when employing train output stride = eval output stride = 16. BU: Bilinear upsampling. (b) Qualitative effect of employing the proposed decoder module compared with the naive bilinear upsampling (denoted as BU). In the examples, we adopt Xception as feature extractor and train output stride = eval output stride = 16.

### 4.5 Experimental Results on Cityscapes

In this section, we experiment DeepLabv3+ on the Cityscapes dataset [3], a large-scale dataset containing high quality pixel-level annotations of 5000 images (2975, 500, and 1525 for the training, validation, and test sets respectively) and about 20000 coarsely annotated images.

本节中，我们在Cityscapes数据集上对DeepLabV3+进行试验，这个大型数据集包含5000幅高质量像素级的标注图像（训练、验证和测试集分别有2975,500和1525幅图像），和大约20000粗糙标注的图像。

As shown in Tab. 7 (a), employing the proposed Xception model as network backbone (denoted as X-65) on top of DeepLabv3 [23], which includes the ASPP module and image-level features [52], attains the performance of 77.33% on the validation set. Adding the proposed decoder module significantly improves the performance to 78.79% (1.46% improvement). We notice that removing the augmented image-level feature improves the performance to 79.14%, showing that in DeepLab model, the image-level features are more effective on the PASCAL VOC 2012 dataset. We also discover that on the Cityscapes dataset, it is effective to increase more layers in the entry flow in the Xception [26], the same as what [31] did for the object detection task. The resulting model building on top of the deeper network backbone (denoted as X-71 in the table), attains the best performance of 79.55% on the validation set.

如图7(a)所示，在DeepLabV3上采用提出的Xception模型作为网络骨干（表示为X-65），包含了ASPP模块和图像层特征，可以在验证集上得到77.33%的性能。增加提出的解码器模块可以将性能显著提升到78.79%（改进1.46%）。我们注意到，去掉增强的图像层特征，会将性能提升到79.14%，说明在DeepLab模型中，图像层的特征在PASCAL VOC 2012数据集上更有效。我们还发现，在Cityscape数据集上，在Xception的entry flow中增加更多的层会更有效，这与[31]中进行目标检测的情况一样。在更深的网络骨干上构建得到的模型（在表中表示为X-71），在验证集上得到了79.55%的最佳性能。

After finding the best model variant on val set, we then further fine-tune the model on the coarse annotations in order to compete with other state-of-art models. As shown in Tab. 7 (b), our proposed DeepLabv3+ attains a performance of 82.1% on the test set, setting a new state-of-art performance on Cityscapes.

在验证集上发现了最佳模型变体后，我们进一步在粗糙标记上精调模型，以与目前最好的模型进行比较。如表7(b)所示，我们提出的DeepLabV3+在测试集上得到了82.1%的性能，在Cityscapes上竖立了新的最佳性能。

Table 7. (a) DeepLabv3+ on the Cityscapes val set when trained with train_fine set. (b) DeepLabv3+ on Cityscapes test set. Coarse: Use train_extra set (coarse annotations) as well. Only a few top models are listed in this table.

(a) val set results

Backbone | Decoder | ASPP | Image-Level | mIOU
--- | --- | --- | --- | ---
X-65 | n | y | y | 77.33
X-65 | y | y | y | 78.79
X-65 | y | y | n | 79.14
X-71 | y | y | n | 79.55

(b) test set results

Method | Coarse | mIOU
--- | --- | ---
ResNet-38 [83] | y | 80.6
PSPNet [24] | y | 81.2
Mapillary [86] | y | 82.0
DeepLabV3 | y | 81.3
DeepLabV3+ | y | 82.1

## 5 Conclusion

Our proposed model “DeepLabv3+” employs the encoder-decoder structure where DeepLabv3 is used to encode the rich contextual information and a simple yet effective decoder module is adopted to recover the object boundaries. One could also apply the atrous convolution to extract the encoder features at an arbitrary resolution, depending on the available computation resources. We also explore the Xception model and atrous separable convolution to make the proposed model faster and stronger. Finally, our experimental results show that the proposed model sets a new state-of-the-art performance on PASCAL VOC 2012 and Cityscapes datasets.

我们提出的DeepLabV3+模型采用了编码器-解码器结构，其中DeepLabV3用作编码器，包含了丰富的上下文信息，解码器简单却有效，可以恢复目标边缘。可以使用孔洞卷积在任意分辨率上提取编码器特征，这取决于可用的计算资源。我们还探索了Xception模型和孔洞可分离卷积，使提出的模型更快速更强。最后，我们的试验结果表明，提出的模型在PASCAL VOC 2012和Cityscapes数据集上竖立了新的最佳标准。