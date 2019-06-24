# Rethinking Atrous Convolution for Semantic Image Segmentation

Liang-Chieh Chen et al. Google Inc.

## Abstract 摘要

In this work, we revisit atrous convolution, a powerful tool to explicitly adjust filter’s field-of-view as well as control the resolution of feature responses computed by Deep Convolutional Neural Networks, in the application of semantic image segmentation. To handle the problem of segmenting objects at multiple scales, we design modules which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates. Furthermore, we propose to augment our previously proposed Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales, with image-level features encoding global context and further boost performance. We also elaborate on implementation details and share our experience on training our system. The proposed ‘DeepLabv3’ system significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2012 semantic image segmentation benchmark.

本文中，我们重新思考了孔洞卷积，在语义分割应用中，这种可以显式调整滤波器视野的强力工具，还可以控制DCNNs计算的特征响应的分辨率。为处理多尺度上的目标分割问题，我们设计了新的模块，使用级联的或并行的孔洞卷积，采用多孔洞率来捕获多尺度上下文。而且，我们提出扩充之前提出的ASPP模块，在多个尺度上探测卷积特征，有图像层的特征包含了全局上下文，可以进一步提升性能。我们还研究了实现细节，分享了我们在训练系统上的经验。提出的DeepLabV3系统对之前的DeepLab进行了显著的改进，在PASCAL VOC 2012语义分割基准测试上，得到了与其他目前最好的模型类似的性能。

## 1. Introduction 引言

For the task of semantic segmentation [20, 63, 14, 97, 7], we consider two challenges in applying Deep Convolutional Neural Networks (DCNNs) [50]. The first one is the reduced feature resolution caused by consecutive pooling operations or convolution striding, which allows DCNNs to learn increasingly abstract feature representations. However, this invariance to local image transformation may impede dense prediction tasks, where detailed spatial information is desired. To overcome this problem, we advocate the use of atrous convolution [36, 26, 74, 66], which has been shown to be effective for semantic image segmentation [10, 90, 11]. Atrous convolution, also known as dilated convolution, allows us to repurpose ImageNet [72] pretrained networks to extract denser feature maps by removing the downsampling operations from the last few layers and upsampling the corresponding filter kernels, equivalent to inserting holes (‘trous’ in French) between filter weights. With atrous convolution, one is able to control the resolution at which feature responses are computed within DCNNs without requiring learning extra parameters.

对语义分割的任务，我们考虑在应用DCNNs时考虑两个挑战。第一个是由于连续的池化操作或卷积步长导致的特征分辨率降低，这使DCNNs学习的越来越抽象的特征表示。但是，这种对局部图像变换的不变性可能不利于密集预测任务，其中需要空间细节信息。为克服这一问题，我们推荐使用孔洞卷积，这在语义分割中非常有效。孔洞卷积，也称为扩张卷积，使我们可以将ImageNet预训练的网络改变为提取更密集的特征图，从最后几层中去除下采样的运算，将对应的滤波核进行上采样，等价于在滤波器权重之间增加孔洞。有了孔洞卷积，可以控制DCNNs计算得到的特征响应的分辨率，而且不需要学习额外的参数。

Another difficulty comes from the existence of objects at multiple scales. Several methods have been proposed to handle the problem and we mainly consider four categories in this work, as illustrated in Fig. 2. First, the DCNN is applied to an image pyramid to extract features for each scale input [22, 19, 69, 55, 12, 11] where objects at different scales become prominent at different feature maps. Second, the encoder-decoder structure [3, 71, 25, 54, 70, 68, 39] exploits multi-scale features from the encoder part and recovers the spatial resolution from the decoder part. Third, extra modules are cascaded on top of the original network for capturing long range information. In particular, DenseCRF [45] is employed to encode pixel-level pairwise similarities [10, 96, 55, 73], while [59, 90] develop several extra convolutional layers in cascade to gradually capture long range context. Fourth, spatial pyramid pooling [11, 95] probes an incoming feature map with filters or pooling operations at multiple rates and multiple effective field-of-views, thus capturing objects at multiple scales.

另一个困难来自目标的存在是在多尺度上的。已经提出了几种方法来处理这个问题，我们在本文中主要考虑四种类别，如图2所示。第一，DCNN用于图像金字塔，以对每个输入尺度提取特征，不同尺度的目标在不同的特征图中变得更显著。第二，编码器-解码器架构利用了编码器部分的多尺度特征，从解码器部分恢复空间分辨率。第三，在原始网络上叠加了更多模块，捕获长程信息。特别是，DenseCRF[45]可以用于提出像素层的成对相似性，还可以利用额外的级联卷积层来逐步捕获长程上下文。第四，空间金字塔池化，用多种滤波器或多种孔洞率和多视野的池化操作来提取输入特征图，可以捕获多尺度的目标。

In this work, we revisit applying atrous convolution, which allows us to effectively enlarge the field of view of filters to incorporate multi-scale context, in the framework of both cascaded modules and spatial pyramid pooling. In particular, our proposed module consists of atrous convolution with various rates and batch normalization layers which we found important to be trained as well. We experiment with laying out the modules in cascade or in parallel (specifically, Atrous Spatial Pyramid Pooling (ASPP) method [11]). We discuss an important practical issue when applying a 3 × 3 atrous convolution with an extremely large rate, which fails to capture long range information due to image boundary effects, effectively simply degenerating to 1 × 1 convolution, and propose to incorporate image-level features into the ASPP module. Furthermore, we elaborate on implementation details and share experience on training the proposed models, including a simple yet effective bootstrapping method for handling rare and finely annotated objects. In the end, our proposed model, ‘DeepLabv3’ improves over our previous works [10, 11] and attains performance of 85.7% on the PASCAL VOC 2012 test set without DenseCRF postprocessing.

本文中，我们重新利用了孔洞卷积，使我们可以有效的增大滤波器的视野，来利用多尺度的上下文，以级联模块的形式和空间金字塔池化的形式。特别是，我们提出的模块包括多种孔洞率的卷积和BN层，我们发现都需要好好训练。我们试验用级联或并行方式进行（具体的，ASPP[11]）。我们讨论了一个重要的实践问题，在使用3×3孔洞卷积时，孔洞率非常大，由于图像边缘效应，无法捕获长程信息，实际上蜕化为1×1卷积；然后提出将图像层次的特征整合入ASPP模块。进一步，我们研究了实现细节，分享了训练我们提出的模型的经验，包括一种简单但有效的bootstrapping方法，可以处理罕见和精细标注的目标。最后，我们提出的模型，DeepLabV3改进了我们之前的工作，在PASCAL VOC 2012测试集上得到了85.7%的性能，没有DenseCRF后处理。

Figure 1. Atrous convolution with kernel size 3 × 3 and different rates. Standard convolution corresponds to atrous convolution with rate = 1. Employing large value of atrous rate enlarges the model’s field-of-view, enabling object encoding at multiple scales.

Figure 2. Alternative architectures to capture multi-scale context. (a) Image Pyramid (b) Encoder-Decoder (c) Deeper w. Atrous Convolution (d) Spatial Pyramid Pooling

## 2. Related Work

It has been shown that global features or contextual interactions [33, 76, 43, 48, 27, 89] are beneficial in correctly classifying pixels for semantic segmentation. In this work, we discuss four types of Fully Convolutional Networks (FCNs) [74, 60] (see Fig. 2 for illustration) that exploit context information for semantic segmentation [30, 15, 62, 9, 96, 55, 65, 73, 87].

已经证明，全局特征或上下文互动对于在语义分割中正确的对像素分类是有帮助的。本文中，我们讨论四种全卷积网络FCNs（见图2），利用上下文信息进行语义分割。

**Image pyramid**: The same model, typically with shared weights, is applied to multi-scale inputs. Feature responses from the small scale inputs encode the long-range context, while the large scale inputs preserve the small object details. Typical examples include Farabet et al. [22] who transform the input image through a Laplacian pyramid, feed each scale input to a DCNN and merge the feature maps from all the scales. [19, 69] apply multi-scale inputs sequentially from coarse-to-fine, while [55, 12, 11] directly resize the input for several scales and fuse the features from all the scales. The main drawback of this type of models is that it does not scale well for larger/deeper DCNNs (e.g., networks like [32, 91, 86]) due to limited GPU memory and thus it is usually applied during the inference stage [16].

**图像金字塔**：共享权重的相同模型应用于多尺度输入上。小尺度输入的特征响应包含了长程上下文，大尺度的输入保存了小目标的细节。一般的例子包括了Farabet等[22]，将输入图像通过一个Laplacian金字塔进行变换，将每个尺度的输入送入一个DCNN，将所有尺度的特征图进行融合。[19,69]将多尺度输入按从粗糙到精细的顺序进行处理，而[55,12,11]直接将输入变换到几个尺度上，将所有尺度的特征进行融合。这类模型的主要缺点是，由于GPU内存有限，扩展到更大/更深的DCNNs效果并不好，所以主要在推理阶段进行应用[16]。

**Encoder-decoder**: This model consists of two parts: (a)the encoder where the spatial dimension of feature maps is gradually reduced and thus longer range information is more easily captured in the deeper encoder output, and (b) the decoder where object details and spatial dimension are gradually recovered. For example, [60, 64] employ deconvolution [92] to learn the upsampling of low resolution feature responses. SegNet [3] reuses the pooling indices from the encoder and learn extra convolutional layers to densify the feature responses, while U-Net [71] adds skip connections from the encoder features to the corresponding decoder activations, and [25] employs a Laplacian pyramid reconstruction network. More recently, RefineNet [54] and [70, 68, 39] have demonstrated the effectiveness of models based on encoder-decoder structure on several semantic segmentation benchmarks. This type of model is also explored in the context of object detection [56, 77].

**编码器-解码器**：这种模型包括两个部分：(a)编码器，其中特征图的空间分辨率逐渐降低，在更深的编码器输出中，更容易捕捉到更长程的信息；(b)解码器，其中目标的细节和空间维度逐渐得到恢复。比如，[60,64]采用解卷积[92]来学习低分辨率特征响应的上采样。SegNet[3]对编码器的池化索引进行了重用，学习了额外的卷积层，使特征响应密集化，而U-Net[71]加入了从编码器特征到对应的解码器激活的跳跃连接，[25]采用了Laplacian金字塔金字塔重建网络。最近，RefineNet[54]和[79,68,39]证明了，基于编码器-解码器结构的模型在几个语义分割基准测试中是有效的。这类模型在目标检测中也有研究[56,77]。

**Context module**: This model contains extra modules laid out in cascade to encode long-range context. One effective method is to incorporate DenseCRF [45] (with efficient high-dimensional filtering algorithms [2]) to DCNNs [10, 11]. Furthermore, [96, 55, 73] propose to jointly train both the CRF and DCNN components, while [59, 90] employ several extra convolutional layers on top of the belief maps of DCNNs (belief maps are the final DCNN feature maps that contain output channels equal to the number of predicted classes) to capture context information. Recently, [41] proposes to learn a general and sparse high-dimensional convolution (bilateral convolution), and [82, 8] combine Gaussian Conditional Random Fields and DCNNs for semantic segmentation.

**上下文模块**：这种模型包括级联连接的额外模块，包含了长程上下文。一种有效的方法是，将DenseCRF[45]（包含高效的高维滤波算法[2]）与DCNNs[10,11]结合到一起。而且，[96,55,73]提出提出对CRF和DCNN进行联合训练，[59,90]在DCNN的特征图上使用了几个额外的卷积层，以捕获上下文信息。最近，[41]提出学习一种通用稀疏高维卷积（横向卷积），[82,8]将高斯CRF和DCNNs结合起来进行语义分割。

**Spatial pyramid pooling**: This model employs spatial pyramid pooling [28, 49] to capture context at several ranges. The image-level features are exploited in ParseNet [58] for global context information. DeepLabv2 [11] proposes atrous spatial pyramid pooling (ASPP), where parallel atrous convolution layers with different rates capture multi-scale information. Recently, Pyramid Scene Parsing Net (PSP) [95] performs spatial pooling at several grid scales and demonstrates outstanding performance on several semantic segmentation benchmarks. There are other methods based on LSTM [35] to aggregate global context [53, 6, 88]. Spatial pyramid pooling has also been applied in object detection [31].

**空间金字塔池化**：这种模型采用空间金字塔池化[28,49]以捕获各种距离的上下文。ParseNet[58]研究了图像层的特征，以利用全局上下文信息。DeepLabv2[11]提出了ASPP，利用不同比率的并行孔洞卷积层捕获多尺度信息。最近，金字塔场景解析网络(PSPNet)[95]在几种网格尺度上进行空间池化，在几个语义分割基准测试上都得到了非凡的性能。还有其他基于LSTM[35]的方法，可以集聚全局上下文[53,6,88]。空间金字塔池化在目标检测上也有应用[31]。

In this work, we mainly explore atrous convolution [36, 26, 74, 66, 10, 90, 11] as a context module and tool for spatial pyramid pooling. Our proposed framework is general in the sense that it could be applied to any network. To be concrete, we duplicate several copies of the original last block in ResNet [32] and arrange them in cascade, and also revisit the ASPP module [11] which contains several atrous convolutions in parallel. Note that our cascaded modules are applied directly on the feature maps instead of belief maps. For the proposed modules, we experimentally find it important to train with batch normalization [38]. To further capture global context, we propose to augment ASPP with image-level features, similar to [58, 95].

本文中，我们主要研究了将孔洞卷积用于一个上下文模块和工具，进行空间金字塔池化。我们提出的框架可以用于任何网络，因此是通用的。具体来说，我们对ResNet[32]最后一个模块进行了几次复制，将其进行级联，重新研究了ASPP模块，其中包含了几个并行的孔洞卷积。注意，我们的级联模块直接用于特征图，而不是belief maps。对于提出的模块，我们通过试验发现，用BN进行训练非常重要[38]。为进一步捕获全局上下文，我们提出用图像层的特征来增强ASPP，这与[58,95]类似。

**Atrous convolution**: Models based on atrous convolution have been actively explored for semantic segmentation. For example, [85] experiments with the effect of modifying atrous rates for capturing long-range information, [84] adopts hybrid atrous rates within the last two blocks of ResNet, while [18] further proposes to learn the deformable convolution which samples the input features with learned offset, generalizing atrous convolution. To further improve the segmentation model accuracy, [83] exploits image captions, [40] utilizes video motion, and [44] incorporates depth information. Besides, atrous convolution has been applied to object detection by [66, 17, 37].

**孔洞卷积**：基于孔洞卷积的模型在语义分割中已经得到了很多研究。比如，[85]修改了孔洞卷积的比率，通过试验研究这对捕获长程信息的影响；[84]采用在ResNet的最后两个模块中采用了混合孔洞率；[18]进一步提出，学习deformable卷积，其中用学习到的偏移来对输入特征进行采样，将孔洞卷积进行了推广。为进一步改进分割模型准确率，[83]研究了图像加标题，[40]利用了视频运动，[44]利用了深度信息。此外，孔洞卷积在目标检测中也有应用[66,17,37]。

## 3. Methods 方法

In this section, we review how atrous convolution is applied to extract dense features for semantic segmentation. We then discuss the proposed modules with atrous convolution modules employed in cascade or in parallel. 本节中，我们回顾了怎样应用孔洞卷积来提取密集特征，进行语义分割。然后我们讨论提出的使用了孔洞卷积的模块，以级联或并行方式进行利用。

### 3.1. Atrous Convolution for Dense Feature Extraction 使用孔洞卷积进行密集特征提取

Deep Convolutional Neural Networks (DCNNs) [50] deployed in fully convolutional fashion [74, 60] have shown to be effective for the task of semantic segmentation. However, the repeated combination of max-pooling and striding at consecutive layers of these networks significantly reduces the spatial resolution of the resulting feature maps, typically by a factor of 32 across each direction in recent DCNNs [47, 78, 32]. Deconvolutional layers (or transposed convolution) [92, 60, 64, 3, 71, 68] have been employed to recover the spatial resolution. Instead, we advocate the use of ‘atrous convolution’, originally developed for the efficient computation of the undecimated wavelet transform in the “algorithme à trous” scheme of [36] and used before in the DCNN context by [26, 74, 66].

DCNNs以全卷积方式部署，对于语义分割任务非常有效。但是，网络中连续层中都有最大池化和步长的重复组合，这显著降低了得到的特征图的空间分辨率，最近的DCNN中一般都会降低32倍。解卷积层（或转置卷积）曾经用于恢复空间分辨率。但是，我们推荐使用孔洞卷积，最开始提出来的时候，是用于计算undecimated小波变换，后来也层用于DCNN中。

Consider two-dimensional signals, for each location i on the output y and a filter w, atrous convolution is applied over the input feature map x: 考虑二维信号的情况，对位置i中的输出y，和滤波器w，孔洞卷积作用于输入特征图x：

$$y[i] = \sum_k x[i+r·k] · w[k]$$(1)

where the atrous rate r corresponds to the stride with which we sample the input signal, which is equivalent to convolving the input x with upsampled filters produced by inserting r − 1 zeros between two consecutive filter values along each spatial dimension (hence the name atrous convolution where the French word trous means holes in English). Standard convolution is a special case for rate r = 1, and atrous convolution allows us to adaptively modify filter’s field-of-view by changing the rate value. See Fig. 1 for illustration.

其中孔洞率r对应步长，我们用r步长来对输入信号进行采样，这等价于，将输入x与上采样的滤波器进行卷积，即在滤波器的两个相邻值中间插入r-1个零值。标准卷积是r=1的特殊情况，孔洞卷积使我们可以通过改变孔洞率来自适应的修改滤波器的视野。见图1。

Atrous convolution also allows us to explicitly control how densely to compute feature responses in fully convolutional networks. Here, we denote by output stride the ratio of input image spatial resolution to final output resolution. For the DCNNs [47, 78, 32] deployed for the task of image classification, the final feature responses (before fully connected layers or global pooling) is 32 times smaller than the input image dimension, and thus output_stride = 32. If one would like to double the spatial density of computed feature responses in the DCNNs (i.e., output_stride = 16), the stride of last pooling or convolutional layer that decreases resolution is set to 1 to avoid signal decimation. Then, all subsequent convolutional layers are replaced with atrous convolutional layers having rate r = 2. This allows us to extract denser feature responses without requiring learning any extra parameters. Please refer to [11] for more details.

孔洞卷积也使我们可以显式的控制全卷积网络中计算特征响应的密集度。这里，我们将输出步长表示为输入图像的空间分辨率与最后输出分辨率的比率。对于图像分类任务的DCNNs，最后的特征响应（在全连接层或全局池化前）比输入图像分辨率小32倍，所以输出步长output_stride = 32。如果希望将DCNNs中计算的特征响应空间密度加倍（即，output_stride = 16），最后一个降低分辨率的池化或卷积层的步长设置为1，以避免信号抽取。然后，所有后续的卷积层都替换为孔洞卷积层，孔洞率r=2。这使我们可以提取更密集的特征，而不需要学习任何额外的参数。详情请参考[11]。

### 3.2. Going Deeper with Atrous Convolution

We first explore designing modules with atrous convolution laid out in cascade. To be concrete, we duplicate several copies of the last ResNet block, denoted as block4 in Fig. 3, and arrange them in cascade. There are three 3 × 3 convolutions in those blocks, and the last convolution contains stride 2 except the one in last block, similar to original ResNet. The motivation behind this model is that the introduced striding makes it easy to capture long range information in the deeper blocks. For example, the whole image feature could be summarized in the last small resolution feature map, as illustrated in Fig. 3 (a). However, we discover that the consecutive striding is harmful for semantic segmentation (see Tab. 1 in Sec. 4) since detail information is decimated, and thus we apply atrous convolution with rates determined by the desired output stride value, as shown in Fig. 3 (b) where output_stride = 16.

我们首先研究一下，以级联的方式用孔洞卷积设计模块。具体来说，我们将ResNet最后的模块复制几个拷贝，如图3的block4所示，然后将其按级联的方式连接起来。在这些模块中，有3个3×3的卷积，最后的卷积的步长为2，除了在最后一个单元，这与原始的ResNet类似。这个模型背后的动机是，引入的步长使其在更深的模块中很容易捕获长程信息。比如，整体图像的特征可以在最后的小分辨率的特征图中得到总结，如图3(a)所示。但是，我们发现连续的步长对于语义分割是有害的（见第4节中的表1），因为细节信息被抽取没了，所以我们采用孔洞卷积，孔洞率由期望的输出步长值来确定，如图3(b)所示，其中output_stride=16。

In this proposed model, we experiment with cascaded ResNet blocks up to block7 (i.e., extra block5, block6, block7 as replicas of block4), which has output_stride = 256 if no atrous convolution is applied.

在提出的模型中，我们用级联的ResNet模块（即，block5,block6,block7都是block4的复制）进行试验，如果没有采用孔洞卷积的话，那么output_stride = 256。

Figure 3. Cascaded modules without and with atrous convolution. (a)Going deeper without atrous convolution; (b)Going deeper with atrous convolution. Atrous convolution with rate > 1 is applied after block3 when output_stride = 16.

#### 3.2.1 Multi-grid Method

Motivated by multi-grid methods which employ a hierarchy of grids of different sizes [4, 81, 5, 67] and following [84, 18], we adopt different atrous rates within block4 to block7 in the proposed model. In particular, we define as Multi_Grid = ($r_1, r_2, r_3$) the unit rates for the three convolutional layers within block4 to block7. The final atrous rate for the convolutional layer is equal to the multiplication of the unit rate and the corresponding rate. For example, when output_stride = 16 and Multi_Grid = (1, 2, 4), the three convolutions will have rates = 2 · (1, 2, 4) = (2, 4, 8) in the block4, respectively.

多网格方法采用不同大小的层次化网格[4,81,5,67]，受此启发，并按照[84,18]的思路，我们在提出的模型中从block4到block7采用不同的孔洞率。特别的，我们定义了多网格Multi_Grid=($r_1, r_2, r_3$)，就是block4到block7之间的三个卷积层的单元孔洞率。卷积层的最终孔洞率等于单元孔洞率和对应的孔洞率的乘积。比如，当output_stride=16，Multi_Grid=(1,2,4)时，block4中三个卷积的孔洞率分别为= 2 · (1, 2, 4) = (2, 4, 8)。

### 3.3. Atrous Spatial Pyramid Pooling

We revisit the Atrous Spatial Pyramid Pooling proposed in [11], where four parallel atrous convolutions with different atrous rates are applied on top of the feature map. ASPP is inspired by the success of spatial pyramid pooling [28, 49, 31] which showed that it is effective to resample features at different scales for accurately and efficiently classifying regions of an arbitrary scale. Different from [11], we include batch normalization within ASPP.

我们再讨论一下[11]中提出的ASPP，其中在特征图之上使用了4个不同孔洞率的并行孔洞卷积。ASPP是受到空间金字塔池化[28,49,31]的成功启发的，即将特征在不同尺度上重新采样可以准确有效的分类任意尺度的区域。与[11]不同，我们在ASPP中使用了BN。

ASPP with different atrous rates effectively captures multi-scale information. However, we discover that as the sampling rate becomes larger, the number of valid filter weights (i.e., the weights that are applied to the valid feature region, instead of padded zeros) becomes smaller. This effect is illustrated in Fig. 4 when applying a 3 × 3 filter to a 65 × 65 feature map with different atrous rates. In the extreme case where the rate value is close to the feature map size, the 3 × 3 filter, instead of capturing the whole image context, degenerates to a simple 1 × 1 filter since only the center filter weight is effective.

使用不同孔洞率的ASPP有效了捕获了多尺度信息。但是，我们发现，随着取样率变得越来越大，滤波器有效权重的数量（即，应用于有效特征区域的权重，而不是补零的部分）变得更小了。这种效应如图4所示，在65×65的特征图中，用不同孔洞率的3×3滤波器进行滤波。在极端情况下，当孔洞率接近特征图大小时，3×3滤波器不会捕获整个图像的上下文，而是蜕化为简单的1×1滤波器，因为只有中间的滤波器权重是有效的。

Figure 4. Normalized counts of valid weights with a 3 × 3 filter on a 65 × 65 feature map as atrous rate varies. When atrous rate is small, all the 9 filter weights are applied to most of the valid region on feature map, while atrous rate gets larger, the 3 × 3 filter degenerates to a 1 × 1 filter since only the center weight is effective.

To overcome this problem and incorporate global context information to the model, we adopt image-level features, similar to [58, 95]. Specifically, we apply global average pooling on the last feature map of the model, feed the resulting image-level features to a 1 × 1 convolution with 256 filters (and batch normalization [38]), and then bilinearly upsample the feature to the desired spatial dimension. In the end, our improved ASPP consists of (a) one 1×1 convolution and three 3 × 3 convolutions with rates = (6, 12, 18) when output_stride = 16 (all with 256 filters and batch normalization), and (b) the image-level features, as shown in Fig. 5. Note that the rates are doubled when output_stride = 8. The resulting features from all the branches are then concatenated and pass through another 1 × 1 convolution (also with 256 filters and batch normalization) before the final 1 × 1 convolution which generates the final logits.

为克服这个问题，在模型中把全局上下文信息也包括在内，我们采用了图像层的特征，与[58,95]类似。具体的，我们在模型最后的特征图上，使用了全局平均池化，将得到的图像层的特征送入1×1卷积中（有256个滤波器，和BN），然后将特征双线性上采样到期望的空间分辨率。最后，我们改进的ASPP包括以下两个部分：(a)一个1×1卷积，和3个3×3卷积，孔洞率=(6,12,18)，output_stride = 16（都是256个滤波器，包含BN）；(b)图像层的特征，如图5所示。注意，当output_stride = 8时，孔洞率会加倍。所有分支得到的特征会拼接到一起，然后送入另一个1×1卷积（也有256个滤波器，和BN），生成最后的logits。

Figure 5. Parallel modules with atrous convolution (ASPP), augmented with image-level features.

## 4. Experimental Evaluation 试验评估

We adapt the ImageNet-pretrained [72] ResNet [32] to the semantic segmentation by applying atrous convolution to extract dense features. Recall that output_stride is defined as the ratio of input image spatial resolution to final output resolution. For example, when output_stride = 8, the last two blocks (block3 and block4 in our notation) in the original ResNet contains atrous convolution with rate = 2 and rate = 4 respectively. Our implementation is built on TensorFlow [1].

我们采用ImageNet预训练的ResNet，通过应用孔洞卷积来提取密集特征，进行语义分割。回顾一下，output_stride定义为输入图像空间分辨率与最终输出分辨率的比率。比如，当output_stride = 8时，原始ResNet中最后两个block（我们的表示中，是block3和block4）分别包含rate=2和rate=4的孔洞卷积。我们的实现是用TensorFlow进行的。

We evaluate the proposed models on the PASCAL VOC 2012 semantic segmentation benchmark [20] which contains 20 foreground object classes and one background class. The original dataset contains 1, 464 (train), 1, 449 (val), and 1, 456 (test) pixel-level labeled images for training, validation, and testing, respectively. The dataset is augmented by the extra annotations provided by [29], resulting in 10, 582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union (IOU) averaged across the 21 classes.

我们在PASCAL VOC 2012语义分割基准测试[20]中评估提出的模型，测试中包含20个前景目标类别，一个背景类别。原始数据集包含1464（训练）、1449（验证）和1456（测试）像素层标注的图像，分别用于训练、验证和测试。数据集由[29]给的额外标注进行了扩充，得到了10582(trainaug)训练图像。性能是用像素IOU在21个类别上的平均度量的。

### 4.1. Training Protocol 训练方法

In this subsection, we discuss details of our training protocol. 在这个小节中，我们讨论了训练方法的细节。

**Learning rate policy**: Similar to [58, 11], we employ a “poly” learning rate policy where the initial learning rate is multiplied by $(1-\frac{iter}{max\_iter})^{power}$ with power = 0.9. **学习速率策略**：与[58,11]类似，我们采用了一种poly学习速率策略，其中初始学习速率乘以$(1-\frac{iter}{max\_iter})^{power}$，power=0.9。

**Crop size**: Following the original training protocol [10, 11], patches are cropped from the image during training. For atrous convolution with large rates to be effective, large crop size is required; otherwise, the filter weights with large atrous rate are mostly applied to the padded zero region. We thus employ crop size to be 513 during both training and test on PASCAL VOC 2012 dataset. **剪切块大小**：根据[10,11]的原始训练方法，剪切块是在训练过程中从图像中剪切出来的。对于大孔洞率的孔洞卷积，如果要有效的话，需要大的剪切大小；否则，大孔洞率的滤波器权重多数都会应用到补零的区域。我们在PASCAL VOC 2012数据集上，在训练和测试时，都采用513的剪切大小。

**Batch normalization**: Our added modules on top of ResNet all include batch normalization parameters [38], which we found important to be trained as well. Since large batch size is required to train batch normalization parameters, we employ output_stride = 16 and compute the batch normalization statistics with a batch size of 16. The batch normalization parameters are trained with decay = 0.9997. After training on the trainaug set with 30K iterations and initial learning rate = 0.007, we then freeze batch normalization parameters, employ output_stride = 8, and train on the official PASCAL VOC 2012 trainval set for another 30K iterations and smaller base learning rate = 0.001. Note that atrous convolution allows us to control output_stride value at different training stages without requiring learning extra model parameters. Also note that training with output_stride = 16 is several times faster than output_stride = 8 since the intermediate feature maps are spatially four times smaller, but at a sacrifice of accuracy since output_stride = 16 provides coarser feature maps.

**批归一化**：我们在ResNet上增加的模块都包含了批归一化的参数[38]，我们发现这对训练也非常重要。由于需要大的batch size来训练BN参数，我们采用output_stride=16，用batch size 16来计算BN统计参数。BN参数用decay=0.9997进行训练。在trainaug集上进行训练30K次迭代后，初始学习速率为0.007，我们就冻结了BN参数，采用output_stride=8，在官方的PASCAL VOC 2012 trainval集上训练另外30K次迭代，用更小的基准学习速率0.001。注意，孔洞卷积使我们可以在不同训练阶段控制output_stride值，而不需要学习额外的模型参数。同时注意，使用output_stride=16训练比采用output_stride=8训练要快上好几倍，因为中间特征图小了4倍，但会牺牲准确率，因为output_stride=16给出的是更粗糙的特征图。

**Upsampling logits**: In our previous works [10, 11], the target groundtruths are downsampled by 8 during training when output_stride = 8. We find it important to keep the groundtruths intact and instead upsample the final logits, since downsampling the groundtruths removes the fine annotations resulting in no back-propagation of details. **上采样logits**：在之前的工作中[10,11]，在output_stride=8时，目标真值在训练中会进行8倍下采样。我们发现保持真值完整是很重要的，所以将最终的logits上采样，因为真值下采样会丢失精细的标注，导致没有反向传播的细节。

**Data augmentation**: We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0) and randomly left-right flipping during training. **数据扩充**：我们通过随机变化输入图像的尺度（从0.5到2.0），和随机左右翻转，来进行数据扩充。

### 4.2. Going Deeper with Atrous Convolution

We first experiment with building more blocks with atrous convolution in cascade. 我们首先用孔洞卷积构建更多的模块形成级联来进行试验。

**ResNet-50**: In Tab. 1, we experiment with the effect of output_stride when employing ResNet-50 with block7 (i.e., extra block5, block6, and block7). As shown in the table, in the case of output_stride = 256 (i.e., no atrous convolution at all), the performance is much worse than the others due to the severe signal decimation. When output_stride gets larger and apply atrous convolution correspondingly, the performance improves from 20.29% to 75.18%, showing that atrous convolution is essential when building more blocks cascadedly for semantic segmentation.

**ResNet-50**：在表1中，在采用ResNet-50用作block7（即，block5,block6,block7）时，我们对output_stride的效果进行试验。如表所示，在output_stride=256时（即，没有孔洞卷积），由于严重的信号抽取，性能比其他情况差很多。当output_stride变大时，即使用了对应的孔洞卷积，性能从20.29%改进到75.18%，说明在对模块进行级联用于语义分割时，孔洞卷积是关键部分。

Table 1. Going deeper with atrous convolution when employing ResNet-50 with block7 and different output stride. Adopting output stride = 8 leads to better performance at the cost of more memory usage.

output_stride | 8 | 16 | 32 | 64 | 128 | 256
--- | --- | --- | --- | --- | --- | ---
mIOU | 75.18 | 73.88 | 70.06 | 59.99 | 42.34 | 20.29

**ResNet-50 vs. ResNet-101**: We replace ResNet-50 with deeper network ResNet-101 and change the number of cascaded blocks. As shown in Tab. 2, the performance improves as more blocks are added, but the margin of improvement becomes smaller. Noticeably, employing block7 to ResNet-50 decreases slightly the performance while it still improves the performance for ResNet-101.

**ResNet-50 vs. ResNet-101**：我们将ResNet-50替换为更深的ResNet-101，改变级联模块的数量。如表2所示，当增加更多的模块时，性能也响应的提升了，但改进的幅度越来越小。值得注意的是，block7采用ResNet-50性能略微降低，而采用ResNet-101则性能还在继续改善。

Table 2. Going deeper with atrous convolution when employing ResNet-50 and ResNet-101 with different number of cascaded blocks at output stride = 16. Network structures ‘block4’, ‘block5’, ‘block6’, and ‘block7’ add extra 0, 1, 2, 3 cascaded modules respectively. The performance is generally improved by adopting more cascaded blocks.

Network | block4 | block5 | block6 | block7
--- | --- | --- | --- | ---
ResNet-50 | 64.81 | 72.14 | 74.29 | 73.88
ResNet-101 | 68.39 | 73.21 | 75.34 | 75.76

**Multi-grid**: We apply the multi-grid method to ResNet-101 with several cascadedly added blocks in Tab. 3. The unit rates, Multi_Grid = ($r_1, r_2, r_3$), are applied to block4 and
all the other added blocks. As shown in the table, we observe that (a) applying multi-grid method is generally better than the vanilla version where ($r_1, r_2, r_3$) = (1, 1, 1), (b) simply doubling the unit rates (i.e., ($r_1, r_2, r_3$) = (2, 2, 2)) is not effective, and (c) going deeper with multi-grid improves the performance. Our best model is the case where block7 and ($r_1, r_2, r_3$) = (1, 2, 1) are employed.

**多网格**：我们对ResNet-101采用multi-grid方法，还有几个级联的添加模块，如表3所示。单元率，Multi_Grid = ($r_1, r_2, r_3$)，应用于block4和所有其他增加的模块。如表所示，我们观察到(a)用multi-grid方法一般都比传统版本要好，即($r_1, r_2, r_3$) = (1, 1, 1)的情况；(b)仅将单元率加倍（即($r_1, r_2, r_3$) = (2, 2, 2)）是无效的；(c)使用multi-grid方法在更深的情况下可以改进性能。我们最好的模型是在block7上使用($r_1, r_2, r_3$) = (1, 2, 1)的情况。

Table 3. Employing multi-grid method for ResNet-101 with different number of cascaded blocks at output stride = 16. The best model performance is shown in bold.

Multi-Grid | block4 | block5 | block6 | block7
--- | --- | --- | --- | ---
(1, 1, 1) | 68.39 | 73.21 | 75.34 | 75.76
(1, 2, 1) | 70.23 | 75.67 | 76.09 | **76.66**
(1, 2, 3) | 73.14 | 75.78 | 75.96 | 76.11
(1, 2, 4) | 73.45 | 75.74 | 75.85 | 76.02
(2, 2, 2) | 71.45 | 74.30 | 74.70 | 74.62

**Inference strategy on val set**: The proposed model is trained with output_stride = 16, and then during inference we apply output_stride = 8 to get more detailed feature map. As shown in Tab. 4, interestingly, when evaluating our best cascaded model with output_stride = 8, the performance improves over evaluating with output_stride = 16 by 1.39%. The performance is further improved by performing inference on multi-scale inputs (with scales = {0.5, 0.75, 1.0, 1.25, 1.5, 1.75}) and also left-right flipped images. In particular, we compute as the final result the average probabilities from each scale and flipped images.

**在val集上的推理策略**：提出的模型在训练时使用output_stride = 16，在推理时，我们使用的是output_stride = 8，以得到更详细的特征图。如表4所示，当评估最好的output_stride = 8级联模型时，性能比output_stride = 16的情况下多了1.39%。如果在多尺度（尺度={0.5,0.75,1.0,1.25,1.5,1.75}）输入上进行推理，以及输入左右翻转的图像，性能会得到进一步改进。特别的，我们计算在各个尺度和翻转图像上的平均概率作为最终结果。

Table 4. Inference strategy on the val set. MG: Multi-grid. OS: output_stride. MS: Multi-scale inputs during test. Flip: Adding left-right flipped inputs.

Method | OS=16 | OS=8 | MS | Flip | mIOU
--- | --- | --- | --- | --- | ---
block 7+ | y | n | n | n | 76.66
MG(1,2,1) | n | y | n | n | 78.05
| | n | y | y | n | 78.93
| | n | y | y | y | 79.35

### 4.3. Atrous Spatial Pyramid Pooling

We then experiment with the Atrous Spatial Pyramid Pooling (ASPP) module with the main differences from [11] being that batch normalization parameters [38] are fine-tuned and image-level features are included. 我们然后采用ASPP模块进行试验，与[11]的主要区别在于BN参数[38]是精调过的，还包括了图像层的特征。

**ASPP**: In Tab. 5, we experiment with the effect of incorporating multi-grid in block4 and image-level features to the improved ASPP module. We first fix ASPP =(6, 12, 18) (i.e., employ rates = (6, 12, 18) for the three parallel 3 × 3 convolution branches), and vary the multi-grid value. Employing Multi_Grid = (1, 2, 1) is better than Multi_Grid = (1, 1, 1), while further improvement is attained by adopting Multi_Grid = (1, 2, 4) in the context of ASPP = (6, 12, 18) (cf., the ‘block4’ column in Tab. 3). If we additionally employ another parallel branch with rate = 24 for longer range context, the performance drops slightly by 0.12%. On the other hand, augmenting the ASPP module with image-level feature is effective, reaching the final performance of 77.21%.

**ASPP**：在表5中，我们在ASPP模块中，在block4加入multi-grid，同时还加入了图像层的特征，以此进行试验。我们首先固定ASPP=(6,12,18)（即，对于三个并行的3×3卷积分支采用孔洞率=(6,12,18)），并改变multi-grid值。采用Multi_Grid = (1, 2, 1)比Multi_Grid = (1, 1, 1)要好，而Multi_Grid = (1, 2, 4)则有进一步的改进（参见表3中的block4列）。如果我们另外多用一个rate=24的并行分支，以获得更长程的上下文，性能反而会略微下降了0.12%。另一方面，用图像级的特征增强ASPP模块是有效的，最终性能达到了77.21%。

Table 5. Atrous Spatial Pyramid Pooling with multi-grid method and image-level features at output_stride = 16.

**Inference strategy on val set**: Similarly, we apply output_stride = 8 during inference once the model is trained. As shown in Tab. 6, employing output_stride = 8 brings 1.3% improvement over using output_stride = 16, adopting multi-scale inputs and adding left-right flipped images further improve the performance by 0.94% and 0.32%, respectively. The best model with ASPP attains the performance of 79.77%, better than the best model with cascaded atrous convolution modules (79.35%), and thus is selected as our final model for test set evaluation.

**在val集上的推理策略**：类似的，模型训练好之后，我们在推理时采用output_stride=8。如表6所示，采用output_stride=8比output_stride=16带来了1.3%的性能改进，使用多尺度输入和增加左右翻转图像分别进一步改进了0.94%和0.32%。使用ASPP的最好模型得到了79.77%的性能，比最好的级联孔洞模块(79.35%)要好，因此我们选择这种模型作为最后在测试集上进行评估的模型。

**Comparison with DeepLabv2**: Both our best cascaded model (in Tab. 4) and ASPP model (in Tab. 6) (in both cases without DenseCRF post-processing or MS-COCO pre-training) already outperform DeepLabv2 (77.69% with DenseCRF and pretrained on MS-COCO in Tab. 4 of [11]) on the PASCAL VOC 2012 val set. The improvement mainly comes from including and fine-tuning batch normalization parameters [38] in the proposed models and having a better way to encode multi-scale context.

**与DeepLabv2的比较**：我们最好的级联模型（在表4中）和ASPP模型（在表6中）（这两种情况下，都没有DenseCRF后处理或MS-COCO预训练），在PASCAL VOC 2012验证集上，都超过了DeepLabv2（77.69%，有DenseCRF后处理和在MS-COCO上预训练，见[11]中表4）。改进主要来自于BN参数的精调，和包含多尺度上下文的方法更好。

**Appendix**: We show more experimental results, such as the effect of hyper parameters and Cityscapes [14] results, in the appendix. 我们在附录中，展示了更多试验结果，比如超参数的影响和在Cityscapes数据集上的结果。

**Qualitative results**: We provide qualitative visual results of our best ASPP model in Fig. 6. As shown in the figure, our model is able to segment objects very well without any DenseCRF post-processing. 我们在图6中给出了最好的ASPP模型的可视化结果。如图所示，我们的模型在没有DenseCRF后处理的情况下，也可以将目标分割的非常好。

**Failure mode**: As shown in the bottom row of Fig. 6, our model has difficulty in segmenting (a) sofa vs. chair, (b) dining table and chair, and (c) rare view of objects. 错误模式：我们在图6的最下面一行，给出了模型很难分割的几种情况：(a)沙发和椅子；(b)餐桌和椅子；(c)目标的少见视角。

Figure 6. Visualization results on the val set when employing our best ASPP model. The last row shows a failure mode.

**Pretrained on COCO**: For comparison with other state-of-art models, we further pretrain our best ASPP model on MS-COCO dataset [57]. From the MS-COCO trainval minus minival set, we only select the images that have annotation regions larger than 1000 pixels and contain the classes defined in PASCAL VOC 2012, resulting in about 60K images for training. Besides, the MS-COCO classes not defined in PASCAL VOC 2012 are all treated as background class. After pretraining on MS-COCO dataset, our proposed model attains performance of 82.7% on val set when using output_stride = 8, multi-scale inputs and adding left-right flipped images during inference. We adopt smaller initial learning rate = 0.0001 and same training protocol as in Sec. 4.1 when fine-tuning on PASCAL VOC 2012 dataset.

**在COCO上预训练**：为和其他目前最好的模型比较，我们进一步将我们最好的ASPP模型在MS-COCO上进行预训练。在MS-COCO tranval-minval集上，我们只选择了那些图像标注区域大于1000像素的，包含定义在PASCAL VOC 2012上的类别的图像，最后得到了60K图像进行训练。此外，在PASCAL VOC 2012中没有定义的MS-COCO类别都被当做背景类别。在MS-COCO数据集上预训练后，我们提出的模型在val集上得到了82.7%的性能，推理时使用的output_stride=8，多尺度输出，并增加了左右翻转图像。我们在PASCAL VOC 2012数据集上精调时，采用了与4.1节类似的初始学习速率0.0001和相同的训练方案。

**Test set result and an effective bootstrapping method**: We notice that PASCAL VOC 2012 dataset provides higher quality of annotations than the augmented dataset [29], especially for the bicycle class. We thus further fine-tune our model on the official PASCAL VOC 2012 trainval set before evaluating on the test set. Specifically, our model is trained with output_stride = 8 (so that annotation details are kept) and the batch normalization parameters are frozen (see Sec. 4.1 for details). Besides, instead of performing pixel hard example mining as [85, 70], we resort to bootstrapping on hard images. In particular, we duplicate the images that contain hard classes (namely bicycle, chair, table, potted-plant, and sofa) in the training set. As shown in Fig. 7, the simple bootstrapping method is effective for segmenting the bicycle class. In the end, our ‘DeepLabv3’ achieves the performance of 85.7% on the test set without any DenseCRF post-processing, as shown in Tab. 7.

**在测试集上的结果和一种有效的提升方法**：我们注意到，PASCAL VOC 2012数据集的标注质量比增强数据集[29]要好，尤其是自行车类别。所以我们进一步在官方的PASCAL VOC 2012 trainval集上精调我们的模型，然后再在测试集上进行评估。具体来说，我们的模型使用output_stride = 8进行训练（这样可以保留标注细节），BN参数冻结住了（详见4.1节）。另外，我们没有进行像素难分样本挖掘[85,70]，而是在难分图像上尝试提升效果。特别的，我们在训练集中对包含难分类别的图像进行复制（即，自行车，椅子，桌子，盆栽，和沙发）。如图7所示，这种简单的提升方法对分割自行车类别是有效的。最后，我们的DeepLabv3在测试集上取得了85.7%的性能，没有采用DenseCRF后处理，如表7所示。

Figure 7. Bootstrapping on hard images improves segmentation accuracy for rare and finely annotated classes such as bicycle.

Table 7. Performance on PASCAL VOC 2012 test set.

Method | mIOU
--- | ---
Adelaide VeryDeep FCN VOC [85] | 79.1
LRR 4x ResNet-CRF [25] | 79.3
DeepLabv2-CRF [11] | 79.7
CentraleSupelec Deep G-CRF [8] | 80.2
HikSeg COCO [80] | 81.4
SegModel [75] | 81.8
Deep Layer Cascade (LC) [52] | 82.7
TuSimple [84] | 83.1
Large Kernel Matters [68] | 83.6
Multipath-RefineNet [54] | 84.2
ResNet-38 MS COCO [86] | 84.9
PSPNet [95] | 85.4
IDW-CNN [83] | 86.3
CASIA IVA SDN [23] | 86.6
DIS [61] | 86.8
DeepLabv3 | 85.7
DeepLabv3-JFT | 86.9

**Model pretrained on JFT-300M**: Motivated by the recent work of [79], we further employ the ResNet-101 model which has been pretraind on both ImageNet and the JFT-300M dataset [34, 13, 79], resulting in a performance of 86.9% on PASCAL VOC 2012 test set.

**在JFT-300M上预训练模型**：受[79]启发，我们进一步采用在ImageNet和JFT-300M数据集上预训练的ResNet-101模型，得到的模型在PASCAL VOC 2012测试集得到了86.9%的性能。

## 5. Conclusion 结论

Our proposed model “DeepLabv3” employs atrous convolution with upsampled filters to extract dense feature maps and to capture long range context. Specifically, to encode multi-scale information, our proposed cascaded module gradually doubles the atrous rates while our proposed atrous spatial pyramid pooling module augmented with image-level features probes the features with filters at multiple sampling rates and effective field-of-views. Our experimental results show that the proposed model significantly improves over previous DeepLab versions and achieves comparable performance with other state-of-art models on the PASCAL VOC 2012 semantic image segmentation benchmark.

我们提出的DeepLabv3模型，采用了上采样滤波器的孔洞卷积，以提取密集特征图，来捕获长程上下文。特别的，为包含多尺度信息，我们提出的级联模块的孔洞率逐渐加倍，而我们提出的ASPP模块使用图像层的特征得到增强，用多种上采样率和多种有效视野来提取特征。我们的试验结果表明，提出的模型比之前的DeepLab版本有显著的改进，在PASCAL VOC 2012语义分割基准测试中得到了与其他目前最好的模型类似的性能。

## A. Effect of hyper-parameters

## B. Asynchronous training

## C. DeepLabv3 on Cityscapes dataset