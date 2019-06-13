# Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs

Liang-Chieh Chen et al. UCLA Google Inc.

## Abstract 摘要

Deep Convolutional Neural Networks (DCNNs) have recently shown state of the art performance in high level vision tasks, such as image classification and object detection. This work brings together methods from DCNNs and probabilistic graphical models for addressing the task of pixel-level classification (also called ”semantic image segmentation”). We show that responses at the final layer of DCNNs are not sufficiently localized for accurate object segmentation. This is due to the very invariance properties that make DCNNs good for high level tasks. We overcome this poor localization property of deep networks by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF). Qualitatively, our “DeepLab” system is able to localize segment boundaries at a level of accuracy which is beyond previous methods. Quantitatively, our method sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 71.6% IOU accuracy in the test set. We show how these results can be obtained efficiently: Careful network re-purposing and a novel application of the ’hole’ algorithm from the wavelet community allow dense computation of neural net responses at 8 frames per second on a modern GPU.

深度卷积神经网络(DCNNs)在高层视觉任务中取得了目前最好的表现，如图像分类和目标检测。本文将DCNNs与概率图模型结合到一起，解决像素层次的分类任务问题（也称为“语义图像分割”）。我们证明了，DCNNs的最终层的响应不足以进行精确的目标分割定位。这是因为其不变性造成的，也正是这种不变性使其在高层视觉任务中表现如此之好。我们将DCNN最终层的响应，与一个全连接的条件随机场(CRF)相结合，解决了深度网络的这种不好的定位性质。定性的说，我们的DeepLab系统可以很准确的定位分割边界，比之前所有方法都要精确。定量的说，我们的方法在PASCAL VOC-2012语义分割任务中得到了最好成绩，在测试集中达到了71.6% IOU准确率。我们展示了怎样高效的取得这种结果：仔细的改变网络的目的，并采用了小波研究中的一种孔洞算法，进行新的应用，可以在现代GPU上以8fps的速度进行神经网络的密集计算。

## 1 Introduction 引言

Deep Convolutional Neural Networks (DCNNs) had been the method of choice for document recognition since LeCun et al. (1998), but have only recently become the mainstream of high-level vision research. Over the past two years DCNNs have pushed the performance of computer vision systems to soaring heights on a broad array of high-level problems, including image classification (Krizhevsky et al., 2013; Sermanet et al., 2013; Simonyan & Zisserman, 2014; Szegedy et al., 2014; Papandreou et al., 2014), object detection (Girshick et al., 2014), fine-grained categorization (Zhang et al., 2014), among others. A common theme in these works is that DCNNs trained in an end-to-end manner deliver strikingly better results than systems relying on carefully engineered representations, such as SIFT or HOG features. This success can be partially attributed to the built-in invariance of DCNNs to local image transformations, which underpins their ability to learn hierarchical abstractions of data (Zeiler & Fergus, 2014). While this invariance is clearly desirable for high-level vision tasks, it can hamper low-level tasks, such as pose estimation (Chen & Yuille, 2014; Tompson et al., 2014) and semantic segmentation - where we want precise localization, rather than abstraction of spatial details.

DCNNs自从1998年LeCun等就是文档识别的一种方法，但直到最近才成为高层视觉研究的主流。在过去两年中，DCNNs将计算机视觉系统的性能在多种高层问题上推向了非常高的性能，包括图像分类，目标检测，细粒度分类等等。这些工作的常见方法是端到端的训练DCNNs，得到的效果比仔细手工设计的表示效果要好的多，如SIFT或HOG特征。这种成功部分归功于DCNNs对局部图像变换的内在不变性，使其可以层次化的学习数据的抽象特征。这种不变性对于高层视觉任务非常理想，但却会影响低层任务，如姿态估计和语义分割，其中我们需要精确的定位，而不是空间细节的抽象。

There are two technical hurdles in the application of DCNNs to image labeling tasks: signal downsampling, and spatial ‘insensitivity’ (invariance). The first problem relates to the reduction of signal resolution incurred by the repeated combination of max-pooling and downsampling (‘striding’) performed at every layer of standard DCNNs (Krizhevsky et al., 2013; Simonyan & Zisserman, 2014; Szegedy et al., 2014). Instead, as in Papandreou et al. (2014), we employ the ‘atrous’ (with holes) algorithm originally developed for efficiently computing the undecimated discrete wavelet transform (Mallat, 1999). This allows efficient dense computation of DCNN responses in a scheme substantially simpler than earlier solutions to this problem (Giusti et al., 2013; Sermanet et al., 2013).

DCNNs在图像标记任务中的应用主要有两个技术障碍：信号的下采样，和对空间的不敏感性（不变性）。第一个问题与信号分辨率的降低有关，这是不断的max-pooling和下采样造成的，这是标准DCNNs的每一层都有的。我们则采用了atrous算法，这是为高效计算undecimated离散小波变换提出的算法。这可以高效计算DCNN的密集响应，比之前的方法要简单的多。

The second problem relates to the fact that obtaining object-centric decisions from a classifier requires invariance to spatial transformations, inherently limiting the spatial accuracy of the DCNN model. We boost our model’s ability to capture fine details by employing a fully-connected Conditional Random Field (CRF). Conditional Random Fields have been broadly used in semantic segmentation to combine class scores computed by multi-way classifiers with the low-level information captured by the local interactions of pixels and edges (Rother et al., 2004; Shotton et al., 2009) or superpixels (Lucchi et al., 2011). Even though works of increased sophistication have been proposed to model the hierarchical dependency (He et al., 2004; Ladicky et al., 2009; Lempitsky et al., 2011) and/or high-order dependencies of segments (Delong et al., 2012; Gonfaus et al., 2010; Kohli et al., 2009; Chen et al., 2013; Wang et al., 2015), we use the fully connected pairwise CRF proposed by Krähenbühl & Koltun (2011) for its efficient computation, and ability to capture fine edge details while also catering for long range dependencies. That model was shown in Krähenbühl & Koltun (2011) to largely improve the performance of a boosting-based pixel-level classifier, and in our work we demonstrate that it leads to state-of-the-art results when coupled with a DCNN-based pixel-level classifier.

第二个问题与下面的事实相关，从分类器中得到以目标为中心的决策，需要对空间变换具有不变性，这根本上限制了DCNN模型的空间准确率。为提升我们模型捕获精细细节的能力，我们采用了一个全连接的条件随机场(CRF)。CRF在语义分割中有广泛的使用，将多路分类器得到的类别分数，与像素和边缘局部互动的底层信息结合起来，或与超像素结合起来。即使提出了更为复杂的模型来对层次化依赖关系和片段的高阶依赖关系进行建模，我们使用全连接的成对CRF进行高效的计算，并捕获边缘的精细细节，同时照顾到长程依赖关系。这个模型在另篇文章中大幅改善了基于boosting的像素层次分类器的性能，在我们的工作中，我们证明了，在与基于DCNN的像素层分类器结合后，可以得到目前最好的结果。

The three main advantages of our “DeepLab” system are (i) speed: by virtue of the ‘atrous’ algorithm, our dense DCNN operates at 8 fps, while Mean Field Inference for the fully-connected CRF requires 0.5 second, (ii) accuracy: we obtain state-of-the-art results on the PASCAL semantic segmentation challenge, outperforming the second-best approach of Mostajabi et al. (2014) by a margin of 7.2% and (iii) simplicity: our system is composed of a cascade of two fairly well-established modules, DCNNs and CRFs.

我们的DeepLab系统的三个主要优势是：(i)速度：由于使用了atrous算法，我们的DCNN可以达到8fps的运行速度，而全连接CRF的Mean Field Inference需要0.5s；(ii)准确度：我们在PASCAL语义分割挑战上得到了目前最好的结果，超过了第二名方法7.2%；(iii)简单：我们的系统由于两类成熟模块级联构成，即DCNNs和CRFs。

## 2 Related Work 相关工作

Our system works directly on the pixel representation, similarly to Long et al. (2014). This is in contrast to the two-stage approaches that are now most common in semantic segmentation with DCNNs: such techniques typically use a cascade of bottom-up image segmentation and DCNN-based region classification, which makes the system commit to potential errors of the front-end segmentation system. For instance, the bounding box proposals and masked regions delivered by (Arbeláez et al., 2014; Uijlings et al., 2013) are used in Girshick et al. (2014) and (Hariharan et al., 2014b) as inputs to a DCNN to introduce shape information into the classification process. Similarly, the authors of Mostajabi et al. (2014) rely on a superpixel representation. A celebrated non-DCNN precursor to these works is the second order pooling method of (Carreira et al., 2012) which also assigns labels to the regions proposals delivered by (Carreira & Sminchisescu, 2012). Understanding the perils of committing to a single segmentation, the authors of Cogswell et al. (2014) build on (Yadollah-pour et al., 2013) to explore a diverse set of CRF-based segmentation proposals, computed also by (Carreira & Sminchisescu, 2012). These segmentation proposals are then re-ranked according to a DCNN trained in particular for this reranking task. Even though this approach explicitly tries to handle the temperamental nature of a front-end segmentation algorithm, there is still no explicit exploitation of the DCNN scores in the CRF-based segmentation algorithm: the DCNN is only applied post-hoc, while it would make sense to directly try to use its results during segmentation.

我们的系统直接在像素表示上工作，与另一篇工作类似。这与两阶段方法形成对比，两阶段方法在DCNNs进行语义分割的工作中现在颇为普遍：这种技术一般使用自下而上的图像分割，与基于DCNN的区域分类相级联，这使系统容易受前置分割系统的错误影响。比如，一篇文献得到的边界框建议和掩膜区域用作另外文献的DCNN的输入，为分类过程提供区域信息。类似的，一篇文献依赖于超像素表示作为输入。这类工作的一个注明的非DCNN先驱，是两阶pooling方法，也使用了另一篇文献提出的对区域指定标签的方法。为理解依赖单一分割的危害，一篇文献探索了基于CRF的多种分割建议。这些分割建议重新进行了排序，根据是为这种重新排序工作特意设计的DCNN。即使这个方法显式的想要处理前置分割算法的不良本质，但还是没有在基于CRF的分割算法中显式的利用DCNN得分：DCNN只是事后进行应用，而应当直接尝试在分割的过程中使用其结果才有意义。

Moving towards works that lie closer to our approach, several other researchers have considered the use of convolutionally computed DCNN features for dense image labeling. Among the first have been Farabet et al. (2013) who apply DCNNs at multiple image resolutions and then employ a segmentation tree to smooth the prediction results; more recently, Hariharan et al. (2014a) propose to concatenate the computed inter-mediate feature maps within the DCNNs for pixel classification, and Dai et al. (2014) propose to pool the inter-mediate feature maps by region proposals. Even though these works still employ segmentation algorithms that are decoupled from the DCNN classifier’s results, we believe it is advantageous that segmentation is only used at a later stage, avoiding the commitment to premature decisions.

也有与我们的方法更接近的工作，一些其他研究者考虑使用DCNN特征进行密集图像标记。第一个在多种图像分辨率上使用DCNN，然后使用一种分割树以平滑预测结果；最近，一位研究者提出，将DCNN计算的中间特征图拼接起来，以进行像素分类，另一位提出对中间特征图通过区域建议进行池化。即使这些工作采用的分割算法仍然与DCNN分类器结果是非耦合的，我们相信还是有优势的，因为分割是用在后一个阶段，防止了早期承诺问题。

More recently, the segmentation-free techniques of (Long et al., 2014; Eigen & Fergus, 2014) directly apply DCNNs to the whole image in a sliding window fashion, replacing the last fully connected layers of a DCNN by convolutional layers. In order to deal with the spatial localization issues outlined in the beginning of the introduction, Long et al. (2014) upsample and concatenate the scores from inter-mediate feature maps, while Eigen & Fergus (2014) refine the prediction result from coarse to fine by propagating the coarse results to another DCNN.

最近，非分割方法直接将DCNNs以滑窗的方式应用到整个图像中，将DCNN的最后一个全连接层替换为卷积层。为处理引言部分提出的空间定位问题，一位研究者对最后结果进行上采样，并与中间特征图的得分进行拼接，另一篇文章通过将粗糙结果输入另一个DCNN中，从而由粗糙到精细的精炼预测结果。

The main difference between our model and other state-of-the-art models is the combination of pixel-level CRFs and DCNN-based ‘unary terms’. Focusing on the closest works in this direction, Cogswell et al. (2014) use CRFs as a proposal mechanism for a DCNN-based reranking system, while Farabet et al. (2013) treat superpixels as nodes for a local pairwise CRF and use graph-cuts for discrete inference; as such their results can be limited by errors in superpixel computations, while ignoring long-range superpixel dependencies. Our approach instead treats every pixel as a CRF node, exploits long-range dependencies, and uses CRF inference to directly optimize a DCNN-driven cost function. We note that mean field had been extensively studied for traditional image segmentation/edge detection tasks, e.g., (Geiger & Girosi, 1991; Geiger & Yuille, 1991; Kokkinos et al., 2008), but recently Krähenbühl & Koltun (2011) showed that the inference can be very efficient for fully connected CRF and particularly effective in the context of semantic segmentation.

我们的模型与其他目前最好的模型之间的区别是我们将像素级的CRFs与基于DCNN的项结合了起来。聚焦在这个方向最接近的工作，一篇文献使用CRFs为一个基于DCNN的重排序系统作为建议机制，另一篇文献将超像素作为一个局部成对CRF的节点，使用图分割进行离散推理；这样他们的结果就受限于超像素计算的错误率，同时忽略了长程超像素的依赖关系。我们的方法则将每个像素都作为CRF节点，利用了长程依赖关系，使用CRF推理直接优化DCNN驱动的损失函数。我们注意到，mean field在传统图像分割/边缘检测任务中得到了广泛的研究，但最近一篇文献证明了对于全连接的CRF会非常高效，在语义分割的情况下尤其高效。

After the first version of our manuscript was made publicly available, it came to our attention that two other groups have independently and concurrently pursued a very similar direction, combining DCNNs and densely connected CRFs (Bell et al., 2014; Zheng et al., 2015). There are several differences in technical aspects of the respective models. In terms of applications, Bell et al. (2014) focus on the problem of material classification. Similarly to us, Zheng et al. (2015) evaluate their system on the problem of semantic image segmentation but their results on the PASCAL VOC 2012 benchmark are somewhat inferior to ours. We refer the interested reader to these papers for different perspectives on the interplay of DCNNs and CRFs.

在我们公开手稿第一版后，我们意识到，其他两个组也在独立并同时的进行类似方向的工作，将DCNNs与密集连接的CRFs结合起来。模型之间与一些区别。应用上来说，一篇关注物质分类的问题。与我们相似的一篇，在语义分割问题上进行试验，但他们在PASCAL VOC 2012上的结果比我们要差一些。我们推荐读者也参考这些文章。

## 3 Convolutional Neural Networks for Dense Image Labelling

Herein we describe how we have re-purposed and finetuned the publicly available Imagenet-pretrained state-of-art 16-layer classification network of (Simonyan & Zisserman, 2014) (VGG-16) into an efficient and effective dense feature extractor for our dense semantic image segmentation system.

在此我们描述一下我们怎样将公开可用的ImageNet预训练的目前最好的16层分类网络VGG-16重新改变用途并精调，用作高效的密集特征提取器，然后进行密集语义分割系统的。

### 3.1 Efficient Dense Sliding Window Feature Extraction with the Hole Algorithm

Dense spatial score evaluation is instrumental in the success of our dense CNN feature extractor. As a first step to implement this, we convert the fully-connected layers of VGG-16 into convolutional ones and run the network in a convolutional fashion on the image at its original resolution. However this is not enough as it yields very sparsely computed detection scores (with a stride of 32 pixels). To compute scores more densely at our target stride of 8 pixels, we develop a variation of the method previously employed by Giusti et al. (2013); Sermanet et al. (2013). We skip subsampling after the last two max-pooling layers in the network of Simonyan & Zisserman (2014) and modify the convolutional filters in the layers that follow them by introducing zeros to increase their length (2×in the last three convolutional layers and 4× in the first fully connected layer). We can implement this more efficiently by keeping the filters intact and instead sparsely sample the feature maps on which they are applied on using an input stride of 2 or 4 pixels, respectively. This approach, illustrated in Fig. 1 is known as the ‘hole algorithm’ (‘atrous algorithm’) and has been developed before for efficient computation of the undecimated wavelet transform (Mallat, 1999). We have implemented this within the Caffe framework (Jia et al., 2014) by adding to the im2col function (it converts multi-channel feature maps to vectorized patches) the option to sparsely sample the underlying feature map. This approach is generally applicable and allows us to efficiently compute dense CNN feature maps at any target subsampling rate without introducing any approximations.

密集空间分数评估在我们的密集CNN特征提取器的成功中非常重要。作为实现这个的第一步，我们将VGG-16的全连接层转化成卷积层，对图像进行卷积式的处理，维持原始分辨率。但这是不够的，因为生成的是非常稀疏的检测分数（步长为32像素）。为更密集的计算分数，得到我们目标的步长8像素，我们提出了一篇文献中这种算法的变体；我们跳过了网络中最后两个max-pooling层后的下采样，修改了层中之后的卷积滤波器，为滤波器中增加零值，以增加其长度（最后三个卷积层增加2倍，最后一个全连接层增加4倍）。我们有一种更高效的实现方式，即保持滤波器不变，但对特征图进行稀疏采样，使用输入步长分别为2像素或4像素。这种方法如图1所示，称为孔洞算法(atrous algorithm)，早年提出这种算法是为了高效的计算undecimated小波变换。我们在Caffe框架下实现了这种计算，即对im2col函数增加了一个选项参数，可以对特征图进行稀疏采样（im2col将多通道特征图转化为向量化的块）。这种方法在一般情况下都是可用的，使我们可以在任何目标下采样率下高效的计算密集CNN特征图，不需要采用任何近似。

Figure 1: Illustration of the hole algorithm in 1-D, when kernel size = 3, input stride = 2, and output stride = 1.

We finetune the model weights of the Imagenet-pretrained VGG-16 network to adapt it to the image classification task in a straightforward fashion, following the procedure of Long et al. (2014). We replace the 1000-way Imagenet classifier in the last layer of VGG-16 with a 21-way one. Our loss function is the sum of cross-entropy terms for each spatial position in the CNN output map (subsampled by 8 compared to the original image). All positions and labels are equally weighted in the overall loss function. Our targets are the ground truth labels (subsampled by 8). We optimize the objective function with respect to the weights at all network layers by the standard SGD procedure of Krizhevsky et al. (2013).

我们对ImageNet预训练的VGG-16精调模型权重，以适应图像分类任务，采用的是文献中的方法。我们将VGG-16中最后一层的1000路的ImageNet分类器替换为21路。我们的损失函数是在CNN输出特征图每个空间位置的交叉熵项的和（与原始图像相比，下采样率为8）。所有位置和标签在总体的损失函数中权重都一样。我们的目标是真值标签（进行8倍下采样）。我们对网络中所有层的权重优化目标函数，使用标准SGD过程。

During testing, we need class score maps at the original image resolution. As illustrated in Figure 2 and further elaborated in Section 4.1, the class score maps (corresponding to log-probabilities) are quite smooth, which allows us to use simple bilinear interpolation to increase their resolution by a factor of 8 at a negligible computational cost. Note that the method of Long et al. (2014) does not use the hole algorithm and produces very coarse scores (subsampled by a factor of 32) at the CNN output. This forced them to use learned upsampling layers, significantly increasing the complexity and training time of their system: Fine-tuning our network on PASCAL VOC 2012 takes about 10 hours, while they report a training time of several days (both timings on a modern GPU).

在测试过程中，我们需要原始图像分辨率大小的类别分数图。如图2所示，在4.1节中进行了详述，类别分数图（对应log概率）非常平滑，使我们可以使用简单的双线性插值来提高8倍分辨率，计算量基本可以忽略。注意，文献中的方法没有使用atrous算法，在CNN输出处生成了非常粗糙的分数（下采样率32）。这使其必须使用学习的上采样层，极大的增加了复杂度，以及系统的训练时间：在PASCAL VOC 2012上精调我们的网络大约需要10小时，而文献中的方法则需要好几天（都是在现代GPU上运行的时间）。

### 3.2 Controlling the Receptive Field Size and Accelerating Dense Computation with Convolutional Nets

Another key ingredient in re-purposing our network for dense score computation is explicitly controlling the network’s receptive field size. Most recent DCNN-based image recognition methods rely on networks pre-trained on the Imagenet large-scale classification task. These networks typically have large receptive field size: in the case of the VGG-16 net we consider, its receptive field is 224 × 224 (with zero-padding) and 404 × 404 pixels if the net is applied convolutionally. After converting the network to a fully convolutional one, the first fully connected layer has 4,096 filters of large 7 × 7 spatial size and becomes the computational bottleneck in our dense score map computation.

我们将网络的目标重新设定为密集分数计算，其另一个关键部分是，显式的控制网络感受野大小。多数现代基于DCNN的图像识别方法，都依赖于在ImageNet大型分类任务上预训练的网络。这些网络一般都有很大的感受野大小：在VGG-16中，其感受野大小为224×224（加上补零），如果网络是卷积方式计算的，那么就是404×404大小。在将网络转化成全卷积之后，第一个全连接层有4096个滤波器，大小7×7，这成为了我们的密集分数图计算的计算瓶颈。

We have addressed this practical problem by spatially subsampling (by simple decimation) the first FC layer to 4×4 (or 3×3) spatial size. This has reduced the receptive field of the network down to 128×128 (with zero-padding) or 308×308 (in convolutional mode) and has reduced computation time for the first FC layer by 2 − 3 times. Using our Caffe-based implementation and a Titan GPU, the resulting VGG-derived network is very efficient: Given a 306×306 input image, it produces 39×39 dense raw feature scores at the top of the network at a rate of about 8 frames/sec during testing. The speed during training is 3 frames/sec. We have also successfully experimented with reducing the number of channels at the fully connected layers from 4,096 down to 1,024, considerably further decreasing computation time and memory footprint without sacrificing performance, as detailed in Section 5. Using smaller networks such as Krizhevsky et al. (2013) could allow video-rate test-time dense feature computation even on light-weight GPUs.

我们解决这个实际问题的方法是，将第一个全连接层进行空间下采样（简单的抽取）成为4×4（或3×3）空间大小。这将网络的感受野降低到了128×128（带补零）或308×308（在卷积模式下），将第一个全连接层的计算时间降低了2-3倍。在一个Titan GPU上使用我们基于Caffe的实现，得到的VGG-衍生网络是非常高效的：给定306×306的输入图像，网络生成39×39的密集原生特征分数，测试时运行速度为8fps。训练时为3fps。我们还将全连接层的通道数从4096减少到了1024，成功的进行了试验，进一步显著降低了计算时间和内存的占用量，而没有牺牲性能，这在第5部分进行了详述。使用更小的网络，即使在轻量GPU上也可以得到视频速率的密集特征计算速度。

## 4 Detailed Boundary Recovery: Fully-Connected Conditional Random Fields and Multi-Scale Prediction

### 4.1 Deep Convolutional Networks and the Localization Chanllenge

As illustrated in Figure 2, DCNN score maps can reliably predict the presence and rough position of objects in an image but are less well suited for pin-pointing their exact outline. There is a natural trade-off between classification accuracy and localization accuracy with convolutional networks: Deeper models with multiple max-pooling layers have proven most successful in classification tasks, however their increased invariance and large receptive fields make the problem of inferring position from the scores at their top output levels more challenging.

如图2所示，DCNN的分数图可以非常可靠的预测目标的存在和大致位置，但不太适合于精确确定其区域。卷积网络在分类准确率和定位准确率之间有很自然的折中关系：更深的模型有多个max-pooling层，在分类任务中非常成功，但是其增强的不变性和大型感受野使得从分数中推断目标位置非常有挑战性。

Recent work has pursued two directions to address this localization challenge. The first approach is to harness information from multiple layers in the convolutional network in order to better estimate the object boundaries (Long et al., 2014; Eigen & Fergus, 2014). The second approach is to employ a super-pixel representation, essentially delegating the localization task to a low-level segmentation method. This route is followed by the very successful recent method of Mostajabi et al. (2014).

在解决定位的问题上，最近的工作有两个方向。第一种是利用卷积网络中多层的信息，来更好的估计目标边缘。第二种方法是采用一种超像素表示，将定位问题交给了底层分割方法。这条线路最近有一个很成功的方法，见文献。

In Section 4.2, we pursue a novel alternative direction based on coupling the recognition capacity of DCNNs and the fine-grained localization accuracy of fully connected CRFs and show that it is remarkably successful in addressing the localization challenge, producing accurate semantic segmentation results and recovering object boundaries at a level of detail that is well beyond the reach of existing methods.

在4.2中，我们采用了一个新的方向，基于将DCNN的识别能力与全连接CRF的细粒度定位准确率结合起来，证明了这在解决定位问题中非常成功，得到了准确的语义分割结果，在很细节的层次恢复了目标边缘，远远超过了现有的方法。

Figure 2: Score map (input before softmax function) and belief map (output of softmax function) for Aeroplane. We show the score (1st row) and belief (2nd row) maps after each mean field iteration. The output of last DCNN layer is used as input to the mean field inference. Best viewed in color.

### 4.2 Fully-Connected Conditional Random Fields for Accurate Localization

Traditionally, conditional random fields (CRFs) have been employed to smooth noisy segmentation maps (Rother et al., 2004; Kohli et al., 2009). Typically these models contain energy terms that couple neighboring nodes, favoring same-label assignments to spatially proximal pixels. Qualitatively, the primary function of these short-range CRFs has been to clean up the spurious predictions of weak classifiers built on top of local hand-engineered features.

传统上，条件随机场(CRFs)已经用于含噪分割图的平滑。这些模型很典型的都包含能量项，将相邻的节点结合起来，倾向于对空间上相邻的像素点指定给相同的标签。定性的说，这些短程CRFs的基本函数清空了弱分类器的虚假预测，这都是在局部手工设计的特征上的。

Compared to these weaker classifiers, modern DCNN architectures such as the one we use in this work produce score maps and semantic label predictions which are qualitatively different. As illustrated in Figure 2, the score maps are typically quite smooth and produce homogeneous classification results. In this regime, using short-range CRFs can be detrimental, as our goal should be to recover detailed local structure rather than further smooth it. Using contrast-sensitive potentials (Rother et al., 2004) in conjunction to local-range CRFs can potentially improve localization but still miss thin-structures and typically requires solving an expensive discrete optimization problem.

与这些更弱的分类器相比，现代DCNN架构，比如我们在本文中所用的，生成的分数图和语义标签预测，是非常不一样的。如图2所示，分数图一般非常平滑，会产生同质的分类结果。在这个领域中，使用短程CRFs是不利的，因为我们的目标应当是复原局部细节，而不是进一步平滑。使用对比度敏感的potentials与局部距离的CRF可能可以改进定位，但仍然会使细的结构确实，这一般需要求解复杂的离散优化问题。

Figure 3: Model Illustration. The coarse score map from Deep Convolutional Neural Network (with fully convolutional layers) is upsampled by bi-linear interpolation. A fully connected CRF is applied to refine the segmentation result. Best viewed in color.

To overcome these limitations of short-range CRFs, we integrate into our system the fully connected CRF model of Krähenbühl & Koltun (2011). The model employs the energy function: 为克服短程CRF的局限，我们将我们的系统与全连接CRFs模型相结合。模型采用了下面的能量函数：

$$E(x) = \sum_i θ_i (x_i) + \sum_{ij} θ_{ij} (x_i, x_j)$$(1)

where x is the label assignment for pixels. We use as unary potential $θ_i (x_i) = − log P(x_i)$, where $P(x_i)$ is the label assignment probability at pixel i as computed by DCNN. The pairwise potential is $θ_{ij} (x_i, x_j) = μ(x_i, x_j) \sum_{m=1}^K w_m · k^m (f_i, f_j)$, where $μ(x_i, x_j) = 1$ if $x_i != x_j$, and zero otherwise (i.e., Potts Model). There is one pairwise term for each pair of pixels i and j in the image no matter how far from each other they lie, i.e. the model’s factor graph is fully connected. Each $k^m$ is the Gaussian kernel depends on features (denoted as f) extracted for pixel i and j and is weighted by parameter $w_m$. We adopt bilateral position and color terms, specifically, the kernels are:

其中x是像素的指定标签。我们使用的一元potential为$θ_i (x_i) = − log P(x_i)$，其中$P(x_i)$是在像素i处DCNN计算得到的指定标签概率。成对的potential为$θ_{ij} (x_i, x_j) = μ(x_i, x_j) \sum_{m=1}^K w_m · k^m (f_i, f_j)$，其中$μ(x_i, x_j) = 1$，如果$x_i != x_j$，否则为0。对于图像中的每个像素i,j对，都有一个成对项，不管这两个像素点距离多远，即，模型的因子图是全连接的。每个$k^m$是高斯核，依赖像素点i和j处提取到的特征（表示为f），权重为$w_m$。我们采用双边位置和色彩项，核心为：

$$w_1 exp(-\frac{||p_i-p_j||^2}{2σ_α^2}-\frac{||I_i-I_j||^2}{2σ_β^2}) + w_2 exp(-\frac{||p_i-p_j||^2}{2σ_γ^2})$$(2)

where the first kernel depends on both pixel positions (denoted as p) and pixel color intensities (denoted as I), and the second kernel only depends on pixel positions. The hyper parameters $σ_α, σ_β$ and $σ_γ$ control the “scale” of the Gaussian kernels. 其中第一个核心依赖像素位置（表示为p）和像素色彩强度（表示为I），第二个核心只依赖像素位置。超参数$σ_α, σ_β$和$σ_γ$控制着高斯核的尺度。

Crucially, this model is amenable to efficient approximate probabilistic inference (Krähenbühl & Koltun, 2011). The message passing updates under a fully decomposable mean field approximation $b(x) = \prod_i b_i (x_i)$ can be expressed as convolutions with a Gaussian kernel in feature space. High-dimensional filtering algorithms (Adams et al., 2010) significantly speed-up this computation resulting in an algorithm that is very fast in practice, less that 0.5 sec on average for Pascal VOC images using the publicly available implementation of (Krähenbühl & Koltun, 2011).

最关键的是，这个模型可以高效的近似概率推理。传递的更新信息可以表示为与一个高斯核在特征空间的卷积。高维滤波算法使这种计算显著加速，得到的算法在实践中也非常快，对于PASCAL VOC图像平均不到0.5s。

### 4.3 Multi-Scale Prediction

Following the promising recent results of (Hariharan et al., 2014a; Long et al., 2014) we have also explored a multi-scale prediction method to increase the boundary localization accuracy. Specifically, we attach to the input image and the output of each of the first four max pooling layers a two-layer MLP (first layer: 128 3x3 convolutional filters, second layer: 128 1x1 convolutional filters) whose feature map is concatenated to the main network’s last layer feature map. The aggregate feature map fed into the softmax layer is thus enhanced by 5 * 128 = 640 channels. We only adjust the newly added weights, keeping the other network parameters to the values learned by the method of Section 3. As discussed in the experimental section, introducing these extra direct connections from fine-resolution layers improves localization performance, yet the effect is not as dramatic as the one obtained with the fully-connected CRF.

我们还探索了一种多尺度预测方法，来增加边缘定位的准确率。具体的，我们在输入图像和四个max-pooling层的第一层的输出都接上了一个2层MLP（第一层，128个3×3卷积滤波器，第二层，128个1×1卷积滤波器），其特征图拼接到主网络的最后一层特征图。累积的特征图送入softmax层，增强到了5*128=640通道。我们只调整新增加的权重，保持其他网络参数为第3节中方法学习到的值。在试验部分中我们讨论了，从精细分辨率层引入这些多余的直接连接，可以改进定位性能，但效果没有引入全连接CRF的增幅那么好。

## 5 Experimental Evaluation

**Dataset**. We test our DeepLab model on the PASCAL VOC 2012 segmentation benchmark (Everingham et al., 2014), consisting of 20 foreground object classes and one background class. The original dataset contains 1, 464, 1, 449, and 1, 456 images for training, validation, and testing, respectively. The dataset is augmented by the extra annotations provided by Hariharan et al. (2011), resulting in 10, 582 training images. The performance is measured in terms of pixel intersection-over-union (IOU) averaged across the 21 classes.

**数据集**。我们在PASCAL VOC 2012分割基准测试上测试我们的DeepLab模型，数据集包括20个前景目标类别和一个背景类别。原始数据集的训练、验证和测试图像数量分别为1464，1449和1456。数据集由额外的标注进行了扩充，得到了10582幅训练图像。性能度量标准是21类IOU的平均。

**Training**. We adopt the simplest form of piecewise training, decoupling the DCNN and CRF training stages, assuming the unary terms provided by the DCNN are fixed during CRF training. 

For DCNN training we employ the VGG-16 network which has been pre-trained on ImageNet. We fine-tuned the VGG-16 network on the VOC 21-way pixel-classification task by stochastic gradient descent on the cross-entropy loss function, as described in Section 3.1. We use a mini-batch of 20 images and initial learning rate of 0.001 (0.01 for the final classifier layer), multiplying the learning rate by 0.1 at every 2000 iterations. We use momentum of 0.9 and a weight decay of 0.0005.

**训练**。我们采用了最简单形式的分片训练，将DCNN与CRF的训练阶段分开，假设在CRF训练时，DCNN提供的一元项是固定的。

对于DCNN训练，我们采用ImageNet预训练的VGG-16网络，我们在VOC 21路像素分类任务中进行精调，采用SGD，交叉熵损失函数，如3.1节所描述。我们使用一个mini-batch 20幅图像，初始学习速率为0.001（对最终的分类层为0.01），每2000次迭代将学习速率乘以0.1。我们使用的动量为0.9，权重衰减为0.0005。

After the DCNN has been fine-tuned, we cross-validate the parameters of the fully connected CRF model in Eq. (2) along the lines of Krähenbühl & Koltun (2011). We use the default values of $w_2 = 3$ and $σ_γ = 3$ and we search for the best values of $w_1, σ_α$, and $σ_β$ by cross-validation on a small subset of the validation set (we use 100 images). We employ coarse-to-fine search scheme. Specifically, the initial search range of the parameters are $w_1$ ∈ [5, 10], $σ_α$ ∈ [50 : 10 : 100] and $σ_β$ ∈ [3 : 1 : 10] (MATLAB notation), and then we refine the search step sizes around the first round’s best values. We fix the number of mean field iterations to 10 for all reported experiments.

在精调了DCNN后，我们交叉验证式(2)中的全连接CRF参数。我们使用默认值$w_2 = 3$ and $σ_γ = 3$，并搜索$w_1, σ_α$, and $σ_β$的最佳值，在验证集的小型子集上进行交叉验证（我们使用100幅图像）。我们采用由粗而细的搜索方案。具体的，参数的初始搜索范围为$w_1$ ∈ [5, 10], $σ_α$ ∈ [50 : 10 : 100] and $σ_β$ ∈ [3 : 1 : 10]，然后在第一轮的最佳值基础上，细化搜索步长。我们在所有试验中，固定mean field迭代次数为10。

**Evaluation on Validation set**. We conduct the majority of our evaluations on the PASCAL ‘val’ set, training our model on the augmented PASCAL ‘train’ set. As shown in Tab. 1 (a), incorporating the fully connected CRF to our model (denoted by DeepLab-CRF) yields a substantial performance boost, about 4% improvement over DeepLab. We note that the work of Krähenbühl & Koltun (2011) improved the 27.6% result of TextonBoost (Shotton et al., 2009) to 29.1%, which makes the improvement we report here (from 59.8% to 63.7%) all the more impressive.

**在验证集上的评估**。我们主要在PASCAL val集上进行评估，在PASCAL train集的扩充集上进行模型训练。如表1(a)所示，模型使用了全连接CRF（表示为DeepLab-CRF）得到了很大的性能提升，比DeepLab提升了4%。我们注意一篇文献将另一篇文献的结果从27.6%改进到了29.1%，这说明了我们的改进（从59.8%到63.7%）更加令人印象深刻。

Turning to qualitative results, we provide visual comparisons between DeepLab and DeepLab-CRF in Fig. 7. Employing a fully connected CRF significantly improves the results, allowing the model to accurately capture intricate object boundaries.

我们在图7给出了DeepLab和DeepLab-CRF的视觉比较。采用全连接CRF显著改进了结果，使模型可以准确的捕捉的目标的精细边缘。

Figure 7: Visualization results on VOC 2012-val. For each row, we show the input image, the segmentation result delivered by the DCNN (DeepLab), and the refined segmentation result of the Fully Connected CRF (DeepLab-CRF). We show our failure modes in the last three rows. Best viewed in color.

**Multi-Scale features**. We also exploit the features from the intermediate layers, similar to Hariharan et al. (2014a); Long et al. (2014). As shown in Tab. 1 (a), adding the multi-scale features to our DeepLab model (denoted as DeepLab-MSc) improves about 1.5% performance, and further incorporating the fully connected CRF (denoted as DeepLab-MSc-CRF) yields about 4% improvement. The qualitative comparisons between DeepLab and DeepLab-MSc are shown in Fig. 4. Leveraging the multi-scale features can slightly refine the object boundaries.

**多尺度特征**。我们还利用了中间层的特征，与其他文献类似。如表1(a)所示，为我们的DeepLab模型增加了多尺度特征（表示为DeepLab-MSc）改进了1.5%的性能，进一步结合全连接CRF（表示为DeepLab-MSc-CRF）得到了4%的改进。DeepLab和DeepLab-MSc的定性比较如图4所示。利用多尺度特征可以略微改进目标边界。

Figure 4: Incorporating multi-scale features improves the boundary segmentation. We show the results obtained by DeepLab and DeepLab-MSc in the first and second row, respectively. Best viewed in color.

**Field of View**. The ‘atrous algorithm’ we employed allows us to arbitrarily control the Field-of-View (FOV) of the models by adjusting the input stride, as illustrated in Fig. 1. In Tab. 2, we experiment with several kernel sizes and input strides at the first fully connected layer. The method, DeepLab-CRF-7x7, is the direct modification from VGG-16 net, where the kernel size = 7×7 and input stride = 4. This model yields performance of 67.64% on the ‘val’ set, but it is relatively slow (1.44 images per second during training). We have improved model speed to 2.9 images per second by reducing the kernel size to 4 × 4. We have experimented with two such network variants with different FOV sizes, DeepLab-CRF and DeepLab-CRF-4x4; the latter has large FOV (i.e., large input stride) and attains better performance. Finally, we employ kernel size 3×3 and input stride = 12, and further change the filter sizes from 4096 to 1024 for the last two layers. Interestingly, the resulting model, DeepLab-CRF-LargeFOV, matches the performance of the expensive DeepLab-CRF-7x7. At the same time, it is 3.36 times faster to run and has significantly fewer parameters (20.5M instead of 134.3M).

**视野**。我们采用的孔洞算法使我们可以任意控制模型的视野，只需要调整输入步长，如图1所示。在表2中，我们在第一个全连接层上试验了几种滤波核大小和输入步长。模型DeepLab-CRF-7×7是在VGG-16网络上的直接修改，其中滤波核大小为7×7，输入步长为4。这个模型在val集上得到的性能为67.64%，但略慢一些（训练时每秒1.44幅图像）。核大小缩小为4×4，可以将速度提升到2.9图像每秒。我们用两个这种网络变体，使用不同的FOV大小进行了试验，即DeepLab-CRF和DeepLab-CRF-4×4；后者FOV大（即，输入步长大），得到了更好的性能。最后，我们采用的核大小为3×3，输入步长为12，在最后两层进一步将滤波器数量从4096降低到1024。有趣的是，得到的模型，DeepLab-CRF-LargeFOV，与耗时的DeepLab-CRF-7×7性能类似。在同样的时间内，速度快了3.36倍，参数数量也少的多（20.5M vs 134.3M）。

The performance of several model variants is summarized in Tab. 1, showing the benefit of exploiting multi-scale features and large FOV. 表1总结了几种模型变体的性能，表明了利用多尺度特征和大的FOV的好处。

Table 1: (a) Performance of our proposed models on the PASCAL VOC 2012 ‘val’ set (with training in the augmented ‘train’ set). The best performance is achieved by exploiting both multi-scale features and large field-of-view.

Method | Mean IOU(%)
--- | ---
DeepLab | 59.80
DeepLab-CRF | 63.74
DeepLab-MSc | 61.30
DeepLab-MSc-CRF | 65.21
DeepLab-7x7 | 64.38
DeepLab-CRF-7x7 | 67.64
DeepLab-LargeFOV | 62.25
DeepLab-CRF-LargeFOV | 67.64
DeepLab-MSc-LargeFOV | 64.21
DeepLab-MSc-CRF-LargeFOV | 68.70

Table 1: (b) Performance of our proposed models (with training in the augmented ‘trainval’ set) compared to other state-of-art methods on the PASCAL VOC 2012 ‘test’ set.

Method | Mean IOU(%)
--- | ---
MSRA-CFM | 61.8
FCN-8s | 62.2
TTI-Zoomout-16 | 64.4
DeepLab-CRF | 66.4
DeepLab-MSc-CRF | 67.1
DeepLab-CRF-7x7 | 70.3
DeepLab-CRF-LargeFOV | 70.3
DeepLab-MSc-CRF-LargeFOV | 71.6

Table 2: Effect of Field-Of-View. We show the performance (after CRF) and training speed on the PASCAL VOC 2012 ‘val’ set as the function of (1) the kernel size of first fully connected layer, (2) the input stride value employed in the atrous algorithm.

Method | Kernel Size | Input Stride | Receptive Field | Parameters | Mean IOU(%) | Training Speed(img/sec)
--- | --- | --- | --- | --- | --- | ---
DeepLab-CRF-7x7 | 7×7 | 4 | 224 | 134.3M | 67.64 | 1.44
DeepLab-CRF | 4x4 | 4 | 128 | 65.1M | 63.74 | 2.90
DeepLab-CRF-4x4 | 4x4 | 8 | 224 | 65.1M | 67.14 | 2.90
DeepLab-CRF-LargeFOV | 3x3 | 12 | 224 | 20.5M | 67.64 | 4.84

**Mean Pixel IOU along Object Boundaries**. To quantify the accuracy of the proposed model near object boundaries, we evaluate the segmentation accuracy with an experiment similar to Kohli et al. (2009); Krähenbühl & Koltun (2011). Specifically, we use the ‘void’ label annotated in val set, which usually occurs around object boundaries. We compute the mean IOU for those pixels that are located within a narrow band (called trimap) of ‘void’ labels. As shown in Fig. 5, exploiting the multi-scale features from the intermediate layers and refining the segmentation results by a fully connected CRF significantly improve the results around object boundaries.

**目标边缘处的平均像素IOU**。为对提出的模型在目标边缘附近的准确率进行量化，我们用与其他文献类似的方法评估分割准确率。具体的，我们使用val集中标注的void标签，经常在目标边缘附近发生。我们计算这些在void标签附近很窄的一条的平均IOU。如图5所示，利用中间层的多尺度特征，通过一个全连接CRF精炼分割结果，显著改进了目标边缘附近的结果。

Figure 5: (a) Some trimap examples (top-left: image. top-right: ground-truth. bottom-left: trimap of 2 pixels. bottom-right: trimap of 10 pixels). Quality of segmentation result within a band around the object boundaries for the proposed methods. (b) Pixelwise accuracy. (c) Pixel mean IOU.

**Comparison with State-of-art**. In Fig. 6, we qualitatively compare our proposed model, DeepLab-CRF, with two state-of-art models: FCN-8s (Long et al., 2014) and TTI-Zoomout-16 (Mostajabi et al., 2014) on the ‘val’ set (the results are extracted from their papers). Our model is able to capture the intricate object boundaries.

**与目前最好结果的比较**。在图6中，我们定型的比较了我们的模型与两个目前最好的模型，即FCN-8s和TTI-Zoomout-16。我们的模型DeepLab-CRF，可以更好的捕获目标边缘细节。

Figure 6: Comparisons with state-of-the-art models on the val set. First row: images. Second row: ground truths. Third row: other recent models (Left: FCN-8s, Right: TTI-Zoomout-16). Fourth row: our DeepLab-CRF. Best viewed in color.

**Reproducibility**. We have implemented the proposed methods by extending the excellent Caffe framework (Jia et al., 2014). We share our source code, configuration files, and trained models that allow reproducing the results in this paper at a companion web site https://bitbucket.org/deeplab/deeplab-public.

**可复现性**。我们在Caffe框架的基础上实现了我们的方法。我们开源了代码、配置文件和训练的模型。

**Test set results**. Having set our model choices on the validation set, we evaluate our model variants on the PASCAL VOC 2012 official ‘test’ set. As shown in Tab. 3, our DeepLab-CRF and DeepLab-MSc-CRF models achieve performance of 66.4% and 67.1% mean IOU, respectively. Our models outperform all the other state-of-the-art models (specifically, TTI-Zoomout-16 (Mostajabi et al., 2014), FCN-8s (Long et al., 2014), and MSRA-CFM (Dai et al., 2014)). When we increase the FOV of the models, DeepLab-CRF-LargeFOV yields performance of 70.3%, the same as DeepLab-CRF-7x7, while its training speed is faster. Furthermore, our best model, DeepLab-MSc-CRF-LargeFOV, attains the best performance of 71.6% by employing both multi-scale features and large FOV.

**测试集结果**。在验证集上设置好了模型参数后，我们在PASCAL VOC 2012官方测试集上评估我们的模型变体。如表3所示，我们的DeepLab-CRF和DeepLab-MSc-CRF模型分别得到了66.4%和67.1%的平均IOU。我们的模型超过了所有其他目前最好的模型（具体的，TTI-Zoomout-16，FCN-8s，和MSRA-CFM）。当我们增加模型FOV时，DeepLab-CRF-LargeFOV得到了70.3%的结果，与DeepLab-CRF-7×7一样，而其速度更快。而且，我们最好的模型，DeepLab-MSc-CRF-LargeFOV，得到了71.6%的最好性能，采用了多尺度特征和大的FOV。

Table 3: Labeling IOU (%) on the PASCAL VOC 2012 test set, using the trainval set for training.

Method | Mean IOU(%)
--- | ---
MSRA-CFM | 61.8
FCN-8s | 62.2
TTI-Zoomout-16 | 64.4
DeepLab-CRF | 66.4
DeepLab-MSc-CRF | 67.1
DeepLab-CRF-7x7 | 70.3
DeepLab-CRF-LargeFOV | 70.3
DeepLab-MSc-CRF-LargeFOV | 71.6

## 6 Discussion

Our work combines ideas from deep convolutional neural networks and fully-connected conditional random fields, yielding a novel method able to produce semantically accurate predictions and detailed segmentation maps, while being computationally efficient. Our experimental results show that the proposed method significantly advances the state-of-art in the challenging PASCAL VOC 2012 semantic image segmentation task.

我们的工作结合了DCNN与全连接CRF的思想，得到了新模型，可以得到精确的语义预测和细节的分割图，计算上效率非常高。我们的试验结果表明，我们提出的方法在PASCAL VOC 2012图像分割任务上明显将目前最好结果提升了很多。

There are multiple aspects in our model that we intend to refine, such as fully integrating its two main components (CNN and CRF) and train the whole system in an end-to-end fashion, similar to Krähenbühl & Koltun (2013); Chen et al. (2014); Zheng et al. (2015). We also plan to experiment with more datasets and apply our method to other sources of data such as depth maps or videos. Recently, we have pursued model training with weakly supervised annotations, in the form of bounding boxes or image-level labels (Papandreou et al., 2015).

我们的模型还可以在多个方面改进，如全面结合其两个主要部分（CNN和CRF），以端到端的形式训练整个系统。我们还计划在更多数据集上进行试验，在其他类型数据上也进行试验，如深度图或视频。最近，我们在进行弱监督标注方面的训练试验，以边界框的或图像标签的形式。

At a higher level, our work lies in the intersection of convolutional neural networks and probabilistic graphical models. We plan to further investigate the interplay of these two powerful classes of methods and explore their synergistic potential for solving challenging computer vision tasks.

在更高层，我们的工作是在CNN和概率图模型的交际上。我们计划进一步研究这两类方法，探索其解决计算机视觉任务的潜力。