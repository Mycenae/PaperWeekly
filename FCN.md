# Fully Convolutional Networks for Semantic Segmentation

Jonathan Long et al. UC Berkeley

## Abstract 摘要

Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build “fully convolutional” networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet [20], the VGG net [31], and GoogLeNet [32]) into fully convolutional networks and transfer their learned representations by fine-tuning [3] to the segmentation task. We then define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves state-of-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes less than one fifth of a second for a typical image.

卷积网络是非常强力的视觉模型，可以产生层次化的特征。我们展示，卷积网络本身，进行端到端，像素到像素的训练，会超过目前最好的语义分割结果。一个关键的思想是构建全卷积网络，输入图像大小任意，经过高效的推理和学习，生成对应大小的输出。我们定义并详述全卷积网络的空间，解释其在空间密集预测任务中的应用，并与之前的模型进行联系。我们将现在的分类网络(AlexNet [20], the VGG net [31], and GoogLeNet [32])改装成全卷积网络，并精调[3]以将其学到的表示迁移到分割任务中。我们然后定义一种跳跃框架，将从深度、粗糙的层中得到的语义信息，与从浅层、精细的层中得到的表现信息结合起来，生成准确细节丰富的分割。我们的全卷积网络在PASCAL VOC上得到了目前最好的分割结果（与2012年的62.2%平均IU相比，相对改进了20%），在NYUDv2和SIFT Flow上也是，而对于一幅典型的图像，其推理时间不到1/5秒。

## 1. Introduction 引言

Convolutional networks are driving advances in recognition. Convnets are not only improving for whole-image classification [20, 31, 32], but also making progress on local tasks with structured output. These include advances in bounding box object detection [29, 10, 17], part and keypoint prediction [39, 24], and local correspondence [24, 8].

卷积网络正在取得可以识别的进展。卷积网络不仅在改进整图分类的效果[20,31,32]，也在结构化输出的局部任务中取得进展。这包括带有边界框的目标检测[29,10,17]，部位和关键点检测[39,24]，和局部对应性[24,8]。

The natural next step in the progression from coarse to fine inference is to make a prediction at every pixel. Prior approaches have used convnets for semantic segmentation [27, 2, 7, 28, 15, 13, 9], in which each pixel is labeled with the class of its enclosing object or region, but with short-comings that this work addresses.

很自然的，从粗糙到精细推理的下一步的进展是，对每个像素进行预测。之前的方法已经有使用卷积网络进行语义分割[27,2,7,28,15,13,9]，其中每个像素都标记为其包围的对象或区域的类别，但是这个要处理的问题总是有很多缺点。

We show that a fully convolutional network (FCN) trained end-to-end, pixels-to-pixels on semantic segmentation exceeds the state-of-the-art without further machinery. To our knowledge, this is the first work to train FCNs end-to-end (1) for pixelwise prediction and (2) from supervised pre-training. Fully convolutional versions of existing networks predict dense outputs from arbitrary-sized inputs. Both learning and inference are performed whole-image-at-a-time by dense feedforward computation and backpropagation. In-network upsampling layers enable pixelwise prediction and learning in nets with subsampled pooling.

我们展示了，端到端、像素到像素训练的全卷积网络(FCN)进行语义分割，超过了目前最好的结果。据我们所知，这是第一项工作对FCNs进行端到端的训练，(1)以进行逐像素的预测，(2)从有监督的预训练中进行的。现有网络的全卷积版本从任意大小的输入中预测密集输出。学习过程和推理过程都是整幅图像一次性的处理，包括密集前向计算和反向传播。网络中的上采样层使逐像素的预测成为可能，以及在有下采样的网络中进行学习。

This method is efficient, both asymptotically and absolutely, and precludes the need for the complications in other works. Patchwise training is common [27, 2, 7, 28, 9], but lacks the efficiency of fully convolutional training. Our approach does not make use of pre- and post-processing complications, including superpixels [7, 15], proposals [15, 13], or post-hoc refinement by random fields or local classifiers [7, 15]. Our model transfers recent success in classification [20, 31, 32] to dense prediction by reinterpreting classification nets as fully convolutional and fine-tuning from their learned representations. In contrast, previous works have applied small convnets without supervised pre-training [7, 28, 27].

这种方法非常高效，其他更复杂的工作都没有必要了。逐块的训练很常用[27,2,7,28,9]，但缺少全卷积训练的效率。我们的方法没有使用那些复杂的预处理和后处理，包括超像素[7,15]，候选[15,13]，或通过随机场或局部分类器[7,15]进行事后精炼。我们的模型将最近在分类中成功的应用[20,31,32]迁移到密集预测中，方法是将分类网络重新解释为全卷积网络，并从其学习到的表示的基础上进行精调。形成对比的是，之前的工作使用了小型卷积网络，而且没有有监督的预训练[7,28,27]。

Semantic segmentation faces an inherent tension between semantics and location: global information resolves what while local information resolves where. Deep feature hierarchies encode location and semantics in a nonlinear local-to-global pyramid. We define a skip architecture to take advantage of this feature spectrum that combines deep, coarse, semantic information and shallow, fine, appearance information in Section 4.2 (see Figure 3).

语义分割面临着语义和位置的内在矛盾：全局信息解决的是什么的问题，局部信息解决的是哪里的问题。深度特征层次将位置和语义信息编码在一个非线性的局部到全局的金字塔中。我们在4.2节中定义了一种跳跃框架，以利用这种特征谱，将深度、粗糙的语义信息，和浅层、精细的表现信息结合起来（见图3）。

In the next section, we review related work on deep classification nets, FCNs, and recent approaches to semantic segmentation using convnets. The following sections explain FCN design and dense prediction tradeoffs, introduce our architecture with in-network upsampling and multilayer combinations, and describe our experimental framework. Finally, we demonstrate state-of-the-art results on PASCAL VOC 2011-2, NYUDv2, and SIFT Flow.

下一节中，我们回顾了深度分类网络，FCNs相关的工作，和最近使用卷积网络进行语义分割的方法。后续的小节解释了FCN的设计和密集预测的折中，提出了我们带有网络内上采样和多层结合的架构，描述了我们的试验框架。最后，我们在PASCAL VOC 2011/2012、NYUDv2和SIFT Flow数据集上给出了目前最好的结果。

Figure 1. Fully convolutional networks can efficiently learn to make dense predictions for per-pixel tasks like semantic segmentation.

## 2. Related work 相关的工作

Our approach draws on recent successes of deep nets for image classification [20, 31, 32] and transfer learning [3, 38]. Transfer was first demonstrated on various visual recognition tasks [3, 38], then on detection, and on both instance and semantic segmentation in hybrid proposal-classifier models [10, 15, 13]. We now re-architect and finetune classification nets to direct, dense prediction of semantic segmentation. We chart the space of FCNs and situate prior models, both historical and recent, in this framework.

我们的方法利用了最近深度网络在图像分类[20,31,32]和迁移学习[3,38]中的进展。迁移学习首先在各种视觉识别任务中[3,38]展现出来，然后在检测中，然后在混合候选-分类模型的实例分割和语义分割模型中[10,15,13]。我们现在改变分类网络的架构并进行精调，以进行直接的语义分割密集预测。我们在本文中将之前的模型与FCNs进行了比较总结，形成了图表。

**Fully convolutional networks**. To our knowledge, the idea of extending a convnet to arbitrary-sized inputs first appeared in Matan et al. [26], which extended the classic LeNet [21] to recognize strings of digits. Because their net was limited to one-dimensional input strings, Matan et al. used Viterbi decoding to obtain their outputs. Wolf and Platt [37] expand convnet outputs to 2-dimensional maps of detection scores for the four corners of postal address blocks. Both of these historical works do inference and learning fully convolutionally for detection. Ning et al. [27] define a convnet for coarse multiclass segmentation of C. elegans tissues with fully convolutional inference.

**全卷积网络**。据我们所知，将卷积网络的输入拓展为任意大小的第一个工作出现在Matan等[26]中，将经典的LeNet[21]拓展到识别数字串中。因为他们的网络限制一维输入字符串，Matan等使用了Viterbi解码以得到其输出。Wolf和Platt[37]将卷积网络的输出拓展到检测分数的二维图，以处理邮政地址块的四角。这些历史性的工作都学习全卷积网络并进行推理以检测。Ning等[27]定义了一个卷积网络，使用全卷积网络的推理，进行C. elegans牌纸巾的粗糙多类分割。

Fully convolutional computation has also been exploited in the present era of many-layered nets. Sliding window detection by Sermanet et al. [29], semantic segmentation by Pinheiro and Collobert [28], and image restoration by Eigen et al. [4] do fully convolutional inference. Fully convolutional training is rare, but used effectively by Tompson et al. [35] to learn an end-to-end part detector and spatial model for pose estimation, although they do not exposit on or analyze this method.

现在很多层的网络也探索了全卷积计算的潜力。Sermanet等[29]的滑窗检测，Pinheiro和Collobert的语义分割和Eigen等[4]的图像恢复，都进行了全卷积推理。全卷积训练倒是非常少见，但Tompson等[35]将之用于学习一个端到端的部位检测器，以及姿态估计的空间模型，但是他们没有探索分析这种方法。

Alternatively, He et al. [17] discard the non-convolutional portion of classification nets to make a feature extractor. They combine proposals and spatial pyramid pooling to yield a localized, fixed-length feature for classification. While fast and effective, this hybrid model cannot be learned end-to-end.

He等抛弃了分类网络的非卷积部分，形成一个特征提取器。他们将候选和空间金字塔池化结合到一起，生成一个局部化的，固定长度的特征，以进行分类。虽然速度很快，计算效率高，但这种混合模型不能进行端到端的学习。

**Dense prediction with convnets**. Several recent works have applied convnets to dense prediction problems, including semantic segmentation by Ning et al. [27], Farabet et al. [7], and Pinheiro and Collobert [28]; boundary prediction for electron microscopy by Ciresan et al. [2] and for natural images by a hybrid convnet/nearest neighbor model by Ganin and Lempitsky [9]; and image restoration and depth estimation by Eigen et al. [4, 5]. Common elements of these approaches include

**使用卷积网络的密集预测**。最近的几项工作将卷积网络用于密集预测问题，包括Ning等[27]，Farabet等[7]，Pinheiro和Collobert [28]的语义分割；Ciresan等[2]对电子显微成像的边界预测，Ganin和Lempitsky[9]的卷积网络/最近邻混合模型处理自然图像；Eigen等[4,5]的图像恢复和深度估计。这些方法的常用元素包括：

- small models restricting capacity and receptive fields; 限制容量和感受野的小型模型；

- patchwise training [27, 2, 7, 28, 9]; 分块训练[27,2,7,28,9]；

- post-processing by superpixel projection, random field regularization, filtering, or local classification [7, 2, 9]; 后处理，包括超像素投影，随机领域正则化，滤波，或局部分类[7,2,9]；

- input shifting and output interlacing for dense output [29, 28, 9]; 输入变换和输出交错的密集输出[29,28,9]；

- multi-scale pyramid processing [7, 28, 9]; 多尺度金字塔处理[7,28,9]；

- saturating tanh nonlinearities [7, 4, 28]; 饱和的tanh非线性[7,4,28]；

- and ensembles [2, 9], 集成学习。

whereas our method does without this machinery. However, we do study patchwise training 3.4 and “shift-and-stitch” dense output 3.2 from the perspective of FCNs. We also discuss in-network upsampling 3.3, of which the fully connected prediction by Eigen et al. [5] is a special case. 然而我们的方法没有使用这些技术。但我们确实在3.4节和3.2节中从FCNs的角度研究了分块训练和shift-and-stitch密集输出。我们还讨论了网络中的上采样（3.3节），Eigen等[5]中的全连接预测是一个特例。

Unlike these existing methods, we adapt and extend deep classification architectures, using image classification as supervised pre-training, and fine-tune fully convolutionally to learn simply and efficiently from whole image inputs and whole image ground truths. 与现有的方法不同，我们将深度分类架构进行了修改和拓展，使用图像分类作为有监督的预训练，从全部输入图像和真值标注中简单有效的精调全卷积网络。

Hariharan et al. [15] and Gupta et al. [13] likewise adapt deep classification nets to semantic segmentation, but do so in hybrid proposal-classifier models. These approaches fine-tune an R-CNN system [10] by sampling bounding boxes and/or region proposals for detection, semantic segmentation, and instance segmentation. Neither method is learned end-to-end. They achieve state-of-the-art segmentation results on PASCAL VOC and NYUDv2 respectively, so we directly compare our standalone, end-to-end FCN to their semantic segmentation results in Section 5.

Hariharan等[15]和Gupta等[13]类似的将深度分类网络改装已进行语义分割，但是在候选-分类的混合模型下进行的。这些方法精调了一个R-CNN系统[10]，即对边界框或候选区域进行采样以进行检测、语义分割和实例分割。这些方法都不是端到端的。他们分别在PASCAL VOC和NYUDv2上取得了目前最好的分割结果，所以我们将我们提出的单独的端到端的FCN与这些模型的语义分割结果在第5节中进行比较。

We fuse features across layers to define a nonlinear local-to-global representation that we tune end-to-end. In contemporary work Hariharan et al. [16] also use multiple layers in their hybrid model for semantic segmentation. 我们跨层融合了特征，以定义一个非线性的局部到全局的表示，对之进行端到端的调节。同时Hariharan等[16]的工作在他们的混合模型中也使用了多层进行语义分割。

## 3. Fully convolutional networks 全卷积网络

Each layer of data in a convnet is a three-dimensional array of size h × w × d, where h and w are spatial dimensions, and d is the feature or channel dimension. The first layer is the image, with pixel size h × w, and d color channels. Locations in higher layers correspond to the locations in the image they are path-connected to, which are called their receptive fields.

卷积网络中的每层数据都是三维阵列，大小为h × w × d，其中h和w为空间尺度，d是特征或通道维数。第一层是输入图像，像素大小为h × w，d为色彩通道。更高层的位置对应着图像中按路径连接的位置，称之为感受野。

Convnets are built on translation invariance. Their basic components (convolution, pooling, and activation functions) operate on local input regions, and depend only on relative spatial coordinates. Writing $x_{ij}$ for the data vector at location (i, j) in a particular layer, and $y_{ij}$ for the following layer, these functions compute outputs $y_{ij}$ by

卷积网络构建的时候就具有平移不变性。其基本部件（卷积，池化和激活函数）在局部输入区域中计算，只依赖于相对的空间坐标。令$x_{ij}$为特定层上(i,j)位置上的数据向量，下一层的则为$y_{ij}$，计算得到的$y_{ij}$的函数为：

$$y_{ij} = f_{ks} (\{x_{si+δi,sj+δj} \}_{0≤δi,δj≤k} )$$

where k is called the kernel size, s is the stride or subsampling factor, and $f_{ks}$ determines the layer type: a matrix multiplication for convolution or average pooling, a spatial max for max pooling, or an elementwise nonlinearity for an activation function, and so on for other types of layers. 其中k称为核大小，s为步长或下采样因子，$f_{ks}$为层的类型：矩阵相乘就是卷积层，或平均池化层，空域max为max池化层，如果是逐元素的非线性函数就是激活函数层，还有其他函数代表的其他层。

This functional form is maintained under composition, with kernel size and stride obeying the transformation rule 泛函形式如下，其核大小与步长遵循下式

$$f_{ks} ◦ g_{k's'} = (f ◦ g) k'+(k−1)s', ss'$$

While a general deep net computes a general nonlinear function, a net with only layers of this form computes a nonlinear filter, which we call a deep filter or fully convolutional network. An FCN naturally operates on an input of any size, and produces an output of corresponding (possibly resampled) spatial dimensions. 一般的深度网络计算的是一般的非线性函数，而只有这些层的网络计算的是一个非线性滤波器，我们称之为深度滤波器，或全卷积网络。FCN的输入可以是任意大小的，生成的输出也是对应大小的空间分辨率（也可能是重采样的）。

A real-valued loss function composed with an FCN defines a task. If the loss function is a sum P over the spatial dimensions of the final layer, $l(x; θ) = \sum_{ij} l'(x_{ij}; θ)$, its gradient will be a sum over the gradients of each of its spatial components. Thus stochastic gradient descent on l computed on whole images will be the same as stochastic gradient descent on l' , taking all of the final layer receptive fields as a minibatch. 实值损失函数构成的FCN定义了一个任务。如果损失函数为最后一层在空域上的求和，$l(x; θ) = \sum_{ij} l'(x_{ij}; θ)$，其梯度就是其每个空域部件的梯度的和。所以，l在整个图像上计算的随机梯度下降，和l'将最终层的所有感受野作为一个minibatch计算随机梯度下降是一样的。

When these receptive fields overlap significantly, both feedforward computation and backpropagation are much more efficient when computed layer-by-layer over an entire image instead of independently patch-by-patch. 当这些感受野有明显的重叠时，前向计算和反向传播在整幅图像上逐层计算，比逐块计算要高效的多。

We next explain how to convert classification nets into fully convolutional nets that produce coarse output maps. For pixelwise prediction, we need to connect these coarse outputs back to the pixels. Section 3.2 describes a trick, fast scanning [11], introduced for this purpose. We gain insight into this trick by reinterpreting it as an equivalent network modification. As an efficient, effective alternative, we introduce deconvolution layers for upsampling in Section 3.3. In Section 3.4 we consider training by patchwise sampling, and give evidence in Section 4.3 that our whole image training is faster and equally effective.

下面我们解释怎样将分类网络转化成全卷积网络，以生成粗糙的输出图。对于逐像素的预测，我们需要将这些粗糙的输出与像素连接起来。3.2节描述了一个技巧，快速扫描[11]，就是为这个目的提出的。我们重新将其解释为等价的网络修正，以更深入的进行理解。我们在3.3节提出解卷积层进行上采样，作为一种高效的、有效的替代。在3.4节，我们采用分块采样进行训练，并在4.3节给出证明，我们的整图训练更快，也更有效。

### 3.1. Adapting classifiers for dense prediction 将分类调整为密集预测

Typical recognition nets, including LeNet [21], AlexNet [20], and its deeper successors [31, 32], ostensibly take fixed-sized inputs and produce non-spatial outputs. The fully connected layers of these nets have fixed dimensions and throw away spatial coordinates. However, these fully connected layers can also be viewed as convolutions with kernels that cover their entire input regions. Doing so casts them into fully convolutional networks that take input of any size and output classification maps. This transformation is illustrated in Figure 2.

典型的识别网络，包括LeNet[21]，AlexNet[20]和其更深的后继者[31,32]，表面上以固定大小的图像为输入，生成非空域的输出。这些网络的全连接层维度固定，不包含空间坐标信息。但是，这些全卷积层也可以视为卷积核覆盖了整个输入区域的卷积。这样可以将其转换成全卷积网络，以任意大小图像为输入，输出分类图。这种变换如图2所示。

Furthermore, while the resulting maps are equivalent to the evaluation of the original net on particular input patches, the computation is highly amortized over the overlapping regions of those patches. For example, while AlexNet takes 1.2 ms (on a typical GPU) to infer the classification scores of a 227×227 image, the fully convolutional net takes 22 ms to produce a 10×10 grid of outputs from a 500×500 image, which is more than 5 times faster than the naı̈ve approach(Assuming efficient batching of single image inputs. The classification scores for a single image by itself take 5.4 ms to produce, which is nearly 25 times slower than the fully convolutional version).

而且，得到的图与原网络在特定输入块的结果应该是等价的，计算量分摊在这些块的重合区域中。比如，AlexNet在227×227的图像中推理分类分数耗时1.2ms（在典型的GPU上），而全卷积层用22ms从一个500×500的图像中生成一个10×10的网格输出，这比naı̈ve方法快了5倍多（假设单个图像输入批量化效率很高，单个图像的分类分数要花费5.4ms，这比全卷积版本要慢25倍）。

The spatial output maps of these convolutionalized models make them a natural choice for dense problems like semantic segmentation. With ground truth available at every output cell, both the forward and backward passes are straightforward, and both take advantage of the inherent computational efficiency (and aggressive optimization) of convolution. The corresponding backward times for the AlexNet example are 2.4 ms for a single image and 37 ms for a fully convolutional 10 × 10 output map, resulting in a speedup similar to that of the forward pass.

这些卷积化模型的空域输出图可以很自然的应用于密集问题，如语义分割。只要在每个输出单元都有真值，前向和反向过程都非常简单，前向和反向过程都利用了卷积内在的计算效率（和极大的优化）。AlexNet的例子对应的反向时间为2.4ms一幅图像，而全卷积的10×10输出图为37ms，其加速效果与前向过程的类似。

While our reinterpretation of classification nets as fully convolutional yields output maps for inputs of any size, the output dimensions are typically reduced by subsampling. The classification nets subsample to keep filters small and computational requirements reasonable. This coarsens the output of a fully convolutional version of these nets, reducing it from the size of the input by a factor equal to the pixel stride of the receptive fields of the output units.

我们对分类网络重新解释为全卷积，可以对任意大小的输入得到输出图，输出维度通常通过下采样得到减小。分类网络下采样以确保滤波器规模小，计算量需求合理。这使这些网络的全卷积版本的输出变得粗糙，从输入的大小变为更低的分辨率，变化因子等于输出单元的感受野的大小。

### 3.2. Shift-and-stitch is filter rarefaction 滤波器稀疏化

Dense predictions can be obtained from coarse outputs by stitching together output from shifted versions of the input. If the output is downsampled by a factor of f, shift the input x pixels to the right and y pixels down, once for every (x, y) s.t. 0 ≤ x, y < f. Process each of these $f^2$ inputs, and interlace the outputs so that the predictions correspond to the pixels at the centers of their receptive fields.

从粗糙的输出得到密集预测的方法，可以是，将输入不断平移，得到不同的输出，将这些输出缝合起来。如果输出下采样率为f，将输入向右平移x像素，向下移动y像素，对每个(x,y)对都进行一次，s.t. 0 ≤ x, y < f。处理这$f^2$个输入，将输出交错，使预测对应感受野的中心。

Although performing this transformation naı̈vely increases the cost by a factor of $f^2$ there is a well-known trick for efficiently producing identical results [11, 29] known to the wavelet community as the à trous algorithm [25]. Consider a layer (convolution or pooling) with input stride s, and a subsequent convolution layer with filter weights $f^{ij}$ (eliding the irrelevant feature dimensions). Setting the lower layer’s input stride to 1 upsamples its output by a factor of s. However, convolving the original filter with the upsampled output does not produce the same result as shift-and-stitch, because the original filter only sees a reduced portion of its (now upsampled) input. To reproduce the trick, rarefy the filter by enlarging it as:

直接进行这种变换代价增加了$f^2$倍，但有一个很有名的技巧可以高效的得到一样的效果[11,29]，研究小波的群体知道这种技巧，叫做atrous算法[25]。如果一个层（卷积或池化）输入的步长为s，后续的卷积层的滤波器权重为$f^{ij}$（忽略不相关的特征维度）。设定较低的层的输入步长为1，可以将输出进行上采样，因子为s。但是，将原始的滤波器与上采样的输出进行卷积，这与shift-and-stitch的结果不一样，因为原始滤波器原始滤波器只看到了输入简化过的一部分（现在则是上采样过的）。为重现这种技巧，我们稀释滤波器，将其放大为：

$f_{ij}' = f_{i/s, j/s}$, if s divides both i and j; 0, otherwise

(with i and j zero-based). Reproducing the full net output of the trick involves repeating this filter enlargement layer-by-layer until all subsampling is removed. (In practice, this can be done efficiently by processing subsampled versions of the upsampled input.) （i，j都是从0开始的）。重现这种技巧的全网络输出，需要逐层重复这种滤波器放大技术，直到所有的下采样都没有了。（在实践中，可以将输入进行上采样，然后处理这个输入的下采样版本，这样更高效一些）

Decreasing subsampling within a net is a tradeoff: the filters see finer information, but have smaller receptive fields and take longer to compute. The shift-and-stitch trick is another kind of tradeoff: the output is denser without decreasing the receptive field sizes of the filters, but the filters are prohibited from accessing information at a finer scale than their original design.

降低网络中的下采样是一种折中：滤波器看到了更精细的信息，但是有更小的感受野，需要更长时间进行计算。shift-and-stich技巧是另一种折中：输出更稠密，也没有降低滤波器的感受野大小，但与原始设计相比，滤波器不能获取更精细的信息了。

Although we have done preliminary experiments with this trick, we do not use it in our model. We find learning through upsampling, as described in the next section, to be more effective and efficient, especially when combined with the skip layer fusion described later on.

虽然我们用这种技巧进行了初步的实验，我们并没有在模型中使用。我们发现通过上采样进行学习更高效有效，尤其与跳跃层融合后，后面会描述这个。

### 3.3. Upsampling is backwards strided convolution 上采样是反向变步长的卷积

Another way to connect coarse outputs to dense pixels is interpolation. For instance, simple bilinear interpolation computes each output $y_{ij}$ from the nearest four inputs by a linear map that depends only on the relative positions of the input and output cells. 将粗糙的输出变为密集的像素的另一种方法是插值。比如，简单的双线性插值从最接近的四个输入点上用线性图上计算每个输出$y_{ij}$，这只与输入与输出单元的相对位置有关。

In a sense, upsampling with factor f is convolution with a fractional input stride of 1/f. So long as f is integral, a natural way to upsample is therefore backwards convolution (sometimes called deconvolution) with an output stride of f. Such an operation is trivial to implement, since it simply reverses the forward and backward passes of convolution. Thus upsampling is performed in-network for end-to-end learning by backpropagation from the pixelwise loss.

在某种意义上，因子为f的上采样就是与分数输入步长为1/f的卷积。只要f是整数，上采样的一种自然方法就是反向卷积（也称为解卷积），输出步长为f。这种运算实现起来很简单，因为只是将卷积的前向和反向过程反了过来。所以上采样在网络内通过反向传播进行，损失函数为逐像素的，以得到端到端的学习。

Note that the deconvolution filter in such a layer need not be fixed (e.g., to bilinear upsampling), but can be learned. A stack of deconvolution layers and activation functions can even learn a nonlinear upsampling. 注意，这样一个层中的解卷积滤波器不需要固定（如，双线性上采样），但可以学习。多个解卷积层和激活函数甚至可以学习一个非线性上采样。

In our experiments, we find that in-network upsampling is fast and effective for learning dense prediction. Our best segmentation architecture uses these layers to learn to upsample for refined prediction in Section 4.2. 在我们的试验中，我们发现网络内上采样可以快速有效的学习密集预测。4.2节中，我们最好的分割架构使用这些层来学习进行上采样以精炼预测。

### 3.4. Patchwise training is loss sampling 分块训练是损失采样

In stochastic optimization, gradient computation is driven by the training distribution. Both patchwise training and fully convolutional training can be made to produce any distribution, although their relative computational efficiency depends on overlap and minibatch size. Whole image fully convolutional training is identical to patchwise training where each batch consists of all the receptive fields of the units below the loss for an image (or collection of images). While this is more efficient than uniform sampling of patches, it reduces the number of possible batches. However, random selection of patches within an image may be recovered simply. Restricting the loss to a randomly sampled subset of its spatial terms (or, equivalently applying a DropConnect mask [36] between the output and the loss) excludes patches from the gradient computation.

在随机优化中，梯度计算由训练分布驱动。分块训练和全卷积训练都可以生成任意分布，虽然其相对运算效率依赖于重叠和minibatch大小。整体图像全卷积训练与分块训练是一样的，其中每个batch包括一幅图像（或图像集）单元所有的感受野。虽然这比单一采样图像块效率要高，但还是减少了可能批次的数量。但是，图像中图像块的随机选择可以很简单的恢复。限制损失函数为其空域项的随机采样子集（或等价的将一个DropConnect掩膜[36]应用于输出和损失函数之间）将梯度计算中的块排除在外。

If the kept patches still have significant overlap, fully convolutional computation will still speed up training. If gradients are accumulated over multiple backward passes, batches can include patches from several images.(Note that not every possible patch is included this way, since the receptive fields of the final layer units lie on a fixed, strided grid. However, by shifting the image right and down by a random value up to the stride, random selection from all possible patches may be recovered)

如果保留下来的块仍然有明显的重叠，全卷积计算将仍然会加速训练。如果梯度在多个反向过程中累积，批次中可能包括几个图像中的图像块。（注意，并不是每个可能的块都以这种方式包含进去，因为最后一层上单元的感受野是在固定的、有步长的网格上。但是，将图像右移下移步长内的随机值，可能会从所有可能的图像块中恢复任意的选择）

Sampling in patchwise training can correct class imbalance [27, 7, 2] and mitigate the spatial correlation of dense patches [28, 15]. In fully convolutional training, class balance can also be achieved by weighting the loss, and loss sampling can be used to address spatial correlation. 分块训练中的采样可以修正类别的不均衡[27,7,2]并弥合密集块之间的空域相关性[28,15]。在全卷积训练中，类别均衡也可以通过对损失函数加权得到，损失采样可以用于解决空域相关的问题。

We explore training with sampling in Section 4.3, and do not find that it yields faster or better convergence for dense prediction. Whole image training is effective and efficient. 我们在4.3节中研究了用带有采样的训练，没有在发现密集预测中可以得到更快或更好的收敛性能。整图训练高效又好用。

## 4. Segmentation Architecture 分割架构

We cast ILSVRC classifiers into FCNs and augment them for dense prediction with in-network upsampling and a pixelwise loss. We train for segmentation by fine-tuning. Next, we add skips between layers to fuse coarse, semantic and local, appearance information. This skip architecture is learned end-to-end to refine the semantics and spatial precision of the output.

我们将ILSVRC分类器变换为FCNs，并将其扩充以进行密集预测，有网络中的上采样，和逐像素的损失。我们通过精调以训练进行分割。下一步，我们在层之间增加跳跃以融合粗糙的外貌信息，包括语义信息和局部信息。这种跳跃架构是端到端学习的，以精炼输出的语义和空域精度。

For this investigation, we train and validate on the PASCAL VOC 2011 segmentation challenge [6]. We train with a per-pixel multinomial logistic loss and validate with the standard metric of mean pixel intersection over union, with the mean taken over all classes, including background. The training ignores pixels that are masked out (as ambiguous or difficult) in the ground truth.

为这个研究，我们在PASCAL VOC 2011分割挑战[6]上进行训练和验证。我们用一个逐像素的多项式logistic损失函数进行训练，并用标准的度量标准，即平均像素IoU进行验证，在所有类别上进行平均，包括背景。训练过程忽略了真值中被掩膜的像素（模糊的，或困难的）。

### 4.1. From classifier to dense FCN 从分类器到密集FCN

We begin by convolutionalizing proven classification architectures as in Section 3. We consider the AlexNet (Using the publicly available CaffeNet reference model) architecture [20] that won ILSVRC12, as well as the VGG nets [31] and the GoogLeNet [32](there is no publicly available version of GoogLeNet, we use our own reimplementation. Our version is trained with less extensive data augmentation, and gets 68.5% top-1 and 88.4% top-5 ILSVRC accuracy) which did exceptionally well in ILSVRC14. We pick the VGG 16-layer net (Using the publicly available version from the Caffe model zoo), which we found to be equivalent to the 19-layer net on this task. For GoogLeNet, we use only the final loss layer, and improve performance by discarding the final average pooling layer. We decapitate each net by discarding the final classifier layer, and convert all fully connected layers to convolutions. We append a 1 × 1 convolution with channel dimension 21 to predict scores for each of the PASCAL classes (including background) at each of the coarse output locations, followed by a deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs as described in Section 3.3. Table 1 compares the preliminary validation results along with the basic characteristics of each net. We report the best results achieved after convergence at a fixed learning rate (at least 175 epochs).

我们从分类网络卷积化开始。我们考虑赢得了ILSVRC12的AlexNet架构[20]（使用公开可用的CaffeNet参考模型），以及在ILSVRC14上表现非常好的VGGNet[31]和GoogLeNet[32]（没有公开可用的GoogLeNet版本，我们进行了重新实现。我们的版本训练时没有用那么多数据扩充，得到的top-1和top-5 ILSVRC准确率为68.5%和88.4%）。我们选择了VGG-16网络（使用Caffe模型库中公开可用的版本），在这个任务中与VGG-19一样的。对于GoogLeNet，我们只使用最终的损失层，并丢弃了最后的平均池化层，以改进效果。我们将每个网络最后的分类层丢弃，并将全连接层转换成卷积层。我们在每个粗糙输出位置上接上了一个1 × 1卷积层，21个通道，以预测每个PASCAL类别（包括背景）的分数，然后跟上一个解卷积层来将粗糙的输出层双线性上采样，成为像素这样密集的输出，如3.3节所示。表1比较了每个网络的初步验证结果和基本性质。我们给出了以固定学习率收敛后的最好结果（至少175轮）。

Fine-tuning from classification to segmentation gave reasonable predictions for each net. Even the worst model achieved ∼ 75% of state-of-the-art performance. The segmentation-equipped VGG net (FCN-VGG16) already appears to be state-of-the-art at 56.0 mean IU on val, compared to 52.6 on test [15]. Training on extra data raises FCN-VGG16 to 59.4 mean IU and FCN-AlexNet to 48.0 mean IU on a subset of val. Despite similar classification accuracy, our implementation of GoogLeNet did not match the VGG16 segmentation result.

从分类网络精调成分割网络，给出了每个网络的合理预测。即使最差的模型也取得了目前最好结果的大约75%的性能。改成了分割的VGG网络(FCN-VGG16)已经似乎是目前最好的模型了，在验证时得到了52.6的MIoU，在测试时得到了52.6[15]。用额外的数据进行训练，将FCN-VGG16的性能提高至59.4MIoU，FCN-AlexNet在验证集的子集上得到了48.0的MIoU。尽管分类准确率类似，但我们实现的GoogLeNet没有达到VGG16的分割结果。

Table 1. We adapt and extend three classification convnets. We compare performance by mean intersection over union on the validation set of PASCAL VOC 2011 and by inference time (averaged over 20 trials for a 500 × 500 input on an NVIDIA Tesla K40c). We detail the architecture of the adapted nets with regard to dense prediction: number of parameter layers, receptive field size of output units, and the coarsest stride within the net. (These numbers give the best performance obtained at a fixed learning rate, not best performance possible.)

| | FCN-AlexNet | FCN-VGG16 | FCN-GoogLeNet
--- | --- | --- | ---
mean IoU | 39.8 | 56.0 | 42.5
forward time |  50ms | 210ms | 59ms
conv. layers | 8 | 16 | 22
parameters | 57M | 134M | 6M
rf size | 355 | 404 | 907
max stride | 32 | 32 | 32

### 4.2. Combining what and where

We define a new fully convolutional net (FCN) for segmentation that combines layers of the feature hierarchy and refines the spatial precision of the output. See Figure 3. 我们定义了一种新的全卷积网络(FCN)以进行分割，结合了特征层次的层，并提炼了输出的空间精度。如图3所示。

Figure 3. Our DAG nets learn to combine coarse, high layer information with fine, low layer information. Pooling and prediction layers are shown as grids that reveal relative spatial coarseness, while intermediate layers are shown as vertical lines. First row (FCN-32s): Our single-stream net, described in Section 4.1, upsamples stride 32 predictions back to pixels in a single step. Second row (FCN-16s): Combining predictions from both the final layer and the pool4 layer, at stride 16, lets our net predict finer details, while retaining high-level semantic information. Third row (FCN-8s): Additional predictions from pool3, at stride 8, provide further precision.

While fully convolutionalized classifiers can be finetuned to segmentation as shown in 4.1, and even score highly on the standard metric, their output is dissatisfyingly coarse (see Figure 4). The 32 pixel stride at the final prediction layer limits the scale of detail in the upsampled output.

分类器全卷积化后可以精调用于分割，这在4.1节叙述过了，在标准度量下甚至取得了很高的分数，其输出令人不太满意，比较粗糙（如图4所示）。最终预测层32像素的步长，限制了上采样输出的细节尺度。

Figure 4. Refining fully convolutional nets by fusing information from layers with different strides improves segmentation detail. The first three images show the output from our 32, 16, and 8 pixel stride nets (see Figure 3).

We address this by adding skips [1] that combine the final prediction layer with lower layers with finer strides. This turns a line topology into a DAG, with edges that skip ahead from lower layers to higher ones (Figure 3). As they see fewer pixels, the finer scale predictions should need fewer layers, so it makes sense to make them from shallower net outputs. Combining fine layers and coarse layers lets the model make local predictions that respect global structure. By analogy to the jet of Koenderick and van Doorn [19], we call our nonlinear feature hierarchy the deep jet.

我们通过增加跳跃[1]来解决这个问题，将最后的预测层和较低的层（步长更小一些）结合到一起。这将线状拓扑转换成了一个有向无环图(Directed Acyclic Graph, DAG)，其边从较低的层连接到较高的层（图3）。由于其感受野较小，更精细的尺度预测应当需要更少的层，所以从较浅的网络输出是可以的。将精细的层和粗糙的层结合到一起，使模型的局部预测更适合全局结构。与Koenderick and van Doorn[19]的jet类比，我们称我们的非线性特征层级为deep jet。

We first divide the output stride in half by predicting from a 16 pixel stride layer. We add a 1 × 1 convolution layer on top of pool4 to produce additional class predictions. We fuse this output with the predictions computed on top of conv7 (convolutionalized fc7) at stride 32 by adding a 2× upsampling layer and summing (Max fusion made learning difficult due to gradient switching) both predictions (see Figure 3). We initialize the 2× upsampling to bilinear interpolation, but allow the parameters to be learned as described in Section 3.3. Finally, the stride 16 predictions are upsampled back to the image. We call this net FCN-16s. FCN-16s is learned end-to-end, initialized with the parameters of the last, coarser net, which we now call FCN-32s. The new parameters acting on pool4 are zero-initialized so that the net starts with unmodified predictions. The learning rate is decreased by a factor of 100.

我们首先将输出步长减半，从步长为16的层进行预测。我们在pool4上增加1 × 1卷积层，产生额外的类别预测。我们将这个输出与conv7（卷积化的fc7）上的预测输出（步长32）结合起来，即增加了一个2倍的上采样层，然后将两个预测求和（最大值融合会使得学习更困难，这是梯度转换的原因）。这个2倍上采样使用双线性插值，但允许参数像3.3节所述的一样进行学习。最后，步长16的预测进行上采样，成为原图像的大小。我们称这个网络为FCN-16s。FCN-16s是端到端学习的，用上一个更粗糙的网络的参数进行初始化，我们称之为FCN-32s。pool4上的新参数初始化为0，这样网络开始于为修正的预测。学习率降低了100倍。

Learning this skip net improves performance on the validation set by 3.0 mean IU to 62.4. Figure 4 shows improvement in the fine structure of the output. We compared this fusion with learning only from the pool4 layer, which resulted in poor performance, and simply decreasing the learning rate without adding the skip, which resulted in an insignificant performance improvement without improving the quality of the output.

学习这个跳跃网络，将在验证集上的性能改进了3.0，MIoU达到了62.4。图4给出了输出的精细结构的改进。我们将这种融合的结果，与只从pool4层进行学习的结果进行了比较，后者得到了较差的性能，只降低了学习率，而没有增加跳跃，得到的性能改进不明显。

We continue in this fashion by fusing predictions from pool3 with a 2× upsampling of predictions fused from pool4 and conv7, building the net FCN-8s. We obtain a minor additional improvement to 62.7 mean IU, and find a slight improvement in the smoothness and detail of our output. At this point our fusion improvements have met diminishing returns, both with respect to the IU metric which emphasizes large-scale correctness, and also in terms of the improvement visible e.g. in Figure 4, so we do not continue fusing even lower layers.

我们继续这种方式，融合从pool3得到的预测结果，将其预测进行2x上采样，然后与pool4和conv7的结果融合，构建出FCN-8s网络。这次得到的MIoU改进较小，达到了62.7，输出的平滑性和细节得到了少许改进。在这一点上，我们的融合没有什么回报了，在强调大规模正确性上的MIoU上如此，在视觉效果上也是，如图4所示，所以我们没有融合更低级的层。

**Refinement by other means**. Decreasing the stride of pooling layers is the most straightforward way to obtain finer predictions. However, doing so is problematic for our VGG16-based net. Setting the pool5 stride to 1 requires our convolutionalized fc6 to have kernel size 14 × 14 to maintain its receptive field size. In addition to their computational cost, we had difficulty learning such large filters. We attempted to re-architect the layers above pool5 with smaller filters, but did not achieve comparable performance; one possible explanation is that the ILSVRC initialization of the upper layers is important.

**精炼的其他方法**。降低池化层的步长是得到更精细预测的最直接方法。但是，这样做对于我们基于VGG16的网络是有问题的。设pool5层的步长为1，需要卷积化的fc6的卷积核大小为14 × 14，以维持其感受野大小。除了其计算量，学习这样大的滤波器是有困难的。我们用更小的滤波器重新构建pool5以上的层，但没有得到很好的性能；一种可能的解释是，上部层的ILSVRC初始化是很重要的。

Another way to obtain finer predictions is to use the shift-and-stitch trick described in Section 3.2. In limited experiments, we found the cost to improvement ratio from this method to be worse than layer fusion. 另一种得到更精细的预测的方法是，使用3.2节中叙述的shift-and-stich技巧。在有限的试验里，我们发现用这种方法改进的效果比融合层更差一些。

### 4.3. Experimental framework 试验框架

**Optimization**. We train by SGD with momentum. We use a minibatch size of 20 images and fixed learning rates of $10^{−3}$, $10^{−4}$, and $5^{−5}$ for FCN-AlexNet, FCN-VGG16, and FCN-GoogLeNet, respectively, chosen by line search. We use momentum 0.9, weight decay of $5^{−4}$ or $2^{−4}$, and doubled learning rate for biases, although we found training to be sensitive to the learning rate alone. We zero-initialize the class scoring layer, as random initialization yielded neither better performance nor faster convergence. Dropout was included where used in the original classifier nets.

**优化**。我们用带有动量的SGD进行训练。我们使用的minibatch大小为20，对FCN-AlexNet，FCN-VGG16和FCN-GoogLeNet分别固定学习率为$10^{−3}$, $10^{−4}$和$5^{−5}$，由线搜索得到。我们使用动量0.9，权重衰减为$5^{−4}$或$2^{−4}$，偏置的学习率为双倍，虽然我们发现训练对于学习率有些敏感。我们对类别分数层进行零值初始化，因为随机初始化既不能得到好性能，也不能得到更快的收敛性。Dropout在原始分类网络中使用的地方也得到了使用。

**Fine-tuning**. We fine-tune all layers by back-propagation through the whole net. Fine-tuning the output classifier alone yields only 70% of the full fine-tuning performance as compared in Table 2. Training from scratch is not feasible considering the time required to learn the base classification nets. (Note that the VGG net is trained in stages, while we initialize from the full 16-layer version.) Fine-tuning takes three days on a single GPU for the coarse FCN-32s version, and about one day each to upgrade to the FCN-16s and FCN-8s versions.

**精调**。我们通过整个网络中的反向传播精调所有的层。只精调输出的分类器，可以得到全精调性能的70%，如表2所示。从头训练是不可行的，因为学习基础分类网络的时间太常。（注意VGG网络是分阶段训练的，而初始化则用的是完整的16层版）精调单个FCN-32s网络在单个GPU上花费了3天，升级到FCN-16s和FCN-8s则各多花费了一天。

Table 2. Comparison of skip FCNs on a subset 7 of PASCAL VOC 2011 segval. Learning is end-to-end, except for FCN-32s-fixed, where only the last layer is fine-tuned. Note that FCN-32s is FCN-VGG16, renamed to highlight stride.

| | pixel acc. | mean acc. | MIoU | f.w. IU
--- | --- | --- | --- | ---
FCN-32s-fixed | 83.0 | 59.7 | 45.4 | 72.0
FCN-32s | 89.1 | 73.3 | 59.4 | 81.4
FCN-16s | 90.0 | 75.7 | 62.4 | 83.0
FCN-8s | 90.3 | 75.9 | 62.7 | 83.2

**More Training Data**. The PASCAL VOC 2011 segmentation training set labels 1112 images. Hariharan et al. [14] collected labels for a larger set of 8498 PASCAL training images, which was used to train the previous state-of-the-art system, SDS [15]. This training data improves the FCN-VGG16 validation score (There are training images from [14] included in the PASCAL VOC 2011 val set, so we validate on the non-intersecting set of 736 images.) by 3.4 points to 59.4 mean IU.

**更多的训练数据**。PASCAL VOC 2011分割训练集标注了1112幅图像。Hariharan等[14]收集并标注了更多(8498)PASCAL训练图像，用于训练出之前最好的系统，SDS[15]。这种训练数据改进了FCN-VGG16验证分数3.4点，到了59.4 MIoU（有[14]中的训练数据包含在了PASCAL VOC 2011 验证集，所以我们在非交叉的736幅图像集合上进行验证）。

**Patch Sampling**. As explained in Section 3.4, our full image training effectively batches each image into a regular grid of large, overlapping patches. By contrast, prior work randomly samples patches over a full dataset [27, 2, 7, 28, 9], potentially resulting in higher variance batches that may accelerate convergence [22]. We study this tradeoff by spatially sampling the loss in the manner described earlier, making an independent choice to ignore each final layer cell with some probability 1 − p. To avoid changing the effective batch size, we simultaneously increase the number of images per batch by a factor 1/p. Note that due to the efficiency of convolution, this form of rejection sampling is still faster than patchwise training for large enough values of p (e.g., at least for p > 0.2 according to the numbers in Section 3.1). Figure 5 shows the effect of this form of sampling on convergence. We find that sampling does not have a significant effect on convergence rate compared to whole image training, but takes significantly more time due to the larger number of images that need to be considered per batch. We therefore choose unsampled, whole image training in our other experiments.

**图像块采样**。如3.4节所述，我们的整图训练有效的将每幅图像弄成了一批规则网格上的大型重叠图像块。对比起来，之前的工作在一个完整数据集上随机取样图像块[27,2,28,9]，得到的图像块之间变化大，这可能会加速收敛[22]。我们用前面叙述过的方法对损失进行采样，以进行折中，并以某概率1-p忽略每个最终层上的单元。为避免改变了有效的批大小，我们同时增加每个批次上的图像数量，增加到1/p大小。注意，由于卷积的效率，对于足够大的p值来说（如，根据3.1节中的数字，至少需要p>0.2），这种形式的rejection采样仍然比逐块训练要快。图5给出了这种形式的采样对收敛的影响。我们发现，与整图训练相比，采样对于收敛没有显著的影响，但花费的时间却多了很多，因为每个批次需要考虑的图像数量要更多。所以我们在其他试验中选择了上采样的整图训练。

**Class Balancing**. Fully convolutional training can balance classes by weighting or sampling the loss. Although our labels are mildly unbalanced (about 3/4 are background), we find class balancing unnecessary. **类别均衡**。全卷积训练可以通过对损失加权或采样，来对类别进行均衡。虽然我们的标签略微不均衡（大约3/4是背景），我们发现类别均衡不太有必要。

**Dense Prediction**. The scores are upsampled to the input dimensions by deconvolution layers within the net. Final layer deconvolutional filters are fixed to bilinear interpolation, while intermediate upsampling layers are initialized to bilinear upsampling, and then learned. **密集预测**。分数要上采样到输入的维度，方法是在网络中的解卷积层。最后一层的解卷积滤波器固定为双线性插值，而中间的上采样层初始化为双线性上采样，然后进行学习。

**Augmentation**. We tried augmenting the training data by randomly mirroring and “jittering” the images by translating them up to 32 pixels (the coarsest scale of prediction) in each direction. This yielded no noticeable improvement. **扩充**。我们尝试扩充训练数据，通过随机镜像和“抖动”图像，即在每个方向平移最多32个像素（预测的最粗糙尺度）。这样产生的改进都很小。

**Implementation**. All models are trained and tested with Caffe [18] on a single NVIDIA Tesla K40c. Our models and code are publicly available at http://fcn.berkeleyvision.org. **实现**。所有的模型都用Caffe[18]进行训练并测试，用的是单个NVidia Tesla K40c GPU。我们的模型和代码可见于如下链接。

## 5. Results 结果

We test our FCN on semantic segmentation and scene parsing, exploring PASCAL VOC, NYUDv2, and SIFT Flow. Although these tasks have historically distinguished between objects and regions, we treat both uniformly as pixel prediction. We evaluate our FCN skip architecture on each of these datasets, and then extend it to multi-modal input for NYUDv2 and multi-task prediction for the semantic and geometric labels of SIFT Flow. 我们将FCN在语义分割和场景解析上进行测试，包括PASCAL VOC，NYUDv2和SIFT Flow。虽然这些任务历史上区分了目标和区域，但我们将两者都统一视为像素预测。我们在每个数据集上对FCN跳跃架构进行评估，然后在NYUDv2上拓展到多模输入，在SIFT Flow上的语义和几何标签上拓展到多任务预测。

**Metrics**. We report four metrics from common semantic segmentation and scene parsing evaluations that are variations on pixel accuracy and region intersection over union (IU). Let $n_{ij}$ be the number of pixels of class i predicted to belong to class j, where there are $n_{cl}$ different classes, and let $t_i = \sum_j n_{ij}$ be the total number of pixels of class i. We compute:

**度量标准**。我们对常见的语义分割和场景解析给出四种度量标准，都是像素准确率和区域IoU的变体。令$n_{ij}$为类别i的像素数预测成为类别j了，其中有$n_{cl}$个不同的类别，令$t_i = \sum_j n_{ij}$为类别i上的像素总数。我们计算：

pixel accuracy: $\sum_i n_{ii}/\sum_i t_i$

mean accuracy: $\frac {1}{n_{cl}} \sum_i \frac {n_{ii}}{t_i}$

mean IoU: $\frac {1}{n_{cl}} \sum_i \frac {n_{ii}}{t_i+\sum_j n_{ji} - n_{ii}}$

frequency weighted IoU: $(\sum_k t_k)^{-1} \sum_i \frac {t_i n_{ii}} {t_i+\sum_j n_{ji} - n_{ii}}$

**PASCAL VOC**. Table 3 gives the performance of our FCN-8s on the test sets of PASCAL VOC 2011 and 2012, and compares it to the previous state-of-the-art, SDS [15], and the well-known R-CNN [10]. We achieve the best results on mean IoU (This is the only metric provided by the test server.) by a relative margin of 20%. Inference time is reduced 114× (convnet only, ignoring proposals and refinement) or 286× (overall).

**PASCAL VOC**。表3给出了我们FCN-8s在PASCAL VOC 2011和2012的测试集上的性能，并与之前最好的模型进行比较，即SDS[15]和著名的R-CNN[10]。我们取得了最好的MIoU结果（这是测试服务器上仅有的度量标准），超过了之前20%。推理时间降低了114倍（仅有卷积网络，忽略了候选和提炼）或286倍（整体）。

Table 3. Our fully convolutional net gives a 20% relative improvement over the state-of-the-art on the PASCAL VOC 2011 and 2012 test sets and reduces inference time.

|| MIoU VOC 2011 test | MIoU VOC2012 test | inference time
--- | --- | --- | ---
R-CNN[10] | 47.9 | - | -
SDS[15] | 52.6 | 51.6 | ~50s
FCN-8s | 62.7 | 62.2 | ~175ms

**NYUDv2** [30] is an RGB-D dataset collected using the Microsoft Kinect. It has 1449 RGB-D images, with pixelwise labels that have been coalesced into a 40 class semantic segmentation task by Gupta et al. [12]. We report results on the standard split of 795 training images and 654 testing images. (Note: all model selection is performed on PASCAL 2011 val.) Table 4 gives the performance of our model in several variations. First we train our unmodified coarse model (FCN-32s) on RGB images. To add depth information, we train on a model upgraded to take four-channel RGB-D input (early fusion). This provides little benefit, perhaps due to the difficultly of propagating meaningful gradients all the way through the model. Following the success of Gupta et al. [13], we try the three-dimensional HHA encoding of depth, training nets on just this information, as well as a “late fusion” of RGB and HHA where the predictions from both nets are summed at the final layer, and the resulting two-stream net is learned end-to-end. Finally we upgrade this late fusion net to a 16-stride version.

**NYUDv2**[30]是一个RGB-D数据集，Gupta等[12]通过微软Kinect收集，有1449幅RGB-D图像，并逐像素的标注，形成40个类别的语义分割任务。我们用标准分割，即795幅图像进行训练，654幅图像进行测试，给出了其结果。（注意，所有模型选择都在PASCAL 2011 val上进行） 表4给出了我们的模型在几种变化下的性能。首先我们训练未修改的粗糙模型(FCN-32s)在RGB图像上的训练结果。为增加深度信息，我们在一个升级的模型上进行训练，这个模型可以增加第四个通道的RGB-D输入（早期融合）。这带来的好处不多，可能是因为通过模型传播有意义的梯度有些困难。由于Gupta等[13]的成功，我们尝试了三维的HHA深度编码，只在这种信息上训练网络，以及RGB和HHA的晚期融合，两种网络的预测在最后一层进行了叠加，得到的两路网络是通过端到端学习的。最后我们将这个晚期融合的网络升级为16步长版本。

Table 4. Results on NYUDv2. RGBD is early-fusion of the RGB and depth channels at the input. HHA is the depth embedding of [13] as horizontal disparity, height above ground, and the angle of the local surface normal with the inferred gravity direction. RGB-HHA is the jointly trained late fusion model that sums RGB and HHA predictions.

 | | pixel acc. | mean acc. | MIoU | f.w. IoU 
--- | --- | --- | --- | ---
Gupta et al. [13] | 60.3 | - | 28.6 | 47.0
FCN-32s RGB | 60.0 | 42.2 | 29.2 | 43.9
FCN-32s RGBD | 61.5 | 42.4 | 30.5 | 45.5
FCN-32s HHA | 57.1 | 35.2 | 24.2 | 40.4
FCN-32s RGB-HHA | 67.3 | 44.9 | 32.8 | 48.0
FCN-16s RGB-HHA | 65.4 | 46.1 | 34.0 | 49.5

**SIFT Flow** is a dataset of 2,688 images with pixel labels for 33 semantic categories (“bridge”, “mountain”, “sun”), as well as three geometric categories (“horizontal”, “vertical”, and “sky”). An FCN can naturally learn a joint representation that simultaneously predicts both types of labels. We learn a two-headed version of FCN-16s with semantic and geometric prediction layers and losses. The learned model performs as well on both tasks as two independently trained models, while learning and inference are essentially as fast as each independent model by itself. The results in Table 5, computed on the standard split into 2,488 training and 200 test images, (Three of the SIFT Flow categories are not present in the test set. We made predictions across all 33 categories, but only included categories actually present in the test set in our evaluation) show state-of-the-art performance on both tasks.

**SIFT Flow**数据集包含2688幅图像，像素标签包括33种语义类别（桥，山，太阳等），以及三种几何类别（水平，垂直和天空）。FCN可以很自然的学习到联合表示，可以同时预测两种类型的标记。我们学习了一种双头版本的FCN-16s，有语义预测和几何预测的层和损失。学习到的模型在两种任务中表现都很好，就像两个独立训练的模型一样，学习和推理的过程就像每个独立的模型各自进行一样快。结果如表5所示，数据集分割也是标准的，2488幅进行训练，200幅进行测试，（SIFT Flow的三个类别没有在测试集中出现，我们对所有33个类别都进行了预测，但在评估时，只包括了在测试集上实际出现的类别），结果显示，我们在两个任务中都取得了目前最好的表现。

Table 5. Results on SIFT Flow with class segmentation (center) and geometric segmentation (right). Tighe [33] is a non-parametric transfer method. Tighe 1 is an exemplar SVM while 2 is SVM + MRF. Farabet is a multi-scale convnet trained on class-balanced samples (1) or natural frequency samples (2). Pinheiro is a multi-scale, recurrent convnet, denoted R-CNN 3 (o3 ). The metric for geometry is pixel accuracy.

 | | pixel acc. | mean acc. | MIoU | f.w. IoU | geom. acc.
--- | --- | --- | --- | --- | ---
Liu et al. [23] | 76.7 | - | - | - | -
Tighe et al. [33] | - | - | - | - | 90.8
Tighe et al. [34] 1 | 75.6 | 41.4 | - | - | -
Tighe et al. [34] 2 | 78.6 | 39.2 | - | - | -
Farabet et al. [7] 1 | 72.3 | 50.8 | - | - | -
Farabet et al. [7] 2 | 78.5 | 29.6 | - | - | -
Pinheiro et al. [28] | 77.7 | 29.8 | - | - | -
FCN-16s | 85.2 | 51.7 | 39.5 | 76.1 | 94.3

## 6. Conclusion 结论

Fully convolutional networks are a rich class of models, of which modern classification convnets are a special case. Recognizing this, extending these classification nets to segmentation, and improving the architecture with multi-resolution layer combinations dramatically improves the state-of-the-art, while simultaneously simplifying and speeding up learning and inference.

全卷积网络是非常多类别的模型，其中现代分类卷积网络是一个特例。认识到这些，并将这些分类网络拓展到分割网络，用多尺度层的结合来改进架构，可以极大的改进目前最好的效果，同时简化并加速学习和推理。