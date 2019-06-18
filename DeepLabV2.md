# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

Liang-Chieh Chen et al. Google Inc/University College London

## Abstract 摘要

In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit. First, we highlight convolution with upsampled filters, or ‘atrous convolution’, as a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation. Second, we propose atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales. Third, we improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance. Our proposed “DeepLab” system sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 79.7% mIOU in the test set, and advances the results on three other datasets: PASCAL-Context, PASCAL-Person-Part, and Cityscapes. All of our code is made publicly available online.

本文中，我们使用深度学习处理语义分割问题，做出了三个主要贡献，通过试验证明，有非常好的效果。第一，我们强调了有上采样滤波器的卷积，或称为孔洞卷积，在密集预测任务中是一个很强的工具。孔洞卷积使我们可以显式的控制DCNN计算得到的特征响应分辨率，它还使我们可以有效的放大滤波器的视野，利用更大范围的上下文，而不增加参数数量或计算量。第二，我们提出了孔洞空间金字塔池化(ASPP)，可以在多个尺度稳健的分割目标。ASPP在多个采样率和有效的视野下下处理卷积特征层，所以在多个尺度上捕获目标以及图像上下文。第三，我们结合了DCNNs和概率图模型，改进了目标边缘的定位。在DCNN中通常使用的最大池化和下采样组合，有着很好的不变性，但不利于定位准确性。我们通过将DCNN最终层的响应，与全连接CRF结合起来，克服了这个问题，改进了定位性能。我们提出的DeepLab系统，在PASCAL VOC 2012语义分割任务中确定了目前最好的结果，在测试集上达到了79.7% mIOU的结果，并在其他三个数据集上推进了结果进步：PASCAL-Context, PASCAL-Person-Part, and Cityscapes。所有代码都已开源。

**Index Terms**—Convolutional Neural Networks, Semantic Segmentation, Atrous Convolution, Conditional Random Fields.

## 1 Introduction 引言

Deep Convolutional Neural Networks (DCNNs) [1] have pushed the performance of computer vision systems to soaring heights on a broad array of high-level problems, including image classification [2], [3], [4], [5], [6] and object detection [7], [8], [9], [10], [11], [12], where DCNNs trained in an end-to-end manner have delivered strikingly better results than systems relying on hand-crafted features. Essential to this success is the built-in invariance of DCNNs to local image transformations, which allows them to learn increasingly abstract data representations [13]. This invariance is clearly desirable for classification tasks, but can hamper dense prediction tasks such as semantic segmentation, where abstraction of spatial information is undesired.

DCNNs在很多计算机视觉高层问题中得到了非常好的性能，包括图像分类，目标检测，其中DCNNs以端到端的方式训练，比手工设计特征的系统得到了好的多的结果。这种成功的主要原因是，DCNNs内在的局部图像变换不变性，这使其学习到了非常抽象的数据表示。这种不变性对于分类任务非常理想，但却不利于密集预测任务，如语义分割，其中空间信息的抽象是不利的。

In particular we consider three challenges in the application of DCNNs to semantic image segmentation: (1) reduced feature resolution, (2) existence of objects at multiple scales, and (3) reduced localization accuracy due to DCNN invariance. Next, we discuss these challenges and our approach to overcome them in our proposed DeepLab system.

特别的，我们考虑DCNNs在语义分割中的三个挑战：(1)特征分辨率降低，(2)在多个尺度都存在目标，(3)DCNNs的不变性降低了定位准确率。下面，我们讨论这三个挑战，和我们提出的DeepLab系统怎样克服。

The first challenge is caused by the repeated combination of max-pooling and downsampling (‘striding’) performed at consecutive layers of DCNNs originally designed for image classification [2], [4], [5]. This results in feature maps with significantly reduced spatial resolution when the DCNN is employed in a fully convolutional fashion [14]. In order to overcome this hurdle and efficiently produce denser feature maps, we remove the downsampling operator from the last few max pooling layers of DCNNs and instead upsample the filters in subsequent convolutional layers, resulting in feature maps computed at a higher sampling rate. Filter upsampling amounts to inserting holes (‘trous’ in French) between nonzero filter taps. This technique has a long history in signal processing, originally developed for the efficient computation of the undecimated wavelet transform in a scheme also known as “algorithme à trous” [15]. We use the term atrous convolution as a shorthand for convolution with upsampled filters. Various flavors of this idea have been used before in the context of DCNNs by [3], [6], [16]. In practice, we recover full resolution feature maps by a combination of atrous convolution, which computes feature maps more densely, followed by simple bilinear interpolation of the feature responses to the original image size. This scheme offers a simple yet powerful alternative to using deconvolutional layers [13], [14] in dense prediction tasks. Compared to regular convolution with larger filters, atrous convolution allows us to effectively enlarge the field of view of filters without increasing the number of parameters or the amount of computation.

第一个挑战是最大池化和下采样（步长）的重复组合导致的，这在为图像分类设计的DCNNs中每一层都存在。如果DCNN是全卷积的形式，这会导致特征图的空间分辨率不断降低。为克服这个问题，高效的生成更密集的特征图，我们从最后几个最大池化层中移除了下采样算子，在后续的卷积层对滤波器进行了上采样，得到的特征图是在较高的采样率下计算的。滤波器上采样，即在滤波器的非零值中间插入孔洞。这种技术在信号处理中有很长的历史，最开始提出时是为了高效计算undecimated小波变换，称为孔洞算法。我们也称这种上采样滤波器的卷积为孔洞算法，计算的特征图更密集，然后进行简单的双线性插值，达到原始图像分辨率大小。在密集预测任务中，这种方案与解卷积层相比，简单但强大。与更大滤波器的传统卷积相比，孔洞卷积使我们有效的扩大了滤波器视野，而没有增加参数数量或计算量。

The second challenge is caused by the existence of objects at multiple scales. A standard way to deal with this is to present to the DCNN rescaled versions of the same image and then aggregate the feature or score maps [6], [17], [18]. We show that this approach indeed increases the performance of our system, but comes at the cost of computing feature responses at all DCNN layers for multiple scaled versions of the input image. Instead, motivated by spatial pyramid pooling [19], [20], we propose a computationally efficient scheme of resampling a given feature layer at multiple rates prior to convolution. This amounts to probing the original image with multiple filters that have complementary effective fields of view, thus capturing objects as well as useful image context at multiple scales. Rather than actually resampling features, we efficiently implement this mapping using multiple parallel atrous convolutional layers with different sampling rates; we call the proposed technique “atrous spatial pyramid pooling” (ASPP).

第二个挑战是目标在多尺度上存在导致的。表示的处理方法应当是，向DCNN中输入变换尺度的同样图像，然后将特征或分数图聚积起来。我们证明了这种方法确实可以提高我们系统的性能，但代价是对于多个尺度的输入图像在所有DCNN层上计算特征。我们受空间金字塔池化[19,20]影响，提出了对一个特征层在多个比率上的重采样的高效计算方法。我们使用多个滤波器，其有效视野互补，来处理原始图像，所以在多个尺度上不仅捕获了目标，还有有用的图像上下文。我们没有真的重采样特征，而是使用多个并行的atrous卷积层来高效的实现这种映射，这些层有着不同的采样率；我们称这种技术为孔洞空间金字塔池化(ASPP)。

The third challenge relates to the fact that an object-centric classifier requires invariance to spatial transformations, inherently limiting the spatial accuracy of a DCNN. One way to mitigate this problem is to use skip-layers to extract “hyper-column” features from multiple network layers when computing the final segmentation result [14], [21]. Our work explores an alternative approach which we show to be highly effective. In particular, we boost our model’s ability to capture fine details by employing a fully-connected Conditional Random Field (CRF) [22]. CRFs have been broadly used in semantic segmentation to combine class scores computed by multi-way classifiers with the low-level information captured by the local interactions of pixels and edges [23], [24] or superpixels [25]. Even though works of increased sophistication have been proposed to model the hierarchical dependency [26], [27], [28] and/or high-order dependencies of segments [29], [30], [31], [32], [33], we use the fully connected pairwise CRF proposed by [22] for its efficient computation, and ability to capture fine edge details while also catering for long range dependencies. That model was shown in [22] to improve the performance of a boosting-based pixel-level classifier. In this work, we demonstrate that it leads to state-of-the-art results when coupled with a DCNN-based pixel-level classifier.

第三个挑战与下面的事实相关，即以目标为中心的分类器需要对空间变换的不变性，内在上就限制了DCNN的空间精确度。一种缓解这个问题的方法是，使用跳跃层来从多个网络层中提取hyper-column特征，然后计算最终的分割结果[14,21]。我们的工作探索了另外一种方法，非常高效。特别的，我们采用一种全连接的条件随机场(CRF)[22]来提升模型捕获精细细节的能力。CRFs在语义分割中有广泛的使用，可以将多路分类器的类别分数与低层信息结合起来，这些低层信息是像素与边缘，或超像素的局部互动捕获到的。即使提出了越来越复杂的模型来对层次依赖关系和片段的高阶依赖关系进行建模，但我们还是使用[22]提出的全连接成对CRF，因为计算效率高，而且能够捕获精细边缘细节，同时照顾到长程依赖关系。模型在[22]中改进了一种基于boosting的像素级分类器。本文中，我们证明了，当与基于DCNN的像素级分类器结合时，可以得到目前最好的结果。

A high-level illustration of the proposed DeepLab model is shown in Fig. 1. A deep convolutional neural network (VGG-16 [4] or ResNet-101 [11] in this work) trained in the task of image classification is re-purposed to the task of semantic segmentation by (1) transforming all the fully connected layers to convolutional layers (i.e., fully convolutional network [14]) and (2) increasing feature resolution through atrous convolutional layers, allowing us to compute feature responses every 8 pixels instead of every 32 pixels in the original network. We then employ bi-linear interpolation to upsample by a factor of 8 the score map to reach the original image resolution, yielding the input to a fully-connected CRF [22] that refines the segmentation results.

图1给出了我们提出的DeepLab模型的概要。用于图像分类的DCNN(VGG-16 or ResNet-101)经过两点修改后用于语义分割任务，(1)将所有的全连接层变换为卷积层（即全卷积网络[14]），(2)通过孔洞卷积层提高特征分辨率，可以每8像素计算一个特征，而不是原网络的32像素。我们然后采用双线性插值来将特征图上采样到原始图像的分辨率，结果输入到全连接CRF中，得到精炼的分割结果。

Fig. 1: Model Illustration. A Deep Convolutional Neural Network such as VGG-16 or ResNet-101 is employed in a fully convolutional fashion, using atrous convolution to reduce the degree of signal downsampling (from 32x down 8x). A bilinear interpolation stage enlarges the feature maps to the original image resolution. A fully connected CRF is then applied to refine the segmentation result and better capture the object boundaries.

Input -> DCNN with Atrous Convolution -> Coarse Score Map -> Bilinear Interpolation -> Fully Connected CRF -> Final Output

From a practical standpoint, the three main advantages of our DeepLab system are: (1) Speed: by virtue of atrous convolution, our dense DCNN operates at 8 FPS on an NVidia Titan X GPU, while Mean Field Inference for the fully-connected CRF requires 0.5 secs on a CPU. (2) Accuracy: we obtain state-of-art results on several challenging datasets, including the PASCAL VOC 2012 semantic segmentation benchmark [34], PASCAL-Context [35], PASCAL-Person-Part [36], and Cityscapes [37]. (3) Simplicity: our system is composed of a cascade of two very well-established modules, DCNNs and CRFs.

从实际的观点来看，我们DeepLab系统的三个主要优势在于：(1)速度：由于采用了孔洞卷积，我们的密集DCNN在NVIDIA Titan X GPU上能够以8 FPS的速度运行，而全连接CRF的Mean Field Inference在CPU上需要0.5s；(2)准确度：我们在几个数据集上取得了目前最好的结果，包括PASCAL VOC 2012语义分割基准测试[34]，PASCAL-Context[35]，PASCAL-Person-Part和Cityscapes；(3)简洁：我们的系统是两个模块的级联，即DCNNs和CRFs。

The updated DeepLab system we present in this paper features several improvements compared to its first version reported in our original conference publication [38]. Our new version can better segment objects at multiple scales, via either multi-scale input processing [17], [39], [40] or the proposed ASPP. We have built a residual net variant of DeepLab by adapting the state-of-art ResNet [11] image classification DCNN, achieving better semantic segmentation performance compared to our original model based on VGG-16 [4]. Finally, we present a more comprehensive experimental evaluation of multiple model variants and report state-of-art results not only on the PASCAL VOC 2012 benchmark but also on other challenging tasks. We have implemented the proposed methods by extending the Caffe framework [41]. We share our code and models at a companion web site http://liangchiehchen.com/projects/DeepLab.html.

与我们第一次在会议发表文章提出的系统相比，本文更新的DeepLab系统有几个改进。我们的新版可以更好的在多尺度上分割目标，可以通过图像多尺度处理的输入，或ASPP。我们采用最新的ResNet图像分类DCNN构建了一个ResNet-DeepLab变体，与原始的VGG-16-DeepLab相比，得到了更好的语义分割性能。最后，我们对多个模型变体进行了更广泛的试验评估，不仅在PASCAL VOC 2012基准测试上得到了最好结果，在其他任务中也是。我们通过扩展Caffe框架实现了提出的方法。代码已开源。

## 2 Related Work 相关工作

Most of the successful semantic segmentation systems developed in the previous decade relied on hand-crafted features combined with flat classifiers, such as Boosting [24], [42], Random Forests [43], or Support Vector Machines [44]. Substantial improvements have been achieved by incorporating richer information from context [45] and structured prediction techniques [22], [26], [27], [46], but the performance of these systems has always been compromised by the limited expressive power of the features. Over the past few years the breakthroughs of Deep Learning in image classification were quickly transferred to the semantic segmentation task. Since this task involves both segmentation and classification, a central question is how to combine the two tasks.

多数过去十年提出的成功的语义分割系统依赖的是手工设计的特征，并与分类器结合，如Boosting，随机森林，或支持矢量机。与更丰富的上下文信息和结构化预测技术相结合，可以得到极大的改进，但这些系统的性能一直受到特征的有限表示能力制约。过去几年深度学习在图像分类中的突破，很快迁移到了语义分割任务中。由于这个任务与分割和分类都有关，所以中心问题是如何将这两个任务结合起来。

The first family of DCNN-based systems for semantic segmentation typically employs a cascade of bottom-up image segmentation, followed by DCNN-based region classification. For instance the bounding box proposals and masked regions delivered by [47], [48] are used in [7] and [49] as inputs to a DCNN to incorporate shape information into the classification process. Similarly, the authors of [50] rely on a superpixel representation. Even though these approaches can benefit from the sharp boundaries delivered by a good segmentation, they also cannot recover from any of its errors.

基于DCNN的语义分割，第一类方法一般采用级联方法，首先是自下而上的图像分割，然后是基于DCNN的区域分类。比如，[47,48]提出的边界框建议和掩膜区域，用于[7,49]中DCNN的输入，将形状信息也用于分类过程。类似的，[50]依靠的是超像素表示。即使这些方法得益于好的分割得到的锐利边缘，但是也不能从其错误中得到恢复。

The second family of works relies on using convolutionally computed DCNN features for dense image labeling, and couples them with segmentations that are obtained independently. Among the first have been [39] who apply DCNNs at multiple image resolutions and then employ a segmentation tree to smooth the prediction results. More recently, [21] propose to use skip layers and concatenate the computed intermediate feature maps within the DCNNs for pixel classification. Further, [51] propose to pool the intermediate feature maps by region proposals. These works still employ segmentation algorithms that are decoupled from the DCNN classifier’s results, thus risking commitment to premature decisions.

第二类方法，使用DCNN卷积计算得到的特征，进行密集图像标记，然后与独立得到的分割结合起来。[39]首先将DCNN用于多种图像分辨率，然后使用分割树来平滑预测结果。最近，[21]提出使用跳跃层并将计算得到的中间特征图与DCNN拼接起来，进行像素分类。更进一步，[51]提出对中间特征图通过区域建议进行池化。这些工作采用的分割算法，仍然与DCNN分类器结果是分离的，所以有过早承诺的风险。

The third family of works uses DCNNs to directly provide dense category-level pixel labels, which makes it possible to even discard segmentation altogether. The segmentation-free approaches of [14], [52] directly apply DCNNs to the whole image in a fully convolutional fashion, transforming the last fully connected layers of the DCNN into convolutional layers. In order to deal with the spatial localization issues outlined in the introduction, [14] upsample and concatenate the scores from intermediate feature maps, while [52] refine the prediction result from coarse to fine by propagating the coarse results to another DCNN. Our work builds on these works, and as described in the introduction extends them by exerting control on the feature resolution, introducing multi-scale pooling techniques and integrating the densely connected CRF of [22] on top of the DCNN. We show that this leads to significantly better segmentation results, especially along object boundaries. The combination of DCNN and CRF is of course not new but previous works only tried locally connected CRF models. Specifically, [53] use CRFs as a proposal mechanism for a DCNN-based reranking system, while [39] treat superpixels as nodes for a local pairwise CRF and use graph-cuts for discrete inference. As such their models were limited by errors in superpixel computations or ignored long-range dependencies. Our approach instead treats every pixel as a CRF node receiving unary potentials by the DCNN. Crucially, the Gaussian CRF potentials in the fully connected CRF model of [22] that we adopt can capture long-range dependencies and at the same time the model is amenable to fast mean field inference. We note that mean field inference had been extensively studied for traditional image segmentation tasks [54], [55], [56], but these older models were typically limited to short-range connections. In independent work, [57] use a very similar densely connected CRF model to refine the results of DCNN for the problem of material classification. However, the DCNN module of [57] was only trained by sparse point supervision instead of dense supervision at every pixel.

第三类方法，使用DCNNs直接得到类别层次的密集像素标记，使其甚至可能直接抛弃分割的使用。[14,52]的无分割方法，直接对全图像以全卷积的方式使用DCNN，将DCNN最后的全连接层转化为卷积层。为解决空间定位问题，[14]对中间特征图进行上采样并拼接其分数，[52]将粗糙结果送到另一个DCNN中，从粗糙到精细的提炼其预测结果。我们的工作在这些工作基础之上，如简介中叙述的，对其进行了拓展，对特征分辨率进行了控制，引入了多尺度池化技术，将DCNN与全连接CRF结合到了一起。我们证明了，这会得到好的多的分割结果，尤其是沿着目标边缘处。DCNN和CRF的结合当然不是新的，但之前的工作只尝试了局部连接的CRF模型。具体的，[53]使用CRFs作为建议机制，进行基于DCNN的重新排序，[39]将超像素作为局部成对CRF的节点，使用图分割进行离散推理。这样他们的模型受限于超像素的错误，或忽略了长程依赖关系。我们的方法则将所有的像素都作为CRF节点。关键是，[22]的全连接CRF的Gaussian CRF potentials可以捕获长程依赖关系，同时模型可以进行快速mean field inference。我们注意到，mean field inference在传统图像分割任务中得到了广泛的研究，但这些较老的模型一般局限与短程连接。在[57]中也使用了类似的密集连接CRF模型，以改进材质分类问题中的DCNN结果。但是，[57]中的DCNN模块只是由稀疏点监督训练的，而不是在每个点上的密集监督训练的。

Since the first version of this work was made publicly available [38], the area of semantic segmentation has progressed drastically. Multiple groups have made important advances, significantly raising the bar on the PASCAL VOC 2012 semantic segmentation benchmark, as reflected to the high level of activity in the benchmark’s leaderboard [17], [40], [58], [59], [60], [61], [62], [63]. Interestingly, most top-performing methods have adopted one or both of the key ingredients of our DeepLab system: Atrous convolution for efficient dense feature extraction and refinement of the raw DCNN scores by means of a fully connected CRF. We outline below some of the most important and interesting advances.

本文的第一版工作[38]已经开源，语义分割领域进展很快。多个小组都取得了重要进展，极大的提升了在PASCAL VOC 2012语义分割测试上的成绩。有趣的是，多数最好成绩的方法都采用了我们的DeepLab系统的一个或两个关键元素：孔洞卷积，可以进行高效的密集特征提取，全连接CRF可以改进DCNN的原始分数。我们列出一些最重要和有趣的进展。

*End-to-end training for structured prediction* has more recently been explored in several related works. While we employ the CRF as a post-processing method, [40], [59], [62], [64], [65] have successfully pursued joint learning of the DCNN and CRF. In particular, [59], [65] unroll the CRF mean-field inference steps to convert the whole system into an end-to-end trainable feed-forward network, while [62] approximates one iteration of the dense CRF mean field inference [22] by convolutional layers with learnable filters. Another fruitful direction pursued by [40], [66] is to learn the pairwise terms of a CRF via a DCNN, significantly improving performance at the cost of heavier computation. In a different direction, [63] replace the bilateral filtering module used in mean field inference with a faster domain transform module [67], improving the speed and lowering the memory requirements of the overall system, while [18], [68] combine semantic segmentation with edge detection.

*结构化预测的端到端训练*在几个相关的工作中探索的很多。我们采用CRF作为后处理方法，而[40,59,62,64,65]成功的探索了DCNN和CRF的联合学习。特别是，[59,65]展开了CRF mean-field inference步骤，将整个系统转化为了一个端到端可训练的前向网络，[62]用可学习滤波器的卷积层，近似了密集CRF mean field inference的一次迭代。另一个有成果的方向[40,66]，通过DCNN学习CRF的成对项，显著改进了性能，但计算量也更多了。另一个方向[63]，使用更快的领域变换模块[67]替换了mean field inference使用的双边滤波模块，改进了整个系统的速度，降低了内存需求，[18,68]将语义分割与边缘检测结合了起来。

*Weaker supervision* has been pursued in a number of papers, relaxing the assumption that pixel-level semantic annotations are available for the whole training set [58], [69], [70], [71], achieving significantly better results than weakly-supervised pre-DCNN systems such as [72]. In another line of research, [49], [73] pursue instance segmentation, jointly tackling object detection and semantic segmentation.

*更弱的监督*是一些文章的研究点，弱化了下面的假设，即像素级的语义标注可用于整个训练集[58,69,70,71]，比弱监督的pre-DCNN系统如[72]，取得了明显更好的结果。在另一条研究线，[49,73]研究的是语义分割，同时处理目标检测和语义分割。

What we call here atrous convolution was originally developed for the efficient computation of the undecimated wavelet transform in the “algorithme à trous” scheme of [15]. We refer the interested reader to [74] for early references from the wavelet literature. Atrous convolution is also intimately related to the “noble identities” in multi-rate signal processing, which builds on the same interplay of input signal and filter sampling rates [75]. Atrous convolution is a term we first used in [6]. The same operation was later called dilated convolution by [76], a term they coined motivated by the fact that the operation corresponds to regular convolution with upsampled (or dilated in the terminology of [15]) filters. Various authors have used the same operation before for denser feature extraction in DCNNs [3], [6], [16]. Beyond mere resolution enhancement, atrous convolution allows us to enlarge the field of view of filters to incorporate larger context, which we have shown in [38] to be beneficial. This approach has been pursued further by [76], who employ a series of atrous convolutional layers with increasing rates to aggregate multiscale context. The atrous spatial pyramid pooling scheme proposed here to capture multiscale objects and context also employs multiple atrous convolutional layers with different sampling rates, which we however lay out in parallel instead of in serial. Interestingly, the atrous convolution technique has also been adopted for a broader set of tasks, such as object detection [12], [77], instance-level segmentation [78], visual question answering [79], and optical flow [80].

我们称为孔洞卷积的技术，开始时是为了高效计算undecimated小波变换。有兴趣的读者可以参考[74]，早期的小波文献。孔洞卷积还与多速率信号处理中的noble identities紧密相关，也是对输入信号和滤波器采样率进行interplay。孔洞卷积这个术语在[6]中首次使用。[76]将这种运算称为扩张卷积，与传统卷积相比，滤波器进行了上采样，所以这样取名。不同的作者之前使用同样的运算，进行更密集的特征提取。除了分辨率的提高，孔洞卷积扩大了滤波器的视野，可以得到更大的上下文，在[38]中说明了这很有好处。这种方法在[76]中得到进一步应用，采用一系列比率不断增加的孔洞卷积层，累积多尺度上下文。这里提出的孔洞空间金字塔池化方案，是为了捕获多尺度目标和上下文，也采用了采样率不同的多尺度孔洞卷积层，我们将其并行使用，而不是串行使用。有趣的是，孔洞卷积技术还被用于更多任务，如目标检测[12,77]，实例级分割[78]，视觉问题回答[79]和光流[80]。

We also show that, as expected, integrating into DeepLab more advanced image classification DCNNs such as the residual net of [11] leads to better results. This has also been observed independently by [81].

我们还证明了，将更先进的图像分类DCNNs整合进DeepLab系统，如ResNet，可以得到更好的结果。[81]也独立的得到了这个观察结果。

## 3 Methods

### 3.1 Atrous Convolution for Dense Feature Extraction and Field-of-View Enlargement

The use of DCNNs for semantic segmentation, or other dense prediction tasks, has been shown to be simply and successfully addressed by deploying DCNNs in a fully convolutional fashion [3], [14]. However, the repeated combination of max-pooling and striding at consecutive layers of these networks reduces significantly the spatial resolution of the resulting feature maps, typically by a factor of 32 across each direction in recent DCNNs. A partial remedy is to use ‘deconvolutional’ layers as in [14], which however requires additional memory and time.

DCNNs用于语义分割，或其他密集预测任务，只要以全卷积的方式部署DCNNs，就可以成功[3，14]。但是，最大池化和步长的结合，以及在网络中连续的存在，使得到的特征图的空间分辨率显著下降，一般下采样率是32。可以通过解卷积[14]进行部分补救，但需要额外的计算时间和内存。

We advocate instead the use of atrous convolution, originally developed for the efficient computation of the undecimated wavelet transform in the “algorithme à trous” scheme of [15] and used before in the DCNN context by [3], [6], [16]. This algorithm allows us to compute the responses of any layer at any desirable resolution. It can be applied post-hoc, once a network has been trained, but can also be seamlessly integrated with training.

我们推荐使用孔洞卷积，最开始提出是用于高效的计算undecimated小波变换[15]，之前在DCNN的上下文中也用过[3,6,16]。算法使我们可以在任何希望的分辨率上计算响应。可以事后应用，一旦网络被训练好，也可以与训练无缝结合。

Considering one-dimensional signals first, the output y[i] of atrous convolution (We follow the standard practice in the DCNN literature and use non-mirrored filters in this definition.) of a 1-D input signal x[i] with a filter w[k] of length K is defined as:

首先考虑一维信号，1-D输入信号为x[i]，滤波器为w[k]，长度为K，孔洞卷积的输入y[i]定义为：

$$y[i] = \sum_{k=1}^K x[i+r·k] w[k]$$(1)

The rate parameter r corresponds to the stride with which we sample the input signal. Standard convolution is a special case for rate r = 1 . See Fig. 2 for illustration. 比率参数r对应我们采样输入信号的步长。标准卷积是r=1的特殊情况。如图2所示。

Fig. 2: Illustration of atrous convolution in 1-D. (a) Sparse feature extraction with standard convolution on a low resolution input feature map. (b) Dense feature extraction with atrous convolution with rate r = 2 , applied on a high resolution input feature map.

We illustrate the algorithm’s operation in 2-D through a simple example in Fig. 3: Given an image, we assume that we first have a downsampling operation that reduces the resolution by a factor of 2, and then perform a convolution with a kernel - here, the vertical Gaussian derivative. If one implants the resulting feature map in the original image coordinates, we realize that we have obtained responses at only 1/4 of the image positions. Instead, we can compute responses at all image positions if we convolve the full resolution image with a filter ‘with holes’, in which we upsample the original filter by a factor of 2, and introduce zeros in between filter values. Although the effective filter size increases, we only need to take into account the non-zero filter values, hence both the number of filter parameters and the number of operations per position stay constant. The resulting scheme allows us to easily and explicitly control the spatial resolution of neural network feature responses.

我们在图3中描述了2D孔洞卷积的简单例子：给定一幅图像， 我们假设首先进行下采样操作，分辨率下降2倍，然后用滤波核进行卷积，这里是竖向高斯导数。如果将得到的特征图植入原始图像坐标系中，我们会发现，我们只在图像1/4的位置上得到了响应。如果我们将完整分辨率的图像与带孔的滤波器进行卷积（原始滤波器的上采样率为2，即在滤波器值之间引入0），则我们在图像所有位置上都计算响应值。虽然滤波器的有效大小增加了，但我们只需要计算滤波器的非零值，所以滤波器参数量和运算量都保持不变。得到的方案使我们可以很容易且显式的控制特征响应的空间分辨率。

Fig. 3: Illustration of atrous convolution in 2-D. Top row: sparse feature extraction with standard convolution on a low resolution input feature map. Bottom row: Dense feature extraction with atrous convolution with rate r = 2, applied on a high resolution input feature map.

In the context of DCNNs one can use atrous convolution in a chain of layers, effectively allowing us to compute the final DCNN network responses at an arbitrarily high resolution. For example, in order to double the spatial density of computed feature responses in the VGG-16 or ResNet-101 networks, we find the last pooling or convolutional layer that decreases resolution (’pool5’ or ’conv5_1’ respectively), set its stride to 1 to avoid signal decimation, and replace all subsequent convolutional layers with atrous convolutional layers having rate r = 2 . Pushing this approach all the way through the network could allow us to compute feature responses at the original image resolution, but this ends up being too costly. We have adopted instead a hybrid approach that strikes a good efficiency/accuracy trade-off, using atrous convolution to increase by a factor of 4 the density of computed feature maps, followed by fast bilinear interpolation by an additional factor of 8 to recover feature maps at the original image resolution. Bilinear interpolation is sufficient in this setting because the class score maps (corresponding to log-probabilities) are quite smooth, as illustrated in Fig. 5. Unlike the deconvolutional approach adopted by [14], the proposed approach converts image classification networks into dense feature extractors without requiring learning any extra parameters, leading to faster DCNN training in practice.

在DCNN中，可以在连续层中使用孔洞卷积，计算得到的DCNN最后层的分辨率可以任意的高。比如，为使在VGG-16或ResNet-101网络中的特征的空间率加倍，我们只需要找到降低分辨率的最后一个池化层或卷积层（分别是pool5或conv5_1），设其步长为1，以避免信号抽取，将后续的所有卷积层替换为r=2的孔洞卷积层。将这种方法应用与整个网络，使我们可以计算的特征分辨率为原始图像分辨率，但这个代价比较大。我们采取的是一种混合方法，取得了较好的效率/准确率折中，使用孔洞卷积将计算的特征图分辨率提高4倍，然后进行快速的双线性插值，比率为8，以将特征图分辨率恢复到原始图像的水平。双线性插值在这种设置中是足够的，因为类别分数图（对数概率值）非常平滑，如图5所示。与[14]中采取的解卷积方法不一样，我们的方法将图像分类网络转化成密集特征提取器，不需要学习任何多余的参数，实践中可以进行更快的DCNN训练。

Atrous convolution also allows us to arbitrarily enlarge the field-of-view of filters at any DCNN layer. State-of-the-art DCNNs typically employ spatially small convolution kernels (typically 3×3 ) in order to keep both computation and number of parameters contained. Atrous convolution with rate r introduces r − 1 zeros between consecutive filter values, effectively enlarging the kernel size of a k×k filter to $k_e = k + (k − 1)(r − 1)$ without increasing the number of parameters or the amount of computation. It thus offers an efficient mechanism to control the field-of-view and finds the best trade-off between accurate localization (small field-of-view) and context assimilation (large field-of-view). We have successfully experimented with this technique: Our DeepLab-LargeFOV model variant [38] employs atrous convolution with rate r = 12 in VGG-16 ‘fc6’ layer with significant performance gains, as detailed in Section 4.

孔洞卷积使我们可以在任意DCNN层任意放大滤波器的视野。目前最好的DCNN一般使用的都是空域上很小的卷积核（一般是3×3），以控制计算量和参数数量。比率为r的孔洞卷积，在滤波器的连续值之间引入了r-1个0，将滤波核大小为k×k的滤波器视野有效的扩大到$k_e = k + (k − 1)(r − 1)$，而没有增加参数数量，也没有增加计算量。所以这提供了有效的控制视野的方法，在准确定位（小视野）和吸收上下文（大视野）之间达到很好的平衡。我们用这种方法进行试验非常成功：我们的DeepLab-LargeFOV模型变体[38]，在VGG-16的fc6层使用r=12的孔洞卷积，得到了明显的性能改进，在第4部分进行详述。

Turning to implementation aspects, there are two efficient ways to perform atrous convolution. The first is to implicitly upsample the filters by inserting holes (zeros), or equivalently sparsely sample the input feature maps [15]. We implemented this in our earlier work [6], [38], followed by [76], within the Caffe framework [41] by adding to the im2col function (it extracts vectorized patches from multi-channel feature maps) the option to sparsely sample the underlying feature maps. The second method, originally proposed by [82] and used in [3], [16] is to subsample the input feature map by a factor equal to the atrous convolution rate r, deinterlacing it to produce $r^2$ reduced resolution maps, one for each of the r×r possible shifts. This is followed by applying standard convolution to these intermediate feature maps and reinterlacing them to the original image resolution. By reducing atrous convolution into regular convolution, it allows us to use off-the-shelf highly optimized convolution routines. We have implemented the second approach into the TensorFlow framework [83].

在实现方面，有两种有效进行孔洞卷积的方法。第一个是显式的对滤波器进行上采样，即增加孔洞（零值），或等价的对输入特征图进行稀疏采样[15]。我们在之前的工作中[6,38]这样实现，[76]也是这样实现的，对im2col函数增加一个参数选项，对特征图进行稀疏采样。第二种方法，首先在[82]中提出，在[3,16]中得到应用，是对输入特征图进行下采样，下采样率等于孔洞卷积率r，对其进行去隔行，得到$r^2$个降低分辨率的特征图，所有这些构成了r×r个可能的偏移。然后对这些中间特诊图进行标准卷积，将其进行隔行叠加成原始分辨率。通过将孔洞卷积蜕化为常规卷积，这使我们可以使用已经高度优化的卷积程序。我们在TensorFlow中实现了第二种方法[83]。

### 3.2 Multiscale Image Representations using Atrous Spatial Pyramid Pooling

DCNNs have shown a remarkable ability to implicitly represent scale, simply by being trained on datasets that contain objects of varying size. Still, explicitly accounting for object scale can improve the DCNN’s ability to successfully handle both large and small objects [6].

DCNNs在隐式的表达尺度上能力非凡，只需要在包含不同大小目标的数据集上进行训练即可。而且，显式的表达目标尺度，也可以改进DCNN的能力，更好的处理大型目标和小型目标[6]。

We have experimented with two approaches to handling scale variability in semantic segmentation. The first approach amounts to standard multiscale processing [17], [18]. We extract DCNN score maps from multiple (three in our experiments) rescaled versions of the original image using parallel DCNN branches that share the same parameters. To produce the final result, we bilinearly interpolate the feature maps from the parallel DCNN branches to the original image resolution and fuse them, by taking at each position the maximum response across the different scales. We do this both during training and testing. Multiscale processing significantly improves performance, but at the cost of computing feature responses at all DCNN layers for multiple scales of input.

我们试验了两种方法，处理语义分割中的尺度变化。第一种方法是标准的多尺度处理[17,18]。我们从原始图像的三个变尺度版中使用共享相同参数的并行DCNN分支提取DCNN分数图。为得到最后结果，我们对并行DCNN分支中得到的特征图进行双线性插值，达到原始图像分辨率，并对其融合，在每个位置都取不同尺度下的最大响应。我们在训练和测试时都这样做。多尺度处理显著改进了性能，但代价是，在多尺度输入上计算所有层的特征响应。

The second approach is inspired by the success of the R-CNN spatial pyramid pooling method of [20], which showed that regions of an arbitrary scale can be accurately and efficiently classified by resampling convolutional features extracted at a single scale. We have implemented a variant of their scheme which uses multiple parallel atrous convolutional layers with different sampling rates. The features extracted for each sampling rate are further processed in separate branches and fused to generate the final result. The proposed “atrous spatial pyramid pooling” (DeepLab-ASPP) approach generalizes our DeepLab-LargeFOV variant and is illustrated in Fig. 4.

第二种方法是受[20]的R-CNN空域金字塔池化的方法启发的，任意尺度的区域，通过对在单一尺度上提取到的特征进行重采样，可以准确高效的进行分类。我们实现了这种方法的一个变体，使用多个并行的孔洞卷积层，比率不同。每个采样率下提取的特征在不同分支下进一步处理，融合以生成最后结果。提出的孔洞空间金字塔池化(DeepLab-ASPP)方法将DeepLab-LargeFOV变体推广了，如图4所示。

Fig. 4: Atrous Spatial Pyramid Pooling (ASPP). To classify the center pixel (orange), ASPP exploits multi-scale features by employing multiple parallel filters with different rates. The effective Field-Of-Views are shown in different colors.

### 3.3 Structured Prediction with Fully-Connected Conditional Random Fields for Accurate Boundary Recovery

A trade-off between localization accuracy and classification performance seems to be inherent in DCNNs: deeper models with multiple max-pooling layers have proven most successful in classification tasks, however the increased invariance and the large receptive fields of top-level nodes can only yield smooth responses. As illustrated in Fig. 5, DCNN score maps can predict the presence and rough position of objects but cannot really delineate their borders.

定位准确率与分类性能的折中似乎是DCNNs固有的：更深的模型有多个最大池化层，在分类任务中非常成功，但是更多的不变性和更大的感受野只能产生平滑的响应。如图5所示，DCNN分数图可以预测目标的存在和大致位置，但不能细致描述其边缘。

Fig. 5: Score map (input before softmax function) and belief map (output of softmax function) for Aeroplane. We show the score (1st row) and belief (2nd row) maps after each mean field iteration. The output of last DCNN layer is used as input to the mean field inference.

Previous work has pursued two directions to address this localization challenge. The first approach is to harness information from multiple layers in the convolutional network in order to better estimate the object boundaries [14], [21], [52]. The second is to employ a super-pixel representation, essentially delegating the localization task to a low-level segmentation method [50].

之前的工作为解决定位的挑战，追求两个方向。第一种方法是利用卷积网络中多层的信息，以更好的估计目标边缘[14,21,52]。第二个是采用超像素表示，将定位任务交给低层分割方法[50]。

We pursue an alternative direction based on coupling the recognition capacity of DCNNs and the fine-grained localization accuracy of fully connected CRFs and show that it is remarkably successful in addressing the localization challenge, producing accurate semantic segmentation results and recovering object boundaries at a level of detail that is well beyond the reach of existing methods. This direction has been extended by several follow-up papers [17], [40], [58], [59], [60], [61], [62], [63], [65], since the first version of our work was published [38].

我们是另一个方向，将DCNN的识别能力和全连接CRFs的细粒度定位能力结合起来，证明这在解决定位挑战、生成准确的语义分割结果、准确恢复目标边界上是非常成功的，已有的方法远达不到这个水平。自从我们发表了论文的第一版[38]，这个方向已经有几篇文章正在研究。

Traditionally, conditional random fields (CRFs) have been employed to smooth noisy segmentation maps [23], [31]. Typically these models couple neighboring nodes, favoring same-label assignments to spatially proximal pixels. Qualitatively, the primary function of these short-range CRFs is to clean up the spurious predictions of weak classifiers built on top of local hand-engineered features.

传统上，CRFs被用于平滑含噪的分割图[23,31]。一般这些模型都与相邻的节点结合，倾向于对空间上临近的像素点指定同样标签。定性的说，短程CRFs的基本函数是为了清理弱分类器的虚假预测，这些弱分类器一般是在手工设计的局部特征上构建的。

Compared to these weaker classifiers, modern DCNN architectures such as the one we use in this work produce score maps and semantic label predictions which are qualitatively different. As illustrated in Fig. 5, the score maps are typically quite smooth and produce homogeneous classification results. In this regime, using short-range CRFs can be detrimental, as our goal should be to recover detailed local structure rather than further smooth it. Using contrast-sensitive potentials [23] in conjunction to local-range CRFs can potentially improve localization but still miss thin-structures and typically requires solving an expensive discrete optimization problem.

与这些更弱的分类器相比，现代DCNN架构，如本文中使用的这些，生成的分数图和语义标签预测，是非常不同的。如图5所示，分数图通常都很平滑，生成的是同质的分类结果。在这个领域中，使用短程CRFs是没有好处的，因为我们的目标应当是恢复局部结构的细节，而不是平滑掉它。使用对比度敏感的potentials[23]，与短程CRFs结合，可以改进定位，但会缺失精细结构，一般需要求解费时的离散优化问题。

To overcome these limitations of short-range CRFs, we integrate into our system the fully connected CRF model of [22]. The model employs the energy function 为克服短程CRFs这些局限，我们将[22]的全连接CRF模型综合进我们的系统中。模型采用下面的能量函数：

$$E(x) = \sum_i θ_i (x_i) + \sum_{ij} θ_{ij} (x_i, x_j)$$(2)

where x is the label assignment for pixels. We use as unary potential $θ_i (x_i) = − log P(x_i)$, where $P(x_i)$ is the label assignment probability at pixel i as computed by a DCNN. The pairwise potential has a form that allows for efficient inference while using a fully-connected graph, i.e. when connecting all pairs of image pixels, i, j . In particular, as in [22], we use the following expression:

其中x是对像素指定的类别。我们使用单势项$θ_i (x_i) = − log P(x_i)$，其中$P(x_i)$是在像素点i的标签指定概率，由DCNN计算得到。成对的势的形式使用的是一个全连接图，可以进行高效的推理，即连接所有的像素对i,j。特别的，和[22]中一样，我们使用下面的表达式：

$$θ_{ij} (x_i, x_j) = μ(x_i, x_j) [w_1 exp(-\frac {||p_i - p_j||^2}{2σ_α^2} - \frac {||I_i - I_j||^2}{2σ_β^2}) + w_2 exp(-\frac {||p_i - p_j||^2} {2σ_γ^2})]$$(3)

where $μ(x_i, x_j) = 1$ if $x_i != x_j$, and zero otherwise, which, as in the Potts model, means that only nodes with distinct labels are penalized. The remaining expression uses two Gaussian kernels in different feature spaces; the first, ‘bilateral’ kernel depends on both pixel positions (denoted as p) and RGB color (denoted as I), and the second kernel only depends on pixel positions. The hyper parameters $σ_α, σ_β$ and $σ_γ$ control the scale of Gaussian kernels. The first kernel forces pixels with similar color and position to have similar labels, while the second kernel only considers spatial proximity when enforcing smoothness.

其中$μ(x_i, x_j) = 1$，如果$x_i != x_j$，否则为0，和Potts模型一样，意味着不同标签的节点才被惩罚。表达式剩下的部分在不同的特征空间使用两个高斯核；第一个，双边核，依赖两个像素位置（表示为p），和RGB颜色（表示为I），第二个核只依赖于像素位置。超参数$σ_α, σ_β$和$σ_γ$控制高斯核的尺度。第一个核使相近颜色相近位置的像素，其标签也相似，而第二个核只考虑空间临近程度，要求平滑性。

Crucially, this model is amenable to efficient approximate probabilistic inference [22]. The message passing updates under a fully decomposable mean field approximation $b(x) = \prod_i b_i (x_i)$ can be expressed as Gaussian convolutions in bilateral space. High-dimensional filtering algorithms [84] significantly speed-up this computation resulting in an algorithm that is very fast in practice, requiring less that 0.5 sec on average for Pascal VOC images using the publicly available implementation of [22].

关键是，这个模型可以进行高效的概率推理近似[22]。全连接CRFs可以表示为高斯卷积在双边空间的。高维滤波算法[84]显著使这个计算加速了，得到的算法在实际中非常快，在PASCAL VOC图像上进行[22]的实现，耗时少于0.5秒。

## 4 Experimental Results

We finetune the model weights of the Imagenet-pretrained VGG-16 or ResNet-101 networks to adapt them to the semantic segmentation task in a straightforward fashion, following the procedure of [14]. We replace the 1000-way Imagenet classifier in the last layer with a classifier having as many targets as the number of semantic classes of our task (including the background, if applicable). Our loss function is the sum of cross-entropy terms for each spatial position in the CNN output map (subsampled by 8 compared to the original image). All positions and labels are equally weighted in the overall loss function (except for unlabeled pixels which are ignored). Our targets are the ground truth labels (subsampled by 8). We optimize the objective function with respect to the weights at all network layers by the standard SGD procedure of [2]. We decouple the DCNN and CRF training stages, assuming the DCNN unary terms are fixed when setting the CRF parameters.

我们将ImageNet预训练的VGG-16或ResNet-101网络的模型权重进行精调，以用于语义分割任务，采用直接精调的方式，与[14]的方法类似。我们将最后一层的1000路的ImageNet分类器，替换为我们任务中的语义类别数量的分类器（包括背景）。我们的损失函数是在每个空间位置上的交叉熵项之和（与原始图像相比，进行因子为8的下采样）。所有的位置和标签在总体损失函数中都是同样的加权（没标签的像素则被忽略）。我们的目标是真值标签（因子8下采样）。我们对网络所有权重优化目标函数，使用标准SGD[2]。我们将DCNN与CRF的训练分割开来，在设定CRF参数时，假设DCNN一元项是固定的。

We evaluate the proposed models on four challenging datasets: PASCAL VOC 2012, PASCAL-Context, PASCAL-Person-Part, and Cityscapes. We first report the main results of our conference version [38] on PASCAL VOC 2012, and move forward to latest results on all datasets. 我们在下面四个数据集上评估提出的模型：PASCAL VOC 2012, PASCAL-Context, PASCAL-Person-Part, and Cityscapes。我们首先给出在PASCAL VOC 2012上会议版论文[38]的主要结果，然后给出在所有数据集上的最新结果。

### 4.1 PASCAL VOC 2012

**Dataset**: The PASCAL VOC 2012 segmentation benchmark [34] involves 20 foreground object classes and one background class. The original dataset contains 1, 464 (train), 1, 449 (val), and 1, 456 (test) pixel-level labeled images for training, validation, and testing, respectively. The dataset is augmented by the extra annotations provided by [85], resulting in 10, 582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union (IOU) averaged across the 21 classes.

**数据集**：PASCAL VOC 2012分割基准测试[34]包括20个前景目标类别和一个背景类别。原始数据集包括1464训练、1449验证和1456测试图像，都是像素级标记的图像。数据集由[85]提供的额外标注进行了扩充，得到了10582(trainaug)训练图像。性能度量采用的是21个类别的像素IOU的平均。

#### 4.1.1 Results from our conference version

We employ the VGG-16 network pre-trained on Imagenet, adapted for semantic segmentation as described in Section 3.1. We use a mini-batch of 20 images and initial learning rate of 0.001(0.01 for the final classifier layer), multiplying the learning rate by 0.1 every 2000 iterations. We use momentum of 0.9 and weight decay of 0.0005.

我们采用的ImageNet预训练的VGG-16网络，进行精调以进行语义分割，方法如3.1节所述。我们使用mini-batch大小为20幅图像，初始学习速率为0.001（对最后一个分类层使用的是0.01），每2000次迭代将学习率乘以0.1。我们使用动量为0.9，权重衰减为0.0005。

After the DCNN has been fine-tuned on trainaug, we cross-validate the CRF parameters along the lines of [22]. We use default values of $w_2 = 3$ and $σ_γ = 3$ and we search for the best values of $w_1, σ_α$, and $σ_β$ by cross-validation on 100 images from val. We employ a coarse-to-fine search scheme. The initial search range of the parameters are $w_1 ∈ [3:6], σ_α ∈ [30:10:100]$ and $σ_β ∈ [3:6]$ (MATLAB notation), and then we refine the search step sizes around the first round’s best values. We employ 10 mean field iterations.

DCNN在trainaug上进行精调后，我们对CRF的参数进行了交叉验证，使用[22]中的方法。我们使用一些默认值，$w_2 = 3$，$σ_γ = 3$，搜索$w_1, σ_α$和$σ_β$的最佳值，使用val中的100幅图像进行交叉验证。我们采用由粗而细的搜索方案。参数的初始搜索范围是$w_1 ∈ [3:6], σ_α ∈ [30:10:100]$和$σ_β ∈ [3:6]$，然后将搜索步长大小细化，在第一轮的最佳值附近搜索。我们采用10 mean field迭代。

**Field of View and CRF**: In Tab. 1, we report experiments with DeepLab model variants that use different field-of-view sizes, obtained by adjusting the kernel size and atrous sampling rate r in the ‘fc6’ layer, as described in Sec. 3.1. We start with a direct adaptation of VGG-16 net, using the original 7 × 7 kernel size and r = 4 (since we use no stride for the last two max-pooling layers). This model yields performance of 67.64% after CRF, but is relatively slow (1.44 images per second during training). We have improved model speed to 2.9 images per second by reducing the kernel size to 4 × 4. We have experimented with two such network variants with smaller (r = 4) and larger (r = 8) FOV sizes; the latter one performs better. Finally, we employ kernel size 3×3 and even larger atrous sampling rate (r = 12), also making the network thinner by retaining a random subset of 1,024 out of the 4,096 filters in layers ‘fc6’ and ‘fc7’. The resulting model, DeepLab-CRF-LargeFOV, matches the performance of the direct VGG-16 adaptation (7 × 7 kernel size, r = 4). At the same time, DeepLab-LargeFOV is 3.36 times faster and has significantly fewer parameters (20.5M instead of 134.3M).

**视野和CRF**：在表1中，我们给出了DeepLab模型变体使用不同视野大小的结果，即调整了核大小和fc6层孔洞采样率r的大小，如3.1节所述。我们开始直接精调VGG-17网络，使用原始的7×7滤波核，r=4（因为我们对于最后两个最大池化层没有使用步长）。这个模型使用CRF后得到了67.64%的性能，但相对较慢（训练时1.44图像每秒）。将滤波核大小降到4×4后，速度提升到了2.9图像每秒。我们对网络的两个变体进行了试验，分别是小视野的r=4和大视野的r=8；后者效果更好。最后，我们采用滤波核大小3×3，使用更大的孔洞采样率r=12，同时使网络更瘦一些，在fc6和fc7中使用4096个滤波器中的随机1024个。得到的模型，DeepLab-CRF-LargeFOV，与VGG-16的直接精调结果接近（7×7滤波核大小，r=4）。同时，DeepLab-LargeFOV速度快了3.36倍，参数数量更少（20.5M）。

TABLE 1: Effect of Field-Of-View by adjusting the kernel size and atrous sampling rate r at ‘fc6’ layer. We show number of model parameters, training speed (img/sec), and val set mean IOU before and after CRF. DeepLab-LargeFOV (kernel size 3×3 , r = 12 ) strikes the best balance.

Kernel | Rate | FOV | Params | Speed | bef/aft CRF
--- | --- | --- | --- | --- | ---
7×7 | 4 | 224 | 134.3M | 1.44 | 64.38/67.64
4×4 | 4 | 128 | 65.1M | 2.90 | 59.80/63.74
4×4 | 8 | 224 | 65.1M | 2.90 | 63.41/67.14
3×3 | 12 | 224 | 20.5M | 4.84 | 62.25/67.64

The CRF substantially boosts performance of all model variants, offering a 3-5% absolute increase in mean IOU. CRF提升了所有模型变体的性能，mIOU绝对值提升了3-5%。

**Test set evaluation**: We have evaluated our DeepLab-CRF-LargeFOV model on the PASCAL VOC 2012 official test set. It achieves 70.3% mean IOU performance. 测试集评估：我们在PASCAL VOC 2012官方测试集上评估了我们的DeepLab-CRF-LargeFOV模型，得到了70.3% mIOU的性能。

#### 4.1.2 Improvements after conference version of this work

After the conference version of this work [38], we have pursued three main improvements of our model, which we discuss below: (1) different learning policy during training, (2) atrous spatial pyramid pooling, and (3) employment of deeper networks and multi-scale processing.

本文的会议版发表后，我们主要追求的是三方面的改进，讨论如下：(1)训练过程中不同的学习策略；(2)孔洞空间金字塔池化，(3)采用更深的网络和多尺度处理方法。

**Learning rate policy**: We have explored different learning rate policies when training DeepLab-LargeFOV. Similar to [86], we also found that employing a “poly” learning rate policy (the learning rate is multiplied by $(1− \frac {iter}{max_iter})^{power}$) is more effective than “step” learning rate (reduce the learning rate at a fixed step size). As shown in Tab. 2, employing “poly” (with power = 0.9) and using the same batch size and same training iterations yields 1.17% better performance than employing “step” policy. Fixing the batch size and increasing the training iteration to 10K improves the performance to 64.90% (1.48% gain); however, the total training time increases due to more training iterations. We then reduce the batch size to 10 and found that comparable performance is still maintained (64.90% vs. 64.71%). In the end, we employ batch size = 10 and 20K iterations in order to maintain similar training time as previous “step” policy. Surprisingly, this gives us the performance of 65.88% (3.63% improvement over “step”) on val, and 67.7% on test, compared to 65.1% of the original “step” setting for DeepLab-LargeFOV before CRF. We employ the “poly” learning rate policy for all experiments reported in the rest of the paper.

**学习速率策略**：我们在训练DeepLab-LargeFOV时，探索了不同的学习速率策略。与[86]类似，我们还发现采用poly学习速率策略比采用step学习速率策略要更有效。如表2所示，采用poly策略，使用相同的batch size、相同的训练迭代次数，比使用step策略得到的性能要高1.17%。固定batch size，增加训练迭代次数到10K，性能可以提升到64.90%（提升1.48%）；但是，总计训练时间由于训练迭代次数更多，所以增加了。然后我们将batch size降低到10，发现可以得到类似的性能(64.90% vs. 64.71%)。最后，我们采用batch size为10，迭代次数20K，以与step策略保持接近的训练时间。令人惊奇的是，这将在val集上的性能提升到了65.88%（比step策略高了3.63%），在test集上为67.7%，比较之下，DeepLab-LargeFOV没有CRF时，采用原始step策略只得到65.1%。我们在本文剩下的试验中都使用poly学习速率策略。

TABLE 2: PASCAL VOC 2012 val set results (%) (before CRF) as different learning hyper parameters vary. Employing “poly” learning policy is more effective than “step” when training DeepLab-LargeFOV.

Learning policy | Batch size | Iteration | mean IOU
--- | --- | --- | ---
step | 30 | 6K | 62.25
poly | 30 | 6K | 63.42
poly | 30 | 10K | 64.90
poly | 10 | 10K | 64.71
poly | 10 | 20K | 65.88

**Atrous Spatial Pyramid Pooling**: We have experimented with the proposed Atrous Spatial Pyramid Pooling (ASPP) scheme, described in Sec. 3.1. As shown in Fig. 7, ASPP for VGG-16 employs several parallel fc6-fc7-fc8 branches. They all use 3×3 kernels but different atrous rates r in the ‘fc6’ in order to capture objects of different size. In Tab. 3, we report results with several settings: (1) Our baseline LargeFOV model, having a single branch with r = 12, (2) ASPP-S, with four branches and smaller atrous rates (r = {2, 4, 8, 12}), and (3) ASPP-L, with four branches and larger rates (r = {6, 12, 18, 24}). For each variant we report results before and after CRF. As shown in the table, ASPP-S yields 1.22% improvement over the baseline LargeFOV before CRF. However, after CRF both LargeFOV and ASPP-S perform similarly. On the other hand, ASPP-L yields consistent improvements over the baseline LargeFOV both before and after CRF. We evaluate on test the proposed ASPP-L + CRF model, attaining 72.6%. We visualize the effect of the different schemes in Fig. 8.

**孔洞空间金字塔池化**：我们用提出的ASPP进行了试验，如3.1节所述。如图7所示，VGG-16的ASPP采用了几个并行的fc6-fc7-fc-8分支。它们都使用3×3滤波器核，但在fc6层中使用不同的孔洞率，以捕获不同大小的目标。在表3中，我们给出了几种设置下的结果：(1)基准的LargeFOV模型，使用单分支，r=12；(2)ASPP-S，四个分支，孔洞率更小一些(r = {2, 4, 8, 12})；(3)ASPP-L，四个分支，孔洞率更大(r = {6, 12, 18, 24})。对每个变体，我们都给出使用CRF之前和之后的结果。如表所示，ASPP-S在CRF之前比基准改进了1.22%。但是，在CRF之后，基准和ASPP-S表现类似。另一方面，ASPP-L在CRF之前和之后都比基准要好。我们在测试集上评估了提出的ASPP-L+CRF模型，得到了72.6%的结果。我们在图8中对不同的方案效果进行了可视化。

Fig. 7: DeepLab-ASPP employs multiple filters with different rates to capture objects and context at multiple scales.

TABLE 3: Effect of ASPP on PASCAL VOC 2012 val set performance (mean IOU) for VGG-16 based DeepLab model. LargeFOV: single branch, r = 12 . ASPP-S: four branches, r = {2, 4, 8, 12}. ASPP-L: four branches, r = {6, 12, 18, 24}.

Method | before CRF | after CRF
--- | --- | ---
LargeFOV | 65.76 | 69.84
ASPP-S | 66.98 | 69.73
ASPP-L | 68.96 | 71.57

Fig. 8: Qualitative segmentation results with ASPP compared to the baseline LargeFOV model. The ASPP-L model, employing multiple large FOVs can successfully capture objects as well as image context at multiple scales.

**Deeper Networks and Multiscale Processing**: We have experimented building DeepLab around the recently proposed residual net ResNet-101 [11] instead of VGG-16. Similar to what we did for VGG-16 net, we re-purpose ResNet-101 by atrous convolution, as described in Sec. 3.1. On top of that, we adopt several other features, following recent work of [17], [18], [39], [40], [58], [59], [62]: (1) Multi-scale inputs: We separately feed to the DCNN images at scale = {0.5, 0.75,1}, fusing their score maps by taking the maximum response across scales for each position separately [17]. (2) Models pretrained on MS-COCO [87]. (3) Data augmentation by randomly scaling the input images (from 0.5 to 1.5) during training. In Tab. 4, we evaluate how each of these factors, along with LargeFOV and atrous spatial pyramid pooling (ASPP), affects val set performance. Adopting ResNet-101 instead of VGG-16 significantly improves DeepLab performance (e.g., our simplest ResNet-101 based model attains 68.72%, compared to 65.76% of our DeepLab-LargeFOV VGG-16 based variant, both before CRF). Multiscale fusion [17] brings extra 2.55% improvement, while pretraining the model on MS-COCO gives another 2.01% gain. Data augmentation during training is effective (about 1.6% improvement). Employing LargeFOV (adding an atrous convolutional layer on top of ResNet, with 3×3 kernel and rate = 12) is beneficial (about 0.6% improvement). Further 0.8% improvement is achieved by atrous spatial pyramid pooling (ASPP). Post-processing our best model by dense CRF yields performance of 77.69%.

**更深的网络和多尺度处理**：我们用最近提出的ResNet-101构建DeepLab并进行了试验。与我们对VGG-16所做的类似，我们将ResNet-101与孔洞卷积结合，如3.1节所示。在这之上，我们采用了一些其他特征，与最近的[17,18,39,40,58,59,62]类似：(1)多尺度输入：我们将尺度为{0.5, 0.75,1}的图像分别送入DCNN，将其分数图进行融合，即在每个位置的不同尺度上取其最大响应[17]；(2)在MS-COCO上预训练模型；(3)对输入图像进行随机尺度变化（从0.5到1.5），从而在训练时数据扩充。在表4中，我们评估了每个因素，包括LargeFOV和ASPP，怎样影响验证集性能。采用ResNet-101，而不是VGG-16显著提升了DeepLab的性能（如，我们最简单的基于ResNet-101的模型得到了68.72%，而DeepLab-LargeFOV VGG-16在使用CRF之前只有65.76%）。多尺度融合[17]带来了额外的2.55%改进，而在MS-COCO上预训练可以得到2.01%改进。训练时的数据扩充也是有效的，可以改进大约1.6%。采用LargeFOV（在ResNet-101上增加一个孔洞卷积层，滤波核3×3，r=12）也是有好处的，改进约0.6%。ASPP还可以进一步改进0.8%。我们最好的模型，经过CRF后处理，得到性能为77.69%。

TABLE 4: Employing ResNet-101 for DeepLab on PASCAL VOC 2012 val set. MSC: Employing mutli-scale inputs with max fusion. COCO: Models pretrained on MS-COCO. Aug: Data augmentation by randomly rescaling inputs.

MSC | COCO | Aug | LargeFOV | ASPP | CRF | mIOU
--- | --- | --- | --- | --- | --- | ---
n | n | n | n | n | n | 68.72
y | n | n | n | n | n | 71.27
y | y | n | n | n | n | 73.28
y | y | y | n | n | n | 74.87
y | y | y | y | n | n | 75.54
y | y | y | n | y | n | 76.35
y | y | y | n | y | y | 77.69

**Qualitative results**: We provide qualitative visual comparisons of DeepLab’s results (our best model variant) before and after CRF in Fig. 6. The visualization results obtained by DeepLab before CRF already yields excellent segmentation results, while employing the CRF further improves the performance by removing false positives and refining object boundaries.

**定性结果**：我们将最好的模型变体在CRF处理之前与之后的结果在图6中进行了可视化对比。没有CRF处理的DeepLab模型的可视化结果已经得到了很好的分割结果，而采用CRF进一步改进了性能，去除了假阳性结果并细化了目标边缘。

Fig. 6: PASCAL VOC 2012 val results. Input image and our DeepLab results before/after CRF.

**Test set results**: We have submitted the result of our final best model to the official server, obtaining test set performance of 79.7%, as shown in Tab. 5. The model substantially outperforms previous DeepLab variants (e.g., DeepLab-LargeFOV with VGG-16 net) and is currently the top performing method on the PASCAL VOC 2012 segmentation leaderboard.

**测试集结果**：我们将最好的模型提交给了官方服务器，得到的测试集性能为79.7%，如表5所示。模型明显超过了之前的DeepLab变体（如，DeepLab-LargeFOV VGG-16），是目前PASCAL VOC 2012分割排行榜上最好的方法。

TABLE 5: Performance on PASCAL VOC 2012 test set. We have added some results from recent arXiv papers on top of the official leadearboard results.

Method | mIOU
--- | ---
DeepLab-CRF-LargeFOV-COCO [58] | 72.7
MERL_DEEP_GCRF [88] | 73.2
CRF-RNN [59] | 74.7
POSTECH_DeconvNet_CRF_VOC [61] | 74.8
BoxSup [60] | 75.2
Context + CRF-RNN [76] | 75.3
$QO_4^{mres}$ [66] | 75.5
DeepLab-CRF-Attention [17] | 75.7
CentraleSuperBoundaries++ [18] | 76.0
DeepLab-CRF-Attention-DT [63] | 76.3
H-ReNet + DenseCRF [89] | 76.8
LRR_4x_COCO [90] | 76.8
DPN [62] | 77.5
Adelaide Context [40] | 77.8
Oxford_TVG_HO_CRF [91] | 77.9
Context CRF + Guidance CRF [92] | 78.1
Adelaide_VeryDeep_FCN_VOC [93] | 79.1
DeepLab-CRF(ResNet-101) | 79.7

**VGG-16 vs. ResNet-101**: We have observed that DeepLab based on ResNet-101 [11] delivers better segmentation results along object boundaries than employing VGG-16 [4], as visualized in Fig. 9. We think the identity mapping [94] of ResNet-101 has similar effect as hyper-column features [21], which exploits the features from the intermediate layers to better localize boundaries. We further quantize this effect in Fig. 10 within the “trimap” [22], [31] (a narrow band along object boundaries). As shown in the figure, employing ResNet-101 before CRF has almost the same accuracy along object boundaries as employing VGG-16 in conjunction with a CRF. Post-processing the ResNet-101 result with a CRF further improves the segmentation result.

**VGG-16 vs. ResNet-101**：如图9所示，我们观察到基于ResNet-101的DeepLab比基于VGG-16的模型在目标边缘处可以得到更好的结果。我们认为，ResNet-101中的恒等映射与超列特征[21]有类似的效果，利用了中间层的特征更好的对边缘定位。我们在图10中进一步量化了这种效果，使用了trimap的效果（沿着目标边缘的窄带区域）。如图所示，采用ResNet-101在CRF处理之前，与VGG-16在CRF之后，在目标边缘处的准确率类似。使用CRF对ResNet-101的结果进行后处理，进一步改进了分割结果。

Fig. 9: DeepLab results based on VGG-16 net or ResNet-101 before and after CRF. The CRF is critical for accurate prediction along object boundaries with VGG-16, whereas ResNet-101 has acceptable performance even before CRF.

Fig. 10: (a) Trimap examples (top-left: image. top-right: ground-truth. bottom-left: trimap of 2 pixels. bottom-right: trimap of 10 pixels). (b) Pixel mean IOU as a function of the band width around the object boundaries when employing VGG-16 or ResNet-101 before and after CRF.

### 4.2 PASCAL-Context

**Dataset**: The PASCAL-Context dataset [35] provides detailed semantic labels for the whole scene, including both object (e.g., person) and stuff (e.g., sky). Following [35], the proposed models are evaluated on the most frequent 59 classes along with one background category. The training set and validation set contain 4998 and 5105 images.

**数据集**：PASCAL-Context数据集给出了整个场景的细节语义标签，包括目标（如person）和stuff（如天空）。与[35]一样，我们提出的模型在最常用的59类中进行评估，其中一个是背景类别。训练集和验证集包括4998和5105幅图像。

**Evaluation**: We report the evaluation results in Tab. 6. Our VGG-16 based LargeFOV variant yields 37.6% before and 39.6% after CRF. Repurposing the ResNet-101 [11] for DeepLab improves 2% over the VGG-16 LargeFOV. Similar to [17], employing multi-scale inputs and max-pooling to merge the results improves the performance to 41.4%. Pretraining the model on MS-COCO brings extra 1.5% improvement. Employing atrous spatial pyramid pooling is more effective than LargeFOV. After further employing dense CRF as post processing, our final model yields 45.7%, outperforming the current state-of-art method [40] by 2.4% without using their non-linear pairwise term. Our final model is slightly better than the concurrent work [93] by 1.2%, which also employs atrous convolution to repurpose the residual net of [11] for semantic segmentation.

**评估**：我们在表6中给出评估结果。我们基于VGG-16的LargeFOV变体在CRF之前结果为37.6%，CRF之后为39.6%。基于ResNet-101的DeepLab比基于VGG-16 LargeFOV的模型改进了2%。与[17]类似，采用了多尺度输入和最大池化以融合结果，性能改进至41.4%。在MS-COCO上预训练模型带来了额外的1.5%改进。采用ASPP比LargeFOV更有效。在采用密集CRF作为后处理后，我们最终模型得到了45.7%的结果，超过了目前最好的方法[40] 2.4%，没有使用它们的非线性成对项。我们的最终模型比并行进行的[93]好1.2%，他们也使用了孔洞卷积，将残差网络用于语义分割。

TABLE 6: Comparison with other state-of-art methods on PASCAL-Context dataset.

**Qualitative results**: We visualize the segmentation results of our best model with and without CRF as post processing in Fig. 11. DeepLab before CRF can already predict most of the object/stuff with high accuracy. Employing CRF, our model is able to further remove isolated false positives and improve the prediction along object/stuff boundaries.

**定性结果**：我们在图11中给出了最好的模型在CRF处理前后的分割可视化结果。在CRF处理之前，DeepLab已经可以基本准确的预测所有目标/stuff。采用了CRF之后，我们的模型可以进一步去除孤立的假阳性预测，改进在目标/stuff边缘处的预测。

Fig. 11: PASCAL-Context results. Input image, ground-truth, and our DeepLab results before/after CRF.

### 4.3 PASCAL-Person-Part

**Dataset**: We further perform experiments on semantic part segmentation [98], [99], using the extra PASCAL VOC 2010 annotations by [36]. We focus on the person part for the dataset, which contains more training data and large variation in object scale and human pose. Specifically, the dataset contains detailed part annotations for every person, e.g. eyes, nose. We merge the annotations to be Head, Torso, Upper/Lower Arms and Upper/Lower Legs, resulting in six person part classes and one background class. We only use those images containing persons for training (1716 images) and validation (1817 images).

**数据集**：我们在语义部位分割[98,99]上进一步进行试验，使用额外的PASCAL VOC 2010标注[36]。我们关注数据集中的身体部位，包含了很多训练数据，目标尺度和人体姿态变化都很大。具体的，数据集包含每个人详细的部分标注，如眼睛，鼻子。我们将这些标注合并成头部、躯干、胳膊上/下部，腿上/下部，得到六个身体部位，和一个背景类别。我们只使用包含人的图像进行训练（1716幅图像）和验证（1817幅图像）。

**Evaluation**: The human part segmentation results on PASCAL-Person-Part is reported in Tab. 7. [17] has already conducted experiments on this dataset with re-purposed VGG-16 net for DeepLab, attaining 56.39% (with multi-scale inputs). Therefore, in this part, we mainly focus on the effect of repurposing ResNet-101 for DeepLab. With ResNet-101, DeepLab alone yields 58.9%, significantly outperforming DeepLab-LargeFOV (VGG-16 net) and DeepLab-Attention (VGG-16 net) by about 7% and 2.5%, respectively. Incorporating multi-scale inputs and fusion by max-pooling further improves performance to 63.1%. Additionally pretraining the model on MS-COCO yields another 1.3% improvement. However, we do not observe any improvement when adopting either LargeFOV or ASPP on this dataset. Employing the dense CRF to post process our final output substantially outperforms the concurrent work [97] by 4.78%.

**评估**：在PASCAL-Person-Part上的人体部位分割结果如表7所示。[17]已经使用DeepLab VGG-16在这个数据集上进行了试验，得到了56.39%的结果（有多尺度输入）。所以，在这一部分，我们主要关注DeepLab ResNet-101的性能。DeepLab ResNet-101得到了58.9%，明显超过了DeepLab-LargeFOV VGG-16和DeepLab-Attention VGG-16，分别超过了7%和2.5%。使用了多尺度输入，最大池化融合，进一步改进性能到63.1%。在MS-COCO上预训练模型进一步改进了1.3%。但是，在这个数据集上采用LargeFOV或ASPP都没有看到任何改进。使用密集CRF进行后处理，我们模型的最终输出明显超过了[97] 4.78%。

**Qualitative results**: We visualize the results in Fig. 12. 定性结果：我们在图12中给出可视化结果。

TABLE 7: Comparison with other state-of-art methods on PASCAL-Person-Part dataset.

Fig. 12: PASCAL-Person-Part results. Input image, ground-truth, and our DeepLab results before/after CRF.

### 4.4 Cityscapes

**Dataset**: Cityscapes [37] is a recently released large-scale dataset, which contains high quality pixel-level annotations of 5000 images collected in street scenes from 50 different cities. Following the evaluation protocol [37], 19 semantic labels (belonging to 7 super categories: ground, construction, object, nature, sky, human, and vehicle) are used for evaluation (the void label is not considered for evaluation). The training, validation, and test sets contain 2975, 500, and 1525 images respectively.

**数据集**：Cityscapes[37]是最近放出的大型数据集，包含高质量的像素级标注，5000幅图像，街道场景，50个不同的城市。遵循评估协议[37]，19个语义标签（属于7个超类：背景、建筑、目标、自然、天空、人和车辆），用于评估（空标签不进行评估）。训练、验证和测试集分别包含2975,500和1525幅图像。

**Test set results of pre-release**: We have participated in benchmarking the Cityscapes dataset pre-release. As shown in the top of Tab. 8, our model attained third place, with performance of 63.1% and 64.8% (with training on additional coarsely annotated images). 我们参与了Cityscapes数据集预放出的基准测试。如表8所示，我们的模型取得了第三名的位置，性能为63.1%，用额外的粗略标注的图像进行训练可以得到64.8%的性能。

TABLE 8: Test set results on the Cityscapes dataset, comparing our DeepLab system with other state-of-art methods.

**Val set results**: After the initial release, we further explored the validation set in Tab. 9. The images of Cityscapes have resolution 2048×1024, making it a challenging problem to train deeper networks with limited GPU memory. During benchmarking the pre-release of the dataset, we downsampled the images by 2. However, we have found that it is beneficial to process the images in their original resolution. With the same training protocol, using images of original resolution significantly brings 1.9% and 1.8% improvements before and after CRF, respectively. In order to perform inference on this dataset with high resolution images, we split each image into overlapped regions, similar to [37]. We have also replaced the VGG-16 net with ResNet-101. We do not exploit multi-scale inputs due to the limited GPU memories at hand. Instead, we only explore (1) deeper networks (i.e., ResNet-101), (2) data augmentation, (3) LargeFOV or ASPP, and (4) CRF as post processing on this dataset. We first find that employing ResNet-101 alone is better than using VGG-16 net. Employing LargeFOV brings 2.6% improvement and using ASPP further improves results by 1.2%. Adopting data augmentation and CRF as post processing brings another 0.6% and 0.4%, respectively.

**验证集结果**：在初始放出后，我们进一步探索了验证集，如表9所示。Cityscapes数据集的图像分辨率为2048×1024，所以使用有限的GPU内存怎样训练更深的网络是一个很有挑战的问题。在对预放出的数据集进行基准测试时，我们将图像进行2倍下采样。但是，我们发现采用原始分辨率进行处理是有好处的。使用相同的训练方案，使用原始分辨率图像进行训练，使用CRF前后可以得到1.9%和1.8%的改进。为在这个数据集上进行高分辨率图像的推理，我们将每幅图像切分成重叠的区域，与[37]类似。我们还将VGG-16替换成ResNet-101。我们没有利用多尺度输入，因为GPU内存有限。我们探索了如下方案：(1)更深的网络，即ResNet-101；(2)数据扩充；(3)LargeFOV或ASPP；(4)使用CRF进行后处理。我们首先发现，使用ResNet-101比使用VGG-16效果更好。采用LargeFOV效果改进了2.6%，使用ASPP进一步改进了1.2%。采用数据扩充和CRF进行后处理，分别带来0.6%和0.4%的改进。

**Current test result**: We have uploaded our best model to the evaluation server, obtaining performance of 70.4%. Note that our model is only trained on the train set. 我们将最好的模型提交到评估服务器上，得到了70.4%的性能。我们的模型只是在训练集上进行的训练。

**Qualitative results**: We visualize the results in Fig. 13. 图13给出了可视化结果。

TABLE 9: Val set results on Cityscapes dataset. Full: model trained with full resolution images.

Fig. 13: Cityscapes results. Input image, ground-truth, and our DeepLab results before/after CRF.

### 4.5 Failure Modes 失败的模式

We further qualitatively analyze some failure modes of our best model variant on PASCAL VOC 2012 val set. As shown in Fig. 14, our proposed model fails to capture the delicate boundaries of objects, such as bicycle and chair. The details could not even be recovered by the CRF post processing since the unary term is not confident enough. We hypothesize the encoder-decoder structure of [100], [102] may alleviate the problem by exploiting the high resolution feature maps in the decoder path. How to efficiently incorporate the method is left as a future work.

我们进一步定性的分析了模型在PASCAL VOC 2012验证集上的一些失败模式。如图14所示，我们提出的模型无法捕获目标的精细边缘，如自行车和椅子。甚至CRF后处理也无法恢复这些细节，因为一元项或然率不够高。我们推测编码-解码器架构[100,102]可能缓解这个问题，因为在解码路径上利用了高分辨率特征图。怎样与这种方法结合，是未来的一个工作。

Fig. 14: Failure modes. Input image, ground-truth, and our DeepLab results before/after CRF.

## 5 Conclusion 结论

Our proposed “DeepLab” system re-purposes networks trained on image classification to the task of semantic segmentation by applying the ‘atrous convolution’ with upsampled filters for dense feature extraction. We further extend it to atrous spatial pyramid pooling, which encodes objects as well as image context at multiple scales. To produce semantically accurate predictions and detailed segmentation maps along object boundaries, we also combine ideas from deep convolutional neural networks and fully-connected conditional random fields. Our experimental results show that the proposed method significantly advances the state-of-art in several challenging datasets, including PASCAL VOC 2012 semantic image segmentation benchmark, PASCAL-Context, PASCAL-Person-Part, and Cityscapes datasets.

我们提出的DeepLab系统将分类网络进行了改造用于语义分割，主要是使用孔洞卷积，对滤波器进行了上卷积，以得到密集特征提取。我们进一步扩展到了孔洞空间金字塔池化，将多尺度上的目标和图像上下文融合到一起。为得到语义上准确的预测和沿着图像边缘的精细分割图，我们还将DCNNs与全连接CRF综合到了一起。我们的试验结果表明，提出的方法在几个数据集上显著推进了目前最好的水平，包括PASCAL VOC 2012语义分割基准测试，PASCAL-Context, PASCAL-Person-Part和Cityscapes数据集。