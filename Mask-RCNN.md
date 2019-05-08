# Mask R-CNN

Kaiming He et al. Facebook AI Research (FAIR)

## Abstract 摘要

We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: https://github.com/facebookresearch/Detectron.

我们提出了一种概念上很简单、灵活且通用的目标实例分割框架。我们的方法可以有效的在图像中检测目标，同时对每个实例生成一个高质量分割掩膜。这种方法我们命名为Mask R-CNN，拓展了Faster R-CNN模型，增加了一个分支，与现有的边界框预测一起，并行的预测目标掩膜。Mask R-CNN训练起来很简单，在Faster R-CNN的基础上增加了很少的计算量，运行速度可以达到5fps。而且，Mask R-CNN很容易就可以泛化到其他任务上，如，在同样的框架中可以估计人体姿态。我们在COCO挑战的三条赛道上都得到了最高的结果，包括实例分割，边界框目标检测，和人体关键点检测。没有使用任何技巧，Mask R-CNN在每个任务上超过了所有现有的单模型参赛者，包括COCO 2016挑战获胜者。我们希望我们简单有效的方法会成为一个坚实的基准，使将来的实例级识别研究更简单。代码已经开源。

## 1. Introduction 引言

The vision community has rapidly improved object detection and semantic segmentation results over a short period of time. In large part, these advances have been driven by powerful baseline systems, such as the Fast/Faster R-CNN [12, 36] and Fully Convolutional Network (FCN) [30] frameworks for object detection and semantic segmentation, respectively. These methods are conceptually intuitive and offer flexibility and robustness, together with fast training and inference time. Our goal in this work is to develop a comparably enabling framework for instance segmentation.

视觉团体已经在过去很短一段时间内，迅速的改进了目标检测和语义分割结果。这些进展大部分都是由强有力的基准系统推动的，如分别进行目标检测和语义分割的Fast/Faster R-CNN[12,36]和全卷积网络(FCN)[30]框架。这些方法概念上都很直观，兼具灵活性和稳健性，训练和推理速度也很快。本文中我们的目标是，开发出一种很好的实例分割框架。

Instance segmentation is challenging because it requires the correct detection of all objects in an image while also precisely segmenting each instance. It therefore combines elements from the classical computer vision tasks of object detection, where the goal is to classify individual objects and localize each using a bounding box, and semantic segmentation, where the goal is to classify each pixel into a fixed set of categories without differentiating object instances. (Following common terminology, we use object detection to denote detection via bounding boxes, not masks, and semantic segmentation to denote per-pixel classification without differentiating instances. Yet we note that instance segmentation is both semantic and a form of detection.) Given this, one might expect a complex method is required to achieve good results. However, we show that a surprisingly simple, flexible, and fast system can surpass prior state-of-the-art instance segmentation results.

实例分割是很有挑战性的，因为需要正确检测图像中的所有目标，同时精确的分割每个实例，所以结合了经典计算机视觉中的目标检测和语义分割的元素，其中目标检测的目标是对单个目标分类并使用边界框定位，语义分割的目标是将每个像素分类为固定类别集合，而不需要区分目标实例。（遵循常用的术语惯例，我们所说的目标检测是指，通过边界框进行检测，而不是通过掩膜进行检测；语义分割是指，逐像素的分类，而不需要区分实例。但是我们指出，实例分割既是一种语义分割，也是一种检测。）给定这个目标，可能会以为需要一种复杂的方法才能得到很好的结果。但是，我们证明了，一种非常简单、灵活、快速的系统可以超过之前最好的实例分割结果。

Our method, called Mask R-CNN, extends Faster R-CNN [36] by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression (Figure 1). The mask branch is a small FCN applied to each RoI, predicting a segmentation mask in a pixel-to-pixel manner. Mask R-CNN is simple to implement and train given the Faster R-CNN framework, which facilitates a wide range of flexible architecture designs. Additionally, the mask branch only adds a small computational overhead, enabling a fast system and rapid experimentation.

我们的方法，称之为Mask R-CNN，拓展了Faster R-CNN[36]，在每个RoI上增加了一个预测分割掩膜的分支，与现有的分类和边界框回归分支是并行关系（图1）。掩膜分支是一个小的全卷积网络，在每个RoI上进行应用，逐像素的预测一个分割掩膜。有了Faster R-CNN框架，Mask R-CNN非常容易实现并训练，可以利用很非常多的灵活的架构设计。另外，掩膜分支只增加了很小的计算量，所以实现的系统非常快，试验也非常迅速。

In principle Mask R-CNN is an intuitive extension of Faster R-CNN, yet constructing the mask branch properly is critical for good results. Most importantly, Faster R-CNN was not designed for pixel-to-pixel alignment between network inputs and outputs. This is most evident in how RoIPool [18, 12], the de facto core operation for attending to instances, performs coarse spatial quantization for feature extraction. To fix the misalignment, we propose a simple, quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations. Despite being a seemingly minor change, RoIAlign has a large impact: it improves mask accuracy by relative 10% to 50%, showing bigger gains under stricter localization metrics. Second, we found it essential to decouple mask and class prediction: we predict a binary mask for each class independently, without competition among classes, and rely on the network’s RoI classification branch to predict the category. In contrast, FCNs usually perform per-pixel multi-class categorization, which couples segmentation and classification, and based on our experiments works poorly for instance segmentation.

原则上，Mask R-CNN是Faster R-CNN的直觉上的拓展，但是合理的构建掩膜分支是得到好结果的关键。最重要的是，Faster R-CNN并不是设计用于在网络输入和输出间进行逐像素的对齐的。最明显的表现是，RoIPool[18,12]，这个实际上的核心运算，在特征提取的时候进行了粗糙的空域量化。为改正不对齐的问题，我们提出一种简单的、不用量化的层，称为RoIAligh层，忠实的保存了精确的空间位置。尽管看上去改动很小，但RoIAlign的影响很大：它改进掩膜正确率的相对幅度在10%到50%，在更严格的度量标准下都取得了更好的改进。第二，我们发现这是将掩膜预测与类别预测分离的基础：我们对每个类别独立的预测一个二值掩膜，类别间没有竞争，依靠的是网络的RoI分类分支来预测类别。比较起来，FCNs通常进行逐像素的多类别归类，这将分割与分类结合到了一起，我们也进行了试验，这对实例分割效果不好。

Without bells and whistles, Mask R-CNN surpasses all previous state-of-the-art single-model results on the COCO instance segmentation task [28], including the heavily-engineered entries from the 2016 competition winner. As a by-product, our method also excels on the COCO object detection task. In ablation experiments, we evaluate multiple basic instantiations, which allows us to demonstrate its robustness and analyze the effects of core factors.

Mask R-CNN没有使用什么特别的技巧，就超过了之前最好的单模型在COCO实例分割任务[28]上的结果，包括2016年竞赛获胜者重手工设计的模型。我们的模型还在COCO目标检测任务上得到了很好的结果，这只是一个副产品。在分离试验中，我们评估了多个基本的实现，这使我们可以证明其稳健性并分析其核心因素的效果。

Our models can run at about 200ms per frame on a GPU, and training on COCO takes one to two days on a single 8-GPU machine. We believe the fast train and test speeds, together with the framework’s flexibility and accuracy, will benefit and ease future research on instance segmentation.

我们的模型在GPU上的运行速度可以达到200ms每帧，COCO上的训练在8 GPU机器上耗时1到2天。我们相信，模型快速的训练和测试速度，和框架的灵活性及准确性，会使将来实例分割的研究更加容易。

Finally, we showcase the generality of our framework via the task of human pose estimation on the COCO keypoint dataset [28]. By viewing each keypoint as a one-hot binary mask, with minimal modification Mask R-CNN can be applied to detect instance-specific poses. Mask R-CNN surpasses the winner of the 2016 COCO keypoint competition, and at the same time runs at 5 fps. Mask R-CNN, therefore, can be seen more broadly as a flexible framework for instance-level recognition and can be readily extended to more complex tasks.

最后，我们在COCO关键点数据集[28]上进行了人体姿态估计，展现了我们框架的一般性。我们将每个关键点看作一个独热的二维掩膜，对Mask R-CNN进行了很小的修改，就可以应用于检测实例的姿态。Mask R-CNN超过了2016 COCO关键点竞赛的获胜者，并且运行速度达到了5 fps。Mask R-CNN因此可以看作是实例识别的灵活框架，可以拓展到更复杂的任务。

We have released code to facilitate future research. 代码已经开源。

## 2. Related Work 相关工作

**R-CNN**: The Region-based CNN (R-CNN) approach [13] to bounding-box object detection is to attend to a manageable number of candidate object regions [42, 20] and evaluate convolutional networks [25, 24] independently on each RoI. R-CNN was extended [18, 12] to allow attending to RoIs on feature maps using RoIPool, leading to fast speed and better accuracy. Faster R-CNN [36] advanced this stream by learning the attention mechanism with a Region Proposal Network (RPN). Faster R-CNN is flexible and robust to many follow-up improvements (e.g., [38, 27, 21]), and is the current leading framework in several benchmarks.

**R-CNN**：R-CNN方法[13]进行边界框目标检测，就是处理一定数量的候选目标区域[42,20]，在每个RoI上独立的用卷积网络[25,24]处理。R-CNN在[18,12]中得到了拓展，可以使用RoIPool在特征图上处理RoI，得到了更快的速度和更高的准确率。Faster R-CNN[36]进一步做了推进，用RPN学习了注意力机制。Faster R-CNN灵活又稳健，有很多随后的改进（如，[38,27,21]），在几个基准测试中都是领先的框架。

**Instance Segmentation**: Driven by the effectiveness of R-CNN, many approaches to instance segmentation are based on segment proposals. Earlier methods [13, 15, 16, 9] resorted to bottom-up segments [42, 2]. DeepMask [33] and following works [34, 8] learn to propose segment candidates, which are then classified by Fast R-CNN. In these methods, segmentation precedes recognition, which is slow and less accurate. Likewise, Dai et al. [10] proposed a complex multiple-stage cascade that predicts segment proposals from bounding-box proposals, followed by classification. Instead, our method is based on parallel prediction of masks and class labels, which is simpler and more flexible.

**实例分割**：受R-CNN有效性的驱动，很多实例分割的方法都是基于分割建议的。较早的方法[13,15,16,9]使用自下而上的分割[42,2]。DeepMask[33]和随后的工作[34,8]学习提出分割候选，然后用Fast R-CNN分类。在这些方法中，分割是在识别之前，这很慢而且准确性不高。类似的，Dai等[10]提出了一种复杂的多阶段级联，从边界框建议中预测分割建议，然后进行分类。而我们的方法则是基于并行的类别标签预测和掩膜预测，更简单也更灵活。

Most recently, Li et al. [26] combined the segment proposal system in [8] and object detection system in [11] for “fully convolutional instance segmentation” (FCIS). The common idea in [8, 11, 26] is to predict a set of position-sensitive output channels fully convolutionally. These channels simultaneously address object classes, boxes, and masks, making the system fast. But FCIS exhibits systematic errors on overlapping instances and creates spurious edges (Figure 6), showing that it is challenged by the fundamental difficulties of segmenting instances.

最近，Li等[26]将[8]的分割建议系统和[11]的目标检测系统结合起来，形成了全卷积实例分割(FCIS)。[8,11,26]的同样思想都是用全卷积预测对位置敏感的输出通道集合。这些通道同时处理目标类别，边界框，和掩膜，所以这个系统很快。但FCIS在重叠的实例上出现了系统性的错误，产生了虚假边缘（图6），说明实例分割的基本困难还是不容易处理。

Another family of solutions [23, 4, 3, 29] to instance segmentation are driven by the success of semantic segmentation. Starting from per-pixel classification results (e.g., FCN outputs), these methods attempt to cut the pixels of the same category into different instances. In contrast to the segmentation-first strategy of these methods, Mask R-CNN is based on an instance-first strategy. We expect a deeper incorporation of both strategies will be studied in the future.

另一类实例分割的方法[23,4,3,29]是受语义分割成功的推动。从逐像素分类结果开始（如FCN输出），这些方法试图将同一类别的像素分成不同的实体。与这些方法中的分类第一的策略相比，Mask R-CNN是基于实例第一的策略。我们期待在将来会有两种策略更深的融合。

## 3. Mask R-CNN

Mask R-CNN is conceptually simple: Faster R-CNN has two outputs for each candidate object, a class label and a bounding-box offset; to this we add a third branch that outputs the object mask. Mask R-CNN is thus a natural and intuitive idea. But the additional mask output is distinct from the class and box outputs, requiring extraction of much finer spatial layout of an object. Next, we introduce the key elements of Mask R-CNN, including pixel-to-pixel alignment, which is the main missing piece of Fast/Faster R-CNN.

Mask R-CNN在概念上很简单：Faster R-CNN对每个候选目标有两个输出，一个目标类别和一个边界框偏移；我们在此之上加上一个分支，输出目标的掩膜。Mask R-CNN所以的思想是自然并且直观的。但额外的掩膜输出与类别和边界框输出不同，需要提取目标更细致的空间位置。下面，我们介绍Mask R-CNN的关键元素，包括像素对像素的对齐，这是Fast/Faster R-CNN主要缺少的部分。

**Faster R-CNN**: We begin by briefly reviewing the Faster R-CNN detector [36]. Faster R-CNN consists of two stages. The first stage, called a Region Proposal Network (RPN), proposes candidate object bounding boxes. The second stage, which is in essence Fast R-CNN [12], extracts features using RoIPool from each candidate box and performs classification and bounding-box regression. The features used by both stages can be shared for faster inference. We refer readers to [21] for latest, comprehensive comparisons between Faster R-CNN and other frameworks.

**Faster R-CNN**：我们开始简要回顾一下Faster R-CNN检测器[36]。Faster R-CNN包括两个阶段。第一阶段称为区域建议网络(RPN)，提出候选目标边界框。第二阶段，是Fast R-CNN的精华[12]，使用RoIPool从每个候选框中提取特征，进行分类和边界框回归。两个阶段用到的特征可以共享，使推理更快。我们推荐读者参考[21]，在Faster R-CNN和其他框架之间进行最新的综合比较。

**Mask R-CNN**: Mask R-CNN adopts the same two-stage procedure, with an identical first stage (which is RPN). In the second stage, in parallel to predicting the class and box offset, Mask R-CNN also outputs a binary mask for each RoI. This is in contrast to most recent systems, where classification depends on mask predictions (e.g. [33, 10, 26]). Our approach follows the spirit of Fast R-CNN [12] that applies bounding-box classification and regression in parallel (which turned out to largely simplify the multi-stage pipeline of original R-CNN [13]).

**Mask R-CNN**：Mask R-CNN采用相同的两阶段过程，第一阶段完全相同（即RPN）。第二阶段，与预测类别和边界框偏移并行的是，Mask R-CNN同时还对每个RoI输出了一个二值掩膜。这与最近的多数系统不一样，他们都使分类依赖于掩膜预测（如[33,10,26]）。我们的方法遵循Fast R-CNN[12]的精神，将边界框分类和回归并行起来（这简化了原始R-CNN[13]的多阶段过程）。

Formally, during training, we define a multi-task loss on each sampled RoI as $L = L_{cls} + L_{box} + L_{mask}$. The classification loss $L_{cls}$ and bounding-box loss $L_{box}$ are identical as those defined in [12]. The mask branch has a $Km^2$-dimensional output for each RoI, which encodes K binary masks of resolution m × m, one for each of the K classes. To this we apply a per-pixel sigmoid, and define $L_{mask}$ as the average binary cross-entropy loss. For an RoI associated with ground-truth class k, $L_{mask}$ is only defined on the k-th mask (other mask outputs do not contribute to the loss).

正式的，在训练过程中，我们在每个取样的RoI上定义一个多类别损失，$L = L_{cls} + L_{box} + L_{mask}$。分类损失$L_{cls}$和边界框损失$L_{box}$与[12]中的定义一样。每个RoI中掩膜分支的输出维度为$Km^2$维，其中K表示类别数目，二值掩膜的分辨率为m × m。为此，我们应用一个逐像素的sigmoid，并定义$L_{mask}$为平均二值交叉熵损失。对于一个RoI，与真值类别k关联上，$L_{mask}$只在第k个掩膜上有定义（其他掩膜输出对这个损失没有贡献）。

Our definition of $L_{mask}$ allows the network to generate masks for every class without competition among classes; we rely on the dedicated classification branch to predict the class label used to select the output mask. This decouples mask and class prediction. This is different from common practice when applying FCNs [30] to semantic segmentation, which typically uses a per-pixel softmax and a multinomial cross-entropy loss. In that case, masks across classes compete; in our case, with a per-pixel sigmoid and a binary loss, they do not. We show by experiments that this formulation is key for good instance segmentation results.

我们定义的$L_{mask}$使网络可以对每个类别生成掩膜，类别间不存在竞争；我们依靠分类分支来预测类别标签，以选择输出掩膜。这将掩膜和类别预测分离开来。当FCN[30]进行语义分割时，会使用逐像素的softmax和一个多项式交叉熵损失，这种情况下，不同类别间的掩膜形成竞争关系；在我们的情况下，因为有逐像素的sigmoid和二值损失，所以没有竞争关系。我们通过试验表明，这是得到好的实例分割结果的关键。

**Mask Representation**: A mask encodes an input object’s spatial layout. Thus, unlike class labels or box offsets that are inevitably collapsed into short output vectors by fully-connected (fc) layers, extracting the spatial structure of masks can be addressed naturally by the pixel-to-pixel correspondence provided by convolutions.

**掩膜表示**：掩膜是输入目标的空间布局。所以，与类别标签或边界框偏移不一样，它们最后都由全连接层坍缩成很短的输出向量，提取掩膜的空间结构的问题，可以很自然的由卷积提供的像素对像素的对应关系解决。

Specifically, we predict an m × m mask from each RoI using an FCN [30]. This allows each layer in the mask branch to maintain the explicit m × m object spatial layout without collapsing it into a vector representation that lacks spatial dimensions. Unlike previous methods that resort to fc layers for mask prediction [33, 34, 10], our fully convolutional representation requires fewer parameters, and is more accurate as demonstrated by experiments.

特别的，我们从每个RoI中使用FCN[30]预测一个m × m的掩膜。这使掩膜分支中的每一层都保持了显式的m × m的目标空间布局，不用坍缩成一个向量表示，缺少空间维度信息。之前的方法都使用fc层来进行掩膜预测[33,34,10]，与之不同，我们的全卷积表示需要更少的参数，试验也证明了，准确度会更高。

This pixel-to-pixel behavior requires our RoI features, which themselves are small feature maps, to be well aligned to faithfully preserve the explicit per-pixel spatial correspondence. This motivated us to develop the following RoIAlign layer that plays a key role in mask prediction.

这种像素对像素的行为需要我们的RoI特征（它们本身就是小的特征图）很好的对齐，忠实的保存每个像素的空间对应关系。这促使我们提出了下面的RoIAlign层，在掩膜预测中扮演了关键的角色。

**RoIAlign**: RoIPool [12] is a standard operation for extracting a small feature map (e.g., 7×7) from each RoI. RoIPool first quantizes a floating-number RoI to the discrete granularity of the feature map, this quantized RoI is then subdivided into spatial bins which are themselves quantized, and finally feature values covered by each bin are aggregated (usually by max pooling). Quantization is performed, e.g., on a continuous coordinate x by computing [x/16], where 16 is a feature map stride and [·] is rounding; likewise, quantization is performed when dividing into bins (e.g., 7×7). These quantizations introduce misalignments between the RoI and the extracted features. While this may not impact classification, which is robust to small translations, it has a large negative effect on predicting pixel-accurate masks.

**RoIAigh**：RoIPool[12]是从每个RoI中提取一个小的特征图（如7×7）的标准操作。RoIPool首先将浮点的RoI量化到离散粒度的特征图，这种量化的RoI然后再分割成空间格子，各个格子都是量化过的，最后每个格子内的特征值进行合计（通常是max pooling）。量化是在连续的坐标x中，如计算[x/16]，其中16是特征图的步长，[·]是四舍五入；类似的，分成小格子（如7×7）时，也进行量化。这些量化会导致RoI和提取的特征间对不齐。这个可能不会影响分类，因为对于小的平移是稳健的，但却对预测精确到像素的掩膜有着很大的负面影响。

To address this, we propose an RoIAlign layer that removes the harsh quantization of RoIPool, properly aligning the extracted features with the input. Our proposed change is simple: we avoid any quantization of the RoI boundaries or bins (i.e., we use x/16 instead of [x/16]). We use bilinear interpolation [22] to compute the exact values of the input features at four regularly sampled locations in each RoI bin, and aggregate the result (using max or average), see Figure 3 for details. We note that the results are not sensitive to the exact sampling locations, or how many points are sampled, as long as no quantization is performed.

为解决这个问题，我们提出一种RoIAlign层，去除了RoIPool的粗糙的量化，将提取的特征与输入进行合理的对齐。我们提出的改变非常简单：我们避免RoI边界或格子的任何量化（即，使用x/16，而不是[x/16]）。我们使用双线性插值[22]来在每个RoI格子中的4个规则取样的位置上计算输入特征的确切值，然后对结果进行合计（使用max或平均），详见图3。我们要说明，结果对于精确的取样位置，或者有多少点进行了采样，是不敏感的，只要没有进行量化。

Figure 3. RoIAlign: The dashed grid represents a feature map, the solid lines an RoI (with 2×2 bins in this example), and the dots the 4 sampling points in each bin. RoIAlign computes the value of each sampling point by bilinear interpolation from the nearby grid points on the feature map. No quantization is performed on any coordinates involved in the RoI, its bins, or the sampling points.

图3. RoIAlign：虚线格子表示特征图，实线表示一个RoI（本例子中有2×2的格子），点表示每个格子中的4个采样点。RoIAlign通过特征图上附近的网格点的双线性插值来计算每个采样点的值。RoI、格子或采样点涉及到的任何坐标点都没进行量化

RoIAlign leads to large improvements as we show in §4.2. We also compare to the RoIWarp operation proposed in [10]. Unlike RoIAlign, RoIWarp overlooked the alignment issue and was implemented in [10] as quantizing RoI just like RoIPool. So even though RoIWarp also adopts bilinear resampling motivated by [22], it performs on par with RoIPool as shown by experiments (more details in Table 2c), demonstrating the crucial role of alignment.

如4.2节所示，RoIAlign带来了很大的改进。我们还与[10]中提出的RoIWarp操作进行了比较。与RoIAlign不同，RoIWarp忽略了对齐的问题，[10]中的实现就是RoI量化，与RoIPool一样。所以即使RoIWarp也受到[22]启发采取了双线性重采样，其表现还是与RoIPool类似，试验表明了这样的结论（详见表2c），表明了对齐的关键作用。

**Network Architecture**: To demonstrate the generality of our approach, we instantiate Mask R-CNN with multiple architectures. For clarity, we differentiate between: (i) the convolutional backbone architecture used for feature extraction over an entire image, and (ii) the network head for bounding-box recognition (classification and regression) and mask prediction that is applied separately to each RoI.

**网络架构**：为表明我们方法的通用性，我们用多个架构对Mask R-CNN进行实例化。为清楚的表达，我们区分：(i)用于在整个图像上进行特征提取的卷积骨干架构，和(ii)用于边界框识别（分类和回归）和掩膜预测的网络头，这在每个RoI上都分别进行了应用。

We denote the backbone architecture using the nomenclature network-depth-features. We evaluate ResNet [19] and ResNeXt [45] networks of depth 50 or 101 layers. The original implementation of Faster R-CNN with ResNets [19] extracted features from the final convolutional layer of the 4-th stage, which we call C4. This backbone with ResNet-50, for example, is denoted by ResNet-50-C4. This is a common choice used in [19, 10, 21, 39].

我们使用网络-深度-特征的命名法表示骨干网络架构。我们采用50层或101层的ResNet[19]和ResNeXt[45]网络。Faster R-CNN的原始实现是用ResNets[19]第4阶段最后的卷积层提取的特征，我们称之为C4。如果骨干网络是ResNet-50，那么就表示为ResNet-50-C4。这在[19,10,21,39]等文献中是通用的表示法。

We also explore another more effective backbone recently proposed by Lin et al. [27], called a Feature Pyramid Network (FPN). FPN uses a top-down architecture with lateral connections to build an in-network feature pyramid from a single-scale input. Faster R-CNN with an FPN backbone extracts RoI features from different levels of the feature pyramid according to their scale, but otherwise the rest of the approach is similar to vanilla ResNet. Using a ResNet-FPN backbone for feature extraction with Mask R-CNN gives excellent gains in both accuracy and speed. For further details on FPN, we refer readers to [27].

我们还使用了Lin等[27]最近提出的更有效的骨干，称为FPN，其使用了带有横向连接的自上而下的架构，从单尺度输入构建了一个网络中的特征金字塔。使用FPN骨干网络的Faster R-CNN，根据其尺度从特征金字塔的不同层中提取RoI特征，但这个方法的其他部分与传统ResNet类似。使用ResNet-FPN骨干网络进行特征提取的Mask R-CNN给出了极好的准确率和速度提升。FPN进一步的细节，读者请参考[27]。

For the network head we closely follow architectures presented in previous work to which we add a fully convolutional mask prediction branch. Specifically, we extend the Faster R-CNN box heads from the ResNet [19] and FPN [27] papers. Details are shown in Figure 4. The head on the ResNet-C4 backbone includes the 5-th stage of ResNet (namely, the 9-layer ‘res5’ [19]), which is compute-intensive. For FPN, the backbone already includes res5 and thus allows for a more efficient head that uses fewer filters.

对于网络头，我们按照之前工作提出的架构来，增加了一个全卷积掩膜预测分支。特别的，我们从ResNet[19]和FPN[27]中拓展Faster R-CNN的网络头。详见图4。ResNet-C4骨架网络的头包括哦第5阶段的ResNet（即，9层的res5[19]），其计算量很大。对于FPN，骨干网络已经包括了res5，所以可以用更更少滤波器的网络头。

We note that our mask branches have a straightforward structure. More complex designs have the potential to improve performance but are not the focus of this work. 我们说明，我们的掩膜分支结构很直接。更复杂的设计有可能进一步改进性能，但不是本文的重点。

Figure 4. Head Architecture: We extend two existing Faster R-CNN heads [19, 27]. Left/Right panels show the heads for the ResNet C4 and FPN backbones, from [19] and [27], respectively, to which a mask branch is added. Numbers denote spatial resolution and channels. Arrows denote either conv, deconv, or fc layers as can be inferred from context (conv preserves spatial dimension while deconv increases it). All convs are 3×3, except the output conv which is 1×1, deconvs are 2×2 with stride 2, and we use ReLU [31] in hidden layers. Left: ‘res5’ denotes ResNet’s fifth stage, which for simplicity we altered so that the first conv operates on a 7×7 RoI with stride 1 (instead of 14×14 / stride 2 as in [19]). Right: ‘×4’ denotes a stack of four consecutive convs.

图4. 头部架构：我们拓展两个已有的Faster R-CNN头部架构[19,27]。左右两边分别是ResNet-C4[19]和FPN[27]骨干网络的头部网络，都增加了一个掩膜分支。数字表示空间分辨率和通道数。箭头表示卷积层，解卷积层，或fc层，可以从上下文中推理出来（卷积层保持空间分辨率，解卷积层提高空间分辨率）。所有卷积层都是3×3的，除了输出的卷积层是1×1，解卷积层是2×2的，步长为2，我们在隐藏层中使用ReLU[31]。左边：res5表示ResNet的第5阶段，简化起见我们进行了改变，在7×7 RoI上进行的第一个卷积步长为1（[19]中是14×14，步长为2）。右：×4表示堆叠连续4个卷积层。

### 3.1. Implementation Details

We set hyper-parameters following existing Fast/Faster R-CNN work [12, 36, 27]. Although these decisions were made for object detection in original papers [12, 36, 27], we found our instance segmentation system is robust to them.

我们按照现有的Fast/Faster R-CNN工作[12,36,27]设置超参数。虽然这些设置是为目标检测准备的[12,36,27]，我们发现我们的实例分割使用起来也很好。

**Training**: As in Fast R-CNN, an RoI is considered positive if it has IoU with a ground-truth box of at least 0.5 and negative otherwise. The mask loss $L_{mask}$ is defined only on positive RoIs. The mask target is the intersection between an RoI and its associated ground-truth mask.

**训练**：像在Fast R-CNN中一样，一个RoI如果与真值边界框的IoU超过0.5，就认为是正的，否则就是负的。掩膜损失$L_{mask}$只在正RoIs上有定义。掩膜对象是RoI与其关联的真值掩膜的交集。

We adopt image-centric training [12]. Images are resized such that their scale (shorter edge) is 800 pixels [27]. Each mini-batch has 2 images per GPU and each image has N sampled RoIs, with a ratio of 1:3 of positive to negatives [12]. N is 64 for the C4 backbone (as in [12, 36]) and 512 for FPN (as in [27]). We train on 8 GPUs (so effective mini-batch size is 16) for 160k iterations, with a learning rate of 0.02 which is decreased by 10 at the 120k iteration. We use a weight decay of 0.0001 and momentum of 0.9. With ResNeXt [45], we train with 1 image per GPU and the same number of iterations, with a starting learning rate of 0.01.

我们采用图像中心的(image-centric)训练[12]。改变图像的大小，使其尺度（短边）为800像素[27]。每个mini-batch图像数量为2图像每GPU，每幅图像有N个取样的RoIs，正负比例为1:3[12]。对于C4骨干网络（和[12,36]中一样），N为64，对于FPN（和[27]中一样），N为512。我们在8个GPU上进行训练，（所以有效mini-batch大小为16），共160k次迭代，学习速率为0.02，在第10k和120k次迭代时减少。权重衰减为0.0001，动量0.9。使用ResNeXt[45]时，我们用每GPU 1幅图像的设置，迭代次数一样，初始学习速率为0.01。

The RPN anchors span 5 scales and 3 aspect ratios, following [27]. For convenient ablation, RPN is trained separately and does not share features with Mask R-CNN, unless specified. For every entry in this paper, RPN and Mask R-CNN have the same backbones and so they are shareable.

RPN锚框有5个尺度，3个宽高比，与[27]中一样。为进行方便的分离试验，RPN分别进行训练，没有与Mask R-CNN共享特征，除非特别说明。对于这边论文中的每种方法，RPN和Mask R-CNN有相同的骨干网络，所以可以共享。

**Inference**: At test time, the proposal number is 300 for the C4 backbone (as in [36]) and 1000 for FPN (as in [27]). We run the box prediction branch on these proposals, followed by non-maximum suppression [14]. The mask branch is then applied to the highest scoring 100 detection boxes. Although this differs from the parallel computation used in training, it speeds up inference and improves accuracy (due to the use of fewer, more accurate RoIs). The mask branch can predict K masks per RoI, but we only use the k-th mask, where k is the predicted class by the classification branch. The m×m floating-number mask output is then resized to the RoI size, and binarized at a threshold of 0.5.

**推理**：在测试时，C4骨干网络的建议数量为300（和[36]中一样），FPN的为1000（和[27]中一样）。我们在这些建议上运行边界框预测分支，然后是非极大抑制[14]；然后在最高得分的100个检测边界框中应用掩膜分支。虽然这与训练时的并行结构有所不同，但加速了推理，提高了准确率（因为使用了更少更准确的RoIs）。掩膜分支可以在每个RoI中预测K个掩膜，但我们只使用第k个掩膜，k分类分支预测的类别。大小为m×m的浮点掩膜输出然后改变成RoI的大小，然后以阈值0.5进行二值化。

Note that since we only compute masks on the top 100 detection boxes, Mask R-CNN adds a small overhead to its Faster R-CNN counterpart (e.g., ∼ 20% on typical models). 注意，由于我们只计算了得分最高的100个检测框的掩膜，Mask R-CNN相比Faster R-CNN增加的计算量是很小的（如在典型模型中大概是20%）。

## 4. Experiments: Instance Segmentation 试验：实例分割

We perform a thorough comparison of Mask R-CNN to the state of the art along with comprehensive ablations on the COCO dataset [28]. We report the standard COCO metrics including AP (averaged over IoU thresholds), $AP_{50}$, $AP_{75}$, and $AP_S$, $AP_M$, $AP_L$ (AP at different scales). Unless noted, AP is evaluating using mask IoU. As in previous work [5, 27], we train using the union of 80k train images and a 35k subset of val images (trainval35k), and report ablations on the remaining 5k val images (minival). We also report results on test-dev [28].

我们将Mask R-CNN与目前最好的模型在COCO数据集[28]上进行彻底的比较，并进行详尽的分离试验。我们在标准的COCO度量标准上汇报结果，包括AP（在各个IoU阈值上进行了平均），$AP_{50}$, $AP_{75}$, and $AP_S$, $AP_M$, $AP_L$（在不同尺度上的AP）。除非另有说明，AP都是使用掩膜IoU进行评估的。与之前的工作[5,27]一样，我们使用COCO的80k训练图像和验证集的35k子集的并集(trainval35k)进行训练，在验证集上的剩下的5k图像上(minival)进行分离试验。我们也在test-dev[28]给出结果。

### 4.1. Main Results

We compare Mask R-CNN to the state-of-the-art methods in instance segmentation in Table 1. All instantiations of our model outperform baseline variants of previous state-of-the-art models. This includes MNC [10] and FCIS [26], the winners of the COCO 2015 and 2016 segmentation challenges, respectively. Without bells and whistles, Mask R-CNN with ResNet-101-FPN backbone outperforms FCIS+++ [26], which includes multi-scale train/test, horizontal flip test, and online hard example mining (OHEM) [38]. While outside the scope of this work, we expect many such improvements to be applicable to ours.

我们在表1中将Mask R-CNN与目前最好的实例分割方法进行了比较，包括MNC[10]和FCIS[26]，分别是COCO 2015和2016年分割竞赛的获胜者。没有使用任何技巧，ResNet-101-FPN骨干网络的Mask R-CNN超过了FCIS+++[26]，而其包括了多尺度训练/测试，水平翻转测试，在线难分样本挖掘(OHEM)[38]。在此工作之外，可能会有很多这样的改进应用到我们的模型中。

Table 1. Instance segmentation mask AP on COCO test-dev. MNC [10] and FCIS [26] are the winners of the COCO 2015 and 2016 segmentation challenges, respectively. Without bells and whistles, Mask R-CNN outperforms the more complex FCIS+++, which includes multi-scale train/test, horizontal flip test, and OHEM [38]. All entries are single-model results.

| | backbone | AP | $AP_{50}$ | $AP_{75}$ | $AP_S$ | $AP_M$ | $AP_L$
--- | --- | --- | --- | --- | --- | --- | ---
MNC [10] | ResNet-101-C4 | 24.6 | 44.3 | 24.8 | 4.7 | 25.9 | 43.6
FCIS [26] +OHEM | ResNet-101-C5-dilated | 29.2 | 49.5 | - | 7.1 | 31.3 | 50.0
FCIS+++ [26] +OHEM | ResNet-101-C5-dilated | 33.6 | 54.5 | - | - | - | -
Mask R-CNN | ResNet-101-C4 | 33.1 | 54.9 | 34.8 | 12.1 | 35.6 | 51.1
Mask R-CNN | ResNet-101-FPN | 35.7 | 58.0 | 37.8 | 15.5 | 38.1 | 52.4
Mask R-CNN | ResNeXt-101-FPN | 37.1 | 60.0 | 39.4 | 16.9 | 39.9 | 53.5

Mask R-CNN outputs are visualized in Figures 2 and 5. Mask R-CNN achieves good results even under challenging conditions. In Figure 6 we compare our Mask R-CNN baseline and FCIS+++ [26]. FCIS+++ exhibits systematic artifacts on overlapping instances, suggesting that it is challenged by the fundamental difficulty of instance segmentation. Mask R-CNN shows no such artifacts.

Mask R-CNN的输出如图2和图5所示。Mask R-CNN即使在很有挑战的条件下也取得了很好的结果。在图6中，我们将Mask R-CNN与FCIS+++[26]进行了比较。FCIS+++在重叠的实例上表现出了系统性的错误，说明不能处理好实例分割的基本困难。Mask R-CNN则没有这种错误。

### 4.2. Ablation Experiments

We run a number of ablations to analyze Mask R-CNN. Results are shown in Table 2 and discussed in detail next. 我们进行了几个分离试验来分析Mask R-CNN。结果如表2所示，下面进行详述。

**Architecture**: Table 2a shows Mask R-CNN with various backbones. It benefits from deeper networks (50 vs. 101) and advanced designs including FPN and ResNeXt. We note that not all frameworks automatically benefit from deeper or advanced networks (see benchmarking in [21]).

**架构**：表2a给出了各种骨干网络的Mask R-CNN，网络越深(50vs101)越先进（包括FPN和ResNeXt），性能越好。我们要说明，并不是所有框架都自动从更深或更先进的网络中受益（见[21]中的基准测试）。

Table 2. Ablations. We train on trainval35k, test on minival, and report mask AP unless otherwise noted.

(a) Backbone Architecture: Better backbones bring expected gains: deeper networks do better, FPN outperforms C4 features, and ResNeXt improves on ResNet.

net-depth-features | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | ---
ResNet-50-C4 | 30.3 | 51.2 | 31.5
ResNet-101-C4 | 32.7 | 54.2 | 34.3
ResNet-50-FPN | 33.6 | 55.2 | 35.3
ResNet-101-FPN | 35.4 | 57.3 | 37.5
ResNeXt-101-FPN | 36.7 | 59.5 | 38.9

**Multinomial vs. Independent Masks**: Mask R-CNN decouples mask and class prediction: as the existing box branch predicts the class label, we generate a mask for each class without competition among classes (by a per-pixel sigmoid and a binary loss). In Table 2b, we compare this to using a per-pixel softmax and a multinomial loss (as commonly used in FCN [30]). This alternative couples the tasks of mask and class prediction, and results in a severe loss in mask AP (5.5 points). This suggests that once the instance has been classified as a whole (by the box branch), it is sufficient to predict a binary mask without concern for the categories, which makes the model easier to train.

**多项式vs.独立掩膜**：Mask R-CNN将掩膜与类别预测分离开来：由于现有的边界框分支预测类别标签，我们对每个类别生成一个掩膜，类别之间不存在竞争关系（通过一个每像素的sigmoid和二值损失）。在表2b中，我们将这种方法与使用每像素的sigmoid和多项式损失进行了比较（多项式损失在FCN[30]中常常使用）。这种选择将掩膜预测的任务和类别预测的任务结合到了一起，得到了掩膜AP严重降低(5.5%)。这说明，一旦实例作为整体被分类（通过边界框分支），预测二值掩膜时不用考虑类别就够了，这使得模型更容易训练。

(b) Multinomial vs. Independent Masks (ResNet-50-C4): Decoupling via per-class binary masks (sigmoid) gives large gains over multinomial masks (softmax).

| | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | ---
softmax | 24.8 | 44.1 | 25.1
sigmoid | 30.3 | 51.2 | 31.5
| | +5.5 | +7.1 | +6.4

**Class-Specific vs. Class-Agnostic Masks**: Our default instantiation predicts class-specific masks, i.e., one m×m mask per class. Interestingly, Mask R-CNN with class-agnostic masks (i.e., predicting a single m×m output regardless of class) is nearly as effective: it has 29.7 mask AP vs. 30.3 for the class-specific counterpart on ResNet-50-C4. This further highlights the division of labor in our approach which largely decouples classification and segmentation.

**类别相关vs类别无关的掩膜**：我们默认的实现预测的是类别相关的掩膜，即，每个类别都有一个m×m的掩膜。有意思的是，类别无关掩膜的Mask R-CNN（即只预测一个m×m的输出，与类别无关）也接近同样有效：类别无关的ResNet-50-C4结果为29.7掩膜AP，类别相关的为30.3掩膜AP。这进一步强调了我们方法中的分支分工，即分类与分割的分工。

**RoIAlign**: An evaluation of our proposed RoIAlign layer is shown in Table 2c. For this experiment we use the ResNet-50-C4 backbone, which has stride 16. RoIAlign improves AP by about 3 points over RoIPool, with much of the gain coming at high IoU ($AP_{75}$). RoIAlign is insensitive to max/average pool; we use average in the rest of the paper.

**RoIAlign**：我们提出的RoIAlign层的评估如表2c所示。对这个试验，我们使用ResNet-50-C4骨干网络，其步长为16。RoIAlign比RoIPool改进了3点AP，大部分改进来自于高IoU部分($AP_{75}$)。RoIAlign对最大/平均池化不敏感；在本文剩下部分，我们都使用平均池化。

Additionally, we compare with RoIWarp proposed in MNC [10] that also adopt bilinear sampling. As discussed in §3, RoIWarp still quantizes the RoI, losing alignment with the input. As can be seen in Table 2c, RoIWarp performs on par with RoIPool and much worse than RoIAlign.

另外，我们还与MNC[10]中的RoIWarp进行了比较，其使用了双线性采样。就像第3部分讨论的那样，RoIWarp仍然对RoI进行量化，丢失了与输入之间的对齐关系。如表2c所示，RoIWarp的表现与RoIPool类似，比RoIAlign差很多。

(c) RoIAlign (ResNet-50-C4): Mask results with various RoI layers. Our RoIAlign layer improves AP by ∼ 3 points and $AP_{75}$ by ∼ 5 points. Using proper alignment is the only factor that contributes to the large gap between RoI layers.

| | align? | bilinear? | agg. | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | --- | --- | --- | ---
RoIPool[12] | n | n | max | 26.9 | 48.8 | 26.4
RoIWarp[10] | n | y | max | 27.2 | 49.2 | 27.1
RoIWarp[10] | n | y | ave | 27.1 | 48.9 | 27.1
RoIAlign | y | y | max | 30.2 | 51.0 | 31.8
RoIAlign | y | y | ave | 30.3 | 51.2 | 31.5

This highlights that proper alignment is key. We also evaluate RoIAlign with a ResNet-50-C5 backbone, which has an even larger stride of 32 pixels. We use the same head as in Figure 4 (right), as the res5 head is not applicable. Table 2d shows that RoIAlign improves mask AP by a massive 7.3 points, and mask $AP_{75}$ by 10.5 points (50% relative improvement). Moreover, we note that with RoIAlign, using stride-32 C5 features (30.9 AP) is more accurate than using stride-16 C4 features (30.3 AP, Table 2c). RoIAlign largely resolves the long-standing challenge of using large-stride features for detection and segmentation.

这也进一步强调了，合理的对齐是关键因素。我们还在ResNet-50-C5骨干网络上评估了RoIAlign，其步长更大，达到了32像素。我们使用了与图4右同样的网络头，因为res5的头不能应用。表2d所示的是RoIAlign改进了掩膜AP很多，达到了7.3点，掩膜$AP_{75}$则改进了10.5点（50%相对改进）。而且，我们要注意，有了RoIAlign后，使用32步长的C5特征(30.9 AP)比使用16步长的C4特征更准确(30.3 AP, Table 2c)。RoIAlign基本解决了使用大步长特征进行检测和分割的长久挑战。

(d) RoIAlign (ResNet-50-C5, stride 32): Mask-level and box-level AP using large-stride features. Misalignments are more severe than with stride-16 features (Table 2c), resulting in big accuracy gaps.

| | AP | $AP_{50}$ | $AP_{75}$ | $AP^{bb}$ | $AP_{50}^{bb}$ | $AP_{75}^{bb}$
--- | --- | --- | --- | --- | --- | ---
RoIPool | 23.6 | 46.5 | 21.6 | 28.2 | 52.7 | 26.9
RoIAlign | 30.9 | 51.8 | 32.1 | 34.0 | 55.3 | 36.4
| | +7.3 | +5.3 | +10.5 | +5.8 | +2.6 | +9.5

Finally, RoIAlign shows a gain of 1.5 mask AP and 0.5 box AP when used with FPN, which has finer multi-level strides. For keypoint detection that requires finer alignment, RoIAlign shows large gains even with FPN (Table 6).

最后，RoIAlign在与FPN一起使用后掩膜AP提升了1.5，框AP提升了0.5，FPN的多层级步长更细致。对于关键点检测，需要更精细的对齐，RoIAlign与FPN一起也得到了很大的改进（表6）。

**Mask Branch**: Segmentation is a pixel-to-pixel task and we exploit the spatial layout of masks by using an FCN. In Table 2e, we compare multi-layer perceptrons (MLP) and FCNs, using a ResNet-50-FPN backbone. Using FCNs gives a 2.1 mask AP gain over MLPs. We note that we choose this backbone so that the conv layers of the FCN head are not pre-trained, for a fair comparison with MLP.

**掩膜分支**：分割是一个像素对像素的任务，我们利用FCN研究掩膜的空间布局。在表2e中，我们比较了MLP与FCN，使用的是ResNet-50-FPN骨干网络。使用FCN比使用MLP的掩膜AP高2.1。我们要说明的是，我们选择这个骨干网络的原因是，FCN的卷积层不是预训练好的，可以与MLP进行公平比较。

(e) Mask Branch (ResNet-50-FPN): Fully convolutional networks (FCN) vs. multi-layer perceptrons (MLP, fully-connected) for mask prediction. FCNs improve results as they take advantage of explicitly encoding spatial layout.

| |mask branch | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | --- | ---
MLP | fc: 1024→1024→80·28^2 | 31.5 | 53.7 | 32.8
MLP | fc: 1024→1024→1024→80·28^2 | 31.5 | 54.0 | 32.6
FCN | conv: 256→256→256→256→256→80 | 33.6 | 55.2 | 35.3

### 4.3. Bounding Box Detection Results 边界框检测结果

We compare Mask R-CNN to the state-of-the-art COCO bounding-box object detection in Table 3. For this result, even though the full Mask R-CNN model is trained, only the classification and box outputs are used at inference (the mask output is ignored). Mask R-CNN using ResNet-101-FPN outperforms the base variants of all previous state-of-the-art models, including the single-model variant of GRMI [21], the winner of the COCO 2016 Detection Challenge. Using ResNeXt-101-FPN, Mask R-CNN further improves results, with a margin of 3.0 points box AP over the best previous single model entry from [39] (which used Inception-ResNet-v2-TDM).

我们将Mask R-CNN与目前最好的边界框目标检测结果进行了比较，如表3所示。在这个结果中，即使训练了完整的Mask R-CNN模型，在推理时也只使用分类和框输出（忽略掉掩膜输出）。使用ResNet-101-FPN的Mask R-CNN超过了之前所有最好的模型的基础变体，包括GRMI[21]的单模型变体，这是COCO 2016检测挑战的获胜者。使用ResNeXt-101-FPN的话，Mask R-CNN可以进一步改进结果，比之前最好的单模型方法[39]（使用的是Inception-ResNet-v2-TDM）的框AP提高了3.0点。

As a further comparison, we trained a version of Mask R-CNN but without the mask branch, denoted by “Faster R-CNN, RoIAlign” in Table 3. This model performs better than the model presented in [27] due to RoIAlign. On the other hand, it is 0.9 points box AP lower than Mask R-CNN. This gap of Mask R-CNN on box detection is therefore due solely to the benefits of multi-task training.

我们还训练了一个没有掩膜分支版本的Mask R-CNN，作为进一步的比较，表示为表3中的“Faster R-CNN, RoIAlign”。这个模型由于使用了RoIAlign，比[27]中的模型效果更好。而它比Mask R-CNN低了0.9的框AP。与Mask R-CNN在框检测上的这个差距，仅仅是因为多任务训练。

Lastly, we note that Mask R-CNN attains a small gap between its mask and box AP: e.g., 2.7 points between 37.1 (mask, Table 1) and 39.8 (box, Table 3). This indicates that our approach largely closes the gap between object detection and the more challenging instance segmentation task.

最后，我们要说明，Mask R-CNN在掩膜AP和框AP之间差距很小：如，表1中的掩膜AP (37.1)与表3中的框AP (39.8)仅差了2.7点。这说明，我们的方法缩小了目标检测和更有挑战性的实例分割任务的差距。

Table 3. Object detection single-model results (bounding box AP), vs. state-of-the-art on test-dev. Mask R-CNN using ResNet-101-FPN outperforms the base variants of all previous state-of-the-art models (the mask output is ignored in these experiments). The gains of Mask R-CNN over [27] come from using RoIAlign (+1.1 $AP^{bb}$), multitask training (+0.9 $AP^{bb}$), and ResNeXt-101 (+1.6 $AP^{bb}$).

| | backbone | $AP^{bb}$ | $AP_{50}^{bb}$ | $AP_{75}^{bb}$ | $AP_S^{bb}$ | $AP_M^{bb}$ | $AP_L^{bb}$
--- | --- | --- | --- | --- | --- | --- | --- | ---
Faster R-CNN+++ [19] | ResNet-101-C4 | 34.9 | 55.7 | 37.4 | 15.6 | 38.7 | 50.9
Faster R-CNN w FPN [27] | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2
Faster R-CNN by G-RMI [21] | Inception-ResNet-v2[41] | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0
Faster R-CNN w TDM [39] | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1
Faster R-CNN, RoIAlign | ResNet-101-FPN | 37.3 | 59.6 | 40.3 | 19.8 | 40.2 | 48.8
Mask R-CNN | ResNet-101-FPN | 38.2 | 60.3 | 41.7 | 20.1 | 41.1 | 50.2
Mask R-CNN | ResNeXt-101-FPN | 39.8 | 62.3 | 43.4 | 22.1 | 43.2 | 51.2

### 4.4. Timing

**Inference**: We train a ResNet-101-FPN model that shares features between the RPN and Mask R-CNN stages, following the 4-step training of Faster R-CNN [36]. This model runs at 195ms per image on an Nvidia Tesla M40 GPU (plus 15ms CPU time resizing the outputs to the original resolution), and achieves statistically the same mask AP as the unshared one. We also report that the ResNet-101-C4 variant takes ∼ 400ms as it has a heavier box head (Figure 4), so we do not recommend using the C4 variant in practice.

**推理**：我们训练一个ResNet-101-FPN模型，在RPN与Mask R-CNN阶段共享特征，沿用Faster R-CNN[36]的四步训练法。这个模型在NVidia Tesla M40 GPU上每幅图像推理时间195ms（加上将输出改变到原始分辨率的15ms CPU运行时间），得到了掩膜AP与不共享特征的统计上一样。我们实现了ResNet-101-C4的变体，耗时大约400ms，因为其框的网络头比较大（图4），所以我们不推荐在实际中使用C4变体。

Although Mask R-CNN is fast, we note that our design is not optimized for speed, and better speed/accuracy tradeoffs could be achieved [21], e.g., by varying image sizes and proposal numbers, which is beyond the scope of this paper.

虽然Mask R-CNN速度很快，我们注意到，我们的设计并没有为速度优化，所以可以得到更好的速度/准确率折中[21]，如，使用不同的图像大小和建议数量，这不在本文讨论范围之内。

**Training**: Mask R-CNN is also fast to train. Training with ResNet-50-FPN on COCO trainval35k takes 32 hours in our synchronized 8-GPU implementation (0.72s per 16-image mini-batch), and 44 hours with ResNet-101-FPN. In fact, fast prototyping can be completed in less than one day when training on the train set. We hope such rapid training will remove a major hurdle in this area and encourage more people to perform research on this challenging topic.

**训练**：Mask R-CNN训练过程也很快。在COCO trainval35k上训练ResNet-50-FPN需要32小时，使用的是8-GPU同步实现（mini-batch为16图像，耗时0.72秒），训练ResNet-101-FPN耗时44小时。实际上，如果在训练集上进行训练，快速原型可以在一天之内得到。我们希望这样快速的训练会去掉这个领域中的一个主要障碍，鼓励更多人研究这个有挑战的课题。

## 5. Mask R-CNN for Human Pose Estimation

Our framework can easily be extended to human pose estimation. We model a keypoint’s location as a one-hot mask, and adopt Mask R-CNN to predict K masks, one for each of K keypoint types (e.g., left shoulder, right elbow). This task helps demonstrate the flexibility of Mask R-CNN.

我们的框架可以轻松的拓展到人体姿态估计。我们将关键点的位置编成独热码，使用Mask R-CNN来预测K个掩膜，每个对应K个关键点类型的一个（如左肩膀，右肘）。这个任务证明了Mask R-CNN的灵活性。

We note that minimal domain knowledge for human pose is exploited by our system, as the experiments are mainly to demonstrate the generality of the Mask R-CNN framework. We expect that domain knowledge (e.g., modeling structures [6]) will be complementary to our simple approach.

我们注意到，我们的系统使用的人体姿态的领域知识是极少的，因为试验主要是证明Mask R-CNN框架的通用性。我们期待领域知识（如建模结构[6]）会成为我们简单方法的补充。

**Implementation Details**: We make minor modifications to the segmentation system when adapting it for keypoints. For each of the K keypoints of an instance, the training target is a one-hot m × m binary mask where only a single pixel is labeled as foreground. During training, for each visible ground-truth keypoint, we minimize the cross-entropy loss over an $m^2$-way softmax output (which encourages a single point to be detected). We note that as in instance segmentation, the K keypoints are still treated independently.

**实现细节**：在关键点预测上应用时，我们对分割系统做了很小的修改。对于一个实例的K个关键点的每一个，训练目标是一个独热的m×m二值掩膜，其中只有一个像素标记为前景。在训练时，对于每个可见的真值关键点，我们在$m^2$路softmax输出上最小化其交叉熵损失（这鼓励只检测到一个点）。我们注意到，就像在实例分割中一样，K个关键点也都是独立对待的。

We adopt the ResNet-FPN variant, and the keypoint head architecture is similar to that in Figure 4 (right). The keypoint head consists of a stack of eight 3×3 512-d conv layers, followed by a deconv layer and 2× bilinear upscaling, producing an output resolution of 56×56. We found that a relatively high resolution output (compared to masks) is required for keypoint-level localization accuracy.

我们使用的是ResNet-FPN变体，关键点头部网络结构与图4右类似。关键点头部网络包括8个3×3 512-d卷积层的堆叠，随后是一个解卷积层，和2x双线性上采样层，生成一个分辨率56×56的输出。我们发现，对于关键点层次的定位，需要一个相对较高分辨率的输出（与掩膜相比）才能得到很好的定位准确率。

Models are trained on all COCO trainval35k images that contain annotated keypoints. To reduce overfitting, as this training set is smaller, we train using image scales randomly sampled from [640, 800] pixels; inference is on a single scale of 800 pixels. We train for 90k iterations, starting from a learning rate of 0.02 and reducing it by 10 at 60k and 80k iterations. We use bounding-box NMS with a threshold of 0.5. Other details are identical as in §3.1.

模型在COCO trainval35k集中包含标注关键点的图像上进行训练。为降低过拟合，由于训练集更小，我们使用尺度为[640,800]像素的随机采样的图像进行训练；推理则是在800像素的单尺度上。我们训练90k次迭代，从初始学习速率0.02开始，在60k和80k次迭代时除以10。我们使用边界框NMS的阈值为0.5。其他细节与3.1节描述的一样。

**Main Results and Ablations**: We evaluate the person keypoint AP ($AP^{kp}$) and experiment with a ResNet-50-FPN backbone; more backbones will be studied in the appendix. Table 4 shows that our result (62.7 $AP^{kp}$) is 0.9 points higher than the COCO 2016 keypoint detection winner [6] that uses a multi-stage processing pipeline (see caption of Table 4). Our method is considerably simpler and faster.

**主要结果和分离试验**：我们使用ResNet-50-FPN骨干网络进行试验并评估人体关键点AP ($AP^{kp}$)；附录中研究了更多的骨干网络。表4给出了我们的结果(62.7 $AP^{kp}$)，比COCO 2016关键点检测的获胜者[6]高了0.9点，他们使用了一个多阶段处理的流程（见表4的标题）。我们的方法更简单，更快速。

Table 4. Keypoint detection AP on COCO test-dev. Ours is a single model (ResNet-50-FPN) that runs at 5 fps. CMU-Pose+++ [6] is the 2016 competition winner that uses multi-scale testing, post-processing with CPM [44], and filtering with an object detector, adding a cumulative ∼ 5 points (clarified in personal communication). † : G-RMI was trained on COCO plus MPII [1] (25k images), using two models (Inception-ResNet-v2 for bounding box detection and ResNet-101 for keypoints).

| | $AP^{kp}$ | $AP^{kp}_{50}$ | $AP^{kp}_{75}$ | $AP^{kp}_M$ | $AP^{kp}_L$
--- | --- | --- | --- | --- | ---
CMU-Pose+++ [6] | 61.8 | 84.9 | 67.5 | 57.1 | 68.2
G-RMI [32] † | 62.4 | 84.0 | 68.5 | 59.1 | 68.1
Mask R-CNN, keypoint-only | 62.7 | 87.0 | 68.4 | 57.4 | 71.1
Mask R-CNN, keypoint & mask | 63.1 | 87.3 | 68.7 | 57.8 | 71.4

More importantly, we have a unified model that can simultaneously predict boxes, segments, and keypoints while running at 5 fps. Adding a segment branch (for the person category) improves the $AP^{kp}$ to 63.1 (Table 4) on test-dev. More ablations of multi-task learning on minival are in Table 5. Adding the mask branch to the box-only (i.e., Faster R-CNN) or keypoint-only versions consistently improves these tasks. However, adding the keypoint branch reduces the box/mask AP slightly, suggesting that while keypoint detection benefits from multitask training, it does not in turn help the other tasks. Nevertheless, learning all three tasks jointly enables a unified system to efficiently predict all outputs simultaneously (Figure 7).

更重要的是，我们有一个统一的模型，可以同时预测边界框，预测分割，预测关键点，同时以5 fps的速度运行。增加一个分割分支（对于人的类别）将$AP^{kp}$改进到63.1（表4）。在minival上更多的多任务分离试验如表5所示。将掩膜分支增加到框上（如Faster R-CNN）或关键点上都会改进这些任务。但是，增加关键点分支会略微降低框/掩膜的AP，说明关键点检测可以从多任务训练中受益，但并没有对其他任务有贡献。但是，同时学习所有三个任务可以用一个统一系统高效的同时预测所有输出（图7）。

We also investigate the effect of RoIAlign on keypoint detection (Table 6). Though this ResNet-50-FPN backbone has finer strides (e.g., 4 pixels on the finest level), RoIAlign still shows significant improvement over RoIPool and increases $AP^{kp}$ by 4.4 points. This is because keypoint detections are more sensitive to localization accuracy. This again indicates that alignment is essential for pixel-level localization, including masks and keypoints.

我们还研究了RoIAlign在关键点检测中的作用（表6）。虽然ResNet-50-FPN骨干网络步长较小（如，在最精细的层只有4个像素），RoIAlign仍然表现出了显著的改进，将$AP^{kp}$提高了4.4点。这是因为，关键点检测对定位准确率更敏感。这再次说明了，对齐是像素级定位的关键因素，包括掩膜和关键点。

Given the effectiveness of Mask R-CNN for extracting object bounding boxes, masks, and keypoints, we expect it be an effective framework for other instance-level tasks. Mask R-CNN在提取目标边界框、掩膜和关键点上都非常高效，我们期待其会成为其他实例级任务的有效框架。

Table 5. Multi-task learning of box, mask, and keypoint about the person category, evaluated on minival. All entries are trained on the same data for fair comparisons. The backbone is ResNet-50-FPN. The entries with 64.2 and 64.7 AP on minival have test-dev AP of 62.7 and 63.1, respectively (see Table 4).

| | $AP^{bb}_{person}$ | $AP^{mask}_{person}$ | $AP^{kp}$
--- | --- | --- | ---
Faster R-CNN | 52.5 | - | -
Mask R-CNN, mask-only | 53.6 | 45.8 | -
Mask R-CNN, keypoint-only | 50.7 | - | 64.2
Mask R-CNN, keypoint & mask | 52.0 | 45.1 | 64.7

Table 6. RoIAlign vs. RoIPool for keypoint detection on minival. The backbone is ResNet-50-FPN.

| | $AP^{kp}$ | $AP_{50}^{kp}$ | $AP_{75}^{kp}$ | $AP^{kp}_M$ | $AP^{kp}_L$
--- | --- | --- | --- | --- | ---
RoIPool | 59.8 | 86.2 | 66.7 | 55.1 | 67.4
RoIAlign | 64.2 | 86.6 | 69.7 | 58.7 | 73.0