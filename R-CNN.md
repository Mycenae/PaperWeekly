# Rich feature hierarchies for accurate object detection and semantic segmentation 利用层次性富特征进行准确目标检测及语义分割

Ross Girshick et al. UC Berkeley

## Abstract 摘要

Object detection performance, as measured on the canonical PASCAL VOC dataset, has plateaued in the last few years. The best-performing methods are complex ensemble systems that typically combine multiple low-level image features with high-level context. In this paper, we propose a simple and scalable detection algorithm that improves mean average precision (mAP) by more than 30% relative to the previous best result on VOC 2012—achieving a mAP of 53.3%. Our approach combines two key insights: (1) one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects and (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. Since we combine region proposals with CNNs, we call our method R-CNN: Regions with CNN features. We also compare R-CNN to OverFeat, a recently proposed sliding-window detector based on a similar CNN architecture. We find that R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset. Source code for the complete system is available at http://www.cs.berkeley.edu/˜rbg/rcnn.

目标检测的性能一般由权威的PASCAL VOC数据集衡量，在过去几年中一直都没有很大进展。最好的方法一般都是将多个底层图像特征与高层上下文进行组合形成的复杂集成系统。在本文中，我们提出一种简单并可扩展的检测算法，与之前在VOC 2012上的最好结果相比，我们的mAP提升了30%以上，达到了53.3%。我们的方法结合了两种关键思想：(1)将高性能的卷积神经网络CNNs应用于自下而上的候选区域中，以定位并分割目标；(2)当有标签的训练数据不足时，先对辅助任务进行有监督的预训练，随后进行领域相关的精调，可以得到性能的显著提升。由于我们将候选区域与CNN结合，我们称此方法为R-CNN：带有CNN特征的区域。我们还将R-CNN与OverFeat方法进行比较，OverFeat是近期提出的一种基于类似的CNN架构的滑窗检测器。我们发现在200类的ILSVRC2013检测数据集上，R-CNN远超OverFeat的性能。全部系统的源码见http://www.cs.berkeley.edu/˜rbg/rcnn.

## 1. Introduction 引言

Features matter. The last decade of progress on various visual recognition tasks has been based considerably on the use of SIFT [29] and HOG [7]. But if we look at performance on the canonical visual recognition task, PASCAL VOC object detection [15], it is generally acknowledged that progress has been slow during 2010-2012, with small gains obtained by building ensemble systems and employing minor variants of successful methods.

特征很重要。过去十年中多种视觉识别任务大多是基于SIFT和HOG特征的。但如果我们看看在权威视觉识别任务，即PASCAL VOC目标检测，中的表现，大家普遍承认在2010-2012年的进步是缓慢的，基本都是构建集成系统和成功方法的微小改变得到的进展。

SIFT and HOG are blockwise orientation histograms, a representation we could associate roughly with complex cells in V1, the first cortical area in the primate visual pathway. But we also know that recognition occurs several stages downstream, which suggests that there might be hierarchical, multi-stage processes for computing features that are even more informative for visual recognition.

SIFT和HOG是分块的方向直方图，这种表示可以大致与V1区域的复杂细胞联系起来，V1区域即在原始视觉通道的第一皮质区域。但我们还知道，识别的动作发生在几个阶段依次进行，这意味着对视觉识别更有信息量的特征计算可能是层次式的、多阶段的过程。

Fukushima’s “neocognitron” [19], a biologically-inspired hierarchical and shift-invariant model for pattern recognition, was an early attempt at just such a process. The neocognitron, however, lacked a supervised training algorithm. Building on Rumelhart et al. [33], LeCun et al. [26] showed that stochastic gradient descent via back-propagation was effective for training convolutional neural networks (CNNs), a class of models that extend the neocognitron.

Fukushima的“神经感知机”[19]是一种受生物学启发的层次性平移不变的模式识别模型，是这种过程的早期尝试。但神经感知机缺少有监督训练算法。在Rumelhart et al. [33]的基础上，LeCun et al. [26]展示了通过反向传播的随机梯度下降对于训练CNNs非常有效，CNNs是神经感知机的延伸模型。

CNNs saw heavy use in the 1990s (e.g., [27]), but then fell out of fashion with the rise of support vector machines. In 2012, Krizhevsky et al. [25] rekindled interest in CNNs by showing substantially higher image classification accuracy on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) [9, 10]. Their success resulted from training a large CNN on 1.2 million labeled images, together with a few twists on LeCun’s CNN (e.g., max(x,0) rectifying non-linearities and “dropout” regularization).

在1990s CNNs得到大量应用，如[27]，但很快被支持矢量机的风头盖过。在2012年，Krizhevsky et al. [25]在ILSVRC-2012上得到了图像分类准确率的极大提升[9,10]，从而重新点燃了CNNs的兴趣。他们的成功是在120万有标签的图像上训练大型CNN得到的，另外还在LeCun的CNN上进行了一些小的改进，即使用了ReLU激活函数和dropout正则化。

The significance of the ImageNet result was vigorously debated during the ILSVRC 2012 workshop. The central issue can be distilled to the following: To what extent do the CNN classification results on ImageNet generalize to object detection results on the PASCAL VOC Challenge?

ImageNet结果的重要性在ILSVRC 2012 workshop上受到了热烈的辩论。提炼出的中心问题如下：CNN在ImageNet上的分类结果能在多大程度上泛化到PASCAL VOC挑战的目标检测结果上？

We answer this question by bridging the gap between image classification and object detection. This paper is the first to show that a CNN can lead to dramatically higher object detection performance on PASCAL VOC as compared to systems based on simpler HOG-like features. To achieve this result, we focused on two problems: localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data.

我们在图像分类与目标检测之间架起了桥梁，从而回答了这个问题。本文第一个展示了CNN可以带来PASCAL VOC上目标检测性能的极大提高，这是与HOG类的相对简单特征构成的系统相比。为得到这个结果，我们聚焦在两个问题上：用深度网络定位目标，和仅用较少的标记检测数据来训练大容量模型。

Unlike image classification, detection requires localizing (likely many) objects within an image. One approach frames localization as a regression problem. However, work from Szegedy et al. [38], concurrent with our own, indicates that this strategy may not fare well in practice (they report a mAP of 30.5% on VOC 2007 compared to the 58.5% achieved by our method). An alternative is to build a sliding-window detector. CNNs have been used in this way for at least two decades, typically on constrained object categories, such as faces [32, 40] and pedestrians [35]. In order to maintain high spatial resolution, these CNNs typically only have two convolutional and pooling layers. We also considered adopting a sliding-window approach. However, units high up in our network, which has five convolutional layers, have very large receptive fields (195 × 195 pixels) and strides (32×32 pixels) in the input image, which makes precise localization within the sliding-window paradigm an open technical challenge.

与图像分类不同，检测需要在图像中定位目标（很可能是很多个目标）。有一种方法将定位视为一个回归问题。但Szegedy et al. [38]的工作表明这种策略在实际情况中不一定有很好的表现，这与我们的工作是同时的（他们在VOC 2007上得到的mAP为30.5%，而我们的结果为58.5%）。另一种方法是构建一个滑窗检测器。这种方法使用CNNs已经有至少20年了，典型的是在特定的目标类别中，如人脸检测[32,40]和行人检测[35]。为保持高空间分辨率，这些CNN一般只有两个卷积层和pooling层。我们也采用滑窗的方法。但是，我们的网络中节点非常多，有5个卷积层，输入图像感受野非常大(195×195像素)，步长也很大(32×32像素)，这使滑窗方案的精确定位成为尚未解决的技术挑战。

Instead, we solve the CNN localization problem by operating within the “recognition using regions” paradigm [21], which has been successful for both object detection [39] and semantic segmentation [5]. At test time, our method generates around 2000 category-independent region proposals for the input image, extracts a fixed-length feature vector from each proposal using a CNN, and then classifies each region with category-specific linear SVMs. We use a simple technique (affine image warping) to compute a fixed-size CNN input from each region proposal, regardless of the region’s shape. Figure 1 presents an overview of our method and highlights some of our results. Since our system combines region proposals with CNNs, we dub the method R-CNN: Regions with CNN features.

我们则在“使用区域的识别”方案[21]中解决了CNN的定位问题，这对于目标检测[39]和语义分割[5]都很成功。在测试时，我们的方法对输入图像产生了大约2000个类别无关的候选区域，用CNN对每个候选区域都提取出一个定长的特征向量，然后用特定类别的线性SVMs对每个区域进行分类。无论候选区域形状如何，我们使用了一种简单技术（仿射图像变形）来从每个候选区域中计算固定大小的CNN输入。图1是我们的方法的概览，并突出强调了我们得到的一些结果。因为我们的模型将候选区域与CNN结合起来，我们称此方法为R-CNN: CNN特征的区域。

Figure 1: Object detection system overview. Our system (1) takes an input image, (2) extracts around 2000 bottom-up region proposals, (3) computes features for each proposal using a large convolutional neural network (CNN), and then (4) classifies each region using class-specific linear SVMs. R-CNN achieves a mean average precision (mAP) of 53.7% on PASCAL VOC 2010. For comparison, [39] reports 35.1% mAP using the same region proposals, but with a spatial pyramid and bag-of-visual-words approach. The popular deformable part models perform at 33.4%. On the 200-class ILSVRC2013 detection dataset, R-CNN’s mAP is 31.4%, a large improvement over OverFeat [34], which had the previous best result at 24.3%.

图1：目标检测模型概览。我们的模型主要分以下几个步骤，(1)接收输入图像，(2)提取出约2000个自下而上的候选区域，(3)对每个候选区域，用大型CNN计算特征，(4)用类别特定的线性SVMs对每个区域进行分类。R-CNN在PASCAL VOC 2010上得到了53.7% mAP，作为比较，[39]用同样的候选区域法，但用了空域金字塔和bag-of-visual-words方法，得到了35.1% mAP，流行的DPM模型得到33.4% mAP。在含有200类的ILSVRC2013检测数据集上，R-CNN得到了31.4% mAP，与之前得到最好结果的OverFeat[34]方法(24.3% mAP)相比改进很多。

In this updated version of this paper, we provide a head-to-head comparison of R-CNN and the recently proposed OverFeat [34] detection system by running R-CNN on the 200-class ILSVRC2013 detection dataset. OverFeat uses a sliding-window CNN for detection and until now was the best performing method on ILSVRC2013 detection. We show that R-CNN significantly outperforms OverFeat, with a mAP of 31.4% versus 24.3%.

在本文的此次更新中，我们给出了R-CNN与最近提出的OverFeat[34]检测模型的直接对比，即在200类的ILSVRC2013检测数据集上运行R-CNN模型。OverFeat使用了滑窗CNN进行检测，之前是ILSVRC2013检测任务的最佳表现模型。我们的结果显示，R-CNN明显优于OverFeat，mAP的对比为31.4%对24.3%。

A second challenge faced in detection is that labeled data is scarce and the amount currently available is insufficient for training a large CNN. The conventional solution to this problem is to use unsupervised pre-training, followed by supervised fine-tuning (e.g., [35]). The second principle contribution of this paper is to show that supervised pre-training on a large auxiliary dataset (ILSVRC), followed by domain-specific fine-tuning on a small dataset (PASCAL), is an effective paradigm for learning high-capacity CNNs when data is scarce. In our experiments, fine-tuning for detection improves mAP performance by 8 percentage points. After fine-tuning, our system achieves a mAP of 54% on VOC 2010 compared to 33% for the highly-tuned, HOG-based deformable part model (DPM) [17, 20]. We also point readers to contemporaneous work by Donahue et al. [12], who show that Krizhevsky’s CNN can be used (without fine-tuning) as a blackbox feature extractor, yielding excellent performance on several recognition tasks including scene classification, fine-grained sub-categorization, and domain adaptation.

检测中面临的第二个挑战是有标签的数据是很稀少的，现有可用的数量是不足以训练一个大型CNN的。解决这个问题的传统方法是先使用无监督预训练，然后进行有监督的精调，如[35]。本文的第二个主要贡献是得出了如下结论，在数据缺少时，首先在大型辅助数据集(ILSVRC)上进行有监督的预训练，然后在一个小数据集(PASCAL)上进行领域相关的精调，是一种有效的方法，可以学习得到大容量CNNs。在我们的试验中，对检测任务的精调改进了mAP大约8个百分点。精调之后，我们的系统在VOC 2010上得到的mAP为54%，与之相比基于HOG特征的高度调谐DPM模型得到的结果为33% mAP。我们还向读者指出，同期Donahue et al. [12]的工作，说明Krizhevsky的CNN（未精调）可以用作黑盒特征提取器，在几个识别任务，包括场景分类、细粒度子分类和领域适应，都得到了极好的表现。

Our system is also quite efficient. The only class-specific computations are a reasonably small matrix-vector product and greedy non-maximum suppression. This computational property follows from features that are shared across all categories and that are also two orders of magnitude lower-dimensional than previously used region features (cf. [39]).

我们的模型也很高效。唯一的与类别有关的计算是一个相对较小的矩阵-矢量乘积，和贪婪非最大抑制。这种计算遵从所有类别共享特征的性质，与之前使用的区域特征相比，计算量低了两个等级[39]。

Understanding the failure modes of our approach is also critical for improving it, and so we report results from the detection analysis tool of Hoiem et al. [23]. As an immediate consequence of this analysis, we demonstrate that a simple bounding-box regression method significantly reduces mislocalizations, which are the dominant error mode.

理解我们方法中的失败模式对改进是很重要的，所以我们用Hoiem et al. [23]的检测分析工具给出了结果。这个分析立刻得到了一个结果，我们证明了简单的边界框回归就可以明显减少错误定位，而这是主要的错误模式。

Before developing technical details, we note that because R-CNN operates on regions it is natural to extend it to the task of semantic segmentation. With minor modifications, we also achieve competitive results on the PASCAL VOC segmentation task, with an average segmentation accuracy of 47.9% on the VOC 2011 test set.

在进入技术细节之前，我们要注意，由于R-CNN是作用于区域上的，将其延伸至语义分割的任务是很自然的。只需要很少修改，我们就可以在PASCAL VOC分割任务中得到很好的结果，在VOC 2011测试集上的平均分割准确率为47.9%。

## 2. Object detection with R-CNN 采用R-CNN的目标检测

Our object detection system consists of three modules. The first generates category-independent region proposals. These proposals define the set of candidate detections available to our detector. The second module is a large convolutional neural network that extracts a fixed-length feature vector from each region. The third module is a set of classspecific linear SVMs. In this section, we present our design decisions for each module, describe their test-time usage, detail how their parameters are learned, and show detection results on PASCAL VOC 2010-12 and on ILSVRC2013.

我们的目标检测模型包括三个模块。第一个生成类别无关的候选区域，这些候选区域定义了我们检测器的候选被检测集。第二个模块是一个大型CNN，从每个区域中提取出固定长度的特征向量。第三个模块是特定类别的线性SVMs集。在这部分中，我们给出每个模块的设计决策，描述它们测试时的用处，详述参数如何学习，并展示了在PASCAL VOC 2010-12和ILSVRC2013的检测结果。

### 2.1. Module design 模块设计

**Region proposals**. A variety of recent papers offer methods for generating category-independent region proposals. Examples include: objectness [1], selective search [39], category-independent object proposals [14], constrained parametric min-cuts (CPMC) [5], multi-scale combinatorial grouping [3], and Cires an et al. [6], who detect mitotic cells by applying a CNN to regularly-spaced square crops, which are a special case of region proposals. While R-CNN is agnostic to the particular region proposal method, we use selective search to enable a controlled comparison with prior detection work (e.g., [39, 41]).

**候选区域**。近期有几篇文章都给出了生成类别无关的候选区域的方法。如，objectness [1], selective search [39], category-independent object proposals [14], constrained parametric min-cuts (CPMC) [5], multi-scale combinatorial grouping [3], and Cires an et al. [6], 他将CNN应用于规则摆放的方形组图来检测有丝分裂细胞，这实际上是候选区域的特例。R-CNN与特定的候选区域法无关，我们使用selective search法方便与之前的其他检测工作进行比较[39,41]。

**Feature extraction**. We extract a 4096-dimensional feature vector from each region proposal using the Caffe [24] implementation of the CNN described by Krizhevsky et al. [25]. Features are computed by forward propagating a mean-subtracted 227×227 RGB image through five convolutional layers and two fully connected layers. We refer readers to [24, 25] for more network architecture details.

**特征提取**。我们从每个区域中提取一个4096维的特征向量，使用的是Caffe[24]实现的Krizhevsky et al. [25]CNN模型。特征计算方法是将减去均值的227×227 RGB图像输入至5个卷积层和2个全连接层的网络。[24,25]包括更多的网络架构细节。

In order to compute features for a region proposal, we must first convert the image data in that region into a form that is compatible with the CNN (its architecture requires inputs of a fixed 227 × 227 pixel size). Of the many possible transformations of our arbitrary-shaped regions, we opt for the simplest. Regardless of the size or aspect ratio of the candidate region, we warp all pixels in a tight bounding box around it to the required size. Prior to warping, we dilate the tight bounding box so that at the warped size there are exactly p pixels of warped image context around the original box (we use p = 16). Figure 2 shows a random sampling of warped training regions. Alternatives to warping are discussed in Appendix A.

为计算一个候选区域的特征，我们首先需要将区域图像转换为与CNN兼容的形式（CNN的架构需要输入图像为227×227像素大小）。对我们的任意形状的区域，可以有很多种变换，我们选择了最简单的。不管候选区域的纵横比，我们将所有像素变形放入一个需求大小的边界框。变形之前，我们将边界框进行扩大，这样原边界框中的内容有p个像素的上下文，我们用p=16。图2所示的是任选的变形后训练区域。附录A讨论了变形的替代选项。

### 2.2. Test-time detection 测试时检测

At test time, we run selective search on the test image to extract around 2000 region proposals (we use selective search’s “fast mode” in all experiments). We warp each proposal and forward propagate it through the CNN in order to compute features. Then, for each class, we score each extracted feature vector using the SVM trained for that class. Given all scored regions in an image, we apply a greedy non-maximum suppression (for each class independently) that rejects a region if it has an intersection-over-union (IoU) overlap with a higher scoring selected region larger than a learned threshold.

在测试时，我们在测试图像上使用selective search方法来提取大约2000个候选区域（我们在所有试验中都使用selective search的快速模式）。我们将每个区域变形并送入CNN计算特征。然后，对于每个类别，我们都用那个类别训练出的SVM对提取出的特征进行计算得分。一幅图像中所有区域的得分都计算后，我们用贪婪非最大值抑制（对每个类分别应用），如果其与更高得分区域的IoU重叠超过一个学习得到的阈值，那就拒绝这个区域。

**Run-time analysis**. Two properties make detection efficient. First, all CNN parameters are shared across all categories. Second, the feature vectors computed by the CNN are low-dimensional when compared to other common approaches, such as spatial pyramids with bag-of-visual-word encodings. The features used in the UVA detection system [39], for example, are two orders of magnitude larger than ours (360k vs. 4k-dimensional).

**运行时分析**。检测高效是两个性质的原因。第一，所有类别都使用同样的CNN参数。第二，与其他常用方法相比，CNN计算得到的特征向量维数较低，如bag-of-visual-word编码的空间金字塔法。UVA检测模型[39]中使用的特征比我们高出两个数量级(360k vs. 4k)。

The result of such sharing is that the time spent computing region proposals and features (13s/image on a GPU or 53s/image on a CPU) is amortized over all classes. The only class-specific computations are dot products between features and SVM weights and non-maximum suppression. In practice, all dot products for an image are batched into a single matrix-matrix product. The feature matrix is typically 2000×4096 and the SVM weight matrix is 4096×N, where N is the number of classes.

这种权值共享的结果是，计算候选区域和特征的耗费时间（GPU上13秒/图像，CPU上53秒/图像）在所有类别中均摊了。唯一与类别有关的运算是特征与SVM权值间的点乘，和非最大值抑制。在实践中，一幅图像的所有点乘进行批次处理，只要计算一次矩阵-矩阵乘积。

This analysis shows that R-CNN can scale to thousands of object classes without resorting to approximate techniques, such as hashing. Even if there were 100k classes, the resulting matrix multiplication takes only 10 seconds on a modern multi-core CPU. This efficiency is not merely the result of using region proposals and shared features. The UVA system, due to its high-dimensional features, would be two orders of magnitude slower while requiring 134GB of memory just to store 100k linear predictors, compared to just 1.5GB for our lower-dimensional features.

这种分析说明，R-CNN可以扩展到数千种目标类别，而不用借助于哈希这样的近似技术。即使有100k个类别，得到的矩阵乘积在现代的多核CPU上也只需10秒左右。这种效率不只是使用了候选区域和共享权值的结果。UVA系统，由于其高维特征，会慢两个数量级，而且需要134G内存存储100k个线性预测期，我们的低维特征只需要1.5G。

It is also interesting to contrast R-CNN with the recent work from Dean et al. on scalable detection using DPMs and hashing [8]. They report a mAP of around 16% on VOC 2007 at a run-time of 5 minutes per image when introducing 10k distractor classes. With our approach, 10k detectors can run in about a minute on a CPU, and because no approximations are made mAP would remain at 59% (Section 3.2).

将R-CNN与最近Dean et al.在使用DPM和哈希的可扩展检测工作[8]相比较也很有趣。他们在VOC 2007上得到的mAP为约16%，当引入10k错误诱导类别时，每幅图像运行时间5分钟。而我们的方法，10k检测器在CPU上运行约1分钟，而由于没有近似，mAP应当还是59%（3.2节）。

### 2.3. Training 训练

**Supervised pre-training**. We discriminatively pre-trained the CNN on a large auxiliary dataset (ILSVRC2012 classification) using image-level annotations only (bounding-box labels are not available for this data). Pre-training was performed using the open source Caffe CNN library [24]. In brief, our CNN nearly matches the performance of Krizhevsky et al. [25], obtaining a top-1 error rate 2.2 percentage points higher on the ILSVRC2012 classification validation set. This discrepancy is due to simplifications in the training process.

**监督预训练**。我们在大型辅助数据集(ILSVRC2012分类数据集)上只使用图像级别的标注来预训练CNN（边界框标签在此数据上不可用）。预训练使用开源Caffe CNN库[24]来进行。简要来说，我们的CNN几乎与Krizhevsky et al. [25]性能相同，在ILSVRC2012分类验证数据集上的top-1错误率只高了2.2%。这个差异是由于训练过程的简化导致的。

**Domain-specific fine-tuning**. To adapt our CNN to the new task (detection) and the new domain (warped proposal windows), we continue stochastic gradient descent (SGD) training of the CNN parameters using only warped region proposals. Aside from replacing the CNN’s ImageNet-specific 1000-way classification layer with a randomly initialized (N + 1)-way classification layer (where N is the number of object classes, plus 1 for background), the CNN architecture is unchanged. For VOC, N = 20 and for ILSVRC2013, N = 200. We treat all region proposals with ≥ 0.5 IoU overlap with a ground-truth box as positives for that box’s class and the rest as negatives. We start SGD at a learning rate of 0.001 (1/10th of the initial pre-training rate), which allows fine-tuning to make progress while not clobbering the initialization. In each SGD iteration, we uniformly sample 32 positive windows (over all classes) and 96 background windows to construct a mini-batch of size 128. We bias the sampling towards positive windows because they are extremely rare compared to background.

**领域特定的精调**。为使我们的CNN适应新任务（检测）和新领域（变形候选窗口），我们只用变形候选区域继续随机梯度下降训练CNN参数。CNN的结构变化只在于，我们将ImageNet特定的1000路分类层，替换成了随机初始化的N+1路分类层，这里N是目标类别数，加1是为背景准备的。对于VOC，N=20，对于ILSVRC2013，N=200。对与真值框重叠度IoU ≥ 0.5的所有候选区域，我们都认为是这个类的正样本，剩下的就是负样本。我们的SGD学习速率为0.001（初始预训练率的1/10），这使精调得以进展，而且不会破坏初始化。在每个SGD循环中，我们统一取32个正样本窗口（在所有类别中）和96个背景窗口来构成一个大小128的mini-batch。我们的取样偏向于正样本窗口，因为与背景相比，正样本窗口很稀少。

**Object category classifiers**. Consider training a binary classifier to detect cars. It’s clear that an image region tightly enclosing a car should be a positive example. Similarly, it’s clear that a background region, which has nothing to do with cars, should be a negative example. Less clear is how to label a region that partially overlaps a car. We resolve this issue with an IoU overlap threshold, below which regions are defined as negatives. The overlap threshold, 0.3, was selected by a grid search over {0,0.1,...,0.5} on a validation set. We found that selecting this threshold carefully is important. Setting it to 0.5, as in [39], decreased mAP by 5 points. Similarly, setting it to 0 decreased mAP by 4 points. Positive examples are defined simply to be the ground-truth bounding boxes for each class.

**目标类别分类器**。考虑训练一个二值分类器来检测汽车的情况。一个紧贴着包含汽车的图像区域很明确是一个正样本；类似的，与汽车毫无关系的背景区域也很明确是一个负样本；如果一个区域与汽车部分重叠的情况则不那么明确。我们通过IoU重叠阈值来解决这个问题，低于这个阈值的就认为是负样本。重叠阈值设定为0.3，这是在验证集上逐个验证集合{0,0.1,...,0.5}得到的，要小心的选择这个阈值，这很重要。如果像[39]中一样设为0.5，那么我们的mAP结果要降低5%。类似的，设置为0使mAP降低4%。正样本定义为每个类别的真值边界框。

Once features are extracted and training labels are applied, we optimize one linear SVM per class. Since the training data is too large to fit in memory, we adopt the standard hard negative mining method [17, 37]. Hard negative mining converges quickly and in practice mAP stops increasing after only a single pass over all images.

一旦特征提取完毕，训练标签也应用了，我们对每个类优化一个线性SVM。由于训练数据太大不能全装载入内存，我们采用标准的hard negative mining method [17,37]。Hard negative mining很快就收敛，在实际情况中，在所有图像都循环过一遍后，mAP就停止增长了。

In Appendix B we discuss why the positive and negative examples are defined differently in fine-tuning versus SVM training. We also discuss the trade-offs involved in training detection SVMs rather than simply using the outputs from the final softmax layer of the fine-tuned CNN.

在附录B中我们讨论了为什么在精调和SVM训练中正样本和负样本的定义不同。我们还讨论了训练检测SVM，而不是仅仅使用精调的CNN中最后的softmax层的输出，其中牵扯到的折中。

### 2.4. Results on PASCAL VOC 2010-12

Following the PASCAL VOC best practices [15], we validated all design decisions and hyperparameters on the VOC 2007 dataset (Section 3.2). For final results on the VOC 2010-12 datasets, we fine-tuned the CNN on VOC 2012 train and optimized our detection SVMs on VOC 2012 trainval. We submitted test results to the evaluation server only once for each of the two major algorithm variants (with and without bounding-box regression).

我们遵从PASCAL VOC最好实践[15]的做法，在VOC 2007数据集上验证了所有的设计决定和超参数（3.2节）。为得到在VOC 2010-12数据集的最终结果，我们在VOC 2012训练数据集上精调CNN，在VOC 2012 训练+验证数据集上优化我们的检测SVMs。我们对两种主要算法变体（带有和不带有边界框回归）的测试结果每种只提交到评估服务器一次。

Table 1 shows complete results on VOC 2010. We compare our method against four strong baselines, including SegDPM [18], which combines DPM detectors with the output of a semantic segmentation system [4] and uses additional inter-detector context and image-classifier rescoring. The most germane comparison is to the UVA system from Uijlings et al. [39], since our systems use the same region proposal algorithm. To classify regions, their method builds a four-level spatial pyramid and populates it with densely sampled SIFT, Extended OpponentSIFT, and RGB-SIFT descriptors, each vector quantized with 4000-word codebooks. Classification is performed with a histogram intersection kernel SVM. Compared to their multi-feature, non-linear kernel SVM approach, we achieve a large improvement in mAP, from 35.1% to 53.7% mAP, while also being much faster (Section 2.2). Our method achieves similar performance (53.3% mAP) on VOC 2011/12 test.

表1所示的是VOC 2010的完全结果。我们将我们的算法与四种强基准进行了比较，包括SegDPM[18]，其结合了DPM检测器和一个语义分割系统[4]的输出，使用额外的内检测器上下文和图像分类器重新评分。最贴切的对比是与Uijlings et al. [39]的UVA系统的对比，因为我们的系统用的是一样的候选区域算法。为对区域分类，他们的方法构建了一个四层空域金字塔，用密集采样的SIFT、扩展OpponentSIFT和RGB-SIFT描述子填充，每个向量用4000-word codebooks量化。分类采用的是直方图交集核SVM。与他们的多特征、非线性核SVM方法相比，我们在mAP上有很大改进，从35.1%到53.7%，而且还更快了（2.2节）。我们的方法在VOC 2011/12测试集上得到了类似的表现。

Table 1: Detection average precision (%) on VOC 2010 test. R-CNN is most directly comparable to UVA and Regionlets since all methods use selective search region proposals. Bounding-box regression (BB) is described in Section C. At publication time, SegDPM was the top-performer on the PASCAL VOC leaderboard. † DPM and SegDPM use context rescoring not used by the other methods.

表1：在VOC 2010测试集上的检测平均精度(%)。R-CNN与UVA和Regionlets最具可比性，因为这几种方法都使用selective search进行区域候选。边界框回归(BB)在C节叙述。在发表时，SegDPM是PASCAL VOC排行榜上的表现最佳模型。† DPM和SegDPM使用上下文重评分，别的方法没有使用。

VOC 2010 test | person plant sheep sofa train tv et al. 20 classes | mAP
--- | --- | ---
DPM v5 [20] † | 47.7 10.8 34.2 20.7 43.8 38.3 | 33.4
UVA [39] | 32.9 15.3 41.1 31.8 47.0 44.8 | 35.1
Regionlets [41] | 43.5 14.3 43.9 32.6 54.0 45.9 | 39.7
SegDPM [18] † | 47.1 14.8 38.7 35.0 52.8 43.1 | 40.4
R-CNN | 53.6 26.7 56.5 38.1 52.8 50.2 | 50.2
R-CNN BB | 58.1 29.5 59.4 39.3 61.2 52.4 | 53.7

### 2.5. Results on ILSVRC2013 detection

We ran R-CNN on the 200-class ILSVRC2013 detection dataset using the same system hyperparameters that we used for PASCAL VOC. We followed the same protocol of submitting test results to the ILSVRC2013 evaluation server only twice, once with and once without bounding-box regression.

我们在200类的ILSVRC2013检测数据集上运行R-CNN，使用的是和PASCAL VOC一样的系统超参数。我们遵循提交测试结果给ILSVRC2013评估服务器的协议，提交了两次，一次有边界框回归，一次没有边界框回归。

Figure 3 compares R-CNN to the entries in the ILSVRC 2013 competition and to the post-competition OverFeat result [34]. R-CNN achieves a mAP of 31.4%, which is significantly ahead of the second-best result of 24.3% from OverFeat. To give a sense of the AP distribution over classes, box plots are also presented and a table of per-class APs follows at the end of the paper in Table 8. Most of the competing submissions (OverFeat, NEC-MU, UvA-Euvision, Toronto A, and UIUC-IFP) used convolutional neural networks, indicating that there is significant nuance in how CNNs can be applied to object detection, leading to greatly varying outcomes.

图3将R-CNN与其他ILSVRC 2013竞赛参赛者进行了比较，也和赛后的OverFeat[34]结果进行了比较。R-CNN取得了31.4%的mAP，远远超过第二名OverFeat的24.3%。为直观感受类别间的AP分布，我们绘出了箱线图，在文章最后的表8给出了每个类别的AP值。大多数参赛的提交算法(OverFeat, NEC-MU, UvA-Euvision, Toronto A, and UIUC-IFP)使用了卷积神经网络，表明CNN应用于目标检测的方式差别明显，导致结果也很多样。

Figure 3: (Left) Mean average precision on the ILSVRC2013 detection test set. Methods preceeded by * use outside training data (images and labels from the ILSVRC classification dataset in all cases). (Right) Box plots for the 200 average precision values per method. A box plot for the post-competition OverFeat result is not shown because per-class APs are not yet available (per-class APs for R-CNN are in Table 8 and also included in the tech report source uploaded to arXiv.org; see R-CNN-ILSVRC2013-APs.txt). The red line marks the median AP, the box bottom and top are the 25th and 75th percentiles. The whiskers extend to the min and max AP of each method. Each AP is plotted as a green dot over the whiskers (best viewed digitally with zoom).

图3：（左）在ILSVRC2013检测测试集上的mAP结果。用星号标记的使用了外部训练数据（各种情况下的ILSVRC分类数据集的图像和标签）。（右）每种方法的200类平均准确率值箱线图。赛后的OverFeat结果缺少，因为每类的AP值还不可用（R-CNN的每类AP值在表8中，也在上传到arXiv.org上的科技报告源中，见R-CNN-ILSVRC2013-APs.txt）。红线表明了AP中值，箱底和顶部是25%和75%值。每种方法的线条延伸至AP最小值和最高值。每个AP都画作一个线条上的绿点（最好在电子档上放大观察）。

In Section 4, we give an overview of the ILSVRC2013 detection dataset and provide details about choices that we made when running R-CNN on it.

在第4节中，我们给出了ILSVRC2013检测数据集的概览，并详述了在数据集上运行R-CNN时做出的选择。

## 3. Visualization, ablation, and modes of error 可视化、简化测试和错误模式

### 3.1. Visualizing learned features 可视化学习到的特征

First-layer filters can be visualized directly and are easy to understand [25]. They capture oriented edges and opponent colors. Understanding the subsequent layers is more challenging. Zeiler and Fergus present a visually attractive deconvolutional approach in [42]. We propose a simple (and complementary) non-parametric method that directly shows what the network learned.

第一层滤波器可以直接可视化，易于理解[25]。捕获的是方向性的边缘和opponent colors。理解后面的层更有挑战性。Zeiler and Fergus给出了可视化效果较好的解卷积方法[42]。我们提出一种简单的（补充性的）非参数方法，直接展示出网络学习到的是什么。

The idea is to single out a particular unit (feature) in the network and use it as if it were an object detector in its own right. That is, we compute the unit’s activations on a large set of held-out region proposals (about 10 million), sort the proposals from highest to lowest activation, perform non-maximum suppression, and then display the top-scoring regions. Our method lets the selected unit “speak for itself” by showing exactly which inputs it fires on. We avoid averaging in order to see different visual modes and gain insight into the invariances computed by the unit.

其思想是选出网络中的一个特殊单元（特征），将其作为一个独立的目标检测器来使用。也就是，我们在大型留存候选区域集（约1000万个）上计算这个单元的激活值，将候选区域按激活值从高到低排序，进行非最大抑制处理，然后显示出最高评分的区域。我们的方法让这个选中的单元“为自己代言”，展示对哪个输入影响最大。我们没有平均，这样可以看到不同的视觉模式，得到这个单元计算的不变性。

We visualize units from layer pool 5 , which is the max-pooled output of the network’s fifth and final convolutional layer. The pool 5 feature map is 6 × 6 × 256 = 9216-dimensional. Ignoring boundary effects, each pool 5 unit has a receptive field of 195×195 pixels in the original 227×227 pixel input. A central pool 5 unit has a nearly global view, while one near the edge has a smaller, clipped support.

我们可视化pool 5层的单元，即网络第5层、最后一个卷积层的max-pooled输出。pool 5特征图维数为6 × 6 × 256 = 9216。忽略边缘效应，每个pool 5单元的感受野为原始227×227输入像素中的195×195像素。中心处的pool 5单元几乎有全局视野，而边缘处的单元则有更小一些的剪切过的范围。

Each row in Figure 4 displays the top 16 activations for a pool 5 unit from a CNN that we fine-tuned on VOC 2007 trainval. Six of the 256 functionally unique units are visualized (Appendix D includes more). These units were selected to show a representative sample of what the network learns. In the second row, we see a unit that fires on dog faces and dot arrays. The unit corresponding to the third row is a red blob detector. There are also detectors for human faces and more abstract patterns such as text and triangular structures with windows. The network appears to learn a representation that combines a small number of class-tuned features together with a distributed representation of shape, texture, color, and material properties. The subsequent fully connected layer fc 6 has the ability to model a large set of compositions of these rich features.

图4中的每行显示了一个pool 5单元的最高的16个激活值，这个单元处于我们在VOC 2007训练验证集上精调过的CNN中。256个功能独特单元中的6个进行了可视化（附录D中包括更多）。选中这些单元展示了网络学习到了什么代表性的样本。在第二行中，我们看到单元学习到的是狗脸和点阵列。第三行对应的单元是一个红色团状物检测器。也有人脸检测器和更抽象的模式检测器，如文字和带窗的三角形结构。网络似乎学到了一种表示，这种表示结合了少数与类别相关的特征，和形状、纹理、颜色、材质性质的分布表示。后面的全连接层fc 6的能力是对这些丰富特征的合成的集合进行建模。

Figure 4: Top regions for six pool 5 units. Receptive fields and activation values are drawn in white. Some units are aligned to concepts, such as people (row 1) or text (4). Other units capture texture and material properties, such as dot arrays (2) and specular reflections (6).

图4：六个pool 5单元的最高响应区域。感受野和激活值用白色表示。一些单元表示概念，如人（第1行）或文字（第4行）。其他单元捕捉纹理或材质性质，比如点阵(2)和镜面反射(6)。

### 3.2. Ablation studies 简化测试研究

**Performance layer-by-layer, without fine-tuning**. To understand which layers are critical for detection performance, we analyzed results on the VOC 2007 dataset for each of the CNN’s last three layers. Layer pool 5 was briefly described in Section 3.1. The final two layers are summarized below.

**逐层性能研究，不含精调**。为理解哪些层对于检测表现是重要的，我们分析了在VOC 2007数据集上CNN最后3层每层的结果。pool 5层在3.1节进行了简述。最后2层总结如下。

Layer fc 6 is fully connected to pool 5 . To compute features, it multiplies a 4096×9216 weight matrix by the pool 5 feature map (reshaped as a 9216-dimensional vector) and then adds a vector of biases. This intermediate vector is component-wise half-wave rectified (x ← max(0,x)).

fc 6层与pool 5层全连接。为计算特征，将4096×9216的权值矩阵与pool 5层的特征图（重整为9216维的矢量）相乘，然后加上一个偏置矢量。这个中间矢量又逐个元素进行了半波整流，即ReLU激活。

Layer fc 7 is the final layer of the network. It is implemented by multiplying the features computed by fc 6 by a 4096 × 4096 weight matrix, and similarly adding a vector of biases and applying half-wave rectification.

fc 7层是网络的最后一层。将fc 6计算得到的特征与4096×4096的权值矩阵相乘，类似的，也加上一个偏置矢量，进行半波整流。

We start by looking at results from the CNN without fine-tuning on PASCAL, i.e. all CNN parameters were pre-trained on ILSVRC 2012 only. Analyzing performance layer-by-layer (Table 2 rows 1-3) reveals that features from fc 7 generalize worse than features from fc 6 . This means that 29%, or about 16.8 million, of the CNN’s parameters can be removed without degrading mAP. More surprising is that removing both fc 7 and fc 6 produces quite good results even though pool 5 features are computed using only 6% of the CNN’s parameters. Much of the CNN’s representational power comes from its convolutional layers, rather than from the much larger densely connected layers. This finding suggests potential utility in computing a dense feature map, in the sense of HOG, of an arbitrary-sized image by using only the convolutional layers of the CNN. This representation would enable experimentation with sliding-window detectors, including DPM, on top of pool 5 features.

我们首先观察没有经过PASCAL精调的CNN的结果，即所有的CNN参数都是只由ILSVRC 2012预训练的。逐层分析性能（见表2行1-3）发现fc 7层的特征泛化能力比fc 6的特征泛化能力要差。这意味着CNN参数的29%，也就是1680万参数，可以从网络中去除，而不影响mAP。更令人惊讶的是，去掉fc 7和fc 6层也可以得到不错的结果，而pool 5层的特征是只用了整个网络6%的参数计算得到的。也就是说，CNN的大部分表示能力是从卷积层计算得到的，而不是从密集连接的全连接层计算得到的。这个发现说明，可以只使用CNN的卷积层来计算任意形状图像的密集特征图（相较于HOG来说）。这种表示可以在pool 5层特征的基础上采用滑窗检测器进行试验，包括DPM方法。

**Performance layer-by-layer, with fine-tuning**. We now look at results from our CNN after having fine-tuned its parameters on VOC 2007 trainval. The improvement is striking (Table 2 rows 4-6): fine-tuning increases mAP by 8.0 percentage points to 54.2%. The boost from fine-tuning is much larger for fc 6 and fc 7 than for pool 5 , which suggests that the pool 5 features learned from ImageNet are general and that most of the improvement is gained from learning domain-specific non-linear classifiers on top of them.

**逐层性能研究，含有精调**。我们现在看看经过VOC 2007训练验证集精调的CNN的结果。其改进是惊人的（表2行4-6）：精调使mAP上升了8个百分点，到了54.2%。精调带来的提升对于fc 6和fc 7层，比对pool 5层要大的多，这意味着从ImageNet学到的pool 5特征是一般性的，大部分的改进是在此基础上学习领域相关的非线性分类器得到的。

**Comparison to recent feature learning methods**. Relatively few feature learning methods have been tried on PASCAL VOC detection. We look at two recent approaches that build on deformable part models. For reference, we also include results for the standard HOG-based DPM [20].

**与最近的特征学习方法的对比**。在PASCAL VOC检测中只进行过相对少数特征学习方法。我们看看两种最近在DPM上建立的方法。作为参考，我们也包括了标准的基于HOG的DPM[20]的结果。

The first DPM feature learning method, DPM ST [28], augments HOG features with histograms of “sketch token” probabilities. Intuitively, a sketch token is a tight distribution of contours passing through the center of an image patch. Sketch token probabilities are computed at each pixel by a random forest that was trained to classify 35×35 pixel patches into one of 150 sketch tokens or background.

第一个DPM特征学习方法，DPM ST[28]，用“sketch token”概率直方图来增强HOG特征。直观上，sketch token是经过图像块中心的轮廓分布。在每个像素上用随机森林计算sktch token概率，随机森林是训练后用于将35×35像素块分类成150个sketch token中的一个或背景。

The second method, DPM HSC [31], replaces HOG with histograms of sparse codes (HSC). To compute an HSC, sparse code activations are solved for at each pixel using a learned dictionary of 100 7 × 7 pixel (grayscale) atoms. The resulting activations are rectified in three ways (full and both half-waves), spatially pooled, unit $L_2$ normalized, and then power transformed ($x ← sign(x)|x|^α$).

第二种方法，DPM HSC[31]，将HOG替换为稀疏码直方图(HSC)。计算HSC的方法是，用学习到的100个7×3像素（灰度）原子字典在每个像素处求解稀疏码激活。得到的激活值经过三种方法整流（全波整流和两种半波整流），空间上pool操作，对每个单元进行$L_2$归一化，最后进行幂变换($x ← sign(x)|x|^α$)。

All R-CNN variants strongly outperform the three DPM baselines (Table 2 rows 8-10), including the two that use feature learning. Compared to the latest version of DPM, which uses only HOG features, our mAP is more than 20 percentage points higher: 54.2% vs. 33.7%—a 61% relative improvement. The combination of HOG and sketch tokens yields 2.5 mAP points over HOG alone, while HSC improves over HOG by 4 mAP points (when compared internally to their private DPM baselines—both use non-public implementations of DPM that underperform the open source version [20]). These methods achieve mAPs of 29.1% and 34.3%, respectively.

所有R-CNN变体的表现都比这三种DPM基准方法好很多（表2行8-10），包括使用特征学习的两种方法。与最新版的只使用HOG特征的DPM比较，我们的mAP高出了20%：54.2% vs. 33.7%，相对改进61%。HOG和sketch token结合得到了2.5%的mAP提升，HSC改进了4%的mAP（当内部与他们的不公开的DPM基准比较时，两种用了非公开的DPM实现比开源版本都要差一点[20]）。这些方法分别得到了29.1%和34.3%的mAP。

Table 2: Detection average precision (%) on VOC 2007 test. Rows 1-3 show R-CNN performance without fine-tuning. Rows 4-6 show results for the CNN pre-trained on ILSVRC 2012 and then fine-tuned (FT) on VOC 2007 trainval. Row 7 includes a simple bounding-box regression (BB) stage that reduces localization errors (Section C). Rows 8-10 present DPM methods as a strong baseline. The first uses only HOG, while the next two use different feature learning approaches to augment or replace HOG.

表2：在VOC 2007测试集上的检测平均准确度(%)。1-3行是没有精调的R-CNN的结果，4-6行是在ILSVRC-2012预训练，然后在VOC 2007训练验证集精调的R-CNN的结果。第7行是包含了一个简单的边界框回归(BB)阶段的算法，减少了定位错误率（C节）。8-10行将DPM方法作为强基准。第一种只用HOG，后面两种用了不同的特征学习方法来增强或替换HOG。

VOC 2007 test | person plant sheep sofa train tv et al. 20 classes | mAP
--- | --- | ---
R-CNN pool 5 | 42.4 23.4 46.1 36.7 51.3 55.7 | 44.2
R-CNN fc 6 | 44.6 25.6 48.3 34.0 53.1 58.0 | 46.2
R-CNN fc 7 | 43.3 23.3 48.1 35.3 51.0 57.4 | 44.7
R-CNN FT pool 5 | 45.8 28.1 50.8 40.6 53.1 56.4 | 47.3
R-CNN FT fc 6 | 52.2 31.3 55.0 50.0 57.7 63.0 | 53.1
R-CNN FT fc 7 | 54.2 31.5 52.8 48.9 57.9 64.7 | 54.2
R-CNN FT fc 7 BB | 58.7 33.4 62.9 51.1 62.5 64.8 | 58.5
DPM v5 [20] | 43.2 12.0 21.1 36.1 46.0 43.5 | 33.7
DPM ST [28] | 32.4 13.3 15.9 22.8 46.2 44.9 | 29.1
DPM HSC [31] | 39.9 12.4 23.5 34.4 47.4 45.2 | 34.3

### 3.3. Network architectures 网络架构

Most results in this paper use the network architecture from Krizhevsky et al. [25]. However, we have found that the choice of architecture has a large effect on R-CNN detection performance. In Table 3 we show results on VOC 2007 test using the 16-layer deep network recently proposed by Simonyan and Zisserman [43]. This network was one of the top performers in the recent ILSVRC 2014 classification challenge. The network has a homogeneous structure consisting of 13 layers of 3 × 3 convolution kernels, with five max pooling layers interspersed, and topped with three fully-connected layers. We refer to this network as “O-Net” for OxfordNet and the baseline as “T-Net” for TorontoNet.

本文的多数结果使用Krizhevsky et al. [25]的网络架构。但我们发现网络架构的选择对R-CNN检测性能有很大影响。在表3中我们给出在VOC 2007测试集上使用Simonyan and Zisserman [43]提出的16层深度网络的结果。这个网络是最近的ILSVRC 2014分类挑战赛的最好表现者之一。这个网络有13个同样结构的3×3卷积核的卷积层，中间散布着5个max pooling层，最后是三层全连接层。我们将这个OxfordNet简称为O-Net，TorontoNet简称为T-Net。

To use O-Net in R-CNN, we downloaded the publicly available pre-trained network weights for the VGG ILSVRC 16 layers model from the Caffe Model Zoo. We then fine-tuned the network using the same protocol as we used for T-Net. The only difference was to use smaller mini-batches (24 examples) as required in order to fit within GPU memory. The results in Table 3 show that R-CNN with O-Net substantially outperforms R-CNN with T-Net, increasing mAP from 58.5% to 66.0%. However there is a considerable drawback in terms of compute time, with the forward pass of O-Net taking roughly 7 times longer than T-Net.

为在R-CNN中使用O-Net，我们从Caffe模型库中下载了公开的VGG16网络预训练权重。然后用与T-Net中使用的相同的协议来精调网络。唯一的区别是使用小一些的mini-batch（24样本），这是由于GPU内存的原因。表3的结果说明，使用O-Net的R-CNN明显优于使用T-Net的，mAP由58.5%增加到了66.0%。但是计算时间大大增加，O-Net的时间是T-Net时间的7倍。

Table 3: Detection average precision (%) on VOC 2007 test for two different CNN architectures. The first two rows are results from Table 2 using Krizhevsky et al.’s architecture (T-Net). Rows three and four use the recently proposed 16-layer architecture from Simonyan and Zisserman (O-Net) [43].

表3：两种不同CNN架构在VOC 2007测试集上的检测平均准确率(%)。前两行是表2使用Krizhevsky et al.架构(T-Net)的结果。第3、4行使用了最近Simonyan and Zisserman (O-Net) [43]提出的16层架构。

VOC 2007 test | person plant sheep sofa train tv et al. 20 classes | mAP
--- | --- | ---
R-CNN T-Net | 54.2 31.5 52.8 48.9 57.9 64.7 | 54.2
R-CNN T-Net BB | 58.7 33.4 62.9 51.1 62.5 64.8 | 58.5
R-CNN O-Net | 59.3 35.7 62.1 64.0 66.5 71.2 | 62.2
R-CNN O-Net BB | 64.2 35.6 66.8 67.2 70.4 71.1 | 66.0

### 3.4. Detection error analysis 检测错误分析

We applied the excellent detection analysis tool from Hoiem et al. [23] in order to reveal our method’s error modes, understand how fine-tuning changes them, and to see how our error types compare with DPM. A full summary of the analysis tool is beyond the scope of this paper and we encourage readers to consult [23] to understand some finer details (such as “normalized AP”). Since the analysis is best absorbed in the context of the associated plots, we present the discussion within the captions of Figure 5 and Figure 6.

我们使用了优秀的Hoiem et al. [23]检测分析工具来检查我们方法的错误模式，理解精调使怎样改变它们的，并研究我们的错误类型与DPM的有何不同。分析工具的完整总结不是本文的任务，推荐读者参考[23]来理解详情（比如归一化AP）。由于分析结果与相关的图上下文一起更好理解，我们将讨论放在图5和图6的标题中。

Figure 5: Distribution of top-ranked false positive (FP) types. Each plot shows the evolving distribution of FP types as more FPs are considered in order of decreasing score. Each FP is categorized into 1 of 4 types: Loc—poor localization (a detection with an IoU overlap with the correct class between 0.1 and 0.5, or a duplicate); Sim—confusion with a similar category; Oth—confusion with a dissimilar object category; BG—a FP that fired on background. Compared with DPM (see [23]), significantly more of our errors result from poor localization, rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG. Loose localization likely results from our use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification. Column three shows how our simple bounding-box regression method fixes many localization errors.

图5：最多的false positive类型分布。每个图都展示了当按照递减的评分考虑更多的FP时，FP类型分布的演化情况。每个FP都是下面4类中的一种：Loc - 定位不准确（与正确类别的IoU重叠在0.1到0.5之间的检测结果，或重复检测结果）；Sim - 与类似的类别混淆；Oth - 与不相似的类别混淆；BG - 将背景识别为目标的检测结果。与DPM[23]相比，我们的错误结果更多的是由于定位不准确的原因，而不是与背景混淆或其他目标类别混淆，这表明CNN特征比HOG特征更具有分辨能力。定位不准确很可能是由于我们使用了自下而上的候选区域，以及预训练的CNN是为整图分类的，从中也学到了位置不变性。第3列是我们的简单边界框回归方法怎样修正了很多定位错误的。

Figure 6: Sensitivity to object characteristics. Each plot shows the mean (over classes) normalized AP (see [23]) for the highest and lowest performing subsets within six different object characteristics (occlusion, truncation, bounding-box area, aspect ratio, viewpoint, part visibility). We show plots for our method (R-CNN) with and without fine-tuning (FT) and bounding-box regression (BB) as well as for DPM voc-release5. Overall, fine-tuning does not reduce sensitivity (the difference between max and min), but does substantially improve both the highest and lowest performing subsets for nearly all characteristics. This indicates that fine-tuning does more than simply improve the lowest performing subsets for aspect ratio and bounding-box area, as one might conjecture based on how we warp network inputs. Instead, fine-tuning improves robustness for all characteristics including occlusion, truncation, viewpoint, and part visibility.

图6：对目标特征的敏感度。每个图都展示了6个不同目标特性的最高和最低表现的子集的平均（在所有类别中）归一化AP[23]，分别是遮挡、截断、、边界框区域、纵横比、视角和部分可见性。我们给出了我们的方法(R-CNN)精调的、未精调的和带有边界框回归(BB)的，和DPM voc-release5的结果图。总体上来说，精调对所有图像特质都不会降低其敏感度，但的确改进了最高和最低表现值。这表明精调不仅仅改进了纵横比子集和边界框区域子集的最低表现，因为可能会基于我们怎样将网络输入变形来进行推断。相反，精调改进了所有特质图像的鲁棒性，包括遮挡、截断、视角和部分可见性。

### 3.5. Bounding-box regression 边界框回归

Based on the error analysis, we implemented a simple method to reduce localization errors. Inspired by the bounding-box regression employed in DPM [17], we train a linear regression model to predict a new detection window given the pool 5 features for a selective search region proposal. Full details are given in Appendix C. Results in Table 1, Table 2, and Figure 5 show that this simple approach fixes a large number of mislocalized detections, boosting mAP by 3 to 4 points.

在错误分析的基础上，我们实现了一个简单的方法来减少定位错误。受DPM[17]中使用的边界框回归方法的启发，我们训练了一个线性回归模型，为selective search的候选区域，在给定pool 5层特征的情况下，预测新的检测窗口。详见附录C。表1、表2和图5的结果显示，这种简单方法解决大量的错误定位的检测问题，将mAP提升了3到4个百分点。

### 3.6. Qualitative results 定量结果

Qualitative detection results on ILSVRC2013 are presented in Figure 8 and Figure 9 at the end of the paper. Each image was sampled randomly from the val 2 set and all detections from all detectors with a precision greater than 0.5 are shown. Note that these are not curated and give a realistic impression of the detectors in action. More qualitative results are presented in Figure 10 and Figure 11, but these have been curated. We selected each image because it contained interesting, surprising, or amusing results. Here, also, all detections at precision greater than 0.5 are shown.

在ILSVRC2013的定量检测结果在文末图8和图9中给出。每个图像都是随机从val 2集中取样的，所有检测器的所有检测精度大于0.5的都进行了展示。注意这些结果没有受到引导，给出了实际使用的检测器的实际印象。更多量化结果在图10和图11中给出，但这些结果是引导过的。每个图像都是经过选择的，因为包含了吸引人的、令人惊讶的或有趣的结果。这里一样，所有检测结果精度大于0.5的才展示出来。

## 4. The ILSVRC2013 detection dataset 检测数据集

In Section 2 we presented results on the ILSVRC2013 detection dataset. This dataset is less homogeneous than PASCAL VOC, requiring choices about how to use it. Since these decisions are non-trivial, we cover them in this section.

在第2节中，我们给出了在ILSVRC2013检测数据集上的结果。这个数据集比PASCAL VOC同质性略差，取决于怎样使用它。因为这些决策都不是细枝末节，所以我们在这一节详述。

### 4.1. Dataset overview 数据集概览

The ILSVRC2013 detection dataset is split into three sets: train (395,918), val (20,121), and test (40,152), where the number of images in each set is in parentheses. The val and test splits are drawn from the same image distribution. These images are scene-like and similar in complexity (number of objects, amount of clutter, pose variability, etc.) to PASCAL VOC images. The val and test splits are exhaustively annotated, meaning that in each image all instances from all 200 classes are labeled with bounding boxes. The train set, in contrast, is drawn from the ILSVRC2013 classification image distribution. These images have more variable complexity with a skew towards images of a single centered object. Unlike val and test, the train images (due to their large number) are not exhaustively annotated. In any given train image, instances from the 200 classes may or may not be labeled. In addition to these image sets, each class has an extra set of negative images. Negative images are manually checked to validate that they do not contain any instances of their associated class. The negative image sets were not used in this work. More information on how ILSVRC was collected and annotated can be found in [11, 36].

ILSVRC2013检测数据集可以分为3部分：训练集(395,918)，验证集(20,121)，测试集(40,152)，其中每个集合的图片数量在括号中。验证和测试数据集是以相同的图像分布抽取出来的。这些图像于PASCAL VOC图像场景类似，复杂度也类似（目标数量、杂乱程度、姿态多样性等）。验证和测试集进行了详尽的标注，意思是每幅图像中200类的所有实例都添加了边界框的标签。训练集是从ILSVRC2013分类数据集中抽取出来的。与只有单个中心目标的图像相比，这些图像复杂度参差不齐。与验证集和测试集不同，训练图像（由于其数量巨大）并没有详尽标注。在任意给定的训练图像中，200类的实例可能有标签，也可能没有标签。除了这些图像数据集，每个类都还有负图像集。负图像都经过手动检查，以确认不包含关联类别的任何实例。本文的工作中没有用到负图像集。ILSVRC是怎样收集的，怎样标注的，可以参考[11,36]。

The nature of these splits presents a number of choices for training R-CNN. The train images cannot be used for hard negative mining, because annotations are not exhaustive. Where should negative examples come from? Also, the train images have different statistics than val and test. Should the train images be used at all, and if so, to what extent? While we have not thoroughly evaluated a large number of choices, we present what seemed like the most obvious path based on previous experience.

这些数据集为训练R-CNN提供了选择。训练图像不能用于难分样本挖掘，因为没有详尽标注。那么负样本从哪里得到呢？还有，与验证集和测试集相比，训练图像有不同的统计特性。应该使用这些训练图像吗，如果用的话，用到什么程度呢？我们没有完全评估非常多的选择，但我们基于以前的经验提出了似乎是最明显的路径。

Our general strategy is to rely heavily on the val set and use some of the train images as an auxiliary source of positive examples. To use val for both training and validation, we split it into roughly equally sized “val 1 ” and “val 2 ” sets. Since some classes have very few examples in val (the smallest has only 31 and half have fewer than 110), it is important to produce an approximately class-balanced partition. To do this, a large number of candidate splits were generated and the one with the smallest maximum relative class imbalance was selected. (Relative imbalance is measured as |a − b|/(a + b) where a and b are class counts in each half of the split.) Each candidate split was generated by clustering val images using their class counts as features, followed by a randomized local search that may improve the split balance. The particular split used here has a maximum relative imbalance of about 11% and a median relative imbalance of 4%. The val 1 /val 2 split and code used to produce them will be publicly available to allow other researchers to compare their methods on the val splits used in this report.

我们的一般性策略是更多的依靠验证集，使用一部分训练图像作为正样本的辅助来源。为使用验证集既进行训练，又进行验证，我们将其分成大致相同的验证集1和验证集2，val 1和val 2。由于一些类别验证集样本很少，最小的只有大约31个样本，半数的样本少于110个，那么生成大致类别均衡的集合就非常重要了。为了达到这个目标，生成了大量候选集，选出相对不均衡最小的类别（相对不均衡为|a − b|/(a + b)，其中a和b是类别中两个验证集的样本数目）。每个候选集的产生方法为，将验证集图像进行聚类，使用类别数目作为特征，然后经过随机本地搜索，这样可能改进集合均衡性。这里用到的集合最大相对不均衡性为11%，中值相对不均衡性为4%。val 1和val 2集和生成集合的代码会进行公开，以便其他研究者在这些集合上比较他们的方法。

###4.2. Region proposals 候选区域

We followed the same region proposal approach that was used for detection on PASCAL. Selective search [39] was run in “fast mode” on each image in val 1 , val 2 , and test (but not on images in train). One minor modification was required to deal with the fact that selective search is not scale invariant and so the number of regions produced depends on the image resolution. ILSVRC image sizes range from very small to a few that are several mega-pixels, and so we resized each image to a fixed width (500 pixels) before running selective search. On val, selective search resulted in an average of 2403 region proposals per image with a 91.6% recall of all ground-truth bounding boxes (at 0.5 IoU threshold). This recall is notably lower than in PASCAL, where it is approximately 98%, indicating significant room for improvement in the region proposal stage.

我们采用的区域候选方法与用在PASCAL检测中的一样。Selective search[39]在val 1、val 2、测试集中的每个图像上以快速模式运行（但没有在训练集中运行）。一个小的改动是，selective search不具有尺度不变性，所以得到的区域数量取决于图像分辨率，这个问题需要解决。ILSVRC图像尺寸的范围从很小到很大都有，所以我们将每幅图像尺寸改变为固定宽度（500像素），然后运行selective search。在val集上，selective search在每幅图像上平均得到了2403个候选区域，召回率91.6%（IoU阈值为0.5）。这个召回率比PASCAL中(98%)的明显低，说明在区域候选阶段还有很多可以改进之处。

### 4.3. Training data 训练数据

For training data, we formed a set of images and boxes that includes all selective search and ground-truth boxes from val 1 together with up to N ground-truth boxes per class from train (if a class has fewer than N ground-truth boxes in train, then we take all of them). We’ll call this dataset of images and boxes val 1 + train N. In an ablation study, we show mAP on val 2 for N ∈ {0,500,1000} (Section 4.5).

对于训练数据，我们形成了图像和边界框集，这个集合中包括val 1集中用selective search产生的边界框和ground-truth边界框，还有在训练集中每类的N个ground-truth边界框（如果某类在训练集中的ground-truth边界框少于N个，那就全部包含进来）。我们称这个图像和边界框的数据集为val 1 + train N。在一个简化测试研究中，我们在4.5节展示在val 2集中，N ∈ {0,500,1000}的mAP。

Training data is required for three procedures in R-CNN: (1) CNN fine-tuning, (2) detector SVM training, and (3) bounding-box regressor training. CNN fine-tuning was run for 50k SGD iteration on val 1 +train N using the exact same settings as were used for PASCAL. Fine-tuning on a single NVIDIA Tesla K20 took 13 hours using Caffe. For SVM training, all ground-truth boxes from val 1 +train N were used as positive examples for their respective classes. Hard negative mining was performed on a randomly selected subset of 5000 images from val 1 . An initial experiment indicated that mining negatives from all of val 1 , versusa 5000 image subset (roughly half of it), resulted in only a 0.5 percentage point drop in mAP, while cutting SVM training time in half. No negative examples were taken from train because the annotations are not exhaustive. The extra sets of verified negative images were not used. The bounding-box regressors were trained on val 1.

R-CNN中再三个过程中需要训练数据：(1)CNN精调，(2)SVM检测器训练，(3)边界框回归器训练。CNN精调在val 1 +train N集上运行了50k次SGD迭代的，这在PASCAL中也是用的这个设定。精调在单张NVIDIA Tesla K20卡上用caffe花了13个小时。对于SVM训练，所有val 1 +train N中的ground-truth边界框都用作相应类别的正样本。难分样本挖掘是在从val 1集中随机选择出来的5000幅图像的子集中进行的。开始的试验表明从val 1集中所有图像中挖掘负样本，和从5000幅图像（大约全部的一半）的子集中挖掘只带来了mAP下降了0.5%，但却使SVM的训练时间缩短了一半。没有从训练集中选取负样本，是因为标注不详尽。额外的验证负图像集没有使用。边界框回归器是在val 1中训练的。

### 4.4. Validation and evaluation 验证和评估

Before submitting results to the evaluation server, we validated data usage choices and the effect of fine-tuning and bounding-box regression on the val 2 set using the training data described above. All system hyperparameters (e.g., SVM C hyperparameters, padding used in region warping, NMS thresholds, bounding-box regression hyperparameters) were fixed at the same values used for PASCAL. Undoubtedly some of these hyperparameter choices are slightly suboptimal for ILSVRC, however the goal of this work was to produce a preliminary R-CNN result on ILSVRC without extensive dataset tuning. After selecting the best choices on val 2 , we submitted exactly two result files to the ILSVRC2013 evaluation server. The first submission was without bounding-box regression and the second submission was with bounding-box regression. For these submissions, we expanded the SVM and bounding-box regressor training sets to use val+train 1k and val, respectively. We used the CNN that was fine-tuned on val 1 +train 1k to avoid re-running fine-tuning and feature computation.

在将结果提交到评估服务器之前，我们验证在val 2集上验证了数据使用选择的效果，和精调的效果，以及边界框回归的效果，使用的训练数据如上所述。所有系统超参数（如，SVM C超参数，区域变形中的padding，NMS阈值，边界框回归超参数）的选取都和PSACAL中使用的一样。这些超参数中一部分肯定对于ILSVRC来说不是最优选项，但任务目标是在没有数据集大调整的情况下，在ILSVRC上生成初步的R-CNN结果。在val 2上进行最好的选择后，我们将两份结果文件提交到ILSVRC2013评估服务器上。第一个提交时没有边界框回归的，第二个是带有边界框回归的。对于这两份提交，我们将SVM和边界框回归的训练集分别扩展到val+train 1k和val集上。我们使用在val 1 +train 1k上精调的CNN来避免重新进行精调和特征计算。

### 4.5. Ablation study 简化测试研究

Table 4 shows an ablation study of the effects of different amounts of training data, fine-tuning, and bounding-box regression. A first observation is that mAP on val 2 matches mAP on test very closely. This gives us confidence that mAP on val 2 is a good indicator of test set performance. The first result, 20.9%, is what R-CNN achieves using a CNN pre-trained on the ILSVRC2012 classification dataset (no fine-tuning) and given access to the small amount of training data in val 1 (recall that half of the classes in val 1 have between 15 and 55 examples). Expanding the training set to val 1 +train N improves performance to 24.1%, with essentially no difference between N = 500 and N = 1000. Fine-tuning the CNN using examples from just val 1 gives a modest improvement to 26.5%, however there is likely significant overfitting due to the small number of positive training examples. Expanding the fine-tuning set to val 1 +train 1k , which adds up to 1000 positive examples per class from the train set, helps significantly, boosting mAP to 29.7%. Bounding-box regression improves results to 31.0%, which is a smaller relative gain than what was observed in PASCAL.

表4所示的是采用不同的训练数据、精调和边界框回归效果的简化测试研究。第一个观察是在val 2上的mAP与在测试集上的非常接近。这说明，在val 2上的mAP能很好的说明在测试集上的表现。第一个结果，是R-CNN使用在ILSVRC2012分类数据集上预训练的CNN（无精调）和在小训练集val 1上训练的结果（回忆一下，val 1集中的半数类只有15到55个样本）。将训练集扩展到val 1 + train N将结果改进为24.1%，在N=500和N=1000的情况下都是这个结果。使用val 1集的数据精调CNN将结果改进至26.5%，但是由于正样本数太少，很可能有过拟合的情况。将精调集扩展至val 1 +train 1k，这在每类中大约从训练集中增加了1000个正样本，明显将mAP提升至29.7%。边界框回归将结果改进到31.0%，这与在PASCAL的情况相比，是一个小一些的提升。

Table 4: ILSVRC2013 ablation study of data usage choices, fine-tuning, and bounding-box regression.

### 4.6. Relationship to OverFeat 与OverFeat的关系

There is an interesting relationship between R-CNN and OverFeat: OverFeat can be seen (roughly) as a special case of R-CNN. If one were to replace selective search region proposals with a multi-scale pyramid of regular square regions and change the per-class bounding-box regressors to a single bounding-box regressor, then the systems would be very similar (modulo some potentially significant differences in how they are trained: CNN detection fine-tuning, using SVMs, etc.). It is worth noting that OverFeat has a significant speed advantage over R-CNN: it is about 9x faster, based on a figure of 2 seconds per image quoted from [34]. This speed comes from the fact that OverFeat’s sliding windows (i.e., region proposals) are not warped at the image level and therefore computation can be easily shared between overlapping windows. Sharing is implemented by running the entire network in a convolutional fashion over arbitrary-sized inputs. Speeding up R-CNN should be possible in a variety of ways and remains as future work.

R-CNN与OverFeat关系很有趣：OverFeat大致可以看作是R-CNN的一个特例。如果将selective search候选区域法替换为常规的方形区域的多尺度金字塔，将每类的边界框回归器替换为单个边界框回归器，那么两个模型就很像了（除去一些在训练方式上的潜在的明显区别：CNN检测精调，使用SVM等）。值的注意的是，OverFeat在速度上优势很大：大约快了9倍，这是基于[34]中的一幅图提到每个图像2秒。这个速度是因为OverFeat的滑窗（也就是候选区域）在图像层没有变形，所以重叠窗口间可以很容易分享计算。分享实现的方式是，在任意大小的输入上以卷积的方式运行整个网络。R-CNN可以以多种方式加速，这在以后进行研究。

## 5. Semantic segmentation 语义分割

Region classification is a standard technique for semantic segmentation, allowing us to easily apply R-CNN to the PASCAL VOC segmentation challenge. To facilitate a direct comparison with the current leading semantic segmentation system (called $O_2P$ for “second-order pooling”) [4], we work within their open source framework. $O_2P$ uses CPMC to generate 150 region proposals per image and then predicts the quality of each region, for each class, using support vector regression (SVR). The high performance of their approach is due to the quality of the CPMC regions and the powerful second-order pooling of multiple feature types (enriched variants of SIFT and LBP). We also note that Farabet et al. [16] recently demonstrated good results on several dense scene labeling datasets (not including PASCAL) using a CNN as a multi-scale per-pixel classifier.

区域分类是语音分割的标准技术，所以我们可以很容易将R-CNN应用于PASCAL VOC的分割挑战上。为能与目前最好的语音分割系统（称为$O_2P$，即二阶pooling[4]）进行比较，我们在其开源框架进行工作。$O_2P$使用CPMC在每幅图像中生成150个候选区域，然后预测每个区域、每个类的质量，使用的是支持矢量回归(SVR)。其方法的高性能源于CMPC区域的质量和强劲的多种特征二阶pooling（SIFT和LBP的增强变体）。我们还注意到Farabet et al. [16]最近在几个密集场景标记数据集（不包括PASCAL）上得到了很好的结果，其中将CNN用于多尺度逐像素分类器。

We follow [2, 4] and extend the PASCAL segmentation training set to include the extra annotations made available by Hariharan et al. [22]. Design decisions and hyperparameters were cross-validated on the VOC 2011 validation set. Final test results were evaluated only once.

我们仿照[2,4]，将PASCAL分割训练集扩展，包含Hariharan et al. [22]进行的额外标注。设计决策与超参数在VOC2011验证集上进行了交叉验证。最终测试结果只进行了一个评估。

**CNN features for segmentation**. We evaluate three strategies for computing features on CPMC regions, all of which begin by warping the rectangular window around the region to 227 × 227. The first strategy (full) ignores the region’s shape and computes CNN features directly on the warped window, exactly as we did for detection. However, these features ignore the non-rectangular shape of the region. Two regions might have very similar bounding boxes while having very little overlap. Therefore, the second strategy (fg) computes CNN features only on a region’s foreground mask. We replace the background with the mean input so that background regions are zero after mean subtraction. The third strategy (full+fg) simply concatenates the full and fg features; our experiments validate their complementarity.

**用CNN特征进行分割**。我们评估三种在CPMC区域上计算特征的策略，三种方法开始都是先将矩形窗口包围的区域大小变形为227×227。第一种策略(full)忽略了区域的形状，直接在变形后窗口计算CNN特征，这与我们进行检测的做法是一样的。但是，这些特征忽略了区域的非矩形形状。两个区域可能边界框很类似，但几乎没有重叠。因此，第二种策略(fg)只在区域的前景mask上计算CNN特征。我们将背景替换为输入的均值，这样背景在进行减去均值的操作后就成为0了。第三中策略(full+fg)仅仅将full特征和fg特征拼接起来；我们的试验验证了它们的互补性。

**Results on VOC 2011**. Table 5 shows a summary of our results on the VOC 2011 validation set compared with $O_2P$. (See Appendix E for complete per-category results.) Within each feature computation strategy, layer fc 6 always outperforms fc 7 and the following discussion refers to the fc 6 features. The fg strategy slightly outperforms full, indicating that the masked region shape provides a stronger signal, matching our intuition. However, full+fg achieves an average accuracy of 47.9%, our best result by a margin of 4.2% (also modestly outperforming $O_2P$), indicating that the context provided by the full features is highly informative even given the fg features. Notably, training the 20 SVRs on our full+fg features takes an hour on a single core, compared to 10+ hours for training on $O_2P$ features.

**在VOC 2011上的结果**。表5所示的是我们的方法与$O_2P$在VOC 2011验证集上的结果对比（见附录E的完整每类结果）。在每种特征计算策略中，fc 6层的结果一直超过fc 7层，所以下面的讨论都是指fc 6层的特征。fg策略略微比full策略好一些，表明masked区域形状提供了更强的信号，这与直觉符合。但是full+fg得到了平均准确率47.9%，我们的最好结果还要再高4.2%（略好与$O_2P$），表明full特征提供的上下文即使在给定fg特征的情况下也是很有信息量的。注意，在我们的full+fg特征上训练20个SVR的时间只需要一个小时，而$O_2P$特征上训练要10+个小时。

Table 5: Segmentation mean accuracy (%) on VOC 2011 validation. Column 1 presents O 2 P; 2-7 use our CNN pre-trained on ILSVRC 2012.

| | full R-CNN | fg R-CNN | full+fg R-CNN
--- | --- | --- | --- | ---
$O_2P$ [4] | fc6 fc7 | fc6 fc7 | fc6 fc7
46.4 | 43.0 42.5 | 43.7 42.1 | 47.9 45.8

In Table 6 we present results on the VOC 2011 test set, comparing our best-performing method, fc 6 (full+fg), against two strong baselines. Our method achieves the highest segmentation accuracy for 11 out of 21 categories, and the highest overall segmentation accuracy of 47.9%, averaged across categories (but likely ties with the $O_2P$ result under any reasonable margin of error). Still better performance could likely be achieved by fine-tuning.

表6中我们给出在VOC 2011测试集上的结果，将我们表现最好的方法，fc6(full+fg)，与两种很强的基准进行比较。我们的方法在21个类别中的11个得到了最高的分割准确率，和总体分割准确率47.9%，这是在所有类别中平均得到的。如果进行精调，还可能得到更好的结果。

Table 6: Segmentation accuracy (%) on VOC 2011 test. We compare against two strong baselines: the “Regions and Parts” (R&P) method of [2] and the second-order pooling ($O_2P$) method of [4]. Without any fine-tuning, our CNN achieves top segmentation performance, outperforming R&P and roughly matching $O_2P$.

## 6. Conclusion 结论

In recent years, object detection performance had stagnated. The best performing systems were complex ensembles combining multiple low-level image features with high-level context from object detectors and scene classifiers. This paper presents a simple and scalable object detection algorithm that gives a 30% relative improvement over the best previous results on PASCAL VOC 2012.

近年来，目标检测性能停滞了。最佳表现的系统都是将多种底层图像特征和目标检测器和场景分类器的高层上下文结合的复杂集成系统。本文提出了一种简单可扩展的目标检测算法，将在PASCAL VOC 2012上得到的最佳结果相对提升了30%。

We achieved this performance through two insights. The first is to apply high-capacity convolutional neural networks to bottom-up region proposals in order to localize and segment objects. The second is a paradigm for training large CNNs when labeled training data is scarce. We show that it is highly effective to pre-train the network -with supervision- for a auxiliary task with abundant data (image classification) and then to fine-tune the network for the target task where data is scarce (detection). We conjecture that the “supervised pre-training/domain-specific fine-tuning” paradigm will be highly effective for a variety of data-scarce vision problems.

我们通过两种思想得到这种结果。第一是将高容量的卷积神经网络用于自下而上的候选区域，以定位并分割目标。第二是当训练数据稀少时训练大型CNN的方案。我们说明了，首先将网络进行有监督的预训练，通常都是有大量数据的辅助任务（图像分类，然后对目标任务进行精调网络，其中数据很少（检测）。我们推测这种“有监督预训练/领域相关的精调”方案会在大量数据稀缺的视觉任务中得到高效应用。

We conclude by noting that it is significant that we achieved these results by using a combination of classical tools from computer vision and deep learning (bottom-up region proposals and convolutional neural networks). Rather than opposing lines of scientific inquiry, the two are natural and inevitable partners.

通过将计算机视觉中的经典工具和深度学习结合起来（自下而上的区域候选和卷积神经网络）得到这种结果是有重大意义的，这种两个自然和不可避免的伙伴。

## Appendix 附录

### A. Object proposal transformations 候选目标变换

### B. Positive vs. negative examples and softmax 正负样本和softmax

### C. Bounding-box regression 边界框回归

### D. Additional feature visualizations 额外的特征可视化

### E. Per-category segmentation results 类前分割结果

### F. Analysis of cross-dataset redundancy 跨数据集冗余性分析