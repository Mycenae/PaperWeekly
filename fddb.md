# FDDB: A Benchmark for Face Detection in Unconstrained Settings (2010)

Vidit Jain et al. University of Massachusetts Amherst

## Abstract 摘要

Despite the maturity of face detection research, it remains difficult to compare different algorithms for face detection. This is partly due to the lack of common evaluation schemes. Also, existing data sets for evaluating face detection algorithms do not capture some aspects of face appearances that are manifested in real-world scenarios. In this work, we address both of these issues. We present a new data set of face images with more faces and more accurate annotations for face regions than in previous data sets. We also propose two rigorous and precise methods for evaluating the performance of face detection algorithms. We report results of several standard algorithms on the new benchmark.

尽管人脸检测研究已经有了一定的成熟度，但对不同的人脸检测算法进行比较仍然比较困难。这部分是由于缺少通用的评估方案。同时，现有的评估人脸检测算法的数据集没有体现出真实世界场景中人脸外表的某些部分。在本文中，我们同时处理这两个问题。我们提出一种新的人脸图像数据集，人脸数量更多，人脸区域的标注比之前的数据集更精确。我们还提出了两种严密精确的评估人脸检测算法的方法。我们在新的测试标准上给出了几种标准算法的结果。

## 1. Introduction 引言

Face detection has been a core problem in computer vision for more than a decade. Not only has there been substantial progress in research, but many techniques for face detection have also made their way into commercial products such as digital cameras. Despite this maturity, algorithms for face detection remain difficult to compare, and are somewhat brittle to the specific conditions under which they are applied. One difficulty in comparing different face detection algorithms is the lack of enough detail to reproduce the published results. Ideally, algorithms should be published with sufficient detail to replicate the reported performance, or with an executable binary. However, in the absence of these alternatives, it is important to establish better benchmarks of performance.

十几年来，人脸检测一直是计算机视觉的一个核心问题。在研究上有重大进展，也有很多人脸检测技术已经得到了商用，如数字相机。尽管人脸检测算法成熟，但仍然难于比较，而且一定程度上难以应用到特定的情况。比较不同的人脸检测算法的一个困难是，缺少足够的细节来重现发表的结果。理想情况下，算法发表应当有足够的细节来复现发表的表现，或者有一个可执行的二进制文件。但是，在没有这些的情况下，建立更好的测试基准就非常重要了。

For a data set to be useful for evaluating face detection, the locations of all faces in these images need to be annotated. Sung et al. [24] built one such data set. Although this data set included images from a wide range of sources including scanned newspapers, all of the faces appearing in these images were upright and frontal. Later, Rowley et al. [18] created a similar data set with images that included faces with in-plane rotation. Schneiderman et al. [20, 21] combined these two data sets with an additional collection of profile face images, which is commonly known as the MIT+CMU data set. Since this resulting collection contains only grayscale images, it is not applicable for evaluating face detection systems that employ color information as well [6]. Some of the subsequent face detection data sets included color images, but they also had several shortcomings. For instance, the GENKI data set [25] includes color images that show a range of head poses (yaw, pitch ±45◦. roll ±20◦), but every image in this collection contains exactly one face. Similarly, the Kodak [13], UCD [23] and VT-AAST [1] data sets included images of faces with occlusions, but the small sizes of these data sets limit their utility in creating effective benchmarks for face detection algorithms.

一个数据集要对评估人脸检测算法有用，图像中所有人脸的位置需要标注出来。Sung等[24]建立了一个这样的数据集。虽然这个数据集包括的图像来源很广，比如扫描的报纸，但这些中出现的所有人脸都是竖直的、正面图像。后来，Rowley等[18]创建了一个类似的数据集，包含了人脸的平面内旋转。Schneiderman等[20,21]综合了这两个数据集，并增加了另外的侧脸图像，这个数据集被称为MIT+CMU数据集。因为这个数据集只有灰度图像，所以不能评估处理彩色信息的人脸检测系统[6]。后来的一些人脸检测数据集包含了彩色图像，但也有一些缺点。比如，GENKI数据集[25]包含的彩色人脸姿态处于一定范围内(yaw, pitch ±45◦. roll ±20◦)，但数据集中的每幅图像都只包含一张脸。类似的，the Kodak[13]，UCD[23]和VT-AAST[1]数据集包含了遮挡部分人脸的图像，但这些数据集的规模都很小，限制了其创建高效的人脸检测算法基准测试的工具性能。

One contribution of this work is the creation of a new data set that addresses the above-mentioned issues. Our data set includes 本文的一项贡献就是创建了一个新的数据集，解决了上述问题。我们的数据集包括：

- 2845 images with a total of 5171 faces; 2845幅图像，共5171张人脸；

- a wide range of difficulties including occlusions, difficult poses, and low resolution and out-of-focus faces; 包含大量特殊情况如遮挡，困难姿态，低分辨率和失焦人脸；

- the specification of face regions as elliptical regions; and 人脸的椭圆形区域描述；

- both grayscale and color images. 既有灰度图像，也有彩色图像。

Another limitation of the existing benchmarks is the lack of a specification for evaluating the output of an algorithm on a collection of images. In particular, as noted by Yang et al. [28], the reported performance measures depend on the definition of a “correct” detection result. The definition of correctness can be subtle. For example, how should we score an algorithm which provides two detections, each of which covers exactly 50% of a face region in an image? Since the evaluation process varies across the published results, a comparison of different algorithms remains difficult. We address this issue by presenting a new evaluation scheme with the following components:

现有的测试基准的另一个局限是缺少评估一个算法处理一个图像集合的输出的具体指标。具体来说，Yang等[28]指出，性能的衡量依赖于“正确”的检测结果的定义。正确的定义可能非常微妙。比如，如果一个算法给出两个检测结果，两个结果每个都检测到了50%的人脸区域，那么怎样评价这个算法呢？由于评估过程在很多发表的结果中都不一样，所以不同算法的比较仍然很困难。我们提出了一套新的评估方案，解决了这个问题，方案包括如下部分：

- An algorithm to find correspondences between a face detector’s output regions and the annotated face regions. 匹配人脸检测输出区域与标注的人脸区域的算法；

- Two separate rigorous and precise methods for evaluating any algorithm’s performance on the data set. These two methods are intended for different applications. 两种不同的方法，都可以严密精确的评估任何算法在数据集上的性能；

- Source code for implementing these procedures. 实现这些目标的源代码。

We hope that our new data set, the proposed evaluation scheme, and the publicly available evaluation software will make it easier to precisely compare the performance of algorithms, which will further prompt researchers to work on more difficult versions of the face detection problem.

我们希望这个新数据集、提出的评估方案和公开发布的评估软件可以使不同算法性能的比较更加容易，这可以推动研究者发展出人脸检测算法更加高级的版本。

The report is organized as follows. In Section 2, we discuss the challenges associated with comparing different face detection approaches. In Section 3, we outline the construction of our data set. Next, in Section 4, we describe a semi-automatic approach for removing duplicate images in a data set. In Section 5, we present the details of the annotation process, and finally in Section 6, we present our evaluation scheme.

本报告组织如下。在第2节，我们讨论了比较不同的人脸检测算法的困难。在第3部分，我们给出数据集的概要轮廓。在第4部分，我们描述了一种从数据集中去除重复图像的半自动方法。在第5部分，我们给出了标注过程的细节，最后在第6部分，我们给出了我们的评估方案。

## 2. Comparing face detection approaches 人脸检测方法的比较

Based of the range of acceptable head poses, face detection approaches can be categorized as 基于可能的头部姿态范围，人脸检测方法可以分类为：

- single pose: the head is assumed to be in a single, up-right pose (frontal [24, 18, 26] or profile [21]); 单姿态：假设头部是单一的、竖直的姿态（正面视图[24,18,26]或侧面视图[21]）；

- rotation-invariant: in-plane rotations of the head are allowed [8, 19]; 旋转不变的：允许头部的平面内旋转[8,19]；

- multi-view: out-of-plane rotations are binned into a pre-determined set of views [7, 9, 12]; 多视角：异面旋转分类成预定义的视角集[7,9,12]；

- pose-invariant: no restrictions on the orientation of the head [16, 22]. 对姿态不变：对头部的方向没有限制[16,22]。

Moving forward from previous comparisons [28] of approaches that focus on limited head orientations, we intend to evaluate different approaches for the most general, i.e., the pose-invariant, face detection task.

之前的比较方法[28]只关注有限的头部方向，我们从这出发，提出了评估不同方法的最一般的方案，即与姿态无关的人脸检测任务评估方法。

One challenge in comparing face detection systems is the lack of agreement on the desired output. In particular, while many approaches specify image regions – e.g., rectangular regions [26] or image patches with arbitrary shape [17] – as hypotheses for face regions, others identify the locations of various facial landmarks such as the eyes [27]. Still others give an estimate of head pose [16] as well.

对比人脸检测系统的一个挑战是，对期望输出没有一致意见。特别是，很多方法指定图像区域（即矩形区域[26]或任意形状的图像块[17]）作为人脸区域的假设，其他的会识别出各种人脸特征点的位置，如眼睛[27]。还有其他的会给出头部的姿态的估计[16]。

The scope of this work is limited to the evaluation of region-based output alone (although we intend to follow this report in the near future with a similar evaluation of 3D pose estimation algorithms). To this end, we annotate each face region with an ellipse of arbitrary size, shape, and orientation, showing the approximate face region for each face in the image. Compared to the traditional rectangular annotation of faces, ellipses are generally a better fit to face regions and still maintain a simple parametric shape to describe the face. We discuss the details of the annotation process in Section 5. Note that our data set is amenable to any additional annotations including facial landmarks and head pose information, which would be beneficial for benchmarking the next generation of face detection algorithms.

本文的工作限定在只评估基于区域的输出（我们在将来会给出类似的3D姿态估计算法的评估）。为此，我们将每个人脸标注成一个椭圆，大小、形状和方向都任意，给出图像中每个人脸的近似区域。与传统的人脸矩形标注相比，椭圆一般来说更匹配人脸区域，同时其参数化表示仍然比较简单。我们讨论了在第5部分详细讨论标注过程。注意我们的数据集是可以增加额外的标注的，包括人脸特征点和头部姿态信息，这对于对下一代人脸检测算法的基准测试是有好处的。

Next we discuss the origins and construction of our database. 下一步我们讨论数据集的起源和构建。

## 3. FDDB: Face Detection Data set and Benchmark

Berg et al. [2] created a data set that contains images and associated captions extracted from news articles (see Figure 1). The images in this collection display large variation in pose, lighting, background and appearance. Some of these variations in face appearance are due to factors such as motion, occlusions, and facial expressions, which are characteristic of the unconstrained setting for image acquisition. The annotated faces in this data set were selected based on the output of an automatic face detector. An evaluation of face detection algorithms on the existing set of annotated faces would favor the approaches with outputs highly correlated with this base detection algorithm. This property of the existing annotations makes them unsuitable for evaluating different approaches for face detection. The richness of the images included in this collection, however, motivated us to build an index of all of the faces present in a subset of images from this collection. We believe that benchmarking face detection algorithms on this data set will provide good estimates of their expected performance in unconstrained settings.

Berg等[2]创建了一个数据集，包括了从新闻文章里提取出的图像及相关的标题（见图1）。数据集中的图像的姿态、光照、背景和外貌变化很大。人脸外貌的一些变化是由于运动、遮挡和人脸表情，这是不受限制的图像获取的特征。这个数据集中标注的人脸的选择是基于一个自动人脸检测器的输出。已有集合上人脸检测算法的评估，就会倾向于那些与基准检测算法输出高度相关的算法。现有标注的这个性质使其不适合评估不同的人脸检测算法。这个集合中包含的图像的丰富性，推动我们建立一个其图像子集中所有人脸的索引。我们相信这个集合上的人脸检测算法基准测试，会成为不受限制的设置中的很好的估计。

Figure 1. Example images from Berg et al.’s data set.

### 3.1. Construction of the data set 数据集的构建

The images in Berg et al.’s data set were collected from the Yahoo! news website, which accumulates news articles from different sources. Although different news organizations may cover a news event independently of each other, they often share photographs from common sources such as the Associated Press or Reuters. The published photographs, however, may not be digitally identical to each other because they are often modified (e.g., cropped or contrast-corrected) before publication. This process has led to the presence of multiple copies of near-duplicate images in Berg et al.’s data set. Note that the presence of such near-duplicate images is limited to a few data collection domains such as news photos and those on the internet, and is not a characteristic of most practical face detection application scenarios. For example, it is uncommon to find near-duplicate images in a personal photo collection. Thus, an evaluation of face detection algorithms on a data set with multiple copies of near-duplicate images may not generalize well across domains. For this reason, we decided to identify and remove as many near duplicates from our collection as possible. We now present the details of the duplicate detection.

Berg等人的数据集的图像是从Yahoo! news网站上收集的，从不同的源中积累新闻文章。虽然不同的新闻组织可以相互独立的报道一个新闻事件，但他们通常会共用一些图片，这些图片是从共有的源得来的，如Associated Press或Reuters。发表的图片不会完全一样，因为发表前通常会做一些修改（如剪切，或对比度修正）。这个过程导致了在Berg等人的数据集中存在同一幅图像的多个拷贝或近似重复图像。注意这些近似重复图像的存在只在于几个子集中，如新闻图像和网络上的图像，这并不是大多数实际中的人脸检测应用场景中的特征。比如，在个人相册中很难找到近似重复的图像。所以，数据集中包含图像的近似重复的多个拷贝，人脸检测算法在这样的数据集中的评估，可能泛化能力不会很好。基于这个原因，我们决定从数据集中尽可能识别并去除这些近似重复图像。现在我们给出重复检测的算法细节。

Figure 2. Outline of the labeling process. Semi-automatic approaches are developed for both of these steps.

Original Collection -> Near-duplicate Detection -> Ellipse Fitting

## 4. Near-duplicate detection 近似重复的检测

We selected a total of 3527 images (based on the chronological ordering) from the image-caption pairs of Berg et al. [2]. Examining pairs for possible duplicates in this collection in the naı̈ve fashion would require approximately 12.5 million annotations. An alternative arrangement would be to display a set of images and manually identify groups of images in this set, where images in a single group are near-duplicates of each other. Due to the large number of images in our collection, it is unclear how to display all the images simultaneously to enable this manual identification of near-duplicates in this fashion.

我们从Berg等[2]人的图像-标题对中选择了总计3527幅图像（按照时间先后顺序）。以一种单纯的方式检查这个集合中的可能的重复对，会需要大约1250万标注。另一种安排可以显示图像集，并手动识别出集合中的近似重复图像。由于我们集合中图像数量巨大，还不太清楚如何同时显示这么多图像并手动识别这些近似重复图像。

Identification of near-duplicate images has been studied for web search [3, 4, 5]. However, in the web search domain, scalability issues are often more important than the detection of all near-duplicate images in the collection. Since we are interested in discovering all of the near-duplicates in our data set, these approaches are not directly applicable to our task. Zhang et al. [29] presented a more computationally intensive approach based on stochastic attribute relational graph (ARG) matching. Their approach was shown to perform well on a related problem of detecting near-identical frames in news video databases. These ARGs represent the compositional parts and part-relations of image scenes over several interest points detected in an image. To compute a matching score between the ARGs constructed for two different images, a generative model for the graph transformation process is employed. This approach has been observed to achieve high recall of near-duplicates, which makes it appropriate for detecting similar images in our data set.

识别近似重复图像已经在网络搜索中进行了研究[3,4,5]。但是，在网络搜索领域，可扩展性问题通常比检测所有的近似重复图像更重要。由于我们想要找到数据集中所有的近似重复图像，所以这些方法不能直接用于我们的任务。Zhang等[29]提出了一个计算量更大的基于随机属性关系图(ARG)匹配。他们的方法在一个相关的问题上表现良好，即在新闻视频数据集中检测接近相同的帧。这些ARGs代表了图像中检测到的感兴趣点的组成部分和各部分关系。为计算两幅不同图像间的ARGs间的匹配分数，采用了一个图变换的生成式模型。这种方法可以得到接近重复样本的高召回率，这在我们的数据集上的就很适合来检测类似的图像。

As with most automatic approaches for duplicate detection, this approach has a trade-off among false positives and false negatives. To restrict the number of false positives, while maintaining a high true positive rate, we follow an iterative approach (outlined in Algorithm 1) that alternates between clustering and manual inspection of the clusters. We cluster (steps 3-5 of Algorithm 1) using a spectral graph-clustering approach [15]. Then, we manually label each non-singleton cluster from the preceding step as either uniform, meaning that it contains images that are all near duplicates of each other, or non-uniform, meaning that at least one pair of images in the cluster are not near duplicates of each other. Finally, we replace each uniform cluster with one of the images belonging to it.

对于多数重复检测的自动方法来说，这种方法在false positive和false negative中有一个折中。为限制false positive的数量，同时保持较高的true positive率，我们采用了一种迭代方法（如算法1所示），不断的进行聚类及手工检查聚类。我们使用一种谱系图聚类方法[15]进行聚类（算法1的步骤3-5）。然后，我们手工的标记上一步骤中得到的每个非单个的聚类，如果包括的图像都是互相的近似重复，那么就是uniform，如果至少一对图像不是近似重复，就标记成non-uniform。最后，我们将每个uniform聚类替换为其中的一幅图像。

For the clustering step, in particular, we construct a fully-connected undirected graph G over all the images in the collection, where the ARG-matching scores are used as weights for the edges between each pair of images. Following the spectral graph-clustering approach [15], we compute the (unnormalized) Laplacian $L_G$ of graph G as

特别的，对于聚类步骤，我们构建一个集合中所有图像的全连接无向图G，使用ARG-匹配分数作为每对图像的边的权重。使用[15]的谱系图聚类算法，我们计算图G的Laplacian $L_G$（未归一化）：

$$L_G = diag(d) − W_G$$(1)

where d is the set of degrees of all the nodes in G, and $W_G$ is the adjacency matrix of G. A projection of the graph G into a subspace spanned by the top few eigenvectors of $L_G$ provides an effective distance metric between all pairs of nodes (images, in our case). We perform mean-shift clustering with a narrow kernel in this projected space to obtain clusters of images.

其中d是G中的所有节点的度，$W_G$是G的邻接矩阵。图G投影到一个由$L_G$的最高几个特征向量支撑的子空间，可以给出所有节点（在我们的情况中就是图像）间的有效距离。我们在这个投影空间中使用一个窄核进行mean-shift聚类，以得到图像的聚类。

Algorithm 1 Identifying near-duplicate images in a collection 算法1 识别集合中的近似重复图像

1: Construct a graph G = {V, E}, where V is the set of images, and E are all pairwise edges with weights as the ARG matching scores.

2: repeat

3: Compute the Laplacian of G, $L_G$.

4: Use the top m eigenvectors of $L_G$ to project each image onto $R^m$.

5: Cluster the projected data points using mean-shift clustering with a small-width kernel.

6: Manually label each cluster as either uniform or non-uniform.

7: Collapse the uniform clusters onto their centroids, and update G.

8: until none of the clusters can be collapsed.

Using this procedure, we were able to arrange the images according to their mutual similarities. Annotators were asked to identify clusters in which all images were derived from the same source. Each of these clusters was replaced by a single exemplar from the cluster. In this process we manually discovered 103 uniform clusters over seven iterations, with 682 images that were near-duplicates. Additional manual inspections were performed to find an additional three cases of duplication.

使用这个程序，我们可以依据其之间的相似度来重新对图像进行安排。要求标注者识别聚类中所有的图像都是相同的源的情况。这样的聚类都替换成其中的一个样本。在这个过程中，我们在7次迭代后手工发现了103个uniform聚类，682幅图像是近似重复的。还有另外的手工检查工作，发现了另外三种重复的情况。

Next we describe our annotation of face regions. 下面我们描述人脸区域的标注。

Figure 3. Near-duplicate images. (Positive) The first two images differ from each other slightly in the resolution and the color and intensity distributions, but the pose and expression of the faces are identical, suggesting that they were derived from a single photograph. (Negative) In the last two images, since the pose is different, we do not consider them as near-identical images.

## 5. Annotating face regions 标注人脸区域

As a preliminary annotation, we drew bounding boxes around all the faces in 2845 images. From this set of annotations, all of the face regions with height or width less than 20 pixels were excluded, resulting in a total of 5171 face annotations in our collection.

作为初步标注，我们对2845幅图像中的所有人脸都画了边界框。这种标注集合中，所有高度或宽度小于20像素的人脸区域都被排除掉，最后得到集合中的共计5171个人脸标注。

Figure 4. Challenges in face labeling. For some image regions, deciding whether or not it represents a “face” can be challenging. Several factors such as low resolution (green, solid), occlusion (blue, dashed), and pose of the head (red, dotted) may make this determination ambiguous.

For several image regions, the decision of labeling them as face regions or non-face regions remains ambiguous due to factors such as low resolution, occlusion, and head-pose (e.g., see Figure 4). One possible approach for handling these ambiguities would be to compute a quantitative measure of the “quality” of the face regions, and reject the image regions with the value below a pre-determined threshold. We were not able, however, to construct a satisfactory set of objective criteria for making this determination. For example, it is difficult to characterize the spatial resolution needed to characterize an image patch as a face. Similarly, for occluded face regions, while a threshold based on the fraction of the face pixels visible could be used as a criterion, it can be argued that some parts of the face (e.g., eyes) are more informative than other parts. Also, note that for the current set of images, all of the regions with faces looking away from the camera have been labeled as non-face regions. In other words, the faces with the angle between the nose (specified as radially outward perpendicular to the head) and the ray from the camera to the person’s head is less than 90 degrees. Estimating this angle precisely from an image is difficult.

对于一些图像区域，是否将其标注为人脸区域或非人脸区域，会由于一些因素如低分辨率、遮挡和头部姿态，而使决定变得困难（如图4）。处理这些模糊情况的一种可能方法是计算人脸区域质量的量化度量，并拒绝那些值低于预定义的阈值的区域。但是，我们无法给出一种令人满意的目标准则来作出这种区分。比如，很难确定所需的空间分辨率来很好的表现一张人脸。类似的，对于遮挡的人脸区域，虽然可以使用基于可见的人脸部分作为一个准则，但也可以说，人脸的一些部分（如眼睛）比其他部分包含更多信息。同时，对于现在的图像集，没有看镜头的所有人脸，都被标注成了非人脸区域。换句话说，鼻子（与头部垂直向外的方向）与镜头人头之间的角度要小于90度。从一幅图像中估算这个角度非常困难。

Due to the lack of an objective criterion for including (or excluding) a face region, we resort to human judgments for this decision. Since a single human decision for determining the label for some image regions is likely to be inconsistent, we used an approach based on the agreement statistics among multiple human annotators. All of these face regions were presented to different people through a web interface to obtain multiple independent decisions about the validity of these image regions as face regions. The annotators were instructed to reject the face regions for which neither of the two eyes (or glasses) were visible in the image. They were also requested to reject a face region if they were unable to (qualitatively) estimate its position, size, or orientation. The guidelines provided to the annotators are described in Appendix A.

由于缺少标注（或排除）一个人脸区域的目标原则，我们就依靠人的判断。由于一个人确定是否标注一些图像区域可能不太可靠，我们采用多个标注者的共同决定。所有的人脸区域都通过网络接口送给不同的标注者，以得到多个独立的决定，确定这些图像区域是否是人脸区域。标注者被告知的原则是，如果两个人眼（或眼镜）在图像中都不可见，那么就判断不是人脸区域。如果不能定性的估计人脸的位置、大小或方向，也要拒绝认为这是人脸区域。给标注者提供的准则在附录A中。

### 5.1. Elliptical Face Regions 椭圆人脸区域

As shown in Figure 5, the shape of a human head can be approximated using two three-dimensional ellipsoids. We call these ellipsoids the vertical and horizontal ellipsoids. Since the horizontal ellipsoid provides little information about the features of the face region, we estimate a 2D ellipse for the orthographic projection of the hypothesized vertical ellipsoid in the image plane. We believe that the resulting representation of a face region as an ellipse provides a more accurate specification than a bounding box without introducing any additional parameters.

如图5所示，人头部的形状可以使用两个三维椭球进行近似。我们称这两个椭球为竖直椭球和水平椭球。由于水平椭球提供的关于人脸区域的特征信息非常少，我们对竖直椭球在图像平面上的正交投影估计一个2D椭圆以近似。我们相信将人脸区域表示成一个椭圆，会比边界框更精确，而且没有引入额外的参数。

Figure 5. Shape of a human head. The shape of a human head (left) can be approximated as the union of two ellipsoids (right). We refer to these ellipses as vertical and horizontal ellipsoids.

图5. 人头部的形状。人头部的形状（左）可以近似看作两个椭球（右）的交集。我们称这两个椭球为水平椭球和竖直椭球。

We specified each face region using an ellipse parameterized by the location of its center, the lengths of its major and minor axes, and its orientation. Since a 2D orthographic projection of the human face is often not elliptical, fitting an ellipse around the face regions in an image is challenging. To make consistent annotations for all the faces in our data set, the human annotators are instructed to follow the guidelines shown in Figure 6. Figure 7 shows some sample annotations. The next step is to produce a consistent and reasonable evaluation criterion.

我们将人脸区域表示成椭圆，参数为中心点位置，主轴和次轴的长度，及其方向。由于人脸的2D正交投影通常不是标准椭圆，将椭圆与人脸区域进行匹配还是有挑战性的。为对数据集中的所有人脸进行一致的标注，给标注者的标注准则如图6所示。图7给出了一些标注例子。下一步是创建一致合理的评估准则。

Figure 6. Guidelines for drawing ellipses around face regions. The extreme points of the major axis of the ellipse are respectively matched to the chin and the topmost point of the hypothetical vertical ellipsoid used for approximating the human head (see Figure 5). Note that this ellipse does not include the ears. Also, for a non-frontal face, at least one of the lateral extremes (left or right) of this ellipse are matched to the boundary between the face region and the corresponding (left or right) ear. The details of our specifications are included in Appendix A.

图6. 在人脸区域周围画椭圆的准则。椭圆主轴的端点分别匹配到下巴，与竖直椭球的上顶点。注意这个椭圆不包括耳朵。同时，对于非正面视角的人脸，椭圆的水平端点至少一个是人脸区域与对应耳朵的边界。规范的细节见附录A。

Figure 7. Sample Annotations. The two red ellipses specify the location of the two faces present in this image. Note that for a non-frontal face (right), the ellipse traces the boundary between the face and the visible ear. As a result, the elliptical region includes pixels that are not a part of the face.

图7. 标注的例子。两个红色椭圆指定了图像中两张人脸的具体位置。注意，对于非正面视角的人脸（右），椭圆到了人脸和可见耳朵的边界。结果是，椭圆区域包括了一部分非人脸的像素。

## 6. Evaluation 评估

To establish an evaluation criterion for detection algorithms, we first specify some assumptions we make about their outputs. We assume that： 为确定检测算法的评估标准，我们首先确定其输出的一些假设。我们假设：

- A detection corresponds to a contiguous image region. 检测结果对应一个连续的图像区域。

- Any post-processing required to merge overlapping or similar detections has already been done. 合并重叠区域或类似检测结果的后处理都已经进行完成。

- Each detection corresponds to exactly one entire face, no more, no less. In other words, a detection cannot be considered to detect two faces at once, and two detections cannot be used together to detect a single face. We further argue that if an algorithm detects multiple disjoint parts of a face as separate detections, only one of them should contribute towards a positive detection and the remaining detections should be considered as false positives.

- 每个检测结果对应一整个人脸，不多，也不少。还句话说，一个检测结果不要一次检测两张人脸，也不能使用两次检测来检测一张人脸。我们进一步要求，如果一个算法将人脸不相交的部分检测为多个结果，那么只有一个被认为是正确检测结果，其余的都认为是假检测。

To represent the degree of match between a detection $d_i$ and an annotated region $l_j$, we employ the commonly used ratio of intersected areas to joined areas:

为表示一个检测结果$d_i$和标注区域$l_j$的匹配程度，我们使用常用的交并比：

$$S(d_i,l_j)=\frac {area(d_i) ∩ area(l_j)} {area(d_i) ∪ area(l_j)}$$(2)

To specify a more accurate annotation for the image regions corresponding to human faces than is obtained with the commonly used rectangular regions, we define an elliptical region around the pixels corresponding to these faces. While this representation is not as accurate as a pixel-level annotation, it is a clear improvement over the rectangular annotations in existing data sets.

一般的标注都是矩形区域标注，为对人脸进行更精确的标注，我们定义了人脸区域的椭圆标注。虽然没有像素级标注精确，但明显比现有的矩形标注有所改进。

To facilitate manual labeling, we start with an automated guess about face locations. To estimate the elliptical boundary for a face region, we first apply a skin classifier on the image pixels that uses their hue and saturation values. Next, the holes in the resulting face region are filled using a flood-fill implementation in MATLAB. Finally, a moments-based fit is performed on this region to obtain the parameters of the desired ellipse. The parameters of all of these ellipses are manually verified and adjusted in the final stage.

为方便手工标注，我们先给出人脸位置的自动猜测。为估计人脸区域的椭圆边界，我们首先使用一个皮肤分类器，根据的是图像的hue值和saturation值。下一步，人脸区域中的孔洞由于Matlab中的flood-fill算法实现。最后，为估计椭圆的参数，使用了一种基于moments的匹配。在最后阶段，这些椭圆的参数经过人工核对和调整。

### 6.1. Matching detections and annotations 匹配检测结果与标注

A major remaining question is how to establish a correspondence between a set of detections and a set of annotations. While for very good results on a given image, this problem is easy, it can be subtle and tricky for large numbers of false positives or multiple overlapping detections (see Figure 8 for an example). Below, we formulate this problem of matching annotations and detections as finding a maximum weighted matching in a bipartite graph (as shown in Figure 9).

余下的一个主要问题是，怎么确定检测结果与标注的对应关系。对于给定图像上的很好的结果，这个问题非常简单，但如果检测结果中有很多false positves或很多重叠的检测结果（见图8中的例子），问题就会比较棘手。下面，我们将匹配标注与检测的问题，表述为一个双向图中寻找最大加权匹配的问题（如图9所示）。

Figure 8. Matching detections and annotations. In this image, the ellipses specify the face annotations and the five rectangles denote a face detector’s output. Note that the second face from left has two detections overlapping with it. We require a valid matching to accept only one of these detections as the true match, and to consider the other detection as a false positive. Also, note that the third face from the left has no detection overlapping with it, so no detection should be matched with this face. The blue rectangles denote the true positives and yellow rectangles denote the false positives in the desired matching.

Figure 9. Maximum weight matching in a bipartite graph. We make an injective (one-to-one) mapping from the set of detected image regions $d_i$ to the set of image regions $l_i$ annotated as face regions. The property of the resulting mapping is that it maximizes the cumulative similarity score for all the detected image regions.

图9. 双向图中的最大加权匹配问题。我们从检测到的图像区域$d_i$到标注的人脸区域$l_i$作一个单射(injective mapping)。得到的映射的性质是，可以最大化所有检测到的图像区域的累积相似度分数。

Let L be the set of annotated face regions (or labels) and D be the set of detections. We construct a graph G with the set of nodes V = L ∪ D. Each node $d_i$ is connected to each label $l_j ∈ L$ with an edge weight $w_{ij}$ as the score computed in Equation 2. For each detection $d_i ∈ D$, we further introduce a node $n_i$ to correspond to the case when this detection $d_i$ has no matching face region in L.

令L为标注的人脸区域集合，D为检测结果集合。我们构建一个图G，其节点集合为V = L ∪ D。每个节点$d_i$都连接到每个标注$l_j ∈ L$，这条边的权重为$w_{ij}$，由式(2)计算其分数。对于每个检测结果$d_i ∈ D$，我们进一步引入一个节点$n_i$，表示检测结果$d_i$在人脸区域集合L中没有匹配到人脸的情况。

A matching of detections to face regions in this graph corresponds to the selection of a set of edges M ⊆ E. In the desired matching of nodes, we want every detection to be matched to at most one labeled face region, and every labeled face region to be matched to at most one detection. Note that the nodes $n_k$ have a degree equal to one, so they can be connected to at most one detection through M as well. Mathematically, the desired matching M maximizes the cumulative matching score while satisfying the following constraints:

这个图中检测结果与人脸区域的匹配，对应着选择边的集合M ⊆ E。在理想的节点匹配中，我们希望每个检测结果都与最多一个标注的人脸区域进行匹配，每个标注的人脸区域最多与一个检测结果进行匹配。注意，节点$n_k$的度为1，所以它们也最多与一个检测结果匹配。数学上来说，理想的匹配M会最大化累计匹配分数，同时满足下述约束：

$$∀d ∈ D, ∃l ∈ {L ∪ N}, d →^M l$$(3)

$$∀l ∈ L, !∃d, d' ∈ D, d →^M l ∧ d' →^M l$$(4)

The determination of the minimum weight matching in a weighted bipartite graph has an equivalent dual formulation as finding the solution of the minimum weighted (vertex) cover problem on a related graph. This dual formulation is exploited by the Hungarian algorithm [11] to obtain the solution for the former problem. For a given image, we employ this method to determine the matching detections and ground-truth annotations. The resulting similarity score is used for evaluating the performance of the detection algorithm on this image.

在加权双向图中确定最小权重匹配，有一个等价对偶的描述，即在一个相关的图中找到最小加权覆盖问题。这个等价对偶描述由Hungarian算法[11]研究，以得到前述问题的解。对于一幅给定的图像，我们用这种方法来确定检测结果与真值标注的匹配。相似性分数的结果用于评估检测算法在这幅图像中的表现。

### 6.2. Evaluation metrics 评估标准

Let $d_i$ and $v_i$ denote the i-th detection and the corresponding matching node in the matching M obtained by the algorithm described in Section 6.1, respectively. We propose the following two metrics for specifying the score $y_i$ for this detection:

令$d_i$和$v_i$表示第i个检测结果和对应的匹配节点，采用6.1节的匹配算法得到。对于检测结果的匹配分数$y_i$，我们提出下属两种度量标准：

Discrete score (DS) : $y_i = δ_{S(d_i, v_i)>0.5}$.

Continuous score (CS): $y_i = S(d_i, v_i)$

For both of these choice of scoring the detections, we recommend analyzing the Receiver Operating Characteristic (ROC) curves to compare the performance of different approaches on this data set. Although comparing the area under the ROC curve is equivalent to a non-parametric statistical hypothesis test (Wilcoxon signed-rank test), it is plausible that the cumulative performances of none of the compared approaches is better than the rest with statistical significance. Furthermore, it is likely that for some range of performance, one approach could outperform another, whereas the relative comparison is reversed for a different range. For instance, one detection algorithm might be able to maintain a high level of precision for low recall values, but the precision drops sharply after a point. This trend may suggest that this detector would be useful for application domains such as biometrics-based access controls, which may require high precision values, but can tolerate low recall levels. The same detector may not be useful in a setting (e.g., surveillance) that would requires the retrieval of all the faces in an image or scene. Hence, the analysis of the entire range of ROC curves should be done for determining the strengths of different approaches.

对于这两种检测评分方法，我们推荐分析ROC曲线来比较不同方法在这个数据集上的性能。比较ROC曲线下的区域面积，与非参数统计假设测试(wilcoxon signed-rank test)是等价的，但被比较的方法的累积性能，在统计性能上都没有剩下的好，这是有可能的。而且，对于一些性能范围来说，很可能一种方法超过了另一种方法，但在另一个范围内结果是相反的。比如，一种检测算法可以精度很高，但召回率很低，但在一个点后精度下降明显。这种趋势说明，这种检测器对于基于生物特征识别的访问控制很有用处，这需要很高的精度值，但可以容忍很低的召回率值。同样的检测器对于另外的设置（如监控）则没有什么用处，需要对场景或图像中的所有人脸进行检索。所以，为确定不同方法的强度，需要分析全部范围的ROC。

## 7. Experimental Setup 试验设置

For an accurate and useful comparison of different approaches, we recommend a distinction based on the training data used for estimating their parameters. In particular, we propose the following experiments:

为精确比较不同方法，我们推荐基于估计其参数的训练数据的比较方法。特别的，我们提出下面的试验：

EXP-1: 10-fold cross-validation 10份交叉验证

For this experiment, a 10-fold cross-validation is performed using a fixed partitioning of the data set into ten folds. The cumulative performance is reported as the average curve of the ten ROC curves, each of which is obtained for a different fold as the validation set.

这个试验中，将数据集固定分成10份，在此之上进行10份交叉验证。累积性能就是10个ROC曲线的平均曲线，每条曲线中，不同的部分作为验证集。

EXP-2: Unrestricted training 不受限的训练

For this experiment, data outside the FDDB data set is permitted to be included in the training set. The above-mentioned ten folds of the data set are separately used as validation sets to obtain ten different ROC curves. The cumulative performance is reported as the average curve of these ten ROC curves.

在这个试验中，训练集可以包含FDDB数据集之外的数据。上面提到的数据集的10份轮流作为验证集，以得到不同的ROC曲线。累积性能就是这10条ROC曲线的均值。

## 8. Benchmark 基准测试

For a proper use of our data set, we provide the implementation (C++ source code) of the algorithms for matching detections and annotations (Section 6.1), and computing the resulting scores (Section 6.2) to generate the performance curves at http://vis-www.cs.umass.edu/fddb/results.html. To use our software, the user needs to create a file containing a list of the output of this detector.

为合理使用我们的数据集，我们给出了匹配检测结果和标注的算法的实现(C++源代码, 6.1节)，计算分数结果(6.2节)来生成性能曲线。为使用我们的软件，用户需要生成一个文件，包含检测器输出的列表。

The format of this input file is described in Appendix B. In Figure 10, we present the results for the following approaches for the above-mentioned EXP-2 experimental setting: 输入文件的格式如附录B所示。在图10中，我们给出了下面方法在EXP-2试验设置上的结果：

- Viola-Jones detector [26] – we used the OpenCV implementation of this approach. We set the scale-factor and minimum number of neighbors parameters to 1.2 and 0, respectively. 我们使用这种方法的OpenCV实现，我们设置其尺度因子和最小邻域数目为1.2和0。

- Mikolajczyk’s face detector [14] – we set the parameter for the minimum distance between eyes in a detected face to 5 pixels. 我们设置在检测到的人脸中双眼的最小距离为5像素。

- Kienzle et al.’s [10] face detection library (fdlib). 人脸检测库

Figure 10. FDDB baselines. These are the ROC curves for different face detection algorithms. Both of these scores (DS and CS) are described in Section 6.2, whereas the implementation details of these algorithms are included in Section 8.

As seen in Figure 10, the number of false positives obtained from all of these face detection systems increases rapidly as the true positive rate increases. Note that the performances of all of these systems on the new benchmark are much worse than those on the previous benchmarks, where they obtain less than 100 false positives at a true positive rate of 0.9. Also note that although our data set includes images of frontal and non-frontal faces, the above experiments are limited to the approaches that were developed for frontal face detection. This limitation is due to the unavailability of a public implementation of multi-pose or pose-invariant face detection system. Nevertheless, the new benchmark includes more challenging examples of face appearances than the previous benchmarks. We hope that our benchmark will further prompt researchers to explore new research directions in face detection.

如图10所示，当true positive率增加时，人脸检测系统检测得到的false positive也迅速增加。注意所有这些系统的性能在新的基准测试中比在之前的都要差好多，在之前的基准测试中，在true positive率0.9时，false positive不到100个。同时也要注意，虽然我们的数据集包括正面视角和非正面视角的人脸，上面的试验中所用的方法都是对正面人脸检测的开发的算法的。这种限制是因为，现在多姿态或对姿态不变的人脸检测系统都没有公开实现。即使这样，新的基准测试比之前的包括了更有挑战性的样本。我们希望我们的基准测试能进一步推动研究者探索人脸检测新的研究方向。

## A. Guidelines for annotating faces using ellipses

## B. Data formats