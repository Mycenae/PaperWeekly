# Selective Search for Object Recognition 目标识别中的选择性搜索算法

J.R.R. Uijlings et al. University of Trento, Italy/University of Amsterdam, the Netherlands

## Abstract

This paper addresses the problem of generating possible object locations for use in object recognition. We introduce Selective Search which combines the strength of both an exhaustive search and segmentation. Like segmentation, we use the image structure to guide our sampling process. Like exhaustive search, we aim to capture all possible object locations. Instead of a single technique to generate possible object locations, we diversify our search and use a variety of complementary image partitionings to deal with as many image conditions as possible. Our Selective Search results in a small set of data-driven, class-independent, high quality locations, yielding 99% recall and a Mean Average Best Overlap of 0.879 at 10,097 locations. The reduced number of locations compared to an exhaustive search enables the use of stronger machine learning techniques and stronger appearance models for object recognition. In this paper we show that our selective search enables the use of the powerful Bag-of-Words model for recognition. The Selective Search software is made publicly available.

本文要解决的是目标识别中的可能目标位置生成问题。我们提出了selective search算法，该算法结合了穷举式搜索和分割的能力。就像分割一样，我们用图像结构来指导我们的采样过程。与穷举式搜索类似，我们要得到所有可能的目标位置。我们没有使用单一方法产生可能的目标位置，而是使我们的搜索多样化，使用多种互补的图像划分方法来处理各种情况的图像。我们的selective search算法得到的目标位置集合小，而且这个集合是由数据驱动的，与类别无关，目标位置质量高，得到的10097个目标位置中，召回率99%(recall = TP/TP+FN, correctly identified objects/objects in samples)，Mean Average Best Overlap 0.879。与穷举式搜索算法相比，我们得到的目标位置数大大减少，所以可使用更强的机器学习算法和目标识别的appearance模型。本文中，我们可以看到，selective search得到的结果可以使用强大的Bag-of-Words识别模型。现在selective search软件已经公开可以取得。

## 1 Introduction

For a long time, objects were sought to be delineated before their identification. This gave rise to segmentation, which aims for a unique partitioning of the image through a generic algorithm, where there is one part for all object silhouettes in the image. Research on this topic has yielded tremendous progress over the past years [3, 6, 13, 26]. But images are intrinsically hierarchical: In Figure 1a the salad and spoons are inside the salad bowl, which in turn stands on the table. Furthermore, depending on the context the term table in this picture can refer to only the wood or include everything on the table. Therefore both the nature of images and the different uses of an object category are hierarchical. This prohibits the unique partitioning of objects for all but the most specific purposes. Hence for most tasks multiple scales in a segmentation are a necessity. This is most naturally addressed by using a hierarchical partitioning, as done for example by Arbelaez et al. [3].

很长时间里，目标在识别之前，要先确定其轮廓。这就引出了分割的问题，其目标是通过通用算法得到图像的唯一划分，使图像中所有目标的轮廓都包含其中。这个课题的研究在过去几年得到了巨大发展[3,6,13,26]。但图像在本质上是分层次的：在图1a中，沙拉和勺子在沙拉碗中，而碗又在桌子上。更进一步，根据图像中桌子的上下文，可以推断出只有木质桌子才能承载桌子上的所有东西。所以图像的本质和目标种类的不同使用都是分层次的。这使得几乎大多数特殊目的的目标的唯一划分都是不可能的。所以，对于大多数任务，多尺度分割是必须的。这主要通过层次化划分来自然的解决，如Arbelaez et al. [3]的例子。

Besides that a segmentation should be hierarchical, a generic solution for segmentation using a single strategy may not exist at all. There are many conflicting reasons why a region should be grouped together: In Figure 1b the cats can be separated using colour, but their texture is the same. Conversely, in Figure 1c the chameleon is similar to its surrounding leaves in terms of colour, yet its texture differs. Finally, in Figure 1d, the wheels are wildly different from the car in terms of both colour and texture, yet are enclosed by the car. Individual visual features therefore cannot resolve the ambiguity of segmentation.

分割除了是层次化的以外，单一策略分割的通用方案是不可能存在的。关于一个区域为什么应当被分成一组，有很多互相冲突的原因：在图1b中的猫可以用色彩来分开，但其纹理是一样的。相反的，在图1c中的变色龙与周围的叶子在颜色上是一样的，但其纹理不一样。最后，在图1d中，轮子与车子其他部分在颜色与纹理上都非常不一样，但却是车的一部分。所以单个的视觉特征不能解决分割的含糊问题。

And, finally, there is a more fundamental problem. Regions with very different characteristics, such as a face over a sweater, can only be combined into one object after it has been established that the object at hand is a human. Hence without prior recognition it is hard to decide that a face and a sweater are part of one object [29].

最后，还有一个更为基础的问题。非常不同特征的区域，比如毛衣上的脸，只有在确定要识别的目标是人之后，才能合并到这个目标中去。因此，在没有识别在前的话，很难确定脸部和毛衣是一个目标的一部分[29]。

This has led to the opposite of the traditional approach: to do localisation through the identification of an object. This recent approach in object recognition has made enormous progress in less than a decade [8, 12, 16, 35]. With an appearance model learned from examples, an exhaustive search is performed where every location within the image is examined as to not miss any potential object location [8, 12, 16, 35].

这就到了传统方法的反面：要通过目标识别来进行定位。目标识别方法在不到10年的时间内已经有了很大进步[8,12,16,35]。通过从样本中学习到的appearance模型，在图像中的每个位置都进行穷举式搜索，以免错过任何潜在的目标位置[8,12,16,35]。

However, the exhaustive search itself has several drawbacks. Searching every possible location is computationally infeasible. The search space has to be reduced by using a regular grid, fixed scales, and fixed aspect ratios. In most cases the number of locations to visit remains huge, so much that alternative restrictions need to be imposed. The classifier is simplified and the appearance model needs to be fast. Furthermore, a uniform sampling yields many boxes for which it is immediately clear that they are not supportive of an object. Rather than sampling locations blindly using an exhaustive search, a key question is: Can we steer the sampling by a data-driven analysis?

但是，穷举式搜索本身有几个缺陷。任何可能的位置都进行搜索，其计算量可能是不可行的。通过使用规则的网格、固定的尺度和固定的纵横比，可以缩小搜索空间。在大多数情况下，需要处理的位置数量仍然非常巨大，有时候需要加入其他限制条件。分类器是简化的，appearance模型需要是快速的。统一的采样得到了很多边界框，有的立刻就可以明确这不是一个目标。如果不使用穷举式搜索，盲目的进行采样定位，那么一个关键问题就是：我们可以转而采用数据驱动分析的采样吗？

In this paper, we aim to combine the best of the intuitions of segmentation and exhaustive search and propose a data-driven selective search. Inspired by bottom-up segmentation, we aim to exploit the structure of the image to generate object locations. Inspired by exhaustive search, we aim to capture all possible object locations. Therefore, instead of using a single sampling technique, we aim to diversify the sampling techniques to account for as many image conditions as possible. Specifically, we use a data-driven grouping-based strategy where we increase diversity by using a variety of complementary grouping criteria and a variety of complementary colour spaces with different invariance properties. The set of locations is obtained by combining the locations of these complementary partitionings. Our goal is to generate a class-independent, data-driven, selective search strategy that generates a small set of high-quality object locations.

在本文中，我们的目标是将最好的分割直觉与穷举式搜索结合起来，提出一种数据驱动的选择性搜索selective search算法。受自下而上分割的启发，我们通过图像的结构来产生目标位置。受穷举式搜索启发，我们要得到所有可能的目标位置。所以，我们不采用单一的取样技术，而采取多样化的取样技术，处理各种各样情况下的图像。特别的，我们使用数据驱动的基于分组的策略，其中我们通过使用多种互补的分组原则和多种互补的带有不同不变性性质的色彩空间来增加多样性。通过组合这些互补划分的位置，得到位置集合。我们的目标是生成类别无关的、数据驱动的选择性搜索策略，从而得到高质量的较小的目标位置集合。

Our application domain of selective search is object recognition. We therefore evaluate on the most commonly used dataset for this purpose, the Pascal VOC detection challenge which consists of 20 object classes. The size of this dataset yields computational constraints for our selective search. Furthermore, the use of this dataset means that the quality of locations is mainly evaluated in terms of bounding boxes. However, our selective search applies to regions as well and is also applicable to concepts such as “grass”.

我们的selective search应用范围是目标识别。所以我们在使用最多的数据集上评估算法，也就是包括20种目标类别的PASCAL VOC检测数据集。该数据集的规模要求我们的selective search有计算量限制。而且，使用这个数据集意味着位置数据的质量主要是以边界框来评估。但是，我们的selective search算法可以应用到区域中，也可以应用在“草”这样的概念中。

In this paper we propose selective search for object recognition. Our main research questions are: (1) What are good diversification strategies for adapting segmentation as a selective search strategy? (2) How effective is selective search in creating a small set of high-quality locations within an image? (3) Can we use selective search to employ more powerful classifiers and appearance models for object recognition?

本文中，我们对目标识别应用提出了selective search算法。我们主要的研究问题是：(1)对于将分割适应到selective search策略中，什么是好的多样化策略？(2)在一幅图像中生成较小的高质量位置集合，selective search效率有多高？(3)我们使用selective search后，是否可以在目标识别中使用更强大的分类器和appearance模型？

Figure 1: There is a high variety of reasons that an image region forms an object. In (b) the cats can be distinguished by colour, not texture. In (c) the chameleon can be distinguished from the surrounding leaves by texture, not colour. In (d) the wheels can be part of the car because they are enclosed, not because they are similar in texture or colour. Therefore, to find objects in a structured way it is necessary to use a variety of diverse strategies. Furthermore, an image is intrinsically hierarchical as there is no single scale for which the complete table, salad bowl, and salad spoon can be found in (a).

## 2 Related Work 相关的工作

We confine the related work to the domain of object recognition and divide it into three categories: Exhaustive search, segmentation, and other sampling strategies that do not fall in either category.

我们将相关的工作限定在目标识别领域中，并分为三类：穷举式搜索，分割和不属于这两类的其他取样策略。

### 2.1 Exhaustive Search 穷举式搜索

As an object can be located at any position and scale in the image, it is natural to search everywhere [8, 16, 36]. However, the visual search space is huge, making an exhaustive search computationally expensive. This imposes constraints on the evaluation cost per location and/or the number of locations considered. Hence most of these sliding window techniques use a coarse search grid and fixed aspect ratios, using weak classifiers and economic image features such as HOG [8, 16, 36]. This method is often used as a preselection step in a cascade of classifiers [16, 36].

由于目标可以在图像的任何位置，尺度也非常不确定，所以很自然需要搜索所有位置[8,16,36]。但是，视觉搜索空间是巨大的，穷举式搜索计算量非常之大。这就要求在每个位置上的评估以及要考虑到的位置数量上施加一定的限制。所以大多数滑动窗口方法都使用较粗糙的搜索网格，以及固定的纵横比，并使用弱分类器以及比较经济的图像特征，如HOG[8,16,36]。这种方法经常在级联分类器中用作预选择步骤[16,36]。

Related to the sliding window technique is the highly successful part-based object localisation method of Felzenszwalb et al. [12]. Their method also performs an exhaustive search using a linear SVM and HOG features. However, they search for objects and object parts, whose combination results in an impressive object detection performance.

与滑动窗口相关的技术是Felzenszwalb et al.非常成功的基于部件的目标定位方法[12]。他们的方法用一个线性SVM和HOG特征进行穷举式搜索。但是，他们搜索目标和目标部件，其组合得到了让人印象深刻的目标检测结果。

Lampert et al. [17] proposed using the appearance model to guide the search. This both alleviates the constraints of using a regular grid, fixed scales, and fixed aspect ratio, while at the same time reduces the number of locations visited. This is done by directly searching for the optimal window within the image using a branch and bound technique. While they obtain impressive results for linear classifiers, [1] found that for non-linear classifiers the method in practice still visits over a 100,000 windows per image.

Lampert et al. [17]提出使用appearance模型来指导搜索。这缓和了使用规则网格、固定尺度和固定纵横比的约束，同时减少了需要考虑的位置数量。这是通过使用 branch and bound 技术直接搜索图像内的最佳窗口实现的。这对于线性分类器得到了很好的结果，而[1]发现对于非线性分类器，使用这种方法仍然每幅图像要计算10万个窗口。

Instead of a blind exhaustive search or a branch and bound search, we propose selective search. We use the underlying image structure to generate object locations. In contrast to the discussed methods, this yields a completely class-independent set of locations. Furthermore, because we do not use a fixed aspect ratio, our method is not limited to objects but should be able to find stuff like “grass” and “sand” as well (this also holds for [17]). Finally, we hope to generate fewer locations, which should make the problem easier as the variability of samples becomes lower. And more importantly, it frees up computational power which can be used for stronger machine learning techniques and more powerful appearance models.

我们没有使用盲目穷举式搜索，也没有使用 branch and bound 搜索，我们提出一种selective search算法。我们使用潜在的图像结构来产生目标位置。与上述方法相比，这会生成一种完全与类别无关的的位置集合。而且，我们并不使用固定的纵横比，所以我们的方法并不限制目标，也可以找到像“草”或“沙子”这样的东西([17]也是这样)。最后，我们希望生成更少的位置，这会使得问题更简单，因为取样的变化更少。更重要的，这减少了计算量，所以可以使用更强的机器学习技术和更强的appearance模型。

### 2.2 Segmentation 分割

Both Carreira and Sminchisescu [4] and Endres and Hoiem [9] propose to generate a set of class independent object hypotheses using segmentation. Both methods generate multiple foreground/background segmentations, learn to predict the likelihood that a foreground segment is a complete object, and use this to rank the segments. Both algorithms show a promising ability to accurately delineate objects within images, confirmed by [19] who achieve state-of-the-art results on pixel-wise image classification using [4]. As common in segmentation, both methods rely on a single strong algorithm for identifying good regions. They obtain a variety of locations by using many randomly initialised foreground and background seeds. In contrast, we explicitly deal with a variety of image conditions by using different grouping criteria and different representations. This means a lower computational investment as we do not have to invest in the single best segmentation strategy, such as using the excellent yet expensive contour detector of [3]. Furthermore, as we deal with different image conditions separately, we expect our locations to have a more consistent quality. Finally, our selective search paradigm dictates that the most interesting question is not how our regions compare to [4, 9], but rather how they can complement each other.

Carreira and Sminchisescu [4] and Endres and Hoiem [9]提出用分割来生成与类别无关的目标集合的假设。两种方法都生成多个前景/背景分割，学习预测前景分割是完整目标的可能性，并用来对这个分割来进行评级。两种算法都很可能精确的描述图像中目标的轮廓，[19]确认了使用[4]可以在像素级的图像分类中得到目前最优的结果。在分割算法中，经常是单个很强的算法来确定好的区域，在这两种算法中也是这样。他们通过随机初始化很多前景和背景种子，来得到很多位置。与之相比，我们通过使用不同的分组策略和不同的表示来显式的处理不同情况的图像。这意味着较低的计算量，因为我们没有采用单独的最佳分割策略，比如使用非常好但很消耗计算量的边缘检测器[3]。而且，由我们分别处理不同情况的图像，我们希望我们的位置质量更加一致。最后，我们的selective search算法得到的最有趣的问题不是我们的区域与[4,9]相比怎样，而是它们怎样互补的。

Gu et al. [15] address the problem of carefully segmenting and recognizing objects based on their parts. They first generate a set of part hypotheses using a grouping method based on Arbelaez et al. [3]. Each part hypothesis is described by both appearance and shape features. Then, an object is recognized and carefully delineated by using its parts, achieving good results for shape recognition. In their work, the segmentation is hierarchical and yields segments at all scales. However, they use a single grouping strategy whose power of discovering parts or objects is left unevaluated. In this work, we use multiple complementary strategies to deal with as many image conditions as possible. We include the locations generated using [3] in our evaluation.

Gu et al. [15]处理的问题是基于目标部件来仔细分割、识别目标。他们首先使用基于Arbelaez et al. [3]分组方法生成了部件集假设。每个假设的部件由appearance和形状特征来描述。然后，使用目标部件来识别目标，并仔细形成目标轮廓，得到形状识别的好结果。在他们的工作中，分割是分层次的，得出各种尺度上的分割。但是，他们使用了单一的分组策略，发现部件或目标的能力没有进行评估。在本文中，我们使用多种互补的策略处理各种情况下的图像。我们在评估中包括了[3]生成的位置。

### 2.3 Other Sampling Strategies 其他取样策略

Alexe et al. [2] address the problem of the large sampling space of an exhaustive search by proposing to search for any object, independent of its class. In their method they train a classifier on the object windows of those objects which have a well-defined shape (as opposed to stuff like “grass” and “sand”). Then instead of a full exhaustive search they randomly sample boxes to which they apply their classifier. The boxes with the highest “objectness” measure serve as a set of object hypotheses. This set is then used to greatly reduce the number of windows evaluated by class-specific object detectors. We compare our method with their work.

Alexe et al. [2]处理穷举式搜索的巨大取样空间问题，提出搜索与类别无关的任何目标。在他们的方法中，他们对含有确定良好形状的目标（相反的是“草”或“沙”的对象），在目标窗口上训练了一个分类器。然后并不进行穷举式搜索，而是随机取样边界框，并应用分类器。有最高objectness值的边界框作为目标假设集。这个集合然后用于极大的降低特定类别的目标检测器要处理的窗口数量。我们会将我们的方法与之比较。

Another strategy is to use visual words of the Bag-of-Words model to predict the object location. Vedaldi et al. [34] use jumping windows [5], in which the relation between individual visual words and the object location is learned to predict the object location in new images. Maji and Malik [23] combine multiple of these relations to predict the object location using a Hough-transform, after which they randomly sample windows close to the Hough maximum. In contrast to learning, we use the image structure to sample a set of class-independent object hypotheses.

另一种策略是使用Bag-of-Words模型中的视觉词来预测目标位置。Vedaldi et al. [34]使用了跳窗[5]，其中学习了单个视觉词和目标位置的关系，来预测目标在新图像中的位置。Maji and Malik [23]将多个这种关系结合起来，用Hough变换来预测目标位置，然后他们随机取样接近Hough极大值的窗口。他们采用学习，而我们使用图像结构来采样与类别无关的目标集假设。

To summarize, our novelty is as follows. Instead of an exhaustive search [8, 12, 16, 36] we use segmentation as selective search yielding a small set of class independent object locations. In contrast to the segmentation of [4, 9], instead of focusing on the best segmentation algorithm [3], we use a variety of strategies to deal with as many image conditions as possible, thereby severely reducing computational costs while potentially capturing more objects accurately. Instead of learning an objectness measure on randomly sampled boxes [2], we use a bottom-up grouping procedure to generate good object locations.

总结一下，我们的创新点如下。我们没有使用穷举式搜索[8,12,16,36]，我们使用分割进行selective search，得到了与类别无关的较小的目标位置集。与[4,9]的分割相比，我们没有聚焦在最佳的分割算法[3]，而是使用多种策略来处理各种条件的图像，所以极大的降低了计算量，而且可能精确的捕捉到更多目标。我们没有在随机取样的框中学习objectness值[2]，而是使用了自下而上的分组过程来生成好的图像位置。

## 3 Selective Search

In this section we detail our selective search algorithm for object recognition and present a variety of diversification strategies to deal with as many image conditions as possible. A selective search algorithm is subject to the following design considerations:

本节中我们详述了我们面向目标识别的selective search算法，提出了几种多样化的策略来处理各种图像条件。Selective search算法主要依据下述设计考虑：

**Capture All Scales**. Objects can occur at any scale within the image. Furthermore, some objects have less clear boundaries then other objects. Therefore, in selective search all object scales have to be taken into account, as illustrated in Figure 2. This is most naturally achieved by using an hierarchical algorithm.

**捕获所有尺度**。目标可能出现在图像的任何尺度中。而且，一些目标的边界可能不太明确。所以，在selective search算法中，所有目标尺度都必须考虑到，如图2所示。这通过使用层次化算法那可以自然得到实现。

**Diversification**. There is no single optimal strategy to group regions together. As observed earlier in Figure 1, regions may form an object because of only colour, only texture, or because parts are enclosed. Furthermore, lighting conditions such as shading and the colour of the light may influence how regions form an object. Therefore instead of a single strategy which works well in most cases, we want to have a diverse set of strategies to deal with all cases.

**多样化**。没有单个的优化策略来分组形成区域。在图1中我们观察到，区域可能因为色彩、纹理或部件封闭性形成目标。而且，光照条件比如阴影和光的颜色可能影响区域形成目标。所以，我们没有采用在多数情况下工作良好的单个策略，而是采用了多样化的策略来处理所有情况。

**Fast to Compute**. The goal of selective search is to yield a set of possible object locations for use in a practical object recognition framework. The creation of this set should not become a computational bottleneck, hence our algorithm should be reasonably fast.

**快速计算**。Selective search的目标是生成可能的目标位置集，然后进行目标识别。集合的生成不能成为计算瓶颈，所以我们的算法应当没那么复杂。

Figure 2: Two examples of our selective search showing the necessity of different scales. On the left we find many objects at different scales. On the right we necessarily find the objects at different scales as the girl is contained by the tv.

### 3.1 Selective Search by Hierarchical Grouping

We take a hierarchical grouping algorithm to form the basis of our selective search. Bottom-up grouping is a popular approach to segmentation [6, 13], hence we adapt it for selective search. Because the process of grouping itself is hierarchical, we can naturally generate locations at all scales by continuing the grouping process until the whole image becomes a single region. This satisfies the condition of capturing all scales.

我们以一个层次化分组算法作为我们selective search算法的基础。自下而上分组是分割的流行方法[6,13]，所以我们在selective search中使用。由于分组这个过程本身就是层次性的，所以可以在分组的过程中可以自然地生成所有尺度上的位置，直到整个图像变成一整个区域。这就满足了捕获所有尺度的条件。

As regions can yield richer information than pixels, we want to use region-based features whenever possible. To get a set of small starting regions which ideally do not span multiple objects, we use the fast method of Felzenszwalb and Huttenlocher [13], which [3] found well-suited for such purpose.

由于区域可以比像素产生更丰富的信息，我们希望尽可能使用基于区域的特征。为得到较小的起始区域集，我们使用了Felzenszwalb and Huttenlocher [13]的快速方法，在[3]中发现这个方法可以满足这个目的。

Our grouping procedure now works as follows. We first use [13] to create initial regions. Then we use a greedy algorithm to iteratively group regions together: First the similarities between all neighbouring regions are calculated. The two most similar regions are grouped together, and new similarities are calculated between the resulting region and its neighbours. The process of grouping the most similar regions is repeated until the whole image becomes a single region. The general method is detailed in Algorithm 1.

我们的分组过程如下。我们首先使用[13]生成初始区域，然后使用贪婪算法迭代分组区域：首先计算相邻区域的相似性，最相似的两个区域首先合并分组，然后计算新的相邻区域的相似性。这个分组过程重复进行，直到整个区域成为一整个区域。一般方法的细节如算法1所示。

**Algorithm 1**: Hierarchical Grouping Algorithm

**Input**: (colour) image \
**Output**: Set of object location hypotheses *L* \
Obtain initial regions $R = {r_1,··· , r_n}$ using [13] \
Initialise similarity set $S = \varnothing$ \
**for each** Neighbouring region pair $(r_i, r_j)$ **do** \
Calculate similarity $s(r_i, r_j)$ \
$S = S∪s(r_i, r_j)$ \

**while** $S \neq \varnothing$ **do** \
Get highest similarity $s(r_i ,r_j) = max(S)$ \
Merge corresponding regions $r_t = r_i ∪r_j$ \
Remove similarities regarding $r_i : S = S - s(r_i ,r_∗ )$ \
Remove similarities regarding $r_j : S = S - s(r_∗ ,r_j )$ \
Calculate similarity set $S_t$ between $r_t$ and its neighbours \
$S = S∪S_t$ \
$R = R∪r_t$ \

Extract object location boxes *L* from all regions in *R*

For the similarity $s(r_i ,r_j)$ between region $r_i$ and $r_j$ we want a variety of complementary measures under the constraint that they are fast to compute. In effect, this means that the similarities should be based on features that can be propagated through the hierarchy, i.e. when merging region $r_i$ and $r_j$ into $r_t$, the features of region $r_t$ need to be calculated from the features of $r_i$ and $r_j$ without accessing the image pixels.

对于区域$r_i$和$r_j$间的相似度$s(r_i ,r_j)$，我们希望有多种互补的度量，并且能够快速计算。实际上，这意味着相似度应当基于特征，在层次之间可以传播，即，当合并区域$r_i$和$r_j$成为$r_t$时，区域$r_t$的特征可以从区域$r_i$和$r_j$的特征计算出来，而不用再从像素中计算得到。

### 3.2 Diversification Strategies

The second design criterion for selective search is to diversify the sampling and create a set of complementary strategies whose locations are combined afterwards. We diversify our selective search (1) by using a variety of colour spaces with different invariance properties, (2) by using different similarity measures $s_{ij}$, and (3) by varying our starting regions.

Selective search的第二个设计准则是取样多样化，生成互补策略集，之后再组合其位置。我们selective search多样化的途径如下：(1)使用带有不同不变性性质的多种色彩空间，(2)使用不同的相似性度量$s_{ij}$，(3)初始区域的变化。

**Complementary Colour Spaces**. We want to account for different scene and lighting conditions. Therefore we perform our hierarchical grouping algorithm in a variety of colour spaces with a range of invariance properties. Specifically, we use the following colour spaces with an increasing degree of invariance: (1) RGB, (2) the intensity (grey-scale image) I, (3) Lab, (4) the rg channels of normalized RGB plus intensity denoted as rgI, (5) HSV, (6) normalized RGB denoted as rgb, (7) C [14] which is an opponent colour space where intensity is divided out, and finally (8) the Hue channel H from HSV. The specific invariance properties are listed in Table 1.

**互补的色彩空间**。我们希望能处理不同的场景和光照条件。所以我们在带有几种不变性性质的多种颜色空间中执行我们的层次性分组算法。特别的，我们使用下述颜色空间，其不变性是逐渐增加的：(1)RGB, (2)灰度图像的灰度I, (3)Lab, (4)归一化RGB后的rg通道和强度，表示为rgI, (5)HSV, (6)归一化的RGB，表示为rgb, (7) [14]中的另一个色彩空间C，其中去除了强度分量，最后(8)HSV空间中的Hue通道H。上述空间的不变性见表1。

Table 1: The invariance properties of both the individual colour channels and the colour spaces used in this paper, sorted by degree of invariance. A “+/-” means partial invariance. A fraction 1/3 means that one of the three colour channels is invariant to said property.

colour channels | R | G | B | I | V | L | a | b | S | r | g | C | H
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Light Intensity | - | - | - | - | - | - | +/- | +/- | + | + | + | + | +
Shadows/shading | - | - | - | - | - | - | +/- | +/- | + | + | + | + | +
Highlights | - | - | - | - | - | - | - | - | - | - | - | +/- | +

colour spaces | RGB | I | Lab | rgI | HSV | rgb | C | H
--- | --- | --- | --- | --- | --- | --- | --- | ---
Light Intensity | - | - | +/- | 2/3 | 2/3 | + | + | +
Shadows/shading | - | - | +/- | 2/3 | 2/3 | + | + | +
Highlights | - | - | - | - | 1/3 | - | +/- | +

Of course, for images that are black and white a change of colour space has little impact on the final outcome of the algorithm. For these images we rely on the other diversification methods for ensuring good object locations.

当然，对于黑白图像，改变色彩空间对算法输出的影响很小。对于这些图像，我们需要其他的多样化方法来确保得到好的目标位置。

In this paper we always use a single colour space throughout the algorithm, meaning that both the initial grouping algorithm of [13] and our subsequent grouping algorithm are performed in this colour space.

本文中我们始终使用单个色彩空间，这意思是初始分组算法[13]和后续的分组算法都在这个色彩空间中进行。

**Complementary Similarity Measures**. We define four complementary, fast-to-compute similarity measures. These measures are all in range [0,1] which facilitates combinations of these measures.

**互补的相似性度量**。我们定义四种互补的，可以快速计算的相似性度量。这些度量值范围都是[0,1]，使用的都是这些度量的组合。

$s_{colour}(r_i, r_j)$ measures colour similarity. Specifically, for each region we obtain one-dimensional colour histograms for each colour channel using 25 bins, which we found to work well. This leads to a colour histogram $C_i = {c^1_i ,··· ,c^n_i}$ for each region $r_i$ with dimensionality n = 75 when three colour channels are used. The colour histograms are normalised using the $L_1$ norm. Similarity is measured using the histogram intersection:

$s_{colour}(r_i, r_j)$是色彩相似性的度量。特别的，对每个区域的每个色彩通道我们都计算25bin一维色彩直方图，试验中发现效果较好。这样对于每个区域$r_i$，在3个色彩通道的情况下，就可以得到色彩直方图$C_i = {c^1_i ,··· ,c^n_i}$，其中n=75,。色彩直方图用$L_1$范数归一化。相似性采用直方图交集来进行度量：

$$s_{colour}(r_i, r_j) = \sum^n_{k=1} min(c^k_i, c^k_j)$$(1)

The colour histograms can be efficiently propagated through the hierarchy by 色彩直方图可以在层次间通过如下方式传播

$$C_t = \frac {size(r_i)×C_i + size(r_j)×C_j} {size(r_i)+size(r_j)}$$(2)

The size of a resulting region is simply the sum of its constituents: $size(r_t) = size(r_i)+size(r_j)$.

得到的区域就是两个较小区域的和：$size(r_t) = size(r_i)+size(r_j)$。

$s_{texture}(r_i,r_j)$ measures texture similarity. We represent texture using fast SIFT-like measurements as SIFT itself works well for material recognition [20]. We take Gaussian derivatives in eight orientations using σ = 1 for each colour channel. For each orientation for each colour channel we extract a histogram using a bin size of 10. This leads to a texture histogram $T_i = {t^1_i, ···, t^n_i}$ for each region $r_i$ with dimensionality n = 240 when three colour channels are used. Texture histograms are normalised using the $L_1$ norm. Similarity is measured using histogram intersection:

$s_{texture}(r_i,r_j)$是纹理相似性度量。由于SIFT对于材质识别效果较好[20]，所以我们用快速类SIFT特征来表示纹理。对每个色彩通道，我们采用8个方向的Gaussian导数，σ = 1。对每个颜色通道的每个方向，我们提取10bin直方图。这样对于每个区域$r_i$，当使用3个颜色通道时，会得到纹理直方图$T_i = {t^1_i, ···, t^n_i}$，其中n=240。纹理直方图采用$L_1$范数归一化。相似性用直方图交集来进行度量：

$$s_{texture}(r_i, r_j) = \sum^n_{k=1} min(t^k_i, t^k_j)$$(3)

Texture histograms are efficiently propagated through the hierarchy in the same way as the colour histograms. 纹理直方图在层次间的传播方式和色彩直方图一样。

$s_{size}(r_i, r_j)$ encourages small regions to merge early. This forces regions in S, i.e. regions which have not yet been merged, to be of similar sizes throughout the algorithm. This is desirable because it ensures that object locations at all scales are created at all parts of the image. For example, it prevents a single region from gobbling up all other regions one by one, yielding all scales only at the location of this growing region and nowhere else. $s_{size}(r_i, r_j)$ is defined as the fraction of the image that $r_i$ and $r_j$ jointly occupy:

$s_{size}(r_i, r_j)$鼓励早期的小区域合并。这使S中的区域，也就是尚未合并的区域，在整个算法过程中类似大小的区域可以合并。这确保了所有尺度的目标位置在图像各个部分都可以生成。比如，这防止了单个区域逐渐一个一个的合并了其他区域，从而只在这个位置上得到各个尺度的合并区域，别的位置都没有。$s_{size}(r_i, r_j)$的定义为区域$r_i$和$r_j$共同占有的图像的比例

$$ s_{size}(r_i, r_j) = 1− \frac {size(r_i)+size(r_j)} {size(im)}$$(4)

where size(im) denotes the size of the image in pixels. 其中size(im)为图像包括的像素数目。

$s_{fill}(r_i, r_j)$ measures how well region $r_i$ and $r_j$ fit into each other. The idea is to fill gaps: if $r_i$ is contained in $r_j$ it is logical to merge these first in order to avoid any holes. On the other hand, if $r_i$ and $r_j$ are hardly touching each other they will likely form a strange region and should not be merged. To keep the measure fast, we use only the size of the regions and of the containing boxes. Specifically, we define $BB_{ij}$ to be the tight bounding box around $r_i$ and $r_j$. Now $s_{fill}(r_i, r_j)$ is the fraction of the image contained in $BB_{ij}$ which is not covered by the regions of $r_i$ and $r_j$:

$s_{fill}(r_i, r_j)$是区域$r_i$和$r_j$相互符合适应性的度量。其思想是补充间隙缺口：如果区域$r_i$包含在$r_j$中，那么应当将其合并，避免出现空洞。另一方面，如果区域$r_i$和$r_j$几乎没有接触，那么有可能形成一种奇怪的区域，不应当被合并。为使这些度量可以快速计算，我们只使用区域的大小和边界框的大小。特别的，我们定义$BB_{ij}$为$r_i$和$r_j$的外围边界框。现在$s_{fill}(r_i, r_j)$定义为$BB_{ij}$中去除$r_i$和$r_j$的剩余区域部分：

$$s_{fill}(r_i, r_j) = 1− \frac {size(BB ij )−size(r i )−size(r i )} {size(im)}$$(5)

We divide by size(im) for consistency with Equation 4. Note that this measure can be efficiently calculated by keeping track of the bounding boxes around each region, as the bounding box around two regions can be easily derived from these.

我们像式(4)一样除以size(im)。注意这个度量可以通过追踪每个区域的边界框很容易的计算出来，因为两个区域的边界框可以很容易推断出来。

In this paper, our final similarity measure is a combination of the above four: 本文中，我们最终的相似性度量为上述四个度量的组合：

$$s(r_i, r_j) = a_1 s_{colour}(r_i, r_j) + a_2 s_{texture}(r_i, r_j) + a_3 s_{size}(r_i, r_j) + a_4 s_{fill}(r_i, r_j)$$(6)

where $a_i ∈ {0,1}$ denotes if the similarity measure is used or not. As we aim to diversify our strategies, we do not consider any weighted similarities.

其中$a_i ∈ {0,1}$为是否使用这几种相似性度量。由于我们需要多样化的策略，我们不考虑带权重的相似性度量。

**Complementary Starting Regions**. A third diversification strategy is varying the complementary starting regions. To the best of our knowledge, the method of [13] is the fastest, publicly available algorithm that yields high quality starting locations. We could not find any other algorithm with similar computational efficiency so we use only this oversegmentation in this paper. But note that different starting regions are (already) obtained by varying the colour spaces, each which has different invariance properties. Additionally, we vary the threshold parameter k in [13].

**互补的起始区域**。另一个多样化的策略是使互补的起始区域有变化。据我们所知，[13]的方法是运算速度最快的，可以得到高质量的起始区域。我们没有找到更好的算法，所以我们在本文中就使用这种算法。但注意在使用不同的色彩空间时就已经得到了起始区域。另外，我们采用了与[13]中不同的阈值。

### 3.3 Combining Locations 位置组合

In this paper, we combine the object hypotheses of several variations of our hierarchical grouping algorithm. Ideally, we want to order the object hypotheses in such a way that the locations which are most likely to be an object come first. This enables one to find a good trade-off between the quality and quantity of the resulting object hypothesis set, depending on the computational efficiency of the subsequent feature extraction and classification method.

本文中，我们将层次式分组算法的几个变体得到的假设目标进行组合。理想情况下，我们希望假设目标排序后，最可能是目标的位置排在前面。这需要使目标假设集的数量和质量有很好的折中，也要看后续的特征提取和分类算法的计算需求。

We choose to order the combined object hypotheses set based on the order in which the hypotheses were generated in each individual grouping strategy. However, as we combine results from up to 80 different strategies, such order would too heavily emphasize large regions. To prevent this, we include some randomness as follows. Given a grouping strategy j, let $r^j_i$ be the region which is created at position i in the hierarchy, where i = 1 represents the top of the hierarchy (whose corresponding region covers the complete image). We now calculate the position value $v^j_i$ as RND×i, where RND is a random number in range [0,1]. The final ranking is obtained by ordering the regions using $v^j_i$.

我们这样对组合的目标假设集进行排序：单个的分组策略产生的假设排列在一起。但是，由于我们要组合的结果来源于多达80个不同的策略，这样排序会过于强调较大的区域。为防止这个问题，我们选用了下面的一些随机性。给定分组策略j，令$r^j_i$为在层次i产生的区域，i=1表示最高层（对应整个图像）。我们现在计算$v^j_i$为RND×i，其中RND为随机数，范围[0,1]。最终排序通过$v^j_i$得到。

When we use locations in terms of bounding boxes, we first rank all the locations as detailed above. Only afterwards we filter out lower ranked duplicates. This ensures that duplicate boxes have a better chance of obtaining a high rank. This is desirable because if multiple grouping strategies suggest the same box location, it is likely to come from a visually coherent part of the image.

当我们用边界框表示位置，我们首先如上述对所有位置排序。然后，我们滤除掉排名很低的重复位置。这确保了重复的边界框更可能得到高的排名。这是值得的，因为如果多个分组策略得到了同样的边界框位置，那么很可能是图像的一致部分。