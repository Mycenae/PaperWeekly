# Contour Detection and Hierarchical Image Segmentation

## 0. Abstract

This paper investigates two fundamental problems in computer vision: contour detection and image segmentation. We present state-of-the-art algorithms for both of these tasks. Our contour detector combines multiple local cues into a globalization framework based on spectral clustering. Our segmentation algorithm consists of generic machinery for transforming the output of any contour detector into a hierarchical region tree. In this manner, we reduce the problem of image segmentation to that of contour detection. Extensive experimental evaluation demonstrates that both our contour detection and segmentation methods significantly outperform competing algorithms. The automatically generated hierarchical segmentations can be interactively refined by user- specified annotations. Computation at multiple image resolutions provides a means of coupling our system to recognition applications.

本文研究了计算机视觉中的两个基本问题：边缘检测和图像分割。我们提出了这两个任务目前最好的算法。我们的边缘检测器基于谱聚类，将多个局部线索综合到一个全局框架中。我们的分割算法是通用机制，将任何边缘检测器的输出，转换成一个层次化的区域树。以这种方式，我们将图像分割问题，变成了一个边缘检测的问题。多个试验评估表明，我们的边缘检测和分割方法，显著超过了其他类似算法。自动生成的层次化的分割，可以由用户指定的标注进行互动优化。在多个图像分辨率上的计算，提供了一种将我们的系统与识别应用结合的方法。

## 1. Introduction

This paper presents a unified approach to contour detection and image segmentation. Contributions include: 本文提出了边缘检测和图像分割的统一方法。贡献包括：

- A high performance contour detector, combining local and global image information. 高性能边缘检测器，将局部和全局图像信息结合到了一起。

- A method to transform any contour signal into a hierarchy of regions while preserving contour quality. 将任何边缘信号转换成层次区域的方法，同时保护了边缘质量。

- Extensive quantitative evaluation and the release of a new annotated dataset. 广泛的定量评估，并放出了一个新的标注的数据集。

Figures 1 and 2 summarize our main results. The two Figures represent the evaluation of multiple contour detection (Figure 1) and image segmentation (Figure 2) algorithms on the Berkeley Segmentation Dataset (BSDS300) [1], using the precision-recall framework introduced in [2]. This benchmark operates by comparing machine generated contours to human ground-truth data (Figure 3) and allows evaluation of segmentations in the same framework by regarding region boundaries as contours.

图1和图2总结了我们的主要结果。这两幅图是多个边缘检测和图像分割算法在BSDS300数据集上的评估，使用的是[2]中提出的精度-召回框架。这个基准测试将算法产生的边缘与人类标注的真值数据（图3）进行比较，并在相同的框架中评估分割，将区域的轮廓视为边缘。

Especially noteworthy in Figure 1 is the contour detector gPb, which compares favorably with other leading techniques, providing equal or better precision for most choices of recall. In Figure 2, gPb-owt-ucm provides universally better performance than alternative segmentation algorithms. We introduced the gPb and gPb-owt-ucm algorithms in [3] and [4], respectively. This paper offers comprehensive versions of these algorithms, motivation behind their design, and additional experiments which support our basic claims.

在图1中尤其值得注意的是，边缘检测器gPb，与其他领先的技术相比，效果要更好，对各种召回值都得到了类似的或更好的精度值。在图2中，gPb-owt-ucm比其他分割算法，提供了一致更好的性能。我们分别在[3,4]中提出了gPB和gPb-owt-ucm算法。本文给出了这些算法的更多版本，这是受其设计所推动的，并提供了更多的试验，支持了我们的基本观点。

We begin with a review of the extensive literature on contour detection and image segmentation in Section 2. 我们在第2部分，回顾了边缘检测和图像分割的很多文献。

Section 3 covers the development of the gPb contour detector. We couple multiscale local brightness, color, and texture cues to a powerful globalization framework using spectral clustering. The local cues, computed by applying oriented gradient operators at every location in the image, define an affinity matrix representing the similarity between pixels. From this matrix, we derive a generalized eigenproblem and solve for a fixed number of eigenvectors which encode contour information. Using a classifier to recombine this signal with the local cues, we obtain a large improvement over alternative globalization schemes built on top of similar cues.

第3部分是gPb边缘检测器的开发。我们将多尺度的局部亮度，色彩和纹理线索结合到一起，形成了一个强力的全局化框架，使用的是谱聚类。局部线索，是在图像的每个位置计算有方向的梯度算子得到的，定义了一个临近矩阵，表示像素之间的相似度。从这个矩阵开始，我们推导出一个一般性的特征值问题，求解特定数量的特征向量，包含了轮廓信息。使用一个分类器，将这个信号与局部线索重新结合起来，与其他在类似的线索上得到的全局方案相比，我们得到了很大的改进。

To produce high-quality image segmentations, we link this contour detector with a generic grouping algorithm described in Section 4 and consisting of two steps. First, we introduce a new image transformation called the Oriented Watershed Transform for constructing a set of initial regions from an oriented contour signal. Second, using an agglomerative clustering procedure, we form these regions into a hierarchy which can be represented by an Ultrametric Contour Map, the real-valued image obtained by weighting each boundary by its scale of disappearance. We provide experiments on the BSDS300 as well as the BSDS500, a superset newly released here.

为得到高质量的图像分割，我们将边缘检测器与第4部分的一个通用分组算法结合起来，分组算法包括2个步骤。第一，我们提出了一种新的图像变换，称为有向分水岭变换，从一个有向的轮廓信号中，构建出一个初始区域集合。第二，使用一种聚积的聚类过程，我们将这些区域形成一个层次结构，可以用一种超度量轮廓图来表示，这是一幅实值图像，是通过对每个边缘通过其消失的尺度来进行加权得到的。我们在BSDS300和BSDS500上给出了试验。

Although the precision-recall framework [2] has found widespread use for evaluating contour detectors, considerable effort has also gone into developing metrics to directly measure the quality of regions produced by segmentation algorithms. Noteworthy examples include the Probabilistic Rand Index, introduced in this context by [5], the Variation of Information [6], [7], and the Segmentation Covering criteria used in the PASCAL challenge [8]. We consider all of these metrics and demonstrate that gPb-owt-ucm delivers an across-the-board improvement over existing algorithms.

虽然精度-召回框架在评估轮廓检测器中得到了广泛的应用，在开发直接度量分割算法得到的区域的质量方面，也有很多努力。值得注意的例子包括，Probabilistic Rand Index [5], Variation of Information [6,7], 和在PASCAL挑战赛上使用的Segmentation Covering[8]。我们考虑了所有这些度量标准，证明了gPb-owt-ucm对已有的算法得到了全面的改进。

Sections 5 and 6 explore ways of connecting our purely bottom-up contour and segmentation machinery to sources of top-down knowledge. In Section 5, this knowledge source is a human. Our hierarchical region trees serve as a natural starting point for interactive segmentation. With minimal annotation, a user can correct errors in the automatic segmentation and pull out objects of interest from the image. In Section 6, we target top-down object detection algorithms and show how to create multiscale contour and region output tailored to match the scales of interest to the object detector.

第5和第6部分探索了一些将我们的纯自下而上的轮廓和分割算法与自上而下的知识进行结合的方法。在第5部分中，这种知识源是人类。我们的层次结构区域树作为互动分割的很自然的初始点。在最小标注下，用户可以在自动分割中修改错误，从图像中得出感兴趣的目标。在第6部分中，我们的目标是自上而下的目标检测算法，展示了怎样创建定制的多尺度轮廓和区域输出，与目标检测器中的感兴趣尺度匹配起来。

Though much remains to be done to take full advantage of segmentation as an intermediate processing layer, recent work has produced payoffs from this endeavor [9], [10], [11], [12], [13]. In particular, our gPb-owt-ucm segmentation algorithm has found use in optical flow [14] and object recognition [15], [16] applications.

要利用分割作为中间处理层，很多工作还需要去做，尽管这样，最新的工作从这些努力中得到了回报。特别是，我们的gPb-owt-ucm分割算法在光流和目标识别应用中找到了应用。

## 2. Previous Work

The problems of contour detection and segmentation are related, but not identical. In general, contour detectors offer no guarantee that they will produce closed contours and hence do not necessarily provide a partition of the image into regions. But, one can always recover closed contours from regions in the form of their boundaries. As an accomplishment here, Section 4 shows how to do the reverse and recover regions from a contour detector. Historically, however, there have been different lines of approach to these two problems, which we now review.

轮廓检测和分割的问题是相互关联的，但并不一致。一般来说，轮廓检测器并不保证会得到闭合的轮廓，因此不一定会将图像分割成区域。但是，利用区域的边缘，一直可以恢复出闭合的轮廓。第4部分的一个成就是，展示了怎样从一个轮廓检测器中恢复出区域。历史上对这个两个问题有不同的研究线路，现在我们回顾一下。

### 2.1 Contours

Early approaches to contour detection aim at quantifying the presence of a boundary at a given image location through local measurements. The Roberts [17], Sobel [18], and Prewitt [19] operators detect edges by convolving a grayscale image with local derivative filters. Marr and Hildreth [20] use zero crossings of the Laplacian of Gaussian operator. The Canny detector [22] also models edges as sharp discontinuities in the brightness channel, adding non-maximum suppression and hysteresis thresholding steps. A richer description can be obtained by considering the response of the image to a family of filters of different scales and orientations. An example is the Oriented Energy approach [21], [36], [37], which uses quadrature pairs of even and odd symmetric filters. Lindeberg [38] proposes a filter-based method with an automatic scale selection mechanism.

早期的轮廓检测方法，其目标是在给定的图像位置，通过局部度量，对边缘是否存在进行量化。Roberts, Sobel和Prewitt算子，通过将灰度图像与局部导数滤波器进行卷积，进行边缘检测。Marr和Hildreth使用的是高斯算子的Laplacian的过零点。Canny检测器也将亮度通道的尖锐不连续点作为边缘，并加入了非最大抑制和滞后阈值的步骤。使用一组不同尺度和方向的滤波器，对图像进行滤波，计算其响应，可以得到更丰富的响应。一个例子是Oriented Energy方法，使用的是奇对称和偶对称的滤波器的四元组。Lindeberg提出了一种基于滤波器的方法，可以自动选择滤波器的尺度。

More recent local approaches take into account color and texture information and make use of learning techniques for cue combination [2], [26], [27]. Martin et al. [2] define gradient operators for brightness, color, and texture channels, and use them as input to a logistic regression classifier for predicting edge strength. Rather than rely on such hand-crafted features, Dollar et al. [27] propose a Boosted Edge Learning (BEL) algorithm which attempts to learn an edge classifier in the form of a probabilistic boosting tree [39] from thousands of simple features computed on image patches. An advantage of this approach is that it may be possible to handle cues such as parallelism and completion in the initial classification stage. Mairal et al. [26] create both generic and class-specific edge detectors by learning discriminative sparse representations of local image patches. For each class, they learn a discriminative dictionary and use the reconstruction error obtained with each dictionary as feature input to a final classifier.

最近的局部方法，将颜色和纹理信息纳入了考虑，使用学习的方法进行线索综合。Martin等[2]定义了灰度，色彩和纹理通道的梯度算子，将其用作逻辑回归分类器的输入，以预测边缘的强度。Dollar等[27]没有依赖于这样的手工特征，提出一种Boosted Edge Learning (BEL)算法，以概率boosting树的形式，从在图像块中计算得到的数千个简单特征中，学习一个边缘分类器。这个方法的一个优势是，在最初的分类阶段，可能处理像平行和填充这样的线索。Mairal等[26]通过从局部图像块中学习有区分性的稀疏表示，创建了通用的和特定类别的边缘检测器。对于每个类别，他们学习一个有区分性的字典，使用得到的重建误差和每个字典一起，作为最终的分类器的特征输入。

The large range of scales at which objects may appear in the image remains a concern for these modern local approaches. Ren [28] finds benefit in combining information from multiple scales of the local operators developed by [2]. Additional localization and relative contrast cues, defined in terms of the multiscale detector output, are fed to the boundary classifier. For each scale, the localization cue captures the distance from a pixel to the nearest peak response. The relative contrast cue normalizes each pixel in terms of the local neighborhood.

目标在图像中出现的尺度很大，这是现代局部方法的一个问题。Ren[28]将多个尺度的局部算子[2]的信息结合起来，发现有所帮助。另外的定位和相对对比度线索，以多尺度检测器输出的形式定义，送入边缘分类器。对于每个尺度，定位线索捕获了一个像素到其最近的峰值响应的距离。相对对比度线索对每个像素以局部邻域的形式进行归一化。

An orthogonal line of work in contour detection focuses primarily on another level of processing, globalization, that utilizes local detector output. The simplest such algorithms link together high-gradient edge fragments in order to identify extended, smooth contours [40], [41], [42]. More advanced globalization stages are the distinguishing characteristics of several of the recent high-performance methods benchmarked in Figure 1, including our own, which share as a common feature their use of the local edge detection operators of [2].

轮廓检测的另一条线主要关注的是另一个处理的层次，即利用局部检测器输出的全局化。最简单的这样的算法，将高梯度的边缘碎片连接起来，以识别出拓展的平滑的轮廓。更高级的全局化步骤，是图1中测试的最近的几种高性能方法，包括我们自己的算法，与其他算法相同的是，也使用了[2]中的局部边缘检测算子。

Ren et al. [23] use the Conditional Random Fields (CRF) framework to enforce curvilinear continuity of contours. They compute a constrained Delaunay triangulation (CDT) on top of locally detected contours, yielding a graph consisting of the detected contours along with the new “completion” edges introduced by the triangulation. The CDT is scale-invariant and tends to fill short gaps in the detected contours. By associating a random variable with each contour and each completion edge, they define a CRF with edge potentials in terms of detector response and vertex potentials in terms of junction type and continuation smoothness. They use loopy belief propagation [43] to compute expectations.

Ren等[23]使用条件随机场(CRF)框架，来保证轮廓的曲线平滑性。他们在局部检测到的轮廓上，计算了一个约束的Delaunay三角剖分(CDT)，生成了一个图，由检测到的轮廓，与由三角剖分带来的新的补全的边缘一起组成。CDT是尺度不变的，可以对检测到的轮廓填充其小的空缺。将每个轮廓和每个补全的边缘与一个随机变量相关联，可以定义一个CRF与可能的边缘，以检测器的响应和顶点可能的形式，以结合点的类型和连续的平滑度的形式。他们使用loopy belief传播来计算期望。

Felzenszwalb and McAllester [25] use a different strategy for extracting salient smooth curves from the output of a local contour detector. They consider the set of short oriented line segments that connect pixels in the image to their neighboring pixels. Each such segment is either part of a curve or is a background segment. They assume curves are drawn from a Markov process, the prior distribution on curves favors few per scene, and detector responses are conditionally independent given the labeling of line segments. Finding the optimal line segment labeling then translates into a general weighted min-cover problem in which the elements being covered are the line segments themselves and the objects covering them are drawn from the set of all possible curves and all possible background line segments. Since this problem is NP-hard, an approximate solution is found using a greedy “cost per pixel” heuristic.

Felzenszwalb等[25]使用一种不同的策略，从一个局部轮廓检测器中，提取出显著的平滑曲线。他们考虑短的有向的线段的集合，将图像中的像素与其邻域的像素连接起来。每个这样的片段，要么是一条曲线的一部分，要么是一个背景片段。他们假设，曲线是从一个Markov过程抽取出来的，曲线的先验分布，倾向于在每个场景中的数量尽力少，给定线段的标注，检测器的响应是条件独立的。找到最优的线段标注，然后转换成了一个通用的加权最小覆盖问题，其中覆盖的元素是线段本身，覆盖了他们的目标，是从所有的可能的曲线的集合和所有可能的背景线段中提取出来的。由于这是一个NP难题，使用一种贪婪的“每个像素的代价”的启发式算法可以得到一个近似的解。

Zhu et al. [24] also start with the output of [2] and create a weighted edgel graph, where the weights measure directed collinearity between neighboring edgels. They propose detecting closed topological cycles in this graph by considering the complex eigenvectors of the normalized random walk matrix. This procedure extracts both closed contours and smooth curves, as edgel chains are allowed to loop back at their termination points.

Zhu等[24]也是从[2]的输出开始，创建了一个加权的边缘图，其中这些权重度量的是相邻的边缘的有向共线性。他们提出在这个图中，通过考虑归一化的随机行走矩阵的复杂的特征向量，检测闭合的拓扑环。这个过程提取了闭合的轮廓和平滑的曲线，因为边缘链可以在其终止点上循环回来。

### 2.2 Regions

A broad family of approaches to segmentation involve integrating features such as brightness, color, or texture over local image patches and then clustering those features based on, e.g., fitting mixture models [7], [44], mode-finding [34], or graph partitioning [32], [45], [46], [47]. Three algorithms in this category appear to be the most widely used as sources of image segments in recent applications, due to a combination of reasonable performance and publicly available implementations.

分割方法的家族很大，是将局部图像块中的灰度、色彩或纹理的特征整合到一起，然后对这些特征进行聚类，比如根据fitting mixture models [7], [44], mode-finding [34], or graph partitioning [32], [45], [46], [47]。这个类别中的三个算法似乎使用最为广泛，在最近的应用中用作图像分割，其性能不错，有公开的实现。

The graph based region merging algorithm advocated by Felzenszwalb and Huttenlocher (Felz-Hutt) [32] attempts to partition image pixels into components such that the resulting segmentation is neither too coarse nor too fine. Given a graph in which pixels are nodes and edge weights measure the dissimilarity between nodes (e.g. color differences), each node is initially placed in its own component. Define the internal difference of a component Int(R) as the largest weight in the minimum spanning tree of R. Considering edges in non-decreasing order by weight, each step of the algorithm merges components R1 and R2 connected by the current edge if the edge weight is less than:

Felzenszwalb and Huttenlocher (Felz-Hutt)[32]支持基于图的区域合并算法，试图将图像像素分割成很多部分，这样得到的分割既不太粗糙，也不太精细。给定一个图，其中像素是节点，边的权重度量的是节点间的不相似性（如，色彩差异），每个节点都放置为其自己的部件。定义一个部件的内部差异Int(R)为，R的最小张树的最大权重。考虑将边以权重非减的方式进行排序，算法的每一步，如果边的权重小于下面的式子，就将组件R1和R2合并：

$$min(Int(R_1)+τ(R_1),Int(R_2)+τ(R_2))$$(1)

where τ(R) = k/|R|. k is a scale parameter that can be used to set a preference for component size. 其中τ(R) = k/|R，k是一个尺度参数，可以设置用于部件的大小。

The Mean Shift algorithm [34] offers an alternative clustering framework. Here, pixels are represented in the joint spatial-range domain by concatenating their spatial coordinates and color values into a single vector. Applying mean shift filtering in this domain yields a convergence point for each pixel. Regions are formed by grouping together all pixels whose convergence points are closer than h_s in the spatial domain and h_r in the range domain, where h_s and h_r are respective bandwidth parameters. Additional merging can also be performed to enforce a constraint on minimum region area.

Mean Shift算法给出了另一个聚类框架。这里，像素表示为空间-范围的联合域中给出，将其空间坐标和色彩值拼接成一个向量。在这个域中进行mean shift滤波，对每个像素得到一个收敛点。对于像素的收敛点，如果其在空间域中距离小于h_s，在范围域中小于h_r，那么就将将这些像素分组到一起，其中h_s和h_r是各自的带宽参数。也可以进行另外的合并，来施加最小区域的限制。

Spectral graph theory [48], and in particular the Normalized Cuts criterion [45], [46], provides a way of integrating global image information into the grouping process. In this framework, given an affinity matrix W whose entries encode the similarity between pixels, one defines diagonal matrix $D_{ii} = \sum_j W_{ij}$ and solves for the generalized eigenvectors of the linear system:

谱图理论[48]，特别是归一化割准则[45,46]，提供了一个整合全局图像信息到分组过程中的方法。在这个框架中，给定一个亲和矩阵W，其元素包含了像素之间的相似度的信息，定义$D_{ii} = \sum_j W_{ij}$，求解线性系统的一般性特征向量：

$$(D − W)v = λDv$$(2)

Traditionally, after this step, K-means clustering is applied to obtain a segmentation into regions. This approach often breaks uniform regions where the eigenvectors have smooth gradients. One solution is to reweight the affinity matrix [47]; others have proposed alternative graph partitioning formulations [49], [50], [51].

传统上来说，在这个步骤后，就应用K均值聚类，以得到一个分割，将图像分割成区域。这个方法通常会将一致性的区域打破，这些区域中会有很平滑的梯度。一个解决方案是，重新对亲和矩阵进行加权[47]；其他人已经提出了别的图割方法[49,50,51]。

A recent variant of Normalized Cuts for image segmentation is the Multiscale Normalized Cuts (NCuts) approach of Cour et al. [33]. The fact that W must be sparse, in order to avoid a prohibitively expensive computation, limits the naive implementation to using only local pixel affinities. Cour et al. solve this limitation by computing sparse affinity matrices at multiple scales, setting up cross-scale constraints, and deriving a new eigenproblem for this constrained multiscale cut.

图像分割的图割方法的一个变体是，Cour等[33]的多尺度归一化割(NCuts)方法。W必须是稀疏的，以避免非常昂贵的计算代价，这个事实限制了只使用局部像素亲和的实现。Cour等通过在多个尺度上计算稀疏亲和矩阵，以求解这个局限，设置了跨尺度约束，对这个约束多尺度割推导出了一个新的特征问题。

Sharon et al. [31] propose an alternative to improve the computational efficiency of Normalized Cuts. This approach, inspired by algebraic multigrid, iteratively coarsens the original graph by selecting a subset of nodes such that each variable on the fine level is strongly coupled to one on the coarse level. The same merging strategy is adopted in [52], where the strong coupling of a subset S of the graph nodes V is formalized as:

Sharon等[31]提出了改进归一化图割计算效率的另一种方法。这种方法，是受到代数多网格启发的，通过选择一个节点子集，这样在精细层次上的每个变量都与在粗糙层次上的紧密结合，迭代的使原始图变得更加粗糙。[52]中采用了同样的合并策略，其中图节点V的一个子集S的强耦合，可以正式写为：

$$\frac {\sum_{j∈S} p_{ij}} {\sum_{j∈V} p_{ij}} > ψ, ∀ i∈V−S$$(3)

where ψ is a constant and p_ij the probability of merging i and j, estimated from brightness and texture similarity. 其中ψ是一个常数，p_ij是合并i和j的概率，从亮度和纹理相似性估计得到。

Many approaches to image segmentation fall into a different category than those covered so far, relying on the formulation of the problem in a variational framework. An example is the model proposed by Mumford and Shah [53], where the segmentation of an observed image u0 is given by the minimization of the functional:

很多图像分割方法是不同类别的，将问题表述为一个变分框架。一个例子是Mumford和Shah提出的模型[53]，其中一幅图像u0的分割，是由下列泛函的最小化得到的：

$$F(u,C) = \int_Ω (u-u_0)^2 dx + μ\int_{Ω\\C} |∇(u)| dx + ν|C|$$

where u is piecewise smooth in Ω\C and μ, ν are weighting parameters. Theoretical properties of this model can be found in, e.g. [53], [54]. Several algorithms have been developed to minimize the energy (4) or its simplified version, where u is piecewise constant in Ω\C . Koepfler et al. [55] proposed a region merging method for this purpose. Chan and Vese [56], [57] follow a different approach, expressing (4) in the level set formalism of Osher and Sethian [58], [59]. Bertelli et al. [30] extend this approach to more general cost functions based on pairwise pixel similarities. Recently, Pock et al. [60] proposed to solve a convex relaxation of (4), thus obtaining robustness to initialization. Donoser et al. [29] subdivide the problem into several figure/ground segmentations, each initialized using low-level saliency and solved by minimizing an energy based on Total Variation.

其中u在Ω\C中是分段平滑的，μ, ν是加权参数。这个模型的理论性质可以在[53,54]中找到。提出了几种算法来最小化(4)的能量，或其简化版，其中u在Ω\C中是分段常数。Koepfler等[55]提出了一个达到这个目的的区域合并方法。Chan和Vese[56,57]提出了一种不同的方法，将(4)表述成一个水平集形式的问题。Bertelli等[30]将此方法拓展到一个更一般性的损失函数中，基于成对像素相似性。最近，Pock等[60]来求解(4)的凸松弛问题，因此对初始化是稳健的。Donoser等[29]将问题细分成几个图/背景分割问题，每个都用底层的显著性初始化，通过求解一个基于全变分的能量最小化问题进行初始化。

### 2.3 Benchmarks

Though much of the extensive literature on contour detection predates its development, the BSDS [2] has since found wide acceptance as a benchmark for this task [23], [24], [25], [26], [27], [28], [35], [61]. The standard for evaluating segmentations algorithms is less clear.

虽然很多轮廓检测的文献早于BSDS[2]的提出，但这个数据集已经成为了这个任务的广为接受的基准测试。而评估分割算法的标准则没有那么明确。

One option is to regard the segment boundaries as contours and evaluate them as such. However, a methodology that directly measures the quality of the segments is also desirable. Some types of errors, e.g. a missing pixel in the boundary between two regions, may not be reflected in the boundary benchmark, but can have substantial consequences for segmentation quality, e.g. incorrectly merging large regions. One might argue that the boundary benchmark favors contour detectors over segmentation methods, since the former are not burdened with the constraint of producing closed curves. We therefore also consider various region-based metrics.

一个选项是，将分割的边缘作为轮廓，以此进行评估。但是，一种能够直接度量分割质量的方法是非常好的。一些类型的错误，如，两个区域之间的边缘缺失了一个像素，在边缘的基准测试中可能不会反应出来，但在分割质量中可能会有很严重的后果，如，错误的将两个区域合并到了一起。我们可以认为，边缘的基准测试会倾向于轮廓检测器，而不倾向于分割方法，因为前者没有受到产生闭合曲线的约束的限制。因此我们也考虑各种基于区域的度量。

#### 2.3.1 Variation of Information

The Variation of Information metric was introduced for the purpose of clustering comparison [6]. It measures the distance between two segmentations in terms of their average conditional entropy given by:

信息变化的度量是为了聚类比较的目的。其度量的是两个分割之间的距离，具体是其平均条件熵，由下式给出：

$$VI(S, S') = H(S) + H(S') − 2I(S, S')$$(5)

where H and I represent respectively the entropies and mutual information between two clusterings of data S and S'. In our case, these clusterings are test and groundtruth segmentations. Although VI possesses some interesting theoretical properties [6], its perceptual meaning and applicability in the presence of several ground-truth segmentations remains unclear.

其中H和I分别表示两个聚类S和S'的熵和互信息。在我们的情况中，这些聚类是真值分割和测试的分割。虽然VI有一些有意思的理论性质，在存在几个真值分割的情况下，其感知意义和可应用性还不是太清楚。

#### 2.3.2 Rand Index

Originally, the Rand Index [62] was introduced for general clustering evaluation. It operates by comparing the compatibility of assignments between pairs of elements in the clusters. The Rand Index between test and groundtruth segmentations S and G is given by the sum of the number of pairs of pixels that have the same label in S and G and those that have different labels in both segmentations, divided by the total number of pairs of pixels. Variants of the Rand Index have been proposed [5], [7] for dealing with the case of multiple ground-truth segmentations. Given a set of ground-truth segmentations {$G_k$}, the Probabilistic Rand Index is defined as:

开始的时候，Rand Index的提出是用于通用聚类评估的。其计算是，在成对的聚类的元素中，比较其指定的可兼容性。测试和真值分割S和G之间的Rand Index，是由S和G中，有相同标签的像素对的数量，和在两个分割中有不同标签的像素对的数量之和，除以像素对的总数量。Rand Index的变体在[5,7]中提出，处理多真值分割的情况。给定一个真值分割的集合{$G_k$}，概率Rand Index定义为：

$$PRI(S,\{G_k\} = \frac{1}{T} \sum_{i<j} [c_{ij} p_{ij} + (1-c_{ij})(1-p_{ij})]$$(6)

where $c_{ij}$ is the event that pixels i and j have the same label and $p_{ij}$ its probability. T is the total number of pixel pairs. Using the sample mean to estimate $p_{ij}$, (6) amounts to averaging the Rand Index among different ground-truth segmentations. The PRI has been reported to suffer from a small dynamic range [5], [7], and its values across images and algorithms are often similar. In [5], this drawback is addressed by normalization with an empirical estimation of its expected value.

其中$c_{ij}$是i和j有相同标签的情况，$p_{ij}$是其概率。T是像素对的总数量。使用样本平均来估计$p_{ij}$，(6)是在多个不同的真值分割中对Rand Index进行平均。PRI据说有动态范围过小的问题，不同图像和算法的值通常很类似。在[5]中，这个缺点通过使用其期望值的经验估计的归一化，进行来处理。

#### 2.3.3 Segmentation Covering

The overlap between two regions R and R', defined as: 两个区域R和R'的重叠，定义为：

$$O(R,R') = \frac {|R∩R'|}{|R∪R'|}$$(7)

has been used for the evaluation of the pixel-wise classification task in recognition [8], [11]. We define the covering of a segmentation S by a segmentation S' as: 广泛用于识别中的逐像素的分类任务的评估。我们定义一个分割S由一个分割S'的覆盖为：

$$C(S'->S) = \frac {1}{N} \sum_{R∈S} |R|·max_{R'∈S'} O(R,R')$$(8)

where N denotes the total number of pixels in the image. 其中N表示图像中的像素总数量。

Similarly, the covering of a machine segmentation S by a family of ground-truth segmentations {Gi} is defined by first covering S separately with each human segmentation Gi, and then averaging over the different humans. To achieve perfect covering the machine segmentation must explain all of the human data. We can then define two quality descriptors for regions: the covering of S by {Gi} and the covering of {Gi} by S.

类似的，一个机器分割S被一族真值分割{Gi}的覆盖，定义为，首先计算S由每个人类分割Gi的覆盖，然后对各个值进行平均。为获得完美的覆盖，机器分割必须解释所有的人类数据。我们可以然后定义区域的两个质量描述子：{Gi}对S的覆盖，和S对{Gi}的覆盖。

## 3. Contour Detection

As a starting point for contour detection, we consider the work of Martin et al. [2], who define a function Pb(x, y, θ) that predicts the posterior probability of a boundary with orientation θ at each image pixel (x,y) by measuring the difference in local image brightness, color, and texture channels. In this section, we review these cues, introduce our own multiscale version of the Pb detector, and describe the new globalization method we run on top of this multiscale local detector.

作为轮廓检测的开始，我们考虑Martin等[2]的工作，他们定义了一个函数Pb(x, y, θ)，通过度量在图像局部亮度、色彩和纹理通道的差异，预测在像素点(x,y)上一个方向θ的边缘的后验概率。在本节中，我们回顾了这些线索，提出了我们自己的多尺度Pb检测器版本，在这个多尺度局部探测器之上，描述了新的全局化的方法。

### 3.1 Brightness, Color, Texture Gradients

The basic building block of the Pb contour detector is the computation of an oriented gradient signal G(x, y, θ) from an intensity image I. This computation proceeds by placing a circular disc at location (x, y) split into two half-discs by a diameter at angle θ. For each half-disc, we histogram the intensity values of the pixels of I covered by it. The gradient magnitude G at location (x,y) is defined by the χ^2 distance between the two half-disc histograms g and h:

Pb轮廓检测器的基本组成模块是，从一幅灰度图像I中，计算有向梯度信号G(x, y, θ)。这个计算是将一个圆形的盘子放在(x,y)位置上，由一条角度为θ的直径分成两个半盘，我们对这个图形覆盖的图像的像素求得其直方图。在(x,y)处的梯度幅度G定义为两个半圆盘的直方图g和h的χ^2距离：

$$χ^2(g,h) = \frac {1}{2} \sum_i \frac {(g(i) − h(i))^2} {g(i)+h(i)}$$(9)

We then apply second-order Savitzky-Golay filtering [63] to enhance local maxima and smooth out multiple detection peaks in the direction orthogonal to θ. This is equivalent to fitting a cylindrical parabola, whose axis is orientated along direction θ, to a local 2D window surrounding each pixel and replacing the response at the pixel with that estimated by the fit.

然后我们使用二阶Savitzky-Golay滤波，以增强局部最大值，平滑掉垂直于θ方向的多个检测峰值。这等价于拟合一个圆柱形的抛物线，其轴的方向是沿着方向θ，到包围每个像素的一个局部2D窗口，并将在这个像素处的响应替换为由这个拟合估计得到的值。

Figure 4 shows an example. This computation is motivated by the intuition that contours correspond to image discontinuities and histograms provide a robust mechanism for modeling the content of an image region. A strong oriented gradient response means a pixel is likely to lie on the boundary between two distinct regions.

图4给出了一个例子。这个计算是受到下面的直觉推动，即轮廓对应的是图像的不连续性，直方图给出了一个图像区域中内容的建模的稳健机制。一个很强的有向梯度响应，意味着一个像素很可能在两个不同的区域的边缘上。

The Pb detector combines the oriented gradient signals obtained from transforming an input image into four separate feature channels and processing each channel independently. The first three correspond to the channels of the CIE Lab colorspace, which we refer to as the brightness, color a, and color b channels. For grayscale images, the brightness channel is the image itself and no color channels are used.

Pb检测器将这些有向梯度信号结合到一起，这些信号是将一幅输入图像变换到四个分离的特征通道中，对每个通道进行了独立处理。前三个对应CIE Lab色彩空间的通道，我们称之为亮度，色彩a和色彩b通道。对于灰度图像，亮度通道就是图像本身，没有使用色彩通道。

The fourth channel is a texture channel, which assigns each pixel a texton id. These assignments are computed by another filtering stage which occurs prior to the computation of the oriented gradient of histograms. This stage converts the input image to grayscale and convolves it with the set of 17 Gaussian derivative and center-surround filters shown in Figure 5. Each pixel is associated with a (17-dimensional) vector of responses, containing one entry for each filter. These vectors are then clustered using K-means. The cluster centers define a set of image-specific textons and each pixel is assigned the integer id in [1, K] of the closest cluster center. Experiments show choosing K = 32 textons to be sufficient.

第四个通道是一个纹理通道，对每个像素指定了一个纹理基元id。这些指定是由另一个滤波阶段计算出来的，在有向梯度直方图的计算之前。这个阶段将输入图像转换到灰度图，将其与17个高斯导数和中心滤波器卷积，如图5所示。每个像素与一个17维的响应向量相关联，即每个滤波器一个维度。这些向量然后用K均值进行聚类。这个聚类中心定义了一个图像专用的纹理基元集，每个像素都指定了[1, K]范围内的一个整数ID，即最接近的聚类中心。试验表明，选择K=32个纹理基元就很足够。

We next form an image where each pixel has an integer value in [1,K], as determined by its texton id. An example can be seen in Figure 6 (left column, fourth panel from top). On this image, we compute differences of histograms in oriented half-discs in the same manner as for the brightness and color channels.

下面，我们形成一幅图像，每个像素值为[1,K]，由其纹理集运id确定。图6是一个例子（左边的列，从上到下第四个面板）。在这幅图像中，我们在有向的半圆盘中计算直方图之差，和亮度和色彩通道方式一样。

Obtaining G(x,y,θ) for arbitrary input I is thus the core operation on which our local cues depend. In the appendix, we provide a novel approximation scheme for reducing the complexity of this computation.

对任意输入I，得到G(x,y,θ)，这是使用局部线索的核心操作。在附录中，我们给出了一个新的近似方案，降低了这个运算的复杂度。

### 3.2 Multiscale Cue Combination

We now introduce our own multiscale extension of the Pb detector reviewed above. Note that Ren [28] introduces a different, more complicated, and similarly performing multiscale extension in work contemporaneous with our own [3], and also suggests possible reasons Martin et al. [2] did not see performance improvements in their original multiscale experiments, including their use of smaller images and their choice of scales.

我们现在提出我们自己的对Pb检测器的多尺度拓展。注意Ren[28]提出了一种不同的，更复杂的，表现类似的多尺度拓展，时间与我们的工作同时[3]，还说明了Martin等[2]在其原始的多尺度试验中为什么没有看到性能改进的可能原因，包括其使用了更小的图像，以及其选择的尺度。

In order to detect fine as well as coarse structures, we consider gradients at three scales: [σ/2 , σ, 2σ] for each of the brightness, color, and texture channels. Figure 6 shows an example of the oriented gradients obtained for each channel. For the brightness channel, we use σ = 5 pixels, while for color and texture we use σ = 10 pixels. We then linearly combine these local cues into a single multiscale oriented signal:

为检测到精细的结构，以及粗糙的结构，我们考虑在三个尺度的梯度[σ/2 , σ, 2σ]，包括灰度，色彩和纹理通道。图6给出了每个通道的有向梯度的例子。对于亮度通道，我们使用σ = 5像素，而对于色彩和纹理，我们使用σ = 10像素。我们然后将这些局部线索线性的结合到一起，成为一个多尺度有向信号：

$$mPb(x,y,θ) = \sum_s \sum_i α_{i,s} G_{i,σ(i,s)} (x,y,θ)$$(10)

where s indexes scales, i indexes feature channels (brightness, color a, color b, texture), and $G_{i,σ(i,s)}(x, y, θ)$ measures the histogram difference in channel i between two halves of a disc of radius σ(i, s) centered at (x, y) and divided by a diameter at angle θ. The parameters $α_{i,s}$ weight the relative contribution of each gradient signal. In our experiments, we sample θ at eight equally spaced orientations in the interval [0,π). Taking the maximum response over orientations yields a measure of boundary strength at each pixel:

其中s是尺度的索引，i是特征通道的索引（亮度，色彩a，色彩b，纹理），$G_{i,σ(i,s)}(x, y, θ)$度量的是通道i中，在两个半圆盘之间的直方图差异，圆盘半径为σ(i, s)，中心在(x, y)，角度为θ，除以一个半径。参数$α_{i,s}$对每个梯度信号的相对贡献进行加权。在我们的试验中，我们在[0,π)范围内采样8个相同间隔的方向θ。在几个方向中，取最大的响应，在每个像素上得到一个边缘强度度量：

$$mPb(x, y) = max_θ \{ mPb(x, y, θ) \}$$(11)

An optional non-maximum suppression step [22] produces thinned, real-valued contours. 还可以进行非最大抑制，以得到细化的，实值的轮廓。

In contrast to [2] and [28] which use a logistic regression classifier to combine cues, we learn the weights $α_{i,s}$ by gradient ascent on the F-measure using the training images and corresponding ground-truth of the BSDS.

[2,28]使用了逻辑回归分类器，将各种线索结合起来，我们学习权重$α_{i,s}$，使用的是在F度量上的梯度上升，使用的是训练图像和对应的BSDS真值。

### 3.3 Globalization

Spectral clustering lies at the heart of our globalization machinery. The key element differentiating the algorithm described in this section from other approaches [45], [47] is the “soft” manner in which we use the eigenvectors obtained from spectral partitioning.

谱聚类是我们全局化机制的核心。本节中所描述的算法，与其他方法[45,47]区分的关键元素，是我们使用谱分割得到的特征向量的软方式。

As input to the spectral clustering stage, we construct a sparse symmetric affinity matrix W using the intervening contour cue [49],[64],[65], the maximal value of mPb along a line connecting two pixels. We connect all pixels i and j within a fixed radius r with affinity:

作为谱聚类阶段的输入，我们使用介入的轮廓线索，构建了一个稀疏对称亲和矩阵W，mPb在连接两个像素的一条线的最大值。我们连接所有像素i和j用的是一个固定的半径r，其亲和度为：

$$W_{ij} = exp(-max_{p∈\overline{ij}} \{ mPb(p)\}/ρ)$$(12)

where $\overline{ij}$ is the line segment connecting i and j and ρ is a constant. We set r=5 pixels and ρ=0.1. 其中$\overline{ij}$是连接i和j的线段，ρ是一个常数。我们设r=5像素，ρ=0.1。

In order to introduce global information, we define $D_{ii} = \sum_j W_{ij}$ and solve for the generalized eigenvectors {v0, v1, ..., vn} of the system (D − W)v = λDv (2), corresponding to the n+1 smallest eigenvalues 0 = λ0 ≤ λ1 ≤ ... ≤ λn. Figure 7 displays an example with four eigenvectors. In practice, we use n = 16.

为引入全局信息，我们定义了$D_{ii} = \sum_j W_{ij}$，以对系统(D − W)v = λDv (2)求解通用特征向量{v0, v1, ..., vn}，对应着n+1个最小的特征值0 = λ0 ≤ λ1 ≤ ... ≤ λn。图7给出了四个特征向量的例子。实际中，我们使用n=16。

At this point, the standard Normalized Cuts approach associates with each pixel a length n descriptor formed from entries of the n eigenvectors and uses a clustering algorithm such as K-means to create a hard partition of the image. Unfortunately, this can lead to an incorrect segmentation as large uniform regions in which the eigenvectors vary smoothly are broken up. Figure 7 shows an example for which such gradual variation in the eigenvectors across the sky region results in an incorrect partition.

在这一点上，标准的归一化图割方法，对每个像素关联了一个长度为n的描述子，是由n个特征向量的entries形成的，使用聚类方法如K-均值，来创建图像的硬分割。不幸的是，这会导致错误的分割，因为大的一致性的区域可能会分裂，其中的特征向量变化非常缓慢。图7给出了一个例子，在天空区域中，特征向量的缓慢变化导致了一个错误分割。

To circumvent this difficulty, we observe that the eigenvectors themselves carry contour information. Treating each eigenvector v_k as an image, we convolve with Gaussian directional derivative filters at multiple orientations θ, obtaining oriented signals {$∇_θ v_k(x,y)$}. Taking derivatives in this manner ignores the smooth variations that previously lead to errors. The information from different eigenvectors is then combined to provide the “spectral” component of our boundary detector:

为防止这种困难，我们观察到，特征向量本身就带有轮廓信息。将每个特征向量v_k作为一幅图像，我们将其与高斯方向导数滤波器在多个方向θ卷积，得到有向信号{$∇_θ v_k(x,y)$}。以这种方式求导数，忽略了平滑的变化，之前这会导致错误。不同特征向量的信息，然后结合到一起，以给出我们的边缘检测器的谱分量：

$$sPb(x,y,θ) = \sum_{k=1}^n \frac {1}{\sqrt {λ_k}} · ∇_θ v_k(x,y)$$(13)

where the weighting by $1/\sqrt {λ_k}$ is motivated by the physical interpretation of the generalized eigenvalue problem as a mass-spring system [66]. Figures 7 and 8 present examples of the eigenvectors, their directional derivatives, and the resulting sPb signal.

其中$1/\sqrt {λ_k}$加权是受到通用特征值问题作为重量弹簧系统的物理解释的启发。图7和图8给出了特征向量，其方向导数，和得到的sPb信号的例子。

The signals mPb and sPb convey different information, as the former fires at all the edges while the latter extracts only the most salient curves in the image. We found that a simple linear combination is enough to benefit from both behaviors. Our final globalized probability of boundary is then written as a weighted sum of local and spectral signals:

信号mPb和sPb传递的是不同的信息，前者对所有边缘起作用，而后者只提取图像中最明显的曲线。我们发现，简单的线性组合就可以从两种行为中受益。我们最终的全局化的边缘概率，就写为局部和谱信号的加权和：

$$gPb(x,y,θ)=\sum_s \sum_i β_{i,s} G_{i,σ(i,s)} (x,y,θ) + γ·sPb(x,y,θ)$$(14)

We subsequently rescale gPb using a sigmoid to match a probabilistic interpretation. As with mPb (10), the weights $β_{i,s}$ and γ are learned by gradient ascent on the F-measure using the BSDS training images.

我们然后使用sigmoid来改变gPb的尺度，以匹配一个概率解释。至于mPb(10)，权重$β_{i,s}$和γ是在F-度量上使用BSDS训练图像通过梯度下降学习到的。

### 3.4 Results

Qualitatively, the combination of the multiscale cues with our globalization machinery translates into a reduction of clutter edges and completion of contours in the output, as shown in Figure 9.

定性的，多尺度线索和我们的全局机制的组合，成为了杂乱边缘的简化，和输出中的轮廓的补全，如图9所示。

Figure 10 breaks down the contributions of the multiscale and spectral signals to the performance of gPb. These precision-recall curves show that the reduction of false positives due to the use of global information in sPb is concentrated in the high thresholds, while gPb takes the best of both worlds, relying on sPb in the high precision regime and on mPb in the high recall regime.

图10将gPb中多尺度和谱信号的贡献分解了开来。这些准确率-召回率的曲线表明，假阳性的减少，是因为在sPb中使用的全局信息关注在高阈值中，而gPb将两个领域中最好的结合了起来，在高精度领域中依赖于sPb，在高召回率中依赖于mPb。

Looking again at the comparison of contour detectors on the BSDS300 benchmark in Figure 1, the mean improvement in precision of gPb with respect to the single scale Pb is 10% in the recall range [0.1, 0.9].

回看一下图1中在BSDS300基准测试中轮廓检测器的比较，gPb相对于单尺度Pb在精度上的改进，在召回率[0.1, 0.9]的范围内是10%。

## 4 Segmentation

The nonmax suppressed gPb contours produced in the previous section are often not closed and hence do not partition the image into regions. These contours may still be useful, e.g. as a signal on which to compute image descriptors. However, closed regions offer additional advantages. Regions come with their own scale estimates and provide natural domains for computing features used in recognition. Many visual tasks can also benefit from the complexity reduction achieved by transforming an image with millions of pixels into a few hundred or thousand “superpixels” [67].

上一节中产生的，gPb的轮廓经过非最大抑制，通常不是那么闭合，因此并没有将图像分割成区域。这些轮廓可能仍然有用，如，作为一种信号来计算图像描述子。但是，闭合的区域会给出额外的好处。区域有其自己的尺度估计，并很自然的给出用于计算特征的领域，用于识别。将一幅有着数百万像素的图像，转换成几百或几千个超像素，这样的复杂度降低，会使很多视觉任务受益。

In this section, we show how to recover closed contours, while preserving the gains in boundary quality achieved in the previous section. Our algorithm, first reported in [4], builds a hierarchical segmentation by exploiting the information in the contour signal. We introduce a new variant of the watershed transform [68], [69], the Oriented Watershed Transform (OWT), for producing a set of initial regions from contour detector output. We then construct an Ultrametric Contour Map (UCM) [35] from the boundaries of these initial regions.

本节中，我们展示了怎样恢复出闭合的轮廓，同时保持上一节中边缘质量的收益。我们的算法，首先在[4]中给出，通过利用轮廓信号中的信息，构建了一个层次化的分割。我们提出了watershed变换的一种新变体，有向watershed变换(OWT)，从轮廓检测器的输出中，给出了一系列初始区域。我们然后从这些初始区域的边缘中，构建了一个超度量的轮廓图。

This sequence of operations (OWT-UCM) can be seen as generic machinery for going from contours to a hierarchical region tree. Contours encoded in the resulting hierarchical segmentation retain real-valued weights indicating their likelihood of being a true boundary. For a given threshold, the output is a set of closed contours that can be treated as either a segmentation or as a boundary detector for the purposes of benchmarking.

这些顺序的操作(OWT-UCM)可以视为从轮廓到层次化的区域树的通用机制。在得到的层次化分割中编码的轮廓，保持了实值的权重，是其作为一个真实边缘的可能性的指示。对于一个给定的阈值，输出是一个闭合轮廓的集合，可以视为一个分割，或边缘检测器，以进行基准测试。

To describe our algorithm in the most general setting, we now consider an arbitrary contour detector, whose output E(x,y,θ) predicts the probability of an image boundary at location (x, y) and orientation θ.

为在最通用的设置中描述我们的算法，我们现在考虑一个任意的轮廓检测器，其输出E(x,y,θ)预测了在(x,y)和方向θ上是图像边缘的概率。

### 4.1 Oriented Watershed Transform

Using the contour signal, we first construct a finest partition for the hierarchy, an over-segmentation whose regions determine the highest level of detail considered. This is done by computing E(x,y) = max_θ E(x,y,θ), the maximal response of the contour detector over orientations. We take the regional minima of E(x,y) as seed locations for homogeneous segments and apply the watershed transform used in mathematical morphology [68], [69] on the topographic surface defined by E(x,y). The catchment basins of the minima, denoted P_0, provide the regions of the finest partition and the corresponding watershed arcs, K_0, the possible locations of the boundaries.

使用轮廓信号，我们首先为这个层次构建了最精细的分割，这是一个过分割，其区域确定了考虑的最高层次的细节。这是通过计算E(x,y) = max_θ E(x,y,θ)得到的，即轮廓检测器在各个方向上的最高响应。我们以E(x,y)的区域最小值，作为均一性片段的种子位置，对E(x,y)中定义的地形曲面，使用数学形态学中的watershed变换。汇水区盆地的最小值点，表示为P_0，给出了最精细的分割的区域，和对应的watershed弧，K_0，边缘的可能位置。

Figure 11 shows an example of the standard watershed transform. Unfortunately, simply weighting each arc by the mean value of E(x,y) for the pixels on the arc can introduce artifacts. The root cause of this problem is the fact that the contour detector produces a spatially extended response around strong boundaries. For example, a pixel could lie near but not on a strong vertical contour. If this pixel also happens to belong to a horizontal watershed arc, that arc would be erroneously upweighted. Several such cases can be seen in Figure 11. As we flood from all local minima, the initial watershed oversegmentation contains many arcs that should be weak, yet intersect nearby strong boundaries.

图11展示了标准watershed变换的一个例子。不幸的是，简单的通过在弧上的每个像素的E(x,y)的均值，对每个弧进行加权，会引入伪影。这个问题的根源在于，轮廓检测器在强边缘的附近会产生空间上拓展的响应。比如，一个像素可能在一个很强的垂直轮廓的周围，但并不在轮廓上。如果这个像素同时还碰巧属于一个水平的watershed弧，那个弧可能会错误的过度加权。图11中可以看到几个这样的例子。在我们从所有的局部极小值进行flood时，初始的watershed过分割包含很多弧，应当是很弱的，但与附近的强边缘相交。

To correct this problem, we enforce consistency between the strength of the boundaries of K0 and the underlying E(x,y,θ) signal in a modified procedure, which we call the Oriented Watershed Transform (OWT), illustrated in Figure 12. As the first step in this reweighting process, we estimate an orientation at each pixel on an arc from the local geometry of the arc itself. These orientations are obtained by approximating the watershed arcs with line segments as shown in Figure 13. We recursively subdivide any arc which is not well fit by the line segment connecting its endpoints. By expressing the approximation criterion in terms of the maximum distance of a point on the arc from the line segment as a fraction of the line segment length, we obtain a scale-invariant subdivision. We assign each pixel (x,y) on a subdivided arc the orientation o(x, y) ∈ [0, π) of the corresponding line segment.

为修正这一问题，我们对边缘强度K0和潜在的E(x,y,θ)信号上以一种修正的过程施加了一致性，我们称之为有向Watershed变换(OWT)，在图12中进行了描述。作为这个重新赋权的过程的第一步，我们在一个弧上的每个像素上，从这个弧本身的局部几何中，估计了一个方向。这些方向是通过用图13中所示的线段近似watershed弧得到的。我们递归的将没有被连接其端点的线段很好的拟合的弧进行分割。我们用弧上的一个点到线段的最大距离，比线段长度本身的分数，作为近似准则，我们得到了一个尺度不变的细分。我们对在一个细分的弧上的每个像素(x,y)指定对应线段的一个方向o(x, y) ∈ [0, π)。

Next, we use the oriented contour detector output E(x, y, θ), to assign each arc pixel (x, y) a boundary strength of E(x, y, o(x, y)). We quantize o(x, y) in the same manner as θ, so this operation is a simple lookup. Finally, each original arc in K0 is assigned weight equal to average boundary strength of the pixels it contains. Comparing the middle left and far right panels of Figure 12 shows this reweighting scheme removes artifacts.

下一步，我们使用有向的轮廓检测器输出E(x, y, θ)，来对每个弧像素(x,y)指定一个边缘强度E(x, y, o(x, y))。我们对o(x,y)进行量化，方式与θ一样，所以这个运算就是一个简单的查表。最后，在K0中每个原始的弧都指定了权重，等于其包含的所有像素的边缘强度的平均。比较图12中的左边和右边面板，表明这种重新赋权的方案去除了伪影。

### 4.2 Ultrametric Contour Map

Contours have the advantage that it is fairly straightforward to represent uncertainty in the presence of a true underlying contour, i.e. by associating a binary random variable to it. One can interpret the boundary strength assigned to an arc by the Oriented Watershed Transform (OWT) of the previous section as an estimate of the probability of that arc being a true contour.

轮廓有一个优势，就是在存在真实的潜在的轮廓时，要表示不确定性，是相对很直接的，如，与之关联一个二值随机变量。可以通过前一节的OWT作为弧是一个真正的轮廓的概率的估计，解释指定给一个弧的边缘强度。

It is not immediately obvious how to represent uncertainty about a segmentation. One possibility, which we exploit here, is the Ultrametric Contour Map (UCM) [35] which defines a duality between closed, non-self-intersecting weighted contours and a hierarchy of regions. The base level of this hierarchy respects even weak contours and is thus an oversegmentation of the image. Upper levels of the hierarchy respect only strong contours, resulting in an under-segmentation. Moving between levels offers a continuous trade-off between these extremes. This shift in representation from a single segmentation to a nested collection of segmentations frees later processing stages to use information from multiple levels or select a level based on additional knowledge.

但如何表示一个分割的不确定性，则不是那么很明显。我们探索的一个可能性，是无度量轮廓图(UCM)，这在闭合的、不会自己相交的加权轮廓和区域层次之间定义了一个对偶。这个层次的基准层次甚至是弱轮廓，因此是图像的一个过分割。更高的层次只包含强的轮廓，得到了一个弱分割。在各个层次之间移动，给出了这些极端之间的连续折中。这种从一个分割，到一个嵌套的分割的集合的转变，将后续的处理过程，即利用多个层次的信息或选择一个基于额外知识的层次，变得很自由。

Our hierarchy is constructed by a greedy graph-based region merging algorithm. We define an initial graph G = (P0, K0, W(K0)), where the nodes are the regions P0, the links are the arcs K0 separating adjacent regions, and the weights W(K0) are a measure of dissimilarity between regions. The algorithm proceeds by sorting the links by similarity and iteratively merging the most similar regions. Specifically:

我们的层次是由一个基于贪婪图的区域合并算法构建的。我们定义一个初始的图G = (P0, K0, W (K0))，其中节点是区域P0，连接是弧K0，将相邻的区域分隔开来，权重W(K0)是区域之间的不相似性的度量。算法对连接通过相似性进行排序，对最相似的区域进行迭代合并。具体的：

1) Select minimum weight contour: 选择权重最小的轮廓：

$$C^* = argmin_{C∈K_0} W(C)$$

2) Let $R_1, R_2∈P_0$ be the regions separated by $C^*$.

3) Set $R=R_1∪R_2$, and update:

$$P_0←P_0 \ \{ R_1, R_2 \} ∪ \{R\}, and K_0←K_0 \ \{ C^* \}$$

4) Stop if K_0 is empty. Otherwise, update weights W(K0) and repeat.

This process produces a tree of regions, where the leaves are the initial elements of P0, the root is the entire image, and the regions are ordered by the inclusion relation.

这个过程产生一个区域树，其中树叶是P0的初始元素，根是整幅图像，区域是通过包含关系进行排序的。

We define dissimilarity between two adjacent regions as the average strength of their common boundary in K0, with weights W(K0) initialized by the OWT. Since at every step of the algorithm all remaining contours must have strength greater than or equal to those previously removed, the weight of the contour currently being removed cannot decrease during the merging process. Hence, the constructed region tree has the structure of an indexed hierarchy and can be described by a dendrogram, where the height H(R) of each region R is the value of the dissimilarity at which it first appears. Stated equivalently, H(R) = W(C) where C is the contour whose removal formed R. The hierarchy also yields a metric on P0×P0, with the distance between two regions given by the height of the smallest containing segment:

我们将两个相邻区域的不相似性定义为，K0中它们共同边缘的平均强度，还有OWT初始化的权重W(K0)。由于在算法的每一步中，所有剩余的轮廓，其强度都会比之前移除掉的轮廓的强度要高，目前要移除的轮廓的权重在合并的过程中不会下降。因此，构建的区域树是一个带有索引的层次结构，可以用树状图来描述，其中每个区域R的高度H(R)是其一开始出现的不相似性值。等价的说，H(R) = W(C)，其中C是去除掉的以形成R的那个轮廓。这个层次还产生了P0×P0上的一个度量，其两个区域之间的距离，是由包含分段的最小高度给出

$$D(R_1, R_2) = min\{ H(R) : R_1, R_2 ⊆ R \}$$(15)

This distance satisfies the ultrametric property: 这个距离满足无度量属性：

$$D(R_1, R_2) ≤ max(D(R_1, R), D(R, R_2))$$(16)

since if R is merged with R1 before R2, then D(R1, R2) = D(R,R2), or if R is merged with R2 before R1, then D(R1, R2) = D(R1, R). As a consequence, the whole hierarchy can be represented as an Ultrametric Contour Map (UCM) [35], the real-valued image obtained by weighting each boundary by its scale of disappearance.

如果R与R1的合并在R2之前，则D(R1, R2) = D(R,R2)，或如果R与R2的融合在R1之前，则D(R1, R2) = D(R1, R)。结果是，整个层次可以表示为无度量轮廓图(UCM)，通过使用其消失的尺度对每个边缘进行加权得到的实值图像。

Figure 14 presents an example of our method. The UCM is a weighted contour image that, by construction, has the remarkable property of producing a set of closed curves for any threshold. Conversely, it is a convenient representation of the region tree since the segmentation at a scale k can be easily retrieved by thresholding the UCM at level k. Since our notion of scale is the average contour strength, the UCM values reflect the contrast between neighboring regions.

图14给出了我们的方法的一个例子。UCM是一个加权的轮廓图，构建的时候有非常好的性质，会对任何阈值产生闭合曲线集合。相反的，它是区域树的方便的表示，由于在尺度k上的分割可以很容易的，通过在UCM上在层次k上的阈值，得到。由于我们的关于尺度的概念是平均轮廓强度，UCM值反应的是相邻区域之间的对比度。

### 4.3 Results

While the OWT-UCM algorithm can use any source of contours for the input E(x,y,θ) signal (e.g. the Canny edge detector before thresholding), we obtain best results by employing the gPb detector [3] introduced in Section 3. We report experiments using both gPb as well as the baseline Canny detector, and refer to the resulting segmentation algorithms as gPb-owt-ucm and Canny-owt-ucm, respectively.

OWT-UCM可以使用任何轮廓源作为输入的E(x,y,θ)信号（如，使用阈值之前的Canny边缘检测器），但使用第3部分提出的gPb检测器，我们可以得到最好的结果。我们使用gPb以及基准的Canny检测器给出试验结果，并称分割结果分别为gPb-owt-ucm和Canny-owt-ucm。

Figures 15 and 16 illustrate results of gPb-owt-ucm on images from the BSDS500. Since the OWT-UCM algorithm produces hierarchical region trees, obtaining a single segmentation as output involves a choice of scale. One possibility is to use a fixed threshold for all images in the dataset, calibrated to provide optimal performance on the training set. We refer to this as the optimal dataset scale (ODS). We also evaluate performance when the optimal threshold is selected by an oracle on a per-image basis. With this choice of optimal image scale (OIS), one naturally obtains even better segmentations.

图15和16给出了gPb-owt-ucm在BSDS500上的结果。由于OWT-UCM算法产生了层次化的区域树，得到单个分割作为输出，涉及到选择尺度。一个可能性是对数据集中的所有图像使用固定的阈值，进行校准，以在训练集上获得最优性能。我们称这个是最优数据集尺度(ODS)。最优阈值是在逐幅图像的基准上进行选择，我们也评估了性能。在最优图像尺度(OIS)的选择上，很自然的可以得到更好的分割结果。

### 4.4 Evaluation

To provide a basis of comparison for the OWT-UCM algorithm, we make use of the region merging (Felz- Hutt) [32], Mean Shift [34], Multiscale NCuts [33], and SWA [31], [52] segmentation methods reviewed in Section 2.2. We evaluate each method using the boundary-based precision-recall framework of [2], as well as the Variation of Information, Probabilistic Rand Index, and segment covering criteria discussed in Section 2.3. The BSDS serves as ground-truth for both the boundary and region quality measures, since the human-drawn boundaries are closed and hence are also segmentations.

为给比较OWT-UCM算法提供一个基准，我们使用了区域合并，Mean Shift，多尺度NCuts，和SWA分割方法，这在2.2节进行了回顾。我们使用了基于边缘的精度-召回框架来评估每种方法，以及2.3节中讨论的Variation of Information, Probabilistic Rand Index和分割覆盖准则。BSDS的作用是作为边缘和区域质量的度量的真值，因为人类画的边缘是闭合的，因此也是分割。

#### 4.4.1 Boundary Quality

Remember that the evaluation methodology developed by [2] measures detector performance in terms of precision, the fraction of true positives, and recall, the fraction of ground-truth boundary pixels detected. The global F-measure, or harmonic mean of precision and recall at the optimal detector threshold, provides a summary score.

[2]中提出的评估方法是按照精度、真阳性的比例、召回、检测到的真值边缘像素来度量检测器的性能。全局的F度量，或在最佳检测器阈值上的精度和召回的调和平均，给出了一个总结的分数。

In our experiments, we report three different quantities for an algorithm: the best F-measure on the dataset for a fixed scale (ODS), the aggregate F-measure on the dataset for the best scale in each image (OIS), and the average precision (AP) on the full recall range (equivalently, the area under the precision-recall curve). Table 1 shows these quantities for the BSDS. Figures 2 and 17 display the full precision-recall curves on the BSDS300 and BSDS500 datasets, respectively. We find retraining on the BSDS500 to be unnecessary and use the same parameters learned on the BSDS300. Figure 18 presents side by side comparisons of segmentation algorithms.

在我们的试验中，我们对一个算法给出三个不同的量：在数据集中对于一个固定尺度(ODS)的最佳F-度量，在数据集中对于每幅图像的最佳尺度(OIS)的累积F度量，在整个召回范围内的平均精度(AP)（等价的说，在准确率-召回曲线下的整个区域）。表1给出了在BSDS数据集上的这些量。图2和图17分别在BSDS300和BSDS500数据集上展示了完整的准确率-召回曲线。我们发现，在BSDS500上重新训练是不太需要的，使用了在BSDS300上学习到的相同的参数。图18给出了分割算法的逐对比较。

Of particular note in Figure 17 are pairs of curves corresponding to contour detector output and regions produced by running the OWT-UCM algorithm on that output. The similarity in quality within each pair shows that we can convert contours into hierarchical segmentations without loss of boundary precision or recall.

在图17中是，轮廓检测器的输出，和在那个输出上运行OWT-UCM算法得到的区域的成对曲线。在每一对中的数量相似性表明，我们能将轮廓转换成层次化的分割，而不损失边缘精度或召回。

#### 4.4.2 Region Quality

Table 2 presents region benchmarks on the BSDS. For a family of machine segmentations {Si}, associated with different scales of a hierarchical algorithm or different sets of parameters, we report three scores for the covering of the ground-truth by segments in {Si}. These correspond to selecting covering regions from the segmentation at a universal fixed scale (ODS), a fixed scale per image (OIS), or from any level of the hierarchy or collection {Si} (Best). We also report the Probabilistic Rand Index and Variation of Information benchmarks.

表2在BSDS上给出了区域的基准测试。对于一族机器分割{Si}，与一个层次化算法或不同的参数集合的不同尺度相关，我们给出{Si}中的分割覆盖真值的三个分数。这对应着在ODS，OIS或任意层次的分割的覆盖区域。我们还给出Probabilistic Rand Index和Variation of Information基准测试。

While the relative ranking of segmentation algorithms remains fairly consistent across different benchmark criteria, the boundary benchmark (Table 1 and Figure 17) appears most capable of discriminating performance. This observation is confirmed by evaluating a fixed hierarchy of regions such as the Quad-Tree (with 8 levels). While the boundary benchmark and segmentation covering criterion clearly separate it from all other segmentation methods, the gap narrows for the Probablilistic Rand Index and the Variation of Information.

分割算法在不同的基准测试准则下的相对排名相对一致，而边缘基准测试显得最能区分性能。这种观察通过评估一个固定的区域层次比如Quad-Tree，可以得到确认。边缘基准测试，和分割覆盖准则，将其与其他分割方法明显的分隔开来，但对于Probablilistic Rand Index和the Variation of Information，差距则更窄一些。

#### 4.4.3 Additional Datasets

We concentrated experiments on the BSDS because it is the most complete dataset available for our purposes, has been used in several publications, and has the advantage of providing multiple human-labeled segmentations per image. Table 3 reports the comparison between Canny-owt-ucm and gPb-owt-ucm on two other publicly available datasets:

我们将BSDS上的试验拼接了起来，因为这是满足我们目的的最完整的可用数据集，在几篇文章中使用过，对每幅图像提供了多个人标注的分割结果。表3给出了Canny-owt-ucm和gPb-owt-ucm在另外两个公开可用的数据集上的比较结果：

- MSRC [71]

The MSRC object recognition database is composed of 591 natural images with objects belonging to 21 classes. We evaluate performance using the ground-truth object instance labeling of [11], which is cleaner and more precise than the original data.

MSRC目标检测数据集由591幅自然图像组成，目标属于21个类别。我们使用[11]的真值目标实例标注来评估性能，比原始数据更加干净，更加精确。

- PASCAL 2008 [8]

We use the train and validation sets of the segmentation task on the 2008 PASCAL segmentation challenge, composed of 1023 images. This is one of the most difficult and varied datasets for recognition. We evaluate performance with respect to the object instance labels provided. Note that only objects belonging to the 20 categories of the challenge are labeled, and 76% of all pixels are unlabeled.

我们使用2008 PASCAL分割挑战赛上的分割任务中的训练和验证集，由1023幅图像组成。这是用于识别的最难变化最多的数据集之一。我们用给出的目标实例标签评估了性能。注意挑战赛上的20个类别的目标得到了标注，所有像素的76%是没有标注的。

#### 4.4.4 Summary

The gPb-owt-ucm segmentation algorithm offers the best performance on every dataset and for every benchmark criterion we tested. In addition, it is straight-forward, fast, has no parameters to tune, and, as discussed in the following sections, can be adapted for use with top-down knowledge sources.

gPb-owt-ucm分割算法在每个数据集上，对我们测试的每个基准测试准则，都给出了最好的结果。另外，这非常直接，快速，不需要调整任何参数，就像在后续的小节中讨论的，可以调整与自上而下的知识一起使用。

## 5 Interactive Segmentation

Until now, we have only discussed fully automatic image segmentation. Human assisted segmentation is relevant for many applications, and recent approaches rely on the graph-cuts formalism [72], [73], [74] or other energy minimization procedure [75] to extract foreground regions.

直到现在，我们只讨论了全自动的图像分割。人类辅助分割与很多应用相关，最近的方法依赖于图割，或其他能量最小化过程，以提取前景区域。

For example, [72] cast the task of determining binary foreground/background pixel assignments in terms of a cost function with both unary and pairwise potentials. The unary potentials encode agreement with estimated foreground or background region models and the pairwise potentials bias neighboring pixels not separated by a strong boundary to have the same label. They transform this system into an equivalent minimum cut/maximum flow graph partitioning problem through the addition of a source node representing the foreground and a sink node representing the background. Edge weights between pixel nodes are defined by the pairwise potentials, while the weights between pixel nodes and the source and sink nodes are determined by the unary potentials. User-specified hard labeling constraints are enforced by connecting a pixel to the source or sink with sufficiently large weight. The minimum cut of the resulting graph can be computed efficiently and produces a cost-optimizing assignment.

比如，[72]将确定前景/背景像素指定的任务，视为一个代价函数，有一元势能和二元势能。一元势能是，估计的前景和背景区域模型，二元势能对于没有被很强的边缘分隔的相邻像素，给定了同样的标签。它们加入了一个源节点，表示前景，一个汇聚节点，表示背景，将这个系统转化成等价的最小割/最大流图分割问题。像素节点间的边缘权重，通过成对的势能定义，而像素节点和源、汇聚节点间的权重，由一元势能确定。用户指定的硬标签约束，通过用足够大的权重连接源或汇聚来施加。得到的图的最小割可以进行高效计算，并得到一个代价优化指定。

It turns out that the segmentation trees generated by the OWT-UCM algorithm provide a natural starting point for user-assisted refinement. Following the procedure of [76], we can extend a partial labeling of regions to a full one by assigning to each unlabeled region the label of its closest labeled region, as determined by the ultrametric distance (15). Computing the full labeling is simply a matter of propagating information in a single pass along the segmentation tree. Each unlabeled region receives the label of the first labeled region merged with it. This procedure, illustrated in Figure 19, allows a user to obtain high quality results with minimal annotation.

OWT-UCM算法生成的分割树，对用户协助的改进给出了一个自然的开始点。按照[76]的过程，我们可以将一个区域的部分标注拓展到完整的标注，对每个未标注的区域指定其最接近的标注区域的标注，由无度量距离来确定。计算完整的标注只是将信息沿着分割树进行传播的问题。每个无标记的区域，得到的是第一个与之合并的区域的标注。这个过程，如图19所示，使一个用户在最少标注的情况下，得到最佳的结果。

## 6 Multiscale For Object Analysis

Our contour detection and segmentation algorithms capture multiscale information by combining local gradient cues computed at three different scales, as described in Section 3.2. We did not see any performance benefit on the BSDS by using additional scales. However, this fact is not an invitation to conclude that a simple combination of a limited range of local cues is a sufficient solution to the problem of multiscale image analysis. Rather, it is a statement about the nature of the BSDS. The fixed resolution of the BSDS images and the inherent photographic bias of the dataset lead to the situation in which a small range of scales captures the boundaries humans find important.

我们的轮廓检测和分割算法，通过综合三个不同尺度的局部梯度线索，来捕获多尺度信息，如3.2节所示。如果多使用尺度，我们在BSDS上没有看到任何性能改进。但是，这并不说明，有限范围的局部线索的简单组合，就足以解决多尺度图像分析的问题。这是BSDS数据集的一种本质反应。BSDS图像的固定分辨率，和固有的图像偏好，得到了这样一种情况，即小范围的尺度就可以捕获人类觉得重要的边缘。

Dealing with the full variety one expects in high resolution images of complex scenes requires more than a naive weighted average of signals across the scale range. Such an average would blur information, resulting in good performance for medium-scale contours, but poor detection of both fine-scale and large-scale contours. Adaptively selecting the appropriate scale at each location in the image is desirable, but it is unclear how to estimate this robustly using only bottom-up cues.

处理高分辨率图像的完全的多样性，包含复杂的场景，需要的不仅仅是在尺度范围内的简单的加权平均。这样一种平均会使信息模糊，对中间尺度的轮廓得到很好的性能，但对于精细尺度和大尺度的轮廓性能就会很差。在图像的每个位置上自适应的选择合适的尺度，是理想的，但只用自下而上的线索，怎样估计这个，是不清楚的。

For some applications, in particular object detection, we can instead use a top-down process to guide scale selection. Suppose we wish to apply a classifier to determine whether a subwindow of the image contains an instance of a given object category. We need only report a positive answer when the object completely fills the subwindow, as the detector will be run on a set of windows densely sampled from the image. Thus, we know the size of the object we are looking for in each window and hence the scale at which contours belonging to the object would appear. Varying the contour scale with the window size produces the best input signal for the object detector. Note that this procedure does not prevent the object detector itself from using multiscale information, but rather provides the correct central scale.

对于一些应用，特别是目标检测，我们可以使用自上而下的过程来引导尺度选择。假设我们希望使用一个分类器，来确定图像的一个子窗口是否包含一个给定目标类别的实例。我们只需要在目标完全填充了一个子窗口时，给出一个肯定的答案，因为检测器会在一系列窗口中运行，这些窗口是在图像中密集取样得到的。因此，我们知道我们在每个窗口中要查找的目标的大小，因此也知道属于这个目标的轮廓会消失的尺度。轮廓尺度随着窗口大小变化，会对目标检测器产生最佳的输入信号。注意，这个过程并没有防止目标检测器本身使用多尺度信息，而是给出了正确的中间尺度。

As each segmentation internally uses gradients at three scales, [σ/2, σ, 2σ], by stepping by a factor of 2 in scale between segmentations, we can reuse shared local cues. The globalization stage (sPb signal) can optionally be customized for each window by computing it using only a limited surrounding image region. This strategy, used here, results in more work overall (a larger number of simpler globalization problems), which can be mitigated by not sampling sPb as densely as one samples windows.

在每个分割中，内部都使用了三个尺度，[σ/2, σ, 2σ]，我们还可以重复使用共享的局部线索。全局化步骤(sPB信号)可以有选择的对每个窗口定制，只使用周围的图像区域进行计算。这里使用的这个策略，总体上导致了更多的工作（更简单的全局化问题更多），这可以通过不要密集的采样来缓解。

Figure 20 shows an example using images from the PASCAL dataset. Bounding boxes displayed are slightly larger than each object to give some context. Multiscale segmentation shows promise for detecting fine-scale objects in scenes as well as making salient details available together with large-scale boundaries.

图20给出了使用PASCAL数据集的图像的例子。展示出的边界框比每个目标略大，以给出一定的上下文。多尺度分割给出了检测细尺度目标的希望，以及检测显著的细节的作用。