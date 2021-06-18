# Machine Learning for High-Speed Corner Detection

Edward Rosten and Tom Drummond

Department of Engineering, Cambridge University, UK

## 0. Abstract

Where feature points are used in real-time frame-rate applications, a high-speed feature detector is necessary. Feature detectors such as SIFT (DoG), Harris and SUSAN are good methods which yield high quality features, however they are too computationally intensive for use in real-time applications of any complexity. Here we show that machine learning can be used to derive a feature detector which can fully process live PAL video using less than 7% of the available processing time. By comparison neither the Harris detector (120%) nor the detection stage of SIFT (300%) can operate at full frame rate.

特征点在实时帧率应用中使用时，就必须要一个高速特征检测器。像SIFT(DoG), Harris和SUSAN这样的特征检测器，都是很好的方法，会得到高质量的特征，但是它们计算量很大，无法在实时应用中使用。这里我们证明了，可以使用机器学习来推导得到一个特征检测器，可以完整处理实时PAL视频，时间少于7%的可用处理时间。比较起来，Harris检测器为120%，SIFT为300%，它们都不能在完全帧率下进行使用。

Clearly a high-speed detector is of limited use if the features produced are unsuitable for downstream processing. In particular, the same scene viewed from two different positions should yield features which correspond to the same real-world 3D locations[1]. Hence the second contribution of this paper is a comparison corner detectors based on this criterion applied to 3D scenes. This comparison supports a number of claims made elsewhere concerning existing corner detectors. Further, contrary to our initial expectations, we show that despite being principally constructed for speed, our detector significantly outperforms existing feature detectors according to this criterion.

很明显，如果产生的特征不适用于下游处理，高速检测器用处有限。特别是，从两个不同的位置观察同样的场景，应当得到对应同样的真实世界3D位置的特征。因此本文的第二贡献是，应用到3D场景中，基于这个规则的角点检测器的比较。这个比较也验证了有关已有的角点检测器的数个结论。而且，与我们的初始期望相反，我们展示了，尽管主要是为了加速构建，根据这个规则，我们的检测器还显著超过了已有的特征检测器。

## 1. Introduction

Corner detection is used as the first step of many vision tasks such as tracking, SLAM (simultaneous localisation and mapping), localisation, image matching and recognition. Hence, a large number of corner detectors exist in the literature. With so many already available it may appear unnecessary to present yet another detector to the community; however, we have a strong interest in real-time frame rate applications such as SLAM in which computational resources are at a premium. In particular, it is still true that when processing live video streams at full frame rate, existing feature detectors leave little if any time for further processing, even despite the consequences of Moore’s Law.

角点检测时很多视觉任务的第一步，如跟踪，SLAM，定位，图像匹配和识别。因此，文献中有大量角点检测器。有了这么多已经可用的，似乎提出另一种检测器，不是很必要；但是，我们有对实时帧率应用非常感兴趣，如SLAM，其中计算资源非常重要。特别是，当以完整帧率处理实时视频流时，现有的特征提取器很难进行实时处理。

Section 2 of this paper demonstrates how a feature detector described in earlier work can be redesigned employing a machine learning algorithm to yield a large speed increase. In addition, the approach allows the detector to be generalised, producing a suite of high-speed detectors which we currently use for real-time tracking [2] and AR label placement [3].

本文第2部分证明了，之前工作描述的特征提取器，可以采用机器学习算法进行重新设计，得到速度的大幅提升。另外，这个方法使检测器可以被推广，生成一族高速检测器，可用于实时跟踪，和AR标签放置。

To show that speed can been obtained without necessarily sacrificing the quality of the feature detector we compare our detector, to a variety of well-known detectors. In Section 3 this is done using Schmid’s criterion [1], that when presented with different views of a 3D scene, a detector should yield (as far as possible) corners that correspond to the same features in the scene. Here we show how this can be applied to 3D scenes for which an approximate surface model is known.

为证明速度的提升可以不用牺牲特征提取器的质量，我们将我们的检测器与一些著名检测器进行比较。在第3部分，我们用Schmid准则来做这个工作，即给定一个3D场景不同视角的图像时，检测器得到的角点，应当对应场景中的同样特征。这里我们展示了怎样在3D场景中应用，其中近似的表面模型是已知的。

### 1.1. Previous Work

The majority of feature detection algorithms work by computing a corner response function (C) across the image. Pixels which exceed a threshold cornerness value (and are locally maximal) are then retained.

主要的特征检测算法工作原理都是，在图像中计算一个角点响应函数C。超过角点值阈值的像素（并且是局部最大值的）就保留下来。

Moravec [4] computes the sum-of-squared-differences (SSD) between a patch around a candidate corner and patches shifted a small distance in a number of directions. C is then the smallest SSD so obtained, thus ensuring that extracted corners are those locations which change maximally under translations.

Moravec[4]在一个角点候选附近的图像块和在几个方向偏移一小段距离的图像块之间计算SSD。C就是得到的最小的SSD，因此确保了提取的角点是那些在平移下变化最大的位置。

Harris[5] builds on this by computing an approximation to the second derivative of the SSD with respect to the shift. The approximation is:

Harris[5]在此基础上，计算了SSD对偏移的二阶导数的近似。这个近似是：

$$H = [\hat {I_x^2}, \hat {I_x I_y}; \hat {I_x I_y}, \hat {I_y^2}]$$(1)

where $\hat I$ denotes averaging performed over the image patch (a smooth circular window can be used instead of a rectangle to perform the averaging resulting in a less noisy, isotropic response). Harris then defines the corner response to be

这里$\hat I$表示在图像块上进行平均（也可以使用光滑的圆形窗口，而不使用矩形窗口来进行平均，得到没那么多噪声的各向同性的响应）。Harris然后将角点响应定义为

$$C = |H| - k(trace H)^2$$(2)

This is large if both eigenvalues of H are large, and it avoids explicit computation of the eigenvalues. It has been shown[6] that the eigenvalues are an approximate measure of the image curvature.

如果H的两个特征值都很大，这个值就很大，而且避免了特征值的显式计算。已经证明了，特征值是图像曲率的近似度量。

Based on the assumption of affine image deformation, a mathematical analysis led Shi and Tomasi[7] conclude that it is better to use the smallest eigen value of H as the corner strength function:

基于仿射图像形变的假设，Shi和Tomasi[7]用数学分析得到结论，用H的最小特征值代表角点强度函数更加合理：

$$C = min (λ_1, λ_2)$$(3)

A number of suggestion have [5, 7, 8, 9] been made for how to compute the corner strength from H and these have been all shown [10] to be equivalent to various matrix norms of H.

[5,7,8,9]作出了几个建议，怎样从H中计算角点强度，这些都表明[10]，与H的各种矩阵范数是等价的。

Zheng et al.[11] perform an analysis of the computation of H, and find some suitable approximations which allow them to obtain a speed increase by computing only two smoothed images, instead of the three previously required.

Zheng等[11]分析了H的计算，找到了合适的近似，使其可以加速计算，只计算两个平滑过的图像，而不是以前的三个。

Lowe [12] obtains scale invariance by convolving the image with a Difference of Gaussians (DoG) kernel at multiple scales, retaining locations which are optima in scale as well as space. DoG is used because it is good approximation for the Laplacian of a Gaussian (LoG) and much faster to compute. An approximation to DoG has been proposed which, provided that scales are $\sqrt 2$ apart, speeds up computation by a factor of about two, compared to the striaghtforward implementation of Gaussian convolution [13].

Lowe[12]通过将图像与DoG核在多个尺度上卷积，保留在尺度和空间上都是极值的点，来得到尺度不变性。使用DoG是因为，这是LoG的很好的近似，而且计算起来非常快。还有对DoG的近似计算加速方法。

It is noted in [14] that the LoG is a particularly stable scale-space kernel. [14]中指出，LoG是一个特别稳定的尺度空间核。

Scale-space techniques have also been combined with the Harris approach in [15] which computes Harris corners at multiple scales and retains only those which are also optima of the LoG response across scales.

尺度空间技术在[15]中还与Harris方法结合到一起，在多个尺度上计算Harris角点，只保留那些在多个尺度上LoG响应都是极值的角点。

Recently, scale invariance has been extended to consider features which are invariant to affine transformations [14, 16, 17].

最近，尺度不变性进行了拓展，考虑了对仿射变换不变的的特征。

An edge (usually a step change in intensity) in an image corresponds to the boundary between two regions. At corners of regions, this boundary changes direction rapidly. Several techniques were developed which involved detecting and chaining edges with a view to finding corners in the chained edge by analysing the chain code[18], finding maxima of curvature [19, 20, 21], change in direction [22] or change in appearance[23]. Others avoid chaining edges and instead look for maxima of curvature [24] or change in direction [25] at places where the gradient is large.

图像中的一个边缘（通常是灰度上的阶跃变化），对应着两个区域之间的边界。在区域的角点上，这个边界迅速改变方向。提出了几种技术，对边缘进行检测和链接起来，对链接起来的边缘，通过分析链码[18]，找到曲率最大点[19,20,21]，方向变化[22]或外观的变化[23]来找到角点。其他工作没有采用链接的边缘，而是在梯度很大的地方，寻找曲率最大点[24]，或方向变化点[25]。

Another class of corner detectors work by examining a small patch of an image to see if it “looks” like a corner. Since second derivatives are not computed, a noise reduction step (such as Gaussian smoothing) is not required. Consequently, these corner detectors are computationally efficient since only a small number of pixels are examined for each corner detected. A corollary of this is that they tend to perform poorly on images with only large-scale features such as blurred images. The corner detector presented in this work belongs to this category.

另一类角点检测器，是检查图像的小块，看是否像是一个角点。由于没有计算二阶导数，所以就不需要进行噪声去除（比如高斯平滑）。结果是，这些角点检测器计算上很快，因为对每个检测到的角点，只检查了一小部分像素。这个的一个推论，对于只有大尺度特征的图像，如模糊的图像，这类算法就表现较差。本文中的角点检测器就是属于这种类别。

The method presented in [26] assumes that a corner resembles a blurred wedge, and finds the characteristics of the wedge (the amplitude, angle and blur) by fitting it to the local image. The idea of the wedge is generalised in [27], where a method for calculating the corner strength is proposed which computes self similarity by looking at the proportion of pixels inside a disc whose intensity is within some threshold of the centre (nucleus) value. Pixels closer in value to the nucleus receive a higher weighting. This measure is known as the USAN (the Univalue Segment Assimilating Nucleus). A low value for the USAN indicates a corner since the centre pixel is very different from most of its surroundings. A set of rules is used to suppress qualitatively “bad” features, and then local minima of the, SUSANs, (Smallest USAN) are selected from the remaining candidates.

[26]中的方法假设，角点与模糊的楔子很像，通过将楔子与局部图像拟合，来找到楔子的特征（幅度，角度和模糊）。楔子的思想泛化到了[27]中，提出了一种计算角点强度的方法，计算的是自相似性，查找一个disc中，像素灰度在中心(nucleus)值的某个阈值之内的比例。像素灰度值与nucleus接近的，权重较高。这个度量就是USAN (the Univalue Segment Assimilating Nucleus)。USAN值比较小，说明是角点，因为中心像素与其周围非常不同。采用了一系列规则来抑制定性的坏特征，然后从剩余的候选中找到USAN的局部最小值，SUSAN。

Trajkovic and Hedley [28] use a similar idea: a patch is not self-similar if pixels generally look different from the centre of the patch. This is measured by considering a circle. $f_C$ is the pixel value at the centre of the circle, and $f_P$ and $f_{P'}$ are the pixel values at either end of a diameter line across the circle. The response function is defined as

Trajkovic和Hedley [28]使用了类似的思想：如果在一个图像块中，像素大部分与图形块中心不一样，那么图像块就不是自相似的。这个度量是采用一个圆来进行的。$f_C$是圆心处的像素值，$f_P$和$f_{P'}$是穿过圆的一个直径的两端的像素值。响应函数定义为：

$$C = min_P (f_P - f_C)^2 + (f_{P'} - f_C)^2$$(4)

This can only be large in the case where there corner. The test is performed on a Bresenham circle. Since the circle is discretized, linear or circular interpolation is used in between discrete orientations in order to give the detector a more isotropic response. To this end, the authors present a method whereby the minimum response function at all interpolated positions between two pixels can be efficiently computed. Computing the response function requires performing a search over all orientations, but any single measurement provides an upper bound on the response. To speed up matching, the response in the horizontal and vertical directions only is checked. If the upper bound on the response is too low, then the potential corner is rejected. To speed up the method further, this fast check is first applied at a coarse scale.

只有在角点处，这个值才会很大。这个测试是在一个Bresenham圆上进行的。由于圆是离散化的，在离散的方向之间使用了线性插值或圆形插值，以给检测器更加各向同性的响应。为此，作者提出了一种方法，以此在两个像素之间在所有插值的位置上的最小响应函数可以进行高效的计算。计算响应函数，需要在所有方向上进行搜索，但任何单个的度量，都给出了响应的上限。为加速匹配，只核实了垂直和水平方向的响应。如果响应的上限太低，那么就拒绝这个可能的角点。为进一步加速这个方法，在一个粗糙的尺度上首先进行快速核实。

A fast radial symmetry transform is developed in [29] to detect points. Points have a high score when the gradient is both radially symmetric, strong, and of a uniform sign along the radius. The scale can be varied by changing the size of the area which is examined for radial symmetry.

[29]提出了一种快速的径向对称变换，以检测点。当梯度径向对称，很强，而且在半径上符号一致，这个点的分数就很高。通过变化区域大小，检查径向对称性，尺度可以变化。

An alternative method of examining a small patch of an image to see if it looks like a corner is to use machine learning to classify patches of the image as corners or non-corners. The examples used in the training set determine the type of features detected. In [30], a three layer neural network is trained to recognise corners where edges meet at a multiple of 45◦, near to the centre of an 8 × 8 window. This is applied to images after edge detection and thinning. It is shown how the neural net learned a more general representation and was able to detect corners at a variety of angles.

检查一个小图像块，看其是否像一个角点的另一种方法，是使用机器学习方法将图像块进行归类为角点或非角点。训练集中所用的样本，确定了检测到的特征类型。在[30]中，训练了一个三层神经网络来识别边缘以45◦相交于8x8窗口的中央附近的角点。首先对图像进行边缘检测和细化，然后再应用这种方法。展示了神经网络是怎样学习到一个更通用的表示，可以检测很多角度的角点的。

## 2. High-Speed Corner Detection

### 2.1. FAST: Features from Accelerated Segment Test

The segment test criterion operates by considering a circle of sixteen pixels around the corner candidate p. The original detector [2, 3] classifies p as a corner if there exists a set of n contiguous pixels in the circle which are all brighter than the intensity of the candidate pixel $I_p$ plus a threshold t, or all darker than $I_p − t$, as illustrated in Figure 1. n was chosen to be twelve because it admits a high-speed test which can be used to exclude a very large number of non-corners: the test examines only the four pixels at 1, 5, 9 and 13 (the four compass directions). If p is a corner then at least three of these must all be brighter than $I_p + t$ or darker than $I_p − t$. If neither of these is the case, then p cannot be a corner. The full segment test criterion can then be applied to the remaining candidates by examining all pixels in the circle. This detector in itself exhibits high performance, but there are several weaknesses:

segment测试准则，通过考虑以角点候选p为中心的16个像素的圆，来进行运算。原始检测器[2,3]对p进行分类，如果在圆上有n个连续的像素，都比候选像素$I_p$加上一个阈值t要大，或都比$I_p − t$要小，如图1所示，就认为p是角点。n选择为12，因为可以进行高速测试，可以用于排除大量非角点：测试只检查四个像素，即1，5，9和13（四个指南针方向）。如果p是一个角点，那么至少3个要比$I_p + t$要亮，或比$I_p - t$要暗。如果都不是这种情况，那么p就不会是角点。完整的segment测试准则可以应用于剩下的候选，检查圆上的所有像素。检测器本身展现出了很高的性能，但有几个弱点：

1. The high-speed test does not generalise well for n < 12. 对于n<12的情况，高速测试的泛化并不好。

2. The choice and ordering of the fast test pixels contains implicit assumptions about the distribution of feature appearance. 快速测试像素的选择和排序，包含特征外观的分布的隐式假设。

3. Knowledge from the first 4 tests is discarded. 前4个测试的知识被抛弃了。

4. Multiple features are detected adjacent to one another. 检测到了多个相邻的特征。

### 2.2. Machine Learning a Corner Detector

Here we present an approach which uses machine learning to address the first three points (the fourth is addressed in Section 2.3). The process operates in two stages. In order to build a corner detector for a given n, first, corners are detected from a set of images (preferably from the target application domain) using the segment test criterion for n and a convenient threshold. This uses a slow algorithm which for each pixel simply tests all 16 locations on the circle around it.

这里我们提出了一种方法，使用的是机器学习来解决前3点（第4点在2.3节中进行处理）。这个过程以两个阶段进行。为对给定的n构建一个角点检测器，首先，对一系列图像使用segment测试准则来检测角点，使用n和一个方便的阈值。这使用的是慢速算法，对每个像素，测试围绕该像素的圆上的所有16个位置。

For each location on the circle x ∈ {1..16}, the pixel at that position relative to p (denoted by p → x) can have one of three states: 对该圆上的每个位置x ∈ {1..16}，在相对于p的该位置上的像素，表示为p → x，可以有三个状态中的一种：

$$S_{p→x} = \left\{ \begin{matrix} d, & I_{p→x}≤I_p-t & (darker) \\ s, & I_p-t<I_{p→x}<I_p+t & (similar) \\ b, & I_p+t≤I_{p→x} & (brighter) \end{matrix} \right.$$(5)

Choosing an x and computing $S_{p→x}$ for all p ∈ P (the set of all pixels in all training images) partitions P into three subsets, $P_d, P_s, P_b$, where each p is assigned to $P_{S_{p→x}}$. 对所有的p ∈ P（在所有训练图像中的所有像素的集合），选择一个x，计算$S_{p→x}$，可以将P分成三个子集，$P_d, P_s, P_b$，其中每个p都指定到$P_{S_{p→x}}$。

Let $K_p$ be a boolean variable which is true if p is a corner and false otherwise. Stage 2 employs the algorithm used in ID3[31] and begins by selecting the x which yields the most information about whether the candidate pixel is a corner, measured by the entropy of $K_p$.

令$K_p$是一个boolean变量，如果p是一个角点，就是true，如果不是角点，就是false。第2阶段采用在ID3[31]中使用的算法，选择一些x，会产生最多的关于候选像素是一个角点的信息，信息采用$K_p$的熵来度量。

The entropy of K for the set P is: 对于集合P的熵K是

$$H(P) = (c+\bar c)log_2 (c+\bar c)-clog_2 c-\bar clog_2\bar c$$(6)

where c = |{p|$K_p$ is true}| (number of coners)

and $\bar c$ = |{p|$K_p$ is false}| (number of non corners)

The choice of x then yields the information gain: x的选择产生了信息增益：

$$H(P) - H(P_d) - H(P_s) - H(P_b)$$(7)

Having selected the x which yields the most information, the process is applied recursively on all three subsets i.e. $x_b$ is selected to partition $P_b$ in to $P_{b,d}, P_{b,s}, P_{b,b}$, $x_s$ is selected to partition $P_s$ in to $P_{s,d}, P_{s,s}, P_{s,b}$ and so on, where each x is chosen to yield maximum information about the set it is applied to. The process terminates when the entropy of a subset is zero. This means that all p in this subset have the same value of $K_p$, i.e. they are either all corners or all non-corners. This is guaranteed to occur since K is an exact function of the learning data.

选择了产生最多信息的x，这个过程迭代的应用到所有三个子集上，即，选择$x_b$将$P_b$分割成$P_{b,d}, P_{b,s}, P_{b,b}$，选择$x_s$将$P_s$分割成$P_{s,d}, P_{s,s}, P_{s,b}$，等等，其中每个x的选择，都对所应用的集合会产生最大的信息。当这个子集的熵为零时，这个过程就停止了。这意味着，这个子集中的所有p都有相同的$K_p$值，即，它们都是角点，或者都不是角点。这肯定会发生的，因为K是学习数据的确切函数。

This creates a decision tree which can correctly classify all corners seen in the training set and therefore (to a close approximation) correctly embodies the rules of the chosen FAST corner detector. This decision tree is then converted into C-code, creating a long string of nested if-then-else statements which is compiled and used as a corner detector. For full optimisation, the code is compiled twice, once to obtain profiling data on the test images and a second time with arc-profiling enabled in order to allow reordering optimisations. In some cases, two of the three subtrees may be the same. In this case, the boolean test which separates them is removed.

这创建了一个决策树，可以正确的分类所有训练集中见过的角点，因此（很好的近似了）正确的体现了选择的FAST角点检测器的规则。这个决策树然后转化成C代码，创建了大量嵌套的if-then-else语句，经过编译用作角点检测器。为进行完全的优化，代码编译了两次，一次是为了在测试图像上得到概述数据，第二次采用arc-profiling，可以允许重新排序的优化。在一些情况下，三个子树的两个可能是一样的。在这种情况下，就将分隔它们的Boolean测试去掉。

Note that since the data contains incomplete coverage of all possible corners, the learned detector is not precisely the same as the segment test detector. It would be relatively straightforward to modify the decision tree to ensure that it has the same results as the segment test algorithm, however, all feature detectors are heuristic to some degree, and the learned detector is merely a very slightly different heuristic to the segment test detector.

注意，由于数据并不是包含了所有可能的角点的完整覆盖，学习得到的检测器与segment测试检测器并不完全一样。修改决策树以确保与segment测试算法与完全相同的结果，会比较直接，但是，所有的特征检测器是在一定程度上启发式的，学习得到的检测器与segment测试检测器只是一个略微不同的启发式。

### 2.3. Non-maximal Suppression

Since the segment test does not compute a corner response function, non maximal suppression can not be applied directly to the resulting features. Consequently, a score function, V must be computed for each detected corner, and non-maximal suppression applied to this to remove corners which have an adjacent corner with higher V . There are several intuitive definitions for V:

由于segment测试并没有计算角点响应函数，NMS并不能直接应用到得到的特征上。结果是，必须对每个检测到的角点计算一个打分函数V，对这个函数应用NMS，以去除临近V值更大的角点。对V，有几个直观的定义：

1. The maximum value of n for which p is still a corner. p仍然是一个角点的n的最大值；

2. The maximum value of t for which p is still a corner. p仍然是一个角点的t的最大值；

3. The sum of the absolute difference between the pixels in the contiguous arc and the centre pixel. 在中间像素和连续弧像素之间的差绝对值的和。

Definitions 1 and 2 are very highly quantised measures, and many pixels share the same value of these. For speed of computation, a slightly modified version of 3 is used. V is given by: 定义1和2是非常量化的度量，很多像素的这些值都是一样的。为计算速度，我们使用3的略微修正版。V由下式给出：

$$V = max (\sum_{x∈S_{bright}} |I_{p→x} - I_p| - t, \sum_{x∈S_{dark}} |I_p - I_{p→x}| - t)$$(8)

with

$$S_{bright} = \{ x|I_{p→x} ≥ I_p + t \}, S_{dark} = \{ x|I_{p→x} ≤ I_p - t \}$$(9)

### 2.4 Timing Results

Timing tests were performed on a 2.6GHz Opteron and an 850MHz Pentium III processor. The timing data is taken over 1500 monochrome fields from a PAL video source (with a resolution of 768×288 pixels). The learned FAST detectors for n = 9 and 12 have been compared to the original FAST detector, to our implementation of the Harris and DoG (difference of Gaussians—the detector used by SIFT) and to the reference implementation of SUSAN[32].

在计算机上进行了计时测试。计时数据是在PAL视频（分辨率为768×288）上的1500个黑白区域进行的。学习到的n=9和n=12时的FAST检测器，与原始FAST检测器进行了比较，与我们实现的Harris和DoG和SUSAN的参考实现也进行了比较。

As can be seen in Table 1, FAST in general offers considerably higher performance than the other tested feature detectors, and the learned FAST performs up to twice as fast as the handwritten version. Importantly, it is able to generate an efficient detector for n = 9, which (as will be shown in Section 3) is the most reliable of the FAST detectors. On modern hardware, FAST consumes only a fraction of the time available during video processing, and on low power hardware, it is the only one of the detectors tested which is capable of video rate processing at all.

如表1所示，FAST与其他测试的特征检测器相比，一般会给出更好的性能，学习得到的FAST比手写的版本快了接近2倍。重要的是，对n=9可以生成一个高效的检测器，这是最可靠的FAST检测器。在现代硬件上，FAST只占可用时间的一小部分，在低能耗硬件上，这是唯一一种可以进行全帧率处理的检测器。

Examining the decision tree shows that on average, 2.26 (for n = 9) and 2.39 (for n = 12) questions are asked per pixel to determine whether or not it is a feature. By contrast, the handwritten detector asks on average 2.8 questions.

检查决策树表明，平均每个像素，问了2.26 (for n = 9)和2.39 (for n = 12)个问题，确定是否是一个特征。对比起来，手写的检测器平均问了2.8个问题。

Interestingly, the difference in speed between the learned detector and the original FAST are considerably less marked on the Opteron processor compared to the Pentium III. We believe that this is in part due to the Opteron having a diminishing cost per pixel queried that is less well modelled by our system (which assumes equal cost for all pixel accesses), compared to the Pentium III.

## 3. A Comparison of Detector Repeatability

Although there is a vast body of work on corner detection, there is much less on the subject of comparing detectors. Mohannah and Mokhtarian[33] evaluate performance by warping test images in an affine manner by a known amount. They define the ‘consistency of corner numbers’ as

虽然在角点检测上有大量工作了，在比较检测器上的工作就少的多了。Mohannah和Mokhtarian[33]通过对测试图像以仿射的方式以已知的量进行变形，进行性能评估。

$$CCN = 100 × 1.1^{-|n_w - n_o|}$$

where $n_w$ is the number of features in the warped image and $n_o$ is the number of features in the original image. They also define accuracy as 其中$n_w$是在变形图像中的特征数量，$n_o$是原始图像中的特征数量。

$$ACU = 100 × (n_a/n_o + n_a/n_g) /2$$

where $n_g$ are the number of ‘ground truth’ corners (marked by humans) and $n_a$ is the number of matched corners compared to the ground truth. This unfortunately relies on subjectively made decisions. 其中$n_g$是真值角点的数量（人工标注的），$n_a$是与真值相比，匹配的角点的数量。不幸的是，这会依赖于主观做出的决定。

Trajkovic and Hedley[28] define stability to be the number of ‘strong’ matches (matches detected over three frames in their tracking algorithm) divided by the total number of corners. This measurement is clearly dependent on both the tracking and matching methods used, but has the advantage that it can be tested on the date used by the system.

Trajkovic和Hedley[28]将稳定性定义为，强匹配的数量（在追踪算法中在超过3帧中检测出来的匹配）除以角点的总数量。这个度量很明显依赖于使用的跟踪和匹配方法，但也有一个优势，即可以在系统所用的时间上进行测试。

When measuring reliability, what is important is if the same real-world features are detected from multiple views [1]. This is the definition which will be used here. For an image pair, a feature is ‘detected’ if is is extracted in one image and appears in the second. It is ‘repeated’ if it is also detected nearby in the second. The repeatability is the ratio of repeated features detected features. In [1], the test is performed on images of planar scenes so that the relationship between point positions is a homography. Fiducial markers are projected on to the planar scene to allow accurate computation of this.

当度量可靠性时，重要的是，是否从多个视角检测到了同样的真实世界特征。这是我们在这里所用的定义。对于一个图像对，如果从一幅图像中提取了一个特征，并在第二幅图像中出现，那么我们就说检测到了一个特征。如果还在第二幅图像中的附近也检测到了，那么就是重复的。可重复性是，重复检测到的特征的比率。在[1]中，对平面场景的图像进行了测试，这样在点位置之间的关系是具有单映性的。标记点投影到了平面场景中，以进行这个的精确计算。

By modelling the surface as planar and using flat textures, this technique tests the feature detectors’ ability to deal with mostly affine warps (since image features are small) under realistic conditions. This test is not so well matched to our intended application domain, so instead, we use a 3D surface model to compute where detected features should appear in other views (illustrated in Figure 2). This allows the repeatability of the detectors to be analysed on features caused by geometry such as corners of polyhedra, occlusions and T-junctions. We also allow bas-relief textures to be modelled with a flat plane so that the repeatability can be tested under non-affine warping.

将表面建模成平面的，使用平坦的纹理，这种技术测试特征检测器在实际情况中处理多数仿射变换的能力（由于图像特征是很小的）。这个测试与我们理想的应用领域并不是那么匹配，所以，我们使用了一个3D表面模型，来计算检测到的特征应当在其他视角中出现（如图2所示）。这使得检测器的重复性，可以通过分析由几何导致的特征，比如，多面体的角点，遮挡和T形结。我们还允许浅浮雕纹理用一个平面进行建模，这样重复性可以在非仿射形变的情况下进行测试。

A margin of error must be allowed because: 必须要允许一定的误差边界，因为：

1. The alignment is not perfect. 对齐不是完美的；
2. The model is not perfect. 模型不是完美的；
3. The camera model (especially regarding radial distortion) is not perfect. 相机模型（尤其是径向形变的）不是完美的；
4. The detector may find a maximum on a slightly different part of the corner. This becomes more likely as the change in viewpoint and hence change in shape of the corner become large. 检测器可能找到角点不同部分处的极大值。这在视角变化时更加可能，因此角点形状的变化会变得更大。

Instead of using fiducial markers, the 3D model is aligned to the scene by hand and this is then optimised using a blend of simulated annealing and gradient descent to minimise the SSD between all pairs of frames and reprojections.

我们没有使用标记点，3D模型与场景采取手工对齐，然后使用模拟退火和梯度下降的几何来最小化所有帧和reprojections对之间的SSD。

To compute the SSD between frame i and reprojected frame j, the position of all points in frame j are found in frame i. The images are then bandpass filtered. High frequencies are removed to reduce noise, while low frequencies are removed to reduce the impact of lighting changes. To improve the speed of the system, the SSD is only computed using 1000 random points (as opposed to every point).

为计算在帧i和reprojected帧j之间的SSD，在帧j中的所有点的位置都在帧i中找到。图像然后经过带通滤波。去掉高频分量，以降低噪声，而去掉低频以降低光照变化的影响。为改进系统的速度，SSD只使用1000个随机点来进行计算（而不是所有点）。

The datasets used are shown in Figure 3, Figure 4 and Figure 5. With these datasets, we have tried to capture a wide range of corner types (geometric and textural). 使用的数据如图3，4，5所示。用这些数据集，我们尝试捕获了大量类型的角点（几何的和纹理的）。

The repeatability is computed as the number of corners per frame is varied. For comparison we also include a scattering of random points as a baseline measure, since in the limit if every pixel is detected as a corner, then the repeatability is 100%.

在每帧的角点数量变化时，计算了重复性。为进行比较，我们还包含了随机散点作为基准度量，由于如果每个点都检测为角点，那么重复性就是100%了。

To test robustness to image noise, increasing amounts of Gaussian noise were added to the bas-relief dataset. It should be noted that the noise added is in addition to the significant amounts of camera noise already present (from thermal noise, electrical interference, and etc).

为测试对图像噪声的稳健性，对浅浮雕数据集加入了高斯噪声。应当指出，这些噪声的加入，是在已有的相机噪声的基础之上的。

## 4. Results and Discussion

Shi and Tomasi [7], derive their result for better feature detection on the assumption that the deformation of the features is affine. In the box and maze datasets, this assumption holds and can be seen in Figure 6B and Figure 6C the detector outperforms the Harris detector. In the bas-relief dataset, this assumption does not hold, and interestingly, the Harris detector outperforms Shi and Tomasi detector in this case.

Shi and Tomasi [7]假设特征的形变是仿射的，推导出了更好的特征检测结果。在box和maze数据集中，这个假设是正确的，如图6b和6c所示，检测器性能超过了Harris检测器。在浅浮雕数据集中，这个假设就不成立了，有趣的是，Harris检测器在这个情况中超过了Shi and Tomasi检测器。

Mikolajczyk and Schmid [15] evaluate the repeatability of the Harris-Laplace detector evaluated using the method in [34], where planar scenes are examined. The results show that Harris-Laplace points outperform both DoG points and Harris points in repeatability. For the box dataset, our results verify that this is correct for up to about 1000 points per frame (typical numbers, probably commonly used); the results are somewhat less convincing in the other datasets, where points undergo non-projective changes.

Mikolajczyk and Schmid [15] 评估了Harris-Laplace检测器的重复性，使用的是[34]中的方法，其中检查的平面场景。结果表明，Harris-Laplace点在重复性上超过了DoG点和Harris点的性能。对于box数据集，我们的结果验证了，在最多每帧1000个点的情况下，这都是正确的；这个结果在其他的点经过非投影变化的数据集上没有那么有说服力。

In the sample implementation of SIFT[35], approximately 1000 points are generated on the images from the test sets. We concur that this a good choice for the number of features since this appears to be roughly where the repeatability curve for DoG features starts to flatten off.

在SIFT的实现中[35]，从测试集中的图像中生成了大约1000个点。我们赞同这个特征的数量是非常好的选择，因为DoG特征的曲线的重复性开始变平了。

Smith and Brady[27] claim that the SUSAN corner detector performs well in the presence of noise since it does not compute image derivatives, and hence, does not amplify noise. We support this claim: although the noise results show that the performance drops quite rapidly with increasing noise to start with, it soon levels off and outperforms all but the DoG detector.

Smith and Brady[27]声称，SUSAN角点检测器在含有噪声的情况下表现也很好，因为没有计算图像导数，因此不会放大噪声。我们支持这个结论：虽然含噪结果表明，随着噪声越来越多，性能下降非常迅速，但很快就变平，然后超过了所有检测器（除了DoG）。

The big surprise of this experiment is that the FAST feature detectors, despite being designed only for speed, outperform the other feature detectors on these images (provided that more than about 200 corners are needed per frame). It can be seen in Figure 6A, that the 9 point detector provides optimal performance, hence only this and the original 12 point detector are considered in the remaining graphs.

The DoG detector is remarkably robust to the presence of noise. Since convolution is linear, the computation of DoG is equivalent to convolution with a DoG kernel. Since this kernel is symmetric, this is equivalent to matched filtering for objects with that shape. The robustness is achieved because matched filtering is optimal in the presence of additive Gaussian noise[36].

FAST, however, is not very robust to the presence of noise. This is to be expected: Since high speed is achieved by analysing the fewest pixels possible, the detector’s ability to average out noise is reduced.

## 5. Conclusions

In this paper, we have used machine learning to derive a very fast, high quality corner detector. It has the following advantages:

- It is many times faster than other existing corner detectors.
- High levels of repeatability under large aspect changes and for different kinds of feature.

However, it also suffers from a number of disadvantages:

- It is not robust to high levels noise.
- It can respond to 1 pixel wide lines at certain angles, when the quantisation of the circle misses the line.
- It is dependent on a threshold.

We were also able to verify a number of claims made in other papers using the method for evaluating the repeatability of corners and have shown the importance of using more than just planar scenes in this evaluation.

The corner detection code is made available from http://mi.eng.cam.ac.uk/∼{}er258/work/fast.html and http://savannah.nongnu.org/projects/libcvd and the data sets used for repeatability are available from http://mi.eng.cam.ac.uk/∼{}er258/work/datasets.html.