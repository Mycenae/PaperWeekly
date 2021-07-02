# ORB: an efficient alternative to SIFT or SURF

Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski

Willow Garage, Menlo Park, California

## 0. Abstract

Feature matching is at the base of many computer vi­sion problems, such as object recognition or structure from motion. Current methods rely on costly descriptors for de­tection and matching. In this paper, we propose a very fast binary descriptor based on BRIEF, called ORB, which is rotation invariant and resistant to noise. We demonstrate through experiments how ORB is at two orders of magni­tude faster than SIFT, while peiforming as well in many situations. The efficiency is tested on several real-world ap­plications, including object detection and patch-tracking on a smart phone.

特征匹配是很多计算机视觉问题的基础，比如目标识别或从运动计算结构。目前的方法依赖于昂贵的描述子进行检测和匹配。在本文中，我们提出了一种基于BRIEF的非常快速的二值描述子，称为ORB，是旋转不变的，具有抗噪声能力。我们通过试验证明了，ORB是怎样比SIFT快了两个数量级的，而且在很多情况下效果一样好。其效率在几个真实世界应用中进行了测试，包括在智能手机中的目标检测和图像块跟踪。

## 1. Introduction

The SIFT keypoint detector and descriptor [17], al­though over a decade old, have proven remarkably success­ful in a number of applications using visual features, in­cluding object recognition [17], image stitching [28], visual mapping [25], etc. However, it imposes a large computa­tional burden, especially for real-time systems such as vi­sual odometry, or for low-power devices such as cell phones. This has led to an intensive search for replacements with lower computation cost; arguably the best of these is SURF [2]. There has also been research aimed at speeding up the computation of SIFT, most notably with GPU devices [26].

SIFT关键点检测器和描述子[17]虽然已经超过10年了，但在很多使用视觉特征的应用中被证明非常成功，包括目标识别，图像拼接，视觉映射，等等。但是其计算代价很大，尤其是对于实时系统，比如视觉里程计，或对于低能耗设备，比如手机。很多研究者都在寻找更低计算量的替代品；最好的可能是SURF[2]。有一些研究的目标是加速SIFT的计算，最著名的是用GPU设备[26]。

In this paper, we propose a computationally-efficient re­placement to SIFT that has similar matching performance, is less affected by image noise, and is capable of being used for real-time performance. Our main motivation is to en­hance many common image-matching applications, e.g., to enable low-power devices without GPU acceleration to per­form panorama stitching and patch tracking, and to reduce the time for feature-based object detection on standard PCs. Our descriptor performs as well as SIFT on these tasks (and better than SURF), while being almost two orders of mag­nitude faster.

本文中，我们提出了一种计算上高效的SIFT替代，匹配性能类似，受到图像噪声的影响更小，可以用于实时处理。我们的主要动机是，增强很多常见的图像匹配应用，如，使没有GPU加速的低能耗设备进行全景拼接和图像块跟踪，降低在标准PC上基于特征的目标检测的计算时间。在这些任务上，我们的描述子与SIFT性能一样（比SURF要好），而速度快了两个数量级。

Our proposed feature builds on the well-known FAST keypoint detector [23] and the recently-developed BRIEF descriptor [6]; for this reason we call it ORB (Oriented FAST and Rotated BRIEF). Both these techniques are at­tractive because of their good performance and low cost. In this paper, we address several limitations of these tech­niques vis-a-vis SIFT, most notably the lack of rotational invariance in BRIEF. Our main contributions are:

我们提出的特征是基于著名的FAST关键点检测器[23]和最近提出的BRIEF描述子[6]；为此，我们称之为ORB（有向FAST和旋转BRIEF）。这两种技术都很吸引人，因为性能好，计算量低。本文中，我们用SIFT来处理这些技术的一些局限，主要是BRIFE的缺少旋转不变性。我们的主要贡献为：

- The addition of a fast and accurate orientation compo­nent to FAST. 对FAST加上了一个快速准确的方向部分；

- The efficient computation of oriented BRIEF features. 有向BRIEF特征的高效计算；

- Analysis of variance and correlation of oriented BRIEF features. 有向BRIEF特征的方差和相关的分析；

- A learning method for de-correlating BRIEF features under rotational invariance, leading to better perfor­mance in nearest-neighbor applications. 去相关BRIEF特征在旋转不变性下的学习方法，在最近邻应用中得到了更好的性能。

To validate ORB, we perform experiments that test the properties of ORB relative to SIFT and SURF, for both raw matching ability, and performance in image-matching applications. We also illustrate the efficiency of ORB by implementing a patch-tracking application on a smart phone. An additional benefit of ORB is that it is free from the licensing restrictions of SIFT and SURF.

为验证ORB的有效性，我们进行了试验，测试了ORB相对于SIFT和SURF的属性，包括原始的匹配能力，和在图像匹配应用中的性能。我们还在手机上实现了一个图像块跟踪的应用，展示了ORB的效率。ORB的额外好处是，没有SIFT和SURF的授权限制。

## 2. Related Work

**Keypoints**. FAST and its variants [23,24] are the method of choice for finding keypoints in real-time systems that match visual features, for example, Parallel Tracking and Mapping [13]. It is efficient and finds reasonable corner keypoints, although it must be augmented with pyramid schemes for scale [14], and in our case, a Harris corner filter [11] to reject edges and provide a reasonable score.

关键点。FAST及其变体是在实时系统中找到关键点进行视觉特征匹配的方法选项，比如，并行跟踪和建图[13]。在找到合理的角点关键点上非常高效，但是必须用尺度金字塔进行扩增[14]，在我们的情况中，是一个Harris角点滤波器[11]来拒绝边缘，给出合理的分数。

Many keypoint detectors include an orientation operator (SIFT and SURF are two prominent examples), but FAST does not. There are various ways to describe the orientation of a keypoint; many of these involve histograms of gradient computations, for example in SIFT [17] and the approxi­mation by block patterns in SURF [2]. These methods are either computationally demanding, or in the case of SURF, yield poor approximations. The reference paper by Rosin [22] gives an analysis of various ways of measuring orienta­tion of corners, and we borrow from his centroid technique. Unlike the orientation operator in SIFT, which can have multiple value on a single keypoint, the centroid operator gives a single dominant result.

很多关键点检测器包含方向算子（SIFT和SURF是两个有名的例子），但FAST没有。有很多方法来描述一个角点的方向；很多都采用了梯度直方图的计算，比如在SIFT[17]和SURF[2]中的块模式的近似。这些方法计算量很大，在SURF中得到的近似还不好。Rosin[22]的参考文章给出了各种测量角点方向的方法的分析，我们借鉴了他的centroid技术。SIFT中的方向算子在一个关键点处有多个值，centroid算子与之不同，会给出一个主要的结果。

**Descriptors**. BRIEF [6] is a recent feature descriptor that uses simple binary tests between pixels in a smoothed image patch. Its performance is similar to SIFT in many respects, including robustness to lighting, blur, and perspective dis­tortion. However, it is very sensitive to in-plane rotation.

BRIEF[6]是最近的一个特征描述子，在平滑的图像块中使用简单的像素间二值测试。其性能与SIFT在很多方面都很类似，包括对光照、模糊和视角扭曲的稳健性。但是，对平面内旋转非常敏感。

BRIEF grew out of research that uses binary tests to train a set of classification trees [4]. Once trained on a set of 500 or so typical keypoints, the trees can be used to re­turn a signature for any arbitrary keypoint [5]. In a similar manner, we look for the tests least sensitive to orientation. The classic method for finding uncorrelated tests is Princi­pal Component Analysis; for example, it has been shown that PCA for SIFT can help remove a large amount of re­dundant information [12]. However, the space of possible binary tests is too big to perform PCA and an exhaustive search is used instead.

BRIEF从使用二值测试来训练分类树的集合[4]的研究发展而来。在500左右典型的关键点上进行训练后，这个树可以用于返回对任意关键点的签名[5]。以类似的形式，我们寻找对方向没那么敏感的测试。寻找不相关测试的经典方法是PCA；比如，已经证明了SIFT使用PCA可以帮助去除大量冗余信息[12]。但是，可能的二值测试的空间对于PCA来说太大了，进而使用了一个穷举搜索。

Visual vocabulary methods [21,27] use offline clustering to find exemplars that are uncorrelated and can be used in matching. These techniques might also be useful in finding uncorrelated binary tests.

视觉词典方法[21,27]使用离线聚类来寻找不相关的样本，可以用于匹配。这种技术在寻找不相关二值测试中也可能是有用的。

The closest system to ORB is [3], which proposes a multi-scale Harris keypoint and oriented patch descriptor. This descriptor is used for image stitching, and shows good rotational and scale invariance. It is not as efficient to com­pute as our method, however.

与ORB最接近的系统是[3]，提出了一个多尺度Harris关键点和有向图像块描述子。这个描述子用于图像拼接，有很好的旋转和尺度不变性。但是，没有我们的方法计算高效。

## 3. oFAST: FAST Keypoint Orientation

FAST features are widely used because of their compu­tational properties. However, FAST features do not have an orientation component. In this section we add an efficiently­ computed orientation.

FAST特征因为其计算性质得到了广泛使用。但是，FAST特征并没有一个方向的组成部分。在本节中，我们加入了一个高效计算的方向。

### 3.1. FAST Detector

We start by detecting FAST points in the image. FAST takes one parameter, the intensity threshold between the center pixel and those in a circular ring about the center. We use FAST-9 (circular radius of 9), which has good per­formance.

我们先在图像中检测FAST关键点。FAST只有一个参数，中间像素和在围绕中间像素的圆环上的像素之间的灰度阈值。我们使用FAST-9（环形半径9），其性能非常好。

FAST does not produce a measure of cornerness, and we have found that it has large responses along edges. We em­ploy a Harris corner measure [11] to order the FAST key­ points. For a target number N of keypoints, we first set the threshold low enough to get more than N keypoints, then order them according to the Harris measure, and pick the top N points.

FAST并没有给出角点度的度量，我们发现在沿着边缘的地方响应很大。我们采用一个Harris角点度量[11]来对FAST关键点进行排序。对N个目标关键点，我们首先讲阈值设的很低，以得到多于N个关键点，然后根据Harris度量对其进行排序，然后选择最高的N个点。

FAST does not produce multi-scale features. We employ a scale pyramid of the image, and produce FAST features (filtered by Harris) at each level in the pyramid.

FAST并不给出多尺度特征。我们采用了图像中的一个尺度金字塔，并在金字塔的每一级产生FAST特征（由Harris进行滤波）。

### 3.2. Orientation by Intensity Centroid

Our approach uses a simple but effective measure of cor­ner orientation, the intensity centroid [22]. The intensity centroid assumes that a corner's intensity is offset from its center, and this vector may be used to impute an orientation. Rosin defines the moments of a patch as:

我们的方法使用一种简单但是有效的角点方向的度量，灰度质心[22]。灰度质心会假设，角点的灰度是从其中心偏移得到的，这个向量可以用于得到一个方向。Rosin定义一个图像块的动量为：

$$m_{pq} = \sum_{x,y} x^p y^q I(x,y)$$(1)

and with these moments we may find the centroid: 用这些动量，我们可以找到质心

$$C = (m_{10}/m_{00}, m_{01}/m_{00})$$(2)

We can construct a vector from the corner's center, O, to the centroid, $\vec{OC}$. The orientation of the patch then simply is:

我们可以从角点的中心O构建一个到质心$\vec{OC}$的向量。图像块的方向就是：

$$\theta = atan2(m_{01},m_{10})$$(3)

where atan2 is the quadrant-aware version of arctan. Rosin mentions taking into account whether the corner is dark or light; however, for our purposes we may ignore this as the angle measures are consistent regardless of the corner type.

atan2是感知象限版的atan。Rosin提到了，将角点是暗的还是亮的考虑进来；但是，为了我们的目的，我们忽略了这个，因为角度度量是连续的，不管角点类型是什么。

To improve the rotation invariance of this measure we make sure that moments are computed with x and y re­maining within a circular region of radius T. We empirically choose T to be the patch size, so that that x and y run from [-T, T] . As |C| approaches 0, the measure becomes unsta­ble; with FAST corners, we have found that this is rarely the case.

为改进这种度量的旋转不变性，我们确保动量的计算时，x和y是在半径为T的圆形区域中的。我们通过经验，将T选择为图像块的大小，这样x和y会在[-T, T]的范围内。当|C|接近0时，度量变得不稳定；用FAST角点，我们发现，这种情况非常少。

We compared the centroid method with two gradient­ based measures, BIN and MAX. In both cases, X and Y gradients are calculated on a smoothed image. MAX chooses the largest gradient in the keypoint patch; BIN forms a histogram of gradient directions at 10 degree inter­vals, and picks the maximum bin. BIN is similar to the SIFT algorithm, although it picks only a single orientation. The variance of the orientation in a simulated dataset (in-plane rotation plus added noise) is shown in Figure 2. Neither of the gradient measures performs very well, while the cen­troid gives a uniformly good orientation, even under large image noise.

我们将质心方法与两种基于梯度的度量BIN和MAX进行了比较。在两种情况下，X和Y梯度的计算都是在平滑图像中的。MAX在关键点图像块中选择最大的梯度；BIN以10度的间隔形成了梯度方向的直方图，并选择了最大的bin。BIN与SIFT算法类似，但是只选择了一个方向。在一个仿真的数据集（平面内旋转加上增加的噪声）中，方向的方差如图2所示。这两种梯度度量的表现都不是很好，而质心给出了很好的方向，即使增加了很大的噪声也是这样。

## 4. rBRIEF: Rotation-Aware BRIEF

In this section, we first introduce a steered BRIEF de­scriptor, show how to compute it efficiently and demon­strate why it actually performs poorly with rotation. We then introduce a learning step to find less correlated binary tests leading to the better descriptor rBRIEF, for which we offer comparisons to SIFT and SURF.

本节中，我们首先提出一种引导BRIEF描述子，展示了怎样进行高效的计算，以及为什么对旋转表现很差。然后我们提出了一种学习步骤，找到相关性没那么大的二值测试，可以得到更好的描述子rBRIEF，然后与SIFT和SURF进行了比较。

### 4.1. Efficient Rotation of the BRIEF Operator

**Brief overview of BRIEF**

The BRIEF descriptor [6] is a bit string description of an image patch constructed from a set of binary intensity tests. Consider a smoothed image patch, p. A binary test τ is defined by:

BRIEF描述子是图像块的一个二值串描述子，从二值灰度测试的集合构建得到。考虑一个平滑的图像块p，二值测试τ定义如下：

$$τ(p;x,y) := \left\{ \begin{matrix} 1 & :p(x)<p(y) \\ 0 & :p(x)≥p(y)  \end{matrix} \right.$$(4)

where p(x) is the intensity of p at a point x. The feature is defined as a vector of n binary tests: 其中p(x)是图像p在点x处的灰度。特征定义为n个二值测试的向量：

$$f_n(p) := \sum_{1≤i≤n} 2^{i-1} τ(p;x_i, y_i)$$(5)

Many different types of distributions of tests were consid­ered in [6]; here we use one of the best performers, a Gaus­sian distribution around the center of the patch. We also choose a vector length n = 256.

[6]中考虑了很多种类型的测试分布；这里我们使用其中表现最好的一种，围绕图像块中心的高斯分布。我们还选择了向量长度n=256。

It is important to smooth the image before performing the tests. In our implementation, smoothing is achieved us­ing an integral image, where each test point is a 5 x 5 sub­window of a 31 x 31 pixel patch. These were chosen from our own experiments and the results in [6].

在进行测试前对图像进行平滑这非常重要。在我们的实现中，平滑是使用积分图像得到的，其中每个测试点是31x31像素块的5x5子窗口。这可以从我们的试验中和[6]中的结果选择得到。

**Steered BRIEF**

We would like to allow BRIEF to be invariant to in-plane rotation. Matching performance of BRIEF falls off sharply for in-plane rotation of more than a few degrees (see Figure 7). Calonder [6] suggests computing a BRIEF descriptor for a set of rotations and perspective warps of each patch, but this solution is obviously expensive. A more efficient method is to steer BRIEF according to the orientation of keypoints. For any feature set of n binary tests at location (xi, yi), define the 2 x n matrix

我们希望让BRIEF对平面内旋转不变。对于平面内稍大一点的旋转（大于几度），BRIEF的匹配性能就急剧下降（见图7）。Calonder[6]建议，对每个图像的数个旋转和视角变形都计算一个BRIEF描述子，但这种解决方法很明显计算量很大。一种更高效的方法是，根据关键点的方向来调整BRIEF。对在位置(xi,yi)处的n个二值测试的任意特征集，定义2xn的矩阵

$$S = \left( \begin{matrix} x_1,...,x_n \\ y_1,...,y_n \end{matrix} \right)$$

Using the patch orientation θ and the corresponding rotation matrix Rθ, we construct a "steered" version Sθ of S: 使用图像块的方向θ和对应的旋转矩阵Rθ，我们构建S的一个修正版Sθ

$$S_θ = R_θ S$$

Now the steered BRIEF operator becomes 现在修正版的BRIEF算子变成

$$g_n(p,θ) := f_n(p)|(x_i,y_i) ∈ S_θ$$(6)

We discretize the angle to increments of 2π/30 (12 de­grees), and construct a lookup table of precomputed BRIEF patterns. As long as the keypoint orientation θ is consistent across views, the correct set of points Sθ will be used to compute its descriptor.

我们将角度离散成以2π/30 (12 de­grees)为单位的递增，构建了预计算BRIEF模式的查找表。只要关键点的方向θ在不同视角之间是连续的，就会使用正确的点集Sθ来计算其描述子。

### 4.2. Variance and Correlation

One of the pleasing properties of BRIEF is that each bit feature has a large variance and a mean near 0.5. Figure 3 shows the spread of means for a typical Gaussian BRIEF pattern of 256 bits over 100k sample keypoints. A mean of 0.5 gives the maximum sample variance 0.25 for a bit feature. On the other hand, once BRIEF is oriented along the keypoint direction to give steered BRIEF, the means are shifted to a more distributed pattern (again, Figure 3). One way to understand this is that the oriented corner keypoints present a more uniform appearance to binary tests.

BRIEF的一个很好的性质是，每个bit的特征变化很大，均值在0.5附近。图3展示了，对一个典型的256点高斯BRIEF模式在100k个样本关键点上的均值分布。均值0.5给出了一个bit特征的最大样本方差为0.25。另一方面，一旦BRIEF沿着关键点的方向得到修正的BRIEF，那么均值就偏向到更加有分布的模式（图3）。一种理解这个的方式是，有向角点关键点给出了二值测试的更均匀的外观。

High variance makes a feature more discriminative, since it responds differentially to inputs. Another desirable prop­erty is to have the tests uncorrelated, since then each test will contribute to the result. To analyze the correlation and variance of tests in the BRIEF vector, we looked at the re­sponse to 100k keypoints for BRIEF and steered BRIEF. The results are shown in Figure 4. Using PCA on the data, we plot the highest 40 eigenvalues (after which the two de­scriptors converge). Both BRIEF and steered BRIEF ex­hibit high initial eigenvalues, indicating correlation among the binary tests - essentially all the information is contained in the first 10 or 15 components. Steered BRIEF has signif­icantly lower variance, however, since the eigenvalues are lower, and thus is not as discriminative. Apparently BRIEF depends on random orientation of keypoints for good per­formance. Another view of the effect of steered BRIEF is shown in the distance distributions between inliers and out­liers (Figure 5). Notice that for steered BRIEF, the mean for outliers is pushed left, and there is more of an overlap with the inliers.

高方差使特征更加具有区分性，因为对输入的响应不同。另一个很好的性质是，测试要不相关，因为每个测试都会对结果有贡献。为分析BRIEF向量的测试的相关性和方差，我们查看了BRIEF和修正BRIEF的100k的关键点的响应。结果如图4所示。对数据使用PCA，我们画出了最高的40个特征值（在这之后，两个描述子就会收敛）。BRIEF和修正BRIEF表现出了很高的初始值，说明在二值测试中有相关性，基本上所有信息都包含在开始10或15个元素里。修正BRIEF的方差低很多，由于特征值更低，因此区分性没有那么高。很明显BRIEF依赖于关键点的随机方向得到很好的性能。修正BRIEF的效果的另一个观察，在内点和外点的距离分布中展示（如图5）。注意对于修正BRIEF，外点的均值向左推了，因此与内点的重叠更多了一些。

### 4.3. Learning Good Binary Features

To recover from the loss of variance in steered BRIEF, and to reduce correlation among the binary tests, we de­velop a learning method for choosing a good subset of bi­nary tests. One possible strategy is to use PCA or some other dimensionality-reduction method, and starting from a large set of binary tests, identify 256 new features that have high variance and are uncorrelated over a large training set. However, since the new features are composed from a larger number of binary tests, they would be less efficient to com­pute than steered BRIEF. Instead, we search among all pos­sible binary tests to find ones that both have high variance (and means close to 0.5), as well as being uncorrelated.

为恢复修正BRIEF中丢失的方差，降低二值测试之间的相关性，我们提出一种学习方法来选择一个好的二值测试的子集。一种可能的策略是使用PCA，或一些其他降维方法，从一个很大的二值测试集合开始，挑选出256个新的特征，在一个大的训练集上方差高，不相关。但是，由于新特征是由一个更大数量的二值测试组成的，它们计算上比修正的BRIEF效率更低。因此我们搜索所有可能的二值测试，以找到方差高的（均值接近0.5），以及不相关的测试。

The method is as follows. We first set up a training set of some 300k keypoints, drawn from images in the PASCAL 2006 set [8]. We also enumerate all possible binary tests drawn from a 31 x 31 pixel patch. Each test is a pair of 5 x 5 sub-windows of the patch. If we note the width of our patch as wp = 31 and the width of the test sub-window as wt = 5, then we have N = (wp - wt)^2 possible sub-windows. We would like to select pairs of two from these, so we have N(N-1)/2 binary tests. We eliminate tests that overlap, so we end up with M = 205590 possible tests. The algorithm is:

方法如下。我们首先设立一个训练集，大约包含300k个关键点，从PASCAL 2006集的图像中提取出来[8]。我们还枚举了所有可能的二值测试，从31x31的图像块中得到。每个测试是图像块的一对5x5的子窗口。如果我们将图像块的宽度表示为wp=31，将测试子窗口的宽度表示为wt=5，然后我们有N = (wp - wt)^2个可能的子窗口。我们会从这些对中选择两个，所以我们有N(N-1)/2个二值测试。我们去除了重叠的测试，所以我们最后有M = 205590种可能的测试。算法是：

1. Run each test against all training patches. 对所有训练图像块运行每个测试；
2. Order the tests by their distance from a mean of 0.5, forming the vector T. 以对均值0.5的距离对测试进行排序，形成向量T。
3. Greedy search: 贪婪搜索：

(a) Put the first test into the result vector R and re­move it from T. 将第一个测试放入得到的向量R中，从T中将其移除；

(b) Take the next test from T, and compare it against all tests in R. If its absolute correlation is greater than a threshold, discard it; else add it to R. 从T中拿出下一个测试，与R中的所有测试进行比较。如果其绝对相关大于某阈值，丢弃之；否则将其加入R。

(c) Repeat the previous step until there are 256 tests in R. If there are fewer than 256, raise the thresh­old and try again. 重复前一步骤，直到在R中有256个测试。如果少于256，将阈值提高，再次尝试。

This algorithm is a greedy search for a set of uncorrelated tests with means near 0.5. The result is called rBRIEF. rBRIEF has significant improvement in the variance and correlation over steered BRIEF (see Figure 4). The eigen­ values of PCA are higher, and they fall off much less quickly. It is interesting to see the high-variance binary tests produced by the algorithm (Figure 6). There is a very pro­nounced vertical trend in the unlearned tests (left image), which are highly correlated; the learned tests show better diversity and lower correlation.

这个算法是对均值接近0.5的不相关测试的集合的贪婪搜索。结果称为rBRIEF。rBRIEF在方差和相关性上比修正BRIEF有明显的提升（见图4）。PCA的特征值更高，下降的更快。看到由这个算法产生的高方差二值测试是很有趣的（图6）。在未学习的测试中，有很明显的竖直的倾向（左图），这相关性非常大；学习到的测试展现出了更好的多样性和更低的相关性。

### 4.4. Evaluation

We evaluate the combination of oFAST and rBRIEF, which we call ORB, using two datasets: images with syn­thetic in-plane rotation and added Gaussian noise, and a real-world dataset of textured planar images captured from different viewpoints. For each reference image, we compute the oFAST keypoints and rBRIEF features, targeting 500 keypoints per image. For each test image (synthetic rotation or real-world viewpoint change), we do the same, then per­form brute-force matching to find the best correspondence. The results are given in terms of the percentage of correct matches, against the angle of rotation.

我们评估了oFAST和rBRIEF的组合，我们称之为ORB，我们使用了两个数据集：合成平面内旋转和加入高斯噪声的图像，和一个真实世界数据集，从不同视角拍摄的纹理平面图像。对每个参考图像，我们计算oFAST关键点和rBRIEF特征，目标是每幅图像500个关键点。对每个测试图像（合成的旋转，或真实世界的视角变化），我们做相同的计算，然后进行暴力匹配，以找到最佳的对应。结果以正确匹配的百分比对旋转角度给出。

Figure 7 shows the results for the synthetic test set with added Gaussian noise of 10. Note that the standard BRIEF operator falls off dramatically after about 10 degrees. SIFT outperforms SURF, which shows quantization effects at 45-degree angles due to its Haar-wavelet composition. ORB has the best performance, with over 70% inliers.

图7给出了带有高斯噪声10的合成测试集的结果。注意标准BRIEF算子在10度之后性能就急剧下降。SIFT比SURF性能要好，这表明由于其Harr小波的组成在45度角有量化效果。ORB的性能最好，有超过70%的内点。

ORB is relatively immune to Gaussian image noise, un­like SIFT. If we plot the inlier performance vs. noise, SIFT exhibits a steady drop of 10% with each additional noise increment of 5. ORB also drops, but at a much lower rate (Figure 8).

与SIFT不一样，ORB对高斯图像噪声相对免疫。如果我们画出内点性能对噪声的曲线，SIFT在噪声每额外增加5的情况下，性能下降10%。ORB也会下降，但是速度会慢很多（图8）。

To test ORB on real-world images, we took two sets of images, one our own indoor set of highly-textured mag­azines on a table (Figure 9), the other an outdoor scene. The datasets have scale, viewpoint, and lighting changes. Running a simple inlier/outlier test on this set of images, we measure the performance of ORB relative to SIFT and SURF. The test is performed in the following manner:

为在真实世界图像上测试ORB，我们采集了两个图像集合，一个是我们自己的室内桌子上高度纹理的杂志集合（图9），另外一个是室外场景。数据集的变化包含尺度，视角和光照变化。在这个图像集合上运行一个简单的内点/外点测试，我们度量了ORB相对于SIFT、SURF的性能。测试进行的方式如下：

1. Pick a reference view V0. 选择参考视角V0；
2. For all Vi, find a homographic warp Hi0 that maps Vi → V0. 对于所有的Vi，找到一个homographic变形Hi0，形成映射Vi → V0；
3. Now, use the Hi0 as ground truth for descriptor matches from SIFT, SURF, and ORB. 使用Hi0作为描述子的真值，用SIFT、SURF和ORB进行匹配。

ORB outperforms SIFT and SURF on the outdoor dataset. It is about the same on the indoor set; [6] noted that blob­ detection keypoints like SIFT tend to be better on graffiti­ type images.

ORB在室外数据集上性能超过了SIFT和SURF。在室内数据集上性能接近，[6]指出blob检测的关键点，像SIFT，在类似涂鸦上的图像上性能会更好。

Magazines | inlier% | N Points
--- | --- | ---
ORB | 36.180 | 548.50
SURF | 38.305 | 513.55
SIFT | 34.010 | 584.15
Boat | inlier% | N Points
ORB | 45.8 | 789
SURF | 28.6 | 795
SIFT | 30.2 | 714

## 5. Scalable Matching of Binary Features

In this section we show that ORB outperforms SIFT/SURF in nearest-neighbor matching over large databases of images. A critical part of ORB is the recovery of variance, which makes NN search more efficient.

本节中，我们展示了，在大型图像数据集中，采用最近邻匹配，ORB性能超过了SIFT/SURF。ORB的一个关键部分，是恢复了方差，这使得NN搜索更加高效。

### 5.1. Locality Sensitive Hashing for rBrief

As rBRIEF is a binary pattern, we choose Locality Sen­sitive Hashing [10] as our nearest neighbor search. In LSH, points are stored in several hash tables and hashed in differ­ent buckets. Given a query descriptor, its matching buckets are retrieved and its elements compared using a brute force matching. The power of that technique lies in its ability to retrieve nearest neighbors with a high probability given enough hash tables.

由于rBRIEF是一个二值模式，我们选择LSH作为我们的最近邻搜索方案。在LSH中，点存储在几个hash表中，哈希在不同的bucket中。给定一个查询算子，获取其匹配的bucket，使用暴力匹配比较其元素。这种技术的能力在于，在给定足够的哈希表时，能够以很高的概率获取其最近的邻域。

For binary features, the hash function is simply a subset of the signature bits: the buckets in the hash tables contain descriptors with a common sub-signature. The distance is the Hamming distance.

对于二值特征，哈希函数就是其签名bits的一个子集：在哈希表中的buckets包含了常见子签名的描述子。距离就是Hamming距离。

We use multi-probe LSH [18] which improves on the traditional LSH by looking at neighboring buckets in which a query descriptor falls. While this could result in more matches to check, it actually allows for a lower number of tables (and thus less RAM usage) and a longer sub-signature and therefore smaller buckets.

我们使用多探测器LSH[18]，改进了传统的LSH，查找临近的查询描述子落入的buckets。这可以给出更多匹配来进行核实，但实际上使用了更少数量的表（因此更少的RAM使用量），和更长的子签名，因此buckets更少。

### 5.2. Correlation and Leveling

rBRIEF improves the speed of LSH by making the buckets of the hash tables more even: as the bits are less correlated, the hash function does a better job at partitioning the data. As shown in Figure 10, buckets are much smaller in average compared to steered BRIEF or normal BRIEF.

rBRIEF改进了LSH的速度，使哈希表的buckets更加均匀：随着bits更加不相关，哈希函数在分割数据时可以表现更好。如图10所示，与修正BRIEF或常规BRIEF相比，平均起来buckets更小。

### 5.3. Evaluation

We compare the performance of rBRIEF LSH with kd­-trees of SIFT features using FLANN [20]. We train the dif­ferent descriptors on the Pascal 2009 dataset and test them on sampled warped versions of those images using the same affine transforms as in [1].

我们比较了rBRIEF LSH，和使用FLANN的SIFT特征kd树[20]。我们在PASCAL 2009数据集上训练了不同的描述子，在这些图像的采样的形变版上，对其进行测试，使用的仿射变换与[1]中相同。

Our multi-probe LSH uses bitsets to speedup the pres­ence of keys in the hash maps. It also computes the Ham­ming distance between two descriptors using an SSE 4.2 optimized popcount.

我们的多探测器LSH使用bitsets来加速哈希图中keys的存在，还使用了SSE 4.2来计算两个描述子之间的Hamming距离。

Figure 11 establishes a correlation between the speed and the accuracy of kd-trees with SIFT (SURF is equiv­alent) and LSH with rBRIEF. A successful match of the test image occurs when more than 50 descriptors are found in the correct database image. We notice that LSH is faster than the kd-trees, most likely thanks to its simplicity and the speed of the distance computation. LSH also gives more flexibility with regard to accuracy, which can be interesting in bag-of-feature approaches [21, 27]. We can also notice that the steered BRIEF is much slower due to its uneven buckets.

图11展示了，使用kd树的SIFT，和LSH rBRIEF，速度和准确率的关系。当在正确的数据集图像中找到了多于50个描述子时，测试图像就算作成功匹配。我们注意到，LSH比kd树要快，很可能是由于其简洁性，以及距离计算的速度。LSH对于速度还给出了更多的灵活性，在bag-of-feature方法中是很有趣的[21,27]。我们还注意到，修正BRIEF会慢很多，这是由于其不均衡的buckets。

## 6. Applications

### 6.1. Benchmarks

One emphasis for ORB is the efficiency of detection and description on standard CPUs. Our canonical ORB detec­tor uses the oFAST detector and rBRIEF descriptor, each computed separately on five scales of the image, with a scal­ing factor of $\sqrt 2$. We used an area-based interpolation for efficient decimation.

对于ORB的一个强调是，在标准CPU上的检测和描述效率都很高。我们的标准ORB检测器使用oFAST检测器和rBRIEF描述子，每个都在图像的5个尺度上单独计算，缩放系数为$\sqrt 2$。我们使用了一个基于面积插值进行高效的抽取。

The ORB system breaks down into the following times per typical frame of size 640x480. The code was executed in a single thread running on an Intel i7 2.8 GHz processor: ORB系统对于典型的640x480大小的帧计算时间分解如下。代码以单线程运行在Intel i7 2.8GHz处理器上：

ORB | Pyramid | oFAST | rBRIEF
--- | --- | --- | ---
Time(ms) | 4.43 | 8.68 | 2.12

When computing ORB on a set of 2686 images at 5 scales, it was able to detect and compute over 2 x 10^6 fea­tures in 42 seconds. Comparing to SIFT and SURF on the same data, for the same number of features (roughly 1000), and the same number of scales, we get the following times:

当在2686幅图像的集合上，以5个尺度计算ORB时，能够在42秒内检测并计算2 x 10^6个特征。在相同的数据上，与SIFT和SURF相比，对于相同数量的特征（大约1000），相同数量的尺度，我们得到下面的时间：

Detector | ORB | SURF | SIFT
--- | --- | --- | ---
Time per frame(ms) | 15.3 | 217.3 | 5228.7

These times were averaged over 24 640x480 images from the Pascal dataset [9]. ORB is an order of magnitude faster than SURF, and over two orders faster than SIFT. 这些时间是在PASCAL数据集上24幅640x480图像进行了平均。ORB比SURF快了一个数量级，比SIFT快了两个数量级。

### 6.2. Textured object detection

We apply rBRIEF to object recognition by implement­ing a conventional object recognition pipeline similar to [19]: we first detect oFAST features and rBRIEF de­scriptors, match them to our database, and then perform PROSAC [7] and EPnP [16] to have a pose estimate.

我们讲rBRIEF应用于目标识别，实现了传统的目标识别流程，与[19]类似：我们首先检测oFAST特征和rBRIEF描述子，并与数据集中的数据进行匹配，然后进行PROSAC和EPnP进行姿态估计。

Our database contains 49 household objects, each taken under 24 views with a 2D camera and a Kinect device from Microsoft. The testing data consists of 2D images of sub­sets of those same objects under different view points and occlusions. To have a match, we require that descriptors are matched but also that a pose can be computed. In the end, our pipeline retrieves 61% of the objects as shown in Figure 12.

我们的数据集包含49个日常目标，每个用2D相机和微软的Kinect设备以24个视角进行拍摄。测试数据包含这些目标在不同视角和遮挡情况下的2D图像子集。为得到匹配，我们需要描述子得到匹配，而且可以计算一个姿态。最后，我们的流程获取了61%目标，如图12所示。

The algorithm handles a database of 1.2M descriptors in 200MB and has timings comparable to what we showed earlier (14 ms for detection and 17ms for LSH matching in average). The pipeline could be speeded up considerably by not matching all the query descriptors to the training data but our goal was only to show the feasibility of object detection with ORB.

算法处理1.2M个描述子的数据库，处理时间与我们之前展示的类似（平均14ms检测，17ms进行LSH匹配）。流程可以显著加速，只要不将所有的查询描述子与训练数据进行匹配，但我们的目标只是要展示使用ORB进行目标检测的可行性。

### 6.3. Embedded real-time feature tracking

Tracking on the phone involves matching the live frames to a previously captured keyframe. Descriptors are stored with the keyframe, which is assumed to contain a planar surface that is well textured. We run ORB on each incom­ing frame, and proceed with a brute force descriptor match­ing against the keyframe. The putative matches from the descriptor distance are used in a PROSAC best fit homog­raphy H.

在手机上进行跟踪，要将之前捕获的关键帧与实时的帧进行匹配。描述子用关键帧的进行存储，假设包含一个平面表面，纹理丰富。我们在每个输入的帧上运行ORB，并与关键帧进行暴力描述子匹配。与描述子的距离的公认匹配，在PROSAC最佳匹配homography H中进行使用。

While there are real-time feature trackers that can run on a cellphone [15], they usually operate on very small images (e.g., 120x160) and with very few features. Systems com­parable to ours [30] typically take over 1 second per image. We were able to run ORB with 640 x 480 resolution at 7 Hz on a cellphone with a 1GHz ARM chip and 512 MB of RAM. The OpenCV port for Android was used for the im­plementation. These are benchmarks for about 400 points per image:

有一些实时的特征追踪器可以在手机上运行[15]，但通常是在很小的图像山（如，120x160），使用的特征很少。与我们的可以相比的系统[30]一般要用一秒才能处理一幅图像。我们用ORB处理640 x 480大小的图像，在1GHz ARM芯片和512MB内存的手机上，可以达到7Hz。Android上的OpenCV用在了实现中。这是每个图像上400个点的基准测试：

 | | ORB | Matching | H Fit
 --- | --- | --- | ---
 Time(ms) | 66.6 | 72.8 | 20.9

## 7. Conclusion

In this paper, we have defined a new oriented descrip­tor, ORB, and demonstrated its performance and efficiency relative to other popular features. The investigation of vari­ance under orientation was critical in constructing ORB and de-correlating its components, in order to get good per­formance in nearest-neighbor applications. We have also contributed a BSD licensed implementation of ORB to the community, via OpenCV 2.3.

本文中，我们定义了一个新的有向描述子，ORB，与其他流行的特征相比，我们证明了其性能和效率。在方向下的变化的研究，是构建ORB和部件解相关的关键，可以在最近邻应用中得到很好的性能。我们还贡献了BSD授权的ORB实现，在OpenCV 2.3中。

One of the issues that we have not adequately addressed here is scale invariance. Although we use a pyramid scheme for scale, we have not explored per keypoint scale from depth cues, tuning the number of octaves, etc.. Future work also includes GPU/SSE optimization, which could improve LSH by another order of magnitude.

我们还没有处理的很好的一个问题是，尺度不变性。虽然我们使用了金字塔的方案，我们还没有从深度的线索上研究每个关键点尺度，调节octaves的数量，等。未来的工作还包括，GPU/SSE的优化，这可能将LSH改进一个数量级。