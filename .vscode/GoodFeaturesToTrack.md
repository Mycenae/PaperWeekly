# Good Features to Track

Jianbo Shi, Carlo Tomasi Cornell University & Stanford University

## 0. Abstract

No feature-based vision system can work unless good features can be identified and tracked from frame to frame. Although tracking itself is by and large a solved problem, selecting features that can be tracked well and correspond to physical points in the world is still hard. We propose a feature selection criterion that is optimal by construction because it is based on how the tracker works, and a feature monitoring method that can detect occlusions, disocclusions, and features that do not correspond to points in the world. These methods are based on a new tracking algorithm that extends previous Newton-Raphson style search methods to work under affine image transformations. We test performance with several simulations and experiments.

只有可以逐帧识别出并跟踪好的特征，基于特征的视觉系统才可以很好的工作。虽然跟踪本身大致是一个已经解决的问题了，选取可以很好的跟踪的特赠，并对应到真实世界，仍然非常困难。我们提出了一个特征选择规则，是最优构建得到的，因为是基于追踪器的工作原理，可以检测遮挡、空洞和特征的特征监视方法，这个特征并不对应于真实世界的点。这些方法是基于一种新的追踪方法，拓展了之前的Newton-Raphson式的搜索方法，在仿射图像变换下也可以很好的工作。我们用几种仿真和试验进行了性能测试。

## 1. Introduction

Is feature tracking a solved problem? The extensive studies of image correlation [4], [3], [15], [18], [7], [17] and sum-of-squared-difference (SSD) methods [2], [1] show that all the basics are in place. With small inter-frame displacements, a window can be tracked by optimizing some matching criterion with respect to translation [10], [1] and linear image deformation [6], [8], [11], possibly with adaptive window size[l4]. Feature windows can be selected based on some measure of texturedness or cornerness, such as a high standard deviation in the spatial intensity profile [13], the presence of zero crossings of the Laplacian of the image intensity [12], and corners [9], [5]. Yet, even a region rich in texture can be poor. For instance, it can straddle a depth discontinuity or the boundary of a reflection highlight on a glossy surface. In either case, the window is not attached to a fixed point in the world, making that feature useless or even harmful to most structure-from-motion algorithms. Furthermore, even good features can become occluded, and trackers often blissfully drift away from their original target when this occurs. No feature-based vision system can be claimed to really work until these issues have been settled.

特征跟踪是一个解决的问题了吗？图像相关和差值平方和(SSD)方法的大量研究表明，基础的研究已经有很多了。在帧间有很小的偏移的时候，通过优化一些对平移和线性图像变换的配准规则，可以对窗口进行跟踪，有时候窗口大小可以是自适应的。特征窗口可以基于纹理程度或角点程度的度量来选择，比如空间灰度分布的标准差很高，存在图像灰度的Laplacian的过零点，和角点。但是，即使是纹理丰富的区域也可能是很差的。比如，可以横跨一个深度的不连续性，或光滑平面的反射的边缘。在任一情况中，窗口并没有与真实世界中的固定点关联，使得特征无用，甚至对多数sfm算法效果有害。而且，好的特征可能被遮挡，当这种情况发生时，跟踪器通常都会产生漂移。除非这些问题得到解决，基于特征的视觉系统才能说真的好用了。

In this paper we show how to monitor the quality of image features during tracking by using a measure of feature dissimilarity that quantifies the change of appearance of a feature between the first and the current frame. The idea is straightforward: dissimilarity is the feature’s rms residue between the first and the current frame, and when dissimilarity grows too large the feature should be abandoned. However, in this paper we make two main contributions to this problem. First, we provide experimental evidence that pure translation is not an adequate model for image motion when measuring dissimilarity, but affine image changes, that is, linear warping and translation, are adequate. Second, we propose a numerically sound and efficient way of determining affine changes by a Newton-Raphson stile minimization procedure, in the style of what Lucas and Kanade [10] do for the pure translation model. In addition, we propose a more principled way to select features than the more traditional "interest" or "cornerness" measures. Specifically, we show that features with good texture properties can be defined by optimizing the tracker’s accuracy. In other words, the right features are exactly those that make the tracker work best. Finally, we submit that using two models of image motion is better than using one. In fact, translation gives more reliable results than affine changes when the inter-frame camera translation is small, but affine changes are necessary to compare distant frames to determine dissimilarity. We define these two models in the next section.

本文中，我们展示了怎样在跟踪时，使用特征不相似度的度量（量化了一个特征在第一帧和当前帧之间的外观变化），监控图像特征的质量。其思想是很直接的：不相似性是第一帧和当前帧之间的特征的残差均方根，当不相似性增长到过大，这个特征就应当被抛弃了。但是，本文中，我们对这个问题作出了两个贡献。第一，我们给出了试验证据，当度量不相似性的时候，纯粹的平移并不是图像运动的充分模型，但仿射变换是，即线性变形和平移。第二，我们提出了一个数值上很高效的方法来计算仿射变换，通过Newton-Raphson式的最小化过程，与Lucas和Kanade在纯平移的模型做的类似。另外，与传统的兴趣或角点度度量相比，我们提出了一个更有原则的选择特征的方法。具体的，我们证明了，具有很好纹理特性的性质，可以通过优化追踪器的准确率来进行定义。换句话说，正确的特征就是那些使追踪器工作最好的特征。最后，我们提出，使用两个图像运动的模型，比使用一个要好。实际上，当帧间相机平移很小时，平移比仿射变换给出更可靠的结果，但仿射变换在比较帧间距离较大时确定不相似性是必须的。我们在下一节来定义这两个模型。

## 2. Two Models of Image Motion

As the camera moves, the patterns of image intensities change in a complex way. However, away from occluding boundaries and near surface markings, these changes can often be described as image motion:

当相机移动时，图像灰度变化的模式以复杂的方式变化。但是，在远离遮挡的边缘和接近表面的标记的地方，这些变化通常可以描述为图像运动：

$$I(x,y,t+τ) = I(x-ξ(x,y,t,τ), y-η(x,y,t,τ))$$(1)

Thus, a later image taken at time t+τ can be obtained by moving every point in the current image, taken at time t, by a suitable amount. The amount of motion δ  = (ξ, η) is called the displacement of the point at x = (x,y).

因此，在时间t+τ时拍摄的图像，可以通过移动在时间t拍摄的目前图像中的每个点得到，移动距离要合适。移动距离δ  = (ξ, η)称为x = (x,y)处点的位移。

The displacement vector δ is a function of the image position x, and variations in δ are often noticeable even within the small windows used for tracking. It then makes little sense to speak of "the" displacement of a feature window, since there are different displacements within the same window. An affine motion field is a better representation:

位移矢量δ是图像位置x的函数，即使是对于用于跟踪的小窗口，δ中的变化也是可以注意到的。因此要说一个特征窗口的偏移，就是没有意义的，因为即使在相同的窗口中，其位移也是不一样的。一个仿射运动场是一个更好的表示

$$δ = Dx + d$$

where

$$D = [d_{xx}, d_{xy}; d_{yx}, d_{yy}]$$

is a deformation matrix, and d is the translation of the feature window’s center. The image coordinates x are measured with respect to the window’s center. Then, a point x in the first image I moves to point Ax + d in the second image J , where A = 1 + D and 1 is the 2 x 2 identity matrix:

这是一个变形矩阵，其中d是特征窗口中心的平移。图像坐标系x是相对于窗口中心度量的。那么，在第一幅图像I中的一个点x，就移到了第二幅图像J中的点Ax+d，其中A=1+D，1是2x2的单位矩阵

$$J(Ax+d) = I(x)$$(2)

Given two images I and J and a window in image I, tracking means determining the six parameters that appear in the deformation matrix D and displacement vector d. The quality of this estimate depends on the size of the feature window, the texturedness of the image within it, and the amount of camera motion between frames. When the window is small, the matrix D is harder to estimate, because the variations of motion within it are smaller and therefore less reliable. However, smaller windows are in general preferable for tracking because they are less likely to straddle a depth discontinuity. For this reason, a pure translation model is preferable during tracking, where the deformation matrix D is assumed to be zero:

给定两幅图像I和J，和图像I中的窗口，跟踪的意思是确定在变形矩阵D和位移向量d中的6个参数。这个估计的质量，依赖于特征窗口的大小，其中图像的纹理度，和在两帧间相机运动的量。当窗口很小时，矩阵D更难以估计，因为其中运动的变化更小，因为更加不可靠。但是，对于跟踪来说，更小的窗口通常更受欢迎，因为不太可能横跨一个深度不连续性。因此，在跟踪时更倾向于一个纯粹的平移模型，其中变形矩阵D假设为0：

$$δ = d$$

The experiments in sections 6 and 7 show that the best combination of these two motion models is pure translation for tracking, because of its higher reliability and accuracy over the small inter-frame motion of the camera, and affine motion for comparing features between the first and the current frame in order to monitor their quality. In order to address these issues quantitatively, however, we first need to introduce our tracking method.

第6和7节中的试验表明，这两个运动模型的最佳组合，对于跟踪来说是纯平移，因为对相机运动的小帧间运动来说更加可靠准确率更高，对于比较第一帧和当前帧的特征来说，是仿射运动，这样可以监控其质量。但是，为定量的处理这些问题，我们首先需要介绍我们的追踪方法。

## 3. Computing Image Motion

Because of image noise and because the affine motion model is not perfect, equation (2) is in general not satisfied exactly. The problem of determining the motion parameters is then that of finding the A and d that minimize the dissimilarity

由于图像噪声和仿射运动模型的不完美性，式(2)的条件并不是严格满足的。确定运动参数的问题，就成了找到A和d，最小化下面的不相似度

$$ϵ = \int \int_W [J(Ax+d) - I(x)]^2 w(x) dx$$(3)

where W is the given feature window and w(x) is a weighting function. In the simplest case, w(x) = 1. Alternatively, w could be a Gaussian-like function to emphasize the central area of the window. Under pure translation, the matrix A is constrained to be equal to the identity matrix. To minimize the residual (3), we differentiate it with respect to the unknown entries of the deformation matrix D and the displacement vector d and set the result to zero. We then linearize the resulting system by the truncated Taylor expansion

其中W是给定的特征窗口，w(x)是一个加权函数。在最简单的情况下，w(x)=1。另外，w可以是一个类高斯的函数，来强调窗口的中央。在纯平移的情况下，矩阵A就等于单位矩阵。为最小化残差(3)，我们将其对形变矩阵D和位移向量d求微分，设结果为0。然后我们将结果的系统进行线性化，对Taylor表示进行截断

$$J(Ax+d) = J(x)+g^T (u)$$(4)

This yields (see [16]) the following linear 6 x 6 system: 这得到了下面的6 x 6线性系统：

$$Tz = a$$(5)

where $z^T = [d_{xx}, d_{yx}, d_{xy}, d_{yy}, d_x, d_y]$ collects the entries of the deformation D and displacement d, the error vector

其中z是形变D和位移d的入口，误差向量a为

$$a = \int \int_W [I(x) - J(x)] [xg_x, xg_y, yg_x, yg_y, g_x, g_y]wdx$$

depends on the difference between the two images, and the 6 x 6 matrix T, which can be computed from one image, can be written as

a依赖于两幅图像的差，6x6的矩阵T，可以从一幅图像中计算得到，可以写为

$$T = \int \int_W [U, V; V^T, Z] wdx$$(6)

where

$$U = [x^2 g_x^2, x^2g_xg_y, xyg_x^2, xyg_xg_y; x^2g_xg_y, x^2g_y^2, xyg_xg_y, xyg_y^2; xyg^2_x, xyg_xg_y, y^2g_x^2, y^2g_xg_y; xyg_xg_y, xyg_y^2, y^2g_xg_y, y^2g_y^2]$$

$$V^T = [xg^2_x, xg_xg_y, yg_x^2, yg_xg_y; xg_xg_y, xg_y^2, yg_xg_y, yg_y^2]$$

$$Z = [g_x^2, g_xg_y; g_xg_y, g_y^2]$$

Even when affine motion is a good model, equation 5 is only approximately satisfied, because of the linearization of equation (4). However, the correct affine change can be found by using equation 5 iteratively in a Newton-Raphson style minimization [16].

即使仿射运动是一个很好的模型，式5也只是近似的满足，因为(4)的线性化。但是，对式5以迭代的Newton-Raphson式的最小化进行使用，就可以得到正确的仿射变换。

During tracking, the affine deformation D of the feature window is likely to be small, since motion between adjacent frames must be small in the first place for tracking to work at all. It is then safer to set D to the zero matrix. In fact, attempting to determine deformation parameters in this situation is not only useless but can lead to poor displacement solutions: in fact, the deformation D and the displacement d interact through the 4 x 2 matrix V of equation (6), and any error in D would cause errors in d. Consequently, when the goal is to determine d, the smaller system

在跟踪中，特征窗口的仿射变换D很可能很小，因为临近帧之间的运动必须很小，这样追踪才能得以进行。因此设D为零矩阵是更安全的。实际上，试图在这种情况下确定形变参数，不仅是无用的，而且会得到很差的位移解：实际上，形变D和位移d通过式6的4x2矩阵V进行相互作用，在D中的任意误差都会得到d中误差。结果是，当目标是确定d时，应当求解下面的更小的系统

$$Zd = e$$(7)

should be solved, where e collects the last two entries of the vector a of equation (5). 其中e是式5中向量的最后两个入口。

When monitoring features for dissimilarities in their appearance between the first and the current frame, on the other hand, the full affine motion system (5) should be solved. In fact, motion is now too large to be described well by the pure translation model. Furthermore, in determining dissimilarity, the whole transformation between the two windows is of interest, and a precise displacement is less critical, so it is acceptable for D and d to interact to some extent through the matrix V.

当监控特征以求得第一帧和当前帧外观的不相似性时，另一方面，应当求解完整的仿射运动系统(5)。实际上，现在运动幅度太大，不能用纯平移模型来描述的很好。而且，在确定不相似性时，在两个窗口之间的整个变换是我们关心的，精确的位移则不那么关键，所以D和d通过矩阵V进行一定程度的相互作用，是可以接受的。

In the next two sections we discuss these issues in more detail: first we determine when system (7) yields a good displacement measurement (section 4) and then we see when equation (5) can be used reliably to monitor a feature’s quality (section 5).

下面两节中，我们更加详细的讨论这些问题：首先在第4部分中我们确定什么时候系统(7)会得到一个很好的位移度量，然后在第5部分中我们看看式(5)什么时候可以可靠的用于监控一个特征的质量。

## 4. Texturedness

Regardless of the method used for tracking, not all parts of an image contain complete motion information (the aperture problem): for instance, only the vertical component of motion can be determined for a horizontal intensity edge. To overcome this difficulty, researchers have proposed to track corners, or windows with a high spatial frequency content, or regions where some mix of second-order derivatives is sufficiently high. However, there are two problems with these "interest operators". First, they are often based on a preconceived and arbitrary idea of what a good window looks like. The resulting features may be intuitive, but are not guaranteed to be the best for the tracking algorithm to produce good results. Second, "interest operators" have been usually defined for the pure translation model of section 2, and the underlying concept are hard to extend to affine motion.

不管用什么方法进行跟踪，一幅图像中并不是所有部分都包含完整的运动信息（孔径问题）：比如，运动的竖直分量，可以定为水平灰度边缘。为克服这个问题，研究者提出要跟踪角点，或跟踪有着很高空间频率内容的窗口，或一些混合二阶导数最够大的区域。但是，对这些感兴趣的算子，有两个问题。第一，它们通常是基于对好窗口是什么样子的预想和任意思想。得到的特征可能是很直观的，但可能并不是跟踪算法可以得到最好结果的。第二，感兴趣的算子通常是为第2节中的纯平移模型来定义的，其潜在的概念很难拓展到仿射运动。

In this paper, we propose a more principled definition of feature quality. With the proposed definition, a good feature is one that can be tracked well, so that the selection criterion is optimal by construction.

本文中，我们提出一个更有原则的特征质量定义。用提出的定义，一个好的特征，是可以很好的跟踪的，所以选择原则从构建上就是最优的。

We can track a window from frame to frame if system 7 represents good measurements, and if it can be solved reliably. Consequently, the symmetric 2 x 2 matrix Z of the system must be both above the image noise level and well-conditioned. The noise requirement implies that both eigenvalues of Z must be large, while the conditioning requirement means that they cannot differ by several orders of magnitude. Two small eigenvalues mean a roughly constant intensity profile within a window. A large and a small eigenvalue correspond to a unidirectional texture pattern. Two large eigenvalues can represent corners, salt-and-pepper textures, or any other pattern that can be tracked reliably.

如果系统7表示了很好的度量，如果可以可靠的求解，那么我们可以从帧到帧的跟踪窗口。结果是，系统的对称的2x2矩阵Z必须高于图像噪声水平，而且要是良态的。噪声需求说明，Z的两个特征值都需要很大，而条件的需求意味着，它们不能差异达到几个数量级。两个小的特征值意味着，在窗口中的灰度变化大致是常数。特征值一大一小，对应着单方向的纹理模式。两个大的特征值可以表示角点，椒盐纹理，或任意其他可以很可靠的跟踪的模式。

In practice, when the smaller eigenvalue is sufficiently large to meet the noise criterion, the matrix Z is usually also well conditioned. In fact, the intensity variations in a window are bounded by the maximum allowable pixel value, so that the greater eigenvalue cannot be arbitrarily large. In conclusion, if the two eigenvalues of Z are $λ_1$ and $λ_2$, we accept a window if

在实践中，当较小的特征值足够大，符合噪声原则时，矩阵Z通常就是良态的。实际上，一个窗口中的灰度变化，是受到最大允许像素值限制的，这样大一些的特征值也不能任意的大。结论是，如果Z的两个特征值是$λ_1$和$λ_2$，如果有下式条件，我们就接受这个窗口

$$min(λ_1, λ_2) > λ$$(8)

where λ is a predefined threshold. 其中λ是一个预定义的阈值。

Similar considerations hold also when solving the full affine motion system (5) for the deformation D and displacement d. However, an essential difference must be pointed out: deformations are used to determine whether the window in the first frame matches that in the current frame well enough during feature monitoring. Thus, the goal is not to determine deformation per se. Consequently, it does not matter if one component of deformation cannot be determined reliably. In fact, this means that that component does not affect the window substantially, and any value along this component will do in the comparison. In practice, the system (5) can be solved by computing the pseudo-inverse of T. Then, whenever some component is undetermined, the minimum norm solution is computed, that is, the solution with a zero deformation along the undetermined component(s).

当求解完整仿射运动系统(5)得到形变D和位移d时，也有类似的考虑。但是，必须指出一个根本的差异：形变必须用于确定，第一帧中的窗口与当前帧的窗口在特征监控中是否很好的匹配。因此，目标并不是用来确定形变本身。结果是，如果形变的一个部件不能被可靠的确定，这是没关系的。实际上，这意味着这个部件并不会从根本上影响这个窗口，这个部件的任意值都可以在比较中起作用。实际上，系统(5)可以通过计算T的伪逆来求解。那么，不管何时一些部件是未定的，就求解最小值范数解，即，在未定部件时的零形变的解。

## 5. Dissimilarity

A feature with a high texture content, as defined in the previous section, can still be a bad feature to track. For instance, in an image of a tree, a horizontal twig in the foreground can intersect a vertical twig in the background. This intersection occurs only in the image, not in the world, since the two twigs are at different depths. Any selection criterion would pick the intersection as a good feature to track, and yet there is no real world feature there to speak of. The measure of dissimilarity defined in equation (3) can often indicate that something is going wrong. Because of the potentially large number of frames through which a given feature can be tracked, the dissimilarity measure would not work well with a pure translation model. To illustrate this, consider figure 1, which shows three out of 21 frame details from Woody Allen's movie, Manhattan. The top row of figure 2 shows the results of tracking the traffic sign in this sequence.

如前节所定义，有着高度纹理内容的特征，对于跟踪来说仍然可能是不好的特征。比如，在树的图像中，前景中一个水平的树枝，可能与背景中的竖直树枝交叉。这个交叉只在图像中发生，而不是在世界中发生，因为两个树枝是在不同的深度的。任何选择规则都会将交叉点选做要跟踪的好特征，但并不对应着真实世界的特征。定义在式(3)中的不相似性度量，经常会指出有东西错了。因为可能有大量帧的要通过给定的特征来跟踪，不相似性度量在纯平移模型中效果不会太好。为描述这个，考虑图1中，展示的是Woody Allen电影Manhattan中21帧中的3帧。图2中的最上行展示的是在这个序列中跟踪交通标志的结果。

While the inter-frame changes are small enough for the pure translation tracker to work, the cumulative changes over 25 frames are rather large. In fact, the size of the sign increases by about 15 percent, and the dissimilarity measure (3) increases rather quickly with the frame number: as shown by the dashed and crossed line of figure 3. The solid and crossed line in the same figure shows the dissimilarity measure when also deformations are accounted for, that is, if the entire system (5) is solved for z. This new measure of dissimilarity remains small and roughly constant. The bottom row of figure 2 shows the same windows as in the top row, but warped by the computed deformations. The deformations make the five windows virtually equal to each other.

帧间的变化足够小，所以纯平移跟踪器可以工作，而25帧的累计变化是非常大的。实际上，标记的大小增加了大约15%，而不相似性度量(3)随着帧数也增加的很快：如图3中的虚线和交叉线所示。图中的实线和交叉线，是计入了形变时的不相似性度量，即，对z求解整个系统(5)。这个新的不相似性度量，仍然很小，大概就是常数。图2的底行给出了上行中一样的窗口，但是用计算得到的形变进行了变形。形变使得5个窗口实际上几乎相等。

The two circled curves in figure 3 refer to another feature from the same sequence, shown in figure 4. The top row of figure 5 shows the feature window through five frames. In the middle frame the traffic sign begins to occlude the original feature. The circled curves in figure 3 are the dissimilarity measures under affine motion (solid) and pure translation (dashed). The sharp jump in the affine motion curve around frame 4 indicates the occlusion. The bottom row of figure 5 shows that the deformation computation attempts to deform the traffic sign into a window.

图3中的两个圆圈曲线，是指同样序列中的另一个特征，如图4所示。图5中的上行展示了5帧中的特征窗口。在中间帧中，交通标志开始遮挡原始特征。图3中的圆圈曲线是在仿射运动和纯平移时的不相似性度量。在帧4不仅的仿射运动的尖锐跳变，表明了是遮挡。图5的底行表明，形变计算试图将交通标志形变到一个窗口。

## 6. Convergence

The simulations in this section show that when the affine motion model is correct our iterative tracking algorithm converges even when the starting point is far removed from the true solution. The first series of simulations are run on the four circular blobs shown in the leftmost column of figure 6. The three motions of table l are considered. To see their effects, compare the first and last column of figure 6. The images in the last column are the images warped, translated, and corrupted with random Gaussian noise with a standard deviation equal to 16 percent of the maximum image intensity. The images in the intermediate columns are the results of the deformations and translations to which the tracking algorithm subjects the images in the leftmost column after 4, 8, and 19 iterations, respectively. The algorithm works correctly, and makes the images in the fourth column of figure 6 as similar as possible to those in the fifth column.

本节中的仿真表明，当仿射运动模型是正确的时候，我们的迭代跟踪算法在起始点离真实解很远时，也会收敛。第一个仿真系列是在图6中的4个圆形blobs上进行的。最后一列的图像是经过变形、平移和随机高斯噪声污染后的图像，其标准差等于最大灰度亮度的16%。中间列的图像是，跟踪算法在4，8，19次迭代后的图像。算法效果正确，使图6的第4列与第5列的图像尽可能的相似。

Figure 7 plots the dissimilarity measure (as a fraction of the maximum image intensity), translation error (in pixels), and deformation error (Frobenius norm of the residual deformation matrix) as a function of the frame number (first three columns), as well as the intermediate displacements and deformations (last two columns). Deformations are represented in the fifth column of figure 7 by two vectors each, corresponding to the two columns of the transformation matrix A = 1 + D. Table 1 shows the final numerical values.

图7画出了不相似性度量（作为最大灰度的一部分），平移误差（以像素为单位），形变误差，作为帧数的函数，以及中间的位移和形变。形变是在图7中第5列用两个矢量表示的，对应着变换矩阵A=1+D中的两列。表1给出了最终的数值。

Figure 8 shows a similar experiment with a more complex image (from MATLAB). Finally, figure 9 shows an attempt to match two completely different images: four blobs and a cross. The algorithm tries to do its best by aligning the blobs with the cross, but the dissimilarity (left plot at the bottom of figure 9) remains high throughout.

图8用更复杂的图像展示了类似的实验。最后，图9展示了两幅完全不同的图像的匹配：4个blobs和一个十字。算法尽了最大努力，将blobs与十字对齐，但不相似度一直保持很高。

## 7. Monitoring Features

This section presents some experiments with real images and shows how features can be monitored during tracking to detect potentially bad features. Figure 10 shows the first frame of a 26-frame sequence. A Pulnix camera equipped with a 16mm lens moves forward 2mm per frame. Because of the forward motion, features loom larger from frame to frame. The pure translation model is sufficient for inter-frame tracking but not to monitor features, as discussed below. Figure 11 displays the 102 features selected according to the criterion introduced in section 4. To limit the number of features and to use each portion of the image at most once, the constraint was imposed that no two feature windows can overlap in the first frame. Figure 12 shows the dissimilarity of each feature under the pure translation motion model, that is, with the deformation matrix D set to zero for all features. This dissimilarity is nearly useless for feature monitoring: except for features 58 and 89, all features have comparable dissimilarities, and no clean discrimination can be drawn between good and bad features.

本节提出了一些真实图像的试验，表明特征怎样在跟踪的过程中进行监控，以检测潜在的坏特征。图10展示了26帧序列中的第一帧。Pulnix相机，16mm镜头，每帧向前进2mm。由于是前向运动，特征在逐帧中逐渐变大。纯平移模型足以进行帧间的跟踪，但不能监控特征，下面会进行叙述。图11展示了根据第4部分引入的规则选择的102个特征。为限制特征的数量，使用图像的每个部分最多一次，施加了下面的约束，没有两个特征窗口在第一帧中可以重合。图12展示了在纯平移运动模型下，每个特征的不相似度，即形变矩阵D对所有特征设为0。这种不相似度对于特征监控几乎是无用的：除了特征58和59，所有特征都有类似的不相似度，在好的特征和不好的特征之间，没有任何区分。

From figure 13 we see that features 58 is at the boundary of the block with a letter U visible in the lower right-hand side of the figure. The feature window straddles the vertical dark edge of the block in the foreground as well as parts of the letters Cra in the word "Crayola" in the background. Six frames of this window are visible in the third row of figure 14. As the camera moves forward, the pure translation tracking stays on top of approximately the same part of the image. However, the gap between the vertical edge in the foreground and the letters in the background widens, and it becomes harder to warp the current window into the window in the first frame, thereby leading to the rising dissimilarity. The changes in feature 89 are seen even more easily. This feature is between the edge of the book in the background and a lamp partially visible behind it in the top right corner of figure 13. As the camera moves forward, the shape of the glossy reflection on the lamp shade changes as it becomes occluded (see the last row of figure 14).

从图13，我们可以看到58是在一个块的边缘上，其右下有一个字母U。特征窗口横跨了前景中模块的竖直暗边缘，和背景中单词Crayola的字母Cra。这个窗口的6帧在图14中的第三行是可见的。当相机向前移动时，纯平移的跟踪保持在图像的同样部分中。但是，前景中竖直边缘，和背景中的字母的间隙变大了，更难将当前窗口变形到第一帧中，因此得到的不相似性迅速变大。在特征89上的变化可以更明显的看到。这个特征是在背景中书的边缘，和在其后部分可见的一个灯中间，如图13的右边角落所示。在相机向前移动时，灯的光泽反射的形状在其逐渐遮挡时在变化（见图14的最后一行）。

Although these bad features would be detected because of their high dissimilarity, many other bad features would pass unnoticed. For instance, feature 3 in the lower right of figure 13 is affected by a substantial disocclusion of the lettering on the Crayola box by the U block as the camera moves forward, as well as a slight disocclusion by the "3M" box on the right (see the top row of figure 14). Yet with a pure translation model the dissimilarity of feature 3 is not substantially different from that of all the other features in figure 12. In fact, the looming caused by the camera's forward motion dominates, and reflects in the overall upward trend of the majority of curves in figure 12. Similar considerations hold, for instance, for features 78 (a disocclusion), 24 (an occlusion), and 4 (a disocclusion) labeled in figure 13.

虽然这些坏的特征由于其很高的不相似性会检测到，但很多其他坏的特征会不被注意到。比如，图13中的右边较低的特征3，在相机向前移动时候会受到Crayola箱子被U块的遮挡，以及右边3M箱子的轻微遮挡。但用纯平移模型，特征3的不相似性与图12中的所有其他特征，并没有什么特别的不同。实际上，相机向前运动的逼近效果占主要作用，反应在图12中大多数曲线总体向上的趋势中。类似的考虑也成立，比如，对于图13中的特征78，24和4。

Now compare the pure translation dissimilarity of figure 12 with the affine motion dissimilarity of figure 15. The thick stripe of curves at the bottom represents all good features, including features 1,21,30,53, labeled in figure 13. These four features are all good, being immune from occlusions or glossy reflections: 1 and 21 are lettering on the "Crayola" box (the second row of figure 14 shows feature 21 as an example), while features 30 and 53 are details of the large title on the book in the background (upper left in figure 13). The bad features 3,4,58,78,89, on the other hand, stand out very clearly in figure 15: discrimination is now possible.

现在对比图12中的纯平移不相似性，和图15中的仿射运动不相似性。在底部的曲线的厚厚的条纹，表示了所有好的特征，包括图13中的1，21，30，53。这4个特征都是好的，对遮挡或光泽反射是免疫的。而坏的特征3,4,58,78,89，则在图15中非常清晰：现在可以进行区分了。

Features 24 and 60 deserve a special discussion, and are plotted with dashed lines in figure 15. These two features are lettering detail on the rubber cement bottle in the lower center of figure 13. The fourth row of figure 14 shows feature 60 as an example. Although feature 24 suffers an additional slight occlusion as the camera moves forward, these two features stand out from the very beginning, and their dissimilarity curves are very erratic throughout the sequence. This is because of aliasing: from the fourth row of figure 14, we see that feature 60 (and similarly feature 24) contains very small lettering, of size comparable to the image's pixel size (the feature window is 25 x 25 pixels). The matching between one frame and the next is haphazard, because the characters in the lettering are badly aliased. This behavior is not a problem: erratic dissimilarities indicate trouble, and the corresponding features ought to be abandoned.

特征24和60值得特别讨论，在图15中以虚线画了出来。图14的第4行，展示了特征60的例子。虽然特征24在相机向前的时候有略微的遮挡，这两个特征从开始就很突出，其不相似性曲线在整个序列中非常不规则。这是因为混淆：从图14中的第4行，我们看到特征60（和类似的特征24）包含非常小的字母，与图像的字母大小非常类似。一帧和下一帧的匹配是随意的，因为字母是不规则的锯齿状的。这种行为并不是一个问题：不规则的不相似性表明是问题，对应的特征应当进行抛弃。

## 8. Conclusion

In this paper, we have proposed a method for feature selection, a tracking algorithm based on a model of affine image changes, and a technique for monitoring features during tracking. Selection specifically maximizes the quality of tracking, and is therefore optimal by construction, as opposed to more ad hoc measures of texturedness. Monitoring is computationally inexpensive and sound, and helps discriminating between good and bad features based on a measure of dissimilarity that uses affine motion as the underlying image change model.

本文中，我们提出了一种特征选择的方法，一种基于仿射图像变换模型的跟踪算法，和一种在跟踪中监控特征的技术。选择特别最大化了跟踪的质量，因此与更加随意的纹理度度量来说，从构建上就是最优秀的。监控在计算上并不复杂，而且很合理，帮助在好的和坏的特征之间进行区分，这是基于使用了仿射运动作为潜在的图像变化模型的不相似性度量。

Of course, monitoring feature dissimilarity does not solve all the problems of tracking. In some situations, a bright spot on a glossy surface is a bad (that is, nonrigid) feature, but may change little over a long sequence: dissimilarity may not detect the problem. However, even in principle, not everything can be decided locally. Rigidity is not a local feature, so a local method cannot be expected to always detect its violation. On the other hand, many problems can indeed be discovered locally and these are the target of the investigation in this paper. Our experiments and simulations show that monitoring is indeed effective in realistic circumstances. A good discrimination at the beginning of the processing chain can reduce the remaining bad features to a few outliers, rather than leaving them an overwhelming majority. Outlier detection techniques at higher levels in the processing chain are then more likely to succeed.

当然，监控特征不相似度并没有解决跟踪的所有问题。在一些情况下，光泽平面上的一个亮点，是一个不好的特征（即，非刚体），但在一个长的序列中的变化可能很小：不相似度可能不会检测到这个问题。但是，即使在原则上，并不是所有事物都可以在局部上决定的。刚性并不是一个局部特征，所以一种局部方法不能期望可以永远检测到这些冲突。另一方面，很多问题确实可以从局部被发现，这些是本文的研究目标。我们的实验和仿真表明，监控在实际的场景中是的确有效的。在处理链的开始的时候就有好的区分，可以降低剩余的坏的特征，成为几个离群点，而不是让其成为主要部分。更高层的离群点检测技术，会很可能成功。