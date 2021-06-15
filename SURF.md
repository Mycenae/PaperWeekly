# SURF: Speeded Up Robust Features

Herbert Bay, et al. ETH Zurich

## 0. Abstract

In this paper, we present a novel scale- and rotation-invariant interest point detector and descriptor, coined SURF (Speeded Up Robust Features). It approximates or even outperforms previously proposed schemes with respect to repeatability, distinctiveness, and robustness, yet can be computed and compared much faster.

本文中，我们提出了一种新的尺度不变，旋转不变的感兴趣点检测器和描述子，称为SURF（加速的稳健特征）。与之前提出的方案相比，在可重复性、区分性和稳健性上，本文都达到了接近或超过的水平，但可以进行更加快速的计算和比较。

This is achieved by relying on integral images for image convolutions; by building on the strengths of the leading existing detectors and descriptors (in casu, using a Hessian matrix-based measure for the detector, and a distribution-based descriptor); and by simplifying these methods to the essential. This leads to a combination of novel detection, description, and matching steps. The paper presents experimental results on a standard evaluation set, as well as on imagery obtained in the context of a real-life object recognition application. Both show SURF’s strong performance.

这是用积分图像进行图像卷积得到的；在现有最好的检测器和描述子之上（在这种情况下，对检测器使用一个基于Hessian矩阵的度量，和一个基于分布的描述子）；并讲这些方法简化到最简。这带来了新的检测、描述和匹配步骤的组合。文章在标准评估集上给出了试验结果，也在实际生活的目标识别应用中得到的图像中进行了评估。两者都表明SURF性能非常强劲。

## 1. Introduction

The task of finding correspondences between two images of the same scene or object is part of many computer vision applications. Camera calibration, 3D reconstruction, image registration, and object recognition are just a few. The search for discrete image correspondences – the goal of this work – can be divided into three main steps. First, ‘interest points’ are selected at distinctive locations in the image, such as corners, blobs, and T-junctions. The most valuable property of an interest point detector is its repeatability, i.e. whether it reliably finds the same interest points under different viewing conditions. Next, the neighbourhood of every interest point is represented by a feature vector. This descriptor has to be distinctive and, at the same time, robust to noise, detection errors, and geometric and photometric deformations. Finally, the descriptor vectors are matched between different images. The matching is often based on a distance between the vectors, e.g. the Mahalanobis or Euclidean distance. The dimension of the descriptor has a direct impact on the time this takes, and a lower number of dimensions is therefore desirable.

在两幅同样场景或目标的图像中找到对应性，是很多计算机视觉应用中的一部分。相机标定，3D重建，图像配准，和目标识别，只是一部分。本文的目标是搜索离散图像对应性，可以分成三个主要步骤。第一，在图像独特的地方选择感兴趣的点，比如，角点，blobs和T结点。感兴趣点检测器的最重要的性质是其可重复性，即，在不同的观测条件下，可以可靠的找到相同的感兴趣点。第二，每个感兴趣点的邻域都用一个特征向量表示。这个描述子必须是独特的，同时对噪声、检测错误、几何形变和灰度变化要稳健。最后，描述子向量在不同图像中进行匹配。匹配通常都是基于向量间的距离，如，Mahalanobis距离或欧式距离。描述子的维度对耗时有直接影响，因此通常希望维度更低。

It has been our goal to develop both a detector and descriptor, which in comparison to the state-of-the-art are faster to compute, while not sacrificing performance. In order to succeed, one has to strike a balance between the above requirements, like reducing the descriptor’s dimension and complexity, while keeping it sufficiently distinctive.

开发一个检测器和描述子，与目前最好的计算起来更快，而且不损失性能，一直是我们的目标。为成功，必须在上述需求之间取得平衡，比如降低描述子的维度和复杂度，而使其保持足够独特。

A wide variety of detectors and descriptors have already been proposed in the literature (e.g. [1–6]). Also, detailed comparisons and evaluations on benchmarking datasets have been performed [7–9]. While constructing our fast detector and descriptor, we built on the insights gained from this previous work in order to get a feel for what are the aspects contributing to performance. In our experiments on benchmark image sets as well as on a real object recognition application, the resulting detector and descriptor are not only faster, but also more distinctive and equally repeatable.

现有文献中已经提出了大量检测器和描述子。同时，[7-9]在基准测试数据集中进行了详细的比较和评估。构建我们的快速检测器和描述子时，我们是在之前的工作上进行的，以感受我们是在哪方面对性能做出了贡献。在我们对基准测试图像数据集的试验，以及一个真实的目标识别应用中，得到的检测器和描述子不仅更快，而且更独特，同时一样可重复。

When working with local features, a first issue that needs to be settled is the required level of invariance. Clearly, this depends on the expected geometric and photometric deformations, which in turn are determined by the possible changes in viewing conditions. Here, we focus on scale and image rotation invariant detectors and descriptors. These seem to offer a good compromise between feature complexity and robustness to commonly occurring deformations. Skew, anisotropic scaling, and perspective effects are assumed to be second-order effects, that are covered to some degree by the overall robustness of the descriptor. As also claimed by Lowe [2], the additional complexity of full affine-invariant features often has a negative impact on their robustness and does not pay off, unless really large viewpoint changes are to be expected. In some cases, even rotation invariance can be left out, resulting in a scale-invariant only version of our descriptor, which we refer to as ’upright SURF’ (U-SURF). Indeed, in quite a few applications, like mobile robot navigation or visual tourist guiding, the camera often only rotates about the vertical axis. The benefit of avoiding the overkill of rotation invariance in such cases is not only increased speed, but also increased discriminative power. Concerning the photometric deformations, we assume a simple linear model with a scale factor and offset. Notice that our detector and descriptor don’t use colour.

在采用局部特征时，第一个要确定的问题是，需要的不变性的层次。很明显，这依赖于期望的几何和灰度变换，这又是由观察条件的可能变化决定的。这里，我们聚焦在尺度和图像旋转不变的检测器和描述子上。这在特征复杂度和对经常发生的形变之间，取得了很好的折中。扭曲，各项异性缩放，和视角的效果，这些都认为是第二级的效果，由描述子的总体稳健性在一定程度上覆盖。Lowe [2]也声称，完整的仿射不变特征的额外复杂度，通常对稳健性有负面效果，而且无法弥补，除非可以期待真的很大的视角变化。在一些情况下，即使旋转不变性也可以去掉，得到一个只有尺度不变版的描述子，我们称之为竖直SURF (U-SURF)。确实，在很多应用中，如移动机器人导航，或视觉旅游导游，相机只沿着垂直轴进行旋转。在这种情况下，避免过多的旋转不变性，不仅增加速度，还增加了区分能力。考虑灰度形变，我们假设了一个简单的线性模型，有尺度因素和偏移。注意，我们的检测器和描述子没有使用色彩。

The paper is organised as follows. Section 2 describes related work, on which our results are founded. Section 3 describes the interest point detection scheme. In section 4, the new descriptor is presented. Finally, section 5 shows the experimental results and section 6 concludes the paper.

本文组织如下。第2部分描述了相关的工作，我们的结果在此之上进行。第3部分描述了感兴趣点检测方案。在第4部分，给出了我们的新的描述子。最后，第5部分给出了试验结果，第6部分进行了总结。

## 2. Related Work

**Interest Point Detectors**. The most widely used detector probably is the Harris corner detector [10], proposed back in 1988, based on the eigenvalues of the second-moment matrix. However, Harris corners are not scale-invariant. Lindeberg introduced the concept of automatic scale selection [1]. This allows to detect interest points in an image, each with their own characteristic scale. He experimented with both the determinant of the Hessian matrix as well as the Laplacian (which corresponds to the trace of the Hessian matrix) to detect blob-like structures. Mikolajczyk and Schmid refined this method, creating robust and scale-invariant feature detectors with high repeatability, which they coined Harris-Laplace and Hessian-Laplace [11]. They used a (scale-adapted) Harris measure or the determinant of the Hessian matrix to select the location, and the Laplacian to select the scale. Focusing on speed, Lowe [12] approximated the Laplacian of Gaussian (LoG) by a Difference of Gaussians (DoG) filter.

感兴趣点检测器。最广泛使用的检测器，可能就是Harris角点检测器[10]，在1988年提出，基于二阶动量矩阵的特征值。但是，Harris角点不是尺度不变的。Lindeberg提出了自动尺度选择的概念[1]。这使得可以在图像中检测兴趣点，每个点都有其特征尺度。他用Hessian矩阵的行列式和Laplacian（对应于Hessian矩阵的迹）进行了试验，以检测类似blob的结构。Mikolajczyk和Schmid提炼了这个方法，创建了稳健、尺度不变的特征检测器，有很高的可重复性，称为Harris-Laplace和Hessian-Laplace[11]。他们使用了适应尺度的Harris度量，或Hessian矩阵的横列式，以选择位置，以及Laplacian来选择尺度。聚焦在速度上，Lowe[12]用DoG滤波器近似LoG。

Several other scale-invariant interest point detectors have been proposed. Examples are the salient region detector proposed by Kadir and Brady [13], which maximises the entropy within the region, and the edge-based region detector proposed by Jurie et al. [14]. They seem less amenable to acceleration though. Also, several affine-invariant feature detectors have been proposed that can cope with longer viewpoint changes. However, these fall outside the scope of this paper.

提出了几个其他的尺度不变的兴趣点检测器。比如，有Kadir和Brady[13]提出的显著区域检测器，对区域内的熵进行最大化，以及Jurie等[14]提出的基于边缘的区域检测器。他们似乎更易于加速。同时，提出了几个仿射不变的特征检测器，可以处理更长的视角变化。但是，这不在本文讨论范围内。

By studying the existing detectors and from published comparisons [15, 8], we can conclude that (1) Hessian-based detectors are more stable and repeatable than their Harris-based counterparts. Using the determinant of the Hessian matrix rather than its trace (the Laplacian) seems advantageous, as it fires less on elongated, ill-localised structures. Also, (2) approximations like the DoG can bring speed at a low cost in terms of lost accuracy.

通过研究现有的检测器，以及从发表的比较中[15,8]，我们得出结论，(1)与基于Harris的检测器相比，基于Hessian的检测器更稳定，更加可重复；使用Hessian矩阵的行列式而不是其迹，似乎效果更好，因为不会对拉长的、位置病态的结构有效果；(2)类似DoG这样的近似，可以以较低代价得到速度提升，损失的准确率很低。

**Feature Descriptors**. An even larger variety of feature descriptors has been proposed, like Gaussian derivatives [16], moment invariants [17], complex features [18, 19], steerable filters [20], phase-based local features [21], and descriptors representing the distribution of smaller-scale features within the interest point neighbourhood. The latter, introduced by Lowe [2], have been shown to outperform the others [7]. This can be explained by the fact that they capture a substantial amount of information about the spatial intensity patterns, while at the same time being robust to small deformations or localisation errors. The descriptor in [2], called SIFT for short, computes a histogram of local oriented gradients around the interest point and stores the bins in a 128-dimensional vector (8 orientation bins for each of the 4 × 4 location bins).

**特征描述子**。提出了的特征描述子更多，如高斯导数[16]，动量不变性[17]，复杂特征[18,19]，可操纵滤波器[20]，基于相位的局部特征[21]，和表示兴趣点邻域内更小尺度特征分布的描述子。后者由Lowe [2]提出，已经证明了比其他的要好[7]。这可以由下面的事实解释，即捕获了很多空间灰度模式变化的信息，同时对小的形变或定位误差更加稳健。[2]中的描述子，称为SIFT，在感兴趣点附近计算了局部有向梯度的直方图，在一个128维的向量中存储这些bins （对每个4x4的位置bins，都有8个方向bins）。

Various refinements on this basic scheme have been proposed. Ke and Sukthankar [4] applied PCA on the gradient image. This PCA-SIFT yields a 36-dimensional descriptor which is fast for matching, but proved to be less distinctive than SIFT in a second comparative study by Mikolajczyk et al. [8] and slower feature computation reduces the effect of fast matching. In the same paper [8], the authors have proposed a variant of SIFT, called GLOH, which proved to be even more distinctive with the same number of dimensions. However, GLOH is computationally more expensive.

在这个基础方案上提出了各种精炼。Ke和Sukthankar[4]对梯度图像使用了PCA。这种PCA-SIFT得到了36维描述子，匹配起来非常快，但在Mikolajczyk等的二次比较工作中[8]，与SIFT相比，独特性没有那么强，特征计算比较慢，也降低了快速匹配的效果。在同样的文章中[8]，作者提出了SIFT的变体，称为GLOH，在相同的维度上特征更加独特。但是，GLOH在计算上更加复杂。

The SIFT descriptor still seems to be the most appealing descriptor for practical uses, and hence also the most widely used nowadays. It is distinctive and relatively fast, which is crucial for on-line applications. Recently, Se et al. [22] implemented SIFT on a Field Programmable Gate Array (FPGA) and improved its speed by an order of magnitude. However, the high dimensionality of the descriptor is a drawback of SIFT at the matching step. For on-line applications on a regular PC, each one of the three steps (detection, description, matching) should be faster still. Lowe proposed a best-bin-first alternative [2] in order to speed up the matching step, but this results in lower accuracy.

SIFT描述子在实际使用中看起来仍然是最有吸引力的描述子，因此现在仍然是使用最广泛的。其独特性很强，相对较快，这对于在线应用是非常关键的。最近，Se等[22]在FPGA上实现了SIFT，将其运行速度提高了一个数量级。但是，描述子维度很高，是SIFT在匹配阶段的一个不足之处。对于在常规PC上的在线应用，三个步骤（检测、描述、匹配）中的每一个都应当更快。Lowe提出了一种best-bin-first的替代，以加速匹配步骤，但这使得准确率下降了。

**Our approach**. In this paper, we propose a novel detector-descriptor scheme, coined SURF (Speeded-Up Robust Features). The detector is based on the Hessian matrix [11, 1], but uses a very basic approximation, just as DoG [2] is a very basic Laplacian-based detector. It relies on integral images to reduce the computation time and we therefore call it the ’Fast-Hessian’ detector. The descriptor, on the other hand, describes a distribution of Haar-wavelet responses within the interest point neighbourhood. Again, we exploit integral images for speed. Moreover, only 64 dimensions are used, reducing the time for feature computation and matching, and increasing simultaneously the robustness. We also present a new indexing step based on the sign of the Laplacian, which increases not only the matching speed, but also the robustness of the descriptor.

**我们的方法**。本文中，我们提出了一种新的检测器-描述子的方案，称为SURF（加速的稳健特征）。检测器是基于Hessian矩阵的，但使用了一种非常基础的近似，就像DoG是一种很基础的基于Laplacian的检测器。它基于积分图像，降低了计算时间，因此我们称之“快速Hessian”检测器。另一方面，描述子描述了在兴趣点邻域中的Haar-小波响应的分布。我们再次利用积分图像加速计算。而且，只使用了64个维度，降低了特征计算和匹配的时间，同时增加了稳健性。

In order to make the paper more self-contained, we succinctly discuss the concept of integral images, as defined by [23]. They allow for the fast implementation of box type convolution filters. The entry of an integral image $I_Σ(x)$ at a location x = (x, y) represents the sum of all pixels in the input image I of a rectangular region formed by the point x and the origin, $I_Σ(x) = \sum^{i≤x}_{i=0} \sum^{j≤y}_{j=0} I(i, j)$. With $I_Σ$ calculated, it only takes four additions to calculate the sum of the intensities over any upright, rectangular area, independent of its size.

为使文章更加独立，我们简要讨论一下积分图像的概念，如[23]中的定义。这使方块类型的卷积滤波器实现更加快速。积分图像$I_Σ(x)$在位置x = (x,y)处的值，表示输入图像I在点x和原点所形成的矩形区域的所有像素的和， $I_Σ(x) = \sum^{i≤x}_{i=0} \sum^{j≤y}_{j=0} I(i, j)$。计算了$I_Σ$后，计算竖直的矩形区域内的灰度和，就只需要四次加法，与矩形区域大小无关。

## 3. Fast-Hessian Detector

We base our detector on the Hessian matrix because of its good performance in computation time and accuracy. However, rather than using a different measure for selecting the location and the scale (as was done in the Hessian-Laplace detector [11]), we rely on the determinant of the Hessian for both. Given a point x = (x, y) in an image I, the Hessian matrix H(x, σ) in x at scale σ is defined as follows

我们的检测器基于Hessian矩阵，因为在计算时间和准确率上性能很好。但是，我们没有使用不同的度量来选择位置和尺度（在Hessian-Laplace检测器[11]上是这样的），我们依赖于Hessian的行列式来选择位置和尺度。给定图像I中的点x = (x,y)，在尺度σ上的x点的Hessian矩阵定义如下

$$H(x,σ) = [L_{xx}(x,σ), L_{xy}(x,σ); L_{xy}(x,σ), L_{yy}(x,σ)]$$(1)

where $L_{xx}(x, σ)$ is the convolution of the Gaussian second order derivative $∂^2/∂x^2 g(σ)$ with the image I in point x, and similarly for $L_{xy}(x, σ)$ and $L_{yy}(x, σ)$. 其中$L_{xx}(x, σ)$是高斯二阶导数$∂^2/∂x^2 g(σ)$与图像I在点x处的卷积，$L_{xy}(x, σ)$和$L_{yy}(x, σ)$也是类似的计算。

Gaussians are optimal for scale-space analysis, as shown in [24]. In practice, however, the Gaussian needs to be discretised and cropped (Fig. 1 left half), and even with Gaussian filters aliasing still occurs as soon as the resulting images are sub-sampled. Also, the property that no new structures can appear while going to lower resolutions may have been proven in the 1D case, but is known to not apply in the relevant 2D case [25]. Hence, the importance of the Gaussian seems to have been somewhat overrated in this regard, and here we test a simpler alternative. As Gaussian filters are non-ideal in any case, and given Lowe’s success with LoG approximations, we push the approximation even further with box filters (Fig. 1 right half). These approximate second order Gaussian derivatives, and can be evaluated very fast using integral images, independently of size. As shown in the results section, the performance is comparable to the one using the discretised and cropped Gaussians.

对于尺度空间分析，高斯函数是最优的，[24]中已经进行了证明。但是，在实践中，高斯函数需要被离散化和剪切（图1左半），即使用高斯滤波器，只要得到的图像进行降采样，就会产生混叠失真。同时，在进入低分辨率时，没有新结构会出现的性质，在1D情况下可能已经得到了证明，但在相关的2D情况下还是不成立的。因此，高斯的重要性在这方面是被高估了，这里我们测试了一个更简单的替代品。因为高斯滤波器在任何情况下都是非理想的，而且Lowe在LoG的近似上非常成功，我们将近似进一步推进到box滤波器的近似（图1右半）。这近似了二阶高斯导数，可以用积分图像进行快速计算，与大小无关。在下面的小节中可以看到，其性能与使用离散化剪切的高斯函数是类似的。

The 9 × 9 box filters in Fig. 1 are approximations for Gaussian second order derivatives with σ = 1.2 and represent our lowest scale (i.e. highest spatial resolution). We denote our approximations by $D_{xx}, D_{yy}$, and $D_{xy}$. The weights applied to the rectangular regions are kept simple for computational efficiency, but we need to further balance the relative weights in the expression for the Hessian’s determinant with $\frac {|L_{xy}(1.2)|_F |D_{xx}(9)|_F} {|L_{xx}(1.2)|_F |D_{xy}(9)|_F}$ = 0.912... ≃ 0.9, where $|x|_F$ is the Frobenius norm. This yields

图1中的9x9 box滤波器是σ = 1.2时的高斯二阶导数的近似，表示最低尺度（即，最高空间分辨率）。我们将近似表示为$D_{xx}, D_{yy}$和$D_{xy}$。对矩形区域应用的权重保持简单，以使计算很高效，但我们需要在表达式中，用$\frac {|L_{xy}(1.2)|_F |D_{xx}(9)|_F} {|L_{xx}(1.2)|_F |D_{xy}(9)|_F}$ = 0.912... ≃ 0.9对Hessian行列式进一步平衡相对权重，其中$|x|_F$是Frobenius范数。这会得到

$$det(H_{approx}) = D_{xx}D_{yy} − (0.9D_{xy})^2$$(2)

Furthermore, the filter responses are normalised with respect to the mask size. This guarantees a constant Frobenius norm for any filter size. 而且，滤波器响应是相对mask大小进行归一化的。这保证了对于任意的滤波器大小，都是常数Frobenius范数。

Scale spaces are usually implemented as image pyramids. The images are repeatedly smoothed with a Gaussian and subsequently sub-sampled in order to achieve a higher level of the pyramid. Due to the use of box filters and integral images, we do not have to iteratively apply the same filter to the output of a previously filtered layer, but instead can apply such filters of any size at exactly the same speed directly on the original image, and even in parallel (although the latter is not exploited here). Therefore, the scale space is analysed by up-scaling the filter size rather than iteratively reducing the image size. The output of the above 9 × 9 filter is considered as the initial scale layer, to which we will refer as scale s = 1.2 (corresponding to Gaussian derivatives with σ = 1.2). The following layers are obtained by filtering the image with gradually bigger masks, taking into account the discrete nature of integral images and the specific structure of our filters. Specifically, this results in filters of size 9×9, 15×15, 21×21, 27×27, etc. At larger scales, the step between consecutive filter sizes should also scale accordingly. Hence, for each new octave, the filter size increase is doubled (going from 6 to 12 to 24). Simultaneously, the sampling intervals for the extraction of the interest points can be doubled as well.

尺度空间通常是用图像金字塔实现的。图像用高斯函数进行平滑，然后进行下采样，这个过程重复进行，就可以得到高阶金字塔。由于使用了box滤波器和积分图像，我们不需要对之前滤波过的层的输出迭代使用同样的滤波器，而是在原始图像上以相同的速度，使用任意大小的滤波器，甚至是并行使用（但后者并没有进行研究）。因此，尺度空间的分析是通过加大滤波器的大小，而不是迭代着降低图像的大小。上面的9x9滤波器的输出就是初始尺度层，我们称之为尺度s = 1.2（对应着s = 1.2的高斯导数）。后续的层，是通过使用逐渐更大的masks来对图像滤波得到的，这将积分图像的离散本质和我们的滤波器的特殊结构就考虑在内了。具体的，这会得到大小为9×9, 15×15, 21×21, 27×27等的滤波器。在更大的尺度上，连续两个滤波器大小之间的步长，也需要进行缩放。因此，对每个新的octave，滤波器大小的增加都加倍了（从6到12到24）。同时，提取兴趣点的采样间隔也可以加倍。

As the ratios of our filter layout remain constant after scaling, the approximated Gaussian derivatives scale accordingly. Thus, for example, our 27 × 27 filter corresponds to σ = 3 × 1.2 = 3.6 = s. Furthermore, as the Frobenius norm remains constant for our filters, they are already scale normalised [26].

因为我们的滤波器布局的比例在缩放后是一样的，近似的高斯导数也进行了相应的缩放。因此，比如，我们的27x27滤波器对应着σ = 3 × 1.2 = 3.6 = s。而且，因为Frobenius范数对我们的滤波器保持常数，他们就已经进行了尺度归一化。

In order to localise interest points in the image and over scales, a non-maximum suppression in a 3 × 3 × 3 neighbourhood is applied. The maxima of the determinant of the Hessian matrix are then interpolated in scale and image space with the method proposed by Brown et al. [27]. Scale space interpolation is especially important in our case, as the difference in scale between the first layers of every octave is relatively large. Fig. 2 (left) shows an example of the detected interest points using our ’Fast-Hessian’ detector.

为在图像中以及多个尺度上定位兴趣点，在3x3x3的邻域上应用了非最大抑制。Hessian矩阵的行列式的最大值，在尺度中和图像空间中进行插值，插值方法为Brown等[27]中提出的方法。尺度空间插值在我们的情况中是尤其重要的，因为在每个octave的第一层的尺度上的差异是相对较大的。图2（左）展示了使用了我们的快速Hessian检测器检测的兴趣点的一个例子。

## 4. SURF Descriptor

The good performance of SIFT compared to other descriptors [8] is remarkable. Its mixing of crudely localised information and the distribution of gradient related features seems to yield good distinctive power while fending off the effects of localisation errors in terms of scale or space. Using relative strengths and orientations of gradients reduces the effect of photometric changes.

与其他描述子相比，SIFT的好性能是显著的。粗糙定位信息和与梯度相关特征的分布的混合，似乎产生了很好的区分能力，同时避开了定位错误的效果。使用相对强度和梯度方向，降低了亮度变化的效果。

The proposed SURF descriptor is based on similar properties, with a complexity stripped down even further. The first step consists of fixing a reproducible orientation based on information from a circular region around the interest point. Then, we construct a square region aligned to the selected orientation, and extract the SURF descriptor from it. These two steps are now explained in turn. Furthermore, we also propose an upright version of our descriptor (U-SURF) that is not invariant to image rotation and therefore faster to compute and better suited for applications where the camera remains more or less horizontal.

提出的SURF描述子是基于类似的性质的，复杂度得到了进一步降低。第一步，在兴趣点附近圈定一个圆形区域，基于此区域的信息，固定一个可复现的方向。然后，沿着选定的方向，构建一个方形区域，从中提取SURF描述子。这两步现在按顺序进行解释。而且，我们还提出描述子的竖直版本(U-SURF)，对图像旋转并不是不变的，因此计算起来更快，更适合相机大致保持水平的应用。

### 4.1 Orientation Assignment

In order to be invariant to rotation, we identify a reproducible orientation for the interest points. For that purpose, we first calculate the Haar-wavelet responses in x and y direction, shown in Fig. 2, and this is a circular neighbourhood of radius 6s around the interest point, with s the scale at which the interest point was detected. Also the sampling step is scale dependent and chosen to be s. In keeping with the rest, also the wavelet responses are computed at that current scale s. Accordingly, at high scales the size of the wavelets is big. Therefore, we use again integral images for fast filtering. Only six operations are needed to compute the response in x or y direction at any scale. The side length of the wavelets is 4s.

为对旋转不变，我们对兴趣点识别了一个可复现的方向。为此，我们首先计算了x和y方向的Haar小波响应，如图2所示，这是一个围绕兴趣点的圆形邻域，半径6s，s是检测到兴趣点的尺度。同时，采样步长是依赖于尺度的，选择为s。在后面保持一致，小波响应也是在目前的尺度s上计算的。相应的，在更高的尺度上，小波的规模是很大的。因此，我们再次使用积分图像进行快速滤波。只需要6次运算来计算x或y方向在任意尺度上的响应。小波的边长为4s。

Once the wavelet responses are calculated and weighted with a Gaussian (σ = 2.5s) centered at the interest point, the responses are represented as vectors in a space with the horizontal response strength along the abscissa and the vertical response strength along the ordinate. The dominant orientation is estimated by calculating the sum of all responses within a sliding orientation window covering an angle of π/3. The horizontal and vertical responses within the window are summed. The two summed responses then yield a new vector. The longest such vector lends its orientation to the interest point. The size of the sliding window is a parameter, which has been chosen experimentally. Small sizes fire on single dominating wavelet responses, large sizes yield maxima in vector length that are not outspoken. Both result in an unstable orientation of the interest region. Note the U-SURF skips this step.

一旦计算了小波响应，并用以兴趣点为中心的高斯(σ = 2.5s)函数进行加权，响应就用空间中的向量来表示，水平响应强度沿着横坐标，垂直响应强度沿着纵坐标。主要的方向的估计，是通过在覆盖角度π/3的滑动方向窗口计算所有响应的和。窗口内的水平和垂直响应进行求和。两个求和的响应产生了一个新的向量。最长的这种向量就是兴趣点的方向。滑动窗口的大小是一个参数，是通过试验选择的。小窗口对应单个主要的小波响应，大窗口产生向量长度上的最大值。两个都会得到兴趣区域的不稳定方向。注意U-SURF跳过了这个步骤。

### 4.2 Descriptor Components

For the extraction of the descriptor, the first step consists of constructing a square region centered around the interest point, and oriented along the orientation selected in the previous section. For the upright version, this transformation is not necessary. The size of this window is 20s. Examples of such square regions are illustrated in Fig. 2.

为提取描述子，第一步要围绕兴趣点为中心构建一个方形区域，其方向为在前一节选择的方向。对于竖直版，变换是没必要的。窗口的大小为20s。这种方形区域的例子如图2所示。

The region is split up regularly into smaller 4 × 4 square sub-regions. This keeps important spatial information in. For each sub-region, we compute a few simple features at 5×5 regularly spaced sample points. For reasons of simplicity, we call d_x the Haar wavelet response in horizontal direction and d_y the Haar wavelet response in vertical direction (filter size 2s). ”Horizontal” and ”vertical” here is defined in relation to the selected interest point orientation. To increase the robustness towards geometric deformations and localisation errors, the responses d_x and d_y are first weighted with a Gaussian (σ = 3.3s) centered at the interest point.

区域分裂成规则的4x4方形子区域。这保留了重要的空间信息。对每个子区域，我们在5x5的规则间隔的采样点上计算几个简单的特征。为简化起见，我们称d_x为水平方向的Haar小波响应，d_y为垂直方向的Haar小波响应（滤波器大小为2s）。这里的水平和垂直，是相对于选定的兴趣点方向定义的。为增大对几何形变和定位误差的稳健性，d_x和d_y的响应首先用以兴趣点为中心的高斯函数(σ = 3.3s)进行加权。

Then, the wavelet responses d_x and d_y are summed up over each subregion and form a first set of entries to the feature vector. In order to bring in information about the polarity of the intensity changes, we also extract the sum of the absolute values of the responses, |d_x| and |d_y|. Hence, each sub-region has a four-dimensional descriptor vector v for its underlying intensity structure
$v = (\sum d_x, \sum d_y, \sum |d_x|, \sum |d_y|)$. This results in a descriptor vector for all 4×4 sub-regions of length 64. The wavelet responses are invariant to a bias in illumination (offset). Invariance to contrast (a scale factor) is achieved by turning the descriptor into a unit vector.

然后，小波响应d_x和d_y在每个子区域求和，形成特征向量的第一个集合。为带来灰度变化的极性信息，我们还提取了响应的绝对值的和，|d_x|和|d_y|。因此，每个子区域都有一个四维描述子向量v，代表灰度结构$v = (\sum d_x, \sum d_y, \sum |d_x|, \sum |d_y|)$。这最后对所有4x4的子区域得到了一个描述子向量，长度为64。小波响应对光照变化是不变的。将描述子变成一个单位向量，就可以获得了对对比度的不变性。

Fig. 3 shows the properties of the descriptor for three distinctively different image intensity patterns within a subregion. One can imagine combinations of such local intensity patterns, resulting in a distinctive descriptor.

图3给出了在三种完全不同的图像灰度模式子区域中的描述子性质。可以想象这种局部灰度模式的组合，会得到很有区分性的描述子。

In order to arrive at these SURF descriptors, we experimented with fewer and more wavelet features, using $d^2_x$ and $d^2_y$, higher-order wavelets, PCA, median values, average values, etc. From a thorough evaluation, the proposed sets turned out to perform best. We then varied the number of sample points and sub-regions. The 4×4 sub-region division solution provided the best results. Considering finer subdivisions appeared to be less robust and would increase matching times too much. On the other hand, the short descriptor with 3 × 3 subregions (SURF-36) performs worse, but allows for very fast matching and is still quite acceptable in comparison to other descriptors in the literature. Fig. 4 shows only a few of these comparison results (SURF-128 will be explained shortly).

为达到这些SURF描述子，我们用更多和更少的小波特征来进行了试验，使用$d^2_x$和$d^2_y$，更高阶小波，PCA，中值，平均值等。从彻底的评估中，提出的集合的表现是最好的。我们然后变化了采样点和子区域的数量。4x4的子区域分割给出了最好的结果。考虑更精细的子分割，似乎会没那么稳健，而且会极大的增加匹配时间。另一方面，3x3子区域的短描述子(SURF-36)表现会更差，但会允许非常快的匹配，与文献中的其他描述子相比，性能还可以接受。图4展示了这里的几个比较结果（SURF-128后面会简要介绍）。

We also tested an alternative version of the SURF descriptor that adds a couple of similar features (SURF-128). It again uses the same sums as before, but now splits these values up further. The sums of d_x and |d_x| are computed separately for d_y < 0 and d_y ≥ 0. Similarly, the sums of d_y and |d_y| are split up according to the sign of d_x, thereby doubling the number of features. The descriptor is more distinctive and not much slower to compute, but slower to match due to its higher dimensionality.

我们还测试了另一个版本的SURF描述子，加上了几个类似的特征(SURF-128)。这个描述子也使用了同样的求和，但现在将这些值分裂的更多。d_x和|d_x|的求和，对d_y < 0和d_y ≥ 0两种情况分别进行。类似的，d_y和|d_y|的求和也根据d_x的符号分别进行，因此特征数量加倍了。描述子现在区分性更强，计算起来也没有慢太多，但由于维度比较高，匹配起来比较慢。

In Figure 4, the parameter choices are compared for the standard ‘Graffiti’ scene, which is the most challenging of all the scenes in the evaluation set of Mikolajczyk [8], as it contains out-of-plane rotation, in-plane rotation as well as brightness changes. The extended descriptor for 4 × 4 subregions (SURF-128) comes out to perform best. Also, SURF performs well and is faster to handle. Both outperform the existing state-of-the-art.

在图4中，参数选择在标准Graffiti场景中进行比较，这是Mikolajczyk [8]的评估集的所有场景中最有挑战性的，因为其包含了平面外旋转，平面内旋转以及亮度变化。对4x4子区域的拓展的描述子(SURF-128)表现最好。同时，SURF表现很好，而且计算快速。这两者都超过了目前最好的结果。

For fast indexing during the matching stage, the sign of the Laplacian (i.e. the trace of the Hessian matrix) for the underlying interest point is included. Typically, the interest points are found at blob-type structures. The sign of the Laplacian distinguishes bright blobs on dark backgrounds from the reverse situation. This feature is available at no extra computational cost, as it was already computed during the detection phase. In the matching stage, we only compare features if they have the same type of contrast. Hence, this minimal information allows for faster matching and gives a slight increase in performance.

在匹配阶段，为了快速索引，潜在兴趣点的Laplacian的符号（即，Hessian矩阵的迹）包含在内。典型的，兴趣点在blob类结构处找到。Laplacian的符号区分黑背景上的亮blobs，和相反的情况。这种特征不需要额外的计算就可以得到，因为在检测阶段就已经计算了。在匹配阶段，我们只比较有相同类型对比度的特征。因此，这种最小信息可以进行快速匹配，性能会略有提升。

## 5. Experimental Results

First, we present results on a standard evaluation set, for both the detector and the descriptor. Next, we discuss results obtained in a real-life object recognition application. All detectors and descriptors in the comparison are based on the original implementations of authors.

第一，我们在标准评估集上给出结果，检测器和描述子都是。下一步，我们讨论在真实生活目标检测应用中得到的结果。所有比较的检测器和描述子都是基于作者的原始实现的。

**Standard Evaluation**. We tested our detector and descriptor using the image sequences and testing software provided by Mikolajczyk. These are images of real textured and structured scenes. Due to space limitations, we cannot show the results on all sequences. For the detector comparison, we selected the two viewpoint changes (Graffiti and Wall), one zoom and rotation (Boat) and lighting changes (Leuven) (see Fig. 6, discussed below). The descriptor evaluations are shown for all sequences except the Bark sequence (see Fig. 4 and 7).

**标准评估**。我们使用Mikolajczyk提供的图像序列和测试软件，来测试我们的检测器和描述子。这都是真实的纹理场景和结构化场景的图像。由于空间限制，我们不能给出所有序列的结果。对于检测器比较，我们选择两个视角变化(Graffiti and Wall)，一个放大和旋转(Boat)，和光照变化(Leuven)（见图6，下面进行讨论）。描述子评估对所有序列都进行了展示，除了Bark（见图4和7）。

For the detectors, we use the repeatability score, as described in [9]. This indicates how many of the detected interest points are found in both images, relative to the lowest total number of interest points found (where only the part of the image that is visible in both images is taken into account).

对于检测器，我们使用的是重复性分数，如[9]中描述。这指示的是在两幅图像中找到了多少检测的兴趣点，相对于发现的最低数量的兴趣点（其中两幅图中只有部分图像是可见的，这种情况已经进行了考虑）。

The detector is compared to the difference of Gaussian (DoG) detector by Lowe [2], and the Harris- and Hessian-Laplace detectors proposed by Mikolajczyk [15]. The number of interest points found is on average very similar for all detectors. This holds for all images, including those from the database used in the object recognition experiment, see Table 1 for an example. As can be seen our ’Fast-Hessian’ detector is more than 3 times faster that DoG and 5 times faster than Hessian-Laplace. At the same time, the repeatability for our detector is comparable (Graffiti, Leuven, Boats) or even better (Wall) than for the competitors. Note that the sequences Graffiti and Wall contain out-of-plane rotation, resulting in affine deformations, while the detectors in the comparison are only rotation- and scale invariant. Hence, these deformations have to be tackled by the overall robustness of the features.

检测器与Lowe[2]的DoG检测器，Mikolajczyk提出的Harris-和Hessian-Laplace检测器进行了比较。对所有检测器来说，发现的兴趣点数量平均起来都非常类似。这对于所有图像都是正确的，包括在目标识别试验中使用的图像，见图1的例子。可以看到，我们的Fast-Hessian检测器比DoG快3倍，比Hessian-Laplace快5倍。同时，我们检测器的重复性与竞争者相比是类似的，甚至更好。注意Graffiti和Wall序列包含平面外旋转，得到了仿射形变，而比较的检测器只是旋转不变和尺度不变的。因此，这些形变需要由特征的整体稳健性来处理。

The descriptors are evaluated using recall-(1-precision) graphs, as in [4] and [8]. For each evaluation, we used the first and the fourth image of the sequence, except for the Graffiti (image 1 and 3) and the Wall scene (image 1 and 5), corresponding to a viewpoint change of 30 and 50 degrees, respectively. In figures 4 and 7, we compared our SURF descriptor to GLOH, SIFT and PCA-SIFT, based on interest points detected with our ’Fast-Hessian’ detector. SURF outperformed the other descriptors for almost all the comparisons. In Fig. 4, we compared the results using two different matching techniques, one based on the similarity threshold and one based on the nearest neighbour ratio (see [8] for a discussion on these techniques). This has an effect on the ranking of the descriptors, yet SURF performed best in both cases. Due to space limitations, only results on similarity threshold based matching are shown in Fig. 7, as this technique is better suited to represent the distribution of the descriptor in its feature space [8] and it is in more general use.

描述子使用recall-(1-precision)图来进行评估，和[4, 8]中一样。对每个评估，我们都使用序列中的第一和第四幅图像，除了Graffiti（图1，3）和Wall（图1，5），分别对应了视角变化了30度和50度。在图4和图7中，我们将SURF描述子与GLOH，SIFT和PCA-SIFT进行了比较，基于用我们的Fast-Hessian检测器检测得到的兴趣点。SURF在几乎所有比较中，都超过了其他描述子。在图4中，我们比较了使用两种不同匹配技术的结果，一种是基于相似性阈值，一种是基于最近邻率。着对描述子的排名有效果，但SURF在两种情况下都表现最好。由于空间限制，在图7中只给出了相似性阈值的匹配结果，因为这种技术更适用于表示描述子在其特征空间中的分布，其使用更加通用。

The SURF descriptor outperforms the other descriptors in a systematic and significant way, with sometimes more than 10% improvement in recall for the same level of precision. At the same time, it is fast to compute (see Table 2). The accurate version (SURF-128), presented in section 4, showed slightly better results than the regular SURF, but is slower to match and therefore less interesting for speed-dependent applications.

SURF描述子系统的、显著的超过了其他描述子，对于同样水平的精度，有时候召回率提升超过10%。同时，其计算非常快速（见表2）。准确版(SURF-128)比常规SURF结果略好，但匹配更慢，因此对于速度要求很高的应用来说，效果没那么吸引人。

Note that throughout the paper, including the object recognition experiment, we always use the same set of parameters and thresholds (see table 1). The timings were evaluated on a standard Linux PC (Pentium IV, 3GHz).

注意，贯穿全文，包括目标识别试验，我们一直使用同样的参数喝阈值集合（见表1）。计时是在标准Linux PC上进行的(Pentium IV, 3GHz)。

**Object Recognition**. We also tested the new features on a practical application, aimed at recognising objects of art in a museum. The database consists of 216 images of 22 objects. The images of the test set (116 images) were taken under various conditions, including extreme lighting changes, objects in reflecting glass cabinets, viewpoint changes, zoom, different camera qualities, etc. Moreover, the images are small (320 × 240) and therefore more challenging for object recognition, as many details get lost.

**目标识别**。我们还在一个实际应用中测试了新的特征，目标是在一个博物馆中识别艺术目标。数据集包含216幅图像，22个目标。测试集(116幅图像)是在各种条件下拍摄的，包括极端的光照变化，在反光的镜面柜子中的目标，视角变化，放大，不同的相机质量，等。而且，图像很小(320 × 240)，所以对于目标识别更加有挑战，因为很多细节都丢失了。

In order to recognise the objects from the database, we proceed as follows. The images in the test set are compared to all images in the reference set by matching their respective interest points. The object shown on the reference image with the highest number of matches with respect to the test image is chosen as the recognised object.

为从数据集中识别目标，我们按下面步骤进行。测试集中的图像，与参考集中的所有图像都进行比较，匹配其兴趣点。在参考图像中展示的目标，与测试图像中匹配得到的最高数量，选为识别得到的目标。

The matching is carried out as follows. An interest point in the test image is compared to an interest point in the reference image by calculating the Euclidean distance between their descriptor vectors. A matching pair is detected, if its distance is closer than 0.7 times the distance of the second nearest neighbour. This is the nearest neighbour ratio matching strategy [18, 2, 7]. Obviously, additional geometric constraints reduce the impact of false positive matches, yet this can be done on top of any matcher. For comparing reasons, this does not make sense, as these may hide shortcomings of the basic schemes. The average recognition rates reflect the results of our performance evaluation. The leader is SURF-128 with 85.7% recognition rate, followed by U-SURF (83.8%) and SURF (82.6%). The other descriptors achieve 78.3% (GLOH), 78.1% (SIFT) and 72.3% (PCA-SIFT).

匹配按照下面步骤进行。测试图像中的一个兴趣点，与参考图像中的一个兴趣点进行比较，计算其描述子向量的欧式距离。如果其距离比第二最近邻的距离的0.7倍要小，那么就得到了一个匹配的对。这是最近邻率匹配策略。很明显，额外的几何约束降低了假阳性匹配的影响，但这可以在任意匹配器上进行。出于比较的原因，这没什么意义，因为这可能会隐藏基础方案的缺陷。平均识别率反应了性能评估的结果。第一名是SURF-128，识别率为85.7%，然后是U-SURF (83.8%)和SURF (82.6%)。其他的描述子性能为78.3% (GLOH), 78.1% (SIFT) 和 72.3% (PCA-SIFT)。

## 6. Conclusion

We have presented a fast and performant interest point detection-description scheme which outperforms the current state-of-the art, both in speed and accuracy. The descriptor is easily extendable for the description of affine invariant regions. Future work will aim at optimising the code for additional speed up. A binary of the latest version is available on the internet.

我们提出了一种快速性能号的兴趣点检测-描述方案，速度和准确率超过了目前最好的结果。描述子很容易拓展到仿射不变区域的描述。未来的工作目标在于优化代码，进一步加速。最新版的可执行稳健已经在网上可用。