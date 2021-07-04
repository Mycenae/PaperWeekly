# KAZE Features

Pablo F. Alcantarilla, Adrien Bartoli, and Andrew J. Davison

Universit´e d’Auvergne, Imperial College London

## 0. Abstract

In this paper, we introduce KAZE features, a novel multiscale 2D feature detection and description algorithm in nonlinear scale spaces. Previous approaches detect and describe features at different scale levels by building or approximating the Gaussian scale space of an image. However, Gaussian blurring does not respect the natural boundaries of objects and smoothes to the same degree both details and noise, reducing localization accuracy and distinctiveness. In contrast, we detect and describe 2D features in a nonlinear scale space by means of nonlinear diffusion filtering. In this way, we can make blurring locally adaptive to the image data, reducing noise but retaining object boundaries, obtaining superior localization accuracy and distinctiviness. The nonlinear scale space is built using efficient Additive Operator Splitting (AOS) techniques and variable conductance diffusion. We present an extensive evaluation on benchmark datasets and a practical matching application on deformable surfaces. Even though our features are somewhat more expensive to compute than SURF due to the construction of the nonlinear scale space, but comparable to SIFT, our results reveal a step forward in performance both in detection and description against previous state-of-the-art methods.

本文中，我们提出了KAZE特征，一种新的再非线性尺度空间的多尺度2D特征检测和描述算法。之前的方法在不同的尺度层次中检测和描述特征，是通过构建或近似一幅图像的高斯尺度空间。但是，高斯模糊并不尊重目标的自然边界，对细节和噪声进行同样程度的平滑，降低了定位准确率和区分性。比较起来，我们检测和描述2D特征是在非线性尺度空间，通过非线性扩散滤波。以这种方式，我们可以使模糊在局部对图像数据自适应，降低噪声但是保持图像的边缘，得到更好的定位准确率和区分性。非线性尺度空间的构建是通过高效的加性算子分裂(Additive Operator Splitting, AOS)技术和可变的电导扩散。我们在基准测试数据集上给出了广泛的评估，在形变的表面上给出了一个实际的匹配应用。虽然我们的特征比SURF的计算要更复杂，因为有非线性尺度空间的构建，但与SIFT相比，我们的结果与之前最好的方法相比，在检测和描述上，性能都有提升。

## 1. Introduction

Multiscale image processing is a very important tool in computer vision applications. We can abstract an image by automatically detecting features of interest at different scale levels. For each of the detected features an invariant local description of the image can be obtained. These multiscale feature algorithms are a key component in modern computer vision frameworks, such as scene understanding [1], visual categorization [2] and large scale 3D Structure from Motion (SfM) [3].

多尺度图像处理是计算机视觉应用中一个非常重要的工具。我们可以通过再不同的尺度级别自动检测感兴趣特征，来对一幅图像进行抽象。对每个检测到的特征，都可以得到图像的一个不变的局部描述。这些多尺度特征算法是现在计算机视觉框架中的一个关键组成部分，比如场景理解，视觉分类和大规模3D SfM。

The main idea of multiscale methods is quite simple: Create the scale space of an image by filtering the original image with an appropriate function over increasing time or scale. In the case of the Gaussian scale space, this is done by convolving the original image with a Gaussian kernel of increasing standard deviation. For larger kernel values we obtain simpler image representations. With a multiscale image representation, we can detect and describe image features at different scale levels or resolutions. Several authors [4, 5] have shown that under some general assumptions, the Gaussian kernel and its set of partial derivatives are possible smoothing kernels for scale space analysis. However, it is important to note here that the Gaussian scale space is just one instance of linear diffusion, since other linear scale spaces are also possible [6].

多尺度方法的主要思想是非常简单的：对原始图像用合适的函数在增加的时间或尺度上进行滤波，创建图像的尺度空间。在高斯尺度空间的情况中，这是通过将原始图像与增加标准差的高斯核进行卷积得到的。对更大的核的值，我们得到更简单的图像表示。有了多尺度的图像表示，我们可以在多个尺度级别或分辨率上检测和描述图像特征。几个作者[4,5]证明了，在一些通用假设的情况下，高斯核和其偏导数的集合，是尺度空间分析的可能的平滑核。但是，必须要指出，高斯尺度空间只是线性扩散的一个例子，因为其他线性尺度空间也是可能的[6]。

The Gaussian kernel is probably the simplest option (but not the only one) to build a scale space representation of an image. However, it has some important drawbacks. In Gaussian scale space, the advantages of selecting coarser scales are the reduction of noise and the emphasis of more prominent structure. The price to pay for this is a reduction in localization accuracy. The reason for this is the fact that Gaussian blurring does not respect the natural boundaries of objects and smoothes to the same degree both details and noise at all scale levels. This loss in localization increases as long as we detect features at coarser scale levels, where the amount of Gaussian blurring is higher.

高斯核可能是构建图像尺度表示的最简单的选项（但并不是唯一的）。但是，它有一些重要的缺陷。在高斯尺度空间中，选择更粗糙的尺度的优势，是降低了噪声，强调更显著的结构。要付出的代价是，降低了定位准确率。原因是，高斯模糊也会把目标的自然边缘模糊掉，在所有尺度层次上对细节和噪声的平滑都是一样的。只要我们在更粗糙的尺度级别上检测特征，高斯滤波的程度就更高，定位的损失也就增加了。

It seems more appropriate to make blurring locally adaptive to the image data so that noise will be blurred, but details or edges will remain unaffected. To achieve this, different nonlinear scale space approaches have been proposed to improve on the Gaussian scale space approach [7, 8]. In general, nonlinear diffusion approaches perform much better than linear ones [9, 10] and impressive results have been obtained in different applications such as image segmentation [11] or denoising [12]. However, to the best of our knowledge, this paper is the first one that exploits nonlinear diffusion filtering in the context of multiscale feature detection and description using efficient schemes. By means of nonlinear diffusion, we can increase repeatability and distinctiviness when detecting and describing an image region at different scale levels through a nonlinear scale space.

让模糊对图像数据做到局部自适应，这似乎是更合理的，这样就会把噪声模糊掉，但是细节或边缘会保持不受影响。为取得这个结果，提出了不同的非线性尺度空间方法，以改进高斯尺度空间方法[7,8]。一般来说，非线性扩散方法比线性方法的表现会好很多，在不同的应用中已经得到了很好的结果，如图像分割[11]或去噪[12]。但是，据我们所知，本文是第一篇将非线性扩散滤波高效的应用到多尺度特征检测和描述中的。通过非线性扩散，我们可以在不同尺度级别检测和描述一个图像区域时，利用非线性尺度空间，增加可重复性和区分性。

Probably one of the reasons why nonlinear diffusion filtering has not been used more often in practical computer vision components such as feature detection and description is the poor efficiency of most of the approaches. These approaches normally consist of the discretization of a function by means of the forward Euler scheme. The Euler scheme requires very small step sizes for convergence, and hence many iterations to reach a desired scale level and high computational cost. Fortunately, Weickert et al. introduced efficient schemes for nonlinear diffusion filtering in [9]. The backbone of these schemes is the use of Additive Operator Splitting (AOS) techniques. By means of AOS schemes we can obtain stable nonlinear scale spaces for any step size in a very efficient way. One of the key issues in AOS schemes is solving a tridiagonal system of linear equations, which can be efficiently done by means of the Thomas algorithm, a special variant of the Gaussian elimination algorithm.

非线性扩散滤波在实际的计算机视觉应用还没有成为应用特别广泛的组成部分，比如特征检测和描述，其中一个原因可能是这类方法计算效率比较低。这些方法一般由一个函数通过前向Euler方案的离散化组成。Euler方案需要很小的步长才能收敛，因此需要很多迭代来达到理想的尺度级别和很高的计算代价。幸运的是，Weickert等提出了非线性扩散滤波的高效计算方法[9]。这些方案的骨干是，使用加性算子分裂(AOS)技术。通过AOS，我们可以高效的在任意步长下得到稳定的非线性尺度空间。AOS方案中的一个关键问题是，求解一个三对角线性方程，这可以由Thomas算法高效进行，这是高斯消除算法的一个特殊变体。

In this paper we propose to perform automatic feature detection and description in nonlinear scale spaces. We describe how to build nonlinear scale spaces using efficient AOS techniques and variable conductance diffusion, and how to obtain features that exhibit high repeatability and distinctiveness under different image transfromations. We evaluate in detail our novel features within standard evaluation frameworks [13, 14] and a practical image matching application using deformable surfaces.

本文中，我们提出在非线性尺度空间中进行自动特征检测和描述。我们描述了怎样利用AOS技术和变化电导扩散来构建非线性尺度空间，怎样得到在不同的图像变换下展现出高重复性和区分性的特征。我们在标准评估框架[13,14]内，详细评估了新特征，并使用形变表面评估了一个实际的图像匹配应用。

Our features are named KAZE, in tribute to Iijima [15], the father of scale space analysis. KAZE is a Japanese word that means wind. In nature wind is defined as the flow of air on a large scale and normally this flow is ruled by nonlinear processes. In this way, we make the analogy with nonlinear diffusion processes in the image domain. The rest of the paper is organized as follows: In Section 2 we describe the related work. Then, we briefly introduce the basis of nonlinear diffusion filtering in Section 3. The KAZE features algorithm is explained in detail in Section 4. Finally, exhaustive experimental results and conclusions are presented in Section 5 and 6 respectively.

我们的特征命名为KAZE，以向尺度空间分析之父Iijima[15]致敬。KAZE是一个日文词，意思是风。在自然中，风定义为大规模的空气流动，正常来说这个流动是受到非线性过程控制的。以这种方式，我们与图像领域中的非线性扩散过程进行类比。本文剩下部分组织如下：第2部分是相关的工作，第3部分是非线性扩散滤波的基础介绍，第4部分详细解释了KAZE特征的算法，最后，在第5和第6部分中分别给出了详尽的试验和结论。

## 2. Related Work

Feature detection and description is a very active field of research in computer vision. Obtaining features that exhibit high repeatability and distinctiveness against different image transformations (e.g. viewpoint, blurring, noise, etc.) is of extreme importance in many different applications. The most popular multiscale feature detection and description algorithms are the Scale Invariant Feature Transform (SIFT) [16] and the Speeded Up Robust Features (SURF) [17].

特征检测和描述是计算机视觉一个非常活跃的研究领域。在很多不同的应用中，得到在不同的图像变换（如，视角，模糊，噪声等）中有很高的重复性和区分性的特征，是非常重要的。最流行的多尺度特征检测和描述算法，是SIFT[16]和SURF[17]。

SIFT features were a milestone in feature detection and image matching and are still widely used in many different fields such as mobile robotics and object recognition. In SIFT, feature locations are obtained as the maxima and minima of the result of a Difference of Gaussians (DoG) operator applied through a Gaussian scale space. For building the scale space, a pyramid of Gaussian blurred versions of the original image is computed. The scale space is composed of different sublevels and octaves. For the set of detected features, a descriptor is built based on the main gradient orientation over a local area of interest of the detected keypoint. Then, a rectangular grid of normally 4 × 4 subregions is defined (according to the main orientation) and a histogram of the gradient orientations weighted by its magnitude is built, yielding a descriptor vector of 128 elements.

SIFT特征是特征检测和图像匹配中的一个里程碑，在很多不同的领域中都广泛的应用，比如移动机器人和目标识别。在SIFT中，将DOG算子在一个高斯尺度空间中应用，其中的最大值和最小值的位置，就是特征位置。为构建尺度空间，要计算原始图像的高斯模糊版的金字塔。尺度空间由不同的子级别和octaves构成。对于检测到的特征的集合，在检测到的关键点的局部感兴趣区域，基于其主要的梯度方向，就可以构建一个描述子。然后，定义一个4x4的子区域的矩形网格（根据主要方向），构建了一个由其幅度加权的梯度方向的直方图，得到了128维的描述子向量。

Inspired by SIFT, Bay et al. proposed the SURF detector and descriptor. SURF features exhibit better results with respect to repeatability, distinctiveness and robustness, but at the same time can be computed much faster thanks to the use of the integral image [18], meaning that Gaussian derivatives at different scale levels can be approximated by simple box filters without computing the whole Gaussian scale space. Similar to SIFT, a rectangular grid of 4 × 4 subregions is defined (according to the main orientation) and a sum of Haar wavelet responses (weighted by a Gaussian centered at the interest keypoint) is computed per region. The final descriptor dimension is normally 64 or 128 in its extended counterpart. In [19], Agrawal and Konolige introduced some improvements over SURF by using center-surround detectors (CenSurE) and the Modified-SURF (M-SURF) descriptor. The M-SURF is a variant of the original SURF descriptor, but handles better descriptor boundary effects and uses a more robust and intelligent two-stage Gaussian weighting scheme.

受SIFT启发，Bay等提出了SURF检测器和描述子。SURF特征在重复性，区分性和稳健性上展现出了更好的结果，但同时，由于使用了积分图像，可以进行非常快的计算，在不同尺度级别上，高斯导数的计算可以由简单的盒形滤波器进行近似，不需要计算整个高斯尺度空间。与SIFT类似的是，定义了一个4x4子区域的矩形网格（根据主要方向），在每个区域中计算了一个Haar小波响应的和（由以兴趣关键点为中心的高斯函数加权）。最终的描述子的维度正常的是64，拓展版是128。在[19]中，Agrawal和Konolige对SURF提出了一些改进，使用中间环绕检测器(CenSurE)和修正SURF(M-SURF)描述子。M-SURF是原始SURF描述子的变体，但可以更好的处理描述子边缘效应，使用了更稳健和智能的二阶段高斯加权方案。

Both of these approaches and the many related algorithms which have followed rely on the use of the Gaussian scale space and sets of Gaussian derivatives as smoothing kernels for scale space analysis. However, to repeat, Gaussian scale space does not respect the natural boundaries of objects and smoothes to the same degree both details and noise at all scale levels. In this paper we will show that by means of nonlinear diffusion filtering it is possible to obtain multiscale features that exhibit much higher repeatability and distinctiveness rates than previous algorithms that are based on the Gaussian scale space. At the cost of a moderate increase in computational cost compared to SURF or CenSurE, our results reveal a big step forward in performance in both feature detection and description.

这两种方法和很多相关的跟进算法，都是依赖于高斯尺度空间，和高斯导数的集合作为平滑核的使用，进行尺度空间分析。但是，我们再重复一下，高斯尺度空间并不会保留目标的自然边缘，对细节和噪声再所有尺度级别上的平滑程度都是一样的。本文中，我们展示了，通过非线性扩散滤波，与基于高斯尺度空间的之前的算法相比，可以得到有更大可重复性和区分性的多尺度特征。其代价是，与SURF或CenSurE相比，计算量有一定程度的增加，我们的结果表明，在特征检测和描述上，性能有较大的进步。

## 3. Nonlinear Diffusion Filtering

Nonlinear diffusion approaches describe the evolution of the luminance of an image through increasing scale levels as the divergence of a certain flow function that controls the diffusion process. These approaches are normally described by nonlinear partial differential equations (PDEs), due to the nonlinear nature of the involved differential equations that diffuse the luminance of the image through the nonlinear scale space. Equation 1 shows the classic nonlinear diffusion formulation:

非线性扩散方法描述的是图像亮度的演化，通过增加尺度级别作为特定流函数的散度，控制着扩散的过程。这些方法一般称之为非线性PDEs，因为涉及到的微分方程的非线性本质，通过非线性尺度空间来扩散图像的亮度。式1展示了经典的非线性扩散方程

$$∂L/∂t = div (c(x, y, t) · ∇L)$$(1)

where div and ∇ are respectively the divergence and gradient operators. Thanks to the introduction of a conductivity function (c) in the diffusion equation, it is possible to make the diffusion adaptive to the local image structure. The function c depends on the local image differential structure, and this function can be either a scalar or a tensor. The time t is the scale parameter, and larger values lead to simpler image representations. In this paper, we will focus on the case of variable conductance diffusion, where the image gradient magnitude controls the diffusion at each scale level.

其中div和∇分别是散度和梯度算子。多亏了电导函数c的引入，可以使扩散对局部图像结构进行自适应。函数c依赖于局部图像差分结构，这个函数可以是一个标量，或一个张量。时间t是尺度参数，更大的值会得到更简单的图像表示。本文中，我们会聚焦在可变电导扩散的情况，其中在每个尺度级别中，图像梯度幅度控制着扩散。

### 3.1 Perona and Malik Diffusion Equation

Nonlinear diffusion filtering was introduced in the computer vision literature in [7]. Perona and Malik proposed to make the function c dependent on the gradient magnitude in order to reduce the diffusion at the location of edges, encouraging smoothing within a region instead of smoothing across boundaries. In this way, the function c is defined as:

计算机视觉中的非线性扩散滤波由[7]提出。Perona和Malik提出，让函数c是梯度幅度的函数，以降低在边缘处的扩散，鼓励区域中的平滑，而不是跨边缘平滑。以这种方式，函数c定义为：

$$c(x, y, t) = g(|∇L_σ (x, y, t)|)$$(2)

where the luminance function $∇L_σ$ is the gradient of a Gaussian smoothed version of the original image L. Perona and Malik described two different formulations for the conductivity function g:

亮度函数$∇L_σ$是高斯平滑版的原始图像L的梯度。Perona和Malik给出了电导率函数g的两种不同的表述

$$g_1 = exp(-|∇L_σ|^2/k^2), g_2 = 1/(1 + |∇L_σ|^2/k^2)$$(3)

where the parameter k is the contrast factor that controls the level of diffusion. The function $g_1$ promotes high-contrast edges, whereas $g_2$ promotes wide regions over smaller ones. Weickert [11] proposed a slightly different diffusion function for rapidly decreasing diffusivities, where smoothing on both sides of an edge is much stronger than smoothing across it. That selective smoothing prefers intraregional smoothing to interregional blurring. This function, which we denote here as $g_3$, is defined as follows:

其中参数k是对比系数，控制扩散的级别。函数$g_1$会保留高对比度边缘，而$g_2$倾向于宽的区域。Weickert[11]提出了一个略微不同的扩散函数，扩散率迅速下降，在边缘的两侧的平滑，比跨边缘的边缘要强的多。这种选择性的平滑倾向于区域内平滑，而不是跨区域平滑。我们将这个函数表示为$g_3$，定义如下：

$$g_3 = \left \{ \begin{matrix} 1 && ,|∇L_σ|^2=0 \\ 1-exp(-3.315/(|∇L_σ|/k)^8) && ,|∇L_σ|^2>0 \end{matrix} \right.$$(4)

The contrast parameter k can be either fixed by hand or automatically by means of some estimation of the image gradient. The contrast factor determines which edges have to be enhanced and which have to be canceled. In this paper we take an empirical value for k as the 70% percentile of the gradient histogram of a smoothed version of the original image. This empirical procedure gives in general good results in our experiments. However, it is possible that for some images a more detailed analysis of the contrast parameter can give better results. Figure 1 depicts the conductivity coefficient $g_1$ in the Perona and Malik equation for different values of the parameter k. In general, for higher k values only larger gradients are taken into account.

对比度参数k可以手动固定，也可以通过估计图像梯度自动设置。对比度因子决定，哪些边缘需要增强，哪些需要取消。本文中，我们通过经验将k设置为，原始图像的平滑版的梯度直方图的70%百分位数。在我们的试验中，这个经验的过程总体上给出了很好的结果。但是，对一些图像来说，对比度参数的更详细分析可能给出更好的结果。图1展示的是在PM方程中，不同的k值得到的电导率系数$g_1$。总体上，对于更高的k值，只有更大的梯度才纳入考虑。

### 3.2 AOS Schemes

There are no analytical solutions for the PDEs involved in nonlinear diffusion filtering. Therefore, one needs to use numerical methods to approximate the differential equations. One possible discretization of the diffusion equation is the so-called linear-implicit or semi-implicit scheme. In a vector-matrix notation and using a similar notation to [9], the discretization of Equation 1 can be expressed as:

对非线性扩散滤波，其PDEs没有解析解。因此，需要使用数值方法来近似微分方程。扩散方程的一种可能的离散表达，是所谓的线性隐式或半隐式方案。在向量-矩阵表达式中，我们采用与[9]类似的表达，式1的离散可以表示为：

$$(L^{i+1} - L^i)/τ = \sum_{l=1}^m A_l(L^i) L^{i+1}$$(5)

where $A_l$ is a matrix that encodes the image conductivities for each dimension. In the semi-implicit scheme, for computing the solution $L_{i+1}$, one needs to solve a linear system of equations. The solution $L_{i+1}$ can be obtained as:

其中$A_l$是一个矩阵，包含对每个维度的图像电导率。在半隐式方案中，对计算解$L_{i+1}$，需要求解一个线性系统方程。解可以通过下式得到：

$$L^{i+1} = (I - τ\sum_{l=1}^m A_l(L^i))^{-1} L^i$$(6)

The semi-implicit scheme is absolutely stable for any step size. In addition, it creates a discrete nonlinear diffusion scale-space for arbitrarily large time steps. In the semi-implicit scheme, it is necessary to solve a linear system of equations, where the system matrix is tridiagonal and diagonally dominant. Such systems can be solved very efficiently by means of the Thomas algorithm, which is a variation of the well-known Gaussian elimination algorithm for tridiagonal systems.

半隐式方案对任意步长大小都是绝对稳定的。另外，对任意大的时间步长，创建了一个离散的非线性扩散尺度空间。在半隐式方案中，必须要求解一个线性系统方程，其中系统矩阵是三对角矩阵，对角占优。这种系统可以通过Thomas算法很高效的求解，这是高斯消除算法对三对角系统的一种变体。

## 4. KAZE Features

In this section, we describe our novel method for feature detection and description in nonlinear scale spaces. Given an input image, we build the nonlinear scale space up to a maximum evolution time using AOS techniques and variable conductance diffusion. Then, we detect 2D features of interest that exhibit a maxima of the scale-normalized determinant of the Hessian response through the nonlinear scale space. Finally, we compute the main orientation of the keypoint and obtain a scale and rotation invariant descriptor considering first order image derivatives. Now, we will describe each of the main steps in our formulation.

在这一节中，我们描述了在非线性尺度空间中进行特征检测和描述的新方法。给定输入图像，我们使用AOS技术和可变电导扩散在最大演化时间内构建非线性尺度空间。然后，我们检测感兴趣的2D特征，在非线性尺度空间中的Hessian响应的尺度归一化行列式是最大值。最后，我们计算关键点的主要方向，得到一个尺度和旋转不变的描述子，考虑一阶图像导数。现在，我们对每个主要步骤进行详述。

### 4.1 Computation of the Nonlinear Scale Space

We take a similar approach as done in SIFT, discretizing the scale space in logarithmic steps arranged in a series of O octaves and S sub-levels. Note that we always work with the original image resolution, without performing any downsampling at each new octave as done in SIFT. The set of octaves and sub-levels are identified by a discrete octave index o and a sub-level one s. The octave and the sub-level indexes are mapped to their corresponding scale σ through the following formula:

我们采用与SIFT类似的方法，将尺度空间以对数步长进行离散化，设置为O个octave和S的子级别。注意，我们一直是用原始的图像分辨率进行计算的，在每个新的octave中并没有进行下采样，而SIFT是进行了下采样的。octaves和子级别的设置，有一个离散octave索引o，和子级别索引s。octave和子级别的索引，通过下面的公式，映射到其对应的尺度σ：

$$σ_i(o, s) = σ_02^{(o+s)/S}, o∈[0 ... O−1], s∈[0 ... S−1], i∈[0 ... N]$$(7)

where $σ_0$ is the base scale level and N is the total number of filtered images. Now, we need to convert the set of discrete scale levels in pixel units $σ_i$ to time units. The reason of this conversion is because nonlinear diffusion filtering is defined in time terms. In the case of the Gaussian scale space, the convolution of an image with a Gaussian of standard deviation σ (in pixels) is equivalent to filtering the image for some time $t = σ^2/2$. We apply this conversion in order to obtain a set of evolution times and transform the scale space $σ_i(o, s)$ to time units by means of the following mapping $σ_i → t_i$:

其中$σ_0$是基础尺度级别，N是滤波图像的总计数量。现在，我们需要将离散的尺度级别集合从以像素为单位的$σ_i$转化到以时间的单位。这种转化的原因是，非线性扩散滤波是以时间的单位定义的。在高斯尺度空间的情况下，图像与一个标准差σ的高斯函数卷积，等价于在时间$t = σ^2/2$内对图像滤波。我们应用了这种转换，为得到一系列演化时间，通过下面的映射$σ_i → t_i$，将尺度空间$σ_i(o, s)$转换到时间单位：

$$t_i = σ_i^2/2, i = \{0 ... N \}$$(8)

It is important to mention here that we use the mapping $σ_i → t_i$ only for obtaining a set of evolution times from which we build the nonlinear scale space. In general, in the nonlinear scale space at each filtered image $t_i$ the resulting image does not correspond with the convolution of the original image with a Gaussian of standard deviation $σ_i$. However, our framework is also compatible with the Gaussian scale space in the sense that we can obtain the equations for the Gaussian scale space by setting the diffusion function g to be equal to 1 (i.e. a constant function). In addition, as long as we evolve through the nonlinear scale space the conductivity function tends to be constant for most of the image pixels except for the strong image edges that correspond to the objects boundaries.

提到下面的事很重要，我们使用映射$σ_i → t_i$只是为了得到演化时间的集合，来构建非线性尺度空间。一般来说，在非线性尺度空间中，每个滤波的图像$t_i$，得到的图像并不对应着原始图像与标准差为$σ_i$的高斯函数的卷积。但是，我们的框架与高斯尺度空间也是兼容的，只要将扩散函数g设为等于1（即，常数函数），就可以得到高斯尺度空间的等式。另外，只要我们通过非线性尺度空间进行演化，电导率函数对于大多数图像像素来说都是常数，除了在很强的图像边缘附近，这对应着目标边缘。

Given an input image, we firstly convolve the image with a Gaussian kernel of standard deviation $σ_0$ to reduce noise and possible image artefacts. From that base image we compute the image gradient histogram and obtain the contrast parameter k in an automatic procedure as described in Section 3.1. Then, given the contrast parameter and the set of evolution times $t_i$, it is straightforward to build the nonlinear scale space in an iterative way using the AOS schemes (which are absolutely stable for any step size) as:

给定输入图像，我们首先将图像与标准差为$σ_0$的高斯核相卷积，以降低噪声和可能的图像伪影。从这个基础图像中，我们计算图像梯度直方图，以3.1节的自动过程得到对比度参数k。然后，给定对比度参数和演化时间集合$t_i$，以迭代的方式用AOS方案构建非线性尺度空间就是很直观的了：

$$L^{i+1} = (I - (t_{i+1} - t_i)·\sum_{l=1}^m A_l(L^i))^{-1} L^i$$(9)

Figure 2 depicts a comparison between the Gaussian scale space and the nonlinear one (using the $g_3$ conductivity function) for several evolution times given the same reference image. As it can be observed, Gaussian blurring smoothes for equal all the structures in the image, whereas in the nonlinear scale space strong image edges remain unaffected.

图2给出了高斯尺度空间和非线性尺度空间（使用$g_3$电导率函数），在给定同样的参考图像下，在几个演化时间下的对比。可以观察到，高斯模糊对图像中所有结构都进行同样的平滑，而在非线性尺度空间中，很强的图像边缘则保持不受影响。

### 4.2 Feature Detection

For detecting points of interest, we compute the response of scale-normalized determinant of the Hessian at multiple scale levels. For multiscale feature detection, the set of differential operators needs to be normalized with respect to scale, since in general the amplitude of spatial derivatives decrease with scale [5]:

为检测感兴趣的点，我们在多个尺度级别上，计算Hessian行列式的尺度归一化响应。对于多尺度特征检测，微分算子的集合需要对尺度进行归一化，因为总体上空间导数的幅度会随着尺度的增加而降低：

$$L_{Hessian} = σ^2 (L_{xx} L_{yy} − L^2_{xy})$$(10)

where ($L_{xx}, L_{yy}$) are the second order horizontal and vertical derivatives respectively, and $L_{xy}$ is the second order cross derivative. Given the set of filtered images from the nonlinear scale space $L^i$, we analyze the detector response at different scale levels $σ_i$. We search for maxima in scale and spatial location. The search for extrema is performed in all the filtered images except i = 0 and i = N. Each extrema is searched over a rectangular window of size $σ_i × σ_i$ on the current i, upper i + 1 and lower i − 1 filtered images. For speeding-up the search for extrema, we firstly check the responses over a window of size 3×3 pixels, in order to discard quickly non-maxima responses. Finally, the position of the keypoint is estimated with sub-pixel accuracy using the method proposed in [20].

其中($L_{xx}, L_{yy}$)分别是二阶水平和垂直导数，而$L_{xy}$是二阶交叉导数。给定非线性尺度空间$L^i$中滤波后图像的集合，我们在不同尺度级别$σ_i$分析检测器响应。我们在尺度和空间位置上搜索极值。极值的搜索是在所有滤波后图像中进行的，除了i = 0和i = N。每个极值都在大小为$σ_i × σ_i$（在目前的i，上面的i+1和下面的i-1滤波后图像中）的矩形窗口中进行搜索。对加速极值的搜索，我们首先检查在3x3大小的窗口上的响应，为迅速抛弃非极值响应。最后，关键点的位置是用[20]中提出的方法以亚像素准确率进行估计的。

The set of first and second order derivatives are approximated by means of 3 × 3 Scharr filters of different derivative step sizes $σ_i$. Second order derivatives are approximated by using consecutive Scharr filters in the desired coordinates of the derivatives. These filters approximate rotation invariance significantly better than other popular filters such as Sobel filters or standard central differences differentiation [21]. Notice here that although we need to compute multiscale derivatives for every pixel, we save computational efforts in the description step, since we re-use the same set of derivatives that are computed in the detection step.

一阶和二阶导数的集合，由不同的微分步长$σ_i$的3x3 Scharr滤波器进行近似。二阶导数的近似，是在适用于导数的坐标系中使用连续Scharr滤波器。这些滤波器近似旋转不变性会比其他流行滤波器会明显更好，比如Sobel滤波器或标准中间差分[21]。注意，虽然我们需要对每个像素计算多尺度导数，我们在描述步骤中节约了计算量，因为我们重用了在检测步骤中同样的导数集合。

### 4.3 Feature Description

**Finding the Dominant Orientation**. For obtaining rotation invariant descriptors, it is necessary to estimate the dominant orientation in a local neighbourhood centered at the keypoint location. Similar to SURF, we find the dominant orientation in a circular area of radius $6σ_i$ with a sampling step of size $σ_i$. For each of the samples in the circular area, first order derivatives $L_x$ and $L_y$ are weighted with a Gaussian centered at the interest point. Then, the derivative responses are represented as points in vector space and the dominant orientation is found by summing the responses within a sliding circle segment covering an angle of π/3. From the longest vector the dominant orientation is obtained.

找到主要方向。为得到旋转不变的描述子，必须估计以关键点位置为中心的局部邻域的主要方向。与SURF类似，我们在一个半径为$6σ_i$的圆形区域中以步长大小为$σ_i$找到主要方向。对圆形区域中的每个样本，一阶导数$L_x$和$L_y$是用以兴趣点为中心的高斯函数进行加权的。然后，导数响应以向量空间的点来表示，主要方向是通过在滑动圆形片段中的响应的和来找到的，这个片段覆盖了π/3的角度的。从最长的向量中，得到主要的方向。

**Building the Descriptor**. We use the M-SURF descriptor adapted to our nonlinear scale space framework. For a detected feature at scale $σ_i$, first order derivatives $L_x$ and $L_y$ of size $σ_i$ are computed over a $24σ_i × 24σ_i$ rectangular grid. This grid is divided into 4×4 subregions of size $9σ_i × 9σ_i$ with an overlap of $2σ_i$. The derivative responses in each subregion are weighted with a Gaussian ($σ_1 = 2.5σ_i$) centered on the subregion center and summed into a descriptor vector $dv = (\sum L_x, \sum L_y, \sum |L_x|, \sum |L_y|)$. Then, each subregion vector is weighted using a Gaussian ($σ_2 = 1.5σ_i$) defined over a mask of 4×4 and centered on the interest keypoint. When considering the dominant orientation of the keypoint, each of the samples in the rectangular grid is rotated according to the dominant orientation. In addition, the derivatives are also computed according to the dominant orientation. Finally, the descriptor vector of length 64 is normalized into a unit vector to achieve invariance to contrast.

构建描述子。我们使用改编的M-SURF描述子，适应我们的非线性尺度空间框架。对于一个在尺度$σ_i$检测到的特征，大小为$σ_i$的一阶导数$L_x$和$L_y$是在$24σ_i × 24σ_i$的矩形网格上计算得到的。这个网络分割成4x4的子区域，大小$9σ_i × 9σ_i$，有$2σ_i$的重叠。在每个子区域的导数响应是用以子区域中央为中心的高斯函数($σ_1 = 2.5σ_i$)进行加权的，并求和成为一个描述子向量$dv = (\sum L_x, \sum L_y, \sum |L_x|, \sum |L_y|)$。然后，每个子区域向量使用一个高斯函数($σ_2 = 1.5σ_i$)进行加权，该高斯函数在一个4x4的掩模上定义，以兴趣点为中心。当考虑关键点的主要方向时，在矩形网格上的每个样本，都是根据主要方向进行旋转的。另外，导数也是根据主要方向来计算的。最后，长度为64的描述子向量进行归一化，成为单位向量，以获得对对比度的不变性。

## 5. Experimental Results and Discussion

In this section, we present extensive experimental results obtained on the standard evaluation set of Mikolajczyk et al. [13, 14] and on a practical image matching application on deformable surfaces. The standard dataset includes several image sets (each sequence generally contains 6 images) with different geometric and photometric transformations such as image blur, lighting, viewpoint, scale changes, zoom, rotation and JPEG compression. In addition, the ground truth homographies are also available for every image transformation with respect to the first image of every sequence.

本节中，我们给出广泛的试验结果，在Mikolajczyk等的标准评估集，和在实际的在形变表面的图像匹配应用中。标准数据集包括几个图像集（每个序列一般包含6幅图像），有着不同的几何和亮度变换，比如图像模糊，光照，视角，尺度变化，缩放，旋转和JPEG压缩。另外，真值homographies对每个对每个序列的第一幅图像的图像变换也都是可用的

We also evaluate the performance of feature detectors and descriptors under image noise transformations. We created a new dataset named Iguazu. This dataset consists of 6 images, where the image transformation is the progressive addition of random Gaussian noise. For each pixel of the transformed images, we add random Gaussian noise with increasing variance considering grey scale value images. The noise variances for each of the images are the following: Image 2 ± N (0, 2.55), Image 3 ± N (0, 12.75), Image 4 ± N (0, 15.00), Image 5 ± N (0, 51.00) and Image 6 ± N (0, 102), considering that the grey value of each pixel in the image ranges from 0 to 255. Figure 3 depicts the Iguazu dataset.

我们还评估了特征检测器和描述子在图像噪声变换下的性能。我们创建了一个新的数据集名为Iguazu。这个数据集包含6幅图像，其中图像变换是逐渐增加随机高斯噪声。变换图像中的每个像素，我们增加随机高斯噪声，考虑灰度尺度值图像，方差逐渐增加。对每个图像的噪声方差如下：图像2 ± N (0, 2.55), 图像3 ± N (0, 12.75), 图像4 ± N (0, 15.00), 图像5 ± N (0, 51.00), 图像6 ± N (0, 102)，考虑图像中的每个像素的灰度值为0到255。图3描述了Iguazu数据集。

We compare KAZE features against SURF, SIFT and CenSurE features. For SURF we use the original closed-source library and for SIFT we use Vedaldi’s implementation [22]. Regarding CenSurE features we use the OpenCV based implementation, which is called STAR detector. After detecting features with the STAR detector, we compute a M-SURF descriptor plus orientation as described in [19]. Therefore, we will denote in this section the STAR method as an approximation of CenSurE feature detector plus the computation of a M-SURF descriptor. We use for all the methods the same number of scales O = 4, and sublevels S = 3 for the SIFT and KAZE cases. The feature detection thresholds of the different methods are set to proper values to detect approximately the same number of features per image.

我们将KAZE特征与SURF，SIFT和CenSurE特征进行了比较。对于SURF，我们使用了原始的闭源库，对于SIFT，我们使用了Vedaldi的实现。对CenSurE特征，我们使用基于OpenCV的实现，称为STAR检测器。在用STAR检测器检测到特征后，我们计算一个M-SURF描述子，以及一个方向，如[19]描述。因此，我们在本节中将STAR方法表示为CenSurE特征检测器和M-SURF描述子计算的一个近似。我们对所有方法使用相同数量的参数，对SIFT和KAZE，尺度O=4，子级别S=3。不同方法的特征检测阈值，都设置为合适的值，在每幅图像中以检测到大约相似数量的特征数量。

### 5.1 KAZE Detector Repeatability

The detector repeatability score between two images as defined in [13], measures the ratio between the corresponding keypoints and the minimum number of keypoints visible in both images. The overlap error is defined as the ratio of the intersection and union of the regions $ϵs = 1 − (A∩H^tBH) / (A∪H^tBH)$, where A and B are the two regions and H is the corresponding homography between the images. When the overlap error between two regions is smaller than 50%, a correspondence is considered.

在两幅图像之间，检测器的重复性分数在[13]中定义，度量了两幅图像中对应的关键点和最小数量的可见关键点的比值。重叠误差定义为，区域交集和并集的比值，$ϵs = 1 − (A∩H^tBH) / (A∪H^tBH)$，其中A和B是两个区域，H是两幅图像对应的homography。当两个区域之间重叠误差小于50%，就得到了一个对应性。

Figure 4 depicts the repeatability scores for some selected sequences from the standard dataset. We show repeatability scores for SURF, SIFT, STAR and KAZE considering the different conductivities (g1, g2, g3) explained in Section 3.1. As it can be observed, the repeatibility score of KAZE features clearly outperforms their competitors by a large margin for all the analyzed sequences. Regarding the Iguazu dataset (Gaussian noise), the repeatability score of the KAZE features is for some images 20% higher than SURF and STAR and 40% higher than SIFT. The reason for this is because nonlinear diffusion filtering smoothes the noise but at the same time keeps the boundaries of the objects, whereas Gaussian blurring smoothes in the same degree details and noise. Comparing the results of the different conductivities, g2 exhibits a slightly higher repeatability. This can be explained by the fact that g2 promotes wide area regions which are more suitable for blob-like features such as the ones detected by the determinant of the Hessian. In contrast g1 and g3 promote high-contrast edges which may be more suitable for corner detection.

图4展示了标准数据集中一些选定的序列的重复性分数。我们展示了SURF，SIFT，STAR和KAZE的重复性分数，包含不同的电导率函数g1,g2,g3。可以观察到，在所有分析的序列中，KAZE特征的重复性分数很明显超过了竞争者很多。对于Iguazu数据集（高斯噪声），KAZE特征的重复性分数在一些图像上比SURF和STAR高了20%，比SIFT高了40%。其原因是因为，非线性扩散滤波平滑了噪声，但是同时保持了目标边缘，而高斯模糊对细节和噪声进行了同样程度的平滑。比较不同电导率的结果，g2的重复性略高一些。因为g2倾向于宽的区域，更适合于blob类的特征，比如Hessian行列式检测到的。比较起来，g1和g3倾向于高对比度边缘，更适合于角点检测。

### 5.2 Evaluation and Comparison of the Overall KAZE Features

We evaluate the joint performance of the detection, description and matching for each of the analyzed methods. Descriptors are evaluated by means of precision-recall graphs as proposed in [14]. This criterion is based on the number of correct matches and the number of false matches obtained for an image pair:

我们对每个分析的方法，评估了检测、描述和匹配的总体性能。描述子的评估是通过[14]提出的precision-recall图。这个原则是基于对一个图像对的，正确匹配的数量，和错误匹配的数量。

$$recall = \frac {#correct matches}{correspondences}, 1-precision = \frac {#false matches}{#all matches}$$(11)

where the number of correct matches and correspondences is determined by the overlap error. For the Bikes, Iguazu, Trees and UBC sequences, we show results for the upright version of the descriptors (no dominant orientation) for all the methods. The upright version of the descriptors is faster to compute and usually exhibits higher performance (compared to its corresponding rotation invariant version) in applications where invariance to rotation is not necessary, such is the case of the mentioned sequences.

其中正确匹配和对应性的数量是由于重叠错误来确定的。对于Bikes, Iguazu, Trees和UBC序列，对所有方法，我们展示竖直版描述子的结果（没有主要的方向）。竖直版的描述子计算起来更快速，在不需要旋转不变性的应用中，通常会有更好的性能（与其旋转不变版相比），提到的序列就是这个情况。

Figure 5 depicts precision-recall graphs considering the nearest neighbor matching strategy. As it can be seen, KAZE features obtain superior results thanks in part due to the much better detector repeatability in most of the sequences. For the Boat and Graffiti sequences SURF and SIFT obtain comparable results to KAZE features. However, the number of found correspondences by KAZE is approximately two times higher than the ones found by SURF and SIFT. Note that in all the analyzed image pairs, except the Boat and Graffiti ones, KAZE features exhibit recall rates sometimes 40% higher than SURF, SIFT and STAR for the same number of detected keypoints.

图5展示了在最近邻匹配策略中的precision-recall图。可以看出，KAZE得到了更好的结果，部分是由于在多数序列中有好的多的检测器重复性。对于Boat和Graffiti序列，SURF和SIFT的结果与KAZE特征结果类似。但是，KAZE找到的对应性的数量，大约是SURF和SIFT找到的两倍。注意，在所有分析的图像对中，除了Boat和Graffiti，KAZE特征的recall率有时候比SURF，SIFT和STAR在相同数量的检测关键点上高出40%。

### 5.3 Image Matching for Deformable Surfaces

Complementary to the extensive evaluation on benchmark datasets, we also show results of image matching in deformable surfaces. In particular, we use the deformable surface detection method described in [23]. This method, based on local surface smoothness, is capable of discarding outliers from a set of putative matches between an image template and a deforming target image. In template-based deformable surface detection and reconstruction [24, 25], is very important to have a high number of good correspondences between the template and the target image to capture more accurately the image deformation.

作为在基准测试数据集上的广泛评估的补充，我们还展示了在形变表面的图像匹配的结果。特别是，我们使用了[23]中的形变表面检测方法。这种方法是基于局部表面平滑性，在一幅图像模板和形变的目标图像之间的推定的匹配集之间，可以抛弃外点。在基于模板的形变表面检测和重建中，在模板和目标图像之间，有很多好的对应性是很重要的，可以更准确的捕获图像形变。

Figure 6(a,b) depicts two frames from the paper dataset [24] where we performed our image matching experiment. We detect features from the first image and then match these features to the extracted features on the second image. Firstly, a set of putative correspondences is obtained by using the nearest neighbor distance ratio (NNDR) strategy as proposed in [16]. This matching strategy takes into account the ratio of distance from the closest neighbor to the distance of the second closest. Then, we use the set of putative matches between the two images (that contains outliers) as the input for the outlier rejection method described in [23]. By varying the distance ratio, we can obtain a graph showing the number of inliers for different values of the distance ratio. Figure 6(c) depicts the number of inliers graphs obtained with SURF, SIFT, STAR and KAZE features for the analyzed experiment. According to the results, we can observe that KAZE features exhibit also good performance for image matching applications in deformable surfaces, yielding a higher number of inliers than their competitors.

图6(a,b)展示了paper数据集中的两帧，我们在此上进行图像匹配试验。我们从第一幅图像中检测的特征，然后将这些特征与在第二幅图像中提取到的特征进行匹配。首先，通过使用[16]中提出的最近邻距离率(NNDR)得到一组认定的对应性。匹配策略考虑了最接近的第二接近的距离率。然后，我们使用两幅图像推定匹配的集合（包含外点）作为输入，进行[23]的外点拒绝。通过变化距离率，我们得到了对于不同的距离率值的内点数量。图6c展示了对于分析的分析，用SURF，SIFT，STAR和KAZE方法得到的内点的数量图。根据结果，我们可以观察到，KAZE特征在形变表面的图像匹配应用中，也得到了好的性能，比其他方法得到了更多数量的内点。

### 5.4 Timing Evaluation

In this section we perform a timing evaluation for the most important operations in the process of computing KAZE features with conductivity function g2 and a comparison with respect to SURF, SIFT and STAR. We take into account both the detection and the description of the features (computing a descriptor and dominant orientation or few of them in the case of SIFT). All timing results were obtained on a Core 2 Duo 2.4GHz laptop computer. Our KAZE code is implemented in C++ based on OpenCV data structures. The source code and the Iguazu dataset can be downloaded from https://www.robesafe.com/personal/pablo.alcantarilla/kaze.html.

本节我们对用g2电导率函数计算KAZE特征的过程中，多数重要运算的计时评估，并于SURF，SIFT和STAR进行了比较。我们考虑了特征的检测和描述（计算一个描述子和主要方向，或在SIFT情况下的几个）。所有计时结果都是在Core 2 Duo 2.4GHz的笔记本电脑上进行的。我们的KAZE代码是用C++实现的，基于OpenCV数据结构。源码和Iguazu数据集都已经开源。

In particular, Table 1 shows timing results in seconds for two images of different resolution from the standard dataset. As it can be observed, KAZE features are computationally more expensive than SURF or STAR, but comparable to SIFT. This is mainly due to the computation of the nonlinear scale space, which is the most consuming step in our method. However, at the cost of a slight increase in computational cost, our results reveal a big step forward in performance. In our implementation, we parallelized the AOS schemes computation for each image dimension, since AOS schemes split the whole diffusion filtering in a sequence of 1D separable processes. Nevertheless, our method and implementation are subject to many improvements that can speed-up the computation of the KAZE features tremendously.

特别的，表1给出了以秒为单位在标准数据集中两幅不同分辨率图像上的计时结果。可以观察到，KAZE特征在计算上比SURF或STAR更加复杂，但于SIFT类似。这主要是因为非线性尺度空间的计算，这是我们方法中最耗时的步骤。但是，计算量略微增加，我们的结果在性能上有了很大进步。在我们的实现中，我们对每个图像维度上的AOS方案计算进行了并行化，因为AOS将整个扩散滤波分裂成了一系列1D可分离过程。尽管如此，我们的方法和实现会受到很多改进影响，可以极大的加速KAZE特征计算。

## 6. Conclusions and Future Work

In this paper, we have presented KAZE features, a novel method for multiscale 2D feature detection and description in nonlinear scale spaces. In contrast to previous approaches that rely on the Gaussian scale space, our method is based on nonlinear scale spaces using efficient AOS techniques and variable conductance diffusion. Despite of moderate increase in computational cost, our results reveal a step forward in performance both in detection and description against previous state-of-the-art methods such as SURF, SIFT or CenSurE.

本文中，我们提出了KAZE特征，一种在非线性尺度空间中进行多尺度2D特征检测和描述的新方法。与之前的依赖于高斯尺度空间的方法相比，我们的方法基于非线性尺度空间，使用了高效AOS技术和可变电导率扩散。尽管计算量略有增加，我们的结果在性能上与之前最好的方法有很大进步，如SURF，SIFT或CenSurE，包括检测和描述。

In the next future we are interested in going deeper in nonlinear diffusion filtering and its applications for feature detection and description. In particular, we think that higher quality nonlinear diffusion filtering such as coherence-enhancing diffusion filtering [21] can improve our current approach substantially. In addition, we will work in the direction of speeding-up the method by simplifying the nonlinear diffusion process and by using GPGPU programming for real-time performance. Furthermore, we are also interested in using KAZE features for large-scale object recognition and deformable 3D reconstruction. Despite a tremendous amount of progress that has been made in the last few years in invariant feature matching, the final word has by no means been written yet, and we think nonlinear diffusion has many things to say.

未来我们会更加深入非线性扩散滤波，及其进行特征检测和描述的应用。特别是，我们认为更高质量的非线性扩散滤波，比如coherence增强扩散滤波[21]，可以显著改进我们的方法。另外，我们会加速这个方法，简化非线性扩散过程，使用GPGPU得到实时的性能。而且，我们还会使用KAZE特征进行大规模目标识别，以及形变3D重建。尽管在不变特征匹配上有了非常多的工作，但仍然没有最终结论，我们认为非线性扩散会有很大的帮助。