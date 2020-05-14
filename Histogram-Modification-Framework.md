# A Histogram Modification Framework and Its Application for Image Contrast Enhancement

Tarik Arici et al. Georgia Institute of Technology, Atlanta

## 0. Abstract

A general framework based on histogram equalization for image contrast enhancement is presented. In this framework, contrast enhancement is posed as an optimization problem that minimizes a cost function. Histogram equalization is an effective technique for contrast enhancement. However, a conventional histogram equalization (HE) usually results in excessive contrast enhancement, which in turn gives the processed image an unnatural look and creates visual artifacts. By introducing specifically designed penalty terms, the level of contrast enhancement can be adjusted; noise robustness, white/black stretching and mean-brightness preservation may easily be incorporated into the optimization. Analytic solutions for some of the important criteria are presented. Finally, a low-complexity algorithm for contrast enhancement is presented, and its performance is demonstrated against a recently proposed method.

提出了一种基于直方图均衡化进行图像对比度增强的一般性框架。在这个框架中，对比度增强是视作一个优化问题，对一个代价函数进行最小化。直方图均衡化是对比度增强的一种有效技术。但是，传统的直方图均衡化(HE)通常会得到过度的对比度增强，被处理的图像其外观会很不自然，产生视觉伪影。我们引入了特别设计的惩罚项，对比度增强的程度可以进行调节；对噪声的稳健性，白/黑延展和平均亮度保持，可以很容易的纳入到优化过程来。我们给出了对于一些重要标准的解析解。最后，给出了对比度增强的一个低复杂度的算法，其性能与最近提出的算法进行了对比。

Index Terms — Histogram equalization, histogram modification, image/video quality enhancement.

## 1. Introduction

Contrast enhancement plays a crucial role in image processing applications, such as digital photography, medical image analysis, remote sensing, LCD display processing, and scientific visualization. There are several reasons for an image/video to have poor contrast: the poor quality of the used imaging device, lack of expertise of the operator, and the adverse external conditions at the time of acquisition. These effects result in under-utilization of the offered dynamic range. As a result, such images and videos may not reveal all the details in the captured scene, and may have a washed-out and unnatural look. Contrast enhancement targets to eliminate these problems, thereby to obtain a more visually-pleasing or informative image or both. Typical viewers describe the enhanced images as if a curtain of fog has been removed from the picture [1].

对比度增强在图像处理应用中扮演着一个关键的角色，比如数字成像，医学图像分析，遥感，LCD显示处理，和科学可视化。图像/视频的对比度较差，可能有以下几个原因：使用的成像设备质量较差，操作人员的专业性不够，在获取图像的时间点上外部条件不利。这些因素下得到的图像，其动态范围利用率是不行的。结果是，这种图像和视频可能不会显示捕获场景的所有细节，可能会有一种褪色的、不自然的外观。对比度增强的目标就是消除这些问题，得到视觉效果更好的或更有信息量的图像。典型的观察者描述增强的图像，好像图像上的一层雾被去除掉了。

Several contrast enhancement techniques have been introduced to improve the contrast of an image. These techniques can be broadly categorized into two groups: direct methods [2], [3] and indirect methods [4], [5]. Direct methods define a contrast measure and try to improve it. Indirect methods, on the other hand, improve the contrast through exploiting the under-utilized regions of the dynamic range without defining a specific contrast term. Most methods in the literature fall into the second group. Indirect methods can further be divided into several sub-groups: i) techniques that decompose an image into high and low frequency signals for manipulation, e.g., homomorphic filtering [6], ii) histogram modification techniques [7]–[17], and iii) transform-based techniques [18]–[22]. Out of these three subgroups, the second subgroup received the most attention due to its straightforward and intuitive implementation qualities.

已经提出了几种对比度增强的技术，来改进图像的对比度。这些技术可以大致分为两组：直接方法[2,3]和间接方法[4,5]。直接法定义了一种对比度度量，试图进行改进。间接方法，在不定义特定对比度项的情况下，通过利用动态范围中没有充分利用的区域，来改进对比度。文献中的多数方法都是第二类的。间接的方法可以进一步分成几个子类：i)将一幅图像分解成高频信号和低频信号进行操作的技术，如，同态滤波[6]；ii)直方图修正技术[7-17]；iii)基于变换的技术[18-22]。在这三个子类之中，第二个子类获得了最多注意力，因为其很直接，其实现质量很很直觉化的。

Contrast enhancement techniques in the second subgroup modify the image through some pixel mapping such that the histogram of the processed image is more spread than that of the original image. Techniques in this subgroup either enhance the contrast globally or locally. If a single mapping derived from the image is used then it is a global method; if the neighborhood of each pixel is used to obtain a local mapping function then it is a local method. Using a single global mapping cannot (specifically) enhance the local contrast [10], [13]. The method presented in this paper is demonstrated as a global contrast enhancement (GCE) method, and can be extended to local contrast enhancement (LCE) using similar approaches.

第二个子类中的对比度增强技术，通过某种像素映射技术对图像进行修正，这样处理过的图像的直方图比原始图像更分散。在这个子类中的技术，可以在全局中或局部增强对比度。如果从图像中推导只得到一个映射进行使用，那么就是全局方法；如果每个像素的邻域用于得到一个局部映射函数，那么就是一个局部方法。使用单个全局映射不能增强局部对比度[10,13]。本文中提出的方法，是一个全局对比度增强(GCE)方法，使用类似的方法，可以拓展为局部对比度增强(LCE)方法。

One of the most popular GCE techniques is histogram equalization (HE). HE is an effective technique to transform a narrow histogram by spreading the gray-level clusters in the histogram [23], [24], and it is adaptive since it is based on the histogram of a given image. However, HE without any modification can result in an excessively enhanced output image for some applications (e.g., display-processing).

一个最流行的GCE方法是直方图均衡化(HE)。HE是一种很有效的技术，通过延展直方图中的灰度团来对一个很窄的直方图进行变换[23,24]，而且由于是基于给定图像的直方图的，所以是一种自适应的方法。但是，原始HE，对于一些应用，会得到过度增强的输出图像（如，显示处理）。

Various methods have been proposed for limiting the level of enhancement, most of which are obtained through modifications on HE. For example, bi-histogram equalization was proposed to reduce the mean brightness change [7]. HE produces images with mean intensity that is approximately in the middle of the dynamic range. To avoid this, two separate histograms from the same image are created and equalized independently. The first is the histogram of intensities that are less than the mean intensity, the second is the histogram of intensities that are greater than the mean intensity. A similar method called equal area dualistic sub-image histogram equalization (DSIHE) was proposed in which the two separate histograms were created using the median intensity instead of the mean intensity [8]. Although they are visually more pleasing than HE, these two techniques cannot adjust the level of enhancement and are not robust to noise, which may become a problem when the histogram has spikes. Also, it should be noted that preserving the brightness does not imply preservation of naturalness. One method to deal with histogram spikes is the histogram low-pass filtering [9]. Another method proposes modifying the “cumulation function” of the histogram to adjust the level of enhancement [10], but both of these methods are still sensitive to problems created by histogram spikes. These two methods apply gaussian blurring in the spatial domain to obtain a low-pass filtered histogram or a modified cumulation function [9], [10]. The image blurring operation alone may still be insufficient for large spikes in the histogram; modifying the cumulation function alone enables adjustment of enhancement but does not directly handle histogram-spike related problems. In addition, both of these methods are LCE methods, which are known to be more computationally complex than GCE methods and they not only highlight details in the image but also enhance noise. One recent method proposed by Wang and Ward [14] suggests modifying the image histogram by weighting and thresholding before histogram equalization. The weighting and thresholding is performed by clamping the original histogram at an upper threshold $P_u$ and at a lower threshold $P_l$, and transforming all the values between the upper and lower thresholds using a normalized power law function with index r>0.

已经提出了各种方法，限制增强的水平，多数都是通过改变HE得到的。比如，双直方图均衡化的提出，就是为了降低平均灰度变化[7]。HE得到的图像，其平均灰度大约在动态范围的中间。为避免这种情况，从同样的图像创建得到两个不同的直方图，并独立进行均衡化。第一个是比平均灰度低的直方图，第二个是比平均灰度要高的直方图。还提出过一个类似的方法，称为同区域二元子图直方图均衡化(DSIHE)，是用中值灰度，而不是平均灰度，创建了两个不同的直方图。虽然这比HE在视觉效果上更好，这两种技术不能调整增强的水平，对于噪声也并不稳健，当直方图有尖状时，就是一个问题。而且，应当指出，保持亮度，并不意味着保持自然性。一种处理直方图尖刺的方法是，直方图低通滤波[9]。另一种方法提出修正直方图的累加函数，以调整增强的水平，但这两种方法仍然对直方图尖刺得到的问题比较敏感。这两种方法在空域中应用高斯模糊，以得到低通滤波的直方图，或修正的累加函数[9,10]。图像模糊运算对于直方图中的大的尖刺，可能仍然是不够的；修正累加函数本身，可以调整增强的程度，但并没有直接处理直方图尖刺相关的问题。另外，这两种方法都是LCE方法，这比GCE方法计算起来更复杂，这些方法不仅会增强细节，还会增强噪声。Wang等[14]最近提出一种方法，通过在直方图均衡之前进行加权并取阈值，修正图像直方图。加权和阈值，是将原始直方图在一个较高阈值$P_u$和较低阈值$P_l$之间clamp进行，将在阈值上下界之间的所有值使用归一化的幂函数进行变换，其索引r>0。

There are also unconventional approaches to the histogram-based contrast enhancement problem [11], [12]. Gray-level grouping (GLG) is such an algorithm that groups histogram bins and then redistributes these groups iteratively [11]. Although GLG can adjust the level of enhancement and is robust to histogram spikes, it is mainly designed for still images. Since gray-level grouping makes hard decisions on grouping histogram bins, and redistributing the bins depends on the grouping, mean brightness intensity in an image sequence can abruptly change in the same scene. This causes flickering, which is one of the most annoying problems in video enhancement. Although a fast version of the algorithm is available, GLG’s computational complexity is high for most applications.

也有非传统的方法，进行基于直方图的对比度增强[11,12]。灰度级分组(GLG)是这样一种算法，将直方图bins分组，然后迭代的对这些组进行重新分配[11]。虽然GLG可以调整增强的水平，对直方图尖刺很稳健，但其主要是设计用于静止图像的，由于灰度级分组对直方图bins分组的决定很难，对这些bins进行重新分配，依赖于分组，一个图像序列中的平均亮度可能在同一场景中突然改变。这导致闪烁，这在视频增强中是最主要的一个问题。虽然这个算法有一个快速版，GLG的计算复杂度对于多数应用来说还是很高的。

Contrast enhancement techniques in the first and third subgroups often use multiscale analysis to decompose the image into different bands and enhance desired global and local frequencies [6], [18]–[22], [25]–[27]. These techniques are computationally complex but enable global and local contrast enhancement at the same time by enhancing the appropriate scales.

在第一和第三类中的对比度增强技术，通常使用多尺度分析，来将图像分解成不同的bands，增强期望的全局和局部频率。这种技术计算起来很复杂，但可以通过增强合适的尺度，来同时进行全局和局部的对比度增强。

The aforementioned contrast enhancement techniques perform well on some images but they can create problems when a sequence of images is enhanced, or when the histogram has spikes, or when a natural looking enhanced image is strictly required. In addition, computational complexity and controllability become an important issue when the goal is to design a contrast enhancement algorithm for consumer products. In summary, our goal in this paper is to obtain a visually pleasing enhancement method that has low-computational complexity and can be easily implemented on FPGAs or ASICs and works well with both video and still images. The contributions of this paper in achieving this goal are:

之前提到的对比度增强技术，在一些图像中效果不错，但在对图像序列进行增强时，或直方图有尖刺时，或严格需要一幅看起来很自然的图像时，可能出现一些问题。另外，当设计一个对比度增强算法用于消费级的产品时，计算复杂度和可控性就成来一个重要的问题。总结起来，本文中我们的目标是得到一个视觉上看起来很不错的增强方法，其计算复杂度要较低，可以很容易的在FPGAs或ASICs上进行实现，对视频和静止图像效果都很好。要实现这个目标，本文的贡献如下：

- to describe the necessary properties of the enhancement mapping T[n], and to obtain T[n] via the solution of a bi-criteria optimization problem; 描述增强映射T[n]的必须性质，并通过一种双标准优化方法得到T[n]；

- to incorporate additional penalty terms into the bi-criteria optimization problem in order to handle noise robustness and black/white stretching; 将额外的惩罚项并入到双标准优化问题中，以处理对噪声的稳健性问题和黑/白延展问题；

- to present a content-adaptive algorithm with low computational complexity. 提出一种对内容自适应的算法，计算复杂度低。

In the next section, contrast enhancement is explained. In Section III, the contrast enhancement using the proposed framework is explained in a progressive manner. Then, the proposed low-complexity method is presented in Section IV. Simulation results and discussions are presented in Section V. Finally, the conclusion is provided in Section VI.

下一节解释了对比度增强。在第III部分，以一种渐进的方式解释了，使用提出的框架的对比度增强。然后，在第IV部分提出了低复杂度方法。在第V部分给出了仿真结果和讨论。最后，在第VI部分给出了结论。

## 2. Contrast enhancement

Histogram-based contrast enhancement techniques utilize the image histogram to obtain a single-indexed mapping T[n] to modify the pixel values. In HE and other histogram-based methods, mapping function is obtained from the histogram or the modified histogram, respectively [23]. HE finds a mapping to obtain an image with a histogram that is as close as possible to a uniform distribution to fully exploit the dynamic range. A histogram, h[n], can be regarded as an un-normalized discrete probability mass function of the pixel intensities. The normalized histogram p[n] of an image gives the approximate probability density function (PDF) of its pixel intensities. Then, the approximate cumulative distribution function (CDF), c[n], is obtained from p[n]. The mapping function is a scaled version of this CDF. HE uses the image histogram to obtain the mapping function; whereas, other histogram-based methods obtain the mapping function via the modified histogram. The mapping function in the discrete form is given as

基于直方图的对比度增强技术，利用了图像直方图，得到一个单索引的映射T[n]，来修改像素值。在HE和其他基于直方图的方法中，分别是从直方图或修正的直方图中获得映射函数[23]。HE找到了一个映射，这个映射得到的图像的直方图，与均匀分布尽可能的接近，以充分利用动态范围。一个直方图h[n]，可以认为是像素灰度的一个未归一化的离散概率质量函数。一幅图像的归一化的直方图p[n]，给出了其像素灰度的近似概率密度分布(PDF)。然后，近似的累积分布函数(CDF)，c[n]，从p[n]可以得到。映射函数是CDF的缩放版本。HE使用图像直方图以得到映射函数；而其他基于直方图的方法，通过修正的直方图获得映射函数。离散形式的映射函数由下式给出

$$T[n] = ⌊(2^B-1)\sum_{j=0}^n p[j]+0.5⌋$$(1)

where B is the number of bits used to represent the pixel values, and n ∈ [0, 2^B-1]. Although the histogram of the processed image will be as uniform as possible, it may not be exactly uniform because of the discrete nature of the pixel intensities.

其中B是用于表示像素值的比特数，n ∈ [0, 2^B-1]。虽然处理过的图像的直方图，会尽可能的均匀，但由于像素灰度的离散特性，不可能是严格均匀的。

It is also possible to enhance the contrast without using the histogram. Black stretching and white stretching are simple but effective techniques used in consumer-grade TV sets [1]. Black stretching makes dark pixels darker, while white stretching makes bright pixels brighter. This produces more natural looking black and white regions; hence, it enhances the contrast of the image. Linear black and white stretching can be achieved by the mapping

不使用直方图，也可能增强对比度。黑延展和白延展是很简单但有效的技术，用于消费者级的电视中。黑延展是的暗的像素更暗，而白延展使亮的像素更亮。这产生的白色和黑色区域外观更加自然；因此，增强了图像的对比度。线性的黑色和白色延展可以通过如下映射得到：

$$T[n] = \left\{ \begin{matrix} n×s_b, & n≤b \\ n×g[n], & b<n<w \\ w+(n-w)×s_w, & w≤n \end{matrix} \right.$$(2)

where b is the maximum gray-level to be stretched to black and w is the minimum gray-level to be stretched to white,      g[n] is any function mapping the intensities in between, and $s_b, s_w$ are black and white stretching factors both of which are less than one.

其中b是要延展到黑色的最大灰度，w是要延展到白色的最小灰度，g[n]是中间的任意函数映射，$s_b, s_w$是黑/白延展因子，两者都小于1。

## 3. Histogram modification

To fully exploit the available dynamic range, HE tries to create a uniformly distributed output histogram by using a cumulated histogram as its mapping function. However, HE often produces overly enhanced unnatural looking images. One problem with HE rises from large backward-difference values of T[n], i.e., T[n]-T[n-1] may be unusually large. To deal with this, the input histogram can be modified without compromising its contrast enhancement potential. The modified histogram can then be accumulated to map input pixels to output pixels, similar to HE.

为完全利用可用的动态范围，HE使用累积直方图作为其映射函数，试图创建一个均匀分布的输出直方图。但是，HE通常生成过度增强的外观不自然的图像，HE的一个问题是，T[n]的反向差值较大，即T[n]-T[n-1]可能非常大。为处理这个问题，输入直方图可以在不损害其对比度增强潜力的情况下进行修正。修正的直方图可以进行累加，以将输入像素映射到输出像素，与HE类似。

It is important to note that when the input distribution is already uniform, the mapping obtained from cumulating the input distribution is T[n]=n, which identically maps input to output. Hence, to lessen the level of enhancement that would be obtained by HE, the input histogram $h_i$ can be altered so that the modified histogram $\tilde h$ is closer to a uniformly distributed histogram u, according to a suitably chosen distance metric.

当输入分布已经是均匀分布时，从累积输入分布得到的映射为T[n]=n，那么就将输入恒等的映射到了输出。因此，为减轻HE得到的增强的水平，输入直方图$h_i$可以进行改变，这样修正的直方图$\tilde h$更接近均匀分布的直方图u，根据合理选择的距离度量。

The modified histogram can be seen as a solution of a bi-criteria optimization problem. The goal is to find a modified histogram $\tilde h$ that is closer to u as desired, but also make the residual $\tilde h - h_i$ small. This modified histogram would then be used to obtain the mapping function via (1). This is a bi-criteria optimization problem, and can be formulated as a weighted sum of the two objectives as

修正的直方图可以视作双规则优化问题的一个解决方案。其目标是找到一个修正的直方图$\tilde h$，尽可能的接近u，但还要使得残差$\tilde h - h_i$比较小。这种修正的直方图，就可以通过(1)用于得到映射函数。这是一个双规则优化问题，可以写成两个目标函数的加权和

$$min||h-h_i||+λ||h-u||$$(3)

where $h,h_i,\tilde h$, and $u∈R^{256×1}$, and λ is a problem parameter. As λ varies over [0,∞), the solution of (3) traces the optimal trade-off curve between the two objectives. HE obtained by λ=0 corresponds to the standard HE, and as λ goes to infinity it converges to preserving the original image. Therefore, various levels of contrast enhancement can be achieved by varying λ.

其中$h,h_i,\tilde h$和$u∈R^{256×1}$，λ是一个问题参数。λ取值范围为[0,∞)，(3)的答案是两个目标函数的最优折中。λ=0对应着标准HE，随着λ趋向于无穷大，收敛于保持原始图像。因此，各种层次的对比度增强都可以通过变化λ来得到。

### 3.1 Adjustable Histogram Equalization

An analytical solution to (3) can be obtained when the squared sum of the Euclidean norm is used, i.e., 当使用欧几里得范数的平方和时，可以得到(3)的解析解

$$\tilde h = argmin_h ||h-h_i||_2^2+λ||h-u||_2^2$$(4)

which results in the quadratic optimization problem 可以得到两次优化问题

$$\tilde h = argmin_h [(h-h_i)^T(h-h_i)+λ(h-u)^T(h-u)]$$(5)

The solution of (5) is 其解为

$$\tilde h = \frac{h_i+λu}{1+λ}=(\frac{1}{1+λ})h_i+(\frac{λ}{1+λ})u$$(6)

The modified histogram $\tilde h$, therefore, turns out to be a weighted average of $h_i$ and u. Simply by changing λ, the level of enhancement can be adjusted instead of the more complex non-linear technique given by Stark [10].

因此修正直方图$\tilde h$是$h_i$和u的加权平均。只要改变λ，增强的程度就可以进行调节，而不需要Stark [10]这样更复杂的非线性技术。

An example image and enhanced images using modified histogram equalization with three different λ values (0, 1, 2) are shown in Fig. 1. When λ is zero, the modified histogram is equal to the input histogram; hence, the standard HE is applied. The resulting image is over-enhanced, with many unnatural details on the door and loss of details on the doorknob. When λ is increased to one, the penalty term comes into play and the enhanced image looks more like the original image. For λ = 2, the level of enhancement is further decreased and the details on the doorknob are mostly preserved. In Fig. 2(a), the mappings for the three λ values are given. As λ increases, the mapping becomes more similar to line. The fixed point observed around gray-level value of 76 is a repelling fixed point. Although the level of enhancement is decreased with increasing λ, the slope of the mapping at the fixed point, $n^*$, is still rather large. The slope at $n^*$ determines how fast the intensities in the enhanced image move away from the fixed point [28]. This may become especially important for images with smooth background in which gray-level differences in neighboring pixels look like noise. An example for this situation is shown in Fig. 9(b) and (c).

图1中给出了使用三种不同的λ值(0,1,2)修正直方图的样本图像和增强图像。当λ为0时，修正的直方图等于输入直方图；因此，使用的就是标准HE。得到的图像是过度增强的，在门上有很多不自然的细节，在门把手上损失了很多细节。当λ为1时，惩罚项开始起作用，增强的图像看起来更像原始图像。当λ为2时，增强的程度进一步降低，门把手上的细节得到了很大的保留。在图2(a)中，给出了三个λ值的映射。当λ增加时，映射逐渐变得与直线更接近。在灰度值76附近观察到的固定点，是一个repelling固定点。虽然随着λ的增大，其增强水平下降了，但在固定点$n^*$处的斜率仍然很大。在$n^*$处的斜率，决定了增强图像中的灰度，从固定点处离开的速度。这对于有着平滑背景的图像，其邻域中的灰度级差异看起来很像噪声。这种情况的一个例子，如图9b,c所示。

Fig. 1. Modified histogram equalization results using (6) for image Door. (a) Original image, (b) enhanced image using (6) with λ=0, (c) enhanced image using (6) with λ=1, (d) enhanced image using (6) with λ=2.

Fig. 2. The mappings and histograms for Fig. 1. (a) Mappings for three different λ values used in Fig. 1, (b) original histogram, modified histogram with λ=2 and the uniform histogram.

The problem of $T[n^*]$ having a large slope arises from spikes in the input histogram. The original histogram given in Fig. 2(b) exhibits spikes and the modified histogram has also spikes at the corresponding intensities. This sensitivity to spikes is observed because $l_2$ norm heavily penalizes large residuals, therefore, is not robust to spikes. One way to deal with histogram spikes is to use $l_1$ norm for the histogram approximation term in the objective while using $l_2$ norm for the penalty term. Hence, the problem in (4) is changed to

$T[n^*]$的斜率很大的问题，是源于输入直方图中的尖刺导致的。图2b中的原始直方图有尖刺，修正的直方图在对应的灰度上还会有尖刺。可以观察到这种对尖刺的敏感，是因为$l_2$范数严重的惩罚了很大的残差，因此，对于尖刺并不稳健。一种处理直方图尖刺的方法，是对目标函数的直方图近似项使用$l_1$范数，而对惩罚项使用$l_2$范数.因此，(4)中的问题变为

$$\tilde h = argmin_h ||h-h_i||_1+λ||h-u||_2^2$$(7)

To transform this mixed norm problem into a constrained quadratic programming problem, the first term can be expressed as a sum of auxiliary variables 为将这种混合范数问题转换成一个受约束的平方规划问题，第一项可以表示为辅助变量的和

$$\tilde h = argmin_h [t^T 1+λ(h-u)^T(h-u)]$$

subject to

$$-t \preceq (h-h_i) \preceq t$$

($\preceq$ symbol denotes vector/componentwise inequality) where $t∈R^{256×1}$ and represents the auxiliary variables, and $1∈R^{256×1}$ is a vector of ones. However, this constrained quadratic programming problem has high computational complexity since there are 512 optimization variables. Hence, this approach will not be pursued and is presented here for completeness.

其中$t∈R^{256×1}$表示辅助变量，$1∈R^{256×1}$是元素为1的向量。但是，这种有约束的平方规划问题计算复杂度很高，因为有512个优化变量。因此，这种方法不会使用，这里提出只是为了完备性。

Another way to deal with the histogram spikes in the input histogram is to use one more penalty term to measure the smoothness of $\tilde h$, which reduces the modified histogram’s sensitivity to spikes.

要处理输入图像直方图尖刺的另一种方法是，使用另一个度量$\tilde h$平滑性的惩罚项，这可以降低修正直方图对尖刺的敏感性。

### 3.2 Histogram Smoothing

To avoid spikes that lead to strong repelling fixed points, a smoothness constraint can be added to the objective. The backward-difference of the histogram, i.e., h[i]-h[i-1], can be used to measure its smoothness. A smooth modified histogram will tend to have less spikes since they are essentially abrupt changes in the histogram.

为避免导致很强的repelling固定点的尖刺，可以对目标函数增加平滑性约束。直方图的反向差值，即h[i]-h[i-1]，可以用于度量其平滑性。平滑修正的直方图其尖刺会更少，因为它们就是直方图中的突然变化

The difference matrix $D∈R^{255×256}$ is bi-diagonal 差值矩阵$D∈R^{255×256}$是双对角的

$$D = \left [ \begin{matrix} -1 & 1 & 0 & \cdots & 0 & 0 & 0 \\ 0 & -1 & 1 & \cdots & 0 & 0 & 0 \\ \vdots & \vdots & \vdots & & \vdots & \vdots & \vdots \\ 0 & 0 & 0 & \cdots & -1 & 1 & 0 \\ 0 & 0 & 0 & \cdots & 0 & -1 & 1 \end{matrix} \right ]$$

with the additional penalty term for smoothness, the optimal trade-off is obtained by 有了平滑的额外的惩罚项，最佳折中由下式得到

$$min ||h-h_i||_2^2 + λ||h-u||_2^2 + γ||Dh||_2^2$$(8)

The solution of this three-criterion problem is 这个三规则问题的解为

$$\tilde h = ((1+λ)I+γD^TD)^{-1} (h_i+γu)$$(9)

While (6) results in a weighted average of $h_i$ and u, (9) further smoothes this weighted average to avoid spikes. The first term in (9), that is, $S^{-1} = ((1+λ)I+γD^TD)^{-1}$ in fact corresponds to a low-pass filtering operation on the averaged histogram. This can be seen by expressing $S = ((1+λ)I+γD^TD)$ explicitly as (10), shown at the bottom of the page, where S is a tridiagonal matrix. Each row of its inverse can be shown to be a zero-phase low-pass filter by using a theorem of Fischer and Usmani [29]. Hence, a penalty term for smoothness corresponds to low-pass filtering the averaged histogram. This shows that the proposed framework provides an explanation for the histogram low-pass filtering approaches investigated in the literature, as in Gauch’s work [9], from a different perspective.

(6)式得到$h_i$和u的加权平均，(9)进一步将这个加权平均进行平滑，以避免尖刺。(9)式中的第一项，即，$S^{-1} = ((1+λ)I+γD^TD)^{-1}$，实际上对应对平均的直方图的一个低通滤波。我们将$S = ((1+λ)I+γD^TD)$写为(10)式，其中S为一个三对角矩阵。其逆的每一行，都是一个零相位的低通滤波器。因此，平滑的惩罚项对应着，对平均直方图的低通滤波。这说明，提出的框架，为直方图低通滤波方法提供了一个解释，这类方法如[9]。

To illustrate the performance of histogram smoothing, the image given in Fig. 3(a), which is captured from a compressed video stream, is enhanced using adjustable histogram equalization with and without histogram smoothing. Fig. 3(b) and (c) adjusts the level of enhancement with $γ=0, λ=1$ and $γ=0, λ=3$, respectively. After enhancement, both exhibit artifacts, which are observed as black grain noise around the text. These artifacts arise from the strong repelling fixed-point in the mapping created by the spikes of the original histogram. The ringing-artifact pixels that have intensities less than the background pixels are mapped to even darker intensities. Histogram smoothing with $γ=1000$ solves this problem as can be seen in Fig. 3(d). The mappings for the corresponding enhanced images are given in Fig. 4. The slope, $\dot T(x)$, at the spike bin gray-level has been successfully reduced with histogram smoothing.

为描述直方图平滑的性能，图3a中的图像使用可调整直方图均衡进行了增强，这是一幅从压缩视频流中得到的图像，分别进行了有直方图平滑和没有直方图平滑的增强。图3b和c分别用$γ=0, λ=1$和$γ=0, λ=3$调整了增强的程度。在增强之后，两者都出现了伪影，即在文字附近的黑色米粒样的噪声。这些伪影是由于映射中很强的repelling定点引起的，而这是原始直方图中的尖刺引起的。Ringing伪影的像素，其灰度比背景像素要低，映射到了更低的灰度值。$γ=1000$的直方图平滑，解决了这个问题，如图3d。对应的增强图像的映射如图4所示。在尖刺bin处的斜率$\dot T(x)$，用直方图平滑成功的降低了。

Although histogram smoothing is successful in avoiding histogram spikes, it has a shortcoming. For a real-time implementation $S^{-1}$ has to be computed for each image as γ needs to be adjusted based on the magnitude of the histogram spikes. Even though there are fast algorithms for inverting tridiagonal matrices that require only O(7n) arithmetic operations [30] as opposed to $O(n^3/3)$, it is still unacceptable because of the application at hand (i.e., LCD display processing). This renders the algorithm not easily implementable on FPGAs. Instead of using (9), a low-pass filtering on the histogram can also be performed. But the number of taps and the transfer function must also be adaptive. Another approach that is less computationally complex is to use a weighted error norm for the approximation error $h-h_i$, which is to be described next.

虽然直方图平滑可以成功的避免直方图尖刺，它也有一个缺点。要进行实时的实现，$S^{-1}$需要对每幅图像都进行计算，因为γ需要根据直方图尖刺的幅度进行调整。即使三对角矩阵的逆有快速算法，只需要O(7n)次代数运算，但对于现有的应用仍然不可接受（即，LCD显示处理）。这说明算法在FPGA上的实现没那么容易。如果不使用(9)，也可以对直方图进行低通滤波。但taps数量和传递函数也需要是自适应的。另一种计算上没那么复杂的方法，是对近似误差$h-h_i$使用加权误差范数，下面进行描述。

Fig. 3. Histogram smoothing results using (9) for image Palermo. (a) Original image, (b) enhanced image using (9) with $γ=0$ and $λ=1$, (c) enhanced image using (9) with $γ=0$ and $λ=3$, (d) enhanced image using (9) with $γ=1000$ and $λ=1$.

Fig. 4. Mappings for the enhanced images given in Fig. 3.

$$S = \left[ \begin{matrix} 2γ+(1+λ) & -2γ & 0 & 0 & \cdots \\ -2γ & 4γ+(1+λ) & -2γ & 0 & \cdots \\ 0 & -2γ & 4γ+(1+λ) & -2γ & \cdots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{matrix} \right]$$(10)

### 3.3. Weighted Histogram Approximation

Histogram spikes occur because of the existence of large number of pixels with exactly the same gray-level values as their neighbors. Histogram spikes cause the forward/backward difference of the mapping at that gray-level to be large. This results in an input-output transformation that maps a narrow range of pixel values to a much wider range of pixel values. Hence, it causes contouring and grainy noise type artifacts in uniform regions. A large number of pixels having exactly the same gray-levels are often due to large smooth areas in the image. Hence, the average local variance of all the pixels with the same gray-level can be used to weight the approximation error, $h-h_i$. Histogram approximation error at the corresponding bin will be weighted with a smaller weight. Therefore, the modified histogram bin will not closely follow the input histogram’s spike bin to minimize the approximation error. The objective function with the weighted approximation error is

直方图尖刺产生的原因是，大量像素与其邻域像素有相同的灰度值。直方图尖刺导致在那个灰度级上，映射的前向/后向差分很大。这导致输入-输出的变换，要将很窄的像素值范围，映射到宽的多的像素值范围。因此，会在均匀区域导致伪轮廓和颗粒状噪声的伪影。大量像素有相同的灰度级，这通常是因为图像中有很大的平滑区域。因此，有相同灰度级的所有像素的平均局部方差，可以用于对近似误差$h-h_i$进行加权。在对应bin的直方图近似误差会用较小的权值进行加权。因此，修正直方图bin不会紧密的遵循输入直方图的尖刺bin来最小化近似误差。带有加权近似误差的目标函数是

$$min(h-h_i)^T W (h-h_i) + λ(h-u)^T(h-u)$$(11)

where $W∈R^{256×256}$ is the diagonal error weight matrix, and W(i,i) measures the average local variance of pixels with gray-level i. The solution of (11) is

其中$W∈R^{256×256}$是对角误差权值矩阵，W(i,i)衡量的是灰度值i的像素的平均局部方差。(11)式的解为

$$\tilde h = (W+λI)^{-1} (Wh_i+λu)$$(12)

This is computationally simpler than (9). Since the first term is a diagonal matrix, taking matrix inverse is avoided, i.e., only simple division operations for the diagonal elements are needed to compute its inverse.

这比(9)式计算上要更简单。由于第一项是一个对角矩阵，对矩阵求逆的运算就可以避免来，即，只需要对对角元素进行简单的除法运算，来求得其逆。

Fig. 5 shows the weighted histogram approximation and histogram smoothing for comparison. The grain-noise-type artifacts around the text are avoided in both methods. The mappings for the two methods is given in Fig. 6. The difference of the mapping corresponding to smooth background pixels has further been reduced. However, the mapping is not as smooth as histogram smoothing since no explicit smoothing is performed on the modified histogram.

图5展示了加权直方图近似和直方图平滑进行比较。文字周围的颗粒状伪影在两种方法中都可以得到避免。两种方法的映射如图6给出。平滑的背景像素处的映射的差值，进一步得到了降低。但是，这个映射并没有直方图平滑那么平滑，因为在修正的直方图上没有显式的平滑。

Fig. 5. Comparison results of histogram smoothing and weighted histogram approximation for image Palermo. (a) Histogram smoothing using (9) with γ=1000 and λ=1, (b) weighted approximation using (12) with λ=1000.

Fig. 6. Mappings for the enhanced images given in Fig. 5.

### 3.4. Black and White Stretching

Black and white (B&W) stretching is one of the oldest image enhancement techniques used in television sets. B&W stretching maps predetermined dark and bright intensities to darker and brighter intensities, respectively. To incorporate B&W stretching into histogram modification, where the gray-level range for B&W stretching is [0,b] and [w,255], respectively, the modified histogram $\tilde h$ must have small bin values for the corresponding gray-level ranges. Since the length of the histogram bins determines the contrast between the mapped intensities, by decreasing the histogram bin length for [0,b] and [w,255], the mapping obtained by accumulating the modified histogram will have a smaller forward/backward difference for these two gray-level ranges.

黑/白延展是最古老的一种图像增强技术，用于电视机当中。B&W延展图预先确定了低的灰度和高的灰度，将其延展为更低的灰度和更高的灰度。为将B&W延展纳入到直方图修正中，其中B&W延展的灰度范围是[0,b]和[w,255]，修正的直方图$\tilde h$需要在对应的灰度范围内有小的bin值。由于直方图bins的长度，确定了映射灰度的对比度，通过降低直方图bin[0,b]和[w,255]的长度，通过累积修正直方图得到的映射，对于这两个灰度级范围，会有一个更小的前向/反向差分。

An additional penalty term for B&W stretching can be added to one of the objective functions presented in previous subsections [e.g., adjustable histogram equalization equation given in (5)]

B&W延展的一种额外的惩罚项，可以加入到前一节的目标函数中（如，(5)式中的可调整直方图均衡化）

$$min (h-h_i)^T(h-h_i)+λ(h-u)^T(h-u)+αh^T I^B h$$(13)

where $I^B$ is a diagonal matrix. $I^B(i,i) = 1$ for $i∈\{ [0,b] ∪ [w,255]\}$, and the remaining diagonal elements are zero. The solution to this minimization problem is

其中$I^B$是一个对角矩阵。对于$i∈\{ [0,b] ∪ [w,255]\}$，有$I^B(i,i) = 1$，其余对角元素为0。这个最小化问题的解为

$$\tilde h = ((1+λ)I+αI^B)^{-1} (h_i+λu)$$(14)

In Fig. 7, histogram smoothing with and without B&W stretching is illustrated. In this experiment, black stretch gray-level range is [0, 20] and white stretch gray-level range is [200, 255] with α set to 5. With the more natural look of the black and white in the image, the contrast has greatly improved. The mapping as given in Fig. 7(d) clearly shows B&W stretching and the smooth transition to nonstretching region.

在图7中，给出了有和没有B&W延展的直方图平滑。在这个试验中，黑色延展灰度的范围是[0,20]，白色延展的灰度级范围是[200,255]，α设置为5。图像中的黑色和白色看起来更自然了，对比度得到了极大的改善。图7d给出的映射，明显表明了B&W延展和平滑迁移到非延展区域。

Fig. 7. Comparison results of histogram smoothing with and without B&W stretching for image Palermo. (a) Original image, (b) enhanced image using (9) with γ=1000 and λ=1, (c) enhanced image using (14) with γ=1000, λ=1 and α=5, (d) mappings for the two enhanced images in (b) and (c).

## 4. Low-complexity Histogram Modification Algorithm

In this section, a low-complexity histogram modification algorithm is presented. The pseudo-code of the algorithm is given in Algorithm 1. It deals with histogram spikes, performs B&W stretching, and adjusts the level of enhancement adaptively so that the dynamic range is better utilized while handling the noise visibility and the natural look requirements. Also, the proposed algorithm does not require any division operation.

本节中，提出了一种低复杂度直方图修正算法。算法的伪代码在算法1中给出。算法会处理直方图尖刺，进行B&W延展，并自适应的调整增强的程度，这样可以更好的利用动态范围，同时处理噪声可见性和外观自然的需求。同时，提出的算法不需要任何除法运算。

Using histogram smoothing or weighted histogram approximation is computationally complex when considering the scarce memory and gate-count/area resources in an hardware implementation. Histogram smoothing requires either solving (9) or explicit low-pass filtering with adaptive filter length and transfer function. On the other hand, weighted approximation with solution given in (12) requires division operation.

使用直方图平滑，或加权直方图近似，计算量是非常大的，尤其是考虑到硬件实现时，内存和门计数/区域是稀缺资源。直方图平滑需要求解(9)，或显式的用自适应滤波器长度和传递函数进行低通滤波。另一方面，加权近似在(12)中给出的解需要除法运算。

### 4.1. Histogram Computation

To deal with histogram spikes in a simple way, instead of smoothing or weighting the input histogram, one can change the way a histogram is computed. Histogram spikes are created because of a large number of pixels that have the same gray-level and these pixels almost always come from smooth areas in the input image when they create artifacts/noise in the enhanced image. Hence, histogram computation can be modified so as to take pixels that have some level of contrast with their neighbors into account, which will solve the histogram spike problem at the very beginning. It is also possible to relate this practical approach with optimization based solutions discussed in the previous section as follows: For a successful contrast enhancement, the histogram should be modified in such a way that the modified histogram, $\tilde h$, represents the conditional probability of a pixel, given that it has a contrast with its neighbors (denoted by C). That is, $\tilde h[i] = p[i|C]$, where $p[i|C]$ denotes the probability of a pixel having gray-level i given the event C. Performing histogram equalization on $\tilde h$ rather than h will enhance the contrast but not the noise, since the former will only utilize the dynamic range for pixels that have some level of contrast with their neighbors. Noting that the histogram modification methods presented in the previous section (e.g., weighting) also aim to increase contrast but not the noise visibility, they must modify the histogram in such a way that the the modified histogram resembles $p[i|C]$ rather than $p[i]$. However, one can simply obtain $p[i|C]$ by counting only those pixels that have contrast, rather than solving complex optimization problems, which in essence corresponds to dealing with histogram spikes resulting from smooth area (noncontrast) pixels after computing the histogram in the conventional way.

为简单的处理直方图尖刺，我们不对输入直方图进行平滑或加权，而可以改变直方图计算的方式。直方图尖刺的生成，是因为很多像素有相同的灰度级，这些像素几乎都是来自输入图像的平滑区域，而在增强的图像中产生伪影/噪声。因此，直方图计算可以进行修改，将与周围邻域有一定的对比度纳入考虑，这就可以在最开始解决直方图尖刺问题。将这种方法与之前讨论的基于优化的方法结合到一起，也是有可能的，如下：对一个成功的对比度增强，其直方图修正的方式应当是，修正的直方图$\tilde h$，表示一个像素的条件概率，只要其与邻域有一个对比度（表示为C）。即，$\tilde h[i] = p[i|C]$，其中$p[i|C]$表示，在给定事件C下，像素灰度值为i的条件概率。对$\tilde h$进行直方图均衡化，而不是对h进行，将会增强对比度，而不增强噪声，因为前者会对一部分像素利用其动态范围，这些像素是与其邻域有一定的对比度的。注意，前节给出的直方图修正方法（如，加权）其目的也是提升对比度，而不是噪声可见性，它们修正直方图的方式，需要是修正的直方图与$p[i|C]$很像，而不是与$p[i]$很像。但是，仅仅靠数那些有对比度的像素就可以得到$p[i|C]$，而不是求解复杂的优化问题，这实质上也对应着处理直方图尖刺的问题。

To obtain the histogram, the local variation of each pixel can be used to decide if a pixel has sufficient contrast with its neighbors. One efficient way of achieving this for hardware simplicity is to use a horizontal variation measure by taking advantage of the row-wise pixel processing architecture, which is available in common video processing hardware platforms. A horizontal one-lagged difference operation is a high-pass filter, which will also measure noise. On the other hand, a horizontal two-lagged difference operation is a band-pass filter which will attenuate high-frequency noise signals. Histogram is created using pixels with a two-lagged difference that has a magnitude larger than a given threshold (steps 5, 6, 7). The number of pixels included in the histogram is also counted for proper normalization.

为得到直方图，每个像素的局部变化可以用于确定，一个像素是否与其邻域有足够的对比度。为硬件实现简单，一种有效的方式是使用一种横向变化度量，这样可以利用逐行的像素处理架构，这在常用的视频处理硬件平台是可用的。一种水平的延迟一个像素的差分运算，是一种高通滤波器，这也会度量噪声。另一方面，水平的差二差分运算，是一种带通滤波，这会使高频噪声信号衰减。直方图是使用差二差分幅度大于给定阈值的像素生成的。直方图中包含的像素也进行了合理的归一化。

### 4.2 Adjusting the Level of Enhancement

As described in Section III-A, it is possible to adjust the level of histogram equalization to achieve natural looking enhanced images. The modified histogram is a weighted average of the input histogram $h_i$ and the uniform histogram u, as given in (6). The contribution of the input histogram in the modified histogram is $κ^* = 1/(1+λ)$. The level of histogram equalization should be adjusted depending on the input image’s contrast. Low contrast images have narrow histograms and with histogram equalization, contouring and noise can be created. Therefore, κ is computed to measure the input contrast using the aggregated outputs of horizontal two-lagged difference operation (step 4). Afterwards, κ is multiplied by a user-controlled parameter g, then gκ is normalized to the range [0, 1] (step 11) to get $κ^*$. It is a good practice to limit the maximum contribution of a histogram, since this will help with the worst-case artifacts created due to histogram equalization. By choosing the maximum value that gκ can take on as a power of two, the normalization step can be done using a bit-shift operation rather than a costly division. To ensure that $h_i$ and u have the same normalization, u is obtained using the number of pixels that are included in the histogram (step 12). $u_{min}$ is used to ensure that very low bin regions of the histogram will not result in very low slope in the mapping function; it will increase the slope in these regions, resulting in increased-utilization of dynamic range.

就像在3.1小节中所述，通过调整直方图均衡化的程度，来得到外观自然的增强图像，这是可能的。修正的直方图是输入直方图$h_i$和均匀直方图的加权平均，如(6)式所述。在修正直方图中，输入直方图的贡献是$κ^* = 1/(1+λ)$。直方图均衡的程度，应当根据输入图像的对比度来进行调整。低对比度图像直方图比较窄，用直方图均衡，会产生伪边缘和噪声。因此，κ的计算是度量输入的对比度，是使用水平差二差分运算的累积输出。然后，κ乘以一个用户控制的参数g，然后gκ归一化到[0, 1]的范围以得到$κ^*$。限制一个直方图的最大贡献，是非常好的，因为这对最差情况下的伪影（直方图均衡化产生）有帮助。选择gκ的最大值为2的幂，归一化步骤可以用bit-shift运算进行，而不需要除法进行。为确保$h_i$和u的归一化相同，u是使用直方图中包含的像素数量得到的。$u_{min}$用于确保直方图中的低bin区域不会在映射函数中有很低的斜率；这会在这些区域中增加斜率，得到动态范围的更大利用率。

B&W stretching is performed using (14) (step 17). Parameters b, w, and α can be adapted with the image content. b and w is usually derived from the histogram as the minimum and maximum intensities. For noise robustness, b should be chosen as the minimum gray-level that is bigger than some predefined number of pixels’ intensities, w can be chosen similarly. It is a good practice to impose limits on b and w. The stretching parameter should also be adapted with image content. For dark images white stretching can be favored, while for bright images black stretching can be favored. α may also depend on the input image’s contrast.

B&W延展采用(14)进行。参数b, w和α可以用图像内容来调整。b和w通常由直方图推导而来，作为最小最大灰度。出于对噪声的稳健性，b应当选择作为最小灰度级，比一些预定义的像素灰度数量更大，w也可以类似的选择。给b和w设置限制，是很好的做法。延展参数应当与图像内容相适应。对于较黑的图像，应当倾向于用白延展，而对较亮的图像，需要用黑延展。α应当依赖于输入图像的对比度。

## 5. Results and Discussion

we will use the following quantitative measures: Absolute Mean Brightness Error (AMBE), the discrete entropy (H), and the measure of enhancement (EME) [3], [16], [18].

### 5.1 Subjective Assessment

1) Gray-Scale Images
2) Color Images

### 5.2 Objective Assessment

### 5.3 Complexity Comparison

### 6. Conclusion

A general framework for image contrast enhancement is presented. A low-complexity algorithm suitable for video display applications is proposed as well. The presented framework employs carefully designed penalty terms to adjust the various aspects of contrast enhancement. Hence, the contrast of the image/video can be improved without introducing visual artifacts that decrease the visual quality of an image and cause it to have an unnatural look.

提出了图像对比度增强的一个一般性框架，还提出了一种低复杂度的算法，适用于视频显示应用。提出的框架采用仔细设计的惩罚项，来调整对比度增强的各个方面。因此，图像/视频的对比度可以得到增强，而且不会引入视觉伪影，降低一幅图像的视觉质量，导致其外观不自然。

To obtain a real-time implementable algorithm, the proposed method avoids cumbersome calculations and memory-bandwidth consuming operations. The experimental results show the effectiveness of the algorithm in comparison to other contrast enhancement algorithms. Obtained images are visually pleasing, artifact free, and natural looking. A desirable feature of the proposed algorithm is that it does not introduce flickering, which is crucial for video applications. This is mainly due to the fact that the proposed method uses the input (conditional) histogram, which does not change significantly within the same scene, as the primary source of information. Then, the proposed method modifies it using linear operations resulting from different cost terms in the objective rather than making algorithmic hard decisions.

为得到可以实时实现的算法，提出的方法避免复杂的计算，和消耗内存带宽的运算。试验结果表明了算法的有效性，与其他对比度增强算法进行了对比。得到的图像视觉效果很好，没有伪影，看起来非常自然。提出的算法的一个很好的性质是，不会引入闪烁，这对于视频应用非常关键。这主要是因为，提出的方法使用了输入（条件）直方图，在同一个场景中不会剧烈变化。然后，提出的方法使用线性运算对其进行修正，这些线性运算是从目标函数的不同的代价项中得到的，而不用进行算法上的艰难选择。

The proposed method is applicable to a wide variety of images and video sequences. It also offers a level of controllability and adaptivity through which different levels of contrast enhancement, from histogram equalization to no contrast enhancement, can be achieved.

提出的方法可用于很多图像和视频序列。算法还提供了一定程度的可控性和自适应性，可以得到不同水平的对比度增强，从直方图均衡到没有对比度增强。