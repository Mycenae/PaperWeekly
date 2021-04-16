# Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns

Timo Ojala et. al. University of Oulu

## 0. Abstract

This paper presents a theoretically very simple, yet efficient, multiresolution approach to gray-scale and rotation invariant texture classification based on local binary patterns and nonparametric discrimination of sample and prototype distributions. The method is based on recognizing that certain local binary patterns, termed “uniform,” are fundamental properties of local image texture and their occurrence histogram is proven to be a very powerful texture feature. We derive a generalized gray-scale and rotation invariant operator presentation that allows for detecting the “uniform” patterns for any quantization of the angular space and for any spatial resolution and presents a method for combining multiple operators for multiresolution analysis. The proposed approach is very robust in terms of gray-scale variations since the operator is, by definition, invariant against any monotonic transformation of the gray scale. Another advantage is computational simplicity as the operator can be realized with a few operations in a small neighborhood and a lookup table. Excellent experimental results obtained in true problems of rotation invariance, where the classifier is trained at one particular rotation angle and tested with samples from other rotation angles, demonstrate that good discrimination can be achieved with the occurrence statistics of simple rotation invariant local binary patterns. These operators characterize the spatial configuration of local image texture and the performance can be further improved by combining them with rotation invariant variance measures that characterize the contrast of local image texture. The joint distributions of these orthogonal measures are shown to be very powerful tools for rotation invariant texture analysis.

本文提出了一种理论上非常简单，但非常高效的多分辨率方法，进行灰度和旋转不变的纹理分类算法，算法基于LBP和对样本和原型分布的非参数区分。本方法是基于识别特定的LBP，我们称为uniform，是局部图像纹理的基础性质，其频数直方图被证明是一种非常强的纹理特征。我们推导出了一种通用的灰度和旋转不变的算子表示，可以对角空间的任意量化和对任意空间分辨率检测uniform模式，提出一种方法结合了多种算子进行多分辨率分析。提出的方法对于灰度变化非常稳健的，因为这个算子在定义上就对灰度级的单调变化是不变的。另一个优势是其计算上非常简单，因为算子可以在很小的邻域中用一个查找表用很少几个计算就可以实现。在实际问题中得到了非常好的试验结果，并有旋转不变性，其中分类器是在一个特定旋转角度下训练的，并用其他旋转角度的样本进行了测试，证明了用简单的旋转不变的LBP的发生统计量，就可以得到很好的分类性能。这些算子描述了局部图像纹理的空间配置特征，与旋转不变的方差度量（这是局部图像纹理的对比度特征）相结合，可以进一步改进性能，这些不相关的度量的联合分布，证明了是旋转不变纹理分析的强有力工具。

**Index Terms** — Nonparametric, texture analysis, Outex, Brodatz, distribution, histogram, contrast.

## 1. Introduction

Analysis of two-dimensional textures has many potential applications, for example, in industrial surface inspection, remote sensing, and biomedical image analysis, but only a limited number of examples of successful exploitation of texture exist. A major problem is that textures in the real world are often not uniform due to variations in orientation, scale, or other visual appearance. The gray-scale invariance is often important due to uneven illumination or great within-class variability. In addition, the degree of computational complexity of most proposed texture measures is too high, as Randen and Husoy [32] concluded in their recent extensive comparative study involving dozens of different spatial filtering methods: “A very useful direction for future research is therefore the development of powerful texture measures that can be extracted and classified with a low-computational complexity.”

二维纹理的分析有很多潜在的应用，比如，工业表面检测，遥感，医学图像分析，但只有几个例子成功的探索了纹理特征。一个主要的问题是，真实世界中的纹理并不是一致的，有方向，尺度或其他视觉外观上的差异。灰度不变性通常是很重要的，因为光照很可能是不均衡的，类内变化也很大。另外，多数提出的纹理度量的计算复杂度都很高，Randen and Husoy [32]在其最近的广泛比较性研究中涉及到了十几种不同的空间滤波方法，得出结论：未来研究的一个非常有用的方向是，研究很强的纹理度量的提取和分类，但计算量要非常低。

Most approaches to texture classification assume, either explicitly or implicitly, that the unknown samples to be classified are identical to the training samples with respect to spatial scale, orientation, and gray-scale properties. However, real-world textures can occur at arbitrary spatial resolutions and rotations and they may be subjected to varying illumination conditions. This has inspired a collection of studies which generally incorporate invariance with respect to one or at most two of the properties spatial scale, orientation, and gray scale.

多数纹理分类方法显式或隐式的假设，要分类的未知样本与训练样本的空间尺度、方向和灰度性质是一样的。但是，真实世界的纹理可以是任意空间分辨率的，任意旋转方向的，而且可能是在不同的光照条件下得到的。这激发了很多研究，一般考虑了一种或两种不变性，包括空间尺度，方向和灰度级。

The first few approaches on rotation invariant texture description include generalized cooccurrence matrices [12], polarograms [11], and texture anisotropy [7]. Quite often an invariant approach has been developed by modifying a successful noninvariant approach such as MRF (Markov Random Field) model or Gabor filtering. Examples of MRF based rotation invariant techniques include the CSAR (circular simultaneous autoregressive) model by Kashyap and Khotanzad [16], the MRSAR (multiresolution simultaneous autoregressive) model by Mao and Jain [23], and the works of Chen and Kundu [6], Cohen et al. [9], and Wu and Wei [37]. In the case of feature-based approaches, such as filtering with Gabor wavelets or other basis functions, rotation invariance is realized by computing rotation invariant features from the filtered images or by converting rotation variant features to rotation invariant features [13], [14], [15], [19], [20], [21], [22], [30], [39]. Using a circular neighbor set, Porter and Canagarajah [31] presented rotation invariant generalizations for all three mainstream paradigms: wavelets, GMRF, and Gabor filtering. Utilizing similar circular neighborhoods, Arof and Deravi obtained rotation invariant features with 1D DFT transformation [2].

有几种方法研究了旋转不变纹理特征，包括通用共现矩阵[12]，polarograms [11], 和texture anisotropy [7]。通常一种不变方法的提出，是对一种成功的非不变方法的修改，如MRF模型或Gabor滤波。基于MRF的旋转不变的技术的例子包括，CSAR (circular simultaneous autoregressive) model by Kashyap and Khotanzad [16], the MRSAR (multiresolution simultaneous autoregressive) model by Mao and Jain [23], and the works of Chen and Kundu [6], Cohen et al. [9], and Wu and Wei [37]。基于特征的方法的情况中，比如用Gabor小波的滤波或其他基函数，旋转不变性的实现，是通过从滤波图像中计算旋转不变特征，或将旋转变化的特征转化到旋转不变的特征。使用了一个圆形的邻域集，Porter and Canagarajah对所有三种主流方案提出了旋转不变的泛化：小波，GMRF和Gabor滤波。利用类似的圆形邻域，Arof and Deravi得到了1D DFT变换的旋转不变特征。

A number of techniques incorporating invariance with respect to both spatial scale and rotation have been presented [1], [9], [20], [22]. [38], [39]. The approach based on Zernike moments by Wang and Healey [36] is one of the first studies to include invariance with respect to all three properties: spatial scale, rotation, and gray scale. In his mid-1990s survey on scale and rotation invariant texture classification, Tan [35] called for more work on perspective projection invariant texture classification, which has received a rather limited amount of attention [5], [8], [17].

提出了几种具有空间尺度和旋转不变性的技术算法。Wang and Healey的基于Zernike矩的方法是第一种具有三种不变性的研究，空间尺度，旋转和灰度级。在他的1990s时对尺度和旋转不变纹理分类综述中，Tan[35]呼吁更多的对视角投影不变的纹理分类工作，这方面的工作比较少。

This work focuses on gray-scale and rotation invariant texture classification, which has been addressed by Chen and Kundu [6] and Wu and Wei [37]. Both studies approached gray-scale invariance by assuming that the gray-scale transformation is a linear function. This is a somewhat strong simplification, which may limit the usefulness of the proposed methods. Chen and Kundu realized gray-scale invariance by global normalization of the input image using histogram equalization. This is not a general solution, however, as global histogram equalization cannot correct intraimage (local) gray-scale variations.

本文关注的是灰度和旋转不变纹理分类，Chen and Kundu [6]和Wu and Wei [37]也处理了这个问题。这两个研究通过假设灰度级变换是一个线性函数，从而达到灰度级不变性。这是一个比较强的简化条件，会限制提出的方法的可用范围。Chen and Kundu通过对输入图像使用直方图均化全局归一化而实现了灰度不变性。但这并不是一个通用解决方案，因为全局直方图均化不能修正图像内的局部灰度级变化。

In this paper, we propose a theoretically and computationally simple approach which is robust in terms of grayscale variations and which is shown to discriminate a large range of rotated textures efficiently. Extending our earlier work [27], [28], [29], we present a gray-scale and rotation invariant texture operator based on local binary patterns. Starting from the joint distribution of gray values of a circularly symmetric neighbor set of pixels in a local neighborhood, we derive an operator that is, by definition, invariant against any monotonic transformation of the gray scale. Rotation invariance is achieved by recognizing that this gray-scale invariant operator incorporates a fixed set of rotation invariant patterns.

本文中，我们提出了一种理论上和计算上都很简单的方法，对灰度变化是非常稳健的，可以高效的区分大量旋转纹理。从早期的工作[27,28,29]进行拓展，我们基于LBP提出了一种灰度级和旋转不变的纹理算子。我们从灰度值在局部邻域中的一个圆形的对称邻域像素集合中的联合分布，推导出了一个算子，从定义上就是对任意灰度级的单调变化是不变的。通过结合了旋转不变模式的固定集合，灰度不变的算子也可以得到旋转不变性。

The main contribution of this work lies in recognizing that certain local binary texture patterns termed “uniform” are fundamental properties of local image texture and in developing a generalized gray-scale and rotation invariant operator for detecting these “uniform” patterns. The term “uniform” refers to the uniform appearance of the local binary pattern, i.e., there are a limited number of transitions or discontinuities in the circular presentation of the pattern. These “uniform” patterns provide a vast majority, sometimes over 90 percent, of the 3x3 texture patterns in examined surface textures. The most frequent “uniform” binary patterns correspond to primitive microfeatures, such as edges, corners, and spots; hence, they can be regarded as feature detectors that are triggered by the best matching pattern.

本文的主要贡献，是发现了一些局部二值纹理模式是局部图像纹理的基本性质，称为uniform，在开发通用灰度和旋转不变的算子是非常基础的，可以检测这些uniform模式。术语uniform指LBP的一致外观，即，模式的圆形表示中的迁移或不连续性数量有限。这些uniform模式在检查的表面纹理中，给出了绝大部分，有时候超过了90%的，3x3纹理模式。最经常出现的uniform二值模式，对应着微特征原语，如边缘，角点，和斑点；因此，它们可以认为是由这些最佳匹配的模式触发的特征检测器。

The proposed texture operator allows for detecting “uniform” local binary patterns at circular neighborhoods of any quantization of the angular space and at any spatial resolution. We derive the operator for a general case based on a circularly symmetric neighbor set of P members on a circle of radius R, denoting the operator as $LBP_{P,R}^{riu2}$. Parameter P controls the quantization of the angular space, whereas R determines the spatial resolution of the operator. In addition to evaluating the performance of individual operators of a particular (P,R), we also propose a straightforward approach for multiresolution analysis, which combines the responses of multiple operators realized with different (P, R).

提出的纹理算子可以在圆形邻域中检测uniform LBP，角度空间量化任意，空间分辨率任意。我们在通用情况下，基于圆形对称邻域集中，半径为R有P个成员，推导了这个算子，表示为$LBP_{P,R}^{riu2}$。参数P控制着角度空间的量化，而R控制着算子的空间分辨率。除了评估对特定的(P,R)的单个算子的性能，我们还提出了一种直接的方法进行多分辨率分析，将不同参数(P,R)的多个算子的响应结合起来。

The discrete occurrence histogram of the “uniform” patterns (i.e., the responses of the $LBP_{P,R}^{riu2}$ operator) computed over an image or a region of image is shown to be a very powerful texture feature. By computing the occurrence histogram, we effectively combine structural and statistical approaches: The local binary pattern detects microstructures (e.g., edges, lines, spots, flat areas) whose underlying distribution is estimated by the histogram.

在图像中，或图像的一个区域中，计算得到的uniform模式的离散频数直方图（即，$LBP_{P,R}^{riu2}$算子的响应），是一种非常强的纹理特征。通过计算频数直方图，我们有效的结合了结构方法和统计方法：LBP检测的微结构（如，边缘，线段，斑点，平坦区域），其潜在分布是由直方图估计的。

We regard image texture as a two-dimensional phenomenon characterized by two orthogonal properties: spatial structure (pattern) and contrast (the “amount” of local image texture). In terms of gray-scale and rotation invariant texture description, these two are an interesting pair: Where spatial pattern is affected by rotation, contrast is not, and vice versa, where contrast is affected by the gray scale, spatial pattern is not. Consequently, as long as we want to restrict ourselves to pure gray-scale invariant texture analysis, contrast is of no interest as it depends on the gray scale.

我们将图像纹理视为一种二维现象，其特征是有两种不相关的性质：空间结构（模式），和对比度（局部图像纹理的量）。就灰度和旋转不变的纹理描述来说，这两个是一个有趣的对：空间模式是受旋转影响的，对比度不是，反之亦然，对比度是受灰度影响的，空间模式不是。结果是，只要我们想将我们限制于纯粹的灰度不变纹理分析，对比度就没什么兴趣，因为其依赖于灰度。

The $LBP_{P,R}^{riu2}$ operator is an excellent measure of the spatial structure of local image texture, but it, by definition, discards the other important property of local image texture, i.e., contrast, since it depends on the gray scale. If only rotation invariant texture analysis is desired, i.e., gray-scale invariance is not required, the performance of $LBP_{P,R}^{riu2}$ can be further enhanced by combining it with a rotation invariant variance measure $VAR_{P,R}$ that characterizes the contrast of local image texture. We present the joint distribution of these two complementary operators, $LBP_{P,R}^{riu2}/VAR_{P,R}$, as a powerful tool for rotation invariant texture classification.

$LBP_{P,R}^{riu2}$算子是局部图像纹理的空间结构的非常好的度量，但从定义上，其抛弃了局部图像纹理的其他重要的性质，即，对比度，因为其依赖于灰度。如果我们只想要旋转不变的纹理分析，即，灰度不变性不需要，那么$LBP_{P,R}^{riu2}$的性能，可以与旋转不变度量$VAR_{P,R}$结合到一起，进行增强，$VAR_{P,R}$描述的是局部图像纹理的对比度特征。我们提出了这两种互补的算子的联合分布，$LBP_{P,R}^{riu2}/VAR_{P,R}$，这是旋转不变纹理分类的强力工具。

As the classification rule, we employ nonparametric discrimination of sample and prototype distributions based on a log-likelihood measure of the dissimilarity of histograms, which frees us from making any, possibly erroneous, assumptions about the feature distributions.

至于分类规则，我们采用了样本的非参数区分和基于log似然度量的直方图不相似性的原型分布，这使我们不需要对特征分布做出任何错误假设。

The performance of the proposed approach is demonstrated with two experiments. Excellent results in both experiments demonstrate that the proposed texture operator is able to produce, from just one reference rotation angle, a representation that allows for discriminating a large number of textures at other rotation angles. The operators are also computationally attractive as they can be realized with a few operations in a small neighborhood and a lookup table.

提出的方法的性能用两个试验进行了证明。两个试验的非常好的结果证明了，提出的纹理算子可以从一个参考旋转角度产生出表示，而且可以对很多其他旋转角度的大量纹理进行区分。算子的计算量也很低，因为可以用一个小型邻域和一个查找表就可以实现。

The paper is organized as follows: The derivation of the operators and the classification principle are described in Section 2. Experimental results are presented in Section 3 and Section 4 concludes the paper.

本文组织如下：第2部分给出了算子的推导和分类准则，第3部分给出了试验结果，第4部分进行了总结。

## 2. Gray Scale and Rotation Invariant Local Binary Patterns

We start the derivation of our gray scale and rotation invariant texture operator by defining texture T in a local neighborhood of a monochrome texture image as the joint distribution of the gray levels of P (P > 1) image pixels:

作为推导灰度和旋转不变的纹理算子的开始，我们将灰度纹理图像的一个局部邻域中的纹理T，定义为P(P>1)个图像像素的联合分布：

$$T = t(g_c, g_0, . . . , g_{P-1})$$(1)

where gray value g_c corresponds to the gray value of the center pixel of the local neighborhood and g_p(p = 0; . . . ; P-1) correspond to the gray values of P equally spaced pixels on a circle of radius R(R>0) that form a circularly symmetric neighbor set.

其中灰度值g_c对应着局部邻域中央的灰度值，g_p(p = 0, . . . , P-1)对应着半径为R的圆上的P个相等间隔的像素，形成了一个圆形对称的邻域集。

If the coordinates of g_c are (0, 0), then the coordinates of g_p are given by (-Rsin(2πp/P), Rcos(2πp/P)). Fig. 1 illustrates circularly symmetric neighbor sets for various (P, R). The gray values of neighbors which do not fall exactly in the center of pixels are estimated by  interpolation.

如果g_c的坐标为(0,0)，那么g_p的坐标为(-Rsin(2πp/P), Rcos(2πp/P))。图1给出了各种(P,R)的圆形对称邻域集。没有落到像素中央的邻域的灰度值，由插值进行估计。

### 2.1 Achieving Gray-Scale Invariance

As the first step toward gray-scale invariance, we subtract, without losing information, the gray value of the center pixel ($g_c$) from the gray values of the circularly symmetric neighborhood $g_p (p=0, . . . , P-1)$, giving:

作为达到灰度不变性的第一步，我们在不损失信息的条件下，从圆形对称邻域$g_p (p=0, . . . , P-1)$中，减去中心像素的灰度值($g_c$)，得到：

$$T = t(g_c, g_0-g_c, g_1-g_c, . . . , g_{P-1}-g_c)$$(2)

Next, we assume that differences $g_p-g_c$ are independent of $g_c$, which allows us to factorize (2): 下一步，我们假设差值$g_p-g_c$与$g_c$无关，这使我们可以将(2)分解：

$$T ≈ t(g_c)t(g_0-g_c, g_1-g_c, . . . , g_{P-1}-g_c)$$(3)

In practice, an exact independence is not warranted; hence, the factorized distribution is only an approximation of the joint distribution. However, we are willing to accept the possible small loss in information as it allows us to achieve invariance with respect to shifts in gray scale. Namely, the distribution $t(g_c)$ in (3) describes the overall luminance of the image, which is unrelated to local image texture and, consequently, does not provide useful information for texture analysis. Hence, much of the information in the original joint gray level distribution (1) about the textural characteristics is conveyed by the joint difference distribution[28]:

实践中，并不保证严格的独立性；因为，分解的分布只是联合分布的一个近似。但是，我们愿意接受信息可能的很小的损失，因为这使我们获得了对灰度变化的不变性。。即，(3)中的分布$t(g_c)$描述了图像中的总体光照，与局部图像纹理不相关，结果是，并没有对纹理分析提供有用的信息。因此，原始联合灰度分布(1)中关于纹理特性的多数信息是通过联合差异分布来传递的：

$$T ≈ t(g_0-g_c, g_1-g_c, . . . , g_{p-1}-g_c)$$(4)

This is a highly discriminative texture operator. It records the occurrences of various patterns in the neighborhood of each pixel in a P-dimensional histogram. For constant regions, the differences are zero in all directions. On a slowly sloped edge, the operator records the highest difference in the gradient direction and zero values along the edge and, for a spot, the differences are high in all directions.

这是一个有高度区分性的纹理算子。其记录了每个像素的邻域中各种模式的频次，成为P维直方图。对于常数区域，所有方向的差值为0。在缓坡的边缘，该算子在梯度方向记录了最高的差值，沿着边缘方向则为0，对于一个斑点，在所有方向的差值都很大。

Signed differences $g_p-g_c$ are not affected by changes in mean luminance; hence, the joint difference distribution is invariant against gray-scale shifts. We achieve invariance with respect to the scaling of the gray scale by considering just the signs of the differences instead of their exact values:

有符号的差值$g_p-g_c$不会受到平均亮度变化的影响；因此，联合差值分布对灰度变化是不变的。我们通过只考虑差值的符号，而不是其准确值，来获得对灰度缩放的不变性：

$$T ≈ t(s(g_0-g_c), s(g_1-g_c), . . . , s(g_{P-1}-g_c))$$(5)

where

$$s(x) = 1, x≥0; 0, x<0$$(6)

By assigning a binomial factor 2^p for each sign s(g_p-g_c), we transform (5) into a unique LBP_{P,R} number that characterizes the spatial structure of the local image texture:

对每个符号s(g_p-g_c)指定一个二项式系数2^p，我们将(5)变换成一个唯一的LBP_{P,R}数，描述了局部图像纹理的空间结构特征：

$$LBP_{P,R} = \sum_{p=0}^{P-1} s(g_p-g_c)2^p$$(7)

The name "Local Binary Pattern" reflects the functionality of the operator, i.e., a local neighborhood is thresholded at the gray value of the center pixel into a binary pattern. LBP_{P,R} operator is by definition invariant against any monotonic transformation of the gray scale, i.e., as long as the order of the gray values in the image stays the same, the output of the LBP_{P,R} operator remains constant.

名称LBP反应的就是算子的功能，即，一个局部邻域用中心像素的灰度值取阈值，得到一个二值模式。LBP_{P,R}算子在定义上就对灰度值的任意单调变换是不变的，即，只要图像中灰度值的顺序保持不变，LBP_{P,R}的输出就保持不变。

If we set (P = 8, R = 1), we obtain LBP_{8,1}, which is similar to the LBP operator we proposed in [27]. The two differences between LBP_{8,1} and LBP are: 1) The pixels in the neighbor set are indexed so that they form a circular chain and 2) the gray values of the diagonal pixels are determined by interpolation. Both modifications are necessary to obtain the circularly symmetric neighbor set, which allows for deriving a rotation invariant version of LBP_{P,R}.

如果我们设(P = 8, R = 1)，我们得到LBP_{8,1}，与我们在[27]中提出的LBP算子类似。LBP_{8,1}与LBP的两个差异是：1)邻域中的像素是有索引的，这样可以形成一个圆形链；2)对角线像素的灰度值是由插值来确定的。这两个改动都是必须的，可以得到圆形对称的邻域，可以推导出一个旋转不变版的LBP_{P,R}。

### 2.2 Achieving Rotation Invariance

The LBP_{P,R} operator produces 2^P different output values, corresponding to the 2^P different binary patterns that can be formed by the P pixels in the neighbor set. When the image is rotated, the gray values g_p will correspondingly move along the perimeter of the circle around g_0. Since g_0 is always assigned to be the gray value of element (0, R) to the right of g_c rotating a particular binary pattern naturally results in a different LBP_{P,R} value. This does not apply to patterns comprising of only 0s (or 1s) which remain constant at all rotation angles. To remove the effect of rotation, i.e., to assign a unique identifier to each rotation invariant local binary pattern we define:

LBP_{P,R}算子产生2^P个不同的输出值，对应着2^P个不同的二值模式，可以由邻域集中的P个像素形成。当图像旋转后，灰度值g_p会相应的沿着圆的周长围绕g_0移动。由g_0永远是元素(0,R)的灰度值，旋转g_c很自然的产生不同的LBP_{P,R}值。这对只有0(或1)的模式不适用，因为它们在所有的旋转角度中都保持一致。为去除旋转的影响，即，对每个旋转不变的LBP，我们指定一个唯一的标识符，我们定义：

$$LBP_{P,R}^{ri} = min\{ROR(LBP_{P,R}, i) | i = 0,1, ..., P-1 \}$$(8)

where ROR(x, i) performs a circular bit-wise right shift on the P-bit number xi times. In terms of image pixels, (8) simply corresponds to rotating the neighbor set clockwise so many times that a maximal number of the most significant bits, starting from g_{P-1}, is 0.

其中ROR(x, i)对这个P-bit的数进行环形逐位的右向位移xi次。以图像像素来说，(8)对应着将邻域集顺时针旋转很多次，这样从g_{P-1}开始的最高有效位的多数都为0。

$LBP^{ri}_{P,R}$ quantifies the occurrence statistics of individual rotation invariant patterns corresponding to certain microfeatures in the image; hence, the patterns can be considered as feature detectors. Fig. 2 illustrates the 36 unique rotation invariant local binary patterns that can occur in the case of P = 8, i.e., $LBP^{ri}_{8, R}$ can have 36 different values. For example, pattern #0 detects bright spots, #8 dark spots and flat areas, and #4 edges. If we set R = 1, $LBP^{ri}_{8, 1}$ corresponds to the gray-scale and rotation invariant operator that we designated as LBPROT in [29].

$LBP^{ri}_{P,R}$量化了单个旋转不变的模式对特定微特征的频次统计量；因此，这些模式可以认为是特征提取器。图2给出了P=8时36个唯一的旋转不变LBP，即$LBP^{ri}_{8, R}$可以有36个不同的值。比如，#0 检测的是亮的斑点，#8 是暗的斑点和平台区域，#4 是边缘。如果我们设R=1，$LBP^{ri}_{8, 1}$对应的是我们在[29]中的灰度和旋转不变的算子LBPROT。

### 2.3 Improved Rotation Invariance with "Uniform" Patterns and Finer Quantization of the Angular Space

Our practical experience, however, has shown that LBPROT as such does not provide very good discrimination, as we also concluded in [29]. There are two reasons: The occurrence frequencies of the 36 individual patterns incorporated in LBPROT vary greatly and the crude quantization of the angular space at 45° intervals.

但是，我们的实践经验证明，LBPROT并没有给出很好的区分性，我们在[29]中给出了这样的结论。有两个原因：LBPROT中的36个独立的模式的发生频率变化太大，以及角度空间45°间隔的粗糙量化。

We have observed that certain local binary patterns are fundamental properties of texture, providing the vast majority, sometimes over 90 percent, of all 3x3 patterns present in the observed textures. This is demonstrated in more detail in Section 3 with statistics of the image data used in the experiments. We call these fundamental patterns "uniform" as they have one thing in common, namely, uniform circular structure that contains very few spatial transitions. “Uniform” patterns are illustrated on the first row of Fig. 2. They function as templates for microstructures such as bright spot (0), flat area or dark spot (8), and edges of varying positive and negative curvature (1-7).

我们观察到，特定的LBP是纹理的基础属性，在观察的纹理中，给出了所有3x3模式的绝大部分，有时候超过90%。这在第3部分给出更多细节，包括在试验中使用的图像的统计量。我们称这些基础模式为uniform，因为它们有一个共同点，即，一致的环形结构，包含非常少的空间变化。Uniform模式在图2的第一行给出。它们是一些微结构的模板，比如亮点(0)，平坦区域或暗点(8)，变化的正曲率与负曲率的边缘(1-7)。

To formally define the "uniform" patterns, we introduce a uniformity measure U("pattern"), which corresponds to the number of spatial transitions (bitwise 0/1 changes) in the "pattern." For example, patterns 00000000_2 and 11111111_2 have U value of 0, while the other seven patterns in the first row of Fig. 2 have U value of 2 as there are exactly two 0/1 transitions in the pattern. Similarly, the other 27 patterns have U value of at least 4. We designate patterns that have U value of at most 2 as "uniform" and propose the following operator for gray-scale and rotation invariant texture description instead of $LBP^{ri}_{P,R}$:

为正式的定义uniform模式，我们提出了uniformity度量U("pattern")，对应着模式中的空间变化（逐位的0/1变化）数量。比如，模式00000000_2和11111111_2的U值为0，图2的第一行的其他7个模式的U值为2，因为模式中的0/1变化数为2。类似的，其他的27个模式的U值至少为4。我们定义U值最多为2的模式为uniform，提出下面的灰度和旋转不变纹理描述算子，替代$LBP^{ri}_{P,R}$：

$$LBP_{P,R}^{riu2} = \sum_{p=0}^{P-1} s(g_p-g_c), if U(LBP_{P,R}) ≤ 2; P+1, otherwise$$(9)

where

$$U(LBP_{P,R}) = |s(g_{P-1}-g_c) - s(g_0-g_c)| + \sum_{p=1}^{P-1} |s(g_p-g_c) - s(g_{p-1}-g_c)|$$(10)

Superscript riu2 reflects the use of rotation invariant "uniform" patterns that have U value of at most 2. By definition, exactly P + 1 "uniform" binary patterns can occur in a circularly symmetric neighbor set of P pixels. Equation (9) assigns a unique label to each of them corresponding to the number of "1" bits in the pattern (0->P), while the "nonuniform" patterns are grouped under the "miscellaneous" label (P + 1). In Fig. 2, the labels of the "uniform" patterns are denoted inside the patterns. In practice, the mapping from $LBP_{P,R}$ to $LBP^{riu2}_{P,R}$, which has P+2 distinct output values, is best implemented with a lookup table of 2^P elements.

上标riu2反应的是使用旋转不变的uniform模式，U值最多为2。从定义上来说，在P像素的环形对称邻域中，会有P+1个uniform二值模式。式9为其中每一个指定了一个唯一的标签，对应着模式中1的数量，而非uniform模式分组在“其他”的标签(P+1)下。在图2中，uniform模式的标签在模式中间有标记。实践中，从$LBP_{P,R}$到$LBP^{riu2}_{P,R}$的映射用一个查找表来实现最好。

The final texture feature employed in texture analysis is the histogram of the operator outputs (i.e., pattern labels) accumulated over a texture sample. The reason why the histogram of “uniform” patterns provides better discrimination in comparison to the histogram of all individual patterns comes down to differences in their statistical properties. The relative proportion of “nonuniform” patterns of all patterns accumulated into a histogram is so small that their probabilities cannot be estimated reliably. Inclusion of their noisy estimates in the dissimilarity analysis of sample and model histograms would deteriorate performance.

纹理分析中最终采用的纹理特征，是算子输出（即，模式标签）在一个纹理样本上累积的直方图。与所有模式的直方图相比，uniform模式的直方图给出更好的区分性的原因，要到其统计属性的差异上来看。所有模式累积成直方图，其中非uniform模式的相对比例太小了，其概率不能可靠的进行估计。将其不可靠的估计纳入到样本和模型直方图的不相似性的分析中，会使性能下降。

We noted earlier that the rotation invariance of LBPROT($LBP^{ri}_{8,1}$) is hampered by the crude 45° quantization of the angular space provided by the neighbor set of eight pixels. A straightforward fix is to use a larger P since the quantization of the angular space is defined by (360°/P). However, certain considerations have to be taken into account in the selection of P. First, P and R are related in the sense that the circular neighborhood corresponding to a given R contains a limited number of pixels (e.g., nine for R=1), which introduces an upper limit to the number of nonredundant sampling points in the neighborhood. Second, an efficient implementation with a lookup table of 2^P elements sets a practical upper limit for P. In this study, we explore P values up to 24, which requires a lookup table of 16 MB that can be easily managed by a modern computer.

我们前面注意了，LBPROT($LBP^{ri}_{8,1}$)的旋转不变性受到8像素的邻域的角空间的粗糙的45°量化所妨碍。一种直接的修复是，使用更大的P，因为角空间的量化是由360°/P定义的。但是，对P的选择必须有特定考虑。首先，P和R是相关的，因为对给定的R的圆形邻域包含有限数量的像素，如，对于R=1，P=9，对非冗余采样点设置了上限。第二，大小为2^P的查找表的高效实现，为P设置了一定的上限。本研究中，我们探索了最大为24的P值，这需要16MB大小的查找表，一般电脑都可以很容易处理。

### 2.4 Rotation Invariant Variance Measures of the Contrast of Local Image Texture

The $LBP^{riu2}_{P,R}$ operator is a gray-scale invariant measure, i.e., its output is not affected by any monotonic transformation of the gray scale. It is an excellent measure of the spatial pattern, but it, by definition, discards contrast. If gray-scale invariance is not required and we wanted to incorporate the contrast of local image texture as well, we can measure it with a rotation invariant measure of local variance:

$LBP^{riu2}_{P,R}$算子是一个灰度不变的度量，即，其输出不会受到灰度的单调变换影响。这是空间模式的一个极好度量，但在定义上就丢弃了对比度。如果不需要灰度不变性，我们希望将局部图像纹理的对比度也纳入进来，我们可以用局部方差的旋转不变度量来进行度量：

$$VAR_{P,R} = \frac {1}{P} \sum_{p=0}^{P-1} (g_p-μ)^2, where μ=\frac {1}{P} \sum_{p=0}^{P-1} g_p$$(11)

$VAR_{P,R}$ is by definition invariant against shifts in gray scale. Since $LBP^{riu2}_{P,R}$ and $VAR_{P,R}$ are complementary, their joint distribution $LBP^{riu2}_{P,R}/VAR_{P,R}$ is expected to be a very powerful rotation invariant measure of local image texture. Note that, even though we in this study restrict ourselves to using only joint distributions of $LBP^{riu2}_{P,R}$ and $VAR_{P,R}$ operators that have the same (P,R) values, nothing would prevent us from using joint distributions of operators computed at different neighborhoods.

$VAR_{P,R}$在定义上就是对灰度的变化不变的。由于$LBP^{riu2}_{P,R}$和$VAR_{P,R}$是互补的，其联合分布$LBP^{riu2}_{P,R}/VAR_{P,R}$就应当是局部图像纹理的一个非常强的旋转不变度量。注意，即使在这个研究中，我们自我限制只使用$LBP^{riu2}_{P,R}$和$VAR_{P,R}$的联合分布，其有着相同的(P,R)值，但我们还可以使用不同邻域的算子联合分布。

### 2.5 Nonparametric Classification Principle

In the classification phase, we evaluate the dissimilarity of sample and model histograms as a test of goodness-of-fit, which is measured with a nonparametric statistical test. By using a nonparametric test, we avoid making any, possibly erroneous, assumptions about the feature distributions. There are many well-known goodness-of-fit statistics such as the chi-square statistic and the G (log-likelihood ratio) statistic [33]. In this study, a test sample S was assigned to the class of the model M that maximized the log-likelihood statistic:

在分类阶段，我们评估了样本和模型直方图之间的不相似性，作为适合度测试，这是用一个非参数的统计测试来度量的。通过使用一个非参数测试，我们避免了对特征分布做出任何可能的错误假设。有很多适合度统计量，比如chi-square统计量，和G(log似然率)统计。在本研究中，如果最大化下面的log似然统计量，则测试样本S被指定为模型M的类别：

$$L(S,M)=\sum_{b=1}^B S_b logM_b$$(12)

where B is the number of bins and S_b and M_b correspond to the sample and model probabilities at bin b, respectively. Equation (12) is a straightforward simplification of the G (log-likelihood ratio) statistic:

其中B是bins的数量，S_b和M_b分别对应着bin b处的样本和模型概率。式12是G统计量的直接简化：

$$G(S,M) = 2\sum_{b=1}^B S_b log \frac {S_b}{M_b} = 2\sum_{b=1}^B [S_b log S_b - S_b log M_b]$$(13)

where the first term of the righthand expression can be ignored as a constant for a given S. 其中对给定的S，右边表达式的第一项可以忽略为常数。

L is a nonparametric pseudometric that measures likelihoods that sample S is from alternative texture classes, based on exact probabilities of feature values of preclassified texture models M. In the case of the joint distribution $LBP^{riu2}_{P,R}/VAR_{P,R}$, (12) was extended in a straightforward manner to scan through the two-dimensional histograms.

L是一个非参数的伪度量，度量的是样本S属于某个纹理类别的似然，基于预先分类的纹理模型M的特征值的准确概率。在联合分布$LBP^{riu2}_{P,R}/VAR_{P,R}$的情况下，(12)可以很直接的拓展到扫描二维直方图的情况。

Sample and model distributions were obtained by scanning the texture samples and prototypes with the chosen operator and dividing the distributions of operator outputs into histograms having a fixed number of B bins. Since $LBP^{riu2}_{P,R}$ has a fixed set of discrete output values (0 -> P+1), no quantization is required, but the operator outputs are directly accumulated into a histogram of P+2 bins. Each bin effectively provides an estimate of the probability of encountering the corresponding pattern in the texture sample or prototype. Spatial dependencies between adjacent neighborhoods are inherently incorporated in the histogram because only a small subset of patterns can reside next to a given pattern.

样本和模型的分布，是通过用选择的算子扫描纹理样本和原型，并将算子输出的分布分成直方图（bins数量固定为B），而得到的。由$LBP^{riu2}_{P,R}$有离散输出值(0 -> P+1)的固定集，不需要任何量化，但算子输出直接累积成P+2 bins的直方图。每个bin有效的给出了在纹理样本或原型中遇到对应模式的概率估计。临近邻域的空间依赖关系是内在于直方图的，因为只有模式的一个小型子集可以在给定模式的附近驻留。

Variance measure $VAR_{P,R}$ has a continuous-valued output; hence, quantization of its feature space is needed. This was done by adding together feature distributions for every single model image in a total distribution, which was divided into B bins having an equal number of entries. Hence, the cut values of the bins of the histograms corresponded to the (100/B) percentile of the combined data. Deriving the cut values from the total distribution and allocating every bin the same amount of the combined data guarantees that the highest resolution of quantization is used where the number of entries is largest and vice versa. The number of bins used in the quantization of the feature space is of some importance as histograms with a too small number of bins fail to provide enough discriminative information about the distributions. On the other hand, since the distributions have a finite number of entries, a too large number of bins may lead to sparse and unstable histograms. As a rule of thumb, statistics literature often proposes that an average number of 10 entries per bin should be sufficient. In the experiments, we set the value of B so that this condition is satisfied.

方差度量$VAR_{P,R}$的输出值为连续的；因此，需要对其特征空间进行量化。这是通过将每个单个模型图像的特征分布加到一起成为总计分布，并分成B个bins有相同数量的entries，而得到的。因此，直方图bins的cut values对应着结合的数据的100/B百分比。从总分布中推导得到cut values，给每个bin分配相同数量的结合数据，确保使用了量化的最高分辨率，entries数量是最大的，反之亦然。在量化特征空间中使用的bins的数量，是与直方图的重要性类似的，bins数量太少，不能提供分布的足够区分性信息。另一方面，由于分布的entries数量有限，bins数量太大，会导致稀疏不稳定的直方图。根据经验，每个bin中平均有10个entries应当足够。试验中，我们设B值以满足这个条件。

### 2.6 Multiresolution Analysis

We have presented general rotation-invariant operators for characterizing the spatial pattern and the contrast of local image texture using a circularly symmetric neighbor set of P pixels placed on a circle of radius R. By altering P and R, we can realize operators for any quantization of the angular space and for any spatial resolution. Multiresolution analysis can be accomplished by combining the information provided by multiple operators of varying (P,R).

我们给出了通用的旋转不变算子，用半径为R包含P个像素的圆形对称邻域，描述局部图像纹理的空间模式和对比度特征。通过变化P和R，我们可以实现角空间任意量化，任意分辨率的算子。通过结合不同(P,R)的多个算子的信息，可以实现多分辨率分析。

In this study, we perform straightforward multiresolution analysis by defining the aggregate dissimilarity as the sum of individual log-likelihoods computed from the responses of individual operators

本研究中，我们通过定义聚积不相似性为单个算子的响应的log似然的和，进行直接的多分辨率分析

$$L_N = \sum_{n=1}^N L(S^n, M^n)$$(14)

where N is the number of operators and S^n and M^n correspond to the sample and model histograms extracted with operator n(n = 1, . . . , N), respectively. This expression is based on the additivity property of the G statistic (13), i.e., the results of several G tests can be summed to yield a meaningful result. If X and Y are independent random events and S_X, S_Y , M_X, and M_Y are the respective marginal distributions for S and M, then $G(S_{XY}, M_{XY}) = G(S_X, M_X) + G(S_Y, M_Y)$[18].

其中N是算子的数量，S^n和M^n分别对应用算子n(n = 1, . . . , N)提取出来的样本和模型的直方图。这个表达式是基于G统计量的可加性质，即，几个G测试的结果可以加到一起，得到一个有意义的结果。如果X和Y是独立的随机事件，S_X, S_Y , M_X和M_Y是S和M的分别的边缘分布，那么$G(S_{XY}, M_{XY}) = G(S_X, M_X) + G(S_Y, M_Y)$[18]。

Generally, the assumption of independence between different texture features does not hold. However, estimation of exact joint probabilities is not feasible due to statistical unreliability and computational complexity of large multidimensional histograms. For example, the joint histogram of $LBP^{riu2}_{8,R}$, $LBP^{riu2}_{16,R}$, and $LBP^{riu2}_{24,R}$ would contain 4,680 (10x18x26) cells. To satisfy the rule of thumb for statistical reliability, i.e., at least 10 entries per cell on average, the image should be of roughly (216+2R)(216+2R) pixels in size. Hence, high-dimensional histograms would only be reliable with really large images, which renders them impractical. Large multidimensional histograms are also computationally expensive, both in terms of computing speed and memory consumption.

一般来说，不同纹理特征之间独立性的假设不能成立。但是，确切的联合概率的估计是不可行的，因为统计的不可靠性和大型多维度直方图的计算复杂性。比如，$LBP^{riu2}_{8,R}$, $LBP^{riu2}_{16,R}$和$LBP^{riu2}_{24,R}$的联合直方图，会包含4680单元。为满足经验上的统计可靠性，即，平均至少每个单元10个entries，图像的大小大致应当是(216+2R)(216+2R)像素。因此，高维直方图只对很大的图像是可靠的，这使其不切实际。大的多维直方图在计算量上的很昂贵，包括计算速度和内存消耗。

We have recently successfully employed this approach also in texture segmentation, where we quantitatively compared different alternatives for combining individual histograms for multiresolution analysis [25]. In this study, we restrict ourselves to combinations of at most three operators.

我们最近成功的采用这种方法进行纹理分割，其中我们定量的比较了结合单个直方图进行多分辨率分析的不同替代。本研究中，我们最多结合3个算子。

## 3. Experiments

We demonstrate the performance of our approach with two different problems of rotation invariant texture analysis. Experiment #1 is replicated from a recent study on rotation invariant texture classification by Porter and Canagarajah [31] for the purpose of obtaining comparative results to other methods. Image data includes 16 source textures captured from the Brodatz album [4]. Considering this in conjunction with the fact that rotated textures are generated from the source textures digitally, this image data provides a slightly simplified but highly controlled problem for rotation invariant texture analysis. In addition to the original experimental setup, where training was based on multiple rotation angles, we also consider a more challenging setup, where the texture classifier is trained at only one particular rotation angle and then tested with samples from other rotation angles.

我们用两个不同的旋转不变纹理分析问题证明了方法的性能。试验#1是从最近Porter and Canagarajah [31]的旋转不变纹理分类的研究中复制过来的，目的是得到与其他方法类似可比较的结果。图像数据包括Brodatz[4]中的16种源纹理。旋转纹理是从源纹理中生成的，这些图像数据提供了一个略微简化的但高度受控的问题，进行旋转不变纹理分析。除了原始的试验设置，其中训练是基于多个旋转角度的，我们还考虑了一个更有挑战性的设置，其中纹理分类器是只在某特定旋转角度训练的，然后再其他旋转角度的样本上进行测试。

Experiment #2 involves a new set of texture images which have a natural tactile dimension and natural appearance of local intensity distortions caused by the tactile dimension. Some source textures have large intraclass variation in terms of color content, which results in highly different gray-scale properties in the intensity images. Adding the fact that the textures were captured using three different illuminants of different color spectra, this image data presents a very realistic and challenging problem for illumination and rotation invariant texture analysis.

试验#2涉及了纹理图像的一个新集合，有着自然的触感维度，和由触感维度导致的局部灰度形变的自然外观。一些源纹理的色彩内容类内变化比较大，结果是灰度图像中有非常不同的灰度性质。加上纹理是用三种不同的颜色光照得到的，这些图像数据是非常贴合实际，非常有挑战性的光照不变、旋转不变纹理分析问题。

To incorporate three different spatial resolutions and three different angular resolutions, we realized $LBP^{riu2}_{P,R}$ and $VAR_{P,R}$ with {P,R} values of (8,1), (16,2), and (24,3) in the experiments. Corresponding circularly symmetric neighborhoods are illustrated in Fig. 1. In multiresolution analysis, we use the three 2-resolution combinations and the one 3-resolution combination these three alternatives can form.

为纳入三种不同的空间分辨率和三种不同的角度分辨率，我们用三种{P,R}值(8,1), (16,2), (24,3)在试验中实现$LBP^{riu2}_{P,R}$和$VAR_{P,R}$。对应圆形对称的邻域，如图1所示。在多分辨率分析中，我们使用三种2分辨率组合，和一种3分辨率组合。

Before going into the experiments, we take a quick look at the statistical foundation of $LBP^{riu2}_{P,R}$. In the case of $LBP^{riu2}_{8,R}$, we choose nine "uniform" patterns out of the 36 possible patterns, merging the remaining 27 under the "miscellaneous" label. Similarly, in the case of $LBP^{riu2}_{16,R}$, we consider only 7 percent (17 out of 243) of the possible rotation invariant patterns. Taking into account a minority of the possible patterns and merging a majority of them could imply that we are throwing away most of the pattern information. However, this is not the case, as the "uniform" patterns appear to be fundamental properties of local image texture, as illustrated by the numbers in Table 1.

在进入试验之前，我们迅速看看$LBP^{riu2}_{P,R}$的统计基础。在$LBP^{riu2}_{8,R}$的情况下，我们从36种可能中选择9个uniform模式，剩下的27种合并成"miscellaneous"标签。类似的，在$LBP^{riu2}_{16,R}$中，我们只考虑7%的可能的旋转不变模式（243个中的17个）。只考虑可能模式中的很小一部分，并将大部分进行合并，可能意味着，我们扔掉了大部分模式信息。但是，这并不是这个情况，因为uniform模式是局部图像纹理的基本性质，如表1的数字所示。

In the case of the image data of Experiment #1, the nine "uniform" patterns of $LBP^{riu2}_{8,1}$ contribute from 76.6 percent up to 91.8 percent of the total pattern data, averaging 87.2 percent. The most frequent individual pattern is symmetric edge detector 00001111_2 with an 18.0 percent share, followed by 00011111_2 (12.8 percent) and 00000111_2 (11.8 percent); hence, these three patterns contribute 42.6 percent of the textures. As expected, in the case of $LBP^{riu2}_{16,1}$, the 17 "uniform" patterns contribute a smaller proportion of the image data, from 50.9 percent up to 76.4 percent of the total pattern data, averaging 66.9 percent. The most frequent pattern is the flat area/dark spot detector 1111111111111111_2 with an 8.8 percent share.

在试验#1的图像数据的情况下，$LBP^{riu2}_{8,1}$的9个uniform模式贡献了76.6%大到91.8%的总计模式数据，平均87.2%。最常见的单个模式是，对称边缘检测器00001111_2，占比18.0%，然后是00011111_2(12.8%)和00000111_2(11.8%)；因此，这三种模式贡献了42.6%的纹理。如同期待一样，在$LBP^{riu2}_{16,1}$的情况下，17个uniform模式贡献的图像数据比例更小一些，是总模式数据的50.9%到76.4%，平均66.9%。最常见的模式是平坦区域/黑斑点检测器1111111111111111_2，占比8.8%。

The numbers for the image data of Experiment #2 are remarkably similar. The contribution of the nine "uniform" patterns of $LBP^{riu2}_{8,1}$ totaled over the three illuminants (see Section 3.2.1) ranges from 82.4 percent to 93.3 percent, averaging 89.7 percent. The three most frequent patterns are again 00001111_2 (18.9 percent), 00000111_2 (15.2 percent), and 00011111_2 (14.5 percent), totalling 48.6 percent of the patterns. The contribution of the 17 "uniform" patterns of $LBP^{riu2}_{16,2}$ ranges from 57.6 percent to 79.6 percent, averaging 70.7 percent. The most frequent patterns is again 1111111111111111_2 with an 8.7 percent share. In the case of $LBP^{riu2}_{24,3}$, the 25 "uniform" patterns contribute 54.0 percent of the local texture. The two most frequent patterns are the flat area/dark spot detector (all bits "1") with an 8.6 percent share and the bright spot detector (all bits "0") with an 8.2 percent share.

试验#2中的图像数据数量非常类似。$LBP^{riu2}_{8,1}$的9个uniform模式在三种光照情况下的贡献从82.4%到93.3%，平均89.7%。三种最常见的模式又是00001111_2 (18.9%)，00000111_2 (15.2%)，00011111_2 (14.5%)，共计所有模式的48.6%。$LBP^{riu2}_{16,2}$的17个uniform模式贡献了57.6%到79.6%，平均70.7%。最常见的模式又是1111111111111111_2，占比8.7%。在$LBP^{riu2}_{24,3}$中，25个uniform模式贡献了局部纹理的54.0%。两种最常见的模式是平坦区域/黑斑点检测器（所有比特都是1），比例8.6%，和亮斑点检测器（所有比特0），比例8.2%。

### 3.1 Experiment #1

In their comprehensive study, Porter and Canagarajah [31] presented three feature extraction schemes for rotation invariant texture classification, employing the wavelet transform, a circularly symmetric Gabor filter, and a Gaussian Markov Random Field with a circularly symmetric neighbor set. They concluded that the wavelet-based approach was the most accurate and exhibited the best noise performance also having the lowest computational complexity.

在Porter and Canagarajah [31]的全面研究中，给出了三种特征提取方案，进行旋转不变的纹理分类，采用了小波变换，圆形对称Gabor滤波器，和带有圆形对称邻域集的高斯Markov随机场。他们得出结论基于小波的方法是最准确的，可以得到最好噪声性能，计算复杂度也最低。

### 3.2 Experiment #2

## 4 Discussion

We presented a theoretically and computationally simple yet efficient multiresolution approach to gray-scale and rotation invariant texture classification based on "uniform" local binary patterns and nonparametric discrimination of sample and prototype distributions. "Uniform" patterns were recognized to be a fundamental property of texture as they provide a vast majority of local texture patterns in examined textures, corresponding to texture microstructures such as edges. By estimating the distributions of these microstructures, we combined structural and statistical texture analysis.

我们提出了一种理论上和计算上都很简单，但是高效的多分辨率方法进行灰度不变、旋转不变的纹理分类，方法是基于uniform LBP和样本、原型分布的非参数区分。Uniform模式是纹理的一种基础性质，因为它们提供了主要的待检纹理的局部纹理模式，对应纹理的微结构如边缘。通过估计这些微结构的分布，我们将结构分析和统计纹理分析结合到了一起。

We developed a generalized gray-scale and rotation invariant operator $LBP^{riu2}_{P,R}$, which allows for detecting "uniform" patterns in circular neighborhoods of any quantization of the angular space and at any spatial resolution. We also presented a simple method for combining responses of multiple operators for multiresolution analysis by assuming that the operator responses are independent.

我们提出了一种通用的灰度不变和旋转不变的算子$LBP^{riu2}_{P,R}$，可以在角空间的任意量化和任意空间分辨率的圆形邻域中检测uniform模式。我们还提出了一种简单的方法，假设算子响应是独立的，将多分辨率分析的多个算子的响应结合到一起。

Excellent experimental results obtained in two problems of true rotation invariance where the classifier was trained at one particular rotation angle and tested with samples from other rotation angles demonstrate that good discrimination can be achieved with the occurrence statistics of "uniform" rotation invariant local binary patterns.

在两个真正的旋转不变问题中得到了非常好的试验结果，在一个特定方向训练了分类器，用其他旋转角度的样本进行了测试，证明了用uniform旋转不变LBP的发生统计量可以得到很好的区分性。

The proposed approach is very robust in terms of grayscale variations caused, e.g., by changes in illumination intensity since the $LBP^{riu2}_{P,R}$ operator is by definition invariant against any monotonic transformation of the gray scale. This should make it very attractive in situations where nonuniform illumination conditions are a concern, e.g., in visual inspection. Gray-scale invariance is also necessary if the gray-scale properties of the training and testing data are different. This was clearly demonstrated in our recent study on supervised texture segmentation with the same image set that was used by Randen and Husoy in their recent extensive comparative study [32]. In our experiments, the basic 3x3 LBP operator provided better performance than any of the methods benchmarked by Randen and Husoy for 10 of the 12 texture mosaics and, in most cases, by a clear margin [28]. Results in Experiment #2, involving three illuminants with different spectra and large intraclass color variations in source textures demonstrate that the proposed approach is also robust in terms of color variations.

提出的方法非常稳健，如对光照亮度变化导致的灰度变化，因为$LBP^{riu2}_{P,R}$在定义上就是对任何灰度的单调变化是不变的。这使其在非一致光照条件的情况非常理想，如，在视觉检查中。如果训练和测试数据的灰度性质是不同的，灰度不变性也是非常必须的。这在我们最近对有监督纹理分割的研究中也得到了证明，使用的图像集与Randen and Husoy[32]的研究一样。在我们的试验中，基础的3x3 LBP算子比Randen and Husoy测试的所有方法都要好，在12种纹理马赛克上的10种中都有显著的改进。在试验#2中的结果，涉及到了3种光照下不同光谱，在源纹理种有很大的类内色彩变化，证明了提出的方法对色彩变化也非常稳健。

Computational simplicity is another advantage as the operators can be realized with a few comparisons in a small neighborhood and a lookup table. This facilitates a very straightforward and efficient implementation, which may be mandatory in time critical applications.

计算上的简单性是另一个优势，因为算子可以用很小邻域种的几个比较和一个查找表就可以实现。这促进了非常直接非常高效的实现，在对时间要求很高的应用中，这是必须的。

If gray-scale invariance is not required, performance can be further improved by combining the $LBP^{riu2}_{P,R}$ operator with the rotation invariant variance measure $VAR_{P,R}$ that characterizes the contrast of local image texture. As we observed in the experiments, the joint distributions of these orthogonal operators are very powerful tools for rotation invariant texture analysis.

如果不需要会不不变性，性能可以得到进一步改进，即将$LBP^{riu2}_{P,R}$算子与旋转不不变度量$VAR_{P,R}$结合起来，后者描述的是局部图像纹理的对比度。我们在试验中可以观察到，两个不相关的算子的联合分布，对旋转不变的纹理分析是非常强的工具。

The spatial size of the operators is of interest. Some may find our experimental results surprisingly good considering how small spatial support our operators have, for example, in comparison to much larger Gabor filters that are often used in texture analysis. However, the built-in spatial support of our operators is inherently larger as only a limited subset of patterns can reside adjacent to a particular pattern. Still, our operators may not be suitable for discriminating textures where the dominant features appear at a very large scale. This can be addressed by increasing the spatial predicate R, which allows generalizing the operators to any neighborhood size.

算子的空间尺度是非常有趣的。一些会发现，我们的试验结果非常好，因为算子的空间支撑非常小，比较起来，纹理分析中常用的Gabor滤波器就非常大。但是，我们算子的内建空间支撑本身就很大，因为模式的有限子集可以与特定模式邻接。而且，我们的算子对于主要特征的尺度非常大的纹理，不适合区分。这可以通过增加空间尺度R来应对，这可以将算子泛化到任意邻域大小。

The performance can be further enhanced by multiresolution analysis. We presented a straightforward method for combining operators of different spatial resolutions for this purpose. Experimental results involving three different spatial resolutions showed that multiresolution analysis is beneficial, except in those cases where a single resolution was already sufficient for a very good discrimination. Ultimately, we would want to incorporate scale invariance, in addition to gray-scale and rotation invariance.

用多分辨率分析，可以进一步改进性能。我们提出了一种直接方法，将不同空间分辨率的算子结合到一起，进行多分辨率分析。三种不同的空间分辨率的试验结果表明，多分辨率分析是有好用的，除非一种分辨率就足以进行很好的区分。最终，我们希望能将尺度不变性，灰度不变性和旋转不变性结合到一起。

Regarding future work, one thing deserving a closer look is the use of a task specific subset of rotation invariant patterns, which may, in some cases, provide better performance than "uniform" patterns. Patterns or pattern combinations are evaluated with some criterion, e.g., classification accuracy on a training data, and the combination providing the best accuracy is chosen. Since combinatorial explosion may prevent an exhaustive search through all possible subsets, suboptimal solutions such as stepwise or beam search should be considered. We have explored this approach in a classification problem involving 16 textures from the Curet database [10] with an 11.25° tilt between training and testing images [24]. Thanks to its invariance against monotonic gray-scale transformations, the methodology is applicable to textures with minor 3D transformations, corresponding to such textures which a human can easily, without attention, classify to the same categories as the original textures. Successful discrimination of Curet textures captured from slightly different viewpoints demonstrates the robustness of the approach with respect to small distortions caused by height variations, local shadowing, etc.

一个未来的工作是使用任务专有的旋转不变模式，这在一些情况下会比uniform模式得到更好的性能。用一些准则可以评估模式或模式组合，比如，训练数据上的分类准确率，选择给出最好准确率的组合。

In a similar fashion to deriving a task-specific subset of patterns, instead of using a general purpose set of operators, the parameters P and R could be “tuned” for the task in hand or even for each texture class separately. We also reported that when classification errors occur, the model of the true class very often ranks second. This suggests that classification could be carried out in stages by selecting operators which best discriminate among remaining alternatives.

Our findings suggest that complementary information of local spatial patterns and contrast plays an important role in texture discrimination. There are studies on human perception that support this conclusion. For example, Tamura et al. [34] designated coarseness, edge orientation, and contrast as perceptually important textural properties. The LBP histograms provide information of texture orientation and coarseness, while the local gray-scale variance characterizes contrast. Similarly, Beck et al. [3] suggested that texture segmentation of human perception might occur as a result of differences in the first-order statistics of textural elements and their parts, i.e., in the LBP histogram.