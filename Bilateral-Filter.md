# Bilateral Filtering for Gray and Color Images

C. Tomasi, R. Manduchi, Stanford University, Apple Computer, Inc.

## 0. Abstract

Bilateral filtering smooths images while preserving edges, by means of a nonlinear combination of nearby image values. The method is noniterative, local, and simple. It combines gray levels or colors based on both their geometric closeness and their photometric similarity, and prefers near values to distant values in both domain and range. In contrast with filters that operate on the three bands of a color image separately, a bilateral filter can enforce the perceptual metric underlying the CIE-Lab color space, and smooth colors and preserve edges in a way that is tuned to human perception. Also, in contrast with standard filtering, bilateral filtering produces no phantom colors along edges in color images, and reduces phantom colors where they appear in the original image.

双边滤波通过附近像素值的非线性组合，可以在保持边缘的同时平滑图像。本方法是非迭代的，局部的，非常简单。算法基于其几何接近程度和灰度相似性将灰度级或色彩值结合到一起，在领域和范围内都倾向于附近的值，而不倾向于较远的值。有的滤波器在彩色图像的三个通道上分别运算，而双边滤波器则在CIE-Lab色彩空间中加入感知度量，以人类感知的方式平滑色彩、保持边缘。同时，与标准滤波相比，双边滤波在彩色图像中的边缘附近不会产生假颜色，对原始图像中的假颜色则有所减少。

## 1. Introduction

Filtering is perhaps the most fundamental operation of image processing and computer vision. In the broadest sense of the term “filtering,” the value of the filtered image at a given location is a function of the values of the input image in a small neighborhood of the same location. In particular, Gaussian low-pass filtering computes a weighted average of pixel values in the neighborhood, in which, the weights decrease with distance from the neighborhood center. Although formal and quantitative explanations of this weight fall-off can be given [11], the intuition is that images typically vary slowly over space, so near pixels are likely to have similar values, and it is therefore appropriate to average them together. The noise values that corrupt these nearby pixels are mutually less correlated than the signal values, so noise is averaged away while signal is preserved.

滤波可能是最基本的图像处理和计算机视觉运算。在术语“滤波”最防范的含义中，滤波过的图像在给定位置的值，是输入图像在同样位置的一个小邻域的值的函数。特别是，高斯低通滤波计算邻域中像素的加权平均，其中权重随着到邻域中心距离的增加而递减。虽然这种权重衰减已经给出了公式的定量解释，但直觉仍然是，图像一般在空间中变化很缓慢，所以临近的像素很可能有类似的像素值，因此将其进行平均是合适的。使附近像素值出错的噪声值，互相之间关系很小，没有像素值的相关性那么大，所以噪声得到平均，而像素值得到保留。

The assumption of slow spatial variations fails at edges, which are consequently blurred by low-pass filtering. Many efforts have been devoted to reducing this undesired effect [1,2,3,4,5,6,7,8,9,10,12,13,14,15,17]. How can we prevent averaging across edges, while still averaging within smooth regions? Anisotropic diffusion [12, 14] is a popular answer: local image variation is measured at every point, and pixel values are averaged from neighborhoods whose size and shape depend on local variation. Diffusion methods average over extended regions by solving partial differential equations, and are therefore inherently iterative. Iteration may raise issues of stability and, depending on the computational architecture, efficiency. Other approaches are reviewed in section 6.

空间慢变化的假设在边缘附近是不成立的，所以低通滤波器会使边缘模糊。很多工作试图减少这种不理想的效果。我们怎样才能在平滑区域进行平均的同时，防止把边缘处也一起平均？各向异性滤波是一种流行的答案：在每一点处衡量局部图像变化，对像素平均所在的邻域，其大小和形状依赖于局部变化。扩散方法在拓展的区域进行平均，区域是通过求解偏微分方程得到的，因此其本质就是迭代的。迭代可能带来稳定性的问题，依赖于计算架构，效率。其他方法在第6部分进行回顾。

In this paper, we propose a noniterative scheme for edge preserving smoothing that is noniterative and simple. Although we claims no correlation with neurophysiological observations, we point out that our scheme could be implemented by a single layer of neuron-like devices that perform their operation once per image.

本文中，我们提出了一种非迭代的方案进行保护边缘的平滑，非常简单。虽然我们声称与神经生理学没有关系，我们也指出，我们的方案可以通过单层神经类设备实现，每幅图像进行一次运算即可。

Furthermore, our scheme allows explicit enforcement of any desired notion of photometric distance. This is particularly important for filtering color images. If the three bands of color images are filtered separately from one another, colors are corrupted close to image edges. In fact, different bands have different levels of contrast, and they are smoothed differently. Separate smoothing perturbs the balance of colors, and unexpected color combinations appear. Bilateral filters, on the other hand, can operate on the three bands at once, and can be told explicitly, so to speak, which colors are similar and which are not. Only perceptually similar colors are then averaged together, and the artifacts mentioned above disappear.

而且，我们的方案可以显式的将任意想要的光度距离表示加入到算法中。这对彩色图像滤波非常重要。如果彩色图像的三通道独立的分别进行滤波，在图像边缘处就会得到变质的色彩。实际上，不同的通道有不同的对比度层级，因此进行了不同的平滑。分别进行平滑会影响色彩平衡，所以会出现奇怪的色彩组合。双边滤波器则不会这样，在三个通道中一次性进行运算，可以显式的告诉滤波器，比如，哪种色彩是类似的，哪种不是。只有感官上类似的色彩是一起平均的，上面提到的伪影，则会消失。

The idea underlying bilateral filtering is to do in the range of an image what traditional filters do in its domain. Two pixels can be close to one another, that is, occupy nearby spatial location, or they can be similar to one another, that is, have nearby values, possibly in a perceptually meaningful fashion. Closeness refers to vicinity in the domain, similarity to vicinity in the range. Traditional filtering is domain filtering, and enforces closeness by weighing pixel values with coefficients that fall off with distance. Similarly, we define range filtering, which averages image values with weights that decay with dissimilarity. Range filters are nonlinear because their weights depend on image intensity or color. Computationally, they are no more complex than standard nonseparable filters. Most importantly, they preserve edges, as we show in section 4.

双边滤波器的思想，是将传统滤波器在其领域中做的事，在图像的领域中再做一次。两个像素可以互相接近，即，占据的空间位置是接近的，或可以互相很相似，即，有相近的灰度值，可能是以感官上有意义的方式。接近是指空间上位置上的接近，相似是指灰度值上的接近。传统滤波器是位置上的滤波，通过对一定距离内在像素进行加权平均，来加上接近性上的限制。类似的，我们定义的灰度值范围上的滤波，其平滑的图像值，是对灰度上不同相似度的像素进行不同的加权。灰度范围滤波器是非线性的，因此其权重依赖于图像灰度或颜色。计算上来说，他们不比标准的不可分割滤波器更复杂。更重要的是，滤波器会保护边缘，这我们在第4部分展示。

Spatial locality is still an essential notion. In fact, we show that range filtering by itself merely distorts an image’s color map. We then combine range and domain filtering, and show that the combination is much more interesting. We denote the combined filtering as bilateral filtering.

空间局部性仍然是一个必要的概念。实际上，我们展示了，图像灰度滤波本质上只是一幅图像的色彩图的扭曲。我们然后将灰度滤波和位置滤波结合到一起，表明其结合是更加有趣的。我们将两种滤波的结合称为双边滤波。

Since bilateral filters assume an explicit notion of distance in the domain and in the range of the image function, they can be applied to any function for which these two distances can be defined. In particular, bilateral filters can be applied to color images just as easily as they are applied to black-and-white ones. The CIE-Lab color space [16] endows the space of colors with a perceptually meaningful measure of color similarity, in which short Euclidean distances correlate strongly with human color discrimination performance [16]. Thus, if we use this metric in our bilateral filter, images are smoothed and edges are preserved in a way that is tuned to human performance. Only perceptually similar colors are averaged together, and only perceptually visible edges are preserved.

由于双边滤波假设图像函数中有位置和灰度量方面的显式的表示，所以可以应用到这两种距离可以进行定义的任何函数中。特别的，双边滤波可以应用到彩色图像中，与灰度图像的应用同样容易。CIE-Lab色彩空间使色彩空间拥有了感知上有意义的色彩相似度度量，其中欧式距离较小与人类看到的色彩差异度不大非常接近。因此，如果我们在双边滤波中使用这种度量，图像可以得到平滑，边缘得到保护，而且这是与人类的表现相协调的。只有感官上相似的色彩在一起平滑，只有感官上可见的边缘得到保护。

In the following section, we formalize the notion of bilateral filtering. Section 3 analyzes range filtering in isolation. Sections 4 and 5 show experiments for black-and-white and color images, respectively. Relations with previous work are discussed in section 6, and ideas for further exploration are summarized in section 7.

下面小节中，我们将双边滤波器的思想用公式进行表述。第3部分单独分析了灰度滤波。第4、5部分分别给出了灰度图像和彩色图像的试验。第6部分讨论了与之前的工作的关系，第7部分总结了更多的探索。

## 2. The Idea

A low-pass domain filter applied to image f(x) produces an output image defined as follows: 对图像f(x)应用的位置低通滤波器，生成的输出图像，定义如下：

$$h(x) = k_d^{-1} (x) \int_{-∞}^∞ \int_{-∞}^∞ f(ξ)c(ξ,x)dξ$$(1)

where $c(ξ,x)$ measures the geometric closeness between the neighborhood center x and a nearby point ξ. The bold font for f and h emphasizes the fact that both input and output images may be multiband. If low-pass filtering is to preserve the dc component of low-pass signals we obtain

其中$c(ξ,x)$衡量的是邻域中心x和附近点ξ在几何上非常接近。f和h的黑体表示，输入和输出图像可能是多通道的。如果低通滤波是保护低通信号的直流部分，那么我们可以得到

$$k_d(x) = \int_{-∞}^∞ \int_{-∞}^∞ c(ξ,x)dξ$$(2)

If the filter is shift-invariant, c(ξ,x) is only a function of the vector difference ξ-x, and k_d is a constant.

如果滤波器是平移不变的，那么c(ξ,x)应当只是向量差ξ-x的函数，k_d是一个常数。

Range filtering is similarly defined: 灰度滤波器的定义是类似的：

$$h(x) = k_r^{-1} (x) \int_{-∞}^∞ \int_{-∞}^∞ f(ξ)s(f(ξ),f(x))dξ$$(3)

except that now s(f(ξ),f(x)) measures the photometric similarity between the pixel at the neighbourhood center x and that of a nearby point ξ. Thus, the similarity function s operates in the range of the image function f, while the closeness function c operates in the domain of f. The normalization constant (2) is replaced by

除了现在s(f(ξ),f(x))度量的是邻域中心点x和附近点ξ的灰度差异。因此，相似性函数s是在图像函数f的范围内运算的，而接近函数c是在f的位置上进行运算的。(2)的归一化常数替换为

$$k_r(x) = \int_{-∞}^∞ \int_{-∞}^∞ s(f(ξ),f(x))dξ$$(4)

Contrary to what occurs with the closeness function c, the normalization for the similarity function s depends on the image f. We say that the similarity function s is unbiased if it depends only on the difference f(ξ)-f(x).

与接近函数c相反，相似度函数s的归一化依赖于函数f。我们说相似度函数s是无偏的，如果其只依赖于差值f(ξ)-f(x)。

The spatial distribution of image intensities plays no role in range filtering taken by itself. Combining intensities from the entire image, however, makes little sense, since image values far away from x ought not to affect the final value at x. In addition, section 3 shows that range filtering by itself merely changes the color map of an image, and is therefore of little use. The appropriate solution is to combine domain and range filtering, thereby enforcing both geometric and photometric locality. Combined filtering can be described as follows:

图像灰度的空间分布在灰度滤波中毫无作用。但将整个图像中的所有灰度结合到一起，基本没有多少意义，因为与x点距离很远的图像灰度值，不应当对x点的最终值有什么影响。另外，第3部分表明，灰度滤波本身，只是改变了图像的色彩图，因此本身是没什么作用的。合理的解决方案是，将位置滤波和灰度滤波结合到一起，因此要同时利用几何局部性和灰度局部性。联合滤波可以描述如下：

$$h(x) = k^{-1}(x) \int_{-∞}^∞ \int_{-∞}^∞ f(ξ) c(ξ,x) s(f(ξ),f(x)) dξ$$(5)

with the normalization: 包含归一化

$$k(x) = \int_{-∞}^∞ \int_{-∞}^∞ c(ξ,x) s(f(ξ),f(x)) dξ$$(6)

Combined domain and range filtering will be denoted as bilateral filtering. It replaces the pixel value at x with an average of similar and nearby pixel values. In smooth regions, pixel values in a small neighborhood are similar to each other, and the normalized similarity function k^{-1}s is close to one. As a consequence, the bilateral filter acts essentially as a standard domain filter, and averages away the small, weakly correlated differences between pixel values caused by noise. Consider now a sharp boundary between a dark and a bright region, as in figure 1(a). When the bilateral filter is centered, say, on a pixel on the bright side of the boundary, the similarity function s assumes values close to one for pixels on the same side, and close to zero for pixels on the dark side. The similarity function is shown in figure 1 (b) for a 23*23 filter support centered two pixels to the right of the step in figure 1 (a). The normalization term k(x) ensures that the weights for all the pixels add up to one. As a result, the filter replaces the bright pixel at the center by an average of the bright pixels in its vicinity, and essentially ignores the dark pixels. Conversely, when the filter is centered on a dark pixel, the bright pixels are ignored instead. Thus, as shown in figure 1 (c), good filtering behavior is achieved at the boundaries, thanks to the domain component of the filter, and crisp edges are preserved at the same time, thanks to the range component.

将位置滤波与灰度滤波结合到一起，我们称之为双边滤波。它将x处的像素值替换为，附近的、相似的像素值的平均。在平滑区域，很小邻域内的像素值是互相很相似的，归一化的相似度函数k^{-1}s接近于1。结果是，双边滤波器实际上就是一个标准的位置滤波器，将很小的、弱相关的像素值之间的噪声导致的差异平滑掉。现在考虑一个暗色区域和亮色区域之间的陡峭边缘，如图1(a)所示。当双边滤波器的中心在亮色处的边缘处，相似度函数s假设接近于1的像素值是在同一侧的，而接近于0的像素是在暗色处的。相似度函数如图1(b)处所示，滤波器支撑为23*23大小，其中心是在图1(a)中阶跃的右侧两个像素处。归一化项k(x)确保了，所有像素的权重求和为1。结果是，滤波器将亮色点替换为亮色点一边处的像素的平均，实际上忽略了较暗的像素。相反的，当滤波器中心在暗色一边时，较亮的像素会被忽略掉。因此，如图1(c)所示，在边缘处得到了好的滤波器行为，多亏了滤波器的位置元素；而同时保持了很好的边缘，这多亏了位置元素。

### 2.1 Example: the Gaussian Case

A simple and important case of bilateral filtering is shift-invariant Gaussian filtering, in which both the closeness function c(ξ,x) and the similarity function s(ϕ,f) are Gaussian functions of the Euclidean distance between their arguments. More specifically, c is radially symmetric

双边滤波器的一种简单和重要的情况，是平移不变的高斯滤波，其中接近函数c(ξ,x)和相似度函数s(ϕ,f)都是参数之间的欧式距离的高斯函数。更具体的，c是径向对称的：

$$c(ξ,x) = e^{-\frac{1}{2} (\frac {d(ξ,x)} {σ_d})^2}$$

where

$$d(ξ,x) = d(ξ-x) = ||ξ-x||$$

is the Euclidean distance between ξ and x. The similarity function s is perfectly analogous to c: 是ξ和x之间的欧式距离。相似度函数s与c是完全类似的：

$$s(ξ,x) = e^{-\frac {1}{2} (\frac {δ(f(ξ),f(x))} {σ_r})^2} \delta$$

where

$$δ(ϕ,f) = δ(ϕ-f) = ||ϕ-f||$$

is a suitable measure of distance between the two intensity values ϕ and f. In the scalar case, this may be simply the absolute difference of the pixel difference or, since noise increases with image intensity, an intensity-dependent version of it. A particularly interesting example for the vector case is given in section 5.

适合于两个灰度值ϕ和f之间的距离度量。在标量的情况下，这可能就是简单的像素差的绝对值，或者由于噪声随着图像灰度增加的话，可能是与灰度无关的版本。第5节会给出一个特别有趣的向量版的函数。

The geometric spread σ_d in the domain is chosen based on the desired amount of low-pass filtering. A large σ_d blurs more, that is, it combines values from more distant image locations. Also, if an image is scaled up or down, σ_d must be adjusted accordingly in order to obtain equivalent results. Similarly, the photometric spread σ_r in the image range is set to achieve the desired amount of combination of pixel values. Loosely speaking, pixels with values much closer to each other than σ_r are mixed together and values much more distant than σ_r are not. If the image is amplified or attenuated, σ_r must be adjusted accordingly in order to leave the results unchanged.

位置上的几何延展σ_d是基于期望的低通滤波来选择的。大的σ_d模糊的更多，即，结合了更远处图像位置上的像素。同时，如果图像进行了缩放，σ_d必须也要相应的进行调整，以得到等价的结果。类似的，灰度延展σ_r设置为合适数量的灰度值的结合。粗略的说，像素值比σ_r更加互相接近的混合在一起，比σ_r大很多的值则没有。如果图像进行了放大或衰减，σ_r也需要相应的调整，以使结果不变。

Just as this form of domain filtering is shift-invariant, the Gaussian range filter introduced above is insensitive to overall additive changes of image intensity, and is therefore unbiased: if filtering f(x) produces h(x), then the same filter applied to f(x)+a yields h(x)+a, since δ(f(ξ)+a, f(x)+a) = δ(f(ξ)+a - (f(x)+a)) = δ(f(ξ)-f(x)). Of course, the range filter is shift-invariant as well, as can be easily verified from expressions (3) and (4).

就像位置滤波的这种形式是平移不变的，上面给出的高斯位置滤波对整体的图像恢复的加性变化是不敏感的，因此也是无偏的：如果对f(x)的滤波产生了h(x)，然后对f(x)+a的同样的滤波会产生h(x)+a，由于δ(f(ξ)+a, f(x)+a) = δ(f(ξ)+a - (f(x)+a)) = δ(f(ξ)-f(x))。当然，灰度滤波器也是平移不变的，这从式(3)和(4)可以很容易的进行验证。

## 3. Range Versus Bilateral Filtering

In the previous section we combined range filtering with domain filtering to produce bilateral filters. We now show that this combination is essential. For notational simplicity, we limit our discussion to black-and-white images, but analogous results apply to multiband images as well. The main point of this section is that range filtering by itself merely modifies the gray map of the image it is applied to. This is a direct consequence of the fact that a range filter has no notion of space.

在前一节，我们将灰度滤波与位置滤波结合到一起，以生成双向滤波。我们现在给出，这种结合是必须的。对表示简单，我们将讨论限制在黑白图像中，但类似的结果也可以用于多通道图像。这一节的主要观点是，灰度滤波器本身，只是对图像的灰度图进行了修正。因为灰度滤波器没有空间的概念，这直接导致了这个结果。

Let ν(ϕ) be the frequency distribution of gray levels in the input image. In the discrete case, ν(ϕ) is the gray level histogram: ϕ is typically an integer between 0 and 255, and ν(ϕ) is the fraction of image pixels that have a gray value of ϕ. In the continuous case, ν(ϕ)dϕ is the fraction of image area whose gray values are between ϕ and ϕ+dϕ. For notational consistency, we continue our discussion in the continuous case, as in the previous section.

令ν(ϕ)为输入图像的灰度的频率分布。在离散情况下，ν(ϕ)是灰度级的直方图：ϕ一般是0到255中的一个整数，ν(ϕ)是灰度值为ϕ的像素的比重。在连续的情况下，ν(ϕ)dϕ是灰度值在ϕ和dϕ之间的像素比例。为表示上的一致性，我们在连续情况下进行讨论，和上一节一样。

Simple manipulation, omitted for lack of space, shows that expressions (3) and (4) for the range filter can be combined into the following:

简单的运算，忽略缺少的空间，表明恢复滤波中的式(3)和(4)可以结合成：

$$h = \int_0^∞ ϕτ(ϕ,f)dϕ$$(7)

where

$$τ(ϕ,f) = \frac {s(ϕ,f)ν(ϕ)} {\int_0^∞ s(ϕ,f)ν(ϕ)dϕ}$$

independently of the position x. Equation (7) shows range filtering to be a simple transformation of gray levels. The mapping kernel τ(ϕ,f) is a density function, in the sense that it is nonnegative and has unit integral. It is equal to the histogram ν(ϕ) weighted by the similarity function s centered at f and normalized to unit area. Since τ is formally a density function, equation (7) represents a mean. We can therefore conclude with the following result:

是与位置x无关的。式(7)表明，灰度滤波器就是灰度级的简单变换。映射核τ(ϕ,f)是一个密度函数，因为其是非负的而且积分值为1。这与直方图ν(ϕ)经过加权是一样的，权值为相似度函数s以f为中心，归一化到单位区域。由于τ是一个密度函数，式(7)表示一个均值。因此我们可以得到下面的结论：

Range filtering merely transforms the gray map of the input image. The transformed gray value is equal to the mean of the input’s histogram values around the input gray level f, weighted by the range similarity function s centered at f.

灰度滤波器只是输入图像的灰度图的变换。变换的灰度图与输入直方图在输入灰度级f附近的均值相等，权重为灰度相似度函数s在以f为中心的值。

It is useful to analyze the nature of this gray map transformation in view of our discussion of bilateral filtering. Specifically, we want to show that 分析这种灰度图变换的本质，对于我们讨论双边滤波是有用的。具体的，我们要展示的是

Range filtering compresses unimodal histograms. 灰度滤波压缩了单峰的直方图。

In fact, suppose that the histogram ν(ϕ) of the input image is a single-mode curve as in figure 2(a), and consider an input value of f located on either side of this bell curve. Since the symmetric similarity function s is centered at f, on the rising flank of the histogram, the product sν produces a skewed density τ(ϕ,f). On the left side of the bell τ is skewed to the right, and vice versa. Since the transformed value h is the mean of this skewed density, we have h > f on the left side and h < f on the right side. Thus, the flanks of the histogram are compressed together.

实际上，假设输入图像的灰度直方图ν(ϕ)是一个单峰的曲线，如图2(a)所示，考虑一个输入值f，在这个钟形曲线的任意一侧。由于相似度函数s是对称的，中心在f，在直方图的上升侧，sν的乘积生成的是歪曲的密度τ(ϕ,f)。在钟形的左侧，τ就向右偏斜，反之亦然。由于变换值h是这个偏斜密度的均值，我们有在左侧h > f，在右侧h < f。因此，直方图的两侧压缩到的一起。

At first, the result that range filtering is a simple remapping of the gray map seems to make range filtering rather useless. Things are very different, however, when range filtering is combined with domain filtering to yield bilateral filtering, as shown in equations (5) and (6). In fact, consider
first a domain closeness function c that is constant within a window centered at x, and is zero elsewhere. Then, the bilateral filter is simply a range filter applied to the window. The filtered image is still the result of a local remapping of the gray map, but a very interesting one, because the remapping is different at different points in the image.

第一，灰度滤波的结果，是灰度图的简单重映射，这似乎使得灰度滤波没什么用处。但当灰度滤波与位置滤波结合到一起时，形成双边滤波，如式(5)和(6)，就非常不一样了。实际上，考虑第一，位置接近函数c，在以x为中心的窗口中是一个常数，在其他位置中为0。那么，双边滤波器只是应用到窗口的灰度滤波器。滤波后的图像，仍然只是灰度图的局部重映射，但是非常有趣的一种，因此这种重映射在图像的不同位置点是不一样的。

For instance, the solid curve in figure 2(b) shows the histogram of the step image of figure 1 (a). This histogram is bimodal, and its two lobes are sufficiently separate to allow us to apply the compression result above to each lobe. The dashed line in figure 2 (b) shows the effect of bilateral filtering on the histogram. The compression effect is obvious, and corresponds to the separate smoothing of the light and dark sides, shown in figure 1 (c). Similar considerations apply when the closeness function has a profile other than constant, as for instance the Gaussian profile shown in section 2, which emphasizes points that are closer to the center of the window.

比如，图2(b)中的实线是图1(a)中的阶跃图像的直方图。这个直方图是双峰的，其两叶是充分分开的，以使我们对每一叶使用上面的压缩结果。图2(b)中的虚线，给出了双边滤波器在灰度直方图上的效果。压缩效果是很显然的，对应着对亮边和暗边的平滑效果，如图1(c)所示。当接近度函数有一个曲线，而不是常数时，也有同样的考虑，比如第2节中的高斯曲线，强调了与接近窗口中间位置的点。

## 4 Experiments with Black-and-White Images

In this section we analyze the performance of bilateral filters on black-and-white images. Figure 5 (a) and 5 (b) in the color plates show the potential of bilateral filtering for the removal of texture. Some amount of gray-level quantization can be seen in figure 5 (b), but this is caused by the printing process, not by the filter. The picture “simplification” illustrated by figure 5 (b) can be useful for data reduction without loss of overall shape features in applications such as image transmission, picture editing and manipulation, image description for retrieval. Notice that the kitten’s whiskers, much thinner than the filter’s window, remain crisp after filtering. The intensity values of dark pixels are averaged together from both sides of the whisker, while the bright pixels from the whisker itself are ignored because of the range component of the filter. Conversely, when the filter is centered somewhere on a whisker, only whisker pixel values are averaged together.

本节中，我们分析了双边滤波器在灰度图像上的性能。图5(a)和(b)表明了双边滤波对去除纹理的效果。图5(b)中可以看到一些灰度级量化，但这是由于打印的过程形成的，并不是滤波器造成的。图5(b)的简化效果对于不损失总体形状特征的数据压缩是有用的，这可以用于图像传输、图像编辑和处理，图像检索中的图像描述。注意猫咪的胡须，比滤波器的窗口小很多，在滤波后仍然保持尖锐。暗色像素的灰度值被忽略了，因为滤波器的灰度部分。相反的，当滤波器在胡须的某个位置为中心时，只有胡须像素值得到平均。

Figure 3 shows the effect of different values of the parameters σ_d and σ_r on the resulting image. Rows correspond to different amounts of domain filtering, columns to different amounts of range filtering. When the value of the range filtering constant σ_r is large (100 or 300) with respect to the overall range of values in the image (1 through 254), the range component of the filter has little effect for small σ_d: all pixel values in any given neighborhood have about the same weight from range filtering, and the domain filter acts as a standard Gaussian filter. This effect can be seen in the last two columns of figure (3). For smaller values of the range filter parameter σ_r (10 or 30), range filtering dominates perceptually because it preserves edges.

图3表明σ_d和σ_r不同参数值对得到的图像的效果。行对应着不同的位置滤波的效果，列对应着不同的灰度滤波量的效果。当灰度滤波常数σ_r的值相对于图像中灰度值的总体范围(1到254)很大(100或300)时，滤波器的灰度部分在σ_d较小时基本没有效果：所有像素值在任何给定的邻域中都从灰度滤波中得到了相同的权重，位置滤波的作用只不过是一个标准的高斯滤波器。这种效果可以在图(3)中的后两列中看到。对于灰度滤波参数σ_r的较小的值(10或30)，灰度滤波在感官上起主要作用，因为其保留了边缘。

However, for σ_d = 10, image details that was removed by smaller values of σ_d reappears. This apparently paradoxical effect can be noticed in the last row of figure 3, and in particularly dramatic form for σ_r=100, σ_d=10. This image is crisper than that above it, although somewhat hazy. This is a consequence of the gray map transformation and histogram compression results discussed in section 3. In fact, σ_d = 10 is a very broad Gaussian, and the bilateral filter becomes essentially a range filter. Since intensity values are simply remapped by a range filter, no loss of detail occurs. Furthermore, since a range filter compresses the image histogram, the output image appears to be hazy. Figure 2 (c) shows the histograms for the input image and for the two output images for σ_r=100, σ_d=3, and for σ_r=100, σ_d=10. The compression effect is obvious.

但是，对于σ_d = 10，由更小的σ_d值去除的图像细节重新出现了。这种显然矛盾的效果可以在图3的最后一行中看到，而且对于σ_r=100, σ_d=10的情况尤其戏剧化。这幅图像比其上面的图像更脆，虽然有一些模糊。这是灰度图变换和直方图压缩的结果，在第3节中讨论过。实际上，σ_d=10是一个很广的高斯函数，双边滤波实际上变成了一个灰度滤波。由于灰度值只是被灰度滤波器简单的重新映射，所以不会损失任何细节。而且，由于灰度滤波压缩了图像的直方图，输出图像看起来比较模糊。图2(c)展示了输入图像和输出图像在σ_r=100, σ_d=3, 和σ_r=100, σ_d=10的直方图。压缩的效果是很明显的。

Bilateral filtering with parameters σ_d = 3 pixels and σ_r = 50, intensity values is applied to the image in figure 4(a) to yield the image in figure 4(b). Notice that most of the fine texture has been filtered away, and yet all contours are as crisp as in the original image.

图4(a)的图像应用了参数σ_d = 3, σ_r = 50的双边滤波，得到了图4(b)的图像。注意，多数细节纹理都被滤除掉了，但所有边缘都与原始图像一样犀利。

Figure 4(c) shows a detail of figure 4(a), and figure 4(d) shows the corresponding filtered version. The two onions have assumed a graphics-like appearance, and the fine texture has gone. However, the overall shading is preserved, because it is well within the band of the domain filter and is almost unaffected by the range filter. Also, the boundaries of the onions are preserved.

图4(c)给出了图4(a)的一个局部细节，图4(d)给出了对应的滤波后的版本。这两个洋葱外观相似，但细节纹理都去掉了。但是，总体的阴影得到了保持，因为这是在位置滤波器的范围内，而并没有受到灰度滤波器影响。而且，洋葱的边缘得到了保持。

In terms of computational cost, the bilateral filter is twice as expensive as a nonseparable domain filter of the same size. The range component depends nonlinearly on the image, and is nonseparable. A simple trick that decreases computation cost considerably is to precompute all values for the similarity function s(ϕ,f). In the Gaussian case, if the image has n levels, there are 2n+1 possible values for s, one for each possible value of the difference ϕ-f.

以计算代价而论，双边滤波器是同样大小的不可分割的位置滤波器的两倍。灰度部分与图像的关系是非线性的，而且是不可分割的。一个简单的降低计算量的技巧，是对相似度函数s(ϕ,f)预先计算所有值。在高斯的情况下，如果图像有n个灰度级，那么s就有2n+1个可能的值，对于差值ϕ-f的每个可能的值都有一个值。

## 5. Experiments with Color Images

For black-and-white images, intensities between any two grey levels are still grey levels. As a consequence, when smoothing black-and-white images with a standard low-pass filter, intermediate levels of gray are produced across edges, thereby producing blurred images. With color images, an additional complication arises from the fact that between any two colors there are other, often rather different colors. For instance, between blue and red there are various shades of pink and purple. Thus, disturbing color bands may be produced when smoothing across color edges. The smoothed image does not just look blurred, it also exhibits odd-looking, colored auras around objects. Figure 6 (a) in the color plates shows a detail from a picture with a red jacket against a blue sky. Even in this unblurred picture, a thin pink-purple line is visible, and is caused by a combination of lens blurring and pixel averaging. In fact, pixels along the boundary, when projected back into the scene, intersect both red jacket and blue sky, and the resulting color is the pink average of red and blue. When smoothing, this effect is emphasized, as the broad, blurred pink-purple area in figure 6 (b) shows.

对于灰度图像，两个灰度级之间仍然是灰度。结果是，当对黑白图像用低通滤波进行平滑时，在边缘两边会产生中间的灰度级，因此产生模糊的图像。对于彩色图像，会产生额外的困难，因为两个色彩之间通常会有很多其他不同的色彩。比如，在蓝色和红色之间，有很多不同的粉色和紫色。因此，当在彩色边缘附近进行滤波时，会产生一些干扰的色彩通道。平滑的图像并不只是看起来模糊，还会有一些奇怪的样子，在目标周围有一些彩色的光环。图6(a)展示了一幅图像的细节，在蓝色天空的周围有一个红色的jacket。即使在这个不模糊的图像中，也可以看到一条细细的粉紫色的线，这是由镜头模糊和像素平均导致的。实际上，在边缘附近的像素，当投影回场景时，与红色jacket和蓝色天空都是相交的，得到的色彩是红色和蓝色的平均，即粉色。当平滑时，这种效果得到了加强，如图6(b)的模糊的粉紫色所描述。

To address this difficulty, edge-preserving smoothing could be applied to the red, green, and blue components of the image separately. However, the intensity profiles across the edge in the three color bands are in general different. Separate smoothing results in an even more pronounced pink-purple band than in the original, as shown in figure 6 (c). The pink-purple band, however, is not widened as it is in the standard-blurred version of figure 6 (b).

为处理这种困难，保护边缘的平滑可以分别应用到图像的红色、绿色和蓝色通道。但是，在三个色彩通道中的边缘的亮度曲线，一般是不同的。分离的平滑结果产生了比原始图像更加明显的粉紫色条，如图6(c)所示。但粉紫色条并没有加宽，而在图6(b)中的标准模糊的版本中则有加宽效果。

A much better result can be obtained with bilateral filtering. In fact, a bilateral filter allows combining the three color bands appropriately, and measuring photometric distances between pixels in the combined space. Moreover, this combined distance can be made to correspond closely to perceived dissimilarity by using Euclidean distance in the CIE-Lab color space [16]. This space is based on a large body of psychophysical data concerning color-matching experiments performed by human observers. In this space, small Euclidean distances correlate strongly with the perception of color discrepancy as experienced by an “average” color-normal human observer. Thus, in a sense, bilateral filtering performed in the CIE-Lab color space is the most natural type of filtering for color images: only perceptually similar colors are averaged together, and only perceptually important edges are preserved. Figure 6 (d) shows the image resulting from bilateral smoothing of the image in figure 6 (a). The pink band has shrunk considerably, and no extraneous colors appear.

采用双边滤波则可以得到更加好的结果。实际上，双边滤波可以将三个色彩通道合适的结合起来，将像素之间的光度距离在结合的空间中进行度量。而且，这种结合的距离，与在CIE-Lab色彩空间中的欧式距离感受到的不相似性，可以很紧密的对应起来。这个空间是基于主要的心理生理学的数据的，与人类观察者进行的色彩匹配试验有很大关系。在这个空间中，欧式距离小，与色彩差异性的感知是相关的，这里是指由平均的色彩正常的人类观察者得到的结果。因此，在某种意义上，在CIE-Lab色彩空间中进行的双边滤波是对彩色图像进行的最自然的滤波：只有感官上相似的色彩在一起进行平均，只有感官上重要的边缘得到了保留。图6(d)给出了图6(a)双边滤波平滑得到的结果。粉色带缩小了很多，没有额外的色彩出现。

Figure 7 (c) in the color plates shows the result of five iterations of bilateral filtering of the image in figure 7 (a). While a single iteration produces a much cleaner image (figure 7 (b)) than the original, and is probably sufficient for most image processing needs, multiple iterations have the effect of flattening the colors in an image considerably, but without blurring edges. The resulting image has a much smaller color map, and the effects of bilateral filtering are easier to see when displayed on a printed page. Notice the cartoon-like appearance of figure 7 (c). All shadows and edges are preserved, but most of the shading is gone, and no “new” colors are introduced by filtering.

图7(c)是图7(a)经过5次迭代双边滤波的结果。一次迭代产生了比原始图像更干净的图像，这对大多数图像处理的需求是足够的，多次迭代的效果是，将图像中的色彩有了相当的平滑，但并没有模糊边缘。得到的图像的色彩图更小，双边滤波的效果在打印出来的页面上更容易看到。注意图7(c)中比较像卡通的效果。所有的阴影和边缘都得到了保留，但多数阴影都没有了，而且滤波并没有引入新的色彩。

## 6 Relations with Previous Work

The literature on edge-preserving filtering is vast, and we make no attempt to summarize it. An early survey can be found in [8], quantitative comparisons in [2], and more recent results in [1]. In the latter paper, the notion that neighboring pixels should be averaged only when they are similar enough to the central pixels is incorporated into the definition of the so-called “G-neighbors.” Thus, G-neighbors are in a sense an extreme case of our method, in which a pixel is either counted or it is not. Neighbors in [1] are strictly adjacent pixels, so iteration is necessary.

保护边缘的滤波的文献很多，我们并不打算对其进行总结。[8]是一个早期的总结，[2]中有定量的对比，[1]是最近的结果。在后面的文章中，相邻的像素应当只在与中心像素足够类似的时候进行平均，这个概念与所谓的G-邻域的概念结合到了一起。因此，G-邻域是我们方法中的一个极端情况，其中一个像素要么得到了计数，要么没有得到。[1]中的邻域是严格相邻的像素，所以迭代是必须的。

A common technique for preserving edges during smoothing is to compute the median in the filter’s support, rather than the mean. Examples of this approach are [6,9], and an important variation [3] that uses K-means instead of medians to achive greater robustness.

滤波的同时保持边缘的一种常见技术是，计算滤波器支撑的中值，而不是其均值。这种方法的例子如[6,9]，一种重要的变体[3]使用K-均值，而没有使用中值，得到了更好的稳健性。

More related to our approach are weighting schemes that essentially average values within a sliding window, but change the weights according to local differential [4, 15] or statistical [10, 7] measures. Of these, the most closely related article is [10], which contains the idea of multiplying a geometric and a photometric term in the filter kernel. However, that paper uses rational functions of distance as weights, with a consequent slow decay rate. This forces application of the filter to only the immediate neighbors of every pixel, and mandates multiple iterations of the filter. In contrast, our bilateral filter uses Gaussians as a way to enforce what Overton and Weimouth call “center pixel dominance.” A single iteration drastically “cleans” an image of noise and other small fluctuations, and preserves edges even when a very wide Gaussian is used for the domain component. Multiple iterations are still useful in some circumstances, as illustrated in figure 7 (c), but only when a cartoon-like image is desired as the output. In addition, no metrics are proposed in [10] (or in any of the other papers mentioned above) for color images, and no analysis is given of the interaction between the range and the domain components. Our discussions in sections 3 and 5 address both these issues in substantial detail.

与我们的方法更加相关的是，在一个滑窗中对像素进行平均，其加权方案的确定，有的权重的变化是根据局部微分，或统计方法。当然，最相关的文章是[10]，包含的思想是在滤波核中乘以一个几何项和光度项。但是，那篇文章使用的rational函数的距离作为权重，衰减率因此非常低。这使滤波器的应用，只是到了每个像素的很小的邻域，并对滤波进行迭代了多次。对比起来，我们的双边滤波器使用高斯函数作为中间像素主导的因素。单次迭代极大的清除了图像的噪声，和其他的波动，即使在很宽的高斯函数应用时，也保持了边缘。多次迭代在一些场合下仍然有用，如图7(c)所示，但只在期望输出一种像卡通的图像时。另外，[10]中并没有给出对彩色图像的度量（或在上述提到的任何文章中），而且也没有给出位置滤波和灰度滤波部分的交互的分析。我们在第3和第5部分的讨论，以很细节的方式讨论了这两个问题。

## 7 Conclusions

In this paper we have introduced the concept of bilateral filtering for edge-preserving smoothing. The generality of bilateral filtering is analogous to that of traditional filtering, which we called domain filtering in this paper. The explicit enforcement of a photometric distance in the range component of a bilateral filter makes it possible to process color images in a perceptually appropriate fashion.

本文中，我们提出了双边滤波的概念，可以进行保持边缘的平滑。双边滤波的一般性可以与传统滤波类比，我们在本文中称之为位置滤波。双边滤波器中的灰度滤波部分，加入了光度距离的部分，这使其可以以一种感知上很合适的方式处理彩色图像。

The parameters used for bilateral filtering in our illustrative examples were to some extent arbitrary. This is however a consequence of the generality of this technique. In fact, just as the parameters of domain filters depend on image properties and on the intended result, so do those of bilateral filters. Given a specific application, techniques for the automatic design of filter profiles and parameter values may be possible.

我们的例子的双边滤波中的参数的使用，是比较随意的。但这是这种技术的一般性的结果。实际上，就像位置滤波器依赖于图像性质和期望的结果，双边滤波器也是一样的。给定特定的应用，自动涉及滤波器响应曲线的参数值是可能的。

Also, analogously to what happens for domain filtering, similarity metrics different from Gaussian can be defined for bilateral filtering as well. In addition, range filters can be combined with different types of domain filters, including oriented filters. Perhaps even a new scale space can be defined in which the range filter parameter σ_r corresponds to scale. In such a space, detail is lost for increasing σ_r, but edges are preserved at all range scales that are below the maximum image intensity value. Although bilateral filters are harder to analyze than domain filters, because of their nonlinear nature, we hope that other researchers will find them as intriguing as they are to us, and will contribute to their understanding.

同时，与在位置滤波中发生的类似，与高斯函数不同的相似度度量，也可以定义并用于双边滤波。另外，灰度滤波可以与不同类型的位置滤波相结合，包括定向滤波器。甚至可以定义一种新的尺度空间，其中灰度滤波参数σ_r对应着尺度。在这种空间中，增加σ_r会损失细节，但在所有灰度尺度上（在图像最大灰度值之下），边缘都会得到保持。虽然双边滤波器比位置滤波器更难分析，由于其非线性的本质，我们希望其他研究者会发现之。