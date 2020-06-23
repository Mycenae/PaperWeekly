# A non-local algorithm for image denoising

## 0. Abstract

We propose a new measure, the method noise, to evaluate and compare the performance of digital image denoising methods. We first compute and analyze this method noise for a wide class of denoising algorithms, namely the local smoothing filters. Second, we propose a new algorithm, the non local means (NL-means), based on a non local averaging of all pixels in the image. Finally, we present some experiments comparing the NL-means algorithm and the local smoothing filters.

我们提出了一种新的度量，the method noise，来评估比较数字图像去噪方法的性能。我们对很多类去噪算法首先计算和分析了这种method noise，这些都是局部平滑滤波器。第二，我们提出了一种新的算法，non-local means，这是基于对图像中所有像素的非局部平均。最后，我们提出一些试验，比较了NL-means算法和局部平滑滤波器。

## 1. Introduction

The goal of image denoising methods is to recover the original image from a noisy measurement, 图像去噪方法的目标，是从含噪的度量中恢复出原始图像：

$$v(i) = u(i) + n(i)$$(1)

where v(i) is the observed value, u(i) is the “true” value and n(i) is the noise perturbation at a pixel i. The best simple way to model the effect of noise on a digital image is to add a gaussian white noise. In that case, n(i) are i.i.d. gaussian  values with zero mean and variance σ^2.

其中v(i)是在像素i处观察到的值，u(i)是在像素i处的真值，n(i)是在像素i处的噪声扰动。对图像中噪声的效果进行建模的最好最简单方式，是加上一个高斯白噪声。在这种情况下，n(i)是iid的高斯值，零均值，方差为σ^2。

Several methods have been proposed to remove the noise and recover the true image u. Even though they may be very different in tools it must be emphasized that a wide class share the same basic remark: denoising is achieved by averaging. This averaging may be performed locally: the Gaussian smoothing model (Gabor [7]), the anisotropic filtering (Perona-Malik [11], Alvarez et al. [1]) and the neighborhood filtering (Yaroslavsky [16], Smith et al. [14], Tomasi et al. [15]), by the calculus of variations: the Total Variation minimization (Rudin-Osher-Fatemi [13]), or in the frequency domain: the empirical Wiener filters (Yaroslavsky [16]) and wavelet thresholding methods (Coiffman-Donoho [5, 4]).

提出了几种方法来去除噪声，恢复出真实图像u。虽然采用的工具都很不一样，但是必须强调，这些算法有一个很大的共同点：去噪是通过平滑来进行的。这种平滑可能是局部进行的：高斯平滑模型，各向异性滤波，邻域滤波，采用变分的方法：全变分最小化，或在频域中的：维纳滤波器，小波阈值方法。

Formally we define a denoising method $D_h$ as a decomposition, 这里我们将一种去噪方法$D_h$正式的定义为一种分解

$$v = D_h v + n(D_h, v)$$

where v is the noisy image and h is a filtering parameter which usually depends on the standard deviation of the noise. Ideally, $D_h v$ is smoother than v and $n(D_h, v)$ looks like the realization of a white noise. The decomposition of an image between a smooth part and a non smooth or oscillatory part is a current subject of research (for example Osher et al. [10]). In [8], Y. Meyer studied the suitable functional spaces for this decomposition. The primary scope of this latter study is not denoising since the oscillatory part contains both noise and texture.

其中v是含噪图像，h是滤波参数，通常依赖噪声的标准方差。理想情况下，$D_h v$比v更平滑，$n(D_h, v)$看起来更像是白噪声的实现。图像分解成一个平滑的部分，和一个不平滑或震荡的部分，是目前研究的主题。[8]中，Y. Meyer等研究了这种分解的合适的函数空间。后者的研究的主要范围，并不是去噪，因为震荡部分包含了噪声和纹理。

The denoising methods should not alter the original image u. Now, most denoising methods degrade or remove the fine details and texture of u. In order to better understand this removal, we shall introduce and analyze the method noise. The method noise is defined as the difference between the original (always slightly noisy) image u and its denoised version.

去噪方法不应当改变原始图像u。现在，多数去噪方法都将u中的细节和纹理去掉了，或减少了。为更好的理解这种去除，我们提出并分析method noise。Method noise定义为原始图像u和去噪版之间的差别。

We also propose and analyze the NL-means algorithm, which is defined by the simple formula 我们还提出并分析了NL-means算法，定义为下面的简单公式

$$NL[u] (x) = \frac{1}{C(x)} \int_Ω e^{-\frac {(G_a*|u(x+.)-u(y+.)|^2)(0)} {h^2}} u(y)dy$$

where $x∈Ω, C(x) = \int_Ω e^{-\frac {(G_a*|u(x+.)-u(y+.)|^2)(0)} {h^2}} dz$ is a normalizing constant, $G_a$ is a Gaussian kernel and h acts as a filtering parameter. This formula amounts to say that the denoised value at x is a mean of the values of all points whose gaussian neighborhood looks like the neighborhood of x. The main difference of the NL-means algorithm with respect to local filters or frequency domain filters is the systematic use of all possible self-predictions the image can provide, in the spirit of [6]. For a more detailed analysis on the NL-means algorithm and a more complete comparison, see [2].

其中$x∈Ω, C(x) = \int_Ω e^{-\frac {(G_a*|u(x+.)-u(y+.)|^2)(0)} {h^2}} dz$是一个归一化常数，$G_a$是一个高斯核，h是一个滤波参数。这个公式是说，在x处的去噪值，是一些点的均值，这些点的高斯邻域与x的邻域类似。NL-means算法与局部滤波器或频域滤波器的主要区别，是图像可以提供的所有可能的自预测点的系统使用，这是源于[6]的思想。NL-means算法更细节的分析和更全面的比较，见[2]。

Section 2 introduces the method noise and computes its mathematical formulation for the mentioned local smoothing filters. Section 3 gives a discrete definition of the NL-means algorithm. In section 4 we give a theoretical result on the consistency of the method. Finally, in section 5 we compare the performance of the NL-means algorithm and the local smoothing filters.

第2部分介绍了method noise，并对提到的局部平滑滤波器计算了其数学公式。第3部分NL-means算法的离散定义。第4部分，我们给出了这种方法的理论结果的一致性。最后，在第5部分，我们比较了NL-means算法和局部平滑滤波器的性能。

## 2. Method noise

**Definition 1 (Method noise)**. Let u be an image and $D_h$ a denoising operator depending on a filtering parameter h. Then, we define the method noise as the image difference 令u为图像，$D_h$是一个去噪算子，滤波参数为h。那么，我们定义method noise为下面的图像差

$$u − D_h u$$

The application of a denoising algorithm should not alter the non noisy images. So the method noise should be very small when some kind of regularity for the image is assumed. If a denoising method performs well, the method noise must look like a noise even with non noisy images and should contain as little structure as possible. Since even good quality images have some noise, it makes sense to evaluate any denoising method in that way, without the traditional “add noise and then remove it” trick. We shall list formulas permitting to compute and analyze the method noise for several classical local smoothing filters: the Gaussian filtering [7], the anisotropic filtering [1, 11], the Total Variation minimization [13] and the neighborhood filtering [16]. The formal analysis of the method noise for the frequency domain filters fall out of the scope of this paper. These method noises can also be computed but their interpretation depends on the particular choice of the wavelet basis.

去噪算法的应用，不应当改变无噪声图像。所以如果假设图像有某种规则性，那么method noise就应当很小。如果一个去噪方法表现很好，method noise就应当看起来像是噪声，所包含的结构信息应当尽量的少。由于质量很好的图像也有一些噪声，所以以这种方式评估任何去噪算法都是可以的，而不需要对图像增加噪声，然后去除掉。我们对几种经典的局部平滑滤波器，列出其计算method noise的公式：高斯滤波器，各向异性滤波，全变分最小化滤波和邻域滤波。对于频域滤波器的method noise的正式分析，不在本文范畴之内。这些method noise也可以计算，但其解释依赖于小波基的具体选择。

### 2.1. The Gaussian filtering

The image isotropic linear filtering boils down to the convolution of the image by a linear symmetric kernel. The paradigm of such kernels is of course the gaussian kernel $x → G_h(x) = \frac {1}{4πh^2} e^{-\frac{|x|^2}{4h^2}}$. In that case, $G_h$ has standard deviation h and it is easily seen that

图像各向同性线性滤波，实际上就是将图像与线性对称核进行卷积。这种核的范例，当然就是高斯核，$x → G_h(x) = \frac {1}{4πh^2} e^{-\frac{|x|^2}{4h^2}}$。这种情况下，$G_h$有标准差h，很容易可以看出

**Theorem 1 (Gabor 1960)** The image method noise of the convolution with a gaussian kernel $G_h$ is 与高斯核$G_h$卷积的图像method noise是

$$u − G_h * u = −h^2∆u + o(h^2)$$

for h small enough. 这是对h足够小的情况的结果。

The gaussian method noise is zero in harmonic parts of the image and very large near edges or texture, where the Laplacian cannot be small. As a consequence, the Gaussian convolution is optimal in flat parts of the image but edges and texture are blurred.

高斯method noise在图像的平滑部分为0，在接近边缘或纹理的时候很大，在这附近的Laplacian也不会很小。结果是，在图像的平滑部分，高斯卷积是最优的，但边缘和纹理则模糊了。

### 2.2. The anisotropic filtering

The anisotropic filter (AF) attempts to avoid the blurring effect of the Gaussian by convolving the image u at x only in the direction orthogonal to Du(x). The idea of such filter goes back to Perona and Malik [11]. It is defined by

各向异性滤波器(AF)试图避免高斯滤波的模糊效果，在x处对u进行卷积时，只在与Du(x)的方向进行。这种滤波器的思想要追溯到Perona and Malik [11]。定义为

$$AF_h u(x) = \int G_h(t) u(x + t \frac {Du(x)^⊥} {|Du(x)|}) dt$$

for x such that Du(x) $\neq$ 0 and where $(x, y)^⊥ = (−y, x)$ and $G_h$ is the one-dimensional Gauss function with variance $h^2$. If one assumes that the original image u is twice continuously differentiable ($C^2$) at x, it is easily shown by a second order Taylor expansion that

对于Du(x) $\neq$ 0的x点，其中$(x, y)^⊥ = (−y, x)$，$G_h$是一维高斯函数，方差$h^2$。如果假设原始图像u在x点处是二阶连续可微的($C^2$)，用二阶泰勒展开很容易可以得到

Theorem 2 The image method noise of an anisotropic filter $AF_h$ is

$$u(x) - AF_h u(x) = \frac {1}{2} h^2 |Du| curv(u)(x) + o(h^2)$$

where the relation holds when Du(x) $\neq$ 0.

By curv(u)(x), we denote the curvature, i.e. the signed inverse of the radius of curvature of the level line passing by x. This method noise is zero wherever u behaves locally like a straight line and large in curved edges or texture (where the curvature and gradient operators take high values). As a consequence, the straight edges are well restored while flat and textured regions are degraded.

curv(u)(x)表示曲率，即，通过x的level线的曲率半径的逆。如果u的局部是一条直线状的，那么method noise就为0；在弯曲的边缘和纹理处就很大（那里曲率和梯度算子的值很大）。结果是，直线边缘恢复的很好，而平坦区域和纹理区域则有降质。

### 2.3. The Total Variation minimization

The Total Variation minimization was introduced by Rudin, Osher and Fatemi [13]. Given a noisy image v(x), these authors proposed to recover the original image u(x) as the solution of the minimization problem

全变分最小化由[13]提出。给定一幅含噪图像v(x)，这些作者提出求解如下最小化问题来恢复原始图像u(x)

$$TVF_λ(v) = argmin_u TV(u) + λ\int |v(x)−u(x)|^2dx$$

where TV(u) denotes the total variation of u and λ is a given Lagrange multiplier. The minimum of the above minimization problem exists and is unique. The parameter λ is related to the noise statistics and controls the degree of filtering of the obtained solution.

其中TV(u)表示u的全变分，λ是给定的Lagrange乘子。上述最小化问题的最小值是存在并唯一的。参数λ与噪声统计值相关，控制着得到的解的滤波程度。

Theorem 3 The method noise of the Total Variation minimization is 全变分最小化的method noise是

$$u(x) − TVF_λ(u)(x) = − \frac {1}{2λ} curv(TVFλ(u))(x)$$

As in the anisotropic case, straight edges are maintained because of their small curvature. However, details and texture can be over smoothed if λ is too small.

在各向异性的情况下，直的边缘由于其曲率很好得到保持。但是，如果λ太小，细节和纹理会被过度平滑。

### 2.4. The neighborhood filtering

We call neighborhood filter any filter which restores a pixel by taking an average of the values of neighboring pixels with a similar grey level value. Yaroslavsky (1985) [16] averages pixels with a similar grey level value and belonging to the spatial neighborhood $B_ρ(x)$,

如果一个滤波器是通过对灰度值相近的邻域像素进行平均进行滤波的，我们就称之为邻域滤波。Yaroslavsky(1985)[16]对属于空域邻域$B_ρ(x)$且灰度值类似的像素进行平均

$$YNF_{h,ρ} u(x) = \frac{1}{C(x)} \int_{B_ρ(x)} u(y) e^{-\frac{|u(y)-u(x)|^2}{h^2}}dy$$(2)

where $x ∈ Ω, C(x) = \int_{B_ρ(x)} e^{-\frac{|u(y)-u(x)|^2}{h^2}}dy$ is the normalization factor and h is a filtering parameter. 其中...是归一化因子，h是滤波参数。

The Yaroslavsky filter is less known than more recent versions, namely the SUSAN filter (1995) [14] and the Bilateral filter (1998) [15]. Both algorithms, instead of considering a fixed spatial neighborhood $B_ρ(x)$, weigh the distance to the reference pixel x,

Yaroslavsky滤波器没有最近的工作出名，即SUSAN滤波器和双边滤波器。这两个算法都没有考虑固定的空域邻域$B_ρ(x)$，而是采用与参考像素x的距离的权重

$$SNF_{h,ρ}u(x) = \frac{1}{C(x)} \int_Ω u(y) e^{-\frac{|y-x|^2}{ρ^2}} e^{-\frac{|u(y)-u(x)|^2}{h^2}} dy$$

where $(x) = \int_Ω e^{-\frac{|y-x|^2}{ρ^2}} e^{-\frac{|u(y)-u(x)|^2}{h^2}} dy$ is the normalization  factor and ρ is now a spatial filtering parameter. In practice, there is no difference between $YNF_{h,ρ}$ and $SNF_{h,ρ}$. If the grey level difference between two regions is larger than h, both algorithms compute averages of pixels belonging to the same region as the reference pixel. Thus, the algorithm does not blur the edges, which is its main scope. In the experimentation section we only compare the Yaroslavsky neighborhood filter.

其中...是归一化因子，ρ是一个空间滤波参数。实践中，$YNF_{h,ρ}$和$SNF_{h,ρ}$没有区别。如果两个区域的灰度差异大于h，两个算法计算的像素平均所用的像素，是与参考像素相同的那个区域。因此，算法不会使得边缘模糊，这是其主要效果。在试验小节，我们只比较了Yaroslavsky邻域滤波器。

The problem with these filters is that comparing only grey level values in a single pixel is not so robust when these values are noisy. Neighborhood filters also create artificial shocks which can be justified by the computation of its method noise, see [3].

这些滤波器的问题是，当灰度值含有噪声时，比较一个像素的灰度值并不具有很好的鲁棒性。邻域滤波器还会产生伪影，可以通过计算其method noise来纠正。

## 3. NL-means algorithm

Given a discrete noisy image v = {v(i) | i ∈ I}, the estimated value NL[v](i), for a pixel i, is computed as a weighted average of all the pixels in the image,

给定一幅离散含噪图像v = {v(i) | i ∈ I}，对于像素i的估计值NL[v](i)，通过计算图像所有像素的加权平均得到

$$NL[v](i) = \sum_{j∈I} w(i,j)v(j)$$

where the family of weights {w(i,j)}_j depend on the similarity between the pixels i and j, and satisfy the usual conditions 0 ≤ w(i,j) ≤ 1 and $\sum_j w(i,j) = 1$.

其中权重族{w(i,j)}_j依赖于像素i和j的相似度，满足通常的条件，0 ≤ w(i,j) ≤ 1，和$\sum_j w(i,j) = 1$。

The similarity between two pixels i and j depends on the similarity of the intensity gray level vectors $v(N_i)$ and $v(N_j)$, where $N_k$ denotes a square neighborhood of fixed size and centered at a pixel k. This similarity is measured as a decreasing function of the weighted Euclidean distance,$||v(N_i) − v(N_j)||^2_{2,a}$, where a > 0 is the standard deviation of the Gaussian kernel. The application of the Euclidean distance to the noisy neighborhoods raises the following equality

两个像素i和j的相似度依赖于灰度向量$v(N_i)$和$v(N_j)$的相似度，其中$N_k$表示固定大小的方形邻域，中心在像素k上。这个相似度是以欧式距离为加权的递减函数，$||v(N_i) − v(N_j)||^2_{2,a}$，其中a>0是高斯核的标准方差。欧式距离在含噪邻域的应用带来了下面的等式

$$E||v(N_i) − v(N_j)||^2_{2,a} = ||u(N_i) − u(N_j)||^2_{2,a} + 2σ^2$$

This equality shows the robustness of the algorithm since in expectation the Euclidean distance conserves the order of similarity between pixels. 这个等式表明了算法的稳健性，因为欧式距离的期望保护了像素之间的相似度阶数。

The pixels with a similar grey level neighborhood to v(N_i) have larger weights in the average, see Figure 1. These weights are defined as, 与v(N_i)有着相似灰度邻域的像素，平均来说有更大的权重，见图1。这些权重定义为

Figure 1. Scheme of NL-means strategy. Similar pixel neighborhoods give a large weight, w(p,q1) and w(p,q2), while much different neighborhoods give a small weight w(p,q3).

$$w(i,j) = \frac{1}{Z(i} e^{-\frac{||v(N_i)-v(N_j)||^2_{2,a}}{h^2}}$$

where Z(i) is the normalizing constant 其中Z(i)是归一化常数

$$Z(i) = \sum_j e^{-\frac{||v(N_i)-v(N_j)||^2_{2,a}}{h^2}}$$

and the parameter h acts as a degree of filtering. It controls the decay of the exponential function and therefore the decay of the weights as a function of the Euclidean distances. 参数h是滤波程度的参数，控制着指数函数的衰减，因此权重的衰减是欧式距离的函数。

The NL-means not only compares the grey level in a single point but the the geometrical configuration in a whole neighborhood. This fact allows a more robust comparison than neighborhood filters. Figure 1 illustrates this fact, the pixel q3 has the same grey level value of pixel p, but the neighborhoods are much different and therefore the weight w(p, q3) is nearly zero.

NL-means不仅比较单点的灰度值，而且包含了整个邻域的几何配置。这比邻域滤波器的比较要更稳健。图1展示了这个事实，像素q3与点p的灰度值一样，但邻域非常不同，因此权重w(p, q3)接近于0。

## 4. NL-means consistency

Under stationarity assumptions, for a pixel i, the NL-means algorithm converges to the conditional expectation of i once observed a neighborhood of it. In this case, the stationarity conditions amount to say that as the size of the image grows we can find many similar patches for all the details of the image.

在平稳性的假设下，对于像素i，NL-means算法在观察了其邻域后，会收敛到i的条件期望。在这种情况下，平稳性的条件即是，在图像的大小不断增长的情况下，我们可以对图像的所有细节找到类似的图像块。

Let V be a random field and suppose that the noisy image v is a realization of V. Let Z denote the sequence of random variables Zi = {Yi , Xi} where Yi = V(i) is real valued and $X_i = V (N_i \backslash \{i\})$ is $R^p$ valued. The NL-means is an estimator of the conditional expectation $r(i) = E[Y_i | X_i = v(N_i \backslash \{i\})]$.

令V是一个随机场，假设含噪图像v是V的一个样本。令Z表示随机变量Zi = {Yi , Xi}的序列，其中Yi = V(i)是实值的，$X_i = V (N_i \backslash \{i\})$是$R^p$值的。NL-means是条件期望$r(i) = E[Y_i | X_i = v(N_i \backslash \{i\})]$的的估计值。

Theorem 4 (Conditional expectation theorem) Let $Z = \{V(i), V(N_i \backslash \{i\})\}$ for i = 1,2,... be a strictly stationary and mixing process. Let $NL_n$ denote the NL-means algorithm applied to the sequence $Z_n = \{V(i),V(N_i \backslash \{ i \})\}_{i=1}^n$. Then,

$$|NL_n(j) - r(j)|→0$$ a.s

for j ∈ {1,...,n}.

The full statement of the hypothesis of the theorem and its proof can be found in a more general framework in [12]. This theorem tells us that the NL-means algorithm corrects the noisy image rather than trying to separate the noise (oscillatory) from the true image (smooth).

这个定理的假设及其证明的完整表述，可以在[12]中找到其更一般性的框架。这个定理告诉我们，NL-means算法修正了含噪图像，而不是试图从真实图像中分离噪声。

In the case that an additive white noise model is assumed, the next result shows that the conditional expectation is the function of $V(N_i \backslash \{i\})$ that minimizes the mean square error with the true image u.

假设是加性白噪声的话，下一个结果表明条件期望是$V(N_i \backslash \{i\})$的函数，最小化了真实图像u的均方差。

Theorem 5 Let V, U, N be random fields on I such that V = U + N , where N is a signal independent white noise. Then, the following statements are hold.

(i) E[V(i) | Xi = x] = E[U(i) | Xi = x] for all i ∈ I and x ∈ R^p.

(ii) The expected random variable E[U(i) | V (Ni \ {i})] is the function of V (Ni \ {i}) that minimizes the mean square error

$$min_g E[U(i)−g(V(N_i \backslash \{i\}))]^2$$

Similar optimality theoretical results have been obtained in [9] and presented for the denoising of binary images. Theoretical links between the two algorithms will be explored in a future work.

## 5. Discussion and experimentation

In this section we compare the local smoothing filters and the NL-means algorithm under three well defined criteria: the method noise, the visual quality of the restored image and the mean square error, that is, the Euclidean difference between the restored and true images.

本节中，我们在三种定义好的准则下比较局部平滑滤波器和NL-means算法：method noise，恢复图像的视觉质量，和均方误差，即恢复的图像和真实图像的欧式差异。

For computational purposes of the NL-means algorithm, we can restrict the search of similar windows in a larger ”search window” of size S × S pixels. In all the experimentation we have fixed a search window of 21×21 pixels and a similarity square neighborhood Ni of 7 × 7 pixels. If N^2 is the number of pixels of the image, then the final complexity of the algorithm is about 49×441×N^2.

为了NL-means算法的计算，我们将搜索类似窗口限制在S × S像素的更大的搜索窗口。在所有试验中，我们的搜索窗口固定为21×21像素，相似度方形邻域Ni为7 × 7像素。如果N^2是图像像素的数量，那么算法最终的复杂度为大约49×441×N^2。

The 7 × 7 similarity window has shown to be large enough to be robust to noise and small enough to take care of details and fine structure. The filtering parameter h has been fixed to 10 * σ when a noise of standard deviation σ is added. Due to the fast decay of the exponential kernel, large Euclidean distances lead to nearly zero weights acting as an automatic threshold, see Fig. 2.

7 × 7的相似度窗口已经足够大，对噪声非常稳健，而且又很小，可以照顾到细节和精细结构。当加入的噪声的标准差为σ时，滤波参数h固定为10 * σ。由于指数核的快速衰减，欧式距离较大的话，就会得到接近于0的权重，是一种自动的阈值，见图2。

In section 2 we have computed explicitly the method noise of the local smoothing filters. These formulas are corroborated by the visual experiments of Figure 4. This figure displays the method noise for the standard image Lena, that is, the difference u − Dh(u), where the parameter h is been fixed in order to remove a noise of standard deviation 2.5. The method noise helps us to understand the performance and limitations of the denoising algorithms, since removed details or texture have a large method noise. We see in Figure 4 that the NL-means method noise does not present any noticeable geometrical structures. Figure 2 explains this property since it shows how the NL-means algorithm chooses a weighting configuration adapted to the local and non local geometry of the image.

在第2节我们计算了局部平滑滤波器的method noise。这些公式由图4的视觉试验所证实。图4展示了对于标准图像Lena的method noise，即，u − Dh(u)之差，其中参数h是固定的，以去除标准差为2.5的噪声。Method noise帮助我们理解去噪算法的性能和局限，因为去掉的细节或纹理的method noise很大。我们在图4中看到，NL-means的method noise没有表现出可注意到的几何结构。图2解释了这种性质，因为其展示了NL-means算法选择的加权方法，与图像的局部和非局部几何相适应。

The human eye is the only one able to decide if the quality of the image has been improved by the denoising method. We display some denoising experiences comparing the NL-means algorithm with local smoothing filters. All experiments have been simulated by adding a gaussian white noise of standard deviation σ to the true image. The objective is to compare the visual quality of the restored images, the non presence of artifacts and the correct reconstruction of edges, texture and details.

人类的眼睛是唯一能够决定，去噪方法是否改进了图像质量的。我们比较了NL-means算法与局部平滑滤波器，展示了一些去噪的试验。所有试验都是对真实图像加入了高斯白噪声，标准差为σ。其目标是，比较恢复的图像的视觉质量，不存在伪影，和边缘、纹理和细节的正确重建。

Due to the nature of the algorithm, the most favorable case for the NL-means is the textured or periodic case. In this situation, for every pixel i, we can find a large set of samples with a very similar configuration. See Figure 2 e) for an example of the weight distribution of the NL-means algorithm for a periodic image. Figure 3 compares the performance of the NL-means and local smoothing filters for a natural texture.

由于算法的本质，NL-means最适合的情况是有纹理的或周期性的情况。在这种情况中，对于每个像素i，我们可以找到很多有类似配置的样本。见图2e的例子，即NL-means算法对于一幅周期性图像的权重分布。图3比较了NL-means和局部平滑滤波器对于自然纹理的性能。

Natural images also have enough redundancy to be restored by NL-means. Flat zones present a huge number of similar configurations lying inside the same object, see Figure 2 (a). Straight or curved edges have a complete line of pixels with similar configurations, see Figure 2 (b) and (c). In addition, natural images allow us to find many similar configurations in far away pixels, as Figure 2 (f) shows. Figure 5 shows an experiment on a natural image. This experience must be compared with Figure 4, where we display the method noise of the original image. The blurred or degraded structures of the restored images coincide with the noticeable structures of its method noise.

自然图像也有充分的冗余，可以被NL-means所恢复。平滑的区域会在同样目标的内部，给出大量类似的配置，见图2a。直线或曲线边缘，在边缘附近的配置都是类似的，见图2b或c。另外，自然图像使我们可以在很远的像素中找到很多类似的配置，如图2f所示。图5在自然图像上进行了试验，需要与图4进行比较，其中我们展示了原始图像的method noise。恢复的图像的模糊的结构，或降质的结构，与其method noise中可以看到的结构是相吻合的。

Finally Table 1 displays the mean square error for the denoising experiments given in the paper. This numerical measurement is the most objective one, since it does not rely on any visual interpretation. However, this error is not computable in a real problem and a small mean square error does not assure a high visual quality. So all above discussed criteria seem necessary to compare the performance of algorithms.

最后，表1给出了本文中的去噪试验的均方误差。这个数值测量是最主观的，因为其不依赖于任何视觉解释。但是，这个误差在真实问题中是不可计算的，很小的均方误差并不保证图像质量很高。所以，上面讨论的原则，似乎对于比较算法性能都是必须的。