# Deep Image Prior

Dmitry Ulyanov et al. Skolkovo Institute of Science and Technology

## Abstract 摘要

Deep convolutional networks have become a popular tool for image generation and restoration. Generally, their excellent performance is imputed to their ability to learn realistic image priors from a large number of example images. In this paper, we show that, on the contrary, the structure of a generator network is sufficient to capture a great deal of low-level image statistics prior to any learning. In order to do so, we show that a randomly-initialized neural network can be used as a handcrafted prior with excellent results in standard inverse problems such as denoising, super-resolution, and inpainting. Furthermore, the same prior can be used to invert deep neural representations to diagnose them, and to restore images based on flash-no flash input pairs.

深度卷积网络已经成为了图像生成和图像恢复的流行工具。一般来说，其优异性能是由于其从大量样本图像中学习实际图像先验的能力。本文中，我们展示了相反的情况，生成网络的结构足以捕捉到大量底层图像统计先验。为达到这个目标，我们展示了，随机初始化的神经网络可以在标准逆问题中，如去噪、超分和inpainting中用作优异结果的手工先验。而且，同样的先验也可以用于逆深度神经表示来进行诊断，以及基于flash和no flash图像对的图像恢复。

Apart from its diverse applications, our approach highlights the inductive bias captured by standard generator network architectures. It also bridges the gap between two very popular families of image restoration methods: learning-based methods using deep convolutional networks and learning-free methods based on handcrafted image priors such as self-similarity.

除了广泛的应用，我们的方法强调通过标准生成网络架构捕捉到的inductive bias。这还在两种常用种类的图像恢复方法之间架起了桥梁：使用深度卷积网络的基于学习的方法，和基于手工图像先验如自相似性的不需要学习的方法。

## 1. Introduction 引言

Deep convolutional neural networks (ConvNets) currently set the state-of-the-art in inverse image reconstruction problems such as denoising [5, 20] or single-image super-resolution[19,29,18]. ConvNets have also been used with great success in more “exotic” problems such as reconstructing an image from its activations within certain deep networks or from its HOG descriptor [8]. More generally, ConvNets with similar architectures are nowadays used to generate images using such approaches as generative adversarial networks [11], variational autoencoders [16], and direct pixelwise error minimization [9, 3].

深度卷积神经网络(ConvNets)目前是逆图像重建问题的最好方法，如去噪[5,20]，或单幅图像超分[19,29,18]。ConvNets在很多奇怪的问题中也得到了成功的应用，如从特定深度网络中的激活中重建图像，或从其HOG描述子中重建图像[8]。更一般的，类似架构的ConvNets目前还用于生成图像，如使用生成式对抗网络[11]，变分自动编码机[16]，和直接的逐像素的误差最小化[9,3]。

State-of-the-art ConvNets for image restoration and generation are almost invariably trained on large datasets of images. One may thus assume that their excellent performance is due to their ability to learn realistic image priors from data. However, learning alone is insufficient to explain the good performance of deep networks. For instance, the authors of [33] recently showed that the same image classification network that generalizes well when trained on genuine data can also overfit when presented with random labels. Thus, generalization requires the structure of the network to “resonate” with the structure of the data. However, the nature of this interaction remains unclear, particularly in the context of image generation.

图像恢复和图像生成任务中目前最好的ConvNets几乎都是在大型图像数据集上训练得到的。因此可以假设，其优异的性能是因为其从数据中学习实际图像先验的能力。但是，单单学习是不足以解释深度网络的良好性能的。比如，[33]的作者最近指出，同样的分类网络，当在真实数据上训练后泛化能力很好的，当给定随机标签时也会过拟合。所以，泛化需要网络的结构与数据的结构产生共振。但是，这种互动的本质仍然不是太清楚，尤其是在图像生成的上下文中。

In this work, we show that, contrary to the belief that learning is necessary for building good image priors, a great deal of image statistics are captured by the structure of a convolutional image generator independent of learning. This is particularly true for the statistics required to solve various image restoration problems, where the image prior is required to integrate information lost in the degradation processes.

一般认为学习对于构建好的图像先验来说是必要的，本文中我们展示了相反的情况，大量图像统计数据都可以用一个卷积图像生成器的结构捕捉到，这与学习无关。对于解决各种图像恢复问题的统计数据来说，这个结论尤其正确，其中需要图像先验整合在降质过程中损失的信息。

To show this, we apply untrained ConvNets to the solution of several such problems. Instead of following the common paradigm of training a ConvNet on a large dataset of example images, we fit a generator network to a single degraded image. In this scheme, the network weights serve as a parametrization of the restored image. The weights are randomly initialized and fitted to maximize their likelihood given a specific degraded image and a task-dependent observation model.

为展示这个结论，我们在几个这样问题的解决方案中应用了未训练的卷积网络。常用的方案是在大型图像样本数据集上训练卷积网络，我们的方案则是在单个降质图像上匹配一个生成器网络。在这个方案中，网络权重的作用是恢复图像的参数。权重随机初始化，目标是在给定降质图像和与任务相关的观察模型时最大化其似然概率。

Stated in a different way, we cast reconstruction as a conditional image generation problem and show that the only information required to solve it is contained in the single degraded input image and the handcrafted structure of the network used for reconstruction.

换句话说，我们将重建问题看作一个条件图像生成问题，并指出解决问题需要的唯一信息包含在单幅降质输入图像中，以及用于重建的手工设计的网络结构中。

We show that this very simple formulation is very competitive for standard image processing problems such as denoising, inpainting and super-resolution. This is particularly remarkable because no aspect of the network is learned from data; instead, the weights of the network are always randomly initialized, so that the only prior information is in the structure of the network itself. To the best of our knowledge, this is the first study that directly investigates the prior captured by deep convolutional generative networks independently of learning the network parameters from images.

我们展示了，这种非常简单的构想对于标准图像处理问题非常有竞争力，如去噪、inpaiting和超分。这非常引人注目，因为网络不是从数据中学习得到的；网络的权重一直是随机初始化的，所以唯一的先验信息就是网络自己的结构。据我们所知，这是第一个这样的研究，直接研究深度卷积生成式网络捕捉到的先验，而不是从图像数据集中学习网络参数。

In addition to standard image restoration tasks, we show an application of our technique to understanding the information contained within the activations of deep neural networks. For this, we consider the “natural pre-image” technique of [21], whose goal is to characterize the invariants learned by a deep network by inverting it on the set of natural images. We show that an untrained deep convolutional generator can be used to replace the surrogate natural prior used in [21] (the TV norm) with dramatically improved results. Since the new regularizer, like the TV norm, is not learned from data but is entirely handcrafted, the resulting visualizations avoid potential biases arising from the use of powerful learned regularizers [8].

除了标准图像恢复任务，我们的技术还可以理解深度神经网络中的激活所包含的信息，为展示这个应用，我们考虑[21]中的自然预图像技术(natural pre-image)，其目标是描述深度网络学习的不变性的特征，方法是逆变换到自然图像集中。我们展示了，一个未训练的深度生成器可用于替换[21]中的自然先验代理(the TV norm)，极大的改进结果。因为新的正则化器，如TV norm，不是从数据中学习到的，而完全是手工设计的，可视化结果避免了由于使用强有力的学习正则化器所带来的潜在的偏移。

Figure 1: Super-resolution using the deep image prior. Our method uses a randomly-initialized ConvNet to upsample an image, using its structure as an image prior; similar to bicubic upsampling, this method does not require learning, but produces much cleaner results with sharper edges. In fact, our results are quite close to state-of-the-art super-resolution methods that use ConvNets learned from large datasets. The deep image prior works well for all inverse problems we could test.

## 2. Method 方法

Deep networks are applied to image generation by learning generator/decoder networks $x = f_θ (z)$ that map a random code vector z to an image x. This approach can be used to sample realistic images from a random distribution [11]. Here we focus on the case where the distribution is conditioned on a corrupted observation $x_0$ to solve inverse problems such as denoising [5] and super-resolution [7].

深度网络应用于图像生成，是要学习一个生成器/解码器网络$x = f_θ (z)$，将随机编码向量z映射到图像x。这种方法可以用于以随机分布对真实图像进行采样[11]。这里我们聚焦于这样的情况，即分布是降质观测$x_0$的条件分布，来解决逆问题如去噪[5]和超分[7]。

Our aim is to investigate the prior implicitly captured by the choice of a particular generator network structure, before any of its parameters are learned. We do so by interpreting the neural network as a parametrization $x = f_θ (z)$ of an image $x ∈ R^{3×H×W}$. Here $z ∈ R^{C'×H'×W'}$ is a code tensor/vector and θ are the network parameters. The network itself alternates filtering operations such as convolution, upsampling and non-linear activation. In particular, most of our experiments are performed using a U-Net type “hourglass” architecture with skip-connections, where z and x have the same spatial size. Our default architecture has two million parameters θ (see Supplementary Material for the details of all used architectures).

我们的目标是研究，选择的特定生成器网络结构，在学习其参数之前，所隐式的捕捉到的先验。我们将神经网络解释为一幅图像$x ∈ R^{3×H×W}$的参数化表示，从而达到这个目标。这里$z ∈ R^{C'×H'×W'}$是张量/向量编码，θ是网络参数。网络本身交替进行滤波操作，如卷积、上采样和非线性激活。特别是，我们的大部分实验都是用带有跳跃连接的U-Net类型的沙漏架构，其中z和x的空间大小是一样的。我们的默认架构有两百万参数θ（见补充资料中的所有使用的架构的细节）。

To demonstrate the power of this parametrization, we consider inverse tasks such as denoising, super-resolution and inpainting. These can be expressed as energy minimization problems of the type

为证明这种参数化的力量，我们考虑一些逆问题，如去噪、超分或inpainting。这可以归结为下述能量最小化问题：

$$x^∗ = min_x E(x;x_0) + R(x)$$(1)

where $E(x;x_0)$ is a task-dependent data term, $x_0$ the noisy/low-resolution/occluded image, and R(x) a regularizer. 其中$E(x;x_0)$是于任务相关的数据项，$x_0$是含噪/低分辨率/遮挡图像，R(x)是正则化器。

The choice of data term $E(x;x_0)$ is dictated by the application and will be discussed later. The choice of regularizer, which usually captures a generic prior on natural images, is more difficult and is the subject of much research. As a simple example, R(x) can be the Total Variation (TV) of the image, which encourages solutions to contain uniform regions. In this work, we replace the regularizer R(x) with the implicit prior captured by the neural network, as follows:

数据项$E(x;x_0)$的选择是由应用决定的，放在后面讨论。正则化项通常捕捉的是自然图像的一般先验，其选择更加困难，需要更多的研究。一个简单的例子是，R(x)可以是图像的全变分(TV)，这会鼓励解中包含均一区域。在本文中，我们将正则化器R(x)替换为神经网络捕捉到的隐式先验，如下：

$$θ^∗ = argmin_θ E(f_θ (z); x_0), x^∗ = f_{θ^∗} (z)$$(2)

The minimizer $θ^∗$ is obtained using an optimizer such as gradient descent starting from a random initialization of the parameters. Given a (local) minimizer $θ^∗$, the result of the restoration process is obtained as $x^∗ = f_{θ^∗}(z)$. Note that while it is also possible to optimize over the code z, in our experiments we do not do that. Thus, unless noted otherwise, z is a fixed 3D tensor with 32 feature maps and of the same spatial size as x filled with uniform noise. We found that additionally perturbing z randomly at every iteration lead to better results in some experiments (c.f. Supplementary material).

要得到最小化的$θ^∗$，可以使用优化器，如参数随机初始化的梯度下降法。给定一个（局部）最小化值$θ^∗$，恢复过程的结果由$x^∗ = f_{θ^∗}(z)$得到。注意虽然也可能对z进行优化，在实验中我们没有那样做。所以，除非另有解释，z是一个固定3D张量，有32个特征图，空间大小与x相同，其内容为均匀噪声。我们发现，每个迭代中随机给z添加扰乱，在一些实验中会得到更好的结果（见参考资料）。

In terms of (1), the prior R(x) defined by (2) is an indicator function R(x) = 0 for all images that can be produced from z by a deep ConvNet of a certain architecture, and R(x) = +∞ for all other signals. Since no aspect of the network is pre-trained from data, such deep image prior is effectively handcrafted, just like the TV norm. We show that this hand-crafted prior works very well for various image restoration tasks.

在式(1)中，由(2)定义的先验R(x)，对于所有可以从z通过某种架构的深度网络生成的图像，都是R(x)=0，对于其他所有信号，都是R(x) = +∞。由于网络没有从数据进行预训练，这样的深度图像先验都是手工设计的，就像TV模值一样。我们会证明，这个手工设计的先验在各种图像恢复任务中都会得到很好的应用。

**A parametrization with high noise impedance**. One may wonder why a high-capacity network $f_θ$ can be used as a prior at all. In fact, one may expect to be able to find parameters θ recovering any possible image x, including random noise, so that the network should not impose any restriction on the generated image. We now show that, while indeed almost any image can be fitted, the choice of network architecture has a major effect on how the solution space is searched by methods such as gradient descent. In particular, we show that the network resists “bad” solutions and descends much more quickly towards naturally-looking images. The result is that minimizing (2) either results in a good-looking local optimum, or, at least, the optimization trajectory passes near one.

**参数化对噪声抵抗很大**。你可能会怀疑高容量网络$f_θ$是否可以用作先验。实际上，你能找到参数θ来恢复所有可能的图像x，包括随机噪声，所以网络不用对生成的图像施加任何限制。我们现在证明，虽然可以容纳几乎任意图像，但网络架构的选择对怎样搜索解空间有主要影响，如梯度下降。特别是，我们证明了，网络抵制坏的解，而且迅速向看起来很自然的图像逼近。结果是，最小化(2)式要么得到看起来很好的局部最优点，或者至少是，优化过程很接近局部最优点。

In order to study this effect quantitatively, we consider the most basic reconstruction problem: given a target image $x_0$, we want to find the value of the parameters $θ^∗$ that reproduce that image. This can be setup as the optimization of (2) using a data term comparing the generated image to $x_0$:

为量化的研究这种效果，我们考虑最基本的重建问题：给定一个目标图像$x_0$，我们希望找到参数$θ^∗$可以重现这幅图像。这可以用优化问题(2)表示为数据项为生成的图像与$x_0$比较：

$$E(x; x_0) = ||x − x_0||^2$$(3)

Plugging this in eq. (2) leads us to the optimization problem 将上式插入(2)中会得到下述优化问题

$$min_θ ||f_θ (z) − x_0||^2$$(4)

Figure 2 shows the value of the energy $E(x; x_0)$ as a function of the gradient descent iterations for four different choices for the image $x_0$: 1) a natural image, 2) the same image plus additive noise, 3) the same image after randomly permuting the pixels, and 4) white noise. It is apparent from the figure that optimization is much faster for cases 1) and 2), whereas the parametrization presents significant “inertia” for cases 3) and 4).

图2所示的是能量$E(x; x_0)$的值作为梯度下降迭代次数的函数的图像，图中对$x_0$有4种不同的选择：(1)自然图像，(2)自然图像加加性噪声，(3)自然图像重新随机排列像素，(4)白噪声。图中可以很明显看出，对于情况(1)(2)优化过程很迅速，而参数化对于情况(3)(4)表现出了显著的迟钝反应。

Figure 2: Learning curves for the reconstruction task using: a natural image, the same plus i.i.d. noise, the same randomly scrambled, and white noise. Naturally-looking images result in much faster convergence, whereas noise is rejected.

Thus, although in the limit the parametrization can fit unstructured noise, it does so very reluctantly. In other words, the parametrization offers high impedance to noise and low impedance to signal. Therefore for most applications, we restrict the number of iterations in the optimization process (2) to a certain number of iterations. The resulting prior then corresponds to projection onto a reduced set of images that can be produced from z by ConvNets with parameters θ that are not too far from the random initialization $θ_0$.

所以，虽然在极限情况下参数化可以适应没有结构的噪声，但是也是非常勉强的。换句话说，参数化对噪声阻抗很高，对信号阻抗很低。所以对于大部分应用，我们限制优化过程(2)中的迭代次数，使其不超过一定的迭代次数。得到的先验对应着向一个有限图像集合的投影，可以从z经过参数为θ的卷积网络生成，这个网络与随机初始化的$θ_0$比较接近。

## 3. Applications 应用

We now show experimentally how the proposed prior works for diverse image reconstruction problems. Due to space limitations, we present a few examples and numbers and include many more in the Supplementary material and the project webpage [30].

我们现给出实验结果，说明提出的先验可以解决各种图像重建问题。由于篇幅限制，我们给出几个例子和一些数据，并在补充材料和项目网页[30]中给出更详细的内容。

**Denoising and generic reconstruction**. As our parametrization presents high impedance to image noise, it can be naturally used to filter out noise from an image. The aim of denoising is to recover a clean image x from a noisy observation $x_0$. Sometimes the degradation model is known: $x_0 = x + \epsilon$ where $\epsilon$ follows a particular distribution. However, more often in blind denoising the noise model is unknown.

**去噪和一般性重建**。由于我们的参数化方案对图像噪声阻抗很高，所以很自然的可以用于从图像中滤除噪声。去噪的目的是从含噪观测$x_0$中恢复出干净图像x。有时降质模型是已知的，$x_0 = x + \epsilon$，其中$\epsilon$服从特定的分布。但是，在盲去噪中噪声模型更多的是未知的。

Here we work under the blindness assumption, but the method can be easily modified to incorporate information about noise model. We use the same exact formulation as eqs. (3) and (4) and, given a noisy image $x_0$, recover a clean image $x^∗ = f_{θ^∗} (z)$ after substituting the minimizer $θ^∗$ of eq. (4).

这里我们在盲恢复的假设下进行，但是这个方法可以很容易的引入噪声模型的信息。我们使用(3)和(4)的设置，给定含噪图像$x_0$，替换掉(4)中的最小化器$θ^∗$后，恢复一个干净的图像$x^∗ = f_{θ^∗} (z)$。

Our approach does not require a model for the image degradation process that it needs to revert. This allows it to be applied in a “plug-and-play” fashion to image restoration tasks, where the degradation process is complex and/or unknown and where obtaining realistic data for supervised training is difficult. We demonstrate this capability by sevral qualitative examples in fig. 4 and in the supplementary material, where our approach uses the quadratic energy (3) leading to formulation (4) to restore images degraded by complex and unknown compression artifacts. Figure 3 (top row) also demonstrates the applicability of the method beyond natural images (a cartoon in this case).

我们的方法不需要图像降质过程的模型来进行图像恢复。这使得我们的方法可以即插即用式的处理图像恢复任务，其中降质过程可以非常复杂或者是未知的，得到实际的数据进行有监督的训练非常困难。我们在图4和附加材料中给出了一些定性的例子，以证明这种能力，其中我们的方法使用了quadratic能量(3)，代入(4)式后可以恢复由复杂未知的噪声降质的图像。图3证明了我们的方法在自然图像之外的应用能力（在这个情况下是一幅卡通图像）。

Figure 3: Blind restoration of a JPEG-compressed image. (electronic zoom-in recommended) Our approach can restore an image with a complex degradation (JPEG compression in this case). As the optimization process progresses, the deep image prior allows to recover most of the signal while getting rid of halos and blockiness (after 2400 iterations) before eventually overfitting to the input (at 50K iterations).

图3：JPEG压缩图像的盲恢复。我们的方法可以恢复复杂降质的图像（在本例中是JPEG压缩）。随着优化过程的进行，深度图像先验可以恢复绝大部分信号，同时（在2400次迭代之后）去除晕轮和块状问题，最后（在50K次迭代后）达到过拟合状态。

Figure 4: Blind image denoising. The deep image prior is successful at recovering both man-made and natural patterns. For reference, the result of a state-of-the-art non-learned denoising approach [6] is shown.

图4.图像盲去噪。深度图像先验在恢复人造图像和自然图像方面都很成功。作为参考，目前最好的非学习去噪方法[6]也在图中给出。

We evaluate our denoising approach on the standard dataset(http://www.cs.tut.fi/foi/GCF-BM3D/index.html#ref_results), consisting of 9 colored images with noise strength of σ = 25. We achieve a PSNR of 29.22 after 1800 optimization steps. The score is improved up to 30.43 if we additionally average the restored images obtained in the last iterations (using exponential sliding window). If averaged over two optimization runs our method further improves up to 31.00 PSNR. For reference, the scores for the two popular approaches CMB3D [6] and Non-local means [4] that do not require pretraining are 31.42 and 30.26 respectively.

我们在标准数据集上评估我们的去噪方法，数据集包括9幅彩色图像，噪声强度σ = 25，我们的方法在1000次迭代后，达到PSNR 29.22。如果我们将迭代得到的恢复图像另外进行平均（使用指数滑动窗口），PSNR可以改进达到30.43。如果在两次优化结果之间进行平均，我们方法可以进一步改进达到31.00 PSNR。为参考，两种流行的方法CMB3D[6]和Non-local means[4]可以达到的PSNR分别为31.42和30.26，它们不需要预训练。

**Super-resolution**. The goal of super-resolution is to take a low resolution (LR) image $x_0 ∈ R^{3×H×W}$ and upsampling factor t, and generate a corresponding high resolution (HR) version $x ∈ R^{3×tH×tW}$. To solve this inverse problem, the data term in (2) is set to:

**超分辨率**。超分辨率的目标是以低分辨率图像(LR)$x_0 ∈ R^{3×H×W}$和上采样因子t为输入，生成对应的高分辨率(HR)版本图像$x ∈ R^{3×tH×tW}$。为求解这个逆问题，(2)中的数据项要设为：

$$E(x; x_0) = ||d(x) − x_0||^2$$(5)

where $d(·) : R^{3×tH×tW} → R^{3×H×W}$ is a downsampling operator that resizes an image by a factor t. Hence, the problem is to find the HR image x that, when downsampled, is the same as the LR image $x_0$. Super-resolution is an ill-posed problem because there are infinitely many HR images x that reduce to the same LR image $x_0$ (i.e. the operator d is far from surjective). Regularization is required in order to select, among the infinite minimizers of (5), the most plausible ones.

其中$d(·) : R^{3×tH×tW} → R^{3×H×W}$是降采样算子，降采样因子为t。所以，问题就成为，找到高分辨率图像x，如果对其进行降采样，得到的结果与低分辨率图像$x_0$相同。超分辨率是一个病态问题，因为有无数个高分辨率图像x可以退化为相同的低分辨率图像$x_0$（即，算子d远不是满射的）。需要正则化来从(5)中的无数个解中选择最可行的那个。

Following eq. (2), we regularize the problem by considering the reparametrization $x = f_θ (z)$ and optimizing the resulting energy w.r.t. θ. Optimization still uses gradient descent, exploiting the fact that both the neural network and the most common downsampling operators, such as Lanczos, are differentiable.

采用(2)的方法，我们对问题进行正则化，即参数化为$x = f_θ (z)$，相对于θ优化得到的能量。优化过程仍然使用梯度下降，因为可以利用下述事实，即神经网络和最常见的降采样算子如Lanczos都是可微分的。

We evaluate super-resolution ability of our approach using Set5 [2] and Set14 [32] datasets. We use a scaling factor of 4 to compare to other works, and show results with scaling factor of 8 in supplementary materials. We fix the number of optimization steps to be 2000 for every image.

我们使用Set5[2]和Set14[32]数据集来评估我们的超分辨率算法的能力。与其他工作相比，我们使用的尺度因子为4，在补充材料中给出了尺度因子为8的结果。对每幅图像，我们将优化迭代次数都固定为2000。

Qualitative comparison with bicubic upsampling and state-of-the art learning-based methods SRResNet [19], LapSRN [29] is presented in fig. 5. Our method can be fairly compared to bicubic, as both methods never use other data than a given low-resolution image. Visually, we approach the quality of learning-based methods that use the MSE loss. GAN-based [11] methods SRGAN [19] and EnhanceNet [28] (not shown in the comparison) intelligently hallucinate fine details of the image, which is impossible with our method that uses absolutely no information about the world of HR images.

与双三次上采样和目前最好的基于学习的方法SRResNet[19]、LapSRN[29]的定性比较如图5所示。我们的方法与双三次上采样的方法比较起来比较公平，因为两种方法都只使用了给定的低分辨率图像。视觉上来说，我们的方法接近基于学习的方法（使用MSE损失）的质量。基于GAN[11]的方法SRGAN[19]和EnhanceNet[28]（没有在对比中给出）可以智能的幻化出图像的精细细节，这对于我们的方法来说是不可能的，因为一点都没有使用高分辨率图像的信息。

We compute PSNRs using center crops of the generated images. Our method achieves 29.90 and 27.00 PSNR on Set5 and Set14 datasets respectively. Bicubic upsampling gets a lower score of 28.43 and 26.05, while SRResNet has PSNR of 32.10 and 28.53. While our method is still outperformed by learning-based approaches, it does considerably better than bicubic upsampling. Visually, it seems to close most of the gap between bicubic and state-of-the-art trained ConvNets (c.f. fig. 1,fig. 5 and suppmat).

我们使用生成图像的中间简切块来计算PSNR。我们的方法在Set5和Set14中分别取得了29.90和27.00的PSNR。双三次上采样的成绩为28.43和26.05，SRResNet的PSNR为32.10和28.53。我们的方法仍然没有基于学习的方法好，但比双三次上采样要好不少。视觉上来说，弥合了双三次和目前最好的训练神经网络的结果间的鸿沟（参考图1、图5和补充资料）。

**Inpainting**. In image inpainting, one is given an image $x_0$ with missing pixels in correspondence of a binary mask $m ∈ \{0, 1\}^{H×W}$; the goal is to reconstruct the missing data. The corresponding data term is given by

**去瑕疵**。在image inpainting中，给定的图像$x_0$有缺失像素，对应二值掩膜$m ∈ \{0, 1\}^{H×W}$；目的是重建缺失的数据。对应的数据项如下式：

$$E(x; x_0) = ||(x − x_0) \odot m||^2$$(6)

where $\odot$ is Hadamard’s product. The necessity of a data prior is obvious as this energy is independent of the values of the missing pixels, which would therefore never change after initialization if the objective was optimized directly over pixel values x. As before, the prior is introduced by optimizing the data term w.r.t. the reparametrization (2).

其中$\odot$为Hadamard乘积。数据先验的必须性是明显的，因为能量与缺失的像素值是无关的，因此如果目标函数直接在像素值x上进行直接优化时，在初始化之后绝对不会变化。与之前一样，先验的引入是通过对参数(2)优化数据项。

In the first example (fig. 7, top row) inpainting is used to remove text overlaid on an image. Our approach is compared to the method of [27] specifically designed for inpainting. Our approach leads to an almost perfect results with virtually no artifacts, while for [27] the text mask remains visible in some regions.

在第一个例子中（图7，第一行），inpainting用于去除图像上方的文字。我们的方法与[27]中的方法进行比较，那是专门为inpainting设计的算法。我们的方法得到的结果几乎是完美的，基本上没有瑕疵，而[27]中的有些区域中文字仍然可见文字掩膜。

Figure 7: Comparison with two recent inpainting approaches. Top – comparison with Shepard networks [27] on text inpainting example. Bottom – comparison with convolutional sparse coding [25] on inpainting 50% of missing pixels. In both cases, our approach performs better on the images used in the respective papers.

Next, fig. 7 (bottom) considers inpainting with masks randomly sampled according to a binary Bernoulli distribution. First, a mask is sampled to drop 50% of pixels at random. We compare our approach to a method of [25] based on convolutional sparse coding. To obtain results for [25] we first decompose the corrupted image $x_0$ into low and high frequency components similarly to [12] and run their method on the high frequency part. For a fair comparison we use the version of their method, where a dictionary is built using the input image (shown to perform better in [25]). The quantitative comparison on the standard data set [14] for our method is given in table 1, showing a strong quantitative advantage of the proposed approach compared to convolutional sparse coding. In fig. 7 (bottom) we present a representative qualitative visual comparison with [25].

下一个，图7（下面）处理的是随机掩膜的inpainting问题，随机掩膜服从二值Bernoulli分布。首先，随机生成50%的掩膜。我们将我们的方法与[25]的方法进行比较，[25]是基于卷积稀疏编码的。为得到[25]的结果，我们首先将降质图像$x_0$分解成低频和高频成分，与[12]类似，使用他们的方法处理高频成分。为公平比较，我们使用他们方法的版本，其中使用输入图像构建了一个字典（[25]中说明这样表现更好）。在标准数据集[14]中与我们的方法的定量比较如图1所示，结果显示，我们的方法与卷积稀疏编码相比，数据上有很大的优势。在图7（下面）中，我们给出了与[25]的定性视觉比较的代表性例子。

Table 1: Comparison between our method and the algorithm in [25]. See fig. 7 bottom row for visual comparison.

| | Barbara | Boat | House | Lena | Peppers | C.man | Couple | Finger | Hill | Man | Montage
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
Papyan et al. | 28.14 | 31.44 | 34.58 | 35.04 | 31.11 | 27.90 | 31.18 | 31.34 | 32.35 | 31.92 | 28.05
Ours | 32.22 | 33.06 | 39.16 | 36.16 | 33.05 | 29.8 | 32.52 | 32.84 | 32.77 | 32.20 | 34.54

We also apply our method to inpainting of large holes. Being non-trainable, our method is not expected to work correctly for “highly-semantical” large-hole inpainting (e.g. face inpainting). Yet, it works surprisingly well for other situations. We compare to a learning-based method of [15] in fig. 6. The deep image prior utilizes context of the image and interpolates the unknown region with textures from the known part. Such behaviour highlights the relation between the deep image prior and traditional self-similarity priors.

我们还将我们的方法应用于大的孔洞的inpainting。我们的方法是非训练型的，因此不能期望对于高度语义化的大的孔洞的inpainting也能正确工作（如，面部inpainting）。但是，对于其他情况效果出奇的好。我们在图6中与一种基于学习的方法[15]进行了比较。深度图像先验利用了图像的上下文，用已知部分的纹理来插值未知区域。这样的行为强调了深度图像先验和传统的自相似先验间的关系。

Figure 6: Region inpainting. In many cases, deep image prior is sufficient to successfully inpaint large regions. Despite using no learning, the results may be comparable to [15] which does. The choice of hyper-parameters is important (for example (d) demonstrates sensitivity to the learning rate), but a good setting works well for most images we tried.

In fig. 8, we compare deep priors corresponding to several architectures. Our findings here (and in other similar comparisons) seem to suggest that having deeper architecture is beneficial, and that having skip-connections that work so well for recognition tasks (such as semantic segmentation) is highly detrimental.

在图8中，我们比较了深度先验对应几种框架。我们在这里的发现（以及在其他类似的比较中）得到的结论似乎是更深的架构更好，包括skip连接的架构对于识别任务效果更好（比如语义分割）效果更差一些。

Figure 8: Inpainting using different depths and architectures. The figure shows that much better inpainting results can be obtained by using deeper random networks. However, adding skip connections to ResNet in U-Net is highly detrimental.

**Natural pre-image**. The natural pre-image method of [21] is a diagnostic tool to study the invariances of a lossy function, such as a deep network, that operates on natural images. Let Φ be the first several layers of a neural network trained to perform, say, image classification. The pre-image is the set $Φ^{−1}(Φ(x_0)) = \{x ∈ X : Φ(x) = Φ(x_0)\}$ of images that result in the same representation $Φ(x_0)$. Looking at this set reveals which information is lost by the network, and which invariances are gained.

**自然预映射**。[21]中的Natral pre-image方法是一个诊断工具，可以研究损失函数的不变性，比如处理自然图像的深度网络。令Φ为一个神经网络的开始几层，训练用于图像分类。pre-image就是图像集合$Φ^{−1}(Φ(x_0)) = \{x ∈ X : Φ(x) = Φ(x_0)\}$，可以得到与$Φ(x_0)$同样的表示。观察这个集合，可以发现网络丢失了哪些信息，保持了那些不变性。

Finding pre-image points can be formulated as minimizing the data term $E(x; x_0) = ||Φ(x) − Φ(x_0)||^2$. However, optimizing this function directly may find “artifacts”, i.e. non-natural images for which the behavior of the network Φ is in principle unspecified and that can thus drive it arbitrarily. More meaningful visualization can be obtained by restricting the pre-image to a set X of natural images, called a natural pre-image in [21].

找到pre-image点可以表示为最小化数据项$E(x; x_0) = ||Φ(x) − Φ(x_0)||^2$。但是，直接最优化这个函数可能找到artifacts，即非自然图像，因为网络Φ的行为原则上是尚未指定的，因此可以任意行动。限制pre-image到自然图像集X中，可以得到更有意义的可视化效果，这在[21]中称为natural pre-image。

In practice, finding points in the natural pre-image can be done by regularizing the data term similarly to the other inverse problems seen above. The authors of [21] prefer to use the TV norm, which is a weak natural image prior, but is relatively unbiased. On the contrary, papers such as [8] learn to invert a neural network from examples, resulting in better looking reconstructions, which however may be biased towards learning data-driven inversion prior. Here, we propose to use the deep image prior (2) instead. As this is handcrafted like the TV-norm, it is not biased towards a particular training set. On the other hand, it results in inversions at least as interpretable as the ones of [8].

在实践中，解决natural pre-image问题可以通过对数据项增加正则化，使其与上面的其他逆问题相似。[21]的作者倾向于使用TV范数，这是一种弱自然图像先验，但相对无偏。相反，像[8]这样的文章学习从样本中逆转一个神经网络，得到更好看的重建，这可能相对于学习数据驱动的逆先验是有偏的。这里，我们提出使用深度图像先验(2)进行替代。因为这是与TV范数类似的手工设计的，这并不倾向于某一特定训练数据集。另一方面，它得到的逆结果至少与[8]的结果一样可解释。

For evaluation, our method is compared to the ones of [22] and [8]. Figure 9 shows the results of inverting representations Φ obtained by considering progressively deeper subsets of AlexNet [17]: conv1, conv2, ..., conv5, fc6, fc7, and fc8. Pre-images are found either by optimizing (2) using a structured prior.

为评估，我们的方法与[12,8]中的算法进行比较。图9给出了逆表示Φ的结果，这是通过逐渐加深的AlexNet[17]的子集：conv1, conv2, ..., conv5, fc6, fc7, and fc8。Pre-images是使用不同的结构化先验优化(2)得到的。

As seen in fig. 9, our method results in dramatically improved image clarity compared to the simple TV-norm. The difference is particularly remarkable for deeper layers such as fc6 and fc7, where the TV norm still produces noisy images, whereas the structured regularizer produces images that are often still interpretable. Our approach also produces more informative inversions than a learned prior of [8], which have a clear tendency to regress to the mean.

如图9所示，我们的方法得到的结果比使用TV范数得到的结果要清晰的多。更深的层如fc6和fc7的结果差异尤其明显，TV范数得到的是噪声非常多的图像，而结构化的正则化得到的结果通常还可以解释。我们的方法与学习的先验[8]比较，得到的逆结果也包含更多信息，[8]倾向于回归到均值。

Figure 9: AlexNet inversion. Given the image on the left, we show the natural pre-image obtained by inverting different layers of AlexNet (trained for classification on ImageNet ISLVRC) using three different regularizers: the Deep Image prior, the TV norm prior of [21], and the network trained to invert representations on a hold-out set [8]. The reconstructions obtained with the deep image prior are in many ways at least as natural as [8], yet they are not biased by the learning process.

**Flash-no flash reconstruction**. While in this work we focus on single image restoration, the proposed approach can be extended to the tasks of the restoration of multiple images, e.g. for the task of video restoration. We therefore conclude the set of application examples with a qualitative example demonstrating how the method can be applied to perform restoration based on pairs of images. In particular, we consider flash-no flash image pair-based restoration [26], where the goal is to obtain an image of a scene with the lighting similar to a no-flash image, while using the flash image as a guide to reduce the noise level.

**Flash-no flash重建**。本文中我们主要聚焦在单幅图像恢复中，但我们提出的方法可以拓展到恢复多幅图像的任务中，如视频恢复的任务。所以我们用一个定性的例子来代表，说明我们的方法可以应用于成对图像的恢复。特别的，我们考虑flash-no flash图像对的恢复[26]，得到的目标图像，其光照条件与非曝光图像类似，而且使用曝光图像作为指引来降低噪声水平。

In general, extending the method to more than one image is likely to involve some coordinated optimization over the input codes z that for single-image tasks in our approach was most often kept fixed and random. In the case of flash-no-flash restoration, we found that good restorations were obtained by using the denoising formulation (4), while using flash image as an input (in place of the random vector z). The resulting approach can be seen as a non-linear generalization of guided image filtering [13]. The results of the restoration are given in the fig. 10.

一般来说，将方法拓展到多于一幅图像很可能会涉及到对输入编码z的协同优化问题，在我们的方法中输入编码z对于单幅图像的任务一般是固定随机的。在曝光-非曝光恢复中，我们发现，在使用曝光图像作为输入时（取代随机向量z），使用去噪的表述(4)可以得到很好的恢复。得到的方法可以看作是引导图像滤波[13]的非线性泛化。恢复的结果如图10所示。

Figure 10: Reconstruction based on flash and no-flash image pair. The deep image prior allows to obtain low-noise reconstruction with the lighting very close to the no-flash image. It is more successful at avoiding “leaks” of the lighting patterns from the flash pair than joint bilateral filtering [26] (c.f. blue inset).

## 4. Related work 相关工作

Our method is obviously related to image restoration and synthesis methods based on learnable ConvNets and referenced above. At the same time, it is as much related to an alternative group of restoration methods that avoid training on the hold-out set. This group includes methods based on joint modeling of groups of similar patches inside corrupted image [4, 6, 10], which are particularly useful when the corruption process is complex and highly variable (e.g. spatially-varying blur [1]). Also in this group are methods based on fitting dictionaries to the patches of the corrupted image [23, 32] as well as methods based on convolutional sparse coding [31], which can also fit statistical models similar to shallow ConvNets to the reconstructed image [25]. The work [20] investigates the model that combines ConvNet with a self-similarity based denoising and thus also bridges the two groups of methods, but still requires train ing on a hold-out set.

我们的方法明显与基于可学习的卷积网络的图像恢复与图像综合方法相关。同时还与另外一类恢复方法相关，即在保留集上避免训练的算法。这些方法是基于降质图像中类似图像块的联合建模[4,6,10]，这在降质过程复杂多变的时候（如随空间变化的模糊[1]）特别有用。还包括的方法是基于降质图像块适配字典的方法[23,32]以及基于卷积稀疏编码[31]的方法，其也可以将类似于浅层卷积网络的统计模型方法适配到重建图像[25]。文章[20]研究了结合卷积网络和自相似性去噪的方法，所以连接起了这两类方法，但仍然需要在保留集上进行训练。

Overall, the prior imposed by deep ConvNets and investigated in this work seems to be highly related to self-similarity-based and dictionary-based priors. Indeed, as the weights of the convolutional filters are shared across the entire spatial extent of the image this ensures a degree of self-similarity of individual patches that a generative ConvNet can potentially produce. The connections between ConvNets and convolutional sparse coding run even deeper and are investigated in [24] in the context of recognition networks, and more recently in [25], where a single-layer convolutional sparse coding is proposed for reconstruction tasks. The comparison of our approach with [25] (fig. 7 and table 1) however suggests that using deep ConvNet architectures popular in modern deep learning-based approaches may lead to more accurate restoration results at least in some circumstances.

总体上，本文研究的深度卷积网络施加的先验似乎与基于自相似性和基于字典的先验高度相关。确实，由于卷积滤波器的权重在整个空域范围内共享，这确保了个体图像块的自相似性，这也是生成式卷积网络可能产生的结果。卷积网络与卷积稀疏编码的关系更加密切，在[24]中的识别网络任务中进行了研究，最近发表的[25]，其中提出了一个单层卷积稀疏编码方法处理重建任务。我们的方法与[25]的对比（图7和表1）认为，使用深度卷积网络架构至少在一些场景下可以得到更精确的恢复结果。

## 5. Discussion 讨论

We have investigated the success of recent image generator neural networks, teasing apart the contribution of the prior imposed by the choice of architecture from the contribution of the information transferred from external images through learning. As a byproduct, we have shown that fitting a randomly-initialized ConvNet to corrupted images works as a “Swiss knife” for restoration problems. While practically slow (taking several minutes of GPU computation per image), this approach does not require modeling of the degradation process or pre-training.

我们研究了近期的图像生成神经网络，梳理了各种不同架构所施加的先验的贡献。一个副产品是，我们发现将一个随机初始化的卷积网络适配到降质图像可以成为恢复问题的瑞士军刀。虽然实践上比较慢（GPU计算要几分钟才能处理一幅图像），这种方法不需要对降质过程进行建模，或预训练。

Our results go against the common narrative that explain the success of deep learning in image restoration to the ability to learn rather than hand-craft priors; instead, random networks are better hand-crafted priors, and learning builds on this basis. This also validates the importance of developing new deep learning architectures.

普通结论认为深度学习能力重在学习，而我们发现更像手工设计的特征；随机初始化的网络是更好的手工设计的先验，学习是在这个基础上的。这也验证了开发新的深度学习框架的重要性。

**Acknowledgements**. DU and VL are supported by the Ministry of Education and Science of the Russian Federation (grant 14.756.31.0001) and AV is supported by ERC 677195-IDIU.