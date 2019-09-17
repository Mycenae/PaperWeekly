# Image-to-Image Translation with Conditional Adversarial Networks

Phillip Isola et al. Berkeley AI Research

## 0. Abstract

We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.

我们研究了条件对抗网络，作为图像到图像翻译问题的通用目标解。这些网络不仅学习从输入图像到输出图像的映射，还学习到了训练这个映射的的一个损失函数。这样，传统上需要非常不同的损失函数的问题，现在可能用同样的通用方法来解决。我们证明了，这种方法在下面的问题中是有效的，从标记图中合成图像，从边缘图中重建目标，为图像上色，等等。自从放出了与本文相关的pix2pix软件后，很多互联网用户（很多是艺术家）发表了使用这套系统的各自的试验结果，进一步证明了其广泛的应用和易用性，不需要对参数进行调节。我们不需要手动设计我们的映射函数，本文也说明，我们在不需要手工设计损失函数的情况下，也可以得到很好的结果。

## 1. Introduction 引言

Many problems in image processing, computer graphics, and computer vision can be posed as “translating” an input image into a corresponding output image. Just as a concept may be expressed in either English or French, a scene may be rendered as an RGB image, a gradient field, an edge map, a semantic label map, etc. In analogy to automatic language translation, we define automatic image-to-image translation as the task of translating one possible representation of a scene into another, given sufficient training data (see Figure 1). Traditionally, each of these tasks has been tackled with separate, special-purpose machinery (e.g., [16, 25, 20, 9, 11, 53, 33, 39, 18, 58, 62]), despite the fact that the setting is always the same: predict pixels from pixels. Our goal in this paper is to develop a common framework for all these problems.

图像处理、计算机图形学和计算机视觉中的很多问题，都可以看作是将输入图像翻译成对应的输出图像。就像一个概念可以用英语或法语表达一样，一个场景可以渲染为一幅RGB图像，梯度场，边缘图，语义标签图，等等。与自动语言翻译类比，我们定义了自动图像到图像翻译，即在足够的训练数据的情况下，将一个场景的可能表示翻译到另一个表示（如图1所示）。传统上，这些任务每个都由不同的、特定目的的算法所解决，但事实是，这些问题的设置都是一样的：从像素预测像素。本文中我们的目标是对所有这些问题提出一种通用的框架。

Figure 1: Many problems in image processing, graphics, and vision involve translating an input image into a corresponding output image. These problems are often treated with application-specific algorithms, even though the setting is always the same: map pixels to pixels. Conditional adversarial nets are a general-purpose solution that appears to work well on a wide variety of these problems. Here we show results of the method on several. In each case we use the same architecture and objective, and simply train on different data.

The community has already taken significant steps in this direction, with convolutional neural nets (CNNs) becoming the common workhorse behind a wide variety of image prediction problems. CNNs learn to minimize a loss function – an objective that scores the quality of results – and although the learning process is automatic, a lot of manual effort still goes into designing effective losses. In other words, we still have to tell the CNN what we wish it to minimize. But, just like King Midas, we must be careful what we wish for! If we take a naive approach and ask the CNN to minimize the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results [43, 62]. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring. Coming up with loss functions that force the CNN to do what we really want – e.g., output sharp, realistic images – is an open problem and generally requires expert knowledge.

研究者们在这个方向上已经有了一些工作，都是采用CNNs作为这些图像预测问题背后的工具。CNNs学习最小化一个损失函数，即为结果的质量打分的目标，虽然学习过程是自动的，但仍然需要很多工作来设计有效的损失函数。换句话说，我们仍然需要为CNN设计要最小化的函数。但我们的愿望必须要小心！如果我们的方法很简单，使CNN最小化预测的像素和真值像素的欧式距离，将会得到模糊的结果。这是因为，欧式距离的最小化是平均所有可行的输出，这会导致模糊。设计的损失函数，使CNN得到我们真正需要的，如输出锐利真实的图像，仍然是一个开放的问题，一般需要专家知识才能解决。

It would be highly desirable if we could instead specify only a high-level goal, like “make the output indistinguishable from reality”, and then automatically learn a loss function appropriate for satisfying this goal. Fortunately, this is exactly what is done by the recently proposed Generative Adversarial Networks (GANs) [24, 13, 44, 52, 63]. GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss. Blurry images will not be tolerated since they look obviously fake. Because GANs learn a loss that adapts to the data, they can be applied to a multitude of tasks that traditionally would require very different kinds of loss functions.

如果我们只需要指定一个高层的目标，如“使输出与真实之间可以区别开来”，然后可以自动学习到一个损失函数，满足这个目标，那么就非常理想了。幸运的是，最近提出的GANs就可以达成这样的目标。GANs学习到的损失函数，可以对输出图像进行分类，得到是一幅真实图像，或生成的假图像，而同时还会训练一个生成式模型，来最小化其损失。系统不会生成模糊图像，因为这看起来明显是假的。因为GANs学习的损失是自适应于数据的，所以可以用于很多任务，而传统上这些任务需要非常不同的损失函数。

In this paper, we explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model [24]. This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image.

本文中，我们研究来条件GANs。就像GANs学习了数据的生成式模型，cGANs学习了条件生成模型。这使cGANs适用于图像到图像的翻译任务，其中我们对输入图像设置条件，生成对应的输出图像。

GANs have been vigorously studied in the last two years and many of the techniques we explore in this paper have been previously proposed. Nonetheless, earlier papers have focused on specific applications, and it has remained unclear how effective image-conditional GANs can be as a general-purpose solution for image-to-image translation. Our primary contribution is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results. Our second contribution is to present a simple framework sufficient to achieve good results, and to analyze the effects of several important architectural choices. Code is available at https://github.com/phillipi/pix2pix.

GANs在过去两年的研究特别多，我们在本文研究的很多技术都是之前提出来的。尽管如此，之前的文章关注的特定的应用，图像cGANs作为图像到图像翻译的通用目标解，其效用仍然是未知的。我们的主要贡献是，证明了在很多问题上，cGANs都可以得到不错的结果。我们的第二个贡献是，提出了一个简单的框架，但足以得到很好的结果，并研究了几种重要架构选项的效果。代码已开源。

## 2. Related work

**Structured losses for image modeling**. Image-to-image translation problems are often formulated as per-pixel classification or regression (e.g., [39, 58, 28, 35, 62]). These formulations treat the output space as “unstructured” in the sense that each output pixel is considered conditionally independent from all others given the input image. Conditional GANs instead learn a structured loss. Structured losses penalize the joint configuration of the output. A large body of literature has considered losses of this kind, with methods including conditional random fields [10], the SSIM metric [56], feature matching [15], nonparametric losses [37], the convolutional pseudo-prior [57], and losses based on matching covariance statistics [30]. The conditional GAN is different in that the loss is learned, and can, in theory, penalize any possible structure that differs between output and target.

**图像建模的结构化损失**。图像到图像的翻译问题，通常表述为逐像素的分类或回归。这些表述将输出空间作为“无结构的”进行对待，即在给定输入图像的情况下，每个输出的像素与其他所有像素都是条件独立的。cGANs学习的则是一个结构化的损失。结构化损失惩罚的是输出的联合配置。很多文献都考虑的是这种损失，如条件随机场[10]，SSIM度量标准[56]，特征匹配[15]，非参数损失[37]，卷积伪先验[57]，和基于匹配协方差统计的损失[30]。cGAN是不同的，其损失函数是学习到的，理论上可以惩罚任何可能的输出与目标之间不同的结构。

**Conditional GANs**. We are not the first to apply GANs in the conditional setting. Prior and concurrent works have conditioned GANs on discrete labels [41, 23, 13], text [46], and, indeed, images. The image-conditional models have tackled image prediction from a normal map [55], future frame prediction [40], product photo generation [59], and image generation from sparse annotations [31, 48] (c.f. [47] for an autoregressive approach to the same problem). Several other papers have also used GANs for image-to-image mappings, but only applied the GAN unconditionally, relying on other terms (such as L2 regression) to force the output to be conditioned on the input. These papers have achieved impressive results on inpainting [43], future state prediction [64], image manipulation guided by user constraints [65], style transfer [38], and superresolution [36]. Each of the methods was tailored for a specific application. Our framework differs in that nothing is application-specific. This makes our setup considerably simpler than most others.

**条件GANs**。我们并不是第一个在条件设置下使用GANs的。之前和同时的工作，有将cGANs应用到离散标签的，文本的，和图像的。以图像为条件的模型处理图像预测问题时，有的使用规范化图，有的进行未来帧的预测，有的进行商品照片生成，有的从系数标注中生成图像。其他几篇文献也使用GANs进行图像到图像的映射，但是在无条件的情况下使用的GAN，依靠其他项（如L2回归）来使输出以输入为条件。这些文章在下面的领域取得来很好的结果，如修补，未来状态预测，采用用户约束来引导图像变换，风格迁移，和超分辨率。每一种方法都是为特定的应用定制的。我们的框架是不同的，因为不是某个应用特有的。这使得我们的设置比其他的算法要简单很多。

Our method also differs from the prior works in several architectural choices for the generator and discriminator. Unlike past work, for our generator we use a “U-Net”-based architecture [50], and for our discriminator we use a convolutional “PatchGAN” classifier, which only penalizes structure at the scale of image patches. A similar PatchGAN architecture was previously proposed in [38] to capture local style statistics. Here we show that this approach is effective on a wider range of problems, and we investigate the effect of changing the patch size.

我们的方法与之前的工作还有几点不同，主要体现在生成器和判别器的架构选择上。与过去的工作不同，我们的生成器，使用一种基于U-Net的架构，我们的判别器使用了一种卷积的PatchGAN分类器，只在图像块的尺度上惩罚结构。[38]提出了类似的PatchGAN架构，用来捕获局部风格统计。这里我们展示了，这种方法在更光范围内的问题中也是有效的，我们研究了改变图像块大小的效果。

## 3. Method

GANs are generative models that learn a mapping from random noise vector z to output image y, G : z → y [24]. In contrast, conditional GANs learn a mapping from observed image x and random noise vector z, to y, G : {x,z} → y. The generator G is trained to produce outputs that cannot be distinguished from “real” images by an adversarially trained discriminator, D, which is trained to do as well as possible at detecting the generator’s “fakes”. This training procedure is diagrammed in Figure 2.

GANs是一种生成式模型，学习的是从随机噪声向量z到输出图像y的映射，G : z → y。比较之下，cGANs学习的是从观察图像x和随机噪声z到输出图像y的映射，G : {x,z} → y。生成器G训练的目标是，其生成的输出，要尽量与真实图像难以区分开来，而经过对抗训练得到的判别器D，其训练目标就是尽可能检测到生成器生成的假图像。这种训练过程如图2所示。

Figure 2: Training a conditional GAN to map edges→photo. The discriminator, D, learns to classify between fake (synthesized by the generator) and real {edge, photo} tuples. The generator, G, learns to fool the discriminator. Unlike an unconditional GAN, both the generator and discriminator observe the input edge map.

### 3.1. Objective

The objective of a conditional GAN can be expressed as: cGAN的目标函数可以表述为：

$$L_{cGAN} (G, D) =E_{x,y} [log D(x, y)] + E_{x,z} [log(1 − D(x, G(x,z))]$$(1)

where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. $G∗ = arg min_G max_D L_{cGAN} (G, D)$. 其中G要最小化目标函数，而对抗函数D则尝试去最大化目标函数，即$G∗ = arg min_G max_D L_{cGAN} (G, D)$。

To test the importance of conditioning the discriminator, we also compare to an unconditional variant in which the discriminator does not observe x: 为测验为判别器增加条件的重要性，我们还与一个变体的无条件GAN进行对比，其中判别器没有对x的观测：

$$L_{GAN} (G, D) =E_y [log D(y)] + E_{x,z} [log(1 − D(G(x, z))]$$(2)

Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance [43]. The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:

之前的方法发现，将GAN的目标函数与传统损失，如L2距离损失混合在一起，会很有好处[43]。判别器的任务保持不变，但生成器的任务不仅是要骗过判别器，而且还要在L2损失下接近真值输出。我们研究了这种选项，使用L1距离，而没有使用L2距离，因为L2会鼓励图像模糊：

$$L_{L1} (G) = E_{x,y,z} [||y−G(x,z)||_1]$$(3)

Our final objective is 最终的目标函数为：

$$G∗ = arg min_G max_D L_{cGAN}(G,D) + λL_{L1}(G)$$(4)

Without z, the net could still learn a mapping from x to y, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function. Past conditional GANs have acknowledged this and provided Gaussian noise z as an input to the generator, in addition to x (e.g., [55]). In initial experiments, we did not find this strategy effective – the generator simply learned to ignore the noise – which is consistent with Mathieu et al. [40]. Instead, for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time. Despite the dropout noise, we observe only minor stochasticity in the output of our nets. Designing conditional GANs that produce highly stochastic output, and thereby capture the full entropy of the conditional distributions they model, is an important question left open by the present work.

在没有z的情况下，网络也能学习到从x到y的映射，但会生成确定性的输出，因此不会形成一个分布，而只是一个delta函数。过去的cGANs承认这一点，因此除了x以外，将高斯噪声z也作为生成器的输入。在初步的试验中，我们没有发现这个策略很有效，生成器只是学习忽略了这个噪声，这个结果与Mathieu等的结果一致。因此，在我们最终的模型中，我们只以dropout的形式提供噪声，在我们的生成器的若干层中使用，训练和测试时都进行使用。虽然是dropout噪声的形式，我们在网络输出中只观察到了很少的随机性。设计生成高度随机性输出的cGANs，因此能捕获建模的条件分布的完全熵，这仍然是本文留下的一个重要的开放问题。

### 3.2. Network architectures

We adapt our generator and discriminator architectures from those in [44]. Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu [29]. Details of the architecture are provided in the supplemental materials online, with key features discussed below.

我们采用DCGAN中的生成器和判别器架构。生成器和判别器使用的模块的形式是卷积-BN-ReLU[29]。架构细节在附加材料中，关键特征下面进行讨论。

#### 3.2.1 Generator with skips

A defining feature of image-to-image translation problems is that they map a high resolution input grid to a high resolution output grid. In addition, for the problems we consider, the input and output differ in surface appearance, but both are renderings of the same underlying structure. Therefore, structure in the input is roughly aligned with structure in the output. We design the generator architecture around these considerations.

图像到图像的翻译问题的主要特征是，将高分辨的输入，映射到高分辨率的输出。另外，对于我们考虑的问题，输入和输出的区别是surface appearance，但两者都是相同的潜在结构的渲染呈现。因此，输入中的结构与输出中的结构是大致对齐的。我们基于这些考虑，设计生成器的架构。

Many previous solutions [43, 55, 30, 64, 59] to problems in this area have used an encoder-decoder network [26]. In such a network, the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed. Such a network requires that all information flow pass through all the layers, including the bottleneck. For many image translation problems, there is a great deal of low-level information shared between the input and output, and it would be desirable to shuttle this information directly across the net. For example, in the case of image colorization, the input and output share the location of prominent edges.

这个领域很多之前的工作都使用了编码器-解码器网络[26]。在这样的网络中，输入经过一些层，逐渐的进行降采样，直到一个瓶颈层，在这一点上整个过程进行逆转。这样的网络需要所有信息都流经所有的层，包括瓶颈层。对于很多图像翻译问题，有很多底层信息是输入和输出所共享的，如果这些信息能够在网络中传播，就是非常理想的。比如，在图像上色的情形中，主要边缘的位置是输入和输出所共享的。

To give the generator a means to circumvent the bottleneck for information like this, we add skip connections, following the general shape of a “U-Net” [50]. Specifically, we add skip connections between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i.

为使生成器有一种方法，可以避免信息流经瓶颈，我们按照U-Net的一般形状，增加了skip连接。具体来说，我们在层i与层n-i之间增加了skip连接，其中n是层的总数。每个跳跃连接只是将层i中的所有通道与层n-i的进行拼接。

Figure 3: Two choices for the architecture of the generator. The “U-Net” [50] is an encoder-decoder with skip connections be- tween mirrored layers in the encoder and decoder stacks.

#### 3.2.2 Markovian discriminator (PatchGAN)

It is well known that the L2 loss – and L1, see Figure 4 – produces blurry results on image generation problems [34]. Although these losses fail to encourage high-frequency crispness, in many cases they nonetheless accurately capture the low frequencies. For problems where this is the case, we do not need an entirely new framework to enforce correctness at the low frequencies. L1 will already do.

众所周知，L2损失和L1损失在图像生成问题中会产生模糊的结果。虽然这些损失不会鼓励高频的锐度，尽管如此，在很多情况下它们会准确的捕获到低频分量。对于这种情况的问题，我们不需要一个全新的框架来得到准确的低频分量，L1损失就可以得到这样的结果。

This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness (Eqn. 4). In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of D.

这样就只需要GAN判别器关注高频结构，而低频分量的正确性就依靠L1项（式4）。为对高频分量进行建模，那么将我们的注意力限制在局部图像块的结构，就足够了。因此，我们设计了一种判别器架构，我们称之为PatchGAN，只对图像块的尺度上的结构进行惩罚。这种判别器对图像中N×N个图像块进行逐个分类，是否是真实图像还是合成图像。我们以卷积的形式，在整个图像中应用这个判别器，并将所有的输出进行平均，得到D最终的输出。

In Section 4.4, we demonstrate that N can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily large images.

在4.4节中，我们证明了N的数值比完整图像大小小很多，也可以得到很高质量的结果。这是有优势的，因为更小的PatchGAN其参数更少，运行的更快，可以应用于任意大小的图像中。

Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. This connection was previously explored in [38], and is also the common assumption in models of texture [17, 21] and style [16, 25, 22, 37]. Therefore, our PatchGAN can be understood as a form of texture/style loss.

这样的判别器，将图像建模为一个Markov随机场，假设图像块直径分隔的像素之间是独立的。[38]研究过这种连接，这在纹理的模型中和风格的模型中是常见的假设。因此，我们的PatchGAN可以理解为一种纹理/风格损失。

### 3.3. Optimization and inference

To optimize our networks, we follow the standard approach from [24]: we alternate between one gradient descent step on D, then one step on G. As suggested in the original GAN paper, rather than training G to minimize log(1 − D(x, G(x, z)), we instead train to maximize log D(x, G(x, z)) [24]. In addition, we divide the objective by 2 while optimizing D, which slows down the rate at which D learns relative to G. We use minibatch SGD and apply the Adam solver [32], with a learning rate of 0.0002, and momentum parameters β1 = 0.5, β2 = 0.999.

为优化我们的网络，我们采用[24]中的标准方法：轮流进行D的一步梯度下降，和G的一步梯度下降。如原始GAN文章中所述，我们训练G不是去最小化log(1 − D(x, G(x, z))，而是来最大化log D(x, G(x, z))。另外，我们在优化D的同时，将目标函数除以2，这会降低D相对于G的学习速度。我们使用mini-batch SGD，使用Adam solver，学习速率为0.0002，动量参数为β1 = 0.5, β2 = 0.999。

At inference time, we run the generator net in exactly the same manner as during the training phase. This differs from the usual protocol in that we apply dropout at test time, and we apply batch normalization [29] using the statistics of the test batch, rather than aggregated statistics of the training batch. This approach to batch normalization, when the batch size is set to 1, has been termed “instance normalization” and has been demonstrated to be effective at image generation tasks [54]. In our experiments, we use batch sizes between 1 and 10 depending on the experiment.

在推理时，我们运行生成器的方式，与在训练阶段是一样的。这与通常情况下，在测试时使用dropout的不太一样，我们使用测试批次的统计量来使用BN，而不是使用训练批次的累积统计。BN的这种方法，当batch size设为1时，称为“实例归一化”，已经证明在图像生成任务中是很有效的[54]。在我们的试验中，不同的试验使用的batch size从1到10有所不同。

## 4. Experiments

To explore the generality of conditional GANs, we test the method on a variety of tasks and datasets, including both graphics tasks, like photo generation, and vision tasks, like semantic segmentation:

为研究cGANs的泛化性，我们在很多任务和数据集上测试本方法，包括图形任务和视觉任务，如图像生成，语义分隔：

- Semantic labels↔photo, trained on the Cityscapes dataset [12]. 语义标签与图像的相互转换，在Cityscapes数据集上训练。
- Architectural labels→photo, trained on CMP Facades [45]. 架构标签到图像的转换，在CMP Facades数据集上训练。
- Map↔aerial photo, trained on data scraped from Google Maps. 地图与空中图像的相互转换，用从谷歌地图上爬取的数据进行训练。
- BW→color photos, trained on [51]. 二值图到彩色图的转换，在[51]上进行的训练。
- Edges→photo, trained on data from [65] and [60]; binary edges generated using the HED edge detector [58] plus postprocessing. 边缘到图像的转换，在[65,60]的数据上进行的训练；二值边缘的生成使用的是HED边缘检测器[58]加上后处理。
- Sketch→photo: tests edges→photo models on human-drawn sketches from [19]. 概略图到图像的转换，在[19]中人类画的草图上测试边缘到图像变换的模型。
- Day→night, trained on [33]. 白天的图到夜晚的图的转换，在[33]上训练。
- Thermal→color photos, trained on data from [27]. 热力图到彩色图像的转换，在[27]的数据上进行训练。
- Photo with missing pixels→inpainted photo, trained on Paris StreetView from [14]. 缺失像素的图像到修补图像的转换，在[14]的Paris StreetView上进行训练。

Details of training on each of these datasets are provided in the supplemental materials online. In all cases, the input and output are simply 1-3 channel images. Qualitative results are shown in Figures 8, 9, 11, 10, 13, 14, 15, 16, 17, 18, 19, 20. Several failure cases are highlighted in Figure 21. More comprehensive results are available at https://phillipi.github.io/pix2pix/.

在每个数据集上的训练细节详见附加材料。在所有情形下，输入和输出都只是1-3通道的图像。定性结果如图8-20所示。一些失败的情况如图21所示。更多结果见网站。

**Data requirements and speed**. We note that decent results can often be obtained even on small datasets. Our facade training set consists of just 400 images (see results in Figure 14), and the day to night training set consists of only 91 unique webcams (see results in Figure 15). On datasets of this size, training can be very fast: for example, the results shown in Figure 14 took less than two hours of training on a single Pascal Titan X GPU. At test time, all models run in well under a second on this GPU.

**数据需求和速度**。我们要说明的是，在更小的数据集上，也能得到不错的结果。我们的facade训练集只有400幅图像（结果如图14所示），白天到夜晚的训练集只包含91个摄像头数据（结果如图15所示）。在这样规模的数据集上，训练速度会很快：比如，图14的结果训练时间只有不到2小时，在一块Pascal Titan X GPU上。在测试时，所有模型在这个GPU上运行时间都远小于1秒钟。

### 4.1. Evaluation metrics

Evaluating the quality of synthesized images is an open and difficult problem [52]. Traditional metrics such as per-pixel mean-squared error do not assess joint statistics of the result, and therefore do not measure the very structure that structured losses aim to capture.

评估合成图像的质量是一个开放的困难问题[52]。传统的度量标准，如逐像素的均方误差，没有评估结果的联合统计值，因此没有衡量结构化损失想要捕获的结构。

To more holistically evaluate the visual quality of our results, we employ two tactics. First, we run “real vs. fake” perceptual studies on Amazon Mechanical Turk (AMT). For graphics problems like colorization and photo generation, plausibility to a human observer is often the ultimate goal. Therefore, we test our map generation, aerial photo generation, and image colorization using this approach.

为更全面的评估我们结果的视觉质量，我们采用了两种策略。首先，在AMT上进行“真实vs合成”的感觉研究。对于图形问题，如上色和图像生成，人类观察者得到的合理性，通常是终极目标。因此，我们使用这种方法来测试地图生成、空中图像生成和图像着色的应用。

Second, we measure whether or not our synthesized cityscapes are realistic enough that off-the-shelf recognition system can recognize the objects in them. This metric is similar to the “inception score” from [52], the object detection evaluation in [55], and the “semantic interpretability” measures in [62] and [42].

第二，我们度量合成的街景是不是足够真实的方法是，采用现有的识别系统，看是否可以识别其中的目标。这种度量标准与[52]中的inception score类似，与[55]中的目标检测评估类似，与[62,42]中的语义解释性度量类似。

**AMT perceptual studies**. For our AMT experiments, we followed the protocol from [62]: Turkers were presented with a series of trials that pitted a “real” image against a “fake” image generated by our algorithm. On each trial, each image appeared for 1 second, after which the images disappeared and Turkers were given unlimited time to respond as to which was fake. The first 10 images of each session were practice and Turkers were given feedback. No feedback was provided on the 40 trials of the main experiment. Each session tested just one algorithm at a time, and Turkers were not allowed to complete more than one session. ∼ 50 Turkers evaluated each algorithm. Unlike [62], we did not include vigilance trials. For our colorization experiments, the real and fake images were generated from the same grayscale input. For map↔aerial photo, the real and fake images were not generated from the same input, in order to make the task more difficult and avoid floor-level results. For map↔aerial photo, we trained on 256 × 256 resolution images, but exploited fully-convolutional translation (described above) to test on 512 × 512 images, which were then downsampled and presented to Turkers at 256 × 256 resolution. For colorization, we trained and tested on 256 × 256 resolution images and presented the results to Turkers at this same resolution.

**AMT感知研究**。对于我们的AMT试验，我们按照[62]中的协议进行：Turkers要进行一系列试验，分辨真实图像与我们的算法生成的合成虚假图像。在每个试验中，每幅图像出现1秒钟，然后图像消失，Turkers有无限的时间来给出哪幅图像是假的。每个session的前10幅图像是练习，会给Turkers以反馈。主要试验的40次判断是没有反馈的。每个session一次测试一种算法，Turkers的测试不能超过一个session。大约50个Turkers评估一个算法。与[62]不同的是，我们没有包含警惕性测试。对于着色试验，真实和虚假图像是从相同的灰度输入生成的。对于地图与空中图像的相互转换，并不是从同一个输入生成的真实和虚假图像，主要目的是使得任务更困难，避免最低水平的结果。对于地图与空中图像的相互转换，我们在256×256分辨率的图像上进行训练，但利用全卷积变换在512×512图像上进行测试，然后降采样，给Turkers呈现的是256×256的图像。对于着色，我们训练和测试都是在256×256分辨率上，给Turkers的也是同样分辨率的。

**“FCN-score”**. While quantitative evaluation of generative models is known to be challenging, recent works [52, 55, 62, 42] have tried using pre-trained semantic classifiers to measure the discriminability of the generated stimuli as a pseudo-metric. The intuition is that if the generated images are realistic, classifiers trained on real images will be able to classify the synthesized image correctly as well. To this end, we adopt the popular FCN-8s [39] architecture for semantic segmentation, and train it on the cityscapes dataset. We then score synthesized photos by the classification accuracy against the labels these photos were synthesized from.

**FCN分数**。对生成式模型的定量评估是非常困难的，但最近的工作尝试使用预训练的语义分类器，来衡量生成的激励的区分性，作为一个伪度量。直觉上来说，如果生成的图像是真实的，在真实图像上训练出来的分类器，也会正确的对合成图像进行分类。为此，我们采用流行的FCN-8s[39]进行语义分割，并在cityscapes数据集上进行训练。然后我们对合成图像进行打分，方法是，相对于合成这些图像的标签来说，分类的准确率。

### 4.2. Analysis of the objective function

Which components of the objective in Eqn. 4 are important? We run ablation studies to isolate the effect of the L1 term, the GAN term, and to compare using a discriminator conditioned on the input (cGAN, Eqn. 1) against using an unconditional discriminator (GAN, Eqn. 2).

式4的目标函数中，哪个组成部分是重要的？我们进行分离对比试验，来观察L1项的效果，GAN项的效果，并比较以输入为条件的判别器（cGAN，式1），与无条件判别器的（GAN，式2）的效果。

Figure 4 shows the qualitative effects of these variations on two labels→photo problems. L1 alone leads to reasonable but blurry results. The cGAN alone (setting λ = 0 in Eqn. 4) gives much sharper results but introduces visual artifacts on certain applications. Adding both terms together (with λ = 100) reduces these artifacts.

图4给出了这些变化在两个标签到图像转换问题上的定性效果。只使用L1损失，可以得到合理但模糊的结果。只有cGAN损失（在式4中设λ=0）可以得到锐利的多的结果，但在一些应用中会带来视觉上杂质。两项结合(λ=100)会减少这些杂质。

We quantify these observations using the FCN-score on the cityscapes labels→photo task (Table 1): the cGAN-based objectives achieve higher scores, indicating that the synthesized images include more recognizable structure. We also test the effect of removing conditioning from the discriminator (labeled as GAN). In this case, the loss does not penalize mismatch between the input and output; it only cares that the output look realistic. This variant results in poor performance; examining the results reveals that the generator collapsed into producing nearly the exact same output regardless of input photograph. Clearly, it is important, in this case, that the loss measure the quality of the match between input and output, and indeed cGAN performs much better than GAN. Note, however, that adding an L1 term also encourages that the output respect the input, since the L1 loss penalizes the distance between ground truth outputs, which correctly match the input, and synthesized outputs, which may not. Correspondingly, L1+GAN is also effective at creating realistic renderings that respect the input label maps. Combining all terms, L1+cGAN, performs similarly well.

我们使用FCN分数对这些观察进行量化，针对的任务是在cityscapes上的标签到图像的转换任务（表1）：基于cGAN的目标函数取得了更高的分数，说明合成的图像包含更容易识别的结构。我们还测试了从判别器中去除条件的效果（标为GAN）。在这种情况下，损失没有惩罚输入和输出之间的不匹配；只是关注输出看起来是否像真的。这种变体得到很差的结果；检查结果发现，生成器退化为，不论输入图像是什么样子的，生成的输出几乎完全一样。这就很清楚了，在这种情况下，损失函数一定要衡量输入和输出之间的匹配程度，这非常重要，在这方面，cGAN比GAN的性能要好的多。但是，要注意的是，增加L1项也会鼓励输出要与输入接近，因为L1损失惩罚的是真值输出和合成输出之间的距离，真值输出与输入是正确匹配的，而合成输出则不是。对应的，L1+GAN也可以高效的生成真实的呈现，与输入标签图匹配很好。将所有的项结合在一起，L1+cGAN的表现也非常好。

Table 1: FCN-scores for different losses, evaluated on Cityscapes labels↔photos.

**Colorfulness**. A striking effect of conditional GANs is that they produce sharp images, hallucinating spatial structure even where it does not exist in the input label map. One might imagine cGANs have a similar effect on “sharpening” in the spectral dimension – i.e. making images more colorful. Just as L1 will incentivize a blur when it is uncertain where exactly to locate an edge, it will also incentivize an average, grayish color when it is uncertain which of several plausible color values a pixel should take on. Specially, L1 will be minimized by choosing the median of the conditional probability density function over possible colors. An adversarial loss, on the other hand, can in principle become aware that grayish outputs are unrealistic, and encourage matching the true color distribution [24]. In Figure 7, we investigate whether our cGANs actually achieve this effect on the Cityscapes dataset. The plots show the marginal distributions over output color values in Lab color space. The ground truth distributions are shown with a dotted line. It is apparent that L1 leads to a narrower distribution than the ground truth, confirming the hypothesis that L1 encourages average, grayish colors. Using a cGAN, on the other hand, pushes the output distribution closer to the ground truth.

**色彩度**。cGANs的一个惊人效果是，可以生成锐利的图像，逼真的空间结构，即使是在输入标签图中是不存在的。可以想象，cGANs在谱维度上也有类似的锐化效果，即，使得图像更加多彩。就像L1，在不确定哪个确切位置会有边缘时，会鼓励模糊效果，当一个像素可能有几个可能的色彩值而不确定哪个时，会鼓励一个平均的、更像灰度的色彩值。具体的，L1会在可能的色彩上，选择条件概率密度函数的中值，得到L1的最小化值。而对抗损失，则会意识到，像灰度的输出是不真实的，鼓励匹配真实色彩分布[24]。在图7中，我们研究的是，cGANs是否真的得到在Cityscapes数据集上的效果。这些图说明了，在Lab色彩空间中，与输出色彩值相比，分布差异是很小的。真值分布是用虚线表示的。很明显，L1得到的分布比真值更窄，确认了关于L1的假设，即L1鼓励平均的、类似灰度的色彩。而使用cGAN，则得到的输出分布与真值非常接近。

### 4.3. Analysis of the generator architecture

A U-Net architecture allows low-level information to shortcut across the network. Does this lead to better results? Figure 5 and Table 2 compare the U-Net against an encoder-decoder on cityscape generation. The encoder-decoder is created simply by severing the skip connections in the U-Net. The encoder-decoder is unable to learn to generate realistic images in our experiments. The advantages of the U-Net appear not to be specific to conditional GANs: when both U-Net and encoder-decoder are trained with an L1 loss, the U-Net again achieves the superior results.

U-Net架构使得底层信息在网络中有捷径流动。这是否会得到更好的结果呢？图5和表2比较了在cityscapes生成任务上，U-Net结构和编码器-解码器架构得到的结果。编码器-解码器结构就是将U-Net中的跳跃连接切断得到的。在我们的试验中，编码器-解码器的学习，不能产生真实的图像。U-Net的优势并不是只是在cGANs中才存在：当U-Net和编码器-解码器都用L1损失来训练时，U-Net再次取得了更好的结果。

Figure 5: Adding skip connections to an encoder-decoder to create a “U-Net” results in much higher quality results.

Table 2: FCN-scores for different generator architectures (and objectives), evaluated on Cityscapes labels↔photos. (U-net (L1-cGAN) scores differ from those reported in other tables since batch size was 10 for this experiment and 1 for other tables, and random variation between training runs.)

### 4.4. From PixelGANs to PatchGANs to ImageGANs

We test the effect of varying the patch size N of our discriminator receptive fields, from a 1 × 1 “PixelGAN” to a full 286 × 286 “ImageGAN”(We achieve this variation in patch size by adjusting the depth of the GAN discriminator. Details of this process, and the discriminator architectures, are provided in the supplemental materials online). Figure 6 shows qualitative results of this analysis and Table 3 quantifies the effects using the FCN-score. Note that elsewhere in this paper, unless specified, all experiments use 70 × 70 PatchGANs, and for this section all experiments use an L1+cGAN loss.

我们对判别器感受野的块大小N进行变化，测试其效果，从1×1的PixelGAN，到完整分辨率的286×286的ImageGAN（我们通过调整GAN判别器的深度，来得到图像块大小变化的效果。这个过程的细节，和判别器的架构，在附加材料中有描述）。在图6中给出这个分析的定性结果，表3使用FCN分数量化来这个效果。注意，在本文的其他地方，除非另有指定，所有的试验都使用70×70的PatchGANs，在本节中，所有试验都使用L1+cGAN的损失。

Figure 6: Patch size variations. Uncertainty in the output manifests itself differently for different loss functions. Uncertain regions become blurry and desaturated under L1. The 1x1 PixelGAN encourages greater color diversity but has no effect on spatial statistics. The 16x16 PatchGAN creates locally sharp results, but also leads to tiling artifacts beyond the scale it can observe. The 70x70 PatchGAN forces outputs that are sharp, even if incorrect, in both the spatial and spectral (colorfulness) dimensions. The full 286x286 ImageGAN produces results that are visually similar to the 70x70 PatchGAN, but somewhat lower quality according to our FCN-score metric (Table 3). Please see https://phillipi.github.io/pix2pix/ for additional examples.

Table 3: FCN-scores for different receptive field sizes of the dis- criminator, evaluated on Cityscapes labels→photos. Note that input images are 256 × 256 pixels and larger receptive fields are padded with zeros.

The PixelGAN has no effect on spatial sharpness but does increase the colorfulness of the results(quantified in Figure 7). For example, the bus in Figure 6 is painted gray when the net is trained with an L1 loss, but becomes red with the PixelGAN loss. Color histogram matching is a common problem in image processing [49], and PixelGANs may be a promising lightweight solution.

PixelGAN在空间锐利度上毫无效果，但增加了结果的色彩度（图7中有量化结果）。比如，在网络使用L1损失进行训练时，图6中的公交车变成了灰色，但在使用PixelGAN损失时，会变成红色。色彩直方图的匹配在图像处理中是一个通用问题，PixelGAN可能是一个轻量级解决方案。

Using a 16×16 PatchGAN is sufficient to promote sharp outputs, and achieves good FCN-scores, but also leads to tiling artifacts. The 70 × 70 PatchGAN alleviates these artifacts and achieves slightly better scores. Scaling beyond this, to the full 286 × 286 ImageGAN, does not appear to improve the visual quality of the results, and in fact gets a considerably lower FCN-score (Table 3). This may be because the ImageGAN has many more parameters and greater depth than the 70 × 70 PatchGAN, and may be harder to train.

使用16x16大小的PatchGAN可以得到更锐利的输出，得到不错的FCN分数，但也带来了一些异物。70x70大小的PatchGAN缓解了异物的问题，得到的分数略好一些。进一步放大，完整的286x286大小的ImageGAN，似乎没有改进结果的视觉质量，但实际上得到的FCN分数更低（表3）。这可能是因为，ImageGAN的参数要多很多，深度也要比70x70的PatchGAN要更深，所以训练起来更困难。

**Fully-convolutional translation**. An advantage of the PatchGAN is that a fixed-size patch discriminator can be applied to arbitrarily large images. We may also apply the generator convolutionally, on larger images than those on which it was trained. We test this on the map↔aerial photo task. After training a generator on 256×256 images, we test it on 512 × 512 images. The results in Figure 8 demonstrate the effectiveness of this approach.

**全卷积翻译**。PatchGAN的一个优势是，固定大小的块判别器可以应用于任意大的图像上。我们也可以将生成器以卷积的形式应用到更大的图像上，比训练时所用的图像大小要更大。我们在地图与空中图像的相互转换任务上进行测试。我们在256x256大小的图像上训练了一个生成器，并在512x512图像上进行测试。图8中的结果证明了这种方法更有效。

### 4.5 Perceptual validation 感知验证

We validate the perceptual realism of our results on the tasks of map↔aerial photograph and grayscale→color. Results of our AMT experiment for map↔photo are given in Table 4. The aerial photos generated by our method fooled participants on 18.9% of trials, significantly above the L1 baseline, which produces blurry results and nearly never fooled participants. In contrast, in the photo->map direction our method only fooled participants on 6.1% of trials, and this was not significantly different than the performance of the L1 baseline (based on bootstrap test). This may be because minor structural errors are more visible in maps, which have rigid geometry, than in aerial photographs, which are more chaotic.

我们在地图与空中图像的相互转换，和灰度图像与彩色图像的相互转换上，验证我们的结果的感知的真实性。我们在地图与空中图像的相互转换试验的AMT结果如表4所示。我们的方法生成的空中图像在18.9%的测验中骗过了参与者，明显高于L1基准，它生成了很模糊的结果，几乎没有骗到参与者。比较之下，空中图像到地图的转换中，我们的方法只在6.1%的测验中骗过了参与者，这与L1基准的性能差别不大。这可能是因为，在地图中，很小的结构性误差也会很明显，因为地图有很严格的几何关系，而空中图像的几何关系更混乱一些。

We trained colorization on ImageNet [51], and tested on the test split introduced by [62, 35]. Our method, with L1+cGAN loss, fooled participants on 22.5% of trials (Table 5). We also tested the results of [62] and a variant of their method that used an L2 loss (see [62] for details). The conditional GAN scored similarly to the L2 variant of [62] (difference insignificant by bootstrap test), but fell short of [62]’s full method, which fooled participants on 27.8% of trials in our experiment. We note that their method was specifically engineered to do well on colorization.

我们在ImageNet上训练上色，在[62,35]中的测试分割集中进行测试。我们的方法，使用的是L1+cGAN损失，在22.5%的测试中骗过了参与者（表5）。我们还测试了[62]的结果，以及它们方法的一个变体，使用的是L2损失。cGAN与[62]的L2变体的得分类似，但却不如[62]的完整方法，在27.8%的测验中骗过了参与者。我们注意到，他们的方法是专门为着色而设计的，因此效果更好。

Figure 9: Colorization results of conditional GANs versus the L2 regression from [62] and the full method (classification with rebalancing) from [64]. The cGANs can produce compelling colorizations (first two rows), but have a common failure mode of producing a grayscale or desaturated result (last row).

### 4.6. Semantic segmentation

Conditional GANs appear to be effective on problems where the output is highly detailed or photographic, as is common in image processing and graphics tasks. What about vision problems, like semantic segmentation, where the output is instead less complex than the input?

cGANs在输出细节很多或是图像的时候，似乎非常有效，这在图像处理和图形学中非常普遍。在视觉问题中，如语义分割，其输出比输入要简单的多的情况下，会怎样呢？

To begin to test this, we train a cGAN (with/without L1 loss) on cityscape photo→labels. Figure 10 shows qualitative results, and quantitative classification accuracies are reported in Table 6. Interestingly, cGANs, trained without the L1 loss, are able to solve this problem at a reasonable degree of accuracy. To our knowledge, this is the first demonstration of GANs successfully generating “labels”, which are nearly discrete, rather than “images”, with their continuous-valued variation(Note that the label maps we train on are not exactly discrete valued, as they are resized from the original maps using bilinear interpolation and saved as jpeg images, with some compression artifacts). Although cGANs achieve some success, they are far from the best available method for solving this problem: simply using L1 regression gets better scores than using a cGAN, as shown in Table 6. We argue that for vision problems, the goal (i.e. predicting output close to the ground truth) may be less ambiguous than graphics tasks, and reconstruction losses like L1 are mostly sufficient.

为开始这样的测试，我们在cityscapes的图像到标签任务上训练了一个cGAN（有L1损失/没有L1损失）。图10给出了定性的结果，定量的分类准确率如表6所示。有趣的是，cGANs在没有L1损失训练的情况下，可以以一定的正确率解决这个问题。据我们所知，这是GANs成功的生成标签的第一次展示，几乎是其连续值变化的离散化（注意，我们训练所用的标签图并不是严格的离散值的，因为它们是从原始图中使用双线性插值变换大小得到的，保存成了jpeg图像，有一些压缩得到的杂质）。虽然cGANs取得了一些成功，但还远远不是解决这类问题最好的可用方法：只使用L1回归，得到的分数就比使用cGAN要好，如表6所示。我们认为，对于视觉问题，其目标（即，预测的输出要与真值接近）比图形学任务歧义更加少，重建损失如L1就已经足够了。

Figure 10: Applying a conditional GAN to semantic segmentation. The cGAN produces sharp images that look at glance like the ground truth, but in fact include many small, hallucinated objects.

### 4.7. Community-driven Research

Since the initial release of the paper and our pix2pix codebase, the Twitter community, including computer vision and graphics practitioners as well as visual artists, have successfully applied our framework to a variety of novel image-to-image translation tasks, far beyond the scope of the original paper. Figure 11 and Figure 12 show just a few examples from the #pix2pix hashtag, including Background removal, Palette generation, Sketch → Portrait, Sketch→Pokemon, ”Do as I Do” pose transfer, Learning to see: Gloomy Sunday, as well as the bizarrely popular #edges2cats and #fotogenerator. Note that these applications are creative projects, were not obtained in controlled, scientific conditions, and may rely on some modifications to the pix2pix code we released. Nonetheless, they demonstrate the promise of our approach as a generic commodity tool for image-to-image translation problems.

自从我们放出论文和pix2pix代码后，很多人成功的将我们的框架应用于大量新颖的图像到图像翻译任务，远超过了原始论文的范围。图11和图12给出了几个例子，包括Background removal, Palette generation, Sketch → Portrait, Sketch→Pokemon, ”Do as I Do” pose transfer, Learning to see: Gloomy Sunday, 以及奇怪流行的 #edges2cats和#fotogenerator. 注意这些应用都是有创意的工程，并不是在可控的，科研条件下得到的，可能对我们放出的pix2pix代码有一些修改。尽管如此，他们展示了我们方法的希望，即图像到图像翻译问题的通用工具。

## 5. Conclusion

The results in this paper suggest that conditional adversarial networks are a promising approach for many image-to-image translation tasks, especially those involving highly structured graphical outputs. These networks learn a loss adapted to the task and data at hand, which makes them applicable in a wide variety of settings.

本文的结果说明，cGAN是很多图像到图像翻译任务有希望的方法，尤其是那些涉及到高度结构化的图形输出的。这些网络学习到一个损失，与任务和数据相适配，这使其可以在很广泛的设置中都可以适用。

## 6. Appendix

### 6.1. Network architectures

We adapt our network architectures from those in [44]. Code for the models is available at https://github.com/phillipi/pix2pix. 我们从[44]中改造出我们的网络架构。代码已经开源。

Let Ck denote a Convolution-BatchNorm-ReLU layer with k filters. CDk denotes a Convolution-BatchNorm-Dropout-ReLU layer with a dropout rate of 50%. All convolutions are 4 × 4 spatial filters applied with stride 2. Convolutions in the encoder, and in the discriminator, downsample by a factor of 2, whereas in the decoder they upsample by a factor of 2.

令Ck表示conv-bn-relu层，有k个滤波器。CDk表示conv-bn-dropout-relu层，dropout率为50%。所有卷积都是4x4的空间滤波器大小，步长为2。编码器和判别器中的卷积，下采样系数为2，解码器中上采样系数为2。

#### 6.1.1 Generator architectures

The encoder-decoder architecture consists of:

encoder: C64-C128-C256-C512-C512-C512-C512-C512

decoder: CD512-CD512-CD512-C512-C256-C128-C64

After the last layer in the decoder, a convolution is applied to map to the number of output channels (3 in general, except in colorization, where it is 2), followed by a Tanh function. As an exception to the above notation, BatchNorm is not applied to the first C64 layer in the encoder. All ReLUs in the encoder are leaky, with slope 0.2, while ReLUs in the decoder are not leaky.

The U-Net architecture is identical except with skip connections between each layer i in the encoder and layer n − i in the decoder, where n is the total number of layers. The skip connections concatenate activations from layer i to layer n − i. This changes the number of channels in the decoder:

U-Net decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128

#### 6.1.2 Discriminator architectures

The 70 × 70 discriminator architecture is: C64-C128-C256-C512

After the last layer, a convolution is applied to map to a 1-dimensional output, followed by a Sigmoid function. As an exception to the above notation, BatchNorm is not applied to the first C64 layer. All ReLUs are leaky, with slope 0.2.
All other discriminators follow the same basic architecture, with depth varied to modify the receptive field size:

1 × 1 discriminator: C64-C128 (note, in this special case, all convolutions are 1 × 1 spatial filters)

16 × 16 discriminator: C64-C128

286 × 286 discriminator: C64-C128-C256-C512-C512-C512

### 6.2. Training details