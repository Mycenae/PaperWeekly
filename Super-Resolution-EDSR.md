# Enhanced Deep Residual Networks for Single Image Super-Resolution

Bee Lim et al. Department of ECE, ASRI, Seoul National University

## Abstract

Recent research on super-resolution has progressed with the development of deep convolutional neural networks (DCNN). In particular, residual learning techniques exhibit improved performance. In this paper, we develop an enhanced deep super-resolution network (EDSR) with performance exceeding those of current state-of-the-art SR methods. The significant performance improvement of our model is due to optimization by removing unnecessary modules in conventional residual networks. The performance is further improved by expanding the model size while we stabilize the training procedure. We also propose a new multi-scale deep super-resolution system (MDSR) and training method, which can reconstruct high-resolution images of different upscaling factors in a single model. The proposed methods show superior performance over the state-of-the-art methods on benchmark datasets and prove its excellence by winning the NTIRE2017 Super-Resolution Challenge [26].

近年来超分辨率随着深度卷积神经网络(DCNN)的发展而进步。特别是，残差学习技术显示出了很好的性能。本文中，我们提出一种增强的深度超分辨率网络(EDSR)，性能超过了目前最好的超分辨率方法。我们的模型显著提升了性能，主要是由于我们去除了传统残差网络中的不必要模块。通过扩展模型规模，同时稳定训练过程，可以进一步改进性能。我们还提出了一种新的多尺度深度超分辨率系统(MDSR)及其训练方法，可以在一个模型中重建不同上采样因子的高分辨率图像。我们提出的方法与目前最好的方法相比，在基准测试数据集上显示出了更好的性能，而且赢得了NTIRE2017超分辨率挑战[26]。

## 1. Introduction 引言

Image super-resolution (SR) problem, particularly single image super-resolution (SISR), has gained increasing research attention for decades. SISR aims to reconstruct a high-resolution image $I^{SR}$ from a single low-resolution image $I^{LR}$. Generally, the relationship between $I^{LR}$ and the original high-resolution image $I^{HR}$ can vary depending on the situation. Many studies assume that $I^{LR}$ is a bicubic downsampled version of $I^{HR}$, but other degrading factors such as blur, decimation, or noise can also be considered for practical applications.

图像超分辨率(SR)问题，特别是单幅图像超分辨率(SISR)，几十年来得到了越来越多的研究关注。SISR的目标是从单幅低分辨率图像$I^{LR}$重建高分辨率图像。一般来说，$I^{LR}$与原始高分辨率图像$I^{HR}$的关系会根据情况有所不同。很多研究假设$I^{LR}$是$I^{HR}$的双三次降采样版本，但是其他降质因素如模糊、抽取或噪声也可以在实际应用中考虑到。

Recently, deep neural networks [11, 12, 14] provide significantly improved performance in terms of peak signal-to-noise ratio (PSNR) in the SR problem. However, such networks exhibit limitations in terms of architecture optimality. First, the reconstruction performance of the neural network models is sensitive to minor architectural changes. Also, the same model achieves different levels of performance by different initialization and training techniques. Thus, carefully designed model architecture and sophisticated optimization methods are essential in training the neural networks.

近年来，深度神经网络[11,12,14]在超分问题中的PSNR上得到了显著的改进。但是，这些网络在架构最优性上有局限。首先，神经网络模型的重建效果对微小的架构改变很敏感；同时，同样的模型在不同的初始化和训练技术下会得到不同水平的结果。所以，仔细设计模型架构和复杂的优化方法对于训练神经网络来说至关重要。

Second, most existing SR algorithms treat super-resolution of different scale factors as independent problems without considering and utilizing mutual relationships among different scales in SR. As such, those algorithms require many scale-specific networks that need to be trained independently to deal with various scales. Exceptionally, VDSR [11] can handle super-resolution of several scales jointly in the single network. Training the VDSR model with multiple scales boosts the performance substantially and outperforms scale-specific training, implying the redundancy among scale-specific models. Nonetheless, VDSR style architecture requires bicubic interpolated image as the input, that leads to heavier computation time and memory compared to the architectures with scale-specific upsampling method [5, 22, 14].

第二，多数现有的超分算法将不同尺度因子的超分看作独立的不同问题，而没有考虑到并利用不同尺度超分问题之间的相互关系。这样的话，这些算法需要很多特定尺度的网络，这些网络必须独立训练，以处理不同的尺度。有一个例外，VDSR[11]可以在单个网络中同时处理几个尺度的超分问题。训练多个尺度的VDSR模型极大的提升了性能，超过了那些特定尺度的训练，这说明了特定尺度模型的冗余性。但是，VDSR类的架构需要双三次插值的图像作为输入，与特定尺度的上采样方法[5,22,14]相比，这会导致更多的计算时间和内存需求。

While SRResNet [14] successfully solved those time and memory issue with good performance, it simply employs the ResNet architecture from He et al. [9] without much modification. However, original ResNet was proposed to solve higher-level computer vision problems such as image classification and detection. Therefore, applying ResNet architecture directly to low-level vision problems like super-resolution can be suboptimal.

SRResNet[14]成功的解决了这些时间和存储问题，性能也非常好，但它只是采用了He等[9]的ResNet架构，没有很多修正。但是，原始ResNet的提出是为了解决更高层次的计算机视觉问题，如图像分类和检测。所以，将ResNet架构直接应用在超分这样的底层视觉问题中，这不是最优的。

To solve these problems, based on the SRResNet architecture, we first optimize it by analyzing and removing unnecessary modules to simplify the network architecture. Training a network becomes nontrivial when the model is complex. Thus, we train the network with appropriate loss function and careful model modification upon training. We experimentally show that the modified scheme produces better results.

为解决这些问题，基于SRResNet架构，我们首先分析并去除不需要的模块，以简化网络架构。当模型复杂的时候，训练一个网络不是一件小事。所以，我们用合适的损失函数训练网络，并在训练的基础上仔细的修正模型。我们通过试验表明，修正的方案得到了更好的结果。

Second, we investigate the model training method that transfers knowledge from a model trained at other scales. To utilize scale-independent information during training, we train high-scale models from pre-trained low-scale models. Furthermore, we propose a new multi-scale architecture that shares most of the parameters across different scales. The proposed multi-scale model uses significantly fewer parameters compared with multiple single-scale models but shows comparable performance.

第二，我们研究了模型训练方法，可以从其他尺度上训练好的模型中传递信息。为在训练过程中利用尺度无关的信息，我们从预训练的低尺度模型中训练了高尺度模型。而且，我们提出了一个新的多尺度架构，在不同尺度间共享大多数参数。提出的多尺度模型使用的参数数量，与多个单尺度模型相比明显更少，但性能接近。

We evaluate our models on the standard benchmark datasets and on a newly provided DIV2K dataset. The proposed single- and multi-scale super-resolution networks show the state-of-the-art performances on all datasets in terms of PSNR and SSIM. Our methods ranked first and second, respectively, in the NTIRE 2017 Super-Resolution Challenge [26].

我们在标准基准测试数据集和新给出的DIV2K数据集上评估了我们的模型。提出的单尺度和多尺度超分网络在所有数据集中都展示出了目前最好的性能，衡量标准为PSNR和SSIM。我们的方法在NTIRE2017超分挑战赛中分别排名第一和第二[26]。

## 2. Related Works 相关工作

To solve the super-resolution problem, early approaches use interpolation techniques based on sampling theory [1, 15, 34]. However, those methods exhibit limitations in predicting detailed, realistic textures. Previous studies [25, 23] adopted natural image statistics to the problem to reconstruct better high-resolution images.

为解决超分问题，早期的方法使用基于采样理论的插值技术[1,15,34]。但是，这些方法在预测细节、实际的纹理时表现出了局限性。之前的研究[25,23]采用问题相关的自然图像统计数据来重建更好的高分辨率图像。

Advanced works aim to learn mapping functions between $I^{LR}$ and $I^{HR}$ image pairs. Those learning methods rely on techniques ranging from neighbor embedding [3, 2, 7, 21] to sparse coding [31, 32, 27, 33]. Yang et al. [30] introduced another approach that clusters the patch spaces and learns the corresponding functions. Some approaches utilize image self-similarities to avoid using external databases [8, 6, 29], and increase the size of the limited internal dictionary by geometric transformation of patches [10].

高级一点的工作目标放在了学习$I^{LR}$和$I^{HR}$图像对的映射函数关系上。这些学习方法采用的技术包括邻域嵌入[3,2,7,21]到稀疏编码[31,32,37,33]。Yang等[30]提出了另一种方法，对图像块空间进行聚类，学习对应的函数。一些方法利用图像的自相似性，以避免使用外部数据库[8,6,29]，扩充有限的内部字典的大小的方式则是图像块的几何变换[10]。

Recently, the powerful capability of deep neural networks has led to dramatic improvements in SR. Since Dong et al. [4, 5] first proposed a deep learning-based SR method, various CNN architectures have been studied for SR. Kim et al. [11, 12] first introduced the residual network for training much deeper network architectures and achieved superior performance. In particular, they showed that skip-connection and recursive convolution alleviate the burden of carrying identity information in the super-resolution network. Similarly to [20], Mao et al. [16] tackled the general image restoration problem with encoder-decoder networks and symmetric skip connections. In [16], they argue that those nested skip connections provide fast and improved convergence.

最近，深度神经网络的强大能力使超分得到了极大发展。自从Dong等[4,5]第一次提出了一个基于深度学习的超分方法，各种CNN架构在超分中的应用都得到了研究。Kim等[11,12]首先引入残差网络以训练深的多的网络架构，得到了更好的表现。特别是，他们展示了跳跃连接和递归卷积减缓了超分网络中恒等信息的重担。与[20]类似，Mao等[16]使用编码器-解码器网络和对称跳跃连接来处理一般的图像恢复问题，在[16]中，他们说嵌套的跳跃连接可以提供快速和改进的收敛性。

In many deep learning based super-resolution algorithms, an input image is upsampled via bicubic interpolation before they fed into the network [4, 11, 12]. Rather than using an interpolated image as an input, training up-sampling modules at the very end of the network is also possible as shown in [5, 22, 14]. By doing so, one can reduce much of computations without losing model capacity because the size of features decreases. However, those kinds of approaches have one disadvantage: They cannot deal with the multi-scale problem in a single framework as in VDSR [11]. In this work, we resolve the dilemma of multi-scale training and computational efficiency. We not only exploit the inter-relation of learned feature for each scale but also propose a new multi-scale model that efficiently reconstructs high-resolution images for various scales. Furthermore, we develop an appropriate training method that uses multiple scales for both single- and multi-scale models.

在很多基于深度学习的超分算法中，输入图像是双三次插值得到的上采样图像，然后送入网络[4,11,12]。不使用插值图像作为输入，在网络的最终一端训练上采样模块也是可能的[5,22,14]。这样做可以减少大多数运算量，而不损失模型能力，因为特征的大小减小了。但是，这种方法有一种缺点：他们不能像VDSR[11]那样在单个框架中处理多尺度问题。本文中，我们解决了多尺度训练和计算效率之间的两难困境。我们不仅研究了每个尺度上学习到的特征之间的关系，而且提出了一种新的多尺度模型，可以高效的重建多个尺度上的高分辨率图像。而且，我们提出了一种合适的训练方法，对单尺度模型和多尺度模型都使用了多尺度信息。

Several studies also have focused on the loss functions to better train network models. Mean squared error (MSE) or L2 loss is the most widely used loss function for general image restoration and is also major performance measure (PSNR) for those problems. However, Zhao et al. [35] reported that training with L2 loss does not guarantee better performance compared to other loss functions in terms of PSNR and SSIM. In their experiments, a network trained with L1 achieved improved performance compared with the network trained with L2.

有几个研究聚焦在损失函数上，以训练更好的网络模型。均方误差(MSE)或L2损失是在图像恢复中最广泛使用的损失函数，PSNR是主要的性能度量指标。但是，Zhao等[35]表示，用L2损失训练与其他损失函数相比，在PSNR和SSIM方面，并不一定有更好的表现。在他们的试验中，用L1损失训练的网络与用L2损失训练的网络相比，得到了更好的表现。

## 3. Proposed Methods 提出的方法

In this section, we describe proposed model architectures. We first analyze recently published super-resolution network and suggest an enhanced version of the residual network architecture with the simpler structure. We show that our network outperforms the original ones while exhibiting improved computational efficiency. In the following sections, we suggest a single-scale architecture (EDSR) that handles a specific super-resolution scale and a multi-scale architecture (MDSR) that reconstructs various scales of high-resolution images in a single model.

本节中，我们描述了提出的模型架构。我们首先分析了最近发表的超分网络，并提出了一种简化结构的增强版残差网络架构。我们展示了，我们的网络比原版的性能要好，同时计算效率要高。在下面的小节里，我们提出了单尺度架构(EDSR)，可以处理特定尺度的超分问题，以及多尺度架构(MDSR)，可以在单模型中重建各种尺度的高分辨率图像。

### 3.1. Residual blocks 残差模块

Recently, residual networks [11, 9, 14] exhibit excellent performance in computer vision problems from the low-level to high-level tasks. Although Ledig et al. [14] successfully applied the ResNet architecture to the super-resolution problem with SRResNet, we further improve the performance by employing better ResNet structure.

近年来，残差网络[11,9,14]在底层到高层的计算机视觉问题中都展现出了优异的性能。虽然Ledig等[14]成功的将ResNet架构应用于超分问题，提出了SRResNet，我们则采用更好的ResNet架构以进一步改进性能。

Figure 2: Comparison of residual blocks in original ResNet, SRResNet, and ours. (a) Original (b) SRResNet (c) Proposed

In Fig. 2, we compare the building blocks of each network model from original ResNet [9], SRResNet [14], and our proposed networks. We remove the batch normalization layers from our network as Nah et al.[19] presented in their image deblurring work. Since batch normalization layers normalize the features, they get rid of range flexibility from networks by normalizing the features, it is better to remove them. We experimentally show that this simple modification increases the performance substantially as detailed in Sec. 4.

在图2中，我们比较了每个网络模型的组成模块，包括原始ResNet[9]，SRResNet[14]以及我们提出的网络。就像Nah等[19]在其图像质量改善工作中展示的那样，我们去除了BN层。因为BN层对特征进行了归一化，网络也就没有了范围的灵活性，去掉反而好一些。我们的试验表明，这种简单的修改极大的提升了性能，详见第4节。

Furthermore, GPU memory usage is also sufficiently reduced since the batch normalization layers consume the same amount of memory as the preceding convolutional layers. Our baseline model without batch normalization layer saves approximately 40% of memory usage during training, compared to SRResNet. Consequently, we can build up a larger model that has better performance than conventional ResNet structure under limited computational resources.

而且，GPU内存使用率也得到了极大降低，因为BN层消耗的内存与前面的卷积层是一样的。与SRResNet相比，我们没有BN层的基准模型在训练中节省了大约40%的内存使用。结果是，我们可以构建一个更大的模型，在有限的计算资源下比传统ResNet结构有更好的表现。

### 3.2. Single-scale model 单尺度模型

The simplest way to enhance the performance of the network model is to increase the number of parameters. In the convolutional neural network, model performance can be enhanced by stacking many layers or by increasing the number of filters. General CNN architecture with depth (the number of layers) B and width (the number of feature channels) F occupies roughly O(BF) memory with O($BF^2$) parameters. Therefore, increasing F instead of B can maximize the model capacity when considering limited computational resources.

强化网络模型的表现的最简单方法就是增加参数数量。在卷积神经网络中，模型表现的增强方式可以是堆叠很多层，或增加滤波器数量。一般的CNN架构设其深度（层数）为B，宽度（特征通道数）为F，其内存使用大致是O(BF)，参数数量为O($BF^2$)。所以，在有限的计算资源下，增加F，而不是增加B，可以最大化模型能力。

However, we found that increasing the number of feature maps above a certain level would make the training procedure numerically unstable. A similar phenomenon was reported by Szegedy et al. [24]. We resolve this issue by adopting the residual scaling [24] with factor 0.1. In each residual block, constant scaling layers are placed after the last convolution layers. These modules stabilize the training procedure greatly when using a large number of filters. In the test phase, this layer can be integrated into the previous convolution layer for the computational efficiency.

但是，我们发现增加特征图数量超过了一定水平，会使得训练过程在数值上不稳定。Szegedy等[24]也报告了类似的现象。我们采用了尺度参数为0.1的残差来解决这个问题。在每个残差模块中，在最后的卷积层后面，都加上常数尺度层。在使用大量滤波器的情况下，这些模块极大的稳定了训练过程。在测试阶段，这个层可以与前面的卷积层集成到一起，以提高计算效率。

We construct our baseline (single-scale) model with our proposed residual blocks in Fig. 2. The structure is similar to SRResNet [14], but our model does not have ReLU activation layers outside the residual blocks. Also, our baseline model does not have residual scaling layers because we use only 64 feature maps for each convolution layer. In our final single-scale model (EDSR), we expand the baseline model by setting B = 32, F = 256 with a scaling factor 0.1. The model architecture is displayed in Fig. 3.

我们用图2中提出的残差模块来构建基准（单尺度）模型。模型结构与SRResNet[14]类似，但我们的模型在残差模块之外没有ReLU激活层。而且，我们的基准模型没有残差尺度层，因为我们在每个卷积层中只使用了64个特征图。在我们最终的单尺度模型中(EDSR)，我们将基准模型进行了扩展，设B=32, F=256，尺度系数0.1。模型架构如图3所示。

Figure 3: The architecture of the proposed single-scale SR network (EDSR).

When training our model for upsampling factor ×3 and ×4, we initialize the model parameters with pre-trained ×2 network. This pre-training strategy accelerates the training and improves the final performance as clearly demonstrated in Fig. 4. For upscaling ×4, if we use a pre-trained scale ×2 model (blue line), the training converges much faster than the one started from random initialization (green line).

当训练上采样系数为x3和x4的模型时，我们用预训练的x2网络来初始化模型参数。这个预训练策略加速了训练，并改进了最终性能，图4中清晰的表明了这一点。对于x4的上采样，如果我们使用预训练的x2模型（蓝线），训练收敛要比使用随机初始化（绿线）要快的多。

Figure 4: Effect of using pre-trained ×2 network for ×4 model (EDSR). The red line indicates the best performance of green line. 10 images are used for validation during training.

### 3.3. Multi-scale model 多尺度模型

From the observation in Fig. 4, we conclude that super-resolution at multiple scales is inter-related tasks. We further explore this idea by building a multi-scale architecture that takes the advantage of inter-scale correlation as VDSR [11] does. We design our baseline (multi-scale) models to have a single main branch with B = 16 residual blocks so that most of the parameters are shared across different scales as shown in Fig. 5.

从对图4的观察中，我们得出结论，多个尺度上的超分任务是存在互相相关的任务。我们进一步探索这种思想，构建了一个多尺度架构，利用了这种尺度间相关性，像VDSR[11]一样。我们设计基准（多尺度）模型，单个主分支有B=16个残差模块，多数参数为不同尺度所共享，如图5所示。

In our multi-scale architecture, we introduce scale-specific processing modules to handle the super-resolution at multiple scales. First, pre-processing modules are located at the head of networks to reduce the variance from input images of different scales. Each of pre-processing module consists of two residual blocks with 5 × 5 kernels. By adopting larger kernels for pre-processing modules, we can keep the scale-specific part shallow while the larger receptive field is covered in early stages of networks. At the end of the multi-scale model, scale-specific upsampling modules are located in parallel to handle multi-scale reconstruction. The architecture of the upsampling modules is similar to those of single-scale models described in the previous section.

在我们的多尺度架构中，我们提出了特定尺度的处理模块来应对多个尺度的超分问题。首先，预处理模块位于网络的头部，以减小不同尺度的输入图像的方差。每个预处理模块包括两个残差模块，卷积核大小为5×5。在预处理模块中采用大一些的卷积核，我们可以使特定尺度的部分保持浅层，而同时大一些的感受野在网络早期就存在。在多尺度模型的最后，特定尺度的上采样模块并行分布以处理多尺度重建。上采样模块的架构与单尺度模型中的类似。

We construct our final multi-scale model (MDSR) with B = 80 and F = 64. While our single-scale baseline models for 3 different scales have about 1.5M parameters each, totaling 4.5M, our baseline multi-scale model has only 3.2 million parameters. Nevertheless, the multi-scale model exhibits comparable performance as the single-scale models. Furthermore, our multi-scale model is scalable in terms of depth. Although our final MDSR has approximately 5 times more depth compared to the baseline multi-scale model, only 2.5 times more parameters are required, as the residual blocks are lighter than scale-specific parts. Note that MDSR also shows the comparable performance to the scale-specific EDSRs. The detailed performance comparison of our proposed models is presented in Table 2 and 3.

我们构建最终的多尺度模型(MDSR)，参数为B=80，F=64。我们三个不同尺度的单尺度基准模型每个有1.5M参数，共计4.5M，而基准多尺度模型只有3.2M参数。不过，多尺度模型的表现与单尺度模型类似。而且，我们的多尺度模型在深度上是可扩展的。虽然我们最终的MDSR的深度比基准多尺度模型深度多了5倍，但只多了2.5倍的参数，因为残差模块比特定尺度的部分更轻量。注意MDSR与特定尺度的EDSR的性能接近。我们模型的详细性能比较见表2和表3。

Table 1: Model specifications.

Options | SRResNet[14] (reproduced) | Baseline (Single / Multi) | EDSR | MDSR
--- | --- | --- | --- | ---
Residual blocks | 16 | 16 | 32 | 80
Filters | 64 | 64 | 256 | 64
Parameters | 1.5M | 1.5M / 3.2M | 43M | 8.0M
Residual scaling | - | - | 0.1 | -
Use BN | Yes | No | No | No
Loss function | L2 | L1 | L1 | L1

Table 2: Performance comparison between architectures on the DIV2K validation set (PSNR(dB) / SSIM). Red indicates the best performance and blue indicates the second best. EDSR+ and MDSR+ denote self-ensemble versions of EDSR and MDSR.

Scale | SRResNet(L2 loss) | SRResNet(L1 loss) | Our baseline(Single-scale) | Our baseline(Multi-scale) | EDSR(Ours) | MDSR(Ours) | EDSR+(Ours) | MDSR+(Ours)
--- | --- | --- | --- | --- | --- | --- | --- | ---
×2 | 34.40/0.9662 | 34.44/0.9665 | 34.55/0.9671 | 34.60/0.9673 | 35.03/0.9695 | 34.96/0.9692 | 35.12/0.9699 | 35.05/0.9696
×3 | 30.82/0.9288 | 30.85/0.9292 | 30.90/0.9298 | 30.91/0.9298 | 31.26/0.9340 | 31.25/0.9338 | 31.39/0.9351 | 31.36/0.9346
×4 | 28.92/0.8960 | 28.92/0.8961 | 28.94/0.8963 | 28.95/0.8962 | 29.25/0.9017 | 29.26/0.9016 | 29.38/0.9032 | 29.36/0.9029

Table 3: Public benchmark test results and DIV2K validation results (PSNR(dB) / SSIM). Red indicates the best performance and blue indicates the second best. Note that DIV2K validation results are acquired from published demo codes.

## 4. Experiments 试验

### 4.1. Datasets 数据集

DIV2K dataset [26] is a newly proposed high-quality (2K resolution) image dataset for image restoration tasks. The DIV2K dataset consists of 800 training images, 100 validation images, and 100 test images. As the test dataset ground truth is not released, we report and compare the performances on the validation dataset. We also compare the performance on four standard benchmark datasets: Set5 [2], Set14 [33], B100 [17], and Urban100 [10].

DIV2K数据集[26]是一个新提出的高质量（2K分辨率）图像恢复任务的数据集。DIV2K数据集包括800幅训练图像，100幅验证图像，和100幅测试图像。由于测试数据集的真值尚未发布，我们在验证数据集上报告并比较性能。我们还在标准基准测试数据集上比较了性能：Set5[2], Set14[33], B100[17]和Urban100[10]。

### 4.2. Training Details 训练细节

For training, we use the RGB input patches of size 48×48 from LR image with the corresponding HR patches. We augment the training data with random horizontal flips and 90 rotations. We pre-process all the images by subtracting the mean RGB value of the DIV2K dataset. We train our model with ADAM optimizer [13] by setting $β_1 = 0.9$, $β_2 = 0.999$, and $\epsilon = 10^{−8}$. We set minibatch size as 16. The learning rate is initialized as $10^{−4}$ and halved at every $2 × 10^5$ minibatch updates.

在训练中，我们使用48×48大小的低分辨率RGB输入图像块以及对应的高分辨率图像块。我们用随机水平翻转和90度旋转来扩充训练数据。我们预处理所有图像的方法为减去DIV2K数据集的均值RGB值。我们训练模型使用ADAM优化器[13]，参数$β_1 = 0.9$, $β_2 = 0.999$, 以及$\epsilon = 10^{−8}$。我们设置minibatch size为16。学习率初始化为$10^{−4}$，每$2 × 10^5$ minibatches减半。

For the single-scale models (EDSR), we train the networks as described in Sec. 3.2. The ×2 model is trained from scratch. After the model converges, we use it as a pretrained network for other scales.

对于单尺度模型(EDSR)，我们如3.2节中训练网络。x2模型是从零开始训练的。模型收敛以后，我们将其用于其他尺度的预训练网络。

At each update of training a multi-scale model (MDSR), we construct the minibatch with a randomly selected scale among ×2,×3 and ×4. Only the modules that correspond to the selected scale are enabled and updated. Hence, scale-specific residual blocks and upsampling modules that correspond to different scales other than the selected one are not enabled nor updated.

在训练多尺度模型(MDSR)的每次迭代中，我们用从x2,x3,x4中随机选择的尺度来构建minibatch。只有对应选择的尺度的那个模块被激活并更新。所以，其余尺度的残差模块和上采样模块都没有被激活，也没有更新。

We train our networks using L1 loss instead of L2. Minimizing L2 is generally preferred since it maximizes the PSNR. However, based on a series of experiments we empirically found that L1 loss provides better convergence than L2. The evaluation of this comparison is provided in Sec. 4.4。

我们使用L1损失训练网络，而不是L2损失。一般都会使用最小化L2损失，因为能使PSNR最大化。但是，经过一系列试验的观察，我们发现L1损失比L2损失的收敛性更好。这种比较的评估放在4.4节中。

We implemented the proposed networks with the Torch7 framework and trained them using NVIDIA Titan X GPUs. It takes 8 days and 4 days to train EDSR and MDSR, respectively. The source code is publicly available online.(https://github.com/LimBee/NTIRE2017)

我们用Torch7框架实现了提出的网络，用NVIDIA Titan X GPU训练。训练EDSR和MDSR分别用了8天和4天。源代码已经开源。

### 4.3. Geometric Self-ensemble 

In order to maximize the potential performance of our model, we adopt the self-ensemble strategy similarly to [28]. During the test time, we flip and rotate the input image $I^{LR}$ to generate seven augmented inputs $I^{LR}_{n,i} = T_i (I_n^{LR})$ for each sample, where $T_i$ represents the 8 geometric transformations including identity. With those augmented low-resolution images, we generate corresponding super-resolved images {$I^{SR}_{n,1}, ..., I^{SR}_{n,8}$} using the networks. We then apply inverse transform to those output images to get the original geometry $\tilde I^{SR}_{n,i} = T_i^{-1}(I^{SR}_{n,i})$. Finally, we average the transformed outputs all together to make the self-ensemble result as follows. $I^{SR}_n = \frac{1}{8} \sum_{i=1}^8 \tilde I^{SR}_{n,i}$.

为最大化我们模型的潜在表现，我们采用了与[28]类似的自集成技术。在测试时，我们将输入图像$I^{LR}$翻转及旋转，对每个样本都生成7个扩充的输入$I^{LR}_{n,i} = T_i (I_n^{LR})$，其中$T_i$表示8个几何变换包括恒等变换。用这些扩充的低分辨率图像，我们用网络生成对应的高分辨率图像{$I^{SR}_{n,1}, ..., I^{SR}_{n,8}$}。然后我们对这些输出图像应用逆变换，以得到原始几何$\tilde I^{SR}_{n,i} = T_i^{-1}(I^{SR}_{n,i})$。最后，我们将变换的输出全部进行平均以得到自集成结果。$I^{SR}_n = \frac{1}{8} \sum_{i=1}^8 \tilde I^{SR}_{n,i}$。

This self-ensemble method has an advantage over other ensembles as it does not require additional training of separate models. It is beneficial especially when the model size or training time matters. Although self-ensemble strategy keeps the total number of parameters same, we notice that it gives approximately same performance gain compared to conventional model ensemble method that requires individually trained models. We denote the methods using self-ensemble by adding ’+’ postfix to the method name; i.e. EDSR+/MDSR+. Note that geometric self-ensemble is valid only for symmetric downsampling methods such as bicubic downsampling.

这种自集成方法比其他集成方法有一个优势，因为不需要训练其他模型。当模型规模比较大，或训练时间很长时，尤其有好处。虽然自集成策略保持所有参数数量相同，我们注意到，相对于传统的需要多个训练模型的集成方法，其得到的性能提升是大约一样的。我们将这种自集成方法表示为原方法名称加上+，即EDSR+/MDSR+。注意几何自集成只对对称下采样方法有效，如双三次下采样。

### 4.4. Evaluation on DIV2K Dataset 在DIK2K数据集上的评估

We test our proposed networks on the DIV2K dataset. Starting from the SRResNet, we gradually change various settings to perform ablation tests. We train SRResNet [14]
on our own.(We confirmed our reproduction is correct by getting comparable results in an individual experiment, using the same settings of the paper [14]. In our experiments, however, it became slightly different to match the settings of our baseline model training. See our codes at https://github.com/LimBee/NTIRE2017.)( We used the original paper (https://arxiv.org/abs/1609.04802v3) as a reference.) First, we change the loss function from L2 to L1, and then the network architecture is reformed as described in the previous section and summarized in Table 1.

我们在DIV2K数据集上测试我们的网络。从SRResNet开始，我们逐渐变换不同的设置，以进行分离试验。我们自己训练了SRResNet[14]。（我们确认了我们的复现是正确的，使用文章[14]中的设置，在实验中得到了类似的结果。但在我们的实验中，要匹配我们基准模型训练的设置，会略有不同。见我们的代码。）（我们使用了原论文作为参考。）第一，我们将L2损失函数变为了L1损失，然后网络架构的变化在前节中已经进行了描述，总结在表1中。

We train all those models with $3 × 10^5$ updates in this experiment. Evaluation is conducted on the 10 images of DIV2K validation set, with PSNR and SSIM criteria. For the evaluation, we use full RGB channels and ignore the (6 + scale) pixels from the border.

在实验中，我们训练所有这些模型$3 × 10^5$次迭代。评估在DIV2K验证集上的10幅图像中进行，标准是PSNR和SSIM。评估时，我们使用全体RGB通道，并忽略边界处的(6+scale)个像素。

Table 2 presents the quantitative results. SRResNet trained with L1 gives slightly better results than the original one trained with L2 for all scale factors. Modifications of the network give an even bigger margin of improvements. The last 2 columns of Table 2 show significant performance gains of our final bigger models, EDSR+ and MDSR+ with the geometric self-ensemble technique. Note that our models require much less GPU memory since they do not have batch normalization layers.

表2中给出了定量结果。用L1损失训练的SRResNet，比原始的用L2损失训练的，在所有尺度上都得到了更好的结果。网络经过修正得到的改进甚至更大。表2的最后两列说明，我们最后的更大的模型，即几何自集成技术得到的EDSR+和MDSR+，有了显著的性能提升。注意我们的模型需要少的多的GPU内存，因为没有BN层。

### 4.5. Benchmark Results 基准测试结果

We provide the quantitative evaluation results of our final models (EDSR+, MDSR+) on public benchmark datasets in Table 3. The evaluation of the self-ensemble is also provided in the last two columns. We trained our models using $10^6$ updates with batch size 16. We keep the other settings same as the baseline models. We compare our models with the state-of-the-art methods including A+ [27], SRCNN [4], VDSR [11], and SRResNet [14]. For comparison, we measure PSNR and SSIM on the y channel and ignore the same amount of pixels as scales from the border. We used MATLAB [18] functions for evaluation. Comparative results on DVI2K dataset are also provided. Our models exhibit a significant improvement compared to the other methods. The gaps further increase after performing self-ensemble. We also present the qualitative results in Fig. 6. The proposed models successfully reconstruct the detailed textures and edges in the HR images and exhibit better-looking SR outputs compared with the previous works.

我们在公开基准测试数据集上给出了我们最终模型(EDSR+, MDSR+)的量化评估结果，如表3所示。自集成的评估在表中的最后两列。我们训练我们模型迭代$10^6$次，batch size为16。我们保持其他设置与基准模型一样。我们将我们的模型与目前最好的模型进行了比较，如A+[27], SRCNN[4], VDSR[11]和SRResNet[14]。比较时，我们在图像的y通道度量PSNR和SSIM，忽略边界处同样数量的像素。我们使用Matlab[18]函数进行评估。也给出了在DIV2K数据集上的比较结果。我们的模型在与其他方法进行比较时，显示处了明显的改善。改进的幅度在进行了自集成后得到了进一步改进。我们还在图6中给出了定性的结果。提出的模型在HR图像中成功的重建了纹理细节和边缘，与之前的工作相比，得到了更好看的SR输出结果。

## 5. NTIRE2017 SR Challenge

This work is initially proposed for the purpose of participating in the NTIRE2017 Super-Resolution Challenge [26]. The challenge aims to develop a single image super-resolution system with the highest PSNR.

本文的工作最初是提交参加NTIRE2017超分挑战的[26]。挑战赛目标是得到PSNR最高的单幅图像超分系统。

In the challenge, there exist two tracks for different degraders (bicubic, unknown) with three downsample scales (×2,3,4) each. Input images for the unknown track are not only downscaled but also suffer from severe blurring. Therefore, more robust mechanisms are required to deal with the second track. We submitted our two SR models (EDSR and MDSR) for each competition and prove that our algorithms are very robust to different downsampling conditions. Some results of our algorithms on the unknown downsampling track are illustrated in Fig. 7. Our methods successfully reconstruct high-resolution images from severely degraded input images. Our proposed EDSR+ and MDSR+ won the first and second places, respectively, with outstanding performances as shown in Table 4.

在挑战赛中，在三个降采样尺度(×2,3,4)上分别存在两类不同的降质（双三次，未知）。未知种类降质的输入图像不止是尺度缩小了，而且有很严重的模糊。所以，需要更多的稳健机制来应对第二类图像（即未知种类降质的图像）。我们将我们的两类SR模型(EDSR和MDSR)提交给相应的比赛，证明了我们的算法对不同的降采样情况都非常稳健。我们算法在未知种类的降采样图像中的一些结果如图7所示。我们的方法成功的从严重降质的输入图像中重建了高分辨率图像。我们的EDSR+和MDSR+分别赢得了第一名和第二名，得到的优异结果如表4所示。

Figure 6: Qualitative comparison of our models with other works on ×4 super-resolution.

Figure 7: Our NTIRE2017 Super-Resolution Challenge results on unknown downscaling ×4 category. In the challenge, we excluded images from 0791 to 0800 from training for validation. We did not use geometric self-ensemble for unknown downscaling category.

Table 4: Performance of our methods on the test dataset of NTIRE2017 Super-Resolution Challenge [26]. The results of top 5 methods are displayed for two tracks and six categories. Red indicates the best performance and blue indicates the second best.

Method | track1 x2 PSNR | x2 SSIM | x3 PSNR | x3 SSIM | x4 PSNR | x4SSIM | track2 x2 PSNR | x2 SSIM | x3 PSNR | x3 SSIM | x4 PSNR | x4 SSIM
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
EDSR+ (Ours) | 34.93 | 0.948 | 31.13 | 0.889 | 29.09 | 0.837 | 34.00 | 0.934 | 30.78 | 0.881 | 28.77 | 0.826
MDSR+ (Ours) | 34.83 | 0.947 | 31.04 | 0.888 | 29.04 | 0.836 | 33.86 | 0.932 | 30.67 | 0.879 | 28.62 | 0.821
3rd method | 34.47 | 0.944 | 30.77 | 0.882 | 28.82 | 0.830 | 33.67 | 0.930 | 30.51 | 0.876 | 28.54 | 0.819
4th method | 34.66 | 0.946 | 30.83 | 0.884 | 28.83 | 0.830 | 32.92 | 0.921 | 30.31 | 0.871 | 28.14 | 0.807
5th method | 34.29 | 0.948 | 30.52 | 0.889 | 28.55 | 0.752 | - | - | - | - | - | -

## 6. Conclusion 结论

In this paper, we proposed an enhanced super-resolution algorithm. By removing unnecessary modules from conventional ResNet architecture, we achieve improved results while making our model compact. We also employ residual scaling techniques to stably train large models. Our proposed singe-scale model surpasses current models and achieves the state-of-the-art performance.

本文中，我们提出了一种增强超分算法。我们将传统ResNet架构中的非必要模块移除，使得模型更紧凑，而且得到了改进的结果。我们还采用了不同尺度的残差scaling技术，以稳定的训练大型模型。我们提出的单尺度模型超过了目前算法的最好性能。

Furthermore, we develop a multi-scale super-resolution network to reduce the model size and training time. With scale-dependent modules and shared main network, our multi-scale model can effectively deal with various scales of super-resolution in a unified framework. While the multi-scale model remains compact compared with a set of single-scale models, it shows comparable performance to the single-scale SR model.

而且，我们提出了多尺度超分网络以降低模型规模，减少训练时间。我们的多尺度模型有与尺度相关的模块和共享的主网络，可以在统一的框架中有效的处理各种尺度的超分问题。我们的多尺度模型，与多个单尺度模型相比，规模非常紧凑，而且与单尺度SR模型的结果很接近。

Our proposed single-scale and multi-scale models have achieved the top ranks in both the standard benchmark datasets and the DIV2K dataset. 我们提出的单尺度和多尺度模型在标准基准测试数据集和DIV2K数据集上取得了最高排名。