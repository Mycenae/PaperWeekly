# Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)

Yi Zhang et al. Sichuan University

## 0. Abstract

Given the potential risk of X-ray radiation to the patient, low-dose CT has attracted a considerable interest in the medical imaging field. Currently, the main stream low-dose CT methods include vendor-specific sinogram domain filtration and iterative reconstruction algorithms, but they need to access raw data whose formats are not transparent to most users. Due to the difficulty of modeling the statistical characteristics in the image domain, the existing methods for directly processing reconstructed images cannot eliminate image noise very well while keeping structural details. Inspired by the idea of deep learning, here we combine the autoencoder, deconvolution network, and shortcut connections into the residual encoder-decoder convolutional neural network (RED-CNN) for low-dose CT imaging. After patch-based training, the proposed RED-CNN achieves a competitive performance relative to the-state-of-art methods in both simulated and clinical cases. Especially, our method has been favorably evaluated in terms of noise suppression, structural preservation, and lesion detection.

X射线辐射对病人有潜在的风险，所以低剂量CT吸引了医学成像领域的很多兴趣。目前，主流低剂量CT方法包括，与供应商相关的sinogram域滤波和迭代重建算法，但他们需要访问原始数据，其格式对于多数用户并不是透明的。由于在图像域中对统计特性建模很困难，现有的直接处理重建图像的方法不能早保持结构细节的同时，很好的消除图像噪声。受到深度学习思想的启发，这里我们将自动编码器、解卷积网络和捷径连接结合到一起，形成残差编码器解码器网络，进行低剂量CT成像。在基于图像块的训练后，提出的RED-CNN域目前最好的方法相比，在仿真和临床情况下，都取得了可以类比的结果。特别是，我们的方法在噪声抑制、结构保持和损伤检测上，都取得了很好的评估结果。

**Index Terms** — Low-dose CT, deep learning, auto-encoder, convolutional, deconvolutional, residual neural network.

## 1. Introduction

X-ray computed tomography (CT) has been widely utilized in clinical, industrial and other applications. Due to the increasing use of medical CT, concerns have been expressed on the overall radiation dose to a patient. The research interest has been strong in CT dose reduction under the well-known guiding principle of ALARA (as low as reasonably achievable) [1]. The most common way to lower the radiation dose is to reduce the X-ray flux by decreasing the operating current and shortening the exposure time of an X-ray tube. In general, the weaker the X-ray flux, the noisier a reconstructed CT image, which degrades the signal-to-noise ratio and could compromise the diagnostic performance. To address this inherent physical problem, many algorithms were designed to improve the image quality for low-dose CT (LDCT). These algorithms can be generally categorized into three categories: (1) sinogram domain filtration, (2) iterative reconstruction, and (3) image processing.

X射线CT在临床、工业等领域有广泛的应用。由于医学CT的使用越来越多，对病人的辐射剂量得到了越来越多的关注。在CT剂量降低上的研究兴趣一直很强，其有名的指导原则是ALARA(as low as reasonably achievable)。降低放射剂量的最常见方式是，通过减少X射线球管的管电流和缩短曝光时间来降低X射线通量。一般来说，X射线通量越弱，重建的CT图像包含的噪声越多，将降低来SNR，使得诊断性能折中。为解决这种内在的物理问题，设计了很多算法来改进低剂量CT的图像质量。这些算法一般可以分为三类：(1)sinogram域滤波，(2)迭代重建，(3)图像处理。

Sinogram filtering techniques perform on either raw data or log-transformed data before image reconstruction, such as filtered backprojection (FBP). The main convenience in the data domain is that the noise characteristic has been well known. Typical methods include structural adaptive filtering [2], bilateral filtering [3], and penalized weighted least-squares (PWLS) algorithms [4]. However, the sinogram filtering methods often suffer from spatial resolution loss when edges in the sinogram domain are not well preserved.

Sinogram滤波技术是在原始数据或log变换的数据上运算，然后再进行图像重建，比如滤波反投影(FBP)。在数据域的主要便利是，噪声特性是知道的很清楚的。典型的方法包括结构自适应滤波[2]，横向滤波[3]和惩罚加权最小二乘(PWLS)算法[4]。但是，sinogram滤波方法通常有空间分辨率降低的问题，因为sinogram域的边缘没有保存的很好。

Over the past decade, iterative reconstruction (IR) algorithms have attracted much attention especially in the field of LDCT. This approach combines the statistical properties of data in the sinogram domain, prior information in the image domain, and even parameters of the imaging system into one unified objective function. With compressive sensing (CS) [5], several image priors were formulated as sparse transforms to deal with the low-dose, few-view, limited-angle and interior CT issues, such as total variation (TV) and its variants [6-9], nonlocal means (NLM) [10-12], dictionary learning [13], low-rank [14], and other techniques. Model based iterative reconstruction (MBIR) takes into account the physical acquisition processes and has been implemented on some current CT scanners [15]. Although IR methods obtained exciting results, there are two weaknesses. First, on most of modern MDCT scanners, IR techniques have replaced FBP based image reconstruction techniques for radiation dose reduction. However, these IR techniques are vendor-specific since the details of the scanner geometry and correction steps are not available to users and other vendors. Second, there are substantial computational overhead costs associated with popular IR techniques. Fully model-based iterative reconstruction techniques have greater potential for radiation dose reduction but slow reconstruction speed and changes in image appearance limit their clinical applications.

在过去的十年中，迭代重建(IR)算法已经吸引了很多注意力，尤其是在LDCT的领域。这种方法数据将sinogram域的统计性质，图像域的先验信息，甚至是成像系统中的参数结合到一个统一的目标函数中。采用压缩感知技术[5]，几种图像先验表示成了稀疏变换，以处理低剂量、少视野、有限角度和CT的内在问题，如全变分(TV)及其变体[6-9]，非局部均值(NLM)[10-12]，字典学习[13]，低秩[14]和其他技术。基于模型的迭代重建(MBIR)将物理获取过程纳入考虑，在几个目前的CT上得到了实现[15]。虽然IR方法得到了令人激动的结果，但有两个弱点。第一，在多数现代MDCT中，IR技术已经替换掉了基于FBP的图像重建技术，以降低辐射剂量。但是，这些IR技术是每个供应商特定的，因为扫描几何的细节和修正步骤对于用户和其他供应商是不可用的。第二，流行的IR方法有很大的计算开销。完全基于模型的迭代重建技术，降低辐射剂量的潜力更大，但重建速度较低，而且图像外貌变化较大，这限制了其临床应用。

An alternative for LDCT is post-processing of reconstructed images, which does not rely on raw data. These techniques can be directly applied on LDCT images, and integrated into any CT system. In [16], NLM was introduced to take advantage of the feature similarity within a large neighborhood in a reconstructed image. Inspired by the theory of sparse representation, dictionary learning [17] was adapted for LDCT denoising, and resulted in substantially improved quality abdomen images [18]. Meanwhile, block-matching 3D (BM3D) was proved efficient for various X-ray imaging tasks [19-21]. In contrast to the other two kinds of methods, the noise distribution in the image domain cannot be accurately determined, which prevents users from achieving the optimal tradeoff between structure preservation and noise supersession.

另一种LDCT的方法，是对重建图像的后处理，这并不依赖于原始数据。这种技术可以直接应用到LDCT图像上，集成进任何CT系统中。在[16]中，引入了NLM，以在重建图像中的大型邻域中利用特征相似度。受稀疏表示理论的启发，[17]采用了字典学习进行LDCT去噪，得到了改进的高质量的腹部图像[18]。同时，块匹配3D(BM3D)证明了其对各种X射线成像任务中都很高效[19-21]。与其他两种方法相比，图像域中的噪声分布不能准确的确定，这妨碍了用户在结构保持和噪声抑制中得到最佳折中。

Recently, deep learning (DL) has generated an overwhelming enthusiasm in several imaging applications, ranging from low-level to high-level tasks from image denoising, deblurring and super resolution to segmentation, detection and recognition [22]. It simulates the information processing procedure by human, and can efficiently learn high-level features from pixel data through a hierarchical network framework [23].

最近，深度学习在几种成像应用中激发了极大的热情，从底层任务到高层任务，包括图像去噪，去模糊和超分辨率，到分割，检测和识别。它模拟了人类信息处理的过程，可以从像素数据中采用层次式的网络架构中高效的学习高层特征。

Several DL algorithms have been proposed for image restoration using different network models [24-31]. As the autoencoder (AE) has a great potential for image denoising, stacked sparse denoising autoencoder (SSDA) and its variant were introduced [24-26]. Convolutional neural networks are powerful tools for feature extraction and were applied for image denoising, deblurring and super resolution [27-29]. Burger et al. [30] analyzed the performance of multi-layer perception (MLP) as applied to image patches and obtained competitive results as compared to the state-of-the-art methods. Previous studies also applied DL for medical image analysis, such as tissue segmentation [32, 33], organ classification [34] and nuclei detection[35]. Furturemore, reports started emerging on tomographic imaging topics. For example, Wang et al. incorporated a DL-based regularization term into a fast MRI reconstruction framework [36]. Chen et al. presented preliminary results with a light-weight CNN-based framework for LDCT imaging [37]. A deeper version using the wavelet transform as inputs was presented [38] which won the second place in the “2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge.” The filtered back-projection (FBP) workflow was mapped to a deep CNN architecture, reducing the reconstruction error by a factor of two in the case of limited-angle tomography [39]. An overall perspective was also published on deep learning, or machine learning in general, for tomographic reconstruction [40].

已经提出了几种DL算法，用不同的网络模型来进行图像恢复[24-31]。由于自编码器(AE)在图像去噪中的潜力很大，所以提出了叠加稀疏去噪自编码器(SSDA)和其变体[24-26]。CNN是特征提取的强力工具，可以用于图像去噪、去模糊和超分辨率[27-29]。Burger等[30]分析了MLP应用到图像块中的性能，与目前最好的方法相比，得到了很有竞争力的结果。之前的研究也将DL用于医学图像分析中，如组织分割[32,33]，器官分类[34]和细胞核检测[35]。而且，在断层成像上也出现了一些文章。比如，Wang等将基于DL的正则化项与快速MRI重建框架[36]结合了起来。Chen等给出了使用轻量级的基于CNN的LDCT成像的框架[37]。[38]提出了使用小波变换作为输入的更深的版本，在“2016 NIH-AAPM-Mayo Clinic低剂量CT挑战赛”中获得了第二名的成绩。[39]将滤波反投影(FBP)的工作流映射到了一个深度CNN架构中，在有限角度断层成像中，降低了一半的重建误差。[40]给出了深度学习、机器学习在断层重建中的总体情况。

Despite the interesting results on CNN for LDCT, the potential of the deep CNN has not been fully realized. Although some studies involved construction of deeper networks [41, 42], most image denoising models had limited layers (usually 2~3 layers) since image denoising is considered as a “low-level” task without intention to extract features. This is in clear contrast to high-level tasks such as recognition or detection, in which pooling and other operations are widely used to bypass image details and capture topological structures.

虽然LDCT中CNN有很有趣的结果，深度CNN的潜力尚未得到完全实现。虽然一些研究涉及到了更深的网络的构建[41,42]，但大多数图像去噪模型层数有限（通常2-3层），因为图像去噪认为是一种低层次任务，不需要进行特征提取。这与高层任务，如识别或检测，形成了鲜明对比，那些任务中广泛使用了pooling和其他运算，以绕过图像的细节，并捕获拓扑结构。

Inspired by the work of [31], we incorporated a deconvolution network [43] and shortcut connections [41, 42] into a CNN model, which is referred to as a residual encoder-decoder convolutional neural network (RED-CNN). In the second section, the proposed network architecture is described. In the third section, the proposed model is evaluated and validated. In the final section, the conclusion is drawn.

受[31]的启发，我们将解卷积网络[43]和捷径连接[41,42]与CNN模型结合起来，称之为残差编码器-解码器卷积神经网络(RED-CNN)。在第二部分，描述了提出的网络架构。在第三部分，评估和验证了提出的模型。在最后的部分，给出了结论。

## 2. Methods

### 2.1 Noise Reduction Model

Our workflow starts with a straightforward FBP reconstruction from a low-dose scan, and the image denoising problem is restricted within the image domain [37]. Since the DL-based methods are independent of the statistical distribution of image noise, the LDCT problem can be simplified to the following one. Assuming that $X∈R^{m×n}$ is a LDCT image and $Y∈R^{m×n}$ is a corresponding normal dose CT (NDCT) image, the relationship between them can be formulated as

我们的工作流开始于，从低剂量扫描的直接FBP重建，图像去噪问题局限在图像域[37]。由于基于DL的方法与图像噪声的统计分布无关，LDCT问题可以简化为下面的问题。假设$X∈R^{m×n}$是一个LDCT图像，$Y∈R^{m×n}$是对应的正常剂量CT(NDCT)图像，两者之间的关系可以表述为

$$X = σ(Y)$$(1)

where $σ :R^{m×n} -> R^{m×n}$ denotes the complex degradation process involving quantum noise and other factors. Then, the problem can be transformed to seek a function f: 其中σ表示复杂的降质过程，涉及到量子噪声和其他因素。然后，问题可以转换为，寻找一个函数f：

$$argmin_f ||f(X)-Y||_2^2$$(2)

where f is regarded as the optimal approximation of $σ^{-1}$, and can be estimated using DL techniques. 其中f被认为是$σ^{-1}$的最佳估计，可以使用DL技术进行估计。

### 2.2 Residual Autoencoder Network

The autoencoder (AE) was originally developed for unsupervised feature learning from noisy inputs, which is also suitable for image restoration. In the context of image denoising, CNN also demonstrated an excellent performance. However, due to its multiple down-sampling operations, some image details can be missed by CNN. For LDCT, here we propose a residual network combining AE and CNN, which has an origin in the work [31]. Rather than adopting fully-connected layers for encoding and decoding, we use both convolutional and deconvolutional layers in symmetry. Furthermore, different from the typical encoder-decoder structure, residual learning [41] with shortcuts is included to facilitate the operations of the convolutional and corresponding deconvolutional layers. There are two modifications to the network described in [31]: (a) the ReLU layers before summation with residuals have been removed to abandon the positivity constraint on learned residuals; and (b) shortcuts have been added to improve the learning process.

自动编码器(AE)最早提出来是用于从含噪声的输入中进行无监督特征学习，这也是适用于图像恢复的。在图像去噪的上下文中，CNN也表现出了很好的性能。但是，由于其有多个下采样运算，CNN会失去一些图像细节。对于LDCT，这里我们提出一个残差网络与AE和CNN的结合，[31]最早提出了这个结构。我们没有用全连接层进行编码解码，而是使用了卷积层和解卷积层。而且，与典型的编码器-解码器结构不同，采用了带有捷径连接的残差学习，以促进卷积层和对应的解卷积层的运算。对[31]中提出的网络有两个改进：(a)去除了与残差求和之前的ReLU层，以放弃学习到的残差中的正值约束；(b)增加了捷径连接，以改进学习过程。

The overall architecture of the proposed RED-CNN network is shown in Fig. 1. This network consists of 10 layers, including 5 convolutional and 5 deconvolutional layers symmetrically arranged. Shortcuts connect matching convolutional and deconvolutional layers. Each layer is followed by its rectified linear units (ReLU) [44]. The details about the network are described as follows.

提出的RED-CNN网络的总体结构如图1所示。网络包含10层，包含5个卷积层和5个解卷积层，对称分布。捷径连接将卷积层与解卷积层匹配起来。每一层都跟着ReLU层。网络细节如下所述。

1) Patch extraction

DL-based methods need a huge number of samples. This requirement cannot be easily met in practice, especially for clinical imaging. In this study, we propose to use overlapped patches in CT images. This strategy has been found to be effective and efficient, because the perceptual differences of local regions can be detected, and the number of samples are significantly boosted [24, 27, 28]. In our experiments, we extracted patches from LDCT and corresponding NDCT images with a fixed size.

基于DL的方法需要大量样本。这种需求在实际中不能轻易得到满足，尤其是对于临床成像来说。在本研究中，我们提出使用CT图像中的重叠块。这种策略已经证明是有效和高效的，因为局部区域的感知差异可以检测的到，而且样本数量可以得到极大的提升。在我们的试验中，我们从LDCT和对应的NDCT中提取出固定大小的图像块。

2) Stacked encoders (Noise and artifact reduction)

Unlike the traditional stacked AE networks, we use a chain of fully-connected convolutional layers as the stacked encoders. Image noise and artifacts are suppressed from low-level to high-level step by step in order to preserve essential information in the extracted patches. Moreover, since the pooling layer (down-sampling) after a convolutional layer may discard important structural details, it is abandoned in our encoder. As a result, there are only two types of layers in our encoder: convolutional layers and ReLU units, and the stacked encoders $C_e^i (x_i)$ can be formulated as

与传统的堆叠AE网络不同，我们使用一系列全连接的卷积层作为堆叠编码器。图像噪声和伪影，从底层到高层，一步一步得到了抑制，在提取出的图像块中保存了必要的信息。而且，由于卷积层后的pooling层（下采样）可能会丢失重要的结构细节，在我们的编码器中就放弃了。结果是，我们的编码器中只有两种类型的层：卷积层和ReLU单元，堆叠编码器$C_e^i (x_i)$可以表述为：

$$C_e^i (x_i) = ReLU(W_i*x_i+b_i), i=0,1,...,N$$(3)

where N is the number of convolutional layers, $W_i$ and $b_i$ denote the weights and biases respectively, * represents the convolution operator, $x_0$ is the extracted patch from the input images, and $x_i(i>0)$ is the extracted features from the previous layers. ReLU(x) = max(0, x) is the activation function. After the stacked encoders, the image patches are transformed into a feature space, and the output is a feature vector $x_N$ whose size is $l_N$.

其中N是卷积层的数量，$W_i$和$b_i$分别表示权重和偏置，*表示卷积运算，$x_0$是从输入图像提出的块，$x_i(i>0)$是从之前的层中提取出的特征。ReLU(x) = max(0, x)是激活函数。在堆叠编码器之后，图像块变换到一个特征空间，输出是一个特征向量$x_N$，其大小为$l_N$。

3) Stacked decoders (Structural detail recovery)

Although the pooling operation is removed, a serial of convolutions, which essentially act as noise filters, will still diminish the details of input signals. Inspired by the recent results on semantic segmentation [45, 46, 47] and biomedical image segmentation [48, 49], deconvolutional layers are integrated into our model for recovery of structural details, which can be seen as image reconstruction from extracted features. We use a chain of fully-connected deconvolutional layers to form the stacked decoders for image reconstruction. Since the encoders and decoders should appear in pair, the convolutional and deconvolutional layers are symmetric in the proposed network. To ensure the input and output of the network match exactly, the convolutional and deconvolutional layers must have the same kernel size. Note that the data flow through the convolutional and deconvolutional layers in our framework follows the rule of “FILO” (First In Last Out). As demonstrated in Fig. 1, the first convolution layer corresponds to the last deconvolutional layer, the last convolution layer corresponds to the first deconvolutional layer, and so on. In other words, this architecture is featured by the symmetry of paired convolution and deconvolution layers.

虽然去掉了pooling算子，一系列卷积仍然会使输入信号的细节逐渐消失，这些卷积实际上起到的是噪声滤波的作用。受到最近在语义分割和医学图像分割中研究的启发，解卷积层整合到了我们的模型中，以恢复结构细节，这可以视作从提取出的特征中进行图像重建。我们使用一系列全连接的解卷积层来作为堆叠解码器，以进行图像重建。由于编码器和解码器是成对出现的，所以在我们提出的工作中，卷积层和解卷积层是成对出现的。为确保网络的输入和输出严格匹配，卷积层和解卷积层必须有相同的核大小。注意，数据在卷积层和解卷积层中的流动，符合先进后出(FILO)的原则。如图1所示，第一个卷积层对应着最后一个解卷积层，最后一个卷积层对应着第一个解卷积层，等等。换句话说，这种架构的特征是，卷积层和解卷积层是成对对称的。

There are two types of layers in our decoder network: deconvolution and ReLU. Thus, the stacked decoders $D_d^i (y_i)$ can be formulated as: 在解码器网络中有两种类型的层：解卷积和ReLU。因此，堆叠解码器$D_d^i (y_i)$可以表述为：

$$D_d^i (y_i) = ReLU(W'_i \bigotimes y_i+b'_i), i=0,1...,N$$(4)

where N is the number of deconvolutional layers, $W'_i$ and $b'_i$ denote the weights and biases respectively, $\bigotimes$ represents the deconvolutional operator, $y_N = x$ is the output feature vector after stacked encoding, $y_i$ (N>i>0) is the reconstructed feature vector from the previous deconvolutional layer, and $y_0$ is the reconstructed patch. After stacked decoding, image patches are reconstructed from features, and can be assembled to reconstruct a denoised image.

其中N是解卷积层的数量，$W'_i$和$b'_i$分别表示权重和偏置，$\bigotimes$表示解卷积算子，$y_N = x$是堆叠编码后的输出特征向量，$y_i$ (N>i>0)是从之前的解卷积层中重建的特征向量，$y_0$是重建得到的图像块。在堆叠解码后，图像块从特征中重建得到，可以组装起来重建一幅去噪的图像。

4) Residual compensation

Like the prior art methods [24, 25], convolution will eliminate some image details. Although the deconvolutional layers can recover some of the details, when the network goes deeper this inverse problem becomes more ill-posed, and the accumulated loss could be quite unsatisfactory for image reconstruction. In addition, when the network depth increases the gradient diffusion could make the network difficult to train.

与之前最好的方法[24,25]类似，卷积会使一些图像细节丢失。虽然解卷积层可以恢复一些细节，但当网络更深时，这种逆问题变得更病态，累加的损失对于图像重建来说，不能很令人满意。另外，当网络深度增加时，梯度弥散的问题使得网络更难以训练。

To address the above two issues, similar to deep residual learning [41, 42] we introduce a residual compensation mechanism into the proposed network. Instead of mapping the input to the output solely by the stacked layers, we adopt a residual mapping, as shown in Fig. 2. Defining the input as I and the output as O, the residual mapping can be denoted as F(I) = O-I, and we use stacked layers to fit this mapping. Once the residual mapping is built, we can reconstruct the original mapping as R(I)=O=F(I)+I. Consequently, we transform the direct mapping problem to a residual mapping problem.

为解决上述两个问题，与深度残差学习类似，我们在提出的网络中加入了一种残差补偿机制。除了用堆叠的层将输入映射到输出，我们采用了一种残差映射，如图2所示。将输入定义为I，输出定义为O，残差映射可以表示为F(I) = O-I，我们使用堆叠的层来适配这种映射。一旦建立了残差映射，我们可以将原始映射重建为R(I)=O=F(I)+I。结果是，我们将直接映射问题，变换成了一种残差映射问题。

There are two benefits associated with the residual mapping. First, it is easier to optimize the residual mapping than optimizing the direct mapping. In other words, it helps avoid the gradient vanishing during training when the network is deep. For example, it would be much easier to train an identity mapping network by pushing the residual to zero than fitting an identity mapping directly. Second, since only the residual is processed by the convolutional and deconvolutional layers, more structural and contrast details can be preserved in the output of the deconvolutional layers, which can significantly enhance the LDCT imaging performance.

有两种与残差映射相关的好处。第一，与优化直接映射相比，很容易优化残差映射。换句话说，当网络很深时，其有助于避免训练时的梯度消失问题。比如，要训练一个恒等映射的话，与直接拟合一个恒等映射相比，拟合接近于0的残差显然更容易。第二，由于卷积和解卷积层只处理了残差，更多的结构细节和对比度细节可以在解卷积层中的输出得到保留，这可以极大的增强LDCT成像的性能。

The use of shortcut connections in [41, 42] was to solve the difficulty in training so that the shortcut connections were only applied across convolutional layers of the same size. In our work, shortcut connections were used for both preservation of structural details and facilitation of training deeper networks. Furthermore, the symmetric structure of convolution and deconvolution layer pairs was also utilized to keep more details while suppressing image noise and artifacts. The CNN layers in [41] are essentially feedforward long short-term memories (LSTMs) without gates, while our RED-CNN network is in general not composed of the standard feedforward LSTMs.

[41,42]中使用捷径连接，解决了训练中的困难，所以捷径连接只在相同大小的卷积层中进行了应用。在我们的工作中，捷径连接用于保留结构细节，并促进深度网络的训练。而且，卷积和解卷积层对的对称结构，也被用于在抑制图像噪声和伪影的同时，保持更多细节。[41]中的CNN层实际上是没有gates的前向LSTM，而我们的RED-CNN网络中一般不含有标准的前向LSTMs。

In [47] and its variants [48, 49], both shortcut connection and deconvolution were used for segmentation. High resolution features were combined with an up-sampled output to improve the image classification. Besides shortcut connection and deconvolution, there are the following new features of the proposed RED-CNN over the networks in [47-49]:

在[47]及其变体[48,49]中，捷径连接和解卷积层都用于分割。高分辨率的特征与上采样的输出相结合，以改进图像分类的结果。除了捷径连接和解卷积，我们提出的RED-CNN与[47-49]中的网络相比，有以下新的特征：

(i). The idea of the autoencoder, which was originally designed for training with noisy samples, was introduced into our model, and convolution and deconvolution layers appeared in pairs; 自动编码器的思想引入了我们的模型，其原始设计是用于训练含噪的样本，卷积层和解卷积层成对出现；

(ii). To avoid losing details, pooling layer was discarded; 为避免损失细节，放弃了pooling层；

(iii). Convolution layers can be seen as noise filters in our application, but filtering leads to loss in details. Deconvolution and shortcutting in our model were used for detail preservation, and in the experiment section we will separately analyze the improvements due to each of these components. Furthermore, the strides of convolution and deconvolution layers in our model were fixed to 1 to avoid down-sampling. 卷积层在我们的应用中可以视作噪声滤波器，但滤波会导致细节的损失。我们模型中的解卷积和捷径用于保存细节，在试验部分中，会单独分析由于各个部件带来的改进。另外，在我们的模型中卷积和解卷积的步长固定为1，以避免下采样。

5) Training

The proposed network is an end-to-end mapping from low-dose CT images to normal-dose images. Once the network is configured, the set of parameters, $Θ ={W_i, b_i, W'_i, b'_i}$ of the convolutional and deconvolutional layers should be estimated to build the mapping function M. The estimation can be achieved by minimizing the loss $F(D;Θ)$ between the estimated CT images and the reference NDCT images X. Given a set of paired patches P={$(X_1, Y_1), (X_2, Y_2), ..., (X_K, Y_K)$} where {$X_i$} and {$Y_i$} denote NDCT and LDCT image patches respectively, and K is the total number of training samples. The mean squared error (MSE) is utilized as the loss function:

提出的网络是一种从低剂量CT图像到正常剂量的图像的端到端映射。一旦网络配置好，卷积层和解卷积层的参数集$Θ ={W_i, b_i, W'_i, b'_i}$，应当估计得到，以构建映射函数M。通过最小化估计的CT图像和参考NDCT图像X之间的损失函数$F(D;Θ)$，就可以估计得到参数。给定成对的图像块的集合P={$(X_1, Y_1), (X_2, Y_2), ..., (X_K, Y_K)$}，其中{$X_i$}和{$Y_i$}分别表示NDCT和LDCT图像块，K是训练样本的总数量。MSE用作损失函数：

$$F(D;Θ) = \frac{1}{N} \sum_{i=1}^N ||X_i-M(Y_i)||^2$$(5)

In this study, the loss function was optimized by Adam [50]. 本研究中，损失函数用Adam进行优化。

## 3. Experimental Design and Results

### 3.1 Data Sources

1) Simulated data

正常剂量的CT来自于下载数据集，然后用仿真的方法得到低剂量CT，仿真的方法是增加Posson分布的噪声；

2) Clinical data

临床数据集来自于Mayo诊所举办的挑战赛。

### 3.2 Parameter selection

图像块大小为55*55。

### 3.3 Experimental Results

1) Simulated data

RED-CNN最好。

2) Clinical data

RED-CNN最好。

### 3.4 Model and Performance Trade-Offs

1) Deconvolutional Decoder

2) Shortcut Connection

3) Number of Layers

4) Patch Size

5) Performance Robustness

6) Computational Cost

## 4. Conclusion

In brief, we have designed a symmetrical convolutional and deconvolutional neural network, aided by shortcut connections. Two well-known databases have been utilized to evaluate and validate the performance of our proposed RED-CNN in comparison with the state of the art methods. The simulated and clinical results have demonstrated a great potential of deep learning for noise suppression, structural preservation, and lesion detection at a high computational speed. In the future, we plan to optimize RED-CNN, extend it to higher dimensional cases such as 3D reconstruction, dynamic/spectral CT reconstruction, and adapt the ideas to other imaging tasks or even other imaging modalities.

简而言之，我们设计了一种对称的卷积和解卷积网络，还含有捷径连接。利用了两个有名的数据集，来评估和验证我们提出的RED-CNN的性能，与目前最好的方法进行了对比。仿真的和临床的结果都证明，深度学习进行去噪、结构保持和损伤检测上有很大的潜力，计算速度还很快。未来，我们计划优化RED-CNN，将其拓展到更高的维度，如3D重建，动态/能谱CT重建，将这种思想调整用于其他成像任务或甚至其他成像模态。