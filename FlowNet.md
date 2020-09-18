# FlowNet: Learning Optical Flow with Convolutional Networks

Philipp Fischer et. al. Technical University of Munich

## 0. Abstract


Convolutional neural networks (CNNs) have recently been very successful in a variety of computer vision tasks, especially on those linked to recognition. Optical flow estimation has not been among the tasks where CNNs were successful. In this paper we construct appropriate CNNs which are capable of solving the optical flow estimation problem as a supervised learning task. We propose and compare two architectures: a generic architecture and another one including a layer that correlates feature vectors at different image locations.

CNNs最近在很多计算机视觉任务中非常成功，尤其是在与识别有关的任务中。CNNs在光流估计中并不成功。在本文中，我们构建的CNNs以一种监督学习的方式求解光流估计问题。我们提出并比较了两种架构：一个通用架构，另一个架构则包含了一个在不同位置关联特征向量的层。

Since existing ground truth datasets are not sufficiently large to train a CNN, we generate a synthetic Flying Chairs dataset. We show that networks trained on this unrealistic data still generalize very well to existing datasets such as Sintel and KITTI, achieving competitive accuracy at frame rates of 5 to 10 fps.

由于已有的真值数据集并不足够大以训练CNN，我们生成一个合成的Flying Chairs数据集。我们表明，在这种非真实的数据上训练出来的网络，对现有的数据集泛化的仍然非常好，如Sintel和KITTI，以5-10fps的帧率，得到很有竞争力的准确率。

## 1. Introduction

Convolutional neural networks have become the method of choice in many fields of computer vision. They are classically applied to classification [25, 24], but recently presented architectures also allow for per-pixel predictions like semantic segmentation [28] or depth estimation from single images [10]. In this paper, we propose training CNNs end-to-end to learn predicting the optical flow field from a pair of images.

CNNs在计算机视觉的很多领域中都得到了应用。典型的应用是分类，但最近提出的架构还可以逐像素的预测，如语义分割，或从单幅图像中进行深度估计。在本文中，我们提出端到端的训练CNNs，从一对图像中学习预测光流。

While optical flow estimation needs precise per-pixel localization, it also requires finding correspondences between two input images. This involves not only learning image feature representations, but also learning to match them at different locations in the two images. In this respect, optical flow estimation fundamentally differs from previous applications of CNNs.

光流估计需要精确的逐像素的定位，也需要在两幅输入图像中找到对应性。这不仅涉及到学习图像特征表示，而且学习在两幅图像的不同位置上匹配之。有鉴于此，光流估计与之前的CNNs的应用在根本上就有所区别。

Since it was not clear whether this task could be solved with a standard CNN architecture, we additionally developed an architecture with a correlation layer that explicitly provides matching capabilities. This architecture is trained end-to-end. The idea is to exploit the ability of convolutional networks to learn strong features at multiple levels of scale and abstraction and to help it with finding the actual correspondences based on these features. The layers on top of the correlation layer learn how to predict flow from these matches. Surprisingly, helping the network this way is not necessary and even the raw network can learn to predict optical flow with competitive accuracy.

由于这个任务是否可以用标准CNN架构来解决并不清楚，我们提出了一个架构，有一个相关层，显式的给出匹配的能力。这个架构是端到端的训练的。其思想是，探索CNNs在多个尺度层次和抽象层次上学习强特征的能力，基于这些特征找到实际的对应性。在相关层之上的层，学习从这些匹配上预测光流。令人惊讶的是，以这种方式帮助网络并不是必要的，即使原始网络可以学习以不错的准确率预测光流。

Training such a network to predict generic optical flow requires a sufficiently large training set. Although data augmentation does help, the existing optical flow datasets are still too small to train a network on par with state of the art. Getting optical flow ground truth for realistic video material is known to be extremely difficult [7]. Trading in realism for quantity, we generate a synthetic Flying Chairs dataset which consists of random background images from Flickr on which we overlay segmented images of chairs from [1]. These data have little in common with the real world, but we can generate arbitrary amounts of samples with custom properties. CNNs trained on just these data generalize surprisingly well to realistic datasets, even without fine-tuning.

训练这样一个网络来预测通用光流，需要一个足够大的训练集。虽然数据扩增确实有帮助，现有的光流数据集仍然非常小，不能训练出一个目前最好的网络。对于真实的视频资料，得到光流的真值，是非常困难的。以真实性换取数量，我们生成了一个合成的Flying Chairs数据集，由Flickr上的随机背景的图像组成，在这些背景之上，我们叠加了[1]中的椅子的分割图像。这些数据与真实世界的数据很少有共同之处，但我们可以以定制的性质生成任意数量的样本。在这些数据上训练得到的CNNs，在真实数据集上的泛化效果很好，甚至不需要精调。

Leveraging an efficient GPU implementation of CNNs, our method is faster than most competitors. Our networks predict optical flow at up to 10 image pairs per second on the full resolution of the Sintel dataset, achieving state-of-the-art accuracy among real-time methods.

利用CNNs的高效GPU实现，我们的方法比多数竞品都要快。我们的网络预测光流的速度，在Sintel数据集上，以完整分辨率可以达到大约10图像对每秒，在实时方法中，得到目前最好的准确率。

## 2. Related Work

**Optical Flow**. Variational approaches have dominated optical flow estimation since the work of Horn and Schunck [19]. Many improvements have been introduced [29, 5, 34]. The recent focus was on large displacements, and combinatorial matching has been integrated into the variational approach [6, 35]. The work of [35] termed Deep-Matching and DeepFlow is related to our work in that feature information is aggregated from fine to coarse using sparse convolutions and max-pooling. However, it does not perform any learning and all parameters are set manually. The successive work of [30] termed EpicFlow has put even more emphasis on the quality of sparse matching as the matches from [35] are merely interpolated to dense flow fields while respecting image boundaries. We only use a variational approach for optional refinement of the flow field predicted by the convolutional net and do not require any handcrafted methods for aggregation, matching and interpolation.

**光流**。自从Horn和Schunck的工作[19]，变分方法就在光流估计中占了主导。[29,5,34]提出了很多改进。最近关注的是较大的偏移，组合匹配与变分方法结合到了一起[6,35]。[35]的工作称为Deep-Matching和DeepFlow，与我们的工作相关，即特征信息从精细到粗糙聚积到了一起，使用的是稀疏卷积和最大池化。但是，它并没有进行任何学习，所有参数都是手动设置的。后续的[30]工作称为EpicFlow更加注意稀疏匹配的质量，因为[35]中的匹配只是插值到了更加密集的光流场，同时考虑了图像的边缘。我们用卷积网络预测了光流场，然后使用了变分方法进行光流的提炼，这不是必须的，并不需要任何手工的方法进行积聚、匹配和插值。

Several authors have applied machine learning techniques to optical flow before. Sun et al. [32] study statistics of optical flow and learn regularizers using Gaussian scale mixtures; Rosenbaum et al. [31] model local statistics of optical flow with Gaussian mixture models. Black et al. [4] compute principal components of a training set of flow fields. To predict optical flow they then estimate coefficients of a linear combination of these ’basis flows’. Other methods train classifiers to select among different inertial estimates [21] or to obtain occlusion probabilities [27].

之前有几位作者将机器学习技术应用到光流估计中。Sun等[32]研究了光流的统计特性，使用高斯尺度混合学习了正则化器；Rosenbaum等[31]对光流的局部统计特性进行建模，使用的是高斯混合模型。Black等[4]计算了一个光流训练集的主成分。为预测光流，他们估计这些基准流的线性组合的系数。其他方法训练分类器来在不同的惯性估计[21]中进行选择，或得到遮挡的概率[27]。

There has been work on unsupervised learning of disparity or motion between frames of videos using neural network models. These methods typically use multiplicative interactions to model relations between a pair of images. Disparities and optical flow can then be inferred from the latent variables. Taylor et al. [33] approach the task with factored gated restricted Boltzmann machines. Konda and Memisevic [23] use a special autoencoder called ‘synchrony autoencoder’. While these approaches work well in a controlled setup and learn features useful for activity recognition in videos, they are not competitive with classical methods on realistic videos.

使用神经网络模型以及无监督学习，来在视频帧之间学习之间的差异和运动，有一些这样的工作。这些方法一般采用乘法的操作来对一对图像之间的关系进行建模。差异和光流可以从潜在的变量推断出来。Taylor等[33]用分解的门空有限Boltzmann机来尝试解决这个任务。Konda and Memisevic [23]使用一种特殊的autoencoder称为‘synchrony autoencoder’。这些方法在一些受控的设置中效果不错，学习用于视频中行为识别的有用特征，在实际的视频中，他们与经典方法相比，并不具有竞争力。

## 3. Network Architectures

**Convolutional Networks**. Convolutional neural networks trained with backpropagation [25] have recently been shown to perform well on large-scale image classification by Krizhevsky et al. [24]. This gave the beginning to a surge of works on applying CNNs to various computer vision tasks.

**卷积网络**。Krizhevsky等[24]最近证明，采用反向传播训练的卷积神经网络，在大规模图像分类中表现很好。这开启了将CNNs应用到各种计算机视觉任务中的浪潮。

While there has been no work on estimating optical flow with CNNs, there has been research on matching with neural networks. Fischer et al. [12] extract feature representations from CNNs trained in supervised or unsupervised manner and match these features based on Euclidean distance. Zbontar and LeCun [36] train a CNN with a Siamese architecture to predict similarity of image patches. A drastic difference of these methods to our approach is that they are patch based and leave the spatial aggregation to post-processing, whereas the networks in this paper directly predict complete flow fields.

采用CNNs估计光流尚无工作，但有一些采用神经网络进行匹配的工作。Fischer等[12]从有监督或无监督训练的CNNs中提取特征表示，基于欧式距离对这些特征进行匹配。Zbontar and LeCun[36]采用一个Siamese架构来训练CNN，以预测图像块的相似性。这些方法与我们方法的差异是，他们是基于图像块的，将空间聚积的任务留给了后处理，而本文中的网络直接预测了完整的流场。

Recent applications of CNNs include semantic segmentation [11, 15, 17, 28], depth prediction [10], keypoint prediction [17] and edge detection [13]. These tasks are similar to optical flow estimation in that they involve per-pixel predictions. Since our architectures are largely inspired by the recent progress in these per-pixel prediction tasks, we briefly review different approaches.

CNNs最近的应用包括语义分割，深度预测，关键点预测和边缘检测。这些任务与光流估计类似，因为涉及到逐点的预测。由于我们的架构主要是受到最近的逐点预测任务的启发，我们简要的回顾一下不同的方法。

The simplest solution is to apply a conventional CNN in a ‘sliding window’ fashion, hence computing a single prediction (e.g. class label) for each input image patch [8, 11]. This works well in many situations, but has drawbacks: high computational costs (even with optimized implementations involving re-usage of intermediate feature maps) and per-patch nature, disallowing to account for global output properties, for example sharp edges. Another simple approach [17] is to upsample all feature maps to the desired full resolution and stack them together, resulting in a concatenated per-pixel feature vector that can be used to predict the value of interest.

最简单的方法是以一种滑窗的方式应用一个传统的CNN，这样对每个输入的图像块都计算一个预测（如，类别标签）。在很多情况下这都很好用，但有一些缺陷：很高的计算代价（即使进行了实现上的优化，对中间层特征图进行了重复利用），以及每个图像块的本质不能顾及到全局输出的性质，比如尖锐的边缘。另一个简单的方法[17]是，将所有的特征图上采样到期望的完整分辨率，将其叠加到一起，得到拼接在一起的逐像素特征向量，可以用于预测感兴趣的值。

Eigen et al. [10] refine a coarse depth map by training an additional network which gets as inputs the coarse prediction and the input image. Long et al. [28] and Dosovitskiy et al. [9] iteratively refine the coarse feature maps with the use of ‘upconvolutional’ layers. Our approach integrates ideas from both works. Unlike Long et al., we ‘upconvolve’ not just the coarse prediction, but the whole coarse feature maps, allowing to transfer more high-level information to the fine prediction. Unlike Dosovitskiy et al., we concatenate the ‘upconvolution’ results with the features from the ‘contractive’ part of the network.

Eigen等[10]通过训练另一个网络，以粗糙的预测和图像为输入，提炼了一个粗糙的深度图。Long等[28]和Dosovitskiy等[9]使用”上卷积层“，迭代的提炼粗糙的特征图。我们的方法结合了两个工作中的思想。与Long等不一样，我们不仅对粗糙的预测进行上卷积，而且是对整个粗糙的特征图进行上卷积，使得可以将更多的高层信息迁移到精细的预测中。与Dosovitskiy等不一样，我们将上卷积的结果与网络收缩的部分拼接到了一起。

Convolutional neural networks are known to be very good at learning input–output relations given enough labeled data. We therefore take an end-to-end learning approach to predicting optical flow: given a dataset consisting of image pairs and ground truth flows, we train a network to predict the x–y flow fields directly from the images. But what is a good architecture for this purpose?

在有足够的标记数据后，CNNs擅长于学习输入输出关系。因此我们以端到端的输入方法来预测光流：给定数据集，包含很多图像对，以及真值光流，我们训练网络从图像中直接预测x-y流。但怎样的结构才能很好的实现这个目标呢？

A simple choice is to stack both input images together and feed them through a rather generic network, allowing the network to decide itself how to process the image pair to extract the motion information. This is illustrated in Fig. 2 (top). We call this architecture consisting only of convolutional layers ‘FlowNetSimple’.

一个简单的选择是，将输入图像叠加到一起，将其送入一个非常通用的网络，使网络来决定其本身怎样处理图像对，提取出运动信息。这如图2上所示。我们称这种只含有卷积层的架构为FlowNetSimple。

In principle, if this network is large enough, it could learn to predict optical flow. However, we can never be sure that a local gradient optimization like stochastic gradient descent can get the network to this point. Therefore, it could be beneficial to hand-design an architecture which is less generic, but may perform better with the given data and optimization techniques.

原则上，如果网络足够大，就可以学习到怎样预测光流。但是，我们永远不能确定，像SGD这样的局部梯度优化是否能够将网络带到那个点上。因此，如果能够手动设计一个架构，不那么通用的话，可能会非常有帮助，在给定的数据和优化技术中，可能会表现更好。

A straightforward step is to create two separate, yet identical processing streams for the two images and to combine them at a later stage as shown in Fig. 2 (bottom). With this architecture the network is constrained to first produce meaningful representations of the two images separately and then combine them on a higher level. This roughly resembles the standard matching approach when one first extracts features from patches of both images and then compares those feature vectors. However, given feature representations of two images, how would the network find correspondences?

一个直接的步骤是，对图像创建两个分离的，但是相同的处理流，在后续的阶段将其结合到一起，如图2下所示。用这个架构的话，网络就可以首先分别产生两幅图像的有意义的表示，然后可以在较高的层级中将其结合到一起。这于标准的匹配方法大致一样，即首先从两幅图像的图像块中提取特征，然后比较这些特征向量。但是，给定两幅图像的特征表示，网络怎样找到对应性呢？

To aid the network in this matching process, we introduce a ‘correlation layer’ that performs multiplicative patch comparisons between two feature maps. An illustration of the network architecture ‘FlowNetCorr’ containing this layer is shown in Fig. 2 (bottom). Given two multi-channel feature maps $f_1, f_2: R^2 → R^c$, with w, h, and c being their width, height and number of channels, our correlation layer lets the network compare each patch from $f_1$ with each patch from $f_2$.

为在这个匹配过程中帮助网络，我们提出了一个“相关层”，在两个特征图之间进行乘性的块比较。网络架构FlowNetCorr包含这个层，如图2下所示。给定两个多通道特征图$f_1, f_2: R^2 → R^c$，w,h,c分别为其宽度，高度和通道数量，我们的相关层让网络对$f_1$的每个块与$f_2$的每个块进行比较。

For now we consider only a single comparison of two patches. The ’correlation’ of two patches centered at $x_1$ in the first map and $x_2$ in the second map is then defined as

目前，我们只考虑两个块的单一比较。在第一个图中以$x_1$为中心，在第二个图中以$x_2$为中心的两个块的相关，然后定义为

$$c(x_1, x_2) = \sum_{o∈[-k,k]×[-k,k]} <f_1(x_1+o), f_2(x_2+o)>$$(1)

for a square patch of size K := 2k+1. Note that Eq.1 is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with other data. For this reason, it has no trainable weights.

对于一个方形块，大小K := 2k+1。注意式1与神经网络中的卷积的一个步骤是一样的，但并不是数据与滤波器卷积，而是数据与其他数据的卷积。因此，没有可训练的权重。

Computing $c(x_1, x_2)$ involves $c · K^2$ multiplications. Comparing all patch combinations involves $w^2 · h^2$ such computations, yields a large result and makes efficient forward and backward passes intractable. Thus, for computational reasons we limit the maximum displacement for comparisons and also introduce striding in both feature maps.

计算$c(x_1, x_2)$包含$c · K^2$个乘法。比较所有的图像块的组合，涉及到$w^2 · h^2$这样的计算，这样会得到一个很大的结果，使前向过程和反向过程非常难以计算。因此，由于计算原因，我们限制了比较的最大偏移，同时在两个特征图中引入了步长。

Given a maximum displacement d, for each location $x_1$ we compute correlations $c(x_1, x_2)$ only in a neighborhood of size D := 2d + 1, by limiting the range of $x_2$. We use strides $s_1$ and $s_2$, to quantize $x_1$ globally and to quantize $x_2$ within the neighborhood centered around $x_1$.

给定最大偏移d，对每个位置$x_1$，我们只在一个大小为D := 2d + 1的邻域计算$c(x_1, x_2)$，限制了$x_2$的区域。我们使用$s_1$和$s_2$的步长，对$x_1$在全局中进行量化，对$x_2$在$x_1$附近的邻域中进行量化。

In theory, the result produced by the correlation is four-dimensional: for every combination of two 2D positions we obtain a correlation value, i.e. the scalar product of the two vectors which contain the values of the cropped patches respectively. In practice we organize the relative displacements in channels. This means we obtain an output of size (w × h × D^2). For the backward pass we implemented the derivatives with respect to each bottom blob accordingly.

理论上，相关层计算出来的结果是四维的：对于2个2D位置的每个组合，我们都会得到一个相关值，即，分别是包含了剪切过的块的值的两个向量的标量积。在实践中，我们以通道来组织相对偏移。这意味着，我们得到的输出的大小为(w × h × D^2)。对反向过程，我们分别实现对每个底部blob的导数。

**Refinement**. CNNs are good at extracting high-level abstract features of images, by interleaving convolutional layers and pooling, i.e. spatially shrinking the feature maps. Pooling is necessary to make network training computationally feasible and, more fundamentally, to allow aggregation of information over large areas of the input images. However, pooling results in reduced resolution, so in order to provide dense per-pixel predictions we need a way to refine the coarse pooled representation.

**精炼**。CNNs通过交替使用卷积层和池化层，特征图在空间上不断缩小，擅长于提取图像特征的高层抽象。要使网络训练在计算上可行，池化是必须的，更基础的是，为允许输入图像在很大区域中的信息聚积。但是，池化使得分辨率下降，所以为了给出密集的逐像素预测，我们需要一个方式来提炼粗糙的池化过的表示。

Our approach to this refinement is depicted in Figure 3. The main ingredient are ‘upconvolutional’ layers, consisting of unpooling (extending the feature maps, as opposed to pooling) and a convolution. Such layers have been used previously [38, 37, 16, 28, 9]. To perform the refinement, we apply the ‘upconvolution’ to feature maps, and concatenate it with corresponding feature maps from the ’contractive’ part of the network and an upsampled coarser flow prediction (if available). This way we preserve both the high-level information passed from coarser feature maps and fine local information provided in lower layer feature maps. Each step increases the resolution twice. We repeat this 4 times, resulting in a predicted flow for which the resolution is still 4 times smaller than the input.

我们进行提炼的方法如图3所示。主要的组成部分是上卷积层，由逆池化（将特征图扩展，与池化相关）和卷积层组成。这样的层以前也曾经用过[38, 37, 16, 28, 9]。为进行提炼，我们将上卷积应用到特征图中，并与从网络收缩部分的对应的特征图、以及上采样的较粗糙的光流预测进行拼接（如果可用的话）。这样，我们既保持了从更粗糙的特征中传递过来的高层信息，也保持了在更低的层的特征图中的精细的局部信息。每个步骤都将分辨率增加两倍。我们将这个过程重复4遍，得到了预测的流，其分辨率比输入小4倍。

We discover that further refinement from this resolution does not significantly improve the results, compared to a computationally less expensive bilinear upsampling to full image resolution. The result of this bilinear upsampling is the final flow predicted by the network.

我们发现，从这个分辨率进一步提炼，并不会显著改进结果，这是与一个计算上没那么复杂的双线性上采样到图像分辨率的方法进行比较的。这个双线性上采样的结果是网络最终预测到的光流。

In an alternative scheme, instead of bilinear upsampling we use the variational approach from [6] without the matching term: we start at the 4 times downsampled resolution and then use the coarse to fine scheme with 20 iterations to bring the flow field to the full resolution. Finally, we run 5 more iterations at the full image resolution. We additionally compute image boundaries with the approach from [26] and respect the detected boundaries by replacing the smoothness coefficient by $α = exp(−λb(x,y)^κ)$, where b(x,y) denotes the thin boundary strength resampled at the respective scale and between pixels. This upscaling method is more computationally expensive than simple bilinear upsampling, but adds the benefits of variational methods to obtain smooth and subpixel-accurate flow fields. In the following, we denote the results obtained by this variational refinement with a ‘+v’ suffix. An example of variational refinement can be seen in Fig. 4.

在另一个方案中，我们没有使用双线性上采样，而是使用[6]中的变分方法，但不带有匹配项：我们从4倍下采样的分辨率开始，然后使用从粗糙到精细的方案，进行了20次迭代，以将光流场恢复到完整分辨率。最终，我们在完整图像分辨率上运行了5次更多的迭代。我们还还采用[26]中的方法计算了图像边缘，将平滑系数替换为$α = exp(−λb(x,y)^κ)$，其中b(x,y)表示细细的边缘强度，在对应的尺度和像素之间进行重新采样的。这种放大尺度的方法，比简单的双线性上采样，计算量要大的多，但将变分方法的益处加到了得到平滑的、亚像素精度的光流场中。在下面中，我们将这种通过变分提炼方法得到的结果表示为'+v'。变分提炼的例子如图4所示。

## 4. Training Data 训练数据

Unlike traditional approaches, neural networks require data with ground truth not only for optimizing several parameters, but to learn to perform the task from scratch. In general, obtaining such ground truth is hard, because true pixel correspondences for real world scenes cannot easily be determined. An overview of the available datasets is given in Table 1.

与传统的方法不同，神经网络需要带有真值的数据，不仅用于优化几个参数，还用于从头学习来进行这个任务。一般来说，得到这样的真值是很困难的，因为在真实世界场景中，真正的像素级的对应性，不是那么容易得到的。可用的数据集如表1所示。

Table 1. Size of already available datasets and the proposed Flying Chairs dataset.

| | Frame pairs | Frames with ground truth | Ground truth density per frame
--- | --- | --- | ---
Middlebury | 72 | 8 | 100%
KITTI | 194 | 194 | ~50%
Sintel | 1041 | 1041 | 100%
Flying Chairs | 22872 | 22872 | 100%

### 4.1. Existing Datasets

The Middlebury dataset [2] contains only 8 image pairs for training, with ground truth flows generated using four different techniques. Displacements are very small, typically below 10 pixels.

Middlebury数据集[2]只有8个图像对用于训练，真值光流的生成使用了四种不同的技术。偏移都非常小，通常少于10个像素。

The KITTI dataset [14] is larger (194 training image pairs) and includes large displacements, but contains only a very special motion type. The ground truth is obtained from real world scenes by simultaneously recording the scenes with a camera and a 3D laser scanner. This assumes that the scene is rigid and that the motion stems from a moving observer. Moreover, motion of distant objects, such as the sky, cannot be captured, resulting in sparse optical flow ground truth.

KITTI数据集[14]更大一些（194个训练图像对），并包含较大的偏移，但只包含了一种非常特殊的运动类型。真值是从真实世界场景得到的，同时用相机和3D激光扫描仪来记录场景。这假设场景是刚性的，运动的产生是由于观察者在移动。但是，远处目标的移动，比如天空，不能被捕获到，得到的是很稀疏的光流真值。

The MPI Sintel [7] dataset obtains ground truth from rendered artificial scenes with special attention to realistic image properties. Two versions are provided: the Final version contains motion blur and atmospheric effects, such as fog, while the Clean version does not include these effects. Sintel is the largest dataset available (1,041 training image pairs for each version) and provides dense ground truth for small and large displacement magnitudes.

MPI Sintel[7]数据集从渲染的人工场景中得到真值，特别关注的真实的图像性质。有两个版本：最终版包含运动模糊，和大气效果，比如雾，而清晰版本并没有包含这样的效果。Sintel是可用的最大的数据集（每个版本包含1041训练图像对），为小的和大的偏移幅度提供了密集真值。

### 4.2. Flying Chairs

The Sintel dataset is still too small to train large CNNs. To provide enough training data, we create a simple synthetic dataset, which we name Flying Chairs, by applying affine transformations to images collected from Flickr and a publicly available rendered set of 3D chair models [1]. We retrieve 964 images from Flickr with a resolution of 1, 024 × 768 from the categories ‘city’ (321), ‘landscape’ (129) and ‘mountain’ (514). We cut the images into 4 quadrants and use the resulting 512 × 384 image crops as background. As foreground objects we add images of multiple chairs from [1] to the background. From the original dataset we remove very similar chairs, resulting in 809 chair types and 62 views per chair available. Examples are shown in Figure 5.

Sintel数据集仍然太小，不能训练大型CNNs。为提供足够的训练数据，我们创建了一个简单的合成数据集，我们命名为Flying Chairs，将仿射变换应用到从Flickr收集到的图像，和一个公开可用的渲染的3D椅子模型[1]。我们从Flickr获取了964幅图像，分辨率为1024 × 768，类别为‘city’ (321), ‘landscape’ (129)和‘mountain’ (514)。我们将图像切割成4块，使用得到的512 × 384图像剪切块，作为背景。前景目标为我们加入的多个椅子。从原始数据集中，我们移除了很类似的椅子，得到809个椅子类型，每个椅子有62个视角。例子如图5所示。

To generate motion, we randomly sample affine transformation parameters for the background and the chairs. The chairs’ transformations are relative to the background transformation, which can be interpreted as both the camera and the objects moving. Using the transformation parameters we render the second image, the optical flow and occlusion regions.

为生成运动，我们对仿射变换参数进行随机采样，用于背景和椅子。椅子的变换是相对于背景变换的，这可以解释为相机和目标的运动。使用变换参数，我们对第二幅图像进行渲染，光流和遮挡区域。

All parameters for each image pair (number, types, sizes and initial positions of the chairs; transformation parameters) are randomly sampled. We adjust the random distributions of these parameters in such a way that the resulting displacement histogram is similar to the one from Sintel (details can be found in the supplementary material). Using this procedure, we generate a dataset with 22,872 image pairs and flow fields (we re-use each background image multiple times). Note that this size is chosen arbitrarily and could be larger in principle.

每个图像对的所有参数（椅子的数量，类型，大小和初始姿态；变换参数）都是随机采样的。我们调整了这些参数的随机分布，其方式是，得到的偏移直方图与Sintel中的类似（细节可以在附加材料中得到）。使用这个过程，我们生成了22872个图像对和光流的数据集（我们多次重新使用每个背景图像）。注意，大小是随意选择的，原则上可以更大。

### 4.3. Data Augmentation

A widely used strategy to improve generalization of neural networks is data augmentation [24, 10]. Even though the Flying Chairs dataset is fairly large, we find that using augmentations is crucial to avoid overfitting. We perform augmentation online during network training. The augmentations we use include geometric transformations: translation, rotation and scaling, as well as additive Gaussian noise and changes in brightness, contrast, gamma, and color. To be reasonably quick, all these operations are processed on the GPU. Some examples of augmentation are given in Fig. 5.

改进神经网络的泛化能力的一个广泛使用的策略是数据扩增。虽然Flying Chairs数据集已经相当大了，我们发现使用扩增对防止过拟合仍然是很关键的。我们在网络训练时在线进行扩增。我们使用的扩增包括几何变形：平移，旋转和缩放，以及加性的高斯噪声，和亮度、gamma值、对比度和色彩的变化。为快速处理，所有这些运算都是在GPU上进行的。扩增的一些例子如图5所示。

As we want to increase not only the variety of images but also the variety of flow fields, we apply the same strong geometric transformation to both images of a pair, but additionally a smaller relative transformation between the two images. We adapt the flow field accordingly by applying the per-image augmentations to the flow field from either side.

由于我们不止要增加图像的多样性，而且还要增加光流的多样性，我们对图像对使用了相同的强几何变换，而且对图像对之间还额外应用了一个更小的相对变换。我们相应的对光流进行自适应，对每幅图像的扩增应用到光流中。

Specifically we sample translation from a the range [−20%, 20%] of the image width for x and y; rotation from [−17◦, 17◦]; scaling from [0.9, 2.0]. The Gaussian noise has a sigma uniformly sampled from [0, 0.04]; contrast is sampled within [−0.8, 0.4]; multiplicative color changes to the RGB channels per image from [0.5, 2]; gamma values from [0.7, 1.5] and additive brightness changes using Gaussian with a sigma of 0.2.

具体的，我们对平移范围进行采样，范围是[−20%, 20%]的图像宽度，包括x和y方向；旋转角度为[−17◦, 17◦]；缩放尺度为[0.9, 2.0]。高斯噪声的sigma从[0, 0.04]均匀采样；对比度在[−0.8, 0.4]内采样；RGB通道的乘性色彩变化，每幅图像为[0.5, 2]；gamma值为[0.7, 1.5]，加性亮度变化使用sigma为0.2的高斯函数。

## 5. Experiments

We report the results of our networks on the Sintel, KITTI and Middlebury datasets, as well as on our synthetic Flying Chairs dataset. We also experiment with fine-tuning of the networks on Sintel data and variational refinement of the predicted flow fields. Additionally, we report runtimes of our networks, in comparison to other methods.

我们在四个数据集上给出我们网络的结果，Sintel，KITTI和Middlebury数据集，以及我们的合成Flying Chairs数据集。我们还进行了实验，在Sintel数据上精调了网络，以及预测流场的变分提炼。另外，我们给出我们网络的运行时间，与其他方法进行了对比。

### 5.1. Network and Training Details

The exact architectures of the networks we train are shown in Fig. 2. Overall, we try to keep the architectures of different networks consistent: they have nine convolutional layers with stride of 2 (the simplest form of pooling) in six of them and a ReLU nonlinearity after each layer. We do not have any fully connected layers, which allows the networks to take images of arbitrary size as input. Convolutional filter sizes decrease towards deeper layers of networks: 7 × 7 for the first layer, 5 × 5 for the following two layers and 3 × 3 starting from the fourth layer. The number of feature maps increases in the deeper layers, roughly doubling after each layer with a stride of 2. For the correlation layer in FlowNetC we chose the parameters k = 0, d = 20, s1 = 1, s2 = 2. As training loss we use the endpoint error (EPE), which is the standard error measure for optical flow estimation. It is the Euclidean distance between the predicted flow vector and the ground truth, averaged over all pixels.

我们训练的网络的精确结构如图2所示。总体上，我们保持不同的网络架构一致：它们都有9个卷积层，其中6个步长为2（形式最简单的池化），每一层都有一个ReLU非线性单元。我们没有任何全连接层，这使得网络可以输入任意大小的图像。卷积滤波器大小随着网络加深逐渐减小：第一层为7 × 7，后面两层为5 × 5，从第4层为3 × 3。特征图的数量在更深的层中会增加，基本上是在步长为2的层后数量加倍。对于在FlowNetC中的相关层，我们选择的参数为k = 0, d = 20, s1 = 1, s2 = 2。训练损失函数，我们使用endpoint误差(EPE)，这是光流估计的标准误差度量。这是预测的光流向量和真值的欧式距离，在所有像素上进行平均。

For training CNNs we use a modified version of the caffe [20] framework. We choose Adam [22] as optimization method because for our task it shows faster convergence than standard stochastic gradient descent with momentum. We fix the parameters of Adam as recommended in [22]: β1 = 0.9 and β2 = 0.999. Since, in a sense, every pixel is a training sample, we use fairly small mini-batches of 8 image pairs. We start with learning rate λ = 1e−4 and then divide it by 2 every 100k iterations after the first 300k. With FlowNetCorr we observe exploding gradients with λ = 1e−4. To tackle this problem, we start by training with a very low learning rate λ = 1e−6, slowly increase it to reach λ = 1e−4 after 10k iterations and then follow the schedule just described.

对于CNNs的训练，我们使用修改的caffe[20]框架。我们选择Adam优化器，因为对于我们的任务，比带有动量的SGD展示出了更快的收敛速度。我们固定Adam的参数为[22]中所推荐的：β1 = 0.9，β2 = 0.999。在某种意义上，每个像素都是一个训练样本，所以我们使用的mini-batch大小相对较小，为8个图像对。我们以学习速率λ = 1e−4开始，，在前300k次迭代中每100k次迭代除以2。对于FlowNetCorr，在λ = 1e−4时，我们观察到了梯度爆炸。为处理这个问题，我们以非常低的训练速率λ = 1e−6开始，在10k次迭代后慢慢的将其增加至λ = 1e−4，然后按照刚刚描述的方案进行。

To monitor overfitting during training and fine-tuning, we split the Flying Chairs dataset into 22, 232 training and 640 test samples and split the Sintel training set into 908 training and 133 validation pairs.

为监控训练和精调过程中过拟合，我们将Flying Chairs数据集分割成22, 232训练集和640个测试集，将Sintel分裂成980个样本的训练集和133个验证对。

We found that upscaling the input images during testing may improve the performance. Although the optimal scale depends on the specific dataset, we fixed the scale once for each network for all tasks. For FlowNetS we do not upscale, for FlowNetC we chose a factor of 1.25.

我们发现在测试时将输入图像放大，会改进性能。虽然最佳尺度依赖于具体的数据集，我们对所有任务对每个网络固定了尺度。对于FlowNetS我们不会放大尺度，对于FlowNetC我们选择的因子为1.25。

**Fine-tuning**. The used datasets are very different in terms of object types and motions they include. A standard solution is to fine-tune the networks on the target datasets. The KITTI dataset is small and only has sparse flow ground truth. Therefore, we choose to fine-tune on the Sintel training set. We use images from the Clean and Final versions of Sintel together and fine-tune using a low learning rate λ = 1e−6 for several thousand iterations. For best performance, after defining the optimal number of iterations using a validation set, we then fine-tune on the whole training set for the same number of iterations. In tables we denote finetuned networks with a ‘+ft’ suffix.

**精调**。使用的数据集在目标类型和包含的运动类型上非常不同。标准方法是在目标数据集上对网络进行精调。KITTI数据集很小，只有稀疏的光流真值。因此，我们选择来精调Sintel训练集。我们使用从清洗过的和最终版的Sintel数据集中的图像一起，使用很低的初始学习率λ = 1e−6迭代数千次迭代来进行精调。对于最佳的性能，在使用验证集定义了最佳数量的迭代次数后，然后我们在整个训练集上进行精调，进行同样次数的迭代。在表格中，我们以'+ft'的后缀来表示精调的网络。

### 5.2. Results

Table 2 shows the endpoint error (EPE) of our networks and several well-performing methods on public datasets (Sintel, KITTI, Middlebury), as well as on our Flying Chairs dataset. Additionally we show runtimes of different methods on Sintel.

表2给出了我们网络和几个表现不错的方法在公开数据集上的endpoint误差(EPE)，包括Sintel，KITTI，Middlebury，以及我们的Flying Chairs。另外，我们给出了不同方法在Sintel上的运行时间。

The networks trained just on the non-realistic Flying Chairs perform very well on real optical flow datasets, beating for example the well-known LDOF [6] method. After fine-tuning on Sintel our networks can outperform the competing real-time method EPPM [3] on Sintel Final and KITTI while being twice as fast.

在非真实的Flying Chairs数据集上训练得到网络，在真实光流数据集上表现也非常好，比如打败了著名的LDOF[6]方法。在Sintel精调了我们的网络后，在Sintel Final和KITTI上表现超过了实时方法EPPM[3]，而且速度快了2倍。

**Sintel**. From Table 2 one can see that FlowNetC is better than FlowNetS on Sintel Clean, while on Sintel Final the situation changes. On this difficult dataset, FlowNetS+ft+v is even on par with DeepFlow. Since the average endpoint error often favors over-smoothed solutions, it is interesting to see qualitative results of our method. Figure 7 shows examples of the raw optical flow predicted by the two FlowNets (without fine-tuning), compared to ground truth and EpicFlow. The figure shows how the nets often produce visually appealing results, but are still worse in terms of endpoint error. Taking a closer look reveals that one reason for this may be the noisy non-smooth output of the nets especially in large smooth background regions. This we can partially compensate with variational refinement.

**Sintel**。从表2中可以看出，FlowNetC比FlowNetS在Sintel Clean上效果要好，而在Sintel Final上表现变了过来。在这个较难的数据集上，FlowNetS+ft+v甚至与DeepFlow效果类似。由于平均端点误差通常在平滑的解决方案中比较好，看到我们的方法的定性结果还是很有趣的。图7给出了两个FlowNets预测出来的原始光流的例子（没有经过精调），并与真值和EpicFlow进行了比较。图示表明，网络通常可以给出视觉上非常不错的结果，但在endpoint误差上仍然非常很差。更近距离看一下，可以看出，一个原因是，网络的含噪非平滑输出，尤其是在平滑的背景区域中。这里我们可以用变分提炼进行部分补偿。

**KITTI**. The KITTI dataset contains strong projective transformations which are very different from what the networks encountered during training on Flying Chairs. Still, the raw network output is already fairly good, and additional fine-tuning and variational refinement give a further boost. Interestingly, fine-tuning on Sintel improves the results on KITTI, probably because the images and motions in Sintel are more natural than in Flying Chairs. The FlowNetS outperforms FlowNetC on this dataset.

**KITTI**。KITTI数据集包含很强的投影变换，与网络在Flying Chairs上训练遇到的非常不同。但是，原始网络输出已经非常不错了，另外的精调和变分精炼会给出进一步的效果提升。有趣的是，在Sintel上精调改进了在KITTI上的结果，可能是因为在Sintel上的图像和运动比在Flying Chairs上更加自然。在这个数据集上，FlowNetS比FlowNetC性能要更好。

**Flying Chairs**. Our networks are trained on the Flying Chairs, and hence are expected to perform best on those. When training, we leave aside a test set consisting of 640 images. Table 2 shows the results of various methods on this test set, some example predictions are shown in Fig. 6. One can see that FlowNetC outperforms FlowNetS and that the nets outperform all state-of-the-art methods. Another interesting finding is that this is the only dataset where the variational refinement does not improve performance but makes things worse. Apparently the networks can do better than variational refinement already. This indicates that with a more realistic training set, the networks might also perform even better on other data.

**Flying Chairs**。我们的网络是在Flying Chairs上训练的，因此应当在这个数据集上表现最好。当训练时，我们分离出来一个测试集，包含640幅图像。表2给出了各种方法在这个测试集中的结果，一些预测例子如图6所示。一个人可以看到，FlowNetC比FlowNetS效果要好，网络比所有目前最好的方法效果都要好。另一个有趣的发现是，只有在这个数据集上，变分提炼并没有改进性能，而是使效果变得更差。很明显，网络可以比变分提炼效果做的更好。这说明，在一个更实际的训练集上，网络甚至会在其他数据上表现更好。

**Timings**. In Table 2 we show the per-frame runtimes of different methods in seconds. Unfortunately, many methods only provide the runtime on a single CPU, whereas our FlowNet uses layers only implemented on GPU. While the error rates of the networks are below the state of the art, they are the best among real-time methods. For both training and testing of the networks we use an NVIDIA GTX Titan GPU. The CPU timings of DeepFlow and EpicFlow are taken from [30], while the timing of LDOF was computed on a single 2.66GHz core.

**计时**。在表2中，我们给出了不同方法的逐帧运行时间，单位为秒。很多方法只给出在单个CPU上的运行时间，而我们的FlowNet在GPU上给出了实现。网络的错误率比目前最好的效果还要低，在实时方法中是最好的。网络的训练和测试，我们使用NVIDIA GTX Titan GPU。DeepFlow和EpicFlow的CPU运行时间是从[30]中得到的，而LDOF的运行时间是在单核2.66GHz上计算得到的。

### 5.3. Analysis

**Training data**. To check if we benefit from using the Flying Chairs dataset instead of Sintel, we trained a network just on Sintel, leaving aside a validation set to control the performance. Thanks to aggressive data augmentation, even Sintel alone is enough to learn optical flow fairly well. When testing on Sintel, the network trained exclusively on Sintel has EPE roughly 1 pixel higher than the net trained on Flying Chairs and fine-tuned on Sintel.

**训练数据**。为检查使用Flying Chairs数据集是否比使用Sintel要好，我们在Sintel上训练了一个网络，分出了一部分验证集，以控制性能。多亏了数据扩增用的比较重，即使是Sintel一个数据集上，也可以学习到不错的光流。当在Sintel上测试时，只在Sintel上训练得到的网络，比在Flying Chairs上训练并在Sintel上精调的的要高大概1个EPE。

The Flying Chairs dataset is fairly large, so is data augmentation still necessary? The answer is positive: training a network without data augmentation on the Flying Chairs results in an EPE increase of roughly 2 pixels when testing on Sintel.

Flying Chairs数据集相对较大，所以数据扩增是否还必要？答案是还需要：在Flying Chairs上不用数据扩增训练得到的网络，在Sintel上进行测试，EPE增大了大概2个像素。

**Comparing the architectures**. The results in Table 2 allow to draw conclusions about strengths and weaknesses of the two architectures we tested.

**比较架构**。从表2中的结果，我们可以对我们测试的两种架构，分析其强处和弱点，得到结论。

First, FlowNetS generalizes to Sintel Final better than FlowNetC. On the other hand, FlowNetC outperforms FlowNetS on Flying chairs and Sintel Clean. Note that Flying Chairs do not include motion blur or fog, as in Sintel Final. These results together suggest that even though the number of parameters of the two networks is virtually the same, the FlowNetC slightly more overfits to the training data. This does not mean the network remembers the training samples by heart, but it adapts to the kind of data it is presented during training. Though in our current setup this can be seen as a weakness, if better training data were available it could become an advantage.

首先，FlowNetS比FlowNetC，可以更好的在Sintel Final上泛化。另一方面，FlowNetC在Flying Chairs和Sintel Clean上，比FlowNetS上要性能更好。注意，Flying Chairs数据集中并没有包含运动模糊或雾，Sintel Final也没有。这些结果一起说明，即使两个网络的参数数量大概类似，FlowNetC略微有些在训练集上过拟合。这并不意味着网络记住了训练数据，而是适应了训练时接触的数据。虽然在我们目前的设置中，这可以认为是一个弱点，如果有更好的训练数据可用，则可以是一个优势。

Second, FlowNetC seems to have more problems with large displacements. This can be seen from the results on KITTI discussed above, and also from detailed performance analysis on Sintel Final (not shown in the tables). FlowNetS+ft achieves an s40+ error (EPE on pixels with displacements of at least 40 pixels) of 43.3px, and for FlowNetC+ft this value is 48px. One explanation is that the maximum displacement of the correlation does not allow to predict very large motions. This range can be increased at the cost of computational efficiency.

第二，FlowNetC似乎对大的偏移有不少问题。这可以从上面讨论的KITTI的结果看到，同时在Sintel Final的细节性能分析中也可以看到（没有在表格中展现）。FlowNetS+ft在s40+误差的数据上（像素上的EPE偏差至少40像素）获得的结果是43.3px，而对于FlowNetC+ft这个结果是48px。一种解释是，关联的最大偏差并不允许大的运动。这个范围可以增加，但代价是计算效率。

## 6. Conclusion

Building on recent progress in design of convolutional network architectures, we have shown that it is possible to train a network to directly predict optical flow from two input images. Intriguingly, the training data need not be realistic. The artificial Flying Chairs dataset including just affine motions of synthetic rigid objects is sufficient to predict optical flow in natural scenes with competitive accuracy. This proves the generalization capabilities of the presented networks. On the test set of the Flying Chairs the CNNs even outperform state-of-the-art methods like DeepFlow and EpicFlow. It will be interesting to see how future networks perform as more realistic training data becomes available.

在最近设计CNN的进展上，我们已经证明，训练一个网络从两幅输入图像中直接预测光流，是可能的。有趣的是，训练数据不需要是真实的。人工合成的Flying Chairs数据集，只包含了合成的刚体的仿射运动，足以在自然场景中以不错的准确率预测光流。这证明了现有网络的泛化能力。在Flying Chairs的测试集上，CNNs甚至超过了目前最好的方法，如DeepFlow和EpicFlow。当有更多的真实训练数据时，网络表现如何，这一定非常有趣。