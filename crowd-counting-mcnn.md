# Single-Image Crowd Counting via Multi-Column Convolutional Neural Network

Yingying Zhang et al. Shanghai Tech University

## Abstract 摘要

This paper aims to develop a method than can accurately estimate the crowd count from an individual image with arbitrary crowd density and arbitrary perspective. To this end, we have proposed a simple but effective Multi-column Convolutional Neural Network (MCNN) architecture to map the image to its crowd density map. The proposed MCNN allows the input image to be of arbitrary size or resolution. By utilizing filters with receptive fields of different sizes, the features learned by each column CNN are adaptive to variations in people/head size due to perspective effect or image resolution. Furthermore, the true density map is computed accurately based on geometry-adaptive kernels which do not need knowing the perspective map of the input image. Since exiting crowd counting datasets do not adequately cover all the challenging situations considered in our work, we have collected and labelled a large new dataset that includes 1198 images with about 330,000 heads annotated. On this challenging new dataset, as well as all existing datasets, we conduct extensive experiments to verify the effectiveness of the proposed model and method. In particular, with the proposed simple MCNN model, our method outperforms all existing methods. In addition, experiments show that our model, once trained on one dataset, can be readily transferred to a new dataset.

本文目标是提出一种可以准确的估计单幅图像中的人群计数，图像视角任意，人群密度任意。为此，我们提出一种简单但有效的多列卷积神经网络(MCNN)架构，将图像映射到人群密度图。提出的MCNN允许输入图像大小任意，或分辨率任意。利用不同大小感受野的滤波器，每列CNN学到的特征对人/头的大小自适应变化，大小的变化通常是由于视角效应或图像分辨率造成的。而且，密度图真值是基于几何自适应的核准确计算得到的，不需要知道输入图像的视角图。既然现有的人群计数数据集不足以覆盖我们工作中考虑到的所有有挑战性的情况，我们收集并标注了一个大型数据集，包括1198幅图像，共标注了大约330000个人头。在这个有挑战性的新数据集上，以及现有的数据集上，我们进行了广泛的实验，以验证我们提出的模型和方法的有效性。特别是，我们提出的MCNN模型超过了现有的全部方法。另外，实验表明，我们的模型在一个数据集上训练过后，可以很容易的泛化到新数据集上。

## 1. Introduction 引言

In the new year eve of 2015, 35 people were killed in a massive stampede in Shanghai, China. Unfortunately, since then, many more massive stampedes have taken place around the world which have claimed many more victims. Accurately estimating crowds from images or videos has become an increasingly important application of computer vision technology for purposes of crowd control and public safety. In some scenarios, such as public rallies and sports events, the number or density of participating people is an essential piece of information for future event planning and space design. Good methods of crowd counting can also be extended to other domains, for instance, counting cells or bacteria from microscopic images, animal crowd estimates in wildlife sanctuaries, or estimating the number of vehicles at transportation hubs or traffic jams, etc.

在2015年新年前夜，35人死于中国上海的一次大型踩踏事故中。不幸的是，从那时起，世界范围内发生了更多的大型踩踏事故，受害者越来越多。从图像或视频中准确的估计人群人数，已经成为计算机视觉技术的一种越来越重要的应用，其目的就是群体控制和公共安全。在一些场景中，比如公开集会和体育活动，参与的人们的密度，对于未来的事件计划和空间设计，是很关键的一部分信息。好的人群技术的方法也可以拓展到其他领域中，比如，从显微图像中对细胞或细菌技术，野生动物聚积场所的动物群体数量估计，或在交通堵塞期间估计交通工具数量，等等。

**Related work**. Many algorithms have been proposed in the literature for crowd counting. Earlier methods [29] adopt a detection-style framework that scans a detector over two consecutive frames of a video sequence to estimate the number of pedestrians, based on boosting appearance and motion features. [19, 30, 31] have used a similar detection-based framework for pedestrian counting. In detection-based crowd counting methods, people typically assume a crowd is composed of individual entities which can be detected by some given detectors [13, 34, 18, 10]. The limitation of such detection-based methods is that occlusion among people in a clustered environment or in a very dense crowd significantly affects the performance of the detector hence the final estimation accuracy.

**相关工作**。文献中提出了许多人群计数的算法。早期的方法[29]采用基于检测的框架，检测器检测两幅连续视频帧，以估计行人的数量，基于boosting外观和运动特征。[19,30,31]使用了类似的基于检测的框架进行行人计数。在基于检测的人群计数方法中，通常假设人群是由个体的人构成的，而且是可以由给定的检测器检测到的[13,34,18,10]。这种基于检测的方法的局限性在于聚集环境中的人与人的遮挡会严重影响检测器的效果，也就影响最终的估计准确率。

In counting crowds in videos, people have proposed to cluster trajectories of tracked visual features. For instance, [24] has used highly parallelized version of the KLT tracker and agglomerative clustering to estimate the number of moving people. [3] has tracked simple image features and probabilistically group them into clusters representing independently moving entities. However, such tracking-based methods do not work for estimating crowds from individual still images.

在视频中进行人群计数，人们曾经提出过对视觉特征轨迹路线进行聚类。比如，[24]使用了高度并行的KLT追踪器和agglomerative聚类来估计运动行人的数量。[3]追踪了简单的图像特征，依据概率将其群聚为聚类，代表运动的个体。但是，这些基于追踪的方法对于在静止图像中的人群估计任务中并不适用。

Arguably the most extensively used method for crowd counting is feature-based regression, see [4, 7, 5, 27, 15, 20]. The main steps of this kind of method are: 1) segmenting the foreground; 2) extracting various features from the foreground, such as area of crowd mask [4, 7, 27, 23], edge count [4, 7, 27, 25], or texture features [22, 7]; 3) utilizing a regression function to estimate the crowd count. Linear [23] or piece-wise linear [25] functions are relatively simple models and yield decent performance. Other more advanced/effective methods are ridge regression (RR) [7], Gaussian process regression (GPR) [4], and neural network (NN) [22].

人群计数最广泛使用的方法是基于特征的回归，见[4,7,5,27,15,20]。这类方法的主要步骤是：1)分割前景；2)从前景中提取各种特征，如人群掩膜区域[4,7,27,23]，边缘计数[4,7,27,25]，或纹理特征[22,7]；3)利用回归函数来估计人群数量。线性函数[23]或分段线性函数[25]是相对简单的模型，可以得到不错的结果。其他更高级/有效的方法包括脊回归(RR)[7]，高斯过程回归(GPR)[4]和神经网络[22]。

There have also been some works focusing on crowd counting from still images. [12] has proposed to leverage multiple sources of information to compute an estimate of the number of individuals present in an extremely dense crowd visible in a single image. In that work, a dataset of fifty crowd images containing 64K annotated humans (UCF CC 50) is introduced. [2] has followed the work and estimated counts by fusing information from multiple sources, namely, interest points (SIFT), Fourier analysis, wavelet decomposition, GLCM features, and low confidence head detections. [28] has utilized the features extracted from a pre-trained CNN to train a support vector machine (SVM) that subsequently generates counts for still images.

也有很多工作关注从静止图像中进行人群计数。[12]提出了利用多源信息来计算单幅图像中极度拥挤的人群的个体数量估计。在这篇文章中，提出了一个数据集，包括50幅拥挤的图像，共计6.4万个标注的人(UCF CC 50)。[2]沿着这个工作，融合了多源信息进行人数估计，包括兴趣点(SIFT)，傅立叶变换，小波分解，GLCM特征和低置信度的人头检测。[28]利用了从预训练CNN中提取出的特征，来训练一个SVM，从静止图像中生成人数估计。

Recently Zhang et al. [33] has proposed a CNN based method to count crowd in different scenes. They first pretrain a network for certain scenes. When a test image from a new scene is given, they choose similar training data to fine-tune the pretrained network based on the perspective information and similarity in density map. Their method demonstrates good performance on most existing datasets. But their method requires perspective maps both on training scenes and the test scene. Unfortunately, in many practical applications of crowd counting, the perspective maps are not readily available, which limits the applicability of such methods.

最近Zhang等[33]提出了在各种不同场景下的基于CNN的人数统计。他们首先对某些场景预训练一个网络。当给定新场景中的测试图像，他们基于视角信息和密度图中的相似性，来选择类似的训练数据来精调预训练的网络。他们的方法在现有的大多数数据集中都给出了很好的表现。但是他们的方法需要训练场景和测试场景的视角图。不幸的是，在很多人群计数的实际应用中，视角图并不一定可用，这限制了这种方法的可用性。

**Contributions of this paper**. In this paper, we aim to conduct accurate crowd counting from an arbitrary still image, with an arbitrary camera perspective and crowd density (see Figure 1 for some typical examples). At first sight this seems to be a rather daunting task, since we obviously need to conquer series of challenges:

**本文贡献**。本文中，我们的目标是从任意的静止图像中进行准确的人群计数，图像的拍摄视角任意，人群密度任意（如图1所示的一些典型例子）。乍一看来这是一个非常让人却步的工作，因为显然要面对一一系列挑战：

- Foreground segmentation is indispensable in most existing work. However foreground segmentation is a challenging task all by itself and inaccurate segmentation will have irreversible bad effect on the final count. In our task, the viewpoint of an image can be arbitrary. Without information about scene geometry or motion, it is almost impossible to segment the crowd from its background accurately. Hence, we have to estimate the number of crowd without segmenting the foreground first.

- 前景分割在多数现有的工作中是必不可少的部分。但是前景分割是一项很有挑战性的任务，分割的不准确，会导致最终计数结果错误，而且不可修复。在我们的任务中，图像的视角是任意的。没有场景几何或运动的信息，几乎不可能正确的从背景中分割出人群。所以，我们不能首先分割前景，然后估计人群数量。

- The density and distribution of crowd vary significantly in our task (or datasets) and typically there are tremendous occlusions for most people in each image. Hence traditional detection-based methods do not work well on such images and situations.

- 我们的任务（或数据集）中，人群密度和分布变化非常大，而且通常每幅图像的多数人有很多的遮挡。所以传统的基于检测的方法在这种图像和情况下不能很好的工作。

- As there might be significant variation in the scale of the people in the images, we need to utilize features at different scales all together in order to accurately estimate crowd counts for different images. Since we do not have tracked features and it is difficult to handcraft features for all different scales, we have to resort to methods that can automatically learn effective features.

- 由于图像中人的尺度会有显著的变化，我们需要利用不同尺度的特征，以准确的估计不同图像中的人群数量。由于我们没有追踪特征，也很难对所有的尺度进行人工设计特征，我们只能依靠能自动学习特征的方法。

To overcome above challenges, in this work, we propose a novel framework based on convolutional neural network (CNN) [9, 16] for crowd counting in an arbitrary still image. More specifically, we propose a multi-column convolutional neural network (MCNN) inspired by the work of [8], which has proposed multi-column deep neural networks for image classification. In their model, an arbitrary number of columns can be trained on inputs preprocessed in different ways. Then final predictions are obtained by averaging individual predictions of all deep neural networks. Our MCNN contains three columns of convolutional neural networks whose filters have different sizes. Input of the MCNN is the image, and its output is a crowd density map whose integral gives the overall crowd count. Contributions of this paper are summarized as follows:

为克服以上的挑战，在本文中，我们提出了一种新的基于CNN的框架[9,16]在任意静止图像中进行人群计数。具体来说，我们受到[8]的启发提出了一种多列CNN(MCNN)，[8]也提出了多列DNN进行图像分类。在其模型中，可以在不同预处理的输入上训练任意数量的列。然后通过平均所有单个深度神经网络的预测，得到最终预测。我们的MCNN包含了三列卷积神经网络，其滤波器大小不同。MCNN的输入为图像，其输出为人群密度图，密度图的积分就是人群计数的总量。本文的贡献总结如下：

- The reason for us to adopt a multi-column architecture here is rather natural: the three columns correspond to filters with receptive fields of different sizes (large, medium, small) so that the features learned by each column CNN is adaptive to (hence the overall network is robust to) large variation in people/head size due to perspective effect or across different image resolutions.

- 我们采用多列架构的原因是非常自然的：三列对应着不同感受野大小的滤波器（大，中，小），每列CNN学习到的特征是自适应于人/头大小的变化的，所以整体的网络对尺度变化是稳健的，这些尺度变化可能是因为视角的效果或不同的图像分辨率。

- In our MCNN, we replace the fully connected layer with a convolution layer whose filter size is 1 × 1. Therefore the input image of our model can be of arbitrary size to avoid distortion. The immediate output of the network is an estimate of the density of the crowd from which we derive the overall count.

- 在我们的MCNN中，我们将全连接层替换为卷积层，滤波器大小为1×1。所以我们模型的输入图像可以是任意大小，这样就可以避免变形。网络的输出是人群密度的估计，从中可以推导得到共计的数量。

- We collect a new dataset for evaluation of crowd counting methods. Existing crowd counting datasets cannot fully test the performance of an algorithm in the diverse scenarios considered by this work because their limitations in the variation in viewpoints (UCSD, WorldExpo’10), crowd counts (UCSD), the scale of dataset (UCSD, UCF CC 50), or the variety of scenes (UCF CC 50). In this work we introduce a new large-scale crowd dataset named Shanghaitech of nearly 1,200 images with around 330,000 accurately labeled heads. As far as we know, it is the largest crowd counting dataset in terms of number annotated heads. No two images in this dataset are taken from the same viewpoint. This dataset consists of two parts: Part A and Part B. Images in Part A are randomly crawled from the Internet, most of them have a large number of people. Part B are taken from busy streets of metropolitan areas in Shanghai. We have manually annotated both parts of images and will share this dataset by request. Figure 1 shows some representative samples of this dataset.

- 我们收集制作了一个新的数据集，来评估人群计数方法。现有的人群计数数据集不能在各种不同场景中完全的测试我们算法的性能，因为数据集的视角变化有限制（UCSD，WorldExpo'10），或人群数量有限制(UCSD)，或数据集的规模(UCSD, UCF CC 50)，或场景的种类有限制(UCF CC 50)。在本文中，我们提出了一个新的大规模人群数据集，名为Shanghaitech，有接近1200幅图像，共计33万个精确标注的人头。据我们所知，这是最大的人群计数的数据集，以标注的人头数量计算。数据集中图像的视角都是不同的。这个数据集包括两部分：Part A和Part B。Part A中的图像是随机从网络上爬取的，多数都有很多人在其中。Part B是从上海市区繁忙的街道上拍摄的。我们手工标注了这两部分图像，并分享了这两个数据集。图1是数据集中的一些代表性例子。

Figure 1: (a) Representative images of Part A in our new crowd dataset. (b) Representative images of Part B in our crowd dataset. All faces are blurred in (b) for privacy preservation.

## 2. Multi-column CNN for Crowd Counting 人群计数的多列CNN

### 2.1. Density map based crowd counting 基于密度图的人群计数

To estimate the number of people in a given image via the Convolutional Neural Networks (CNNs), there are two natural configurations. One is a network whose input is the image and the output is the estimated head count. The other one is to output a density map of the crowd (say how many people per square meter), and then obtain the head count by integration. In this paper, we are in favor of the second choice for the following reasons:

为通过卷积神经网络CNN估计给定图像中的人数，有两个很自然的选择。一种是，网络输入是图像，输出是估计的人头数。另一个是输出人群的密度图（如每平方米有多少人），然后通过积分得到人头的数量。在本文中，我们选择第二个选项，因为：

- Density map preserves more information. Compared to the total number of the crowd, density map gives the spatial distribution of the crowd in the given image, and such distribution information is useful in many applications. For example, if the density in a small region is much higher than that in other regions, it may indicate something abnormal happens there.

- 密度图保留了更多的信息。与人群总数相比，密度图给出了给定图像中的人群空间分布，这种分布信息在很多应用都很有用。比如，如果一个小区域中的密度比其他区域都高的多，这就说明那里发生了什么异常的事。

- In learning the density map via a CNN, the learned filters are more adapted to heads of different sizes, hence more suitable for arbitrary inputs whose perspective effect varies significantly. Thus the filters are more semantic meaningful, and consequently improves the accuracy of crowd counting.

- 在通过CNN学习密度图的过程中，学习的滤波器更适应不同大小的人头，所以更适应任意大小的输入，视角也变化多样。所以滤波器在语义上更加有意义，结果改进了人群计数的准确度。

### 2.2. Density map via geometry-adaptive kernels 通过几何自适应核的密度图

Since the CNN needs to be trained to estimate the crowd density map from an input image, the quality of density given in the training data very much determines the performance of our method. We first describe how to convert an image with labeled people heads to a map of crowd density. If there is a head at pixel $x_i$, we represent it as a delta function $δ(x − x_i)$. Hence an image with N heads labeled can be represented as a function

由于CNN需要训练从输入图像中进行人群密度图的估计，训练数据中的密度质量很大程度上决定了我们方法的性能。我们首先描述一下，怎样将标注了人头的图像转化为人群密度图。如果在像素点$x_i$处有一个人头，我们将其表示成一个delta函数$δ(x − x_i)$。所以标注了N个头的图像可以表示为

$$H(x) = \sum_{i=1}^N δ(x-x_i)$$

To convert this to a continuous density function, we may convolve this function with a Gaussian kernel[17] $G_σ$ so that the density is $F(x) = H(x) ∗ G_σ (x)$. However, such a density function assumes that these $x_i$ are independent samples in the image plane which is not the case here: In fact, each $x_i$ is a sample of the crowd density on the ground in the 3D scene and due to the perspective distortion, and the pixels associated with different samples $x_i$ correspond to areas of different sizes in the scene.

为将其转化成一个连续的密度函数，我们可以将这个函数与一个高斯核[17]$G_σ$进行卷积，所以得到的密度图就是$F(x) = H(x) ∗ G_σ (x)$。但是，这样的密度函数假设这些$x_i$在图像平面中是独立的样本，而情况不是这样的：实际上，每个$x_i$都是3D场景中地面上人群密度的一个样本，由于视角变形，不同样本$x_i$的像素对应的区域有这不同的大小。

Therefore, to accurately estimate the crowd density F, we need to take into account the distortion caused by the homography between the ground plane and the image plane. Unfortunately, for the task (and datasets) at hand, we typically do not know the geometry of the scene. Nevertheless, if we assume around each head, the crowd is somewhat evenly distributed, then the average distance between the head and its nearest k neighbors (in the image) gives a reasonable estimate of the geometric distortion (caused by the perspective effect).

所以，为准确估计人群密度F，我们需要考虑进变形的因素。不幸的是，对眼下的任务（和数据集），我们通常不知道场景的几何关系。尽管如此，如果我们假设对于每个人头，人群是某种均匀分布的，那么这个人头与其周围最近的k个邻近样本的平均距离，可以给出几何形变的合理估计（由视角原因导致）。

Therefore, we should determine the spread parameter σ based on the size of the head for each person within the image. However, in practice, it is almost impossible to accurately get the size of head due to the occlusion in many cases, and it is also difficult to find the underlying relationship between the head size the density map. Interesting we found that usually the head size is related to the distance between the centers of two neighboring persons in crowded scenes (please refer to Figure 2). As a compromise, for the density maps of those crowded scenes, we propose to data-adaptively determine the spread parameter for each person based on its average distance to its neighbors.(For the images given the density or perspective maps, we directly use the given density maps in our experiments or use the density maps generated from perspective maps. For those data only contain very few persons and the sizes of heads are similar, we use the fixed spread parameter for all the persons.)

所以，我们应当在图像中每个人的人头的大小的基础上，确定延展参数σ。但是，在实践中，几乎不可能准确的得到人头的大小，因为很多情况下都有遮挡的问题，也很难找到人头大小与密度图的潜在关系。有趣的是，我们发现，在拥挤的场景中，人头大小通常都与相邻的两个人之间的距离有关（请参见图2）。作为折中，对于拥挤场景的密度图，我们提出根据数据自适应的确定每个人的延展参数，主要是依据与其邻域人头的平均距离。（对于给定密度或视角图的图像，我们在实验中直接使用给定的密度图，或使用从视角图中生成的密度图。因为这些数据只包含很少几个人，人头的大小也很相似，对于所有人我们都使用固定的延展参数）

For each head $x_i$ in a given image, we denote the distances to its m nearest neighbors as $\{ d_1^i, d_2^i, ... , d_m^i \}$. The average distance is therefore $\bar d^i = \frac {1}{m} \sum_{j=1}^m d_j^i$. Thus, the pixel associated with $x_i$ corresponds to an area on the ground in the scene roughly of a radius proportional to $\bar d^i$. Therefore, to estimate the crowd density around the pixel $x_i$, we need to convolve $δ(x − x_i)$ with a Gaussian kernel with variance $σ_i$ proportional to $\bar d^i$. More precisely, the density F should be

对于给定图像的每个人头$x_i$，与周围m个最近邻域人头的距离表示为$\{ d_1^i, d_2^i, ... , d_m^i \}$，平均距离因此是$\bar d^i = \frac {1}{m} \sum_{j=1}^m d_j^i$。因此，与$x_i$相关的像素，对应着场景中地面上的一个区域，其半径大致正比于$\bar d^i$。因此，为估计像素$x_i$周围的人群密度，我们需要将$δ(x − x_i)$与高斯核卷积，其方差$σ_i$正比于$\bar d^i$。更精确的，密度图F应当是

$$F(x) = \sum_{i=1}^N δ(x-x_i)*G_{σ_i}(x), with σ_i = β \bar d^i$$

for some parameter β. In other words, we convolve the labels H with density kernels adaptive to the local geometry around each data point, referred to as geometry-adaptive kernels. In our experiment, we have found empirically β = 0.3 gives the best result. In Figure 2, we have shown so-obtained density maps of two exemplar images in our dataset.

换句话说，我们将标注H和密度核进行卷积，卷积核自适应于每个数据点周围的局部几何特征，称为几何自适应的卷积核。在我们的实验中，我们通过经验发现β = 0.3给出最佳的结果。在图2中，我们展示了我们数据集中的两幅图像样本，通过这样得到的密度图。

Figure 2: Original images and corresponding crowd density maps obtained by convolving geometry-adaptive Gaussian kernels.

### 2.3. Multi-column CNN for density map estimation

Due to perspective distortion, the images usually contain heads of very different sizes, hence filters with receptive fields of the same size are unlikely to capture characteristics of crowd density at different scales. Therefore, it is more natural to use filters with different sizes of local receptive field to learn the map from the raw pixels to the density maps. Motivated by the success of Multi-column Deep Neural Networks (MDNNs) [8], we propose to use a Multi-column CNN (MCNN) to learn the target density maps. In our MCNN, for each column, we use the filters of different sizes to model the density maps corresponding to heads of different scales. For instance, filters with larger receptive fields are more useful for modeling the density maps corresponding to larger heads.

由于视角变形，图像中通常包含不同大小的人头，所以固定感受野大小的滤波器不能捕捉到不同尺度的人群密度特征。所以，很自然的可以使用不同感受野大小的滤波器，来从原始像素中学习得到密度图。受到[8]的多列深度神经网络(MDNNs)成功的激励，我们提出使用多列CNN(MCNN)来学习目标的密度图。在我们的MCNN中，我们在每列中都使用不同大小的滤波器，以对不同大小人头的密度图建模。比如，较大感受野的滤波器，对于更大头部的密度图建模更有用处。

The overall structure of our MCNN is illustrated in Figure 3. It contains three parallel CNNs whose filters are with local receptive fields of different sizes. For simplification, we use the same network structures for all columns (i.e., conv–pooling–conv–pooling) except for the sizes and numbers of filters. Max pooling is applied for each 2 × 2 region, and Rectified linear unit (ReLU) is adopted as the activation function because of its good performance for CNNs [32]. To reduce the computational complexity (the number of parameters to be optimized), we use less number of filters for CNNs with larger filters. We stack the output feature maps of all CNNs and map them to a density map. To map the features maps to the density map, we adopt filters whose sizes are 1 × 1 [21]. Then Euclidean distance is used to measure the difference between the estimated density map and ground truth. The loss function is defined as follows:

我们的MCNN整体结构如图3所示，包括三个平行的CNN，其滤波器的局部感受野大小不同。简化起见，我们对所有列使用相同的网络结构，即，卷积-池化-卷积-池化，不一样的地方只有滤波器的大小和数量。对每个2×2区域进行最大池化，使用ReLU作为激活函数。为降低计算复杂度（需要优化的参数数量），我们使用较少数量的滤波器，但尺寸更大一些。我们将所有CNNs的输出并列在一起，映射到一个密度图上。为将特征图映射到密度图，我们采用大小为1×1的滤波器[21]，然后使用欧几里得距离来度量估计的密度图和真值密度图之间的差异。损失函数定义如下：

$$L(Θ) = \frac {1}{2N} \sum_{i=1}^N ||F(X_i; Θ) - F_i||_2^2$$(1)

where Θ is a set of learnable parameters in the MCNN. N is the number of training image. $X_i$ is the input image and $F_i$ is the ground truth density map of image $X_i$. $F (X_i ; Θ)$ stands for the estimated density map generated by MCNN which is parameterized with Θ for sample $X_i$. L is the loss between estimated density map and the ground truth density map.

其中Θ是MCNN的可学习的参数集合。N是训练图像的数量。$X_i$是输入图像，$F_i$是图像$X_i$的真值密度图。$F (X_i ; Θ)$代表的是MCNN生成的样本$X_i$的估计密度图，其参数为Θ。L是估计密度图与真值密度图之间的损失。

**Remarks** i) Since we use two layers of max pooling, the spatial resolution is reduced by 1/4 for each image. So in the training stage, we also down-sample each training sample by 1/4 before generating its density map. ii) Conventional CNNs usually normalize their input images to the same size. Here we prefer the input images to be of their original sizes because resizing images to the same size will introduce additional distortion in the density map that is difficult to estimate. iii) Besides the fact that the filters have different sizes in our CNNs, another difference between our MCNN and conventional MDNNs is that we combine the outputs of all CNNs with learnable weights (i.e.,1×1 filters). In contrast, in MDNNs proposed by [8], the outputs are simply averaged.

**注释** i)由于我们使用了两层最大池化，每幅图像的分辨率变成了原图的1/4，所以在训练阶段，我们也将训练样本降采样为1/4并生成其密度图； ii)传统CNNs通常将输入图像归一化为一样的大小。这里我们保留输入图像的原始大小，因为改变图像到相同的大小会带来密度图的额外变形，更加难以估计； iii)我们的CNNs与传统MDNN相比，除了滤波器大小不同，另一个不同是，我们将所有CNNs的输入结合起来，其权重可以学习（即，1×1滤波器）。相比之下，[8]中提出的MDNNs，其输出只是进行了简单的平均。

### 2.4. Optimization of MCNN 优化

The loss function (1) can be optimized via batch-based stochastic gradient descent and backpropagation, typical for training neural networks. However, in reality, as the number of training samples are very limited, and the effect of gradient vanishing for deep neural networks, it is not easy to learn all the parameters simultaneously. Motivated by the success of pre-training of RBM [11], we pre-train CNN in each single column separately by directly mapping the outputs of the fourth convolutional layer to the density map. We then use these pre-trained CNNs to initialize CNNs in all columns and fine-tune all the parameters simultaneously.

损失函数(1)可以通过基于批次的随机梯度下降和反向传播优化，一般训练神经网络都是这样。但是，在实际中，由于训练样本非常有限，以及深度神经网络的梯度消失效应，不可能同时学习所有的参数。受到RBM[11]预训练的成功的启发，我们在每个单列中分别预训练CNN，直接将第4个卷积层映射到密度图。然后我们使用这些预训练的CNNs来初始化所有列的CNNs，然后同时精调所有参数。

### 2.5. Transfer learning setting 迁移学习设置

One advantage of such a MCNN model for density estimation is that the filters are learned to model the density maps of heads with different sizes. Thus if the model is trained on a large dataset which contains heads of very different sizes, then the model can be easily adapted (or transferred) to another dataset whose crowd heads are of some particular sizes. If the target domain only contains a few training samples, we may simply fix the first several layers in each column in our MCNN, and only fine-tune the last few convolutional layers. There are two advantages for fine-tuning the last few layers in this case. Firstly, by fixing the first several layers, the knowledge learnt in the source domain can be preserved, and by fine-tuning the last few layers, the models can be adapted to the target domain. So the knowledge in both source domain and target domain can be integrated and help improve the accuracy. Secondly, comparing with fine-tuning the whole network, fine-tuning the last few layers greatly reduces the computational complexity.

MCNN模型进行密度估计的一个优点是学习的滤波器可以估计不同大小的头部的密度图。所以如果模型在很大的数据集上训练，包含不同大小的头部，那么模型可以轻松的迁移到别的头部大小类似的数据集上。如果目标领域只包含一些训练样本，我们可以直接固定MCNN每一列的前几层，只精调最后几个卷积层。这种情况下，只精调最后几层有两个优势。首先，固定前面几层，保留了在源领域学习的知识，通过精调最后几层，模型可以适用于目标领域。所以，源领域的知识和目标领域的知识可以结合在一起，来改进计数准确率。第二，如果对整个网络进行精调，计算复杂度会非常高，只精调最后几层极大的减少了计算量。

## 3. Experiments 实验

We evaluate our MCNN model on four different datasets – three existing datasets and our own dataset. Although comparing to most DNN based methods in the literature, the proposed MCNN model is not particularly deep nor sophisticated, it has nevertheless achieved competitive and often superior performance in all the datasets. In the end, we also demonstrate the generalizability of such a simple model in the transfer learning setting (as mentioned in section 2.5). Implementation of the proposed network and its training are based on the Caffe framework developed by [14].

我们在四个不同的数据集上评估我们的MCNN模型，3个现有的数据集和我们自制的数据集。与现有的基于DNN的方法相比，我们提出的MCNN模型不是特别深或特别复杂，但是却在所有数据集上取得了非常有竞争力，而且经常是最好的结果。最后，我们也证明了这样一个简单模型在迁移学习设置中的泛化能力（如2.5节所述）。网络及其训练是基于Caffe框架[14]实现的。

### 3.1. Evaluation metric 评估标准

By following the convention of existing works [28, 33] for crowd counting, we evaluate different methods with both the absolute error (MAE) and the mean squared error (MSE), which are defined as follows: 和已有的人群计数的工作[28,33]一样，我们用MAE和MSE评估不同的方法，定义如下：

$$MAE = \frac {1}{N} \sum_{i=1}^N |z_i - \hat z_i|, MSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (z_i - \hat z_i)^2}$$(2)

where N is the number of test images, $z_i$ is the actual number of people in the i-th image, and $\hat z_i$ is the estimated number of people in the i-th image. Roughly speaking, MAE
indicates the accuracy of the estimates, and MSE indicates the robustness of the estimates. 其中N是测试图像的数量，$z_i$是第i幅图像的真实人数，$\hat z_i$是第i幅图像的估计人数。大致说来，MAE是估计的准确度，MSE是估计的稳健度。

### 3.2. Shanghaitech dataset

As exiting datasets are not entirely suitable for evaluation of the crowd count task considered in this work, we introduce a new large-scale crowd counting dataset named Shanghaitech which contains 1198 annotated images, with a total of 330,165 people with centers of their heads annotated. As far as we know, this dataset is the largest one in terms of the number of annotated people. This dataset consists of two parts: there are 482 images in Part A which are randomly crawled from the Internet, and 716 images in Part B which are taken from the busy streets of metropolitan areas in Shanghai. The crowd density varies significantly between the two subsets, making accurate estimation of the crowd more challenging than most existing datasets. Both Part A and Part B are divided into training and testing: 300 images of Part A are used for training and the remaining 182 images for testing;, and 400 images of Part B are for training and 316 for testing. Table1 gives the statistics of Shanghaitech dataset and its comparison with other datasets. We also give the crowd histograms of images in this dataset in Figure 4. If the work is accepted for publication, we will release the dataset, the annotations, as well as the training/testing protocol.

由于现有的数据集并不完全适合评估本文进行的人群计数任务，我们提出了一个新的大规模人群计数数据集，名为Shanghaitech，包含1198幅标注的图像，共计330165个人，标注了其人头的中心点。据我们所知，这个数据集在标注的人的数量上来说，是最大的数据集。这个数据集包括两部分：Part A包含482幅图像，是从互联网上随机爬取得到的，Part B包含716幅图像，是从上海市区最繁忙的街道上拍的。这两个子集中的人群密度很不一样，这使得准确的估计人群计数比多数现有的数据集更有挑战性。Part A和Part B都分成了训练集和测试集：Part A的300幅图像用于训练，剩余182幅图像用于测试；Part B的400幅图像用于训练，剩下的316幅图像用于测试。表1给出了Shanghaitech数据集的统计值，以及与其他数据集的对比。我们还在图4中给出了人群数量的直方图。我们会公开数据集、标注，以及训练、测试协议。

Table 1: Comparation of Shanghaitech dataset with existing datasets: Num is the number of images; Max is the maximal crowd count; Min is the minimal crowd count; Ave is the average crowd count; Total is total number of labeled people.

Dataset | Resolution | Num | Max | Min | Ave | Total
--- | --- | --- | --- | --- | --- | ---
UCSD | 158×238 | 2000 | 46 | 11 | 24.9 | 49885
UCF_CC_50 | different | 50 | 4543 | 94 | 1279.5 | 63974
WorldExpo | 576×720 | 3980 | 253 | 1 | 50.2 | 199923
Shanghaitech part_A | different | 482 | 3139 | 33 | 501.4 | 241677
Shanghaitech part_B | 768×1024 | 716 | 578 | 9 | 123.6 | 88488

To augment the training set for training the MCNN, we cropped 9 patches from each image at different locations, and each patch is 1/4 size of the original image. All the patches are used to train our MCNN model. For Part A, as the crowd density is usually very high, we use our geometry-adaptive kernels to generate the density maps, and the predicted density at overlapping region is calculated by averaging. For Part B, since the crowd is relatively sparse, we use the same spread in Gaussian kernel to generate the (ground truth) density maps. In our implementation, we first pre-train each column of MCNN independently. Then we fine-tune the whole network. Figure 5 shows examples of ground truth density maps and estimated density maps of images in Part A.

为扩充MCNN训练的训练集，我们从每幅图像中在不同位置剪切出9个图像块，每个图像块都是原图像的1/4大小。所有图像块都用于训练我们的MCNN模型。对于Part A，由于人群密度通常都非常高，我们使用几何自适应的核来生成密度图，在重叠区域预测的密度通过平均进行计算。对于Part B，由于人群相对稀疏，我们在高斯核中使用相同的扩展系数来生成真值密度图。在我们的实现中，我们首先单独预训练每列MCNN；然后精调整个网络。图5是Part A中图像的真值密度图和预测的密度图。

We compare our method with the work of Zhang et al. [33], which also uses CNNs for crowd counting and achieved state-of-the-art accuracy at the time. Following the work of [33], we also compare our work with regression based method, which uses Local Binary Pattern (LBP) features extracted from the original image as input and uses ridge regression (RR) to predict the crowd number for each image. To extract LBP features, each image is uniformly divided into 8 × 8 blocks in Part A and 12 × 16 blocks in Part B, then a 59-dimensional uniform LBP in each block is extracted and all uniform LBP features are concatenated together to represent the image. The ground truth is a 64D or 192D vector where each entry is the total number of persons in corresponding patch. We compare the performances of all the methods on Shanghaitech dataset in Table 2.

我们将我们的方法与Zhang等[33]的工作进行了比较，他们也使用了CNNs进行人群计数，并得到了当时最好的结果。与[33]一样，我们也与基于回归的方法进行了比较，即使用了从原图提取的LBP特征作为输入，并使用脊回归(RR)来预测每幅图像中的人群计数。为提取LBP特征，Part A中的每幅图像统一分割成8×8的小块，Part B中的分成12×16的小块，然后每个小块提取出59维的LBP特征，拼接到一起，来代表整个图像的特征。真值是64D或192D矢量，每个entry是对应小块的人数。表2中给出了所有方法在Shanghaitech数据集上的性能对比。

Table 2: Comparing performances of different methods on Shanghaitech dataset.

Method | partA MAE | partA MSE | partB MAE | partB MSE
--- | --- | --- | --- | ---
LBP+RR | 303.2 | 371.0 | 59.1 | 81.7
Zhang et al. [33] | 181.8 | 277.7 | 32.0 | 49.8
MCNN-CCR | 245.0 | 336.1 | 70.9 | 95.9
MCNN | 110.2 | 173.2 | 26.4 | 41.3

**The effect of pretraining in MCNN**. We show the effect of our model without pretraining on Shanghaitech dataset Part A in Figure 6. We see that pretrained network outperforms the network without pretraining. The result verifies the necessity of pretraining for MCNN as optimization starting from random initialization tends to fall into local minima.

**MCNN预训练的效果**。图6中我们给出了我们的模型没有经过预训练在Shanghaitech数据集上的效果。我们可以看到预训练网络的性能超过了没有预训练的网络。结果确认了MCNN预训练的必要性，因为随机初始化的优化过程可能会陷入局部极小值。

**Single column CNNs vs MCNN**. Figure 6 shows the comparison of single column CNNs with MCNN on Shanghaitech dataset Part A. It can be seen that MCNNs significantly outperforms each single column CNN for both MAE and MSE. This verifies the effectiveness of the MCNN architecture.

**单列CNNs vs MCNN**。图6给出了单列CNNs和MCNN在Shanghaitech数据集Part A部分的比较。可以看出，MCNN在MAE和MSE方面明显超过了每个单列CNN。这确认了我们MCNN架构的有效性。

**Comparison of different loss functions**. We evaluate the performance of our framework with different loss functions. Other than mapping the images to their density maps, we can also map the images to the total head counts in the image directly. For the input image $X_i (i = 1, . . . , N)$, its total head count is $z_i$, and $F(X_i; Θ)$ stands for the estimated density map and Θ is the parameters of MCNN. Then we arrive the following objective function:

**不同损失函数的比较**。我们用不同损失函数来评估我们框架的性能。除了将图像映射到其密度图之外，我们还可以将图像直接映射到图像中的人头总数。对于输入图像$X_i (i = 1, . . . , N)$，其人头总数为$z_i$，$F(X_i; Θ)$代表估计的密度图，Θ是MCNN的参数。然后我们得到下面的目标函数：

$$L(Θ) = \frac {1}{2N} \sum_{i=1}^N || \int_S F(X_i; Θ)dxdy - z_i ||^2$$(3)

Here S stands for the spatial region of estimated density map, and ground truth of the density map is not used. For this loss, we also pretrain CNNs in each column separately. We call such a baseline as MCNN based crowd count regression (MCNN-CCR). Performance based on such loss function is listed in Table 2, which is also compared with two existing methods as well as the method based on density map estimation (simply labeled as MCNN). We see that the results based on crowd count regression is rather poor. In a way, learning density map manages to preserve more information of the image, and subsequently helps improve the count accuracy.

这里S代表密度图估计的空间区域，密度图真值没有使用。对于这个损失函数，我们也对每列进行单独的预训练CNNs。我们称这个基准为基于MCNN的人群计数回归(MCNN-CCR)。基于这个损失函数的性能如表2所示，与两种已有的方法进行了比较，也包括基于密度图估计的方法（记为MCNN）。我们看到，人群计数回归的结果比较差。学习密度图保留了图像的更多信息，然后帮助改进了计数准确率。

In Figure 7, we compare the results of our method with those of Zhang et al. [33] in more details. We group the test images in Part A and Part B into 10 groups according to crowd counts in an increasing order. We have 182+316 test images in Part A and Part B. Except for the 10th group which contains 20+37 images, other groups all have 18+31 images each. From the plots in the figure, we can see that our method is much more accurate and robust to large variation in crowd number/density.

在图7中，我们将我们的方法与Zhang等[33]的结果更细节的进行了比较。我们将Part A和Part B中的测试图像，按照人群计数增加的顺序分成了10组。我们在Part A和Part B中有182+316幅测试图像。除了第10组包括20+37幅图像，其他组都是18+31幅图像。从图中，我们看到我们的方法更准确，而且对人群数量/密度上大的变化非常稳健。

### 3.3. The UCF CC 50 dataset

The UCF CC 50 dataset is firstly introduced by H. Idrees et al. [12]. This dataset contains 50 images from the Internet. It is a very challenging dataset, because of not only limited number of images, but also the crowd count of the image changes dramatically. The head counts range between 94 and 4543 with an average of 1280 individuals per image. The authors provided 63974 annotations in total for these fifty images. We perform 5-fold cross-validation by following the standard setting in [12]. The same data augmentation approach as in that in Shanghaitech dataset.

UCF CC 50数据集由H. Idrees等[12]首先提出。这个数据集包含网络得到的50幅图像，是一个非常有挑战的数据集，因为图像数量有限，而且图像中的人群计数变化剧烈。人头数量在94和4543之间，平均每幅图像1280个个体，共计标注了63794个标注。我们按照[12]中的方法进行5部交叉验证。数据扩充的方式和在Shanghaitech数据集上使用的一样。

We compare our method with four existing methods on UCF CC 50 dataset in Table 3. Rodriguez et al. [26] employs density map estimation to obtain better head detection results in crowd scenes. Lempitsky et al. [17] adopts dense SIFT features on randomly selected patches and the MESA distance to learn a density regression model. The method presented in [12] gets the crowd count estimation by using multi-source features. The work of Zhang et al. [33] is based on crowd CNN model to estimate the crowd count of an image. Our method achieves the best MAE, and comparable MSE with existing methods.

我们将我们的方法与四种已有的方法在UCF CC 50数据集上进行了比较，如表3所示。Rodriguez等[26]采用了密度图估计的方法，得到了人群场景中更好的人头检测结果。Lempitsky等[17]采用随机选择的图像块上的密集SIFT特征和MESA距离来学习密度回归模型。[12]中提出的方法使用多源特征得到人群计数估计。Zhang等[33]的工作是基于crowd CNN模型来估计图像中的人群计数。我们的方法在MAE的结果上最好，MSE上与现有的结果类似。

Table 3: Comparing results of different methods on the UCF CC 50 dataset.

Method | MAE | MSE
--- | --- | ---
Rodriguez et al. [26] | 655.7 | 697.8
Lempitsky et al. [17] | 493.4 | 487.1
Idrees et al. [12] | 419.5 | 541.6
Zhang et al. [33] | 467.0 | 498.5
MCNN | 377.6 | 509.1

### 3.4. The UCSD dataset

We also evaluate our method on the UCSD dataset [4]. This dataset contains 2000 frames chosen from one surveillance camera in the UCSD campus. The frame size is 158 × 238 and it is recoded at 10 fps. There are only about 25 persons on average in each frame (Please refer to Table 1) The dataset provides the ROI for each video frame.

我们还在UCSD数据集[4]上评估了我们的方法。这个数据集包括2000帧从UCSD校园监控摄像头中选出的图像。帧大小为158 × 238，重新编码为10fps。平均每帧有25人，数据集对每个视频帧都提供了ROI。

By following the same setting with [4], we use frames from 601 to 1400 as training data, and the remaining 1200 frames are used as test data. This dataset does not satisfy assumptions that the crowd is evenly distributed. So we fix the σ of the density map. The intensities of pixels out of ROI is set to zero, and we also use ROI to revise the last convolution layer. Table 4 shows the results of our method and other methods on this dataset. The proposed MCNN model outperforms both the foreground segmentation based methods and CNN based method [33]. This indicates that our model can estimate not only images with extremely dense crowds but also images with relative sparse people.

采用[4]中相同的设置，我们用601-1400帧作为训练数据，剩余的1200帧作为测试数据。这个数据集没有满足人群均匀分布的假设。所以我们固定密度图的σ。ROI以外像素的亮度设为0，我们也使用ROI来修改最后一个卷积层。表4的结果表明，我们的模型不仅能预测非常密集人群的人数，也能估计相对稀少的人数。

Table 4: Comparing results of different methods on the UCSD dataset.

Method | MAE | MSE
--- | --- | ---
Kernel Ridge Regression [1] | 2.16 | 7.45
Ridge Regression [7] | 2.25 | 7.82
Gaussian Process Regression [4] | 2.24 | 7.97
Cumulative Attribute Regression [6] | 2.07 | 6.86
Zhang et al. [33] | 1.60 | 3.31
MCNN | 1.07 | 1.35

### 3.5. The WorldExpo’10 dataset

WorldExpo’10 crowd counting dataset was firstly introduced by Zhang et al. [33]. This dataset contains 1132 annotated video sequences which are captured by 108 surveillance cameras, all from Shanghai 2010 WorldExpo. The authors of [33] provided a total of 199,923 annotated pedestrians at the centers of their heads in 3980 frames. 3380 frames are used in training data. Testing dataset includes five different video sequences, and each video sequence contains 120 labeled frames. Five different regions of interest (ROI) are provided for the test scenes.

WorldExpo'10人群计数数据集由Zhang等[33]首先提出。这个数据集包括108个监控摄像头捕捉到的1132个标注了的视频序列，都是从上海2012世博会上拍摄的。[33]的作者共标注了3980帧中的199923个行人，标注在其头部中央。3380帧用作训练数据，测试数据集包括5个不同的视频序列，每个视频序列包含120帧标注数据。测试场景中，给出了5个不同的ROI。

In this dataset, the perspective maps are given. For fair comparison, we followed the work of [33], generated the density map according to perspective map with the relation σ = 0.2 ∗ M (x), M (x) denotes that the number of pixels in the image representing one square meter at that location. To be consistent with [33], only ROI regions are considered in each test scene. So we modify the last convolution layer based on the ROI mask, namely, setting the neuron corresponding to the area out of ROI to zero. We use the same evaluation metric (MAE) suggested by the author of [33]. Table 5 reports the results of different methods in the five test video sequences. Our method also achieves better performance than Fine-tuned Crowd CNN model [33] in terms of average MAE.

在这个数据集中给出了视角图。为进行公平比较，我们按照[33]的方法，根据视角图生成了密度图，其关系为σ = 0.2 ∗ M (x)，其中M(x)代表图像位置中代表一平方米的像素数。为与[33]一致，只考虑了测试场景中的ROI区域。所以我们根据ROI掩膜修改了最后一个卷积层，即，设ROI区域外的神经元为0。我们的方法也比精调的Crowd CNN模型[33]取得了更好的效果。

Table 5: Mean absolute errors of the WorldExpo’10 crowd counting dataset.

Method | Scene 1 | Scene 2 | Scene 3 | Scene 4 | Scene 5 | Average
--- | --- | --- | --- | --- | --- | ---
LBP + RR | 13.6 | 59.8 | 37.1 | 21.8 | 23.4 | 31.0
Zhang et al. [33] | 9.8 | 14.1 | 14.3 | 22.2 | 3.7 | 12.9
MCNN | 3.4 | 20.6 | 12.9 | 13.0 | 8.1 | 11.6

### 3.6. Evaluation on transfer learning 迁移学习的评估

To demonstrate the generalizability of the learned model in our method, we test our method in the transfer learning setting by using the Part A of Shanghaitech dataset as the source domain and using the UCF CC 50 dataset as the target domain. Specifically, we train a MCNNs model with data in the source domain. For the crowd counting task in the target domain, we conduct two settings, i.e., (i) no training samples in the target domain, and (ii) There are only a few samples in the target domain. For case (i), we directly use our model trained on Part A of Shanghaitech dataset for evaluation. For case (ii), we use the training samples in the target domain to fine-tune the network. The performance of different settings is reported in Table 6. The accuracy differences between models trained on UCF CC 50 and Part A are similar (377.7 vs 397.7), which means the model trained on Part A is already good enough for the task on UCF CC 50. By fine-tuning the last two layers of MCNN with training data on UCF CC 50, the accuracy can be greatly boosted (377.7 vs. 295.1). However, if the whole network is fine-tuned rather than only the last two layers, the performance drops significantly (295.1 vs 378.3), but still comparable (377.7 vs 378.31) with the MCNN model trained with the training data of the target domain. The performance gap between fine-tuning the whole network and fine-tuning the last couple of layers is perhaps due to the reason that we have limited training samples in the UCF CC 50 dataset. Fine-tuning the last two layers ensures that the output of the model is adapted to the target domain, and keeping the first few layers of the model in tact ensures that good features/filters learned from adequate data in the source domain will be preserved. But if the whole network is fine-tuned with inadequate data in the target domain, the learned model becomes similar to that learned with only the training data in the target domain. Hence the performance degrades to that of the model learned in the latter case.

为展示我们方法学习的模型的可泛化能力，我们对方法进行迁移学习测试，使用Shanghaitech数据集的Part A作为源领域，使用UCF CC 50数据集作为目标领域。具体来说，我们在源领域训练一个MCNN模型。对于目标领域的人群计数任务，我们进行两种设置，即(i)没有标领域的训练样本，(ii)只有目标领域的少部分样本。对于情况(i)，我们直接使用在Shanghaitech数据集Part A上训练的模型进行评估。对于情况(ii)，我们使用目标领域的训练样本来精调网络。表6给出了不同设置的性能结果。用UCF CC 50数据集训练的模型和用Part A数据训练的模型准确度结果类似(377.7 vs 397.7)，这意味着在Part A上训练的模型已经可以用于UCF CC 50上的任务。通过使用UCF CC 50的数据精调MCNN的最后两层，准确率可以得到很大的提升(377.7 vs 295.1)。但是，如果整个网络都进行精调，性能反而会显著下降(295.1 vs 378.3)，但仍然与在UCF CC 50数据集上训练的模型可以比较(377.7 vs 378.31)。精调整个网络和精调网络最后几层的区别，可能是由于，UCF CC 50数据集上的训练样本很有限。精调最后两层，可以保证输出的模型适应了目标领域，而且保留了开始几层的模型，确保了从充足数据上学习得到的很好的特征提取器/滤波器。但如果整个网络都用不充分的目标领域数据进行精调，学习到的模型与只用目标领域数据训练的模型就很类似了。所以性能下降到后面这种模型的性能。

Table 6: Transfer learning across datasets. “MCNN w/o transfer” means we train the MCNN using the training data in UCF CC 50 only, and data from the source domain are not used. “MCNN trained on Part A” means we do not use the training data in the target domain to fine-tune the MCNN trained in the source domain.

Method | MAE | MSE
--- | --- | ---
MCNN w/o transfer | 377.7 | 509.1
MCNN trained on Part A | 397.7 | 624.1
Finetune the whole MCNN | 378.3 | 594.6
Finetune the last two layers | 295.1 | 490.23

## 4. Conclusion 结论

In this paper, we have proposed a Multi-column Convolution Neural Network which can estimate crowd number accurately in a single image from almost any perspective. To better evaluate performances of crowd counting methods under practical conditions, we have collected and labelled a new dataset named Shanghaitech which consists of two parts with a total of 330,165 people annotated. This is the largest dataset so far in terms of the annotated heads for crowd counting. Our model outperforms the state-of-art crowd counting methods on all datasets used for evaluation. Further, our model trained on a source domain can be easily transferred to a target domain by fine-tuning only the last few layers of the trained model, which demonstrates good generalizability of the proposed model.

本文中，我们提出了一种多列卷积神经网络，可以从任意视角的单幅图像中准确的估计人群数量。为更好的评估实际情况中的人群计数方法性能，我们收集并标注了一个新的数据集，名为Shanghaitech数据集，包括两部分，共标注了330165个人头。这是目前人群计数标注人头数最大的数据集。我们的模型在所有数据集上都超过了目前最好的人群计数方法。而且，我们在源领域训练的模型可以很容易的迁移到目标领域，只需要精调网络的最后几层，这展示出了模型很好的泛化能力。