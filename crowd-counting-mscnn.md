# Multi-Scale Convolutional Neural Networks for Crowd Counting (ICIP 2017)

Lingke Zeng et al. South China University of Technology

## Abstract 摘要

Crowd counting on static images is a challenging problem due to scale variations. Recently deep neural networks have been shown to be effective in this task. However, existing neural-networks-based methods often use the multi-column or multi-network model to extract the scale-relevant features, which is more complicated for optimization and computation wasting. To this end, we propose a novel multi-scale convolutional neural network (MSCNN) for single image crowd counting. Based on the multi-scale blobs, the network is able to generate scale-relevant features for higher crowd counting performances in a single-column architecture, which is both accuracy and cost effective for practical applications. Complemental results show that our method outperforms the state-of-the-art methods on both accuracy and robustness with far less number of parameters.

静态图像中的人群计数由于尺度变化的原因，非常有挑战性。最近深度神经网络在这个任务中非常有效。但是，现有的基于神经网络的方法经常使用多列或多网络模型来提取与尺度相关的特征，这在优化时就更加复杂，浪费了很多计算量。为此，我们提出了一种新的多尺度卷积神经网络(MSCNN)进行单幅图像的人群计数。基于多尺度模块，网络在单列架构下就可以生成与尺度相关的特征，得到更好的人群计数的性能，这在实际应用中可以得到更高的准确率，计算效率也很好。补充结果显示，我们的方法在准确度和稳健性上，都超过了现有最好的方法，而且所需参数也少的多。

**Index Terms**— Multi-scale CNN, scale-relevant architectures, crowd counting. 多尺度CNN，尺度相关架构，人群计数

## 1. Introduction 引言

Crowd counting aims to estimate the number of people in the crowded images or videos feed from surveillance cameras. Overcrowding in scenarios such as tourist attractions and public rallies can cause crowd crushes, blockages and even stampedes. It has been much significant to public safety to produce an accurate and robust crowd count estimation using computer vision techniques.

人群计数的目标是从监控摄像机给出的人群图像或视频中估计出人数。在游客聚集、公众集会这样的场景中，过度拥挤可能导致群体挤压、拥堵甚至踩踏。使用计算机视觉技术，生成准确且稳健的人群计数，这对于公共安全来说非常关键。

Existing methods of crowd counting can be generally divided into two categories: detection-based methods and regression-based methods.

人群计数的现有方法可以大致分为两类：基于检测的方法和基于回归的方法。

Detection-based methods generally assume that each person on the crowd images can be detected and located by using the given visual object detector [1, 2, 3], and obtain the counting result by accumulating each detected person. However, these methods [4, 5, 6] need huge computing resource and they are often limited by person occlusions and complex background in practical scenarios, resulting at a relatively low robustness and accuracy.

基于检测的方法一般假设，使用给定的视觉目标检测器[1,2,3]，人群图像中可以检测并定位到每个人，通过累加每个检测到的人，得到计数结果。但是这些方法[4,5,6]需要大量计算资源，在实际应用中受限于人的遮挡和复杂的背景，得到的准确率和稳健性都不太高。

Regression-based methods regress the crowd count from the image directly. Chan et al. [7] used handcraft features to translate the crowd counting task into a regression problem. Following works [8, 9] proposed more kinds of crowd-relevant features including segment-based features, structural-based features and local texture features. Lempitsky et al. [10] proposed a density-based algorithm that obtain the count by integrating the estimated density map.

基于回归的方法直接从图像回归得到人群数量。Chan等[7]使用人工设计的特征来将人群计数问题转化为回归问题。之后的工作[8,9]提出了更多的人群相关的特征包括基于分割的特征，基于结构的特征和局部纹理特征。Lempitsky等[10]提出了一种基于密度的算法，通过对估计得到的密度图进行积分来得到人群数量。

Recently, deep convolutional neural networks have been shown to be effective in crowd counting. Zhang et al. [11] proposed a convolutional neural network (CNN) to alternatively learn the crowd density and the crowd count. Wang et al. [12] directly used a CNN-based model to map the image patch to its people count value. However, these single-CNN-based algorithms are limited to extract scale-relevant features and hard to address the scale variations on crowd images. Zhang et al. [13] proposed a multi-column CNN to extract multi-scale features by columns with different kernel sizes. Boominathan et al. [14] proposed a multi-network CNN that used a deep and shallow network to improve the spatial resolution. These improved algorithms can relatively suppress the scale variations problem, but they still have two shortages:

最近，深度卷积神经网络在人群计数中也得到了应用。Zhang等[11]提出了一个卷积神经网络(CNN)交替学习人群密度和人群计数。Wang等[12]直接使用了一种基于CNN的模型来从图像块中直接得到人数。但是，这些基于单个CNN模型的算法在提取与尺度相关的特征时能力有限，很难处理人群图像中的尺度变化。Zhang等[13]提出了一种多列CNN使用不同大小的核来提取多尺度特征。Boominathan等[14]提出了一种多网络的CNN，使用了一个深度和浅层网络来改进空间分辨率。这些改进的算法可以相对的抑制尺度相关的问题，但仍然有两个缺点：

- Multi-column/network need pre-trained single-network for global optimization, which is more complicated than end-to-end training. 多列/网络需要预训练的单个网络来进行全局优化，这比端到端的训练更复杂；

- Multi-column/network introduce more parameters to consume more computing resource, which make it hard for practical application. 多列/网络需要更多的参数，消耗了更多的计算资源，对实际应用来说更难。

In this paper, we propose a multi-scale convolutional neural network (MSCNN) to extract scale-relevant features. Rather than adding more columns or networks, we only introduce a multi-scale blob with different kernel sizes similar to the naive Inception module [15]. Our approach outperforms the state-of-the-art methods on the ShanghaiTech and UCF CC 50 dataset with a small number of parameters.

在本文中，我们提出了一种多尺度卷积神经网络(MSCNN)来提取出与尺度相关的特征。我们没有增加更多的列或网络，而只是引入了一个多尺度模块，其核的大小不同，与原生Inception模块类似[15]。我们的方法在ShanghaiTech和UCF CC 50数据集上，超过了现有最好的算法，参数数量更少。

## 2. Multi-scale CNN for Crowd Counting 人群计数的多尺度CNN

Crowd images are usually consisted of various sizes of persons pixels due to perspective distortion. Single-network is hard to counter scale variations with the same sized kernels combination. In [15], a Inception module is proposed to process visual information at various scales and aggregated to the next stage. Motivated by it, we designed a multi-scale convolutional neural network (MSCNN) to learn the scale-relevant density maps from original images.

人群图像中的人所占的像素通常不同，由视角变形导致。单网络在相同大小的核时，很难处理尺度变化的情况。在[15]中，提出了一种Inception模块，来处理不同尺度下的视觉信息，聚积到下一阶段。受此启发，我们设计了一种多尺度卷积神经网络(MSCNN)来从原始图像中学习尺度相关的密度图。

### 2.1. Multi-scale Network Architecture 多尺度网络框架

An overview of MSCNN is illustrated in Figure. 1, including feature remapping, multi-scale feature extraction, and density map regression. The first convolution layer is a traditional convolutional layer with single-sized kernels to remap the image feature. Multi-Scale Blob (MSB) is a Inception-like model (as Figure. 2) to extract the scale-relevant features, which consists of multiple filters with different kernel size (including 9×9, 7×7, 5×5 and 3×3). A multi-layer perceptron (MLP) [16] convolution layer works as a pixel-wise fully connection, which has multiple 1 × 1 convolutional filters to regress the density map. Rectified linear unit (ReLU) [17] is applied after each convolution layer, which works as the activation function of previous convolutional layers except the last one. Since the value in density map is always positive, adding ReLU after last convolutional layer can enhance the density map restoration. Detailed parameter settings are listed in Table 1.

MSCNN的概览如图1所示，包括特征重映射，多尺度特征提取，和密度图回归。第一个卷积层是传统卷积层，单一核心大小，用来重新映射图像特征。多尺度模块(MSB)是一个与Inception类似的模型（如图2所示），用于提取与尺度相关的特征，包括多个不同核心大小的滤波器（包括9×9, 7×7, 5×5 and 3×3）。多层感知机(MLP)[16]卷积层进行逐像素的全卷积，有多个1×1的卷积滤波器，进行密度图的回归。在每个卷积层之后都应用ReLU[17]，作为前一个卷积层的激活函数。由于密度图中的值永远是正值，在最后一个卷积层后增加一个ReLU可以改进密度图恢复。表1详细列出了参数设置。

### 2.2. Scale-relevant Density Map 与尺度相关的密度图

Following Zhang et al. [13], we estimate the crowd density map directly from the input image. To generate a scale-relevant density map with high quality, the scale-adaptive kernel is currently the best choice. For each head annotation of the image, we represent it as a delta function $δ(x − x_i)$ and describe its distribution with a Gaussian kernel $G_σ$ so that the density map can be represented as $F(x) = H(x) ∗ G_σ(x)$ and finally accumulated to the crowd count value. If we assume that the crowd is evenly distributed on the ground plane, the average distance $\bar d_i$ between the head $x_i$ and its nearest 10 annotations can generally characterize the geometric distortion caused by perspective effect using the Eq. (1), where M is the total number of head annotations in the image and we fix β = 0.3 as [13] empirically.

我们采用Zhang等[13]的工作，直接从输入图像中估计人群密度图。为生成高质量的与尺度相关的密度图，根据尺度自适应的核是目前最好的选择。对于图像中的每个头部标注，我们用一个delta函数$δ(x − x_i)$来表示，用一个高斯核$G_σ$来描述其分布，所以其密度图可以表示为$F(x) = H(x) ∗ G_σ(x)$，最终累加成为人群计数值。如果我们假设人群在地表平面均匀分布，人头$x_i$和其最近的10个标注之间的平均距离$\bar d_i$一般可以代表视角造成的几何形变，如式(1)所示，其中M是图像中所有人头标注的数量，根据[13]的经验，我们固定β = 0.3。

$$F(x) = \sum_{i=1}^{M} δ(x-x_i) * G_{σ_i}, with σ_i = β \bar d_i$$(1)

### 2.3. Model Optimization 模型优化

The output from our model is mapped to the density map, Euclidean distance is used to measure the difference between the output feature map and the corresponding ground truth. The loss function that needs to be optimized is defined as Eq. (2), where Θ represents the parameters of the model while $F (X_i ; Θ)$ represents the output of the model. $X_i$ and $F_i$ are respectively the i-th input image and density map ground truth.

我们模型的输出被映射到密度图，使用欧几里得距离来度量输出特征图和对应的真值的差异。需要优化的损失函数如(2)式定义，其中Θ表示模型的参数，$F (X_i ; Θ)$表示模型的输出。$X_i$和$F_i$分别是第i幅输入图像和真值密度图。

$$L(Θ) = \frac {1}{2N} \sum_{i=1}^{N} ||F(X_i; Θ) - F_i||_2^2$$(2)

## 3. Experiments 实验

We evaluate our multi-scale convolutional neural network (MSCNN) for crowd counting on two different datasets, which include the ShanghaiTech and UCF CC 50 datasets. The experimental results show that our MSCNN outperforms the state-of-the-art methods on both accuracy and robustness with far less parameter. All of the convolutional neural networks are trained based on Caffe [18].

我们在两个不同的数据集上评估我们人群计数的多尺度卷积神经网络(MSCNN)算法，即ShanghaiTech和UCF CC 50数据集。试验结果表明，我们的MSCNN在准确度和稳健性上都超过了目前最好的方法，而参数数量则只有之前的非常小一部分。所有的卷积神经网络都是在Caffe[18]上训练的。

### 3.1. Evaluation Metric 评估度量标准

Following existing state-of-the-art methods [13], we use the mean absolute error (MAE), the mean squared error (MSE) and the number of neural networks parameters (PARAMS) to evaluate the performance on the testing datasets. The MAE and the MSE are defined in Eq. (3) and Eq. (4).

与目前最好的方法[13]一样，我们使用平均绝对值误差(MAE)，平均平方误差(MSE)和神经网络参数数量(PARAMS)来评估在测试集上的性能。MAE和MSE如式(3)和式(4)定义。

$$MAE = \frac {1}{N} \sum_{i=1}{N} |z_i - \hat z_i|$$(3)

$$MSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (z_i - \hat z_i)^2}$$(4)

Here N represents the total number of images in the testing datasets, $z_i$ and $ẑ_i$ are the ground truth and the estimated value respectively for the i-th image. In general, MAE, MSE and PARAMS can respectively indicate the accuracy, robustness and computation complexity of a method.

这里N代表测试集中图像的总数量，$z_i$和$ẑ_i$分别是第i幅图像的真值和估计值。一般来说，MAE、MSE和PARAMS分别可以指示一种方法的准确率、稳健性和计算复杂度。

### 3.2. The ShanghaiTech Dataset 数据集

The ShanghaiTech dataset is a large-scale crowd counting dataset introduced by [13]. It contains 1198 annotated images with a total of 330,165 persons. The dataset consists of 2 parts: Part A has 482 images crawled from the Internet and Part B has 716 images taken from the busy streets. Following [13], both of them are divided into a training set with 300 images and a testing set with the remainder.

ShanghaiTech数据集是一个大型人群计数数据集，在[13]中提出，数据集包括1198幅标注的图像，共计330165人。数据集包括两个部分：Part A是从互联网爬取的482幅图像，Part B是从繁忙的大街上拍摄的716幅图像。我们采用[13]中的方法，两部分都分成训练集和测试集，训练集300幅图像，剩下的作为测试集。

#### 3.2.1. Model Training 模型训练

To ensure a sufficient number of data for model training, we perform data augmentation by cropping 9 patches from each image and flipping them. We simply fix the 9 cropped points as top, center and bottom combining with left, center and right. Each patch is 90% of the original size.

为确保模型训练的数据充分，我们进行数据扩充，从每幅图像中截取出9个图像块，并进行翻转。我们将9个剪切点固定为顶部、中间和底部，结合左边、中间和右边的点。每个图像块是原始尺寸的90%。

In order to facilitate comparison with MCNN architecture [13], the network was designed similar to the largest column of MCNN but with MSB, of which detailed settings are described in Table 1. All convolutional kernels are initialized with Gaussian weight setting standard deviation to 0.01. As described in Sec. 2.3, we use the SGD optimization with momentum of 0.9 and weight decay as 0.0005.

为更好的与MCNN[13]架构进行比较，我们将网络设计的与MCNN最大的一列类似，除了多尺度模块MSB，网络的设置细节如表1所示。所有的卷积核权重初始化为高斯权重设置，标准差为0.01。如2.3节所述，我们使用SGD优化，动量0.9，权重衰减0.0005。

#### 3.2.2. Results 结果

We compare our method with 4 existing methods on the ShanghaiTech dataset. The LBP+RR method used LBP feature to regress the function between the counting value and the input image. Zhang et al. [11] designed a convolutional network to regress both the density map and the crowd count value from original pixels. A multi-column CNN [13] is proposed to estimate the crowd count value (MCNN-CCR) and crowd density map (MCNN).

我们将我们的方法与已有的四种方法在ShanghaiTech数据集上进行了比较。LBP+RR使用LBP特征来回归输入图像和计数值之间的函数。Zhang等[11]设计了一个卷积网络来从原始像素中对密度图和人群计数值进行回归。[13]提出了多列CNN来估计人群计数值(MCNN-CCR)和人群密度图(MCNN)。

In Table 2, the results illustrate that our approach achieves the state-of-the-art performance on the ShanghaiTech dataset. In addition, it should be emphasized that the number of our parameters is far less than other two CNN-based algorithms. MSCNN uses approximately 7× fewer parameters than the state-of-the-art method (MCNN) with higher accuracy and robustness.

在表2中，结果表明，我们的方法在ShanghaiTech数据集上取得了目前最好的性能。另外，应当强调的是，我们模型的参数数量比其他两种基于CNN的算法少了很多。MSCNN的参数数量是MCNN的1/7，而且取得了更好的准确率和稳健性。

Table. 2. Performances of methods on ShanghaiTech dataset.

Method | part_A MAE | part_A MSE | part_B MAE | part_B MSE | PARAMS
--- | --- | --- | --- | --- | ---
LBP+RR | 303.2 | 371.0 | 59.1 | 81.7 | -
MSNN-CCR[13] | 245.0 | 336.1 | 70.9 | 95.9 | -
Zhang et al.[11] | 181.8 | 277.7 | 32.0 | 49.8 | 7.1M
MCNN[13] | 110.2 | 173.2 | 26.4 | 41.3 | 19.2M
MSCNN | 83.8 | 127.4 | 17.7 | 30.2 | 2.9M

### 3.3. The UCF CC 50 Dataset

The UCF CC 50 dataset [19] contains 50 gray scale images with a total 63,974 annotated persons. The number of people range from 94 to 4543 with an average 1280 individuals per image. Following [11, 13, 14], we divide the dataset into five splits evenly so that each split contains 10 images. Then we use 5-fold cross-validation to evaluate the performance of our proposed method.

UCF CC 50数据集[19]包括50幅灰度图像，共计63974个标注的人，每幅图像中的人数从94到4543个，平均1280人。与[11,13,14]一样，我们将数据集分等成5部分，每部分包括10幅图像。然后我们使用5组交叉验证来评估我们的方法。

#### 3.3.1. Model Training 模型训练

The most challenging problem of the UCF CC 50 dataset is the limited number of images for training while the people count in the images span too large. To ensure enough number of training data, we perform a data augmentation strategy following [14] by randomly cropping 36 patches with size 225×225 from each image and flipping them as similar in Sec. 3.2.1.

UCF CC 50数据集上最有挑战性的问题是，用于训练的图像很有限，每幅图像中的人数范围太宽。为确保训练数据数量足够，我们按照[14]中的方法进行数据扩充，从每幅图像中随即剪切出36个225×225大小的图像，然后像3.2.1节中类似的进行翻转。

We train 5 models using 5 splits of training set. The MAE and the MSE are calculated after all the 5 models obtained the estimated results of the corresponding validation set. During training, the MSCNN model is initialized almost the same as the experiment on the ShanghaiTech dataset except that the learning rate is fixed to be 1e-7 to guarantee the model convergence.

我们用训练集的5个分割训练了5个模型。在5个模型在相应的验证集上得到估计结果后，计算MAE和MSE。在训练过程中，MSCNN模型初始化的方式几乎与在ShanghaiTech数据集上相同，除了学习速率固定为1e-7，以确保模型的收敛性。

#### 3.3.2. Results 结果

We compared our method on the UCF CC 50 dataset with 6 existing methods. In [20, 10, 19], handcraft features are used to regress the density map from the input image. Three CNN-based methods [11, 14, 13] proposed to used multi-column/network and perform evaluation on the UCF CC 50 dataset.

我们将我们的方法与现有的6种方法在UCF CC 50数据集上进行比较。在[20,10,19]中，使用了人工设计的特征来从输入图像中回归密度图。三种基于CNN的方法[11,14,13]使用了多列/网络来在UCF CC 50数据集上进行评估。

Table 3 illustrates that our approach also achieves the state-of-the-art performance on the UCF CC 50 dataset. Here our parameters number is approximately 5× fewer than the CrowdNet model, demonstrating that our proposed MSCNN can work more accurately and robustly.

表3的结果表明，我们的方法在UCF CC 50数据集上也给出了目前最好的结果。这里我们的参数数量只有CrowdNet模型1/5，表明我们的MSCNN模型更准确、更稳健。

Table. 3. Performances of methods on UCF CC 50 dataset.

Method | MAE | MSE | PARAMS
--- | --- | --- | ---
Rodriguez et al.[20] | 655.7 | 697.8 | -
Lempitsky et al.[10] | 493.4 | 487.1 | -
Idrees et al.[19] | 419.5 | 541.6 | -
Zhang et al.[11] | 467.0 | 498.5 | 7.1M
CrowdNet [14] | 452.5 | - | 14.8M
MCNN [13] | 377.6 | 509.1 | 19.2M
MSCNN | 363.7 | 468.4 | 2.9M

## 4. Conclusion 结论

In this paper, we proposed a multi-scale convolutional neural network (MSCNN) for crowd counting. Compared with the recent CNN-based methods, our algorithm can extract scale-relevant features from crowd images using a single column network based on the multi-scale blob (MSB). It is an end-to-end training method with no requirement for multi-column/network pre-training works. Our method can achieve more accurate and robust crowd counting performance with far less number of parameters, which make it more likely to extend to the practical application.

在本文中，我们提出了一种多尺度的卷积神经网络(MSCNN)进行人群计数。与最近的基于CNN的方法比较，我们的算法可以使用带有多尺度模块(MSB)的单列网络从人群图像中提取出尺度相关的特征。这是一种端到端的训练方法，不需要多列/网络预训练工作。我们的方法可以得到更准确更稳健的人群计数性能，而使用的参数数量更少，这更容易拓展到实际应用中。