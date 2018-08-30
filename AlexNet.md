# ImageNet Classification with Deep Convolutional Neural Networks
# 采用深度卷积神经网络进行ImageNet数据集分类

Alex Krizhevsky et. al    University of Toronto

## Abstract 摘要

We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.

我们训练了一个大型的深度卷积神经网络来对ImageNet LSVRC-2010比赛的120万高分辨率图像分成1000类。在测试数据集上，我们取得了top-1和top-5错误率分别是37.5%和17%的结果，这比以前最好的结果要好很多。我们神经网络有6000万参数，650000个神经元，由5个卷积层组成，有些后面跟着max-pooling层，三个全连接层和最后一个1000通道的softmax层。为使训练速度更快，我们使用了非饱和神经元，并使用了高效的GPU运算来实现卷积操作。为减少全连接层的过拟合问题，我们使用了最近提出的正则化方法，称为dropout，证明是非常有效的。我们在ILSVRC-2012比赛中使用了本模型的一个变体，取得了top-5的错误率15.3%的结果，相比下次好的成绩是26.2%。

## 1 Introduction

Current approaches to object recognition make essential use of machine learning methods. To improve their performance, we can collect larger datasets, learn more powerful models, and use better techniques for preventing overfitting. Until recently, datasets of labeled images were relatively small — on the order of tens of thousands of images (e.g., NORB [16], Caltech-101/256 [8, 9], and CIFAR-10/100 [12]). Simple recognition tasks can be solved quite well with datasets of this size, especially if they are augmented with label-preserving transformations. For example, the current-best error rate on the MNIST digit-recognition task (<0.3%) approaches human performance [4]. But objects in realistic settings exhibit considerable variability, so to learn to recognize them it is necessary to use much larger training sets. And indeed, the shortcomings of small image datasets have been widely recognized (e.g., Pinto et al. [21]), but it has only recently become possible to collect labeled datasets with millions of images. The new larger datasets include LabelMe [23], which consists of hundreds of thousands of fully-segmented images, and ImageNet [6], which consists of over 15 million labeled high-resolution images in over 22,000 categories.

目前的目标识别方法中机器学习算法不可或缺。为改善性能，我们可以收集更大的数据集，学习出更强大的模型，使用更好的技术防止过拟合。标记的图像数据集一直都相对比较小，在数万幅图的量级上（比如NORB [16], Caltech-101/256 [8, 9], and CIFAR-10/100 [12]），在这种规模的数据集上简单的识别任务可以很好的解决，如果将图像用保留标签的变换处理过后，增大图像集则更佳。比如，目前在MNIST数字识别任务中错误率最好的结果(<0.3%)接近人类的表现[4]。但目标真实世界的目标变化很大，为了学习识别目标，有必要采用大的多的训练集。确实，小型图像数据集的缺点得到了广泛的承认（如Pinto et al. [21]），但直到最近收集带标签的百万规模的图像数据集才成为可能。这些更大型的数据集包括LabelMe[23]，包括数十万张全分割的图像，和ImageNet图像集，包括超过1500万张带标记的高分辨率图像，类别数超过22000。

To learn about thousands of objects from millions of images, we need a model with a large learning capacity. However, the immense complexity of the object recognition task means that this problem cannot be specified even by a dataset as large as ImageNet, so our model should also have lots of prior knowledge to compensate for all the data we don’t have. Convolutional neural networks (CNNs) constitute one such class of models [16, 11, 13, 18, 15, 22, 26]. Their capacity can be controlled by varying their depth and breadth, and they also make strong and mostly correct assumptions about the nature of images (namely, stationarity of statistics and locality of pixel dependencies). Thus, compared to standard feedforward neural networks with similarly-sized layers, CNNs have much fewer connections and parameters and so they are easier to train, while their theoretically-best performance is likely to be only slightly worse.

为从数百万图像中学习得到几千个目标，我们需要一个学习容量大的模型。但目标识别任务的极度复杂性意味着，这个问题不能只针对ImageNet一个数据集（即使其规模非常大），我们的模型应当有很多先验知识，以弥补那些我们没有的数据，卷积神经网络(CNNs)就包括一类这样的模型[16, 11, 13, 18, 15, 22, 26]。其容量可以通过网络深度和宽度控制，模型对图像的本质做出了很强但是大多正确的假设，即，统计平稳性，像素局部相互依赖性。所以，与规模类似的标准前馈神经网络相比，CNN的连接和参数非常少，所以训练起来更容易，同时其理论上的最好性能可能只会略微下降。

Despite the attractive qualities of CNNs, and despite the relative efficiency of their local architecture, they have still been prohibitively expensive to apply in large scale to high-resolution images. Luckily, current GPUs, paired with a highly-optimized implementation of 2D convolution, are powerful enough to facilitate the training of interestingly-large CNNs, and recent datasets such as ImageNet contain enough labeled examples to train such models without severe overfitting.

尽管CNN质量吸引人，尽管其局部结构相对高效，但在大规模高分辨率图像上应用起来的代价仍然让人负担不起。幸运的是，现在的GPU都可以高度优化的实现2D卷积，已经强大到可以便利的训练超大型CNN，而最近的数据集如ImageNet包含足够的标签样本，可以训练这些模型，不出现严重的过拟合。

The specific contributions of this paper are as follows: we trained one of the largest convolutional neural networks to date on the subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012 competitions [2] and achieved by far the best results ever reported on these datasets. We wrote a highly-optimized GPU implementation of 2D convolution and all the other operations inherent in training convolutional neural networks, which we make available publicly (http://code.google.com/p/cuda-convnet/). Our network contains a number of new and unusual features which improve its performance and reduce its training time, which are detailed in Section 3. The size of our network made overfitting a significant problem, even with 1.2 million labeled training examples, so we used several effective techniques for preventing overfitting, which are described in Section 4. Our final network contains five convolutional and three fully-connected layers, and this depth seems to be important: we found that removing any convolutional layer (each of which contains no more than 1% of the model’s parameters) resulted in inferior performance.

本文的具体贡献如下：我们在ILSVRC-2010和ILSVRC-2012比赛上，在ImageNet的子集上训练了迄今为止最大的卷积神经网络之一，得到了至今报道出来的最好结果。我们用GPU高度优化的实现了2D卷积和所有其他卷积神经网络训练中的内部运算，并已经公开代码(http://code.google.com/p/cuda-convnet/)。我们的网络包含了一些新的不寻常的特征，这使其性能改进，同时减少训练时间，这在第三部分有详述。网络的规模使得过拟合成为了一个重要问题，即使在120万标记图像训练下也是问题，所以我们使用了几个有效的技术来防止过拟合，这在第四部分有详述，我们最后得到的网络包括5个卷积层，3个全连接层，这个深度似乎是重要的：我们发现去掉任何一个卷积层（每层的参数都不超过模型总共参数的1%）都会使性能下降明显。

In the end, the network’s size is limited mainly by the amount of memory available on current GPUs and by the amount of training time that we are willing to tolerate. Our network takes between five and six days to train on two GTX 580 3GB GPUs. All of our experiments suggest that our results can be improved simply by waiting for faster GPUs and bigger datasets to become available.

在最后，我们网络的规模主要受目前GPUs的内存大小以及我们能忍受的训练时间所限制，我们的网络在2块GTX 580 3GB的GPU上要训练5到6天。我们所有的试验都说明，只要GPU速度加快，数据集规模增大，得到的结果就会得到改善。

## 2 The Dataset

ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories. The images were collected from the web and labeled by human labelers using Amazon’s Mechanical Turk crowd-sourcing tool. Starting in 2010, as part of the Pascal Visual Object Challenge, an annual competition called the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held. ILSVRC uses a subset of ImageNet with roughly 1000 images in each of 1000 categories. In all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images.

ImageNet数据库包含超过1500万标记高分辨率图像，对应大约22000个类别。图像从网络收集，使用Amazon的Mechanical Turk众包工具进行人工标记。从2010年起，作为Pascal视觉目标挑战赛的一部分，每年都会举办ILSVRC，即ImageNet大规模视觉识别挑战赛。ILSVRC使用ImageNet的子集，包括1000个类别，每类约1000幅图像。总计有大约120万训练图像，5万幅验证图像，15万测试图像。

ILSVRC-2010 is the only version of ILSVRC for which the test set labels are available, so this is the version on which we performed most of our experiments. Since we also entered our model in the ILSVRC-2012 competition, in Section 6 we report our results on this version of the dataset as well, for which test set labels are unavailable. On ImageNet, it is customary to report two error rates: top-1 and top-5, where the top-5 error rate is the fraction of test images for which the correct label is not among the five labels considered most probable by the model.

ILSVRC-2010是唯一一届测试集有标签的，所以我们在这个数据集上进行了我们的大部分试验。我们的模型参加了ILSVRC-2012比赛，在第六节中我们也给出了这个数据集版本的结果，这个数据集上是没有标签的。在ImageNet比赛中，习惯的给出两个错误率，top-1和top-5，而top-5错误率是指，模型给出的5个标签都不是图像的正确标签的错误率。

ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality. Therefore, we down-sampled the images to a fixed resolution of 256 × 256. Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then cropped out the central 256×256 patch from the resulting image. We did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. So we trained our network on the (centered) raw RGB values of the pixels.

ImageNet中的图像分辨率不是一致的，而我们的模型需要输入图像维度固定，所以，我们将图像降采样到固定分辨率256×256。给定一个矩形图像，我们首先改变图像尺寸，使其较短一边长度为256，然后从图像中截取中间的256×256块。我们将每个像素减去其训练集的均值，除此以外没有对图像进行其他的任何预处理，这样我们用像素的原始RGB值（中心化，或零均值化处理后）训练网络。

## 3 The Architecture

The architecture of our network is summarized in Figure 2. It contains eight learned layers — five convolutional and three fully-connected. Below, we describe some of the novel or unusual features of our network’s architecture. Sections 3.1-3.4 are sorted according to our estimation of their importance, with the most important first.

图2所示的是我们网络的架构，包括8个学习好的层，即5个卷积层和3个全连接层，下面我们介绍一下我们网络结构中的新颖或不寻常的特征。3.1节到3.4节内容依据重要性排列，最重要的排前面。

### 3.1 ReLU Nonlinearity ReLU非线性处理