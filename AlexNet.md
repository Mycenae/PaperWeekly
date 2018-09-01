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

The standard way to model a neuron’s output *f* as a function of its input *x* is with *f(x) = tanh(x)* or $f(x)=(1+e^{-x})^{-1}$。 In terms of training time with gradient descent, these saturating nonlinearities are much slower than the non-saturating nonlinearity *f(x) = max(0,x)*. Following Nair and Hinton [20], we refer to neurons with this nonlinearity as Rectified Linear Units (ReLUs). Deep convolutional neural networks with ReLUs train several times faster than their equivalents with tanh units. This is demonstrated in Figure 1, which shows the number of iterations required to reach 25% training error on the CIFAR-10 dataset for a particular four-layer convolutional network. This plot shows that we would not have been able to experiment with such large neural networks for this work if we had used traditional saturating neuron models.

神经元输出*f*与输入*x*的标准函数是*f(x) = tanh(x)*或$f(x)=(1+e^{-x})^{-1}$。就梯度下降方法的训练时间来说，采用这种饱和性非线性关系的与非饱和非线性关系如*f(x) = max(0,x)*相比，要慢多了。我们将带有这种非线性关系的神经元叫做ReLU单元，遵循Nair and Hinton [20]的习惯。带有ReLU的深度卷积网络训练起来比相应的带tanh单元的要快几倍，如图1所示，是在CIFAR-10的数据集上训练一个特定的4层卷积网络达到25%的错误率所需的迭代次数。

Figure 1: A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network with tanh neurons (dashed line). The learning rates for each network were chosen independently to make training as fast as possible. No regularization of any kind was employed. The magnitude of the effect demonstrated here varies with network architecture, but networks with ReLUs consistently learn several times faster than equivalents with saturating neurons.

图1. 一个采用了ReLU（实线）4层卷积神经网络在CIFAR-10数据集上训练达到25%的错误率比带有tanh神经元的相应网络（虚线）快6倍。为每个网络单独选择了学习率以使其训练速度越快越好，没采用任何正则化措施。具体的效果根据具体的网络结构不同而不同，但带有ReLU的网络一直比带有饱和性神经元的网络训练要快好几倍。

We are not the first to consider alternatives to traditional neuron models in CNNs. For example, Jarrett et al.[11] claim that the nonlinearity *f(x)* = |*tanh(x)*| works particularly well with their type of contrast normalization followed by local average pooling on the Caltech-101 dataset. However, on this dataset the primary concern is preventing overfitting, so the effect they are observing is different from the accelerated ability to fit the training set which we report when using ReLUs. Faster learning has a great influence on the performance of large models trained on large datasets.

我们不是第一个考虑替换CNN的传统神经元的。比如，Jarrett et al.[11]声称非线性关系*f(x)* = |*tanh(x)*|在Caltech-101数据集上使用他们的对比度归一化和局部平局pooling效果特别好。但是，在这个数据集上其首要考虑是防止过拟合，所以他们观察的这种效果与我们用ReLU来训练数据集的加速能力是不一样的。更加快速的学习对大型模型在大型数据集上训练影响非常大。

### 3.2 Training on Multiple GPUs 多GPU训练

A single GTX 580 GPU has only 3GB of memory, which limits the maximum size of the networks that can be trained on it. It turns out that 1.2 million training examples are enough to train networks which are too big to fit on one GPU. Therefore we spread the net across two GPUs. Current GPUs are particularly well-suited to cross-GPU parallelization, as they are able to read from and write to one another’s memory directly, without going through host machine memory. The parallelization scheme that we employ essentially puts half of the kernels (or neurons) on each GPU, with one additional trick: the GPUs communicate only in certain layers. This means that, for example, the kernels of layer 3 take input from all kernel maps in layer 2. However, kernels in layer 4 take input only from those kernel maps in layer 3 which reside on the same GPU. Choosing the pattern of connectivity is a problem for cross-validation, but this allows us to precisely tune the amount of communication until it is an acceptable fraction of the amount of computation.

单个GTX 580 GPU只有3G内存，这限制了所能训练的网络的最大规模。120万样本对于训练网络是足够的，但模型太大不能再一个GPU上训练，所以我们将网络扩展到两块GPU上。现在的GPU非常适合跨GPU并行计算，因为它们可以互相直接读取或写入数据，不需要再经过主机内存。我们采用的并行化策略就是在每个GPU上放入模型的一半神经元，还有一个技巧：GPU只在某些层进行通信，比如，第3层的神经元的输入对应所有第2层的神经元的输出，但第4层的输入对应的第3层输出的神经元都在同一个GPU上。选择连接的模式对于交叉验证来说是一个问题，但这使我们可以精确的调节通信规模，直到低于计算量的某一很小百分比。

The resultant architecture is somewhat similar to that of the “columnar” CNN employed by Cire¸ san et al. [5], except that our columns are not independent (see Figure 2). This scheme reduces our top-1 and top-5 error rates by 1.7% and 1.2%, respectively, as compared with a net with half as many kernels in each convolutional layer trained on one GPU. The two-GPU net takes slightly less time to train than the one-GPU net. ( The one-GPU net actually has the same number of kernels as the two-GPU net in the final convolutional layer. This is because most of the net’s parameters are in the first fully-connected layer, which takes the last convolutional layer as input. So to make the two nets have approximately the same number of parameters, we did not halve the size of the final convolutional layer (nor the fully-conneced layers which follow). Therefore this comparison is biased in favor of the one-GPU net, since it is bigger than “half the size” of the two-GPU
net.)

最后得到的架构与Cire¸ san et al. [5]采用的柱状CNN有些相似，但我们的各列不是相互独立的（见图2）。这个方案将top-1和top-5错误率分别降低了1.7%和1.2%，这是与另一个网络的训练结果比较的，这个网络在每个卷积层中，每个GPU都有一半的核心在进行计算，这个双GPU网络与单GPU网络训练相比，时间减少的很少。（单GPU网络实际上与双GPU网络在最后卷积层的核心数是一样的，这是因为网络多数的参数在第一个全连接层，这个全连接层以最后一个卷积层为输入，所以为了使两个网络参数数量大致相同，我们没有使最后的卷积层规模减半，后续的全连接层也是。所以这个对比是对单GPU网络有利的，因为这比一半规模的双GPU网络要大。）

Figure 2: An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities between the two GPUs. One GPU runs the layer-parts at the top of the figure while the other runs the layer-parts at the bottom. The GPUs communicate only at certain layers. The network’s input is 150,528-dimensional, and the number of neurons in the network’s remaining layers is given by 253,440–186,624–64,896–64,896–43,264–4096–4096–1000.

图2. 我们的CNN的架构示意图，明确指出了两个GPU的责任划分。一个GPU负责运行图上部的层，另一个负责运行底部的层，两个GPU只在特定的层通信。网络输入是150528维，网络其他层的神经元数为253440,186624,64896,64896,43264,4096,4096,1000。

### 3.3 Local Response Normalization 局部响应归一化

ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. If at least some training examples produce a positive input to a ReLU, learning will happen in that neuron. However, we still find that the following local normalization scheme aids generalization. Denoting by $a^i_{x,y}$ the activity of a neuron computed by applying kernel *i* at position (*x,y*) and then applying the ReLU nonlinearity, the response-normalized activity $b^i_{x,y}$ is given by the expression

ReLU有个很好的性质，就是不需要将输入归一化以防止饱和。如果有几个训练样本对一个ReLU产生了正的输入，就会在那个神经元中产生学习。但是，我们还是发现下面的局部归一化方案对泛化有帮助。将核心*i*在(*x,y*)点卷积，然后进行ReLU非线性化，得到的结果记为$a^i_{x,y}$，那么归一化的响应$b^i_{x,y}$就是

$$b_{x,y}^i = \frac{a_{x,y}^i}{(k+ \alpha \sum_{j=max(0,i-n/2}^{min(N-1,i+n/2)} (a^j_{x,y})^2)^{\beta}}$$

where the sum runs over *n* “adjacent” kernel maps at the same spatial position, and *N* is the total number of kernels in the layer. The ordering of the kernel maps is of course arbitrary and determined before training begins. This sort of response normalization implements a form of lateral inhibition inspired by the type found in real neurons, creating competition for big activities amongst neuron outputs computed using different kernels. The constants *k,n,α*, and *β* are hyper-parameters whose values are determined using a validation set; we used *k* = 2, *n* = 5, *α* = $10^{−4}$ , and *β* = 0.75. We applied this normalization after applying the ReLU nonlinearity in certain layers (see Section 3.5).

这里求和是对在同一空域点的*n*个毗邻的核心图，*N*是整个层中核心的全部数量。核心图的顺序是任意的，在训练开始时就确定好。这种响应归一化应用了一种侧抑制机制，这是受真实神经元中发现的模式启发得到的，在用不同核心计算得到的神经元输出中的较大活动中造成竞争。常数*k,n,α* 和 *β*为超参数，其值由验证集确定，我们使用*k* = 2, *n* = 5, *α* = $10^{−4}$ , *β* = 0.75。我们在某些层中的ReLU非线性处理后，再进行这种归一化。（见3.5节）

This scheme bears some resemblance to the local contrast normalization scheme of Jarrett et al. [11], but ours would be more correctly termed “brightness normalization”, since we do not subtract the mean activity. Response normalization reduces our top-1 and top-5 error rates by 1.4% and 1.2%, respectively. We also verified the effectiveness of this scheme on the CIFAR-10 dataset: a four-layer CNN achieved a 13% test error rate without normalization and 11% with normalization (We cannot describe this network in detail due to space constraints, but it is specified precisely by the code and parameter files provided here: http://code.google.com/p/cuda-convnet/).

这个方案与Jarrett et al. [11]的局部对比度归一化方案有些类似，但我们的方案称为“亮度归一化”更为准确，因为我们没有减去其均值。响应归一化使我们的top-1和top-5错误率分别下降1.4%和1.2%，我们还在CIFAR-10数据集上验证了这个方案的有效性：4层CNN没有归一化时错误率13%，有归一化时11%（限于篇幅无法详细描述网络，但在代码和参数文件中描述的很具体）。

### 3.4 Overlapping Pooling

Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map. Traditionally, the neighborhoods summarized by adjacent pooling units do not overlap (e.g., [17, 11, 4]). To be more precise, a pooling layer can be thought of as consisting of a grid of pooling units spaced *s* pixels apart, each summarizing a neighborhood of size *z × z* centered at the location of the pooling unit. If we set *s = z*, we obtain traditional local pooling as commonly employed in CNNs. If we set *s < z*, we obtain overlapping pooling. This is what we use throughout our network, with *s* = 2 and *z* = 3. This scheme reduces the top-1 and top-5 error rates by 0.4% and 0.3%, respectively, as compared with the non-overlapping scheme *s* = 2,*z* = 2, which produces output of equivalent dimensions. We generally observe during training that models with overlapping pooling find it slightly more difficult to overfit.

CNN中的Pooling层将同一个核心图中的神经元的邻域内的输出进行处理。一般临近pooling单元的处理邻域不重叠，如[17, 11, 4]。更精确的说，pooling层可以认为是由pooling单元网格组成，每个单元有*s × s*个像素，对*z × z*大小的邻域进行处理。如果*s = z*，那就是CNN采用的传统pooling。如果*s < z*，我们就得到了重叠pooling。我们在整个网络中都应用了这个设置，*s* = 2, *z* = 3。这个方案使top-1和top-5错误率分别减少了0.4%和0.3%，这是与*s* = 2,*z* = 2的非重叠pooling方案比较得到的。我们发现在训练模型时，重叠pooling方案更难过拟合。

### 3.5 Overall Architecture

Now we are ready to describe the overall architecture of our CNN. As depicted in Figure 2, the net contains eight layers with weights; the first five are convolutional and the remaining three are fully-connected. The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. Our network maximizes the multinomial logistic regression objective, which is equivalent to maximizing the average across training cases of the log-probability of the correct label under the prediction distribution.

现在我们可以描述我们的CNN整体结构了。如图2所示，网络包含8层，都有参数，前5层为卷积层，剩下3个为全连接层。最后一个全连接层输出到一个1000路softmax层，最后得到1000个类别的输出标签。我们的网络最大化多项式logistic回归的目标函数，这与最大化所有训练案例的正确标签在预测分布下的log概率是等价的。

The kernels of the second, fourth, and fifth convolutional layers are connected only to those kernel maps in the previous layer which reside on the same GPU (see Figure 2). The kernels of the third convolutional layer are connected to all kernel maps in the second layer. The neurons in the fully-connected layers are connected to all neurons in the previous layer. Response-normalization layers follow the first and second convolutional layers. Max-pooling layers, of the kind described in Section 3.4, follow both response-normalization layers as well as the fifth convolutional layer. The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.

第2,4,5个卷积层的核心只与前一层在相同GPU上的核心相连（见图2），第3个卷积层的核心则与第2层所有的核心相连。全连接层与前一层所有神经元相连。响应归一化层在第1和第2卷积层后有，3.4节中提到的max-pooling层在1,2层后的响应归一化层后有，在第5层后也有。ReLU非线性处理在所有卷积层与全连接层中都有。

The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels (this is the distance between the receptive field centers of neighboring neurons in a kernel map). The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48. The third, fourth, and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers. The third convolutional layer has 384 kernels of size 3 × 3 × 256 connected to the (normalized, pooled) outputs of the second convolutional layer. The fourth convolutional layer has 384 kernels of size 3 × 3 × 192 , and the fifth convolutional layer has 256 kernels of size 3 × 3 × 192. The fully-connected layers have 4096 neurons each.

第1个卷积层对224×224×3的输入图像进行滤波，共96个核心，尺寸为11×11×3，卷积步长(stride)为4个像素（这是感受野中心与邻域神经元的距离）。第2个卷积层直接以第1个卷积层的输出（经过响应归一化和pooling处理）作为输入，滤波器有256个，尺寸5×5×48。第3，4,5个卷积层依次连接，没有pooling或归一化层；第3个卷积层有384个核，尺寸3×3×256，以第2卷积层输出后经过归一化和pooling处理作为输入；第4个卷积层有384个核，尺寸3×3×192，第5个卷积层有256个核心，尺寸3×3×192。每个全连接层有4096个神经元。

## 4 Reducing Overfitting 减少过拟合

Our neural network architecture has 60 million parameters. Although the 1000 classes of ILSVRC make each training example impose 10 bits of constraint on the mapping from image to label, this turns out to be insufficient to learn so many parameters without considerable overfitting. Below, we describe the two primary ways in which we combat overfitting.

我们的神经网络架构有6000万参数。虽然1000个类别使每个训练对象从图像到标签的映射有10个字节的约束，但这对于学习这么多参数的训练来说，还是会有相当的过拟合。下面，我们列出了两种对抗过拟合的主要方法。

### 4.1 Data Augmentation 数据膨胀

The easiest and most common method to reduce overfitting on image data is to artificially enlarge the dataset using label-preserving transformations (e.g., [25, 4, 5]). We employ two distinct forms of data augmentation, both of which allow transformed images to be produced from the original images with very little computation, so the transformed images do not need to be stored on disk. In our implementation, the transformed images are generated in Python code on the CPU while the GPU is training on the previous batch of images. So these data augmentation schemes are, in effect, computationally free.

对图像数据减少过拟合的最简单最常见的方法就是用保留标签的变换(e.g., [25, 4, 5])人工增大数据集，我们采用了两种不同的数据膨胀方法，这两种方法都使用很小的计算量从原图像进行变换，所以不需要把变换后的图像存储到磁盘上。在我们的实现中，图像变换用python写，在CPU上处理，同时GPU在处理上一批图像。所以这种数据膨胀的方法实际上是不增加运算量的。

The first form of data augmentation consists of generating image translations and horizontal reflections. We do this by extracting random 224×224 patches (and their horizontal reflections) from the 256×256 images and training our network on these extracted patches(This is the reason why the input images in Figure 2 are 224 × 224 × 3-dimensional). This increases the size of our training set by a factor of 2048, though the resulting training examples are, of course, highly interdependent. Without this scheme, our network suffers from substantial overfitting, which would have forced us to use much smaller networks. At test time, the network makes a prediction by extracting five 224 × 224 patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions made by the network’s softmax layer on the ten patches.

第一种数据膨胀方法是生成图像位移和水平翻转。我们从256×256的图像中随机截取224×224大小的图像块，然后水平翻转，在这些提取出来的图像块上训练我们的网络（这也是图2中我们的输入数据为什么是224×224×3的）。这使我们的训练集的规模膨胀了2048倍，虽然得到的训练样本是高度相关的。不用这个方案的话，我们的网络过拟合问题非常严重，这迫使我们使用小的多的网络。在测试时，网络预测是通过提取5个224×224的图像块（4个角的块和1个中心块），以及其水平翻转块，总计10块，这10块通过softmax层，然后再平均。

The second form of data augmentation consists of altering the intensities of the RGB channels in training images. Specifically, we perform PCA on the set of RGB pixel values throughout the ImageNet training set. To each training image, we add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1. Therefore to each RGB image pixel $I_{xy}=[I^R_{xy},I^G_{xy},I^B_{xy}]^T$ we add the following quantity:

第二种图像膨胀方法是改变训练图像RGB通道的亮度。具体来说，我们对整个ImageNet训练集的RGB值做一个PCA，对每个训练图像，都加上得到的主成分的倍数，其倍数与对应特征值，乘以一个零均值方差0.1的高斯分布随机值，所以对于每个RGB图像像素$I_{xy}=[I^R_{xy},I^G_{xy},I^B_{xy}]^T$我们加上下面的值：

$$[p_1,p_2,p_3][α_1λ_1 ,α_2λ_2 ,α_3λ_3]^T$$

where $p_i$ and $λ_i$ are *i*th eigenvector and eigenvalue of the 3 × 3 covariance matrix of RGB pixel values, respectively, and $α_i$ is the aforementioned random variable. Each $α_i$ is drawn only once for all the pixels of a particular training image until that image is used for training again, at which point it is re-drawn. This scheme approximately captures an important property of natural images, namely, that object identity is invariant to changes in the intensity and color of the illumination. This scheme reduces the top-1 error rate by over 1%.

这里$p_i$和$λ_i$分别是RGB值的3×3协方差阵的第*i*个特征矢量和特征值，$α_i$是前面提到的随机变量，对每个训练图像其所有像素都用同一个$α_i$，直到下次再用这个图像训练。这个方案抓住了自然图像的一个重要特征，即，目标个体对亮度、色彩的不变性。这个方案使top-1错误率降低1%。

### 4.2 Dropout

Combining the predictions of many different models is a very successful way to reduce test errors [1, 3], but it appears to be too expensive for big neural networks that already take several days to train. There is, however, a very efficient version of model combination that only costs about a factor of two during training. The recently-introduced technique, called “dropout” [10], consists of setting to zero the output of each hidden neuron with probability 0.5. The neurons which are “dropped out” in this way do not contribute to the forward pass and do not participate in backpropagation. So every time an input is presented, the neural network samples a different architecture, but all these architectures share weights. This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. At test time, we use all the neurons but multiply their outputs by 0.5, which is a reasonable approximation to taking the geometric mean of the predictive distributions produced by the exponentially-many dropout networks.

将不同模型的预测组合起来可以有效的减少测试错误[1,3]，但这对于要花好几天来训练的大型神经网络来说似乎代价太过高昂。有一种很有效的模型组合方法，训练中代价大概是两倍吧。最近提出的技术，称为dropout[10]，使每个神经元的输出以0.5的概率置零。这样被dropout的神经元对前向传播不起作用，也不参与反向传播。所以每次给定输入后，神经网络抽样出一种不同的架构，但所有这些架构共享权重。这种技术减少了神经元之间复杂的互相适应，因为一个神经元不能依靠特定其他神经元的存在。所以，必须学习更加鲁棒的特征，在与其他神经元的很多随机子集的组合中也可以有用。在测试时，我们将所有神经元的输出乘以0.5，因为dropout网络数量为指数级的，这是对其预测分布的几何均值的一个合理近似。

We use dropout in the first two fully-connected layers of Figure 2. Without dropout, our network exhibits substantial overfitting. Dropout roughly doubles the number of iterations required to converge.

我们在图2的前两个全连接层使用dropout，没有dropout的话，我们的网络表现出严重的过拟合，dropout大约将收敛需要的迭代数增加了一倍。

## 5 Details of learning

We trained our models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005. We found that this small amount of weight decay was important for the model to learn. In other words, weight decay here is not merely a regularizer: it reduces the model’s training error. The update rule for weight *w* was

我们用随机梯度下降法训练模型，batch size为128样本，momentum为0.9，weight decay为0.0005。我们发现这种很小的weight decay对于模型学习来说很重要，换句话说，weight decay不仅仅是正则化措施，它可以减少模型的训练错误。权重*w*的更新规则为

$$v_{i+1}:=0.9 \cdot v_i - 0.0005 \cdot \epsilon \cdot w_i - \epsilon \cdot <\frac{∂L}{∂w} \vert _{w_i}>_{D_i}$$
$$w_{i+1}:=w_i+v_{i+1}$$

where *i* is the iteration index, *v* is the momentum variable, $\epsilon$ is the learning rate, and $<\frac{∂L}{∂w} \vert _{w_i}>_{D_i}$ is the average over the *i*th batch $D_i$ of the derivative of the objective with respect to *w*, evaluated at $w_i$.

这里*i*是迭代次数索引，*v*是momentum变量，$\epsilon$是学习率，$<\frac{∂L}{∂w} \vert _{w_i}>_{D_i}$是目标函数*L*对*w*求导在$w_i$点的值，并在整个第*i*个batch上求平均。

We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01. We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs. We initialized the neuron biases in the remaining layers with the constant 0.

我们用零均值标准差0.01高斯分布的随机数初始化各层的权重。我们用常数1初始化第2,4,5卷积层的神经元偏置，以及全连接隐藏层。这个初始化加速了初期的学习，因为给ReLU提供了正的输入。我们在其他层以0值初始化神经元偏置。

We used an equal learning rate for all layers, which we adjusted manually throughout training. The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and reduced three times prior to termination. We trained the network for roughly 90 cycles through the training set of 1.2 million images, which took five to six days on two NVIDIA GTX 580 3GB GPUs.

我们在所有层上所用的学习速度是一样的，这是在训练过程中手动调整得到的。我们遵循的规则是以目前的学习率不能再改进验证错误率时，就把学习率除以10。学习率初始值是0.01，在结束前数值降低了3次。我们在120万幅图上大约用了90个循环训练网络，这在2块NVIDIA GTX 580 3GB GPU上用了5到6天。

## 6 Results

Our results on ILSVRC-2010 are summarized in Table 1. Our network achieves top-1 and top-5 test set error rates of 37.5% and 17.0% (The error rates without averaging predictions over ten patches as described in Section 4.1 are 39.0% and 18.3%.). The best performance achieved during the ILSVRC-2010 competition was 47.1% and 28.2% with an approach hat averages the predictions produced from six sparse-coding models trained on different features [2], and since then the best published results are 45.7% and 25.7% with an approach that averages the predictions of two classifiers trained on Fisher Vectors (FVs) computed from two types of densely-sampled features [24].

我们在ILSVRC-2010上的结果如表1所示，我们网络的top-1和top-5测试集错误率为37.5%和17.5%（像4.1节，预测结果没有在10图像块上平均得到的错误率，为39.0%和18.3%）。ILSVRC-2010上取得的最好成绩是47.1%和28.2%，其方法是将在不同特征上训练得出的6个稀疏编码模型的预测结果进行平均[2]，比赛过后的工作得到的最好结果是45.7%和25.7%，方法是用两个Fisher Vectors训练得到的分类器的预测结果平均，Fisher Vector是从两种稠密抽样的特征计算得到的[24]。

Table 1: Comparison of results on ILSVRC-2010 test set. In italics are best results achieved by others.

表1 ILSVRC-2010测试集上的结果对比，斜体是别人得到的最好结果

Model | Top-1 | Top-5
--- | --- | ---
*Sparse coding [2]* | *47.1%* | *28.2%*
*SIFT + FVs [24]* | *45.7%* | *25.7%*
CNN | 37.5% | 17.0%

We also entered our model in the ILSVRC-2012 competition and report our results in Table 2. Since the ILSVRC-2012 test set labels are not publicly available, we cannot report test error rates for all the models that we tried. In the remainder of this paragraph, we use validation and test error rates interchangeably because in our experience they do not differ by more than 0.1% (see Table 2). The CNN described in this paper achieves a top-5 error rate of 18.2%. Averaging the predictions of five similar CNNs gives an error rate of 16.4%. Training one CNN, with an extra sixth convolutional layer over the last pooling layer, to classify the entire ImageNet Fall 2011 release (15M images, 22K categories), and then “fine-tuning” it on ILSVRC-2012 gives an error rate of 16.6%. Averaging the predictions of two CNNs that were pre-trained on the entire Fall 2011 release with the aforementioned five CNNs gives an error rate of 15.3%. The second-best contest entry achieved an error rate of 26.2% with an approach that averages the predictions of several classifiers trained on FVs computed from different types of densely-sampled features [7].

我们参加了ILSVRC-2012比赛，取得的结果如表2所示。由于ILSVRC-2012测试集的标签并不是公开可用的，我们不能给出尝试的所有模型的错误率。在本段剩下的部分，我们交替使用验证错误率和测试错误率，因为以我们的经验，其差值不会超过0.1%（见表2）。本文方法取得了top-5错误率18.2%的结果。对5个类似的CNN的预测结果进行平均，得到的错误率为16.4%。再训练一个CNN，在最后的pooling层后多加第6个卷积层，然后对整个ImageNet 2011秋季发布版（1500万图像，2.2万类别）进行分类，然后在数据集上精调参数，得到的错误率为16.6%。在整个2011秋季版数据集上预训练两个CNN，然后与前面提到的5个CNN平均，得到的错误率为15.3%。次好的参赛结果错误率为26.2%，其方法是对几个在FV上训练得到的分类器的预测平均，FV是从不同种类的稠密抽样特征计算得到的[7]。

Table 2: Comparison of error rates on ILSVRC-2012 validation and test sets. In italics are best results achieved by others. Models with an asterisk* were “pre-trained” to classify the entire ImageNet 2011 Fall release. See Section 6 for details.

表2 在ILSVRC验证集和测试集上的错误率对比。斜体表示是别人的最好结果，带星号的是预训练后对整个ImageNet 2011秋季发布版进行分类的结果。详见第6部分。

Model | Top-1 (val) | Top-5 (val) | Top-5 (test)
--- | --- | --- | ---
*SIFT + FVs [7]* | — | — | *26.2%*
1 CNN | 40.7% | 18.2% | —
5 CNNs | 38.1% | 16.4% | 16.4%
1 CNN* | 39.0% | 16.6% | —
7 CNNs* | 36.7% | 15.4% | 15.3%

Finally, we also report our error rates on the Fall 2009 version of ImageNet with 10,184 categories and 8.9 million images. On this dataset we follow the convention in the literature of using half of the images for training and half for testing. Since there is no established test set, our split necessarily differs from the splits used by previous authors, but this does not affect the results appreciably. Our top-1 and top-5 error rates on this dataset are 67.4% and 40.9%, attained by the net described above but with an additional, sixth convolutional layer over the last pooling layer. The best published results on this dataset are 78.1% and 60.9% [19].

最后，我们还对2009年秋季版的ImageNet做了试验，其有890万图像，10184个类别。在这个数据集上，我们依据习惯，一半图像用作训练，一半图像用作测试。由于没有固定的测试集，所以我们的数据集分割与以前作者的分割肯定不同，但这不会显著影响结果。我们的top-1和top-5错误率为67.4%和40.9%，采用的是上述的网络结构，但在最后的pooling层后添加了第6个卷积层。已经发布出来的最好结果是78.1%和60.9%。

### 6.1 Qualitative Evaluations 定性评估

Figure 3 shows the convolutional kernels learned by the network’s two data-connected layers. The network has learned a variety of frequency- and orientation-selective kernels, as well as various colored blobs. Notice the specialization exhibited by the two GPUs, a result of the restricted connectivity described in Section 3.5. The kernels on GPU 1 are largely color-agnostic, while the kernels on on GPU 2 are largely color-specific. This kind of specialization occurs during every run and is independent of any particular random weight initialization (modulo a renumbering of the GPUs).

图3所示的是网络的卷积层学到的卷积核。网络学到了一组频率选择性和方向选择性核心，还有不同颜色的块。注意两个GPU表现出的特别之处，这是3.5中提到的连接限定方案的结果。GPU 1上的核心大多与颜色无关，而GPU 2上的核心大多有特定的颜色。这种特点每次运行都会出现，而且与权值初始化的方式无关（除非GPU重新排序）。

Figure 3: 96 convolutional kernels of size 11×11×3 learned by the first convolutional layer on the 224×224×3 input images. The top 48 kernels were learned on GPU 1 while the bottom 48 kernels were learned on GPU 2. See Section 6.1 for details.

图3 第一个卷积层学习得到的96个11×11×3大小的卷积核，输入图像大小224×224×3。上面48个核在GPU 1上学习，下面48核在GPU 2上学习。详见6.1节。

In the left panel of Figure 4 we qualitatively assess what the network has learned by computing its top-5 predictions on eight test images. Notice that even off-center objects, such as the mite in the top-left, can be recognized by the net. Most of the top-5 labels appear reasonable. For example, only other types of cat are considered plausible labels for the leopard. In some cases (grille, cherry) there is genuine ambiguity about the intended focus of the photograph.

图4左边我们给8幅图计算top-5预测，从而定性的评估网络学习到了什么。注意即使是偏离中心的物体，比如左上的小虫子图像，也可以通过网络识别出来。大多top-5标签是合理的，比如对于豹子图像，只预测其可能是其他类型的猫。在一些情况下（汽车散热器的护栅，樱桃）才对图像的焦点产生了疑义。

Another way to probe the network’s visual knowledge is to consider the feature activations induced by an image at the last, 4096-dimensional hidden layer. If two images produce feature activation vectors with a small Euclidean separation, we can say that the higher levels of the neural network consider them to be similar. Figure 4 shows five images from the test set and the six images from the training set that are most similar to each of them according to this measure. Notice that at the pixel level, the retrieved training images are generally not close in L2 to the query images in the first column. For example, the retrieved dogs and elephants appear in a variety of poses. We present the results for many more test images in the supplementary material.

另一种可视化网络的方法是考虑在最后的4096维隐藏层中由图像带来的特征激活。如果两个图像生成的特征激活向量在欧式空间中距离很小，我们可以说更高层的神经网络认为它们是类似的。图4所示的测试集中的5幅图，和根据这种度量，在训练集中与每幅图最类似的6幅图。注意在像素层级上，得到的训练图像在L2距离上与第一列的查询图像并不接近。例如，得到的狗和大象的图像其姿态各不相同。我们在补充材料中给出了更多测试图像的结果。

Computing similarity by using Euclidean distance between two 4096-dimensional, real-valued vectors is inefficient, but it could be made efficient by training an auto-encoder to compress these vectors to short binary codes. This should produce a much better image retrieval method than applying auto-encoders to the raw pixels [14], which does not make use of image labels and hence has a tendency to retrieve images with similar patterns of edges, whether or not they are semantically similar.

通过计算两个4096维的实值向量的欧式距离来计算相似性是不够的，但可以通过训练一个auto-encoder来压缩这些向量成短的二进制编码，这样就可以了。与将auto-encoder直接应用于原始像素[14]相比，这应当能产生一个好的多的图像检索方法，[14]没有利用图像的标签，所以倾向于检索有类似边缘模式的图像，而与图像是否在语义上类似则无关。

## 7 Discussion

Our results show that a large, deep convolutional neural network is capable of achieving record-breaking results on a highly challenging dataset using purely supervised learning. It is notable that our network’s performance degrades if a single convolutional layer is removed. For example, removing any of the middle layers results in a loss of about 2% for the top-1 performance of the network. So the depth really is important for achieving our results.

我们结果说明一个大型深度卷积神经网络是可以在一个非常有挑战的数据集上采用纯监督学习取得破纪录的结果的。值得注意，我们的网络如果去掉任何一个卷积层，其表现都会变差。比如，去掉中间任何一层，都会导致top-1性能降低2%，所以我们能取得这样的结果，深度是很重要的。

To simplify our experiments, we did not use any unsupervised pre-training even though we expect that it will help, especially if we obtain enough computational power to significantly increase the size of the network without obtaining a corresponding increase in the amount of labeled data. Thus far, our results have improved as we have made our network larger and trained it longer but we still have many orders of magnitude to go in order to match the infero-temporal pathway of the human visual system. Ultimately we would like to use very large and deep convolutional nets on video sequences where the temporal structure provides very helpful information that is missing or far less obvious in static images.

为了简化我们的试验，我们不使用任何无监督的预训练，虽然我们认为这会有帮助，尤其是，如果我们的计算能力足够，可以显著增加网络规模，而没有在标记数据规模上获得相应的增加。迄今为止，我们的网络规模变大，训练时间变长，所以结果得到了改进，但我们距离人眼视觉系统的能力还有很多个数量级的差距。最终我们会用规模非常大非常深的卷积网络对视频序列进行处理，其中的时间结构可以提供很多信息，这在静态图像中是没有的，或者非常不明显。

## References

- [1] R. M. Bell and Y. Koren. Lessons from the netflix prize challenge. ACM SIGKDD Explorations Newsletter, 9(2):75–79, 2007.
- [2] A. Berg, J. Deng, and L. Fei-Fei. Large scale visual recognition challenge 2010. www.image-net.org/challenges. 2010.
- [3] L. Breiman. Random forests. Machine learning, 45(1):5–32, 2001.
- [4] D. Cire¸ san, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. Arxiv preprint arXiv:1202.2745, 2012.
- [5] D.C. Cire¸ san, U. Meier, J. Masci, L.M. Gambardella, and J. Schmidhuber. High-performance neural networks for visual object classification. Arxiv preprint arXiv:1102.0183, 2011.
- [6] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.
- [7] J. Deng, A. Berg, S. Satheesh, H. Su, A. Khosla, and L. Fei-Fei. ILSVRC-2012, 2012. URL http://www.image-net.org/challenges/LSVRC/2012/.
- [8] L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. Computer Vision and Image Understanding, 106(1):59–70, 2007.
- [9] G. Griffin, A. Holub, and P. Perona. Caltech-256 object category dataset. Technical Report 7694, California Institute of Technology, 2007. URL http://authors.library.caltech.edu/7694.
- [10] G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R.R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012.
- [11] K. Jarrett, K. Kavukcuoglu, M. A. Ranzato, and Y. LeCun. What is the best multi-stage architecture for object recognition? In International Conference on Computer Vision, pages 2146–2153. IEEE, 2009.
- [12] A. Krizhevsky. Learning multiple layers of features from tiny images. Master’s thesis, Department of Computer Science, University of Toronto, 2009.
- [13] A. Krizhevsky. Convolutional deep belief networks on cifar-10. Unpublished manuscript, 2010.
- [14] A. Krizhevsky and G.E. Hinton. Using very deep autoencoders for content-based image retrieval. In ESANN, 2011.
- [15] Y. Le Cun, B. Boser, J.S. Denker, D. Henderson, R.E. Howard, W. Hubbard, L.D. Jackel, et al. Handwritten digit recognition with a back-propagation network. In Advances in neural information processing systems, 1990.
- [16] Y. LeCun, F.J. Huang, and L. Bottou. Learning methods for generic object recognition with invariance to pose and lighting. In Computer Vision and Pattern Recognition, 2004. CVPR 2004. Proceedings of the 2004 IEEE Computer Society Conference on, volume 2, pages II–97. IEEE, 2004.
- [17] Y. LeCun, K. Kavukcuoglu, and C. Farabet. Convolutional networks and applications in vision. In Circuits and Systems (ISCAS), Proceedings of 2010 IEEE International Symposium on, pages 253–256. IEEE, 2010.
- [18] H. Lee, R. Grosse, R. Ranganath, and A.Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 609–616. ACM, 2009.
- [19] T. Mensink, J. Verbeek, F. Perronnin, and G. Csurka. Metric Learning for Large Scale Image Classification: Generalizing to New Classes at Near-Zero Cost. In ECCV - European Conference on Computer Vision, Florence, Italy, October 2012.
- [20] V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In Proc. 27th International Conference on Machine Learning, 2010.
- [21] N. Pinto, D.D. Cox, and J.J. DiCarlo. Why is real-world visual object recognition hard? PLoS computational biology, 4(1):e27, 2008.
- [22] N. Pinto, D. Doukhan, J.J. DiCarlo, and D.D. Cox. A high-throughput screening approach to discovering good forms of biologically inspired visual representation. PLoS computational biology, 5(11):e1000579, 2009.
- [23] B.C. Russell, A. Torralba, K.P. Murphy, and W.T. Freeman. Labelme: a database and web-based tool for image annotation. International journal of computer vision, 77(1):157–173, 2008.
- [24] J. Sánchez and F. Perronnin. High-dimensional signature compression for large-scale image classification. In Computer Vision and Pattern Recognition(CVPR),2011 IEEE Conference on, pages 1665–1672.IEEE, 2011.
- [25] P.Y. Simard, D. Steinkraus, and J.C. Platt. Best practices for convolutional neural networks applied to visual document analysis. In Proceedings of the Seventh International Conference on Document Analysis and Recognition, volume 2, pages 958–962, 2003.
- [26] S.C. Turaga, J.F. Murray, V. Jain, F. Roth, M. Helmstaedter, K. Briggman, W. Denk, and H.S. Seung. Convolutional networks can learn to generate affinity graphs for image segmentation. Neural Computation, 22(2):511–538, 2010.