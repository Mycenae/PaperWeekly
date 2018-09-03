# Visualizing and Understanding Convolutional Networks 卷积网络的可视化与理解

Matthew D. Zeiler, Rob Fergus Dept. of Computer Science, Courant Institute, New York University

## Abstract

Large Convolutional Network models have recently demonstrated impressive classification performance on the ImageNet benchmark (Krizhevsky et al., 2012). However there is no clear understanding of why they perform so well, or how they might be improved. In this paper we address both issues. We introduce a novel visualization technique that gives insight into the function of intermediate feature layers and the operation of the classifier. Used in a diagnostic role, these visualizations allow us to find model architectures that outperform Krizhevsky et al. on the ImageNet classification benchmark. We also perform an ablation study to discover the performance contribution from different model layers. We show our ImageNet model generalizes well to other datasets: when the softmax classifier is retrained, it convincingly beats the current state-of-the-art results on Caltech-101 and Caltech-256 datasets.

大型卷积网络模型近期在ImageNet基准测试(Krizhevsky et al., 2012)中的分类任务中表现让人印象深刻，但为什么性能这么好，或怎样得到的改进，没有一个明晰的理解。在这篇文章中，我们对两个问题都有涉猎。我们提出了一种新的可视化技术，可以了解中间特征层的功能，以及分类器的操作。这些可视化起到的是诊断的作用，也让我们能找到超过Krizhevsky et al.在ImageNet分类测试性能的模型架构。我们还进行了解剖研究，看看每一层对模型的性能有多少贡献。我们的试验表明，在ImageNet上试验的我们的模型到其他数据集上也泛化的很好，只需要重新训练softmax层，在Caltech-101和Caltech-256数据集中的表现超过了近期最好的结果很多。

## 1. Introduction 介绍

Since their introduction by (LeCun et al., 1989) in the early 1990’s, Convolutional Networks (convnets) have demonstrated excellent performance at tasks such as hand-written digit classification and face detection. In the last year, several papers have shown that they can also deliver outstanding performance on more challenging visual classification tasks. (Ciresan et al., 2012) demonstrate state-of-the-art performance on NORB and CIFAR-10 datasets. Most notably, (Krizhevsky et al., 2012) show record beating performance on the ImageNet 2012 classification benchmark, with their convnet model achieving an error rate of 16.4%, compared to the 2nd place result of 26.1%. Several factors are responsible for this renewed interest in convnet models: (i) the availability of much larger training sets, with millions of labeled examples; (ii) powerful GPU implementations, making the training of very large models practical and (iii) better model regularization strategies, such as Dropout (Hinton et al., 2012).

自从20世纪90年代初提出卷积网络(convnets, LeCun et al., 1989)以来，convnets在手写数字分类和人脸识别等任务上性能表现优异。去年有几篇论文表示他们也可以在更有挑战性的视觉分类任务中得到非常好的结果。(Ciresan et al., 2012)在NORB和CIFAR-10数据集上证实得到了目前最好的结果。最值得提到的是，(Krizhevsky et al., 2012)破纪录的在ImageNet 2012分类测试中，提出的convnet模型错误率为16.4%，远超第二名的26.1%。对卷积网络模型的复兴，有几个因素很重要：(i)规模大的多的训练数据集的出现，其标记样本可以达到数百万；(ii)性能强劲的GPU实现，使大型模型的训练得以实现；(iii)更好的模型正则化策略，比如dropout (Hinton et al., 2012)。

Despite this encouraging progress, there is still little insight into the internal operation and behavior of these complex models, or how they achieve such good performance. From a scientific standpoint, this is deeply unsatisfactory. Without clear understanding of how and why they work, the development of better models is reduced to trial-and-error. In this paper we introduce a visualization technique that reveals the input stimuli that excite individual feature maps at any layer in the model. It also allows us to observe the evolution of features during training and to diagnose potential problems with the model. The visualization technique we propose uses a multi-layered Deconvolutional Network (deconvnet), as proposed by (Zeiler et al., 2011), to project the feature activations back to the input pixel space. We also perform a sensitivity analysis of the classifier output by occluding portions of the input image, revealing which parts of the scene are important for classification.

尽管这个过程非常令人振奋，但这些复杂模型内部的操作和行为，也就是为什么会得到这样好的结果，很少有人了解。从科学的角度来说，这是非常令人不满意的。不能清楚的理解它们怎样工作，为什么这样工作，要开发更好的模型，就只能去不断试错探索了。在本文中，我们提出了一种可视化技术，可以揭示在模型中任意一层输入刺激是怎样激发出每一幅特征图的。这也使我们可以观察训练过程中特征的演化，诊断模型潜在的问题。我们提出的可视化技术使用了一种多层反卷积网络(deconvnet)，这是由(Zeiler et al., 2011)提出的，用来将特征激活值投射回输入像素空间。我们还对分类器输出进行了敏感度分析，方法是遮挡部分输入图像，这可以发现对于分类来说，图像的哪一部分是重要的。

Using these tools, we start with the architecture of (Krizhevsky et al., 2012) and explore different architectures, discovering ones that outperform their results on ImageNet. We then explore the generalization ability of the model to other datasets, just retraining the softmax classifier on top. As such, this is a form of supervised pre-training, which contrasts with the unsupervised pre-training methods popularized by (Hinton et al., 2006) and others (Bengio et al., 2007; Vincent et al., 2008). The generalization ability of convnet features is also explored in concurrent work by (Donahue et al., 2013).

运用这些工具，我们首先从(Krizhevsky et al., 2012)的架构开始分析，并探索不同的架构，发现在ImageNet上性能更好的架构。然后我们研究一下模型在其他数据集上的泛化能力，其方法是仅仅重新训练一下模型最上面的softmax分类器。这样，这就成了一种有监督的预训练，与之形成比较的是那些无监督预训练方法，如(Hinton et al., 2006), (Bengio et al., 2007; Vincent et al., 2008)等。同时(Donahue et al., 2013)也研究了convnets特征的泛化能力。

### 1.1. Related Work 相关工作

Visualizing features to gain intuition about the network is common practice, but mostly limited to the 1st
layer where projections to pixel space are possible. In higher layers this is not the case, and there are limited methods for interpreting activity. (Erhan et al., 2009) find the optimal stimulus for each unit by performing gradient descent in image space to maximize the unit’s activation. This requires a careful initialization and does not give any information about the unit’s invariances. Motivated by the latter’s short-coming, (Le et al., 2010) (extending an idea by (Berkes & Wiskott, 2006)) show how the Hessian of a given unit may be computed numerically around the optimal response, giving some insight into invariances. The problem is that for higher layers, the invariances are extremely complex so are poorly captured by a simple quadratic approximation. Our approach, by contrast, provides a non-parametric view of invariance, showing which patterns from the training set activate the feature map. (Donahue et al., 2013) show visualizations that identify patches within a dataset that are responsible for strong activations at higher layers in the model. Our visualizations differ in that they are not just crops of input images, but rather top-down projections that reveal structures within each patch that stimulate a particular feature map.

注：本文的卷积神经网络是从下至上的，下面是输入，上面是输出。

对特征进行可视化以得到关于网络的直观感觉，这是比较普遍进行的工作，但多数都只集中在第1层，因为只有这一层才可以直接将特征投射到像素空间。在更高的层中就不是这样的，所以解释更高层的行为的方法就非常有限了。(Erhan et al., 2009)通过图像空间中的梯度下降法对每个单元找到最佳刺激，来对单元的激活函数最大化。这需要小心的初始化，也没有给出单元不变性的任何信息。受到其局限的激发，(Le et al., 2010)拓展了(Berkes & Wiskott, 2006)的思想，展示了一个单元的Hessian矩阵可以理解其不变性，并通过最佳响应计算出来。问题是对于高层，不变性非常复杂，简单的二次逼近特征捕捉能力很差。而我们的方法给出的是不变性的非参数观察，展示了训练集内的哪些特征能激活特征图。(Donahue et al., 2013)的可视化是确定数据集内哪些块能使模型中的高层产生强烈的激活。我们的可视化与之不同，不仅仅是输入图像的剪切块，而是从上至下的映射，反映出激发出某特定特征图的每个块中的结构。

## 2. Approach 方法

We use standard fully supervised convnet models throughout the paper, as defined by (LeCun et al., 1989) and (Krizhevsky et al., 2012). These models map a color 2D input image $x_i$ , via a series of layers, to a probability vector $\hat y_i$ over the *C* different classes. Each layer consists of (i) convolution of the previous layer output (or, in the case of the 1st layer, the input image) with a set of learned filters; (ii) passing the responses through a rectified linear function (*relu(x)* = max(*x,0*)); (iii) [optionally] max pooling over local neighborhoods and (iv) [optionally] a local contrast operation that normalizes the responses across feature maps. For more details of these operations, see (Krizhevsky et al., 2012) and (Jarrett et al., 2009). The top few layers of the network are conventional fully-connected networks and the final layer is a softmax classifier. Fig. 3 shows the model used in many of our experiments.

本文中我们一直使用标准的全监督convnet模型，这由(LeCun et al., 1989) and (Krizhevsky et al., 2012)定义。这些模型将彩色2D输入图像$x_i$，经过一系列层的处理，得到*C*维概率向量$\hat y_i$，*C*是类别数。每层包括(i)上一层输出的卷积（如果是第一层，就是输入的卷积），卷积核是学习得到的滤波器；(ii)卷积结果经过一个ReLU函数(*relu(x)* = max(*x,0*))；(iii)[可选]邻域max-pooling；(iv)[可选]在整个特征图上的响应归一化，这是一个局部对比度操作。这些操作的细节见(Krizhevsky et al., 2012)和(Jarrett et al., 2009)。网络最上面的几层是传统的全连接层，最后一层是softmax分类器。图3所示的是我们的试验中所用到的模型。

![Image](https://www.baidu.com/sample.png)

Figure 3. Architecture of our 8 layer convnet model. A 224 by 224 crop of an image (with 3 color planes) is presented as the input. This is convolved with 96 different 1st layer filters (red), each of size 7 by 7, using a stride of 2 in both x and y. The resulting feature maps are then: (i) passed through a rectified linear function (not shown), (ii) pooled (max within 3x3 regions, using stride 2) and (iii) contrast normalized across feature maps to give 96 different 55 by 55 element feature maps. Similar operations are repeated in layers 2,3,4,5. The last two layers are fully connected, taking features from the top convolutional layer as input in vector form (6 · 6 · 256 = 9216 dimensions). The final layer is a *C*-way softmax function, *C* being the number of classes. All filters and feature maps are square in shape.

图3. 8层convnet模型架构。输入为224×224×3的图像，在第1层与96个不同的滤波器（红色）卷积，滤波器大小7×7，卷积步长2（x和y方向都是），得到的特征图然后：(i)经过一个ReLU（未在图像显示），(ii)max-pooling（3×3取最大值，步长2），(iii)在整个特征图上对比度归一化，最后输出96个不同的55×55特征图。类似的操作在第2,3,4,5层中重复进行。最后两层为全连接层，从最高卷积层取输入，输入为向量形式（6 · 6 · 256 = 9216维）。最后一层是*C*路softmax函数，*C*是类别数目。所有的滤波器和特征图都是方形的。

We train these models using a large set of *N* labeled images {*x,y*}, where label $y_i$ is a discrete variable indicating the true class. A cross-entropy loss function, suitable for image classification, is used to compare $\hat y_i$ and $y_i$. The parameters of the network (filters in the convolutional layers, weight matrices in the fully-connected layers and biases) are trained by backpropagating the derivative of the loss with respect to the parameters throughout the network, and updating the parameters via stochastic gradient descent. Full details of training are given in Section 3.

我们在很大一个数据集上训练这些模型，集合包含*N*个标记图像{*x,y*}，这里$y_i$是离散变量，表示真实类别。我们采用一个适用于图像分类的交叉熵损失函数来比较$\hat y_i$和$y_i$。网络参数（卷积层的滤波器，全连接层的权值矩阵和偏置）的训练方法如下，在整个网络里反向传播损失函数对参数的导数，用随机梯度下降法来更新参数。第3部分里给出了训练的细节。

### 2.1. Visualization with a Deconvnet 通过Deconvnet的可视化

Understanding the operation of a convnet requires interpreting the feature activity in intermediate layers. We present a novel way to map these activities back to the input pixel space, showing what input pattern originally caused a given activation in the feature maps. We perform this mapping with a Deconvolutional Network (deconvnet) (Zeiler et al., 2011). A deconvnet can be thought of as a convnet model that uses the same components (filtering, pooling) but in reverse, so instead of mapping pixels to features does the opposite. In (Zeiler et al., 2011), deconvnets were proposed as a way of performing unsupervised learning. Here, they are not used in any learning capacity, just as a probe of an already trained convnet.

要理解卷积网络convnet的操作，需要解释中间层特征的行为。我们提出一种将这些特征行为映射回输入图像空间的新方法，展示出最初的输入模型怎样导致了特征图中的激活。我们用一个反卷积网络(deconvnet)(Zeiler et al., 2011)来实现这种映射。反卷积网络deconvnet可以认为是一种convnet模型，使用的是相同的组件（滤波、pooling等），但是相反的，所以不是从像素映射到特征，而是从特征映射到像素。在(Zeiler et al., 2011)中，deconvnet是一种进行无监督学习的方法。这里，并没有用它们的学习能力，只用作对已经训练好的convnet进行探测。

To examine a convnet, a deconvnet is attached to each of its layers, as illustrated in Fig. 1(top), providing a continuous path back to image pixels. To start, an input image is presented to the convnet and features computed throughout the layers. To examine a given convnet activation, we set all other activations in the layer to zero and pass the feature maps as input to the attached deconvnet layer. Then we successively (i) unpool, (ii) rectify and (iii) filter to reconstruct the activity in the layer beneath that gave rise to the chosen activation. This is then repeated until input pixel space is reached.

为了对卷积网络convnet进行检查，在其每一层后，都附加了一层反卷积网络deconvnet，如图1所示（上部），这样就有了回到图像像素层的连续路径。开始是这样的，输入图像到convnet，然后通过每一层，计算出相应的特征图。为了检查一个给定的convnet激活，我们将这一层的所有其他激活都设为零，将特征图输入到连接着的deconvnet层，然后我们(i) unpool, (ii)rectify, (iii)filter来在下面一层重建造成给定激活的行为。重复进行这一步骤，直到到达像素空间。

![Image](https://www.baidu.com/sample.png)

Figure 1. Top: A deconvnet layer (left) attached to a convnet layer (right). The deconvnet will reconstruct an approximate version of the convnet features from the layer beneath. Bottom: An illustration of the unpooling operation in the deconvnet, using *switches* which record the location of the local max in each pooling region (colored zones) during pooling in the convnet.

图1 上：一个deconvnet层（左）接着一个convnet层（右边），deconvnet将重建出下面一层convnet特征的近似版本；下：描述了deconvnet中的unpooling操作，在convnet的pooling过程中，用*switches*记录在pooling区域（彩色区域）的局部最大值的位置。

**Unpooling**: In the convnet, the max pooling operation is non-invertible, however we can obtain an approximate inverse by recording the locations of the maxima within each pooling region in a set of *switch* variables. In the deconvnet, the unpooling operation uses these *switches* to place the reconstructions from the layer above into appropriate locations, preserving the structure of the stimulus. See Fig. 1(bottom) for an illustration of the procedure.

**Unpooling**: 在convnet中，max-pooling操作是不可逆的，但我们可以在每个pooling区域记录局部极大值的位置，位置存储在*switch*变量集中，这样就可以得到一个近似的逆。在deconvnet中，unpooling操作利用这些*switches*将那些从上面一层来的数据重建回下面一层的适当位置上，保护激励的结构。图1下部所示的就是这个过程。

**Rectification**: The convnet uses ReLU non-linearities, which rectify the feature maps thus ensuring the feature maps are always positive. To obtain valid feature reconstructions at each layer (which also should be positive), we pass the reconstructed signal through a ReLU non-linearity.

**Rectification**: convnet采用ReLU非线性处理，这对特征图进行了校正因此确保了特征图都是正值。为了得到每一层有效的特征重建（也应当是正值），我们将重建的信号通过一个ReLU非线性处理。

**Filtering**: The convnet uses learned filters to convolve the feature maps from the previous layer. To invert this, the deconvnet uses transposed versions of the same filters, but applied to the rectified maps, not the output of the layer beneath. In practice this means flipping each filter vertically and horizontally.

**Filtering**: convnet使用学习好的滤波器来对前面一层的特征图进行卷积，为了逆转这个过程，deconvnet使用相同滤波器的转置，对校正过的特征图进行卷积，而不是对下面一层的输出卷积。在实践中，这意味着将每个滤波器水平和垂直方向都翻转一次。

Projecting down from higher layers uses the switch settings generated by the max pooling in the convnet on the way up. As these switch settings are peculiar to a given input image, the reconstruction obtained from a single activation thus resembles a small piece of the original input image, with structures weighted according to their contribution toward to the feature activation. Since the model is trained discriminatively, they implicitly show which parts of the input image are discriminative. Note that these projections are not samples from the model, since there is no generative process involved.

在convnet向上推理的过程中max pooling生成switch，使用switch从高层向下投射。由于switch设置对于给定输入图像是特有的，从一个单独的激活得到的重建与原输入图像的一小部分是相似的，这部分图像的结构根据其贡献加权，得到特征激活。由于模型训练对每个点是有区别的，它们隐含的表示出输入图像的哪一部分是与其他部分区分开来的。注意这些投射不是模型中的样本，因为没有生成式的过程。

## 3. Training Details 训练细节

We now describe the large convnet model that will be visualized in Section 4. The architecture, shown in Fig. 3, is similar to that used by (Krizhevsky et al., 2012) for ImageNet classification. One difference is that the sparse connections used in Krizhevsky’s layers 3,4,5 (due to the model being split across 2 GPUs) are replaced with dense connections in our model. Other important differences relating to layers 1 and 2 were made following inspection of the visualizations in Fig. 6, as described in Section 4.1.

我们现在描述一下大型convnet模型，并将在第四部分对其进行可视化。如图3所示为其架构，这与(Krizhevsky et al., 2012)用作图像分类的架构是类似的。一个不同之处是，Krizhevsky在第3,4,5层所用的稀疏连接（由于模型分割在2个GPU上训练）在我们的模型中由稠密连接替代。其他重要的区别与第1,2层有关，见图6中可视化部分，这在4.1节有描述。

Figure 6. (a): 1st layer features without feature scale clipping. Note that one feature dominates. (b): 1st layer features from (Krizhevsky et al., 2012). (c): Our 1st layer features. The smaller stride (2 vs 4) and filter size (7x7 vs 11x11) results in more distinctive features and fewer “dead” features. (d): Visualizations of 2nd layer features from (Krizhevsky et al., 2012). (e): Visualizations of our 2nd layer features. These are cleaner, with no aliasing artifacts that are visible in (d).

图6. (a):第1层特征，没有特征尺度剪切，注意有一个特征特别突出；(b):(Krizhevsky et al., 2012)的第1层(c):我们第1层的特征，较小的卷积步长(2 vs 4)和滤波器尺寸(7×7和11×11)会带来更明显的特征和更少的死特征；(d):(Krizhevsky et al., 2012)中第2层特征的可视化；(e):我们的第2层特征的可视化，更干净一些，没有(d)中的重叠混杂的杂质。

The model was trained on the ImageNet 2012 training set (1.3 million images, spread over 1000 different classes). Each RGB image was preprocessed by resizing the smallest dimension to 256, cropping the center 256x256 region, subtracting the per-pixel mean (across all images) and then using 10 different sub-crops of size 224x224 (corners + center with(out) horizontal flips). Stochastic gradient descent with a mini-batch size of 128 was used to update the parameters, starting with a learning rate of $10^{−2}$, in conjunction with a momentum term of 0.9. We anneal the learning rate throughout training manually when the validation error plateaus. Dropout (Hinton et al., 2012) is used in the fully connected layers (6 and 7) with a rate of 0.5. All weights are initialized to $10^{−2}$ and biases are set to 0.

模型在ImageNet 2012训练集上进行训练，该集合有130万幅图片，分布在1000个不同的类别中。每个RGB图像经过的预处理包括，将宽和高最小的维度变为256，剪切出中间的256×256大小区域，减去每个像素平均值（所有图像），然后使用10个不同的剪切出的224×224大小的子区域，包括角落和中间的，水平翻转过的和原始图像。mini-batch规模128的随机梯度下降用来更新参数，初始学习速率为$10^{−2}$，momentum为0.9。当验证错误率停滞不变时，训练全过程手动对学习速率退火。在全连接层(6,7)使用了Dropout (Hinton et al., 2012)，比率为0.5，所有的权重初始化为$10^{−2}$，偏置设为0。

Visualization of the first layer filters during training reveals that a few of them dominate, as shown in Fig. 6(a). To combat this, we renormalize each filter in the convolutional layers whose RMS value exceeds a fixed radius of $10^{−1}$ to this fixed radius. This is crucial, especially in the first layer of the model, where the input images are roughly in the [-128,128] range. As in (Krizhevsky et al., 2012), we produce multiple different crops and flips of each training example to boost training set size. We stopped training after 70 epochs, which took around 12 days on a single GTX580 GPU, using an implementation based on (Krizhevsky et al., 2012).

训练过程中第1层滤波器的可视化揭示出，有几个滤波器起主导作用，如图6(a)所示。为避免这种情况，我们对卷积层的一些滤波器进行了重新归一化，这些滤波器的RMS(root mean square)值超过了一个固定半径的$10^{−1}$。这非常重要，尤其是模型第1层，这里输入图像大致范围是[-128,128]。就像在(Krizhevsky et al., 2012)里一样，我们对每个训练样本生成多个不同的剪切或翻转图，来提升训练集大小。我们在70个epoch后停止训练，大约时间为12天，训练用了单个的GTX580 GPU，实现方法是基于(Krizhevsky et al., 2012)的。

## 4. Convnet Visualization 卷积网络可视化

Using the model described in Section 3, we now use the deconvnet to visualize the feature activations on
the ImageNet validation set.

采用第3部分中描述的模型，我们现在用deconvnet来对ImageNet验证数据集上的特征激活进行可视化。

**Feature Visualization**: Fig. 2 shows feature visualizations from our model once training is complete. However, instead of showing the single strongest activation for a given feature map, we show the top 9 activations. Projecting each separately down to pixel space reveals the different structures that excite a given feature map, hence showing its invariance to input deformations. Alongside these visualizations we show the corresponding image patches. These have greater variation than visualizations as the latter solely focus on the discriminant structure within each patch. For example, in layer 5, row 1, col 2, the patches appear to have little in common, but the visualizations reveal that this particular feature map focuses on the grass in the background, not the foreground objects.

**特征可视化**：图2所示的是我们的模型训练完成后的特征可视化。但是，对于给定的特征图，我们不是显示最强的激活，我们显示最高的9个激活。将每个都投射到像素空间，可以发现激励出特定特征图的不同结构，然后发现对输入变形的不变性。与这些可视化效果一起，我们还展示了对应的图像块。这些图像块比相应的可视化结果变化更大，因为后者只关注每个块中区别性的结构。比如，在第5层，第1行第2列，图像块显示其没有什么共同点，但可视化效果中的这个特征图却都聚焦在背景的草地上，而不是前面的物体上。

Figure 2. Visualization of features in a fully trained model. For layers 2-5 we show the top 9 activations in a random subset of feature maps across the validation data, projected down to pixel space using our deconvolutional network approach. Our reconstructions are not samples from the model: they are reconstructed patterns from the validation set that cause high activations in a given feature map. For each feature map we also show the corresponding image patches. Note: (i) the strong grouping within each feature map, (ii) greater invariance at higher layers and (iii) exaggeration of discriminative parts of the image, e.g. eyes and noses of dogs (layer 4, row 1, cols 1). Best viewed in electronic form.

图2. 训练完成的模型的特征可视化。对于2-5层，我们在验证数据集特征图的随机子集中将最高的9个激活，用我们的反卷积网络deconvnet方法投射回像素空间显示出来。我们的重建结果不是模型中的样本：它们是从验证集中重建的模式，它们是给定的特征图中高激活的原因。对于每个特征图我们还显示了对应的图像块，注意：(i)每个特征图中强烈的群聚现象；(ii)更高的层中有更多的不变性；(iii)图像突出部分的夸大，比如，狗的眼部和鼻子（第4层，第1行第1列）。电子版形式查看效果最佳。

The projections from each layer show the hierarchical nature of the features in the network. Layer 2 responds to corners and other edge/color conjunctions. Layer 3 has more complex invariances, capturing similar textures (e.g. mesh patterns (Row 1, Col 1); text (R2,C4)). Layer 4 shows significant variation, but is more class-specific: dog faces (R1,C1); bird’s legs (R4,C2). Layer 5 shows entire objects with significant pose variation, e.g. keyboards (R1,C1) and dogs (R4).

每一层的投射展示了网络中特征的分级的性质。第2层对应着角点和其他边缘/色彩连接处。第3层的不变性更复杂一些，捕捉到的类似的纹理（比如，第1行第1列的网状结构，第2行第4列的文字）。第4层是显著的变化，但更多与类别相关：第1行第1列的狗脸，第4行第2列的鸟腿。第5层是明显姿势变化的整个物体，如第1行第1列的键盘，和第4行的狗。

**Feature Evolution during Training**: Fig. 4 visualizes the progression during training of the strongest activation (across all training examples) within a given feature map projected back to pixel space. Sudden jumps in appearance result from a change in the image from which the strongest activation originates. The lower layers of the model can be seen to converge within a few epochs. However, the upper layers only develop after a considerable number of epochs (40-50), demonstrating the need to let the models train until fully converged.

**训练过程中的特征演化**：图4将训练过程中最强的激活的演化过程（对所有的训练样本）可视化表现了出来，方法也是在一个给定的特征图中投射回像素空间。出现的突变是由最强激活源自的图像变化导致的。模型中较低的层经过几个epoch之后就收敛了，而较高的层在很多epoch (40-50)之后才演化完成，说明需要模型训练直到充分收敛。

Figure 4. Evolution of a randomly chosen subset of model features through training. Each layer’s features are displayed in a different block. Within each block, we show a randomly chosen subset of features at epochs [1,2,5,10,20,30,40,64]. The visualization shows the strongest activation (across all training examples) for a given feature map, projected down to pixel space using our deconvnet approach. Color contrast is artificially enhanced and the figure is best viewed in electronic form.

图4. 随机选择的模型特征子集在训练过程中的演化。每一层的特征都展示在不同的方块里。在每一块中，我们展示了随机选择的特征子集在第[1,2,5,10,20,30,40,64] 8个epoch的结果。可视化结果展示了对于给定的特征图的最强激活（在所有训练样本中），方法是用我们的deconvnet方法投射回像素空间。色彩对比经过了人工强化，图像以电子版形式查看最佳。

**Feature Invariance**: Fig. 5 shows 5 sample images being translated, rotated and scaled by varying degrees while looking at the changes in the feature vectors from the top and bottom layers of the model, relative to the untransformed feature. Small transformations have a dramatic effect in the first layer of the model, but a lesser impact at the top feature layer, being quasilinear for translation & scaling. The network output is stable to translations and scalings. In general, the output is not invariant to rotation, except for object with rotational symmetry (e.g. entertainment center).

**特征不变性**：图5显示了经过不同程度的平移、旋转和尺度变换的样本图像，并观察在模型从上到下的层中的特征向量中变化，并与未变换特征对比。很小的变化在模型第1层就有比较大的变化，但在高层变化就小一些，对于平移和尺度变化是拟线性的。网络输出对于平移和尺度变化是稳定的。一般来说，输出对于旋转来说不是不变的，当然对于具有旋转对称性的物体除外（比如，娱乐中心）。

Figure 5. Analysis of vertical translation, scale, and rotation invariance within the model (rows a-c respectively). Col 1: 5 example images undergoing the transformations. Col 2 & 3: Euclidean distance between feature vectors from the original and transformed images in layers 1 and 7 respectively. Col 4: the probability of the true label for each image, as the image is transformed.

图5. 模型的垂直平移、尺度变化和旋转不变性分析（分别对应a行b行c行）。第1列：经过变换的样本图像；第2、3列：原始图像和变换图像在第1到7层的特征向量间的欧式距离；第4列：图像变换后，每个图像的真值标签概率。

### 4.1. Architecture Selection 架构选择

While visualization of a trained model gives insight into its operation, it can also assist with selecting good architectures in the first place. By visualizing the first and second layers of Krizhevsky et al. ’s architecture (Fig. 6(b) & (d)), various problems are apparent. The first layer filters are a mix of extremely high and low frequency information, with little coverage of the mid frequencies. Additionally, the 2nd layer visualization shows aliasing artifacts caused by the large stride 4 used in the 1st layer convolutions. To remedy these problems, we (i) reduced the 1st layer filter size from 11x11 to 7x7 and (ii) made the stride of the convolution 2, rather than 4. This new architecture retains much more information in the 1st and 2nd layer features, as shown in Fig. 6(c) & (e). More importantly, it also improves the classification performance as shown in Section 5.1.

训练好的模型的可视化可以了解其内部操作，首先还可以协助选择好的架构。通过对Krizhevsky et al.模型第1和第2层进行可视化（图6b、6d），有几个问题非常明显。第1层是极高和极低的频率信息的混合，中间频率的信息很少；另外第2层的可视化结果出现混叠的杂物，这是由在第1层里用到的大卷积步长4导致的。为解决这个问题，我们(i)将第1层的滤波器尺寸由11×11改为7×7；(ii)将卷积步长由4改为2。这种新的结构在第1层和第2层保留的信息多了很多，如图6c和6e所示。更重要的是，还提高了分类性能，这在5.1节有介绍。

### 4.2. Occlusion Sensitivity 对遮挡的敏感性

With image classification approaches, a natural question is if the model is truly identifying the location of the object in the image, or just using the surrounding context. Fig. 7 attempts to answer this question by systematically occluding different portions of the input image with a grey square, and monitoring the output of the classifier. The examples clearly show the model is localizing the objects within the scene, as the probability of the correct class drops significantly when the object is occluded. Fig. 7 also shows visualizations from the strongest feature map of the top convolution layer, in addition to activity in this map (summed over spatial locations) as a function of occluder position. When the occluder covers the image region that appears in the visualization, we see a strong drop in activity in the feature map. This shows that the visualization genuinely corresponds to the image structure that stimulates that feature map, hence validating the other visualizations shown in Fig. 4 and Fig. 2.

对于图像分类算法，一个自然的问题是，模型是真的能识别目标在图像内的位置，还是靠周围的上下文。图7试图回答这个问题，试验中我们用一个灰色方块系统性的遮挡输入图像的不同部分，并监控分类器的输出。例子清楚的说明，模型对场景中的物体进行了定位，当目标被遮挡的时候，正确类别的概率明显下降。图7还包括最高卷积层的最强特征图的可视化，还有这个特征图（对所有空域位置求和）以遮挡物位置为参数的函数。当遮挡物盖住了在可视化图中出现的图像区域，我们看到了特征图中活动的剧烈下降。这表明，可视化结果真正对应激励出特征图的图像结构，因此验证了其他可视化结果（图4和图2）。

Figure 7. Three test examples where we systematically cover up different portions of the scene with a gray square (1st column) and see how the top (layer 5) feature maps ((b) & (c)) and classifier output ((d) & (e)) changes. (b): for each position of the gray square, we record the total activation in the layer 5 feature map (the one with the strongest response in the unoccluded image). (c): a visualization of this feature map projected down into the input image (black square), along with visualizations of this map from other images. The first row example shows the strongest feature to be the dog’s face. When this is covered-up the activity in the feature map decreases (blue area in (b)). (d): a map of correct class probability, as a function of the position of the gray square. E.g. when the dog’s face is obscured, the probability for “pomeranian” drops significantly. (e): the most probable label as a function of occluder position. E.g. in the 1st row, for most locations it is “pomeranian”, but if the dog’s face is obscured but not the ball, then it predicts “tennis ball”. In the 2nd example, text on the car is the strongest feature in layer 5, but the classifier is most sensitive to the wheel. The 3rd example contains multiple objects. The strongest feature in layer 5 picks out the faces, but the classifier is sensitive to the dog (blue region in (d)), since it uses multiple feature maps.

图7. 三个测试例子，我们用一个灰色方块系统的遮挡场景的不同部分（a列），观察最高层（第5层）特征图（b列c列）和分类器输出（d列e列）如何变化。(b):对灰色方块的每个位置，我们都记录了第5层特征图的总共激活（未遮挡图像的最强响应）；(c):这个特征图投射回输入图像空间的可视化结果（黑色方块），还有其他图像的这张图的可视化，第1行的例子表明最强特征是狗脸，当遮挡时特征图的活动会衰减（b中的蓝色区域）；(d):正确分类概率图，以灰色方块位置为参数的函数，比如，当狗脸被遮挡时，pomeranian的概率明显下降；(e):标签概率与遮挡位置的函数关系图，比如在第一行，大多数位置都是pomeranian，单如果狗脸被遮挡（而不是那个球），那么预测为tennis ball，在第二个例子中，车上的文字是第5层的最强特征，但分类器对轮子最为敏感，第3行的例子包含多个目标，其第5层的最强特征应当是人脸，但分类器对狗敏感（d中的蓝色区域），因为使用了多特征图。

### 4.3. Correspondence Analysis 对应的分析

Deep models differ from many existing recognition approaches in that there is no explicit mechanism for establishing correspondence between specific object parts in different images (e.g. faces have a particular spatial configuration of the eyes and nose). However, an intriguing possibility is that deep models might be implicitly computing them. To explore this, we take 5 randomly drawn dog images with frontal pose and systematically mask out the same part of the face in each image (e.g. all left eyes, see Fig. 8). For each image *i*, we then compute:  $ϵ_i^l = x_i^l - \tilde x_i^l$, where $x_i^l$ and $\tilde x_i^l$ are the feature vectors at layer *l* for the original and occluded images respectively. We then measure the consistency of this difference vector *ϵ* between all related image pairs (*i,j*): $∆_l = \sum _{i,j=1,i \neq j}^5 H(sign(ϵ_i^l) - sign(ϵ_j^l))$, where *H* is Hamming distance. A lower value indicates greater consistency in the change resulting from the masking operation, hence tighter correspondence between the same object parts in different images (i.e. blocking the left eye changes the feature representation in a consistent way). In Table 1 we compare the ∆ score for three parts of the face (left eye, right eye and nose) to random parts of the object, using features from layer *l* = 5 and *l* = 7. The lower score for these parts, relative to random object regions, for the layer 5 features show the model does establish some degree of correspondence.

深度模型与很多现有的识别方法都不一样，不同之处在于，没有明显的机制来确定特定目标与不同图像对应起来（比如，人脸有特殊的眼睛和鼻子的特殊空域配置）。但是，深度模型可能是隐式的计算它们。为了探寻原因，我们用5个随机选出来的狗图像，前面是狗鼻子，并系统性的对每幅图像遮挡住脸的相同部分（比如，都是左眼，见图8）。对每幅图*i*，我们计算$ϵ_i^l = x_i^l - \tilde x_i^l$，这里$x_i^l$ 和 $\tilde x_i^l$分别是原图和遮挡图的第*l*层的特征矢量。然后我们衡量差矢*ϵ*对于所有相关图像对(*i,j*)的一致性：$∆_l = \sum _{i,j=1,i \neq j}^5 H(sign(ϵ_i^l) - sign(ϵ_j^l))$，这里*H*是Hamming距离。值较小，说明遮挡变化的一致性较强，也就是不同图像中同样的物体部分有较紧的对应关系（即，挡住左眼，对特征表示的改变比较一致）。在表1中，我们比较了脸部三个部位（左眼，右眼和鼻子）的∆值和其他随机部分的值，采用第5层和第7层的特征值。这些部分较低的值，相对于随机目标区域，在第5层上的特征，表明模型确实确立了一定的对应性。

Figure 8. Images used for correspondence experiments. Col 1: Original image. Col 2,3,4: Occlusion of the right eye, left eye, and nose respectively. Other columns show examples of random occlusions.

图8. 用于对照性试验的图像。第1列，原始图像；第2,3,4列，分别遮挡右眼、左眼、鼻子；其他列为遮挡随机区域的例子。

Table 1. Measure of correspondence for different object parts in 5 different dog images. The lower scores for the eyes and nose (compared to random object parts) show the model implicitly establishing some form of correspondence of parts at layer 5 in the model. At layer 7, the scores are more similar, perhaps due to upper layers trying to discriminate between the different breeds of dog.

表1. 5中不同的狗的图像，不同目标部分的对应性的衡量。眼睛和鼻子部分对应的较小的值（与随机目标部分相比）说明，模型在第5层隐式的确定了一些不同部分的对应性。在第7层，值更加接近，可能是由于高层在区分不同种类的狗。

Occlusion Location | Layer 5 | Layer 7
--- | --- | ---
Right Eye | 0.067 ± 0.007 | 0.069 ± 0.015
Left Eye | 0.069 ± 0.007 | 0.068 ± 0.013
Nose | 0.079 ± 0.017 | 0.069 ± 0.011
Random | 0.107 ± 0.017 | 0.073 ± 0.014

## 5. Experiments

### 5.1. ImageNet 2012

This dataset consists of 1.3M/50k/100k training/validation/test examples, spread over 1000 categories. Table 2 shows our results on this dataset.

Table 2. ImageNet 2012 classification error rates. The ∗ indicates models that were trained on both ImageNet 2011 and 2012 training sets.

Error % | Val Top-1 | Val Top-5 | Test Top-5
--- | --- | --- | ---
(Gunji et al., 2012) | - | - | 26.2
(Krizhevsky et al., 2012), 1 convnet | 40.7 | 18.2 | −−
(Krizhevsky et al., 2012), 5 convnets | 38.1 | 16.4 | 16.4
(Krizhevsky et al., 2012) ∗ , 1 convnets | 39.0 | 16.6 | −−
(Krizhevsky et al., 2012) ∗ , 7 convnets | 36.7 | 15.4 | 15.3
Our replication of (Krizhevsky et al., 2012), 1 convnet | 40.5 | 18.1 | −−
1 convnet as per Fig. 3 | 38.4 | 16.5 | −−
5 convnets as per Fig. 3 – (a) | 36.7 | 15.3 | 15.3
1 convnet as per Fig. 3 but with layers 3,4,5: 512,1024,512 maps – (b) | 37.5 | 16.0 | 16.1
6 convnets, (a) & (b) combined | 36.0 | 14.7 | 14.8

Using the exact architecture specified in (Krizhevsky et al., 2012), we attempt to replicate their result on the validation set. We achieve an error rate within 0.1% of their reported value on the ImageNet 2012 validation set.

Next we analyze the performance of our model with the architectural changes outlined in Section 4.1 (7×7 filters in layer 1 and stride 2 convolutions in layers 1 & 2). This model, shown in Fig. 3, significantly outperforms the architecture of (Krizhevsky et al., 2012), beating their single model result by 1.7% (test top-5). When we combine multiple models, we obtain a test error of 14.8%, the best published performance
on this dataset (This performance has been surpassed in the recent Imagenet 2013 competition (http://www.image-net.org/challenges/LSVRC/2013/results.php), despite only using the 2012 training set). We note that this error is almost half that of the top non-convnet entry in the ImageNet 2012 classification challenge, which obtained 26.2% error (Gunji et al., 2012).

**Varying ImageNet Model Sizes**: In Table 3, we first explore the architecture of (Krizhevsky et al., 2012) by adjusting the size of layers, or removing them entirely. In each case, the model is trained from scratch with the revised architecture. Removing the fully connected layers (6,7) only gives a slight increase in error. This is surprising, given that they contain the majority of model parameters. Removing two of the middle convolutional layers also makes a relatively small different to the error rate. However, removing both the middle convolution layers and the fully connected layers yields a model with only 4 layers whose performance is dramatically worse. This would suggest that the overall depth of the model is important for obtaining good performance. In Table 3, we modify our model, shown in Fig. 3. Changing the size of the fully connected layers makes little difference to performance (same for model of (Krizhevsky et al., 2012)). However, increasing the size of the middle convolution layers goes give a useful gain in performance. But increasing these, while also enlarging the fully connected layers results in overfitting.

Table 3. ImageNet 2012 classification error rates with various architectural changes to the model of (Krizhevsky et al., 2012) and our model (see Fig. 3).

Error % | Train Top-1 | Val Top-1 | Val Top-5
--- | --- | --- | ---
Our replication of (Krizhevsky et al., 2012), 1 convnet | 35.1 | 40.5 | 18.1
Removed layers 3,4 | 41.8 | 45.4 | 22.1
Removed layer 7 | 27.4 | 40.0 | 18.4
Removed layers 6,7 | 27.4 | 44.8 | 22.4
Removed layer 3,4,6,7 | 71.1 | 71.3 | 50.1
Adjust layers 6,7: 2048 units | 40.3 | 41.7 | 18.8
Adjust layers 6,7: 8192 units | 26.8 | 40.0 | 18.1
Our Model (as per Fig. 3) | 33.1 | 38.4 | 16.5
Adjust layers 6,7: 2048 units | 38.2 | 40.2 | 17.6
Adjust layers 6,7: 8192 units | 22.0 | 38.8 | 17.0
Adjust layers 3,4,5: 512,1024,512 maps | 18.8 | 37.5 | 16.0
Adjust layers 6,7: 8192 units and Layers 3,4,5: 512,1024,512 maps | 10.0 | 38.3 | 16.9

### 5.2. Feature Generalization

### 5.3. Feature Analysis

## 6. Discussion

We explored large convolutional neural network models, trained for image classification, in a number ways. First, we presented a novel way to visualize the activity within the model. This reveals the features to be far from random, uninterpretable patterns. Rather, they show many intuitively desirable properties such as compositionality, increasing invariance and class discrimination as we ascend the layers. We also showed how these visualization can be used to debug problems with the model to obtain better results, for example improving on Krizhevsky et al. ’s (Krizhevsky et al., 2012) impressive ImageNet 2012 result. We then demonstrated through a series of occlusion experiments that the model, while trained for classification, is highly sensitive to local structure in the image and is not just using broad scene context. An ablation study on the model revealed that having a minimum depth to the network, rather than any individual section, is vital to the model’s performance.

Finally, we showed how the ImageNet trained model can generalize well to other datasets. For Caltech-101 and Caltech-256, the datasets are similar enough that we can beat the best reported results, in the latter case by a significant margin. This result brings into question to utility of benchmarks with small (i.e. < $10^4$ ) training sets. Our convnet model generalized less well to the PASCAL data, perhaps suffering from dataset bias (Torralba & Efros, 2011), although it was still within 3.2% of the best reported result, despite no tuning for the task. For example, our performance might improve if a different loss function was used that permitted multiple objects per image. This would naturally enable the networks to tackle the object detection as well.

## Acknowledgments

The authors are very grateful for support by NSF grant IIS-1116923, Microsoft Research and a Sloan Fellow-
ship.

## References

- Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. Greedy layer-wise training of deep networks. In NIPS, pp. 153–160, 2007.
- Berkes, P. and Wiskott, L. On the analysis and interpretation of inhomogeneous quadratic forms as receptive fields. Neural Computation, 2006.
- Bo, L., Ren, X., and Fox, D. Multipath sparse coding using hierarchical matching pursuit. In CVPR, 2013.
- Ciresan, D. C., Meier, J., and Schmidhuber, J. Multi-column deep neural networks for image classification. In CVPR, 2012.
- Dalal, N. and Triggs, B. Histograms of oriented gradients for pedestrian detection. In CVPR, 2005.
- Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., and Darrell, T. DeCAF: A deep convolutional activation feature for generic visual recognition. In arXiv:1310.1531, 2013.
- Erhan, D., Bengio, Y., Courville, A., and Vincent, P. Visualizing higher-layer features of a deep network. In Technical report, University of Montreal, 2009.
- Fei-fei, L., Fergus, R., and Perona, P. One-shot learning of object categories. IEEE Trans. PAMI, 2006.
- Griffin, G., Holub, A., and Perona, P. The caltech 256. In Caltech Technical Report, 2006.
- Gunji, N., Higuchi, T., Yasumoto, K., Muraoka, H., Ushiku, Y., Harada, T., and Kuniyoshi, Y. Classification entry. In Imagenet Competition, 2012.
- Hinton, G. E., Osindero, S., and The, Y. A fast learning algorithm for deep belief nets. Neural Computation, 18:1527–1554, 2006.
- Hinton, G.E., Srivastave, N., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. R. Improving neural networks by preventing co-adaptation of feature detectors. arXiv:1207.0580, 2012.
- Jarrett, K., Kavukcuoglu, K., Ranzato, M., and LeCun, Y. What is the best multi-stage architecture for object recognition? In ICCV, 2009.
- Jianchao, Y., Kai, Y., Yihong, G., and Thomas, H. Linear spatial pyramid matching using sparse coding for image classification. In CVPR, 2009.
- Krizhevsky, A., Sutskever, I., and Hinton, G.E. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.
- Le, Q. V., Ngiam, J., Chen, Z., Chia, D., Koh, P., and Ng, A. Y. Tiled convolutional neural networks. In NIPS, 2010.
- LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., and Jackel, L. D. Backpropagation applied to handwritten zip code recognition. Neural Comput., 1(4):541–551, 1989.
- Sande, K., Uijlings, J., Snoek, C., and Smeulders, A. Hybrid coding for selective search. In PASCAL VOC Classification Challenge 2012, 2012.
- Sohn, K., Jung, D., Lee, H., and Hero III, A. Efficient learning of sparse, distributed, convolutional feature representations for object recognition. In ICCV, 2011.
- Torralba, A. and Efros, A. A. Unbiased look at dataset bias. In CVPR, 2011.
- Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P. A. Extracting and composing robust features with denoising autoencoders. In ICML, pp. 1096–1103, 2008.
- Yan, S., Dong, J., Chen, Q., Song, Z., Pan, Y., Xia, W., Huang, Z., Hua, Y., and Shen, S. Generalized hierarchical matching for sub-category aware object classification. In PASCAL VOC Classification Challenge 2012, 2012.
- Zeiler, M., Taylor, G., and Fergus, R. Adaptive deconvolutional networks for mid and high level feature learning. In ICCV, 2011.