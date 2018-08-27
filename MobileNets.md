# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
# MobileNets: 面向移动视觉应用的高效卷积神经网络

Andrew G. Howard et. al, Google Inc.

**Abstract 摘要**

We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depthwise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.

我们提出了一类面向移动和嵌入式视觉应用的高效模型，即MobileNet。MobileNets是流线型结构的，使用了depthwise seperable convolutions来构建轻量级的深度神经网络。我们引入了两个简单的全局超参数，这样就可以在延迟和准确性上进行有效的折中取舍。这些超参数使模型建构者在面对应用问题时，基于其约束，选择正确规模的模型。对ImageNet分类问题，我们在资源和准确率的折中方面进行了广泛的实验，与其他受欢迎的模型相比，MobileNet有突出的表现。我们展示了MobileNet在很多应用中的有效性，包括目标检测、细粒度图像分类、人脸属性识别和~~大型地标的地理定位~~。

## 1. Introduction 简介

Convolutional neural networks have become ubiquitous in computer vision ever since AlexNet [19] popularized deep convolutional neural networks by winning the ImageNet Challenge: ILSVRC 2012 [24]. The general trend has been to make deeper and more complicated networks in order to achieve higher accuracy [27, 31, 29, 8]. However, these advances to improve accuracy are not necessarily making networks more efficient with respect to size and speed. In many real world applications such as robotics, self-driving car and augmented reality, the recognition tasks need to be carried out in a timely fashion on a computationally limited platform.

利用深度卷积神经网络，AlextNet[19]赢得了2012年的ImageNet挑战赛[24]，自此CNN大受欢迎，现在在计算机视觉领域CNN变得无处不在。一般的方向是让网络更深更复杂，以提高准确性[27,31,29,8]，但这些进展在准确性上的改善并没有使网络在大小和速度方面也同样得到提高。现实中的很多应用比如机器人，自动驾驶汽车和增强现实，必须在一个计算能力有限的平台上及时的运行识别任务。

This paper describes an efficient network architecture and a set of two hyper-parameters in order to build very small, low latency models that can be easily matched to the design requirements for mobile and embedded vision applications. Section 2 reviews prior work in building small models. Section 3 describes the MobileNet architecture and two hyper-parameters width multiplier and resolution multiplier to define smaller and more efficient MobileNets. Section 4 describes experiments on ImageNet as well a variety of different applications and use cases. Section 5 closes with a summary and conclusion.

这篇文章提出了一种高效的网络结构，包括两个超参数来构建小型的、低延迟的模型，很容易满足移动和嵌入式视觉应用的设计需求。第二部分回顾了构建小型模型的已有工作；第三部分描述了MobileNet架构和两个超参数，分别是width multiplier和resolution multiplier，通过它们来定义更小型更有效的MobileNet。第四部分描述了在ImageNet上的实验和很多不同的应用、应用场景。第五部分以总结和结论结束。

## 2. Prior Work 已有工作

There has been rising interest in building small and efficient neural networks in the recent literature, e.g. [16, 34, 12, 36, 22]. Many different approaches can be generally categorized into either compressing pretrained networks or training small networks directly. This paper proposes a class of network architectures that allows a model developer to specifically choose a small network that matches the resource restrictions (latency, size) for their application. MobileNets primarily focus on optimizing for latency but also yield small networks. Many papers on small networks focus only on size but do not consider speed.

最近的文献对构建小型高效的神经网络兴趣不断提高，如[16,34,12,36,22]。这些不同的方法一般可以归为两类，一类是压缩预训练的网络，一类是直接训练小型网络。本文提出了一类网络结构，使模型开发者可以明确的选择小型网络来满足应用的资源限制（如延迟，规模）。MobileNet优先聚焦在延迟的优化上，但也可以生成小型网络。许多关于小型网络的文章主要关注在规模，但没有考虑速度。

MobileNets are built primarily from depthwise separable convolutions initially introduced in [26] and subsequently used in Inception models [13] to reduce the computation in the first few layers. Flattened networks [16] build a network out of fully factorized convolutions and showed the potential of extremely factorized networks. Independent of this current paper, Factorized Networks[34] introduces a similar factorized convolution as well as the use of topological connections. Subsequently, the Xception network [3] demonstrated how to scale up depthwise separable filters to out perform Inception V3 networks. Another small network is Squeezenet [12] which uses a bottleneck approach to design a very small network. Other reduced computation networks include structured transform networks [28] and deep fried convnets [37].

MobileNet是从depthwise seperable convolutions构建出来的，最早在[26]中提出，然后在Inception模型[13]中应用，主要作用是减少最初几层的计算量。Flattened networks [16]由全分解卷积(fully factorized convolutions)构建，显示出了重度分解网络(extremely factorized networks)的潜力。Factorized Networks[34]同时提出了类似的分解卷积(factorized convolution)，也应用了拓扑连接。随后，Xception网络[3]展示了怎样利用depthwise seperable filters来超越Inception V3网络的表现。另一个小型网络是Squeezenet[12]，使用了一种瓶颈方法来设计非常小型的网络。其他缩减计算量的网络包括structured transform networks[28]和deep fried convnets[37]。

A different approach for obtaining small networks is shrinking, factorizing or compressing pretrained networks. Compression based on product quantization [36], hashing [2], and pruning, vector quantization and Huffman coding [5] have been proposed in the literature. Additionally various factorizations have been proposed to speed up pretrained networks [14, 20]. Another method for training small networks is distillation [9] which uses a larger network to teach a smaller network. It is complementary to our approach and is covered in some of our use cases in section 4. Another emerging approach is low bit networks [4, 22, 11].

另一种网络小型化的方法是压缩预训练网络。一些文献提出了基于product quantization[36], hashing[2]和pruning, vector quantization and Huffman coding[5]的压缩方法。另外，提出了各种分解方法(factorization)来加速预训练网络[14,20]。训练小型网络的另一种方法是distillation[9]，其使用了一个更大型的网络教出一个更小型的网络。这是对我们方法的一种补充，在第四部分的一些应用场景中有所使用。另一种刚刚兴起的方法是low bit networks[4,22,11]。

## 3. MobileNet Architecture MobileNet结构
In this section we first describe the core layers that MobileNet is built on which are depthwise separable filters. We then describe the MobileNet network structure and conclude with descriptions of the two model shrinking hyper-parameters width multiplier and resolution multiplier.

在这部分中我们首先描述了设计MobileNet的核心层，就是depthwise seperable filters，然后我们描述了MobileNet的结构，最后描述了使模型缩小的两个超参数，width multiplier和resolution multiplier。

### 3.1. Depthwise Separable Convolution

The MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a 1×1 convolution called a pointwise convolution. For MobileNets the depthwise convolution applies a single filter to each input channel. The pointwise convolution then applies a 1×1 convolution to combine the outputs the depthwise convolution. A standard convolution both filters and combines inputs into a new set of outputs in one step. The depthwise separable convolution splits this into two layers, a separate layer for filtering and a separate layer for combining. This factorization has the effect of drastically reducing computation and model size. Figure 2 shows how a standard convolution 2(a) is factorized into a depthwise convolution 2(b) and a 1 × 1 pointwise convolution 2(c).

MobileNet模型是基于depthwise seperable convolutions的，这是factorized convolutions的一种形式，而factorized convolution就是将标准的卷积操作分解成一种depthwise convolution，其中1×1的卷积被称为pointwise convolution。MobileNet中，depthwise convolution对每个输入通道都有一个单独的滤波器。然后应用pointwise convolution来将depthwise convolution的输出组合起来。标准卷积对输入既进行滤波，也进行组合，一步得到新的输出集合，而depthwise seperable convolution将这分成两层，一层用来滤波，一层用来组合。这种分解可以大幅降低计算量和模型规模。图2展示了标准卷积2(a)如何分解成depthwise convolution 2(b)和1×1的pointwise convolution 2(c)。

A standard convolutional layer takes as input a $D_F × D_F × M$ feature map **F** and produces a $D_F × D_F × N$ feature map **G** where $D_F$ is the spatial width and height of a square input feature map(We assume that the output feature map has the same spatial dimensions as the input and both feature maps are square. Our model shrinking results generalize to feature maps with arbitrary sizes and aspect ratios), $M$ is the number of input channels(input depth), $D_G$ is the spatial width and height of a square output feature map and $N$ is the number of output channel (output depth).

标准卷积层输入为$D_F × D_F × M$的特征图**F**，产生$D_F × D_F × N$的特征图**G**，这里$D_F$是正方形输入特征图的空间高度和宽度（我们假设输出特征图与输入空间大小一样，两个特征图都是正方形的，我们模型可以一般化应用到具有任意大小和纵横比的特征图），$M$是输入的通道数（输入深度），$D_G$是正方形输出特征图的空间高度和宽度，$N$是输出的通道数（输出深度）。

The standard convolutional layer is parameterized by convolution kernel **K** of size $D_K ×D_K × M × N$ where $D_K$ is the spatial dimension of the kernel assumed to be square and $M$ is number of input channels and $N$ is the number of output channels as defined previously.

标准卷积层的参数为卷积核**K**，大小为$D_K ×D_K × M × N$，其中$D_K$是卷积核的空间维度的大小，并假设是方形的，$M$是输入通道数，$N$是输出通道数。

The output feature map for standard convolution assuming stride one and padding is computed as:

标准卷积层的输出特征图（假设卷积步长stride为1，并有padding）计算为：

$${\bold G}_{k,l,n}=\sum_{i,j,m}{\bold K}_{i,j,m,n}\cdot{\bold F}_{k+i-1,l+j-1,m}$$(1)

Standard convolutions have the computational cost of:

标准卷积层的计算量为：

$$D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$(2)

where the computational cost depends multiplicatively on the number of input channels $M$, the number of output channels $N$ the kernel size $D_k × D_k$ and the feature map size $D_F × D_F$ . MobileNet models address each of these terms and their interactions. First it uses depthwise separable convolutions to break the interaction between the number of output channels and the size of the kernel.

这里计算量乘性依赖于输入通道数量$M$，输出通道数量$N$，卷积核大小$D_k × D_k$，特征图大小$D_F × D_F$。MobileNet模型会处理每一项以及它们的相互作用。首先模型用depthwise separable convolutions来使输出通道数量与卷积核大小没有相互作用。

The standard convolution operation has the effect of filtering features based on the convolutional kernels and combining features in order to produce a new representation. The filtering and combination steps can be split into two steps via the use of factorized convolutions called depthwise separable convolutions for substantial reduction in computational cost.

标准卷积操作有两种作用，一个是用卷积核对特征进行滤波，还有对特征进行组合，以产生新的表示。滤波步骤和组合步骤可以通过卷积分解分成两步，这种卷积分解叫做depthwise separable convolutions，这样计算量可以得到实质性的减少。

Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions. We use depthwise convolutions to apply a single filter per each input channel (input depth). Pointwise convolution, a simple 1×1 convolution, is then used to create a linear combination of the output of the depthwise layer. MobileNets use both batchnorm and ReLU nonlinearities for both layers.

Depthwise separable convolution由两层组成：depthwise convolutions和pointwise convolutions。我们用depthwise convolutions来对每个输入通道（输入深度）进行单独滤波。Pointwise convolution，就是简单的1×1卷积，用于对depthwise convolution层的输出进行线性组合，产生输出。MobileNet对两层分别进行batchnorm和ReLU非线性处理。

Depthwise convolution with one filter per input channel(input depth) can be written as:

对每个输入通道（输入深度）都有一个单独滤波器的Depthwise convolution可以写为：

$${\hat \bold G}_{k,l,m} = \sum_{i,j}{\hat \bold K}_{i,j,m}\cdot{\bold F_{k+i-1,l+j-1,m}}$$(3)

where ${\hat \bold K}$ is the depthwise convolutional kernel of size $D_K × D_K × M$ where the $m_{th}$ filter in ${\hat \bold K}$ is applied to the $m_{th}$ channel in **F** to produce the $m_{th}$ channel of the filtered output feature map ${\hat \bold G}$.

这里${\hat \bold K}$是depthwise卷积核，大小$D_K × D_K × M$，其中${\hat \bold K}$里的第*m*个滤波器作用到输入**F**的第*m*个通道，产生滤波过的输出特征图${\hat \bold G}$的第*m*个通道分量。

Depthwise convolution has a computational cost of:

Depthwise convolution的计算量为：

$$D_K \cdot D_K \cdot M  \cdot D_F \cdot D_F$$(4)

Depthwise convolution is extremely efficient relative to standard convolution. However it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise  convolution via 1 × 1 convolution is needed in order to generate these new features.

Depthwise convolution与标准卷积相比效率的得到极大提高，但是只对输入通道进行了滤波，没有将输出进行组合得到新特征，所以为了产生这些新特征，需要额外的一层来计算depthwise convolution的输出的线性组合，这是通过1 × 1卷积得到的。

The combination of depthwise convolution and 1 × 1 (pointwise) convolution is called depthwise separable convolution which was originally introduced in [26].

Depthwise convolution和1 × 1 (pointwise) convolution的组合，称为depthwise separable convolution，[26]最先提出了这种计算方法。

Depthwise separable convolutions cost: 计算量

$$D_K \cdot D_K \cdot M  \cdot D_F \cdot D_F + M  \cdot N \cdot D_F \cdot D_F$$(5)

which is the sum of the depthwise and 1×1 pointwise convolutions.

也就是depthwise和pointwise convolution两层的计算量之和。

By expressing convolution as a two step process of filtering and combining we get a reduction in computation of:

通过将卷积分解为滤波和组合两步，计算量缩减为：

$$\frac{D_K \cdot D_K \cdot M  \cdot D_F \cdot D_F + M  \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F}=\frac{1}{N}+\frac{1}{D_K^2}$$

MobileNet uses 3 × 3 depthwise separable convolutions which uses between 8 to 9 times less computation than standard convolutions at only a small reduction in accuracy as seen in Section 4.

MobileNet使用3 × 3 depthwise separable convolutions，其计算量为标准卷积的1/9到1/8，在第四部分的实验中可以看到，其准确度只有很小的下降。

Additional factorization in spatial dimension such as in [16, 31] does not save much additional computation as very little computation is spent in depthwise convolutions.

空间维度的额外分解，如[16,31]，并没有节省很多额外的计算量，这是因为depthwise convolutions的计算量已经很小了。

![Image](https://www.gitee.com/sample.png)

### 3.2. Network Structure and Training 网络结构和训练

The MobileNet structure is built on depthwise separable convolutions as mentioned in the previous section except for the first layer which is a full convolution. By defining the network in such simple terms we are able to easily explore network topologies to find a good network. The MobileNet architecture is defined in Table 1. All layers are followed by a batchnorm [13] and ReLU nonlinearity with the exception of the final fully connected layer which has no nonlinearity and feeds into a softmax layer for classification. Figure 3 contrasts a layer with regular convolutions, batchnorm and ReLU nonlinearity to the factorized layer with depthwise convolution, 1 × 1 pointwise convolution as well as batchnorm and ReLU after each convolutional layer. Down sampling is handled with strided convolution in the depthwise convolutions as well as in the first layer. A final average pooling reduces the spatial resolution to 1 before the fully connected layer. Counting depthwise and pointwise convolutions as separate layers, MobileNet has 28 layers.

如上一节所讲，MobileNet的结构是建立在depthwise separable convolutions的基础上，但第一层是个例外，这是一个全卷积层。网络定义在这些简单单元之上，我们就可以很容易的探索网络拓扑结构，来找到一个好的网络。MobileNet架构定义于表格1。所有层后都有一个BatchNorm[13]和ReLU非线性处理，除了最后一个全连接层是个例外，没有非线性处理，其结果输入到一个softmax层，进行分类。图3对比了常规卷积层和分解后的层，左边是带有batchnorm和ReLU非线性处理的常规卷积层，右边是包括depthwise convolution, 1 × 1 pointwise convolution和每层后的batchnorm and ReLU非线性化处理，这是常规卷积分解后的层。下采样是通过增加卷积步长的depthwise convolutions得到的，第一层也是这样的。最终的平均池化将空间分辨率降为1，随后是全连接层。将depthwise和pointwise卷积层算作单独的层，MobileNet共有28层。

Table 1. MobileNet Body Architecture 主体结构

Type/Stride  | Filter Shape      | Input Size
---          | ---               | ---
Conv / s2    | 3 × 3 × 3 × 32    | 224 × 224 × 3
Conv dw / s1 | 3 × 3 × 32 dw     | 112 × 112 × 32
Conv / s1    | 1 × 1 × 32 × 64   | 112 × 112 × 32
Conv dw / s2 | 3 × 3 × 64 dw     | 112 × 112 × 64
Conv / s1    | 1 × 1 × 64 × 128  | 56 × 56 × 64
Conv dw / s1 | 3 × 3 × 128 dw    | 56 × 56 × 128
Conv / s1    | 1 × 1 × 128 × 128 | 56 × 56 × 128
Conv dw / s2 | 3 × 3 × 128 dw    | 56 × 56 × 128
Conv / s1    | 1 × 1 × 128 × 256 | 28 × 28 × 128
Conv dw / s1 | 3 × 3 × 256 dw    | 28 × 28 × 256
Conv / s1    | 1 × 1 × 256 × 256 | 28 × 28 × 256
Conv dw / s2 | 3 × 3 × 256 dw    | 28 × 28 × 256
Conv / s1    | 1 × 1 × 256 × 512 | 14 × 14 × 256
5× Conv dw / s1 | 3 × 3 × 512 dw | 14 × 14 × 512
5× Conv / s1 | 1 × 1 × 512 × 512 | 14 × 14 × 512
Conv dw / s2 | 3 × 3 × 512 dw    | 14 × 14 × 512
Conv / s1    | 1 × 1 × 512 × 1024| 7 × 7 × 512
Conv dw / s2 | 3 × 3 × 1024 dw   |7 × 7 × 1024
Conv / s1    | 1 × 1 × 1024 × 1024 | 7 × 7 × 1024
Avg Pool / s1 | Pool 7 × 7       | 7 × 7 × 1024
FC / s1      | 1024 × 1000       | 1 × 1 × 1024
Softmax / s1 | Classifier        | 1 × 1 × 1000

3×3 conv -> BN -> ReLU

3×3 depthwise conv -> BN -> ReLU -> 1×1 conv -> BN -> ReLU

Figure 3 Standard convolution layer and Depthwise separable convolution layer 

It is not enough to simply define networks in terms of a small number of Mult-Adds. It is also important to make sure these operations can be efficiently implementable. For instance unstructured sparse matrix operations are not typically faster than dense matrix operations until a very high level of sparsity. Our model structure puts nearly all of the computation into dense 1×1 convolutions. This can be implemented with highly optimized general matrix multiply(GEMM) functions. Often convolutions are implemented by a GEMM but require an initial reordering in memory called im2col in order to map it to a GEMM. For instance, this approach is used in the popular Caffe package [15]. 1×1 convolutions do not require this reordering in memory and can be implemented directly with GEMM which is one of the most optimized numerical linear algebra algorithms. MobileNet spends 95% of it’s computation time in 1 × 1 convolutions which also has 75% of the parameters as can be seen in Table 2. Nearly all of the additional parameters are in the fully connected layer.

仅仅用用较少数量的乘法、加法运算定义了网络是不够的。同样重要的是，确保这些操作是可以高效实现的。例如，结构不规律的稀疏矩阵操作并不比稠密矩阵操作快的不很明显，只有稀疏性程度很高才会明显加快。我们模型的结构几乎所有的计算量都在稠密的1×1卷积上，这可以通过高度优化过的GEMM(通用矩阵乘法, general matrix multiply)函数实现。通常卷积是通过GEMM实现的，但开始时需要在内存中重新排序，函数是im2col，这样可以映射到一个GEMM。比如，这种方法在Caffe库[15]中就有应用。1×1卷积不需要在内存中重新排序，可以直接用GEMM实现，这种线性代数算法得到了最大程度的优化。MobileNet在1×1卷积上耗费了95%的计算时间，如表2所示，这其中包括了75%的参数，几乎所有剩余的参数都在全连接层。

Table 2. Resource Per Layer Type 每种卷积层的资源分布

Type            | Mult-Adds | Parameters
---             | ---       | ---
Conv 1 × 1      | 94.86%    | 74.59%
Conv DW 3 × 3   | 3.06%     | 1.06%
Conv 3 × 3      | 1.19%     | 0.02%
Fully Connected | 0.18%     | 24.33%

MobileNet models were trained in TensorFlow [1] using RMSprop [33] with asynchronous gradient descent similar to Inception V3 [31]. However, contrary to training large models we use less regularization and data augmentation techniques because small models have less trouble with overfitting. When training MobileNets we do not use side heads or label smoothing and additionally reduce the amount image of distortions by limiting the size of small crops that are used in large Inception training [31]. Additionally, we found that it was important to put very little or no weight decay (L2 regularization) on the depthwise filters since there are so few parameters in them. For the ImageNet benchmarks in the next section all models were trained with same training parameters regardless of the size of the model.

MobileNet模型在TensorFlow[1]中训练，用的是带有异步梯度下降的RMSprop[33]，这与Inception V3[31]类似。但是，与训练大型模型不同，我们很少用正则化和数据增强技术，因为小型模型的过拟合问题比较少。当训练MobileNet时，我们不需要使用side heads或者label smoothing，在大型Inception[31]训练中，通过限制小规模裁剪来减少扭曲图像的数量进行正则化，在MobileNet训练中也不需要。另外，我们发现，在depthwise滤波器中，只需要极少或不需要权值衰减(weight decay, L2正则化)，因为其中参数非常少。下一节中的ImageNet基准测试中，所有模型都用了相同的参数进行训练，参数与模型的规模无关。

### 3.3. Width Multiplier: Thinner Models

Although the base MobileNet architecture is already small and low latency, many times a specific use case or application may require the model to be smaller and faster. In order to construct these smaller and less computationally expensive models we introduce a very simple parameter *α* called width multiplier. The role of the width multiplier *α* is to thin a network uniformly at each layer. For a given layer and width multiplier *α*, the number of input channels *M* becomes *αM* and the number of output channels *N* becomes *αN*.

尽管基本的MobileNet架构已经属于小型、低延迟的，但很多场景下、应用中还是可能需要更小更快的模型。为了建构这种模型更小、计算量更少的模型，我们引入了一个很简单的参数*α*，称为width multiplier。其角色是对网络的每一层进行一致瘦身。对一给定层和width multiplier *α*，输入通道数量*M* 变为*αM*，输出通道数量*N*变为*αN*。

The computational cost of a depthwise separable convolution with width multiplier *α* is:

带有width multiplier *α*的depthwise separable convolution其计算量为：

$$D_K \cdot D_K \cdot \alpha M  \cdot D_F \cdot D_F + \alpha M  \cdot \alpha N \cdot D_F \cdot D_F$$(6)

where *α* ∈ (0,1] with typical settings of 1, 0.75, 0.5 and 0.25. *α* = 1 is the baseline MobileNet and *α* < 1 are reduced MobileNets. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly $α^2$ . Width multiplier can be applied to any model structure to define a new smaller model with a reasonable accuracy, latency and size trade off. It is used to define a new reduced structure that needs to be trained from scratch.

这里*α* ∈ (0,1]，典型值为1，0.75，0.5，0.25。*α* = 1是标准MobileNet，*α* < 1是退化MobileNets。Width multiplier可以减少运算量，并以大致$α^2$的速度减少参数数量。width multiplier可以应用在任何模型结构中，并定义出更小的模型，其准确度、延迟、规模都还折中的比较合理。由此得到的新退化结构模型需要重新训练。

### 3.4. Resolution Multiplier: Reduced Representation

The second hyper-parameter to reduce the computational cost of a neural network is a resolution multiplier *ρ*. We apply this to the input image and the internal representation of every layer is subsequently reduced by the same multiplier. In practice we implicitly set *ρ* by setting the input resolution.

第二个可以减少神经网络运算量的超参数是resolution multiplier *ρ*，我们将其应用于输入图像，那么随后每一层的内部表示都被相同的乘子缩减。在实践中我们通过设定输入图像的分辨率，也就隐式的设定了参数*ρ*。

We can now express the computational cost for the core layers of our network as depthwise separable convolutions with width multiplier *α* and resolution multiplier *ρ*:

我们现在可以将带有width multiplier *α*和resolution multiplier *ρ*的网络核心层，即depth separable convolution，的计算量表示如下：

$$D_K \cdot D_K \cdot \alpha M  \cdot \rho D_F \cdot \rho D_F + \alpha M  \cdot \alpha N \cdot \rho D_F \cdot \rho D_F$$(7)

where *ρ* ∈ (0,1] which is typically set implicitly so that the input resolution of the network is 224, 192, 160 or 128. ρ = 1 is the baseline MobileNet and ρ < 1 are reduced computation MobileNets. Resolution multiplier has the effect of reducing computational cost by $ρ^2$.

这里*ρ* ∈ (0,1]，通常其值设置后使输入图像分辨率为224、192、160或128。ρ = 1是标准MobileNet，ρ < 1时的MobileNets计算量降低。Resolution multiplier可以将计算量按照$ρ^2$速度降低。

As an example we can look at a typical layer in MobileNet and see how depthwise separable convolutions, width multiplier and resolution multiplier reduce the cost and parameters. Table 3 shows the computation and number of parameters for a layer as architecture shrinking methods are sequentially applied to the layer. The first row shows the Mult-Adds and parameters for a full convolutional layer with an input feature map of size 14×14×512 with a kernel **K** of size 3 × 3 × 512 × 512. We will look in detail in the next section at the trade offs between resources and accuracy.

在表3中我们可以看到一个典型的网络层，在depthwise separable convolution, width multiplier, resolution multiplier的影响下如何减少计算量和参数数量的。第一行是一个全卷积层的乘法加法数和参数数量，输入特征图的大小是14×14×512，卷积核**K**的大小是3 × 3 × 512 × 512。下一节中我们会更详细的看到，资源和准确度是如何折中的。

Table 3. Resource usage for modifications to standard convolution. Note that each row is a cumulative effect adding on top of the previous row. This example is for an internal MobileNet layer with $D_K$ = 3, *M* = 512, *N* = 512, $D_F$ = 14. M for million.

表3 标准卷积及改进后的资源使用情况，注意每一行的效果是叠加在前一行的，例子中是MobileNet的内部层，参数为$D_K$ = 3, *M* = 512, *N* = 512, $D_F$ = 14

Layer/Modification | Mult-Adds | Parameters 
---                | ---       | ---
Convolution        | 462M      | 2.36M
Depthwise Separable Conv | 52.3M | 0.27M
*α* = 0.75         | 29.6M     | 0.15M
*ρ* = 0.714        | 15.1M     | 0.15M

## 4. Experiments 试验

In this section we first investigate the effects of depthwise convolutions as well as the choice of shrinking by reducing the width of the network rather than the number of layers. We then show the trade offs of reducing the network based on the two hyper-parameters: width multiplier and resolution multiplier and compare results to a number of popular models. We then investigate MobileNets applied to a number of different applications.

在本节中我们首先探讨一下depthwise convolution的效果，也看看通过减少网络宽度来对收缩模型的影响，减少网络深度及层数就不用研究了。然后我们再看一下通过两个超参数，width multiplier和resolution multiplier，来使模型收缩的代价，并与一些受欢迎的模型进行比较。最后我们研究在一些不同场景中的MobileNet应用情况。

### 4.1. Model Choices 模型选择

First we show results for MobileNet with depthwise separable convolutions compared to a model built with full convolutions. In Table 4 we see that using depthwise separable convolutions compared to full convolutions only reduces accuracy by 1% on ImageNet was saving tremendously on
mult-adds and parameters.

首先我们将depthwise separable convolution的MobileNet的结果与全卷积的模型相比，在表4中我们可以看到，使用depthwise separable convolutions的MobileNet算法在ImageNet上的准确度，与全卷积网络相比，只减少了1%，但计算量和参数数量却急剧减少。

Table 4. Depthwise Separable vs Full Convolution MobileNet

Model | Accuracy | Mult-Adds | Parameters
--- | --- | --- | ---
Conv MobileNet | 71.7% | 4866 | 29.3
MobileNet | 70.6% | 569 | 4.2

We next show results comparing thinner models with width multiplier to shallower models using less layers. To make MobileNet shallower, the 5 layers of separable filters with feature size 14 × 14 × 512 in Table 1 are removed. Table 5 shows that at similar computation and number of
parameters, that making MobileNets thinner is 3% better than making them shallower.

然后我们将使用width multiplier的瘦模型结果，与使用较少层的浅模型进行比较。将MobileNet变浅的方法，是将表1中的5层separable filters(特征尺寸14 × 14 × 512)移除掉，表5说明，计算量和参数数量都相似，但瘦MobileNet模型比浅模型准确率高3%。

Table 5. Narrow vs Shallow MobileNet

Model | Accuracy | Mult-Adds | Parameters
--- | --- | --- | ---
0.75 MobileNet | 68.4% | 325 | 2.6
Shallow MobileNet | 65.3% | 307 | 2.9

### 4.2. Model Shrinking Hyperparameters

