# Very Deep Convolutional Networks for Large-Scale Image Recognition
# 大规模图像识别应用中非常深的卷积网络

Karen Simonyan & Andrew Zisserman Visual Geometry Group, Department of Engineering Science, University of Oxford

## ABSTRACT

In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

本文我们研究了大规模图像识别应用中卷积神经网络的深度对准确率的影响。我们的主要贡献是使用非常小的卷积核(3×3)组成的架构彻底对网络深度进行了评估，结果显示如果带权重的层增加到16-19个，在现有配置下，网络性能会有明显提升。以这些发现为基础，我们向2014年ImageNet挑战赛提交了自己的模型，我们的团队在定位任务中得到了第一的位置，在分类任务中得到了第二的成绩。我们的模型泛化能力也非常好，实际上是目前最好的结果。我们公布了两个最好的卷积网络，为进一步在计算机视觉中研究深度视觉表示提供便利。

## 1 I NTRODUCTION

Convolutional networks (ConvNets) have recently enjoyed a great success in large-scale image and video recognition (Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014; Simonyan & Zisserman, 2014) which has become possible due to the large public image repositories, such as ImageNet(Deng et al., 2009), and high-performance computing systems, such as GPUs or large-scale distributed clusters (Dean et al., 2012). In particular, an important role in the advance of deep visual recognition architectures has been played by the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) (Russakovsky et al., 2014), which has served as a testbed for a few generations of large-scale image classification systems, from high-dimensional shallow feature encodings (Perronnin et al., 2010) (the winner of ILSVRC-2011)to deep ConvNets (Krizhevsky et al., 2012) (the winner of ILSVRC-2012).

卷积网络(ConvNets)近些年在大规模图像和视频识别非常成功(Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014; Simonyan & Zisserman, 2014)，这是因为有了大规模图像数据集如ImageNet(Deng et al., 2009)，和高性能计算系统如GPU和大规模分布式集群(Dean et al., 2012)。特别是ILSVRC(Russakovsky et al., 2014)在深度视觉识别的发展中起到了很重要的作用，几代大规模图像分类系统都是以它为试验台发展起来的，从ILSVRC-2011冠军的高维度浅层特征编码(Perronnin et al., 2010)到ILSVRC-2012冠军的深度卷积网络(Krizhevsky et al., 2012)。

With ConvNets becoming more of a commodity in the computer vision field, a number of attempts have been made to improve the original architecture of Krizhevsky et al. (2012) in a bid to achieve better accuracy. For instance, the best-performing submissions to the ILSVRC-2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014) utilised smaller receptive window size and smaller stride of the first convolutional layer. Another line of improvements dealt with training and testing the networks densely over the whole image and over multiple scales (Sermanet et al., 2014; Howard, 2014). In this paper, we address another important aspect of ConvNet architecture design – its depth. To this end, we fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small (3 × 3) convolution filters in all layers.

随着卷积网络在计算机视觉领域变得日益普遍，有一些对原版Krizhevsky et al. (2012)网络架构进行改进的尝试，以获得更好的准确度。比如，ILSVRC-2013冠军得主(Zeiler & Fergus, 2013; Sermanet et al., 2014)使用了小一些的感受野窗口和小一些的卷积步长（在第1个卷积层中）。另一个改进的思路想要解决在整个图像上多尺度密集训练测试网络(Sermanet et al., 2014; Howard, 2014)。本文中，我们探讨了卷积网络架构设计问题的另一个重要方面，也就是深度。为了这个目的，我们解决架构的其他参数问题，然后稳步增加更多的卷积层以增加网络深度，我们在所有层中都用了很小的卷积滤波器(3×3)，所以是可行的。

As a result, we come up with significantly more accurate ConvNet architectures, which not only achieve the state-of-the-art accuracy on ILSVRC classification and localisation tasks, but are also applicable to other image recognition datasets, where they achieve excellent performance even when used as a part of a relatively simple pipelines (e.g. deep features classified by a linear SVM without fine-tuning). We have released our two best-performing models (http://www.robots.ox.ac.uk/~vgg/research/very_deep/) to facilitate further research.

结果我们得到了准确度高的多的ConvNet架构，在ILSVRC分类和定位任务中取得了目前最好的准确率，还可以用于其他识别数据集，即使在一个简单结构的一部分，也可以取得非常好的结果（如将深度特征用线性SVM分类，不精调）。我们公开了两个最好的模型为进一步的研究提供便利(http://www.robots.ox.ac.uk/~vgg/research/very_deep/)。

The rest of the paper is organised as follows. In Sect. 2, we describe our ConvNet configurations. The details of the image classification training and evaluation are then presented in Sect. 3, and the configurations are compared on the ILSVRC classification task in Sect. 4. Sect. 5 concludes the paper. For completeness, we also describe and assess our ILSVRC-2014 object localisation system in AppendixA, and discuss the generalisation of very deep features to other datasets in AppendixB. Finally, Appendix C contains the list of major paper revisions.

文章剩余部分组织如下：在第2部分中，我们列出了卷积网络的配置，第3部分给出了图像分类训练和测试的细节，第4部分在ILSVRC分类任务中进行了比较，第5部分进行了总结，附录A中评价了我们的ILSVRC目标定位系统，附录B中讨论了深度特征在其他数据集上的泛化，附录C包括了文章的主要修改过程。

## 2 CONVNET CONFIGURATIONS

To measure the improvement brought by the increased ConvNet depth in a fair setting, all our ConvNet layer configurations are designed using the same principles, inspired by Ciresan et al. (2011); Krizhevsky et al. (2012). In this section, we first describe a generic layout of our ConvNet configurations(Sect.2.1) and then detail the specific configurations used in the evaluation (Sect.2.2). Our design choices are then discussed and compared to the prior art in Sect. 2.3.

为了公平的度量增加卷积网络深度带来的改进，受Ciresan et al. (2011)、Krizhevsky et al. (2012)启发，我们的卷积网络层的配置设计时都用了同样的原则。在本节中，我们首先描述了卷积网络配置的一般性结构(2.1节)，然后详细叙述了评估中用到的特定配置(2.2节)。2.3节讨论了我们的设计选择并与以前的技术进行了比较。

### 2.1 ARCHITECTURE

During training, the input to our ConvNets is a fixed-size 224 × 224 RGB image. The only preprocessing we do is subtracting the mean RGB value, computed on the training set, from each pixel. The image is passed through a stack of convolutional(conv.) layers, where we use filters with a very small receptive field: 3 × 3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations we also utilise 1 × 1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv.layer input is such that the spatial resolutionis preserved after convolution, i.e. the padding is 1 pixel for 3 × 3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv. layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2 × 2 pixel window, with stride 2.

训练期间，网络输入是固定的224×224 RGB图像，唯一的预处理是在整个训练集上减去了每个像素的RGB均值。输入图像经过了一连串卷积层，其中使用的滤波器感受野非常小，只有3×3。其中一处我们还使用了1×1卷积核，可以看做是输入通道的线性变换（后面是非线性处理）。卷积步长固定为1；卷积层输入的填充原则是卷积输出的尺寸不改变大小，即，对于3×3卷积填充是1个像素。一些卷积层后有pooling层，共有5个。max-pooling操作在2×2像素窗口，步长为2。

A stack of convolutional layers (which has a different depth in different architectures)is followed by three Fully-Connected (FC) layers: the first two have 4096 channels each, the third performs 1000-way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.

卷积层的叠加（不同的架构有不同的深度）后是3个全连接层：前两层各有4096个通道，第三层进行ILSVRC 1000类的分类，所以有1000个通道（每类一个）。最后是softmax层。全连接层在所有网络里配置都是一样的。

All hidden layers are equipped with the rectification (ReLU (Krizhevsky et al., 2012)) non-linearity. We note that none of our networks (except for one) contain Local Response Normalisation (LRN) normalisation (Krizhevsky et al., 2012): as will be shown in Sect. 4, such normalisation does not improve the performance on the ILSVRC dataset, but leads to increased memory consumption and computation time. Where applicable, the parameters for the LRN layer are those of (Krizhevsky et al., 2012).

所有的隐藏层都进行了ReLU非线性处理，我们的网络只有1个包含局部响应归一化(LRN)，将在第4部分看到，这种归一化不会改进ILSVRC上的准确率，但是内存使用和计算时间都会增加。应用的LRN的网络其参数是从(Krizhevsky et al., 2012)得到的。

### 2.2 CONFIGURATIONS

The ConvNet configurations, evaluated in this paper, are outlined in Table 1, one per column. In the following we will refer to the nets by their names (A–E). All configurations follow the generic design presented in Sect. 2.1, and differ only in the depth: from 11 weight layers in the network A (8 conv.and 3 FC layers) to 19 weight layers in the network E (16 conv.and 3 FC layers). The width of conv. layers (the number of channels) is rather small, starting from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer, until it reaches 512.

本文卷积网络的配置见表1，每个模型一列。下文中我们用网络的名字A-E进行指代。所有的配置都遵循2.1节中的一般设计原则，不同的只有深度，从网络A的11层（8卷积层3全连接层）到网络E的19层（16卷积层3全连接层）。卷积层的宽度（通道数）非常小，从第1层的64，每过一个max-pooling层增加1倍，一直到512通道。

Table 1: ConvNet configurations (shown in columns). The depth of the configurations increases from the left (A) to the right (E), as more layers are added (the added layers are shown in bold). The convolutional layer parameters are denoted as “conv < receptive field size > - < number of channels >”. The ReLU activation function is not shown for brevity.

表1 卷积网络配置（每列1个模型），从A到E配置的层数逐渐增加，卷积层的参数用conv感受野大小 - 通道数来表示。为简洁，ReLU激活函数没有显示。

A | A-LRN | B | C | D | E
--- | --- | --- | --- | --- | ---
11 weight layers | 11 weight layers | 13 weight layers | 16 weight layers | 16 weight layers | 19 weight layers
| input | 224 × 224 | RGB image |
conv3-64 | conv3-64 | conv3-64 | conv3-64 | conv3-64 | conv3-64
| | LRN | conv3-64 | conv3-64 | conv3-64 | conv3-64
| maxpool |
conv3-128 | conv3-128 | conv3-128 | conv3-128 | conv3-128 | conv3-128
| | | conv3-128 | conv3-128 | conv3-128 | conv3-128
| maxpool |
conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256
conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256 | conv3-256
| | | | conv1-256 | conv3-256 | conv3-256
| | | | | | conv3-256
| maxpool| 
conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512
conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512
| | | | conv1-512 | conv3-512 | conv3-512
| | | | | | conv3-512
| maxpool |
conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512
conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512 | conv3-512
| | | | conv1-512 | conv3-512 | conv3-512
| | | | | | conv3-512
| maxpool |
| FC-4096 |
| FC-4096 |
| FC-1000 |
| soft-max |

In Table 2 we report the number of parameters for each configuration. In spite of a large depth, the number of weights in our nets is not greater than the number of weights in a more shallow net with larger conv. layer widths and receptive fields (144M weights in (Sermanet et al., 2014)).

在表2中列出了每种配置的参数数量。尽管层数众多，与那些更浅但宽度更大感受野更大的网络比，我们网络中权重的数量并不多，比如(Sermanet et al., 2014)的网络中有1.44亿个参数。

Table 2: Number of parameters 参数数量(亿)

Network | A,A-LRN | B | C | D | E
--- | --- | --- | --- | --- | ---
Number of parameters | 1.33 | 1.33 | 1.34 | 1.38 | 1.44

### 2.3 DISCUSSION

Our ConvNet configurations are quite different from the ones used in the top-performing entries of the ILSVRC-2012 (Krizhevsky et al., 2012) and ILSVRC-2013 competitions (Zeiler & Fergus, 2013; Sermanet et al., 2014). Rather than using relatively large receptive fields in the first conv. layers (e.g. 11×11with stride 4 in (Krizhevsky et al., 2012), or 7×7 with stride 2 in (Zeiler & Fergus, 2013; Sermanet et al., 2014)), we use very small 3 × 3 receptive fields throughout the whole net,which are convolved with the input at every pixel (with stride 1). It is easy to see that a stack of two 3×3 conv.layers (without spatial pooling in between) has an effective receptive field of 5×5; three such layers have a 7 × 7 effective receptive field. So what have we gained by using, for instance, a stack of three 3×3 conv.layers instead of a single 7×7 layer? First, we incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative. Second, we decrease the number of parameters: assuming that both the input and the output of a three-layer 3 × 3 convolution stack has C channels, the stack is parametrised by 3($3^2 C^2$) = 27 $C^2$ weights; at the same time, a single 7 × 7 conv. layer would require $7^2 C^2 = 49C^2$ parameters, i.e. 81% more. This can be seen as imposing a regularisation on the 7 × 7 conv. filters, forcing them to have a decomposition through the 3 × 3 filters (with non-linearity injected in between).

我们的ConvNet配置与ILSVRC-2012的最佳得主(Krizhevsky et al., 2012)和ILSVRC-2013最佳得主(Zeiler & Fergus, 2013; Sermanet et al., 2014)很不一样。(Krizhevsky et al., 2012)在第一层使用了较大的感受野（即滤波器，11×11，步长4），
(Zeiler & Fergus, 2013; Sermanet et al., 2014)为7×7，步长2，我们在全网中都使用了很小的滤波器，3×3，步长1。容易理解，2层3×3的滤波器（中间没有pooling）的感受野与5×5是一样的，3层就是7×7。那么我们用3层3×3的来替代7×7的有什么别的益处呢？首先，可以使用3个非线性ReLU层，这使判别函数更有区分力；第二，减少了参数数量。假设输入输出层都有C个通道，那么3层累加的参数为3($3^2 C^2$) = 27 $C^2$，同时7×7的层参数数量为$7^2 C^2 = 49C^2$，多了81%。这可以看做是在7×7卷积层增加了正则化，使其分解为3个3×3的滤波器（中间还有非线性处理）。

The incorporation of 1 × 1 conv. layers (configuration C, Table 1) is a way to increase the non-linearity of the decision function without affecting the receptive fields of the conv. layers. Even though in our case the 1×1 convolution is essentially a linear projection onto the space of the same dimensionality (the number of input and output channels is the same), an additional non-linearity is introduced by the rectification function. It should be noted that 1×1 conv. layers have recently been utilised in the “Network in Network” architecture of Lin et al. (2014).

采用1×1卷积层（表1中C模型）是一种增加判别函数非线性的方式，而同时又不影响卷积层的感受野。我们的模型中1×1卷积实际上是向相同维度空间的线性映射（输入输出的通道数一样），额外的非线性是由ReLU函数带来的。Lin et al. (2014)在Network in Network架构中也使用了1×1卷积层。

Small-size convolution filters have been previously used by Ciresan et al. (2011), but their nets are significantly less deep than ours, and they did not evaluate on the large-scale ILSVRC dataset. Goodfellow et al. (2014) applied deep ConvNets (11 weight layers) to the task of street number recognition, and showed that the increased depth led to better performance. GoogLeNet (Szegedy et al., 2014), a top-performing entry of the ILSVRC-2014 classification task, was developed independently of our work, but is similar in that it is based on very deep ConvNets (22 weight layers) and small convolution filters (apart from 3 × 3, they also use 1 × 1 and 5 × 5 convolutions). Their network topology is, however, more complex than ours, and the spatial resolution of the feature maps is reduced more aggressively in the first layers to decrease the amount of computation. As will be shown in Sect. 4.5, our model is outperforming that of Szegedy et al. (2014) in terms of the single-network classification accuracy.

Ciresan et al. (2011)使用过小型卷积核，但是其网络比我们的要浅的多，也没用用在ImageNet数据集上。Goodfellow et al. (2014)将深层网络（11层）用在了街道编号识别上，结果显示深度增加改进了性能。GoogLeNet (Szegedy et al., 2014)在ILSVRC-2014分类任务中表现最好，其深度也增加很多（22个权重层），使用的卷积滤波器也很小（除了3×3，还有1×1和5×5）。其网络拓扑比我们的要复杂很多，特征图的空间分辨率在第1层减少很多，是为了降低计算量。如我们在4.5节中看到的，我们的模型比Szegedy et al. (2014)在单个网络的分类准确率上表现要好。

## 3 CLASSIFICATION FRAMEWORK

In the previous section we presented the details of our network configurations. In this section, we describe the details of classification ConvNet training and evaluation.

前一节中，我们详述了网络的配置。这一节中，我们详述我们网络的分类训练和评估。

### 3.1 TRAINING

The ConvNet training procedure generally follows Krizhevsky et al. (2012) (except for sampling the input crops from multi-scale training images, as explained later). Namely, the training is carried out by optimising the multinomial logistic regression objective using mini-batch gradient descent (based on back-propagation (LeCun et al., 1989)) with momentum. The batch size was set to 256, momentum to 0.9. The training was regularised by weight decay (the L2 penalty multiplier set to 5·$10^{−4}$ ) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5). The learning rate was initially set to $10^{−2}$ , and then decreased by a factor of 10 when the validation set accuracy stopped improving. In total, the learning rate was decreased 3 times, and the learning was stopped after 370K iterations (74 epochs). We conjecture that in spite of the larger number of parameters and the greater depth of our nets compared to (Krizhevsky et al., 2012), the nets required less epochs to converge due to (a) implicit regularisation imposed by greater depth and smaller conv. filter sizes; (b) pre-initialisation of certain layers.

卷积网络训练过程一般按照Krizhevsky et al. (2012)进行，除了从多尺度的训练图像中进行输入块的取样，这将在后面解释。进行训练，就是用带有动量的mini-batch梯度下降法最优化多项式logistic回归目标函数。每批包含256个样本，动量0.9。训练正则化方法为权重衰减（L2惩罚因子设为5·$10^{−4}$），前两层全卷积层采用dropout正则化，dropout ratio为0.5。学习率初始值设为$10^{−2}$，当验证集准确率不再改进时，就除以10。我们推测，即使我们的网络比(Krizhevsky et al., 2012)参数要多，深度要深，收敛的epoch数要少，因为(a)更深层网络和更小的卷积核带来的隐藏正则化作用；(b)一些层的预初始化。

The initialisation of the network weights is important, since bad initialisation can stall learning due to the instability of gradient in deep nets. To circumvent this problem, we began with training the configuration A (Table 1), shallow enough to be trained with random initialisation. Then, when training deeper architectures, we initialised the first four convolutional layers and the last three fully-connected layers with the layers of net A (the intermediate layers were initialised randomly). We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning. For random initialisation (where applicable), we sampled the weights from a normal distribution with the zero mean and $10^{−2}$ variance. The biases were initialised with zero. It is worth noting that after the paper submission we found that it is possible to initialise the weights without pre-training by using the random initialisation procedure of Glorot & Bengio (2010).

网络权值的初始化是非常重要的，因为不良的初始化会使网络停止学习，因为深度网络的梯度不稳定。为防止这种问题，我们首先训练表1中的模型A，用随机初始化就可以训练这样浅的网络。然后，训练更深的架构时，我们把前4层卷积层和最后三个全连接层用网络A的参数初始化（其余中间层随机初始化）。对于预初始化层，我们没有降低学习速率，在训练中再改变其值。对于随机初始化，用的是零均值方差为0.01的正态分布随机数。偏置初始化为0。论文提交后我们发现，可以用Glorot & Bengio (2010)的随机初始化方法，而不用预初始化权值。

To obtain the fixed-size 224×224 ConvNet input images, they were randomly cropped from rescaled training images (one crop per image per SGD iteration). To further augment the training set, the crops underwent random horizontal flipping and random RGB colour shift (Krizhevsky et al., 2012). Training image rescaling is explained below.

为得到固定尺寸的224×224 ConvNets输入图像，将训练图像变换尺寸后，随机进行剪切（每个图像每次SGD迭代一个剪切块）。为进一步扩充训练集，这些剪切块还进行了随机的水平翻转，随机的RGB色彩变换(Krizhevsky et al., 2012)。训练图像变换尺寸介绍如下。

**Training image size**. Let S be the smallest side of an isotropically-rescaled training image, from which the ConvNet input is cropped (we also refer to S as the training scale). While the crop size is fixed to 224 × 224, in principle S can take on any value not less than 224: for S = 224 the crop will capture whole-image statistics, completely spanning the smallest side of a training image; for S ≫ 224 the  crop will correspond to a small part of the image, containing a small object or an object part.

**训练图像大小**。令S表示训练图像按比例缩放后的图像较小的那个边，当剪切块大小固定为224×224时，原则上S可以是不小于224的任意值，S=224时，剪切块就充满了图像的一边，在另一条边上滑动；S≫224时剪切块就只是图像的一小部分，包含一个小目标或目标的一部分。

We consider two approaches for setting the training scale S. The first is to fix S, which corresponds to single-scale training (note that image content within the sampled crops can still represent multi-scale image statistics). In our experiments, we evaluated models trained at two fixed scales: S = 256 (which has been widely used in the prior art (Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014)) and S = 384. Given a ConvNet configuration, we first trained the network using S = 256. To speed-up training of the S = 384 network, it was initialised with the weights pre-trained with S = 256, and we used a smaller initial learning rate of $10^{−3}$.

S取值的确定有两种方法。第一种是固定S，这对应着单尺度训练（取样后的剪切块内的图像内容仍然可以代表多尺度图像统计信息）。我们的试验中，评估了两个固定的S值，S=256和S=384, 256的值层被广泛使用(Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014)。给定一个ConvNet配置，我们首先用S=256训练网络。然后将训练结果作为S=384网络的初始值，进一步进行训练，学习速率初始值设定较小$10^{−3}$。

The second approach to setting S is multi-scale training, where each training image is individually rescaled by randomly sampling S from a certain range [$S_{min} ,S_{max}$ ] (we used $S_{min}$ = 256 and $S_{max}$ = 512). Since objects in images can be of different size, it is beneficial to take this into account during training. This can also be seen as training set augmentation by scale jittering, where a  single model is trained to recognise objects over a wide range of scales. For speed reasons, we trained multi-scale models by fine-tuning all layers of a single-scale model with the same configuration, pre-trained with fixed S = 384.

第二种方法是多尺度训练设定，每个训练图像改变大小时，S值从一个范围[$S_{min} ,S_{max}$]内随机选择，我们使用$S_{min}$ = 256, $S_{max}$ = 512。由于图像中的目标可能尺寸不一样，训练中将这一点考虑进来是有好处的。这也可以看作是通过尺度变化来扩充训练集，训练出一个模型可以识别不同尺度的目标。由于速度的原因，我们通过精调单尺度模型参数来训练多尺度模型，预训练的是S=384的模型。

### 3.2 TESTING

At test time, given a trained ConvNet and an input image, it is classified in the following way. First, it is isotropically rescaled to a pre-defined smallest image side, denoted as Q (we also refer to it as the test scale). We note that Q is not necessarily equal to the training scale S (as we will show in Sect. 4, using several values of Q for each S leads to improved performance). Then, the network is applied densely over the rescaled test image in a way similar to (Sermanet et al., 2014). Namely, the fully-connected layers are first converted to convolutional layers (the first FC layer to a 7 × 7 conv. layer, the last two FC layers to 1 × 1 conv. layers). The resulting fully-convolutional net is then applied to the whole (uncropped) image. The result is a class score map with the number of channels equal to the number of classes, and a variable spatial resolution, dependent on the input image size. Finally, to obtain a fixed-size vector of class scores for the image, the class score map is spatially averaged (sum-pooled). We also augment the test set by horizontal flipping of the images; the soft-max class posteriors of the original and flipped images are averaged to obtain the final scores for the image.

测试时，给定一个训练好的卷积网络和一个输入图像，分类过程是这样进行的。首先，将图像尺寸按比例缩放到预定义的短边大小，用Q表示（也称为测试尺度），Q不一定等于S（我们将在第4部分看到，对于每个S有几个值的Q可以改进性能）；然后，改变大小的测试图像经过网络，与(Sermanet et al., 2014)类似，即，全连接层首先转化为卷积层（第1全连接层转化为7×7的卷积网络，后两个全连接层转化为1×1的卷积网络），得到的全卷积网络应用在整个未剪切的图像。得到的结果是一个类别分数图，通道数与类别数相同，还有依赖于输入图像大小的可变的空间分辨率。最后，为了得到图像分类得分的固定大小向量，类别得分图在空间上进行了平均。我们对测试集也进行了扩充，通过图像水平翻转；原始图像和翻转图像的softmax分类结果进行平均，作为图像的最后得分。

Since the fully-convolutional network is applied over the whole image, there is no need to sample multiple crops at test time (Krizhevsky et al., 2012), which is less efficient as it requires network re-computation for each crop. At the same time, using a large set of crops, as done by Szegedy et al. (2014), can lead to improved accuracy, as it results in a finer sampling of the input image compared to the fully-convolutional net. Also, multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions: when applying a ConvNet to a crop, the convolved feature maps are padded with zeros, while in the case of dense evaluation the padding for the same crop naturally comes from the neighbouring parts of an image (due to both the convolutions and spatial pooling), which substantially increases the overall network receptive field, so more context is captured. While we believe that in practice the increased computation time of multiple crops does not justify the potential gains in accuracy, for reference we also evaluate our networks using 50 crops per scale (5×5 regular grid with 2 flips), for a total of 150 crops over 3 scales, which is comparable to 144 crops over 4 scales used by Szegedy et al. (2014).

由于是在整个图像上应用全卷积网络，所以没有必要在测试时剪切出多个图像块(Krizhevsky et al., 2012)，这样会导致效率低下，因为网络会为每个剪切块重新计算。同时，使用多个图像剪切块，就像Szegedy et al. (2014)那样，可以改进准确率，因为这与全卷积网络相比，可以对输入图像进行精细的取样。多块评估与稠密评估是互补的，因为它们卷积的边界条件不一样：当一个剪切块经过网络时，卷积的特征图用0补齐，稠密评估时，同样的剪切块其补齐是来自图像中的邻域（因为卷积和空间pooling），这基本上就是增加了网络的感受野，所以捕捉到了更多的上下文信息。我们相信，实践中多个剪切块计算增加的计算量不会带来相应的准确度提升，作为参考，我们还评估了网络在每个尺度上用50个剪切块进行试验（5×5的普通网格，2个翻转），3个尺度共150幅图像，Szegedy et al. (2014)使用的4个尺度144幅图。

### 3.3 IMPLEMENTATION DETAILS

Our implementation is derived from the publicly available C++ Caffe toolbox (Jia, 2013) (branched out in December 2013), but contains a number of significant modifications, allowing us to perform training and evaluation on multiple GPUs installed in a single system,as well as train and evaluate on full-size (uncropped) images at multiple scales (as described above). Multi-GPU training exploits data parallelism, and is carried out by splitting each batch of training images into several GPU batches, processed in parallel on each GPU. After the GPU batch gradients are computed, they are averaged to obtain the gradient of the full batch. Gradient computation is synchronous across the GPUs, so the result is exactly the same as when training on a single GPU.

我们的实现基于Caffe工具箱，但有一些较大的改动，使我们能在多个GPU上进行训练和评估，还有对全尺寸的多尺度图像进行训练和评估。多GPU训练利用数据并行性，通过将一个batch的训练图像分成几个GPU batch训练，在每个GPU上并行进行。在GPU上计算了batch梯度后，要进行平均，以得到整个batch的梯度。梯度计算在GPU间是同步的，所以结果和在一个GPU上进行一样。

While more sophisticated methods of speeding up ConvNet training have been recently proposed (Krizhevsky, 2014), which employ model and data parallelism for different layers of the net, we have found that our conceptually much simpler scheme already provides a speedup of 3.75 times on an off-the-shelf 4-GPU system, as compared to using a single GPU. On a system equipped with four NVIDIA Titan Black GPUs, training a single net took 2–3 weeks depending on the architecture.

(Krizhevsky, 2014)提出了很复杂的方法来加速卷积网络训练，在网络的不同层次都利用了模型并行性和数据并行性，而与单GPU系统相比，我们的简单方案在这个4 GPU系统上已经有了约3.75倍的加速。我们的系统装备了4个NVIDIA Tian Black GPU，训练一个网络大约花费2-3星期。

## 4 CLASSIFICATION EXPERIMENTS

**Dataset**. In this section, we present the image classification results achieved by the described ConvNet architectures on the ILSVRC-2012 dataset (which was used for ILSVRC 2012–2014 challenges). The dataset includes images of 1000 classes, and is split into three sets: training (1.3M images), validation (50K images), and testing (100K images with held-out class labels). The classification performance is evaluated using two measures: the top-1 and top-5 error. The former is a multi-class classification error, i.e. the proportion of incorrectly classified images; the latter is the main evaluation criterion used in ILSVRC, and is computed as the proportion of images such that the ground-truth category is outside the top-5 predicted categories.

**数据集**。本节中，我们用提出的卷积网络模型在ILSVRC-2012数据集（在ILSVRC 2012-2014年的挑战赛上使用）上进行分类试验。数据集包括1000类图像，分成3个集合，训练集（130万图像），验证集（5万图像），测试集（10万图像，带类别标签）。分类性能用两个指标来衡量，top-1和top-5错误率。top-1是多类分类错误，即错误分类的图像概率；top-5是ILSVRC使用的主要评估准则，计算的是前5个预测类别中没有真值类别的概率。

For the majority of experiments, we used the validation set as the test set. Certain experiments were also carried out on the test set and submitted to the official ILSVRC server as a “VGG” team entry to the ILSVRC-2014 competition (Russakovsky et al., 2014).

试验中主要把验证集当成测试集，一些试验也在测试集中进行了，提交给了ILSVRC官方服务器，名称为VGG。

### 4.1 SINGLE SCALE EVALUATION

We begin with evaluating the performance of individual ConvNet models at a single scale with the layer configurations described in Sect. 2.2. The test image size was set as follows: Q = S for fixed S, and Q = 0.5($S_{min} + S_{max}$) for jittered $S ∈ [S_{min} ,S_{max}]$. The results of are shown in Table 3.

我们用2.2节叙述的网络配置，在单个尺度上对单个卷积网络模型评估其性能。测试图像尺寸设置如下：对于固定的S，Q=S；对于不固定的$S ∈ [S_{min} ,S_{max}]$，Q = 0.5($S_{min} + S_{max}$)。结果如表3所示。

Table 3: ConvNet performance at a single test scale. 卷积网络模型在单测试尺度的表现

ConvNet | train(S) | test(Q) | top-1 | top-5
--- | --- | --- | --- | ---
A | 256 | 256 | 29.6 | 10.4
A-LRN | 256 | 256 | 29.7 | 10.5
B | 256 | 256 | 28.7 | 9.9
C | 256 | 256 | 28.1 | 9.4
C | 384 | 384 | 28.1 | 9.3
C | [256;512] | 384 | 27.3 | 8.8
D | 256 | 256 | 27.0 | 8.8
D | 384 | 384 | 26.8 | 8.7
D | [256;512] | 384 | 25.6 | 8.1
E | 256 | 256 | 27.3 | 9.0
D | 384 | 384 | 26.9 | 8.7
D | [256;512] | 384 | 25.5 | 8.0

First, we note that using local response normalisation (A-LRN network) does not improve on the model A without any normalisation layers. We thus do not employ normalisation in the deeper architectures (B–E).

首先，我们发现使用局部响应归一化(A-LRN网络)并没有对模型A有所改进。所以我们没有在更深的架构中(B-E)中使用归一化手段。

Second, we observe that the classification error decreases with the increased ConvNet depth: from 11 layers in A to 19 layers in E. Notably, in spite of the same depth, the configuration C (which contains three 1 × 1 conv. layers), performs worse than the configuration D, which uses 3 × 3 conv. layers throughout the network. This indicates that while the additional non-linearity does help (C is better than B), it is also important to capture spatial context by using conv. filters with non-trivial receptive fields (D is better than C). The error rate of our architecture saturates when the depth reaches 19 layers, but even deeper models might be beneficial for larger datasets. We also compared the net B with a shallow net with five 5 × 5 conv. layers, which was derived from B by replacing each pair of 3×3 conv. layers with a single 5×5 conv. layer (which has the same receptive field as explained in Sect. 2.3). The top-1 error of the shallow net was measured to be 7% higher than that of B (on a center crop), which confirms that a deep net with small filters outperforms a shallow net with larger filters.

第二，我们观察到，随着网络深度增加，分类错误率也在减少。注意，模型C和模型D深度相同，但C包含3个1×1卷积层，D对应的是3个3×3卷积层，C的性能没有D好。这说明额外的非线性确实有帮助（C模型比B模型要好），但用有意义的卷积滤波器来捕捉空间上下文同样重要（D比C要好）。当深度到达19时，我们架构的错误率会饱和，但更深的模型可能应用在更大的数据集上效果会更好。我们还将模型B与一个浅层的5个5×5滤波器卷积层进行了比较，模型从B中衍生出来，将其中的3×3滤波器对换成了5×5的，这样其感受野是一样大的。浅层模型的top-1错误率比模型B要高7%，这说明了小型滤波器的深层网络比较大滤波器的浅层网络效果要差。

Finally, scale jittering at training time (S ∈ [256;512]) leads to significantly better results than training on images with fixed smallest side (S = 256 or S = 384), even though a single scale is used at test time. This confirms that training set augmentation by scale jittering is indeed helpful for capturing multi-scale image statistics.

最后，训练时的尺度变化(S ∈ [256;512])会明显比使用固定尺寸(S = 256 or S = 384)要好，但测试时我们还是用了单个尺度的。这说明用尺度变化扩充训练集是确实有用的。

### 4.2 MULTI-SCALE EVALUATION

Having evaluated the ConvNet models at a single scale, we now assess the effect of scale jittering at test time. It consists of running a model over several rescaled versions of a test image(corresponding to different values of Q), followed by averaging the resulting class posteriors. Considering that a large discrepancy between training and testing scales leads to a drop in performance, the models trained with fixed S were evaluated over three test image sizes, close to the training one: Q = {S − 32,S,S + 32}. At the same time, scale jittering at training time allows the network to be applied to a wider range of scales at test time, so the model trained with variable S ∈ [$S_{min};S_{max}$] was evaluated over a larger range of sizes Q = {$S_{min},0.5(S_{min}+ S_{max}),S_{max}$}.

在单尺度上评估了模型后，我们在测试时评估尺度变化的效果。这要将测试图像改变几个尺寸版本（对应不同的Q值），然后对这些图像运行模型，最后对分类结果进行平均，作为最后结果。如果训练和测试的尺度相差较大，则会带来性能的降低，用固定S训练出来的模型在三种测试图像尺寸上（与训练尺寸较接近）进行了评估，即Q = {S − 32,S,S + 32}。同时，训练时尺度变化使网络在测试时运行在更广范围的尺度上，所以用S ∈ [$S_{min};S_{max}$]训练出来的模型在更大的尺寸范围内评估Q = {$S_{min},0.5(S_{min}+ S_{max}),S_{max}$}。

The results, presented in Table 4, indicate that scale jittering at test time leads to better performance (as compared to evaluating the same model at a single scale, shown in Table 3). As before, the deepest configurations (D and E) perform the best, and scale jittering is better than training with a fixed smallest side S. Our best single-network performance on the validation set is 24.8%/7.5% top-1/top-5 error (highlighted in bold in Table 4). On the test set, the configuration E achieves 7.3% top-5 error.

表4给出了结果，显示测试时的尺度变化会得到更好的结果（与表3中同样的模型在一个尺度上评估的结果相比）。和前面一样，最深的配置（D和E）得到了最佳的结果，尺度变化比固定训练尺度效果要好。我们最好的单网络性能在训练集上是24.8%/7.5%的top-1/top-5错误率。在测试集上，配置E的top-5错误率达到了7.3%。

Table 4: ConvNet performance at multiple test scales.

ConvNet | train (S) | test (Q) | top-1 | top-5
--- | --- | --- | --- | ---
B | 256 | 224,256,288 | 28.2 | 9.6
C | 256 | 224,256,288 | 27.7 | 9.2
C | 384 | 352,384,416 | 27.8 | 9.2
C | [256;512] | 256,384,512 | 26.3 | 8.2
D | 256 | 224,256,288 | 26.6 | 8.6
D | 384 | 352,384,416 | 26.5 | 8.6
D | [256;512] | 256,384,512 | 24.8 | 7.5
E | 256 | 224,256,288 | 26.9 | 8.7
E | 384 | 352,384,416 | 26.7 | 8.6
E | [256;512] | 256,384,512 | 24.8 | 7.5

### 4.3 MULTI-CROP EVALUATION

In Table 5 we compare dense ConvNet evaluation with mult-crop evaluation (see Sect. 3.2 for details). We also assess the complementarity of the two evaluation techniques by averaging their softmax outputs. As can be seen, using multiple crops performs slightly better than dense evaluation, and the two approaches are indeed complementary, as their combination outperforms each of them. As noted above, we hypothesize that this is due to a different treatment of convolution boundary conditions.

表5中我们比较了密集卷积网络评估与多剪切块评估（详见3.2节）。我们还对它们的softmax输出进行平均，从而评估了这两种评估方法的互补性。将会看到，使用多剪切块比密集评估略好一些，这两种方法确实是互补的，因为其组合比任何一个单独都要好。如前所述，我们认为这是卷积边界条件的不同导致的。

Table 5: ConvNet evaluation techniques comparison. In all experiments the training scale S was sampled from [256;512], and three test scales Q were considered: {256,384,512}.

ConvNet | Evaluation method | top-1 | top-5
:---: | --- | --- | ---
D | dense | 24.8 | 7.5
D | multi-crop | 24.6 | 7.5
D | multi-crop & dense | 24.4 | 7.2
E | dense | 24.8 | 7.5
E | multi-crop | 24.6 | 7.4
E | multi-crop & dense | 24.4 | 7.1

### 4.4 CONVNET FUSION

Up until now, we evaluated the performance of individual ConvNet models. In this part of the experiments, we combine the outputs of several models by averaging their soft-max class posteriors. This improves the performance due to complementarity of the models, and was used in the top ILSVRC submissions in 2012 (Krizhevsky et al., 2012) and 2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014).

前面我们评价了单个卷积网络的性能。这一部分的试验中，我们将几个模型的softmax输出类别信息进行平均，由于模型之间有互补性，这可以改进性能，这也是ILSVRC 2012和2013冠军得主的做法。

The results are shown in Table 6. By the time of ILSVRC submission we had only trained the single-scale networks, as well as a multi-scale model D (by fine-tuning only the fully-connected layers rather than all layers). The resulting ensemble of 7 networks has 7.3% ILSVRC test error. After the submission, we considered an ensemble of only two best-performing multi-scale models (configurations D and E), which reduced the test error to 7.0% using dense evaluation and 6.8% using combined dense and multi-crop evaluation. For reference, our best-performing single model achieves 7.1% error (model E, Table 5).

结果如表6所示。在提交ILSVRC的时候，我们只训练了单尺度的网络和多尺度的D模型（只精调全连接层）。得到的7网络集成模型在ILSVRC上只有7.3%的测试错误率。提交之后，我们考虑只集成2个最好的多尺度模型（D和E），将测试错误率降到了7.0%（使用稠密评估）和6.8%（使用稠密和多剪切块评估结合）。作为参考，我们的单模型最好成绩为7.1%（表5中的模型E）。

Table 6: Multiple ConvNet fusion results.

Combined ConvNet models | top-1 val | top-5 val | top-5 test
--- | --- | --- | --- 
(D/256/224,256,288), (D/384/352,384,416), (D/[256;512]/256,384,512) | 24.7 | 7.5 | 7.3
(C/256/224,256,288), (C/384/352,384,416) |
(E/256/224,256,288), (E/384/352,384,416) |
(D/[256;512]/256,384,512), (E/[256;512]/256,384,512), dense eval. | 24.0 | 7.1 | 7.0
(D/[256;512]/256,384,512), (E/[256;512]/256,384,512) , multi-crop | 23.9 | 7.2 | -
(D/[256;512]/256,384,512), (E/[256;512]/256,384,512), multi-crop & dense eval. | 23.7 | 6.8 | 6.8

### 4.5 COMPARISON WITH THE STATE OF THE ART

Finally, we compare our results with the state of the art in Table 7. In the classification task of ILSVRC-2014 challenge (Russakovsky et al., 2014), our “VGG” team secured the 2nd place with 7.3% test error using an ensemble of 7 models. After the submission, we decreased the error rate to 6.8% using an ensemble of 2 models.

最后，我们与其他最好的方法进行了比较，如表7所示。在ILSVRC-2014分类任务中，我们的VGG团队取得了第二名的成绩，测试错误率7.3%，集成了7个模型。模型提交以后，我们用2个模型的集成将错误率降到了6.8%。

As can be seen from Table 7, our very deep ConvNets significantly outperform the previous generation of models, which achieved the best results in the ILSVRC-2012 and ILSVRC-2013 competitions. Our result is also competitive with respect to the classification task winner (GoogLeNet with 6.7% error) and substantially outperforms the ILSVRC-2013 winning submission Clarifai, which achieved 11.2% with outside training data and 11.7% without it. This is remarkable, considering that our best result is achieved by combining just two models – significantly less than used in most ILSVRC submissions. In terms of the single-net performance, our architecture achieves the best result (7.0% test error), outperforming a single GoogLeNet by 0.9%. Notably, we did not depart from the classical ConvNet architecture of LeCun et al. (1989), but improved it by substantially increasing the depth.

如表7所示，我们的深度卷积网络比以前的模型(ILSVRC-2012 & ILSVRC-2013)明显要好很多。我们的结果与分类任务第一名(GoogLeNet错误率6.7%)也是差距不大的。我们只集成了2个模型，这是最了不起的。在单个网络的性能方面，我们的架构取得了最佳成果7.0%错误率，比GoogLeNet高0。9%。而且我们的模型与最早的LeCun et al. (1989)相似度很高，只是增加了不少深度。

Table 7: Comparison with the state of the art in ILSVRC classification. Our method is denoted as “VGG”. Only the results obtained without outside training data are reported.

Method | top-1 val | top-5 val | top-5 test
--- | --- | --- | ---
VGG (2 nets, multi-crop & dense eval.) | 23.7 | 6.8 | 6.8
VGG (1 net, multi-crop & dense eval.) | 24.4 | 7.1 | 7.0
VGG (ILSVRC submission, 7 nets, dense eval.) | 24.7 | 7.5 | 7.3
GoogLeNet (Szegedy et al., 2014) (1 net) | - | 7.9
GoogLeNet (Szegedy et al., 2014) (7 nets) | - | 6.7
MSRA (He et al., 2014) (11 nets) | - | - | 8.1
MSRA (He et al., 2014) (1 net) | 27.9 | 9.1 | 9.1
Clarifai (Russakovsky et al., 2014) (multiple nets) | - | - | 11.7
Clarifai (Russakovsky et al., 2014) (1 net) | - | - | 12.5
Zeiler & Fergus (Zeiler & Fergus, 2013) (6 nets) | 36.0 | 14.7 | 14.8
Zeiler & Fergus (Zeiler & Fergus, 2013) (1 net) | 37.5 | 16.0 | 16.1
OverFeat (Sermanet et al., 2014) (7 nets) | 34.0 | 13.2 | 13.6
OverFeat (Sermanet et al., 2014) (1 net) | 35.7 | 14.2 | -
Krizhevsky et al. (Krizhevsky et al., 2012) (5 nets) | 38.1 | 16.4 | 16.4
Krizhevsky et al. (Krizhevsky et al., 2012) (1 net) | 40.7 | 18.2 | -

## 5 CONCLUSION

In this work we evaluated very deep convolutional networks (up to 19 weight layers) for large-scale image classification. It was demonstrated that the representation depth is beneficial for the classification accuracy, and that state-of-the-art performance on the ImageNet challenge dataset can be achieved using a conventional ConvNet architecture(LeCun et al., 1989; Krizhevsky et al., 2012) with substantially increased depth. In the appendix, we also show that our models generalise well to a wide range of tasks and datasets, matching or outperforming more complex recognition pipelines built around less deep image representations. Our results yet again confirm the importance of depth in visual representations.

本文中我们评估了大规模图像分类中的非常深度卷积网络（最多19权值层）。得到结论是，增加深度是可以提高分类正确率的，只需要将传统卷积网络的架构增加深度，就可以提高ImageNet挑战的最好成绩。附录中表示，我们的模型在其他任务和数据集上的泛化能力也很好，达到或超过了没那么深的但更复杂的识别模型的表现。我们的结论再次证明了，深度对于视觉表示的重要性。

## ACKNOWLEDGEMENTS

This work was supported by ERC grant VisRec no. 228180. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the GPUs used for this research.

## REFERENCES

- Bell, S., Upchurch, P., Snavely, N., and Bala, K. Material recognition in the wild with the materials in context database. CoRR, abs/1412.0623, 2014.
- Chatfield, K., Simonyan, K., Vedaldi, A., and Zisserman, A. Return of the devil in the details: Delving deep into convolutional nets. In Proc. BMVC., 2014.
- Cimpoi, M., Maji, S., and Vedaldi, A. Deep convolutional filter banks for texture recognition and segmentation. CoRR, abs/1411.6836, 2014.
- Ciresan, D. C., Meier, U., Masci, J., Gambardella, L. M., and Schmidhuber, J. Flexible, high performance convolutional neural networks for image classification. In IJCAI, pp. 1237–1242, 2011.
- Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M., Mao, M., Ranzato, M., Senior, A., Tucker, P., Yang, K., Le, Q. V., and Ng, A. Y. Large scale distributed deep networks. In NIPS, pp. 1232–1240, 2012.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In Proc. CVPR, 2009.
- Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E.,and Darrell, T. Decaf: A deep convolutional activation feature for generic visual recognition. CoRR, abs/1310.1531, 2013.
- Everingham, M., Eslami, S. M. A., Van Gool, L., Williams, C., Winn, J., and Zisserman, A. The Pascal visual object classes challenge: A retrospective. IJCV, 111(1):98–136, 2015.
- Fei-Fei, L., Fergus, R., and Perona, P. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In IEEE CVPR Workshop of Generative Model Based Vision, 2004.
- Girshick, R. B., Donahue, J., Darrell, T., and Malik, J. Rich feature hierarchies for accurate object detection and semantic segmentation. CoRR, abs/1311.2524v5, 2014. Published in Proc. CVPR, 2014.
- Gkioxari, G., Girshick, R., and Malik, J. Actions and attributes from wholes and parts. CoRR, abs/1412.2604, 2014.
- Glorot, X. and Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. In Proc. AISTATS, volume 9, pp. 249–256, 2010.
- Goodfellow, I. J., Bulatov, Y., Ibarz, J., Arnoud, S., and Shet, V. Multi-digit number recognition from street view imagery using deep convolutional neural networks. In Proc. ICLR, 2014.
- Griffin, G., Holub, A., and Perona, P. Caltech-256 object category dataset. Technical Report 7694, California Institute of Technology, 2007.
- He, K., Zhang, X., Ren, S., and Sun, J. Spatial pyramid pooling in deep convolutional networks for visual
recognition. CoRR, abs/1406.4729v2, 2014.
- Hoai, M. Regularized max pooling for image categorization. In Proc. BMVC., 2014.
- Howard, A. G. Some improvements on deep convolutional neural network based image classification. In Proc. ICLR, 2014.
- Jia, Y. Caffe: An open source convolutional architecture for fast feature embedding. http://caffe.berkeleyvision.org/, 2013.
- Karpathy, A. and Fei-Fei, L. Deep visual-semantic alignments for generating image descriptions. CoRR, abs/1412.2306, 2014.
- Kiros, R., Salakhutdinov, R., and Zemel, R. S. Unifying visual-semantic embeddings with multimodal neural language models. CoRR, abs/1411.2539, 2014.
- Krizhevsky, A. One weird trick for parallelizing convolutional neural networks. CoRR, abs/1404.5997, 2014.
- Krizhevsky, A., Sutskever, I., and Hinton, G. E. ImageNet classification with deep convolutional neural networks. In NIPS, pp. 1106–1114, 2012.
- LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., and Jackel, L. D. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541–551, 1989.
- Lin, M., Chen, Q., and Yan, S. Network in network. In Proc. ICLR, 2014.
- Long, J., Shelhamer, E., and Darrell, T. Fully convolutional networks for semantic segmentation. CoRR, abs/1411.4038, 2014.
- Oquab, M., Bottou, L., Laptev, I., and Sivic, J. Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks. In Proc. CVPR, 2014.
- Perronnin, F., Sánchez, J., and Mensink, T. Improving the Fisher kernel for large-scale image classification. In Proc. ECCV, 2010.
- Razavian, A., Azizpour, H., Sullivan, J., and Carlsson, S. CNN Features off-the-shelf: an Astounding Baseline for Recognition. CoRR, abs/1403.6382, 2014.
- Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. ImageNet large scale visual recognition challenge. CoRR, abs/1409.0575, 2014.
- Sermanet, P., Eigen, D., Zhang, X., Mathieu, M., Fergus, R., and LeCun, Y. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks. In Proc. ICLR, 2014.
- Simonyan, K. and Zisserman, A. Two-stream convolutional networks for action recognition in videos. CoRR, abs/1406.2199, 2014. Published in Proc. NIPS, 2014.
- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. CoRR, abs/1409.4842, 2014.
- Wei, Y., Xia, W., Huang, J., Ni, B., Dong, J., Zhao, Y., and Yan, S. CNN: Single-label to multi-label. CoRR, abs/1406.5726, 2014.
- Zeiler, M. D. and Fergus, R. Visualizing and understanding convolutional networks. CoRR, abs/1311.2901, 2013. Published in Proc. ECCV, 2014.

