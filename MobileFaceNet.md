# MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices

Sheng Chen et al. Beijing Jiaotong University

## Abstract 摘要

We present a class of extremely efficient CNN models, MobileFaceNets, which use less than 1 million parameters and are specifically tailored for high-accuracy real-time face verification on mobile and embedded devices. We first make a simple analysis on the weakness of common mobile networks for face verification. The weakness has been well overcome by our specifically designed MobileFaceNets. Under the same experimental conditions, our MobileFaceNets achieve significantly superior accuracy as well as more than 2 times actual speedup over MobileNetV2. After trained by ArcFace loss on the refined MS-Celeb-1M, our single MobileFaceNet of 4.0MB size achieves 99.55% accuracy on LFW and 92.59% TAR@FAR1e-6 on MegaFace, which is even comparable to state-of-the-art big CNN models of hundreds MB size. The fastest one of MobileFaceNets has an actual inference time of 18 milliseconds on a mobile phone. For face verification, MobileFaceNets achieve significantly improved efficiency over previous state-of-the-art mobile CNNs. 

我们提出一类极其高效的CNN模型，即MobileFaceNets，使用的参数少于一百万，特意为移动和嵌入式设备上的高准确度实时人脸验证定制的。我们首先简单分析了一下常见移动网络对人脸验证上的弱点。我们特别设计的MobileFaceNets很好的克服了这些弱点。在同样的试验条件下，我们的MobileFaceNets比MobileNetV2准确率明显提高，同时速度快了2倍。在用ArcFace损失函数，在MS-Celeb-1M上训练后，我们的单个MobileFaceNet模型大小4M，在LFW上取得了99.55%的准确率，在MegaFace上取得了92.59% TAR@FAR1e-6的成绩，甚至于目前最好的数百MB大小的CNN模型可以相比。最快的MobileFaceNet在手机上的实际推理时间达到了18ms。对于人脸验证，MobileFaceNets比之前最好的移动CNN模型效率要明显提高。

**Keywords**: Mobile network, face verification, face recognition, convolutional neural network, deep learning.

## 1 Introduction 引言

Face verification is an important identity authentication technology used in more and more mobile and embedded applications such as device unlock, application login, mobile payment and so on. Some mobile applications equipped with face verification technology, for example, smartphone unlock, need to run offline. To achieve user-friendliness with limited computation resources, the face verification models deployed locally on mobile devices are expected to be not only accurate but also small and fast. However, modern high-accuracy face verification models are built upon deep and big convolutional neural networks (CNNs) which are supervised by novel loss functions during training stage. The big CNN models requiring high computational resources are not suitable for many mobile and embedded applications. Several highly efficient neural network architectures, for example, MobileNetV1 [1], ShuffleNet [2], and MobileNetV2 [3], have been proposed for common visual recognition tasks rather than face verification in recent years. It is a straight-forward way to use these common CNNs unchanged for face verification, which only achieves very inferior accuracy compared with state-of-the-art results according to our experiments (see Table 2).

人脸验证是一种非常重要的身份认证技术，正用于越来越多的移动和嵌入式应用中，如设备解锁，应用登录，移动支付等等。一些装备了人脸验证技术的移动设备需要离线运行，如智能手机解锁。在有限的计算资源中取得用户友好的体验，移动设备本地部署的人脸验证模型不仅需要准确，还需要小型快速。但是，现代高准确率人脸验证模型，都是基于大型深度卷积神经网络(CNNs)的，在训练阶段使用的是新型损失函数。大型CNN模型需要大量计算资源，不适用于很多移动和嵌入式应用。一些效率很高的神经网络架构，如MobileNetV1[1], ShuffleNet[2]和MobileNetV2[3]，都是用于通用视觉识别任务的，而不是人脸验证。直接使用这些CNN进行人脸验证非常方便，但是我们进行了试验，得到的效果不是很好，如表2所示。

In this paper, we make a simple analysis on common mobile networks’ weakness for face verification. The weakness has been well overcome by our specifically designed MobileFaceNets, which is a class of extremely efficient CNN models tailored for high-accuracy real-time face verification on mobile and embedded devices. Our MobileFaceNets use less than 1 million parameters. Under the same experimental conditions, our MobileFaceNets achieve significantly superior accuracy as well as more than 2 times actual speedup over MobileNetV2. After trained on the refined MS-Celeb-1M [4] by ArcFace [5] loss from scratch, our single MobileFaceNet model of 4.0MB size achieves 99.55% face verification accuracy (see Table 3) on LFW [6] and 92.59% TAR@FAR10-6 (see Table 4) on MegaFace Challenge 1 [7], which is even comparable to state-of-the-art big CNN models of hundreds MB size. Note that many existing techniques such as pruning [37], low-bit quantization [29], and knowledge distillation [16] are able to improve MobileFaceNets’ efficiency additionally, but these are not included in the scope of this paper.

本文中，我们简单分析了常见移动网络在人脸验证上的缺点。我们专门设计的MobileFaceNets则很好的克服了这些缺点，这是一类效率很高的CNN模型，为移动和嵌入式设备上的高准确度实时人脸验证定制的。我们的MobileFaceNets使用的参数数量少于一百万。在同样的试验条件下，我们的MobileFaceNets与MobileNetV2相比，取得了明显很优秀的准确率，速度也快了2倍多。在改进的MS-Celeb-1M[4]上用ArcFace[5]损失函数从头训练后，我们的单MobileFaceNet模型大小4.0M，在LFW[6]上取得了99.55%的验证准确率（见表3），在MegaFace挑战1[7]上取得了92.59% TAR@FAR10-6的成绩（见表4），与目前最好的大型CNN模型（几百M）的结果都差不多。注意很多现有的技术，如剪枝[37]，低精度量化[29]和知识蒸馏[16]都可以用于进一步改进MobileFaceNets的效率，但这并不是本文中的重点。

The major contributions of this paper are summarized as follows: (1) After the last (non-global) convolutional layer of a face feature embedding CNN, we use a global depthwise convolution layer rather than a global average pooling layer or a fully connected layer to output a discriminative feature vector. The advantage of this choice is also analyzed in both theory and experiment. (2) We carefully design a class of face feature embedding CNNs, namely MobileFaceNets, with extreme efficiency on mobile and embedded devices. (3) Our experiments on LFW, AgeDB ([8]), and MegaFace show that our MobileFaceNets achieve significantly improved efficiency over previous state-of-the-art mobile CNNs for face verification.

本文的主要贡献如下：(1)在人脸特征嵌入CNN的最后一个（非全局）卷积层后，我们使用了一个全局depthwise卷积层，而不是一个全局平均池化层，或一个全连接层，来输出一个有区分能力的特征向量。这种选择的优势，我们进行了理论分析和试验验证。(2)我们仔细设计了一类人脸特征嵌入CNNs，即MobileFaceNets，在移动和嵌入式设备中效率非常高。(3)我们在LFW、AgeDB[8]和MegaFace上的试验表明，MobileFaceNets比之前最好的移动模型在人脸验证上取得了明显改进的效率。

## 2 Related Work 相关工作

Tuning deep neural architectures to strike an optimal balance between accuracy and performance has been an area of active research for the last several years [3]. For common visual recognition tasks, many efficient architectures have been proposed recently [1, 2, 3, 9]. Some efficient architectures can be trained from scratch. For example, SqueezeNet ([9]) uses a bottleneck approach to design a very small network and achieves AlexNet-level [10] accuracy on ImageNet [11, 12] with 50x fewer parameters (i.e., 1.25 million). MobileNetV1 [1] uses depthwise separable convolutions to build lightweight deep neural networks, one of which, i.e., MobileNet-160 (0.5x), achieves 4% better accuracy on ImageNet than SqueezeNet at about the same size. ShuffleNet [2] utilizes pointwise group convolution and channel shuffle operation to reduce computation cost and achieve higher efficiency than MobileNetV1. MobileNetV2 [3] architecture is based on an inverted residual structure with linear bottleneck and improves the state-of-the-art performance of mobile models on multiple tasks and benchmarks. The mobile NASNet [13] model, which is an architectural search result with reinforcement learning, has much more complex structure and much more actual inference time on mobile devices than MobileNetV1, ShuffleNet, and MobileNetV2. However, these lightweight basic architectures are not so accurate for face verification when trained from scratch (see Table 2).

深度神经架构调参，在准确率和性能之间达到最优平衡，在过去几年中一直是活跃的研究领域[3]。对于常见的视觉识别任务，已经提出了很多高效的架构[1,2,3,9]。一些高效的架构是从头训练的，如SqueezeNet[9]使用了一种瓶颈方法来设计非常小型的网络，在ImageNet[11,12]上取得了AlexNet级别[10]的准确率，用到的参数少了50倍（即，1.25 million）。MobileNetV1[1]使用depthwise separable卷积构建了轻量级深度卷积网络，其中一个网络，即MobileNet-160(0.5x)，在与SquuezeNet大小基本相同时，在ImageNet上取得的准确率还要高4%。ShuffleNet[2]使用pointwise分组卷积和通道混洗操作，降低了运算代价，取得了比MobileNetV1更高的效率。MobileNetV2[3]架构是基于逆残差和线性瓶颈的，改进了移动模型在多项任务和基准测试上的最好效果。移动NASNet[13]模型，这是用强化学习进行架构搜索得到的结果，结构更为复杂，在移动设备上比MobileNetV1, ShuffleNet和MobileNetV2实际推理时间更长。但是，这些轻量级的基本架构对于人脸验证从头训练的话，效果不是很好（见表2）。

Accurate lightweight architectures specifically designed for face verification have been rarely researched. [14] presents a light CNN framework to learn a compact embedding on the large-scale face data, in which the Light CNN-29 model achieves 99.33% face verification accuracy on LFW with 12.6 million parameters. Compared with MobileNetV1, Light CNN-29 is not lightweight for mobile and embedded platform. Light CNN-4 and Light CNN-9 are much less accurate than Light CNN-29. [15] proposes ShiftFaceNet based on ShiftNet-C model with 0.78 million parameters, which only achieves 96.0% face verification accuracy on LFW. In [5], an improved version of MobileNetV1, namely LMobileNetE, achieves comparable face verification accuracy to state-of-the-art big models. But LMobileNetE is actually a big model of 112MB model size, rather than a lightweight model. All above models are trained from scratch.

对人脸验证特别设计准确的轻量级架构少有研究。[14]提出了一种轻量的CNN架构，在大规模人脸数据上学习了一种紧凑嵌入，其中的Light CNN-29模型在LFW上取得了99.33的人脸验证准确率，参数数量为12.6 million。与MobileNetV1相比，Light CNN-29对于移动和嵌入式平台并不是很小。Light CNN-4和Light CNN-9比Light CNN-29准确度要差很多。[15]基于Shift-C模型提出ShiftFaceNet，只使用了0.78 million参数，在LFW上只取得了96.0%的人脸验证准确率。在[5]中提出了一种改进版的MobileNetV1，即LMobileNetE，与目前最好的大型模型取得了类似的人脸验证准确率。但LMobileNEtE实际上是一个大型模型，模型大小112MB，所以不是一个轻量模型。上述所有模型都是从头训练的。

Another approach for obtaining lightweight face verification models is compressing pretrained networks by knowledge distillation [16]. In [17], a compact student network (denoted as MobileID) trained by distilling knowledge from the teacher network DeepID2+ [33] achieves 97.32% accuracy on LFW with 4.0MB model size. In [1], several small MobileNetV1 models for face verification are trained by distilling knowledge from the pretrained FaceNet [18] model and only face verification accuracy on the authors’ private test dataset are reported. Regardless of the small student models’ accuracy on public test datasets, our MobileFaceNets achieve comparable accuracy to the strong teacher model FaceNet on LFW (see Table 3) and MegaFace (see Table 4).

另一种得到轻量级人脸验证模型的方法是，通过知识蒸馏[16]压缩训训练网络。在[17]中，从老师网络DeepID2+[33]进行知识蒸馏训练得到了一个紧凑的学生网络（表示为MobileID），在LFW上得到了97.32%的准确率，大小4.0M。在[1]中，从预训练的FaceNet[18]中进行知识蒸馏训练，使用几个小型的MobileNetV1模型进行人脸验证，只在作者的私有测试数据集上给出了人脸验证准确率结果。除去小的学生模型在公开测试数据集上的准确率，我们的MobileFaceNets在LFW和MegaFace上取得了与大型老师FaceNet模型类似的准确率（见表3和表4）。

## 3 Approach 方法

In this section, we will describe our approach towards extremely efficient CNN models for accurate real-time face verification on mobile devices, which overcome the weakness of common mobile networks for face verification. To make our results totally reproducible, we use ArcFace loss to train all face verification models on public datasets, following the experimental settings in [5].

本节中，我们将描述为移动设备上准确实时的人脸验证任务设计极其高效的CNN模型的方法，克服了常见移动网络进行人脸验证的缺点。为使我们的结果完全可复制，我们使用ArcFace损失在公开数据集上来训练所有的人脸验证模型，使用[5]中的试验设置。

### 3.1 The Weakness of Common Mobile Networks for Face Verification 常见移动网络在人脸验证上的缺点

There is a global average pooling layer in most recent state-of-the-art mobile networks proposed for common visual recognition tasks, for example, MobileNetV1, ShuffleNet, and MobileNetV2. For face verification and recognition, some researchers ([14], [5], etc.) have observed that CNNs with global average pooling layers are less accurate than those without global average pooling. However, no theoretical analysis for this phenomenon has been given. Here we make a simple analysis on this phenomenon in the theory of receptive field [19].

最近提出的常见视觉识别任务的最好移动网络都有一个全局平均池化层，如MobileNetV1，ShuffleNet和MobileNetV2。对于人脸验证和识别，一些研究者[14,5]观察到，使用全局平均池化层的CNN比没有使用的准确度要低。但是，没有给出这种现象的理论分析。这里我们对这种现象采用感受野理论进行简单的分析[19]。

A typical deep face verification pipeline includes preprocessing face images, extracting face features by a trained deep model, and matching two faces by their features’ similarity or distance. Following the preprocessing method in [5, 20, 21, 22], we use MTCNN [23] to detect faces and five facial landmarks in images. Then we align the faces by similarity transformation according to the five landmarks. The aligned face images are of size 112 × 112, and each pixel in RGB images is normalized by subtracting 127.5 then divided by 128. Finally, a face feature embedding CNN maps each aligned face to a feature vector, as shown in Fig. 1. 

典型的深度人脸验证流程包括人脸图像的预处理，使用训练好的深度模型提取人脸特征，通过两个人脸的特征相似度或距离进行匹配。与[5,20,21,22]的预处理方法相同，我们也使用MTCNN[23]来检测图像中的人脸和5个面部特征点。然后我们进行人脸对齐，根据5个特征点通过相似度变换进行。对齐的人脸图像大小为112×112，RGB图像，进行归一化，即减去127.5然后除以128。最后，人脸特征嵌入CNN将每个对齐的人脸映射为一个特征向量，如图1所示。

Without loss of generality, we use MobileNetV2 as the face feature embedding CNN in the following discussion. To preserve the same output feature map sizes as the original network with 224 × 224 input, we use the setting of stride = 1 in the first convolutional layer instead of stride = 2, where the latter setting leads to very poor accuracy. So, before the global average pooling layer, the output feature map of the last convolutional layer, denoted as FMap-end for convenience, is of spatial resolution 7 × 7. Although the theoretical receptive fields of the corner units and the central units of FMap-end are of the same size, they are at different positions of the input image. The receptive fields’ center of FMap-end’s corner units is in the corner of the input image and the receptive fields’ center of FM-end’s central units are in the center of the input image, as shown in Fig. 1. According to [24], pixels at the center of a receptive field have a much larger impact on an output and the distribution of impact within a receptive field on the output is nearly Gaussian. The effective receptive field [24] sizes of FMap-end’s corner units are much smaller than the ones of FMap-end’s central units. When the input image is an aligned face, a corner unit of FMap-end carries less information of the face than a central unit. Therefore, different units of FMap-end are of different importance for extracting a face feature vector.

不失一般性，我们在下述讨论中使用MobileNetV2作为人脸特征嵌入的CNN。网络输入大小为224×224，为保持同样输出特征图大小，我们设第一个卷积层的步长为1，而不是2，步长为2会导致很差的准确率。所以，在全局平均池化层之前，最后一个卷积层的输出特征图，表示为FMap-end，其空间分辨率为7×7。虽然FMap-end上的中心单元和角点单元上的理论感受野大小应当相同，但它们在输入图像中位于不同的图像位置。FMap-end角点的感受野的中心在输入图像的角点上，FM-end上中心点的感受野的中心在输入图像的中心，如图1所示。根据[24]，在感受野中心的像素对输出的影响要大的多，感受野内部对输出影响的分布是接近高斯的。FMap-end的角点的有效感受野[24]的大小，比FMap-end中心点的有效感受野大小要小的多。当输入图像是一个对齐的人脸时，FMap-end的角点所携带的人脸信息比中心点要小。所以，FMap-end上的不同点对于提取人脸特征向量的重要性是不同的。

In MobileNetV2, the flattened FMap-end is unsuitable to be directly used as a face feature vector since it is of a too high dimension 62720. It is a natural choice to use the output of the global average pooling (denoted as GAPool) layer as a face feature vector, which achieves inferior verification accuracy in many researchers’ experiments [14, 5] as well as ours (see Table 2). The global average pooling layer treats all units of FMap-end with equal importance, which is unreasonable according to the above analysis. Another popular choice is to replace the global average pooling layer with a fully connected layer to project FMap-end to a compact face feature vector, which adds large number of parameters to the whole model. Even when the face feature vector is of a low dimension 128, the fully connected layer after FMap-end will bring additional 8 million parameters to MobileNetV2. We do not consider this choice since small model size is one of our pursuits.

在MobileNetV2中，展平的FMap-end直接作为人脸特征向量是不合适的，因为其维度太高，有62720。很自然的，可以使用全局平均池化层(GAPool)的输出作为人脸特征向量，但会得到较差的验证准确率，这在很多研究者的试验[14,5]中包括我们的都得到了验证（见表2）。全局平均池化层对待所有FMap-end上的单元都一样，这是不合理的，见上述分析。另一种流行的选择是将全局平均池化层替换为全连接层，将FMap-end投影到一个更紧凑的人脸特征向量，这会为整个模型增加太多参数。即使人脸特征向量维度低至128，FMap-end后的全连接层也会给MobileNetV2带来额外的8 millions参数。我们不考虑这种选项，因为我们的目标是小型模型。

### 3.2 Global Depthwise Convolution 全局depthwise卷积

To treat different units of FMap-end with different importance, we replace the global average pooling layer with a global depthwise convolution layer (denoted as GDConv). A GDConv layer is a depthwise convolution (c.f. [25, 1]) layer with kernel size equaling the input size, pad = 0, and stride = 1. The output for global depthwise convolution layer is computed as:

为区别FMap-end上不同单元的不同重要性，我们将全局平均池化层替换为全局depthwise卷积层（表示为GDConv）。一个GDConv层是一个depthwise卷积层（参考[25,1]），核心大小与输入大小一样，没有补零，步长为1。全局depthwise卷积层的输出计算如下：

$$G_m = \sum_{i,j} K_{i,j,m} ⋅ F_{i,j,m}$$(1)

where F is the input feature map of size W × H × M , K is the depthwise convolution kernel of size W × H × M, G is the output of size 1 × 1 × M, the m-th channel in G has only one element $G_m$, (i, j) denotes the spatial position in F and K, and m denotes the channel index. 其中F是输入特征图，大小为W×H×M；K是depthwise卷积核，大小为W×H×M；G是输出，大小为1×1×M；G中的第m通道只有一个元素$G_m$，(i,j)表示F和K中的空间位置，m表示通道索引。

Global depthwise convolution has a computational cost of: 全局depthwise卷积的计算量为：

$$W⋅H⋅M$$(2)

When used after FMap-end in MobileNetV2 for face feature embedding, the global depthwise convolution layer of kernel size 7 × 7 × 1280 outputs a 1280-dimensional face feature vector with a computational cost of 62720 MAdds (i.e., the number of operations measured by multiply-adds, c.f. [3]) and 62720 parameters. Let MobileNetV2-GDConv denote MobileNetV2 with global depthwise convolution layer. When both MobileNetV2 and MobileNetV2-GDConv are trained on CIASIA-Webface [26] for face verification by ArcFace loss, the latter achieves significantly better accuracy on LFW and AgeDB (see Table 2). Global depthwise convolution layer is an efficient structure for our design of MobileFaceNets.

在MobileNetV2的FMap-end后，使用全局depthwise卷积进行人脸特征嵌入，其核大小为7×7×1280，输出一个1280维的人脸特征向量，计算量为62720 MAdds，参数数量62720。令MobileNetV2-GDConv表示带有全局depthwise卷积层的MobileNetV2模型。我们把MobileNetV2和MobileNetV2-GDConv都在CIASIA-Webface[26]上进行训练，使用ArcFace损失，进行人脸验证，后者得到的准确率比前者明显的高，测试数据集为LFW和AgeDB（见表2）。全局depthwise卷积层是我们的MobileFaceNet设计的有效结构。

### 3.3 MobileFaceNet Architectures

Now we describe our MobileFaceNet architectures in detail. The residual [38] bottlenecks proposed in MobileNetV2 [3] are used as our main building blocks. For convenience, we use the same conceptions as those in [3]. The detailed structure of our primary MobileFaceNet architecture is shown in Table 1. Particularly, expansion factors for bottlenecks in our architecture are much smaller than those in MobileNetV2. We use PReLU [27] as the non-linearity, which is slightly better for face verification than using ReLU (see Table 2). In addition, we use a fast downsampling strategy at the beginning of our network, an early dimension-reduction strategy at the last several convolutional layers, and a linear 1 × 1 convolution layer following a linear global depthwise convolution layer as the feature output layer. Batch normalization [28] is utilized during training and batch normalization folding (c.f. Section 3.2 of [29]) is applied before deploying.

现在我们详细描述一下我们的MobileFaceNet架构。MobileNetV2 [3]中提出的残差[38]瓶颈层作为我们的主要部件使用。为方便起见，我们使用与[3]中同样的构思。我们的基本MobileFaceNet架构如表1所示。特别的，我们架构中的瓶颈的扩展系数比MobileNetV2中的要小很多。我们使用PReLU[27]作为人脸验证的非线性部分，这比使用ReLU略微好一些（见表2）。另外，我们在网络开始的时候使用快速下采样策略，在最后的几个卷积层中使用早期的降维策略，线性全局depthwise卷积层后是线性1×1卷积层，作为特征输出层。训练的时候使用了批归一化[28]，在部署前，则应用了批归一化折叠（参考[29]的3.2节）。

Our primary MobileFaceNet network has a computational cost of 221 million MAdds and uses 0.99 million parameters. We further tailor our primary architecture as follows. To reduce computational cost, we change input resolution from 112 × 112 to 112 × 96 or 96 × 96. To reduce the number of parameters, we remove the linear 1 × 1 convolution layer after the linear GDConv layer from MobileFaceNet, the resulting network of which is called MobileFaceNet-M. From MobileFaceNet-M, removing the 1 × 1 convolution layer before the linear GDConv layer produces the smallest network called MobileFaceNet-S. These MobileFaceNet networks’ effectiveness is demonstrated by the experiments in the next section.

我们的基本MobileFaceNet网络计算量为221 millions MAdds，使用0.99 million参数。我们进一步将基本架构进行如下定制。为降低计算量，我们将输入分辨率从112×112降低到112×96或96×96。为减少参数数量，我们从MobileFaceNet中去除了线性GDConv层后的线性1×1卷积层，得到的网络我们称为MobileFaceNet-M。从MobileFaceNet-M中，去掉线性GDConv层之前的1×1卷积层，得到的最小网络，称之为MobileFaceNet-S。这些MobileFaceNet网络的有效性在下一节的试验中进行展示。

Table 1. MobileFaceNet architecture for feature embedding. We use almost the same notations as MobileNetV2 [3]. Each line describes a sequence of operators, repeated n times. All layers in the same sequence have the same number c of output channels. The first layer of each sequence has a stride s and all others use stride 1. All spatial convolutions in the bottlenecks use 3 × 3 kernels. The expansion factor t is always applied to the input size. GDConv7x7 denotes GDConv of 7 × 7 kernels.

Input | Operator | t | c | n | s
--- | --- | --- | --- | --- | ---
112^2 × 3 | conv3×3 | - | 64 | 1 | 2
56^2 × 64 | depthwise conv3×3 | - | 64 | 1 | 1
56^2 × 64 | bottleneck | 2 | 64 | 5 | 2
28^2 × 64 | bottleneck | 4 | 128 | 1 | 2
14^2 × 128 | bottleneck | 2 | 128 | 6 | 1
14^2 × 128 | bottleneck | 4 | 128 | 1 | 2
7^2 × 128 | bottleneck | 2 | 128 | 2 | 1
7^2 × 128 | conv1×1 | - | 512 | 1 | 1
7^2 × 512 | linear GDConv7×7 | - | 512 | 1 | 1
1^2 × 512 | linear conv1×1 | - | 128 | 1 | 1

## 4 Experiments 试验

In this section, we will first describe the training settings of our MobileFaceNet models and our baseline models. Then we will compare the performance of our trained face verification models with some previous published face verification models, including several state-of-the-art big models. 在本节中，我们首先给出MobileFaceNet模型的训练设置和基准模型。然后我们比较训练好的人脸验证模型与之前发表的一些人脸验证模型，包括几个之前最好的大型模型。

### 4.1 Training settings and accuracy comparison on LFW and AgeDB 训练设置和在LFW和AgeDB上的准确度比较

We use MobileNetV1, ShuffleNet, and MobileNetV2 (with stride = 1 for the first convolutional layers of them since the setting of stride = 2 leads to very poor accuracy) as our baseline models. All MobileFaceNet models and baseline models are trained on CASIA-Webface dataset from scratch by ArcFace loss, for a fair performance comparison among them. We set the weight decay parameter to be 4e-5, except the weight decay parameter of the last layers after the global operator (GDConv or GAPool) being 4e-4. We use SGD with momentum 0.9 to optimize models and the batch size is 512. The learning rate begins with 0.1 and is divided by 10 at the 36K, 52K and 58K iterations. The training is finished at 60K iterations. Then, the face verification accuracy on LFW and AgeDB-30 is compared in Table 2.

我们使用MobileNetV1, ShuffleNet和MobileNetV2（第一个卷积层的步长为1，因为步长为2会得到很差的准确率结果）作为基准模型。所有的MobileFaceNet模型和基准模型都在CASIA-Webface数据集上从头训练，使用ArcFace损失，以进行公平的性能比较。我们设置权重衰减参数为4e-5，除了全局算子(GDConv or GAPool)之后的最后几层的权重衰减参数设为4e-4。我们使用SGD，动量0.9，batch size为512。学习速率开始设置为0.1，在第36K、52K和58K次迭代的时候除以10。训练总计进行60K次迭代。然后，在表2中比较在LFW和AgeDB-30上的人脸验证准确率。

Table 2. Performance comparison among mobile models trained on CASIA-Webface. In the last column, we report actual inference time in milliseconds (ms) on a Qualcomm Snapdragon 820 CPU of a mobile phone with 4 threads (using NCNN [30] inference framework).

Network | LFW | AgeDB-30 | Params | Speed
--- | --- | --- | --- | ---
MobileNetV1 | 98.63% | 88.95% | 3.2M | 60ms
ShuffleNet (1×,g=3) | 98.70% | 89.27% | 0.83M | 27ms
MobileNetV2 | 98.58% | 88.81% | 2.1M | 49ms
MobileNetV2-GDConv | 98.88% | 90.67% | 2.1M | 50ms
MobileFaceNet | 99.28% | 93.05% | 0.99M | 24ms
MobileFaceNet (112 × 96) | 99.18% | 92.96% | 0.99M | 21ms
MobileFaceNet (96 × 96) | 99.08% | 92.63% | 0.99M | 18ms
MobileFaceNet-M | 99.18% | 92.67% | 0.92M | 24ms
MobileFaceNet-S | 99.00% | 92.48% | 0.84M | 23ms
MobileFaceNet (ReLU) | 99.15% | 92.83% | 0.98M | 23ms
MobileFaceNet (expansion factor ×2) | 99.10% | 92.81% | 1.1M | 27ms

As shown in Table 2, compared with the baseline models of common mobile　networks, our MobileFaceNets achieve significantly better accuracy with faster　inference speed. Our primary MobileFaceNet achieves the best accuracy and　MobileFaceNet with a lower input resolution of 96 × 96 has the fastest inference　speed. Note that our MobileFaceNets are more efficient than those with larger　expansion factor such as MobileFaceNet (expansion factor ×2) and MobileNetV2-GDConv.

如表2所示，与普通移动网络的基准模型相比，我们的MobileFaceNet得到了明显更好的准确率，而且推理速度也更快。我们的基本MobileFaceNet模型得到了最好的准确率，低分辨率输入(96×96)推理速度最快。注意我们的MobileFaceNet比那些更大扩展因子的模型，如MobileFaceNet (expansion factor ×2) and MobileNetV2-GDConv。

To pursue ultimate performance, MobileFaceNet, MobileFaceNet (112 × 96), and MobileFaceNet (96 × 96) are also trained by ArcFace loss on the cleaned training set of MS-Celeb-1M database [5] with 3.8M images from 85K subjects. The accuracy of our primary MobileFaceNet is boosted to 99.55% and 96.07% on LFW and AgeDB-30, respectively. The three trained models’ accuracy on LFW is compared with previous published face verification models in Table 3.

为追求极致性能，我们在MS-Celeb-1M数据集[5]的清理过的训练集上（3.8M图像，85K个subjects）用ArcFace损失训练了MobileFaceNet, MobileFaceNet (112 × 96)和MobileFaceNet (96 × 96)。我们的基本MobileFaceNet在LFW和AgeDB-30上的准确率分别提升到了99.55%和96.07%。这三个训练好的模型在LFW上的准确率与之前发表的模型的比较，如表3所示。

Table 3. Performance comparison with previous published face verification models on LFW.

Method | Training Data | Net | Model Size | LFW Acc.
--- | --- | --- | --- | ---
Deep Face [31] | 4M | 3 | - | 97.35%
DeepFR [32] | 2.6M | 1 | 0.5G | 98.95%
DeepID2+ [33] | 0.3M | 25 | - | 99.47%
Center Face [34] | 0.7M | 1 | 105MB | 99.28%
DCFL [35] | 4.7M | 1 | - | 99.55%
SphereFace [20] | 0.49M | 1 | - | 99.47%
CosFace [22] | 5M | 1 | - | 99.73%
ArcFace (LResNet100E-IR) [5] | 3.8M | 1 | 250MB | 99.83%
FaceNet [18] | 200M | 1 | 30MB | 99.63%
ArcFace (LMobileNetE) [5] | 3.8M | 1 | 112MB | 99.50%
Light CNN-29 [14] | 4M | 1 | 50MB | 99.33%
MobileID [17] | - | 1 | 4.0MB | 97.32%
ShiftFaceNet [15] | - | 1 | 3.1MB | 96.00%
MobileFaceNet | 3.8M | 1 | 4.0MB | 99.55%
MobileFaceNet (112 × 96) | 3.8M | 1 | 4.0MB | 99.53%
MobileFaceNet (96 × 96) | 3.8M | 1 | 4.0MB | 99.52%

### 4.2 Evaluation on MegaFace Challenge1

In this paper, we use the Facescrub [36] dataset as the probe set to evaluate the verification performance of our primary MobileFaceNet on Megaface Challenge 1. Table 4 summarizes the results of our models trained on two protocols of MegaFace where the training dataset is regarded as small if it has less than 0.5 million images, large otherwise. Our primary MobileFaceNet shows comparable accuracy for the verification task on both the protocols.

本文中，我们使用Facescrub[36]数据集作为probe集，评估我们的基础MobileFaceNet在MegaFace挑战1上的人脸验证性能。表4总结我们模型在MegaFace两种协议下训练的结果，其中如果训练数据集的图像数量少于0.5 million，就为小型，否则就是大型。我们的基本MobileFaceNet模型在验证任务的两个协议下都展现出了不错的准确度。

Table 4. Face verification evaluation on Megafce Challenge 1. “VR” refers to face verification TAR (True Accepted Rate) under 10 -6 FAR (False Accepted Rate). MobileFaceNet (R) are evaluated on the refined version of MegaFace dataset (c.f. [5]).

Method | VR(large protocol) | VR(small protocol)
--- | --- | ---
SIAT MMLAB [34] | 87.27% | 76.72%
DeepSense V2 | 95.99% | 82.85%
SphereFace-Small [20] | - | 90.04%
Google-FaceNet v8 [18] | 86.47% | -
Vocord-deepVo V3 | 94.96% | -
CosFace (3-patch) [22] | 97.96% | 92.22%
iBUG_DeepInsight (ArcFace [5]) | 98.48% | -
MobileFaceNet | 90.16% | 85.76%
MobileFaceNet (R) | 92.59% | 88.09%

## 5 Conclusion

We proposed a class of face feature embedding CNNs, namely MobileFaceNets, with extreme efficiency for real-time face verification on mobile and embedded devices. Our experiments show that MobileFaceNets achieve significantly improved efficiency over previous state-of-the-art mobile CNNs for face verification.

我们提出了一类人脸特征嵌入CNNs，名为MobileFaceNets，在移动和嵌入式设备上可以进行实时人脸验证，计算效率非常高。我们的试验表明了MobileFaceNets比之前最好的移动CNNs在人脸验证上明显改进了计算效率。