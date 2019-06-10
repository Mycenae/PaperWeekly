# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

Mingxing Tan, Quoc V. Le Google Research, Brain Team

## Abstract

Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.

ConvNets一般是在固定的资源预算下开发的，如果有更多的可用资源，那么就进行网络缩放，得到更好的准确率。在本文中，我们系统的研究了模型尺度缩放，发现在网络的深度、宽度和分辨率之间取得平衡，会得到更好的性能。基于这种观察结果，我们提出一种新的尺度缩放方法，使用一种简单但高效的复合系数，对深度/宽度/分辨率进行统一尺度缩放。我们在对MobileNets和ResNet的尺度变换中，证明了这种方法的有效性。

To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1/97.1% top-5 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet.

更进一步，我们使用神经架构搜索来设计一种新的基准网络，并进行网络缩放，得到一族模型，称为EfficientNet，比之前的ConvNets得到了好很多的准确率和效率。特别是，我们的EfficientNet-B7在ImageNet上得到了目前最好的top-1准确率84.8%，top-5准确率97.1%，但比现有最好的ConvNet小了8.4倍，推理快了6.1倍。我们的EfficientNets在CIFAR-100、Flowers和其他三个迁移学习数据集上迁移性能很好，得到了目前最好的准确率91.7%、98.8%，参数数量少了一个量级。代码已开源。

## 1. Introduction 引言

Scaling up ConvNets is widely used to achieve better accuracy. For example, ResNet (He et al., 2016) can be scaled up from ResNet-18 to ResNet-200 by using more layers; Recently, GPipe (Huang et al., 2018) achieved 84.3% ImageNet top-1 accuracy by scaling up a baseline model four time larger. However, the process of scaling up ConvNets has never been well understood and there are currently many ways to do it. The most common way is to scale up ConvNets by their depth (He et al., 2016) or width (Zagoruyko & Komodakis, 2016). Another less common, but increasingly popular, method is to scale up models by image resolution (Huang et al., 2018). In previous work, it is common to scale only one of the three dimensions – depth, width, and image size. Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency.

把ConvNet缩放可以得到更好的准确率，这在实际中使用广泛。比如，ResNet通过使用更多的层，可以从ResNet-18放大到ResNet-200；最近，GPipe通过将一个基准模型放大了4倍，在ImageNet上得到了84.3%的top-1准确率。但是，ConvNets缩放过程一直没有得到透彻的理解，现在有很多缩放的方法。最常用的方法是增加ConvNets的深度，或宽度。另一种不是太常用，但非常流行的方法是，增加图像分辨率以缩放模型。在之前的工作中，常用的是缩放三个维度中的一个，深度、宽度和图像大小。虽然可以任意缩放两个或三个维度，但任意缩放需要繁琐的人工调节，仍然可能得到次优的准确率和效率。

In this paper, we want to study and rethink the process of scaling up ConvNets. In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency? Our empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio. Based on this observation, we propose a simple yet effective compound scaling method. Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use $2^N$ times more computational resources, then we can simply increase the network depth by $α^N$, width by $β^N$, and image size by $γ^N$, where α, β, γ are constant coefficients determined by a small grid search on the original small model. Figure 2 illustrates the difference between our scaling method and conventional methods.

本文中，我们研究并重新思考缩放ConvNets的过程。特别是，我们研究了这个中心问题：有没有缩放ConvNets的方法原则，可以得到更好的准确率和效率？我们的经验研究表明，在网络宽度/深度/分辨率各个维度之间的均衡非常关键，而这种均衡可以通过简单的对各个维度进行常数比率的缩放进行。基于这种观察，我们提出了一种简单但有效的复合缩放方法。传统实践中会任意缩放这些因子，我们与其不同，会统一缩放网络深度、宽度和分辨率，使用固定的缩放系数集。比如，如果我们希望用$2^N$倍的计算资源，那么我们可以简单的将网络深度增加$α^N$倍，宽度增加$β^N$倍，图像大小增加$γ^N$倍，其中α, β, γ是常数系数，可以通过在原始小型模型上进行小型网格搜索确定。图2是我们的缩放方法与传统方法的区别。

Figure 2. Model Scaling. (a) is a baseline network example; (b)-(d) are conventional scaling that only increases one dimension of network width, depth, or resolution. (e) is our proposed compound scaling method that uniformly scales all three dimensions with a fixed ratio.

Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image. In fact, previous theoretical (Raghu et al., 2017; Lu et al., 2018) and empirical results (Zagoruyko & Komodakis, 2016) both show that there exists certain relationship between network width and depth, but to our best knowledge, we are the first to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.

直觉上来说，这种复合缩放方法是有合理意义的，因为如果输入图像分辨率更大一点，那么网络需要更多的层来增大感受野，更多的通道来捕获更细粒度的模式。实际上，之前的理论和经验结果都说明，在网络深度和宽度之间有一定关系，但据我们所知，我们是第一个通过经验量化这种关系的，即网络宽度、深度和分辨率之间的关系。

We demonstrate that our scaling method work well on existing MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Notably, the effectiveness of model scaling heavily depends on the baseline network; to go even further, we use neural architecture search (Zoph & Le, 2017; Tan et al., 2019) to develop a new baseline network, and scale it up to obtain a family of models, called EfficientNets. Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets. In particular, our EfficientNet-B7 surpasses the best existing GPipe accuracy (Huang et al., 2018), but using 8.4x fewer parameters and running 6.1x faster on inference. Compared to the widely used ResNet (He et al., 2016), our EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% with similar FLOPS. Besides ImageNet, EfficientNets also transfer well and achieve state-of-the-art accuracy on 5 out of 8 widely used datasets, while reducing parameters by up to 21x than existing ConvNets.

我们的网络缩放方法在现有的MobileNets和ResNet上效果不错。需要指出，模型缩放方法的有效性，严重依赖基准网络；更进一步，我们使用NAS提出了一种新的基准网络，将其缩放可以得到一族模型，称为EfficientNets。图1总结了在ImageNet上的性能，其中我们的EfficientNets明显超过了其他ConvNets。特别是，我们的EfficientNet-B7超过了目前最好的Gpipe准确率，但参数量少了8.4倍，推理速度快了6.1倍。与广泛使用的ResNet相比，我们的EfficientNet-B4与ResNet-50的运算量类似，但准确率从76.3%改进到了82.6%。除了ImageNet，EfficientNets迁移学习性能也非常好，在8个广泛使用的数据集中的5个上，都取得了目前最好的效果，而参数量与已有的ConvNet比，最多降低了21倍。

Figure 1. Model Size vs. ImageNet Accuracy. All numbers are for single-crop, single-model. Our EfficientNets significantly outperform other ConvNets. In particular, EfficientNet-B7 achieves new state-of-the-art 84.4% top-1 accuracy but being 8.4x smaller and 6.1x faster than GPipe. EfficientNet-B1 is 7.6x smaller and 5.7x faster than ResNet-152. Details are in Table 2 and 4.

## 2. Related Work 相关的工作

**ConvNet Accuracy**: Since AlexNet (Krizhevsky et al., 2012) won the 2012 ImageNet competition, ConvNets have become increasingly more accurate by going bigger: while the 2014 ImageNet winner GoogleNet (Szegedy et al., 2015) achieves 74.8% top-1 accuracy with about 6.8M parameters, the 2017 ImageNet winner SENet (Hu et al., 2018) achieves 82.7% top-1 accuracy with 145M parameters. Recently, GPipe (Huang et al., 2018) further pushes the state-of-the-art ImageNet top-1 validation accuracy to 84.3% using 557M parameters: it is so big that it can only be trained with a specialized pipeline parallelism library by partitioning the network and spreading each part to a different accelerator. While these models are mainly designed for ImageNet, recent studies have shown better ImageNet models also perform better across a variety of transfer learning datasets (Kornblith et al., 2019), and other computer vision tasks such as object detection (He et al., 2016; Tan et al., 2019). Although higher accuracy is critical for many applications, we have already hit the hardware memory limit, and thus further accuracy gain needs better efficiency.

**ConvNet准确率**：自从AlexNet赢得ImageNet竞赛以来，ConvNet越变越大，准确率也越来越高：2014年ImageNet获胜者GoogLeNet获得了74.8%的top-1准确率，参数量6.8M，2017年的ImageNet获胜者SENet取得了82.7%的top-1准确率，参数量145M。最近，GPipe进一步将目前最好的ImageNet top-1验证准确率推高到了84.3%，使用了557M参数：模型太大了，只能使用专用的流水并行库训练，将网络分割，每个不同的部分放在不同的加速器上。这些模型主要设计用于ImageNet，但最近的研究表明，在ImageNet上表现更好的模型，也可以通过迁移学习在其他数据集上得到很好的性能，也包括其他计算机视觉任务，如目标检测。虽然更好的准确率对于更多应用来说非常关键，我们已经触碰到了硬件内存的极限，所以进一步改进准确率，需要更好的效率才可以。

**ConvNet Efficiency**: Deep ConvNets are often over-parameterized. Model compression (Han et al., 2016; He et al., 2018; Yang et al., 2018) is a common way to reduce model size by trading accuracy for efficiency. As mobile phones become ubiquitous, it is also common to handcraft efficient mobile-size ConvNets, such as SqueezeNets (Iandola et al., 2016; Gholami et al., 2018), MobileNets (Howard et al., 2017; Sandler et al., 2018), and ShuffleNets (Zhang et al., 2018; Ma et al., 2018). Recently, neural architecture search becomes increasingly popular in designing efficient mobile-size ConvNets (Tan et al., 2019; Cai et al., 2019), and achieves even better efficiency than hand-crafted mobile ConvNets by extensively tuning the network width, depth, convolution kernel types and sizes. However, it is unclear how to apply these techniques for larger models that have much larger design space and much more expensive tuning cost. In this paper, we aim to study model efficiency for super large ConvNets that surpass state-of-the-art accuracy. To achieve this goal, we resort to model scaling.

**ConvNet效率**：深度ConvNet经常参数量过多。模型压缩是一种常用的降低模型大小的方法，主要是以准确率为代价换取效率。由于手机变得十分普遍，所以有很多文章手工设计高效的移动大小的ConvNets，如SqueezeNets，MobileNets和ShuffleNets。最近，NAS变得越来越流行，可以设计高效的移动规模的ConvNets，比手工设计的移动ConvNet甚至得到了更好的效率，主要是通过在很广的范围内调节网络宽度、深度、卷积核类型和大小。但是，怎样将这些技术应用于更大的模型中仍然不是很清楚，这些模型有着更大的设计空间，调整起来的代价大的多。本文中，我们的目标是研究超大型ConvNet的效率，这些超大型模型有着目前最好的准确率。为取得这个目标，我们使用模型缩放技术。

**Model Scaling**: There are many ways to scale a ConvNet for different resource constraints: ResNet (He et al., 2016) can be scaled down (e.g., ResNet-18) or up (e.g., ResNet-200) by adjusting network depth (#layers), while WideResNet (Zagoruyko & Komodakis, 2016) and MobileNets (Howard et al., 2017) can be scaled by network width (#channels). It is also well-recognized that bigger input image size will help accuracy with the overhead of more FLOPS. Although prior studies (Raghu et al., 2017; Lin & Jegelka, 2018; Sharir & Shashua, 2018; Lu et al., 2018) have shown that network deep and width are both important for ConvNets’ expressive power, it still remains an open question of how to effectively scale a ConvNet to achieve better efficiency and accuracy. Our work systematically and empirically studies ConvNet scaling for all three dimensions of network width, depth, and resolutions.

**模型缩放**：有很多缩放ConvNet的的方法，适用于不同的资源限制下：ResNet可以通过调整网络深度（层）来放大(e.g., ResNet-200)或缩小(e.g., ResNet-18)，而WideResNet和MobileNets可以通过调整网络宽度（通道）来改变网络尺度。更大的输入图像大小，也会使准确率得到改进，这也是广为承认的，不过计算量更大一些。虽然之前的研究表明，网络深度和宽度对于ConvNet的表示能力非常重要，但如何有效的缩放ConvNet，以得到更好的效率和准确率，仍然是一个开放问题。我们的工作从经验上系统的研究了三个维度上的ConvNet缩放，即宽度、深度和分辨率。

## 3. Compound Model Scaling 复合模型缩放

In this section, we will formulate the scaling problem, study different approaches, and propose our new scaling method. 本节中，我们公式化阐述网络缩放问题，研究不同的方法，提出新的缩放方法。

### 3.1. Problem Formulation 问题公式化表述

A ConvNet Layer i can be defined as a function: $Y_i = F_i (X_i)$, where $F_i$ is the operator, $Y_i$ is output tensor, $X_i$ is input tensor, with tensor shape <$H_i, W_i, C_i$>(For the sake of simplicity, we omit batch dimension), where $H_i$ and $W_i$ are spatial dimension and $C_i$ is the channel dimension. A ConvNet N can be represented by a list of composed layers: $N = F_k ⊙... ⊙ F_2 ⊙ F_1 (X_1) = ⊙_{j=1...k} F_j (X_1)$. In practice, ConvNet layers are often partitioned into multiple stages and all layers in each stage share the same architecture: for example, ResNet (He et al., 2016) has five stages, and all layers in each stage has the same convolutional type except the first layer performs down-sampling. Therefore, we can define a ConvNet as:

ConvNet层i可以定义为一个函数：$Y_i = F_i (X_i)$，其中$F_i$是算子，$Y_i$是输出张量，$X_i$为输入张量，张量形状为<$H_i, W_i, C_i$>（简化起见，我们忽略了批次维度），其中$H_i$和$W_i$是空间维度，$C_i$是通道维度。一个ConvNet N可以表示为层的组合列表：$N = F_k ⊙... ⊙ F_2 ⊙ F_1 (X_1) = ⊙_{j=1...k} F_j (X_1)$。实际中，ConvNet层经常分为多个阶段，每个阶段中的所有层结构都一样：比如，ResNet有5个阶段，每个阶段的所有层都有相同的卷积类型，除了第一层进行了下采样。所以，我们可以将一个ConvNet定义为：

$$N = ⊙_{i=1...s} F_i^{L_i} (X_{<H_i,W_i,C_i>})$$(1)

where $F_i^{L_i}$ denotes layer $F_i$ is repeated $L_i$ times in stage i, <$H_i, W_i, C_i$> denotes the shape of input tensor X of layer i. Figure 2(a) illustrate a representative ConvNet, where the spatial dimension is gradually shrunk but the channel dimension is expanded over layers, for example, from initial input shape <224, 224, 3> to final output shape <7, 7, 512>.

其中$F_i^{L_i}$表示层$F_i$在阶段i重复了$L_i$次，<$H_i, W_i, C_i$>表示输入张量X在第i层的形状。图2(a)所示的是一类代表性的ConvNet，其中空间维度逐渐缩水，但通道维度逐层扩大，比如，对于初始输入形状<224,224,3>，其最终输出形状为<7,7,512>。

Unlike regular ConvNet designs that mostly focus on finding the best layer architecture $F_i$, model scaling tries to expand the network length ($L_i$), width ($C_i$), and/or resolution ($H_i, W_i$) without changing $F_i$ predefined in the baseline network. By fixing $F_i$, model scaling simplifies the design problem for new resource constraints, but it still remains a large design space to explore different $L_i, C_i, H_i, W_i$ for each layer. In order to further reduce the design space, we restrict that all layers must be scaled uniformly with constant ratio. Our target is to maximize the model accuracy for any given resource constraints, which can be formulated as an optimization problem:

常规ConvNet设计大多聚焦在寻找最好的层架构$F_i$上，模型缩放与之不同，$F_i$是预定义的基准网络，不改变$F_i$，对网络的长度($L_i$)、宽度($C_i$)和/或分辨率($H_i, W_i$)进行缩放。通过保持$F_i$不变，模型缩放对新的资源限制简化了设计问题，但对每一层寻找不同的$L_i, C_i, H_i, W_i$组合，仍然是很大的设计空间。为进一步缩小设计空间，我们限制所有层都必须用常数比率统一缩放。我们的目标是在任何给定的资源限制下，最大化模型准确率，这可以表述为一个优化问题：

$$\begin{split} max_{d,w,r} &Accuracy(N(d,w,r)) \\ s.t. &N(d,w,r) = ⊙_{i=1...s} F̂_i^{d·L̂_i} (X_{< r·Ĥ_i, r·Ŵ_i, w·Ĉ_i>}) \\ &Memory(N)≤target_memory \\ &FLOPS(N)≤target_flops \end{split}$$(2)

where w, d, r are coefficients for scaling network width, depth, and resolution; $F̂_i, L̂_i, Ĥ_i, Ŵ_i, Ĉ_i$ are predefined parameters in baseline network (see Table 1 as an example). 其中w,d,r是缩放网络宽度、深度和分辨率的系数；$F̂_i, L̂_i, Ĥ_i, Ŵ_i, Ĉ_i$是在基准网络中的预定义参数（见表1中的例子）。

### 3.2. Scaling Dimensions 缩放的维度

The main difficulty of problem 2 is that the optimal d, w, r depend on each other and the values change under different resource constraints. Due to this difficulty, conventional methods mostly scale ConvNets in one of these dimensions: 问题(2)的主要困难是，最优的d,w,r是互相依赖的，而且在不同的资源限制下取值也不同。由于这个困难，传统方法主要对下列维度之一进行缩放：

**Depth(d)**: Scaling network depth is the most common way used by many ConvNets (He et al., 2016; Huang et al., 2017; Szegedy et al., 2015; 2016). The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem (Zagoruyko & Komodakis, 2016). Although several techniques, such as skip connections (He et al., 2016) and batch normalization (Ioffe & Szegedy, 2015), alleviate the training problem, the accuracy gain of very deep network diminishes: for example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers. Figure 3 (middle) shows our empirical study on scaling a baseline model with different depth coefficient d, further suggesting the diminishing accuracy return for very deep ConvNets.

**深度(d)**：对网络深度进行缩放是很多ConvNet最常用的方法。直觉上，更深的ConvNet可以捕获到更丰富和更复杂的特征，在新数据集上的泛化能力也更好。但是，更深的网络也更难训练，因为存在梯度消失的问题。虽然有几种技术可以缓解这个训练问题，如跳跃连接、BN等，但非常深的网络的准确率收益却是渐渐消失的：比如，ResNet-1000与ResNet-100的准确率就是类似的，虽然深了很多。图3（中）给出了用不同的深度系数d对基准模型进行缩放的经验研究结果，进一步说明了，对于非常深的网络，准确率回报逐渐消失的问题。

**Width(w)**: Scaling network width is commonly used for small size models (Howard et al., 2017; Sandler et al., 2018; Tan et al., 2019) (In some literature, scaling number of channels is called “depth multiplier”, which means the same as our width coefficient w.). As discussed in (Zagoruyko & Komodakis, 2016), wider networks tend to be able to capture more fine-grained features and are easier to train. However, extremely wide but shallow networks tend to have difficulties in capturing higher level features. Our empirical results in Figure 3 (left) show that the accuracy quickly saturates when networks become much wider with larger w.

**宽度(w)**：用宽度缩放网络在小型模型中经常使用（在一些文献中，通道数量的变化称为“深度乘子”，这与我们的宽度系数w是一样的）。如文献中所述，更宽的网络可以捕捉更细粒度的特征，更容易进行训练。但是，非常宽但很浅的网络可能难以捕获更高层的特征。我们的经验结果表明（图3左），当网络用较大的w变得很宽时，准确率会迅速的饱和。

**Resolution(r)**: With higher resolution input images, ConvNets can potentially capture more fine-grained patterns. Starting from 224x224 in early ConvNets, modern ConvNets tend to use 299x299 (Szegedy et al., 2016) or 331x331 (Zoph et al., 2018) for better accuracy. Recently, GPipe (Huang et al., 2018) achieves state-of-the-art ImageNet accuracy with 480x480 resolution. Higher resolutions, such as 600x600, are also widely used in object detection ConvNets (He et al., 2017; Lin et al., 2017). Figure 3 (right) shows the results of scaling network resolutions, where indeed higher resolutions improve accuracy, but the accuracy gain diminishes for very high resolutions (r = 1.0 denotes resolution 224x224 and r = 2.5 denotes resolution 560x560).

**分辨率(r)**：输入图像分辨率越高，ConvNets可能捕获到更多细粒度的模式。早期的ConvNets从224×224分辨率开始，现代ConvNets倾向于使用299×299或331×331，以获得更高的准确率。最近，GPipe以480×480的分辨率在ImageNet上获得了目前最好的准确率。更高的准确率，如600×600，在目标检测ConvNets中也有广泛的使用。图3（右）所示的是缩放网络分辨率的结果，更高的分辨率确实会改进准确率，但准确率的提升在很高的分辨率时就逐渐消失了（r=1.0表示分辨率224×224，r=2.5表示分辨率560×560）。

The above analyses lead us to the first observation: 上面的分析可以得到第一个观察结论：

Observation 1 – Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models. 观察1 - 放大宽度、深度和分辨率的任意网络维度可以改进准确率，但对于更大的模型，准确率收益逐渐消失。

Figure 3. Scaling Up a Baseline Model with Different Network Width (w), Depth (d), and Resolution (r) Coefficients. Bigger networks with larger width, depth, or resolution tend to achieve higher accuracy, but the accuracy gain quickly saturate after reaching 80%, demonstrating the limitation of single dimension scaling. Baseline network is described in Table 1.

### 3.3. Compound Scaling 复合缩放

We empirically observe that different scaling dimensions are not independent. Intuitively, for higher resolution images, we should increase network depth, such that the larger receptive fields can help capture similar features that include more pixels in bigger images. Correspondingly, we should also increase network depth when resolution is higher, in order to capture more fine-grained patterns with more pixels in high resolution images. These intuitions suggest that we need to coordinate and balance different scaling dimensions rather than conventional single-dimension scaling.

经验上，我们观察到，不同的缩放尺度并不是独立的。直觉上来说，对于更高分辨率的图像，我们应当增加网络深度，这样更大的感受野可以帮助捕获更大的图像中更多像素点类似的特征。对应的，分辨率更高时，我们应当也增加网络宽度，为捕获更细粒度的模式。这些直觉说明，我们需要协调并平衡不同的缩放维度，而不是传统的单维度缩放。

To validate our intuitions, we compare width scaling under different network depths and resolutions, as shown in Figure 4. If we only scale network width w without changing depth (d=1.0) and resolution (r=1.0), the accuracy saturates quickly. With deeper (d=2.0) and higher resolution (r=2.0), width scaling achieves much better accuracy under the same FLOPS cost. These results lead us to the second observation:

为验证我们的直觉，我们比较了不同的网络深度和分辨率下的宽度缩放效果，如图4所示。如果我们保持深度(d=1.0)和分辨率(r=1.0)不变，只缩放网络宽度w，那么准确率很快就饱和了。随着深度更深(d=2.0)，分辨率更高(r=2.0)，宽度缩放在相同的FLOPS下下得到了好很多的准确率。这种结果可以得出第二个观察结论：

Observation 2 – In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling. 观察2 - 为追求更好的准确率和效率，在ConvNet缩放时，保持网络宽度、深度和分辨率的平衡至关重要。

Figure 4. Scaling Network Width for Different Baseline Networks. Each dot in a line denotes a model with different width coefficient (w). All baseline networks are from Table 1. The first baseline network (d=1.0, r=1.0) has 18 convolutional layers with resolution 224x224, while the last baseline (d=2.0, r=1.3) has 36 layers with resolution 299x299.

In fact, a few prior work (Zoph et al., 2018; Real et al., 2019) have already tried to arbitrarily balance network width and depth, but they all require tedious manual tuning. 实际上，之前的一些工作已经尝试在网络深度和宽度之间进行一些平衡，但它们都需要繁琐的人力调节。

In this paper, we propose a new compound scaling method, which use a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way: 本文中，我们提出一种新的复合缩放方法，使用一个复合系数φ来统一缩放网络宽度、深度和分辨率，是一种有原则的方式：

$$depth: d = α^φ, width: w = β^φ, resolution: r = γ^φ, s.t. α · β^2 · γ^2 ≈ 2, α ≥ 1, β ≥ 1, γ ≥ 1$$(3)

where α, β, γ are constants that can be determined by a small grid search. Intuitively, φ is a user-specified coefficient that controls how many more resources are available for model scaling, while α, β, γ specify how to assign these extra resources to network width, depth, and resolution respectively. Notably, the FLOPS of a regular convolution op is proportional to d, $w^2$, $r^2$, i.e., doubling network depth will double FLOPS, but doubling network width or resolution will increase FLOPS by four times. Since convolution ops usually dominate the computation cost in ConvNets, scaling a ConvNet with equation 3 will approximately increase total FLOPS by $(α · β^2 · γ^2)^φ$. In this paper, we constraint $α · β^2 · γ^2$ ≈ 2 such that for any new φ, the total FLOPS will approximately increase by $2^φ$(FLOPS may differ from theocratic value due to rounding).

其中α, β, γ为常数，可以通过一个小型网格搜索确定。直觉上来说，φ是一个用户指定的系数，控制模型缩放所需要的计算资源，而α, β, γ将这些额外的计算资源分别分配给宽度、深度和分辨率。值得注意的是，常规卷积算子的FLOPS与d, $w^2$, $r^2$成正比，即网络深度加倍会使FLOPS加倍，网络宽度加倍或分辨率加倍会使FLOPS增加到4倍。由于卷积算子通常是ConvNets中计算量的主要部分，用式(3)对ConvNet进行缩放，会使FLOPS大约增加至$(α · β^2 · γ^2)^φ$倍。本文中，我们限制$α · β^2 · γ^2$ ≈ 2，这样对于任意新的φ，总计FLOPS会大约增加$2^φ$倍。

## 4. EfficientNet Architecture 网络架构

Since model scaling does not change layer operators $F̂_i$ in baseline network, having a good baseline network is also critical. We will evaluate our scaling method using existing ConvNets, but in order to better demonstrate the effectiveness of our scaling method, we have also developed a new mobile-size baseline, called EfficientNet.

由于模型缩放并不改变基准网络$F̂_i$中层的算子，有一个很好的基准网络就非常关键了。我们将使用现有的ConvNets评估我们的缩放方法，但为了更好的证明我们缩放方法的有效性，我们还提出了一种新的移动设备规模的基准模型，称为EfficientNet。

Inspired by (Tan et al., 2019), we develop our baseline network by leveraging a multi-objective neural architecture search that optimizes both accuracy and FLOPS. Specifically, we use the same search space as (Tan et al., 2019), and use $ACC(m)×[FLOPS(m)/T]^w$ as the optimization goal, where ACC(m) and FLOPS(m) denote the accuracy and FLOPS of model m, T is the target FLOPS and w=-0.07 is a hyperparameter for controlling the trade-off between accuracy and FLOPS. Unlike (Tan et al., 2019; Cai et al., 2019), here we optimize FLOPS rather than latency since we are not targeting any specific hardware device. Our search produces an efficient network, which we name EfficientNet-B0. Since we use the same search space as (Tan et al., 2019), the architecture is similar to MnasNet, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Table 1 shows the architecture of EfficientNet-B0. Its main building block is mobile inverted bottleneck MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation optimization (Hu et al., 2018).

受之前工作启发，我们利用一种多目标神经架构搜索，对准确率和FLOPS都进行优化，来得到我们的基准网络。特别是，我们使用与之相同的搜索空间，使用$ACC(m)×[FLOPS(m)/T]^w$作为优化目标，其中ACC(m)和FLOPS(m)分别表示模型m的准确率和FLOPS，T是目标FLOPS，w=-0.07是一个超参数，控制准确率和FLOPS间的折中。与其不同的是，这里我们优化FLOPS而不是延迟，因为我们没有特别的硬件设备目标。我们的搜索得到了一种高效的网络，我们命名为EfficientNet-B0。由于我们使用与其相同的搜索空间，其架构与MnasNet类似，除了我们的EfficientNet略大一些，因为目标FLOPS更大一些（我们的FLOPS目标为400M）。表1是EfficientNet-B0的架构。其主要模块是移动的逆瓶颈MBConv，我们还在其上增加了SE优化。

Table 1. EfficientNet-B0 baseline network – Each row describes a stage i with $L̂_i$ layers, with input resolution <$Ĥ_i, Ŵ_i$> and output channels $Ĉ_i$. Notations are adopted from equation 2.

stage i | Operator $F̂_i$ | Resolution $Ĥ_i×Ŵ_i$ | Channels $Ĉ_i$ | Layers $L̂_i$
--- | --- | --- | --- | ---
1 | Conv3x3 | 224×224 | 32 | 1
2 | MBConv1, k3x3 | 112×112 | 16 | 1
3 | MBConv6, k3x3 | 112×112 | 24 | 2
4 | MBConv6, k5x5 | 56×56 | 40 | 2
5 | MBConv6, k3x3 | 28×28 | 80 | 3
6 | MBConv6, k5x5 | 28×28 | 112 | 3
7 | MBConv6, k5x5 | 14×14 | 192 | 4
8 | MBConv6, k3x3 | 7×7 | 320 | 1
9 | Conv1x1 & Pooling & FC | 7×7 | 1280 | 1

Starting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps: 从我们的基准EfficientNet-B0开始，我们应用我们的复合缩放方法，用两步对其进行缩放：

- STEP 1: we first fix φ = 1, assuming twice more resources available, and do a small grid search of α, β, γ based on Equation 2 and 3. In particular, we find the best values for EfficientNet-B0 are α = 1.2, β = 1.1, γ = 1.15, under constraint of $α · β^2 · γ^2$ ≈ 2. 第1步：我们固定φ = 1，假设有2倍的计算资源可用，基于式(2)和式(3)对α, β, γ进行一个小型网格搜索。特别的，我们发现对于EfficientNet-B0的最优值为α = 1.2, β = 1.1, γ = 1.15，这是在$α · β^2 · γ^2$ ≈ 2的约束下得到的。

- STEP 2: we then fix α, β, γ as constants and scale up baseline network with different φ using Equation 3, to obtain EfficientNet-B1 to B7 (Details in Table 2). 第2步：我们固定α, β, γ为常数，使用式(3)用不同的φ缩放基准网络，得到EfficientNet-B0到B7（详见表2）。

Notably, it is possible to achieve even better performance by searching for α, β, γ directly around a large model, but the search cost becomes prohibitively more expensive on larger models. Our method solves this issue by only doing search once on the small baseline network (step 1), and then use the same scaling coefficients for all other models (step 2).

值得注意的是，对一个大型模型，直接搜索α, β, γ可能得到更好的性能，但搜索代价在更大的模型上变得及其昂贵。我们的方法，通过在小型基准网络上进行一次性搜索，解决这个问题，然后对所有其他模型使用相同的缩放系数。

Table 2. EfficientNet Performance Results on ImageNet (Russakovsky et al., 2015). All EfficientNet models are scaled from our baseline EfficientNet-B0 using different compound coefficient φ in Equation 3. ConvNets with similar top-1/top-5 accuracy are grouped together for efficiency comparison. Our scaled EfficientNet models consistently reduce parameters and FLOPS by an order of magnitude (up to 8.4x parameter reduction and up to 16x FLOPS reduction) than existing ConvNets.

Model | Top1-1 Acc. | Top-5 Acc. | Params | Ratio-to-EfficientNet | FLOPS | Ratio-to-EfficientNet
--- | --- | --- | --- | --- | --- | ---
EfficientNet-B0 | 76.3% | 93.2% | 5.3M | 1x | 0.39B | 1x
ResNet-50 | 76.0% | 93.0% | 26M | 4.9x | 4.1B | 11x
DenseNet-169 | 76.2% | 93.2% | 14M | 2.6x | 3.5B | 8.9x
---- |
EfficientNet-B1 | 78.8% | 94.4% | 7.8M | 1x | 0.70B | 1x
ResNet-152 | 77.8% | 93.8% | 60M | 7.6x | 11B | 16x
DenseNet-264 | 77.9% | 93.9% | 34M | 4.3x | 6.0B | 8.6x
Inception-v3 | 78.8% | 94.4% | 24M | 3.0x | 5.7B | 8.1x
Xception | 79.0% | 94.5% | 23M | 3.0x | 8.4B | 12x
---- | 
EfficientNet-B2 | 79.8% | 94.9% | 9.2M | 1x | 1.0B | 1x
Inception-v4 | 80.0% | 95.0% | 48M | 5.2x | 13B | 13x
Inception-ResNet-v2 | 80.1% | 95.1% | 56M | 6.1x | 13B | 13x
---- |
EfficientNet-B3 | 81.1% | 95.5% | 12M | 1x | 1.8B | 1x
ResNeXt-101 | 80.9% | 95.6% | 84M | 7.0x | 32B | 18x
PolyNet | 81.3% | 95.8% | 92M | 7.7x | 35B | 19x
---- |
EfficientNet-B4 | 82.6% | 96.3% | 19M | 1x | 4.2B | 1x
SENet | 82.7% | 95.2% | 146M | 7.7x | 42B | 10x
NASNet-A | 82.7% | 96.2% | 89M | 4.7x | 24B | 5.7x
AmoebaNet-A | 82.8% | 96.1% | 87M | 4.6x | 23B | 5.5x
PNASNet | 82.9% | 96.2% | 86M | 4.5x | 23B | 6.0x
---- |
EfficientNet-B5 | 83.3% | 96.7% | 30M | 1x | 9.9B | 1x
AmoebaNet-C | 83.5% | 96.5% | 155M | 5.2x | 41B | 4.1x
---- |
EfficientNet-B6 | 84.0% | 96.9% | 43M | 1x | 19B | 1x
---- |
EfficientNet-B7 | 84.4% | 97.1% | 66M | 1x | 37B | 1x
GPipe | 84.3% | 97.0% | 557M | 8.4x | - | -

## 5. Experiments 试验

In this section, we will first evaluate our scaling method on existing ConvNets and the new proposed EfficientNets. 本节中，我们首先在现有的ConvNets和我们提出的EfficientNets上评估我们的缩放方法。

### 5.1. Scaling Up MobileNets and ResNets 缩放MobileNets和ResNets

As a proof of concept, we first apply our scaling method to the widely-used MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Table 3 shows the ImageNet results of scaling them in different ways. Compared to other single-dimension scaling methods, our compound scaling method improves the accuracy on all these models, suggesting the effectiveness of our proposed scaling method for general existing ConvNets.

作为概念验证，我们首先对广泛应用的MobileNets和ResNets使用我们的缩放方法。表3给出了使用不同缩放方法在ImageNet上得到的结果。与现有的单维度缩放方法相比，我们的复合缩放方法对所有模型的准确率都有所改进，说明我们提出的缩放方法对一般的现有方法的有效性。

Table 3. Scaling Up MobileNets and ResNet.

Model | FLOPS | Top-1 Acc.
--- | --- | ---
Baseline MobileNetV1 | 0.6B | 70.6%
Scale MobileNetV1 by width(w=2) | 2.2B | 74.2%
Scale MobileNetV1 by resolution(r=2) | 2.2B | 72.7%
compound scale(d=1.4,w=1.2,r=1.3) | 2.3B | 75.6%
Baseline MobileNetV2 | 0.3B | 72.0%
Scale MobileNetV2 by depth(d=4) | 1.2B | 76.8%
Scale MobileNetV2 by width(w=2) | 1.1B | 76.4%
Scale MobileNetV2 by resolution(r=2) | 1.2B | 74.8%
MobileNetV2 compound scale | 1.3B | 77.4%
Baseline ResNet-50 | 4.1B | 76.0%
Scale ResNet-50 by depth(d=4) | 16.2B | 78.1%
Scale ResNet-50 by withd(w=2) | 14.7B | 77.7%
Scale ResNet-50 by resolution(r=2) | 16.4B | 77.5%
ResNet-50 compound scale | 16.7B | 78.8%

### 5.2. ImageNet Results for EfficientNet

We train our EfficientNet models on ImageNet using similar settings as (Tan et al., 2019): RMSProp optimizer with decay 0.9 and momentum 0.9; batch norm momentum 0.99; weight decay 1e-5; initial learning rate 0.256 that decays by 0.97 every 2.4 epochs. We also use swish activation (Ramachandran et al., 2018; Elfwing et al., 2018), fixed AutoAugment policy (Cubuk et al., 2019), and stochastic depth (Huang et al., 2016) with drop connect ratio 0.3. As commonly known that bigger models need more regularization, we linearly increase dropout (Srivastava et al., 2014) ratio from 0.2 for EfficientNet-B0 to 0.5 for EfficientNet-B7.

我们在ImageNet上使用与(Tan et al., 2019)类似的设置训练我们的EfficientNet模型：RMSProp优化器，衰减0.9，动量0.9，BN动量0.99，权重衰减1e-5，初始学习速率0.256，每2.4 epochs衰减0.97。我们使用swish激活，固定的AutoAugment策略，drop connect率0.3的随机深度。众所周知的是，更大的模型需要更多的正则化，我们我们将dropout率从0.2线性增加到EfficientNet-B0的0.2，到EfficientNet-B7的0.5。

Table 2 shows the performance of all EfficientNet models that are scaled from the same baseline EfficientNet-B0. Our EfficientNet models generally use an order of magnitude fewer parameters and FLOPS than other ConvNets with similar accuracy. In particular, our EfficientNet-B7 achieves 84.4% top1 / 97.1% top-5 accuracy with 66M parameters and 37B FLOPS, being more accurate but 8.4x smaller than the previous best GPipe (Huang et al., 2018).

表2给出了所有EfficientNet模型的性能，都是从相同的EfficientNet-B0缩放得到的。我们的EfficientNet模型与取得类似准确率的ConvNets相比，使用的参数量和FLOPS一般少了一个数量级。特别的，我们的EfficientNet-B7的ImageNet top-1和top-5准确率为84.4%, 97.1%，参数量为66M，FLOPS 37B，与之前最好的GPipe相比，准确率更高，但是小了8.4x。

Figure 1 and Figure 5 illustrates the parameters-accuracy and FLOPS-accuracy curve for representative ConvNets, where our scaled EfficientNet models achieve better accuracy with much fewer parameters and FLOPS than other ConvNets. Notably, our EfficientNet models are not only small, but also computational cheaper. For example, our EfficientNet-B3 achieves higher accuracy than ResNeXt-101 (Xie et al., 2017) using 18x fewer FLOPS.

图1和图5给出了代表性ConvNets的参数量-准确率和FLOPS-准确率曲线，其中我们的缩放EfficientNet模型用少的多的参数量和FLOPS取得了更高的准确率。值得注意的是，我们的EfficientNet模型不仅小，而且计算量更少。比如，我们的EfficientNet-B3与ResNeXt-101相比，使用的FLOPS小了18x，准确率却更高。

Figure 5. FLOPS vs. ImageNet Accuracy – Similar to Figure 1 except it compares FLOPS rather than model size.

To validate the computational cost, we have also measured the inference latency for a few representative CovNets on a real CPU as shown in Table 4, where we report average latency over 20 runs. Our EfficientNet-B1 runs 5.7x faster than the widely used ResNet-152 (He et al., 2016), while EfficientNet-B7 runs about 6.1x faster than GPipe (Huang et al., 2018), suggesting our EfficientNets are indeed fast on real hardware.

为验证计算量，我们还测量了一些有代表性的ConvNets在CPU上的推理耗时，如表4所示，我们给出20次运行的平均耗时。我们的EfficientNet-B1运行时间比广泛使用的ResNet-152快了5.7x，而EfficientNet-B7比GPipe快了大约6.1倍，说明我们的EfficientNet在真实硬件上却是运行更快。

Table 4. Inference Latency Comparison – Latency is measured with batch size 1 on a single core of Intel Xeon CPU E5-2690.

Model | Acc. | Latency
--- | --- | ---
ResNet-152 | 77.8% | 0.554s
EfficientNet-B1 | 78.8% 0.098s
Speedup | 5.7x
GPipe | 84.3% | 19.0s
EfficientNet-B7 | 84.4% | 3.1s
Speedup | 6.1x

### 5.3. Transfer Learning Results for EfficientNet 迁移学习结果

We have also evaluated our EfficientNet on a list of commonly used transfer learning datasets, as shown in Table 6. We borrow the same training settings from (Kornblith et al., 2019) and (Huang et al., 2018), which take ImageNet pretrained checkpoints and finetune on new datasets.

我们还评估了EfficientNet在常用的迁移学习数据集上的性能，如表6所示。我们使用了两篇文章相同的训练设置，即使用ImageNet预训练的检查点，并在新数据集上进行精调。

Table 6. Transfer Learning Datasets.

Dataset | Train Size | Test Size | Classes
--- | --- | --- | ---
CIFAR-10 | 50000 | 10000 | 10
CIFAR-100 | 50000 | 10000 | 100
Birdsnap | 47386 | 2443 | 500
Stanford Cars | 8144 | 8041 | 196
Flowers | 2040 | 6149 | 102
FGVC Aircraft | 6667 | 3333 | 100
Oxford-IIIT Pets | 3680 | 3369 | 37
Food-101 | 75750 | 25250 | 101

Table 5 shows the transfer learning performance: (1) Compared to public available models, such as NASNet-A (Zoph et al., 2018) and Inception-v4 (Szegedy et al., 2017), our EfficientNet models achieve better accuracy with 4.7x average (up to 21x) parameter reduction. (2) Compared to state-of-the-art models, including DAT (Ngiam et al., 2018) that dynamically synthesizes training data and GPipe (Huang et al., 2018) that is trained with specialized pipeline parallelism, our EfficientNet models still surpass their accuracy in 5 out of 8 datasets, but using 9.6x fewer parameters.

表5给出了迁移学习的性能：(1)与公开可用的模型相比，如NASNet-A和Inception-v4，我们的EfficientNet模型得到了更好的准确率，但是平均参数降低了4.7x（最多21x）；(2)与目前最好的模型相比，包括动态合成训练数据的DAT，和使用特殊的并行流程训练的GPipe，我们的EfficientNet在8个数据集中的5个上面超过了它们的准确率，但使用的参数量少了9.6倍。

Table 5. EfficientNet Performance Results on Transfer Learning Datasets. Our scaled EfficientNet models achieve new state-of-the-art accuracy for 5 out of 8 datasets, with 9.6x fewer parameters on average.

| | Model | Acc. | Param | Our Model | Acc. | Param(ratio) | Model | Acc. | Param | Our Model | Acc. | Param(ratio)
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
CIFAR-10 | NASNet-A | 98.0% | 85M | EfficientNet-B0 | 98.1% | 4M(21x) | GPipe | 99.0% | 556M | EfficientNet-B7 | 98.9% | 64M(8.7x)
CIFAR-100 | NASNet-A | 87.5% | 85M | EfficientNet-B0 | 88.1% | 4M(21x) | GPipe | 91.3% | 556M | EfficientNet-B7 | 91.7% | 64M(8.7x)
Birdsnap | Inception-v4 | 81.8% | 41M | EfficientNet-B5 | 82.0% | 28M(1.5x) | GPipe | 83.6% | 556M | EfficientNet-B7 | 84.3% | 64M(8.7x)
Stanford Cars | Inception-v4 | 93.4% | 41M | EfficientNet-B3 | 93.6% | 10M(4.1x) | DAT | 94.8% | - | EfficientNet-B7 | 94.7% | -
Flowers | Inception-v4 | 98.5% | 41M | EfficientNet-B5 | 98.5% | 28M(1.5x) | DAT | 97.7% | - | Efficient-B7 | 98.8% | -
FGVC Aircraft | Inception-v4 | 90.9% | EfficientNet-B3 | 10M(4.1x) | DAT | 92.9% | - | EfficientNet-B7 | 92.9% | -
Oxford-IIIT Pets | ResNet-152 | 94.5% | 58M | EfficientNet-B4 | 94.8% | 17M(5.6x) | GPipe | 95.9% | 556M | EfficientNet-B6 | 95.4% | 41M(14x)
Food-101 | Inception-v4 | 90.8% | 41M | EfficientNet-B4 | 91.5% | 17M(2.4x) | GPipe | 93.0% | 556M | EfficientNet-B7 | 93.0% | 64M(8.7x)
Geo-Mean | 4.7x | 9.6x

Figure 6 compares the accuracy-parameters curve for a variety of models. In general, our EfficientNets consistently achieve better accuracy with an order of magnitude fewer parameters than existing models, including ResNet (He et al., 2016), DenseNet (Huang et al., 2017), Inception (Szegedy et al., 2017), and NASNet (Zoph et al., 2018).

图6比较了很多不同模型的准确率-参数量曲线。一般来说，我们的EfficientNets与现有模型相比，总是会得到更好的准确率，而且使用的参数量会少一个数量级，比较对象包括ResNet，DenseNet，Inception和NASNet。

Figure 6. Model Parameters vs. Transfer Learning Accuracy – All models are pretrained on ImageNet and finetuned on new datasets.

## 6. Discussion 讨论

To disentangle the contribution of our proposed scaling method from the EfficientNet architecture, Figure 8 compares the ImageNet performance of different scaling methods for the same EfficientNet-B0 baseline network. In general, all scaling methods improve accuracy with the cost of more FLOPS, but our compound scaling method can further improve accuracy, by up to 2.5%, than other single-dimension scaling methods, suggesting the importance of our proposed compound scaling.

为区分我们提出的缩放方法和EfficientNet架构之间的贡献，图8比较了使用EfficientNet-B0基准架构，但使用不同缩放方法，在ImageNet上的性能。一般来说，所有的缩放方法在使用更多FLOPS时，都会改进准确率，但我们的复合缩放方法可以更一步的改进准确率，比其他单维度缩放方法，最高达到2.5%，说明我们提出的复合缩放方法的有效性和重要性。

Figure 8. Scaling Up EfficientNet-B0 with Different Methods.

In order to further understand why our compound scaling method is better than others, Figure 7 compares the class activation map (Zhou et al., 2016) for a few representative models with different scaling methods. All these models are scaled from the same baseline, and their statistics are shown in Table 7. Images are randomly picked from ImageNet validation set. As shown in the figure, the model with compound scaling tends to focus on more relevant regions with more object details, while other models are either lack of object details or unable to capture all objects in the images.

为更进一步理解为什么我们的复合缩放方法比其他方法更好，图7比较了几种有代表性模型的类别激活图，使用了不同的缩放方法。所有这些模型都是从相同的基准上进行缩放，其统计数据在表7中进行了比较。图像是从ImageNet验证集中随机选取的。如图所示，复合缩放方法的模型会聚焦在更相关的区域中，包含更多的目标细节，而其他模型要么缺少目标细节，或不能捕获图像中的所有目标。

Figure 7. Class Activation Map (CAM) (Zhou et al., 2016) for Different Models in Table 7 - Our compound scaling method allowsthe scaled model (last column) to focus on more relevant regions with more object details. Model details are in Table 7.

Table 7. Scaled Models Used in Figure 7.

Model | FLOPS | Top-1 Acc.
--- | --- | ---
Baseline model (EfficientNet-B0) | 0.4B | 76.3%
Scale model by depth (d=4) | 1.8B | 79.0%
Scale model by width (w=2) | 1.8B | 78.9%
Scale model by resolution (r=2) | 1.9B | 79.1%
Compound Scale (dd =1.4, w =1.2, r =1.3) | 1.8B | 81.1%

## 7. Conclusion 结论

In this paper, we systematically study ConvNet scaling and identify that carefully balancing network width, depth, and resolution is an important but missing piece, preventing us from better accuracy and efficiency. To address this issue, we propose a simple and highly effective compound scaling method, which enables us to easily scale up a baseline ConvNet to any target resource constraints in a more principled way, while maintaining model efficiency. Powered by this compound scaling method, we demonstrate that a mobile-size EfficientNet model can be scaled up very effectively, surpassing state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS, on both ImageNet and five commonly used transfer learning datasets.

在本文中，我们系统的研究了ConvNet缩放方法，并发现小心的在网络宽度、深度和分辨率之间平衡是一个重要的被忽略的因素，这样可以获得更好的准确率和效率。为解决这个问题，我们提出了一种简单但高效的复合缩放方法，这使我们可以很容易的缩放一个基准ConvNet，达到任何目标计算资源限制，缩放方法更加有规则，而且保持了模型效率。采用这种复合缩放方法，我们证明了一个移动规模的EfficientNet模型，可以非常高效的进行缩放，超过了目前最好的模型，而且参数数量和FLOPS少了一个数量级，在ImageNet和常用的迁移学习数据集上都进行了试验。