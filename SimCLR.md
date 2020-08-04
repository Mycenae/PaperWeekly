# A Simple Framework for Contrastive Learning of Visual Representations

Ting Chen et. al. Google Research, Brain Team

## 0. Abstract

This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100× fewer labels.

本文提出了SimCLR：一种简单的进行视觉表示的对比学习的框架。我们简化了最近提出的对比自监督学习算法，不需要专门的架构或存储组。为理解什么使对比预测任务来学习有用的表示，我们系统的研究我们的架构的主要组成部分。我们表示，(1)数据扩增在定义高效的预测任务中扮演了关键的角色，(2)在表示和对比损失之间，提出一种可学习的非线性变换，会显著改进学习的表示的质量，(3)与监督学习相比，对比学习从更大的batch规模，和更多的训练步骤中受益。将这些发现组合到一起，我们在ImageaNet的自监督和半监督任务中显著超过了之前的方法。在SimCLR学习到自监督表示上训练的一个线性分类器，得到了76.5%的top-1准确率，比之前最好的结果相对改进了7%，与ResNet-50的监督版相媲美。如果在1是%的标签上进行精调，我们得到了85.8%的top-5准确率，用少了100x的标签超过了AlexNet的水平。

## 1. Introduction

Learning effective visual representations without human supervision is a long-standing problem. Most mainstream approaches fall into one of two classes: generative or discriminative. Generative approaches learn to generate or otherwise model pixels in the input space (Hinton et al., 2006; Kingma & Welling, 2013; Goodfellow et al., 2014). However, pixel-level generation is computationally expensive and may not be necessary for representation learning. Discriminative approaches learn representations using objective functions similar to those used for supervised learning, but train networks to perform pretext tasks where both the inputs and labels are derived from an unlabeled dataset. Many such approaches have relied on heuristics to design pretext tasks (Doersch et al., 2015; Zhang et al., 2016; Noroozi & Favaro, 2016; Gidaris et al., 2018), which could limit the generality of the learned representations. Discriminative approaches based on contrastive learning in the latent space have recently shown great promise, achieving state-of-the-art results (Hadsell et al., 2006; Dosovitskiy et al., 2014; Oord et al., 2018; Bachman et al., 2019).

在没有人类监督的情况下学习有效的视觉表示，是一个长期的问题。多数主要的方法都归于以下两类：生成式的或判别式的。生成式的方法在输入空间中生成像素或对像素进行建模。但是，像素级的生成计算量很大，对于表示学习不一定是必须的。判别式的方法使用目标函数来学习表示，目标函数与用于监督学习的类似，但训练网络用于pretext任务，其中输入和标签都是从无标记的数据集推断出来的。很多这样的方法都依赖于启发式来定义pretext任务，这可能限制了学习到的表示的一般性。在隐空间中基于对比学习的判别式的方法最近表现出了很大的希望，获得了目前最好的效果。

In this work, we introduce a simple framework for contrastive learning of visual representations, which we call SimCLR. Not only does SimCLR outperform previous work (Figure 1), but it is also simpler, requiring neither specialized architectures (Bachman et al., 2019; Hénaff et al., 2019) nor a memory bank (Wu et al., 2018; Tian et al., 2019; He et al., 2019a; Misra & van der Maaten, 2019).

本文中，我们提出了一种简单的框架，进行视觉表示的对比学习，我们称之为SimCLR。SimCLR逼近超过了之前的工作，同时还更简单，不需要专门的架构，或一个存储体。

In order to understand what enables good contrastive representation learning, we systematically study the major components of our framework and show that: 为理解什么成就了好的对比表示学习，我们系统的研究了我们框架的主要组件，表示：

- Composition of multiple data augmentation operations is crucial in defining the contrastive prediction tasks that yield effective representations. In addition, unsupervised contrastive learning benefits from stronger data augmentation than supervised learning. 多个数据扩增操作，对于定义对比预测任务非常关键，可以生成有效表示。另外，无监督对比学习从更强的数据扩增中，获益比监督学习更多。

- Introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations. 在表示和对比损失之间，提出了一种可学习的非线性变换，极大的改进了学习到的表示的质量。

- Representation learning with contrastive cross entropy loss benefits from normalized embeddings and an appropriately adjusted temperature parameter. 采用对比交叉熵的表示学习，会归一化的嵌入和合适调整的温度参数中受益。

- Contrastive learning benefits from larger batch sizes and longer training compared to its supervised counterpart. Like supervised learning, contrastive learning benefits from deeper and wider networks. 与监督学习相比，对比学习从更大的batch规模和更长的训练中受益更多。与监督学习类似，更深更宽的网络也是加强对比学习。

We combine these findings to achieve a new state-of-the-art in self-supervised and semi-supervised learning on ImageNet ILSVRC-2012 (Russakovsky et al., 2015). Under the linear evaluation protocol, SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art (Hénaff et al., 2019). When fine-tuned with only 1% of the ImageNet labels, SimCLR achieves 85.8% top-5 accuracy, a relative improvement of 10% (Hénaff et al., 2019). When fine-tuned on other natural image classification datasets, SimCLR performs on par with or better than a strong supervised baseline (Kornblith et al., 2019) on 10 out of 12 datasets.

我们将这些发现结合到一起，在ImageNet ILSVRC-2012中的自监督和半监督学习中获得了新的目前最好的结果。在线性评估的协议下，SimCLR获得了76.5是%的top-1准确率，比之前的最好结果有7%的相对改进。如果使用1%的ImageNet标签进行精调，SimCLR获得了86.5%的top-5准确率，相对改进幅度达10%。当在其他自然图像分类数据集上进行精调，SimCLR在12个数据集中的10个中，与一个很强的有监督基准相比，效果接近或更好。

## 2. Method

### 2.1. The Contrastive Learning Framework 对比学习框架

Inspired by recent contrastive learning algorithms (see Section 7 for an overview), SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. As illustrated in Figure 2, this framework comprises the following four major components.

受到最近的对比学习算法的启发（见第7节的综述），SimCLR通过对相同数据的不同扩增的视角，在隐空间的对比度损失的一致性进行最大化，从而学习表示。如图2所示，框架由以下四个主要部件组成。

- A stochastic data augmentation module that transforms any given data example randomly resulting in two correlated views of the same example, denoted $\tilde x_i$ and $\tilde x_j$, which we consider as a positive pair. In this work, we sequentially apply three simple augmentations: random cropping followed by resize back to the original size, random color distortions, and random Gaussian blur. As shown in Section 3, the combination of random crop and color distortion is crucial to achieve a good performance.

随机数据扩增模块，将任意给定的数据样本随机变换，得到相同样本的两个相关视图，表示为$\tilde x_i$和$\tilde x_j$，这我们认为是一个正对。本文中，我们顺序的应用三个简单的扩增：随机剪切，然后是变换到原始图像大小，随机色彩变换，和随机高斯模糊。如第3节所示，随机剪切和色彩变换的组合，对于取得一个好的性能很关键。

- A neural network base encoder f (·) that extracts representation vectors from augmented data examples. Our framework allows various choices of the network architecture without any constraints. We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain $h_i = f(\tilde x_i)=ResNet(\tilde x_i)$ where $h_i ∈ R^d$ is the output after the average pooling layer.

神经网络基础编码器f(·)，从扩增的数据样本中提取出表示向量。我们的框架可以选择不同的网络架构，没有任何限制。简单起见，我们采用常用的ResNet，得到$h_i = f(\tilde x_i)=ResNet(\tilde x_i)$，其中$h_i ∈ R^d$，是平均池化层后的输出。

- A small neural network projection head g(·) that maps representations to the space where contrastive loss is applied. We use a MLP with one hidden layer to obtain $z_i = g(h_i) = W^{(2)} σ(W^{(1)} h_i)$ where σ is a ReLU non-linearity. As shown in section 4, we find it beneficial to define the contrastive loss on $z_i$’s rather than $h_i$’s.

小的神经网络投影头g(·)，将表示映射到使用对比损失的空间。我们使用带有一个隐藏层的MLP，得到$z_i = g(h_i) = W^{(2)} σ(W^{(1)} h_i)$，其中σ是一个ReLU非线性函数。如第4部分所述，我们发现在$z_i$上定义对比损失，比在$h_i$上更加更加有好处。

- A contrastive loss function defined for a contrastive prediction task. Given a set {$\tilde x_k$} including a positive pair of examples $\tilde x_i$ and $\tilde x_j$, the contrastive prediction task aims to identify $\tilde x_j$ in {$\tilde x_k$}$_{k\neq i}$ for a given $\tilde x_i$.

对比损失函数，用于对比预测任务。给定集合{$\tilde x_k$}，包含一个正对样本$\tilde x_i$和$\tilde x_j$，对比预测任务目标是，在{$\tilde x_k$}$_{k\neq i}$中，对一个给定的$\tilde x_i$，识别$\tilde x_j$。

We randomly sample a minibatch of N examples and define the contrastive prediction task on pairs of augmented examples derived from the minibatch, resulting in 2N data points. We do not sample negative examples explicitly. Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N−1) augmented examples within a minibatch as negative examples. Let $sim(u,v) = u^⊤v/∥u∥∥v∥$ denote the cosine similarity between two vectors u and v. Then the loss function for a positive pair of examples (i, j) is defined as

我们随机采样一个minibatch，包含N个样本，在这个minibatch中的一对扩增的样本上，定义对比预测任务，得到2N个数据点。我们不显式的采样负样本。而是，给定一个正对，我们认为这个mini-batch中其他2(N-1)个扩增的样本为负样本。令$sim(u,v) = u^⊤v/∥u∥∥v∥$表示两个向量u和v的cosine相似度。那么一个正对样本(i, j)之间的损失函数定义为：

$$l_{i,j} = -log \frac {exp(sim(z_i, z_j)/τ)} {\sum_{k=1}^{2N} 1_{[k\neq i]} exp(sim(z_i,z_k)/τ)}$$(1)

where $1_{[k\neq i]} ∈$ {0, 1} is an indicator function evaluating to 1 iff $k\neq i$ and τ denotes a temperature parameter. The final loss is computed across all positive pairs, both (i,j) and (j,i), in a mini-batch. This loss has been used in previous work (Sohn, 2016; Wu et al., 2018; Oord et al., 2018); for convenience, we term it NT-Xent (the normalized temperature-scaled cross entropy loss).

其中$1_{[k\neq i]} ∈$ {0, 1}是一个指示函数，在且只在$k\neq i$时取值为1，τ表示一个温度参数。最终的损失在所有正对上进行计算，包括在一个mini-batch中的(i,j)和(j,i)。这个损失函数在之前的工作中曾经用过；为方便，我们称之为NT-Xent（归一化温度尺度的交叉熵损失）。

Algorithm 1 summarizes the proposed method. 算法1总结了提出的方法。

### 2.2. Training with Large Batch Size

We do not train the model with a memory bank (Wu et al., 2018). Instead, we vary the training batch size N from 256 to 8192. A batch size of 8192 gives us 16382 negative examples per positive pair from both augmentation views. Training with large batch size may be unstable when using standard SGD/Momentum with linear learning rate scaling (Goyal et al., 2017). To stabilize the training, we use the LARS optimizer (You et al., 2017) for all batch sizes. We train our model with Cloud TPUs, using 32 to 128 cores depending on the batch size.

我们并没有使用存储组来训练模型。我们使训练的batch大小进行变化，从256到8192。8192的batch大小，对每个正对的两个扩增视图，会给出16382个负样本。用大型batch大小进行训练，当使用标准SGD/动量和线性学习速率缩放时，可能是不稳定的。为稳定训练，我们对所有的batch大小都使用LARS优化器。我们使用云TPUs训练我们的模型，依据batch大小使用32到128个核。

**Global BN**. Standard ResNets use batch normalization (Ioffe & Szegedy, 2015). In distributed training with data parallelism, the BN mean and variance are typically aggregated locally per device. In our contrastive learning, as positive pairs are computed in the same device, the model can exploit the local information leakage to improve prediction accuracy without improving representations. We address this issue by aggregating BN mean and variance over all devices during the training. Other approaches include shuffling data examples (He et al., 2019a), or replacing BN with layer norm (Hénaff et al., 2019).

**全局BN**。标准的ResNets使用了批归一化。在数据并行的分布式训练中，BN的均值和方差一般都是在每个设备上局部聚集的。在我们的对比学习中，由于正对是在同一设备上计算得到的，模型可以利用局部信息的泄漏来改进预测准确率，而不需要改进表示。我们通过在所有设备上在训练时聚集BN的均值和方差，来解决这个问题。其他的方法包括混洗数据样本，或用层的范数替换BN。

### 2.3. Evaluation Protocol

Here we lay out the protocol for our empirical studies, which aim to understand different design choices in our framework.

这里我们为我们的经验研究提出协议，其目的是理解我们框架中不同的设计选择。

**Dataset and Metrics**. Most of our study for unsupervised pretraining (learning encoder network f without labels) is done using the ImageNet ILSVRC-2012 dataset (Russakovsky et al., 2015). Some additional pretraining experiments on CIFAR-10 (Krizhevsky & Hinton, 2009) can be found in Appendix B.7. We also test the pretrained results on a wide range of datasets for transfer learning. To evaluate the learned representations, we follow the widely used linear evaluation protocol (Zhang et al., 2016; Oord et al., 2018; Bachman et al., 2019; Kolesnikov et al., 2019), where a linear classifier is trained on top of the frozen base network, and test accuracy is used as a proxy for representation quality. Beyond linear evaluation, we also compare against state-of-the-art on semi-supervised and transfer learning.

**数据集和度量**。多数我们的无监督预训练研究（在没有标签的情况下学习编码器网络f），是使用ImageNet ILSVRC-2012数据集完成的。一些额外的在CIFAR-10上的预训练试验可以在附录B.7中找到。我们还在很多数据集上测试了预训练结果，进行迁移学习。为评估学到的表示，我们按照广泛使用的线性评估协议，其中在冻结的基础网络之上训练了一个线性分类器，测试准确率用作表示质量的代理。在线性评估之外，我们还在半监督和迁移学习上与目前最好的结果进行了比较。

**Default setting**. Unless otherwise specified, for data augmentation we use random crop and resize (with random flip), color distortions, and Gaussian blur (for details, see Appendix A). We use ResNet-50 as the base encoder network, and a 2-layer MLP projection head to project the representation to a 128-dimensional latent space. As the loss, we use NT-Xent, optimized using LARS with linear learning rate scaling (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10e-6. We train at batch size 4096 for 100 epochs. Further more, we use linear warmup for the first 10 epochs, and decay the learning rate with the cosine decay schedule without restarts (Loshchilov & Hutter, 2016).

**默认设置**。除非另外指定，对数据扩增，我们使用随机剪切和改变大小（带有随机翻转），色彩变换，和高斯模糊（对于细节，见附录A）。我们使用ResNet-50作为基础编码器网络，并使用一个2层的MLP投影头来将表示投影到128维的隐空间。至于损失，我们使用的是NT-Xent，使用LARS优化器，线性学习速率缩放（即，学习速率=0.3x批规模/256），权重衰减为10e-6。我们训练的批规模为4096，进行100轮训练。而且，我们对前10轮训练使用线性预热，学习速率衰减使用cosine方案，没有重启。

## 3. Data Augmentation for Contrastive Representation Learning

**Data augmentation defines predictive tasks**. While data augmentation has been widely used in both supervised and unsupervised representation learning (Krizhevsky et al., 2012; Hénaff et al., 2019; Bachman et al., 2019), it has not been considered as a systematic way to define the contrastive prediction task. Many existing approaches define contrastive prediction tasks by changing the architecture. For example, Hjelm et al. (2018); Bachman et al. (2019) achieve global-to-local view prediction via constraining the receptive field in the network architecture, whereas Oord et al. (2018); Hénaff et al. (2019) achieve neighboring view prediction via a fixed image splitting procedure and a context aggregation network. We show that this complexity can be avoided by performing simple random cropping (with resizing) of target images, which creates a family of predictive tasks subsuming the above mentioned two, as shown in Figure 3. This simple design choice conveniently decouples the predictive task from other components such as the neural network architecture. Broader contrastive prediction tasks can be defined by extending the family of augmentations and composing them stochastically.

**数据扩增定义了预测任务**。数据扩增在有监督和无监督表示学习中得到了广泛的应用，但还并没有被认为是一种系统的定义对比预测任务的方法。很多已有的方法通过改变架构来定义对比预测任务。比如，Hjelm et al. (2018); Bachman et al. (2019)通过限制网络架构中的感受野，来得到全局到局部的视图，而Oord et al. (2018); Hénaff et al. (2019)通过固定的图像分裂过程和上下文聚集网络来得到邻域视图预测。我们展示了，这种复杂性可以通过目标图像的简单随机剪切（并改变大小）来避免，这创建了一类预测任务，将上面提到的两个都归入一起，如图3所示。这种简单的设计选择很方便的将预测任务与其他组件分离开来，如网络架构。更宽泛的对比预测任务，可以通过拓展扩增的种类，并将其随机组合得到。

### 3.1. Composition of data augmentation operations is crucial for learning good representations

To systematically study the impact of data augmentation, we consider several common augmentations here. One type of augmentation involves spatial/geometric transformation of data, such as cropping and resizing (with horizontal flipping), rotation (Gidaris et al., 2018) and cutout (De Vries & Taylor, 2017). The other type of augmentation involves appearance transformation, such as color distortion (including color dropping, brightness, contrast, saturation, hue) (Howard, 2013; Szegedy et al., 2015), Gaussian blur, and Sobel filtering. Figure 4 visualizes the augmentations that we study in this work.

为系统的研究数据扩增的影响，我们考虑几种常见的扩增。一类扩增为数据的空间/几何变换，比如剪切和变换大小（带有水平翻转），旋转和cutout。另一种扩增类型为外观变换，如色彩变换（包括色彩丢失，亮度，对比度，饱和度，hue），高斯模糊和Sobel滤波。图4对本文中我们研究的扩增进了可视化。

To understand the effects of individual data augmentations and the importance of augmentation composition, we investigate the performance of our framework when applying augmentations individually or in pairs. Since ImageNet images are of different sizes, we always apply crop and resize images (Krizhevsky et al., 2012; Szegedy et al., 2015), which makes it difficult to study other augmentations in the absence of cropping. To eliminate this confound, we consider an asymmetric data transformation setting for this ablation. Specifically, we always first randomly crop images and resize them to the same resolution, and we then apply the targeted transformation(s) only to one branch of the framework in Figure 2, while leaving the other branch as the identity (i.e. $t(x_i) = x_i$). Note that this asymmetric data augmentation hurts the performance. Nonetheless, this setup should not substantively change the impact of individual data augmentations or their compositions.

为理解单个数据扩增的效果，和扩增组成的重要性，我们研究了当应用单个的扩增，或成对的应用时，我们框架的性能。既然ImageNet图像的大小都不一样，我们一直应用的是剪切和改变图像大小，这使得在缺少剪切的情况下研究其他扩增很困难。为消除这种困惑，我们考虑了一种非对称的数据变换设置。具体的，我们永远首先随机剪切图像并将其改变到同样的分辨率，然后我们只对图2中的框架的一枝进行目标变换，而使另一枝为恒等变换（即，$t(x_i) = x_i$）。注意，这种非对称数据扩增会使性能有损失。即使这样，这种设置不应当改变单个数据扩增或其组合的影响。

Figure 5 shows linear evaluation results under individual and composition of transformations. We observe that no single transformation suffices to learn good representations, even though the model can almost perfectly identify the positive pairs in the contrastive task. When composing augmentations, the contrastive prediction task becomes harder, but the quality of representation improves dramatically.

图5给出了在单个和组合变换时的线性评估结果。我们观察到，单个变换是不足以学到很好的表示的，即使模型可以几乎完美的识别出对比任务中的正对。当组合扩增时，对比预测任务变得更难，但表示质量得到了极大改进。

One composition of augmentations stands out: random cropping and random color distortion. We conjecture that one serious issue when using only random cropping as data augmentation is that most patches from an image share a similar color distribution. Figure 6 shows that color histograms alone suffice to distinguish images. Neural nets may exploit this shortcut to solve the predictive task. Therefore, it is critical to compose cropping with color distortion in order to learn generalizable features.

一种扩增的组合效果非常好：随机剪切和随机色彩变换。我们推测，当使用随机剪切作为数据扩增时，一个严重的问题是，一幅图像中的多数图像块是服从类似的色彩分布的。图6所示的是，色彩直方图就足以分辨图像。神经网络可能会利用这种捷径来求解预测任务。因此，将剪切与色彩变换组合到一起，以学习到可泛化的特征，这是很关键的。

### 3.2. Contrastive learning needs stronger data augmentation than supervised learning

To further demonstrate the importance of the color augmentation, we adjust the strength of color augmentation as shown in Table 1. Stronger color augmentation substantially improves the linear evaluation of the learned unsupervised models. In this context, AutoAugment (Cubuk et al., 2019), a sophisticated augmentation policy found using supervised learning, does not work better than simple cropping + (stronger) color distortion. When training supervised models with the same set of augmentations, we observe that stronger color augmentation does not improve or even hurts their performance. Thus, our experiments show that unsupervised contrastive learning benefits from stronger (color) data augmentation than supervised learning. Although previous work has reported that data augmentation is useful for self-supervised learning (Doersch et al., 2015; Bachman et al., 2019; Hénaff et al., 2019), we show that data augmentation that does not yield accuracy benefits for supervised learning can still help considerably with contrastive learning.

为进一步证明色彩扩增的重要性，我们调整了色彩扩增的强度，如表1所示。更强的色彩扩增极大了改善了学习到的无监督模型的线性评估性能。在这个上下文中，AutoAugment这种在有监督学习中使用的复杂的扩增策略，并没有比简单的剪切+（更强的）色彩变形更好。我们使用相同的扩增训练有监督模型，我们观察到，更强的色彩扩增并没有改进性能，甚至会对性能有损伤。因此，我们的试验表明，无监督对比学习会从更强的色彩数据扩增中获益，比有监督学习获益更多。虽然之前的工作表明，数据扩增对自监督学习是有用的，但我们证明了，数据扩增不会对有监督学习产生准确率提升，但对对比学习可以有很大的帮助。

## 4. Architectures for Encoder and Head

### 4.1. Unsupervised contrastive learning benefits (more) from bigger models

Figure 7 shows, perhaps unsurprisingly, that increasing depth and width both improve performance. While similar findings hold for supervised learning (He et al., 2016), we find the gap between supervised models and linear classifiers trained on unsupervised models shrinks as the model size increases, suggesting that unsupervised learning benefits more from bigger models than its supervised counterpart.

图7证明了，增加深度和宽度都会改进性能，这并不令人惊讶。虽然对于有监督学习也有类似的发现，但我们发现，当模型规模增加时，有监督模型与在无监督模型上训练的线性分类器的差距变小了，说明无监督学习从更大的模型中受益更多。

### 4.2. A nonlinear projection head improves the representation quality of the layer before it

We then study the importance of including a projection head, i.e. g(h). Figure 8 shows linear evaluation results using three different architecture for the head: (1) identity mapping; (2) linear projection, as used by several previous approaches (Wu et al., 2018); and (3) the default nonlinear projection with one additional hidden layer (and ReLU activation), similar to Bachman et al. (2019). We observe that a nonlinear projection is better than a linear projection (+3%), and much better than no projection (>10%). When a projection head is used, similar results are observed regardless of output dimension. Furthermore, even when nonlinear projection is used, the layer before the projection head, h, is still much better (>10%) than the layer after, z = g(h), which shows that the hidden layer before the projection head is a better representation than the layer after.

我们然后研究了包含一个投影头的重要性，即g(h)，图8展示了使用三种不同的架构作为头的线性评估结果：(1)恒等映射；(2)线性投影，之前的几种方法使用都是这个；(3)默认的非线性投影，带有一个额外的隐含层（和ReLU激活）。我们观察到，非线性投影比线性投影效果要好(+3%)，比没有投影要好很多(>10%)。当使用了一个投影头时，不论输出的维度是多少，都可以看到类似的结果。而且，即使是使用了非线性投影，在投影头之前的层h，仍然比这个层之后的z=g(h)要好很多(>10%)，这表明在投影头之前的隐含层比投影层之后的是一个更好的表示。

We conjecture that the importance of using the representation before the nonlinear projection is due to loss of information induced by the contrastive loss. In particular, z = g(h) is trained to be invariant to data transformation. Thus, g can remove information that may be useful for the downstream task, such as the color or orientation of objects. By leveraging the nonlinear transformation g(·), more information can be formed and maintained in h. To verify this hypothesis, we conduct experiments that use either h or g(h) to learn to predict the transformation applied during the pretraining. Here we set $g(h) = W^{(2)} σ(W^{(1)}h)$, with the same input and output dimensionality (i.e. 2048). Table 3 shows h contains much more information about the transformation applied, while g(h) loses information.

我们推测，使用非线性投影之前的表示的重要性，是由于对比损失导致的信息损失。特别是，z=g(h)的训练，是要对数据变换是不变的。因此，g会去除掉一些对于下游任务有用的信息，比如色彩或目标方向。通过利用非线性变换g(·)，可以形成更多的信息，并保存在h中。为验证这个假设，我们进行了试验，使用h或g(h)在预训练的时候来学习预测应用的变换。这里我们设$g(h) = W^{(2)} σ(W^{(1)}h)$，输入和输出的维度是相同的（即2048）。表3表明，h包含了更多的关于应用的变换的信息，而g(h)损失了信息。

## 5. Loss Functions and Batch Size

### 5.1. Normalized cross entropy loss with adjustable temperature works better than alternatives

We compare the NT-Xent loss against other commonly used contrastive loss functions, such as logistic loss (Mikolov et al., 2013), and margin loss (Schroff et al., 2015). Table 2 shows the objective function as well as the gradient to the input of the loss function. Looking at the gradient, we observe 1) l2 normalization along with temperature effectively weights different examples, and an appropriate temperature can help the model learn from hard negatives; and 2) unlike cross-entropy, other objective functions do not weigh the negatives by their relative hardness. As a result, one must apply semi-hard negative mining (Schroff et al., 2015) for these loss functions: instead of computing the gradient over all loss terms, one can compute the gradient using semi-hard negative terms (i.e., those that are within the loss margin and closest in distance, but farther than positive examples).

我们比较了NT-Xent损失与其他常用的对比损失函数，比如logistic损失，和margin损失。表2所示的是，目标函数，以及损失函数对输入的梯度。查看这些梯度，我们发现，(1)l2归一化和温度有效的加权了不同的样本，一个合适的温度可以帮助模型从难分负样本中学习，(2)与交叉熵不同，其他目标函数没有通过其相对难度对负样本进行加权。结果是，必须对这些损失函数，应用半难分负样本挖掘：不是在所有损失项上计算梯度，而是使用半难分负项计算梯度（即，那些在损失边界以内，在距离上比较接近的，但比正样本更远的）。

To make the comparisons fair, we use the same l2 normalization for all loss functions, and we tune the hyperparameters, and report their best results. Table 4 shows that, while (semi-hard) negative mining helps, the best result is still much worse than our default NT-Xent loss.

为使比较更公平，我们对所有损失函数使用相同的l2归一化，我们调整超参数，以给出其最好的结果。表4表明，（半难分）负样本挖掘有帮助，但最好的结果仍然比我们默认的NT-Xent损失要差。

We next test the importance of the l2 normalization and temperature τ in our default NT-Xent loss. Table 5 shows that without normalization and proper temperature scaling, performance is significantly worse. Without l2 normalization, the contrastive task accuracy is higher, but the resulting representation is worse under linear evaluation.

我们下一步测试l2归一化和温度τ在我们默认的NT-Xent损失中的重要性。表5表明，没有归一化和合适的温度缩放，性能会显著恶化。没有l2归一化，对比任务准确率更高，但得到的表示在线性评估下要更差。

### 5.2. Contrastive learning benefits (more) from larger batch sizes and longer training

Figure 9 shows the impact of batch size when models are trained for different numbers of epochs. We find that, when the number of training epochs is small (e.g. 100 epochs), larger batch sizes have a significant advantage over the smaller ones. With more training steps/epochs, the gaps between different batch sizes decrease or disappear, provided the batches are randomly resampled. In contrast to supervised learning (Goyal et al., 2017), in contrastive learning, larger batch sizes provide more negative examples, facilitating convergence (i.e. taking fewer epochs and steps for a given accuracy). Training longer also provides more negative examples, improving the results.

图9给出了批规模的影响，模型训练的轮数也不一样。我们发现，当训练轮数很小时（如，100轮），更大的批规模比更小的有很显著的优势。在更多的训练步数/轮数下，不同批规模之间的差异逐渐下降或消失，如果这些批都随机重采样的。与有监督学习相比，在对比学习中，更大的批规模会给出更多的负样本，会促进收敛（即，对于给定的准确率，耗费更少的轮数和步数就会收敛）。训练的时间更长，也会给出更多的负样本，改进结果。

## 6. Comparison with State-of-the-art

In this subsection, similar to Kolesnikov et al. (2019); He et al. (2019a), we use ResNet-50 in 3 different hidden layer widths (width multipliers of 1×, 2×, and 4×). For better convergence, our models here are trained for 1000 epochs.

在本节中，我们使用三种不同的隐含层宽度的ResNet-50（宽度乘子1x, 2x和4x)。为更好的收敛，我们的模型训练1000轮。

**Linear evaluation**. Table 6 compares our results with previous approaches (Zhuang et al., 2019; He et al., 2019a; Misra & van der Maaten, 2019; Hénaff et al., 2019; Kolesnikov et al., 2019; Donahue & Simonyan, 2019; Bachman et al., 2019; Tian et al., 2019) in the linear evaluation setting. Table 1 shows more numerical comparisons among different methods. We are able to use standard networks to obtain substantially better results compared to previous methods that require specifically designed architectures. The best result obtained with our ResNet-50 (4×) can match the supervised pretrained ResNet-50.

**线性评估**。表6将我们的结果与之前的方法，在线性评估的设置下进行了比较。表1给出了不同方法的更多的数值比较。我们可以使用标准网络来得到更好的结果。用ResNet-50(4x)得到的最好结果，可以与监督预训练ResNet-50相媲美。

**Semi-supervised learning**. We follow Zhai et al. (2019) and sample 1% or 10% of the labeled ILSVRC-12 training datasets in a class-balanced way (i.e. around 12.8 and 128 images per class respectively). We simply fine-tune the whole base network on the labeled data without regularization (see Appendix B.5). Table 7 shows the comparisons of our results against recent methods (Zhai et al., 2019; Xie et al., 2019; Sohn et al., 2020; Wu et al., 2018; Donahue & Simonyan, 2019; Misra & van der Maaten, 2019; Hénaff et al., 2019). Again, our approach significantly improves over state-of-the-art with both 1% and 10% of the labels.

**半监督学习**。我们按照Zhai et al. (2019)，采样1%或10%的标记的ILSCRC-12训练数据集，并在不同类别之间平衡（即，每类12.8或128幅图像）。我们简单的将整个基础网络在标记的数据上进行精调，没有正则化。表7给出了我们的结果与最近的方法的比较。我们的方法再一次比之前最好的结果有很大改进。

**Transfer learning**. We evaluate transfer learning performance across 12 natural image datasets in both linear evaluation (fixed feature extractor) and fine-tuning settings. Following Kornblith et al. (2019), we perform hyperparameter tuning for each model-dataset combination and select the best hyperparameters on a validation set. Table 8 shows results with the ResNet-50 (4×) model. When fine-tuned, our self-supervised model significantly outperforms the supervised baseline on 5 datasets, whereas the supervised baseline is superior on only 2 (i.e. Pets and Flowers). On the remaining 5 datasets, the models are statistically tied. Full experimental details as well as results with the standard ResNet-50 architecture are provided in Appendix B.6.

**迁移学习**。我们在12个自然图像数据集中评估迁移学习的性能，包括线性评估（固定的特征提取器）和精调的设置。按照Kornblith et al. (2019)，我们对每个模型-数据集的组合进行超参数调节，并在一个验证集上选择最好的超参数。表8给出了ResNet-50(4x)模型的结果。当进行了精调，我们的自监督模型在5个数据集上显著超过了有监督的基准，而有监督的基准只在2个数据集上结果超前（即，宠物和花）。在剩下的5个数据集中，模型在统计上是类似的。完整的试验细节，以及使用标准ResNet-50架构的结果，在附录B.6中。

We note that the superiority of our framework relative to previous work is not explained by any single design choice, but by their composition. We provide a comprehensive comparison of our design choices with those of previous work in Appendix C.

我们指出，我们框架比之前的工作的优越性，并不是因为任何单个的设计选择，而是其组合。我们将我们的设计选择，与之前的工作，进行了综合的比较，如附录C所示。

## 7. Related Work

The idea of making representations of an image agree with each other under small transformations dates back to Becker & Hinton (1992). We extend this idea by leveraging recent advances in data augmentation, network architecture and contrastive losses. A similar consistency idea has been explored in other contexts such as semi-supervised learning (Xie et al., 2019; Berthelot et al., 2019).

使得一幅图像在很小的变换下，其表示相互符合，这种思想在1992年就有了。我们拓展了这种思想，利用最近在数据扩增，网络架构和对比损失的进展。一种类似的一致性思想在其他上下文中也得到了研究，比如半监督学习。

**Handcrafted pretext tasks**. The recent renaissance of self-supervised learning began with artificially designed pretext tasks, such as relative patch prediction (Doersch et al., 2015), solving jigsaw puzzles (Noroozi & Favaro, 2016), colorization (Zhang et al., 2016) and rotation prediction (Gidaris et al., 2018). Although good results can be obtained with bigger networks and longer training (Kolesnikov et al., 2019), these pretext tasks rely on somewhat ad-hoc heuristics, which limits the generality of learned representations.

**手工设计的pretext任务**。最近复兴的自监督学习，开始于人工设计的pretext任务，比相对图像块预测，求解拼图谜题，上色，旋转预测等。虽然使用更大的网络和更长时间的训练，可以得到很好的结果，但这些pretext任务依赖于某种ad-hoc启发式，这限制了学习到的表示的泛化性。

**Contrastive visual representation learning**. Dating back to Hadsell et al. (2006), these approaches learn representations by contrasting positive pairs against negative pairs. Along these lines, Dosovitskiy et al. (2014) proposes to treat each instance as a class represented by a feature vector (in a parametric form). Wu et al. (2018) proposes to use a memory bank to store the instance class representation vector, an approach adopted and extended in several recent papers (Zhuang et al., 2019; Tian et al., 2019; He et al., 2019a; Misra & van der Maaten, 2019). Other work explores the use of in-batch samples for negative sampling instead of a memory bank (Ye et al., 2019; Ji et al., 2019).

**对比视觉表示学习**。追溯到Hadsell et al. (2006)，这些方法通过对比正对和负对来学习表示。沿着这条线，Dosovitskiy et al. (2014)提出将每个实例都按照一个类别来对待，表示为一个特征向量（以一种参数化的形式）。Wu et al. (2018)提出使用存储组来存储实例类别表示向量，随后被几篇文章采用并拓展。其他工作探索了使用in-batch样本进行负样本采样，而不采用存储组。

Recent literature has attempted to relate the success of their methods to maximization of mutual information between latent representations (Oord et al., 2018; Hénaff et al., 2019; Hjelm et al., 2018; Bachman et al., 2019). However, it is not clear if the success of contrastive approaches is determined by the mutual information, or by the specific form of the contrastive loss (Tschannen et al., 2019). Further comparison of our method to related methods are in Appendix C.

最近的文献试图将其方法的成功，与隐表示的互信息最大化关联起来。但是，对比方法的成功，是否是由互信息决定的，或是由于对比损失的具体形式，还不是很明了。我们的方法与相关方法的进一步比较如附录C所示。

## 8. Conclusion

In this work, we present a simple framework and its instantiation for contrastive visual representation learning. We carefully study its components, and show the effects of different design choices. By combining our findings, we improve considerably over previous methods for self-supervised, semi-supervised, and transfer learning.

本文中，我们提出了一种简单的框架及其实现，进行对比视觉表示学习。我们仔细的研究了其组件，展示了不同设计选择的效果。通过将我们的发现结合，我们比之前的自监督、半监督和迁移学习改进了很多。

Our results show that the complexity of some previous methods for self-supervised learning is not necessary to achieve good performance. Our approach differs from standard supervised learning on ImageNet only in the choice of data augmentation, the use of a nonlinear head at the end of the network, and the loss function. The strength of this simple framework suggests that, despite a recent surge in interest, self-supervised learning remains undervalued.

我们的结果表明，一些之前的自监督学习方法的复杂性不是必须的，就可以得到很好的效果。我们的方法与标准的在ImageNet上的监督学习方法不同，包括数据扩增的方法，在网络最后使用非线性头，损失函数。这种简单框架的力度说明，尽管最近有很多这方面的研究兴趣，自监督学习仍然是低估的。