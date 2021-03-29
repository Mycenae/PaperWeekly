# Revisiting ResNets: Improved Training and Scaling Strategies

Irwan Bello et. al. Google Brain

## 0. Abstract

Novel computer vision architectures monopolize the spotlight, but the impact of the model architecture is often conflated with simultaneous changes to training methodology and scaling strategies. Our work revisits the canonical ResNet (He et al., 2015) and studies these three aspects in an effort to disentangle them. Perhaps surprisingly, we find that training and scaling strategies may matter more than architectural changes, and further, that the resulting ResNets match recent state-of-the-art models. We show that the best performing scaling strategy depends on the training regime and offer two new scaling strategies: (1) scale model depth in regimes where overfitting can occur (width scaling is preferable otherwise); (2) increase image resolution more slowly than previously recommended (Tan & Le, 2019). Using improved training and scaling strategies, we design a family of ResNet architectures, ResNet-RS, which are 1.7x - 2.7x faster than EfficientNets on TPUs, while achieving similar accuracies on ImageNet. In a large-scale semi-supervised learning setup, ResNet-RS achieves 86.2% top-1 ImageNet accuracy, while being 4.7x faster than EfficientNet-NoisyStudent. The training techniques improve transfer performance on a suite of downstream tasks (rivaling state-of-the-art self-supervised algorithms) and extend to video classification on Kinetics-400. We recommend practitioners use these simple revised ResNets as baselines for future research.

新的计算机视觉架构垄断了注意力，但模型架构的影响，通常与训练方法，和缩放策略的变化相混合。我们的工作重新回顾了经典ResNet，研究了这三个方面以分析清楚。可能令人惊讶的是，我们发现训练和缩放策略要比架构变化的影响更大，而且，得到的ResNets与目前最好的结果相似。我们证明了，最好表现的缩放策略依赖于训练方法，给出了两个新的缩放策略：(1)在有过拟合的情况时，缩放模型深度（宽度缩放在其他情况时更好）；(2)比之前推荐的分辨率增加速度更慢。使用改进的训练和缩放策略，我们设计了一族ResNet架构，ResNet-RS，在TPUs上比EfficientNets快1.7x-2.7x，同时在ImageNet上获得了类似的准确率。在大规模半监督学习的设置中，ResNet-RS获得了86.2%的top-1 ImageNet准确率，比EfficientNet-NoisyStudent要快4.7x。训练技术在下游任务中改进了迁移学习的性能（与目前最好的自监督算法类似），拓展到了在Kinetics-400上的视频分类中。我们推荐在未来的研究中使用这些简单修改过的ResNets作为基准。

## 1. Introduction

The performance of a vision model is a product of the architecture, training methods and scaling strategy. However, research often emphasizes architectural changes. Novel architectures underlie many advances, but are often simultaneously introduced with other critical – and less publicized – changes in the details of the training methodology and hyperparameters. Additionally, new architectures enhanced by modern training methods are sometimes compared to older architectures with dated training methods (e.g. ResNet-50 with ImageNet Top-1 accuracy of 76.5% (He et al., 2015)). Our work addresses these issues and empirically studies the impact of training methods and scaling strategies on the popular ResNet architecture (He et al., 2015).

视觉模型的性能，是架构，训练方法和缩放策略的结果。但是，研究通常强调架构的变化。新的架构是很多进步的基础，但通常同时与其他关键但是宣传很少的变化同时引入，包括训练方法和超参数的细节。另外，现代训练方法增强的新架构，有时候与用老的训练方法的老架构进行比较（如，ResNet-50，ImageNet top-1准确率76.5%）。我们的工作处理这些问题，从经验上研究训练方法和缩放策略在流行的ResNet架构上的影响。

We survey the modern training and regularization techniques widely in use today and apply them to ResNets (Figure 1). In the process, we encounter interactions between training methods and show a benefit of reducing weight decay values when used in tandem with other regularization techniques. An additive study of training methods in Table 1 reveals the significant impact of these decisions: a canonical ResNet with 79.0% top-1 ImageNet accuracy is improved to 82.2% (+3.2%) through improved training methods alone. This is increased further to 83.4% by two small and commonly used architectural improvements: ResNet-D (He et al., 2018) and Squeeze-and-Excitation (Hu et al., 2018). Figure 1 traces this refinement over the starting ResNet in a speed-accuracy Pareto curve.

我们研究了现在广泛使用的现代训练和正则化技术，将其应用与ResNets（图1）。在这个过程中，我们遇到了训练方法之间的交互，证明了降低权重衰减值与一些其他正则化技术一起使用的好处。表1中的训练方法加性研究表明，这些决策的显著影响：经典ResNet的top-1 ImageNet准确率为79.0%，用改进的训练方法本身就改进到了82.2% (+3.2%)。又通过两个小但经常使用的架构改进，性能增加到了83.4%：ResNet-D和Squeeze-and-Excitation。图1追踪了这些方法相对于起使ResNet的改进，给出了速度-准确率的Pareto曲线。

We offer new perspectives and practical advice on scaling vision architectures. While prior works extrapolate scaling rules from small models (Tan & Le, 2019) or from training for a small number of epochs (Radosavovic et al., 2020), we design scaling strategies by exhaustively training models across a variety of scales for the full training duration (e.g. 350 epochs instead of 10 epochs). In doing so, we uncover strong dependencies between the best performing scaling strategy and the training regime (e.g. number of epochs, model size, dataset size). These dependencies are missed in any of these smaller regimes, leading to sub-optimal scaling decisions. Our analysis leads to new scaling strategies summarized as (1) scale the model depth when overfitting can occur (scaling the width is preferable otherwise) and (2) scale the image resolution more slowly than prior works (Tan & Le, 2019).

我们在缩放视觉架构上给出了新的观点和实践建议。之前的工作从小模型上外插得到缩放规则，或从很少数量的epochs训练上外插，我们通过在很多尺度上长时间穷举式训练模型来设计缩放策略（350 epochs，而不是10 epochs）。这样做，我们揭示了最佳性能的缩放策略和训练方法的很强关联性（如，epochs数量，模型大小，数据集大小）。这些依赖关系在更小的研究范畴中是缺失的，带来了次优的缩放策略。我们的分析得到了新的缩放策略，总结为：(1)当过拟合发生的时候，缩放模型的深度（缩放宽度在其他情况下效果更好），(2)比之前的工作更慢的缩放图像分辨率。

Using the improved training and scaling strategies, we design re-scaled ResNets, ResNet-RS, which are trained across a wide range of model sizes, as shown in Figure 1. ResNet-RS models use less memory during training and are 1.7x - 2.7x faster on TPUs (2.1x - 3.3x faster on GPUs) than the popular EfficientNets on the speed-accuracy Pareto curve. In a large-scale semi-supervised learning setup, ResNet-RS obtains a 4.7x training speedup on TPUs (5.5x on GPUs) over EfficientNet-B5 when co-trained on ImageNet and an additional 130M pseudo-labeled images.

使用改进的训练和缩放策略，我们设计了重新缩放的ResNets，即ResNet-RS，在大范围的模型大小上进行了训练，如图1所示。在速度-准确率Pareto曲线上，ResNet-RS模型比流行的EfficientNets，在训练时使用了更少的内存，在TPUs上速度快了1.7x-2.7x（GPUs上快了2.1x-3.3x）。在大规模半监督的训练设置中，在ImageNet与另外130M伪标注的图像的联合训练中，ResNet-RS比EfficientNet-B5，在TPUs上训练加速了4.7x（在GPUs加速了5.5x）。

Finally, we conclude with a suite of experiments testing the generality of the improved training and scaling strategies. We first design a faster version of EfficientNet using our scaling strategy, EfficientNet-RS, which improves over the original on the speed-accuracy Pareto curve. Next, we show that the improved training strategies yield representations that rival or outperform those from self-supervised algorithms (SimCLR (Chen et al., 2020a) and SimCLRv2 (Chen et al., 2020b)) on a suite of downstream tasks. The improved training strategies extend to video classification as well. Applying the training strategies to 3D-ResNets on the Kinetics-400 dataset yields an improvement from 73.4% to 77.4% (+4.0%).

最后，我们用一系列试验测试了改进的训练和缩放策略的泛化性，得出了结论。我们首先用我们的缩放策略设计了一个更快版本的EfficientNet，EfficientNet-RS，在原始版本上改进了速度-准确率Pareto曲线。下一步，我们证明了，改进的训练策略得到的表示，与自监督算法得到的表示相比，在一系列下游任务中都类似或有所超越。改进的训练策略还拓展到了视频分类应用。将训练策略应用到3D-ResNets，在Kinetics-400数据集上将性能从73.4%改进到了77.4% (+4.0%)。

Through combining minor architectural changes (used since 2018) and improved training and scaling strategies, we discover the ResNet architecture sets a state-of-the-art baseline for vision research. This finding highlights the importance of teasing apart each of these factors in order to understand what architectures perform better than others.

将一些次要的架构变化，与改进的训练和缩放策略，结合到一起，我们发现ResNet架构确立了新的目前最好基准。这个发现强调了梳理这些因素的重要性，以理解什么架构要比其他的要好。

We summarize our contributions: 我们总结了我们的贡献：

- An empirical study of regularization techniques and their interplay, which leads to a regularization strategy that achieves strong performance (+3% top-1 accuracy) without having to change the model architecture. 正则化技术和其相互作用的相互研究，得到了可以获得最强性能的正则化策略（+3% top-1准确率），而不用改变模型架构。

- A simple scaling strategy: (1) scale depth when overfitting can occur (scaling width can be preferable otherwise) and (2) scale the image resolution more slowly than prior works (Tan & Le, 2019). This scaling strategy improves the speed-accuracy Pareto curve of both ResNets and EfficientNets. 简单的缩放策略：(1)当过拟合发生时，缩放深度（缩放宽度在其他情况下更好用），(2)比之前的工作更慢的缩放图像分辨率。这种缩放策略改进了ResNets和EfficientNets的速度-准确率的Pareto曲线。

- ResNet-RS: a Pareto curve of ResNet architectures that are 1.7x - 2.7x faster than EfficientNets on TPUs (2.1x - 3.3x on GPUs) by applying the training and scaling strategies. 使用这些训练和缩放策略，得到了ResNet-RS，这种ResNet架构的Pareto曲线，比EfficientNets在TPUs上快了1.7x-2.7x（在GPUs上快了2.1x-3.3x）。

- Semi-supervised training of ResNet-RS with an additional 130M pseudo-labeled images achieves 86.2% top-1 ImageNet accuracy, while being 4.7x faster on TPUs (5.5x on GPUs) than the corresponding EfficientNet-NoisyStudent (Xie et al., 2020). ResNet-RS用额外的130M伪标签图像的半监督训练，得到了86.2%的top-1 ImageNet准确率，比对应的EfficientNet-NoisyStudent在TPUs上快了4.7x（在GPUs上快了5.5x）。

- ResNet checkpoints that, when fine-tuned on a diverse set of computer vision tasks, rival or outperform state-of-the-art self-supervised representations from SimCLR (Chen et al., 2020a) and SimCLRv2 (Chen et al., 2020b). 当在很多计算机视觉任务中进行精调时，ResNet的checkpoints与目前最好的自监督学习类似或有所超越。

- 3D ResNet-RS by extending our training methods and architectural changes to video classification. The resulted model improves the top-1 Kinetics-400 accuracy by 4.8% over the baseline. 将我们的训练方法和架构改变拓展到视频分类，得到了3D ResNet-RS。得到的模型与基准相比，改进了Kinetics-400的准确率4.8%。

## 2. Characterizing Improvements on ImageNet

Since the breakthrough of AlexNet (Krizhevsky et al., 2012) on ImageNet (Russakovsky et al., 2015), a wide variety of improvements have been proposed to further advance image recognition performance. These improvements broadly arise along four orthogonal axes: architecture, training/regularization methodology, scaling strategy and using additional training data.

自从AlexNet在ImageNet上的爆发，已经提出了很多改进，以进一步推动图像识别的性能。这些改进大致沿着四个不同的方向：架构，训练/正则化方法，缩放策略和使用额外的训练数据。

**Architecture**. The works that perhaps receive the most attention are novel architectures. Notable proposals since AlexNet (Krizhevsky et al., 2012) include VGG (Simonyan & Zisserman, 2014), ResNet (He et al., 2015), Inception (Szegedy et al., 2015; 2016), and ResNeXt (Xie et al., 2017). Automated search strategies for designing architectures have further pushed the state-of-the-art, notably with NasNet-A (Zoph et al., 2018), AmoebaNet-A (Real et al., 2019) and EfficientNet (Tan & Le, 2019). There have also been efforts in going beyond standard ConvNets for image classification, by adapting self-attention (Vaswani et al., 2017) to the visual domain (Bello et al., 2019; Ramachandran et al., 2019; Hu et al., 2019; Shen et al., 2020; Dosovitskiy et al., 2020) or using alternatives such as lambda layers (Bello, 2021).

**架构**。获得最多注意力的工作可能是新的架构。自从AlexNet之后，值得注意的包括VGG，ResNet，Inception和ResNeXt。设计架构的自动搜索策略进一步推进了最好效果，包括NasNet-A，AmoebaNet-A和EfficientNet。在标准卷积网络以外进行图像分类也有一些工作，包括修改自注意力进行视觉识别，或使用比如lambda层的工作。

**Training and Regularization Methods**. ImageNet progress has been boosted by innovations in training and regularization approaches. When training models for more epochs, regularization methods such as dropout (Srivastava et al., 2014), label smoothing (Szegedy et al., 2016), stochastic depth (Huang et al., 2016), dropblock (Ghiasi et al., 2018) and data augmentation (Zhang et al., 2017; Yun et al., 2019; Cubuk et al., 2018; 2019) have significantly improved generalization. Improved learning rate schedules (Loshchilov & Hutter, 2016; Goyal et al., 2017) have further increased final accuracy. While benchmarking architectures in a short non-regularized training setup facilitates fair comparisons with prior work, it is unclear whether architectural improvements are sustained at larger scales and improved training setups. For example, the RegNet architecture (Radosavovic et al., 2020) shows strong speedups over baselines in a short non-regularized training setup, but was not tested in a state-of-the-art ImageNet setup (best top-1 is 79.9%).

**训练和正则化的方法**。通过新的训练和正则化方法也可以推进ImageNet上的效果。当模型训练更多的epochs，一些正则化方法可以显著改进泛化效果，如dropout，label smoothing, stochastic depth, dropblock, 和data augmentation。改进的学习速率方案进一步改进的最终的准确率。虽然在短期的非正则化的训练设置中，对架构进行基准测试，促进了与之前的工作的公平比较，但在更大的尺度上，在改进的训练设置中，架构改进是否仍能够保持，这是不清楚的。比如，RegNet架构在短的非正则化训练设置中，展现出了比基准的很强的加速，但在目前最好的ImageNet设置中，并未进行测试（最佳的top-1是79.9%）。

**Scaling Strategies**. Increasing the model dimensions (e.g. width, depth and resolution) has been another successful axis to improve quality (Rosenfeld et al., 2019; Hestness et al., 2017). Sheer scale was exhaustively demonstrated to improve performance of neural language models (Kaplan et al., 2020) which motivated the design of ever larger models including GPT-3 (Brown et al., 2020) and Switch Transformer (Fedus et al., 2021). Similarly, scale in computer vision has proven useful. Huang et al. (2018) designed and trained a 557 million parameter model, AmoebaNet, which achieved 84.4% top-1 ImageNet accuracy. Typically, ResNet architectures are scaled up by adding layers (depth): ResNets, suffixed by the number of layers, have marched onward from ResNet-18 to ResNet-200, and beyond (He et al., 2016; Zhang et al., 2020; Bello, 2021). Wide ResNets (Zagoruyko & Komodakis, 2016) and MobileNets (Howard et al., 2017) instead scale the width. Increasing image resolutions has also been a reliable source of progress. Thus as training budgets have grown, so have the image resolutions: EfficientNet uses 600 image resolutions (Tan & Le, 2019) and both ResNeSt (Zhang et al., 2020) and TResNet (Ridnik et al., 2020) use 448 image resolutions for their largest model. In an attempt to systematize these heuristics, EfficientNet proposed the compound scaling rule, which recommended balancing the network depth, width and image resolution. However, Section 7.2 shows this scaling strategy is sub-optimal for not only ResNets, but EfficientNets as well.

**缩放策略**。增加模型的维度（如，深度，宽度和分辨率）是改进质量的另一种成功方法。纯粹只用尺度已经证明就可以改进NLP模型的性能，这促使人们设计更大的模型，博爱阔GPT-3和Switch Transformer。类似的，计算机视觉中的尺度也证明了是有用的。Huang等设计训练了一个557M参数的模型AmoebaNet，获得了ImageNet top-1 84.4%的准确率。典型的，ResNet架构通过增加层数（深度），从ResNet-18到ResNet-200，还有更深的模型。Wide ResNets和MobileNets对宽度进行缩放。增加图像分辨率也是改进的一个可靠方法。因此训练代价增加了，图像分辨率也增加了：EfficientNet在其最大的模型中使用600的图像分辨率，而ResNeSt和TResNet在其最大的模型中使用448的图像分辨率。为将这些探索系统化，EfficientNet提出复合缩放规则，建议将网络深度、宽度和图像分辨率进行平衡。但是，7.2节证明了，这种缩放策略对ResNets和EfficientNets都不是最优的。

**Additional Training Data**. Another popular way to further improve accuracy is by training on additional sources of data (either labeled, weakly labeled, or unlabeled). Pretraining on large-scale datasets (Sun et al., 2017; Mahajan et al., 2018; Kolesnikov et al., 2019) has significantly pushed the state-of-the-art, with ViT (Dosovitskiy et al., 2020) and NFNets (Brock et al., 2021) recently achieving 88.6% and 89.2% ImageNet accuracy respectively. Noisy Student, a semi-supervised learning method, obtained 88.4% ImageNet top-1 accuracy by using pseudo-labels on an extra 130M unlabeled images (Xie et al., 2020). Meta pseudo-labels (Pham et al., 2020), an improved semi-supervised learning technique, currently holds the ImageNet state-of-the-art (90.2%). We present semi-supervised learning results in Table 4 and discuss how our training and scaling strategies transfer to large data regimes in Section 8.

**额外的训练数据**。另一种改进准确率的方法是，在更多的数据上进行训练（可以是标注的，弱标注的或无标注的）。在大规模数据集上进行预训练明显提升了最好的效果，ViT和NFNets最近分别获得了88.6%和89.2%的ImageNet准确率。Noisy Student是一种半监督的学习方法，在额外的130M无标签图像中使用伪标签，获得了88.4%的ImageNet top-1准确率。Meta pseudo-labels是一种改进的半监督的学习技术，目前保持着ImageNet上的最好效果(90.2%)。我们在表4中给出了半监督的学习结果，在第8部分中讨论了我们的训练和缩放方法如何迁移到大数据范畴内。

## 3. Related Work on Improving ResNets

Improved training methods combined with architectural changes to ResNets have routinely yielded competitive ImageNet performance (He et al., 2018; Lee et al., 2020; Ridnik et al., 2020; Zhang et al., 2020; Bello, 2021; Brock et al., 2021). He et al. (2018) achieved 79.2% top-1 ImageNet accuracy (a +3% improvement over their ResNet-50 baseline) by modifying the stem and downsampling block while also using label smoothing and mixup. Lee et al. (2020) further improved the ResNet-50 model with additional architectural modifications such as Squeeze-and-Excitation (Hu et al., 2018), selective kernel (Li et al., 2019), and anti-alias downsampling (Zhang, 2019), while also using label smoothing, mixup, and dropblock to achieve 81.4% accuracy. Ridnik et al. (2020) incorporated several architectural modifications to the ResNet architectures along with improved training methodologies to outperform EfficientNet-B1 to EfficientNet-B5 models on the speed-accuracy Pareto curve.

改进的训练方法与架构变化相结合，在ImageNet上得到了很好的结果。He等修改了stem和下采样模块，还使用了label smoothing和mixup，获得了79.2%的ImageNet top-1准确率。Lee等用额外的架构修改，比如SE，selective kernel和anti-alias下采样，同时使用了label smoothing，mixup和dropblock，进一步改进了ResNet-50模型，获得了81.4%的准确率。Ridnik等将几种架构修正与ResNet架构结合，使用改进的训练方法，在速度-准确率的Pareto曲线上超过了EfficientNet-B1到EfficientNet-B5。

Most works, however, put little emphasis on identifying strong scaling strategies. In contrast, we only consider lightweight architectural changes routinely used since 2018 and instead focus on the training and scaling strategies to build a Pareto curve of models. Our improved training and scaling methods lead to ResNets that are 1.7x - 2.7x faster than EfficientNets on TPUs. Our scaling improvements are orthogonal to the aforementioned methods and we expect them to be additive.

但是，多数工作都没有强调缩放策略。相比，我们考虑的架构变化很少，而聚焦在训练和缩放策略中，以构建模型的Pareto曲线。我们改进的训练和缩放方法带来的ResNets，在TPU上运行比EfficientNets快1.7x-2.7x。我们的缩放改进与之前提到的方法是不相关的，我们期望这些改进是加性的。

## 4. Methodology

We describe the base ResNet architecture and the training methods used throughout this paper.

### 4.1. Architecture

Our work studies the ResNet architecture, with two widely used architecture changes, the ResNet-D (He et al., 2018) modification and Squeeze-and-Excitation (SE) in all bottleneck blocks (Hu et al., 2018). These architectural changes are used in used many architectures, including TResNet, ResNeSt and EfficientNets.

ResNet-D (He et al., 2018) combines the following four adjustments to the original ResNet architecture. First, the 7×7 convolution in the stem is replaced by three smaller 3×3 convolutions, as first proposed in Inception-V3 (Szegedy et al., 2016). Second, the stride sizes are switched for the first two convolutions in the residual path of the downsampling blocks. Third, the stride-2 1×1 convolution in the skip connection path of the downsampling blocks is replaced by stride-2 2×2 average pooling and then a non-strided 1×1 convolution. Fourth, the stride-2 3×3 max pool layer is removed and the downsampling occurs in the first 3×3 convolution in the next bottleneck block. We diagram these modifications in Figure 6.

Squeeze-and-Excitation (Hu et al., 2018) reweighs channels via cross-channel interactions by average pooling signals from the entire feature map. For all experiments we use a Squeeze-and-Excitation ratio of 0.25 based on preliminary experiments. In our experiments, we sometimes use the original ResNet implementation without SE (referred
to as ResNet) to compare different training methods. Clear denotations are made in table captions when this is the case.

### 4.2. Training Methods

We study regularization and data augmentation methods that are routinely used in state-of-the art classification models and semi/self-supervised learning. 我们研究了在目前最好的分类模型中和半/自监督学习中经常使用的正则化和数据扩增方法。

**Matching the EfficientNet Setup**. Our training method closely matches that of EfficientNet, where we train for 350 epochs, but with a few small differences. (1) We use the cosine learning rate schedule (Loshchilov & Hutter, 2016) instead of an exponential decay for simplicity (no additional hyperparameters). (2) We use RandAugment (Cubuk et al., 2019) in all models, whereas EfficientNets were originally trained with AutoAugment (Cubuk et al., 2018). We reran EfficientNets B0-B4 with RandAugment and found it offered no performance improvement and report EfficientNet B5 and B7 with the RandAugment results from Cubuk et al. (2019). (3) We use the Momentum optimizer instead of RMSProp for simplicity. See Table 10 in the Appendix C for a comparison between our training setup and EfficientNet.

**与EfficientNet设置匹配**。我们的训练方法与EfficientNet很接近，训练了350 epochs，但有一点点差异。(1)哦我们使用cosine学习速率方案，而不是指数衰减，这样更简单一些（没有额外的超参数）；(2)我们在所有模型中使用RandAugment，而EfficientNets最早是使用AutoAugment进行训练的。我们用RandAugment重新运行了EfficientNets B0-B4，发现并没有性能改进，而EfficientNet B5到B7的RandAugment结果使用的是Cubuk等；(3)我们使用Momentum优化器，而不是RMSProp。见附录C中的表10，与EfficientNet的训练设置进行比较。

**Regularization**. We apply weight decay, label smoothing, dropout and stochastic depth for regularization. Dropout (Srivastava et al., 2014) is a common technique used in computer vision and we apply it to the output after the global average pooling occurs in the final layer. Stochastic depth (Huang et al., 2016) drops out each layer in the network (that has residual connections around it) with a specified probability that is a function of the layer depth.

**正则化**。我们进行了权重衰减，标签平滑，dropout和随机深度进行正则化。Dropout是在计算机视觉中常用的技术，我们将其使用在最后一层的全局平均池化之后。随机深度用指定的概率丢弃网络中的每一层（周围有残差连接），该概率是层深的函数。

**Data Augmentation**. We use RandAugment (Cubuk et al., 2019) data augmentation as an additional regularizer. RandAugment applies a sequence of random image transformations (e.g. translate, shear, color distortions) to each image independently during training. As mentioned earlier, originally EfficientNets uses AutoAugment (Cubuk et al., 2018), which is a learned augmentation procedure that slightly underperforms RandAugment.

**数据扩增**。我们使用RandAugment数据扩增技术，作为额外的正则化。RandAugment在训练时对每幅图像独立的使用一系列随机的图像变换（如，平移，剪切，色彩变换）。之前提及过，原始的EfficientNets使用的是AutoAugment，这是一种学习得到的扩增方法，比RandAugment效果略差。

**Hyperparameter Tuning**. To select the hyperparameters for the various regularization and training methods, we use a held-out validation set comprising 2% of the ImageNet training set (20 shards out of 1024). This is referred to as the minival-set and the original ImageNet validation set (the one reported in most prior works) is referred to as validation-set. The hyperparameters of all ResNet-RS models are in Table 8 in the Appendix B.

**超参数调节**。为对各种正则化和训练方法选择超参数，我们使用一个保留的验证集，是由ImageNet训练集的2%构成的。这称为mini-val集，原始的ImageNet验证集称之为validation集。所有的ResNet-RS模型的超参数如附录B的表8所示。

## 5. Improved Training Methods

### 5.1. Additive Study of Improvements

We present an additive study of training, regularization methods and architectural changes in Table 1. The baseline ResNet-200 gets 79.0% top-1 accuracy. We improve its performance to 82.2% (+3.2%) through improved training methods alone without any architectural changes. When adding two common and simple architectural changes (Squeeze-and-Excitation and ResNet-D) we further boost the performance to 83.4%. Training methods alone cause 3/4 of the total improvement, which demonstrates their critical impact on ImageNet performance.

我们在表1中给出了训练方法、正则化方法和架构变化的加性研究。基准的ResNet-200 top-1准确率为79.0%。我们通过改进训练方法，性能改进到82.2%。当加上了两个常用的简单架构变化，进一步改进到了83.4%。训练方法的改进占了3/4，这证明了其关键影响。

### 5.2. Importance of decreasing weight decay when combining regularization methods

Table 2 highlights the importance of changing weight decay when combining regularization methods together. When applying RandAugment and label smoothing, there is no need to change the default weight decay of 1e-4. But when we further add dropout and/or stochastic depth, the performance can decrease unless we further decrease the weight decay. The intuition is that since weight decay acts as a regularizer, its value must be decreased in order to not overly regularize the model when combining many techniques. Furthermore, Zoph et al. (2020a) presents evidence that the addition of data augmentation shrinks the L2 norm of the weights, which renders some of the effects of weight decay redundant. Other works use smaller weight decay values, but do not point out the significance of the effect when using more regularization (Tan et al., 2019; Tan & Le, 2019).

表2强调了，在与正则化方法相结合时，改变权重衰减的重要性。当使用了RandAugment和标签平滑时，不需要改变默认的权重衰减1e-4。但当我们加入了dropout和/或随机深度时，性能会下降，除非我们进一步降低权重衰减。直觉是这样的，由于权重衰减是一个正则化器，其值必须降低，以在与很多技术结合时，不过度正则化模型。而且，Zoph等给出了证据，加上了数据扩增，会使权重的L2范数缩减。其他工作使用了更小的权重衰减值，但并没有指出在使用更多的正则化时这个动作的重要性。

## 6. Improved Scaling Strategies

The prior section demonstrates the significant impact of training methodology and we now show the scaling strategy is similarly important. In order to establish scaling trends, we perform an extensive search on ImageNet over width multipliers in [0.25, 0.5, 1.0, 1.5, 2.0], depths of [26, 50, 101, 200, 300, 350, 400] and resolutions of [128, 160, 224, 320, 448]. We train these architectures for 350 epochs, mimicking the training setup of state-of-the-art ImageNet models. We increase the regularization as the model size increases to limit overfitting. See Appendix E for regularization and model hyperparameters.

前一节证明了训练方法的显著影响，我们现在展示缩放策略的重要性也是类似的。为确立缩放的趋势，我们在ImageNet上进行了广泛的搜索，包括宽度乘子范围[0.25, 0.5, 1.0, 1.5, 2.0]，深度[26, 50, 101, 200, 300, 350, 400]和分辨率[128, 160, 224, 320, 448]。我们训练这些架构350 epochs，模仿目前最好的ImageNet模型的训练设置。我们在模型大小增加时，也增加正则化，也防止过拟合。附录E中给出了正则化和模型的超参数。

**FLOPs do not accurately predict performance in the bounded data regime**. Prior works on scaling laws observe a power law between error and FLOPs in unbounded data regimes (Kaplan et al., 2020; Henighan et al., 2020). In order to test whether this also holds in our scenario, we plot ImageNet error against FLOPs for all scaling configurations in Figure 2. For the smaller models, we observe an overall power law trend between error and FLOPs, with minor dependency on the scaling configuration (i.e. depth versus width versus image resolution). However, the trend breaks for larger model sizes. Furthermore, we observe a large variation in ImageNet performance for a fixed amount of FLOPs, especially in the higher FLOP regime. Therefore the exact scaling configuration (i.e. depth, width and image resolution) can have a big impact on performance even when controlling for the same amount of FLOPs.

**FLOPs在有限的数据范围内不能准确的预测性能**。之前在缩放规则上的工作，在没有限制的数据范围内，观察到了错误率和FLOPs之间有幂次规则。为测试在我们的场景中是否也是如此，我们在图2中画出了ImageNet错误率和FLOPs在所有缩放配置下的图。对于较小的模型，我们观察到错误率和FLOPs之间有大致的幂次规则，与缩放配置也有少许关系（即，深度vs宽度vs图像分辨率）。但是，对于更大的模型，这个趋势就不成立了。而且，我们在固定的FLOPs下，观察到ImageNet性能有很大的变化，尤其是在更高的FLOPs区域中。因此，在固定数量的FLOPs下，确切的缩放配置（即，深度，宽度和图像分辨率）对性能有很大的影响。

**The best performing scaling strategy depends on the training regime**. We next look directly at latencies on the hardware of interest to identify scaling strategies that improve the speed-accuracy Pareto curve. Figure 3 presents accuracies and latencies of models scaled with either width or depth across four image resolutions and three different training regimes (10, 100 and 350 epochs). We observe that the best performing scaling strategy, especially whether to scale depth and/or width, highly depends on the training regime.

**最佳性能的缩放策略依赖于训练方案**。下一步我们直接观察了感兴趣硬件的延迟，以发现改进速度-准确率Pareto曲线的缩放策略。图3给出了模型的准确率和延迟，在四种图像分辨率上和三种不同的训练方法上（10，100，350 epochs）缩放了宽度或深度。我们观察到，最佳性能的缩放策略是高度依赖于训练方法的，尤其是是否缩放深度还是宽度。

### 6.1. Strategy #1 - Depth Scaling in Regimes Where Overfitting Can Occur

**Depth scaling outperforms width scaling for longer epoch regimes**. In the 350 epochs setup (Figure 3, right panel), we observe depth scaling to significantly outperform width scaling across all image resolutions. Scaling the width is subject to overfitting and sometimes hurts performance even with increased regularization. We hypothesize that this is due to the larger increase in parameters when scaling the width. The ResNet architecture maintains constant FLOPs across all block groups and multiplies the number of parameters by 4× every block group. Scaling the depth, especially in the earlier layers, therefore introduces fewer parameters compared to scaling the width.

**在更长的训练方案中，深度缩放性能超过了宽度缩放**。在350 epoch设置中，我们观察到，深度缩放在所有分辨率设置中都超过了宽度缩放。缩放宽度受到过拟合的影响，即使增加正则化，有时候也会损伤性能。我们推测，这是因为增加深度时，参数也增加了。缩放深度与缩放宽度比起来，尤其是在更早的层中，带来的参数更少。

**Width scaling outperforms depth scaling for shorter epoch regimes**. In contrast, width scaling is better when only training for 10 epochs ( Figure 3, left panel). For 100 epochs (Figure 3, middel panel), the best performing scaling strategy varies between depth scaling and width scaling, depending on the image resolution. The dependency of the scaling strategy on the training regime reveals a pitfall of extrapolating scaling rules. We point out that prior works also choose to scale the width when training for a small number of epochs on large-scale datasets (e.g. ∼40 epochs on 300M images), consistent with our experimental findings that scaling the width is preferable in shorter epoch regimes. In particular, Kolesnikov et al. (2019) train a ResNet-152 with 4x filter multiplier while Brock et al. (2021) scales the width with ∼1.5x filter multiplier.

**在更短的训练方案中，宽度缩放性能超过了深度缩放**。比较起来，宽度缩放在只训练10 epoch时，效果更好。对于100 epoch，最佳性能的缩放策略既有深度缩放，也有宽度缩放，这依赖于图像分辨率。缩放策略对训练方案的依赖性，说明外插缩放规则有一定的漏洞。我们指出，之前的工作在大规模数据集上训练epoch更少时也选择缩放宽度（300M图像时，大约40 epochs），与我们的试验发现类似。

### 6.2. Strategy #2 - Slow Image Resolution Scaling

In Figure 2, we also observe that larger image resolutions yield diminishing returns. We therefore propose to increase the image resolution more gradually than previous works. This contrasts with the compound scaling rule proposed by EfficientNet which leads to very large images (e.g. 600 for EfficientNet-B7, 800 for EfficientNet-L2 (Xie et al., 2020)). Other works such as ResNeSt (Zhang et al., 2020) and TResNet (Ridnik et al., 2020)) scale the image resolution up to 448. Our experiments indicate that slower image scaling improves not only ResNet architectures, but also EfficientNets on a speed-accuracy basis (Section 7.2).

在图2中，我们还观察到，图像分辨率更大，得到的回报会逐渐消失。我们因此提出，比之前的工作更慢的增加图像分辨率。这也与EfficientNet提出的复合缩放规则形成对比，其会带来很大的图像（如，对EfficientNet-B7是600分辨率的图像，对EfficientNet-L2是800分辨率的图像）。其他的工作，如ResNeSt和TResNet将图像分辨率增加到了448。我们的试验说明，更慢的图像缩放，不仅改进ResNet架构，而且也改进了EfficientNets的速度-准确率基准。

### 6.3. Two Common Pitfalls in Designing Scaling Strategies

Our scaling analysis surfaces two common pitfalls in prior research on scaling strategies: 我们的缩放分析说明，之前研究中的缩放策略有两个主要的缺陷：

(1) **Extrapolating scaling strategies from small-scale regimes**. Scaling strategies found in small scale regimes (e.g. on small models or with few training epochs) can fail to generalize to larger models or longer training iterations. The dependencies between the best performing scaling strategy and the training regime are missed by prior works which extrapolate scaling rules from either small models (Tan & Le, 2019) or shorter training epochs (Radosavovic et al., 2020). We therefore do not recommend generating scaling rules exclusively in a small scale regime because these rules can break down.

(1)**从小尺度范围外插缩放策略**。在小尺度范围内发现的缩放策略（如，小模型，或很少的训练epochs），泛化到更大的模型或更长的训练迭代次数时，可能会失败。最佳性能的缩放策略和训练方法的依赖关系，之前的工作中是没有的，他们都是从小模型或更短的训练epochs中进行缩放规则外插。我们因此不推荐只从小规模的范围内生成缩放规则，因为这些规则可能会失败。

(2) **Extrapolating scaling strategies from a single and potentially sub-optimal initial architecture**. Beginning from a sub-optimal initial architecture can skew the scaling results. For example, the compound scaling rule derived from a small grid search around EfficientNet-B0, which was obtained by architecture search using a fixed FLOPs budget and a specific image resolution. However, since this image resolution can be sub-optimal for that FLOPs budget, the resulting scaling strategy can be sub-optimal. In contrast, our work designs scaling strategies by training models across a variety of widths, depths and image resolutions.

(2)**从单个初始架构和可能是次优的初始架构中外插缩放策略**。从一个次优的初始架构开始，可能会使缩放结果扭曲。比如，从EfficientNet-B0附近的小网格搜索，推导得到的复合缩放规则，这是使用固定的FLOPs预算和指定的图像分辨率通过架构搜索得到的。但是，因为图像分辨率对于那个FLOPs预算可能是次优的，得到的缩放策略可能是次优的。比较之下，我们的工作通过在很大范围的宽度，深度和图像分辨率训练模型，来设计缩放策略。

### 6.4. Summary of Improved Scaling Strategies

For a new task, we recommend running a small subset of models across different scales, for the full training epochs, to gain intuition on which dimensions are the most useful across model scales. While this approach may appear more costly, we point out that the cost is offset by not searching for the architecture.

对于一个新任务，我们推荐在不同的尺度上运行模型的一个小的子集，完全的训练epochs，以得到在哪个维度上是在各种模型尺度上最有用的。虽然这种方法可能代价比较昂贵，我们指出，如果不对架构进行搜索，代价可能是有偏的。

For image classification, the scaling strategies are summarized as (1) scale the depth in regimes where overfitting can occur (scaling the width is preferable otherwise) and (2) slow image resolution scaling. Experiments indicate that applying these scaling strategies to ResNets (ResNet-RS) and EfficientNets (EfficientNet-RS) leads to significant speed-ups over EfficientNets. We note that similar scaling strategies are also employed in recent works that obtain large speed-ups over EfficientNets such as LambdaResNets (Bello, 2021) and NFNets (Brock et al., 2021).

对图像分类，缩放策略总结如下，(1)出现过拟合时，增加深度（增加宽度则在其他情况下较好）；(2)减缓图像分辨率的增加。试验表明，对ResNets和EfficientNet使用这些缩放策略会带来显著的加速。我们指出，最近得到很大的加速的工作中，也采用了类似的缩放策略，如LambdaResNets和NFNets。

## 7. Experiments with Improved Training and Scaling Strategies

### 7.1. ResNet-RS on a Speed-Accuracy Basis

Using the improved training and scaling strategies, we design ResNet-RS, a family of re-scaled ResNets across a wide range of model scales (see Appendix B and D for experimental and architectural details). Figure 4 compares EfficientNets against ResNet-RS on a speed-accuracy Pareto curve. We find that ResNet-RS match Efficient-Nets' performance while being 1.7x - 2.7x faster on TPUs. This large speed-up over EfficientNet may be non-intuitive since EfficientNets significantly reduce both the parameter count and the FLOPs compared to ResNets. We next discuss why a model with fewer parameters and fewer FLOPs (EfficientNet) is slower and more memory-intensive during training.

使用改进的训练和缩放策略，我们设计了ResNet-RS，在很宽的模型尺度上进行重新缩放的ResNets族。图4比较了EfficientNets和ResNet-RS的速度-准确率Pareto曲线。我们发现ResNet-RS与EfficientNets性能类似，但在TPU上速度快了1.7x-2.7x。这种加速可能不太直观，因为EfficientNets与ResNets相比，显著降低了参数数量和FLOPs。下面我们讨论，为什么一个参数更少FLOPs更少的模型，在训练时会更慢，使用更多的内存。

**FLOPs vs Latency**. While FLOPs provide a hardware-agnostic metric for assessing computational demand, they may not be indicative of actual latency times for training and inference (Howard et al., 2017; 2019; Radosavovic et al., 2020). In custom hardware architectures (e.g. TPUs and GPUs), FLOPs are an especially poor proxy because operations are often bounded by memory access costs and have different levels of optimization on modern matrix multiplication units (Jouppi et al., 2017). The inverted bottlenecks (Sandler et al., 2018) used in EfficientNets employ depthwise convolutions with large activations and have a small compute to memory ratio (operational intensity) compared to the ResNet's bottleneck blocks which employ dense convolutions on smaller activations. This makes EfficientNets less efficient on modern accelerators compared to ResNets. Table 3 illustrates this point: a ResNet-RS model with 1.8x more FLOPs than EfficientNet-B6 is 2.7x faster on a TPUv3 hardware accelerator. 

FLOPs给出了评估计算需求的设备无关度量，但对于训练和推理，却并不能指示其实际延迟。在定制的硬件架构中，FLOPs是非常差的度量，因为运算通常受到内存访问代价的限制，在现代的矩阵乘法单元中，有不同层次的优化。EfficientNets中使用的逆瓶颈结构，采用了分层卷积，有很多激活值，其计算量对存储的比值很小，比较起来，ResNet中的瓶颈结构使用了更小的激活上的密集卷积。这使得EfficientNet在现代加速器上效率要更低一些。表3描述了这一点：一个FLOPs 1.8x的ResNet-RS模型，在TPUv3上比EfficientNet-B6快了2.7x。

**Parameters vs Memory**. Parameter count does not necessarily dictate memory consumption during training because memory is often dominated by the size of the activations. The large activations used in EfficientNets also cause larger memory consumption, which is exacerbated by the use of large image resolutions, compared to our rescaled ResNets. A ResNet-RS model with 3.8x more parameters than EfficientNet-B6 consumes 2.3x less memory for a similar ImageNet accuracy (Table 3). We emphasize that both memory consumption and latency are tightly coupled to the software and hardware stack (TensorFlow on TPUv3) due to compiler optimizations such as operation layout assignments and memory padding.

参数数量在训练过程中并不一定指示内存消耗量，因为内存通常是由激活的大小决定的。EfficientNet中使用了很大的激活，使其内存消耗很大，其图像分辨率很大也加剧了这一点。参数数量3.8x的ResNet-RS模型，比EfficientNet-B6占用的内存少了2.3x，其ImageNet准确率是类似的。我们强调，内存消耗和延迟与软件和硬件是紧密相关的，因为编译器进行了相当的优化。

### 7.2. Improving the Efficiency of EfficientNets

The scaling analysis from Section 6 reveals that scaling the image resolution results in diminishing returns. This suggests that the scaling rules advocated in EfficientNets which increases model depth, width and resolution independently of model scale is sub-optimal. We apply the slow image resolution scaling strategy (Strategy #2) to EfficientNets and train several versions with reduced image resolutions, without changing the width or depth. The RandAugment magnitude is set to 10 for image resolution 224 or smaller, 20 for image resolution larger than 320 and 15 otherwise. All other hyperparameters are kept the same as per the original EfficientNets. Figure 5 demonstrates a marked improvement of the re-scaled EfficientNets (EfficientNet-RS) on the speed-accuracy Pareto curve over the original EfficientNets.

第6部分的缩放分析说明，增加图像分辨率带来的提升会越来越小。这说明，EfficientNet中的缩放规则，即与模型大小无关的增加模型深度、宽度和分辨率，并不是最优的。我们对EfficientNets使用了更慢的图像分辨率增加策略，用降低的图像分辨率训练了几个版本，不改变宽度或深度。RandAugment幅度对图像分辨率224或更小的设置为10，对大于320的设置为20，其他为15。所有其他超参数与原始EfficientNets保持一样。图5证明了，EfficientNet-RS在速度-准确率的Pareto曲线上比原始EfficientNet改进了很多。

### 7.3. Semi-Supervised Learning with ResNet-RS

We measure how ResNet-RS performs as we scale to larger datasets in a large scale semi-supervised learning setup. We train ResNets-RS on the combination of 1.2M labeled ImageNet images and 130M pseudo-labeled images, in a similar fashion to Noisy Student (Xie et al., 2020). We use the same dataset of 130M images pseudo-labeled as Noisy Student, where the pseudo labels are generated from an EfficientNet-L2 model with 88.4% ImageNet accuracy. Models are jointly trained on both the labeled and pseudo-labeled data and training hyperparameters are kept the same. Table 4 reveals that ResNet-RS models are very strong in the semi-supervised learning setup as well. We obtain a top-1 ImageNet accuracy of 86.2%, while being 4.7x faster on TPU (5.5x on GPU) than the corresponding Noisy Student EfficientNet-B5 model.

在一个大规模半监督学习设置中，我们在更大的数据集上进行训练，我们检测ResNet-RS表现怎样。我们在1.2M标注的ImageNet图像和130M伪标注图像上训练ResNets-RS，与Noisy Student方式类似。我们使用与Noisy Student使用了相同的130M图像伪标签，其中伪标签是由EfficientNet-L2生成的，其ImageNet准确率为88.4%。模型在标注和伪标注的图像上进行联合训练，训练超参数保持一致。表4给出了，ResNet-RS模型在半监督设置中效果也非常好。我们得到的top-1 ImageNet准确率为86.2%，与对应的Noisy Student EfficientNet-B5模型相比，在TPU上快了4.7x（GPU上快了5.5x）。

### 7.4. Transfer Learning of ResNet-RS

We now investigate whether the improved supervised training strategies yield better representations for transfer learning and compare them with self-supervised learning algorithms. Recent self-supervised learning algorithms claim to surpass the transfer learning performance of supervised learning and create more universal representations (Chen et al., 2020a;b). Self-supervised algorithms, however, make several changes to the training methods (e.g training for more epochs, data augmentation) making comparisons to supervised learning difficult. Table 5 compares the transfer performance of improved supervised training strategies (denoted RS) against self-supervised SimCLR (Chen et al., 2020a) and SimCLRv2 (Chen et al., 2020b). In an effort to closely match SimCLR's training setup and provide fair comparisons, we restrict the RS training strategies to a subset of its original methods. Specifically, we employ data augmentation (RandAugment), label smoothing, dropout, decreased weight decay and cosine learning rate decay for 400 epochs but do not use stochastic depth or exponential moving average (EMA) of the weights. We choose this subset to closely match the training setup of SimCLR: longer training, data augmentation and a temperature parameter for their contrastive loss. We use the vanilla ResNet architecture without the ResNet-D modifications or Squeeze-and-Excite, matching the SimCLR and SimCLRv2 architectures.

我们现在研究，改进的有监督训练策略，在迁移学习中是否得到更好的表示，并将其与自监督学习算法比较。最近的自监督学习算法声称，超过了有监督学习的迁移学习性能，创造了更通用的表示。自监督算法对训练方法进行了一些改变（如，训练epochs更多，数据扩增），这使得与有监督学习的比较更加困难。表5比较了，改进的有监督训练策略与自监督学习。为接近SimCLR的训练设置，进行公平的比较，我们将RS的训练方法限制到了原始方法的一个子集。具体的，我们采用了数据扩增(RandAugment)，标签平滑，dropout，降低了权重衰减，cosine学习速率，进行了400 epochs训练，但并没有使用随机深度或权重的EMA。我们选择这个子集，以与SimCLR的训练设置尽可能接近：更长的训练，数据扩增，对比损失的温度参数。我们使用传统ResNet架构，没有使用ResNet-D修正或SE，与SimCLR和SimCLRv2架构匹配。

We evaluate the transfer performance on five downstream tasks: CIFAR-100 Classification (Krizhevsky et al., 2009), Pascal Detection & Segmentation (Everingham et al., 2010), ADE Segmentation (Zhou et al., 2017) and NYU Depth (Silberman et al., 2012). We find that, even when restricted to a smaller subset, the improved training strategies improve transfer performance. The improved supervised representations (RS) outperform SimCLR on 5/10 downstream tasks and SimCLRv2 on 8/10 tasks. Furthermore, the improved training strategies significantly outperform the standard supervised ResNet representations, highlighting the need for using modern training techniques when comparing to self-supervised learning. While self-supervised learning can be used on unlabeled data, our results challenge the notion that self-supervised algorithms lead to more universal representations than supervised learning when labels are available.

我们在5个下游任务中评估了迁移性能：CIFAR-100分类，PASCAL检测和分割，ADE分割和NYU Depth。我们发现，即使是使用了更小的子集，改进的训练策略也改进了迁移学习的性能。改进的有监督表示(RS)在5/10的下游任务中超过了SimCLR，在8/10的任务中超过了SimCLRv2。而且，改进的训练策略显著超过了标准有监督ResNet表示，强调了使用现代训练技术的必要性。虽然自监督可以使用在未标注的数据中，我们的结果还是挑战了这个概念，即自监督算法比有监督学习带来了更通用的表示。

### 7.5. Revised 3D ResNet for Video Classification

We conclude by applying the training strategies to the Kinetics-400 video classification task, using a 3D ResNet as the baseline architecture (Qian et al., 2020) (see Appendix G for experimental details). Table 6 presents an additive study of the RS training recipe and architectural improvements.

我们将训练策略应用于Kinetics-400视频分类任务，使用3D ResNet作为基准架构，得到结论。表6给出了RS训练方案和架构改进的加性研究。

The training strategies extend to video classification, yielding a combined improvement from 73.4% to 77.4% (+4.0%). The ResNet-D and Squeeze-and-Excitation architectural changes further improve the performance to 78.2% (+0.8%). Similarly to our study on image classification (Table 1), we find that most of the improvement can be obtained without architectural changes. Without model scaling, 3D ResNet-RS-50 is only 2.2% less than the best number reported on Kinetics-400 at 80.4% (Feichtenhofer, 2020).

训练策略拓展到视频分类，将形从73.4%改进到了77.4%。ResNet-D和SE架构变化进一步改进到了78.2%。与我们在图像分类中的研究类似（表1），我们发现，不需要架构变化，我们也能得到大部分改进。不进行模型缩放，3D ResNet-RS-50只比Kinetics-400上最佳表现的80.4%性能低了2.2%。

## 8. Discussion

**Why is it important to tease apart improvements coming from training methods vs architectures**? Training methods can be more task-specific than architectures (e.g. data augmentation is more helpful on small datasets). Therefore, improvements coming from training methods do not necessarily generalize as well as architectural improvements. Packaging newly proposed architectures together with training improvements makes accurate comparisons between architectures difficult. The large improvements coming from training strategies, when not being controlled for, can overshadow architectural differences.

**为什么将训练方法和架构中得到的改进区分开来**？训练方法比架构更加依赖于任务（如，数据扩增在小型数据集上帮助更大）。因此，从训练方法中得到的改进并不像架构改进泛化效果那么好。将新提出的架构，与训练改进放到一起，使准确的比较架构更难。训练策略中得到的很大改进，如果不进行控制，会对架构的差异造成很大影响。

**How should one compare different architectures**? Since training methods and scale typically improve performance (Lee et al., 2020; Kaplan et al., 2020), it is critical to control for both aspects when comparing different architectures. Controlling for scale can be achieved through different metrics. While many works report parameters and FLOPs, we argue that latencies and memory consumption are generally more relevant (Radosavovic et al., 2020). Our experimental results (Section 7.1) re-emphasize that FLOPs and parameters are not representative of latency or memory consumption (Radosavovic et al., 2020; Norrie et al., 2021).

**应当怎样比较不同的架构**？由于训练方法和缩放一般都会改进性能，当比较不同的架构时，控制这两个方面就非常关键。控制缩放可以通过不同的度量来得到。很多工作都会给出参数数量和FLOPs，我们认为延迟和内存消耗更加相关。我们的试验结果重新强调了，FLOPs和参数数量并不代表延迟或内存消耗。

**Do the improved training strategies transfer across tasks**? The answer depends on the domain and dataset sizes available. Many of the training and regularization methods studied here are not used in large-scale pretraining (e.g. 300M images) (Kolesnikov et al., 2019; Dosovitskiy et al., 2020). Data augmentation is useful for small datasets or when training for many epochs, but the specifics of the augmentation method can be task-dependent (e.g. scale jittering instead of RandAugment in Table 6).

**改进的训练策略可以迁移到不同任务中吗**？答案依赖于可用的领域和数据集大小。这里研究的很多训练和正则化方法，在大规模预训练中并没有使用（如，300M图像）。数据扩增对小型数据集很有用，在训练很多epochs也很有用，但扩增方法更加依赖于任务。

**Do the scaling strategies transfer across tasks**? The best performing scaling strategy depends on the training regime and whether overfitting is an issue, as discussed in Section 6. When training for 350 epochs on ImageNet, we find scaling the depth to work well, whereas scaling the width is preferable when training for few epochs (e.g. 10 epochs). This is consistent with works employing width scaling when training for few epochs on large-scale datasets (Kolesnikov et al., 2019). We are unsure how our scaling strategies apply in tasks that require larger image resolutions (e.g. detection and segmentation) and leave this to future work.

**缩放策略可以迁移到不同任务中吗**？性能最佳的缩放策略依赖于训练方法和是否过拟合，这是一个问题，这在第6部分进行了讨论。当在ImageNet上训练350 epochs时，我们发现增加深度效果最好，而增加宽度在训练epochs更少时效果更好（如，10 epochs）。这与其他工作是一致的，在大规模数据集上训练少量epochs时采用宽度缩放。我们对于需要更大的图像分辨率的任务中，我们的缩放策略如何表现，不是非常确定，这将在未来的工作中进行研究。

**Are architectural changes useful**? Yes, but training methods and scaling strategies can have even larger impacts. Simplicity often wins, especially given the non-trivial performance issues arising on custom hardware. Architecture changes that decrease speed and increase complexity may be surpassed by scaling up faster and simpler architectures that are optimized on available hardware (e.g convolutions instead of depthwise convolutions for GPUs/TPUs). We envision that future successful architectures will emerge by co-design with hardware, particularly in resource-tight regimes like mobile phones (Howard et al., 2019).

**架构变化有用吗**？是的，但是训练方法和缩放策略的影响甚至更大。简单的架构通常会获胜，尤其是在定制的硬件中性能问题更加non-trivial。架构变化如果降低了速度，增加了复杂度，就可能被缩放更快更简单的架构超过，尤其是在可用的硬件中进行了优化（如，卷积，而不是分层卷积）。我们认为，未来的成功架构会从硬件的协同设计中出现，尤其是在资源有限的范围内，如手机。

**How should one allocate a computational budget to produce the best vision models**? We recommend beginning with a simple architecture that is efficient on available hardware (e.g. ResNets on GPU/TPU) and training several models, to convergence, with different image resolutions, widths and depths to construct a Pareto curve. Note that this strategy is distinct from Tan & Le (2019) which instead allocate a large portion of the compute budget for identifying an optimal initial architecture to scale. They then do a small grid search to find the compound scaling coefficients used across all model scales. RegNet (Radosavovic et al., 2020) does most of their studies when training for only 10 epochs.

**应当怎样分配计算预算，以得到最佳的视觉模型**？我们推荐从简单架构开始，在可用的硬件上高效（如，在GPU/TPU上的ResNets），训练几个模型收敛，用几种不同的图像分辨率，宽度和深度，以构建Pareto曲线。注意，这种策略与Tan&Le不同，他们分配了大量计算预算来得到最佳的初始架构，以进行缩放。他们然后进行了一个小的网格搜索，以找到复合缩放系数，在所有模型尺度上使用。RegNet训练了10 epochs，进行了他们大部分的研究。

## 9. Conclusion

By updating the de facto vision baseline with modern training methods and an improved scaling strategy, we have revealed the remarkable durability of the ResNet architecture. Simple architectures set strong baselines for state-of-the-art methods. We hope our work encourages further scrutiny in maintaining consistent methodology for both proposed innovations and baselines alike.

用现代训练方法和改进的缩放策略，我们更新了视觉基准，揭示了ResNet架构的耐久性。简单的架构对目前最好的方法设置了很强的基准。我们希望我们的工作，对提出的创新和基准，鼓励检查一致的方法。