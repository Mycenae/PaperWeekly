# Speed/accuracy trade-offs for modern convolutional object detectors 现代卷积型目标检测器的速度/准确率折中

Jonathan Huang et al. Google Inc.

## Abstract 摘要

The goal of this paper is to serve as a guide for selecting a detection architecture that achieves the right speed/memory/accuracy balance for a given application and platform. To this end, we investigate various ways to trade accuracy for speed and memory usage in modern convolutional object detection systems. A number of successful systems have been proposed in recent years, but apples-to-apples comparisons are difficult due to different base feature extractors (e.g., VGG, Residual Networks), different default image resolutions, as well as different hardware and software platforms. We present a unified implementation of the Faster R-CNN [31], R-FCN [6] and SSD [26] systems, which we view as “meta-architectures” and trace out the speed/accuracy trade-off curve created by using alternative feature extractors and varying other critical parameters such as image size within each of these meta-architectures. On one extreme end of this spectrum where speed and memory are critical, we present a detector that achieves real time speeds and can be deployed on a mobile device. On the opposite end in which accuracy is critical, we present a detector that achieves state-of-the-art performance measured on the COCO detection task.

本文的目的是成为选择检测架构的指南，对于一个给定的应用和平台，检测架构要能正确的平衡速度/内存/准确度。为此，我们研究了多种方法，再现代的卷积型目标检测系统中，牺牲准确率换得速度和内存使用。近几年来提出了几个成功的系统，但同类型的比较比较困难，因为使用了不同的基础特征提取器（比如，VGG，残差网络），不同的默认图像分辨率，以及不同的硬件和软件平台。我们提出了三种目标检测系统的统一实现，Faster R-CNN[31], R-FCN[6]和SSD[26]，我们称之为“元架构”，勾画出这些元架构的速度/准确率的折中曲线。在这个谱系的一个极端，其中速度和内存是关键，我们提出了一种达到实时速度的检测器，可以部署在移动设备上。在相反的极端情况中，准确率是最关键的，我们提出一种检测器，在COCO检测任务中达到了目前最好的表现。

## 1. Introduction 引言

A lot of progress has been made in recent years on object detection due to the use of convolutional neural networks (CNNs). Modern object detectors based on these networks — such as Faster R-CNN [31], R-FCN [6], Multibox [40], SSD [26] and YOLO [29] — are now good enough to be deployed in consumer products (e.g., Google Photos, Pinterest Visual Search) and some have been shown to be fast enough to be run on mobile devices.

近几年目标检测有很多进展，这主要是由于使用了卷积神经网络(CNNs)。基于这些网络的现代目标检测器，比如Faster R-CNN [31], R-FCN [6], Multibox [40], SSD [26] and YOLO [29]，现在已经足够好，可以部署在用户产品中（比如，Google Photos, Pinterest Visual Search），一些速度已经很快，可以运行在移动设备中。

However, it can be difficult for practitioners to decide what architecture is best suited to their application. Standard accuracy metrics, such as mean average precision (mAP), do not tell the entire story, since for real deployments of computer vision systems, running time and memory usage are also critical. For example, mobile devices often require a small memory footprint, and self driving cars require real time performance. Server-side production systems, like those used in Google, Facebook or Snapchat, have more leeway to optimize for accuracy, but are still subject to throughput constraints. While the methods that win competitions, such as the COCO challenge [25], are optimized for accuracy, they often rely on model ensembling and multicrop methods which are too slow for practical usage.

但是，实践者可能很难决定哪种架构最适合他们的应用。标准的准确率度量标准，如mAP，没有给出算法的全貌，因为对于计算机视觉系统的真实部署，运行时间和内存使用率也非常关键。比如，移动设备需要内存使用量小，而自动驾驶汽车需要实时的性能。服务器端的生产系统，比如那些用于Google、Facebook或Snapchat的，有更多的余地来为准确率做优化，但仍然要看吞吐量的限制。那些赢得竞赛的方法是为准确率优化的，比如COCO挑战赛[25]，他们经常依靠模型集成和多剪切块方法，速度太慢，不能实用。

Unfortunately, only a small subset of papers (e.g., R-FCN [6], SSD [26] YOLO [29]) discuss running time in any detail. Furthermore, these papers typically only state that they achieve some frame-rate, but do not give a full picture of the speed/accuracy trade-off, which depends on many other factors, such as which feature extractor is used, input image sizes, etc.

不幸的是，只有一小部分文章（如R-FCN [6], SSD [26] YOLO [29]）在细节上讨论运行时的情况。而且，这些文章一般只说明他们取得了多少FPS，但没有给出速度/准确率折中的全貌，这与很多其他因素有关，比如使用了哪个特征提取器，输入图像大小，等等。

In this paper, we seek to explore the speed/accuracy trade-off of modern detection systems in an exhaustive and fair way. While this has been studied for full image classification( (e.g., [3]), detection models tend to be significantly more complex. We primarily investigate single-model/single-pass detectors, by which we mean models that do not use ensembling, multi-crop methods, or other “tricks” such as horizontal flipping. In other words, we only pass a single image through a single network. For simplicity (and because it is more important for users of this technology), we focus only on test-time performance and not on how long these models take to train.

在本文中，我们详尽且公平的研究了现代检测系统中速度和准确率的折中方法。在整幅图像分类中，这个问题已经被研究过（如[3]），但检测模型要复杂的多。我们主要研究了单模型/单发的检测器，意思是，我们只研究不使用集成方法、多剪切块方法或其他技巧如水平翻转的模型。换句话说，我们只将单幅图像送入单个网络。简单起见（而且因为这种技术对于用户更重要），我们只聚焦在测试时的性能，并不关注这些模型训练时花多少时间。

Though it is impractical to compare every recently proposed detection system, we are fortunate that many of the leading state of the art approaches have converged on a common methodology (at least at a high level). This has allowed us to implement and compare a large number of detection systems in a unified manner. In particular, we have created implementations of the Faster R-CNN, R-FCN and SSD meta-architectures, which at a high level consist of a single convolutional network, trained with a mixed regression and classification objective, and use sliding window style predictions.

虽然比较近期提出的每个检测系统不太现实，但幸运的是，很多领先的最好方法都汇聚到了一个通用方法中（至少在高层次上是这样）。这使我们可以用一种统一的方式实现并比较很多检测系统。特别是，我们已经实现了Faster R-CNN, R-FCN和SSD元架构，在高层次上，这个架构包含单个卷积网络，用回归和分类的混合目标函数进行了训练，并使用滑窗样式的预测。

To summarize, our main contributions are as follows: 总结一下，我们的主要贡献如下：

- We provide a concise survey of modern convolutional detection systems, and describe how the leading ones follow very similar designs.
- 我们给出了现代卷积检测系统的简单调查，并说明了，这些最好的系统是怎样遵循类似的设计的。
- We describe our flexible and unified implementation of three meta-architectures (Faster R-CNN, R-FCN and SSD) in Tensorflow which we use to do extensive experiments that trace the accuracy/speed tradeoff curve for different detection systems, varying meta-architecture, feature extractor, image resolution, etc.
- 我们在TensorFlow中给出了三种元架构灵活而统一的实现(Faster R-CNN, R-FCN, SSD)，并进行了广泛的试验，对不同检测系统、不同元架构、特征提取器和不同分辨率等情况，描绘出了准确率/速度的折中曲线。
- Our findings show that using fewer proposals for Faster R-CNN can speed it up significantly without a big loss in accuracy, making it competitive with its faster cousins, SSD and R-FCN. We show that SSDs performance is less sensitive to the quality of the feature extractor than Faster R-CNN and R-FCN. And we identify sweet spots on the accuracy/speed trade-off curve where gains in accuracy are only possible by sacrificing speed (within the family of detectors presented here).
- 我们的发现证明了使用更小的候选可以明显加速Faster R-CNN，而且准确率损失并不大，使其与其他模型具有竞争力，如SSD和R-FCN。我们证明了SSD的性能对特征提取器的敏感性，并没有Faster R-CNN和R-FCN大。我们还找出了准确率/速度折中曲线上的理想点，在这些点的情况中，准确率的提升只有牺牲速度才可以得到（只对这里给出的检测器家族来说）。
- Several of the meta-architecture and feature-extractor combinations that we report have never appeared before in literature. We discuss how we used some of these novel combinations to train the winning entry of the 2016 COCO object detection challenge.
- 我们给出的几种元架构和特征提取器组合，在之前的文献中从未出现。我们讨论了怎样使用这些新颖的组合训练并赢得2016 COCO目标检测挑战的。

## 2. Meta-architectures 元架构

Neural nets have become the leading method for high quality object detection in recent years. In this section we survey some of the highlights of this literature. The R-CNN paper by Girshick et al. [11] was among the first modern incarnations of convolutional network based detection. Inspired by recent successes on image classification [20], the R-CNN method took the straightforward approach of cropping externally computed box proposals out of an input image and running a neural net classifier on these crops. This approach can be expensive however because many crops are necessary, leading to significant duplicated computation from overlapping crops. Fast R-CNN [10] alleviated this problem by pushing the entire image once through a feature extractor then cropping from an intermediate layer so that crops share the computation load of feature extraction.

近年来神经网络已经成了高质量目标检测系统的先进方法。本节中我们研究了这些文献中的亮点。Girshick等[11]等人提出的R-CNN是第一个基于卷积网络的检测系统。受到图像分类[20]中成功进展的启发，R-CNN方法采用了直接的方法，将外部算法计算出的输入图像候选框，剪切出来送入神经网络分类器。因为需要很多剪切块，导致在重叠剪切块上明显的重复计算，所以这种方法计算量很大。Fast R-CNN[10]将整幅图像一次性送入特征提取器，然后从中间层中剪切，所以这些剪切共享了特征提取的计算量，这样缓解了问题。

While both R-CNN and Fast R-CNN relied on an external proposal generator, recent works have shown that it is possible to generate box proposals using neural networks as well [41, 40, 8, 31]. In these works, it is typical to have a collection of boxes overlaid on the image at different spatial locations, scales and aspect ratios that act as “anchors” (sometimes called “priors” or “default boxes”). A model is then trained to make two predictions for each anchor: (1) a discrete class prediction for each anchor, and (2) a continuous prediction of an offset by which the anchor needs to be shifted to fit the groundtruth bounding box.

R-CNN和Fast R-CNN都依靠外部的候选生成算法，但近期的工作表明，可以使用神经网络生成候选框[41,40,8,31]。在这些工作中，一般都有在图像的不同位置、不同尺度上不同纵横比的框的集合，称之为“锚框”（有时也称为“先验框”或“默认框”）。训练的模型对每个锚框进行两个预测：(1)每个锚框预测一个类别，，(2)预测一个连续的偏移，锚框通过这个偏移来适配真值边界框。

Papers that follow this anchors methodology then minimize a combined classification and regression loss that we now describe. For each anchor a, we first find the best matching groundtruth box b (if one exists). If such a match can be found, we call a a “positive anchor”, and assign it (1) a class label $y_a ∈ {1...K}$ and (2) a vector encoding of box b with respect to anchor a (called the box encoding $φ(b_a ;a)$). If no match is found, we call a a “negative anchor” and we set the class label to be $y_a = 0$. If for the anchor a we predict box encoding $f_{loc}(I;a,θ)$ and corresponding class $f_{cls}(I;a,θ)$, where I is the image and θ the model parameters, then the loss for a is measured as a weighted sum of a location-based loss and a classification loss:

使用这种锚框技术的文章，构建了一种综合了分类和回归的损失函数，并进行最小化，下面我们给出这个损失。对于每个锚框a，首先我们找到最匹配的真值框b（如果存在的话）。如果找到了这种匹配，我们称之为“正锚框”，并为之指定(1)一个类别标签$y_a ∈ {1...K}$，和(2)锚框a对应的编码向量b（称为框编码$φ(b_a ;a)$）。如果没有找到匹配项，我们称a为一个“负锚框”，并设其类别标签为$y_a = 0$。如果对于锚框a，我们预测的框位置编码为$f_{loc}(I;a,θ)$，对应的类别为$f_{cls}(I;a,θ)$，其中I是图像，θ为模型参数，那么对于a的损失就是位置损失和分类损失的加权之和：

$$L(a,I;θ) = α · 1[a \space is \space positive] · l_{loc}(φ(b_a ;a) − f_{loc}(I;a,θ)) + β · l_{cls} (y_a ,f_{cls}(I;a,θ))$$(1)

where α,β are weights balancing localization and classification losses. To train the model, Equation 1 is averaged over anchors and minimized with respect to parameters θ.

其中α,β是平衡定位损失和分类损失的权重。为训练这个模型，式1在所有锚框上平均，并求使其达到最小值的参数θ。

The choice of anchors has significant implications both for accuracy and computation. In the (first) Multibox paper [8], these anchors (called “box priors” by the authors) were generated by clustering groundtruth boxes in the dataset. In more recent works, anchors are generated by tiling a collection of boxes at different scales and aspect ratios regularly across the image. The advantage of having a regular grid of anchors is that predictions for these boxes can be written as tiled predictors on the image with shared parameters (i.e., convolutions) and are reminiscent of traditional sliding window methods, e.g. [44]. The Faster R-CNN [31] paper and the (second) Multibox paper [40] (which called these tiled anchors “convolutional priors”) were the first papers to take this new approach.

锚框的选择对于准确率和计算量都有重要的含义。在（第一篇）MultiBox文章[8]中，这些锚框（作者称之为“先验框”）通过对数据集内的真值框进行聚类得到。在最近的文章中，在图像中的不同位置、不同尺度中堆叠不同纵横比的框，这样生成锚框。锚框的规则网格分布，其优势是这些框的预测可以作为图像中的堆叠预测器，共享参数（即卷积的参数），并让人联想起传统的滑窗方法，如[44]。论文Faster R-CNN[31]和（第二篇）Multibox[40]（称这些堆叠的锚框为“卷积先验”）是首先使用这种新方法的。

### 2.1. Meta-architectures 元架构

In our paper we focus primarily on three recent (meta)-architectures: SSD (Single Shot Multibox Detector [26]), Faster R-CNN [31] and R-FCN (Region-based Fully Convolutional Networks [6]). While these papers were originally presented with a particular feature extractor (e.g., VGG, Resnet, etc), we now review these three methods, decoupling the choice of meta-architecture from feature extractor so that conceptually, any feature extractor can be used with SSD, Faster R-CNN or R-FCN.

本文我们主要研究三种近期提出的元架构：SSD[26], Faster R-CNN[31]和R-FCN[6]。这些文章提出的时候，都有特定的特征提取器（如VGG，ResNet等），我们现在回顾这些方法的时候，将元架构与特征提取器分离开来，所以从概念上来说，任何特征提取器都可以用于SSD, Faster R-CNN或R-FCN上。

#### 2.1.1 Single Shot Detector (SSD).

Though the SSD paper was published only recently (Liu et al., [26]), we use the term SSD to refer broadly to architectures that use a single feed-forward convolutional network to directly predict classes and anchor offsets without requiring a second stage per-proposal classification operation (Figure 1a). Under this definition, the SSD meta-architecture has been explored in a number of precursors to [26]. Both Multibox and the Region Proposal Network (RPN) stage of Faster R-CNN [40, 31] use this approach to predict class-agnostic box proposals. [33, 29, 30, 9] use SSD-like architectures to predict final (1 of K) class labels. And Poirson et al., [28] extended this idea to predict boxes, classes and pose.

虽然SSD论文只是最近才发表[26]，我们使用SSD来指代一类广泛的结构，即使用单个前向卷积网络来直接预测类别和锚框偏移，而不需要另一个阶段进行每个候选框的分类操作（见图1a）。在这种定义下，SSD元架构在[26]之前就有一些研究了。Multibox和Faster R-CNN的RPN[40,31]使用这种方法来预测与类别无关的候选框。[33,29,30,9]使用类似SSD的架构来预测最终类别标签（K类种的一个）。Poirson等人[28]将这种思想延申到预测边界框、类别和姿态。

#### 2.1.2 Faster R-CNN.

In the Faster R-CNN setting, detection happens in two stages (Figure 1b). In the first stage, called the region proposal network (RPN), images are processed by a feature extractor (e.g., VGG-16), and features at some selected intermediate level (e.g., “conv5”) are used to predict class-agnostic box proposals. The loss function for this first stage takes the form of Equation 1 using a grid of anchors tiled in space, scale and aspect ratio.

在Faster R-CNN的设置种，检测是两阶段的（见图1b）。在第一阶段，称为RPN，图像经过特征提取器处理（如VGG-16），选定了几个中间层（如conv5），器特征用于预测类别无关的候选框。这一阶段的损失函数就是式1的形式，使用了以位置、尺度和纵横比的方式堆叠的锚框网格。

In the second stage, these (typically 300) box proposals are used to crop features from the same intermediate feature map which are subsequently fed to the remainder of the feature extractor (e.g., “fc6” followed by “fc7”) in order to predict a class and class-specific box refinement for each proposal. The loss function for this second stage box classifier also takes the form of Equation 1 using the proposals generated from the RPN as anchors. Notably, one does not crop proposals directly from the image and re-run crops through the feature extractor, which would be duplicated computation. However there is part of the computation that must be run once per region, and thus the running time depends on the number of regions proposed by the RPN.

在第二阶段，这些候选框（一般为300）用于从相同的中间层特征图中剪切出特征，然后送入特征提取器的剩余部分（如fc6后面接着fc7），以预测类别和对每个候选框进行优化提炼的特定类别的框。第二阶段框分类器的损失函数也是式1的形式，使用RPN生成的候选作为锚框。尤其是，没有直接从图像中剪切出候选，然后送入特征提取器重新运行，这省去了很多重复计算。但是，有一部分计算必须每个区域都运行一次，所以运行时间取决于RPN生成的候选区域数量。

Since appearing in 2015, Faster R-CNN has been particularly influential, and has led to a number of follow-up works [2, 35, 34, 46, 13, 5, 19, 45, 24, 47] (including SSD and R-FCN). Notably, half of the submissions to the COCO object detection server as of November 2016 are reported to be based on the Faster R-CNN system in some way.

自从2015年提出后，Faster R-CNN影响力很大，引出了一系列随后的工作[2,35,34,46,13,5,19,45,24,47]（包括SSD和R-FCN）。尤其是，2016年11月提交给COCO目标识别服务器的算法中，一半都是在某种程度上基于Faster R-CNN的。

Figure 1: High level diagrams of the detection meta-architectures compared in this paper. (a) SSD. (b) Faster RCNN. (c) R-FCN.

### 2.2. R-FCN

While Faster R-CNN is an order of magnitude faster than Fast R-CNN, the fact that the region-specific component must be applied several hundred times per image led Dai et al. [6] to propose the R-FCN (Region-based Fully Convolutional Networks) method which is like Faster R-CNN, but instead of cropping features from the same layer where region proposals are predicted, crops are taken from the last layer of features prior to prediction (Figure 1c). This approach of pushing cropping to the last layer minimizes the amount of per-region computation that must be done. Dai et al. argue that the object detection task needs localization representations that respect translation variance and thus propose a position-sensitive cropping mechanism that is used instead of the more standard ROI pooling operations used in [10, 31] and the differentiable crop mechanism of [5]. They show that the R-FCN model (using Resnet 101) could achieve comparable accuracy to Faster R-CNN often at faster running times. Recently, the R-FCN model was also adapted to do instance segmentation in the recent TA-FCN model [22], which won the 2016 COCO instance segmentation challenge.

Faster R-CNN比Fast R-CNN速度快了一个量级，但其中每幅图像区域特定的组件都必须运行几百次，这使Dai等人[6]提出了R-FCN。与Faster R-CNN很像，但Faster R-CNN是在预测候选区域的层中剪切出特征，而R-FCN是在预测之前的最后一层中剪切出特征（如图1c）。这种剪切最后一层的方法，将每个区域都要进行的运算的运算量最小化了。Dai等人认为目标检测需要与平移变换有关的定位表示，所以提出了一种区分位置的剪切机制，取代了[10,31]中使用的标准ROI pooling操作，和[5]中的可区分的剪切机制。他们表示，R-FCN模型（使用ResNet 101）可以取得与Faster R-CNN类似的准确率，但运算时间更少。最近，TA-FCN模型[22]将R-FCN模型经过调整用于实例分割，赢得了2016 COCO实例分割挑战。

## 3. Experimental setup 实验设置

The introduction of standard benchmarks such as Imagenet [32] and COCO [25] has made it easier in recent years to compare detection methods with respect to accuracy. However, when it comes to speed and memory, apples-to-apples comparisons have been harder to come by. Prior works have relied on different deep learning frameworks (e.g., DistBelief [7], Caffe [18], Torch [4]) and different hardware. Some papers have optimized for accuracy; others for speed. And finally, in some cases, metrics are reported using slightly different training sets (e.g., COCO training set vs. combined training+validation sets).

标准测试基准的引入，如ImageNet[32]和COCO[25]，使最近几年比较检测方法的准确率变得更容易。但是，当比较速度和内存使用时，同类型的比较更难一些。之前的工作依靠的是不同的深度学习框架，如DistBelief[7], Caffe[18], Torch[4]，以及不同的硬件。一些文章针对准确率进行了优化，另一些针对的是速度优化。最后，在一些情况中，使用了不同的训练集给出了度量标准（如COCO训练集 vs. 训练验证集）。

In order to better perform apples-to-apples comparisons, we have created a detection platform in Tensorflow [1] and have recreated training pipelines for SSD, Faster R-CNN and R-FCN meta-architectures on this platform. Having a unified framework has allowed us to easily swap feature extractor architectures, loss functions, and having it in Tensorflow allows for easy portability to diverse platforms for deployment. In the following we discuss ways to configure model architecture, loss function and input on our platform — knobs that can be used to trade speed and accuracy.

为更好的进行同类型的对比，我们在TensorFlow中创建了一个检测平台，在这个平台上重建了SSD、Faster R-CNN和R-FCN元架构的训练过程。有一个统一的框架使我们更容易替换特征提取器架构，损失函数，在TensorFlow中实现使我们在各种平台上都容易移植和部署。下面我们就讨论在我们的平台上配置这些模型架构、损失函数和输入数据的方法，这些都是可以用于平衡速度和准确率的部分组件。

### 3.1. Architectural configuration 架构配置

#### 3.1.1 Feature extractors. 特征提取器

In all of the meta-architectures, we first apply a convolutional feature extractor to the input image to obtain high-level features. The choice of feature extractor is crucial as the number of parameters and types of layers directly affect memory, speed, and performance of the detector. We have selected six representative feature extractors to compare in this paper and, with the exception of MobileNet [14], all have open source Tensorflow implementations and have had sizeable influence on the vision community.

在所有的元架构中，我们首先都会将输入图像送入一个卷积特征提取器中，来得到高层特征。特征提取器的选择是非常关键的，因为参数数量和层的类型都直接影响到了检测器的内存使用、速度和性能。我们选择了6种有代表性的特征提取器在本文种进行比较，除了MobileNet[14]，都有开源的TensorFlow实现，它们都在计算机视觉种有很大的影响。

In more detail, we consider the following six feature extractors. We use VGG-16 [37] and Resnet-101 [13], both of which have won many competitions such as ILSVRC and COCO 2015 (classification, detection and segmentation). We also use Inception v2 [16], which set the state of the art in the ILSVRC2014 classification and detection challenges, as well as its successor Inception v3 [42]. Both of the Inception networks employed ‘Inception units’ which made it possible to increase the depth and width of a network without increasing its computational budget. Recently, Szegedy et al. [38] proposed Inception Resnet(v2), which combines the optimization benefits conferred by residual connections with the computation efficiency of Inception units. Finally, we compare against the new MobileNet network[14], which has been shown to achieve VGG-16 level accuracy on Imagenet with only 1/30 of the computational cost and model size. MobileNet is designed for efficient inference in various mobile vision applications. Its building blocks are depthwise separable convolutions which factorize a standard convolution into a depthwise convolution and a 1 × 1 convolution, effectively reducing both computational cost and number of parameters.

详细来说，我们使用了以下6种特征提取器。我们使用了VGG-16[37]和ResNet-101[13]，两个都赢得了很多比赛，比如ILSVRC和COCO 2015（分类、检测和分割）。我们还使用了Inception v2[16]，这是ILSVRC2014分类和检测挑战赛种最好的模型，以及Inception v3[42]。两种Inception网络都使用了Inception单元，这使网络在不增加计算预算的情况下，可以增加网络深度和宽度。最近，Szegedy等[38]提出了Inception ResNet v2，结合了残差连接的高度优化性和Inception单元的计算高效性。最后，我们还与新提出的MobileNet[14]进行了比较，这种网络在ImageNet上得到了VGG-16级别的准确度，但计算量和模型大小只有1/30。MobileNet设计用于各种移动视觉应用的高效推理，其基本组成模块是depthwise separable卷积，它将标准卷积分解成了一个depthwise卷积和1×1卷积，有效的降低了计算量和参数数量。

For each feature extractor, there are choices to be made in order to use it within a meta-architecture. For both Faster R-CNN and R-FCN, one must choose which layer to use for predicting region proposals. In our experiments, we use the choices laid out in the original papers when possible. For example, we use the ‘conv5’ layer from VGG-16 [31] and the last layer of conv_4_x layers in Resnet-101 [13]. For other feature extractors, we have made analogous choices. See supplementary materials for more details.

对每个特征提取器，都需要作出一些选择以用于元架构种。对于Faster R-CNN和R-FCN，必须选择用哪些层来预测候选区域。在我们的试验种，如果可能我们就使用原始论文种的选择。比如，我们使用VGG-16[31]的conv5层和ResNet-101种conv_4_x层的最后一层[13]。对于其他特征提取器，我们也做出类似的选择。详见补充资料。

Liu et al. [26] showed that in the SSD setting, using multiple feature maps to make location and confidence predictions at multiple scales is critical for good performance. For VGG feature extractors, they used conv4_3, fc7 (converted to a convolution layer), as well as a sequence of added layers. In our experiments, we follow their methodology closely, always selecting the topmost convolutional feature map and a higher resolution feature map at a lower level, then adding a sequence of convolutional layers with spatial resolution decaying by a factor of 2 with each additional layer used for prediction. However unlike [26], we use batch normalization in all additional layers.

Liu等[26]证明了在SSD的设置下，使用多个特征图来在多个尺度进行位置和置信度预测是取得好性能的关键。对于VGG特征提取器，使用了conv4_3, fc7（转化成一个卷积层），以及新增加了几个层。在我们的试验中，我们遵循这种方法，选择最上层的卷积图和一个更低层的高分辨率特征图，然后增加一系列卷积层，每增加一层特征图分辨率就除以2，将这些特征图用于预测。但是与[26]不同的是，我们在所有新加的层中都使用了批归一化。

For comparison, feature extractors used in previous works are shown in Table 1. In this work, we evaluate all combinations of meta-architectures and feature extractors, most of which are novel. Notably, Inception networks have never been used in Faster R-CNN frameworks and until recently were not open sourced [36]. Inception Resnet (v2) and MobileNet have not appeared in the detection literature to date.

为进行比较，之前工作中使用的特征提取器在表1中进行了比较。在本文中，我们评估所有元架构和特征提取器的组合，其中很多都是新的。尤其是，Inception网络一直没有在Faster R-CNN框架中使用过，直到最近也没有开源[36]。Inception ResNet(v2)和MobileNet到目前为止，还没有在检测文献中使用过。

Table 1: Convolutional detection models that use one of the meta-architectures described in Section 2. Boxes are encoded with respect to a matching anchor a via a function φ (Equation 1), where [$x_0, y_0, x_1, y_1$] are min/max coordinates of a box, $x_c, y_c$ are its center coordinates, and w,h its width and height. In some cases, $w_a, h_a$, width and height of the matching anchor are also used. Notes: (1) We include an early arXiv version of [26], which used a different configuration from that published at ECCV 2016; (2) [29] uses a fast feature extractor described as being inspired by GoogLeNet [39], which we do not compare to; (3) YOLO matches a groundtruth box to an anchor if its center falls inside the anchor (we refer to this as BoxCenter).

表1：使用第2节中描述的元结构的卷积检测模型。边界框是和相关的匹配锚框a通过函数φ进行编码的（见式1），其中[$x_0, y_0, x_1, y_1$]是边界框的最大最小坐标值，$x_c, y_c$是其中心坐标，w,h是其宽和高。在一些情况中，也使用了匹配到的锚框的宽度和高度$w_a, h_a$。注：(1)我们使用[26]的一个较早的arXiv版本，与在ECCV2016发表的配置不同；(2)[29]使用一种快速特征提取器，是受到GoogLeNet[39]启发得到的，我们没有与之比较；(3)YOLO匹配真值框与锚框的条件是，真值框的中心在锚框中（我们称之为BoxCenter）。

Paper | Meta-architecture | Feature Extractor | Matching | Box Encoding $φ(b_a, a$) | Location Loss functions
--- | --- | --- | --- | --- | ---
Szegedy et al. [40] | SSD | InceptionV3 | Bipartite | [$x_0, y_0, x_1, y_1$] | $L_2$
Redmon et al. [29] | SSD | Custom (GoogLeNet inspired) | Box Center | [$x_c, y_c, \sqrt w, \sqrt h$] | $L_2$
Ren et al. [31] | Faster R-CNN | VGG | Argmax | [$x_c/w_a, y_c/h_a, logw, logh$] | Smooth$L_1$
He et al. [13] | Faster R-CNN | ResNet-101 | Argmax | [$x_c/w_a, y_c/h_a, logw, logh$] | Smooth$L_1$
Liu et al. [26] (v1) | SSD | InceptionV3 | Argmax | [$x_0, y_0, x_1, y_1$] | $L_2$
Liu et al. [26] (v2, v3) | SSD | VGG | Argmax | [$x_c/w_a, y_c/h_a, logw, logh$] | Smooth$L_1$
Dai et al [6] | R-FCN | ResNet-101 | Argmax | [$x_c/w_a, y_c/h_a, logw, logh$] | Smooth$L_1$

#### 3.1.2 Number of proposals. 候选数量

For Faster R-CNN and R-FCN, we can also choose the number of region proposals to be sent to the box classifier at test time. Typically, this number is 300 in both settings, but an easy way to save computation is to send fewer boxes potentially at the risk of reducing recall. In our experiments, we vary this number of proposals between 10 and 300 in order to explore this trade-off.

对Faster R—CNN和R-FCN来说，我们也可以选择候选区域的数量，然后在测试时送入框分类器。一般来说，这个数量是300，一种降低计算量的简单方法就是送入更少的框进行分类，但这有降低召回率的风险。在我们的试验中，我们测试了10-300不同数量的候选区域，以确定其折中关系。

#### 3.1.3 Output stride settings for Resnet and Inception Resnet.

Our implementation of Resnet-101 is slightly modified from the original to have an effective output stride of 16 instead of 32; we achieve this by modifying the conv5_1 layer to have stride 1 instead of 2 (and compensating for reduced stride by using atrous convolutions in further layers) as in [6]. For Faster R-CNN and R-FCN, in addition to the default stride of 16, we also experiment with a (more expensive) stride 8 Resnet-101 in which the conv4_1 block is additionally modified to have stride 1. Likewise, we experiment with stride 16 and stride 8 versions of the Inception Resnet network. We find that using stride 8 instead of 16 improves the mAP by a factor of 5%(i.e., (map8 - map16) / map16 = 0.05), but increased running time by a factor of 63%.

我们实现的ResNet-101与原版略有不同，我们的输出步长为16而不是32；我们修改了conv5_1层，将其步长由2改为1，得到了输出步长的改变，就像[6]中一样（为补偿减小的步长，在后面的层中使用了atrous卷积）。对于Faster R-CNN和R-FCN，在默认的步长16之外，我们还试验了（计算量更大的）步长8的ResNet-101，其中conv4_1模块的步长改为了1。类似的，我们试验了步长为16和8的Inception ResNet网络。我们发现，使用步长8（比使用步长16）使mAP改进了5%（即，(mAP8-mAP16)/mAP16=0.05），但计算时间增加了63%。

### 3.2. Loss function configuration 损失函数配置

Beyond selecting a feature extractor, there are choices in configuring the loss function (Equation 1) which can impact training stability and final performance. Here we describe the choices that we have made in our experiments and Table 1 again compares how similar loss functions are configured in other works.

除了选择特征提取器，还有配置损失函数（式1）的选项，这可以影响训练稳定性和最终性能。这里我们给出试验中的选项，表1比较了类似的损失函数是怎样在其他工作中配置的。

#### 3.2.1 Matching.匹配

Determining classification and regression targets for each anchor requires matching anchors to groundtruth instances. Common approaches include greedy bipartite matching (e.g., based on Jaccard overlap) or many-to-one matching strategies in which bipartite-ness is not required, but matchings are discarded if Jaccard overlap between an anchor and groundtruth is too low. We refer to these strategies as Bipartite or Argmax, respectively. In our experiments we use Argmax matching throughout with thresholds set as suggested in the original paper for each meta-architecture. After matching, there is typically a sampling procedure designed to bring the number of positive anchors and negative anchors to some desired ratio. In our experiments, we also fix these ratios to be those recommended by the paper for each meta-architecture.

对每个锚框确定分类和回归的目标，需要将锚框与真值实例进行匹配。一般的方法包括，贪婪双边匹配（如基于交并比的），或多对一匹配策略，其中不需要双边性，但是如果锚框和真值框之间的交并比过低，那就抛弃这种匹配。我们分别称这些策略为Bipartite或Argmax。在我们的试验中，我们通篇使用Argmax匹配，对每种元架构来说，阈值设定采用原论文推荐的值。匹配后，通常都有一个取样过程，其作用是将正锚框和负锚框的数量固定到期望的比率。在我们的试验中，我们对每种元架构也采用其论文中设定的比率。

#### 3.2.2 Box encoding.边界框编码

To encode a groundtruth box with respect to its matching anchor, we use the box encoding function $φ(b_a ;a) = [10 ·x_c/w_a, 10·y_c/h_a, 5·logw, 5·logh]$ (also used by [11, 10, 31, 26]). Note that the scalar multipliers 10 and 5 are typically used in all of these prior works, even if not explicitly mentioned.

为编码真值框与匹配的锚框，我们使用编码函数$φ(b_a ;a) = [10 ·x_c/w_a, 10·y_c/h_a, 5·logw, 5·logh]$（也在[11, 10, 31, 26]中使用）。注意标量乘子10和5在之前所有的工作中都这样使用，有的没有说明的也是。

#### 3.2.3 Location loss ($l_{loc}$).定位损失

Following [10, 31, 26], we use the Smooth L1 (or Huber [15]) loss function in all experiments.与[10,31,26]一样，我们在所有试验中使用Smooth L1 (or Huber [15])损失函数。

### 3.3. Input size configuration.输入大小配置

In Faster R-CNN and R-FCN, models are trained on images scaled to M pixels on the shorter edge whereas in SSD, images are always resized to a fixed shape M × M. We explore evaluating each model on downscaled images as a way to trade accuracy for speed. In particular, we have trained high and low-resolution versions of each model. In the “high-resolution” settings, we set M = 600, and in the “low-resolution” setting, we set M = 300. In both cases, this means that the SSD method processes fewer pixels on average than a Faster R-CNN or R-FCN model with all other variables held constant.

在Faster R-CNN和R-FCN中，输入图像重制为短边M像素，然后训练模型，而在SSD中，图像大小固定为M×M。我们在每个模型上都评估输入图像大小与速度的折中关系。特别的，我们对每个模型都训练了高分辨率版本和低分辨率版本。在高分辨率设置中，我们设M=600，在低分辨率设置中，M=300。在两种情况下，其他变量保持不变，这都意味着SSD方法处理的图像要比Faster R-CNN或R-FCN模型要少。

### 3.4. Training and hyperparameter tuning 训练和超参数调节

We jointly train all models end-to-end using asynchronous gradient updates on a distributed cluster [7]. For Faster R-CNN and R-FCN, we use SGD with momentum with batch sizes of 1 (due to these models being trained using different image sizes) and for SSD, we use RMSProp [43] with batch sizes of 32 (in a few exceptions we reduced the batch size for memory reasons). Finally we manually tune learning rate schedules individually for each feature extractor. For the model configurations that match works in literature ([31, 6, 13, 26]), we have reproduced or surpassed the reported mAP results. (In the case of SSD with VGG, we have reproduced the number reported in the ECCV version of the paper, but the most recent version on ArXiv uses an improved data augmentation scheme to obtain somewhat higher numbers, which we have not yet experimented with.)

我们对所有模型进行端到端的联合训练，在一个分布式集群中使用异步梯度更新[7]。对于Faster R-CNN和R-FCN，我们使用带有动量的SGD，batch size为1（因为这些模型使用不同大小的图像进行训练），对SSD，我们使用RMSProp[43]，batch size为32（有一些例外情况，我们根据内存情况减小batch size）。最后对于每种特征提取器，我们逐个手工调节学习率变化方案。对于文献中有的模型配置([31, 6, 13, 26])，我们复现或超过其给出的mAP结果。（在VGG-SSD的情况中，我们复现了ECCV版本论文的结果，但其在ArXiv上最新的版本使用了一种改进的数据扩充方案，得到了更好一些的结果，我们还没有进行这个试验）

Note that for Faster R-CNN and R-FCN, this end-to-end approach is slightly different from the 4-stage training procedure that is typically used. Additionally, instead of using the ROI Pooling layer and Position-sensitive ROI Pooling layers used by [31, 6], we use Tensorflow’s “crop and resize” operation which uses bilinear interpolation to resample part of an image onto a fixed sized grid. This is similar to the differentiable cropping mechanism of [5], the attention model of [12] as well as the Spatial Transformer Network[17]. However we disable backpropagation with respect to bounding box coordinates as we have found this to be unstable during training.

注意对于Faster R-CNN和R-FCN，这种端到端的方法与一般使用的4阶段训练过程略有不同。另外，我们没有使用[31,6]中的ROI pooling层和与位置有关的ROI pooling层，而是使用了TensorFlow中的“剪切并改变大小”操作，操作中使用了双线性插值来将图像的一部分重采样成为固定大小的网格。这与[5]中的可微分剪切机制、[12]中的注意力机制以及[17]的空域变换器网络类似。但是我们没有将与边界框坐标有关的量反向传播，因为我们发现这会使训练不稳定。

Our networks are trained on the COCO dataset, using all training images as well as a subset of validation images, holding out 8000 examples for validation. (We remark that this dataset is similar but slightly smaller than the trainval35k set that has been used in several papers, e.g., [2, 26]) Finally at test time, we post-process detections with non-max suppression using an IOU threshold of 0.6 and clip all boxes to the image window. To evaluate our final detections, we use the official COCO API [23], which measures mAP averaged over IOU thresholds in [0.5 : 0.05 : 0.95], amongst other metrics.

我们的网络在COCO数据集中训练，使用所有的训练图像和一部分验证图像，保留了8000幅图像进行验证。（我们说明，这个数据集与在很多文章[2,26]中使用的trainval35k集类似，但略小）最后在测试时，我们用非最大抑制来处理检测结果，使用的IOU阈值为0.6，剪切了所有的图像窗口的边界框。为评估我们最终的检测结果，我们使用了官方的COCO API[23]，在IOU阈值取[0.5 : 0.05 : 0.95]上平均所有mAP。

### 3.5. Benchmarking procedure 基准测试过程

To time our models, we use a machine with 32GB RAM, Intel Xeon E5-1650 v2 processor and an Nvidia GeForce GTX Titan X GPU card. Timings are reported on GPU for a batch size of one. The images used for timing are resized so that the smallest size is at least k and then cropped to k × k where k is either 300 or 600 based on the model. We average the timings over 500 images.

我们使用的机器内存为32G，处理器为Intel Xeon E5-1650 v2，GPU为Nvidia GeForce GTX Titan X。计时是在GPU上，batch size为1。图像大小改变为短边为k，然后剪切成k×k大小，其中k是300或600。我们在500幅图像上平均所计时间。

We include postprocessing in our timing (which includes non-max suppression and currently runs only on the CPU). Postprocessing can take up the bulk of the running time for the fastest models at ∼ 40ms and currently caps our maximum framerate to 25 frames per second. Among other things, this means that while our timing results are comparable amongst each other, they may not be directly comparable to other reported speeds in the literature. Other potential differences include hardware, software drivers, framework (Tensorflow in our case), and batch size (e.g., the Liu et al. [26] report timings using batch sizes of 8). Finally, we use tfprof [27] to measure the total memory demand of the models during inference; this gives a more platform independent measure of memory demand. We also average the memory measurements over three images.

我们计时的时候包括了后处理的过程（这包括非最大抑制，现在只在CPU上运行）。在最快的模型中，后处理可以占到大部分的运行时间，大约40ms，所以目前我们的最高帧率为25FPS。这意味着虽然我们的计时结果相互之间可以比较，但可能与其他文献中给出的速度不能直接比较。其他潜在的不同之处包括硬件，软件驱动，框架（我们使用TensorFlow），batch size（如Liu等[26]使用batch size 8来给出时间结果）。最后，我们使用tfprof[27]来度量模型推理时所需的内存；这给出了一个更加平台无关的结果。我们还对三幅图像所需的内存进行平均。

### 3.6. Model Details 模型细节

Table 2 summarizes the feature extractors that we use. All models are pretrained on ImageNet-CLS. We give details on how we train the object detectors using these feature extractors below.

表2总结了我们使用的特征提取器。所有模型都在ImageNet-CLS进行了预训练。使用这些特征提取器怎样训练目标检测器，我们下面给出其中的细节。

Table 2: Properties of the 6 feature extractors that we use. Top-1 accuracy is the classification accuracy on ImageNet.

Model | Top-1 accuracy | Num. Params.
--- | --- | ---
VGG-16 | 71.0 | 14,714,688
MobileNet | 71.1 | 3,191,072
Inception V2 | 73.9 | 10,173,112
ResNet-101 | 76.4 | 42,605,504
Inception V3 | 78.0 | 21,802,784
Inception Resnet V2 | 80.4 | 54,336,736

#### 3.6.1 Faster R-CNN

We follow the original implementation of Faster R-CNN [31] closely, but but use Tensorflow’s “crop and resize” operation instead of standard ROI pooling. Except for VGG, all the feature extractors use batch normalization after convolutional layers. We freeze the batch normalization parameters to be those estimated during ImageNet pretraining. We train Faster R-CNN with asynchronous SGD with momentum of 0.9. The initial learning rates depend on which feature extractor we used, as explained below. We reduce the learning rate by 10x after 900K iterations and another 10x after 1.2M iterations. 9 GPU workers are used during asynchronous training. Each GPU worker takes a single image per iteration; the minibatch size for RPN training is 256, while the minibatch size for box classifier training is 64.

我们基本采用了Faster R-CNN的原始实现，但是使用了TensorFlow的"crop and resize"操作，而没有使用标准的ROI pooling。除了VGG，所有特征提取器都在卷积层后使用批归一化。我们将批归一化的参数固定为在ImageNet预训练时估计得到的参数。我们用动量0.9的异步SGD训练Faster R-CNN。初始的学习速率取决于我们所使用的特征提取器，这在下面详解。我们在900k次迭代后将学习速率除以10，再训练1.2M次迭代后再除以10。异步训练中使用了9块GPU。每个GPU每次迭代处理一幅图像，RPN的minibatch大小为256，框分类器的minibatch大小为64。

- VGG [37]: We extract features from the “conv5” layer whose stride size is 16 pixels. Similar to [5], we crop and resize feature maps to 14x14 then maxpool to 7x7. The initial learning rate is 5e-4.
- VGG[37]：我们从conv5层中提取特征，其步长大小为16像素。与[5]类似，我们将特征图剪切并改变大小为14×14，同时maxpool到7×7。初始学习速率为5e-4。
- Resnet 101 [13]: We extract features from the last layer of the “conv4” block. When operating in atrous mode, the stride size is 8 pixels, otherwise it is 16 pixels. Feature maps are cropped and resized to 14x14 then maxpooled to 7x7. The initial learning rate is 3e-4.
- ResNet-101[13]：我们从conv4模块的最后一层提取特征。当再atrous模式下操作时，步长大小为8像素，否则是16像素。特征图剪切为14×14大小，然后maxpool到7×7大小。初始学习速率为3e-4。
- Inception V2 [16]: We extract features from the “Mixed_4e” layer whose stride size is 16 pixels. Feature maps are cropped and resized to 14x14. The initial learning rate is 2e-4.
- Inception V2[16]：我们从Mixed_4e层提取特征，其步长大小为16像素。特征图剪切后变换到14×14大小。初始学习速率为2e-4。
- Inception V3 [42]: We extract features from the “Mixed_6e” layer whose stride size is 16 pixels. Feature maps are cropped and resized to 17x17. The initial learning rate is 3e-4.
- Inception V3[42]：我们从Mixed_6e层中提取特征，其步长大小为16像素。特征图剪切然后改变大小到17×17。初始学习速率为3e-4。
- Inception Resnet [38]: We extract features the from “Mixed_6a” layer including its associated residual layers. When operating in atrous mode, the stride size is 8 pixels, otherwise is 16 pixels. Feature maps are cropped and resized to 17x17. The initial learning rate is 1e-3.
- Inception ResNet[38]：我们从Mixed_6a层中提取特征，包括其相关残差层。当再atrous模式操作时，步长大小为8像素，否则是16像素。特征图剪切然后改变到17×17大小。初始学习速率为1e-3。
- MobileNet [14]: We extract features from the “Conv2d_11” layer whose stride size is 16 pixels. Feature maps are cropped and resized to 14x14. The initial learning rate is 3e-3.
- MobileNet[14]：从conv2d_11层中提取特征，其步长大小为16像素。特征图剪切后改变到14×14大小。初始学习速率为3e-3。

#### 3.6.2 R-FCN

We follow the implementation of R-FCN[6] closely, but use Tensorflow’s “crop and resize” operation instead of ROI pooling to crop regions from the position-sensitive score maps. All feature extractors use batch normalization after convolutional layers. We freeze the batch normalization parameters to be those estimated during ImageNet pretraining. We train R-FCN with asynchronous SGD with momentum of 0.9. 9 GPU workers are used during asynchronous training. Each GPU worker takes a single image per iteration; the minibatch size for RPN training is 256. As of the time of this submission, we do not have R-FCN results for VGG or Inception V3 feature extractors.

我们基本采用了R-FCN[6]的原始实现，但使用了TensorFlow的"crop and resize"操作，而没有使用ROI pooling，来从位置敏感的分数图中剪切区域。所有特征提取器都在卷积层后使用批归一化。我们将批归一化的参数固定为在ImageNet上预训练时估计得到的值。我们用动量为0.9的异步SGD来训练R-FCN。异步训练的过程中使用了9块GPU。每个GPU每次迭代处理一幅图像；RPN的minibatch大小为256。本次提交我们没有使用VGG或Inception V3作为特征提取器实现R-FCN。

- Resnet 101 [13]: We extract features from “block3” layer. When operating in atrous mode, the stride size is 8 pixels, otherwise it is 16 pixels. Position-sensitive score maps are cropped with spatial bins of size 7x7 and resized to 21x21. We use online hard example mining to sample a minibatch of size 128 for training the box classifier. The initial learning rate is 3e-4. It is reduced by 10x after 1M steps and another 10x after 1.2M steps.
- ResNet-101[13]：我们从block3层中提取特征。当在atrous模式时，步长为8，否则为16。位置敏感的分数图剪切为7×7大小的空间块，然后改变大小到21×21。我们使用在线难分样本挖掘来训练框分类器，样本minibatch大小为128。初始学习速率为3e-4，1M迭代后除以10，再1.2M次迭代后再除以10。
-  Inception V2 [16]: We extract features from “Mixed_4e” layer whose stride size is 16 pixels. Position-sensitive score maps are cropped with spatial bins of size 3x3 and resized to 12x12. We use online hard example mining to sample a minibatch of size 128 for training the box classifier. The initial learning rate is 2e-4. It is reduced by 10x after 1.8M steps and another 10x after 2M steps.
- Inception V2[16]：我们从Mixed_4e层中提取特征，其步长为16。位置敏感的分数图剪切成3×3大小的空间块，然后改变大小为12×12。我们使用在线难分样本挖掘来训练框分类器，样本minibatch大小为128。初始学习率为2e-4，1.8M次迭代后除以10，再2M迭代后再除以10。
- Inception Resnet [38]: We extract features from “Mixed_6a” layer including its associated residual layers. When operating in atrous mode, the stride size is 8 pixels, otherwise it is 16 pixels. Position-sensitive score maps are cropped with spatial bins of size 7x7 and resized to 21x21. We use all proposals from RPN for box classifier training. The initial learning rate is 7e-4. It is reduced by 10x after 1M steps and another 10x after 1.2M steps.
- Inception ResNet[38]：我们从Mixed_6a层中提取特征，包括相关的残差层。当在atrous模式时，步长为8，否则步长为16。位置敏感的分数图剪切为7×7大小的空间块，然后改变大小为21×21。我们使用RPN生成的所有候选进行框分类器训练。初始学习速率为7e-4，1M迭代后除以10，再1.2M迭代后再除以10。
- MobileNet [14]: We extract features from “Conv2d_11” layer whose stride size is 16 pixels. Position-sensitive score maps are cropped with spatial bins of size 3x3 and resized to 12x12. We use online hard example mining to sample a minibatch of size 128 for training the box classifier. The initial learning rate is 2e-3. Learning rate is reduced by 10x after 1.6M steps and another 10x after 1.8M steps.
- MobileNet[14]：我们从conv2d_11层中提取特征，其步长为16。位置敏感的分数图剪切成3×3大小的空间块，然后改变大小到12×12。我们使用在线难分样本挖掘来训练框分类器，样本minibatch大小为128。初始学习速率为2e-3，1.6M次迭代后除以10，再1.8M次迭代后再除以10。

#### 3.6.3 SSD

As described in the main paper, we follow the methodology of [26] closely, generating anchors in the same way and selecting the topmost convolutional feature map and a higher resolution feature map at a lower level, then adding a sequence of convolutional layers with spatial resolution decaying by a factor of 2 with each additional layer used for prediction. The feature map selection for Resnet101 is slightly different, as described below.

就像在文章中讲的一样，我们基本采用[26]中的方法，用同样的方式生成锚框，选择最高层的卷积特征图和更低层的高分辨率特征图，然后增加一系列卷积层，每增加一层，就将分辨率缩小一半，这些特征图用于预测。ResNet-101的特征图选择略有不同，下面详述。

Unlike [26], we use batch normalization in all additional layers, and initialize weights with a truncated normal distribution with a standard deviation of σ = .03. With the exception of VGG, we also do not perform “layer normalization” (as suggested in [26]) as we found it not to be necessary for the other feature extractors. Finally, we employ distributed training with asynchronous SGD using 11 worker machines. Below we discuss the specifics for each feature extractor that we have considered. As of the time of this submission, we do not have SSD results for the Inception V3 feature extractor and we only have results for high resolution SSD models using the Resnet 101 and Inception V2 feature extractors.

与[26]不同的是，我们在所有增加的层里都使用了批归一化，使用截断正态分布随机初始化权重，标准差σ = .03。在VGG中，我们没有进行层归一化（[26]中也这样建议），在其他特征提取器中，我们也发现这没有必要。最后，我们使用11块GPU采用异步SGD的分布式训练。下面我们讨论每个特征提取器中的特定情况。在我们提交论文的时候，我们还没有Inception V3特征提取器的SSD结果，使用ResNet-101和Inception V2特征提取器的只有高分辨率SSD模型的结果。

- VGG [37]: Following the paper, we use conv4_3, and fc7 layers, appending five additional convolutional layers with decaying spatial resolution with depths 512,256, 256, 256, 256, respectively. We apply $L_2$ normalization to the conv4_3 layer, scaling the feature norm at each location in the feature map to a learnable scale, s, which is initialized to 20.0. During training, we use a base learning rate of $lr_{base} = .0003$, but use a warm-up learning rate scheme in which we first train with a learning rate of $0.8^2 ·lr_{base}$ for 10K iterations followed by $0.8 · lr_{base}$ for another 10K iterations.
- VGG[37]：按照文章中所说，我们使用conv4_3和fc7层，另外增加5个卷积层，每层分辨率递减，通道数分别为512，256，256，256，256。我们对conv4_3层使用$L_2$归一化，将特征图中每个位置中的特征范数数值变成一个可以学习的量，s，并初始化为20.0。在训练过程中，我们使用的基础学习速率为$lr_{base} = .0003$，但使用一种预热的学习率方案，首先使用学习速率$0.8^2 ·lr_{base}$进行10k迭代，然后使用$0.8 · lr_{base}$再进行10k次迭代。
- Resnet 101 [13]: We use the feature map from the last layer of the “conv4” block. When operating in atrous mode, the stride size is 8 pixels, otherwise it is 16 pixels. Five additional convolutional layers with decaying spatial resolution are appended, which have depths 512, 512, 256, 256, 128, respectively. We have experimented with including the feature map from the last layer of the “conv5” block. With “conv5” features, the mAP numbers are very similar, but the computational costs are higher. Therefore we choose to use the last layer of the “conv4” block. During training, a base learning rate of 3e-4 is used. We use a learning rate warm up strategy similar to the VGG one.
- ResNet-101[13]：我们使用conv4模块最后一层的特征图。当在atrous模式时，步长为8，否则为16。对于新增加的5个卷积层，分辨率逐渐减小，通道数分别为512，512，256，256，128。我们还对使用conv5模块的最后一层的特征图进行了试验，结果是mAP数值差不多，但计算量更大。所以我们选择使用conv4模块的最后一层。在训练过程中，基础学习率为3e-4。我们使用的预热学习速度策略与VGG类似。
- Inception V2 [16]: We use Mixed_4c and Mixed_5c, appending four additional convolutional layers with decaying resolution with depths 512, 256, 256, 128 respectively. We use ReLU6 as the non-linear activation function for each conv layer. During training, we use a base learning rate of 0.002, followed by learning rate decay of 0.95 every 800k steps.
- Inception V2[16]：我们使用Mixed_4c和Mixed_5c层，另外增加了4个卷积层，分辨率逐步降低，通道数分别为512，256，256，128。对每个卷积层，我们使用ReLU6作为非线性激活函数。在训练过程中，我们使用基础学习率0.002，每800k次迭代衰减0.95。
- Inception Resnet [38]: We use Mixed_6a and Conv2d_7b, appending three additional convolutional layers with decaying resolution with depths 512, 256, 128 respectively. We use ReLU as the non-linear activation function for each conv layer. During training, we use a base learning rate of 0.0005, followed by learning rate decay of 0.95 every 800k steps.
- Inception ResNet[38]：我们使用Mixed_6a和Conv2d-7b层，另外增加3个卷积层，分辨率逐步降低，通道数分别为512，256，128。对每个卷积层，我们使用ReLU作为非线性激活函数。在训练中，我们使用基础学习率0.0005，每800k次迭代衰减0.95。
- MobileNet [14]: We use conv_11 and conv_13, appending four additional convolutional layers with decaying resolution with depths 512, 256, 256, 128 respectively. The non-linear activation function we use is ReLU6 and both batch norm parameters β and γ are trained. During training, we use a base learning rate of 0.004, followed by learning rate decay of 0.95 every 800k steps.
- MobileNet[14]：我们使用conv_11和conv_13，另外增加4个卷积层，分辨率逐步衰减，通道数分别为512，256，256，128。使用的非线性激活函数为ReLU6，两个批归一化参数β和γ都进行了训练。训练过程中，基础学习率为0.004，每800k次迭代衰减0.95。

## 4. Results 结果

In this section we analyze the data that we have collected by training and benchmarking detectors, sweeping over model configurations as described in Section 3. Each such model configuration includes a choice of meta-architecture, feature extractor, stride (for Resnet and Inception Resnet) as well as input resolution and number of proposals (for Faster R-CNN and R-FCN).

在本节中，我们将第3节中不同模型不同配置的检测器都进行了训练和基准测试，对收集的数据进行了分析。每个模型配置都包括了元架构、特征提取器、步长（对ResNet和Inception ResNet来说）的选择，还有输入分辨率和候选数量（对Faster R-CNN和R-FCN）的选择。

For each such model configuration, we measure timings on GPU, memory demand, number of parameters and floating point operations as described below. We make the entire table of results available in the supplementary material, noting that as of the time of this submission, we have included 147 model configurations; models for a small subset of experimental configurations (namely some of the high resolution SSD models) have yet to converge, so we have for now omitted them from analysis.

对每个这样的模型配置，我们在GPU上衡量了运行时间，所需的内存，参数数量和浮点数操作，描述如下。我们在附加材料中给出了所有可用数据的整个表格，在我们这次提交的时候，我们囊括了147种模型配置；一小部分模型配置（即几种高分辨率SSD模型）还未收敛，所以我们在分析种忽略了它们。

### 4.1. Analyses 分析

#### 4.1.1 Accuracy vs time 准确率与时间

Figure 2 is a scatterplot visualizing the mAP of each of our model configurations, with colors representing feature extractors, and marker shapes representing meta-architecture. Running time per image ranges from tens of milliseconds to almost 1 second. Generally we observe that R-FCN and SSD models are faster on average while Faster R-CNN tends to lead to slower but more accurate models, requiring at least 100 ms per image. However, as we discuss below, Faster R-CNN models can be just as fast if we limit the number of regions proposed. We have also overlaid an imaginary “optimality frontier” representing points at which better accuracy can only be attained within this family of detectors by sacrificing speed. In the following, we highlight some of the key points along the optimality frontier as the best detectors to use and discuss the effect of the various model configuration options in isolation.

图2是每个模型配置的mAP的可视化散点图，其中颜色代表特征提取器，记号形状代表元架构。每幅图像的运行时间从几十ms到接近1秒。一般来说，我们观察到R-FCN和SSD模型平均更快一些，而Faster R-CNN更慢但更准确，每幅图像至少需要100ms。但是，就像我们下面讨论的，如果我们减少候选区域数量，Faster R-CNN也可以很快。我们我们还画了一条想象的最佳前沿，在这些点上要想在本族检测器种得到更好的准确率，必须牺牲速度。下面，我们强调最佳前沿上的一些关键点，这都是可以使用的最佳检测器，然后讨论不同模型配置选项的影响。

Figure 2: Accuracy vs time, with marker shapes indicating meta-architecture and colors indicating feature extractor. Each (meta-architecture, feature extractor) pair can correspond to multiple points on this plot due to changing input sizes, stride, etc.

#### 4.1.2 Critical points on the optimality frontier. 最佳前沿的关键点

(Fastest: SSD w/MobileNet): On the fastest end of this optimality frontier, we see that SSD models with Inception v2 and Mobilenet feature extractors are most accurate of the fastest models. Note that if we ignore postprocessing costs, Mobilenet seems to be roughly twice as fast as Inception v2 while being slightly worse in accuracy.

（最快的：SSD w/MobileNet）：在最佳前沿的最快的一端，我们看到采用InceptionV2和MobileNet特征提取器的SSD模型是最快的模型种非常准确的。注意我们忽略了后处理的代价，MobileNet比Inception v2大约快了2倍，但准确率略低。

(Sweet Spot: R-FCN w/Resnet or Faster R-CNN w/Resnet and only 50 proposals): There is an “elbow” in the middle of the optimality frontier occupied by R-FCN models using Residual Network feature extractors which seem to strike the best balance between speed and accuracy among our model configurations. As we discuss below, Faster R-CNN w/Resnet models can attain similar speeds if we limit the number of proposals to 50.

（最佳位置：R-FCN w/ResNet 或 Faster R-CNN w/ResNet，50候选）：在最佳前沿中间有一个肘部，是使用ResNet作为特征提取器的R-FCN模型，在速度和准确率的均衡上做到了最佳。我们下面会讨论到，Faster R-CNN w/ResNet模型如果将候选限制在50个，也可以做到类似的速度。

(Most Accurate: Faster R-CNN w/Inception Resnet at stride 8): Finally Faster R-CNN with dense output Inception Resnet models attain the best possible accuracy on our optimality frontier, achieving, to our knowledge, the state-of-the-art single model performance. However these models are slow, requiring nearly a second of processing time. The overall mAP numbers for these 5 models are shown in Table 3.

（最准确：Faster R-CNN w/Inception-ResNet，步长8）：最后密集输出的Inception-ResNet构成的Faster R-CNN模型在最佳前沿种达到了最佳准确率，取得了我们所知的最佳效果。但是这些模型很慢，需要接近1秒的处理时间。这5种模型的总体mAP如表3所示。

Table 3: Test-dev performance of the “critical” points along our optimality frontier.

Model summary | minival mAP | test-dev mAP
--- | --- | ---
(Fastest) SSD w/MobileNet (Low Resolution) | 19.3 | 18.8
(Fastest) SSD w/Inception V2 (Low Resolution) | 22 | 21.6
(Sweet Spot) Faster R-CNN w/Resnet 101, 100 Proposals | 32 | 31.9
(Sweet Spot) R-FCN w/Resnet 101, 300 Proposals | 30.4 | 30.3
(Most Accurate) Faster R-CNN w/Inception Resnet V2, 300 Proposals | 35.7 | 35.6

#### 4.1.3 The effect of the feature extractor.特征提取器的效果

Intuitively, stronger performance on classification should be positively correlated with stronger performance on COCO detection. To verify this, we investigate the relationship between overall mAP of different models and the Top-1 Imagenet classification accuracy attained by the pretrained feature extractor used to initialize each model. Figure 3 indicates that there is indeed an overall correlation between classification and detection performance. However this correlation appears to only be significant for Faster R-CNN and R-FCN while the performance of SSD appears to be less reliant on its feature extractor’s classification accuracy.

直觉上来说，分类性能更强，应当在COCO检测上取得更好的效果。为验证这个，我们研究了不同模型的总体mAP，和对应使用的预训练的特征提取器所取得Top-1 ImageNet分类准确率。图3表明，确实有分类和检测性能的总体关联。但是这种关联只对于Faster R-CNN和R-FCN比较明显，而SSD的性能似乎不太依赖于器特征提取器的分类准确率。

Figure 3: Accuracy of detector (mAP on COCO) vs accuracy of feature extractor (as measured by top-1 accuracy on ImageNet-CLS). To avoid crowding the plot, we show only the low resolution models.

#### 4.1.4 The effect of object size.目标大小的影响

Figure 4 shows performance for different models on different sizes of objects. Not surprisingly, all methods do much better on large objects. We also see that even though SSD models typically have (very) poor performance on small objects, they are competitive with Faster RCNN and R-FCN on large objects, even outperforming these meta-architectures for the faster and more lightweight feature extractors.

图4所示的是不同的目标大小对不同模型的性能的影响。所有的方法都对大目标效果好，这不让人惊讶。我们还看到，即使SSD模型在小目标上表现一般不好，但在大目标上和Faster R-CNN、R-FCN也有相近的性能，甚至在一些更快更轻量级的特征提取器上超过了这些元架构的性能。

Figure 4: Accuracy stratified by object size, meta-architecture and feature extractor, We fix the image resolution to 300.

#### 4.1.5 The effect of image size.图像大小的影响

It has been observed by other authors that input resolution can significantly impact detection accuracy. From our experiments, we observe that decreasing resolution by a factor of two in both dimensions consistently lowers accuracy (by 15.88% on average) but also reduces inference time by a relative factor of 27.4% on average.

其他作者已经观察到，图像分辨率可以显著影响检测准确率。从我们的试验来看，我们观察到，两个维度上都将分辨率降低一半，准确率全部都有所下降（平均下降15.88%），但推理时间也平均降低了27.4%。

One reason for this effect is that high resolution inputs allow for small objects to be resolved. Figure 5 compares detector performance on large objects against that on small objects, confirms that high resolution models lead to significantly better mAP results on small objects (by a factor of 2 in many cases) and somewhat better mAP results on large objects as well. We also see that strong performance on small objects implies strong performance on large objects in our models, (but not vice-versa as SSD models do well on large objects but not small).

这种效果的一个原因是高分辨率输入使小目标的问题得以解决。图5比较了大目标与小目标的检测器性能，确认了高分辨率模型在小目标上会得到更好的mAP结果，在大目标上mAP也有所提升。我们还看到，在小目标上表现好，意味着在大目标上表现也好，（但反过来不成立，SSD模型在大目标上表现很好，但小目标表现不好）。

Figure 5: Effect of image resolution

#### 4.1.6 The effect of the number of proposals.候选数量的效果

For Faster R-CNN and R-FCN, we can adjust the number of proposals computed by the region proposal network. The authors in both papers use 300 boxes, however, our experiments suggest that this number can be significantly reduced without harming mAP (by much). In some feature extractors where the “box classifier” portion of Faster R-CNN is expensive, this can lead to significant computational savings. Figure 6a visualizes this trade-off curve for Faster R-CNN models with high resolution inputs for different feature extractors. We see that Inception Resnet, which has 35.4% mAP with 300 proposals can still have surprisingly high accuracy (29% mAP) with only 10 proposals. The sweet spot is probably at 50 proposals, where we are able to obtain 96% of the accuracy of using 300 proposals while reducing running time by a factor of 3. While the computational savings are most pronounced for Inception Resnet, we see that similar tradeoffs hold for all feature extractors.

对于Faster R-CNN和R-FCN，我们可以调整RPN计算得到的候选数量。两篇文章的作者都使用了300框，但是，我们的试验表明，这个数量可以明显减少，而不太影响mAP。在一些特征提取器中，Faster R-CNN中的框分类器占比较大，这可以显著的节约计算量。图6a画出了高分辨率输入的Faster R-CNN模型，在不同的特征提取器下，候选框的数量与mAP的关系图。我们看到，Inception ResNet在300候选时mAP为35.4%，而在10个候选时，还是有很高的准确率(29% mAP)。最佳点大约是50个候选，这里可以达到300候选准确率的96%，运行时间可以降低3倍。对于Inception ResNet来说，计算量的减少最为显著，对于其他所有特征提取器，我们可以看到类似的折衷效果。

Figure 6b visualizes the same trade-off curves for R-FCN models and shows that the computational savings from using fewer proposals in the R-FCN setting are minimal — this is not surprising as the box classifier (the expensive part) is only run once per image. We see in fact that at 100 proposals, the speed and accuracy for Faster R-CNN models with ResNet becomes roughly comparable to that of equivalent R-FCN models which use 300 proposals in both mAP and GPU speed.

图6b是R-FCN的相同的折中曲线，这表明在R-FCN中减少候选并没有减少多少计算量，这并不令人惊讶，因为其框分类器（计算量大的部分）每幅图像只运行一次。我们看到在100个候选时，使用ResNet的Faster R-CNN的速度和准确率，和相应的R-FCN模型使用300候选时的mAP和GPU速度大致类似。

Figure 6: Effect of proposing increasing number of regions on mAP accuracy (solid lines) and GPU inference time (dotted). Surprisingly, for Faster R-CNN with Inception Resnet, we obtain 96% of the accuracy of using 300 proposals by using only 50 proposals, which reduces running time by a factor of 3.

#### 4.1.7 FLOPs analysis.

Figure 7 plots the GPU time for each model combination. However, this is very platform dependent. Counting FLOPs (multiply-adds) gives us a platform independent measure of computation, which may or may not be linear with respect to actual running times due to a number of issues such as caching, I/O, hardware optimization etc.

图7画出了每种模型组合的GPU时间。但是，这与平台关系很大。FLOPs数量（乘法加法数量）给出了平台无关的计算量度量，这可能与实际运行时间线性相关，也可能不是线性相关，有几个因素影响，比如缓存、I/O、硬件优化等等。

Figure 7: GPU time (milliseconds) for each model, for image resolution of 300.

Figures 8a and 8b plot the FLOP count against observed wallclock times on the GPU and CPU respectively. Interestingly, we observe in the GPU plot (Figure 8a) that each model has a different average ratio of flops to observed running time in milliseconds. For denser block models such as Resnet 101, FLOPs/GPU time is typically greater than 1, perhaps due to efficiency in caching. For Inception and Mobilenet models, this ratio is typically less than 1 — we conjecture that this could be that factorization reduces FLOPs, but adds more overhead in memory I/O or potentially that current GPU instructions (cuDNN) are more optimized for dense convolution.

图8a和8b给出了FLOPs数量与观察到的运行时间的关系，分GPU和CPU两种。有趣的是，我们在GPU图中（图8a）看到，每个模型的flops与观察到的运行时间的平均比率都不一样。对于更密集模块的模型，如ResNet-101，FLOPs/GPU时间一般大于1，可能是因为缓存的效率原因。对于Inception和MobileNet模型来说，这个比率一般小于1，我们推测这可能是因为分解减少了FLOPs，但增加了内存I/O的代价，或者可能是现在GPU指令(cuDNN)为密集卷积优化的更好。

#### 4.1.8 Memory analysis.内存分析

For memory benchmarking, we measure total usage rather than peak usage. Figures 10a, 10b plot memory usage against GPU and CPU wallclock times. Overall, we observe high correlation with running time with larger and more powerful feature extractors requiring much more memory. Figure 9 plots some of the same information in more detail, drilling down by meta-architecture and feature extractor selection. As with speed, Mobilenet is again the cheapest, requiring less than 1Gb (total) memory in almost all settings.

对于内存基准测试，我们衡量的是总计使用量，而不是峰值使用量。图10a, 10b画出了在CPU和GPU上的内存使用量。总体上来说，我们看到更大更强的特征提取器需要更多的内存。图9所示的有一些相同的信息，更详细，分元架构和特征提取器选择显示了信息。至于在速度上，MobileNet还是非常快速，在所有设置中需要的内存都不到1Gb。

Figure 9: Memory (Mb) usage for each model. Note that we measure total memory usage rather than peak memory usage. Moreover, we include all data points corresponding to the low-resolution models here. The error bars reflect variance in memory usage by using different numbers of proposals for the Faster R-CNN and R-FCN models (which leads to the seemingly considerable variance in the Faster-RCNN with Inception Resnet bar).

#### 4.1.9 Good localization at .75 IOU means good localization at all IOU thresholds.在IOU阈值为0.75时定位好意味着在所有阈值上都好

While slicing the data by object size leads to interesting insights, it is also worth nothing that slicing data by IOU threshold does not give much additional information. Figure 11 shows in fact that both mAP@.5 and mAP@.75 performances are almost perfectly linearly correlated with mAP@[.5:.95]. Thus detectors that have poor performance at the higher IOU thresholds always also show poor performance at the lower IOU thresholds. This being said, we also observe that mAP@.75 is slightly more tightly correlated with mAP@[.5:.95] (with $R^2$ > .99), so if we were to replace the standard COCO metric with mAP at a single IOU threshold, we would likely choose IOU=.75.

当按照目标大小分割数据时，我们看到有意思的结论。但按照IOU阈值分割数据没有给出更多信息。图11所示的是mAP@.5和mAP@.75的性能几乎是完全线性相关的，mAP@[.5:.95]也是。所以如果模型在更高的IOU阈值上表现不太好时，在更低的IOU阈值上表现也不会好。我们还观察到，mAP@.75与mAP@[.5:.95]的相关性更强一些（$R^2$ > .99），所以如果我们可以将标准COCO度量标准的mAP替换成一个mAP阈值的话，我们会选择IOU=.75。

Figure 11: Overall COCO mAP (@[.5:.95]) for all experiments plotted against corresponding mAP@.50IOU and mAP@.75IOU. It is unsurprising that these numbers are correlated, but it is interesting that they are almost perfectly correlated so for these models, it is never the case that a model has strong performance at 50% IOU but weak performance at 75% IOU.

### 4.2. State-of-the-art detection on COCO 目前在COCO上最好的检测结果

Finally, we briefly describe how we ensembled some of our models to achieve the current state of the art performance on the 2016 COCO object detection challenge. Our model attains 41.3% mAP@[.5, .95] on the COCO test set and is an ensemble of five Faster R-CNN models based on Resnet and Inception Resnet feature extractors. This outperforms the previous best result (37.1% mAP@[.5, .95]) by MSRA, which used an ensemble of three Resnet-101 models [13]. Table 4 summarizes the performance of our model and highlights how our model has improved on the state-of-the-art across all COCO metrics. Most notably, our model achieves a relative improvement of nearly 60% on small object recall over the previous best result. Even though this ensemble with state-of-the-art numbers could be viewed as
an extreme point on the speed/accuracy tradeoff curves (requires ∼50 end-to-end network evaluations per image), we have chosen to present this model in isolation since it is not comparable to the “single model” results that we focused on in the rest of the paper.

最后，我们简要介绍一下，我们是怎样将我们最好的一些模型集成起来，在2016 COCO目标检测挑战赛上得到目前最好的表现结果的。我们的模型在COCO测试集上得到了41.3% mAP@[.5, .95]，这是5个Faster R-CNN模型的集成，使用的特征提取器为ResNet和Inception ResNet。这超过了之前最好的结果MSRA(37.1% mAP@[.5, .95])，它使用了三个ResNet-101模型的集成[13]。表4总结了我们的模型的表现，强调了我们的模型在所有的COCO度量标准上是怎样改进了目前最好的结果。最值得注意的是，我们的模型在小目标召回率上比之前最好的结果有了60%的相对进步。即使这种得到了最好成绩的集成模型可以视作速度/准确率折中曲线的一个极端点（每幅图像大约需要50个端到端的网络评估），我们还是单独将其拿出来展示，因为与单模型的结果这是不可比的，本文还是主要集中在单模型。

Table 4: Performance on the 2016 COCO test-challenge dataset. AP and AR refer to (mean) average precision and average recall respectively. Our model achieves a relative improvement of nearly 60% on small objects recall over the previous state-of-the-art COCO detector.

| | AP | AP@.50IOU | AP@.75IOU | AP_small | AP_med | AP_large | AR@100 | AR_small | AR_med | AR_large
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
Ours | 0.413 | 0.62 | 0.45 | 0.231 | 0.436 | 0.547 | 0.604 | 0.424 | 0.641 | 0.748
MSRA2015 | 0.371 | 0.588 | 0.398 | 0.173 | 0.415 | 0.525 | 0.489 | 0.267 | 0.552 | 0.679
Trimps-Soushen | 0.359 | 0.58 | 0.383 | 0.158 | 0.407 | 0.509 | 0.497 | 0.269 | 0.557 | 0.683

To construct our ensemble, we selected a set of five models from our collection of Faster R-CNN models. Each of the models was based on Resnet and Inception Resnet feature extractors with varying output stride configurations, retrained using variations on the loss functions, and different random orderings of the training data. Models were selected greedily using their performance on a held-out validation set. However, in order to take advantage of models with complementary strengths, we also explicitly encourage diversity by pruning away models that are too similar to previously selected models (c.f., [21]). To do this, we computed the vector of average precision results across each COCO category for each model and declared two models to be too similar if their category-wise AP vectors had cosine distance greater than some threshold.

为构建我们的集成模型，我们选择了5种Faster R-CNN模型。每个模型都是基于ResNet和Inception ResNet特征提取器的，以及不同的输出步长配置，使用不同的损失函数重新训练，训练数据有着不同的随机次序。模型是在一个验证集上，根据其表现贪婪选取的。但是，为选择互补的模型，我们鼓励模型的多样性，将与之前选择的模型过于类似的去除掉[21]。为了达到这个目标，我们对每个模型都计算了所有COCO类别的平均精度结果的向量，如果两个模型的分类别AP向量的cosine距离大于某个阈值，就称之为过于相似。

Table 5 summarizes the final selected model specifications as well as their individual performance on COCO as single models. (Note that these numbers were computed on a held-out validation set and are not strictly comparable to the official COCO test-dev data results (though they are expected to be very close).) Ensembling these five models using the procedure described in [13] (Appendix A) and using multi-crop inference then yielded our final model. Note that we do not use multiscale training, horizontal flipping, box refinement, box voting, or global context which are sometimes used in the literature. Table 6 compares a single model’s performance against two ways of ensembling, and shows that (1) encouraging for diversity did help against a hand selected ensemble, and (2) ensembling and multicrop were responsible for almost 7 points of improvement over a single model.

表5列出了最终选择的模型的指标，以及其单个模型在COCO上的表现。（注意这些数据是在一个保留的验证集上计算的，与COCO官方的test-dev数据结果并不严格可比，但是它们确实应当是很接近的）用[13]种的方法集成这5种模型（附录A），并用多剪切块推理就可以得到我们的最终模型。注意我们没有使用多尺度训练，水平翻转，边界框优化，边界框投票，或全局上下文，这些有时候在文献中会用到。表6比较了单个模型的性能与两种方法的集成，这表明(1)鼓励模型的多样性确实由于手工选择模型集成，(2)模型集成和多剪切块方法比单个模型有接近7%的改进。

Table 5: Summary of single models that were automatically selected to be part of the diverse ensemble. Loss ratio refers to the multipliers α,β for location and classification losses, respectively.

AP | Feature Extractor | Output stride | loss ratio | Location loss function 
--- | --- | --- | --- | ---
32.93 | Resnet 101 | 8 | 3:1 | SmoothL1
33.3 | Resnet 101 | 8 | 1:1 | SmoothL1
34.75 | Inception Resnet (v2) | 16 | 1:1 | SmoothL1
35.0 | Inception Resnet (v2) | 16 | 2:1 | SmoothL1
35.64 | Inception Resnet (v2) | 8 | 1:1 | SmoothL1 + IOU

Table 6: Effects of ensembling and multicrop inference. Numbers reported on COCO test-dev dataset. Second row (hand selected ensemble) consists of 6 Faster RCNN models with 3 Resnet 101 (v1) and 3 Inception Resnet (v2) and the third row (diverse ensemble) is described in detail in Table 5.

| | AP | AP@.50IOU | AP@.75IOU | AP_small | AP_med | AP_large
--- | --- | --- | --- | --- | --- | ---
Faster RCNN with Inception Resnet (v2) | 0.347|  0.555 | 0.367 | 0.135 | 0.381 | 0.52
Hand selected Faster RCNN ensemble w/multicrop | 0.41 | 0.617 | 0.449 | 0.236 | 0.43 | 0.542
Diverse Faster RCNN ensemble w/multicrop | 0.416 | 0.619 | 0.454 | 0.239 | 0.435 | 0.549

### 4.3. Example detections

In Figures 12 to 17 we visualize detections on images from the COCO dataset, showing side-by-side comparisons of five of the detectors that lie on the “optimality frontier” of the speed-accuracy trade-off plot. To visualize, we select detections with score greater than a threshold and plot the top 20 detections in each image. We use a threshold of .5 for Faster R-CNN and R-FCN and .3 for SSD. These thresholds were hand-tuned for (subjective) visual attractiveness and not using rigorous criteria so we caution viewers from reading too much into the tea leaves from these visualizations. This being said, we see that across our examples, all of the detectors perform reasonably well on large objects — SSD only shows its weakness on small objects, missing some of the smaller kites and people in the first image as well as the smaller cups and bottles on the dining table in the last image.

图12-图17中，我们给出在COCO数据集一些图像上的检测结果，给出了在最佳前沿上的5种检测器的对比。我们选择了分数大于某一阈值的检测结果，给出了每幅图像的top20检测结果。对于Faster R-CNN和R-FCN，我们使用阈值为0.5，对于SSD，阈值为0.3。这些阈值是根据主观的视觉效果手工调整得到的，没有使用严格的标准。我们观察到，所有的模型在大目标上表现的都很好，SSD只对于小目标表现略差，在第一幅图像种丢失了一些小的风筝和人，在最后一幅图像中丢失了一些餐桌上的杯子和瓶子。

## 5. Conclusion

We have performed an experimental comparison of some of the main aspects that influence the speed and accuracy of modern object detectors. We hope this will help practitioners choose an appropriate method when deploying object detection in the real world. We have also identified some new techniques for improving speed without sacrificing much accuracy, such as using many fewer proposals than is usual for Faster R-CNN.

我们对现代目标检测器中影响速度和准确率的一些主要方面进行了试验对比。我们希望这会帮助实践者在真实世界部署目标检测系统时，选择合适的方法。我们还发现了一些新技术，可以在不太损失准确率的情况下提高运算速度，比如对于Faster R-CNN来说，使用很少的候选区域。