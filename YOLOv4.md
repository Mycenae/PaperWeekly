# YOLOv4: Optimal Speed and Accuracy of Object Detection

Alexey Bochkovskiy et. al. 

## 0. Abstract

There are a huge number of features which are said to improve Convolutional Neural Network (CNN) accuracy. Practical testing of combinations of such features on large datasets, and theoretical justification of the result, is required. Some features operate on certain models exclusively and for certain problems exclusively, or only for small-scale datasets; while some features, such as batch-normalization and residual-connections, are applicable to the majority of models, tasks, and datasets. We assume that such universal features include Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT) and Mish-activation. We use new features: WRC, CSP, CmBN, SAT, Mish activation, Mosaic data augmentation, CmBN, DropBlock regularization, and CIoU loss, and combine some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50) for the MS COCO dataset at a real-time speed of ∼65 FPS on Tesla V100. Source code is at https://github.com/AlexeyAB/darknet.

有大量特征据说可以改进CNN的准确率。需要将这些特征的组合在大型数据集上进行实际测试，以及对结果的理论解释。一些特征只在特定模型上使用，只在特定问题上使用，或只对小型数据集有用；而另外一些特征，比如批归一化和残差连接，对大部分模型、任务和数据集都可以应用。我们假设，这样的通用特征包括，加权残差连接(WRC)，跨阶段部分连接(CSP)，跨批次归一化(CmBN)，自对抗训练(SAT)和Mish激活。我们使用新特征：WRC，CSP，CmBN，SAT，Mish激活，马赛克数据扩增，CmBN，DropBlock正则化，和CIoU损失，将其中的一部分结合，就可以得到目前最好的结果：在MSCOCO数据集上，使用Tesla V100，可以以实时的速度约65 FPS，得到43.5% AP (65.7% AP50)。代码已开源。

## 1. Introduction

The majority of CNN-based object detectors are largely applicable only for recommendation systems. For example, searching for free parking spaces via urban video cameras is executed by slow accurate models, whereas car collision warning is related to fast inaccurate models. Improving the real-time object detector accuracy enables using them not only for hint generating recommendation systems, but also for stand-alone process management and human input reduction. Real-time object detector operation on conventional Graphics Processing Units (GPU) allows their mass usage at an affordable price. The most accurate modern neural networks do not operate in real time and require large number of GPUs for training with a large mini-batch-size. We address such problems through creating a CNN that operates in real-time on a conventional GPU, and for which training requires only one conventional GPU.

大部分基于CNN的目标检测器，主要只对推荐系统应用。比如，通过市区视频相机寻找可用的停车空间，是通过很慢的准确模型进行的，而车辆冲突的警告则与快速不准确的模型相关。改进实时目标检测器的准确率，使其不仅对生成线索的推荐系统有用，也对独立的过程管理和人类输入缩减可用。实时目标检测器的运算在传统GPU上，使得可以在可接受的价格下进行大量使用。最精确的现代神经网络并不是实时的，在minibatch大小较大时，需要大量的GPU进行训练。我们通过创建在传统GPU上可以实时运行的CNN，来解决这个问题，而训练也只需要一个传统GPU。

The main goal of this work is designing a fast operating speed of an object detector in production systems and optimization for parallel computations, rather than the low computation volume theoretical indicator (BFLOP). We hope that the designed object can be easily trained and used. For example, anyone who uses a conventional GPU to train and test can achieve real-time, high quality, and convincing object detection results, as the YOLOv4 results shown in Figure 1. Our contributions are summarized as follows:

本工作的主要目标，是设计一个生产系统中的快速目标检测器，对并行计算进行了优化，而不是对低计算体理论指示器(BFLOP)优化。我们希望，设计的目标可以很简单的进行训练和使用。比如，任何使用传统GPU来进行训练、测试的，都可以得到实时、高质量的，可信的目标检测结果，如图1的YOLOv4结果。我们的贡献总结如下：

1. We develope an efficient and powerful object detection model. It makes everyone can use a 1080 Ti or 2080 Ti GPU to train a super fast and accurate object detector. 我们提出了一个高效的、强力的目标检测模型。这使得每个人都可以使用一个1080Ti或2080Ti GPU来训练一个超级快速准确的目标检测器。
   
2. We verify the influence of state-of-the-art Bag-of-Freebies and Bag-of-Specials methods of object detection during the detector training. 我们验证了目标检测方法中，在检测器训练时，目前最好的Bag-of-Freebies和Bag-of-Specials的影响。

3. We modify state-of-the-art methods and make them more effecient and suitable for single GPU training, including CBN [89], PAN [49], SAM [85], etc. 我们修改了目前最好的方法，使其更高效，更适用于单GPU训练，包括CBN，PAN，SAM，等。

## 2. Related work

### 2.1. Object detection models

A modern detector is usually composed of two parts, a backbone which is pre-trained on ImageNet and a head which is used to predict classes and bounding boxes of objects. For those detectors running on GPU platform, their backbone could be VGG [68], ResNet [26], ResNeXt [86], or DenseNet [30]. For those detectors running on CPU platform, their backbone could be SqueezeNet [31], MobileNet [28, 66, 27, 74], or ShuffleNet [97, 53]. As to the head part, it is usually categorized into two kinds, i.e., one-stage object detector and two-stage object detector. The most representative two-stage object detector is the R-CNN [19] series, including fast R-CNN [18], faster R-CNN [64], R-FCN [9], and Libra R-CNN [58]. It is also possible to make a two-stage object detector an anchor-free object detector, such as RepPoints [87]. As for one-stage object detector, the most representative models are YOLO [61, 62, 63], SSD [50], and RetinaNet [45]. In recent years, anchor-free one-stage object detectors are developed. The detectors of this sort are CenterNet [13], CornerNet [37, 38], FCOS [78], etc. Object detectors developed in recent years often insert some layers between backbone and head, and these layers are usually used to collect feature maps from different stages. We can call it the neck of an object detector. Usually, a neck is composed of several bottom-up paths and several top-down paths. Networks equipped with this mechanism include Feature Pyramid Network (FPN) [44], Path Aggregation Network (PAN) [49], BiFPN [77], and NAS-FPN [17]. In addition to the above models, some researchers put their emphasis on directly building a new backbone (DetNet [43], DetNAS [7]) or a new whole model (SpineNet [12], HitDetector [20]) for object detection.

一个现代检测器通常由两个部分组成，一个骨干网络，在ImageNet上预训练，和一个头，用于预测目标的类别的边界框。对这些运行在GPU平台上的检测器，其骨干可以是VGG，ResNet，ResNeXt，或DenseNet。对这些运行在CPU平台上的检测器，其骨干可以是SqueezeNet，MobileNet，或ShuffleNet。至于头部分，通常归类于两种，即，单阶段目标检测器，和两阶段目标检测器。最有代表性的两阶段目标检测器是R-CNN系列，包括fast R-CNN，faster R-CNN，R-FCN和Libra R-CNN。将两阶段的目标检测器变成无anchor的目标检测器，也是可能的，比如RepPoints。至于一阶段目标检测器，最有代表性的就是YOLO，SSD和RetinaNet。近年来，也提出来了无anchor的一阶段检测器。这个类型的检测器有CenterNet，CornerNet，FCOS等。在最近这些年提出的目标检测器，通常在骨干和头里插入一些层，这些层通常用于从不同的阶段收集特征图。我们可以称之为目标检测器的脖子。通常，一个脖子是由一些自下向上的路径和几个自上向下的路径构成的。带有这种机制的网络包括，FPN，PAN，BiFPN和NAS-FPN。除了上述这些模型，一些研究者直接构建了一个新的骨干(DetNet, DetNAS)，或整个最新的模型(SpineNet, HitDetector)进行目标检测。

To sum up, an ordinary object detector is composed of several parts: 总结起来，一个正常的目标检测器由下面几个部分构成：

• Input: Image, Patches, Image Pyramid 输入：图像，图像块，图像金字塔

• Backbones: VGG16 [68], ResNet-50 [26], SpineNet [12], EfficientNet-B0/B7 [75], CSPResNeXt50 [81], CSPDarknet53 [81] 骨干网络

• Neck: 脖子部分
• Additional blocks: SPP [25], ASPP [5], RFB [47], SAM [85] 额外的模块
• Path-aggregation blocks: FPN [44], PAN [49], NAS-FPN [17], Fully-connected FPN, BiFPN [77], ASFF [48], SFAM [98] 路径聚积模块

• Heads: 头
• Dense Prediction (one-stage): 密集预测（一阶段）
◦ RPN [64], SSD [50], YOLO [61], RetinaNet [45] (anchor based)
◦ CornerNet [37], CenterNet [13], MatrixNet [60], FCOS [78] (anchor free)
• Sparse Prediction (two-stage): 稀疏预测（两阶段）
◦ Faster R-CNN [64], R-FCN [9], Mask R-CNN [23] (anchor based)
◦ RepPoints [87] (anchor free)

### 2.2. Bag of freebies

Usually, a conventional object detector is trained offline. Therefore, researchers always like to take this advantage and develop better training methods which can make the object detector receive better accuracy without increasing the inference cost. We call these methods that only change the training strategy or only increase the training cost as “bag of freebies.” What is often adopted by object detection methods and meets the definition of bag of freebies is data augmentation. The purpose of data augmentation is to increase the variability of the input images, so that the designed object detection model has higher robustness to the images obtained from different environments. For examples, photometric distortions and geometric distortions are two commonly used data augmentation method and they definitely benefit the object detection task. In dealing with photometric distortion, we adjust the brightness, contrast, hue, saturation, and noise of an image. For geometric distortion, we add random scaling, cropping, flipping, and rotating.

通常，一个传统目标检测器是离线训练的。因此，研究者通常都会利用这一点，开发出更好的训练方法，使目标检测器得到更好的准确率，而不增加推理代价。我们称这种只改变训练策略，或增加训练代价的方法为，bag of freebies。目标检测方法通常采用的，符合bag of freebies定义的，是数据扩增。数据扩增的目的是增加输入图像的变化，这样设计的目标检测模型对不同环境得到的图像，有更高的稳健性。比如，光学变化和几何变化，是两种常用的图像扩增方法，这肯定会对目标检测任务有好处。在光学变化中，我们调整亮度，对比度，色调，饱和度，和图像的噪声。对几何变化，我们增加了随机缩放，剪切，翻转和旋转。

The data augmentation methods mentioned above are all pixel-wise adjustments, and all original pixel information in the adjusted area is retained. In addition, some researchers engaged in data augmentation put their emphasis on simulating object occlusion issues. They have achieved good results in image classification and object detection. For example, random erase [100] and CutOut [11] can randomly select the rectangle region in an image and fill in a random or complementary value of zero. As for hide-and-seek [69] and grid mask [6], they randomly or evenly select multiple rectangle regions in an image and replace them to all zeros. If similar concepts are applied to feature maps, there are DropOut [71], DropConnect [80], and DropBlock [16] methods. In addition, some researchers have proposed the methods of using multiple images together to perform data augmentation. For example, MixUp [92] uses two images to multiply and superimpose with different coefficient ratios, and then adjusts the label with these superimposed ratios. As for CutMix [91], it is to cover the cropped image to rectangle region of other images, and adjusts the label according to the size of the mix area. In addition to the above mentioned methods, style transfer GAN [15] is also used for data augmentation, and such usage can effectively reduce the texture bias learned by CNN.

上面提到的数据扩增方法，都是逐像素的调整，在调整区域中，所有的原始像素信息都得到保留。另外，一些数据扩增的研究者，还研究了模拟目标遮挡的问题。在图像分类和目标检测中，他们得到了很好的结果。比如，随机擦除和CutOut可以随机选择图像中的矩形区域，然后用零值或随机值进行填充。在hide-and-seek [69]和grid mask [6]中，他们随机的或平均的选择图像中的多个矩形区域，将其都替换成零。如果类似的概念应用到特征图，这就是DropOut，DropConnect和DropBlock方法。另外，一些研究者提出了使用多幅图像进行数据扩增的方法。比如，MixUp使用两幅图像用不同的系数相乘和叠加的方法，然后用叠加的系数调整标签。至于CutMix，是将剪切掉的图像用其他图像的矩形区域进行覆盖，根据混合区域的面积来调整标签。除了上面提到的方法，style transfer GAN也用于数据扩增，这种用途可以有效的降低CNN学到的纹理偏差。

Different from the various approaches proposed above, some other bag of freebies methods are dedicated to solving the problem that the semantic distribution in the dataset may have bias. In dealing with the problem of semantic distribution bias, a very important issue is that there is a problem of data imbalance between different classes, and this problem is often solved by hard negative example mining [72] or online hard example mining [67] in two-stage object detector. But the example mining method is not applicable to one-stage object detector, because this kind of detector belongs to the dense prediction architecture. Therefore Lin et al. [45] proposed focal loss to deal with the problem of data imbalance existing between various classes. Another very important issue is that it is difficult to express the relationship of the degree of association between different categories with the one-hot hard representation. This representation scheme is often used when executing labeling. The label smoothing proposed in [73] is to convert hard label into soft label for training, which can make model more robust. In order to obtain a better soft label, Islam et al. [33] introduced the concept of knowledge distillation to design the label refinement network.

与上面提到的各种方法不同的是，一些其他bag of freebies方法想要解决的问题是，数据集中的语义分布可能包含偏差。为解决语义分布偏差的问题，一个非常重要的问题是，不同类别间有数据不均衡的问题，在两阶段目标检测器中，这个问题通常是由难分负样本挖掘或在线难分样本挖掘来解决。但样本挖掘方法不能应用于单阶段目标检测器，因为这种检测器属于密集预测的架构。因此，Lin等[45]提出了focal loss来处理类别间数据不均衡的问题。另一个非常重要的问题是，用one-hot hard表示，很难表达不同类别之间的关系。这种表示方法通常是用于进行标注的。[73]中提出的标签平滑，是将硬标签转化成软标签，以进行训练，这会使得模型更加稳健。为得到更好的软标签，Islam等[33]提出了知识蒸馏的概念，设计了标签提炼网络。

The last bag of freebies is the objective function of Bounding Box (BBox) regression. The traditional object detector usually uses Mean Square Error (MSE) to directly perform regression on the center point coordinates and height and width of the BBox, i.e., {x-center, y-center, w, h}, or the upper left point and the lower right point, i.e., {x-top-left, y-top-left, x-bottom-right, y-bottom-right}. As for anchor-based method, it is to estimate the corresponding offset, for example {x-center-offset, y-center-offset, w-offset, h-offset} and {x-top-left-offset, y-top-left-offset, x-bottom-right-offset, y-bottom-right-offset}. However, to directly estimate the coordinate values of each point of the BBox is to treat these points as independent variables, but in fact does not consider the integrity of the object itself. In order to make this issue processed better, some researchers recently proposed IoU loss [90], which puts the coverage of predicted BBox area and ground truth BBox area into consideration. The IoU loss computing process will trigger the calculation of the four coordinate points of the BBox by executing IoU with the ground truth, and then connecting the generated results into a whole code. Because IoU is a scale invariant representation, it can solve the problem that when traditional methods calculate the l1 or l2 loss of {x, y, w, h}, the loss will increase with the scale. Recently, some researchers have continued to improve IoU loss. For example, GIoU loss [65] is to include the shape and orientation of object in addition to the coverage area. They proposed to find the smallest area BBox that can simultaneously cover the predicted BBox and ground truth BBox, and use this BBox as the denominator to replace the denominator originally used in IoU loss. As for DIoU loss [99], it additionally considers the distance of the center of an object, and CIoU loss [99], on the other hand simultaneously considers the overlapping area, the distance between center points, and the aspect ratio. CIoU can achieve better convergence speed and accuracy on the BBox regression problem.

最后的bag of freebies是BBox的目标函数回归。传统的目标检测器通常使用MSE对BBox的中心点、高度和宽度进行直接回归，即，{x-center, y-center, w, h}，或左上的点和右下的点，即 {x-top-left, y-top-left, x-bottom-right, y-bottom-right}。至于基于anchor的方法，是要估计对应的offset，比如，{x-center-offset, y-center-offset, w-offset, h-offset}，和{x-top-left-offset, y-top-left-offset, x-bottom-right-offset, y-bottom-right-offset}。但是，为直接估计BBox每个点的坐标值，是将这些点都当作独立的变量来处理，但实际上，并没有考虑目标本身的完整性。为使这个问题更好的处理，一些研究者提出了IoU损失，将预测的BBox的覆盖的面积，和真值BBox的面积纳入考虑。IoU损失的计算过程，触发计算BBox的四个坐标点的方式是，将IoU与真值进行处理，然后将生成的结果与完整代码连接起来。因为IoU是一个尺度不变的表示，所以这可以解决传统方法计算{x,y,w,h}的l1或l2损失的问题，损失会随着尺度的增加而增加。最近，一些研究者在持续改进IoU损失。比如，GIoU损失将目标的形状和方向也考虑进来，也包括覆盖面积。他们提出要找到最小面积的BBox，同时覆盖预测的BBox和真值BBox，将这个BBox作为分母，替换掉在IoU损失中使用的分母。至于DIoU损失，其额外的考虑了目标的中央的距离，CIoU损失，同时考虑了重叠区域，中心点之间的距离，和纵横比。CIoU在BBox回归问题中，可以得到更好的收敛速度和准确率。

### 2.3. Bag of specials

For those plugin modules and post-processing methods that only increase the inference cost by a small amount but can significantly improve the accuracy of object detection, we call them “bag of specials”. Generally speaking, these plugin modules are for enhancing certain attributes in a model, such as enlarging receptive field, introducing attention mechanism, or strengthening feature integration capability, etc., and post-processing is a method for screening model prediction results.

一些插件模块和后处理方法，使推理代价增加了一点点，但极大的改进了目标检测的准确率，我们称之为bag of specials。一般来说，这些插件模块是为了强化模型的特定属性，比如，增大感受野，引入注意力机制，或强化特征整合能力，等，后处理是筛选模型预测结果的方法。

Common modules that can be used to enhance receptive field are SPP [25], ASPP [5], and RFB [47]. The SPP module was originated from Spatial Pyramid Matching (SPM) [39], and SPMs original method was to split feature map into several d × d equal blocks, where d can be {1, 2, 3, ...}, thus forming spatial pyramid, and then extracting bag-of-word features. SPP integrates SPM into CNN and use max-pooling operation instead of bag-of-word operation. Since the SPP module proposed by He et al. [25] will output one dimensional feature vector, it is infeasible to be applied in Fully Convolutional Network (FCN). Thus in the design of YOLOv3 [63], Redmon and Farhadi improve SPP module to the concatenation of max-pooling outputs with kernel size k × k, where k = {1, 5, 9, 13}, and stride equals to 1. Under this design, a relatively large k × k maxpooling effectively increase the receptive field of backbone feature. After adding the improved version of SPP module, YOLOv3-608 upgrades AP50 by 2.7% on the MS COCO object detection task at the cost of 0.5% extra computation. The difference in operation between ASPP [5] module and improved SPP module is mainly from the original k×k kernel size, max-pooling of stride equals to 1 to several 3 × 3 kernel size, dilated ratio equals to k, and stride equals to 1 in dilated convolution operation. RFB module is to use several dilated convolutions of k×k kernel, dilated ratio equals to k, and stride equals to 1 to obtain a more comprehensive spatial coverage than ASPP. RFB [47] only costs 7% extra inference time to increase the AP50 of SSD on MS COCO by 5.7%.

常见的用于增强感受野的方法是SPP，ASPP和RFB。SPP模块是来自于SPM的，SPMs的原始方法是将特征图分割成几个d × d的相同模块，其中d可以是{1, 2, 3, ...}，因此形成了空间金字塔，然后提取了bag-of-word特征。SPP将SPM集成到CNN中，使用max-pooling运算，而不是bag-of-words运算。由于He等[25]提出的SPP模块会输出一维特征向量，所以在FCN中进行应用，是不可行的。因此在设计YOLOv3时，Redmon和Farhadi将SPP模块改进成，核大小为k × k的max-pooling输出的拼接，其中k = {1, 5, 9, 13}，步长为1。在这种设计下，相对较大的k × k maxpooling有效的增大了骨干特征的感受野。在加上了改进版的SPP模块后，YOLOv3-608将AP50在MS COCO目标检测任务上改进了2.7%，代价是增加了0.5%的计算量。ASPP和改进的SPP模块之间的差异，主要是原始的kxk核大小，步长为1的max-pooling，到几个3x3的核大小，dilated率等于k，在dilated卷积运算中步长等于1。RFB模块是用几个k×k大小的dilated卷积，dilated率为k，步长为1，得到比ASPP更综合的空间覆盖。RFB增加了额外7%的推理时间，将SSD在MS COCO上的AP50增加了5.7%。

The attention module that is often used in object detection is mainly divided into channel-wise attention and pointwise attention, and the representatives of these two attention models are Squeeze-and-Excitation (SE) [29] and Spatial Attention Module (SAM) [85], respectively. Although SE module can improve the power of ResNet50 in the ImageNet image classification task 1% top-1 accuracy at the cost of only increasing the computational effort by 2%, but on a GPU usually it will increase the inference time by about 10%, so it is more appropriate to be used in mobile devices. But for SAM, it only needs to pay 0.1% extra calculation and it can improve ResNet50-SE 0.5% top-1 accuracy on the ImageNet image classification task. Best of all, it does not affect the speed of inference on the GPU at all.

注意力模块在目标检测中经常使用，主要分为逐通道的注意力，和逐点的注意力，这两种注意力模块的代表，分别是SE和SAM。SE模块可以将ResNet50在ImageNet图像分类上的top-1准确率提高1%，其代价是增加了计算量2%，但在GPU上，其推理时间增加了10%，所以更适合于在移动设备上使用。但SAM只需要增加0.1%的计算量，就将ResNet50-SE在ImageNet分类上的top-1准确率提高了0.5%。最好的是，在GPU上的推理速度并不影响。

In terms of feature integration, the early practice is to use skip connection [51] or hyper-column [22] to integrate low-level physical feature to high-level semantic feature. Since multi-scale prediction methods such as FPN have become popular, many lightweight modules that integrate different feature pyramid have been proposed. The modules of this sort include SFAM [98], ASFF [48], and BiFPN [77]. The main idea of SFAM is to use SE module to execute channel-wise level re-weighting on multi-scale concatenated feature maps. As for ASFF, it uses softmax as point-wise level reweighting and then adds feature maps of different scales. In BiFPN, the multi-input weighted residual connections is proposed to execute scale-wise level re-weighting, and then add feature maps of different scales.

在特征整合上，早期的实践是使用跳跃连接，或hyper-column，以将低层物理特征与高层语义特征整合到一起。由于多尺度预测方法，如FPN，已经非常流行，所以提出了很多整合了不同特征金字塔的轻量级模块。这种类型的模块包括，SFAM，ASFF，和BiFPN。SFAM的主要思想是，使用SE模块来在多尺度拼接特征图上执行逐通道级的重新赋权。至于ASFF，其使用softmax作为逐点级的重新赋权，然后将不同尺度的特征图加到一起。在BiFPN中，提出了多输入的加权残差连接，来进行逐尺度级的重赋权，然后将不同尺度的特征图加在一起。

In the research of deep learning, some people put their focus on searching for good activation function. A good activation function can make the gradient more efficiently propagated, and at the same time it will not cause too much extra computational cost. In 2010, Nair and Hinton [56] propose ReLU to substantially solve the gradient vanish problem which is frequently encountered in traditional tanh and sigmoid activation function. Subsequently, LReLU [54], PReLU [24], ReLU6 [28], Scaled Exponential Linear Unit (SELU) [35], Swish [59], hard-Swish [27], and Mish [55], etc., which are also used to solve the gradient vanish problem, have been proposed. The main purpose of LReLU and PReLU is to solve the problem that the gradient of ReLU is zero when the output is less than zero. As for ReLU6 and hard-Swish, they are specially designed for quantization networks. For self-normalizing a neural network, the SELU activation function is proposed to satisfy the goal. One thing to be noted is that both Swish and Mish are continuously differentiable activation function.

在深度学习的研究中，一些研究者寻找好的激活函数。一个好的激活函数可以使得梯度更高效的传播，同时不会消耗太多计算量。在2010年，Nair和Hinton[56]提出ReLU，基本上解决了梯度消失问题，这在传统的tanh和sigmoid中经常遇到。后来，LReLU，PReLU，ReLU6，SELU，Swish，hard-Swish，和Mish等，在不同文献中提出，也用于解决梯度消失问题。LReLU和PReLU的主要目标是解决，当输出小于零时，ReLU的梯度为零的问题。而ReLU6和hard-Swish，则是专门设计用于解决量化网络问题的。对于自正则化一个神经网络来说，SELU激活函数可以满足这个目标。要注意的一个事情是，Swish和Mish是连续可微的激活函数。

The post-processing method commonly used in deep-learning-based object detection is NMS, which can be used to filter those BBoxes that badly predict the same object, and only retain the candidate BBoxes with higher response. The way NMS tries to improve is consistent with the method of optimizing an objective function. The original method proposed by NMS does not consider the context information, so Girshick et al. [19] added classification confidence score in R-CNN as a reference, and according to the order of confidence score, greedy NMS was performed in the order of high score to low score. As for soft NMS [1], it considers the problem that the occlusion of an object may cause the degradation of confidence score in greedy NMS with IoU score. The DIoU NMS [99] developers way of thinking is to add the information of the center point distance to the BBox screening process on the basis of soft NMS. It is worth mentioning that, since none of above postprocessing methods directly refer to the captured image features, post-processing is no longer required in the subsequent development of an anchor-free method.

在基于深度学习的目标检测中，常用的后处理方法是NMS，可用于将错误的预测同一目标的BBoxes滤除掉，只保留高响应的候选BBoxes。NMS作出改进的方式，与优化一个目标函数的方法是一致的。NMS提出的原始方法，并没有考虑上下文信息，所以Girshick等[19]在R-CNN中增加了分类置信度分数作为参考，根据置信度分数的顺序，从高分到低分进行贪婪NMS。至于soft NMS，其考虑的问题是，目标的遮挡会导致在贪婪NMS中用IoU分数的置信度分数的下降。DIoU NMS是在soft NMS的基础上，将中心点距离加入到BBox筛选过程中。值得提到的是，由于上面提到的后处理方法，都没有直接参考捕获的图像特征，在anchor-free方法的后续开发中，后处理就不需要了。

## 3. Methodology

The basic aim is fast operating speed of neural network, in production systems and optimization for parallel computations, rather than the low computation volume theoretical indicator (BFLOP). We present two options of real-time neural networks:

基本的目标是，神经网络运算速度快，在生产系统中和并行计算的优化中都要快，而不是很低的计算体积理论指示器(BFLOP)。我们提出了实时神经网络的两个选项：

• For GPU we use a small number of groups (1 - 8) in convolutional layers: CSPResNeXt50 / CSPDarknet53 对于GPU，我们使用几组(1-8)卷积层

• For VPU - we use grouped-convolution, but we refrain from using Squeeze-and-excitement (SE) blocks - specifically this includes the following models: EfficientNet-lite / MixNet [76] / GhostNet [21] / MobileNetV3 对于VPU，我们使用分组卷积，但是我们没有使用SE模块，具体的，这包括了下面的模型。

### 3.1. Selection of architecture

Our objective is to find the optimal balance among the input network resolution, the convolutional layer number, the parameter number (filter size2 * filters * channel / groups), and the number of layer outputs (filters). For instance, our numerous studies demonstrate that the CSPResNext50 is considerably better compared to CSPDarknet53 in terms of object classification on the ILSVRC2012 (ImageNet) dataset [10]. However, conversely, the CSPDarknet53 is better compared to CSPResNext50 in terms of detecting objects on the MS COCO dataset [46].

我们的目标是，在输入网络分辨率，卷积层数量，参数数量，层的输出的数量之间找到最佳平衡。比如，我们的很多研究证明了，CSPResNext50比CSPDarknet53，在ILSVRC2012目标分类中要好很多。但是，在MS COCO目标检测任务中，CSPDarknet53与CSPResNext50相比要好很多。

The next objective is to select additional blocks for increasing the receptive field and the best method of parameter aggregation from different backbone levels for different detector levels: e.g. FPN, PAN, ASFF, BiFPN.

下一个目标是，选择额外的模块，以增加感受野，和从不同的骨干层次不同的检测器层次中最佳的参数聚积方法，如，FPN，PAN，ASFF，BiFPN。

A reference model which is optimal for classification is not always optimal for a detector. In contrast to the classifier, the detector requires the following:

对分类是最佳的参考模型，对于检测器并不一定是最佳的。与分类器相比，检测器需要下面的性质：

• Higher input network size (resolution) – for detecting multiple small-sized objects 输入网络大小要更高（分辨率） - 这样可以检测到多个小型的目标

• More layers – for a higher receptive field to cover the increased size of input network 更多的层，感受野要更大，以覆盖输入网络增加的大小。

• More parameters – for greater capacity of a model to detect multiple objects of different sizes in a single image 更多的参数，模型要可以在单幅图像中检测不同大小的多个目标。

Hypothetically speaking, we can assume that a model with a larger receptive field size (with a larger number of convolutional layers 3 × 3) and a larger number of parameters should be selected as the backbone. Table 1 shows the information of CSPResNeXt50, CSPDarknet53, and EfficientNet B3. The CSPResNext50 contains only 16 convolutional layers 3 × 3, a 425 × 425 receptive field and 20.6 M parameters, while CSPDarknet53 contains 29 convolutional layers 3 × 3, a 725 × 725 receptive field and 27.6 M parameters. This theoretical justification, together with our numerous experiments, show that CSPDarknet53 neural network is the optimal model of the two as the backbone for a detector.

假设来说，我们可以认为，有更大感受野（卷积层数量大），和更大参数数量的模型，应当选择为骨干。表1展示了CSPResNeXt50, CSPDarknet53, 和EfficientNet B3的信息。CSPResNext50只有16个卷积层，感受野大小425 × 425，20.6M参数，而CSPDarknet53包含29个卷积层，725 × 725感受野大小，27.6M参数。理论依据和我们很多试验一起表明，CSPDarknet53作为检测器的骨干网络是更好的。

The influence of the receptive field with different sizes is summarized as follows: 不同大小感受野的影响，总结如下：

• Up to the object size - allows viewing the entire object 达到目标大小，可以看到整个目标；

• Up to network size - allows viewing the context around the object 达到网络大小，可以看到目标周围的上下文；

• Exceeding the network size - increases the number of connections between the image point and the final activation 超过了网络大小，增加图像点与最终的激活之间的连接数量；

We add the SPP block over the CSPDarknet53, since it significantly increases the receptive field, separates out the most significant context features and causes almost no reduction of the network operation speed. We use PANet as the method of parameter aggregation from different backbone levels for different detector levels, instead of the FPN used in YOLOv3.

我们在CSPDarknet53之上增加了SPP模块，由于其显著增加了感受野大小，分离出了最显著的上下文特征，而网络运算速度几乎没有影响。我们使用PANet作为参数聚积方法，从不同的骨干层次对不同的检测器层次聚积参数，而没有使用YOLOv3中的FPN。

Finally, we choose CSPDarknet53 backbone, SPP additional module, PANet path-aggregation neck, and YOLOv3 (anchor based) head as the architecture of YOLOv4.

最后，我们选择了CSPDarknet53作为骨干，SPP为额外的模块，PANet作为路径聚积的脖子，YOLOv3的头（基于anchor）作为YOLOv4的架构。

In the future we plan to expand significantly the content of Bag of Freebies (BoF) for the detector, which theoretically can address some problems and increase the detector accuracy, and sequentially check the influence of each feature in an experimental fashion.

在未来，我们计划将检测器的BoF进行极大扩充，这在理论上可以解决一些问题，增加检测器的准确率，以试验按顺序检查每个特征的影响。

We do not use Cross-GPU Batch Normalization (CGBN or SyncBN) or expensive specialized devices. This allows anyone to reproduce our state-of-the-art outcomes on a conventional graphic processor e.g. GTX 1080Ti or RTX 2080Ti.

我们没有使用Cross-GPU的批归一化，或昂贵的专用设备。这使得任何人都可以在一个传统的GPU上复现我们的目前最好效果，如GTX 1080Ti，或RTX 2080Ti。

### 3.2. Selection of BoF and BoS

For improving the object detection training, a CNN usually uses the following: 为改进目标检测的训练，CNN通常会使用下面的元素：

• Activations: ReLU, leaky-ReLU, parametric-ReLU, ReLU6, SELU, Swish, or Mish 激活函数

• Bounding box regression loss: MSE, IoU, GIoU, CIoU, DIoU 边界框回归损失

• Data augmentation: CutOut, MixUp, CutMix 数据扩增

• Regularization method: DropOut, DropPath [36], Spatial DropOut [79], or DropBlock 正则化方法

• Normalization of the network activations by their mean and variance: Batch Normalization (BN) [32], Cross-GPU Batch Normalization (CGBN or SyncBN) [93], Filter Response Normalization (FRN) [70], or Cross-Iteration Batch Normalization (CBN) [89] 用均值和方差对网络激活进行归一化

• Skip-connections: Residual connections, Weighted residual connections, Multi-input weighted residual connections, or Cross stage partial connections (CSP) 跳跃连接

As for training activation function, since PReLU and SELU are more difficult to train, and ReLU6 is specifically designed for quantization network, we therefore remove the above activation functions from the candidate list. In the method of reqularization, the people who published Drop-Block have compared their method with other methods in detail, and their regularization method has won a lot. Therefore, we did not hesitate to choose DropBlock as our regularization method. As for the selection of normalization method, since we focus on a training strategy that uses only one GPU, syncBN is not considered.

至于训练激活函数，由于PReLU和SELU更难以训练，ReLU6是专门为量化网络设计的，因此我们从候选列表中移除了上述激活函数。对于正则化的方法，发表Drop-Block的人已经比较了他们的方法与其他方法，这种正则化方法有很大优势。因此，我们选择了DropBlock作为正则化方法。至于归一化方法的选择，由于我们关注的是使用一个GPU的训练策略，syncBN就不在我们的考虑范围内了。

### 3.3. Additional improvements

In order to make the designed detector more suitable for training on single GPU, we made additional design and improvement as follows: 为使设计的检测器更适用于在一个GPU上进行训练，我们进行了下面的额外设计和改进：

• We introduce a new method of data augmentation Mosaic, and Self-Adversarial Training (SAT)，提出了一种数据扩增的新方法

• We select optimal hyper-parameters while applying genetic algorithms，我们应用遗传算法，选择最优的超参数

• We modify some exsiting methods to make our design suitble for efficient training and detection - modified SAM, modified PAN, and Cross mini-Batch Normalization (CmBN) 我们对一些现有方法进行改进，使我们现有的设计更适合于高效的训练和检测，包括修改的SAM，PAN和CmBN；

Mosaic represents a new data augmentation method that mixes 4 training images. Thus 4 different contexts are mixed, while CutMix mixes only 2 input images. This allows detection of objects outside their normal context. In addition, batch normalization calculates activation statistics from 4 different images on each layer. This significantly reduces the need for a large mini-batch size.

Mosaic表示一种新的数据扩增方法，将4幅训练图像进行混合。因此混合了4种不同的上下文，而CutMix只混合了两幅输入图像。这使得可以在其正常上下文之外进行检测目标。另外，BN是从每层的4幅不同的图像计算的激活统计。这显著的降低了大mini-batch size的需要。

Self-Adversarial Training (SAT) also represents a new data augmentation technique that operates in 2 forward backward stages. In the 1st stage the neural network alters the original image instead of the network weights. In this way the neural network executes an adversarial attack on itself, altering the original image to create the deception that there is no desired object on the image. In the 2nd stage, the neural network is trained to detect an object on this modified image in the normal way.

SAT也表示了一种新的数据扩增方法，在两个前向反向阶段进行运算。在第一阶段，神经网络改变了原始图像，而不是改变网络权重。这样神经网络对其自己进行了对抗攻击，改变原始图像创建了一种假象，即在图像中没有理想的目标。在第二阶段，神经网络训练用于在修改的图像中，以正常的方式来检测目标。

CmBN represents a CBN modified version, as shown in Figure 4, defined as Cross mini-Batch Normalization (CmBN). This collects statistics only between mini-batches within a single batch. CmBN是CBN的一个修正版本，如图4所示，定义为CmBN，在一个batch中的mini-batches之间收集统计。

We modify SAM from spatial-wise attention to pointwise attention, and replace shortcut connection of PAN to concatenation, as shown in Figure 5 and Figure 6, respectively. 我们对SAM从空间的注意力修正为逐点的注意力，将PAN的捷径连接替换为拼接，分别如图5和6所示。

### 3.4. YOLOv4

In this section, we shall elaborate the details of YOLOv4. YOLOv4 consists of: 本节中，我们详述一下YOLOv4中的细节。YOLOv4包括：

• Backbone: CSPDarknet53 [81] • Neck: SPP [25], PAN [49] • Head: YOLOv3 [63]

YOLO v4 uses:

• Bag of Freebies (BoF) for backbone: CutMix and Mosaic data augmentation, DropBlock regularization, Class label smoothing

• Bag of Specials (BoS) for backbone: Mish activation, Cross-stage partial connections (CSP), Multi-input weighted residual connections (MiWRC)

• Bag of Freebies (BoF) for detector: CIoU-loss, CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training, Eliminate grid sensitivity, Using multiple anchors for a single ground truth, Cosine annealing scheduler [52], Optimal hyper-parameters, Random training shapes

• Bag of Specials (BoS) for detector: Mish activation, SPP-block, SAM-block, PAN path-aggregation block, DIoU-NMS

## 4. Experiments

We test the influence of different training improvement techniques on accuracy of the classifier on ImageNet (ILSVRC 2012 val) dataset, and then on the accuracy of the detector on MS COCO (test-dev 2017) dataset.

我们在ImageNet数据集上测试了不同训练改进技术对分类器准确率的影响，在MS COCO数据集上测试了对准确率的的影响。

### 4.1. Experimental setup

In ImageNet image classification experiments, the default hyper-parameters are as follows: the training steps is 8,000,000; the batch size and the mini-batch size are 128 and 32, respectively; the polynomial decay learning rate scheduling strategy is adopted with initial learning rate 0.1; the warm-up steps is 1000; the momentum and weight decay are respectively set as 0.9 and 0.005. All of our BoS experiments use the same hyper-parameter as the default setting, and in the BoF experiments, we add an additional 50% training steps. In the BoF experiments, we verify MixUp, CutMix, Mosaic, Bluring data augmentation, and label smoothing regularization methods. In the BoS experiments, we compared the effects of LReLU, Swish, and Mish activation function. All experiments are trained with a 1080 Ti or 2080 Ti GPU.

在ImageNet图像分类试验中，默认的超参数如下：训练步数为8,000,000，batch大小和mini-batch大小分别是128和32；学习速率初始值为0.1，采用了多项式衰减的学习速率方案；预热的步数为1000；动量和权重衰减分别是0.9和0.005。我们所有的BoS试验都使用默认的超参数设置，在BoF试验中，我们加入了额外的50%训练步数。在BoF试验中，我们验证了MixUp, CutMix, Mosaic, Bluring数据扩增，和标签平滑的正则化方法。在BoS试验中，我们比较了LReLU，Swish和Mish激活函数。所有试验都用1080 Ti或2080 Ti GPU进行训练。

In MS COCO object detection experiments, the default hyper-parameters are as follows: the training steps is 500,500; the step decay learning rate scheduling strategy is adopted with initial learning rate 0.01 and multiply with a factor 0.1 at the 400,000 steps and the 450,000 steps, respectively; The momentum and weight decay are respectively set as 0.9 and 0.0005. All architectures use a single GPU to execute multi-scale training in the batch size of 64 while mini-batch size is 8 or 4 depend on the architectures and GPU memory limitation. Except for using genetic algorithm for hyper-parameter search experiments, all other experiments use default setting. Genetic algorithm used YOLOv3-SPP to train with GIoU loss and search 300 epochs for min-val 5k sets. We adopt searched learning rate 0.00261, momentum 0.949, IoU threshold for assigning ground truth 0.213, and loss normalizer 0.07 for genetic algorithm experiments. We have verified a large number of BoF, including grid sensitivity elimination, mosaic data augmentation, IoU threshold, genetic algorithm, class label smoothing, cross mini-batch normalization, self-adversarial training, cosine annealing scheduler, dynamic mini-batch size, DropBlock, Optimized Anchors, different kind of IoU losses. We also conduct experiments on various BoS, including Mish, SPP, SAM, RFB, BiFPN, and Gaussian YOLO [8]. For all experiments, we only use one GPU for training, so techniques such as syncBN that optimizes multiple GPUs are not used.

在MS COCO目标检测试验中，默认的超参数如下：训练步数为500,500；步数衰减学习速率方案为，初始学习速率0.01，在400,000和450,000步数时分别乘以0.1；动量和权重衰减分别设为0.9和0.0005。所有架构都使用单个GPU来执行多尺度训练，batch大小为64，mini-batch大小为8或4，这与架构和GPU内存限制有关。除了使用遗传算法进行超参数搜索试验，所有其他试验都使用默认设置。遗传算法使用YOLOv3-SPP和GIoU损失来训练，在min-val 5k集合上搜索300 epochs。对遗传算法试验，我们采用的搜索学习速率为0.00261，动量0.949，指定真值的IoU阈值为0.213，损失归一化0.07。我们验证了大量BoF，包括grid sensitivity elimination, mosaic data augmentation, IoU threshold, genetic algorithm, class label smoothing, cross mini-batch normalization, self-adversarial training, cosine annealing scheduler, dynamic mini-batch size, DropBlock, Optimized Anchors, different kind of IoU losses。我们还对各种BoS上进行了试验，包括Mish, SPP, SAM, RFB, BiFPN, 和Gaussian YOLO。对于所有的试验，我们只使用一个GPU进行训练，所以像syncBN这样的优化多个GPUs的技术就没有进行使用。

### 4.2. Influence of different features on Classifier training

First, we study the influence of different features on classifier training; specifically, the influence of Class label smoothing, the influence of different data augmentation techniques, bilateral blurring, MixUp, CutMix and Mosaic, as shown in Fugure 7, and the influence of different activations, such as Leaky-ReLU (by default), Swish, and Mish.

首先，我们研究了不同特征对分类器训练的影响；具体的，Class label smoothing, different data augmentation techniques, bilateral blurring, MixUp, CutMix and Mosaic的影响如图7所示。还有different activations的影响, such as Leaky-ReLU (by default), Swish, and Mish。

In our experiments, as illustrated in Table 2, the classifier’s accuracy is improved by introducing the features such as: CutMix and Mosaic data augmentation, Class label smoothing, and Mish activation. As a result, our BoF-backbone (Bag of Freebies) for classifier training includes the following: CutMix and Mosaic data augmentation and Class label smoothing. In addition we use Mish activation as a complementary option, as shown in Table 2 and Table 3.

在我们的试验中，如表2所示，分类器的准确率改进，是通过引入下面的特征，比如：CutMix and Mosaic data augmentation, Class label smoothing, and Mish activation。结果是，我们的分类器训练的BoF-backbone包括了下面的：CutMix和Mosaic data augmentation和Class label smoothing。除此以外，我们使用Mish激活作为补充选择，如表2和表3所示。

### 4.3. Influence of different features on Detector training

Further study concerns the influence of different Bag-of-Freebies (BoF-detector) on the detector training accuracy, as shown in Table 4. We significantly expand the BoF list through studying different features that increase the detector accuracy without affecting FPS:

进一步的研究考虑了，不同BoF在检测器训练准确率上的影响，如表4所示。我们通过研究不同的特征，增加检测器准确率而不影响FPS，显著扩展了BoF列表：

• S: Eliminate grid sensitivity the equation $b_x = σ(t_x) + c_x, b_y = σ(t_y)+c_y$, where $c_x$ and $c_y$ are always whole numbers, is used in YOLOv3 for evaluating the object coordinates, therefore, extremely high $t_x$ absolute values are required for the $b_x$ value approaching the $c_x$ or $c_x + 1$ values. We solve this problem through multiplying the sigmoid by a factor exceeding 1.0, so eliminating the effect of grid on which the object is undetectable.

• M: Mosaic data augmentation - using the 4-image mosaic during training instead of single image

• IT: IoU threshold - using multiple anchors for a single ground truth IoU (truth, anchor) > IoU threshold

• GA: Genetic algorithms - using genetic algorithms for selecting the optimal hyperparameters during network training on the first 10% of time periods

• LS: Class label smoothing - using class label smoothing for sigmoid activation

• CBN: CmBN - using Cross mini-Batch Normalization for collecting statistics inside the entire batch, instead of collecting statistics inside a single mini-batch

• CA: Cosine annealing scheduler - altering the learning rate during sinusoid training

• DM: Dynamic mini-batch size - automatic increase of mini-batch size during small resolution training by using Random training shapes

• OA: Optimized Anchors - using the optimized anchors for training with the 512x512 network resolution

• GIoU, CIoU, DIoU, MSE - using different loss algorithms for bounded box regression

Further study concerns the influence of different Bag-of-Specials (BoS-detector) on the detector training accuracy, including PAN, RFB, SAM, Gaussian YOLO (G), and ASFF, as shown in Table 5. In our experiments, the detector gets best performance when using SPP, PAN, and SAM.

### 4.4. Influence of different backbones and pretrained weightings on Detector training

Further on we study the influence of different backbone models on the detector accuracy, as shown in Table 6. We notice that the model characterized with the best classification accuracy is not always the best in terms of the detector accuracy.

我们还研究了不同的骨干模型对检测器准确率的影响，如表6所示。我们注意到，最佳分类准确率的模型，并不总是最佳检测准确率。

First, although classification accuracy of CSPResNeXt50 models trained with different features is higher compared to CSPDarknet53 models, the CSPDarknet53 model shows higher accuracy in terms of object detection.

首先，虽然带有不同特征的CSPResNeXt50模型的分类准确率，比CSPDarknet53模型要高，但CSPDarknet53模型得到了更高的目标检测准确率。

Second, using BoF and Mish for the CSPResNeXt50 classifier training increases its classification accuracy, but further application of these pre-trained weightings for detector training reduces the detector accuracy. However, using BoF and Mish for the CSPDarknet53 classifier training increases the accuracy of both the classifier and the detector which uses this classifier pre-trained weightings. The net result is that backbone CSPDarknet53 is more suitable for the detector than for CSPResNeXt50.

第二，在CSPResNeXt50分类器上使用BoF和Mish训练，增加了分类准确率，但进一步使用这些预训练的权重用于检测器的训练，却降低了检测器的准确率。但是，使用BoF和Mish进行CSPDarknet53的分类器训练，使用分类器预训练的权重，却同时增加了分类器和检测器的准确率。网络结果是，CSPDarknet53骨干比CSPResNeXt50更适用于检测器。

We observe that the CSPDarknet53 model demonstrates a greater ability to increase the detector accuracy owing to various improvements. 我们观察到，CSPDarknet53证明了有更大的能力来改进检测器的准确率，可以有各种改进。

### 4.5. Influence of different mini-batch size on Detector training

Finally, we analyze the results obtained with models trained with different mini-batch sizes, and the results are shown in Table 7. From the results shown in Table 7, we found that after adding BoF and BoS training strategies, the mini-batch size has almost no effect on the detector’s performance. This result shows that after the introduction of BoF and BoS, it is no longer necessary to use expensive GPUs for training. In other words, anyone can use only a conventional GPU to train an excellent detector.

最后，我们分析了模型在不同mini-batch大小下训练得到的结果，结果如表7所示。从表7的结果可以看到，在加上了BoF和BoS之后的训练策略，改变mini-batch大小对检测器性能几乎没有影响。这个结果说明，在引入了BoF和BoS之后，不需要使用昂贵的GPUs来进行训练了。换句话说，任何人都可以用一个传统GPU来训练一个优秀的检测器。

## 5. Results

Comparison of the results obtained with other state-of-the-art object detectors are shown in Figure 8. Our YOLOv4 are located on the Pareto optimality curve and are superior to the fastest and most accurate detectors in terms of both speed and accuracy. Since different methods use GPUs of different architectures for inference time verification, we operate YOLOv4 on commonly adopted GPUs of Maxwell, Pascal, and Volta architectures, and compare them with other state-of-the-art methods. Table 8 lists the frame rate comparison results of using Maxwell GPU, and it can be GTX Titan X (Maxwell) or Tesla M40 GPU. Table 9 lists the frame rate comparison results of using Pascal GPU, and it can be Titan X (Pascal), Titan Xp, GTX 1080 Ti, or Tesla P100 GPU. As for Table 10, it lists the frame rate comparison results of using Volta GPU, and it can be Titan Volta or Tesla V100 GPU.

用其他目前最好的检测器得到的结果比较，如图8所示。我们的YOLOv4是在Pareto最优曲线上的，比最快的最准确的检测器还要好。由于不同的方法使用不同架构的GPUs验证推理时间，我们在常用的Maxwell, Pascal, 和Volta架构上运算YOLOv4，与其他目前最好的方法进行比较。表8列出了使用Maxwell GPU的帧率比较结果。表9给出了使用Pascal GPU的比较结果。表10给出了使用Volta GPU的帧率比较结果。

## 6. Conclusions

We offer a state-of-the-art detector which is faster (FPS) and more accurate (MS COCO AP50...95 and AP50) than all available alternative detectors. The detector described can be trained and used on a conventional GPU with 8-16 GB-VRAM this makes its broad use possible. The original concept of one-stage anchor-based detectors has proven its viability. We have verified a large number of features, and selected for use such of them for improving the accuracy of both the classifier and the detector. These features can be used as best-practice for future studies and developments.

我们给出了目前最好的检测器，比所有可用的检测器更快更准确。描述的检测器可以使用8-16GB VRAM的传统GPU进行训练，这使其可以更广泛的使用。单阶段基于anchor的检测器证明了其可行性。我们验证了大量特征，选择使用其中的一些改进分类器和检测器的准确率。这些特征可以作为未来研究和开发的最佳实践来使用。