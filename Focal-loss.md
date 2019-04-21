# Focal Loss for Dense Object Detection 密集目标检测的聚焦损失

Kaiming He et al. Facebook AI Research

## Abstract 摘要

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. Code is at: https://github.com/facebookresearch/Detectron.

目前准确率最高的目标检测器是基于两阶段的方法，即基于R-CNN的方法，其中将分类器用于稀疏候选目标位置。比较起来，应用于常规密集采样的可能目标位置的单阶段检测器可能更快更简单一些，但在准确率上目前落后于两阶段检测器。在本文中，我们研究了为什么会是这样。我们发现，密集检测器在训练时遇到的极端的前景背景类别不均衡，是中心原因。我们提出解决这种类别不均衡问题，方法是改变标准的交叉熵损失的形状，使其对分类很好的样本的损失的权重下降。这种新的聚焦损失，主要在难分样本的稀疏集上集中训练，防止数量巨大的容易的负样本在训练时淹没了检测器。为评估我们的损失函数的有效性，我们设计并训练了一个简单的密集检测器，我们称之为RetinaNet。我们的结果表明，当用聚焦损失训练时，RetinaNet能够达到单阶段检测器的速度，而且超过所有目前最好的两阶段检测器的准确率。

## 1. Introduction 引言

Current state-of-the-art object detectors are based on a two-stage, proposal-driven mechanism. As popularized in the R-CNN framework [11], the first stage generates a sparse set of candidate object locations and the second stage classifies each candidate location as one of the foreground classes or as background using a convolutional neural network. Through a sequence of advances [10, 28, 20, 14], this two-stage framework consistently achieves top accuracy on the challenging COCO benchmark [21].

目前最好的目标检测器是基于两阶段的、候选驱动式的机制。最开始是R-CNN[11]框架，第一阶段生成候选目标位置的稀疏集，第二阶段使用一个卷积神经网络对每个候选进行分类，成为前景类别或背景。经过一系列发展[10,28,20,14]，这种两阶段框架在COCO基准测试中[21]一直是最高准确率的方法。

Despite the success of two-stage detectors, a natural question to ask is: could a simple one-stage detector achieve similar accuracy? One stage detectors are applied over a regular, dense sampling of object locations, scales, and aspect ratios. Recent work on one-stage detectors, such as YOLO [26, 27] and SSD [22, 9], demonstrates promising results, yielding faster detectors with accuracy within 10-40% relative to state-of-the-art two-stage methods.

虽然两阶段检测器非常成功，有一个要问的很自然的问题：简单的单阶段检测器是否也能取得类似的准确率？单阶段检测器的应用是在目标位置、尺度和纵横比的规则、密集采样中进行的。最近关于单阶段的检测器的工作，如YOLO[26,27]，SSD[22,9]，给出了很有希望的结果，与目前最好的两阶段方法的准确率相差10%-40%，而且速度更快。

This paper pushes the envelop further: we present a one-stage object detector that, for the first time, matches the state-of-the-art COCO AP of more complex two-stage detectors, such as the Feature Pyramid Network (FPN) [20] or Mask R-CNN [14] variants of Faster R-CNN [28]. To achieve this result, we identify class imbalance during training as the main obstacle impeding one-stage detector from achieving state-of-the-art accuracy and propose a new loss function that eliminates this barrier.

本文将继续推进这方面的工作：我们给出一个单阶段目标检测器，第一次与目前最好的复杂得多的两阶段检测器可以匹配，如FPN(Feature Pyramid Network)，或Mask R-CNN, Faster R-CNN[28]的变体。为得到这个结果，我们认定训练过程中的类别不均衡是阻碍单阶段检测器得到最好的准确率的主要障碍，并提出了一种新的损失函数，去除这个障碍。

Class imbalance is addressed in R-CNN-like detectors by a two-stage cascade and sampling heuristics. The proposal stage (e.g., Selective Search [35], EdgeBoxes [39], DeepMask [24, 25], RPN [28]) rapidly narrows down the number of candidate object locations to a small number (e.g., 1-2k), filtering out most background samples. In the second classification stage, sampling heuristics, such as a fixed foreground-to-background ratio (1:3), or online hard example mining (OHEM) [31], are performed to maintain a manageable balance between foreground and background.

类别不均衡在R-CNN类的检测器中是用两阶段级连和启发式采样进行处理的。候选阶段（如Selctive Search[35], EdgeBoxes[39], DeepMask[24,25], RPN[28]）迅速将候选目标位置降到很小的数目（如1-2k），滤除掉了多数背景样本。在第二阶段的分类中，启发式的采样，如固定前景-背景比率(1:3)，或在线难分样本挖掘(OHEM)[31]，被用来保持前景和背景样本的均衡。

In contrast, a one-stage detector must process a much larger set of candidate object locations regularly sampled across an image. In practice this often amounts to enumerating ∼ 100k locations that densely cover spatial positions, scales, and aspect ratios. While similar sampling heuristics may also be applied, they are inefficient as the training procedure is still dominated by easily classified background examples. This inefficiency is a classic problem in object detection that is typically addressed via techniques such as bootstrapping [33, 29] or hard example mining [37, 8, 31].

比较起来，单阶段检测器要处理的候选目标位置数量要大的多，因为是在图像中规则取样得到的。实践中，这经常导致枚举大约100k个位置，密集的覆盖了所有的空间位置、尺度和纵横比。虽然可以使用类似的启发式采样，但却很没有效率，因为训练过程中大多数仍然是很容易分类的背景样本。这种低效性是目标检测中的经典问题，通常用bootstrpping[33,29]或难分样本挖掘[37,8,31]的技术来解决。

In this paper, we propose a new loss function that acts as a more effective alternative to previous approaches for dealing with class imbalance. The loss function is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases, see Figure 1. Intuitively, this scaling factor can automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples. Experiments show that our proposed Focal Loss enables us to train a high-accuracy, one-stage detector that significantly outperforms the alternatives of training with the sampling heuristics or hard example mining, the previous state-of-the-art techniques for training one-stage detectors. Finally, we note that the exact form of the focal loss is not crucial, and we show other instantiations can achieve similar results.

本文中，我们提出一种新的损失函数，可以作为之前方法的更有效的替代，以处理类别不均衡问题。这个损失函数是一个动态尺度的交叉熵损失，其中尺度因子在正确分类置信度增加的时候减少到0，如图1所示。直觉上来说，这种尺度因子可以自动的使容易样本在训练过程中的贡献变小，使模型迅速聚焦到难分样本中。试验表明，我们提出的Focal Loss使我们训练出了一个高准确率的单阶段检测器，显著超过了那些之前最好的单阶段检测器，它们使用的是启发式采样策略，或难分样本挖掘。最后，我们注意到，focal loss的精确形式不是决定性的，我们证明，其他的例示也可以类似的结果。

Figure 1. We propose a novel loss we term the Focal Loss that adds a factor $(1 − p_t)^γ$ to the standard cross entropy criterion. Setting γ > 0 reduces the relative loss for well-classified examples ($p_t > .5$), putting more focus on hard, misclassified examples. As our experiments will demonstrate, the proposed focal loss enables training highly accurate dense object detectors in the presence of vast numbers of easy background examples.

图1. 我们提出了一种新的损失函数，称为Focal Loss，在标准的交叉熵准则上加上了一个因子$(1 − p_t)^γ$。设γ > 0，可以减少良好分类样本的相对损失($p_t > .5$)，分类器的注意力就会更多的放在难分的、错分的样本上。我们的试验会证明，提出的focal loss可以训练出非常准确的密集目标检测器，而且有非常多的容易分类的背景样本。

To demonstrate the effectiveness of the proposed focal loss, we design a simple one-stage object detector called RetinaNet, named for its dense sampling of object locations in an input image. Its design features an efficient in-network feature pyramid and use of anchor boxes. It draws on a variety of recent ideas from [22, 6, 28, 20]. RetinaNet is efficient and accurate; our best model, based on a ResNet-101-FPN backbone, achieves a COCO test-dev AP of 39.1 while running at 5 fps, surpassing the previously best published single-model results from both one and two-stage detectors, see Figure 2.

为证明我们提出的focal loss的有效性，我们设计了一种简单的单阶段目标检测器，称为RetinaNet，因为其在输入图像上的目标位置进行了密集采样。其设计的特点是高效的网络中特征金字塔，并使用了锚框，这从最近的工作[22,6,28,20]中得到了很多启发。RetinaNet是高效又准确的，我们最好的模型是以ResNet-101-FPN为骨架的，取得了COCO test-dev AP为39.1，运行速度为5 fps，超过了发表的之前最好的单阶段和两阶段的单模型结果，如图2所示。

Figure 2. Speed (ms) versus accuracy (AP) on COCO test-dev. Enabled by the focal loss, our simple one-stage RetinaNet detector outperforms all previous one-stage and two-stage detectors, including the best reported Faster R-CNN [28] system from [20]. We show variants of RetinaNet with ResNet-50-FPN (blue circles) and ResNet-101-FPN (orange diamonds) at five scales (400-800 pixels). Ignoring the low-accuracy regime (AP<25), RetinaNet forms an upper envelope of all current detectors, and an improved variant (not shown) achieves 40.8 AP. Details are given in §5.

图2. 在COCO test-dev上的速度(ms)准确度(AP)对比。使用了我们提出的focal loss，我们简单的单阶段RetinaNet检测器超过了之前所有的单阶段和双阶段检测器，包括[20]中最好的Faster R-CNN系统[28]。我们给出了RetinaNet的变体，即ResNet-50-FPN（蓝圈）和ResNet-101-FPN（橘色菱形），在五个尺度（400-800像素）上的结果。忽略了低准确率的方法(AP<25)，RetinaNet成为了所有目前检测器的上外包线，有一个改进的变体（没有在此显示）得到了40.8 AP。细节在第5部分给出。

## 2. Related Work 相关工作

**Classic Object Detectors**: The sliding-window paradigm, in which a classifier is applied on a dense image grid, has a long and rich history. One of the earliest successes is the classic work of LeCun et al. who applied convolutional neural networks to handwritten digit recognition [19, 36]. Viola and Jones [37] used boosted object detectors for face detection, leading to widespread adoption of such models. The introduction of HOG [4] and integral channel features [5] gave rise to effective methods for pedestrian detection. DPMs [8] helped extend dense detectors to more general object categories and had top results on PASCAL [7] for many years. While the sliding-window approach was the leading detection paradigm in classic computer vision, with the resurgence of deep learning [18], two-stage detectors, described next, quickly came to dominate object detection.

**经典目标检测器**：滑窗方法，其中一个分类器在密集的图像网格上应用，有着很长又丰富的历史。最早的成功是LeCun等人的经典工作，他们将卷积神经网络应用于手写数字识别[19,36]。Viola和Jones[37]使用boosted目标检测器进行人脸检测，使得人们广泛采用这种模型。HOG[4]的提出和积分通道特征(Integrated Channel Features)[5]使得高效行人检测方法崛起。DPMs[8]帮助将密集检测器拓展到更一般性的目标类别，在PASCAL[9]上领先了好多年。滑窗法在经典计算机视觉上领导着检测方案，但随着深度学习[18]的崛起，两阶段检测器很快就主导了目标检测的研究。

**Two-stage Detectors**: The dominant paradigm in modern object detection is based on a two-stage approach. As pioneered in the Selective Search work [35], the first stage generates a sparse set of candidate proposals that should contain all objects while filtering out the majority of negative locations, and the second stage classifies the proposals into foreground classes / background. R-CNN [11] upgraded the second-stage classifier to a convolutional network yielding large gains in accuracy and ushering in the modern era of object detection. R-CNN was improved over the years, both in terms of speed [15, 10] and by using learned object proposals [6, 24, 28]. Region Proposal Networks (RPN) integrated proposal generation with the second-stage classifier into a single convolution network, forming the Faster R-CNN framework [28]. Numerous extensions to this framework have been proposed, e.g. [20, 31, 32, 16, 14].

**两阶段检测器**：现代目标检测的主导性方案是基于两阶段方法的。第一阶段生成候选建议的稀疏集，应当包含所有的目标，同时过滤掉大部分非目标位置，这个阶段的先驱是Selective Search法[35]；第二阶段将候选分类成前景类别/背景。R-CNN[11]将第二阶段的分类器升级为卷积网络，大幅提升了准确率，开启了目标检测的新纪元。R-CNN持续得到改进，包括速度方面[15,10]和使用学习好的目标建议[6,24,28]。区域建议网络(Region Proposal Networks, RPN)将候选生成和第二阶段的分类器整合到一个卷积网络中，形成了Faster R-CNN框架[28]。提出了很多这种框架的拓展，如[20,31,32,16,14]。

**One-stage Detectors**: OverFeat [30] was one of the first modern one-stage object detector based on deep networks. More recently SSD [22, 9] and YOLO [26, 27] have renewed interest in one-stage methods. These detectors have been tuned for speed but their accuracy trails that of two-stage methods. SSD has a 10-20% lower AP, while YOLO focuses on an even more extreme speed/accuracy trade-off. See Figure 2. Recent work showed that two-stage detectors can be made fast simply by reducing input image resolution and the number of proposals, but one-stage methods trailed in accuracy even with a larger compute budget [17]. In contrast, the aim of this work is to understand if one-stage detectors can match or surpass the accuracy of two-stage detectors while running at similar or faster speeds. The design of our RetinaNet detector shares many similarities with previous dense detectors, in particular the concept of ‘anchors’ introduced by RPN [28] and use of features pyramids as in SSD [22] and FPN [20]. We emphasize that our simple detector achieves top results not based on innovations in network design but due to our novel loss.

**单阶段检测器**：OverFeat[30]是第一个现代的基于深度网络的单阶段目标检测器。更近来的SSD[22,9]和YOLO[26,27]更新了单阶段方法。这些检测器是为速度提出来的，其准确率落后于两阶段方法。SSD的AP要低10%-20%，而YOLO则聚焦于更极端的速度/准确率折中。见图2。最近的工作表明，两阶段检测器如果降低输入图像分辨率和候选数量，则就可以加快速度；但单阶段方法准确率落后，其计算量则更大一些[17]。比较起来，本文的目标是理解，如果单阶段检测器可以匹配或超过两阶段检测器的准确率，是否能够在类似或更快的速度上运行。我们的RetinaNet检测器的设计与之前的密集检测器非常类似，特别是RPN[28]引入的锚的概念，以及SSD[22]和FPN[20]使用的特征金字塔的概念。我们强调，我们简单的检测器取得最好的结果，不是因为网络设计上的创新，而是我们新颖的损失函数。

**Class Imbalance**: Both classic one-stage object detection methods, like boosted detectors [37, 5] and DPMs [8], and more recent methods, like SSD [22], face a large class imbalance during training. These detectors evaluate $10^4 - 10^5$ candidate locations per image but only a few locations contain objects. This imbalance causes two problems: (1) training is inefficient as most locations are easy negatives that contribute no useful learning signal; (2) en masse, the easy negatives can overwhelm training and lead to degenerate models. A common solution is to perform some form of hard negative mining [33, 37, 8, 31, 22] that samples hard examples during training or more complex sampling/reweighing schemes [2]. In contrast, we show that our proposed focal loss naturally handles the class imbalance faced by a one-stage detector and allows us to efficiently train on all examples without sampling and without easy negatives overwhelming the loss and computed gradients.

**类别不均衡**：经典的单阶段目标检测方法，如boosted检测器[37,5]和DPM[8]，和最近的方法，如SSD[22]，在训练时都会面临着很大的类别不均衡问题。这些检测器每分钟要评估$10^4 - 10^5$个候选位置，但只有几个位置包含目标。这种不均衡导致两个问题：(1)训练效率低，因为多数位置都很容易判断是负样本，没有贡献有用的学习信号；(2)这些容易的负样本会在训练中占据绝大部分，导致降质模型。通常的解决方案是进行一定形式的难分样本挖掘[33,37,8,31,22]，在训练过程中对难分样本进行采样，或更复杂的取样/重新赋权方案[2]。比较起来，我们证明了，我们提出的focal loss可以很自然的处理单阶段检测器面临的类别不均衡问题，使我们高效的在所有样本上训练，不需要采样，容易分类样本也不会占据损失的决定性地位并计算梯度。

**Robust Estimation**: There has been much interest in designing robust loss functions (e.g., Huber loss [13]) that reduce the contribution of outliers by down-weighting the loss of examples with large errors (hard examples). In contrast, rather than addressing outliers, our focal loss is designed to address class imbalance by down-weighting inliers (easy examples) such that their contribution to the total loss is small even if their number is large. In other words, the focal loss performs the opposite role of a robust loss: it focuses training on a sparse set of hard examples.

**稳健的估计**：有很多工作设计稳健的损失函数（如Huber损失[13]），降低离群样本的贡献，将大错误的样本对损失的权重降低。比较起来，我们不是处理离群样本，我们的focal loss是设计用来处理类别不均衡问题的，方法是降低内部样本（容易分类样本）的权重，这样其对总体损失的贡献会变小，即使其数量巨大。换句话说，focal loss是robust loss的反面：其聚焦在难分样本稀疏集的训练上。

## 3. Focal Loss

The Focal Loss is designed to address the one-stage object detection scenario in which there is an extreme imbalance between foreground and background classes during training (e.g., 1:1000). We introduce the focal loss starting from the cross entropy (CE) loss for binary classification(Extending the focal loss to the multi-class case is straightforward and works well; for simplicity we focus on the binary loss in this work):

Focal Loss的设计是用于处理单阶段目标检测场景下，训练时前景和背景类别极度不均衡的问题（如1:1000）。我们从二类分类的交叉熵损失开始介绍focal loss（将focal loss扩展到多类别的情况是很直接的，运行结果也很好；为简化起见，我们在本文中关注点在二值分类损失）：

$$CE(p,y) = -log(p), if. y=1; -log(1-p), otherwise$$(1)

In the above y ∈ {±1} specifies the ground-truth class and p ∈ [0, 1] is the model’s estimated probability for the class with label y = 1. For notational convenience, we define $p_t$: 上面y ∈ {±1}指定了真值类别，p ∈ [0, 1]是模型估计的y=1的类别概率。为表示上的方便，我们定义$p_t$：

$$p_t = p, if. y=1; 1-p, otherwise$$(2)

and rewrite $CE(p, y) = CE(p_t) = −log(p_t)$.

The CE loss can be seen as the blue (top) curve in Figure 1. One notable property of this loss, which can be easily seen in its plot, is that even examples that are easily classified ($p_t >> .5$) incur a loss with non-trivial magnitude. When summed over a large number of easy examples, these small loss values can overwhelm the rare class. 交叉熵损失可以看作图1中的蓝线（上面）。这种损失的一个重要性质是，即使样本分类很容易($p_t >> .5$)，也会带来有一定幅度的损失，这可以从图中轻松看出来。当很多容易分类样本相加，这些小的损失会把那些稀有的类别淹没掉。

### 3.1. Balanced Cross Entropy 平衡的交叉熵

A common method for addressing class imbalance is to introduce a weighting factor α ∈ [0, 1] for class 1 and 1 − α for class −1. In practice α may be set by inverse class frequency or treated as a hyperparameter to set by cross validation. For notational convenience, we define $α_t$ analogously to how we defined $p_t$. We write the α-balanced CE loss as:

处理类别不均衡问题的普通方法是对类别1引入权重因子α ∈ [0, 1]，对类别-1引入1 − α。实践中α可以通过类别频率逆来设置，或当作一个通过交叉验证设置的超参数。为表示上的方便，我们定义$α_t$，与我们定义的$p_t$类似。我们将α-均衡的损失表示如下：

$$CE(p_t) = −α_t log(p_t)$$(3)

This loss is a simple extension to CE that we consider as an experimental baseline for our proposed focal loss. 这个损失是交叉熵损失的简单扩展，作为我们提出的focal loss的一个试验基准。

### 3.2. Focal Loss Definition 定义

As our experiments will show, the large class imbalance encountered during training of dense detectors overwhelms the cross entropy loss. Easily classified negatives comprise the majority of the loss and dominate the gradient. While α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples. Instead, we propose to reshape the loss function to down-weight easy examples and thus focus training on hard negatives.

我们的试验将会表明，密集检测器在训练中遇到的大的类别不均衡会把淹没交叉熵损失。容易分类的负样本成为了损失的大部分，主导了梯度。α平衡了正样本和负样本的重要性，但并没有区分容易分类的样本和难以分类的样本。相反，我们提出让损失函数对简单分类样本的权重降低，从而聚焦在难分样本的训练上。

More formally, we propose to add a modulating factor $(1 − p_t)^γ$ to the cross entropy loss, with tunable focusing parameter γ ≥ 0. We define the focal loss as: 正式的，我们提出给交叉熵损失加上一个调制因子$(1 − p_t)^γ$，其中有一个可调节的聚焦参数γ ≥ 0。我们定义focal loss为：

$$FL(p_t) = −(1 − p_t)^γ log(p_t)$$(4)

The focal loss is visualized for several values of γ ∈ [0, 5] in Figure 1. We note two properties of the focal loss. (1) When an example is misclassified and $p_t$ is small, the modulating factor is near 1 and the loss is unaffected. As $p_t$ → 1, the factor goes to 0 and the loss for well-classified examples is down-weighted. (2) The focusing parameter γ smoothly adjusts the rate at which easy examples are down-weighted. When γ = 0, FL is equivalent to CE, and as γ is increased the effect of the modulating factor is likewise increased (we found γ = 2 to work best in our experiments).

图1中有几种γ ∈ [0, 5]值下focal loss的图示。我们注意到focal loss有两个性质。(1)当样本误分类，$p_t$很小时，调制因子接近于1，损失并为受影响。当$p_t$接近1时，因子趋向于0，容易分类的样本的损失权重降低；(2)聚焦参数γ平滑的调整容易分类的样本降低权重的速度。当γ=0，focal loss与交叉熵损失相同，当γ增加时，调制因子的影响也响应增加（我们发现γ=2在我们的试验中效果最好）。

Intuitively, the modulating factor reduces the loss contribution from easy examples and extends the range in which an example receives low loss. For instance, with γ = 2, an example classified with $p_t = 0.9$ would have 100× lower loss compared with CE and with $p_t ≈ 0.968$ it would have 1000× lower loss. This in turn increases the importance of correcting misclassified examples (whose loss is scaled down by at most 4× for $p_t$ ≤ .5 and γ = 2).

直觉上来说，调制因子降低容易分类样本贡献的损失，增大样本损失小的范围。比如，在γ=2时，以概率$p_t = 0.9$分类的样本的损失与交叉熵损失比较，就降低了100倍；而$p_t ≈ 0.968$的，损失则降低了1000倍。这也增加了纠正错误分类样本的重要性（在$p_t$ ≤ .5，γ = 2时，其损失最多降低了4倍）。

In practice we use an α-balanced variant of the focal loss: 实践中，我们使用focal loss的α-均衡变体：

$$FL(p_t) = −α_t (1 − p_t)^γ log(p_t)$$(5)

We adopt this form in our experiments as it yields slightly improved accuracy over the non-α-balanced form. Finally, we note that the implementation of the loss layer combines the sigmoid operation for computing p with the loss computation, resulting in greater numerical stability. 试验中我们采取这种形式，因为与非α均衡的形式相比，可以得到略微改进的准确率。我们指出，损失层的实现，与计算p的sigmoid操作的损失一起，可以得到很好的数值稳定性。

While in our main experimental results we use the focal loss definition above, its precise form is not crucial. In the appendix we consider other instantiations of the focal loss and demonstrate that these can be equally effective. 在我们的主要试验结果中，我们使用的都是上面的损失定义，但是其精确形式并不是关键的。在附录中，我们考虑了focal loss的其他实现，证明了也可以一样的有效。

### 3.3. Class Imbalance and Model Initialization 类别不均衡和模型初始化

Binary classification models are by default initialized to have equal probability of outputting either y = −1 or 1. Under such an initialization, in the presence of class imbalance, the loss due to the frequent class can dominate total loss and cause instability in early training. To counter this, we introduce the concept of a ‘prior’ for the value of p estimated by the model for the rare class (foreground) at the start of training. We denote the prior by π and set it so that the model’s estimated p for examples of the rare class is low, e.g. 0.01. We note that this is a change in model initialization (see §4.1) and not of the loss function. We found this to improve training stability for both the cross entropy and focal loss in the case of heavy class imbalance.

二值分类模型默认初始化为不同的输出概率是一样的，即y = −1或1。在这样一个初始化的情况下，如果有类别不均衡，那么样本多的类别会主导整个损失函数，导致训练早期的不稳定性。为解决这个问题，我们模型对较少的类（前景）估计的p值，在训练开始的时候就引入先验的概念。我们将这个先验表示为π，其值的设置会使模型估计的p值对于少数的类是很低的，如0.01。我们指出，这是模型初始化的一个变化（见4.1节），并不是损失函数的变化。我们发现，在类别极度不均衡的情况下，对于交叉熵损失和focal loss来说，这都会改进训练的稳定性。

### 3.4 Class Imbalance and Two-stage Detectors 类别不均衡和两类检测器

Two-stage detectors are often trained with the cross entropy loss without use of α-balancing or our proposed loss. Instead, they address class imbalance through two mechanisms: (1) a two-stage cascade and (2) biased minibatch sampling. The first cascade stage is an object proposal mechanism [35, 24, 28] that reduces the nearly infinite set of possible object locations down to one or two thousand. Importantly, the selected proposals are not random, but are likely to correspond to true object locations, which removes the vast majority of easy negatives. When training the second stage, biased sampling is typically used to construct minibatches that contain, for instance, a 1:3 ratio of positive to negative examples. This ratio is like an implicit α-balancing factor that is implemented via sampling. Our proposed focal loss is designed to address these mechanisms in a one-stage detection system directly via the loss function.

两阶段检测器通常使用交叉熵损失训练，而且没有使用α-均衡，也没有用我们提出的损失。相反，他们通过两种机制来解决类别不均衡问题：(1)两阶段级联；(2)有偏向的minibatch取样。级联第一阶段是一个目标建议机制[35,24,28]，将接近无限数量的可能目标位置降低到1k到2k的数量级。重要的是，选择的建议不是随机的，但很可能对应着真实目标位置，这就去除了大多数容易分类的负样本。在第二阶段的训练时，有偏向的取样通常用于构建minibatches，其正负样本比例比如说可以是1:3。这种比率就像一种隐式的α-均衡因子，是通过取样来实现的。我们提出的focal loss就是设计用于在单阶段检测系统中通过损失函数来解决这种机制。

## 4. RetinaNet Detector

RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks. The backbone is responsible for computing a convolutional feature map over an entire input image and is an off-the-self convolutional network. The first subnet performs convolutional object classification on the backbone’s output; the second subnet performs convolutional bounding box regression. The two subnetworks feature a simple design that we propose specifically for one-stage, dense detection, see Figure 3. While there are many possible choices for the details of these components, most design parameters are not particularly sensitive to exact values as shown in the experiments. We describe each component of RetinaNet next.

RetinaNet是单一的统一网络，包括一个骨干网络和两个与任务相关的子网络。骨干网络负责在整个输入图像上计算卷积特征图，是一个现成的卷积网络。第一个子网络进行对骨干网络的输出进行卷积目标分类，第二个子网络进行卷积边界框回归。这两个子网络都有一个简单的设计特征，是我们特意为单阶段密集预测提出来的，如图3所示。虽然有这些组建的细节有很多可能的选择，多数设计参数不是很敏感，不一定选择试验中确定的特定值。下面我们逐一描述各个组件。

**Feature Pyramid Network Backbone**: We adopt the Feature Pyramid Network (FPN) from [20] as the backbone network for RetinaNet. In brief, FPN augments a standard convolutional network with a top-down pathway and lateral connections so the network efficiently constructs a rich, multi-scale feature pyramid from a single resolution input image, see Figure 3(a)-(b). Each level of the pyramid can be used for detecting objects at a different scale. FPN improves multi-scale predictions from fully convolutional networks (FCN) [23], as shown by its gains for RPN [28] and DeepMask-style proposals [24], as well at two-stage detectors such as Fast R-CNN [10] or Mask R-CNN [14].

**特征金字塔网络为骨干**：我们采用特征金字塔网络(FPN)[20]作为RetinaNet的骨干网络。简单来说，FPN用自上而下的通道和横向连接来增强标准的卷积网络，使网络可以从单分辨率输入图像中高效的构建出丰富的多尺度特征金字塔，见图3(a)-(b)。金字塔的每层都可以用于在不同的尺度检测目标。FPN从全卷积网络(FCN)[23]改进了多尺度预测，从其对RPN[28]和DeepMask-style的建议[24]的改进就可以看出，以及对两阶段检测器如Fast R-CNN[10]或Mask R-CNN[14]的改进。

Following [20], we build FPN on top of the ResNet architecture [16]. We construct a pyramid with levels P3 through P7, where l indicates pyramid level ($P_l$ has resolution $2^l$ lower than the input). As in [20] all pyramid levels have C = 256 channels. Details of the pyramid generally follow [20] with a few modest differences. While many design choices are not crucial, we emphasize the use of the FPN backbone is; preliminary experiments using features from only the final ResNet layer yielded low AP. (RetinaNet uses feature pyramid levels P3 to P7, where P3 to P5 are computed from the output of the corresponding ResNet residual stage (C3 through C5) using top-down and lateral connections just as in [20], P6 is obtained via a 3×3 stride-2 conv on C5, and P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6. This differs slightly from [20]: (1) we don’t use the high-resolution pyramid level P2 for computational reasons, (2) P6 is computed by strided convolution instead of downsampling, and (3) we include P7 to improve large object detection. These minor modifications improve speed while maintaining accuracy.)

遵循[20]，我们在ResNet架构[16]上构建FPN。我们从P3到P7层构建了一个金字塔，其中l表示金字塔层级（$P_l$的分辨率比输入层低$2^l$倍）。就像在[20]中一样，所有的金字塔都有C=256个通道。金字塔的细节一般都采用[20]，有几个小的差异。虽然很多设计选择不是关键性的，但我们要强调，使用FPN作为骨干网络是很关键的；初步试验结果表明，只使用ResNet最终层的特征，得到的AP很低。（RetinaNet使用特征金字塔的层次为P3到P7，其中P3到P5是从对应的ResNet残差阶段中计算出来的（C3到C5），使用的自上而下结构和横向连接与[20]中一样，P6是对C5进行一个3×3步长为2的卷积得到的，P7是P6通过一个3×3步长为2的卷积然后再进行ReLU运算后得到的。下面两点与[20]略有不同：(1)我们没有使用高分辨率的金字塔层P2，是因为计算量的原因；(2)P6是通过有步长的卷积计算得到的，而不是降采样；(3)我们纳入了P7，以改进大型目标的检测。这些小的改动在保持了准确率的同时，提升了速度）。

**Anchors**: We use translation-invariant anchor boxes similar to those in the RPN variant in [20]. The anchors have areas of $32^2$ to $512^2$ on pyramid levels P3 to P7, respectively. As in [20], at each pyramid level we use anchors at three aspect ratios {1:2, 1:1, 2:1}. For denser scale coverage than in [20], at each level we add anchors of sizes {$2^0 ,2^{1/3}, 2^{2/3}$} of the original set of 3 aspect ratio anchors. This improve AP in our setting. In total there are A = 9 anchors per level and across levels they cover the scale range 32 - 813 pixels with respect to the network’s input image.

**锚框**：我们使用平移不变的锚框，与RPN变体在[20]中的类似。这些锚框的区域大小从P3到P7分别为$32^2$到$512^2$。就像在[20]中一样的，在每个金字塔层级上，我们使用的锚框都有三个纵横比{1:2, 1:1, 2:1}。为比[20]中覆盖更密集的尺度，在每个层次我们在原始3个纵横比上增加锚框的大小{$2^0 ,2^{1/3}, 2^{2/3}$}。这在我们设置中改进了AP值结果。总计在每个层次上有A=9个锚框，在不同层次上，覆盖了32像素-813像素的尺度，根据输入图像大小不同而略有不同。

Each anchor is assigned a length K one-hot vector of classification targets, where K is the number of object classes, and a 4-vector of box regression targets. We use the assignment rule from RPN [28] but modified for multi-class detection and with adjusted thresholds. Specifically, anchors are assigned to ground-truth object boxes using an intersection-over-union (IoU) threshold of 0.5; and to background if their IoU is in [0, 0.4). As each anchor is assigned to at most one object box, we set the corresponding entry in its length K label vector to 1 and all other entries to 0. If an anchor is unassigned, which may happen with overlap in [0.4, 0.5), it is ignored during training. Box regression targets are computed as the offset between each anchor and its assigned object box, or omitted if there is no assignment.

每个锚框都指定了一个长度为K的独热向量，作为分类目标，其中K是目标类别的数量，还指定了4个坐标作为框回归的目标。我们使用RPN[28]中的指定规则，但对于多类别检测的情况进行了修正，也调整了阈值。特别的，锚框指定到真值目标框时，使用的IoU阈值为0.5；指定为背景时，其IoU是[0,0.4)。由于每个锚框最多指定一个目标框，我们在长度K的标注向量中设置对应的条目为1，其他条目为0。如果一个锚框并未指定，这在IoU [0.4,0.5)的情况下是可能发生的，那么就在训练时忽略掉。框回归的目标，是计算每个锚框及其指定的目标框的偏移，如果没有指定，那么就忽略掉。

Figure 3. The one-stage RetinaNet network architecture uses a Feature Pyramid Network (FPN) [20] backbone on top of a feedforward ResNet architecture [16] (a) to generate a rich, multi-scale convolutional feature pyramid (b). To this backbone RetinaNet attaches two subnetworks, one for classifying anchor boxes (c) and one for regressing from anchor boxes to ground-truth object boxes (d). The network design is intentionally simple, which enables this work to focus on a novel focal loss function that eliminates the accuracy gap between our one-stage detector and state-of-the-art two-stage detectors like Faster R-CNN with FPN [20] while running at faster speeds.

图3. 单阶段RetinaNet网络架构使用FPN[20]作为骨架网络，FPN位于一个前向的ResNet架构[16]之上，(a)生成丰富的、多尺度卷积特征金字塔；(b)在这个骨架RetinaNet上，连接着两个子网络，一个用于对锚框进行分类；(c)另一个用于从锚框到真值目标框进行回归；(d)网络故意设计的很简单，这使网络可以聚焦在新的损失函数的贡献，弥补了单阶段检测器和目前最好的两阶段检测器之间的间隙，如带有FPN[20]的Faster R-CNN，而且还能运行的更快。

**Classification Subnet**: The classification subnet predicts the probability of object presence at each spatial position for each of the A anchors and K object classes. This subnet is a small FCN attached to each FPN level; parameters of this subnet are shared across all pyramid levels. Its design is simple. Taking an input feature map with C channels from a given pyramid level, the subnet applies four 3×3 conv layers, each with C filters and each followed by ReLU activations, followed by a 3×3 conv layer with KA filters. Finally sigmoid activations are attached to output the KA binary predictions per spatial location, see Figure 3 (c). We use C = 256 and A = 9 in most experiments.

**分类子网络**：分类子网络，在A个锚框的每个空间位置，预测K个目标类别每种存在的概率。这个子网络是一个小的FCN，连接在每个FPN层上；这个子网络的参数在所有金字塔层中共享。其设计非常简单。从给定的金字塔层中以其C通道特征图为输入，子网络进行4层的3×3卷积层处理，每层都有C个滤波器，每层之后都有ReLU激活，随后是有KA个滤波器的3×3卷积层。最后在每个空间位置上，在KA个二值预测的输出上，接着的是一个sigmoid激活层，见图3(c)。在大部分试验中，我们使用C=256，A=9。

In contrast to RPN [28], our object classification subnet is deeper, uses only 3×3 convs, and does not share parameters with the box regression subnet (described next). We found these higher-level design decisions to be more important than specific values of hyperparameters.

与RPN[28]形成对比，我们的目标分类子网络更深，只使用了3×3卷积，与边界框回归子网络并不共享权重。我们发现，这些较高层的设计决定比具体超参数的取值更加重要。

**Box Regression Subnet**: In parallel with the object classification subnet, we attach another small FCN to each pyramid level for the purpose of regressing the offset from each anchor box to a nearby ground-truth object, if one exists. The design of the box regression subnet is identical to the classification subnet except that it terminates in 4A linear outputs per spatial location, see Figure 3 (d). For each of the A anchors per spatial location, these 4 outputs predict the relative offset between the anchor and the groundtruth box (we use the standard box parameterization from R-CNN [11]). We note that unlike most recent work, we use a class-agnostic bounding box regressor which uses fewer parameters and we found to be equally effective. The object classification subnet and the box regression subnet, though sharing a common structure, use separate parameters.

**边界框回归子网络**：与目标分类子网络并行的是，我们在每个金字塔层上还连接着一个小的FCN，其目标是每个锚框与最近的真值目标的偏移回归，如果对应的真值目标框存在的话。边界框回归子网络的设计与分类子网络大部分一样，除了其在每个空间位置上是4A个线性输出，见图3(d)。对于每个空间位置上A个锚框中的每一个，这4个输出预测的是锚框和真值框之间的偏移（我们使用R-CNN[11]中的标准框参数）。我们注意到，与多数最近的工作不同，我们使用了类别无关的边界框回归器，使用了更少的参数，我们发现同样有效。目标分类子网络和边界框回归子网络，虽然使用共同的框架，但是使用的是不同的参数。

### 4.1. Inference and Training 推理和训练

**Inference**: RetinaNet forms a single FCN comprised of a ResNet-FPN backbone, a classification subnet, and a box regression subnet, see Figure 3. As such, inference involves simply forwarding an image through the network. To improve speed, we only decode box predictions from at most 1k top-scoring predictions per FPN level, after thresholding detector confidence at 0.05. The top predictions from all levels are merged and non-maximum suppression with a threshold of 0.5 is applied to yield the final detections.

**推理**：RetinaNet形成一个单独的FCN，包含一个ResNet-FPN骨干网络，分类子网络，以及框回归子网络，见图3所示。这样，推理的问题就只是使一幅图像前向经过这个网络。为改进速度，我们在每个FPN层上只对排名前1k的预测解码其边界框预测，检测器上的置信度阈值为0.05。所有层次上的最高预测都合并到一起，然后使用阈值为0.5的NMS，以产生最终的预测。

**Focal Loss**: We use the focal loss introduced in this work as the loss on the output of the classification subnet. As we will show in §5, we find that γ = 2 works well in practice and the RetinaNet is relatively robust to γ ∈ [0.5, 5]. We emphasize that when training RetinaNet, the focal loss is applied to all ∼ 100k anchors in each sampled image. This stands in contrast to common practice of using heuristic sampling (RPN) or hard example mining (OHEM, SSD) to select a small set of anchors (e.g., 256) for each minibatch. The total focal loss of an image is computed as the sum of the focal loss over all ∼ 100k anchors, normalized by the number of anchors assigned to a ground-truth box. We perform the normalization by the number of assigned anchors, not total anchors, since the vast majority of anchors are easy negatives and receive negligible loss values under the focal loss. Finally we note that α, the weight assigned to the rare class, also has a stable range, but it interacts with γ making it necessary to select the two together (see Tables 1a and 1b). In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).

**Focal Loss**：我们在分类子网络中使用本文提出的focal loss。我们将在§5给出结果，我们发现γ = 2在实践中效果很好，RetinaNet在γ ∈ [0.5, 5]的范围内都比较稳定。我们强调，当训练RetinaNet的时候，focal loss在每一幅取样图像中的所有大约100k个锚框上都进行了使用。这在试验中与RPN使用的启发式取样，或SSD中使用的在线难分样本挖掘(OHEM)进行比较，OHEM对每个minibatch只选择了一小部分锚框（如256）。一幅图像的总计focal loss计算为所有大约100k个锚框的focal loss之和，并除以指定到真值边界框的锚框数量以归一化。我们用指定的锚框数量进行归一化，而非总计的锚框数量，因为大部分的锚框都是容易分类的负样本，在focal loss下都是可以忽略的损失值。最后我们注意到，指定给较少类别的权重α，也有一个稳定的范围，但α与γ相互作用，所以需要同时选择两个参数的值（见表1a和1b）。一般来说，如果γ增加的话，那么α应当略微下降（对于γ = 2, α取值最佳为0.25）。

**Initialization**: We experiment with ResNet-50-FPN and ResNet-101-FPN backbones [20]. The base ResNet-50 and ResNet-101 models are pre-trained on ImageNet1k; we use the models released by [16]. New layers added for FPN are initialized as in [20]. All new conv layers except the final one in the RetinaNet subnets are initialized with bias b = 0 and a Gaussian weight fill with σ = 0.01. For the final conv layer of the classification subnet, we set the bias initialization to b = − log((1 − π)/π), where π specifies that at the start of training every anchor should be labeled as foreground with confidence of ∼ π. We use π = .01 in all experiments, although results are robust to the exact value. As explained in §3.3, this initialization prevents the large number of background anchors from generating a large, destabilizing loss value in the first iteration of training.

**初始化**：我们使用ResNet-50-FPN和ResNet-101-FPN作为骨干网络进行试验。基准ResNet-50和ResNet-101是在ImageNet1k上进行预训练的；我们使用[16]中给出的模型。为FPN新增加的层的初始化与[20]相同。在RetinaNet子网络中，所有新的卷积层，除了最后一层，其初始化都是偏置b=0，权重为零均值Gauss随机值，σ = 0.01。对于分类子网络的最后卷积层，我们设偏置的初始值为b = − log((1 − π)/π)，其中π是在训练开始时指定的值，每个锚框都应当以概率大约为π标注为前景。我们在所有试验中使用π=0.01，但是结果对于具体的值还是很稳健的。如3.3节所解释，这种初始化使大量背景锚框，在训练的第一次迭代中，不会生成很大的不稳定的损失。

**Optimization**: RetinaNet is trained with stochastic gradient descent (SGD). We use synchronized SGD over 8 GPUs with a total of 16 images per minibatch (2 images per GPU). Unless otherwise specified, all models are trained for 90k iterations with an initial learning rate of 0.01, which is then divided by 10 at 60k and again at 80k iterations. We use horizontal image flipping as the only form of data augmentation unless otherwise noted. Weight decay of 0.0001 and momentum of 0.9 are used. The training loss is the sum the focal loss and the standard smooth L1 loss used for box regression [10]. Training time ranges between 10 and 35 hours for the models in Table 1e.

**优化**：RetinaNet使用SGD训练。我们使用同步SGD在8个GPU上进行训练，每个minibatch 16幅图像（每个GPU 2幅图像）。除了另有指定，所有模型都训练90k次迭代，初始学习率为0.01，在第60k和80k次迭代中学习率除以10。我们使用水平图像翻转作为图像扩充的方式，除非另有陈述。权重衰减为0.0001，动量0.9。训练损失为focal loss和用于边界框回归的标准的平滑L1损失[10]。对于表1e中的模型，训练时间在10小时到35小时之间。

## 5. Experiments 试验

We present experimental results on the bounding box detection track of the challenging COCO benchmark [21]. For training, we follow common practice [1, 20] and use the COCO trainval35k split (union of 80k images from train and a random 35k subset of images from the 40k image val split). We report lesion and sensitivity studies by evaluating on the minival split (the remaining 5k images from val). For our main results, we report COCO AP on the test-dev split, which has no public labels and requires use of the evaluation server.

我们在COCO基准测试[21]的边界框检测上给出试验结果。对于训练，我们遵循[1,20]的常用方法，使用COCO trainval35k分割（训练集的80k图像与验证集的40k图像的随机35k图像子集）。我们在minival分割（val中剩余的5k图像）上进行评估，给出损伤和敏感性的研究。我们主要的结果，都是在COCO test-dev分割上给出COCO AP结果，在这上面没有公开的标签，需要使用评估服务器。

### 5.1. Training Dense Detection

We run numerous experiments to analyze the behavior of the loss function for dense detection along with various optimization strategies. For all experiments we use depth 50 or 101 ResNets [16] with a Feature Pyramid Network (FPN) [20] constructed on top. For all ablation studies we use an image scale of 600 pixels for training and testing.

我们进行了几个试验，分析损失函数在密集检测上的表现，以及各种优化策略的表现。对于所有的试验，我们使用深度50或101的ResNet[101]，其上是特征金字塔网络(FPN)[20]的组合。所有的分离试验中，我们在训练和测试中使用的图像尺度为600像素。

**Network Initialization**: Our first attempt to train RetinaNet uses standard cross entropy (CE) loss without any modifications to the initialization or learning strategy. This fails quickly, with the network diverging during training. However, simply initializing the last layer of our model such that the prior probability of detecting an object is π = .01 (see §4.1) enables effective learning. Training RetinaNet with ResNet-50 and this initialization already yields a respectable AP of 30.2 on COCO. Results are insensitive to the exact value of π so we use π = .01 for all experiments.

**网络初始化**：我们对RetinaNet的第一次训练，使用的是标准交叉熵(CE)损失，没有修改初始化策略或学习策略。这很快失败了，网络在训练中发散。但是，对模型仅初始化最后一层，使检测到目标的先验概率为π = .01（见4.1节），就可以有效的进行训练了。使用ResNet-50训练RetinaNet，这种初始化已经在COCO熵得到了30.2 AP的不错结果。结果对于π的确切值并不敏感，所以我们在所有试验中都使用π = .01。

**Balanced Cross Entropy**: Our next attempt to improve learning involved using the α-balanced CE loss described in §3.1. Results for various α are shown in Table 1a. Setting α = .75 gives a gain of 0.9 points AP.

**均衡的交叉熵**：我们的下一个改进学习的尝试是使用3.1节中所述的α-均衡的交叉熵损失。各种α值的结果如表1a所示。设α=.75使AP提升了0.9个点。

**Focal Loss**: Results using our proposed focal loss are shown in Table 1b. The focal loss introduces one new hyperparameter, the focusing parameter γ, that controls the strength of the modulating term. When γ = 0, our loss is equivalent to the CE loss. As γ increases, the shape of the loss changes so that “easy” examples with low loss get further discounted, see Figure 1. FL shows large gains over CE as γ is increased. With γ = 2, FL yields a 2.9 AP improvement over the α-balanced CE loss.

**Focal Loss**：使用我们提出的focal loss得到的结果如表1b所示。Focal loss带来了一个新的超参数，聚焦参数γ，控制着调制项的强度。当γ=0时，我们的损失函数与交叉熵损失是一样的。当γ增加时，损失函数的形状也相应的变化，容易分类的样本的低损失函数进一步降低，见图1。γ增加时，FL比CE的提升更多。当γ=2时，FL比α-均衡的交叉熵损失带来了2.9 AP的改进。

For the experiments in Table 1b, for a fair comparison we find the best α for each γ. We observe that lower α’s are selected for higher γ’s (as easy negatives are down-weighted, less emphasis needs to be placed on the positives). Overall, however, the benefit of changing γ is much larger, and indeed the best α’s ranged in just [.25,.75] (we tested α ∈ [.01, .999]). We use γ = 2.0 with α = .25 for all experiments but α = .5 works nearly as well (.4 AP lower).

对于表1b中的试验，为公平对比，我们对于每个γ值，都找到了最佳的α值。我们观察到，更高的γ值，对应着更低的α值（由于容易分类的负样本的权重降低，所以也不需要对正样本强调的过多）。但总体上，改变γ的收益是更大的，确实最佳的α值是在[.25, .75]的范围内（对于α ∈ [.01, .999]都进行了测试）。我们在所有试验中使用γ = 2.0，α = .25的选择，但α = .5也同样很好（只低了.4 AP）。

**Analysis of the Focal Loss**: To understand the focal loss better, we analyze the empirical distribution of the loss of a converged model. For this, we take take our default ResNet-101 600-pixel model trained with γ = 2 (which has 36.0 AP). We apply this model to a large number of random images and sample the predicted probability for ∼ $10^7$ negative windows and ∼ $10^5$ positive windows. Next, separately for positives and negatives, we compute FL for these samples, and normalize the loss such that it sums to one. Given the normalized loss, we can sort the loss from lowest to highest and plot its cumulative distribution function (CDF) for both positive and negative samples and for different settings for γ (even though model was trained with γ = 2).

**Focal Loss的分析**：为更好的理解focal loss，我们分析了一个收敛模型的损失的经验分布。为此，我们使用了默认的ResNet-101 600像素模型，使用γ = 2进行训练，结果为36.0 AP。我们将这个模型用于大量随机图像，取样了大约∼ $10^7$的负样本窗口和∼ $10^5$个正样本窗口的预测概率。下一步，对于这些正样本和负样本，我们分别计算其FL，并对损失进行归一化，使其和为1。有了这些归一化的损失，我们可以对其从低到高排序，对于正样本和负样本，在各种不同的γ设置下（即使模型是在γ=2的情况下训练出来的），分别绘出其累加分布函数(CDF)。

Cumulative distribution functions for positive and negative samples are shown in Figure 4. If we observe the positive samples, we see that the CDF looks fairly similar for different values of γ. For example, approximately 20% of the hardest positive samples account for roughly half of the positive loss, as γ increases more of the loss gets concentrated in the top 20% of examples, but the effect is minor.

正样本和负样本的累加分布函数如图4所示。如果我们观察正样本，我们会发现CDF对于不同的γ值是非常类似的。比如，大约20%的最难分类的样本，贡献了大约一半的正样本损失，随着γ增加，损失越来越集中于样本的前20%上，但效果并不明显。

The effect of γ on negative samples is dramatically different. For γ = 0, the positive and negative CDFs are quite similar. However, as γ increases, substantially more weight becomes concentrated on the hard negative examples. In fact, with γ = 2 (our default setting), the vast majority of the loss comes from a small fraction of samples. As can be seen, FL can effectively discount the effect of easy negatives, focusing all attention on the hard negative examples.

γ对于负样本的效果则非常不同。对于γ=0，正样本和负样本的CDFs颇为类似。但是，随着γ增加，损失一下子就集中到了难分负样本上。实际上，对于γ=2（我们的默认设置上），绝大多数的损失都在很小一部分样本上。可以看到，FL可以有效的降低容易分类的负样本的效果，将所有的注意力都放在难以分类的负样本上。

Figure 4. Cumulative distribution functions of the normalized loss for positive and negative samples for different values of γ for a converged model. The effect of changing γ on the distribution of the loss for positive examples is minor. For negatives, however, increasing γ heavily concentrates the loss on hard examples, focusing nearly all attention away from easy negatives.

图4. 一个收敛模型，在不同γ值下，正样本和负样本的归一化损失的累积分类函数。改变γ对正样本的损失分布的影响是很小的。但对于负样本，增加γ值迅速将损失集中到了难分样本上，容易分类样本的损失几乎降到没有。

**Online Hard Example Mining (OHEM)**: [31] proposed to improve training of two-stage detectors by constructing minibatches using high-loss examples. Specifically, in OHEM each example is scored by its loss, non-maximum suppression (nms) is then applied, and a minibatch is constructed with the highest-loss examples. The nms threshold and batch size are tunable parameters. Like the focal loss, OHEM puts more emphasis on misclassified examples, but unlike FL, OHEM completely discards easy examples. We also implement a variant of OHEM used in SSD [22]: after applying nms to all examples, the minibatch is constructed to enforce a 1:3 ratio between positives and negatives to help ensure each minibatch has enough positives.

**在线难分样本挖掘(OHEM)**：[31]提出通过使用高损失的样本构建minibatch来改进两阶段检测器。特别的，在OHEM中，每个样本由其损失来打分，然后应用非最大抑制(NMS)，然后用最大损失的样本构建一个minibatch。NMS阈值和批数量都是可调节的参数。与focal loss很像，OHEM特别强调错误分类的样本，但与FL不一样的是，OHEM完全抛弃了容易分类的样本。我们还实现了SSD[22]中使用的OHEM变体：对所有的样本进行NMS，构建minibatch的原则是正样本和负样本的比例为1:3，以确保每个minibatch都有足够的正样本。

We test both OHEM variants in our setting of one-stage detection which has large class imbalance. Results for the original OHEM strategy and the ‘OHEM 1:3’ strategy for selected batch sizes and nms thresholds are shown in Table 1d. These results use ResNet-101, our baseline trained with FL achieves 36.0 AP for this setting. In contrast, the best setting for OHEM (no 1:3 ratio, batch size 128, nms of .5) achieves 32.8 AP. This is a gap of 3.2 AP, showing FL is more effective than OHEM for training dense detectors. We note that we tried other parameter setting and variants for OHEM but did not achieve better results.

我们在单阶段检测算法中测试了两个OHEM变体，算法都对应着很大的类别不均衡。原始OHEM策略和OHEM 1:3的策略选择批数量，以及不同nms的阈值的结果如表1d所示。这些结果使用ResNet-101，我们使用FL训练的基准使用这种设置得到了36.0 AP。比较起来，OHEM最好的设置（非1:3比例，批数量128，nms阈值0.5）得到了32.8 AP。这是3.2 AP的差距，显示FL在训练密集检测器上比OHEM更有效。要说明的是，我们尝试了其他的参数设置，和OHEM的变体，但是也没有取得更好的结果。

**Hinge Loss**: Finally, in early experiments, we attempted to train with the hinge loss [13] on $p_t$, which sets loss to 0 above a certain value of $p_t$. However, this was unstable and we did not manage to obtain meaningful results. Results exploring alternate loss functions are in the appendix.

**Hinge Loss**：最后，在早期的试验中，我们尝试用hinge loss[13]在$p_t$上训练，在$p_t$值高于一定值的时，损失为0。但是，这很不稳定，没有得到有意义的结果。附录中给出了探索其他损失函数得出的结果。

Table 1. Ablation experiments for RetinaNet and Focal Loss (FL). All models are trained on trainval35k and tested on minival unless noted. If not specified, default values are: γ = 2; anchors for 3 scales and 3 aspect ratios; ResNet-50-FPN backbone; and a 600 pixel train and test image scale. (a) RetinaNet with α-balanced CE achieves at most 31.1 AP. (b) In contrast, using FL with the same exact network gives a 2.9 AP gain and is fairly robust to exact γ/α settings. (c) Using 2-3 scale and 3 aspect ratio anchors yields good results after which point performance saturates. (d) FL outperforms the best variants of online hard example mining (OHEM) [31, 22] by over 3 points AP. (e) Accuracy/Speed trade-off of RetinaNet on test-dev for various network depths and image scales (see also Figure 2).

表1 RetinaNet和Focal Loss(FL)的分离试验。所有模型都用trainval35k进行训练，在minival上进行测试，除非另有陈述。如果没有指定，默认值为，γ = 2，锚框有3个尺度、3个纵横比，骨干网络为ResNet-50-FPN，训练和测试图像的尺度为600像素。(a)α-均衡的交叉熵损失的RetinaNet得到了最多31.1 AP；(b)比较之下，同样的网络，使用FL得到了2.9 AP的提升，对于不同的γ/α设置还是很稳健的；(c)使用2-3个尺度和3个纵横比的锚框得到很好的结果，在这之后性能会饱和；(d) FL超过了最好的OHEM变体[31,22]大约3点AP；(e)RetinaNet在test-dev上各种不同的网络深度和图像尺度上的准确率/速度折中。

(a) Varying α for CE loss (γ = 0)

α | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | ---
.10 | 0.0 | 0.0 | 0.0
.25 | 10.8 | 16.0 | 11.7
.50 | 30.2 | 46.7 | 32.8
.75 | 31.1 | 49.4 | 33.0
.90 | 30.8 | 49.7 | 32.3
.99 | 28.7 | 47.4 | 29.9
.999 | 25.1 | 41.7 | 26.1

(b) Varying γ for FL (w. optimal α)

γ | α | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | --- | ---
0 | .75 | 31.1 | 49.4 | 33.0
0.1 | .75 | 31.4 | 49.9 | 33.1
0.2 | .75 | 31.9 | 50.7 | 33.4
0.5 | .50 | 32.9 | 51.7 | 35.2
1.0 | .25 | 33.7 | 52.0 | 36.2
2.0 | .25 | 34.0 | 52.5 | 36.5
5.0 | .25 | 32.2 | 49.6 | 34.8

(c) Varying anchor scales and aspects

sc | ar | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | --- | ---
1 | 1 | 30.3 | 49.0 | 31.8
2 | 1 | 31.9 | 50.0 | 34.0
3 | 1 | 31.8 | 49.4 | 33.7
1 | 3 | 32.4 | 52.3 | 33.9
2 | 3 | 34.2 | 53.1 | 36.5
3 | 3 | 34.0 | 52.5 | 36.5
4 | 3 | 33.8 | 52.1 | 36.2

(d) FL vs. OHEM baselines (with ResNet-101-FPN)

method | batch size  | nms thr | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | --- | --- | ---
OHEM | 128 | .7 | 31.1 | 47.2 | 33.2
OHEM | 256 | .7 | 31.8 | 48.8 | 33.9
OHEM | 512 | .7 | 30.6 | 47.0 | 32.6
OHEM | 128 | .5 | 32.8 | 50.3 | 35.1
OHEM | 256 | .5 | 31.0 | 47.4 | 33.0
OHEM | 512 | .5 | 27.6 | 42.0 | 29.2
OHEM 1:3 | 128 | .5 | 31.1 | 47.2 | 33.2
OHEM 1:3 | 256 | .5 | 28.3 | 42.4 | 30.3
OHEM 1:3 | 512 | .5 | 24.0 | 35.5 | 25.8
FL | n/a | n/a | 36.0 | 54.9 | 38.7

(e) Accuracy/speed trade-off RetinaNet (on test-dev)

depth | scale | AP | $AP_{50}$ | $AP_{75}$ | $AP_S$ | $AP_M$ | $AP_L$ | time
--- | --- | --- | --- | --- | --- | --- | --- | ---
50 | 400 | 30.5 | 47.8 | 32.7 | 11.2 | 33.8 | 46.1 | 64
50 | 500 | 32.5 | 50.9 | 34.8 | 13.9 | 35.8 | 46.7 | 72
50 | 600 | 34.3 | 53.2 | 36.9 | 16.2 | 37.4 | 47.4 | 98
50 | 700 | 35.1 | 54.2 | 37.7 | 18.0 | 39.3 | 46.4 | 121
50 | 800 | 35.7 | 55.0 | 38.5 | 18.9 | 38.9 | 46.3 | 153
101 | 400 | 31.9 | 49.5 | 34.1 | 11.6 | 35.8 | 48.5 | 81
101 | 500 | 34.4 | 53.1 | 36.8 | 14.7 | 38.5 | 49.1 | 90
101 | 600 | 36.0 | 55.2 | 38.7 | 17.4 | 39.6 | 49.7 | 122
101 | 700 | 37.1 | 56.6 | 39.8 | 19.1 | 40.6 | 49.4 | 154
101 | 800 | 37.8 | 57.5 | 40.8 | 20.2 | 41.1 | 49.2 | 198

### 5.2. Model Architecture Design 模型架构设计

**Anchor Density**: One of the most important design factors in a one-stage detection system is how densely it covers the space of possible image boxes. Two-stage detectors can classify boxes at any position, scale, and aspect ratio using a region pooling operation [10]. In contrast, as one-stage detectors use a fixed sampling grid, a popular approach for achieving high coverage of boxes in these approaches is to use multiple ‘anchors’ [28] at each spatial position to cover boxes of various scales and aspect ratios.

**锚框密度**：单阶段检测系统的一个重要设计因素是，覆盖的可能的图像框空间有多密集。两阶段检测器可以使用区域pooling操作[10]，在任何位置、尺度和纵横比上对边界框进行分类。比较起来，单阶段检测器使用一个固定的采样网络，在这些方法中，边界框的广泛覆盖率的一个好办法是，在每个空间位置使用多个锚框[28]，覆盖各种尺度和纵横比的边界框。

We sweep over the number of scale and aspect ratio anchors used at each spatial position and each pyramid level in FPN. We consider cases from a single square anchor at each location to 12 anchors per location spanning 4 sub-octave scales ($2^{k/4}$, for k ≤ 3) and 3 aspect ratios [0.5, 1, 2]. Results using ResNet-50 are shown in Table 1c. A surprisingly good AP (30.3) is achieved using just one square anchor. However, the AP can be improved by nearly 4 points (to 34.0) when using 3 scales and 3 aspect ratios per location. We used this setting for all other experiments in this work.

我们看一看在FPN中，在每个金字塔层次的每个空间位置上，不同尺度和不同纵横比的锚框的数量。我们考虑在每个位置上的单个方形的锚框的情况，到每个位置12个锚框的情况，12个框包括了4个尺度($2^{k/4}$, for k ≤ 3)和3个纵横比[0.5, 1, 2]。使用ResNet-50的结果，如表1c所示。即使使用一个方形的锚框，也得到了很好的AP结果(30.3)。但是，在每个空间位置上使用3个尺度3个纵横比，可以将AP提高接近4个点（到34.0）。我们本文中的所有其他试验都使用这个设置。

Finally, we note that increasing beyond 6-9 anchors did not shown further gains. Thus while two-stage systems can classify arbitrary boxes in an image, the saturation of performance w.r.t. density implies the higher potential density of two-stage systems may not offer an advantage.

最后，我们说明，6-9个锚框以上的试验并没有得到更好的结果。所以虽然两阶段系统可以对图像中的任意框进行分类，但随着密度增加而得到的饱和性能说明，两阶段系统可能更高的密度并没不是一个优点。

**Speed versus Accuracy**: Larger backbone networks yield higher accuracy, but also slower inference speeds. Likewise for input image scale (defined by the shorter image side). We show the impact of these two factors in Table 1e. In Figure 2 we plot the speed/accuracy trade-off curve for RetinaNet and compare it to recent methods using public numbers on COCO test-dev . The plot reveals that RetinaNet, enabled by our focal loss, forms an upper envelope over all existing methods, discounting the low-accuracy regime. RetinaNet with ResNet-101-FPN and a 600 pixel image scale (which we denote by RetinaNet-101-600 for simplicity) matches the accuracy of the recently published ResNet-101-FPN Faster R-CNN [20], while running in 122 ms per image compared to 172 ms (both measured on an Nvidia M40 GPU). Using larger scales allows RetinaNet to surpass the accuracy of all two-stage approaches, while still being faster. For faster runtimes, there is only one operating point (500 pixel input) at which using ResNet-50-FPN improves over ResNet-101-FPN. Addressing the high frame rate regime will likely require special network design, as in [27], and is beyond the scope of this work. We note that after publication, faster and more accurate results can now be obtained by a variant of Faster R-CNN from [12].

**速度vs准确率**：骨干网络越深，得到的准确率越高，但推理速度也越慢。对于输入图像尺度也有类似的结论（定义为图像短边的像素数）。我们在表1e中给出这两个因素的影响。在图2中，我们绘制出了RetinaNet的速度/准确率折中曲线，并与两种最近的方法在COCO test-dev上使用公开数据进行了比较。绘图说明，RetinaNet使用focal loss形成了所有现有方法的上包线。使用ResNet-101-FPN的RetinaNet在600像素图像尺度上（表示为RetinaNet-101-600）与最近发表的ResNet-101-FPN Faster R-CNN[20]性能匹配，而运算速度达到了122ms每图像，比较起来后者为172ms（都在NVidia M40 GPU上度量得到）。使用更大的尺度使RetinaNet超过了所有两阶段方法的准确率，而速度依然更快。对于更快的速度来说，使用ResNet-50-FPN在500像素输入的超过了Res-101-FPN的性能。如果想要更高的处理速度，很可能需要特殊的网络设计，像[27]一样，这不在本文的讨论范围内。我们注意到，在本文发表后，[12]提出的Faster R-CNN的变体，得到了更快的运算速度和更准确的结果。

### 5.3. Comparison to State of the Art 与目前最好的方法的比较

We evaluate RetinaNet on the challenging COCO dataset and compare test-dev results to recent state-of-the-art methods including both one-stage and two-stage models. Results are presented in Table 2 for our RetinaNet-101-800 model trained using scale jitter and for 1.5× longer than the models in Table 1e (giving a 1.3 AP gain). Compared to existing one-stage methods, our approach achieves a healthy 5.9 point AP gap (39.1 vs. 33.2) with the closest competitor, DSSD [9], while also being faster, see Figure 2. Compared to recent two-stage methods, RetinaNet achieves a 2.3 point gap above the top-performing Faster R-CNN model based on Inception-ResNet-v2-TDM [32]. Plugging in ResNeXt-32x8d-101-FPN [38] as the RetinaNet backbone further improves results another 1.7 AP, surpassing 40 AP on COCO.

我们在COCO数据集上评估RetinaNet，并将其在test-dev上的结果与目前最好的方法进行比较，包括单阶段和两阶段的模型。结果如表2所示，包括我们的RetinaNet-101-800模型，使用尺度jitter进行了训练，比表1e中的模型长1.5倍（得到了1.3 AP的提升）。与现有的单阶段方法相比，我们的方法比第二名的DSSD[9]提升了5.9 AP(39.1 vs 33.2)，速度也更快一些，如图2所示。与最近的两阶段方法比，RetinaNet比表现最好的Faster R-CNN模型（基于Inception-ResNet-v2-TDM[32]）的AP仍然高2.3点。使用ResNeXt-32x8d-101-FPN[38]作为RetinaNet的骨干网络，可以进一步提升1.7 AP，在COCO上超过了40 AP。

Table 2. Object detection single-model results (bounding box AP), vs. state-of-the-art on COCO test-dev. We show results for our RetinaNet-101-800 model, trained with scale jitter and for 1.5× longer than the same model from Table 1e. Our model achieves top results, outperforming both one-stage and two-stage models. For a detailed breakdown of speed versus accuracy see Table 1e and Figure 2.

| | backbone | AP | $AP_{50}$ | $AP_{75}$ | $AP_S$ | $AP_M$ | $AP_L$
--- | --- | --- | --- | --- | --- | --- | ---
Faster R-CNN+++ [16] | ResNet-101-C4 | 34.9 | 55.7 | 37.4 | 15.6 | 38.7 | 50.9
Faster R-CNN w FPN [20] | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2
Faster R-CNN by G-RMI [17] | Inception-ResNet-v2 [34] | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0
Faster R-CNN w TDM [32] | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1
YOLOv2 [27] | DarkNet-19 [27] | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5
SSD513 [22, 9] | ResNet-101-SSD | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8
DSSD513 [9] | ResNet-101-DSSD | 33.2 | 53.3 | 35.2 | 13.0 | 35.4 | 51.1
RetinaNet (ours) | ResNet-101-FPN | 39.1 | 59.1 | 42.3 | 21.8 | 42.7 | 50.2
RetinaNet (ours) | ResNeXt-101-FPN | 40.8 | 61.1 | 44.1 | 24.1 | 44.2 | 51.2

## 6. Conclusion 结论

In this work, we identify class imbalance as the primary obstacle preventing one-stage object detectors from surpassing top-performing, two-stage methods. To address this, we propose the focal loss which applies a modulating term to the cross entropy loss in order to focus learning on hard negative examples. Our approach is simple and highly effective. We demonstrate its efficacy by designing a fully convolutional one-stage detector and report extensive experimental analysis showing that it achieves state-of-the-art accuracy and speed. Source code is available at https://github.com/facebookresearch/Detectron [12].

本文中，我们认为单阶段目标检测器不能超过最好的两阶段检测器的主要原因是类别不均衡。为解决这个问题，我们提出了focal loss，对交叉熵使用了一个调制项，以聚焦难以分类的负样本的学习。我们的方法非常间单高效。我们设计了一个全卷积的单阶段检测器，进行广泛的试验并分析，结果表明取得了目前最好的准确率和速度，证明了focal loss的有效性。源码可见与facebook的Detectron[12]。

## Appendix A: Focal Loss*

The exact form of the focal loss is not crucial. We now show an alternate instantiation of the focal loss that has similar properties and yields comparable results. The following also gives more insights into properties of the focal loss.

Focal loss的准确形式并不重要。我们给出focal loss的另一种形式，有着类似的性质，得到了可比的结果。下面也给出了focal loss性质的更多思考。

We begin by considering both cross entropy (CE) and the focal loss (FL) in a slightly different form than in the main text. Specifically, we define a quantity $x_t$ as follows: 我们首先考虑交叉熵损失和focal loss的略微不同的形式。特别的，我们定义$x_t$如下：

$$x_t = yx$$(6)

where y ∈ {±1} specifies the ground-truth class as before. We can then write $p_t = σ(x_t)$ (this is compatible with the definition of $p_t$ in Equation 2). An example is correctly classified when $x_t$ > 0, in which case $p_t$ > .5. 其中y ∈ {±1}表示真值类别。我们可以写成$p_t = σ(x_t)$（这与$p_t$在式2中的定义是兼容的）。样本正确分类的情况就是，$x_t$ > 0，$p_t$ > .5。

We can now define an alternate form of the focal loss in terms of $x_t$. We define $p^∗_t$ and $FL^∗$ as follows: 我们现在用$x_t$定义另一种形式的focal loss。我们定义$p^∗_t$和$FL^∗$如下：

$$p^∗_t = σ(γx t + β)$$(7)
$$FL^∗ = − log(p^∗_t )/γ$$(8)

$FL^∗$ has two parameters, γ and β, that control the steepness and shift of the loss curve. We plot $FL^∗$ for two selected settings of γ and β in Figure 5 alongside CE and FL. As can be seen, like FL, $FL^∗$ with the selected parameters diminishes the loss assigned to well-classified examples.

Figure 5. Focal loss variants compared to the cross entropy as a function of $x_t = yx$. Both the original FL and alternate variant $FL^∗$ reduce the relative loss for well-classified examples ($x_t$ > 0).

We trained RetinaNet-50-600 using identical settings as before but we swap out FL for $FL^∗$ with the selected parameters. These models achieve nearly the same AP as those trained with FL, see Table 3. In other words, $FL^∗$ is a reasonable alternative for the FL that works well in practice.

Table 3. Results of FL and $FL^∗$ versus CE for select settings.

loss | γ | β | AP | $AP_{50}$ | $AP_{75}$
--- | --- | --- | --- | --- | ---
CE | - | - | 31.1 | 49.4 | 33.0
FL | 2.0 | - | 34.0 | 52.5 | 36.5
FL* | 2.0 | 1.0 | 33.8 | 52.7 | 36.3
FL* | 4.0 | 0.0 | 33.9 | 51.8 | 36.4

We found that various γ and β settings gave good results. In Figure 7 we show results for RetinaNet-50-600 with $FL^∗$ for a wide set of parameters. The loss plots are color coded such that effective settings (models converged and with AP over 33.5) are shown in blue. We used α = .25 in all experiments for simplicity. As can be seen, losses that reduce weights of well-classified examples (x t > 0) are effective.

Figure 7. Effectiveness of $FL^∗$ with various settings γ and β. The plots are color coded such that effective settings are shown in blue.

More generally, we expect any loss function with similar properties as FL or $FL^∗$ to be equally effective.

## Appendix B: Derivatives