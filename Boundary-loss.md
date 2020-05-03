# Boundary loss for highly unbalanced segmentation

Hoel Kervadec et. al. 

## 0. Abstract

Widely used loss functions for convolutional neural network (CNN) segmentation, e.g., Dice or cross-entropy, are based on integrals (summations) over the segmentation regions. Unfortunately, for highly unbalanced segmentations, such regional losses have values that differ considerably – typically of several orders of magnitude – across segmentation classes, which may affect training performance and stability. We propose a boundary loss, which takes the form of a distance metric on the space of contours (or shapes), not regions. This can mitigate the difficulties of regional losses in the context of highly unbalanced segmentation problems because it uses integrals over the boundary (interface) between regions instead of unbalanced integrals over regions. Furthermore, a boundary loss provides information that is complimentary to regional losses. Unfortunately, it is not straightforward to represent the boundary points corresponding to the regional softmax outputs of a CNN. Our boundary loss is inspired by discrete (graph-based) optimization techniques for computing gradient flows of curve evolution. Following an integral approach for computing boundary variations, we express a non-symmetric L2 distance on the space of shapes as a regional integral, which avoids completely local differential computations involving contour points. This yields a boundary loss expressed with the regional softmax probability outputs of the network, which can be easily combined with standard regional losses and implemented with any existing deep network architecture for N-D segmentation. We report comprehensive evaluations on two benchmark datasets corresponding to difficult, highly unbalanced problems: the ischemic stroke lesion (ISLES) and white matter hyperintensities (WMH). Used in conjunction with the region-based generalized Dice loss (GDL), our boundary loss improves performance significantly compared to GDL alone, reaching up to 8% improvement in Dice score and 10% improvement in Hausdorff score. It also yielded a more stable learning process. Our code is publicly available.

CNN分割最常使用的损失函数，如Dice或交叉熵，是基于对分割区域的积分（求和）的。不幸的是，对于高度不均衡的分割，这样的区域性损失函数在不同的分割类别中的有很大差异，通常差了几个数量级，这可能会影响训练性能和稳定性。我们提出一种边缘损失，boundary loss，其形式是在边缘/形状空间中的距离度量，而并不是区域。这可以缓解高度非均衡分割问题的区域损失的难度，因为其使用了区域间的边缘（连接处）的积分，而不是在区域上的不均衡积分。而且，边缘损失boundary loss给出的信息，与区域性的损失是互补的。不幸的是，要表示CNN的区域softmax输出所对应的边缘点，要进行表示并不是很直观的。我们的boundary loss是受离散（基于图的）优化技术启发得到的，这个技术是用于曲线演化中计算梯度流用的。按照计算边缘变分的积分方法，我们在形状空间上将非对称的L2距离表示为一个区域积分，这避免了涉及到边缘点的完全局部的微分计算。这得到的boundary loss，是用网络的区域softmax概率输出表示的，这就可以很容易的与标准区域损失结合到一起，并可以用任何N-D分割的深度学习架构来实现。我们在两个基准数据集上给出了综合评估，对应着困难，高度非平衡的问题：缺血性中风损伤(ISLES)和白质过亮(WMH)。与基于区域的通用dice损失(GDL)一起，我们的boundary loss与只使用GDL相比，显著改进了性能，在dice分数上达到了8%的改进，在hausdorff分数上达到了10%的改进。同时这还带来了很稳定的学习过程。我们的代码已经开源。

**Keywords**: Boundary loss, unbalanced data, semantic segmentation, deep learning, CNN

## 1. Introduction

Recent years have witnessed a substantial growth in the number of deep learning methods for medical image segmentation (Litjens et al., 2017; Shen et al., 2017; Dolz et al., 2018; Ker et al., 2018). Widely used loss functions for segmentation, e.g., Dice or cross-entropy, are based on regional integrals, which are convenient for training deep neural networks. In practice, these regional integrals are summations over the segmentation regions of differentiable functions, each invoking directly the softmax probability outputs of the network. Therefore, standard stochastic optimizers such SGD are directly applicable. Unfortunately, difficulty occurs for highly unbalanced segmentations, for instance, when the size of target foreground region is several orders of magnitude less than the background size. For example, in the characterization of white matter hyperintensities (WMH) of presumed vascular origin, the foreground composed of WMH regions may be 500 times smaller than the background (see the typical example in Fig. 1). In such cases, quite common in medical image analysis, standard regional losses contain foreground and background terms that have substantial differences in their values, typically of several orders of magnitude, which might affect performance and training stability (Milletari et al., 2016; Sudre et al., 2017).

最近几年见证了在医学图像分割中，深度学习方法的急剧增长。分割中广泛使用的损失函数，如dice或交叉熵，都是基于区域积分的，这对于深度神经网络的训练非常方便。在实践中，这些区域积分是在可微分函数的分割区域上进行求和得到的，每个都直接使用了网络的softmax概率输出。因此，标准的随机优化器，如SGD，都是可以直接应用的。不幸的是，对于高度不均衡的分割会产生困难，如当目标前景区域的大小比背景区域大小小了几个数量级。比如，在推测的血管原点，白质亮化(WHM)的特征，组成前景的WMH区域，可能比背景区域小500倍（如图1中的典型样本）。在这种情况下，在医学图像分析中非常常见，标准的区域损失包括前景和背景项，其值差异会非常大，一般都会有几个数量级的差异，这会影响性能和训练的稳定性。

Figure 1: A visual comparison that shows the positive effect of our boundary loss on a validation data from the WMH dataset. Our boundary loss helped recovering small regions that were otherwise missed by the generalized Dice loss (GDL). Best viewed in colors.

Segmentation approaches based on convolutional neural networks (CNN) are typically trained by minimizing the cross-entropy (CE), which measures an affinity between the regions defined by probability softmax outputs of the network and the corresponding groundtruth regions. The standard regional CE has well-known drawbacks in the context of highly unbalanced problems. It assumes identical importance distribution of all the samples and classes. To achieve good generalization, it requires a large training set with balanced classes. For unbalanced data, CE typically results in unstable training and leads to decision boundaries biased towards the majority classes. Class-imbalanced learning aims at mitigating learning bias by promoting the importance of infrequent labels. In medical image segmentation, a common strategy is to re-balance class prior distributions by down-sampling frequent labels (Havaei et al., 2017; Valverde et al., 2017). Nevertheless, this strategy limits the information of the images used for training. Another common practice is to assign weights to the different classes, inversely proportional to the frequency of the corresponding labels (Brosch et al., 2015; Ronneberger et al., 2015; Kamnitsas et al., 2017; Long et al., 2015; Yu et al., 2017). In this scenario, the standard cross-entropy (CE) loss is modified so as to assign more importance to the rare labels. Although effective for some unbalanced problems, such weighting methods may undergo serious difficulties when dealing with highly unbalanced datasets, as seen with WMH segmentation. The CE gradient computed over the few pixels of infrequent labels is typically noisy, and amplifying this noise with a high class weight may lead to instability.

基于CNN的分割方法，一般通过最小化交叉熵来训练，这衡量的是区域和对应的真值区域的相似性，这是通过网络的概率softmax输出来定义的。标准的区域交叉熵，在高度非均衡的问题中，有很著名的缺点，因为其假设的是在所有的样本和类别中都有相同的重要性分布。为取得很好的泛化效果，这需要很大的训练集，类别还必须均衡。对于不均衡数据，CE一般训练不稳定，得到的决策边界对倾向于主要类别。类别不均衡学习，其目标是，通过增强不经常的标签的重要性，来弥补学习的偏差。在医学图像分割中，一种主要的策略是，通过对高频的标签进行下采样，来重新均衡类别先验分布。尽管如此，这种策略限制了用于训练的图像的信息。另一种常见的方法是，为不同的类别指定权重，与对应的类别频度成反比。在这种场景中，标准的CE损失要进行修改，以对稀有标签指定更多的重要性。虽然这对一些不均衡的问题是有效的，这种加权方法可能会在高度不均衡的数据集时，产生严重的困难，比如在WHM分割中看到的。在频度低的标签上的很少像素上计算的CE梯度，一般是含有噪声的，用一个很高的类别权重来放大这种噪声，会带来不稳定性。

The well-known Dice overlap coefficient was also adopted as a regional loss function, typically outperforming CE in unbalanced medical image segmentation problems (Milletari et al., 2016, 2017; Wong et al., 2018). Sudre et al. (Sudre et al., 2017) generalized the Dice loss (Milletari et al., 2016) by weighting according to the inverse of class-label frequency. Despite these improvements over CE (Milletari et al., 2016; Sudre et al., 2017), regional Dice losses may undergo difficulties when dealing with very small structures. In such highly unbalanced scenarios, mis-classified pixels may lead to large decreases of the loss, resulting in unstable optimization. Furthermore, Dice corresponds to the harmonic mean between precision and recall, implicitly using the arithmetic mean of false positives and false negatives. False positives and false negatives are, therefore, equally important when the true positives remain the same, making this loss mainly appropriate when both types of errors are equally bad. The recent research in (Salehi et al., 2017; Abraham and Khan, 2018) investigated losses based on the Tversky similarity index in order to provide a better trade-off between precision and recall. It introduced two parameters that control the importance of false positives and false negatives. Other recent advances in class-imbalanced learning for computer vision problems have been adopted in medical image segmentation. For example, inspired by the concept of focal loss (Lin et al., 2018), Dice and Tvserky losses have been extended to integrate a focal term, which is parameterized by a value that controls the importance between easy and hard training samples (Abraham and Khan, 2018; Wong et al., 2018). The main objective of these losses is to balance the classes not only by their relative class sizes, but also by the level of segmentation difficulty.

著名的Dice重叠系数也被用于一种区域损失函数，在不均衡的医学图像分割问题中一般会超过交叉熵。Sudre等通过根据类别标签频率的逆进行加权，将Dice损失函数一般化，尽管有这些交叉熵的改进，区域Dice损失在处理非常小的结构时，会遇到很多困难。在这种高度不均衡的场景中，误分类的像素可能会导致损失降低很多，从而得到不稳定的优化。而且，Dice对应着precision和recall的调和平均，隐式的使用了假阳性和假阴性的代数平均。假阳性和假阴性，在真阳性是一样的时候，因此是同等重要的，使这种损失的适用情况是，两种类型的错误的程度是差不多的时候。最近的工作研究了基于Tversky相似性索引的损失函数，以在precision和recall之间取得更好的折中。其引入了两个参数，控制着假阳性和假阴性的重要程度。在计算机视觉问题的类别不均衡的学习中，其他最近的进展也在医学图像分割中得到了采用。比如，受focal loss的启发，Dice损失和Tversky损失得到了拓展，加入了一个focal项，即有一个参数值控制容易的训练样本和困难的训练样本之间的重要性。这些损失的主要目标，是不仅从相对类别规模中平衡类别，还从分割难度水平中平衡类别。

Contributions 贡献

All the above-mentioned losses are region-based. In this paper, we propose a boundary loss that takes the form of a distance metric on the space of contours (or shapes), not regions. We argue that a boundary loss can mitigate the issues related to regional losses in highly unbalanced segmentation problems. Rather than using unbalanced integrals over the regions, a boundary loss uses integrals over the boundary (interface) between the regions. Furthermore, it provides information that is complimentary to regional losses. It is, however, challenging to represent the boundary points corresponding to the regional softmax outputs of a CNN. This difficulty may explain why boundary losses have been avoided in the context of deep segmentation networks. Our boundary loss is inspired by techniques in discrete (graph-based) optimization for computing gradient flows of curve evolution (Boykov et al., 2006). Following an integral approach for computing boundary variations, we express a non-symmetric L2 distance on the space of shapes (or contours) as a regional integral, which avoids completely local differential computations involving contour points. This yields a boundary loss expressed as the sum of linear functions of the regional softmax probability outputs of the network. Therefore, it can be easily combined with standard regional losses and implemented with any existing deep network architecture for N-D segmentation.

所有上述损失都是基于区域的。本文中，我们提出一种boundary损失，其形式是一种在轮廓（或形状）空间的距离度量，而不是区域空间中的。我们认为，boundary损失可以在高度不均衡的分割问题中缓解与区域性损失相关的问题。这种损失没有使用区域中的不均衡积分，而是使用区域间的边缘上的积分。而且，这种损失提供的是与区域性损失互补的信息。但是，用CNN的区域性softmax输出，表示边界点，是比较有挑战的。这种困难性，可能是boundary损失在深度分割网络中一直没有使用的原因。我们的boundary loss是受到曲线演化中计算梯度流的（基于图的）离散优化所启发的。按照计算边界变分的积分方法，我们将形状（或轮廓）空间中的非对称L2距离，表示成一种区域积分，这完全避免了采用轮廓点的局部差分计算。这样得到的boundary loss，是用网络的区域性softmax概率性输出的线性函数之和表示的。因此，这可以很容易的与标准的区域性损失进行结合，并用任何现有的N-D分割深度网络框架进行实现。

We evaluated our boundary loss in conjunction with the region-based generalized Dice loss (GDL) (Sudre et al., 2017) on two challenging and highly unbalanced segmentation problems – the Ischemic Stroke Lesion (ISLES) and the White Matter Hyperintensities (WMH) benchmark datasets. The results indicate that the proposed boundary loss provides a more stable learning process, and significantly outperforms GDL alone, yielding up to 8% improvement in Dice score and 10% improvement in Hausdorff score.

我们将boundary loss与基于区域的GDL，在两个很有挑战的、非常不均衡的分割问题中进行评估，即the Ischemic Stroke Lesion (ISLES)和the White Matter Hyperintensities (WMH)基准测试数据集。结果表明，提出的bounary loss可以得到一个更稳定的学习过程，并明显超过了单独的GDL，dice分数提高了8%，Hausdorff分数提高了10%。

## 2. Formulation

Let $I : Ω ⊂ R^{2,3} → R$ denotes a training image with spatial domain Ω, and g : Ω → {0,1} a binary ground-truth segmentation of the image: g(p) = 1 if pixel/voxel p belongs to the target region G ⊂ Ω (foreground region) and 0 otherwise, i.e., p ∈ Ω \ G (background region). Let $s_θ : Ω → [0, 1]$ denotes the softmax probability output of a deep segmentation network, and $S_θ ⊂ Ω$ the corresponding segmentation region: $S_θ = {p ∈ Ω|s_θ(p) ≥ δ}$ for some threshold δ. Widely used segmentation loss functions involve a regional integral for each segmentation region in Ω, which measures some similarity (or overlap) between the region defined by the probability outputs of the network and the corresponding ground-truth. In the two-region case, we have an integral of the general form $\int_Ω g(p)f(s_θ(p))dp$ for the foreground, and of the form  $\int_Ω (1−g(p))f(1−s_θ(p))dp$ for the background. For instance, the standard two-region cross-entropy loss corresponds to a summation of these two terms for f = −log(·). Similarly, the generalized Dice loss (GDL) (Sudre et al., 2017) involves regional integrals with f = 1, subject to some normalization, and is given as follows for the two-region case:

令$I : Ω ⊂ R^{2,3} → R$表示一个训练图像，其空域范围为Ω，g : Ω → {0,1}是图像的二值真值分割：如果像素/体素p属于目标区域G ⊂ Ω（前景区域），则g(p) = 1，否则等于0，即p ∈ Ω \ G（背景区域）。令$s_θ : Ω → [0, 1]$表示一个深度分割网络的softmax概率输出，且$S_θ ⊂ Ω$是对应的分割区域：对于某阈值，$S_θ = {p ∈ Ω|s_θ(p) ≥ δ}$。广泛使用的分割损失函数，都涉及到对每个分割区域Ω中的区域性积分，度量的是网络概率输出定义的区域，和对应的真值区域的相似度（或重叠部分）。在两区域的情况中，对前景，我们有通用的积分形式$\int_Ω g(p)f(s_θ(p))dp$，对于背景区域，有$\int_Ω (1−g(p))f(1−s_θ(p))dp$的形式。比如，标准的两个区域的交叉熵损失，对应着这两项的求和，其中f = −log(·)。类似的，一般化的Dice损失，GDL，用f=1改进了区域性积分，对于两个区域的情况，有下式：

$$L_{GD}(θ) = 1 - 2 \frac {w_G \int_{p∈Ω} g(p)s_θ(p)dp + w_B \int_{p∈Ω} (1 − g(p))(1 − s_θ(p))dp} {w_G \int_Ω [s_θ(p)+g(p)]dp+w_B \int_Ω [2−s_θ(p)−g(p)]dp}$$(1)

where coefficients $w_G = 1/(\int_{p∈Ω} g(p)dp)^2$ and $w_B = 1/(\int_Ω(1 − g(p))dp)^2$ are introduced to reduce the well-known correlation between the Dice overlap and region size.

其中系数$w_G = 1/(\int_{p∈Ω} g(p)dp)^2$和$w_B = 1/(\int_Ω(1 − g(p))dp)^2$的引入，是为了降低Dice重叠和区域大小之间的著名关联。

Regional integrals are widely used because they are convenient for training deep segmentation networks. In practice, these regional integrals are summations of differentiable functions, each invoking directly the softmax probability outputs of the network, $s_θ(p)$. Therefore, standard stochastic optimizers such SGD are directly applicable. Unfortunately, extremely unbalanced segmentations are quite common in medical image analysis, where, e.g., the size of the target foreground region is several orders of magnitude smaller than the background size. This represents challenging cases because the foreground and background terms have substantial differences in their values, which affects segmentation performance and training stability (Milletari et al., 2016; Sudre et al., 2017).

区域积分使用广泛，因为对于训练深度分割网络非常方便。实践中，这些区域积分是可微分函数的求和，每个都直接调用了网络的softmax概率输出，$s_θ(p)$。因此，标准随机优化器，如SGD，可以直接应用。不幸的是，极端不均衡的分割在医学图像分析中是非常常见的，如目标前景区域比背景区域小了几个数量级。这是很有挑战的情况，其中前景项和背景项对应的值有非常大的差异，这影响了分割的性能和训练的稳定性。

Our purpose is to build a boundary loss Dist($∂G,∂S_θ$), which takes the form of a distance metric on the space of contours (or region boundaries) in Ω, with ∂G denoting a representation of the boundary of ground-truth region G (e.g., the set of points of G, which have a spatial neighbor in background Ω\G) and $∂S_θ$ denoting the boundary of the segmentation region defined by the network output. On the one hand, a boundary loss should be able to mitigate the above-mentioned difficulties for unbalanced segmentations: rather than using unbalanced integrals within the regions, it uses integrals over the boundary (interface) between the regions. Furthermore, a boundary loss provides information that is different from and, therefore, complimentary to regional losses. On the other hand, it is not clear how to represent boundary points on $∂S_θ$ as a differentiable function of regional network outputs $s_θ$. This difficulty might explain why boundary losses have been, to the best of our knowledge, completely avoided in the context of deep segmentation networks.

我们的目的是构建一个boundary损失Dist($∂G,∂S_θ$)，其形式是Ω中的轮廓（或区域边缘）空间上的距离度量，其中∂G表示真值区域G的边缘（如，G的点中，与背景Ω\G有空间邻域的点集），$∂S_θ$表示网络输出定义的分割区域的边缘。一方面，boundary损失应当可以缓和不均衡分割的上述问题：没有使用区域中的不均衡积分，而使用的是区域之间的边缘（界面）上的积分。而且，boundary损失提供的信息与区域损失是不同的，因此是互补的。另一方面，怎样将$∂S_θ$的边缘点表示为网络区域输出$s_θ$的可微分函数，还不是很明确。这种困难可能解释了，boundary损失在深度分割网络中完全没有应用。

Our boundary loss is inspired from discrete (graph-based) optimization techniques for computing gradient flows of curve evolution (Boykov et al., 2006). Similarly to our problem, curve evolution methods require a measure for evaluating boundary changes (or variations). Consider the following non-symmetric L2 distance on the space of shapes, which evaluates the change between two nearby boundaries ∂S and ∂G (Boykov et al., 2006):

我们的boundary损失是受到曲线演化中计算梯度流的离散（基于图）优化技术所启发得到的。与我们的问题类似，曲线演化方法需要一个度量，来评估边界的变化（或变分）。考虑下面的形状空间的非对称L2距离，评估的是两个临近的边缘∂S和∂G的变化：

$$Dist(∂G,∂S) = \int_{∂G} ||y_{∂S}(p) − p||^2 dp$$(2)

where p ∈ Ω is a point on boundary ∂G and $y_{∂S}(p)$ denotes the corresponding point on boundary ∂S, along the direction normal to ∂G, i.e., $y_{∂S}(p)$ is the intersection of ∂S and the line that is normal to ∂G at p (See Fig. 2.a for an illustration). ||⋅|| denotes the L2 norm. In fact, this differential framework for evaluating boundary change is in line with standard variational curve evolution methods (Mitiche and Ben Ayed, 2011), which compute the motion of each point p on the evolving curve as a velocity along the normal to the curve at point p. Similarly to any contour distance invoking directly points on contour ∂S, expression (2) cannot be used directly as a loss for ∂S = $∂S_θ$. However, it is easy to show that the differential boundary variation in (2) can be expressed using an integral approach (Boykov et al., 2006), which avoids completely local differential computations involving contour points and represents boundary change as a regional integral:

其中p∈Ω是边缘∂G上的一个点，$y_{∂S}(p)$表示沿着∂G的法线方向上，在边缘∂S上的对应点，即，$y_{∂S}(p)$是∂S和∂G在p点的法线的交点（见图2a）。||⋅||表示L2范数。实际上，这种评估边缘变化的微分框架，与标准的变分曲线演化方法是一致的，这将演化曲线中每个点p的运动，视为曲线上p点的法线上的速度来计算。式(2)与任何直接调用边缘∂S上的点的边缘距离类似，不能直接用作损失函数∂S = $∂S_θ$。但是，很容易可以证明，(2)式中的微分边缘变分可以用积分方法进行表示，这就完全避免了与边缘点相关的局部微分计算，将边缘变化表示为区域积分：

$$Dist(∂G, ∂S) = 2 \int_{ΔS} D_G(q)dq$$(3)

where ∆S denotes the region between the two contours and $D_G : Ω → R^+$ is a distance map with respect to boundary ∂G, i.e., $D_G(q)$ evaluates the distance between point q ∈ Ω and the nearest point $z_{∂G}(q)$ on contour ∂G: $D_G(q) = ||q − z_{∂G}(q)||$. Fig. 2.b illustrates this integral framework for evaluating the boundary distance in Eq. (2). To show that Eq. (3) holds, it suffices to notice that integrating the distance map $2D_G(q)$ over the normal segment connecting a point p on ∂G and $y_{∂S} (p)$ yields $||y_{∂S} (p) − p||^2$. This follows directly from a variable change:

其中∆S表示两个边缘之间的区域，$D_G : Ω → R^+$是一个到边缘∂G的距离图，即，$D_G(q)$计算的是点q∈Ω与∂G上最近的点$z_{∂G}(q)$的距离：$D_G(q) = ||q − z_{∂G}(q)||$。图2b描述了计算式(2)的边缘距离的积分框架。为说明式(3)是对的，需要注意的是，将距离图$2D_G(q)$在连接∂G上的点p和$y_{∂S} (p)$的法相片段上进行积分，会得到$||y_{∂S} (p) − p||^2$。这就是一个简单的变量变换：

$$\int_p^{y_{∂S}(p)} 2D_G(q)dq = \int_0^{||y_{∂S}(p)-p||}2D_GdD_G = ||y_{∂S}(p) − p||^2$$

Thus, from Eq. (3), the non-symmetric L2 distance between contours in Eq. (2) can be expressed as a sum of regional integrals based on a level set representation of boundary ∂G:

因此，从(3)式可以得出，(2)式中边缘间的非对称的L2距离，可以表达为，基于边缘∂G的水平集表示的区域积分之和：

$$\frac{1}{2}Dist(∂G,∂S) = \int_S φ_G(q)dq − \int_G φ_G(q)dq = \int_Ω φ_G(q)s(q)dq − \int_Ω φ_G(q)g(q)dq$$(4)

where s : Ω → {0, 1} is binary indicator function of region S: s(q) = 1 if q ∈ S belongs to the target and 0 otherwise. $φ_G : Ω → R$ denotes the level set representation of boundary ∂G: $φ_G(q) = −D_G(q)$ if q ∈ G and $φ_G(q) = D_G(q)$ otherwise. Now, for $S = S_θ$, i.e., replacing binary variables s(q) in Eq. (4) by the softmax probability outputs of the network $s_θ(q)$, we obtain the following boundary loss which, up to a constant independent of θ, approximates boundary distance $Dist(∂G, ∂S_θ)$:

其中s: Ω→{0,1}是区域S的一个二值指示器函数：如果q∈S属于目标，则s(q) = 1，否则为0。$φ_G : Ω → R$表示边缘∂G的水平集表示：如果q ∈ G，则$φ_G(q) = −D_G(q)$，否则$φ_G(q) = D_G(q)$。现在，对于$S = S_θ$，即，将式(4)中的二值变量s(q)替换为网络$s_θ(q)$的softmax概率输出，我们得到下面的boundary loss，给定一个独立的常量θ，近似的boundary距离$Dist(∂G, ∂S_θ)$：

$$L_B(θ) = \int_Ω φG(q)s_θ(q)dq$$(5)

Notice that we omitted the last term in Eq. (4) as it is independent of network parameters. The level set function φG is pre-computed directly from the ground-truth region G. In practice, our boundary loss in Eq. (5) is the sum of linear functions of the regional softmax probability outputs of the network. Therefore, it can be easily combined with standard regional losses and implemented with any existing deep network architecture for N-D segmentation. In the experiments, we will use our boundary loss in conjunction with the regional generalized Dice loss:

注意我们忽略了式(4)的最后一项，因为与网络参数是无关的。水平集函数φG是直接从ground-truth区域G中预计算出来的。实践中，我们式(5)中的boundary loss是网络的区域性softmax概率输出的线性函数的相加。因此，可以很容易的与标准区域性损失结合，并用任何现有的N-D分割深度学习架构框架实现。实验中，我们会将boundary loss与区域性GDL结合使用：

$$αL_{GD} (θ) + (1 − α)L_B (θ)$$(6)

Finally, it is worth noting that our boundary loss uses ground-truth boundary information via pre-computed level-set function $φ_G(q)$, which encodes the distance between each point q and ∂G. In Eq. (5), the softmax for each point q is weighted by the distance function. Such distance-to-boundary information is omitted in widely used regional losses, where all the points within a given region are treated equally, independently of their distances from the boundary.

最后，值得注意的是，我们的boundary loss通过预先计算的水平集函数$φ_G(q)$来使用真值边界信息，这个函数里包含了每个点q与∂G的距离。在式(5)中，每个点q的softmax由距离函数加权。这种到边缘的距离的信息，在广泛使用的区域损失中被忽略了，在区域性损失中，在给定区域中的所有点都是统一对待的，与其到边缘的距离无关。

## 3. Experiments

### 3.1. Datasets

To evaluate the proposed boundary loss, we selected two challenging brain lesion segmentation tasks, each corresponding to highly unbalanced classes.

为评估提出的boundary loss，我们选择了两个很有挑战的大脑损伤分割任务，每个都对应着高度不均衡的类别。

**ISLES**: The training dataset provided by the ISLES organizers is composed of 94 ischemic stroke lesion multi-modal scans. In our experiments, we split this dataset into training and validation sets containing 74 and 20 examples, respectively. Each scan contains Diffusion maps (DWI) and Perfusion maps (CBF, MTT, CBV, Tmax and CTP source data), as well as the manual ground-truth segmentation. More details can be found in the ISLES website.

**ISLES**：ISLES组织者提供的训练数据集由94个缺血性中风损伤的多模态scans组成。在我们的实验中，我们将这个数据集分成训练集和验证集，分别包含74个样本和20个样本。每个scan包含Diffusion maps (DWI)和Perfusion maps (CBF, MTT, CBV, Tmax and CTP source data)，以及手动真值分割。详见ISLES网站。

**WMH**: The public dataset of the White Matter Hyperintensities (WMH) MICCAI 2017 challenge contains 60 3D T1-weighted scans and 2D multi-slice FLAIR acquired from multiple vendors and scanners in three different hospitals. In addition, the ground truth for the 60 scans is provided. From the whole set, 50 scans were used for training, and the remaining 10 for validation.

**WMH**：MICCAI 2017中白质过亮(WMH)的公开数据集包含60个3D T1加权的扫描和2D多slice FLAIR，是从多个供应商和scanners在三个不同的医院得到的。另外，提供了60个scans的真值。对整个数据集，50个scans进行了训练，剩下的10个进行验证。

### 3.2. Implementation details

**Data pre-processing**. While the scans are provided as 3D images, we process them as a stack of independent 2D images, which are fed into the network. In fact, the scans in some datasets, such as ISLES, contain between 2 and 16 slices, making them ill-suited for 3D convolutions in those cases. The scans were normalized between 0 and 1 before being saved as a set of 2D matrices, and re-scaled to 256×256 pixels if needed. When several modalities were available, all of them were concatenated before being used as input to the network. We did not use any data augmentation in our experiments.

**数据预处理**。这些scans是以3D图像的形式提供的，但我们将其作为独立的2D图像的堆叠进行处理，然后送入网络。实际上，一些数据集中的scans，如ISLES，包含的slices从2到16都有，这对于3D卷积来说就是不适宜的。这些scans归一化到0到1范围内，然后存储到一系列2D矩阵中，然后如果有需要，改变其大小到256×256像素。当有几种模态可用时，所有模态都拼接起来，然后用作网络输入。我们在实验中不进行任何数据扩充。

**Architecture and training**. We employed UNet (Ronneberger et al., 2015) as deep learning architecture in our experiments. To train our model, we employed Adam optimizer, with a learning rate of 0.001 and a batch size equal to 8. The learning rate is halved if the validation performances do not improve during 20 epochs. We did not use early stopping.

**架构和训练**。我们在实验中采用U-Net作为深度学习架构。为训练我们的模型，我们采用Adam优化器，学习速率为0.001，batch size为8。如果验证性能在20轮数据中都没有改进，那么学习速率就减半。我们没有使用早停。

To compute the level set function φG in Eq. (5), we used standard SciPy functions. Note that, for slices containing only the background region, we used a zero-distance map, assuming that the GDL is sufficient in those cases. Furthermore, during training, the value of α in Eq. (6) was initially set to 1, and decreased by 0.01 after each epoch, following a simple scheduling strategy, until it reached the value of 0.01. In this way, we give more importance to the regional loss term at the beginning while gradually increasing the impact of the boundary loss term. We empirically found that this simple scheduling strategy was less sensitive to the choice of α while giving consistently similar or better results than a constant value. In addition, we evaluated the performance when the boundary loss is the only objective, i.e., α = 0.

为计算式(5)中的水平集函数，我们使用标准的SciPy函数。注意，对于只包含背景区域的slices，我们使用的是一种zero-distance图，假设在这种情况中GDL是充分的。而且，在训练中，式(6)中的α值初始化为1，每过一轮就减少0.01，按照这个简单的计划策略，直到其值达到0.01。以这种方式，我们在初始的时候，给区域性损失更多的重要性，同时逐渐增加boundary loss项的重要性。我们通过经验发现，这种简单的计划策略对α值的选择不那么敏感，同时比常数值给出了类似的或更好的结果。而且，我们在boundary loss是唯一目标函数时也评估了其性能，即α=0的情况。

For our implementation, we used PyTorch (Paszke et al., 2017), and ran the experiments on a machine equipped with an NVIDIA GTX 1080 Ti GPU with 11GBs of memory. Our code (data pre-processing, training and testing scripts) is publicly available.

我们使用PyTorch进行实现，运行实验的机器的显卡为NVIDIA GTX 1080 Ti，有11GB显存。代码已经开源，包含数据预处理，训练和测试。

**Evaluation**. For evaluation purposes, we employ the common Dice Similarity Coefficient (DSC) and Hausdorff Distance (HD) metrics. 为进行评估，我们采用通用的Dice相似性系数和Hausdorff距离作为度量。

### 3.3. Results

**Quantative evaluation**. Table 1 reports the DSC and HD performance for our experiments using GDL alone and the loss we proposed in Eq. (6) on the ISLES and WMH datasets. Adding our boundary loss term to the GDL consistently improves segmentation performance, which is reflected in significantly higher DSC and HD values. While this boundary loss term brings a DSC improvement of around 2% on the WMH dataset, it achieves 8% better DSC on the ISLES segmentation task. The same trend is observed with the HD metric, where the gain is larger on the ISLES dataset than on WMH.

**定量的评估**。表1给出了我们实验的DSC和HD结果，使用的损失包括GDL和提出的(6)式的损失，数据集为ISLES和WMH。GDL与boundary loss一起，可以改进分割性能，其DSC和HD值要高很多。Boundary loss项在WMH数据集上可以带来大约2%的DSC改进，在ISLES分割任务上的DSC提高了8%。HD度量得到了相同的趋势，在ISLES数据集上的提升比WMH上要大。

Table 1: DSC and HD values achieved on the validation subset. The values represent the mean performance (and standard deviation) of 2 runs for each setting.

Loss | ISLES DSC | ISLES HD(mm) | WMH DSC | WMH HD(mm)
--- | --- | --- | --- | ---
LB | 0.321(0.000) | NA | 0.569(0.000) | NA
LGD | 0.575(0.028) | 4.009(0.016) | 0.727(0.006) | 1.045 (0.014)
LGD + LB | 0.656(0.023) | 3.562(0.009) | 0.748(0.005) | 0.987 (0.010)

Using the boundary loss alone does not yield the same competitive results as a joint loss (i.e., boundary and region), making the network collapse quickly into empty foreground regions, i.e., softmax predictions close to zero. We believe that this is due to the following technical facts. In theory, the global optimum of the boundary loss corresponds to a negative value, as a perfect overlap sums only over the negative values of the distance map. In this case, the softmax probabilities correspond to a non-empty foreground. However, an empty foreground (null values of the softmax probabilities almost everywhere) corresponds to low gradients. Therefore, this trivial solution is close a local minimum or a saddle point. This is not the case when we use our boundary loss in conjunction with a regional loss, which guides the training during the first epochs and avoids getting stuck in such a trivial solution. The scheduling method then increases the weight of the boundary loss, with the latter becoming very dominant towards the end of the training process. This behaviour of boundary terms is conceptually similar to the behaviour of classical and popular contour-based energies for level set segmentation, e.g., geodesic active contours (Caselles et al., 1997), which also require additional regional terms to avoid trivial solutions (i.e., empty foreground regions).

只使用boundary loss，并没有使用联合损失（即，boundary loss和区域性损失）效果好，这会使网络迅速坍缩成空的前景区域，即softmax预测接近于0。我们相信，这是因为下列事实。理论上，boundary loss的全局最优解对应着一个负值，因为一个完美的重叠，其只在距离图的负值上求和。在这种情况下，softmax概率对应一个非空的前景。但是，空的前景（在几乎所有地方其softmax概率都为空值）对应的梯度值很低。因此，这种无意义的解，是一个局部最优值或是一个鞍点值。但当我们将boundary loss与区域性损失一起使用时，就不一样了，区域性损失会在第一轮数据训练时进行引导，不会停留在无意义的解上。这种方法增加了boundary loss的权值，在训练过程的最后阶段，后者的权重已经占主要作用。Boundary项的这种行为，概念上与经典流行的基于contour能量的水平集分割类似，如，geodesic active contours，这也需要额外的区域性项来防止出现无意义解（即，空的前景区域）。

The learning curves depicted in Figure 3 show the gap in performance between the GDL alone and the GDL augmented with our boundary loss, with the difference becoming significant at convergence. In addition to outperforming GDL, we can also observe that the boundary loss term helps stabilizing the training process, yielding a much smoother curve as the network training converges. This behaviour is consistent for both metrics and both dataset, which clearly shows the benefits of employing the proposed boundary loss term.

图3所示的学习曲线表明，只使用GDL进行训练，和使用GDL+boundary loss得到的性能差距，在收敛的时候，差距会变得很明显。除了超过GDL，我们还可以观察到，boundary loss项帮助稳定了训练过程，在网络训练收敛时，得到了更平滑的曲线。这种行为在不同的度量标准和不同的数据集上都是一致的，清楚的说明了采用提出的boundary loss项的好处。

Figure 3: Evolution of DSC and HD values on the validation subset when training on ISLES and WMH dataset. The blue curve shows the performance of the network trained using the GDL loss, while the red curve represents the optimization process with the GDL + our proposed boundary loss term.

**Qualitative evaluation**. Qualitative results are depicted in Fig. 4. Inspecting these results visually, we can observe that there are two major types of improvements when employing the proposed boundary loss. First, as the methods based on DSC losses, such as GDL, do not use spatial information, prediction errors are treated equally. This means that the errors for pixels/voxels in an already detected object have the same importance as the errors produced in completely missed objects. On the contrary, as our boundary loss is based on the distance map from the ground-truth boundary ∂G, it will penalize much more such cases, helping to recover small and far regions. This effect is best illustrated in Fig. 1 and Fig. 4 (third row). False positives (first row in Fig. 4) will be far away from the closest foreground, getting a much higher penalty than with the GDL alone. This helps in reducing the number of false positives.

**定性评估**。定性的结果如图4所示。从视觉上观察这些结果，我们可以观察到，在采用了提出的boundary loss时，有两种主要的改进。第一，基于DSC损失的方法，如GDL，并没有使用空间信息，预测的错误是一样的对待的。这意味着，在已经预测的目标中，对于pixels/voxels的错误，与在完全错误的目标中预测的错误，有相同的重要性。相反的，由于我们的boundary loss是基于距离真值边缘∂G的距离图的，则会对这种情况进行更多的惩罚，帮助恢复更小更远的区域。这种效果在图1和图4的第三行表现的最好。假阳性（在图4中的第一行）与最接近的前景会很远，与GDL相比，会得到高的多的惩罚。这帮助降低了假阳性的数量。

Figure 4: Visual comparison on two different datasets from the validation set.

**Computational complexity**. It is worth mentioning that, as the proposed boundary loss term involves an element-wise product between two matrices – i.e., the pre-computed level-set function $φ_G$ and the softmax output $s_θ(p)$ – the complexity that it adds is negligible.

**计算复杂度**。值得提到的是，由于提出的boundary loss项包含两个矩阵的逐元素乘积，即，预计算的水平集函数$φ_G$和softmax输出$s_θ(p)$，所以其增加的计算复杂度是可以忽略的。

## 4. Conclusion

We proposed a boundary loss term that can be easily combined with standard regional losses to tackle the segmentation task in highly unbalanced scenarios. Furthermore, the proposed term can be implemented in any existing deep network architecture and for any N-D segmentation problem. Our experiments on two challenging and highly unbalanced datasets demonstrated the effectiveness of including the proposed boundary loss term during training. It consistently improved the performance, with a large margin on one data set, and enhanced training stability. Even though we limited the experiments to 2-D segmentation problems, the proposed framework can be trivially extended to 3-D, which could further improve the performance of deep networks, as more context is analyzed.

我们提出了一种boundary loss项，可以很容易的与标准区域性损失结合到一起，以解决高度不均衡场景的分割任务。而且，提出的项可以用任何现有的深度学习架构实现，是一个N-D分割问题。我们在两个很有挑战的高度不均衡的数据集上的试验，证明了在训练时包含提出的boundary loss项的有效性。其可以一直提高性能，在一个数据集上可以提升很多，并增强了训练稳定性。我们将试验限制在2D分割问题，提出的框架可以很容易的拓展到3D中，这可以进一步改进深度网络的性能，因为分析了更多的上下文。