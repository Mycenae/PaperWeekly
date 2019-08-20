# U-Net: Convolutional Networks for Biomedical Image Segmentation

Olaf Ronneberger et al. University of Freiburg, Germany

## Abstract 摘要

There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.

深度网络的成功训练，需要很多标注的训练样本。本文中，我们提出了一种网络和训练策略，在可用的标注样本上，进行了很强的数据扩充，从而更高效的利用了数据。这个架构包含了收缩的路径以捕获上下文，和一个对称的扩展路径，可以进行精确的定位。我们证明了，这样一种网络可以用很少的图像在端到端训练后，在ISBI神经结构分割挑战赛上，性能超过之前最好的方法（是一种滑窗卷积网络）。使用同样的网络，在光传输显微图像（相位对比和DIC）上进行训练，我们大幅度赢得了ISBI 2015细胞追踪挑战。而且，这个网络很快。512×512图像的分割任务在GPU上耗时不到1秒。完整实现（基于Caffe）和训练好的网络已开源。

## 1 Introduction 引言

In the last two years, deep convolutional networks have outperformed the state of the art in many visual recognition tasks, e.g. [7,3]. While convolutional networks have already existed for a long time [8], their success was limited due to the size of the available training sets and the size of the considered networks. The breakthrough by Krizhevsky et al. [7] was due to supervised training of a large network with 8 layers and millions of parameters on the ImageNet dataset with 1 million training images. Since then, even larger and deeper networks have been trained [12].

过去两年，深度卷积网络在很多视觉识别任务中超过了目前最好的方法，如[7,3]。虽然卷积网络已经存在了很长时间，但其性能一直受限于可用的训练集大小和网络规模的大小。[7]的突破是因为网络深度达到了8层，参数达到了数百万，在ImageNet数据集上进行了监督训练，其中有一百万训练图像。从此以后，训练了更大的更深的网络[12]。

The typical use of convolutional networks is on classification tasks, where the output to an image is a single class label. However, in many visual tasks, especially in biomedical image processing, the desired output should include localization, i.e., a class label is supposed to be assigned to each pixel. Moreover, thousands of training images are usually beyond reach in biomedical tasks. Hence, Ciresan et al. [1] trained a network in a sliding-window setup to predict the class label of each pixel by providing a local region (patch) around that pixel as input. First, this network can localize. Secondly, the training data in terms of patches is much larger than the number of training images. The resulting network won the EM segmentation challenge at ISBI 2012 by a large margin.

卷积网络的典型应用是在分类任务上，输入是一幅图像，输出是单个类别标签。但是，在很多视觉任务中，尤其是在生物医学图像处理中，期望的输出应当包括定位，即，一个类别标签应当指定到每个像素。而且，在生物医学任务中，数千幅图像通常是很难得到的。因此，Ciresan等[1]以滑窗的设置训练了一个网络，类别标签预测到每个像素，因为在每个像素周围都给出了一个局部区域作为输入。首先，这个网络可以定位。第二，以图像块为单位的训练数据比训练图像数量要大的多。得到的网络大幅赢得了ISBI 2012 EM分割挑战赛。

Obviously, the strategy in Ciresan et al. [1] has two drawbacks. First, it is quite slow because the network must be run separately for each patch, and there is a lot of redundancy due to overlapping patches. Secondly, there is a trade-off between localization accuracy and the use of context. Larger patches require more max-pooling layers that reduce the localization accuracy, while small patches allow the network to see only little context. More recent approaches [11,4] proposed a classifier output that takes into account the features from multiple layers. Good localization and the use of context are possible at the same time.

显然，Ciresan等[1]的策略有两个缺点。首先，算法非常慢，因为对每个图像块都要分别运行网络，由于图像块的重叠，所以有很多冗余性。第二，在定位准确性和上下文的使用上有折中。更大的图像块需要更多的最大池化层，这降低了定位准确率，而小型图像块使得网络只能看到很小的上下文。最近的方法[11,4]提出了一个分类器输出，考虑了多层的特征。这就可能同时得到很好的定位和上下文的使用。

In this paper, we build upon a more elegant architecture, the so-called “fully convolutional network” [9]. We modify and extend this architecture such that it works with very few training images and yields more precise segmentations; see Figure 1. The main idea in [9] is to supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators. Hence, these layers increase the resolution of the output. In order to localize, high resolution features from the contracting path are combined with the upsampled output. A successive convolution layer can then learn to assemble a more precise output based on this information.

本文中，我们的架构则更优雅，也就是所谓的全卷积网络。我们修改并拓展了这个架构，使其可以在很少训练图像上工作，得到更精确的分割；如图1所示。[9]中的主要思想是在后续层中附加了一个普通的收缩网络，其中的池化算子替换为了上采样算子。因此，这些层增大了输出的分辨率。为更好的定位，收缩路径上的高分辨率特征与上采样的输出结合到了一起。后续的卷积层可以学习与在这些信息基础上的更精确的输出结合起来。

Fig. 1. U-net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.

One important modification in our architecture is that in the upsampling part we have also a large number of feature channels, which allow the network to propagate context information to higher resolution layers. As a consequence, the expansive path is more or less symmetric to the contracting path, and yields a u-shaped architecture. The network does not have any fully connected layers and only uses the valid part of each convolution, i.e., the segmentation map only contains the pixels, for which the full context is available in the input image. This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy (see Figure 2). To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image. This tiling strategy is important to apply the network to large images, since otherwise the resolution would be limited by the GPU memory.

我们架构中的一个重要修改是，在上采样部分，我们也有很多特征通道数，这使网络可以讲上下文信息传播到更高分辨率的层。结果是，扩展通道与收缩通道大致是对称的，得到了一个U形的架构。网络没有任何全连接层，只使用了每个卷积的有效部分，即，分割图只包含了其完整上下文在输入图像中可用的像素。这个策略利用了一种重叠块的策略，可以在任意大小的图像上进行无缝的分割（见图2）。为预测图像边缘区域的像素，这些缺失的上下文，通过将输入图像进行镜像来进行外插值。这种并列平铺的策略对于将网络用于大型图像非常重要，因为如果不这样的话，分辨率就会受到GPU内存的限制。

As for our tasks there is very little training data available, we use excessive data augmentation by applying elastic deformations to the available training images. This allows the network to learn invariance to such deformations, without the need to see these transformations in the annotated image corpus. This is particularly important in biomedical segmentation, since deformation used to be the most common variation in tissue and realistic deformations can be simulated efficiently. The value of data augmentation for learning invariance has been shown in Dosovitskiy et al. [2] in the scope of unsupervised feature learning.

而我们的任务中，可用的训练数据是很少的，所以我们使用了过度数据扩充技术，对可用的训练图像使用了弹性变形。这使得网络可以学习到对这种变形的不变性，而不需要在标注的图像集中看到这些变形。这对生物医学图像分割非常重要，因为变形是组织中的最常见变化，这样实际的变形可以得到很好的模拟。在学习不变性中数据扩充的价值，在Dosovitskiy等[2]中已经有了展示，这是在无监督特征学习的范畴中。

Another challenge in many cell segmentation tasks is the separation of touching objects of the same class; see Figure 3. To this end, we propose the use of a weighted loss, where the separating background labels between touching cells obtain a large weight in the loss function.

在很多细胞分割任务中，另一个挑战是相同类别的接触目标的分离；如图3所示。为此，我们提出使用一种加权损失，其中接触的细胞间的分离背景标签在损失函数中有很大的权重。

The resulting network is applicable to various biomedical segmentation problems. In this paper, we show results on the segmentation of neuronal structures in EM stacks (an ongoing competition started at ISBI 2012), where we outperformed the network of Ciresan et al. [1]. Furthermore, we show results for cell segmentation in light microscopy images from the ISBI cell tracking challenge 2015. Here we won with a large margin on the two most challenging 2D transmitted light datasets.

得到的网络对多种生物医学分割问题都可以应用。在本文中，我们给出了EM栈上的神经元结构的分割结果（自从ISBI 2012就开始进行的一个比赛），我们超过了Ciresan等[1]网络的结果。而且，我们给出了光学显微镜图像中的细胞分割结果，图像来自ISBI细胞跟踪挑战2015。这里我们在两个最有挑战性的2D传输光学数据集上以大幅度超过了其他算法。

## 2 Network Architecture 网络架构

The network architecture is illustrated in Figure 1. It consists of a contracting path (left side) and an expansive path (right side). The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.

网络架构如图1所示。包含了一条收缩路径（左边）和一条扩张路径（右边）。收缩路径就是一个卷积网络的典型架构，包含了2个3×3卷积层（未补零卷积）的重复堆叠，每个后面都跟着一个ReLU和一个2×2最大池化运算，步长为2以进行下采样。在每个下采样步骤中，我们将特征通道数量加倍。扩张通道的每个步骤包含一个特征图的上采样，随后是一个2×2卷积（上卷积），特征通道数量减半，然后与收缩通道中对应的剪切特征图拼接，2个3×3卷积，每个后面都有ReLU。剪切是很有必要的，因为每个卷积中都丢失了边缘像素。在最后的层中，使用了一个1×1卷积将每个64个元素的特征向量映射到期望的类别数量。网络总计有23个卷积层。

To allow a seamless tiling of the output segmentation map (see Figure 2), it is important to select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size.

为将输出分割图进行无缝平铺（见图2），选择输入的平铺尺寸就很重要了，这样所有的2×2最大池化运算应用的层中，x和y的大小很均衡。

Fig. 1. U-net architecture (example for 32x32 pixels in the lowest resolution). Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the box. The x-y-size is provided at the lower left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations.

Fig. 2. Overlap-tile strategy for seamless segmentation of arbitrary large images (here segmentation of neuronal structures in EM stacks). Prediction of the segmentation in the yellow area, requires image data within the blue area as input. Missing input data is extrapolated by mirroring.

## 3 Training

The input images and their corresponding segmentation maps are used to train the network with the stochastic gradient descent implementation of Caffe [6]. Due to the unpadded convolutions, the output image is smaller than the input by a constant border width. To minimize the overhead and make maximum use of the GPU memory, we favor large input tiles over a large batch size and hence reduce the batch to a single image. Accordingly we use a high momentum (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.

输入图像及其对应的分割图用于训练网络，使用Caffe实现的SGD。由于卷积未补零，输出图像是比输入要小的，小一个常数边界宽度。为最小化计算开支，最大化使用GPU内存，我们倾向于大批次数量上的大输入堆叠，因此一个批次缩减到了一幅图像。相应的，我们使用的动量值很大0.99，这样之前训练过的大量训练样本才能确定当前优化步骤中的更新值。

The energy function is computed by a pixel-wise soft-max over the final feature map combined with the cross entropy loss function. The soft-max is defined as $p_k (x) = \frac {exp(a_k(x))} {\sum_{k'=1}^K exp(a_{k'} (x))}$ where $a_k (x)$ denotes the activation in feature channel k at the pixel position x ∈ Ω with $Ω ⊂ Z^2$. K is the number of classes and $p_k (x)$ is the approximated maximum-function. I.e. $p_k (x) ≈ 1$ for the k that has the maximum activation $a_k (x)$ and $p_k (x) ≈ 0$ for all other k. The cross entropy then penalizes at each position the deviation of $p_{l(x)} (x)$ from 1 using

能量函数的计算是在最终特征图上计算逐像素的softmax，并与交叉熵损失函数结合到一起。softmax的定义为$p_k (x) = \frac {exp(a_k(x))} {\sum_{k'=1}^K exp(a_{k'} (x))}$，其中$a_k (x)$表示的是特征通道k中在像素位置x ∈ Ω的激活，$Ω ⊂ Z^2$。K是类别数量，$p_k (x)$是近似的最大函数。即，对于有最大激活$a_k (x)$的k，有$p_k (x) ≈ 1$；对所有其他k，有$p_k (x) ≈ 0$。交叉熵在每个位置惩罚$p_{l(x)} (x)$与1之间的偏移，使用下式：

$$E = \sum_{x∈Ω} w(x) log(p_{l(x)}(x))$$(1)

where l : Ω → {1, . . . , K} is the true label of each pixel and w : Ω → R is a weight map that we introduced to give some pixels more importance in the training. 其中l : Ω → {1, . . . , K}是每个像素的真实标签，w : Ω → R是我们引入的权重图，在训练中给一些像素更多重要性。

We pre-compute the weight map for each ground truth segmentation to compensate the different frequency of pixels from a certain class in the training data set, and to force the network to learn the small separation borders that we introduce between touching cells (See Figure 3c and d).

我们为每个真值分割预计算权重图，以补偿特定类别像素在训练数据集中的不同频率，迫使网络学习很小的分离边缘，这是我们在接触的细胞中引入的（见图3c和3d）。

Fig. 3. HeLa cells on glass recorded with DIC (differential interference contrast) microscopy. (a) raw image. (b) overlay with ground truth segmentation. Different colors indicate different instances of the HeLa cells. (c) generated segmentation mask (white: foreground, black: background). (d) map with a pixel-wise loss weight to force the network to learn the border pixels.

The separation border is computed using morphological operations. The weight map is then computed as 分离边缘使用形态学计算。权重图然后计算如下

$$w(x) = w_c(x) + w_0 · exp(-\frac {(d_1(x)+d_2(x))^2} {2σ^2})$$(2)

where $w_c : Ω → R$ is the weight map to balance the class frequencies, $d_1: Ω → R$ denotes the distance to the border of the nearest cell and $d_2 : Ω → R$ the distance to the border of the second nearest cell. In our experiments we set $w_0 = 10$ and σ ≈ 5 pixels.

其中$w_c : Ω → R$是均衡类别频率的权重图，$d_1: Ω → R$表示到最近的细胞的边缘的距离，$d_2 : Ω → R$是到第二近的细胞的边缘的距离。在我们的试验中，我们设$w_0 = 10$，σ ≈ 5像素。

In deep networks with many convolutional layers and different paths through the network, a good initialization of the weights is extremely important. Otherwise, parts of the network might give excessive activations, while other parts never contribute. Ideally the initial weights should be adapted such that each feature map in the network has approximately unit variance. For a network with our architecture (alternating convolution and ReLU layers) this can be achieved by drawing the initial weights from a Gaussian distribution with a standard deviation of $\sqrt{2/N}$, where N denotes the number of incoming nodes of one neuron [5]. E.g. for a 3x3 convolution and 64 feature channels in the previous layer N = 9 · 64 = 576.

在很多卷积层的深度网络中，如果网络中有不同的路径，那么权重很好的初始化就非常重要。否则，部分网络可能给出过多的激活，而其他部分则没有任何贡献。理想情况下，初始值应当使网络中的每个特征图的方差都大约是1。对于我们的网络架构（交替的卷积和ReLU层），可以将初始权重设为零均值标准差为$\sqrt{2/N}$的高斯分布的随机量，其中N表示一个神经元的输入节点。如，对于一个3×3卷积、64特征通道的层来说，N = 9 · 64 = 576。

### 3.1 Data Augmentation 数据扩充

Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available. In case of microscopical images we primarily need shift and rotation invariance as well as robustness to deformations and gray value variations. Especially random elastic deformations of the training samples seem to be the key concept to train a segmentation network with very few annotated images. We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid. The displacements are sampled from a Gaussian distribution with 10 pixels standard deviation. Per-pixel displacements are then computed using bicubic interpolation. Drop-out layers at the end of the contracting path perform further implicit data augmentation.

要将网络训练出想要的不变性和稳健性，在只有一些训练样本可用的时候，数据扩充至关重要。在显微图像的情况中，我们主要需要平移和旋转不变性，也需要对变形和灰度值变化的稳健性。特别是训练样本的随机弹性变换，似乎是使用很少标准样本训练出一个分割网络的关键概念。我们在一个粗糙的3×3网格上使用随机位移矢量，生成平滑的变形。这个位移是10像素标准差的高斯分布。逐像素的位移使用双线性插值计算出来。收缩路径下的dropout层进行了额外的隐式数据扩充。

## 4 Experiments 试验

We demonstrate the application of the u-net to three different segmentation tasks. The first task is the segmentation of neuronal structures in electron microscopic recordings. An example of the data set and our obtained segmentation is displayed in Figure 2. We provide the full result as Supplementary Material. The data set is provided by the EM segmentation challenge [14] that was started at ISBI 2012 and is still open for new contributions. The training data is a set of 30 images (512x512 pixels) from serial section transmission electron microscopy of the Drosophila first instar larva ventral nerve cord (VNC). Each image comes with a corresponding fully annotated ground truth segmentation map for cells (white) and membranes (black). The test set is publicly available, but its segmentation maps are kept secret. An evaluation can be obtained by sending the predicted membrane probability map to the organizers. The evaluation is done by thresholding the map at 10 different levels and computation of the “warping error”, the “Rand error” and the “pixel error” [14].

我们将U-Net应用于三种不同的分割任务。第一个任务是电子显微镜图像中神经结构的分割。图2给出了数据集中的一个例子和我们得到的分割结果。我们在附加材料中给出完整的结果。这个数据集是EM分割挑战中的，从ISBI 2012时开始，现在仍然是开放的。训练数据是30幅图像（512×512大小）。每幅图像都有对应的完全标注的真值分割图，包含细胞（白）和细胞膜（黑）。测试集是公开可用的，但其真值分割图没有公开。评估需要将预测的细胞膜概率图提交给组织者。将概率图用10个不同的阈值进行切分，并计算“变形误差”，“Rand error”和“像素误差”。

The u-net (averaged over 7 rotated versions of the input data) achieves without any further pre- or postprocessing a warping error of 0.0003529 (the new best score, see Table 1) and a rand-error of 0.0382.

我们的U-Net的数据数据进行7种旋转，其结果进行了平均，在没有更多的预处理或后处理的情况下，得到了warping error为0.0003529（新的最好分数，见表1），和rand-error 0.0382。

This is significantly better than the sliding-window convolutional network result by Ciresan et al. [1], whose best submission had a warping error of 0.000420 and a rand error of 0.0504. In terms of rand error the only better performing algorithms on this data set use highly data set specific post-processing methods applied to the probability map of Ciresan et al. [1].

这比Ciresan等[1]的滑窗卷积网络的结果好很多，其最好结果的warping error为0.000420和rand error 0.0504。以rand error来说，在这个数据集上唯一表现更好的算法，使用了与数据集高度相关的后处理方法，处理了Ciresan等[1]得到的概率图。

We also applied the u-net to a cell segmentation task in light microscopic images. This segmenation task is part of the ISBI cell tracking challenge 2014 and 2015 [10,13]. The first data set “PhC-U373” contains Glioblastoma-astrocytoma U373 cells on a polyacrylimide substrate recorded by phase contrast microscopy (see Figure 4a,b and Supp. Material). It contains 35 partially annotated training images. Here we achieve an average IOU (“intersection over union”) of 92%, which is significantly better than the second best algorithm with 83% (see Table 2). The second data set “DIC-HeLa” are HeLa cells on a flat glass recorded by differential interference contrast (DIC) microscopy (see Figure 3, Figure 4c,d and Supp. Material). It contains 20 partially annotated training images. Here we achieve an average IOU of 77.5% which is significantly better than the second best algorithm with 46%.

我们还将U-Net用于了一种光学显微图像的细胞分割任务中。这个分割任务是ISBI细胞追踪挑战赛2014和2015的一部分。第一个数据集PhC-U373包含了XXXX U373细胞（见附加材料中的图4a,b），包含了35幅部分标注的训练图像。这里我们得到的平均IOU为92%，比第二名的83%要好很多（见表2）。第二个数据集DIC-HeLa是HeLa细胞图（见图3，图4c，d和附加材料）。包含20幅部分标注的训练图像。这里我们得到了平均IOU 77.5%，比第二名的46%要好很多。

Fig. 4. Result on the ISBI cell tracking challenge. (a) part of an input image of the “PhC-U373” data set. (b) Segmentation result (cyan mask) with manual ground truth (yellow border) (c) input image of the “DIC-HeLa” data set. (d) Segmentation result (random colored masks) with manual ground truth (yellow border).

Table 2. Segmentation results (IOU) on the ISBI cell tracking challenge 2015.

Name | PhC-U373 | DIC-HeLa
--- | --- | ---
IMCB-SG (2014) | 0.2669 | 0.2935
KTH-SE (2014) | 0.7953 | 0.4607
HOUS-US (2014) | 0.5323 | -
second-best 2015 | 0.83 | 0.46
u-net (2015) | 0.9203 | 0.7756

## 5 Conclusion

The u-net architecture achieves very good performance on very different biomedical segmentation applications. Thanks to data augmentation with elastic deformations, it only needs very few annotated images and has a very reasonable training time of only 10 hours on a NVidia Titan GPU (6 GB). We provide the full Caffe[6]-based implementation and the trained networks. We are sure that the u-net architecture can be applied easily to many more tasks.

U-Net架构在非常不同的生物医学分割应用中取得了非常好的表现。多亏了弹性变形的数据扩充，只需要非常少的标注图像，在NVidia Titan GPU (6GB)上只需要10个小时的时间就可以得到很好的结果。我们给出了基于Caffe的完整实现和训练好的网络。我们非常确信，U-Net架构可以很轻松的用于更多任务。