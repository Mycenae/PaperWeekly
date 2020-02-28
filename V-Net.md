# V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

Fausto Milletari et al. Technische Universit ̈at Mu ̈nchen, Germany Johns Hopkins University

## 0. Abstract

Convolutional Neural Networks (CNNs) have been recently employed to solve problems from both the computer vision and medical image analysis fields. Despite their popularity, most approaches are only able to process 2D images while most medical data used in clinical practice consists of 3D volumes. In this work we propose an approach to 3D image segmentation based on a volumetric, fully convolutional, neural network. Our CNN is trained end-to-end on MRI volumes depicting prostate, and learns to predict segmentation for the whole volume at once. We introduce a novel objective function, that we optimise during training, based on Dice coefficient. In this way we can deal with situations where there is a strong imbalance between the number of foreground and background voxels. To cope with the limited number of annotated volumes available for training, we augment the data applying random non-linear transformations and histogram matching. We show in our experimental evaluation that our approach achieves good performances on challenging test data while requiring only a fraction of the processing time needed by other previous methods.

CNNs最近已经用于解决计算机视觉和医学图像分析问题。尽管这么流行，但大多方法都只能处理2D图像，同时大多数临床上使用的医学数据都是由3D体组成的。本文中，我们提出了一种基于体的、全卷积的神经网络，进行3D图像分割。我们的CNN是在前列腺MRI体上进行的端到端训练，一次性对整个体预测其分割。我们提出一种新的基于Dice系数的目标函数，在训练的过程中进行优化。以这种方法，我们可以处理前景像素和背景像素数量很不平衡的问题。为处理可用的标注体数量少的问题，我们对数据进行扩增，使用随机非线性变换和直方图匹配。我们在试验评估中展示了，我们的方法在很有挑战性的测试数据上得到了很好的表现，所需的时间与之前的方法相比，只有一小部分。

## 1. Introduction and Related Work

Recent research in computer vision and pattern recognition has highlighted the capabilities of Convolutional Neural Networks (CNNs) to solve challenging tasks such as classification, segmentation and object detection, achieving state-of-the-art performances. This success has been attributed to the ability of CNNs to learn a hierarchical representation of raw input data, without relying on handcrafted features. As the inputs are processed through the network layers, the level of abstraction of the resulting features increases. Shallower layers grasp local information while deeper layers use filters whose receptive fields are much broader that therefore capture global information [19].

计算机视觉和模式识别的最近研究，其亮点是CNN在很多有挑战性的任务中的应用，如分类，分割和目标检测，都得到了目前最好的结果。这种成功归功于CNNs学习原始输入数据的层次化表示的能力，而不需要依赖于手工设计的特征。由于输入是通过网络层处理的，得到特征的抽象层次逐渐增加。较浅的层抓住的是局部信息，而更深的层使用的滤波器其感受野更大，因此捕获的是全局信息。

Segmentation is a highly relevant task in medical image analysis. Automatic delineation of organs and structures of interest is often necessary to perform tasks such as visual augmentation [10], computer assisted diagnosis [12], interventions [20] and extraction of quantitative indices from images [1]. In particular, since diagnostic and interventional imagery often consists of 3D images, being able to perform volumetric segmentations by taking into account the whole volume content at once, has a particular relevance. In this work, we aim to segment prostate MRI volumes. This is a challenging task due to the wide range of appearance the prostate can assume in different scans due to deformations and variations of the intensity distribution. Moreover, MRI volumes are often affected by artefacts and distortions due to field inhomogeneity. Prostate segmentation is nevertheless an important task having clinical relevance both during diagnosis, where the volume of the prostate needs to be assessed [13], and during treatment planning, where the estimate of the anatomical boundary needs to be accurate [4,20].

分割是在医学图像分析中是一个高度有意义的工作。感兴趣器官和结构的自动勾画对于一些任务是必须的，如视觉增强[10]，计算机辅助诊断[12]，介入[20]和从图像中提取量化索引[1]。特别是，由于诊断和介入图像通常是由3D图像组成的，如果可以一次性对整个体考虑体的分割，则是非常有意义的。本文中，我们的目标是分割前列腺MRI体。这是一个很有挑战性的任务，因为前列腺可能会有很大的形状变化，在不同的扫描中，会有变形和灰度分布的变化。而且，MRI体通常会受伪影和形变，主要原因是磁场的非均一性。前列腺分割是一个非常重要的任务，在诊断中有重要的临床意义，其中需要评估前列腺的体积，在治疗计划时，解剖边界的估计需要非常准确。

CNNs have been recently used for medical image segmentation. Early approaches obtain anatomy delineation in images or volumes by performing patchwise image classification. Such segmentations are obtained by only considering local context and therefore are prone to failure, especially in challenging modalities such as ultrasound, where a high number of mis-classified voxel are to be expected. Post-processing approaches such as connected components analysis normally yield no improvement and therefore, more recent works, propose to use the network predictions in combination with Markov random fields [6], voting strategies [9] or more traditional approaches such as level-sets [2]. Patch-wise approaches also suffer from efficiency issues. When densely extracted patches are processed in a CNN, a high number of computations is redundant and therefore the total algorithm runtime is high. In this case, more efficient computational schemes can be adopted.

CNNs最近用于进行医学图像分割。早期在图像或体中进行解剖结构勾画的方法，是进行逐块的图像分类。这种分割只考虑了局部上下文，因此很容易失败，尤其是在一些有挑战性的模态中，如超声，这样会造成很多体素被误分类。后处理方法，如连接组件分析，通常不会有什么改进，因此最近的工作提出使用网络预测与Markov随机场的结合，与投票策略的结合，或更传统的方法，如level-sets。逐块的方法还有效率低下的问题。当密集抽取出的块送入CNNs处理，很多计算是冗余的，因此总体算法运行时间是很高的。这种情况下，可以采用更高效的计算方案。

Fully convolutional network trained end-to-end were so far applied only to 2D images both in computer vision [11,8] and microscopy image analysis [14]. These models, which served as an inspiration for our work, employed different network architectures and were trained to predict a segmentation mask, delineating the structures of interest, for the whole image. In [11] a pre-trained VGG network architecture [15] was used in conjunction with its mirrored, de-convolutional, equivalent to segment RGB images by leveraging the descriptive power of the features extracted by the innermost layer. In [8] three fully convolutional deep neural networks, pre-trained on a classification task, were refined to produce segmentations while in [14] a brand new CNN model, especially tailored to tackle biomedical image analysis problems in 2D, was proposed.

端到端训练的全卷积网络，目前只应用于了2D图像，包括计算机视觉和显微图像分析。这些模型都是我们工作的启发源泉，它们采用了不同的网络架构，经过训练预测一个分割掩膜，对整幅图像勾画感兴趣的结构。在[11]中，使用了预训练的VGG网络架构及其镜像的、解卷积对应部分来分割RGB图像，利用了最深处的层所提取的非常有描述能力的特征。在[8]中，三个全卷积深度神经网络，在一个分类任务进行了预训练，经过提炼得到分割结果；而在[14]中，提出了全新的CNN模型，来处理2D医学图像分析问题。

In this work we present our approach to medical image segmentation that leverages the power of a fully convolutional neural networks, trained end-to-end, to process MRI volumes. Differently from other recent approaches we refrain from processing the input volumes slice-wise and we propose to use volumetric convolutions instead. We propose a novel objective function based on Dice coefficient maximisation, that we optimise during training. We demonstrate fast and accurate results on prostate MRI test volumes and we provide direct comparison with other methods which were evaluated on the same test data.

本文中，我们提出了我们的医学图像分割的方法，利用了全卷积神经网络，进行了端到端的训练，以处理MRI体。与最近的方法不同的是，我们没有对输入的体逐层进行处理，而是提出使用了体卷积。我们提出一种新的目标函数，基于Dice系数最大化，我们在训练的过程中进行了优化。我们证明了，在前列腺MRI测试体上可以快速准确的得到结果，与其他方法进行了直接比较，都在同样的测试数据上进行了评估。

Fig. 1. Slices from MRI volumes depicting prostate. This data is part of the PROMISE2012 challenge dataset [7].

## 2 Method

In Figure 2 we provide a schematic representation of our convolutional neural network. We perform convolutions aiming to both extract features from the data and, at the end of each stage, to reduce its resolution by using appropriate stride. The left part of the network consists of a compression path, while the right part decompresses the signal until its original size is reached. Convolutions are all applied with appropriate padding.

图2中，我们给出了我们CNN的图示。我们进行卷积的目标是，从数据中提取特征，同时在每个阶段最后，使用适当的步幅来降低其分辨率。网络的左边部分是压缩路径，而右边部分对信号进行解压，直到达到其原始大小。在进行卷积的时候，使用了适当的padding。

The left side of the network is divided in different stages that operate at different resolutions. Each stage comprises one to three convolutional layers. Similarly to the approach presented in [3], we formulate each stage such that it learns a residual function: the input of each stage is (a) used in the convolutional layers and processed through the non-linearities and (b) added to the output of the last convolutional layer of that stage in order to enable learning a residual function. As confirmed by our empirical observations, this architecture ensures convergence in a fraction of the time required by a similar network that does not learn residual functions.

网络的左边分成了几个不同的阶段，在不同的分辨率上进行计算。每个阶段是由1-3个卷积层组成。与[3]中的方法类似，我们对每个阶段进行公式化表达，这样学习到一个残差函数：每个阶段的输入(a)用于卷积层中，进而通过非线性处理，(b)加入到这个阶段的最后一个卷积层的输出上，以学习到一个残差函数。由我们的经验化观察可以得到，这种架构与没有学习残差函数的相比，所需的收敛时间大大减少。

Fig. 2. Schematic representation of our network architecture. Our custom implementation of Caffe [5] processes 3D data by performing volumetric convolutions. Best viewed in electronic format.

The convolutions performed in each stage use volumetric kernels having size 5×5×5 voxels. As the data proceeds through different stages along the compression path, its resolution is reduced. This is performed through convolution with 2 × 2 × 2 voxels wide kernels applied with stride 2 (Figure 3). Since the second operation extracts features by considering only non overlapping 2×2×2 volume patches, the size of the resulting feature maps is halved. This strategy serves a similar purpose as pooling layers that, motivated by [16] and other works discouraging the use of max-pooling operations in CNNs, have been replaced in our approach by convolutional ones. Moreover, since the number of feature channels doubles at each stage of the compression path of the V-Net, and due to the formulation of the model as a residual network, we resort to these convolution operations to double the number of feature maps as we reduce their resolution. PReLu non linearities are applied throughout the network.

每个阶段进行的卷积都使用了体核，大小为5×5×5体素。由于数据在压缩路径上通过了不同的阶段，其分辨率降低了。这是以步幅2进行的与2 × 2 × 2大小的核进行的卷积（图3）。由于第二种运算是对不重叠的2×2×2的体块提取特征，得到的特征图大小减半了。这种策略达到的目标，与池化层类似，我们参考了[16]和其他工作，发现在CNNs中使用最大池化运算的作用是不太好的，所以用我们方法中的卷积替代。而且，由于特征通道的数量在V-Net压缩路径的每个阶段都翻倍了，而且由于模型采用的是残差网络，我们借助于这些卷积运算来使特征图的数量翻倍，因为我们降低了其分辨率。在我们的网络中，一直使用的是PReLU非线性函数。

Fig. 3. Convolutions with appropriate stride can be used to reduce the size of the data. Conversely, de-convolutions increase the data size by projecting each input voxel to a bigger region through the kernel.

Replacing pooling operations with convolutional ones results also to networks that, depending on the specific implementation, can have a smaller memory footprint during training, due to the fact that no switches mapping the output of pooling layers back to their inputs are needed for back-propagation, and that can be better understood and analysed [19] by applying only de-convolutions instead of un-pooling operations.

将池化运算替换为卷积运算，配合特定的实现，在训练时会耗费更小的内存，由于将池化层的输出映射回其输入的开关在反向传播中不需要了，而我们只进行了反卷积运算，而没有使用反池化运算，这也可以得到更好的理解和分析。

Downsampling allows us to reduce the size of the signal presented as input and to increase the receptive field of the features being computed in subsequent network layers. Each of the stages of the left part of the network, computes a number of features which is two times higher than the one of the previous layer.

下采样使我们降低输入信号的尺寸，增加后续网络层中计算的特征的感受野。网络左边部分的每个阶段，计算的一些特征比之前的层是高两倍的。

The right portion of the network extracts features and expands the spatial support of the lower resolution feature maps in order to gather and assemble the necessary information to output a two channel volumetric segmentation. The two features maps computed by the very last convolutional layer, having 1×1×1 kernel size and producing outputs of the same size as the input volume, are converted to probabilistic segmentations of the foreground and background regions by applying soft-max voxelwise. After each stage of the right portion of the CNN, a de-convolution operation is employed in order increase the size of the inputs (Figure 3) followed by one to three convolutional layers involving half the number of 5 × 5 × 5 kernels employed in the previous layer. Similar to the left part of the network, also in this case we resort to learn residual functions in the convolutional stages.

网络右边的部分也提取特征，并扩展较低分辨率特征图的空间占用，以收集组合必须的信息，以输出一个两通道的体分割。最后一个卷积层计算得到的两个特征图，其核大小为1×1×1，生成的输出大小与输入体相同，并通过逐体素的soft-max运算，转换为前景区域和背景区域的概率分割。CNN右边部分的每个阶段，都采用了一个解卷积运算，以增加输入的大小，然后再进行一到三个卷积层，核大小为5 × 5 × 5，数量比前一层减半。与网络左边类似，我们在每个卷积阶段都学习一个残差函数。

Similarly to [14], we forward the features extracted from early stages of the left part of the CNN to the right part. This is schematically represented in Figure 2 by horizontal connections. In this way we gather fine grained detail that would be otherwise lost in the compression path and we improve the quality of the final contour prediction. We also observed that when these connections improve the convergence time of the model.

与[14]类似，我们将左边部分提取到的特征，都转送到右边部分，在图2中，我们将其表示为水平接连。以这种方法，我们聚集了细粒度的细节，否则就会在压缩路径上损失掉，这样我们改进了最终轮廓预测的质量。我们还观察到了，这些连接改进了模型训练的收敛时间。

We report in Table 1 the receptive fields of each network layer, showing the fact that the innermost portion of our CNN already captures the content of the whole input volume. We believe that this characteristic is important during segmentation of poorly visible anatomy: the features computed in the deepest layer perceive the whole anatomy of interest at once, since they are computed from data having a spatial support much larger than the typical size of the anatomy we seek to delineate, and therefore impose global constraints.

我们在表1中给出了网络每层的感受野，说明我们网络的最中间层，已经捕获了正个输入体的所有内容。我们相信这种特性在分割过程中，对于看起来不那么清晰的解剖结构是很重要的：最深一层所计算的特征，一次性感知到了感兴趣的整个解剖结构，因为其计算所用的数据，其空间支撑比我们要勾画的解剖结构的典型大小要大的多，因此施加了全局约束。

Table 1. Theoretical receptive field of the 3 × 3 × 3 convolutional layers of the network.

Layer | Input Size | Receptive Field | Layer | Input Size | Receptive Field
--- | --- | --- | --- | --- | ---
L-Stage 1 | 128 | 5×5×5 | R-Stage 4 | 16 | 476×476×476
L-Stage 2 | 64 | 22×22×22 | R-Stage 3 | 32 | 528×528×528
L-Stage 3 | 32 | 72×72×72 | R-Stage 2 | 64 | 546×546×546
L-Stage 4 | 16 | 172 × 172 × 172 | R-Stage 1 | 128 | 551×551×551
L-Stage 5 | 8 | 372 × 372 × 372 | Output | 128 | 551×551×551

## 3 Dice loss layer

The network predictions, which consist of two volumes having the same resolution as the original input data, are processed through a soft-max layer which outputs the probability of each voxel to belong to foreground and to background. In medical volumes such as the ones we are processing in this work, it is not uncommon that the anatomy of interest occupies only a very small region of the scan. This often causes the learning process to get trapped in local minima of the loss function yielding a network whose predictions are strongly biased towards background. As a result the foreground region is often missing or only partially detected. Several previous approaches resorted to loss functions based on sample re-weighting where foreground regions are given more importance than background ones during learning. In this work we propose a novel objective function based on dice coefficient, which is a quantity ranging between 0 and 1 which we aim to maximise. The dice coefficient D between two binary volumes can be written as:

网络预测，是由两个体组成的，与原始输入数据有相同的分辨率，经过soft-max层处理，输出每个体素属于前景还是背景的概率。在医学体中，如本文我们处理的这些，感兴趣的解剖结构是占了一个扫描的很小一部分，这是很正常的。这通常导致学习过程会陷入损失函数的局部极小值，得到的网络其预测很倾向于背景。结果是，前景区域通常缺失，或只是部分检测到。几种之前的方法，采用基于样本重新赋权的损失函数，在学习过程中，前景区域的权重更高一些，背景的更低一些。在本文中，我们提出一种新的目标函数，是基于Dice系数的，其值在0，1之间，这是我们要最大化的。两个二值体之间的Dice系数D可以写为：

$$D = \frac {2\sum_i^N p_i g_i} {\sum_i^N p_i^2 + \sum_i^N g_i^2}$$

where the sums run over the N voxels, of the predicted binary segmentation volume $p_i ∈ P$ and the ground truth binary volume $g_i ∈ G$. This formulation of Dice can be differentiated yielding the gradient

其中对预测的二值分割体$p_i ∈ P$和真值二值体$g_i ∈ G$，在N个体素上计算和。这个Dice公式求其微分，得到下面的梯度

$$\frac {∂D}{∂p_j} = 2[\frac {g_j (\sum_i^N p_i^2 + \sum_i^N g_i^2) - 2p_j (\sum_i^N p_i g_i)} {(\sum_i^N p_i^2 + \sum_i^N g_i^2)^2}]$$

computed with respect to the j-th voxel of the prediction. Using this formulation we do not need to assign weights to samples of different classes to establish the right balance between foreground and background voxels, and we obtain results that we experimentally observed are much better than the ones computed through the same network trained optimising a multinomial logistic loss with sample re-weighting (Fig. 6).

这是对第j个体素的预测计算的其梯度。使用这个公式，我们不需要对不同类别的样本指定其权重，以对前景像素和背景像素进行正确的平衡，我们得到的结果，通过试验观察，比那些使用同样网络对多项式逻辑带有样本重新赋权的损失进行优化要好的多（图6）。

### 3.1 Training 训练

Our CNN is trained end-to-end on a dataset of prostate scans in MRI. An example of the typical content of such volumes is shown in Figure 1. All the volumes processed by the network have fixed size of 128 × 128 × 64 voxels and a spatial resolution of 1 × 1 × 1.5 millimeters.

我们的CNN在一个MRI扫描的前列腺数据集上进行了端到端的训练，图1是这种体的典型内容的一个例子。网络处理的所有体有固定的大小，即128 × 128 × 64体素，空间分辨率是1 × 1 × 1.5 mm。

Annotated medical volumes are not easy to obtain due to the fact that one or more experts are required to manually trace a reliable ground truth annotation and that there is a cost associated with their acquisition. In this work we found necessary to augment the original training dataset in order to obtain robustness and increased precision on the test dataset.

标注的医学体没那么容易得到，因为需要一个或多个专家手动跟踪一个可靠的真值标注，获得这种标注是有代价的。本文中，我们发现需要对原始训练数据集进行增强，以得到稳健性，增加在测试数据集上的精度。

During every training iteration, we fed as input to the network randomly deformed versions of the training images by using a dense deformation field obtained through a 2 × 2 × 2 grid of control-points and B-spline interpolation. This augmentation has been performed ”on-the-fly”, prior to each optimisation iteration, in order to alleviate the otherwise excessive storage requirements. Additionally we vary the intensity distribution of the data by adapting, using histogram matching, the intensity distributions of the training volumes used in each iteration, to the ones of other randomly chosen scans belonging to the dataset.

在每次训练迭代中，我们对网络的输入是对训练图像的随机形变版，即使用2 × 2 × 2网格控制点和B样条插值的密集形变场进行的变形。这种扩充是即时进行的，在每次优化迭代之前，否则就会需要很多存储空间。另外，我们对数据的灰度分布进行变化，使用直方图匹配，改变每次迭代中，训练体的灰度分布，改变为数据集中随机选取的其他scans的直方图。

### 3.2 Testing

A previously unseen MRI volume can be segmented by processing it in a feed-forward manner through the network. The output of the last convolutional layer, after soft-max, consists of a probability map for background and foreground. The voxels having higher probability (> 0.5) to belong to the foreground than to the background are considered part of the anatomy.

之前未曾见到过的MRI体送入网络中，通过前向过程，可以对其进行分割。最后一个卷积层经过soft-max的输出，包含了背景和前景的概率图。如果体素被认为是前景的概率(> 0.5)，超过被认为是背景的概率，那么就认为这是解剖结构的一部分。

## 4 Results

We trained our method on 50 MRI volumes, and the relative manual ground truth annotation, obtained from the ”PROMISE2012” challenge dataset [7]. This dataset contains medical data acquired in different hospitals, using different equipment and different acquisition protocols. The data in this dataset is representative of the clinical variability and challenges encountered in clinical settings. As previously stated we massively augmented this dataset through random transformation performed in each training iteration, for each mini-batch fed to the network. The mini-batches used in our implementation contained two volumes each, mainly due to the high memory requirement of the model during training. We used a momentum of 0.99 and a initial learning rate of 0.0001 which decreases by one order of magnitude every 25K iterations.

我们在50个MRI体以及相关的手动真值标注上进行训练，这是从PROMISE2012挑战赛数据集[7]上进行训练的。这个数据集包含不同医院取得的医学数据，使用不同的设备和不同的获取协议。这个数据集中的数据是临床设置中所遇到的临床变化和挑战的代表。就像之前说的那样，我们对这个数据集，每次喂入网络的一个mini-batch，通过在每次训练迭代中进行随机形变，进行大规模扩充。我们的实现中，使用的mini-batch每次包含2个体，主要是因为模型训练时占用的内存非常大。我们使用的动量为0.99，初始学习速率为0.0001，每25K次迭代降低一个数量级。

We tested V-Net on 30 MRI volumes depicting prostate whose ground truth annotation was secret. All the results reported in this section of the paper were obtained directly from the organisers of the challenge after submitting the segmentation obtained through our approach. The test set was representative of the clinical variability encountered in prostate scans in real clinical settings [7].

我们在30个MRI体上测试了V-Net，里面都包含了前列腺，其真值标注是秘密的。这个部分得到的所有结果，都是我们将我们方法的分割结果提交给挑战赛组织方，从组织方那里直接得到的。测试集代表了真实临床设置中前列腺scans所遇到的临床变化。

We evaluated the approach performance in terms of Dice coefficient, Hausdorff distance of the predicted delineation to the ground truth annotation and in terms of score obtained on the challenge data as computed by the organisers of ”PROMISE 2012” [7]. The results are shown in Table 2 and Fig. 5.

我们评估方法性能，是以预测勾画与真值标注之间的Dice系数和Hausdorff距离，在PROMISE 2012挑战赛的数据上进行的。结果如表2图5所示。

Fig.5. Distribution of volumes with respect to the Dice coefficient achieved during segmentation.

Table 2. Quantitative comparison between the proposed approach and the current best results on the PROMISE 2012 challenge dataset.

Algorithm | Avg. Dice | Avg. Hausdorff distance | Score on challenge task
--- | --- | --- | ---
V-Net + Dice-based loss | 0.869 ± 0.033 | 5.71 ± 1.20 mm | 82.39
V-Net + mult. logistic loss | 0.739 ± 0.088 | 10.55 ± 5.38 mm | 63.30
Imorphics [18] | 0.879 ± 0.044 | 5.935 ± 2.14 mm | 84.36
ScrAutoProstate | 0.874 ± 0.036 | 5.58 ± 1.49 mm | 83.49
SBIA | 0.835 ± 0.055 | 7.73 ± 2.68 mm | 78.33
Grislies | 0.834 ± 0.082 | 7.90 ± 3.82 mm | 77.55

Our implementation was realised in python, using a custom version of the Caffe[5] framework which was enabled to perform volumetric convolutions via CuDNN v3. All the trainings and experiments were ran on a standard workstation equipped with 64 GB of memory, an Intel(R) Core(TM) i7-5820K CPU working at 3.30GHz, and a NVidia GTX 1080 with 8 GB of video memory. We let our model train for 48 hours, or 30K iterations circa, and we were able to segment a previously unseen volume in circa 1 second. The datasets were first normalised using the N4 bias filed correction function of the ANTs framework [17] and then resampled to a common resolution of 1 × 1 × 1.5 mm. We applied random deformations to the scans used for training by varying the position of the control points with random quantities obtained from gaussian distribution with zero mean and 15 voxels standard deviation. Qualitative results can be seen in Fig. 4.

我们是用python实现的，使用的是Caffe的定制版，通过CuDNN v3即可以进行体的卷积。所有的训练和试验都是在一个标准工作站上进行的，有64GB内存，一个Intel(R) Core(TM) i7-5820K CPU，工作频率为3.30GHz，一个NVidia GTX 1080，8GB显存。我们的模型训练了48个小时，或大约30K次迭代，然后我们就可以对一个之前未曾见过的体在大约1秒内完成分割。数据集首先使用N4 bias场修正函数，在ANTs框架中进行归一化，然后重采样到一个常见的分辨率1 × 1 × 1.5 mm。我们对训练用的scans进行随机形变，用随机量变换控制点的位置，随机量为零均值和15个体素的高斯分布。定性结果如图4所示。

Fig. 4. Qualitative results on the PROMISE 2012 dataset [7].

## 5 Conclusion

We presented an approach based on a volumetric convolutional neural network that performs segmentation of MRI prostate volumes in a fast and accurate manner. We introduced a novel objective function that we optimise during training based on the Dice overlap coefficient between the predicted segmentation and the ground truth annotation. Our Dice loss layer does not need sample re-weighting when the amount of background and foreground pixels is strongly unbalanced and is indicated for binary segmentation tasks. Although we inspired our architecture to the one proposed in [14], we divided it into stages that learn residuals and, as empirically observed, improve both results and convergence time. Future works will aim at segmenting volumes containing multiple regions in other modalities such as ultrasound and at higher resolutions by splitting the network over multiple GPUs.

我们提出一种基于体卷积神经网络的方法，对前列腺MRI体进行快速准确的分割。我们提出了一种基于预测分割与真值分割的Dice系数的新目标函数，在训练中对其进行优化。我们的Dice损失层在前景和背景像素极度不平衡时，不需要样本重赋权，是适用于二值分割任务的。虽然我们是从[14]中提出的网络受到启发，我们将其分为几个阶段，对残差进行学习，通过经验观察到，可以改进结果和收敛时间。未来的工作目标是，在其他模态中分割包含多个区域的体，如超声，或在更高分辨率上进行，将网络分到数个GPUs上进行训练。