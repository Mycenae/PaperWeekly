# Experimentally Defined Convolutional Neural Network Architechture Variants For Non-Temporal Real-Time Fire Detection

Andrew J. Dunnings et al. Durham University, UK

## Abstract

In this work we investigate the automatic detection of fire pixel regions in video (or still) imagery within real-time bounds without reliance on temporal scene information. As an extension to prior work in the field, we consider the performance of experimentally defined, reduced complexity deep convolutional neural network architectures for this task. Contrary to contemporary trends in the field, our work illustrates maximal accuracy of 0.93 for whole image binary fire detection, with 0.89 accuracy within our superpixel localization framework can be achieved, via a network architecture of signficantly reduced complexity. These reduced architectures additionally offer a 3-4 fold increase in computational performance offering up to 17 fps processing on contemporary hardware independent of temporal information. We show the relative performance achieved against prior work using benchmark datasets to illustrate maximally robust real-time fire region detection.

本文中我们研究了视频（或静止图像）中的火焰像素区域自动实时检测，而不需要与时间相关的场景信息。作为这个领域先前工作的拓展，我们考虑了一种实验确定的、简化复杂度的深度卷积神经网络架构来处理这项任务。我们的工作在整图二值火焰检测中得到了0.93的最高准确率，而超像素定位框架可以得到0.89的准确率，我们的工作是基于一种显著简化了复杂度的网络架构。这些简化的架构在计算性能上有3-4倍的提升，在现有的硬件性能上可以得到17fps的性能，而且与时间有关的信息是无关的。我们在基准测试数据集上给出了与之前工作的相对比较，展示了最大的稳健性的实时火焰区域检测。

**Index Terms**— simplified CNN, fire detection, real-time, non-temporal, non-stationary visual fire detection 简化的CNN，火焰检测，实时，非时间的，非静态视觉火焰检测

## 1. Introduction

A number of factors have driven forward the increased need for fire (or flame) detection within video sequences for deployment in a wide variety of automatic monitoring tasks. The increasing prevalence of industrial, public space and general environment monitoring using security-driven CCTV video systems has given rise to the consideration of these systems as secondary sources of initial fire detection (in addition to traditional smoke/heat based systems). Furthermore, the on-going consideration of remote vehicles for fire detection and monitoring tasks [1, 2, 3] adds further to the demand for autonomous fire detection from such platforms. In the latter case, attention turns not only to the detection of fire itself but also its internal geography of the fire and temporal development [4].

很多因素推动了在许多自动监测任务中部署视频序列火焰检测的需要。工业应用，公共场合和通用环境监测都越来越喜欢用安全驱动的CCTV视频系统，这使得人们开始考虑采用这些系统作为初始火焰监测的次要来源（作为传统的烟基于雾/热的系统的补充）。此外，采用遥控飞行器进行火焰监测与监控任务[1,2,3]的考虑一直存在，这进一步增加了从这种平台进行自动火焰监测的需要。在后一种情况下，注意力并不仅仅局限在火焰监测上，还在其火焰的内在地理位置上及其随着时间的发展上[4]。

Traditional approaches in this area concentrate either on the use of a purely colour based approach [5, 6, 7, 8, 9, 4] or a combination of colour and high-order temporal information [10, 11, 12, 13]. Early work emanated from the colour-threshold approach of [5] which was extended with the basic consideration of motion by [10]. Later work considered the temporal variation (flame flicker) of fire imagery within the Fourier domain [11] with further studies formulating a Hidden Markov Model problem [12]. More recently work considering the temporal aspect of the problem has investigated time-derivatives over the image [13]. Although flame flicker is generally not sinusoidal or periodic under all conditions, a frequency of 10Hz has been observed in generalised observational studies [14]. As such, [15] considered the use of the wavelet transform as a temporal feature. In later applications [7], we still see the basic approaches of [10] underlying colour-driven approaches although more sophisticated colour models based on a derivative of background segmentation [9] and consideration of alternative  colour spaces [8] are proposed. In general these works report ~98-99% (true positive) detection at 10-40 frames per second (fps) on relatively small image sizes (CIF or similar) [9, 8].

这个领域中的传统方法要么是使用纯基于颜色的方法[5,6,7,8,9,4]，或使用结合了颜色和高阶时间信息[10,11,12,13]。早期的工作起源于[5]的颜色阈值方法，又与运动的基本考虑[10]结合在一起进行了拓展。后来的工作考虑了时间上的变化（火焰的闪烁）在频域中的表现[11]，进一步研究形成了一个隐马尔科夫问题[12]。近来的工作考虑了这个问题在时间方面中信息，研究了图像序列对时间的导数[13]。虽然火焰的闪烁在所有情况下一般不是正弦或周期性的，但在很多通用观察研究中[14]都观察到了10Hz的这个频率。在这种情况下，[15]考虑了使用小波变换作为一种时域特征。在后来的应用中[7]，我们仍然看到了[10]的基本方法即潜在的颜色驱动的方法，虽然是更复杂的基于背景分割的导数的颜色模型[9]，或考虑了其他的颜色空间[8]。一般来说，这些工作给出了~98-99%的真阳性检测结果，速度为10-40fps，处理的图像相对较小（CIF或类似的）[9,8]。

More recent work has considered machine learning based classification approaches to the fire detection problem [3, 16, 17]. The work of [3] considers a colour-driven approach utilising temporal shape features as an input to a shallow neural network and similarly the work of [16] utilises wavelet co-efficients as an input to a SVM classifier. Chenebert et al. [17] consider the use of a non-temporal approach with the combined use of colour-texture feature descriptors as an input to decision tree or shallow neural network classification (80-90% mean true positive detection, 7-8% false positive). Other recent approaches consider the use shape-invariant features [18] or simple patches [19] within varying machine learning approaches. However, the majority of recent work is temporally dependent considering a range of dynamic features [20] and motion characteristics [21, 22] between consecutive video frames with the most recent work of [22] considering convolutional neural networks (CNN) for fire detection within this context.

更近来的工作考虑了基于机器学习的分类方法处理火焰检测问题[3,16,17]。[3]考虑了一种颜色驱动的方法，利用了时域形状特征，输入到一个浅层神经网络中；类似的，[16]将了小波系数输入到一个SVM分类器中。Chenebert等[17]使用了一种非时域方法，综合使用了色彩-纹理特征描述子作为决策树或浅层神经网络分类器的输入（80-90%的平均真阳性检测，7-8%的假阳性检测结果）。其他最近的方法考虑了在不同的机器学习方法中使用形状不变的特征[18]或简单的图象块[19]。但是，大部分最近的工作都在连续的视频帧中暂时依赖不同的特征[20]和运动特征[21,22]，更近来的工作[22]开始考虑使用CNN进行火焰检测。

Here, by contrast to previous classifier-driven work [3, 16, 4, 21, 20, 22], we instead consider a non-temporal classification model for fire detection following the theme non-temporal fire detection championed by Chenebert et al. [17] and further supporting by the non-stationary camera visual fire detection challenge posed by Steffans et al. [23]. Non-temporal detection models are highly suited to the non-stationary fire detection scenario posed by the future use of autonomous systems in a fire fighting context [23]. Within this work we show that comparable fire detection results are achievable to the recent temporally dependent work of [21, 20, 22], both exceeding the prior non-temporal approach of Chenebert et al. [17] and within significantly lower CNN model complexity than the recent work of [22]. Our reduced complexity network architectures are experimentally defined as architectural subsets of seminal CNN architectures offering maximal performance for the fire detection task. Furthermore, we extend this concept to incorporate in-frame localization via the use of superpixels [24] and benchmark comparison using the fire non-stationary (moving camera) visual fire detection dataset released under [23].

这里，与之前的分类驱动的工作[3,16,4,21,20,22]相比，我们考虑一种非时域的分类模型进行火焰检测，沿用了Chenbert等[17]夺冠的非时域火焰检测算法，而且由Steffans等[23]提出的非静态摄像头视觉火焰检测挑战所支持。非时域检测模型非常适合于非静态火焰检测场景，而这正是未来消防领域自动系统的使用所提出的问题[23]。在本文中，我们给出了与最近的依赖时域信息的工作[21,20,22]可以比较的火焰检测结果，超过了之前Chenbert等[17]的非时域方法，而且使用的是复杂度显著低于[22]的CNN模型。我们的降低复杂度的网络架构是由实验确定的，是火焰检测任务中给出了最佳结果的初始CNN架构的一个子集架构。而且，我们将这种概念进行了拓展，通过使用超像素[24]包含了帧内定位的功能，使用[23]给出的非静态（移动摄像头）视觉火焰检测数据集，与其基准测试进行了比较。

## 2. Approach 方法

Our approach centres around the development of low-complexity CNN architectural variants (Section 2.1) operating on single image inputs (non-temporal) experimentally optimized for the fire detection task (Section 2.2). This is then expanded into a superpixel based localization approach (Section 2.3) to offer a complete detection solution.

我们的方法集中于低复杂度CNN架构变体的发展（2.1节），输入为单幅图像（非时域），在火焰检测任务中进行实验优化（2.2节），然后扩展成为一种基于超像素的定义方法（2.3节），形成一个完整的检测方案。

### 2.1. Reference CNN Architectures 参考的CNN架构

We consider several candidate architectures, with reference to general object recognition performance within [25], to cover varying contemporary CNN design principles [26] that can then form the basis for our reduced complexity CNN approach. 我们考虑几种候选架构，参考[25]中的通用目标识别性能，以覆盖现在多样的CNN设计原则[26]，然后可以形成我们的简化复杂度CNN方法的基础。

**AlexNet** [27] represents the seminal CNN architecture comprising of 8 layers. Initially, a convolutional layer with a kernel size of 11 is followed by another convolutional layer of kernel size 5. The output of each of these layers is followed by a max pooling layer and local response normalization. Three more convolutional layers then follow, each having a kernel size of 3, and the third is followed by a max pooling layer and local response normalization. Finally, three fully connected layers are stacked to produce the classification output.

**AlexNet**[27]代表了最开始的CNN架构，包含8层。从开始处，是卷积核大小为11的卷积层，然后是另一个卷积核大小为5的卷积层，然后是一个max pooling层和局部响应归一化，然后是3个卷积层，卷积核大小都是3，最后是max pooling层和局部响应归一化层。最后，三个全卷积层叠加起来生成分类输出。

**VGG-16** [28] is a network architecture based on the principle of prioritizing simplicity and depth over complexity – all convolutional layers have a kernel size of 3, and the network has a depth of 16 layers. This model consists of groups of convolutional layers, and each group is followed by a max pooling layer. The first group consists of two convolutional layers, each with 64 filters, and is followed by a group of two convolutional layers with 128 filters each. Subsequently, a group of three layers with 256 filters each, and another two groups of three layers with 512 filters each feed into three fully connected layers which produce the output. Here we implement the 13-layer variant of this network by removing one layer from each of the final three groups of convolutional layers (denoted VGG-13).

**VGG-16**[28]的网络结构原则是简单优先和深度先于复杂度，全部都是卷积核大小为3的卷积层，网络深度为16层。这个模型包括几个卷积层组，每个组后面都接一个max pooling层。第一组包括2个卷积层，每层64个滤波器，后面又接了一组卷积层，包含2个卷积层，每个128个滤波器。然后，一组包含3层，每层256个滤波器，然后是两组，每组3层，每层512个滤波器，然后送入三个全卷积层，生成输出。这里我们实现其13层变体，从最后3组中每组去掉1层卷积层，表示为VGG-13。

**Inception-V1** ([29], GoogLeNet) is a network architecture composed almost entirely of a single repeating inception module element consisting of four parallel strands of computation, each containing different layers. The theory behind this choice is that rather than having to choose between convolutional filter parameters at each stage in the network, multiple different filters can be applied in parallel and their outputs concatenated. Different sized filters may be better at classifying certain inputs, so by applying many filters in parallel the network will be more robust. The four strands of computation are composed of convolutions of kernel sizes 1 × 1, 3 × 3, and 5 × 5, as well as a 3 × 3 max pooling layer. 1 × 1 convolutions are included in each strand to provide a dimension reduction – ensuring that the number of outputs does not increase from stage to stage, which would drastically decrease training speed. The Inception-V1 architecture offers a contrasting 22 layer deep network architecture to AlexNet (8 layers), offering superior benchmark performance [29], whilst having 12 times fewer parameters through modularization that make use of 9 inception modules in its standard configuration.

**Inception-V1**[29]，即GoogLeNet，是几乎都是由重复的Inception模块构成的网络架构，Inception模块是由4个并行的计算部分构成，每个包含不同的层。这种选择背后的理论是，在网络的各个阶段，不用在不同的卷积滤波器参数之间进行选择，多种不同的滤波器可以并行应用，并将其输出进行拼接。不同大小的滤波器擅长于对特定输入进行分类，所以并行的应用很多滤波器，网络会更稳健。4个计算部分是由卷积核大小为1 × 1, 3 × 3, and 5 × 5以及一个3 × 3 max pooling层。每个部分都包含了1 × 1卷积，以进行维度压缩，确保输出的数量在从一层到另一层时不会增加，这会极大的减少训练速度。Inception-V1架构的网络达到了22层，与AlexNet的8层形成了明显对比，并得到了非常好的基准测试结果[29]，网络使用了9个标准的Inception模块，模型参数只有1/12。

### 2.2. Simplified CNN Architectures 简化的CNN架构

Informed by the relative performance of the three representative CNN architectures (AlexNet, VGG-13, InceptionV1) on the fire detection task (Table 1, upper), an experimental assessment of the marginally better performing AlexNet and InceptionV1 architectures is performed.

这三种代表性的CNN架构在火焰检测任务中的表现如表1上所示，我们对其架构进行修改，得到效果略好的模型。

Our experimental approach systematically investigated variations in architectural configuration of each network against overall performance (statistical accuracy) on the fire image classification task. Performance was measured using the same evaluation parameters set out in Section 3 with network training performed on 25% of our fire detection training dataset and evaluated upon the same test dataset.

我们实验性的方法系统性的研究了每种网络架构配置的变化与火焰图像分类任务中性能的关系（统计性的准确率）。性能度量是使用第3节中设置的相同的评估参数，网络训练是在我们的火焰检测训练集的25%上进行，在相同的测试集上进行评估。

For AlexNet we consider six variations to the architectural configuration by removing layers from the original architecture, denoted by C1-C6 as follows: C1 removed layer 3 only, C2 removed layers 3, 4, C3 removed layers 3, 4, 5, C4 removed layers 6 only, C5 removed layers 3, 4, 6 and C6 removed layer 2 only. The results in terms of statistical accuracy for fire detection plotted against the number of parameters present in the resulting network model are shown in Figure 3 (left) where C7 represents the performance of the original AlexNet architecture [27].

对于AlexNet我们对架构配置进行6个变体，即从原始架构中去除一些层，表示为C1-C6：C1是只去掉第3层，C2是去掉第3、4层，C3是去掉3、4、5层，C4是去掉第6层，C5是去掉3、4、6层，C6是去掉第2层。得到的网络，其火焰检测统计准确率的结果与参数数量的关系如图3（左）所示，其中C7表示原始AlexNet架构[27]。

For the Inception-V1 architecture we consider eight variations to the architectural configuration by removing up to 8 inception modules from the original configuration of 9 present [29]. The results in terms of statistical accuracy for fire detection plotted against the number of parameters present in the resulting model are shown in Figure 3 (right) where label i ∈ {1..8} represents the resulting network model with only i inception modules present and i = 9 represents the performance of the original Inception-V1 architecture [29].

对于Inception-V1架构，我们考虑原始架构的8个变体，也是从原始架构的9个模块中去除一些Inception模块（最多8个）[29]。得到的模型，其火焰检测统计准确率的结果与参数数量的关系如图3（右）所示，其中标签i∈{1..8}代表只有i个Inception模块存在，i=9代表原始Inception-V1架构的表现[29]。

From the results shown in Figure 3 (left) we can see that configuration C2 improves upon the accuracy of all other architectural variations whilst containing significantly less parameters than several other configurations, including the original AlexNet architecture. Similarly, from the results shown in Figure 3 (right) we can see that accuracy tends to slightly decrease as the number of inception modules decreases, whereas the number of parameters decreases significantly. The exception to this variation is using only one inception module, for which performance is significantly reduced. An architecture containing only three inception modules is the variation with the fewest parameters which retains performance in the highest band (Figure 3, right).

从图3（左）的结果中可以看到，C2的配置在所有结构变体中准确率的改进最好，同时包含的参数数量显著减少。类似的，从图3（右）的结果中可以看出，随着Inception模块的减少，准确率会逐渐越来越低，而参数数量则显著减少。其中的一个例外情况是只使用一个Inception模块的时候，其性能也显著的下降了。包含3个Inception模块的架构是参数数量最少，同时性能最高的一个（图3，右）。

Overall from our experimentation on this subset of the main task (i.e. 25% training data), we can observe both explicit over-fitting within these original high-complexity CNN architectures such as the performance of reduced CNN C2 vs. original AlexNet architecture C7 (Figure 3, left) and also the potential for over-fitting where significantly increased architectural complexity within a Inception-V1 modular paradigm offers only marginal performance gains (Figure 3, right). Based on these findings, we propose two novel reduced complexity CNN architectures targeted towards performance on the fire detection task (illustrated in Figure 2).

在这个主任务的子集（即25%的训练数据）的实验中，我们可以看到，在这些原来高度复杂的CNN架构中，有明显的过拟合情况，如C2与原始AlexNet架构C7（图3，左），也有潜在的过拟合状况，如Inception-V1模块方案中，显著增长的架构复杂度只得到了一点点的性能增加（图3，右）。基于这些发现，我们提出两种新的降低复杂度的CNN架构，以进行火焰检测任务（如图2所示）。

**FireNet** is based on our C2 AlexNet configuration such that it contains only three convolutional layers of sizes 64, 128, and 256, with kernel filter sizes 5 × 5, 4 × 4, and 1 × 1 respectively. Each convolutional layer is followed by a max pooling layer with a kernel size 3 × 3 and local response normalization. This set of convolutional layers are followed by two fully connected layers, each with 4096 incoming connections and using tanh() activation. Dropout of 0.5 is applied across these two fully connected layers during training to offset residual over-fitting. Finally we have a fully connected layer with 2 incoming connections and soft-max activation output. The architecture of FireNet is illustrated in Figure 2 (left) following the illustrative style of the original AlexNet work to aid comparison.

**FireNet**是基于我们的C2 AlexNet配置，这样就只包含了3个卷积层，卷积核数量64，128和256，卷积核大小分别为5 × 5, 4 × 4和1 × 1。每个卷积层后面都跟着一个max pooling层，大小3 × 3，和局部响应归一化。这些卷积层之后是2个全连接层，每个都有4096个连接，使用tanh()激活函数。训练时，两个全连接层之间使用了0.5的dropout，防止过拟合。最后，有一个全连接层，只有2个连接，和一个softmax激活输出。FireNet的架构如图2左所示，与原始AlexNet的画图风格一致，以便比较。

**InceptionV1-OnFire** is based on the use of a reduced InceptionV1 architecture only with three consecutive inception modules. Each individual module follows the same definition as the original work [29], using these first three in the same interconnected format as in the full InceptionV1 architecture. As shown in Figure 2 (right), following the illustrative style of the original InceptionV1 work to aid comparison, the same unchanged configuration of pre-process and post-process layers are used around this three module set.

**InceptionV1-OnFire**是基于只有3个连续的Inception模块的缩减InceptionV1架构。每个单独的Inception模块都采用原始工作[29]的相同定义，使用完整InceptionV1架构的前3个相同的连接格式。如图2（右）所示，与原始InceptionV1架构的展示风格一致，以便比较，在这三个模块前后，使用了相同的预处理和后处理层。

### 2.3. Superpixel Localization 超像素定位

In contrast to earlier work [17, 8] that largely relies on colour-based initial localization, we instead adopt the use of superpixel regions [24]. Superpixel based techniques over-segment an image into perceptually meaningful regions which are similar in colour and texture (Figure 4). Specifically we use simple linear iterative clustering (SLIC) [24], which essentially adapts the k-means clustering to reduced spatial dimensions, for computational efficiency. An example of superpixel based localization for fire detection is shown in Figure 4A with classification akin to [30, 31] via CNN (Figure 4B).

之前的工作[17,8]主要依靠基于颜色的初始定位，而我们则使用超像素区域[24]。基于超像素的技术对图像进行过分割，成为感知上有意义的区域，区域内的颜色和纹理都是类似的（如图4所示）。具体来说，我们使用简单线性迭代聚类(SLIC)[24]，实质上就是修改了k均值聚类以降低空间维度，以降低计算量。图4A是一个基于超像素的火焰检测定位，图4B则是一个类似的基于CNN的分类例子[30,31]。

## 3. Evaluation 评估

For the comparison of the simplified CNN architectures outlined we consider the True Positive Rate (TPR) and False Positive Rate (FPR) together with the F-score (F), Precision (P) and accuracy (A) statistics in addition to comparison against the state of the art in non-temporal fire detection [17]. We address two problems for the purposes of evaluation:- (a) full-frame binary fire detection (i.e. fire present in the image as whole - yes/no?) and (b) superpixel based fire region localization against ground truth in-frame annotation [23].

为比较上述的简化CNN架构，我们考虑计算其真阳性率(True Positive Rate, TPR)和假阳性率(False Positive Rate, FPR)，以及F-score(F)，精度(Precision, P)和准确率(Accuracy, A)，还有与目前最好的非时域火焰检测算法[17]的比较。为评估，我们解决两个问题：(a)整帧二值火焰检测，即图像中是否存在火焰？(b)基于超像素的火焰区域定位与真值帧内标注[23]。

CNN training and evaluation was performed using fire image data compiled from Chenebert et al. [17] (75,683 images) and also the established visual fire detection evaluation dataset of Steffens et al. [23] (20593 images) in addition to material from public video sources (youtube.com: 269,426 images) to give a wide variety of environments, fires and non-fire examples (total dataset: 365,702 images). From this dataset a training set of 23,408 images was extracted for training and testing a full-frame binary fire detection problem (70:30 data split) with a secondary validation set of 2931 images used for statistical evaluation. Training is from random initialisation using stochastic gradient descent with a momentum of 0.9, a learning rate of 0.001, a batch size of 64 and categorical cross-entropy loss. All networks are trained using a a Nvidia Titan X GPU via TensorFlow (1.1 + TFLearn 0.3).

CNN训练和评估所用的数据集为Chenbert等[17]的火焰图像数据（75683幅图像），和Steffens等[23]构建的视觉火焰检测评估数据集（20593幅图像），和公共视频资源中的材料（youtube.com: 269426幅图像），以得到多种环境的火焰、非火焰样本（数据集共有365702幅图像）。从这个数据集中，提取了一个训练集，包含23408幅图像，测试的是整帧二值火焰检测问题（70:30数据划分），验证集2931幅图像，用于统计评估。训练是从随机初始化开始的，使用随机梯度下降，动量0.9，学习率为0.001，批规模为64，类别交叉熵损失。所有网络都用NVidia Titan X GPU在TensorFlow(1.1 + TFLearn 0.3)上训练

From the results presented in Table 1, addressing the full-frame binary fire detection problem, we can see that the InceptionV1-OnFire architecture matches the maximal performance of its larger parent network InceptionV1 (0.93 accuracy / 0.96 TPR, within 1% on other metrics). Furthermore, we can see a similar performance relationship between the FireNet architecture and its AlexNet parent.

表1所示的是处理全帧二值火焰检测问题的结果，从中我们可以看到，InceptionV1-OnFire架构与原始InceptionV1架构的最高性能是一样的（0.93准确率/0.96TPR，在其他度量上误差不超过1%）。而且，我们在FireNet架构和AlexNet之间也可以看到一种类似的关系。

Table 1. Statistical performance - full-frame fire detection.

| | TPR | FPR | F | P | A
--- | --- | --- | --- | --- | ---
AlexNet | 0.91 | 0.07 | 0.93 | 0.95 | 0.92
InceptionV1 | 0.96 | 0.09 | 0.95 | 0.94 | 0.93
VGG-13 | 0.93 | 0.11 | 0.93 | 0.92 | 0.91
FireNet | 0.92 | 0.09 | 0.93 | 0.93 | 0.92
InceptionV1-OnFire | 0.96 | 0.10 | 0.94 | 0.93 | 0.93

Computational performance at run-time was performed using at average of 100 image frames of 608 × 360 RGB colour video on a Intel Core i5 2.7GHz CPU and 8GB of RAM. The resulting frames per second (fps) together with a measure of architecture complexity (parameter complexity, C), percentage accuracy (A) and ratio A : C are shown in Table 2. From the results presented in Table 2, we observe significant run-time performance gains for the reduced complexity FireNet and InceptionV1-OnFire architectures compared to their parent architectures. Whilst FireNet provides a maximal 17 fps throughput, it is notable that InceptionV1-OnFire provides the maximal accuracy to complexity ratio. Whilst the accuracy of FireNet is only slightly worse than that of AlexNet, it can perform a classification 4.2× times faster. Similarly InceptionV1-OnFire matches the accuracy of InceptionV1 but can perform a classification 3.3× faster.

运行时的计算性能是这样计算的，处理100幅从608×360大小的视频帧中提取出的帧图像求平均，处理平台为Intel Core i5 2.7GHz CPU和8GB RAM。得到的FPS与架构复杂度（参数复杂度，C），准确率(A)和比率A:C，如表2所示。从表2中的结果可以看到，降低复杂度的FireNet和InceptionV1-OnFire架构，与其父架构相比，运行时性能有显著的提升。FireNet最高达到17fps，而InceptionV1-OnFire得到了最高的准确度复杂度比率。FireNet只比AlexNet在准确度上略微差了一点，其分类速度快了4.2倍。类似的，InceptionV1-OnFire与InceptionV1准确度类似，但速度快了3.3倍。

Table 2. Statistical results - size, accuracy and speed (fps).

| | C($×10^6$) | A(%) | A:C | fps
--- | --- | --- | --- | ---
AlexNet | 71.9 | 91.7 | 1.3 | 4.0
FireNet | 68.3 | 91.5 | 1.3 | 17.0
InceptionV1 | 6.1 | 93.4 | 15.4 | 2.6
InceptionV1-OnFire | 1.2 | 93.4 | 77.9 | 8.4
Chenbert et al. [17] | - | - | - | 0.16

To evaluate within the context of in-frame localization (Section 2.3), we utilise the ground truth annotation available from Steffens et al. [23] to label image superpixels for training, test and validation. The InceptionV1-OnFire architecture is trained over a set of 54,856 fire (positive) and 167,400 non-fire (negative) superpixel examples extracted from 90% of the image frames within [23]. Training is performed as per before with validation against the remaining 10% of frames comprising 1178 fire (positive) and 881 non-fire (negative) examples. The resulting contour from any fire detected superpixels is converted to a bounding rectangle and tested for intersection with the ground truth annotation (Similarity, S: correct if union over ground truth>0.5 as per [23]). From the results presented in Table 3 (lower), we can see that the combined localization approach of superpixel region identification and localized InceptionV1-OnFire CNN classification performs marginally worse than the competing state of the art Chenebert et al. [17] but matching overall full-frame detection (Table 3, upper). However, as can be seen from Table 2, this prior work [17] has significantly worse computational throughput than any of the CNN approaches proposed here. Example detection and localization are shown in Figures 1 and 4B (fire = green, no-fire = red).

为评估帧内定位的性能（2.3节），我们利用了Steffens等[23]提供的真值标注，来标记图像超像素进行训练、测试和评估。InceptionV1-OnFire架构在54856个火焰样本（正样本）和167400个非火焰（负样本）超像素样本中进行训练，这是从[23]中90%的图像帧得到的样本。剩余10%的帧，包括1178个火焰样本（正）和881个非火焰样本（负）为验证集。检测得到的任何火焰超像素，都转换成边界框矩形，与真值标注之间计算交集（相似度，S为真，如果与真值间的交集比率高于50%，如[23]中一样）。如表3（下）中所示，我们可以看到，超像素区域辨识和局部InceptionV1-OnFire CNN分类构成的定位方法，与目前最好的Chenbert等[17]相比，略差一点，但整体的整帧检测性能类似（表3，上）。但是，就像我们在表2中看到的一样，这个之前的工作[17]计算性能要差的多。样本检测和定位如图1和图4B所示（火为绿色，非火为红色）。

Table 3. Statistical results - localization

Detection(full-frame) | TPR | FPR | F | P | A
--- | --- | --- | --- | --- | ---
Chenbert et al.[17] | 0.99 | 0.28 | 0.92 | 0.86 | 0.89
InceptionV1-OnFire | 0.92 | 0.17 | 0.90 | 0.88 | 0.89

Localization(pixel region) | TPR | F | P | S
--- | --- | --- | --- | ---
Chenbert et al.[17] | 0.98 | 0.90 | 0.83 | 0.80
InceptionV1-OnFire | 0.92 | 0.88 | 0.84 | 0.78

## 4. Conclusions 结论

Overall we show that reduced complexity CNN, experimentally defined from leading architectures in the field, can achieve 0.93 accuracy for the binary classification task of fire detection. This significantly outperforms prior work in the field on non-temporal fire detection [17] at lower complexity than prior CNN based fire detection [22]. Furthermore, reduced complexity FireNet and InceptionV1-OnFire architectures offer classification accuracy within less than 1% of their more complex parent architectures at 3-4× of the speed (FireNet offering 17 fps). To these ends, we illustrate more generally a architectural reduction strategy for the experimentally driven complexity reduction of leading multi-class CNN architectures towards efficient, yet robust performance on simpler binary classification problems.

整体上，我们给出了降低复杂度的CNN，这是从领先的架构中通过实验确定的，在火焰检测二值分类任务中得到了93%的准确率。这显著高于之前这个领域的非时域火焰检测工作结果[17]，计算复杂度比之前基于CNN的火焰检测[22]要低很多。而且，降低复杂度的FireNet和InceptionV1-OnFire架构得到的分类准确率，比其父架构降低了不到1%，但是计算速度提高了3-4倍（FireNet给出了17fps的结果）。为此，我们给出了更一般的架构精简策略，即通过实验确定这些领先的多类CNN架构的复杂度精简方法，在二值分类问题中形成高效但是性能稳健的模型。