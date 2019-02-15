# Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

Kaipeng Zhang et al.

## Abstract 摘要

Face detection and alignment in unconstrained environment are challenging due to various poses, illuminations and occlusions. Recent studies show that deep learning approaches can achieve impressive performance on these two tasks. In this paper, we propose a deep cascaded multi-task framework which exploits the inherent correlation between them to boost up their performance. In particular, our framework adopts a cascaded structure with three stages of carefully designed deep convolutional networks that predict face and landmark location in a coarse-to-fine manner. In addition, in the learning process, we propose a new online hard sample mining strategy that can improve the performance automatically without manual sample selection. Our method achieves superior accuracy over the state-of-the-art techniques on the challenging FDDB and WIDER FACE benchmark for face detection, and AFLW benchmark for face alignment, while keeps real time performance.

在不受限制的环境中进行人脸检测和人脸对齐非常有挑战性，因为姿态、光照和遮挡等情况众多。最近的研究显示深度学习方法可以在这两种任务中取得让人印象深刻的效果。在本文中，我们提出了一种深度级联多任务框架，探索了两个任务的内在联系，利用其提升性能。特别的，我们的框架采用了一种级联结构，是三个阶段仔细设计的深度卷积网络，以从粗糙到精细的方式来预测人脸和特征点的位置。另外，在学习的过程中，我们提出一种新的在线难分样本挖掘策略，可以自动改善性能，不用手动的选取样本。我们的方法与目前最好的方法相比，在FDDB和WIDER FACE的人脸检测基准数据集和AFLW人脸对齐数据集中，取得了更好的准确率性能，同时还保持了实时的性能。

**Index Terms** - Face detection, face alignment, cascaded convolutional neural network

**关键词** - 人脸检测，人脸对齐，级联卷积神经网络

## I. Introduction 引言

Face detection and alignment are essential to many face applications, such as face recognition and facial expression analysis. However, the large visual variations of faces, such as occlusions, large pose variations and extreme lightings, impose great challenges for these tasks in real world applications.

人脸检测和对齐对于许多人脸应用来说非常关键，比如人脸识别和人脸表情分析。但是，人脸的巨大视觉差异，比如遮挡、大的姿态变化和极端的光照，会对真实世界的应用施加巨大的挑战。

The cascade face detector proposed by Viola and Jones [2] utilizes Haar-Like features and AdaBoost to train cascaded classifiers, which achieve good performance with real-time efficiency. However, quite a few works [1, 3, 4] indicate that this detector may degrade significantly in real-world applications with larger visual variations of human faces even with more advanced features and classifiers. Besides the cascade structure, [5, 6, 7] introduce deformable part models (DPM) for face detection and achieve remarkable performance. However, they need high computational expense and may usually require expensive annotation in the training stage. Recently, convolutional neural networks (CNNs) achieve remarkable progresses in a variety of computer vision tasks, such as image classification [9] and face recognition [10]. Inspired by the good performance of CNNs in computer vision tasks, some of the CNNs based face detection approaches have been proposed in recent years. Yang et al. [11] train deep convolution neural networks for facial attribute recognition to obtain high response in face regions which further yield candidate windows of faces. However, due to its complex CNN structure, this approach is time costly in practice. Li et al. [19] use cascaded CNNs for face detection, but it requires bounding box calibration from face detection with extra computational expense and ignores the inherent correlation between facial landmarks localization and bounding box regression.

Viola和Jones[2]提出的级联人脸检测器利用了类Harr特征和Adaboost来训练级联分类器，算法以实时的效率得到了很好的效果。但是，不少工作[1,3,4]指出，在真实世界应用中，人脸有较大的视觉变化时，这种检测器即使使用更高级的特征和分类其，也会性能会显著下降。除了这种级联结构，[5,6,7]还提出了可变部件模型(DPM)进行人脸检测，取得了令人印象深刻的性能。但是，这需要很好的计算代价，而且在训练阶段通常需要昂贵的标注。最近，卷积神经网络(CNNs)在很多计算机视觉任务中取得了令人印象深刻的进展，比如图像分类[9]和人脸识别[10]。受到CNNs在计算机视觉中的优秀性能的启发，近年来提出了一些基于CNNs的人脸检测方法。Yang等[11]训练了深度卷积神经网络进行人脸属性识别，以在人脸区域中得到更高的响应，这会进一步生成人脸的候选窗口。但是，由于其复杂的CNN结构，这种方法在实践中非常耗时。Li等[19]使用了级联CNNs进行人脸检测，但需要人脸检测的边界框校准，需要额外的计算代价，忽略了人脸特征点的定位和边界框回归的内在关联。

Face alignment also attracts extensive interests. Regression-based methods [12, 13, 16] and template fitting approaches [14, 15, 7] are two popular categories. Recently, Zhang et al. [22] proposed to use facial attribute recognition as an auxiliary task to enhance face alignment performance using deep convolutional neural network.

人脸对齐也吸引了广泛的关注。基于回归的方法[12,13,16]和模板匹配方法[14,15,7]是两个重要的类别。最近，Zhang等[22]提出了使用人脸属性识别作为辅助任务来增强使用卷积神经网络的人脸对齐的性能。

However, most of the available face detection and face alignment methods ignore the inherent correlation between these two tasks. Though there exist several works attempt to jointly solve them, there are still limitations in these works. For example, Chen et al. [18] jointly conduct alignment and detection with random forest using features of pixel value difference. But, the handcraft features used limits its performance. Zhang et al. [20] use multi-task CNN to improve the accuracy of multi-view face detection, but the detection accuracy is limited by the initial detection windows produced by a weak face detector.

但是，大多可用的人脸检测和人脸对齐方法都忽略了这两个任务的内在关联。虽然有几项工作尝试同时解决这两个问题，但这些工作都有一些局限，比如，Chen等[18]使用随机森林利用像素值差异的特征同时进行对齐和检测。但是，使用手工设计的特征限制了其性能。Zhang等[20]使用多任务CNN来改进多视角人脸检测的准确率，但检测准确率被一个弱人脸检测器生成的初始检测窗口所限制。

On the other hand, in the training process, mining hard samples in training is critical to strengthen the power of detector. However, traditional hard sample mining usually performs an offline manner, which significantly increases the manual operations. It is desirable to design an online hard sample mining method for face detection and alignment, which is adaptive to the current training process automatically.

另一方面，在训练过程中，在训练过程中挖掘难分样本对增强检测器的能力非常关键。但是，传统的难分样本挖掘通常都是离线模式进行的，这显著增加了手动运算量。设计一个在线的难分样本挖掘方法对于人脸检测和对齐来说非常理想，可以自动的适应当前的训练过程。

In this paper, we propose a new framework to integrate these two tasks using unified cascaded CNNs by multi-task learning. The proposed CNNs consist of three stages. In the first stage, it produces candidate windows quickly through a shallow CNN. Then, it refines the windows to reject a large number of non-faces windows through a more complex CNN. Finally, it uses a more powerful CNN to refine the result and output facial landmarks positions. Thanks to this multi-task learning framework, the performance of the algorithm can be notably improved. The major contributions of this paper are summarized as follows: (1) We propose a new cascaded CNNs based framework for joint face detection and alignment, and carefully design lightweight CNN architecture for real time performance. (2) We propose an effective method to conduct online hard sample mining to improve the performance. (3) Extensive experiments are conducted on challenging benchmarks, to show the significant performance improvement of the proposed approach compared to the state-of-the-art techniques in both face detection and face alignment tasks.

在本文中，我们提出一个新的框架来整合这两种任务，使用的是统一的级联CNN进行多任务学习。提出的CNNs包括三个阶段。在第一阶段，通过一个浅层CNN迅速生成候选窗口。然后，进行窗口提炼，通过一个更复杂的CNN拒绝了大量非人脸窗口。最后，使用了一个更强大的CNN来提炼结果，输出人脸特征点位置。使用这种多任务学习框架，算法的性能可以得到显著改进。本文的主要贡献总结如下：(1)我们提出一个新的基于级联CNNs的框架同时进行人脸检测和对齐，同时仔细设计轻量级CNN架构得到实时性能；(2)我们提出一种有效的方法来进行在线难分样本挖掘来改进性能；(3)在几个有挑战的数据集上进行广泛的实验，与现在最好的方法相比，我们提出的方法在人脸检测和人脸对齐任务中有显著改进的性能。

## II. Approach 方法

In this section, we will describe our approach towards joint face detection and alignment. 在本节中，我们描述一下提出的同时进行人脸检测和对齐的方法。

### A. Overall Framework 总体框架

The overall pipeline of our approach is shown in Fig. 1. Given an image, we initially resize it to different scales to build an image pyramid, which is the input of the following three-stage cascaded framework:

我们方法总体的流程如图1所示。给定一幅图像，我们开始先将其变换到不同的尺度大小，构建一个图像金字塔，然后作为下面的三阶段级联架构的输入：

Stage 1: We exploit a fully convolutional network[?], called Proposal Network (P-Net), to obtain the candidate windows and their bounding box regression vectors in a similar manner as [29]. Then we use the estimated bounding box regression vectors to calibrate the candidates. After that, we employ non-maximum suppression (NMS) to merge highly overlapped candidates.

阶段1：我们利用一个全卷积网络，称为候选网络(P-Net)，像[29]一样得到候选窗口和其边界框回归向量。然后我们使用估计的边界框回归向量来校准候选。在这之后，我们使用非最大抑制(NMS)来合并高度重叠的候选。

Stage 2: all candidates are fed to another CNN, called Refine Network (R-Net), which further rejects a large number of false candidates, performs calibration with bounding box regression, and NMS candidate merge.

阶段2：所有候选都送入另一个CNN，称为提炼网络(R-Net)，这会进一步拒绝大量假候选，用边界框候选进行校准，以及NMS候选合并。

Stage 3: This stage is similar to the second stage, but in this stage we aim to describe the face in more details. In particular, the network will output five facial landmarks’ positions.

阶段3：这个阶段与第二阶段类似，但在这个阶段我们的目标是更细节的描述人脸。特别是，网络会输出5个脸部重要标志位置。

Fig. 1. Pipeline of our cascaded framework that includes three-stage multi-task deep convolutional networks. Firstly, candidate windows are produced through a fast Proposal Network (P-Net). After that, we refine these candidates in the next stage through a Refinement Network (R-Net). In the third stage, The Output Network (O-Net) produces final bounding box and facial landmarks position.

图1. 我们的级联框架的流程，包括三阶段多任务深度卷积网络。首先，通过一个快速候选网络(P-Net)生成候选窗口。在这之后，我们在下一个阶段通过一个提炼网络(R-Net)提炼这些候选。在第三个阶段，输出网络(O-Net)生成最终的边界框和人脸特征点位置。

### B. CNN Architectures CNN架构

In [19], multiple CNNs have been designed for face detection. However, we noticed its performance might be limited by the following facts: (1) Some filters lack diversity of weights that may limit them to produce discriminative description. (2) Compared to other multi-class objection detection and classification tasks, face detection is a challenge binary classification task, so it may need less numbers of filters but more discrimination of them. To this end, we reduce the number of filters and change the 5×5 filter to a 3×3 filter to reduce the computing while increase the depth to get better performance. With these improvements, compared to the previous architecture in [19], we can get better performance with less runtime (the result is shown in Table 1. For fair comparison, we use the same data for both methods). Our CNN architectures are showed in Fig. 2.

在[19]中设计了多个CNNs来进行人脸检测。但是，我们注意到其性能受到下面的事实的限制：(1)一些滤波器缺少权重的多样性，这会限制其产生有区分能力的描述；(2)与其他多类别目标识别和分类任务相比，人脸检测是二值分类任务，所以需要的滤波器数量教少，但是需要滤波器更具有区分性。为此，我们减少滤波器数量，并将5×5的滤波器更换为3×3的滤波器，来减少计算量，同时得到更好的性能。通过这些改进，与之前[19]的架构相比，我们可以在更少的运行时间内得到更好的性能（结果如表1所示，为公平对比，我们对两种方法使用相同的数据）。我们的CNN架构如图2所示。

Table I. Comparison of speed and validation accuracy of our CNNs and previous CNNs [19]

Group | CNN | 300 Times Forward | Accuracy
--- | --- | --- | ---
Group1 | 12-Net [19] | 0.038s | 94.4%
Group1 | P-Net | 0.031s | 94.6%
Group2 | 24-Net [19] | 0.738s | 95.1%
Group2 | R-Net | 0.458s | 95.4%
Group3 | 48-Net [19] | 3.577s | 93.2%
Group3 | O-Net | 1.347s | 95.4%

### C. Training 训练

We leverage three tasks to train our CNN detectors: face/non-face classification, bounding box regression, and facial landmark localization. 我们利用三类任务才能训练我们的CNN检测器：人脸/非人脸分类，边界框回归，和人脸特征点定位。

1) Face classification: The learning objective is formulated as a two-class classification problem. For each sample, we use the cross-entropy loss: 人脸分类：学习目标函数为两类分类问题。对于每个样本，我们使用交叉熵损失：

$$L_i^{det} = -(y_i^{det} log(p_i) + (1-y_i^{det}) (1-log(p_i)))$$(1)

where $p_i$ is the probability produced by the network that indicates a sample being a face. The notation $y_i^{det} ∈ \{0,1\}$ denotes the ground-truth label. 其中$p_i$是网络生成的概率，指示样本为一张脸的概率。符号$y_i^{det} ∈ \{0,1\}$表示真值标签。

2) Bounding box regression: For each candidate window, we predict the offset between it and the nearest ground truth (i.e., the bounding boxes’ left top, height, and width). The learning objective is formulated as a regression problem, and we employ the Euclidean loss for each sample $x_i$:

2) 边界框回归：对每个候选窗口，我们预测其与最接近的真值间的偏移（即，边界框的左上角，高度和宽度）。学习目标函数是一个回归问题，对每个样本$x_i$，我们采用欧几里得损失：

$$L_i^{box} = ||\hat y_i^{box} - y_i^{box}||_2^2$$(2)

where ̂$\hat y_i^{box}$ regression target obtained from the network and $y_i^{box}$ is the ground-truth coordinate. There are four coordinates, including left top, height and width, and thus $y_i^{box} ∈ R^4$. 其中$\hat y_i^{box}$是网络得到的目标回归值，$y_i^{box}$为真值坐标。有4个坐标值，包括左上角的坐标，高度和宽度，即$y_i^{box} ∈ R^4$。

3) Facial landmark localization: Similar to the bounding box regression task, facial landmark detection is formulated as a regression problem and we minimize the Euclidean loss: 人脸特征点定位：与边界框回归任务类似，人脸特征点检测也可以看作一个回归问题，最小化如下欧几里得损失：

$$L_i^{landmark} = ||\hat y_i^{landmark} - y_i^{landmark}||_2^2$$(3)

where $\hat y_i^{landmark}$ is the facial landmark’s coordinate obtained from the network and $y_i^{landmark}$ is the ground-truth coordinate. There are five facial landmarks, including left eye, right eye, nose, left mouth corner, and right mouth corner, and thus $y_i^{landmark} ∈ R^{10}$.

其中$\hat y_i^{landmark}$是通过网络得到的人脸特征点的坐标，$y_i^{landmark}$是真值坐标。有5个人脸特征点，包括左眼，右眼，鼻子，左边嘴角，右边嘴角，所以$y_i^{landmark} ∈ R^{10}$。

4) Multi-source training: Since we employ different tasks in each CNNs, there are different types of training images in the learning process, such as face, non-face and partially aligned face. In this case, some of the loss functions (i.e., Eq. (1)-(3) ) are not used. For example, for the sample of background region, we only compute $L_i^{det}$, and the other two losses are set as 0. This can be implemented directly with a sample type indicator. Then the overall learning target can be formulated as:

4) 多源训练：由于我们在每个CNNs中进行多个任务，在训练过程中，就有不同类型的训练数据，比如人脸、非人脸和部分对齐的人脸。在这种情况下，一些损失函数（即式1-3）没有使用。比如，对于背景区域样本，我们只计算$L_i^{det}$，另外两个损失函数设为0。这可以直接通过一个样本类型指示器来实现。那么整体的学习目标可以表示为：

$$min \sum_{i=1}^N \sum_{j∈\{ det,box,landmark\}} α_j β_i^j L_i^j$$(4)

where N is the number of training samples. $α_j$ denotes on the task importance. We use $α_{det} = 1, α_{box} = 0.5, α_{landmark} = 0.5$ in P-Net and R-Net, while $α_{det} = 1, α_{box} = 0.5, α_{landmark} = 1$ in O-Net for more accurate facial landmarks localization. $β_i^j ∈ \{0,1\}$ is the sample type indicator. In this case, it is natural to employ stochastic gradient descent to train the CNNs.

其中N是训练样本数量。$α_j$表示任务的重要性，我们在P-Net和R-Net中使用$α_{det} = 1, α_{box} = 0.5, α_{landmark} = 0.5$，在O-Net中使用$α_{det} = 1, α_{box} = 0.5, α_{landmark} = 1$，以得到更精确的人脸特征点位置。$β_i^j ∈ \{0,1\}$是样本类型指示器。在这个例子中，很自然可以使用随机梯度下降法来训练CNNs。

5) Online Hard sample mining: Different from conducting traditional hard sample mining after original classifier had been trained, we do online hard sample mining in face classification task to be adaptive to the training process.

5) 在线难分样本挖掘：传统的难分样本挖掘是在原始分类器训练好之后进行，与之不同，我们在人脸分类任务中进行在线难分样本挖掘，以自适应训练过程。

In particular, in each mini-batch, we sort the loss computed in the forward propagation phase from all samples and select the top 70% of them as hard samples. Then we only compute the gradient from the hard samples in the backward propagation phase. That means we ignore the easy samples that are less helpful to strengthen the detector while training. Experiments show that this strategy yields better performance without manual sample selection. Its effectiveness is demonstrated in the Section III.

特别的，在每个mini-batch，我们对所有样本计算前向传播过程的损失并排序，选取其中的前70%作为难分样本。然后我们在反向传播过程中只计算难分样本的梯度。那意味着我们忽略容易的样本，它们对于在训练过程中加强分类器的帮助不大。实验表明，这种策略会得到更好的性能，而不用手工选择样本。其有效性在第III部分进行展示。

## III. Experiments 实验

In this section, we first evaluate the effectiveness of the proposed hard sample mining strategy. Then we compare our face detector and alignment against the state-of-the-art methods in Face Detection Data Set and Benchmark (FDDB) [25], WIDER FACE [24], and Annotated Facial Landmarks in the Wild (AFLW) benchmark [8]. FDDB dataset contains the annotations for 5,171 faces in a set of 2,845 images. WIDER FACE dataset consists of 393,703 labeled face bounding boxes in 32,203 images where 50% of them for testing into three subsets according to the difficulty of images, 40% for training and the remaining for validation. AFLW contains the facial landmarks annotations for 24,386 faces and we use the same test subset as [22]. Finally, we evaluate the computational efficiency of our face detector.

在本节中，我们首先评估了提出的难分样本挖掘策略的有效性。然后我们将我们的人脸检测器和人脸对齐算法与目前最好的方法进行了比较，所用的数据集为人脸检测数据集和基准测试(FDDB)[25]、WIDER FACE[24]和标注野外人脸特征点(AFLW)[8]。FDDB数据集包含2845幅图像，5171个标注的人脸。WIDER FACE数据集包含32203幅图像，393703个标注的人脸边界框，其中40%是训练样本，10%为验证集，剩下50%按照难度分为三个测试集。AFLW包括24386个人脸特征点标注，我们使用其子集，与[22]相同。最后，我们评估了我们人脸检测器的计算效率。

### A. Training Data 训练数据

Since we jointly perform face detection and alignment, here we use four different kinds of data annotation in our training process: (i) Negatives: Regions that the Intersection-over-Union (IoU) ratio less than 0.3 to any ground-truth faces; (ii) Positives: IoU above 0.65 to a ground truth face; (iii) Part faces: IoU between 0.4 and 0.65 to a ground truth face; and (iv) Landmark faces: faces labeled 5 landmarks’ positions. Negatives and positives are used for face classification tasks, positives and part faces are used for bounding box regression, and landmark faces are used for facial landmark localization. The training data for each network is described as follows:

既然我们同时进行人脸检测和对齐，这里我们在训练过程中使用四个不同类型的数据标注：(i)负样本：与任何真值人脸的IOU比率都小于0.3的区域；(ii)正样本：与某个真值人脸的IOU比率大于0.65；(iii)部分人脸：与某个真值人脸的IOU比率在0.4与0.65之间；(iv)特征点人脸：标注了5个特征点位置的人脸。负样本和正样本用于人脸分类任务，正样本和部分人脸样本用于边界框回归，特征点人脸用于人脸特征点定位。每个网络的训练数据描述如下：

1) P-Net: We randomly crop several patches from WIDER FACE [24] to collect positives, negatives and part face. Then, we crop faces from CelebA [23] as landmark faces; 我们从WIDER FACE[24]中随机剪切出若干图像块来收集正样本、负样本和部分人脸样本。然后，我们从CelebA[23]剪切出人脸作为特征点人脸；

2) R-Net: We use first stage of our framework to detect faces from WIDER FACE [24] to collect positives, negatives and part face while landmark faces are detected from CelebA [23]. 我们使用框架的第一阶段从WIDER FACE[24]中检测人脸来收集正样本、负样本和部分人脸样本，从CelebA[23]中检测特征点人脸。

3) O-Net: Similar to R-Net to collect data but we use first two stages of our framework to detect faces. 与R-Net类似的收集数据，但是我们使用框架的前两阶段来检测人脸。

### B. The effectiveness of online hard sample mining 在线难分样本挖掘的有效性

To evaluate the contribution of the proposed online hard sample mining strategy, we train two O-Nets (with and without online hard sample mining) and compare their loss curves. To make the comparison more directly, we only train the O-Nets for the face classification task. All training parameters including the network initialization are the same in these two O-Nets. To compare them easier, we use fix learning rate. Fig. 3 (a) shows the loss curves from two different training ways. It is very clear that the hard sample mining is beneficial to performance improvement.

为评估提出的在线难分样本挖掘策略的贡献，我们训练了2个O-Nets（分别是使用了和没使用在线难分样本挖掘的网络），并比较其损失曲线。为使这个比较更直接，我们只训练了O-Nets进行人脸分类任务。所有训练参数包括网络初始化在这两个O-Nets中都是一样的。为更简单的比较，我们使用固定的学习速率。图3(a)展示了两种不同训练方式的损失曲线。非常清楚的可以看到，在线难分样本挖掘对于性能改进是有好处的。

Fig. 3. (a) Validation loss of O-Net with and without hard sample mining. (b) “JA” denotes joint face alignment learning while “No JA” denotes do not joint it. “No JA in BBR” denotes do not joint it while training the CNN for bounding box regression.

图3.(a)包含与不包含难分样本挖掘和的O-Net验证损失；(b)JA表示同时进行人脸对齐学习，No JA表示没有同时进行人脸对齐。No JA in BBR表示没有同时进行人脸对齐，而且训练CNN进行边界框回归。

### C. The effectiveness of joint detection and alignment 同时训练和对齐的有效性

To evaluate the contribution of joint detection and alignment, we evaluate the performances of two different O-Nets (joint facial landmarks regression task and do not joint it) on FDDB (with the same P-Net and R-Net for fair comparison). We also compare the performance of bounding box regression in these two O-Nets. Fig. 3 (b) suggests that joint landmarks localization task learning is beneficial for both face classification and bounding box regression tasks.

为评估同时检测和对齐的贡献，我们在FDDB上评估两个不同的O-Nets的性能（同时进行人脸特征点回归任务，以及不同时进行这个任务）（为公平进行比较，使用一样的P-Net和R-Net）。我们还比较了这两个O-Nets边界框回归的性能。图3(b)说明同时进行特征点定位任务的学习对于人脸分类和边界框回归任务都是有好处的。

### D. Evaluation on face detection 人脸检测的评估

To evaluate the performance of our face detection method, we compare our method against the state-of-the-art methods [1, 5, 6, 11, 18, 19, 26, 27, 28, 29] in FDDB, and the state-of-the-art methods [1, 24, 11] in WIDER FACE. Fig. 4 (a)-(d) shows that our method consistently outperforms all the previous approaches by a large margin in both the benchmarks. We also evaluate our approach on some challenge photos(Examples are showed in http://kpzhang93.github.io/SPL/index.html).

为评估我们人脸检测方法的性能，我们将我们的方法与在FDDB上目前最好的方法[1,5,6,11,18,19,26,27,28,29]进行了比较，在WIDER FACE上目前最好的方法[1,24,11]也进行了比较。图4(a)-(d)显示我们的方法在两个基准测试中超过了所有之前的方法很多。我们还在一些有挑战性的图像中评估我们的方法（例子见链接）。

Fig. 4. (a) Evaluation on FDDB. (b-d) Evaluation on three subsets of WIDER FACE. The number following the method indicates the average accuracy. (e) Evaluation on AFLW for face alignment

### E. Evaluation on face alignment 人脸对齐的评估

In this part, we compare the face alignment performance of our method against the following methods: RCPR [12], TSPM [7], Luxand face SDK [17], ESR [13], CDM [15], SDM [21], and TCDCN [22]. In the testing phase, there are 13 images that our method fails to detect face. So we crop the central region of these 13 images and treat them as the input for O-Net. The mean error is measured by the distances between the estimated landmarks and the ground truths, and normalized with respect to the inter-ocular distance. Fig. 4 (e) shows that our method outperforms all the state-of-the-art methods with a margin.

在本部分中，我们比较了我们方法与如下方法在人脸对齐上的性能：RCPR[12]、TSPM[7]、Luxand face SDK[17]，ESR[13]，CDM[15]，SDM[21]和TCDCN[22]。在测试阶段，有13幅图像没有检测到人脸。所以我们剪切这13幅图像的中间区域，作为O-Net的输入。平均误差由估计的特征点和真值特征点的距离估计得到，并根据双眼间距离进行归一化。图4(e)表明，我们的方法超过了目前最好的方法不少。

### F. Runtime efficiency

Given the cascade structure, our method can achieve very fast speed in joint face detection and alignment. It takes 16fps on a 2.60GHz CPU and 99fps on GPU (Nvidia Titan Black). Our implementation is currently based on un-optimized MATLAB code.

给定级联结构，我们的方法可以以很快的速度进行同时人脸检测和对齐。在2.60GHz的CPU的速度为16fps，在Nvidia Titan Black GPU上为99fps。我们的实现目前是基于未经优化的MATLAB代码。

## IV. Conclusion 结论

In this paper, we have proposed a multi-task cascaded CNNs based framework for joint face detection and alignment. Experimental results demonstrate that our methods consistently outperform the state-of-the-art methods across several challenging benchmarks (including FDDB and WIDER FACE benchmarks for face detection, and AFLW benchmark for face alignment) while keeping real time performance. In the future, we will exploit the inherent correlation between face detection and other face analysis tasks, to further improve the performance.

在本文中，我们提出了一种多任务基于级联CNNs框架同时进行人脸检测和对齐。实验结果表明，我们的方法在几个有挑战性的基准测试中全都超过了目前最好的方法（包括FDDB和WIDER FACE人脸检测基准测试，和AFLW人脸对齐基准测试），同时保持了实时的性能。在将来，我们会发掘人脸检测和其他人脸分析任务的内在联系，以进一步改进其性能。