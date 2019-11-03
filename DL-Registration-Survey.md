# Deep Learning in Medical Image Registration: A Survey

Grant Haskins et al. Rensselaer Polytechnic Institute

## 0 Abstract

The establishment of image correspondence through robust image registration is critical to many clinical tasks such as image fusion, organ atlas creation, and tumor growth monitoring, and is a very challenging problem. Since the beginning of the recent deep learning renaissance, the medical imaging research community has developed deep learning based approaches and achieved the state-of-the-art in many applications, including image registration. The rapid adoption of deep learning for image registration applications over the past few years necessitates a comprehensive summary and outlook, which is the main scope of this survey. This requires placing a focus on the different research areas as well as highlighting challenges that practitioners face. This survey, therefore, outlines the evolution of deep learning based medical image registration in the context of both research challenges and relevant innovations in the past few years. Further, this survey highlights future research directions to show how this field may be possibly moved forward to the next level.

通过图像配准确定图像的对应性，是很多临床任务的关键，如图像融合，生成器官图谱，和肿瘤生长的监控，是一个很难的问题。自从最近深度学习复兴的开始，医学图像研究团体就提出了很多基于深度学习的方法，在很多应用中取得了很多当时最好的结果，包括图像配准。基于深度学习的图像配准需要一个综述。这需要关注一些不同的研究领域，并关注人们面临的挑战。因此，本文给出了基于深度学习的医学图像配准的发展。而且，本文关注了未来的研究方向，展望了这个领域未来可能怎样更上一个台阶。

## 1 Introduction

Image registration is the process of transforming different image datasets into one coordinate system with matched imaging contents, which has significant applications in medicine. Registration may be necessary when analyzing a pair of images that were acquired from different viewpoints, at different times, or using different sensors/modalities [47, 146]. Until recently, image registration was mostly performed manually by clinicians. However, many registration tasks can be quite challenging and the quality of manual alignments are highly dependent upon the expertise of the user, which can be clinically disadvantageous. To address the potential shortcomings of manual registration, automatic registration has been developed. Although other methods for automatic image registration have been extensively explored prior to (and during) the deep learning renaissance, deep learning has changed the landscape of image registration research [3]. Ever since the success of AlexNet in the ImageNet challenge of 2012 [2], deep learning has allowed for state-of-the-art performance in many computer vision tasks including, but not limited to: object detection [100], feature extraction [43], segmentation [103], image classification [2], image denoising [135], and image reconstruction [138].

图像配准是将不同的图像数据集转换到同一个坐标系统中，匹配图像内容的过程，在医学中有很多应用。在分析从不同角度得到的图像时，在不同时间得到的图像时，或使用不同的传感器/模态得到的图像时，是非常必须的。直到最近，图像配准都是由临床医生手工进行的。但是，很多配准任务可能很困难，手工配准的质量高度依赖于用户的专业程度，这在临床上是不利的。为解决手工配准的问题，就提出了自动配准。虽然在深度学习复兴之前（之中），有很多其他方法的自动配准，但深度学习改变了这个状况。自从AlexNet在2012年ImageNet挑战赛上的成功，深度学习在很多计算机视觉任务中得到了最好的性能，包括：目标检测，特征提取，分割，图像分类，图像去噪，和图像重建。

Initially, deep learning was successfully used to augment the performance of iterative, intensity based registration. Soon after this initial application, several groups investigated the intuitive application of reinforcement learning to registration. Further, demand for faster registration methods later motivated the development of deep learning based one-step transformation estimation techniques. The challenges associated with procuring/generating ground truth data have recently motivated many groups to develop unsupervised frameworks for one-step transformation estimation. One of the hurdles associated with this framework is the familiar challenge of image similarity quantification. Recent efforts that use information theory based similarity metrics, segmentations of anatomical structures, and generative adversarial network like frameworks to address this challenge have shown promising results. As the trends visualized in Figures 1 and 2 suggest, this field is moving very quickly to surmount the hurdles associated with deep learning based medical image registration and several groups have already enjoyed significant successes for their applications.

最初，深度学习成功的用于改进迭代的、基于灰度的配准的性能。在最初的应用后，几个小组研究了强化学习在配准中的应用。而且，快速配准方法的需求，推动了基于深度学习的一步变换估计技术的提出与发展。真值数据的获得非常困难，这促使很多小组提出无监督的一步变换估计的框架。这种框架相关的困难是，图像相似度量化的困难。最近使用了很多基于信息理论的相似度度量，解剖结构的分割，和GAN类的框架，来解决这些挑战，得到了有希望的结果。这些趋势如图1和图2所示，这个领域正迅速的解决与深度学习图像配准相关的困难，几个小组在其应用中得到了显著的成功。

Therefore, the purpose of this article is to comprehensively survey the field of deep learning based medical image registration, highlight common challenges that practitioners face, and discuss future research directions that may address these challenges. Prior to surveying deep learning based medical image registration works, background information pertaining to deep learning is discussed in Section 2. The methods surveyed in this article were divided into the following three categories: Deep Iterative Registration, Supervised Transformation Estimation, and Unsupervised Transformation Estimation. Following a discussion of the methods that belong to each of the aforementioned categories in Sections 3, 4, and 5 respectively, future research directions and current trends are discussed in Section 6.

因此，本文的目标是广泛的调查基于深度学习的医学图像配准，关注常见的挑战，讨论未来的研究方向。在调查基于深度学习的医学图像配准任务之前，有关深度学习的背景信息在第2部分进行讨论。本文总结的方法，可以分为下面的三个类别：深度迭代配准，有监督的变换估计，无监督的变换估计，这分别在第3、4、5部分进行讨论，第6部分讨论未来的研究方向和目前的趋势。

## 2 Deep Learning

Deep learning belongs to a larger class of machine learning that uses neural networks with a large number of layers to learn representations of data [38]. Based on the way that networks are trained, most deep learning approaches fall into one of two categories: supervised learning and unsupervised learning. Supervised learning involves the designation of a desired neural network output, while unsupervised learning involves drawing inferences from a set of data without the use of any manually defined labels [38, 107]. Both supervised and unsupervised learning, allow for the use of a variety of deep learning paradigms. In this section, several of those approaches will be explored, including: convolutional neural networks, recurrent neural networks, reinforcement learning, and generative adversarial networks. Note that there are many publicly available libraries that can be used to build the networks described in the section, for example TensorFlow [1], MXNet [17], Keras [22], Caffe [58], and PyTorch [95].

深度学习是机器学习的一个类别，使用很多层神经网络来学习数据的表示。按照网络训练的方式，多数深度学习方法可以分为以下两类：有监督的学习和无监督学习。有监督学习指定了神经网络的期望输出，而无监督学习推理所使用的数据，是没有任何手工定义的标签的。监督学习和无监督学习，都使用很多深度学习范式。本节中，研究了几种方法，包括：卷积神经网络，循环神经网络，强化学习和生成式对抗网络。现在有很多公开可用的库，可以用于构造本节所述的网络，比如，TensorFlow, MXNet, Keras, Caffe和PyTorch。

### 2.1 Convolutional neural networks

Convolutional neural networks (CNNs) and their variants (such as the fully convolutional neural network (FCN) are among the most commonly used deep neural networks for computer vision and image processing and analysis applications. CNNs are feed forward neural networks that are often used to analyze images and perform tasks such as classification and object detection/recognition [67, 115]. They utilize a variation of the multilayer perceptron (MLP) and are famously translation invariant due to their parameter weight sharing [38]. In each layer of these networks, a number of convolutional filters “slide” across the feature maps from the previous layer. The output is another set of feature maps that are constructed from the inner products of the kernel and the corresponding patches associated with previous feature maps. The feature maps that result from these convolutions are stacked and inputted into the next layer of the network. This allows for hierarchical feature extraction of the image. Further, these operations can be performed patch-wise, which is useful for a number of computer vision tasks [49, 141]. Because the construction of a feature map from either an input image/image pair or a feature map is linear, non-linear activation functions are used to introduce non-linearities and enhance the expressivity of feature maps [38].

CNNs及其变体（如FCN）是最常用的深度神经网络。CNNs是前向神经网络，通常用于分析图像，进行分类、目标检测/识别这样的任务。其利用了MLP的变体，具有平移不变形，因为有参数权重共享机制。在网络中的每一层，几个卷积滤波器滑过前一层的特征图。输出是另一组特征图，是卷积核和前一层特征图的内积。这些卷积得到的特征图堆叠起来，输出到网络的下一层。这样就可以进行层次化的图像特征提取。而且，这些操作可以逐块进行，这对于一些计算机视觉应用很有用。由于特征图的形成是线性运算，因此还需要非线性的激活函数来增进特征图的表示能力。

Convolutional filters and their activations are often combined with pooling layers, either average or max, in a typical CNN to reduce the dimensionality of feature maps [123, 69]. Batch normalization (BN) is commonly used after convolutional layers as well [54] because of its ability to reduce internal covariate shift. Furthermore, many modern neural networks make use of residual connections [122] and depthwise separable convolutions [21]. These networks can be trained in an end-to-end fashion using back propagation to iteratively update the parameters that constitute the network [38, 69].

卷积滤波器及其激活通常与池化层结合起来，要么是平均池化，要么是最大池化，在典型的CNN中，这用于降低特征图的维度。BN也通常用在卷积层的后面，因为可以降低内部的协方差变化。而且，很多现代神经网络使用残差连接和逐层可分离卷积。这些网络可以进行端到端的训练，使用反向传播迭代的更新参数，构成网络。

Additionally, randomly dropping connections in certain layers of a model during training, a strategy known as dropout, allows for the implicit use of an ensemble of models [118]. This is a popular regularization strategy that is frequently used to prevent overfitting.

另外，在模型训练过程中，在一些层中随机丢弃一些连接，这种策略称为dropout，这也是模型集成的隐式应用。这是一个流行的正则化策略，经常使用，以防止过拟合。

### 2.2 Recurrent Neural Networks

Although CNNs and their variants are typically used to analyze data that exists in the spatial domain, recurrent neural networks (RNNs) that are composed of several of the network components described in the above section can be used to analyze time series data. Each element (e.g image) in the time series data is mapped to a feature representation and the “current” representation is determined by a combination of the previous representations and the “current” input datum. RNNs can be “many-to-one” or “many-to-many” (i.e. the output of the RNN can be a single datum or time series data). Further, a gated variant of RNNs- Long Short-Term Memory Networks (LSTMs) [48]- can be used to model long term dependencies by helping to prevent gradient vanishing/explosion.

CNNs及其变体通常用于分析空域数据，RNNs也是由上节所述的部件组成的网络，可以用于分析时序数据。时间序列中的每个元素（如图像）都映射到一个特征表示，“目前的”表示由之前的表示和“目前的”输入数据的组合计算得到。RNNs可以是多对一的，或多对多的，（即RNN的输出可以是单个数据或时间序列数据）。而且，RNNs的一种门控变体-LSTM-可以用于对长期依赖关系进行建模，对防止梯度消失/爆炸有帮助。

### 2.3 Reinforcement learning

Another popular deep learning strategy is reinforcement learning. Problems that use reinforcement learning can essentially be cast as Markov decision processes [75, 81, 98] associated with a tuple of a state, action, transition probability, reward, and discount factor. When an agent is in a particular state, it uses a policy in order to determine an action to take among a set of state-dependent actions [60]. Upon performing the action that was selected by the policy, the agent transitions into the next state with a given probability and receives a reward. The goal of the agent is to maximize the total reward that it receives while performing a given task. Because the rewards that the agent will receive are subject to stochastic processes, the agent will seek to maximize the cumulative expected rewards, while using the discount factor in order to prioritize longer term rewards. The primary goal is to learn the optimal policy with respect to the expected future rewards [27, 91, 131]. Instead of doing this directly, most reinforcement learning paradigms learn the action-value function Q by using the Bellman equation [9]. The process through which Q functions are approximated is referred to as Q-learning. These approaches utilize value functions, action-value functions [131] that determine the advantageous nature of a given state and state-action pair respectively[126, 131]. Further, an advantage function determines the advantageous nature of a given state-action pair relative to the other pairs [126]. These approaches have been applied to various video/board games and have often been able to demonstrate superhuman performance [15, 40, 41, 112, 113, 124]. The performances of such methods are often used as benchmarks that indicate the current state of deep learning research.

另一种流行的深度学习策略是强化学习。使用强化学习的问题，都可以表述为Markov决策过程，与状态、行为、转移概率、回报和折扣因素的元组有关。当一个agent在一个特定的状态时，就使用一种策略来确定一个进行的行为，这个行为是一个行为集合中的元素，集合中的行为都是与状态有关的。进行了策略选择的这个动作，agent会以一定的概率迁移到下一个状态，并得到一个回报。Agent的目标是在进行给定的任务时，使得到的总计回报最大化。因为agent得到的回报是与随机过程相关的，agent就会试图最大化累积期待回报，同时使用折扣因子以使更长期的回报更优先。基本的目标是，学习到与期望未来回报相关的最优策略。多数强化学习范式没有直接这样做，而是使用Bellman等式学习行为值函数Q。Q函数近似的过程称为Q-学习。这个方法利用了值函数，行为值函数，这个函数在给定的状态-行为对中，确定了一个给定状态的优势。这些方法已经用于多种视频/桌面游戏，经常得到超过人类的表现。这些方法的性能经常用作基准测试，是目前深度学习研究的状态指示。

### 2.4 Generative Adversarial Networks

A generative adversarial network (GAN) [39] is composed of two competing neural networks: a generator and a discriminator. The generator maps data from one domain to another. In their original implementation, they mapped a random noise vector to an image domain associated with a particular dataset. The discriminator is tasked with discerning between real data that originated from said domain and data produced by the generator. The goal for training GANs is to converge to a differentiable Nash Equilibrium [99], at which point generated data and real data are indistinguishable [37].

GAN是由两个竞争的神经网络构成：生成器和判别器。生成器将一个领域的数据映射到另一个领域。在原始实现中，是将一个随机噪声向量映射到另一个图像领域，这个领域与一个特定数据集相关。判别器的任务是区分真实数据与生成器生成的数据。训练GANs的目标，是收敛到一个可微分的Nash均衡，在这个状态点上，真实数据与生成数据不可区分。

When GANs are applied to medical image registration, they are commonly used for regularization. The generator predicts a transformation and the discriminator takes the resulting resampled images as its input. The discriminator is trained to discern between aligned image pairs and resampled image pairs following the generator’s prediction. The generator is typically trained using a linear combination of an adversarial loss function term (based on the discriminator’s predictions) and a target loss function term (e.g euclidean distance from ground truth). For both the generator and discriminator, a binary cross entropy (BCE) loss function is commonly used.

GANs用在医学图像配准时，通常是用作正则化。生成器预测一个变换，判别器以得到的重新取样的图像作为输入。判别器的训练目标，是辨别对齐的图像对，和重新取样的图像对。生成器通常使用对抗损失项（基于判别器的预测）和目标损失函数项（如与真值的欧式距离）的线性组合来训练。对于生成器和判别器，通常都会使用二值交叉熵(Binary Cross Entropy, BCE)。

Further discussion of deep learning based medical image analysis and various deep learning research directions outlined above is outside of the scope of this article. However, comprehensive review articles that survey the application of deep learning to medical image analysis [70, 74], reinforcement learning [60], and the application of GANs to medical image analysis [61] are recommended to the interested readers.

基于深度学习的医学图像分析的进一步讨论，和各种深度学习研究方向，不在本文的范畴之内。但是，我们推荐了一些综述文章给读者，如基于深度学习的医学图像分析[70,74]，强化学习[60]，和GANs在医学图像分析中的应用[61]。

## 3 Deep Iterative Registration

Automatic intensity-based image registration requires both a metric that quantifies the similarity between a moving image and a fixed image and an optimization algorithm that updates the transformation parameters such that the similarity between the images is maximized. Prior to the deep learning renaissance, several manually crafted metrics were frequently used for such registration applications, including: sum of squared differences (SSD), cross-correlation (CC), mutual information (MI) [84, 129], normalized cross correlation (NCC), and normalized mutual information (NMI). Early applications of deep learning to medical image registration are direct extensions of this classical framework [114, 132, 133]. Several groups later used a reinforcement learning paradigm to iteratively estimate a transformation [64, 73, 83, 88] because this application is more consistent with how practitioners perform registration.

基于灰度信息的自动图像配准，需要两个要素，一个是度量移动图像和固定图像的相似度的标准，和一个优化算法，使得变换参数下的图像相似度度量最大化。在深度学习的复兴之前，几种手工定制的度量标准，经常用于这样的配准应用，包括：差值平房和(SSD)，互相关(CC)，互信息(MI)，归一化的互相关(NCC)，归一化的互信息(NMI)。深度学习在医学图像配准中的早期应用，是经典框架的直接拓展。后来几个小组使用强化学习范式，来对变换进行迭代估计，因为这个应用与实践中进行的配准非常相似。

A description of both types of methods is given in Table 1. We will survey earlier methods that used deep similarity based registration in Section 3.1 and then some more recently developed methods that use deep reinforcement learning based registration in Section 3.2.

两类方法的描述如表1所述。我们在3.1节中叙述了早期的使用深度相似度的配准，在3.2节中叙述了最近提出的使用深度强化学习的配准方法。

### 3.1 Deep Similarity based Registration

In this section, methods that use deep learning to learn a similarity metric are surveyed. This similarity metric is inserted into a classical intensity-based registration framework with a defined interpolation strategy, transformation model, and optimization algorithm. A visualization of this overall framework is given in Fig. 3. The solid lines represent data flows that are required during training and testing, while the dashed lines represent data flows that are required only during training. Note that this is the case for the remainder of the figures in this article as well.

本节中我们综述的方法，是使用深度学习以学习一个相似度度量的方法。这种相似性度量插入到了经典的基于灰度值的配准框架中，并且是一种特定定义的差值策略，变换模型和优化算法来插入的。图3给出了这种框架的总体概览。实线代表在训练和测试时都需要的数据流，而虚线则是只在训练时才需要的数据流。本文其他的图，也遵循这个规范。

#### 3.1.1 Overview of Works

Although manually crafted similarity metrics perform reasonably well in the uni-modal registration case, deep learning has been used to learn superior metrics. This section will first discuss approaches that use deep learning to augment the performance of unimodal intensity based registration pipelines before multimodal registration.

虽然手工定义的相似度度量，在单模态配准情况中，表现还不错，但深度学习已经用于学习更好的度量标准。本节首先讨论了使用深度学习来改进单模态的基于灰度的配准流程，然后讨论多模态的情况。

**3.1.1.1 Unimodal Registration.** Wu et al. [132,133] were the first to use deep learning to obtain an application specific similarity metric for registration. They extracted the features that are used for unimodal, deformable registration of 3D brain MR volumes using a convolutional stacked auto-encoder (CAE). They subsequently performed the registration using gradient descent to optimize the NCC of the two sets of features. This method outperformed diffeomorphic demons [127] and HAMMER [110] based registration techniques.

Wu等第一个使用深度学习得到配准的特定应用专用的相似度度量。他们使用卷积堆叠自动编码机提取的特征，用于3D脑部MR体的单模、形变配准。后来他们进行的配准，使用的是梯度下降，来优化两个集合特征的NCC。这种方法超过了基于diffeomorphic demons[127]和HAMMER[110]的配准方法。

Recently, Eppenhof et al. [32] estimated registration error for the deformable registration of 3D thoracic CT scans (inhale-exhale) in an end-to-end capacity. They used a 3D CNN to estimate the error map for inputted inhale-exhale pairs of thoracic CT scans. Like the above method, only learned features were used in this work.

最近，Eppenhof等估计了3D胸CT扫描（吸气-呼气）的形变配准的误差，这个估计是端到端的。他们使用了一个3D CNN，对输入的吸气-呼气对的胸腔CT扫描，估计了其误差图。与上面的方法相似的是，本文中只使用了学习到的特征。

Instead, Blendowski et al. [10] proposed the combined use of both CNN-based descriptors and manually crafted MRF-based self-similarity descriptors for lung CT registration. Although the manually crafted descriptors outperformed the CNN-based descriptors, optimal performance was achieved using both sets of descriptors. This indicates that, in the unimodal registration case, deep learning may not outperform manually crafted methods. However, it can be used to obtain complementary information.

Blendowski等[10]提出了将基于CNN的描述子和手工设计的基于MRF的自相似描述子结合起来，用于肺部CT配准。虽然手工设计的描述子超过了基于CNN的描述子，但使用两种描述子集合，得到了最佳性能。这说明了，在单模态配准的情况下，深度学习不一定能超过手工设计的方法。但是，可以用于获取互补的信息。

**3.1.1.2 Multimodal Registration.** The advantages of the application of deep learning to intensity based registration are more obvious in the multimodal case, where manually crafted similarity metrics have had very little success.

在多模态的情况下，基于深度学习的配准算法，比基于灰度值的算法显示出了更多优势，这里，手工设计的相似度度量很少有成功的例子。

Cheng et al. [18, 19] recently used a stacked denoising autoencoder to learn a similarity metric that assesses the quality of the rigid alignment of CT and MR images. They showed that their metric outperformed NMI and local cross correlation (LCC) for their application.

Cheng等最近使用了堆叠去噪自动编码机，学习了一个相似度度量，用于评估CT和MR图像的刚性配准的质量。他们表明，这种度量标准，在他们的应用中，超过了NMI和局部交叉相关(LCC)。

In an effort to explicitly estimate image similarity in the multimodal case, Simonovsky et al. [114] used a CNN to learn the dissimilarity between aligned 3D T1 and T2 weighted brain MR volumes. Given this similarity metric, gradient descent was used in order to iteratively update the parameters that define a deformation field. This method was able to outperform MI based registration and set the stage for deep intensity based multimodal registration.

Simonovsky等在多模态的情况下对图像相似度进行了估计，使用了一个CNN来学习对齐的3D T1和T2加权的脑MR体的不相似度。给定这种相似度度量，用梯度下降来迭代的更新参数，形成形变场的定义。这种方法可以超过基于MI的配准，形成基于灰度的深度配准的性能基准。

Additionally, Sedghi et al. [108] performed the rigid registration of 3D US/MR (modalities with an even greater appearance difference than MR/CT) abdominal scans by using a 5-layer neural network to learn a similarity metric that is then optimized by Powells method. This approach also outperformed MI based registration.

另外，Sedghi等进行了3D US/MR的腹部扫描刚性配准，使用了一个5层的神经网络，学习了一个相似性度量，然后用Powell方法进行优化。这种方法也超过了基于MI的配准。

Haskins et al. [42] learned a similarity metric for multimodal rigid registration of MR and transrectal US (TRUS) volumes by using a CNN to predict target registration error (TRE). Instead of using a traditional optimizer like the above methods, they used an evolutionary algorithm to explore the solution space prior to using a traditional optimization algorithm because of the learned metric’s lack of convexity. This registration framework outperformed MIND [44] and MI based registration.

Haskins等[42]也针对MR和经直肠US体的多模刚性配准学习了一个相似性度量，使用了一个CNN来预测目标配准误差。他们没有使用上述的传统优化器，而是使用了演化算法来探索解空间，然后使用传统的优化算法，因为学习的度量非凸。这种配准框架超过了MIND[44]和基于MI的配准。

In stark contrast to the above methods, Wright et al. [87] used LSTM spatial co-transformer networks to iteratively register MR and US volumes group-wise. The recurrent spatial co-transformation occurred in three steps: image warping, residual parameter prediction, parameter composition. They demonstrated that their method is more capable of quantifying image similarity than a previous multimodal image similarity quantification method that uses self-similarity context descriptors [45].

与上述方法形成对比的是，Wright等[87]使用了LSTM空域协同变换器网络来迭代的分组配准MS和US。循环空域协同变换在下面三个步骤中发生：图像形变，残差参数预测，参数形成。他们证明了，他们的方法比之前的多模态图像相似度量化方法（使用了一个自相似上下文描述子），更能量化一个图像相似度。

#### 3.1.2 Discussion and Assessment

Recent works have confirmed the ability of neural networks to assess image similarity in multimodal medical image registration. The results achieved by the approaches described in this section demonstrate that deep learning can be successfully applied to challenging registration tasks. However, the findings from [10] suggest that learned image similarity metrics may be best suited to complement existing similarity metrics in the unimodal case. Further, it is difficult to use these iterative techniques for real time registration.

最近的工作也确认了，神经网络在多模态图像配准中，评估图像相似度的能力。本节的方法所得的结果说明，深度学习可以成功的用于配准任务。但是，[10]的发现说明，在单模态的情况下，学习到的图像相似度度量，最好与现有的相似度度量互补使用。而且，很难使用这些迭代技术进行实时配准。

### 3.2 Reinforcement Learning based Registration

In this section, methods that use reinforcement learning for their registration applications are surveyed. Here, a trained agent is used to perform the registration as opposed to a pre-defined optimization algorithm. A visualization of this framework is given in Fig. 4. Reinforcement learning based registration typically involves a rigid transformation model. However, it is possible to use a deformable transformation model.

本节中，回顾了使用强化学习进行配准应用的方法。这里，使用训练好的agent来进行配准，而不是使用预定义好的优化算法。图4是这种框架的图示。基于强化学习的配准一般是刚性变换模型。但是，使用形变变换模型，也是有可能的。

Liao et al. [73] were the first to use reinforcment learning based registration to perform the rigid registration of cardiac and abdominal 3D CT images and cone-beam CT (CBCT) images. They used a greedy supervised approach for end-to-end training with an attention-driven hierarchical strategy. Their method outperformed MI based registration and semantic registration using probability maps.

Liao等[73]第一个使用强化学习进行配准，是对心脏和腹部的3D CT和CBCT图像进行刚性配准。他们使用了一种贪婪的监督方法进行端到端的训练，是一种注意力驱动的层次化策略。他们的方法超过了基于MI的配准和使用概率图的语义配准。

Shortly after, Kai et al. [83] used a reinforcement learning approach to perform the rigid registration of MR/CT chest volumes. This approach is derived from Q-learning and leverages contextual information to determine the depth of the projected images. The network used in this method is derived from the dueling network architecture [131]. Notably, this work also differentiates between terminal and non-terminal rewards. This method outperforms registration methods that are based on iterative closest points (ICP), landmarks, Hausdorff distance, Deep Q Networks, and the Dueling Network [131].

此后，Kai等[83]使用了一种强化学习方法，进行胸腔MR/CT的体刚性配准。这种方法是从Q学习中推导而来，利用了上下文信息来确定投影图像的深度。此方法中使用的网络是从dueling网络架构[131]中推导而来的。这种方法还区分了terminal回报和non-terminal回报。这种方法超过了基于ICP、特征点、Hausdorff距离、Deep Q网络和Dueling Network的方法。

Instead of training a single agent like the above methods, Miao et al. [88] used a multi-agent system in a reinforcement learning paradigm to rigidly register X-Ray and CT images of the spine. They used an auto-attention mechanism to observe multiple regions and demonstrate the efficacy of a multi-agent system. They were able to significantly outperform registration approaches that used a state-of-the-art similarity metric given by [24].

Miao[88]不是像上面的方法只训练了一个agent，而是使用了强化学习范式的多agent系统，以对脊柱的X射线和CT图像进行刚性配准。他们使用了一种auto-attention机制来观测多个区域，证明了多agent系统的效率。他们可以远超[24]的配准方法，而[24]使用目前最好的相似度度量。

As opposed to the above rigid registration based works, Krebs et al. [64] used a reinforcement learning based approach to perform the deformable registration of 2D and 3D prostate MR volumes. They used a low resolution deformation model for the registration and fuzzy action control to influence the stochastic action selection. The low resolution deformation model is necessary to restrict the dimensionality of the action space. This approach outperformed Elastix [62] and LCC-Demons [80] based registration techniques.

与上面的刚性配准应用不同，Krebs[64]使用强化学习方法进行形变配准，对象是前列腺2D和3D MR体。他们使用了一个低分辨率形变模型进行配准，使用了模糊行为控制来影响随机行为选择。低分辨率形变模型对于限制行为空间的维度是必须的。这种方法超过了基于Elastix[62]和LCC-Demons[80]的配准技术。

The use of reinforcement learning is intuitive for medical image registration applications. One of the principle challenges for reinforcement learning based registration is the ability to handle high resolution deformation fields. There are no such challenges for rigid registration. Because of the intuitive nature and recency of these methods, we expect that such approaches will receive more attention from the research community in the next few years.

强化学习的使用，对于医学图像配准应用是很自然的。基于强化学习的配准的一个主要挑战是，处理高分辨率形变场的能力。对于刚性配准，没有这样的挑战。希望这种方法在未来几年会得到更多的关注。

## 4 Supervised Transformation Estimation

Despite the early success of the previously described approaches, the transformation estimation in these methods is iterative, which can lead to slow registration. This is especially true in the deformable registration case where the solution space is high dimensional [70]. This motivated the development of networks that could estimate the transformation that corresponds to optimal similarity in one step. However, fully supervised transformation estimation (the exclusive use of ground truth data to define the loss function) has several challenges that are highlighted in this section.

上述方法早期比较成功，但这些方法中的变换估计是迭代的，这就导致配准速度较慢。这在形变配准中尤其如此，因为其解空间是高维的[70]。这就促使提出了能够一步估计最优相似性的变换的网络。但是，全监督的变换估计（使用真值数据来定义损失函数）有几个重大挑战。

A visualization of supervised transformation estimation is given in Fig. 5 and a description of notable works is given in Table 2. This section first discusses methods that use fully supervised approaches in Section 4.1 and then discusses methods that use dual/weakly supervised approaches in Section 4.2.

监督变换估计如图5所示，表2给出来相关的工作。本节首先在4.1节中讨论使用全监督方法，在4.2节中讨论对偶/弱监督方法的使用。

### 4.1 Fully Supervised Transformation Estimation

In this section, methods that used full supervision for single-step registration are surveyed. Using a neural network to perform registration as opposed to an iterative optimizer significantly speeds up the registration process.

本节中，回顾了使用全监督方法进行单步配准的方法。使用神经网络进行配准，与迭代优化器相比，显著加速了配准过程。

#### 4.1.1 Overview of works

Because the methods discussed in this section use a neural network to estimate transformation parameters directly, the use of a deformable transformation model does not introduce additional computational constraints. This is advantageous because deformable transformation models are generally superior to rigid transformation models [96]. This section will first discuss approaches that use a rigid transformation model and then discuss approaches that use a deformable transformation model.

因为本节讨论的方法使用神经网络直接估计变换参数，因此形变变换模型并不会带来额外的计算限制。这是有优势的，因为形变变换模型一般比刚性变换模型要好。本节首先讨论使用刚性变换模型，然后讨论使用形变变换模型。

**4.1.1.1 Rigid Registration.** Miao et al. [89, 90] were the first to use deep learning to predict rigid transformation parameters. They used a CNN to predict the transformation matrix associated with the rigid registration of 2D/3D X-ray attenuation maps and 2D X-ray images. Hierarchical regression is proposed in which the 6 transformation parameters are partitioned into 3 groups. Ground truth data was synthesized in this approach by transforming aligned data. This is the case for the next three approaches that are described as well. This approach outperformed MI, CC, and gradient correlation based iterative registration approaches.

Miao等首先使用深度学习来预测刚性变换参数。他们使用CNN来预测变换矩阵，这是2D/3D X射线衰减图和2D X射线图的刚性配准相关的。提出了层次回归，将6个变换参数分成了3组。真值数据是通过对对齐的数据进行变换，合成得到的。这和后面三种方法是一样的。这种方法超过了基于MI、CC和梯度相关的迭代配准方法。

Recently, Chee et al. [16] used a CNN to predict the transformation parameters used to rigidly register 3D brain MR volumes. In their framework, affine image registration network (AIRNet), the MSE between the predicted and ground truth affine transforms is used to train the network. They are able to outperform iterative MI based registration for both the unimodal and multimodal cases.

最近，Chee等[16]使用了一个CNN来预测变换参数，来对3D脑MR体进行刚性配准。在其框架中，仿射图像配准网络(AIRNet)，预测图像和真值图像的仿射变换的MSE用于训练网络。在单模态和多模态的情况下，他们都可以超过基于迭代MI的配准。

That same year, Salehi et al. [106] used a deep residual regression network, a correction network, and a bivariant geodesic distance based loss function to rigidly register T1 and T2 weighted 3D fetal brain MRs for atlas construction. The use of the residual network to initially register the image volumes prior to the forward pass through the correction network allowed for an enhancement of the capture range of the registration. This approach was evaluated for both slice-to-volume registration and volume-to-volume registration. They validated the efficacy of their geodesic loss term and outperformed NCC registration.

同一年，Salehi等[106]使用一个深度残差回归网络，一个修正网络，和一个基于二元测地距离的损失函数，来对T1加权和T2加权的3D胎儿脑部MR进行刚性配准，以进行图集构建。使用残差网络来初始化配准图像，然后用修正网络进行前向过程，这使得配准的捕获范围可以扩大。这种方法用于slice-to-volume配准的评估，和volume-to-volume配准的评估。他们验证了测地距离损失的作用，这超过了NCC配准。

Additionally, Zheng et al. [143] proposed the integration of a pairwise domain adaptation module (PDA) into a pre-trained CNN that performs the rigid registration of pre-operative 3D X-Ray images and intraoperative 2D X-ray images using a limited amount of training data. Domain adaptation was used to address the discrepancy between synthetic data that was used to train the deep model and real data.

另外，Zheng等[143]提出了将一个成对domain adaptation模块与预训练的CNN整合到一起，进行手术前的3D X射线图像刚性配准，和手术中的2D X射线图像刚性配准，使用的训练数据数量有限。Domain adaptation用于处理合成数据和真实数据之间的差异，合成数据是用于训练深度模型的。

Sloan et al. [116] used a CNN is used to regress the rigid transformation parameters for the registration of T1 and T2 weighted brain MRs. Both unimodal and multimodal registration were investigated in this work. The parameters that constitute the convolutional layers that were used to extract low-level features in each image were only shared in the unimodal case. In the multimodal case, these parameters were learned separately. This approach also outperformed MI based image registration.

Sloan等[116]使用CNN来回归刚性变换参数，以配准T1加权和T2加权的脑MR。这片文章中进行了单模态和多模态的配准。构成卷积层（用于提取底层特征）的参数只在单模态的情况下共享。在多模态的情况下，这些参数是分开进行学习的。这种方法也超过了基于MI的图像配准。

**4.1.1.2 Deformable Registration.** Unike the previous section, methods that use both real and synthesized ground truth labels will be discussed. Methods that use clinical/publicly available ground truth labels for training are discussed first. This ordering is reflective of the fact that simulating realistic deformable transformations is more difficult than simulating realistic rigid transformations.

与前一节不同，这里会讨论使用真实的真值数据和合成的真值数据的方法。首先讨论使用临床上/公开可用的真值标签进行训练的方法。这个顺序也反应了，模拟真实的形变变换，比模拟真实的刚性变换，要困难的多。

First, Yang et al. [137] predicted the deformation field with an FCN that is used to register 2D/3D intersubject brain MR volumes in a single step. A U-net like architecture [103] was used in this approach. Further, they used large diffeomorphic metric mapping to provide a basis, used the initial momentum values of the pixels of the image volumes as the network input, and evolved these values to obtain the predicted deformation field. This method outperformed semi-coupled dictionary learning based registration [11].

首先，Yang等[137]使用FCN预测了形变场，用于一步配准2D/3D脑MR体。这种方法中使用了一种类U-Net的架构。而且，他们将diffeomorphic度量标准映射作为基础，使用图像volume的像素的初始动量值作为网络输入，将这些值通过迭代演化，得到预测的形变场。这种方法超过了基于半耦合字典学习的配准。

The following year, Rohe et al. [102] also used a U-net [103] inspired network to estimate the deformation field used to register 3D cardiac MR volumes. Mesh segmentations are used to compute the reference transformation for a given image pair and SSD between the prediction and ground truth is used as the loss function. This method outperformed LCC Demons based registration [80].

下一年，Rohe等[102]也使用了类U-Net的网络来估计形变场，以配准3D心脏MR volume。网格分割用于计算给定图像对的参考变换，预测值和真值之间的SSD用于损失函数。这种方法超出了基于LCC Demons的配准。

That same year, Cao et al. [13] used a CNN to map input image patches of a pair of 3D brain MR volumes to their respective displacement vector. The totality of these displacement vectors for a given image constitutes the deformation field that is used to perform the registration. Additionally, they used the similarity between inputted image patches to guide the learning process. Further, they used equalized active-points guided sampling strategy that makes it so that patches with higher gradient magnitudes and displacement values are more likely to be sampled for training. This method outperforms SyN [5] and Demons [127] based registration methods.

同一年，Cao等[13]使用CNN将输入的成对3D脑MR volume的图像块映射到其偏移向量。给定图像的所有偏移向量，构成了变形场，用于进行配准。另外，他们使用输入图像块之间的相似度来引导学习过程。而且，they used equalized active-points guided sampling strategy that makes it so that patches with higher gradient magnitudes and displacement values are more likely to be sampled for training. 这种方法超过来基于SyN和Demons的配准方法。

Recently, Jun et al. [82] used a CNN to perform the deformable registration of abdominal MR images to compensate for the deformation that is caused by respiration. This approach achieved registration results that are superior to those obtained using non-motion corrected registrations and local affine registration. Recently, unlike many of the other approaches discussed in this paper, Yang et al. [136] quantified the uncertainty associated with the deformable registration of 3D T1 and T2 weighted brain MRs using a low-rank Hessian approximation of the variational gaussian distribution of the transformation parameters. This method was evaulated on both real and synthetic data.

最近，Jun等[82]使用了CNN进行腹部MR图像的形变配准，这些形变是由呼吸运动导致的。这种方法得到的配准结果，好过于那些没有进行运动修正的配准结果，和局部放射配准结果。最近，与本文讨论的其他方法都不同，Yang等[136]量化了与3D T1和T2加权的脑MR的形变配准相关的不确定性，使用的是变换参数的变分高斯分布的低秩的Hessian近似。这种方法在真实数据和合成数据上都进行了评估。

Just as deep learning practitioners use random transformations to enhance the diversity of their dataset, Sokooti et al. [117] used random DVFs to augment their dataset. They used a multi-scale CNN to predict a deformation field. This deformation is used to perform intra-subject registration of 3D chest CT images. This method used late fusion as opposed to early fusion, in which the patches are concatenated and used as the input to the network. The performance of their method is competitive with B-Spline based registration [117].

深度学习者使用随机变换来增强数据集的多样性，Sokooti等[117]使用随机DVFs来对数据集进行扩充。他们使用一个多尺度CNN来预测形变场。这种形变是用于进行胸腔3D CT图像的目标内配准。这种方法使用了后期融合，而不是早期融合，图像块拼接后用于网络的输入。他们的方法的性能与基于B-spline的配准类似。

Such approaches have notable, but also limited ability to enhance the size and diversity of datasets. These limitations motivated the development of more sophisticated ground truth generation. The rest of the approaches described in this section use simulated ground truth data for their applications.

这种方法增强数据集的规模和多样性的能力有限。其局限促使人们提出了更复杂的真值数据生成方法。本节剩下所述的方法，使用的是模拟的真值数据。

For example, Eppenhof et al. [31] used a 3D CNN to perform the deformable registration of inhale-exhale 3D lung CT image volumes. A series of multi-scale, random transformations of aligned image pairs eliminate the need for manually annotated ground truth data while also maintaining realistic image appearance. Further, as is the case with other methods that generate ground truth data, the CNN can be trained using relatively few medical images in a supervised capacity.

例如，Eppenhof等[31]使用一个3D CNN来进行形变配准，对象是呼气-吸气的3D肺部CT图像volume。对对齐的图像对进行了一系列多尺度、随机的变换，这样就不需要手工标注真值数据，同时还保持了真实的图像外观。而且，其他方法也采用同样的真值数据生成方法，CNN在监督情况下进行训练，可以使用相对较少的医学图像。

Unlike the above works, Uzunova et al. [125] generated ground truth data using statistical appearance models (SAMs). They used a CNN to estimate the deformation field for the registration of 2D brain MRs and 2D cardiac MRs, and adapt FlowNet [29] for their application. They demonstrated that training FlowNet using SAM generated ground truth data resulted in superior performance to CNNs trained using either randomly generated ground truth data or ground truth data obtained using the registration method described in [30].

与上述工作不同，Uzunova等[125]使用统计外观模型(SAM)来生成真值数据。他们使用CNN来估计形变场，以进行2D脑MR和2D心脏MR的配准，并修改了FlowNet[29]用在他们的应用中。他们证明了，使用SAM生成的真值数据训练FlowNet，会优于其他方法生成的真值数据的训练结果。

Unlike the other methods in this section that use random transformations or manually crafted methods to generate ground truth data, Ito et al. [56] used a CNN to learn plausible deformations for ground truth data generation. They evaluated their approach on the 3D brain MR volumes in the ADNI dataset and outperformed the MI based approach proposed in [53].

本节中的其他方法使用的是随机变换，或手工设计的方法来生成真值数据，Ito等[56]与这些方法不同，使用CNN学习了一种可行的变形以进行真值数据生成。他们使用其方法对ADNI数据集的3D脑MR进行评估，超过了基于MI的方法[53]。

#### 4.1.2 Discussion and Assessment

Supervised transformation estimation has allowed for real time, robust registration across applications. However, such works are not without their limitations. Firstly, the quality of the registrations using this framework is dependent on the quality of the ground truth registrations. The quality of these labels is, of course, dependent upon the expertise of the practitioner. Furthermore, these labels are fairly difficult to obtain because there are relatively few individuals with the expertise necessary to perform such registrations. Transformations of training data and the generation of synthetic ground truth data can address such limitations. However, it is important to ensure that simulated data is sufficiently similar to clinical data. These challenges motivated the development of partially supervised/unsupervised approaches, which will be discussed next.

有监督的变换估计可以在不同的应用中进行实时的，稳健的配准。但是，这些工作是有其局限性的。首先，使用这种框架进行配准的质量是依赖于真值配准的质量的。这些标签的质量，是依赖于实践者的专业能力的。而且，这些标签是较难得到的，因为很少有人既有这种专业能力还有这种需求去进行这种配准。训练数据的变换，和合成真值数据的生成，可以解决这种局限。但是，要确保合成的数据，与临床的数据要足够相似。这些挑战促使了部分监督/无监督的方法的研究，下面会进行讨论。

### 4.2 Dual/Weakly Supervised Transformation Estimation

Dual supervision refers to the use of both ground truth data and some metric that quantifies image similarity to train a model. On the other hand, weak supervision refers to using the overlap of segmentations of corresponding anatomical structures to design the loss function. This section will discuss the contributions of such works in Section 4.2.1 and then discuss the overall state of this research direction in Section 4.2.2.

双监督指的是使用真值数据和一些量化图像相似度的度量标准来训练模型。另一方面，弱监督指的是使用对应解剖结构的分割的重叠来指定损失函数。在4.2.1节讨论的是这样工作的贡献，在4.2.2节中讨论这个研究方向的总体状态。

#### 4.2.1 Overview of works

First, this section will discuss methods that use dual supervised and then will discuss methods that use weak supervision. Recently, Fan et al. [34] used hierarchical, dual-supervised learning to predicted the deformation field for 3D brain MR registration. They amend the traditional U-Net architecture [103] by using “gap-filling” (i.e., inserting convolutional layers after the U-type ends or the architecture) and coarse-to-fine guidance. This approach leveraged both the similarity between the predicted and ground truth transformations, and the similarity between the warped and fixed images to train the network. The architecture detailed in this method outperformed the traditional U-Net architecture and the dual supervision strategy is verified by ablating the image similarity loss function term. A visualization of dual supervised transformation estimation is given in Fig. 6.

第一，本节会讨论使用双监督的方法，然后讨论使用弱监督的方法。最近，Fan等[34]使用层次化的双监督学习来预测变形场，进行3D脑MR的配准。他们修补了传统的U-Net架构，进行了gap-filling（即在U型末端或架构之后增加卷积层），并进行了粗糙到精细的引导。这个方法利用了预测和真值变换之间的相似度，和变形图像和固定图像的相似度来训练网络。这个架构性能超过了传统的U-Net架构，双监督策略的性能通过分离试验得到了验证。图6给出了双监督变换估计的框图。

On the other hand, Yan et al. [134] used a GAN [39] framework to perform the rigid registration of 3D MR and TRUS volumes. In this work, the generator was trained to estimate a rigid transformation. While, the discriminator was trained to discern between images that were aligned using the ground truth transformations and images that were aligned using the predicted transformations. Both Euclidean distance to ground truth and an adversarial loss term are used to construct the loss function in this method, which outperformed both MIND based registration and MI based registration. Note that the adversarial supervision strategy that was used in this approach is similar to the ones that are used in a number of unsupervised works that will be described in the next section. A visualization of adversarial transformation estimation is given in Fig. 7.

另一方面，Yan等[134]使用了GAN框架来进行3D MR和TRUS的刚性配准。在这个方法中，生成器的训练用于估计刚性变换。同时，判别器的训练用于区分两类图像，即用真值变形对齐的图像，和与预测的变形对齐的图像。这个方法中的损失函数是到真值的欧式距离和对抗损失项，算法性能超过了基于MIND的配准和基于MI的配准。注意，这个方法中使用的对抗监督策略，与下一小节中描述的几个无监督的方法使用的类似。对抗变换估计如图7所示。

Unlike the above methods that used dual supervision, Hu et al. [51, 52] recently used label similarity to train their network to perform MR-TRUS registration. In their initial work, they used two neural networks: local-net and global-net to estimate the global affine transformation with 12 degrees of freedom and the local dense deformation field respectively [51]. The local-net uses the concatenation of the transformation of the moving image given by the global-net and the fixed image as its input. However, in their later work [52], they combine these networks in an end-to-end framework. This method outperformed NMI based and NCC based registration. A visualization of weakly supervised transformation estimation is given in Fig. 8. In another work, Hu et al. [50] simultaneously maximized label similarity and minimized an adversarial loss term to predict the deformation for MR-TRUS registration. This regularization term forces the predicted transformation to result in the generation of a realistic image. Using the adversarial loss as a regularization term is likely to successfully force the transformation to be realistic given proper hyper parameter selection. The performance of this registration framework was inferior to the performance of their previous registration framework described above. However, they showed that adversarial regularization is superior to standard bending energy based regularization. Similar to the above method, Hering et al. [46] built upon the progress made with respect to both dual and weak supervision by introducing a label and similarity metric based loss function for cardiac motion tracking via the deformable registration of 2D cine-MR images. Both segmentation overlap and edge based normalized gradient fields distance were used to construct the loss function in this approach. Their method outperformed a multilevel registration approach similar to the one proposed in [104].

上面的方法是双监督的方法，Hu等[51,52]与之不同，使用了标签相似性来训练其网络，进行MR-TRUS配准。在其开始的工作中，他们使用了两个神经网络：local-net和global-net来估计全局仿射变换，有12个自由度，也估计局部密集变形场。Local-net使用移动图像的变形的拼接（由global-net给出），和固定图像作为输入。但是，在其后续的工作中[52]，他们将网络综合到一起，形成了一个端到端的框架。这个方法性能超过了基于NMI的和NCC的配准算法。图8给出了弱监督变换估计的图示。

在另一个工作中，Hu等[50]同时最大化标签相似度和最小化一个对抗损失，来为MS-TRUS配准预测形变。正则化项迫使预测的变换生成真实的图像。使用对抗损失作为正则化项，只要合适选择超参数，很可能会迫使变换是真实的。这个配准框架的性能，比上面所述的他们前一个框架的性能要差一些。但是，他们表明了，对抗正则化是优于标准的基于bending energy的正则化的。与上述方法类似，Hering等在这基础上，提出了一个基于标签和相似性度量的损失函数，通过2D cine-MR图像形变配准进行心脏运动追踪。这个方法中，使用了基于分割重叠和边缘的归一化梯度场距离来构建损失函数。他们的方法超过了一种多级配准方法，与[104]提出的类似。

#### 4.2.2 Discussion and Assessment

Direct transformation estimation marked a major breakthrough for deep learning based image registration. With full supervision, promising results have been obtained. However, at the same time, those techniques require a large amount of detailed annotated images for training. Partially/weakly supervised transformation estimation methods alleviated the limitations associated with the trustworthiness and expense of ground truth labels. However, they still require manually annotated data (e.g ground truth and/or segmentations). On the other hand, weak supervision allows for similarity quantification in the multimodal case. Further, partial supervision allows for the aggregation of methods that can be used to assess the quality of a predicted registration. As a result, there is growing interest in these research areas.

直接变换估计是基于深度学习的图像配准的主要突破。在完全监督下，可以得到很有希望的配准结果。但是，同时，这项技术需要大量标注的图像进行训练。部分监督/弱监督的变换估计方法，缓解了这个局限。但是，他们仍然需要手工标注的数据（如，真值和分割）。另一方面，弱监督在多模态情况下允许相似度量化。而且，部分监督可以进行各种方法的积聚，可以用于预测的配准的质量评估。在这个研究领域的研究兴趣是逐渐增加的。

## 5 Unsupervised Transformation Estimation

Despite the success of the methods described in the previous sections, the difficult nature of the acquisition of reliable ground truth remains a significant hindrance. This has motivated a number of different groups to explore unsupervised approaches. One key innovation that has been useful to these works is the spatial transformer network (STN) [57]. Several methods use an STN to perform the deformations associated with their registration applications. This section discusses unsupervised methods that utilize image similarity metrics (Section 5.1) and feature representations of image data (Section 5.2) to train their networks. A description of notable works is given in Table 3.

前一节所述的方法是很成功的，但可靠的真值的获取是很困难的，这仍然是一个明显的阻碍。这促使了一些小组探索无监督的方法。一个关键的创新是spatial transformer network(STN)[57]。几种方法都使用了STN来进行与配准应用相关的变形。本节讨论了无监督方法，利用图像相似度度量（5.1节）和图像数据的特征表示（5.2节）来训练网络。表3给出了相关的文章。

### 5.1 Similarity Metric based Unsupervised Transformation Estimation

#### 5.1.1 Standard Methods

This section begins by discussing approaches that use a common similarity metric with common regularization strategies to define their loss functions. Later in the section, approaches that use more complex similarity metric based strategies are discussed. A visualization of standard similarity metric based transformation estimation is given in Fig. 9.

本节讨论的方法是使用通用相似度度量和通用正则化策略来定义其损失函数。后面，讨论了使用基于更复杂的相似性度量的策略的方法。基于标准相似度度量的变换估计如图9所示。

Inspired to overcome the difficulty associated with obtaining ground truth data, Li et al. [71, 72] trained an FCN to perform deformable intersubject registration of 3D brain MR volumes using ”self-supervision.” NCC between the warped and fixed images and several common regularization terms (e.g smoothing constraints) constitute the loss function in this method. Although many manually defined similarity metrics fail in the multimodal case (with the occasional exception of MI), they are often suitable for the unimodal case. The method detailed in this work outperforms ANTs based registration and the deep learning methods proposed by Sokooti et al. [117] (discussed previously) and Yoo et al. [140] (discussed in the next section).

为克服获得真值数据的困难，Li等[71,72]训练了一个FCN来进行3D脑部MR体的形变目标内配准，使用的是自监督。变形图像和固定图像间的NCC，和几种常见的正则化项（如，平滑限制）构成了这种方法的损失函数。虽然很多手工定义的相似度度量在多模态的情况下会失败（使用MI则有时候会成功），他们更适合于单模态的情况。这篇文章的方法超过了基于ANT的配准，超过了Sokooti等[117]和Yoo等[140]提出的深度学习方法。

Further, de Vos et al. [26] used NCC to train an FCN to perform the deformable registration of 4D cardiac cine MR volumes. A DVF is used in this method to deform the moving volume. Their method outperforms Elastix based registration [62].

另外，de Vos等[26]使用NCC来训练FCN来进行4D心脏cine MR体的形变配准。这种方法使用了DVF来对移动volume进行变形。这种方法超过了基于Elastix的配准。

In another work, de Vos et al. [25] use a multistage, multiscale approach to perform unimodal registration on several datasets. NCC and a bending-energy regularization term are used to train the networks that predict an affine transformation and subsequent coarse-to-fine deformations using a B-Spline transformation model. In addition to validating their multi-stage approach, they show that their method outperforms simple elastix based registration with and without bending energy.

在另一项工作中，de Vos等[25]使用了一个多阶段，多尺度方法在几个数据集上进行单模态配准。使用了NCC和bending-energy正则化项来训练网络，预测一个仿射变换和后续的粗糙到精细的变形，使用的是B-spline转换模型。除了验证了他们的多阶段的方法，他们的方法还超过了基于简单的elastix的配准方法。

The unsupervised deformable registration framework used by Ghosal et al. [36] minimizes the upper bound of the SSD (UB SSD) between the warped and fixed 3D brain MR images. The design of their network was inspired by the SKIP architecture [79]. This method outperforms log-demons based registration.

Ghosal等[36]的无监督形变配准框架对变形的和固定的3D脑MR图像之间的SSD上限进行最小化。其网络设计受到SKIP架构启发[79]。这种方法超过了基于log-demons的配准。

Shu et al. [111] used a coarse-to-fine, unsupervised deformable registration approach to register images of neurons that are acquired using a scanning electron microscope (SEM). The mean squared error (MSE) between the warped and fixed volumes is used as the loss function here. Their approach is competitive with and faster than the sift flow framework [76].

Shu等[111]使用了一种由粗到细的、无监督的形变配准方法，对使用SEM得到的神经元图像进行配准。变形的和固定的MSE用作损失函数。他们的方法比sift flow框架[76]性能类似，速度要更快。

Sheikhjafari et al. [109] used learned latent representations to perform the deformable registration of 2D cardiac cine MR volumes. Deformation fields are thus obtained by embedding. This latent representation is used as the input to a network that is composed of 8 fully connected layers to obtain the transformation. The sum of absolute errors (SAE) is used as the loss function. This method outperforms a moving mesh correspondence based method described in [97].

Sheikhjafari等[109]使用了学习到的隐藏表示来对2D心脏cine MR体进行形变配准。变形场是通过嵌入得到的。这种隐藏表示用作网络输入，网络由8层全连接层组成，用于获得变换信息。损失函数用SAE。这种方法超过了[97]中基于moving mesh correspondance的方法。

Stergios et al. [119] used a CNN to both linearly and locally register inhale-exhale pairs of lung MR volumes. Therefore, both the affine transformation and the deformation are jointly estimated. The loss function is composed of an MSE term and regularization terms. Their method outperforms several state-of-the-art methods that do not utilized ground truth data, including Demons [80], SyN [5], and a deep learning based method that uses an MSE loss term. Further, the inclusion of the regularization terms is validated by an ablation study.

Stergios等[119]使用CNN来对肺部MR体的呼气-吸气对进行线性配准和局部配准。因此，对仿射变换和形变是联合估计的。损失函数由MSE项和正则化项构成。他们的方法几种不使用真值数据的目前最好方法，包括Demons[80]，SyN[5]，和一种基于深度学习的方法（使用MSE损失项）。另外，正则化项的有效应通过一个分离试验进行了验证。

The successes of deep similarity metric based unsupervised registration motivated Neylon et al. [94] to use a neural network to learn the relationship between image similarity metric values and TRE when registering CT image volumes. This is done in order to robustly assess registration performance. The network was able to achieve subvoxel accuracy in 95% of cases. Similarly inspired, Balakrishnan et al. [7, 8] proposed a general framework for unsupervised image registration, which can be either unimodal or multimodal theoretically. The neural networks are trained using a selected, manually-defined image similarity metric (e.g. NCC, NMI, etc.).

基于相似性度量的深度无监督配准方法的成功，促使Neylon等[94]在配准CT图像volume时，使用神经网络来学习图像相似度度量值和TRE之间的关系。这是为了稳健的评估配准性能。网络可以在95%的情况下得到subvoxel的精度。类似的，Balakrishnan等[7,8]提出了无监督图像配准的通用框架，理论上可以是单模的，也可以是多模的。网络的训练使用的是手动定义的选择的图像相似度度量（如，NCC，NMI等）。

In a follow-up paper, Dalca et al. [23] casted deformation prediction as variational inference. Diffeomorphic integration is combined with a transformer layer to obtain a velocity field. Squaring and rescaling layers are used to integrate the velocity field to obtain the predicted deformation. MSE is used as the similarity metric that, along with a regularization term, define the loss function. Their method outperforms ANTs based registration [6] and the deep learning based method described in [7].

在后续的文章中，Dalca等[23]将形变预测表述为变分推理问题。Diffeomorphic integration与transformer layer结合到一起，形成速度场。平方层和改变尺度的层用于将速度场整合到一起，得到预测的形变场。相似性度量用的是MSE，和一个正则化项，形成了损失函数。他们的方法性能超过了基于ANTs的配准[6]和基于深度学习的方法[7]。

Shortly after, Kuang et al. [68] used a CNN and STN inspired framework to perform the deformable registration of T1-weighted brain MR volumes. The loss function is composed of a NCC term and a regularization term. This method uses Inception modules, a low capacity model, and residual connections instead of skip connections. They compare their method with VoxelMorph (the method proposed by Balakrishnan et al., described above) [8] and uTIlzReg GeoShoot [128] using the LBPA40 and Mindboggle 101 datasets and demonstrate superior performance with respect to both.

很快，Kuang等[68]使用了CNN和STN启发得到的框架，进行T1加权的脑MR的形变配准。损失函数是由NCC项和正则化项构成。这种方法使用Inception模块，所以是一种低容量模型，使用了残差连接，而没有使用skip连接。他们将方法与VoxelMorph等方法进行了比较，表明其方法非常优秀。

Building upon the progress made by the previously described metric-based approaches, Ferrante et al. [35] used a transfer learning based approach to perform unimodal registration of both X-ray and cardiac cine images. In this work, the network is trained on data from a source domain using NCC as the primary loss function term and tested in a target domain. They used a U-net like architecture [103] and an STN [57] to perform the feature extraction and transformation estimation respectively. They demonstrated that transfer learning using either domain as the source or the target domain produces effective results. This method outperformed the Elastix registration technique [62].

在上述工作基础之上，Ferrante等[35]使用了基于迁移学习的方法进行X射线图像和心脏cine图像的单模态配准。这片文章中，网络的训练是在source domain进行，使用NCC作为基本的损失函数，并在target domain进行测试。他们使用了一个U-Net类的架构和STN来分别进行特征提取和变换估计。他们证明了，无论使用哪个domain作为source或target，采用迁移学习都会得到有效的结果。这种方法超过了Elastix配准技术。

Although applying similarity metric based approaches to the multimodal case is difficult, Sun et al. [120] proposed an unsupervised method for 3D MR/US brain registration that uses a 3D CNN that consists of a feature extractor and a deformation field generator. This network is trained using a similarity metric that incorporates both pixel intensity and gradient information. Further, both image intensity and gradient information are used as inputs into the CNN.

虽然将基于相似性度量的方法用在多模的情况很困难，Sun等[120]提出了一种无监督方法进行3D MR/US脑配准，使用的是一个3D CNN，包含一个feature extractor和一个deformation field generator。这个网络的训练使用的是相似性度量，也包含了像素灰度和梯度信息。而且，图像灰度和梯度信息都用作CNN的输入。

#### 5.1.2 Extensions

Cao et al. [12] also applied similarity metric based training to the multimodal case. Specifically, they used intra-modality image similarity to supervise the multimodal deformable registration of 3D pelvic CT/MR volumes. The NCC between the moving image that is warped using the ground truth transformation and the moving image that is warped using the predicted transformation is used as the loss function. This work utilizes ”dual” supervision (i.e. the intra-modality supervision previously described is used for both the CT and the MR images). This is not to be confused with the dual supervision strategies described earlier.

Cao等[12]将基于相似度度量的训练也应用到了多模态的情况。具体的，他们对3D盆部CT/MR体，用模态内的图像相似度来监督多模态的形变配准。真值变换变形的图像，和预测变换变形的图像，其NCC用作损失函数。这片文章使用了“双“监督（即，CT和MR的模态内监督）。不要与之前所述的双监督策略混淆。

Inspired by the limiting nature of the asymmetric transformations that typical unsupervised methods estimate, Zhang et al. [142] used their network Inverse-Consistent Deep Network (ICNet)-to learn the symmetric diffeomorphic transformations for each of the brain MR volumes that are aligned into the same space. Different from other works that use standard regularization strategies, this work introduces an inverse-consistent regularization term and an anti-folding regularization term to ensure that a highly weighted smoothness constraint does not result in folding. Finally, the MSD between the two images allows this network to be trained in an unsupervised manner. This method outperformed SyN based registration [5], Demons based registration [80], and several deep learning based approaches.

典型的无监督方法估计的都是非对称的变换，这个局限启发了Zhang等[142]，使用其ICNet对每个脑MR volume（对齐到同样的空间）学习对称的diffeomorphic的变换。其他的文章一般使用标准正则化策略，与之不同，这片文章提出了一种inverse-consistent正则化项，和一个anti-folding正则化项，来确保平滑限制的高权重，不会导致folding。最后，两个图像的MSD使网络可以以无监督的方式进行训练。这种方法超过了基于SyN的配准，基于Demons的配准，和其他基于深度学习的方法。

The next three approaches described in this section used a GAN for their applications. Unlike the GAN-based approaches described previously, these methods use neither ground truth data nor manually crafted segmentations. Mahapatra et al. [85] used a GAN to implicitly learn the density function that represents the range of plausible deformations of cardiac cine images and multimodal retinal images (retinal colour fundus images and fluorescein angiography (FA) images). In addition to NMI, structual similarity index measure (SSIM), and a feature perceptual loss term (determined by the SSD between VGG outputs), the loss function is comprised of conditional and cyclic constraints, which are based on recent advances involving the implementation of adversarial frameworks. Their approach outperforms Elastix based registration and the method proposed by de Vos et al. [26].

本节后面讨论的三种方法，都使用了GAN。与之前讨论的基于GAN的方法不同，这些方法没有使用真值数据，也没有使用手工分割结果。Mahapatra等[85]使用一个GAN隐式的学习了一个密度函数，代表了两类模态的成像。

Further, Fan et al. [33] used a GAN to perform unsupervised deformable image registration of 3D brain MR volumes. Unlike most other unsupervised works that use a manually crafted similarity metric to determine the loss function and unlike the previous approach that used a GAN to ensure that the predicted deformation is realistic, this approach uses a discriminator to assess the quality of the alignment. This approach outperforms Diffeomorphic Demons and SyN registration on every dataset except for MGH10. Further, the use of the discriminator for supervision of the registration network is superior to the use of ground truth data, SSD, and CC on all datasets.

Different from the hitherto previously described works (not just the GAN based ones), Mahapatra et al. [86] proposed simultaneous segmentation and registration of chest X-rays using a GAN framework. The network takes 3 inputs: reference image, floating image, and the segmentation mask of the reference image and outputs the segmentation mask of the transformed image, and the deformation field. Three discriminators are used to assess the quality of the generated outputs (deformation field, warped image, and segmentation) using cycle consistency and a dice metric. The generator is additionally trained using NMI, SSIM, and a feature perceptual loss term.

Finally, instead of predicting a deformation field given a fixed parameterization as the other methods in this section do, Jiang et al. [59] used a CNN to learn an optimal parameterization of an image deformation using a multi-grid B-Spline method and L1-norm regularization. They use this approach to parameterize the deformable registration of 4D CT thoracic image volumes. Here, SSD is used as the similarity metric and L-BFGS-B is used as the optimizer. The convergence rate using the parameterized deformation model obtained using the proposed method is faster than the one obtained using a traditional L1-norm regularized multi-grid parameterization.

#### 5.1.3 Discussion and Assessment

Image similarity based unsupervised image registration has received a lot of attention from the research community recently because it bypasses the need for expert labels of any kind. This means that the performance of the model will not depend on the expertise of the practitioner. Further, extensions of the original similarity metric based method that introduce more sophisticated similarity metrics (e.g the discriminator of a GAN) and/or regularization strategies have yielded promising results. However, it is still difficult to quantify image similarity for multimodal registration applications. As a result, the scope of unsupervised, image similarity based works is largely confined to the unimodal case. Given that multimodal registration is often needed in many clinical applications, we expect to see more papers in the near future that will tackle this challenging problem.

基于图像相似度的无监督图像配准最近得到非常多的关注，因为这不需要专家标注的标签。这意味着模型的性能，不会依赖于人的专业能力。而且，基于原始相似度度量的方法的扩展，带来了更复杂的相似度度量标准（如，GAN的判别器），和正则化策略，得到了很有希望的结果。但是，对于多模态配准，仍然很难量化图像相似度。结果，无监督的基于图像相似度的工作大概只能局限于单模态的情况。鉴于多模态的配准在临床应用中很多，我们期待未来看到更多文章来解决这个问题。

### 5.2 Feature based Unsupervised Transformation Estimation

In this section, methods that use learned feature representations to train neural networks are surveyed. Like the methods surveyed in the previous section, the methods surveyed in this section do not require ground truth data. In this section, approaches that create unimodal registration pipelines are presented first. Then, an approach that tackles multimodal image registration is discussed. A visualization of featured based transformation estimation is given in Fig. 10.

本节中，回顾了使用学习到的特征表示来训练神经网络的方法。与之前的小节回顾的方法类似，本节中的方法也不需要真值数据。本节中，先给出单模态配准的流程。然后，再讨论多模态配准的方法。图10给出了基于特征的变换估计的方法的图示。

#### 5.2.1 Unimodal Registration

Yoo et al. [140] used an STN to register serial-section electron microscopy images (ssEMs). An autoencoder is trained to reconstruct fixed images and the L2 distance between reconstructed fixed images and corresponding warped moving images is used along with several regularization terms to construct the loss function. This approach outperforms the bUnwarpJ registration technique [4] and the Elastic registration technique [105].

Yoo等[140]使用STN来配准ssEMs。

In the same year, Liu et al. [78] proposed a tensor based MIND method using a principle component analysis based network (PCANet) [14] for both unimodal and multimodal registration. Both inhale-exhale pairs of thoracic CT volumes and multimodal pairs of brain MR images are used for experimental validation of this approach. MI and residual complexity (RC) based [92], and the original MIND-based [44] registration techniques were outperformed by the proposed method.

Krebs et al. [65, 66] performed the registration of 2D brain and cardiac MRs and bypassed the need for spatial regularization using a stochastic latent space learning approach. A conditional variational autoencoder [28] is used to ensure that the parameter space follows a prescribed probability distribution. The negative log liklihood of the fixed image given the latent representation and the warped volume and KL divergence of the latent distribution from a prior distribution are used to define the loss function. This method outperforms the Demons technique [80] and the deep learning method described in [7].

#### 5.2.2 Multimodal Registration

Unlike all of the other methods described in this section, Kori et al. perform feature extraction and affine transformation parameter regression for the multimodal registration of 2-D T1 and T2 weighted brain MRs in an unsupervised capacity using pre-trained networks [63]. The images are binarized and then the Dice score between the moving and the fixed images is used as the cost function. As the appearance difference between these two modalities is not significant, the use of these pre-trained models can be reasonably effective.

#### 5.2.3 Discussion and Assessment

Performing multimodal image registration in an unsupervised capacity is significantly more difficult than performing unimodal image registration because of the difficulty associated with using manually crafted similarity metrics to quantify the similarity between the two images, and generally using the unsupervised techniques described above to establish/detect voxel-to-voxel correspondence. The use of unsupervised learning to learn feature representations to determine an optimal transformation has generated significant interest from the research community recently. Along with the previously discussed unsupervised image registration method, we expect feature based unsupervised registration to continue to generate significant interest from the research community. Further, extension to the multimodal case (especially for applications that use image with significant appearance differences) is likely to be a prominent research focus in the next few years.

## 6 Research Trends and Future Directions

In this section, we summarize the current research trends and future directions of deep learning in medical image registration. As we can see from Fig. 2, some research trends have emerged. First, deep learning based medical image registration seems to be following the observed trend for the general application of deep learning to medical image analysis. Second, unsupervised transformation estimation methods have been garnering more attention recently from the research community. Further, deep learning based methods consistently outperform traditional optimization based techniques [93]. Based on the observed research trends, we speculate that the following research directions will receive more attention in the research community.

本节中，我们总结了深度学习在医学图像配准中目前的研究趋势和未来的方向。从图2中可以看出，一些研究方向已经出现。首先，基于深度学习的医学图像配准似乎遵循深度学习在医学图像分析的通用应用中的趋势。第二，无监督变换估计方法正得到越来越多的关注。而且，基于深度学习的方法一直在超越基于传统优化的技术。基于观察到的研究趋势，我们推测下面的研究方向会得到更多的关注。

### 6.1 Deep Adversarial Image Registration

We further speculate that GANs will be used more frequently in deep learning based image registration in the next few years. As described above, GANs can serve several different purposes in deep learning based medical image registration: ensuring that predicted transformations are realistic, using a discriminator as a learned similarity metric, and using a GAN to perform image translation to transform a multimodal registration problem into a unimodal registration problem.

我们进一步推测，在未来几年，GANs会更多的应用于基于深度学习的图像配准。如上所述，GANs在基于深度学习的医学图像配准中有几个作用：确保预测的变换是真实的，使用判别器作为学习到的相似性度量，使用GAN进行图像转换，将多模配准问题转化成单模态配准问题。

Unconstrained deformation field prediction can result in warped moving images with unrealistic organ appearances. A common approach to add the L2 norm of the predicted deformation field to the loss function. However, Hu et al. [50] explored the use of a GAN like framework to produce realistic deformations. Constraining the deformation prediction using a discriminator results in superior performance relative to the use of L2 norm regularization.

不受限的形变场预测，得到的变形moving图像，其器官外表会不太真实。常用的方法是将预测的形变场的L2范数加入到损失函数中。但是，Hu等[50]探索使用一种类似GAN的框架，以得到真实的变形。使用判别器的结果限制变形预测，会得到很好的性能，比使用L2范数正则化要好。

Further, GANs have been used in several works to obtain a learned similarity metric. Several recent works [33, 134] use a discriminator to discern between aligned and misaligned image pairs. This is particularly useful in the multimodal registration case where manually crafted similarity metrics famously have little success. Because this allows the generator to be trained without ground truth transformations, further research into using discriminators as similarity metric will likely allow for unsupervised multimodal registration.

而且，GANs已经用在了几篇文章中，以得到学习好的相似度度量。几篇最近的文章使用判别器来区别对齐的图像和未对齐的图像。这在多模态配准的情况下尤其有用，因为手工设计的相似度度量在这个领域几乎没有成功。这使得生成器不用真值变换就可以得到训练，使用判别器作为相似性度量，将来可能使无监督的多模配准成为可能。

Lastly, GANs can be used to map medical images in a source domain (e.g MR) to a target domain (e.g CT) [20, 55, 77, 139], regardless of whether or not paired training data is available [145]. This would be advantageous because many unimodal unsupervised registration methods use similarity metrics, which often fail in the multimodal case, to define their loss functions. If image translation could be performed as a pre-processing step, then commonly used similarity metrics could be used to define the loss function.

最后，GANs可以用于将source domain的医学图像（如MR）映射到target domain（如CT）中，有没有成对的训练数据，都是可以的。这是很有优势的，因为很多单模态无监督配准方法使用的是相似度度量，这在多模态情况就会失败，不能定义出好的损失函数。如果图像转化可以作为一个预处理步骤，那么常用的相似度度量比标准就可以用于定义损失函数了。

### 6.2 Reinforcement Learning based Registration

We also project that reinforcement learning will also be more commonly used for medical image registration in the next few years because it is very intuitive and can mimic the manner in which physicians perform registration. It should be noted that there are some unique challenges associated with deep learning based medical image registration: including the dimensionality of the action space in the deformable registration case. However, we believe that such limitations are surmountable because there is already one proposed method that uses reinforcement learning based registration with a deformable transformation model [64].

我们还预测，强化学习也会更多的用于医学图像配准中，因为很直观，可以模仿物理师进行配准的方式。应当指出，基于深度学习的医学图像配准，有一些独特的挑战：包括在形变配准的情况中，action空间的维度问题。但是，我们相信，这些局限是可以克服的，因为已经有人提出使用基于强化学习的配准，其中是形变变换的模型。

### 6.3 Raw Imaging Domain Registration

This article has focused on surveying methods performing registration using reconstructed images. However, we speculate that it is possible to incorporate reconstruction into an end-to-end deep learning based registration pipeline. In 2016, Wang [130] postulated that deep neural networks could be used to perform image reconstruction. Further, several works [138, 144, 101] recently demonstrated the ability of deep learning to map data points in the raw data domain to the reconstructed image domain. Therefore, it is reasonable to expect that registration pipelines that take raw data as input and output registered, reconstructed images can be developed within the next few years.

本文关注的是使用重建的图像进行配准。但是，我们推测，将重建过程纳入到端到端的深度学习配准系统中，是有可能的。在2016年，Wang[130]推测到，CNN可以用于进行图像重建。而且，最近的几篇文章证实了，可以使用深度学习将原始数据domain映射到重建数据domain。因此，可以期待，以原始数据作为输入，并输出配准的重建图像，在未来几年会出现。

## 7 Conclusion

In this article, the recent works that use deep learning to perform medical image registration have been examined. As each application has its own unique challenges, the creation of the deep learning based frameworks must be carefully designed. Most deep learning based medical image registration applications share similar challenges (e.g. lack of a large database, the difficulty associated with robustly labeling medical images). Recent successes have demonstrated the impact of the application of deep learning to medical image registration. This trend can be observed across medical imaging applications. Many future exciting works are sure to build on the recent progress that has been outlined in this paper.

在本文中，对近年来使用深度学习进行医学图像配准的方法进行了综述。由于每个应用都有其独特的挑战，基于深度学习的框架需要仔细设计。多数基于深度学习的医学图像配准应用，都有相同的挑战（如，缺少大型数据集，标注医学图像的困难等）。近期的成功表明，深度学习对医学图像配准的影响。这种趋势在不同的医学图像应用中都可以观察到。未来肯定会在本文列出的这些进展的基础上，提出更成功的模型。