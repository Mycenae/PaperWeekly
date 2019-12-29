# Deep Learning vs. Traditional Computer Vision

Niall O’ Mahony et al. IMaR Technology Gateway, Institute of Technology Tralee, Ireland

**Abstract**. Deep Learning has pushed the limits of what was possible in the domain of Digital Image Processing. However, that is not to say that the traditional computer vision techniques which had been undergoing progressive development in years prior to the rise of DL have become obsolete. This paper will analyse the benefits and drawbacks of each approach. The aim of this paper is to promote a discussion on whether knowledge of classical computer vision techniques should be maintained. The paper will also explore how the two sides of computer vision can be combined. Several recent hybrid methodologies are reviewed which have demonstrated the ability to improve computer vision performance and to tackle problems not suited to Deep Learning. For example, combining traditional computer vision techniques with Deep Learning has been popular in emerging domains such as Panoramic Vision and 3D vision for which Deep Learning models have not yet been fully optimised.

**摘要** 深度学习拓展了数字图像处理领域的边界。但是，这并不是说，过去很多年逐渐发展起来的传统计算机视觉技术已经被废弃。本文会分析每种方法的优势和缺点。本文的目的是推动大家讨论，经典计算机视觉技术是否应当继续维护发展。本文同时还会探索计算机视觉的这两种技术怎样结合到一起。我们对最近的几种混合方法进行了回顾，这些方法都可以改进计算机视觉的性能，解决深度学习无法解决的问题。比如，将传统计算机视觉技术与深度学习结合起来，这在新出现的领域中很流行，如全景视觉和3D视觉，在这些领域中，深度学习模型还没有得到充分的优化。

**Keywords**: Computer Vision, Deep Learning, Hybrid techniques.

## 1 Introduction

Deep Learning (DL) is used in the domain of digital image processing to solve difficult problems (e.g. image colourization, classification, segmentation and detection). DL methods such as Convolutional Neural Networks (CNNs) mostly improve prediction performance using big data and plentiful computing resources and have pushed the boundaries of what was possible. Problems which were assumed to be unsolvable are now being solved with super-human accuracy. Image classification is a prime example of this. Since being reignited by Krizhevsky, Sutskever and Hinton in 2012 [1], DL has dominated the domain ever since due to a substantially better performance compared to traditional methods.

深度学习在数字图像处理的领域有很多应用，以解决困难的问题（如图像上色，分类，分割和检测）。DL方法，如CNNs，主要是利用了大数据和大量计算能力以改进预测性能，拓展了研究前沿。以前认为解决不了的问题，现在问题解决得到的精确度超过人类的表现。图像分类是一个主要例子。DL的研究在2012年被Krizhevsky等重新点燃后，就一直主导着计算机视觉的研究，因为与传统方法相比，会有很高的性能提升。

Is DL making traditional Computer Vision (CV) techniques obsolete? Has DL superseded traditional computer vision? Is there still a need to study traditional CV techniques when DL seems to be so effective? These are all questions which have been brought up in the community in recent years [2], which this paper intends to address.

DL是否使得传统计算机视觉技术废弃了吗？DL是否超过了传统计算机视觉呢？DL看起来效果非常好，是否还有研究传统CV技术的需要呢？在过去几年中，研究团体不断提出这些问题，本文打算来处理这些问题。

Additionally, DL is not going to solve all CV problems. There are some problems where traditional techniques with global features are a better solution. The advent of DL may open many doors to do something with traditional techniques to overcome the many challenges DL brings (e.g. computing power, time, accuracy, characteristics and quantity of inputs, and among others).

另外，DL不可能解决所有CV问题。在一些问题中，采用全局特征的传统技术可以有更好的解决。DL的出现，使得用传统技术也可以做一些事情，以克服DL带来的很多挑战（如，计算能力，时间，准确度，输入的特征和数量，等等）。

This paper will provide a comparison of deep learning to the more traditional handcrafted feature definition approaches which dominated CV prior to it. There has been so much progress in Deep Learning in recent years that it is impossible for this paper to capture the many facets and sub-domains of Deep Learning which are tackling the most pertinent problems in CV today. This paper will review traditional algorithmic approaches in CV, and more particularly, the applications in which they have been used as an adequate substitute for DL, to complement DL and to tackle problems DL cannot.

本文比较了深度学习与更传统的手工设计特征的方法，这在DL出现之前是主流研究。最近几年深度学习的发展太多了，本文不可能覆盖深度学习的所有方面和子领域。本文会回顾CV中的传统算法方法，特别的，可以作为DL的替代品的那些应用，以作为DL的补充，解决DL解决不了的问题。

The paper will then move on to review some of the recent activities in combining DL with CV, with a focus on the state-of-the-art techniques for emerging technology such as 3D perception, namely object registration, object detection and semantic segmentation of 3D point clouds. Finally, developments and possible directions of getting the performance of 3D DL to the same heights as 2D DL are discussed along with an outlook on the impact the increased use of 3D will have on CV in general.

本文然后回顾最近的几篇文章，将DL和CV结合到一起，尤其是那些最新的技术，解决的是3D感知的问题，即目标配准，目标检测和3D点云的语义分割。最后，讨论了将3D DL发展到2D DL的水平，可能的方向，还展望了3D的使用对CV的一般性影响。

## 2 A Comparison of Deep Learning and Traditional Computer Vision

### 2.1 What is Deep Learning

To gain a fundamental understanding of DL we need to consider the difference between descriptive analysis and predictive analysis.

为对DL有基本的理解，我们需要知道descriptive analysis和predictive analysis的区别。

Descriptive analysis involves defining a comprehensible mathematical model which describes the phenomenon that we wish to observe. This entails collecting data about a process, forming hypotheses on patterns in the data and validating these hypotheses through comparing the outcome of descriptive models we form with the real outcome [3]. Producing such models is precarious however because there is always a risk of unmodelled variables that scientists and engineers neglect to include due to ignorance or failure to understand some complex, hidden or non-intuitive phenomena [4].

Descriptive analysis，是定义了一个可理解的数学模型，描述了我们要观察的现象。这需要收集这个过程的数据，对这些数据的模式形成假设，并通过比较用descriptive model模型得到的输出与真实输出，验证这些假设。生成这些模型是不可靠的，因为总有漏掉的未建模的变量的风险，或者是因为忽略了，或者是因为没有理解一些复杂的、隐藏的或非直观的现象。

Predictive analysis involves the discovery of rules that underlie a phenomenon and form a predictive model which minimise the error between the actual and the predicted outcome considering all possible interfering factors [3]. Machine learning rejects the traditional programming paradigm where problem analysis is replaced by a training framework where the system is fed a large number of training patterns (sets of inputs for which the desired outputs are known) which it learns and uses to compute new patterns [5].

Predictive analysis是发现现象底层的规则，形成一个predictive model，最小化实际的输出与预测的输出之间的误差，考虑到所有可能的干扰因素。机器学习与传统编程范式不同，用一个训练框架代替了问题分析，用很多训练数据送入系统（输入对应的期望输出是已知的），系统进行学习并用于计算新的模式。

DL is a subset of machine learning. DL is based largely on Artificial Neural Networks (ANNs), a computing paradigm inspired by the functioning of the human brain. Like the human brain, it is composed of many computing cells or ‘neurons’ that each perform a simple operation and interact with each other to make a decision [6]. Deep Learning is all about learning or ‘credit assignment’ across many layers of a neural network accurately, efficiently and without supervision and is of recent interest due to enabling advancements in processing hardware [7]. Self-organisation and the exploitation of interactions between small units have proven to perform better than central control, particularly for complex non-linear process models in that better fault tolerance and adaptability to new data is achievable [7].

DL是深度学习的一个子集。DL主要是基于人工神经网络，这是受到人类大脑的功能启发得到的计算范式。如同人类大脑一样，这是由很多计算单元或神经元组成，每个神经元执行一个简单的运算，并互相交互以得到一个决策。深度学习的过程就是学习的过程，是给神经网络很多层的单元准确的指定权重的过程，这个过程是高效的，没有监督的，由于近年来处理硬件的发展，得到了广泛的关注。小单元的自组织和相互作用的利用，比集中控制的系统性能要好，尤其是在复杂的、非线性的过程模型中，对错误的容忍性更好，对新数据的适应性也更好。

### 2.2 Advantages of Deep Learning

Rapid progressions in DL and improvements in device capabilities including computing power, memory capacity, power consumption, image sensor resolution, and optics have improved the performance and cost-effectiveness of further quickened the spread of vision-based applications. Compared to traditional CV techniques, DL enables CV engineers to achieve greater accuracy in tasks such as image classification, semantic segmentation, object detection and Simultaneous Localization and Mapping (SLAM). Since neural networks used in DL are trained rather than programmed, applications using this approach often require less expert analysis and fine-tuning and exploit the tremendous amount of video data available in today’s systems. DL also provides superior flexibility because CNN models and frameworks can be re-trained using a custom dataset for any use case, contrary to CV algorithms, which tend to be more domain-specific.

DL的迅速发展，和设备性能的改进，包括计算能力，存储能力，能耗，图像传感器的分辨率，光学性能，改进了基于视觉的算法的性能，和进一步加速算法应用的效能比。与传统CV算法相比，DL使得CV工程师可以得到更高的准确率，如图像分类，语义分割，目标检测和SLAM。由于DL中使用的神经网络是训练的，而不是编程得到的，使用这种方法的应用通常需要的专家分析和精调很少，而更多的利用今天系统中可用的视频数据。DL也有非常好的灵活性，因为CNN模型和框架可以使用自定义数据集进行重新训练，这与传统CV算法不一样，一般都是更加适应特定领域的。

Taking the problem of object detection on a mobile robot as an example, we can compare the two types of algorithms for computer vision:

以移动机器人上的目标检测问题为例，我们可以比较这两类计算机视觉算法：

The traditional approach is to use well-established CV techniques such as feature descriptors (SIFT, SURF, BRIEF, etc.) for object detection. Before the emergence of DL, a step called feature extraction was carried out for tasks such as image classification. Features are small “interesting”, descriptive or informative patches in images. Several CV algorithms, such as edge detection, corner detection or threshold segmentation may be involved in this step. As many features as practicable are extracted from images and these features form a definition (known as a bag-of-words) of each object class. At the deployment stage, these definitions are searched for in other images. If a significant number of features from one bag-of-words are in another image, the image is classified as containing that specific object (i.e. chair, horse, etc.).

传统方法使用的CV技术是成熟的，如用特征描述子(SIFT, SURF, BRIEF, etc.)进行目标检测。在DL出现之前，对于图像分类这种应用，会有一个步骤称为特征提取。特征是图像的小块，descriptive or informative。几种CV算法，如边缘检测，角点检测或阈值分割，也可以在这个步骤中出现。从图像中提取出尽可能多的特征，然后对每个目标类别，这些特征形成了一个定义（被称为bag-of-words）。在部署阶段，在其他图像中搜寻这种定义。如果在另一幅图像中找到了一个bag-of-words中的很多特征，那么图像就分类为包含这种特定的目标（即，椅子，马匹，等）。

The difficulty with this traditional approach is that it is necessary to choose which features are important in each given image. As the number of classes to classify increases, feature extraction becomes more and more cumbersome. It is up to the CV engineer’s judgment and a long trial and error process to decide which features best describe different classes of objects. Moreover, each feature definition requires dealing with a plethora of parameters, all of which must be fine-tuned by the CV engineer.

这种传统方法的困难在于，在给定的图像中需要选择哪种特征是必须的。随着要分类的类别数量增加，特征提取变得越来越麻烦。需要CV工程师进行判断，这是一个冗长的试错过程，来决定哪些特征最能描述不同类别的目标。而且，每个特征定义都涉及到很多参数，所有这些参数都需要由CV工程师精调。

DL introduced the concept of end-to-end learning where the machine is just given a dataset of images which have been annotated with what classes of object are present in each image [7]. Thereby a DL model is ‘trained’ on the given data, where neural networks discover the underlying patterns in classes of images and automatically works out the most descriptive and salient features with respect to each specific class of object for each object. It has been well-established that DNNs perform far better than traditional algorithms, albeit with trade-offs with respect to computing requirements and training time. With all the state-of-the-art approaches in CV employing this methodology, the workflow of the CV engineer has changed dramatically where the knowledge and expertise in extracting hand-crafted features has been replaced by knowledge and expertise in iterating through deep learning architectures as depicted in Fig. 1.

DL引入来端到端学习的概念，只需要给机器一个标注过的图像数据集，说明每幅图像中存在哪些类别的目标。然后在给定的数据上训练DL模型，神经网络会发现各类别图像的潜在模式，对每个特定类别的目标的，自动得到最具有描述力的最显著的特征。已经确定的是，DNNs比传统算法的效果要好的多，尽管在计算量需求和训练时间上还有折中。CV中目前最好的算法都采用了这种方法，所以CV工程师的工作流程有了很大变化，以前需要很多知识进行提取手工设计的特征，现在已经变成了深度学习架构的迭代的知识，如图1所示。

The development of CNNs has had a tremendous influence in the field of CV in recent years and is responsible for a big jump in the ability to recognize objects [9]. This burst in progress has been enabled by an increase in computing power, as well as an increase in the amount of data available for training neural networks. The recent explosion in and wide-spread adoption of various deep-neural network architectures for CV is apparent in the fact that the seminal paper ImageNet Classification with Deep Convolutional Neural Networks has been cited over 3000 times [2].

CNNs的发展对CV领域有巨大的影响，因此目标识别的能力有了很大的进步。这种进步的原因，是计算能力的进步，还有用于训练神经网络的可用数据的增长。最近采用各种DNN架构解决CV问题的极大发展，其根源是采用DNN进行ImageNet分类的最初文章引用数量超过了3000次。

CNNs make use of kernels (also known as filters), to detect features (e.g. edges) throughout an image. A kernel is just a matrix of values, called weights, which are trained to detect specific features. As their name indicates, the main idea behind the CNNs is to spatially convolve the kernel on a given input image check if the feature it is meant to detect is present. To provide a value representing how confident it is that a specific feature is present, a convolution operation is carried out by computing the dot product of the kernel and the input area where kernel is overlapped (the area of the original image the kernel is looking at is known as the receptive field [10]).

CNNs利用滤波核，进行特征检测（如边缘）。滤波核就是一个矩阵，称为权重，训练用于检测特定的特征。CNNs的主要思想是，用滤波核对给定输入图像在空间上进行卷积，看要检测的特征是否存在。我们给定一个值，代表指定特征是否存在的置信度，卷积操作的进行，是通过计算卷积核和输入的点积。

To facilitate the learning of kernel weights, the convolution layer’s output is summed with a bias term and then fed to a non-linear activation function. Activation Functions are usually non-linear functions like Sigmoid, TanH and ReLU (Rectified Linear Unit). Depending on the nature of data and classification tasks, these activation functions are selected accordingly [11]. For example, ReLUs are known to have more biological representation (neurons in the brain either fire or they don’t). As a result, it yields favourable results for image recognition tasks as it is less susceptible to the vanishing gradient problem and it produces sparser, more efficient representations [7].

为学习卷积核的权重，卷积层的输出与偏置项相加，送入非线性激活函数。激活函数通常是非线性函数，如sigmoid，tanh和ReLU。随着数据和分类任务的不同，相应的选择不同的激活函数。比如，ReLU更能代表生物特性（大脑中的神经元要么是开启的，要么不是）。结果是，图像识别任务可以得到很好的结果，因为消失梯度的问题更不容易出现，而且可以得到更稀疏的更高效的表示。

To speed up the training process and reduce the amount of memory consumed by the network, the convolutional layer is often followed by a pooling layer to remove redundancy present in the input feature. For example, max pooling moves a window over the input and simply outputs the maximum value in that window effectively reducing to the important pixels in an image [7]. As shown in Fig. 2, deep CNNs may have several pairs of convolutional and pooling layers. Finally, a Fully Connected layer flattens the previous layer volume into a feature vector and then an output layer which computes the scores (confidence or probabilities) for the output classes/features through a dense network. This output is then passed to a regression function such as Softmax [12], for example, which maps everything to a vector whose elements sum up to one [7].

为加速训练过程，减少网络消耗的内存，卷积层后通常都是池化层，以去除输入特征的冗余性。比如，最大池化用滑窗滑过输入，输出窗口中的最大值，选出图像中的重要像素。如图2所示，DCNNs有若干对卷积层和池化层的组合。最后，全连接层将前一层的输出拉平，成为一个特征向量，然后输出层会对输出的类别（特征）通过一个密集网络来计算分数（置信度或概率）。这个输出然后送入一个回归函数中，如softmax，将其映射到一个向量，向量元素之和相加为1。

But DL is still only a tool of CV. For example, the most common neural network used in CV is the CNN. But what is a convolution? It’s in fact a widely used image processing technique (e.g. see Sobel edge detection). The advantages of DL are clear, and it would be beyond the scope of this paper to review the state-of-the-art. DL is certainly not the panacea for all problems either, as we will see in following sections of this paper, there are problems and applications where the more conventional CV algorithms are more suitable.

但DL只是CV的一个工具。比如，在CV中最常用的神经网络是CNN。但什么是卷积呢？这实际上是广泛应用于图像处理技术的（如，Sobel边缘检测）。DL的优势的清楚的，本文也不能回顾所有目前最好的方法。DL当然不是所有问题的灵丹妙药，我们在文章后面部分会看到，在一些问题和应用中，传统CV算法会更加好用一些。

### 2.3 Advantages of Traditional Computer Vision Techniques

This section will detail how the traditional feature-based approaches such as those listed below have been shown to be useful in improving performance in CV tasks: 本节会详细介绍传统的基于特征的方法，如下列这些，都可以改进CV算法的性能：

- Scale Invariant Feature Transform (SIFT) [14]
- Speeded Up Robust Features (SURF) [15]
- Features from Accelerated Segment Test (FAST) [16]
- Hough transforms [17]
- Geometric hashing [18]

Feature descriptors such as SIFT and SURF are generally combined with traditional machine learning classification algorithms such as Support Vector Machines and K-Nearest Neighbours to solve the aforementioned CV problems. 特征描述子，如SIFT和SURF通常与传统的机器学习分类算法结合使用，如SVM和K近邻，以解决前面所述的CV问题。

DL is sometimes overkill as often traditional CV techniques can solve a problem much more efficiently and in fewer lines of code than DL. Algorithms like SIFT and even simple colour thresholding and pixel counting algorithms are not class-specific, that is, they are very general and perform the same for any image. In contrast, features learned from a deep neural net are specific to your training dataset which, if not well constructed, probably won’t perform well for images different from the training set. Therefore, SIFT and other algorithms are often used for applications such as image stitching/3D mesh reconstruction which don’t require specific class knowledge. These tasks have been shown to be achievable by training large datasets, however this requires a huge research effort and it is not practical to go through this effort for a closed application. One needs to practice common sense when it comes to choosing which route to take for a given CV application. For example, to classify two classes of product on an assembly line conveyor belt, one with red paint and one with blue paint. A deep neural net will work given that enough data can be collected to train from. However, the same can be achieved by using simple colour thresholding. Some problems can be tackled with simpler and faster techniques.

DL有时候会使用过度，因为通常传统CV算法可以更高效的解决问题，而且代码数量也比DL要少。像SIFT或简单的颜色阈值和像素计数这种算法，并不是对特定类别适用的算法，而是非常通用的，对任何图像都非常适用的算法。比较之下，DNN学习到的特征是只适用于训练集的，如果数据集的构建有什么问题的话，那么对于与训练集不同的图像来说，可能表现就不会那么好。因此，SIFT和其他算法的通常应用包括，图像缝合，3D网格重建，这些应用不需要特定的类别知识。这些任务通过训练大型数据集也是可以实现的，但是这需要很多的研究工作，对于一个已经成熟的应用来说，再去进行这样的努力也是不可行的。当为给定的应用选择技术路线时，需要一些常识。比如，在流水线传送带上对两类产品进行分类，一种有红色涂料，一种有蓝色涂料。一个深度神经网络在收集来足够多的数据进行训练的时候，是有用的。但是，用简单的色彩阈值方法，也是可以的。一些问题可以用更简单更快速的方法进行解决。

What if a DNN performs poorly outside of the training data? If the training dataset is limited, then the machine may overfit to the training data and not be able to generalize for the task at hand. It would be too difficult to manually tweak the parameters of the model because a DNN has millions of parameters inside of it each with complex inter- relationships. In this way, DL models have been criticised to be a black box in this way [5]. Traditional CV has full transparency and the one can judge whether your solution will work outside of a training environment. The CV engineer can have insights into a problem that they can transfer to their algorithm and if anything fails, the parameters can be tweaked to perform well for a wider range of images.

如果在训练数据之外，DNN模型效果较差，怎么办呢？如果训练数据集有限，那么机器可能在训练数据上过拟合，无法在现有的任务上很好的泛化。手工调整模型参数是非常困难的，因为一个DNN会有数百万个参数，其之间有非常复杂的关系。人们曾经批评过DL的这种黑箱特性。传统CV则是完全透明的，人们可以判断一个算法在训练环境之外是否能够很好的工作。CV工程师对问题有非常深入的理解，可以将其算法迁移到新环境中，如果有问题的话，可以调整参数，以对更多的图像都可以得到更好的表现。

Today, the traditional techniques are used when the problem can be simplified so that they can be deployed on low cost microcontrollers or to limit the problem for deep learning techniques by highlighting certain features in data, augmenting data [19] or aiding in dataset annotation [20]. We will discuss later in this paper how many image transformation techniques can be used to improve your neural net training. Finally, there are many more challenging problems in CV such as: Robotics [21], augmented reality [22], automatic panorama stitching [23], virtual reality [24], 3D modelling [24], motion estimation [24], video stabilization [21], motion capture [24], video processing [21] and scene understanding [25] which cannot simply be easily implemented in a differentiable manner with deep learning but benefit from solutions using "traditional" techniques.

今天，当问题可以简化时，就使用传统方法，将其部署到低功耗的微控制器上，或采用深度学习方法，强调数据中的一些特征，扩充数据，并在数据标注上进行帮助。我们后面会讨论，有多少种图像变换技术可以用于改进神经网络训练。最后，在CV中，有很多更有挑战性的事情，比如：机器人，增强现实，自动全景拼接，虚拟现实，3D建模，运动估计，视频稳定化，运动捕获，视频处理和场景理解，这些应用用深度学习进行实现是非常困难的，但采用传统方法时则更容易有解决方案。

## 3 Challenges for Traditional Computer Vision

### 3.1 Mixing Hand-Crafted Approaches with DL for Better Performance

There are clear trade-offs between traditional CV and deep learning-based approaches. Classic CV algorithms are well-established, transparent, and optimized for performance and power efficiency, while DL offers greater accuracy and versatility at the cost of large amounts of computing resources.

在传统CV算法和基于深度学习的方法中有很明显的折中关系。经典CV算法已经很成熟，透明，性能和功耗上都得到了优化，而DL则可以得到更高的准确率，更广的适用性，但代价是需要很大的计算量。

Hybrid approaches combine traditional CV and deep learning and offer the advantages traits of both methodologies. They are especially practical in high performance systems which need to be implemented quickly. For example, in a security camera, a CV algorithm can efficiently detect faces or other features [26] or moving objects [27] in the scene. These detections can then be passed to a DNN for identity verification or object classification. The DNN need only be applied on a small patch of the image saving significant computing resources and training effort compared to what would be required to process the entire frame.

将传统CV算法和深度学习方法结合起来，同时具有两类方法的优点。这样在高性能系统中尤其实用，可以很快的得到实现。比如，在安保摄像头中，CV算法可以高效的检测到人脸或其他特征或运动物体。这些检测然后可以送入DNN，进行身份验证，或目标分类。这样DNN就只需要应用到图像的小块中，就可以节约大量计算资源和训练的耗时，如果处理整帧图像，则需要很长时间，运算量也很大。

The fusion of Machine Learning metrics and Deep Network have become very popular, due to the simple fact that it can generate better models. Hybrid vision processing implementations can introduce performance advantage and ‘can deliver a 130X-1,000X reduction in multiply-accumulate operations and about 10X improvement in frame rates compared to a pure DL solution. Furthermore, the hybrid implementation uses about half of the memory bandwidth and requires significantly lower CPU resources’ [28].

机器学习的度量标准与深度学习的融合，已经非常流行，这是因为可以生成更好的模型。混合视觉处理的实现可以带来更好的性能，而且“与基于DL的算法相比，所需的乘法-累加运算的数量减少了130X-1300X，帧率有10X的提升；而且，混合实现使用的内存带宽只有一半，需要明显更低的CPU资源”。

### 3.2 Overcoming the Challenges of Deep Learning

There are also challenges introduced by DL. The latest DL approaches may achieve substantially better accuracy; however this jump comes at the cost of billions of additional math operations and an increased requirement for processing power. DL requires a these computing resources for training and to a lesser extent for inference. It is essential to have dedicated hardware (e.g. high-powered GPUs[29] and TPUs [30] for training and AI accelerated platforms such as VPUs for inference [31]) for developers of AI.

DL也带来了很多挑战。最新的DL方法可以得到更好的准确率，但是，其代价是运算量大的多，处理能力的需求激增。DL需要这样的计算资源进行训练，但推理所需的计算量则没有那么大。对于AI开发者来说，要有很好的硬件设备非常重要（如，高性能的GPUs和TPUs进行训练，以及加速AI的平台，如VPU进行推理）。

Vision processing results using DL are also dependent on image resolution. Achieving adequate performance in object classification, for example, requires high-resolution images or video – with the consequent increase in the amount of data that needs to be processed, stored, and transferred. Image resolution is especially important for applications in which it is necessary to detect and classify objects in the distance, e.g. in security camera footage. The frame reduction techniques discussed previously such as using SIFT features [26, 32] or optical flow for moving objects [27] to first identify a region of interest are useful with respect to image resolution and also with respect to reducing the time and data required for training.

使用DL进行视觉处理的结果，也依赖于图像分辨率。要在目标分类中得到很好的性能，就需要高分辨率的图像或视频，但结果就是要处理、存储和迁移的数据量变大了。对于要对有一定距离的目标进行检测和分类的应用来说，图像分辨率是非常重要的，如安保摄像头录像。前面提到的帧压缩技术，如SIFT特征，或运动目标的光流，首先识别感兴趣区域在是有用的，不止是图像分辨率上，而且可以降低训练所需的时间和数据。

DL needs big data. Often millions of data records are required. For example, PASCAL VOC Dataset consists of 500K images with 20 object categories [26][33], ImageNet consists of 1.5 million images with 1000 object categories [34] and Microsoft Common Objects in Context (COCO) consists of 2.5 million images with 91 object categories [35]. When big datasets or high computing facility are unavailable, traditional methods will come into play.

DL需要大数据。通常需要数百万数据。比如，PASCAL VOC数据集包含500K图像，分属20个目标类别，ImageNet包含150万幅图像，分属1000个目标类别，COCO数据集包含250万图像，分属91个目标类别。当大数据或高性能计算平台不可用时，传统方法就有了用武之地。

Training a DNN takes a very long time. Depending on computing hardware availability, training can take a matter of hours or days. Moreover, training for any given application often requires many iterations as it entails trial and error with different training parameters. The most common technique to reduce training time is transfer learning [36]. With respect to traditional CV, the discrete Fourier transform is another CV technique which once experienced major popularity but now seems obscure. The algorithm can be used to speed up convolutions as demonstrated by [37, 38] and hence may again become of major importance.

训练DNN需要很长时间。计算硬件可用性不一样，训练耗时可能是几个小时或几天。而且，给定任务的训练通常需要很多次迭代，因为这包含很多试错过程，训练参数也不一样。减少训练时间的最常见方法，是迁移学习。在传统CV中，离散Fourier变换是另一种CV技术，曾经很流行，但现在似乎不太流行了。这种算法可以用于加速卷积，如[37,38]所述，因此可能变得更加重要。

However, it must be said that easier, more domain-specific tasks than general image classification will not require as much data (in the order of hundreds or thousands rather than millions). This is still a considerable amount of data and CV techniques are often used to boost training data through data augmentation or reduce the data down to a particular type of feature through other pre-processing steps.

但是，需要指出的是，更简单的，领域更明确的任务，需要的数据量没那么大（在数百数千的量级上，而不是数百万的量级上）。这仍然是非常多的数据，CV技术通常还会用于对训练数据进行扩充，通过数据扩充技术，或通过其他预处理步骤，提出数据某一方面的特征。

Pre-processing entails transforming the data (usually with traditional CV techniques) to allow relationships/patterns to be more easily interpreted before training your model. Data augmentation is a common pre-processing task which is used when there is limited training data. It can involve performing random rotations, shifts, shears, etc. on the images in your training set to effectively increase the number of training images [19]. Another approach is to highlight features of interest before passing the data to a CNN with CV-based methods such as background subtraction and segmentation [39].

预处理包含对数据的变换（通常是用传统CV技术），这使得在训练之间图像的关系/模式更容易得到解释。数据扩充是一个常用的预处理任务，通常用于训练数据有限的情形。这可能包括对训练数据集中图像的随机旋转，平移，剪切等，可以有效的增加训练数据的数量。另一种方法是，用基于CV的方法强调感兴趣的特征，如背景差值和分割，然后将数据送入CNN。

### 3.3 Making Best Use of Edge Computing

If algorithms and neural network inferences can be run at the edge, latency, costs, cloud storage and processing requirements, and bandwidth requirements are reduced compared to cloud-based implementations. Edge computing can also privacy and security requirements by avoiding transmission of sensitive or identifiable data over the network.

如果算法和神经网络推理可以在端侧运行，那么与云端实现相比，延迟、运算代价、云存储和处理需求、带宽需求就都没有了。端侧计算也能满足隐私和安全的需求，避免了敏感或可辨识数据在网络上的传输。

Hybrid or composite approaches involving conventional CV and DL take great advantage of the heterogeneous computing capabilities available at the edge. A heterogeneous compute architecture consists of a combination of CPUs, microcontroller coprocessors, Digital Signal Processors (DSPs), Field Programmable Gate Arrays (FPGAs) and AI accelerating devices [31] and can be power efficient by assigning different workloads to the most efficient compute engine. Test implementations show 10x latency reductions in object detection when DL inferences are executed on a DSP versus a CPU [28].

传统CV与DL的混合方法，充分利用了端侧可用的异质计算的能力。异质计算架构包括CPU、微控制器协处理器、DSP、FPGA和AI加速器的组合，将不同的计算任务交给效率最高的计算引擎，可以得到效率很高的计算。测试实现表明，DL推理在DPS上运行时，比在CPU上运行，会得到10x的加速。

Several hybrids of deep learning and hand-crafted features based approaches have demonstrated their benefits in edge applications. For example, for facial-expression recognition, [41] propose a new feature loss to embed the information of hand-crafted features into the training process of network, which tries to reduce the difference between hand-crafted features and features learned by the deep neural network. The use of hybrid approaches has also been shown to be advantageous in incorporating data from other sensors on edge nodes. Such a hybrid model where the deep learning is assisted by additional sensor sources like synthetic aperture radar (SAR) imagery and elevation like synthetic aperture radar (SAR) imagery and elevation is presented by [40]. In the context of 3D robot vision, [42] have shown that combining both linear subspace methods and deep convolutional prediction achieves improved performance along with several orders of magnitude faster runtime performance compared to the state of the art.

有几个深度学习和手工设计的特征的混合算法，已经证明了在端侧的优势。比如，对于人脸表情识别，[41]提出了一种新的特征损失，将手工设计特征的信息嵌入到网络的训练过程中，缩小了手工设计特征和DNN学习到的特征的差异。混合方法的使用，也利于将其他传感器数据纳入到端侧节点中。这样一种混合模型，即深度学习由额外的传感器源协助，如SAR图像，见[40]。在3D机器人视觉的应用中，[42]将线性子空间方法和DNN预测方法结合，得到了更好的性能，而且运算速度也快了几个数量级。

### 3.4 Problems Not Suited to Deep Learning

There are many more changing problems in CV such as: Robotic, augmented reality, automatic panorama stitching, virtual reality, 3D modelling, motion estimation, video stabilization, motion capture, video processing and scene understanding which cannot simply be easily implemented in a differentiable manner with deep learning but need to be solved using the other "traditional" techniques.

在CV领域中，有很多更挑战的问题，如：机器人，增强现实，自动全景拼接，虚拟现实，3D建模，运动估计，视频稳定化，运动捕获，视频处理和场景理解，这些想要用深度学习用可微的形式进行实现都是不太容易的，需要用其他更传统的技术解决。

DL excels at solving closed-end classification problems, in which a wide range of potential signals must be mapped onto a limited number of categories, given that there is enough data available and the test set closely resembles the training set. However, deviations from these assumptions can cause problems and it is critical to acknowledge the problems which DL is not good at. Marcus et al. present ten concerns for deep learning, and suggest that deep learning must be supplemented by other techniques if we are to reach artificial general intelligence [43]. As well as discussing the limitations of the training procedure and intense computing and data requirements as we do in our paper, key to their discussion is identifying problems where DL performs poorly and where it can be supplemented by other techniques.

DL在解决封闭式的分类问题时表现出色，其中各种可能的图像都要映射到几个有限的类别中，如果有足够的可用数据，测试集与训练集非常像。但是，与这些假设的偏差，会导致问题，必须承认有一些问题DL并不擅长。Marcus等提出对深度学习的10个担忧，如果我们想要迈向通用人工智能的目标，建议深度学习由其他技术进行补充。

One such problem is the limited ability of DL algorithms to learn visual relations, i.e. identifying whether multiple objects in an image are the same or different. This limitation has been demonstrated by [43] who argue that feedback mechanisms including attention and perceptual grouping may be the key computational components to realising abstract visual reasoning.

一个这种问题是，DL算法在学习视觉关系时的有限能力，即，鉴别多个目标是同一目标还是不同目标的情况。这种限制在[43]中有所展示，其中反馈机制，包括注意力机制和感知分组，在实现抽象视觉推理时可能是最关键的计算单元。

It is also worth noting that ML models find it difficult to deal with priors, that is, not everything can be learnt from data, so some priors must be injected into the models [44], [45]. Solutions that have to do with 3D CV need strong priors in order to work well, e.g. image-based 3D modelling requires smoothness, silhouette and illumination information [46].

还要指出的是，ML模型很难处理先验知识，即，从数据中无法学到所有知识，所以必须将先验知识加入到模型中。与3D CV有关的解决方案，需要非常强的先验知识，才能很好的工作，如，基于图像的3D建模需要平滑性信息，轮廓信息和光照信息[46]。

Below are some emerging fields in CV where DL faces new challenges and where classic CV will have a more prominent role. 下面是CV中的一些新兴领域，其中DL面临着新的挑战，而传统CV算法会有更显著的角色。

### 3.5 3D Vision

3D vision systems are becoming increasingly accessible and as such there has been a lot of progress in the design of 3D Convolutional Neural Networks (3D CNNs). This emerging field is known as Geometric Deep Learning and has multiple applications such as video classification, computer graphics, vision and robotics. This paper will focus on 3DCNNs for processing data from 3D Vision Systems. Wherein 2D convolutional layers the kernel has the same depth so as to output a 2D matrix, the depth of a 3D convolutional kernel must be less than that of the 3D input volume so that the output of the convolution is also 3D and so preserve the spatial information.

3D视觉系统正变得越来越可行，所以3D CNNs的设计有了很多进展。这个新兴的研究领域称为几何深度学习，有很多应用，如视频分类，计算机图形学，视觉和机器人。本文关注的是处理3D视觉系统数据的3DCNNs。在2D卷积层中，卷积核的深度是一样的，这样可以输出为一个2D矩阵，但3D卷积核的深度必须少于3D输入体，这样卷积的输出才能是3D的，以保持空间信息。

The size of the input is much larger in terms of memory than conventional RGB images and the kernel must also be convolved through the input space in 3 dimensions (see Fig. 3). As a result, the computational complexity of 3D CNNs grows cubically with resolution. Compared to 2D image processing, 3D CV is made even more difficult as the extra dimension introduces more uncertainties, such as occlusions and different cameras angles as shown in Fig. 4.

输入的大小，以存储来说，比是传统RGB图像是更大的，其核心也需要在输入的3D空间中进行卷积，如图3所示。结果是，3D CNNs的计算复杂度随着分辨率成三次方增长。与2D图像处理相比，3D CV难度更大，因为多的一个维度带来了更多不确定性，比如遮挡和不同的摄像机角度，如图4所示。

FFT based methods can optimise 3D CNNs reduce the amount of computation, at the cost of increased memory requirements however. Recent research has seen the implementation of the Winograd Minimal Filtering Algorithm (WMFA) achieve a two-fold speedup compared to cuDNN(NVIDIA’s language/API for programming on their graphics cards) without increasing the required memory. The next section will include some solutions with novel architectures and pre-processing steps to various 3D data representations which have been proposed to overcome these challenges.

基于FFT的方法可以优化3D CNNs，减少计算量，其代价是增加内存需求。最近的研究实现了Winograd Minimal Filtering Algorithm(WMFA)，与cuDNN相比，提速两倍，所需内存没有增加。下一节会介绍一些新架构和预处理步骤的解决方案，处理的是3D数据，可以克服这些挑战。

Geometric Deep Learning (GDL) deals with the extension of DL techniques to 3D data. 3D data can be represented in a variety of different ways which can be classified as Euclidean or non-Euclidean [50]. 3D Euclidean-structured data has an underlying grid structure that allows for a global parametrization and having a common system of coordinates as in 2D images. This allows existing 2D DL paradigms and 2DCNNs can be applied to 3D data. 3D Euclidean data is more suitable for analysing simple rigid objects such as, chairs, planes, etc e.g. with voxel-based approaches [51]. On the other hand, 3D non-Euclidean data do not have the gridded array structure where there is no global parametrization. Therefore, extending classical DL techniques to such representations is a challenging task and has only recently been realized with architectures such as Pointnet [52].

几何深度学习(GDL)是DL技术的延伸，处理的是3D数据。3D数据可以用几种方式表示，这些方式可以归类为欧式的，或非欧的。3D欧式结构的数据有网格的结构，可以进行全局参数化，就像在2D图像中一样，有一个通用的坐标系系统。这使得现有的2D DL范式和2DCNNs可以应用于3D数据。3D欧式数据更适用于分析简单的刚性目标，如椅子，飞机等，用基于voxel的方法。另一方面，3D非欧数据则没有网格阵列结构，也就没有全局参数化。因此，将经典的DL技术拓展到这种表示中，是一个有挑战的任务，只在最近得到了一些实现，如PointNet[52]。

Continuous shape information that is useful for recognition is often lost in their conversion to a voxel representation. With respect to traditional CV algorithms, [53] propose a single dimensional feature that can be applied to voxel CNNs. A novel rotation-invariant feature based on mean curvature that improves shape recognition for voxel CNNs was proposed. The method was very successful in that when it was applied to the state-of-the-art recent voxel CNN Octnet architecture a 1% overall accuracy increase on the ModelNet10 dataset was achieved.

对于识别有用的连续形状信息，在转换到voxel表示的时候，通常会丢失。在传统CV算法方面，[53]提出了一种一维特征，可以用于voxel CNNs，提出了一种新的基于平均曲率的旋转不变特征，改进了voxel CNNs的形状识别能力。这个方法非常成功，在应用到最新的voxel CNN OctNet架构上时，在ModelNet10数据集上有1%的总体准确率提升。

### 3.6 SLAM

Visual SLAM is a subset of SLAM where a vision system is used instead of LiDAR for the registration of landmarks in a scene. Visual SLAM has the advantages of photogrammetry (rich visual data, low-cost, lightweight and low power consumption) without the associated heavy computational workload involved in post-processing. The visual SLAM problem consists of steps such as environment sensing, data matching, motion estimation, as well as location update and registration of new landmarks [54].

视觉SLAM是SLAM的一个子集，即使用了视觉系统代替了LiDAR对场景中的特征点进行配准。视觉SLAM有很多摄影测量的优势（丰富的视觉数据，代价小，轻量级和低能耗），在后处理时没有很重的计算负担。视觉SLAM问题包含以下步骤，环境感知，数据匹配，运动估计，以及位置更新和新特征点的配准。

Building a model of how visual objects appear in different conditions such as 3D rotation, scaling, lighting and extending from that representation using a strong form of transfer learning to achieve zero/one shot learning is a challenging problem in this domain. Feature extraction and data representation methods can be useful to reduce the amount of training examples needed for an ML model [55].

构建一个模型，描述视觉目标如何在不同的条件中出现，如3D旋转，缩放，使用一个很强形式的迁移学习从这种表示中拓展，得到单步学习的结果，这是非常有挑战性的。特征提取和数据表示方法可以用于减少ML模型需要的训练样本。

A two-step approach is commonly used in image based localization: place recognition followed by pose estimation. The former computes a global descriptor for each of the images by aggregating local image descriptors, e.g. SIFT, using the bag-of-words approach. Each global descriptor is stored in the database together with the camera pose of its associated image with respect to the 3D point cloud reference map. Similar global descriptors are extracted from the query image and the closest global descriptor in the database can be retrieved via an efficient search. The camera pose of the closest global descriptor would give us a coarse localization of the query image with respect to the reference map. In pose estimation, the exact pose of the query image calculated more precisely with algorithms such as the Perspective-n-Point (PnP) [13] and geometric verification [18] algorithms. [56]

两步方法通常用于基于图像的定位：位置识别，然后进行姿态估计。前者对每幅图像计算局部图像描述子，如SIFT，然后聚集起来成为全局描述子，使用的是bag-of-words方法。每个全局描述子都存储在数据库中，与图像相关的摄像头的姿态一起，相对于3D点云的参考地图。类似的全局描述子从查询图像中提取出来，从数据库中用高效的搜索检索到最接近的全局描述子。在姿态估计中，计算出查询图像的精确姿态，使用的算法可以是PnP[13]和几何验证[18]算法。

The success of image based place recognition is largely attributed to the ability to extract image feature descriptors. Unfortunately, there is no algorithm to extract local features similar to SIFT for LiDAR scans. A 3D scene is composed of 3D points and database images. One approach has associated each 3D point to a set of SIFT descriptors corresponding to the image features from which the point was triangulated. These descriptors can then be averaged into a single SIFT descriptor that describes the appearance of that point [57].

基于图像的位置识别的成功，主要归功于提取图像特征描述子的能力。不幸的是，对于LiDAR扫描，没有像SIFT这样的提取局部特征的算法。一个3D场景是由3D点和数据库图像组成的。一种方法将每个3D点都与一个SIFT描述子集合结合起来，这些描述子对应的图像特征是三角剖分的点。这些描述子然后平均为一个SIFT描述子上，描述这个点的样子。

Another approach constructs multi-modal features from RGB-D data rather than the depth processing. For the depth processing part, they adopt the well-known colourization method based on surface normals, since it has been proved to be effective and robust across tasks [58]. Another alternative approach utilizing traditional CV techniques presents the Force Histogram Decomposition (FHD), a graph-based hierarchical descriptor that allows the spatial relations and shape information between the pairwise structural subparts of objects to be characterized. An advantage of this learning procedure is its compatibility with traditional bags-of-features frameworks, allowing for hybrid representations gathering structural and local features [59].

另一种方法构建多模态特征时，是从RGB-D数据中，而不是从深度信息处理中。对于深度处理部分，他们采用的是著名的色彩化方法，基于表面法线，因为这在不同的任务中都被证明了有效性和稳定性。另一种方法利用的是传统CV技术，FHD，是一种基于图的层次化描述子，可以描述目标不同部分的相互结构的空间关系和形状信息的特征。这种学习过程的优势是，与传统的bags-of-features框架是兼容的，可以与结构化、局部特征进行混合表示。

### 3.7 360 cameras

A 360 camera, also known as an omnidirectional or spherical or panoramic camera is a camera with a 360-degree field of view in the horizontal plane, or with a visual field that covers (approximately) the entire sphere. Omnidirectional cameras are important in applications such as robotics where large visual field coverage is needed. A 360 camera can replace multiple monocular cameras and eliminate blind spots which obviously advantageous in omnidirectional Unmanned Ground Vehicles (UGVs) and Unmanned Aerial Vehicles (UAVs). Thanks to the imaging characteristic of spherical cameras, each image captures the 360◦ panorama of the scene, eliminating the limitation on available steering choices. One of the major challenges with spherical images is the heavy barrel distortion due to the ultra-wide-angle fisheye lens, which complicates the implementation of conventional human vision inspired methods such as lane detection and trajectory tracking. Additional pre-processing steps such as prior calibration and deworming are often required. An alternative approach which has been presented by [60], who circumvent these pre-processing steps by formulating navigation as a classification problem on finding the optimal potential path orientation directly based on the raw, uncalibrated spherical images.

360相机，也称为全向相机，球形相机，全景相机，是FOV在水平平面上有360度的相机，或视野覆盖了几乎整个球体的相机。全向相机有一些很重要的应用，如机器人，需要大视野覆盖。360相机可以取代多个单眼相机，消除盲点，这在全向无人地面交通器和无人飞行器中有明显的优势。多亏了球形相机的成像特性，每幅图像都捕获了场景的360度全景，消除了可用的转向系统的限制。球形成像的一个主要挑战是，由于极广角的鱼眼镜头，所以有很严重的桶状变形，而一些应用，如车道检测和轨迹追踪，其算法是基于传统的人眼视觉的，这就使得情况变得比较复杂了。还需要一些预处理步骤，先验校准，和deworm。[60]给出了另一种方法，将导航视为一种分类问题，基于原始的未校准的球形图像直接找到最佳的潜在路径方向，这就不需要这些预处理步骤了。

Panorama stitching is another open research problem in this area. A real-time stitching methodology [61] uses a group of deformable meshes and the final image and combine the inputs using a robust pixel-shader. Another approach [62], combine the accuracy provided by geometric reasoning (lines and vanishing points) with the higher level of data abstraction and pattern recognition achieved by DL techniques (edge and normal maps) to extract structural and generate layout hypotheses for indoor scenes. In sparsely structured scenes, feature-based image alignment methods often fail due to shortage of distinct image features. Instead, direct image alignment methods, such as those based on phase correlation, can be applied. Correlation-based image alignment techniques based on discriminative correlation filters (DCF) have been investigated by [23] who show that the proposed DCF-based methods outperform phase correlation-based approaches on these datasets.

全景缝合是这个领域中的另一个开放研究问题。一种实时缝合方法[61]使用了一组可变形网格和最终的图像，使用了一个稳健的pixel-shader将输入结合到一起。另一种方法[62]将几何推理（线段和消失的点）和更高层的数据抽象给出的准确性，与基于DL技术得到的模式识别（边缘和法线图）的准确性，结合到了一起，以提取结构，对室内场景生成布局假设。在稀疏结构的场景中，基于特征的图像对齐方法通常会失败，因为缺少明显的图像特征。相反，可以使用直接的图像对齐方法，如基于相位相关的。基于相关的图像对齐技术，是基于判别性相关滤波器(DCF)的，[23]对此进行了研究，提出的基于DCF的方法，在这些数据集上超过了基于相位相关的方法。

### 3.8 Dataset Annotation and Augmentation

There are arguments against the combination of CV and DL and they summarize to the conclusion that we need to re-evaluate our methods from rule-based to data-driven. Traditionally, from the perspective of signal processing, we know the operational connotations of CV algorithms such as SIFT and SURF methods, but DL leads such meaning nowhere, all you need is more data. This can be seen as a huge step forward, but may be also a backward move. Some of the pros and cons of each side of this debate have been discussed already in this paper, however, if future-methods are to be purely data-driven then focus should be placed on more intelligent methods for dataset creation.

有人认为，CV和DL的结合是有问题的，他们得出结论，我们需要重新评估从基于规则的到数据驱动的方法。传统上，从信号处理的角度来说，我们知道CV算法中的运算含义，如SIFT和SURF，但DL的意义我们是不知道的，我们只需要更多的数据。这可以认为是一个巨大的跳跃，也可以认为是一次倒退的行为。本文中讨论了一些支持观点和反对观点，但是，如果未来的方法纯粹是数据驱动的，那么应当关注创建数据集的更智能的方法。

The fundamental problem of current research is that there is no longer enough data for advanced algorithms or models for special applications. Coupling custom datasets and DL models will be the future theme to many research papers. So many researchers’ outputs consist of not only algorithms or architectures, but also datasets or methods to amass data. Dataset annotation is a major bottleneck in the DL workflow which requires many hours of manual labelling. Nowhere is this more problematic than in semantic segmentation applications where every pixel needs to be annotated accurately. There are many useful tools available to semi-automate the process as reviewed by [20], many of which take advantage of algorithmic approaches such as ORB features [55], polygon morphing [63], semi-automatic Area of Interest (AOI) fitting [55] and all of the above [63].

目前研究的基本问题是，对于特殊的应用来说，那些先进的算法或模型没有足够的数据。将定制的数据集和DL模型结合，是未来很多研究文章的主题。很多研究者的输出包含了不仅包含算法或架构，还有数据集或积累数据的方法。数据集标注是DL工作流的一个主要瓶颈，需要很多小时的手工标注工作。语义分割应用的问题最大，因为每个像素都需要准确的进行标注。[20]总结流很多有用半自动化这个过程的工具，很多利用了算法的方法，如ORB特征[55]，多边形变形[63]，自动ROI适配[55]和所有上述方法[63]。

The easiest and most common method to overcome limited datasets and reduce overfitting of deep learning models for image classification is to artificially enlarge the dataset using label-preserving transformations. This process is known as dataset augmentation and it involves the artificial generation of extra training data from the available ones, for example, by cropping, scaling, or rotating images [64]. It is desirable for data augmentation procedures to require very little computation and to be implementable within the DL training pipeline so that the transformed images do not need to be stored on disk. Traditional algorithmic approaches that have been employed for dataset augmentation include Principle Component Analysis (PCA) [1], adding noise, interpolating or extrapolating between samples in a feature space [65] and modelling the visual context surrounding objects from segmentation annotations [66].

克服数据集有限性、降低深度学习模型对图像分类过拟合最简单最常见的方法是，使用保持标签的变换人工增大数据集。这个过程称为数据扩充，是从现有数据中额外生成更多的训练数据，比如，剪切，缩放，或旋转。数据扩充的过程需要很少计算，在DL训练过程中是可以实现的，变换的图像也不需要存储在磁盘上。数据集扩充采用了传统的算法方法，包括PCA，增加噪声，在不同样本的特征空间中插值或外插值，从语义标注中对目标周围的视觉上下文进行建模。

## Conclusion

A lot of the CV techniques invented in the past 20 years have become irrelevant in recent years because of DL. However, knowledge is never obsolete and there is always something worth learning from each generation of innovation. That knowledge can give you more intuitions and tools to use especially when you wish to deal with 3D CV problems for example. Knowing only DL for CV will dramatically limit the kind of solutions in a CV engineer’s arsenal.

过去20年中提出的很多CV技术与最近的DL技术相关性不大。但是，知识是永不废弃的，从每一代的创新中永远都可以学到一些值得学习的东西。这些知识可以给你更多的直觉和工具，尤其是在你要解决3D CV问题时。只知道CV中的DL知识，会极大的限制解决方案的种类。

In this paper we have laid down many arguments for why traditional CV techniques are still very much useful even in the age of DL. We have compared and contrasted traditional CV and DL for typical applications and discussed how sometimes traditional CV can be considered as an alternative in situations where DL is overkill for a specific task.

本文中，关于为什么传统CV技术在DL时代仍然非常有用，我们给出了很多意见。我们比较了传统CV和DL在典型应用中的表现，讨论了对一些特殊应用，DL会过犹不及的时候，传统CV可以是一个很好的替代。

The paper also highlighted some areas where traditional CV techniques remain relevant such as being utilized in hybrid approaches to improve performance. DL innovations are driving exciting breakthroughs for the IoT (Internet of Things), as well as hybrid techniques that combine the technologies with traditional algorithms. Additionally, we reviewed how traditional CV techniques can actually improve DL performance in a wide range of applications from reducing training time, processing and data requirements to being applied in emerging fields such as SLAM, Panoramic- stitching, Geometric Deep Learning and 3D vision where DL is not yet well established.

本文还强调了一些领域，其中传统CV技术仍然是有用的，如构建混合模型改进性能。DL的创新在IoT方面有很大突破，以及将DL与传统算法结合到一起的混合技术。另外，我们还回顾了传统CV技术在很多应用中是怎样改进DL性能的，如缩短训练时间，在新兴领域，如SLAM，全景缝合，几何DL和3D视觉中的处理和数据需求，这里DL还没有得到很好的应用。

The digital image processing domain has undergone some very dramatic changes recently and in a very short period. So much so it has led us to question whether the CV techniques that were in vogue prior to the AI explosion are still relevant. This paper hopefully highlight some cases where traditional CV techniques are useful and that there is something still to gain from the years of effort put in to their development even in the age of data-driven intelligence.

数字图像处理领域最近有了极大的发展。我们会问到，AI爆发之前的CV技术还有用吗？本文给出了传统CV技术仍然有用的几个领域，在数据驱动智能的时代，之前多年的努力还是可以结合到最新的进展中的。