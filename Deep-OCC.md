# Learning Deep Features for One-Class Classification

Pramuditha Perera, Vishal M. Patel Johns Hopkins University

## 0. Abstract

We present a novel deep-learning based approach for one-class transfer learning in which labeled data from an unrelated task is used for feature learning in one-class classification. The proposed method operates on top of a Convolutional Neural Network (CNN) of choice and produces descriptive features while maintaining a low intra-class variance in the feature space for the given class. For this purpose two loss functions, compactness loss and descriptiveness loss are proposed along with a parallel CNN architecture. A template matching-based framework is introduced to facilitate the testing process. Extensive experiments on publicly available anomaly detection, novelty detection and mobile active authentication datasets show that the proposed Deep One-Class (DOC) classification method achieves significant improvements over the state-of-the-art.

我们提出了一种的新的基于深度学习的方法进行一类迁移学习，用不相关任务的标注数据进行特征学习，进行一类分类。提出的方法是在CNN上实现的，会产生有描述性的特征，同时对给定的类别，在特征空间中保持类内变化幅度很低。为达到这个目标，提出了两种损失函数，紧凑型损失和描述性损失，以及相应的CNN架构。提出了一种基于模板匹配的框架，使测试过程很方便。在公开可用的异常检测、新异类检测和移动主动认证数据集上进行了广泛的试验，表明提出的DOC分类方法在过去最好的结果上获得了显著的改进。

**Index Terms** — One-class classification, anomaly detection, novelty detection, deep learning.

## 1. Introduction

One-class classification is a classical machine learning problem that has received considerable attention in the recent literature [1], [40], [5], [32], [31], [29]. The objective of one-class classification is to recognize instances of a concept by only using examples of the same concept [16] as shown in Figure 1. In such a training scenario, instances of only a single object class are available during training. In the context of this paper, all other classes except the class given for training are called alien classes. During testing, the classifier may encounter objects from alien classes. The goal of the classifier is to distinguish the objects of the known class from the objects of alien classes. It should be noted that one-class classification is different from binary classification due to the absence of training data from a second class.

一类分类是一个经典的机器学习问题，最近很多文献在关注。一类分类的目标是，使用相同概念的样本，识别这个概念的实例，如图1所示。在这样的一个训练场景中，在训练中只能使用一个目标类别的实例。在本文的上下文中，给定训练类别以外的所有其他类别都称为外来类别。在测试时，分类器会遇到外来类别的目标。分类器的目标是区分已知类别的目标和外来类别的目标。应当注意，一类分类与二分类是不同的，因为缺少其他类别的训练数据。

One-class classification is encountered in many real-world computer vision applications including novelty detection [25], anomaly detection [6], [37], medical imaging and mobile active authentication [30], [33], [28], [35]. In all of these applications, unavailability of samples from alien classes is either due to the openness of the problem or due to the high cost associated with obtaining the samples of such classes. For example, in a novelty detection application, it is counter intuitive to come up with novel samples to train a classifier. On the other hand, in mobile active authentication, samples of alternative classes (users) are often difficult to obtain due to the privacy concerns [24].

一类分类在很多真实世界计算机视觉应用中都会遇到，包括新异类检测，异常检测，医学成像和移动主动认证。在所有这些应用中，外来类别样本的不可用性，要么是因为问题的开放性，要么是获得这些类别的样本的代价很大。比如，在新异类检测应用中，用异类样本来训练这样的类别是违反直觉的。另外，在移动主动认证中，其他类别（用户）的样本很难获得，因为有隐私方面的考虑。

Despite its importance, contemporary one-class classification schemes trained solely on the given concept have failed to produce promising results in real-world datasets ( [1], [40] has achieved an Area Under the Curve in the range of 60%-65% for CIFAR10 dataset [21] [31]). However, we note that computer vision is a field rich with labeled datasets of different domains. In this work, in the spirit of transfer learning, we step aside from the conventional one-class classification protocol and investigate how data from a different domain can be used to solve the one-class classification problem. We name this specific problem One-class transfer learning and address it by engineering deep features targeting one-class classification tasks.

尽管很重要，目前的在给定概念上训练的一类分类方案，在真实世界数据集中很难取得很好的效果。但是，我们注意到，计算机视觉领域中，不同领域的标注数据集非常丰富。本文中，在迁移学习的思想下，我们避开了传统一类分类的协议，研究其他领域的数据怎样用于解决一类分类问题。我们将这个特定问题命名为一类迁移学习，通过定制专为一类分类任务的深度特征，来进行处理。

In order to solve the One-class transfer learning problem, we seek motivation from generic object classification frameworks. Many previous works in object classification have focused on improving either the feature or the classifier (or in some cases both) in an attempt to improve the classification performance. In particular, various deep learning-based feature extraction and classification methods have been proposed in the literature and have gained a lot of traction in recent years [22], [44]. In general, deep learning-based classification schemes have two subnetworks, a feature extraction network (g) followed by a classification sub network (hc), that are learned jointly during training. For example, in the popular AlexNet architecture [22], the collection of convolution layers may be regarded as (g) where as fully connected layers may collectively be regarded as (hc). Depending on the output of the classification sub network (hc), one or more losses are evaluated to facilitate training. Deep learning requires the availability of multiple classes for training and extremely large number of training samples (in the order of thousands or millions). However, in learning tasks where either of these conditions are not met, the following alternative strategies are used:

为解决一类迁移学习问题，我们从通用目标分类框架中寻找动力。很多之前的目标分类工作聚焦在改进特征或改进分类器（有时候两者都有），以改进分类性能。特别是，文献中提出了各种基于深度学习的特征提取和分类方法，近些年获得了很多关注。一般来说，基于深度学习的分类方案有两个子网络，一个特征提取网络(g)，一个分类网络(hc)，在训练的时候是联合学习的。比如，在流行的AlexNet架构中[22]，卷积层的集合可以认为是(g)，而全连接层可以认为是(hc)。依赖于分类子网络(hc)的输出，计算一个或多个损失，以促进训练。深度学习需要多个类别的可用性和大量训练样本以进行训练（数量级为数千或上百万）。但是，在这些条件不具备的学习任务中，可以使用下面的替代策略：

(a) **Multiple classes, many training samples**: This is the case where both requirements are satisfied. Both feature extraction and classification networks, g and hc are trained end-to-end (Figure 2(a)). The network parameters are initialized using random weights. Resultant model is used as the pre-trained model for fine tuning [22], [17], [34].

**多个类别，很多训练样本**：这是两种要求都满足的情况。特征提取g和分类网络hc都可以进行端到端的训练（图2a）。网络参数使用随机权重进行初始化。得到的模型作为预训练模型进行精调。

(b) **Multiple classes, low to medium number of training samples**: The feature extraction network from a pre-trained model is used. Only a new classification network is trained in the case of low training samples (Figure 2(b)). When medium number of training samples are available, feature extraction network (g) is divided into two sub-networks - shared feature network (gs) and learned feature network (gl), where g = gs ◦ gl. Here, gs is taken from a pre-trained model. gl and the classifier are learned from the data in an end-to-end fashion (Figure 2(c)). This strategy is often referred to as fine-tuning [13].

**多个类别，训练样本数量少或中等数量**：使用预训练模型的特征提取网络。在训练样本很少的情况下，只训练一个新的分类网络（图2b）。当可用的训练样本数量不少时，特征提取网络g分成两个子网络 - 共享特征网络gs和学习的特征网络gl，其中g = gs ◦ gl。这里，gs是预训练模型中的。gl和分类器是从数据中端到端学来的（图2c）。这种策略通常称为精调[13]。

(c) **Single class or no training data**: A pre-trained model is used to extract features. The pre-trained model used here could be a model trained from scratch (as in (a)) or a model resulting from fine-tuning (as in (b)) [10], [24] where training/fine-tuning is performed based on an external dataset. When training data from a class is available, a one-class classifier is trained on the extracted features (Figure 2(d)).

**单类别或无训练数据**：用预训练模型来提取特征。这里使用的预训练模型可以从头训练的模型（与a中一样），或精调得到的模型（和b中一样），这里说的训练、精调是基于外部数据集进行的。当一类的训练数据可用时，就在提取的数据中训练一个一类分类器（图2d）。

In this work, we focus on the task presented in case (c) where training data from a single class is available. Strategy used in case (c) above uses deep-features extracted from a pre-trained model, where training is carried out on a different dataset, to perform one-class classification. However, there is no guarantee that features extracted in this fashion will be as effective in the new one-class classification task. In this work, we present a feature fine tuning framework which produces deep features that are specialized to the task at hand. Once the features are extracted, they can be used to perform classification using the strategy discussed in (c).

本文中，我们聚焦在情况c中的工作，即单类训练数据可用的情况。在情况c中使用的策略，使用从预训练模型中提取的深度特征，其中训练是在不同的数据集上进行的，以进行一类分类。但是，这种方式提取的特征无法保证在新的一类分类任务中是有效的。本文中，我们提出一个特征精调框架，产生的深度特征专用于眼前的任务。一旦提取了特征，就可以用于使用c中的策略进行分类。

In our formulation (shown in Figure 2 (e)), starting from a pre-trained deep model, we freeze initial features (gs) and learn (gl) and (hc). Based on the output of the classification sub-network (hc), two losses compactness loss and descriptiveness loss are evaluated. These two losses, introduced in the subsequent sections, are used to assess the quality of the learned deep feature. We use the provided one-class dataset to calculate the compactness loss. An external multi-class reference dataset is used to evaluate the descriptiveness loss. As shown in Figure 3, weights of gl and hc are learned in the proposed method through back-propagation from the composite loss. Once training is converged, system shown in setup in Figure 2(d) is used to perform classification where the resulting model is used as the pre-trained model.

在我们的表述中（图2e），从一个预训练深度模型开始，我们冻结初始特征gs和学习特征gl和hc。基于分类子网络hc的输出，计算两个损失，即紧凑性损失和描述性损失。这两个损失在后面的章节中介绍，用于评估学习到的深度特征的质量。我们使用已有的一类数据集来计算紧凑性损失。使用一个外部的多类参考数据集，来评估描述性损失。如图3所示，gl和hc的权重是用提出的方法通过复合损失的反向传播学习得到的。一旦训练收敛，图2d中所示的设置的系统用于进行分类，得到的模型用作预训练模型。

In summary, this paper makes the following three contributions. 总结起来，本文有如下三个贡献。

- We propose a deep-learning based feature engineering scheme called Deep One-class Classification (DOC), targeting one-class problems. To the best of our knowledge this is one of the first works to address this problem. 我们提出了一种基于深度学习的特征工程方案，称为深度一类分类(DOC)，解决一类问题。据我们所知，这是这种类型中第一个解决这种问题的。
   
- We introduce a joint optimization framework based on two loss functions - compactness loss and descriptiveness loss. We propose compactness loss to assess the compactness of the class under consideration in the learned feature space. We propose using an external multi-class dataset to assess the descriptiveness of the learned features using descriptiveness loss. 我们提出一个联合优化框架，基于两个损失函数，紧凑性损失和描述性损失。我们提出了紧凑性损失，以评估在考虑的类别在学习的特征空间的紧凑性。我们提出使用外部多类别数据集使用描述性损失来评估学习的特征的描述性。
   
- On three publicly available datasets, we achieve state-of-the-art one-class classification performance across three different tasks. 在三种公开可用的数据集中，我们在三个不同的任务中获得了目前最好的一类分类性能。

Rest of the paper is organized as follows. In Section II, we review a few related works. Details of the proposed deep one-class classification method are given in Section III. Experimental results are presented in Section IV. Section V concludes the paper with a brief discussion and summary.

本文组织如下。在第2部分中，我们回顾了一些相关的工作。第3部分给出了提出的深度一类分类方法的细节。第4部分给出了试验结果。第5部分以简短的讨论和摘要总结了本文。

## 2. Related Work

Generic one-class classification problem has been addressed using several different approaches in the literature. Most of these approaches predominately focus on proposing a better classification scheme operating on a chosen feature and are not specifically designed for the task of one-class image classification. One of the initial approaches to one-class learning was to estimate a parametric generative model based on the training data. Work in [36], [3], propose to use Gaussian distributions and Gaussian Mixture Models (GMMs) to represent the underlying distribution of the data. In [19] comparatively better performances have been obtained by estimating the conditional density of the one-class distribution using Gaussian priors.

通用的一类分类问题，文献中有几种不同的方法进行了处理。多数这种方法聚焦在，在选定的特征上提出更好的分类方法，并不是专门为一类图像分类问题设计。一类学习最开始方法中的一个，是基于训练数据估计一个参数化生成式模型。[36,3]的工作提出来使用高斯分布和高斯混合模型GMM来表示数据的潜在分布。在[19]中，通过使用高斯先验估计一类分布的条件密度，得到了相对更好的性能。

The concept of Support Vector Machines (SVMs) was extended for the problem of one-class classification in [43]. Conceptually this algorithm treats the origin as the out-of-class region and tries to construct a hyperplane separating the origin with the class data. Using a similar motivation, [45] proposed Support Vector Data Description (SVDD) algorithm which isolates the training data by constructing a spherical separation plane. In [23], a single layer neural network based on Extreme Learning Machine is proposed for one-class classification. This formulation results in an efficient optimization as layer parameters are updated using closed form equations. Practical one-class learning applications on different domains are predominantly developed based on these conceptual bases.

[43]将SVMs的概念拓展到一类分类问题中。概念上来说，这个算法将原点认为是类别外的区域，尝试构建一个超平面，将原点与类别数据区分开来。使用类似的动机，[45]提出SVDD算法，将训练数据通过构建一个球形分割平面，来孤立出来。在[23]中，提出了基于单层神经网络的ELM进行一类分类。这种表述带来了很高效的优化，因为层的参数使用封闭形式的等式进行更新。在不同领域中的实际的一类学习应用，主要是基于这些概念来建立的。

In [39], visual anomalies in wire ropes are detected based on Gaussian process modeling. Anomaly detection is performed by maximizing KL divergence in [38], where the underlying distribution is assumed to be a known Gaussian. A detailed description of various anomaly detection methods can be found in [6].

在[39]中，基于高斯过程建模检测了钢丝绳中的视觉异常。[38]中通过最大化KL散度进行异常检测，其中潜在的分布假设是已知的高斯分布。各种异常检测方法的详细描述可以见[6]。

Novelty detection based on one-class learning has also received a significant attention in recent years. Some of the earlier works in novelty detection focused on estimating a parametric model for data and to model the tail of the distribution to improve classification accuracy [8], [37]. In [4], null space-based novelty detection framework for scenarios when a single and multiple classes are present is proposed. However, it is mentioned in [4] that their method does not yield superior results compared with the classical one-class classification methods when only a single class is present. An alternative null space-based approach based on kernel Fisher discriminant was proposed in [11] specifically targeting one-class novelty detection. A detailed survey of different novelty detection schemes can be found in [25], [26].

基于一类学习的新异类检测在近年来获得了非常多的注意力。新异类检测中的一些早期工作聚焦在为数据估计参数模型，并对分布的尾巴进行建模，以改进分类准确率。在[4]中，在存在单类和多类的情况下，提出了一种基于null space的新异类检测框架。但是，在[4]中提到了，当只有一类存在时，他们的方法并没有得到比经典一类分类方法更好的结果。另一种基于null space的方法是基于核Fisher判别的，在[11]中提出专门用于一类新异类检测。[25,26]给出了不同的新异类检测方法的详细综述。

Mobile-based Active Authentication (AA) is another application of one-class learning which has gained interest of the research community in recent years [30]. In mobile AA, the objective is to continuously monitor the identity of the user based on his/her enrolled data. As a result, only the enrolled data (i.e. one-class data) are available during training. Some of the recent works in AA has taken advantage of CNNs for classification. Work in [42], uses a CNN to extract attributes from face images extracted from the mobile camera to determine the identity of the user. Various deep feature-based AA methods have also been proposed as benchmarks in [24] for performance comparison.

移动主动认证是一类学习的另一种应用，最近几年有很多研究。在移动AA中，目标是基于用户登记的数据，连续监测用户的身份。结果是，只有登记的数据（即，一类数据）在训练时可用。AA中的最近一些工作利用CNNs进行分类。[42]使用CNN来从面部图像中来提取属性，确定用户的身份。各种基于深度特征的AA方法在[24]中作为基准测试，用于性能比较。

Since one-class learning is constrained with training data from only a single class, it is impossible to adopt a CNN architectures used for classification [22], [44] and verification [7] directly for this problem. In the absence of a discriminative feature generation method, in most unsupervised tasks, the activation of a deep layer is used as the feature for classification. This approach is seen to generate satisfactory results in most applications [10]. This can be used as a starting point for one-class classification as well. As an alternative, autoencoders [47], [15] and variants of autoencoders [48], [20] can also to be used as feature extractors for one-class learning problems. However, in this approach, knowledge about the outside world is not taken into account during the representation learning. Furthermore, none of these approaches were specifically designed for the purpose of one-class learning. To the best of our knowledge, one-class feature learning has not been addressed using a deep-CNN architecture to date.

由于一类学习局限于单类的训练数据，无法用用于分类和验证的CNN架构，来直接解决这个问题。在缺少具有区分性的特征生成方法的情况下，在多数无监督任务中，深度层的激活值就用于分类的特征。这种方法在多数应用中都得到了令人满意的结果。这也可以用作一类分类的起始点。另一种方法是，自编码器和各种变体也可以用于特征提取器，解决一类学习问题。但是，在这种方法中，在表示学习中，并没有将外部世界的知识考虑进去。而且，这些方法都不是专门设计用于一类学习的目的的。据我们所知，一类特征学习还没有使用深度CNN架构来进行处理。

## 3. Deep One-Class Classification (DOC)

### 3.1. Overview

In this section, we formulate the objective of one-class feature learning as an optimization problem. In the classical multiple-class classification, features are learned with the objective of maximizing inter-class distances between classes and minimizing intra-class variances within classes [2]. However, in the absence of multiple classes such a discriminative approach is not possible.

本节中，我们将一类特征学习的目的表述成一个优化问题。在经典多类分类中，特征的学习是以最大化类别间距离，最小化类别内距离为目的的。但是，在缺少多类的情况下，这样一种判别式方法是不可能的。

In this light, we outline two important characteristics of features intended for one-class classification. 从这一角度看，我们列出两个重要的特点，进行以特征为目的的一类分类。

**Compactness C**. A desired quality of a feature is to have a similar feature representation for different images of the same class. Hence, a collection of features extracted from a set of images of a given class will be compactly placed in the feature space. This quality is desired even in features used for multi-class classification. In such cases, compactness is quantified using the intra-class distance [2]; a compact representation would have a lower intra-class distance.

**紧凑性C**。特征的一个期望质量是，对于同类的不同图像由类似的特征表示。因此，一个给定类别的图像集合所提取的特征，应当在特征空间中的紧凑排列的。对于多类别分类的情况，特征的这个质量也是很理想的。在这种情况下，紧凑性是用类内距离量化的[2]；一个紧凑的表示应当类内距离更小。

**Descriptiveness D**. The given feature should produce distinct representations for images of different classes. Ideally, each class will have a distinct feature representation from each other. Descriptiveness in the feature is also a desired quality in multi-class classification. There, a descriptive feature would have large inter-class distance [2].

**描述性D**。给定的特征应当对不同类型的图像产生独特的表示。理想情况下，每个类别应当都有独特的特征表示。特征中的描述性在多类分类中也是理想的质量。有描述性的特征，应当有大的类间距离[2]。

It should be noted that for a useful (discriminative) feature, both of these characteristics should be satisfied collectively. Unless mutually satisfied, neither of the above criteria would result in a useful feature. With this requirement in hand, we aim to find a feature representation g that maximizes both compactness and descriptiveness. Formally, this can be stated as an optimization problem as follows,

值的注意的是，对于一个有用的（有区分性）的特征，这些特性都应当满足。除非互相满足，上面的哪个准则都不会得到有用的特征。有了这些要求，我们的目标是找到一个特征表示g，最大化紧凑性和描述性。正式的，这可以表述为下面的优化问题：

$$\hat g = max_g D(g(t)) + λC(g(t))$$(1)

where t is the training data corresponding to the given class and λ is a positive constant. Given this formulation, we identify three potential strategies that may be employed when deep learning is used for one-class problems. However, none of these strategies collectively satisfy both descriptiveness and compactness.

其中t是给定类别的训练数据，λ是一个正常数。给定这个公式，我们认为当深度学习用于一类问题时，可以采用三种可能的策略。但是，这三种策略都没有满足描述性和紧凑性两者。

(a) **Extracting deep features**. Deep features are first extracted from a pre-trained deep model for given training images. Classification is done using a one-class classification method such as one-class SVM, SVDD or k-nearest neighbor using extracted features. This approach does not directly address the two characteristics of one-class features. However, if the pre-trained model used to extract deep features was trained on a dataset with large number of classes, then resulting deep features are likely to be descriptive. Nevertheless, there is no guarantee that the used deep feature will possess the compactness property.

(a) **提取深度特征**。对给定的训练图像，深度特征首先由预训练深度模型提取。分类是使用一类分类方法进行，比如使用提取的特征进行一类SMV，SVDD或k-nn。这个方法没有直接处理一类特征的两个性质。但是，如果用于提取深度特征的预训练模型是在大量类别上训练的，得到的深度特征很可能是具有描述性的。尽管如此，并不能保证，使用的深度特征具有紧凑性的属性。

(b) **Fine-tune a two class classifier using an external dataset**. Pre-trained deep networks are trained based on some legacy dataset. For example, models used for the ImageNet challenge are trained based on the ImageNet dataset [9]. It is possible to fine tune the model by representing the alien classes using the legacy dataset. This strategy will only work when there is a high correlation between alien classes and the legacy dataset. Otherwise, the learned feature will not have the capacity to describe the difference between a given class and the alien class thereby violating the descriptiveness property.

(b) **使用外部数据集精调两类分类器**。预训练的深度网络是基于一些经典数据集训练的。比如，用在ImageNet挑战上的模型，是基于ImageNet数据集训练的。使用经典数据集来表示外部类别，来精调模型，是可能的。这个策略只有当外部类别与经典数据集有很高的关联，才会好用。否则，学习到的特征不会有能力来描述给定类别和外部类别之间的差异，因为就不符合描述性的性质。

(c) **Fine-tune using a single class data**. Fine-tuning may be attempted by using data only from the given single class. For this purpose, minimization of the traditional cross-entropy loss or any other appropriate distance could be used. However, in such a scenario, the network may end up learning a trivial solution due to the absence of a penalty for miss-classification. In this case, the learned representation will be compact but will not be descriptive.

(c) **使用单个类别数据来精调**。也可以只使用给定的单个类别的数据来精调。为此，可以使用传统交叉熵或其他合适的距离的最小化来进行。但是，在这样一个场景中，网络最后会成为学习一个无意义的解，因为缺少多类别的惩罚。在这种情况下，学习到的表示会是紧凑的，但是没有描述性。

Let us investigate the appropriateness of these three strategies by conducting a case study on the abnormal image detection problem where the considered class is the normal chair class. In abnormal image detection, initially a set of normal chair images are provided for training as shown in Figure 4(a). The goal is to learn a representation such that, it is possible to distinguish a normal chair from an abnormal chair.

我们通过在非正常图像检测问题上进行案例研究，来研究这三个策略的合适性，其中考虑的类别是正常椅子的类别。在非正常图像检测中，有正常椅子图像的集合可用于训练，如图4a所示。其目标是学习一个表示，这样可能将正常椅子和非正常椅子区分开来。

The trivial approach to this problem is to extract deep features from an existing CNN architecture (solution (a)). Let us assume that the AlexNet architecture [22] is used for this purpose and fc7 features are extracted from each sample. Since deep features are sufficiently descriptive, it is reasonable to expect samples of the same class to be clustered together in the extracted feature space. Illustrated in Figure 4(b) is a 2D visualization of the extracted 4096 dimensional features using t-SNE [46]. As can be seen from Figure4(b), the AlexNet features are not able to enforce sufficient separation between normal and abnormal chair classes.

对这个问题的平常方法是，从现有的CNN架构中提取深度特征（方法a）。我们假设，AlexNet架构用于此目的，从每个样本中提取fc7特征。由于深度特征是足够有描述性的，我们期待，同样类别的样本，在提取出的特征空间中，是可以聚类到一起的，这是合理的。图4b中所述的，是提取出的4096维特征使用t-SNE的2D可视化。如图4b中所示，AlexNet特征不足以区分正常的椅子类别和非正常的椅子类别。

Another possibility is to train a two class classifier using the AlexNet architecture by providing normal chair object images and the images from the ImageNet dataset as the two classes (solution (b)). However, features learned in this fashion produce similar representations for both normal and abnormal images, as shown in Figure4(c). Even though there exist subtle differences between normal and abnormal chair images, they have more similarities compared to the other ImageNet objects/images. This is the main reason why both normal and abnormal images end up learning similar feature representations.

另一种可能性，是使用AlexNet架构训练一个两类分类器，将正常的椅子目标的图像和ImageNet数据集中的图像作为两类图像（方法b）。但是，以这种方式学习到的特征，对正常图像和非正常图像都产生了类似的表示，如图4c所示。即使正常椅子图像和非正常椅子图像之间有很微小的差异，与其他的ImageNet目标图像相比，它们之间还是有更多的相似性的。这就是正常图像和非正常图像都得到类似的特征表示的主要原因。

A naive, and ineffective, approach would be to fine-tune the pre-trained AlexNet network using only the normal chair class (solution (c)). Doing so, in theory, should result in a representation where all normal chairs are compactly localized in the feature space. However, since all class labels would be identical in such a scenario, the fine-tuning process would end up learning a futile representation as shown in Figure4(d). The reason why this approach ends up yielding a trivial solution is due to the absence of a regularizing term in the loss function that takes into account the discriminative ability of the network. For example, since all class labels are identical, a zero loss can be obtained by making all weights equal to zero. It is true that this is a valid solution in the closed world where only normal chair objects exist. But such a network has zero discriminative ability when abnormal chair objects appear.

一种朴素的、效率不高的方法是，只使用正常的椅子类别来精调预训练的Alexnet网络（方法c）。这样，理论上可以得到的结果是，所有的正常的椅子的表示会紧凑的分布在特征空间中的一个局部。但是，由于在这种场景中，所有类别标签都是一样的，精调过程结果会学习到一个无效的表示，如图4d所示。这种方法得到无意义解的原因，是因为缺少损失函数中的正则化项，来将网络的区分能力考虑进去。比如，由于所有的类别标签都是一样的，因此将所有权重置为零的话，那么就可以得到零损失。在只存在正常椅子目标的情况下，这是一个有效的解。但当非正常椅子出现的时候，这样一个网络的区分能力为零。

None of the three strategies discussed above are able to produce features that are both compact and descriptive. We note that out of the three strategies, the first produces the most reasonable representation for one-class classification. However, this representation was learned without making an attempt to increase compactness of the learned feature. Therefore, we argue that if compactness is taken into account along with descriptiveness, it is possible to learn a more effective representation.

上面讨论的三种策略，没有一个可以产生既紧凑又有描述性的特征。我们注意到，在这三种策略中，第一种可以得到最合理一类分类的表示。但是，这个表示的学习，并没有增加学习到的特征的紧凑性的目的。因此，我们认为，如果紧凑性与描述性同时进行考虑，是可能学到一种更有效的表示的。

### 3.2. Proposed Loss Functions

In this work, we propose to quantify compactness and descriptiveness in terms of measurable loss functions. Variance of a distribution has been widely used in the literature as a measure of the distribution spread [14]. Since spread of the distribution is inversely proportional to the compactness of the distribution, it is a natural choice to use variance of the distribution to quantify compactness. In our work, we approximate variance of the feature distribution by the variance of each feature batch. We term this quantity as the compactness loss (lC).

On the other hand, descriptiveness of the learned feature cannot be assessed using a single class training data. However, if there exists a reference dataset with multiple classes, even with random object classes unrelated to the problem at hand, it can be used to assess the descriptiveness of the engineered feature. In other words, if the learned feature is able to perform classification with high accuracy on a different task, the descriptiveness of the learned feature is high. Based on this rationale, we use the learned feature to perform classification on an external multi-class dataset, and consider classification loss there as an indicator of the descriptiveness of the learned feature. We call the cross-entropy loss calculated in this fashion as the descriptiveness loss (lD). Here, we note that descriptiveness loss is low for a descriptive representation.

With this formulation, the original optimization objective in equation (1) can be re-formulated as,

$$\hat g = min_g l_D(r) + λl_C(t)$$(2)

where lC and lD are compactness loss and descriptiveness loss, respectively and r is the training data corresponding to the reference dataset. The tSNE visualization of the features learned in this manner for normal and abnormal images are shown in Figure 4(e). Qualitatively, features learned by the proposed method facilitate better distinction between normal and abnormal images as compared with the cases is shown in Figure 2(b)-(d).

### 3.3. Terminology 术语

Based on the intuition given in the previous section, the architecture shown in Figure 5 (a) is proposed for one-class classification training and the setup shown in Figure 5 (b) for testing. They consist of following elements:

基于前一节给出的直觉知识，我们提出了图5a所示的架构，用于一类分类的训练，图5b中的设置用于测试。这包括下面的元素：

**Reference Network** (R): This is a pre-trained network architecture considered for the application. Typically it contains a repetition of convolutional, normalization, and pooling layers (possibly with skip connections) and is terminated by an optional set of fully connected layers. For example, this could be the AlexNet network [22] pre-trained using the ImageNet [9] dataset. Reference network can be seen as the composition of a feature extraction sub-network g and a classification sub-network hc. For example, in AlexNet, conv1-fc7 layers can be associated with g and fc8 layer with hc. Descriptive loss (lD) is calculated based on the output of hc.

**参考网络**(R)：这是一个预训练的网络架构。一般包括卷积层，归一化层和池化层的重复组合（很可能有跳跃连接），最后以若干个全连接层的结束。比如，可能是用ImageNet数据集预训练的AlexNet网络。参考网络可以视为由特征提取子网络g和分类子网络hc构成。比如，在AlexNet中，conv1-fc7可以视为g，fc8就是hc。描述性损失lD就是基于hc的输出来计算得到的。

**Reference Dataset** (r): This is the dataset (or a subset of it) used to train the network R. Based on the example given, reference dataset is the ImageNet dataset [9] (or just a subset of the ImageNet dataset). 参考数据集r：这是用于训练网络R的数据集（或子集）。基于给定的例子，参考数据集是ImageNet数据集（或是其子集）。

**Secondary Network** (S): This is a second CNN where the network architecture is structurally identical to the reference network. Note that g and hc are shared in both of these networks. Compactness loss (lC) is evaluated based on the output of hc. For the considered example, S would have the same structure as R (AlexNet) up to fc8. 第二网络S：这是第二个CNN，其网络架构与参考网络的结构是一样的。注意g和hc是在这些网络中共享的。紧凑性损失lC就是基于hc的输出计算得到的。对于考虑的例子，S与R的网络结构是一样的，即到fc8的AlexNet。

**Target Dataset** (t): This dataset contains samples of the class for which one-class learning is used for. For example, for an abnormal image detection application, this dataset will contain normal images (i.e. data samples of the single class considered). 目标数据集t：这个数据集是一类学习所用于的类别。比如，对于一个异常的图像检测应用，这个数据集应当包含正常的图像（即，考虑单个类别的数据样本）。

**Model** (W): This corresponds to the collection of weights and biases in the network, g and hc. Initially, it is initialized by some pre-trained parameters W0. 模型W：这对应着网络g和hc中的权重和偏置。开始，这是由一些预训练的参数W0初始化的。

**Compactness loss** (lC) : All the data used during the training phase will belong to the same class. Therefore they all share the same class label. Compactness loss evaluates the average similarity between the constituent samples of a given batch. For a large enough batch, this quantity can be expected to represent average intra-class variance of a given class. It is desirable to select a smooth differentiable function as lC to facilitate back propagation. In our work, we define compactness loss based on the Euclidean distance. 紧凑性损失lC：所有训练阶段用到的数据，都是属于同一类别的。所以它们都共享相同的类别标签。紧凑性损失评估的是，一个给定的批次中的样本之间的相似性。对于一个足够大的批次，这个量就可以用于表示给定类别中的平均类内变化程度。选择一个平滑可微的函数作为lC，是很理想的，可以为反向传播提供方便。在我们的工作中，我们基于欧式距离定义紧凑性损失。

**Descriptiveness loss** (lD) : Descriptiveness loss evaluates the capacity of the learned feature to describe different concepts. We propose to quantify discriminative loss by the evaluation of cross-entropy with respect to the reference dataset (R). 描述性损失lD：描述性损失评估学习的特征的能力，以描述不同的概念。我们提出来通过评估相对于参考数据集R的交叉熵，来量化区分性损失。

For this discussion, we considered the AlexNet CNN architecture as the reference network. However, the discussed principles and procedures would also apply to any other CNN of choice. In what follows, we present the implementation details of the proposed method.

本文中，我们考虑使用AlexNet架构作为参考网络。但是，讨论的原则和过程应当也可以应用于其他的CNN选项。下面，我们给出提出方法的实现细节。

### 3.4. Architecture

The proposed training architecture is shown in Figure 5 (a) . The architecture consists of two CNNs, the reference network (R) and the secondary network (S) as introduced in the previous sub-section. Here, weights of the reference network and secondary network are tied across each corresponding counterparts. For example, weights between convi (where, i = 1, 2.., 5) layer of the two networks are tied together forcing them to be identical. All components, except Compactness loss, are standard CNN components. We denote the common feature extraction sub-architecture by g(⋅) and the common classification by sub-architecture by hc(⋅). Please refer to Appendix for more details on the architectures of the proposed method based on the AlexNet and VGG16 networks.

提出的训练架构如图5a所示。这个架构包含2个CNNs，参考网络R和第二网络S在前一节中已经介绍了。这里，参考网络和第二网络的权重与各自对应的部分绑定到一起。比如，两个网络的convi(i = 1, 2.., 5)层的权重是绑定到一起，迫使它们相等。所有组成部分，除了紧凑性损失，都是标准的CNN组成部分。我们将通用特征提取子架构表示为g(⋅)，通用分类子架构表示为hc(⋅)。附录中有提出的方法架构的更多的细节，基于AlexNet和VGG16。

### 3.5. Compactness loss

Compactness loss computes the mean squared intra-batch distance within a given batch. In principle, it is possible to select any distance measure for this purpose. In our work, we design compactness loss based on the Euclidean distance measure. Define X = {$x_1, x_2, ..., x_n$} ∈ $R^{n×k}$ to be the input to the loss function, where the batch size of the input is n.

紧凑性损失计算的是给定批次中的批次内均方根距离。原则上，可以选择任意的距离度量。在我们的工作中，我们基于欧式距离度量设计了紧凑性损失。定义X = {$x_1, x_2, ..., x_n$} ∈ $R^{n×k}$是损失函数的输入，其中输入批次的大小为n。

**Forward Pass**: For each ith sample $x_i ∈ R^k$, where 1 ≤ i ≤ n, the distance between the given sample and the rest of the samples z_i can be defined as, 对给定的第i个样本$x_i ∈ R^k$，其中1 ≤ i ≤ n，给定的样本与其他样本z_i之间的距离可以定义为

$$z_i = x_i − m_i$$(3)

where, $m_i = \frac {1}{n-1} \sum_{j\neq i} x_j$ is the mean of rest of the samples. Then, compactness loss lC is defined as the average Euclidean distance as in, 其中m_i是其余样本的均值。那么，紧凑性损失lC定义为平均欧式距离

$$l_C = \frac {1}{nk} \sum_{i=1}^n z_i^T z_i$$(4)

**Backpropagation**: In order to perform back-propagation using this loss, it is necessary to assess the contribution each element of the input has on the final loss. Let x_i = {x_i1, x_i2, . . . , x_ik}. Similarly, let m_i = {m_i1, m_i2, . . . , m_ik}. Then, the gradient of lb with respect to the input x_ij is given as, 为使用这个损失进行反向传播，必须要评估输入的每个元素对最终损失的影响。lb对输入x_ij的梯度为

$$\frac {∂l_C} {∂x_{ij}} = \frac {2}{(n-1)nk} [n×(x_{ij} − m_{ij}) - \sum_{k=1}^n (x_{ik} − m_{ik})]$$(5)

Detailed derivation of the back-propagation formula can be found in the Appendix. The loss lC calculated in this form is equal to the sample feature variance of the batch multiplied by a constant (see Appendix). Therefore, it is an inverse measure of the compactness of the feature distribution. 反向传播公式的详细推导见附录。以这个公式计算的损失lC，等于批次样本特征的方差乘以一个常数。因此，这是特征分布的紧凑性的一个逆度量。

### 3.6. Training

During the training phase, initially, both the reference network (R) and the secondary network (S) are initialized with the pre-trained model weights W0. Recall that except for the types of loss functions associated with the output, both these networks are identical in structure. Therefore, it is possible to initialize both networks with identical weights. During training, all layers of the network except for the last four layers are frozen as commonly done in network fine-tuning. In addition, the learning rate of the training process is set at a relatively lower value (5 × 10^−5 is used in experiments). During training, two image batches, each from the reference dataset and the target dataset are simultaneously fed into the input layers of the reference network and secondary network, respectively. At the end of the forward pass, the reference network generates a descriptiveness loss (lD), which is same as the cross-entropy loss by definition, and the secondary network generates compactness loss (lC). The composite loss (l) of the network is defined as,

在训练阶段，参考网络R和第二网络S在开始的时候都用预训练权重W0进行初始化。回忆一下，除了与输出相关的损失函数类型，这两个网络在结构上是一样的。因此，是可以用相同的权重来对两个网络进行初始化的。在训练的过程中，网络的所有层，除了最后4层，都是冻结的，这在网络精调时是很常见的。另外，训练过程的学习速率设为相对低的值（试验中使用5 × 10^−5）。在训练过程中，从参考数据集和目标数据集的两个图像批次，同时分别送入参考网络和第二网络的输入层。在前向过程的最后，参考网络生成了一个描述性损失(lD)，其在定义上就和交叉熵损失一样，第二网络生成紧凑性损失(lC)。网络的复合损失l定义为

$$l(r, t) = l_D(r|W) + λl_C(t|W)$$(6)

where λ is a constant. It should be noted that, minimizing this loss function leads to the minimization of the optimization objective in (2).

其中λ是常数。应当注意，损失函数的最小化，得到(2)中优化目标的最小化。

In our experiments, λ is set equal to 0.1. Based on the composite loss, the network is back-propagated and the network parameters are learned using gradient descent or a suitable variant. Training is carried out until composite loss l(r,t) converges. A sample variation of training loss is shown in Figure 6. In general, it was observed that composite loss converged in around two epochs (here, epochs are defined based on the size of the target dataset).

在我们的实验中，λ设为0.1。基于复合损失，网络进行反向传播，网络参数使用梯度下降进行学习。训练一直进行，直到复合损失l(r,t)收敛。训练损失的变化如图6所示。一般来说，可以观察得到，复合损失在两轮左右就可以收敛（这里，epochs是按照目标数据集的大小定义的）。

Intuitively, the two terms of the loss function lD and lC measure two aspects of features that are useful for one-class learning. Cross entropy loss values obtained in calculating descriptiveness loss lD measures the ability of the learned feature to describe different concepts with respect to the reference dataset. Having reasonably good performance in the reference dataset implies that the learned features are discriminative in that domain. Therefore, they are likely to be descriptive in general. On the other hand, compactness loss (lC) measures how compact the class under consideration is in the learned feature space. The weight λ governs the mutual importance placed on each requirement.

直觉上来说，损失函数的两项lD和lC度量的是特征对一类学习有用的两个方面。计算描述性损失lD得到的交叉熵的值，度量的是学习的特征描述参考数据集中不同的概念的能力。在参考数据集中有很好的性能，说明学习到的特征在那个领域中是有区分性的。因此，一般来说，它们是具有描述性的。另一方面，紧凑性损失lC度量的是，要考虑的类别在学习的特征空间中的紧凑程度。权重λ代表在每个要求上的相互重要性。

If λ is made large, it implies that the descriptiveness of the feature is not as important as the compactness. However, this is not a recommended policy for one-class learning as doing so would result in trivial features where the overlap between the given class and an alien class is significantly high. As an extreme case, if λ = 0 (this is equivalent to removing the reference network and carrying out training solely on the secondary network (Figure 2 (d)), the network will learn a trivial solution. In our experiments, we found that in this case the weights of the learned filters become zero thereby making output of any input equal to zero.

如果λ比较大，这说明特征的描述性没有紧凑性重要。但是，对一类学习来说，这并不推荐，因为这样做会使结果得到没有意义特征，给定的类别和外部类别的重叠非常高。作为例子，如果λ=0（这相当于去掉参考网络，只用第二网络进行训练，图2d），网络会学习得到无意义的解。在我们的实验中，我们发现，这种情况下，学习到的滤波器的权重变成了哦0，因此使得任何输入的输出都等于0。

Therefore, for practical one-class learning problems, both reference and secondary networks should be present and more prominence should be given to the loss of the reference network.

因此，对于实际上的一类学习问题，参考网络和第二网络都应当存在，而且应当给参考网络更大的重要性。

### 3.7. Testing

The proposed testing procedure involves two phases - template generation and matching. For both phases, secondary network with weights learned during training is used as shown in Figure 5 (b). During both phases, the excitation map of the feature extraction sub-network is used as the feature. For example, layer 7, fc7 can be considered from a AlexNet-based network. First, during the template generation phase a small set of samples v = {v1, v2, . . . , vn} are drawn from the target (i.e. training) dataset where v ∈ t. Then, based on the drawn samples a set of features g(v1), g(v2), . . . , g(vn) are extracted. These extracted features are stored as templates and will be used in the matching phase.

测试过程包含两个阶段 - 模板生成和匹配。在这两个阶段中，都使用训练过程中学习得到权重的第二网络，如图5b所示。在这两个阶段中，特征提取子网络的激发图用作特征。比如，第7层，fc7可以认为是基于AlexNet的网络。第一，在模板生成阶段，从目标（即，训练）数据集中取出一个样本集合v = {v1, v2, . . . , vn}，v ∈ t。然后，基于取出的样本，提取出一个特征集合g(v1), g(v2), . . . , g(vn)。提取出来的特征存储为模板，会在匹配阶段进行使用。

Based on stored template, a suitable one-class classifier, such as one-class SVM [43], SVDD [45] or k-nearest neighbor, can be trained on the templates. In this work, we choose the simple k-nearest neighbor classifier described below. When a test image y is present, the corresponding deep feature g(y) is extracted using the described method. Here, given a set of templates, a matched score Sy is assigned to y as

基于存储的模板，在模板上训练一个合适的一类分类器，如一类SVM，SVDD或k近邻。本文中，我们选择下述的简单k近邻分类器。当存在测试图像y时，用描述的方法提取对应的深度特征g(y)。这里，给定模板集，给y指定一个匹配分数Sy

$$S_y = f(g(y)|g(t_1), g(t_2), . . . , g(t_n))$$(7)

where f(⋅) is a matching function. This matching function can be a simple rule such as the cosine distance or it could be a more complicated function such as Mahalanobis distance. In our experiments, we used Euclidean distance as the matching function. After evaluating the matched score, y can be classified as follows,

其中f(⋅)是一个匹配函数。匹配函数可以是一个简单的规则，比如cosine距离，或者可以是一个更复杂的函数，比如Mahalanobis距离。在我们的实验中，我们使用欧式距离作为匹配函数。在计算匹配分数后，y可以按照下式进行分类

$$class(y) = 1, if S_y≤δ; 0, if S_y>δ$$(8)

where 1 is the class identity of the class under consideration and 0 is the identity of other classes and δ is a threshold. 其中1是在考虑的类别的类别标记，0是其他类别的标记，δ是一个阈值。

### 3.8. Memory Efficient Implementation

Due to shared weights between the reference network and the secondary network, the amount of memory required to store the network is nearly twice as the number of parameters. It is not possible to take advantage of this fact with deep frameworks with static network architectures (such as caffe [18]). However, when frameworks that support dynamic network structures are used (e.g. PyTorch), implementation can be altered to reduce memory consumption of the network.

由于参考网络和第二网络之间权重共享，所以存储网络所需的内存接近是参数数量的两倍。用静态网络架构的深度框架（如Caffe），不太可能利用这个事实。但是，当框架支持动态网络架构时（如PyTorch），可以换一种实现，来降低网络的内存消耗。

In the alternative implementation, only a single core network with functions g and hc is used. Two loss functions lC and lD branch out from the core network. However in this setup, descriptiveness loss (lD) is scaled by a factor of 1 − λ. In this formulation, first λ is made equal to 0 and a data batch from the reference dataset is fed into the network. Corresponding loss is calculated and resulting gradients are calculated using back-propagation. Then, λ is made equal to 1 and a data batch from the target dataset is fed into the network. Gradients are recorded same as before after back-propagation. Finally, the average gradient is calculated using two recorded gradient values, and network parameters are updated accordingly. In principle, despite of having a lower memory requirement, learning behavior in the alternative implementation would be identical to the original formulation.

在另一种实现中，只使用单个核心网络g和hc。两个损失函数lC和lD从核心网络中产生分支。但是，在这个设置中描述性损失lD乘以了一个系数1 − λ。在这个公式中，λ开始设为0，然后参考数据集中的数据批次送入网络。计算对应的损失，得到的梯度使用反向传播进行计算。然后，λ设为1，然后目标数据集的数据批次送入网络。梯度也像以前一样，在反向传播后进行记录。最后，用两个记录得到的梯度值，计算平均梯度，网络参数也相应的进行更新。原则上，尽管内存消耗降低了，这种实现的学习行为与原始公式中应当是一样的。

## 4. Experimental Results

In order to assess the effectiveness of the proposed method, we consider three one-class classification tasks: abnormal image detection, single class image novelty detection and active authentication. We evaluate the performance of the proposed method in all three cases against state of the art methods using publicly available datasets. Further, we provide two additional CNN-based baseline comparisons.

为评估提出的方法的有效性，我们考虑三个一类分类任务：非正常图像检测，单类图像新异类检测，和主动认证。我们将提出的方法在所有三种情况中与目前最好的方法，使用公开可用的数据集进行对比评估。而且，我们还给出了另外两种基于CNN的基准比较。

### 4.1. Experimental Setup

Unless otherwise specified, we used 50% of the data for training and the remaining data samples for testing. In all cases, 40 samples were taken at random from the training set to generate templates. In datasets with multiple classes, testing was done by treating one class at a time as the positive class. Objects of all the other classes were considered to be alien. During testing, alien object set was randomly sampled to arrive at equal number of samples as the positive class. As for the reference dataset, we used the validation set of the ImageNet dataset for all tests. When there was an object class overlap between the target dataset and the reference dataset, the corresponding overlapping classes were removed from the reference dataset. For example, when novelty detection was performed based on the Caltech 256, classes appearing in both Caltech 256 and ImageNet were removed from the ImageNet dataset prior to training.

除非另外说明，我们使用50%的数据进行训练，剩下的数据进行测试。在所有情况下，都从训练集随机取40个样本，以生成模板。在多类的数据集中，每次将一类作为正类，进行测试，所有其他类都当做外部类别。在测试时，外部目标集都是随机进行取样，达到与正类相同数量的样本。至于参考数据集，我们使用ImageNet的验证集。在目标数据集和参考数据集中有目标类别的重叠时，从参考数据集中将对应的重叠类别去掉。比如，在基于Caltech 256进行新异类检测时，在Caltech 256和ImageNet中都出现的类别就从ImageNet中去掉，然后再进行训练。

The Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) Curve are used to measure the performance of different methods. The reported performance figures in this paper are the average AUC figures obtained by considering multiple classes available in the dataset. In all of our experiments, Euclidean distance was used to evaluate the similarity between a test image and the stored templates. In all experiments, the performance of the proposed method was evaluated based on both the AlexNet [22] and the VGG16 [44] architectures. In all experimental tasks, the following experiments were conducted.

ROC曲线的AUC用于度量不同方法的性能。本文中给出的性能数字，是在考虑数据集中可用的多个类别的平均AUC数字。在我们所有的实验中，测试图像和存储的模板之间的相似度，是用欧式距离进行计算的。在所有的实验中，提出的方法的性能是基于AlexNet和VGG16架构进行计算的。在所有的实验任务中，进行了下列实验。

**AlexNet Features and VGG16 Features (Baseline)**. One-class classification is performed using k-nearest neighbor, One-class SVM [43], Isolation Forest [3] and Gaussian Mixture Model [3] classifiers on fc7 AlexNet features and the fc7 VGG16 features, respectively. 对AlexNet和VGG16的fc7特征，分别使用k近邻，一类SVM，Isolation森林，高斯混合模型分类器进行一类分类。

**AlexNet Binary and VGG16 Binary (Baseline)**. A binary CNN is trained by having ImageNet samples and one-class image samples as the two classes using AlexNet and VGG16 architectures, respectively. Testing is performed using k-nearest neighbor, One-class SVM [43], Isolation Forest [3] and Gaussian Mixture Model [3] classifiers. 用AlexNet和VGG16，将ImageNet和一类图像样本作为两类，分别训练了一个二值CNN。测试时使用了k近邻，一类SVM，Isolation森林和高斯混合模型分类器。

**One-class Neural Network (OCNN)**. Method proposed in [5] applied on the extracted features from the AlexNet and VGG16 networks. 从AlexNet和VGG16网络中提取的特征，使用[5]中的方法进行应用。

**Autoencoder** [15]. Network architecture proposed in [15] is used to learn a representation of the data. Reconstruction loss is used to perform verification. 用[15]中的网络架构来学习数据的一个表示。用重建损失来进行验证。

**Ours (AlexNet) and ours (VGG16)**. Proposed method applied with AlexNet and VGG16 network backbone architectures. The fc7 features are used during testing. 将AlexNet和VGG16的骨干网络架构用于我们提出的方法中。fc7特征在测试时进行使用。

In addition to these baselines, in each experiment we report the performance of other task specific methods. 除了上述这些基准，在每个试验中，我们给出其他任务专有的方法。

### 4.2. Results

**Abnormal Image Detection**: The goal in abnormal image detection is to detect abnormal images when the classifier is trained using a set of normal images of the corresponding class. Since the nature of abnormality is unknown a priori, training is carried out using a single class (images belonging to the normal class). The 1001 Abnormal Objects Dataset [41] contains 1001 abnormal images belonging to six classes which are originally found in the PASCAL [12] dataset. Six classes considered in the dataset are Airplane, Boat, Car, Chair, Motorbike and Sofa. Each class has at least one hundred abnormal images in the dataset. A sample set of abnormal images and the corresponding normal images in the PASCAL dataset are show in Figure 7(a). Abnormality of images has been judged based on human responses received on the Amazon Mechanical Turk. We compare the performance of abnormal detection of the proposed framework with conventional CNN schemes and with the comparisons presented in [41]. It should be noted that our testing procedure is consistent with the protocol used in [41].

**非正常图像检测**：非正常图像检测的目标是，使用对应类别的正常图像集合来训练一个分类器，来检测非正常图像。由于非正常的本质并不是先验，训练是使用单个类别来进行的（属于正常类别的图像）。1001非正常目标数据集[41]包含1001幅非正常图像，属于6个PASCAL数据集中的类别，这6个类别是飞机，船，车，椅子，摩托车和沙发。数据集中每个类别至少有100张非正常的图像。非正常图像的样本集，和PASCAL数据集中对应的正常图像，如图7a所示。图像的非正常性是基于AMT上的人类响应的评判的。我们将提出的方法的非正常检测性能，与传统CNN方法和[41]中的方法进行了比较。应当指出，我们的测试过程与[41]中提出的方案是一致的。

Results corresponding to this experiment are shown in Table I. Adjusted graphical model presented in [41] has outperformed methods based on traditional deep features. The introduction of the proposed framework has improved the performance in AlexNet almost by a margin of 14%. Proposed method based on VGG produces the best performance on this dataset by introducing a 4.5% of an improvement as compared with the Adjusted Graphical Method proposed in [41].

这个实验的结果如表1所示。[41]提出的Adjusted graphical model超过了基于传统深度特征的方法。提出的框架将AlexNet的性能提高了接近14%。基于VGG的提出的方法在这个数据集上得到了最好的性能，与[41]提出的Adjusted graphical model相比改进了4.5%。

**One-Class Novelty Detection**: In one-class novelty detection, the goal is to assess the novelty of a new sample based on previously observed samples. Since novel examples do not exist prior to test time, training is carried out using one-class learning principles. In the previous works [4], [11], the performance of novelty detection has been assessed based on different classes of the ImageNet and the Caltech 256 datasets. Since all CNNs used in our work have been trained using the ImageNet dataset, we use the Caltech 256 dataset to evaluate the performance of one-class novelty detection. The Caltech 256 dataset contains images belonging to 256 classes with total of 30607 images. In our experiments, each single class was considered separately and all other classes were considered as alien. Sample images belonging to three classes in the dataset are shown in Figure 7 (b). First, consistent with the protocol described in [11], AUC of 20 random repetition were evaluated by considering the American Flag class as the known class and by considering boom-box, bulldozer and cannon classes as alien classes. Results corresponding to different methods are tabulated in Table II.

**一类新异类检测**：在一类新异类检测中，目标是基于之前观察的样本，来评估新样本的新颖程度。由于在测试之前新样本并不存在，训练是使用一类学习的原则进行的。在之前的工作中[4,11]，新异类检测的性能，是基于ImageNet和Caltech 256数据集的不同类别来进行评估的。由于在我们的工作中所有的CNN都是用ImageNet数据集训练过的，我们使用Caltech 256数据集来评估一类新异类检测的性能。Caltech 256数据集包含256类图像，共计30607幅图像。在我们的试验中，每个类别单独进行考虑，其他所有类别都认为是外部类别。图7b所示的是数据集中三类的样本图像。第一，与[11]中描述的原则一致，认为美国国旗的类别是已知类别，boom-box，bulldozer和cannon类别是外部类别，计算20个随机重复的AUC。不同方法的对应结果如表2所示。

In order to evaluate the robustness of our method, we carried out an additional test involving all classes of the Caltech 256 dataset. In this test, first a single class is chosen to be the enrolled class. Then, the effectiveness of the learned classifier was evaluated by considering samples from all other 255 classes. We did 40 iterations of the same experiment by considering first 40 classes of the Caltech 256 dataset one at a time as the enrolled class. Since there are 255 alien classes in this test as opposed to the first test, where there were only three alien classes, performance is expected to be lower than in the former. Results of this experiment are tabulated in Table III.

为评估我们方法的稳健性，我们进行了额外的测试，涉及到Caltech 256数据集的所有类别。在这个测试中，首先选择了单个类别为登记的类别。然后，考虑所有其他255个类别的样本，来评估学习的分类器的有效性。我们将同样的实验迭代了40次，考虑Caltech 256数据集的前40个类别，每次一个类别，作为登记的类别。由于在这个测试中有255个外部类别，而前面的测试只有3个外部类别，因此本次的性能期待应当比前者要低。这个实验的结果如表3所示。

It is evident from the results in Table II that a significant improvement is obtained in the proposed method compared to previously proposed methods. However, as shown in Table III this performance is not limited just to a American Flag. Approximately the same level of performance is seen across all classes in the Caltech 256 dataset. Proposed method has improved the performance of AlexNet by nearly 13% where as the improvement the proposed method has had on VGG16 is around 9%. It is interesting to note that binary CNN classifier based on the VGG framework has recorded performance very close to the proposed method in this task (difference in performance is about 1%). This is due to the fact that both ImageNet and Caltech 256 databases contain similar object classes. Therefore, in this particular case, ImageNet samples are a good representative of novel object classes present in Caltech 256. As a result of this special situation, binary CNN is able to produce results on par with the proposed method. However, this result does not hold true in general as evident from other two experiments.

从表2的结果中可以看到，很明显，我们提出的方法比之前提出的方法有明显的改进。但是，从表3中也可以看到，这个性能也并不是局限在美国国旗这个类别上的。在Caltech 256数据集上，在所有类别上都可以看到接近一样级别的性能。提出的方法对AlexNet的性能改进了接近13%，对VGG的改进大约是9%。很有趣要说明的是，基于VGG的二值CNN分类器与本文提出的方法在这个任务上性能非常接近（差异小于1%）。这是因为，ImageNet和Caltech 256数据集包含的都是类似的目标类别。因此，在这个特殊的情况中，ImageNet样本是Caltech 256中新异类目标类别的很好的代表。这个特殊情况的结果是，二值CNN可以得到与本文提出的方法类似的结果。但是，这个结果一般并不成立，从其他两个试验可以很明显看到。

**Active Authentication (AA)**: In the final set of tests, we evaluate the performance of different methods on the UMDAA-02 mobile AA dataset [24]. The UMDAA-02 dataset contains multi-modal sensor observations captured over a span of two months from 48 users for the problem of continuous authentication. In this experiment, we only use the face images of users collected by the front-facing camera of the mobile device. The UMDAA-02 dataset is a highly challenging dataset with large amount of intra-class variation including pose, illumination and appearance variations. Sample images from the UMDAA-02 dataset are shown in Figure 7 (c). As a result of these high degrees of variations, in some cases the inter-class distance between different classes seem to be comparatively lower making recognition challenging.

**主动认证(AA)**：在最后一个测试中，我们在UMDAA-02 mobile AA数据集上评估不同方法的性能。UMDAA-02数据集包含多模态传感器数据，是从48个用户中持续时间超过2个月收集的，为解决连续验证的问题。在这个实验中，我们只使用移动设备的前置摄像头拍摄的用户面部图像。UMDAA-02数据集是一个高度有挑战性的数据集，类内变化很大，包括姿态，光照和外貌变化。UMDAA-02中的样本图像如图7c所示。变化很大的结果是，在一些情况中，不同类别间的类间距离似乎相对更低，这使得识别非常有挑战性。

During testing, we considered first 13 users taking one user at a time to represent the enrolled class where all the other users were considered to be alien. The performance of different methods on this dataset is tabulated in Table IV.

在测试时，我们考虑前13个用户，一次一个，来表示登记的类别，其他所有用户都认为是外部类别。不同方法在这个数据集上的性能如表4所示。

Recognition results are comparatively lower for this task compared to the other tasks considered in this paper. This is both due to the nature of the application and the dataset. However, similar to the other cases, there is a significant performance improvement in proposed method compared to the conventional CNN-based methods. In the case of AlexNet, improvement induced by the proposed method is nearly 8% whereas it is around 6% for VGG16. The best performance is obtained by the proposed method based on the VGG16 network.

与本文中其他任务相比，这个任务的识别结果相对较低。这是因为应用和数据集的本质。但是，与其他情况类似，提出的方法与传统基于CNN的方法相比，还是有明显的性能改进的。在AlexNet的情况下，提出的方法带来的改进是接近8%，对VGG16是大约6%。提出的方法基于VGG16网络，得到了最好的性能。

### 4.3 Discussion

**Analysis on mis-classifications**: The proposed method produces better separation between the class under consideration and alien samples as presented in the results section. However, it is interesting to investigate on what conditions the proposed method fails. Shown in Figure 8 are a few cases where the proposed method produced erroneous detections for the problem of one-class novelty detection with respect to the American Flag class (in this experiment, all other classes in Caltech256 dataset were used as alien classes). Here, detection threshold has been selected as δ = 0. Mean detection scores for American Flag and alien images were 0.0398 and 8.8884, respectively.

**误分类的分析**：提出的方法可以在考虑的类别和外部样本之间给出更好的区分，在结果部分也看到了。但是，研究一下在什么条件下提出的方法会失败，这是很有趣的。图8给出了，提出的方法给出了错误检测的几种情况，其问题是美国国旗类别的一类新异类检测（在这个实验中，Caltech 256数据集中的所有其他类别都被用于外部类别）。这里，检测阈值选择为δ = 0。美国国旗和外部图像的平均检测分数分别为0.0398和8.8884。

As can be see from Figure 8, in majority of false negative cases, the American Flag either appears in the background of the image or it is too closer to clearly identify its characteristics. On the other hand, false positive images either predominantly have American flag colors or the texture of a waving flag. It should be noted that the nature of mis-classifications obtained in this experiment are very similar to that of multi-class CNN-based classification.

从图8中可以看到，在假阴性的主要情况中，美国国旗要么是在图像背景中，要么太接近不能识别出其特征。另一方面，假阳性图像都有美国国旗的颜色或旗帜的纹理。应当指出，在这个实验中得到的误分类的本质，与基于CNN的多类分类非常类似。

**Using a subset of the reference dataset**: In practice, the reference dataset is often enormous in size. For example, the ImageNet dataset has in excess of one million images. Therefore, using the whole reference dataset for transfer learning may be inconvenient. Due to the low number of training iterations required, it is possible to use a subset of the original reference dataset in place of the reference dataset without causing over-fitting. In our experiments, training of the reference network was done using the validation set of the ImageNet dataset. Recall that initially, both networks are loaded with pre-trained models. It should be noted that these pre-trained models have to be trained using the whole reference dataset. Otherwise, the resulting network will have poor generalization properties.

**使用参考数据集的子集**：在实践中，参考数据集通常非常大。比如，ImageNet数据集超过100万幅图像。因此，使用整个参考数据集进行迁移学习可能不方便。由于训练迭代次数不多，可以使用原始参考数据集的子集来替代参考数据集，也不会造成过拟合。在我们的实验中，参考网络的训练是使用ImageNet数据集的验证集来进行的。回忆一下，两个网络开始的时候都用预训练模型进行了初始化。应当注意，这些预训练的模型要使用整个参考数据集进行训练，否则，得到的网络的泛化性质会比较差。

**Number of training iterations**: In an event when only a subset of the original reference dataset is used, the training process should be closely monitored. It is best if training can be terminated as soon as the composite loss converges. Training the network long after composite loss has converged could result in inferior features due to over-fitting. This is the trade-off of using a subset of the reference dataset. In our experiments, convergence occurred around 2 epochs for all test cases (Figure 6). We used a fixed number of iterations (700) for each dataset in our experiments.

**训练迭代的数量**：当只使用原始参考数据集的子集时，训练过程应当进行密切监控。最好复合损失收敛时，就立刻停止。复合损失收敛后再训练很长时间，可能会由于过拟合，导致特征不太好。这是使用参考数据集的子集的代价。在我们的实验中，在所有测试情况下，2轮后就收敛了（图6）。在我们的试验中，对每个数据集，我们使用的迭代数量固定(700)。

**Effect of number of templates**: In all conducted experiments, we fixed the number of templates used for recognition to 40. In order to analyze the effect of template size on the performance of our method, we conducted an experiment by varying the template size. We considered two cases: first, the novelty detection problem related to the American Flag (all other classes in Caltech256 dataset were used as alien classes), where the recognition rates were very high at 98%; secondly, the AA problem where the recognition results were modest. We considered Ph01USER002 from the UMDAA-02 dataset for the study on AA. We carried out twenty iterations of testing for each case. The obtained results are tabulated in Table V.

**模板数量的效果**：在所有进行的试验中，我们用作识别的模板的数量固定为40。为分析模板数量对性能的影响，我们改变模板数量以进行试验。我们考虑两种情况：第一，美国国旗类的新异类检测问题（Caltech 256中的所有其他类别都用作外部类别），其识别率非常高98%；第二，识别结果中等的AA问题。我们在AA的研究中，考虑UMDAA-02数据集中的Ph01USER002。我们对每个情况，考虑20次迭代的测试。结果如表5所示。

According to the results in Table V, it appears that when the proposed method is able to isolate a class sufficiently, as in the case of novelty detection, the choice of the number of templates is not important. Note that even a single template can generate significantly accurate results. However, this is not the case for AA. Reported relatively lower AUC values in testing suggests that all faces of different users lie in a smaller subspace. In such a scenario, using more templates have generated better AUC values.

根据表5的结果，当提出的方法能够很充分的孤立一个类别时，就像新异类检测的情况，模板数量的选择似乎并不重要。注意，一个模板就可以生成非常好的结果了。但是，对于AA就不是这个情况。在测试中的AUC值相对较低，说明不同用户的脸部都在一个很小的子空间中。。在这种情况下，使用更多的模板可以得到更好的AUC值。

### 4.4. Impact of Different Features

In this subsection, we investigate the impact of different choices of hc and g has on the recognition performance. Feature was varied from fc6 to fc8 and the performance of the abnormality detection task was evaluated. When fc6 was used as the feature, the sub-network g consisted layers from conv1 to fc6, where layers fc7 and fc8 were associated with the sub network hc. Similarly, when the layer fc7 was considered as the feature, the sub-networks g and hc consisted of layers conv1 − fc7 and fc8, respectively.

在这个小节中，我们研究了hc和g的不同选择，对识别性能的影响。我们选择了从fc6到fc8的不同特征，评估了非正常检测的任务。当fc6用作特征时，子网络g是从conv1到fc6的层，而fc7和fc8是子网络hc的层。类似的，当fc7选为特征时，子网络g和hc由于conv1到fc7或fc8的层构成。

In Table VI, the recognition performance on abnormality image detection task is tabulated for different choices of hc and g. From Table VI we see that in both AlexNet and VGG16 architectures, extracting features at a later layer has yielded in better performance in general. For example, for VGG16 extracting features from fc6, fc7 and fc8 layers has yielded AUC of 0.856, 0.956 and 0.969, respectively. This observation is not surprising on two accounts. First, it is well-known that later layers of deep networks result in better generalization. Secondly, Compactness Loss is minimized with respect to features of the target dataset extracted in the fc8 layer. Therefore, it is expected that fc8 layer provides better compactness in the target dataset.

在表6中，在非正常图像检测任务中，对hc和g的不同选择，列出了不同的识别性能。从表6中可以看出，在AlexNet和VGG架构中，在更后面的层提取特征，总体上会得到更好的性能。比如，对于VGG16来说，从fc6，fc7，fc8提取特征，分别可以得到0.856, 0.956和0.969的AUC值。这个观察并不令人惊讶。第一，众所周知，深度网络中更后面的层有着更好的泛化性能。第二，紧凑性损失是对目标数据集在fc8层提取的特征进行最小化的。因此，fc8层在目标数据集中获得更好的紧凑性，这是正常的。

### 4.5. Impact of the Reference Dataset

The proposed method utilizes a reference dataset to ensure that the learned feature is informative by minimizing the descriptiveness loss. For this scheme to result in effective features, the reference dataset has to be a non-trivial multi-class object dataset. In this subsection, we investigate the impact of the reference dataset on the recognition performance. In particular, abnormal image detection experiment on the Abnormal 1001 dataset was repeated with a different choice of the reference dataset. In this experiment ILSVRC12 [9], Places365 [49] and Oxford Flowers 102 [27] datasets were used as the reference dataset. We used publicly available pre-trained networks from caffe model zoo [18] in our evaluations.

提出的方法利用参考数据集，通过最小化描述性损失，来确保学习的特征是可以提供有用信息的。为了这种方法能够得到有效的特征，参考数据集需要是有意义的多类别目标数据集。在这个小节中，我们研究了参考数据集对识别性能的影响。具体的，选择了不同的参考数据集，在Abnormal 1001数据集上重复了非正常图像检测试验。在这个实验中，ILSVRC12 [9], Places365 [49] 和 Oxford Flowers 102 [27]用作参考数据集。我们使用公开可用的caffe模型库中的预训练网络进行评估。

In Table VII the recognition performance for the proposed method as well as the baseline methods are tabulated for each considered dataset. From Table VII we observe that the recognition performance has dropped when a different reference dataset is used in the case of VGG16 architecture. Places365 has resulted in a drop of 0.038 whereas the Oxford flowers 102 dataset has resulted in a drop of 0.026. When the AlexNet architecture is used, a similar trend can be observed. Since Places365 has smaller number of classes than ILVRC12, it is reasonable to assume that the latter is more diverse in content. As a result, it has helped the network to learn more informative features. On the other hand, although Oxford flowers 102 has even fewer classes, it should be noted that it is a fine-grain classification dataset. As a result, it too has helped to learn more informative features compared to Places365. However, due to the presence of large number of non-trivial classes, the ILVRC12 dataset has yielded the best performance among the considered cases.

表7给出了选择不同的数据集时，提出的方法以及基准方法的识别性能。从表7中，我们看到，在使用VGG16时，使用其他参考数据集，会使得识别性能下降。使用Places365性能下降了0.038，使用Oxford flowers 102数据集性能下降了0.026。当使用AlexNet架构时，可以看到类似的趋势。由于Places365比ILSVRC12的类别数量更少，假设后者内容更加多样化这是很自然的。结果是，这可以帮助网络学习更有信息量的特征。另一方面，虽然Oxford flowers 102的类别数量更少，应当指出的是，这是一个细粒度分类数据集。结果是，与Places365相比，也学习到了更加具有信息量的特征。但是，由于存在大量有意义的类别，ILSVRC12数据集在所有数据集中得到了最好的性能。

## 5. Conclusion

We introduced a deep learning solution for the problem of one-class classification, where only training samples of a single class are available during training. We proposed a feature learning scheme that engineers class-specific features that are generically discriminative. To facilitate the learning process, we proposed two loss functions descriptiveness loss and compactness loss with a CNN network structure. Proposed network structure could be based on any CNN backbone of choice. The effectiveness of the proposed method is shown in results for AlexNet and VGG16-based backbone architectures. The performance of the proposed method is tested on publicly available datasets for abnormal image detection, novelty detection and face-based mobile active authentication. The proposed method obtained the state-of-the-art performance in each test case.

我们对一类学习提出了一种深度学习解决方法，其中只有一个类别的训练样本在训练时可用。我们提出了一个特征学习方案，设计出了具有区分能力的类别专用的特征。为方便学习过程，我们提出了两个损失函数，描述性损失和紧凑性损失，以及一个CNN网络架构。提出的网络结构可以基于任意CNN骨干网络。提出方法的有效性使用AlexNet和VGG16骨干架构得到了结果。提出的方法的性能在公开可用的数据集上进行了测试，包括非正常图像检测，新异类检测和基于人脸的移动主动认证。提出的方法在每个测试的情况中都得到了目前最好的性能。