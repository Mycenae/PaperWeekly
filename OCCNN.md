# One-Class Convolutional Neural Network

Poojan Oza, and Vishal M. Patel Johns Hopkins University

## 0. Abstract

We present a novel Convolutional Neural Network (CNN) based approach for one class classification. The idea is to use a zero centered Gaussian noise in the latent space as the pseudo-negative class and train the network using the cross-entropy loss to learn a good representation as well as the decision boundary for the given class. A key feature of the proposed approach is that any pre-trained CNN can be used as the base network for one class classification. The proposed One Class CNN (OC-CNN) is evaluated on the UMDAA-02 Face, Abnormality-1001, FounderType-200 datasets. These datasets are related to a variety of one class application problems such as user authentication, abnormality detection and novelty detection. Extensive experiments demonstrate that the proposed method achieves significant improvements over the recent state-of-the-art methods. The source code is available at: https://github.com/otkupjnoz/oc-cnn.

我们提出一种新的基于CNN的方法进行一类分类。其思想是使用在隐空间使用零均值高斯噪声作为伪负类，使用交叉熵损失来训练网络，为给定类别学习一个好的表示以及决策边界。提出的方法的一个关键特征是，任意预训练的CNN都可以用作一类分类的基础网络。提出的一类CNN(OCCNN)在UMDAA-02 Face, Abnormality-1001, FounderType-200数据集上进行评估。这些数据集与很多一类应用问题相关，比如用户验证，非正常检测和新异类检测。广泛的试验证明，提出的方法比最近最好的方法有显著的改进。代码已开源。

**Index Terms**—One Class Classification, Convolutional Neural Networks, Representation Learning.

## 1. Introduction

Multi-class classification entails classifying an unknown object sample into one of many pre-defined object categories. In contrast, in one-class classification, the objective is to identify objects of a particular class (also known as positive class data or target class data) among all possible objects by learning a classifier from a training set consisting of only the target class data. The absence of data from the negative class(es) makes the one-class classification problem difficult.

多类分类要将未知目标样本分类成很多预定义的目标类别中的一个。对比起来，在一类分类中，其目标是在所有可能的目标中识别特定类别的目标（也称为正类数据或目标类数据），但要在只有目标类别数据的训练集中学习一个分类器。负类数据的缺失，使一类分类问题非常困难。

One-class classification has many applications such as anomaly or abnormality detection [1], [2], [3], [4], [5], novelty detection [6], [7], [8], and user authentication [9], [10], [11], [12], [13], [14]. For example, in novelty detection, it is normally assumed that one does not have a priori knowledge of the novel class data. Hence, the learning process involves only the target class data.

一类分类有很多应用，比如异常检测，非正常检测，新异类检测，和用户验证。比如，在新异类检测中，一般假设，对新异类数据并没有先验知识。因此，学习过程只涉及到目标类别数据。

Various methods have been proposed in the literature for one-class classification. In particular, many one-class classification methods are based on the Support Vector Machines (SVM) formulation [15], [8], [16]. SVMs are based on the concept of finding a boundary that maximizes the margin between two classes and are shown to work well for binary and multi-class classification. However, in one-class problems the infromation regarding the negative class data is unavailable. To deal with this issue, Scholkopf et al. [17] proposed one-class SVM (OC-SVM), which tackles the absence of negative class data by maximizing the boundary with respect to the origin. Another popular approach inspired by the SVM formulation is Support Vector Data Description (SVDD) introduced by Tax et al. [18], in which a hypersphere that encloses the target class data is sought. Various extensions of OC-SVM and SVDD have been proposed in the literature over the years. We refer readers to [19] for a survey of different one-class classification methods. Another approach for one-class classification is based on the Minimax Probability Machines (MPM) formulation [20]. Single class MPM [21], [22] seeks to find a hyper-plane similar to that of OC-SVM by taking second order statistics of data into consideration. Hence, single class MPM learns a decision boundary that generalizes well to the underlying data distribution. Fig. 1 presents a high-level overview of different one-class classification methods. Though, these approaches are powerful tools in identifying the decision boundary for target data, their performance depends on the features used to represent the target class data.

文献中已经提出了各种方法进行异类分类。特别是，很多一类分类方法是基于SVM的。SVM是基于下面思想的，即找到一条边界，使两类之间的间隙最大化，对两类分类和多类分类效果都很好。但是，在一类问题中，负类数据的信息是不可用的。为处理这个问题，Scholkopf等[17]提出了一类SVM，通过最大化与原点之间的距离，来处理负类缺失的问题。另一种受到SVM启发的方法是SVDD，由Tax等[18]提出，它们寻找的是一个超球面，包围着目标类别。过去在文献中提出了OC-SVM和SVDD的各种拓展。我们推荐读者参考[19]的一类分类方法综述。另一种一类分类的方法是基于Minimax概率机(MPM)的。单类MPM通过考虑数据的二阶统计量，来寻找一个与OC-SVM类似的超平面。因此，一类MPM学习的决策面，对潜在的数据分布泛化的很好。图1给出了各种一类分类方法的高层概览。虽然，这些方法都是很好的工具，可以找到目标数据的决策边界，但其性能依赖于用来表示目标类别数据的特征。

Over the last five years, methods based on Deep Convolutional Neural Networks (DCNNs) have shown impressive performance improvements for object detection and recognition problems. Taking classification task as an example, the top-5 error rate of vision systems on the ImageNet dataset [23] has dropped from ∼ 25% to 2.25% in the last five years. This has been made possible due to the availability of large annotated datasets, a better understanding of the non-linear mapping between input images and class labels as well as the affordability of Graphics Processing Units (GPUs). These networks learn distinct features for a particular class against another using the cross entropy loss. However, for one class problems training such networks in an end-to-end manner becomes difficult due to the absence of negative class data.

在过去五年中，DCNN的方法在目标检测和识别问题中有显著的改进。以分类问题作为一个例子，视觉系统在ImageNet数据集中的top-5错误率，已经在过去五年中从大约25%降低到了2.25%。因为有大量可用的标注数据集，在输入图像和类别标签之间的非线性映射的更好理解，和GPUs的可用性，这才成为可能。这些网络学习使用交叉熵，到了特定类别的独特特征。但是，对于一类问题，以端到端的方式训练这种网络非常困难，因为缺少负类数据。

In recent years, several attempts have been made to counter the problem of training a neural network for one-class classification [5], [24], [25], [26], [27], [28], [29], [30]. These approaches can be broadly classified in to two categories, generative approaches [27], [28], [29] and discriminative approaches [25], [30]. Generative approaches use generative frameworks such as auto-encoders or Generative Adversarial Networks (GAN) [31] for one-class classification. For example, Ravanbakhsh et al. [27] and Sabokrou et al. [24] proposed deep auto-encoder networks for event anomaly detection in surveillance videos. However, in their approaches the focus is mainly on the image-level one-class classification. Work by Lawson et al. [26] developed a GAN-based approach for abnormality detection. Sabokrou et al. [5] extended that idea for detecting outliers from image data using an auto-encoder based generator with adversarial training. In general, these generative models such as GANs are very difficult to train as compared to the discriminative classification networks [32].

近年来，有一些解决训练一类分类神经网络的尝试。这些方法可以大致分为两类，生成式方法和判别式方法。生成式方法使用生成式框架，比如自动编码器或GAN进行一类分类。比如，Ravanbakhsh等[27]和Sabokrou等[24]提出了深度自动编码器网络，解决监控视频中的事件异常检测。但是，在他们的方法中，其焦点主要是在图像级的一类分类。Lawson等[26]提出了一种基于GAN的非正常检测方法。Sabokrou等[5]拓展了用基于自动编码器的生成器，和对抗训练的方法，来从图像数据中检测离群的思想。总体上，这些生成式模型比如GANs，与判别式分类网络相比，非常难以训练。

Compared to the generative approaches, discriminative approaches for one-class classification have not been well explored in the literature. One such approach by Perera and Patel [30] utilize an external reference dataset as the negative class to train a deep network for one-class classification using a novel loss function. In contrast to this method, we do not make use of any negative class data in our approach. In another approach, Chalapathy et al. [25] proposed a novel SVM inspired loss function to train a neural network for anomaly detection. With some inspirations from other statistical approaches for one-class classification (i.e. taking origin as a reference to find the decision boundary), we propose a novel method called, One-Class CNN (OC-CNN), to learn representations for one-class problems with CNNs trained end-to-end in a discriminative manner. This paper makes the following contributions:

与生成式方法相比，一类分类的判别式方法在文献中没有得到很好的探索。一种这样的方法Perera and Patel [30]利用外部参考数据集作为负类来训练一个深度网络进行一类分类，使用了一个新的损失函数。与这种方法相比，我们没有使用任何负类数据。在另一种方法中，Chalapathy等[25]提出了一种新的受到SVM启发的损失函数，来训练神经网络进行异常检测。受到其他统计方法进行一类分类的启发（即，以原点为参考来找到决策界面），我们提出了一种新的方法，称为一类CNN(OC-CNN)，来对一类问题用端到端的CNN以判别的方式来学习表示。本文贡献如下：

- A new approach is proposed based on CNN for one class classification which is end-to-end trainable. 提出了一种新的基于CNN的方法进行一类分类，是端到端可训练的。
  
- Through experiments, we show that proposed approach outperforms other statistical and deep learning-based one class classification methods and generalizes well across a variety of one class applications. 通过试验，我们证明了提出的方法超过了其他基于统计的和基于深度学习的一类分类方法，在很多一类分类应用中泛化非常好。

## 2. Proposed Approach

Fig. 2 gives an overview of the proposed CNN-based approach for one-class classification. The overall network consists of a feature extractor network and a classifier network. The feature extractor network essentially embeds the input target class images into a feature space. The extracted features are then appended with the pseudo-negative class data, generated from a zero centered Gaussian in the feature space. The appended features are then fed into a classification network which is characterized by a fully connected neural network. The classification network assigns a confidence score for each feature representation. The output of the classification network is either 1 or 0. Here, 1 corresponds to the data sample belonging to the target class and 0 corresponds to the data sample belonging to the negative class. The entire nework is trained end-to-end using binary cross-entropy loss.

图2给出了提出的基于CNN的一类分类方法的概览。总体网络由一个特征提取器网络和分类网络构成。特征提取器网络将输入的目标类别图像嵌入到一个特征空间中。提取的特征然后拼接到伪负类数据之后，后者是在特征空间中以一个零均值的高斯函数中生成的。拼接的特征然后送入分类网络中，一般是一个全连接网络。分类网络对每个特征表示指定了一个置信度分数。分类网络的输出是1或0。这里，1对应着属于目标类别的数据样本，0对应着属于负类的数据样本。整个网络是使用二值交叉熵损失端到端训练的。

### 2.1. Feature Extractor

Any pre-trained CNN can be used as the feature extractor. In this paper, we use the pre-trained AlexNet [33] and VGG16 [34] networks by removing the softmax regression layers (i.e. the last layer) from their networks. During training, we freeze the convolution layers and only train the fully-connected layers. Assuming that the extracted features are D-dimensional, the features are appended with the pseudo-negative data generated from a Gaussian, $N(\overline µ, σ^2 · I)$, where σ and $\overline µ$ are the parameters of the Gaussian and I is a D × D identity matrix. Here, $N(\overline µ, σ^2 · I)$ can be seen as generating D independent one dimensional gaussian with σ standard deviation.

任意预训练的CNN都可以用作特征提取器。本文中，我们使用预训练的AlexNet和VGG16，并将其softmax回归层（即，最后一层）去掉。在训练中，我们冻结卷积层，只训练全连接层。假设提取的特征是D维的，特征拼接到高斯函数生成的伪负类数据中，$N(\overline µ, σ^2 · I)$，其中σ和$\overline µ$是高斯函数的参数，I是D × D的单位矩阵。这里，$N(\overline µ, σ^2 · I)$可以视为独立的生成D个一维高斯函数，标准差为σ。

### 2.2. Classification Network

Due to the appending of the pseudo-negative data with the original features, the classifer network observes the input in the batch size of 2. A simple fully-connected layer followed by a softmax regression layer is used as the classifier network. The dimension of the fully-connected layer is kept the same as the feature dimension. The number of outputs from the softmax layer are set equal to two.

由于将原始特征与伪负类数据拼接到了一起，分类器网络观察到的输入的批次大小为2。简单的全连接层和softmax回归层用作分类网络。全连接层的维度与特征维度相同。softmax层的输出的数量设置为2。

### 2.3. Loss Function

The following binary cross-entropy loss function is used to train the entire network 下列的二值交叉熵损失函数用于训练整个网络

$$L_c = -\frac {1}{2K} \sum_{j=1}^{2K} (ylogp+(1-y)log(1-p))$$(1)

where, y ∈ {0, 1} indicates whether the classifier input corresponds to the feature extractor, (i.e. y = 0), or it is sampled from $N(\overline µ, σ^2 · I)$, (i.e. y = 1). Here, p denotes the softmax probability of y = 0. 其中，y ∈ {0, 1}指示了分类器输入是否对应着特征提取器（即y=0），还是从$N(\overline µ, σ^2 · I)$中取样的（即y=1）。这里，p表示y=0的softmax概率。

The network is optimized using the Adam optimizer [35] with learning rate of 10^−4. The input image batch size of 64 is used in our approach. For all experiments, the parameters $\overline µ$ and σ are set equal to 0 and 0.01, respectively. Instance normalization [36] is used before the classifier network as it was found to be very useful in stabilizing the training procedure.

网络使用Adam优化器进行优化，学习速率为10^-4。我们的方法中输入图像的批次大小为64.在所有试验中，参数$\overline µ$和σ分别设为0和0.01。在分类器网络之前使用了实例归一化，因为对于稳定训练过程非常有用。

## 3. Experimental Results

We evaluate the performance of the proposed approach on three different one-class classification problems - abnormality detection, face-based user authentication, and novelty detection. Abnormality-1001 [37], UMDAA-02 [38] and FounderType-200 [39] datasets are used to conduct experiments for the abnormality detection, user authentication and novelty detection problems. For all methods compared here, the data is aligned such that objects are at the center with minimal background.

我们在三个不同的一类分类问题上评估了提出的方法的性能，非正常检测，基于人脸的用户验证，和新异类检测，分别使用了Abnormality-1001 [37], UMDAA-02 [38]和FounderType-200 [39]数据集。对于所有比较的方法，数据都进行了对齐，这样目标都处于中间，背景最小。

The proposed approach is compared with following one-class classification methods: 提出的方法与下列一类分类问题进行了比较：

- OC-SVM: One-Class Support Vector Machine is used as formulated in [15], trained using the AlexNet and VGG16 features. 使用了一类SVM[15]，训练使用AlexNet和VGG16。
- BSVM: Binary SVM is used where the zero centered Gaussian noise is used as the negative data. AlexNet and VGG16 features extracted from the target class data are used as the positive class data. 二值SVM，负类数据使用的是零均值高斯噪声。AlexNet和VGG16用于从目标类别数据中提取数据，作为正类数据。
- MPM: MiniMax Probability Machines are used as formulated in [20]. Since, the MPM algorithm involves computing covariance matrix from the data, Principal component analysis (PCA) is used to reduce the dimensionality of the features before computing the covariance matrix. 由于MPM要从数据中计算协方差矩阵，用PCA来对特征进行降维，然后再计算协方差矩阵。
- SVDD: Support Vector Data Description is used as formulated in [18], trained on the AlexNet and VGG16 features.
- OC-NN: One-class neural network (OC-NN) is used as formulated in [25]. Here, for fair comparison, instead of using the feature extractor trained using an auto-encoder (as per [25] methodology), AlexNet and VGG16 networks, the same as the proposed method, are used. As described in [25], we evaluate OC-NN using three different activation functions - linear, Sigmoid and ReLU. 为公平进行比较，并没有使用[25]的自动编码器特征提取器，而是使用了我们方法中的AlexNet和VGG16网络。像[25]中一样，我们使用三种不同的激活函数来评估OC-NN，线性的，Sigmoid和ReLU。
- OC-CNN: One-class CNN is the method proposed in this paper.
- OC-SVM+: OCSVM+ is another method used in this paper where OC-SVM is utilized on top of the features extracted from the network trained using OC-CNN. However, since it uses OC-SVM for classification, it is not end-to-end trainable. OC-SVM+是本文中使用的另一种方法，其中在OC-CNN的网络提取的特征上，使用OC-SVM进行分类。但是，由于使用了OC-SVM进行分类，其并不是端到端可训练的。

### 3.1. Abnormality Detection

Abnormality detection (also referred as anomaly detection or outlier rejection) deals with identifying instances that are dissimilar to the target class instances (i.e. abnormal instances). Note that, the abnormal instances are not known a priori and only the normal instances are available during training. Such problem can be addressed by one-class classification algorithms. The Abnormality-1001 dataset [37] is widely used for visual abnormality detection. This dataset consists of 1001 abnormal images belonging to six classes such as Chair, Car, Airplane, Boat, Sofa and Motorbike which have their respective normal classes in the PASCAL VOC dataset [40]. Sample images from the Abnormality-1001 dataset are shown in Fig. 3 (a). Normal images obtained from the PASCAL VOC dataset are split into train and test sets such that the number of abnormal and normal images in test set are equal. Reported results are averaged for all six classes.

非正常检测（也称为异常检测或离群点拒绝）处理的是，识别与目标类别实例不相似的实例的问题。注意，非正常实例并不是先验已知的，在训练时只有正常实例是可用的。这样的问题可以通过一类分类算法进行解决。Abnormality-1001数据集[37]广泛的用于视觉非正常检测。这个数据集由1001幅非正常图像组成，属于6个类别，包括椅子，车，飞机，船，沙发和摩托车，在PASCAL VOC数据集中有其对应的正常类别。Abnormality-1001数据集中的图像例子如图3a所示。正常图像是从PASCAL VOC数据集中得到的，分成训练和测试集，这样测试集中的非正常和正常图像数量相等。得到的结果是在6个类别上进行平均的。

### 3.2. User Active Authentication

Active authentication refers to the problem of identifying the enrolled user based on his/her biometric data such as face, swipe patterns, and accelerometer patterns [13]. The problem can be viewed as identifying the abnormal user behaviour to reject the unauthorized user. The active authentication problem has been viewed as one-class classification problem [12]. The UMDAA-02 dataset [38] is widely used dataset for user active authentication on mobile devices. The UMDAA-02 dataset has multiple modalities corresponding to each user such as face, accelerometer, gyroscope, touch gestures, etc. Here, we only use the face data provided in this dataset since face is one of the most commonly used modality for authentication. The face data consists of 33209 face images corresponding to 48 users. Sample images corresponding to a few subjects from this dataset are shown in Fig. 3(b). As can be seen from this figure, the images contains large variations in pose, illumination, appearance, and occlusions. For each class, train and test sets are created by maintaining 80/20 ratio. Network is trained using the train set of a target user and tested on the test set of the target user against the rest of the user test set data. This process is repeated for all the users and average results are reported.

主动验证是指基于生物数据，如人脸，滑动样式和加速度计的样式来识别登记的用户。问题可以视为识别非正常用户行为，以拒绝未认证的用户。主动认证问题也被视为异类分类问题。UMDAA-02数据集广泛用于移动设备上的用于主动认证问题。UMDAA-02数据集有多种模态，比如人脸，加速器计，陀螺仪，接触姿态等。这里，我们只使用其中的人脸数据，因为这是最经常用于认证的数据模态。人脸数据由48个用户的33209幅人脸图像组成。图3b中是这个数据集中一些目标的图像样本。从这个图中可以看出，图像在姿态，光照，外观和遮挡上的变化很大。对于每个类别，训练集和测试集按照80/20的比例进行分割。网络训练使用一个目标用户的训练集，并在这个用户以及其他用户的测试数据上进行测试。这个过程对所有用户进行重复，给出平均的结果。

### 3.3. Novelty Detection

The FounderType-200 dataset was introduced for the purpose of novelty detection by Liu et al. in [39]. The FounderType-200 dataset, contains 6763 images from 200 different types of fonts created by the company FounderType. Fig. 3(c) shows some sample images from this dataset. For experiments, first 100 classes are used as the target classes and remaining 100 classes are used as the novel data. The first 100 class data are split into train and test set having equal number of images. For novel data, a novel set is created having 50 images from each of the novel classes. For each class, train set from the known data is used for training the network and known class test set and novel set data are used for evaluation. For example, class i (i ∈ {1, 2, .., 100}) train set is used for training the network. The trained network is then evaluated with class i test set tested against the novel set (containing data of class 101-200). This is repeated for all classes i where, i ∈ {1, 2, .., 100} and average results are reported.

FounderType-200数据集由Liu等[39]提出，进行新异类检测。FounderType-200数据集包含6763幅图像，分属200个不同的字体类型，由FouderType公司创建。图3c给出了这个数据集中的一些样本图像。对于试验，前100个类别用作目标类别，剩下的100个类别用作新异类数据。前100个类别的数据分为训练和测试集，数量相等。对于新异类数据，创建了一个新的集合，每个新异类都有50幅图像。对每个类别，训练使用已知数据的训练集，已知类别的测试集和新异类数据集用于评估。比如，类别i (i ∈ {1, 2, .., 100})的训练集用于训练网络。训练好的网络用类别i和新异类集合（由类别101-200的数据组成）的测试集进行测试。这对所有类别i进行重复，这里i ∈ {1, 2, .., 100}，并给出平均的结果。

## 4. Results and Discussion

The performance is measured using the area under the receiver operating characteristic (ROC) curve (AUROC), most commonly used metric for one-class problems. The results are tabulated in Table II and Table I corresponding to the VGG16 and AlexNet networks. AlexNet and VGG16 pretrained features are used to compute the results for OC-SVM, BSVM, SVDD and MPM. The OC-NN results are computed using the linear, sigmoid and relu activations after training on the target class data. The OC-CNN results are computed after training on the target class and for OC-SVM+, an one-class SVM is trained on top of the features extracted from the trained AlexNet/VGG16, and AUROC is computed from the SVM classifier scores.

性能用ROC曲线下的面积AUROC进行度量，这是一类问题最常用的度量。结果如表2和表1所示，对应VGG16和AlexNet网络。AlexNet和VGG16的预训练特征用于计算OC-SVM，BSVM，SVDD和MPM的结果。OC-NN的结果在目标类别数据上进行训练后，用线性、sigmoid和ReLU激活进行了计算。OC-CNN结果在目标类别上训练后计算得到的，对于OC-SVM+，在从训练好的AlexNet/VGG16提取了特征后，训练了一个一类SVM，对SVM的分类分数，计算了AUROC。

From the Tables I and II, it can be observed that either OC-CNN or OC-SVM+ achieves the best performance on all three datasets. MPM and OC-SVM achieve similar performances, while BSVM with Gaussian data as the negative class doesn't work as well. With the BSVM baseline, we show that similar trick we used for proposed algorithm doesn't work well for statistical approaches like SVM. Among the other one-class approaches, OC-NN with linear activation performs the best. However, OC-NN results are inconsistent. For couple of experiments, SVDD was found to be working better than OC-NN. The reason behind this inconsistent performance can be due to the differences in the evaluation protocol used for OC-NN in [25] and this paper. The ratio of the number of target class images to novel/abnormal class images in our evaluation protocol is much higher than the ratio used by Chalpathy et al. [25]. When the ratio is close to one, as is the case for Abnormality-1001 dataset, the OC-NN performs better than SVDD for both AlexNet and VGG16. However, when the ratio is increased (which is more realistic scenario), as is the case for UMDAA-02 and FounderType-200, the performance of OC-NN becomes inconsistent. Whereas, using the proposed approach performs consistently well, providing ∼4%, ∼10% and ∼5% improvements over OC-NN for Abnormality-1001, UMDAA02- Face and FounderType-200 datasets, respectively. Since, the proposed approach is built upon the traditional discriminative learning framework for deep neural networks, it is able to learn better features than OC-NN.

从表1和表2中可以看出，OC-CNN或OC-SVM+在所有三个数据集上获得了最好的性能。MPM和OC-SVM获得了类似的性能，而将高斯噪声作为负类的BSVM效果也不是很好。有了BSVM作为基准，我们展示了，对提取的方法我们使用类似的技巧，对统计类方法如SVM效果也不好。其他的一类方法中，线性激活的OC-NN效果最好。但是，OC-NN的结果并不一致。在几个试验中，SVDD的效果比OC-NN要好。这种性能不一致的原因，可能是OC-NN与本文使用的评估准则不一样。目标类别图像的数量，与新异类/非正常类别图像的比例，在我们的评估准则中，比Chalpathy等[25]中所用的比例要高的多。当这个比例接近于1，就像在Abnormality-1001数据集的情况中，OC-NN比SVDD效果要好，在AlexNet和VGG16两种情况中都是。但是，当这个比例增大（这是更加实际的场景），就像在UMDAA-02和FounderType-200的情况下，OC-NN的性能变得不一致起来。使用提出的方法的表现一致很好，在Abnormality-1001, UMDAA02-Face和FounderType-200上表现分别比OC-NN提升了∼4%, ∼10%和∼5%。由于提出的方法是基于传统的判别式学习框架中对深度神经网络构建起来的，其可以比OC-NN学到更好的特征。

Also as expected, methods based on the VGG16 network work better than the methods based on the AlexNet network. Apart from the FounderType-200 dataset where, OC-CNN with AlexNet works better than VGG16, for all methods VGG16 works better than AlexNet. However, it should be noted that better OC-SVM+ performance for VGG16 indicates that features learned with the proposed approach for VGG16 are better than AlexNet for FounderType-200. Overall, VGG16 gives ∼2% improvement over AlexNet.

基于VGG16网络的方法比基于AlexNet网络的方法效果要好，这符合期待。除了在FounderType-200中，AlexNet的OC-CNN比VGG16的要好，在其他所有试验中，VGG16都比AlexNet效果要好。但是，应当注意，对于OC-SVM+，采用VGG16的性能更好，说明，提出的方法用VGG16比用AlexNet在FounderType-200上学习到了更好的特征。总体上，VGG16比AlexNet有~2%的性能改进。

Another interesting comparison is between OC-SVM and OC-SVM+. OC-SVM uses features extracted from a pretrained AlexNet/VGG16 network. On the other hand, OC-SVM+ uses features extracted from AlexNet/VGG16 network trained using the proposed approach. OC-SVM+ performs ∼18% and ∼17% better than OC-SVM on average across all datasets for AlexNet and VGG16, respectively. This result shows the ability of our approach to learn better representations. So, apart from being an end-to-end learnable standalone system, our approach can also be used to extract target classfriendly features. Also, using sophisticated classifier has shown to improve the performance over OC-CNN (i.e., OC-SVM+) in majority of cases.

OC-SVM与OC-SVM+是另一个有趣的比较。OC-SVM使用的是从预训练的AlexNetVGG16提取的特征。而OC-SVM+使用的是提出的方法用AlexNet/VGG16提取出的特征。OC-SVM+比OC-SVM，在所有数据集上，对于AlexNet和VGG16，平均表现分别要好∼18%和∼17%。这个结果说明，我们的方法可以提取出更好的表示。所以，除了作为一个端到端可学习的系统，我们的方法还可以用于提取目标类别友好的特征。同时，使用复杂的分类器已经证明可以改进主要情况中的性能。

## 5. Conclusion

We proposed a new one-class classification method based on CNNs. A pseudo-negative Gaussian data was introduced in the feature space and the network was trained using a binary cross-entropy loss. Apart from being a standalone one-class classification system, the proposed method can also be viewed as good feature extractor for the target class data (i.e. OCSVM+) as well. Furthermore, the consistent performance improvements over all the datasets related to authentication, abnormality and novelty detection showed the ability of our method to work well on a variety of one-class classification applications. In this paper, experiments were performed over data with objects centrally aligned. In the future, we will explore the possibility of developing an end-to-end deep one class method that does joint detection and classification.

我们提出了一种新的基于CNNs的一类分类方法。在特征空间中引入了伪负高斯数据，网络训练使用二值交叉熵损失。除了作为独立的一类分类系统，提出的方法也可以视为目标类别数据的很好的特征提取器。而且，在所有数据集上的一致性能改进表明，我们的方法在很多一类分类应用中都效果很好。在本文中，试验是在将目标中间对齐的数据中进行的。未来，我们会探索提出端到端深度一类方法，并同时进行检测和分类的可能性。