# MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection

Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger

MVTec Software GmbH

## 0. Abstract

The detection of anomalous structures in natural image data is of utmost importance for numerous tasks in the field of computer vision. The development of methods for unsupervised anomaly detection requires data on which to train and evaluate new approaches and ideas. We introduce the MVTec Anomaly Detection (MVTec AD) dataset containing 5354 high-resolution color images of different object and texture categories. It contains normal, i.e., defect-free, images intended for training and images with anomalies intended for testing. The anomalies manifest themselves in the form of over 70 different types of defects such as scratches, dents, contaminations, and various structural changes. In addition, we provide pixel-precise ground truth regions for all anomalies. We also conduct a thorough evaluation of current state-of-the-art unsupervised anomaly detection methods based on deep architectures such as convolutional autoencoders, generative adversarial networks, and feature descriptors using pre-trained convolutional neural networks, as well as classical computer vision methods. This initial benchmark indicates that there is considerable room for improvement. To the best of our knowledge, this is the first comprehensive, multi-object, multi-defect dataset for anomaly detection that provides pixel-accurate ground truth regions and focuses on real-world applications.

在自然图像数据中检测异常结构，在计算机视觉中很多任务中是具有极端的重要性的。无监督异常检测方法的发展，需要训练和评估新方法和思想的数据。我们提出MVTec异常检测数据集，包含5354幅高分辨率彩色图像，包含不同的目标和纹理类别。数据集包含正常的图像，即无缺陷的图像，用于训练，以及有缺陷的图像用于测试。异常的形式超过70种，比如划痕，凹痕，污染，以及各种结构变化。另外，我们对所有缺陷还给出了逐像素的真值区域。我们还彻底评估了目前最好的基于深度架构的无监督缺陷检测方法，比如卷积自动编码器，生成式对抗网络，和使用预训练CNN的特征描述子，以及经典的计算机视觉方法。这个初始基准测试表明，要改进的空间还很大。据我们所知，这是第一个用于缺陷检测的，给出了精确到像素的真值区域，聚焦在真实世界的应用的综合多目标多缺陷数据集。

## 1. Introduction

Humans are very good at recognizing if an image is similar to what they have previously observed or if it is something novel or anomalous. So far, machine learning systems, however, seem to have difficulties with such tasks.

人类非常擅长于识别，图像与之前看到的图像是否类似，或是否有些新东西或异常的东西。但是，迄今为止，机器学习系统在这种任务上有一些困难。

There are many relevant applications that must rely on unsupervised algorithms that can detect anomalous regions. In the manufacturing industry, for example, optical inspection tasks often lack defective samples or it is unclear what kinds of defects may appear. In active learning systems, structures that are identified as anomalous might indicate the necessity of including a specific image for training. Therefore, it is not surprising that recently a significant amount of interest has been directed towards novelty detection in natural image data using modern machine learning architectures. A number of algorithms have been proposed that test whether a network is able to detect if new input data matches the distribution of the training data. Many of these algorithms, however, focus on classification settings in which the inlier and outlier distributions differ significantly. This is commonly known as outlier detection or one-class-classification. A common evaluation protocol is to arbitrarily label a number of classes from existing object classification datasets as outlier classes and use the remaining classes as inliers for training. It is then measured how well the trained algorithm can distinguish between previously unseen outlier and inlier samples.

有很多相关的应用，必须依赖于无监督算法，进行异常区域检测。比如，在制造业中，光学检查任务经常缺少缺陷样本，或不知道什么类型的缺陷会出现。在主动学习系统中，识别为异常的结构，可能表明，需要将特定的图像加入到训练中。因此，最近有大量关于在自然图像中使用现代机器学习架构进行异常检测的工作，这就很正常了。提出了很多算法，来测试是否一个网络可以检测新输入的数据与训练数据的分布是否匹配。但是，很多这种算法聚焦在分类的设置，其中正常值和异常值的分布差异非常大。这通常称为离群值检测，或一类分类。常用的评估准则是，从现有的目标分类数据集中任意标注一定数量的类别作为异常类别，使用剩下的类别作为正常类别，进行训练。然后看训练好的算法对之前没见过的正常和异常样本的区分程度，来对算法进行度量。

While this classification on an image level is important, it is unclear how current state-of-the-art methods perform on what we call anomaly detection tasks. The problem setting is to find novelties in images that are very close to the training data and differ only in subtle deviations in possibly very small, confined regions. Clearly, to develop machine learning models for such and other challenging scenarios we require suitable data. Curiously, there is a lack of comprehensive real-world datasets available for such scenarios.

这种在图像级别的分类是很重要的，但是目前最好的方法在异常检测任务中表现如何，尚不清楚。问题设置是，在图像中找到新东西，这些图像与训练数据非常接近，只在很小，非常局限的区域中与训练图像有很小的差异。很明显，对这样和其他有挑战的场景开发机器学习模型，我们需要合适的数据。但是现在对这种场景仍然缺少综合的真实世界数据集。

Large-scale datasets have led to incredible advances in many areas of computer vision in the last few years. Just consider how closely intertwined the development of new classification methods is with the introduction of datasets such as MNIST [16], CIFAR10 [14], or ImageNet [15].

在过去几年中，大规模数据集引领了计算机视觉很多领域的惊人进展。随着像MNIST、CIFAR10或ImageNet数据集的提出，新的分类方法也不断涌现。

To the best of our knowledge, no comparable dataset exists for the task of unsupervised anomaly detection. As a first step to fill this gap and to spark further research in the development of methods for unsupervised anomaly detection, we introduce the MVTec Anomaly Detection (MVTec AD or MAD for short) dataset that facilitates a thorough evaluation of such methods. We identify industrial inspection tasks as an ideal and challenging real-world use-case for these scenarios. Defect-free example images of objects or textures are used to train a model that must determine whether an anomaly is present during test time. Unsupervised methods play a significant role here since it is often unknown beforehand what types of defects might occur during manufacturing. In addition, industrial processes are optimized to produce a minimum amount of defective samples. Therefore, only a very limited amount of images with defects is available, in contrast to a vast amount of defect-free samples that can be used for training. Ideally, methods should provide a pixel-accurate segmentation of anomalous regions. All this makes industrial inspection tasks perfect benchmarks for unsupervised anomaly detection methods that work on natural images. Our contribution is twofold:

据我们所知，对无监督异常检测，不存在类似的数据集。为填补这个空白，进一步促进无监督异常检测方法的发展，我们提出了MVTec异常检测数据集，方便这种方法的彻底评估。我们认为工业检测任务是这种场景的理想的，有挑战的真实世界使用案例。目标或纹理的无缺陷样本图像用于训练模型，在测试时用于决定是否存在异常。无监督方法这里起到了主要角色，因为尚不知道什么类型的缺陷在生产中会出现。另外，工业过程是不断优化的，产生的缺陷样本会尽量的少。因此，缺陷样本的图像会非常有限，而会有大量无缺陷的样本，可以用于训练。理想情况下，算法应当给出异常区域的精确到像素的分割。所有这些，使得工业检测任务是无监督缺陷检测方法的完美基准测试，而且是在自然图像中使用的。我们的贡献是两方面的：

- We introduce a novel and comprehensive dataset for the task of unsupervised anomaly detection in natural image data. It mimics real-world industrial inspection scenarios and consists of 5354 high-resolution images of five unique textures and ten unique objects from different domains. There are 73 different types of anomalies in the form of defects in the objects or textures. For each defect image, we provide pixel-accurate ground truth regions (1888 in total) that allow to evaluate methods for both one-class classification and anomaly detection.

我们提出了一种新的综合数据集，用于自然图像数据的无监督异常检测任务。模拟了真实世界工业检测场景，包含了5354幅高分辨率图像，有不同领域的5种独特的纹理，和10种独特的目标。有73种不同类型的异常，表现为目标或纹理的缺陷。对每种缺陷图像，我们给出了像素级的真值区域标注（总计有1888幅），可以对一类分类方法和异常检测方法进行评估。

- We conduct a thorough evaluation of current state-of-the-art methods as well as more traditional methods for unsupervised anomaly detection on the dataset. Their performance for both segmentation and classification of anomalous images is assessed. Furthermore, we provide a well-defined way to detect anomalous regions in test images using hyperparameters that are estimated without the knowledge of any anomalous images. We show that the evaluated methods do not perform equally well across object and defect categories and that there is considerable room for improvement.

我们对无监督异常检测中，目前最好的方法和更多传统方法，在这个数据集上进行了彻底评估。其对异常图像的分割和分类性能进行了评估。而且，我们提出了一种定义良好的方法来检测测试图像中的异常区域，使用的超参数不需要任何异常图像就可以进行估计。我们展示了，评估的方法在不同的目标和缺陷类别中表现不同，而且有很大的改进空间。

## 2. Related Work

### 2.1. Existing Datasets for Anomaly Detection

We first give a brief overview of datasets that are commonly used for anomaly detection in natural images and demonstrate the need for our novel dataset. We distinguish between datasets where a simple binary decision between defect and defect-free images must be made and datasets that allow for the segmentation of anomalous regions.

我们首先给出常用于自然图像缺陷检测的数据集的概览，表明需要我们的新数据集。我们区分了两种类型的数据集，一种是只需要简单的区分缺陷图像和无缺陷图像，一种是可以分割出异常区域的数据集。

#### 2.1.1 Classification of Anomalous Images

When evaluating methods for outlier detection in mutliclass classification scenarios, a common practice is to adapt existing classification datasets for which class labels are already available. The most prominent examples are MNIST [16], CIFAR10 [14], and ImageNet [15]. A popular approach [1, 7, 21] is to select an arbitrary subset of classes, re-label them as outliers, and train a novelty detection system solely on the remaining inlier classes. During the testing phase, it is checked whether the trained model is able to correctly predict whether a test sample belongs to one of the inliner classes. While this immediately provides a large amount of training and testing data, the anomalous samples differ significantly from the samples drawn from the training distribution. Therefore, when performing evaluations on such datasets, it is unclear how a proposed method would generalize to data where anomalies manifest themselves in less significant differences from the training data manifold.

当在多类分类场景中评估异常检测的方法时，一种常见的实践是采用已有的分类数据集，其中有已经可用的类别标签。最著名的例子是MNIST，CIFAR10和ImageNet。一种流行的方法是，选择任意类别子集，将其重新标注为异常，在剩下的正常类别上训练一个异常检测系统。在测试阶段，检查训练好的模型是否可以正确的预测测试样本属于正常类别中的一个。这立刻就有了大量训练和测试数据，但异常样本与训练分布的样本差异非常大。因此，当在这种数据集上进行评估时，提出的方法是否可以泛化到其他情况，是不太清楚的，比如异常图像与训练数据差异并没有那么明显的。

For this purpose, Saleh et al. [22] propose a dataset that contains six categories of abnormally shaped objects, such as oddly shaped cars, airplanes, and boats, obtained from internet search engines that should be distinguished from regular samples of the same class in the PASCAL VOC dataset [8]. While their data might be closer to the training data manifold, the decision is again based on entire images rather than finding the parts of the images that make them novel or anomalous.

为此，Saleh等[22]提出了一个数据集，包含了6种外形不正常的目标，比如形状奇怪的车，飞机和船，是从互联网上搜索得到的，与PASCAL VOC数据集中的相同类别的常规样本可以明显区分开来。虽然其数据与训练数据可能接近，但其决策还是基于整幅图像的，而不是找到使图像异常的部分。

#### 2.1.2 Segmentation of Anomalous Regions

For the evaluation of methods that segment anomalies in images, only very few public datasets are currently available. All of them focus on the inspection of textured surfaces and, to the best of our knowledge, there does not yet exist a comprehensive dataset that allows for the segmentation of anomalous regions in natural images.

为评估分割图像中的异常的方法，目前可用的只有极少几个公开数据集。它们都关注的是纹理表面的检查，据我们所知，还不存在一个综合数据集，可以分割自然图像中的异常区域。

Carrera et al. [6] provide NanoTWICE, a dataset of 45 gray-scale images that show a nanofibrous material acquired by a scanning electron microscope. Five defect-free images can be used for training. The remaining 40 images contain anomalous regions in the form of specks of dust or flattened areas. Since the dataset only provides a single kind of texture, it is unclear how well algorithms that are evaluated on this dataset generalize to other textures of different domains.

Carrera等[6]给出了NanoTWICE，包含45幅灰度图像的数据集，是通过扫描电子显微镜展示的纳米纤维材料。5幅无缺陷的图像可用于训练。剩余的40幅图像包含异常区域，形式为尘埃斑点，或展平的区域。由于数据集只有一种纹理，在这个数据集上的算法能否泛化到其他类型的纹理，尚不清楚。

A dataset that is specifically designed for optical inspection of textured surfaces was proposed during a 2007 DAGM workshop by Wieler and Hahn [28]. They provide ten classes of artificially generated gray-scale textures with defects weakly annotated in the form of ellipses. Each class comprises 1000 defect-free texture patches for training and 150 defective patches for testing. However, their annotations are quite coarse and since the textures were generated by very similar texture models, the variance in appearance between the different textures is quite low. Furthermore, artificially generated datasets can only be seen as an approximation to the real world.

在2007 DAGM workshop上，Wieler和Hahn [28]专门为纹理表面的光学检查提出了一个数据集。数据集给出了10种人工生成的灰度纹理，缺陷是用椭圆进行的弱标注。每个类别由1000幅无缺陷的纹理块（用于训练）和150幅缺陷图像块（用于测试）组成。但是，他们的标注是非常粗糙的，由于其纹理是由非常类似的纹理模型生成的，不同纹理的外观的变化其实很小。而且，人工生成的数据集只可以视为真实世界的一种近似。

### 2.2. Methods

The landscape of methods for unsupervised anomaly detection is diverse and many approaches have been suggested to tackle the problem [1, 19]. Pimentel et al. [20] give a comprehensive review of existing work. We restrict ourselves to a brief overview of current state-of-the art methods, focusing on those that serve as baseline for our initial benchmark on the dataset.

无监督异常检测方法非常多，提出了很多方法来解决这个问题[1,19]。Pimentel等[20]给出了已有工作的综合回顾。我们只简要回顾一下目前最好的方法，聚焦在那些作为我们初始基准测试的基准的方法。

#### 2.2.1 Generative Adversarial Networks

Schlegl et al. [23] propose to model the manifold of the training data by a generative adversarial network (GAN) [10] that is trained solely on defect-free images. The generator is able to produce realistically looking images that fool a simultaneously trained discriminator network in an adversarial way. For anomaly detection, the algorithm searches for a latent sample that reproduces a given input image and manages to fool the discriminator. An anomaly segmentation can be obtained by a per-pixel comparison of the reconstructed image with the original input.

Schlegl等[23]提出通过GAN来对训练数据进行建模，只用无缺陷图像。生成器可以产生真实外观的图像，来以一种对抗的方式来欺骗一个同时训练的判别器网络。对异常检测，算法搜索隐样本，产生一个给定的输入图像，来欺骗判别器。异常分割可以通过逐像素比较重建图像与原始输入。

#### 2.2.2 Deep Convolutional Autoencoders

Convolutional Autoencoders (CAEs) [9] are commonly used as a base architecture in unsupervised anomaly detection settings. They attempt to reconstruct defect-free training samples through a bottleneck (latent space). During testing, they fail to reproduce images that differ from the data that was observed during training. Anomalies are detected by a per-pixel comparison of the input with its reconstruction. Recently, Bergmann et al. [4] pointed out the disadvantages of per-pixel loss functions in autoencoding frameworks when used in anomaly segmentation scenarios and proposed to incorporate spatial information of local patch regions using structural similarity [27] for improved segmentation results.

卷积自动编码器[9]通常以无监督异常检测的设置用作基准架构，试图通过一个瓶颈（隐空间）来重建无缺陷训练样本。在测试时，如果数据与训练时的数据不同，重建图像时就会失败。异常的检测是通过逐像素的比较输入及其重建。最近，Bergmann等[4]指出自动编码器框架中的逐像素损失函数，在用于异常分割场景时的缺点，提出将局部图像块区域的空间信息使用结构相似性纳入进来，以改进分割结果。

There exist various extensions to CAEs such as the variational autoencoders (VAEs) [13] that have been used by Baur et al. [3] for the unsupervised segmentation of anomalies in brain MR scans. Baur et al., however, do not report significant improvements over using standard CAEs. This coincides with the observations made by Bergmann et al. [4]. Nalisnick et al. [17] and Hendrycks et al. [12] provide further evidence that probabilities obtained from VAEs and other deep generative models might fail to model the true likelihood of the training data. Therefore, we restrict ourselves to deterministic autoencoder frameworks in the initial evaluation of the dataset below.

有各种CAEs的扩展，比如VAE[13]，Baur等[3]将其用于大脑MR扫描的无监督异常分割，但是，与使用标准CAE相比，并没有明显的改进。这与Bergmann等[4]的观察一致。Nalisnick等[17]和Hendrycks等[12]给出了进一步的证据，VAEs与其他生成式模型得到的概率可能不能对训练数据的真实似然进行建模。因此，我们在数据集的初步评估中，只研究确定性的自动编码器框架。

#### 2.2.3 Features of Pre-trained Convolutional Neural Networks

The aforementioned approaches attempt to learn feature representations solely from the provided training data. In addition, there exist a number of methods that use feature descriptors obtained from CNNs that have been pre-trained on a separate image classification task.

之前提到的方法试图只从给定的训练数据中学习特征表示。另外，还有一些方法使用CNNs得到的特征描述子，这些CNNs是从其他的图像分类任务预训练得到的。

Napoletano et al. [18] propose to use clustered feature descriptions obtained from the activations of a ResNet-18 [11] classification network pre-trained on ImageNet [15] to distinguish normal from anomalous data. They achieve state-of-the-art results on the NanoTWICE dataset. Being designed for one-class classification, their method only provides a binary decision whether an input image contains an anomaly or not. In order to obtain a spatial anomaly map, the classifier must be evaluated at multiple image locations, ideally at each single pixel. This quickly becomes a performance bottleneck for large images. To increase performance in practice, not every pixel location is evaluated and the resulting anomaly maps are coarse.

Napoletano等[18]提出使用在ImageNet上预训练的ResNet-18分类网络的激活值作为聚类的特征描述，来区分正常数据和异常数据，在NanoTWICE数据集上得到了目前最好的结果。他们的方法是设计用于一类分类的，只给出了二值决策，即输入图像是否包含异常。为得到空间异常图，分类器要在多个图像位置进行评估，最理想的是对每个像素进行评估。这对于大图像迅速变成了性能瓶颈。为在实践中增加性能，并没有对每个像素位置进行评估，得到的异常图是粗糙的。

#### 2.2.4 Traditional Methods

In addition to the methods described above, we consider two traditional methods for our benchmark. Bottger and Ulrich [5] extract hand-crafted feature descriptors from defect-free texture images. The distribution of feature vectors is modeled by a Gaussian Mixture Model (GMM) and anomalies are detected for extracted feature descriptors for which the GMM yields a low probability. Their algorithm can only be applied to images of regular textures.

除了上面描述的方法，我们在基准测试中考虑两个传统方法。Bottger和Ulrich[5]从无缺陷的纹理图像中提取手工设计的特征描述子。特征向量的分布是用高斯混合模型GMM来进行建模的，检测异常并提取的特征描述子，会使GMM得到一个低的概率。其算法只能应用于常规纹理的图像。

In order to obtain a simple baseline for the non-texture objects in the dataset, we consider the variation model [26, Chapter 3.4.1.4]. This method requires a prior alignment of the object contours and calculates the mean and standard deviation for each pixel. This models the gray-value statistics of the training images. During testing, a statistical test is performed for each image pixel that measures the deviation of the pixel’s gray-value from the mean. If the deviation is larger than a threshold, an anomalous pixel is detected.

为对数据集中的非纹理目标得到简单的基准，我们考虑变分模型。这个方法需要对目标轮廓进行先验的对齐，并对每个像素计算均值和标准差。这对训练图像的灰度值统计量进行了建模。在测试时，对每个像素进行了统计测试，度量了像素与均值的灰度偏差。如果偏移大于一个阈值，那么就检测出一个异常像素。

## 3. Dataset Description

The MVTec Anomaly Detection dataset comprises 15 categories with 3629 images for training and validation and 1725 images for testing. The training set contains only images without defects. The test set contains both: images containing various types of defects and defect-free images. Table 1 gives an overview for each object category. Some example images for every category together with an example defect are shown in Figure 2. We provide further example images of the dataset in the supplementary material. Five categories cover different types of regular (carpet, grid) or random (leather, tile, wood) textures, while the remaining ten categories represent various types of objects. Some of these objects are rigid with a fixed appearance (bottle, metal nut), while others are deformable (cable) or include natural variations (hazelnut). A subset of objects was acquired in a roughly aligned pose (e.g., toothbrush, capsule, and pill) while others were placed in front of the camera with a random rotation (e.g., metal nut, screw, and hazelnut). The test images of anomalous samples contain a variety of defects, such as defects on the objects’ surface (e.g., scratches, dents), structural defects like distorted object parts, or defects that manifest themselves by the absence of certain object parts. In total, 73 different defect types are present, on average five per category. The defects were manually generated with the aim to produce realistic anomalies as they would occur in real-world industrial inspection scenarios.

MVTec AD数据集包含15个类别，3626幅图像进行训练和验证，1725幅图像进行测试。训练集只包含没有缺陷的图像。测试集包含两种：无缺陷图像，和包含各种缺陷的图像。表1给出了每个目标类别的概览。每个类别的一些图像例子，和缺陷的例子，如图2所示。我们在附加材料中给出了数据集更多的图像。5个类别是规则或随机的纹理，其他10个类别表示不同类型的目标。这些目标中的一些是刚体，有固定的外观，其他则是可形变的，或包含自然变化。以大致对齐的姿态获得了一些目标（如牙刷，胶囊和药片），而其他的则以随机的旋转放在了相机前面。缺陷样本的测试图像包含各种缺陷，比如在目标表面上的缺陷，结构上的缺陷如扭曲的目标部位，或缺少特定目标部位的缺陷。共计给出了73种缺陷类型，平均每个类别有5种。数据集是手工生成的，目标是产生实际的缺陷，因为这会在真实世界工业检测场景中发生。

All images were acquired using a 2048 × 2048 pixel high-resolution industrial RGB sensor in combination with two bilateral telecentric lenses [26, Chapter 2.2.4.2] with magnification factors of 1:5 and 1:1, respectively. Afterwards, the images were cropped to a suitable output size. All image resolutions are in the range between 700 × 700 and 1024 × 1024 pixels. Since gray-scale images are also common in industrial inspection, three object categories (grid, screw, and zipper) are made available solely as single-channel images. The images were acquired under highly controlled illumination conditions. For some object classes, however, the illumination was altered intentionally to increase variability. We provide pixel-precise ground truth labels for each defective image region. In total, the dataset contains almost 1900 manually annotated regions. Some examples of labels for selected anomalous images are displayed in Figure 1.

所有图像都是用2048x2048高分辨率工业RGB传感器与两个双边远心镜头组合获得的，放大因子分别为1:5和1:1。然后，图像剪切成合适的输出大小。所有图像分辨率都在700x700到1024x1024的范围内。由于灰度图像在工业检测中很常见，三个目标类别都给出了单通道图像。图像都是在高度控制的光照条件下获得的。但对于一些目标类别，光照刻意进行了变化，以增加变化程度。我们对每个缺陷图像区域都给出了像素精度的真值标签。总计，数据集包含几乎1900个手工标注的区域。精选的缺陷图像的一些标签样本在图1中进行了展示。

## 4. Benchmark

We conduct a thorough evaluation of multiple state-of-the-art methods for unsupervised anomaly detection as an initial benchmark on our dataset. It is intended to serve as a baseline for future methods. Moreover, we provide a well-defined way to detect anomalous regions in test images using hyperparameters that are estimated solely from anomaly-free validation images. We then discuss the strengths and weaknesses of each method on the various objects and textures of the dataset. We show that, while each method can detect anomalies of certain types, none of the evaluated methods manages to excel for the entire dataset.

我们进行了多个目前最好的无监督异常检测方法的彻底评估，作为我们数据集的初始基准测试，目的是作为未来方法的基准。而且，我们给出了一个定义良好的方法来在测试图像中检测异常区域，使用的超参数是只从无异常的验证图像中估计得到的。然后我们在数据集的各种目标和纹理中讨论每种方法的强项和弱项。我们展示了，每种方法可以检测特定类型的异常，但是评估的方法中，没有哪种方法在整个数据集上都表现非常好。

### 4.1. Evaluated Methods

#### 4.1.1 AnoGAN

For the evaluation of AnoGAN, we use the publicly available implementation on Github. The GAN’s latent space dimension is fixed to 64 and generated images are of size 128 × 128 pixels, which results in relatively stable training for all categories of the dataset. Training is conducted for 50 epochs with an initial learning rate of 0.0002. During testing, 300 iterations of latent space search are performed with an initial learning rate of 0.02. Anomaly maps are obtained by a per-pixel ℓ2-comparison of the input image with the generated output.

对于AnoGAN的评估，我们使用Github上公开可用的实现。GAN的隐空间维度固定为64，生成的图像大小为128x128大小，对数据集的所有类别，都得到了相对稳定的训练。训练进行了50轮，初始学习速率为0.0002。在测试时，进行了隐空间搜索的300次迭代，初始学习速率为0.02。异常图是通过输入图像与生成的输出的逐像素l2比较得到的。

For the evaluation of objects, both training and testing images are zoomed to the input size of 128 × 128 pixels. For textures, we zoom all dataset images to size 512 × 512 and extract training patches of size 128 × 128. For training, data augmentation techniques are used as described in Section 4.2. During testing, a patchwise evaluation is performed with a horizontal and vertical stride of 128 pixels. In general, one could also imagine to choose a smaller stride and average the estimated anomaly scores. However, this is not feasible due to the relatively long runtimes of AnoGAN’s latent-space optimization.

对于目标的评估，训练图像和测试图像都放大到了128x128大小。对于纹理，我们将所有数据集图像都放大到512x512大小，提取出128x128大小的训练图像块。对于训练，我们使用了数据扩增技术，如4.2节所述。在测试时，进行了逐个图像块的评估，水平和垂直步长都为128像素。总体上，也可以选择一个更小的步长，对估计的异常分数进行平均。但是，由于AnoGAN的隐空间优化运行时间相对较长，这并不可行。

#### 4.1.2 L2 and SSIM Autoencoder

For the evaluation of the L2 and SSIM autoencoder on the texture images, we use the same CAE architecture as described by Bergmann et al. [4]. They reconstruct texture patches of size 128 × 128, employing either a per-pixel ℓ2 loss or a loss based on the structural similiarity index (SSIM). For the latter, we find an SSIM window size of 11 × 11 pixels to work well in our experiments. The latent space dimension is chosen to be 100. Larger latent space dimensions do not yield significant improvements in reconstruction quality while lower dimensions lead to degenerate reconstructions.

对于在纹理图像上的L2和SSIM自动编码器的评估，我们使用了与Bergmann等[4]相同的CAE架构。重建的纹理图像块大小为128x128，采用的是逐像素的l2损失，或基于SSIM的损失。对于后者，我们找到了一个SSIM窗口大小为11x11，在试验中效果不错。隐空间维度选择为100。更大的隐空间维度在重建质量上的改进并不明显，而更低的维度则会使重建质量更差。

Since we deem an image size of 128 × 128 too small for the reconstruction of entire objects in the dataset, we extend the architecture used for textures by an additional convolution layer to process object images at resolution 256 × 256.

由于我们认为图像大小128x128对于在数据集中重建整个目标太小，我们对用于纹理的架构进行了拓展，加上了一个额外的卷积层，处理256x256大小的目标图像。

For objects, anomaly maps are generated by passing an image through the autoencoder and comparing the reconstruction with its respective input using either per-pixel ℓ2 comparisons or SSIM. For textures, we reconstruct patches at a stride of 30 pixels and average the resulting anomaly maps. Since SSIM does not operate on color images, for the training and evaluation of the SSIM-autoencoder the images are converted to gray-scale. Data augmentation is performed as described in Section 4.2.

对于目标，异常图的生成是通过，将图像送入到自动编码器中，比较其重建结果与其输入，使用逐像素的l2或SSIM。对于纹理，我们重建的图像块的步长为30像素，对结果异常图进行平均。由于SSIM并没有对彩色图进行运算，对于SSIM-自动编码器的训练和评估，图像都转化成灰度图。数据扩增也得到了使用，如4.2所示。

#### 4.1.3 CNN Feature Dictionary

We use our own implemenation of the CNN feature dictionary proposed by Napoletano et al. [18], which extracts features from the 512-dimensional avgpool layer of a ResNet-18 pretrained on ImageNet. Principal Component Analysis (PCA) is performed on the extracted features to explain 95% of the variance, which typically results in a reduction to a feature vector with around 100 components. For K-means, we vary the number of cluster centers and identify ten cluster centers to be a good value, which agrees with the findings of Napoletano et al. We extract patches of size 16 × 16 for both the textures and objects. Objects are evaluated on image size 256 × 256 and texture images are zoomed to size 512 × 512. For evaluation, a stride of four pixels is chosen to create a coarse anomaly map. For gray-scale images, the channels are triplicated for ResNet feature extraction since the feature extractor only operates on three-channel input images.

我们自己实现了Napoletano等[18]提出的CNN特征字典并进行使用，在ImageNet上预训练了ResNet-18，从512维的avgpool层提取了特征。对提取的特征进行了PCA，以解释95%的变化，一般会降维到大约100维的特征向量。对于K-均值，我们变化了聚类中心的数量，得出10个聚类中心的效果是较好的，与Napoletano等的发现是一致的。我们对纹理和目标提取的图像块大小都是16x16。目标评估的图像大小为256x256，纹理图像为512x512。对评估，选择了像素步长为4，以得到粗糙的异常图。对于灰度级图像，通道数要复制为3个，用ResNet进行特征提取，因为特征提取器只对于三通道输入图像进行运算。

#### 4.1.4 GMM-Based Texture Inspection Model

For the texture inspection model [5], an optimized implementation is available in the HALCON machine vision library. Texture images are downscaled to an input size of 400 × 400 pixels and a four-layer image pyramid is constructed for training and evaluation. The patch size of examined texture regions on each pyramid level is set to 7×7 pixels. We use a total of ten randomly selected images from the original training set for training the texture model. Anomaly maps are obtained by evaluating the negative log-likelihood for each image pixel using the trained GMM. The method automatically provides a threshold that can be used to convert continuous anomaly maps to binarized segmentations of anomalous regions.

对于纹理检查模型[5]，在HALCON库中有一个优化的实现是可用的。纹理图像的大小缩放到400x400的输入大小，构建了4层图像金字塔用于训练和评估。在每个金字塔层次中，检查的纹理区域的图像块大小设为7x7像素。我们从原始的训练集随机选择了10幅图像来训练纹理模型。异常图的得到，是通过对每个像素使用训练的GMM评估其负log似然。这种方法自动给出了一个阈值，可用于将连续的异常图转化成异常区域的二值分割。

#### 4.1.5 Variation Model

For the evaluation of object categories using the variation model, we first attempt to align each category using shape-based matching [24, 25]. Since near pixel-accurate alignment is not possible for every object in the dataset, we restrict the evaluation of this method to a subset of objects (Table 2). We use 30 randomly selected training images of each object category in its original size to train the mean and variance parameters at each pixel location. All images are converted to gray-scale before evaluation.

为使用变化模型评估目标类别，我们首先使用基于形状的匹配来对齐每个类别。由于对数据集中的每个目标都进行接近像素精度的对齐，是不可能的，我们将这种方法的评估局限到目标的子集中（表2）。我们对每个目标类别使用30个随机选择的训练图像，分辨率为原始大小，训练在每个像素位置上的均值和方差。所有图像在评估之前都转化为灰度级图像。

Anomaly maps are obtained by computing the distance of each test pixel’s gray value to the predicted pixel mean relative to its predicted standard deviation. As for the GMM-based texture inspection, we use the optimized implementation of the HALCON machine vision library.

异常图是通过计算每个测试像素的灰度值与预测的像素均值相对于其预测的标准差来得到的。至于基于GMM的纹理检查，我们使用HALCON机器视觉库的优化的实现。

### 4.2. Data Augmentation

Since the evaluated methods based on deep architectures are typically trained on large datasets, data augmentation is performed for these methods for both textures and objects. For the texture images, we randomly crop rotated rectangular patches of fixed size from the training images. For each object category, we apply a random translation and rotation. Additional mirroring is applied where the object permits it. We augment each category to create 10000 training patches.

由于基于深度架构的方法评估一般是基于大型数据集训练的，我们对这些方法都进行了数据扩增，包括纹理和目标。对于纹理图像，我们从训练图像中随机剪切了固定大小的旋转的矩形图像块。对每个目标类别，我们使用了随机平移和旋转。在目标允许的地方，还使用了镜像。我们扩增每个类别到10000个训练图像块。

### 4.3. Evaluation Metric

Each of the evaluated methods provides a one-channel spatial map in which large values indicate that a certain pixel belongs to an anomalous region. To obtain a final segmentation result and make a binary decision for each pixel, a threshold must be determined. Only the GMM-based texture inspection provides a suitable threshold out of the box. For all other methods, we propose a well-defined way to estimate the threshold from a set of randomly selected validation images that we exclude from the training set.

每种评估的方法都给出了单通道空间图，较大的值表明特定像素属于异常区域。为找到最终的分割结果，对每个像素得到二值决策，必须确定一个阈值。只有基于GMM的纹理检测会给出一个可用的合适的阈值。对所有其他方法，我们提出了一种明确的方法来从随机选择的验证图像中估计阈值。

For every category, we define a minimum defect area that a connected component in the thresholded anomaly map must have to be classified as a defective region. For each evaluated method, we then successively segment the anomaly maps of the anomaly-free validation set with increasing thresholds. This procedure is stopped when the area of the largest anomalous region on the validation set is just below the user-defined area and the threshold that yielded this segmentation is used for further evaluation.

对每个类别，我们定义了一个最小的缺陷区域，阈值后的异常图的连通部分如果大于此区域，就需要被分类为缺陷区域。对于每种评估的方法，我们然后依次用越来越大的阈值来分割无异常的验证集的异常图。当验证集中最大的异常区域的面积低于用户定义的区域，这个过程就停止了，产生这个分割的阈值就用于进一步的评估。

Given this threshold, we evaluate the performance of each method when applied to both the anomaly classification and segmentation task. For the classification scenario, we compute the accuracy of correctly classified images for anomalous and anomaly-free test images. To assess segmentation performance, we evaluate the relative per-region overlap of the segmentation with the ground truth. To get an additional performance measure that is independent of the determined threshold, we compute the area under the receiver operating characteristic curve (ROC AUC). We define the true positive rate as the percentage of pixels that were correctly classified as anomalous across an evaluated dataset category. The false positive rate is the percentage of pixels that were wrongly classified as anomalous.

给定这个阈值，我们评估每种方法在应用到异常分类和分割任务时的性能。对于分类场景，我们计算测试图像中，异常图像和无异常图像的正确分类的准确率。为评估分割性能，我们评估分割区域与真值相比的每个区域的相对重叠。为得到与确定的阈值无关的额外的性能度量，我们计算ROC AUC曲线下的面积。我们定义了TP率为正确分类为异常的像素的比例。FP率为错误的分类为异常的像素的比例。

### 4.4. Results

Evaluation results for the classification of anomalous images and segmentation of anomalous regions are given for all methods and dataset categories in Tables 2 and 3, respectively. None of the methods manages to consistently perform well across all object and texture classes.

异常图像的分类和异常区域的分割的评估结果，对所有方法和数据集类别，分别如表2和表3所示。目前没有方法能够在所有目标和纹理类别中表现都非常好。

Looking at the five texture categories as a whole, none of the evaluated methods emerges as a clear winner. Considering only the ROC AUC, the CNN Feature Dictionary manages to perform the most consistently.

将5种纹理类别看作一个整体，评估的方法中没有出现清楚的获胜者。只考虑ROC AUC的话，CNN特征字典方法表现更加具有一致性。

For the ten object categories, the autoencoder architectures achieve the best results. Which one of these two performs better depends on the object under consideration. The L2 autoencoder achieves better per-region overlap values, indicating that the estimation of the anomaly threshold may have worked better for this method.

对于10个目标类别，自动编码器的架构获得了最好的结果。这两个中哪一个表现最好，依赖于考虑的目标。L2自动编码器获得了更好的逐区域重叠结果，说明异常阈值的估计对这种方法效果较好。

Table 3 shows that a high ROC AUC does not necessarily coincide with a high per-region overlap of the segmentation for the estimated threshold. In these cases, the ROC AUC shows that the anomaly maps successfully represent anomalies in the images but the segmentation nevertheless fails due to bad estimation of the threshold. This highlights the difficulty in trying to find a good threshold based solely on a set of anomaly-free images. In a supervised setting, i.e., with knowledge of a set of anomalous images, this estimation might often be an easier task.

表3展示了，高ROC AUC与估计阈值下的每个区域分割的高重叠并不一定一致。在这些情况下，ROC AUC表明，异常图成功的表示了图像中的异常，但分割却失败了，因为阈值估计很差。这强调了只基于无缺陷的图像来找到很好的阈值的困难性。在有监督的情况下，即，有缺陷图像的情况下，这个估计可能是一个更加容易的工作。

We now discuss for each method its overall evaluation result and provide examples for both failure cases and images for which the methods worked well (Figure 3).

我们现在讨论一下对每种方法，其总体的评估结果，给出失败案例的例子，以及效果很好的例子。

#### 4.4.1 AnoGAN

We observe a tendency of GAN training to result in mode collapse [2]. The generator then often completely fails to reproduce a given test image since all latent samples generate more or less the same image. As a consequence, AnoGAN has great difficulties with object categories for which the objects appear in various shapes or orientations in the dataset. It performs better for object categories that contain less variations, such as bottle and pill. This can be seen in Figure 3a, where AnoGAN manages to detect the crack on the pill. However, it fails to generate small details on the pill such as the colored speckles, which it also detects as anomalies. For the category carpet, AnoGAN is unable to model all the subtle variations of the textural pattern, which results in a complete failure of the method as can be seen in the bottom row of Figure 3a.

我们观察到GAN训练结果导致模式坍塌的倾向[2]。生成器通常不能复现给定的测试图像，因为所有的隐式样本生成的或多或少都是一样的图像。结果是，AnoGAN对于数据集中目标有各种形状或方向的类别，有很大困难。对变化很少的目标类别，效果较好，比如瓶子和药片。如图3a所示，其中AnoGAN能够检测出药片上的裂痕。但是，没有能生成药片上的小的细节，如有色的斑点，这也检测成了异常。对于毯子的类别，AnoGAN不能对纹理模式的细微变化进行建模，这导致了方法失败，如图3a的下一行所示。

#### 4.4.2 L2 and SSIM Autoencoder

We observe stable training across all dataset categories with reasonable reconstructions for both the SSIM and L2 autoencoder. Especially for the object categories of the dataset, both autoencoders outperform all other evaluated methods in the majority of cases. For some categories, however, both autoencoders fail to model small details, which results in rather blurry image reconstructions. This is especially the case for high-frequency textures, which appear, for example, in tile and zipper. The bottom row of Figure 3b shows that for tile, the L2 autoencoder, in addition to the cracked surface, detects many false positive regions across the entire image. A similar behavior can be observed for the SSIM autoencoder.

对SSIM和L2自动编码器，我们在所有数据集类别上都看到了稳定的训练，合理的重建结果。尤其是对于数据集中的目标类别，自动编码器在多数情况都超过了其他评估的方法。但是，对于一些类别，两种自动编码器在对小的细节进行建模上都失败了，得到了非常模糊的图像重建。这对于高频的纹理尤其是这种情况，比如，tile和zipper。图3b的底行表明，对于tile，L2自动编码器在裂痕表面之外，还检测到了很多FP区域。SSIM自动编码器也可以看到类似的行为。

#### 4.4.3 CNN Feature Dictionary

As a method proposed for the detection of anomalous regions in textured surfaces, the feature dictionary based on CNN features achieves satisfactory results for all textures except grid. Since it does not incorporate additional information about the spatial location of the extracted features, its performance degenerates when evaluated on objects. Figure 3c demonstrates good anomaly segmentation performance for carpet with only few false positives, while the color defect on metal nut is only partially found.

这是一种用于检测纹理表面的异常区域的方法，基于CNN特征的特征字典在所有纹理上都获得了很好的效果，除了grid。由于没有利用提取特征的空间位置信息，其性能在目标上就会下降。图3c表明，对毯子的异常分割结果就很好，只有少量几个FP，而对于金属nut的彩色缺陷则只找到了一部分。

#### 4.4.4 GMM-Based Texture Inspection Model

Specifically designed to operate on texture images, the GMM-based texture inspection model performs well across most texture categories of the dataset. On grid, however, it cannot achieve satisfactory results due to many small defects for which its sensitivity is not high enough (Figure 3d). Furthermore, since it only operates on gray-scale images, it fails to detect most color-based defects.

基于GMM的纹理检查模型是专门设计用于纹理图像的，在数据集的多数纹理类别中表现都很好。但是，在grid上不能得到很好的结果，因为有很多小的缺陷，其敏感度不够高（图3d）。而且，由于其只处理灰度级图像，对多数彩色缺陷，则无法检测。

#### 4.4.5 Variation Model

For the variation model, good performance can be observed on screw, toothbrush, and bottle, while it yields comparably bad results for metal nut and capsule. This is mostly due to the fact that the latter objects contain certain random variations on the objects’ surfaces, which prevents the variation model from learning reasonable mean and variance values for most of the image pixels. Figure 3e illustrates this behavior: since the imprint on the capsule can appear at various locations, it will always be misclassified as a defect.

对于变化模型，在screw, toothbrush, 和bottle上可以得到很好的性能，而在metal nut和capsule上则得到相对较差的结果。这主要是因为，后者在目标的表面上有一定的随机变化，这使得变化模型无法对图像中的大多数像素学习到合理的均值和方差值。图3e描述了这种行为：由于在capsule上的imprint可以在多个位置出现，所以一直被检测为缺陷。

## 5. Conclusions

We introduce the MVTec Anomaly Detection dataset, a novel dataset for unsupervised anomaly detection mimicking real-world industrial inspection scenarios. The dataset provides the possibility to evaluate unsupervised anomaly detection methods on various texture and object classes with different types of anomalies. Because pixel-precise ground truth labels for anomalous regions in the images are provided, it is possible to evaluate anomaly detection methods for both image-level classification as well as pixel-level segmentation.

我们提出了MVTec AD数据集，这是模仿真实世界工业检测场景的用于无监督异常检测的新数据集。数据集可以评估无监督异常检测方法，包含多种纹理和目标类别，有多种异常类型。由于给出了图像中的异常区域的像素精度的真值标签，所以可以评估图像分类和像素分割的异常检测方法。

Several state-of-the-art methods as well as two classical methods were thoroughly evaluated on this dataset. The evaluations provide a first benchmark on this dataset and show that there is still considerable room for improvement. It is our hope that the proposed dataset will stimulate the development of new unsupervised anomaly detection methods.

几种目前最好的方法，以及两种经典方法，在这个数据集上进行了彻底评估。评估在这个数据集上给出了第一个基准测试，表明还有很大的改进空间。我们希望提出的数据集会刺激新的无监督异常检测方法的发展。