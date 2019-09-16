# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

Alec Radford et al. 

## Abstract

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

最近几年，采用CNNs的监督学习在计算机视觉中得到了广泛的应用。比较起来，采用CNNs的无监督学习得到的关注就很少。本文中，我们希望弥合CNNs在监督学习和无监督学习之间的差距。我们提出一类CNNs，称为DCGANs，有特定的架构限制，表明它们可以进行高效的无监督学习。在多个图像数据集上进行训练后，我们展示了很有说服力的证据，说明我们的深度卷积对抗对学习到了很好的层次化表示，从目标的部分到场景中都可以进行学习，在生成器和判别器中都可以学习到。另外，我们用学习到的特征进行新的任务，可以应用于通用图像表示。

## 1 Introduction

Learning reusable feature representations from large unlabeled datasets has been an area of active research. In the context of computer vision, one can leverage the practically unlimited amount of unlabeled images and videos to learn good intermediate representations, which can then be used on a variety of supervised learning tasks such as image classification. We propose that one way to build good image representations is by training Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), and later reusing parts of the generator and discriminator networks as feature extractors for supervised tasks. GANs provide an attractive alternative to maximum likelihood techniques. One can additionally argue that their learning process and the lack of a heuristic cost function (such as pixel-wise independent mean-square error) are attractive to representation learning. GANs have been known to be unstable to train, often resulting in generators that produce nonsensical outputs. There has been very limited published research in trying to understand and visualize what GANs learn, and the intermediate representations of multi-layer GANs.

从大型未标记的数据集中学习可复用的特征表示，一直是一个活跃的研究领域。在计算机视觉的上下文中，可以利用实际上无限量的未标记的图像和视频，来学习很好的中间表示，可以用于很多监督学习任务中，如图像分类。我们提出，一种构建很好的图像表示的方法是，训练GAN，然后将生成器和判别器的部分网络进行复用，作为特征提取器，用作监督任务。GAN可以很好的替代最大似然技术。还可以认为，其学习过程和缺少启发性的代价函数（如逐像素的独立均方误差）对于表示学习是很有吸引力的。GANs的训练过程不是太稳定，通常会导致生成器产生无意义的结果。理解和可视化GANs学习到了什么，和多层GANs的中间表示，这方面的研究很少。

In this paper, we make the following contributions 本文中，我们作出了如下贡献：

- We propose and evaluate a set of constraints on the architectural topology of Convolutional GANs that make them stable to train in most settings. We name this class of architectures Deep Convolutional GANs (DCGAN). 我们提出并评估了，卷积GANs拓扑上的架构约束，使其在多数设置中都可以稳定的训练。我们称这类架构为DCGAN。

- We use the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms. 我们使用训练好的判别器进行图像分类，得到的性能与其他无监督算法非常类似。

- We visualize the filters learnt by GANs and empirically show that specific filters have learned to draw specific objects. 我们将GANs学习到的滤波器进行了可视化，经验上表明，学习到的特定的滤波器，是用来描述特定目标的。

- We show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples. 我们表明，生成器有很有趣的向量代数性质，可以很容易改变生成样本的很多语义性质。

## 2 Related Work

### 2.1 Representation Learning from Unlabelled Data

Unsupervised representation learning is a fairly well studied problem in general computer vision research, as well as in the context of images. A classic approach to unsupervised representation learning is to do clustering on the data (for example using K-means), and leverage the clusters for improved classification scores. In the context of images, one can do hierarchical clustering of image patches (Coates & Ng, 2012) to learn powerful image representations. Another popular method is to train auto-encoders (convolutionally, stacked (Vincent et al., 2010), separating the what and where components of the code (Zhao et al., 2015), ladder structures (Rasmus et al., 2015)) that encode an image into a compact code, and decode the code to reconstruct the image as accurately as possible. These methods have also been shown to learn good feature representations from image pixels. Deep belief networks (Lee et al., 2009) have also been shown to work well in learning hierarchical representations.

无监督表示学习，在通用计算机视觉研究中，和图像的上下文中，是一个研究的较为透彻的问题。无监督表示学习的经典方法是，对数据进行聚类（如使用K均值），利用这些聚类进行改进分类分数。在图像的上下文中，可以对图像块进行层次化聚类，以学习到强力的图像表示。另一个流行的方法是训练自动编码机（卷积的，堆叠的），将图像编码为一个紧凑的码，并对这个码进行解码以尽可能准确的重建图像。这些方法也可以从图像像素中学习到很好的特征表示。DBN也可以进行很好的学习层次化表示。

### 2.2 Generating Natural Images

Generative image models are well studied and fall into two categories: parametric and non-parametric. 生成式图像模型也研究的很多，可以分为两个类别：参数化模型和非参数化模型。

The non-parametric models often do matching from a database of existing images, often matching patches of images, and have been used in texture synthesis (Efros et al., 1999), super-resolution (Freeman et al., 2002) and inpainting (Hays & Efros, 2007).

非参数化模型通常将现有的图像与数据库中的图像进行匹配，通常匹配图像块，曾用于纹理合成，超分辨率和图像修补。

Parametric models for generating images has been explored extensively (for example on MNIST digits or for texture synthesis (Portilla & Simoncelli, 2000)). However, generating natural images of the real world have had not much success until recently. A variational sampling approach to generating images (Kingma & Welling, 2013) has had some success, but the samples often suffer from being blurry. Another approach generates images using an iterative forward diffusion process (Sohl-Dickstein et al., 2015). Generative Adversarial Networks (Goodfellow et al., 2014) generated images suffering from being noisy and incomprehensible. A laplacian pyramid extension to this approach (Denton et al., 2015) showed higher quality images, but they still suffered from the objects looking wobbly because of noise introduced in chaining multiple models. A recurrent network approach (Gregor et al., 2015) and a deconvolution network approach (Dosovitskiy et al., 2014) have also recently had some success with generating natural images. However, they have not leveraged the generators for supervised tasks.

生成图像的参数化模型也得到了广泛的研究（如，在MNIST数字上的研究，或进行纹理合成的研究）。但是，生成真实世界的自然图像直到最近才有所成功。一种变分采样生成图像的方法有一些成功，但这些样本通常会有模糊的问题。另一种方法使用了一种迭代前向扩散过程来生成图像。GAN生成的图像噪声较大，理解上有困难。这种方法的Laplacian金字塔扩展得到了更高质量的图像，但仍然有目标看起来摇晃的问题，因为在链接多个模型的时候引入了噪声。一种RNN的方法和一种解卷积网络的方法也有一些成功，可以生成图像。但是，它们没有利用生成器进行监督任务。

### 2.3 Visualizing the Internals of CNNS

One constant criticism of using neural networks has been that they are black-box methods, with little understanding of what the networks do in the form of a simple human-consumable algorithm. In the context of CNNs, Zeiler et. al. (Zeiler & Fergus, 2014) showed that by using deconvolutions and filtering the maximal activations, one can find the approximate purpose of each convolution filter in the network. Similarly, using a gradient descent on the inputs lets us inspect the ideal image that activates certain subsets of filters (Mordvintsev et al.).

使用神经网络的一个批评是，这种方法是黑箱的方法，很难以简单的人类可以理解的算法，来理解网络做了什么。在CNNs的上下文中，Zeiler等展示了，通过使用解卷积和对最大激活进行滤波，可以发现网络中每个卷积滤波器的近似目的。类似的，对输入使用梯度下降，使我们可以发现激活特定滤波器子集的理想图像。

## 3 Approach and Model Architechture

Historical attempts to scale up GANs using CNNs to model images have been unsuccessful. This motivated the authors of LAPGAN (Denton et al., 2015) to develop an alternative approach to iteratively upscale low resolution generated images which can be modeled more reliably. We also encountered difficulties attempting to scale GANs using CNN architectures commonly used in the supervised literature. However, after extensive model exploration we identified a family of architectures that resulted in stable training across a range of datasets and allowed for training higher resolution and deeper generative models.

使用CNNs来构建GANs的努力，之前都不太成功。这推动了LAPGAN的作者，提出了一种替代的方法，来迭代的放大生成的低分辨率图像，这种建模可以更为可靠。我们在使用CNN来构建GANs的的过程中遇到来困难。但是，在探索了很多模型后，我们发现了一族架构，可以在很多数据集上进行稳定的训练，可以训练得到更高分辨率的更深的生成式模型。

Core to our approach is adopting and modifying three recently demonstrated changes to CNN architectures. 我们方法的核心是，采用了最近提出的三种CNN架构，并进行修改。

The first is the all convolutional net (Springenberg et al., 2014) which replaces deterministic spatial pooling functions (such as maxpooling) with strided convolutions, allowing the network to learn its own spatial downsampling. We use this approach in our generator, allowing it to learn its own spatial upsampling, and discriminator.

第一种是全卷积网络，将确定性的空间池化函数（如最大池化）替换为带有步长的卷积，使网络学习学习其自己的空间降采样。我们在我们的生成器中使用这种方法，使其可以学习自己的空间上采样，也用在了判别器中。

Second is the trend towards eliminating fully connected layers on top of convolutional features. The strongest example of this is global average pooling which has been utilized in state of the art image classification models (Mordvintsev et al.). We found global average pooling increased model stability but hurt convergence speed. A middle ground of directly connecting the highest convolutional features to the input and output respectively of the generator and discriminator worked well. The first layer of the GAN, which takes a uniform noise distribution Z as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack. For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output. See Fig. 1 for a visualization of an example model architecture.

第二是去掉卷积特征之上的全连接层的趋势。最强的一个例子是，全局平均池化层，这用在了目前最好的图像分类模型中。我们发现全局平均池化增加了模型的稳定性，但收敛速度有所降低。将最高的卷积特征直接分别连接到生成器和判别器的输入或输出很好用。GAN的第一层，以均匀分布的噪声Z为输入，可以算作是全连接的，因为这只是一个矩阵乘积，但结果重新变形为一个4维张量，用到卷积堆叠的开始。对于判别器，最后一个卷积层被拉平，然后送入一个sigmoid输出。图1是一个模型架构样本的可视化结果。

Third is Batch Normalization (Ioffe & Szegedy, 2015) which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance. This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models. This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs. Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability. This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.

第三个是BN层，通过将每一层的输入归一化到零均值和单位方差，可以稳定学习。这有助于解决训练中的问题，这是因为不良初始化造成的，也可以帮助梯度在更深的模型中流动。这对于深度生成器开始学习非常关键，防止生成器生成的所有样本坍塌到一个点上，这是GANs训练中观察到的一个常见失败模式。但是，直接将BN应用到所有层，会得到样本震荡的结果，以及模型不稳定。我们没有将BN应用到生成器的输出和判别器的输入，从而避免了这个问题。

The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function. We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution. Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well, especially for higher resolution modeling. This is in contrast to the original GAN paper, which used the maxout activation (Goodfellow et al., 2013).

生成器中用到了ReLU激活，但输出层使用了tanh激活，是一个例外。我们观察到，使用有界的激活，可以使得模型更快的学习，并饱和，覆盖了训练分布的色彩空间。在判别器中，我们发现，使用leaky ReLU激活效果很好，尤其是对于更高分辨率的建模。这是与原始GAN文章相比得到的，它使用了maxout激活。

Architecture guidelines for stable Deep Convolutional GANs: 稳定的DCGANs的架构指南：

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator). 将池化层替换成带有步长的卷积（判别器）和分数步长的卷积（生成器）。
- Use batchnorm in both the generator and the discriminator. 在生成器和判别器中使用了BN层。
- Remove fully connected hidden layers for deeper architectures. 在更深的架构中去除了全连接层。
- Use ReLU activation in generator for all layers except for the output, which uses Tanh. 在生成器的所有层中使用ReLU激活，除了输出层使用了tanh。
- Use LeakyReLU activation in the discriminator for all layers. 在判别器的所有层中使用leaky ReLU激活。

## 4 Details of Adversarial Training

We trained DCGANs on three datasets, Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset. Details on the usage of each of these datasets are given below.

我们在三个数据集上训练了DCGANs，大规模场景理解(LSUN)，ImageNet-1k和新组成的人脸数据集。每个数据集的使用细节如下所述。

No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1]. All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128. All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02. In the LeakyReLU, the slope of the leak was set to 0.2 in all models. While previous GAN work has used momentum to accelerate training, we used the Adam optimizer (Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested learning rate of 0.001, to be too high, using 0.0002 instead. Additionally, we found leaving the momentum term β1 at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.

训练图像的预处理只是将其缩放到tanh函数的范围中[-1,1]。所有模型都使用mini-batch的SGD进行训练，mini-batch的大小为128。所有权重初始化为零均值的正态分布的随机变量，标准差为0.02。在leaky ReLU中，所有模型leak的斜率都设为0.2。之前有关GAN的工作使用了动量来加速训练，我们使用的则是Adam优化器，并对超参数进行调节。我们发现建议的学习速率0.001太高了，使用的是0.0002。另外，我们还发现，动量项β1在建议值0.9时，会得到训练震荡和不稳定，将其降低到0.5会帮助稳定训练。

Figure 1: DCGAN generator used for LSUN scene modeling. A 100 dimensional uniform distribu- tion Z is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions (in some recent papers, these are wrongly called deconvolutions) then convert this high level representation into a 64 × 64 pixel image. Notably, no fully connected or pooling layers are used.

### 4.1 LSUN

As visual quality of samples from generative image models has improved, concerns of over-fitting and memorization of training samples have risen. To demonstrate how our model scales with more data and higher resolution generation, we train a model on the LSUN bedrooms dataset containing a little over 3 million training examples. Recent analysis has shown that there is a direct link between how fast models learn and their generalization performance (Hardt et al., 2015). We show samples from one epoch of training (Fig.2), mimicking online learning, in addition to samples after convergence (Fig.3), as an opportunity to demonstrate that our model is not producing high quality samples via simply overfitting/memorizing training examples. No data augmentation was applied to the images.

由于生成式图像模式生成的样本的视觉质量得到了改进，所以过拟合和记住训练样本的风险就变大了。为证明我们的模型可以随着生成更多的数据和更高的分辨率而缩放，我们在LSUN卧室数据集上训练一个模型，数据集包含超过300万训练样本。最近的分析表明，模型学习速度有多快，与其泛化性能有直接的关系。我们给出一轮数据训练得到的例子（图2），模仿在线学习，以及收敛后的样本（图3），这说明了，我们的模型不是通过简单的过你和或记住训练样本来得到更高质量的样本。对训练图像没有使用数据增强。

Figure2: Generatedbedroomsafteronetrainingpassthroughthedataset.Theoretically,themodel could learn to memorize training examples, but this is experimentally unlikely as we train with a small learning rate and minibatch SGD. We are aware of no prior empirical evidence demonstrating memorization with SGD and a small learning rate.

Figure3: Generatedbedroomsafterfiveepochsoftraining.Thereappearstobeevidenceofvisual under-fitting via repeated noise textures across multiple samples such as the base boards of some of the beds.

#### 4.1.1 Deduplication

To further decrease the likelihood of the generator memorizing input examples (Fig.2) we perform a simple image de-duplication process. We fit a 3072-128-3072 de-noising dropout regularized RELU autoencoder on 32x32 downsampled center-crops of training examples. The resulting code layer activations are then binarized via thresholding the ReLU activation which has been shown to be an effective information preserving technique (Srivastava et al., 2014) and provides a convenient form of semantic-hashing, allowing for linear time de-duplication. Visual inspection of hash collisions showed high precision with an estimated false positive rate of less than 1 in 100. Additionally, the technique detected and removed approximately 275,000 near duplicates, suggesting a high recall.

为进一步降低生成器记住输入样本的可能性（图2），我们进行了一个简单的图像去重复化的过程。我们取训练样本的中间剪切块，降采样到32✖32大小，然后对其使用了一个3072-128-3072的去噪dropout正则化ReLU的自动编码机。得到的编码层激活，通过对ReLU激活使用阈值进行二值化，这已经证明是一个有效的信息保存技术，是语义哈希的简便形式，可以进行时间线性的去重复化。对哈希冲突的情况进行视觉上的检查，表明精度非常高，估计的假阳性率小于1%。另外，这个技术检测并去掉了大约275000个接近的重复，说明召回率非常高。

### 4.2 Faces

We scraped images containing human faces from random web image queries of peoples names. The people names were acquired from dbpedia, with a criterion that they were born in the modern era. This dataset has 3M images from 10K people. We run an OpenCV face detector on these images, keeping the detections that are sufficiently high resolution, which gives us approximately 350,000 face boxes. We use these face boxes for training. No data augmentation was applied to the images.

我们在网上随机检索人名，爬取包含人脸的图像。人名是通过dbpedia得到的，选取准则是生于现代的人。数据集有10K人的3M图像。我们对这些图像用opencv进行人脸检测，保证检测到的人脸分辨率足够高，这样得到了大约350000人脸框。我们使用这些人脸框进行训练。这些图像没有使用数据扩充。

### 4.3 ImageNet-1K

We use Imagenet-1k (Deng et al., 2009) as a source of natural images for unsupervised training. We train on 32 × 32 min-resized center crops. No data augmentation was applied to the images. 我们采用ImageNet-1k作为无监督训练的自然图像数据源。我们改变图像的大小，并截取32✖32的中间块。对这些图像没有进行数据扩充。

## 5 Empirical Validation of DCGANs Capabilities

### 5.1 Classifying CIFAR-10 Using GANs as a Feature Extractor

One common technique for evaluating the quality of unsupervised representation learning algorithms is to apply them as a feature extractor on supervised datasets and evaluate the performance of linear models fitted on top of these features. 评估无监督表示学习算法的质量，一个常用的技术是，在有监督的数据集上将其用作特征提取器，在这些特征上拟合出线性模型，并评估其性能。

On the CIFAR-10 dataset, a very strong baseline performance has been demonstrated from a well tuned single layer feature extraction pipeline utilizing K-means as a feature learning algorithm. When using a very large amount of feature maps (4800) this technique achieves 80.6% accuracy. An unsupervised multi-layered extension of the base algorithm reaches 82.0% accuracy (Coates & Ng, 2011). To evaluate the quality of the representations learned by DCGANs for supervised tasks, we train on Imagenet-1k and then use the discriminator’s convolutional features from all layers, maxpooling each layers representation to produce a 4 × 4 spatial grid. These features are then flattened and concatenated to form a 28672 dimensional vector and a regularized linear L2-SVM classifier is trained on top of them. This achieves 82.8% accuracy, outperforming all K-means based approaches. Notably, the discriminator has many less feature maps (512 in the highest layer) compared to K-means based techniques, but does result in a larger total feature vector size due to the many layers of 4 × 4 spatial locations. The performance of DCGANs is still less than that of Exemplar CNNs (Dosovitskiy et al., 2015), a technique which trains normal discriminative CNNs in an unsupervised fashion to differentiate between specifically chosen, aggressively augmented, exemplar samples from the source dataset. Further improvements could be made by finetuning the discriminator’s representations, but we leave this for future work. Additionally, since our DCGAN was never trained on CIFAR-10 this experiment also demonstrates the domain robustness of the learned features.

在CIFAR-10数据集上，一个很强的基准性能是，用k-均值作为特征学习算法，很好的调整这个单层特征提取的流程。当使用大量特征图(4800)时，这种技术可以取得80.6%的准确率。这种基准算法的一种无监督的多层拓展，可以达到82.0%的准确率。为评估DCGANs学习到的表示的质量，以进行有监督的任务，我们在ImageNet-1k上进行训练，然后使用判别器在所有层的卷积特征，对每一层的表示进行最大池化，生成4✖4的空间格子。这些特征然后拉平，进行拼接，形成一个28672维的向量，然后在这些特征之上训练一个正则化的线性L2-SVM分类器。这可以得到82.8%的准确率，超过所有基于K-均值的方法。值得注意的是，与基于K-均值的方法相比，判别器的特征图数量要少的多（最高的层有512个特征图），但却得到了更大的总计特征向量大小，这是因为有很多层的4✖4空间位置点。DCGANs的性能仍然比Exemplar CNNs的性能要差一些，这是一种以无监督的式样来训练判别式CNNs的方法，可以区分源数据集的图像与特意选定的、经过激进扩充的范例样本。要做进一步的改进，可以精调判别器的表示，但这是我们将来的工作。另外，由于我们的DCGAN从来没有在CIFAR-10上训练过，这个试验也证明了学习的特征的领域稳健性。

Table 1: CIFAR-10 classification results using our pre-trained model. Our DCGAN is not pretrained on CIFAR-10, but on Imagenet-1k, and the features are used to classify CIFAR-10 images.

5.2 Classifying SVHN Digits Using GANs as a Feature Extractor

On the StreetView House Numbers dataset (SVHN)(Netzer et al., 2011), we use the features of the discriminator of a DCGAN for supervised purposes when labeled data is scarce. Following similar dataset preparation rules as in the CIFAR-10 experiments, we split off a validation set of 10,000 examples from the non-extra set and use it for all hyperparameter and model selection. 1000 uniformly class distributed training examples are randomly selected and used to train a regularized linear L2-SVM classifier on top of the same feature extraction pipeline used for CIFAR-10. This achieves state of the art (for classification using 1000 labels) at 22.48% test error, improving upon another modifcation of CNNs designed to leverage unlabled data (Zhao et al., 2015). Additionally, we validate that the CNN architecture used in DCGAN is not the key contributing factor of the model’s performance by training a purely supervised CNN with the same architecture on the same data and optimizing this model via random search over 64 hyperparameter trials (Bergstra & Bengio, 2012). It achieves a signficantly higher 28.87% validation error.

在街景房间牌号(SVHN)数据集中，我们以有监督的目的，来使用DCGAN的判别器的特征，而标注的数据则非常少。与在CIFAR-10试验中的数据集准备非常类似，我们从非额外集分割出10000样本的验证集，将其用于超参数和模型选择。从所有类别中随机选择1000个均匀分布于各类的训练样本，用于训练一个正则化的线性L2-SVM分类器，使用的特征提取流程与CIFAR-10上一样。这得到了目前最好的测试误差22.48%（对于使用1000个标签的分类问题），在另一个使用无标签的数据模型基础上改进了效果，这个模型也是修改了CNNs模型的。另外，我们还验证了，DCGAN中使用的CNN架构，并不是模型性能的关键贡献因素；我们使用相同的架构、相同的数据，以纯监督的CNN方式进行了训练，并对超参数进行64种随机搜索，来优化模型。这样得到的验证错误率明显更高，达到来28.87%。

Table 2: SVHN classification with 1000 labels

## 6 Investigating and Visualizing the Internals of the Networks

We investigate the trained generators and discriminators in a variety of ways. We do not do any kind of nearest neighbor search on the training set. Nearest neighbors in pixel or feature space are trivially fooled (Theis et al., 2015) by small image transforms. We also do not use log-likelihood metrics to quantitatively assess the model, as it is a poor (Theis et al., 2015) metric.

我们以很多方式研究了训练好的生成器和判别器。我们没有在训练集上进行任何形式的最近邻搜索。在像素空间或特征空间中的最近邻，通过小型图像变换可以很容易的进行欺骗。我们也没有使用log概率的度量标准，以量化的评估模型，因为这不是一种好的度量标准。

### 6.1 Walking in the Latent Space

The first experiment we did was to understand the landscape of the latent space. Walking on the manifold that is learnt can usually tell us about signs of memorization (if there are sharp transitions) and about the way in which the space is hierarchically collapsed. If walking in this latent space results in semantic changes to the image generations (such as objects being added and removed), we can reason that the model has learned relevant and interesting representations. The results are shown in Fig.4.

我们所做的第一个试验，是理解隐藏空间的样子。在学习到的流形中进行探索，通常可以看到记住的迹象（如果有尖锐的迁移），还可以看到空间是怎样层次化的坍塌的。如果在隐藏空间的探索，看到了图像生成的过程中，有语义变化（如增加了目标，或去除了目标），我们就可以推理得到，模型学习到了相关的有趣的表示。结果如图4所示。

Figure 4: Top rows: Interpolation between a series of 9 random points in Z show that the space learned has smooth transitions, with every image in the space plausibly looking like a bedroom. In the 6th row, you see a room without a window slowly transforming into a room with a giant window. In the 10th row, you see what appears to be a TV slowly being transformed into a window.

### 6.2 Visualizing the Discriminator Features

Previous work has demonstrated that supervised training of CNNs on large image datasets results in very powerful learned features (Zeiler & Fergus, 2014). Additionally, supervised CNNs trained on scene classification learn object detectors (Oquab et al., 2014). We demonstrate that an unsupervised DCGAN trained on a large image dataset can also learn a hierarchy of features that are interesting. Using guided backpropagation as proposed by (Springenberg et al., 2014), we show in Fig.5 that the features learnt by the discriminator activate on typical parts of a bedroom, like beds and windows. For comparison, in the same figure, we give a baseline for randomly initialized features that are not activated on anything that is semantically relevant or interesting.

之前的工作已经证明了，在大型图像数据集上对CNNs进行有监督的训练，会得到很强大的学习特征。另外，在场景分类情况下训练的有监督的CNNs可以学习到目标检测器。我们则证明了，在大型数据集上训练的无监督的DCGAN，也可以学习到很有趣的特征层次。使用其他文章中提出的引导反向传播，我们在图5中给出，判别器学习到的特征在卧室的某一部分会得到激活，如床或窗户。为进行比较，在图5中，我们给出了一个基准，特征是随机初始化的，没有在任何语义相关或有趣的目标上激活。

Figure 5: On the right, guided backpropagation visualizations of maximal axis-aligned responses for the first 6 learned convolutional features from the last convolution layer in the discriminator. Notice a significant minority of features respond to beds - the central object in the LSUN bedrooms dataset. On the left is a random filter baseline. Comparing to the previous responses there is little to no discrimination and random structure.

### 6.3 Manipulating the Generator Representation

#### 6.3.1 Forgetting to Draw Certain Objects

In addition to the representations learnt by a discriminator, there is the question of what representations the generator learns. The quality of samples suggest that the generator learns specific object representations for major scene components such as beds, windows, lamps, doors, and miscellaneous furniture. In order to explore the form that these representations take, we conducted an experiment to attempt to remove windows from the generator completely.

除了判别器学习到的特征，还有生成器学习到的什么特征的问题。样本的质量说明，生成器学习了主要场景组件的特定目标表示，如床，窗户，灯，门和各种家具。为探索这些表示所采用的形式，我们进行了试验，从生成器中把窗户完全去除。

On 150 samples, 52 window bounding boxes were drawn manually. On the second highest convolution layer features, logistic regression was fit to predict whether a feature activation was on a window (or not), by using the criterion that activations inside the drawn bounding boxes are positives and random samples from the same images are negatives. Using this simple model, all feature maps with weights greater than zero ( 200 in total) were dropped from all spatial locations. Then, random new samples were generated with and without the feature map removal.

在150个样本中，52个窗户的边界框是手动画出来的。在次高的卷积层特征上，进行了logistic regression，预测这个特征激活是否是一个窗户，使用的规则是，在边界框之内的激活是正的，同一图像中的随机样本是负的。使用这个简单的模型，权重大于0的所有的特征图（总计有200个），在空间位置上全部丢弃掉。然后，在特征图去除和不去除的情况下，生成随机的新的样本。

The generated images with and without the window dropout are shown in Fig.6, and interestingly, the network mostly forgets to draw windows in the bedrooms, replacing them with other objects.

图6给出了，丢弃和不丢弃情况下，生成的图像，有趣的是，网络主要是在卧室里忘记了画窗户，将其替换为了其他物体。

Figure 6: Top row: un-modified samples from model. Bottom row: the same samples generated with dropping out ”window” filters. Some windows are removed, others are transformed into objects with similar visual appearance such as doors and mirrors. Although visual quality decreased, overall scene composition stayed similar, suggesting the generator has done a good job disentangling scene representation from object representation. Extended experiments could be done to remove other objects from the image and modify the objects the generator draws.

#### 6.3.2 Vector Arithmetic on Face Samples

In the context of evaluating learned representations of words (Mikolov et al., 2013) demonstrated that simple arithmetic operations revealed rich linear structure in representation space. One canonical example demonstrated that the vector(”King”) - vector(”Man”) + vector(”Woman”) resulted in a vector whose nearest neighbor was the vector for Queen. We investigated whether similar structure emerges in the Z representation of our generators. We performed similar arithmetic on the Z vectors of sets of exemplar samples for visual concepts. Experiments working on only single samples per concept were unstable, but averaging the Z vector for three examplars showed consistent and stable generations that semantically obeyed the arithmetic. In addition to the object manipulation shown in (Fig. 7), we demonstrate that face pose is also modeled linearly in Z space (Fig. 8).

评估学习到的语句表示，有文章表明，简单的代数运算，就可以揭示表示空间中丰富的线性结构。一个经典的例子表明，vector("King") - vector("Man") + vector("woman")得到的向量，其最近邻是Queen的向量。我们研究了，在我们的生成器的Z表示中，是否也有类似的结构。我们在视觉概念的范例样本集上，对其Z向量进行了类似的代数运算。每个概念只取一个样本，其试验是不稳定的，但对三个样本的Z向量进行平均，会得到持续稳定的生成结果，语义上符合其代数运算。除了图7中给出的目标操作，我们还证明了，人脸姿态在Z空间中也是线性建模的（图8）。

Figure 7: Vector arithmetic for visual concepts. For each column, the Z vectors of samples are averaged. Arithmetic was then performed on the mean vectors creating a new vector Y . The center sample on the right hand side is produce by feeding Y as input to the generator. To demonstrate the interpolation capabilities of the generator, uniform noise sampled with scale +-0.25 was added to Y to produce the 8 other samples. Applying arithmetic in the input space (bottom two examples) results in noisy overlap due to misalignment.

Figure 8: A ”turn” vector was created from four averaged samples of faces looking left vs looking right. By adding interpolations along this axis to random samples we were able to reliably transform their pose.

These demonstrations suggest interesting applications can be developed using Z representations learned by our models. It has been previously demonstrated that conditional generative models can learn to convincingly model object attributes like scale, rotation, and position (Dosovitskiy et al., 2014). This is to our knowledge the first demonstration of this occurring in purely unsupervised models. Further exploring and developing the above mentioned vector arithmetic could dramatically reduce the amount of data needed for conditional generative modeling of complex image distributions.

这些证明说明了，使用我们的模型学习到的Z表示，可以开发出有趣的应用。之前就证明过，条件生成式模型可以学习到对目标属性进行很好的建模，如尺度、旋转和位置。据我们所知，这是在纯无监督模型中，第一个这样的例子。进一步探索和发展上面提到的向量代数，可以极大的减少复杂图像分布的条件生成模型所需的数据。

## 7 Conclusion and Future Work

We propose a more stable set of architectures for training generative adversarial networks and we give evidence that adversarial networks learn good representations of images for supervised learning and generative modeling. There are still some forms of model instability remaining - we noticed as models are trained longer they sometimes collapse a subset of filters to a single oscillating mode.

我们提出了一个训练GAN的更稳定的架构集，对于监督学习和生成式建模，对抗网络学习到了很好的表示，我们给出了这方面的证据。仍然还有一些建模不稳定性，我们注意到，当模型训练更长时间时，一些滤波器子集会坍缩到一种震荡模式。

Further work is needed to tackle this from of instability. We think that extending this framework to other domains such as video (for frame prediction) and audio (pre-trained features for speech synthesis) should be very interesting. Further investigations into the properties of the learnt latent space would be interesting as well.

为处理这种不稳定性，需要进一步的工作。我们认为，将这个框架拓展到其他领域中，如视频（进行帧预测），和音频（用预训练特征进行语音合成）应当非常有趣。对学习到的隐藏空间进行进一步的探索，也应当非常有趣。

## 8 Supplementary Material

### 8.1 Evaluating DCGANs Capability to Capture Data Distributions

We propose to apply standard classification metrics to a conditional version of our model, evaluating the conditional distributions learned. We trained a DCGAN on MNIST (splitting off a 10K validation set) as well as a permutation invariant GAN baseline and evaluated the models using a nearest neighbor classifier comparing real data to a set of generated conditional samples. We found that removing the scale and bias parameters from batchnorm produced better results for both models. We speculate that the noise introduced by batchnorm helps the generative models to better explore and generate from the underlying data distribution. The results are shown in Table 3 which compares our models with other techniques. The DCGAN model achieves the same test error as a nearest neighbor classifier fitted on the training dataset - suggesting the DCGAN model has done a superb job at modeling the conditional distributions of this dataset. At one million samples per class, the DCGAN model outperforms InfiMNIST (Loosli et al., 2007), a hand developed data augmentation pipeline which uses translations and elastic deformations of training examples. The DCGAN is competitive with a probabilistic generative data augmentation technique utilizing learned per class transformations (Hauberg et al., 2015) while being more general as it directly models the data instead of transformations of the data.

我们提出将标准的分类度量标准，应用于我们模型的条件式版本，评估学习到的条件式分布。我们在MNIST上训练一个DCGAN（切分出一个10K的验证集），还训练了一个排列不变的GAN基准，使用最近邻分类器评估模型，将真实数据与生成的条件式样本进行比较。我们发现，从BN中去掉尺度和偏置参数，对两种模型都会得到更好的结果。我们推测，BN带来的噪声，帮助了生成式模型从潜在的数据分布中更好的探索和生成数据。结果如表3所示，将我们的模型与其他方法进行了比较。DCGAN模型与一个最近邻分类器得到了相同的测试错误率，最近邻分类器在训练数据集上进行了拟合，说明DCGAN模型很好的建模了这个数据集的条件分布。在每个类别100万样本的情况下，DCGAN模型超过了InfiMNIST，这是一个手动提出的数据扩充流程，对训练样本使用了平移和弹性形变。DCGAN与另一个概率生成式数据扩充技术效果类似，但更一般化，因为直接对数据进行建模，而不是对数据的变换进行的建模。