# A Survey on Deep Learning in Medical Image Analysis

Geert Litjens et al. Radboud University Medical Center, The Netherlands

## Abstract 摘要

Deep learning algorithms, in particular convolutional networks, have rapidly become a methodology of choice for analyzing medical images. This paper reviews the major deep learning concepts pertinent to medical image analysis and summarizes over 300 contributions to the field, most of which appeared in the last year. We survey the use of deep learning for image classification, object detection, segmentation, registration, and other tasks. Concise overviews are provided of studies per application area: neuro, retinal, pulmonary, digital pathology, breast, cardiac, abdominal, musculoskeletal. We end with a summary of the current state-of-the-art, a critical discussion of open challenges and directions for future research.

深度学习算法，特别是卷积网络，已经迅速成为医学图像分析的一种方法。本文回顾了与医学图像分析相关的深度学习概念，并总结了该领域的超过300个贡献点，其大多数出现在去年。我们调查了深度学习在图像分类、目标检测、分割、对齐和其他任务中的应用。给出了每个应用领域中的简练综述：脑神经，视网膜，肺部，数字病理学，乳腺，心脏病，腹部，肌肉骨骼。我们最后总结了目前最好的方法，讨论了开放性挑战，并指出了未来研究的方向。

**Keywords**: deep learning, convolutional neural networks, medical imaging, survey

## 1. Introduction 引言

As soon as it was possible to scan and load medical images into a computer, researchers have built systems for automated analysis. Initially, from the 1970s to the 1990s, medical image analysis was done with sequential application of low-level pixel processing (edge and line detector filters, region growing) and mathematical modeling (fitting lines, circles and ellipses) to construct compound rule-based systems that solved particular tasks. There is an analogy with expert systems with many if-then-else statements that were popular in artificial intelligence in the same period. These expert systems have been described as GOFAI (good old-fashioned artificial intelligence) (Haugeland, 1985) and were often brittle; similar to rule-based image processing systems.

自从可以将医学图像扫描并存放到计算机中的时候，研究者就构建了自动分析的系统。最初的时候，从1970s到1990s，医学图像分析是首先进行低层像素处理（边缘和线段检测滤波器，区域增长）和数学建模（拟合线段、圆形和椭圆形），然后构建复合的基于规则的系统，以解决特定的问题。同一时期人工智能领域流行的专家系统，有一个类比，就是很多的if-then-else语句。这些专家系统被描述成GOFAI（好用的老式人工智能），性能通常很脆弱；基于规则的图像处理系统也很类似。

At the end of the 1990s, supervised techniques, where training data is used to develop a system, were becoming increasingly popular in medical image analysis. Examples include active shape models (for segmentation), atlas methods (where the atlases that are fit to new data form the training data), and the concept of feature extraction and use of statistical classifiers (for computer-aided detection and diagnosis). This pattern recognition or machine learning approach is still very popular and forms the basis of many successful commercially available medical image analysis systems. Thus, we have seen a shift from systems that are completely designed by humans to systems that are trained by computers using example data from which feature vectors are extracted. Computer algorithms determine the optimal decision boundary in the high-dimensional feature space. A crucial step in the design of such systems is the extraction of discriminant features from the images. This process is still done by human researchers and, as such, one speaks of systems with handcrafted features.

1990年代末期，监督方法在医学图像分析中变得越来越流行，即用训练数据来构建一个系统。例子包括active shape models（进行分割），atlas方法（其中atlas适配到新数据中，形成训练数据），特征提取的概念，和统计分类器的使用（进行计算机辅助检测和诊断）。这种模式识别或机器学习方法现在仍然很流行，是很多成功的商用医学图像分析系统的基础。所以，我们见证了这样的转变，即完全由人类设计的系统，到由计算机从样本数据中提取特征训练出来的系统。计算机算法确定了在高维特征空间中最优决策界面。这种系统的设计中，一个关键步骤是，从图像中提取有区分力的特征。这个过程仍然由人类研究者完成，因此，这种系统使用的是手工设计的特征。

A logical next step is to let computers learn the features that optimally represent the data for the problem at hand. This concept lies at the basis of many deep learning algorithms: models (networks) composed of many layers that transform input data (e.g. images) to outputs (e.g. disease present/absent) while learning increasingly higher level features. The most successful type of models for image analysis to date are convolutional neural networks (CNNs). CNNs contain many layers that transform their input with convolution filters of a small extent. Work on CNNs has been done since the late seventies (Fukushima, 1980) and they were already applied to medical image analysis in 1995 by Lo et al. (1995). They saw their first successful real-world application in LeNet (LeCun et al., 1998) for hand-written digit recognition. Despite these initial successes, the use of CNNs did not gather momentum until various new techniques were developed for efficiently training deep networks, and advances were made in core computing systems. The watershed was the contribution of Krizhevsky et al. (2012) to the ImageNet challenge in December 2012. The proposed CNN, called AlexNet, won that competition by a large margin. In subsequent years, further progress has been made using related but deeper architectures (Russakovsky et al., 2014). In computer vision, deep convolutional networks have now become the technique of choice.

逻辑上来说，下一步是让计算机学习特征，以得到相关问题数据的最优表示。这个概念是很多深度学习算法的基础：由很多层组成的模型（网络），学习越来越高层次的特征，将输入数据（如图像）转化为输出（如存在的/不存在的疾病）。目前最成功的图像处理模型类型是卷积神经网络(CNNs)。CNNs包含很多层，每层将其输入用卷积滤波器进行变换。CNNs方面的工作可以追溯到1970s末期，在1995年就曾用于医学图像分析。第一个成功的真实世界应用是LeNet，进行手写数字识别。尽管有这些开始的成功，但直到发展出了新技术进行高效训练深度网络，以及核心计算系统有了新的进展，才聚集了足够的动能使用CNNs。分水岭出现在2012年AlexNet在ImageNet挑战赛上的成功，大比分超越了之前的成绩。后来，使用类似但更深的结构，取得了进一步的进展。在计算机视觉中，深度卷积网络已经成为了必选的技术。

The medical image analysis community has taken notice of these pivotal developments. However, the transition from systems that use handcrafted features to systems that learn features from the data has been gradual. Before the breakthrough of AlexNet, many different techniques to learn features were popular. Bengio et al. (2013) provide a thorough review of these techniques. They include principal component analysis, clustering of image patches, dictionary approaches, and many more. Bengio et al. (2013) introduce CNNs that are trained end-to-end only at the end of their review in a section entitled Global training of deep models. In this survey, we focus particularly on such deep models, and do not include the more traditional feature learning approaches that have been applied to medical images. For a broader review on the application of deep learning in health informatics we refer to Ravi et al. (2017), where medical image analysis is briefly touched upon.

医学图像分析团体注意到了这些关键核心发展。但是，从使用手工设计的特征的系统，转移到从数据中学习特征的系统，这个过程是渐进的。在AlexNet的突破之前，很多不同的学习特征的技术很流行。Bengio等给出了这些技术的完全回顾。这包括PCA，图像块聚类、字典方法和其他很多。Bengio等只在其综述的最后，用一节的篇幅，标题为Global training of deep models，介绍了可以进行端到端训练的CNNs。本文中，我们特别关注这些深度模型，没有包含更传统的应用于医学图像的特征学习方法。要更广泛的回顾深度学习在健康信息学中的应用，我们推荐Ravi等人的文章，其中简要介绍了医学图像分析。

Applications of deep learning to medical image analysis first started to appear at workshops and conferences, and then in journals. The number of papers grew rapidly in 2015 and 2016. This is illustrated in Figure 1. The topic is now dominant at major conferences and a first special issue appeared of IEEE Transaction on Medical Imaging in May 2016 (Greenspan et al., 2016).

深度学习在医学图像分析中的应用，首先出现在研讨会和会议上，然后出现在期刊上。论文数量从2015到2016年增长迅速。这如图1所示。这个课题在主要的会议上已经占据主流，IEEE Transaction on Medical Imaging在2016年5月专门出版了一期特刊。

One dedicated review on application of deep learning to medical image analysis was published by Shen et al. (2017). Although they cover a substantial amount of work, we feel that important areas of the field were not represented. To give an example, no work on retinal image analysis was covered. The motivation for our review was to offer a comprehensive overview of (almost) all fields in medical imaging, both from an application and a methodology-drive perspective. This also includes overview tables of all publications which readers can use to quickly assess the field. Last, we leveraged our own experience with the application of deep learning methods to medical image analysis to provide readers with a dedicated discussion section covering the state-of-the-art, open challenges and overview of research directions and technologies that will become important in the future.

深度学习在医疗图像分析中的一篇很好的综述是Shen等发表的文章。虽然他们包含了非常多的工作，我们感觉有的重要领域没有表现出来。比如，没有包含视网膜图像分析。我们综述的动机就是给出几乎所有医学成像领域的综合回顾，包含应用的视角，和方法论的视角。还包含了所有出版物的综述表，读者可以迅速评估这个领域。最后，我们利用我们对深度学习在医学图像分析中的理解和经验，给出了一个专门的讨论部分，覆盖了目前最好的方法，开放的挑战和研究方向的回顾。

This survey includes over 300 papers, most of them recent, on a wide variety of applications of deep learning in medical image analysis. To identify relevant contributions PubMed was queried for papers containing (”convolutional” OR ”deep learning”) in title or abstract. ArXiv was searched for papers mentioning one of a set of terms related to medical imaging. Additionally, conference proceedings for MICCAI (including workshops), SPIE, ISBI and EMBC were searched based on titles of papers. We checked references in all selected papers and consulted colleagues. We excluded papers that did not report results on medical image data or only used standard feed-forward neural networks with handcrafted features. When overlapping work had been reported in multiple publications, only the publication(s) deemed most important were included. We expect the search terms used to cover most, if not all, of the work incorporating deep learning methods. The last update to the included papers was on February 1, 2017. The appendix describes the search process in more detail.

本文覆盖了超过300篇文章，大部分都是最近的，包含深度学习在医学图像分析中的广泛应用。为识别相关的贡献，查询了PubMed中包含卷积或深度学习（在标题或摘要）的文章。搜索了ArXiv上与医学成像相关的文章。另外，搜索了MICCAI、SPIE、ISBI和EMBC的会议论文。我们检查了所有选择的文献的参考文献，咨询了相关人员。我们排除了没有在医疗图像数据中给出结果的文章，或只在手工设计的特征上使用标准前向神经网络的文章。当多个文献中出现重复的工作时，则只包含更重要的文章。我们希望搜索项包含了绝大部分深度学习方法的工作。包含的文章截止到2017年2月。附录更详细的描述了这个搜索过程。

Summarizing, with this survey we aim to: 总结下来，这个综述的目的是：

- show that deep learning techniques have permeated the entire field of medical image analysis; 说明了深度学习技术已经渗透了医学图像分析的整个领域；
- identify the challenges for successful application of deep learning to medical imaging tasks; 找到深度学习在医疗图像任务中成功应用的挑战；
- highlight specific contributions which solve or circumvent these challenges. 强调解决或绕过这些挑战的特定贡献。

The rest of this survey as structured as followed. In Section 2 we introduce the main deep learning techniques that have been used for medical image analysis and that are referred to throughout the survey. Section 3 describes the contributions of deep learning to canonical tasks in medical image analysis: classification, detection, segmentation, registration, retrieval, image generation and enhancement. Section 4 discusses obtained results and open challenges in different application areas: neuro, ophthalmic, pulmonary, digital pathology and cell imaging, breast, cardiac, abdominal, musculoskeletal, and remaining miscellaneous applications. We end with a summary, a critical discussion and an outlook for future research.

本文剩余部分组织如下。第2部分，我们介绍了用于医学图像分析的主要深度学习技术，以及本文中参考的这些技术。第3部分描述了深度学习对医学图像处理中经典任务的贡献：分类，检测，分割，配准，检索，图像生成和增强。第4部分讨论了不同应用领域中已有的结果和开放的挑战：脑神经，眼科，肺部，数字病理学和细胞成像，乳腺，心脏病，腹部，肌肉骨骼，和其余各种应用。最后是一个总结，一个关键的讨论，和未来研究的展望。

Figure 1: Breakdown of the papers included in this survey in year of publication, task addressed (Section 3), imaging modality, and application area (Section 4). The number of papers for 2017 has been extrapolated from the papers published in January.

## 2. Overview of deep learning methods 深度学习方法概览

The goal of this section is to provide a formal introduction and definition of the deep learning concepts, techniques and architectures that we found in the medical image analysis papers surveyed in this work.

本节的目的是，给出我们发现的，在本文中综述的，在医学图像分析中的深度学习的概念、技术和架构的正式介绍和定义。

### 2.1. Learning algorithms 学习算法

Machine learning methods are generally divided into supervised and unsupervised learning algorithms, although there are many nuances. In supervised learning, a model is presented with a dataset D = {x, y}$_{n=1}^N$ of input features x and label y pairs, where y typically represents an instance of a fixed set of classes. In the case of regression tasks y can also be a vector with continuous values. Supervised training typically amounts to finding model parameters Θ that best predict the data based on a loss function L(y, ŷ). Here ŷ denotes the output of the model obtained by feeding a data point x to the function f(x; Θ) that represents the model.

机器学习方法一般可以分为监督学习算法和无监督学习算法，其中有很多细微差别。在监督学习中，将数据集D = {x, y}$_{n=1}^N$送入模型中，其中x是输入特征，y是标签，典型的y值为固定类别集合的值。在回归任务的情况中，y也可以是连续值的向量。监督训练一般是寻找模型参数Θ，基于损失函数L(y, ŷ)可以对数据的预测结果达到最佳。这里ŷ表示将数据点x送入代表模型的函数f(x; Θ)中，得到的输出。

Unsupervised learning algorithms process data without labels and are trained to find patterns, such as latent subspaces. Examples of traditional unsupervised learning algorithms are principal component analysis and clustering methods. Unsupervised training can be performed under many different loss functions. One example is reconstruction loss L(x, x̂) where the model has to learn to reconstruct its input, often through a lower-dimensional or noisy representation.

无监督学习算法不用标签来处理数据，算法训练后以寻找模式，如潜在子空间。传统的无监督学习算法的例子是PCA和聚类算法。无监督训练可以用很多很多损失函数进行。一个例子是重建损失L(x, x̂)，其中模型要重建其输入，通常是通过低维或含噪的表示。

### 2.2. Neural Networks 神经网络

Neural networks are a type of learning algorithm which forms the basis of most deep learning methods. A neural network comprises of neurons or units with some activation a and parameters Θ = {W, B}, where W is a set of weights and B a set of biases. The activation represents a linear combination of the input x to the neuron and the parameters, followed by an element-wise non-linearity σ(·), referred to as a transfer function:

神经网络是一种学习算法，是大多数深度学习算法的基础。一个神经网络由神经元组成，包含一些激活a和参数Θ = {W, B}，其中W是权重集合，B是偏置集合。激活是神经元输入x和参数的线性组合，经过逐元素的非线性函数σ(·)的计算的结果，这称为转移函数：

$$a = σ(w^T x + b)$$(1)

Typical transfer functions for traditional neural networks are the sigmoid and hyperbolic tangent function. The multi-layered perceptrons (MLP), the most well-known of the traditional neural networks, have several layers of these transformations:

传统神经网络的典型转移函数为sigmoid和双曲余弦函数。多层感知机(MLP)是最著名的传统神经网络，有几层这种转换：

$$f(x; Θ) = σ(W^T σ(W^T . . . σ(W^T x + b)) + b)$$(2)

Here, W is a matrix comprising of columns $w_k$, associated with activation k in the output. Layers in between the input and output are often referred to as ’hidden’ layers. When a neural network contains multiple hidden layers it is typically considered a ’deep’ neural network, hence the term ’deep learning’.

这里W是一个矩阵，由列$w_k$组成，这与输出的激活k相关。输入与输出层之间的层通常称为隐含层。当一个神经网络包含多个隐含层时，一般就称为深度神经网络，因此就有了深度学习的术语。

At the final layer of the network the activations are mapped to a distribution over classes P(y|x; Θ) through a softmax function: 在网络的最后一层，激活通过softmax函数映射到一个类别分布P(y|x; Θ)：

$$P(y|x;Θ) = softmax(x;Θ) = \frac {e^{w_i^T x}+b_i} {\sum_{k=1}^K e^{w_k^T x}+b_k}$$(3)

where $w_i$ indicates the weight vector leading to the output node associated with class i. A schematic representation of three-layer MLP is shown in Figure 2. 其中$w_i$表示与类别i相关的权重向量。三层的MLP如图2所示。

Maximum likelihood with stochastic gradient descent is currently the most popular method to fit parameters Θ to a dataset D. In stochastic gradient descent a small subset of the data, a mini-batch, is used for each gradient update instead of the full data set. Optimizing maximum likelihood in practice amounts to minimizing the negative log-likelihood:

随机梯度下降的最大似然是目前将参数Θ拟合到数据集D的最常用方法。在随机梯度下降中，数据的一个小子集，即一个mini-batch，用于每个梯度的更新。最优化最大似然，在实际中就是最小化下面的负log自然函数：

$$argmin_Θ -\sum_{n=1}^N log[P(y_n|x_n;Θ)]$$(4)

This results in the binary cross-entropy loss for two-class problems and the categorical cross-entropy for multi-class tasks. A downside of this approach is that it typically does not optimize the quantity we are interested in directly, such as area under the receiver-operating characteristic (ROC) curve or common evaluation measures for segmentation, such as the Dice coefficient.

这得到了两类问题的二值交叉熵损失，和多类任务的类别交叉熵。这种方法的一个缺点是，没有直接优化我们感兴趣的量，比如ROC曲线下的区域面积，或分割的常用评估标准，如Dice系数。

For a long time, deep neural networks (DNN) were considered hard to train efficiently. They only gained popularity in 2006 (Bengio et al., 2007; Hinton and Salakhutdinov, 2006; Hinton et al., 2006) when it was shown that training DNNs layer-by-layer in an unsupervised manner (pre-training), followed by supervised fine-tuning of the stacked network, could result in good performance. Two popular architectures trained in such a way are stacked auto-encoders (SAEs) and deep belief networks (DBNs). However, these techniques are rather complex and require a significant amount of engineering to generate satisfactory results.

很长时间以来，一直认为深度神经网络(DNN)很难高效训练。直到2006年以后才逐渐流行起来，那个时候发现，一层一层的采用无监督的方式训练DNNs（预训练），然后对堆叠的网络进行有监督的精调，可以得到很好的性能。这种方式训练得到了两种流行的架构，即堆叠自动编码机(SAEs)，和深度置信网络(DBNs)。但是，这些技术非常复杂，需要大量的工程工作以得到满意的结果。

Currently, the most popular models are trained end-to-end in a supervised fashion, greatly simplifying the training process. The most popular architectures are convolutional neural networks (CNNs) and recurrent neural networks (RNNs). CNNs are currently most widely used in (medical) image analysis, although RNNs are gaining popularity. The following sections will give a brief overview of each of these methods, starting with the most popular ones, and discussing their differences and potential challenges when applied to medical problems.

现在，最流行的模型是以监督方式进行端到端的训练，极大的简化了训练过程。最受欢迎的架构是CNNs和RNNs。CNNs目前在（医学）图像处理中应用最为广泛，RNNs也应用越来越多。下面会简要回顾一下每种方法，从最受欢迎的开始，当应用于医学问题时，会讨论其差异和可能的挑战。

### 2.3. Convolutional Neural Networks (CNNs)

There are two key differences between MLPs and CNNs. First, in CNNs weights in the network are shared in such a way that it the network performs convolution operations on images. This way, the model does not need to learn separate detectors for the same object occurring at different positions in an image, making the network equivariant with respect to translations of the input. It also drastically reduces the amount of parameters (i.e. the number of weights no longer depends on the size of the input image) that need to be learned. An example of a 1D CNN is shown in Figure 2.

MLPs和CNNs有两点关键的差异。首先，在CNNs中，网络中的权重是共享的，其形式就是网络对图像进行卷积运算。这样，模型对于图像中不同位置中的同样目标，不需要另外学习别的检测器，使得网络对于输入的平移是等变的。这还使得需要学习的参数数量急剧下降（即，权重数量与输入图像大小无关）。1D CNN的一个例子如图2所示。

At each layer, the input image is convolved with a set of K kernels W = {$W_1, W_2, . . ., W_K$} and added biases B = {$b_1, . . ., b_K$}, each generating a new feature map $X_k$. These features are subjected to an element-wise non-linear transform σ(·) and the same process is repeated for every convolutional layer l:

在每一层，输入图像与K个滤波核进行卷积W = {$W_1, W_2, . . ., W_K$}，并加上偏置B = {$b_1, . . ., b_K$}，与每个核的卷积都生成一个新的特征图$X_k$。这些特征是通过一个逐元素的非线性变换σ(·)得到的，对每个卷积层l都重复相同的过程：

$$X_k^l = σ(W_k^{l-1} * X^{l-1} + b_k^{l-1})$$(5)

The second key difference between CNNs and MLPs, is the typical incorporation of pooling layers in CNNs, where pixel values of neighborhoods are aggregated using a permutation invariant function, typically the max or mean operation. This induces a certain amount of translation invariance and again reduces the amount of parameters in the network. At the end of the convolutional stream of the network, fully-connected layers (i.e. regular neural network layers) are usually added, where weights are no longer shared. Similar to MLPs, a distribution over classes is generated by feeding the activations in the final layer through a softmax function and the network is trained using maximum likelihood.

CNNs与MLPs的第二个关键不同之处，是在CNNs中一般都会使用池化层，其中邻域的像素值进行聚合运算，使用的是一个对排列方式不变的函数，典型的是求最大或求均值运算。这带来了一定的平移不变性，也再次降低了网络的参数数量。在网络卷积的最后，通常会加入一些全连接层（即，常规的神经网络层），其中的权重不再进行共享。与MLPs类似的是，将最终层的激活值送入softmax函数，就会得到类别概率分布，然后网络使用最大似然进行训练。

### 2.4. Deep CNN Architectures

Given the prevalence of CNNs in medical image analysis, we elaborate on the most common architectures and architectural differences among the widely used models. 鉴于CNNs在医学图像分析中很流行，我们列举一下广泛使用的模型中，那些最常用的架构及其差异。

#### 2.4.1. General classification architectures 一般分类架构

LeNet (LeCun et al., 1998) and AlexNet (Krizhevsky et al., 2012), introduced over a decade later, were in essence very similar models. Both networks were relatively shallow, consisting of two and five convolutional layers, respectively, and employed kernels with large receptive fields in layers close to the input and smaller kernels closer to the output. AlexNet did incorporate rectified linear units instead of the hyperbolic tangent as activation function.

LeNet和AlexNet实际上是很相似的模型。两种网络相对较浅，分别包含2个和5个卷积层，在与输入接近的层中使用了较大感受野的卷积核，在与输出接近的层中使用了较小的卷积核。AlexNet使用了ReLU，而没有使用双曲余弦作为激活函数。

After 2012 the exploration of novel architectures took off, and in the last three years there is a preference for far deeper models. By stacking smaller kernels, instead of using a single layer of kernels with a large receptive field, a similar function can be represented with less parameters. These deeper architectures generally have a lower memory footprint during inference, which enable their deployment on mobile computing devices such as smartphones. Simonyan and Zisserman (2014) were the first to explore much deeper networks, and employed small, fixed size kernels in each layer. A 19-layer model often referred to as VGG19 or OxfordNet won the ImageNet challenge of 2014.

从2012年开始，对新架构的探索开始了，在过去三年，更倾向于使用深的多的模型。通过堆叠更小的卷积核，而不是使用单层大感受野的卷积核，这样可以使用更少的参数得到类似的效果。这些更深的架构在推理的时候占用的内存通常更少，所以可以在移动计算设备上得以部署，比如手机。Simonyan等是第一个探索这种深的多的模型的，在每一层中使用了小型的固定大小的卷积核。2014年，这种19层的模型，通常称为VGG-19，或OxfordNet赢得了ImageNet挑战赛。

On top of the deeper networks, more complex building blocks have been introduced that improve the efficiency of the training procedure and again reduce the amount of parameters. Szegedy et al. (2014) introduced a 22-layer network named GoogLeNet, also referred to as Inception, which made use of so-called inception blocks (Lin et al., 2013), a module that replaces the mapping defined in Eq. (5) with a set of convolutions of different sizes. Similar to the stacking of small kernels, this allows a similar function to be represented with less parameters. The ResNet architecture (He et al., 2015) won the ImageNet challenge in 2015 and consisted of so-called ResNet-blocks. Rather than learning a function, the residual block only learns the residual and is thereby pre-conditioned towards learning mappings in each layer that are close to the identity function. This way, even deeper models can be trained effectively.

在这些更深的网络之上，引入了更复杂的模块，改进了训练过程的效率，进一步降低了参数数量。Szegedy等提出了一种22层的网络，称为GoogLeNet，也称为Inception，使用了inception模块，该模块使用几种不同大小的卷积的集合，替换了式(5)定义的映射。与小核心堆叠类似，这样可以用更少的参数表达类似的功能。ResNet架构赢得了2015年的ImageNet挑战，包含了所谓的ResNet模块。残差模块学习的不是函数，而是只学习残差，因此预先设定了每层都学习的接近于恒等函数。这样，即使更深的模型也可以得到高效的训练。

Since 2014, the performance on the ImageNet benchmark has saturated and it is difficult to assess whether the small increases in performance can really be attributed to ’better’ and more sophisticated architectures. The advantage of the lower memory footprint these models provide is typically not as important for medical applications. Consequently, AlexNet or other simple models such as VGG are still popular for medical data, though recent landmark studies all use a version of GoogleNet called Inception v3 (Gulshan et al., 2016; Esteva et al., 2017; Liu et al., 2017). Whether this is due to a superior architecture or simply because the model is a default choice in popular software packages is again difficult to assess.

从2014年开始，ImageNet基准测试的性能就饱和了，很难确定这些性能的小幅提升是真的更好了，可能只是因为结构更复杂了。这些模型使用的内存更少，对于医学应用来说也不是很重要。结果是，AlexNet或其他简单的模型如VGG在医疗数据处理中仍然很受欢迎，而最近的研究都使用Inception v3。这是因为这是一种更先进的架构，或只是因为这个模型是流行的软件包的默认设置，也很难评估。

#### 2.4.2. Multi-stream architectures 多流架构

The default CNN architecture can easily accommodate multiple sources of information or representations of the input, in the form of channels presented to the input layer. This idea can be taken further and channels can be merged at any point in the network. Under the intuition that different tasks require different ways of fusion, multi-stream architectures are being explored. These models, also referred to as dual pathway architectures (Kamnitsas et al., 2017), have two main applications at the time of writing: (1) multi-scale image analysis and (2) 2.5D classification; both relevant for medical image processing tasks.

默认的CNN架构可以很容易的适应多源信息输入或多源表示输入，只需要都成为输入层的通道就可以了。这种思想可以进一步发展，通道可以在网络中的任一点处合并。直观上可以感觉到，不同的任务需要不同的融合方法，于是就探索了多流架构。这些模型，也称为双路架构，目前有两个主要的应用：(1)多尺度图像分析，和(2) 2.5D分类；这两个都与医学图像处理任务相关。

For the detection of abnormalities, context is often an important cue. The most straightforward way to increase context is to feed larger patches to the network, but this can significantly increase the amount of parameters and memory requirements of a network. Consequently, architectures have been investigated where context is added in a down-scaled representation in addition to high resolution local information. To the best of our knowledge, the multi-stream multi-scale architecture was first explored by Farabet et al. (2013), who used it for segmentation in natural images. Several medical applications have also successfully used this concept (Kamnitsas et al., 2017; Moeskops et al., 2016a; Song et al., 2015; Yang et al., 2016c).

在畸形检测中，上下文通常是一个重要的线索。增加上下文最直接的方法是将更大的图像块送入网络，但这会显著的增加网络的参数数量和内存需求。结果是，可以将上下文表示为以下采样的表示与高分辨率的局部信息一起，这样的架构进行了研究。据我们所知，多流多尺度框架首先由Farabet等提出，用于自然图像的分割。一些医疗应用也成功的使用了这个概念。

As so much methodology is still developed on natural images, the challenge of applying deep learning techniques to the medical domain often lies in adapting existing architectures to, for instance, different input formats such as three-dimensional data. In early applications of CNNs to such volumetric data, full 3D convolutions and the resulting large amount of parameters were circumvented by dividing the Volume of Interest (VOI) into slices which are fed as different streams to a network. Prasoon et al. (2013) were the first to use this approach for knee cartilage segmentation. Similarly, the network can be fed with multiple angled patches from the 3D-space in a multi-stream fashion, which has been applied by various authors in the context of medical imaging (Roth et al., 2016b; Setio et al., 2016). These approaches are also referred to as 2.5D classification.

由于在自然图像处理中提出了很多方法，所以将深度学习技术应用于医疗领域的挑战，其实是将现有的架构改造，比如适用于不同的输入格式，如三维数据。在早期的CNNs在体结构数据应用中，通过将VOI分割成切片，分别送入不同流的网络，防止使用完整的3D卷积以及得到的大量参数。Prasoon等是第一个使用这种方法进行膝软骨分割的。类似的，可以用3D数据的多角度切片以多流的方式送入网络，不同的作者将其应用于不同的医疗图像处理领域。这些方法也被称为2.5D分类。

#### 2.4.3. Segmentation Architectures 分割架构

Segmentation is a common task in both natural and medical image analysis and to tackle this, CNNs can simply be used to classify each pixel in the image individually, by presenting it with patches extracted around the particular pixel. A drawback of this naive ’sliding-window’ approach is that input patches from neighboring pixels have huge overlap and the same convolutions are computed many times. Fortunately, the convolution and dot product are both linear operators and thus inner products can be written as convolutions and vice versa. By rewriting the fully connected layers as convolutions, the CNN can take input images larger than it was trained on and produce a likelihood map, rather than an output for a single pixel. The resulting ’fully convolutional network’ (fCNN) can then be applied to an entire input image or volume in an efficient fashion.

分割是自然图像和医学图像处理中常见的任务，CNNs可以直接用于对图像中的每个像素进行分类，只要将特定像素周围的图像块送入网络中就可以。这种简单的滑窗方法的缺点是，相邻像素周围的输入图像有很大的重叠，相同的卷积会计算很多次。幸运的是，卷积和点乘都是线性算子，因此内积也可以写为卷积，反之也是。通过讲全连接层重写为卷积层，CNN的输入图像大小可以大于训练时的输入，并生成一个似然图，而不是输出单个像素。得到的“全卷积网络”fCNN可以用于整个输入图像或体，效率也很高。

However, because of pooling layers, this may result in output with a far lower resolution than the input. ’Shift-and-stitch’ (Long et al., 2015) is one of several methods proposed to prevent this decrease in resolution. The fCNN is applied to shifted versions of the input image. By stitching the result together, one obtains a full resolution version of the final output, minus the pixels lost due to the ’valid’ convolutions.

但是，由于池化层，这会导致输出比输入的分辨率要低好多。“shift-and-stitch”是防止分辨率降低的几种方法之一。fCNN应用于输入图像的平移版。通过将结果缝合起来，就可以得到最终输出的完整分辨率版，但是由于卷积时有valid区域，所以要去掉一些像素丢失。

Ronneberger et al. (2015) took the idea of the fCNN one step further and proposed the U-net architecture, comprising a ’regular’ fCNN followed by an upsampling part where ’up’-convolutions are used to increase the image size, coined contractive and expansive paths. Although this is not the first paper to introduce learned upsampling paths in convolutional neural networks (e.g. Long et al. (2015)), the authors combined it with so called skip-connections to directly connect opposing contracting and expanding convolutional layers. A similar approach was used by Çiçek et al. (2016) for 3D data. Milletari et al. (2016b) proposed an extension to the U-Net layout that incorporates ResNet-like residual blocks and a Dice loss layer, rather than the conventional cross-entropy, that directly minimizes this commonly used segmentation error measure.

Ronneberger等将fCNN的思想进一步发展，提出了U-net架构，由常规fCNN后加上一个上采样部分构成，其中上卷积部分用于增加图像大小，命名为收缩通道和扩张通道。虽然这不是第一篇提出在CNN中学习上采样路径的文章，但作者将其与跳跃连接结合到一起，直接将相反的收缩卷积层和扩张卷积层连接了起来。Çiçek等将类似的方法用于3D数据。Milletari提出U-Net的扩展，纳入了类似ResNet的残差模块和Dice损失层，而不是传统的交叉熵损失，直接最小化这种常见的分割误差度量。

### 2.5. Recurrent Neural Networks (RNNs)

Traditionally, RNNs were developed for discrete sequence analysis. They can be seen as a generalization of MLPs because both the input and output can be of varying length, making them suitable for tasks such as machine translation where a sentence of the source and target language are the input and output. In a classification setting, the model learns a distribution over classes $P(y|x_1, x_2, . . ., x_T; Θ)$ given a sequence $x_1, x_2, . . ., x_T$, rather than a single input vector x.

传统上，RNNs的提出是进行离散序列分析的。这可以看做是MLPs的泛化，因为输入和输出的长度都是可变的，使其适合进行机器翻译之类的任务，输入是一个句子，输出是目标语言的句子。在分类的设置中，给定输入徐略$x_1, x_2, . . ., x_T$，模型学习到的不同的类别的概率分布$P(y|x_1, x_2, . . ., x_T; Θ)$，而不是单个输入向量x。

The plain RNN maintains a latent or hidden state h at time t that is the output of a non-linear mapping from its input $x_t$ and the previous state $h_{t−1}$: 普通的RNN在t时刻有隐藏状态h，是输入$x_t$和之前的状态$h_{t−1}$的非线性映射：

$$h_t = σ(Wx_t + Rh_{t−1} + b)$$(6)

where weight matrices W and R are shared over time. For classification, one or more fully connected layers are typically added followed by a softmax to map the sequence to a posterior over the classes. 其中权重矩阵W和R是不同时刻共享的。对于分类，会增加一个或多个全连接层，然后是一个softmax层，将序列映射为分类别的后验概率。

$$P(y|x_1, x_2, . . ., x_T; Θ) = softmax(h_T; W_{out}, b_{out})$$(7)

Since the gradient needs to be backpropagated from the output through time, RNNs are inherently deep (in time) and consequently suffer from the same problems with training as regular deep neural networks (Bengio et al., 1994). To this end, several specialized memory units have been developed, the earliest and most popular being the Long Short Term Memory (LSTM) cell (Hochreiter and Schmidhuber, 1997). The Gated Recurrent Unit (Cho et al., 2014) is a recent simplification of the LSTM and is also commonly used.

由于梯度需要从输出按时间反向传播，RNN天然就是深度的，所以也有常规深度神经网络训练存在的问题。为此，提出了几种专用的存储单元，最早的和最流行的是LSTM单元。GRU是最近对LSTM的简化，也是常用的。

Although initially proposed for one-dimensional input, RNNs are increasingly applied to images. In natural images ’pixelRNNs’ are used as autoregressive models, generative models that can eventually produce new images similar to samples in the training set. For medical applications, they have been used for segmentation problems, with promising results (Stollenga et al., 2015) in the MRBrainS challenge.

虽然RNN的提出是用于一维输入的，但也越来越多的用于图像中。在自然图像中，pixelRNNs用于自动回归模型和生成式模型，可以最后生成与训练集中样本类似的图像。对于医学应用，常用于解决分割问题，在MRBrainS挑战中可以得到很有希望的结果。

### 2.6. Unsupervised models

#### 2.6.1. Auto-encoders (AEs) and Stacked Auto-encoders (SAEs)

AEs are simple networks that are trained to reconstruct the input x on the output layer x' through one hidden layer h. They are governed by a weight matrix $W_{x,h}$ and bias $b_{x,h}$ from input to hidden state and $W_{h,x'}$ with corresponding bias $b_{h,x'}$ from the hidden layer to the reconstruction. A non-linear function is used to compute the hidden activation:

AEs是简单的网络，训练用于从输出层x'通过一个隐藏层h重建输入x。其中的参数是从输入到隐藏状态的权重矩阵$W_{x,h}$和偏置$b_{x,h}$，和从隐藏状态到重建的$W_{h,x'}$和对应的偏置$b_{h,x'}$。用了一个非线性函数来计算隐藏激活：

$$h = σ(W_{x,h}x + b_{x,h})$$(8)

Additionally, the dimension of the hidden layer |h| is taken to be smaller than |x|. This way, the data is projected onto a lower dimensional subspace representing a dominant latent structure in the input. Regularization or sparsity constraints can be employed to enhance the discovery process. If the hidden layer had the same size as the input and no further non-linearities were added, the model would simply learn the identity function.

另外，隐藏层|h|的维度通常比输入的维度|x|要小。这样，数据投影到低维度子空间，代表了输入的主要隐藏结构。正则化或稀疏约束可以用于强化发现的过程。如果隐藏层与输入有同样的大小，而且也没有加入其他非线性，那么模型就会直接学习恒等函数。

The denoising auto-encoder (Vincent et al., 2010) is another solution to prevent the model from learning a trivial solution. Here the model is trained to reconstruct the input from a noise corrupted version (typically salt-and-pepper-noise). SAEs (or deep AEs) are formed by placing auto-encoder layers on top of each other. In medical applications surveyed in this work, auto-encoder layer were often trained individually (‘greedily’) after which the full network was fine-tuned using supervised training to make a prediction.

去噪的AE是为防止网络学习了一个无意义的解的另一个解决方案。这里模型训练用于从含噪降质版（一般是椒盐噪声）重建输入。SAEs（或深度AEs）是将AE层堆叠起来形成的。在本文总结的医学应用中，在整个网络使用监督训练进行精调后，AE层通常单独训练，以进行预测。

#### 2.6.2. Restricted Boltzmann Machines (RBMs) and Deep Belief Networks (DBNs)

RBMs (Hinton, 2010) are a type of Markov Random Field (MRF), constituting an input layer or visible layer $x = (x_1, x_2, . . ., x_N)$ and a hidden layer $h = (h_1, h_2, . . ., h_M$) that carries the latent feature representation. The connections between the nodes are bidirectional, so given an input vector x one can obtain the latent feature representation h and also vice versa. As such, the RBM is a generative model, and we can sample from it and generate new data points. In analogy to physical systems, an energy function is defined for a particular state (x, h) of input and hidden units:

RBMs是一类马尔科夫随机场(MRF)，由一个输入层（或可见层）$x = (x_1, x_2, . . ., x_N)$，和一个隐含层$h = (h_1, h_2, . . ., h_M$)组成，隐含层是潜在的特征表示。节点间的连接是双向的，所以给定输入向量x，可以得到潜在的特征表示h，反之亦然。这样，RBM是一个生成式模型，我们可以对其取样，生成新的数据点。与物理系统类比，对输入和隐含单元的特定状态(x,h)定义了一个能量函数：

$$E(x, h) = h^T Wx − c^T x − b^T h$$(9)

with c and b bias terms. The probability of the ‘state’ of the system is defined by passing the energy to an exponential and normalizing: 其中c和b是偏置项。系统状态的概率，通过将能量函数进行指数运算并归一化后定义：

$$p(x,h) = \frac {1}{Z} exp(-E(x,h))$$(10)

Computing the partition function Z is generally intractable. However, conditional inference in the form of computing h conditioned on v or vice versa is tractable and results in a simple formula: 计算分割函数Z一般是很难的。但是，在给定v的条件下计算h，或反之，这样的条件推理是可以进行的，结果是下面这样简单的公式：

$$P(h_j|x) = \frac {1}{1+exp(-b_j-W_jx)}$$(11)

Since the network is symmetric, a similar expression holds for $P(x_i|h)$. 因为这个网络是对称的，对$P(x_i|h)$也有类似的表达式。

DBNs (Bengio et al., 2007; Hinton et al., 2006) are essentially SAEs where the AE layers are replaced by RBMs. Training of the individual layers is, again, done in an unsupervised manner. Final fine-tuning is performed by adding a linear classifier to the top layer of the DBN and performing a supervised optimization.

DBNs实际上就是SAEs，其中AE层替换成了RBM。单独一层的训练还是以无监督的形式进行的。最终的精调，是在DBN的最上层增加了一个线性分类器，然后以监督优化的方式进行的。

#### 2.6.3. Variational Auto-Encoders and Generative Adverserial Networks

Recently, two novel unsupervised architectures were introduced: the variational auto-encoder (VAE) (Kingma and Welling, 2013) and the generative adversarial network (GAN) (Goodfellow et al., 2014). There are no peer-reviewed papers applying these methods to medical images yet, but applications in natural images are promising. We will elaborate on their potential in the discussion.

最近，提出了两种新的无监督架构：变分自动编码机(VAE)和生成式对抗网络(GAN)。还没有文献对这两种方法在医学图像中进行应用，但在自然图像中的应用是很有希望的。我们在讨论中叙述了其潜力。

### 2.7. Hardware and Software

One of the main contributors to steep rise of deep learning has been the widespread availability of GPU and GPU-computing libraries (CUDA, OpenCL). GPUs are highly parallel computing engines, which have an order of magnitude more execution threads than central processing units (CPUs). With current hardware, deep learning on GPUs is typically 10 to 30 times faster than on CPUs.

深度学习的崛起，一个重要的贡献者是GPU和GPU计算库的广泛使用(CUDA, OpenCL)。GPU是高度并行的计算引擎，比CPU的计算线程多了一个数量级。在现有的硬件上，深度学习在GPU上的运算一般比在CPUs上快10到30倍。

Next to hardware, the other driving force behind the popularity of deep learning methods is the wide availability of open source software packages. These libraries provide efficient GPU implementations of important operations in neural networks, such as convolutions; allowing the user to implement ideas at a high level rather than worrying about low-level efficient implementations. At the time of writing, the most popular packages were (in alphabetical order):

除了硬件，深度学习流行的其他推动力，还包括开源软件包的广泛使用。这些库给出了神经网络中重要运算的高效的GPU实现，如卷积；使用户可以在更高层实现其思想，而不是考虑底层的高效实现。在本文写就的时候，使用最广泛的软件包有：

- Caffe (Jia et al., 2014). Provides C++ and Python interfaces, developed by graduate students at UC Berkeley.

- Tensorflow (Abadi et al., 2016). Provides C++ and Python and interfaces, developed by Google and is used by Google research.

- Theano (Bastien et al., 2012). Provides a Python interface, developed by MILA lab in Montreal.

- Torch (Collobert et al., 2011). Provides a Lua interface and is used by, among others, Facebook AI research.

There are third-party packages written on top of one or more of these frameworks, such as Lasagne (https://github.com/Lasagne/Lasagne) or Keras (https://keras.io/). It goes beyond the scope of this paper to discuss all these packages in detail.

## 3. Deep Learning Uses in Medical Imaging

### 3.1. Classification

#### 3.1.1. Image/exam classification

Image or exam classification was one of the first areas in which deep learning made a major contribution to medical image analysis. In exam classification one typically has one or multiple images (an exam) as input with a single diagnostic variable as output (e.g., disease present or not). In such a setting, every diagnostic exam is a sample and dataset sizes are typically small compared to those in computer vision (e.g., hundreds/thousands vs. millions of samples). The popularity of transfer learning for such applications is therefore not surprising.

图像分类/检查分类是深度学习对医学图像分析做出的第一个主要贡献。在检查分类中，输入是一幅/多幅图像（一个检查结果），输出是单个诊断变量（如，存在病症或不存在）。在这样一个设置中，每个诊断检查都是一个样本，数据集大小与计算机视觉中的相比，相对很小（如，几百/几千 vs 上百万样本）。迁移学习在这种应用中是非常流行的。

Transfer learning is essentially the use of pre-trained networks (typically on natural images) to try to work around the (perceived) requirement of large data sets for deep network training. Two transfer learning strategies were identified: (1) using a pre-trained network as a feature extractor and (2) fine-tuning a pre-trained network on medical data. The former strategy has the extra benefit of not requiring one to train a deep network at all, allowing the extracted features to be easily plugged in to existing image analysis pipelines. Both strategies are popular and have been widely applied. However, few authors perform a thorough investigation in which strategy gives the best result. The two papers that do, Antony et al. (2016) and Kim et al. (2016a), offer conflicting results. In the case of Antony et al. (2016), fine-tuning clearly outperformed feature extraction, achieving 57.6% accuracy in multi-class grade assessment of knee osteoarthritis versus 53.4%. Kim et al. (2016a), however, showed that using CNN as a feature extractor outperformed fine-tuning in cytopathology image classification accuracy (70.5% versus 69.1%). If any guidance can be given to which strategy might be most successful, we would refer the reader to two recent papers, published in high-ranking journals, which fine-tuned a pre-trained version of Google’s Inception v3 architecture on medical data and achieved (near) human expert performance (Esteva et al., 2017; Gulshan et al., 2016). As far as the authors are aware, such results have not yet been achieved by simply using pre-trained networks as feature extractors.

迁移学习是利用预训练网络（一般在自然图像中），以绕过深度网络训练需要大型数据集的需要。有两种迁移学习的策略：(1)使用预训练网络作为特征提取器，(2)在医疗数据上精调一个预训练网络。前一个策略的优势是根本不需要训练网络，提取出的特征可以很容易的合并到现有的图像分析流程中。两种策略都非常流行，都得到了广泛的使用。但是，很少有作者进行了彻底的调查，到底哪种策略可以得到最好的结果。有两篇文章进行了这样的试验，但得到了的结果互相冲突。在Antony等的文章中，精调的性能明显超过了特征提取，在膝部骨关节炎的多类别等级评估中，得到了57.6%的准确率，超过了特征提取的53.4%。Kim等的试验则表明，使用CNN作为特征提取器，在细胞病理学图像分类中，超过了精调的性能(70.5% vs 69.1%)。如果想知道哪种策略更成功，我们推荐读者参考最近的两篇文章，在高评分期刊上发表的，在医疗数据上，精调了一个预训练的Inception V3架构网络，得到了（接近）人类专家的性能。要是只使用预训练的网络作为特征提取器，是不会得到这样的结果的。

With respect to the type of deep networks that are commonly used in exam classification, a timeline similar to computer vision is apparent. The medical imaging community initially focused on unsupervised pre-training and network architectures like SAEs and RBMs. The first papers applying these techniques for exam classification appeared in 2013 and focused on neuroimaging. Brosch and Tam (2013), Plis et al. (2014), Suk and Shen (2013), and Suk et al. (2014) applied DBNs and SAEs to classify patients as having Alzheimer’s disease based on brain Magnetic Resonance Imaging (MRI). Recently, a clear shift towards CNNs can be observed. Out of the 47 papers published on exam classification in 2015, 2016, and 2017, 36 are using CNNs, 5 are based on AEs and 6 on RBMs. The application areas of these methods are very diverse, ranging from brain MRI to retinal imaging and digital pathology to lung computed tomography (CT).

关于在检查分类中常用的深度网络类型，很明显，与计算机视觉的时间线是类似的。医学图像团体开始关注的是无监督的预训练网络架构，如SAEs和RBMs。将这种技术应用于检查分类的第一批文章，出现在2013年，关注的是神经细胞成像。一些作者基于脑MRI，将DBNs和SAEs用于将病人分类为有老年痴呆症。最近，大家都在向CNNs转移。2015-2017年发表的关于检查分类的47篇文章中，36篇使用了CNNs，5篇基于AEs，6篇基于RBMs。这些方法的应用领域非常多样，从脑MRI到视网膜成像，和数字病理学，到肺部CT。

In the more recent papers using CNNs authors also often train their own network architectures from scratch instead of using pre-trained networks. Menegola et al. (2016) performed some experiments comparing training from scratch to fine-tuning of pre-trained networks and showed that fine-tuning worked better given a small data set of around a 1000 images of skin lesions. However, these experiments are too small scale to be able to draw any general conclusions from.

最近的文章中，使用CNNs的作者通常从头训练其网络，而不是使用预训练网络。Menegola等进行了一些试验，比较了从头训练的，与从预训练模型精调的，在给定的1000幅皮肤损伤图像上，精调的效果更好一些。但是，这些试验的规模太小，不能得到什么一般性的结论。

Three papers used an architecture leveraging the unique attributes of medical data: two use 3D convolutions (Hosseini-Asl et al., 2016; Payan and Montana, 2015) instead of 2D to classify patients as having Alzheimer; Kawahara et al. (2016b) applied a CNN-like architecture to a brain connectivity graph derived from MRI diffusion-tensor imaging (DTI). In order to do this, they developed several new layers which formed the basis of their network, so-called edge-to-edge, edge-to-node, and node-to-graph layers. They used their network to predict brain development and showed that they outperformed existing methods in assessing cognitive and motor scores.

三篇文章使用的架构，利用了医学数据的独特属性：两篇使用了3D卷积（而不是2D）来分类患者是否患有老年痴呆症；一篇使用了一种类CNN架构，对象是从MRI DTI推导出的脑连接性图。为此，他们提出了几个新的层，形成了其网络的基础，称为edge-to-edge, edge-to-node, node-to-graph层。他们使用其网络预测脑部的发展，表明在评估认知和机动分数中，超过了现有的方法。

Summarizing, in exam classification CNNs are the current standard techniques. Especially CNNs pretrained on natural images have shown surprisingly strong results, challenging the accuracy of human experts in some tasks. Last, authors have shown that CNNs can be adapted to leverage intrinsic structure of medical images.

总结一下，在检查分类中，CNNs是现在标准的技术。尤其是，在自然图像上预训练的CNNs表现出了很好的效果，在一些任务上可以挑战人类专家。最后，作者表明，可以改造CNNs以利用医学图像的内在结构。

#### 3.1.2. Object or lesion classification

Object classification usually focuses on the classification of a small (previously identified) part of the medical image into two or more classes (e.g. nodule classification in chest CT). For many of these tasks both local information on lesion appearance and global contextual information on lesion location are required for accurate classification. This combination is typically not possible in generic deep learning architectures. Several authors have used multi-stream architectures to resolve this in a multi-scale fashion (Section 2.4.2). Shen et al. (2015b) used three CNNs, each of which takes a nodule patch at a different scale as input. The resulting feature outputs of the three CNNs are then concatenated to form the final feature vector. A somewhat similar approach was followed by Kawahara and Hamarneh (2016) who used a multi-stream CNN to classify skin lesions, where each stream works on a different resolution of the image. Gao et al. (2015) proposed to use a combination of CNNs and RNNs for grading nuclear cataracts in slit-lamp images, where CNN filters were pre-trained. This combination allows the processing of all contextual information regardless of image size. Incorporating 3D information is also often a necessity for good performance in object classification tasks in medical imaging. As images in computer vision tend to be 2D natural images, networks developed in those scenarios do not directly leverage 3D information. Authors have used different approaches to integrate 3D in an effective manner with custom architectures. Setio et al. (2016) used a multi-stream CNN to classify points of interest in chest CT as a nodule or non-nodule. Up to nine differently oriented patches extracted from the candidate were used in separate streams and merged in the fully-connected layers to obtain the final classification output. In contrast, Nie et al. (2016c) exploited the 3D nature of MRI by training a 3D CNN to assess survival in patients suffering from high-grade gliomas.

目标分类通常关注的是将医学图像中的一小部分（之前检测得到的）分类为两个或多个类别（如，胸部CT中的小瘤分类）。很多这些任务中，要进行准确的分类，都需要损伤外表的局部信息和损伤位置的全局上下文信息。在通用深度学习架构中，这种组合一般不太可能。几个作者使用了多流架构以多尺度的方式解决了这个问题（2.4.2节）。Shen等使用了三个CNNs，每个网络以不同尺度下的小瘤图像块作为输入。这三个CNNs得到的特征输出，拼接到一起，以形成最终的特征向量。Kawahara等沿用了有些类似的方法，使用一个多流CNN来分类皮肤损伤，其中每个流处理不同分辨率的图像。Gao等提出使用CNNs和RNNs的组合，对slit-lamp图像中的白内障细胞核进行评分，其中CNN滤波器是预训练。这种组合在各种图像大小下都可以处理所有的上下文信息。结合3D信息，通常也是医学图像处理中目标分类任务中得到好性能所必须的。因为计算机视觉中的图像一般是2D自然图像，这些场景下提出的网络不会直接利用3D信息。一些作者利用了不同的方法来将3D信息整合，得到了自定义的架构。Setio等使用了一种多流CNN，对胸部CT的感兴趣点进行分类成小瘤或非小瘤。从候选中提取出最多9个不同方向的图像块，用于分离的流中，然后融合到全连接层，以得到最终的分类结果。比较之下，Nie等研究了MRI的3D本性，训练了一个3D CNN，评估患有严重神经胶细胞瘤的病人的存活率。

Almost all recent papers prefer the use of end-to-end trained CNNs. In some cases other architectures and approaches are used, such as RBMs (van Tulder and de Bruijne, 2016; Zhang et al., 2016c), SAEs (Cheng et al., 2016a) and convolutional sparse auto-encoders (CSAE) (Kallenberg et al., 2016). The major difference between CSAE and a classic CNN is the usage of unsupervised pre-training with sparse auto-encoders.

几乎所有最近的文章都喜欢使用端到端训练的CNNs。在一些情况下也用到了其他架构和方法，比如RBMs，SAEs和卷积稀疏自动编码机。CSAE和经典CNN的主要区别是，使用了无监督的预训练和稀疏自动编码机。

An interesting approach, especially in cases where object annotation to generate training data is expensive, is the integration of multiple instance learning (MIL) and deep learning. Xu et al. (2014) investigated the use of a MIL-framework with both supervised and unsupervised feature learning approaches as well as handcrafted features. The results demonstrated that the performance of the MIL-framework was superior to handcrafted features, which in turn closely approaches the performance of a fully supervised method. We expect such approaches to be popular in the future as well, as obtaining high-quality annotated medical data is challenging.

在一些情况下，用目标标注来生成训练数据是非常昂贵的，这时一个有趣的方法就是整合多实例学习(MIL)和深度学习。Xu等研究了在MIL框架中，使用有监督和无监督特征学习方法，和手工设计的特征。结果表明，MIL框架的性能是比手工设计的特征要好的，很接近全监督方法的性能。我们希望这种方法在未来也会流行，因为得到高质量标注的医学图像是很有挑战的。

Overall, object classification sees less use of pretrained networks compared to exam classifications, mostly due to the need for incorporation of contextual or three-dimensional information. Several authors have found innovative solutions to add this information to deep networks with good results, and as such we expect deep learning to become even more prominent for this task in the near future.

总体上来说，目标分类与检查分类相比，使用更少的预训练网络，主要是因为需要整合上下文信息或三维信息。几位作者提出了一些新方法，将这些信息加入到深度网络中，得到了很好的结果，所以我们希望，深度学习在未来在这个任务中会取得越来越显著的位置。

### 3.2. Detection

#### 3.2.1. Organ, region and landmark localization

Anatomical object localization (in space or time), such as organs or landmarks, has been an important preprocessing step in segmentation tasks or in the clinical workflow for therapy planning and intervention. Localization in medical imaging often requires parsing of 3D volumes. To solve 3D data parsing with deep learning algorithms, several approaches have been proposed that treat the 3D space as a composition of 2D orthogonal planes. Yang et al. (2015) identified landmarks on the distal femur surface by processing three independent sets of 2D MRI slices (one for each plane) with regular CNNs. The 3D position of the landmark was defined as the intersection of the three 2D slices with the highest classification output. de Vos et al. (2016b) went one step further and localized regions of interest (ROIs) around anatomical regions (heart, aortic arch, and descending aorta) by identifying a rectangular 3D bounding box after 2D parsing the 3D CT volume. Pretrained CNN architectures, as well as RBM, have been used for the same purpose (Cai et al., 2016b; Chen et al., 2015b; Kumar et al., 2016), overcoming the lack of data to learn better feature representations. All these studies cast the localization task as a classification task and as such generic deep learning architectures and learning processes can be leveraged.

身体结构目标定位（空间或时间上的），比如器官或特征点，是分割任务的一个重要预处理步骤，或诊断计划和诊断干预的临床工作流的一个重要预处理步骤。医学图像中的定位通常需要解析3D体结构。为用深度学习算法求解3D数据解析问题，提出了几种方法，将3D空间视为2D正交平面的组合。Yang等使用常规CNNs处理三个独立集合的2D MRI切片（每个平面一个集合），识别末股骨平面上的关键点。关键点的3D位置定义为，三个最高分类输出的2D切片的相交位置。de Vos等更进一步，在身体结构区域（心脏，主动脉弓，和降主动脉）附近定位感兴趣区域(ROIs)，在对3D CT体进行2D解析后，检测一个3D矩形边界框。预训练CNN架构和RBM曾都用于相同的目的，克服数据缺少的问题，学习到更好的特征表示。所有这些研究将定位任务当成一个分类任务，这样就可以利用通用的深度学习架构和学习过程。

Other authors try to modify the network learning process to directly predict locations. For example, Payer et al. (2016) proposed to directly regress landmark locations with CNNs. They used landmark maps, where each landmark is represented by a Gaussian, as ground truth input data and the network is directly trained to predict this landmark map. Another interesting approach was published by Ghesu et al. (2016a), in which reinforcement learning is applied to the identification of landmarks. The authors showed promising results in several tasks: 2D cardiac MRI and ultrasound (US) and 3D head/neck CT.

其他作者尝试修改网络学习过程，以直接预测位置。如，Payer等提出直接用CNNs回归关键点位置。他们使用特征点图，其中每个特征点表示为一个高斯函数作为真值输入数据，网络直接训练用于预测这个关键点图。另一种有趣的方法是Ghesu等人的，用强化学习来检测关键点。作者在几个任务中都给出了很有希望的结果，如2D心脏MRI和超声，和3D头部、颈部CT。

Due to its increased complexity, only a few methods addressed the direct localization of landmarks and regions in the 3D image space. Zheng et al. (2015) reduced this complexity by decomposing 3D convolution as three one-dimensional convolutions for carotid artery bifurcation detection in CT data. Ghesu et al. (2016b) proposed a sparse adaptive deep neural network powered by marginal space learning in order to deal with data complexity in the detection of the aortic valve in 3D transesophageal echocardiogram.

由于复杂度变高，只有几种方法处理3D图像中关键点和区域的直接定位问题。Zheng等通过将3D卷积分解成3个一维卷积，降低了复杂度，处理了CT数据中的颈动脉分叉检测。Ghesu等提出一种稀疏自适应DNN，可以学习边际空间，以处理3D食管超声心动图的主动脉瓣中的数据高度复杂性。

CNNs have also been used for the localization of scan planes or key frames in temporal data. Baumgartner et al. (2016) trained CNNs on video frame data to detect up to 12 standardized scan planes in mid-pregnancy fetal US. Furthermore, they used saliency maps to obtain a rough localization of the object of interest in the scan plan (e.g. brain, spine). RNNs, particularly LSTM-RNNs, have also been used to exploit the temporal information contained in medical videos, another type of high dimensional data. Chen et al. (2015a), for example, employed LSTM models to incorporate temporal information of consecutive sequence in US videos for fetal standard plane detection. Kong et al. (2016) combined an LSTM-RNN with a CNN to detect the end-diastole and end-systole frames in cine-MRI of the heart.

CNNs还用于时序数据的扫描平面定位或关键帧定位。Baumgartner等在视频帧数据中训练CNNs，以在中孕期胎儿超声中检测最多12个标准的扫描平面。而且，他们使用显著性图以得到感兴趣目标在扫描平面（如，脑部，脊椎）中的大致定位。RNNs，特别是LSTM-RNNs，也曾用于挖掘医疗视频中的时间信息，这是另一种高维数据。例如，Chen等采用LSTM模型，利用超声视频的连续序列的时间信息，检测胎儿的标准平面。Kong等将LSTM-RNN与CNN结合起来，以检测心脏影像MRI中的舒张末期帧和收缩末期帧。

Concluding, localization through 2D image classification with CNNs seems to be the most popular strategy overall to identify organs, regions and landmarks, with good results. However, several recent papers expand on this concept by modifying the learning process such that accurate localization is directly emphasized, with promising results. We expect such strategies to be explored further as they show that deep learning techniques can be adapted to a wide range of localization tasks (e.g. multiple landmarks). RNNs have shown promise in localization in the temporal domain, and multi-dimensional RNNs could play a role in spatial localization as well.

总结起来，通过CNN进行2D图像分类的定位，似乎是最受欢迎的策略，可以在定位器官、区域和关键点中得到很好的结果。但是，几篇最近的文章拓展了这个概念，修改了学习过程，这样可以直接强调精确的定位过程，得到很有希望的结果。我们期望这样的策略得到更多的探索，因为这表明，深度学习技术经过修改适用于非常多的定位任务（如，多关键点）。RNNs在时序领域的定位中很有应用，多维RNN也可以用于空间定位。

#### 3.2.2. Object or lesion detection

The detection of objects of interest or lesions in images is a key part of diagnosis and is one of the most labor-intensive for clinicians. Typically, the tasks consist of the localization and identification of small lesions in the full image space. There has been a long research tradition in computer-aided detection systems that are designed to automatically detect lesions, improving the detection accuracy or decreasing the reading time of human experts. Interestingly, the first object detection system using CNNs was already proposed in 1995, using a CNN with four layers to detect nodules in x-ray images (Lo et al., 1995).

感兴趣目标的检测，或损伤检测，是诊断的关键部分，是临床医师最繁重的工作之一。一般，这个任务包括，在整幅图像中定位并识别小的损伤。计算机辅助检测系统有很长的研究历史，用于自动检测损伤，改进检测准确率或降低人类专家读图的时间。有趣的是，第一个使用CNNs的目标检测系统在1995年就提出来了，使用了一个4层的CNN检测X涉嫌图像中的小瘤。

Most of the published deep learning object detection systems still uses CNNs to perform pixel (or voxel) classification, after which some form of post processing is applied to obtain object candidates. As the classification task performed at each pixel is essentially object classification, CNN architecture and methodology are very similar to those in section 3.1.2. The incorporation of contextual or 3D information is also handled using multi-stream CNNs. In Section 2.4.2, for example by Barbu et al. (2016) and Roth et al. (2016b). Teramoto et al. (2016) used a multi-stream CNN to integrate CT and Positron Emission Tomography (PET) data. Dou et al. (2016c) used a 3D CNN to find micro-bleeds in brain MRI. Last, as the annotation burden to generate training data can be similarly significant compared to object classification, weakly-supervised deep learning has been explored by Hwang and Kim (2016), who adopted such a strategy for the detection of nodules in chest radiographs and lesions in mammography.

大多数已经发表的深度学习目标检测系统，仍然使用CNNs来进行像素（或体素）分类，然后进行一定的后处理，得到候选目标。由于对每个像素的分类任务实质上仍然是目标分类，CNN架构和方法与3.1.2节中的非常类似。上下文信息或3D信息的利用，可以使用多流CNNs处理。如在2.4.2节中，几篇文献使用多流CNN将CT数据和PET数据进行综合处理。Dou等使用3D CNN在脑MRI中发现微出血。最后，由于生成训练数据的标注负担可能和目标分类一样重，Hwang等探索了弱监督的深度学习，使用了这样一种策略来检测胸透中的小瘤和乳腺中的损伤。

There are some aspects which are significantly different between object detection and object classification. One key point is that because every pixel is classified, typically the class balance is skewed severely towards the non-object class in a training setting. To add insult to injury, usually the majority of the non-object samples are easy to discriminate, preventing the deep learning method to focus on the challenging samples. van Grinsven et al. (2016) proposed a selective data sampling in which wrongly classified samples were fed back to the network more often to focus on challenging areas in retinal images. Last, as classifying each pixel in a sliding window fashion results in orders of magnitude of redundant calculation, fCNNs, as used in Wolterink et al. (2016), are important aspect of an object detection pipeline as well.

目标检测和目标分类有几个方面非常不同。一个关键是，由于每个像素都被分类了，在训练设置中，类别均衡的情况一般会导致非目标的类别特别多。通常大多数非目标样本都很容易区分出来，这样导致深度学习方法就很难聚焦关注那些很有挑战性的样本。van Grinsven等提出选择性数据取样方法，其中错误分类的样本更多的被送入网络，以聚焦视网膜图像中有挑战性的区域。最后，由于以滑窗方式分类每个像素会导致很多冗余计算，fCNNs是非常重要的目标检测方法之一。

Challenges in meaningful application of deep learning algorithms in object detection are thus mostly similar to those in object classification. Only few papers directly address issues specific to object detection like class imbalance/hard-negative mining or efficient pixel/voxel-wise processing of images. We expect that more emphasis will be given to those areas in the near future, for example in the application of multi-stream networks in a fully convolutional fashion.

深度学习目标检测算法的挑战与目标分类的类似。只有很少文章处理目标检测中的特定问题，如类别不均衡，难分样本挖掘，或高效的像素/体素处理。我们期望未来会更多的关注这些领域，比如全卷积形式的多流网络应用。

### 3.3. Segmentation

#### 3.3.1. Organ and substructure segmentation

The segmentation of organs and other substructures in medical images allows quantitative analysis of clinical parameters related to volume and shape, as, for example, in cardiac or brain analysis. Furthermore, it is often an important first step in computer-aided detection pipelines. The task of segmentation is typically defined as identifying the set of voxels which make up either the contour or the interior of the object(s) of interest. Segmentation is the most common subject of papers applying deep learning to medical imaging (Figure 1), and as such has also seen the widest variety in methodology, including the development of unique CNN-based segmentation architectures and the wider application of RNNs.

医学图像中的器官和其他子结构的分割，可以对与体积和形状相关的临床参数进行量化分析，比如心脏或脑部分析。而且，在计算机辅助检测的流程中，这通常是很重要的一步。分割的任务一般定义为确定组成感兴趣目标的边缘或内部的像素/体素。分割是深度学习应用于医学图像分析最常见的文章（图1），因此方法最为多样，包括提出了唯一的基于CNN的分割架构和RNN更广泛的应用。

The most well-known, in medical image analysis, of these novel CNN architectures is U-net, published by Ronneberger et al. (2015) (section 2.4.3). The two main architectural novelties in U-net are the combination of an equal amount of upsampling and downsampling layers. Although learned upsampling layers have been proposed before, U-net combines them with so-called skip connections between opposing convolution and deconvolution layers. This which concatenate features from the contracting and expanding paths. From a training perspective this means that entire images/scans can be processed by U-net in one forward pass, resulting in a segmentation map directly. This allows U-net to take into account the full context of the image, which can be an advantage in contrast to patch-based CNNs. Furthermore, in an extended paper by Çiçek et al. (2016), it is shown that a full 3D segmentation can be achieved by feeding U-net with a few 2D annotated slices from the same volume. Other authors have also built derivatives of the U-net architecture; Milletari et al. (2016b), for example, proposed a 3D-variant of U-net architecture, called V-net, performing 3D image segmentation using 3D convolutional layers with an objective function directly based on the Dice coefficient. Drozdzal et al. (2016) investigated the use of short ResNet-like skip connections in addition to the long skip-connections in a regular U-net.

在医学图像分析中，最著名的CNN架构就是U-Net。U-Net架构的两个主要的创新点是相同数量的上采样和下采样层的结合。虽然学习到的上采样层之前就有提出过，但U-Net将其与所谓的跳跃连接结合到一起，将相反的卷积和解卷积层连接了起来，这将收缩通道和扩张通道的特征拼接到了一起。从训练的角度看，这意味着整个图像/扫描在一个前向过程中就可以为U-Net处理掉，直接得到分割图。这使得U-Net可以考虑图像的整个上下文，这与基于图像块的CNNs相比，是一个优势。而且，在Cicek等的文章中证明了，将同一个体中的几个2D切片标注后送入U-Net，可以得到完整的3D分割。其他作者也构建了衍生的U-Net架构；比如，Milletari等提出了一个U-Net 3D变体架构，称为V-Net，使用3D卷积层进行3D图像分割，目标函数直接基于Dice系数。Drozdzal等研究了在常规U-Net中使用短的类ResNet跳跃连接和长跳跃连接。

RNNs have recently become more popular for segmentation tasks. For example, Xie et al. (2016b) used a spatial clockwork RNN to segment the perimysium in H&E-histopathology images. This network takes into account prior information from both the row and column predecessors of the current patch. To incorporate bidirectional information from both left/top and right/bottom neighbors, the RNN is applied four times in different orientations and the end-result is concatenated and fed to a fully-connected layer. This produces the final output for a single patch. Stollenga et al. (2015) where the first to use a 3D LSTM-RNN with convolutional layers in six directions. Andermatt et al. (2016) used a 3D RNN with gated recurrent units to segment gray and white matter in a brain MRI data set. Chen et al. (2016d) combined bi-directional LSTM-RNNs with 2D U-net-like-architectures to segment structures in anisotropic 3D electron microscopy images. Last, Poudel et al. (2016) combined a 2D U-net architecture with a gated recurrent unit to perform 3D segmentation.

RNNs最近在分割任务变得越来越流行。比如，Xie等使用了一个空域clockwork RNN来在H&E组织病理学图像中分割肌外膜。这个网络考虑了目前图像块中row and column predecessors的先验信息。为将左/上和右/下邻域的双向信息都纳入进来，在不同的方向上将RNN应用四次，最终结果拼接到一起，送入一个全连接层，这对单个图像块生成了最终输出。Stollenga等是第一个将3D LSTM-RNN与卷积层在六个方向上应用到一起的。Andermatt等使用了一个3D RNN（含有门循环单元），在一个脑MRI数据集中分割灰色和白色的物质。Chen等将双向LSTM-RNN与2D U-Net类的架构结合起来，以在各项异性3D电子显微镜图像中分割结构。最后，Poudel等将2D U-Net架构与门循环单元结合到一起进行3D分割。

Although these specific segmentation architectures offered compelling advantages, many authors have also obtained excellent segmentation results with patch-trained neural networks. One of the earliest papers covering medical image segmentation with deep learning algorithms used such a strategy and was published by Ciresan et al. (2012). They applied pixel-wise segmentation of membranes in electron microscopy imagery in a sliding window fashion. Most recent papers now use fCNNs (subsection 2.4.3) in preference over sliding-window-based classification to reduce redundant computation.

虽然这些特定的分割架构得到了很有竞争力的优势，很多作者用图像块训练的神经网络得到了非常好的分割结果。使用这种策略的深度学习医学图像分割算法，其中最早的一篇是Ciresan等2012年发表的。他们分割电子显微镜图像中的细胞膜，以滑窗的方式。大多数现有的文章使用fCNNs，而不采用基于滑窗的分类，以减少冗余的计算。

fCNNs have also been extended to 3D and have been applied to multiple targets at once: Korez et al. (2016), used 3D fCNNs to generate vertebral body likelihood maps which drove deformable models for vertebral body segmentation in MR images, Zhou et al. (2016) segmented nineteen targets in the human torso, and Moeskops et al. (2016b) trained a single fCNN to segment brain MRI, the pectoral muscle in breast MRI, and the coronary arteries in cardiac CT angiography (CTA).

fCNNs也扩展到了3D，曾被用于多目标分割：Korez等使用3D fCNNs，来生成脊椎骨身体似然图，将MR图像中可变性模型进行脊椎骨身体分割的结果向前推动了一步。Zhou等在人体躯干中分割了19个目标，Moeskops等训练了一个fCNN来分割脑MRI，在胸MRI中分割胸肌，在心脏CT血管造影中分割冠状动脉血管。

One challenge with voxel classification approaches is that they sometimes lead to spurious responses. To combat this, groups have tried to combine fCNNs with graphical models like MRFs (Shakeri et al., 2016; Song et al., 2015) and Conditional Random Fields (CRFs) (Alansary et al., 2016; Cai et al., 2016a; Christ et al., 2016; Dou et al., 2016c; Fu et al., 2016a; Gao et al., 2016c) to refine the segmentation output. In most of the cases, graphical models are applied on top of the likelihood map produced by CNNs or fCNNs and act as label regularizers.

体素分割方法的一个挑战是，有时候会得到虚假的响应。为防止这种情况，研究者尝试将fCNN与图模型结合起来，如MRF和CRF，以提炼分割输出。在大多数情况下，CNNs或fCNNs得到的置信度图，送入图模型进行处理，图模型的作用是标签正则化器。

Summarizing, segmentation in medical imaging has seen a huge influx of deep learning related methods. Custom architectures have been created to directly target the segmentation task. These have obtained promising results, rivaling and often improving over results obtained with fCNNs.

总结起来，医学图像分割涌入了大量深度学习相关的方法。有一些定制的专用架构来直接处理分割任务。取得了一些很有希望的结果，与fCNN得到的结果类似或有所改进。

#### 3.3.2. Lesion segmentation

Segmentation of lesions combines the challenges of object detection and organ and substructure segmentation in the application of deep learning algorithms. Global and local context are typically needed to perform accurate segmentation, such that multi-stream networks with different scales or non-uniformly sampled patches are used as in for example Kamnitsas et al. (2017) and Ghafoorian et al. (2016b). In lesion segmentation we have also seen the application of U-net and similar architectures to leverage both this global and local context. The architecture used by Wang et al. (2015), similar to the U-net, consists of the same downsampling and upsampling paths, but does not use skip connections. Another U-net-like architecture was used by Brosch et al. (2016) to segment white matter lesions in brain MRI. However, they used 3D convolutions and a single skip connection between the first convolutional and last deconvolutional layers.

损伤分割将目标检测的挑战与器官、子结构的分割结合了起来，采用深度学习算法进行处理。一般需要全局和局部上下文来进行准确的分割，这样就可以使用不同尺度的多流网络，或非均匀取样的图像块，如Kamnitsas等和Ghafoorian等。在损伤分割中，我们还看到了U-Net和类似架构的应用，利用了这种全局和局部上下文。Wang等使用的架构，与U-Net架构类似，包含相同的下采样和上采样路径，但没有使用跳跃连接。另一个与U-Net类似的架构是Brosch等，分割的是脑MRI的白色物质损伤。但是，他们使用了3D卷积和单个跳跃连接，将第一个卷积和最后一个解卷积层连接了起来。

One other challenge that lesion segmentation shares with object detection is class imbalance, as most voxels/pixels in an image are from the non-diseased class. Some papers combat this by adapting the loss function: Brosch et al. (2016) defined it to be a weighted combination of the sensitivity and the specificity, with a larger weight for the specificity to make it less sensitive to the data imbalance. Others balance the data set by performing data augmentation on positive samples (Kamnitsas et al., 2017; Litjens et al., 2016; Pereira et al., 2016). Thus lesion segmentation sees a mixture of approaches used in object detection and organ segmentation. Developments in these two areas will most likely naturally propagate to lesion segmentation as the existing challenges are also mostly similar.

损伤分割和目标检测共有的另一个挑战是类别不均衡，因为图像中的大多数体素/像素都是非病态类别。一些文章通过调整损失函数来应对：Brosch等将损失函数定义为根据敏感性和专用性的加权组合，对数据不均衡更不敏感的，权重要更大一些。其他通过对正样本进行数据扩充，来平衡数据集。所以在损伤分割中，是目标检测和器官分割方法的混合。这两个领域的发展会很自然的传播到损伤分割中，因为存在的挑战大多很相似。

### 3.4. Registration

Registration (i.e. spatial alignment) of medical images is a common image analysis task in which a coordinate transform is calculated from one medical image to another. Often this is performed in an iterative framework where a specific type of (non-)parametric transformation is assumed and a pre-determined metric (e.g. L2-norm) is optimized. Although segmentation and lesion detection are more popular topics for deep learning, researchers have found that deep networks can be beneficial in getting the best possible registration performance. Broadly speaking, two strategies are prevalent in current literature: (1) using deep-learning networks to estimate a similarity measure for two images to drive an iterative optimization strategy, and (2) to directly predict transformation parameters using deep regression networks.

医学图像配准（即，空间对齐），是一个常见的图像分析任务，其中计算一幅图像到另一幅图像的坐标变换。通常这是在一个迭代框架下计算的，其中假定一种特有的参数/非参数变换，和预先确定的度量标准（如，L2范数），然后进行优化。虽然分割和损伤检测是更流行的深度学习课题，研究者发现，深度网络可以得到最好的配准性能。广义来说，目前的文献中流行的是两种策略：(1)使用深度学习网络估计两幅图像的相似度度量，以驱动迭代优化策略，(2)使用深度回归网络直接预测变换参数。

Wu et al. (2013), Simonovsky et al. (2016), and Cheng et al. (2015) used the first strategy to try to optimize registration algorithms. Cheng et al. (2015) used two types of stacked auto-encoders to assess the local similarity between CT and MRI images of the head. Both auto-encoders take vectorized image patches of CT and MRI and reconstruct them through four layers. After the networks are pre-trained using unsupervised patch reconstruction they are fine-tuned using two prediction layers stacked on top of the third layer of the SAE. These prediction layers determine whether two patches are similar (class 1) or dissimilar (class 2). Simonovsky et al. (2016) used a similar strategy, albeit with CNNs, to estimate a similarity cost between two patches from differing modalities. However, they also presented a way to use the derivative of this metric to directly optimize the transformation parameters, which are decoupled from the network itself. Last, Wu et al. (2013) combined independent subspace analysis and convolutional layers to extract features from input patches in an unsupervised manner. The resultant feature vectors are used to drive the HAMMER registration algorithm instead of handcrafted features.

Wu等使用第一种策略尝试优化配准算法。Cheng等使用两类SAEs来评估头部CT和MRI图像的局部相似度。两种类型的AEs以CT和MRI图像块的向量化形式为输入，通过四层进行重建。在网络使用无监督图像块重建进行预训练后，使用两个预测层进行精调，这是在SAE的第三层上堆叠起来的。这些预测层确定了两个图像块是相似的（类别1）还是不相似（类别2）。Simonovsky等使用了一种类似的策略，使用CNNs估计两种不同模态下的图像块的相似度。但是，他们还提出了一种方法，使用这种度量的导数来直接优化变换参数，这就与网络本身进行了解耦合。最后，Wu等将独立的子空间分析与卷积层结合，从输入图像块中以一种无监督的方式提取特征。得到的特征向量，用来驱动HAMMER配准算法，而不是用手工设计的特征来进行。

Miao et al. (2016) and Yang et al. (2016d) used deep learning algorithms to directly predict the registration transform parameters given input images. Miao et al. (2016) leveraged CNNs to perform 3D model to 2D x-ray registration to assess the pose and location of an implanted object during surgery. In total the transformation has 6 parameters, two translational, 1 scaling and 3 angular parameters. They parameterize the feature space in steps of 20 degrees for two angular parameters and train a separate CNN to predict the update to the transformation parameters given an digitally reconstructed x-ray of the 3D model and the actual inter-operative x-ray. The CNNs are trained with artificial examples generated by manually adapting the transformation parameters for the input training data. They showed that their approach has significantly higher registration success rates than using traditional - purely intensity based - registration methods. Yang et al. (2016d) tackled the problem of prior/current registration in brain MRI using the OASIS data set. They used the large deformation diffeomorphic metric mapping (LDDMM) registration methodology as a basis. This method takes as input an initial momentum value for each pixel which is then evolved over time to obtain the final transformation. However, the calculation of the initial momentum map is often an expensive procure. The authors circumvent this by training a U-net like architecture to predict the x- and y-momentum map given the input images. They obtain visually similar results but with significantly improved execution time: 1500x speed-up for 2D and 66x speed-up for 3D.

Miao等和Yang等使用深度学习算法，在给定输入图像的情况下，来直接预测配准变换参数。Miao等利用CNNs来进行3D模型到2D x射线的配准，在手术过程中评估植入目标的姿态和位置。这个变换总计有6个参数，两个平移参数，一个尺度参数，和三个角度参数。他们将特征空间进行参数化，以20度进行步进，有两个角度参数，训练一个单独的CNN，预测变换参数的更新，之前给定数字重建的x射线3D模型，和实际的操作内的x射线。CNNs使用人工制造的样本进行训练，样本是通过对输入训练数据手工调整变换参数得到的。他们证明了，他们的方法比传统的完全基于灰度的配准方法，有着明显更好的配准成功率。Yang等处理的问题是脑MRI中的先验/目前的配准问题，使用的是OASIS数据集。他们使用LDDMM配准方法作为基准。这种方法对每个像素输入一个初始的动量值，然后随着时间演化，得到最终的变换。但是，初始动量图的计算通常是非常耗时耗力的。作者在给定输入图像的情况下，通过训练一个类U-Net架构的网络来预测x-和y-动量图，来避免了这个问题。他们得到视觉上非常类似的结果，但运行时间大大减少：2D图像得到了1500倍的加速，3D图像得到了66倍的加速。

In contrast to classification and segmentation, the research community seems not have yet settled on the best way to integrate deep learning techniques in registration methods. Not many papers have yet appeared on the subject and existing ones each have a distinctly different approach. Thus, giving recommendations on what method is most promising seems inappropriate. However, we expect to see many more contributions of deep learning to medical image registration in the near future.

与分类和分割问题相比，研究团体还未将深度学习完全应用到配准方法中。这个课题的文章还不够多，现有的则有着完全不同的方法。所以，推荐最有希望的方法是不太合适的。但是，我们希望未来看到深度学习在医学图像配准中的更多贡献。

### 3.5. Other tasks in medical imaging

#### 3.5.1. Content-based image retrieval

Content-based image retrieval (CBIR) is a technique for knowledge discovery in massive databases and offers the possibility to identify similar case histories, understand rare disorders, and, ultimately, improve patient care. The major challenge in the development of CBIR methods is extracting effective feature representations from the pixel-level information and associating them with meaningful concepts. The ability of deep CNN models to learn rich features at multiple levels of abstraction has elicited interest from the CBIR community.

基于内容的图像检索(CBIR)是大规模数据库中的知识发现技术，识别相似的历史案例，理解罕见疾病，最终改进病人的护理。CBIR方法发展的主要挑战是，从像素级的信息中，提取出有效的特征表示，将其与有意义的概念相关联。深度CNN模型的学习多个抽象层次的丰富特征的能力，得到CBIR团体的兴趣和关注。

All current approaches use (pre-trained) CNNs to extract feature descriptors from medical images. Anavi et al. (2016) and Liu et al. (2016b) applied their methods to databases of X-ray images. Both used a five-layer CNN and extracted features from the fully-connected layers. Anavi et al. (2016) used the last layer and a pre-trained network. Their best results were obtained by feeding these features to a one-vs-all support vector machine (SVM) classifier to obtain the distance metric. They showed that incorporating gender information resulted in better performance than just CNN features. Liu et al. (2016b) used the penultimate fully-connected layer and a custom CNN trained to classify X-rays in 193 classes to obtain the descriptive feature vector. After descriptor binarization and data retrieval using Hamming separation values, the performance was inferior to the state of the art, which the authors attributed to small patch sizes of 96 pixels. The method proposed by Shah et al. (2016) combines CNN feature descriptors with hashing-forests. 1000 features were extracted for overlapping patches in prostate MRI volumes, after which a large feature matrix was constructed over all volumes. Hashing forests were then used to compress this into descriptors for each volume.

所有目前的方法都使用（预训练的）CNNs来从医学图像中提取特征的描述子。Anavi等和Liu等将其方法应用于X射线图像的数据库，他们都使用了五层的CNN，从全连接层中提取出特征。Anavi等使用最后一层和一个预训练的网络。其最好结果是通过将特征送入一个一对多的SVM分类器，以得到距离度量。他们证明了，将性别信息纳入之后，会得到比只用CNN特征更好的结果。Liu等使用倒数第二个全连接层，和定制的CNN，训练用于x射线图像的分类，得到193类，得到描述的特征向量。在描述子二值化，使用Hamming分离值进行数据检索后，性能比目前最好的要差一些，作者将其归于较小的图像块大小的原因，只有96像素。Shah等提出的方法，将CNN特征描述子与hashing-forests相结合。在前列腺MRI体中，重叠的图像块提取出1000个特征，然后在所有体上构造了一个巨大的特征矩阵。Hashing forests然后用于将之压缩成每个体的描述子。

Content-based image retrieval as a whole has thus not seen many successful applications of deep learning methods yet, but given the results in other areas it seems only a matter of time. An interesting avenue of research could be the direct training of deep networks for the retrieval task itself.

基于内容的图像检索，整体上尚未看到很多成功的基于深度学习的方法的应用，但在其他领域中应用势头良好，所以似乎这只是时间的问题。一种有趣的研究路线可能是，直接将深度学习训练用于检索任务本身。

#### 3.5.2. Image Generation and Enhancement

A variety of image generation and enhancement methods using deep architectures have been proposed, ranging from removing obstructing elements in images, normalizing images, improving image quality, data completion, and pattern discovery.

有很多使用深度学习的图像生成和图像增强算法被提出，包括从图像中移除阻碍元素，归一化图像，改进图像质量，数据补全，和模式发现。

In image generation, 2D or 3D CNNs are used to convert one input image into another. Typically these architectures lack the pooling layers present in classification networks. These systems are then trained with a data set in which both the input and the desired output are present, defining the differences between the generated and desired output as the loss function. Examples are regular and bone-suppressed X-ray in Yang et al. (2016c), 3T and 7T brain MRI in Bahrami et al. (2016), PET from MRI in Li et al. (2014), and CT from MRI in Nie et al. (2016a). Li et al. (2014) even showed that one can use these generated images in computer-aided diagnosis systems for Alzheimer’s disease when the original data is missing or not acquired.

在图像生成中，2D或3D CNNs用于将一幅输入图像转换为另一幅。一般这些架构缺少分类网络中的池化层。这些系统然后用一个数据集训练，其中有输入和期望的输出，生成的输出和期望输出之间的差异就是损失函数。样本是常规X射线和骨抑制X射线，3T和7T脑MRI，PET，和CT数据。Li等甚至证明了，可以原始数据丢失或不能得到的时候，在老年痴呆症的计算机辅助诊断系统中使用这些生成的图像。

With multi-stream CNNs super-resolution images can be generated from multiple low-resolution inputs (section 2.4.2). In Oktay et al. (2016), multi-stream networks reconstructed high-resolution cardiac MRI from one or more low-resolution input MRI volumes. Not only can this strategy be used to infer missing spatial information, but can also be leveraged in other domains; for example, inferring advanced MRI diffusion parameters from limited data (Golkov et al., 2016). Other image enhancement applications like intensity normalization and denoising have seen only limited application of deep learning algorithms. Janowczyk et al.(2016a) used SAEs to normalize H&E-stained histopathology images whereas Benou et al. (2016) used CNNs to perform denoising in DCE-MRI time-series.

使用多流CNN，可以从多个低分辨率输入中生成超分辨率图像。在Oktay等的文章中，多流网络从一幅或多幅低分辨率输入的MRI体中重建了高分辨率的心脏MRI。这种策略不仅可以用推理丢失的空间信息，还可以用在其他领域中；如，从有限的数据中推理高级MRI扩散参数。其他图像增强应用，如灰度归一化和去噪，只有很有限的深度学习算法应用。Janowczyk等使用SAEs来归一化H&E-stained组织病理学图像，同时，Benou等使用CNNs在DCE-MRI时间序列中进行去噪。

Image generation has seen impressive results with very creative applications of deep networks in significantly differing tasks. One can only expect the number of tasks to increase further in the future. 在很不同的任务中，深度网络都在图像生成中得到了很好的结果。未来这类应用肯定会进一步增加。

#### 3.5.3. Combining Image Data With Reports

The combination of text reports and medical image data has led to two avenues of research: (1) leveraging reports to improve image classification accuracy (Schlegl et al., 2015), and (2) generating text reports from images (Kisilev et al., 2016; Shin et al., 2015, 2016a; Wang et al., 2016e); the latter inspired by recent caption generation papers from natural images (Karpathy and Fei-Fei, 2015). To the best of our knowledge, the first step towards leveraging reports was taken by Schlegl et al. (2015), who argued that large amounts of annotated data may be difficult to acquire and proposed to add semantic descriptions from reports as labels. The system was trained on sets of images along with their textual descriptions and was taught to predict semantic class labels during test time. They showed that semantic information increases classification accuracy for a variety of pathologies in Optical Coherence Tomography (OCT) images.

文本报告与医学图像数据的结合，有两条研究方向：(1)利用报告来改进图像分类准确率，和(2)从图像生成文本报告；后者是受到从自然图像中生成标题的文章的启发。据我们所知，Schlegl等首先开始利用报告，他们认为大量标注数据很难获得，提出用报告中的语义描述作为标签。这个系统的训练是用的图像集合，和其文本描述，在测试时预测的是语义类别标签。他们证明了，在OCT图像中的很多病理中，语义信息提高了分类准确率。

Shin et al. (2015) and Wang et al. (2016e) mined semantic interactions between radiology reports and images from a large data set extracted from a PACS system. They employed latent Dirichlet allocation (LDA), a type of stochastic model that generates a distribution over a vocabulary of topics based on words in a document. In a later work, Shin et al.(2016a) proposed a system to generate descriptions from chest X-rays. A CNN was employed to generate a representation of an image one label at a time, which was then used to train an RNN to generate sequence of MeSH keywords. Kisilev et al. (2016) used a completely different approach and predicted categorical BI-RADS descriptors for breast lesions. In their work they focused on three descriptors used in mammography: shape, margin, and density, where each have their own class label. The system was fed with the image data and region proposals and predicts the correct label for each descriptor (e.g. for shape either oval, round, or irregular).

Shin等和Wang等分析了放射报告和图像的语义相互影响，数据是从一个PACS系统中提取出的大型数据集。他们利用一种随机模型(LDA)，在文档的词语的基础上，生成一种多个话题的概率分布。更晚一点，Shin等提出了一个系统，从胸部X射线图像中生成描述。使用一个CNN来生成一幅图像的表示，一次一个标签，然后用于训练一个RNN来生成MeSH关键字的序列。Kisilev等使用一种完全不同的方法，对胸部损伤预测分类别的BI-RADS描述子。在其工作中，他们聚焦于乳腺X光造影术中的三个描述子：形状，margin和密度，其中每个都有其自己的类别标签。图像数据和区域建议送入系统，对每个描述子预测正确的标签（如，对于形状来说，是椭圆，圆形，或不规则形状）。

Given the wealth of data that is available in PACS systems in terms of images and corresponding diagnostic reports, it seems like an ideal avenue for future deep learning research. One could expect that advances in captioning natural images will in time be applied to these data sets as well.

PACS系统中可用数据是很大的财富，包括图像和对应的诊断报告，似乎这是未来深度学习研究的理想方向。可以期望，在自然图像中的标题生成的进展，会逐渐应用到这个数据集中。

## 4. Anatomical application areas

This section presents an overview of deep learning contributions to the various application areas in medical imaging. We highlight some key contributions and discuss performance of systems on large data sets and on public challenge data sets. All these challenges are listed on http:\\www.grand-challenge.org.

本节给出深度学习在医学图像中对各种应用领域的贡献的概览。我们强调了一些关键贡献，讨论了在大型数据集上的性能，和在公开的挑战数据集上的性能。所有这些挑战都在网站上有列出。

### 4.1. Brain

DNNs have been extensively used for brain image analysis in several different application domains (Table 1). A large number of studies address classification of Alzheimer’s disease and segmentation of brain tissue and anatomical structures (e.g. the hippocampus). Other important areas are detection and segmentation of lesions (e.g. tumors, white matter lesions, lacunes, micro-bleeds).

DNNs在几个不同的应用领域广泛应用于脑部图像分析（表1）。大量研究处理的是老年痴呆症的分类和脑部组织、结构结构（如海马体）的分割。其他重要的领域包括检测和损伤分割（如，肿瘤，白色物质损伤，腔隙，微出血）。

Apart from the methods that aim for a scan-level classification (e.g. Alzheimer diagnosis), most methods learn mappings from local patches to representations and subsequently from representations to labels. However, the local patches might lack the contextual information required for tasks where anatomical information is paramount (e.g. white matter lesion segmentation). To tackle this, Ghafoorian et al. (2016b) used non-uniformly sampled patches by gradually lowering sampling rate in patch sides to span a larger context. An alternative strategy used by many groups is multi-scale analysis and a fusion of representations in a fully-connected layer.

除了目标是扫描级的分类的方法（如老年痴呆症诊断）外，多数方法学习从局部图像块到表示的映射，然后是对表示到标签的映射。但是，局部图像块可能缺少任务需要的上下文信息，其中解剖信息至关重要（如，白色物质损伤分割）。为解决这个问题，Ghafoorian等使用非均匀采样的图像块，逐渐降低图像块的采样率，构成更大的上下文。另一种替代策略是多尺度分析，和在全连接层的表示融合。

Even though brain images are 3D volumes in all surveyed studies, most methods work in 2D, analyzing the 3D volumes slice-by-slice. This is often motivated by either the reduced computational requirements or the thick slices relative to in-plane resolution in some data sets. More recent publications had also employed 3D networks.

虽然调研的脑部图像都是3D体，但大部分工作都是2D的，逐个切片的分析3D体。这通常是因为可以降低计算量，或在一些数据集中与面内分辨率相比，切片非常厚。最近的文献也常常使用3D网络。

DNNs have completely taken over many brain image analysis challenges. In the 2014 and 2015 brain tumor segmentation challenges (BRATS), the 2015 longitudinal multiple sclerosis lesion segmentation challenge, the 2015 ischemic stroke lesion segmentation challenge (ISLES), and the 2013 MR brain image segmentation challenge (MRBrains), the top ranking teams to date have all used CNNs. Almost all of the aforementioned methods are concentrating on brain MR images. We expect that other brain imaging modalities such as CT and US can also benefit from deep learning based analysis.

DNNs完全主导了很多脑部图像分析的挑战。在2014和2015年脑部肿瘤分割挑战(BRATS)上，2015年多场景损伤分割挑战上，2015年缺血性中风损伤分割挑战(ISLES)上，和2013 MR脑图像分割挑战(MRBrains)，排名最高的小组都使用的是CNNs。几乎所有前面提到的方法都很关注脑部MR图像。我们希望其他脑部成像模态，如CT和US，也可以从基于深度学习的分析中受益。

### 4.2. Eye 眼睛

Ophthalmic imaging has developed rapidly over the past years, but only recently are deep learning algorithms being applied to eye image understanding. As summarized in Table 2, most works employ simple CNNs for the analysis of color fundus imaging (CFI). A wide variety of applications are addressed: segmentation of anatomical structures, segmentation and detection of retinal abnormalities, diagnosis of eye diseases, and image quality assessment.

眼科成像过去几年发展迅速，但直到最近深度学习算法才应用到眼部图像理解中。如表2总结，多数工作采用简单的CNNs进行彩色眼底成像(CFI)的分析。包含很多应用：解剖结构的分割，视网膜疾病的分割与检测，眼部疾病的诊断，和图像质量评估。

In 2015, Kaggle organized a diabetic retinopathy detection competition: Over 35,000 color fundus images were provided to train algorithms to predict the severity of disease in 53,000 test images. The majority of the 661 teams that entered the competition applied deep learning and four teams achieved performance above that of humans, all using end-to-end CNNs. Recently Gulshan et al. (2016) performed a thorough analysis of the performance of a Google Inception v3 network for diabetic retinopathy detection, showing performance comparable to a panel of seven certified ophthalmologists.

在2015年，Kaggle组织了一场糖尿病性视网膜病变检测比赛：提供了超过35000幅彩色眼底图像进行训练算法，预测53000幅测试图像中的疾病严重程度。进行比赛的主要661个小组都使用了深度学习，四个小组取得了超过人类预测性能的成绩，使用的都是端到端的CNNs。最近，Gulshan等对Inception V3网络对糖尿病性视网膜病变检测的性能的完全分析，得到的性能与7个合格的眼科医生组成的小组类似的性能。

### 4.3. Chest 胸部

In thoracic image analysis of both radiography and computed tomography, the detection, characterization, and classification of nodules is the most commonly addressed application. Many works add features derived from deep networks to existing feature sets or compare CNNs with classical machine learning approaches using handcrafted features. In chest X-ray, several groups detect multiple diseases with a single system. In CT the detection of textural patterns indicative of interstitial lung diseases is also a popular research topic.

在X射线和CT的胸部图像分析中，小瘤的检测、特征化和分类是要处理的最常见应用。很多工作将深度网络生成的特征加入到现有的特征集里，或将CNNs方法与使用手工设计特征的经典机器学习方法相比较。在胸部X-ray中，几个小组用单个系统检测多种疾病。在CT中，检测指示肺部疾病的纹理模式，也是一个流行的研究课题。

Chest radiography is the most common radiological exam; several works use a large set of images with text reports to train systems that combine CNNs for image analysis and RNNs for text analysis. This is a branch of research we expect to see more of in the near future.

胸部X光照相是最常见的放射检查；几个工作使用了大型图像集合与文本报告来训练系统，将CNNs的图像分析与RNNs的文本分析结合起来。这是我们期望看到更多成果的一个研究分支。

In a recent challenge for nodule detection in CT, LUNA16, CNN architectures were used by all top performing systems. This is in contrast with a previous lung nodule detection challenge, ANODE09, where handcrafted features were used to classify nodule candidates. The best systems in LUNA16 still rely on nodule candidates computed by rule-based image processing, but systems that use deep networks for candidate detection also performed very well (e.g. U-net). Estimating the probability that an individual has lung cancer from a CT scan is an important topic: It is the objective of the Kaggle Data Science Bowl 2017, with 1 million dollars in prizes and more than one thousand participating teams.

在最近的CT小瘤检测挑战(LUNA16)中，表现最好的系统都使用了CNN架构。与之相比，之前的肺部小瘤检测挑战，ANODE09，其中使用了手工设计的特征来分类候选小瘤。LUNA16中的最佳系统仍然依赖于基于规则的图像处理计算得到的小瘤候选，但使用深度学习得到的候选检测结果的系统表现也非常好（如，U-Net）。从CT扫描中估计肺癌的概率是一个重要的课题：这也是Kaggle数据科学碗2017的目标，有100万美元奖金，超过了1000个参赛小组。

### 4.4. Digital pathology and microscopy 数字病理学和显微术

The growing availability of large scale gigapixel whole-slide images (WSI) of tissue specimen has made digital pathology and microscopy a very popular application area for deep learning techniques. The developed techniques applied to this domain focus on three broad challenges: (1) Detecting, segmenting, or classifying nuclei, (2) segmentation of large organs, and (3) detecting and classifying the disease of interest at the lesion or WSI-level. Table 5 presents an overview for each of these categories.

可用的组织样本的大规模十亿像素的全幅图像(WSI)越来越多，使得数字病理学和显微术成为深度学习技术的一个重要应用领域。在这个领域应用的先进技术聚焦在三个广泛的挑战中：(1)检测、分割或分类细胞核；(2)大型器官的分割；(3)在损伤处或WSI级别上，检测分类感兴趣的疾病。表5给出了每个类别的研究概览。

Deep learning techniques have also been applied for normalization of histopathology images. Color normalization is an important research area in histopathology image analysis. In Janowczyk et al. (2016a), a method for stain normalization of hematoxylin and eosin (H&E) stained histopathology images was presented based on deep sparse auto-encoders. Recently, the importance of color normalization was demonstrated by Sethi et al. (2016) for CNN based tissue classification in H&E stained images.

深度学习技术也应用于组织病理学图像的归一化。色彩归一化是组织病理学图像分析中的一个重要研究领域。Janowczyk等基于深度稀疏AE提出了一种对H&E着色的组织病理学图像进行着色归一化的方法。最近，Sethi等在基于CNN的H&E着色图像中进行组织分类中证明了色彩归一化的重要性。

The introduction of grand challenges in digital pathology has fostered the development of computerized digital pathology techniques. The challenges that evaluated existing and new approaches for analysis of digital pathology images are: EM segmentation challenge 2012 for the 2D segmentation of neuronal processes, mitosis detection challenges in ICPR 2012 and AMIDA 2013, GLAS for gland segmentation and, CAMELYON16 and TUPAC for processing breast cancer tissue samples.

数字病理学中引入了挑战赛，这促进了计算机数字病理学技术的发展。挑战赛评估数字病理学图像分析的现有的和新的方法，包括：EM分割挑战赛2012，进行神经元的2D分割，细胞有丝分裂检测挑战赛ICPR 2012和AMIDA 2013，分割挑战赛GLAS，处理乳腺癌组织样本的CAMELYON16和TUPAC。

In both ICPR 2012 and the AMIDA13 challenges on mitosis detection the IDSIA team outperformed other algorithms with a CNN based approach (Cires ¸an et al., 2013). The same team had the highest performing system in EM 2012 (Ciresan et al., 2012) for 2D segmentation of neuronal processes. In their approach, the task of segmenting membranes of neurons was performed by mild smoothing and thresholding of the output of a CNN, which computes pixel probabilities.

在ICPR 2012和AMIDA 2013的细胞有丝分裂检测挑战赛中，IDSIA小组使用基于CNN的方法赢得了比赛。在EM 2012中他们也得到了表现最好的成绩，内容是神经元过程的2D分割。在他们的方法中，分割神经元细胞膜的任务是将CNN的输出进行平滑并用阈值进行分割的，计算的是像素的概率。

GLAS addressed the problem of gland instance segmentation in colorectal cancer tissue samples. Xu et al. (2016d) achieved the highest rank using three CNN models. The first CNN classifies pixels as gland versus non-gland. From each feature map of the first CNN, edge information is extracted using the holistically nested edge technique, which uses side convolutions to produce an edge map. Finally, a third CNN merges gland and edge maps to produce the final segmentation.

GLAS处理的是结肠直肠癌组织样本的腺个体分割问题。Xu等使用三个CNN模型取得了最高的评分。第一个CNN的分类是腺体vs非腺体。从第一个CNN的每个特征图中，提取出边缘信息（使用全面嵌套的边缘技术），使用边卷积来生成边缘图。最后，第三个CNN将腺体和边缘图合并，生成最终的分割。

CAMELYON16 was the first challenge to provide participants with WSIs. Contrary to other medical imaging applications, the availability of large amount of annotated data in this challenge allowed for training very deep models such as 22-layer GoogLeNet (Szegedy et al., 2014), 16-layer VGG-Net (Simonyan and Zisserman, 2014), and 101-layer ResNet (He et al., 2015). The top-five performing systems used one of these architectures. The best performing solution in the Camelyon16 challenge was presented in Wang et al. (2016b). This method is based on an ensemble of two GoogLeNet architectures, one trained with and one without hard-negative mining to tackle the challenge. The latest submission of this team using the WSI standardization algorithm by Ehteshami Bejnordi et al. (2016) achieved an AUC of 0.9935, for task 2, which outperformed the AUC of a pathologist (AUC = 0.966) who independently scored the complete test set.

CAMELYON16是给参与者提供WSIs的第一个挑战赛。与其他医学图像应用相反，这个挑战赛中大规模标注图像的可用性，使得可以训练非常深的模型，如22层的GoogLeNet，16层的VGG和101层的ResNet。表现最好的5个系统都使用了这些架构中的一个。Camelyon16中表现最好的是Wang等。这个方法是基于两个GoogLeNet架构的集成，一个用难分样本挖掘进行训练，一个没有用。这个小组最近一次提交使用了Ehteshami Bejnordi等的WSI标准化算法，在任务2中得到了0.9935的AUC，一个病理学家独立的完成了测试集，得到了AUC为0.966。

The recently held TUPAC challenge addressed detection of mitosis in breast cancer tissue, and prediction of tumor grading at the WSI level. The top performing system by Paeng et al. (2016) achieved the highest performance in all tasks. The method has three main components: (1) Finding high cell density regions, (2) using a CNN to detect mitoses in the regions of interest, (3) converting the results of mitosis detection to a feature vector for each WSI and using an SVM classifier to compute the tumor proliferation and molecular data scores.

最近举行的TUPAC挑战赛乳腺癌组织的细胞有丝分裂检测，在WSI级预测肿瘤的评级。Paeng等是所有任务中表现最好的系统。其方法包含三个主要部分：(1)发现细胞高度密集区域；(2)使用CNN在感兴趣区域检测有丝分裂；(3)对每个WSI，将有丝分裂检测的结果转化为一个特征向量，使用一个SVM分类器来计算肿瘤增殖和分子数据分数。

### 4.5. Breast

One of the earliest DNN applications from Sahiner et al. (1996) was on breast imaging. Recently, interest has returned which resulted in significant advances over the state of the art, achieving the performance of human readers on ROIs (Kooi et al., 2016). Since most breast imaging techniques are two dimensional, methods successful in natural images can easily be transferred. With one exception, the only task addressed is the detection of breast cancer; this consisted of three subtasks: (1) detection and classification of mass-like lesions, (2) detection and classification of micro-calcifications, and (3) breast cancer risk scoring of images. Mammography is by far the most common modality and has consequently enjoyed the most attention. Work on tomosynthesis, US, and shear wave elastography is still scarce, and we have only one paper that analyzed breast MRI with deep learning; these other modalities will likely receive more attention in the next few years. Table 6 summarizes the literature and main messages.

Sahiner等的一个最高的DNN应用就是乳腺图像处理。最近，兴趣的回归导致目前最好的方法有了显著进展，在感兴趣区域中达到了人类解读者的水平。由于多数乳腺成像技术都是二维的，自然图像成功应用的方法可以很容易迁移过来。有一个例外，就是要处理的唯一任务是乳腺癌的检测；这由三个子任务组成：(1)类似重物的损伤的检测与分类；(2)微钙化的检测与分类；(3)图像中含有的乳腺癌分数。乳腺摄影术是目前最常见的形态，因此得到了最多的关注。断层合成、超声和横波弹性成像的研究仍然非常少，只有一篇文章采用深度学习分析乳腺MRI的；其他形态的成像在以后几年会得到更多的注意力。表6总结了相关文献和主要信息。

Since many countries have screening initiatives for breast cancer, there should be massive amounts of data available, especially for mammography, and therefore enough opportunities for deep models to flourish. Unfortunately, large public digital databases are unavailable and consequently older scanned screen-film data sets are still in use. Challenges such as the recently launched DREAM challenge have not yet had the desired success.

因为很多国家都有乳腺癌的筛选计划，因为有大量数据可用，尤其是乳腺X光成像，因为深度模型有充分的机会发展壮大。不幸的是，大量公开可用的数据库都是不可用的，结果更老的扫描的胶片数据仍然在使用。最近提出的DREAM挑战，尚未取得应有的成功。

As a result, many papers used small data sets resulting in mixed performance. Several projects have addressed this issue by exploring semi-supervised learning (Sun et al., 2016a), weakly supervised learning (Hwang and Kim, 2016), and transfer learning (Kooi et al., 2017; Samala et al., 2016b)). Another method combines deep models with handcrafted features (Dhungeletal.,2016), which have been shown to be complementary still, even for very big data sets (Kooi et al., 2016). State of the art techniques for mass-like lesion detection and classification tend to follow a two-stage pipeline with a candidate detector; this design reduces the image to a set of potentially malignant lesions, which are fed to a deep CNN (Fotin et al., 2016; Kooi et al., 2016). Alternatives use a region proposal network (R-CNN) that bypasses the cascaded approach (Akselrod-Ballin et al., 2016; Kisilev et al., 2016).

结果是，很多文章使用的都是小型数据集，得到的性能不一。几个项目都在处理这个问题，探索半监督学习，弱监督学习，和迁移学习。另一种方法，将深度学习模型与手工设计的特征结合起来，证明了其功能是互补的，即使是对非常大型的数据集也是这样。对于类似重物的损伤检测，目前最好的技术是两阶段流程，使用一个候选检测器；这种设计将图像缩小为可能的恶心损伤集合，然后送入深度CNN。替代方法使用R-CNN，绕过了级联的方法。

When large data sets are available, good results can be obtained. At the SPIE Medical Imaging conference of 2016, a researcher from a leading company in the mammography CAD field told a packed conference room how a few weeks of experiments with a standard architecture (AlexNet) - trained on the company’s proprietary database - yielded a performance that was superior to what years of engineering handcrafted feature systems had achieved (Fotin et al., 2016).

当大型数据集可用时，可以得到很好的结果。在2016年SPIE医学图像处理会议上，一位乳腺X射线成像CAD领域的大型公司的研究者，在拥挤的会议室中，讲述了采用标准架构AlexNet用几个星期的试验，在公司的私有数据库上进行训练，得到的性能超过了手工设计特征几年的工作所取得的成绩。

### 4.6. Cardiac

Deep learning has been applied to many aspects of cardiac image analysis; the literature is summarized in Table 7. MRI is the most researched modality and left ventricle segmentation the most common task, but the number of applications is highly diverse: segmentation, tracking, slice classification, image quality assessment, automated calcium scoring and coronary centerline tracking, and super-resolution.

深度学习已经应用的心脏图像分析的很多方面了，文献总结于表7中。MRI是研究最多的形态，左心室的分割是最常见的任务，但应用数量非常多样：分割，跟踪，切片分类，图像质量评估，自动钙评分，和冠状动脉中心线追踪，和超分辨率。

Most papers used simple 2D CNNs and analyzed the 3D and often 4D data slice by slice; the exception is Wolterink et al. (2016) where 3D CNNs were used. DBNs are used in four papers, but these all originated from the same author group. The DBNs are only used for feature extraction and are integrated in compound segmentation frameworks. Two papers are exceptional because they combined CNNs with RNNs: Poudel et al. (2016) introduced a recurrent connection within the U-net architecture to segment the left ventricle slice by slice and learn what information to remember from the previous slices when segmenting the next one. Kong et al. (2016) used an architecture with a standard 2D CNN and an LSTM to perform temporal regression to identify specific frames and a cardiac sequence. Many papers use publicly available data. The largest challenge in this field was the 2015 Kaggle Data Science Bowl where the goal was to automatically measure end-systolic and end-diastolic volumes in cardiac MRI. 192 teams competed for 200,000 dollars in prize money and the top ranking teams all used deep learning, in particular fCNN or U-net segmentation schemes.

大多数文章使用简单的2D CNNs，逐个切片的分析3D和4D数据。例外情况是Wolterink等，使用了3D CNNs。DBNs在四篇文章中得到了使用，但都源自于同一作者群体。DBNs只用于特征提取，整合到复合分割框架。有两篇例外文章，将CNNs与RNNs结合了起来：Poudel等在U-Net架构中引入了循环连接，以逐切片的分割左心室，当分割下一个切片时，学习从上一个切片中记住什么信息。Kong等使用的架构是标准2D CNN与LSTM的结合，进行时域回归，确定特定的帧和心脏序列。很多文章使用公开可用的数据。这个领域最大的挑战是2015年的Kaggle数据科学碗，其中的目标是在心脏MRI中自动度量收缩末期和舒张末期的体积。192个小组竞争20万美元的奖金，最高排名的队伍都使用了CNNs，特别是分割方案使用fCNN或U-Net。

### 4.7. Abdomen 腹腔

Most papers on the abdomen aimed to localize and segment organs, mainly the liver, kidneys, bladder, and pancreas (Table 8). Two papers address liver tumor segmentation. The main modality is MRI for prostate analysis and CT for all other organs. The colon is the only area where various applications were addressed, but always in a straightforward manner: A CNN was used as a feature extractor and these features were used for classification.

腹腔方面的多数文章目标是定位并分割器官，主要是肝脏，肾脏，膀胱和胰腺（表8）。两篇文章处理肝脏肿瘤分割的问题。前列腺分析的主要模态是MRI，其他器官主要是CT。各种应用都会处理结肠这个部位，但都是很直接的方式：用CNN作为特征提取器，这些特征用作分类。

It is interesting to note that in two segmentation challenges - SLIVER07 for liver and PROMISE12 for prostate - more traditional image analysis methods were dominant up until 2016. In PROMISE12, the current second and third in rank among the automatic methods used active appearance models. The algorithm from IMorphics was ranked first for almost five years (now ranked second). However, a 3D fCNN similar to U-net (Yu et al., 2017) has recently taken the top position. This paper has an interesting approach where a sum operation was used instead of the concatenation operation used in U-net, making it a hybrid between a ResNet and U-net architecture. Also in SLIVER07 - a 10-year-old liver segmentation challenge - CNNs have started to appear in 2016 at the top of the leaderboard, replacing previously dominant methods focused on shape and appearance modeling.

应当指出，在两个分割挑战赛中，肝脏的SLIVER07和前列腺的PROMISE12，占主导的是更多传统的图像分析，直到2016年。在PROMISE12中，目前的排名第二和第三的方法使用的是active appearance models。IMorphics的算法排名第一，保持了几乎5年（现在排名第2）。但是，一个与U-Net类似的3D fCNN最近到达了第一的位置。这篇文章的方法很有意思，没有使用U-Net中的拼接运算，而是使用了一个求和运算，使其成为了ResNet和U-Net的混合架构。在SLIVER07中，一个10年的肝脏分割挑战赛上，CNNs在2016年开始出现在排行榜上，替换了之前主导的，关注形状和外形的建模方法。

### 4.8. Musculoskeletal 肌肉骨骼

Musculoskeletal images have also been analyzed by deep learning algorithms for segmentation and identification of bone, joint, and associated soft tissue abnormalities in diverse imaging modalities. The works are summarized in Table 9.

肌肉骨骼图像也用深度学习算法进行了分析，对骨骼、关节和关联的软组织异常进行分割或识别，成像模态很多样。这些工作如表9所示。

A surprising number of complete applications with promising results are available; one that stands out is Jamaludin et al. (2016) who trained their system with 12K discs and claimed near-human performances across four different radiological scoring tasks.

有很多结果很好的完整应用都是可用的。Jamaludin等用12K个discs训练其系统，声称在四个不同的放射评分任务中达到了接近人类的性能。

### 4.9. Other

This final section lists papers that address multiple applications (Table 10) and a variety of other applications (Table 11). 表10给出了多应用的文章列表，表11给出了其他各种应用的文章列表。

It is remarkable that one single architecture or approach based on deep learning can be applied without modifications to different tasks; this illustrates the versatility of deep learning and its general applicability. In some works, pre-trained architectures are used, sometimes trained with images from a completely different domain. Several authors analyze the effect of fine-tuning a network by training it with a small data set of images from the intended application domain. Combining features extracted by a CNN with ‘traditional’ features is also commonly seen.

基于深度学习的一种架构或方法，不经修改就可以应用于不同的任务中，这是很非凡的成就；这说明了深度学习应用广泛，和其强大的泛化能力。在一些工作中，使用了预训练的架构，有时候使用完全不同的领域的数据进行训练。几个作者分析了精调网络的效果，用目标应用领域的小型数据集对其进行训练。将CNN提取出来的特征与传统特征进行结合，也是很常见的。

From Table 11, the large number of papers that address obstetric applications stand out. Most papers address the groundwork, such as selecting an appropriate frame from an US stream. More work on automated measurements with deep learning in these US sequences is likely to follow.

从表11中，大量处理产科应用的文章很突出。多数文章处理的是基础工作，比如从一个US流中选择一个合适的帧。后面应当会出现更多的用深度学习自动度量这些US序列中的工作。

The second area where CNNs are rapidly improving the state of the art is dermoscopic image analysis. For a long time, diagnosing skin cancer from photographs was considered very difficult and out of reach for computers. Many studies focused only on images obtained with specialized cameras, and recent systems based on deep networks produced promising results. A recent work by Esteva et al. (2017) demonstrated excellent results with training a recent standard architecture (Google’s Inception v3) on a data set of both dermoscopic and standard photographic images. This data set was two orders of magnitude larger than what was used in literature before. In a thorough evaluation, the proposed system performed on par with 30 board certified dermatologists.

深度学习迅速推进最好效果的第二个领域是皮肤镜图像分析。在很长时间中，从图像中诊断皮肤癌是非常困难的，计算机也无法进行处理。很多研究关注的只是专用相机得到的图像，最近的基于深度网络的系统得到了很有希望的结果。Esteva等的最近工作证明了，使用最近的标准架构(Inception V3)在由皮肤镜图像和标准光学图像组成的数据集上进行训练可以得到非常好的结果。这个数据集比之前的文献使用的数据集大了两个数量级。在一个彻底的评估中，提出的系统与30位注册的皮肤科医生相媲美。

## 5. Discussion 讨论

*Overview* 概览

From the 308 papers reviewed in this survey, it is evident that deep learning has pervaded every aspect of medical image analysis. This has happened extremely quickly: the vast majority of contributions, 242 papers, were published in 2016 or the first month of 2017. A large diversity of deep architectures are covered. The earliest studies used pre-trained CNNs as feature extractors. The fact that these pre-trained networks could simply be downloaded and directly applied to any medical image facilitated their use. Moreover, in this approach already existing systems based on handcrafted features could simply be extended. In the last two years, however, we have seen that end-to-end trained CNNs have become the preferred approach for medical imaging interpretation (see Figure 1). Such CNNs are often integrated into existing image analysis pipelines and replace traditional handcrafted machine learning methods. This is the approach followed by the largest group of papers in this survey and we can confidently state that this is the current standard practice.

从我们调查的308篇文章中，很明显深度学习已经渗透到了医学图像分析的每个方面。这种情况的发生非常的迅速：有242篇文章，即绝大部分，是在2016和2017的第一个月份发表的。覆盖了非常多深度学习架构。最早的研究使用预训练的CNNs作为特征提取器。预训练网络可以直接下载并直接应用于任何医学图像中，这使其方便使用。而且，使用这种方法，基于手工设计特征的现有的方法可以很容易得到拓展。过去两年中，我们看到端到端训练的CNNs成为了医学图像解释中的流行方法（见图1）。这种CNNs经常整合进现有的图像分析流程中，替换掉传统的手工设计的机器学习方法。这是大部分文章的方法，我们很自信的说，这是目前的标准实践。

*Key aspects of successful deep learning methods* 成功的深度学习方法的关键方面

After reviewing so many papers one would expect to be able to distill the perfect deep learning method and architecture for each individual task and application area. Although convolutional neural networks (and derivatives) are now clearly the top performers in most medical image analysis competitions, one striking conclusion we can draw is that the exact architecture is not the most important determinant in getting a good solution. We have seen, for example in challenges like the Kaggle Diabetic Retinopathy Challenge, that many researchers use the exact same architectures, the same type of networks, but have widely varying results. A key aspect that is often overlooked is that expert knowledge about the task to be solved can provide advantages that go beyond adding more layers to a CNN. Groups and researchers that obtain good performance when applying deep learning algorithms often differentiate themselves in aspects outside of the deep network, like novel data preprocessing or augmentation techniques. An example is that the best performing method in the CAMELYON16-challenge improved significantly (AUC from 0.92 to 0.99) by adding a stain normalization pre-processing step to improve generalization without changing the CNN. Other papers focus on data augmentation strategies to make networks more robust, and they report that these strategies are essential to obtain good performance. An example is the elastic deformations that were applied in the original U-Net paper (Ronneberger et al., 2015).

回顾了这么多文章之后，可以提炼出每个单独任务和应用领域最适合的深度学习方法和架构。虽然CNNs（和衍生）现在明显是多数医学图像分析竞赛表现最好的模型，我们可以得出的一个惊人结论是，确切的架构，在得到一个很好的解决方案上，不是最重要的有决定性的因素。我们已经观察到，比如在Kaggle糖尿病视网膜病变这样的挑战中，很多研究者使用完全相同的架构，同一类型的网络，但得到了非常不一样的结果。通常忽视的一个关键方面是，关于要解决的任务的专家知识，要比为CNN增加几层要更加重要。研究者们使用深度学习算法取得很好的结果，其区别因素往往不在深度网络中，比如新的数据预处理技术，或数据增强技术。一个例子是，在CAMELYON16挑战上表现最好的方法，在没有改变CNN结构的情况下，增加了一个着色归一化预处理步骤，以改进泛化能力，就显著改进了性能，AUC从0.92增加到0.99。其他文章聚焦在数据扩充策略上，使网络更加稳健，他们报道说，这些策略对于得到很好的性能是非常关键的。一个例子是，在原始U-Net文章上应用了弹性形变。

Augmentation and pre-processing are, of course, not the only key contributors to good solutions. Several researchers have shown that designing architectures incorporating unique task-specific properties can obtain better results than straightforward CNNs. Two examples which we encountered several times are multi-view and multi-scale networks. Other, often underestimated, parts of network design are the network input size and receptive field (i.e. the area in input space that contributes to a single output unit). Input sizes should be selected considering for example the required resolution and context to solve a problem. One might increase the size of the patch to obtain more context, but without changing the receptive field of the network this might not be beneficial. As a standard sanity check researchers could perform the same task themselves via visual assessment of the network input. If they, or domain experts, cannot achieve good performance, the chance that you need to modify your network input or architecture is high.

数据扩充和预处理，当然不是得到好的解决方案的唯一关键贡献者。几个研究者表明，为任务设计特定架构的网络，可以得到更好的结果。我们好几次遇到的两个例子是，多视野和多尺度网络。其他的，通常低估的，网络设计的一部分是网络输入大小和感受野（输入空间中对单个输出单元有贡献的区域）。输入大小的选择要考虑，为解决一个问题需要的分辨率和上下文。可以增加图像块的大小，以得到更多的上下文，但不修改网络的感受野，这可能没什么好处。作为标准的检查，研究者可以对同样的任务通过视觉检查网络输入。如果他们，或者领域专家，不能得到很好的性能，那么很可能就需要修改网络输入，或网络架构。

The last aspect we want to touch on is model hyper-parameter optimization (e.g. learning rate, dropout rate), which can help squeeze out extra performance from a network. We believe this is of secondary importance with respect to performance to the previously discussed topics and training data quality. Disappointingly, no clear recipe can be given to obtain the best set of hyper-parameters as it is a highly empirical exercise. Most researchers fall back to an intuition-based random search (Bergstra and Bengio, 2012), which often seems to work well enough. Some basic tips have been covered before by Bengio (2012). Researchers have also looked at Bayesian methods for hyper-parameter optimization (Snoek et al., 2012), but this has not been applied in medical image analysis as far as we are aware of.

我们想要接触的最后一个方面是，模型超参数优化（如，学习速率，dropout率），这可以帮助挤压出网络额外的性能。我们相信，除了上面讨论的话题和训练数据的质量，这是具有仅次的重要性的。令人失望的是，还不能给出得到最佳超参数的方案，因为这是非常依赖经验的。多数研究者回到了基于直觉的随机搜索，这似乎就挺好的。一些基本的技巧由Bengio讨论过。研究者还研究过Bayesian方法进行超参数优化，但据我们所知，这还没有应用到医学图像分析中。

*Unique challenges in medical image analysis* 医学图像分析中的独特挑战

It is clear that applying deep learning algorithms to medical image analysis presents several unique challenges. The lack of large training data sets is often mentioned as an obstacle. However, this notion is only partially correct. The use of PACS systems in radiology has been routine in most western hospitals for at least a decade and these are filled with millions of images. There are few other domains where this magnitude of imaging data, acquired for specific purposes, are digitally available in well-structured archives. PACS-like systems are not as broadly used for other specialties in medicine, like ophthalmology and pathology, but this is changing as imaging becomes more prevalent across disciplines. We are also seeing that increasingly large public data sets are made available: Esteva et al. (2017) used 18 public data sets and more than 10^5 training images; in the Kaggle diabetic retinopathy competition a similar number of retinal images were released; and several chest x-ray studies used more than 10^4 images.

很清楚的是，将深度学习算法应用到医学图像分析中，有几个独特的挑战。缺少大规模训练数据集，通常说是一个障碍。但是，这个说法只是部分正确的。放射医疗中使用PACS系统已经是例行工作了，这在西方医院已经至少十年了，至少有数百万张图像。很少有其他领域，为特定目的收集这么多图像数据，还是在存档中有良好的结构的数字影像。类PACS的系统对于其他专家使用的并没有那么广泛，如眼科学和病理学，但这种情况正在改变，因为成像在不同领域正变得越来越流行。我们还观察到，越来越多的大型公开数据集正变得可用：Esteva等使用了18个公开数据集，超过了10^5幅训练图像；在Kaggle糖尿病视网膜病变竞赛中，放出了类似数量的视网膜图像；几个胸部x-射线的研究使用了多于10^4的图像。

The main challenge is thus not the availability of image data itself, but the acquisition of relevant annotations/labeling for these images. Traditionally PACS systems store free-text reports by radiologists describing their findings. Turning these reports into accurate annotations or structured labels in an automated manner requires sophisticated text-mining methods, which is an important field of study in itself where deep learning is also widely used nowadays. With the introduction of structured reporting into several areas of medicine, extracting labels to data is expected to become easier in the future. For example, there are already papers appearing which directly leverage BI-RADS categorizations by radiologist to train deep networks (Kisilev et al., 2016) or semantic descriptions in analyzing optical coherence tomography images (Schlegl et al., 2015). We expect the amount of research in optimally leveraging free-text and structured reports for network training to increase in the near future.

所以主要的挑战并不在图像数据的可用性本身，而是为这些图像得到相关的标注。传统上的PACS系统存储免费的文本报告，这是放射科医生描述的其发现。将这些报告自动变成准确的标注或有结构的标签，需要复杂的文本挖掘方法，这本身就是一个重要的研究领域，而深度学习已经在其中得到了广泛的应用。随着有结构的报告引入到医学的几个领域，从数据中提取标签应当在未来更加容易。比如，已经有文章直接利用BI-RADS类别（放射科医生生成的）来训练深度网络，或用语义描述来分析光学连续性体层摄影图像。我们期待未来在利用免费的文本和有结构的报告来训练网络的研究会有增长。

Given the complexity of leveraging free-text reports from PACS or similar systems to train algorithms, generally researchers request domain experts (e.g. radiologist, pathologists) to make task-specific annotations for the image data. Labeling a sufficiently large dataset can take a significant amount of time, and this is problematic. For example, to train deep learning systems for segmentation in radiology often 3D, slice-by-slice annotations need to be made and this is very time consuming. Thus, learning efficiently from limited data is an important area of research in medical image analysis. A recent paper focused on training a deep learning segmentation system for 3D segmentation using only sparse 2D segmentations (C¸ic ¸ek et al., 2016). Multiple-instance or active learning approaches might also offer benefit in some cases, and have recently been pursued in the context of deep learning (Yan et al., 2016). One can also consider leveraging non-expert labels via crowd-sourcing (Rajchl et al., 2016b). Other potential solutions can be found within the medical field itself; in histopathology one can sometimes use specific immunohistochemical stains to highlight regions of interest, reducing the need for expert experience (Turkki et al., 2016).

利用PACS或类似系统的免费文本报告来训练算法非常复杂，一般研究者要求领域专家（如放射科医生，病理学医生）来对图像数据进行任务专有的标注。标注足够大的数据集需要耗时非常多，这个问题很大。比如，为在放射科中训练分割的深度学习系统，通常需要3D的逐个切片的标注，这个耗时非常的长。因此，从有限的数据中进行高效的学习是医疗图像处理中的重要研究领域。一篇最近的文章聚焦在，为3D分割训练一个深度学习分割系统，使用的只是稀疏的2D分割。多实例或积极学习的方法，在一些情况下可能会有好处，最近也在深度学习的上下文中得到追捧。也可以考虑利用非专家的标签，这是通过众包获得的。其他可能的解决方法可以从医学领域本身获得；在组织病理学中，可以有时候利用特定的免疫组织化学着色来强调感兴趣区域，对专家经验的需要就没有那么多了。

Even when data is annotated by domain expert, label noise can be a significant limiting factor in developing algorithms, whereas in computer vision the noise in the labeling of images is typically relatively low. To give an example, a widely used dataset for evaluating image analysis algorithms to detect nodules in lung CT is the LIDC-IDRI dataset (Armato et al., 2011). In this dataset pulmonary nodules were annotated by four radiologists independently. Subsequently the readers reviewed each others annotations but no consensus was forced. It turned out that the number of nodules they did not unanimously agreed on to be a nodule, was three times larger than the number they did fully agree on. Training a deep learning system on such data requires careful consideration of how to deal with noise and uncertainty in the reference standard. One could think of solutions like incorporating labeling uncertainty directly in the loss function, but this is still an open challenge.

即使数据是由领域专家标注的，标签噪声也可能是发展算法的显著制约因素，而在计算机视觉领域，标签的噪声一般相对较低。一个例子是，一个广泛使用的数据集，评估图像分析算法，来检测肺部CT的小瘤，就是LIDC-IDRI数据集。在这个数据集中，肺部小瘤是由四个放射科医生独立标注的。然后读者审阅其他每个人的标注，但不会强制得到一致意见。结果是，他们没有一致同意的小瘤数，是他们一致同意的数量的三倍。在这样的数据中训练一个深度学习系统，需要仔细考虑怎样处理这些噪声和不确定性。可以采用一些方法，如在损失函数中直接考虑标签的不确定性，但这仍然是一个开放的问题。

In medical imaging often classification or segmentation is presented as a binary task: normal versus abnormal, object versus background. However, this is often a gross simplification as both classes can be highly heterogeneous. For example, the normal category often consists of completely normal tissue but also several categories of benign findings, which can be rare, and may occasionally include a wide variety of imaging artifacts. This often leads to systems that are extremely good at excluding the most common normal subclasses, but fail miserably on several rare ones. A straightforward solution would be to turn the deep learning system in a multi-class system by providing it with detailed annotations of all possible subclasses. Obviously this again compounds the issue of limited availability of expert time for annotating and is therefore often simply not feasible. Some researchers have specifically looked into tackling this imbalance by incorporating intelligence in the training process itself, by applying selective sampling (van Grinsven et al., 2016) or hard negative mining (Wang et al., 2016b). However, such strategies typically fail when there is substantial noise in the reference standard. Additional methods for dealing with within-class heterogeneity would be highly welcome.

在医学影像中，通常分类或分割是一个二值问题：正常的或不正常的，目标或背景。但是，这通常是一个简化，因为两个类别都是高度异质的。比如，正常的类别通常包含了完全正常的组织，但也有几个类别的良性发现，可能非常罕见，但偶尔包含各种不同的杂质。这通常导致系统非常善于排除最常见的正常子类别，但在几种罕见的情况下失败。一种直接的解决方案是，将深度学习系统变成一个多类别系统，给出所有可能的子类别的详细标注。显然，这再次加剧了专家的有限时间进行标注的问题，通常是不太可行的。一些研究者专门研究处理这个不均衡的问题，在训练过程中引入智能，使用了选择性的取样或难分样本挖掘。但是，在有相当多的噪声的时候，这种策略一般都会失败。处理这种类别内异质性的方法会非常受欢迎。

Another data-related challenge is class imbalance. In medical imaging, images for the abnormal class might be challenging to find, depending on the task at hand. As an example, the implementation of breast cancer screening programs has resulted in vast databases of mammograms that have been established at many locations world-wide. However, the majority of these images are normal and do not contain any suspicious lesions. When a mammogram does contain a suspicious lesion this is often not cancerous, and even most cancerous lesions will not lead to the death of a patient. Designing deep learning systems that are adept at handling this class imbalance is another important area of research. A typical strategy we encountered in current literature is the application of specific data augmentation algorithms to just the underrepresented class, for example scaling and rotation transforms to generate new lesions. Pereira et al. (2016) performed a thorough evaluation of data augmentation strategies for brain lesion segmentation to combat class imbalance.

另一个与数据相关的挑战是类别不均衡。在医学图像处理中，非正常类别的图像可能很难找到，这要看待处理的问题。一个例子是，乳腺癌检查项目的进行，得到了巨大的乳腺X光检查数据库，在世界范围内很多地方都有。但是，这些图像的大多数都是正常的，不包含任何可以的损伤。当一个乳腺X光片确实包含可疑的损伤时，通常不是癌症，甚至是癌性最强的损伤也不会使病人致死。设计深度学习系统，要适合处理这种类别不均衡性，是另一个重要的研究领域。我们在文献中遇到的一个典型的策略是，使用特定的数据扩充算法，来纠正不充分表达的类别，比如缩放和旋转变换，以生成新的损伤。Pereira等深入评估了脑损伤分割的数据扩充策略，以处理类别不均衡性问题。

In medical image analysis useful information is not just contained within the images themselves. Physicians often leverage a wealth of data on patient history, age, demographics and others to arrive at better decisions. Some authors have already investigated combining this information into deep learning networks in a straightforward manner (Kooi et al., 2017). However, as these authors note, the improvements that were obtained were not as large as expected. One of the challenges is to balance the number of imaging features in the deep learning network (typically thousands) with the number of clinical features (typically only a handful) to prevent the clinical features from being drowned out. Physicians often also need to use anatomical information to come to an accurate diagnosis. However, many deep learning systems in medical imaging are still based on patch classification, where the anatomical location of the patch is often unknown to network. One solution would be to feed the entire image to the deep network and use a different type of evaluation to drive learning, as was done by, for example, Milletari et al. (2016b), who designed a loss function based on the Dice coefficient. This also takes advantage of the fact that medical images are often acquired using a relatively static protocol, where the anatomy is always roughly in the same position and at the same scale. However, as mentioned above, if the receptive field of the network is small feeding in the entire image offers no benefit. Furthermore, feeding full images to the network is not always feasible due to, for example, memory constraints. In some cases this might be solved in the near future due to advances in GPU technology, but in others, for example digital pathology with its gigapixel-sized images, other strategies have to be invented.

在医学图像分析中 ，有用的信息不止包含在图像中。医生通常更多的利用病患的历史，年龄，人口统计数据和其他的来得到更好的决策。一些作者已经研究了，以一种直接的方式，将这些信息与深度学习网络结合起来。但是，这些作者也说明了，得到的改进并不像期望的那么高。一个挑战是，在深度学习网络的图像特征数量（一般是数千），和临床特征数量（通常只有几个）之间做出均衡，以防止临床特征被淹没。医生通常也需要使用解剖学信息来得到准确的诊断。但是，医学图像分析中的很多深度学习系统还是基于图像块分类的，而这图像块中的解剖学位置通常对于网络是未知的。一个解决方法是，将整幅图像送入深度网络，使用不同类型的评估来驱动学习，如Milletari等就是这样做的，他基于Dice系数设计了一个损失函数。这也利用了一点，即医学图像通常是使用相对静止的协议得到的，其中解剖结构永远大致是相同的位置，相同的尺度。但是，如上面所述的，如果网络的感受野是小的，将整幅图像送入网络毫无益处。而且，将整幅图像送入网络不是永远可行的，比如，由于内存限制。在一些情况中，这可能由于GPU的发展得到解决，但在其他情况中，比如数字病理学的十亿像素大小的图像，必须采用其他的策略。

*Outlook* 展望

Although most of the challenges mentioned above have not been adequately tackled yet, several high-profile successes of deep learning in medical imaging have been reported, such as the work by Esteva et al. (2017) and Gulshan et al. (2016) in the fields of dermatology and ophthalmology. Both papers show that it is possible to outperform medical experts in certain tasks using deep learning for image classification. However, we feel it is important to put these papers into context relative to medical image analysis in general, as most tasks can by no means be considered ’solved’. One aspect to consider is that both Esteva et al. (2017) and Gulshan et al. (2016) focus on small 2D color image classification, which is relatively similar to the tasks that have been tackled in computer vision (e.g. ImageNet). This allows them to take advantage of well-explored network architectures like ResNet and VGG-Net which have shown to have excellent results in these tasks. However, there is no guarantee that these architectures are optimal in for example regressions/detection tasks. It also allowed the authors to use networks that were pre-trained on a very well-labeled dataset of millions of natural images, which helps combat the lack of similarly large, labeled medical datasets. In contrast, in most medical imaging tasks 3D gray-scale or multi-channel images are used for which pre-trained networks or architectures dont exist. In addition this data typically has very specific challenges, like anisotropic voxel sizes, small registration errors between varying channels (e.g. in multi-parametric MRI) or varying intensity ranges. Although many tasks in medical image analysis can be postulated as a classification problem, this might not always be the optimal strategy as it typically requires some form of post-processing with non-deep learning methods (e.g. counting, segmentation or regression tasks). An interesting example is the paper by Sirinukunwattana et al. (2016), which details a method directly predicting the center locations of nuclei and shows that this outperforms classification-based center localization. Nonetheless, the papers by Esteva et al. (2017) and Gulshan et al. (2016) do show what ideally is possible with deep learning methods that are well-engineered for specific medical image analysis tasks.

虽然上面提到的大多数挑战仍未充分解决，但深度学习在医学图像处理中有几个很成功的案例得到了报道，如Esteva等和Gulshan等，应用领域为皮肤病学和眼科学。两篇文章表明，在某些任务中，使用深度学习进行图像分类，是可能超过医学专家的表现的。但是，我们感觉，将这些文章放入医学图像分析的上下文下非常重要，因为大多数任务都尚未达到解决的程度。要考虑到的一个方面是，Esteva等和Gulshan等关注的都是较小的2D彩色图像分类，这与计算机视觉中已经解决的问题比较接近（如ImageNet）。这使其可以利用很成熟的网络架构，如ResNet和VGG-Net，在这些任务中都有非常好的表现。但是，这些架构在回归/检测这样的任务中是不是最优的，是没有保证的。作者也可能使用预训练的网络，这可能是在标注的非常好的包含上百万幅图像的数据集上训练的，而类似大型的标注好的医学图像数据集则是缺乏的。比较之下，在大多数医学图像任务中，会使用3D灰度或多通道图像，这种情况下预训练的网络架构是不存在的。而且，这些数据一般都有很特定的挑战，如各向异性的voxel大小，不同通道中很小的配准误差（如，多参数MRI），或不同的灰度范围。虽然医学图像分析中的很多任务都可以假定为分类问题，但这可能不一定是最优的策略，因为一般都需要一些非深度学习方法的后处理（如，计数，分割或回归任务）。一个有趣的例子是Sirinukunwattana等的文章，详细描述了一个预测细胞核中心位置的方法，表明这超过了基于分类的中心定位方法。尽管如此，Esteva等和Gulshan等的文章确实表明，深度学习方法对特定医学图像处理任务是非常理想的。

Looking at current trends in the machine learning community with respect to deep learning, we identify a key area which can be highly relevant for medical imaging and is receiving (renewed) interest: unsupervised learning. The renaissance of neural networks started around 2006 with the popularization of greedy layerwise pre-training of neural networks in an unsupervised manner. This was quickly superseded by fully supervised methods which became the standard after the success of AlexNet during the ImageNet competition of 2012, and most papers in this survey follow a supervised approach. However, interest in unsupervised training strategies has remained and recently has regained traction.

目前机器学习中深度学习的研究趋势，我们研究后发现了一个关键领域，与医学图像分析高度相关，并不断得到关注：无监督学习。神经网络的复兴大约是从2006年开始的，其训练的方式是逐层无监督预训练。这迅速的被完全监督方法训练超过了，在AlexNet在ImageNet 2012 比赛成功之后，成为了标准方法，本文中大多数文章都使用的是监督的方法。但是，无监督训练的方法仍然有很多研究。

Unsupervised methods are attractive as they allow (initial) network training with the wealth of unlabeled data available in the world. Another reason to assume that unsupervised methods will still have a significant role to play is the analogue to human learning, which seems to be much more data efficient and also happens to some extent in an unsupervised manner; we can learn to recognize objects and structures without knowing the specific label. We only need very limited supervision to categorize these recognized objects into classes. Two novel unsupervised strategies which we expect to have an impact in medical imaging are variational auto-encoders (VAEs), introduced by Kingma and Welling (2013) and generative adversarial networks (GANs), introduced by Goodfellow et al. (2014). The former merges variational Bayesian graphical models with neural networks as encoders/decoders. The latter uses two competing convolutional neural networks where one is generating artificial data samples and the other is discriminating artificial from real samples. Both have stochastic components and are generative networks. Most importantly, they can be trained end-to-end and learn representative features in a completely unsupervised manner. As we discussed in previous paragraphs, obtaining large amounts of unlabeled medical data is generally much easier than labeled data and unsupervised methods like VAEs and GANs could optimally leverage this wealth of information.

无监督的方法非常有吸引力，因为网络可以用未标注的数据进行训练，这是非常多的。我们假设无监督方法仍然会有显著的影响，另一个原因是与人类学习的类比，人类在学习上非常高效，某种程度上也是无监督的；我们可以学习识别物体和结构，而不需要知道其具体的标签。我们只需要非常有限的监督，来讲识别的物体分类。两个新的无监督策略，我们期望在医学图像领域中会有影响，是变分自动编码机(VAEs)，和生成对抗网络(GANs)。前者将变分贝叶斯图模型与神经网络结合，成为编码器/解码器。后者使用两个竞争的CNNs，一个生成人工数据样本，另一个将人工数据与真实样本区分开来。两者都有随机的成分，都是生成式网络。最重要的是，它们都可以进行端到端的训练，以一种无监督的方式，学习到有代表性的特征。如同我们的之前的段落中所讨论的，得到大量未标记的医学图像数据是非常容易的，无监督方法如VAEs和GANs可以很好的利用这些信息。

Finally, deep learning methods have often been described as ‘black boxes’. Especially in medicine, where accountability is important and can have serious legal consequences, it is often not enough to have a good prediction system. This system also has to be able to articulate itself in a certain way. Several strategies have been developed to understand what intermediate layers of convolutional networks are responding to, for example deconvolution networks (Zeiler and Fergus, 2014), guided back-propagation (Springenberg et al., 2014) or deep Taylor composition (Montavon et al., 2017). Other researchers have tied prediction to textual representations of the image (i.e. captioning) (Karpathy and FeiFei,2015), which is another useful avenue to understand what a network is perceiving. Last, some groups have tried to combine Bayesian statistics with deep networks to obtain true network uncertainty estimates Kendall and Gal (2017). This would allow physicians to assess when the network is giving unreliable predictions. Leveraging these techniques in the application of deep learning methods to medical image analysis could accelerate acceptance of deep learning applications among clinicians, and among patients. We also foresee deep learning approaches will be used for related tasks in medical imaging, mostly unexplored, such as image reconstruction (Wang, 2016). Deep learning will thus not only have a great impact in medical image analysis, but in medical imaging as a whole.

最后，深度学习方法通常被描述为黑盒子。尤其在医学领域中，责任是非常重要的，可能会有严重的法律后果，只有一个好的预测系统是不够的。这个系统还必须要以某种方式很好的表达自己。已经提出了几种策略来理解，CNNs的中间层响应的是什么，比如解卷积网络，引导反向传播，或深度Taylor分解。其他的研究者将预测与图像的文本描述结合到了一起（即，加标题），这对理解网络感知也是非常有好处的。最后，一些小组尝试将贝叶斯统计与深度网络结合到一起，得到真正的网络不确定性估计。这使得医生可以评估，网络什么适合会给出不可靠的预测。在深度学习在医学图像处理中的应用中利用这些技术，可以使深度学习技术更加为医生所接受，也为病人所接受。我们还预测，深度学习方法可以用于医学图像处理的相关任务，大多数还尚未探索，比如，图像重建。深度学习将不止在医学图像分析中有影响，也在医学成像中会有影响。