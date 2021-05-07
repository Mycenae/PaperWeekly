# FFT-Based Deep Learning Deployment in Embedded Systems

Sheng Lin, et. al. Syracuse University etc.

## 0. Abstract

Deep learning has delivered its powerfulness in many application domains, especially in image and speech recognition. As the backbone of deep learning, deep neural networks (DNNs) consist of multiple layers of various types with hundreds to thousands of neurons. Embedded platforms are now becoming essential for deep learning deployment due to their portability, versatility, and energy efficiency. The large model size of DNNs, while providing excellent accuracy, also burdens the embedded platforms with intensive computation and storage. Researchers have investigated on reducing DNN model size with negligible accuracy loss. This work proposes a Fast Fourier Transform (FFT)-based DNN training and inference model suitable for embedded platforms with reduced asymptotic complexity of both computation and storage, making our approach distinguished from existing approaches. We develop the training and inference algorithms based on FFT as the computing kernel and deploy the FFT-based inference model on embedded platforms achieving extraordinary processing speed.

深度学习在很多应用领域中都展现了实力，尤其是图像和语音识别。作为深度学习的骨干，DNNs是由多个层组成的，每层都是成百上千个各种类型的神经元。嵌入式平台是深度学习部署的关键，因为其便携，通用，能耗低。DNNs的大型模型准确率非常好，但计算量大，需要存储空间大，构成了嵌入式平台的负担。研究者调查了降低DNN模型的大小，同时准确率损失要可以忽略不计。本文提出了基于FFT的DNN训练和推理模型，适用于嵌入式平台，计算和存储的渐进复杂度都下降了，相比于已有方法更加优秀。我们提出了基于FFT的训练和推理算法作为计算核心，将基于FFT的推理模型部署到了嵌入式系统，获得了极高的处理速度。

## 1. Introduction

Recently deep learning has outstood from traditional machine learning techniques in many application areas, especially in image and speech recognition [1], [2]. The excellence of deep learning has also resulted in explorations of several emerging real-world applications, such as self-driving systems [3], automatic machine translations [4], drug discovery and toxicology [5]. The deep learning is based on the structure of deep neural networks (DNNs), which consist of multiple layers of various types and hundreds to thousands of neurons in each layer. Recent evidence has revealed that the network depth is of crucial importance to the success of deep learning, and many deep learning models for the challenging ImageNet dataset are sixteen to thirty layers deep [1]. Deep learning achieves significant improvement in overall accuracy by extracting complex and high-level features at the cost of considerable up-scaling in the model size.

最近深度学习在很多应用领域中超越了传统机器学习技术，尤其是在图像和语音识别中。深度学习的优越性也使得对一些正在出现的真实世界应用进行探索，比如自动驾驶系统，自动机器翻译，药物发现和毒理学。深度学习是基于DNN的架构，是由多层组成的，每层包含成百上千个各种类型的神经元。最近的证据表明，网络深度是深度学习成功的关键，对ImageNet数据集，很多深度学习模型都是16-30层深。深度学习通过提取复杂的高层次的特征，取得了显著的准确率提升，其代价是模型大小的极大扩大。

In the big data era and driven by the development of semiconductor technology, embedded systems are now becoming an essential computing platform with ever-increasing functionalities. At the same time, researchers around the world from both academia and industry have devoted significant efforts and resources to investigate, improve, and promote the applications of deep learning in embedded systems [6]. Despite the advantages in DNN recognition accuracy, the deep layered structure and large model size of DNNs also increase computational complexity and memory requirement. Researchers are faced with the following challenges when deploying deep learning models on embedded systems: (i) Confined by the communication bandwidth of embedded systems, which are usually mobile terminals, it is still challenging to download large-size DNN models, even which can be offline-trained in data centers. (ii) The large model size of deep learning also imposes stringent requirements on the computing resources and memory size of embedded systems.

在大数据时代，受半导体技术开发的驱动，嵌入式系统正成为计算平台的关键，其功能日益增加。同时，世界范围内的研究者，包括学术界和工业界，都投入了很大的精力和资源，来研究、改进、推动深度学习在嵌入式系统中的应用。尽管DNN识别准确率有优势，其深度架构和模型大体积，也增加了计算复杂度和内存需求。研究者面临着在嵌入式系统中部署深度学习，面临着下面的困难：(i)受限于嵌入式系统的通信带宽，通常是移动终端，要下载很大的DNN模型，仍然有挑战，这些模型是在数据中心离线训练的，(ii)深度学习模型很大，对嵌入式系统的计算资源和内存大小也提出了很严苛的要求。

Motivated by these challenges, it is intuitive to implement a reduced-size deep learning model with negligible accuracy loss. In fact, the state-of-the-art DNNs are often over-parameterized, hence the removal of redundant parameters in the deep learning models, if performed properly, will produce similar overall accuracy as the original models [1]. Encouraged by this discovery, various deep learning model compression approaches have been investigated [6]–[10], including weight precision reduction, network pruning, weight matrix factorization, etc. In this work, we propose a Fast Fourier Transform (FFT)-based DNN training and inference model suitable for embedded systems due to reduced asymptotic complexity of both computation and storage. Our approach has obvious advantages over existing works on deep learning model compression e.g., [6], [8], [9] in that those approaches result in an irregular network architecture that increases training and inference computation time, while our approach facilitates computation. Please also note the our proposed framework is distinct from the prior work of using FFT for convolutional layer acceleration by LeCun et al. [11], because this prior work can only achieve convolutional layer acceleration instead of simultaneous compression. We develop the training and inference algorithms based on FFT as the computing kernel and deploy the FFT-based inference model on embedded platforms. Experimental test results demonstrate that our model provides the optimization in different languages and achieve a significant improvement.

受到这些挑战推动，实现一个缩小版的深度学习模型，准确率下降可以忽略，这就是一个很直觉的问题。实际上，目前最好的DNNs通常都是过参数化的，因此去除深度学习模型中的冗余参数，如果合理的进行，会与原始模型得到类似的总体准确率。受到这个发现鼓励，各种深度学习模型压缩方法都进行了研究，包括降低权重精度，网络剪枝，权重矩阵分解，等。本文中，我们提出了一种基于FFT的DNN训练和推理模型，适用于嵌入式系统，因为计算和存储复杂度都极大了降低了。我们的方法比现有的模型压缩工作有明显的优势，如[6,8,9]中的方法得到了不规则的网络结构，增加了训练和推理的计算时间，而我们的方法则促进了计算。还要注意到，我们提出的框架，与之间LeCun等[11]提出的用FFT加速CNN的方法是不一样的，因为[11]只能对卷积层进行加速，而不同进行同时压缩。我们提出的基于FFT的训练和推理算法是计算核心，将基于FFT的推理模型部署到嵌入式系统中。试验性的测试结果表明，我们的模型以不同的方式进行了优化，获得了明显的改进。

## 2. Related Work

Over the past decade, a substantial number of techniques and strategies have been proposed to compress neural network size. Weight pruning [6] is a well-known effective approach, in which many weights with values of 0 are pruned to achieve high compression ratio. Other techniques such as threshold setting [6], biased weight decay [9], etc., could be integrated to the weight pruning procedure. Another simple and popular approach to DNN model compression is the low-rank approximation of the weight matrix [12]. To overcome the potential high accuracy loss after low-rank approximation, [13] proposed to perform fine-tuning for the post-factorization of low-rank weight matrices to retain accuracy. Lowering the presentation precision of weights is also an straightforward technique to reduce both the model size and computation cost of DNNs. A fixed-point implementation was explored to replace the original floating-point models [14]. Furthermore, designs with ultra-low precision weights, such as binary (-1/+1) or ternary (-1/0/+1) representation were proposed [15], [16]. By exploring the local and global characteristics of the weight matrix, weight clustering was proposed to reduce the number of weights linearly [17]. In addition, with the aid of gradients clustering in the training phase, the accuracy loss incurred by the weight clustering can be negligible [6].

在过去十年，提出了很多压缩网络大小的方法。权重剪枝是一种著名的有效方法，其中很多值为0的权重被剪掉，以获得很高的压缩率。其他技术比如阈值设置，偏置权重衰减等，可以集成到权重剪枝过程中。另一种简单和流形的DNN模型压缩方法是权重矩阵的低秩近似。为克服低秩近似的高准确率损失，[13]提出对低秩权重矩阵的后分解进行精调，以重新获得准确率。降低权重的精度也是一种很直接的降低模型大小和计算代价的技术。[14]探索了一种定点实现，替换了原始的浮点模型。而且，[15,16]提出了极低精度权重的设计，比如二值表示(-1/+1)或三值表示(-1/0/+1)。[17]探索了权重矩阵的局部和全局特征，提出了权重聚类以线性的降低权重数量。另外，[6]在训练阶段的梯度聚类的帮助下，权重聚类带来的准确率损失是可以忽略不计的。

Some recent works adopted structured weight matrices in order to reduce the model size. In [18], weight matrices of fully-connected (FC) layers were constructed in the Toeplitz-like format to remove the redundancy of the DNN model. In [19], the circulant matrix was introduced to enable further reduction in model size. An n-by-n circulant matrix has a smaller number of parameters i.e., n than that of a same-size Toeplitz matrix i.e., 2n. In this work, we generalize the structured weight matrix method in that (1) we utilize block-circulant matrices for weight matrix representation, which achieves a trade-off between compression ratio and accuracy loss; (2) we extend the structured matrix method to convolutional (CONV) layers besides the FC layers; (3) we propose FFT-based DNN training and inference model and algorithm, which is highly suitable for deployment in embedded systems; and (4) we implement and test the FFT-based DNN inference in various embedded platforms.

一些最近的工作采用了结构化的权重矩阵，以降低模型大小。[18]中，全连接层的权重矩阵，用Toeplitz类的格式进行了构建，以去除了DNN模型的冗余性。[19]中，用循环矩阵进一步降低模型大小。一个nxn的循环矩阵的参数数量更少，即，同样大小的Toeplitz矩阵参数为2n，而循环矩阵则为n。本文中，我们推广了权重矩阵结构化的方法，(1)利用了分块循环矩阵进行权重矩阵表示，在压缩率和准确率损失之间得到了平衡，(2)我们将结构化矩阵从FC层拓展到了卷积层中，(3)我们提出了基于FFT的DNN训练和推理模型和算法，非常适用于在嵌入式系统中的部署，(4)我们在多个嵌入式平台中实现并测试了基于FFT的推理。

## 3. Background

In this section, we introduce basic concepts of deep neural networks (DNNs), Fast Fourier Transform (FFT), and structured matrices, as the background of our proposed FFT-based training and inference algorithms. Specifically, we explain the various DNN layer types, the Cooley-Tukey algorithm for FFT, and the block-circulant matrices as the adopted structured matrices.

本节中，我们提出了DNN，FFT和结构化矩阵的基本概念，作为我们提出的基于FFT的训练和推理算法的背景。具体的，我们解释了各种DNN层的类型，Cooley-Tukey FFT算法，采用的结构化矩阵即分块循环矩阵。

### 3.1. Deep Neural Networks

Deep neural networks (DNNs) are distinguished from other types of neural networks by their depth and have dramatically improved the state-of-the-art in speech recognition, object detection, etc. Some commonly adopted DNN models include deep convolutional neural networks, deep belief networks, and recurrent neural networks. Despite the various network topologies targeting for different applications, these DNN models comprise of multiple functional layers with some commonly used structures. Following are the most commonly used layer structures in the state-of-the-art DNN models:

DNNs以其深度并显著改进了语音识别，目标检测等的性能而区别于其他神经网络类型。一些广泛采用的DNN模型包括，DCNN，深度置信网络，RNN。尽管对各种应用有各种网络拓扑，这些DNN模型包含多个功能层，有一些常用的结构。下面是最常用的层的结构：

The fully-connected (FC) layer is the most storage-intensive layer in DNN models [20] since each of its neurons is fully connected with all the neurons in the previous layer. The computation procedure of a FC layer consists of matrix-vector arithmetics (multiplication and addition) and transformation by the activation function, described as follows:

全连接FC层是DNN模型中最需要存储的层，因为每个神经元都与前一层的所有神经元完全连接。一个FC层的计算过程，由矩阵-向量代数运算（相乘和相加），和激活函数的变换组成，如下式：

$$y = ψ(W^T x + θ)$$(1)

where y and x are outputs of this layer and the previous layer, respectively; $W ∈ R^{m×n}$ is the weight matrix of the synapses between this FC layer (with n neurons) and its previous layer (with m neurons); θ ∈ R^n is the bias vector; and ψ(⋅) is the activation function. The Rectified Linear Unit (ReLU) ψ(x) = max(0, x) is the most widely utilized activation function in DNNs.

其中y和x是这个层和前一层的输出；$W ∈ R^{m×n}$是这一层（n个神经元）和前一层（m个神经元）的突触的权重矩阵；θ ∈ R^n是偏置向量；ψ(⋅)是激活函数。ReLU单元ψ(x) = max(0, x)是DNNs中最广泛使用的激活函数。

The convolutional (CONV) layer, as the name implies, performs two-dimensional convolution of its input to extract features that will be fed into subsequent layers for higher-level feature extracting. A CONV layer is associated with a set of learnable filters [21], which are activated when specific types of features are found at some spatial positions from the inputs. Filter-sized moving windows are applied to the inputs to obtain a set of feature maps, by calculating the convolution of the filter and inputs in the moving window. Each convolutional neuron, representing one pixel in a feature map, takes a set of inputs and the corresponding filter weights to calculate the inner-product. Given input feature map X and the r×r-sized filter (i.e., the convolutional kernel) F, the output feature map Y is calculated as

卷积层对输入进行二维卷积，提取出的特征送入后续的层中，进行更高层的特征提取。卷积层与可学习的滤波器集合有关，当特定类型的特征在输入的某空间位置发现时，就被激活了。滤波器大小的滑窗应用到输入中，计算滤波器和输入在滑窗中的卷积，以得到特征图集。每个卷积神经元，表示特征图中的一个像素，以输入和对应的滤波器权重为输入，来计算点积。给定输入特征图X和rxr大小的滤波器（即，卷积核）F，输出特征图Y计算如下

$$y_{a,b} = \sum_{i=1}^r \sum_{j=1}^r x_{a+i-1, b+j-1} × f_{i,j}$$(2)

where $y_{a,b}, x_{a+i-1, b+j-1}$ and $f_{i,j}$ are elements in Y, X, and F, respectively. Multiple convolutional kernels can be adopted to extract different features in the same input feature map. Multiple input feature maps can be convolved with the same filter and results are summed up to derive a single feature map.

其中$y_{a,b}, x_{a+i-1, b+j-1}$和$f_{i,j}$分别是Y，X和F中的元素。可以采用多个卷积核，在相同的输入特征图中提取不同的特征。多个输入特征图可以与同一个滤波器相卷积，结果叠加到一起，得到单个特征图。

### 3.2 Fast Fourier Transforms

The Fast Fourier Transform (FFT) is an efficient procedure for computing the discrete Fourier transform (DFT) of time series. It takes advantage of the fact that the calculation of the coefficients of the DFT can be carried out iteratively, which results in a considerable savings of computation time. The FFT not only reduces the computational complexity, but also substantially reduces round-off errors associated with these computations. In fact, both the computation time and round-off error are essentially reduced by a factor of n/(log_2 n) where n is the number of data samples in the time series [22]. Fig. 1 shows the simplest and most common form of FFT, which is based on the Cooley-Tukey algorithm [23]. It uses a divide and conquer approach to recursively break down the DFT of an arbitrary composite size N = N1 ⋅ N2 into many smaller DFTs of sizes N1 and N2, in order to reduce the computation time to O(n log n) for highly composite N [23].

FFT是计算离散序列DFT的一种高效方法。它利用了计算DFT系数的过程可以迭代进行的事实，这样就可以极大的节省计算时间。FFT不仅降低了计算复杂度，还极大的降低了与这些计算相关的四舍五入误差。实际上，计算时间和舍入误差极大了降低了n/(log_2 n)倍，其中n是时序中的数据样本数量。图1展示了最简单最常用的FFT形式，这是基于Cooley-Tukey算法的。它使用了分解征服的方法，来迭代的将任意组合大小N = N1 ⋅ N2的DFT分解成很多更小的大小为N1和N2的DFTs，以将计算时间O(n log n)降低到N。

### 3.3 Structured Matrices

An n-by-m matrix A is called a structured matrix when it has a low displacement rank υ [18]. One of the most important characteristics of structured matrices is their low number of independent variables. The number of independent parameters is O(n) for an n-by-n structured matrix instead of O(n^2), which indicates that the storage complexity can be potentially reduced to O(n). As a representative example, a circulant matrix W ∈ R^n×n is defined by a vector w = (w1, w2, ..., wn) as follows:

一个nxm的矩阵A称为结构化矩阵，如果其位移秩U很低。结构化矩阵的一个重要特性是，独立变量数量很少。对于一个nxn的结构化矩阵，其独立参数量是O(n)，而不是O(n^2)，这说明，其存储复杂度可能降到O(n)。作为一个代表性例子，一个循环矩阵W ∈ R^n×n可以由一个向量w = (w1, w2, ..., wn)来进行定义，如下：

$$\left[ \begin{matrix} w_1 & w_n & ... & w_3 & w_2 \\ w_2 & w_1 & w_n & ... & w_3 \\ ... & ... & ... & ... & ... \\ w_{n-1} & ... & ... & ... & w_n \\ w_n & w_{n-1} & ... & w_2 & w_1 \end{matrix} \right]$$

The definition and analysis of structured matrices have been generalized to the case of m-by-n matrices where m $\neq$ n, e.g., the block-circulant matrices. Besides, the computational complexity for many matrix operations, such as matrix-vector multiplication, matrix inversion, etc., can be significantly reduced when operating on structured matrices.

结构化据怎的定义和分析已经泛化到了mxn的矩阵，其中m $\neq$ n，如，分块循环矩阵。除此以外，很多矩阵运算的计算复杂度，比如矩阵向量乘积，矩阵逆，等，当矩阵是结构化矩阵时，可以得到显著的降低。

## 4. Fast Fourier Transform-Based DNN Model

In this section, we propose an efficient inference algorithm and explain the training algorithm in deep neural networks by using block-circulant matrices. We achieve a simultaneous and significant reduction in computational complexity of inference and training processes, and also weight storage. Besides, we have performed theoretical analysis to prove the effectiveness of substituting matrix multiplication with the Fast Fourier Transform method and utilizing block-circulant matrices, thereby guaranteeing applicability of the proposed framework on a wide variety of applications and emerging deep learning models.

本节中，我们提出了一种高效的推理算法，用分块循环矩阵解释了DNN中的训练算法。在训练和推理过程的计算复杂度，和权重存储上，我们同时获得了显著的下降。另外，我们进行了理论分析，证明了将矩阵相乘替换为FFT方法并利用分块循环矩阵的有效性，因此确保了提出的框架可以应用在非常广泛的应用和正在出现的深度学习模型中。

## 4.1. Block-Circulant Matrix-Based Inference and Training Algorithms for FC Layers

Cheng et al. proposed circulant matrix-based DNN training and inference algorithms for FC layers [19]. However, in many practical applications such schemes cannot be directly used because: (1) It is very common that the weight matrices of DNNs are non-square matrices due to the specific need of different applications; and (2) Even if the weight matrices are square, in many cases the compression is too aggressive and hence causes non-negligible accuracy loss. To address the above challenges, we present the block-circulant matrix-based inference and training algorithms.

Cheng等[19]提出了基于循环矩阵的DNN训练和推理算法。但是，在很多实际的应用中，这样的方案不能直接应用，因为：(1)由于不同的应用有专门的需要，DNNs的权重矩阵不一定是方阵；(2)即使权重矩阵是方阵，在很多情况中，压缩方法太激进了，导致了准确率有一些下降。为处理上述挑战，我们提出基于分块循环矩阵的推理和训练算法。

Recall that the forward propagation during the inference phase of a FC layer is performed as $y = ψ(W^T x + θ)$, where ψ is the activation function, W is the weight matrix, x is the input vector, and θ is the biases. The computation bottleneck is the calculation of W^T x. When using a block-circulant matrix for representing W, a fast multiplication algorithm for W^T x exists, which will result in a significant reduction in computational complexity. Assume that the weight matrix is an m-by-n block-circulant matrix W = [C1|C2|...|Ck]^T; the input vector is x = (x1|x2|...|xk); and the bias vector is θ = (θ1|θ2|...| θk). Each circulant matrix Ci ∈ R^n×n is defined by a length-n vector wi = (wi,1, wi,2, ..., wi,n)^T, i ∈ {1, ..., k}, m = kn, and xi = (xi,1, xi,2, ..., xi,n)^T. Hence, W^T x, as the key computation bottleneck in the inference phase, can be simplified as below:

回忆一下，推理阶段的前向传播中，一个FC层是$y = ψ(W^T x + θ)$，其中ψ是激活函数，W是权重矩阵，x是输入向量，θ是偏置。计算瓶颈是W^T x的计算。当使用分块循环矩阵来表示W时，计算W^T x存在一个快速相乘算法，计算复杂度会有明显的下降。假设权重矩阵是一个mxn的分块循环矩阵W = [C1|C2|...|Ck]^T；输入向量是x = (x1|x2|...|xk)；偏置向量θ = (θ1|θ2|...| θk)。每个循环矩阵Ci ∈ R^n×n由一个长度为n的向量wi = (wi,1, wi,2, ..., wi,n)^T定义，i ∈ {1, ..., k}，m=kn，xi = (xi,1, xi,2, ..., xi,n)^T。因此，W^T x，作为推理阶段的关键计算瓶颈，可以简化为：

$$W^T x = \sum_{i=1}^k C_ix_i = \sum_{i=1}^k IFFT(FFT(w_i)◦FFT(x_i))$$(3)

where FFT, IFFT, and ◦ represent a Fast Fourier transform (FFT), an inverse FFT, and an element wise multiplication, respectively. This “FFT → component-wise multiplication → IFFT" procedure to implement W^T x shown in Fig. 2 is derived from the circular convolution theorem [24], [25]. The overall computational complexity in this FC layer will be O(n log n), achieving a significant reduction compared to O(n^2) when calculating W^T x directly. In order to store the weights for the inference phase, we can simply keep the FFT result FFT(w_i) (which is a vector) instead of the whole matrix W, thereby reducing the storage complexity to O(n) for an FC layer. Algorithm 1 summarizes the FFT-based inference algorithm.

这里FFT，IFFT和◦分别表示FFT，IFFT和逐元素的乘积。这个“FFT→逐元素乘积→IFFT”的过程实现W^T x的方法，如图2所示，这是从循环卷积定理推理得到的。在这个FC层中的总体计算复杂度是O(n log n)，当直接计算W^T x时为O(n^2)，这有了显著的下降。为存储推理阶段的权重，我们可以简单的保存FFT的结果FFT(w_i)（这是一个向量），而不是整个矩阵W，因此将FC层的存储复杂度降低到O(n)。算法1总结了基于FFT的推理算法。

Besides the inference procedure, the reformulated training (weight updating) algorithm in the scenario of using block-circulant matrices will also result in significant accelerations. We denote a = W^T x + θ = (a1|a2|...|ak)^T and ai = (ai,1, ai,2, ..., ai,n)^T, then the weight updating rule for the block-circulant FC layer is given by:

除了推理的过程，在这个场景中使用分块循环矩阵来重新表述训练（权重更新）的算法，也会得到显著的加速。我们表示a = W^T x + θ = (a1|a2|...|ak)^T，ai = (ai,1, ai,2, ..., ai,n)^T，然后对分块循环FC层的权重更新规则由下式给出

$$w_i ← w_i - ϵ ⋅ IFFT(FFT(\frac {∂L}{∂a_i}) ∘ FFT(x'_i))⋅I$$(4)

where L, I, ϵ, and x'_i represent the loss function, an all-one column vector, the learning rate, and the base vector that defines the circulant matrix ∂a_i/∂w_i (which is formally derived), respectively. Notice that since ∂a_i/∂w_i is a circulant matrix, similar to inference, we can utilize the “FFT→component-wise multiplication→IFFT" procedure to accelerate the matrix-vector multiplication. The computational complexity will be O(n log n) in each updating step in this layer, which is a significant reduction from O(n^2) in traditional backpropagation procedure. Algorithm 2 summarizes the FFT-based training algorithm.

其中L表示损失函数，I表示全1列向量，ϵ为学习速率，x'_i为定义了循环矩阵∂a_i/∂w_i的基础向量。注意，由于∂a_i/∂w_i是一个循环矩阵，与推理类似，我们可以利用“FFT→点乘→IFFT”的过程，来加速矩阵向量的乘法。计算复杂度在每个更新步骤中在这一层是O(n log n)，传统的反向传播过程是O(n^2)，这是显著的下降。算法2总结了基于FFT的训练算法。

### 4.2. Block-Circulant Matrix-Based Inference and Training Algorithms for CONV Layer

The use of block-circulant matrices can also enable significant reduction in computational and storage complexities of the Convolutional layer. The Convolutional layers are often associated with multiple input and output feature maps in DNNs. Therefore, the computation of the Convolutional layer is described using tensor format as follows:

使用分块循环矩阵，也可以使卷积层的计算复杂度和存储复杂度得到显著下降。DNNs中的卷积层，一般是多输入和多输出的。因此，卷积层的计算是用张量形式描述的，如下所示：

$$y(x,y,p) = \sum_{i=1}^r \sum_{j=1}^r \sum_{c=1}^C F(i,j,c,p) χ(x+i-1,y+j-1,c)$$(5)

where χ∈R^W×H×C, y∈R^(W-r+1)×(H-r+1)×P, F∈R^r×r×C×P denote the input, output, and weight "tensors" of the Convolutional layer, correspondingly. C is the number of input maps. W and H are the spatial dimensions of the input maps. P is the total number of output maps, and r is the size of the convolutional kernel.

其中χ∈R^W×H×C, y∈R^(W-r+1)×(H-r+1)×P, F∈R^r×r×C×P分别对应着卷积层的输入，输出和权重张量。C是输入图的数量。W和H是输入图的空间维度。P是输出图的总计数量，r是卷积核的大小。

We generalize the "block-circulant structure" as rank-4 tensor (F) in the Convolutional layer, i.e., each slice F (⋅, ⋅, i, j) is a circulant matrix. Then, we reformulate the inference and training algorithms of the Convolutional layer to matrix-based operations.

我们将卷积层中的分块循环结构泛化为秩为4的张量F，即，每个slice F (⋅, ⋅, i, j)都是一个循环矩阵。然后，我们重新表述卷积层中的推理和训练算法为基于矩阵的运算。

In the Convolutional layer, to enhance the implementation efficiency, software tools provide an efficient approach of changing tensor-based operations to matrix-based operations equivalently [26], [27]. Fig. 3 demonstrates the application of the method to reformulate Eqn. (3) to the matrix multiplication Y = XF , where X ∈ R^(W-r+1)(H-r+1)×Cr^2, Y ∈ R^(W-r+1)(H-r+1)×P , and F ∈ R^Cr^2×P.

在卷积层中，为强化实现效率，软件工具给出了高效的方法，将基于张量的运算变换为基于矩阵的运算。图3表明了将这个方法应用到式3上重新表述矩阵乘积Y=XF，其中X ∈ R^(W-r+1)(H-r+1)×Cr^2, Y ∈ R^(W-r+1)(H-r+1)×P , 和F ∈ R^Cr^2×P。

Based on the reshaping principle between F and FF, we have: 基于F和FF的重新改变形状的原则，我们有：

$$f_{a+C(i-1)+Cr(j-1),b} = f_{C(i-1)+Cr(j-1), b-a}, ∀a, b$$(6)

where FF is a block-circulant matrix. Therefore, the "FFT→component-wise multiplication →IFFT" procedure can be applied to accelerate Y = XF, leading to the acceleration of (3). With the assist of the proposed approach, the computational complexity for (3) is reduced from O(WHr^2CP) to O(WHQ log Q), where Q = max(r^2C, P).

其中F是一个分块循环矩阵。因此，“FFT→逐元素乘积→IFFT”过程可以应用，以加速Y=XF的计算，得到(3)的加速。有了提出的方法的支持，(3)的计算复杂度从O(WHr^2CP)下降到O(WHQ log Q)，其中Q = max(r^2C, P)。

## 5. Software Implementation

In this section, we provide detailed explanation of our software implementation, experimental setup, and evaluation of the proposed inference framework on various Android-based platforms with embedded processors and various datasets. The purpose of this software implementation is to reveal the potential of embedded systems in running real time applications that involve deep neural networks.

本节中，我们给出了软件实现、试验设置的详细解释，并将提出的推理框架在各种带有嵌入式处理的基于Android的平台和各种数据集上进行评估。软件实现的目的是，展现嵌入式系统在运行DNNs的实时应用的潜力。

The software implementation of proposed inference framework for Android-based platforms is comprised of four high-level modules. The first module is responsible for constructing the network architecture. The second module reads a file that contains trained weights and biases. The third module loads test data that consists of input features and predefined classification labels, and finally, the fourth module performs inference for predicting labels. Fig. 4 depicts these high-level building blocks of the software implementation, along with their interactions. It should be noted that the test data may be loaded from a file, camera, etc.

提出的推理框架的软件实现，是在基于Android的平台上的，由四个高层模块组成。第一个模块负责构建网络架构，第二个模块读取训练好的权重和偏置文件，第三个模块载入测试数据，由输入特征和预定义的分类标签组成，最后，第四个模块进行推理，得到预测的标签。图4展示了软件实现的高层组成模块，与其相互作用一起。应当指出，测试数据可能是从文件或相机中等载入的。

We utilize the OpenCV[28] as core computing library in our project. OpenCV is an open-source cross-platform library of programming functions that is mainly targeted for computer vision applications and includes efficient implementation of aforementioned operations. OpenCV is written in C++, and it provides the API (Application Program Interface) for both C++ and Java. We implement two versions of software for inference: one that uses OpenCV's Java API, which is more convenient for Android development, and another one that is developed in C++ using Android NDK (Native Development Kit), uses OpenCV's C++ API, and is expected to have a better performance.

在我们的工程中，我们利用OpenCV作为核心计算库。OpenCV是一个开源跨平台库，主要面向计算机视觉应用，包含前面提到的算子的高效实现。OpenCV是用C++实现的，有C++和Java的API结构。我们实现了两个版本的软件进行推理：一个使用了OpenCV的Java API，对Android开发非常方便，另一个是用C++开发的，使用Android NDK，使用OpenCV的C++ API，可以有更好的性能。

### 5.1. Experimental Setup

We run the inference application on various platforms of different generations in order to evaluate the applicability of the inference on embedded systems. Table I summarizes the specifications of test platforms.

我们在各种平台上运行推理应用，以评估推理在嵌入式系统的可应用性。表1总结了测试平台的指标。

The OpenCV Manager is installed on all target platforms in order to link OpenCV libraries dynamically and reduce memory usage. Additionally, hardware specific optimizations are applied by OpenCV Manager for an application’s supported platforms.

OpenCV Manager在各种目标平台上进行了安装，以将OpenCV库动态的链接，减少内存消耗。另外，OpenCV Manager对一个应用支持的平台，有硬件专用的优化。

In order to standardize the evaluation process on all platforms, the airplane mode is switched on to eliminate telecommunication overhead; all other running applications are closed to ensure they do not affect runtime; and the device is plugged in to avoid performance throttling applied by a platform’s governor. Though this is the standard setup, we will study the performance of inference process in situations where the device is running on its battery.

为标准化在所有平台的评估过程，调整到了飞行模式，以消除远程通信的开销；所有其他运行的应用都关掉了，以确保不会影响运行时间；设备插入充电状态，以避免平台主宰的性能遏制。虽然这是标准设置，我们会研究设备在电池的状态运行时，推理过程的性能。

### 5.2. MNIST

MNIST dataset [29] is a handwritten digits dataset which includes 28×28 greyscale images with 60,000 images for training and 10,000 images for testing. The original images in the MNIST dataset are resized using a bilinear transformation, and such transformation is used for both training and testing. Various neural network architectures are explored for each dataset and a few of them are presented in this paper.

MNIST数据集是手写数字数据集，包含28x28的灰度图，60000幅训练图像，10000幅测试图像。MNIST数据集中的原始图像使用双线性变换来改变大小，这种变换在训练和测试时都进行使用。各种神经网络架构都在每个数据集上进行了探索，其中一些试验在本文中给出。

For the MNIST dataset, two different neural network architectures are evaluated. In the first architecture (Arch. 1), the input layer consists of 256 neurons that represent the resized MNIST images. The next two layers comprise of 128 neurons each and are based on block-circulant matrix based FC layers. Finally, the last layer is a softmax layer that consists of 10 neurons representing the ten possible predictions for the digits. The second architecture (Arch. 2) has 121 neurons in the input layer, 64 neurons in the two hidden layers, and similar to Arch. 1, a softmax layer as the output layer. Table II summarizes the runtime of each round of inference process using these architectures and on various mobile platforms.

对于MNIST数据集，评估了两种不同的神经网络架构。在第一个架构中(Arch.1)，输入层包含256个神经元，表示变化了大小的MNIST图像，下面两个层每个包含128个神经元，是基于分块循环矩阵的FC层，最后一层是softmax层，包含10个神经元，表示10个可能的数字预测。第二个架构(Arch.2)在输入层包含121个神经元，在两个隐含层中包含64个神经元，与Arch.1类似，输出层为softmax层。表2总结了每轮推理过程的运行时间总结，使用了这两种架构，在各种移动平台上进行了实现。

Based on the results summarized in Table II, the C++ implementation is about 60-65% faster than the Java implementation. One of the reasons for this superior performance is related to memory limitations and management policy in Android. While applications written in C++ have an unlimited heap size, Java applications are restricted to platform-specific heap sizes. As a result, a constraint is imposed on the amount of data that an application written in Java can deal with at each instance of time.

在表II总结的结果中，C++实现比Java实现要快了60%-65%。一个原因是Android中的内存限制和管理策略。用C++写的应用，其heap大小没有限制，Java应用则限制在平台专有的heap大小中。结果是，Java写的应用，每次可以处理的数据量就有了限制。

Another potential reason that may explain the considerable performance difference between the two implementations is the overhead due to switching from Java to C++ and vice versa. Because the OpenCV library is written in C++, it needs to covert data from C++ data types to Java data types whenever the Java API is used. We believe that these conversions do not affect the runtime significantly, but can cause certain difference in performance across the two implementations.

另一个可能的原因是，因为从Java切换到C++以及反过来切换。因为OpenCV是用C++写的，在使用Java API时，需要从C++数据类型转换到Java数据类型。我们相信这种转换并没有明显影响运行时间，但会导致性能上两种实现的特定差异。

Considering different architectures mentioned in Table II, one can observe that going from the smaller network to a bigger network increases the accuracy by about 2% while it increases the memory required for storing parameters by a factor of about two and increases the runtime of Java and C++ implementations by about 2% and 9%, respectively. It should be noted that when the device is running on its battery, the runtime will increase by about 14% in the Java implementation, but remains unchanged in the C++ implementation.

考虑表2中提到的不同的架构，我们可以观察到，更小的网络到更大的网络，准确率增加了2%，而存储参数所需要的内存增加了2倍，Java和C++实现的运行时间分别增加了2%和9%。应当指出，当设备是在电池上运行时，Java实现的运行时间增加了14%，但C++实现则没有变化。

### 5.3. CIFAR-10

The CIFAR-10 [30] dataset contains 32×32 color images from 10 classes, where there are 50,000 training images and 10,000 testing images. The structure of deep neural network can be denoted as 128x3x32x32-64Conv3-64Conv3-128Conv3-128Conv3-512F-1024F-1024F-10F (Arch. 3). Here 128x3x32x32 represents that (i) the batch size is 128; (ii) the number of input channel is 3, (iii) and the feature size of input data is 32x32. In addition, 128Conv3 indicates that 128 3x3 convolutional filters are used in the convolutional layer. In addition, 512F or 10F means that the number of neurons in the FC layer is 512 or 10, respectively. In addition, both the original and compressed models are trained with learning rate 0.001 and momentum 0.9. In this network architecture, the first two convolutional layers are traditional convolutional layers (no block circulant, which is treated as preprocessing similar to the IBM TrueNorth paper [31]). Based on the results summarized in Table III, the C++ implementation is about 130% faster than the Java implementation.

CIFAR-10数据集包含32×32彩色图像，10个类别，有50000训练图像，10000测试图像。深度神经网络的结构可以表示为128x3x32x32-64Conv3-64Conv3-128Conv3-128Conv3-512F-1024F-1024F-10F (Arch. 3)。这里128x3x32x32表示，(i)batch大小为128，(ii)输入通道数量为3，(iii)输入数据的特征大小为32x32。另外，128Conv3表示，卷积层中使用了128个3x3的卷积滤波器。另外，512F或10F意思是，FC层中的神经元数量分别为512或10。另外，原始和压缩的模型训练时的学习速率为0.001，动量为0.9。在这个网络架构中，前两个卷积层是传统的卷积层（没有分块循环，当作是预处理）。结果如表3所示，可以看出，C++实现要比Java实现快了130%。

### 5.4. Comparison Results on Performance and Accuracy

In this section, we provide comprehensive comparison results on MNIST, CIFAR-10, and IBM TrueNorth [31], [32]. Our test platform consists of one or two qual-core ARM, while the IBM TrueNorth includes 4,096 ASIC cores, which is around 500-1000 times more than our testing platform. In Fig. 5, compared with IBM TrueNorth results on MNIST [32], our model performs 10× faster than IBM TrueNorth with a little accuracy reduction on the best device result. The accuracy for IBM TrueNorth is 95% and the runtime is 1000µs per image on MNIST. Compared with IBM TrueNorth results on CIFAR-10 [31], with 500-1000 times less cores, our model performs 10× slower than IBM TrueNorth. The accuracy for IBM TrueNorth is 83.41% and the runtime is 800µs per image. We can see that the later work [31] in 2016 on CIFAR-10 is optimized more efficiently compared with the former work [32] in 2015. Although our mobile phone based framework achieves lower performance compared with IBM TrueNorth on CIFAR-10, it is still reasonably good result considering the dramatic difference in computational resources. These results have demonstrated the effectiveness of the proposed framework.

本节中，我们给出了在MNIST，CIFAR-10和IBM TrueNorth上的综合比较。我们的测试平台由1或2个四核ARM组成，而IBM TrueNorth包含4096个ASIC核，比我们的测试平台要多500-1000倍。在图5中，与IBM TrueNorth在MNIST上的结果相比较，我们的模型比IBM TrueNorth快了10倍，与在最好设备上的结果相比，准确率有略微的下降。IBM TrueNorth在MNIST上的准确率是95%，运行时间是每幅图像1000µs。与IBM TrueNorth在CIFAR-10上的结果相比，我们的模型所用的核数少了500-1000倍，比IBM TrueNorth慢了10倍。IBM TrueNorth在准确率是83.41%，运行时间是每幅图像800µs。我们可以看到[31]在2016年在CIFAR-10上的工作，与[32]在2015年的工作相比，优化了很多。虽然我们基于移动手机的框架，与IBM TrueNorth在CIFAR-10的结果相比，性能差了一些，考虑到其在计算资源上的差异，性能还是非常好的。这些结果证明了提出的框架的有效性。

## 6. Conclusions

This paper presented a design optimization framework for Fast Fourier Transform-based deep neural network inference on embedded system. The proposed approach results in significant reduction in storage requirement for model parameters and improves runtime without affecting accuracy significantly. Our implementation on ARM-based embedded systems achieves runtime improvement on image classification tasks compared to IBM TrueNorth.

本文给出了基于FFT的DNN在嵌入式系统中的推理的设计优化框架。提出的方法在存储模型参数所需的内存上有显著下降，在不显著影响准确率的情况下，改进了运行时间。我们的实现是基于ARM的嵌入式系统，与IBM TrueNorth相比，在图像分类任务上，获得了运行时间的改进。