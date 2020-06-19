# Non-local Neural Networks

Xiaolong Wang et. al. Carnegie Mellon University Facebook AI Research

## 0. Abstract

Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. In this paper, we present non-local operations as a generic family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method [4] in computer vision, our non-local operation computes the response at a position as a weighted sum of the features at all positions. This building block can be plugged into many computer vision architectures. On the task of video classification, even without any bells and whistles, our non-local models can compete or outperform current competition winners on both Kinetics and Charades datasets. In static image recognition, our non-local models improve object detection/segmentation and pose estimation on the COCO suite of tasks. Code is available at https://github.com/ facebookresearch/video-nonlocal-net.

卷积操作和循环操作都是基础模块，每次处理一个局部的邻域。本文中，我们提出非局部的运算，是捕获长程依赖关系的一族基础模块。受到计算机视觉中经典非局部方法的启发，我们的非局部算子，计算所有位置的特征的加权和为某一个位置的响应。这种模块可以插入到很多计算机视觉架构中。在视觉分类的任务中，即使不用各种技巧，我们的非局部模型在Kinetics和Charades数据集上超过了目前的参赛的获胜者。在静止图像识别中，我们的非局部模型在COCO数据集上改进了目标检测/分割和姿态估计的结果。

## 1. Introduction

Capturing long-range dependencies is of central importance in deep neural networks. For sequential data (e.g., in speech, language), recurrent operations [38, 23] are the dominant solution to long-range dependency modeling. For image data, long-distance dependencies are modeled by the large receptive fields formed by deep stacks of convolutional operations [14, 30].

捕获长程依赖关系是DNN中非常重要。对于序列数据（如，语音，语言），循环运算是长程依赖关系建模的主要解决方案。对于图像数据，长程依赖关系是由大的感受野进行建模的，这是由卷积运算的深度堆叠形成的。

Convolutional and recurrent operations both process a local neighborhood, either in space or time; thus long-range dependencies can only be captured when these operations are applied repeatedly, propagating signals progressively through the data. Repeating local operations has several limitations. First, it is computationally inefficient. Second, it causes optimization difficulties that need to be carefully addressed [23, 21]. Finally, these challenges make multihop dependency modeling, e.g., when messages need to be delivered back and forth between distant positions, difficult.

卷积和循环运算处理的都是一个局部邻域，要么是空间上的，要么是时间上的；因此长程依赖关系只有在这些运算重复进行时，才能被捕获到，将信号逐渐的传播到数据中。重复局部运算有几个限制。第一，计算上效率不高。第二，导致优化困难，需要很小心的处理。最后，这些挑战使得多跳依赖关系建模很困难，如，信息需要在很远的位置上反复传递。

In this paper, we present non-local operations as an efficient, simple, and generic component for capturing long-range dependencies with deep neural networks. Our proposed non-local operation is a generalization of the classical non-local mean operation [4] in computer vision. Intuitively, a non-local operation computes the response at a position as a weighted sum of the features at all positions in the input feature maps (Figure 1). The set of positions can be in space, time, or spacetime, implying that our operations are applicable for image, sequence, and video problems.

本文中，我们提出了non-local运算，这是一个高效的、简单的、通用的部件，可以在深度神经网络中捕获长程关系。我们提出的non-local运算是经典的计算机视觉中的non-local平均运算的泛化。从直觉上来说，一个non-local运算计算的将一个位置上的响应，计算为输入特征图所有位置上特征的加权和（图1）。位置集合可以是空间，时间，或空间时间，说明我们的运算对于图像、序列和视频问题都是适用的。

There are several advantages of using non-local operations: (a) In contrast to the progressive behavior of recurrent and convolutional operations, non-local operations capture long-range dependencies directly by computing interactions between any two positions, regardless of their positional distance; (b) As we show in experiments, non-local operations are efficient and achieve their best results even with only a few layers (e.g., 5); (c) Finally, our non-local operations maintain the variable input sizes and can be easily combined with other operations (e.g., convolutions as we will use).

适用non-local运算有几个好处：(a)与循环运算和卷积运算的渐进行为相比，non-local运算直接捕获长程依赖关系，计算的是任意两个位置之间的互动，不管其位置距离的关系；(b)我们在试验中给出了，non-local运算是高效的，即使只使用少数几层，就可以得到最好的结果；(c)最后，我们的non-local运算保持变量的输入大小，可以很容易的与其他运算结合到一起（如，卷积运算）。

We showcase the effectiveness of non-local operations in the application of video classification. In videos, long-range interactions occur between distant pixels in space as well as time. A single non-local block, which is our basic unit, can directly capture these spacetime dependencies in a feedforward fashion. With a few non-local blocks, our architecures called non-local neural networks are more accurate for video classification than 2D and 3D convolutional networks [48] (including the inflated variant [7]). In addition, non-local neural networks are more computationally economical than their 3D convolutional counterparts. Comprehensive ablation studies are presented on the Kinetics [27] and Charades [44] datasets. Using RGB only and without any bells and whistles (e.g., optical flow, multi-scale testing), our method achieves results on par with or better than the latest competitions winners on both datasets.

我们在视频分类的应用中展示了non-local运算的有效性。在视频中，长程互动在距离较远的像素空间中以及时间中都有发生。单个non-local模块，即我们的基本单元，可以直接以前向的形式捕获这些时空依赖关系。只用几个non-local模块，我们的架构（称为non-local神经网络）比2D和3D卷积网络更精确的进行视频分类（包括膨胀的变体）。另外，non-local神经网络计算量也比3D卷积网络更少。在Kinetics和Charades数据集上进行了很多分离试验。只使用RGB图像，不用一些技巧（如，光流，多尺度测试），我们的方法在两个数据集上与最新的获胜者的结果类似或更好。

To demonstrate the generality of non-local operations, we further present object detection/segmentation and pose estimation experiments on the COCO dataset [33]. On top of the strong Mask R-CNN baseline [19], our non-local blocks can increase accuracy on all three tasks at a small extra computational cost. Together with the evidence on videos, these image experiments show that non-local operations are generally useful and can become a basic building block in designing deep neural networks.

为证明non-local运算的泛化性，我们进一步给出在COCO数据集上的目标检测/分割和姿态估计的试验。在很强的Mask R-CNN基准上[19]，我们的non-local模块可以在三个任务中提高准确率，计算量只有少量增加。与在视频上的结果一起，这些图像试验说明，non-local运算一般来说是很有用的，可以成为设计DNN的一种基本模块。

## 2. Related Work

**Non-local image processing**. Non-local means [4] is a classical filtering algorithm that computes a weighted mean of all pixels in an image. It allows distant pixels to contribute to the filtered response at a location based on patch appearance similarity. This non-local filtering idea was later developed into BM3D (block-matching 3D) [10], which performs filtering on a group of similar, but non-local, patches. BM3D is a solid image denoising baseline even compared with deep neural networks [5]. Block matching was used with neural networks for image denoising [6, 31]. Non-local matching is also the essence of successful texture synthesis [12], super-resolution [16], and inpainting [1] algorithms.

**Non-local图像处理**。Non-local平均[4]是一种经典滤波算法，计算图像中所有像素的加权平均。这使得很远的像素也可以基于图像块外表的相似性对一个位置的滤波响应有所贡献。这种non-local滤波的思想后来发展成BM3D（3D模块匹配）方法，在一组类似的但non-local的图像块上进行滤波。即使与DNN相比，BM3D是一种很强的图像去噪的基准。模块匹配可以与神经网络一起，用于图像去噪。Non-local匹配也是成功的纹理合成、超分辨率和修复算法的本质。

**Graphical models**. Long-range dependencies can be modeled by graphical models such as conditional random fields (CRF) [29, 28]. In the context of deep neural networks, a CRF can be exploited to post-process semantic segmentation predictions of a network [9]. The iterative mean-field inference of CRF can be turned into a recurrent network and trained [56, 42, 8, 18, 34]. In contrast, our method is a simpler feedforward block for computing non-local filtering. Unlike these methods that were developed for segmentation, our general-purpose component is applied for classification and detection. These methods and ours are also related to a more abstract model called graph neural networks [41].

**图形模型**。长程依赖关系可以用图像学模型建模，如条件随机场。在DNN的上下文中，CRF可以用于对语义分割的预测结果进行后处理。CRF的迭代平均场推理，可以转换成一种循环网络并进行训练。对比起来，我们的方法是一种更简单的前向模块，用以计算non-local滤波。与这些用于分割的方法不同，我们的通用目的的组件是用于分类和检测的。这些方法和我们的方法与一种更抽象的模型，称为图神经网络相关。

**Feedforward modeling for sequences**. Recently there emerged a trend of using feedforward (i.e., non-recurrent) networks for modeling sequences in speech and language [36, 54, 15]. In these methods, long-term dependencies are captured by the large receptive fields contributed by very deep 1-D convolutions. These feedforward models are amenable to parallelized implementations and can be more efficient than widely used recurrent models.

**对序列的前向建模**。最近在语音和语言中，出现了一种使用前向网络（即，非循环的）对序列进行建模的趋势。在这些方法中，长程依赖关系是通过大型感受野捕获的，即很深的1D卷积。这些前向模型可以进行并行实现，比广泛使用的循环模型效率更高。

**Self-attention**. Our work is related to the recent self-attention [49] method for machine translation. A self-attention module computes the response at a position in a sequence (e.g., a sentence) by attending to all positions and taking their weighted average in an embedding space. As we will discuss in the next, self-attention can be viewed as a form of the non-local mean [4], and in this sense our work bridges self-attention for machine translation to the more general class of non-local filtering operations that are applicable to image and video problems in computer vision.

**自注意力**。我们的工作与最近的用于机器翻译的自注意力模型是相关的。一个自注意力模块，通过在一个嵌入空间注意所有位置并取其加权平均，计算一个序列中一个位置的响应（如，一个语句）。我们在下面会讨论，自注意力可以视为non-local平均的一种形式，在这个意义上，我们的工作将自注意力与更通用的non-local滤波运算连接到了一起，可以在计算机视觉中用于图像和视频问题。

**Interaction networks**. Interaction Networks (IN) [2, 52] were proposed recently for modeling physical systems. They operate on graphs of objects involved in pairwise interactions. Hoshen [24] presented the more efficient Vertex Attention IN (VAIN) in the context of multi-agent predictive modeling. Another variant, named Relation Networks [40], computes a function on the feature embeddings at all pairs of positions in its input. Our method also processes all pairs, as we will explain ($f(x_i, x_j)$ in Eq.(1)). While our non-local networks are connected to these approaches, our experiments indicate that the non-locality of the model, which is orthogonal to the ideas of attention/interaction/relation (e.g., a network can attend to a local region), is the key to their empirical success. Non-local modeling, a long-time crucial element of image processing (e.g., [12, 4]), has been largely overlooked in recent neural networks for computer vision.

**互动网络**。互动网络(IN)最近提出用于对物理系统进行建模。它们在目标的图上进行运算，涉及到成对的互动。Hoshen[24]提出更高效的Vertex注意力IN，用于多agent预测性建模。另一种变体，即相关网络[40]，在其输入的所有位置对的特征嵌入上计算一个函数。我们的方法也处理所有对，我们会在后面解释。我们的non-local网络与这些方法相关，而我们的试验则表明，模型的非局部性，与注意力/互动/相关的思想则是正交的（如，一个网络可以关注一个局部区域），是经验性成功的关键。Non-local建模，图像处理中的长期内的一个关键元素，在最近的计算机视觉中的神经网络中，基本被忽略了。

**Video classification architectures**. A natural solution to video classification is to combine the success of CNNs for images and RNNs for sequences [55, 11]. In contrast, feed-forward models are achieved by 3D convolutions (C3D) [26, 48] in spacetime, and the 3D filters can be formed by “inflating” [13, 7] pre-trained 2D filters. In addition to end-to-end modeling on raw video inputs, it has been found that optical flow [45] and trajectories [50, 51] can be helpful. Both flow and trajectories are off-the-shelf modules that may find long-range, non-local dependency. A systematic comparison of video architectures can be found in [7].

**视频分类架构**。视频分类的一个自然解决方案是，将用于图像的CNNs和用于序列的RNNs的成功结合起来。比较起来，前向模型是采用时空3D卷积(C3D)实现的，3D滤波器是通过对预训练的2D滤波器膨胀得到的。除了在原始视频输入的端到端的建模，光流和轨迹也很有用。光流和轨迹，都是立刻可用的模块，可以发现长程的，non-local依赖关系。视频架构的系统性比较可以参考[7]。

## 3. Non-local Neural Networks

We first give a general definition of non-local operations and then we provide several specific instantiations of it. 我们首先non-local运算的一般性定义，然后给出几种具体实现。

### 3.1. Formulation

Following the non-local mean operation [4], we define a generic non-local operation in deep neural networks as: 按照non-local平均的运算[4]，我们定义DNN中的一个通用non-local运算为：

$$y_i = \frac{1}{C(x)} \sum_{∀j} f(x_i,x_j) g(x_j)$$(1)

Here i is the index of an output position (in space, time, or spacetime) whose response is to be computed and j is the index that enumerates all possible positions. x is the input signal (image, sequence, video; often their features) and y is the output signal of the same size as x. A pairwise function f computes a scalar (representing relationship such as affinity) between i and all j. The unary function g computes a representation of the input signal at the position j. The response is normalized by a factor C(x).

这里i是输出位置的索引（在空间，时间或时空域中），要在这个位置计算响应；j是枚举了所有可能的位置的索引。x是输入信号（图像，序列，视频；通常是其特征），y是输出信号，与x大小一样。成对的函数f，计算的是i和所有j之间的一个标量（表示一种关系，如关系密切程度）。一元函数g计算的是输入信号在位置j上的表示。响应用因子C(x)进行归一化。

The non-local behavior in Eq.(1) is due to the fact that all positions (∀j) are considered in the operation. As a comparison, a convolutional operation sums up the weighted input in a local neighborhood (e.g.,i−1≤j≤i+1 in a 1D case with kernel size 3), and a recurrent operation at time i is often based only on the current and the latest time steps (e.g.,j = i or i−1).

式(1)中的non-local行为是因为在这个运算中考虑了所有位置(∀j)。比较起来，卷积运算只在一个局部邻域对其进行加权求和（如，在1D情况下核大小为3时为i−1≤j≤i+1），在时间i时的循环运算通常只是基于目前的和最后时间的运算（如，j=i或i-1）。

The non-local operation is also different from a fully-connected (fc) layer. Eq.(1) computes responses based on relationships between different locations, whereas fc uses learned weights. In other words, the relationship between $x_j$ and $x_i$ is not a function of the input data in fc, unlike in non-local layers. Furthermore, our formulation in Eq.(1) supports inputs of variable sizes, and maintains the corresponding size in the output. On the contrary, an fc layer requires a fixed-size input/output and loses positional correspondence (e.g.,that from $x_i$ to $y_i$ at the position i).

Non-local运算与全连接层也是不同的。式(1)是基于不同位置之间的关系来计算响应的，而fc使用的是学习到的权重。换句话说，在fc中，$x_j$和$x_i$之间的关系不是输入数据的函数，这与non-local层不同。而且，我们在式(1)中的表述，支持输入的大小可变，而且保持与输入的对应大小一样。相反，fc层需要固定大小的输入输出，失去了位置上的对应性（如，在位置i上从$x_i$到$y_i$的对应性）。

A non-local operation is a flexible building block and can be easily used together with convolutional/recurrent layers. It can be added into the earlier part of deep neural networks, unlike fc layers that are often used in the end. This allows us to build a richer hierarchy that combines both non-local and local information.

一个non-local运算是一个灵活的模块，可以很容易的与卷积/循环层一起使用。可以加入DNN更早的部分，与fc层通常用在最后不太一样。这使我们可以构建更丰富的层次关系，将non-local和local信息结合到一起。

### 3.2. Instantiations 实例化

Next we describe several versions of f and g. Interestingly, we will show by experiments (Table 2a) that our non-local models are not sensitive to these choices, indicating that the generic non-local behavior is the main reason for the observed improvements.

下面我们描述f和g的几个版本。有趣的是，我们会通过试验表明（表2a），我们的non-local模型对这些选择并不敏感，说明通用non-local行为是观察到的改进的主要原因。

For simplicity, we only consider g in the form of a linear embedding: $g(x_j) = W_gx_j$, where $W_g$ is a weight matrix to be learned. This is implemented as, e.g., 1×1 convolution in space or 1×1×1 convolution in spacetime.

简单起见，我们只考虑g的线性嵌入形式：$g(x_j) = W_gx_j$，其中$W_g$是一个要学习的权重矩阵。这在空间中是1×1的卷积，在时空中是1×1×1的卷积。

Next we discuss choices for the pairwise function f. 下面我们讨论成对函数f的选择。

**Gaussian**. Following the non-local mean [4] and bilateral filters [47], a natural choice of f is the Gaussian function. In this paper we consider: 按照non-local平均和双边滤波器，f的一个自然选择是高斯函数。本文中我们考虑：

$$f(x_i, x_j) = e^{x_i^T x_j}$$(2)

Here $x_i^T x_j$ is dot-product similarity. Euclidean distance as used in [4, 47] is also applicable, but dot product is more implementation-friendly in modern deep learning platforms. The normalization factor is set as $C(x) = \sum_{∀j} f(x_i, x_j)$.

这里$x_i^T x_j$是点积的相似性。[4,47]中使用的欧式距离也是可用的，但在深度学习平台上点积实现上更友好。归一化因子设为$C(x) = \sum_{∀j} f(x_i, x_j)$。

**Embedded Gaussian**. A simple extension of the Gaussian function is to compute similarity in an embedding space. In this paper we consider: 高斯函数的一个简单拓展是，在一个嵌入空间中计算相似度，本文中我们考虑：

$$f(x_i, x_j) = e^{θ(x_i)^T φ(x_j)}$$(3)

Here $θ(x_i) = W_θ x_i$ and $φ(x_j) = W_φ x_j$ are two embeddings. As above, we set $C(x) = \sum_{∀j} f(x_i, x_j)$.

We note that the self-attention module [49] recently presented for machine translation is a special case of non-local operations in the embedded Gaussian version. This can be seen from the fact that for a given i, $\frac{1}{C(x)} f(x_i, x_j)$ becomes the softmax computation along the dimension j. So we have $y = softmax(x^T W_θ^T W_φ x)g(x)$, which is the self-attention form in [49]. As such, our work provides insight by relating this recent self-attention model to the classic computer vision method of non-local means [4], and extends the sequential self-attention network in [49] to a generic space/spacetime non-local network for image/video recognition in computer vision.

我们要说明的是，最近提出来用于机器翻译的自注意力模块[49]是嵌入高斯版non-local运算的的特殊情况。我们可以看到，对于给定的i，$\frac{1}{C(x)} f(x_i, x_j)$就是沿着维度j方向的softmax计算。所以我们有$y = softmax(x^T W_θ^T W_φ x)g(x)$，这就是[49]中的自注意力模型。这样，我们的工作将最近的自注意力模型与经典计算机视觉中的non-local均值方法关联了起来，将[49]中的序列自注意力网络拓展到了通用的空间/时空non-local网络，可以在计算机视觉中进行图像/视频识别。

Despite the relation to [49], we show that the attentional behavior (due to softmax) is not essential in the applications we study. To show this, we describe two alternative versions of non-local operations next.

尽管与[49]有关，我们表明其注意力的行为（由softmax）在我们研究的应用中并不关键。为展示这个结果，我们再给出两种non-local运算的替代版本。

**Dot product**. f can be defined as a dot-product similarity:

$$f(x_i, x_j) = θ(x_i)^T φ(x_j)$$(4)

Here we adopt the embedded version. In this case, we set the normalization factor as C(x) = N, where N is the number of positions in x, rather than the sum of f, because it simplifies gradient computation. A normalization like this is necessary because the input can have variable size.

这里我们采用嵌入的版本。在这种情况下，我们设归一化因子为C(x) = N，其中N是x中的位置数量，而不是f的求和，因为这简化了梯度计算。一种这样的归一化是很必要的，因为输入的大小是可变的。

The main difference between the dot product and embedded Gaussian versions is the presence of softmax, which plays the role of an activation function.

点积和高斯嵌入版本的主要差别是softmax的存在，扮演的是激活函数的角色。

**Concatenation**. Concatenation is used by the pairwise function in Relation Networks [40] for visual reasoning. We also evaluate a concatenation form of f: 在相关网络[40]中，拼接是用作成对函数的，进行视觉推理。我们可以采用下面的拼接形式作为f：

$$f(x_i,x_j) = ReLU(w_f^T [θ(x_i), φ(x_j)])$$(5)

Here [·, ·] denotes concatenation and $w_f$ is a weight vector that projects the concatenated vector to a scalar. As above, we set C(x) = N. In this case, we adopt ReLU [35] in f.

这里[·, ·]表示拼接，$w_f$是加权向量，将拼接的向量映射为一个标量。像上面一样，我们设C(x) = N。在这种情况中，我们在f中采用ReLU。

The above several variants demonstrate the flexibility of our generic non-local operation. We believe alternative versions are possible and may improve results. 上述几个变体表明了我们通用的non-local运算的灵活性。我们相信，其他的版本也是可能的，可能会改进结果。

### 3.3. Non-local Block

We wrap the non-local operation in Eq.(1) into a non-local block that can be incorporated into many existing architectures. We define a non-local block as: 我们将式(1)中的non-local运算包装到一个non-local模块中，可以与很多现有的架构一起使用。我们定义一个non-local模块为：

$$z_i = W_z y_i + x_i$$(6)

where $y_i$ is given in Eq.(1) and “$+x_i$” denotes a residual connection [21]. The residual connection allows us to insert a new non-local block into any pre-trained model, without breaking its initial behavior (e.g., if $W_z$ is initialized as zero). An example non-local block is illustrated in Figure 2. The pairwise computation in Eq.(2), (3), or (4) can be simply done by matrix multiplication as shown in Figure 2; the concatenation version in (5) is straightforward.

其中$y_i$是式(1)给出的，“$+x_i$”表示一个残差连接。残差连接使我们可以插入一个新的non-local模块到任何预训练模型中，而不需要破坏你初始的行为（如，$W_z$初始化为0）。Non-local模块的一个例子如图2所示。式2,3,4中的成对计算可以通过图2中的矩阵乘法进行；式5中的拼接是非常直接的。

The pairwise computation of a non-local block is lightweight when it is used in high-level, sub-sampled feature maps. For example, typical values in Figure 2 are T = 4, H = W = 14 or 7. The pairwise computation as done by matrix multiplication is comparable to a typical convolutional layer in standard networks. We further adopt the following implementations that make it more efficient.

一个non-local模块的成对计算，当用在高层的、下采样的特征图时，计算量是很小的。比如，图2中的典型值为T = 4, H = W = 14 or 7。由矩阵相乘进行的成对计算与标准网络中典型的卷积层是可以类比的。我们进一步采用下列的实现，使其更加有效率。

**Implementation of Non-local Blocks**. We set the number of channels represented by $W_g, W_θ$, and $W_φ$ to be half of the number of channels in x. This follows the bottleneck design of [21] and reduces the computation of a block by about a half. The weight matrix $W_z$ in Eq.(6) computes a position-wise embedding on $y_i$, matching the number of channels to that of x. See Figure 2.

**Non-local模块的实现**。我们设由$W_g, W_θ$和$W_φ$表示的通道数为x中通道数的一半。这与[21]的瓶颈设计一样，将一个模块的计算降低了一半。式6中的权重矩阵$W_z$在$y_i$上计算了一个逐个位置的嵌入，将通道数量与x的相匹配。见图2。

A subsampling trick can be used to further reduce computation. We modify Eq.(1) as: $yi = \frac {1}{C(\hat x) \sum_{∀j} f(x_i, \hat x_j)g(\hat x_j)$, where $\hat x$ is a subsampled version of x (e.g., by pooling). We perform this in the spatial domain, which can reduce the amount of pairwise computation by 1/4. This trick does not alter the non-local behavior, but only makes the computation sparser. This can be done by adding a max pooling layer after φ and g in Figure 2.

下采样的一个技巧可以用于进一步降低计算。我们将式1修改为：$yi = \frac {1}{C(\hat x) \sum_{∀j} f(x_i, \hat x_j)g(\hat x_j)$，其中$\hat x$是x的下采样（如，池化）。我们在空域进行，这可以将成对计算的计算量降低到1/4。这种技巧并没有改变non-local行为，但只会使得计算更稀疏。在图2中就是在φ和g的后面增加一个最大池化层。

We use these efficient modifications for all non-local blocks studied in this paper. 我们在本文中将这种高效的修改加入到所有的non-local模块。

## 4. Video Classification Models

To understand the behavior of non-local networks, we conduct comprehensive ablation experiments on video classification tasks. First we describe our baseline network architectures for this task, and then extend them into 3D ConvNets [48, 7] and our proposed non-local nets.

为理解non-local网络的行为，我们在视频分类任务中进行了详尽的分离试验。首先我们对这个任务中描述了我们的基准网络架构，然后将其拓展到3D卷积网络和我们提出的non-local网络。

**2D ConvNet baseline (C2D)**. To isolate the temporal effects of our non-local nets vs. 3D ConvNets, we construct a simple 2D baseline architecture in which the temporal dimension is trivially addressed (i.e., only by pooling).

**2D卷积网络基准(C2D)**。为将我们的non-local网络和3D卷积网络的时域效果孤立开来，我们构建了一个简单的2D基准架构，其中时域维度的处理是非常简单的（即，只通过池化）。

Table 1 shows our C2D baseline under a ResNet-50 backbone. The input video clip has 32 frames each with 224×224 pixels. All convolutions in Table 1 are in essence 2D kernels that process the input frame-by-frame (implemented as 1×k×k kernels). This model can be directly initialized from the ResNet weights pre-trained on ImageNet. A ResNet-101 counterpart is built in the same way.

表1展示了使用ResNet-50骨干的C2D基准。输入的视频片段有32帧，每帧224×224像素。表1中的所有卷积本质上是2D核，逐帧处理输入（实现为1×k×k的核）。这个模型可以直接用在ImageNet上预训练的权重进行初始化。用同样的方法构建了一个ResNet-101的对应模型。

The only operation involving the temporal domain are the pooling layers. In other words, this baseline simply aggregates temporal information. 时域中涉及到的计算是只有池化层。换句话说，这种基准只是聚积了时域信息。

**Inflated 3D ConvNet (I3D)**. As done in [13, 7], one can turn the C2D model in Table 1 into a 3D convolutional counterpart by “inflating” the kernels. For example, a 2D k×k kernel can be inflated as a 3D t×k×k kernel that spans t frames. This kernel can be initialized from 2D models (pre-trained on ImageNet): each of the t planes in the t×k×k kernel is initialized by the pre-trained k×k weights, rescaled by 1/t. If a video consists of a single static frame repeated in time, this initialization produces the same results as the 2D pre-trained model run on a static frame.

**膨胀的3D卷积网络**。如[13,7]中的工作，可以将表1中的C2D模型转化为对应的3D卷积模型，即将卷积核膨胀。比如，2D的k×k核重复t帧，就可以膨胀成一个3D的t×k×k的核。这个核可以从2D模型中进行初始化（在ImageNet上预训练的模型）：在这个t×k×k的核中的t个平面中，通过预训练的k×k权重进行初始化，乘以1/t以归一化。如果视频是由一帧静态影像重复得到的，这种初始化得到的结果，与2D预训练模型在一个静态帧上的结果一样。

We study two cases of inflations: we either inflate the 3×3 kernel in a residual block to 3×3×3 (similar to [7]), or the first 1×1 kernel in a residual block to 3×1×1 (similar to [13]). We denote these as $I3D_{3×3×3}$ and $I3D_{3×1×1}$. As 3D convolutions are computationally intensive, we only inflate one kernel for every 2 residual blocks; inflating more layers shows diminishing return. We inflate $conv_1$ to 5×7×7.

我们研究两种膨胀：我们将残差模块中的3×3膨胀成3×3×3，或将1×1核膨胀成3×1×1。我们将这两种表示为$I3D_{3×3×3}$和$I3D_{3×1×1}$。由于3D卷积计算量很大，我们对每两个残差单元只膨胀一个核；膨胀更多的层会带来递减的收益。我们将$conv_1$膨胀为5×7×7。

The authors of [7] have shown that I3D models are more accurate than their CNN+LSTM counterparts. [7]的作者表明，I3D模型比其CNN+LSTM算法要更加准确。

**Non-local network**. We insert non-local blocks into C2D or I3D to turn them into non-local nets. We investigate adding 1, 5, or 10 non-local blocks; the implementation details are described in the next section in context. 我们将non-local模块插入到C2D或I3D中，将其变成non-local网络。我们研究了加入1，5，或10个non-local模块；实现细节在下一节介绍。

### 4.1. Implementation Details

**Training**. Our models are pre-trained on ImageNet [39]. Unless specified, we fine-tune our models using 32-frame input clips. These clips are formed by randomly cropping out 64 consecutive frames from the original full-length video and then dropping every other frame. The spatial size is 224×224 pixels, randomly cropped from a scaled video whose shorter side is randomly sampled in [256, 320] pixels, following [46]. We train on an 8-GPU machine and each GPU has 8 clips in a mini-batch (so in total with a mini-batch size of 64 clips). We train our models for 400k iterations in total, starting with a learning rate of 0.01 and reducing it by a factor of 10 at every 150k iterations (see also Figure 4). We use a momentum of 0.9 and a weight decay of 0.0001. We adopt dropout [22] after the global pooling layer, with a dropout ratio of 0.5. We fine-tune our models with BatchNorm (BN) [25] enabled when it is applied. This is in contrast to common practice [21] of fine-tuning ResNets, where BN was frozen. We have found that enabling BN in our application reduces overfitting.

**训练**。我们的模型在ImageNet[39]上进行预训练。除非另外指定，我们使用32帧的输入片段来精调我们的模型。这些片段是从原始长度的视频中截取出64连续帧，然后隔一帧丢掉一帧。空间大小为224×224像素，是从缩放的视频中随机剪切出来的，其短边是[256, 320]像素的随机值，这是与[46]的工作是相同的。我们在一个8-GPU机器上训练的，每个GPU的一个mini-batch有8个片段（所以总共mini-batch大小为64个视频片段）。我们训练模型总共400k次迭代，学习速率从0.01开始，每150k次迭代除以10（见图4）。我们使用的动量为0.9，权重衰减为0.0001。我们在全局池化层后采用了dropout，dropout率为0.5。我们使用BN精调模型，。这与[21]中精调ResNets形成对比，其中BN进行了冻结。我们发现，在我们的应用中使用BN会降低过拟合。

We adopt the method in [20] to initialize the weight layers introduced in the non-local blocks. We add a BN layer right after the last 1×1×1 layer that represents $W_z$; we do not add BN to other layers in a non-local block. The scale parameter of this BN layer is initialized as zero, following [17]. This ensures that the initial state of the entire non-local block is an identity mapping, so it can be inserted into any pre-trained networks while maintaining its initial behavior.

我们采用[20]中的方法来初始化non-local模块的权重层。我们在最后的1×1×1层后加入一个BN层，表示$W_z$；我们在non-local模块中没有对其他层增加BN。这个BN层的缩放参数初始化为0，这是按照[17]的方法。这确保了整个non-local模块的初始状态是一个恒等映射，所以可以插入到任何预训练网络中，同时保持其初始行为。

**Inference**. Following [46] we perform spatially fully-convolutional inference on videos whose shorter side is rescaled to 256. For the temporal domain, in our practice we sample 10 clips evenly from a full-length video and compute the softmax scores on them individually. The final prediction is the averaged softmax scores of all clips.

**推理**。按照[46]，我们在视频上进行空间的全卷积推理，其短边缩放到256。对于时域，在我们的实践中，我们从一个完整长度的视频中采样10个片段，分别计算其softmax值。最后的预测是所有片段的softmax分数的平均。

## 5. Experiments on Video Classification

We perform comprehensive studies on the challenging Kinetics dataset [27]. We also report results on the Charades dataset [44] to show the generality of our models. 我们在Kinetics数据集上进行了综合的研究。我们也在Charades数据集上给出了结果，表明我们模型的泛化性。

## 5.1. Experiments on Kinetics

Kinetics [27] contains ∼246k training videos and 20k validation videos. It is a classification task involving 400 human action categories. We train all models on the training set and test on the validation set.

Kinetics数据集包含～246k训练视频，20k验证视频。这是包含了400类人类行为的分类任务。我们在训练集上训练所有模型，在验证集上进行测试。

Figure 4 shows the curves of the training procedure of a ResNet-50 C2D baseline vs. a non-local C2D with 5 blocks (more details in the following). Our non-local C2D model is consistently better than the C2D baseline throughout the training procedure, in both training and validation error.

图4给出了两个模型的训练曲线，即一个ResNet-50 C2D基准，和一个包含5个模块的non-local C2D。我们的non-local C2D模型在整个训练过程中一直优于C2D基准。包括训练误差和验证误差。

Figure 1 and Figure 3 visualize several examples of the behavior of a non-local block computed by our models. Our network can learn to find meaningful relational clues regardless of the distance in space and time.

图1和3对一个non-local模块的行为进行了可视化。我们的网络可以学习到有意义的线索，不管在时间和空间范围内的距离有多少。

Table 2 shows the ablation results, analyzed as follows: 表2给出了分离试验结果，分析如下：

**Instantiations**. Table 2a compares different types of a single non-local block added to the C2D baseline (right before the last residual block of res4). Even adding one non-local block can lead to ∼1% improvement over the baseline. 表2a比较了不同类似的non-local模块加入到C2D基准后的结果。即使只增加一个non-local模块，也会给基准带来约1%的改进。

Interestingly, the embedded Gaussian, dot-product, and concatenation versions perform similarly, up to some random variations (72.7 to 72.9). As discussed in Sec. 3.2, the non-local operations with Gaussian kernels become similar to the self-attention module [49]. However, our experiments show that the attentional (softmax) behavior of this module is not the key to the improvement in our applications; instead, it is more likely that the non-local behavior is important, and it is insensitive to the instantiations.

有趣的是，嵌入高斯，点积和拼接版的性能类似，有一定的随机差异。如3.2节所讨论，采用高斯核的non-local运算与自注意力模块类似。但是，我们的试验表明，这个模块的注意力行为不是在我们应用关键的改进；而我们的non-local行为则是更重要的，对于具体的实现并不敏感。

In the rest of this paper, we use the embedded Gaussian version by default. This version is easier to visualize as its softmax scores are in the range of [0, 1]. 本文的剩余部分，我们默认使用嵌入高斯版。这个版本更容易进行可视化，因为其softmax分数是在[0, 1]的范围内。

**Which stage to add non-local blocks?** Table 2b compares a single non-local block added to different stages of ResNet. The block is added to right before the last residual block of a stage. The improvement of a non-local block on res2, res3, or res4 is similar, and on res5 is slightly smaller. One possible explanation is that res5 has a small spatial size (7×7) and it is insufficient to provide precise spatial information. More evidence of a non-local block exploiting spatial information will be investigated in Table 2d. 表2b比较了将一个non-local模块加入到ResNet的不同阶段的性能比较。这个模块加入到一个阶段最后的残差模块之前。在res2, res3, res4之前的non-local模块的改进是类似的，在res5上则略小一些。一种可能的解释是，res5的空间大小比较小(7×7)，不足以给出精确的空间信息。更多的证据在表2d中也有。

**Going deeper with non-local blocks**. Table 2c shows the results of more non-local blocks. We add 1 block (to res4), 5 blocks (3 to res4 and 2 to res3, to every other residual block), and 10 blocks (to every residual block in res3 and res4) in ResNet-50; in ResNet-101 we add them to the corresponding residual blocks. Table 2c shows that more non-local blocks in general lead to better results. We argue that multiple non-local blocks can perform long-range multi-hop communication. Messages can be delivered back and forth between distant positions in spacetime, which is hard to do via local models.

表2c给出了更多non-local模块的结果。我们在ResNet-50中加入了1个模块、5个模块和10个模块；在ResNet-101中也加入了对应的残差模块。表2c表明，更多的non-local模块一般会带来更好的结果。我们认为non-local模块可以进行长程多跳通信。信息可以在时空域中较远的位置来回传递，这在loca模型中很难做到。

It is noteworthy that the improvement of non-local blocks is not just because they add depth to the baseline model. To see this, we note that in Table 2c the non-local 5-block ResNet-50 model has 73.8 accuracy, higher than the deeper ResNet-101 baseline’s 73.1. However, the 5-block ResNet-50 has only ∼70% parameters and ∼80% FLOPs of the ResNet-101 baseline, and is also shallower. This comparison shows that the improvement due to non-local blocks is complementary to going deeper in standard ways.

值得说明的是，non-local模块的改进，并不只是因为，他们对基准模型增加了深度。为看到这一点，我们看一下表2c，5个non-local模块的ResNet模型有73.8%准确率，比更深的ResNet-101基准的73.1要高。但是，5个模块的ResNet-50与ResNet-101基准比较起来，只有70%的参数和80%的FLOPs，也更浅。这种比较表明，由于non-local模块带来的改进，与标准方式变得更深比起来，是一种互补。

We have also tried to add standard residual blocks, instead of non-local blocks, to the baseline models. The accuracy is not increased. This again shows that the improvement of non-local blocks is not just because they add depth. 我们也尝试了为基准模型增加标准的残差模块，而不是non-local模块。准确率并没有增加。这再一次表明，non-local模块的改进并不只是因为增加了深度。

**Non-local in spacetime**. Our method can naturally handle spacetime signals. This is a nice property: related objects in a video can present at distant space and long-term time interval, and their dependency can be captured by our model. 我们的方法可以很自然的处理时空信号。这是一个很好的性质：视频中的相关目标可以在较远的空间和很长的时间间隔内出现，其依赖关系可以通过我们的模型捕获到。

In Table 2d we study the effect of non-local blocks applied along space, time, or spacetime. For example, in the space-only version, the non-local dependency only happens within the same frame: i.e., in Eq.(1) it only sums over the index j in the same frame of the index i. The time-only version can be set up similarly. Table 2d shows that both the space-only and time-only versions improve over the C2D baseline, but are inferior to the spacetime version.

在表2d中，我们研究了non-local模块在空间、时间和时空中应用的效果。比如，只在空间中的版本，non-local依赖关系只在相同的帧之间发生：即，在式1中，只对索引j进行求和。只在时间中的版本，可以类似的设置。表2d表明，space-only和time-only版都会改进C2D基准，但比spacetime版本要差一些。

**Non-local net vs. 3DConvNet**. Table2e compares our non-local C2D version with the inflated 3D ConvNets. Non-local operations and 3D convolutions can be seen as two ways of extending C2D to the temporal dimensions. 表2e比较了non-local C2D模型和膨胀3D卷积网络。Non-local运算和3D运算可以视为两种将C2D拓展到时域维度的方法。

Table 2e also compares the number of parameters and FLOPs, relative to the baseline. Our non-local C2D model is more accurate than the I3D counterpart (e.g., 75.1 vs. 74.4), while having a smaller number of FLOPs (1.2× vs. 1.5×). This comparison shows that our method can be more effective than 3D convolutions when used alone.

表2e也与基准比较了参数数量和FLOPs数量。我们的non-local模型比I3D模型更准确，而FLOPs则更少。这种比较说明，我们的方法在单独使用时，比3D卷积更加有效。

**Non-local 3D ConvNet**. Despite the above comparison, non-local operations and 3D convolutions can model different aspects of the problem: 3D convolutions can capture local dependency. Table 2f shows the results of inserting 5 non-local blocks into the I3D3×1×1 models. These non-local I3D (NL I3D) models improve over their I3D counterparts (+1.6 point accuracy), showing that non-local operations and 3D convolutions are complementary. 尽管有上面的比较，non-local运算和3D卷积可以对问题的不同方面进行建模：3D卷积可以捕获local依赖关系。表2f给出了在I3D3×1×1模型中插入5个non-local模块的结果。这些non-local I3D模型在I3D模型的基础上有所改进，表明non-local运算和3D卷积是互补的。

**Longer sequences**. Finally we investigate the generality of our models on longer input videos. We use input clips consisting of 128 consecutive frames without subsampling. The sequences throughout all layers in the networks are thus 4× longer compared to the 32-frame counterparts. To fit this model into memory, we reduce the mini-batch size to 2 clips per GPU. As a result of using small mini-batches, we freeze all BN layers in this case. We initialize this model from the corresponding models trained with 32-frame inputs. We fine-tune on 128-frame inputs using the same number of iterations as the 32-frame case (though the mini-batch size is now smaller), starting with a learning rate of 0.0025. Other implementation details are the same as before.

最后我们研究了我们的模型在更长的输入视频中的泛化性。我们使用包含128个连续帧的输入片段，没有进行下采样。为将这个模型装到内存中，我们将mini-batch大小降低了到每GPU 2片段。使用小mini-batch的结果，我们在这种情况下冻结了所有BN层。我们从32帧输入的对应的模型中初始化这个模型。我们在128帧输入中精调，使用了相同次数的迭代，学习速率初始为0.0025。其他的实现细节与之前相同。

Table 2g shows the results of 128-frame clips. Comparing with the 32-frame counterparts in Table 2f, all models have better results on longer inputs. We also find that our NL I3D can maintain its gain over the I3D counterparts, showing that our models work well on longer sequences.

表2g给出了128帧片段的结果。与在图2f中的32帧的比较，所有模型在更长的输入下有了更好的结果。我们还发现，NL I3D模型保持了对I3D模型的改进，表明我们的模型在更长的序列中效果也很好。

**Comparisons with state-of-the-art results**. Table3 shows the results from the I3D authors [7] and from the Kinetics 2017 competition winner [3]. We note that these are comparisons of systems which can differ in many aspects. Nevertheless, our method surpasses all the existing RGB or RGB + flow based methods by a good margin. Without using optical flow and without any bells and whistles, our method is on par with the heavily engineered results of the 2017 competition winner.

表3给出了I3D作者的结果和Kinetics 2017比赛获胜者的结果。我们注意到，这些比较在很多方面会很不同。尽管如此，我们的方法超过了所有的RGB或RGB+光流的方法很多。我们的方法没有使用光流，也没有使用任何技巧，与2017比赛获胜的结果是类似的。

### 5.2. Experiments on Charades

Charades [44] is a video dataset with ∼8k training, ∼1.8k validation, and ∼2k testing videos. It is a multi-label classification task with 157 action categories. We use a per-category sigmoid output to handle the multi-label property. Charades是一个视频数据集，有约8k训练视频，约1.8k验证视频，约2k测试视频。这是一个多标签分类任务，有157个行为类别。我们对每个类别使用了一个sigmoid输出，以处理多标签的性质。

We initialize our models pre-trained on Kinetics (128-frame). The mini-batch size is set to 1 clip per GPU. We train our models for 200k iterations, starting from a learning rate of 0.00125 and reducing it by 10 every 75k iterations. We use a jittering strategy similar to that in Kinetics to determine the location of the 224×224 cropping window, but we rescale the video such that this cropping window outputs 288×288 pixels, on which we fine-tune our network. We test on a single scale of 320 pixels.

我们用在Kinetics 128帧上预训练的结果初始化模型。Mini-batch大小设为每GPU一个片段。我们的模型训练了200k次迭代，学习速率初始为0.00125，每75k迭代就除以10。我们使用了一种抖动的策略，与在Kinetics中的类似，确定224×224剪切窗口的位置，但我们对视频大小进行了缩放，使这个剪切窗口输出288×288像素，在此之上，我们精调了我们的网络。我们在单尺度320像素上进行测试。

Table 4 shows the comparisons with the previous results on Charades. The result of [7] is the 2017 competition winner in Charades, which was also fine-tuned from models pre-trained in Kinetics. Our I3D baseline is higher than previous results. As a controlled comparison, our non-local net improves over our I3D baseline by 2.3% on the test set.

表4给出了在Charades与之前结果的对比。[7]的结果是在Charades的2017比赛的获胜者，也是在Kinetics上预训练的模型进行精调得到的。我们的I3D基准比之前的结果要更高。作为一个受控的比较，我们的non-local网络在I3D基准上有2.3%的改进。

## 6. Extension: Experiments on COCO

We also investigate our models on static image recognition. We experiment on the Mask R-CNN baseline [19] for COCO [33] object detection/segmentation and human pose estimation (keypoint detection). The models are trained on COCO train2017 (i.e., trainval35k in 2014) and tested on val2017 (i.e., minival in 2014).

我们还研究了我们的模型在静态图像识别中的效果。我们用Mask R-CNN基准在COCO上进行目标检测/分割，和人体姿态估计（关键点检测）的试验。模型在COCO train2017上进行训练，在val2017上进行测试。

**Object detection and instance segmentation**. We modify the Mask R-CNN backbone by adding one non-local block (right before the last residual block of res4). All models are fine-tuned from ImageNet pre-training. We evaluate on a standard baseline of ResNet-50/101 and a high baseline of ResNeXt-152 (X152) [53]. Unlike the original paper [19] that adopted stage-wise training regarding RPN, we use an improved implementation with end-to-end joint training similar to [37], which leads to higher baselines than [19].

**目标检测和实例分割**。我们将Mask R-CNN骨干进行了修改，增加了一个non-local模块。模型都是从ImageNet预训练上进行精调的。我们有一个标准的基准ResNet-50/101，还有一个高的基准ResNeXt-152。原始文章[19]采用了分阶段的训练，我们采用了一种改进的实现，即与[37]类似的端到端的联合训练，比[19]得到了更高的基准。

Table 5 shows the box and mask AP on COCO. We see that a single non-local block improves all R50/101 and X152 baselines, on all metrics involving detection and segmentation. APbox is increased by ∼1 point in all cases (e.g., +1.3 point in R101). Our non-local block is complementary to increasing the model capacity, even when the model is upgraded from R50/101 to X152. This comparison suggests that non-local dependency has not been sufficiently captured by existing models despite increased depth/capacity.

表5给出了在COCO上的box和mask AP。我们看到，一个non-local模块对所有的R50/101和X152基准都有所改进，在所有的检测和分割的度量上都是。APbox大约增加了1个点。我们的non-local模块与增加模型容量来说是互补的。这种比较说明，即使现有的模型增加了深度/容量，non-local依赖关系仍然没有得到充分的利用。

In addition, the above gain is at a very small cost. The single non-local block only adds <5% computation to the baseline model. We also have tried to use more non-local blocks to the backbone, but found diminishing return.

此外，上面的改进的代价是很小的。与基准模型相比，单个non-local模块只增加了不到5%的计算量。我们还尝试了使用更多的non-local模块，但发现收益逐渐减少。

**Keypoint detection**. Next we evaluate non-local blocks in Mask R-CNN for keypoint detection. In [19], Mask R-CNN used a stack of 8 convolutional layers for predicting the keypoints as 1-hot masks. These layers are local operations and may overlook the dependency among keypoints across long distance. Motivated by this, we insert 4 non-local blocks into the keypoint head (after every 2 convolutional layers).

下面我们对关键点检测中的Mask R-CNN评估non-local模块。在[19]中，Mask R-CNN使用了8个卷积层的堆叠，将关键点预测为one-hot masks。这些层都是local运算，可能会忽视关键点之间长距离的依赖关系。受此启发，我们在关键点头上插入了4个non-local模块。

Table 6 shows the results on COCO. On a strong baseline of R101, adding 4 non-local blocks to the keypoint head leads to a ∼1 point increase of keypoint AP. If we add one extra non-local block to the backbone as done for object detection, we observe an in total 1.4 points increase of keypoint AP over the baseline. In particular, we see that the stricter criterion of AP75 is boosted by 2.4 points, suggesting a stronger localization performance.

表6给出了在COCO上的结果。在R101的强基准上，给关键点头增加4个non-local模块，带来了大约1个点的提升。如果我们再增加一个non-local模块，我们会得到1.4点的性能提升。特别的，我们看到了AP75的更严格的原则提升了2.4点，说明定位性能非常好。

## 7. Conclusion

We presented a new class of neural networks which capture long-range dependencies via non-local operations. Our non-local blocks can be combined with any existing architectures. We show the significance of non-local modeling for the tasks of video classification, object detection and segmentation, and pose estimation. On all tasks, a simple addition of non-local blocks provides solid improvement over baselines. We hope non-local layers will become an important component of future network architectures.

我们提出了一种新的神经网络类别，通过non-local运算捕获长程依赖关系。我们的non-local模块可以与任何以后的架构结合。我们证明了non-local模型在视频分类、目标检测和分割、姿态估计中的有效性。在所有任务中，简单的加入non-local模块就可以在基准上得到很好的改进。我们希望non-local层会变成未来网络架构的一个重要组件。