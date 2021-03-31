# Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis

Patrice Y. Simard, Dave Steinkraus, John C. Platt Microsoft Research

## 0. Abstract

Neural networks are a powerful technology for classification of visual inputs arising from documents. However, there is a confusing plethora of different neural network methods that are used in the literature and in industry. This paper describes a set of concrete best practices that document analysis researchers can use to get good results with neural networks. The most important practice is getting a training set as large as possible: we expand the training set by adding a new form of distorted data. The next most important practice is that convolutional neural networks are better suited for visual document tasks than fully connected networks. We propose that a simple “do-it-yourself” implementation of convolution with a flexible architecture is suitable for many visual document problems. This simple convolutional neural network does not require complex methods, such as momentum, weight decay, structure-dependent learning rates, averaging layers, tangent prop, or even finely-tuning the architecture. The end result is a very simple yet general architecture which can yield state-of-the-art performance for document analysis. We illustrate our claims on the MNIST set of English digit images.

神经网络是文档图像输入分类的很好技术。但是，在文献和工业中，有很多不同种类的神经网络在使用。本文描述了一些最好的做法，文档分析研究者用这些，可以在神经网络上得到很好的结果。最重要的是得到一个尽可能大的训练集：我们通过增加一种新形式的变形数据，对训练集进行拓展。也很重要的是，卷积神经网络比全连接网络更适合视觉文档任务。我们提出了一种简单的卷积的DIY实现，有着灵活的架构，适合于很多视觉文档问题。这种简单的CNN不需要复杂的方法，比如动量，权重衰减，依赖于结构的学习速率，平均层，tangent prop，或甚至是精调架构。最终结果是一种非常简单但通用的架构，对文档分析可以得到目前最好的结果。我们在英文数字图像的MNIST上，阐述我们的结果。

## 1. Introduction

After being extremely popular in the early 1990s, neural networks have fallen out of favor in research in the last 5 years. In 2000, it was even pointed out by the organizers of the Neural Information Processing System (NIPS) conference that the term “neural networks” in the submission title was negatively correlated with acceptance. In contrast, positive correlations were made with support vector machines (SVMs), Bayesian networks, and variational methods.

神经网络在1990s早期非常流行，但在后来的5年中就没有那么受欢迎了。在2000年，NIPS的组织者甚至指出，提交论文中带有神经网络术语的，不太容易被接收。比较之下，SVM，贝叶斯网络和变分方法更加受欢迎。

In this paper, we show that neural networks achieve the best performance on a handwriting recognition task (MNIST). MNIST [7] is a benchmark dataset of images of segmented handwritten digits, each with 28x28 pixels. There are 60,000 training examples and 10,000 testing examples.

本文中，我们展示了神经网络在手写数字识别任务(MNIST)中得到了最好的结果。MNIST是一个手写数字图像基准测试数据集，每个图像大小28x28像素。有60000个训练样本，10000个测试样本。

Our best performance on MNIST with neural networks is in agreement with other researchers, who have found that neural networks continue to yield state-of-the-art performance on visual document analysis tasks [1, 2].

我们在MNIST上用神经网络得到的最佳性能，是得到其他研究者同意的，他们也发现，神经网络在视觉文档分析任务中一直可以得到最好的性能。

The optimal performance on MNIST was achieved using two essential practices. First, we created a new, general set of elastic distortions that vastly expanded the size of the training set. Second, we used convolutional neural networks. The elastic distortions are described in detail in Section 2. Sections 3 and 4 then describe a generic convolutional neural network architecture that is
simple to implement.

在MNIST上的最佳性能，是用两种基本操作得到的。第一，我们创建了一组新的通用弹性形变，极大的拓展了训练集的大小。第二，我们使用了卷积神经网络。弹性形变在第2部分中详述。第3和第4部分描述了一种通用CNN架构，实现起来很简单。

We believe that these two practices are applicable beyond MNIST, to general visual tasks in document analysis. Applications range from FAX recognition, to analysis of scanned documents and cursive recognition (using a visual representation) in the Tablet PC.

我们相信，这两种操作是可以在MNIST之外应用的，在通用文本分析的通用视觉任务中。应用可以从FAX识别，到扫描文档分析，和平板PC中的手写识别（使用视觉表示）。

## 2. Expanding Data Sets through Elastic Distortions

Synthesizing plausible transformations of data is simple, but the “inverse” problem – transformation invariance – can be arbitrarily complicated. Fortunately, learning algorithms are very good at learning inverse problems. Given a classification task, one may apply transformations to generate additional data and let the learning algorithm infer the transformation invariance. This invariance is embedded in the parameters, so it is in some sense free, since the computation at recognition time is unchanged. If the data is scarce and if the distribution to be learned has transformation-invariance properties, generating additional data using transformations may even improve performance [6]. In the case of handwriting recognition, we postulate that the distribution has some invariance with respect to not only affine transformations, but also elastic deformations corresponding to uncontrolled oscillations of the hand muscles, dampened by inertia.

对数据的可行变换进行合成是简单的，但是其逆问题，变换不变性，的复杂度可能是任意的。幸运的是，学习算法非常擅长于学习逆问题。给定一个分类任务，可以使用变换来生成额外的数据，让学习算法来推理变换不变性。这种不变性是嵌入到参数中的，所以在某种意义上是免费的，因为在识别时的计算是不变的。如果数据很稀少，如果要学习的分布有变换不变的性质，用变换来生成额外的数据甚至会改进性能。在手写数字识别的情况中，我们假设分布不仅对仿射变换有一些不变性，而且还有弹性形变，对应着手部肌肉的不受控的震荡，受到惯性抑制。

Simple distortions such as translations, rotations, and skewing can be generated by applying affine displacement fields to images. This is done by computing for every pixel a new target location with respect to the original location. The new target location, at position (x,y) is given with respect to the previous position. For instance if ∆x(x,y)=1, and ∆y(x,y)=0, this means that the new location of every pixel is shifted by 1 to the right. If the displacement field was: ∆x(x,y)= αx, and ∆y(x,y)= αy, the image would be scaled by α, from the origin location (x,y)=(0,0). Since α could be a non-integer value, interpolation is necessary.

简单的形变，比如平移，旋转，和扭曲可以通过将仿射偏移场应用到图像中生成。这是通过将对每个像素计算一个相对于原始位置的新目标位置。这种新的目标位置，在(x,y)位置上，是相对于之前的位置给出的。比如，如果∆x(x,y)=1, ∆y(x,y)=0，这意味着每个像素的新位置都向右平移了1。如果偏移场是∆x(x,y)= αx, ∆y(x,y)= αy，图像可能从原始位置(x,y)=(0,0)缩放了系数α。由于α可能是一个非整数值，那么就需要进行插值。

Figure 1 illustrates how to apply a displacement field to compute new values for each pixel. In this example, the location of A is assumed to be (0,0) and the numbers 3, 7, 5, 9 are the grey levels of the image to be transformed, at the locations (1,0), (2,0), (1,-1) and (2,-1) respectively. The displacements for A are given by ∆x(0,0) = 1.75 and ∆y(0,0) = -0.5 as illustrated in the figure the arrow. The new grey value for A in the new (warped) image is computed by evaluating the grey level at location (1.75,-0.5) from the original image. A simple algorithm for evaluating the grey level is “bilinear interpolation” of the pixel values of the original image. Although other interpolation schemes can be used (e.g., bicubic and spline interpolation), the bilinear interpolation is one of the simplest and works well for generating additional warped characters image at the chosen resolution (29x29). Interpolating the value horizontally, followed by interpolating the value vertically, accomplishes the evaluation. To compute the horizontal interpolations, we first compute the location where the arrow ends with respect to the square in which it ends. In this case, the coordinates in the square are (0.75, 0.5), assuming the origin of that square is bottomleft (where the value 5 is located). In this example, the new values are: 3 + 0.75 × (7-3) = 6; and 5 + 0.75 × (9-5) = 8. The vertical interpolation between these values yields 8 + 0.5 × (6-8) = 7, which is the new grey level value for pixel A. A similar computation is done for all pixels. All pixel locations outside the given image are assumed to have a background value (e.g. 0).

图1描述了怎样对每个像素应用一个偏移场，计算像素的新值。在这个例子中，位置A假设为(0,0)，数字3, 7, 5, 9是要变换的图像灰度值，所在位置分别为(1,0), (2,0), (1,-1)和(2,-1)。A点在新的（变形）图像中的新灰度值，是通过计算在相对于原始位置(1.75, -0.5)上的灰度值得到的。一个简单的计算灰度值的算法是原始图像灰度值的双线性插值。虽然其他的插值方法也可以使用（如，双三次和样条插值），双线性插值是最简单的一种，在给定的分辨率(29x29)上生成额外形变的数字图像效果很好。水平方向的插值，然后进行垂直方向的插值，这样就完成了整个计算。为计算水平方向的插值，我们首先计算箭头指向的位置，相对于其结束时的方形。在这种情况下，在这个方形中的坐标是(0.75, 0.5)，假设这个方形的原点是左下（值5所在的位置）。在这个例子中，新的值是3 + 0.75 × (7-3) = 6; 和5 + 0.75 × (9-5) = 8。这两个值之间垂直方向的插值得到，8 + 0.5 × (6-8) = 7，这就是像素A的新灰度值。对所有像素进行类似的计算。在给定图像之外的所有像素位置假设是背景值（如0）。

Affine distortions greatly improved our results on the MNIST database. However, our best results were obtained when we used elastic deformations. The image deformations were created by first generating random displacement fields, that is ∆x(x,y) = rand(-1,+1) and ∆y(x,y)=rand(-1,+1), where rand(-1,+1) is a random number between -1 and +1, generated with a uniform distribution. The fields ∆x and ∆y are then convolved with a Gaussian of standard deviation σ (in pixels). If σ is large, the resulting values are very small because the random values average 0. If we normalize the displacement field (to a norm of 1), the field is then close to constant, with a random direction. If σ is small, the field looks like a completely random field after normalization (as depicted in Figure 2, top right). For intermediate σ values, the displacement fields look like elastic deformation, where σ is the elasticity coefficient. The displacement fields are then multiplied by a scaling factor α that controls the intensity of the deformation.

仿射形变极大的改进了在MNIST数据集上的结果。但是，我们最好的结果是当我们使用弹性形变得到的。图像形变的创建，首先是生成随机偏移场，即∆x(x,y) = rand(-1,+1)和∆y(x,y)=rand(-1,+1)，其中rand(-1,+1)是一个在-1和+1之间的随机数，用均匀分布生成。场∆x和∆y然后和一个标准差σ的高斯函数进行卷积。如果σ很大，得到的值非常小，因为随机值的均值为0。如果我们对偏移场进行归一化（归一到范数为1），那么场就接近于常数，方向随机。如果σ很小，这个场在归一化后就会像一个完全随机的场（如图2右上所示）。对于中间的σ值，偏移场看起来很像弹性形变，其中σ是弹性系数。偏移场然后乘以一个缩放系数α，控制形变的强度。

Figure 2 shows example of a pure random field (σ=0.01), a smoothed random field corresponding to the properties of the hand (σ=8), and a smoothed random field corresponding to too much variability (σ=4). If σ is large, the displacements become close to affine, and if σ is very large, the displacements become translations.

图2展示了一个完全随机场的例子(σ=0.01)，一个平滑的随机场，对应着手的性质(σ=8)，和一个平滑的随机场，对应着太多的变化(σ=4)。如果σ很大，偏移就变得接近于仿射，如果σ非常大，那么偏移场就变成了平移。

In our MNIST experiments (29x29 input images), the values that yielded the best results were σ=4 and α=34. 在我们的MNIST试验中（输入图像为29x29），得到最好结果的值为σ=4和α=34。

## 3. Neural Networks Architectures for Visual Tasks

We considered two types of architectures neural network architectures for the MNIST data set. The simplest architecture, which is a universal classifier, is a fully connected network with two layers [4]. A more complicated architecture is a convolutional neural network, which has been found to be well-suited for visual document analysis tasks [3]. The implementation of standard neural networks can be found in textbooks, such as [5]. Section 4 describes a new, simple implementation of convolutional neural networks.

我们考虑两类神经网络架构在MNIST数据集上进行处理。最简单的架构，是一个通用分类器，是一个两层的全连接网络。一个更复杂的架构是卷积神经网络，很适合于视觉文档分析任务。标准神经网络的实现可以在书本上发现，比如[5]。第4部分描述了卷积神经网络的一种新的简单实现。

To test our neural networks, we tried to keep the algorithm as simple as possible, for maximum reproducibility. We only tried two different error functions: cross-entropy (CE) and mean squared error (MSE) (see [5, chapter 6] for more details). We avoided using momentum, weight decay, structure-dependent learning rates, extra padding around the inputs, and averaging instead of subsampling. (We were motivated to avoid these complications by trying them on various architecture/distortions combinations and on a train/validation split of the data and finding that they did not help.)

为测试我们的神经网络，我们试图让算法尽可能简单，以尽可能可以复现。我们只尝试了两种不同的误差函数：交叉熵(CE)和均方误差(MSE)。我们没有使用动量，权重衰减，依赖于结构的学习速率，对输入额外的padding，用平均替换了下采样。

Our initial weights were set to small random values (standard deviation = 0.05). We used a learning rate that started at 0.005 and is multiplied by 0.3 every 100 epochs.

我们的初始权重设置为小的随机值（标准差为0.05）。我们使用的学习速率初始为0.005，每100个epochs就乘以0.3。

### 3.1. Overall architecture for MNIST

As described in Section 5, we found that the convolutional neural network performs the best on MNIST. We believe this to be a general result for visual tasks, because spatial topology is well captured by convolutional neural networks [3], while standard neural networks ignore all topological properties of the input. That is, if a standard neural network is retrained and retested on a data set where all input pixels undergo a fixed permutation, the results would be identical.

如第5部分所述，我们发现卷积神经网络在MNIST上表现最好。我们相信这是视觉任务的通用结果，因为CNN可以很好的捕获空间拓扑，而标准神经网络忽略了输入的所有拓扑性质。即，如果一个标准神经网络在一个数据集上进行了重新训练和重新测试，这个数据集上的所有输入像素都经历了固定的变换，那么结果会是完全一样的。

The overall architecture of the convolutional neural network we used for MNIST digit recognition is depicted in Figure 3. 我们用于MNIST数字识别的CNN的总体架构如图3所示。

The general strategy of a convolutional network is to extract simple features at a higher resolution, and then convert them into more complex features at a coarser resolution. The simplest was to generate coarser resolution is to sub-sample a layer by a factor of 2. This, in turn, is a clue to the convolutions kernel's size. The width of the kernel is chosen be centered on a unit (odd size), to have sufficient overlap to not lose information (3 would be too small with only one unit overlap), but yet to not have redundant computation (7 would be too large, with 5 units or over 70% overlap). A convolution kernel of size 5 is shown in Figure 4. The empty circle units correspond to the subsampling and do not need to be computed. Padding the input (making it larger so that there are feature units centered on the border) did not improve performance significantly. With no padding, a subsampling of 2, and a kernel size of 5, each convolution layer reduces the feature size from n to (n-3)/2. Since the initial MNIST input size 28x28, the nearest value which generates an integer size after 2 layers of convolution is 29x29. After 2 layers of convolution, the feature size of 5x5 is too small for a third layer of convolution. The first feature layer extracts very simple features, which after training look like edge, ink, or intersection detectors. We found that using fewer than 5 different features decreased performance, while using more than 5 did not improve it. Similarly, on the second layer, we found that fewer than 50 features (we tried 25) decreased performance while more (we tried 100) did not improve it. These numbers are not critical as long as there are enough features to carry the information to the classification layers.

卷积网络的总体策略是，在更高分辨率上提取简单的特征，然后将其转化成更粗糙分辨率上的更复杂特征。生成更粗糙的分辨率，最简单的是要将一个层下采样2倍。这是卷积核大小的一个线索。核的宽度要选择成可以使一个单元在中央（即奇数的大小），要有足够的覆盖，不要损失信息（3可能就太小了，因为只覆盖了1个像素），但不能有荣誉的计算（7可能就太大了，有5个单元或超过了70%的重叠）。图4给出了一个大小为5的卷积核。空心的圆圈对应着下采样，不需要进行计算。将输入进行padding（使其变大，这样边缘的特征单元会在中央）并没有显著改进性能。在没有padding的情况下，下采样2倍，核大小为5，每个卷积层将特征从n降低到(n-3)/2。由于初始的MNIST输入大小为28x28，在两层卷积后生成整数大小的最近的值是29x29。在2层卷积后，5x5的特征大小太小了，不能进行第三次卷积。第一个特征层提取出了很简单的特征，在训练后看起来像边缘，ink或相交的检测器。我们发现，使用少于5个不同的特征会降低性能，而使用多余5个也并不会有所改进。类似的，在第二层，我们发现少于50个特征（我们尝试了25）会降低性能，而更多的（我们尝试了100）并不会改进性能。只要有足够的特征将信息输送到分类层，这些数值并不是很关键。

The first two layers of this neural network can be viewed as a trainable feature extractor. We now add a trainable classifier to the feature extractor, in the form of 2 fully connected layers (a universal classifier). The number of hidden units is variable, and it is by varying this number that we control the capacity, and the generalization, of the overall classifier. For MNIST (10 classes), the optimal capacity was reached with 100 hidden units.

这个神经网络的前两层可以视为可训练的特征提取器。我们现在对特征提取器加入了一个可训练的分类器，形式是2个全连接层（一个通用分类器）。隐藏单元的数量的可变的，通过改变这个数量，我们控制着总体分类器的能力和泛化。对于MNIST（10类），当有100个隐藏单元时，可以达到最佳能力。

## 4. Making Convolutional Neural Networks Simple

Convolutional neural networks have been proposed for visual tasks for many years [3], yet have not been popular in the engineering community. We believe that is due to the complexity of implementing the convolutional neural networks. This paper presents new methods for implementing such networks that are much easier that previous techniques and allow easy debugging.

CNN提出用于视觉任务很多年了，但在工程团体中并不是很流行。我们相信，这是因为实现CNN非常复杂。本文提出了实现这样的网络的简单方法，比之前的技术更加简单，可以进行简单的调试。

### 4.1. Simple Loops for Convolution

Fully connected neural networks often use the following rules to implement the forward and backward propagation: 全连接神经网络使用下面的规则来实现前向和反向传播：

$$x_j^{L+1} = \sum_i w_{j,i}^{L+1} x_i^L$$(1.1)

$$g_i^L = \sum_j w_{j,i}^{L+1} g_j^{L+1}$$(1.2)

where $x_i^L$ and $g_i^L$ are respectively the activation and the gradient of unit i at layer L, and $w_{j,i}^{L+1}$ is the weight connecting unit i at layer L to unit j at layer L+1. This can be viewed as the activation units of the higher layer “pulling” the activations of all the units connected to them. Similarly, the units of the lower layer are pulling the gradients of all the units connected to them. The pulling strategy, however, is complex and painful to implement when computing the gradients of a convolutional network. The reason is that in a convolution layer, the number of connections leaving each unit is not constant because of border effects.

其中$x_i^L$和$g_i^L$分别是L层单元i的激活和梯度， $w_{j,i}^{L+1}$是连接L层的单元i和L+1层的单元j的权重。这可以视为更高层的激活单元拉着与之相连的所有单元的激活。类似的，更低层的单元拉着与之相连的所有单元的梯度。这种拉着的策略，在计算卷积网络的梯度时，实现是复杂痛苦的。原因是，在一个卷积层中，离开每个单元的连接数量并不是常数，因为有边界效应。

This is easy to see on Figure 4, where all the units labeled $g^0_i$ have a variable number of outgoing connections. In contrast, all the units on the upper layer have a fixed number of incoming connections. To simplify computation, instead of pulling the gradient from the lower layer, we can “push” the gradient from the upper layer. The resulting equation is:

在图4中可以很容易看到，其中所有标注了$g^0_i$的单元的出射连接数量都有变化。对比起来，上面一层的所有单元的进入连接数量是固定的。为简化计算，而不是从更低的层中拉这些梯度，我们从上层推这些梯度。得到的方程是：

$$g_{j+1}^L += w_i^{L+1} g_j^{L+1}$$(1.3)

For each unit j in the upper layer, we update a fixed number of (incoming) units i from the lower layer (in the figure, i is between 0 and 4). Because in convolution the weights are shared, w does not depend on j. Note that pushing is slower than pulling because the gradients are accumulated in memory, as opposed to in pulling, where gradient are accumulated in a register. Depending on the architecture, this can sometimes be as much as 50% slower (which amounts to less than 20% decrease in overall performance). For large convolutions, however, pushing the gradient may be faster, and can be used to take advantage of Intel's SSE instructions, because all the memory accesses are contiguous. From an implementation standpoint, pulling the activation and pushing the gradient is by far the simplest way to implement convolution layers and well worth the slight compromise in speed.

对于上层的每个单元j，我们更新从低层来的单元i的固定数量的入射（在本图中，i是在0到4之间）。因为在卷积中权重是共享的，w并不依赖于j。注意pushing比pulling更慢一些，因为梯度是在内存中累加的，而对于pulling，其梯度是在寄存器中累加的。依赖于架构，这有时候可能会慢50%（会带来共计20%的总体性能下降）。对于大型卷积，pushing梯度可能会更快，还可以受益于Intel的SSE指令集，因为所有的内存访问都是相邻的。从实现的角度来看，pulling激活，pushing梯度，是目前最简单的实现卷积层的方法，对于在速度上的些许折中，是值得的。

### 4.2. Modular debugging

Back-propagation has a good property: it allows neural networks to be expressed and debugged in a modular fashion. For instance, we can assume that a module M has a forward propagation function which computes its output M(I,W) as a function of its input I and its parameters W. It also has a backward propagation function (with respect to the input) which computes the input gradient as a function of the output gradient, a gradient function (with respect to the weight), which computes the weight gradient with respect to the output gradient, and a weight update function, which adds the weight gradients to the weights using some updating rules (batch, stochastic, momentum, weight decay, etc). By definition, the Jacobian matrix of a function M is defined to be $J_{ki} = \frac {∂M_k}{∂x_i}$ (see [5] p. 148 for more information on Jacobian matrix for neural network). Using the backward propagation function and the gradient function, it is straightforward to compute the two Jacobian matrices $\frac {∂I}{∂M(I,W)}$ and $\frac {∂W}{∂M(I,W)}$ by simply feeding the (gradient) unit vectors ∆M_k(I,W) to both of these functions, where k indexes all the output units of M, and only unit k is set to 1 and all the others are set to 0. Conversely, we can generate arbitrarily accurate estimates of the Jacobians matrices $\frac {∂M(I,W)}{∂I}$ and $\frac {∂M(I,W)}{∂W}$ by adding small variations ε to I and W and calling the M(I,W) function. Using the equalities:

反向传播有一个很好的性质：它可以使神经网络以模块化的方式进行表示和调试。比如，我们可以假设一个模块M有一个前向传播函数，计算其输出M(I,W)为其输入I和其参数W的函数。它也有一个反向传播函数（对输入来说），将输入梯度计算为输出梯度的函数，一个梯度函数（对权重），对输出梯度计算的权重梯度，一个权重更新函数，使用一些更新规则将权重梯度加入到权重上（批次，随机，动量，权重衰减，等）。从定义上来说，一个函数M的Jacobian矩阵定义为$J_{ki} = \frac {∂M_k}{∂x_i}$。使用反向传播函数和梯度函数，计算两个Jacobian矩阵$\frac {∂I}{∂M(I,W)}$ and $\frac {∂W}{∂M(I,W)}$就非常直观简单了，只需要将梯度单元向量∆M_k(I,W)送入到这两个函数中，其中k是M的所有输出单元的索引，只有单元k设为1，其余设为0。相反的，我们可以生成Jacobian矩阵$\frac {∂M(I,W)}{∂I}$ and $\frac {∂M(I,W)}{∂W}$任意准确的估计，只要对I和W加入很小的扰动，调用M(I,W)函数。使用下式：

$$\frac {∂I}{∂M} = F(\frac {∂M}{∂I})^T, \frac {∂W}{∂M} = F(\frac {∂M}{∂W})^T$$

where F is a function which takes a matrix and inverts each of its elements, one can automatically verify that the forward propagation accurately corresponds to the backward and gradient propagations (note: the backpropagation computes F(∂I/∂M(I,W)) directly so only a transposition is necessary to compare it with the Jacobian computed by the forward propagation). In other words, if the equalities above are verified to the precision of the machine, learning is implemented correctly. This is particularly useful for large networks since incorrect implementations sometimes yield reasonable results. Indeed, learning algorithms tend to be robust even to bugs. In our implementation, each neural network is a C++ module and is a combination of more basic modules. A module test program instantiates the module in double precision, sets ε=10^−12 (the machine precision for double is 10^−16), generates random values for I and W, and performs a correctness test to a precision of 10^−10. If the larger module fails the test, we test each of the sub-modules until we find the culprit. This extremely simple and automated procedure has saved a considerable amount of debugging time.

其中F是一个函数，将一个矩阵的所有元素求逆，可以自动验证，前向传播准确的对应着反向和梯度传播（注意：反向传播直接计算F(∂I/∂M(I,W))，所以只需要进行转置，然后与前向传播计算的Jacobian进行比较）。换句话说，如果上面的量与机器的精度验证相符，学习就准确的实现了。这对于大型网络尤其有用，因为不准确的实现有时候也会得到合理的结果。确实，学习算法倾向于对bugs都稳健。在我们的实现中，每个神经网络都是一个C++模块，是更基本的模块的组合。一个模块测试程序以双精度实例化这个模块，设置ε=10^−12（机器上浮点数的精度是10^-16），对I和W生成随机值，对正确性的测试精度为10^-10。如果更大的模块在测试中失败了，我们测试每个子模块，直到我们找到出问题的地方。这个极其简单和自动化的过程，节省了大量调试时间。

## 5. Results

For both fully connected and convolutional neural networks, we used the first 50,000 patterns of the MNIST training set for training, and the remaining 10,000 for validation and parameter adjustments. The result reported on test set where done with the parameter values that were optimal on validation. The two-layer Multi-Layer Perceptron (MLP) in this paper had 800 hidden units, while the two-layer MLP in [3] had 1000 hidden units. The results are reported in the table below:

对全连接网络和CNN，我们都使用MNIST的前50000个样本进行训练，剩下的10000进行验证和参数调整。得到的在测试集上的结果，是在验证集上的最优参数进行的。本文中的两层MLP有800个隐藏单元，而[3]中的两层MLP有1000个节点。结果如下表所示：

There are several interesting results in this table. The most important is that elastic deformations have a considerable impact on performance, both for the 2 layer MLP and our convolutional architectures. As far as we know, 0.4% error is the best result to date on the MNIST database. This implies that the MNIST database is too small for most algorithms to infer generalization properly, and that elastic deformations provide additional and relevant a-priori knowledge. Second, we observe that convolutional networks do well compared to 2-layer MLPs, even with elastic deformation. The topological information implicit in convolutional networks is not easily inferred by MLP, even with the large training set generated with elastic deformation. Finally, we observed that the most recent experiments yielded better performance than similar experiments performed 8 years ago and reported in [3]. Possible explanations are that the hardware is now 1.5 orders of magnitude faster (we can now afford hundreds of epochs) and that in our experiments, CE trained faster than MSE.

这个表中有几个有趣的结果。最重要的是，弹性形变对性能有很大的影响，对2层MLP和CNN都是。据我们所知，0.4%的误差是在MNIST数据集上的目前最好结果。这说明，MNIST数据集太小了，多数算法都不能很好的推理出泛化，弹性形变提供了额外的和相关的先验知识。第二，我们观察到，CNN比2层MLP效果更好，即使是带有弹性形变的2层MLP。在CNN中隐式存在的拓扑信息，即使是用弹性形变生成的大型训练集，MLP也是很难推断得到的。最后，我们观察到，最近的试验比8年前[3]中的类似试验，得到了更好的性能。可能的解释是，硬件速度已经快了1.5倍（我们现在可以进行几百epochs的训练），在我们的试验中，CE的训练比MSE要快。

## 6. Conclusions

We have achieved the highest performance known to date on the MNIST data set, using elastic distortion and convolutional neural networks. We believe that these results reflect two important issues.

我们使用了弹性形变和CNN，在MNIST数据集上得到了目前最好的性能。我们相信，这些结果反应了两个重要的问题。

Training set size: The quality of a learned system is primarily dependent of the size and quality of the training set. This conclusion is supported by evidence from other application areas, such as text[8]. For visual document tasks, this paper proposes a simple technique for vastly expanding the training set: elastic distortions. These distortions improve the results on MNIST substantially.

训练集大小：一个学习得到的系统的质量，基本上是由训练集的大小和数量决定的。这个结论其他应用领域的证据也是支持的，比如文本[8]。对于视觉文档任务，本文提出一种简单的技术，极大的拓展了训练集：弹性形变。这些形变极大的改进了MNIST的训练结果。

Convolutional Neural Networks: Standard neural networks are state-of-the-art classifiers that perform about as well as other classification techniques that operate on vectors, without knowledge of the input topology. However, convolutional neural network exploit the knowledge that the inputs are not independent elements, but arise from a spatial structure.

CNN：标准神经网络是目前最好的分类器，与其他在向量上进行运算的分类技术性能一样好，不需要输入拓扑的知识。但是，CNN探索了如下知识，即输入并不是独立的元素，而是呈现出一种空间结构的。

Research in neural networks has slowed, because neural network training is perceived to require arcane black magic to get best results. We have shown that the best results do not require any arcane techniques: some of the specialized techniques may have arisen from computational speed limitations that are not applicable in the 21st Century.

对神经网络的研究已经变慢，因为神经网络训练被认为是需要奥术黑魔法来得到最好结果的。我们展示了不需要任何神秘技术的最好结果：一些专用技术可能是由计算速度限制中出现的。