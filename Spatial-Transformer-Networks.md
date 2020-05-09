# Spatial Transformer Networks

Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu

Google DeepMind, London, UK

## 0. Abstract

Convolutional Neural Networks define an exceptionally powerful class of models, but are still limited by the lack of ability to be spatially invariant to the input data in a computationally and parameter efficient manner. In this work we introduce a new learnable module, the Spatial Transformer, which explicitly allows the spatial manipulation of data within the network. This differentiable module can be inserted into existing convolutional architectures, giving neural networks the ability to actively spatially transform feature maps, conditional on the feature map itself, without any extra training supervision or modification to the optimisation process. We show that the use of spatial transformers results in models which learn invariance to translation, scale, rotation and more generic warping, resulting in state-of-the-art performance on several benchmarks, and for a number of classes of transformations.

CNNs定义了一类非常强大的模型，但在计算上高效、参数也足够少的情况下，仍然受限于缺少对于输入数据的空间不变形的能力。本文中，我们提出了一种新的可学习的模块，即空间变换器(Spatial Transformer)，可以在网络中对数据进行显式的空间变换。这种可微分的模块，可以插入到现有的卷积架构中，使神经网络具有对特征图进行积极的空间变换的能力，只需要特征图本身，而不需要任何额外的训练监督，也不需要对优化过程有任何修改。我们证明了，运用spatial transformer得到的模型，可以学习到对平移、缩放、旋转和更通用的形变都不变的特征，在几个基准测试中都得到了目前最好的性能，对几类变换都是。

## 1 Introduction

Over recent years, the landscape of computer vision has been drastically altered and pushed forward through the adoption of a fast, scalable, end-to-end learning framework, the Convolutional Neural Network (CNN) [21]. Though not a recent invention, we now see a cornucopia of CNN-based models achieving state-of-the-art results in classification [19, 28, 35], localisation [31, 37], semantic segmentation [24], and action recognition [12, 32] tasks, amongst others.

最近几年，CNN在计算机视觉得到了广泛的使用。虽然不是最近的发明，我们现在CNN模型在分类，检测，语义分割和行为检测任务中得到了很多目前最好的结果。

A desirable property of a system which is able to reason about images is to disentangle object pose and part deformation from texture and shape. The introduction of local max-pooling layers in CNNs has helped to satisfy this property by allowing a network to be somewhat spatially invariant to the position of features. However, due to the typically small spatial support for max-pooling (e.g. 2 × 2 pixels) this spatial invariance is only realised over a deep hierarchy of max-pooling and convolutions, and the intermediate feature maps (convolutional layer activations) in a CNN are not actually invariant to large transformations of the input data [6, 22]. This limitation of CNNs is due to having only a limited, pre-defined pooling mechanism for dealing with variations in the spatial arrangement of data.

一个能够对图像进行推理的系统的理想性质，应当可以将目标姿态、部件变形与纹理和形状区分开。CNNs中局部最大池化层的引入，使得网络在某种程度上对特征的位置是不变的，部分满足了这个性质。但是，由于max-pooling的空间支撑域通常很小（如，2×2像素），这种空间不变性只是通过max-pooling和卷积的深度层次化来实现的，中间的特征层（卷积层激活）实际上对于输入数据的较大的变化是不变的。这种CNNs的限制，是因为为了处理数据空间排布的变化，只有有限的、预定义的pooling机制。

In this work we introduce a Spatial Transformer module, that can be included into a standard neural network architecture to provide spatial transformation capabilities. The action of the spatial transformer is conditioned on individual data samples, with the appropriate behaviour learnt during training for the task in question (without extra supervision). Unlike pooling layers, where the receptive fields are fixed and local, the spatial transformer module is a dynamic mechanism that can actively spatially transform an image (or a feature map) by producing an appropriate transformation for each input sample. The transformation is then performed on the entire feature map (non-locally) and can include scaling, cropping, rotations, as well as non-rigid deformations. This allows networks which include spatial transformers to not only select regions of an image that are most relevant (attention), but also to transform those regions to a canonical, expected pose to simplify recognition in the following layers. Notably, spatial transformers can be trained with standard back-propagation, allowing for end-to-end training of the models they are injected in.

本文中，我们提出了一个Spatial Transformer模块，可以包含在标准的神经网络架构中，以提供空间变换能力。Spatial transformer的行为，是以单个数据样本为条件的，当然还有在训练中学习的合适的行为（没有额外的监督）。Pooling层的感受野是固定的，是局部的，与之不同，spatial transformer模块是一个动态机制，通过对每个输入样本产生一个合适的变换，从而主动的从空间上对一幅图像进行变换。其变换然后在整个特征图（非局部的）上进行，可以包括缩放，剪切，旋转，以及非刚性变形。这使得包含spatial transformer的网络，不仅可以选择图像中最相关的区域（最受关注的），而且将这些区域变换到一个标准的期望姿态，以简化下面的层中的识别。值得注意的是，spatial transformer可以用标准的反向传播进行训练，允许在插入的模型中进行端到端的训练。

Spatial transformers can be incorporated into CNNs to benefit multifarious tasks, for example: (i) image classification: suppose a CNN is trained to perform multi-way classification of images according to whether they contain a particular digit – where the position and size of the digit may vary significantly with each sample (and are uncorrelated with the class); a spatial transformer that crops out and scale-normalizes the appropriate region can simplify the subsequent classification task, and lead to superior classification performance, see Fig. 1; (ii) co-localisation: given a set of images containing different instances of the same (but unknown) class, a spatial transformer can be used to localise them in each image; (iii) spatial attention: a spatial transformer can be used for tasks requiring an attention mechanism, such as in [14, 39], but is more flexible and can be trained purely with backpropagation without reinforcement learning. A key benefit of using attention is that transformed (and so attended), lower resolution inputs can be used in favour of higher resolution raw inputs, resulting in increased computational efficiency.

Spatial transformers可以接入到CNN中，使多种任务受益，比如：(i)图像分类：假设CNN的训练是进行多路分类，判断图像是否包含特定数字，其中每个样本中数字的位置和大小会变化很显著（而且与类别无关）；对合适区域剪切出来并标准缩放的spatial transformer可以对后续的分类进行简化，并得到非常好的分类性能，如图1所示；(ii)协同定位：给定一个图像集，包含相同类别（但未知）的不同实例，可以用spatial transformer在每幅图像中进行定位；(iii)空间注意力：spatial transformer可以用于需要注意力机制的任务，比如[14,39]，但更加灵活，可以完全使用反向传播进行训练，不需要强化学习。使用注意力的一个关键好处是，可以使用变换过的（所以是关注的）低分辨率输入，而不实用高分辨率原始输入，得到提升的计算效率。

Figure 1: The result of using a spatial transformer as the first layer of a fully-connected network trained for distorted MNIST digit classification. (a) The input to the spatial transformer network is an image of an MNIST digit that is distorted with random translation, scale, rotation, and clutter. (b) The localisation network of the spatial transformer predicts a transformation to apply to the input image. (c) The output of the spatial transformer, after applying the transformation. (d) The classification prediction produced by the subsequent fully-connected network on the output of the spatial transformer. The spatial transformer network (a CNN including a spatial transformer module) is trained end-to-end with only class labels – no knowledge of the groundtruth transformations is given to the system.

The rest of the paper is organised as follows: Sect. 2 discusses some work related to our own, we introduce the formulation and implementation of the spatial transformer in Sect. 3, and finally give the results of experiments in Sect. 4. Additional experiments and implementation details are given in Appendix A.

本文组织如下：第2部分讨论了与本文相关的工作，第3部分提出了spatial transformer的公式和实现，最后在第4部分给出了试验结果。附录A中给出了额外的试验和实现。

## 2 Related Work

In this section we discuss the prior work related to the paper, covering the central ideas of modelling transformations with neural networks [15, 16, 36], learning and analysing transformation-invariant representations [4, 6, 10, 20, 22, 33], as well as attention and detection mechanisms for feature selection [1, 7, 11, 14, 27, 29].

本节中，我们讨论与本文相关的前人工作，覆盖了用神经网络对变换进行建模、学习并分析对变换不变的表示的中心思想，以及进行特征选择的注意力和检测机制。

Early work by Hinton [15] looked at assigning canonical frames of reference to object parts, a theme which recurred in [16] where 2D affine transformations were modeled to create a generative model composed of transformed parts. The targets of the generative training scheme are the transformed input images, with the transformations between input images and targets given as an additional input to the network. The result is a generative model which can learn to generate transformed images of objects by composing parts. The notion of a composition of transformed parts is taken further by Tieleman [36], where learnt parts are explicitly affine-transformed, with the transform predicted by the network. Such generative capsule models are able to learn discriminative features for classification from transformation supervision.

早期Hinton的工作[15]尝试过对目标部位指定标准的参考帧，在[16]中进行了同样思想的工作，对2D仿射变换进行建模，以创建一个生成式模型，由被变换的部位组成。生成式训练方案的目标是变换过的输入图像，输入图像和目标之间的变换是网络的一个额外输入。其结果是一个生成式模型，可以学习生成变换的图像，由各个部位一起组成目标。变换的部位的组合的概念，由Tieleman[36]进行了进一步研究，其中学习到的部位是显式的仿射变换，这个变换是由网络预测到的。这种生成式模块模型可以在变换监督下学习区分性的特征进行分类。

The invariance and equivariance of CNN representations to input image transformations are studied in [22] by estimating the linear relationships between representations of the original and transformed images. Cohen & Welling [6] analyse this behaviour in relation to symmetry groups, which is also exploited in the architecture proposed by Gens & Domingos [10], resulting in feature maps that are more invariant to symmetry groups. Other attempts to design transformation invariant representations are scattering networks [4], and CNNs that construct filter banks of transformed filters [20, 33]. Stollenga et al. [34] use a policy based on a network’s activations to gate the responses of the network’s filters for a subsequent forward pass of the same image and so can allow attention to specific features. In this work, we aim to achieve invariant representations by manipulating the data rather than the feature extractors, something that was done for clustering in [9].

CNN表示对输入图像变换的不变性和等变性在[22]中得到了研究，估计了原始图像和变换图像的表示的线性关系。Cohen & Welling [6]分析了这种行为与对称组之间的关系，这在Gens & Domingos [10]提出的架构中也得到了研究，得到的特征图，对对称组具有更多的不变性。其他的设计对变换不变的表示的努力有散射网络[4]，和构建由于变换滤波器组成的滤波器组的CNNs[20,33]。Stollenga等[34]使用一种基于网络激活的策略来对网络滤波器的响应进行门控，响应的输入信号是同样图像的前向序列，所以可以对特定的特征允许使用注意力机制，本文中，我们的目标是，通过对数据进行处理，而不是通过特征提取器，而得到不变的表示，在[9]中对聚类进行了这样的处理。

Neural networks with selective attention manipulate the data by taking crops, and so are able to learn translation invariance. Work such as [1, 29] are trained with reinforcement learning to avoid the need for a differentiable attention mechanism, while [14] use a differentiable attention mechansim by utilising Gaussian kernels in a generative model. The work by Girshick et al. [11] uses a region proposal algorithm as a form of attention, and [7] show that it is possible to regress salient regions with a CNN. The framework we present in this paper can be seen as a generalisation of differentiable attention to any spatial transformation.

有选择性注意力的神经网络通过剪切来处理数据，所以可以学习到平移不变性。[1,29]这样的工作用强化学习进行了训练，以避免需要可微分的注意力机制，而[14]通过在生成式模型中利用高斯核，用了一种可微分的注意力机制。Girshick等[11]的工作使用了一种区域建议算法作为一种形式的注意力，[7]表明，通过CNN回归得到显著性区域是可能的。我们在本文中提出的框架，可以看作是可微分注意力推广到了任意空间变换中。

## 3 Spatial Transformers

In this section we describe the formulation of a spatial transformer. This is a differentiable module which applies a spatial transformation to a feature map during a single forward pass, where the transformation is conditioned on the particular input, producing a single output feature map. For multi-channel inputs, the same warping is applied to each channel. For simplicity, in this section we consider single transforms and single outputs per transformer, however we can generalise to multiple transformations, as shown in experiments.

本节中，我们描述了spatial transformer的表达。这是一个可微分模块，在一个前向过程中，对特征图使用一种空间变换，其中变换是对特定的输入的，生成单个输出的特征图。对多通道输入，对每个通道都进行同样的变形。为简化起见，本节中，我们考虑每个transformer进行单个变换和当个输出，但是，我们可以泛化到多变换中，在试验中可以看到。

The spatial transformer mechanism is split into three parts, shown in Fig. 2. In order of computation, first a localisation network (Sect. 3.1) takes the input feature map, and through a number of hidden layers outputs the parameters of the spatial transformation that should be applied to the feature map – this gives a transformation conditional on the input. Then, the predicted transformation parameters are used to create a sampling grid, which is a set of points where the input map should be sampled to produce the transformed output. This is done by the grid generator, described in Sect. 3.2. Finally, the feature map and the sampling grid are taken as inputs to the sampler, producing the output map sampled from the input at the grid points (Sect. 3.3).

Spatial transformer机制分成三部分，如图2所示。为进行计算，首先特征图输入到一个localisation网络(3.1节)，通过几个隐含层输出空间变换的参数，这些参数应用到特征图中，这对给定输入进行了特定的变换。然后，预测的变换参数用于生成一个采样网格，这是一个点集，在输入特征图中的特定点上进行采样，以生成变换的输出。这通过网格生成器完成，如3.2节所述。最后，特征图和采样网格作为采样器的输入，生成输出图，即对输入在网格点上的采样。

Figure 2: The architecture of a spatial transformer module. The input feature map U is passed to a localisation network which regresses the transformation parameters θ. The regular spatial grid G over V is transformed to the sampling grid $T_θ(G)$, which is applied to U as described in Sect. 3.3, producing the warped output feature map V. The combination of the localisation network and sampling mechanism defines a spatial transformer.

The combination of these three components forms a spatial transformer and will now be described in more detail in the following sections.

这三个组件组成了一个spatial transformer，下面的小节更详细的进行描述。

### 3.1 Localisation Network 定位网络

The localisation network takes the input feature map $U ∈ R_{H×W×C}$ with width W, height H and C channels and outputs θ, the parameters of the transformation $T_θ$ to be applied to the feature map: $θ = f_{loc}(U)$. The size of θ can vary depending on the transformation type that is parameterised, e.g. for an affine transformation θ is 6-dimensional as in (10).

定位网络以特征图$U ∈ R_{H×W×C}$为输入，宽度W，高度H，通道数量C，输出θ，即要应用到特征图的变换的参数$T_θ$：$θ = f_{loc}(U)$。θ的大小根据要参数化的变换的类型而定，即，对于仿射变换，θ是6维的，如(10)所示。

The localisation network function $f_{loc}()$ can take any form, such as a fully-connected network or a convolutional network, but should include a final regression layer to produce the transformation parameters θ.

定位网络函数$f_{loc}()$可以采取任何形式，比如全连接网络，或卷积网络，但必须包含最后的回归层，以生成变换参数θ。

### 3.2 Parameterised Sampling Grid 参数化的取样网格

To perform a warping of the input feature map, each output pixel is computed by applying a sampling kernel centered at a particular location in the input feature map (this is described fully in the next section). By pixel we refer to an element of a generic feature map, not necessarily an image. In general, the output pixels are defined to lie on a regular grid G = {$G_i$} of pixels $G_i = (x^t_i, y_i^t)$, forming an output feature map $V ∈ R^{H'×W'×C}$, where H' and W' are the height and width of the grid, and C is the number of channels, which is the same in the input and output.

为对输入特征图进行形变，每个输出像素是通过在输入特征图的特定位置应用一个采样核计算得到的（这在下一节详细叙述）。这里pixel是指通用特征图的一个元素，并不一定是图像。一般来说，输出像素是在规则网格点G = {$G_i$}上的，其像素表示为$G_i = (x^t_i, y_i^t)$，形成一个输出特征图$V ∈ R^{H'×W'×C}$，其中H'和W'是网格的高度和宽度，C是通道数量，这在输入和输出中是一样的。

For clarity of exposition, assume for the moment that $T_θ$ is a 2D affine transformation $A_θ$. We will discuss other transformations below. In this affine case, the pointwise transformation is

为更清楚的进行阐述，假设现在$T_θ$是一个2D仿射变换$A_θ$。下面我们会讨论其他变换。在这个仿射的情形中，逐点的变换是：

$$\left( \begin{matrix} x_i^s \\ y_i^s \end{matrix} \right) = T_θ(G_i) = A_θ \left( \begin{matrix} x_i^t \\ y_i^t \\ 1 \end{matrix} \right) = \left[ \begin{matrix} θ_{11} & θ_{12} & θ_{13} \\ θ_{21} & θ_{22} & θ_{23} \end{matrix} \right] \left( \begin{matrix} x_i^t \\ y_i^t \\ 1 \end{matrix} \right)$$(1)

where ($x^t_i, y_i^t$) are the target coordinates of the regular grid in the output feature map, ($x^s_i, y_i^s$) are the source coordinates in the input feature map that define the sample points, and $A_θ$ is the affine transformation matrix. We use height and width normalised coordinates, such that $−1 ≤ x^t_i, y_i^t ≤ 1$ when within the spatial bounds of the output, and $−1 ≤ x^s_i, y_i^s ≤ 1$ when within the spatial bounds of the input (and similarly for the y coordinates). The source/target transformation and sampling is equivalent to the standard texture mapping and coordinates used in graphics [8].

其中($x^t_i, y_i^t$)是在输出特征图中常规网格的目标坐标系，($x^s_i, y_i^s$)是输入特征图的源坐标系，定义了采样点，$A_θ$是仿射变换矩阵。我们使用的是宽度和高度归一化的坐标系，比如在输出的空间边界中$−1 ≤ x^t_i, y_i^t ≤ 1$，在输入的空间边界中$−1 ≤ x^s_i, y_i^s ≤ 1$（对y坐标也是类似的）。源/目标变换和采样是与[8]中使用在图形学中的标准纹理映射和坐标是等价的。

The transform defined in (10) allows cropping, translation, rotation, scale, and skew to be applied to the input feature map, and requires only 6 parameters (the 6 elements of $A_θ$) to be produced by the localisation network. It allows cropping because if the transformation is a contraction (i.e. the determinant of the left 2 × 2 sub-matrix has magnitude less than unity) then the mapped regular grid will lie in a parallelogram of area less than the range of $x_i^s, y_i^s$. The effect of this transformation on the grid compared to the identity transform is shown in Fig. 3.

(10)定义的变换允许对输入特征图进行剪切、平移、旋转、缩放和偏斜，只需要定位网络生成的6个参数（$A_θ$的6个参数）。其可以允许剪切，是因为如果变换是一种收缩的话（即，左边的2×2的子矩阵其行列式小于1），那么映射过来的常规网格会平行四边形区域中，比$x_i^s, y_i^s$的范围要小。在网格中，这种变换的效果与恒等变换的比较，如图3所示。

Figure 3: Two examples of applying the parameterised sampling grid to an image U producing the output V. (a) The sampling grid is the regular grid $G = T_I(G)$, where I is the identity transformation parameters. (b) The sampling grid is the result of warping the regular grid with an affine transformation $T_θ(G)$.

The class of transformations $T_θ$ may be more constrained, such as that used for attention 变换$T_θ$的类别可能会更加首先，例如用于注意力的

$$A_θ = \left[ \begin{matrix} s & 0 & t_x \\ 0 & s & t_y \end{matrix} \right]$$(2)

allowing cropping, translation, and isotropic scaling by varying s, t_x, and t_y. The transformation $T_θ$ can also be more general, such as a plane projective transformation with 8 parameters, piecewise affine, or a thin plate spline. Indeed, the transformation can have any parameterised form, provided that it is differentiable with respect to the parameters – this crucially allows gradients to be backpropagated through from the sample points $T_θ(G_i)$ to the localisation network output θ. If the transformation is parameterised in a structured, low-dimensional way, this reduces the complexity of the task assigned to the localisation network. For instance, a generic class of structured and differentiable transformations, which is a superset of attention, affine, projective, and thin plate spline transformations, is $T_θ = M_θB$, where B is a target grid representation (e.g. in (10), B is the regular grid G in homogeneous coordinates), and $M_θ$ is a matrix parameterised by θ. In this case it is possible to not only learn how to predict θ for a sample, but also to learn B for the task at hand.

可以进行剪切、平移和各向同性的缩放，分别对应变化s, t_x和t_y。变换$T_θ$也可以更一般性，比如一个8个参数的平面投影变换，分段仿射，或薄板样条。实际上，变换可以有任何参数化的形式，只要对于其参数是可微分的，这很关键，允许梯度可以从采样点$T_θ(G_i)$反向传播回定位网络输出θ。如果变换是以一种结构化的、低维的方式进行的参数化，这降低了指定给定位网络的任务的复杂度。比如，一个结构化的可微分的变换的类别，是注意力、仿射、投影和薄板样条变换的超集，$T_θ = M_θB$，其中B是目标网格表示（如，在(10)中，B是均一性坐标系中的常规网格G），$M_θ$是以θ为参数的矩阵。本情况中，不仅可能学习怎么预测一个样本的参数θ，还可能学习任务中的B。

### 3.3 Differentiable Image Sampling

To perform a spatial transformation of the input feature map, a sampler must take the set of sampling points $T_θ(G)$, along with the input feature map U and produce the sampled output feature map V.

为对输入特征图进行空域变换，必须有一个sampler对采样点$T_θ(G)$上进行采样，与输入特征图U一起，生成采样的输出特征图V。

Each ($x^s_i, y_i^s$) coordinate in $T_θ(G)$ defines the spatial location in the input where a sampling kernel is applied to get the value at a particular pixel in the output V. This can be written as

$T_θ(G)$中的每个($x^s_i, y_i^s$)坐标，定义了输入中的空间位置，其中一个采样核进行了应用，以在输出V的一个特定像素位置上得到其值。这可以写成

$$V_i^c = \sum_n^H \sum_m^W U_{nm}^c k(x_i^s-m; Φ_x) k(y_i^s-n; Φ_y), ∀i∈[1...H'W'], ∀c∈[1...C]$$(3)

where $Φ_x$ and $Φ_y$ are the parameters of a generic sampling kernel k() which defines the image interpolation (e.g. bilinear), $U_{nm}^c$ is the value at location (n, m) in channel c of the input, and $V_i^c$ is the output value for pixel i at location ($x^t_i,y_i^t$) in channel c. Note that the sampling is done identically for each channel of the input, so every channel is transformed in an identical way (this preserves spatial consistency between channels).

其中$Φ_x$和$Φ_y$是一个通用采样核心k()的参数，定义了图像差值（如，双线性），$U_{nm}^c$是在位置(n,m)通道c上输入的值，$V_i^c$是像素i在位置($x^t_i,y_i^t$)在通道c中的值。注意，采样对输入的每个通道都同样的进行，所以每个通道都是用同样的方式进行变换（这在通道之间保持了空间一致性）。

In theory, any sampling kernel can be used, as long as (sub-)gradients can be defined with respect to $x_i^s$ and $y_i^s$. For example, using the integer sampling kernel reduces (3) to

理论上，任何采样核都可以使用，只要可以对$x_i^s$和$y_i^s$定义梯度。比如，使用整数采样核，(3)就成为了

$$V_i^c = \sum_n^H \sum_m^W U_{nm}^c δ(⌊x^s_i + 0.5⌋ − m)δ(⌊y_i^s + 0.5⌋ − n)$$(4)

where ⌊x + 0.5⌋ rounds x to the nearest integer and δ() is the Kronecker delta function. This sampling kernel equates to just copying the value at the nearest pixel to ($x^s_i, y_i^s$) to the output location ($x^t_i, y_i^t$). Alternatively, a bilinear sampling kernel can be used, giving

其中⌊x + 0.5⌋将x四舍五入到最近的整数，δ()是Kronecker delta函数。这个采样核等于将与($x^s_i, y_i^s$)最近的像素拷贝到输出位置($x^t_i, y_i^t$)。或者，也可以使用一个双线性采样核，得到

$$V_i^c = \sum_n^H \sum_m^W U_{nm}^c max(0, 1-|x_i^s-m|) max(0, 1-|y_i^s-n|)$$(5)

To allow backpropagation of the loss through this sampling mechanism we can define the gradients with respect to U and G. For bilinear sampling (5) the partial derivatives are

为允许损失沿着这种采样机制进行反向传播，我们可以定义对U和G的梯度。对于双线性采样(5)，其偏微分是

$$\frac {∂V_i^c}{∂U_{nm}^c} = \sum_n^H \sum_m^W max(0,1-|x_i^s-m|) max(0, 1-|y_i^s-n|)$$(6)
$$\frac {∂V_i^c}{∂x_i^s} = \sum_n^H \sum_m^W U_{nm}^c max(0, 1-|y_i^s-n|)  \left \{ \begin{matrix} 0 & if |m-x_i^s|≥1 \\ 1 & if m≥x_i^s \\ -1 & if m<x_i^s \end{matrix} \right.$$(7)

and similarly to (7) for $\frac{∂V_i^c}{∂y_i^s}$.

This gives us a (sub-)differentiable sampling mechanism, allowing loss gradients to flow back not only to the input feature map (6), but also to the sampling grid coordinates (7), and therefore back to the transformation parameters θ and localisation network since $\frac{∂x_i^s}{∂θ}$ and $\frac{∂y_i^s}{∂θ}$ can be easily derived from (10) for example. Due to discontinuities in the sampling fuctions, sub-gradients must be used. This sampling mechanism can be implemented very efficiently on GPU, by ignoring the sum over all input locations and instead just looking at the kernel support region for each output pixel.

这给了我们一个可微分的采样机制，使损失梯度的反向流动，不仅到输入特征图(6)，同时到采样网格坐标(7)中，因此反向到了变换参数θ和定位网络中，因为$\frac{∂x_i^s}{∂θ}$和$\frac{∂y_i^s}{∂θ}$可以很容易的从(10)推导出来。由于采样函数的不连续性，必须使用子梯度。这种采样机制可以在GPU上很高效的实现，只需忽略在所有输入位置上的和，而对每个输出像素只观察核支撑区域。

### 3.4 Spatial Transformer Networks

The combination of the localisation network, grid generator, and sampler form a spatial transformer (Fig. 2). This is a self-contained module which can be dropped into a CNN architecture at any point, and in any number, giving rise to spatial transformer networks. This module is computationally very fast and does not impair the training speed, causing very little time overhead when used naively, and even speedups in attentive models due to subsequent downsampling that can be applied to the output of the transformer.

定位网络、网格生成器和采样器的组合，形成了一个spatial transformer（图2）。这是一个自包含的模块，可以在CNN架构的任何点以任何数量插入，得到spatial transformer网络。这个模块计算起来很快，不影响训练速度，直接使用也不会带来很多时间开销，注意力模型甚至会有加速效果，因为后续的降采样可以用于transformer的输出。

Placing spatial transformers within a CNN allows the network to learn how to actively transform the feature maps to help minimise the overall cost function of the network during training. The knowledge of how to transform each training sample is compressed and cached in the weights of the localisation network (and also the weights of the layers previous to a spatial transformer) during training. For some tasks, it may also be useful to feed the output of the localisation network, θ, forward to the rest of the network, as it explicitly encodes the transformation, and hence the pose, of a region or object.

将spatial transformer放在一个CNN当中，使网络可以怎样主动的将特征图进行变换，以在训练时帮助最小化总计的网络损失函数。怎样变换每个训练样本的知识，在训练时是压缩并缓存在定位网络的权重中的（以及一个spatial transformer之前的层的权重）。对于一些任务，将定位网络的输出θ前向送入网络剩余部分，也可能有用，因为其显式对变换进行了编码，因此也包括了目标一个区域的姿态。

It is also possible to use spatial transformers to downsample or oversample a feature map, as one can define the output dimensions H' and W' to be different to the input dimensions H and W. However, with sampling kernels with a fixed, small spatial support (such as the bilinear kernel), downsampling with a spatial transformer can cause aliasing effects.

使用spatial transformer来对一个特征图进行下采样或上采样，也是可能的，因为可以定义输出维度H'和W'与输入维度H和W不一样。但是，采用一个固定的小型空间支撑的采样核（如双线性核），用spatial transformer进行下采样会导致aliasing的效果。

Finally, it is possible to have multiple spatial transformers in a CNN. Placing multiple spatial transformers at increasing depths of a network allow transformations of increasingly abstract representations, and also gives the localisation networks potentially more informative representations to base the predicted transformation parameters on. One can also use multiple spatial transformers in parallel – this can be useful if there are multiple objects or parts of interest in a feature map that should be focussed on individually. A limitation of this architecture in a purely feed-forward network is that the number of parallel spatial transformers limits the number of objects that the network can model.

最后，在一个CNN中是可以有多个spatial transformer的。在越来越深的网络中，植入多个spatial transformers，使越来越抽象的表示也可以进行变换，也给定位网络潜在的更有信息的表示，预测的变换参数也是以此为基础的。我们也可以并行的使用多个spatial transformers，如果在一个特征图中有多个感兴趣的目标或部位，需要单独关注，那么这就有用了。在一个纯前向网络中，这种架构的一种局限是，并行spatial transformers的数量，限制了网络可以建模的目标数量。

## 4 Experiments

In this section we explore the use of spatial transformer networks on a number of supervised learning tasks. In Sect. 4.1 we begin with experiments on distorted versions of the MNIST handwriting dataset, showing the ability of spatial transformers to improve classification performance through actively transforming the input images. In Sect. 4.2 we test spatial transformer networks on a challenging real-world dataset, Street View House Numbers [25], for number recognition, showing state-of-the-art results using multiple spatial transformers embedded in the convolutional stack of a CNN. Finally, in Sect. 4.3, we investigate the use of multiple parallel spatial transformers for fine-grained classification, showing state-of-the-art performance on CUB-200-2011 birds dataset [38] by discovering object parts and learning to attend to them. Further experiments of MNIST addition and co-localisation can be found in Appendix A.

本节中，我们探索了spatial transformer网络在几个有监督学习任务中的使用。在4.1节中，我们在变形的MNIST手写字体数据集上进行试验，表明spatial transformer通过主动对输入图像进行变形，改进了分类性能。在4.2节中，我们在一个很有挑战的真实世界数据集上测试了spatial transformer网络，房间号街景数据集，进行数字识别，表明在CNN中嵌入了多个spatial transformer模块，可以得到目前最好的结果。最后，在4.3节中，我们研究了使用多个并行的spatial transformers进行细粒度分类，在CUB-200-2011鸟类数据集中得到了目前最好的性能，发现了目标的部分，学习对其进行处理。附录A中有更多的MNIST上的试验和共同定位的结果。

### 4.1 Distorted MNIST

In this section we use the MNIST handwriting dataset as a testbed for exploring the range of transformations to which a network can learn invariance to by using a spatial transformer.

本节中，我们使用MNIST手写数字数据集作为试验，探索网络使用spatial transformer可以学习到对多大范围的变形可以学习到不变性。

We begin with experiments where we train different neural network models to classify MNIST data that has been distorted in various ways: rotation (R), rotation, scale and translation (RTS), projective transformation (P), and elastic warping (E) – note that elastic warping is destructive and can not be inverted in some cases. The full details of the distortions used to generate this data are given in Appendix A. We train baseline fully-connected (FCN) and convolutional (CNN) neural networks, as well as networks with spatial transformers acting on the input before the classification network (ST-FCN and ST-CNN). The spatial transformer networks all use bilinear sampling, but variants use different transformation functions: an affine transformation (Aff), projective transformation (Proj), and a 16-point thin plate spline transformation (TPS) [2]. The CNN models include two max-pooling layers. All networks have approximately the same number of parameters, are trained with identical optimisation schemes (backpropagation, SGD, scheduled learning rate decrease, with a multinomial cross entropy loss), and all with three weight layers in the classification network.

我们试验的开始，训练不同的神经网络模型来分类MNIST数据，这些数据以不同的方式进行了变形：旋转R，旋转、缩放和平移RTS，投影变形P，和弹性变形E，注意弹性变形是毁灭性的，在一些情形中是不可逆的。用于产生这些数据的变形的完整细节，如附录A所示。我们训练的基准全连接网络和CNN网络，以及在分类前对输入进行空间变换的网络(ST-FCN和ST-CNN)。Spatial transformer网络都使用双线性采样，但变体使用的是不同的变换函数：一个仿射变换(Aff)，投影变换(proj)，和一个16点的薄板样条变换(TPS)。CNN模型包含两个max-pooling层。所有网络的参数数量大致相同，都用相同的优化方案进行训练（反向传播，SGD，学习速率下降方案，多项式交叉熵损失），在分类网络中都带有3个权重层。

The results of these experiments are shown in Table 1 (left). Looking at any particular type of distortion of the data, it is clear that a spatial transformer enabled network outperforms its counterpart base network. For the case of rotation, translation, and scale distortion (RTS), the ST-CNN achieves 0.5% and 0.6% depending on the class of transform used for $T_θ$, whereas a CNN, with two max-pooling layers to provide spatial invariance, achieves 0.8% error. This is in fact the same error that the ST-FCN achieves, which is without a single convolution or max-pooling layer in its network, showing that using a spatial transformer is an alternative way to achieve spatial invariance. ST-CNN models consistently perform better than ST-FCN models due to max-pooling layers in ST-CNN providing even more spatial invariance, and convolutional layers better modelling local structure. We also test our models in a noisy environment, on 60 × 60 images with translated MNIST digits and background clutter (see Fig. 1 third row for an example): an FCN gets 13.2% error, a CNN gets 3.5% error, while an ST-FCN gets 2.0% error and an ST-CNN gets 1.7% error.

这些试验的结果，如表1左所示。观察任意特定类型的数据形变，很明显，带有spatial transformer的网络超过了其对应的基准网络。对于旋转、平移和缩放的形变(RTS)，ST-CNN取得了0.5%和0.6%的结果，结果依赖于用于$T_θ$的变换类型，而带有2个max-pooling层的CNN得到的错误率为0.8%。这实际上是ST-FCN的错误率，在其网络中没有卷积或max-pooling层，因为ST-CNN中的max-pooling层会提供更多的不变性，而卷积层更好的对局部结构进行建模。我们还在一个含噪的环境中测试了我们的模型，在60 × 60图像中，有平移的MNIST数字，背景有群聚（见图1中的第三行为例子）：FCN得到的13.2%的错误率，CNN错误率为3.5%，而ST-FCN错误率2.0%，ST-CNN的错误率为1.7%。

Looking at the results between different classes of transformation, the thin plate spline transformation (TPS) is the most powerful, being able to reduce error on elastically deformed digits by reshaping the input into a prototype instance of the digit, reducing the complexity of the task for the classification network, and does not over fit on simpler data e.g. R. Interestingly, the transformation of inputs for all ST models leads to a “standard” upright posed digit – this is the mean pose found in the training data. In Table 1 (right), we show the transformations performed for some test cases where a CNN is unable to correctly classify the digit, but a spatial transformer network can. Further test examples are visualised in an animation here https://goo.gl/qdEhUu.

观察不同类别变换之间的结果，薄板样条变换(TPS)是最强的，能够在弹性变形的数字上降低错误率，将输入变形到数字的原始样子，降低了分类网络任务的难度，在更简单的数据上也不会过拟合，如R类型的变换。有趣的是，对于ST模型，输入的变换带来了一种标准的竖立的数字，这是训练数据中的平均姿态。在表1右中，我们展示了在一些测试案例上进行的变换，而CNN是无法对这些数字进行正确分类的，但spatial transformer网络是可以的。更多测试样本可以见https://goo.gl/qdEhUu。

Table 1: Left: The percentage errors for different models on different distorted MNIST datasets. The different distorted MNIST datasets we test are TC: translated and cluttered, R: rotated, RTS: rotated, translated, and scaled, P: projective distortion, E: elastic distortion. All the models used for each experiment have the same number of parameters, and same base structure for all experiments. Right: Some example test images where a spatial transformer network correctly classifies the digit but a CNN fails. (a) The inputs to the networks. (b) The transformations predicted by the spatial transformers, visualised by the grid Tθ (G). (c) The outputs of the spatial transformers. E and RTS examples use thin plate spline spatial transformers (ST-CNN TPS), while R examples use affine spatial transformers (ST-CNN Aff) with the angles of the affine transformations given. For videos showing animations of these experiments and more see https://goo.gl/qdEhUu.

### 4.2 Street View House Numbers

We now test our spatial transformer networks on a challenging real-world dataset, Street View House Numbers (SVHN) [25]. This dataset contains around 200k real world images of house numbers, with the task to recognise the sequence of numbers in each image. There are between 1 and 5 digits in each image, with a large variability in scale and spatial arrangement.

我们现在在一个很有挑战性的真实世界数据集上测试我们的spatial transformer网络，街景房屋号(SVHN)。这个数据集包含大约200k真实世界的房屋号图像，任务是要识别每幅图像中的数字序列。每幅图像中有1到5个数字，其尺度和空间布局都有很大的变化。

We follow the experimental setup as in [1, 13], where the data is preprocessed by taking 64 × 64 crops around each digit sequence. We also use an additional more loosely 128×128 cropped dataset as in [1]. We train a baseline character sequence CNN model with 11 hidden layers leading to five independent softmax classifiers, each one predicting the digit at a particular position in the sequence. This is the character sequence model used in [19], where each classifier includes a null-character output to model variable length sequences. This model matches the results obtained in [13].

我们与[1,13]中的试验设置一样，其中数据的预处理是在每个数字序列附近进行64 × 64的剪切。我们还使用了额外更宽松的128×128剪切的数据集，和[1]中一样。我们训练的一个基准数字序列CNN模型，有11个隐含层，带来5个独立的softmax分类器，每个在一个特定的位置预测序列中的数字。这是[19]中使用的字符串模型，每个分类器包含一个null字符输出，以对可变长度的序列进行建模。这个模型与[13]中得到的结果相匹配。

We extend this baseline CNN to include a spatial transformer immediately following the input (ST-CNN Single), where the localisation network is a four-layer CNN. We also define another extension where before each of the first four convolutional layers of the baseline CNN, we insert a spatial transformer (ST-CNN Multi), where the localisation networks are all two layer fully connected networks with 32 units per layer. In the ST-CNN Multi model, the spatial transformer before the first convolutional layer acts on the input image as with the previous experiments, however the subsequent spatial transformers deeper in the network act on the convolutional feature maps, predicting a transformation from them and transforming these feature maps (this is visualised in Table 2 (right) (a)). This allows deeper spatial transformers to predict a transformation based on richer features rather than the raw image. All networks are trained from scratch with SGD and dropout [17], with randomly initialised weights, except for the regression layers of spatial transformers which are initialised to predict the identity transform. Affine transformations and bilinear sampling kernels are used for all spatial transformer networks in these experiments.

我们对这个基准CNN进行拓展，以在输入后就包含一个spatial transformer(ST-CNN Single)，其中定位网络是一个4层CNN。我们还定义了另一种拓展，在基准CNN的每个卷积层之前，我们插入一个spatial transformer(ST-CNN Multi)，其中定位网络都是两层的全连接网络，每层32个单元。在ST-CNN Multi模型中，在第一个卷积层之前的spatial transformer对输入图像进行处理，与前一个试验一样，但是在网络更深处的后续的spatial transformer都处理的是卷积特征图，从其预测一个变换，并对这些特征图进行变换（这在表2右有可视化）。这使得更深的spatial transformer可以基于更丰富的特征预测到一个变换，而不是原始图像。所有网络都是从头用SGD和dropout训练的，权重随机初始化，除了spatial transformer的回归层，其初始化是用于预测恒等变换的。在这些试验中，所有spatial transformer网络都使用了仿射变换和双线性采样核。

The results of this experiment are shown in Table 2 (left) – the spatial transformer models obtain state-of-the-art results, reaching 3.6% error on 64 × 64 images compared to previous state-of-the-art of 3.9% error. Interestingly on 128 × 128 images, while other methods degrade in performance, an ST-CNN achieves 3.9% error while the previous state of the art at 4.5% error is with a recurrent attention model that uses an ensemble of models with Monte Carlo averaging – in contrast the ST- CNN models require only a single forward pass of a single model. This accuracy is achieved due to the fact that the spatial transformers crop and rescale the parts of the feature maps that correspond to the digit, focussing resolution and network capacity only on these areas (see Table 2 (right) (b) for some examples). In terms of computation speed, the ST-CNN Multi model is only 6% slower (forward and backward pass) than the CNN.

试验结果如表2左所示，spatial transformer模型得到了目前最好的结果，在64 × 64的图像中得到了3.6%的错误率，而之前最好的结果是3.9%。在128 × 128图像中，其他方法性能下降了，而ST-CNN得到了3.9%的错误率，而之前最好的模型是4.5%的错误率，是一个循环注意力模型，使用的是模型集成，并用Monte Carlo进行了平均，比较起来，ST-CNN模型只需要一个模型的一个前向过程。这种准确率的取得，是因为spatial transformer对特征图的部分进行了剪切和改变大小，而这部分对应着数字。分辨率和网络容量都聚焦在这些区域中（见表2右的一些例子）。以计算速度来说，ST-CNN Multi模型只比CNN模型慢了6%。

Table 2: Left: The sequence error for SVHN multi-digit recognition on crops of 64 × 64 pixels (64px), and inflated crops of 128 × 128 (128px) which include more background. 

### 4.3 Fine-Grained Classification

In this section, we use a spatial transformer network with multiple transformers in parallel to perform fine-grained bird classification. We evaluate our models on the CUB-200-2011 birds dataset [38], containing 6k training images and 5.8k test images, covering 200 species of birds. The birds appear at a range of scales and orientations, are not tightly cropped, and require detailed texture and shape analysis to distinguish. In our experiments, we only use image class labels for training.

本节中，我们使用了一个spatial transformer网络，有多个并行的transformer，进行细粒度鸟的分类。我们在CUB-200-2011数据集上评估我们的模型，数据集包含6k训练图像，5.8k测试图像，覆盖鸟类的200个种类。这些会在一定尺度和方向范围内出现，并不是紧贴着剪切的，需要细节纹理和形状分析来区分。在我们的试验中，我们只使用图像类别标签进行训练。

We consider a strong baseline CNN model – an Inception architecture with batch normalisation [18] pre-trained on ImageNet [26] and fine-tuned on CUB – which by itself achieves the state-of-the-art accuracy of 82.3% (previous best result is 81.0% [30]). We then train a spatial transformer network, ST-CNN, which contains 2 or 4 parallel spatial transformers, parameterised for attention and acting on the input image. Discriminative image parts, captured by the transformers, are passed to the part description sub-nets (each of which is also initialised by Inception). The resulting part representations are concatenated and classified with a single softmax layer. The whole architecture is trained on image class labels end-to-end with backpropagation (full details in Appendix A).

我们考虑一个更强的基准CNN模型，带有批归一化的Inception架构，在ImageNet上进行预训练，在CUB上进行精调，得到了目前最好的准确率82.3%（之前最好的结果是81.0%）。我们然后训练一个spatial transformer网络，ST-CNN，包含2或4个并行的spatial transformer，对输入图像进行处理。可区分的图像部分，被transformer捕获得到，送入部位描述子网络（其中每个都是由Inception网络初始化的）。得到的部位表示拼接到一起，由单个softmax层进行分类。整个架构都是用图像类别标签，用反向传播进行的端到端训练。

The results are shown in Table 3 (left). The ST-CNN achieves an accuracy of 84.1%, outperforming the baseline by 1.8%. It should be noted that there is a small (22/5794) overlap between the ImageNet training set and CUB-200-2011 test set – removing these images from the test set results in 84.0% accuracy with the same ST-CNN. In the visualisations of the transforms predicted by 2×ST- CNN (Table 3 (right)) one can see interesting behaviour has been learnt: one spatial transformer (red) has learnt to become a head detector, while the other (green) fixates on the central part of the body of a bird. The resulting output from the spatial transformers for the classification network is a somewhat pose-normalised representation of a bird. While previous work such as [3] explicitly define parts of the bird, training separate detectors for these parts with supplied keypoint training data, the ST-CNN is able to discover and learn part detectors in a data-driven manner without any additional supervision. In addition, the use of spatial transformers allows us to use 448px resolution input images without any impact in performance, as the output of the transformed 448px images are downsampled to 224px before being processed.

## 5 Conclusion

In this paper we introduced a new self-contained module for neural networks – the spatial transformer. This module can be dropped into a network and perform explicit spatial transformations of features, opening up new ways for neural networks to model data, and is learnt in an end-to-end fashion, without making any changes to the loss function. While CNNs provide an incredibly strong baseline, we see gains in accuracy using spatial transformers across multiple tasks, resulting in state-of-the-art performance. Furthermore, the regressed transformation parameters from the spatial transformer are available as an output and could be used for subsequent tasks. While we only explore feed-forward networks in this work, early experiments show spatial transformers to be powerful in recurrent models, and useful for tasks requiring the disentangling of object reference frames, as well as easily extendable to 3D transformations (see Appendix A.3).