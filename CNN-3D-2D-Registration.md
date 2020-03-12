# A CNN Regression Approach for Real-Time 2D/3D Registration

Shun Miao et al. University of British Columbia

## Abstract

In this paper, we present a Convolutional Neural Network (CNN) regression approach to address the two major limitations of existing intensity-based 2-D/3-D registration technology: 1) slow computation and 2) small capture range. Different from optimization-based methods, which iteratively optimize the transformation parameters over a scalar-valued metric function representing the quality of the registration, the proposed method exploits the information embedded in the appearances of the digitally reconstructed radiograph and X-ray images, and employs CNN regressors to directly estimate the transformation parameters. An automatic feature extraction step is introduced to calculate 3-D pose-indexed features that are sensitive to the variables to be regressed while robust to other factors. The CNN regressors are then trained for local zones and applied in a hierarchical manner to break down the complex regression task into multiple simpler sub-tasks that can be learned separately. Weight sharing is furthermore employed in the CNN regression model to reduce the memory footprint. The proposed approach has been quantitatively evaluated on 3 potential clinical applications, demonstrating its significant advantage in providing highly accurate real-time 2-D/3-D registration with a significantly enlarged capture range when compared to intensity-based methods.

本文中，我们提出了一种CNN回归方法，来解决现有基于灰度的2D-3D配准技术的两个主要局限：1)计算慢；2)捕获范围小。基于优化的配准方法，是有一个标量值的度量函数，表示配准的质量，对这个函数进行迭代优化，得到形变参数；提出的方法利用DRR和X射线图像的外观中包含的信息，采用CNN回归器来直接估计形变参数。我们引入了一种自动特征提取步骤，来计算3D pose-indexed特征，对要回归的变量很敏感，对其他要素则很稳健。CNN回归器然后对局部区域进行训练，以层次化的方式进行应用，将复杂的回归任务分解成为多个更简单的子任务，可以分别学习到。CNN回归模型还采取了权重共享，以减少内存占用。提出的方法在3个潜在的临床应用中进行了量化评估，与基于灰度的方法相比，可以进行高准确度的实时2D/3D配准，捕获范围明显增大，具有显著的优势。

**Index Terms** 2-D/3-D registration, convolutional neural network, deep learning, image guided intervention.

## I. Introduction

Two-dimensional to three-dimensional registration represents one of the key enabling technologies in medical imaging and image-guided interventions [1]. It can bring the pre-operative 3-D data and intra-operative 2-D data into the same coordinate system, to facilitate accurate diagnosis and/or provide advanced image guidance. The pre-operative 3-D data generally includes Computed Tomography (CT), Cone-beam CT (CBCT), Magnetic Resonance Imaging (MRI) and Computer Aided Design (CAD) model of medical devices, while the intra-operative 2-D data is dominantly X-ray images. In this paper, we focus on registering a 3-D X-ray attenuation map provided by CT or CBCT with a 2-D X-ray image in real-time. Depending on the application, other 3-D modalities (e.g., MRI and CAD model) can be converted to a 3-D X-ray attenuation map before performing 2-D/3-D registration.

2D-3D配准是医学成像和图像引导介入手术所需的的关键技术，可以将术前3D数据和术中2D数据放到一个坐标系中，可以促进准确的诊断以及高级图像引导。术前3D数据一般包括CT，CBCT，MRI和CAD模型，而术中2D数据主要是X射线图像。本文中，我们关注的是，将CT或CBCT提供的3D X射线衰减图，与2D X射线图像进行实时配准。按应用不同，其他3D模态（如，MRI和CAD模型），可以在进行2D/3D配准前转换为3D X射线衰减图。

In existing methods, accurate 2-D/3-D registration is typically achieved by intensity-based 2-D/3-D registration methods [2]–[5]. In these methods, a simulated X-ray image, referred to as Digitally Reconstructed Radiograph (DRR), is derived from the 3-D X-ray attenuation map by simulating the attenuation of virtual X-rays. An optimizer is employed to maximize an intensity-based similarity measure between the DRR and X-ray images. Intensity-based methods are known to be able to achieve high registration accuracy [6], but at the same time, they suffer from two major drawbacks: 1) long computation time and 2) small capture range. Specifically, because intensity-based methods involve a large number of evaluations of the similarity measure, each requiring heavy computation in rendering the DRR, they typically resulted in above 1s running time, and therefore are not suitable for real-time applications. In addition, because the similarity measures to be optimized in intensity-based methods are often highly non-convex, the optimizer has a high chance of getting trapped into local maxima, which leads to a small capture range of these methods.

现有的方法中，通常是用基于灰度的2D/3D配准方法[2-5]进行准确的配准。这些方法中，仿真的X射线图像，称之为DRR，是从3D X射线衰减图中计算得到的，对虚拟X射线的衰减进行仿真。采用一个优化器来最大化基于灰度的DRR与X 射线图像之间的相似性度量。基于灰度的方法可以得到很高的准确率，但同时有两个主要的缺点：1)计算时间很长；2)捕获的范围很小。具体的，因为基于灰度的方法涉及到很多对相似度度量的评估，每个都需要很大的计算量来得到DRR，通常都会超过1s的运行时间，因此不适用于实时的应用。另外，因为在基于灰度的方法中，要优化的相似性度量通常是高度非凸的，优化器很可能陷入局部极值中，导致这些方法的捕获范围都很小。

The small capture range of intensity-based methods is often addressed by employing initialization methods before registration [7], [8]. However, initialization methods typically utilize dominant features of the target object for pose recovery and therefore are very application specific. For example, Varnavas et al. [7] applied Generalized Hough Transform (GHT) for initial pose estimation of spine vertebrae. This method is specific to applications where spine vertebrae edges are clearly visible in both the X-ray and CT images. Miao et al. [8] proposed to use shape encoding combined with template matching for initial pose estimation. This method can only be applied on metal implants, which are highly X-ray opaque objects that can be reliably segmented from X-ray images for shape encoding.

基于灰度的方法的小捕获范围，其解决方法通常是在配准之前采用初始化方法[7,8]。但是，初始化方法，通常都会使用目标的主要特征进行姿态恢复，因此非常依赖于具体应用。比如，Varnavas等[7]使用通用Hough变换(GHT)对椎骨进行初始姿态估计。这种方法对于在CT和X射线图像中，脊椎的边缘都很清楚的应用，是很适用的。Miao等[8]提出使用形状编码与模板匹配进行初始姿态估计。这种方法只能用于金属植入物中，这对于X射线是高度不透明的，可以很可靠的从X射线图像中分割出来，进行形状编码。

Some efforts have been made toward accelerating DRR generation for fast 2D/3D registration. One strategy for faster DRR generation is sparse sampling, where a subset of the pixels are statistically chosen for DRR rendering and similarity measure calculation [9], [10]. However, only a few similarity measures are suitable to be calculated on a random subset of the image, e.g., Mutual Information (MI) [9] and Stochastic Rank Correlation (SRC) [10]. Another strategy is splatting, which is a voxel-based volume rendering technique that directly projects single voxels to the imaging plane [11], [12]. Splatting allows to only use voxels with intensity above certain threshold for rendering, which significantly reduces the number of voxels to be visited. However, one inherent problem of splatting is that due to aliasing artifacts, the image quality of the generated DRR is significantly degraded, which subsequently degrades the registration accuracy compared to the DRR generated by the standard Ray Casting algorithm [13].

对于快速2D/3D配准，有一些加速DRR生成的努力。更快的生成DRR的一种策略是稀疏采样，即选择一部分像素来渲染DRR，计算相似性度量[9,10]。但是，只有几种相似性度量适合于在图像的随机子集上进行计算，如，互信息(MI)[9]和随机阶相关(SRC)[10]。另一种策略是splatting，这是基于体素的体绘制技术，直接将单个体素投影到成像平面。Splatting可以只使用灰度大于某阈值的体素进行渲染，这显著降低了需要访问的体素数量。但是，splatting的一个内在问题是，由于存在混淆伪影，生成DRR的图像质量被显著降低质量，与使用标准的光线投射算法相比，然后会使配准准确率有所下降。

Supervised learning has also been explored for 2-D/3-D registration. Several metric learning methods have been proposed to learn similarity measures using supervised learning [14], [15]. While learned metrics could have better capture range and/or accuracy over general purpose similarity measures on specific applications or image modalities, 2-D/3-D registration methods using learned metrics still fall into the category of intensity-based methods with a high computational cost. As a new direction, several attempts have been made recently toward learning regressors to solve 2-D/3-D registration problems in real-time [16], [17]. Gouveia et al. [16] extracted a handcrafted feature from the X-ray image and trained a Multi-Layer Perceptron (MLP) regressor to estimate the 3-D transformation parameters. However, the reported accuracy is much lower than that can be achieved using intensity-based methods, suggesting that the handcrafted feature and MLP are unable to accurately recover the underlying complex transformation. Chou et al. [17] computed the residual between the DRR and X-ray images as a feature and trained linear regressors to estimate the transformation parameters to reduce the residual. Since the residual is a low-level feature, the mapping from it to the transformation parameters is highly non-linear, which cannot be reliably recovered using linear regressors, as will be shown in our experiment.

2D/3D配准曾使用监督学习。已经提出一些度量学习算法，使用监督学习来学习相似性度量。与通用目的的相似度度量相比，学习到的度量，在特定的应用或图像模态上，可能有更好的捕获范围，更好的准确率，但使用学习到的度量进行2D/3D配准，仍然会有基于灰度的方法的问题，即计算量很大。一个新的方向是，学习一个回归器来实时解决2D/3D配准问题。Gouveia等[16]从X射线图像中手动提取了特征，训练了一个MLP回归器来估计3D变换参数。但是，得到的准确率与基于灰度的方法相比要低的多，说明手工特征和MLP不能准确的恢复潜在的复杂变换。Chou等[17]计算了DRR和X射线之间的残差作为特征，训练了一个线性回归器来估计变换参数，以reduce这个残差。由于残差是一个底层特征，从其到形变参数的映射是高度非线性的，使用线性回归不能准确的得到恢复，后面我们的试验会展示这个结论。

In recent years, promising results on object matching for computer vision tasks have been reported using machine learning methods [18]–[21]. While these methods are capable of reliably recovering the object's location and/or pose for computer vision tasks, they are unable to meet the accuracy requirement of 2-D/3-D registration tasks in medical imaging, which often target at a very high accuracy (i.e., sub-millimeter) for diagnosis and surgery guidance purposes. For example, Wohlhart et al. [18] proposed to train a Convolutional Neural Networks (CNN) to learn a pose differentiating descriptor from range images, and use k-Nearest Neighbor for pose estimation. While global pose estimation can be achieved using this method, its accuracy is relatively low, i.e., the success rate for angle error less than 5 degrees is below 60% for k=1. Dollár et al. [19] proposed to train cascaded regressors on a pose-indexed feature that is only affected by the difference between the ground truth and initial pose parameters for pose estimation. This method solves 2-D pose estimation from RGB images with hundreds of iterations, and therefore are not applicable for 2-D/3-D registration problems with a real-time (e.g., a few updates) requirement.

最近几年，计算机视觉中的目标匹配已经有了一些很有希望的结果，使用的是机器学习的方法。这些方法可以在计算机视觉任务中可靠的恢复目标的位置及姿态，但他们无法满足医学成像中2D/3D配准任务的精度需要，因为一般诊断和手术引导的精度要求是非常高的（如，亚毫米）。比如，Wohlhart等[18]提出训练CNN来从range图像中学习一个姿态区分描述子，然后使用k近邻进行姿态估计。虽然全局姿态估计可以用这种方法进行，但其精度相对较低，即，k=1时角度误差小于5度的成功率低于60%。Dollár等[19]提出在pose-indexed特征上训练一个级联的回归器，姿态估计只受真值和初始姿态参数的差异影像。这种方法从RGB图像中，采用数百个迭代的计算，进行2D姿态估计，因此不适用于2D/3D配准问题，尤其是还要满足实时的要求。

In this paper, a CNN regression approach, referred to as Pose Estimation via Hierarchical Learning (PEHL), is proposed to achieve real-time 2-D/3-D registration with a large capture range and high accuracy. The key of our approach is to train CNN regressors to recover the mapping from the DRR and X-ray images to the difference of their underlying transformation parameters. Such mapping is highly complex and training regressors to recover the mapping is far from being trivial. In the proposed method, we achieve this by first simplifying the non-linear relationship using the following three algorithmic strategies and then capturing the mapping using CNN regressors with a strong non-linear modeling capability.

本文中，我们提出了一种CNN回归方法，称为层次化学习的姿态估计(PEHL)，可以实时的进行2D/3D配准，捕获范围大，准确率高。我们方法的关键是训练了一个CNN回归器，将DRR到X射线图像的映射恢复为其变换参数的差异。这种映射是非常复杂的，训练回归器来恢复其映射不是那么简单的工作。在提出的方法中，我们首先用下面三种算法策略来简化非线性关系，然后使用具有很强的非线性建模能力的CNN回归器来捕获这个映射。

- Local image residual (LIR): To simplify the underlying mapping to be captured by the regressors, we introduce an LIR feature for regression, which is approximately 3-D pose-indexed, i.e., it is only affected by the difference between the initial and ground truth transformation parameters.

- 局部图像残差(LIR)：为简化潜在的映射，以被回归器捕获，我们提出了LIR特征进行回归，这个特征是近似3D pose-indexed，即只受初始变换参数和真值变换参数的差异的影像。

- Parameter space partitioning (PSP): We partition the transformation parameter space into zones and train CNN regressors in each zone separately, to break down the complex regression task into multiple simpler sub-tasks.

- 参数空间分割(PSP)：我们将变换参数空间分割成几个区域，在每个区域中分别训练CNN回归器，以将复杂的回归任务分解成多个更简单的子任务。

- Hierarchical parameter regression (HPR): We decompose the transformation parameters and regress them in a hierarchical manner, to achieve highly accurate parameter estimation in each step.

- 层次化参数回归(HPR)：我们将变换参数分解，并对其进行层次化回归，在每个步骤中进行高精确度的参数估计。

The remainder of this paper is organized as follows. Section II provides backgrounds of 2-D/3-D registration, and formulates it as a regression problem. Section III presents the proposed PEHL approach. Validation datasets, evaluation metrics and experiment configurations are described in Section IV, and experimental results are shown in Section V. Section VI concludes the paper with a discussion of our findings.

本文的剩下部分组织如下。第II部分给出2D/3D配准的背景，将其表述为一个回归问题。第III部分给出提出的PEHL方法。第IV部分描述了验证集，评估度量标准和试验配置，第V部分给出试验结果。第VI部分进行讨论并总结本文。

## II. Problem Formulation

### A. X-Ray Imaging Model

Assuming that the X-ray imaging system corrects the beam divergence and the X-ray sensor has a logarithm static response, X-ray image generation can be described by the following model:

假设X射线成像系统修正了束流的发散，X射线传感器的响应是log静态的，X射线图像生成可以用下面的模型来描述：

$$I(p) = \int μ(L(p,r))dr$$(1)

Where I(p) is the intensity of the X-ray image at point p, L(p,r) is the ray from the X-ray source to point p, parameterized by r, and μ(⋅) is the X-ray attenuation coefficient. Denoting the X-ray attenuation map of the object to be imaged as $J:R^3->R$, and the 3-D transformation from the object coordinate system to the X-ray imaging coordinate system as $T:R^3->R^3$, the attenuation coefficient at point x in the X-ray imaging coordinate system is

其中I(p)是X射线图像在p点的灰度，L(p,r)是从X射线源到点p的射线，参数为r，μ(⋅)是X射线衰减系数。将目标的X射线衰减图表示为$J:R^3->R$，从目标坐标系到X射线成像坐标系的3D变换表示为$T:R^3->R^3$，那么在X射线坐标系中点x的衰减系数为

$$μ(x) = J(T^{-1}⋅x)$$(2)

Combining (1) and (2), we have

$$I(p) = \int J(T^{-1}⋅L(p,r)) dr$$(3)

In 2-D/3-D registration problems, L is determined by the X-ray imaging system, J is provided by the 3-D data (e.g., CT intensity), and the transformation T is to be estimated from the input X-ray image I. Note that given J, L and T, a synthetic X-ray image I(⋅) can be computed following (3) using Ray-Casting algorithm [13], and the generated image is referred to as DRR.

在2D/3D配准问题中，L是由X射线成像系统决定的，J是由3D数据提供的（如，CT灰度），而且变换T是要从输入X射线图像I中估计得到的。注意给定J，L和T，用(3)和光线投射算法可以合成一幅X射线图像I(⋅)，生成的图像称为DRR。

### B. 3-D Transformation Parameterization

A rigid-body 3-D transformation T can be parameterized by a vector t with 6 components. In our approach, we parameterize the transformation by 3 in-plane and 3 out-of-plane transformation parameters [22], as shown in Fig. 1. In particular, in-plane transformation parameters include 2 translation parameters, $t_x$ and $t_y$, and 1 rotation parameter, $t_θ$. The effects of in-plane transformation parameters are approximately 2-D rigid-body transformations. Out-of-plane transformation parameters include 1 out-of-plane translation parameter, $t_z$, and 2 out-of-plane rotation parameters, $t_α$ and $t_β$. The effects of out-of-plane translation and rotations are scaling and shape changes, respectively.

刚体3D变换T可以用一个6维向量参数表示。在我们的方法中，我们将变换参数化为3个平面内参数，3个平面外参数，如图1所示。特别的，平面内变换参数包含2个平移参数$t_x$, $t_y$，和一个旋转参数$t_θ$。平面内变换参数的作用是近似2D刚体变换。平面外变换参数包括一个平面外平移参数$t_z$，和两个平面外旋转参数$t_α$和$t_β$。平面外平移和旋转分别是缩放和形状变化。

Fig. 1. Effects of the 6 transformation parameters.

### C. 2-D/3-D Registration via Regression

Based on (3), we denote the X-ray image with transformation parameters t as $I_t$, where the variables L and J are omitted for simplicity because they are non-varying for a given 2-D/3-D registration task. The inputs for 2-D/3-D registration are: 1) a 3-D object described by its X-ray attenuation map J, 2) an X-ray image $I_{t_{gt}}$, where $t_{gt}$ denotes the unknown ground truth transformation parameters, and 3) initial transformation parameters $t_{ini}$. The 2-D/3-D registration problem can be formulated as a regression problem, where a set of regressors f(⋅) are trained to reveal the mapping from a feature $X(t_{ini}, I_{t_{gt}})$ extracted from the inputs to the parameter residuals, $t_{gt} - t_{ini}$, as long as it is within a capture range $ϵ$:

基于(3)式，我们将X射线图像用变换参数t表示为$I_t$，简化起见，忽略掉其中的变量L和J，因为对于给定的2D/3D配准任务来说，是不变的。2D/3D配准的输入为：1)X射线衰减图J描述的3D目标，2)X射线图像$I_{t_{gt}}$，其中$t_{gt}$表示未知的真值变换参数，3)初始的变换参数$t_{ini}$。2D/3D配准问题可以表示为一个回归问题，训练一族回归器来揭示从输入提取的特征$X(t_{ini}, I_{t_{gt}})$到参数残差$t_{gt} - t_{ini}$的映射，只要是在一定捕获范围$ϵ$内就可以：

$$t_{gt} - t_{ini} ≈ f(X(t_{ini}, I_{t_{gt}})), ∀ t_{gt} - t_{ini} ∈ ϵ$$(4)

An estimation of $t_{gt}$ is then obtained by applying the regressors and incorporating the estimated parameter residuals into $t_{ini}$:

$t_{gt}$的估计，通过使用回归器，然后将估计的参数残差与$t_{ini}$一起计算：

$$\hat t_{gt} = t_{ini} + f(X(t_{ini},I_{t_{gt}}))$$(5)

It is worth noting that the range ϵ in (4) is equivalent to the capture range of optimization-based registration methods. Based on (4), our problem formulation can be expressed as designing a feature extractor X(⋅) and training regressors f(⋅), such that

值得注意的是，(4)中的范围ϵ等价于基于优化的配准方法的捕获范围。基于(4)，我们的问题描述可以表述为，设计一个特征提取器X(⋅)，并训练回归器f(⋅)，以使得

$$δt ≈ f(X(t, I_{t+δt})), ∀δt ∈ ϵ$$(6)

In the next section, we will discuss in detail 1) how the feature $X(t, I_{t+δt})$ is calculated and 2) how the regressors f(⋅) are designed, trained and applied.

下一节，我会详细讨论 1)特征$X(t, I_{t+δt})$是怎么计算的，和2)回归器f(⋅)是怎样设计、训练和应用的。

## III. Pose Estimation Via Hierarchical Learning 通过层次化学习进行姿态估计

### A. Parameter Space Partitioning 参数空间分割

We aim at training regressors to recover the mapping from the feature $X(t, I_{t+δt})$ to the parameter residuals δt. Since the feature naturally depends on t, the target mapping could vary significantly as t changes, which makes it highly complex and difficult to be accurately recovered. Ideally, we would like to extract a feature that is sensitive to the parameter residuals δt, and is insensitive to the parameters t. Such feature is referred to as pose-index feature, and the property can be expressed as:

我们的目标是训练回归器，以得到从特征$X(t, I_{t+δt})$到参数残差δt的映射。由于特征很自然的依赖于t的，目标映射会随着t的变化有很大变化，这使得要准确的恢复就非常复杂和困难。理想情况中，我们提取出的特征应当对参数残差δt敏感，而对参数t不敏感。这样的特征称为pose-index的特征，其性质可以描述为：

$$X(t_1, I_{t_1+δt}) ≈ X(t_2, I_{t_2+δt}), ∀(t_1, t_2)$$(7)

As we will show in Section III-B, we use ROIs to make $X(t, I_{t+δt})$ invariant to the in-plane and scaling parameters, $(t_x, t_y, t_z, t_θ)$. However, we are unable to make $X(t, I_{t+δt})$ insensitive to $t_α$ and $t_β$, because they cause complex appearance changes in the projection image. To solve this problem, we partition the parameter space spanned by $t_α$ and $t_β$ into a 18✖18 grid (empirically selected in our experiment). Each square in the grid covers a 20✖20 degrees area, and is referred to as a zone. We will show in Section III-B that for ($t_α$, $t_β$) within each zone, our LIR feature is approximately pose-indexed, i.e.,

我们在III-B节中会展示，我们使用ROIs来使得$X(t, I_{t+δt})$对平面内和缩放参数$(t_x, t_y, t_z, t_θ)$不变。但是，我们不能使$X(t, I_{t+δt})$对$t_α$和$t_β$不敏感，因为他们会在投影图像中带来复杂的形状改变。为解决这个问题，我们将$t_α$和$t_β$张成的参数空间分割成一个18✖18的网格（在我们的试验中是根据经验选择的）。网格中每个方块覆盖20✖20度的区域，并称为一个区域。我们会在III-B中展示，对于每个区域中的($t_α$, $t_β$)，我们的LIR特征是近似pose-indexed，即

$$X_k(t_1, I_{t_1+δt}) ≈ X_k(t_2, I_{t_2+δt}), ∀(t_1, t_2)∈Ω_k$$(8)

where $X_k(⋅, ⋅)$ denotes the LIR feature extractor for the k-th zone, and $Ω_k$ denotes the area covered by the k-th zone. The regressors therefore are trained separately for each zone to recover the simplified mapping that is insensitive to t.

这里$X_k(⋅, ⋅)$表示从第k个区域的LIR特征提取器，$Ω_k$表示第k个区域覆盖的区域。因此对每个区域分别训练回归器，以恢复出对t不敏感的简化的映射。

### B. Local Image Residual

**1) Calculation of Local Image Residual**: The LIR feature is calculated as the difference between the DRR rendered using transformation parameters t, denoted by $I_t$, and the X-ray image $I_{t+δt}$ in local patches. To determine the locations, sizes and orientations of the local patches, a number of 3-D points are extracted from the 3-D model of the target object following the steps described in Section III-B2. Given a 3-D point p and parameters t, a square local ROI is uniquely determined in the 2-D imaging plane, which can be described by a triplet, $(q,w,ϕ)$, denoting the ROI's center, width and orientation, respectively. The center q is the 2-D projection of p using transformation parameters t. The width $w = w_0⋅D/t_z$, where $w_0$ is the size of the ROI in mm and D is the distance between the X-ray source and detector. The orientation $ϕ=t_θ$, so that it is always aligned with the object. We define an operator $H_p^t(⋅)$ that extracts the image patch in the ROI determined by p and t, and re-sample it to a fixed size (52✖52 in our applications). Given 3-D points, P={$p_1,...,p_N$}, the LIR feature is then computed as

**1)局部图像残差的计算**：LIR特征计算的是局部图像块中的X射线图像$I_{t+δt}$和变换参数t渲染的DRR的差值，DRR表示为$I_t$。为确定局部图像块的位置、大小和方向，从目标物体的3D模型中提取出一些3D点，具体步骤如III-B2所述。给定一个3D点p和参数t，在2D成像平面可以确定唯一一个正方形局部ROI，可以用一个三元组$(q,w,ϕ)$描述，分别表示ROI的中心、宽度和方向。中心q是p点用变换参数t得到的2D投影。宽度$w = w_0⋅D/t_z$，其中$w_0$是ROI的大小，以mm为单位，D是X射线源和探测器之间的距离。方向$ϕ=t_θ$，这样就会和目标对齐了。我们定义一个算子$H_p^t(⋅)$，从ROI中用p和t提取图像块，并重新采样到固定大小（在我们的应用中是52✖52）。给定3D点P={$p_1,...,p_N$}，LIR特征计算为

$$X(t, I_{t+δt}, P) = {H_{p_i}^t(I_t) - H_{p_i}^t(I_{t+δt})}_{i=1,...,N}$$(9)

In a local area of $I_t$, the effect of varying $t_α$ and $t_β$ within a zone is approximately 2-D translation. Therefore, by extracting local patches from ROIs selected based on t, the effects of all 6 transformation parameters in t are compensated, making $H_p^t(I_t)$ approximately invariant to t. Since the difference between $H_p^t(I_{t+δt})$ and $H_p^t(I_t)$ is merely additional 2-D transformation caused by $δt$, $H_p^t(I_{t+δt})$ is also approximately invariant to t. The workflow of LIR feature extraction is shown in Fig. 2.

在$I_t$的一个局部区域，在一个区域内改变$t_α$和$t_β$的效果，近似的是2D平移。因此，基于t选择ROIs，提取局部图像块，t中所有6个变换参数的效果得到了补偿，使$H_p^t(I_t)$对t是近似不变的。由于$H_p^t(I_{t+δt})$和$H_p^t(I_t)$的差值，仅仅是由于$δt$造成的额外的2D变换，$H_p^t(I_{t+δt})$对于t也是近似不变的。LIR特征提取的工作流如图2所示。

Fig. 2. Workflow of LIR feature extraction, demonstrated on X-ray Echo Fusion data. The local ROIs determined by the 3-D points and the transformation parameters are shown as red boxes. The blue box shows a large ROI that covers the entire object, used in compared methods as will be discussed in Section IV-D.

**2) Extraction of 3-D Points**: The 3-D points used for calculating the LIR feature are extracted separately for each zone in two steps. First, 3-D points that correspond to 2-D edges are extracted as candidates. Specifically, the candidates are extracted by thresholding pixels with high gradient magnitudes in a synthetic X-ray image (i.e., generated using DRR) with $t_α$ and $t_β$ at the center of the zone, and then back-projecting them to the corresponding 3-D structures. The formation model of gradients in X-ray images has been shown in [23] as:

**2) 3D点的提取**：用于计算LIR特征的3D点的提取，是在每个区域中，用两步分别提取的。首先，提取与2D边缘对应的3D点作为候选。具体的，候选的提取，是对合成X射线图像(DRR)中高梯度值的像素采用阈值选取出来候选，DRR的生成中$t_α$和$t_β$是在区域中心，然后反投影到对应的3D结构中。X射线图像中的模型梯度如[23]中所示：

$$g(p) = \int η(L(p,r))dr$$(10)

where g(p) is the magnitute of the X-ray image gradient at the point p, and $η(⋅)$ can be computed from $μ(⋅)$ and the X-ray perspective geometry[23]. We back-project p to $L(p,r_0)$, where

其中，g(p)是X射线图像梯度在p点的幅度，$η(⋅)$可以从$μ(⋅)$和X射线透视图几何中计算出来。我们将p点反投影到$L(p,r_0)$，其中

$$r_0  = argmax_r L(p,r)$$(11)

if

$$\int_{r_0-σ}^{r_0+σ} η(L(p,r)) dr ≥ 0.9⋅g(p)$$(12)

The condition in (12) ensures that the 3-D structure around $L(p,r_0)$ “essentially generates” the 2-D gradient g(p), because the contribution of η(⋅) within a small neighborhood (i.e., σ=2mm) of $L(p,r_0)$ leads to the majority (i.e., ≥90%) of the magnitude of g(p). In other words, we find the dominant 3-D structure corresponding to the gradient in the X-ray image.

(12)中的条件确保了$L(p,r_0)$附近的3D结构“确实生成了”2D梯度g(p)，因为η(⋅)在$L(p,r_0)$中一个很小的邻域(σ=2mm)中贡献了g(p)幅度的大部分（即≥90%）。换句话说，我们找到了X射线图像中梯度对应的主要3D结构。

Second, the candidates are filtered so that only the ones leading to LIR satisfying (7) and also not significantly over-
lapped are kept. To achieve this, we randomly generate $\{ t_j \}_{j=1}^M$ with $t_α$ and $t_β$ within the zone and $\{ δt_k \}_{k=1}^M$ within the capture range ϵ (M=1000 in our applications). The intensity of the n-th pixel of $H_{p_i}^{t_j}(I_{t_j}) - H_{p_i}^{t_j}(I_{t_j+δt_k})$ is denoted as $h_{n,i,j,k}$. The following two measurements are computed for all candidates:

第二，候选要经过筛选，这样只保留那些满足(7)式LIR的，同时没有显著重叠的。为达到这个目标，我们随机生成$\{ t_j \}_{j=1}^M$，$t_α$和$t_β$在区域之中，$\{ δt_k \}_{k=1}^M$在捕获范围ϵ中（在我们的应用中，M=1000）。$H_{p_i}^{t_j}(I_{t_j}) - H_{p_i}^{t_j}(I_{t_j+δt_k})$的第n个像素的灰度，表示为$h_{n,i,j,k}$。对所有候选，计算下面的两个度量

$$E_i = <(h_{n,i,j,k} - <h_{n,i,j,k}>_j)^2>_{n,j,k}$$(13)

$$F_i = <(h_{n,i,j,k} - <h_{n,i,j,k}>_k)^2>_{n,j,k}$$(14)

where <⋅> is an average operator with respect to all indexes in the subscript. Since $E_i$ and $F_i$ measure the sensitivity of $H_{p_i}^t(I_t) - H_{p_i}^t(I_{t+δt})$ with respect to t and δt, respectively, an ideal LIR should have a small $E_i$ to satisfy (7) and a large $F_i$ for regressing δt. Therefore, the candidate list is filtered by picking the candidate with the largest $F_i/E_i$ in the list, and then removing other candidates with ROIs that have more than 25% overlapping area. This process repeats until the list is empty.

其中<⋅>是一个对下标中的所有索引计算的平均算子。由于$E_i$和$F_i$度量的分别是$H_{p_i}^t(I_t) - H_{p_i}^t(I_{t+δt})$对t和δt的敏感度，理想的LIR应当$E_i$很小并满足(7)，但$F_i$很大，以回归得到δt。因此，候选列表的筛选，是要选择列表中$F_i/E_i$最大的候选，然后去掉与之重叠超过25%的其他候选ROIs。这个过程进行重复，直到列表清空。

### C. Hierarchical Parameter Regression

Instead of regressing the 6 parameters together, which makes the mapping to be regressed more complex as multiple confounding factors are involved, we divide them into the following 3 groups, and regress them hierarchically:

一起回归6个参数会使得这个映射非常复杂，因为会有多个复杂的参数混杂到一起，所以我们将参数分成了3组，层次化的对其进行回归：

- Group 1: In-plane parameters: $δt_x, δt_y, δt_θ$
- Group 2: Out-of-plane rotation parameters: $δt_α, δt_β$,
- Group 3: Out-of-plane translation parameter: $δt_z$

Among the 3 groups, the parameters in Group 1 are considered to be the easiest to be estimated, because they cause simple while dominant rigid-body 2-D transformation of the object in the projection image that are less affected by the variations of the parameters in the other two groups. The parameter in Group 3 is the most difficult one to be estimated, because it only causes subtle scaling of the object in the projection image. The difficulty in estimating parameters in Group 2 falls in-between. Therefore we regress the 3 groups of parameters sequentially, from the easiest group to the most difficult one. After a group of parameters are regressed, the feature $X(t, I_{t+δt})$ is re-calculated using the already-estimated parameters for the regression of the parameters in the next group. This way the mapping to be regressed for each group is simplified by limiting the dimension and removing the compounding factors coming from those parameters in the previous groups.

在这三组中，组1中的参数应当是最容易估计的参数，因为这些参数在投影图像中，带来的是简单的主要是目标的刚体2D变换，与其他两组参数的变化相比，其影响更小。组3中的参数是最难估计的，因为在投影图像中只会带来目标的微小缩放。组2中参数估计的难度是在两者中间的。因此我们按顺序回归3组参数，从最简单的组，到最困难的组。在一组参数回归得到后，就用已经估计的参数，重新计算特征$X(t, I_{t+δt})$，用于下一组参数的回归。这种方法中，每一组中要回归的映射都进行了简化，限制了维数，去除了前一组参数带来的复杂因素。

The above HPR process can be repeated for a few iterations to achieve the optimal accuracy, and the result of the current iteration is used as the starting position for the next iteration (Algorithm 1). The number of iterations can be empirically selected (e.g., 3 times in our application), as will be described in Section V-A.

上述HPR过程可以重复进行，在几次迭代之后得到最佳准确率，当前迭代的结果，用于下一次迭代的开始位置（算法1）。迭代的次数可以通过经验进行选择（如在我们的应用中是3次），这将在V-A部分中进行描述。

### D. CNN Regression Model

In the proposed regression approach, challenges in designing the CNN regression model are two folds: 1) it needs to be flexible enough to capture the complex mapping from $X(t, I_{t+δt})$ to δt, and 2) it needs to be light-weighted enough to be forwarded in real-time and stored in Random-Access Memory (RAM). Managing memory footprint is particularly important because regressors for all zones (in total 324) need to be loaded to RAM for optimal speed. We employ the following CNN regression model to address these 2 challenges.

在提出的回归方法中，设计CNN回归模型的挑战是双重的：1)需要足够灵活，捕获从$X(t, I_{t+δt})$到δt的复杂映射；2)需要是足够轻量级的，以进行实时推理，存储在RAM中。管理内存占用是尤其重要的，因为对所有区域（总计324个）的回归器需要装载到RAM中，以得到最优速度。我们采用下面的CNN回归模型，以处理这两个挑战。

**1) Network Structure**: A CNN [24] regression model with the architecture shown in Fig. 3 is trained for each group in each zone. According to (9), the input of the regression model consists of N channels, corresponding to N LIRs. The CNN shown in Fig. 4 is applied on each channel for feature extraction. The CNN consists of five layers, including two 5 × 5 convolutional layers (C1 and C2), each followed by a 2 × 2 max-pooling layers (P1 and P2) with stride 2, and a fully-connected layer (F1) with 100 Rectified Linear Unit (ReLU) activations neurons. The feature vectors extracted from all input channels are then concatenated and connected to another fully-connected layer (F2) with 250 ReLU activations neurons. The output layer (F3) is fully-connected to F2, with each output node corresponding to one parameter in the group. Since the N input channels have the same nature, i.e., they are LIRs at different locations, the weights in the N CNNs are shared to reduce the memory footprint by N times.

**1)网络结构**：CNN回归模型架构如图3所示，对每个区域的每组参数都训练一个模型。根据(9)，回归模型的输入包含N个通道，对应着N个LIRs。图4中的CNN应用于每个通道进行特征提取。CNN包含5层，包含两个5×5卷积层(C1和C2)，每个都跟随着一个2×2最大池化层(P1和P2)，步长为2，以及一个全连接层(F1)，有100个ReLU激活神经元。从所有输入通道中提取出的特征向量然后拼接在一起，连接到另一个全连接层(F2)，有250个ReLU激活的神经元。输出层F3与F2是全连接的，每个输出节点对应着组中的一个参数。N个输入通道其本质都是一样的，即是在不同位置上的LIRs，所以N个CNNs的权重是共享的，以降低内存占用N倍。

In our experiment, we empirically selected the size of the ROI, which led to N ≈ 18. Using the CNN model shown in Fig. 3 with weight sharing, there are in total 660,500 weights for each group in each zone, excluding the output layer, which only has 250×$N_t$ weights, where $N_t$ is the number of parameters in the group. If the weights are stored as 32-bit float, around 2.5 MB is required for each group in each zone. Given 3 groups and 324 zones, there are in total 972 CNN regression models and pre-loading all of them into RAM requires 2.39 GB, which is manageable for modern computers.

我们的试验中，我们通过经验来选取ROI的大小，得到了N ≈ 18。使用图3中的CNN模型，进行了权重共享，对每个区域的每一组，共有660500个权重，输出层只有250×$N_t$个权重，没有计算在内，其中$N_t$是组中的参数数量。如果参数存储为32-bit浮点型，大约每个区域每一组需要2.5MB内存。给定3组324个区域，总计大约需要972个CNN回归模型，将所有模型全预装载到RAM中，需要2.39GB内存，这对于现代计算机是可以承受的。

**2) Training**: The CNN regression models are trained exclusively on synthetic X-ray images, because they provide reliable ground truth labels with little needs on laborious manual annotation, and the quantity of real X-ray images could be limited. For each group in each zone, we randomly generate 25,000 pairs of t and δt. The parameters t follow a uniform distribution with $t_α$ and $t_β$ constrained in the zone. The parameter errors δt also follow a uniform distribution, while 3 different distribution ranges are used for the 3 groups, as shown in Table I. The distribution ranges of δt for Group 1 are the target capture range that the regressors are designed for. The distribution ranges of $δt_x$, $δt_y$ and $δt_θ$ are reduced for Group 2, because they are close to zero after the regressors in the first group are applied. For the same reason, the distribution ranges of $δt_α$ and $δt_β$ are reduced for Group 3. For each pair of t and δt, a synthetic X-ray image $I_{t+δt}$  is generated and the LIR feature $X(t, I_{t+δt})$ is calculated following (9).

CNN回归模型只在合成X射线图像中进行的训练，因为有可靠的真值标签，基本不需要繁复的手动标注，而且真实的X射线图像数量肯定是有限的。对于每个区域中的每组参数，我们随机生成25000对t和δt。参数t遵循均匀分布，$t_α$和$t_β$在一定区间内。参数误差δt也遵循均匀分布，3组不同的参数使用3个不同的分布范围，如表1所示。组1中δt的分布范围是回归器设计的目标捕获范围。对于组2，$δt_x$, $δt_y$和$δt_θ$的分布范围是缩小的，因为在应用了第1组的回归器之后，它们接近于0。同样的原因，对于组3来说，$δt_α$和$δt_β$的分布范围也是缩小的。对于每一对t和δt，会生成一个合成X射线图像$I_{t+δt}$，然后用(9)式计算LIR特征$X(t, I_{t+δt})$。

Table 1 Distribution of randomly generated $δt$. $u(a,b)$ denotes the uniform distribution between a and b. The units for translations and rotations are mm and degree, respectively.

Group 1 | Group 2 | Group 3
--- | --- | ---
$δt_x≈u(-1.5, 1.5)$ | $δt_x≈u(-0.2, 0.2)$ | $δt_x≈u(-0.15, 0.15)$
$δt_y≈u(-1.5, 1.5)$ | $δt_y≈u(-0.2, 0.2)$ | $δt_y≈u(-0.15, 0.15)$
$δt_z≈u(-15, 15)$ | $δt_z≈u(-15, 15)$ | $δt_z≈u(-15, 15)$
$δt_θ≈u(-3, 3)$ | $δt_θ≈u(-0.5, 0.5)$ | $δt_θ≈u(-0.5, 0.5)$
$δt_α≈u(-15, 15)$ | $δt_α≈u(-15, 15)$ | $δt_α≈u(-0.75, 0.75)$
$δt_β≈u(-15, 15)$ | $δt_β≈u(-15, 15)$ | $δt_β≈u(-0.75, 0.75)$

The objective function to be minimized during the training is Euclidean loss, defined as: 训练过程中要最小化的目标函数为欧几里得损失，定义为：

$$Φ=\frac{1}{K} \sum_{i=1}^K ||y_i - f(X_i;W)||_2^2$$(15)

where K is the number of training samples, $y_i$ is the label for the i-th training sample, W is a vector of weights to be learned, $f(X_i;W)$ is the output of the regression model parameterized by W on the i-th training sample. The weights W are learned using Stochastic Gradient Descent (SGD)[24], with a batch size of 64, momentum of m=0.9 and weight decay of d = 0.0001. The update rule for W is:

其中K是训练样本的数量，$y_i$是第i个训练样本的标签，W是要学习的权重矢量，$f(X_i;W)$是回归模型的输出，参数为在第i个训练样本的W。权重W使用SGD学习，批规模为64，动量为m=0.9，权重衰减为d = 0.0001。W的更新规则为：

$$V_{i+1} := m⋅V_i - d⋅κ_i⋅W_i - κ_i⋅<\frac{∂Φ}{∂W}|_{W_i}>_{D_i}$$(16)

$$W_{i+1} := W_i + V_{i+1}$$(17)

where i is the iteration index, V is the momentum variable, $κ_i$ is the learning rate at the i-th iteration, and $<\frac{∂Φ}{∂W}|_{W_i}>_{D_i}$ is the derivative of the objective function computed on the i-th batch $D_i$ with respect to W, evaluated at $W_i$. The learning rate $κ_i$ is decayed in each iteration following

其中i是迭代索引，V是动量变量，$κ_i$是在第i次迭代的学习速率，$<\frac{∂Φ}{∂W}|_{W_i}>_{D_i}$是在第i个批次$D_i$上计算的目标函数对W的导数，评估为$W_i$。学习速率$κ_i$在每次迭代中按如下公式衰减

$$κ_i=0.0025⋅(1+0.0001⋅i)^{-0.75}$$(18)

The derivative $∂Φ/∂W$ is calculated using back-propagation. For weights shared in multiple paths, their derivatives in all paths are back-propagated separately and summed up for weight update. The weights are initialized using the Xavier method[25], and mini-batch SGD is performed for 12,500 iterations (32 epochs).

$∂Φ/∂W$的导数用反向传播计算，对于在多个路径中共享的路径，其在所有路径上的导数分别反向传播，然后求和进行权重更新。权重采用Xavier初始化，mini-batch SGD进行12500次迭代（32轮）。

## IV Experiments

### A. Datasets

We evaluated PEHL on datasets from the following 3 clinical applications to demonstrate its wide applicability for real-time 2-D/3-D registration:

我们在以下三个临床应用中的数据集上评估PHEL，以证明其可以在很多应用中进行实时2D/3D配准：

1) Total Knee Arthroplasty (TKA) Kinematics: In the study of the kinematics of TKA, 3-D kinematics of knee prosthesis can be estimated by matching the 3-D model of the knee prosthesis with the fluoroscopic video of the prosthesis using 2-D/3-D registration [26]. We evaluated PEHL on a fluoroscopic video consisting of 100 X-ray images of a patient's knee joint taken at the phases from full extension to maximum flexion after TKA. The size of the X-ray images is 1024 × 1024 with a pixel spacing of 0.36 mm. A 3-D surface model of the prosthesis was acquired by a laser scanner, and was converted to a binary volume for registration.

1) 全膝关节造形术动力学：在TKA的动力学研究中，膝盖假体的3D动力学的估计，要通过膝盖假体的3D模型与假体的透视视频用2D/3D配准[26]得到。我们在一个透视视频中评估PEHL，包含一个病人的膝盖关节的100幅X射线图像，在TKA后拍摄的从完全伸展到最大弯曲的状态。X射线图像的大小为1024×1024，像素间隔为0.36mm。假体的3D平面模型是通过激光扫描仪取得的，转换到二值体进行配准。

2) Virtual Implant Planning System (VIPS): VIPS is an intraoperative application that was established to facilitate the planning of implant placement in terms of orientation, angulation and length of the screws [27]. In VIPS, 2-D/3-D registration is performed to match the 3-D virtual implant with the fluoroscopic image of the real implant. We evaluated PEHL on 7 X-ray images of a volar plate implant mounted onto a phantom model of the distal radius. The size of the X-ray images is 1024 × 1024 with a pixel spacing of 0.223 mm. A 3-D CAD model of the volar plate was converted to a binary volume for registration.

2) 虚拟植入计划系统(VIPS)：VIPS是一种术中应用，是为了帮助植入定位的计划，包括螺丝的方向，角度和长度。在VIPS中，2D/3D配准的进行，是将3D的虚拟植入物与真实植入物的透射图像进行匹配得到的。我们在7幅X射线图像中评估PEHL，是一个volar plate植入物，装入桡骨远端的模体模型。X射线图像大小为1024×1024，像素间隔为0.223mm。Volar plate的3D CAD模型，转换到二值体中进行配准。

3) X-ray Echo Fusion (XEF): 2-D/3-D registration can be applied to estimate the 3-D pose of a transesophageal echocardiography (TEE) probe from X-ray images, which brings the X-ray and TEE images into the same coordinate system and enables the fusion of the two modalities [28]. We evaluated PEHL on 2 fluoroscopic videos with in total 94 X-ray images acquired during an animal study using a Siemens Artis Zeego C-Arm system. The size of the X-ray images is 1024 × 1024 with a pixel spacing of 0.154 mm. A micro-CT scan of the Siemens TEE probe was used for registration.

3) X射线Echo Fusion(XEF)：2D/3D配准可用于从X射线图像中估计经食管超声心动图(TEE)探头的3D姿态，将X射线和TEE图像放入相同的坐标系系统中，使两种模态得以融合[28]。我们在2个透射视频和94幅X射线图像上评估PEHL，图像是使用西门子Artis Zeego C形臂系统对一个小动物进行研究时拍的。X射线图像的大小为1024 × 1024，像素间隔0.154mm。西门子TEE探头的微型CT扫描用于配准。

Example datasets of the above 3 clinical applications are shown in Fig. 5. Examples of local ROIs and LIRs extracted from the 3 datasets are also shown in Fig. 6.

图5展示了上面3个临床应用的数据集例子。3个数据集中提取的局部ROIs和LIRs的例子如图6所示。

Fig. 5. Example data, including a 3-D model and a 2-D X-ray image of the object. (a) TKA. (b) VIPS. (c) XEF.

Fig. 6. Examples of local ROIs and LIRs. (a) TKA. (b) VIPS. (c) XEF.

Ground truth transformation parameters used for quantifying registration error were generated by first manually registering the target object and then applying an intensity-based 2-D/3-D registration method using Powell's method combined with Gradient Correlation (GC) [12]. Perturbations of the ground truth were then generated as initial transformation parameters for 2-D/3-D registration. For TKA and XEF, 10 perturbations were generated for each X-ray image, leading to 1,000 and 940 test cases, respectively. Since the number of X-ray images for VIPS is limited (i.e., 7), 140 perturbations were generated for each X-ray image to create 980 test cases. The perturbation for each parameter followed the normal distribution with a standard deviation equal to 2/3 of the training range of the same parameter (i.e., Group 1 in Table I). In particular, the standard deviations for $(t_x, t_y, t_z, t_θ, t_α, t_β)$ are 1 mm, 1 mm, 10 mm, 2 degrees, 10 degrees, 10 degrees. With this distribution, 42.18% of the perturbations have all 6 parameters within the training range, while the other 57.82% have at least one parameter outside of the training range.

用于量化配准误差的真值变换参数的生成，第一步手动配准目标物体，然后用基于灰度的2D/3D配准方法使用Powell的方法与梯度相关(GC)方法相结合生成。生成真值的扰动，作为初始变换参数，用于2D/3D配准。对于TKA和XEF，对每幅X射线图像生成了10个扰动，分别带来了1000和940个测试cases。由于用于VIPS的X射线图像是有限的（即，7），对每幅X射线图像生成了140个扰动，以生成980个测试cases。每个参数的扰动遵循正态分布，标准差等于同样参数训练范围的2/3（即表1中的组1）。特别的，$(t_x, t_y, t_z, t_θ, t_α, t_β)$的标准差为1mm，1mm，10mm，2度，10度，10度。在这个分布下，42.18%的扰动的6个参数都在训练范围内，而其他57.82%都至少有一个参数超出了训练范围。

### B. Synthetic Training Data Generation

The synthetic X-ray images used for training were generated by blending a DRR of the object with a background from real X-ray images:

用于训练的合成X射线图像，是通过将目标的DRR图像与真实X射线图像的背景混合到一起生成的：

$$I = I_{X-ray} + γ⋅G_σ*I_{DRR} + N(a,b)$$(19)

where $I_{X-ray}$ is the real X-ray image, $I_{DRR}$ is the DRR, $G_σ$ denotes a Gaussian smoothing kernel with a standard deviation σ simulating X-ray scattering effect, f*g denotes the convolution of f and g, γ is the blending factor, and N(a,b) is a random noise uniformly distributed between [a, b]. The parameters (γ, σ, a, b) were empirically tuned for each object (i.e., implants and TEE probe) to make the appearance of the synthetic X-ray image realistic. These parameters were also randomly perturbed within a neighborhood for each synthetic X-ray image to increase the variation of the appearance of the synthetic X-ray images, so that the regressors trained on them can be generalized well on real X-ray images. The background image use for a given synthetic image was randomly picked from a group of real X-ray images irrespective of the underlying clinical procedures so that the trained network will not be over-fitted for any specific type of background, which could vary significantly from case to case clinically. Examples of synthetic X-ray images of TKA, VIPS and XEF are shown in Fig. 7.

其中$I_{X-ray}$是真实的X射线图像，$I_{DRR}$是DRR图像，$G_σ$表示高斯平滑核，标准差为σ，仿真的是X射线散射效果，f*g表示f和g的卷积，γ是混合因子，N(a,b)是随机噪声，[a, b]间均匀分布。参数(γ, σ, a, b)对每个目标通过经验调节（即，植入物和TEE探头），以使得合成的X射线图像看起来更真实。这些参数也是对每幅合成X射线图像在一个邻域内随机扰动的，以增加合成X射线图像的外观变化，这样在这上面训练的回归器在真实X射线图像上可以很好的泛化。一个给定合成图像的背景图像是从一组真实X射线图像中随机选出的，与可能的临床过程是无关的，这样训练的网络不会对任意特定类型的背景过拟合，这在临床上随着病例的不同会有很大变化。TKA, VIPS和XEF合成X射线图像的例子如图7所示。

Fig.7. Example synthetic X-ray images used for training. (a)TKA, (b)VIPS, (c)XEF.

### C. Evaluation Metrics

The registration accuracy was assessed with the mean Target Registration Error in the projection direction (mTREproj) [29], calculated at the 8 corners of the bounding box of the target object. We regard mTREproj less than 1% of the size of the target object (i.e., diagonal of the bounding box) as a successful registration. For TKA, VIPS and XEF, the sizes of the target objects are 110 mm, 61 mm and 37 mm, respectively. Therefore, the success criterion for the three applications were set to mTREProj less than 1.10 mm, 0.61 mm and 0.37 mm, which are equivalent to 2.8 pixels, 3.7 pixels and 3.5 pixels on the X-ray image, respectively. Success rate was defined as the percentage of successful registrations. Capture range was defined as the initial mTREproj for which 95% of the registration were successful [29]. Capture range is only reported for experiments where there are more than 20 samples within the capture range.

配准准确率是通过在投影方向的平均目标配准误差(mTREproj)来评估的，这是在目标物体的边界框的8个角点上计算得到的。我们认为mTREproj小于目标物体尺寸的1%（即边界框的对角线），就是一个成功的配准。对于TKA，VIPS和XEF，目标物体的大小分别为110mm，61mm和37mm。因此，这三个应用的成功准则设为mTREProj小于1.10 mm, 0.61mm和0.37mm，这分别等价于X射线图像中的2.8像素，3.7像素和3.5像素。成功率定义为成功配准的比例。捕获范围定义为95%的配准都成功的初始mTREproj。捕获范围内有超过20个样本的试验，我们才给出捕获范围。

### D. Performance Analysis

We conducted the following experiments for detailed analysis of the performance and property of PEHL. The dataset from XEF was used for the demonstration of performance analysis because the structure of the TEE probe is more complex than the implants in TKA and VIPS, leading to an increased difficulty for an accurate registration. As described in Section III-C, PEHL can be applied for multiple iterations. We demonstrate the impact of the number of iterations on performance, by applying PEHL for 10 iterations and showing the registration success rate after each iteration. We also demonstrate the importance of the individual core components of PEHL, i.e., the CNN regression model and 3 algorithmic strategies, LIR, HPR and PSP, by disabling them and demonstrating the detrimental effects on performance. The following 4 scenarios were evaluated for 10 iterations to compare with PEHL:

我们进行了下列试验，详细分析了PEHL的性能和属性。XEF的数据集用于性能分析的展示，因为TEE探头的结构比TKA和VIPS的植入物更复杂，这导致进行准确的配准会非常困难。如III-C所述，PEHL可以进行多次迭代应用。我们证明了迭代次数在性能的影响，将PEHL进行了10次迭代，展示了每次迭代后的配准成功率。我们还证明了PEHL的单个核心组件的重要性，即，CNN回归模型和3种算法策略，LIR，HPR和PSP，通过将其关闭来展示对性能的有害影响。下面4种场景评估了10次迭代，以与PEHL进行比较：

- Without CNN: We implemented a companion algorithm using HAAR feature with Regression Forest as an alternative to the proposed CNN regression model. We extract 8 HAAR features as shown in Fig. 8 from the same training data used for training the CNNs. We mainly used edge and line features because δt largely corresponds to lines and edges in LIR. On these HAAR features, we trained a Regression Forest with 500 trees.

- 不用CNN：我们实现了一个伴随算法，使用HAAR特征与回归森林，替换掉提出的CNN回归模型。我们提取出8个HAAR特征，如图8所示，所用的数据集与训练CNNs的一样。我们主要采用边缘和线段特征，因为δt主要对应于LIR中的线段和边缘。在这些HAAR特征上，我们训练了一个500棵树的回归森林。

Fig. 8. HAAR features used in the experimenet “Without CNN”.

- Without LIR: A global image residual covering the whole object was used as the input for regression (shown in Fig. 2 as blue boxes). The CNN regression model was adapted accordingly. It has five hidden layers: two 5 × 5 convolutional layers, each followed by a 3 × 3 max-pooling layer with stride 3, and a fully-connected layer with 250 ReLU activation neurons. For each group in each zone, the network was trained on the same dataset used for training PEHL.

- 不用LIR：将覆盖了整个目标的全局图像残差用作输入，进行回归（如图2中的蓝框所示）。CNN回归模型也相应的采用了这个。其有5个隐藏层：两个5 × 5卷积层，每个后面接着一个3 × 3最大池化层，步长为3，一个全连接层，有250个ReLU激活的神经元。对于每个区域的每个组中，网络都在与训练PEHL相同的数据集上进行训练。

- Without HPR: The proposed CNN regression model shown in Fig. 4 was employed, but the output layer has 6 nodes, corresponding to the 6 parameters for rigid-body transformation. For each zone, the network was trained on the dataset used for training PEHL for Group 1.

- 没有HPR：采用了图4中提出的CNN回归模型，但是输出层有6个节点，对应着刚体变换的6个参数。对于每个区域，网络的训练都是在用于训练PEHL组1参数的数据集上进行。

- Without PSP: For each group of parameters, one CNN regression model was applied for the whole parameter space. Because LIR cannot be applied without PSP, the CNN regression model described in “without LIR” was employed in this scenario. The network was trained on 500,000 synthetic training samples with $t_α$ and $t_β$ uniformly distributed in the parameter space.

- 没有PSP：对于每个参数组，一个CNN回归模型用于整个参数空间。因为没有PSP就不能应用LIR，所以这个场景中采用了“不用LIR”部分描述的CNN回归模型。网络训练在50万幅合成训练样本中，$t_α$和$t_β$在参数空间中均匀分布。

We also conducted an experiment to analyze the precision of PEHL (i.e., the ability to generate consistent results starting from different initial parameters). To measure the precision, we randomly selected an X-ray image from the XEF dataset, and generated 100 perturbations of the ground truth following the same distribution described in Section IV-A. PEHL and the best performed intensity-based method, MI_GC_Powell (will be detailed in the Section IV-E) were applied starting from the 100 perturbations. The precision of the registration method was then quantified by the root mean squared distance in the projection direction (RMSDproj) from the registered locations of each target to their centroid. Smaller RMSDproj indicates higher precision.

我们还进行了试验来分析PEHL的精度（即，从不同的初始参数开始，生成一致结果的能力）。为度量精度，我们从XEF数据集中随机选择了一幅X射线图像，生成了真值的100个扰动，生成的数据分布与IV-A部分描述的一样。PEHL和性能最好的基于灰度的方法，MI_GC_Powell（在IV-E中详细描述），都是从这100个扰动开始应用的。配准方法的精度，是用从每个目标的配准位置到其重心的投影方向的均方根距离(RMSDproj)量化的。更小的RMSDproj说明精度更高。

### E. Comparison With State-of-the-Art Methods

We first compare PEHL with several state-of-the-art intensity-based 2-D/3-D registration methods. An intensity-based method consists of two core components, an optimizer and a similarity measure. A recent study compared four popular optimizers (Powell's method, Nelder-Mead, BFGS, CMA-ES) for intensity-based 2-D/3-D registration, and concluded that Powell's method achieved the best performance [30]. Therefore, in all evaluated intensity-based methods, we used Powell's method as the optimizer. We evaluated three popular similarity measures, MI, Cross Correlation (CC) and GC, which have also been reported to be effective in recent literature [3], [4], [12]. For example, MI has been adopted in [3] for monitoring tumor motion during radiotherapy. CC computed on splatting DRR has been adopted in [12] for 5 Degree of Freedom pose estimation of TEE probe. GC has been adopted in [4] for the assessing the positioning and migration of bone implants. The above three intensity-based methods are referred to as MI_Powell, CC_Powell and GC_Powell, indicating the adopted similarity measure and optimization method.

我们首先将PEHL与几种目前最好的基于灰度的2D/3D配准方法进行比较。基于灰度的方法包含两个核心部件，一个优化器，和一个相似度度量。一个最近的研究比较了四种流行的优化器(Powell's method, Nelder-Mead, BFGS, CMA-ES)在基于灰度的2D/3D配准方法中的应用，得出结论是，Powell方法取得了最佳性能。因此，在所有评估的基于灰度方法中，我们使用Powell方法作为优化器。我们评估了三种流行的相似性度量，MI，交叉相关(CC)和GC，在最近的文献中发现这几种度量很有用。比如，MI用于[3]中，在放射治疗中监控肿瘤运动。在splatting DRR中计算的CC被[12]采用，计算TEE探头的5自由度姿态估计。GC在[4]中采用，评估移植骨结构的位置和移动量。上述三种基于灰度的方法分别称为MI_Powell, CC_Powell和GC_Powell，表明采用的相似性度量和优化方法。

In addition to the above three methods, we implemented another intensity-based method combining MI and GC to achieve improved robustness and accuracy to compete with PEHL. MI focuses on the match of the histograms at the global scale, which leads to a relatively large capture range, but lacks fine accuracy. GC focuses on matching image gradients, which leads to high registration accuracy, but limits the capture range. The combined method, referred to as MI_GC_Powell, first applies MI_Powell to bring the registration into the capture range of GC, and then applies GC_Powell to refine the registration.

除了上述三种方法，我们实现了另一种基于灰度的方法，结合了MI和GC，得到改进的稳定性和准确率，以与PEHL进行竞争。MI关注的是全局尺度的直方图匹配，这带来了相对更大的捕获范围，但缺少精细的准确率。GC关注的是匹配图像梯度，这可以带来很高的配准准确率，但限制了捕获范围。结合的方法，我们称为MI_GC_Powell，首先应用MI_Powell以将配准带到GC的捕获范围，然后应用GC_Powell来提炼配准结果。

We also compared PEHL with CLARET, a linear regression-based 2-D/3-D registration method introduced in [16], which is closely related to PEHL, as it iteratively applies regressors on the image residual to estimate the transformation parameters. In [16], the linear regressors were reported to be trained on X-ray images with fixed ground truth transformation parameters, and therefore can only be applied on X-ray images with poses within a limited range. Since the input X-ray images used in our experiment do not have such limitation, we applied the PSP strategy to train linear regressors separately for each zone. For each zone, the linear regressor was trained on the dataset used for training PEHL for Group 1.

我们还将PEHL与CLARET进行了比较，[16]中提出的一种线性的基于回归的2D/3D配准方法，与PEHL紧密相关，因为这种方法是对图像残差迭代应用回归器，以估计变换参数。在[16]中，在X图像上训练了线性回归器，有固定的真值变换参数，因此只能应用到有限范围姿态变化的X射线图像上。由于我们试验中的输入X射线图像没有这样的限制，我们使用PSP策略，在每个区域中分别训练线性回归器。对于每个区域，都在用于训练PEHL组1的数据集上训练线性回归器。

### F. Experiment Environment

The experiments were conducted on a workstation with Intel Core i7-4790k CPU, 16GB RAM and Nvidia GeForce GTX 980 GPU. For intensity-based methods, the most computationally intensive component, DRR renderer, was implemented using the Ray Casting algorithm with hardware-accelerated 3-D texture lookups on GPU. Similarity measures were implemented in C++ and executed in a single CPU core. Both DRRs and similarity measures were only calculated within an ROI surrounding the target object, for better computational efficiency. In particular, ROIs of size 256 × 256, 512 × 512 and 400 × 400 were used for TKA, VIPS and XEF, respectively. For PEHL, the neural network was implemented with cuDNN acceleration using an open-source deep learning framework, Caffe [31].

试验是在一台Intel Core i7-4790k CPU, 16GB RAM和Nvidia GeForce GTX 980 GPU的工作站上进行的。对于基于灰度的方法，计算量最大的部分是DRR渲染，是用射线投射算法实现的，是在GPU上用3D纹理查找用硬件加速的。相似性度量是用C++实现的，在单个CPU核上执行的。DRRs和相似性度量都是在围绕目标的ROI上进行的，以得到更好的计算效率。特别的，对于TKA, VIPS和XEF分别采用了256 × 256, 512 × 512和400 × 400大小的ROI。对于PEHL，神经网络是用Caffe实现的，进行了cuDNN加速。

## V Results

### A. Performance Analysis

Fig. 9 shows the success rate as the number of iterations increases from 1 to 10 for five analyzed scenarios. The results show that the success rate of PEHL increased rapidly in the first 3 iterations (i.e., from 44.6% to 94.8%), and kept raising slowly afterward until 9 iterations (i.e., to 99.6%). The computation time of PEHL is linear to the number of iterations, i.e., each iteration takes ~34ms. Therefore, applying PEHL for 3 iterations is the optimal setting for the trade-off between accuracy and efficiency, which achieves close to the optimal success rate and a real-time registration of ~10 frames per second(fps). Therefore, in the rest of the experiment, PEHL was tested with 3 iterations unless stated otherwise.

图9给出了，对于5种分析的场景，当迭代次数从1增加到10时，成功率的情况。结果表明，PEHL的成功率在前3次迭代中迅速增加（即，从44.6%增加到94.8%），然后到第9次迭代会慢慢增加（即到99.6%）。PEHL的计算时间与迭代次数是线性关系，即，每次迭代大约耗时34ms。因此，PEHL进行3次迭代是准确率与效率的折中的最佳设置，会得到接近最佳的成功率，和实时的配准效率，即大约10fps。因此，在下面的试验中，PEHL一般都进行3次迭代，除非另有说明。

Fig. 9. Success rates of PEHL with 1 to 10 iterations. Four individual core components of PEHL, i.e., CNN, LIR, HPR and PSP, were disabled one at a time to demonstrate their detrimental effects on performance. Harr feature (HR) + Regression Forest (RF) was implemented to show the effect on performance without CNN. These results were generated on the XEF dataset.

The results show that the 3 proposed strategies, LIR, HPR and PSP, and the use of CNN all noticeably contributed to the final registration accuracy of PEHL. In particular, if the CNN regression model is replaced with HAAR feature + Regression Forest, the success rate at the 3rd iteration dropped to 70.7%, indicating that the strong non-linear modeling capability of CNN is critical to the success of PEHL. If the LIR is replaced with a global image residual, the success rate at the 3rd iteration dropped significantly to 52.2%, showing that LIR is a necessary component to simplify the target mapping so that it can be robustly regressed with the desired accuracy. When HPR and PSP are disabled, the system almost completely failed, dropping the success rate at the 3rd iteration to 19.5% and 14.9%, respectively, suggesting that HPR and PSP are key components that make the regression problem solvable using the proposed CNN regression model.

结果表明，提出的3种策略，LIR, HPR和PSP，以及CNN的使用都对PEHL的配准准确率有显著的贡献。特别是，如果CNN回归模型替换为HAAR特征+回归森林，第3次迭代的成功率会降低到70.7%，说明了CNN强力的非线性建模能力对于PEHL的成功非常关键。如果LIR替换为全局图像残差，在第3次迭代时的成功率显著降低到了52.2%，说明LIR是简化目标映射的必须部件，这样才能稳健的回归到期望的准确率。当HPR和PSP禁用时，系统几乎完全失败，第3次迭代的成功率分别降低到了19.5%和14.9%，说明HPR和PSP是很关键的部分，这样使用提出的CNN回归模型才能回归问题可解。

Fig. 10 shows the RMSDproj from registered target points to their corresponding centroid using both MI_GC_Powell and PEHL with 1 to 10 iterations. The results show that as the number of iteration increases, the RMSDproj of PEHL approaches zero, indicating that with sufficient number of iterations, PEHL can reliably reproduce the same result starting from different positions (e.g., 6 iterations leads to RMSEproj = 0.005mm). At the 3rd iteration, the RMSDproj of PEHL is 0.198 mm, which is 62% smaller than that of MI_GC_Powell, i.e., 0.52 mm. These results suggest that PEHL has a significant advantage over MI_GC_Powell in terms of precision.

图10展示了配准的目标点到其对应的重心的RMSDproj，使用的是MI_GC_Powell和PEHL算法，从1到10次迭代。结果表明，随着迭代次数增加，PEHL的RMSDproj逐渐趋于0，说明只要迭代次数足够多，PEHL可以从不同的位置可靠的复现相同的结果（如，6次迭代会得到RMSEproj = 0.005mm）。在第3次迭代时，PEHL的RMSDproj是0.198mm，比MI_GC_Powell的0.52mm要小62%。结果说明，PEHL比MI_GC_Powell在精度上有显著的优势。

Fig. 10. RMSDproj from the registered locations of each target to their centroid using MI_GC_Powell and PEHL with 1 to 10 iterations. At Number of Iterations = 0, the RMSEproj at the perturbed positions without registration is shown. These results were generated on the XEF dataset.

### B. Comparison With State-of-the-Art Methods

We first observed that the linear regressor in CLARET completely failed in our experiment setup. Table II shows the root mean squared error (RMSE) of the 6 parameters yielded by PEHL and CLARET on the synthetic training data for XEF. The linear regression resulted in very large errors on the training data (i.e., larger than the perturbation), indicating that the mapping from the global image residual to the underlying transformation parameters is highly non-linear, and therefore cannot be reliably captured by a linear regressor. In comparison, PEHL employs the 3 algorithmic strategies to simplify the non-linear relationship and captures it using a CNN with a strong non-linear modeling capability. As a result, PEHL resulted in a very small error on the synthetic training data. Its ability to generalize the performance to unseen testing data was then accessed on real data from the three clinical applications.

我们首先观察到，CLARET中的线性回归器在我们的试验设置中完全失败。表2给出了PEHL和CLARET在XEF的合成训练数据上给出的6个参数的均方根误差(RMSE)。线性回归在训练数据上得到的误差很大（即，大于扰动值），说明从全局图像残差到潜在的变换参数的映射是高度非线性的，因此不能被线性回归器可靠的捕获到。比较之下，PEHL采用三种算法策略，简化了非线性关系，使用具有很强非线性建模能力的CNN捕获到了这种映射。结果是，PEHL在合成训练数据上误差很小。其将性能泛化到未曾见过的测试数据的能力，在三个临床应用的真实数据上得到了验证。

Table II RMSE of the 6 transformation parameters yielded by PEHL and CLARET on the training data for XEF

| |  $t_x$(mm) | $t_y$(mm) | $t_z$(mm) | $t_θ$(°) | $t_α$(°) | $t_β$(°)
--- ｜ --- ｜ --- ｜ --- ｜ --- ｜ --- ｜ ---
Start | 0.86 | 0.86 | 8.65 | 1.71 | 8.66 | 8.66
PEHL | 0.04 | 0.04 | 0.32 | 0.06 | 0.18 | 0.18
CLARET | 0.51 | 0.88 | 34.85 | 2.00 | 19.41 | 17.52

Table III summarizes the success rate, capture range, percentiles of mTREproj and average running time per registration for PEHL and four intensity-based methods on the three applications. The results show that the four intensity-based methods, MI_Powell, GC_Powell, CC_Powell and MI_GC_Powell, all resulted in relatively small capture ranges and slow speeds that are incapable for real-time registration. The small capture range is owning to the limitation of non-convex optimization. Because the intensity-based similarity measures are highly non-convex, the optimizer is likely to get trapped in local maxima if the starting position is not close enough to the global maxima. The relatively slow speed is owning to the large number of DRR renderings and similarity measure calculations during the optimization. The fastest intensity-based method is CC_Powell, which took 0.4 ~ 0.9 s per registration, is still significantly slower than typical fluoroscopic frame rates (i.e., 10 ~ 15 fps). The success rates for MI_Powell, GC_Powell and CC_Powell are also very low, mainly due to two different reasons: 1) MI and CC are unable to resolve small mismatch; 2) GC is unable to recover large mismatch. By employing MI_Powell to recover large mismatch and GC_Powell to resolve small mismatch, MI_GC_Powell achieved much higher success rates, which is in line with our discussion in Section IV-E.

表3总结了三个应用中，4种基于灰度的方法和PEHL在每次配准中的成功率，捕获范围，mTREproj的百分比和平均运行时间。结果表明，4种基于灰度的方法，MI_Powell, GC_Powell, CC_Powell和MI_GC_Powell，其结果都是捕获范围较小，速度较慢，不能进行实时配准。小的捕获范围是因为非凸优化的局限。因为基于灰度的相似性度量是高度非凸的，如果初始位置与全局极值不够近，优化器很可能陷入局部极值点。相对较低的速度是因为有很多DRR渲染的计算，在优化时也有很多相似度度量的计算。基于灰度的方法，最快的是CC_Powell，每次配准需要0.4 ~ 0.9s，但也比典型的透视帧率（即10-15fps）明显要低。MI_Powell, GC_Powell和CC_Powell的成功率也很低，主要是两个不同的原因：1)MI和CC不能解决小的误匹配；2)GC不能恢复大的误匹配。通过采用MI_Powell，恢复大的误匹配，GC_Powell来解决小的不匹配，MI_GC_Powell可以得到高的多的成功率，这与IV-E的讨论是相符的。

Table III Quantitative experiment results of PEHL and baseline methods. Success rate is the percentage of successful registrations in each experiment. Capture range is the initial mTREproj for which 95% of the registrations were successful. The 10th, 25th, 50th, 75th and 90th percentiles of mTREproj are reported. Running time records the average and standard deviation of the computation time for each registration computed in each experiment. Capture range is only reported for experiments where there are more than 20 samples within the capture range.

The results show that PEHL achieved the best success rate and capture range among all evaluated methods on all three applications, and is capable for real-time registration. The advantage of PEHL in capture range compared to the 2nd best-performed method, i.e., MI_GC_Powell, is significant. In particular, on the three applications, PEHL resulted in 155% (on TKA), 99% (on VIPS) and 306% (on XEF) larger capture range than MI_GC_Powell, respectively. The success rates of PEHL are also higher than that of MI_GC_Powell by 27.8% (on TKA), 5% (on VIPS) and 5.4% (on XEF). The advantage of PEHL in capture range and robustness is primarily owning to the learning of the direct mapping from the LIR to the residual of the transformation parameters, which eliminates the need of optimizing over a highly non-convex similarity measure. PEHL resulted in a running time of ~0.1s per registration for all three applications, which is 20 ~ 45 times faster than that of MI_GC_Powell and leads to real-time registration at ~10 fps. In addition, because the computation involved in PEHL is fixed for each registration, the standard deviation of the running time of PEHL is almost zero, so that PEHL can provide real-time registration at a stable frame rate. In comparison, intensity-based methods require different numbers of iterations for each registration, depending on the starting position, which leads to a relatively large standard deviation of the running time. The mTREproj percentiles show that at lower percentiles (e.g., 10th and 25th), the mTREproj of PEHL is in general larger than that of MI_GC_Powell. This is partially owning to the fact that the ground truth parameters were generated using GC, which could bear a slight bias toward intensity-based methods using GC as the similarity measure. For higher percentiles (e.g., 75th and 90th), the mTREproj of PHEL becomes smaller than that of MI_GC_Powell, showing that PHEL is more robust than MI_GC_Powell. The distributions of mTREproj before and after registration using MI_GC_Powell and PEHL on the three applications are shown in Fig. 11.

结果表明，PEHL在所有三个应用中，在所有评估的方法中，取得了最佳成功率和捕获范围，而且可以进行实时配准。PEHL与第二位表现的方法，即MI_GC_Powell，在捕获范围中的优势中比较起来，是非常大的。特别是，在这三个应用中，PEHL比MI_GC_Powell在这三个应用中的捕获范围分别大了155% (on TKA), 99% (on VIPS) and 306% (on XEF)。PEHL的成功率也比MI_GC_Powell高了27.8% (on TKA), 5% (on VIPS) and 5.4% (on XEF)。PEHL在捕获范围和稳定性上的优势，主要是由于从LIR到变换参数的残差的直接映射的学习，这就不需要在高度非凸的相似性度量中进行优化了。PEHL的运行时间为，在所有三种应用中，每次配准~0.1s，这比MI_GC_Powell要快20～45倍，可以达到~10 fps的实时配准率。另外，因为PEHL的计算在每次配准中是固定的，PEHL运行时间的标准差几乎是0，这样PEHL的实时配准速度很稳定。比较下来，基于灰度的方法，对于配准需要不同数量的迭代，这与开始位置有关系，这样运行时间的标准差会相对较大。mTREproj的百分位说明，在较低的百分位上（如10th and 25th），PEHL的mTREproj一般比MI_GC_Powell更大一些。这部分是因为，真值参数是使用GC生成的，这会给使用GC作为相似性度量的基于灰度的方法带来一些偏差。对于较高的百分位（如75th和90th），PHEL的mTREproj比MI_GC_Powell的变得要小，说明PHEL比MI_GC_Powell更稳定。使用MI_GC_Powell和PEHL，在三种应用中，mTREproj在配准前后的分布如图11所示。

Fig. 11. mTREproj before and after registration using MI_GC_Powell and PEHL on TKA, VIPS and XEF applications.

## VI Discussion and conclusion

In this paper, we presented a CNN regression approach, PEHL, for real-time 2-D/3-D registration. To successfully solve 2-D/3-D registration problems using regression, we introduced 3 novel algorithmic strategies, LIR, HPR and PSP, to simplify the underlying mapping to be regressed, and designed a CNN regression model with strong non-linear modeling capability to capture the mapping. We furthermore validated that all 3 algorithmic strategies and the CNN model are important to the success of PEHL, by disabling them from PEHL and showing the detrimental effect on performance. We empirically found that applying PEHL for 3 iterations is the optimal setting, which leads to close to the optimal success rate and a real-time registration speed of ~ 10 fps. We also demonstrated that PEHL has a strong ability to reproduce the same registration result from different initial positions, by showing that the RMSDproj of registered targets approaches to almost zero (i.e., 0.005 mm) as the number of iterations of PEHL increases to 6. In comparison, the RMSEproj using the best performed intensity-based method, MI_GC_Powell, is 0.52 mm. On three potential clinical applications, we compared PEHL with 4 intensity-based 2-D/3-D registration methods and a linear regression-based method, and showed that PEHL achieved much higher robustness and larger capture range. In particular, PEHL increased the capture range by 99%~306% and the success rate by 5%~27.8%, compared to MI_GC_Powell. We also showed that PEHL achieved significantly higher computational efficiency than intensity-based methods, and is capable of real-time registration.

本文中，我们提出了一种CNN回归方法，PEHL，可以进行2D/3D配准。为成功的使用回归来解决2D/3D配准，我们提出了3种新的算法策略，LIR，HPR和PSP，以简化要回归的潜在映射，设计了一种CNN回归模型，有很强的非线性建模能力，可以捕获到这种映射。我们进一步验证了PEHL的成功，其三种算法策略和CNN模型都是很重要的，从PEHL中禁用某个功能，都会对性能有不好的影响。我们通过经验发现，PEHL应用3次迭代是最佳的设置，这会带来接近最优的成功率，和实时的配准速度，约10fps。我们还证明了，PEHL可以从不同的初始位置复现同样的配准结果，随着PEHL的迭代次数增加到6，配准目标的RMSDproj的逐渐趋近于0（即，0.005mm）。比较之下，采用性能最好的基于灰度的方法MI_GC_Powell，其RMSEproj是0.52mm。在三个临床应用中，我们比较了PEHL和4个基于灰度的2D/3D配准方法，和一个基于线性回归的方法，表明PEHL取得了高很多的稳定性和大的捕获范围。特别的，与MI_GC_Powell相比，PEHL增加了99%~306%的捕获范围，5%~27.8%的成功率。我们还展示了，PEHL比基于灰度的方法，取得了明显更高的计算效率，可以进行实时的配准。

The significant advantage of PEHL in robustness and computational efficiency over intensity-based methods is mainly owning to the fact that CNN regressors are trained to capture the mapping from LIRs to the underlying transformation parameters. In every iteration, PEHL fully exploits the rich information embedded in LIR to make an informed estimation of the transformation parameters, and therefore it is able to achieve highly robust and accurate registration with only a minimum number of iterations. In comparison, intensity-based methods always map the DRR and X-ray images to a scalar-valued merit function, where the information about the transformation parameters embedded in the image intensities is largely lost. The registration problem is then solved by heuristically optimizing this scalar-valued merit function, which leads to an inefficient iterative computation and a high chance of getting trapped into local maxima.

PEHL在稳定性和计算效率上，比基于灰度方法的明显优势，主要归功于，CNN回归器是训练用于捕获从LIRs到潜在的变换参数的映射。在每次迭代中，PEHL完整的利用了LIR中的丰富信息，以进行变换参数的估计，因此可以通过最少的迭代次数，得到高度稳定和准确的配准。比较之下，基于灰度的方法，都是将DRR和X射线图像映射到一个标量的度量函数，其中关于变换参数的信息主要是在图像灰度中的，大部分都丢失了。配准问题的解决，是通过优化标量值的度量函数，这会带来低效的迭代计算，而且很可能会陷入到局部极值中。

The results also show that PEHL is more accurate and robust than two accelerated intensity-based 2-D/3-D registration methods, Sparse Histogramming MI (SHMI) [9] and Direct Splatting Correlation (DSC) [12], which employ sub-sampled DRR and splatting DRR to quickly compute approximated MI and CC, respectively. Because of the approximation, SHMI and DSC theoretically achieve the same or degraded accuracy compared to using original MI and CC. As shown in Table III, all reported mTREproj percentiles of PEHL are lower than that of MI_Powell and CC_Powell, and the differences at mid-range percentiles (i.e., 25th, 50th and 75th) are quite significant. In particular, at the 50th percentile, the mTREproj of PEHL are 25%~65% lower than that of MI_Powell and CC_Powell on all three applications. These results suggest that PEHL significantly outperforms SHMI and DSC in terms of robustness and accuracy. In terms of computational efficiency, while all three methods are capable of real-time registration, with an efficient GPU implementation, DSC reported the highest registration speed (i.e., 23.6~92.3 fps) [12].

结果还表明，PEHL与两种加速过的基于灰度的2D/3D配准方法，Sparse Histogramming MI (SHMI) [9]和Direct Splatting Correlation (DSC) [12]相比，要更准确，更稳定，这两种方法都分别使用了子取样的DRR和splatting DRR，以更快的计算近似的MI和CC。因为有近似，与使用原始的MI和CC相比，SHMI和DSC理论上取得了相同或更低的准确率。如表III所示，所有给出的mTREproj百分位都比MI_Powell和CC_Powell要更低，在中等百分位（即，25th, 50th和75th）上的差别非常明显。特别是，在50th百分位上，PEHL的mTREproj比MI_Powell和CC_Powell，在所有三个应用中，都要低25%~65%。这些结果说明，PEHL在稳定性和准确率上，显著超过了SHMI和DSC。在计算效率上，所有三种方法在进行高效的GPU实现时，都可以进行实时配准，DSC给出了最高的配准速度（即23.6~92.3 fps）[12]。

PEHL employs HPR and PSP to break the complex regression problem to several much simpler problems. The PSP strategy divides the parameter space of out-of-plane rotations into small zones, and trains regressors for each zone separately. Smaller zones will make the regression task simpler, but at the same time increase the training effort and memory consumption at the runtime. In this paper, we empirically selected the size of each zone in the PSP strategy to be 20 × 20 degrees, which leads to satisfactory registration accuracy and robustness and keeps the number of zones manageable (i.e., 324). The HPR strategy divides the 6 parameters into 3 groups, and trains regressors for each group separately. Therefore, there are in total 324 × 3 = 972 regressors to be trained and loaded at the runtime. In order to make the memory footprint required by the 972 regressors manageable at runtime, we use a weight sharing mechanism in the CNN model, which leads to 2.39 GB total memory consumption. In comparison, without weight sharing, pre-loading the same CNN regression model used in PEHL requires 13.7 GB RAM, which could be impractical in a clinical setup.

PEHL采用HPR和PSP，以将复杂的回归问题，分解成几个更简单的问题。PSP策略将平面外旋转的参数空间分成很小的区域，对每个区域分别训练回归器。更小的区域会使回归任务更简单，但同时在运行时增加了训练负担和内存消耗。本文中，我们通过经验，选择了PSP策略中每个区域的大小为20 × 20度，这可以得到满意的配准准确率和稳定性，区域数量也不会太多（即，324）。HPR策略将6个参数分成3组，对每个组分别训练回归器。因此，总共有324 × 3 = 972个回归器要训练，并装载到运行时中。为使972个回归器的内存占用在运行时可行，我们在CNN模型中使用权重分享的机制，这可以达到2.39GB的总计内存消耗。比较之下，没有权重分享，PEHL中使用同样的CNN回归模型会占用13.7GB内存消耗，这在临床应用中是不太现实的。

Like any machine learning-based methods, an important factor for the success of PEHL is the quantity and quality of the training data. For PEHL, it has been a challenge to obtain sufficient amount of annotated real X-ray images for training, because accurate annotation of 3-D transformation on X-ray projection image is very difficult, especially for those out-of-plane parameters. We have shown that by generating well-simulated synthetic data and training the CNN network on synthetic data only, we could achieve a high performance when applying PEHL on real X-ray images. However, it's worth noting that if the object to be registered is a device or implant that is manufactured with a fixed design, it is also possible to have a factory setup to massively acquire real X-ray images with a known ground truth for training PEHL.

就像其他的基于机器学习的方法一样，PEHL成功的一个重要因素是，训练数据的数量和质量。对于PEHL，要得到足够的标注过的真实X射线图像进行训练，是非常困难的，因为在X射线投影图像中准确的标注3D变换非常困难，尤其是那些平面外参数。我们证明了，通过生成很好的模拟的合成数据，并只在合成数据中训练CNN网络，我们将PEHL应用到真实X射线图像中，可以得到很好的性能。但是，值得注意的是，如果要配准的目标是一个设备或植入物，是用固定的设计生产的，用其工厂设置来得到大量真实的X射线图像，这样真值也是已知的，用来训练PEHL，这也是可能的。

One of our future work is to investigate the possibility of sharing the CNN weights across groups and/or zones, so that the memory footprint can be further reduced, making more complex and deeper network structures affordable. The difficulty of sharing the CNN across groups and/or zones lies in the fact that the training data for different groups and zones are different, which makes the training of the shared CNN a non-trivial task. Another future work is to extend PEHL for multi-plane 2-D/3-D registration, i.e., registering one 3-D model to multiple X-ray images acquired from different angles. This is currently a limitation of PEHL compared to intensity-based methods, which can be straightforwardly extended from mono-plane to multi-plane setup by simply combining the similarity measures from all the planes into one value. One possible way to achieve PEHL for multi-plane registration would be to apply regression on each plane separately and then combine all regression results into one estimation.

我们的一个未来工作是，研究在不同组或区域中共享CNN权重的可能性，这样内存占用可以得到进一步降低，使得可以承受更复杂更深度的网络结构。在不同组或区域中共享CNN的困难是，对于不同组和区域的训练数据是不同的，这使得共享CNN的训练比较麻烦。另一项未来的工作是将PEHL拓展到多平面2D/3D配准，即，将一个3D模型与多个X射线图像进行配准，这些图像是从不同的角度拍摄得到的。与基于灰度的方法相比，这是PEHL目前的一个限制，不能很直接的从单平面拓展到多平面的设置，而基于灰度的方法可以很简单的将所有平面的相似性度量结合到一起，得到一个值，从而完成这个任务。一个可能的用PEHL进行多平面配准的方法，是对每个平面都进行回归，将所有回归结果综合到一个估计中。