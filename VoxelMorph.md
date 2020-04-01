# VoxelMorph: A Learning Framework for Deformable Medical Image Registration

Guha Balakrishnan et. al. Computer Science and Artificial Intelligence Lab, MIT

## 0. Abstract

We present VoxelMorph, a fast learning-based framework for deformable, pairwise medical image registration. Traditional registration methods optimize an objective function for each pair of images, which can be time-consuming for large datasets or rich deformation models. In contrast to this approach, and building on recent learning-based methods, we formulate registration as a function that maps an input image pair to a deformation field that aligns these images. We parameterize the function via a convolutional neural network (CNN), and optimize the parameters of the neural network on a set of images. Given a new pair of scans, VoxelMorph rapidly computes a deformation field by directly evaluating the function. In this work, we explore two different training strategies. In the first (unsupervised) setting, we train the model to maximize standard image matching objective functions that are based on the image intensities. In the second setting, we leverage auxiliary segmentations available in the training data. We demonstrate that the unsupervised model’s accuracy is comparable to state-of-the-art methods, while operating orders of magnitude faster. We also show that VoxelMorph trained with auxiliary data improves registration accuracy at test time, and evaluate the effect of training set size on registration. Our method promises to speed up medical image analysis and processing pipelines, while facilitating novel directions in learning-based registration and its applications. Our code is freely available at http://voxelmorph.csail.mit.edu.

我们提出了VoxelMorph，可以对成对的、有形变的医学图像进行配准，是一个快速的学习框架。传统配准方法对每一对图像优化一个目标函数，对于大型数据集或很多形变的模型，会非常耗时。与这种方法相比，在最近的基于学习的方法的基础上，我们将配准问题描述为一个函数，将输入图像对映射到一个形变场，这个形变场可以对这些图像进行对齐。我们用CNN对这个函数进行参数化，在图像数据集中对网络参数进行优化。给定一对图像，VoxelMorph通过直接评估函数，可以快速计算形变场。本文中，我们探索了两种不同的训练策略。在第一种设置中（无监督），我们训练模型，来最大化标准的基于图像灰度的匹配目标函数。在第二种设置中，我们在训练数据中利用了可用的辅助分割结果。我们证明了，无监督模型的准确率与目前最好的方法是差不多的，而速度则明显更快。我们还证明了，用辅助数据训练的VoxelMorph，在测试时改进了配准准确率，并评估了在配准时的训练数据集的大小。我们的方法加速了医学图像分析和处理的流程，而同时促进基于学习的配准的新方向，及其应用。代码已经开源。

**Index Terms** — registration, machine learning, convolutional neural networks

## 1. Introduction

Deformable registration is a fundamental task in a variety of medical imaging studies, and has been a topic of active research for decades. In deformable registration, a dense, non-linear correspondence is established between a pair of images, such as 3D magnetic resonance (MR) brain scans. Traditional registration methods solve an optimization problem for each volume pair by aligning voxels with similar appearance while enforcing constraints on the registration mapping. Unfortunately, solving a pairwise optimization can be computationally intensive, and therefore slow in practice. For example, state-of-the-art algorithms running on the CPU can require tens of minutes to hours to register a pair of scans with high accuracy [1]–[3]. Recent GPU implementations have reduced this runtime to just minutes, but require a GPU for each registration [4].

形变配准是很多医学图像研究中的基础任务，数十年中一直是积极的研究领域。在形变配准中，一对图像，比如3D MRI脑部扫描，要确定一个密集的非线性的对应性。传统配准方法通过将体素与类似外观的体素进行对齐，同时对配准映射加入一些约束，对每个体对求解一个优化问题。不幸的是，求解成对的优化问题计算量会非常大，因此实践中会非常低效。比如，目前在CPU上运行的最好的算法，要对一对scans进行高精度的配准，可能需要几十分钟到几个小时。最近的GPU实现，可以将运行时间降低到几分钟，但需要GPU进行配准。

We present a novel registration method that learns a parametrized registration function from a collection of volumes. We implement the function using a convolutional neural network (CNN), that takes two n-D input volumes and outputs a mapping of all voxels of one volume to another volume. The parameters of the network, i.e. the convolutional kernel weights, can be optimized using only a training set of volumes from the dataset of interest. The procedure learns a common representation that enables alignment of a new pair of volumes from the same distribution. In essence, we replace a costly optimization solved for each test image pair with one global function optimization during a training phase. Registration of a new test scan pair is achieved by simply evaluating the learned function on the given volumes, resulting in rapid registration, even on a CPU. We implement our method as a general purpose framework, VoxelMorph, available at http://voxelmorph.csail.mit.edu (We implement VoxelMorph as a flexible framework that includes the methods proposed in this manuscript, as well as extensions that are beyond the scope of this work [5]).

我们提出一种新的配准方法，从体数据集中学习一种参数化的配准函数。我们使用CNN来实现这个函数，以两个n-D体为输入，输出从一个体到另一个体的所有体素的映射。网络的参数，即卷积核的权重，用感兴趣的数据集的训练数据进行优化得到。这个过程学习到了一个通用表示，可以对相同分布的新的体对进行对齐。本质上，我们将对每个测试图像对求解的耗时很长的优化问题，替换为了训练阶段的全局函数优化问题。新的测试样本对的配准，只需要在给定的体中评估学习到的函数就可以得到，所以可以进行快速的配准，即使在CPU上都可以。我们将我们的方法实现为通用目标的框架，VoxelMorph，代码已开源。

In the learning-based framework of VoxelMorph, we are free to adopt any differentiable objective function, and in this paper we present two possible choices. The first approach, which we refer to as unsupervised(We use the term unsupervised to underscore the fact that VoxelMorph is a learning method (with images as input and deformations as output) that requires no deformation fields during training. Alternatively, such methods have also been termed self-supervised, to highlight the lack of supervision, or end-to-end, to highlight that no external computation is necessary as part of a pipeline (such as computing ’true’ deformation fields)), uses only the input volume pair and the registration field computed by the model. Similar to traditional image registration algorithms, this loss function quantifies the dissimilarity between the intensities of the two images and the spatial regularity of the deformation. The second approach also leverages anatomical segmentations available at training time for a subset of the data, to learn network parameters.

在基于学习的框架VoxelMorph中，我们可以自由选择任何可微分的目标函数，本文中，我们给出两种可能的选择。第一种方法，我们称之为无监督的（我们使用无监督这个词，以强调VoxelMorph是一种学习方法，输入是图像，输出是形变场，在训练时不需要形变场；此外，这种方法也称为自监督的，以强调不需要监督，或端到端的，以强调这个流程中不需要其他计算，比如计算真值形变场），只使用输入体对，和模型计算得到的形变场。与传统的图像配准算法相似的是，这个损失函数量化的是两幅图像的灰度之间的不相似性，和形变场的空间规范性。第二种方法，还在训练时利用了，一部分数据的可用的解剖分割结果，以学习网络参数。

Throughout this study, we use the example of registering 3D MR brain scans. However, our method is broadly applicable to other registration tasks, both within and beyond the medical imaging domain. We evaluate our work on a multi-study dataset of over 3,500 scans containing images of healthy and diseased brains from a variety of age groups. Our unsupervised model achieves comparable accuracy to state-of-the-art registration, while taking orders-of-magnitude less time. Registration with VoxelMorph requires less than a minute using a CPU and under a second on a GPU, in contrast to the state-of-the-art baselines which take tens of minutes to over two hours on a CPU.

贯穿整篇研究，我们使用的例子是，对3D MRI脑部图像进行配准。但是，我们的方法可以广泛的应用于其他配准任务，包括医学成像范围内和范围外的。我们在一个多study的数据集上评估我们的工作，包含超过3500个scans，包含很多年龄组的健康和不健康的脑MRI图像。我们的无监督模型，与目前最好的配准模型可以取得差不多的准确率，而所需的时间则少的多。用VoxelMorph进行配准，使用CPU的话，所需时间少于1min，用GPU则只需要1s，目前最好的基准，在GPU上耗时超过10分钟，在CPU上则耗时超过2小时。

This paper extends a preliminary version of the work presented at the 2018 International Conference on Computer Vision and Pattern Recognition [6]. We build on that work by expanding analyses, and introducing an auxiliary learning model that can use anatomical segmentations during training to improve registration on new test image pairs for which segmentation maps are not available. We focus on providing a thorough analysis of the behavior of the VoxelMorph algorithm using two loss functions and a variety of settings, as follows. We test the unsupervised approach on more datasets and both atlas-based and subject-to-subject registration. We then explore cases where different types and numbers of anatomical region segmentations are available during training as auxiliary information, and evaluate the effect on registration of test data where segmentations are not available. We present an empirical analysis quantifying the effect of training set size on accuracy, and show how instance-specific optimization can improve results. Finally, we perform sensitivity analyses with respect to the hyperparameter choices, and discuss an interpretation of our model as amortized optimization.

本文有一个初始版本，发表在2018 ICCVPR，本文对其进行了拓展。我们拓展了分析，引入了一个辅助学习模型，可以在训练过程中使用解剖分割信息，以在新的测试图像对上（没有分割信息）改进配准。我们关注的是，对使用两种损失函数和几种设置的VoxelMorph算法的行为给出彻底的分析，如下所述。我们在更多的数据集中测试了无监督方法，也测试了基于atlas的配准，和subject-to-subject配准方法。我们然后探索了，在训练过程中，可以使用不同类型和数量的解剖区域分割作为辅助信息的方法，并在分割不可用的测试数据中评估配准的效果。我们给出一个经验分析结果，量化了训练集大小对准确率的效果，展示了具体实例的优化是确实可以改进结果。最后，我们对超参数的选择进行了敏感性分析，并将我们的模型进行讨论解释，解释成分次的优化。

The paper is organized as follows. Section 2 introduces medical image registration and Section 3 describes related work. Section 4 presents our methods. Section 5 presents experimental results on MRI data. We discuss insights of the results and conclude in Section 6.

本文组织如下。第2部分介绍了医学图像配准，第3部分描述了相关的工作。第4部分给出了我们的方法。第5部分给出了在MRI数据上的试验结果。我们讨论了结果了内涵，并在第6部分进行总结。

## 2. Background

In the traditional volume registration formulation, one (moving or source) volume is warped to align with a second (fixed or target) volume. Fig. 1 shows sample 2D coronal slices taken from 3D MRI volumes, with boundaries of several anatomical structures outlined. There is significant variability across subjects, caused by natural anatomical brain variations and differences in health state. Deformable registration enables comparison of structures between scans. Such analyses are useful for understanding variability across populations or the evolution of brain anatomy over time for individuals with disease. Deformable registration strategies often involve two steps: an initial affine transformation for global alignment, followed by a much slower deformable transformation with more degrees of freedom. We concentrate on the latter step, in which we compute a dense, nonlinear correspondence for all voxels.

在传统的体配准的表述中，一个（移动的或源）体通过变形，与另一个（固定的或目标）体进行对齐。图1给出了3D MRI体中的2D冠状切片的样本，勾画出了几个解剖结构的边缘。在不同目标间有很大的变化，主要是因为自然的脑解剖结构之间的变化，和不同健康状况的差异。形变配准使得不同扫描之间的结构得以比较。这种分析对于理解不同人群之间的差异是有用的，或个人随着病情的进展，脑部解剖结构的演化。形变配准策略通常有两个步骤：先进行初始的仿射变换，以进行全局配准，然后进行慢的多的形变变换，自由度更多。我们聚焦在后面这一步骤，其中我们回计算出所有体素间的密集、非线性对应性。

Fig. 1: Example coronal slices from the MRI brain dataset, after affine alignment. Each column is a different scan (subject) and each row is a different coronal slice. Some anatomical regions are outlined using different colors: L/R white matter in light/dark blue, L/R ventricles in yellow/red, and L/R hippocampi in purple/green. There are significant structural differences across scans, necessitating a deformable registration step to analyze inter-scan variations.

Most existing deformable registration algorithms iteratively optimize a transformation based on an energy function [7]. Let f and m denote the fixed and moving images, respectively, and let φ be the registration field that maps coordinates of f to coordinates of m. The optimization problem can be written as:

多数已有的形变配准算法，基于一个能量函数对变换进行优化。令f和m分别表示固定图像和移动图像，令φ为配准场，将f的坐标系映射到m中。优化问题可以表示为：

$$\hat φ = arg min_φ L(f, m, φ)$$(1)
$$= arg min_φ L_{sim}(f, m ◦ φ) + λL_{smooth}(φ)$$(2)

where m ◦ φ represents m warped by φ, function $L_{sim} (·, ·)$ measures image similarity between its two inputs, $L_{smooth}(·)$ imposes regularization, and λ is the regularization trade-off parameter.

其中m ◦ φ表示用φ对m进行形变，函数$L_{sim} (·, ·)$度量的是其两个输入之间的图像相似度，$L_{smooth}(·)$对其施加正则化，λ是正则化折中函数。

There are several common formulations for φ, $L_{sim}$ and $L_{smooth}$. Often, φ is characterized by a displacement vector field u specifying the vector offset from f to m for each voxel: φ = Id + u, where Id is the identity transform [8]. Diffeomorphic transforms model φ through the integral of a velocity vector field, preserving topology and maintaining invertibility on the transformation [9]. Common metrics used for $L_{sim}$ include intensity mean squared error, mutual information [10], and cross-correlation [11]. The latter two are particularly useful when volumes have varying intensity distributions and contrasts. $L_{smooth}$ enforces a spatially smooth deformation, often modeled as a function of the spatial gradients of u.

φ，$L_{sim}$和$L_{smooth}$有几种常见的表示形式。通常，φ要有一个偏移矢量场u，指定了从f到m每个体素的向量偏移：φ = Id + u，其中Id是恒等变换。通过对速度向量场进行积分，得到的微分同胚的变换模型φ，保留了拓扑结构，保持了变换的可逆性。用于$L_{sim}$的常见度量，包括灰度均方误差，互信息，和互相关。当体有变化的灰度分布和对比度时，后两者尤其有用。$L_{smooth}$加入了一个空间平滑形变，通常建模为空间梯度u的函数。

Traditional algorithms optimize (1) for each volume pair. This is expensive when registering many volumes, for example as part of population-wide analyses. In contrast, we assume that a field can be computed by a parameterized function of the data. We optimize the function parameters by minimizing the expected energy of the form of (1) over a dataset of volume pairs. Essentially, we replace pair-specific optimization of the deformation field by global optimization of the shared parameters, which in other domains has been referred to as amortization [12]–[15]. Once the global function is estimated, a field can be produced by evaluating the function on a given volume pair. In this paper, we use a displacement-based vector field representation, and focus on various aspects of the learning framework and its advantages. However, we recently demonstrated that velocity-based representations are also possible in a VoxelMorph-like framework, also included in our codebase [5].

传统算法对每个体对优化(1)式。当对很多体进行配准时，这运算量非常大，比如进行人口间的分析时。相比之下，我们假设，形变场可以由数据的参数化函数计算得到。我们通过在体对数据集上，最小化期望(1)式的能量函数，来优化这个函数参数。本质上来说，我们将形变场的成对优化，替换成了共享参数的全局优化，这在其他领域中被称为amortization，分期。一旦估计出了全局函数，通过在一个给定的体对中评估这个函数，就可以生成一个场。本文中，我们使用基于偏移的向量场表示，关注学习框架的各种方面，及其优势。但是，我们最近证明了，基于速度的表示在类VoxelMorph类的框架中，也是可能的，这也是在我们的代码库中的。

## 3. Related Work

### A. Medical Image Registration (Non-learning-based)

There is extensive work in 3D medical image registration [8], [9], [11], [16]–[21]. Several studies optimize within the space of displacement vector fields. These include elastic-type models [8], [22], [23], statistical parametric mapping [24], free-form deformations with b-splines [25], discrete methods [17], [18] and Demons [19], [26]. Diffeomorphic transforms, which are topology-preserving, have shown remarkable success in various computational anatomy studies. Popular formulations include Large Diffeomorphic Distance Metric Mapping (LDDMM) [9], [21], [27]–[32], DARTEL [16], diffeomorphic demons [33], and standard symmetric normalization (SyN) [11]. All of these non-learning-based approaches optimize an energy function for each image pair, resulting in slow registration. Recent GPU-based algorithms build on these concepts to reduce algorithm runtime to several minutes, but require a GPU to be available for each registration [4], [34].

在3D医学图像配准中有很多工作。一些研究对偏移向量场进行优化。这包括弹性类的模型的，统计参数映射的，带有b样条的自由形态形变，离散方法和Demons。微分同胚变换是保持拓扑的，在各种计算解剖研究中表现出了很大的成功。流行的表述包括，大型微分同胚距离度量映射(LDDMM)，DARTEL，微分同胚demons，和标准对称归一化(SyN)。所有这些基于非学习的方法，对每个图像对的能量函数进行优化，配准速度非常慢。最近的基于GPU的算法，在这些概念的基础上，降低算法的运行时间到几分钟，但需要GPU运行每次配准。

### B. Medical Image Registration (Learning-based)

There are several recent papers proposing neural networks to learn a function for medical image registration. Most of these rely on ground truth warp fields [35]–[39], which are either obtained by simulating deformations and deformed images, or running classical registration methods on pairs of scans. Some also use image similarity to help guide the registration [35]. While supervised methods present a promising direction, ground truth warp fields derived via conventional registration tools as ground truth can be cumbersome to acquire and can restrict the type of deformations that are learned. In contrast, VoxelMorph is unsupervised, and is also capable of leveraging auxiliary information such as segmentations during training if those are available.

最近有几篇文章提出，用神经网络学习一个函数进行医学图像配准。多数这些方法都是基于真值形变场的，这通常都是通过模拟形变，和形变图像获得的，或在图像对中运行经典配准方法。一些方法使用图像相似度来帮助引导配准。虽然有监督方法提出了很有希望的方向，但通过传统配准工具推导得到的真值形变场作为真值，是很难处理得到的，会限制学习到的形变类型。对比起来，VoxelMorph是无监督的，可以利用辅助信息，如分割结果进行训练，如果可用的话。

Two recent papers [40], [41], were the first to present unsupervised learning based image registration methods. Both propose a neural network consisting of a CNN and spatial transformation function [42] that warps images to one another. However, these two initial methods are only demonstrated on limited subsets of volumes, such as 3D subregions [41] or 2D slices [40], and support only small transformations [40].

两篇最近的文章第一个提出无监督学习的图像配准方法。他们都提出了一个神经网络，包含一个CNN和空间变换函数，使一幅图像经过变形，成为另一幅。但是，这两种方法的有效性只在有限的体上进行了证实，比如3D子区域或2D slices，而且只支持小的变换。

A recent method has proposed a segmentation driven cost function to be used in registering different imaging modalities – T2w MRI and 3D ultrasound – within the same subject [43], [44]. The authors demonstrate that a loss functions based solely on segmentation maps can lead to an accurate within-subject cross-modality registration network. Parallel to this work, in one of our experiments, we demonstrate the use of segmentation maps during training in subject-to-atlas registration. We provide an analysis of the effect of different anatomical label availability on overall registration quality, and evaluate how a combination of segmentation and image based losses behaves in various scenarios. We find that a segmentation-based loss can be helpful, for example if the input segment labels are the same as those we evaluate on (consistent with [43], and [44]). We also show that the image-based and smoothness losses are still necessary, especially when we evaluate registration accuracy on labels not observed during training, and to encourage deformation regularity.

一种最近的方法提出了一种分割驱动的损失函数，用于配准同一目标的不同模态图像，T2w MRI和3D超声。作者证明了，一种只基于分割图的损失函数，可以得到准确的目标内不同模态间的配准网络。与这个工作并行的是，在我们的一个试验中，我们证明了，在训练中使用分割图，在目标对模板的配准中是有效的。我们对不同解剖标签的可用性，对总体配准质量的影响提出了分析，评估了在不同场景中，分割和基于图像的损失的组合，表现的怎样。我们发现，基于分割的损失有帮助，比如，如果输入的分割标签与我们评估的是一样的（与[43,44]一样）。我们还证明了，基于图像损失和平滑损失都是必须的，尤其是当我们在训练中没有的标签上评估配准准确率时，以鼓励形变正则性。

### C. 2D Image Alignment

Optical flow estimation is a related registration problem for 2D images. Optical flow algorithms return a dense displacement vector field depicting small displacements between a pair of 2D images. Traditional optical flow approaches typically solve an optimization problem similar to (1) using variational methods [45]–[47]. Extensions that better handle large displacements or dramatic changes in appearance include feature-based matching [48], [49] and nearest neighbor fields [50].

光流估计对于2D图像来说是一种相关的配准问题。光流算法返回一个密集的偏移向量场，表示一对2D图像的小偏移值。传统的光流方法，一般使用变分方法求解与(1)类似的优化问题。更好的处理较大的偏移，或更大的变化的拓展，包括基于特征的匹配，和最近邻场。

In recent years, several learning-based approaches to optical flow estimation using neural networks have been proposed [51]–[56]. These algorithms take a pair of images as input, and use a convolutional neural network to learn image features that capture the concept of optical flow from data. Several of these works require supervision in the form of ground truth flow fields [52], [53], [55], [56], while we build on a few that use an unsupervised objective [51], [54]. The spatial transform layer enables neural networks to perform both global parametric 2D image alignment [42] and dense spatial transformations [54], [57], [58] without requiring supervised labels. An alternative approach to dense estimation is to use CNNs to match image patches [59]–[62]. These methods require exhaustive matching of patches, resulting in slow runtime.

最近几年中，提出了几种基于学习的，采用神经网络的光流估计方法。这些算法以一对图像作为输入，使用CNN来学习图像特征，从数据中捕获光流的概念。这些工作中的几个是需要监督的，其形式是真值光流场，而我们是构建在几种无监督的目标的方法之上的。这种空间变换层使神经网络可以进行全局参数化2D图像对齐，和密集空间变换，而不需要有监督的标签。一种密集估计的替代方法是，使用CNNs来匹配图像块。这种方法需要图像块的穷举式匹配，然后导致很慢的运行速度。

We build on these ideas and extend the spatial transformer to achieve n-D volume registration, and further show how leveraging image segmentations during training can improve registration accuracy at test time.

我们在这些概念的基础上，拓展了空间变换层，以得到n-D体配准的结果，进一步展示了，怎样在训练时利用图像分割来在测试时改进配准准确率。

## 4 Method

Let f, m be two image volumes defined over an n-D spatial domain Ω ⊂ $R^n$. For the rest of this paper, we focus on the case n = 3 but our method and implementation are dimension independent. For simplicity we assume that f and m contain single-channel, grayscale data. We also assume that f and m are affinely aligned as a preprocessing step, so that the only source of misalignment between the volumes is nonlinear. Many packages are available for rapid affine alignment.

令f,m是两个图像体，定义于n-D空间域Ω ⊂ $R^n$。本文的剩下部分，我们关注n=3的情况，但我们的方法和实现都是与维度无关的。简化起见，我们假设f和m都是单通道的灰度数据。我们还假设，f和m都是经过预处理进行仿射对齐过的，这样体之间没有对齐的唯一因素就是非线性的。很多软件包都可以进行快速仿射对齐。

We model a function $g_θ(f,m) = u$ using a convolutional neural network (CNN), where θ are network parameters, the kernels of the convolutional layers. The displacement field u between f and m is in practice stored in a n + 1-dimensional image. That is, for each voxel p ∈ Ω, u(p) is a displacement such that f(p) and [m◦φ] (p) correspond to similar anatomical locations, where the map φ = Id + u is formed using an identity transform and u.

我们使用CNN建模得到函数$g_θ(f,m) = u$，其中θ是网络参数，卷积层的核。f和m之间的偏移场u是存储在n+1维的图像中。那是，对于每个体素p ∈ Ω，u(p)是一个偏移，这样f(p)和[m◦φ] (p)对应着相似的解剖位置，其中映射φ = Id + u是使用恒等映射和u形成的。

Fig. 2 presents an overview of our method. The network takes f and m as input, and computes φ using a set of parameters θ. We warp m to m◦φ using a spatial transformation function, enabling evaluation of the similarity of m ◦ φ and f. Given unseen images f and m during test time, we obtain a registration field by evaluating $g_θ(f, m)$.

图2给出了我们方法的一个概览。网络以f和m作为输入，使用一个参数集θ计算φ。我们使用一个空间变形参数，将m变形为m◦φ，使得可以评估m ◦ φ和f的相似性。在测试时给定未曾见过的图像f和m，我们通过评估$g_θ(f, m)$得到配准场。

Fig. 2: Overview of the method. We learn parameters θ for a function $g_θ(·,·)$, and register 3D volume m to a second, fixed volume f. During training, we warp m with φ using a spatial transformer function. Optionally, auxiliary information such as anatomical segmentations $s_f$, $s_m$ can be leveraged during training (blue box).

We use (single-element) stochastic gradient descent to find optimal parameters $\hat θ$ by minimizing an expected loss function using a training dataset. We propose two unsupervised loss functions in this work. The first captures image similarity and field smoothness, while the second also leverages anatomical segmentations. We describe our CNN architecture and the two loss functions in detail in the next sections.

我们使用一个训练数据集，最小化一个期望的损失函数，使用（单元素）SGD找到最优参数$\hat θ$。我们在本文中提出两个无监督损失函数。第一个包含了图像相似度和场的平滑性，第二个还利用了解剖分割信息。我们在下一节描述了我们的CNN架构和两个损失函数。

### A. VoxelMorph CNN Architecture

In this section we describe the particular architecture used in our experiments, but emphasize that a wide range of architectures may work similarly well and that the exact architecture is not our focus. The parametrization of $g_θ(·,·)$ is based on a convolutional neural network architecture similar to UNet [63], [64], which consists of encoder and decoder sections with skip connections.

本节中，我们描述了在我们试验中使用的特定架构，但强调了，很多架构也可以得到类似好的结果，细节架构并不是我们关注的对象。$g_θ(·,·)$的参数化，是基于CNN架构的，与U-Net类似，是由含有skip连接的编码器与解码器部分。

Fig. 3 depicts the network used in VoxelMorph, which takes a single input formed by concatenating m and f into a 2-channel 3D image. In our experiments, the input is of size 160 × 192 × 224 × 2, but the framework is not limited by a particular size. We apply 3D convolutions in both the encoder and decoder stages using a kernel size of 3, and a stride of 2. Each convolution is followed by a LeakyReLU layer with parameter 0.2. The convolutional layers capture hierarchical features of the input image pair, used to estimate φ. In the encoder, we use strided convolutions to reduce the spatial dimensions in half at each layer. Successive layers of the encoder therefore operate over coarser representations of the input, similar to the image pyramid used in traditional image registration work.

图3描述了VoxelMorph中使用的网络，其单个输入是将m和f拼接成一个双通道3D图像得到的。在我们的试验中，输入的大小是160 × 192 × 224 × 2，但框架并没有受限于特定大小。我们在编码器和解码器阶段中都使用了3D卷积，卷积核的大小为3，步长为2。每个卷积都带有LeakyReLU，参数为0.2。卷积层捕获的输入图像对的层次化的特征，用于估计参数φ。在编码器中，我们使用带有步长的卷积，以在每层中降低空间维度到一半。后续的编码器因此会在输入的更粗糙的表示上进行运算，与传统图像配准工作中所用的图像金字塔类似。

Fig. 3: Convolutional UNet architecture implementing $g_θ(f,m)$. Each rectangle represents a 3D volume, generated from the preceding volume using a 3D convolutional network layer. The spatial resolution of each volume with respect to the input volume is printed underneath. In the decoder, we use several 32-filter convolutions, each followed by an upsampling layer, to bring the volume back to full resolution. Arrows represent skip connections, which concatenate encoder and decoder features. The full-resolution volume is further refined using several convolutions.

In the decoding stage, we alternate between upsampling, convolutions and concatenating skip connections that propagate features learned during the encoding stages directly to layers generating the registration. Successive layers of the decoder operate on finer spatial scales, enabling precise anatomical alignment. The receptive fields of the convolutional kernels of the smallest layer should be at least as large as the maximum expected displacement between corresponding voxels in f and m. In our architecture, the smallest layer applies convolutions over a volume (1/16)^3 of the size of the input images.

在解码阶段，我们交替进行上采样，卷积和拼接的skip连接，将编码器阶段学习到的特征直接传播到后续的层中，最后生成配准结果。解码器的后续的层中，是在更精细的空间尺度上进行运算的，可以得到更精确的解剖对齐。最小层的卷积核的感受野，应当至少与f和m之间对应体素之间的最大期望偏移一样大。在我们的架构中，最小的层在输入图像大小的(1/16)^3的体上进行运算。

### B. Spatial Transformation Function 空间变换函数

The proposed method learns optimal parameter values in part by minimizing differences between m◦φ and f. In order to use standard gradient-based methods, we construct a differentiable operation based on spatial transformer networks [42] to compute m ◦ φ.

我们提出的方法，部分通过最小化m◦φ和f之间的值，学习到最佳的参数值。为使用标准的基于梯度的方法，我们基于空间变换器网络[42]构建了一种可微分的运算，以计算m ◦ φ。

For each voxel p, we compute a (subpixel) voxel location p′ = p + u(p) in m. Because image values are only defined at integer locations, we linearly interpolate the values at the eight neighboring voxels:

对于每个体素p，我们在m中计算出一个（亚像素）的体素位置p′ = p + u(p)。因为图像值都是在整数位置上定义的，我们对这些值在8个相邻体素上进行线性插值：

$$m◦φ(p) = \sum_{q ∈ Z(p′)} m(q) \prod_{d∈\{x,y,z\}} (1-|p'_d-q_d|)$$(3)

where Z(p′) are the voxel neighbors of p′, and d iterates over dimensions of Ω. Because we can compute gradients or sub- gradients(The absolute value is implemented with a subgradient of 0 at 0), we can backpropagate errors during optimization.

其中Z(p′)是p′的相邻体素，d在Ω的维度中进行迭代。因为我们可以计算梯度或子梯度（在0位置中实现的子梯度的绝对值为0）。我们可以在优化的过程中对误差进行反向传播。

### C. Loss Functions

In this section, we propose two loss functions: an unsupervised loss $L_{us}$ that evaluates the model using only the input volumes and generated registration field, and an auxiliary loss $L_a$ that also leverages anatomical segmentations at training time.

本节中，我们提出两个损失函数：一个无监督损失$L_{us}$，只使用输入体和生成的配准场评估模型，和一个辅助损失$L_a$，在训练时也利用了解剖分割信息。

1) Unsupervised Loss Function: The unsupervised loss $L_{us}$(·, ·, ·) consists of two components: $L_{sim}$ that penalizes differences in appearance, and $L_{smooth}$ that penalizes local spatial variations in φ:

1) 无监督损失函数：无监督损失$L_{us}$(·, ·, ·)包含两部分，$L_{sim}$只对外观上的差异进行惩罚，$L_{smooth}$惩罚的是φ的局部空间变化：

$$L_{us}(f, m, φ) = L_{sim}(f, m ◦ φ) + λL_{smooth}(φ)$$(4)

where λ is a regularization parameter. We experimented with two often-used functions for $L_{sim}$. The first is the mean squared voxelwise difference, applicable when f and m have similar image intensity distributions and local contrast:

这里λ是一个正则化参数。我们使用两种常用的$L_{sim}$来进行试验。第一种是逐体素的均方差，在f和m有类似的图像灰度分布和局部对比度时可用：

$$MSE(f,m◦φ) = \frac{1}{|Ω|} \sum_{p∈Ω} [f(p)−[m◦φ](p)]^2$$(5)

The second is the local cross-correlation of f and m◦φ, which is more robust to intensity variations found across scans and datasets [11]. Let $\hat f(p)$ and $[\hat m ◦ φ](p)$ denote local mean intensity images: $\hat (p) = \frac {1}{n^3} \sum_{p_i} f(p_i))$, where $p_i$ iterates over a $n^3$ volume around p, with n = 9 in our experiments. The local cross-correlation of f and m ◦ φ is written as:

第二种是f和m◦φ的局部互相关，我们在数据集和scans之间发现，这对灰度变化更稳健。令$\hat f(p)$和$[\hat m ◦ φ](p)$表示局部平均灰度图像：$\hat (p) = \frac {1}{n^3} \sum_{p_i} f(p_i))$，其中$p_i$在p点附近的$n^3$大小的体中迭代，在我们的试验中n = 9。f和m ◦ φ的局部互相关可以写为：

$$CC(f, m ◦ φ) = \sum_{p∈Ω} \frac {(\sum_{p_i} (f(p_i)-\hat f(p)) ([m ◦ φ](p_i) − [\hat m ◦ φ](p)) )^2} {(\sum_{p_i} (f(p_i)-\hat f(p))^2) (\sum_{p_i} ([m ◦ φ](p_i) − [\hat m ◦ φ](p))^2)}$$(6)

A higher CC indicates a better alignment, yielding the loss function: $L_{sim}(f, m, φ) = −CC(f, m ◦ φ)$.

CC值越高，说明对齐的越好，所以我们使用的损失函数为$L_{sim}(f, m, φ) = −CC(f, m ◦ φ)$。

Minimizing $L_{sim}$ will encourage m ◦ φ to approximate f, but may generate a non-smooth φ that is not physically realistic. We encourage a smooth displacement field φ using a diffusion regularizer on the spatial gradients of displacement u:

对$L_{sim}$最小化，会鼓励m ◦ φ接近f，但会生成不平滑的φ，这在物理上是不平滑的。我们鼓励一个平滑的偏移场φ，使用的是对偏移u的空间梯度的扩散正则化器：

$$L_{smooth}(φ) = \sum_{p∈Ω} ||∇u(p)||^2$$(7)

and approximate spatial gradients using differences between neighboring voxels. Specifically, for $∇u(p) = (\frac {∂u(p)}{∂x}, \frac {∂u(p)}{∂y}, \frac {∂u(p)}{∂z})$, we approximate $\frac {∂u(p)}{∂x} ≈ u((p_x + 1,p_y,p_z)) − u((p_x,p_y,p_z))$, and use similar approximations for $\frac {∂u(p)}{∂y}$ and $\frac {∂u(p)}{∂z}$.

使用相邻体素的差异近似空间梯度。具体的，对于$∇u(p) = (\frac {∂u(p)}{∂x}, \frac {∂u(p)}{∂y}, \frac {∂u(p)}{∂z})$，我们近似$\frac {∂u(p)}{∂x} ≈ u((p_x + 1,p_y,p_z)) − u((p_x,p_y,p_z))$，并对$\frac {∂u(p)}{∂y}$和$\frac {∂u(p)}{∂z}$使用类似的近似。

2) Auxiliary Data Loss Function: Here, we describe how VoxelMorph can leverage auxiliary information available during training but not during testing. Anatomical segmentation maps are sometimes available during training, and can be annotated by human experts or automated algorithms. A segmentation map assigns each voxel to an anatomical structure. If a registration field φ represents accurate anatomical correspondences, the regions in f and m ◦ φ corresponding to the same anatomical structure should overlap well.

2) 辅助数据损失函数：这里，我们描述VoxelMorph怎样在训练中利用可用的辅助信息，而不用在测试时利用这些信息。解剖分割图在训练时有时候是可用的，而且可以由人类专家或自动算法进行标注。分割图将每个体素指定为一个解剖结构。如果一个配准场φ表示准确的解剖对应性，则f中的区域和m ◦ φ中的区域对应着相同的解剖结构，应当重叠的很好。

Let $s^k_f, s^k_m ◦φ$ be the voxels of structure k for f and m◦φ, respectively. We quantify the volume overlap for structure k using the Dice score [65]:

令$s^k_f, s^k_m ◦φ$分别为f和m◦φ的结构k的体素。我们使用Dice分数来量化结构k的体积重叠：

$$Dice(s^k_f, s^k_m ◦φ) = 2⋅ \frac {|s^k_f ∩ s^k_m ◦φ|} {|s^k_f|+|s^k_m ◦φ|}$$(8)

A Dice score of 1 indicates that the anatomy matches perfectly, and a score of 0 indicates that there is no overlap. We define the segmentation loss $L_seg$ over all structures k ∈ [1, K] as:

Dice分数为1，说明解剖结构完美匹配，分数为0则说明，没有重叠部分。我们在所有结构k ∈ [1, K]上定义分割损失$L_seg$为：

$$L_{seg}(s_f, s_m ◦φ) = −\frac {1}{K} \sum_1^K Dice(s^k_f, s^k_m ◦φ)$$(9)

$L_{seg}$ alone does not encourage smoothness and agreement of image appearance, which are essential to good registration. We therefore combine $L_{seg}$ with (4) to obtain the objective:

$L_{seg}$本身并不鼓励平滑性和图像外观的一致性，而这些才是对配准最主要的约束。因此我们将$L_{seg}$与(4)结合到一起，以得到目标：

$$L_a(f,m,s_f,s_m,φ) = L_{us}(f,m,φ)+γL_{seg}(s_f,s_m ◦φ)$$(10)

where γ is a regularization parameter. 其中γ是正则化参数。

In our experiments, which use affinely aligned images, we demonstrate that loss (10) can lead to significant improvements. In general, and depending on the task, this loss can also be computed in a multiscale fashion as introduced in [43], depending on quality of the initial alignment.

在我们的试验中使用的是仿射对齐过的图像，我们证明了损失(10)可以带来显著的改进。总体上，根据任务的不同，这种损失可以以多尺度的形式进行计算，如[43]中提出的，这还要依赖于初始对齐质量的不同。

Since anatomical labels are categorical, a naive implementation of linear interpolation to compute $s_m ◦ φ$ is inappropriate, and a direct implementation of (8) might not be amenable to auto-differentiation frameworks. We design $s_f$ and $s_m$ to be image volumes with K channels, where each channel is a binary mask specifying the spatial domain of a particular structure. We compute $s_m ◦ φ$ by spatially transforming each channel of $s_m$ using linear interpolation. We then compute the numerator and denominator of (8) by multiplying and adding $s_f$ and $s_m ◦ φ$, respectively.

因为解剖标签是类别式的，用线性插值的简单实现来计算$s_m ◦ φ$是不合适的，而直接实现的(8)式对于自动微分架构来说可能不是可行的。我们设计$s_f$和$s_m$是有K个通道的图像体，其中每个通道是一个二值掩模，指定特定结构的空间区域。我们通过使用线性插值对$s_m$的每个通道来进行空间形变，来计算$s_m ◦ φ$。我们然后通过$s_f$和$s_m ◦ φ$的相乘和相加，分别计算(8)式的分子和分母。

### D. Amortized Optimization Interpretation

Our method substitutes the pair-specific optimization over the deformation field φ with a global optimization of function parameters θ for function $g_θ(·,·)$. This process is sometimes referred to as amortized optimization [66]. Because the function $g_θ(·,·)$ is tasked with estimating registration between any two images, the fact that parameters θ are shared globally acts as a natural regularization. We demonstrate this aspect in Section V-C (Regularization Analysis). In addition, the quality and generalizability of the deformations outputted by the function will depend on the data it is trained on. Indeed, the resulting deformation can be interpreted as simply an approximation or initialization to the optimal deformation $φ^∗$, and the resulting difference is sometimes referred to as the amortization gap [15], [66]. If desired, this initial deformation field could be improved using any instance-specific optimization. In our experiments, we accomplish this by treating the resulting displacement u as model parameters, and fine-tuning the deformation for each particular scan independently using gradient descent. Essentially, this implements an auto-differentiation version of conventional registration, using VoxelMorph output as initialization. However, most often we find that the initial deformation, the VoxelMorph output, is already as accurate as state of the art results. We explore these aspects in experiments presented in Section V-D.

我们的方法，将对形变场φ的具体对的优化，替换成对函数$g_θ(·,·)$的参数θ的全局优化。这个过程有些时候被称为amortized优化。因为函数$g_θ(·,·)$的任务是，估计任意两幅图像间的配准，那么参数θ肯定是全局共享的，这本身就是一个很自然的正则化。我们在V-C部分证明了这一点（正则化分析）。另外，这个函数输出的形变的质量和泛化性，依赖于其训练集的数据。确实，得到的形变可以解释为，最优形变$φ^∗$的的一个近似，或初始化，得到的差异有时候称为amortization间隙。如果很理想的话，初始的形变场，可以用任何实例具体的优化来改进。在我们的试验中，我们通过将得到的偏移u作为模型参数来完成这一目标，并对每个特定的scan独立的使用梯度下降来精调形变。本质上，这个实现的是传统配准的自动微分版本，使用VoxelMorph的输出作为初始化结果。但是，多数情况下我们发现，初始形变，即VoxelMorph输出，已经与目前最好的结果一样精确了。我们在V-D中的试验，探索了这些方面。

## 5. Experiments

We demonstrate our method on the task of brain MRI registration. We first (Section V-B) present a series of atlas-based registration experiments, in which we compute a registration field between an atlas, or reference volume, and each volume in our dataset. Atlas-based registration is a common formulation in population analysis, where inter-subject registration is a core problem. The atlas represents a reference, or average volume, and is usually constructed by jointly and repeatedly aligning a dataset of brain MR volumes and averaging them together [67]. We use an atlas computed using an external dataset [1], [68]. Each input volume pair consists of the atlas (image f) and a volume from the dataset (image m). Fig. 4 shows example image pairs using the same fixed atlas for all examples. In a second experiment (Section V-C), we perform hyper-parameter sensitivity analysis. In a third experiment (Section V-D), we study the effect of training set size on registration, and demonstrate instance-specific optimization. In the fourth experiment (Section V-E) we present results on a dataset that contains manual segmentations. In the next experiment (Section V-F), we train VoxelMorph using random pairs of training subjects as input, and test registration between pairs of unseen test subjects. Finally (Section V-G), we present an empirical analysis of registration with auxiliary segmentation data. All figures that depict brains in this paper show 2D slices, but all registration is done in 3D.

我们在脑MRI配准任务中证明了我们方法的有效性。我们首先在V-B中给出了基于atlas的一系列配准试验，其中我们在一个atlas，或称参考体，与我们数据集之间的每个体之间计算得到一个配准场。基于atlas的配准在population分析中是一种常见的表述，其中目标间的配准是一个核心问题。Atlas表示一个参考或平均体，通常通过联合并重复对脑MRI体数据集进行对齐，并对其平均得到。我们使用外部数据集计算得到的atlas。每个输入体对由atlas（图像f）和数据集中的一个体（图像m）组成。图4展示了图像对的样本，对所有样本使用的是同样的固定atlas。在第二个试验中（V-C小节），我们进行了超参数敏感性分析。在第三个试验（V-D）中，我们研究了训练集大小对配准的效果，并证明了具体案例优化的效果。在第四个试验（V-E小节）中，我们在一个包含手工分割结果的数据集上给出了结果。在下一个试验（V-F小节）中，我们使用训练目标的随机对作为输入来训练VoxelMorph，并在未曾见过的测试目标上测试配准效果。最后（V-G小节），我们提出了采用辅助分割数据的配准的经验性分析。本文中的所有的图像都展示的是2D slices的脑部结构，但所有配准都是在3D中进行的。

Fig. 4: Example MR coronal slices extracted from input pairs (columns 1-2), and resulting m ◦ φ for VoxelMorph using different loss functions. We overlaid boundaries of a few structures: ventricles (blue/dark green), thalami (red/pink), and hippocampi (light green/orange). A good registration will cause structures in m◦φ to look similar to structures in f. Our models are able to handle various changes in shape of structures, including expansion/shrinkage of the ventricles in rows 2 and 3, and stretching of the hippocampi in row 4.

### A. Experimental Setup 试验设置

1) Dataset: We use a large-scale, multi-site, multi-study dataset of 3731 T1–weighted brain MRI scans from eight publicly available datasets: OASIS [69], ABIDE [70], ADHD200 [71], MCIC [72], PPMI [73], HABS [74], Harvard GSP [75], and the FreeSurfer Buckner40 [1]. Acquisition details, subject age ranges and health conditions are different for each dataset. All scans were resampled to a 256×256×256 grid with 1mm isotropic voxels. We carry out standard pre-processing steps, including affine spatial normalization and brain extraction for each scan using FreeSurfer [1], and crop the resulting images to 160 × 192 × 224. All MRIs were anatomically segmented with FreeSurfer, and we applied quality control using visual inspection to catch gross errors in segmentation results and affine alignment. We include all anatomical structures that are at least 100 voxels in volume for all test subjects, resulting in 30 structures. We use the resulting segmentation maps in evaluating our registration as described below. We split our dataset into 3231, 250, and 250 volumes for train, validation, and test sets respectively, although we highlight that we do not use any supervised information at any stage. In addition, the Buckner40 dataset is only used for testing, using manual segmentations.

1) 数据集：我们使用一个大规模、多部位、多study的数据集，有3731幅T1-加权的脑MRI扫描，从8个公开可用的数据集中来：OASIS [69], ABIDE [70], ADHD200 [71], MCIC [72], PPMI [73], HABS [74], Harvard GSP [75] 和 the FreeSurfer Buckner40 [1]。对于每个数据集，获取的细节，对象年龄范围和健康状况都不同。所有scans重采样成256×256×256网格大小，1mm各向同性的体素。我们进行标准的预处理步骤，包括使用FreeSurfer对每个scan进行仿射空间归一化和脑部提取，并将得到的图像剪切成160 × 192 × 224。所有MRIs都用FreeSurfer进行解剖分割，我们采用视觉检查来得到分割结果和仿射对齐的大致误差。我们对于所有的测试对象，都包含了超过100个体素的解剖结构，得到了30个结构。我们使用得到的分割图，评估我们配准结果，下面有详述。我们将我们的数据集分别分割成训练集，验证集和测试集，各有3231、250、250个体，虽然我们强调了，我们在任何阶段，都不使用任何监督信息。另外，Buckner40数据集只用于测试，使用的是手工分割结果。

2) Evaluation Metrics: Obtaining dense ground truth registration for these data is not well-defined since many registration fields can yield similar looking warped images. We first evaluate our method using volume overlap of anatomical segmentations. If a registration field φ represents accurate correspondences, the regions in f and m ◦ φ corresponding to the same anatomical structure should overlap well (see Fig. 4 for examples). We quantify the volume overlap between structures using the Dice score (8). We also evaluate the regularity of the deformation fields. Specifically, the Jacobian matrix $J_φ(p) = ∇φ(p) ∈ R^{3×3}$ captures the local properties of φ around voxel p. We count all non-background voxels for which |$J_φ(p)$| ≤ 0, where the deformation is not diffeomorphic [16].

2) 评估标准度量：对这些数据得到密集真值配准，这不是一个定义明确的问题，因为很多配准场会得到类似的变形图像。我们首先使用解剖结构的体重叠，来评估我们的方法。如果一个配准场φ代表精确的对应，那么f和m ◦ φ对应着相同的解剖结构的区域应当重叠的很好（如图4中的例子）。我们使用Dice来量化结构间的体重叠。我们还会评估形变场的规则性。具体的，我们使用Jacobian矩阵$J_φ(p) = ∇φ(p) ∈ R^{3×3}$来捕获体素p附近的局部性质。我们对|$J_φ(p)$| ≤ 0的所有非背景体素进行计数，其中形变不是微分同胚的。

3) Baseline Methods: We use Symmetric Normalization (SyN) [11], the top-performing registration algorithm in a comparative study [2] as a first baseline. We use the SyN implementation in the publicly available Advanced Normalization Tools (ANTs) software package [3], with a cross-correlation similarity measure. Throughout our work with medical images, we found the default ANTs smoothness parameters to be sub-optimal for applying ANTs to our data. We obtained improved parameters using a wide parameter sweep across multiple datasets, and use those in these experiments. Specifically, we use SyN step size of 0.25, Gaussian parameters (9, 0.2), at three scales with at most 201 iterations each. We also use the NiftyReg package, as a second baseline. Unfortunately, a GPU implementation is not currently available, and instead we build a multi-threaded CPU version. We searched through various parameter settings to obtain improved parameters, and use the CC cost function, grid spacing of 5, and 500 iterations.

3) 基准方法：我们使用对称归一化(Symmetric Normalization, SyN)作为第一个基准，这是一个比较研究中性能最好的配准算法。我们使用ANTs软件包中的SyN实现，并使用交叉相关的相似性度量。在我们对医学图像的工作中，我们发现默认的ANTs平滑性参数应用到我们的数据中时，并不是最优的。我们在多个数据集中进行了宽泛的参数sweep，得到了改进的参数，并在我们的试验中进行了使用。具体的，我们使用SyN的步长为0.25，Gaussian参数为(9,0.2)，在三个尺度上，每个尺度最多迭代201次。我们还使用了NiftyReg软件包作为第二种基准。不幸的是，GPU实现目前不可用，我们构建了一个多线程的CPU版本。我们搜索了各种参数设置，以得到改进的参数，使用了CC代价函数，网格间隔5，500次迭代。

4) VoxelMorph Implementation: We implemented our method using Keras [76] with a Tensorflow backend [77]. We extended the 2D linear interpolation spatial transformer layer to n-D, and here use n = 3. We use the ADAM optimizer [78] with a learning rate of $10^{-4}$. While our implementation allows for mini-batch stochastic gradient descent, in our experiments each training batch consists of one pair of volumes. Our implementation includes a default of 150,000 iterations. Our code and model parameters are available online at http://voxelmorph.csail.mit.edu.

4) VoxelMorph实现：我们使用TensorFlow backend Keras实现了我们的方法。我们将2D线性插值spatial transformer层拓展到了n-D，这里我们使用的是n=3。我们使用ADAM优化器，学习率为$10^{-4}$。我们的实现可以进行mini-batch SGD，在我们的试验中，每个训练batch由一对体对构成。我们的实现默认是15万次迭代。我们的代码和模型参数已开源。

### B. Atlas-based Registration 基于atlas的配准

In this experiment, we train VoxelMorph for atlas-based registration. We train separate VoxelMorph networks with different λ regularization parameters. We then select the network that optimizes Dice score on our validation set, and report results on our test set.

本试验中，我们训练VoxelMorph进行基于atlas的配准。我们用不同的λ正则化参数来训练不同的VoxelMorph网络。我们然后选择在验证集上优化Dice分数的网络，并在我们的测试集中给出结果。

Table I presents average Dice scores computed for all subjects and structures for baselines of only global affine alignment, ANTs, and NiftyReg, as well as VoxelMorph with different losses. VoxelMorph variants perform comparably to ANTs and NiftyReg in terms of Dice(Both VoxelMorph variants are different from ANTs with paired t-test p-values of 0.003 and 0.008 and with slightly higher Dice values. There is no difference between VoxelMorph (CC) and NiftyReg (p-value of 0.21), and no significant difference between VoxelMorph (CC) and VoxelMorph (MSE) (p-value of 0.09)), and are significantly better than affine alignment. Example visual results of the warped images from our algorithms are shown in Figs. 4 and 6. VoxelMorph is able to handle significant shape changes for various structures.

表1给出了对所有对象和结构，采用只有全局仿射对齐，ANTs和NiftyReg，和不同损失的VoxelMorph，各种方法的平均Dice分数。VoxelMorph的各种变体与ANTs和NiftyReg在Dice中表现类似（两个VoxelMorph的变体在paired t-test p-values 0.003和0.008时其Dice值更高。在VoxelMorph(CC)和NiftyReg（p-value为0.21）之间没有差异，VoxelMorph (CC)和VoxelMorph (MSE)在p-value为0.09时也没有很大差异），比仿射对齐明显要好。我们算法的变形图像的视觉效果的例子，如图4和图6所示。VoxelMorph对于各种结构的明显形状变化都可以处理。

Table I: Average Dice scores and runtime results for affine alignment, ANTs, NiftyReg and VoxelMorph for the first experiment. Standard deviations across structures and subjects are in parentheses. The average Dice score is computed over all structures and subjects. Timing is computed after preprocessing. Our networks yield comparable results to ANTs and NiftyReg in Dice score, while operating orders of magnitude faster during testing. We also show the number and percentage of voxels with a non-positive Jacobian determinant for each method, for our volumes with 5.2 million voxels within the brain. All methods exhibit less than 1 percent such voxels.

Method | Dice | GPU sec | CPU sec | $|J_φ|≤0$ | % of $|J_φ|≤0$
--- | --- | --- | --- | --- | ---
Affine only | 0.584 (0.157) | 0 | 0 | 0 | 0
ANTs SyN (CC) | 0.749(0.136) | ~ | 9059(2023) | 9662(6258) | 0.140(0.091)
NiftyReg (CC) | 0.755(0.143) | ~ | 2347(202) | 41251(14336) | 0.600(0.208)
VoxelMorph (CC) | 0.753(0.145) | 0.45(0.01) | 57(1) | 19077(5928) | 0.366(0.114)
VoxelMorph (MSE) | 0.752(0.140) | 0.45(0.01) | 57(1) | 9606(4516) | 0.184(0.087)

Fig. 5 presents the Dice scores for each structure as a boxplot. For ease of visualization, we average Dice scores of the same structures from the two hemispheres into one score, e.g., the left and right hippocampi scores are averaged. The VoxelMorph models achieve comparable Dice measures to ANTs and NiftyReg for all structures, performing slightly better on some structures such as the lateral ventricles, and worse on others such as the hippocampi.

图5给出了每个结构的Dice分数，画成了柱状图。为可视化简单，我们将两个半球中相同结构的左右两部分进行平均，得到一个分数，如左海马和右海马的分数进行了平均。VoxelMorph模型对所有结构，与ANTs和NiftyReg得到了类似的Dice分数，在一些结构，如侧心室，表现略好，在其他的一些，如海马上表现略差。

Table I includes a count of voxels for which the Jacobian determinant is non-positive. We find that all methods result in deformations with small islands of such voxels, but are diffeomorphic at the vast majority of voxels (99.4% - 99.9%). Figs. 6 and Fig. 11 in the supplemental material illustrate several example VoxelMorph deformation fields. VoxelMorph has no explicit constraint for diffeomorphic deformations, but in this setting the smoothness loss leads to generally smooth and well-behaved results. ANTs and NiftyReg include implementations that can enforce or strongly encourage diffeomorphic deformations, but during our parameter search these negatively affected runtime or results. In this work, we ran the baseline implementations with configurations that yielded the best Dice scores, which also turned out to produce good deformation regularity.

表I包含了Jacobian行列式值非负的体素的数量。我们发现，所有的方法都会得到有一些孤立体素的形变的集合，但在绝大部分体素(99.4% - 99.9%)都是diffeomorphic的。图6和补充资料中的图11，描述了VoxelMorph形变场的一些例子。VoxelMorph对于diffeomorphic的变形，有隐式的约束，但在这种设置中，平滑性约束一般会带来平滑的、表现很好的结果。ANTs和NiftyReg中的实现，可以增强或强烈鼓励diffeomorphic形变，但在我们的参数搜索过程中，这些对运行时间或最后结果有负面影响。在本文中，我们运行的这些基准实现，其配置都是能得到最好的Dice分数的，其也能产生好的形变规则性。

**1) Runtime**: Table I presents runtime results using an Intel Xeon (E5-2680) CPU, and a NVIDIA TitanX GPU. We report the elapsed time for computations following the affine alignment preprocessing step, which all of the presented methods share, and requires just a few minutes even on a CPU. ANTs requires two or more hours on the CPU, while NiftyReg requires roughly 39 minutes for the given setting. ANTs runtimes vary widely, as its convergence depends on the difficulty of the alignment task. Registering two images with VoxelMorph is, on average, 150 times faster on the CPU compared to ANTs, and 40 times faster than NiftyReg. When using the GPU, VoxelMorph computes a registration in under a second. To our knowledge, there is no publicly available ANTs implementation for GPUs. It is likely that the SyN algorithm would benefit from a GPU implementation, but the main advantage of VoxelMorph comes from not requiring an optimization on each test pair, as can be seen in the CPU comparison. Unfortunately, the NiftyReg GPU version is unavailable in the current source code on all available repository history.

**1) 运行时间**：表1给出了使用Intel Xeon (E5-2680) CPU和NVIDIA TitanX GPU的运行时间结果。仿射对齐的预处理过程是所有给出的方法都共享的，我们据此给出计算持续时间，在CPU上只需要几分钟。ANTs在CPU上需要2个小时或更多，而NiftyReg在给定的设置中需要大约39分钟。ANTs的运行时间变化很大，因为其收敛性依赖于对齐任务的难度。将两幅图像用VoxelMorph进行配准，在CPU上平均比ANTs快150倍，比NiftyReg快40倍。当使用GPU时，VoxelMorph计算配准在一秒以内。据我们所知，ANTs的实现并没有公开的GPU版的。SyN算法很可能会受益于GPU实现，但VoxelMorph的主要优势来自于，不需要对每个测试对进行优化，这可以在CPU比较中看到。不幸的是，NiftyReg GPU版目前并不可用。

Fig. 5: Boxplots of Dice scores for various anatomical structures for ANTs, NiftyReg, and VoxelMorph results for the first (unsupervised) experiment. We average Dice scores of the left and right brain hemispheres into one score for this visualization. Structures are ordered by average ANTs Dice score.

### C. Regularization Analysis 正则化分析

Fig. 7 shows average Dice scores for the validation set for different values of the smoothness regularization parameter λ. The results vary smoothly over a large range of λ values, illustrating that our model is robust to choice of λ. Interestingly, even setting λ = 0, which enforces no explicit regularization on registration, results in a significant improvement over affine registration. This is likely because the optimal network parameters θ need to register all pairs in the training set well, yielding an implicit dataset regularization for the function $g_θ(·, ·)$.

图7给出了在不同平滑性正则化参数λ值的情况下，验证集得到的不同平均dice分数。在λ值的很大范围内，结果变化很平缓，说明我们的模型对λ的选择很稳健。有趣的是，即使设λ=0，即没有对配准给出显式的正则化，与仿射配准相比，也会有显著的改进。这很可能是因为，最优网络参数θ需要对训练集中的所有体对进行配准，对函数$g_θ(·, ·)$有隐式的数据集正则化。

Fig. 7: Dice score of validation data for VoxelMorph with varied regularization parameter λ.

### D. Training Set Size and Instance-Specific Optimization

We evaluate the effect of training set size on accuracy, and the relationship between amortized and instance-specific optimization. Because MSE and CC performed similarly for atlas-based registration, in this section we use MSE. We train VoxelMorph on subsets of different sizes from our training dataset, and report Dice scores on: (1) the training subset, (2) the held out test set, and (3) the test set when each deformation is further individually optimized for each test image pair. We perform (3) by fine-tuning the displacements u obtained from VoxelMorph using gradient descent for 100 iterations on each test pair, which took 23.7 ± 0.4 seconds on the GPU or 628.0 ± 4.2 seconds on a single-threaded CPU.

我们评估训练集对准确率的影响，以及amortized优化和具体实例相关的优化之间的关系。因为MSE和CC对基于atlas的配准表现类似，本节中我们使用MSE。从我们的训练集中，我们取出不同大小的子集，来训练VoxelMorph，并在如下集中给出Dice分数：(1)训练子集，(2)保留的测试集，(3)测试集，对每个测试图像对，进行了独立的优化，得到各自的形变。我们通过对从VoxelMorph得到的偏移u，使用梯度下降在每个测试对上迭代100次，这样进行精调，这在GPU上会耗费23.7 ± 0.4s，在单线程CPU上耗费628.0 ± 4.2s。

Fig. 8 presents our results. A small training set size of 10 scans results in slightly lower train and test Dice scores compared to larger training set sizes. However, there is no significant difference in Dice scores when training with 100 scans or the full dataset. Further optimizing the VoxelMorph parameters on each test image pair results in better test Dice scores regardless of training set size, comparable to the state-of-the-art.

图8给出了我们的结果。10 scans的小型训练集大小，与更大的训练集大小比较起来，会得到较低的训练和测试dice分数。但是，训练集100 scans或完整数据集，其dice分数并没有显著差异。在每个测试图像对上，进一步优化VoxelMorph参数，会得到更好的测试dice分数，不管训练集大小是多少，与目前最好的结果是类似的。

Fig. 8: Effect of training set size on accuracy. Also shown are results of instance-specific optimization of deformations, after these are initialized with VoxelMorph outputs using the optimal global parameters resulting from the training phase.

### E. Manual Anatomical Delineations 解剖结构的手动勾画

Since manual segmentations are not available for most datasets, the availability of FreeSurfer segmentations enabled the broad range of experiments above. In this experiment, we use VoxelMorph models already trained in Section V-B to test registration on the (unseen) Buckner40 dataset containing 39 scans. This dataset contains expert manual delineations of the same anatomical structures used in previous experiments, which we use here for evaluation. We also compute VoxelMorph with instance-specific optimization, as described in Section V-D. The Dice score results, shown in Table II, show that VoxelMorph using cross-correlation loss behaves comparably to ANTs and NiftyReg using the same cost function, consistent with the first experiment where we evaluated on FreeSurfer segmentations. VoxelMorph with instance-specific optimization further improves the results, similar to the previous experiment. On this dataset, results using VoxelMorph with MSE loss obtain slightly lower scores, but are improved by the instance-specific optimization procedure to be comparable to ANTs and NiftyReg.

由于手动勾画在多数数据集中并不可用，FreeSurfer分割的可用性使得多数试验成为可能。在本试验中，我们使用的VoxelMorph模型，是已经在V-B小节中训练好的，在模型未曾见过的Buckner40数据集上的39个scans上测试配准效果。这种数据集包含专家的手动勾画，勾画的解剖结构与之前的试验中使用的一样，我们在这里用作评估。我们还对instance-specific的优化也计算了VoxelMorph，如V-D小节所述。表2中给出的Dice分数结果表明，使用交叉相关损失的VoxelMorph与使用相同损失函数的ANTs和NiftyReg性能相似，与在第一个试验中，我们评估FreeSurfer分割的情况类似。进行instance-specific优化的VoxelMorph进一步改进了结果，与之前的试验一样。在这个数据集中，使用MSE损失的VoxelMorph的结果，其分数略低，但通过instance-specific优化过程可以改进，改进后性能与ANTs和NiftyReg是可媲美的。

TABLE II: Results for manual annotation experiment. We show affine, ANTs, NiftyReg, and VoxelMorph, where “inst.” indicates additional instance-specific optimization, as de- scribed in Section V-D. The average Dice score is computed over all structures and subjects, with standard deviations across structures and subjects in parentheses.

Method | Dice
--- | ---
Affine only | 0.608 (0.175)
ANTs SyN (CC) | 0.776 (0.130)
NiftyReg (CC) | 0.776 (0.132)
VoxelMorph (MSE) | 0.766 (0.133)
VoxelMorph (MSE) inst. | 0.776 (0.132)
VoxelMorph (CC) | 0.774 (0.133)
VoxelMorph (CC) inst. | 0.786 (0.132)

### F. Subject-to-Subject Registration 对象对对象的配准

In this experiment, we train VoxelMorph for subject-to-subject registration. Since there is more variability in each registration, we double the number of features for each network layer. We also compute VoxelMorph with instance-specific optimization, as described in Section V-D. Table III presents average test Dice scores on 250 randomly selected test pairs for registration. Consistent with literature, we find that the normalized cross correlation loss leads to more robust results compared to using the MSE loss. VoxelMorph (with doubled feature counts) Dice scores are comparable with ANTs and slightly below NiftyReg, while results from VoxelMorph with instance-specific optimization are comparable to both baselines.

本试验中，我们训练VoxelMorph进行对象对对象的配准。由于在每个配准中有更多变化，我们对每层网络的特征数量进行加倍。我们还用instance-specific优化来计算VoxelMorph，如V-D小节所述。表III在250个随机选择的配准测试对中给出平均测试dice分数。与其他文献一致，我们发现归一化互相关损失与MSE损失相比，会得到更稳健的结果。VoxelMorph（特征数量加倍）Dice分数与ANTs可类比，比NiftyReg略低，而有instance-specific优化的VoxelMorph与两个基准都是可比的。

TABLE III: Results for subject-to-subject alignment using affine, ANTs, and VoxelMorph variants, where “x2” refers to a model where we doubled the number of features to account for the increased inherent variability of the task, and “inst.” indicates additional instance-specific optimization.

Method | Dice
--- | ---
Affine only | 0.579 (0.173)
ANTs SyN (CC) | 0.761 (0.117)
NiftyReg (CC) | 0.772 (0.117)
VoxelMorph (MSE) | 0.727 (0.146)
VoxelMorph x2 (MSE) | 0.750 (0.058)
VoxelMorph x2 (MSE) inst. | 0.764 (0.048)
VoxelMorph (CC) | 0.737 (0.139)
VoxelMorph x2 (CC) | 0.763 (0.049)
VoxelMorph x2 (CC) inst. | 0.772 (0.119)

### G. Registration with Auxiliary Data 带有辅助数据的配准

In this section, we evaluate VoxelMorph when using segmentation maps during training with loss function (10). Because MSE and CC performed similarly for atlas-based registration, in this section we use MSE with λ = 0.02. We present an evaluation of our model in two practical scenarios: (1) when subsets of anatomical structure labels are available during training, and (2) when coarse segmentations labels are available during training. We use the same train/validation/test split as the previous experiments.

本节中，我们在训练时使用分割图，使用损失函数(10)，评估VoxelMorph。因为MSE和CC在基于atlas的配准上表现类似，本节中我们使用MSE，λ = 0.02。我们在两种实用场景中评估我们的模型：(1)在训练时，解剖结构标签的子集是可用的；(2)在训练时，粗糙的分割标签是可用的。我们使用的训练/验证/测试分割与之前的试验相同。

**1) Training with a subset of anatomical labels**: In many practical settings, it may be infeasible to obtain training segmentations for all structures. We therefore first consider the case where segmentations are available for only a subset of the 30 structures. We refer to structures present in segmentations as observed, and the rest as unobserved. We considered three scenarios, when: one, 15 (half), and 30 (all) structure segmentations are observed. The first two experiments essentially simulate different amounts of partially observed segmentations. For each experiment, we train separate models on different subsets of observed structures, as follows. For single structure segmentations, we manually selected four important structures for four folds (one for each fold) of the experiment: hippocampi, cerebral cortex, cerebral white matter, and ventricles. For the second experiment, we randomly selected 15 of the 30 structures, with a different selection for each of five folds. For each fold and each subset of observed labels, we use the segmentation maps at training, and show results on test pairs where segmentation maps are not used.

**1)用解剖标签的子集进行训练**：在很多实际的设置中，是可以得到所有结构的训练分割的。因此我们首先考虑这种情况，即有30个结构的子集是可用的。我们称分割中给出的解剖是观察到的，剩下的为未观察到的。我们考虑下面三种场景：即当1个、15个（一半）和30个（所有）结构是观察到的情况。前两个试验是在模拟不同数量的部分观察到的分割。对于每个试验，我们在不同的观察到的结构的子集训练不同的模型，如下。对于单个结构的分割，我们手动选择试验中4个重要的结构进行four-fold（每个fold一个）：hippocampi, cerebral cortex, cerebral white matter, and ventricles。对于第二个试验，我们随机选择30个结构中的15个，对five fold中的每一个都进行不同的选择。对每个fold和观察到的标签的每个子集，我们在训练时使用分割，在没有使用分割图的测试对上给出结果。

Fig. 9a-c shows Dice scores for both the observed and unobserved labels when sweeping γ in (10), the auxiliary regularization trade-off parameter. We train our models with FreeSurfer annotations, and show results on both the general test set using FreeSurfer annotations (top) and the Buckner40 test set with manual annotations (bottom). The extreme values γ=0 (or logγ=−∞) and γ=∞ serve as theoretical extremes, with γ = 0 corresponding to unsupervised VoxelMorph, and γ = ∞ corresponding to VoxelMorph trained only with auxiliary labels, without the smoothness and image matching objective terms.

图9a-c给出的是各种试验中的dice分数，改变的是(10)中的γ，即辅助正则化折中参数，对于观察到的和未观察到的标签都给出了结果。我们用FreeSurfer标注来训练我们的模型，用FreeSurfer标注在一般测试集上给出结果（上），用手工标注在Buckner40测试集中给出结果（下）。极端值γ=0 (或logγ=−∞)和γ=∞是理论极限，γ = 0对应着无监督VoxelMorph，γ = ∞对应着只用辅助标签训练的VoxelMorph，没有平滑项和图像配准目标。

In general, VoxelMorph with auxiliary data significantly outperforms (largest p-value < 10^−9 among the four settings) unsupervised VoxelMorph (equivalent to γ = 0 or logγ = −∞) and ANTs on observed structures in terms of Dice score. Dice score on observed labels generally increases with an increase in γ.

总体上，用辅助数据训练的VoxelMorph明显超过了无监督的VoxelMorph，，也超过了ANTs，评价指标是在Dice分数中的观察到的结构。在观察到的标签中，Dice分数一般随着γ的增大而增加。

Fig. 9: Results on test scans when using auxiliary data during training. Top: testing on the FreeSurfer segmentation of the general test set. Bottom: testing the same models on the manual segmentation of the Buckner40 test set. We test having varying number of observed labels (a-c), and having coarser segmentation maps (d). Error bars indicate standard deviations across subjects. The leftmost datapoint in each graph for all labels, corresponding to γ = 0, indicates results of VoxelMorph without using auxiliary data (unsupervised). γ = ∞ is achieved by setting the image and smoothness terms to 0. We show Dice scores for results from ANTs with optimal parameters, which does not use segmentation maps, for comparison.

Interestingly, VoxelMorph (trained with auxiliary data) yields improved Dice scores for unobserved structures compared to the unsupervised variant for a range of γ values (see Fig. 9a-b), even though these segmentations were not explicitly observed during training. When all structures that we use during evaluation are observed during training, we find good Dice results at higher γ values (Fig 9c.). Registration accuracy for unobserved structures starts declining when γ is large, in the range log γ ∈ [−3, −2]. This can be interpreted as the range where the model starts to over-fit to the observed structures - that is, it continues to improve the Dice score for observed structures while harming the registration accuracy for the other structures (Fig. 9c).

有趣的是，VoxelMorph（用辅助数据训练得到的）与无监督的变体相比，在γ值的一定范围内，对未观察到的结构也会得到改进的dice结果（图9a-b），即使这些分割在训练时并没有显式的看到。当我们在评估时使用的所有结构在训练时都观察到了，我们发现γ值越高，会得到更好的结果（图9c）。对于未观察到的结构，配准准确率在γ很大时，开始降低，范围是log γ ∈ [−3, −2]。这可以解释为，模型在这个范围内会对观察到的结构过拟合，即，我们对于观察到的结构会持续进行改进，而同时对其他结构会伤害配准准确率（图9c）。

**2) Training with coarse labels**: We consider the scenario where only coarse labels are available, such as when all the white matter is segmented as one structure. This situation enables evaluation of how the auxiliary data affects anatomical registration at finer scales, within the coarsely delineated structures. To achieve this, we merge the 30 structures into four broad groups: white matter, gray matter, cerebral spinal fluid (CSF) and the brain stem, and evaluate the accuracy of the registration on the original structures.

**2) 使用粗糙的标签进行训练**：我们考虑的场景是，只有粗糙的标签是可用的，比如所有白质分割成了一个结构。这种情况下，可以对辅助数据如何在粗糙勾画的结构中，影响更细节尺度上的解剖结构的配准进行评估。为达到这个目标，我们将30个结构合并如4个更大的组：白质，灰质，cerebral spinal fluid (CSF) 和the brain stem，在原始结构上评估配准准确率。

Fig. 9d (top) presents mean Dice scores over the original 30 structures with varying γ. With γ of 0.01, we obtain an average Dice score of 0.78±0.03 on FreeSurfer segmentations. This is roughly a 3 Dice point improvement over VoxelMorph without auxiliary information (p-value < 10^−10).

图9d（上）给出了，在γ变化时，原始的30个结构的平均dice分数。γ为0.01时，我们在FreeSurfer分割中得到的平均dice分数为0.78±0.03。这与没有辅助信息的VoxelMorph相比，大约是3个dice点的改进。

**3) Regularity of Deformations**: We also evaluate the regularity of the deformation fields both visually and by computing the number of voxels for which the determinant of the Jacobian is non-positive. Table IV provides the quantitative regularity measure for all γ values, showing that VoxelMorph deformation regularity degrades slowly as a function of γ (shown on a log scale), with roughly 0.2% of the voxels exhibiting folding at the lowest parameter value, and at most 2.3% when γ = 0.1. Deformations from models that don’t encourage smoothness, at the extreme value of γ = ∞, exhibit 10–13% folding voxels. A lower γ value such as γ = 0.01 therefore provides a good compromise of high Dice scores for all structures while avoiding highly irregular deformation fields, and avoiding overfitting as described above. Fig 10 shows examples of deformation fields for γ = 0.01 and γ = ∞, and we include more figures in the supplemental material for each experimental setting.

**3) 形变的规则性**：我们通过两种方法评估形变场的规则性，第一是视觉的方式，第二是计算Jacobian的行列式是非负体素的数量。表IV给出了对于所有的γ值，规则性度量的量化值，表明VoxelMorph形变的规则性，随着γ的增加逐渐缓慢降低（以log尺度展示），大约0.2%的体素在最低的参数值时，表现出了folding，在γ=0.1时，最多有2.3%。在γ = ∞的极限值，模型中的形变并不鼓励平滑性，表现出了10-13%的folding体素。较低的γ值，如γ = 0.01，会在所有结构的高dice分数，和避免得到高度非规则性的形变场之间，得到很好的折衷，并避免过拟合。图10给出了γ = 0.01和γ = ∞的形变场例子，我们在附加材料中给出了每种试验设置的更多图像。

Fig. 10: Effect of γ on warped images and deformation fields. We show the moving image, fixed image, and warped image (columns 1-3) with the structures that were observed at train time overlaid. The resulting deformation field is visualized in columns 4 and 5. While providing better Dice scores for observed structures, the deformation fields resulting from training with γ = ∞ are far more irregular than those using γ = 0.01. Similarly, the warped image are visually less coherent for γ = ∞.

**4) Testing on Manual Segmentation Maps**: We also test these models on the manual segmentations in the Buckner40 dataset used above, resulting in Fig. 9 (bottom). We observe a behavior consistent with the conclusions above, with smaller Dice score improvements, possibly due to the higher baseline Dice scores achieved on the Buckner40 data.

**4) 在手动分割图上的测试**：我们还在上述的Buckner40数据集中的手动分割中测试了这些模型，结果如图9（底部）所示。我们观察到与上述一致的结论，还有较小的dice分数改进，可能是因为在Buckner40数据集中有更高基准dice分数。

## 6. Discussion and conclusion

VoxelMorph with unsupervised loss performs comparably to the state-of-the-art ANTs and NiftyReg software in terms of Dice score, while reducing the computation time from hours to minutes on a CPU and under a second on a GPU. VoxelMorph is flexible and handles both partially observed or coarsely delineated auxiliary information during training, which can lead to improvements in Dice score while still preserving the runtime improvement.

用无监督损失的VoxelMorph与目前最好的ANTs和NiftyReg软件比较，在dice分数上表现相当，而计算时间从几小时，减少到了在CPU上的几分钟，或者在GPU上的小于一秒钟。VoxelMorph是很灵活的，可以在训练时，利用部分观察到的或粗糙勾画的辅助信息，带来dice分数的改变，同时保持运行时的改进。

VoxelMorph performs amortized optimization, learning global function parameters that are optimal for an entire training dataset. As Fig. 8 shows, the dataset need not be large: with only 100 training images, VoxelMorph leads to state-of-the-art registration quality scores for both training and test sets. Instance-specific optimization further improves VoxelMorph performance by one Dice point. This is a small increase, illustrating that amortized optimization can lead to nearly optimal registration.

VoxelMorph进行amortized改进，学习全局函数参数，对于整个训练数据集是最优的。如图8所示，数据集不需要很大：只需要100幅训练图像，VoxelMorph就可以在训练和测试集上，得到目前最好的配准质量分数。Instance-specific优化可以进一步在VoxelMorph的基础上，改进1个点的dice分数。这是一个很小的改进，说明amortized优化可以带来接近最优的配准。

We performed a thorough set of experiments demonstrating that, for a reasonable choice of γ, the availability of anatomical segmentations during training significantly improves test registration performance with VoxelMorph (in terms of Dice score) while providing smooth deformations (e.g. for γ = 0.01, less than 0.5% folding voxels). The performance gain varies based on the quality and number of anatomical segmentations available. Given a single labeled anatomical structure during training, the accuracy of registration of test subjects for that label increases, without negatively impacting other anatomy. If half or all of the labels are observed, or even a coarse segmentation is provided at training, registration accuracy improves for all labels during test. While we experimented with one type of auxiliary data in this study, VoxelMorph can leverage other auxiliary data, such as different modalities or anatomical keypoints. Increasing γ also increases the number of voxels exhibiting a folding of the registration field. This effect may be alleviated by using a diffeomorphic deformation representation for VoxelMorph, as introduced in recent work [5].

我们进行了很多试验，证明了，合理的选择γ值的大小，解剖结构分割在训练时，可以显著改进VoxelMorph的配准性能（以dice分数为指标），而且给出平滑的形变（如，对于γ = 0.01，小于0.5%的folding体素）。基于可用的解剖结构分割的数量和质量，性能提升会有变化。在训练时给定单标签的解剖结构，测试对象的配准精度在这个标签上就会有提升，而对其他解剖结构没有正面影响。如果可以观察到一般或所有的标签，或在训练时可以用一个粗糙的分割图，配准准确率在测试时就会对所有标签都有改进。我们对一种类型的辅助数据进行了试验，VoxelMorph还可以利用其他类型的辅助数据，比如不同模态的成像，或解剖结构关键点。增加γ也会使得，配准场中有folding的体素数量有所增加。这种效果，通过使用微分同胚形变表示进行VoxelMorph，可以得到缓解，这在最近的工作中有介绍[5]。

VoxelMorph is a general learning model, and is not limited to a particular image type or anatomy – it may be useful in other medical image registration applications such as cardiac MR scans or lung CT images. With an appropriate loss function such as mutual information, the model can also perform multimodal registration. VoxelMorph promises to significantly speed up medical image analysis and processing pipelines, while opening novel directions in learning-based registration.

VoxelMorph是一个通用学习模型，而且不限于某种特定图像类型，或解剖结构，在其他医学配准应用中也会有用，如心脏MR scans，或肺部CT图像。在适当的损失函数下，如互信息，模型也可以进行多模态配准。VoxelMorph可以显著加速医学图像分析和处理的流程，同时开启了基于学习的配准新方向。