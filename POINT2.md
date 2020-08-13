# Multiview 2D/3D Rigid Registration via a Point-Of-Interest Network for Tracking and Triangulation

Haofu Liao et. al. University of Rochester

## 0. Abstract

We propose to tackle the multiview 2D/3D rigid registration problem via a Point-Of-Interest Network for Tracking and Triangulation (POINT2). POINT2 learns to establish 2D point-to-point correspondences between the pre- and intra-intervention images by tracking a set of point-of-interests (POIs). The 3D pose of the pre-intervention volume is then estimated through a triangulation layer. In POINT2, the unified framework of the POI tracker and the triangulation layer enables learning informative 2D features and estimating 3D pose jointly. In contrast to existing approaches, POINT2 only requires a single forward-pass to achieve a reliable 2D/3D registration. As the POI tracker is shift-invariant, POINT2 is more robust to the initial pose of the 3D pre-intervention image. Extensive experiments on a large-scale clinical cone-beam computed tomography dataset show that the proposed POINT2 method outperforms the existing learning-based method in terms of accuracy, robustness and running time. Furthermore, when used as an initial pose estimator, our method also improves the robustness and speed of the state-of-the-art optimization-based approaches by ten folds.

我们提出，通过感兴趣点网络进行跟踪和三角测量(POINT2)，解决多视2D/3D刚体配准问题。POINT2学习确立介入前和介入中图像的2D点到点的对应性，跟踪感兴趣点集(POIs)。介入前体的3D姿态通过一个三角测量层进行估计。在POINT2中，POI跟踪和三角测量层的统一框架，可以同时学习信息很多的2D特征，并估计3D姿态。与现有的方法相比，POINT2只需要一个前向过程就可以得到可靠的2D/3D配准。由于POI跟踪是平移不变的，POINT2对介入前的3D图像的初始姿态更加稳健。在一个大规模临床CBCT数据集上进行了广泛的试验，表明提出的POINT2方法超过了现有的基于学习的方法，包括精度、稳定性和运行时间。而且，当用作初始姿态估计时，我们的方法还改进了稳健性和速度，比之前最好的基于优化的方法好了10倍。

## 1. Introduction

In 2D/3D rigid registration for intervention, the goal is to find a rigid pose of a pre-intervention 3D data, e.g., computed tomography (CT), such that it aligns with a 2D intra-intervention image of a patient, e.g., fluoroscopy. In practice, CT is usually a preferred 3D pre-intervention data as digitally reconstructed radiographs (DRRs) can be produced from CT using ray casting [21]. The generation of DRRs simulates how an X-ray is captured, which makes them visually similar to the X-rays. Therefore, they are leveraged to facilitate the 2D/3D registration as we can observe the misalignment between the CT and patient by directly comparing the intra-intervention X-ray and the generated DRR (See Figure 1 and Section 3.1 for details).

在介入中的2D/3D刚体配准中，其目的是找到一个介入前3D数据的刚体姿态，如CT，这样与2D的介入中患者数据可以对齐，如透视图像。实践中，CT通常是常见的3D介入前3D数据，因为可以从CT中使用射线追踪法生成DRRs。DRRs的生成模拟了X射线捕获的过程，视觉上与X射线图像类似。因此，可以利用CT来促进2D/3D配准，因为我们可以通过直接比较介入中的X射线图像和生成的DRR，来观察CT和患者之间的错位情况。

One of the most commonly used 2D/3D registration strategies [12] is through an optimization-based approach, where a similarity metric is first designed to measure the closeness between the DRRs and the 2D data, and then the 3D pose is iteratively searched and optimized for the best similarity score. However, the iterative pose searching scheme usually suffers from two problems. First, the generation of DRRs incurs high computation, and the iterative pose searching requires a significant number of DRRs for the similarity measure, making it computationally slow. Second, iterative pose searching relies on a good initialization. When the initial position is not close enough to the correct one, the method may converge to local extrema, and the registration fails. Although many studies have been proposed to address these two problems [4, 16, 15, 8, 5, 7, 19], trade-offs still have to be made between sampling good starting points and less costly registration.

最常用的2D/3D配准策略之一[12]是通过基于优化的方法，首先设计一个相似度度量，来衡量DRRs和2D数据的接近程度，然后迭代的搜索3D姿态，优化得到最佳的相似性分数。但是，迭代姿态搜索方案通常有两种问题。首先，DRRs的生成计算量很大，迭代姿态搜索需要非常多的DRRs，以计算相似性度量，这使其计算上就很慢。第二，迭代姿态搜索依赖于好的初始化条件。当初始位置与正确的位置不接近时，这种方法会收敛到局部极值，配准就会失败。虽然提出了很多研究来解决这两个问题，但在好的初始点和耗费较少的配准中，仍然需要进行折中。

In recent years, the development of deep neural networks (DNNs) has enabled a learning-based strategy for medical image registration [13, 22, 11, 14] that aims to estimate the pose of the 3D data without searching and sampling the pose space at a large scale. Despite the efficiency, there are still two limitations of the existing learning-based methods. First, the learning-based methods usually require generating a huge number of DRRs for training. The corresponding poses for the DRRs have to be dense in the entire searching space to avoid overfitting. Considering that the number of required DRRs is exponential with respect to the dimension of the pose space (which is usually six), this is computationally prohibitive, thus making the learning-based methods less reliable during testing. Second, the current state-of-the-art learning-based methods [13, 22, 11] require an iterative refinement of the estimated pose and use DNNs to predict the most plausible update direction for faster convergence. However, the iterative approach still introduces a non-negligible computational cost, and the DNNs may direct the searching to an unseen state, which fails the registration quickly.

最近几年来，DNN的发展可以对医学图像配准采用一种基于学习的策略，目标是估计3D数据的姿态，而不用在很大尺度上搜索并对姿态空间进行采样。尽管效率很高，但现有的基于学习的方法仍然有两种局限。首先，基于学习的方法通常需要生成大量DRRs进行训练。这些DRRs的对应姿态在整个搜索空间中必须是密集的，以防止过拟合。考虑到需要的DRRs的数量对于姿态空间的维度（通常是6）是指数级的，这在计算量上就非常大，因此使得基于学习的方法在测试时不太可靠。第二，目前最好的基于学习的方法需要对估计的姿态进行迭代精炼，使用DNNs来预测最可行的更新方向，以得到更快的收敛。但是，迭代方法仍然引入了不可忽视的计算代价，DNNs可能会将搜索引导到一种未曾见过的状态，会迅速的使配准失败。

In this paper, we introduce a novel learning-based approach, which is referred to as a Point-Of-Interest Network for Tracking and Triangulation (POINT2). POINT2 directly aligns the 3D data with the patient by using DNNs to establish a point-to-point correspondence between multiple views of DRRs and X-ray images. The 3D pose is then estimated by aligning the matched points. Specifically, these are achieved by tracking a set of points of interest (POIs). For 2D correspondence, we use the POI tracking network to map the 2D POIs from the DRRs to the X-ray images. For 3D correspondence, we develop a triangulation layer that projects the tracked POIs in the X-ray images of multiple views back into 3D. We highlight that since the point-to-point correspondence is established in a shift-invariant manner, the requirement of dense sampling in the entire pose space is avoided.

本文中，我们提出了一种新颖的基于学习的方法，称为进行跟踪和三角测量的感兴趣点网络(POINT2)。POINT2直接将3D数据与患者进行对齐，在多个视角的DRRs和X射线图像中，使用DNNs来确立点对点的对应性。然后通过对齐匹配的点，估计3D姿态。具体的，这是通过跟踪感兴趣点集(POIs)来得到的。对于2D对应性，我们使用POI跟踪网络，将2D POIs从DRRs映射到X射线图像中。对于3D对应性，我们提出了一种三角测量层，将跟踪的多视角X射线中POIs投影回到3D中。我们强调，由于点到点的对应性是通过平移不变的方式确定的，就可以避免在整个姿态空间中进行密集采样的需要。

The contributions of this paper are as follows: 本文的贡献如下：

- A novel learning-based multiview 2D/3D rigid registration method that directly measures the 3D misalignment by exploiting the point-to-point correspondence between the X-rays and DRRs, which avoids the costly and unreliable iterative pose searching, and thus delivers faster and more robust registration. 一种新型的基于学习的多视2D/3D刚体配准方法，通过探索X射线图像和DRRs的点对点对应性，直接测量3D错位，避免了昂贵的不可靠的迭代姿态搜索，因此得到了更加快速更加稳定的配准。

- A novel POI tracking network constructed using a Siamese U-Net with POI convolution to enable a fine-grained feature extraction and effective POI similarity measure, and more importantly, to offer a shift-invariant 2D misalignment measure that is robust to in-plane offsets. 使用孪生U-Net和POI卷积，构建了一种新型的POI跟踪网络，可以进行细粒度特征提取，高效的度量POI相似度，更重要的是，可以给出一种偏移不变的2D错位度量，对平面内偏移是稳定的。

- A unified framework of the POI tracker and the triangulation layer, which enables (i) end-to-end learning of informative 2D features and (ii) 3D pose estimation. POI跟踪和三角测量层的统一框架，可以进行(i)端到端的学习信息丰富的2D特征，(ii)3D姿态估计。

- An extensive evaluation on a large-scale and challenging clinical cone-beam CT (CBCT) dataset, which shows that the proposed method performs significantly better than the state-of-the-art learning-based approaches, and, when used as an initial pose estimator, it also greatly improves the robustness and speed of the state-of-the-art optimization-based approaches. 在一个大型的有挑战的临床CBCT数据集上进行了广泛的测试，表明提出的方法比之前最好的基于学习的方法表现要好很多，并且，当用作初始姿态估计时，在现有的最好的基于优化的方法的基础上，稳定性和速度得到了极大的改进。

## 2. Related Work

**Optimization-Based Approaches**. Optimization-based approaches usually suffer from high computational cost and is sensitive to the initial estimate. To reduce the computational cost, many works have been proposed to improve the efficiency in hardware-level [10, 8, 15] or software-level [19, 27, 9]. Although these works have successfully reduced the DRR generation time to a reasonable range, the overall registration time is still non-negligible [19, 15] and the registration accuracy might be compromised for faster speed [27, 19]. For better initial pose estimation, many attempts have been made by either sampling better initial position [7, 5], using multistart strategies [26, 16], or a carefully designed objective function that is less sensitive to the initial position selection [15]. However, these methods usually achieve a more robust registration at the cost of longer running time as more locations, and the corresponding DRRs need to be sampled and generated, respectively, to avoid being trapped in the local extrema.

**基于优化的方法**。基于优化的方法通常有计算代价过高，和对初始估计敏感的问题。为降低计算代价，提出了很多工作来改进效率问题，包括硬件级的或软件级的工作。虽然这些工作成功的把DRR生成的时间降低到了一种合理的范围，总体的配准时间仍然是不可忽视的，配准准确率与更快的速度是可以相互折中的。对于更好的初始姿态估计，已经进行了很多尝试，要么采样更好的初始姿态，使用多起始策略，或一种仔细设计的目标函数，对初始位置的选择不那么敏感。但是，这些方法得到更稳定的配准的代价，通常是更长的运行时间，因为更多的位置和对应的DRRs需要分别进行采样和生成，以防止陷入局部极值。

**Learning-Based Approaches**. Early learning-based approach [14] aims to train the DNNs to directly predict the 3D pose given a pair of DRR and X-ray images. However, this approach is generally too ambitious and hence relies on the existence of opaque objects, such as medical implants, that provide strong features for robustness. Alternatively, it has been shown that formulating the registration as a Markov decision process (MDP) is viable [11]. Instead of directly regressing the 3D pose, MDP-based methods propose first to train an agent that predicts the most possible search direction and then the registration is iteratively repeated until a fixed number of steps is reached. However, the MDP-based approach requires the agent to be trained on a large number of samples such that the registration can follow the expected trajectory. Though mitigated with a multi-agent design [13], it is still inevitable that the neighborhood search may reach an unseen pose and the registration fails. Moreover, the MDP-based approach cannot guarantee convergence and hence limits its registration accuracy. Therefore, the MDP-based approach [13] is usually used to find a good initial pose for the registration, and a combination with an optimization-based method is applied for better performance. Another possible approach is by directly tracking landmarks from multiple views of X-ray images [2]. However, the landmark-based tracking approach does not make use of the information from the CT volume and requires the landmarks to be present in the X-ray images, making it less robust and applicable to clinical applications.

**基于学习的方法**。早期基于学习的方法[14]目标是训练DNNs在给定的DRR和X射线图像对上直接预测3D姿态。但是，这种方法通常目标太大，因此依赖于不透明的目标的存在，如医学植入物，这可以提供很强的特征，使得算法很稳健。此外，已经证明了，将配准标注为一个Markov决策过程(MDP)是可行的[11]。基于MDP的方法没有直接回归3D姿态，而是首先训练一个代理，预测最可能的搜索方向，然后迭代的重复进行配准，直到达到固定数量的步骤。但是，基于MDP的方法需要代理在很大数量的样本上进行训练，这样配准可以沿着期望的轨迹进行。不可避免的是，邻域搜索可能会达到一个未曾见过的姿态，配准过程会失败，通过一个多代理的设计，可以缓解这个问题。而且，基于MDP的方法不能保证收敛，因此限制了其配准准确率。因此，基于MDP的方法通常用于找到一个很好的初始姿态进行配准，并与一种基于优化的方法组合使用，得到更好的性能。另一种可能的方法是，从多视角X射线图像中直接跟踪特征点[2]。但是，基于特征点的跟踪方法，没有使用CT体的信息，需要特征点在X射线图像中，使其不那么稳定，不太能应用到临床应用中。

## 3. Methodology

### 3.1. Problem Formulation

Following the convention in the literature [12], we assume a 2D/3D rigid registration problem and also assume that the 3D data is a CT or CBCT volume, which is the most accessible and allows the generation of DRR. For the 2D data, we use X-rays. As single-view 2D/3D registration is an ill-posed problem (due to the ambiguity introduced by the out-plane offset), X-rays from multiple views are usually captured during the intervention. Therefore, we also follow the literature [12] and tackle a multiview 2D/3D registration problem. Without loss of generality, most of the studies in this work are conducted under two views, and it is easy to extend our work to the cases with more views.

按照文献[12]，我们假设要解决的是2D/3D刚性配准问题，同时假设3D数据是CT或CBCT体，这是最可用的可以生成DRR的数据形式。对于2D数据，我们使用X射线图像。单视角2D/3D配准是一个病态问题（由于异面偏移带来的疑义性），多视角的X射线图像通常是在介入手术过程中拍摄的。因此，我们按照[12]，处理的是一个多视角2D/3D配准问题。不失一般性，本文中的多数研究都是在两视角的情况下进行的，很容易将我们的工作拓展到多视角的情况中。

**2D/3D Rigid Registration with DRRs**. In 2D/3D rigid registration, the misalignment between the patient and the CT volume V is formulated through a transformation matrix T that brings V from its initial location to the patient’s location under the same coordinate. As illustrated in Figure 2, T is usually parameterized by three translations $t=(t_x, t_y, t_z)T$ and three rotations $θ=(θ_x, θ_y, θ_z)^T$ about the axes, and can be written as a 4 × 4 matrix under the homogeneous coordinate

**与DRRs的2D/3D刚性配准**。在2D/3D刚性配准中，患者与CT体V之间的错位，通过一个变换矩阵T来描述，将V从其初始位置在同一个坐标系中，带到患者的位置。如图2所示，T的参数通常为三个平移$t=(t_x, t_y, t_z)T$和三个旋转$θ=(θ_x, θ_y, θ_z)^T$，在齐次坐标系下可以写成一个4 × 4的矩阵

$$T = \left[ \begin{matrix} R(θ) & t \\ 0 & 1 \end{matrix} \right]$$(1)

where R is the rotation matrix that controls the rotation of V around the origin. 其中R是旋转矩阵，控制着V围绕原点的旋转。

As demonstrated in Figure 1, casting simulated X-rays through the CT volume creates a DRR on the detector. Similarly, passing a real x-ray beam through the patient’s body gives an X-ray image. Hence, the misalignment between the CT volume and the patient can be observed from the detector by comparing the DRR and the X-ray image. Given a transformation matrix T and a CT volume V, the DRR $I^D$ can be computed by

如图1所示，将模拟的X射线投射到CT体上，得到探测器上的DRR。类似的，将真实的X射线束投射过患者的身体，得到一幅X射线图像。因此，CT体和患者的错位可以通过观察探测器，比较DRR和X射线图像得到。给定变换矩阵T和CT体V，DRR $I^D$可以计算为

$$I^D(x) = \int_{p∈l(x)} V(T^{-1}p)dp$$(2)

where l(x), whose parameters are determined by the imaging model, is a line segment connecting the X-ray source and a point x on the detector. Therefore, let $I^X$ denote the X-ray image, the 2D/3D registration can be seen as finding the optimal $T^*$ such that $I^X$ and $I^D$ are aligned.

其中l(x)参数是由成像模型决定的，是一条线段，连接X射线源和探测器上的点x。因此，令$I^X$表示X射线图像，2D/3D配准可以视为找到最佳变换$T^*$，使$I^X$和$I^D$对齐。

**X-Ray Imaging Model**. An X-ray imaging system is usually modeled as a pinhole camera [3, 6], as illustrated in Figure 2, where the X-ray source serves as the camera center and the X-ray detector serves as the image plane. Following the convention in X-ray imaging [3], we assume an isocenter coordinate system whose origin lies at the isocenter. Without loss of generality, we also assume the imaging model is calibrated, and there is no X-ray source offset and detector offset. Thus, the X-ray source, the isocenter, and the detector’s origin are collinear, and the line from the X-ray source to the isocenter (referred to as the principal axis) is perpendicular to the detector. Let d denote the distance between the X-ray source and the detector origin, and c denote the distance between the X-ray source and the isocenter, then for a point $X = (X, Y, Z)^T$ in the isocenter coordinate, its projection x on the detector is given by

**X射线成像模型**。一个X射线成像系统通常建模为一个针孔成像系统，如图2所示，其中X射线源是相机中心，X射线探测器是图像平面。按照X射线成像中的惯例，我们假设一个等中心坐标系统，其原点在等中心处。不失一般性，我们还假设成像模型是校准过的，没有X射线源的偏移和成像板的偏移。因此，X射线源，等中心，探测器的中心是共线的，从X射线源到等中心的线（称为主轴）与探测器是垂直的。令d表示X射线源到探测器原点的距离，c表示X射线源到等中心的距离，那么对于等中心坐标系的点$X = (X, Y, Z)^T$，其在成像板上的投影x为

$$x' = K[I,h]\left( \begin{matrix} X \\ 1 \end{matrix} \right)$$(3)

where

$$K = \left[ \begin{matrix} -d & 0 & 0 \\ 0 & -d & 0 \\ 0 & 0 & 1 \end{matrix} \right], h = \left( \begin{matrix} 0 \\ 0 \\ -c \end{matrix} \right)$$

Here $x' = (x', y', z')$ if defined under the homogeneous coordinate and its counterpart under the detector coordinate can be written as x = (x, y) = (x'/z', y'/z').

这里$x' = (x', y', z')$是在齐次坐标系中定义的，在探测器坐标系中的对应点可以写为x = (x, y) = (x'/z', y'/z')。

In general, an X-ray is usually not captured at the canonical view as discussed above. Let $T_{view}$ be a transformation matrix that converts a canonical view to a non-canonical view (Figure 2), then the projection of X for the non-canonical view can be written as

一般来说，X射线并不是在前面所述的典型视角上捕获到的。令$T_{view}$为一个变换矩阵，将典型视角变换到非典型视角（图2），X在非典型视角上的投影可以写成

$$x' = K[R_{view}, t_{view}+h] \left( \begin{matrix} X \\ 1 \end{matrix} \right)$$(4)

where $R_{view}$ and $t_{view}$ perform the rotation and translation, respectively, as in Equation (1). Similarly, we can rewrite Equation (2) at a non-canonical view as

其中$R_{view}$和$t_{view}$分别进行的是旋转和平移，如式(1)。类似的，我们可以在非典型视角下重写式(2)为

$$I_{view}^D(x) = \int_{p∈l(x)} V(T^{-1}T^{-1}_{view}p)dp$$(5)

### 3.2. The Proposed POINT2 Approach

An overview of the proposed method with two views is shown in Figure 3. Given a set of DRR and X-ray pairs of different views, our approach first selects a set of POIs in 3D from the CT volume and projects them to each DRR using Equation (4) as shown in Figure 3(a). Then, the approach measures the misalignment between each pair of DRR and X-ray by tracking the projected DRR POIs from the X-ray (Figure 3(b)). Using the tracked POIs on the X-rays, we can estimate their corresponding 3D POIs on the patient through triangulation (Figure 3(c)). Finally, by aligning CT POIs with patient POIs, the pose misalignment $T^{**}$ between the CT and the patient can be calculated (Figure 3(d)).

提出的方法的概览如图3所示。给定不同视角的DRR和X射线图像对集合，我们的方法首先在CT体中选择一组3D POIs，使用式(4)将其投影到每个DRR上，如图3(a)。然后，本方法通过在X射线图像中追踪投影的DRR POIs，度量每对DRR和X射线的错位（图3(b)）。使用在X射线图像上跟踪的POIs，我们可以通过三角测量估计在患者身上对应的3D POIs（图3(c)）。最后，通过将CT POIs与患者POIs对齐，CT和患者之间的姿态错位$T^{**}$可以得到计算出来（图3(d)）。

**POINT**. One of the key components of the proposed method is a Point-Of-Interest Network for Tracking (POINT) that finds the point-to-point correspondence between two images, that is, we use this network to track the POIs from DRR to X-ray. Specifically, the network takes a DRR and X-ray pair ($I^D, I^X$) and a set of projected DRR POIs {$x^D_1, x^D_2, . . . , x^D_m$} as the input and outputs the tracked X-ray POIs in the form of heatmaps {$\hat M^X_1, \hat M^X_2,..., \hat M^X_m$}.

提出的方法的一个关键组件是，进行跟踪的感兴趣点网络(POINT)，可以找到两幅图像中的点对点的对应关系，即，我们使用这个网络来从DRR跟踪POIs到X射线图像。具体的，网络以DRR和X射线图像对($I^D, I^X$)为输入，还有投影的DRR POIs集合{$x^D_1, x^D_2, . . . , x^D_m$}为输入，输出跟踪的X射线图像POIs，形式为热力图{$\hat M^X_1, \hat M^X_2,..., \hat M^X_m$}。

The structure of the network is illustrated in Figure 4. We construct this network under a Siamese architecture [1, 23] with each branch φ having an U-Net like structure [18]. The weights of the two branches are shared. Each branch takes an image as the input and performs fine-grained feature extraction at pixel-level. Thus, the output is a feature map with the same resolution as the input image, and for an image with size M×N, the size of the feature map is M×N×C where C is the number of channels. We denote the extracted feature maps of DRR and X-ray as $F^D = φ(I^D)$ and $F^X = φ(I^X)$, respectively.

网络的结构如图4所示。我们以孪生结构来构建这个网络，每个分支φ都有一个U-Net类的结构[18]。两个分支的权重是共享的。每个分支以一幅图像为输入，进行像素级的细粒度特征提取。因此，输出是与输入分辨率相同的特征图，对于M×N大小的图像，特征图的大小为M×N×C，其中C是通道数。我们将DRR和X射线图像的提取出来的特征图分别表示为$F^D = φ(I^D)$和$F^X = φ(I^X)$。

With feature map $F^D$, the feature vector of a DRR POI $x_i^D$ can be extracted by interpolating $F^D$ at $x_i^D$. The feature extraction layer (FE layer) in Figure 4 performs this operation and we denote its output as a feature kernel $F^D(x^D_i)$. For a richer feature representation, the neighbor feature vectors around $x^D_i$ may also be used. A neighbor of size K gives in total (2K+1)×(2K+1) feature vectors and the feature kernel $F^D(x^D_i)$ in this case has a size (2K+1)×(2K+1)×C.

有了特征图$F^D$，一个DRR POI $x_i^D$的特征向量可以通过在$x_i^D$对$F^D$内插提取得到。图4中的特征提取层(FE层)进行这个操作，我们将其输出表示为一个特征核$F^D(x^D_i)$。为得到更加丰富的特征表示，在$x^D_i$附近的临近特征向量也可能进行使用。大小为K的邻域可以得到(2K+1)×(2K+1)个特征向量，特征核$F^D(x^D_i)$在这种情况下的大小为(2K+1)×(2K+1)×C。

Similarly, a feature kernel at x of the X-ray feature map can be extracted and denoted as $F^X(x)$. Then, we may apply a similarity operation to $F^D(x^D_i)$ and $F^X(x)$ to give a similarity score of the two locations $x^D_i$ and x. When the similarity check is operated exhaustively over all locations on the X-ray, the location $x^*$ with the highest similarity score is regarded as the corresponding POI of $x^D_i$ on the X-ray. Such an exhaustive search on $F^X$ can be performed effectively with convolution and is denoted as a POI convolution layer in Figure 4. The output of the layer is a heatmap $\hat M^X_i$ and is computed by

类似的，在X射线图像特征图位置x上也可以提取一个特征核，表示为$F^X(x)$。然后，我们可以对$F^D(x^D_i)$和$F^X(x)$进行一个相似度运算，给出两个位置$x^D_i$和x的相似度分数。当相似度检查在X射线图像的所有位置进行穷举式运算，相似度分数最高的位置$x^*$被认为是在X射线图像上$x^D_i$的对应POI。在$F^X$上的这样一个穷举式搜索可以用卷积进行高效的实现，表示为图4中的一个POI卷积层。这一层的输出是一个热力图$\hat M^X_i$，由下式计算：

$$\hat M^X_i = F^X * (W⊙F^D(x^D_i))$$(6)

where W is a learned weight that selects the features for better similarity. Each element $\hat M^X_i(x)$ denotes a similarity score of the corresponding location x on the X-ray.

其中W是一个学习到的权重，选择具有更好相似度的特征。每个元素$\hat M^X_i(x)$都表示在X射线图像中位置x上的一个相似度分数。

**POINT2**. With the tracked POIs from different views of X-rays, we can obtain their 3D locations on the patient using triangulation as shown in Figure 3(c). However, this work seeks a uniform solution that formulates the POINT network and the triangulation under the same framework so that the two tasks can be trained jointly in an end-to-end fashion which could potentially benefit the learning of the tracking network. An illustration of this end-to-end design for two views is shown in Figure 5. For an n-view 2D/3D registration problem, the proposed design will include n POINT networks as discussed above. Each of the networks will track POIs for the designated view and, therefore, the weights are not shared among the networks. Given a set of DRR and X-ray pairs {$(I_1^D, I_1^X),(I_2^D, I_2^X),...,(I_n^D, I_n^X)$} of the n views, these networks output the tracked X-ray POIs of each view in the form of heatmaps.

**POINT2**。有了在不同视角的X射线图像中跟踪的POIs，我们可以使用图3(c)中的三角定位，得到其在患者身上的3D位置。但是，本文试图得到一致的解决方案，将POINT网络和三角测量的工作在同一框架中表述，这样这两个任务可以进行联合训练，并以端到端的方式进行，这可能使跟踪网络的学习也受益。这种双视角的端到端的设计的描述，如图5所示。对于一个n视角的2D/3D配准问题，提出的设计会包含n个POINT网络，如上所述。每个网络都跟踪指定视角的POIs，因此，权重在网络之间并不是共享的。给定n个视角的DRR和X射线图像集合{$(I_1^D, I_1^X),(I_2^D, I_2^X),...,(I_n^D, I_n^X)$}，这些网络输出每个视角的跟踪的X射线POIs，输出形式是热力图。

After obtaining the heatmaps, we introduce a triangulation layer that localizes a 3D point by forming triangles to it from the 2D tracked POIs from the heatmaps. Formally, we denote $M_j = \{ \hat M_{1j}^X, \hat M_{2j}^X, ..., \hat M_{nj}^X\}$ the set of heatmaps from different views but all corresponding to the same 3D POI $\hat X_j^X$. Here, $\hat M_{ij}^X$ is the heatmap of the j-th X-ray POI from the i-th view, and we obtain the 2D X-ray POI by

在得到热力图后，我们引入一个三角测量层，作用是定位3D点，方法是通过从热力图中跟踪的2D POIs形成三角形。正式的，我们用$M_j = \{ \hat M_{1j}^X, \hat M_{2j}^X, ..., \hat M_{nj}^X\}$表示热力图集合，来自不同视角，但是都对应着相同的3D POI $\hat X_j^X$。这里，$\hat M_{ij}^X$是第i个视角中的第j个X射线图像POI的热力图，我们通过下式得到2D X射线POI：

$$\hat x_{ij}^X = \frac {1}{\sum_x \hat M_{ij}^X(x)} \sum_x \hat M_{ij}^X (x)x$$(7)

Next, we rewrite Equation (4) as 然后，我们重写式(4)

$$D(x) R_{view} X = cx - D(x) t_{view}$$(8)

where 其中

$$D(x) = \left[ \begin{matrix} d & 0 \\ 0 & d \end{matrix} \space x \right]$$

Thus, by applying Equation (8) for each view, we can get 对每个视角应用式(8)，我们可以得到

$$\left\{ \begin{matrix} D(\hat x_{1j}^X)R_1 \hat X_j^X & =c\hat x_{1j}^X-D(\hat x_{1j}^X)t_1 \\ D(\hat x_{2j}^X)R_2 \hat X_j^X & =c\hat x_{2j}^X-D(\hat x_{2j}^X)t_2 \\ \vdots \\ D(\hat x_{nj}^X)R_n \hat X_j^X & =c\hat x_{nj}^X-D(\hat x_{nj}^X)t_n \end{matrix} \right.$$(9)

Let

$$A=\left[ \begin{matrix} D(\hat x_{1j}^X)R_1 \\ D(\hat x_{2j}^X)R_2 \\ \vdots \\ D(\hat x_{nj}^X)R_n \end{matrix} \right], b=\left[ \begin{matrix} c\hat x_{1j}^X-D(\hat x_{1j}^X)t_1 \\ c\hat x_{2j}^X-D(\hat x_{2j}^X)t_2 \\ \vdots \\ c\hat x_{nj}^X-D(\hat x_{nj}^X)t_n \end{matrix} \right]$$(10)

then $\hat X_j^X$ is given by

$$\hat X_j^X = A^+b$$(11)

The triangulation can be plugged into a loss function that regulates the training of POINT networks of different views. 这个三角测量可以插入到损失函数中，对不同视角的POINT网络的训练进行规范。

$$L=\frac{1}{mn} \sum_i \sum_j BCE(σ(\hat M_{ij}^X), σ(M_{ij}^X)) + \frac{w}{n} \sum_j ||\hat X_j^X-X_j^X||_2$$(12)

where $M_{ij}^X$ is the ground truth heatmap, $X^X_j$ is the ground truth 3D POI, BCE is the pixel-wise binary cross entropy function, σ is the sigmoid function, and w is a weight balancing the losses between tracking and triangulation errors.

其中$M_{ij}^X$是真值热力图，$X^X_j$是真值3D POI，BCE是逐像素的二值交叉熵函数，σ是sigmoid函数，w是一个权重，在跟踪和三角测量的误差之间进行均衡。

**Shape Alignment**. Let $P^D =[X_1^D, X_2^D, ..., X_m^D]$ be the selected CT POIs and $P^X = [\hat X_1^X, \hat X_2^X, ..., \hat X_m^X]$ be the estimated 3D POIs. The shape alignment finds a transformation matrix $T^*$ such that the transformed $P^D$ aligns closely with $P^X$, i.e.,

**形状对齐**。令$P^D =[X_1^D, X_2^D, ..., X_m^D]$为选择的CT POIs，$P^X = [\hat X_1^X, \hat X_2^X, ..., \hat X_m^X]$为估计的3D POIs。形状对齐要找个一个变换矩阵$T^*$，这样变换的$P^D$与$P^X$对齐，即

$$T^* = argmin_T ||TP^D-P^X||_F, s.t., RR^T=I$$(13)

This problem is solved analytically through Procrustes analysis [20]. 这个问题可以通过Procrustes分析进行解析求解。

## 4. Experiments

### 4.1. Dataset

The dataset we use in the experiments is a cone-beam CT (CBCT) dataset captured for radiation therapy. The dataset contains 340 raw CBCT scans with each has 780 X-ray images. Each X-ray image comes with a geometry file that provides the registration ground truth as well as the information to reconstruct the CBCT volume. Each CBCT volume is reconstructed from the 780 X-ray images, and in total, we have 340 CBCT volumes (one for each CBCT scan). We use 300 scans for training and validation, and 40 scans for testing. The size of the CBCT volumes is 448 × 448 × 768 with 0.5 mm voxel spacing, and the size of the X-ray images is 512×512 with 0.388 mm pixel spacing. During the experiments, the CBCT volumes are treated as the 3D pre-intervention data, and the corresponding X-ray images are treated as the 2D intra-intervention data. Sample X-ray images from our dataset are shown in Figure 6. Note that unlike many existing approaches [15, 17, 25] that evaluate their methods on small datasets (typically about 10 scans) which are captured under relatively ideal scenarios, we use a significantly larger dataset with complex clinical settings, e.g., diverse field-of-views, surgical instruments/implants, various image contrast and quality, etc.

我们在试验中使用的数据集是CBCT数据集，进行放射治疗用的。数据集包含340幅原始CBCT scans，每个scans有780幅X射线图像。每幅X射线图像都有一个几何文件，给出配准的真值，以及用于重建CBCT体的信息。CBCT体就是从这780幅X射线图像中重建得到的，总计我们有340个CBCT体（对每个CBCT scan都有一个体）。我们使用300 scans进行训练和验证，40 scans进行测试。CBCT体的大小是448 × 448 × 768，体素间隔为0.5mm，X射线图像的大小为512×512，像素间隔为0.388mm。在试验中，CBCT体用作介入前3D数据，对应的X射线图像用作介入中2D数据。图6中给出了我们数据集中的X射线样本图像。注意，现有很多方法在小型数据集上（一般大约10 scans）评估其方法，这些小型数据集一般都是在相对理想的场景中获取的，我们与之不同，使用的是一个明显大很多的数据集，有着复杂的临床设置，如，不同的FOV，手术设备/植入物，各种图像对比度和质量，等。

We consider two common views during the experiment: the anterior-posterior view and the lateral view. Hence, only X-rays that are close to (±5◦) these views are used for training and testing. Note that this selection does not tightly constrain the diversity of the X-rays as the patient may be subject to movements with regard to the operating bed. To train the proposed method, X-ray and DRR pairs are selected and generated with a maximum of 10◦ rotation offset and 20 mm translation offset. We first invert all the raw X-ray images and then apply histogram equalization to both the inverted X-ray images and DRRs to facilitate the similarity measurement. For each of the scan, we also annotate their landmarks on the reconstructed CBCT volume for further evaluation.

我们在试验中考虑两个常见的视角：前后视角和侧向视角。因此，只有与这些视角接近(±5◦)的X射线图像用于训练和测试。注意，这种选择没有限制X射线图像的多样性，因为患者相对于手术床有各种运动。为训练提出的方法，选择和生成了X射线图像和DRR对，其旋转偏移最大10◦，平移偏移最大20mm。我们首先颠倒所有的原始X射线图像，然后对颠倒的X射线图像和DRRs应用直方图均化，以方便相似度度量。对每个scan，我们都在重建的CBCT体上标注了其特征点，以进行进一步的评估。

### 4.2. Implementation and Training Details

We implement the proposed approach under the Pytorch framework with GPU acceleration. For the POINT network, each of the Siamese branch φ has five encoding blocks (BatchNorm, Conv, and LeakyReLU) followed by five decoding blocks (BatchNorm, Deconv, and ReLU), thus forming a symmetric structure, and we use skip-connections to shuttle the lower-level features from an encoding block to its symmetric decoding counterpart (see details in the supplementary material). The triangulation layer is implemented according to Equation (11) with the backpropagation automatically supported by Pytorch. We train the proposed approach in a two-stage fashion. In the first stage, we train the POINT network of each view independently for 30 epochs. Then, we fine-tune POINT2 for 20 epochs. We find this mechanism converges faster than training POINT2 from scratch. For the optimization, we use the mini-batch stochastic gradient descent with 0.01 learning rate for the first stage and 0.001 for the second. We set the loss weight as w = 0.01, which we empirically find it works well during training. For the X-ray imaging model, we use d = 1, 500 mm and c = 1, 000 mm.

我们在Pytorch实现了提出的方法，带有GPU加速。对于POINT网络，每个孪生分支φ都有5个编码模块(BatchNorm, Conv, and LeakyReLU)，随后是5个解码模块(BatchNorm, Deconv, and ReLU)，因此形成了一种对称结构，我们使用跳跃连接来将编码模块中的较低层的特征传输到对称的解码模块中（详见附加材料）。三角测量层是根据式(11)实现的，反向传播由Pytorch自动支持。我们以两阶段的方式训练提出的方法。在第一阶段，我们独立训练每个视角的POINT网络，30轮。然后，我们精调POINT2网络20轮。我们发现这种机制比从头训练POINT2收敛的更快.对于优化，我们使用mini-batch SGD，学习速率在第一阶段为0.01，第二阶段为0.001。我们设置损失权重为w = 0.01，我们通过经验发现这在训练时效果很好。对于X射线成像模型，我们设置d=1500mm，c=1000mm。

### 4.3. Ablation Study

This section discusses an ablation study of the proposed POINT network. As the network tracks POIs in 2D, we use mean projected distance (mPD) [24] to evaluate different models with specific design choices. The evaluation results are given in Table 1.

本节中，我们讨论了提出的POINT网络的分离研究。由于网络是在2D中跟踪POIs，我们使用平均投影距离(mean projected distance, mPD)来评估带有具体设计选择的不同模型。评估结果如表1所示。

**POI Selection**. The first step of the proposed approach requires selecting a set of POIs to set up a point-to-point correspondence. In this experiment, we investigate different POI selection strategies. First, we investigate directly using landmarks as the POIs since they usually have strong semantic meaning and can be annotated before the intervention. Second, we also investigate an automatic solution that uses the Harris corners as the POIs to avoid the labor work of annotation. Finally, we try random POI selection.

**POI选择**。提出的方法的第一步，需要选择POIs集合，以设置点对点的对应性。在我们的试验中，我们研究了不同的POI选择策略。首先，我们研究了直接使用标志点作为POIs，因为有很强的语义意义，可以在介入手术前进行标注。第二，我们还研究了一种自动解决方案，使用Harris角点作为POIs，以避免标注的工作。最后，我们尝试了随机POI选择。

As shown in Figure 7(a), we find our approach is prone to overfitting when trained with landmark POIs. This is actually reasonable as each CBCT volume only contains about a dozen of landmarks, which in total is about 3, 000 POIs. Considering the variety of the field of views of our dataset, this is far from enough and leads to the overfitting. For the Harris corners, a few hundreds of POIs are selected from each CBCT volume, and we can see an improvement in performance, but the overfitting still exists (Figure 7 (b)). We find the use of random POIs gives the best performance and generalizes well to unseen data (Figure 7 (c)). This seemly surprising observation is, in fact, reasonable as it forces the model to learn a more general way to extract features at a fine-grained level, instead of memorizing some feature points that may look different when projected from a different view.

如图7(a)所示，我们发现在使用特征点POIs时，容易陷入过拟合。这实际上是很合理的，因为每个CBCT体只包含十几个特征点，总计就是约3000个POIs。考虑到我们数据集中的FOV的多样性，这远远不够，会带来过拟合。对于Harris角点，每个CBCT体中可以选择出几百个POIs，我们看到性能得到了改进，但过拟合的现象仍然存在（图7(b)）。我们发现使用随机POIs会得到最好的性能，在未曾见过的数据上泛化的也最好（图7(c)）。这看起来很令人惊讶，但实际上是合理的，因为这迫使模型学习到了一个更一般的方法来在一个更细粒度的层次上提取出特征，而不是记住一些特征点，当从一个不同的视角进行投影时，看起来会不同。

**POI Convolution**. We also explore two design options for the POI convolution layer. First, it is worth knowing that how much neighborhood information around the POI is necessary to extract a distinctive feature while the learning can still be easily generalized. To this end, we try different sizes of the feature kernel for POI convolution as given in Equation (6). Rows 1-3 in Table 1 show the performance of the POINT network with different feature kernel sizes. We observe that a 1 × 1 kernel does not give features distinctive enough for better similarity measure and a 5 × 5 kernel seems to include too much neighborhood information (and use more computation) that is harder for the model to figure out a general representation. In general, a 3 × 3 kernel serves better for the feature similarity measure. It should also be noted that a 1 × 1 kernel does not mean only the information at the current pixel location is used since each element of $F^D$ or $F^X$ is supported by the receptive field of the U-Net that readily provides rich neighborhood information. Second, we compare the performance of the POINT network with or without having the weight W in Equation (6). Rows 2 and 6 show that it is critical to have a weighted feature kernel convolution so that discriminate features can be highlighted in the similarity measure.

**POI卷积**。我们还研究了POI卷积层的两个设计选项。首先，应该直到，POI附近多少邻域信息是必须的，才能提取出一个有区分性的特征，同时学习仍然可以容易的泛化。为此，我们尝试了特征核的不同大小以进行POI卷积，如式(6)给出。表1中的第1-3行，展示了采用不同核大小的POINT网络的性能。我们观察到，1×1的核不会给出很有区分性的特征，以得到更好的相似性度量，而5×5的核似乎包含了太多邻域信息（而且使用了太多计算），模型很难得到一种一般化的表示。总体来说，3×3大小的核可以更好的满足特征相似性度量。应当指出，1×1的核并不是意味着只使用当前像素的信息，因为$F^D$或$F^X$的每个元素都是由U-Net的感受野计算得到的，已经提供了丰富的邻域信息。第二，我们比较了POINT网络在有或没有式6中的权重w的性能。行2和6表明，有一个加权特征核卷积是非常关键的，这样有区分性的特征可以在相似性度量中得到强调。

**Shift-Invariant Tracking**. The POINT network benefits from the shift invariant property of the convolution operation, which makes it less sensitive to the in-plane offset of the DRRs. Figure 8 shows some tracking results from the POINT network. Here the odd rows show the (a) X-ray and (b-d) DRR images. The heatmap below each DRR shows the tracking result between this DRR and the leftmost X-ray image. The red and the blue marks on the X-ray and DRR images denote the POIs. The red and the blue marks on the heatmaps are the ground truth POIs and the tracked POIs, respectively. The green blobs are the heatmap responses and they are used to generate the tracked POIs (blue) according to Equation (7). The numbers under each DRR denote the mPD scores before and after the tracking. As we can observe that the tracking results are consistently good, no matter how much initial offset there is between the DRR and the X-ray image. This shows that our POINT network indeed benefits from the POI convolution layer and provide more consistent outputs regardless of the in-plane offsets.

**平移不变追踪**。POINT网络从卷积运算的平移不变性质中受益，这使其对DRRs的平面内偏移并不敏感。图8展示了一些POINT网络的一些追踪效果。这里奇数行展示了(a)X射线图像和(b-d)DRR图像。每个DRR下面的热力图展示了这个DRR和最左边的X射线图像的跟踪结果。在X射线和DRR图像上的红点和蓝点表示POIs。在热力图上的红点和蓝点，分别是POIs的真值和跟踪的POIs。绿色区域是热力图响应，用于根据式(7)生成跟踪的POIs（蓝色）。在每个DRR下面的数字表示跟踪前后的mPD。我们可以观察到，跟踪结果一直很好，无论DRR和X射线图像的初始偏移到底有多大。这表明，我们的POINT网络确实从POI卷积层中获益了，不论平面内偏移有多少，都可以给出很一致的结果。

### 4.4. 2D/3D Registration

We compare our method with one learning-based (MDP [13]) and three optimization-based methods (Opt-GC [4], Opt-GO [4] and Opt-NGI [16]). To further evaluate the performances of the proposed method as an initial pose estimator, we also compare two approaches that use MDP or our method to initialize the optimization. We denote these two approaches as MDP+opt and POINT2+opt, respectively. Finally, we investigate the registration performance of our method that only uses the POINT network without the triangulation layer, and denote the corresponding models as POINT and POINT+opt. For MDP+opt, POINT+opt and POINT2+opt, we use the Opt-GC method during the optimization as we find it converges faster when the initial pose is close to the global optima.

我们的方法与一种基于学习的方法(MDP[13])，和三种基于优化的方法(Opt-GC [4], Opt-GO [4] and Opt-NGI [16])进行了比较。为进一步评估提出的方法作为一种初始姿态估计器的性能，我们还比较了使用MDP或我们的方法来初始化优化的效果。我们将这两种方法表示为MDP+opt和POINT2+opt。最后，我们研究了我们的方法的配准性能，但只使用POINT，没有三角测量层，并将对应的模型表示为POINT和POINT+opt。对MDP+opt, POINT+opt和POINT2+opt，我们在优化时使用Opt-GC方法，因为我们发现，当初始姿态与全局最优比较接近时，其收敛的更快一些。

Following the standard in 2D/3D registration [24], the performances of the proposed method and the baseline methods are evaluated with mean target registration error (mTRE), i.e., the mean distance (in mm) between the patient landmarks and the aligned CT landmarks in 3D. The mTRE results are reported in forms of the 50th, 75th, and 95th percentiles to demonstrate the robustness of the compared methods. In addition, we also report the gross failure rate (GFR) and average registration time, where GFR is defined as the percentage of the tested cases with a TRE greater than 10 mm [13].

按照2D/3D配准的标准[24]，提出的方法和基准方法的性能用mTRE评估，即，患者特征点和对齐的CT特征点在3D中的平均距离，单位为mm。mTRE结果以50th, 75th和95th百分位进行给出，以证明比较的方法的稳定性。另外，我们还给出总体失败率(gross failure rate, GFR)和平均配准时间，其中GFR定义为TRE大于10mm的测试案例。

The evaluation results are given in Table 2. We find that the optimization-based methods generally require a good initialization for accurate registration. Otherwise, they fail quickly. Opt-NGI overall is less sensitive to the initial location than Opt-GO and Opt-GC, with more than half of the registration results have less than 1 mm mTRE. Despite the high accuracy, it still suffers from the high failure rate and long registration time and so do the Opt-GO and Opt-GC methods. On the other hand, MDP achieves a better GFR and registration time by learning a function that guides the iterative pose searching. This also demonstrates the benefit of using a learning-based approach to guide the registration. However, due to the problems we have mentioned in Section 1, it still has a relatively high GFR and a noticeable registration time. In contrast, our base model POINT already achieves comparable performance to MDP; however, it runs over twice faster. Further, by including the triangulation layer, POINT2 performs significantly better than both POINT and MDP in terms of mTRE and GFR. It means that the triangulation layer that brings the 3D information to the training of the POINT network is indeed useful.

评估结果如表2所示。我们发现，基于优化的方法一般需要很好的初始化，才能进行精确的配准。否则，就会很快的失败。与Opt-GO和Opt-GC比较起来，Opt-NGI总体上对初始位置更加不敏感一些，超过一半的配准结果mTRE小于1mm。尽管准确率很高，但仍然有很高的失败率，配准时间也很长，Opt-GO和Opt-GC方法也是这样。另外，MDP通过学习一个函数，引导迭代的姿态搜索，其GFR和配准时间都更好。这也证明了，使用基于学习的方法来引导配准的好处。但是，由于我们在第1部分提到的问题，其GFR仍然相对较高，配准时间也不可忽略。对比起来，我们的基准模型POINT已经取得了与MDP类似的性能；但是，运行速度已经快了2倍。而且，加入三角测量层之后，POINT2比POINT和MDP在mTRE和GFR两个指标上都明显要好。这意味着，三角测量层将3D信息带到POINT网络的训练中，确实是有用的。

In addition, we notice that when our method is combined with an optimization-based method (POINT2 + Opt) the GFR is greatly reduced, which demonstrates that our method provides initial poses that are close to the global optima such that the optimization is unlikely to fall into local optima. The speed is also significantly improved due to faster convergence and less sampling over the pose space.

此外，我们注意到，当我们的方法与基于优化的方法结合到一起后(POINT2+Opt)，GFR会极大的降低，这说明，我们的方法会给出很好的初始姿态，与全局最优值是接近的，这样优化就不太会陷入局部极值。由于收敛更快，在姿态空间中需要的采样更少，速度也显著得到了改进。

## 5. Limitations

First, similar to other learning-based approaches, our method requires a considerably large dataset from the targeting medical domain for learning reliable feature representations. When the data is insufficient, the proposed method may fail. Second, although our method alone is quite robust and its accuracy is state-of-the-art through a combination with the optimization-based approach, it is still desirable to come up with a more elegant solution to solve the problem directly. Finally, due to the use of triangulation, our method requires X-rays from at least two views to be available. Hence, for the applications where only a single view is acceptable, our method will render an estimate of registration parameter with inherent ambiguity.

第一，与其他基于学习的方法类似，我们的方法需要一个相对较大的数据集，以学习可靠的特征表示。当数据不足时，提出的方法可能会失败。第二，虽然我们的方法很稳定，其准确率与基于优化的方法一起得到了目前最好的性能，但如果能有一种更优雅的解决方法，直接解决这个问题，则更理想。最后，由于使用的三角测量，我们的方法需要至少2个视角的X射线图像可用。因此，对于只有单个视角的图像可用的应用，我们的方法得到的配准参数估计，会带有内在的模糊性。

## 6. Conclusion

We proposed a fast and robust method for 2D/3D registration. The proposed method avoids the often costly and unreliable iterative pose searching by directly aligning the CT with the patient through a novel POINT2 framework, which first establishes the point-to-point correspondence between the pre- and intra-intervention data in both 2D and 3D, and then performs a shape alignment between the matched points to estimate the pose of the CT. We evaluated the proposed POINT2 framework on a challenging and large-scale CBCT dataset and showed that 1) a robust POINT network should be trained with random POIs, 2) a good POI convolution layer should be convolved with weighted 3 × 3 feature kernel, and 3) the POINT network is not sensitive to in-plane offsets. We also demonstrated that the proposed POINT2 framework is significantly more robust and faster than the state-of-the-art learning-based approach. When used as an initial pose estimator, we also showed that the POINT2 framework can greatly improve the speed and robustness of the current optimization-based approach while attaining a higher registration accuracy. Finally, we discussed several limitations of the POINT2 framework which we will address in our future work.

我们提出了一种快速稳定的方法，进行2D/3D配准。提出的方法避免了迭代姿态搜索的昂贵计算和不可靠性，将CT与患者通过新颖的POINT2框架进行直接对齐，首先确立了介入前和介入中的数据的点到点的对应性，然后对匹配的点之间进行形状对齐，以估计CT的姿态。我们在一个很有挑战性的大型CBCT数据集上评估提出的POINT2框架，表明1)应当用随机POIs训练一个稳健的POINT网络，2)POI卷积层应当用加权3 × 3特征核进行卷积，3)POINT网络对平面内平移不敏感。我们还证明了，提出的POINT2框架比目前最好的基于学习的方法要明显更稳定更快速。当用作初始姿态估计时，我们表明POINT2框架可以极大的改进速度和稳定性，同时保持很高的配准准确率。最后，我们讨论了POINT2框架的几个局限，我们在未来的工作中会进行解决。