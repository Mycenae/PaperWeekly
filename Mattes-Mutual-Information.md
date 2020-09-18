# Nonrigid multi-modality image registration

David Mattes et. al. University of Washington

## 0. Abstract

We have designed, implemented, and validated an algorithm capable of 3D PET-CT registration in the chest, using mutual information as a similarity criterion. Inherent diferences in the imaging protocols produce significant non-linear motion between the two acquisitions. To recover this motion, local deformations modeled with cubic B-splines are incorporated into the transformation. The deformation is defined on a regular grid and is parameterized by potentially several thousand coefficients. Together with a spline-based continuous representation of images and Parzen histogram estimates, the deformation model allows for closed-form expressions of the criterion and its gradient. A limited-memory quasi-Newton optimization package is used in a hierarchical multiresolution framework to automaticaly align the images. To characterize the performance of the algorithm, 27 scans from patients involved in routine lung cancer screening were used in a validation study. The registrations were assessed visually by two observers in specific anatomic locations using a split window validation technique. The visually reported errors are in the 0-6 mm range and the average computation time is 10 minutes.

我们设计实现并验证了一种算法，可以对胸部的3D PET-CT进行配准，使用互信息作为相似度准则。成像协议的内在差异，在两种图像获取方式中产生了显著的非线性运动。为恢复这种运动，使用三次B样条建模的局部变形纳入到了变换中。变形是在规则网格上定义的，可能有数千个系数的参数。与图像的一种基于样条的连续表示和Parzen直方图估计一起，形变模型可以进行规则及其梯度的封闭形式的表示。有限存储的拟牛顿优化包，用在层次化的多分辨率框架中，对图像进行自动对齐。为表现算法性能的特征，肺癌筛查的27位患者的scans用于验证研究。配准由两位观察者通过肉眼观察，对特定解剖位置进行评估，使用的是一种分割窗口验证方法。视觉上给出的误差在0-6mm范围，平均计算时间为10分钟。


**Keywords**: registration, nonrigid, deformation, nonlinear, multimodality, validation, multiresolution, mutual information, positron emission tomography(PET),computed tomography(CT).

## 1. Introduction

We define the problem of medical image registration as follows. Given two image sets acquired from the same patient but at different times or with different devices, we seek the geometric transformation g between the two respective image-based coordinate systems that maps a point x in the first image set to the point g(x) in the second set that has the same patient-based coordinates, i.e. represents the same anatomic location. This notion presupposes that the anatomy is the same in the two image sets, an assumption that may not be precisely true if, for example, the patient has had a surgical resection between the two acquisitions. The situation becomes more complicated if two image sets that reflect different tissue characteristics (e.g. CT and PET) are to be registered. The idea can still be used that, if a candidate registration g matches features in the first set to similar features in the second set, it is probably correct. For example, according to the principle of mutual information homogeneous regions of the first image set should generally map into homogeneous regions in the second set[1],[2]. The beauty of the information theoretic measures is that they make no assumptions about the actual intensity values in the images, instead they measure statistical relationships between the two images. The mutual information metric has been effective in various applications where the requisite transformations are linear[2],[3]and more recently in cases involving non-linear motion descriptions[4].

我们将医学图像配准问题定义如下。给定两幅图像集，是在不同时间或使用不同设备对同一病人获取的图像，我们在两个基于图像的坐标系系统中找到几何变换g，将第一幅图像集中的点x映射到第二个图像集中的g(x)，有着相同的基于患者的坐标系，即，表示同样的解剖位置。这种表示有一种预设的假设，即解剖结构在两个图像集中是一样的，这种假设可能不会是很精确的正确的，比如，如果患者在两个获取的图像集之间做了手术切除。如果两个图像集反应的是不同的组织特性（如CT和PET），要对其进行配准，情况就会变得更加复杂。如果一个候选的配准g将第一个集合中的特征与第二个集合中类似的特征进行了匹配，那么很有可能是正确的，则这个思想也可以应用。比如，根据互信息准则，第一幅图像集中的同种类区域一般会映射到第二个集合中的同种类区域[1,2]。信息论的度量的美感在于，他们对图像中的实际灰度值不做假设，而度量的是，两幅图像中的统计关系。互信息度量在很多应用中都是有效的，其中必须的变换是线性的[2,3]，最近涉及的更多的是非线性运动描述[4]。

We have concentrated our efforts on PET-to-CT image registration in the chest, where we attempt to fuse images from a modality with high anatomic detail (CT) with images from a modality delineating biological function (PET). Although PET is a functional imaging modality (it measures uptake of radioactive tracers in cells with high metabolic activity), a transmission (TR) image is acquired immediately before acquisition of the emission image and is therefore in near-perfect registration with the functional scan. The TR image is similar to a CT attenuation map but it uses a higher energy radiation beam, resulting in less soft-tissue detail than the CT, and detector configuration limits its in-plane resolution. If we can register the TR and CT images, we can apply the resulting transformation to the emission or standard uptake value (SUV) image for improved PET image interpretation.

我们关注的是胸部的PET-CT图像配准，其中我们希望融合两个模态的图像，一个模态有很高的解剖细节(CT)，另一个模态描述的是生物功能(PET)。虽然PET是一种功能成像模态（其衡量的是放射示踪剂在高代谢活动细胞中的吸收量），但传输(TR)的图像在获取发射图像前就立刻获取了，因此与功能scan是接近完美的配准。TR图像与CT衰减图类似，但使用的是更高能量的放射射束，比CT在软组织上有更少的细节，而且探测器的配置限制了其平面内的分辨率。如果我们可以配准TR和CT图像，我们可以将得到的变换应用到发射图像或标准吸收值(SUV)图像中，以改进PET图像解释。

Patient and anatomic motion during acquisition blurs the images. Sharp anatomic outlines in CT scans are obtained by requesting the patient to maintain maximum inspiration during the 30 seconds required for acquisition. To avoid attenuation of the arms in the X-ray beam, the patient also holds the arms overhead if possible. Anatomicaly, this arm motion causes expansion of the arm muscles. Also,the expansion of the lungs and chest wall due to the breath hold cause descent of the diaphragm and abdominal organs. Most patients can not endure an arms-up posture for the duration of a PET scan, which can last up to 30 minutes, and will be engaged in normal tidal breathing. As a result, PET scans show an average of the anatomy over the respiratory cycle. Clearly, a linear transformation model is not sufficient to match anatomic regions under these wildly varying conditions.

在获取图像的过程中，患者和解剖结构的运动会使得图像变得模糊。在CT scans中，尖锐的解剖结构的轮廓的获得，要求患者在30秒的图像获取时间内，保持最大吸气状态。为避免手臂对X射线射束的衰减，如果可能的话，会要求患者将手臂举过头部。解剖上，手臂的这种姿态会导致手臂肌肉的扩张。同时，由于屏气导致的肺部和胸廓的扩张，会导致隔膜和腹部器官的下降。PET扫描一般会持续30分钟，多数病人不能忍受这么长时间手臂上举的姿势，会进入正常的呼吸状态。结果是，PET扫描表现的是呼吸循环中解剖结构的平均状态。很明显，线性变换模型在这种狂野变化的条件下，不足以匹配解剖区域。

Some example images of the relevant anatomy are shown in figure 2. Why is registration useful?Generally speaking there are many answers to this question. In the context of our efforts, we have identified several areas in which our methods are directly applicable to a clinical or research setting and have divided them into four groups: improved PET image interpretation,response to therapy, tumor biology, and image-guided therapy.

相关解剖结构的一些例子图像，如图2所示。为什么配准会有用？一般来说，对这个问题有很多答案。在我们的工作的上下文中，我们发现几个区域，我们的方法在临床或科研设置下可以直接应用，并将其分为四个组：改进的PET图像解释，对治疗的响应，肿瘤的生物学，以及图像引导治疗。

## 2. Methodology

Registration is the process of aligning two datasets (a test and a reference image) according to a given metric. We use mutual information as our image similarity function. Mutual information is a statisticaly-based measure of image alignment derived from probabilistic measures of image intensity values and is robust in the presence of noise and varying amounts of image overlap. Our formulation requires that we express mutual information as a continuous function to allow its explicit differentiation. Doing so requires that the components used in computing mutual information also be expressed as continuous functions. This is accomplished in three ways: using a B-spline basis to represent the test image, estimating the joint probability distribution between the test and reference images with a Parzen window, and implementing deformations modeled with cubic B-splines. We draw on the work of Thevenaz and Unser [3] for the mathematical development of the first two points. We introduce our non-linear transformation model which is used to incorporate deformations into the geometric manipulation of the test image.

配准是根据给定度量，对齐两个数据集（一个测试和一个参考图像）的过程。我们使用互信息作为我们的图像相似度函数。互信息是基于统计的图像对准度量，从图像灰度的概率度量中导出，对噪声的存在和不同程度的图像重叠是稳定的。我们的表述需要，我们将互信息表述为一个连续函数，以允许显式的微分。这样需要在计算互信息时使用的组件也表示为连续函数。这以三种方式完成：使用一个B样条基来表示测试图像，使用Parzen窗在测试图像和参考图像之间估计联合概率分布，实现用三次B样条建模的变形。我们从Thevenaz和Unser[3]中提取出了前两个点的数学发展。我们提出我们的非线性变换模型，用于将变形纳入到测试图像的几何操作中。

### 2.1 Deformations

One important aspect of our algorithm is the expression for the transformation of image coordinates. We model deformations on cubic B-splines, because of their computational efficiency (via separability in multidimensional expressions), smoothness, and local control. A deformation is defined by placing a sparse, regular grid of control points over the volume and varied by moving the control points. Cubic splines are used to distribute this coarse grid of deformation coefficients over the entire image. Using a spline convolution kernel to interpolate the deformation values between the control points produces a locally-controlled, globbaly-smooth deformation. The region of support of the 3D cubic spline convolution window means each pixel in the test image is deformed by the cube of 64 nearest deformation parameters. Using splines for a deformation model also allows simple computation of deformation derivatives with respect to both a deformation parameter (for computing the gradient of mutual information) and a spatial coordinate (useful for estimating inverse coordinates).

我们的算法一个重要的方面是，表示图像坐标系的变换。我们在三次B样条上对形变进行建模，因为其计算效率高（通过在不同维度的表示上的分离），平滑度，和局部控制。一个形变是通过将一个稀疏的规则网格的控制点置于体中定义的，通过移动控制点来变化。三次样条用于将粗糙网格的形变参数分布到整个图像中。使用一个样条卷积核来在控制点中对形变值插值，会产生一个局部控制的、全局平滑的形变。3D三次样条卷积窗的支持区域，意味着在测试图像中每个像素是由64个最接近的形变参数的立方体来形变的。对形变模型使用样条，还使得形变对形变参数的导数（用于计算互信息的梯度）和空间坐标系的导数（用于估计逆坐标）可以很简单的计算。

The resolution of the deformation is defined by the spacing of the grid, which can vary for each image dimension. The vector of grid point densities is a compact notation for the deformation resolution: 形变的分辨率，是通过网格的间隔定义的，对每个图像维度来说都不太一样。网格点密度的向量是形变分辨率的紧凑表示：

$$n_D = [n_x, n_y, n_z]^T$$(1)

Once nD is specified, the control points are defined by their coordinates in the volume; there will be nx•ny•nz such control points, each with a 3-tuple of (x,y,z) coordinates and a constant spacing between them. If the image dimensions are q = (q_x, q_y, q_z), then the vector of grid spacings is:

一旦指定了nD，控制点用在体中的坐标系进行定义；会有nx•ny•nz个这样的控制点，每个都带有一个三元组的(x,y,z)坐标，和之间的常数间隔。如果图像维度为q = (q_x, q_y, q_z)，那么网格间距的矢量为：

$$Δx_D = [Δx_D, Δy_D, Δz_D]^T = [\frac {q_x-1}{n_x-1}, \frac {q_y-1}{n_y-1}, \frac {q_z-1}{n_z-1}]^T$$(2)

Given the grid spacing, we can compute the coordinates of the control points. The control points are represented by the set of control point matrices, {x_D} = {x_D, y_D, z_D}. Each of these components is a 3D matrix of x-,y-,or z-coordinates, respectively; i.e. each control point is specified by a (x,y,z) coordinate in the volume. Any (i,j,k)element of these matrices is computed as:

给定网格间距，我们可以计算控制点的坐标。控制点是通过控制点矩阵的集合表示的，{x_D} = {x_D, y_D, z_D}。这些组成部分的每一个都是一个3D矩阵，分别是x-,y-,或 z-坐标；即，每个控制点都是由体中的一个(x,y,z)坐标来指定。这些矩阵的任意(i,j,k)元素计算为：

$$x_D(i,j,k) = i⋅Δx_D$$
$$y_D(i,j,k) = j⋅Δy_D$$(3)
$$z_D(i,j,k) = k⋅Δz_D$$

Each control point $x_D(i) = (x_D(i), y_D(i), z_D(i)$ has an associated deformation term $c_D(i) = (c_{D,x}(i), c_{D,y}(i), c_{D,z}(i)$ that is composed of 3 separate parameters.

每个控制点$x_D(i) = (x_D(i), y_D(i), z_D(i)$都有一个相关的形变项$c_D(i) = (c_{D,x}(i), c_{D,y}(i), c_{D,z}(i)$，由三个分离的参数组成。

We express the general form of a transformation as a locally-perturbed rigid body transformation. Given a 3x3 homogeneous rotation matrix R, a 3-element transformation vector T, and a deformation term $D(c_D;x)$, we can apply non-linear transformations to the test image:

我们将一般形式的变换表达为一个局部扰动的刚体变换。给定一个3x3的齐次旋转矩阵R，一个3元素的变换向量T，和一个形变项$D(c_D;x)$，我们可以对测试图像进行非线性变换：

$$g(x;μ) = R(x—x_C)—(T—x_C)+D(c_D;x)$$(4)

where $x_c = (nx/2, ny/2, nz/2)^T$ is the location of the center of the volume. With this transformation model, the set of transformation parameters becomes:

其中$x_c = (nx/2, ny/2, nz/2)^T$是体的中心的位置。有了这个变换模型，变换参数变成了：

$$μ=\{ γ, θ, ϕ, t_x, t_y, t_z; \{c_D\} \}$$(5)

where (γ,θ,ϕ) are the Roll-Pitch-Yaw(RPY) Euler angles of the rotation matrix, ($t_x,t_y,t_z$) are the translation components, and {$c_D$} is the set of deformation coefficients. In practice the set of deformation coefficients contains ~1O^2-1O^3 members.

其中(γ,θ,ϕ)是Roll-Pitch-Yaw(RPY)旋转矩阵的欧拉角，($t_x,t_y,t_z$)是平移参数，{$c_D$}是形变系数集合。实践中，形变系数的集合包含~1O^2-1O^3个参数。

The deformation is defined on a uniform grid by the grid control points {$x_D$} the spacing between the control points $Δx_D$ and the set of deformation coefficients {$c_D$} and is locally distributed to neighboring voxels with cubic B-splines:

形变是由控制点{$x_D$}，控制点之间的间距$Δx_D$，和形变系数{$c_D$}的集合，定义在均匀网格上，用三次B样条分布到邻近的体素中：

$$D(c_D;x) = \sum_h c_D(h) β^{(3)} (\frac {x-x_D(r)}{Δx_D} - h)$$(6)

where r = $⌊x/(Δx_D)⌋$ is the central grid point, and the index of summation is determined from r. 其中r = $⌊x/(Δx_D)⌋$是中央网格点，求和的索引是由r来决定的。

### 2.2 Interpolation

The test image is resampled at locations determined by applying a transformation g(x;μ) to the voxel coordinates x∈V of the region of image overlap according to the transformation parameters μ. The interpolation scheme relies on a cubic B-spline basis representation of the test image. The spline coefficients are determined through an efficient recursive filtering algorithm [5]. Image interpolation is accomplished via separable convolution of the spline coefficients {c} with the sampled cubic B-spline window [3]:

测试图像在一些位置进行重采样，这些位置是由对体素坐标x∈V应用一个变换g(x;μ)来决定的，μ是变换参数。插值方案依赖于测试图像的三次B样条基的表示。样条系数是通过一个高效的递归滤波算法决定的。图像插值是通过样条系数{c}与采样的三次B样条窗的可分离的卷积完成的：

$$f_T(g(x;μ)) = \sum_i c(x_i) β^{(3)} (g(x;μ)-x_i)$$(7)

where $x=[x,y,z]^T$ is any voxel location in the reference image and the summation range is over the support of the cubic spline window.

其中$x=[x,y,z]^T$是在参考图像中的任意体素位置，求和范围是在三次样条窗的支持域中。

The gradient of the transformed test image at each of the voxel locations is resampled in a similar manner,but a derivative operator is applied to the convolution, which is just the derivative of the spline window in the respective dimension of each gradient component:

变换的测试图像在每个体素位置上的体素，也通过类似的方式进行重采样，但对卷积算子使用了导数运算，即是样条窗对每个梯度部分在分别维度上的导数：

$$\frac {df_T(t)}{dt} = \sum_i c(x_i) \left[ \begin{matrix} \frac{∂β^{(3)}(u)}{∂u}|_{u=(t-x_i)_x} β^{(3)}((t-x_i)_y) β^{(3)}((t-x_i)_z) \\ β^{(3)}((t-x_i)_x) \frac{∂β^{(3)}(u)}{∂u}|_{u=(t-x_i)_y} β^{(3)}((t-x_i)_z) \\ β^{(3)}((t-x_i)_x) β^{(3)}((t-x_i)_y) \frac{∂β^{(3)}(u)}{∂u}|_{u=(t-x_i)_z} \end{matrix} \right]$$(8)

### 2.3 Mutual information

We pose the task of medical image registration as a function optimization problem. We desire the set of transformation parameters μ that maximizes an image similarity function S:

我们将医学影像配准的问题作为一个函数优化问题来解决。我们期望变换参数μ的集合可以最大化图像相似度函数S：

$$μ_{opt} = argmax_μ S(μ)$$(9)

In our implementation, mutual information is used as for the image similarity function. We hypothesize that the set of transformation parameters {$μ_opt$} that maximizes the similarity function also brings the transformed test image into best corespondence with the reference image. Having determined this set of parameters does not ensure that images are actualy geometrically aligned, only that these parameters maximize (perhaps only locally) the similarity criterion.

在我们的实现中，互信息用作图像相似度度量。我们假设变换参数的集合{$μ_opt$}，可以将相似度函数最大化，也会使变换过后的测试图像与参考图像与最佳的对应性。这个参数集合的确定，并不保证图像在实际中已经几何对齐了，只是说这些参数将相似度准则最大化了（可能只是局部的）。

The problem as presented in equation 9 is stated as a maximization problem, but we will actually minimize the negative of the function S. Let L_T and L_R be discrete sets of intensities associated to the test and reference image,respectively. The negative of mutual information, S, between the reference image and the transformed test image is expressed as a function of the transformation parameters, μ[3]:

式(9)中给出的问题表述成了一个最大化问题，但我们实际上会对函数-S进行最小化。令L_T和L_R分别为测试图像和参考图像相关的离散灰度集。参考图像和变换的测试图像的互信息S的负值，表述为变换参数μ的函数：

$$S(μ) = \sum_{ι∈L_T} \sum_{κ∈L_R} p(ι,κ;μ) log\frac{p(ι,κ;μ)}{p_T(ι;μ)p_R(κ)}$$(10)

where p, p_T and p_R are the joint, marginal test, and marginal reference probability distributions, and will be derived shortly.

其中p, p_T和p_R是联合概率分布，测试图像边缘概率分布，参考图像边缘概率分布，马上我们进行推导得到这些值。

Due to the high dimensionality of the space of transformation parameters, the gradient of the criterion facilitates the search for its maximum. The gradient of mutual information is given as

由于变换参数空间维度特别高，准则的梯度会有助于搜索其最大值。互信息的梯度由下式给出

$$∇S = [\frac{∂S}{∂μ_1}, \frac{∂S}{∂μ_2}, ..., \frac{∂S}{∂μ_i}, ..., \frac{∂S}{∂μ_n}]^T$$(11)

A single component of the gradient is found by diferentiating equation 10 with respect to a transformation parameter:

梯度的单个组件，是将式10对一个变换参数求微分：

$$\frac{∂S}{∂μ_i} = -\sum_{ι∈L_T} \sum_{κ∈L_R} \frac{∂p(ι,κ;μ)}{∂μ_i} log \frac {p(ι,κ;μ)}{p_T(ι;μ)}$$(12)

where $∂p(ι,κ;μ)/∂μ_i$ is the i-th partial derivative of the joint distribution. 其中$∂p(ι,κ;μ)/∂μ_i$是联合分布的第i个偏导数。

The probability distributions used to compute mutual information are based on marginal and joint histograms of the reference and test images. Parzen windowing is used to form continuous estimates of the underlying image histograms,also reducing the effects of quantization from interpolation and discretization from binning the data[3]. The joint distribution is therefore an explicitly differentiable function. Let $β^{(3)}$ be a cubic spline Parzen window and $β^{(0)}$ be a zero-order spline Parzen window (centered unit pulse), both of which satisfy the partition of unity constraint[3]. The joint discrete probability is given by:

用于计算互信息的概率分布，是基于参考图像和测试图像的边缘直方图和联合直方图。Parzen窗用于形成潜在的图像直方图的连续估计，也降低数据分组中的插值和离散化的量化的效果。联合分布因此是一个显式可微分的函数。令$β^{(3)}$为一个三次样条Parzen窗，$β^{(0)}$为一个零阶样条Parzen窗（中央为单位脉冲），两者都满足分割的单位约束。联合离散概率由下式给出：

$$p(ι,κ;μ)=α\sum_{x∈V} β^{(0)} (κ-\frac{f_R(x)-f'_R}{Δb_R}) β^{(3)} (ι-\frac{f_T(g(x;μ))-f'_T}{Δb_T})$$(13)

where α is a normalization factor that ensures $\sum p(ι,κ)=1$, ι∈L_T and κ∈L_R, and f_R(x) and f_T(g(X;μ) are samples of the reference and interpolated test images, respectively. Each contribution is normalized by the minimum intensity value, f'_R or f'_T, and the intensity range of each bin, Δb_R or Δb_T, to fit into a specified number of bins in the intensity distribution. The summation range V is the set of voxel pairs that contribute to the distribution.

其中α是一个归一化因子，确保$\sum p(ι,κ)=1$, ι∈L_T, κ∈L_R, f_R(x)和f_T(g(X;μ)分别是参考图像和插值的测试图像的样本。每个贡献都由最小灰度值f'_R或f'_T进行归一化，每个分组的灰度范围Δb_R或Δb_T，在灰度分布中适应特定数量的分组数量。求和范围V是对这个分布有贡献的体素对的集合。

The marginal discrete probability for the test image is computed from the joint distribution: 测试图像的边缘离散概率从联合分布中计算得到：

$$p_T(ι;μ) = \sum_{κ∈L_R} p(ι,κ;μ)$$(14)

The marginal discrete probability for the reference image can be computed independently of the transformation parameters by noting that the B-spline Parzen window satisfies the partition of unity constraint. The reference marginal distribution is computed as:

参考图像的边缘离散概率，可以独立于变换参数计算，要注意B样条Parzen窗满足分割的单位约束。参考边缘分布计算为：

$$p_R(κ) = α\sum_{x∈V} β^{(0)} (κ-\frac{f_R(x)-f'_R}{Δb_R})$$(15)

The derivative of the joint distribution with respect to one of the transformation parameters is: 联合分布对一个变换参数的导数为：

$$\frac {∂p(ι,κ)}{∂μ} = \frac {1}{Δb_T |V|} \sum β^{(0)} (κ-\frac{f_R(x)-f'_R}{Δb_R}) \frac {∂β^{(3)}(ξ)}{∂ξ}|_{ξ = ι-\frac{f_T(g(x;μ))-f'_T}{Δb_T}} (-\frac{df_T(t)}{dt}|_{t=g(x;μ)})^T \frac {∂g(x;μ)}{∂μ}$$(16)

where #V is the number of voxels used in the sumation. 其中#V是求和中用到的体素数量。

The final term to discuss from equation 16 is the expression for the partial derivatives of the transformation $\frac {∂g(x;μ)}{∂μ}$. This is the variation in position due to a variation in transformation parameter, and depends on geometry and the transformation model. The linearity of the expression of the transformation makes the differentiation of equation 4 straightforward, i.e. derivatives with respect to the rotation angles will depend only on the term $R(x—x_C)$, while the derivatives with respect to the transformation parameters are even simpler. The partial derivatives with respect to the deformation parameters are more complicated and merit additional attention.

式16要讨论的最后一个项，是变换的偏导数$\frac {∂g(x;μ)}{∂μ}$。这是由于变换参数的变化导致的位置变化，依赖于几何参数和变换模型。表达式对变换的线性性使得式4的微分很直观，即，对旋转角度的导数会只依赖于项$R(x—x_C)$，而对变换参数的导数甚至更简单。对变换参数的偏导数更加复杂，需要更多的注意力。

If we expand the sum in equation 6, keeping the vector notation but omitting the vector permutations of the index of summation h, we have: 如果我们将式6中的求和进行扩展，保持矢量的表示，但忽略求和h的索引的矢量扰动，则有：

$$D(x) = c_D(r-1) β^{(3)} (\frac {x-x_D(r)}{Δx_D} -(r-1)) + c_D(r) β^{(3)} (\frac {x-x_D(r)}{Δx_D} -r) + c_D(r+1) β^{(3)} (\frac {x-x_D(r)}{Δx_D} -(r+1)) + c_D(r+2) β^{(3)} (\frac {x-x_D(r)}{Δx_D} -(r+2))$$(17)

When we take the partial derivative of equation 17 with respect to a 3-tuple of deformation coefficients c_D(r), we have a single non-zero term:

当我们取式17我们对一个3元组的变换系数c_D(r)的偏导数，我们会得到一个非零项：

$$\frac {∂D(x)}{∂c_D(r)} = β^{(3)} (\frac {x-x_D(r)}{Δx_D} -r)$$(18)

This is the 3-tuple of deformation derivatives: 这是形变导数的3元组：

$$\frac {∂D(x)}{∂c_D(r)} = [\frac {∂D(x)}{∂c_{D,x}(r_x)}, \frac {∂D(x)}{∂c_{D,y}(r_y)}], \frac {∂D(x)}{∂c_{D,z}(r_z)}^T$$(19)

We will only be interested in the deformation derivative with respect to a single parameter. However, the various deformation derivatives will still be 3-element vectors. The derivative with respect to a x-component deformation coefficient is (the y- and z-components follow similarly):

我们只对对单个参数的形变导数感兴趣。但是，各种形变导数都将是3元素向量。对x元素形变系数的导数是（对y和z元素的都有类似的结果）：

$$\frac {∂D(x)}{∂c_{D,x}(r_x)} = [β^{(3)} (\frac {x-x_D(r_x)} {Δx_D} - r_x), 0, 0]^T$$(20)

### 2.4 Multiresolution optimization strategy

Minimization problems for large sample-based datasets are often benefited by a multi-resolution approach. Instead of constructing an image pyramid, we approach this technique in a somewhat different fashion. The registration process is automated by varying the deformations in the test image such that the mutual information of the two images is maximized, at which point corresponding structures in the reference image are brought into geometric alignment. We use L-BFGS-B[6], a limited-memory, quasi-Newton minimization package, to descend the expression for mutual information in equation 10 until termination criteria are satisfied. In order to avoid local minima, and to decrease computation time,the algorithm has been cast into the framework of a hierarchical multiresolution optimization scheme. A flow chart of the algorithm in the multiresolution context is shown in figure 1.

基于样本的大型数据集的最小化问题，多分辨率方法通常会使问题受益。我们没有构建一个图像金字塔，而是采用了一种有些不同的方法。配准过程的自动化，是通过变化测试图像的形变完成的，然后使得两幅图像的互信息最大化，在这一点上参考图像中的对应结构得到几何上的对准。我们使用L-BFGS-B，一个有限内存的拟牛顿最小化包，以减少式10中的互信息值，直到满足停止条件。为避免陷入局部极小值，并减少计算时间，算法放入了一个层次化的多分辨率优化方案中。算法在多分辨率下的流程图，如图1所示。

We keep the image size the same for each resolution step, but vary several other parameters as the minimization proceeds from coarser to finer resolution: the resolution of the deformation, the number of image samples used in computing mutual information, the degree of Gaussian image bluring, and the optimizer's termination criteria. We vary the parameters according to an empirically determined schedule, but following a general rule that the number of image samples is proportional to the deformation resolution. The resolution steps are denoted as n_1,..., n_m, where m is the number of multiresolution steps(typically m=4). The solution vectors of transformation parameters at coarser resolution levels become the starting vectors for the finer resolution levels.

我们在每个分辨率步骤中都保持同样的图像大小，但当最小化的过程从粗糙变换到更细节的分辨率中，会变化几个其他的参数：形变的分辨率，计算互信息时使用的图像样本的数量，高斯图像模糊的程度，和优化器的停止准则。我们根据经验确定的方案来变化参数，但遵循一个通用的规则，即图像样本的数量与形变分辨率成正比。分辨率步骤表示为n_1,..., n_m, 其中m是多分辨率步骤的数量（一般m=4）。在更粗糙的分辨率中的变换参数解向量会成为更精细的分辨率层次中的初始解向量。

One aspect of our multiresolution strategy is the hierarchical recovery of the deformation. In our approach, we initially recover a global, overall deformation in an attempt to match up the patient motion and lung boundaries. As we increase the resolution (in a multiresolution sense), we try to recover increasingly fine deformations. The resolution of the deformation grid is specified in terms of the number of control points placed over the image and is specified separately for each dimension of the deformation. As the number of control points increases, so does the deformation resolution, and the spacing between the control points decreased. If $n_D = [n_x, n_y, n_z]$ is a 3-tuple of numbers of control points in the deformation, and $n_{D,1}, ..., n_{D,4}$ are the control points density vectors for each of the four multiresolution steps, then we proceed from resolution 1 to resolution 4 as follows. A deformation at resolution level r is positioned on the volume at the control point locations {$X_{D,r}$} referenced by its spline deformation coefficients {$C_{D,r}$}, i.e.{$x_{D,3}$} and {$c_{D,3}$} represent the deformation at resolution step 3.

我们的多分辨率策略的一个方面，是形变信息的层次化恢复。在我们的方法中，我们开始只恢复一个全局的、总体的形变信息，以匹配患者的运动和肺部边缘。当我们增加分辨率（在多分辨率的意义下），我们尝试越来越细致的形变。形变网格的分辨率，是由图像上控制点的数量来指定的，对每个形变的维度是单独指定的。随着控制点数量的增加，形变分辨率也会增加，控制点之间的间距下降。如果$n_D = [n_x, n_y, n_z]$是形变中的3元组控制点，$n_{D,1}, ..., n_{D,4}$是4个多分辨率步骤中每个步骤的控制点密度向量，那么我们从分辨率1到分辨率4按照如下进行。在分辨率层级r上的形变，在体中放置在控制点位置{$X_{D,r}$}上，由其样条形变系数{$C_{D,r}$}参考，即，{$x_{D,3}$}和{$c_{D,3}$}表示在分辨率步骤3上的形变。

If we are at current deformation resolution $n_{D,1}$ and have a deformation defined on this grid, given by the deformation coefficients at this resolution {$C_{D,1}$} then we first place a new deformation grid over the volume at resolution $n_{D,2}$. Next we compute what the deformation coefficients {$C_{D,2}$} need to be in order to have the same deformation at this higher resolution (this becomes our starting point for the second resolution step). We do this by calculating the deformation {$C_{D,1}$} at the first resolution level for each of the control point locations{$x_{D,2}$} of the second:

如果我们在目前的形变分辨率$n_{D,1}$上，有一个定义在此网格上的形变，由在此分辨率{$C_{D,1}$}上的形变系数给出，那么我们首先在分辨率$n_{D,2}$的体上放置一个新的形变网格。下一步我们计算形变系数{$C_{D,2}$}需要是什么样子，才能在更高的分辨率中有相同的形变（这成为了我们在第二个分辨率步骤中的初始点）。我们通过在第一个分辨率层次计算形变{$C_{D,1}$}，计算每个控制点在第二个分辨率层次上的位置{$x_{D,2}$}：

$$D(x_{D,2}) = \sum_i c_{D,1}(i) β^{(3)} (\frac {x_{D,2}-x_{D,1}(i)}{Δx_{D,1}})$$(21)

Then we compute the spline coefficients of the new grid of deformation values: 然后我计算新的网格的形变值的样条系数：

$$c_{D,2} = Γ(D(x_{D,2}))$$(22)

where Γ is the same recursive filter used to compute a spline basis of the test image. 其中Γ是相同的递归滤波器，用于计算测试图像的样条基。

We want the lower resolution steps to converge quickly and bring the independent variables(that define the deformation)close to maximizing mutual information. Besides starting with low resolution deformations, Gaussian blurring is applied to the images with a kernel that narrows as multiresolution proceeds. Smoothing the images tends to smooth the mutual information critereion, thereby avoiding local maxima.

我们希望较低的分辨率步骤快速收敛，以使独立的变量（定义了形变）接近于最大化互信息。除了从低分辨率形变开始，高斯平滑也用于图像，其平滑核随着多分辨率的进行逐渐收窄。平滑图像会倾向于平滑互信息的准则，因此避免局部极值。

### 2.5 Validation

Validating the performance of an image registration algorithm presents a host of challenges, and the poor scores for literature searches on this topic highlight the difficulties. The lack of a gold standard complicates matters further, preventing any quantitative assessment of registration accuracy. Even if individuals trained to interpret medical images are involved in a validation experiment, providing a method for consistently assessing individual images is difficult, and time-consuming. There is a trade-off between the number of images that can be assessed and the time required to assess each one, a situation that often forces researchers to validate based on a limited sample size.

验证图像配准算法的性能，包含很多挑战，搜索文献的结果也很少，说明这个问题难度也很大。缺少金标准使得这个问题更加复杂，不能得到配准准确率的定量评估。即使受到过解读医学影像训练的人进行验证试验，要一致的评估单个图像也是很难的，非常耗时的。要评估的图像数量，每幅图像评估的耗时，有一个折中关系，这种情况要求研究者只能基于有限的样本大小来进行验证。

For algorithms with linear transformation models, retrospective validation can be performed if fiducial markers are in place at the time the scan is performed, a method that will in general fail as the imaged anatomy is subject to non-linear motion either as the patient is moved between scanners or due to physiological effects such as breathing. In cases where placing fiducial markers is impractical or imposible, researchers often conduct validation experiments using simulate data, generated with a phantom or an image simulation algorithm. Such studies provide insight into the performance range of the algorithm but still mean little in terms of absolute performance once the algorithm is applied to real data. Landmark identification is another common method of validation, and was considered for our validation experiments. A difficulty with this technique is that there are few (if any) point landmarks in the anatomy. Structures like the carina, which shows the split in the trachea and is well-defined in our images, do not terminate at a single 3D coordinate, but are distributed over a surface. There will always be some degree of error in the selection of corresponding points.

对于线性变换模型的算法，如果scan时是带有标记的，那么可以进行回顾式的验证的，但由于成像的解剖结构是存在非线性运动的，比如患者在扫描时在scanners之间移动了，或是由于生理效应，如呼吸，所以这类验证方法通常会失效。如果放置基准标记不切实际或不可能，研究者通常使用模拟数据进行验证试验，用模体或图像仿真算法来生成。这样的研究会得到算法性能范围的洞见，但一旦算法应用到了实际数据中，其绝对性能的意义仍然很小。特征点的识别是另一种验证的常见方法，在我们的验证试验中也进行了考虑。这种技术的难处是，在解剖结构中的特征点很少，几乎没有。像隆凸这样的结构，在气管这样的结构中展现出分裂，在图像中可以看的很清楚，在一个3D坐标系中并没有结束，而是在一个曲面中分布的。在选择对应点时，总有一定程度的误差。

We would like to avoid the pit falls in landmark identification, but the capacity of the human eye to rapidly and accurately (albeit only qualitatively) determine the quality of a registration should not be underscored. While validation methods involving human assessment will be prone to error, bias, and inconsistencies, if two images are presented in a conducive manner, experienced radiologists can rapidly assess the gestalt quality of the registration in a short amount of time. Furthermore, they have a priori knowledge of where in the anatomy the registration should be precise and where the required accuracy can be relaxed.

我们希望避免在特征点识别中的陷阱，但人眼能够迅速准确的确定一个配准的质量的能力，不应当被低估。人类进行评估的验证方法，通常容易有误差、偏移和不一致性，如果两幅图像是以容易的方式给出，有经验的放射学家可以在很短的时间内迅速的评估配准的gestalt质量。而且，他们有一些先验知识，在解剖结构的什么地方，配准必须精确，在哪些地方，精度可以放宽一些。

Our aim with the experiment is to assess the algorithm only in anatomically relevant areas. We are unsure of where the algorithm will be sucessful and where it will fail, and certain regions are of more clinical importance to researchers interested in utilizing the registration algorithm. We will take 7 images from each patient set: 5 axial slices (4 regularly spaced through the lungs, and 1 in the upper abdomen), 1 coronal slice at the carina, and 1 sagittal midline slice. Additionaly, we will include 10% duplicate slices to measure intra-observer consistency.

我们试验的目标是评估算法在解剖相关区域的性能。我们不确定算法在哪里会成功，哪里会失败，一些区域对研究者更具有临床重要性。我们在每位患者图像集合中采用了7幅图像：5个横断面slices（4个沿着肺部均匀间隔，1个上腹部），1个隆突部位的冠状slice，1个矢状面中线slice。另外，我们还多取了10%的重复slice以衡量观察者间的一致性。

We created a user interface for validation that allows rapid visual assessment of registered images. To provide a measure of consistency between multiple observers, we allow only 2D image navigation, and present only specific images for assessment. The interface is based on the split window, a display technique that fixes one image over another and allows the user to vary the lines of transition from the top image to the image below. The images are stationary, but with the mouse button held the vertical and horizontal lines of transition move with the mouse cursor. To quantify the perceived error, a ruler is placed over the image and fixed to the mouse cursor. A set of error bars physically sized according to the ruler gradations is used to assess the registration accuracy for the given image pair. The observer makes two error assessments for each image pair: overall and maximum. Using this method, the two strongest visual tests will be the presence/absence of similar anatomic structures and the magnitude of discontinuities in tissue boundaries. In the case absence of anatomy out of the plane of the image, the observer will have to make the best assessment possible based on his/her knowledge of the anatomy.

我们创建了一个用户界面进行验证，可以对配准的图像进行快速的视觉评估。为度量多个观察者之间的一致性，我们只允许2D图像导航，只对特定的图像进行评估。这个界面是基于分割窗口的，这种显示技术将一幅图像固定在另一幅上，用户可以变化从上面的图像到下面的图像的迁移线。图像是静态的，但随着鼠标按下，水平和垂直的迁移线随着鼠标光标移动。为使得感受到的误差量化，图像上放置了一个尺子，固定到鼠标光标上。对给定的图像对，我们采用带有物理尺寸的误差条，根据尺子的刻度来评估配准准确率。观察者对每个图像对进行两个误差评估：总体误差和最大误差。使用这种方法，两个最强的视觉测试是，有/没有类似的解剖结构和组织边缘的不连续性幅度。在图像平面之外没有解剖结构的情况下，观察者会根据其对解剖结构的了解，进行最佳的评估。

Since the intensity values for CT and TR scans are not in the 8-bit display range, window and level functionality is neccesary for rendering specific anatomic structure. Window and level settings for CT images are standardized, but there is no comparable parameterization for attenuation values. Instead, we apply an empirically determined intensity transformation to the PET attenuation values and display the TR image in converted Hounsfield units. Slider bars are available for adjusting the window and level settings for both images and there are presets for common values to render anatomy such as the mediastinum and the lungs.

由于CT和TR scans的灰度值范围超过了8-bit显示范围，要显示特定的解剖结构，窗宽和窗位的功能是必须的。对CT图像的窗宽和窗位是标准的，但对于衰减值，没有可比较的参数。我们对PET衰减值，通过经验确定了其灰度变换，以转换的Housfield值显示了TR图像。滑动条可以用于调整两幅图像的窗宽和窗位设置，为显示像纵隔和肺这样的解剖结构，有预设的常见值。

## 3. Data

The images acquired for the validation studies presented here are part of a lung study at the University of Washington Medical Center. They are scans from patients who have been screened for lung nodules, lymphoma, etc. All patients had PET scans on the UWMC's GE Advance Scanner; a 15 minute TR scan with an energy beam of 51 keV was acquired immediately before the emission data was recorded. A diagnostic CT scan was also performed at the UWMC, however using a variety of GE scaners. The patients received the scans within at most 2 months from one another, and did not have surgical resection or a high degree cancer development between acquisitions. The TR scans occupy 3 fields-of-view(FOV) and are exported from the PET station as a single 3D floating-point binary dataset of size 128x128x103. The TR voxels have a physical size of 4.29x4.29x4.25mm; therefore each TR scan axially images 437.35mm. The CT scans are axial images with varying slice thicknesses that in practice vary from 2-7mm. Once the patient scan is acquired, the in-plane resolution is set to magnify the patient anatomy as much as possible, within the range of 0.7-1.1mm. The portion of anatomy imaged in the CT scans varies from patient to patient, and in general the CT FOV is smaller than that of the TR scan(a typical CT FOV covers 350mm of axial anatomy).

这里进行验证研究获取的图像，是在华盛顿大学医学中心的肺研究的一部分。这是进行肺结节、淋巴瘤等筛查的病人。所有病人都在UWMC的GE Advance Scanner上进行了PET扫描；在发射数据记录之前，立刻获取了51keV能量束的15分钟TR扫描。在UWMC还扫描了诊断CT，但是使用了几台GE scanners。患者进行的扫描每人之间最多间隔2个月，在这之间没有进行过手术，或高强度的癌症治疗。TR scans占据了3个FOV，是从PET站中导出的，作为一个单个的3D浮点二值数据集，大小为128x128x103。TR体素的物理大小为4.29x4.29x4.25mm；因此每个TR scan在轴向上成像437.35mm。CT scans是轴向的图像，有着不同的slice厚度，实际中在2-7mm之间变化。一旦得到了患者的扫描，平面内的分辨率设置为，最大程度上放大患者的解剖结构，在0.7mm-1.1mm的范围。在CT中成像的解剖结构的部分，随着病人不同，总体上CT FOV比TR scan的要小（典型的CT FOV覆盖350mm的轴向解剖结构）。

Preprocessing steps are performed before registering the images. The images are filtered with a Gaussian blur, the TR image is resliced to have isotropic voxels, and the spatial resolution of the CT is reduced to match that of the TR. As explained in the introduction the arms are present in the PET images and are extraneous features for which there is no correspondence in the CT. Manually-placed polygons in several axial slices are extended throught the volume. The voxels within this volume (V in equation 16) represent an optimal set of image samples with which to compute mutual information.

在配准图像之前要进行预处理步骤。图像是用高斯模糊来滤波的，TR图像重新进行分层，以得到各向同性的体素，CT的空间分辨率要进行降低以与TR图像进行匹配。如同在介绍中进行解释的，PET图像中存在手臂，这是无关的特征，在CT中没有对应。在几个轴向slice中，手工放置了多边形，在整个体中进行了拓展。在这个体中的体素（式16中的V）代表图像样本的最佳集合，用于计算互信息。

## 4. Results

28 patient sets were registered for the validation experiment, with 1 failing due to a limited CT FOV; all data presented is for the 27 patients succesfully registered, which was validated by 2 radiologists. Example images from a registered dataset are shown in figure 2 with a locked cursor, while figure 4 shows sample images from the validation user interface. The errors in ranges are recorded by the user interface as error points, with 0 points representing the lowest error and 4 points being the largest possible error selection; i.e. each point corresponds to a range of error. The means, medians, and standard deviations of the overall and maximum assessed errors for both observers are given in table I. The errors statistics fall into the first and second smallest error ranges, respectively. We can also group the results by anatomic region and compare assessed errors in all patients for a given region and for all the anatomy. The assessed error by anatomy is shown in figure 5. The assessed errors in axial slices in the upper lung regions are small, with the poorest performance in the abdomen. The transaxial slices (sagittal and coronal) are also assessed poorly compared to the lung regions. There is some tendency for observer 2 to grade more harshly than observer 1, but their choices are fairly well corelated (ρ=0.68). Also, the intraobserver consistency is well within acceptable ranges.

在验证试验中，对28个病人进行了配准，有1个失败了，由于CT FOV是有限的；27个病人的所有数据都得到了成功的配准，这由2名放射科医生进行了验证。图2展示了一个配准的数据集的例子图像，有一个锁住的光标，而图4所示的是验证用户界面的样本图像。范围上的误差由用户界面记录为误差点，0个点表示最低的误差，4个点表示最大的误差；即，每个点对应着一个误差范围。两个观察者的总体误差和最大误差的均值、中值和标准差，如表I所示。误差统计分别落入最小和次小误差范围。我们还可以通过解剖区域来对结果进行分组，比较所有患者的在给定区域内对所有解剖结构的评估的误差。通过解剖结构评估的误差如图5所示。评估的误差在轴向的slice在上肺部区域是最小的，在腹部区域是最差的。矢状和冠状slices也进行了评估，与肺部区域比较起来，也很差。观察者2与观察者1比起来，评分会更加严格，但其选择的相关性是非常好的(ρ=0.68)。同时，观察者之内的一致性也是非常好的，在可接受的范围内。

The fully automated registration process takes an average of ～100 minutes on a moderate work station to determine the optimal set of transformation parameters that maps coordinates in the test image to coordinates in the reference image. The process is divided into two registrations: a rigid body(10 minutes, 6 parameters) followed by a deformation(90 minutes, 20 parameters) recovery. Each registration is performed within a multiresolution framework that proceeds from "coarse" to "fine" resolutions in 4 steps. Computation time is most strongly effected by the number of transformation parameters and the number of voxels used to estimate mutual information. The multiresolution parameters used for the registrations presented here are summarized in table 2.

全自动的配准过程，在一个中等的工作站上，平均消耗大约100min的时间，可以确定最佳的变换参数集合，将测试图像中的坐标映射到参考图像中的坐标。这个过程分成两个配准：一个刚体（10分钟，6个参数），然后是一个变形恢复过程（90分钟，20个参数）。每个配准都在一个多分辨率框架中进行，在4个步骤中从粗糙的分辨率到精细的分辨率进行。计算时间受到变形参数数量和用于估计互信息的体素数量的很强的影响。这里给出的用于配准的多分辨率参数，在表2中进行了总结。

## 5. Conclusions

The results of the validation study indicate that the algorithm is capable of accurate registrations in the thorax, and the radiologists who validated the results feel the errors are generally within clinically acceptable ranges. We have identified regions in the anatomy for which the algorithm succeeds in varying degrees. According to visual assessments of 27 patient datasets by two observers, the overall error in the considered anatomy is 0.54 error points, which is in the 0-6 mm error range. Recall that the PET voxel size is ~4.3mm on a side. Divided into 5 regimes, the possible range of error is 0-29mm. The mid to upper lung regions are registered the most accurately with a mean overall assessed error of 0.24 error points, which is also in the 0-6mm error range. The assessed performance in the abdomen is the worst in general with a mean overall assessed error of 1.37 error points, which is in the 6-11 mm error range.

验证研究的结果表明，算法可以在胸部进行准确的配准，放射科医生验证了这个结果，认为误差一般都是在临床可接受的范围内。我们在解剖结构中识别出了区域，算法在各种程度中都得到了成功的结果。根据两个观察者用视觉评估的27个患者的数据集，对考虑的解剖结构，总体的误差是在0.54个误差点，也就是0-6mm的误差范围。回顾一下PET的体素大小为大约4.3mm。分成5个区域，误差的可能范围为0-29mm。肺部中间和上部区域，其配准最为精确，平均总体误差为0.24个误差点，也是0-6mm的误差范围。在腹部的评估性能，总体上是最差的，平均总体误差为1.37个误差点，也就是6-11mm的误差范围。

The algorithm produces registrations with relatively large assessed error in the coronal and sagittal planes. Anatomically, however, the chest varies much more transaxially than in any given axial slice and we may have a situation in which increasing the deformation resolution in this dimension may yield better results. Additionally, larger lung and soft tissue regions are perhaps driving the deformation, and the smaller airway anatomy is moved along with them. This highlights the principal limitation of the algorithm: the inability to define deformations on a non-uniform grid. We do however have ideas on how to approach this extension.

算法在冠状面和矢状面得到的配准结果误差相对较大。但是，解剖上来说，胸部在transaxial方向上的变化要更大，因此在这个维度上增加形变的分辨率会得到更好的结果。另外，更大的肺部和软组织区域可能会导致变形，更小的气道解剖结构随着这些形变而运动。这强调了这个算法的主要局限：不能在非均匀网格上定义形变。但我们还是有想法，对这个进行拓展的。

From the validation study, we notice a poor assessment in the abdomen. At 51 keV, the TR scan shows little intensity differences between fat, muscle, and soft tissue, resulting in a uniform image in the abdomen(except for gas bubbles). At 140 keV, the CT scan of the abdomen, on the other hand, shows clear delineation of the anatomy. Since there are few shared structures in the TR and CT scan of the abdomenon which the algorithm can anchor, a poor registration results. In this case, the PET emission or SUV scan may be a better candidate for alignment with CT. An alternative approach includes attenuation and emission values simultaneously, which may possible improve the overall accuracy in the torso.

从验证研究中，我们注意到在腹部的评估结果很差。在51keV上，TR scan在脂肪、肌肉和软组织上表现出很小的灰度差异，在腹部得到一幅灰度均一的图像（除了气泡）。另一方面，在140keV，腹部的CT scan，可以展现出解剖结构的明显边缘。因为在腹部，TR和CT scan的共有结构非常少，算法无法利用，表现出很差的配准结果。在这种情况下，PET或SUV scan可能是与CT进行对齐的更好的候选。一种替代的方法包括，同时包含衰减值和emission值，这可能改进在躯干部分的总体准确率。

In addition to the more general question of how fine a deformation grid do we need, we often face the question of how fine a deformation grid can we use. If we notice troublespots in the registration, and we do not have the capability to use an irregular grid, why not increase the resolution of the deformation until enough control points are clustered in the difficult region? We believe that the robustness of the deformation is due in part to the coarseness of the grid, in that mutual information is adjusted such that large features are moved into alignment. When the grid density is increased, smaller and more local anatomic variations will be driving the deformation, and we might expect some rather violent local deformations. In this sense the coarseness of the grid somewhat regularizes the deformation as a whole. The registration accuracy in certain anatomic regions such as along the trachea can likely benefit from some form of clustering of deformation points.

我们需要一个变形的多精细的网格，这是一个更一般性的问题，除此以外，我们通常面对的问题是，我们能使用多精细的形变网格。如果我们注意到配准中的问题，我们没有使用不规则网格的能力，为什么不增加形变的分辨率，直到足够多的控制点积聚到困难区域中？我们相信，形变的稳健性是部分因为网格的粗糙性，互信息经过调整，大的特征经过移动得到配准。当网格密度增加时，更小的更局部的解剖变化会推动形变，我们会得到一些非常剧烈的局部形变。在这个意义上，网格的粗糙性在一定程度上对形变有整体上的正则化的作用。在特定解剖区域中的配准准确率，比如沿着气管的区域，很可能会从一些形式的形变点的聚积中受益。

We have validated the algorithm only for TR to CT matching problem. It is natural to question if it will work well for other inter- or intra-modality image pairs. PET images are among the worst (except perhaps SPECT) of the 3D imaging modalities in terms of tissue delineation and signal to noise ratio. We believe that given the success of the TR-CT registration problem, the other modality combinations will be no more difficult or significantly easier.

我们已经验证了算法在TR与CT的配准问题中的有效性。很自然要问，对其他同模态或不同模态的图像对是否会好用。PET可能是3D成像模态中质量最差的模态，这是以组织勾画和信噪比来说的。我们相信，有了TR-CT配准问题的成功，其他模态组合不会更加困难，也不会容易太多。