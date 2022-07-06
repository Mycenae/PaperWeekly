# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

Ben Mildenhall et. al.

## 0. Abstract

We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully connected (nonconvolutional) deep network, whose input is a single continuous 5D coordinate (spatial location (x, y, z) and viewing direction (θ, φ)) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis.

我们提出一种方法，在合成复杂场景的新视野下取得了目前最好的结果，这是使用输入视野的稀疏集合，通过优化潜在的连续体场景函数来实现的。我们的算法使用一个全连接（非卷积）深层网络表示一个场景，其输入是单个连续的5D坐标（空间位置(x, y, z)，和视野方向(θ, φ)），其输出是体密度和依赖于视野的在空间位置处的发射的辐射。我们沿着相机射线通过查询5D坐标，使用经典的体渲染技术，将输出的颜色和密度投影到一幅图像中，来合成视野。因为体渲染在自然上就是可微分的，要优化我们的表示，需要的唯一输入，就是已知相机姿态的图像集合。我们描述怎样有效的优化神经辐射场，来渲染具有照片真实感的场景新视野，带有复杂的几何和外观，展示了优于之前的神经渲染和视野合成的结果。

## 1. Introduction

In this work, we address the long-standing problem of view synthesis in a new way. View synthesis is the problem of rendering new views of a scene from a given set of input images and their respective camera poses. Producing photorealistic outputs from new viewpoints requires correctly handling complex geometry and material reflectance properties. Many different scene representations and rendering methods have been proposed to attack this problem; however, so far none have been able to achieve photorealistic quality over a large camera baseline. We propose a new scene representation that can be optimized directly to reproduce a large number of high-resolution input views and is still extremely memory-efficient (see Figure 1).

本文中，我们以一种新的方式处理一直存在的视野合成问题。视野合成是在给定一定的输入图像集合，及其相应的相机姿态，渲染场景的新视野的问题。从新的视野生成具有照片真实感的输出，需要正确的处理复杂的几何和材料反射属性。曾经提出了很多不同的场景表示和渲染方法来解决这个问题；但是，目前为止，在大相机基线的情况下，没有能产生具有照片真实感质量的。我们提出了一种新的场景表示方法，可以进行直接优化，以重现大量高分辨率输入视野，而且需要的内存也很少（见图1）。

We represent a static scene as a continuous 5D function that outputs the radiance emitted in each direction (θ, φ) at each point (x, y, z) in space, and a density at each point which acts like a differential opacity controlling how much radiance is accumulated by a ray passing through (x, y, z). Our method optimizes a deep fully connected neural network without any convolutional layers (often referred to as a multilayer perceptron or MLP) to represent this function by regressing from a single 5D coordinate (x, y, z, θ, φ) to a single volume density and view-dependent RGB color. To render this neural radiance field (NeRF) from a particular viewpoint, we: 1) march camera rays through the scene to generate a sampled set of 3D points, 2) use those points and their corresponding 2D viewing directions as input to the neural network to produce an output set of colors and densities, and 3) use classical volume rendering techniques to accumulate those colors and densities into a 2D image. Because this process is naturally differentiable, we can use gradient descent to optimize this model by minimizing the error between each observed image and the corresponding views rendered from our representation. Minimizing this error across multiple views encourages the network to predict a coherent model of the scene by assigning high-volume densities and accurate colors to the locations that contain the true underlying scene content. Figure 2 visualizes this overall pipeline.

我们将一个静态场景表示为一个连续5D函数，输出空间中每个方向(θ, φ)在每个点(x, y, z)上的发射的辐射，在每个点上的密度，就像一个微分不透明度，控制光线透过(x,y,z)时积累了多少辐射。我们的方法优化一个深层全连接神经网络，没有任何卷积层（通常称为一个多层感知机，MLP）来表示这个函数，从单个5D坐标(x, y, z, θ, φ)回归到一个体密度和依赖于视野的RGB颜色。为从特定视角渲染这个神经辐射场(NeRF)，我们：1)让相机射线发射通过场景，以产生采样的3D点集，2)使用这些点及其对应的2D视野方向作为神经网络的输入，来产生色彩和密度的输出集合，3)使用经典的体渲染技术，来累积这些色彩和密度，成为一幅2D图像。因为这个过程是自然可微分的，我们可以使用梯度下降来优化这个模型，将每幅观察到的图像，和从我们的表示中渲染得到的对应视野，之间的误差最小化。在多个视野中最小化这个误差，会使网络通过给包含真实的潜在场景内容的点，指定高体的密度和精确的色彩，可以预测场景的连续模型。图2对这个总体上的流程进行了可视化。

We find that the basic implementation of optimizing a neural radiance field representation for a complex scene does not converge to a sufficiently high-resolution representation. We address this issue by transforming input 5D coordinates with a positional encoding that enables the MLP to represent higher frequency functions.

我们发现，对一个复杂场景优化一个神经辐射场表示，其基本实现不会收敛到一个足够高分辨率的表示。我们通过将输入的5D坐标，用一个位置编码，转换成使MLP能表示更高频的函数的形式，来处理这个问题。

Our approach can represent complex real-world geometry and appearance and is well suited for gradient-based optimization using projected images. By storing a scene in the parameters of a neural network, our method overcomes the prohibitive storage costs of discretized voxel grids when modeling complex scenes at high resolutions. We demonstrate that our resulting neural radiance field method quantitatively and qualitatively outperforms state-of-the-art view synthesis methods, such as works that fit neural 3D representations to scenes as well as works that train deep convolutional networks (CNNs) to predict sampled volumetric representations. This paper presents the first continuous neural scene representation that is able to render high-resolution photorealistic novel views of real objects and scenes from RGB images captured in natural settings.

我们的方法可以表示复杂的真实世界几何和外观，非常适用于基于梯度的优化，可以使用投影的图像。将一个场景存储在神经网络的参数中，我们的方法克服了在对复杂场景用高分辨率进行建模的时候，离散的体素网格的存储代价过高的问题。我们展示了，得到的神经辐射场方法，在定性的指标，和定性的感受上，都超过了目前最好的视野合成方法，比如将拟合场景为神经3D表示的方法，以及训练深度CNN来预测采样的体表示的方法。本文给出了第一个连续的神经场景表示，从自然状态捕获的RGB图像中，可以渲染出高分辨率的具有照片真实感的真实物体的新视野。

## 2. Related Work

A promising recent direction in computer vision is encoding objects and scenes in the weights of an MLP that directly maps from a 3D spatial location to an implicit representation of the shape, such as the signed distance at that location. However, these methods have so far been unable to reproduce realistic scenes with complex geometry with the same fidelity as techniques that represent scenes using discrete representations such as triangle meshes or voxel grids. In this section, we review these two lines of work and contrast them with our approach, which enhances the capabilities of neural scene representations to produce state-of-the-art results for rendering complex realistic scenes.

最近，计算机视觉中一个有希望的方向是将目标和场景编码成一个MLP，直接从一个3D空间位置映射到形状的一个隐式表示，比如这个位置的带符号距离。但是，这些方法到目前为止，还没有能将真实的带有复杂几何的场景，复现到与将场景表示成离散表示的方法相同的真实度，比如三角网格，或体素网格。在本节中，我们回顾了这两条线的工作，将其与我们的方法进行比较，这增强了神经场景表示的能力，在渲染复杂的真实场景时产生了目前最好的结果。

### 2.1. Neural 3D shape representations

Recent work has investigated the implicit representation of continuous 3D shapes as level sets by optimizing deep networks that map xyz coordinates to signed distance functions or occupancy fields. However, these models are limited by their requirement of access to ground truth 3D geometry, typically obtained from synthetic 3D shape datasets such as ShapeNet. Subsequent work has relaxed this requirement of ground truth 3D shapes by formulating differentiable rendering functions that allow neural implicit shape representations to be optimized using only 2D images. Niemeyer et al. represent surfaces as 3D occupancy fields and use a numerical method to find the surface intersection for each ray, then calculate an exact derivative using implicit differentiation. Each ray intersection location is provided as the input to a neural 3D texture field that predicts a diffuse color for that point. Sitzmann et al. use a less direct neural 3D representation that simply outputs a feature vector and RGB color at each continuous 3D coordinate, and propose a differentiable rendering function consisting of a recurrent neural network that marches along each ray to decide where the surface is located.

最近的工作研究了连续3D形状隐式表示为水平集的问题，优化将xyz坐标映射到带符号距离函数或占用场的深度网络。但是，这些模型需要知道真值3D几何，这限制了其发展，这些3D几何一般是通过合成3D形状数据集得到的，比如ShapeNet。后续的工作放松了对真值3D形状的要求，使用可微分的渲染函数，使神经隐式形状表示可以只使用2D图像被优化。Niemeyer等将表面表示成3D占用场，使用一种数值方法找到对每条光线的相交表面，然后计算使用隐式微分来计算严格的微分。每条光线相交的位置作为神经3D纹理场的输入，对该点预测一个扩散颜色。Sitzmann等使用了一个没那么直接的神经3D表示，在每个连续的3D坐标上简单的输出一个特征向量和RGB色彩，提出了一个可微分的渲染函数，由一个循环神经网络组成，沿着每条光线行进，决定平面的位置。

Though these techniques can potentially represent complicated and high-resolution geometry, they have so far been limited to simple shapes with low geometric complexity, resulting in oversmoothed renderings. We show that an alternate strategy of optimizing networks to encode 5D radiance fields (3D volumes with 2D viewdependent appearance) can represent higher resolution geometry and appearance to render photorealistic novel views of complex scenes.

虽然这些技术潜在的可能表示复杂、高分辨的几何，但他们目前仍局限在简单的形状，几何复杂度低，得到了过度平滑的渲染。我们展示了另一个策略，对网络进行优化，以编码5D辐射场（3D体，和2D依赖于视角的外观），这个策略可以表示更高分辨率的几何和外观，可以渲染出像真实照片的新视角的复杂场景。

### 2.2. View synthesis and image-based rendering

The computer vision and graphics communities have made significant progress on the task of novel view synthesis by predicting traditional geometry and appearance representations from observed images. One popular class of approaches uses mesh-based scene representations. Differentiable rasterizers or pathtracers can directly optimize mesh representations to reproduce a set of input images using gradient descent. However, gradient-based mesh optimization based on image reprojection is often difficult, likely because of local minima or poor conditioning of the loss landscape. Furthermore, this strategy requires a template mesh with fixed topology to be provided as an initialization before optimization, which is typically unavailable for unconstrained real-world scenes.

计算机视觉和图形学团体在新视图合成上取得了显著的进展，从观察到的图像中预测传统的几何和外观表示。一类流行的方法使用基于mesh的场景表示。可微分的rasterizer或pathtracer可以直接优化mesh表示，以使用梯度下降来复现输入图像集。但是，基于图像重投影的基于梯度的mesh优化通常是很难的，很可能是因为局部最小值，或损失函数的条件不太好。而且，这种策略需要提供一个有固定拓扑的模板mesh作为初始化，然后才能进行优化，对于没有约束的真实世界场景，这一般是不存在的。

Another class of methods use volumetric representations to address the task of high-quality photorealistic view synthesis from a set of input RGB images. Volumetric approaches are able to realistically represent complex shapes and materials, are well suited for gradient-based optimization, and tend to produce less visually distracting artifacts than mesh-based methods. Early volumetric approaches used observed images to directly color voxel grids. More recently, several methods have used large datasets of multiple scenes to train deep networks that predict a sampled volumetric representation from a set of input images, and then use either alpha compositing or learned compositing along rays to render novel views at test time. Other works have optimized a combination of CNNs and sampled voxel grids for each specific scene, such that the CNN can compensate for discretization artifacts from low-resolution voxel grids or allow the predicted voxel grids to vary based on input time or animation controls. Although these volumetric techniques have achieved impressive results for novel view synthesis, their ability to scale to higher resolution imagery is fundamentally limited by poor time and space complexity due to their discrete sampling—rendering higher resolution images requires a finer sampling of 3D space. We circumvent this problem by instead encoding a continuous volume within the parameters of a deep fully connected neural network, which not only produces significantly higher quality renderings than prior volumetric approaches but also requires just a fraction of the storage cost of those sampled volumetric representations.

另一类方法使用体表示来处理从输入的RGB图像集中合成高质量具有真实图像感的视图的任务。体方法可以真实的表示复杂形状和材料，非常适用于基于梯度的优化，比基于mesh的方法，通常产生的视觉伪影更少。早期的体方法使用观察到的图像来直接对体网格上色。更最近的，几种方法使用多个场景的大型数据集来训练深度网络，从输入图像集中来预测采样的体表示，然后使用alpha compositing或learned compositing沿着光线来在测试时渲染新的视图。其他工作对每个特定的场景优化了CNNs和采样体网格的组合，这样CNN可以从低分辨率的体素网格补偿离散的伪影，或允许预测的体素网格基于输入时间或动画控制来变化。虽然这些体素技术在新视图合成上获得了令人印象深刻的结果，但放大到更高分辨率图像的能力是在基本上受限的，渲染更高分辨率的图像，需要3D空间更精细的采样。我们则避免了这个问题，在一个深度全连接神经网络的参数中编码了一个连续体，这不仅比之前的体方法产生了显著更高质量的渲染，同时只要求体采样表示的一小部分存储就可以。

## 3. Neural Radiance Field Scene Representation

We represent a continuous scene as a 5D vector-valued function whose input is a 3D location x = (x, y, z) and 2D viewing direction (θ, φ), and whose output is an emitted color c = (r, g, b) and volume density σ. In practice, we express direction as a 3D Cartesian unit vector d. We approximate this continuous 5D scene representation with an MLP network FΘ: (x, d) → (c, σ) and optimize its weights Θ to map from each input 5D coordinate to its corresponding volume density and directional emitted color.

我们将一个连续场景表示为5D向量值的函数，其输入是一个3D位置x = (x, y, z)和2D视角方向(θ, φ)，其输出是一个发射的色彩c=(r,g,b)和体密度σ。在实践中，我们将方向表示为一个3D笛卡尔单位向量d。我们用一个MLP网络FΘ: (x, d) → (c, σ)，来近似这个连续的5D场景表示，优化其权重Θ，来将每个输入5D坐标映射到其对应的体密度和方向发射色彩。

We encourage the representation to be multiview consistent by restricting the network to predict the volume density σ as a function of only the location x, while allowing the RGB color c to be predicted as a function of both location and viewing direction. To accomplish this, the MLP FΘ first processes the input 3D coordinate x with 8 fully connected layers (using ReLU activations and 256 channels per layer), and outputs σ and a 256-dimensional feature vector. This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color.

我们鼓励这个表示是多视角一致的，约束网络预测体密度σ只是位置x的函数，而RGB色彩c是位置和视角方向的函数。为此，MLP FΘ首先用8个全连接层（使用ReLU激活和每层256个通道）处理输入3D坐标x，输出σ和一个256维的特征向量。这个特征向量然后与相机光线的视角方向拼接到一起，送入另一个全连接层（使用ReLU激活和128通道），输出依赖于视角的RGB色彩。

See Figure 3 for an example of how our method uses the input viewing direction to represent non-Lambertian effects. As shown in Figure 4, a model trained without view dependence (only x as input) has difficulty representing specularities.

图3是我们的方法怎样使用输入视角方向来表示non-Lambertian效果的一个例子。如图4所示，在没有视角依赖下训练出来的模型（只有x作为输入），在表示镜面反射时会有困难。

## 4. Volume Rendering with Radiance Fields

Our 5D neural radiance field represents a scene as the volume density and directional emitted radiance at any point in space. We render the color of any ray passing through the scene using principles from classical volume rendering. The volume density σ(x) can be interpreted as the differential probability of a ray terminating at an infinitesimal particle at location x. The expected color C(r) of camera ray r(t) = o + td with near and far bounds tn and tf is:

我们的5D神经辐射场将一个场景表示为在空间中任意点处的体密度和方向发射辐射。我们使用经典体渲染的原则，来渲染任意光线穿过场景的色彩。体密度σ(x)可以解释为，一条光线在位置x处的无线小粒子处终止的微分概率。相机光线r(t)=o+td，有近和远的界限tn和tf，其期望的色彩C(r)为：

$$C(r) = \int_{tn}^{tf} T(t) σ(r(t)) c(r(t),d) dt$$(1)

$$where T(t) = exp(-\int_{tn}^t σ(r(s)ds))$$(2)

The function T(t) denotes the accumulated transmittance along the ray from tn to t, that is, the probability that the ray travels from tn to t without hitting any other particle. Rendering a view from our continuous neural radiance field requires estimating this integral C(r) for a camera ray traced through each pixel of the desired virtual camera.

函数T(t)表示沿着光线从tn到t处的累积传输，即，光线传输tn到t没有撞击到任意其他粒子的概率。从我们的连续神经辐射场渲染一个视图，需要对追踪穿过期望虚拟相机的每个像素的相机光线估计这个积分C(r)。

We numerically estimate this continuous integral using quadrature. Deterministic quadrature, which is typically used for rendering discretized voxel grids, would effectively limit our representation’s resolution because the MLP would only be queried at a fixed discrete set of locations. Instead, we use a stratified sampling approach where we partition [tn, tf] into N evenly spaced bins and then draw one sample uniformly at random from within each bin:

我们使用求积法从数值上估计这个连续积分。确定性的数值求积法，一般用于渲染离散的体网格，会有效的限制我们的表示分辨率，因为MLP只会在位置的固定离散集合上查询计算。我们则使用了一种分层采样方法，我们将[tn, tf]分割成N个均匀间隔的bins，在每个bin中随机均匀采样一个样本点：

$$ti ~ u[tn+\frac{i-1}{N} (tf-tn), tn+\frac{i}{N}(tf-tn)]$$(3)

Although we use a discrete set of samples to estimate the integral, stratified sampling enables us to represent a continuous scene representation because it results in the MLP being evaluated at continuous positions over the course of optimization. We use these samples to estimate C(r) with the quadrature rule discussed in the volume rendering review by Max:

虽然我们使用了样本的离散集合来估计积分，但分层采样使我们可以表示一个连续场景表示，因为这样的结果是，MLP在优化的过程中，在连续位置进行评估。我们使用这些样本来估计C(r)，使用的数值积分规则是Max回顾的体渲染中讨论的：

$$\hat C(r) = \sum_{i=1}^N T_i (1-exp(-σ_i δ_i))c_i$$(4)

$$T_i = exp(-\sum_{j=1}^{i-1} σ_i δ_i)$$(5)

where $δ_i = t_{i+1} − t_i$ is the distance between adjacent samples. 其中$δ_i = t_{i+1} − t_i$是相邻样本之间的距离。

This function for calculating $\hat C(r)$ from the set of (ci, σi) values is trivially differentiable and reduces to traditional alpha compositing with alpha values σi = 1 − exp(−σiδi).

从(ci, σi)值的集合中计算$\hat C(r)$的这个函数是可微的，使传统的alpha compositing为σi = 1 − exp(−σiδi)。

## 5. Optimizing a Neural Radiance Field

In the previous section, we have described the core components necessary for modeling a scene as a neural radiance field and rendering novel views from this representation. However, we observe that these components are not sufficient for achieving state-of-the-art quality. We introduce two improvements to enable representing high-resolution complex scenes. The first is a positional encoding of the input coordinates that assists the MLP in representing highfrequency functions. The second is a hierarchical sampling procedure that we do not describe here; for details, see the original paper.

在上一节中，我们描述了将场景建模为神经辐射场，和从这种表示中渲染出新视图的核心组成部分。但是，我们观察到，这些组成部分不足以获得目前最好的质量。我们提出两种改进，以表示高分辨率复杂场景。第一个是输入坐标的位置编码，帮助MLP表示高频函数。第二个是一种层次化的采样过程，在这里我们没有描述；细节请见原始论文。

### 5.1. Positional encoding

Despite the fact that neural networks are universal function approximators, we found that having the network FΘ directly operate on xyzθφ input coordinates results in renderings that perform poorly at representing high-frequency variation in color and geometry. This is consistent with recent work by Rahaman et al., which shows that deep networks are biased toward learning lower frequency functions. They additionally show that mapping the inputs to a higher dimensional space using high-frequency functions before passing them to the network enables better fitting of data that contains high-frequency variation.

尽管神经网络可以近似任意函数，我们发现，如果让网络FΘ直接对xyzθφ输入坐标进行运算，得到的渲染在表示色彩和几何的高频变化上会比较差。这与Rahaman等最近的工作是一致的，证明了深度网络比较倾向于学习较低频的函数。他们还展示了，将输入映射到更高维空间中，在送入网络前使用高频函数，会使包含高频变化的数据得到更好的拟合。

We leverage these findings in the context of neural scene representations, and show that reformulating FΘ as a composition of two functions FΘ = F'Θ ◦ γ, one learned and one not, significantly improves performance (see Figure 4). Here γ is a mapping from R into a higher dimensional space R^2L, and F'Θ is still simply a regular MLP. Formally, the encoding function we use is:

我们在神经场景表示中利用这些发现，展示了将FΘ重新表示为两个函数的组合，FΘ = F'Θ ◦ γ，一个是学习得到的，一个不是，这样显著改进了性能（见图4）。这里γ是从R到更高空间R^2L的映射，F'Θ仍然是一个常规的MLP。形式上，我们使用的编码函数是：

$$γ(p) = (sin(2^0 πp), cos(2^0 πp), ..., sin(2^{L-1} πp), cos(2^{L-1} πp))$$(6)

This function γ(·) is applied separately to each of the three coordinate values in x (which are normalized to lie in [−1, 1]) and to the three components of the Cartesian viewing direction unit vector d (which by construction lie in [−1, 1]). In our experiments, we set L = 10 for γ(X) and L = 4 for γ(d).

这个函数γ(·)分别应用到x中三个坐标值的每一个（归一化为[-1, 1]），还应用到笛卡尔视角方向单元向量d的3个组成部分中（构造时就在[-1, 1]的范围内）。在我们的试验中，我们对γ(X)设置L=10，对γ(d)设置L=4。

This mapping is studied in more depth in subsequent work [22] which shows how positional encoding enables a network to more rapidly represent higher frequency signals. 这个映射在后续工作中有深入研究[22]，展示了位置编码怎样使网络能更快的表示更高频的信号。

### 5.2. Implementation details

We optimize a separate neural continuous volume representation network for each scene. This requires only a dataset of captured RGB images of the scene, the corresponding camera poses and intrinsic parameters, and scene bounds (we use ground truth camera poses, intrinsics, and bounds for synthetic data, and use the COLMAP structure-from-motion package [18] to estimate these parameters for real data). At each optimization iteration, we randomly sample a batch of camera rays from the set of all pixels in the dataset. We query the network at N random points along each ray and then use the volume rendering procedure described in Section 4 to render the color of each ray using these samples. Our loss is simply the total squared error between the rendered and true pixel colors:

我们对每个场景优化一个单独的神经连续体表示网络。这只需要一个场景中捕获的RGB图像数据集，对应的相机姿态和内部参数，和场景界限（我们使用真值相机姿态，内蕴，和合成数据的界限，使用COLMAP sfm包来对真实数据估计这些参数）。在每个优化迭代中，我们从数据集中的所有像素集中，随机采样一批相机射线。我们在N个随机点上沿着每条射线查询网络，然后使用第4部分描述的体渲染过程，使用这些样本来渲染每条射线的颜色。我们的损失就是，渲染的和真实的像素色彩的总计均方误差：

$$L = \sum_{r∈R} ||\hat C(r) - C(r)||_2^2$$(7)

where R is the set of rays in each batch, and C(r), $\hat C(r)$ are the ground truth and predicted RGB colors for ray r. 这里R是在每个批次中的射线集合，C(r), $\hat C(r)$是射线r的真值和预测的RGB色彩。

In our experiments, we use a batch size of 4096 rays, each sampled at N = 192 coordinates. (These are divided between two hierarchical “coarse” and “fine” networks; for details see the original paper.[13]) We use the Adam optimizer [6] with a learning rate that begins at 5 × 10^{−4} and decays exponentially to 5 × 10^{−5}. The optimization for a single scene typically takes about 1–2 days to converge on a single GPU.

在我们的试验中，我们使用一批4096条射线，每个都在N=192个坐标点上进行采样。（这些分成了层次化的粗糙和精细网络；细节请查看原始论文）我们使用Adam优化器，学习速率在开始的时候是5 × 10^{−4}，然后衰减到5 × 10^{−5}。在单个CPU的单个场景的优化，一般需要1-2天收敛。

## 6. Results

We quantitatively (Table 1) and qualitatively (see Figures 5 and 6) show that our method outperforms prior work. We urge the reader to view our accompanying video to better appreciate our method’s significant improvement over baseline methods when rendering smooth paths of novel views. Videos, code, and datasets can be found at https://www.matthew.

我们定性的，定量的展示了，我们的方法超过了之前的工作。读者可以查看伴随的视频来更好的欣赏我们方法比基准方法显著的改进，渲染了新视图的平滑路径。视频，代码和数据集可以在https://www.matthew中找到。

### 6.1. Datasets

**Synthetic renderings of objects**. We first show experimental results on two datasets of synthetic renderings of objects (Table 1, “Diffuse Synthetic 360°” and “Realistic Synthetic 360°”). The DeepVoxels [20] dataset contains four Lambertian objects with simple geometry. Each object is rendered at 512 × 512 pixels from viewpoints sampled on the upper hemisphere (479 as input and 1000 for testing). We additionally generate our own dataset containing pathtraced images of eight objects that exhibit complicated geometry and realistic non-Lambertian materials. Six are rendered from viewpoints sampled on the upper hemisphere, and two are rendered from viewpoints sampled on a full sphere. We render 100 views of each scene as input and 200 for testing, all at 800 × 800 pixels.

目标的合成渲染。我们首先展示了在两个目标合成渲染的数据集上的试验结果（表1，Diffuse Synthetic 360度，和Realistic Synthetic 360度）。DeepVoxels数据集包含4个Lambertian目标，几何很简单。每个目标渲染了512x512像素，视角是从上半球上采样得到的（479个用作输入，1000个用作测试）。我们还生成我们自己的数据集，包含了跟踪路径的8个目标的图像，有复杂的几何，和真实的non-Lambertian材质。6个是从上半球采样的视角渲染的，2个是在全球采样的视角渲染的。我们对每个场景渲染了100个视角作为输入，200个作为测试，像素量为800x800。

**Real images of complex scenes**. We show results on complex real-world scenes captured with roughly forward-facing images (Table 1, “Real ForwardFacing”). This dataset consists of eight scenes captured with a handheld cellphone (five taken from the local light field fusion (LLFF) paper and three that we capture), captured with 20 to 62 images, and hold out 1/8 of these for the test set. All images are 1008 × 756 pixels.

复杂场景的真实图像。我们在复杂的真实世界场景中展示结果，包含了粗糙的向前的图像。这个数据集包含8个场景，用手持的手机进行拍摄（5个场景是LLFF论文中，3个是我们自己拍摄的），拍摄了20到62幅图像，保留了1/8用于测试。所有的图像都是1008x756像素。

### 6.2. Comparisons

To evaluate our model we compare against current top-performing techniques for view synthesis, detailed here. All methods use the same set of input views to train a separate network for each scene except LLFF, [12] which trains a single 3D CNN on a large dataset, then uses the same trained network to process input images of new scenes at test time.

为评估我们的模型，我们与目前表现最好的视角合成技术进行比较。所有方法都使用相同的输入试图集合，对每个场景都训练一个单独的网络，除了LLFF，LLFF在一个大型数据集上训练一个3D CNN，然后使用相同的训练好的网络，来在测试时处理新场景的输入图像。

Neural Volumes (NV) [8] synthesizes novel views of objects that lie entirely within a bounded volume in front of a distinct background (which must be separately captured without the object of interest). It optimizes a deep 3D CNN to predict a discretized RGBα voxel grid with 128 [3] samples as well as a 3D warp grid with 32 [3] samples. The algorithm renders novel views by marching camera rays through the warped voxel grid.

NV合成目标的新试图，目标需要完全在明显的背景前的一个有限的体积内，而背景需要在没有目标的时候单独拍摄。其优化一个深度3D CNN，来预测一个离散的RGBα体素网格，有128个样本，以及有32个样本的3D形变网格。算法通过让相机射线行进通过形变的体素网格，来渲染新视图。

Scene Representation Networks (SRN) [21] represent a continuous scene as an opaque surface, implicitly defined by an MLP that maps each (x, y, z) coordinate to a feature vector. They train a recurrent neural network to march along a ray through the scene representation by using the feature vector at any 3D coordinate to predict the next step size along the ray. The feature vector from the final step is decoded into a single color for that point on the surface. Note that SRN is a better-performing follow-up to DeepVoxels [20] by the same authors, which is why we do not include comparisons to DeepVoxels.

场景表达网络(SRN)表示一个连续场景为一个不透明的表面，通过一个MLP进行隐式的定义，将每个(x,y,z)坐标映射到一个特征向量。他们训练一个循环神经网络，将一条射线行进通过场景表示，使用在任何3D坐标上的特征向量，来沿着射线来预测下一步长。从最后步骤得到的特征向量，解码为在这个表面上的那个点的单个色彩。注意，SRN是DeepVoxels同样作者的后续工作，所以我们就没有与DeepVoxel进行比较。

LLFF [12] is designed for producing photorealistic novel views for well-sampled forward-facing scenes. It uses a trained 3D CNN to directly predict a discretized frustum-sampled RGBα grid (multiplane image or MPI [25]) for each input view, then renders novel views by alpha compositing and blending nearby MPIs into the novel viewpoint.

LLFF设计用于对采样很好的面向前方的场景产生具有照片真实感的新视图。其使用一个训练好的3D CNN来对每个输入视角直接预测一个离散的frustum采样的RGBα网格，然后通过alpha合成和混合附近的MPIs成新视角来渲染新视图。

### 6.3. Discussion

We thoroughly outperform both baselines that also optimize a separate network per scene (NV and SRN) in all scenarios. Furthermore, we produce qualitatively and quantitatively superior renderings compared to LLFF (across all except one metric) while using only their input images as our entire training set.

我们在所有场景中，对同样优化每个场景一个单独网络的基准(NV和SRN)，都超过了他们。而且，与LLFF相比，在使用他们的输入图像作为我们的整个训练集时，在所有度量上（除了一个），在定性和定量两个维度上，都产生了更好的渲染效果。

The SRN method produces heavily smoothed geometry and texture, and its representational power for view synthesis is limited by selecting only a single depth and color per camera ray. The NV baseline is able to capture reasonably detailed volumetric geometry and appearance, but its use of an underlying explicit 128 [3] voxel grid prevents it from scaling to represent fine details at high resolutions. LLFF specifically provides a “sampling guideline” to not exceed 64 pixels of disparity between input views, so it frequently fails to estimate correct geometry in the synthetic datasets which contain up to 400–500 pixels of disparity between views. Additionally, LLFF blends between different scene representations for rendering different views, resulting in perceptually distracting inconsistency as is apparent in our supplementary video.

SRN方法产生了非常平滑的几何和纹理，对视图合成的表示能力是有限的，对每条相机射线只选择一个深度和色彩。NV基准可以捕获相当细节的体几何和外观，但其使用的是潜在的隐式的128个体素网格，这使其无法放大表示分辨率更高的更精细的细节。LLFF特别的给出了一个采样指南，使输入视图之间不要超过64像素视差，所以在合成数据集中，视图之间的视差有400-500个像素，该方法经常不能估计正确的几何。另外，LLFF将不同的场景表示混合，以渲染不同的视图，得到上在感官上不一致的效果，在附属的视频中很明显。

The biggest practical trade-offs between these methods are time versus space. All compared single scene methods take at least 12 hours to train per scene. In contrast, LLFF can process a small input dataset in under 10 min. However, LLFF produces a large 3D voxel grid for every input image, resulting in enormous storage requirements (over 15GB for one “Realistic Synthetic” scene). Our method requires only 5MB for the network weights (a relative compression of 3000× compared to LLFF), which is even less memory than the input images alone for a single scene from any of our datasets.

这些方法之间最实际的折中是空间与时间上的。所有比较的单场景方法在每个场景中的训练时间都至少12个小时。比较之下，LLFF可以在少于10分钟内处理一个小型输入数据集。但是，LLFF对每个输入图像产生一个大型3D体素网格，所以存储需求很大（对一个Realistic Synthetic场景需要超过15GB）。我们的方法只需要5MB，用于网络权重（与LLFF相比，压缩了3000x），这甚至比输入图像要更小。

## 7. Conclusion

Our work directly addresses deficiencies of prior work that uses MLPs to represent objects and scenes as continuous functions. We demonstrate that representing scenes as 5D neural radiance fields (an MLP that outputs volume density and view-dependent emitted radiance as a function of 3D location and 2D viewing direction) produces better renderings than the previously dominant approach of training deep CNNs to output discretized voxel representations.

我们的工作直接处理了之前工作中的缺点，使用MLPs来表示目标和场景为连续函数。我们展示了，将场景表示为5D神经辐射场（一个MLP，输入为3D位置和2D视图方向，输出为体密度和依赖于视图的发射的辐射），与之前主流的训练深层CNNs来输出离散化体素表示的方法相比，会产生更好的渲染。

We believe that this work makes progress toward a graphics pipeline based on real-world imagery, where complex scenes could be composed of neural radiance fields optimized from images of actual objects and scenes. Indeed, many recent methods have already built upon the neural radiance field representation presented in this work and extended it to enable more functionality such as relighting, deformations, and animation.

我们相信，这个网络基于真实世界图像可以进行图形学处理，其中的复杂场景可以由神经辐射场组成，可以从真实目标和场景的图像优化得到。很多最近的方法已经在神经辐射场表示的基础上进行，拓展了更多功能，比如，重新加入光照，形变和动画。