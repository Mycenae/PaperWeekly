# SimpleITK Documentation

## 1. Fundamental Concepts

The two basic elements which are at the heart of SimpleITK are images and spatial transformations. These follow the same conventions as the ITK components which they represent. The fundamental underlying concepts are described below.

SimpleITK核心的两个基本要素是图像和空间变换。这遵循与ITK部件所表示的传统是一样的。基本的潜在概念如下所述。

### 1.1 Images

The fundamental tenet of an image in ITK and consequentially in SimpleITK is that an image is defined by a set of points on a grid occupying a physical region in space. This significantly differs from many other image analysis libraries that treat an image as an array which has two implications: (1) pixel/voxel spacing is assumed to be isotropic and (2) there is no notion of an image’s location in physical space.

在ITK中一幅图像的基本原则，也就是在SimpleITK中的，是一幅图像是定义在一个网格的点集中的，占据了空间中一定的物理区域。这与很多其他图像处理库有显著的不同，那些库将图像作为一个阵列来对待，这有两个含义：(1)像素/体素的间距被认为是各向同性的，(2)没有图像在物理空间中的位置的概念。

SimpleITK images are multi-dimensional (the default configuration includes images from 2D upto 5D) and can be a scalar, labelmap (scalar with run length encoding), complex value or have an arbitrary number of scalar channels (also known as a vector image). The region in physical space which an image occupies is defined by the image’s:

SimpleITK图像是多维的（默认的配置包括2D到5D的图像），可以是一个标量，标签图（带有行程编码的标量），复值或任意数量的标量通道（也称为向量图）。在物理空间中，一幅图像占据的区域，是由图像的下列要素定义的：

1. Origin (vector like type) - location in the world coordinate system of the voxel with all zero indexes. 原点（类似向量的类别）。在世界坐标系中，索引都是0的体素的位置。
2. Spacing (vector like type) - distance between pixels along each of the dimensions. 间距（类似向量的类型）。沿着每个维度，像素之间的距离。
3. Size (vector like type) - number of pixels in each dimension. 大小（类似向量的类型）。在每个维度中像素的数量。
4. Direction cosine matrix (vector like type representing matrix in row major order) - direction of each of the axes corresponding to the matrix columns. 方向cosine矩阵（类似向量的类型，以行优先的顺序来表示矩阵）。每个轴的方向对应着矩阵列。

The meaning of each of these meta-data elements is visually illustrated in this [figure](https://simpleitk.readthedocs.io/en/master/_images/ImageOriginAndSpacing.svg). 每个元数据元素的意义，在此图中可视化的显示。

An image in SimpleITK occupies a region in physical space which is defined by its meta-data (origin, size, spacing, and direction cosine matrix). Note that the image’s physical extent starts half a voxel before the origin and ends half a voxel beyond the last voxel.

在SimpleITK中的一幅图像，占据了物理空间中的一个区域，由其元数据定义（原点，大小，间距和方向旋转矩阵）。注意图像的物理范围从原点前半个体素开始，在最后一个体素后半个体素结束。

In SimpleITK, when we construct an image we specify its dimensionality, size and pixel type, all other components are set to reasonable default values: 在SimpleITK中，当我们构建一幅图像，我们指定其维度，大小和像素类型，所有其他元素都设置为合理的默认值：

1. origin - all zeros. 原点 - 都是0。
2. spacing - all ones. 间距 - 都是1。
3. direction - identity. 方向 - 单位向量。
4. intensities in all channels - all zero. 在所有通道中的灰度 - 都是0。

In the following Python code snippet we illustrate how to create a 2D image with five float valued channels per pixel, origin set to (3, 14) and a spacing of (0.5, 2). Note that the metric units associated with the location of the image origin in the world coordinate system and the spacing between pixels are unknown (km, m, cm, mm,…). It is up to you the developer to be consistent. More about that below.

在下面的Python代码片段中，我们描述了怎样创建一个2D图像，每个像素有5个浮点值的通道，原点设置在(3,14)，间距为(0.5,2)。注意在世界坐标系中的与图像原点位置相关的度量单位和像素间距是未知的（km, m, cm, mm,…）。需要开发者来保持一致。

```
image = sitk.Image([10,10], sitk.sitkVectorFloat32, 5)
image.SetOrigin((3.0, 14.0))
image.SetSpacing((0.5, 2))
```

The tenet that images occupy a spatial location in the physical world has to do with the original application domain of ITK and SimpleITK, medical imaging. In that domain images represent anatomical structures with metric sizes and spatial locations. Additionally, the spacing between voxels is often non-isotropic (most commonly the spacing along the inferior-superior/foot-to-head direction is larger). Viewers that treat images as an array will display a distorted image as shown in this [figure](https://simpleitk.readthedocs.io/en/master/_images/nonisotropicVsIsotropic.svg).

图像在物理世界中占据一个空间位置的原则，与ITK和SimpleITK的原始应用领域有关，即医学影像。在这个领域中，图像表示有度量大小的和空间位置的解剖结构。另外，体素之间的间距通常不是各向同性的（最常见的是，沿着头脚方向的间距更大）。将图像视为阵列的观察者，会展示出一幅扭曲的图像，如图所示。

As an image is also defined by its spatial location, two images with the same pixel data and spacing may not be considered equivalent. Think of two CT scans of the same patient acquired at different sites. This [figure](https://simpleitk.readthedocs.io/en/master/_images/spatialRelationship.svg) illustrates the notion of spatial location in the physical world, the two images are considered different even though the intensity values and pixel spacing are the same.

由于一幅图像也是由其空间位置所定义的，有相同像素数据和间隔的两幅图像，也不一定是等价的。相像一下有同一个病人在不同位置获取的两个CT scans。下图描述了在物理世界中的空间位置的概念，即使其灰度值和像素间距是一样的，这两幅图像也被认为是不同的。

Two images with exactly the same pixel data, positioned in the world coordinate system. In SimpleITK these are not considered the same image, because they occupy different spatial locations. The image on the left has its origin at (-136.3, -20.5) with a direction cosine matrix, in row major order, of (0.7, -0.7, 0.7, 0.7). The image on the right’s origin is (16.9, 21.4) with a direction cosine matrix of (1,0,0,1).

As SimpleITK images occupy a physical region in space, the quantities defining this region have metric units (cm, mm, etc.). In general SimpleITK assume units are in millimeters (historical reasons, due to DICOM standard). In practice SimpleITK is not aware of the specific units associated with each image, it just assumes that they are consistent. Thus, it is up to you the developer to ensure that all of the images you read and created are using the same units. Mixing units and using wrong units has not ended well in the past.

由于SimpleITK的图像占据了空间中的一个物理区域，定义了这个区域的量有度量单位（cm, mm等）。总体上，SimpleITK假设单位是mm（历史原因，由于DICOM标准）。在实际中，SimpleITK并不知道与每幅图像相关的具体的单位，只假设它们是一致的。因此，需要开发者来确保所有读取创建的图像使用相同的单位。混合单位和使用错误的单位在过去都没有得到很好的结果。

Finally, having convinced you to think of images as objects occupying a physical region in space, we need to answer two questions:

最后，说服你将图像视为占据空间中一定物理区域的目标，我们需要回答两个问题：

1. How do you access the pixel values in an image: 怎样访问一幅图像中的像素值：

```
image.GetPixel((0,0))
```
SimpleITK functions use a zero based indexing scheme. The toolkit also includes syntactic sugar that allows one to use the bracket operator in combination with the native zero/one based indexing scheme (e.g. a one based indexing in R vs. the zero based indexing in Python).

SimpleITK的函数使用开始为0的索引方案。工具箱也包括一些句法，使人可以使用括号运算符与本地的开始为0/1的索引方案（如，R中开始为1的索引，与Python中开始为0的索引）。

2. How do you determine the physical location of a pixel: 怎样确定一个像素的空间位置：
```
image.TransformIndexToPhysicalPoint((0,0))
```
This computation can also be done manually using the meta-data defining the image’s spatial location, but we highly recommend that you do not do so as it is error prone.

这种计算也可以手动进行，使用元数据中定义的图像空间位置，但我们高度推荐这样做，因为手动进行容易出错。

#### Channels

As stated above, a SimpleITK image can have an arbitrary number of channels with the content of the channels being a scalar or complex value. This is determined when an image is created.

如上所述，一幅SimpleITK图像可以有任意数量的通道，通道中的内容可以是标量，或复数值。这在图像创建时确定。

In the medical domain, many image types have a single scalar channel (e.g. CT, US). Another common image type is a three channel image where each channel has scalar values in [0,255], often people refer to such an image as an RGB image. This terminology implies that the three channels should be interpreted using the RGB color space. In some cases you can have the same image type, but the channel values represent another color space, such as HSV (it decouples the color and intensity information and is a bit more invariant to illumination changes). SimpleITK has no concept of color space, thus in both cases it will simply view a pixel value as a 3-tuple.

在医学领域，很多图像类型都有一个标量通道（如，CT，US）。另一种常见的图像类型是一个三通道图像，每个通道都是[0,255]范围内的标量值，通常称这种图像为RGB图像。这种术语意思是，这三个通道应当解释为RGB色彩空间。在一些情况下，我们可以有相同的图像类型，但通道值表示另一种色彩空间，如HSV（这将色彩和灰度信息进行了解耦，对光照变化具有更多的不变性）。SimpleITK没有色彩空间的概念，因此在两种情况中，只会简单的将像素值视为一个三元组。

Word of caution: In some cases looks may be deceiving. Gray scale images are not always stored as a single channel image. In some cases an image that looks like a gray scale image is actually a three channel image with the intensity values repeated in each of the channels. Even worse, some gray scale images can be four channel images with the channels representing RGBA and the alpha channel set to all 255. This can result in a significant waste of memory and computation time. Always become familiar with your data.

### 1.2 Transforms

SimpleITK supports two types of spatial transforms, ones with a global (unbounded) spatial domain and ones with a bounded spatial domain. Points in SimpleITK are mapped by the transform using the TransformPoint method.

SimpleITK支持两种类型的空间变换，一种是全局的（无限制的）空间领域，一种是有限的空间领域。SimpleITK中的点通过变换映射，是使用TransformPoint方法。

All global domain transforms are of the form: 所有的全局领域变换具有下面的形式：

$$T(x)=A(x-c)+t+c$$

The nomenclature used in the documentation refers to the components of the transformations as follows: 本文献中的术语中，变换的各个部分的意义如下：

- Matrix - the matrix A. 矩阵A。
- Center - the point c. 中心，点c。
- Translation - the vector t. 平移，向量t。
- Offset - the expression t+c−Ac. 偏移，表达式t+c-Ac。

A variety of global 2D and 3D transformations are available (translation, rotation, rigid, similarity, affine…). Some of these transformations are available with various parameterizations which are useful for registration purposes.

有很多全局2D和3D变换可用（平移，旋转，刚体，相似度，仿射等）。这些变换中的一些有各种参数，可用于配准目的。

The second type of spatial transformation, bounded domain transformations, are defined to be identity outside their domain. These include the B-spline deformable transformation, often referred to as Free-Form Deformation, and the displacement field transformation.

第二种类型的空间变换，有界限的空间变换，在其界限之外定义为恒等变换。这包括B样条形变变换，通常称为自由形态形变，和偏移场变换。

The B-spline transform uses a grid of control points to represent a spline based transformation. To specify the transformation the user defines the number of control points and the spatial region which they overlap. The spline order can also be set, though the default of cubic is appropriate in most cases. The displacement field transformation uses a dense set of vectors representing displacement in a bounded spatial domain. It has no implicit constraints on transformation continuity or smoothness.

B样条变换使用控制点网格来表示基于样条的变换。为指定变换，用户定义控制点的数据和它们覆盖的空间区域。样条阶数也可以设置，在多数情况下，默认的三阶就很合适。偏移场变换使用向量密集集合来表示一个有限空间领域的偏移。在变换连续性或平滑性上，没有隐式的约束。

Finally, SimpleITK supports a composite transformation with either a bounded or global domain. This transformation represents multiple transformations applied one after the other $T_0(T_1(T_2(...T_n(p)...)))$. The semantics are stack based, that is, first in last applied:

最后，SimpleITK支持复合变换，可以是有界的，也可以是全局的。这个变换表示多个变换一个接一个应用$T_0(T_1(T_2(...T_n(p)...)))$。其语义是基于累积的，即第一个进入，最后应用。

```
composite_transform = CompositeTransform([T0, T1])
composite_transform.AddTransform(T2)
```

In the context of registration, if you use a composite transform as the transformation that is optimized, only the parameters of the last transformation Tn will be optimized over.

在配准的上下文中，如果我们使用一个复合变换作为要优化的变换，只有最后一个变换的参数会被优化。

### 1.3 Resampling

Resampling, as the verb implies, is the action of sampling an image, which itself is a sampling of an original continuous signal.

重采样，如同这个动词的意思一样，是对一幅图像进行采样的动作，是对原始的连续信号的一种采样。

Generally speaking, resampling in SimpleITK involves four components: 一般来说，SimpleITK中的重采样包含四个元素：

一般来说，SimpleITK中的重采样包括4个部分：

1. Image - the image we resample, given in coordinate system m. 图像 - 我们重采样的图像，在给定的坐标系统m中。
2. Resampling grid - a regular grid of points given in coordinate system f which will be mapped to coordinate system m. 重采样的网格 - 在坐标系统f中给定的规则网格点，映射到坐标中m中。
3. Transformation $T_f^m$ - maps points from coordinate system f to coordinate system m, $^mp=T_f^m(^fp)$. 变换$T_f^m$。将坐标系统f中的点映射到坐标系统m中，$^mp=T_f^m(^fp)$。
4. Interpolator - method for obtaining the intensity values at arbitrary points in coordinate system m from the values of the points defined by the Image. 插值器 - 从图像中定义的点的灰度值中，在坐标系统m中，得到任意点的灰度值的方法。

While SimpleITK provides a large number of interpolation methods, the two most commonly used are sitkLinear and sitkNearestNeighbor. The former is used for most interpolation tasks and is a compromise between accuracy and computational efficiency. The later is used to interpolate labeled images representing a segmentation. It is the only interpolation approach which will not introduce new labels into the result.

SimpleITK有很多插值方法，两种最常用的是sitkLinear和sitkNearestNeighbor。前者用于多数插值任务，是准确率与计算效率的折中。后者用于插值标注图像，表示一个分割。这是唯一一种不会向结果中引入新的标签的插值方法。

The SimpleITK interface includes three variants for specifying the resampling grid: SimpleITK的界面，在重采样网格上，包含三个变体：

1. Use the same grid as defined by the resampled image. 使用重采样图像定义的同样的网格；
2. Provide a second, reference, image which defines the grid. 提供另一个参考的图像，定义了网格；
3. Specify the grid using: size, origin, spacing, and direction cosine matrix. 使用下面的事来指定网格：大小，原点，间距和方向cosine矩阵。

Points that are mapped outside of the resampled image’s spatial extent in physical space are set to a constant pixel value which you provide (default is zero).

映射超出重采样的图像的物理空间范围的点，设置为我们给定的常数值（默认为0）。

Common Errors 常见错误

It is not uncommon to end up with an empty (all black) image after resampling. This is due to: 重采样后得到一个全空（全黑）的图像，这也会经常出现。这是因为：

1. Using wrong settings for the resampling grid (not too common, but does happen). 对重采样的网格使用了错误的设置（并不太常见，但经常发生）。
2. Using the inverse of the transformation $T_f^m$. This is a relatively common error, which is readily addressed by invoking the transformation’s GetInverse method. 使用了变换$T_f^m$的逆。这是一个相对常见的错误，可以通过调用变换的GetInverse方法进行解决。

## 2. Registration Overview

The goal of registration is to estimate the transformation which maps points from one image to the corresponding points in another image. The transformation estimated via registration is said to map points from the fixed image coordinate system to the moving image coordinate system.

配准的目标是估计一个变换，从一幅图像中将点映射到另一幅图像中的对应点。通过配准估计的变换，一般说成，将固定图像坐标系中的点映射到移动图像坐标系中。

SimpleITK provides a configurable multi-resolution registration framework, implemented in the ImageRegistrationMethod class. In addition, a number of variations of the Demons registration algorithm are implemented independently from this class as they do not fit into the framework.

SimpleITK提供了一个可配置的多分辨率配准框架，在ImageRegistrationMethod类中实现。另外，Demons配准算法的几个变体也在这个类中有独立的实现，因为与这个框架并不适配。

### 2.1 ImageRegistrationMethod

To create a specific registration instance using the ImageRegistrationMethod you need to select several components which together define the registration instance: 为使用ImageRegistrationMethod创建一个具体的配准实例，你需要选择几个部分，共同定义了配准的实例：

1. Transformation. 变换
2. Similarity metric. 相似度度量
3. Optimizer. 优化器
4. Interpolator. 插值器

#### 2.1.1 Transform

The type of transformation defines the mapping between the two images. SimpleITK supports a variety of global and local transformations. The available transformations include:

变换的类型定义了两幅图像之间的映射。SimpleITK支持很多全局和局部变换。可用的变换包括：

- TranslationTransform.
- VersorTransform.
- VersorRigid3DTransform.
- Euler2DTransform.
- Euler3DTransform.
- Similarity2DTransform.
- Similarity3DTransform.
- ScaleTransform.
- ScaleVersor3DTransform.
- ScaleSkewVersor3DTransform.
- AffineTransform.
- BSplineTransform.
- DisplacementFieldTransform.
- Composite Transform.

The parameters modified by the registration framework are those returned by the transforms GetParameters() method. This requires special attention when the using a composite transform, as the specific parameters vary based on the content of your composite transformation.

配准框架修正的参数是那些由变换的GetParameters()方法返回的值。这在使用复合变换的时候，需要特殊的注意，因为复合变换具体内容的不同，具体的参数会变化很多。

#### 2.1.2 Similarity Metric

The similarity metric reflects the relationship between the intensities of the images (identity, affine, stochastic…). The available metrics include: 相似性度量反映了图像灰度之间的关系（恒等，仿射，随机...）。可用的度量包括：

- MeanSquares. 均方差
- Demons.
- Correlation. 相关
- ANTSNeighborhoodCorrelation. ANTS邻域相关
- JointHistogramMutualInformation. 联合直方图互信息
- MattesMutualInformation. Mattes互信息

In the ITKv4 and consequentially in SimpleITK all similarity metrics are minimized. For metrics whose optimum corresponds to a maximum, such as mutual information, the metric value is negated internally. The selection of a similarity metric is done using the ImageRegistrationMethod’s SetMetricAsX() methods. 在ITKv4以及SimpleITK中，所有的相似性度量都是最小化的。对于有的度量，其最优值对应一个最大值，比如互信息，度量值在内部就取了负值。相似性度量的选择，使用ImageRegistrationMethod的SetMetricAsX()方法。

#### 2.1.3 Optimizer

The optimizer is selected using the SetOptimizerAsX() methods. When selecting the optimizer you will also need to configure it (e.g. set the number of iterations). The available optimizers include: 优化器的选择是使用SetOptimizerAsX()方法。当选择优化器时，同时还需要取配置它（如，设置迭代次数）。可用的优化器包括：

Gradient free:

- Exhaustive.
- Nelder-Mead downhill simplex (Amoeba).
- Powell.
- 1+1 evolutionary optimizer.

Gradient based:

- Gradient Descent.
- Gradient Descent Line Search.
- Regular Step Gradient Descent.
- Conjugate Gradient Line Search.
- L-BFGS-B. Limited memory Broyden, Fletcher, Goldfarb, Shannon, Bound Constrained (supports the use of simple constraints).

#### 2.1.4 Interpolator

SimpleITK has a large number of interpolators. In most cases linear interpolation, the default setting, is sufficient. Unlike the similarity metric and optimizer, the interpolator is set using the SetInterpolator method which receives a parameter indicating the interpolator type.

SimpleITK有很多插值器。多数情况下，线性插值器就足够了，也就是默认设置。与相似性度量和优化器不同，插值器是使用SetInterpolator方法来设置的，有一个参数指示插值器的类型。

#### 2.1.5 Features of Interest

**Transforms and image spaces**

While the goal of registration, as defined above, refers to a single transformation and two images, the ITKv4 registration and the SimpleITK ImageRegistrationMethod provide additional flexibility in registration configuration.

如上定义，配准的目标涉及到一个变换和两幅图像，ITKv4配准和SimpleITK ImageRegistrationMethod方法在配准配置中提供了额外的灵活性。

From a coordinate system standpoint ITKv4 introduced the virtual image domain, making registration a symmetric process so that both images are treated similarly. As a consequence the ImageRegistrationMethod has methods for setting three transformations:

从坐标系的观点，ITKv4引入了虚拟图像域，使配准成为了一个对称过程，这样两幅图像都是类似对待的。结果是，ImageRegistrationMethod有设置三种变换的方法：

1. SetInitialTransform To - composed with the moving initial transform, maps points from the virtual image domain to the moving image domain, modified during optimization.

2. SetFixedInitialTransform Tf - maps points from the virtual image domain to the fixed image domain, never modified.

3. SetMovingInitialTransform Tm - maps points from the virtual image domain to the moving image domain, never modified.

The transformation that maps points from the fixed to moving image domains is thus:

$$p_{moving} = T_o (T_m (T_f^{-1}(p_{fixed})))$$

**Multi Resolution Framework**

The ImageRegistrationMethod supports multi-resolution, pyramid, registration via two methods SetShrinkFactorsPerLevel and SetSmoothingSigmasPerLevel. The former receives the shrink factors to apply when moving from one level of the pyramid to the next and the later receives the sigmas to use for smoothing when moving from level to level. Sigmas can be specified either in voxel units or physical units (default) using SetSmoothingSigmasAreSpecifiedInPhysicalUnits.

ImageRegistrationMethod通过两个方法支持多分辨率、金字塔配准，SetShrinkFactorsPerLevel和SetSmoothingSigmasPerLevel。前者接收一个收缩系数的参数，以从金字塔的一个层级，到另一个层级；后者接收一个sigmas的参数，在从一个层级到另一个层级的时候，进行平滑。Sigmas可以以体素单位指定，或物理单位（默认是这个设置）指定，使用的函数是SetSmoothingSigmasAreSpecifiedInPhysicalUnits。

**Sampling**

For many registration tasks one can use a fraction of the image voxels to estimate the similarity measure. Aggressive sampling can significantly reduce the registration runtime. The ImageRegistration method allows you to specify how/if to sample the voxels, SetMetricSamplingStrategy, and if using a sampling, what percentage, SetMetricSamplingPercentage.

对很多配准任务，可以使用图像体素的一部分，来估计相似性度量。激进的取样可以显著降低配准运行时间。ImageRegistration方法使你可以指定怎样/如果去采样体素的话，SetMetricSamplingStrategy，以及如果使用一个采样的话，使用怎样的百分比，SetMetricSamplingPercentage。

**Scaling in Parameter Space**

The ITKv4 framework introduced automated methods for estimating scaling factors for non-commensurate parameter units. These change the step size per parameter so that the effect of a unit of change has similar effects in physical space (think rotation of 1 radian and translation of 1 millimeter). The relevant methods are SetOptimizerScalesFromPhysicalShift, SetOptimizerScalesFromIndexShift and SetOptimizerScalesFromJacobian. In many cases this scaling is what determines if the the optimization converges to the correct optimum.

ITKv4框架引入了自动方法，以对参数单位估计缩放因子。这对每个参数都改变了步长大小，这样单位变化的效果在物理空间也有类似的效果（相像一下1弧度的旋转和1mm的平移）。相关的方法是SetOptimizerScalesFromPhysicalShift，SetOptimizerScalesFromIndexShift和SetOptimizerScalesFromJacobian。在很多情况中，这种缩放是确定优化过程是否收敛到正确最优值的因素。

**Observing Registration Progress**

The ImageRegistrationMethod enables you to observe the registration process as it progresses. This is done using the Command-Observer pattern, associating callbacks with specific events. To associate a callback with a specific event use the AddCommand method.

ImageRegistrationMethod使你可以观察到配准过程。这是通过使用Command-Observer模式完成的，将callbacks与具体的事件关联起来。为将一个callback与一个具体的事件联系起来，我们使用AddCommand方法。