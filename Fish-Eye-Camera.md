# A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses

Juho Kannala, and Sami S. Brandt

University of Oulu, Helsinki University of Technology

## 0. Abstract

Fish-eye lenses are convenient in such applications where a very wide angle of view is needed, but their use for measurement purposes has been limited by the lack of an accurate, generic, and easy-to-use calibration procedure. We hence propose a generic camera model, which is suitable for fish-eye lens cameras as well as for conventional and wide-angle lens cameras, and a calibration method for estimating the parameters of the model. The achieved level of calibration accuracy is comparable to the previously reported state-of-the-art.

鱼眼镜头在需要极宽视角的应用中非常方便，但由于一直缺乏准确、通用和容易使用的校准过程，其测量目的一直受到限制。因此我们提出了一个通用相机模型，适用于鱼眼镜头相机，以及传统的和广角镜头相机，以及估计模型参数的标定方法。获得的标定准确率水平，与之前的最好结果类似。

**Index Terms**—Camera model, camera calibration, lens distortion, fish-eye lens, wide-angle lens.

## 1. Introduction

THE pinhole camera model accompanied with lens distortion models is a fair approximation for most conventional cameras with narrow-angle or even wide-angle lenses [1], [2], [3]. But, it is still not suitable for fish-eye lens cameras. Fish-eye lenses are designed to cover the whole hemispherical field in front of the camera and the angle of view is very large, about 180°. Moreover, it is impossible to project the hemispherical field of view on a finite image plane by a perspective projection, so fish-eye lenses are designed to obey some other projection model. This is the reason why the inherent distortion of a fish-eye lens should not be considered only as a deviation from the pinhole model [4].

针孔相机模型与镜头形变模型，是对多数窄角度传统相机，甚至是广角镜头相机，都是一个不错的近似。但是，对鱼眼镜头相机仍然并不合适。鱼眼镜头在设计上就是要覆盖相机前面的整个半球形区域，大概180度。而且，将半球场视角用透视投影法投影到一个有限的图像平面，这是不可能的，所以鱼眼镜头从设计上就是遵从一些其他投影模型的。这就是为什么，鱼眼镜头的内在形变，不应当只被考虑为针孔模型的一种变化。

There have been some efforts to model the radially symmetric distortion of fish-eye lenses with different models [5], [6], [7]. The idea in many of these approaches is to transform the original fish-eye image to follow the pinhole model. In [6] and [7], the parameters of the distortion model are estimated by forcing straight lines straight after the transformation, but the problem is that the methods do not give the full calibration. They can be used to “correct” the images to follow the pinhole model but their applicability is limited when one needs to know the direction of a back-projected ray corresponding to an image point. The calibration procedures in [8] and [9] instead aim at calibrating fish-eye lenses generally. However, these methods are slightly cumbersome, in practice, because a laser beam or a cylindrical calibration object is required.

用不同的模型来建模鱼眼镜头的镜像对称畸变，有一些工作。这些方法中很多都是对鱼眼图像进行变换，使其符合针孔模型。在[6,7]中，畸变模型的参数是通过，迫使变换后为直线，来进行估计的，但问题是，这些方法没有给出完整的标定。他们可以用于将图像进行修正，以符合针孔模型，但其应用范围是有限的，如果想要知道一个图像点对应的反向投影的射线的方向，就不行了。[8,9]中的标定过程的目标就是对鱼眼镜头进行通用标定。但是，这些方法在实践中略微麻烦，因为需要用一个激光束或圆柱形的标定目标。

Recently, the first autocalibration methods for fish-eye lens cameras have also emerged [10], [11], [12]. Micusik and Pajdla [10] proposed a method for simultaneous linear estimation of epipolar geometry and an omnidirectional camera model. Claus and Fitzgibbon [11] presented a distortion model which likewise allows the simultaneous linear estimation of camera motion and lens geometry, and Thirthala and Pollefeys [12] used the multi-view geometry of radial 1D cameras to estimate a nonparametric camera model. In addition, the recent work by Barreto and Daniilidis [13] introduced a radial fundamental matrix for correcting the distortion of wide-angle lenses. Nevertheless, the emphasis in these approaches is more in the autocalibration techniques than in the precise modeling of real lenses.

最近，出现了鱼眼镜头相机的第一个自动标定方法[10,11,12]。[10]提出了一个方法，对对极几何和全向相机模型的同时线性估计。[11]提出一个畸变模型，也可以进行相机运动和镜头几何的同时线性估计，[12]使用径向1D相机的多视几何，来估计非参数相机模型。另外，[13]的最近工作提出了一个径向的基础矩阵，修正广角镜头的畸变。尽管如此，这些方法中的重点是在自动标定技术，而不是对真实镜头的精确建模。

In this paper, we concentrate on accurate geometric modeling of real cameras. We propose a novel calibration method for fish-eye lenses that requires that the camera observes a planar calibration pattern. The calibration method is based on a generic camera model that will be shown to be suitable for different kind of omnidirectional cameras as well as for conventional cameras. First, in Section 2, we present the camera model and, in Section 3, theoretically justify it by comparing different projection models. In Section 4, we describe a procedure for estimating the parameters of the camera model, and the experimental results are presented and discussed in Sections 5 and 6.

本文中，我们聚焦在真实相机的精确几何建模上。我们对鱼眼镜头提出一种新的标定方法，需要相机观察一个平面标定模式。标定方法是基于通用相机模型的，对不同类型的全向相机以及传统相机，都是适用的。在第2部分，我们提出相机模型，在第3部分，通过比较不同的投影模型来进行理论上的证明。在第4部分，我们描述了估计相机模型参数的过程，在第5第6部分给出了试验结果并进行讨论。

## 2. Generic Camera Model

Since the perspective projection model is not suitable for fish-eye lenses we use a more flexible radially symmetric projection model. This basic model is introduced in Section 2.1 and then extended with asymmetric distortion terms in Section 2.2. Computation of back-projections is described in Section 2.3.

由于透视投影模型不适用于鱼眼相机，我们适用一种更灵活的径向对称投影模型。在2.1节中给出这个基本模型，在2.2节中用非对称畸变项进行拓展。反向投影的计算在2.3节中给出。

### 2.1 Radially Symmetric Model

The perspective projection of a pinhole camera can be described by the following formula 针孔相机的透视投影可以用下式描述

$$r = f tan(θ),  (i. perspective projection)$$(1)

where θ is the angle between the principal axis and the incoming ray, r is the distance between the image point and the principal point, and f is the focal length. Fish-eye lenses instead are usually designed to obey one of the following projections:

其中θ是主轴和入射光线之间的角度，r是图像点和主点之间的距离，f是焦距。鱼眼镜头在设计上则是遵从下面其中一个投影的：

$$r = 2f tan(θ/2), (ii. stereographic projection)$$(2)
$$r = fθ, (iii. equidistance projection)$$(3)
$$r = 2f sin(θ/2), (iv. equisolid angle projection)$$(4)
$$r = f sin(θ), (v. orthogonal projection)$$(5)

Perhaps the most common model is the equidistance projection. The behavior of the different projections is illustrated in Fig. 1a and the difference between a pinhole camera and a fish-eye camera is shown in Fig. 1b.

最常用的模型可能是等距投影。不同投影如图1a所示，针孔相机和鱼眼相机的差别如图1b所示。

The real lenses do not, however, exactly follow the designed projection model. From the viewpoint of automatic calibration, it would also be useful if we had only one model suitable for different types of lenses. Therefore, we consider projections in the general form

真实的镜头不会严格遵循设计的投影模型。从自动标定的角度来看，如果我们有一个模型适用于不同类型的镜头，那会非常有用。因此，我们考虑下面通用形式的投影

$$r(θ) = k_1 θ + k_2 θ^3 + k_3 θ^5 + k_4 θ^7 + k_5 θ^9 + ...,$$(6)

where, without any loss of generality, even powers have been dropped. This is due to the fact that we may extend r onto the negative side as an odd function while the odd powers span the set of continuous odd functions. For computations, we need to fix the number of terms in (6). We found that first five terms, up to the ninth power of θ, give enough degrees of freedom for good approximation of different projection curves. Thus, the radially symmetric part of our camera model contains the five parameters, k1, k2, . . . , k5.

其中，不失一般性，偶数次方都被丢掉了。这是因为，我们会将r拓展到负值，作为一个奇函数，而奇次幂可以张成连续奇函数的集合。为计算需要，我们需要固定(6)中的项的数量。我们发现前5项，到θ的9次方，可以对不同的投影曲线的近似给出足够的自由度。因此，我们的相机模型中的径向对称部分包含5个参数k1, k2, . . . , k5。

Let F be the mapping from the incoming rays to the normalized image coordinates 令F为输入光线到归一化图像坐标系的映射

$$\left( \begin{matrix} x \\ y \end{matrix} \right) = r(θ) \left( \begin{matrix} cosφ \\ sinφ \end{matrix} \right) = F(Φ)$$(7)

where r(θ) contains the first five terms of (6) and $Φ = (θ, φ)^T$ is the direction of the incoming ray. For real lenses, the values of parameters k_i are such that r(θ) is monotonically increasing on the interval [0, θ_max], where θ_max is the maximum viewing angle. Hence, when computing the inverse of F, we may solve θ by numerically finding the roots of a ninth order polynomial and then choosing the real root between 0 and θ_max.

其中r(θ)包含(6)中的前5项，$Φ = (θ, φ)^T$是入射光线的方向。对于真实的镜头，参数k_i的值要确保在区间[0, θ_max]中，r(θ)是单调递增的，其中θ_max是最大观察角度。因此，当计算F的逆时，我们会对θ进行数值求解，找到一个9阶多项式的根，选择在0和θ_max之间的真正的根。

### 2.2 Full Model

Real lenses may deviate from precise radial symmetry and, therefore, we supplement our model with an asymmetric part. For instance, the lens elements may be inaccurately aligned causing that the projection is not exactly radially symmetric. With conventional lenses, this kind of distortion is called decentering distortion [1], [15]. However, there are also other possible sources of imperfections in the optical system and some of them may be difficult to model. For example, the image plane may be tilted with respect to the principal axis or the individual lens elements may not be precisely radially symmetric. Therefore, instead of trying to model all different physical phenomena in the optical system individually, we propose a flexible mathematical distortion model that is just fitted to agree with the observations.

真实的镜头与精确的径向对称模型会有差别，因此，我们对模型加入一个非对称项。比如，镜头单元并没有精确的对齐，导致投影并不是严格的径向对称。对于传统镜头，这种畸变称为非中心畸变。但是，在光学系统中，也有其他可能的缺陷源，一些会很难建模。比如，图像平面对主轴是倾斜的，或者单个镜头单元并不是精确的径向对称的。因此，我们没有去对光学系统中的所有不同物理现象单独进行建模，而是提出了一种灵活的数学畸变模型，对观察到的图像进行正确的拟合。

To obtain a widely applicable, flexible model, we propose to use two distortion terms as follows: One distortion term acts in the radial direction

为得到一个广泛可用的灵活模型，我们提出利用如下两个畸变项，一个畸变项为径向方向

$$Δ_r(θ, φ) = (l_1 θ + l_2 θ^3 + l_3 θ^5) (i_1 cosφ + i_2 sinφ + i_3 cos2φ + i_4 sin 2φ)$$(8)

and the other in the tangential direction 另一个是切向方向

$$Δ_t(θ, φ) = (m_1 θ + m_2 θ^3 + m_3 θ^5) (j_1 cosφ + j_2 sinφ + j_3 cos2φ + j_4 sin 2φ)$$(9)

where the distortion functions are separable in the variables θ and φ. Because the Fourier series of any 2π-periodic continuous function converges in the L2-norm and any continuous odd function can be represented by a series of odd polynomials we could, in principle, model any kind of continuous distortion by simply adding more terms to (8) and (9), as they both now have seven parameters.

其中畸变函数对于变量θ和φ是可分离的。因为任意2π周期的连续函数的Fourier序列都以L2范数收敛，任意连续的奇函数都可以表示为奇多项式的序列，因此原则上我们可以对任意连续畸变进行建模，只要对(8)和(9)加入更多的项，现在它们都有7个参数。

By adding the distortion terms to (7), we obtain the distorted coordinates $xd = (x_d, y_d)^T$ by

对(7)加入畸变项，我们可以用下式得到畸变的坐标$xd = (x_d, y_d)^T$

$$x_d = r(θ) u_r(φ) + Δ_r(θ, φ) u_r(φ) +  Δ_t(θ, φ) u_φ(φ)$$(10)

where u_r(φ) and u_φ(φ) are the unit vectors in the radial and tangential directions. To achieve a complete camera model, we still need to transform the sensor plane coordinates into the image pixel coordinates. By assuming that the pixel coordinate system is orthogonal, we get the pixel coordinates (u, v)^T from

其中u_r(φ)和u_φ(φ)是径向和切向方向的单位向量。为获得完整的相机模型，我们仍然需要将传感器平面的坐标系转换到图像像素坐标系。假设像素坐标系统是正交的，我们从下式得到像素坐标(u, v)^T

$$\left( \begin{matrix} u \\ v \end{matrix} \right) = \left[ \begin{matrix} m_u && 0 \\ 0 && m_v \end{matrix} \right] \left( \begin{matrix} x_d \\ y_d \end{matrix} \right) + \left( \begin{matrix} u_0 \\ v_0 \end{matrix} \right) = A(x_d)$$(11)

where (u_0, v_0)^T is the principal point and m_u and m_v give the number of pixels per unit distance in horizontal and vertical directions, respectively.

其中(u_0, v_0)^T是主点，m_u和m_v分别是水平和垂直方向单位距离上的像素数量。

By combining (10) and (11), we have the forward camera model 将(10)和(11)结合起来，我们就有了前向相机模型

$$m = P_c(Φ)$$(12)

where m = (u, v)^T. This full camera model contains 23 parameters and it is denoted by p_23 in the following. Since the asymmetric part of the model is very flexible, it may sometimes be reasonable to use a reduced camera model in order to avoid over-fitting. This is the case if, for instance, the control points do not cover the whole image area. Leaving out the asymmetric part gives the camera model p_9 with nine parameters: five in the radially symmetric part (7) and four in the affine transformation (11). We did experiments also with the six-parametric model p_6 which contains only two parameters in the radially symmetric part.

其中m = (u, v)^T。这个完整的相机模型包含23个参数，下面表示为p_23。由于模型的非对称部分是非常灵活的，有时候适用一个缩减版的相机模型，以防止过拟合，这是很合理的。比如，如果控制点没有覆盖整个图像区域，就会是这种情况。省略掉非对称部分，会得到9个参数的相机模型p_9：5个(7)中的径向对称部分参数，4个(11)中的仿射变换。我们还有只有6个参数的模型p_6进行试验，这个模型中的径向对称部分只包含2个参数。

### 2.3 Backward Model

Above, we have described our forward camera model P_c. In practice, one also needs to know the backward model

上面，我们描述了前向相机模型P_c。实践中，我们还需要知道反向模型

$$Φ = P_c^{-1} (m)$$(13)

which is the mapping from the image point m = (u, v)^T to the direction of an incoming light ray, Φ = (θ, φ)^T. We write P_c as the composite function P_c = A ∘ D ∘ F, where F is the transformation (7) from the ray direction Φ to the ideal Cartesian coordinates x = (x, y)^T on the image plane, D is the distortion mapping from x to the distorted coordinates $x_d = (x_d, y_d)^T$ and A is the affine transformation (11). We decompose the projection model in this form because, for the inverse transform $P^{-1}_c = F^{-1} ∘ D^{-1} ∘ A^{-1}$, it is straightforward to compute $F^{-1}$ and $A^{-1}$. The more difficult part is to numerically compute $D^{-1}$.

这是从图像点m = (u, v)^T到入射光线方向Φ = (θ, φ)^T的映射。我们将P_c写为复合函数P_c = A ∘ D ∘ F，其中F是变换(7)，从入射光线方向Φ到图像平面理想笛卡尔坐标系x = (x, y)^T，D是畸变映射，从x到畸变坐标$x_d = (x_d, y_d)^T$，A是仿射变换(11)。我们将投影模型分解成这个形式是因为，要计算逆变换$P^{-1}_c = F^{-1} ∘ D^{-1} ∘ A^{-1}$，计算$F^{-1}$ and $A^{-1}$都是很直接的。更难的部分是从数值上计算$D^{-1}$。

Given a point $x_d$, finding $x = D^{-1}(x_d)$ is equivalent to computing the shift s into the expression $x = x_d - s$, where

给定一个点$x_d$，找到$x = D^{-1}(x_d)$等价于，计算一个偏移s，得到表示$x = x_d - s$，其中

$$s = S(Φ) = Δ_r(θ, φ) u_r(φ) +  Δ_t(θ, φ) u_φ(φ)$$(14)

Moreover, we may write $S(Φ) = (S ∘ F^{-1})(x)$ and approximate the shift by the first order Taylor expansion of $S ∘ F^{-1}$ around $x_d$ that yields

进一步，我们可以写成$S(Φ) = (S ∘ F^{-1})(x)$，通过$S ∘ F^{-1}$在$x_d$附近的一阶Taylor展开近似这个偏移，得到

$$s ≃ (S ∘ F^{-1})(x_d) + \frac {∂(S ∘ F^{-1})}{∂x}(x_d)(x-x_d) = S(Φ_d) - \frac{∂S}{∂Φ} (\frac{∂F}{∂Φ}(Φ_d))^{-1}s$$

where $Φ_d = F^{-1}(x_d)$ may be numerically evaluated. Hence, we may compute the shift s from

其中$Φ_d = F^{-1}(x_d)$可以进行数值计算。因此，我们会用下式计算偏移s

$$s ≃ (I + \frac{∂S}{∂Φ}(Φ_d) (\frac{∂F}{∂Φ}(Φ_d))^{-1})^{-1} S(Φ_d)$$(15)

where the Jacobians $∂S/∂Φ$ and $∂F/∂Φ$ may be computed from (14) and (7), respectively. So, finally,

其中Jacobian行列式$∂S/∂Φ$和$∂F/∂Φ$分别可以从(14)和(7)中计算。所以，最后

$$D^{-1}(x_d) ≃ x_d - (I + (\frac{∂S}{∂Φ}∘F^{-1})(x_d) ((\frac{∂F}{∂Φ}∘F^{-1})(x_d))^{-1})^{-1} (S∘ F^{-1})(x_d)$$(16)

It seems that the first order approximation for the asymmetric distortion function D is tenable, in practice, because the backward model error is typically several degrees smaller than the calibration accuracy for the forward model, as will be seen in detail in Section 5.

似乎非对称畸变函数D的一阶近似就可以了，实践中，因为反向模型误差通常比前向模型的标定准确度低几个数量级，这在第5部分中可以看到。

## 3. Justification of the Projection Model

The traditional approach for camera calibration is to take the perspective projection model as a starting point and then supplement it with distortion terms [1], [3], [16]. However, this is not a valid approach for fish-eye lenses because, when θ approaches π/2, the perspective model projects points infinitely far and it is not possible to remove this singularity with the conventional distortion models. Hence, we base our calibration method to the more generic model (6).

相机标定的传统方法是，以透视变换模型开始，附之以畸变项。但是，这对于鱼眼镜头并不有效，因为，当θ接近π/2时，透视模型会把点投影到无限远处，因此用传统畸变模型不能去除掉这个奇点。因此，我们以更通用的模型(6)作为我们标定方法的基础。

We compared the polynomial projection model (6) to the two two-parametric models proposed by Micusık [17] for fish-eye lenses 我们将多项式投影模型(6)与[17]中为鱼眼镜头提出的双参数模型进行了比较

$r = asin(bθ)/b$ (M1) and $r = (a-\sqrt{(a^2-4bθ^2)})/(2bθ)$ (M2)

In Fig. 2, we have plotted the projection curves (1), (2), (3), (4), and (5) and their least-squares approximations with models M1, M2, and P3, where P3 is the polynomial model (6) with the first two terms. Here, we used the value f = 200 pixels which is a reasonable value for a real camera. The projections were approximated between 0 and θ_max, where the values of θ_max were 60°, 110°, 110°, 110°, and 90°, respectively. The interval [0, θ_max] was discretized using the step of 0.1° and the models M1 and M2 were fitted by using the Levenberg-Marquardt method. It can be seen from Fig. 2 that the model M1 is not suitable at all for the perspective and stereographic projections and that the model M2 is not accurate for the orthogonal projection.

在图2中，我们画出了投影曲线(1), (2), (3), (4), 和(5)，以及用模型M1，M2和P3的最小二乘近似，其中P3是多项式模型(6)只有前2项。这里，我们使用值f = 200像素，对一个真实的相机来说，这是合理的。投影在0到θ_max之间进行了近似，其中θ_max分别为60°, 110°, 110°, 110°, 90°。区间[0, θ_max]用0.1°的步长进行离散化，模型M1和M2的拟合是使用Levenberg-Marquardt方法。可以从图2中看到，模型M1对透视和球面投影非常不合适，模型M2对于正交投影不太精确。

In Table 1, we have tabulated the maximum approximation errors for each model, i.e., the maximum vertical distances between the desired curve and the approximation in Fig. 2. Here, we also have the model P9 which is the polynomial model (6) with the first five terms. It can be seen that the model P3 has the best overall performance from all of the two-parametric models and that the sub-pixel approximation accuracy for all the projection curves requires the five-parametric model P9. These results show that the radially symmetric part of our camera model is well justified.

在表1中，我们给出了对每个模型的最大近似误差，即，图2中理想曲线与近似的最大竖直距离。这里，我们还包括了模型P9，即式(6)有前5项的多项式模型。可以看到，模型P3比两个双参数模型都要好，而对所有投影曲线的亚像素近似，则需要5参数模型P9。这些结果表明，我们相机模型的径向对称部分得到了证明。

## 4. Calibrating the Generic Model

Next, we describe a procedure for estimating the parameters of the camera model. The calibration method is based on viewing a planar object which contains control points in known positions. The advantage over the previous approaches is that also fish-eye lenses, possibly having a field of view larger than 180°, can be calibrated by simply viewing a planar pattern. In addition, a good accuracy can be achieved if circular control points are used, as described in Section 4.2.

下一步，我们描述了估计相机模型参数的过程。标定方法是基于观察平面目标的形式，在已经位置上包含控制点。与之前方法相比的优势在于，即使对于视角大于180度的鱼眼镜头，也可以简单的通过一个平面模式进行标定。另外，如果使用圆形控制点，可以得到更好的精度，如4.2所示。

### 4.1. Calibration Algorithm

The calibration procedure consists of four steps that are described below. We assume that M control points are observed in N views. For each view, there is a rotation matrix $R_j$ and a translation vector $t_j$ describing the position of the camera with respect to the calibration plane such that

标定过程包括4个步骤。我们假设以N个视角观察了M个控制点。对每个视角，都有一个旋转矩阵$R_j$和一个平移向量$t_j$，描述了相机相对于标定平面的位置，有

$$X_c = R_j X + t_j, j = 1, ..., N$$(17)

We choose the calibration plane to lie in the XY-plane and denote the coordinates of the control point i with $X^i = (X^i, Y^i, 0)^T$. The corresponding homogeneous coordinates in the calibration plane are denoted by $x^i_p = (X^i, Y^i, 1)^T$ and the observed coordinates in the view j by $m^i_j = (u^i_j, v^i_j)^T$. The first three steps of the calibration procedure involve only six internal camera parameters and for these we use the short-hand notation $p_6  = (k_1, k_2, m_u, m_v, u_0, v_0)$. The additional parameters of the full model are inserted only in the final step.

我们选择标定平面在XY平面上，将控制点i的坐标表示为$X^i = (X^i, Y^i, 0)^T$。在标定平面中对应的齐次坐标表示为$x^i_p = (X^i, Y^i, 1)^T$，在视角j中得到的观察坐标表示为$m^i_j = (u^i_j, v^i_j)^T$。标定过程的前3步，只涉及到6个内部相机参数，对此，我们使用简写$p_6  = (k_1, k_2, m_u, m_v, u_0, v_0)$。完整模型的其他参数在最后一步中再插入。

Step 1: Initialization of internal parameters. The initial guesses for k1 and k2 are obtained by fitting the model $r = k_1θ + k_2θ^3$ to the desired projection, (1)-(5), with the manufacturer’s values for the nominal focal length f and the angle of view θ_max. Then, we also obtain the radius of the image on the sensor plane by $r_{max} = k_1θ_{max} + k_2θ_{max}^3$.

步骤1：内部参数初始化。将模型$r = k_1θ + k_2θ^3$对期望的投影(1)-(5)进行拟合，用制造商的标称焦距f和视角值θ_max，对k1和k2的初始猜测。然后，我们还得到在传感器平面上的图像的半径$r_{max} = k_1θ_{max} + k_2θ_{max}^3$。

With a circular image fish-eye lens, the actual image fills only a circular area inside the image frames. In pixel coordinates, this circle is an ellipse

对于鱼眼镜头的圆形图像，实际的图像只会填充图像帧内部的圆形区域。在像素坐标系中，这个圆是一个椭圆

$$(\frac {u-u_0}{a})^2 + (\frac {v-v_0}{b})^2 = 1$$

whose parameters can be estimated. Consequently, we obtain initial guesses for the remaining unknowns $m_u, m_v, u_0$, and $v_0$ in p, where $m_u = a/r_{max}$ and $m_v = b/r_{max}$. With a full-frame lens, the best thing is probably to place the principal point to the image center and use the reported values of the pixel dimensions to obtain initial values for $m_u$ and $m_v$.

其参数可以进行估计。结果是，我们得到了p中的其余未知数$m_u, m_v, u_0$,和$v_0$的估计，其中$m_u = a/r_{max}$，$m_v = b/r_{max}$。用一个全画幅镜头，最好的是将主点放到图像中央，使用像素维度的给出值来得到$m_u$和$m_v$的初始值。

Step 2: Back-projection and computation of homographies. With the internal parameters p6, we may back-project the observed points $m^i_j$ onto the unit sphere centered at the camera origin (see Fig. 1b). The points on the sphere are denoted by $\tilde x^i_j$. Since the mapping between the points on the calibration plane and on the unit sphere is a central projection, there is a planar homography $H_j$ so that $s\tilde x^i_j = H_jx^i_p$.

步骤2：反向投影并计算单应矩阵。用内部参数p6，我们可以将观察到的点$m^i_j$反向投影到以相机原点为中心的单位球体上（见图1b）。球体上的点表示为$\tilde x^i_j$。由于在标定平面和单位球上的点的映射，是一个中心投影，所以有一个平面单应矩阵$H_j$，使得$s\tilde x^i_j = H_jx^i_p$。

For each view j the homography H_j is computed as follows: 对每个视角j，单应矩阵H_j计算如下：

1. Back-project the control points by first computing the normalized image coordinates 控制点反向投影，首先计算归一化图像坐标

$$\left( \begin{matrix} x^i_j \\ y^i_j \end{matrix} \right) = \left[ \begin{matrix} 1/m_u && 0 \\ 0 && 1/m_v \end{matrix} \right] \left( \begin{matrix} u^i_j - u_0 \\ v^i_j - v_0 \end{matrix} \right)$$

transforming them to the polar coordinates $(r^i_j, φ^i_j) = (x^i_j, y^i_j)$ and, finally, solving $θ^i_j$ from the cubic equation $k_2(θ^i_j)^3 + k_1 θ^i_j - r^i_j = 0$.

将其变换到极坐标，最后，从三次方程求解$θ^i_j$。

2. Set $\tilde x^i_j = (sinφ^i_j sinθ^i_j, cosφ^i_j sinθ^i_j, cosθ^i_j)$.

3. Compute the initial estimate for $H_j$ from the correspondences $\tilde x^i_j ↔ x^i_p$ by the linear algorithm with data normalization [18]. Define $\hat x^i_j$ as the exact image of $x^i_p$ under $H_j$ such that $\hat x^i_j = H_jx^i_p/||H_jx^i_p||$. 从对应性$\tilde x^i_j ↔ x^i_p$中计算$H_j$的初始估计，用[18]的有数据归一化的线性算法。定义$\hat x^i_j$为在$H_j$下的$x^i_p$的精确图像，$\hat x^i_j = H_jx^i_p/||H_jx^i_p||$。

4. Refine the homography H_j by minimizing $\sum_i sin^2 α^i_j$, where $α^i_j$ is the angle between the unit vector $\tilde x^i_j$ and $\hat x^i_j$. 通过最小化$\sum_i sin^2 α^i_j$来精炼单应矩阵H_j，这里$α^i_j$时单位向量$\tilde x^i_j$和$\hat x^i_j$之间的角度。

Step 3: Initialization of external parameters. The initial values for the external camera parameters are extracted from the homographies H_j. It holds that

步骤3：外部参数的初始化。外部相机参数的初始值从单应矩阵H_j中提取出来。下式是成立的

$$s\tilde x^i_j =[R_j, t_j] \left( \begin{matrix} X^i \\ Y^i \\ 0 \\ 1 \end{matrix} \right) = [r^1_j, r^2_j, t_j] \left( \begin{matrix} X^i \\ Y^i \\ 1 \end{matrix} \right)$$

which implies $H_j = [r^1_j, r^2_j, t_j]$, up to scale. Furthermore,

$$r^1_j = λ_j h^1_j, r^2_j = λ_j h^2_j, r^3_j = r^1_j × r^2_j, t_j = λ_j h^3_j$$

where $λ_j = sign(H_j^{3,3})/||h_j^1||$. Because of estimation errors, the obtained rotation matrices are not orthogonal. Thus, we use the singular value decomposition to compute the closest orthogonal matrices in the sense of Frobenius norm [19] and use them as initial guess for each $R_j$.

其中$λ_j = sign(H_j^{3,3})/||h_j^1||$。由于估计误差，得到的旋转矩阵并不是正交的。因此，我们使用SVD来计算最接近的Frobenius范数意义下的正交矩阵，将其用作每个$R_j$的初始估计。

Step 4: Minimization of projection error. If the full model p_23 or the model p_9 is used the additional camera parameters are initialized to zero at this stage. As we have the estimates for the internal and external camera parameters, we use (17), (7) or (10), and (11) to compute the imaging function P_j for each camera, where a control point is projected to $\hat m^i_j = P_j(X^i)$. The camera parameters are refined by minimizing the sum of squared distances between the measured and modeled control point projections

步骤4：投影误差最小化。如果使用了完整模型p_23或模型p_9，额外的相机参数在这个阶段就初始化为0。由于我们有了相机内部和外部参数的估计，我们(17), (7)或(10), 和(11)来对每个相机计算成像函数P_j，其中一个控制点投影成$\hat m^i_j = P_j(X^i)$。相机参数的优化，是最小化测量到的和建模得到的控制点投影的距离均方差

$$\sum_{j=1}^N \sum_{i=1}^M d(m^i_j, \hat m^i_j)^2$$(18)

using the Levenberg-Marquardt algorithm. 使用的是Levenberg-Marquardt算法。

### 4.2. Modification for Circular Control Points

In order to achieve an accurate calibration, we used a calibration plane with white circles on black background since the centroids of the projected circles can be detected with a subpixel level of accuracy [20]. In this setting, however, the problem is that the centroid of the projected circle is not the image of the center of the original circle. Therefore, since $m^i_j$ in (18) is the measured centroid, we should not project the centers as points $\hat m^i_j$.

为获得精确的标定，我们使用一个带有黑色背景白色圆形的标定平面，因为投影的圆形的重心可以以亚像素的精度进行检测到。但是，在这个设置中，问题时投影的圆的重心，并不是原始圆的中心。因此，因为(18)中的$m^i_j$是测量的重心，我们不应当将中心投影为点$\hat m^i_j$。

To avoid the problem above, we propose solving the centroids of the projected circles numerically. We parameterize the interior of the circle at (X0, Y0) with radius R by $X(ρ,α) = (X_0 + ρsinα, Y_0 + ρcosα, 0)^T$. Given the camera parameters, we get the centroid $\hat m$ for the circle by numerically evaluating

为避免上述问题，我们提出用数值方法求解投影的圆的重心。我们将半径为R的圆内部的点(X0, Y0)参数化为$X(ρ,α) = (X_0 + ρsinα, Y_0 + ρcosα, 0)^T$。给定相机参数，我们通过数值求解下式，得到圆的重心$\hat m$

$$\hat m = \frac {\int_0^R \int_0^{2π} \hat m(ρ,α) |detJ(ρ,α)| dαdρ} {\int_0^R \int_0^{2π} |detJ(ρ,α)| dαdρ}$$(19)

where $\hat m(ρ,α) = P(X(ρ,α))$ and J(ρ,α) is the Jacobian of the composite function P ∘ X. The analytical solving of the Jacobian is a rather tedious task, but it can be computed by mathematical software such as Maple.

其中$\hat m(ρ,α) = P(X(ρ,α))$，J(ρ,α)是复合函数P ∘ X的Jacobian矩阵。Jacobian矩阵的解析求解是非常麻烦的工作，但可以用数学软件来计算，如Maple

## 5. Calibration Experiments

### 5.2. Conventional and Wide-angle Lens Camera

The proposed camera model was compared to the camera model used by Heikkila [3]. This model is the skew-zero pinhole model accompanied with four distortion parameters and it is denoted by δ8 in the following.

提出的相机模型与[3]中的相机模型进行了比较。这个模型是skew-zero针孔模型，有4个畸变参数，下面用δ8来表示。

In the first experiment, we used the same data, provided by Heikkila, as in [3]. It was originally obtained by capturing a single image of a calibration object consisting of two orthogonal planes, each with 256 circular control points. The camera was a monochrome CCD camera with a 8.5 mm Cosmicar lens. The second experiment was performed with the Sony DFW-VL500 camera and a wide-angle conversion lens, with total focal length of 3.8 mm. In this experiment, we used six images of the calibration object. There were 1,328 observed control points in total and they were localized by computing their gray-scale centroids [20].

在第一个实验中，我们使用相同的数据，[3]中进行了使用。图像的获得，是拍摄标定目标的单幅图像，包含两个正交平面，每个都有256个圆形的控制点。相机是黑白CCD相机，8.5mm Cosmicar镜头。第二个实验室用Sony DFW-VL500相机和一个广角转换镜头进行的，总计焦距为3.8mm。在这个实验中，我们使用了标定目标的6幅图像。总计有1328个观察到的控制点，通过计算其灰度重心来进行定位。

The obtained RMS residual errors, i.e., the root-mean-squared distances between the measured and modeled control point positions, are shown in Table 2. Especially interesting is the comparison between models δ8 and p9 because they both have eight degrees of freedom. Model p9 gave slightly smaller residuals although it does not contain any tangential distortion terms. The full model p23 gave the smallest residuals.

得到的RMS残差误差，即，测量的和建模的控制点位置的距离的均方根，如表2所示。尤其有趣的是δ8和p9模型的比较，因为都有8个自由度。模型p9给出了略微更小的残差，虽然并不包含任何切向畸变项。完整模型p23给出了最小的残差。

However, in the first experiment the full model may have been partly fitted to the systematic errors of the calibration data. This is due to the fact that there were measurements only from one image where the illumination was not uniform and all corners were not covered by control points. To illustrate the fact, the estimated asymmetric distortion and remaining residuals for the model p23 are shown in Fig. 3. The relatively large residuals in the lower right corner of the calibration image (Fig. 3b) seem to be due to inaccurate localization, caused by nonuniform lighting.

但是，在第一个实验中，完整模型可能部分拟合到了标定数据的系统误差。这是因为，只有一幅图像的测量结果，其光照并不是均匀的，控制点并不包含所有角点。为描述这个事实，模型p23估计的非对称畸变和剩余的残差如图3所示。标定图像右下角的相对较大的残差，似乎是因为非均匀光照导致的不精确定位。

In the second experiment, the calibration data was better, so the full model is likely to be more useful. This was verified by taking an additional image of the calibration object and solving the corresponding external camera parameters with given internal parameters. The RMS projection error for the additional image was 0.049 pixels for p23 and 0.071 for p9. This indicates that the full model described the true geometry of the camera better than the simpler model p9.

在第二个实验中，标定数据更好一些，所以完整模型很可能会更有用。我们多取了一幅标定目标的图像，用给定的内部参数来求解对应的外部参数，对其进行了验证。额外图像的RMS投影误差，对p23模型是0。049像素，对p9是0.071。这说明，完整模型描述的相机的几何，比更简单的p9模型要更好。

Finally, we estimated the backward model error for p23, caused by the first order approximation of the asymmetric distortion function (see Section 2.3). This was done by back-projecting each pixel and then reprojecting the rays. The maximum displacement in the reprojection was 2.1⋅10^-5 pixels for the first camera and 4.6⋅10^-4 pixels for the second. Both values are very small so it is justified to ignore the backward model error in practice.

最后，我们估计了p23模型的反向模型误差，这是由非对称畸变函数的一阶近似导致的（见2.3节）。这是由反向投影每个像素，然后重新投影射线得到的。重投影的最大偏移，对第一个相机为2.1⋅10^-5像素，对第二个相机为4.6⋅10^-4像素。两个值都非常小，所以证明了，在实践中可以忽略反向模型误差。

### 5.2. Fish-Eye Lens Cameras

The first experimented fish-eye lens was an equidistance lens with the nominal focal length of 1.178 mm and it was attached to a Watec 221S CCD color camera. The calibration object was a 2x3 m^2 plane containing white circles with the radius of 60 mm on the black background. The calibration images were digitized from an analog video signal to 8-bit monochrome images, whose size was 640 by 480 pixels.

第一个实验的鱼眼镜头是一个等距镜头，标称焦距为1.178 mm，连接到了一个Watec 221S CCD彩色相机上。标定目标是一个2x3 m^2的平面，包含黑色背景的白色圆，半径60mm。标定图像用模拟视频信号进行数字化，得到8-bit黑白图像，其大小为640x480像素。

The calibration of a fish-eye lens can be performed even from a single image of the planar object as Fig. 4 illustrates. In that example, we used the model p6 and 60 control points. However, for the most accurate results, the whole field of view should be covered with a large number of measurements. Therefore, we experimented our method with 12 views and 680 points in total; the results are in Table 3. The extended model p23 had the smallest residual error but the radially symmetric model p9 gave almost as good results. Nevertheless, there should be no risk of overfitting because the number of measurements is large. The estimated asymmetric distortion and the residuals are displayed in Fig. 5.

鱼眼镜头的标定，只用一幅平面目标的图像就可以进行，如图4所示。在这个例子中，我们使用了模型p6，和60个控制点。但是，要得到最精确的结果，整个FOV需要覆盖大量测量。因此，在我们的实验中，我们用了12个视角，总计680个点；结果如表3所示。拓展模型p23残差最小，但径向对称模型p9几乎给出了同样好的结果。尽管如此，并没有过拟合的风险，因为测量的数量很大。估计的非对称畸变和残差如图5所示。

The second fish-eye lens was ORIFL190-3 lens manufactured by Omnitech Robotics. This lens has a 190 degree field of view and it clearly deviates from the exact equidistance projection model. The lens was attached to a Point Grey Dragonfly digital color camera having 1024x768 pixels; the calibration object was the same as in Section 5.1. The obtained RMS residual errors for a set-up of 12 views and 1,780 control points are shown in Table 3. Again, the full model had the best performance and this was verified with an additional calibration image. The RMS projection error for the additional image, after fitting the external camera parameters, was 0.13 pixels for p23 and 0.16 pixels for p9. The backward model error for p23 was evaluated at each pixel within the circular images. The maximum displacement was 9.7⋅10^-6 pixels for the first camera and 3.4⋅10^-3 pixels for the second. Again, it is justified to ignore such small errors in practice.

第二个鱼眼镜头是ORIFL190-3镜头，由Omnitech Robotics制造。镜头的FOV为190度，与精确的等距投影模型有明显偏差。镜头连接到一个Point Grey Dragonfly数字彩色相机，分辨率为1024x768；标定目标与5.1一样。采用12个视角，1780个控制点，得到的RMS残差误差如表3所示。完整模型再次有了最好的性能，这由额外的标定图像进行了验证。额外图像的RMS投影误差，在拟合了外部相机参数后，对p23为0.13像素，对p9为0.16像素。对圆形图像中的每个像素计算了p23的反向模型误差。对第一个相机，最大偏差为9.7⋅10^-6像素，对第二个为3.4⋅10^-3像素。再一次，我们证明了在实践中可以忽略这样小的误差。

### 5.3. Synthetic Data

In order to evaluate the robustness of the proposed calibration method we did experiments also with synthetic data. The ground truth values for the camera parameters were obtained from the real fish-eye lens experiment that was illustrated in Fig. 5. So, we used the full camera model and we had 680 circular control points in 12 synthetic calibration images, where the gray level values of control points and background were 180 and 5, respectively. In order to make the synthetic images to better correspond real images, they were blurred by a Gaussian pdf (σ = 1 pixel) and quantized to the 256 gray levels.

为评估提出的标定方法的稳健性，我们还用合成数据进行了试验。相机参数的真值是从真实鱼眼镜头试验中获得的，如图5所示。所以，我们使用完整相机模型，在12个合成的标定图像中有680个圆形控制点，控制点和背景的灰度值分别为180和5。为使合成图像更好的对应真实图像，用高斯pdf (σ = 1 pixel)进行了模糊，量化成了256个灰度级。

First, we estimated the significance of the centroid correction proposed in Section 4.2. In the above setting the RMS distance between the centroids of the projected circles and the projected centers of the original circles was 0.45 pixels. It is a significantly larger value than the RMS residual errors reported in the real experiment (Table 3). This indicates that, without the centroid correction, the estimated camera parameters would have been biased and it is likely that the residual error would have been larger.

首先，我们估计了4.2节中估计的重心修正的显著性。在上面的设置中，投影的圆形的重心与原始圆形的投影中心RMS距离为0.45像素。这比真实实验中得到的RMS残差误差明显要大。这说明，没有重心修正，估计的相机参数会有偏差，残差误差很可能会更大。

Second, we estimated the effect of noise to the calibration by adding Gaussian noise to the synthetic images and performing 10 calibration trials at each noise level. The standard deviation of the noise varied between 0 and 15 pixels. The control points were localized from the noisy images by first thresholding them using a fixed threshold. Then, the centroid of each control point was measured by computing the gray-level-weighted center-of-mass.

第二，我们估计了噪声对标定的效果，对合成图像加入了高斯噪声，对每个噪声级别进行了10次标定。噪声的标准偏差从0到15像素变化。控制点从含噪图像中的定位，首先用固定的阈值进行处理，然后，每个控制点的重心通过计算灰度加权的重心来得到。

The simulation results are shown in Fig. 6, where we have plotted the average RMS measurement, RMS residual and RMS estimation errors. There is small error also at the zero noise level because of the discrete pixel representation and gray level quantization. The fact that the RMS errors approximately satisfy the Pythagorean equality indicates that the calibration algorithm has converged to the true global minimum [18]. Moreover, the low values of the RMS estimation error indicate that the estimated camera model is close to the true one even at large noise levels.

仿真结果如图6所示，其中我们画了平均RMS度量，RMS残差和RMS估计误差。在0噪声水平上，因为离散像素表示和灰度级量化，误差很小。RMS误差近似的满足Pythagorean等式，说明标定算法收敛到了真正的全局最小值。而且，RMS估计误差的值很小，说明估计的相机模型，即使在很大的噪声水平上，与真实的也是很接近的。

## 6. Conclusion

We have proposed a novel camera calibration method for fish-eye lens cameras that is based on viewing a planar calibration pattern. The experiments verify that the method is easy-to-use and provides a relatively high level of accuracy with circular control points. The proposed camera model is generic, easily expandable, and suitable also for conventional cameras with narrow or wide-angle lenses. The achieved level of accuracy for fish-eye lenses is better than it has been reported with other approaches and, for narrow-angle lenses, it is comparable to the results in [3]. This is promising considering especially the aim of using fish-eye lenses in measurement purposes. The calibration method is implemented as a calibration toolbox on Matlab and is available on the authors’ Web page.

我们提出了鱼眼镜头相机的一种新的相机标定方法，基于观察平面标定模式。实验验证了，该方法很容易使用，用圆形控制点，给出了相对很高的准确率。提出的相机模型是通用的，很容易扩展，适用于传统的窄视角或广角相机。对鱼眼镜头获得的精确度水平，比其他方法给出的结果要好，对于窄角相机，与[3]中的结构是类似的。考虑到这是给鱼眼镜头设计的算法，所以该结果还是非常不错的。标定方法进行了实现，做成了Matlab的标定工具箱，在作者的主页上可用。