# A Flexible New Technique for Camera Calibration

Zhengyou Zhang, Senior Member, IEEE

## 0. Abstract

We propose a flexible new technique to easily calibrate a camera. It only requires the camera to observe a planar pattern shown at a few (at least two) different orientations. Either the camera or the planar pattern can be freely moved. The motion need not be known. Radial lens distortion is modeled. The proposed procedure consists of a closed-form solution, followed by a nonlinear refinement based on the maximum likelihood criterion. Both computer simulation and real data have been used to test the proposed technique and very good results have been obtained. Compared with classical techniques which use expensive equipment such as two or three orthogonal planes, the proposed techniques is easy to use and flexible. It advances 3D computer vision one more step from laboratory environments to real world use. The corresponding software is available from the author's Web page.

我们提出了一种灵活的新技术，可以很容易对相机进行标定。这种技术只需要相机在几个（至少两个）不同的方向观察一个平面模式。相机或平面模式都可以自由移动。运动不需要知道。径向镜头畸变进行了建模。提出的过程由一个闭合形式解，和一个基于最大似然准则的非线性优化。计算机模拟和真实数据都用于测试提出的技术，得到了很好的结果。与经典的适用昂贵的设备的技术相比，比如两个或三个正交的平面，我们提出的技术很容易使用，也很灵活。这使得3D计算机视觉技术更进一步，从实验室环境，到了真实世界的使用。对应的软件在作者的网页上可用。

**Index Terms** - Camera calibration, calibration from planes, 2D pattern, flexible plane-based calibration, absolute conic, projective mapping, lens distortion, closed-form solution, maximum likelihood estimation, flexible setup.

## 1. Motivation

Camera calibration is a necessary step in 3D computer vision in order to extract metric information from 2D images. Much work has been done, starting in the photogrammetry community (see [2], [4] to cite a few), and more recently in computer vision ([9], [8], [23], [7], [25], [24], [16], [6] to cite a few). We can classify those techniques roughly into two categories: photogrammetric calibration and self-calibration.

相机标定是3D计算机视觉的一个必要步骤，可以从2D图像中提取测量信息。已经做了很多工作，最初是从摄影测量中开始的，最近到了计算机视觉领域中。我们可以将这些技术大致分为两类：摄影测量标定和自标定。

- **Three-dimensional reference object-based calibration**. Camera calibration is performed by observing a calibration object whose geometry in 3D space is known with very good precision. Calibration can be done very efficiently [5]. The calibration object usually consists of two or three planes orthogonal to each other. Sometimes a plane undergoing a precisely known translation is also used [23]. These approaches require an expensive calibration apparatus, and an elaborate setup.

基于三维参考目标的标定。相机标定的进行，是通过观察一个标定目标，其在3D空间中的几何是已知的，而且精度非常好。标定可以很高效的进行[5]。标定目标通常包含两到三个平面，互相垂直。有时候还要将一个平面进行精确的平移[23]。这些方法需要比较昂贵的标定设备，以及复杂的设置。

- **Self-calibration**. Techniques in this category do not use any calibration object. Just by moving a camera in a static scene, the rigidity of the scene provides in general two constraints [16], [14] on the cameras’ internal parameters from one camera displacement by using image information alone. Therefore, if images are taken by the same camera with fixed internal parameters, correspondences between three images are sufficient to recover both the internal and external parameters which allow us to reconstruct 3D structure up to a similarity [15], [12]. While this approach is very flexible, it is not yet mature [1]. Because there are many parameters to estimate, we cannot always obtain reliable results.

自标定。这个类别中的技术，不使用任何标定目标。只要将相机在静态场景中移动，只使用图像信息，场景的刚性一般会从一个相机平移对相机的内部参数给出两个约束。因此，如果图像是用同样的相机拍摄的，内部参数固定，三幅图像的对应性，就足以恢复相机的内部和外部参数，这使我们重建3D结构，只差一个相似变换。这种方法是很灵活的，但并不成熟。因为有很多参数需要进行估计，我们不能一直得到可靠的结果。

Other techniques exist: vanishing points for orthogonal directions [3], [13], and calibration from pure rotation [11], [20]. 还存在其他的技术：正交方向的消失点[3,13]，以及从纯旋转进行标定[11,20]。

Our current research is focused on a desktop vision system (DVS) since the potential for using DVSs is large. Cameras are becoming inexpensive and ubiquitous. A DVS aims at the general public who are not experts in computer vision. A typical computer user will perform vision tasks only from time to time, so they will not be willing to invest money for expensive equipment. Therefore, flexibility, robustness, and low cost are important. The camera calibration technique described in this paper was developed with these considerations in mind.

我们目前的研究关注的是桌面视觉系统(DVS)，因为使用DVS的潜力是很巨大的。相机正变得没那么昂贵，无处不在。DVS的目标，是并不是计算机视觉专家的一般公众。一个典型的计算机用户会时不时的进行视觉任务，所以他们不会愿意投资昂贵的设备。因此，灵活性，稳健性和低代价是很重要的。本文描述的相机标定技术是考虑了这些要素进行开发的。

The proposed technique only requires the camera to observe a planar pattern shown at a few (at least two) different orientations. The pattern can be printed on a laser printer and attached to a “reasonable” planar surface (e.g., a hard book cover). Either the camera or the planar pattern can be moved by hand. The motion need not be known. The proposed approach, which uses 2D metric information, lies between the photogrammetric calibration, which uses explicit 3D model, and self-calibration, which uses motion rigidity or equivalently implicit 3D information. Both computer simulation and real data have been used to test the proposed technique and very good results have been obtained. Compared with classical techniques, the proposed technique is considerably more flexible: Anyone can make a calibration pattern by him/herself and the setup is very easy. Compared with self-calibration, it gains a considerable degree of robustness. We believe the new technique advances 3D computer vision one step from laboratory environments to the real world.

提出的技术只需要相机观察几个不同方向（至少两个）的平面模式。这个模式可以用激光打印机打印出来，附到一个合理的平面表面上（比如，一本硬皮书）。相机或平面模式都可以进行手动移动。运动不需要是已知的。提出的方法使用了2D测量信息，介于摄影标定和自标定之间，它们分别使用了显式的3D模型，和运动的刚性，即等价的隐式3D信息。我们使用了计算机模拟和真实数据对提出的技术进行了测试，得到了非常好的结果。与经典技术相比，提出的技术更加灵活：任何人都可以自己制作一个标定模式，设置也非常容易。与自标定相比，该方法的稳健程度很高。我们相信，该新技术将3D计算机视觉又发展了一步，从实验室环境到了真实世界。

Note that Triggs [22] recently developed a self-calibration technique from at least five views of a planar scene. His technique is more flexible than ours, but has difficulty to initialize. Liebowitz and Zisserman [13] described a technique of metric rectification for perspective images of planes using metric information, such as a known angle, two equal though unknown angles, and a known length ratio. They also mentioned that calibration of the internal camera parameters is possible provided at least three such rectified planes, although no experimental results were shown.

注意，Triggs[22]最近提出了一种自标定技术，需要用一个平面场景的至少5个视角。他的技术比我们的更加灵活，但初始化非常困难。Liebowitz和Zisserman[13]描述了一种技术，使用测量信息对平面的透视图像进行测量修正，使用的信息可以是已知的角度，两个相等但未知的角度，一个已知的长度比。他们还提到了，给定至少3个这样的修正平面，就可以进行内部相机参数的标定，但是没有给出任何试验结果。

During the revision of this paper, we notice the publication of an independent but similar work by Sturm and Maybank [21]. They use a simplified camera model (image axes are orthogonal to each other) and have studied the degenerate configurations exhaustively for the case of one and two planes, which are very important in practice if only one or two views are used for camera calibration.

在本文的修改过程中，我们注意到发表了一个独立但是类似的工作，即Sturm和Maybank[21]。他们使用了一个简化的相机模型（图像的轴之间互相垂直），对一个平面和两个平面的情况详细研究了降质配置，如果只用一个视角或两个视角进行相机标定，这在实践中非常重要。

The paper is organized as follows: Section 2 describes the basic constraints from observing a single plane. Section 3 describes the calibration procedure. We start with a closed-form solution, followed by nonlinear optimization. Radial lens distortion is also modeled. Section 4 provides the experimental results. Both computer simulation and real data are used to validate the proposed technique. In the Appendix, we provides a number of details, including the techniques for estimating the homography between the model plane and its image.

本文组织如下：第2部分描述观察单个平面的基本约束。第3部分描述了标定过程。我们从一个闭合形式解开始，然后进行非线性优化。镜头径向畸变也进行了建模。第4部分给出了试验结果。计算机模拟和真实数据都用于验证提出的技术。在附录中，我们给出了几个细节，包括用于估计建模平面及其图像的单应矩阵的技术。

## 2. Basic Equations

We examine the constraints on the camera’s intrinsic parameters provided by observing a single plane. We start with the notation used in this paper. 我们研究了观察一个平面所能给出的对相机内部参数的约束。我们从本文所使用的表示开始。

### 2.1. Notation

A 2D point is denoted by $m = [u, v]^T$. A 3D point is denoted by $M = [X, Y, Z]^T$. We use $\tilde x$ to denote the augmented vector by adding 1 as the last element: $\tilde m = [u, v, 1]^T$ and $\tilde M = [X, Y, Z, 1]^T$. A camera is modeled by the usual pinhole: The relationship between a 3D point M and its image projection m is given by

一个2D点表示为$m = [u, v]^T$，3D点表示为$M = [X, Y, Z]^T$。我们使用$\tilde x$来表示扩增的向量，即将1加入作为最后一个元素：$\tilde m = [u, v, 1]^T$，$\tilde M = [X, Y, Z, 1]^T$。一个相机可以使用通常的针孔进行建模：一个3D点M及其图像投影m的关系，可以表示为

$$s\tilde m = A[R, t]\tilde M, A = \left[ \begin{matrix} α && γ && u_0 \\ 0 && β && v_0 \\ 0 && 0 && 1 \end{matrix} \right]$$(1)

where s is an arbitrary scale factor, (R, t), called the extrinsic parameters is the rotation and translation which relates the world coordinate system to the camera coordinate system, and A is called the camera intrinsic matrix, with (u_0, v_0) the coordinates of the principal point, α and β the scale factors in image u and v axes, and the parameter γ describing the skew of the two image axes. We use the abbreviation $A^{-T}$ for $(A^{-1})^T$ or $(A^T)^{-1}$.

其中s是一个任意大小的尺度因子，(R, t)称为外部参数，是旋转和平移，将世界坐标系与相机坐标系关联起来，A称为相机内部参数，(u_0, v_0)是主点的坐标，α和β是图像u和v轴的缩放因子，参数γ描述了两个图像轴的倾斜。对$(A^{-1})^T$或$(A^T)^{-1}$，我们使用简写$A^{-T}$。

### 2.2 Homography between the Model Plan and Its Image

Without loss of generality, we assume the model plane is on Z = 0 of the world coordinate system. Let’s denote the i-th column of the rotation matrix R by r_i. From (1), we have

不失一般性，我们假设模型平面是在Z=0的世界坐标系中。我们将旋转矩阵R的第i列表示为r_i。从(1)式中，我们有

$$s\left[ \begin{matrix} u \\ v \\ 1 \end{matrix} \right] = A [r_1, r_2, r_3, t] \left[ \begin{matrix} X \\ Y \\ 0 \\ 1 \end{matrix} \right] = A [r_1, r_2, t] \left[ \begin{matrix} X \\ Y \\ 1 \end{matrix} \right]$$

By abuse of notation, we still use M to denote a point on the model plane, but $M = [X, Y]^T$ since Z is always equal to zero. In turn, $\tilde M = [X, Y, 1]^T$. Therefore, a model point M and its image m is related by a homography H:

我们过度使用一下表示，仍然用M表示在模型平面上的一个点，但$M = [X, Y]^T$，因为Z永远是等于0的。所以，$\tilde M = [X, Y, 1]^T$。因此，模型点M及其图像m是通过单应矩阵H关联到一起的：

$$s\tilde m = H\tilde M, H = [r_1, r_2, t]$$(2)

As is clear, the 3 x 3 matrix H is defined up to a scale factor. 很清楚，3x3矩阵H除了一个尺度因子，是很明确的。

### 2.3. Constraints on the Intrinsic Parameters

Given an image of the model plane, an homography can be estimated (see the Appendix). Let's denote it by H = [h1, h2, h3]. From (2), we have

给定模型平面上的一幅图像，可以估计一个单映矩阵。我们将其表示为H = [h1, h2, h3]，从(2)中，我们有

$$[h_1, h_2, h_3] = λ A [r_1, r_2, t]$$

where λ is an arbitrary scalar. Using the knowledge that r1 and r2 are orthonormal, we have 其中λ是任意标量。r1和r2是正交的，使用这个知识，我们有

$$h_1^T A^{-T} A^{-1} h_2 = 0$$(3)
$$h_1^T A^{-T} A^{-1} h_1 = h_2^T A^{-T} A^{-1} h_2$$(4)

These are the two basic constraints on the intrinsic parameters, given one homography. Because a homography has 8 degrees of freedom and there are six extrinsic parameters (three for rotation and three for translation), we can only obtain two constraints on the intrinsic parameters. Note that $A^{-T} A^{-1}$ actually describes the image of the absolute conic [15]. In the next section, we will give a geometric interpretation.

在给定一个单映矩阵的情况下，这是对内部参数的两个基本约束。因为单映矩阵有8个自由度，而有6个外部参数（3个旋转，3个平移），我们对内部参数只能得到2个约束。注意$A^{-T} A^{-1}$实际上描述了绝对二次曲线的图像。在下一节中，我们会给出一个几何解释。

### 2.4. Geometric Interpretation

We are now relating (3) and (4) to the absolute conic [16], [15]. 我们现在将(3)(4)与绝对二次曲线关联起来。

It is not difficult to verify that the model plane, under our convention, is described in the camera coordinate system by the following equation:

在我们的表示下，模型平面是在相机坐标系系统中，由下式进行描述，这不难得到验证：

$$\left[ \begin{matrix} r_3 \\ r_3^T t \end{matrix} \right] ^T \left[ \begin{matrix} x \\ y \\ z \\ w \end{matrix} \right] = 0$$

where w = 0 for points at infinity and w = 1 otherwise. This plane intersects the plane at infinity at a line and we can easily see that

w = 0是代表无限远的点，w = 1是其他情况。这个平面与在无限远处的平面相交于一条线，我们可以很容易看到

$[r_1, 0]^T$ and $[r_2, 0]^T$ are two particular points on that line. Any point on it is a linear combination of these two points, i.e.,

$[r_1, 0]^T$ 和 $[r_2, 0]^T$是这条线上两个特殊的点。在其上的任意点，是这两个点的线性组合，即

$$x_​∞ = a[r_1, 0]^T + b[r_2, 0]^T = [ar_1+br_2, 0]^T$$

Now, let’s compute the intersection of the above line with the absolute conic. By definition, the point $x​_∞$, known as the circular point [18], satisfies: $x^T_​∞ x_​∞ = 0$, i.e., $(ar_1 + br_2)^T (ar_1 + br_2) = 0$, or $a^2 + b^2 = 0$. The solution is $b = ±ai$, where i^2 = -1. That is, the two intersection points are

现在，让我们计算上面这条线与绝对二次曲线的交点。从定义，点$x​_∞$，被称为circular点，满足$x^T_​∞ x_​∞ = 0$，即$(ar_1 + br_2)^T (ar_1 + br_2) = 0$，或$a^2 + b^2 = 0$。其解为$b = ±ai$，其中i^2 = -1。即，两个相交的点为

$$x_​∞ = a[r_1±ir_2, 0]^T$$

The significance of this pair of complex conjugate points lies in the fact that they are invariant to Euclidean transformations. Their projection in the image plane is given, up to a scale factor, by

这对复共轭点的重要性在于，它们对欧式变换是不变的。它们在图像的平面由下式给出，尺度因子是未定的

$$\tilde m_∞ = A(r_1±ir_2) = h_1±ih_2$$

Point $\tilde m_∞$ is on the image of the absolute conic, described by $A^{-T} A^{-1}$ [15]. This gives

$$(h_1±ih_2)^T A^{-T} A^{-1} (h_1±ih_2) = 0$$

Requiring that both real and imaginary parts be zero yields (3) and (4).

## 3. Solving Camear Calibration

This section provides the details how to effectively solve the camera calibration problem. We start with an analytical solution, followed by a nonlinear optimization technique based on the maximum-likelihood criterion. Finally, we take into account lens distortion, giving both analytical and nonlinear solutions.

本节给出了怎样高效的求解相机标定问题的细节。我们从解析解开始，然后是基于最大似然准则的非线性优化技术。最后，我们考虑了相机畸变，给出了解析解和非线性解。

### 3.1. Closed-Form Solution

Let

$$B = A^{-T} A^{-1} = \left[ \begin{matrix} B_{11} && B_{12} && B_{13} \\ B_{21} && B_{22} && B_{23} \\ B_{31} && B_{32} && B_{33} \end{matrix} \right] = \left[ \begin{matrix} 1/α^2 && -γ/α^2β && (v_0γ-u_0β)/(α^2β) \\ -γ/α^2β && γ^2/α^2β^2 + 1/β^2 && -(γ(v_0γ - u_0β))/(α^2β^2) - v_0/β^2 \\ (v_0γ-u_0β)/(α^2β) && -(γ(v_0γ - u_0β))/(α^2β^2) - v_0/β^2 && ((v_0γ - u_0β)^2)/(α^2β^2) + v_0^2/β^2 + 1 \end{matrix} \right]$$(5)

Note that B is symmetric, defined by a 6D vector

$$b = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^T$$(6)

Let the i-th column vector of H be $hi = [h_{i1}, h_{i2}, h_{i3}]^T$. Then, we have

$$h_i^T B h_j = v_{ij}^T b$$(7)

with

$$v_{ij} = [h_{i1} h_{j1}, h_{i1} h_{j2} + h_{i2} h_{j1}, h_{i2} h_{j2}, h_{i3} h_{j1} + h_{i1} h_{j3}, h_{i3} h_{j2} + h_{i2} h_{j3}, h_{i3} h_{j3}]]^T$$

Therefore, the two fundamental constraints (3) and (4), from a given homography, can be rewritten as two homogeneous equations in b:

因此，从一个给定的单映矩阵中，两个基础的约束(3)(4)可以重写为两个对b的一致的等式：

$$[v_{12}^T, (v_{11} - v_{22})^T]^T b = 0$$(8)

If n images of the model plane are observed, by stacking n such equations as (8), we have

如果观察到了模型平面的n幅图像，堆叠(8)这样的等式n个，我们有

$$Vb = 0$$(9)

where V is a 2n x 6 matrix. If n ≥ 3, we will have in general a unique solution b defined up to a scale factor. If n = 2, we can impose the skewless constraint γ=0, i.e., [0, 1, 0, 0, 0, 0]b = 0, which is added as an additional equation to (9). (If n = 1, we can only solve two camera intrinsic parameters, e.g., α and β, assuming u_0 and v_0 are known (e.g., at the image center) and γ=0, and that is indeed what we did in [19] for head pose determination based on the fact that eyes and mouth are reasonably coplanar. In fact, Tsai [23] already mentions that focal length from one plane is possible, but incorrectly says that aspect ratio is not.) The solution to (9) is well-known as the eigenvector of $V^T V$ associated with the smallest eigenvalue (equivalently, the right singular vector of V associated with the smallest singular value).

其中V是一个2n x 6的矩阵。如果n ≥ 3，我们一般就有了b的唯一解，只有一个尺度因子未定。如果n=2，我们可以加上无倾斜的约束γ=0，即[0, 1, 0, 0, 0, 0]b = 0，这是可以加入(9)的额外等式。如果n=1，我们就只能求解两个相机内参，如，α和β，假设u_0和v_0是已知的（如，在图像中心），和γ=0，我们在[19]中就是这样的，基于眼睛和嘴巴是基本上共面的事实，进行头部姿态确定。实际上，[23]已经提到了，从一个平面估计焦距是可能的，但也说了纵横比是不可能的，这是不正确的。(9)式的解是$V^T V$最小特征值的特征向量（等价的，V最小的奇异值的右奇异向量）。

Once b is estimated, we can compute all camera intrinsic parameters as follows. The matrix B, as described in Section 3.1, is estimated up to a scale factor, i.e., $B = λA^{-T}A$ with λ an arbitrary scale. Without difficulty, we can uniquely extract the intrinsic parameters from matrix B.

一旦估计出了b，我们可以按照如下方式计算所有相机内参。3.1节中提到的矩阵B，可以估计到取决于一个尺度因子，即$B = λA^{-T}A$，λ是一个任意的尺度。我们可以从矩阵B中唯一的提取出相机的内参。

$$v_0 = (B_{12}B_{13} - B_{11}B_{23})/(B_{11}B_{22}-B_{12}^2)$$
$$λ = B_{33} - [B_{13}^2 + v_0 (B_{12}B_{13} - B_{11}B_{23})]/B_{11}$$
$$α = \sqrt{λ/B_{11}}$$
$$β = \sqrt{λB_{11}/(B_{11}B_{22}-B_{12}^2)}$$
$$γ = -B_{12}α^2β/λ$$
$$u_0 = γv_0/α - B_{13}α^2/γ$$

Once A is known, the extrinsic parameters for each image is readily computed. From (2), we have

一旦A是已知的，每幅图像的外参也可以立即计算得到。从(2)中，我们有

$$r_1 = λA^{-1} h_1, r_2 = λA^{-1} h_2, r_3 = r_1 x r_2, t = λA^{-1} h_3$$

with $λ = 1/||A^{-1}h_1|| = 1/||A^{-1}h_2||$. Of course, because of noise in data, the so-computed matrix R = [r1, r2, r3] does not, in general, satisfy the properties of a rotation matrix. The best rotation matrix can then be obtained through for example singular value decomposition [10], [26].

当然，由于数据中有噪声，计算得到的矩阵R = [r1, r2, r3]一般来说并不满足旋转矩阵的性质。最好的旋转矩阵可以通过SVD来得到。

### 3.2 Maximum-Likelihood Estimation

The above solution is obtained through minimizing an algebraic distance which is not physically meaningful. We can refine it through maximum-likelihood inference.

上面的解是通过最小化代数距离得到的，在物理上没有意义。我们可以通过最大似然推理来进行提炼。

We are given n images of a model plane and there are m points on the model plane. Assume that the image points are corrupted by independent and identically distributed noise. The maximumlikelihood estimate can be obtained by minimizing the following functional:

给定模型平面的n幅图像，在模型平面上有m个点。假设图像点附有独立同分布的噪声。通过最小化如下泛函，可以得到最大似然估计：

$$\sum_{i=1}^n \sum_{j=1}^m ||m_{ij} - \hat m(A, R_i, t_i, M_j)||^2$$(10)

where $\hat m(A, R_i, t_i, M_j)$ is the projection of point M_j in image i, according to (2). A rotation R is parameterized by a vector of three parameters, denoted by r, which is parallel to the rotation axis and whose magnitude is equal to the rotation angle. R and r are related by the Rodrigues formula [5]. Minimizing (10) is a nonlinear minimization problem, which is solved with the Levenberg-Marquardt Algorithm as implemented in Minpack [17]. It requires an initial guess of A, {R_i, t_i | i = 1, ..., n} which can be obtained using the technique described in the previous section.

其中根据(2)，$\hat m(A, R_i, t_i, M_j)$是点M_j在图像i中的投影。旋转R的参数是三个参数的向量，表示为r，与旋转轴平行，其幅度等于旋转角。R和r的关联为Rodrigues公式。最小化(10)是一个非线性最小化问题，可以用Minpack中的Levenberg-Marquardt算法求解。这需要对A, {R_i, t_i | i = 1, ..., n}的初始估计，可以使用前一节中的技术得到。

Desktop cameras usually have visible lens distortion, especially the radial components. We have included these while minimizing (10). Refer to the technical report, [26], for more details.

桌面的相机通常都有可见的镜头畸变，尤其是径向的部分。我们在最小化(10)的时候已经包括了这些。参考科技报告[26]的细节。

### 3.3. Summary

The recommended calibration procedure is as follows: 推荐的标定流程如下：

1. Print a pattern and attach it to a planar surface. 打印一个模式，将其附着到平面表面上。
2. Take a few images of the model plane under different orientations by moving either the plane or the camera. 通过移动平面或相机，在不同的方向拍摄模型平面的几幅图像。
3. Detect the feature points in the images. 检测图像中的特征点。
4. Estimate the five intrinsic parameters and all the extrinsic parameters using the closed-form solution, as described in Section 3.1. 使用闭合形式解估计5个内参，和所有的外参，如3.1节所示。
5. Refine all parameters, including lens distortion parameters, by minimizing (10). 最小化(10)，精炼所有参数，包括径向畸变参数。

There is a degenerate configuration in my technique when planes are parallel to each other. Refer to the technical report, [26], for a more detailed description.

在我们的技术中，有一种降质配置，当所有平面都互相平行的时候。参考科技报告[26]的详细描述。

## 4. Experimental Results

The proposed algorithm has been tested on both computer simulated data and real data. The closed-form solution involves finding a singular value decomposition of a small 2n x 6 matrix, where n is the number of images. The nonlinear refinement within the Levenberg-Marquardt Algorithm takes 3 to 5 iterations to converge. Due to space limitation, we describe in this section one set of experiments with real data when the calibration pattern is at different distances from the camera. The reader is referred to [26] for more experimental results with both computer simulated and real data, and to the following Web page: http://research.microsoft.com/~zhang/Calib/ for some experimental data and the software.

提出的算法在计算机仿真数据和真实数据上都进行了测试。闭合形式解涉及到对小型2n x 6的矩阵进行奇异值分解，其中n是图像的数量。采用Levenberg-Marquardt算法的非线性优化需要3到5次迭代才能收敛。由于空间限制，我们在本节中描述了一组真实数据的试验，标定模式与相机的距离在不同的距离上。可以参考[26]中更多的试验结果，计算机模拟和真实数据都有。

The example is shown in Fig. 1. The camera to be calibrated is an off-the-shelf PULNiX CCD camera with 6 mm lens. The image resolution is 640 x 480. As can be seen in Fig. 1, the model plane contains 9 x 9 squares with nine special dots which are used to identify automatically the correspondence between reference points on the model plane and square corners in images. It was printed on a A4 paper with a 600 DPI laser printer and attached to a cardboard.

图1是一个例子。待标定的相机是一个开箱即用的PULNiX CCD相机，镜头6mm。图像分辨率为640x480。如图1所示，模型平面包含9x9个方形，还有9个特殊的点，用于自动识别模型平面中的参考点和图像中的方形角点之间的对应性。这是打印在一张A4纸上的，用的是一个600 DPI的激光打印机，附着到了一个硬纸板上。

In total, 10 images of the plane were taken (six of them are shown in Fig. 1). Five of them (called Set A) were taken at close range, while the other five (called Set B) were taken at a larger distance. We applied our calibration algorithm to Set A, Set B, and also to the whole set (called Set A+B). The results are shown in Table 1. For intuitive understanding, we show the estimated angle between the image axes, θ, instead of the skew factor γ. We can see that the angle θ is very close to 90°, as expected with almost all modern CCD cameras. The cameras parameters were estimated consistently for all three sets of images, except the distortion parameters with Set B. The reason is that the calibration pattern only occupies the central part of the image in Set B, where lens distortion is not significant and therefore cannot be estimated reliably.

总计拍摄了10幅图像（图1展示了6幅）。近距离拍摄的5幅称为集合A，更大距离拍摄的5幅称为集合B。我们将标定算法应用于集合A，集合B和集合A+B。结果如表1所示。对了更直观的理解，我们给出了估计的图像轴之间的角度θ，而不是倾斜因子γ。我们可以看到角度θ是非常接近90度的，在所有的现代CCD相机中都是这样的。估计的相机参数，对三组图像集合都很一致，除了集合B的畸变参数。原因是，在集合B中，标定模式只占了图像的中央部分，这里镜头畸变并不明显，因此不能估计的很可靠。

## 5. Conclusion

In this paper, we have developed a flexible new technique to easily calibrate a camera. The technique only requires the camera to observe a planar pattern from a few different orientations. Although the minimum number of orientations is two if pixels are square, we recommend four or five different orientations for better quality. We can move either the camera or the planar pattern. The motion does not need to be known, but should not be a pure translation. When the number of orientations is only two, one should avoid positioning the planar pattern parallel to the image plane. The pattern could be anything, as long as we know the metric on the plane. For example, we can print a pattern with a laser printer and attach the paper to a reasonable planar surface such as a hard book cover. We can even use a book with known size because the four corners are enough to estimate the plane homographies.

本文中，我们提出了一种灵活的技术来很容易的标定相机。这种技术只需要相机从几个不同的方向拍摄一个平面模式。如果像素是方的，最小的方向数量是2，但是我们推荐4到5个不同的方向，可以得到更好的质量。我们可以移动相机，或平面模式。运动不需要已知，但不应当是纯平移。当方向数量只有2个时，应当避免将平面模式与图像平面平行。模式可以是任何事物，只要我们知道平面上的度量。比如，我们可以用激光打印机打印一个模式，将其附着在一个合理的平面表面上，比如硬皮书。我们甚至可以使用一本已知大小的书，因为四个角就足以估计平面单映矩阵。

Radial lens distortion is modeled. The proposed procedure consists of a closed-form solution, followed by a nonlinear refinement based on a maximum-likelihood criterion. Both computer simulation and real data have been used to test the proposed technique and very good results have been obtained. Compared with classical techniques which use expensive equipment such as two or three orthogonal planes, the proposed technique gains considerable flexibility.

径向镜头畸变进行了建模。提出的过程包含闭合形式解，然后是一个非线性优化，基于最大似然准则。计算机模拟和真实数据都用于测试提出的技术，得到了非常好的结果。经典的技术通常使用昂贵的设备，比如两到三个正交的平面，与其相比，提出的技术有着相当的灵活性。