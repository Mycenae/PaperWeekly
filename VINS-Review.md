# Visual-Inertial Navigation - A Concise Review

Guoquan (Paul) Huang, University of Delaware

## 0. Abstract

As inertial and visual sensors are becoming ubiquitous, visual-inertial navigation systems (VINS) have prevailed in a wide range of applications from mobile augmented reality to aerial navigation to autonomous driving, in part because of the complementary sensing capabilities and the decreasing costs and size of the sensors. In this paper, we survey thoroughly the research efforts taken in this field and strive to provide a concise but complete review of the related work – which is unfortunately missing in the literature while being greatly demanded by researchers and engineers – in the hope to accelerate the VINS research and beyond in our society as a whole.

由于惯性和视觉传感器正变得无处不在，视觉-惯性导航系统(VINS)已经在大量应用中流形起来，从移动AR，到空中导航，到无人驾驶，部分是因为其互补的感知能力，和传感器越来越低的价格和大小。本文中，我们彻底回顾了这个领域的研究努力，努力给出了相关工作的简介但完整的回顾，希望能加速VINS研究。

## 1. Introduction

Over the years, inertial navigation systems (INS) [1, 2] have been widely used for estimating the 6DOF poses (positions and orientations) of sensing platforms (e.g., autonomous vehicles), in particular, in GPS-denied environments such as underwater, indoor, in the urban canyon, and on other planets. Most INS rely on a 6-axis inertial measurement unit (IMU) that measures the local linear acceleration and angular velocity of the platform to which it is rigidly connected. With the recent advancements of hardware design and manufacturing, low-cost light-weight micro-electro-mechanical (MEMS) IMUs have become ubiquitous [3, 4, 5], which enables high-accuracy localization for, among others, mobile devices [6] and micro aerial vehicles (MAVs) [7, 8, 9, 10, 11], holding huge implications in a wide range of emerging applications from mobile augmented reality (AR) [12, 13] and virtual reality (VR) [14] to autonomous driving [15, 16]. Unfortunately, simple integration of high-rate IMU measurements that are corrupted by noise and bias, often results in pose estimates unreliable for long-term navigation. Although a high-end tactical-grade IMU exists, it remains prohibitively expensive for widespread deployments. On the other hand, a camera that is small, light-weight, and energy-efficient, provides rich information about the environment and serves as an ideal aiding source for INS, yielding visual-inertial navigation systems (VINS).

过去这些年，惯性导航系统(INS)已经广泛用于估计传感器平台的6 DOF姿态（位置和方向），如自动驾驶，特别是，在没有GPS的环境中，比如水下，室内，在城市峡谷中，或在其他星球上。多数INS依赖于6轴惯性测量单元IMU，测量局部的线性加速度，和刚性连接的平台的角速度。随着硬件设计和制造的最新进展，低价格轻量MEMS IMUs已经无处不在，这使得移动设备和微型空中飞行设备MAV等可以进行高精度定位，意味着大量应用可能实现，比如移动AR，VR到自动驾驶。不幸的是，高速率IMU的测量受到噪声和偏移的污染，简单的整合，会导致长期导航中的姿态估计不可靠。虽然存在高端的战术级IMU，但要进行大规模的部署，仍然是非常昂贵的。另一方面，现在的相机都是小型的，轻量的，能耗很低，可以感知环境的丰富信息，可以作为INS的理想辅助，得到视觉-惯性导航系统(VINS)。

While this problem is challenging because of the lack of global information to reduce the motion drift accumulated over time (which is even exacerbated if low-cost, low-quality sensors are used), VINS have attracted significant attentions over the last decade. To date, many VINS algorithms are available for both visual-inertial SLAM [17] and visual-inertial odometry (VIO) [18, 19], such as the extended Kalman filter (EKF) [17, 18, 20, 21, 22], the unscented Kalman filter (UKF) [23, 24, 25], the batch or incremental smoother [26, 27], and (window) optimization-based approaches [28, 29, 30, 31]. Among these, the EKF-based methods remain popular because of its efficiency. For example, as a state-of-the-art solution of VINS on mobile devices, Project Tango [32] (or ARCore [12]) appears to use an EKF to fuse the visual and inertial measurements for motion tracking. Nevertheless, recent advances of preintegration have also allowed for efficient inclusion of high-rate IMU measurements in graph optimization-based formulations [29, 30, 33, 34, 35].

这个问题是非常有挑战性的，因为缺少全局信息来减少随着时间累积的运动漂移（如果使用低价格，低质量传感器，那么这个问题还会被加剧），所以在过去十年中VINS吸引了大量的注意力。迄今为止，很多VINS算法已经可用于视觉-惯性SLAM和视觉-惯性里程计，比如扩展卡尔曼滤波EKF，无迹卡尔曼滤波UKF，the batch or incremental smoother，和基于优化的方法。在这些当中，基于EKF的方法很流形，因为效率很高。比如，Project Tango是目前最好的VINS在移动设备上的解决方案，就使用了EKF来融合视觉和惯性测量，来进行运动跟踪。尽管如此，预整合的最近进展，也可以将高速率IMU测量很高效的纳入到基于图优化的表达中。

As evident, VINS technologies are emerging, largely due to the demanding mobile perception/navigation applications, which has given rise to a rich body of literature in this area. However, to the best of out knowledge, there is no contemporary literature review of VINS, although there are recent surveys broadly about SLAM [16, 36] while not specializing on VINS. This has made difficult for researchers and engineers in both academia and industry, to effectively find and understand the most important related work to their interests, which we have experienced over the years when we are working on this problem. For this reason, we are striving to bridge this gap by: (i) offering a concise (due to space limitation) but complete review on VINS while focusing on the key aspects of state estimation, (ii) providing our understandings about the most important related work, and (iii) opening discussions about the challenges remaining to tackle. This is driven by the hope to (at least) help researchers/engineers track and understand the state-of-the-art VINS algorithms/systems, more efficiently and effectively, thus accelerating the VINS research and development in our society as a whole.

很明显，VINS技术的崛起，主要是因为移动感知/导航应用 要求很高，这带来了这个领域的文献的增长。但是，据我们所知，VINS尚没有综述，虽然最近又很多SLAM的综述，但并没有专注到VINS中。这使得研究者和工业界的工程师都觉得不便。为此，我们努力填补这一空白：(i)给出了VINS的简洁但完整的综述，主要关注状态估计这一方面，(ii)给出最重要的相关工作的理解，(iii)对需要解决的挑战进行讨论。这使得研究者/工程师可以跟踪和理解目前最好的VINS算法和系统，加速VINS的研究和开发。

## 2. Visual-Inertial Navigation

In this section, we provide some basic background of canonical VINS, by describing the IMU propagation and camera measurement models within the EKF framework.

本节中，我们给出经典VINS的一些基本背景，描述了EKF框架中的IMU传播和相机测量模型。

### 2.1. IMU Kinematic Model 运动模型

The EKF uses the IMU (gyroscope and accelerometer) measurements for state propagation, and the state vector consists of the IMU states $x_I$ and the feature position $^Gp_f$:

EKF使用IMU（陀螺仪和加速度计）的测量进行状态传播，状态向量包括IMU的状态$x_I$和特征位置$^Gp_f$：

$$x = [x_I^T, ^Gp_f^T]^T = [^I_G \bar q^T, b_g^T, ^Gv^T, b_a^T, ^Gp^T, ^Gp_f^T]^T$$(1)

where $^I_G \bar q$ is the unit quaternion that represents the rotation from the global frame of reference {G} to the IMU frame {I} (i.e., different parametrization of the rotation matrix $C(^I_G \bar q) =: ^I_G C$); $^G p$ and $^G v$ are the IMU position and velocity in the global frame; and bg and ba denote the gyroscope and accelerometer biases, respectively.

其中$^I_G \bar q$是单位四元数，表示从全局参考框架{G}到IMU框架{I}的旋转（即，旋转矩阵$C(^I_G \bar q) =: ^I_G C$的不同参数化）；$^G p$和$^G v$是全局框架中的IMU位置和速度；bg和ba分别表示陀螺仪和加速度计的偏差。

By noting that the feature is static (with trivial dynamics), as well as using the IMU motion dynamics [37], the continuous-time dynamics of the state (1) is given by:

特征是静态的（有很小的运动），我们使用IMU运动动力学，状态(1)的连续时间动力学由下式给出：

$$^I_G \dot{\bar q}(t) = \frac {1}{2} Ω(^Iω(t)) ^I_G \bar q(t), ^G \dot p(t) = ^G v(t), ^G \dot v(t) = ^G a(t)$$
$$\dot b_g(t) = n_{wg} (t), \dot b_a(t) = n_{wa}(t), ^G \dot p_f(t) = 0_{3×1}$$(2)

where $^I ω = [ω_1, ω_2, ω_3]^T$ is the rotational velocity of the IMU, expressed in {I}, $^Ga$ is the IMU acceleration in {G}, $n_{wg}$ and $n_{wa}$  are the white Gaussian noise processes that drive the IMU biases, and $Ω(ω) = \left[ \begin{matrix} -⌊ω×⌋ && ω \\ ω^T && 0 \end{matrix} \right]$ where ⌊ω×⌋  is the skew-symmetric matrix.

其中$^I ω = [ω_1, ω_2, ω_3]^T$是IMU的旋转速度，在{I}中表示，$^Ga$是{G}中的IMU加速度，$n_{wg}$和$n_{wa}$是高斯白噪声过程，驱动着IMU偏移，$Ω(ω) = \left[ \begin{matrix} -⌊ω×⌋ && ω \\ ω^T && 0 \end{matrix} \right]$，其中⌊ω×⌋是斜对称矩阵。

A typical IMU provides gyroscope and accelerometer measurements, $ω_m$ and $a_m$, both of which are expressed in the IMU local frame {I} and given by:

一个典型的IMU会给出陀螺仪和加速度计的测量$ω_m$和$a_m$，两者都是在局部框架{I}中表示的，由下式给出：

$$ω_m(t) = ^I ω(t) + b_g(t) + n_g(t)$$(3)
$$a_m(t) = C(^I_G \bar q(t)) (^G a(t) - ^Gg) + b_a(t) + n_a(t)$$(4)

where $^G g$ is the gravitational acceleration expressed in {G}, and ng and na are zero-mean, white Gaussian noise.

其中$^G g$是在{G}中表示的重力加速度，ng和na是零均值高斯白噪声。

Linearization of (2) at the current state estimate yields the continuous-time state-estimate propagation model [21]:

(2)在当前状态下的线性化，给出了连续时间状态估计的传播模型：

$$^I_G \dot{\hat{\bar q}}(t) = \frac {1}{2} Ω(^I\hat ω(t)) ^I_G \hat{\bar q}(t), ^G \dot {\hat p}(t) = ^G \hat v(t), ^G \dot {\hat v}(t) = ^G \hat a(t)$$
$$\dot {\hat b}_g(t) = 0_{3×1}, \dot {\hat b}_a(t) = 0_{3×1}, ^G \dot p_f(t) = 0_{3×1}$$(5)

where $\hat a = a_m - \hat b_a$ and $\hat ω = ω_m - \hat b_g$. The error state of dimension 18 × 1 is hence defined as follows [see (1)]:

$$\tilde x(t) = [^I \tilde θ^T(t), \tilde b^T_g(t), ^G \tilde v^T(t), \tilde b^T_a(t), ^G \tilde p^T(t), ^G \tilde p^T_f(t)]^T$$(6)

where we have employed the multiplicative error model for a quaternion [37]. That is, the error between the quaternion $\bar q$ and its estimate $\hat {\bar q}$ is the 3×1 angle-error vector, $^I \tilde θ$, implicitly defined by the error quaternion: $δ\bar q = \bar q ⊗ \hat {\bar q} ≃ \left[ \begin{matrix} ^I \tilde θ/2 \\ 1 \end{matrix} \right]$, where $δ\bar q$ describes the small rotation that causes the true and estimated attitude to coincide. The advantage of this parametrization permits a minimal representation, 3 × 3 covariance matrix E[$^I \tilde θ ^I \tilde θ^T$], for the attitude uncertainty.

这里我们对一个四元数采用了乘性误差模型。即，四元数$\bar q$及其估计$\hat {\bar q}$之间的误差是一个3×1的角度误差向量，$^I \tilde θ$，由误差四元素隐式的定义。其中$δ\bar q$描述的是小的旋转，导致真正的和估计的姿态要重合。这种参数化的优势，对姿态的不确定性，允许一个最小的表示，3 × 3的协方差矩阵E[$^I \tilde θ ^I \tilde θ^T$]。

Now the continuous-time error-state propagation is: 现在连续时间误差状态传播是：

$$\dot {\tilde x} (t) = F_c(t) \tilde x(t) + G_c(t) n(t)$$(7)

where $n = [n_g^T, n_{wg}^T, n_a^T, n_{wa}^T]^T$ is the system noise, $F_c$ is the continuous-time error-state transition matrix, and $G_c$ is the input noise matrix, which are given by (see [37]):

其中n是系统噪声，Fc是连续时间误差状态转移矩阵，Gc是输入噪声矩阵，由下式给出：

$$F_c = \left[ \begin{matrix} -⌊\hat ω×⌋ && -I_3 && 0_3 && 0_3 && 0_3 && 0_3 \\ 0_3 && 0_3 && 0_3 && 0_3 && 0_3 && 0_3 \\ -C^T(^I_G \hat{\bar q})⌊\hat a×⌋ && 0_3 && 0_3 && -C^T(^I_G \hat{\bar q}) && 0_3 && 0_3 \\ 0_3 && 0_3 && 0_3 && 0_3 && 0_3 && 0_3 \\ 0_3 && 0_3 && I_3 && 0_3 && 0_3 && 0_3 \\ 0_3 && 0_3 && 0_3 && 0_3 && 0_3 && 0_3 \end{matrix} \right]$$(8)

$$G_c = \left[ \begin{matrix} -I_3 && 0_3 && 0_3 && 0_3 \\ -0_3 && I_3 && 0_3 && 0_3 \\ -I_3 && 0_3 && -C^T(^I_G \hat{\bar q}) && 0_3 \\ -I_3 && 0_3 && 0_3 && I_3 \\ -I_3 && 0_3 && 0_3 && 0_3 \\ 0_3 && 0_3 && 0_3 && 0_3 \end{matrix} \right]$$(9)

The system noise is modelled as zero-mean white Gaussian process with autocorrelation $E[n(t)n(τ)^T] = Q_c δ(t-τ)$, which depends on the IMU noise characteristics.

系统噪声建模为零均值高斯白噪声，自相关为$E[n(t)n(τ)^T] = Q_c δ(t-τ)$，这依赖于IMU的噪声特性。

We have described the continuous-time propagation model using IMU measurements. However, in any practical EKF implementation, the discrete-time state-transition matrix, $Φ_k := Φ(t_{k+1}, t_k)$, is required in order to propagate the error covariance from time t_k to t_k+1. Typically it is found by solving the following matrix differential equation:

我们描述了使用IMU度量的连续时间传播模型。但是，在任意实际的EKF实现中，都需要离散时间状态转移矩阵$Φ_k := Φ(t_{k+1}, t_k)$，以将误差协方差从时间t_k传播到t_k+1。一般来说，通过求解下面的矩阵微分方程来得到：

$$\dot Φ(t_{k+1}, t_k) = F_c(t_{k+1}) Φ(t_{k+1}, t_k)$$(10)

with the initial condition $Φ(t_k, t_k) = I_{18}$. This can be solved either numerically [21, 37] or analytically [19, 38, 39, 40]. Once it is computed, the EKF propagates the covariance as [41]:

其初始条件为$Φ(t_k, t_k) = I_{18}$。这可以通过数值进行求解，或得到解析解。一旦计算完成，EKF将协方差传播为

$$P_{k+1 | k} = Φ_k P_{k|k} Φ^T_k + Q_{d,k}$$(11)

where $Q_{d,k}$ is the discrete-time system noise covariance matrix computed as follows:

$$Q_{d,k} = \int_{t_k}^{t_{k+1}} Φ(t_{k+1}, τ) G_c(τ) Q_c G^T_c (τ) Φ^T (t_{k+1}, τ)dτ$$

### 2.2. Camera Measurement Model

The camera observes visual corner features, which are used to concurrently estimate the ego-motion of the sensing platform. Assuming a calibrated perspective camera, the measurement of the feature at time-step k is the perspective projection of the 3D point, $^{C_k}p_f$, expressed in the current camera frame {Ck}, onto the image plane, i.e.,

相机观察视觉尚的角点特征，用于并发的估计传感平台的自我运动。假设相机的观测角度已经标定了，特征在时间步骤k时的测量是3D点$^{C_k}p_f$透视投影到图像平面

$$z_k = \frac {1}{z_k} \left[ \begin{matrix} x_k \\ y_k \end{matrix} \right] + n_{f_k}$$(12)
$$\left[ \begin{matrix} x_k \\ y_k \\ z_k \end{matrix} \right] = ^{C_k}p_f = C(^C_I \bar q) C(^I_G \bar q_k) (^G p_f - ^G p_k) + ^C p_I$$(13)

where $n_{f_k}$ is the zero-mean, white Gaussian measurement noise with covariance R_k. In (13), {$^C_I \bar q, ^Cp_I$} is the rotation and translation between the camera and the IMU. This transformation can be obtained, for example, by performing camera-IMU extrinsic calibration offline [42]. However, in practice when the perfect calibration is unavailable, it is beneficial to VINS consistency to include these calibration parameters in the state vector and concurrently estimate them along with the IMU/camera poses [40].

其中$n_{f_k}$是零均值，高斯白噪声，协方差矩阵为R_k。在(13)中，{$^C_I \bar q, ^Cp_I$}是相机和IMU之间的旋转和平移。这个变换可以通过进行相机-IMU外部标定[42]得到。但是，在实际中，完美的标定是不存在的，将这些标定参数包括在状态向量中，与IMU/相机姿态进行同时估计，这样对VINS的一致性是有好处的。

For the use of EKF, linearization of (12) yields the following measurement residual [see (6)]:

$$\tilde z_k = H_k \tilde x_{k|k+1} + n_{f_k} = H_{I_k} \tilde x_{I_{k|k+1}} + H_{f_k} \space ^G\tilde p_{f_{k|k+1}} + n_{f_k}$$(14)

where the measurement Jacobian Hk is computed as:

$$H_k = [H_{I_k}, H_{f_k}] = H_{proj} C(^C_I \bar q) [H_{θ_k}, 0_{3×9}, H_{p_k}, C(^I_G \hat {\bar q}_k)])$$(15)
$$H_{proj} = \frac {1}{\hat z_k^2} \left[ \begin{matrix} \hat z_k && 0 && -\hat x^k \\ 0 && \hat z_k && -\hat y_k \end{matrix} \right]$$(16)
$$H_{θ_k} = ⌊C(^I_G \hat {\bar q}_k) (^G\hat p_f - ^G\hat p_k)×⌋, H_{p_k} = - C(^I_G \hat {\bar q}_k)$$(17)

Once the measurement Jacobian and residual are computed, we can apply the standard EKF update equations to update the state estimates and error covariance [41].

一旦计算得到Jacobian和残差，我们可以应用标准EKF更新方程，来更新状态估计和误差协方差矩阵。

## 3. State Estimation

It is clear from the preceding section that at the core of visual-inertial navigation systems (VINS) is a state estimation algorithm [see (2) and (12)], aiming to optimally fuse IMU measurements and camera images to provide motion tracking of the sensor platform. In this section, we review the VINS literature by focusing on the estimation engine.

从前一节中可以很明显看到，视觉-惯性导航系统VINS的核心是一个状态估计算法，目标是将IMU的测量和相机图像最优的融合，以给出传感器平台的运动跟踪。在本节中，我们回顾了VINS的文献，关注估计的引擎。

### 3.1. Filtering-based vs. Optimization-based Estimation

Mourikis and Roumeliotis [18] developed one of the earliest successful VINS algorithms, known as the multi-state constraint Kalman filter (MSCKF), which later was applied to the application of spacecraft descent and landing [21] and fast UAV autonomous flight [43]. This approach uses the quaternion-based inertial dynamics [37] for state propagation tightly coupled with an efficient EKF update. Specifically, rather than adding features detected and tracked over the camera images to the state vector, their visual bearing measurements are projected onto the null space of the feature Jacobian matrix (i.e., linear marginalization [44]), thereby retaining motion constraints that only relate to the stochastically cloned camera poses in the state vector [45]. While reducing the computational cost by removing the need to co-estimate potentially hundreds and thousands of point features, this operation prevents the relinearization of the features’ nonlinear measurements at later times, yielding approximations deteriorating its performance.

Mourikis和Roumeliotis [18]开发出了最早的成功的VINS算法，称之为多状态约束Kalman滤波器(MSCKF)，后来应用到了航天器的下降和降落的应用，和快速UAV自动飞行[43]。这种方法使用的是基于四元数的惯性动力学进行状态传播，与高效的EKF更新进行紧密耦合。具体来说，这种方法没有将相机图像中检测和跟踪的特征加入到特征向量中，其视觉测量投影到特征Jacobian矩阵的null空间中，因此，保留的运动约束，只与状态向量中随机克隆的相机姿态有关。因为没有了共同估计数百上千个特征点的需要，计算量就降低下来了，这个操作避免了后来特征的非线性测量的重新线性化，得到的估计使其性能恶化。

The standard MSCKF [18] recently has been extended and improved along different directions. In particular, by exploiting the observability-based methodology proposed in our prior work [46, 47, 48, 49], different observability-constrained (OC)-MSCKF algorithms have been developed to improve the filter consistency by enforcing the correct observability properties of the linearized VINS [19, 38, 39, 40, 50, 51, 52]. A square-root inverse version of the MSCKF, i.e., the square-root inverse sliding window filter (SR-ISWF) [6, 53] was introduced to improve the computational efficiency and numerical stability to enable VINS running on mobile devices with limited resources while not sacrificing estimation accuracy. We have introduced the optimal state constraint (OSC)-EKF [54, 55] that first optimally extracts all the information contained in the visual measurements about the relative camera poses in a sliding window and then uses these inferred relative-pose measurements in the EKF update. The (right) invariant Kalman filter [56] was recently employed to improve filter consistency [25, 57, 58, 59, 60], as well as the (iterated) EKF that was also used for VINS in robocentric formulations [22, 61, 62, 63]. On the other hand, in the EKF framework, different geometric features besides points have also been exploited to improve VINS performance, for example, line features used in [64, 65, 66, 67, 68] and plane features in [69, 70, 71, 72]. In addition, the MSCKF-based VINS was also extended to use rolling-shutter cameras with inaccurate time synchronization [64, 73], RGBD cameras [69, 74], multiple cameras [53, 75, 76] and multiple IMUs [77]. While the filtering-based VINS have shown to exhibit high-accuracy state estimation, they theoretically suffer from a limitation; that is, nonlinear measurements (12) must have a one-time linearization before processing, possibly introducing large linearization errors into the estimator and degrading performance.

标准MSCKF最近从不同的方向进行了拓展和改进。特别是，利用我们之前工作提出的基于观测能力的方法论，提出了不同的观测能力约束的MSCKF算法，通过加入了线性化VINS的正确的观测能力属性，可以改进滤波器的一致性。提出了一种MSCKF的平方根逆版本，即平方根逆滑窗滤波器，来改进计算效率和数值稳定性，来使得VINS可以运行在计算能力有限的移动设备上，而不牺牲估计的准确率。我们还提出了最佳状态约束的EKF，首先最优的提取视觉测量中包含的在一个滑窗中关于相对相机姿态的所有信息，然后在EKF更新中使用这些推断的相对姿态测量。最近采用了不变Kalman滤波器来改进滤波器的一致性，迭代EKF也在robocentric表述中用于VINS。另一方面，在EKF框架中，除了点以外，还利用了其他几何特征来改进VINS的性能，比如，线的特征，和面的特征。另外，基于MSCKF的VINS还拓展到使用同步不精确的卷帘相机，RGBD相机，多相机，和多个IMU。基于滤波的VINS已经被证明可以得到高精度的状态估计，但在理论上仍然有局限；即，非线性测量必须在处理之前进行one-time线性化，可能会对估计引入大的线性化误差，降低性能。

Batch optimization methods, by contrast, solve a nonlinear least-squares (bundle adjustment or BA [78]) problem over a set of measurements, allowing for the reduction of error through relinearization [79, 80] but with high computational cost. Indelman et al. [27] employed the factor graph to represent the VINS problem and then solved it incrementally in analogy to iSAM [81, 82]. To achieve constant processing time when applied to VINS, typically a bounded-size sliding window of recent states are only considered as active optimization variables while marginalizing out past states and measurements [28, 31, 83, 84, 85]. In particular, Leutenegger et al. [28] introduced a keyframe-based optimization approach (i.e., OKVIS), whereby a set of non-sequential past camera poses and a series of recent inertial states, connected with inertial measurements, was used in nonlinear optimization for accurate trajectory estimation. Qin et al. [31] recently presented an optimization-based monocular VINS that can incorporate loop closures in a non-real time thread, while our recent VINS [86] is able to efficiently utilize loop closures in a single thread with linear complexity.

比较起来，批优化方法是在一个测量集合上求解一个非线性最小二乘问题(Bundle Adjustment, or BA)，可以通过重线性化来降低误差，但是计算量巨大。Indelman等[27]采用因子图来表示VINS问题，然后类比iSAM进行逐渐求解。当应用到VINS时，为获得恒定处理时间，在最近的状态附近的有限大小的滑窗，被认为是活跃优化变量，忽略过去的状态和测量。特别是，Leutenegger等[28]提出了一种基于关键帧的优化方法(即，OKVIS)，利用过去的相机姿态的非顺序集合，和一系列最近的惯性状态，与惯性测量一起，用于非线性优化，以得到准确的轨迹估计。Qin等[31]最近提出了几种基于优化的单目VINS，可以在一个非实时线程中进行回环闭合，而我们最近的VINS可以在一个线程中以线性的复杂度高效的利用回环闭合。

### 3.2. Tightly-coupled vs. Loosely-coupled Sensor Fusion

There are different schemes for VINS to fuse the visual and inertial measurements which can be broadly categorized into the loosely-coupled and the tightly-coupled. Specifically, the loosely-coupled fusion, in either filtering or optimization-based estimation, processes the visual and inertial measurements separately to infer their own motion constraints and then fuse these constraints (e.g., [27, 87, 88, 89, 90, 91]). Although this method is computationally efficient, the decoupling of visual and inertial constraints results in information loss. By contrast, the tightly-coupled approaches directly fuse the visual and inertial measurements within a single process, thus achieving higher accuracy (e.g., [18, 28, 34, 40, 85, 92]).

VINS将视觉测量与惯性测量融合到一起，有不同的方案，一般来说可以分为松散耦合和紧耦合方案。具体的，松散耦合的融合，在基于滤波或基于优化的估计中，分别单独处理视觉测量和惯性测量，以推断其自己的运动约束，然后将这些约束融合到一起。这种方法计算量不高，但视觉和惯性约束的解耦会得到信息损失。比较起来，紧耦合的方法直接将视觉和惯性测量在一个过程中融合到一起，可以得到更高的准确率。

### 3.3. VIO vs. SLAM

By jointly estimating the location of the sensor platform and the features in the surrounding environment, SLAM estimators are able to easily incorporate loop closure constraints, thus enabling bounded localization errors, which has attracted significant research efforts in the past three decades [16, 36, 93, 94]. VINS can be considered as an instance of SLAM (using particular visual and inertial sensors) and broadly include the visual-inertial (VI)-SLAM [28, 33, 85] and the visual-inertial odometry (VIO) [18, 22, 39, 40, 52, 95]. The former jointly estimates the feature positions and the camera/IMU pose that together form the state vector, whereas the latter does not include the features in the state but still utilizes the visual measurements to impose motion constraints between the camera/IMU poses. In general, by performing mapping (and thus loop closure), the VI-SLAM gains the better accuracy from the feature map and the possible loop closures while incurring higher computational complexity than the VIO, although different methods have been proposed to address this issue [18, 21, 28, 85, 96, 97]. However, VIO estimators are essentially odometry (dead reckoning) methods whose localization errors may grow unbounded unless some global information (e.g., GPS or a priori map) or constraints to previous locations (i.e., loop-closures) are used. Many approaches leverage feature observations from different keyframes to limit drift over the trajectory [28, 98]. Most have a two-thread system that optimizes a small window of “local” keyframes and features limiting drift in the short-term, while a background process optimizes a long-term sparse pose graph containing loop-closure constraints enforcing long-term consistency [31, 83, 99, 100]. For example, VINS-Mono [31, 100] uses loop-closure constraints in both the local sliding window and in the global batch optimization. Specifically, during the local optimization, feature observations from keyframes provide implicit loop-closure constraints, while the problem size remains small by assuming the keyframe poses are perfect (thus removing them from optimization).

通过对传感器平台位置和周围环境中特征的联合估计，SLAM估计器可以很容易的利用回环闭合约束，使得定位误差变得有限，这在过去三十年吸引了很多研究者。VINS可以认为是SLAM的一个例子（使用了视觉和惯性传感器），大致包括了视觉-惯性SLAM和视觉-惯性里程计。前者对特征位置和相机/IMU姿态进行联合估计，这共同形成了状态向量，而后者在状态中不包括特征，但仍然利用视觉测量对相机/IMU姿态施加运动约束。一般来说，VI-SLAM通过进行建图（因此也有回环闭合），从特征图和可能的回环闭合中得到了更好的准确率，因此计算量比VIO更大，虽然提出了不同的方法来处理这个问题。但是，VIO估计器实际上就是里程计方法，其定位误差可能会无限制的增长，除非使用一些全局信息（比如，GPS，或先验图）或对之前位置的约束（即，回环闭合）。很多方法利用不同关键帧的特征观察，来限制轨迹上的漂移。多数都有双线程系统，对局部关键帧和特征的小窗口进行优化，限制短期的漂移，同时还有一个后台进程，优化长期的稀疏姿态图，包含回环闭合约束，确保长期的一致性。比如，VINS-Mono在局部滑窗和全局批优化上使用回环闭合约束。具体的，在局部优化时，关键帧得到的特征观察给出了隐式的回环闭合约束，而通过假设关键帧的姿态是完美的（因此将其从优化中移除），这个问题的规模保持较小。

In particular, whether or not performing loop closures in VINS either via mapping [83, 101, 102] and/or place recognition [103, 104, 105, 106, 107, 108] is one of the key differences between VIO and SLAM. While it is essential to utilize loop-closure information to enable bounded-error VINS performance, it is challenging due to the inability to remain computationally efficient without making inconsistent assumptions such as treating keyframe poses to be true, or reusing information. To this end, a hybrid estimator was proposed in [109] that used the MSCKF to perform real-time local estimation, and triggered global BA on loop-closure detection. This allows for the relinearization and inclusion of loop-closure constraints in a consistent manner, while requiring substantial additional overhead time where the filter waits for the BA to finish. Recently, Lynen et al. [110] developed a large-scale map-based VINS that uses a compressed prior map containing feature positions and their uncertainties and employs the matches to features in the prior map to constrain the estimates globally. DuToit et al. [102] exploited the idea of Schmidt KF [111] and developed a Cholesky-Schmidt EKF, which, however, uses a prior map with its full uncertainty and relaxes all the correlations between the mapped features and the current state variables; while our latest Schmidt-MSCKF [86] integrates loop closures in a single thread. Moreover, the recent point-line VIO [67] treats the 3D positions of marginalized keypoints as “true” for loop closure, which may lead to inconsistency.

特别的，VIO和SLAM的一个关键区别，是在VINS中是否进行回环闭合，可以通过建图，和/或通过位置识别。利用回环闭合信息，对于限制VINS的性能的误差非常关键，但这非常有挑战，因为不能进行不一致的假设，从而导致计算量非常大，比如认为关键帧的姿态是正确的，或重用信息。为此，[109]中提出了一个复合估计器，使用MSCKF来进行实时局部估计，在检测到回环时，触发全局BA。这使得回环闭合约束可以重新线性化并纳入到考虑中，并且是以一种一致的形式，同时需要非常多额外的开销时间，其中滤波器等待BA结束。最近，Lynen等[110]提出了一个大规模基于地图的VINS，利用一个压缩的包含特征位置和不确定性的先验地图，利用其在先验地图中的匹配来对估计进行全局约束。DuToit等[102]利用了Schmidt KF的思想，提出了一种Cholesky-Schmidt EKF，使用了一种带有完全不确定性的先验地图，对建图的特征和目前的状态变量之间的关系进行了松弛；而我们最新的Schmidt-MSCKF [86]在一个线程中集成了回环闭合。而且，最近的点线VIO[67]将回环闭合中的边缘化关键点的3D位置认为是真，这可能会导致不一致。

### 3.4. Direct vs. Indirect Visual Processing

Visual processing pipeline is one of the key components to any VINS, responsible for transforming dense imagery data to motion constraints that can be incorporated into the estimation problem, whose algorithms can be categorized as either direct or indirect upon the visual residual models used. Seen as the classical technique, indirect methods [18, 28, 30, 40, 51] extract and track point features in the environment, while using geometric reprojection constraints during estimation. An example of a current state-of-the-art indirect visual SLAM is the ORB-SLAM2 [83, 112], which performs graph-based optimization of camera poses using information from 3D feature point correspondences.

视觉处理流程对任何VINS来说都是一个关键组成部分，负责将密集的成像数据变换成运动约束，可以整合到估计问题中，对使用的视觉残差模型，其算法可以归类为直接或间接的。间接方法被视为经典技术，对环境中的特征点进行提取和跟踪，在估计时使用几何重投影约束。目前最好的间接视觉SLAM的一个例子是ORB-SLAM2，利用3D特征点对应性的信息，来进行基于图的相机姿态优化。

In contrast, direct methods [96, 113, 114, 115] utilize raw pixel intensities in their formulation and allow for inclusion of a larger percentage of the available image information. LSD-SLAM [114] is an example of a state-of-the-art direct visual-SLAM which optimizes the transformation between pairs of camera keyframes based on minimizing their intensity error. Note that this approach also optimizes a separate graph containing keyframe constraints to allow for the incorporation of highly informative loop-closures to correct drift over long trajectories. This work was later extended from a monocular sensor to stereo and omnidirectional cameras for improved accuracy [116, 117]. Other popular direct methods include [118] and [119] which estimate keyframe depths along with the camera poses in a tightly-coupled manner, offering low-drift results. Application of direct methods to VINS has seen recent attention due to their ability to robustly track dynamic motion even in low-texture environments. For example, Bloesch et al. [61, 62] used a patch-based direct method to provide updates with an iterated EKF; Usenko et al. [96] introduced a sliding-window VINS based on the discrete preintegration and direct image alignment; Ling et al. [9], Eckenhoff et al. [115] integrated direct image alignment with different IMU preintegration [34, 35, 85] for dynamic motion estimation.

比较起来，直接方法直接利用原始像素灰度，可以利用更多的图像信息。LSD-SLAM是目前最好的直接视觉SLAM的一个例子，基于相机关键帧对的灰度误差的最小化，优化其之间的变换。注意这个方法还优化了一个独立的图，其中包含的关键帧约束可以纳入信息量很大的回环闭合，以修正长轨迹形成的漂移。这个工作后来从单目传感器拓展到了立体相机和全向相机，以改进准确率。其他流行的直接方法包括[118,119]，以紧耦合的方式，对关键帧的深度和相机的姿态进行联合估计，得到低漂移的结果。直接方法在VINS中的应用近期已有关注，因为能够在低纹理的环境中稳健的跟踪动态运动。比如，Bloesch等使用了一种基于图像块的直接方法来给迭代EKF进行更新；Usenko等提出了一种滑窗VINS，基于离散预整合和直接图像对齐；Ling等[9]，Eckenhoff等[115]将直接的图像对齐与不同的IMU预整合整合到一起，用于动态运动估计。

While direct image alignments require a good initial guess and high frame rate due to the photometric consistency assumption, indirect visual tracking consumes extra computational resources on extracting and matching features. Nevertheless, indirect methods are more widely used in practical applications due to its maturity and robustness, but direct approaches have potentials in textureless scenarios.

直接图像对齐需要一个很好的初始估计和很高的帧率，因为光度一致性假设，间接的视觉跟踪则需要额外的计算资源来提取特征，匹配特征。尽管如此，间接方法在实际的应用中使用的更广泛，由于其比较成熟，而且很稳健，但直接方法在无纹理的场景中有潜力。

### 3.5. Inertial Preintegration

Lupton and Sukkarieh [33] first developed the IMU preintegration, a computationally efficient alternative to the standard inertial measurement integration, which peforms the discrete integration of the inertial measurement dynamics in a local frame of reference, thus preventing the need to reintegrate the state dynamics at each optimization step. While this addresses the computational complexity issue, this method suffers from singularities due to the use of Euler angles in the orientation representation. To improve the stability of this preintegration, an on-manifold representation was introduced in [29, 34] which presents a singularity-free orientation representation on the SO(3) manifold, incorporating the IMU preintegration into graph-based VINS.

Lupton和Sukkarieh[33]首先提出了IMU预整合，这是标准惯性测量整合的一种计算量很小的替代，标准惯性测量整合，是将惯性测量动力学离散整合到参考局部框架，这就避免了在每个优化步骤中都需要重新整合状态动力学。这解决了计算上复杂的问题，但这种方法存在奇异点问题，因为使用了角度表示的Euler角。为改进这种预整合的稳定性，[29,34]提出了一种流形上的表示，在SO(3)流形中给出了无奇异点的方向表示，将IMU预整合整合到基于图的VINS中。

While Shen et al. [85] introduced preintegration in the continuous form, they still discretely sampled the measurement dynamics without offering closed-form solutions, which left a significant gap in the theoretical completeness of preintegration theory from a continuous-time perspective. As compared to the discrete approximation of the preintegrated measurement and covariance calculations used in previous methods, in our prior work [35, 63], we have derived the closed-form solutions to both the measurement and covariance preintegration equations and showed that these solutions offer improved accuracy over the discrete methods, especially in the case of highly dynamic motion.

Shen等[85]提出了连续形式的预整合，但他们仍然对测量动力学进行离散采样，没有给出闭合形式的解，这在预整合理论理论完备性上从连续时间的角度留下了一个明显的空白。与之前的方法中使用的预整合测量和协方差计算的离散近似相比，在我们之前的工作[35,63]中，我们对测量和协方差预整合方程推导出了闭合形式解，表明这些解比离散方法的准确率是有改进的，尤其是在高动态运动的情况下。

### 3.6. State Initialization

Robust, fast initialization to provide of accurate initial state estimates is crucial to bootstrap real-time VINS estimators, which is often solved in a linear closed form [7, 84, 120, 121, 122, 123, 124]. In particular, Martinelli [123] introduced a closed-form solution to the monocular visual-inertial initialization problem and later extended to the case where gyroscope bias calibration is also included [125] as well as to the cooperative scenario [126]. These approaches fail to model the uncertainty in inertial integration since they rely on the double integration of IMU measurements over an extended period of time. Faessler et al. [127] developed a re-initialization and failure recovery algorithm based on SVO [113] within a loosely-coupled estimation framework, while an additional downward-facing distance sensor is required to recover the metric scale. Mur-Artal and Tard´os [83] introduced a high-latency (about 10 seconds) initializer built upon their ORB-SLAM [112], which computes initial scale, gravity direction, velocity and IMU biases with the visual-inertial full BA given a set of keyframes from ORB-SLAM. In [7, 84] a linear method was recently proposed for noise-free cases, by leveraging relative rotations obtained by short-term IMU (gyro) pre-integration but without modeling the gyroscope bias, which may be unreliable in real world in particular when distant visual features are observed.

提供稳健快速的准确初始状态估计，对于提升实时VINS估计器的性能是非常关键的，这通常是以线性闭合的形式求解的。特别是，Martinelli对单目视觉-惯性初始化问题提出了闭合形式解，后来拓展到了包括陀螺仪偏移标定的情况，以及合作的场景中。这些方法没有对惯性整合中的不确定性进行建模，因为这依赖于较长时间段内的IMU测量的二次整合。Faessler等[127]提出了一种基于SVO的重初始化和故障恢复算法，在一个松散耦合的估计框架中，而需要一个额外的面向下的距离传感器，以恢复度量尺度。[83]提出了一种高延迟（约10s）初始器，是在ORB-SLAM的基础上构建起来的，在给定了关键帧的集合后，用ORB-SLAM和视觉-惯性完整BA计算了初始的尺度，重力方向，速度和IMU偏移。在[7,84]中，对无噪声的情况提出了一种线性方法，利用短期IMU（陀螺仪）预整合得到的相对旋转，大没有对陀螺仪的偏移进行建模，这可能在真实世界中是不可靠的，尤其是观察远距离的视觉特征时。

## 4. Sensor Calibration

When fusing the measurements from different sensors, it is critical to determine in high precision both the spatial and temporal sensor calibration parameters. In particular, we should know accurately the rigid-body transformation between the camera and the IMU in order to correctly fuse motion information extracted from their measurements. In addition, due to improper hardware triggering, transmission delays, and clock synchronization errors, the timestamped sensing data of each sensor may disagree and thus, a timeline misalignment (time offset) between visual and inertial measurements might occur, which will eventually lead to unstable or inaccurate state estimates. It is therefore important that these time offsets should also be calibrated. The problem of sensor calibration of the spatial and/or temporal parameters has been the subject of many recent VINS research efforts [42, 128, 129, 130, 131, 132]. For example, Mirzaei and Roumeliotis [42] developed an EKF-based spatial calibration between the camera and IMU. Nonlinear observability analysis [133] for the calibration parameters was performed to show that these parameters are observable given random motion. Similarly, Jones and Soatto [128] examined the identifiability of the spatial calibration of the camera and IMU based on indistinguishable trajectory analysis and developed a filter based online calibration on an embedded platform. Kelly and Sukhatme [129] solved for the rigid-body transformation between the camera and IMU by aligning rotation curves of these two sensors via an ICP-like matching method.

不同传感器的测量融合时，要高精度的确定，传感器标定的空间和时间参数，这是非常关键的。特别是，我们应当精确的知道，相机和IMU之间的刚体变换，以正确的融合它们的测量提取出运动信息。另外，由于不合适的硬件触发，传输延迟，和时钟同步误差，每个传感器的打上时间标签的传感数据可能会不一致，因此，视觉和惯性测量之间的时间线不对齐可能会出现，这最终会导致不稳定或不精确的状态估计。因此，这些时间偏差也应当进行标定。传感器标定问题中的空间和/或时间参数，是最近很多VINS研究的主题。比如，[42]提出了一种基于EKF的相机和IMU之间的空间校准。对校准参数的非线性可观测性分析[133]表明，给定随机运动的时候，这些参数是可观测的。类似的，[128]研究了相机和IMU之间的空间标定的可识别性，这是基于无法分辨的轨迹分析，在一个嵌入式平台提出了一种基于滤波器的在线标定。[129]通过对这些两个传感器的旋转曲线，采用类似ICP的匹配方法进行对齐，求解了相机和IMU之间的刚体变换。

Many of these research efforts have been focused on offline processes that often require additional calibration aids (fiducial tags) [42, 130, 134, 135, 136]. In particular, as one of the state-of-the-art approaches, the Kalibr calibration toolbox [130, 134] uses a continuous-time basis function representation [137] of the sensor trajectory to calibrate both the extrinsics and intrinsics of a multi-sensor system in a batch fashion. As this B-spline representation allows for the direct computation of expected local angular velocity and local linear acceleration, the difference between the expected and measured inertial readings serve as errors in the batch optimization formulation. A downside of offline calibration is that it must be performed every time a sensor suite is reconfigured. For instance, if a sensor is removed for maintenance and returned, errors in the placement could cause poor performance, requiring a time-intensive recalibration.

很多这些研究努力，聚焦在一个离线过程中，需要额外的标定辅助（基准tag）。特别的，Kalibr标定工具箱是目前最好的一种方法，使用了传感器轨迹的连续时间基函数表示，来以一种批次的方式标定一个多传感器系统的内参和外参。这种B样条的表示可以直接计算期望的局部角速度，和局部线性加速度，期望和测量的惯性读数之间的差异，是批优化表示中的误差。离散标定的一个不好之处，是每次传感器重新配置时，都需要进行标定。比如，如果一个传感器进行了维护并重新安装，放置导致的误差会导致很差的性能，需要一个非常耗时的重新标定。

Online calibration methods, by contrast, estimate the calibration parameters during every operation of the sensor suite, thereby making them more robust to and easier to use in such scenarios. Kim et al. [138] reformulated the IMU preintegration [33, 34, 35] by transforming the inertial readings from the IMU frame into a second frame. This allows for calibration between IMUs and other sensors (including other IMUs), but does not include temporal calibration and also relies on computing angular accelerations from gyroscope measurements. Li and Mourikis [131] performed navigation with simultaneous calibration of both the spatial and temporal extrinsics between a single IMU-camera pair in a filtering framework for use on mobile devices, which was later extended to include the intrinsics of both the camera and the IMU [139]. Qin and Shen [132] extended their prior work on batch-based monocular VINS [31] to include the time offset between the camera and IMU by interpolating the locations of features on the image plane. Schneider et al. [140] proposed the observability-aware online calibration utilizing the most informative motions. While we recently have also analyzed the degenerate motions of spatiotemporal calibration [141], it is not fully understood how to optimally model intrinsics and simultaneously calibrate them along with extrinsics [142, 143].

比较之下，在线标定方法，在传感器的每次操作时都估计标定参数，因此使其更加稳健，并在这种场景中更加容易使用。Kim等[138]重新表述了IMU预整合问题，将惯性器件读数从IMU框架中变换到另一个框架中。这使得IMU可以与其他传感器（包括其他IMU）进行标定，但并没有包括时间上的标定，也依赖于从陀螺仪的测量中计算角加速度。[131]在导航的同时进行标定，包括单个IMU-相机对在空间和时间上的外参，使用的是滤波的框架，在移动设备上进行使用，后来拓展了包括相机和IMU的内参。[31]是基于图像块的单目VINS，[132]对其进行了拓展，对在图像平面上的特征位置进行了插值，包括了相机和IMU之间的时间偏移。[140]提出了考虑了观测性的在线标定过程，利用了最有信息的运动。我们最近还分析了空间时间标定中的退化运动情况[141]，怎样最优的对内参进行建模，同时与外参进行同时标定，这还没有得到完全理解

## 5. Observability Analysis 观测能力分析

System observability plays an important role in the design of consistent state estimation [49], which examines whether the information provided by the available measurements is sufficient for estimating the state/parameters without ambiguity [133, 144, 145]. When a system is observable, the observability matrix is invertible, which is also closely related to the Fisher information (or covariance) matrix [146, 147]. Since this matrix describes the information available in the measurements, by studying its nullspace we can gain insights about the directions in the state space along which the estimator should acquire information. In our prior work [46, 47, 48, 52, 146, 148, 149, 150], we have been the first to design observability-constrained (OC) consistent estimators for robot localization problems. Since then, significant research efforts have been devoted to the observability analysis of VINS (e.g., [19, 38, 151, 152]).

系统观测能力在一致状态估计中扮演了重要的角色，这研究的是可用的测量提供的信息是否足以毫无疑义的估计状态/参数。当一个系统是可观测的，观测矩阵就是可逆的，这与Fisher信息（或协方差）矩阵是紧密关联的。由于这个矩阵描述了观测中可用的信息，通过研究其核空间，我们可以洞察其状态空间的方向，沿着这个方向估计器应该可以获取信息。在我们之前的工作中，我们是第一个设计观测能力约束的一致性估计器进行机器人定位问题的。自动那时起，很多研究工作都致力于VINS的观测能力分析。

In particular, VINS nonlinear observability analysis has been studied using different nonlinear system analysis techniques. For example, Jones and Soatto [128], Hernandez et al. [153] the system’s indistinguishable trajectories [154] were examined from the observability perspective. By employing the concept of continuous symmetries as in [155], Martinelli [122] analytically derived the closed-form solution of VINS and identified that IMU biases, 3D velocity, global roll and pitch angles are observable. He has also examined the effects of degenerate motion [156], minimum available sensors [157], cooperative VIO [126] and unknown inputs [158, 159] on the system observability. Based on the Lie derivatives and observability matrix rank test [133], Hesch et al. [51] analytically showed that the monocular VINS has 4 unobservable directions, corresponding to the global yaw and the global position of the exteroceptive sensor. Guo and Roumeliotis [69] extended this method to the RGBD-camera aided INS that preserves the same unobservable directions if both point and plane measurements are available. With the similar idea, in [74, 129, 160], the observability of IMU-camera (monocular, RGBD) calibration was analytically studied, which shows that the extrinsic transformation between the IMU and camera is observable given generic motions. Additionally, in [161, 162], the system with a downward-looking camera measuring point features from horizontal planes was shown to have the observable global z position of the sensor.

特别是，VINS非线性观测能力分析已经使用不同的非线性系统分析技术进行了研究。比如，[128,153,154]都从观测能力的角度进行了研究。通过采用[155]中的连续对称性的概念，[122]解析的推导了VINS的闭合形式解，认为IMU偏移，3D速度，全局roll和pitch角是可观测的。他还研究了退化运动的效果[156]，最小可用传感器[157]，合作式VIO[126]和未知的输入[158,159]对系统观测能力的影响。基于李导数和观测能力矩阵等级测试[133]，[51]解析的表明，单目VINS有4个不可观测的方向，对应的全局的yaw和感受外界刺激传感器的全局位置。[69]将这种方法拓展到了RGBD相机附属的INS，如果点和面的测量都可用，那么不可观测的方向就是一样的。[74, 129, 160]有类似的思想，IMU-相机（单目，RGBD）标定的观测能力进行了解析研究，表明IMU和相机之间的外参变换在给定的通用运动时是可观测的。另外，在[161,162]中，有向下观测的相机的系统，从水平平面上测量特征点，可以观测到传感器的全局z位置。

As in practice VINS estimators are typically built upon the linearized system, what is practically more important is to perform observability analysis for the linearized VINS. In particular, the observability matrix [41, 163] for the linearized VINS system over the time interval [ko k] has the nullspace (i.e., unobservable subspace) that ideally spans four directions:

在实践中，VINS估计器一般是在线性化系统上构建起来的，实践上更重要的是，对线性化的VINS进行观测能力分析。特别是，线性化VINS系统在时间间隔[ko, k]上的观测能力矩阵的核空间（即，不可观测子空间）可以理想的张成四个方向：

$$M = \left[ \begin{matrix} H_{k_o} \\ H_{k_o + 1}Φ_{k_o} \\ ... \\ H_kΦ_{k−1} · · · Φ_{k_o} \end{matrix} \right] ⇒^{MN = 0} N = \left[ \begin{matrix} 0_3 && C(^I_G \bar q_k)^Gg \\ 0_3 && 0_3 \\ 0_3 && -⌊^G v_k×⌋^Gg \\ 0_3 && 0_3 \\ I_3 && -⌊^G p_k×⌋^Gg \\ I_3 && -⌊^G p_f×⌋^Gg \end{matrix} \right]$$(18)

Note that the first block column of N in (18) corresponding to the global translation while the second block column corresponds to the and global rotation about the gravity vector [19, 38, 40, 51]. When designing a nonlinear estimator for VINS, we would like the system model employed by the estimator to have an unobservable subspace spanned by these directions. However, this is not the case for the standard EKF as shown in [19, 38, 39, 40, 51]. In particular, the standard EKF linearized system, which linearizes system and measurement functions at the current state estimate, has an unobservable subspace of three, instead of four dimensions. This implies that the filter gains non-existent information from available measurements, leading to inconsistency. To address this issue, the first-estimates Jacobian (FEJ) idea [47] was adopted to improve MSCKF consistency [19, 40], and the OC methodology [48] was employed in developing the OC-VINS [38, 39, 50]. We recently have also developed the robocentric VIO (R-VIO) [22, 63] which preserves proper observability properties independent of linearization points.

注意(18)中N的第一块列，对应着全局平移，而第二块列对应着围绕着重力向量的全局旋转。当设计VINS的非线性估计器时，我们希望估计器采用的系统模型其不可观测子空间是由这些方向张成的。但是，对于标准EKF来说并不是这个情况。特别的，标准EKF线性化系统，在当前状态估计处对系统和度量函数进行了线性化，不可观测子空间的维度是3，而不是4。这说明，滤波器从可用的测量中得到了并不存在的信息，导致了非一致性。为解决这个问题，[47]采用了第一估计Jacobian（FEJ）来改进MSCKF的一致性，[48]采用了OC方法提出了OC-VINS。我们最近提出了robocentric VIO (R-VIO)，保持了合理的观测能力属性，与线性化点无关。

## 6. Discussions and Conclusions

As inertial and visual sensors are becoming ubiquitous, visual-inertial navigation systems (VINS) have incurred significant research efforts and witnessed great progresses in the past decade, fostering an increasing number of innovative applications in practice. As a special instance of the well-known SLAM problem, VINS researchers have been quickly building up a rich body of literature on top of SLAM [36]. Given the growing number of papers published in this field, it has become harder (especially for practitioners) to keep up with the state of the art. Moreover, because of the particular sensor characteristics, it is not trivial to develop VINS algorithms from scratch without understanding the pros and cons of existing approaches in the literature (by noting that each method has its own particular focus and does not necessarily explain all the aspects of VINS estimation). All these have motivated us to provide this review on VINS, which, to the best of our knowledge, is unfortunately lacked in the literature and thus should be a useful reference for researchers/engineers who are working on this problem. Upon our significant prior work in this domain, we have strived to make this review concise but complete, by focusing on the key aspects about building a VINS algorithm including state estimation, sensor calibration and observability analysis.

惯性和视觉传感器正变得无处不在，视觉-惯性导航系统(VINS)有很多研究工作，在过去十年经理了很大进展，在实践中有很多创新应用。作为著名的SLAM问题的一个特殊例子，VINS研究者在SLAM的基础上构建了很多文献。在这个领域发表的文献很多，要跟踪目前最新的结果变得困难。而且，由于特殊的传感器特性，从头开发VINS算法而不理解现有方法的pros and cons，就比较困难。所有这些都使我们给出这篇VINS的综述。在我们在这个领域的工作基础之上，我们努力使这个综述简洁但完整，聚焦在构建VINS算法的关键部分，包括状态估计，传感器标定和观测能力分析。

While there are significant progresses on VINS made in the past decade, many challenges remain to cope with, and in the following we just list a few open to discuss:

在过去十年中，VINS有了显著的进展，但仍然有很多挑战，下面我们列出了一些可以进行讨论：

- Persistent localization: While current VINS are able to provide accurate 3D motion tracking, but, in small-scale friendly environments, they are not robust enough for long-term, large-scale, safety-critical deployments, e.g., autonomous driving, in part due to resource constraints [95, 97, 164]. As such, it is demanding to enable persistent VINS even in challenging conditions (such as bad lighting and motions), e.g., by efficiently integrating loop closures or building and utilizing novel maps.

持续定位：当前的VINS可以给出准确的3D运动跟踪，但在小尺度友好的环境中，对长期、大规模、安全上很关键的部署中，仍然不够稳健，如，自动驾驶，部分是因为资源的限制。这样，即使在有挑战的环境中，也要能够进行持续的VINS（比如很差的光照和运动），如，高效的集成回环闭合，或构建和利用新的地图。

- Semantic localization and mapping: Although geometric features such as points, lines and planes [151, 165] are primarily used in current VINS for localization, these handcrafted features may not be work best for navigation, and it is of importance to be able to learn best features for VINS by leveraging recent advances of deep learning [166]. Moreover, a few recent research efforts have attempted to endow VINS with semantic understanding of environments [167, 168, 169, 170], which is only sparsely explored but holds great potentials.

语义定位和建图：目前的VINS用于定位的主要是点，线和平面这样的几何特征，但这些手工设计的特征对于导航来说并不一定是最好的，利用最近深度学习的进展对VINS学习最好的特征，这非常重要。而且，最近的几个研究已经尝试让VINS对环境进行语义理解，这方面的探索不多，但是有很大的潜力。
  
- High-dimensional object tracking: When navigating in dynamic complex environments, besides high-precision localization, it is often necessary to detect, represent, and track moving objects that co-exist in the same space in real time, for example, 3D object tracking in autonomous navigation [92, 171, 172].

高维目标跟踪：当在动态复杂环境中导航时，除了高精度定位，对在同样空间中同时存在的运动目标进行实时检测，表示和跟踪也很有必要，比如，在自动导航中的3D目标跟踪。

- Distributed cooperative VINS: Although cooperative VINS have been preliminarily studied in [126, 173], it is still challenging to develop real-time distributed VINS, e.g., for crowd sourcing operations. Recent work on cooperative mapping [174, 175] may shed some light on how to tackle this problem.

分布式合作式VINS：虽然合作式VINS已经在[126,173]中进行了初步研究，但开发实时分布式VINS仍然是很有挑战的，比如，对于众包的操作。最近在合作式建图上的工作对于怎样解决这个问题可能有一点启发。

- Extensions to different aiding sensors: While optical cameras are seen an ideal aiding source for INS in many applications, other aiding sensors may more proper for some environments and motions, for example, acoustic sonars may be instead used in underwater [176]; low-cost light-weight LiDARs may work better in environments, e.g., with poor lighting conditions [71, 177]; and event cameras [178, 179] may better capture dynamic motions [180, 181]. Along this direction, we should investigate in-depth VINS extensions of using different aiding sources for applications at hand.

拓展到不同的辅助传感器：对INS来说，光学相机是一个很理想的辅助，但其他的辅助传感器可能更适合于一些环境和运动，比如，声学声纳更适合于在水下使用；低价轻量的激光雷达可能在光照不好的环境中更加实用；event相机可以更好的捕获动态运动。沿着这个方向，我们应当研究更深入的VINS拓展，对手头上的应用使用不同的辅助源。