# A general framework and review of scatter correction methods in cone beam CT. Part 2: Scatter estimation approaches

## 0. Abstract

The main components of scatter correction procedures are scatter estimation and a scatter compensation algorithm. This paper completes a previous paper where a general framework for scatter compensation was presented under the prerequisite that a scatter estimation method is already available. In the current paper, the authors give a systematic review of the variety of scatter estimation approaches. Scatter estimation methods are based on measurements, mathematical–physical models, or combinations of both. For completeness they present an overview of measurement-based methods, but the main topic is the theoretically more demanding models, as analytical, Monte-Carlo, and hybrid models. Further classifications are 3D image-based and 2D projection-based approaches. The authors present a system-theoretic framework, which allows to proceed top–down from a general 3D formulation, by successive approximations, to efficient 2D approaches. A widely useful method is the beam-scatter-kernel superposition approach. Together with the review of standard methods, the authors discuss their limitations and how to take into account the issues of object dependency, spatial variance, deformation of scatter kernels, external and internal absorbers. Open questions for further investigations are indicated. Finally, the authors refer on some special issues and applications, such as bow-tie filter, offset detector, truncated data, and dual-source CT.

散射修正过程的主要组成部分是，散射估计和散射补偿算法。前一篇文章提出了一个散射补偿的框架，本文则总结了散射估计的方法。本文中，我们给出了各种散射估计方法的系统回顾。散射估计方法是基于测量的，数学-物理模型，或两者的组合的。为了完整性，我们给出了基于测量的方法的概览，但主要话题是理论上要求更高的模型，比如解析的，Monte-Carlo的，和混合模型。更多分类是，基于3D图像的，和基于2D投影的方法。作者给出了一种系统理论的框架，这可以从一个通用的3D公式，自上而下的进行处理，通过连续的近似，到高效的2D方法。广泛使用的方法是，射束-散射-核叠加方法。与标准方法的回顾一起，作者讨论了其局限，以及怎样考虑目标依赖性，空间变化，散射核的形变，外部吸收和内部吸收问题。给出了开放的问题以进行进一步的研究。最后，我们给出了一些特殊问题和应用，比如bow-tie滤波器，偏移检测器，截断数据，和双能CT。

**Key words**: cone beam CT, image quality, scatter correction, scatter suppression, scatter kernels, scatter kernel superposition, scatter integral transform, Monte Carlo

## 1. Introduction

### 1.1 Organization and aim of the paper

A leading observation in our previous paper was that the structure of scatter correction algorithms comprises two main components: a scatter estimation model and a scatter compensation algorithm. Typically, scatter estimation and scatter compensation are intertwined in iterative algorithms. The aim of the current paper is to present a systematic framework for the multitude of scatter estimation approaches.

前一篇文章的结论是，散射修正算法的结构由两个主要部分组成：散射估计模型，和散射补偿算法。一般来说，散射估计和散射补偿在迭代算法中是纠缠在一起的。本文的目的是，给出大量散射估计方法的系统框架。

In Fig. 1, we complete the structogram of scatter suppression methods just given in our companion paper, where the scatter estimation methods had been omitted. In Fig. 1, here, complementarily the scatter estimation methods are inserted in detail, and the corresponding chapters in the current paper are indicated, whereas hardware rejection and software compensation methods are only hinted.

在图1中，我们补全了散射抑制方法的结构图，在另一篇文章中，我们给出了另外一部分，其中散射估计方法被忽略了。在图1中，散射估计方法给出了具体细节，给出了本文中对应的章节，而硬件反射和软件补偿方法只给出了标题。

In Section I B we mention the state-of-knowledge with respect to admissible approximations in simulation of coherent and incoherent scattering for CT imaging. Although our paper is focused at software methods, for completeness in Sec. II, the main measurement-based scatter estimation methods, i.e., collimator-shadow, beam-stop, and primary modulation method are discussed.

在1.2中，我们在允许的近似范围内，给出了现状总结，对CT成像的相干散射和不相干散射进行了仿真。虽然本文关注的是软件方法，为了完整性，在第2部分中，主要的基于测量的散射估计方法也进行了讨论，即，collimator-shadow, beam-stop, 和primary modulation方法。

In Sec. III, our general framework of mathematical–physical scatter estimation approaches is formulated by an integral transform consisting in a scatter source term and a propagation kernel. This approach is specialized in a 3D formulation based on a point-scatter-kernel model in Sec. III B and in a 2D formulation based on a beam-scatter-kernel model in Sec. III C. The 2D approach can be derived from the 3D approach, as is demonstrated in Appendix A. Realizations of the 3D approach are analytic models (Sec. III B 2 and Appendix A) and Monte Carlo models (Sec. V C).

在第3部分中，我们的数学-物理散射估计方法的总框架，是由一个积分变换表述的，由一个散射源和传播核组成。这种方法专用于3D表示，基于点散射核的模型，在3.2节，在3.3节中是基于射束-散射-核的模型的2D表述。2D方法可以从3D方法中推导得到，如附录A所示。3D方法的实现是解析模型（见3.2.2和附录A），以及Monte Carlo模型（5.3节）。

Section IV is devoted to the practically important 2D methods, represented by the water-equivalent thickness related beam-scatter-kernel (BSK) superposition method, which is described in Sec. IV A and can be specialized to the variety of widely applied convolution approaches (Sec. IV B). Modifications of the convolution approaches to take into consideration object dependency, lateral thickness changes, edge effects are treated in Sec. IV C. Our main concern is object scatter, i.e., radiation scattered by the object or patient to be imaged. For completeness, external and detector scatter are discussed in Secs. IV E and IV F. Some open issues are discussed in Sec. IV G. How to treat bow-tie filters is outlined in Sec. IV D. In Sec. V, we resume 3D image-based scatter estimation methods, as geometric model, Monte Carlo, and hybrid approaches. For completeness, we also mention heuristic “empirical” approaches to remove scatter induced artifacts, although they are unphysical without a scatter estimation model.

第4部分是实践上很重要的2D方法，表示为与等效水厚度相关的BSK叠加方法，如4.1节表述，专用于广泛使用的卷积方法（见4.2节）。卷积方法的修正，将目标依赖性，横向厚度变化，边缘效果的影响纳入考虑，这在4.3节讨论。我们主要的考虑是目标散射，即由目标，或要成像的病人散射的辐射。为完整起见，外部散射和探测器散射在4.5节和4.6节进行讨论。在4.7节中讨论了一些开放性问题。在4.4节和第5章中讨论了怎样处理bow-tie滤波器，我们恢复了基于3D图像的散射估计方法，作为几何模型，Monte Carlo和混合方法。为完整，我们也提及了启发性的经验方法，以去除散射引入的伪影，虽然这些方法并没有使用散射估计模型，是非物理的方法。

References to scatter estimation approaches from other fields, as in diagnostic x-ray radiology and emission tomography, which have not yet been sufficiently appreciated in CT, are given as ever pertinent.

其他领域的散射估计方法的参考，如诊断X射线放射学，和辐射断层成像，这在CT中并没有得到充分利用，也给出了论述。

In Sec. VI, miscellaneous applications and issues, such as offset detector positioning, truncation, dual-source CT, and dual-energy imaging are discussed. 在第6章中，讨论了各种应用和问题，如偏移检测摆位，截断，双能CT，和双能成像。

Section VII gives a summary and outlook to possible improvements by future work. 第7章中给出了总结，和未来可能的改进方向。

### 1.2 Some tissues of scatter physics

#### 1.2.1 Compton (incoherent) scatter

To calculate the Compton scatter angular distribution, it is common to apply the in scientific literature it’s named “Klein-Nishima formula”, formula, multiplied with the incoherent scattering function for effects of electron energy binding energies. The latter may be neglected.

#### 1.2.2 Rayleigh (coherent) scatter

Contrary to Yao and Leszczynski, who claim that coherent scatter may be neglected in CT modeling, Johns and Yaffe, Neitzel et al., Leliveld et al., Persliden and Carlsson, Poletti et al., Cardoso et al., Kyriakou et al., and Engel et al. have pointed out that the coherent scatter plays a significant role and they stress that the Thomson formula modified by the molecular coherent form factors should be used. But Poludniowski et al. have recently demonstrated that the independent atom approximation is good enough for standard CT applications with extended scattering objects, since errors would mainly be concentrated near density discontinuities and be negligibly low.

### 1.3 Mathematical notations

To avoid complicated formula, we prefer compact notations using operator symbols whereever possible. We frequently omit space variables, where functions in object space are implicitly defined in a domain ⊂ R^3 and functions in projection space defined in a domain of R^2 × [0, 2π) where the detector coordinates are u ∈ R^2 and the projection view is denoted by an angular variable α.

For projection data, we will use normalized intensities Φ, i.e., measured or virtual primary fluence data I_P are divided by the calibration data corresponding to measurements I_0 without attenuating object and without scattered radiation. We assume that all usual calibration and precorrection procedures to compensate for detector related errors have yet been carried out.

## 2. Measurement-based scatter estimation

Although our main topic is the theoretically more demanding mathematical approaches, for completeness, we discuss the measurement-based scatter estimation methods below. Pictorial illustrations will be found in the corresponding references. A final judgment is scarcely possible up to now, since improvements of the methods in Secs. 2.2 and 2.3 are still topics of research.

### 2.1 Collimator-shadow continuation method

Siewerdsen et al. proposed to measure scatter fluence within the full shadow of the collimator outside the field-of-view. This needs several detector rows and columns near the border of the detector and, therefore, reduction of field-size. The estimation of scatter distribution in the interior of the field-of-view is obtained by interpolation from the measured boundary data. This method either can be used directly for a simple scatter correction or may be used to update software corrections. The conclusion by Akbarzadeh et al. that the collimator-shadow measurement be superior to the beam-stop method had been obtained with a narrow cone-beam geometry of 4 cm axial coverage and does not hold for wider axial cone-beam geometries.

### 2.2 Beam-absorber array method

Ning et al. and Liu et al. proposed to use a beam-absorber (beam-stop and beam-blocker) array (BAA) between x-ray source and patient to measure scatter behind the BAA shadows. The BAA has to be mounted and rotated with the x-ray tube and the detector. A disadvantage is the additional acquisition, although a coarse sampling of projection views and low dose is sufficient. The influence of finite size and scattering of the blockers has to be addressed.

In order to avoid a second acquisition, Zhu et al. proposed to install the BAA permanently and move it during rotation such that artifacts caused by the BAA are reduced. Yan et al. compared different acquisition and interpolation methods and achieved promising artifact reduction by an interpolation method based on projection view correlation. In another study, semitransparent beam-absorbers are proposed; with respect to the remaining artifacts after scatter correction an optimum at about 50% transparency was found.

Jin et al. use a collinear 1D beam-absorber grid, composed of lead septa. Projection image data are obtained from the grid interspace regions, while scatter is measured in the shadow regions under the grid. This technique combines scatter and dose reduction due to collimation by grid septa and scatter measurement by the BAA method. The missing projection data are obtained differently in different acquisition modes, by interpolation or by dual-rotation and=or grid shift. In spite of remaining registration errors and loss of resolution (<1 mm), the method seems to be promising for cone-beam computed tomography (CBCT) imaging in image-guided radiotherapy applications.

Although the (opaque or semitransparent) BAA approaches have not attained a break-through up to now due to the inconvenience of mechanics and propensity to residual artifacts, these methods are still topics of research and improvement.

### 2.3 Primary Modulation Method

The method, already mentioned by Maher and Malone, was proposed for CBCT scatter estimation by Bani-Hashemi et al. in an abstract and was elaborated and improved by Zhu et al. and Gao et al. in various papers. The idea is adopted from communication theory and is based on the hypothesis that scatter distributions are spatial low-frequency “signals.” Without applying scatter physics, the separation of scatter from primary signals is obtained by use of a 2D modulator grid and application of demodulation techniques. The modulator grid consists, similar to BAA, of a regular 2D array of small disks or squares, however, semitransmittent. The spatial frequency of the modulator grid sampling must be far higher than the frequency content of the scatter distribution (except noise). Main problems are nonlinearity errors from spatially varying transmittance of the modulator elements due to x-ray polychromacy, and artifacts caused by the modulator grid overlaying the projection images. Promising results have been reported by Gao et al. The method is still a topic of research.

## 3. System-theoretic framework

Former system-theoretic approaches to describe the effect of scatter by point-spread and modulation-transfer functions had been presented by Barrett and Swindell, Boone et al., Smith and Kruger. Main properties such as low spatial frequency and decrease of contrast-to-noise ratio have been shown by means of Fourier methods. However, the challenge of more accurate estimation of spatial scatter distributions from irradiated objects or from projection images demands to go beyond the restrictions of spatial invariance and linearity. For systematic reasons, contrary to the historical development, we will present a top–down approach.

In a system-theoretic approach, the system is represented by an input–output model and a transformation between input and output. In a discrete linear system, the transformation is described by a system matrix (transfer-matrix), in a nonlinear system, the system matrix may change with the input signal. In a continuous model, the system matrix is replaced by a transfer (propagation) kernel.

### 3.1 General approach

#### 3.1.1 General formulation

The “output” scatter signal is represented by an integral transform of a modified “input” or “source” function ψ multiplied with a scatter propagation kernel K_f

$$S(u) = \int_F (Tψ)(t)K_f(t,u)dt$$(1)

where the meaning of the symbols is S(u) = S(u, α) normalized scatter fluence at detector pixel u for projection angle α; t integration vector variable (in object volume or at detector area, depending on whether the approach is 3D or 2D); F field of integration, depending on collimation [Fig. 2(a)]; (Tψ) (modified) “scatter source term” or “scatter potential”; ψ normalized primary fluence; K_f scatter propagation kernel, which attributes to each “scatter source” point t, e.g., the probability (including further scatter events) to be detected at u at the detector; K_f depends on many parameters, such as α, acquisition parameters, and on the scattering object.

Particular interpretations of Eq. (1) are given in Secs. 3.2 and 3.3 below. The modification operator T is derived from a physical model as, e.g., by Eq. (5) below or its purpose is a proper scaling of the kernel K_f as given by the examples in Sec. 3.3.2.

The integration vector variable t may also comprise photon energy; however, we prefer to consider the functions ψ and K_f as integrated over the energy spectrum.

The scatter kernel K_f depends on many parameters, such as acquisition geometry, air-gap between object and detector, the effective x-ray energy spectrum (determined by x-ray tube voltage, beam filters, and detector energy response), and the design of antiscatter grid (in case of use).

The implicit dependency of K_f on part of or the whole object is indicated by a parameter vector

$$f = (f, μ_s, \frac {dσ}{dΩ}, ρ, ...)$$(2)

which describes the relevant physical object properties at any point x in the object volume, where f is the total linear attenuation coefficient and μ_s is the linear attenuation coefficient for scatter only, ρ is the density, dσ/dΩ is the differential scatter cross-section per solid angle, etc.

#### 3.1.2 Computational complexity

In order to point out the power of dimensional reduction, we anticipate a rough estimation of the computational expense of the different approaches, which will be discussed later. For the different realizations of Eq. (1), we assume that both the object volume and the projection image (in any pro- jection view) are discretized in each coordinate direction by an order of n [denoted by O(n^1)] voxels and pixels, respectively, i.e., the volume contains O(n^3) voxels and the projection image O(n^2) pixels. Standard ray-tracing from an arbitrary object voxel to an arbitrary detector pixel takes about O(n^1) voxels. Consequently, volume-based algorithms applying ray-tracing for single-scatter simulation usually need O(n^3)×O(n^1)×O(n^2)=O(n^6) basic operations. However, the approximative projection-based algorithms have the benefit of being much faster, due to O(n^4) with scatter-beam-spread functions or convolution approaches, the latter may be accelerated to O(n^2 log n).

The high order dependency on discretization underscores the efficiency of downsizing by reducing discretization as much as justified by the smoothness of scatter distributions.

### 3.2 3D (image-based) interpretation

Starting from the Boltzmann integro-differential transport equation, Inanc has given an integral-kernel formulation for single-scatter and an iterative scheme for multiple-order scatter fluence. However, we will present in Sec. 3.2.1 a compact integral-kernel formulation for multiple-order scatter, a specialization of which is the single-scatter model (Appendix A). Our representation is similar to ideas for simulation and correction of scatter in emission tomography where the radiation source, contrary to transmission CT, is distributed in the interior of the scattering object. However, in transmission CT every object volume element at point x acts as a local scatter source by interacting with the primary radiation at x.

#### 3.2.1 Point-scatter-kernel model

The idea is adopted from emission tomography where a Monte Carlo generated system matrix approach had been proposed by Floyd et al., see also the concept of point scatter distribution functions by Msaki et al. Our continuous description corresponds to the discretized system matrix representation. Propagating from an arbitrary volume element at point x, where the first scatter event is assumed to occur, the spatial distribution of detected scattered radiation at the detector as a function of the detector pixel coordinates u is described by [Fig. 2(b)]

$$K_f(x,u)$$(3)

which we will name Point-Scatter-Kernel (PSK). It also comprises further multiple scatter events between x and point of detection u [as indicated in Fig. 2(b) by a bended dashed arrow]. In general, the kernel depends on the projection angle α, on the acquisition parameters and on physical properties of the object as indicated by f in Eq. (2). It cannot be calculated in closed analytical form but might be obtained by Monte Carlo simulation. The PSK concept is introduced as an intermediate step toward the 2D BSK approach in Sec. 3.3.1. The normalized detected scatter fluence results as a superposition integral

$$S(u) = \int \int \int_{V_f} ψ(x) μ_s(x) K_f(x,u) dx$$(4)

which is an example of the general form of Eq. (1). V_f is the domain of integration of the object, depending on the collimation [Fig. 2(a)]. ψ(x) is the primary fluence at x, including the attenuation within the object before the scatter event at x, as described in Appendix A 1. μ_s(x) is the linear attenuation coefficient for scatter at x. This means that

$$(Tψ)(x) = ψ(x) μ_s(x)$$(5)

is the volume source density for scatter at x.

**3.2.1.a. System matrix approach**. Sauve et al. presented an example in micro-ECT with a very long calculation of the system matrix, which is the discrete version of expression (3). At present, it does not seem practical in diagnostic CBCT. However, looking toward the far future, one should not be deterred by “megalopinakophobia,” as recommended by Barrett et al.

#### 3.2.2 Analytic single-scatter model

The general concept includes multiple scatter. A common two-step approach is a single-scatter model followed by an estimate for the contribution of multiple scatter. We shift the description to Appendix A.

Approximations by Rinkel et al. and Yao and Leszczynski aim at reducing 3D-integration in a single-scatter approach of Eq. (4) to 2D-integrations; see Appendix A 3.

### 3.3 2D (projection-based) interpretation

#### 3.3.1 From 3D to 2D: Beam-scatter-kernels BSK approach

Keeping in mind, Table I suggests itself dimensional reduction. The underlying idea is to change the integration over the volume V_f in Eq. (4) by integrating along “beams,” which connect the x-ray point source and each detector pixel, and summing up the contributions of all beams. The beams can be thought of as pencil beams. The integration volume V_f is the union of the set of all beams of the cone on the CBCT acquisition geometry. The integration in Eq. (4) over a beam [Fig. 3(a)] delivers the image of scattered radiation caused by all primary radiation within the beam and is closely related to the notion of BSK. BSKs can be measured or simulated by Monte Carlo. Theoretically, a general (spatially varying and object-dependent) BSK is based on Eq. (4) reducing from 3D to 2D dimension

$$H_f (u', u) = \frac {1} {(TΦ)(u')} \int_{b(u')} ψ(x) μ_s(x) K_f (x,u) dx$$(6)

where the integration domain b(u') is defined by the beam. The purpose of dividing by (TΦ)(u') is an appropriate scaling of the BSK, see Sec. 3.3.2 below. The term Φ(u') denotes the normalized fluence at the detector and is redefined by Φ(u') = ψ(x_d, u'), where x_D is the distance of the detector from the focal spot of the x-ray source in a local source-detector coordinate system [Fig. 3(b)]. As derived in Appendix A, by performing the integration in Eq. (4) for all beams, the 3D integral is reduced to a 2D integral over the effective detector area D [Fig. 2(a)],

$$S(u) = \int \int_D (TΦ) (u') H_f(u',u) du'$$(7)

which is again of the general form of Eq. (1). We have written explicitly the 3D integrals in Eq. (4) and the 2D integrals in Eq. (7) to emphasize the dimensional reduction. In the following, we will prefer the short-hand notation using one common integral symbol.

#### 3.3.2. Scatter potentials or scatter source term scaling

In the literature different sclaing functions TΦ in Eq. (6), respectively, scatter source term in Eq. (7), are applied. The concept is similar to that of "scatter potential" by Rinkel et al. The open field representation, 

$$TΦ = Φ_0$$(8)

was introduced by Swindell and Evans and Hansen et al., where Φ_0(u') denotes the distribution (at the detector) of normalized fluence without object, i.e., Φ_0(u') = 1 without bow-tie filter, otherwise Φ_0(u') related to the transmission profile of a bow-tie filter projected at the detector plane. In the following, we omit the bow-tie filter version but come back to int in Sec 4.4. Frequently, the normalized primary intensity representation is used

$$TΦ = Φ$$(9)

Since the amplitude of the beam-scatter-kernels, which correspond to the open field or the primary intensity scalings, increases strongly with object thickness, the following so-called forward-scatter representation (see also Sec. 4.2.2):

$$TΦ = -Φ lnΦ = φ exp(-φ)$$(10)

It has the advantage that the corresponding BSK amplitudes depend only weakly on the path-length within the scatter object. In Eq. (10), φ = -ln Φ is the CT projection value at the detector.

A more general modified forward-scatter approach was introduced by Ohnesorge et al.

$$TΦ = Φ^{β_1} |ln Φ|^{β_2}$$(11)

where β_1 and β_2 are empirical parameters obtained by data fitting. This model was recently adopted by Star-Lack et al. and Sun and Star-Lack, who use the term scatter amplitude factor for the expression $Φ^{β_1 - 1} |ln Φ|^{β_2}$.

Note: Due to the definitions in Eqs. (6) and (7), the different scaling functions TΦ correspond to equivalent representations of the scatter distribution in Eq. (7).

## 4. Projection-based scatter kernel approaches

Approximations are aiming at facilitating the evaluation of Eq. (7). The large variety of approximations justifies to be treated in a separate section.

### 4.1 Thickness related BSK approach

#### 4.1.1 Empirical beam-scatter-kernels

The integral representation Eq. (7) also holds with empirical beam-scatter kernels Hf, which can be measured or calculated by Monte Carlo simulations. The idea was first introduced in mega-voltage imaging by Swindell and Evans and Hansen et al., where the notion pencil-beam-scatter-kernel was used. The kernels were generated by MC simulation for homogeneous water slabs as function of slab thickness and correspondingly for water disks with different diameters by Spies et al. The method was adopted for kilovolt CT applications by Maltz et al., Reitz et al., and Wiegert et al. An edge-spread function method to obtain scatter kernels by measurements is proposed by Li et al.

### 4.1.2 Thickness related BSK superposition method

The kernel approach in Eq. (7) is exempted from its specific object dependency by the assumption that the scatter contribution of a primary pencil beam depends only on the integrated material along the path, which can be expressed by the radiological or water-equivalent thickness w-e-t t(u') = t(Φ(u')) traversed by the primary beam. Lateral changes in the neighborhood of each beam are neglected.The w-e-t is defined as the water-thickness, which causes the same effective primary attenuation Φ, including spectral hardening and energy dependent detector response. The direct and iterative calculation of “thickness maps” t(u') corresponding to primary “attenuation maps” Φ(u') is, e.g., described in our companion paper (Ref. 1, p.4308). Replacing in Eq. (7), the index f with t(u') and the argument (u', u) with (u-u') by centering the kernel function at u', we obtain

$$S(u) = \int_D (TΦ)(u') H_{t(u')} (u-u') du'$$(12)

In the literature, this has been called scatter-kernel-superposition method. Equation (12) is not a stationary convolution, since the kernel can change with each pixel u' and, therefore, is not spatially invariant in general.

By normalization of the kernels, a more convenient formula,

$$S(u) = \int_D (\tilde TΦ)(u') h_{t(u')} (u-u') du'$$(13)

is obtained, where the following abbreviations are used:

$$(\tilde TΦ)(u') = (TΦ)(u') w_h(t(u'))$$(14)

$$w_h(t) = \int H_t(u) du$$(15)

$$h_t(u) = \frac {1}{w_h(t)} H_t(u)$$(16)

Equation (14) can, in the case TΦ = Φ, be interpreted as a weighting of the primary fluence by the scatter-to-primary-ratio corresponding to a w-e-t of t, i.e., Eq. (14) has the meaning of a scatter fluence. In Eqs. (13) and (16), the normalized kernels h_t with integral = 1 are smoothing kernels, whose smoothing widths are increasing with local w-e-t, which may vary from pixel to pixel.

#### 4.1.3 Kernel parameterizations

Some authors prefer to store MC generated rotationally symmetric beam-scatter-kernels for slabs and circular disks of stepwise increasing thickness in data bases, e.g., Maltz et al. and Reitz et al. Other authors would fit the kernels to appropriate parametric functions, e.g., Star-Lack et al., Sun and Star-Lack, and Meyer et al. by a linear combination of two Gaussians with differenet widths and amplitutes. In radiography Love and Kruger had investigated a variety of parametric scatter-kernels with images from different anthropomorphic phantoms and found that 2D exponential kernels fitted best. Lo et al. had found, using neural network techniques, that scatter kernels resembled exponential shape in lungs and Gaussian shape in denser regions. Hinshaw and Dobbins modeled a rather complicated kernel function with four tunable parameters to adapt local smoothing width and kernel shape (varying between pure exponential and Gaussian) depending on local fluence.

Empirically, the width and amplitude have much more influence than the specific shape of symmetrical kernels. Practically, the computational efficiency is a crucial point. In the case of convolution approximations, Gaussian kernels are often preferred due to their separability, i.e., 2D-convolution can be reduced to two successive 1D-convolutions in row and column direction.

### 4.2 Convolution approaches

#### 4.2.1 Scatter-kernel convolution approximation

An approximation as a true (spatially invariant) convolution is straightforward on the grounds of Eqs. (13)–(16), by replacing the t-dependent kernels h_t with a unique common kernel h = h_{t_0} for a constant (e.g., mean) t = t_0. Consequently, Eq. (13) becomes the convolution formula as follows:

$$S(u) = ((\tilde TΦ)**h)(u) = \int (\tilde TΦ)(u') h(u-u') du'$$(17)

where ** denotes 2D convolution. As is well-known, the importance of convolution is due to efficient processing by FFT.

#### 4.2.2 Forward-scatter convolution approach

A fundamental relation stating that in a simplified strictly forward-single-scattering model, the scatter-to-primary-ratio is determined by the expression given in Eq. (10) had been observed independently by Hangartner, Swindell and Evans (p.72), and by Ohnesorge et al. The expression TΦ = -Φ lnΦ in Eq. (10) was named forward-scatter-function. The forward-scatter approach represents the scatter fluence by a convolution of the appropriately weighted forward-scatter-function with a scatter smoothing kernel. The convolution kernel may vary with object size or clinical application (e.g., different kernels for head, chest, and abdominal studies). Spatially variant generalization based on thickness-dependent scatter smoothing kernels is obvious, as used by Meyer et al.

Examples of clinical images are given in Zellerhoff et al. and Reiser et al. (pp. 39-40)

#### 4.2.3 Backflash: Convolution approaches in radiography

Since the digital revolution in diagnostic radiology, a lot of papers dealing with heuristic convolution/deconvolution approaches for modeling and correcting scatter effects (together with glare effects in image intensifiers) in projection images appeared, which in our notion Eq. (17) would correspond to the simple approach of a spatially invariant kernel to be convolved with the primary fluence distribution Eq. (9) (e.g., Shaw et al., Naimuddin et al., Molloi and Mistretta,Boone et al., Boone and Seibert, Love and Kruger, and others). Generalizations beyond the standard spatial invariance model through variable weighting were introduced by Molloi and Mistretta and Naimuddin et al., regionally varying convolution kernels by Kruger et al., and combination with thickness estimation by Ersahin and Molloi et al.

### 4.3 Object dependency and nonstationary beam-scatter-kernels

#### 4.3.1 Parametrical account for object dependency

A problem of the BSK superposition approach in the diagnostic kilovolt range is that the beam-scatter-kernels will depend on inhomogenities in the neighborhood of the beam, in general on the lateral size of the object (perpendicular to the beam). An instructive example is given by Maltz et al. (Fig. 2), where scatter kernels for 6 MV and 121 kV x-ray beams and water thickness 25 cm are shown, and in the case of 121 kV for an infinite slab and for a disk with 16 cm diameter. The latter is no longer monotonically decreasing but has wings. To account for this, the scatter-kernel in the BSK-superposition formula, Eq. (12), is updated by an additional parameter, i.e., the apparent lateral diameter of the object.

#### 4.3.2 Regionally varying BSK convolution approaches

**4.3.2.a. Spatially varying weighting**. In agreement with Molloi and Mistretta, the weighting function $w_h(t)$ in Eq. (14) can be understood as a local SPR, a multiparametric function which depends on local thickness t(u') but may also depend on nonstationary contributions such as thickness changes. The importance of thickness changes is recognized there, though a method how to deal with thickness gradients is missing. We will discuss an explicit method in Sec. 4.3.3.

**4.3.2.b. Regionally varying convolution kernel**. To account for different scattering properties in lung, mediastinum, and spine regions in chest radiography, Kruger et al. proposed to apply regionally adapted specific scatter kernels. The resulting scatter distribution is a superposition of weighted contribution from the different regions; the weighting functions are pixel dependent in order to obtain smooth transitions between abutting regions and to prevent discontinuities. Star-Lack et al. and Sun and Star-Lack have used a similar method in their CBCT scatter correction scheme. They create the different "regions" by segmentation with respect to contiguous w-e-t intervals, which are called “thickness groups.”

#### 4.3.3 Continuously and asymmetrically varying BSK approach

Object dependency if recognized as a global property can be accounted for by object-size dependent kernels or weighting factors, but beam-scatter kernels remain symmetric. However, the observation that scatter kernels are changing shape and become asymmetric depending on the gradient of w-e-t thickness poses a more challenging problem. Asymmetric kernels have been reported, e.g., by Lubberink et al., Star-Lack et al., and Sun and Star-Lack. The issue of spatially varying and asymmetric scatter kernels has been addressed by the latter, who have measured the BSKs across a wedge phantom and compared with the symmetric BSKs of slab phantoms of corresponding thickness. We present a slight generalization of their deformation approach to generate asymmetric kernels from symmetric ones by

$$\tilde h(u', u) = a(u') h(u-u') (1 - γ[t(\hat u_λ)-t(u')])$$(18)

where $\tilde h$ is the resulting asymmetric kernel, h is a symmetric kernel, γ and λ are empirical scalar  parameters. We have introduced the additional factor a(u'), which is missing in the original paper, in order to take into consideration that the kernel amplitude may change with the deformation. There are two different implementations of Eq. (18):

(a) The special case of $\hat u_λ = u$ first introduced in the former paper was later called "fast adaptive scatter kernel" method, since it can be shown that substituting h(u-u') in formula Eq. (17) by $\tilde h(u', u)$ from Eq. (18) results simply in an expression composed of two stationary convolutions.

(b) In the more general "adaptive scatter kernel" method as formulated in Eq. (18), the term $\hat u_λ$ is a point linearly interpolated between u and u', i.e., defined by $\hat u_λ = λu + (1-λ)u'$, where 0≤λ≤1 is an interpolation parameter, which controls an extent of spatial stretching of the scatter distribution. This model reduces no longer to convolutions (unless $\hat u = u$, i.e., λ=1), however, proved superior to the model (a) due to more accurate fitting to true scatter distributions.

Notes:

1. Parameters γ, λ, and weighting function a depend on "thickness groups."

2. Our approach is more general than in the cited literature where the deformation model was derived specifically for the modified forward-scatter model corresponding to formula in Eq. (11) and parameterization of BSK by a linear combination of two Gaussian kernels.

#### 4.3.4 Edge effects

Near object borders not only strong asymmetric deformation occur, but also amplitudes of beam-scatter kernels decrease due to lack of scattering material outside the border. This is taken into account by Sun and Star-Lack by multiplication of the "scatter source term" $(\tilde TΦ)(u')$ in Eq. (13) with an empirical windowing function g(u')≤1 near the border. In our more general approach of Eq. (18) this window function can be put together with a(u').

### 4.4 Energy dependency with Bow-tie filter

A bow-tie filter in front of the x-ray source has the effect of homogenization of the attenuation profile at the detector: it compensates the shorter x-ray paths near the border of the object section with the consequence of dose and scatter reduction, efficient use of dynamic range, and reduction of beam-hardening artifacts.

With a bow-tie filter Eq. (12) has to be modified by

$$S(u) = \int_D (TΦ)(u') Φ_0(u') H_{t(u'), τ_B(u')} (u-u') du'$$(19)

where τ_B(u') denotes the pathlength within the bow-tie material passed by the primary x-ray, which reaches at detector pixel at u', and Φ_0(u') denotes the attenuation (corresponding to Beer's exponential law) of the primary intensity due to that length τ_B(u') by the bow-tie material. Note that for different lengths τ_B, the initial polychromatic photon energy spectrum suffers different spectral hardening. Due to this spatially varying spectral prefiltering by the bow-tie filter, the corresponding scatter kernels depend on two parameters t, τ_B. This double-parametric collection of kernels ($H_{t, τ_B}$) can be obtained by Monte Carlo simulation or by measurements.

### 4.5 External scatter

Although by collimation, the x-ray cone is restricted to the volume-of-interest to be imaged, it is unavoidable that the object is positioned on a table, which causes additional scattered radiation. This can be neglected in most applications in diagnostic CT, but with offset detector scans in image-guided radiotherapy where larger patient tables are used significant scatter-related artifacts can occur. Image uniformity and HU accuracy were greatly improved by an algorithm by Sun et al. applying separate kernels to estimate scatter from the table and the patient.

### 4.6 Detector scatter and glare

A low-frequency signal drop in flat-panel detectors attributed to scattering of incident x-ray photons in the detector panel and housing has been discussed, e.g., by Rinkel et al. and Poludniowski et al. It is called "detector glare" in analogy to "veiling glare" in image intensifiers by diffusion of optical phontons in the scintillator. Detector glare can be corrected by deconvolution. Correcting object scatter in CBCT is not enough for quantitative purposes; scatter glare must also be accounted for.

### 4.7 Some open issues

#### 4.7.1 Internal absorbers

Internal absorbers, such as operation clips, platinum coils, and metalic implants may block primary x-rays to reach the detector. Such missing data are usually corrected for by sophisticated metal artifact removal (MAR) algorithms. However, scattered radiation might feign semitransmittant material in spite of opaque absorbers. On the other hand, nonzero fluence in the shadow of an opaque absorber would offer an estimation of scatter fluence in analogy to the beam-stop method. Further investigations have not come to our knowledge, but seem to be worthwhile.

#### 4.7.2 Dependency on Material Distribution along the Ray

In Sec. 4.3, methods to come up with changes of water-equivalent thickness in the lateral neighborhood of the beam have been discussed. However, object dependent changes by materials other than soft tissue, such as bone, are not taken into account. Furthermore, the effect of longitudinal changes of density and material along the ray has not been investigated up to now. Now we give an example.

Scatter intensity is different when the ray first penetrates the bone before penetrating the soft tissue, compared with the opposite succession of scattering and absorbing materials. The plots in Fig. 4 show the scatter intensity profiles for two different material distributions. Although the primary intensity profiles are the same (neglecting noise), scatter intensity is much higher (and the profile is slightly broader) for the succession bone-first and water-last than for water-first and bone-last. The posterior bone absorbs much more from the radiation scattered down to lower energies by the anterior water layer, than conversely. This phenomenon shows the limitations of projection-based approaches, where the inner composition of the object is not taken into consideration. However, this is possible in 3D image-based analytical and Monte Carlo methods, which are to be applied after—albeit approximate—3D image reconstruction of the object.

## 5. Image-based, iterative, hybrid and heuristic approaches

### 5.1 Geometric model-based methods

Wiegert et al. proposed to fit the reconstructed object to an ellipsoid, and in a second reconstruction, the projection data are corrected by a constant scatter background which was previously stored in a data-base for a large variety of water-equivalent ellipsoids. In later papers the method was improved.

Meyer et al. applied a BSK-approach using a forward-scatter-model and parametric kernels with two Gaussians. After a first reconstruction, the object size is fitted to an elliptic cylinder, and the corresponding Gaussian kernel parameters are searched from a database. Then, the BSK-convolution is performed. Using the database an optimized scaling factor is determined, such that the scatter magnitude of the convolution model fits best with the data from the database. Empirically, a projection-based object size estimation algorithm was superior to the image-based method. Also a stable method for truncated data was demonstrated.

A model-based method for estimation and correction of cross-scatter in dual-source CT is presented by Petersilka et al.

### 5.2 Analytic and mixed approaches

Analytic models have already been outlined in Sec. 3.2.2 and Appendix A.

Kyriakou et al. investigated the idea to calculate single-scatter fluence by coarsely discretized analytic integration and to simulate the very smooth (in the mean) multiple-order scatter contribution by accelerated Monte Carlo.

### 5.3 Accelerated Monte-Carlo

Given a radiation source and an object, i.e., a spatial distribution of linear attenuation coefficient, the Monte-Carlo method can be considered as a numerical approach to solve the Boltzmann transport equation. In a postreconstructive iterative framework, the object will be a successively updated 3D-image as output of a cone-beam reconstruction algorithm. The mathematical–physical operator, which describes the acquisition of projection data including scatter, is realized by a Monte-Carlo program. This shows obviously that the acquisition operator itself is object-dependent.

The usefulness of MC driven scatter simulation for scatter correction has been shown by Jarry et al. Strong downsampling of voxels in object volume, and pixels at the detector, and number of projection views is mandatory. The ultimate "screw" to accelerate computation is dramatic reduction of the number of photon trajectories propogated through the volume, however, with the drawback of large spatial fluctuations. Therefore, efficient denoising techniques have been developed by Kawrakow et al., Mainegra-Hing and Kawrakow (using adaptive Savitzky–Golay filtering), Colijn and Beekman, and Zbijewski and Beekman 88(applying the Richardson–Lucy method). Mainegra-Hing and Kawrakow86,89 state an acceleration factor of about 10 by their denoising method. Zbijewski and Beekman reduced the number of photon tracks from ≥ 10^8 down to 10^4 - 10^5. Poludniowski et al. proposed the combination of very coarse discretization and a special fixed-forced-detection method to further reduce number of photon tracks.

### 5.4 Nonphysical heuristic correction approaches

Heuristic postprocessing image corrections are aimed at reducing cupping effects and other artifacts in reconstructed images phenomenologically, without using an explicit physical scatter model, and seem to be useful to mitigate residual artifacts, which remain after application of scatter correction algorithms based on simplified physical models. We have already given references in our previous paper. Wiegert et al. combined in an iteration loop projection-based scatter correction with image postprocessing for elimination of cupping artifacts.

Prior-image constraint correction approach was proposed by Kachelriess et al. and Brunner et al. The image aimed at is obtained from the CBCT reconstruction of a polynomial expansion in powers of the projection data, where the polynomial coefficients are optimized by minimization of the mean square difference with respect to an artifact-free prior template image.

## 6. Miscellaneous Applications

### 6.1 Offset detector geometry

Off-centric positioning of the detector is a method to extend the field-of-view of reconstruction with a full rotation (360°) scan or with two successive scans. A nontrivial problem is the estimation of scatter distribution at the discontinuity caused by the detector border, which cuts the interior object shadow. The discussed method by Star-Lack et al. and Sun and Star-Lack proved successful to tackle the problem of half-fan detector geometry.

### 6.2 Truncated object

Truncated projections due to limited detector size cause artifacts and object density deformations in the reconstructed volume. Bertram et al. developed object completion methods such that reasonable results are obtained by image-based Monte-Carlo estimation of scatter distributions. The method of Meyer et al. also worked satisfactorily with truncated data.

### 6.3 Dual-source CT

Dual-source CT consists of two source-detector systems, which are about 90° apart. If both x-ray sources are used with the same voltage, acquisition time can be reduced by almost a factor of two. With different voltages, typically 80 and 140 kVp, fast dual-energy CT imaging is realized. A main problem with synchronical dual-source CT is, in addition to the direct scatter of each system, cross-scatter arising from the other system. Image quality has been studied by Kyriakou and Kalender, Engel et al., and Petersilka et al., the latter have also developed an efficient online cross-scatter correction method.

### 6.4 Dual-energy imaging

The same scatter correction methods as used in standard single spectrum CT can be applied in dual-energy CT; of course, the different spectra have to be taken into account. Two-component or multicomponent material decomposition and quantitation make high demands on scatter correction accuracy. The impact of scatter in dual-energy CT has been studied by Wiegert et al. In synchronic dual-source CT systems the additional cross-scatter demands appropriately adapted scatter correction procedures.

## 7. Synopsis

Scatter management is one of the most important issues in large-detector volume CT. First of all scattered radiation should be suppressed by hardware prepatient devices (e.g., collimator) and postpatient means (e.g., air-gap) to a level as low as reasonably achievable. Antiscatter grids are discussed controversially; however, most researchers agree that ASGs are favorable concerning contrast-to-noise-ratio with thick objects.

散射管理是大探测器体CT中的一个最重要问题。首先，散射的辐射应当用到达患者前的硬件设施（如，准直器），和患者后的方法（如，空气间隔），来抑制到一个尽可能低的水平。反散射网格也进行来讨论；但是，大多数研究者同意，在考虑到有很厚的物体的CNR时，ASGs是很好的。

We have discussed a multitude of scatter estimation approaches, including measurements and mathematical–physical models. Although the beam-stop technique is a standard method of scatter measurement in the laboratory, the online BAA method has not attained a break-through due to the inconvenience of mechanics and propensity to residual artifacts, which is also true of the primary modulation method. A final judgment of measurement-based methods is scarcely possible up to now, since improvements of the techniques are still topics of research.

我们讨论了很多散射估计方法，包括测量的，和数学物理模型。虽然beam-stop技术是实验室中散射度量的标准方法，在线BAA方法由于机械方面不方便，很容易有残余的伪影（这对于primary modulation方法也是这样），还没有得到技术突破。对于基于度量的方法的最终评价现在基本不太可能，因为这些方法的改进仍然在研究中。

The choice of the appropriate and optimal scatter correction method depends widely on the accuracy, speed, and limitations of the acquisition and imaging system and on the application task, e.g., the demands of accuracy are increasing between CBCT imaging for image-guided radiotherapy, C-arm CBCT for interventional procedures, and finally high-end CT. On the other hand, there are specific preferences by different groups of researchers.

选择合理的散射修正方法，取决于成像系统的准确率、速度和限制，以及不同的任务，如，在图像引导的放射治疗中的CBCT成像对准确率要求就高，还有介入治疗中的C形臂CBCT，和高端CT。另一方面，对不同类型的研究者有具体的倾向。

With the simple collimator-shadow method an accuracy of 30 HU in head scans was achieved and a reduction of cupping by a factor of >2 in a body phantom. Simple software convolution approaches can compete with this modest accuracy, which might be sufficient for less demanding applications.

用简单的collimator-shadow方法，在头部的扫描中可以获得30 HU的准确率，在体部模体中，cupping伪影可以降低>2。简单的卷积方法，可以达到一般的准确率，这对于需求没有那么高的应用，是很充分的。

In order to achieve optimal results of higher accuracy, the scatter estimation models have to be as close to reality as reasonably possible, and the scatter correction procedures have to take into account the stochastic nature of signal and scatter in a statistical framework.

为了得到更好的效果，散射估计模型需要尽量与实际相符，散射修正过程需要考虑到信号与散射的随机本质，在统计的框架下进行求解。

With sophisticated iterative approaches, a reduction of scatter-induced artifacts (i.e., cupping) by a factor of 10 and HU-accuracy about 10–20 HU can be achieved. Rinkel et al. claimed that the accuracy of their analytical method competes with beam-stop measurements; however, the statement that the latter demand nine times higher dose is due to their special experimental setup but is disproved by Ning et al., who stated an increase of total exposure by only 4% due to an additional low-dose acquisition using a BAA.

在复杂的迭代过程下，可以降低10倍散射引入的伪影（如，cupping），HU-准确率可以降低大约10-20 HU。Rinkel等宣称，其解析方法的准确率与beam-stop测量接近；但是，后者需要9倍高的剂量，这是因为其特殊的试验设置，Ning等人不同意这样的观点，表明总辐射量增加了4%，由于使用BAA有了额外的低剂量获取装置。

With respect to computational efficiency of mathematical–physical models, the beam-scatter-kernel approach is the most promising, see Table I. Its potentials are not yet exhausted. We see at least two directions of further research. First, the modeling of asymmetric BSK distortion depending on 2D fluence gradients has to be further investigated and refined. Second, longitudinal effects of BSK dependency on material distribution along the ray and neglected in the literature until now should be incorporated. However, the achievable accuracy is limited by fundamental assumptions and approximations within the BSK model.

在数学-物理模型的计算效率上，BSK方法是最有希望的，见表1。其可能性并未被完全探索。未来至少有2个研究方向。第一，非对称BSK形变的建模，依赖于2D通量的梯度，需要进一步研究和提炼。第二，BSK对物质分布沿着射线方向径向效果的依赖应当纳入考虑。但是，可获得的准确率受限于基本的假设，和BSK模型的近似。

With Monte Carlo programs being closest to reality the limitations of physical modeling can be subdued. However, to overcome the computational demands MC methods are compromised by coarse discretization, acceleration, and fitting techniques. There is hope that progress in fast computing technology will make online MC simulation possible once upon a time. Up to now hybrid approaches combining measurements, 2D projection-based BSK, 3D geometric model based, and (more or less accelerated) Monte Carlo based approaches might be the best possible solution for high accuracy scatter estimation.

MC程序仿真与真实情况最接近，可以抑制物理建模的局限。但是，为克服MC方法的计算需求，需要进行粗糙的离散，加速和拟合技术。快速计算技术可能会使在线MC仿真成为可能。直到现在，将测量、基于projection的2D BSK，3D几何建模，和（或多或少加速的）基于MC的混合方法，可能是对于高精度散射估计来说最有可能的解决方案。