# A general framework and review of scatter correction methods in x-ray cone-beam computerized tomography. Part 1: Scatter compensation approaches

Ernst-Peter Ruhrnschopf et. al. Siemens

Since scattered radiation in cone-beam volume CT implies severe degradation of CT images by quantification errors, artifacts, and noise increase, scatter suppression is one of the main issues related to image quality in CBCT imaging. The aim of this review is to structurize the variety of scatter suppression methods, to analyze the common structure, and to develop a general framework for scatter correction procedures. In general, scatter suppression combines hardware techniques of scatter rejection and software methods of scatter correction. The authors emphasize that scatter correction procedures consist of the main components scatter estimation (by measurement or mathematical modeling) and scatter compensation (deterministic or statistical methods). The framework comprises most scatter correction approaches and its validity also goes beyond transmission CT. Before the advent of cone-beam CT, a lot of papers on scatter correction approaches in x-ray radiography, mammography, emission tomography, and in Megavolt CT had been published. The opportunity to avail from research in those other fields of medical imaging has not yet been sufficiently exploited. Therefore additional references are included when ever it seems pertinent. Scatter estimation and scatter compensation are typically intertwined in iterative procedures. It makes sense to recognize iterative approaches in the light of the concept of self-consistency. The importance of incorporating scatter compensation approaches into a statistical framework for noise minimization has to be underscored. Signal and noise propagation analysis is presented. A main result is the preservation of differential-signal-to-noise-ratio (dSNR) in CT projection data by ideal scatter correction. The objective of scatter compensation methods is the restoration of quantitative accuracy and a balance between low-contrast restoration and noise reduction. In a synopsis section, the different deterministic and statistical methods are discussed with respect to their properties and applications. The current paper is focused on scatter compensation algorithms. The multitude of scatter estimation models will be dealt with in a separate paper.

由于在CBCT中散射辐射意味着CT图像的严重降质，表现为量化误差，伪影，和噪声增加，在CBCT成像中，与图像质量相关的主要问题就是散射抑制。本文的目的是整理出各种散射抑制方法的结构，分析通用结构，提出散射修正过程的通用框架。一般来说，散射抑制是硬件上的散射抑制和软件上的散射修正。作者强调了，散射修正过程主要包括了散射估计（通过测量或数学建模），和散射补偿（确定性的或统计的方法）。框架包含了多数散射修正方法，其有效性超越了传输CT。在CBCT出现之前，在X射线放射成像，乳腺X射线成像，发射断层成像，和MV CT应用中发表了很多散射修正的文章。在其他医学成像领域中应用利用这些研究尚未得到充分的探索。因此在需要的时候，我们包括了其他参考文献。散射估计和散射补偿通常是迭代交叠在一起的。在噪声最小化的统计框架中，纳入散射补偿方法的重要性，是被低估的。给出了信号分析和噪声传播分析。一个主要结果是，通过理想的散射修正，在CT投影数据中保持dSNR。散射补偿方法的目标是，恢复定量准确率，在低对比度恢复和噪声去除中保持平衡。在总结中，讨论了不同的确定性和统计方法，及其性质和应用。本文聚焦在散射补偿算法。各种散射估计模型在另外一篇文章中讨论。

**Key words**: cone-beam CT, image quality, scatter correction, scatter suppression, differential signal-to-noise ratio, contrast-to-noise ratio, iterative correction, consistency, iterative reconstuction, maximum likelihood reconstruction, Bayes reconstruction, noise propagation, noise filtering, statistical optimization

## 1. Introduction

### 1.1 Review of scatter relevance in CT

In the last decade, cone-beam CT (CBCT) has become increasingly important due to development and improvement of flat-panel detectors (FPDs), new acquisition and scanning techniques (helical CT with multirow detectors; C-arm systems with FPD, robotic control), versatile clinical applications (from diagnostics to intervention), accompanied by novel approaches and improvements of mathematical reconstruction algorithms (e.g., with acquisition geometries allowing exact reconstruction formula). On the other hand, the drawback of high scattered radiation levels caused by large irradiated volumes has become a challenging issue related to image quality with CBCT.

在过去十年中，由于平板探测器(FPDs)、新的获取和扫描技术（多列探测器的螺旋CT，带FPD的C形臂，机器人控制）、多样化的临床应用（从诊断到介入）的开发和改进，以及数学重建算法的新方法和改进（如，在已知获取参数时，可以得到精确的重建公式），CBCT已经变得越来越重要。另一方面，大型辐照体的高散射辐射水平的缺点，已经成为了CBCT成像质量的一个有挑战性的问题。

Although scatter in CT was addressed at first in 1976 by Stonestrom and Macovski, followed by some basic papers in 1982 by Johns and Yaffe, Joseph and Spital, and Glover, scattered radiation played a negligible role with single-slice third generation fan-beam CT scanners due to detector septal collimation and restriction of the scattering volume. Scatter got some attention in fourth generation inverse geometry CT with a stationary detector ring where septal collimation is restricted. However, the development of multislice CT scanners and CBCT volume reconstruction with area detectors triggered an increasing interest in scatter correction techniques since the 1990s and was boosting publications in the first decade of the new century.

虽然在CT中的散射在1976年就得到了处理，然后是1982年的一些文章，在单slice的第三代扇形束CT扫描器时，散射辐射的角色可以忽略，由探测器有单独的准直，并限制了散射的体。在第四代逆几何CT时，散射得到了一些关注，因为检测器环是静态的，这限制了单独的准直。但是，自从1990s，多slice CT和CBCT体重建的开发，都带有面探测器，这对散射修正都有很高的要求，促进了新世纪第一个十年的文献发表。

Extending previous studies cited above, one of the most comprehensive study about scatter effects in CBCT is that by Siewerdsen and Jaffray. As yet known from x-ray radiology, scatter fluence may exceed primary fluence by an order of magnitude and may still be in the order of primary fluence even after application of hardware scatter rejection devices such as antiscatter grids. This underscores the importance of additional application of software scatter correction. In CT images, scattered radiation impairs image quality by deterministic inhomogenity and quantification errors and by increased noise. The deterministic errors are showing up as (a) spatial low frequency grey value deformations, known as cupping; (b) streaks, bars, or shadows, particularly in the vicinity and between highly absorbing regions; (c) decrease of soft tissue contrasts. The cupping and shadowing artifacts look similar to those from beam hardening but are often more severe.

将上述的之前研究进行拓展，关于CBCT中的散射研究的最广泛的研究之一是Siewerdsen and Jaffray。从X射线放射成像可知，散射通量可能会超过初级通量一个量级，即使使用了硬件的散射抑制设备，如反散射网格，仍然会和初级通量在一个量级上。这强调了应用软件散射修正的重要性。在CT图像中，散射辐射损害了图像质量，包括确定性的不均一性，量化误差和增加的噪声。确定性的误差展示为，(a)空间上的低频灰度值变形，如cupping，(b)条纹，条状，或阴影，尤其是在高吸收区域的附近和之间，(c)降低软组织对比度。对于射线硬化得到的，cupping和阴影伪影看起来很相似，但通常会更严重。

Long before CBCT, a lot of papers on scatter correction approaches in x-ray radiography, mammography, emission tomography, and in Megavolt CT for radiation therapy had been published. So diagnostic CBCT could synergetically avail from research in other fields of medical imaging, an opportunity which, to our knowledge, has not yet been exploited sufficiently.

CBCT之前很久，发表了很多在X射线放射成像，乳腺X射线成像，发射断层成像和MV CT中的散射修正方法文章。所以诊断级的CBCT会从其他医学成像领域中获益，据我们所知，还并没有得到充分利用。

### 1.2 Organization and aim of the article

At first, we will discuss and propose the terminology to be used in this article. Different terms have been used in the literature to label techniques aiming at reducing scatter and its effects on image quality: scatter compensation, control, correction, management, mitigation, reduction, rejection, removal, subtraction, or suppression. We will prefer the term scatter suppression for both hardware means of scatter rejection and software means for scatter correction. A crucial observation is that the structure of scatter correction algorithms comprises two main components: a scatter estimation model and a scatter compensation algorithm (see Sec. III). Typically, scatter estimation and scatter compensation are intertwined in iterative algorithms. The aim of the current paper is to present a systematic framework of scatter compensation approaches, whereas the multitude of scatter estimation approaches will be postponed to a separate paper. Forthesake of completeness, we add a short overview of hardware techniques for scatter suppression in Sec. II.

首先，我们会讨论并提出本文中使用的术语。文献中使用了不同的术语，来表示降低散射及其在图像质量中的影响：散射补偿，控制，修正，管理，弥补，降低，拒绝，移除，减去，或抑制。我们倾向于使用散射抑制，包括硬件的散射rejection方法和软件的散射修正方法。一个关键的观察是，散射修正算法的结构包括两个主要部分：一个散射估计模型，和一个散射补偿算法（见第3部分）。典型的是，散射估计和散射补偿在迭代算法中是紧密联系的。本文的目的是，给出散射补偿方法的系统性框架，而散射估计方法会单独占用一篇文章。对完整的研究，我们在第2部分加入了硬件散射抑制技术的简短回顾。

In Fig. 1, we present a structogram of scatter suppression methods and components, together with indicators of the corresponding sections in the current paper. The further ramifications of the scatter estimation branch is omitted, since the details will be presented later. Also scatter compensation approaches from other fields, which have not yet been sufficiently appreciated in CT, are included.

在图1中，我们给出了散射抑制方法和组成部分的结构图，以及在本文中的对应小节。散射估计在后续文章中会给出。其他领域中的散射补偿方法，在CT中并没有充分利用的，可包括在本文中。

In Sec. III, different types of structure of scatter correction procedures are distinguished, based on projection data or on reconstructed images. For reconstruction-based scatter compensation (Sec. IV), a self-consistency principle is pointed out which expresses the aim that the mathematical–physical forward projection (as a virtual model of the physical data acquisition including scatter) of the corrected images should be close to the real measured data. A weaker consistency condition is given for projection-based scatter compensation approaches (Sec. V). The consistency conditions give rise to several iterative algorithms.

在第3部分中，我们区分了不同类型的散射修正过程，基于投影数据的，或基于重建图像的。对于基于重建图像的散射补偿（第4部分），我们指出了一个自一致的原则，表达了修正图像的数学-物理前向投影的目标（作为获取物理数据，包括散射的虚拟模型），应当与真实测量到的数据相接近。对于基于投影的散射补偿算法，给出了一个更弱的一致性条件（第5部分）。一致性条件带来了几种迭代算法。

Another classification is the distinction of deterministic and statistical approaches. Deterministic approaches are treated in Sec. V. Statistical approaches aiming at control of scatter-induced noise are treated in Sec. VI. Signal and noise propagation is analyzed in Sec. VII, where the crucial concept of differential-signal-to-noise-ratio (dSNR) is introduced. The main results are compiled in Table I. It might be useful for readers not familiar with to anticipate Sec. VII before beginning with Sec. IV. In Sec. VII B 4, we mention some novel approaches in the literature which include scatter into a generalized concept of detective quantum efficiency (DQE).

另一种分类是，确定性方法和统计性方法。确定性的方法在第5部分。统计性的方法目标是，控制散射引入的噪声，在第6部分讨论。信号和噪声传播在第7部分分析，其中提出了dSNR的关键概念。主要结果汇编在表1中。在第7节B4中，我们提到了文献中的一些新方法，将散射纳入到了一个更加一般性的概念中，即检测量子效率(DQE)。

In Sec. VIII, the different deterministic and statistical methods are discussed with respect to their properties and applications.

在第8部分中，讨论了不同的确定性和统计性方法的性质和应用。

Particular mathematical derivations are shifted to the appendix. 附录中包含了特定的数学推导。

### 1.3 Mathematical notations

To avoid complicated formula, we prefer compact notations using operator symbols when ever possible. We frequently omit space variables, where functions in object space are defined in a domain ⊂ R^3 and functions in projection space defined in a domain of R^2 × [0, 2π) where the detector coordinates are u ∈ R^2 and the projection view is denoted by an angular variable α. For projection data, we will use normalized intensities Φ, Φ_T (or their negative logarithms φ, φ_T, respectively), i.e., measured or virtual fluence data I_T, I_P are divided by the calibration data I_0 corresponding to measurements without attenuating object and without scattered radiation. We assume that all usual calibration and precorrection procedures to compensate for detector related errors have been carried out.

## 2. Hardware techniques of scatter suppression

Two classes of preprocessing techniques of mechanical or geometrical scatter rejection can be distinguished (Fig. 1): prepatient source-side collimators and beam-shapers, and postpatient detector-side techniques as air-gap, antiscatter grid, or interpixel septal collimation at the detector. An essential advantage of prepatient scatter rejection is reduction of patient dose. We cite some historical references which we owe to Kalende and Barnes.

有两类机械的或几何的散射抑制预处理技术（图1）：在患者之前的源方的准直器和射束形成器，患者后方的探测器一侧的技术，如air-gap，反散射网格，或探测器上的像素间准直。患者前的散射抑制的关键好处是，降低患者的剂量。我们引用了一些历史文献。

### 2.1 Collimation

Restriction of field-size to the volume-of-interest (VOI) by source-side collimation of beam is imperative for dose minimization. However, if field-size is chosen too small in CT imaging, the trade-off between truncation and scatter artifacts has to be considered.

通过射线源这一边的射束准直，将射野的大小限制到感兴趣体中，对于剂量最小化是非常重要的。但是，如果射野大小在CT成像中太小，则必须要考虑截断和散射的伪影之间的折中。

VOI-CBCT has been investigated by Chen et al. and Lai et al. who propose to combine a normal or high exposure acquisition using a VOI collimator and low exposure full-field acquisition to extrapolate the truncated VOI projection data. Inside the VOI, scatter-to-primary ratios were reduced by an order of magnitude due to the smaller irradiated volume restricted by the collimator. An alternative was proposed by Chityala et al. to use a VOI filter which is a type of beam shaper (see Sec. II B), which attenuates x-rays outside the VOI. The VOI filter method needs only one acquisition.

Chen等和Lai等对VOI-CBCT进行了研究，他们提出在使用VOI准直器时用正常的或高剂量辐射获取图像，在低剂量时对全野获取图像，并将截断的VOI投影数据进行外插。在VOI内部，散射对初级射线的比，由于准直器约束了放射照射的体，下降了一个量级。一种替代方法是，使用一个VOI滤波器，这是一种射束形成器（见第2部分B），在VOI外对X射线进行衰减。VOI滤波器方法只需要一次图像获取。

The slit technique to reduce scatter in x-ray radiography goes back to an idea as early as 1903: the object is scanned by a moving slit which collimates the beam such that only a small amount of scatter is generated by the object. Standard or helical CT scanners with a single or few detector rows are examples of moving slit realizations. Multiple slit scanning assemblies in radiography were proposed by Barnes et al. Recently, Shikhaliev published a feasibility study for a multiple slit scanning CT system with energy resolving counting detectors. Scanning beam digital x-ray systems (SBDX) consist of a large-area scanned x-ray source, opposite a small detector, and a multihole collimator. The x-ray source generates a series of narrow beams using an electronically deflected focal spot. The beams are sequentially captured by a small stationary detector. Due to the narrow x-ray beams, scatter is significantly reduced. Simulations showed reduction of scatter-to-primary-ratio (SPR) by an order of magnitude. However, despite a feasibility study by Schmidt et al., the loss of x-rays by the strong collimation would induce severe heat loading problems and scanning speed is strongly limited by the small detector.

降低X射线放射成像中的散射的Slit技术，可以追溯到1903年的思想：目标被一个移动的slit扫描，使射束得到准直，这样目标只能产生一小部分散射。只有一个或几行检测器的标准CT或螺旋CT，是移动slit的实现的例子。多个slit扫描，由Barnes等提出。最近，Shikhaliev发表了一种可行性研究，是多slit扫描CT系统，带有能量分辨计数探测器。SBDX由一个大面积X射线源，小的探测器，和一个多孔准直器构成。X射线源产生一系列很窄的射束，使用的是电子偏移的焦点。这些射束由一个小型静止检测器按顺序捕获到。由于X射束很窄，散射得到了显著降低。仿真表明，散射与初级射束比(SPR)降低了一个数量级。但是，尽管有可行性研究，很强的准直带来的X射线的损失，带来了严重的散热问题，扫描速度也被小型探测器限制。

### 2.2 Beam shaper(Bow-tie filter)

A bow-tie filter in front of x-ray source can be used to homogenize the attenuation profile: it compensates the shorter x-ray paths within the object section near the border with the consequence of dose reduction, efficient use of dynamic range, and reduction of beam-hardening artifacts. The effect of scatter reduction has been studied by Glover, Tkaczyk et al., Kwan et al., Graham et al., Mail et al., Menser et al., and Bootsma et al. A method for evaluation of bow-tie filter performance is given by Boone.

在X射线源前的Bow-tie滤波器，可以用于衰减的均匀化：它对目标中接近边缘的较短的x射线路径进行了补偿，其结果是剂量降低，可以高效的使用动态范围，降低射束硬化的伪影。很多人研究了其散射降低的效果。评估Bow-tie滤波器性能的方法由Boone给出。

### 2.3 Air-Gap

Air-gap is defined as the distance between scattering object and detector. The observation that scatter fluence at the detector decreases with increasing air-gap was first stated in 1926. Main theoretical and experimental studies on image quality in diagnostic imaging related to air-gaps have been reported by Sorenson and Floch, Neitzel, and Persliden and Carlsson. In CT systems, typically the air-gap ranges between about 20 and 50 cm, depending on geometry and patient size. But space limitations restrict increase of air-gap with standard CBCT systems.

Air-gap定义为，散射目标和探测器之间的距离。在探测器上的散射通量，会随着air-gap的增加而降低，这在1926年首次发现。在诊断成像的图像质量中，与air-gap相关主要的理论和试验研究，有文章给出。在CT系统中，一般air-gap范围在20-50cm，与几何关系和患者大小相关。但在标准CBCT系统中，空间的限制限制了air-gap的增大。

### 2.4 Antiscatter grid

The first focused antiscatter grid (ASG) goes back to Bucky in 1913. Main experimental and Monte Carlo studies on image quality in diagnostic imaging related to ASGs are published by Kalender, Chan and Doi, Chang et al., and Neitzel. Today, high-precision focused two-dimensional ASGs are topics of research. Image quality effects of conventional one-dimensional ASGs in CBCT have been investigated by Siewerdsen et al., Wiegert et al., Kyriakou and Kalender, and Rinkel et al. ASGs are effective devices to reduce scatter fluence at the detector by up to an order of magnitude, however, at the cost of decrease of transmittance of primary radiation by typically about 30%. Efficiency of ASGs decreases with increasing air-gap and photon energy; without ASG, there is only a weak dependency on x-ray tube voltage.

第一个聚焦反散射网格(ASG)可以追溯到1913年。文章包含了与ASGs相关的诊断图像质量上的主要试验和蒙卡研究。今天，高精度聚焦二维ASGs是一个研究主题。传统的一维ASGs在CBCT中的图像质量效果由文章进行了研究。ASGs是在探测器端降低散射通量的有效设备，可以达到一个量级的减少效果，但是，其代价是初级辐射的透射率会降低大约30%。ASGs的效果会随着增加air-gap和光子能量而降低；在没有ASG时，在x射线管电压上的依赖性很低。

The benefit of ASGs in CBCT is discussed controversially. Most researchers agree that ASGs are favorable concerning contrast-to-noise-ratio (CNR) with thick objects. However, Rinkel et al. have studied the dependence of the SNR-improvement factor (SIF) on SPR and object diameter and have given examples where SIF, depending on the additional electronic noise level, scarcely exceeded a value of 1 even with high SPR. Nevertheless even with efficient ASGs cupping and bar artifacts are not sufficiently reduced. There is common agreement that additional software scatter correction is recommended at any case.

ASGs在CBCT中的效果是有争议的。多数学者同意，在考虑厚目标的CNR时，ASGs的效果是比较好的。但是，Rinkel等研究了SNR的改进系数对SPR和目标半径的依赖性，并给出了例子，即使在SPR很高的情况下，SIF也很少高于1。尽管有ASGs，cupping和bar伪影也没有得到很好的降低。大家一致同意，在任何情况下，都需要进行额外的软件散射修正。

## 3. Structure of Scatter Correction Methods

The structure of scatter correction schemes comprises two essential components: a scatter estimation model and a scatter compensation procedure. In addition scatter correction schemes differ in the type of data from which the scatter estimation is obtained and at what stage in the CT data processing chain the scatter compensation operates. Mixed strategies are possible.

散射修正方案的结构包含两个主要部分：散射估计模型，和散射补偿过程。除此以外，如果得到了不同的散射估计，或是在CT数据处理链上散射补偿在哪个阶段进行，散射修正方案在数据类型上会不一样。可能是混合策略。

### 3.1 Image-based scatter correction

With image-based approaches, scatter is estimated using analytical or Monte Carlo methods from a previously reconstructed approximate image and fed back for scatter compensation of the projection data (see the simplified scheme in Fig. 2). The first reconstruction may be without scatter correction (initialization of scatter image 0), but it is recommended to use some simple projection-based scatter estimation model for an initial guess to improve the first reconstruction. Generally, image-based scatter estimation methods are superior to projection-based scatter models, however, at the cost of bigger computational expense.

在基于图像的方法中，散射是用解析方法或蒙卡方法，从一个之前重建的近似图像中估计的，并送回进行投影数据的散射补偿（图2）。第一次重建可能是没有散射修正的（散射图像初始化为0），但推荐使用一些简单的基于投影的散射估计模型作为初始估计，来改进第一次重建。一般来说，基于图像的散射估计方法比基于投影的散射模型要好，但是，其代价是更大的计算代价。

### 3.2 Projection-based scatter correction

The structure of projection-based scatter correction in Fig. 3 is similar with measured scatter data or with virtual scatter data estimated by a mathematical model. Scatter images are obtained from each projection image separately. They are much faster than image-based techniques. However, the underlying mathematical models are less accurate due to simplifications of the x-ray scatter propagation process.

基于投影的散射修正如图3所示，与测量的散射数据，或与数学模型估计的虚拟散射数据相似。散射图像是从每个投影图像单独获得的。他们比基于图像的技术更快。但是，潜在的数学模型不是那么精确，因为x射线散射传播过程的简化。

### 3.3 Mixed strategies

In a simple kind of mixed strategy updated, object-dependent parameters are obtained from the reconstructed image in order to improve the next projection-based scatter estimation cycle, for example, parameters for thickness-dependent scatter convolution kernels as used, e.g., in Meyer et al. The corresponding modification in the projection-based scatter correction scheme (Fig. 3) would be a dotted arrow from the “reconstructed image” box back to the “scatter estimation model” box.

在一个简单的混合策略中，依赖于目标的参数是从重建图像获得的，以改进下一个基于投影的散射估计循环，比如，使用了依赖于厚度的散射的卷积核参数，如，在Meyer等[59]。在基于投影的散射修正方案中（图3），对应的修改会是从重建图像的盒子回到散射估计模型的盒子的虚箭头。

An other mixed strategy is the iterative improvement of a projection-based scatter correction with an image-based one.

另一种混合的策略是，一种基于投影的散射修正和一种基于图像的迭代改进。

### 3.4 Downsizing

Since scatter distributions are spatially smooth in the statistical mean, it is recommended to reduce the computational expense by downsampling the projection data and image data. After scatter estimation done at the coarser resolution scale, the estimated scatter distribution is upsampled to the initial resolution scale. The scatter compensation is finally carried through the original sampling. The corresponding modification of the block diagrams (Figs. 2 and 3) including down- and upsampling is straightforward.

由于散射分布在统计上是空间平滑的，通过对投影数据和图像数据进行降采样，可以降低计算代价。在更粗糙的分辨率尺度上估计了散射之后，估计的散射分布可以上采样到初始分辨率尺度。散射补偿在原始采样下进行。对模块图（图2和图3）的对应修改，包括上采样和下采样，是很直接的。

## 4. Image Reconstruction Based Scatter Compensation Approaches

### 4.1 Iterative Improvement Reconstruction

In the following, we suppose that an appropriate scatter estimation procedure is already available. Scatter correction is then started from a reconstructed image in object space, although obtained from projection data which are still corrupted by scatter. The general iterative image-based correction approach to solve the underlying nonlinear integral equation [Eq. (2)] has been motivated by modification of ideas in Censor et al. Instead of iterative refinement, we prefer to use the notion iterative improvement. The workflow of the algorithm is presented in Fig. 4. We emphasize that this is a very general approach which is neither restricted to CT nor to scatter correction only. For CT application, we directly use CT projection data φ, φ_T which are usually obtained from normalized intensity data Φ, Φ_T by

下面，我们假设有一个合适的散射估计过程是已经可用的。散射修正是从目标空间的一个重建图像开始，虽然重建图像是从投影数据得到的，而投影数据已经是受到散射污染了。通用的基于图像的迭代修正方法，来求解非线性积分方程（式2）是Censor的思想的改进。我们没有采用迭代精炼，我们只是喜欢使用迭代精炼的概念。算法的流程如图4所示。我们强调，这是一个非常通用的方法，并不局限于CT，也不局限用于散射修正。对于CT应用，我们直接使用CT的投影数据φ, φ_T，这通常是由归一化的亮度数据Φ, Φ_T由下式得到的

$$φ = -lnΦ, φ_T = -lnΦ_T$$(1)

**Notations and definitions**:

$\tilde A_S$ mathematical–physical aquisition operator to generate normalized CT projection data including scatter from a 3D image of an object; the usual negative logarithm Equation (1) is incorporated;

$\tilde A_S$表示数学-物理获取算子，来生成归一化的CT投影数据，包含了目标的3D图像的散射；将式1的通常的负的log也考虑了进去；

B approximate linear CT reconstruction algorithm; B近似了线性CT重建算法；

$\tilde R$ regularization operator in image space. $\tilde R$表示图像空间的正则化算子。

The nonlinear aquisition operator $\tilde A_S$ is explained in more detail in Appendix A. For examples of B, see review article by Defrise and Gullberg; we particularly mention the approximate CBCT reconstruction algorithm of Feldkamp et al., the generalization for partial and irregular near-circular orbits by Wiesent et al., and exact algorithms as derived by Katsevich. Note that B is in any case not only an approximation of the inverse of $\tilde A_S$ due to nonlinear effects such as scatter (and beamhardening).

**Self-Consistency**: The triple ($φ_T, \tilde A_S, f$) is said to be self-consistent, or the image f is consistent with the measured CT data $φ_T = -lnΦ_T$ with respect to the acquisition operator $\tilde A_S$, if

$$\tilde A_S f = φ_T$$(2)

The relation should be interpreted in an approximate sense to be valid within a tolerance band about the data vector φ_T, a band corresponding to the specific noise level, limited accuracy of the acquisition model, etc.

这个关系应当以一种近似的意义来解释，在数据向量φ_T的一定的误差容忍度内就认为是有效的，对应具体的噪声层次，获取模型的有限准确率等的误差范围。

Attention: It must be emphasized that the acquisition model $\tilde A_S$ should be as close to reality as possible. Any errors in the model will be transferred as artifacts into the reconstructed image.

注意：必须强调的是，获取模型$\tilde A_S$应当尽可能与真实接近。模型中的任何误差都会传导为重建图像中的伪影。

#### 4.1.1.1 Iterative improvement in data space

For ease, an intermediate vector in projection data space $ψ = ψ(u,α)$ is introduced. Optionally, a more or less simple (e.g., Grimmer and Kachelriess) projection-based scatter correction C may be combined with the CT reconstruction algorithm B. For this combination, we use the abbreviation

为计算简便，引入了投影数据空间$ψ = ψ(u,α)$的中间向量。一个基于投影的散射修正C与CT重建算法B相结合到一起。对于这种结合，我们使用下面的缩写：

$$B_c = BC$$(3)

If $B_c$ is nontrivial, then $B_c$ is no longer linear. If C is omitted then $B_c = B$. Alternatively, C may be used only at iteration 0 to improve the initial guess and switched off for later iterations.

如果$B_c$是有意义的，那么$B_c$就不是线性的。如果忽略了C，那么$B_c = B$。另外，C可能只在第0次迭代中使用来改进初始的猜测，在后续的迭代中就关掉不用了。

Iteration start: 迭代开始

$$ψ^{(0)} = φ_T$$(4)

Iteration update: 迭代更新

$$f^{(n)} = B_c ψ^{(n)}, n ≥ 0$$(5)
$$ψ^{(n+1)} = ψ^{(n)} + λ^{(n)}(φ_T - \tilde A_S f^{(n)})$$(6)

Iteration stop criteria (with appropriate error bounds ε1, ε2):

$$||φ_T - \tilde A_S f^{(n)}||≤ε1, ||f^{(n+1)} - f^{(n)}||≤ε2$$(7)

A proof of convergence is sketched in Appendix B. 附录B中简要证明了收敛性。

Discussion: (1) The adjustable $λ^{(n)}$ denotes a sequence of relaxation scalars which may be combined with a smoothing operator in projection image space to control convergence of
the iteration cycles.

讨论：(1) 可调整的$λ^{(n)}$表示松弛标量序列，可以在投影图像空间与平滑算子结合到一起，以控制迭代循环中的收敛情况。

(2) Update expression [Eq. (6)] is a general formulation where the operator $\tilde A_S$ is not restricted to scatter but may also include other physical phenomena, as spectral hardening, etc.

(2) 式6的更新表达式是一个通用表达式，其中算子$\tilde A_S$并不限于散射，也可能包括了其他物理现象，如谱硬化等。

(3) The stop criteria are understood as combined by an “OR” condition. The distance measures in Eq. (7) might be, e.g., appropriately weighted maximum or root-mean-squares norms dampening effects of object borders. Reasonable error bounds have to account for data noise level and limited accuracy of the acquisition model.

(3)停止准则可理解为用或运算结合到一起。式7中的距离度量可能是，比如，近似加权最大值，或均方根范数。合理的误差限必须考虑到数据噪声水平，和图像获取模型的有限的准确率。

Note: An iteration scheme very similar to Eq. (6) is given by Mainegra-Hing and Kawrakow with a Monte Carlo based acquisition model.

注意：有文章给出了基于MC的图像获取模型的一个迭代方案。

#### 4.1.1.1 Iterative improvement in object image space

We present an alternative formulation in object space, where for simplicity, we leave away the precorrection C, let alone for initialization, and replace B_c with a standard CT reconstruction operator B, which is essentially linear.

我们提出了在目标空间的另一种表述，简化起见，我们丢弃掉预修正C，也丢弃掉初始化，将B_c替换成标准的CT重建算子B，其实质上是线性的。

Iteration start: 迭代开始

$$f^{(0)} = BCφ_T$$(8)

Iteration update: 迭代更新

$$f^{(n+1)} = f^{(n)} + B λ^{(n)} (φ_T - \tilde A_S f^{(n)}), n ≥ 0$$(9)

The iteration stop criteria are the same as in Eq. (7). 迭代停止条件与式(7)中一样。

A regularization approach as outlined in Sec. 4.2 is recommended.

### 4.2 Regularization

To control noise and to stabilize the update images in the course of iterations, an additional term to Eq. (9) may be introduced, which imposes adaptive smoothness and edge-preserving constraints to neighborhoods of voxels, preventing adjacent voxels to deviate too much from each other. A regularized generalization of Eq. (9) is

为控制噪声，在迭代的过程中稳定更新的图像，要给式9加入一个额外的项，对相邻的体素施加了自适应的平滑和保护边缘的限制，防止相邻的体素灰度差距过大。式9的一种正则化形式为

$$f^{(n+1)} = f^{(n)} + B λ^{(n)} (φ_T - \tilde A_S f^{(n)}) - ρ^{(n)} \tilde R f^{(n)}$$(10)

$ρ^{(n)}$ controls the strength given to the regularizing operation $\tilde R$. Sophisticated approaches are presented in Bruder et al. Regularization in a broader context is discussed by Puetter et al.

$ρ^{(n)}$控制的是正则化操作$\tilde R$的强度。其他文献给出了更多正则化方法。

### 4.3 Postprocessing image corrections

Heuristic postprocessing image corrections have been proposed by Altunbas et al., Marchant et al., and Kyriakou et al. The methods are aimed at reduction of cupping effects and other artifacts in reconstructed images phenomenologically, however, without using an explicit physical scatter estimation model, and seem to be useful to mitigate residual artifacts which remain after the application of scatter correction algorithms based on simplified physical models. This reasoning was leading Wiegert et al. to combine in an iteration loop projection-based scatter correction with image postprocessing to eliminate cupping artifacts.

启发式的后处理图像修正有几篇文献进行了处理。这些方法的目的是，消除重建图像中的cupping效果和其他伪影，但是，不使用一种显式的物理散射估计模型，这在使用了基于简化的物理模型的散射修正算法后，消除剩余的伪影上，很有作用。这种推理使Wiegert等将迭代的基于投影的散射修正与图像后处理结合到一起，以消除cupping效果。

## 5. Projection-based Deterministic Scatter Compensation Approaches

Since image-based scatter estimation is much expensive, projection-based approaches are more frequently preferred. Such scatter estimation models can in most cases be formalized as a transformation S(Φ) of the normalized primary intensity distribution at the detector Φ. Well-known methods are, e.g., convolution approaches or the more general spatially variant scatter-kernel superposition method; we give a short outline in Appendix C. In the following, we will use normalized intensity data Φ, Φ_T instead of logarithmic CT data as in Sec. 4. Logarithmic CT projection data can be regained by the relations in Eq. (1). We will write corresponding operators on projection space functions without the tilde sign. Keep in mind that the operators depend on the projection view α, which we omit for the sake of notational simplicity. The consistency equation in projection space for every projection view reads

由于基于图像的散射修正非常昂贵，基于投影的方法使用的更加频繁。这种散射估计模型在多数情况下，可以表述为探测器Φ处的归一化初级灰度分布的一个变换S(Φ)。有名的方法如，卷积方法，或更加通用的空间变化散射核叠加方法；我们在附录C中给出了一个简短的概述。后面，我们会使用归一化亮度数据Φ, Φ_T，替代第4部分中的CT数据的对数。通过式1，可以重新得到CT投影数据的对数值。我们会在投影空间函数上写出对应的算子，没有波浪符号。记住，算子依赖于投影的视角α，简化起见我们忽略之。在投影空间中，对每个投影视角的一致性方程写作

$$A_s(Φ) = Φ + S(Φ) = Φ_T$$(11)

which is a weaker condition than the self-consistency condition (2). 这是自一致条件(2)的一个弱化版本。

### 5.1 Matrix inversion and deconvolution approaches

In general, S is a nonlinear and nonstationary (i.e., spatially variant) operator, due to multiorder scatter and specific object dependency, e.g., with the scatter-kernel method. Even if a linear model in Eq. (11) is adopted and a scatter propagation matrix can be built, the high dimension matrix inversion would be numerically difficult including noise. In a stationary linear model, Eq. (11) could be solved by deconvolution. We refer to the literature in diagnostic x-ray radiology, e.g., Barrett and Swindell, Seibert and Boone, Floyd et al., Maher and Malone, and regularization methods were reported by Close et al. and Abbott et al.

一般来说，S是非线性，非平稳的（即，在空间中变化的）算子，由于多阶散射和具体的目标依赖性，如，具体的散射核方法。即使式11采用一个线性模型，建立了一个散射传播矩阵，矩阵的高维求逆在数值上是很难的，还包含了噪声。在一个平稳的线性模型中，式11可以用解卷积求解。我们参考诊断X射线成像中的文献，以及一些正则化方法。

### 5.2 Fixed-point equation and iterative subtractive algorithms

Rearranging Eq. (11) leads to the fixed-point equation 重新整理式11，得到定点方程

$$Φ = Φ_T - S(Φ)$$(12)

A solution Φ=Φ_c of Eq. (12) is called a fixed-point since the right-hand-side (r.h.s.) operator maps Φ_c onto itself. The implicit Eq. (12), generally an integral equation, has to be solved for every intensity projection image Φ = Φ(u; α) for every projection view angle α. The 2D functions Φ and S(Φ) have to be nonnegative, since they represent (normalized) physical radiation intensities or fluences. An iterative scheme analogous to the image-based algorithm, Eq. (6), is the iterative subtractive algorithm with relaxation,

式12的一个解Φ=Φ_c称为定点，因为右手算子将Φ_c映射到其本身。式12的一般形式是一个积分方程，对每一幅亮度投影图像Φ = Φ(u; α)的每个角度α都要进行求解。2D函数Φ和S(Φ)需要是非负的，因为它们表示归一化的物理辐射亮度或通量。与基于图像的式6的算法类似的一个迭代方案，是带有松弛的迭代相减算法

$$Φ^{(n+1)} = Φ^{(n)} + λ^{(n)}(Φ_T - (Φ^{(n)}+S(Φ^{(n)}))), n ≥ 0$$(13)

$λ^{(n)}$ is a relaxation parameter and may be combined with a smoothing operator. In a more general formulation of Eq. (13), the scatter estimation operator may vary with the iteration index as $S^{(n)}$; see Sec. 6.2.

$λ^{(n)}$是一个松弛参数，可以与一个平滑算子相结合。在一个更一般化的式13中，散射估计算子可能随着$S^{(n)}$迭代索引变化，见6.2节。

Discussion: (1) By an appropriate sequence of underrelaxation parameters $λ^{(n)} ≤ 1$ negative intensities and divergence can be avoided.

讨论：(1) 通过一个合适的欠松弛参数$λ^{(n)} ≤ 1$序列，可以避免负亮度和发散的情况。

(2) For $λ^{(n)} ≡ 1$, the update equation would reduce to the standard Banach’s fixed point iteration corresponding to Eq. (12), however, assuring convergence only for SPR < 1.

(2) 对$λ^{(n)} ≡ 1$，升级函数就会蜕化为标准Banach定点迭代，有式12相对应，但是只对于SPR<1确保收敛。

(3) An update similar to Eq. (13) arises in a statistical Gaussian maximum likelihood approach, see Sec. 6.1. 

(3) 与式13类似的更新是以统计高斯最大似然方法呈现的，见6.1节。

Note: An iterative algorithm similar to Eq. (13) is proposed by Yao and Leszczynski. in the context of an analytical first-order scatter model; after two or three iterations, sufficient convergence was reached.

注：有文献提出了与与式13类似的迭代算法。散射模型是一阶解析的上下文；在两到三个迭代之后，就会达到充分的收敛。

### 5.3 Iterative multiplicative algorithm

Some problems that occur with the subtractive algorithm (without underrelaxation) are also avoided by a multiplicative update 相减算法（没有欠松弛）带来的一些问题，可以由相乘更新来避免

$$Φ^{(n+1)} = Φ^{(n)} \frac {Φ_T} {Φ^{(n)} + S(Φ^{(n)})}, , n ≥ 0$$(14)

Discussion: (1) If $Φ^{(0)}>0$, the iteration never becomes negative.

(2) The iteration even converges for SPR ≫ 1.

(3) Asymptotically for n → ∝, the r.h.s. fraction tends to 1, which corresponds to the consistency condition (11).

(4) The algorithm turns out as an approximation of another multiplicative scheme which occurs with a statistical ML approach based on Poisson statistics, see Sec. 6.2.

Clinical images obtained with a multiplicative algorithm and a generalized convolution model are given in Zellerhoff et al. and and Reiser et al.

## 6. Projection-based statistical scatter compensation approaches

In a statistical framework, the total, primary, and scatter intensities are considered as random functions. The task in Secs. 6.1 - 6.3 is to derive algorithms for statistical estimators of the expectations of the primary and scatter intensity distributions, given the measured total intensity data (primary + scatter) and given a mathematical–physical scatter model S. In Sec. 6.4, we alternatively deal with an image-processing restoration approach in projection data space to control noise amplification of the scatter compensation algorithm. An approach based on a heuristic filtering method is sketched in Sec. 6.5.

在一个统计的框架下，总计亮度，初级亮度，和散射亮度都被认为是随机函数。6.1-6.3节的任务是对初级和散射亮度分布的期望的统计估计器推导出算法，给定的值是测量出的总计亮度数据（初级+散射），并给定一个数学物理散射模型S。在6.4节，我们在投影数据空间解决图像处理恢复问题，以控制散射补偿算法的噪声放大效应。6.5节概述了一种启发式的滤波方法。

### 6.1 Gaussian maximum likelihood

In Xia et al., the statistics of the total, primary, and scatter intensities are assumed to be Gaussian. Interestingly, maximization of the logarithmic likelihood functional leads to an iterative update equation which is equivalent to the subtractive algorithm with relaxation equation (13), where the relaxation parameter is given by the ratio of the variance of the primary fluence and the sum of variances of primary and scatter fluence.

在Xia等[83]中，总计亮度，初级亮度和散射亮度假设是高斯分布的。有趣的是，似然函数对数的最大化，带来的是一个迭代更新方程，与带有松弛的相减算法式13是等价的，其中松弛参数由初级通量的方差比初级和散射通量之和的比值。

### 6.2 Poisson maximum likelihood approach

#### 6.2.1 Poisson MLEM in emission tomography

There are some similarities between the problem of scatter correction and the reconstruction problem in single photon emission tomography. We refer to the elegant presentation by Natterer and Wubbeling [pp. 45 and 119]. In emission computed tomography (ECT), the number of radioactive decay events r_k in voxel k of a volume of interest has to be reconstructed from the number of quanta q_l detected in detector pixel l, q = (q_1, ..., q_m)'; prime ()' indicates vector transpose. The measurement acquisition model is described by a transition matrix A where the elements a_kl are defined as the probability of quanta emitted in voxel k to be detected in detector pixel l. Within the statistical framework, the r_k are thought of as the expectations of Poisson random variables. r = (r_1, ..., r_n)' is determined by the maximum likelihood method. Shepp and Vardi derived the well-known maximum likelihood expectation maximization (MLEM) algorithm, the iteration update equation of which writes in compact form

散射修正和单光子发射断层成像的问题有一些相似度。我们参考了文献中的表示。在ECT中，在感兴趣体中的体素k上的放射衰减事件r_k数量，要从探测器像素l检测到的光子数量q_l中重建出来，q = (q_1, ..., q_m)'；()'表示向量转置。测量获取模型由一个迁移矩阵A描述，其元素a_kl定义为从体素k中发射的光子被探测器像素l探测到的概率。在统计框架中，r_k被认为是Possion随机变量的期望。r = (r_1, ..., r_n)'由最大似然方法确定。Shepp等推导了著名的最大似然期望最大化(MLEM)算法，其迭代更新方程写成紧凑形式

$$r^{(n+1)} = r^{(n)} \overline A' (\frac {q} {Ar^{(n)}})$$(15)

The operation within the brackets is a pointwise division. The prime indicates matrix transpose. The matrix $\overline A$ is formed by normalizing A such that each column sum is equal to 1

括号中的运算是一个点除。撇号代表矩阵转置。矩阵$\overline A$是将A的每一列进行归一化形成的

$$\overline A' = 1$$(16)

The symbol 1 signifies a vector with all components equal to 1. The operation A can be interpreted as a forward-projection and $\overline A'$ as a backprojection.

符号1表示一个向量，所有元素都等于1。算子A可以解释为一个前向投影，$\overline A'$可以理解为反向投影。

#### 6.2.2 Poisson MLEM for Scatter Correction

Poisson MLEM for scatter correction in radiography was introduced by Floyd et al., Baydush and Floyd, and Ogden et al. The ECT model outlined above is transferred by replacing r by the unknown primary intensity Φ and q with the measured total intensity distribution at the detector Φ_T, respectively. The transition matrix A is replaced with the acquisition operator

在放射成像的散射修正的Possion MELM方法由Floyd等提出。上面列出的ECT模型是通过将r替换为未知的初级亮度Φ，q替换为测量到的探测器Φ_T上的总计亮度分布，而迁移到放射成像中。迁移矩阵A替换为获取算子

$$A = A_s = I + S$$(17)

where I is the identity operator and the operator S describes the transformation of a primary intensity distribution at the detector Φ to a scattered intensity distribution at the detector S(Φ). In this interpretation, Φ represents the expectation values of Poisson random variables. The MLEM algorithm then reads

其中I是恒等算子，算子S描述了将探测器Φ上的初级亮度分布，转化成探测器S(Φ)上的散射亮度分布的变换。在这个解释中，Φ表示Possion随机变量的期望值。MLEM算法如下

$$Φ^{(n+1)} = Φ^{(n)} \overline A' (\frac {Φ_T} {AΦ^{(n)}})$$(18)

Explicitly, the iteration update for every detector pixel k corresponding to Eq. (18) is given by

显式的，对每个探测器像素k对应式18的迭代更新如下式

$$Φ^{(n+1)}_k = Φ^{(n)}_k \sum_j ((\frac {δ_{jk}+s_{jk}} {1+\sum_i s_{ik}}) \frac {(Φ_T)_j} {Φ_j^{(n)} + \sum_l s_{jl} Φ_l^{(n)}})$$(19)

$δ_{jk}$ is the Kronecker symbol defined by $δ_{jj}=1$, $δ_{jk}=0, (j \neq k)$，and $s_{jl}$ describes the contribution of a unit primary fluence at detector pixel l to the scattered fluence at detector pixel j. A generalization consists in replacing the fixed scatter generation operator S in Eq. (17) with an adaptive operator $S^{(n)}$ which may change from iteration to iteration,

$δ_{jk}$是Kronecker符号，$s_{jl}$描述的是在探测器像素l上的单位初级通量对探测器像素上的散射通量j的贡献。一种推广包括将固定的散射生成算子S，替换为自适应的算子$S^{(n)}$，每次迭代都可能不一样

$$A^{(n)} = I + S^{(n)}$$(20)

As a nontrivial example, we give a sketch of the well-known scatter kernel approach in Appendix C. 作为一个一般例子，我们在附录C中给出了一个著名的散射核方法。

Discussion: (1) The quotient expression in Eq. (18) tends to the constant vector 1, making the operator $\overline A'$ superfluous for n → ∞.

(2) This shows the similarity of Eq. (18) to the multiplicative algorithm Eq. (14), where the operator $\overline A'$ is omitted. Baydush et al. have preferred to use the multiplicative update Eq. (14) as an efficient approximation of the MELM update Eq. (18).

(3) Acceleration techniques using overrelaxation parameters have been discussed by Metz and Chen.

(4) The iteration equation (18) coincides with the Richardson–Lucy restoration algorithm, which also plays a role in efficient smoothing of Monte Carlo generated scatter distributions.

Note: An application to CBCT scatter correction using experimentally obtained spatially variant scatter kernels was presented by Li et al.

### 6.3 Bayesian maximum a posteriori probability or penalized likelihood method

The MLEM algorithm is known to increase noise in the course of iterations. To overcome this, several advices are given (Ref. 84, p.123): (1) stop the sequence of iterations early before a given tolerated noise level is surpassed; (2) smoothing after each iteration; or (3) subtract a penalty term from the logarithmic likelihood function to be maximized, i.e., too much roughness of the image is prevented by penalization during the iteration process. This penalization method is equivalent to the generalization of the ML approach in a Bayesian maximum a posteriori probability (MAP) framework, where the penalty arises from prior statistical information about the class of images. Various priors are described in Green. The philosophical controversy among classisist and Bayesian statistians is discussed by Frieden. The duality between Bayesian methods and regularization is pointed out by Puetter et al. [p. 171].

Baydush et al. and Floyd et al. have extended the Poisson MLEM approach and have derived their algorithm in a Bayesian framework. The MELM iteration update [Eq. (18) replaced by Eq. (14)] is completed by a simple second step based on an approach by Hebert and Leahy. The evaluations show that the new technique can reduce scatter even to levels far below those provided by an antiscatter grid and can increase CNR without loss of resolution in chest radiography and mammography.

Those findings seem to contradict the dSNR preservation rule for scatter correction (Sec. VII B 2). Apparently the incorporation of prior information into the Bayesian approach allows to subdue the limitation of CNR improvement.

### 6.4 Penalized weighted least squares method

From the point-of-view of scatter compensation, there is no essential need to distinguish between measured scatter data and those estimated by a theoretical model. A penalized weighted least squares approach is aiming at suppression of image noise during the process of scatter compensation. Particularly in regions of large local SPR, it is desirable to dampen the noise amplification at the cost of reduced local contrast recovery. To deal with this trade-off, scatter compensation is formulated as a restoration problem based on an appropriate cost functional. The penalized weighted least squares (PWLS) method is a penalized ML or Bayesian MAP approach for independent Gaussian noise. Instead of restoration for normalized primary intensity data, the approach is applied directly to (logarithmic) CT projection image data. This is justified by the observation in an experimental study that the noise in CT projection image data can well be approximated by uncorrelated Gaussian noise. Minimization of the cost functional F results in an optimal restoration of primary (scatter-compensated) CT projection data $\hat φ = -ln (\hat Φ)$

$$F(\hat φ) = Z^{Σ} (φ_c - \hat φ) + γ R(\hat φ)$$(21)

The first r.h.s. part is a $χ^2$ expression, i.e., the sum of squared deviations between previously scatter corrected (by a deterministic method) CT projection data φ_c and the optimal CT projection data $\hat φ$ aimed at, and the deviations are weighted by the statistical variances of the estimated CT data

$$Z^{Σ} (φ_c - \hat φ) + (φ_c - \hat φ)' Σ^{(-1)} (φ_c - \hat φ)$$(22)

Σ is a diagonal matrix where diagonal entry number k is the variance $σ^2_k$ of pixel k, and k indicates pixel coordinates u=(u,v) of φ(u). The variances $σ^2_k$ can be estimated by well-known methods.

In Eq. (21), the second r.h.s. expression R is a penalty term which quantifies the roughness (or smoothness) of the restored primary intensity image by evaluating the variation of adjacent pixels in the vicinity of every pixel (u, v). The factor γ in Eq. (21) controls the trade-off between roughness of the restored image and the discrepancy between the data from the estimation model and the data used for the restored image. In the special case of γ = 0, i.e., without any smoothness constraint, the optimization of Eq. (21) results in the trivial solution $\hat φ = φ_c$. As an adaptive quadratic smoothness constraint functional is proposed

$$R(\hat φ) = \sum_k \sum_{l∈N(k)} w_{kl} (\hat φ_l - \hat φ_k)^2$$(23)

where l∈N(k) represent a set of neighboring pixels about pixel index k and $w_{kl}$ are the weights of pixel l∈N(k). To preserve edges adaptive weights have to be used, such that the weight of neighbor pixel l∈N(k) is nonlinearly damped down according to the difference with respect to the pixel k of concern, e.g.,

$$w_{kl} = exp(-\frac {(\hat φ_l - \hat φ_k)^2} {δ^2})$$(24)

The strength of edge preservation is controlled by an adjustable parameter δ. Minimization of Eq. (21) can be performed by the iteration algorithm, with initialization $\hat φ^{(0)} = φ_c$ and update

$$φ^{(n+1)}_k = \frac {φ^{(0)}_k + γσ_k^2 \sum_{l∈N(k)} w_{kl} \hat φ_l^{(n)}} {1 + γσ_k^2 \sum_{l∈N(k)} w_{kl}}$$(25)

A slightly faster Gauss–Seidel version is obvious.

CT images of a Catphan and an anthropomorphic phantom show promising noise reduction.

### 6.5 Split and smooth method

The method is based on heuristic reasoning but delivers qualitatively similar results. Since deterministic scatter compensation restores contrasts at the cost of corresponding noise increase (see Sec. VII B 3), the simple idea is to split off the scatter correction term and smooth it appropriately before applying it again for compensation. Replacing $φ_c = -lnΦ$ and $S_c = S(Φ)$ in Eq. (12), the scatter correction term reads

$$δφ_c = -ln(1 - \frac{S_c}{Φ_T})$$(26)

where S_c is the normalized scatter intensity distribution estimated by measurement or by mathematical models, and

$$φ_c = φ_T + δφ_c$$(27)

denotes the scatter-corrected CT projection data. The idea consists in replacing $φ_c$ by

$$φ_c^G = φ_T + G(δφ_c)$$(28)

where G denotes a smoothing operator. Smoothing may be realized by low-pass filtering, but a straightforward generalization is replacement of the low-pass filter by spatially variant adaptive smoothing.

## 7. Error Analysis and Image Quality Issues

Scatter error propagation has been repeatedly studied in various papers. We present here our original derivation.

### 7.1 Deterministic error analysis of scatter propagation in CT

#### 7.1.1 Deterministic CT projection error due to scatter

CT projection data are obtained by the negative logarithm of normalized intensity data. Thus, the measured projection data corrupted by scatter are (the projection view angle a is omitted)

$$φ_T(u) = -ln(Φ(u)+S(u)) = φ(u)-ln(1+s(u))$$(29)

where we have introduced the abbreviation

$$s(u) = \frac {S(u)}{Φ(u)}$$(30)

This expression is the spatially variant SPR. From Eq. (29), the CT projection error due to scatter is

$$δφ_T(u) = -ln(1+s(u))$$(31)

which has strong spatial variations in inhomogenious objects, even if the scatter intensity S(u) in Eq. (30) is constant across the object. Equation (29) plays a fundamental role and shows that due to scatter CT projection values are definitely underestimated.

#### 7.1.2 Cupping and shadow artifacts in CT images

The fundamental relation (29) explains the dark cupping effects and dark streak and bar artifacts which typically occur in reconstructed CT images due to scatter, since the projection profiles decreased by scatter are transferred into the image via backprojection.

A rule of thumb for the cupping effect can be derived based on Eq. (29): Consider a circular cylinder of water with diameter L, suppose a scatter-to-primary-ratio SPR=s. Without scatter, CT projection value for an x-ray passing through the cylinder axis is φ = Lμ_0, where μ_0 is the linear attenuation coefficient of water. According to Eq. (29), φ is reduced due to scatter by δφ_s = -ln(1+s), correspondingly the mean linear attenuation coefficient is reduced by

$$δμ = -\frac{1}{L} ln(1+s)$$(32)

This is true for all projection views and rays passing through the center. The reconstructed linear attenuation coefficient in the center of the cylinder will be reduced approximately by Eq. (32).

#### 7.1.3 "Contrast" degradation due to scatter

Let us imagine a CT projection value φ, which corresponds to the logarithmic attenuation of an x-ray passing through an object, as a “signal” and consider a small signal change ∆φ, e.g., caused by a tiny lesion. Let us further assume a scatter intensity background S. Now we ask, how the signal change ∆φ is transferred to a scatter corrupted signal change ∆φ_T. Applying differential calculus to Eq. (29) by ∆φ_T = (∂φ_T/∂φ)∆φ leads to the relation

$$\frac {∆φ_T}{∆φ} = (1+s)^{-1}$$(33)

Note: The factor (1+s)^{-1} coincides with the scatter degradation factor known from digital radiography (Ref. 9, p. 199).

If s in the center of a rotationally symmetric object is equal for every projection view, the degradation factor will be transferred directly to the CT image in the center. In general, the effective degradation in an arbitrary voxel will be an average over all rays crossing that voxel for all projection views.

### 7.2 Statistical error analysis of scatter in CT

#### 7.2.1 Differential-signal-to-noise-ratio dSNR

We prefer to use the notion differential-signal-to-noise-ratio dSNR (as introduced by Swindell and Evans), which is closely related to signal-difference-to-noise-ratio SDNR. Contrary to the traditional notion of CNR, they have the convenient properties of being dimensionless scalars and independent of adding a constant. The adjective differential is attributed to the use of differential calculus.

dSNR: Let X = X(u) be a noisy signal, e.g., a spatial function of radiation fluence at the detector corrupted by noise with mean value function $\overline X = \overline X(u)$ and σ_X the standard deviation of X(u), and let ∆X=∆X(u) be a small change of $\overline X$, then dSNR is defined by the ratio

$$d_{SNR}(X) = \frac {|∆X|}{σ_X}$$(34)

Note: This is closely related to the concept of detectability index.

dSNR preservation property: For a random variable X with finite variance, the differential-signal-to-noise-ratio $d_{SNR}(X)$ remains unchanged for any differentiable strictly monotonic function transform $g: X→g(X)$

$$d_{SNR}(g(X)) = d_{SNR}(X)$$(35)

Proof: The partial derivatives of the d_SNR expression (34) in the nominator, due to Taylor’s differential rule, and the denominator, due to Gaussian error propagation rule, cancel away.

Note: Special cases are addition of or multiplication with a constant; a nontrivial example is logarithm.

#### 7.2.2 Contrast recovery, noise amplification and dSNR preservation by scatter correction

In Table I, the different scatter related differential signal and noise propagation behavior for intensity data, logarithmic CT-projection data, and “ideal” scatter-corrected data are presented. With respect to the latter, we assume that there exists a scatter correction technique which allows to estimate the (expectation of) scatter intensity S_c(u) ≈ S(u) with sufficient accuracy such that the residual error is negligible, i.e., the correction term

$$δφ_c(u) = ln(1+s(u))$$(36)

is such that

$$φ_c(u) = φ_T(u) + δφ_c(u) = φ(u)$$(37)

We will call this ideal (deterministic) scatter correction.

In Table I, the deterministic entry (b) is given by Eq. (33), (a) and (c) are trivial. The noise propagation for intensity data (d) is derived in Appendix D by Eq. (D3). The last row is obtained as the ratio of the first and the second rows (below the headline) corresponding to definition of dSNR by Eq. (34) and confirms the preservation property by Eq. (35). Comparison of expressions (b) and (c) and expressions (e) and (f), respectively, show that by ideal scatter correction, the loss of differential signal (contrast) is completely restored; however, noise is increased by the same factor (1 + s).

Conclusion: Since ideal scatter correction does not improve dSNR, adaptive noise filtering or statistical compensation techniques are strongly suggested.

#### 7.2.3 dSNR propagation via backprojection

For rotationally symmetric conditions of the object and scatter geometry, the dSNR-preservation property for an ideal scatter correction is still valid. However, this is not maintained in a strong sense for conditions of dSNR varying with projection view angle a. Since contrast is averaged linearly, however, noise in the mean square sense, the resulting dSNR after backprojection in the image, is reduced. This phenomenon once again underscores the importance of adaptive noise suppression methods.

#### 7.2.4 Note on DQE analysis

Siewerdsen and Jaffray have analyzed the impact of object scatter on the DQE. For input quantum limitation of the imager, the DQE is reduced in proportion to (1-s)^-1, where s = SPR. The analytic form of DQE derived shows that x-ray scatter can be treated like an additive quantum noise source that appears in the denominator of the DQE in a manner analogous to additive electronic noise (Ref. 121, p. 1913). However, the degradation in DQE cannot be restored by increased exposure, since SPR is independent of exposure. Similarly, in more recent papers, the notion of DQE was generalized by including object generated scattered radiation, where Kyprianou et al. introduced the notion GDQE, see also Jain et al., and correspondingly Ranger et al. used the extended notion eDQE.

## 8. Synopsis

### 8.1 Synopsis of scatter compensation approaches

The final quality of a scatter correction method is the result of both the specific scatter estimation method and the compensation algorithm. However, we have to leave out consideration of the influence of the scatter estimation methods which will be the topic of a separate paper. Nevertheless, we can present in Table II a synopsis of the different scatter compensation algorithms we had described. We could not avoid that some pros and cons and recommendations represent our personal views based on our own experiences, and some readers might not agree in every point.

散射修正方法的最终质量，是具体的散射估计方法和补偿算法的结果。但是，本文我们没有讨论散射估计方法的影响，这是另一篇文章的主题。尽管如此，我们在表2中给出了不同散射补偿算法的总结。我们给出了一些支持观点、反对观点和推荐意见，不可避免的反应了我们自己经验和观点，有的读者会有不同意的可能性。

The reconstruction-based iterative improvement approach is intrinsically more accurate than projection-based methods due to applicability of more realistic physical–mathematical scatter estimation models, however, is very demanding and might be restricted to high-end CT systems or recommended as a final step after use of faster projection-based methods. In the third column of Table II, we have noted the presence of empirical parameters as a possible drawback, since the efficiency of the methods depend on the experienced user. We do not recommend heuristic image postprocessing approaches, since in our opinion, they are symptomatic and cosmetical cures but are not modeling and tackling the physical causes. There is a risk of unexpected effects and quantitative errors. Nevertheless, cosmetically reducing artifacts might deliver some nice images for marketing.

基于重建的迭代改进方法，与基于投影的方法相比，本身就会更准确，因为可以应用更实际的物理-数学散射估计模型，但是需求也更高，可能仅限于更高端的CT系统，或在使用了更快的基于投影的方法后，作为最后的处理步骤。在表2的第三列，
我们指出经验参数的存在，是可能的缺点，因为方法的效率取决于有经验的用户。我们并不推荐启发式的图像后处理方法，因为在我们的意见中，这些方法比较肤浅，并没有对真实的物理原因进行建模和处理。有一些不可预期的后果和量化误差的风险。尽管如此，修饰性的降低伪影可能会得到很不错的图像。

Not all scatter correction approaches from other fields, such as digital radiography, mammography, and emission CT, have yet been exploited in CBCT. This is true particularly for Bayesian methods, which seem to have the potentiality to subdue the dSNR preservation limitation by incorporating a priori information.

其他领域，如DR，乳腺成像，和ECT，的散射修正方法，并没有在CBCT中得到完全的利用。对于贝叶斯方法来说这尤其正确，这类方法似乎有这种潜力，通过纳入先验信息处理保持dSNR限制的问题。

The choice of the appropriate and optimal scatter correction method depends widely on the accuracy, speed, and limitations of the acquisition and imaging systems and on the application task. If the image quality is mainly restricted by residual errors of the mechanical and the detecting system and not limited by quantum noise, then fast projection-based deterministic scatter approaches might be sufficient. However, in high-end CT systems with excellent low-contrast detection capability or in low-dose applications more sophisticated methods including statistical optimization models are recommended.

选择合适最佳的散射修正方法，依赖于图像获取成像系统的准确率、速度和限制。如果图像质量主要受到机械和检测系统的误差限制，而并不是受到噪声的限制，那么快速的基于投影的确定性散射方法可能就可以。但是，在高端的CT系统中，有很好的低对比度检测能力，或在低剂量的应用中，我们推荐更复杂的方法，包括统计优化模型。

### 8.2 Conclusion

Scatter management is one of the most important issues in large-detector volume CT. First of all, scattered radiation should be suppressed by hardware prepatient and postpatient devices to a level as low as reasonably achievable.

散射管理在大探测器体CT中是一个重要的问题。首先，散射辐射应当由患者前和患者后的设备抑制到尽量低的水平。

Adapting the quotation from Lanczos “a lack of information cannot be remedied by any mathematical trickery,” we would emphasize scatter once present cannot be annihilated by mathematical trickery. What indeed can be done by mathematical correction methods is to mitigate the detrimental effects of scatter on image quality. In order to achieve optimal results, the models have to be as close as possible to reality, and the correction procedures have to take into account the stochastic nature of signal and scatter in a statistical framework. The consistency relation between measured data and reconstructed image should be kept in mind. By optimization approaches, an optimal balance of noise suppression and image characteristics by introducing adaptive regularization priors can be obtained. In this field of research, improvements can still be expected.

Lanczos有句话，信息的确实不能由任何数学技巧来弥补，我们从这句话进行一下修改，我们需要强调，散射一旦存在，就不能由数学技巧进行消除。数学修正方法所能做的，是弥补散射对图像质量的有害效果。为取得最优的结果，模型要尽可能接近实际，修正过程必须在一个统计框架中，考虑到信号的随机本质和散射。测量的数据和重建图像的一致性要考虑到。通过优化方法，可以在噪声抑制和引入自适应的正则化先验的图像特征中，找到一个平衡。在这个研究领域中，可以期待有进一步的改进。