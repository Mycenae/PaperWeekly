# State of the Art of CT Detectors and Sources: A Literature Review

## 0. Abstract

The three CT components with the greatest impact on image quality are the X-ray source, detection system and reconstruction algorithms. In this paper, we focus on the first two. We describe the state-of-the-art of CT detection systems, their calibrations, software corrections and common performance metrics. The components of CT detection systems, such as scintillator materials, photodiodes, data acquisition electronics and anti-scatter grids, are discussed. Their impact on CT image quality, their most important characteristics, as well as emerging future technology trends for each, are reviewed. The use of detection for multi-energy CT imaging is described. An overview of current CT X-ray sources, their evolution to support major trends in CT imaging and future trends is provided.

对影像质量影响最大的三个CT组件是，X射线源，检测系统和重建算法。本文中，我们关注前两个部分。我们描述了目前最好的CT检测系统，其校准，软件修正和常用的性能度量标准。CT检测系统的组成部分，比如闪烁体材料，光敏二极管，数据获取电路和反散射网格，我们进行了讨论。其对CT图像质量的影响，其最重要的特征，以及正在兴起的未来技术，都进行了回顾。多能CT成像中的应用也进行了讨论。目前的CT X射线源进行了回顾，其在CT成像中的主流趋势和未来趋势进行了总结。

**Keywords** CT detection systems / CT sources / Detection based spectral CT / CT detection components

## 1. Introduction

Over the last two decades, CT detection and source technologies evolved to support three major CT imaging trends [1–4]: increasing number of slices, increased speed of acquisition and dose reduction.

在过去二十年中，CT探测器和源的技术经过进化，有三个主要的成像趋势：slice数量增加，获取速度增加，剂量降低。

Solid-state detectors, segmented into detector elements arrays, were the main enabler for the emergence of multi-slice CT scanners. CT sources evolved to support larger coverage per rotation.

固态探测器，分割成了探测器元素阵列，是多slice CT扫描仪兴起的主要赋能因素。CT源的进化，可以在每次旋转中支持更大的覆盖面积。

In order to enable increased speed of acquisition, X-ray sources were required to support greater accelerations and increased peak power while the detections systems evolved to support shorter integration periods.

为增加获取数据的速度，要求X射线源支持更快的加速和增加峰值能量，而探测系统要支持更短的集成周期。

Finally, over the last few years, the emergence of iterative reconstruction is driving the use of low and ultralow dose acquisition. This has a direct impact on the need for a lower noise floor in the detection systems.

最后，在过去很多年中，迭代重建算法的兴起，驱动着使用超低剂量获取图像。这要求探测系统中有更低的噪声。

## 2. State of the Art CT Detectors

**Current Detector Designs**

Most manufacturers share a common detector design. The compact design has three essential layers: conversion of X-ray to light (scintillator), light to current (photodiode), and a substrate to provide the mechanical and electrical infrastructure (Fig. 1).

制造商们的探测器设计多数是一样的。紧凑的设计有三个基本层：将X射线转化成可见光（闪烁体），可见光转化到电流（光敏二极管），提供机械和电力基础的基底。

Detectors span roughly one meter with varying axial coverage in order to image most of the population. See Fig. 2.

探测器的长度可以达到大致一米，轴向长度不同，以对不同的人进行成像，如图2所示。

For efficiency, detector modules are built from arrays of detector elements. Figures 3 and 4 show a scintillator and a photodiode segmented into elements. A typical module structure is shown in Fig. 5.

为了效率，探测器模块是由探测器元素阵列构成的。图3和图4展示了一个闪烁体和一个光敏二极管分割成单元的样子。图5给出了一个典型的模块结构。

**Detector Systems**

Detector characteristics are crucial for obtaining good CT image quality. The main requirements are: accuracy, dynamic range, stability (short- and long-term), uniformity, speed of response, resolution, geometric efficiency, detector quantum efficiency and cross-talk (spatial and temporal).

探测器的特性，对于得到高质量的CT图像是非常关键的。主要的需求是：准确率，动态范围，稳定性（短期和长期），均匀性，响应速度，分辨率，几何效率，探测器量子效率和串扰（空间上的和时间上的）。

Accuracy: Measuring X-ray flux is essential in order to measure small differences in tissue density, e.g., ～0.1 % for brain imaging [5].

准确率：测量X射线通量，对于测量组织密度间的很小的差异，是非常关键的，如，对大脑成像中的大约0.1%的变化。

Dynamic range: Due to the exponential behavior of attenuation, signal in the detectors can vary significantly over 10^4. Current CT systems can have dynamic ranges above 10^5.

动态范围：由于衰减的指数性质，到探测器上的信号可以以10^4的量级变化。目前的CT系统的动态范围超过10^5。

Stability: Third-generation CT detectors have to produce the same signal for the same irradiation within a scan and between system calibrations.

稳定性：三代CT探测器要在一次扫描中，在系统校准之间，为同样的辐射，给出同样的信号。

Speed of response: In modern scanners, the frame acquisition time can be of the order of 135 μs, like in the Philips iCT scanner [6].

响应速度：在现代扫描仪中，帧获取时间可以达到135 μs，如在Philips iCT扫描仪中。

Resolution: The geometric resolution of a scanner is dominated by the size of the detection element and X-ray source focal spot size. For a uniform focal spot s and uniform detector element d, the resolution of a scanner with Magnification M is [7]

分辨率：一个扫描仪的几何分辨率，由探测元素的尺寸，和X射线源的焦点大小决定。对于均匀的焦点s，和均匀的探测器元素d，一个放大率为M的扫描仪的分辨率为

$$Reff = \frac{1}{M} \sqrt {d^2 + (M-1)^2 s^2}$$

Detector pitch, or sampling size: Sampling will limit the observability of small objects through Nyquist theorem. The pitch is usually larger than the detector element; therefore, special care is required to avoid sampling artifacts.

探测器螺距，或采样大小：采样会通过Nyquist定理，限制小目标的可观测性。螺距通常比探测器元素要大；因此，需要特别的关注，以避免采样伪影。

Cross-talk [8]: Defines how much of the signal generated on one detector element influences (or leaks) into a neighboring pixel. Normally in arrays this number is a few percent.

串扰：定义了一个探测器元素上生成的信号，对临近像素的影响（或泄漏到临近的像素中）。在阵列中，这个数值正常是几个百分点。

To simplify and expand beyond 64 slices, most vendors opted for sub-modules within a module [9–11]. Philips chose the concept of tiles [12]. Coverage can be obtained by adding additional tiles, see Fig. 6.

为简化，并拓展到超过64个slices，多数供应商选择在一个模块中有几个子模块。Philips选择了tile的概念。通过增加更多的tiles，可以得到更多的覆盖面积，如图6所示。

**Detection Performance Metrics**

Geometric detection efficiency (GDE) is the ratio between the incident X-ray photons and the photons hitting the detection active area. It mainly depends on the fraction of the detector active area relative to its size (fill factor). Typically this number is above 70 % for most scanners.

几何探测效率(GDE)是入射X射线光子和撞击到探测器活跃区域的光子的比率。它主要取决于探测器活跃区域占其总尺寸的百分比（填充率）。对大多数扫描仪，这个数值都超过70%。

Detective quantum efficiency (DQE) [13] is the ratio between the square of the signal-to-noise SNR^2 at the detection output and the incident SNR^2 [14, 15]:

探测量子效率DQE是探测输出的信噪比平方与入射的信噪比平方之比：

$$DQE = \frac {SNR^2_{out}} {SNR^2_{in}}$$

Most X-ray DQE literature addresses radiography where the image is directly related to the detection performance. IEC 62220-1 introduced DQE as an international standard. While this standard excludes CT, it is still useful to quantify it for CT detection [1].

多数X射线DQE文献处理的是X射线放射成像，其中图像与探测器性能直接相关。IEC 62220-1提出DQE作为一个国际标准。虽然这个标准排除了CT，但对于CT探测器的性能量化，还是有用的。

Detection efficiency (DE) is determined by the GDE and the DQE [1]: 探测效率DE由GDE和DQE共同决定：

$$DE = GDE × DQE$$

Another method to characterize DQE is to derive it from measured quantities [16, 17]: 另一个表征DQE的方法是将其从测量的量中推导出来：

$$DQE(f) = \frac {g^2 MTF^2 · Ø} {NPS(f)} = \frac {s^2 MTF^2} {NPS(f) · Ø}$$

where S is the signal, MTF (modulation transfer function) the frequency response, NPS the noise power spectrum, and Ø the X-ray quanta per area at the detector input, which corresponds to the measured NPS. Using the right hand form with the measures signal S is only applicable when the detector response is linear and has zero intercept.

其中S是信号，MTF是频率响应，NPS是噪声能量谱，Ø是在探测器输入处的单位面积中的X射线量子，对应着测量到的NPS。对测量的信号S使用右手形式，只有当探测器响应是线性的，截距为0时，才能应用。

This formulation of the DQE is widely used to quantify DE [18]. Ranger et al. [19] compared the IEC method with two other ones and found that the measurements technique can bias the results by up to 12.5%. In another publication [20] Ranger et al. proposed ''effective DQE'' (eDQE) which takes into account the entire system, including the focal spot blurring, scatter, and more.

DQE的这个公式广泛用于量化DE[18]。Ranger等[19]比较了IEC方法与其他两种，发现测量技术可以使结果偏移达12.5%。在另一篇文献[20]中，Ranger提出了有效DQE (eDQE)，考虑到了整个系统，包括焦点模糊，散射和更多的东西。

Measured DQE often include the geometrical efficiency. Luhta et al. [12] reported DQE of 0.78 with fill factor of 0.82, namely GDE = 0.82 and quantum efficiency of 0.95. The DQE decreases at low dose since electronic noise starts to play a role in the overall NPS.

测量的DQE通常包括了几何效率。Luhta等[12]给出的DQE为0.78，填充系数为0.82，即GDE=0.82，量子效率为0.95。DQE在低剂量率时会降低，因为电子噪声在NPS中起到了很大的作用。

**Detectors SW Corrections and Calibrations**

The image formation chain in CT includes several detector-oriented calibrations and corrections, essential to prevent artifacts (scatter corrections not included in this section). The output signal of each detector pixel is supposed to be linear with the input radiation intensity (the energy sum of all incident photons within a unit time). The overall gain of each pixel, normalized to the incident flux, is measured separately without the presence of any object. This has to be repeated periodically, depending on the system and temperature stability. However, the emerging of MDCT raised the need to correct gain variations along the axial direction using the scan raw data [21]. Detectors’ offset are sampled before every scan, without X-ray. Image quality is sensitive to offset stability within the scan, especially at low radiation signal (low dose), causing ring artifacts [1], (Fig. 7).

在CT中的图像形成链中，包括几种面向探测器的校准和修正，对于防止伪影的出现是很关键的（散射修正并没有在本节中叙述）。每个探测器像素的输出信号，假设与输入辐射强度（在单位时间中所有入射光子的能量和）呈线性关系。每个像素的总体增益，对输入通量进行归一化，在不存在任何目标的情况下，独立进行测量。这需要周期性的重复，依赖于系统和温度稳定性。但是，MDCT的出现，提出了沿着轴向修正增益变化的需求，使用的是扫描的原始数据[21]。探测器的偏移在每次扫描之前，在没有X射线的情况下进行取样。图像质量对于扫描中的偏移稳定性很敏感，尤其是在低辐射信号时（低剂量），会导致环状伪影（图7）。

The overall non-linear response of CT detectors, causing streaks and rings artifacts, is composed mostly of spectral non-linear effects, and electronics readout non-linearity (the latter is expressed mainly at low dose). It is corrected, relative to the average response of all the detector pixels, using special calibration phantoms [22]. The results are implemented then as part of channel-to-channel corrections.

CT探测器的总体非线性响应，导致了条纹伪影和环状伪影，这主要是由谱非线性效应，电子读出的非线性构成的，后者主要在低剂量处表达出。使用特殊的校准模体，相对于探测器所有像素的平均响应，可以进行纠正。结果的实现是通道对通道纠正的一部分。

Detector displacement, beyond a certain tolerance, from their positions as considered by the reconstruction system is a cause of artifacts and image distortions. Correcting it along the fan is based on a direct measurement with special off-centered phantoms [23], while for a 2D detector array, some techniques taken from less rigid 3D tomographic systems as C-Arm CT and SPECT are used [24].

在其重建系统认为的位置的探测器偏移，超过了一定的容忍度，是伪影和图像形变的一个原因。沿着fan对其进行修正，是基于直接的测量，使用了特殊的中心偏移模体，而对于一个2D探测器阵列，使用了一些没有那么严格的3D断层系统的技术，如C形臂和SPECT。

The 2D array of small detection pixels, characterizing modern MDCT, is susceptible to a fairly large signal cross-talk. The cross-talk is mainly of optical type, through the thin reflectors separating between scintillator pixels, and through the optical interface with the photodiodes, while a smaller fraction of cross-talk is in the photodiode arrays. By itself, it mainly causes a reduction in MTF (Fig. 8), while ring artifacts may show up for non-uniform cross-talk along the array. Corrections may vary from simple linear subtraction to sophisticated deconvolution schemes [25].

小探测像素的2D阵列，是现代MDCT的特征，可能形成较大的信号串扰。串扰主要是光学类型的，穿过了分离闪烁体像素的细细的反射器，穿过了带有光敏二极管的光学界面，而一小部分串扰是在光敏阵列中。其主要导致了MTF的下降，而对于沿着阵列的非一致串扰，环状伪影就可能出现。有多种修正方法，从简单的线性相减，到复杂的解卷积方案。

Finally, scintillator afterglow can cause ring artifacts, MTF reduction, and image distortion, especially for fast-rotating CT modes with short integration periods. The need for correction depends on the amount of afterglow of the scintillator in use. For low-afterglow scintillators like GOS, this is not required for most CT scans, except the very fast ones that use large number of frames. For slower scintillators with large afterglow, like the GE HiLight, an after-glow correction is required for most scan modes [26], while an advanced correction solution, applicable for both afterglow and crosstalk, is described in Ref. [25] (see Fig. 9).

最后，闪烁体的余晖会导致环状伪影，MTF降低，和图像扭曲，尤其是对于带有快速集成周期的快速旋转CT模式。修正的需求依赖于使用的闪烁体的余晖的量。对于低余晖的闪烁体，如GOS，多数CT扫描都不需要进行修正，除了非常快速的，使用大量帧。对于慢速的闪烁体，有很大的余晖，如GE HiLight，对于多数模式都需要进行余晖修正，一种高级的修正解决方案，可以应用于余晖和串扰，在[25]中进行了描述。

## 3. Detection Components

**Scintillators for CT**

A scintillator is a luminescent material converting linearly high-energy photons into visible light [27, 28, 29•], available for readout and further processing using a suitable photo-detector. Scintillators are widely used in detection systems for medical imaging, industrial inspection, nuclear medicine, and high-energy physics [30–33]. Specifically, inorganic scintillators have been an integral part of CT detection systems ever since their introduction into clinical practice.

闪烁体是一种发光材料，将高能量光子线性的转换成可见光，使用一个合适的光探测器，就可以用于读出，和进一步的处理。闪烁体广泛用于医学成像、工业检测、核医学和高能物理的探测系统中。特别的，自从CT引入到临床实践中后，无机闪烁体就一直是其一个组成部分。

The requirements for scintillators used in CT detectors are probably the most demanding among the various medical imaging modalities. These include high light output (accounting for X-ray conversion efficiency and optical transparency), high X-ray stopping power, good spectral match with the photo-detector, short primary decay time (up to tens of μs), low afterglow, radiation damage resistance, light-output stability (time, temperature), compact packaging, and easy machining. In many cases it is uniformity of a certain property that is more important and more challenging to achieve, rather than meeting a required absolute value.

在CT探测器中使用的闪烁体的要求，在各种医学成像模态中可能是要求最高的。这包括很高的光输出（包括X射线转化效率，光学透明性），很高的X射线停止能量，与光探测器的频谱要匹配，初级衰减时间要短（最多几十μs），低余晖，抗辐射损伤，光输出的稳定性（时间，温度），紧凑封装，易于加工。在很多情况下，是一种特定性质的均匀性非常重要，难以达到，而不是达到一个特定的绝对值。

These demanding CT requirements make single crystals and polycrystalline ceramics the most suitable types of scintillators. Among these, the list of useful materials has been quite limited; to date, the scintillators mostly used have been CdWO_4 [29•], Gd_2 O_2 S:Pr,Ce (GOS) [34], (Y,Gd)_2 O_3:Eu [35], and recently the GE Gemstone^TM [36•].

这些要求很高的CT需求，使得单晶和多晶陶瓷最适宜做闪烁体。在这些里面，有用材料的列表非常有限；迄今为止，用的最多的闪烁体是CdWO_4 [29•], Gd_2 O_2 S:Pr,Ce (GOS) [34], (Y,Gd)_2 O_3:Eu [35], 和最近的GE Gemstone^TM [36•]。

Scintillators for multi-slice CT geometry are made in two-dimensional (2D) arrays, with a typical pixel size of ∼1 mm. The arrays packaging also includes a reflective material matrix, typically consisting of a mixture of a high-reflectance pigment (e.g., TiO2) and a binder (e.g., optical epoxy), or a certain multi-layer structure (e.g., sputtered silver on a polymer). The purpose of the reflective matrix is both to mechanically support the pixilated scintillator array and to efficiently transport the isotropically emitted scintillation light to the photo-detector, with minimal crosstalk.

多slice CT几何的闪烁体做成了2D阵列，典型的像素大小为∼1mm。阵列封装还包括了反射材料矩阵，通常由高反射颜料（如，TiO2）和黏合剂（如，光学环氧树脂），或特定的多层结构（如，在聚合物上溅射上银）。反射矩阵的目的，是从机械上支持闪烁体阵列，并高效的将各向同性发射的闪烁光传输到光探测器上，串扰达到最小。

In the pursuit of the ''ideal scintillator'' [37], new materials, packaging and geometries that will meet the high rotation speed, high resolution, and lower patient doses requirements of modern CT scanners are evaluated. The investigation focuses on light yield, speed, afterglow, and transparency. The light yield of scintillators currently in use in CT detectors is quite far from the theoretical limit given by N_ph = 10^6/E_g, where N_ph is the number of visible photons per 1 MeV gamma photon, E_g is the scintillator band gap, and β is a numerical factor of ≈2.5 (see [33] and references within). For example, the theoretical limit for GOS is N_ph ∼90,000.

在追求理想闪烁体的过程中，满足高转速，高分辨率，低剂量需求的新材料，封装和几何，需要进行评估。调查聚焦在光产生率，速度，余晖，和透明度上。目前CT探测器中使用的闪烁体的光产生率与理论极限相距较远，理论极限是N_ph = 10^6/E_g，其中N_ph是每1MeV gamma光子产生的可见光光子数量，E_g是闪烁体能谱间隙，β是一个数值因子≈2.5。比如，GOS的理论极限是N_ph ∼90,000。

Several groups of new materials are being evaluated for modern CT applications. One very promising group of materials is the garnet of the type (Lu,Gd,Y,Tb)_3 (Ga,Al)5 O_12. These materials, available both in single-crystal and polycrystalline-ceramics forms, offer superior transparency, increased light yield, very short decay times, and improved spectral match with the photo-detectors [38]. The GE Gemstone^TM has been the first garnet-scintillator introduced commercially for CT detection. Another group of materials, evaluated for the Philips dual-layer detectors, are low-Z scintillators such as ZnSe:Te, used for detecting the low-energy part X-ray spectra [39]. Examples of raw-material (wafer) samples of a garnet-type, GOS, and ZnSe:Te scintillators are demonstrated in Fig. 10.

几组新材料已经进行了评估。一种非常有希望的材料组是，石榴石类型(Lu,Gd,Y,Tb)_3 (Ga,Al)5 O_12。这些材料，以单晶和多晶陶瓷的形式存在，有很多的透明性，光产出增加了，很短的衰减时间，与光探测器的谱匹配也变好了。GE Gemstone^TM是第一种使用石榴石闪烁体进行商用的CT探测系统。另一组材料，在Philips双层探测器上进行了评估，是低Z闪烁体，如ZnSe:Te，用于探测低能X射线谱。图10中给出了石榴石类型的，GOS和ZnSe:Te闪烁体的原始材料（晶圆）例子。

Additional materials with potential implementation in CT are halide ''super-bright'' scintillators, for example SrI2:Eu, the light yield of which is reported to exceed 90,000 photons/MeV [40].

可能应用在CT上的其他材料为，卤化物'超亮'闪烁体，比如SrI2:Eu，其光产出率据称超过90,000 光子/MeV。

New packaging geometries evaluated for future usage in CT include scintillating fiber-optics arrays, thin layers of printed scintillators, and composite scintillators. The latter technology, in which a powdered scintillator is dispersed in an optically matched organic resin, is highly attractive as it allows avoiding crystal growth or sintering processes, thus significantly reducing production costs.

评估用于CT的新的封装几何，包括闪烁光纤阵列，打印闪烁体薄层，和复合闪烁体。后者的技术中，粉末状的闪烁体散布于光学匹配好的有机树脂上，是非常有吸引力的，因为这允许晶体生长，或烧结的过程，因此显著的降低了生产成本。

**Photovoltaic Detector Array (PDA)**

The PDA collects light signals from scintillator arrays and converts them, linearly, to electric signals. It is a quantum detector where photons are converted to electron–hole pairs. These pairs are diffused to the junction area and contribute to the detector current.

PDA从闪烁体阵列收集光信号，并将其线性转换到电信号。这是一种量子探测器，其中光子转换成了电子-空穴对。这些对扩散到节点区域，对探测器电流有贡献。

Legacy CT detectors were based on front-illuminated PDA (Fig. 11). The newer CT detectors are based on back-illuminated PDA (Fig. 12). Back-illuminated PDA enables vertical integration that is required for tiled detectors [41, 42].

Legacy CT探测器是基于前向照明的PDA（图11）。更新的CT探测器是基于反向照亮的PDA（图12）。反向照明的PDA可以进行垂直整合，这是tiled探测器所需要的。

In order to reduce leakage currents, the PDA is operated at zero bias (between 100 and -100 μV).

为降低漏电流，PDA在零偏置时操作（在100μV和-100μV之间）。

Listed below are the main PDA characteristics required for a high-end CT scanner: 下面是一台高端CT扫描仪所需的主要的PDA特性：

- Responsivity (output current/input power) should be as close as possible to quantum efficiency of 100 % at the relevant wavelength. Typical value for 510 nm is >0.35 A/W. The design of the junction depth and the optical layers above the junction should be optimized to the scintillator wavelength (Fig. 13).

- 响应度（输出电流/输入能量）在相关的波长时，应当尽可能接近量子效率100%。对于510nm的典型的值是>0.35 A/W。关节深度的设计和关节之上的光学层应当对闪烁体的波长进行优化。

- Shunt resistance: should be high enough to minimize leakage currents and guarantee negligible noise level. Typical value is in the order of 1G ohm at room temperature (Fig. 14).

- 分流器电阻：应当足够高，以最小化漏电流，并保证噪声水平可忽略。典型的值是，在室温时，在1G ohm的数量级上。

- Cross-talk: Most of the electron–hole pairs are collected by the relevant junction, but some can diffuse to the neighbored junction and contribute to cross-talk. Acceptable cross-talk values are in the order of up to 4 % [43•].

- 串扰：多数电子-空穴对都是由相关的节点收集的，但一些可以扩散到相邻的节点，并形成串扰。可接受的串扰值是在最大到4%的水平上。

- Linearity: The detector characteristics should be linear along the full range of signals (typical range is from 1 pA to 1 μA). Acceptable nonlinearity is in the order of ±0.1 %.

- 线性度：探测器特性在信号的整个范围内应当是线性的（典型的范围是从1 pA到1 μA）。可接受的非线性度是在±0.1%的数量级。

- Response time: CT has short integration periods. A fast response in the order of 10–30 μs is required.

- 响应时间：CT的整合周期很短。需要在10-30 μs数量级的快速响应。

An emerging structure of PDA, (Fig. 15) is based on epitaxial silicon grown on thick silicon substrate plus through silicon via (TSV) delivering the contacts to the back side.

PDA的一种正在出现的结构（图15），是基于在厚硅基底上的外延硅生长。

**Data Acquisition Electronics**

CT data acquisition electronics, in integration mode, collects the electrical signals from the PDA and convert them to digital signals with the required noise level, bandwidth and dynamic range.

CT数据获取电路，在整合模式下，从PDA收集电信号，将其转化成数字信号，噪声水平、带宽和动态范围都要达到要求。

Each photodiode (representing a pixel) is connected to a dedicated low noise pre-amplifier. The signal is integrated over a certain period (integration period) and sent to the next stage after conversion to digital format.

每个光敏二极管（表示一个像素）都连接到一个精密的低噪声预放大器。信号的几个周期积分到一起（积分周期），在转换成数字格式之后送入到下一个阶段。

The classical acquisition electronics is based on array of low noise preamplifiers and analog-to-digital converter [44]. In Fig. 16, 128 inputs from the photodiodes are connected to preamplifiers and to two analog-to-digital converters.

经典采集电路是基于低噪声预放大和ADC的阵列的。在图16中，从光电二极管的128个输入，连接到预放大器，和两个ADC上。

Philips presented a different concept of data acquisition electronics [12] based on a current to frequency converter as shown in Fig. 17.

Philips提出了一种不同的数据采集电路，是基于电流到频率的转换器的，如图17所示。

There are several important characteristics required for high-performance data acquisition electronics of CT scanner: CT扫描仪的高性能数据采集电路有几个重要的特征：

- Dynamic range: the dynamic range is in the order of 18 bits, which covers a range of input current from 1 μA down to a few pA. 动态范围：动态范围是在18bit范围，输入电流可以从1 μA到几pA。

- Noise: for low-dose applications, the total electronic noise is significant and should be in the range of a few pA. The 1/f noise is also of importance, as it may cause rings or bands in the image. 噪声：对低剂量应用，总电路噪声是显著的，应当在几pA的范围之内。1/f噪声也是很重要的，可能会导致图像中的环或带。

- Linearity is critical in order to achieve high image quality, and its deviation should be better than ±0.05 % relative to an ideal linear response. 线性性对于得到高质量的图像是很关键的，与理想线性响应相比，其偏差应当在±0.05%以下。

- The sampling rate of the A/D converter is in the range of 3kSPS–10kSPS. ADC的采样率应当在3kSPS–10kSPS之间。

- Power dissipation: the number of channels is in the order of 40,000–60,000 for a 4-cm detector. This dictates careful design with power dissipation in the range of few mW per channel. 能量耗散：对于一个4cm探测器来说，其通道数是在4-6万之间。这需要仔细的设计，其能量耗散在每个通道几mW的水平。

A technology trend of incorporating the PDA and the data acquisition electronics [45] may help to achieve improvement in performance and lower cost. 一个技术趋势是将PDA和数据采集电路整合到一起，可能得到改进的性能和更低的功耗。

The PDA is connected to the data acquisition electronics using a TSV technology. PDA与数据采集系统的连接方式是TSV技术。

**Anti-Scatter Grids**

Current CT reconstruction theory assumes that X-ray photons are absorbed or pass through scanned objects without interaction. In practice, only a small portion of the incident X-ray radiation is directly absorbed by the photo-electric effect, while most undergo coherent (Rayleigh) [46, 47•] or incoherent (Compton) [47•, 48] scattering. For materials with high Z, K-fluorescence should also be considered [47•].

目前的CT重建理论假设，X射线光子在经过要扫描的目标时是吸收或通过的，没有相互作用。在实践中，只有一小部分入射X射线辐射是通过光电效应直接吸收的，而大多数要经过相干或非相干散射。对于高Z值的材料，还要考虑K-荧光效应。

The main contributor to the scattered radiation-related artifacts and image quality degradation is multiple Compton scattering effect [47•]. It has a substantial influence on conventional (non-spectral) CT systems, and yet a more significant effect on spectral CT, and should be treated accordingly [47•, 49]. When scanning large patients without means for scatter reduction, the scattered radiation contribution reaches and even overcomes the direct one [47•].

与散射辐射相关的伪影和图像质量下降，其主要贡献者是多级Compton散射效果。这对传统CT（非能谱）系统有重大的影响，而对能谱CT有更大的影响，需要相应的进行处理。当扫描很大的患者，不进行散射抑制，散射辐射的贡献达到甚至超过了直接辐射。

Provided that the ratio of scattered to direct photons is sufficiently low, the image quality is not much affected. With increasing scatter to primary ratio (SPR), image artifacts emerge, mainly in the form of cupping, streaks and degradation in image quality, mainly low CNR and CT numbers shifts [50, 51].

假设散射光子与直接透射光子的比足够低，图像质量并没有受到很大影响。随着散射初级比(SPR, scatter to primary ratio)增加，图像伪影出现了，主要是cupping，条纹和图像质量降低的形式，表现在CNR较低，和CT值偏移。

Various SW algorithms target scattered radiation artifacts [50–53], but even in a hypothetical case of ideal correction the total noise increases by a factor of $\sqrt{1+SPR}$ comparing to scatter-free case [51].

各种软件算法的目标都是散射辐射伪影，但即使在假想的理想修正情况下，与无散射情况相比，其总计噪声增加了$\sqrt{1+SPR}$。

Today, a key solution for effectively reducing scattered radiation are anti-scatter grids (ASGs) used as collimators in front of detectors [54], enabling scatter reduction by over a factor of 10 [47•]. Both 1D and 2D ASGs are used in CT scanners [47•, 55, 57] (Fig. 18); 2D ASGs generally reduce more effectively scattering [47, 55], especially for scanners with a large axial collimation (Fig. 19).

今天，有效降低散射辐射的一种关键方法是反散射网格(ASGs, anti-scatter grids)，其作用是探测器前的准直器，使得散射降低了超过10倍。1D和2D ASGs都用于CT扫描仪；2D ASGs一般可以更有效的降低散射，尤其是对于有大型径向准直的扫描仪。

Materials used to make AS lamellas are of high Z numbers allowing effective absorption of scattered radiation.

制造反散射薄片的材料，是高Z值的材料，可以有效的吸收散射辐射。

Different aspects of ASGs should be carefully designed to avoid induced artifacts and image quality degradation. These include impact on dose utilization, precise alignment, non-uniform scatter rejection, thermal and mechanical instabilities, and reliability related issues. Cost of ASG is also significant consideration.

需要仔细设计ASGs的不同方面，以避免引入伪影，降低图像质量。这包括利用剂量的效果，精确对齐的效果，非一致散射抑制的效果，热和机械不稳定性，和可靠性相关的问题。ASG的价格也是显著的考虑。

Though ASGs reduce the scattered radiation to tolerable levels, algorithmic corrections are still required to further suppress scatter-induced artifacts and image quality degradation. Use of ASGs with scatter correction algorithms (Fig. 20) and techniques reducing scattered radiation from the X-ray source and its surroundings is the most successful approach for MDCT [47•, 52, 57].

虽然ASGs将散射辐射降低到了可忍受的水平，仍然需要算法的修正，以进一步抑制散射带来的伪影和图像质量降低。使用ASGs和散射修正算法（图20）和技术，降低X射线源及其周围的散射辐射，是MDCT的最成功的方法。

## 4. Detector-Based Spectral CT

**Dual-Layer Detector**

Material decomposition through energy-selective CT was proposed by Alvarez and Macovski in 1976 [58]. A dual-layer detector for a simultaneous acquisition of two energies in CT was first proposed by Brooks and Chiro [59] in 1978. A Philips Healthcare team proposed a different configuration and implementation of that idea [60••, 61] through two attached scintillator layers, optically separated, and read by a side-looking, edge-on, silicon photodiode, thin enough to maintain the same detector pitch and geometrical efficiency as a conventional CT detector (Fig. 21).

通过能量选择CT来进行材料分解，这由Alvarez和Macovski在1976年提出[58]。双层探测器，在CT中同时获取两种能量，首先由Brooks和Chiro在1978年提出。Philips一个小组提出了一种不同的配置，通过两个相连的闪烁体层，光学上是分离的，由一个side-looking，edge-on，硅发光二极管，足够薄以维持与传统CT探测器相同的探测器螺距和几何效率，实现了这个想法。

The top scintillator layer’s atomic number and thickness have been optimized to maximize energy separation at 140 kVp, while maintaining high enough signal statistics for the low-energy raw data even for a large patient. ZnSe advantage in light yield [39] (～70 % better than GOS) contributes to a high SNR in the top (low-energy) layer detector, enabling it to function at very low dose without causing artifacts, typical to electronic-noise dominant signals.

闪烁体上层的原子序数和厚度是经过优化的，最大化140kVp的能量分离，而对于低能原始数据即使在大型患者的情况下也保持足够高的信号统计数据。ZnSe在光子产生率上的优势（～70%，比GOS要好），使得上层探测器（低能）得到高SNR，使其在很低的剂量时也能使用，而且不产生伪影，这是典型的电子-噪声主导的信号。

The mean energy separation of the dual-layer detector, at 140 kVp, with and without a 32-cm water absorber, is shown in Fig. 22.

双层探测器在140kVp的平均能量分离，在有和没有32cm水模体的情况，如图22所示。

The decrease in energy separation with increasing patient size, as well as the unavoidable overlap of the two spectra, had already been noticed to reduce the SNR of material decomposition in this method compared to using two separate kV values [62]. Some of these drawbacks are compensated by the fully simultaneous acquisition of the two energy spectra by the dual-layer detector, leading to a more accurate determination of material concentrations (e.g., iodine). In addition, the dual-layer detector method doesn’t suffer from some of the practical constraints on high-enough dose delivery of 80 kVp at fast scanning modes that are characteristic of modern multidetector CT. Furthermore, at 120 kVp, the dual-layer configuration still enables iodine quantification from soft tissues with an SNR lower than that obtained at 140 kVp by only ～18 %. The latter observation, together with the dual-energy acquisition characteristic, independent of the CT scan protocol and field of view, enables the use of the system as a dual-energy CT in a retrospective mode after the scan, upon users’ request and need. Using edge-on photodiodes between the detector columns prevents optical cross-talk along the detection-arc dimension, leading to a better MTF than in conventional CT detector.

随着患者体型增大，能量分离的降低，以及不可避免的两个能谱的重叠，已经被注意到了，能降低这个方法中材料分解的SNR，这是与使用两个分离的kV值相比较。一些缺陷由这两种能谱被双层探测器的同时获取得到了补偿，带来了材料聚焦的更准确确定值。另外，双层探测器方法比现代多探测器CT有一个优势，即在80kVp时，多探测器CT在快速扫描模式时需要很高的剂量，双层探测器没有这个限制。而且，在120kVp时，双层的配置仍然可以从软组织中分离出碘的量，比在140kVp时的SNR低了～18%。后者的观察，与双能获取的特征，与CT扫描的协议和FOV独立，使得系统可以作为双能CT的回顾模式使用，这是在扫描之后，视用户的要求和需求而定的。在探测器列之间使用edge-on光电二极管，沿着探测弧防止光串扰，可以比传统CT探测器得到更好的MTF。

Figure 23 displays an iodine image, obtained with a dual-layer CT prototype in Hadassah MC Jerusalem, demonstrating that the iodine SNR is good enough to detect a 2-mm non-perfused nodule [63•], while Fig. 24 displays a virtual-non-contrast image obtained with the same system.

图23给出了一幅碘图，用双层CT原型机得到的，说明碘SNR非常好，可以检测到一个2mm非弥散性结节，而图24展示了同样的系统得到的一个虚拟非对比图像。

**Photon Counting Detectors**

In photon counting (PhC) detectors X-ray photons are counted individually and their energy is assessed. To achieve counting of individual photons, the scintillator and photodiode are replaced by a direct conversion material (DiCo) [64, 65] and the signal integrating ASIC is replaced by a fast counting ASIC, enabling the processing of charge clouds formed by individual X-ray photons [66–69]. Figure 25 illustrates the difference between energy integration–based detection, dual-layer detection, and PhC detection, while Fig. 26 describes the operation principle of a PhC detector.

在光子计数探测器(PhC)中，X射线光子逐个计数，其能量进行评估。为得到单个光子的计数，闪烁体和光电二极管替换成了直接转换材料(DiCo)，信号集成的ASIC替换成了快速计数ASIC，使得单个X射线光子形成的电荷云可以进行处理。图25描述了基于能量集成的检测，双层检测，和PhC检测，图26则描述了PhC检测的操作准则。

Photon-counting CT is a disruptive technology, considering the features that contribute to its attractiveness. It is capable of operating at lower dose, as counting is less disturbed by electronic noise [70]; and an improved CNR by different energy weighting of the detected photons [71–73].

光子计数CT是一种破坏性的技术。其可以在更低的剂量下运行，因为计数受到电子噪声的扰动很少；探测到的光子的不同能量加权也可以得到更好的CNR。

An appropriate DiCo material has a high X-ray attenuation coefficient, low e-hole generation energy, and high mobility-lifetime product, contributing to charge collection efficiency, affecting the DQE and energy resolution. CdTe and Cd(Zn)Te are promising candidates, and several detectors, animal scanners [74••, 75] and limited (in energy bins or flux) prototypes have been presented with these II–VI materials [67, 76]. However, several challenges, including polarization, stability and low yield of high-quality material remain the main roadblocks for full commercialization. Silicon and GaAs have been considered too for this purpose.

一种合适的DiCo材料，有一个很高的X射线衰减系数，很低的电子空穴生成能量，很高的移动生命时间的产品，对电荷收集效率会有很大贡献，影响DQE和能量分辨率。CdTe和Cd(Zn)Te是有希望的候选材料，几种探测器，动物扫描仪和有限原型已经给出了。但是，有几个挑战，包括极化，稳定性和高质量材料的低产出率，是完全商业化的主要障碍。硅和GaAs已经考虑做此用途。

A typical ASIC for PhC would have a charge sensitive amplifier to amplify the charge cloud signal and a shaper to shape the signal before discriminating it into energy bin counters [66–69]. Pixel summing and pile-up correction algorithms are employed to correct the distorted measured spectra [68] and signal pile-up [77].

PhC的典型ASIC，需要有一个对电荷敏感的放大器，以放大电荷云信号，并有一个shaper，在将其区分进入能量bin计数器之前以形成信号。像素求和和堆积修正算法用于修正扭曲的测量的谱和信号堆积。

Photon-counting CT can go beyond dual-energy imaging, as typically more than two energy windows are available, improving material separation performance. Once measured photons are divided into three or more energy bins, k-edge imaging becomes possible using special targeted contrast materials and enabling new applications [74••, 78].

PhC CT可以超过双能成像，因为一般可用的是多于两个能量窗口，改进材料分离性能。一旦测量的光子分成三个或更多的能量bins，k-edge成像就成为可能，使用特殊目的的对比材料，并使新的应用成为可能。

## 5. CT Sources

**Current CT Sources**

Although alternatives have been thoroughly studied, CT still relies on Bremsstrahlung. Improvements of X-ray sources during the last decade have enabled a great extension of the use of CT.

CT仍然依赖于韧致辐射，其他的方法也进行了彻底研究。在过去十年中，X射线源的改进已经将CT的使用进行了很大的拓展。

Novel X-ray tubes like the Siemens Straton^TM tube [79] or the Philips iMRC^TM [80], see Fig. 27, are very different from past designs. Instead of electrostatic means, Siemens and Philips use magnetic dipoles for electron beam positioning in-plane and cross-plane for removal of image artifacts. The iMRC^TM features excellent focusing capability of dual magnetic quadrupole lenses, mounted along a straight electron path. The stable segmented 200 mm all metal anode is supported by a liquid metal hydrodynamic straddle bearing in vacuum, which provides direct heat conduction to oil down to low temperatures and has a long service life. With the Straton^TM series, Siemens has taken a different route. The entire tube frame spins on ball bearings in oil, which directly cools the 120-mm anode. While Philips and Siemens rely on heat conduction, GE and Varian utilize 200–238 mm graphite-backed anodes with enhanced heat storage and heat radiation cooling. Modern high-end tubes are anode grounded. Other than in glass tubes, 40 % of heat from scattered electrons is not returning to the anode. Off-focal radiation is practically nonexistent, which is essential for CT.

新的X射线球管，如Siemens Straton管，或Philips iMRC管，见图27，与过去的设计都非常不一样。Siemens和Philips都没有使用静电的方法，而是使用了磁偶极子进行电子束平面内和平面外定位，以去除图像伪影。iMRC的特点是极好的聚焦能力。...

To better compare conceptual different X-ray tubes, the widely used term anode heat storage capacity (''Mega Heat Units'') has been abandoned and replaced by the most practical term CT power in the revised standard IEC 60613 (for special questions CTSPI).

为更好的比较不同的X射线球管，广泛使用的术语热容量已经取消了，在IEC 60613中已经由最常用的CT power所替代。

**CT Source Evolution to Support Major Trends**

Modern systems with great photon capture rates need less electrical energy for photon generation than legacy systems. Why is there a race for sophisticated solutions, then? Gantry speed has doubled in the past decade, and with it the instantaneous power needed. Detector cells and focal spots have shrunken for better spatial resolution. The anode angle has been widened with detector coverage. Thus, the physical power density in the focal spot has risen. It rises with the previously almost-neglected focal track speed, which now exceeds 100 m/s in the iMRC^TM tube. Tube currents have doubled, and lowest tube voltages have gone down. So, even in CT, electronic space charge in front of the cathode has begun to limit the tube current at low kV, where it is needed the most (S/N). Because of their excellent electronic brightness, flat emitters at have been introduced in both, the Philips iMRC and the Siemens Straton tube. The iMRC is able to reliably deliver more than 1,000 mA at 80 kV, and is easily scalable to higher values. GE has introduced flat top coiled filaments, combined with electrostatic means for focusing and deflection.

Gantry speeds of 220 r.p.m. in a Philips iCT^TM system cause centrifugal accelerations of the tube of 32 g. Hydrodynamic anode bearings have proven to be well scalable with increasing loadings. Their service life time is independent of load and rotor speed. The Siemens Straton concept is also well scalable in this sense.

Rising power demand of up to 120 kW per unit and fast kVp switching has pushed the development of h/v generators, too (see Fig. 28). Fast voltage transitions cause high currents for charging of capacitors and cables. In the future, the inverter frequency will rise to further enhance the compactness. Close cooperation of tube and generator development have shown to be essential to handle the complexity of the interfaces and physical effects.

**Other CT Source Concepts**

There are attempts to reduce cone beam artifacts and scatter radiation by use of multiple sources [81•] in an inverse CT geometry. Blocks of stationary anode tubes are rotated on the gantry, and are switched in microseconds.

Aiming to build non-rotating ring tubes, field emission electron emitters have been investigated, too. Due to dose constraints, systems with ring tubes still require rotating detectors with ASGs. The costs of switch gear have shown to be significant for both of these concepts.

Single source rotating anode tubes are expected to be dominating also in the foreseeable future. They are compact, and can be made highly reliable, when manufactured in a competent environment.

## 6. Conclusions

Detection system and X-ray sources are two of the CT components driving the scanner performance and image quality. State of the art technology enables large coverage at sub-millimeter isotropic resolution, low dose and fast acquisition.

The latest developments in detection technology will enable spectral imaging as well as imaging at ultra-low dose with high resolution. Emerging source technologies may support breakthrough CT concepts such as inverse geometry or phase contrast imaging.