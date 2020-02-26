# TI DLP® technology - An industry leader in display and advanced light control solutions

## 1. Overview

### 1.1 Advanced light control 高级光控制

Incredibly fast pattern rates along with ultraviolet and infrared chip offerings fuel diverse and innovative industrial solutions. 极快速的模式变换，可以选择从紫外到红外芯片，可以为多种新型工业解决方案提供支持。

### 1.2 Automotive solutions

Enhance the driving experience with bright, wide field-of-view and augmented reality head-up display (HUD) capability. 用明亮的宽视野和增强现实的平视显示器(HUD)来提升驾驶体验。

### 1.3 Display and projection

Bring HD resolution, high brightness and impressive colors to life with a portfolio of display chips ranging in size and performance to match your product needs. 用多种大小不同、性能不同的显示芯片，带来高清分辨率，高亮度和鲜艳的色彩，满足产品需求。

## 2. DLP products – Getting started

Learn how micromirrors work, select the right chip for your performance imaging needs, and design faster with reference designs, 3rd party solutions and E2E engineering support.

学习微镜是如何工作的，为你的成像性能需求，选择正确的芯片，用参考设计、第三方解决方案和E2E工程支持进行更快的设计。

### 2.1 How does DLP technology work?

At the heart of every DLP chipset is an array of highly reflective aluminum micromirrors known as the digital micromirror device (DMD). The DMD is an electrical input, optical output micro-electrical-mechanical system (MEMS) that allows developers to perform high speed, efficient, and reliable spatial light modulation. Using TI's proven semiconductor manufacturing capabilities, each DMD contains up to 8 million individually controlled micromirrors built on top of an associated CMOS memory cell.

在每个DLP芯片组的核心，是一个高反射性的铝质微镜阵列，称为数字微镜设备(DMD)。DMD是一个电输入，光输出的微机电系统(MEMS)，使开发者能够进行高速高效可靠的空间光调制。使用TI已证明的半导体加工能力，每个DMD包含最多8百万个可单体控制的微镜，这是在一个相关的CMOS存储单元上构建起来的。

During operation, the DMD controller loads each underlying memory cell with a '1' or a '0'. Next, a mirror reset pulse is applied, which causes each micromirror to be electrostatically deflected about a hinge to the associated +/- degree state. The deflection angle of these two valid states is very repeatable due to a physical stop against two spring tips.

在操作时，DMD控制器从下面的存储单元中读取存储内容，即'1'或'0'。然后，送入一个镜面重置脉冲并起作用，使得每个微镜通过静电作用在一个铰链附近偏转，到相关的+/-状态。这两个有效状态的偏转角度是可重复的，因为有两个顶针作用的实体停止作用。

- In a projection system, the + degree state corresponds to an 'on' pixel, and the - degree state corresponds to an 'off' pixel. Grayscale patterns are created by programming the on/off duty cycle of each mirror, and multiple light sources can be multiplexed to create full RGB color images.

- 在投影系统中，+状态对应着on像素，-状态对应着off像素。灰度模式的创建，是对每个镜面的duty cycle进行编程实现的，多光源可以进行多路传输，以创建完全RGB的彩色图像。

- In other applications, the +/- degree states offer two general purpose output ports with a pattern and its inverse.

- 在其他应用中，+/-状态可以提供两种通用目的的输出端口。

The TI DLP portfolio offers chips with +/-12° states as well as +/-17° states. The newest TRP micromirror architecture, with +/- 17° states, is an exciting addition as it enables smaller DMDs that are well suited for more portable end equipment needing good brightness and power efficiency.

TI DLP组合可以提供+/-12°以及+/-17°状态的芯片。最新的TRP微镜架构，使用的是+/-17°状态，是一个激动人心的技术，因为可以制造更小的DMDs，制造的设备有更好的移动性，更好的亮度，更低的功耗。

![](http://www.ti.com/content/dam/ticom/images/products/ic/dlp-products/chips/dlp-get-to-know-chip-shot-dm1987.png)
DMD for digital cinema next to a DMD for pico displays

![](http://www.ti.com/content/dam/ticom/images/products/ic/dlp-products/diagrams/dlp-trp-mirror-diagram-1987.png)
DLP TRP micromirror in -17 degree state

![](http://www.ti.com/content/dam/ticom/images/products/ic/dlp-products/diagrams/dlp-trp-right-chip-diagram-dm1987.png)
DLP TRP micromirror in +17 degree state

### 2.2 Why choose DLP technology?

- Unmatched experience and collaboration with customers has led to the creation of thousands of commercial products:

Provide vivid, crisp image quality to traditional and emerging display applications;

Trusted by innovators in a diverse range of industrial and automotive end products;

DLP technology Academy Award® winning* DLP Cinema® technology is by far the most compelling choice in projection, powering more than eight out of 10 digital movie screens worldwide;

- Powerful, flexible chipsets provide customers high speed and precise control of millions of mirrors with pixel data rates up to 60 Gbps;

- Ample support resources ease the design process such as online references designs, evaluation modules, software, E2E community support and Design Network solutions.

![Delivering precise high speed control of millions of mirrors](http://www.ti.com/content/dam/ticom/images/products/ic/dlp-products/chips/dlp-control-chip-shot-dm1987.png)

### 2.3 What’s included in a DLP chipset?

DLP chipsets allow users to obtain the maximum pattern rates for the associated DMD(s), as well as the ability to synchronize pattern display with external sensors, cameras, motors, or other devices. Furthermore, developers also benefit from TI's extensive knowledge about how to reliably drive the micromirrors across a wide range of operating conditions.

DLP芯片组使得用户对相关的DMDs得到最大化的模式率，以及与外部传感器、相机、发动机或其他设备同步模式显示的能力。而且，TI对于怎样在很多操作环境中驱动微镜有广泛的经验，开发者可以从中受益。

The DLP chipset includes:

- DMD – Digital micromirror device 数字微镜设备；
- DMD Controller – provides the convenient interface for reliable, high speed control of the micromirrors 控制器 - 为可靠、高速控制微镜提供方便的接口；
- DMD Micromirror Drivers – deliver analog clocking and reset signals 微镜驱动器 - 随着模拟时钟和重置信号推送；
- Configuration PROM – configures select DMD Controllers (sold as a programmed IC or offered as downloadable firmware) 配置PROM - 配置选定的DMD控制器（作为已编程IC进行售卖，或作为可下载固件提供）；

### 2.4 Find the right DMD for your design

TI offers a full portfolio of DLP chipsets that span various resolutions, array sizes, micromirror sizes, micromirror tilt angles and wavelengths. Below is a selection guide to learn about the entire DLP portfolio and choose the right chipset for your design, as well as some tables to assist in selecting the DMD best suited for your display or advanced light control solution.

TI的DLP芯片组覆盖了不同的分辨率，阵列大小，微镜大小，微镜倾斜角度和波长。

## 3. Advanced light control

### 3.1 Overview

Highly reliable spatial light modulation for versatile light pattern programming. 高可靠性的空间光调制，可进行多种光模式编程。

- High speed visible - This portfolio of chips offers very fast pattern rates and pixel data rates well suited for solutions requiring programmable, high speed light patterns.

- Ultraviolet (UV) - Targeting 363 - 420 nm, these DMDs steer patterns of UV light to interact with light-sensitive materials.

- Near-infrared (NIR) - Targeting 700-2500nm, these DMDs offer programmable NIR patterns for spectral modulation in optical sensing or precision exposure for 3D and 2D printing, marking and sintering.

### 3.2 Applications

TI DLP® chipsets have enabled powerful, flexible, and programmable light control solutions. The DLP advanced light control product portfolio extends this industry-leading MEMS display technology into ultra-violet and infrared wavelengths, while also enabling faster pattern rates and more advanced pixel control. Through complete reference designs and easy-to-use development tools, TI is accelerating innovative new product development into industrial light control applications.

#### 3.2.1 3D machine vision

DLP systems can produce non-contact, highly accurate 3D data in real-time using programmable structured light patterns. By projecting a series of patterns onto an object and capturing the light distortion with a camera or sensor, one can generate a 3D point cloud.

The point cloud can be used directly for analysis of the object’s surface area, volume, or feature sizes. It can also be exported to a variety of CAD modeling formats.

3D machine vision applications

- Automated optical inspection 自动光学检查
- 3D metrology 3D度量
- Intra-oral scanners (IOS) 口腔内扫描
- 3D scanner accessories 3D扫描仪附件
- Factory automation 工厂自动化
- Medical imaging 医学成像
- Consumer 3D scanners 消费级3D扫描仪
- Biometrics 生物测定学
- Dental scanners 牙齿扫描仪
- Reverse engineering 反向工程

Benefits of 3D machine vision

**Optical MEMS device (up to 4M pixels)**. Enables non-invasive, non-contact 3D scans reliable over time and temperature.

**External triggers**. Synchronize to external cameras and sensors.

**Extended wavelength support (up to 2500 nm)**. Supports wide range of light sources for best scans of varying materials and colored objects.

**Programmable, high speed pattern generation (up to 32 kHz)**. Real-time scan data, optimized for multiple objects and environments using adaptive pattern sets.

**High bit depth**. Higher accuracy and resolution.

**Small form factor**. Portable, lower cost solutions when combined with TI's embedded processors.

#### 3.2.2 3D printing

3D Printing is the additive manufacturing process of building a three-dimensional object with successive layers of material. The process allows manufacturers to speed up design cycles, make quicker prototype adjustments, and print production parts.

3D打印是增材制造的过程，将材料逐层叠加，制造三维目标。这个过程使得制造商加速设计循环，更快的形成原型调整，打印产品部件。

A 3D Computer Aided Design (CAD) model of the object is converted into a series of cross-sectional slices that are sent to the 3D printer. The TI DLP DMD (Digital Micromirror Device) is used to project patterned light for a given layer, which eventually builds the object.  For DLP stereolithography (SLA) printers, liquid resins are hardened by light exposure.  For selective laser sintering (SLS) systems, fine powders are fused together by laser thermal energy.

一个目标的3D CAD模型，转化为一系列横截面切片，送入3D打印机。TI DLP DMD用于对给定的层投影某种光模式，最终制造出这个目标。对于DLP立体光刻(SLA)打印机，液体树脂是用暴露在光下得到硬化的。对于选择性激光烧结(SLS)系统，精细的粉末是通过激光热能融合到一起。

Using DLP technology, build speeds remain constant independent of layer complexity. Projection optics can also be used to control the resolution on the image plane and adjust the layer thickness, leading to smooth finished parts with fine feature sizes. These benefits, combined with its proven reliability, make DLP technology an ideal solution for additive manufacturing solutions.

使用DLP的技术，制造速度与每层的复杂度无关了，维持常速。投影光学也可以用于在图像平面控制分辨率，并调整层厚，可以得到光滑的部件，有着精细的特征尺寸。这些优点，与高可靠性一起，使得DLP技术对于增材制造是一种理想的解决方案。

3D printing applications

- Rapid prototyping 快速原型
- Direct manufacturing 直接制造
- Tooling and casting 工具和铸造
- Dental printers 牙齿打印机
- Custom fit products 定制合适的产品
- 3D printer accessories 3D打印机附件

Benefits of 3D printing

**2D light pattern generation**. Exposes entire print layer on a single shot for fast build times independent of layer complexity.

**High resolution micromirror array**. Enables sub 50μm resolution on the image plane.

**Extended wavelength support (363 - 2500 nm)**. Compatible with a wide range of polymers, resins, sintering powders and other build materials.

**Reliable MEMS technology**. No expensive parts to replace.

#### 3.2.3 Digital lithography 数字平版光刻

Digital lithography is used for PCB manufacturing, flat panel display repair, laser marking, and other light exposure systems. In digital lithography, DLP technology provides high speed and high resolution light patterns to expose photoresist films and other photosensitive materials without using contact masks. This reduces material cost, improves production rates, and allows for rapid pattern changes, especially ideal for use cases where fine feature sizes require double patterning.

数字平版光刻可用于PCB生产，平板显示修复，激光标记，以及其他光暴露系统。在数字平版光刻中，DLP技术可以将高速高分辨率光模式投到光刻胶上，或其他光敏感的材料上，而不需要使用接触性掩膜。这降低了材料损耗，改进了生产率，可以快速更改模式，对于精细的特征大小，需要加倍形成模式的应用，尤其理想。

Digital lithography applications

- Printed circuit boards
- Flant panels
- Industrial printers
- Computer-to-plate printing
- Flexographic printers
- Dynamic laser marking and coding
- Ablation and repair

Benefits of digital lithography

**High speed digital pattern generation (up to 32 kHz)**. Improve manufacturing throughput and eliminate physical masks or print plates.

**Multiple micromirror sizes available (7,10,13 µm)**. Achieve micron-level feature sizes.

**Extended wavelength support (363 - 2500 nm)**. Cure a variety of photosensitive materials or interact with thermally sensitive films.

#### 3.2.4 Spectroscopy 光谱学

All molecules have unique responses to different wavelengths of light. Spectroscopy is an analysis technique that uses these unique responses to identify and characterize materials. In a spectrometer design, the TI DLP Digital Micromirror Device (DMD) can be used as a programmable wavelength selector. Broadband light goes through an optical slit. Then the individual wavelengths of light are dispersed onto the micromirror array using a diffraction grating or prism, allowing subsets of the micromirror array to be mapped to specific wavelengths. Specific wavelengths of light can then be switched to a single-element detector. This powerful design architecture eliminates the need for linear array detectors or motors to generate a spectral scan over a wavelength range, enabling chemical analysis with higher performance and smaller form factors at lower costs.

所有分子对于不同波长的光都有不同的反应。光谱学是一种利用这些不同的反应来识别并给出物质特征的分析技术。在一个分光计的设计中，TI DLP DMD可以用作可编程的波长选择器。宽光谱的光通过一个光学缝隙，使用衍射栅格或棱镜，使单波长的光分散投射到微镜阵列上，使得微镜阵列的一部分可映射到特定的波长；特定波长的光，然后可转变到一个单元素检测器上。这种强力的设计架构，就不需要线性阵列检测器，或发动机来生成一定波长范围内的光谱扫描，使得化学元素分析性能更高，体积更小，消耗更低。

Spectroscopy applications

- Agriculture
- Oil and gas analysis
- Food and drug inspection
- Water and air quality
- Chemical and material identification

Benefits of spectroscopy

**High resolution, programmable optical MEMS array**. Use a large single element detector to capture more light than with linear array detects without sacrificing wavelength resolution.

**Extended wavelength support (up to 2500 nm)**. Enables a single spectral engine customizable for a variety of solids and liquids and multiple light sources.

**High speed switching**. Generate fast spectral scans for real-time material analysis with adjustable scan parameters.

**Reliable MEMS technology**. Stable over temperature and lifetime and enables compact, robust designs.

### 3.3 Optics & electronics

DLP optical module and electronics manufacturers are independent 3rd party companies with expertise in designing and manufacturing systems utilizing TI DLP Products. These off-the-shelf optical modules and electronics are specifically designed for industrial applications and may include a digital micromirror device (DMD), an illumination source, optics, and associated mechanics and electronics. Customers can accelerate product development and reduce time to market by procuring these production-ready solutions directly from this list of manufacturers.

DLP光学模块和电子器件制造商，是独立的第三方公司，专精于设计和制造利用TI DLP产品的系统。这些开箱即用的光学模块和电子器件是设计用于特定工业应用的，可能包含一个DMD，一个光源，光学器件，以及相关的机械和电子器件。通过从这些制造商列表获得这些可用于产品的解决方案，客户可以加速产品开发，降低投放到市场的时间。

Note: most of the 3rd party companies also provide design services and support other DLP chips beyond the ones listed above. For more details, please check out their company profile pages.

注意：多数第三方公司也可以提供设计服务，并支持其他DLP芯片，超出上述列表范围。详情请查询公司简介。

**What is an optical module?**

Buy production ready optical modules online from a worldwide supplier. An optical module is a compact assembly that includes a digital micromirror device (DMD), illumination (LEDs), optics and associated mechanics. Drive electronics are required to supply power and video to the optical module. Drive electronics can either be custom built using TI Designs reference designs or sourced from independent 3rd party companies as an evaluation platform.

Consult our definitions of optical module parameters and design considerations before you get started.

**Which optical module do I need?**

Use the DLP products optical module search tool to identify production ready optical modules from independent third party companies with expertise in designing and manufacturing optical modules utilizing DLP Products. These off-the-shelf modules can help accelerate development and time to market.

**Design service providers**

DLP design service providers are independent 3rd party companies that have proven DLP design experience and provide services such as custom optical design, electronics design, system integration, prototyping, and evaluation modules.

If assistance developing a prototype or technical consultation is needed, design services providers can offer development platforms and guidance to get you started.

**Component providers**

Component providers are independent 3rd party companies that provide necessary adjacent components in a DLP system such as illumination light sources, DMD connectors, and optical components.

If you are designing your own DLP system, this is where you can find useful information about the components needed.

**Extended ecosystem**

Extended ecosystem collaborators are independent 3rd party companies that provide products such as software packages or custom hardware accessories useful for specific DLP applications, such as 3D machine vision, 3D printing, and spectroscopy. 

## 4. Automotive chipsets

High performance automotive display and lighting 高性能汽车显示和光照

DLP products for automotive deliver the highest resolution headlights, with over 1.3 million individually controllable pixels, as well as augmented reality head-up displays with superior solar load performance and image quality. All DLP products for automotive applications are Q1 qualified and on the road today.

DLP在汽车上的产品，可以给出最高分辨率的车头灯，有超过130万单个可控的像素，以及增强现实的平视显示器，有很高的阳光下性能和图像分辨率。所有DLP在汽车上的产品都是有Q1质量的，已经上路使用。

## 5. Display & Projection

Powerful DLP chips use millions of digital mirrors to deliver vivid, crisp images.

### 5.1 Pico chipsets

With mirror arrays ranging from 0.2 to 0.47-inches, DLP Pico products are ideal for small applications and have been designed into hundreds of consumer electronics. From nHD to 4K UHD resolutions, DLP Pico technology brings colorful and crisp images to virtually any surface.

镜片阵列大小从0.2英寸到0.47英寸，DLP Pico产品对于小型应用非常理想，已经设计用于几百种消费电子中。从nHD到4K UHD分辨率，DLP Pico技术可以为几乎所有平面带来色鲜艳丽的图像。

#### 5.1.1 Getting started

Typical system block diagram 

DLP Pico chipsets consist of two primary components: the Digital Micromirror Device (DMD) and the display controller chip. Certain chipsets also include a power management chip.

DLP Pico芯片组是由两个基本组件构成：DMD和显示控制芯片。有些芯片组也会包含能源管理芯片。

The DMD is installed in an optical module with optics and illumination to create the heart of the projection display. The display controller is installed in the electronics board near the optical module to control the DMD and perform necessary data formatting and image processing.

DMD安装在光学模块中，包含光学器件和照明器件，以创建投影显示的核心。显示控制器安装在电子器件板中，在光学模块附近，控制DMD，提供必要的数据格式化和图像处理功能。

![](http://www.ti.com/content/dam/ticom/images/products/ic/dlp-products/diagrams/dlp-chipset-block-diagram-1987.png/dlp-pico-chipset-dm8593.png)

#### 5.1.2 Applications for DLP® Pico™ chipsets

With a broad portfolio ranging from nHD to 4K UHD resolutions, DLP Pico chipsets enable a wide variety of innovative, high performance ultra-mobile and compact display applications. Product developers can quickly go to market by selecting a chipset and optical module that meet the application’s requirements. DLP Pico chipsets can also be used for non-display applications such as 3D Printing and 3D Machine Vision.

DLP Pico芯片组支持很多分辨率，从nHD到4K UHD分辨率，使得非常多有创新性的高性能超移动的紧凑显示应用成为可能。产品开发者可以迅速到市场上选择一个满足应用需求的芯片组和光学模块。DLP Pico芯片组也可以用于非显示的应用，如3D打印和3D机器视觉。

**Pico projectors**

Pico projectors can be used as portable big-screen displays for any device with video output, such as laptops, smartphones, tablets, and gaming consoles. They can offer users an easy and lightweight means to project large and colorful video in a variety of settings.

Pico投影仪可用于任何带视频输出设备的移动大屏显示，比如笔记本，智能手机，平板电脑和游戏控制器，可以给用户提供一种在多种设置下简单轻量级的投射大型色彩斑斓视频的方法。

Benefits:

- High optical efficiency
- Small size, high resolution
- High contrast
- DLP Intellibright™ algorithms
- Mature ecosystem

**Smart home displays**

"Smart home" encompasses a broad category of products and services that bring automation and interconnectivity to a variety of devices in the home, such as lighting, thermostats, appliances, and entertainment devices. While basic smart speakers use audio for feedback, smart displays (or smart speakers with display) supplement the user experience with visual content such as how-to videos, recipes and photos. Mobile Smart TVs are compact devices that can wirelessly stream content and project it onto virtually any surface. They can take the place of a TV in the home or serve as a portable display. Both applications can leverage ultra-short throw optics to project onto surfaces just inches away from the unit.

智慧家庭包括一大类产品和服务，将家庭中很多设备都进行自动化并互联，如，光照，自动调温器，家用电器，和娱乐设备。基本的智能音箱使用音频进行反馈，智能显示（或带有显示的智能音箱）为用户体检增加了视觉内容，如how-to视频，食谱和照片。移动智能TVs是可以无线流媒体的袖珍设备，可以将内容投影到几乎任何平面，可以取代TV在家的地位，或作为移动显示器。这两种应用都可以利用极短光学设置，以投影到只有几英寸远的平面上。

Benefits:

- High optical efficiency
- On demand display
- High resolution
- Displays of any shape on virtually any surface
- Solid state illumination compatible

**Industrial displays**

Digital signage is a category of displays designed for commercial and industrial spaces, including retailers, stadiums, casinos, hotels, restaurants, and airports. Digital signage delivers up-to-date information such as advertising, menus, event status, and maps in locations where people gather. Some use cases may require larger display sizes and brightness – learn more about digital signage with DLP standard chipsets. Free-form projection capability from a compact device also makes DLP Pico products a great fit for other industrial products such as humanoids (service robots), and commercial gaming machines.

数字看板是设计用于商业和工业空间的一类显示，包括零售，体育场，赌场，旅馆，饭店和机场。数字看板投送的是最新信息，如广告，菜单，事件状态和人们聚集在地图上的位置。一些使用案例可能需要更大的显示大小和亮度，这就需要用到使用DLP标准芯片组的数字看板。从袖珍设备中自由形态投影的能力，也使DLP Pico产品非常适合其他工业产品，如人形机器人（服务机器人），和商用游戏机。

Benefits:

- Display on any surface
- Free-form displays
- On-demand display
- Swappable and replaceable displays
- Solid state illumination compatible

**Augmented reality glasses**

Wearable displays are devices that are worn as a helmet, headset, or glasses by the user and create an image in the user’s field of view. The display can either be see-through (augmented reality) or opaque (immersive or virtual reality). Products in this category include head-mounted displays (HMDs) and near-eye displays.

可穿戴显示器是可以佩戴为头盔、头戴式耳机或眼镜式的设备，在用户视野中创建一幅图像。这种显示可以是穿透的（增强现实）或不透明的（沉浸式的或虚拟现实）。这个类别的产品包括头戴现实(HMDs)和离眼镜很近的显示。

The DLP Pico chip is a reflective microdisplay technology used in the optical module in a wearable display. It is typically illuminated by RGB LEDs, and intelligently reflects light through pupil forming optics into a final optical element, such as a waveguide or curved combiner, which relays the image into the eye.

DLP Pico芯片是一种反射式的微显示技术，用在可穿戴式显示设备中的光学模块，通常是由RGB LEDs照明的，智能的反射光穿过瞳孔状的光学元件到最终的光学元素，如波导或弯曲的连接器，将图像接力送到眼镜中，

Benefits:

- Low power consumption
- High contrast
- High optical efficiency
- Small size, high resolution
- High speed: fast refresh rates, low latency

**Smartphones & tablets**

Integrating pico displays into smartphones and tablets enables larger screen and power-efficient projection on virtually any surface. On-demand second screens can be used to share photos, show a presentation, watch a movie, or even play a videogame.

将pico显示整合进智能手机和平板中，可以使更大的显示和更高效的投射到任意平面成为可能。按需的第二个屏幕，可以用于共享图片，展示PPT，观看电影，或甚至玩一个视频游戏。

Benefits:

- Small size, high resolution
- DLP Intellibright™ algorithms
- Mature ecosystem
- High optical efficiency
- High contrast

**Other applications**

There are hundreds of other applications that could benefit from the high brightness and high contrast free-form, on-demand displays that DLP Pico technology offers. For example, there are numerous ways projection displays can be incorporated into appliances. These features enable developers to bring significant product differentiation and can quickly go to market by selecting a chipset and off the shelf optical module that meets the application’s requirements.

DLP Pico技术提供的高亮度、高对比度、自由形态、按需显示，可以使上百种其他应用受益。比如，有无数种方式投影显示可以整合到家用电器中。这些特征使开发者可以带来明显的产品差异，可以通过选择一个芯片组和开箱即用的光学模块，以满足应用需求，迅速的走向市场。

### 5.2 Standard Chipsets

Standard chipset are optimal for large displays with high brightness and resolution. 标准芯片组最适合于高亮度高分辨率的大型显示。

#### 5.2.1 Applications

TI DLP chips in the standard portfolio enable a myriad of high brightness, high resolution projection solutions that are capable of vibrant, colorful displays. Product developers can select a chipset and optical module that best meets the specific application’s display requirements.

TI DLP的标准芯片可以有无数的高亮度高分辨率的投影解决方案，可以进行鲜艳的色彩艳丽的显示。产品开发者可以选择一个芯片组和光学模块，与特定应用的显示需求进行最佳匹配。

**Laser TV**

Laser TV products are projection displays that utilize a laser light source, typically in conjunction with ultra-short throw optics, to create large, bright, beautiful pictures and video. This portable solution is a desirable alternative to large flat panel TVs since they can be lighter, have a smaller footprint and provide greater mobility from room to room or house to house. DLP standard chipsets are compatible with laser and laser phosphor illumination and enhance Laser TV applications by delivering high brightness capabilities and sharp 4K UHD resolution. If you are interested in a small, low power device, see mobile Smart TV products.

Laser TV产品是利用激光光源的投影显示，一般与超短距光学元件配合使用，以生成大型、明亮、美丽的图像和视频。这种可移动解决方案是大型平板TV的完美替代，因为可以更轻，所占面积更小，可以在房间到房间，或屋子到屋子间提供更好的移动性。DLP标准芯片组与激光和激光磷光体照明是兼容的，通过其高亮度能力和清晰的4K UHD分辨率可以增强激光TV的应用。如果对小型、低能耗设备感兴趣，可以看看移动智能TV产品。

Laser TV applications

- Screenless sound
- Mounted System
- Ultra-short throw projection
- Outdoor media sharing

Benefits of DLP technology for Laser TV

**High thermal capability**. DLP standard chipsets enable high brightness, large displays for excellent laser TV viewing experiences in virtually any room. DLP标准芯片组可以达到很高的亮度，进行大型显示，在任何房间内得到优异激光TV观影体验。

**Single chip projection engine**. Optical engines can be designed with a single DLP DMD chip, offering system cost savings and alignment-free setup versus other projection technologies. 光学引擎可以用单个DLP DMD芯片进行设计，与其他投影技术相比，系统价格可以得到降低，而且不用进行对齐设置。

**Light source agnostic**. DLP chips are compatible with solid state illumination - a key enabler for impressive laser TV image quality and small end product form factors. DLP芯片与固态照明是兼容的，这是高质量激光TV图像质量和小型化终端产品的一个必要条件。

**High resolution**. Delivers up to 4K UHD resolution to the screen from compact form factors, providing a compelling alternative to traditional TV technologies. 可以用袖珍设备投射出4K的UHD分辨率投影，这是传统TV技术的一个很有竞争力的替代。

**High contrast**. Enables vivid colors and dark blacks for enhanced laser TV viewing experiences. 鲜亮的颜色和灰度对比，得到强化的激光TV观赏体验。

**Digital signage**

Digital signage is a display solution to convey information and content to a viewer. Benefits include flexibility to adapt the content on demand as well as ability to project on multiple surface types. Bright and eye-catching content might include menus, navigation guides, event updates or advertising in hotels, restaurants, airports and other large public areas. Enabling various display sizes and resolutions of up to 4K UHD, DLP standard chipsets enable large, bright and free-form displays on virtually any surface. Ultra-mobile digital signage is a related application that DLP Pico™ chipsets enable.

数字看板是一种给观察者传递信息和内容的显示解决方案，其优势包括可以灵活的按需调整内容，以及可以投射到多种平面类型中。明亮抢眼的内容可能包括菜单，导航指引，事件更新或广告，可以在宾馆、饭店、机场和其他大型公共区域中。显示大小和分辨率可以多样，最高到4K UHD，DLP标准芯片组可以在几乎任何平面上进行大型、明亮和自由形态的显示。具有超级移动性的数字看板是相关的应用，DLP Pico芯片组才能产生这种效果。

Digital signage applications

- 3D entertainment
- On-demand retail signage
- Free-form imaging
- Bright and large advertising
- Interactive display
- Engaging projection

Benefits of DLP technology for digital signage

**Display on any surface**. Digital projection using DLP chips can transform virtually any surface into a large format display area to convey desired signage content or information. 使用DLP芯片的数字投影来将几乎任何平面变成一个大型显示区域，以传达期望的看板内容或信息。

**Light source agnostic**. DLP technology is compatible with virtually any light source, including lasers and LEDs, allowing designers maximum versatility across signage use cases. DLP技术与几乎任意光圈都兼容，包括激光和LEDs，使得设计者可以最大化看板使用的情况。

**High thermal capability**. DLP standard chipsets have thermal properties capable of large, high brightness displays for vivid and attention-grabbing digital signage. DLP标准芯片组的热性质，可以进行大型、高亮度显示，在数字看板上展现出靓丽吸睛的数字看板。

**Free-form displays**. DLP technology can project displays on custom shapes and curves to create unique and artistic digital signage. DLP技术可以在定制的形状投影显示，以生成独特的、有艺术性的数字看板。

**Ecosystem partners**. Optical module manufacturers and design houses offer production ready optical engines and design services to ease the design process and quickly integrate DLP technology in digital signage applications. 光学模块制造商和设计公司可以提供可用的光学引擎产品和设计服务，使得设计过程非常简单，可以迅速的将DLP技术整合进数字看板应用。

**Business & education**

Business and education displays continue to play an important part of everyday life. DLP technology has been a proven choice for projector manufacturers for over 20 years. The latest chips in the DLP standard portfolio are offering even more brightness and higher resolution. Developers can incorporate high brightness, sharp projection systems ideal for vivid classroom multimedia experiences and high resolution business data sharing.

商务和教育显示在日常生活中一直是重要的部分。DLP技术已经在投影仪技术中使用了20年，非常有效。所有DLP标准芯片中最新的，可以提供更高的分辨率和更高的亮度。开发者可以使用更高的亮度，更锐利的投影系统，进行理想中的教室多媒体投影，以及高分辨率的商务数据共享。

商务和教育显示

Benefits of DLP technology for business and education

**High speed imager**. Delivers up to 4K UHD resolution to the screen making text and data sharp even to those in the back of the room. 提供高达4K的UHD分辨率，即使是对于房间最后排的人，文本和数据也非常清晰锐利。

**High native contrast**. Enables vivid colors and dark blacks, offering outstanding readability of fine text for applications such as PowerPoint and Excel files. 可以有鲜明的色彩和黑白对比，对于PPT和Excel文件的文本，给出极好的可读性。

**High thermal capability**. DLP standard chipsets enable large, high brightness displays so users can have an optimal viewing experience in virtually any classroom or meeting environment. DLP标准芯片组给出大型高对比度的显示，在几乎任何教室或会议环境中，用户都可以拥有最佳的观影体验。

**Single chip projection engine**. Optical engines can be designed with a single high resolution DLP chip, offering system cost savings and alignment-free setup, a key consideration for enterprise and educational procurement teams. 光学引擎可以用单片高分辨率DLP芯片设计，系统价钱可以降低，也可以不用进行对齐，这对于企业和教育采购小组来说是一个很关键的考虑点。

**Light source agnostic**. DLP chips are compatible with LEDs and lasers, which can result in projectors with lower maintenance costs compared to lamp-based systems. DLP芯片与LEDs和激光都是兼容的，与基于灯泡的系统相比，可以得到需要很少维护的投影仪。