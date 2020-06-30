# The UltraSound ToolBox

## 0. Abstract

We present the UltraSound ToolBox (USTB), a processing framework for ultrasound signals. USTB aims to facilitate the comparison of imaging techniques and the dissemination of research results. It fills the void of tools for algorithm sharing and verification, and enables a solid assessment of the correctness and relevance of new approaches. It also aims to boost research productivity by cutting down implementation time and code maintenance.

我们给出超声工具箱(USTB)，一种超声信号的处理框架。USTB的目标是促进成像技术的比较，和研究结果的传播。USTB填补了算法共享和验证的工具空白，可以进行可靠的正确性的评估和相关的新方法的评估，还可以通过减少实现的时间和代码的维护，来提升研究的生产率。

USTB is a MATLAB toolbox for processing 2D and 3D ultrasound data, supporting both MATLAB and C++ implementations. Channel data from any origin, simulated and experimental, and using any kind of sequence, e.g. synthetic transmit aperture imaging (STAI) or coherent plane-wave compounding (CPWC), can be processed with USTB.

USTB是处理2D和3D超声数据的Matlab工具箱，支持Matlab和C++实现。任何来源的通道数据，仿真的或试验的，使用任意序列，如，合成传输孔径成像(STAI)，或一致平面波复合(CPWC)，都可以用USTB进行处理。

Here we describe some of the elements of USTB such as: the ultrasound file format, the concept of the general beamformer, and the signal processing pipeline. We also show a minimal code example, and demonstrate that USTB can be used with the most used transmit sequences: STAI, CPWC, diverging wave imaging (DWI), focused imaging (FI), and retrospective transmit beamforming (RTB).

这里我们描述USTB的一些元素，如：超声文件格式，通用波束形成的概念，和信号处理流程。我们还展示了一个最小代码示例，并展示了USTB可以用于最常见的传输序列：STAI，CPWC，发散波成像(DWI)，聚焦成像(FI)和追溯传输波束形成(RTB)。

Keywords—Beamforming, signal processing, comparison, toolbox

## 1. Introduction

Urged by the motto “publish or perish”, the Academia faces a challenge of biblical proportions: the universal research-paper flood.

受到格言“发表或消亡”的敦促，学术界正面临重大挑战：普遍的研究文章大爆发。

There are two factors that will unleash Armageddon in the ultrasound community. First is our inability to keep track of all the publications relevant to our work. This is due mainly to time constrains, but also because of the increasing specialization required to implement modern methods.

在超声界，有两个因素会导致大决战。第一是我们不能追踪与我们工作相关的所有发表文章。这主要是因为时间约束，但也因为要实现现代的方法需要越来越多的专业技能。

Second is the overloading of the peer-reviewing system. With an ever increasing number of submissions, few reviewers have the luxury of implementing themselves the method described in the manuscript. This makes it very difficult - if not impossible - to assess the correctness and relevance of the proposed method. It reduces the assessment of the method to the observation of some results, often 2D images, that the authors compare to some reference algorithm, often delay-and-sum (DAS).

第二是同行评议系统的过载。随着提交论文的越来越多，很少有评议着会亲自实现文稿中提出的方法。这使其非常困难（如果不是不可能的话）去评估提出的方法的正确性和相关性。这使得提出的方法的评估，只能通过观察结果来进行，通常是2D图像，也就是作者与一些相关算法的比较，通常就是delay-and-sum(DAS)算法。

In this way the question of whether the method is correct and relevant becomes a matter of opinion; a learned opinion perhaps, but an opinion nonetheless.

这样，这个方法是否正确是否相关的问题，变成了一种想法的问题；可能是一种博学的想法，但仍然是一种想法。

We ask the reviewers to assess the performance of a nearly black-box by mere observation of a few of its outputs, a request that should set all our scientific alarms off. Not surprisingly, the attention scholars devote to new research is dropping [1] while the proportion of scientific fraud has increased tenfold since 1975 [2]. In 2012, a study published in Nature [3] found that 47 out of 53 medical research papers on cancer research were irreproducible.

我们请评议者来评估一个接近黑箱算法的性能，只能通过一些其输出进行观察。并不意外的是，关注新研究的学者越来越少，而科研欺诈正自从1975年以来增加了10倍。在2012年，Nature上发表的一片文章发现，53篇医学研究文章中的47篇是不可复制的。

As the scientific community becomes aware of this trend reaction starts to take shape. Initiatives such as open research [4], [5], and reproducible research [6], [7] try to fight this. Using modern tools they aim to reconnect the peer-review system to the spirit that got academic journals started in the XVII century.

在科研团体意识到这个趋势的同时，应对措施也正在成型。比如开放研究、可复现研究的倡议，正试图解决这个问题。使用现代工具使得同行评议系统重新得到了学术期刊在XVII世纪开始的精神。

In the ultrasound community, a recent attempt to address this issue was made with the PICMUS challenge [8] (Tours, IUS 2016). Some researchers have started to take interest in the comparison of beamforming methods [9], and there seems to be a clear consensus on the need for tools and data to support the verification of all kind of signal processing algorithms.

在超声团体，最近要解决这个问题的一个尝试是PICMUS挑战。一些研究者开始对波束形成方法产生兴趣，似乎都一致认为，对所有信号处理算法的验证，需要工具和数据支持。

## 2. Thesis

We believe this problem can be solved by developing a common set of tools, including: 我们相信这个问题可以通过开发一套通用工具来解决，包括：

1) an ultrasound file format, and 超声文件格式，和

2) a framework for ultrasound signal processing. 超声信号处理的框架。

A common ultrasound file format makes it possible to share processed and unprocessed datasets. Making those datasets publicly available facilitates the definition of a set of standard test cases for verification purposes. Inserting the same input dataset to different algorithms makes for a perfect scenario for comparison. Using a known data format allows for further inspection of the output of a method.

通用超声文件格式使其可以共享处理过的和未处理过的数据集。使这些数据集公开可用，可以促进确定一个标准测试案例的集合，以进行验证。将同样的输入数据集送入不同的算法，可以进行很好的比较。使用已知的数据格式，可以进一步调查一个方法的输出。

However, the file format alone does not allow a reviewer to check whether a method is correct or relevant for a given application. To address that problem a framework has to be defined so that also the algorithm can be shared. This way reviewers, and other researchers, can test the algorithm on their own terms, inserting new datasets, designing new tests.

但是，文件格式本身，不能使一个评议者检查一个方法，对于一个应用来说是否是正确的或相关的。为解决这个问题，必须定义一套框架，这样算法也可以共享。通过这种方法，评议者和其他研究者，可以以其自己的方式测试算法，插入新的数据集，定义新的测试集。

Four institutions (Norwegian University of Science and Technology, University of Oslo, University of Lyon, and Johns Hopkins University) have come together to start the development of such a framework: the UltraSound ToolBox (USTB).

四个研究机构一起，开始了这样一个框架的开发：超声工具箱。

The USTB is a free MATLAB toolbox for processing ultrasonic signals, and comes with its own ultrasound file format (UFF). USTB aims to cover all processing techniques from tissue and flow visualization to other image processing techniques. More information about the USTB initiative can be found at the website https://www.ustb.no. Access to the repository https://bitbucket.org/ustb/ustb is granted for anyone interested in development or testing.

USTB是一个免费的Matlab工具箱，进行超声信号处理，有其自己的超声文件格式(UFF)。USTB的目标是覆盖所有处理技术，从组织和流体可视化，到其其他图像处理技术。

## 3. Methods

In what follows we review part of the USTB structure and we disclose some of the elements in USTB. 下面我们看一下USTB的结构和一些元素。

### 3.1. The Ultrasound File Format

UFF data structure originates from the data class organization in USTB. A UFF class family (shown in Fig. 1) is defined, composed of a uff class and seven children classes. To minimize memory use uff is defined as a handle class.

UFF数据结构源自于USTB中的数据集类组织。定义了一个UFF类族（如图1所示），由一个uff类和7个子类组成。为最小化内存使用，uff定义为一个handle类。

All UFF classes can be dumped into HDF5 files (Hierarchical Data Format v5). HDF5 is an open file format to manage extremely large and complex data collections [10]. Originally developed by the National Center for Supercomputing Applications it is now maintained by the HDF Group. Several software platforms support HDF5 such as Java, MATLAB, Python, Octave, Mathematica, etc.

所有的UFF类都可以转换为HDF5文件。HDF5是一种公开文件格式，可以管理极大极复杂的数据集合。最开始是超级计算应用国家中心提出的，目前由HDF小组维护。几种软件平台都支持HDF5，如Java, Matlab, Python, Octave, Mathematica, 等。

For clarity, we refer to the files dumped by USTB as UFF files (instead of HDF5) and we use the extension .uff. 为表达清楚，我们称USTB转换的文件格式为UFF文件（而没用HDF5），使用的后缀为.uff。

### 3.2. Children and grandchildren classes

We follow the policy that general, unconstrained classes must be defined as children classes, while specific, constrained objects are defined as grandchildren classes. For instance we can define an arbitrary probe with uff.probe by specifying the position of each element; but with uff.linear_array we can directly define a linear array just by specifying the pitch and number of elements.

我们遵循的原则是，通用的不受约束的类需要定义为子类，而特定的有约束的目标要定义为孙类。比如，我们可以用uff.probe定义一个任意的探头，只要指定每个元素的位置；但使用uff.linear_array我们只要指定了pitch和元素数量就能直接定义一个线性阵列。

Processing is greatly simplified by making all processes take only children classes as input, rather than all possible grandchildren classes. The only requirement is then that every grandchild class must know how to define itself as its parent class.

通过使所有处理都只以子类为输入，而不是以孙类为输入，处理过程得到了极大的简化。唯一的需求是，每个孙类都必须知道怎样将其本身定义为其父类。

### 3.3. The general beamformer 一般的射束形成

Both UFF and USTB revolve around the concept of the general beamformer. The wavefronts in most ultrasound sequences can be fully defined using a single point source P: in focused imaging (FI) and retrospective transmit beamforming (RTB) P is on the transmit focal point in front of the probe, in diverging wave imaging (DWI) P is at the wave origin behind the probe, in synthetic transmit aperture imaging (STAI) P lies on the active element, in coherent plane-wave compounding (CPWC) P is at an infinite distance but in a given direction. Using point sources to define all those waves it is possible to beamform all sequences with a single algorithm.

UFF和USTB都是围绕着一般射束形成的概念的。在多数超声序列中的波前可以完全使用单个点源P来定义：在聚焦成像(FI)和回顾性传输波束形成(RTB)中，P是在探头前面的传输焦点上，在发散波成像(DWI)中P是在探头后面的波原点上，在合成传输孔径成像(STAI)中，P是在积极元素上，在连贯平面波合成(CPWC)中，P是在一个给定方向上的无限远距离上。使用点源来定义所有这些波，可以使用单个算法来射束形成所有序列。

This is important to reduce the number of beamforming codes in the framework, facilitate intercomparison, and reduce code maintenance.

这对于减少框架中的射束形成代码数量，促进相互之间的对比，减少代码维护，是很重要的。

### 3.4. Time zero convention

It is necessary, however, to set an unequivocal convention for what is defined as time zero. This is often defined as the moment the first element in the array is fired. While this is perfectly correct, in USTB we favor a convention that does not depend on the probe geometry. USTB assumes that time zero corresponds to the moment the transmitted wave passes through the origin of coordinates (0, 0, 0).

对时间0的定义，设置一个明确的协议，这是非常必要的。这通常定义为，阵列中第一个元素出现的时刻。虽然这是非常正确的，但在USTB中，我们倾向于将其定义为与探头集合形状是无关的。USTB假设时间零对应着，传输的波穿过坐标为(0,0,0)的原点的时刻。

### 3.5. Data dimensions

The data in a uff.channel_data structure have four dimensions: [time, channel, event, frame]. 数据的结构为uff.channel_data，包含4个维度：[time, channel, event, frame]。

The first two dimensions run along the time sample and the channel number. The third is the event dimension whose length will often be equal to the number of waves in the sequence. The fourth dimension is the frame number.

前两个维度为时间和通道数量。第三个event维度，其长度会等于序列中的波的数量。第四个维度是帧数。

There are two situations where the number of waves in the sequence will differ from the number of events. One is in case of multi-line transmissions (MLT), i.e. when more than one wave is transmitted in a single transmit/receive event. In such case there will be more waves in the sequence than events in uff.channel_data. Those are addressed with an event index in the uff.wave structure that indicates which event holds the data of a certain wave.

有两种情况，序列中波的数量与event数量不一样。一个是多线传输的情况(MLT)，即在单个传输/接收事件中，传输了超过一个波。在这种情况下，uff.channel_data中的序列中波的数量会比event数量多。这通过在uff.wave结构中的event索引来解决，表明了哪个event有多少个波的数据。

The other case is in packet acquisitions, i.e. when the same wave is transmitted in consecutive events. In this case there are more events than waves in the sequence. These events are addressed by specifying the event indexes in uff.wave as an array of integers.

另外的情况是，包的获取，即，当相同的波是在连续的event中传输的。在这种情况下，序列中的events要多余波的数量。这些events是通过在uff.wave中指定event索引，作为整数阵列来处理的。

The data in a uff.beamformed_data structure have also four dimensions: [pixel, channel, event, frame]. 一个uff.beamformed_data结构中的数据也有四个维度：[pixel, channel, event, frame]。

The first dimension of the data runs along the pixels in the spatial map. The third dimension separates the contribution of each wave to image formation, a contribution which is also referred to as “low resolution image” in the context of STAI or CPWC. The second dimension separates the contribution of each channel to image formation, equivalently to the contribution of each transmitted wave. The fourth dimension is the frame number.

数据的第一个维度是空间图中的像素。第三个维度将每个波对图像形成的贡献分隔开来，这种贡献在STAI或CPWC的情况下也称为低分辨率图像。第二个维度将每个通道对图像形成的贡献分隔开来，与每个传输的波的贡献是等价的。第四个维度是帧数。

While the number of dimensions remains constant along the processing pipeline (always four), the number of elements in any dimensions could vary throughout the processing chain. For instance: after compounding the wave and channel dimensions can become singleton, the number of frames may change after clutter filtering.

在处理流程中，维度数量保持是常数（永远是4），但在任意维度中，元素的数量在处理链条中可能会变化。比如，在复合后，波和通道维度会变成单个，在群聚滤波后，帧数可能会改变。

### 3.6. The signal processing pipeline

An interesting approach to the development of a data processing framework is to set a processing pipeline that allows inserting an arbitrary number of processors (filters or gadgets) in a more or less arbitrary order. Such strategy has been successfully implemented in other disciplines of medical imaging [11]. Maximum flexibility is achieved when the processors are “atomic”, i.e. as small as possible, so that complex processes can be built by combination of them. This strategy boost productivity by increasing code reutilization and reducing maintenance.

开发数据处理框架的一个有趣方法是，设置一个处理管道，可以插入任意数量的处理器（滤波器或工具），其顺序可以是比较任意的。这种策略已经在其他医学处理准则中成功的实现了。当处理器是原子型的时候，即，处理器要越小越好，可以获得最大的灵活性，这样可以通过其组合建立复杂的过程。这种策略通过增加代码重用和降低维护，来提升生产力。

In USTB we define a set of processors that we divide into three types: preprocessors, midprocessors, and postprocessors. A preprocessors takes a uff.channel_data class and delivers another uff.channel_data class. A midprocessor takes a uff.channel_data and delivers a uff.beamformed_data. A postprocessor both takes and delivers uff.beamformed_data classes. A pipeline is built by connecting the outputs and inputs of a set of processors such as it is shown in Fig. 2. Additional data can be fed into the pipeline as parameters of every processor class. Processors can be implemented both in MATLAB or C++.

在USTB中，我们定义一系列处理器，可以分为三类：预处理器，中间处理器，和后处理器。预处理器以uff.channel_data为输入，生成另一个uff.channel_data类。中间处理器以uff.channel_data输入，输出uff.beamformed_data。后处理器的输入和输出都是uff.beamformed_data。通过将一系列处理器的输入和输出连接起来，如图2所示，就可以构建一个流程了。这个流程还需要一些额外的数据输入，作为每个处理器的参数。处理器可以用Matlab或C++实现。

Note that the end result of the pipeline does not have to be a B-mode image. Other physical properties, different from scattering intensity, can be estimated through the pipeline and stored in a uff.beamformed_data structure: Doppler shift, blood flow, sound speed, elasticity, attenuation, etc.

注意流程的最终结果不一定是B模式图像。其他的物理性质，与散射亮度不同，可以通过流程进行估计，存储在uff.beamformed_data数据结构中：多普勒频移，血流，声速，弹性，衰减，等。

Several processors are available in USTB implementing adaptive beamforming techniques, such as the coherence factor, phase coherence factor, generalized coherence factor, delay-multiply-and-sum, and short-lag spatial coherence. USTB also features a large set of example scripts including data import from the Verasonics and Alpinion platforms, interaction with the Field II[12] program, as well as examples of acoustic radiation force imaging, multi-line acquisition, etc. Several UFF datasets are available on the USTB website.

在USTB中，实现自适应波束形成技术时，有几个处理器是可用的，如一致性因素，相位一致性因素，一般化的一致性因素，delay-multiply-and-sum，和短延迟空间一致性。USTB还有一个脚本范例集，包括从Verasonics和Alpinion平台的数据倒入，与Field II程序的交互，以及声音辐射力成像的例子，多线获取的例子，等。在USTB网站上，有几个可用的UFF数据集。

## 4. Results and Conclusions

To demonstrate the flexibility of USTB, five datasets (FI, STAI, CPWC, DWI, and RTB) have been simulated with USTB’s built-in simulator and reconstructed with the general beamformer approach. It is well known that CPWC becomes equivalent to optimal FI (or STAI) under certain circumstances. Here we use USTB to show that the same result can be extrapolated to DWI and RTB, if the transmit apodization is transformed according to [13]. The equivalence is demonstrated in terms of the full width half maximum (FWHM) and side lobe level (SLL).

为展现USTB的灵活性，使用USTB的内建模拟器仿真了5个数据集，并使用通用波束形成方法进行了重建。大家都知道，CPWC在特定情况下，与最优FI（或STAI）是等价的。这里我们使用USTB来展示，如果传输apodization是根据[13]变换的，相同结果可以外插到DWI和RTB中。这种等价性用full width half maximum (FWHM) 和 side lobe level (SLL)进行了展示。

Fig. 3 shows the point spread function (PSF) of the tested imaging sequences showing very similar images. Table I displays the quality indexes FWHM and SLL.

图3展示了测试成像序列的点扩展函数，展示了很类似的图像。表1展示了FWHM和SLL的质量。

We observe a bimodal distribution: nearly identical values for STAI and FI (relative standard deviation of σFWHM=0.52% and σSLL=0.24%), and for CPWC, DWI, and RTB (σFWHM=0.10% and σSLL=0.15%). This is due to the two-step combination of waves that occurs in the latter methods. Between the two distributions we observe a relative error of eFWHM= 4.66% and eSLL=5.75% that can be considered a negligible drop in image quality.

我们观察了一个双峰分布：STAI和FI的值基本相同，CPWC, DWI和RTB的值基本相同。这是因为在后面的方法中，有波的两步结合。在这两个分布中，我们观察到相对误差eFWHM= 4.66%和eSLL=5.75%，这种图像质量的降低是可以忽略的。

The code used to perform this comparison is available at http://www.ustb.no/code/IUS2017_abstract.m. No data is needed to run the code, but the USTB must be available in MATLAB’s path.

Listing 1 show a minimal code example for USTB. The code downloads a CPWC dataset from the PICMUS challenge [8], beamforms it, and displays it. It gives a quick overview of the kind of code handling that can be achieved with USTB. The resulting image is shown in Fig. 4.

列表1是USTB的一个最短示例代码。这段代码从PICMUS挑战上下载了一个CPWC数据集，进行波束形成，并进行展示。这快速展示了用USTB编成的代码可以做的事。得到的图像如图4所示。

```
% download UFF file
tools.download(’PICMUS_carotid_long.uff’, ’http:// ustb.no/datasets/’, [ustb_path(),’/data/’]);
% read channel data from UFF
channel_data=uff.read_object([local_path filename],’ /channel_data’);
% define scan
scan=uff.linear_scan();
scan.x_axis=linspace(-19e-3,19e-3,256).’;
scan.z_axis=linspace(5e-3,30e-3,256).’;
% initialize pipeline
bmf=beamformer();
bmf.channel_data=channel_data;
bmf.scan=scan;
% set up tx/rx apodization
bmf.receive_apodization.window=uff.window.tukey50;
bmf.receive_apodization.f_number=1.2;
bmf.receive_apodization.origo=uff.point(’xyz’,[0, 0, -Inf]);
bmf.transmit_apodization=bmf.receive_apodization;
% launch pipeline: DAS + coherent compounding
b_data=bmf.go({process.das_mex_process.coherent_compounding});
% display image
b_data.plot();
```

The UltraSound Toolbox (USTB) aims to facilitate the comparison of imaging techniques and the dissemination of research results. But it may also become a formidable research booster. Imagine how much faster would it be to test new ideas if we had a plug-and-play implementation of all the methods in the state-of-the-art. Consider how much time could be saved in fruitless recoding and invested in taking the field forward.

USTB的目标是促进成像技术的比较，和研究结果的传播。但这也可能成为一种厉害的研究推进器。想象一下，如果我们有了所有目前最好方法的即插即用的实现，我们测试新想法的速度会快很多。重新写代码会浪费很多时间，采用USTB会带领这个领域向前发展的更好。