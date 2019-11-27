# How Good Is My Test Data? Introducing Safety Analysis for Computer Vision

Oliver Zendel et al. Austrian Institute of Technology

## 0 Abstract

Good test data is crucial for driving new developments in computer vision (CV), but two questions remain unanswered: which situations should be covered by the test data, and how much testing is enough to reach a conclusion? In this paper we propose a new answer to these questions using a standard procedure devised by the safety community to validate complex systems: the hazard and operability analysis (HAZOP). It is designed to systematically identify possible causes of system failure or performance loss. We introduce a generic CV model that creates the basis for the hazard analysis and—for the first time—apply an extensive HAZOP to the CV domain. The result is a publicly available checklist with more than 900 identified individual hazards. This checklist can be utilized to evaluate existing test datasets by quantifying the covered hazards. We evaluate our approach by first analyzing and annotating the popular stereo vision test datasets Middlebury and KITTI. Second, we demonstrate a clearly negative influence of the hazards in the checklist on the performance of six popular stereo matching algorithms. The presented approach is a useful tool to evaluate and improve test datasets and creates a common basis for future dataset designs.

要推动计算机视觉的新发展，好的测试数据非常关键，但还有两个没有解决的问题：测试数据应当涵盖哪些情况，要测试多少，才能得到一个结论？本文中，我们对这些问题提出一个解答，使用的是安全专家验证复杂系统提出的标准过程：危害与可操作性分析(HAZOP)。其目的是系统的确认系统失败或性能降低的可能原因。我们提出了一个通用CV模型，这个模型是危险分析的基础；我们第一次对CV领域进行广泛的HAZOP分析。得到的结果是一个检查表，包含900多个危害点。这个检查表可以用于评估已有的测试数据集，对这些危害点进行量化。我们对流行的立体视觉测试数据集Middlebury和KITTI进行分析和标注，以评估我们的方法。第二，我们对6个流行的立体匹配算法，展示了列表中的危害点的负面影响。我们提出的方法，是评估和改进测试数据集的有用工具，是将来的数据集的设计的基础。

**Keywords** Test data · Testing · Validation · Safety analysis · Hazard analysis · Stereo vision

## 1 Introduction

Many safety-critical systems depend on CV technologies to navigate or manipulate their environment and require a thorough safety assessment due to the evident risk to human lives (Matthias et al. 2010). The most common software safety assessment method is testing on pre-collected datasets. People working in the field of CV often notice that algorithms scoring high in public benchmarks perform rather poor in real world scenarios. It is easy to see why this happens:

很多系统基于CV技术探索其环境，这些系统对安全要求很高，由于很明显会对人的生命造成威胁，所以需要彻底的安全评估。安全评估方法的最常用软件，是在预先收集的数据集上进行测试。CV领域工作的人都会注意到，在公开基准测试数据集上评分很高的算法，在真实世界情况中的表现比较差。很容易可以看出来，这种情况发生的原因：

- The limited information present in these finite samples can only be an approximation of the real world. Thus we cannot expect that an algorithm which performs well under these limited conditions will necessarily perform well for the open real-world problem. 有限样本中的有限信息，只能是真实世界的近似。所以一个算法在有限情况下表现的好，我们不能期待其在开放的真实世界问题中也表现的好。

- Testing in CV is usually one-sided: while every new algorithm is evaluated based on benchmark datasets, the datasets themselves rarely have to undergo independent evaluation. This is a serious omission as the quality of the tested application is directly linked to the quality and extent of test data. Sets with lots of gaps and redundancy will match poorly to actual real-world challenges. Tests conducted using weak test data will result in weak conclusions. CV的测试通常是片面的：每个新算法都是在基准测试数据集上进行评估，但数据集本身很少进行独立的评估。这是一个严重的疏忽，被测试的应用的质量是与测试数据的质量和广度直接联系的。如果数据集有很多缺失和冗余，那么与真实世界的挑战就会相差甚远。在弱测试集上进行的测试，得到的将是弱的结论。

This work presents a new way to facilitate a safety assessment process to overcome these problems: a standard method developed by the safety community is applied to the CV domain for the first time. It introduces an independent measure to enumerate the challenges in a dataset for testing the robustness of CV algorithms.

本文为解决这个问题提出了一种新方法，以促进安全评估：这是安全领域提出的标准方法，第一次应用于CV领域。提出了一种独立的度量，列举出数据集测试CV算法的稳健性中的挑战。

The typical software quality assurance process uses two steps to provide objective evidence that a given system fulfills its requirements: verification and validation (International Electrotechnical Commission 2010). Verification checks whether or not the specification was implemented correctly (i.e. no bugs) (Department of Defense 1991). Validation addresses the question whether or not the algorithm is appropriate for the intended use, i.e., is robust enough under difficult circumstances. Validation is performed by comparing the algorithm’s output against the expected results (ground truth, GT) on test datasets. Thus, the intent of validation is to find shortcomings and poor performance by using “difficult” test cases (Schlick et al. 2011). While general methods for verification can be applied to CV algorithms, the validation step is rather specific. A big problem when validating CV algorithms is the enormous set of possible test images. Even for a small 8bit monochrome image of 640 × 480 pixels, there are already 256^640×480 ≈ 10^739811 possible images. Even if many of these combinations are either noise or render images which are no valid sensor output, exhaustive testing is still not feasible for CV. An effective way to overcome this problem is to find equivalence classes and to test the system with a representative of each class. Defining equivalence classes for CV is an unsolved problem: how does one describe in mathematical terms all possible images that show for example “a tree” or “not a car”? Thus, mathematical terms do not seem to be reasonable but the equivalence classes for images are still hard to define even if we stick to the semantic level. A systematic organization of elements critical to the CV domain is needed and this work will present our approach to supply this.

典型的软件质量保证过程，使用两个步骤来对给定系统满足其需求提出客观证据：verification和validation。Verification是检查，特定的目标是否得到了正确的实现（即，没有bugs）。Validation要处理的是，算法是否实现了想要的目的，即，在一些很困难的情况下是否足够稳健。Validation的进行，是在测试数据集中，将算法的输出与期望的结果（即真值，GT）比较。所以，validation的目的是通过使用困难的测试案例，来发现缺陷和表现差的地方。虽然一般的verification方法可以应用于CV算法，但validation的步骤是很独特的。Validating CV算法的一个大问题是，可能的测试图像有很多数据集。即使是一个很小的640 × 480大小的8bit灰度图像，已经有256^640×480 ≈ 10^739811个可能的图像。即使这些组合很多可能是噪声，或者是没有意义的图像，但要进行穷举测试的话，仍然是不太可行的。克服这个问题的一个有效途径是，找到等效的类别，用每个类别的代表来测试整个系统。在CV中定义一个等效的类别是一个尚未解决的问题：怎样在数学上，对所有展示一颗树，或不是一辆车的图像，进行统一描述呢？因此，数学项可能是不太行的，但即使我们在语义这个层级上，等效类别也仍然很难定义。需要对CV领域的元素进行系统性组织，本文给出的我们的方法就会给出这样的组织。

All in all, the main challenges for CV validation are: 总体来说，CV validation的主要挑战是：

- What should be part of the test dataset to ensure that the required level of robustness is achieved? 数据集应当包含什么样的数据，才能保证有足够的稳健性？

- How can redundancies be reduced (to save time and remove bias due to repeated elements)? 怎样才能去除冗余性（以节约时间并去除重复元素导致的偏移）？

Traditional benchmarking tries to characterize performance on fixed datasets to create a ranking of multiple implementations. On the contrary, validation tries to show that the algorithm can reliably solve the task at hand, even under difficult conditions. Although both use application specific datasets, their goals are different and benchmarking sets are not suited for validation.

传统基准测试在固定数据集上测定性能特征，以对不同的算法实现进行排序。相反，validation试图证明算法可以很好的处理手头的任务，即使在很困难的情形下。虽然都使用的是应用专用的数据集，但它们的目标不同，基准测试数据集并不是适合进行validation。

The main challenge for validation in CV is listing elements and relations which are known to be “difficult” for CV algorithms (comparable to optical illusions for humans). In this paper, the term visual hazard will refer to such elements and specific relations (see Fig. 1 for examples).

CV领域的validation的主要挑战，是将对CV算法很困难的元素及其关系列举出来（对人类来说，类似的是光学上的假象）。本文中，我们称这样的元素和关系为视觉危害(visual hazard)（图1为例子）。

By creating an exhaustive checklist of these visual hazards we meet the above challenges: 对这些视觉危害列出详细的检查单，以满足以下挑战：

- Ensure completeness of test datasets by including all relevant hazards from the list. 将列表中所有相关的危害点都包含在数据集中，确保测试数据集的完备性。

- Reduce redundancies by excluding test data that only contains hazards that are already identified. 将只包含已经鉴别出的危害点的测试数据排除，以降低冗余性。

Our main contributions presented in this paper are: 本文中我们的主要贡献为：

- application of the HAZOP risk assessment method to the CV domain (Sect. 3), 将HAZOP风险评估方法应用到CV领域；

- introduction of a generic CV system model useful for risk analysis (Sect. 3.1), 提出了进行风险分析的通用CV系统模型；

- a publicly available hazard checklist (Sect. 3.7) and a guideline for using this checklist as a tool to measure hazard coverage of test datasets (Sec. 4). 提出了一份公开可用的危害点检查单，和一份指南，使用这个检查单作为工具来度量测试数据集的危害点覆盖度。

To evaluate our approach, the guideline is applied to three stereo vision test datasets: KITTI, Middlebury 2006 and Middlebury 2014 (see Sect. 5). As a specific example, the impact of identified hazards on the output of multiple stereo vision algorithms is compared in Sect. 6.

为评估我们的方法，在三个立体视觉数据集上使用了这个指南，即KITTI, Middlebury 2006和Middlebury 2014。作为一个具体例子，我们在第6部分比较了，识别到的危害点对多个立体视觉算法的影响。

## 2 Related Work

Bowyer and Phillips (1998) analyze the problems related to validating CV systems and propose that the use of sophisticated mathematics goes hand in hand with specific assumptions about the application. If those assumptions are not correct, the actual output in real-world scenarios will deviate from the expected output.

Bowyer等分析了与CV系统validation相关的问题，提出使用复杂的数学与应用相关的具体假设结合起来的思路。如果这些假设不正确的话，真实世界情况的实际输出将会与期望输出有所偏移。

Ponce et al. (2006) analyze existing image classification test datasets and report a strong database bias. Typical poses and orientations as well as lack of clutter create an unbalanced training set for a classifier that should work robustly in real-world applications.

Ponce等分析了已有的图像分类测试数据集，说明数据集的片面性很严重。典型的姿态和方向，以及缺少杂乱的情况，形成了分类器的不均衡训练集，而这个数据集应当在真实世界应用中稳健的应用。

Pinto et al. (2008) demonstrate by a neuronal net, used for object recognition, that the currently used test datasets are significantly biased. Torralba and Efros (2011) successfully train image classifiers to identify the test dataset itself (not its content), thus, showing the strong bias each individual dataset contains.

Pinto等用一个目标识别神经网络证明了，目前使用的测试数据集有明显的片面性。Torralba等训练了图像分类器，成功的识别了测试数据集本身（而不是其内容），因此说明了，每个单独的数据集都包含很强的片面性。

A very popular CV evaluation platform is dedicated to stereo matching, the Middlebury stereo database. Scharstein and Szeliski (2002) developed an online evaluation platform which provides stereo datasets consisting of the image pair and the corresponding GT data. The datasets show indoor scenes and GT are created with a structured light approach (Scharstein and Szeliski 2003). Recently, an updated and enhanced version was presented which includes more challenging datasets as well as a new evaluation method (Scharstein et al. 2014). To provide a similar evaluation platform for road scenes, the KITTI database was introduced by (Geiger et al. 2012).

有一个非常流行的CV评估平台是针对立体匹配的，Middlebury立体视觉数据集。Scharstein等开发了一个在线评估平台，提供立体视觉数据集，包含图像对和对应的GT数据。数据集中的室内场景和GT是用结构光方法创建的。最近，平台进行了更新和升级，包含了更有挑战性的数据集，以及新的评估方法。为对公路场景进行类似的评估，Geiger等提出了KITTI数据集。

A general overview of CV performance evaluation can be found in (Thacker et al. 2008). They summarize and categorize the current techniques for performance validation of algorithms in different subfields of CV. Some examples are shown in the following: Bowyer et al. (2001) present a work for edge detection evaluation based on receiver operator characteristics (ROCs) curves for 11 different edge detectors. Min et al. (2004) describe an automatic evaluation framework for range image segmentation which can be generalized to the broader field of region segmentation algorithms. In Kondermann (2013) the general principles and types of ground truth are summarized. They pointed out, that thorough engineering of requirements is the first step to determine which kind of ground truth is required for a given task. Strecha et al. (2008) present a multi-view stereo evaluation dataset that allows evaluation of pose estimation and multi-view stereo with and without camera calibration. They additionally incorporate GT quality in their LIDAR-based method to enable fair comparisons between benchmark results. Kondermann et al. (2015) discuss the effect of GT quality on evaluation and propose a method to add error bars to disparity GT. Honauer et al. (2015) reveal stereo algorithm-specific strengths and weaknesses through new evaluation metrics addressing depth discontinuities, planar surfaces, and fine geometric structures. All of these are examples of visual hazards.

Thacker等对CV性能评估进行了概览。他们对CV不同领域的目前算法性能验证的技术进行了总结并分类。一些例子如下：Bowyer等提出了边缘检测的评估方法，这是对11种不同的边缘检测器的基于ROCs曲线评估。Min等对range图像分割提出了一个自动评估框架，这种方法可以泛化更广泛的区域分割算法中。Kondermann等总结了真值的一般原则和类型。他们指出，对于给定的任务来说，对需求的彻底研究是确定需要什么样真值类型的第一步。Strecha等提出了一个多视图的立体视觉评估数据集，可以评估姿态估计和多视角立体视觉，不管摄像头有没有校准过。他们还将基于LIDAR方法的GT质量纳入进来，以对不同基准测试结果进行公平的比较。Kondermann等讨论了GT质量对评估的影响，提出了一种方法对差异GT增加了误差条。Honauer等提出的新的评估度量标准解决了深度不连续性、平面表面和精细几何结构的问题，给出了不同立体视觉算法的优势和弱点。所有这些都是视觉危害的例子。

Current test datasets neither provide clear information about which challenges are covered nor which issues remain uncovered. Our approach can fill both gaps: By assigning a reference-table entry with a unique identifier to each challenging hazard, we create a checklist applicable to any dataset. To the best knowledge of the authors there is no published work considering the vision application as a whole, which identifies risks on such a generic level.

目前的测试数据集既没有给出那些挑战是覆盖到的信息，也没有哪些问题仍然未显露出的信息。我们的方法可以填补这两个空白：对每个有挑战性的危害，我们都指定参考表格中的一个唯一识别符，即创建了一个检查表，对任意数据集都可以应用。据我们所知，还没有文章将视觉应用作为一个整体来进行考虑，在这样通用的级别鉴别其中的风险。

### 2.1 Robustness

Depending on context, robustness can refer to different characteristics of the considered system. In the safety context, robustness is about the correct handling of abnormal situations or input data. For instance, in the basic standard for functional safety (International Electrotechnical Commission 2010), it is defined via the system’s behavior in a hazardous situation or hazardous event. This also includes the ability to perform a required function in the presence of implementation faults (internal sources) or cope with faulty and noisy input data (external sources). A method to evaluate robustness against the first type is fault injection (Hampel 1971) (e.g., bit flips in registers or bus) while fuzz testing (Takanen et al. 2008) can be used for assessing the robustness against abnormal input data.

依上下文的不同，稳健性可以是指系统的不同特性。在安全的上下文中，稳健性是指对输入数据的非正常情况进行正确处理。比如，在功能安全性上的基础标准上(International Electrotechnical Commission 2010)，稳健性的定义是在有危害的情况或危害性事件上的系统行为。这也包括，在有实现错误（内部源）时执行一个需要的功能，或应对含有错误或噪声的输入数据（外部源）。评估第一种类型的稳健性的方法是fault injection（如，在寄存器或总线上进行bit flips），而应对非正常的输入数据，则可以用fuzz testing进行评估。

In computer vision, robustness usually refers to coping with distorted or low-quality input. Popular methods are random sample consensus (RANSAC) (Fischler and Bolles 1981), M-Estimators (Huber 1964), or specific noise modeling techniques, which arose from the need to use systems in “real-world applications”. In the work described in this paper, we do not exclude these issues, but aim to cover all influences that may cause a degraded or false performance of a CV solution. This in particular includes aspects that can usually be part of observed scenes, such as lacking or highly regular textures, reflections, occlusions, or low contrasts. Figure 1 illustrates some examples.

在计算机视觉中，稳健性通常是指，应对变形的或低质量的输入数据。常用的方法是随机样本consensus(RANSAC)，M-Estimators，或特定的噪声建模技术，这些都是将系统在真实世界情况中应用的需求中得到的。在本文所述的工作中，我们没有排除这些问题，但目标是涵盖所有会导致CV算法性能下降或失效的影响。特别包括的方面，通常是所观察场景的一部分，比如缺少有规律的纹理，或高度有规律的纹理，倒影，遮挡，或低对比度。图1给出了一些样本。

### 2.2 Risk Analysis

Risk-oriented analysis methods are a subset of validation and verification methods. All technical risk analysis methods assess one or several risk-related attributes (e.g. safety or reliability) of systems, components or even processes with respect to causes and consequences. Some techniques additionally try to identify existing risk reduction measures and propose additional measures where necessary.

面向风险的分析方法，是validation和verification方法的一个子集。所有技术上的风险分析方法，都对系统的一个或几个风险相关的属性（如安全或可靠性）进行分析，或是对组件，甚至是与原因和结果相关的过程进行分析。一些技术还尝试研究已有的风险降低技术，并在需要的时候提出额外的措施。

Originally, risk identification techniques have been developed by the chemical industries, but nowadays they are successfully applied to software quality assurance as well (see Fenelon and Hebbron 1994 and Goseva-Popstojanova et al. 2003 for UML models). The most commonly used methods are: 开始的时候，风险识别技术是由化学工厂提出的，但现在也已经成功的应用于软件质量保证。最常用的方法有：

- HAZOP [7], (Kletz 1983)—hazard and operability analysis,

- FME(C)A (Department of Defense 1949 )—failure modes, effects, (and criticality) analysis,

- FTA (Vesely et al. 1981; Laprie 1992)—fault tree analysis.

Each risk analysis method defines a systematic process to identify potential risks. The first step in a HAZOP is to identify the essential components of the system to be analyzed. The parameters for each component, which define its behavior, have to be identified. These parameters often describe the input output characteristics of the component. A set of predefined guide words which describe deviations are applied to the parameters (e.g. “less” or “other than”) and the resulting combinations are interpreted by experts in order to identify possible consequences (potential hazards) and counteractions. While FME(C)A also starts with identifying the systems components and their operating modes, it then identifies the potential failure modes of the individual components. Further steps deal with identifying potential effects of these failures, their probability of occurrence, and risk reduction measures similar to HAZOP. FTA starts with a hazardous “top event” as root of the fault tree. Leaves are added recursively to the bottom events representing Boolean combinations which contain possible causes for their parent event (e.g. “own car hits the front car” if “speed too high” and “braking insufficient”). This refinement is executed until only elementary events are encountered.

每种风险分析方法都定义了一个系统性的过程，以辨别可能的风险。HAZOP的第一步是，鉴别要分析的系统的基础部件。每个部件的参数都要进行鉴别，因为这定义了其行为。这些参数通常描述了部件的输入输出性质。预定义的guide words描述了偏差，应用于参数（如less，或other than），得到的组合由专家进行解读，以预测可能的后果（可能的危害）和应对行为。FME(C)A也从鉴别系统部件和操作模式开始，然后鉴别单个模块的可能失败模式。后续的步骤处理的是，鉴别这些失败模式的可能后果，其发生的概率，降低风险的措施（与HAZOP类似）。FTA从一个有危害的top event开始，作为fault tree的根节点。在根事件的上面迭代的添加叶节点，代表的是Boolean组合，包含了其父事件的可能结果（如，如果速度过高，并且刹车不够，就会导致车撞上前面的车）。这种改良不断的执行，直到遇到基础事件。

## 3 CV-HAZOP

The identification and collection of CV hazards should follow a systematic manner and the results should be applicable to many CV solutions. The process has to be in line with well-established practices from the risk and safety assessment community to create an accepted tool for validation of CV systems. The most generic method HAZOP (Kletz 1983) is chosen over FME(C)A and FTA because it is feasible for systems for which little initial knowledge is available. In addition, the concept of guide words adds a strong source of inspiration that all other concepts are missing.

CV危害的鉴别和收集，应当是系统性的，其结果应当对很多CV算法都适用。这个过程应当与已经确定的risk and safety assessment团体的过程一致，以创建一个CV系统可以接受的validation工具。我们选用了最通用的HAZOP方法，而没有选择FME(C)A和FTA，因为这对于知道很多初始信息的系统都是可用的。除此以外，guide words的概念是灵感的重要源泉。

The following Sections address the main steps of a HAZOP: 下面的部分是HAZOP的主要步骤：

- Model the system. 对系统建模。

- Partition the model into subcomponents, called locations. 将模型分解为子部分，称为locations。

- Find appropriate parameters for each location which describe its configuration. 对每个location，寻找合适的参数，以描述其配置。

- Define useful guide words. 定义有用的guide words。

- Assign meanings for each guide word/parameter combination and derive consequences from each meaning. 为每个guide word/parameter的组合指定意义，并为每个意义推导出结果。

- Give an example clarifying the entry using for a specific application (e.g. in the context of stereo vision, object tracking, face detection). 对特定的应用（如立体视觉，目标追踪，人脸检测），对每个entry都给出一个example。

### 3.1 Generic Model

The first step of any HAZOP is deriving a model of the system that should be investigated. In case of this HAZOP, the generic CV algorithm has to be modeled together with the observable world (its application). Marr (1982) proposes a model for vision and image perception from the human perception perspective. Aloimonos and Shulman (1989) extended it by the important concepts of stability and robustness. We propose a novel model which is entirely based on the idea of information flow: The common goal of all CV algorithms is the extraction of information from image data. Therefore “information” is chosen to be the central aspect handled by the system. It should be noted, that “information” is used in the context “Information is data which has been assigned a meaning.” Van der Spek and Spijkervet (1997) rather than in a strict mathematical sense (Shannon and Weaver 1949). In this context, hazards are all circumstances and relations that cause a loss of information. Even though hazards ultimately propagate to manifest themselves in the output of the algorithm, an effective way to find a feasible list of hazards is to look at the entire system and attribute the hazard to the location where it first occurred (e.g. unexpected scene configuration or sensor errors). Multiple inputs from different disciplines are used to create the system model:

任何HAZOP的第一步骤是，对要研究的系统进行建模。在本文的HAZOP的情况下，通用CV算法与其可观察的世界（其应用）一起进行建模。Marr从人类感知的角度，提出了视觉和图像感知的模型。Aloimonos等对其进行了拓展，提出了稳定性和稳健性的重要概念。我们提出一个新的模型，完全是基于信息流的思想的：所有CV算法的通用目标是，从图像数据中提出信息。因此，我们选取信息作为系统要处理的核心内容。应当指出，信息是在如下上下文中使用的，“信息是指定了意义的数据”。Van der Spek等在严格的数学意义上讨论了这个问题。在这个语境下，危害是所有会导致信息损失的情况和关系。即使危害最终传播，在算法的输出中得到自我表达，但要找到危害的可行列表，一个有效的方法是整个系统中查找，将危害归因到其首先发生的location（如，非预期的场景配置，或传感器错误）。我们使用不同原则的多个输入来创建系统模型：

**Information Theory**. Communication can be abstracted according to information theory (Shannon and Weaver 1949) as information flow from the transmitter at the source—with the addition of noise—to the receiver at the destination. 通信可以抽象为信息论，因为信息从源的发射器（和加入的噪声一起）传送到目标的接收器。

**Sampling Theorem**. Sampling is a key process in the course of transforming reality into discrete data. Artifacts that can be caused by this process, according to (Nyquist 1928; Shannon 1949), will result in a loss of information. 在将现实转化为离散数据的过程中，采样是一个关键的过程。这个过程可能导致伪影，这会导致信息损失。

**Rendering Equation**. The rendering equation (Kajiya 1986) is a formal description of the process of simulating the output of a virtual camera within a virtual environment. The different parts of the standard rendering equation amount to the different influences that arise when projecting a scenery light distribution into a virtual camera. 渲染方程是模拟虚拟摄像机在一个虚拟环境中的输出的过程的正式数学表示。标准渲染方程的不同部分，负责在场景光分布投射到虚拟摄像机时，所产生的不同影响。

**Control Theory**. The general system theory (e.g. Von Bertalanffy 1968) and especially cybernetics interpret and model the interactions of systems and the steps of acquiring, processing, as well as reacting to information from the environment. 通用系统理论，尤其是控制论，对模型的互相作用，和从环境中获取信息，处理信息以及对信息的反应，进行翻译和建模。

The entire flow of information is modeled as follows: 信息流的整个过程建模如下：

- Since in CV the sensor is a camera, all data within the observed scene available to a CV component can only be provided by the electromagnetic spectrum (simply referred to as light in this paper) received by the observer (i.e. the sensor/camera) from any point in the scene. Hence, light represents data and, as soon as a meaning is assigned, information. 在CV中传感器是摄像机，在观察场景中可用的所有数据，对CV部分来说，都只能由观察者（即，传感器/摄像机）在场景中的任意一点接收到的电磁光谱（本文中称之为光）所提供。因此，光代表了数据，只要为其指定一个意义，那么就是信息。

- At the same time, any unexpected generation of light and unwanted interaction of light with the scene distorts and reduces this information. 同时，任何非预期的光的产生，和光与场景的非预期的相互作用，都会使得这种信息扭曲并有损失。

- The sensing process, i.e. the transformation of received light into digital data, further reduces and distorts the information carried by the received light. 感知的过程，即，接收到的光转换称数字数据，进一步降低并扭曲了接收到的光所携带的信息。

- Finally, the processing of this data by the CV algorithm also reduces or distorts information (through rounding errors, integration etc.). 最后，CV算法对这些数据的处理，也使得信息有损失或扭曲（通过四舍五入的误差，积分等等）。

In essence, two information carriers are distinguished: light outside of the system under test (SUT) and digital data within the SUT. This is visualized in Fig. 2 by two types of arrows: solid arrows for light and a dashed line for digital data. At each transition, this information can potentially be distorted (e.g. by reduction, erasure, transformation, and blending). Benign interactions, e.g. interaction of a pattern specifically projected to create texture for structured light applications, are not entered into the list. We are interested in situations and aspects that can potentially reduce output quality. Nevertheless, the failing of such expected benign interactions (e.g. inference effects of multiple projectors) are a risk and, thus, included in the analysis.

本质上说，区分出了两种信息携带者：要测试的系统(SUT)的外部的光，和SUT内部的数字数据。在图2中，用两种类型的箭头对其进行可视化：实线箭头代表光，虚线代表数字数据。在每种转换时，这些信息都可能被扭曲（如，信息减少，抹除，变换，混合）。良性的相互作用，如结构光的应用，没有列入到列表中。我们对可能降低输出质量的情况和方面感兴趣。尽管如此，这种良性相互作用的失败（如，多个投影仪的干涉作用）是一个风险，因此也包含在本文的分析中。

### 3.2 Locations

The system model is now partitioned into specific locations (i.e. subsystems) of the overall system. Light sources that provide illumination start the process flow (illustrated in Fig. 2). The light traverses through a medium until it either reaches the observer or interacts with objects. This subprocess is recursive and multiple interactions of light with multiple objects are possible. The observer is a combination of optical systems, the sensor, and data pre-processing. Here the light information is converted into digital data as input for a CV algorithm. The CV algorithm processes the data to extract information from it.

系统的模型现在分解成整体系统的具体location（即，子系统）。光源提供了照明，开始了这个过程（如图2所示）。光线穿过介质，到达观察者，或与目标进行相互作用。这个子过程是循环的，光可能与多个目标进行多种相互作用。观察者是光学系统、传感器和数据预处理过程的组合。这里，光线的信息转化成了数字数据，成为CV算法的输入。CV算法对数据进行处理，并从中提取出信息。

Each entity (box in Fig. 2) represents a location for the HAZOP. The recursive loop present in the model results in an additional location called “Objects” for aspects arising from the interactions between multiple objects. The observer is modeled by two components: “Observer—Optomechanics” and “Observer—Electronics”. This reduces complexity for the analysis and allows to focus on the different aspects of the image capturing process.

每个实体（图2中的方框）都代表了HAZOP的一个location。模型中的循环过程会得到更多的location，我们称之为Objects，即在多个objects之间的相互作用导致的。观察者由两个部分进行建模：Observer—Optomechanics和Observer—Electronics。这减少了分析了复杂度，使我们可以关注图像获取过程的不同部分。

### 3.3 Parameters

Each location is characterized by parameters. They refer to physical and operational aspects describing the configuration of the subcomponent. The set of parameters chosen for a single location during the HAZOP should be adequate for its characterization. Table 2 shows the parameters chosen for the location “Medium” as an example. Too few parameters for a location means that it is insufficiently modeled and that the analysis will likely contain gaps. Performing an analysis with too many parameters would require too much effort and create redundancy. A full listing of all parameters is available at the website vitro-testing.com.

每个location都有相应的参数。这指的是子部分配置的物理方面和操作方面。在HAZOP过程中，为某个location选择的参数集合，应当足以描述其特征。表2展示了为location Medium选择的参数，是一个例子。为一个location选择的参数过少，意味着其建模是不充分的，所以对其的分析很可能存在空白。用过多的参数进行分析，会导致耗费过多，产生冗余。所有参数的完整列表在网站上可见。

### 3.4 Guide Words

A guide word is a short expression to trigger the imagination of a deviation from the design/process intent. Number and extent of guide words must be selected to ensure a broad view on the topic. Nevertheless, their number is proportional to the time needed for performing the HAZOP, so avoiding redundant guide words is essential. The provided examples in Table 1 show all guide words we used in the analysis. Exemplary meanings for the deviations caused by each guide words are given, but the experts are not limited to these specific interpretations during the risk analysis. The first seven “basic” guide words are standard guide words used in every HAZOP. The remainder are adaptations and additions that provide important aspects specific for CV: spatial and temporal deviations (Table 2).

Guide word是一个短语，触发从设计/处理意图的偏移的想象。要选择guide words的数量和程度，以确保主题覆盖的视野很宽。尽管如此，数量的选择与进行HAZOP分析的时间成正比，所以避免冗余的guide words非常重要。表1中给出的例子是我们在分析中用到的所有的guide words。每个guide words导致的偏移的范例意思如表1所示，但专家在风险分析的时候不会局限于这些特定的解释。前7个基本的guide words是在每个HAZOP分析中用到的标准guide words。剩下的是为CV领域所作出的修正和添加：即空间偏移，或时间偏移。

Table 1 Guide Words used in the CV-HAZOP

Basic Guide word | Meaning | Example
--- | --- | ---
No | No information can be derived  | No light at all is reflected by a surface
More | Quantitative increase (of parameter) above expected level | Spectrum has a higher average frequency than expected
Less | Quantitative decrease below expected level | Medium is thinner than expected
As well as | Qualitative increase (additional situational element) | Two lights shine on the same object
Part of | Qualitative decrease (only part of the situational element) | Part of an object is occluded by another object
Reverse | Logical opposite of the design intention occurs | Light source casts a shadow instead of providing light
Other than | Complete substitution—another situation encountered | Light source emits a different light texture

Additional Guide word - Spatial | Meaning | Example
--- | --- | ---
Where else | “Other than” for position/direction related aspects | Light reaches the sensor from an unexpected direction
Spatial periodic | Parameter causes a spatially regular effect | A light source projects a repeating pattern
Spatial aperiodic | Parameter causes a spatially irregular effect | The texture on object shows a stochastic pattern
Close/remote | Effects caused when s.t. is close to/remote of s.t. else | Objects at large distance appear too small
In front of/behind | Effects caused by relative positions to other objects | One object completely occludes another object

Additional Guide word - Temporal | Meaning | Example
--- | --- | ---
Early/Late | Deviation from temporal schedule | Camera iris opens too early
Before/after | A step is affected out of sequence,relative to other events | Flash is triggered after exposure of camera terminated
Faster/slower | A step is not done with the right timing | Object moves faster than expected
Temporal periodic | Parameter causes a temporally regular effect | Light flickers periodically with 50 Hz
Temporal aperiodic | Parameter causes a temporally irregular effect | Intensity of light source has stochastic breakdowns

Table 2 Parameters used in the location Medium

Parameter | Meaning
--- | ---
Transparency | Dimming factor per wavelength and distance unit
Spectrum | Color, i.e. richness of medium with respect to absorption spectrum (isotropic or anisotropic)
Texture | Generated by density fluctuations and at surfaces (e.g. water waves)
Wave properties | Polarization, coherence
Particles | Influences and effects of the particles that make up the medium

### 3.5 Implementation

The actual implementation of the HAZOP is the systematic investigation of each combination of guide words and parameters at every location in the system. It is performed redundantly by multiple contributors. Afterwards, the results are compared and discussed to increase quality and completeness. Each HAZOP contributor assigns at least one meaning to a combination. In addition, for each meaning found the contributors investigate the direct consequences of this deviation on the system. One meaning can result in multiple consequences at different levels. Each entry in the list represents an individual hazard which can lead to actual decreases in the total system’s performance or quality. Combinations that result in meaningful interpretations by any contributor are considered to be “meaningful” entries while combinations without a single interpretation are considered to be “meaningless”.

HAZOP的实际实现，是系统中每个guide words和参数在每个location的组合的系统性研究。这是由多个贡献者冗余执行的。然后，结果进行比较并讨论，以提高质量和完备性。每个HAZOP贡献者对一个组合至少指定一个意义。另外，对发现的每个意义，贡献者要对在系统中导致的偏差的直接结果进行调查。一个意义可以在不同的层次导致很多结果。列表中的每个条目代表一个单独的危害，都可以导致系统总体性能或质量的降低。任何贡献者可以将组合推理为有意义的解释的，这个组合就是有意义的条目，而不能得到解释的，就认为是无意义的组合。

### 3.6 Execution

The execution of the CV-HAZOP, including various meetings and discussions by the contributors (with expertise in testing, analysis, and CV), took one year. Each location is covered by at least three of the authors. The additional experts are mentioned in the acknowledgments. The 52 parameters from all seven locations, combined with the 17 guide words, result in 884 combinations. Each combination can have multiple meanings assigned to it. Finally, 947 unique and meaningful entries have been produced. Table 3 shows an excerpt of entries from the final HAZOP and Fig. 3 shows visualizations for each hazard mentioned. The entries in the list can include multiple meanings for each parameter as well as multiple consequences and hazards per meaning. The whole resulting dataset of the CV-HAZOP is publicly available at www.vitro-testing.com.

CV-HAZOP的执行，包括贡献者的各种不同的会议和讨论（与专家在CV领域进行测试、分析），会有长达一年的时间。每个location都需要由至少3个作者研究。在致谢中提到了更多的专家。7个locations的52个参数，与17个guide words进行组合，得到884个组合。每个组合可以指定多个意义。最后，产生了947个唯一并有意义的条目。表3给出了最终HAZOP的条目的节选，图3给出了每个提到的危害的可视化。列表中的条目，在每个参数时可以包括多个意义，每个意义可以有多个后果和危害。CV-HAZOP得到的完整数据集在www.vitro-testing.com可以公开访问。

### 3.7 Resulting List

In total, 947 entries are considered meaningful by the experts. A detailed analysis of the meaningful entries achieved for each guide word/parameter combination is shown in Fig. 4. One goal is to maximize the meaningful entries—and the graphic shows reasonably high entries for most of the basic guide words (see Table 1). Lower valued entries in the matrix can be explained as well: The concepts of the spatial aspects “Close” and “Remote” are simply not applicable to the properties of the electronic part of the observer (obs. electronics) and the concept of space in general is not applicable to a number of parameters at various locations. This also holds true for the temporal guide words which do not fit to the optomechanical and medium locations. Nevertheless, even here the usage of guide word/parameter combinations inspire the analysts to find interpretations which would have been hard to find otherwise. Each hazard entry is assigned a unique hazard identifier (HID) to facilitate referencing of individual entries of the checklist.

专家认为总计有947个有意义的条目。图4给出了有意义的条目在每个guide word/参数的组合下的详细分析。一个目标是使得有意义的条目数最大化，图4给出了大多数基本guide words的条目还是挺高的（见表1）。矩阵中值较低的条目也可以得到解释：空间中的概念close和remote，对于观察者的电子部分都是不能应用的，空间的概念总体上在几个location上对几个参数都是无法应用的。这个结论对于时间guide words也是对的，在光力学和媒介的location中也是无法应用的。尽管如此，这里guide word/参数的组合启发了分析者找到了一些解释，在其他地方曾是很难找到的。每个危害条目都指定了唯一的危害识别符(Hazard Identifier)，以帮助参考检查列表的条目。

## 4 Application

The remainder of this paper focuses on the application of the checklist as an evaluation tool for existing test datasets. On the one hand, we show that the CV-HAZOP correctly identifies challenging situations and on the other hand, we provide a guideline for all researches to do their own analysis of test data.

本文剩下的部分关注检查列表的应用，作为工具对现有的测试数据集进行评估。另一方面，我们展示了CV-HAZOP正确的识别了很有挑战的情况，另一方面，我们为所有研究者对测试数据进行其自己的分析提供了一个指南。

Initially, the evaluators have to clarify the intent and domain of the specific task at hand. This specification creates the conceptual borders that allow the following analysis to filter the hazards. The intent includes a description of the goals, the domain defines the conditions and the environment under which any algorithm performing the task should work robustly. With the intent and domain specified, the evaluators can now check each entry of the CV-HAZOP list to see if that entry applies to the task at hand. Often it is useful to reformulate the generic hazard entry for the specific algorithm to increase readability. In the following a process outline is given:

开始，评估者需要明晰意图，以及手头的特定任务。这种明确性约束了概念，使得滤除hazard后续的分析成为可能。意图包括目标的描述，领域定义了条件和环境，在这些条件和环境下，任何算法执行的任务都应当很稳健的工作。在指定了意图和领域后，评估者可以检查每个CV-HAZOP列表的条目，以看这个条目是否可以应用到当前的任务。对特定的算法，重组通用的危害条目，通常是很有用的，可以增加可读性。下面，给出了过程概述：

1. Check if the preconditions defined by the column Meaning and the according Consequences apply. 检查Meaning列和Consequences列是否适用；
2. Check if the Example matches the specific task at hand. 检查Example是否与手头的特定任务匹配；
3. Each row represents a unique Hazard and has a unique Hazard ID (HID). If the Hazard is too generic to be feasible, add a new row for the specific task using a matching Example. 每一行代表唯一的一个Hazard，并有唯一的Hazard ID(HID)。如果Hazard太通用了，不可行，对特定任务使用一个匹配的Example增加一个新行；
4. Evaluate if the Hazard can be detected (i.e. is visible in the test data). 评估Hazard是否可以被检测到（即，在测试数据中是否可见）；
5. Store the identity of test cases which fulfill relevant HIDs. Create a new test case should none of the current test cases fulfill this Hazard. 对符合相关HID的测试案例，存储其ID。创建一个新的测试案例，直到目前的测试案例都不符合这个Hazard的条件。

Previous evaluations for comparable tasks can be used as templates to speed up this process and to reduce the effort compared to evaluating the whole generic list. Specialized hazards can be added to the checklist so that they can be used directly in future evaluations.

在类似任务上的之前的评估，可以用作模板，以加速此过程，与评估完整的通用列表相比，可以降低耗时耗力。特定的危害可以加到检查列表中，这样可以在未来的评估中直接使用。

With the reduced list of possible hazards, the evaluators are able to go through test datasets and mark the occurrence of a hazard. Usually a simple classification per test case is enough. Individual pixel-based annotations can also be used to indicate the location of specific hazards in test images (see Sect. 5). After this process, the missing hazards are known and quantifiable (e.g. 70% of all relevant hazards are tested using this test dataset). This is a measure of completeness which can be used to compare datasets. Even more important: If a hazard cannot be found in the test data, the CV-HAZOP entry states an informal specification for creating a new test case to complement the test dataset. The extensiveness of the checklist allows a thorough and systematic creation of new test datasets without unnecessary clutter.

可能危害的列表缩减后，评估者可以翻查测试数据集，对危害的情况标记下来。通常每个测试案例有一个简单的分类就足够了。像素级标注的图像也可以用于指明在测试图像中特定危害的位置（见第5节）。在这个过程后，缺失的危害就知道了，也可以量化了（如，所有相关的危害的70%都用这个测试数据集进行了测试）。这是一个完备性的度量，可以用于比较数据集。更重要的是：如果在测试数据集中找不到这个危害，CV-HAZOP条目给出一个非正式的说明，并创建一个新的测试案例，以补全测试数据集。检查列表的广泛性，使新的测试数据集的创建既彻底又具有系统性，而且还没有不必要的杂乱。

Each hazard entry in the check list has a unique hazard identifier (HID). This allows to easily reference individual hazards and compare results from different CV implementations. The checklist approach allows for a top-down evaluation of CV (starting from the problem definition down to the pixel level). This is a good complement to regular benchmarks which tend to be focused on the detailed pixel level (bottom-up evaluation).

在检查列表上的每个危害条目，都有唯一的危害识别符HID。这可以很容易的查找单个危害，比较不同的CV实现的结果。检查列表方法可以对CV进行自上而下的评估（从问题定义开始，到像素级）。常规benchmarks通常关注的像素级的细节（自下而上的评估），这对其是一个很好的补充。

## 5 Example

As proof of concept, the authors applied the described process to a specific task. We chose canonical stereo vision: 作为概念验证，作者对特定任务应用了上述的过程。我们选择了经典的立体视觉：

The intent of the algorithm is the calculation of a dense disparity image (correspondence between the pixels of the image pair) with a fixed epipolar, two camera setup. To further simplify the analysis, we only use greyscale information and assume that the cameras are perfectly synchronous (exposure starts and stops at the same instants), and omit the use of any history information so that many time artifacts can be disregarded. The domains of the algorithm are indoor rooms or outdoor road scenarios. Conditions like snow, fog, and rain are included in the problem definition. This was done to keep the problem definition sufficiently generic to allow room for the analysis.

算法的目的是计算密集差异图像（图像对的像素的对应性），并有一个固定的epipolar，双摄像头设置。为进一步简化分析，我们只使用灰度信息，并假设相机是完美同步的（曝光的开始和停止是在相同的瞬间），并忽略了任何历史信息的使用，所以就可以不用考虑多次的伪影。算法的domain是室内环境，或室外道路场景的。像雪，雾和雨的条件，都包括在问题定义中。这使得问题定义充分通用，也有分析的空间。

Note that this evaluation is not designed to compare stereo vision algorithms themselves or to compare the quality of the specific datasets (will be done in future works). However, this paper provides the first step: a clear proof of concept of the CV-HAZOP list as a tool for validation. The simplifications in domain/intent analysis and algorithm evaluation were performed to reduce complexity/workload and should be re-engineered for a specific stereo vision evaluation.

注意，这种评估的设计，并不是为了比较立体视觉算法本身，或比较特定数据集的质量（这在将来的工作中会进行）。但是，本文给出了第一步：CV-HAZOP列表作为validation工具的清晰概念验证。Domain/intent分析的简化，和算法评估的进行，可以降低复杂度，应当重新进行工程设计，以适用于特定的立体视觉评估。

First, six experts in the field of CV (some had experience with the CV-HAZOP list, others were new to the concept) analyzed the initial 947 entries and identified those applying to the stereo vision use case. During this step, 552 entries were deemed to be not applicable and 106 entries were non-determinable (not verifiable by only surveying the existing test data; more background knowledge needed). The remaining 289 entries were deemed to be relevant for stereo vision. See Table 4 and Fig. 5 for examples from the datasets. About 20% of the hazard formulations were further specified to simplify the following annotation work while the rest were already specific enough. The experts analyzed three test datasets commonly used for stereo vision evaluation (see Table 5) individually for each of the identified hazard.

第一，六位CV领域的专家（一些有CV-HAZAOP列表的经验，其他则对这个没有概念）分析了初始的947个条目，并识别其对立体视觉应用的适用性。在这一步骤中，552个条目认为不适用，106个条目不太确定（只调查现有的测试数据，发现不可验证；需要更多的背景知识）。剩下的289个条目认为与立体视觉应用相关。见表4和图5，是数据集中的例子。大约20%的危害都进一步进行了指定，以简化后续的标注工作，剩下的则已经很具体了。专家分析了三个常用的立体视觉测试数据集（见表5），对每个鉴别到的危害都进行了单独分析。

The hazard entries were evenly distributed among evaluators. All evaluators had the task to annotate each assigned hazard at least once in each dataset (if present at all). The step to annotate all occurrences of individual hazards in all images was omitted as the required effort would exceed the resources reasonable for this proof of concept. One representative of each hazard is deemed sufficient for the purpose of this proof-of-concept but certainly requires a larger sample size for a detailed evaluation of a CV algorithm. Ideally, a test dataset should include a systematically increasing influence of each hazard so that the algorithm’s point of failure can be evaluated.

危害实体在评估者中均匀分布。所有评估者都有一个任务，就是将每个指定的危害在每个数据集中至少标注一次（如果存在的话）。在所有图像中标注单个危害的所有发生情况，这在工作量上是不太可行的，所以就不进行这个工作了。每个危害有一个代表性例子，对于概念验证，就可以认为足够了，但是当然需要更多的样本数量，来CV算法进行evaluation。理想情况下，测试数据集应当每个危害都有一定数量的样本，这样算法的失败点可以得到评估。

The annotation tool was set to randomly choose the access order to reduce annotation bias by removing the influence of image sequence ordering. Table 5 summarizes the results of the evaluation showing the number of images with hazards and the number of uniquely identified hazards. It is not a surprise that KITTI contains the most hazards: it is the largest dataset and is also created in the least controlled environment (outdoor road scenes). It contains many deficiencies in recording quality manifesting as hazards and it includes images with motion blur as well as reflections on the wind-shield.

标注工具会随机选择访问顺序，去除了图像顺序的影响，就降低了标注偏差。表5总结了评估的结果，展示了有危害的图像数量，和识别到的唯一危害的数量。KITTI有最多的危害点，这也是符合预期的：这是最大的数据集，也是在受控最少的环境中创建的数据集（室外道路场景）。在录像质量上，其包含了很多缺陷，表现为危害点，也包含运动模糊的图像，和挡风玻璃上的反射图像。

Many effects stemming from interactions of multiple light sources, medium effects, and sensor effects are missing in all three test datasets. The majority of hazards present in the data deal with specific situations that produce overexposure (HIDs 26, 125, 479, 482, 655, 707, 1043, 1120), underexposure (HIDs 21, 128, 651, 1054, 1072, 1123), little texture (HIDs 444, 445, 449) and occlusions (HIDs 608, 626).

很多由于多个光源、媒质和传感器的互相作用的效果，在所有三个测试数据集上都是没有的。数据中的主要的危害点对应的都是特定的情况，如过曝(HIDs 26, 125, 479, 482, 655, 707, 1043, 1120), 曝光不足(HIDs 21, 128, 651, 1054, 1072, 1123), 无纹理(HIDs 444, 445, 449) 和遮挡(HIDs 608, 626).

## 6 Evaluation

In this section we evaluate the effect of identified hazards on algorithm output quality. The goal is to show that the entries of the CV-HAZOP are meaningful and that the checklist is a useful tool to evaluate robustness of CV algorithms. A specific hazard can only impact the system if it is visible in the image. Thus, we need to annotate areas in images corresponding to specific hazards to show that the annotated area itself (and, thus, the included hazard) is responsible for the output quality decrease. Initially it was unclear how accurate these areas have to be defined. For this purpose two different types of annotations were evaluated: a manually selected outline and a bounding box calculated from the outline.

本节中，我们评估了鉴别到的危害对算法输出质量的效果。目标是表明，CV-HAZOP的条目是有意义的，检查列表是评估CV算法稳健性的有用工具。特定的危害，只有在其在图像中是可见时，才会对系统有影响。所以，我们需要注明特定危害在图像中的对应区域，以表明注明的区域（因此，也是对应的危害）是要输出质量的下降的原因。初始的时候，是不太清楚，这个区域应当是怎样定义的。为这个目的，我们评估了两种类型的标注：一个是手工选定的轮廓，还有一个是从轮廓计算得到的边界框。

We potentially add another bias to our analysis by evaluating only areas that contain annotations. This has two influences: (i) We only look at frames that have annotations while ignoring all other frames in the dataset without any annotations, (ii) We average over small sampling windows that often contain relatively little data due to missing values in the GT.

我们可能会对我们的分析增加另一个偏差，只评估包含标注的区域。这有两个影响：(i)我们只查看包含标注的帧，而忽略数据集中所有其他不包含任何标注的帧；(ii)我们在小采样窗中进行平均，这些窗中通常包含相对较少的数据，因为GT中没有相应的值。

To quantify these influences we generated another set of control annotations: for each annotation in the dataset we generated a mask with a random position but the same size as the annotated hazard in the respective frame.

为量化这些影响，我们生成了另一个控制标注的集合：对数据集中的每个标注，我们生成一个位置随机的mask，但大小与相应帧中的标注危害相同。

At last the overall performance of an algorithm was needed as a base line value. For this the whole image was evaluated. All in all we generated four types of masks from the annotations for our evaluation.

最后，需要一个算法的总体性能作为基准线值。对于这个来说，要对整幅图像进行评估。总的来说，我们从标注中生成了四种类型的mask，以进行评估。

The different masks represent a step-by-step increase of influence of the annotated areas: 不同的masks代表对标注区域的一步一步逐渐增加的影响：

- shape masks with the annotated outlines as filled polygons, 带有标注的轮廓的形状masks，填充成多边形；
- box masks with boxes of equal size and centroid as each annotated outline, 矩形masks，其大小和重心与每个标注的轮廓相同；
- rand masks with boxes of equal size as the annotated outlines but a randomly placed centroid, 随机masks，其框的大小与标注的轮廓相同，但随机放置重心；
- all masks with all pixels except the left border region (to exclude occlusions). all masks是除了左边边缘区域的所有像素，以排除遮挡。

Figure 6 gives an example of the generated masks. Not every image in the test datasets contains annotations. The masks shape, box, and rand are evaluated for the subset of images containing at least one annotation while all is evaluated for all images of the datasets.

图6给出了生成的masks的一个例子。并不是测试数据集的每一幅图像都有标注。形状masks，矩形masks和随机masks是对至少含有一个标注的图像进行的评估，而all masks是对数据集的所有图像进行的评估。

The rand masks only represent the annotated area’s size as well as the subset of annotated frames. A total of 100 random masks are generated for each annotation that share its size but are randomly displaced. Statistics can thus be evaluated over the whole set of random masks which increases the significance. Annotation box represents area and position while shape represents the full annotation.

随机masks只代表标注区域的大小，以及标注的帧的子集。对每个标注，总计100个随机masks，其大小相同，但随机放置位置。这样就可以在整个随机masks组成的集合上进行评估，得到其统计值，以增加其显著性。标注框代表区域和位置，而形状表示其完整标注。

The rand versus all masks verify if the output quality is affected by using smaller image parts for evaluation instead of the whole image as well as a subset of frames, while box versus shape evaluates the influence of specific shapes of the annotations.

随机masks与all masks验证的是，如果使用很小的图像部分进行评估，而不使用整幅图像，输出质量是否受到影响；而框masks和形状masks评估的是特定形状标注的影响。

Table 5 lists the resulting number of annotations created for each dataset. Some hazards require the selection of split areas, resulting in multiple annotations. We only use pixels with valid GT information for evaluation. Unfortunately, many of the hazards (e.g. reflections, transparencies, occlusions, very dark materials) also have a negative influence on the laser scanner used for the GT generation in KITTI. The GT data is generally sparse and even more sparse in the annotated areas.

表5列出了为每个数据集创建的标注数量的得到值。一些危害需要选择分裂区域，这样会得到多个标注。我们只使用含有有效GT信息的像素进行评估。不幸的是，很多危害（如，反射，透明，遮挡，非常暗的材料）对生成KITTI的真值的激光扫描器也有负面作用。GT数据一般来说是稀疏的，在标注的区域是更稀疏的。

### 6.1 Performance Evaluation

For evaluation of the stereo vision test dataset we used the following popular stereo vision algorithms: SAD + texture thresholding (TX) & connected component filtering (CCF) (Konolige 1998), SGBM + TX & CCF (Hirschmüller 2008), census-based BM + TX & CCF (Humenberger et al. 2010; Kadiofsky et al. 2012), cost-volume filtering (CVF) & weighted median post processing filtering (WM) (Rhemann et al. 2011), PatchMatch (PM) & WM (Bleyer et al. 2011), and cross-scale cost aggregation using census and segment-trees (SCAA) & WM (Zhang et al. 2014), (Mei et al. 2013). The resulting disparities of each stereo vision algorithm are compared to the GT disparities of the test dataset. The number of wrong pixels (with an error threshold of >2px) is then compared to the number of pixels within the respective mask that had valid ground truth values. Invalids in the result are counted as being above any threshold. We consider each disparity pixel $d_i ∈ R^⋆$ to either be valid (∈ R) or invalid (denoted by the star value “⋆”). Where $R^⋆$ = R ∪ {⋆}. The same holds for each corresponding ground truth pixel value $g_i ∈ R^⋆$. We consider every $d_i$ for which $correct(d_i, g_i) = true$ to be true, and $correct : R⋆ × R⋆ → true$, false to be defined by:

评估立体视觉测试数据集，我们使用下列流行的立体视觉算法：SAD+纹理阈值(TX)&连接分量滤波(CCF)，SGBM+TX & CCF，基于census的BM+ TX & CCF，cost-volume filtering (CVF) & weighted median post processing filtering (WM)，PatchMatch (PM) & WM，cross-scale cost aggregation using census and segment-trees (SCAA) & WM。每个立体视觉算法得到的偏差值，与测试数据集的真值偏差值进行比较。错误像素的数量（误差阈值>2像素），与在相应的mask中有效真值的像素数量进行比较。无效结果被认为是高于任意阈值的。我们认为每个差值像素$d_i ∈ R^⋆$，可以是有效的(∈ R)，或无效的（表示为星号⋆），其中$R^⋆$ = R ∪ {⋆}。对于每个对应的真值像素$g_i ∈ R^⋆$也是这样的。我们认为每个满足$correct(d_i, g_i) = true$的$d_i$为真，$correct : R⋆ × R⋆ → true$，假是定义为：

$$correct(g_i, d_i) = \begin{cases} true, & for \space d_i = R ∧ g_i \neq ⋆ ∧ |d_i-g_i|<2 \\ false & else \end{cases}$$(1)

The actual comparison is performed for each dataset independently according to the average error $ē_m$ as defined by (2) where $D_m, G_m$ are the disparity and GT values selected by a given mask m ∈ { “shape”, “box”, “rand”, “all”}.

实际的比较是对每个数据集独立进行的，根据的是由式(2)定义的平均误差$ē_m$，其中$D_m, G_m$是差异值和GT值，m代表给定的mask，m ∈ { “shape”, “box”, “rand”, “all”}。

$$ē_m = \frac {|\{∀d_i ∈ D_m, g_i ∈ G_m : ¬correct(d_i, g_i)\}|} {|\{∀g_i ∈ G_m: g_i ∈ R\}|}$$(2)

Figure 7 shows the result of the evaluation for all three datasets and all four mask types. The arithmetic average of the performance evaluated for 100 random masks are reported as rand. We chose to use a high threshold of 2pxl to distinguish the coarse cases “algorithm succeeded at finding a good correspondence” versus “algorithm could not determine a correct correspondence” as opposed to measuring small measurement errors. The performances of the different mask types creates a distinct picture. Section 6.2 will first interpret the results. The following Sect. 6.3 will then assign statistical significance to these interpretations.

图7展示了所有三个数据集和所有四种mask类型的评估结果。对100个随机masks评估的性能的代数平均，注为rand。我们用了很高的阈值，即2像素，来区分“找到很好的对应性的算法”，和“不能确定正确的对应性的算法”。不同的mask类型的性能，形成了一个鲜明的图像。6.2节会第一次解释这个结果。后面的6.3节会给这些解释指定统计的显著性。

### 6.2 Interpretation

The effect of applying the masks based on the identified hazards can be clearly seen. Table 6 summarizes the ratios between the error values of shape and all. The correctly masked areas (shape) have higher error ratios than the mean for the full image (all). The results for KITTI are much more erratic than the rest. The large amount of missing GT data in this dataset reduced its value for this evaluation drastically. The majority of shape mask areas have higher error ratios than the same-sized box mask areas. Newer and more complex algorithms generally score lower errors and have lower absolute differences between shape and all errors. There are two distinct groupings: rand masks have comparable results as all masks while box is comparable to shape. This suggests that box annotations can often be used instead of the time-consuming shape annotations. This allows for the following conclusions based on the different maskings: algorithms have higher error rates at annotated areas and score even higher error rates if the annotation’s shape is preserved (shape vs. box). The effect of sampling patches of different sizes in each image is not prevalent (rand vs. box) and can be neglected.

基于鉴别到的危害使用masks的效果可以很明显的看出来。表6总结了采用形状masks的误差值和all masks的误差值的比值。正确掩膜的区域(shape)，其错误率比全图的均值要高。KITTI数据集的结果比其他数据集更不稳定。这个数据集中大量丢失的GT数据，使得对这个数据集的评估值急剧下降。主要的形状mask区域，比相同大小的框mask区域，其错误率要高。更新的、更复杂的算法，一般错误得分更低，其形状mask的错误和all masks的错误的绝对差异较小。有两个显著的分组：随机masks和all masks其结果可以比较，而框masks的结果与形状masks的结果可以比较。这说明，可以使用框标注，而不用使用更耗时的形状标注。这可以得到以下基于不同maskings的结论：在标注区域错误率更高的算法，如果标注的形状得到保留(shape vs. box)，则会有更高的错误率。在每幅图像中不同大小的采样块的效果相差不大(rand vs. box)，可以被忽略。

### 6.3 Statistical Significance

The intuitive grouping of the mask into groups (all, rand) and (shape, box) is now evaluated for its statistical significance. The null hypothesis H0 we will test is that the average performance evaluated at two different mask-types is not distinguishable. More specifically, that the differences between pairings of measurements (x_i, y_i) are symmetrically distributed around zero. This hypothesis should be valid between the grouped mask types and invalid between the other types.

将mask的进行分组(all, rand)和(shape, box)，现在对其统计显著性进行评估。我们要测试的null假设H0是，在两个不同的mask类型上的平均性能是无法区分的。更具体的，(x_i, y_i)的成对的度量的差别是在零附近对称分布的。这种假设在分组的mask类型内是有效的，在其他类型间是无效的。

To test the hypothesis, parametric and non-parametric tests can be used. Parametric tests (e.g. T-test) need to make assumptions about the underlying distribution. Such assumptions would be detrimental for our analysis as they could introduce bias. From the possible non-parametric tests we chose the Wilcoxon signed rank test (Wilcoxon 1945) because of its robustness and the possibility to evaluate over all three datasets in one joined analysis (see Demšar 2006) for a comparison between similar suited tests). The evaluation of all three datasets in one test statistic increases the sampling size and, thus, the test’s significance.

为测试这个假设，可以使用参数化的测试和非参数化的测试。参数化的测试（如，T-test）需要对分布进行假设。这种假设对我们的分析是有害的，因为这会带来偏差。非参数化的测试，我们选择Wilcoxon signed rank测试，因为这种测试非常稳健，可以对所有三个数据集在一个联合分析中进行评估（Demšar等进行了类似的评估，可以进行比较）。所有三个数据集在一个测试中的评估，增大了采样规模，因此也增加了测试的显著性。

The Wilcoxon signed rank test works by calculating the absolute difference for each pair of measurements from the two distributions and sorting those differences in ascending order. The rank in this order is now summed up using the original sign of each of the differences and the absolute value of this sum is used as the test statistic W. Ties in the ranking receive all the same average over the tying ranks. The number of differences not equal to zero is denoted with N_r.

Wilcoxon signed rank测试，要从两种分布中对每一对度量计算绝对误差，对这些差异以升序进行排列。这种顺序的排序用原始的正负符号叠加在一起，然后用其绝对值叠加在一起作为测试统计值W。非零的差异值的数量记为N_r。

Distributions with a symmetry around zero will yield a sum that has an expected value of zero and a variance of $var_W = N_r (N_r +1)(2N_r +1)/6$. For $N_r > 9$ the distribution of W approaches a normal distribution with $σ_W = \sqrt{var_W}$ and $z_W = W/σ_W$. These resulting probability values $z_W$ can be used as a measure for rejecting the null-hypothesis if $z_W$ is larger than $z_{Wc}$ based on the selected significance level.

在零附近对称分布的值，求和得到的期望值为0，方差为$var_W = N_r (N_r +1)(2N_r +1)/6$。对于$N_r > 9$，W的分布趋近于正态分布，$σ_W = \sqrt{var_W}$，$z_W = W/σ_W$。如果基于选定的显著性级别，得到的概率值$z_W$大于$z_{Wc}$，那么$z_W$可以用于可以用于拒绝null假设。

In our case we calculate the differences using average performance between two mask variants for each single test case (stereo image pair) from the datasets and then sort all differences by their absolute value. The resulting sum of the signed ranks is divided by σ_W for the corresponding N_r of that comparison yielding a single z value each. This test is performed for all relevant pairings of masks and for each algorithm, but we will combine the differences for all datasets. Finally we also calculate the overall z value for each pairing by evaluating the cumulation of all algorithm results. Table 7 shows the summarized results for all tests. The 100 samples of each mask generated for rand are used to calculate 100 times the value of z_W for each combination that contains rand. The table entry contains the arithmetic average of all 100 values. For this evaluation we keep the sign of the resulting test statistic to preserve the direction of each comparison. The decision whether to accept or reject the null hypothesis (distribution of results from different masks are the same) is based on the selected significance level. This percentage describes the probability of rejecting a true null hypothesis (type I error). We now apply a significance level of 5% to the data which translates to a z value of +/− 1.96. All null hypothesis with an absolute z_W value of higher than z_Wc = 1.96 can be rejected.

在我们的情况中，我们对两种mask情况，对数据集中每个单个测试案例（立体视觉对），使用平均性能计算其差异，然后用其绝对值对所有差异进行排序。得到的求和除以σ_W，这对应着N_r个对比，对每个产生一个z值。这个测试对所有相关的mask对、每个算法都进行，但我们会把所有数据集的差异进行汇总结合。最后，我们还会对每个成对数据计算总体的z值，对所有算法结果进行累加得到。表7给出了所有的测试的总结结果。每个mask生成的100个随机样本用于计算100次z_W值，当然这100次组合都包含rand。这个表格的条目包含所有100个值的代数平均。对于这个评估，我们保持得到的测试统计值的符号，以保留每个比较的方向。是否接受或拒绝null假设的决定（不同masks的结果的分布是相同的），是基于所选的显著性级别。这个百分比描述了拒绝一个真的null假设的概率（I型错误）。我们现在对数据应用5%的显著性级别，得到的z值为+/− 1.96。所有的null假设中，其绝对z_W值大于z_Wc = 1.96的，都可以被拒绝掉。

This results in the following observations: 这样就得到了下面的观察结果：

- (all, rand) is not significantly different, the null-hypothesis that both confirm to the same distribution can be accepted; (all,rand)之间差别不大，两个对相同的分布都确认的null假设，可以被接受；
- (shape, box) is significantly different, shape is more difficult than box; (shape,box)是非常不同的，shape比box更复杂；
- (all, shape) has the most significant difference, shape is much more difficult than all. The pairing all, box is also presenting the same level of significant differences. (shape, rand) and (box, rand) show slightly less significance but are still very definite: both shape and box are significantly more difficult than rand; (all,shape)之间的差异最大，shape比all难的很多。成对的(all,box)也是同样程度的差异显著程度的。(shape, rand)和(box, rand)之间的差异略低，但差异仍然很确定：shape和box都要比rand要难很多；
- The significance of the results varies widely between the different algorithms. Older and real-time algorithms tend to show the highest test statistics. SCAA results in the same trends as the remaining algorithms but stays always below the significance level of 5%. 结果的显著性对不同的算法变化很大。更老的和实时的算法通常测试统计值最高。SCAA与剩下的算法有相同的趋势，但其显著性级一直低于5%.

The evaluation paints a clear overall picture: areas identified by the CV experts as containing a visual hazard guided by the CV-HAZOP checklist are especially challenging for the selected CV algorithms. Focusing on these challenging areas is beneficial for robustness evaluations since it creates more meaningful test cases.

评估得到了很清晰的总体图景：CV专家鉴别出的包含视觉危害的区域（通过CV-HAZOP检查列表）对于选定的CV算法尤其具有挑战性。关注在这些有挑战的区域，对于稳健性评估是有好处的，因为这会得到更有意义的测试案例。

## 7 Conclusion

Many critical situations and relations have the potential to reduce the quality and functionality of CV systems. The creation of a comprehensive checklist containing these elements is a crucial component on the road towards systematic validation of CV algorithms. This paper presents the efforts of several experts from the fields of CV as well as risk and safety assessment to systematically create such a list. To the authors’ best knowledge, this is the first time that the risk analysis method HAZOP has been applied extensively to the field of computer vision.

很多关键的情况和关系，可能会降低CV系统的质量和功能。创建一个综合的检查列表，包含这些元素，对于CV算法的系统性validation，是非常关键的。本文给出了几位CV领域专家的努力，以及风险和安全性评估，以系统性的创建这样一个列表。据作者所知，这是风险分析方法HAZOP第一次广泛应用于CV领域。

The CV-HAZOP is performed by first introducing a generic CV model which is based upon information flow and transformation. The model partitions the system into multiple subsystems which are called locations. A set of parameters for each location is defined, that characterize the location’s individual influence on information. Additional special CV-relevant “guide words” are introduced that represent deviations of parameters with the potential to create hazards. The execution of the HAZOP was performed by a number of authors in parallel, assigning meanings to each combination of guide words and parameters to identify hazards. The individual findings were discussed and merged into one resulting CV-HAZOP list. A guideline for using the hazard list as a tool for evaluating and improving the quality and thoroughness of test datasets is provided.

CV-HAZOP的应用，首先要基于信息流和变换提出通用的CV模型。模型将系统分割成多个子系统，我们称为locations。在每个location定义一个参数集合，这就是这个location对信息的影响的特征。另外还提出了特殊的与CV有关的guide words，代表了参数的偏差，可能会形成危害。HAZOP的执行有几个作者同时进行，为每个guide words和参数的组合指定意义，以鉴别出危害。对鉴别出的危害进行讨论，并融合成一个CV-HAZOP列表。我们给出了使用这个危害列表的指南，以评估和改进测试数据集的质量和周全性。

The CV-HAZOP has produced a comprehensive checklist of hazards for the generic CV algorithm with over 900 unique entries. Each individual hazard is now referable by a unique hazard identifier (HID). It supports structured analysis of existing datasets and calculation of their hazard coverage in respect to the checklist. We present an example by applying the proposed guidelines to popular stereo vision datasets and finally evaluate the impact of identified hazards on stereo vision performance. The results show a clear correlation: identified hazards reduce output quality.

CV-HAZOP已经为通用CV算法生成了一个综合的危害检查列表，有超过900个条目。每个危害都可以用唯一的HID进行索引。对现有的数据集，支持结构性的分析，并可以针对检查列表计算其覆盖的危害。我们将提出的指南应用到流行的立体视觉数据集上，最终评估了鉴别出的危害对立体视觉的影响。结果表明了清晰的相关性：鉴别出的危害降低了输出的质量。

## 8 Outlook

The creation or combination and completion of test datasets using our checklist is the logical next step. We plan to guide the creation of a stereo vision test dataset with known coverage of hazards from our checklist. Another idea is the creation of test data that gradually increases the influence of specific hazards (e.g. amount of low contrast textures). This allows to find the point of failure and get an accurate estimation about the robustness of an algorithm when facing a specific hazard. The usage of our checklist can also be streamlined. Pre-filtered lists for common applications and domains provide specific lists without the need of manual adjustments. We are also investigating the automatic detection of hazards, i.e. algorithmic checks to determine if and where a hazard is present in a test image. This will reduce the manual task of categorizing test data and in the long run should lead to a fully automatic CV validation framework.

使用我们的检查列表，创建、组合并补全测试数据集，是下一步的工作。我们计划创建立体视觉测试数据集的时候，用已知的我们的检查列表里的危害进行指引创建。另一个想法是，创建测试数据，逐渐增加特定危害的影响（如，低对比度纹理的数量）。这可以找到失败的点，当面对特定的危害时，得到算法稳健性的准确估计。我们的检查列表的作用也可以成为流线型的。对常见的应用和领域，预滤除的列表，给出了具体的列表，不需要进行手工调整。我们还在研究危害的自动检测，即，用算法来检测确定在测试图像中是否存在危害，在哪里存在危害。这会减少测试数据归类的手工任务量，长期看来会得到全自动的CV validation框架。

Our HAZOP checklist is not considered final. It will be updated to include lessons learned during evaluations and testing or even after tested systems are put into operation. By sharing this information with the community over our public HAZOP database we hope to increase quality and reduce effort in CV robustness evaluation. At this stage, the CV-HAZOP becomes a structured and accessible reference hub for sharing experiences with CV algorithm development, usage, and maintenance.

我们的HAZOP检查列表并不是最终版。会更新，以将评估和测试过程中得到的经验也包括进去，甚至是在系统测试过后投入使用后的经验。通过与团体共享我们的公开HAZOP数据集信息，我们希望在CV算法稳健性评估中提高质量，降低工作量。在这个阶段，CV-HAZOP会成为一个结构化的，可访问的参考hub，在CV算法的开发、使用和维护中共享经验。