# BOOM-Explorer: RISC-V BOOM Microarchitecture Design Space Exploration Framework

Chen Bai, et. al. @ The Chinese University of Hong Kong & Tsinghua University

## 0. Abstract

The microarchitecture design of a processor has been increasingly difficult due to the large design space and time-consuming verification flow. Previously, researchers rely on prior knowledge and cycle-accurate simulators to analyze the performance of different microarchitecture designs but lack sufficient discussions on methodologies to strike a good balance between power and performance. This work proposes an automatic framework to explore microarchitecture designs of the RISC-V Berkeley Out-of-Order Machine (BOOM), termed as BOOM-Explorer, achieving a good trade-off on power and performance. Firstly, the framework utilizes an advanced microarchitecture-aware active learning (MicroAL) algorithm to generate a diverse and representative initial design set. Secondly, a Gaussian process model with deep kernel learning functions (DKL-GP) is built to characterize the design space. Thirdly, correlated multi-objective Bayesian optimization is leveraged to explore Pareto-optimal designs. Experimental results show that BOOM-Explorer can search for designs that dominate previous arts and designs developed by senior engineers in terms of power and performance within a much shorter time.

处理器的微架构设计非常困难，由于设计空间非常大，验证流程非常耗时。之前，研究者依赖先验知识和cycle-accurate模拟器来分析不同微架构设计的性能，但缺少对方法论的充分讨论，在性能和功耗间取得很好的平衡。本文提出了一个自动框架，探索RISC-V BOOM微架构设计，称为BOOM-Explorer，在功耗和性能之间获得了很好的折中。首先，框架利用了一个感知微架构的高级主动学习MicroAL算法，来生成一个多样的有代表性的初始设计集合。第二，构建了一个带有深度核学习函数的高斯过程模型(DKL-GP)，来表述设计空间的特征。第三，利用相关的多目标贝叶斯优化，来探索Pareto最优设计。试验结果表明，BOOM-Explorer可以搜索到的设计，在功耗性能平衡上，超过了之前最好的结果，和高级工程师开发得到的设计，而且耗时短的多。

## 1. Introduction

Recently, RISC-V, an open-source instruction set architecture (ISA) gains much attention and also receives strong support from academia and industry. Berkeley Out-of-Order Machine (BOOM) [1], [2], a RISC-V design fully in compliance with RV64GC instructions, is competitive in power and performance against low-power, embedded out-of-order cores in academia. By adopting Chisel hardware construction language [3], BOOM can be parametric, providing great opportunities to explore a series of microarchitecture designs that have a better balance on power and performance for different purposes of use.

最近RISC-V开源ISA获得了很多关注，也受到了学术界和工业界的很强支持。BOOM是一个完全按照RV64GC指令的RISC-V设计，与学术界的低功耗，嵌入式乱序核，在性能和功耗上很有竞争力。采用了Chisel语言，BOOM是参数化的，为探索一系列微架构设计提供了极好的机会，在不同的用处上有更好的功耗和性能平衡。

Microarchitecture defines the implementation of an ISA in a processor. Due to different organizations and combinations of components inside a processor, microarchitecture designs under a specific technology process can affect power dissipation, performance, die area, etc. of a core [4], [5]. Finding a good microarchitecture that can accommodate a good balance between power and performance is a notorious problem because of two restrictions. On the one hand, the design space is extremely large and the size of it can be exponential with more components to be considered, e.g., special queues, buffers, branch predictors, vector execution unit, external co-processors, etc. Thus, we cannot traverse and evaluate each microarchitecture to retrieve the best one. On the other hand, it costs a lot of time to acquire metrics, e.g., power, performance, etc. when we verify one microarchitecture with diverse benchmarks.

微架构定义了一个ISA在处理器中的实现。由于处理器中对很多组成部分有不同的组织和组合，在特定工艺下的微架构设计会影响一个核的功耗，性能，die面积等。找到一个好的微架构，在功耗和性能之间达到好的平衡，是非常困难的问题，因为有两个限制。第一，设计空间非常大，考虑的组成部分越多，其大小会以指数级增长，如特殊队列，buffers，分支预测器，向量执行单元，外部协处理器，等。因此，我们不能探索评估每个微架构，来获得最佳的那个。另一方面，当我们对一个微架构用不同的基准测试进行验证，获得度量结果会耗费很长的时间，如，功耗，性能，等。

In industry, the traditional solution is based on prior engineering experience from computer architects. However, it lacks scalability for newly emerged processors. In academia, to overcome these two obstacles, researchers proposed various arts, which can be categorized as two kinds of methodologies. First, in view of the difficulty in constructing an analytical model, researchers can otherwise characterize a microarchitecture design space with fewer samples as much as possible by leveraging statistical sampling and predictive black-box models. Li et al. [6] proposed AdaBoost Learning with novel sampling algorithms to explore the design space. Second, to search for more designs within a limited time budget, researchers often rely on coarse-grained simulation infrastructure rather than a register-transfer level (RTL) verification flow to accelerate the process [7]–[10]. Moreover, by decreasing redundant overhead, the simulation can be further speed up [11]–[14].

在工业中，传统的解决方案是基于计算机架构师的先验工程经验。但是，对新出现的处理器缺少可扩展性。在学术界，为克服这两个障碍，研究者提出各种arts，可以归类于两种方法。第一，构建一个解析模型有很大困难，研究者用尽可能少的样本来描述一个微架构设计空间的特征，利用统计采样和预测性黑盒模型。Li等[6]提出带有新的采样算法的AdaBoost学习来探索设计空间。第二，为用有限的时间预算来搜索更多设计，研究者通常依赖于粗糙粒度的仿真基础设施，而不是利用RTL验证流来加速这个过程。而且，通过降低冗余的耗费，这个仿真可以进一步得到加速。

Unfortunately, both of these academic solutions contain several limitations. In the first place, despite the fact that statistical analysis performs well when highly reliable models can be constructed, it fails to embed prior knowledge on microarchitectures to further improve design space exploration. For another, to accelerate the simulation, coarse-grained simulation infrastructure is used widely. Nevertheless, most of them lose sufficient accuracy, especially for distinct processors. The low quality of results is generated often due to the misalignment between simulation and real running behaviors of processors. More importantly, because it is difficult to model the power dissipation of modern processors at the architecture level [15], some infrastructure cannot provide power value, e.g., [8], [10]. In general, because of the afore-mentioned limitations, academia lacks sufficient discussions on methodologies that can explore microarchitecture designs achieving a good trade-off between power and performance.

不幸的是，这两种学术解决方法包含几种限制。第一，当可以构建非常可靠的模型时，统计分析的效果会很好，尽管如此，其不能将微架构的先验知识纳入其中，进一步改进设计空间探索。另一个，为加速仿真，粗糙粒度的仿真基础设施得到了广泛利用。尽管如此，多数都损失了足够的准确率，尤其对那些独特的处理器。结果的低质量，通常是由于仿真和处理器真实运行行为的不匹配。更重要的是，由于很难在架构层次对现代处理的功耗进行建模，一些基础设施不能提供功耗值，如[8,10]。总体上，由于之前提到的局限，对于能够很好的探索功耗和性能之间很好的折中的微架构探索的方法，学术界缺少足够的讨论。

In this paper, following the first strategy, we propose BOOM-Explorer address these issues. In BOOM-Explorer, without sacrificing the accuracy of a predictive model, we embed prior knowledge of BOOM to form a microarchitecture-aware active learning (MicroAL) algorithm based on transductive experimental design [16] by utilizing BOOM RTL samples among the entire design space as few as possible. Secondly, a novel Gaussian process model with deep kernel learning functions (DKL-GP) initialized through MicroAL, is proposed to characterize the features of different microarchitectures. The design space is then explored via correlated multi-objective Bayesian optimization flow [17] based on DKL-GP. Our framework can not only take advantage of fewer microarchitecture designs as much as possible but also helps us to find superior designs that have a better balance between power and performance.

本文中，按照第一种策略，我们提出了BOOM-Explorer，处理这个问题。在BOOM-Explorer中，不需要牺牲预测模型的准确率，我们将BOOM的先验知识纳入其中，以形成一个感知到微架构的主动学习算法(MicroAL)，这是基于transductive实现设计的，利用了整个设计空间中尽可能少的BOOM RTL样本进行的。第二，一个新的带有深度核学习函数的高斯过程模型(DKL-GP)，通过MicroAL进行初始化，提出用于描述不同微架构的特征。设计空间的探索，是通过相关的基于DKL-GP的多目标贝叶斯优化流进行的。我们的框架利用尽可能少的微架构设计，而且可以找到更好的设计，在功耗和性能之间有更好的均衡。

Our contributions are summarized as follows: 我们的贡献总结如下：

• A microarchitecture-aware active learning methodology based on transductive experimental design is introduced for the first time to attain the most representative designs from an enormous RISC-V BOOM design space. 第一次提出了一种基于transductive试验设计的感知微架构的主动学习方法，从巨量的RISC-V BOOM设计空间中，得到最有代表性的设计。

• A novel Gaussian process model with deep kernel learning and correlated multi-objective Bayesian optimization are leveraged to characterize the microarchitecture design space. With the help of DKL-GP, Pareto optimality is explored between power and performance. 利用一种带有深度核学习的新型高斯过程，和相关的多目标贝叶斯优化，来描述微架构设计空间的特征。有了DKL-GP的帮助，在功耗和性能之间探索Pareto最优性。

• We verify our framework with BOOM under advanced 7-nm technology. The experimental results demonstrate the outstanding performance of BOOM-Explorer on various BOOM microarchitectures. 我们在7-nm工艺下，用BOOM验证了我们的框架。试验结果表明，BOOM-Explorer在各种BOOM微架构上都有很好的性能。

The remainder of this paper is organized as follows. Section II introduces the RISC-V BOOM core and the problem formulation. Section III provides detailed explanations on the framework. Section IV conducts several experiments on BOOM core to confirm the outstanding performance of the proposed framework. Finally, Section V concludes this paper.

本文的余下部分组织如下。第二部分介绍了RISC-V BOOM核，和问题的表述。第三部分给出了框架的详细解释。第四部分在BOOM核上进行了几个试验，以确认了提出的框架有优异的性能。最后，第五部分总结了本文。

## 2. Preliminaries

### 2.1. RISC-V BOOM Core

BOOM is an open-source superscalar out-of-order RISC-V processor in academia and it is proved to be industry-competitive in low-power, embedded application scenarios [1], [2].

BOOM是一个学术上开源的超标量乱序RISC-V处理器，在低功耗嵌入式应用场景中，是有工业竞争力的。

Fig. 1 demonstrates the organization of BOOM. Consist of four main parametric modules, i.e., FrontEnd, IDU, EU, and LSU, BOOM can execute benchmarks in distinct behaviors via choosing different candidate values for each component inside these modules. FrontEnd fetches instructions from L2 Cache, packs these instructions as a sequence of fetch packages, and sends them to IDU. IDU decodes instructions as micro-ops and dispatches these micro-ops w.r.t. their categories to issue queues in EU, the latter of which, triggered by corresponding micro-ops and related logics, is responsible for manipulating operands in an out-of-order manner. Finally, some memory-related operations, i.e., loading data and storing data, interact with LSU after EU calculates the results. In addition, BOOM also integrates branch predictors, floating-point execution units, vector execution units, etc.

图1展示了BOOM的组织。包含4个主要的参数化模块，即，前端，IDU，EU和LSU，BOOM执行基准测试的行为非常不同，通过对这些模块中的每个组织部分选择不同的候选值而得到。前端从L2 Cache中取指，将这些指令打包为取到的包的序列，将其送往IDU。IDU将指令解码为微操作，将这些微操作按照其类别发送到EU单元中的发射队列，EU的后半部分，受到对应的微操作和相关逻辑的触发，负责以乱序的方式操作操作数。最后，一些与内存相关的运算，即载入数据，储存数据，在EU计算得到结果后，与LSU进行互动。另外，BOOM还集成了分支预测器，浮点执行单元，向量执行单元，等。

Thanks to the parameterized modules provided by BOOM, various BOOM microarchitectures can be acquired by configuring the core with different parameters. Thus, divergent trade-offs between power dissipation and performance to meet various design requirements can be achieved, e.g., low-power, and embedded applications. However, a satisfying microarchitecture design is non-trivial to be found.

多亏了BOOM提供的参数化模块，各种BOOM微架构可以通过用不同的参数配置core得到。因此，功耗和性能之间不同的折中，可以满足不同的设计需求，如，低功耗，和嵌入式应用。但是，要找到一个令人满意的微架构设计，绝不是容易的事。

Across all parametric modules, a microarchitecture design space of BOOM is constructed and shown in TABLE I. Each row of TABLE I defines structures of a component inside the module. RasEntry and BranchCount in FrontEnd are considered since they have great impacts on the behaviors of branch predictions, and thus incur different power and performance. Because caches, e.g., D-Cache in LSU, often runs at a lower frequency compared to other modules, the component might be hotspots when many memory-related requests occur in the instructions pipeline. A suitable structure of D-Cache can alleviate the burden, therefore different organizations of D-Cache (i.e., associativity, block width, TLB size, etc.) are also included for exploration. Besides, I-Cache is also considered in the design space.

用所有参数化模块，可以构建BOOM的微架构设计空间，如表1所示。表1中的每列，定义了模块中一个组成部分的结构。Frontend中的RasEntry和BranchCount要进行考虑，因为对分支预测的行为有很大的影响，因此会带来不同的功耗和性能。缓存，如LSU中的D-Cache，与其他模块相比，通常用更低的频率运行，因此，如果在指令流水线中发生了很多与内存相关的请求，这个组成部分就可能成为热点。合适结构的D-Cache，可以缓解这种压力，因此D-Cache的不同组织方式（即，关联度，block宽度，TLB大小，等），也纳入进行探索。此外，I-Cache也在设计空间中进行了探索。

Different BOOM microarchitecture designs can be constructed with various combinations of candidate values. However, some combinations do not observe constraints of BOOM design specifications as shown in TABLE II. Thus they are illegal and cannot be compiled to Verilog. For example, each entry of the reorder buffer traces status of every in-flight but decoded instruction in the pipeline. If a microarchitecture does not obey rule 2, reorder buffer may not reserve enough entries for each decoded instruction or may contain redundant entries that cannot be fully used at all. The last three rules in TABLE II are added to simplify the design space. They require the same number of entries or registers in respective components and their additions will not affect the performance of BOOM-Explorer. After we prune the design space w.r.t. rules in TABLE II, the size of the legal microarchitecture design space is approximately 1.6 × 10^8.

不同的BOOM微架构设计，可以通过候选值的不同组合得到。但是，一些组合会不符合BOOM设计指标的一些约束，如表2所示。因此这些组合是非法的，不能编译成Verilog。比如，ROB的每个条目，跟踪的是流水线中每个在飞行中，但已经解码过的指令的状态。如果一个微架构不遵守规则2，ROB可能不会为每个解码过的指令保留足够的条目，或可能包含冗余的条目，不能被完全使用。表2的最后3条指令，是用来简化设计空间的。在各种组成部分中，需要相同数量的条目或寄存器，这些规则的加入不会影响BOOM-Explorer的性能。在我们用表2的规则修剪过设计空间后，合法的微架构设计空间的大小，大约是1.6 × 10^8。

### 2.2. Problem Formulation

**Definition 1** (Microarchitecture Design). Microarchitecture design is to define a combination of candidate values given in TABLE I. A microarchitecture design is legal if it satisfies all constraints as referred to in TABLE II. Every legal microarchitecture design to be determined is encoded as a feature vector among the entire design space D. The feature vector is denoted as x. For convenience, microarchitecture and microarchitecture design in the following sections are the same.

定义1. 微架构设计。微架构设计是，对表1中给出的候选值定义一个组合。如果一个微架构设计满足表2中的所有约束，那么就是合法的。每个要确定的合法的微架构设计，可以用整个设计空间D中的一个特征向量来进行编码定义。特征向量表示为x。如下章节中的微架构和微架构设计是一样的。

**Definition 2** (Power). The power is to be defined as the summation of dynamic power dissipation, short-circuit power dissipation, and leakage power dissipation.

定义2.功耗。功耗定义为动态功耗，短路功耗和泄露功耗的和。

**Definition 3** (Clock Cycle). The clock cycle is to be defined as the clock cycles consumed when a BOOM microarchitecture design runs a specific benchmark.

定义3.时钟周期。时钟周期定义为，BOOM微架构设计运行特定的基准测试时，消耗的时钟周期数量。

Provided with the same benchmark, power and clock cycle are a pair of trade-off metrics since the lower cycles are, the more power will be dissipated when a design integrates more hardware resources to accelerate instructions execution. Together, They reflect whether a microarchitecture design is good or not. Power and clock cycle are denoted as y.

给定相同的基准测试，功耗和时钟周期是一对折中的度量，因为时钟周期越少，散发的能量就越多，设计要集成更多的硬件资源，来加速指令执行。它们一起能反应一个微架构设计是好还是不好。功耗和时钟周期表示为y。

**Definition 4** (Pareto Optimality). For a n-dimensional minimization problem, an objective vector f(x) is said to be dominated by f(x') if

定义4.Pareto最优。对于一个n维最小化问题，目标向量f(x)由f(x') dominate，如果

$$∀ i ∈ [1, n], f_i(x) ≤ f_i(x'); ∃j ∈ [1, n], f_j (x) < f_j (x')$$ (1)

In this way, we denote x' >= x. In the entire design space, a set of designs that are not dominated by any other is called the Pareto-optimal set and they form the Pareto optimality in this space.

这样，我们表示为x' >= x。在整个设计空间中，没有被其他设计所dominate的设计集合，成为Pareto最优集，它们形成了这个空间中的Pareto最优。

In this paper, our objective is to explore Pareto optimality defined in Definition 4 w.r.t. power and clock cycle for various BOOM microarchitectures. Due to the power and clock cycle are a pair of negatively correlated metrics, a microarchitecture belonged to the Pareto-optimal set cannot improve one metric without sacrificing another metric. To guarantee high quality of results, rather than use coarse-grained simulation infrastructure introduced in Section I, we evaluate power and performance using commercial electronic automation (EDA) tools and they are referred to as the VLSI flow. Based on the above definitions, our problem can be formulated.

本文中，我们的目标是，对各种BOOM微架构，对功耗和时钟周期，探索定义中的Pareto最优性。由于功耗和时钟周期是一对负相关的度量，属于Pareto最优集合的微架构，要改进一个度量，就必须牺牲另外一个。为保证结果的高质量，而不是使用第1部分中介绍的粗粒度的仿真基础架构，我们使用商用EDA工具来评估功耗和性能。基于上述定义，可以表述我们的问题。

**Problem 1** (BOOM Microarchitecture Design Space Exploration). Given a search space D, each microarchitecture design inside D is regarded as a feature vector x. Power and clock cycle form the power-performance space Y. Through VLSI flow, the power and cycles y ∈ Y can be obtained according to x. BOOM microarchitecture design space exploration is to be defined as to find a series of features X that form the Pareto optimality among the corresponding Y ⊂ Y. Hence, Y = {y|y' < y, ∀y' ∈ Y}, X = {x|f(x) ∈ Y , ∀x ∈ D}.

问题1 BOOM微架构设计空间探索。给定一个搜索空间D，D中的每个微架构设计称为一个特征向量x。功耗和时钟周期形成了功耗-性能空间Y。通过VLSI流，功耗和时钟周期y ∈ Y可以根据x得到。BOOM微架构设计空间探索定义为，找到一系列特征X，形成Pareto最优性，对应到Y ⊂ Y中。因此，Y = {y|y' < y, ∀y' ∈ Y}, X = {x|f(x) ∈ Y , ∀x ∈ D}。

## 3. BOOM-Explorer

### 3.1. Overview of BOOM-Explorer

Fig. 2 shows an overview of BOOM-Explorer. Firstly, the active learning algorithm MicroAL is adopted to sample a set of initial microarchitectures from the large design space. In this step, domain-specific knowledge is used as the prior information to guide the sampling of the initial designs. Then, a Gaussian process model with deep kernel learning functions (DKL-GP) is built on the initial set. To explore the optimal microarchitecture, the multi-objective Bayesian optimization algorithm is used, with the Expected Improvement of Pareto Hypervolume as the acquisition function, and the DKL-GP model as the surrogate model. During this process, BOOM-Explorer interacts with the VLSI flow to get the accurate performance and power values of designs according to different benchmarks. Finally, The outputs of BOOM-Explorer are the set of explored microarchitectures in the iterative optimization process, and the Pareto optimality is gained from the set.

图2展示了BOOM-Explorer的概览。首先，采用主动学习算法MicroAL，从很大的设计空间中采样初始微架构集合。在这个步骤中，使用领域特定知识作为先验知识，来引导初始设计的采样。然后，一个带有深度核学习函数的高斯过程模型(DKL-GP)，在初始集合的基础上构建。为探索最优微架构，使用多目标贝叶斯最优化算法，用Expected Improvement of Pareto Hypervolume作为采集函数，DKL-GP模型作为代理模型。在这个过程中，BOOM-Explorer与VLSI流进行互动，根据不同的基准测试得到设计的准确性能和功耗值。最后，BOOM-Explorer的输出是在迭代优化过程中的探索的微架构集合，从集合中得到了Pareto最优性。

### 3.2. Microarchitecture-aware Active Learning Algorithm

Due to the time-consuming VLSI flow, to save time, only a limited number of designs will be synthesized practically to obtain power and performance. To guarantee that adequate information is covered in the data set, two principles are considered during the initialization. First, feature vectors should cover the entire design space uniformly. Second, their diversity should fully represent the characteristics of the design space. Within a limited time budget, only push the most representative microarchitecture to VLSI flow can we alleviate the burden to get power and performance.

由于VLSI流非常耗时，只能对有限数量的设计进行综合，以通过实践得到功耗和性能。为确保数据集中覆盖了足够的信息，在初始化的时候考虑了两个原则。第一，特征向量应当均匀的覆盖整个设计空间。第二，它们的多样性应当完全表征设计空间的特征。在有限的时间预算内，只推动最有代表性的微架构到VLSI流中，可以缓解压力，得到功耗和性能。

A naive solution is to sample microarchitectures randomly. In literature, most previous works [18], [19] choose this simple method directly for convenience. In addition, by appraising the importance of each feature vector with suitable distance measurement, greedy sampling [20] can be facilitated to select representative microarchitecture designs.

一个简单的解决方法是对微架构进行随机采样。在文献中，多数之前的工作[18,19]为方便起见，直接选择这种简单方法。另外，为了用合适的距离度量来强调每个特征向量的重要性，可以用贪婪采样[20]来选择有代表性的微架构设计。

To further improve the sampling, orthogonal design [6], [21] is also utilized to pick up dissimilar microarchitectures that are distributed orthogonally across the design space. Nevertheless, the aforementioned methodologies are failed to capture the key components of different microarchitectures that bring great impacts to the trade-off in the power-performance space.

为进一步改进采样，还利用了正交设计[6,21]，来选择不相似的微架构，它们在设计空间中的分布是正交的。尽管如此，之前提到的方法不能捕获不能微架构中的关键组成部分，这些部分为功耗性能空间的折中带来的是很大的影响。

Recently, witnessing the great performance improvement attained by transductive experimental design (TED) in the design space exploration of high-level synthesis [22], [23], compilation and deployment of deep neural networks [24], and etc., we introduce this method into the exploration of microarchitectures for the first time.

最近，TED在高层次综合、编译的设计空间探索，DNN的部署得到了非常好的性能，我们第一次将这种方法引入到微架构的探索中。

TED tends to choose microarchitecture that can spread across the feature space to retain most of the information among the whole design space [16]. A pool of representative feature vectors can be acquired with high mutual divergences, by iteratively maximizing the trace of the distance matrix constructed on a newly sampled design and unsampled ones. Algorithm 1 shows the backbone of TED, where f represents the distance function used in computing the distance matrix K. Note that any suitable distance functions can be applied without restrictions.

TED选择的微架构，要在特征空间中都有分布，以得到整个设计空间的多数信息。一些有代表性的特征向量的获得，是用之间的高度差异性，在新采样的设计和未采样的设计之间的距离矩阵，其矩阵迹进行迭代最大化而获得。算法1展示了TED的骨干结构，其中f表示距离函数，用于计算距离矩阵K。任意合适的距离函数都可以应用，不需要限制。

Unfortunately, TED cannot fully guarantee to generate a good initial data set owing to a lack of prior knowledge of microarchitecture designs. We are motivated to embed the domain knowledge to improve its performance.

不幸的是，TED不能完全保证生成生成一个很好的初始数据集，因为缺少微架构设计的初始知识。我们因此将领域知识嵌入进，以改进其性能。

DecodeWidth as referred to in TABLE I decides the maximal number of instructions to be decoded as corresponding micro-ops simultaneously. In consequence, it can affect the execution bandwidth of EU and LSU (i.e., instructions executed per clock cycle). Assigning a larger candidate value to DecodeWidth and allocating a balanced amount of hardware resources, on the one hand, can lead to greater performance improvement. On the other hand, power dissipation will also increase significantly. By clustering w.r.t. DecodeWidth, the power-performance space can be separated along the potential Pareto optimality, as shown in Fig. 3. Each cluster in Fig. 3 represents a group of microarchitectures with different candidate values for DecodeWidth in the power-performance space. The entire design space is discrete and non-smooth but nonetheless a large number of microarchitectures with the same DecodeWidth achieve similar power-performance characteristics within their sub-regions respectively. It inspires us that we can select microarchitectures on the possible sub-area from the initial design space, to better cover the entire design space at the same time improve the diversity of samples.

表1中的DecodeWidth决定了同时解码的最大数量的指令。结果是，这会影响EU和LSU的执行带宽（即，每时钟周期执行的指令数量）。对DecodeWidth指定一个更大的候选值，配置均衡数量的硬件资源，一方面可以带来更大的性能改进。另一方面，功耗也会显著增加。对DecodeWidth进行聚类，功耗性能空间可以沿着潜在的Pareto最优性进行分离，如图3所示。图3中的每个聚类都代表一组微架构，DecodeWidth候选值不同。整个设计空间是离散的，非平滑的，尽管如此，DecodeWidth值相同的大量微架构，会得到类似的功耗性能特性值，分别形成一个子区域。这启发了我们，可以从初始设计空间中可能的子区域选择微架构，以更好的覆盖整个设计空间，同时改进样本的多样性。

Inside each cluster, Algorithm 1 can be applied instead of choosing the centroid to enlarge the initial data set. The clustering w.r.t. DecodeWidth, together with TED, forms MicroAL and the pseudo code is detailed in Algorithm 2.

在每个聚类中，可以用算法1来选择重心，以增大初始数据集。对DecodeWidth的聚类，与TED一起，形成了MicroAL，伪代码如算法2所示。

First, we cluster the entire design space according to Φ, which is the distance function with a higher penalty along the dimension of DecodeWidth. One possible alternative can be Φ = (x_i − c_j)^T Λ(x_i − c_j), with i ∈ {1, · · · , |U|} and j ∈ {1, · · · , k}, where Λ is a pre-defined diagonal weight matrix. Next, we apply TED for each cluster to sample the most representative feature vectors, i.e., line 9 in Algorithm 2. Finally, containing all of the sampled microarchitectures, the initial data set is formed.

首先，我们根据Φ对整个设计空间进行聚类，Φ是距离函数，沿着DecodeWidth的维度有着更高的惩罚，一个可能的替代品是Φ = (x_i − c_j)^T Λ(x_i − c_j), 其中i ∈ {1, · · · , |U|}, j ∈ {1, · · · , k}，其中Λ是预定义的对角权重矩阵。下一步，我们对每个聚类应用TED，来采样最具有代表性的特征向量，即，算法2中的第9行。最后，包含了所有采样的微架构，就形成了初始数据集。

### 3.3. Gaussian Process with Deep Kernel Learning

Given the initial data set, it is hard to build a reliable model to fully capture the characteristics of the design space yet.

给定初始的数据集，很难构建一个可靠的模型，以完整的捕获设计空间的特征。

However, thanks to robustness and non-parametric approximation features reside in Gaussian process (GP) models, they have been applied in various domains [24]–[26]. In view of the success, BOOM-Explorer adopts GP as well.

但是，多亏了高斯过程模型中的稳健性和非参数近似特征，这已经应用到了多个领域[24-26]。有了这些成功，BOOM-Explorer也采用了GP。

Assume that we have feature vectors X = {x1, x2, ...xn} and they index a set of corresponding power or clock cycles y = {y1, y2, ..., yn}. GP provides a prior over the value function f as f(x) ∼ GP(µ, k_θ), where µ is the mean value and the kernel function k is parameterized by θ. Then, Gaussian distributions can be constructed with any collection of value functions f according to Equation (2)

假设我们有了特征向量X = {x1, x2, ..., xn}，这对应了功耗或时钟周期的集合y = {y1, y2, ..., yn}。GP对价值函数f给出了一个先验，f(x) ∼ GP(µ, k_θ)，其中µ是均值，核函数k以θ为参数。然后，可以用任意集合的值函数f根据式(2)构建高斯分布

$$f = [f(x1), f(x2), ...f(xn)]^T ∼ N(µ, K_{XX|θ}),$$(2)

where $K_{XX|θ}$ is the intra-covariance matrix among all feature vectors and calculated via $[K_{XX|θ}]_{ij} = k_θ(x_i, x_j)$. A Gaussian noise N(f(x), σ^2_e) is necessary to model uncertainties of power or clock cycles generated by different microarchitecture designs. Thus, given a newly sampled feature vector x∗, the predictive joint distribution f∗ that depends on y can be calculated according to Equation (3)

其中$K_{XX|θ}$是内协方差矩阵，在所有的特征向量中，通过$[K_{XX|θ}]_{ij} = k_θ(x_i, x_j)$来计算。需要高斯噪声N(f(x), σ^2_e)来对功耗或时钟周期的不确定性进行建模，这些不确定性是由不同的微架构设计生成的。因此，给定一个新采样的特征向量x*，预测的联合分布f*可以根据式(3)来进行计算，这是依赖于y的

$$f*|y ~ N([µ, µ*]^T, [K_{XX|θ}+σ^2_e I, K_{Xx*|θ}; K_{x* X|θ}, k_{x* x*|θ}])$$(3)

By maximizing the marginal likelihood of GP, θ is optimized to sense the entire design space. Nevertheless, the performance of GP normally depends on the expressiveness and hyper-parameters of kernel functions kθ, e.g., radial bias functions, and etc. Therefore, a suitable kernel function is necessary to the performance of GP.

通过最大化GP的边缘概率，θ进行了优化，以感知整个设计空间。尽管如此，GP的性能一般依赖于核函数kθ的表达力和超参数，如半径偏置函数，等。因此，合适的核函数对于GP的性能是必须的。

In the recent years, deep neural networks (DNN) have shown great potential in various applications and tasks as the black-box model to extract useful features [27]–[29]. Thus, with the help of DNN as a meta-learner for kernel functions, we can relieve workloads in tuning hyper-parameters of kernel functions. By leveraging multi-layer non-linear transformations and weights to calibrate kernel functions with Equation (4) [30], deep kernel functions can provide better performance.

近年来，DNN在各种应用中展示出了很大的潜力，作为一个黑盒模型提取有用的特征。因此，DNN可以作为一个元学习者，学习核函数，我们可以减轻调整核函数的超参数的工作负担。通过利用多层非线性变换和权重，来与式(4)校准核函数，DNN可以提供更好的性能。

$$k_θ(x_i, x_j) → k_{w,θ}(ϕ(x_i, w), ϕ(x_j, w))$$(4)

ϕ in Equation (4) denotes non-linear transformation layers stacked by DNN and w denotes weights in DNN. Enhanced with the expressive power of DNN, DKL-GP is constructed and then plugged into Bayesian optimization as the surrogate model.

式(4)中的ϕ表示DNN累积的非线性变换层，W表示DNN中的权重。有了DNN的表示能力的强化，DKL-GP构建以后，然后插入到贝叶斯优化中，作为代理模型。

### 3.4. Correlated Multi-Objective Design Exploration

Notwithstanding DKL-GP can be used to evaluate a single object (i.e., power or clock cycles) well, to find the Pareto-optimal set still remains an issue, especially for such negatively correlated objectives.

尽管DKL-GP可以用于很好的评估单个目标（如，功耗或时钟周期），但找到Pareto最佳集仍然是一个问题，尤其是对于这样负相关的目标。

A traditional methodology usually integrates different acquisition functions to solve it. Lyu et al. [31] combines Expectation Improvement (EI), Probability Improvement (PI) and Confidence Bound (i.e., UCB and LCB) to form a multi-objective optimization framework in analog circuit design. It still leads to sub-optimal results except that we can select a good composite of acquisition functions for specific problems. To solve the problem more efficiently, we introduce Expected Improvement of Pareto Hypervolume (EIPV) [32] and demonstrate its usability to characterize the trade-off in the power-performance space of different microarchitecture designs.

一种传统的方法通常将不同的采集函数整合到一起进行解决。Lyu等[31]在模拟电路设计中，将Expectation Improvement(EI)，Probability Improvement(PI)和Confidence Bound（即，UCB和LCB）结合到一起，以形成一个多目标优化框架。这得到的仍然是次优的结果，除非我们对特定的问题选择了采集函数的很好的组合。为更高效的解决问题，我们提出了Expected Improvement of Pareto Hypervolume (EIPV)[32]，并证明了其可用性，描述了不同微架构设计的功耗性能空间的折中的特征。

In our problem, a better microarchitecture can not only run faster (i.e., it gets fewer average clock cycles among all benchmarks) but also dissipate less power. Given a reference point v_{ref} ∈ Y, Pareto hypervolume bounded above from v_{ref} is the Lebesgue measure of the space dominated by the Pareto optimality as shown in Fig. 4(a) [33]. The shaded area in orange, indicating Pareto hypervolume w.r.t. the current Pareto-optimal set P(Y) is calculated by Equation (5)

在我们的问题中，更好的微架构，不仅要运行的更快（在所有的基准测试中，得到更少的平均时钟周期数），而且还要耗散更少的能量。给定一个参考点v_{ref} ∈ Y，由v_{ref}限制住的Pareto hypervolume是图4(a)所示的Pareto最优性dominate的空间的Lebesgue度量。橙色的阴影区域，是对目前的Pareto最优集P(Y)的Pareto hypervolume，是由式(5)计算得到的

$$PVol_{v_{ref}} (P(y)) = \int_y 1[y >= v_{ref}] [1 - \prod_{y*∈P(Y)} 1[y* !>= y]] dy$$(5)

where 1(·) is the indicator function, which outputs 1 if its argument is true and 0 otherwise.

其中1(·)是指示函数，如果参数为真，则输出1，否则输出0。

v_{ref} is carefully chosen for convenience of calculation. Ideally, a feature vector x' that can increase the likelihood of DKL-GP maximally should be picked up from the design space D in every iteration. Thus, a better predictive Pareto-optimal set, enveloping the previous one by improving PVol_{v_{ref}}(P(Y)) is the direct solution according to Equation (6) where f : x → y ∈ Y is denoted as DKL-GP. Then the feature vector x∗ = arg max_{x'∈D} EIPV(x'|D) can be sampled as a new candidate for the predictive Pareto optimality.

v_{ref}是仔细选择的，以计算方便。理想来说，一个特征向量x'可以最大的增加DKL-GP的可能性，就应当在每一次迭代中从设计空间D中选择出来。因此，一个更好的预测性的Pareto最优集，包含了之前的一个，改进了PVol_{v_{ref}}(P(Y))，根据式(6)是直接的解，其中f : x → y ∈ Y表示为DKL-GP。然后特征向量x∗ = arg max_{x'∈D} EIPV(x'|D)可以采样为新的候选，以作为预测性的Pareto最优性。

$$EIPV(x'|D) = E_{p(f(x')|D)} [PVol_{v_{ref}} (P(Y) ∪ f(x')) - − PVol_{v_{ref}}(P(Y))]$$(6)

By decomposing the power-performance space as grid cells shown in Fig. 4, Equation (6) can be further simplified as Equation (7)

将图4中所示的网格单元的功耗性能空间结构，式(6)可以进一步简化为式(7)

$$EIPV(x'|D) = \sum_{C∈C_{nd}} \int_C PVol_{v_C}(y)p(y, |D)dy$$(7)

where C_nd denotes non-dominated cells. Region colored in green as referred to Fig. 4(b) shows the improvement of Pareto hypervolume. In Equation (7), p(y|D) is modeled as a multi-objective GP for power and clock cycles where the kernel function in Equation (4) parameterized by DNN can be Matern 5/2 kernel.

其中C_nd表示非dominated单元。图4(b)中绿色的区域，表明了Pareto hypervolume的改进。在式(7)中，p(y|D)对功耗和时钟周期建模为多目标GP，其中式(4)中的核函数可以是Matern 5/2核，由DNN参数化。

Equipped with all aforementioned methodologies, Algorithm 3 provides the end-to-end flow of BOOM-Explorer. We first leverage Algorithm 2 to sample representative microarchitectures (e.g., different branch prediction capability, cache organization, various structures of issue unit, etc.). DKL-GP is then built to characterize the design space. Finally, with Bayesian optimization, the Pareto-optimal set is explored via the maximization of EIPV.

用了上面所述的所有方法后，算法3给出了端到端的BOOM-Explorer流。我们首先利用算法2来采样代表性的微架构（如，不同的分支预测能力，缓存组织，不同发射单元的结构，等）。DKL-GP然后构建以描述设计空间的特征。最后，用贝叶斯优化，通过EIPV的最大化，探索了Pareto最优集。

## 4. Experiments

We conduct comprehensive experiments to evaluate the proposed BOOM-Explorer. Chipyard framework [34] is leveraged to compile various BOOM RTL designs. We utilize 7-nm ASAP7 PDK [35] for the VLSI flow. Cadence Genus 18.12-e012_1 is used to synthesize every sampled RTL design, and Synopsys VCS M-2017.03 is used to simulate the design running at 2GHz with different benchmarks. PrimeTime PX R-2020.09-SP1 is finally used to get power value for all benchmarks.

我们进行了综合的试验，评估提出的BOOM-Explorer。利用了Chipyard框架[34]来编译各种BOOM RTL设计。我们在VLSI流中使用7-nm ASAP7 PDK。用Cadence Genus 18.12-e012_1来综合每个采样的RTL设计，利用Synopsys VCS M-2017.03仿真设计，对不同的基准测试运行在2GHz上。最后利用PrimeTime PX R-2020.09-SP1来对所有基准测试得到能量值。

### 4.1. Benchmarks and Baselines

Since it is time-consuming to verify every sampled microarchitecture design online, we construct an offline data set. Consisting of 994 legal microarchitectures, the offline data set is sampled randomly and uniformly from the BOOM design space as referred to in TABLE I. Each design is fed to the VLSI flow to get power and clock cycles with high fidelity for all benchmarks and the corresponding time to conduct the flow is also recorded. The VLSI flow for each design takes approximately from 6 hours to more than 14 hours to finish. All of experiments are conducted on this data set. Several benchmarks are selected to test the performance of microarchitectures, i.e., median, mt-vvadd, whetstone, and mm from commonly used CPU benchmark suites. These four benchmarks are complete to all RISC-V instructions, e.g., instructions that transfer data between registers and memory, floating-point manipulations, multi-threading executions, vector instructions, etc. The average clock cycles and power on the four benchmarks are denoted as the performance and power value for each design respectively.

由于在线验证每个采样的微架构设计是非常耗时的，我们构建了一个离线数据集。这个数据集由994个合法的微架构组成，从BOOM设计空间中进行随机均匀采样，如表1所示。每个设计送到VLSI流中，对所有基准测试得到功耗和时钟周期数，进行这个流的对应时间也被记录下来。对每个设计的VLSI流，耗费大约6小时到14小时才结束。所有试验都在这个数据集中进行。选择了几种基准测试来测试微架构的性能，即，median, mt-vvadd, whetstone, mm，这些都是常用的CPU基准测试。这四个基准测试对所有RISC-V指令是完备的，即，在寄存器和内存之间传输数据的指令，浮点操作指令，多线程执行指令，向量指令，等。这四个基准测试的平均时钟周期数和功耗，分别表示了每个设计的性能和功耗值。

Several representative baselines are compared with BOOM-Explorer. The ANN-based method [18] (shorted as ASP-LOS’06), stacks ANN to predict the performance of designs, including a complicated chip multiprocessor. The regression-based method [19] (termed HPCA’07), leverages regression models with non-linear transformations to explore the power-performance Pareto curve on POWER4/POWER5 designs. The AdaBoost-RT-based method [6] (abbreviated as DAC’16), utilizes OA sampling and active learning-based AdaBoost regression tree models to explore microarchitectures w.r.t. their performance. The aforementioned arts are proved effective in their works of the exploration of microarchitectures respectively. Therefore, it is requisite to compare these methodologies with BOOM-Explorer. The HLS predictive model-based method [36] (named DAC’19), exploring the high-level synthesis design is also chosen as our baseline. Although the starting point is different, their method is proved to be robust and transferable. Moreover, we also compare BOOM-Explorer with traditional machine learning models, i.e., support vector regression (SVR), random forest, and XGBoost [37]. For fair comparisons, experimental settings of the baselines are the same as those mentioned in their papers. Simulated annealing is leveraged for traditional machine learning algorithms, e.g., SVR, Random Forest, and XGBoost.

几种代表性的基准与BOOM-Explorer进行了比较。基于ANN的方法（缩写为ASP-LOS'06），将ANN堆叠起来，预测设计的性能，包含了一个复杂的芯片多处理器。基于回归的方法[19]（缩写为HPCA'07），利用了非线性变换的回归模型来在POWER4/POWER5设计上探索功耗-性能Pareto曲线。基于AdaBoost-RT的方法[6]（缩写为DAC'16），利用OA采样和基于主动学习的AdaBoost回归树模型，来探索微架构的性能。之前的提到的arts，在其对微架构的探索中被证明是有效的。因此，将这些方法与BOOM-Explorer进行比较是必须的。HLS预测性基于模型的方法[36]（称为DAC'19），探索了高层次综合设计，也选作了我们的基准。虽然开始点是不同的，其方法也证明了是稳健的，可迁移的。而且，我们还比较了BOOM-Explorer和传统的机器学习方法，即，支持矢量机回归，随机森林，和XGBoost[37]。为进行公平比较，基准的试验设置与其文章中是一样的。对传统机器学习算法利用了模拟退火，如SVR，随机森林和XGBoost。

### 4.2 Experiments Settings

In the settings of BOOM-Explorer, DKL-GP is stacked with three hidden layers, each of which has 1000, 500, and 50 hidden neurons respectively, and it adopts ReLU as the non-linear transformation for deep kernel learning. The Adam optimizer [38] is used, with an initial learning rate equals to 0.001. DKL-GP is initialized with 5 microarchitectures sampled according to MicroAL and then BOOM-Explorer performs Bayesian exploration with 9 rounds sequentially. All experiments together with baselines are repeated 10 times and we report corresponding average results.

在BOOM-Explorer的设置中，DKL-GP用三个隐含层堆叠起来，每个分别有1000，500和50个隐藏神经元，采用了ReLU作为深度核学习的非线性变换。使用了Adam优化器，初始学习速率为0.001。DKL-GP用5个微架构初始化，根据MicroAL采样，BOOM-Explorer然后进行贝叶斯探索，顺序进行了9轮。所有试验与基准一起，重复10次，我们给出对应的平均值。

Average distance to reference set (ADRS) and overall running time (ORT) are two metrics for performance comparisons. ADRS, as shown in Equation (8), is widely used in design space exploration problems to measure how close a learned Pareto-optimal set to the real Pareto-optimal set of the design space.

对参考集的平均距离(ADRS)和总体运行时间(ORT)是性能比较的两种度量。ADRS如式(8)所示，在设计空间探索问题中广泛使用，以度量设计空间中学习到的Pareto最优集与真实的Pareto最优集的距离有多近。

$$ADRS(Γ, Ω) = \sum_{γ∈Γ} min_{} f_{ω∈Ω}(γ, ω) / |Γ|$$(8)

where f is the Euclidean distance function. Γ is the real Pareto-optimal set and Ω is the learned Pareto-optimal set. ORT measures the total time of algorithms including initialization and exploring.

其中f是欧式距离函数。Γ是真实的Pareto最优集，Ω是学习到的Pareto最优集。ORT度量的是算法的总时间，包括初始化和探索。

### 4.3. Results Analysis

Fig. 5 shows the learned Pareto-optimal sets obtained by the baselines and BOOM-Explorer. The results show that the Pareto-optimal set learned by BOOM-Explorer is much closer to the real Pareto-optimal set and thus outperforming baselines remarkably.

图5展示了基准和BOOM-Explorer学习的Pareto最优集。结果表明，BOOM-Explorer学习到的Pareto最优集，与真实的Pareto最优集是最近的，因此明显超过了基准的性能。

The normalized results of ADRS and ORT are listed in TABLE III. BOOM-Explorer outperforms ASPLOS’06, HPCA’07, DAC’16, and DAC’19 by 70%, 66%, 29%, and 64% in ADRS, respectively. Meanwhile, it accelerates the exploring by more than 88% compared with DAC’16. Since the prior knowledge of BOOM microarchitecture designs is embedded in our method, DKL-GP can outperform baselines by a large margin in ADRS and ORT. The effectiveness of the proposed MicroAL is demonstrated by conducting comparative experiments of BOOM-Explorer with random sampling instead of MicroAL. The corresponding results are listed as BOOM-Explorer w/o MicroAL. The results show that without MicroAL, the performance of BOOM-Explorer would be close to DAC’16.

ADRS和ORT的归一化的结果如表3所示。BOOM-Explorer超过了ASPLOS’06, HPCA’07, DAC’16, DAC’19，在ADRS上分别有70%, 66%, 29%, 64%。同时，探索过程也得到了加速，比DAC'16加速了88%。由于BOOM微架构设计的先验知识是嵌入到我们的方法中的，DKL-GP会超过基准很多。提出的MicroAL的有效性的证明，是将BOOM-Explorer与随机采样一起进行试验，然后进行比较而得到的。对应的结果列为BOOM-Explorer w/o MicroAL。结果表明，没有MicroAL，BOOM-Explorer的性能与DAC'16接近。

If a larger initialization set is sampled via MicroAL, BOOM-Explorer will be able to gain a better predictive Pareto-optimal set. Finally, we can achieve different designs to strike good balances between power and performance.

如果用MicroAL采样一个更大的初始化集合，BOOM-Explorer会得到更好的预测性Pareto最优集。最后，我们会得到不同的设计，在功耗的性能之间得到更好的平衡。

### 4.4. The Optimal BOOM Microarchitecture Design

Our Pareto design is chosen from the Pareto-optimal set found by BOOM-Explorer and it is compared with a two-wide BOOM developed by senior engineers [1].

我们的Pareto设计是用BOOM-Explorer找到的Pareto最优集，从中选择出的，与资深工程师设计的2宽度BOOM进行了比较。

The aforementioned two microarchitectures of BOOM are listed in TABLE IV. Indicated by “Design Parameters” in TABLE IV, our Pareto design has the same DecodeWidth compared with the two-wide BOOM. However, the Pareto design reduces hardware components on the branch predictor (i.e., RasEntry, BranchCount, etc.), entries of the reorder buffer, etc., but enlarges instructions issue width, LDQ, STQ, etc. Moreover, it has different cache organizations, e.g., different associate sets. Because LSU introduced in Section II-A tends to become a bottleneck, the Pareto design increases hardware resources for LDQ, STQ, and meanwhile increases associate sets and MSHR entries for D-Cache to overcome more data conflicts. Furthermore, the Pareto design reduces resources of RAS and BTB since there are not many branches or jump instructions in these benchmarks. Via reducing redundant hardware resources while increasing necessary components, our Pareto design achieves a better trade-off on power and performance.

之前提到的两个BOOM微架构如表4所示。在表4中的设计参数表示了两个设计，我们的Pareto设计的DecodeWidth与2宽度BOOM相同。但是，Pareto设计减少了一些组成部分的数量，包括分支预测器（即，RasEntry，BranchCount，等），ROB的条目，等，但加大了指令发射宽度，LDQ，STQ等。而且，缓存组织也不一样，即，不同的关联集。因为2.1中引入的LSU会成为一个瓶颈，Pareto设计增加了LDQ，STQ的硬件资源，同时增加了D-Cache的关联集和MSHR条目，以克服更多的数据冲突。而且，Pareto设计降低了RAS和BTB的资源，因为在基准测试中没有很多分支或跳转指令。通过降低冗余的硬件资源，同时增加必须的组成部分，我们的Pareto设计获得了更好的功耗和性能的折中。

To demonstrate the superiority of the Pareto design compared with the two-wide BOOM, both of them are evaluated on more benchmarks, and TABLE IV shows the average power and clock cycles of all these benchmarks. These benchmarks are chosen from different application scenarios, e.g., add-int, add-fp, etc. are from ISA basic instructions, iir, firdim, etc. are from DSP-oriented algorithms [39], compress, duff, etc. are from real-time computing applications [40], etc. Fig. 6 shows the comparison of power and performance between them. For all of these benchmarks, our Pareto design runs approximately 2.11% faster and at the same time dissipates 3.45% less power than the two-wide BOOM.

为证明Pareto设计与2宽度BOOM更优，这两者都在更多的基准测试中进行了评估，表4展示了在所有这些基准测试中的平均功耗和时钟周期。这些基准测试是从不同的应用场景中选择出来的，如，add-int, add-fp等是ISA的基础指令，iir, firdim等，是DSP向的算法，compress, duff等是实时计算应用。图6给出了它们之间的性能和功耗比较。对所有这些基准测试，我们的Pareto设计，与2宽度BOOM相比，运行速度快了2.11%，同时功耗少了3.45%。

## 5. Conclusions

In this paper, BOOM-Explorer is proposed to search for Pareto optimality among the microarchitecture design space within a short time. To the best of our knowledge, this is the first work introducing automatic design space exploration solution to the RISC-V community. We expect to see a lot of researches in our community to further improve microarchitecture design space explorations of processors.

本文中提出了BOOM-Explorer，用很短的时间在微架构设计空间中探索了Pareto最优性。据我们所知，这是对RISC-V第一个提出了自动设计空间探索的工作。我们期待看到更多的研究者进一步改进处理器的微架构设计空间探索。