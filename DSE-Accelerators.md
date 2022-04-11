# Design Space Exploration of Heterogeneous-Accelerator with Hyperparameter Optimization

Thanh Cong, Francois Charot @ INRIA, France

## 0. Abstract

Modern SoC systems consist of general-purpose processor cores augmented with large numbers of specialized accelerators. Building such systems requires a design flow allowing the design space to be explored at the system level with an appropriate strategy. In this paper, we describe a methodology allowing to explore the design space of power-performance heterogeneous SoCs by combining an architecture simulator (gem5-Aladdin) and a hyperparameter optimization method (Hyperopt). This methodology allows different types of parallelism with loop unrolling strategies and memory coherency interfaces to be swept. The flow has been applied to a convolutional neural network algorithm. We show that the most energy efficient architecture achieves a 2x to 4x improvement in energy-delay-product compared to an architecture without parallelism. Furthermore, the obtained solution is more efficient than commonly implemented architectures (Systolic, 2D-mapping, and Tiling). We also applied the methodology to find the optimal architecture including its coherency interface for a complex SoC made up of six accelerated-workloads. We show that a hybrid interface appears to be the most efficient; it reaches 22% and 12% improvement in energy-delay-product compared to just only using non-coherent and only LLC-coherent models, respectively.

现代SoC系统是由通用目标处理器核，和大量专用加速器组成的。构建这样的系统，需要一个设计流程，在系统级用合适的策略来探索设计空间。本文中，我们描述了一种方法论，将架构模拟器gem5-Aladdin和超参数优化方法Hyperopt结合起来，探索功耗性能异质SoCs的设计空间。这种方法论使不同类型的并行可以共同进行，循环展开策略和内存一致性接口。这个流程已经应用到了一个卷积神经网络算法。我们展示了功耗效率最高的架构，比没有并行性的架构，功耗延迟乘积有2x到4x的改进。而且，得到的解比通常实现的架构(Systolic, 2D-mapping, and Tiling)要更高效。我们还应用这种方法论，对一个由6个加速的workloads组成的复杂的SoC，来找到最优架构，包含其一致性接口。我们展示了，混合接口似乎是最高效的；与只使用非一致性模型和LLC一致性模型的系统相比，其功耗延迟乘积分别有22%和12%的改进。

**Keywords** Heterogeneous architecture design, System-on-chip, Hardware accelerators, Hyperparmeter optimization, Simulation

## 1. Introduction

The energy efficiency gap between application-specific integrated circuits (ASICs) and general-purpose processors motivates the design of heterogeneous-accelerator system-on-chip (SoC) architectures, the latter have received increasing interest in recent years [22]. To support several heavy demanding workloads simultaneously, and reduce unpowered silicon area, computer architects design many special-purpose on-chip accelerators implemented in ASIC and share them among multiple processor cores. Such architectures offer much better performance and lower power compared to performing the same task on a general-purpose CPU. Designing heterogeneous-accelerator SoCs is extremely expensive and time-consuming. The designer has to face many design issues such as the choice of the parallelism degree and the resource utilization of accelerators, their interfaces with the memory hierarchy, etc. Design space exploration methodologies are of major importance.

ASICs和通用目标处理器之间的功耗效率差距，促使异质SoC架构的设计，最近几年对后者有越来越多的兴趣。为同时支持几种需求很大的workloads，并降低没有功耗的硅区域，计算机架构师设计了很多特殊目的的片上加速器，以ASIC实现，并在多个处理器核之间共享。与在通用目标CPU上执行相同的任务相比，这种架构可以给出更好的性能，更低的功耗。设计异质加速器SoC非常昂贵，非常耗时。设计者需要面对很多设计问题，比如并行程度和加速器资源利用的选择，与存储层次结构的接口，等。设计空间探索方法是非常重要的。

In this paper, we present a methodology for designing modern SoC architectures which combine many specialized hardware accelerators and processor cores. We explore the design space of power-performance accelerator-based systems with a SoC simulator and determine the optimal configuration using a hyperparameter optimization algorithm. The proposed simulation infrastructure is based on the use of two tools: gem5-Aladdin [24] and Hyperopt [1]. gem5-Aladdin is an architectural simulator that supports the modeling of complex systems made up of heterogeneous accelerators. Hyperopt is a library implementing different hyperparameter optimization algorithms for solving optimization problems with an unknown objective function [21], such as architecture simulation in our case. The main contributions of this work are as follows.

本文中，我们给出一种方法设计现代SoC架构，将很多专用硬件加速器和处理器核结合到一起。我们用SoC模拟器，探索基于加速器的功耗性能系统的设计空间，使用超参数优化算法来确定最优配置。提出的仿真基础设置是基于两个工具：gem5-Aladdin和Hyperopt。gem5-Aladdin是一个架构模拟器，支持异质加速器组成的复杂系统的建模。Hyperopt是一个库，实现了不同的超参数优化算法，求解未知目标函数的优化问题，比如在我们这种情况下的架构仿真。本文的主要贡献如下。

- A framework for determining, at the system-level, the microarchitecture with the best efficiency, in terms of performance-power ratio.

在系统级确定架构的性能功耗比最佳效率的框架。

- A case study allowed us identifying the most energy efficient architecture for a convolutional neural network (CNN). We showed that the solution obtained achieves a 2x to 4x improvement in energy-delay-product (EDP) compared to an architecture without parallelism. Furthermore this solution is more efficient than commonly implemented architectures (Systolic, 2D-mapping, and Tiling).

使我们找到CNN的最佳功耗效率架构的案例研究。我们证明了，得到的解与没有并行的架构相比，功耗延迟乘积EDP的改进有2x到4x。而且，这个解比通常实现的架构(Systolic, 2D-mapping, and Tiling)要更加高效。

- To demonstrate the efficiency of the heterogeneous-accelerator SoC design approach, we determined the optimal architecture including its coherency interface for a complex SoC made up of six common accelerated-workloads. Three possible coherency models are considered: a software-managed direct memory access (DMA), a shared last level cache (LLC-coherent) and a fully-coherent cache. Our framework allowed to determine that a hybrid interface appears to be the most efficient; it reaches 22% and 12% improvement in EDP compared to just only using non-coherent and only LLC-coherent models, respectively.

为展示异质加速器SoC设计方法的效率，我们对一个由6个常见的加速workloads组成的复杂SoC，确定了最优架构，包括其一致性接口。考虑了三种可能的一致性模型：一种软件管理的DMA，一种共享的LLC，和一种全一致的缓存。我们的框架发现，混合接口似乎是最高效的；与只使用非一致性和LLC一致性模型的相比，EDP分别改进了22%和12%。

## 2. Related Work

Simulation platforms adapted to accelerator-centric architectures are proposed in PARADE [7] and gem5-Aladdin [24]. Both provide simulation platforms enabling the exploration of many-accelerator designs. PARADE is a simulation platform that can be automatically generated through a high-level synthesis (HLS) description of the accelerator. Unlike PARADE, gem5-Aladdin models accelerators based on a dataflow representation extracted from the profiling of the dynamic execution of the program, enabling fast design space exploration. The gem5 simulator [4] is then used for the simulation. With such approaches, the designer is faced with the problem of compromise between accuracy and speed of the simulation. To speed up the simulation, there exist approaches where performance models are deployed on FPGA-based platforms [8].

在PARADE[7]和gem5-Aladdin[24]中，提出了以加速器为中心架构的修改的仿真平台。两者都给出了探索很多加速器设计的仿真平台。PARADE是可以通过加速器的HLS描述来自动生成的仿真平台。与PARADE不同，gem5-Aladdin对加速器进行的建模，是基于从程序的动态执行中收集的信息中提取的数据流表示的，可以进行快速设计空间探索。gem5模拟器然后用于仿真。用这些方法，设计者面临的问题是准确度和仿真的速度的折中。为加速仿真，有一些方法将性能模型部署在基于FPGA的平台上。

There are several projects whose goal is to be able to rapidly evaluate accelerator-based hardware design. Embedded Scalable Platforms (ESP) [19] uses HLS to design accelerator SoCs. Cosmos [18] leverages both HLS and memory optimization tools to improve exploration of the accelerator design space. Centrifuge [12] is able to generate and evaluate heterogeneous accelerators SoCs by combining HLS with FireSim [13], a FPGA-accelerated simulation platform. All these works provide design frameworks for evaluating accelerators and it is up to the user to select the optimal one. Unlike these projects, we aim to provide a unified framework for the design, simulation and optimization of the architecture of accelerator-based SoCs.

有几个项目，其目标是迅速的评估基于加速器的硬件设计。嵌入可扩展平台(ESP)使用HLS来设计加速器SoCs。Cosmos[18]利用HLS和内存优化工具来改进加速器设计空间的探索。Centrifuge[12]通过将HLS与FireSim（一个FPGA加速的仿真平台）结合到一起，可以生成和评估异质加速器SoCs。所有这些工作提供了评估加速器的设计框架，用户可以选择最优的那个。与这些项目不同的是，我们的目标是为基于加速器的SoC架构的设计，仿真和优化提供一个统一的框架。

Machine learning quickly became a powerful tool in computer architecture, with established applicability to design, optimization, simulation, etc., as attested by the survey proposed by Penny et al. [17]. These techniques offer interesting opportunities for architecture simulation, especially in the early stages of the design process. Bhardwaj et al. present in [3] a Bayesian optimization-based framework for determining the optimal hybrid coherency interface for many-accelerator SoCs in terms of performance.

机器学习迅速成为了计算机架构中的一个强力工具，可以应用到设计、优化和仿真等过程中。这些技术为架构仿真给出了有趣的机会，尤其是在设计过程的早期阶段。[3]提出了一种基于贝叶斯优化的框架，对众加速器SoCs，以性能为目标确定最优混合一致性接口。

## 3. Design Approach Overview

### 3.1 Heterogeneous-Accelerator SoC

Figure 1 shows the organization of a typical heterogeneous-accelerator SoC architecture. It includes a number of processor cores and many specialized accelerators. Each accelerator is made up of several dedicated datapaths that implement parts or all of an algorithm of a specific application domain. Each accelerator has a local memory (scratchpad memory or private cache) to speed up data transfer. The architecture also includes DMA, last level cache controller, and coherent memory controllers shared by both processor cores and accelerators. At the system-level, a customized network-on-chip enables the communications between these different components.

图1展示了一种典型的异质加速器SoC架构，包含数个处理器核，和很多专用的加速器。每个加速器是由几个专用的数据通路构成，实现了特定应用领域的一个算法的部分或所有。每个加速器都有本地存储（便签存储或私有缓存），以加速数据传输。这个架构还包括DMA，LLC控制器，和处理器核和加速器共享的一致内存控制器。在系统层次，定制的片上网络让不同的组成部分之间进行通信。

### 3.2 Accelerator Modeling and SoC Simulation

The Aladdin trace-based accelerator simulator [23] profiles the dynamic execution trace of an accelerated workload initially expressed in C and estimates its performance, power, and area. The process mainly consists in building a dynamic data dependence graph (DDDG) of the workload which can be viewed as a data flow representation of the accelerator. This graph is then scheduled taking into account the resource constraints by the way of user-defined hardware parameters such as loop unrolling, loop pipelining, and number of memory ports. The underlying model of the Aladdin simulator is a standalone datapath and its local memories.

基于trace的加速器仿真器Aladdin，对被加速的workload（用C写成的）的动态执行trace进行分析，估计其性能，功耗和面积。这个过程主要构建workload的一个动态数据依赖性图(DDDG)，可以视为加速器的数据流表示。这个图然后安排纳入资源限制，以用户定义的硬件参数，如循环展开，循环流水线，和存储端口的数量。Aladdin仿真器的潜在模型是一个独立数据通路，及其本地存储。

The interactions at the SoC view level, that is to say between the accelerators and the other components of the system, are managed by gem5-Aladdin [24], which realizes the coupling of Aladdin with the gem5 architectural simulator [4]. gem5-Aladdin is capable of evaluating interactions between accelerators and processor cores, DMAs, caches, virtual memory, in SoC architectures such as the one illustrated in Figure 1. gem5-Aladdin supports three coherency models for accelerators. (i) non-coherent: using software-managed DMAs (ii) LLC-coherent: by directly accessing the coherent data in the last-level-cache (LLC) without having a private cache; (iii) fully-coherent caches: each accelerator can use its private cache to access the main memory.

在SoC的观察层次的互动，即加速器和系统其他组成部分之间的，由gem5-Aladdin来管理，实现了Aladdin和gem5架构模拟器之间的耦合。gem5-Aladdin可以评估加速器和处理器核，DMAs，缓存，虚拟存储之间的互动，如图1中的SoC架构。gem5-Aladdin对加速器支持三种一致性模型。(i)非一致的：使用软件管理DMAs，(ii)LLC一致性：在LLC中直接访问一致的数据，没有私有缓存；(iii)完全一致的缓存：每个加速器都可以使用其私有缓存来访问主存。

### 3.3 Hyperparameter Optimization Method

Performing design space exploration manually leads to inefficient and time-consuming processes. Approaches based on hyperparameter optimization prove to be very effective in optimizing unknown objective functions as stated in works presented in [3, 21]; they are more powerful than heuristic optimization in terms of convergence and quality of obtained solutions.

手工进行设计空间探索是低效耗时的过程。基于超参数优化的方法在优化未知目标函数上非常有效，如[3,21]所述，以收敛性和得到的结的质量而论，比启发式优化方法会更加高效。

There are several approaches to hyperparameter optimization. Bayesian hyperparameter optimization (also known as sequential model-based optimization, SMBO) is used here. The approach is to build a probability model of the objective function which is used to select the most promising hyperparameters to evaluate in the true objective function. Several variants of SMBO methods exist in practice, based on Gaussian Processes, Random Forests Regressions and Tree Parzen Estimators (TPE). Hyperopt package implements TPE [2]. One of the main advances of TPE over other probabilistic methods is that it is able to retain dependencies between parameters as it models the density of a parameter for "good" experiments and compares this to its density for "bad" experiments. It can then use these models to determine an expected improvement of the objective function for any values a parameter can take. P(x|y), which is the probability of the hyperparameters given the score on the objective function, is expressed as:

有几种超参数优化的方法。贝叶斯超参数优化（也称为基于模型的序列优化，SMBO）可以用在这里。这个方法要构建目标函数的一个概率模型，该目标函数用于选择在评估真正的目标函数上最有希望的超参数。实践中存在SMBO方法的几种变体，基于高斯过程的，随机森林回归和TPE的。Hyperopt包实现了TPE。TPE相对于其他概率模型的主要优势是，保持了参数之间的依赖关系，因为对一个参数在好的试验上建模其密度，并与其在坏试验上的密度相比较。可以使用这些模型，对任何可以接受的参数值，确定目标函数的期望改进(EI)。P(x|y)是在给定目标函数分数的情况下，超参数的概率，表示为：

$$P(x|y) = l(x), if y < y*; g(x), if y > y*$$(1)

where y < y∗ represents a lower value of the objective function than the threshold. As equation 1 shows, two different distributions for the hyperparameters are calculated: one where the value of the objective function is less than the threshold, l(x), and one where the value of the objective function is greater than the threshold, g(x). Once l(x) and g(x) have been expressed, TPE is able to identify the next parameter x_next considering that x_next = argmin_x g(x)/l(x).

其中y < y*表示目标函数值比阈值要低。如式1所示，计算了超参数的两个不同的分布：一种是目标函数值低于阈值的l(x)，一种是目标函数值高于阈值的g(x)。一旦l(x)和g(x)有了表达式，TPE可以找到下一个参数x_next，即x_next = argmin_x g(x)/l(x)。

TPE builds a search history and predicts at each iteration the best trial to try next. TPE itself has many parameters that can be tuned to improve the effectiveness of the TPE algorithm. Adaptive-TPE (ATPE), a modified version of TPE, uses a pre-trained machine-learning model to help optimize faster and more accurately.

TPE构建了一个搜索历史，在每次迭代中预测下一次的最佳尝试。TPE本身有很多参数，可以调节以改进TPE算法的有效性。自适应TPE是TPE的一个修正版本，使用一个预训练的机器学习模型，来帮助优化的更快更准确。

## 4. Designing SoC with this Flow

### 4.1 Parallel Accelerator Exploration

To improve the performance of applications, computation-intensive parts, typically loops, are mapped to hardware accelerators. Loop nests of the considered workloads define the exploration space, as illustrated in Figure 3, with the convolutional layer of a Convolutional Neural Network (CNN) application. This layer exhibits intensive parallelism at the feature map, neuron, and kernel levels. There are four parameters: M (number of output feature maps), N (number of input feature maps), S (output feature map size, or number of neurons), and K (kernel size). These 6 nested-loops offer an interesting exploration space as it is possible to play with different loop unrolling factors.

为改进应用的性能，计算量很大的部分，典型的如循环，会映射到硬件加速器。考虑的workloads的循环嵌套，定义了探索空间，如图3所示，是CNN应用的卷积层。这层在特征图、神经元和核的层次上表现出了很强的并行性。有4个参数，M是输出特征图的数量，N是输入特征图的数量，S是输出特征图的大小，或神经元的数量，还有K是核的大小。这个6层嵌套给出了一个有趣的探索空间，因为可能用不同的循环展开因子进行试验。

### 4.2 Memory Coherency Models Exploration

Giri et al. [11] identified three common coherency interfaces used to integrate accelerators with the memory hierarchy in a loosely-coupled architecture.

Giri等[11]列出了三种常用的一致性接口，用于集成加速器到松散耦合架构的内存层次结构中。

- In a non-coherent interfacing model, the accelerator has a scratchpad memory (SPM) for local storage and uses DMA to load data from DRAM, as illustrated in Figure 1.

在非一致接口模型中，加速器有一个scratchpad存储(SPM)用于本地存储，使用DMA来从DRAM中载入数据，如图1所示。

- LLC-coherent accelerators send DMA requests to the LLC. Their implementation is similar to non-coherent accelerators, but the LLC-coherent DMA requests/responses are routed to the cache-coherent module instead of the DMA.

LLC一致的加速器，向LLC发送DMA请求。其实现类似于非一致加速器，但LLC一致的DMA请求/响应路由到缓存一致模块，而不是DMA。

- In a fully-coherent model, each accelerator has its private cache which implements a cache coherence protocol such as MESI or MOESI, similar to a processor’s cache.

在一个完全一致的模型中，每个加速器都有其私有缓存，实现了一个缓存一致性协议，比如MESI或MOESI，与处理器核类似。

Each of these three coherency models offers interesting power-performance trade-offs. In a SoC integrating several accelerators to support a versatile application, a single coherency interface used by all accelerators may not be the most optimal in terms of power and performance as shown by Giri et al. [11]. Each accelerator may have its own coherency model, and this is what we explore here.

这三个一致性模型的每一个都有有趣的性能-功耗折中。在一个SoC中，集成了几个加速器，以支持多种应用，所有加速器都使用一种一致性接口，可能在性能和功耗上并不是最优的，如Giri等[11]所示。每个加速器都有其自己的一致性模型，这就是我们所探索的地方。

### 4.3 Hyperopt-gem5-Aladdin Framework

As shown in Figure 2, the parallelism of an accelerator is set by design pragma directives such as the loop unrolling factor during the accelerator modeling phase. This phase is used to evaluate and update the power-performance of accelerators.

如图2所示，加速器的并行性是在加速器建模阶段，由设计pragma directives设置的，如循环展开因子。这个阶段用于评估和更新加速器的功耗性能。

The gem5-Aladdin simulator is able to model SoCs including several accelerators that can use different coherency models, and run various workloads concurrently. The complete SoC is specified using a SoC configuration file which describes the configuration of processors, accelerators, memories/caches, and interconnect. The gem5-Aladdin simulator is an objective function of the optimization method; it provides performance, power, delay time of SoC architectures that we want to optimize.

gem5-Aladdin仿真器能够对包含几个加速器的SoCs进行建模，这几个加速器使用了不同的一致性模型，并同时运行各种workloads。完整的SoC使用一个SoC配置文件来指定，描述了处理器核，加速器，内存/缓存，和互联的配置。gem5-Aladdin模拟器是优化方法的一个目标函数；其给出了我们想要优化的SoC架构的性能，功耗，和延迟时间。

Algorithm 1 presents the pseudo-code of the method used in Hyperopt. The algorithm starts by randomly selecting k architecture configurations and simulates them with the gem5-Aladdin simulator (line 2). The search history is initiated (line 3), it consists of k pairs of configuration and its associated EDP. The next steps of the method are iterative and are performed N − k times, where N is the budget on the number of architecture simulations. The search space is narrowed down from the search history and a new configuration for the next simulation step is suggested using equations presented in Section 3.3 (lines 4, 5, 6, 7, 8, 9). Once all the iterations have been completed, the optimal architecture configuration set which reaches the minimum EDP is selected (line 10).

算法1给出了Hyperopt中使用的方法的伪代码。算法的起始，随机选择了k个架构配置，用gem5-Aladdin模拟器进行仿真（第2行）。初始化搜索历史（第3行），由k对配置和其相关的EDP组成。该方法下面的步骤是迭代的，进行N-k次，N是架构仿真的次数的预算。搜索空间从搜索历史窄化了，使用3.3节中的式子建议出下一次仿真的新的配置（第4，5，6，7，8，9行）。一旦完成了所有的迭代，选择达到最小EDP的最优架构配置集合。

## 5. Experiments

To show the effectiveness of our design approach, we present two experiments: the first one concerns the design of CNN accelerators and the second one the design of a SoC including six accelerator tiles. The SoC configuration setup of the gem5-Aladdin simulator of the two experiments is given in Table 1.

为展示我们的设计方法的有效性，我们给出两个试验：第一个关心的是CNN加速器的设计，第二个关心的是SoC的设计，包含了6个加速器tiles。两个试验的gem5-Aladdin仿真器的SoC配置设置，在表1中给出。

Table 1: gem5-Aladdin SoC Architechture Configuration

Component | Description
--- | ---
CPU Type | OoO X86
System Clock | 100MHz
Cache Line Size | 64 bits
L2 Cache(LLC) | 2MB, 16-way, LRU
Memory | DDR3_1600_8x8, 4GB
Hardware Prefetchers | Strided
Data Transfer Mechanism | DMA/Cache

### 5.1 CNN Accelerator in a SoC

As discussed in Section 4.1, CNN layers are highly computation intensive and exhibit fine-grained parallelism at feature map (FP), neuron (NP), and synapse (SP) levels. This potential parallelism offers many opportunities to speed up the calculations. However, most of existing CONV accelerators exploit the parallelism only at one level [16]. Systolic architectures can only exploit synapse parallelism [5], 2D-Mapping architectures neuron parallelism [10], and Tiling architectures feature map parallelism [6]. There is a lack of architectural studies trying to exploit these different types of fine-grained parallelism simultaneously. By exploring all possible types of parallelism, and depending on user constraints, greater efficiency can be expected.

在4.1节中讨论过，CNN层的计算量非常大，在特征图FP，神经元NP和突触SP级都表现出了细粒度的并行性。这种潜在的并行性有很多机会来加速其计算。但是，多数现有的CONV加速器只在一个层次上利用了并行性。Systolic架构只能利用突触级的并行性，2D-Mapping架构利用神经元级别的并行性，Tiling架构利用特征图级的并行性。缺少同时利用这些不同类型的细粒度并行性的架构研究。通过探索所有可能的并行类型，依赖于用户的约束，可以得到更好的效率。

The calculations of a CONV layer, as shown with the code in Figure 3, can be unrolled in different ways. The labels in the code (loop_m, loop_n, loop_r, loop_c, loop_i, loop_j) are used to set the unrolling factors and quantify the parallelism degree of each loop. According to the different unrolling strategies of the loops, there are three types of parallelism.

CONV层的计算，如图3的代码所示，可以以不同的方式进行展开。代码中的标签(loop_m, loop_n, loop_r, loop_c, loop_i, loop_j)用于设置展开因子，量化每个循环的并行程度。根据循环不同的展开策略，有三种类型的并行性。

- Feature map Parallelism (FP), loop_m output feature maps, and loop_n input feature maps are processed at a time (maximum factors are M and N respectively).

特征图并行FP，loop_m输出特征图，loop_n输入特征图是在同时处理的（最大因子分别为M和N）。

- Neuron Parallelism (NP), loop_r , and loop_c neurons of one output feature map are processed at a time (maximum factor is S).

神经元并行性NP，一个输出特征图的loop_r和loop_c个神经元是同时处理的（最大因子为S）。

- Synapse Parallelism (SP), loop_i, and loop_j synapses of one kernel are computed at a time (maximum factor is K).

突触并行性SP，一个核的loop_i和loop_j个突触是同时处理的（最大因子为K）。

The design space is built by combining these three types of parallelism. As an example, an architecture may handle a single input feature map and a single output feature map (loop_m = 1 and loop_n = 1), one neuron of each output feature map (loop_r = 1 and loop_c = 1), but multiple synapses of each kernel at a time (loop_i > 1 or loop_j > 1). This style of parallel computing is named Single Feature map, Single Neuron, Multiple Synapses (SFSNMS). It is obviously possible to define other processing styles: SFSNSS, SFMNSS, SFMNMS, MFSNSS, MFSNMS, MFMNSS and MFMNMS [16].

设计空间是通过将这三种类型的并行性结合到一起而构建的。例如，一种架构可以同时处理一个输入特征图和一个输出特征图(loop_m = 1 and loop_n = 1)，每个输出特征图的一个神经元(loop_r = 1 and loop_c = 1)，但每个核的多个突触(loop_i > 1 or loop_j > 1)。这种类型的并行计算命名为SFSNMS。显然，还可以定义其他的处理类型：SFSNSS，SFMNSS，SFMNMS，MFSNSS，MFSNMS，MFMNSS和MFMNMS。

We evaluated three common workloads. LeNet-5 [14], the most famous handwriting recognition model, FR[9] implementing a face recognition model and HG [15] used to recognize hand gestures of human. In this experiment, we used a non-coherent interface model, it has a private scratchpad memory for local storage and uses DMA to request data from the main memory.

我们评估三种常见的workloads。LeNet-5是最著名的手写识别模型，FR实现了一个人脸识别模型，HG用于识别人类手部姿势。在这个试验中，我们使用非一致接口模型，本地存储有一个私有的便签内存，使用DMA来从主存中请求数据。

Table 2 gives the configuration of three well-known parallel architectures (tiling, 2D-mapping and systolic) for each of the considered workloads. The fourth architecture, called selection, corresponds to that resulting from our exploration.

表2对每个workloads，给出了三种著名并行架构的配置(tiling, 2D-mapping and systolic)。第四种架构称为selection，是我们的探索得到的结果。

Figure 4 shows the EDP results for the six workloads. The horizontal axis denotes the workloads and the vertical axis denotes EDP value normalized by EDP of a baseline architecture without any parallelism whose parameters are (1,1,1,1,1,1). The columns in one benchmark represent the normalized EDP of the different architectures.

图4展示了6种workloads的EDP结果。水平轴表示workloads，垂直轴表示归一化的EDP值，基准架构是没有任何并行性的，其参数为(1,1,1,1,1,1)。在一个基准测试中的列表示不同架构的归一化EDP。

The different EDP improvements of these architectures, illustrated in Figure 4, can be explained by two main reasons: data reuse and use of computing resources. Systolic and 2D-Mapping architectures have a comparable improvement in terms of energy. Systolic has a higher latency than 2D-Mapping because of the long initialization phase to fill the chain of processing elements. But systolic has a higher data reuse factor than 2D-Mapping, therefore systolic consumes less energy than 2D-Mapping for most workloads. At the SoC level, most of the energy is consumed by the data movement, so if data reuse increases, EDP also increases. In the case of tiling, the EDP improvement is very low because of low computing resource utilization. It has the poorest energy efficiency due to high latency and poor data reuse. Our selected configuration combines systolic and 2D-Mapping. This corresponds to configurations having maximal synapse parallelism to increases data reuse, and a high neuron parallelism to balance computing resource utilization and local memory load/store power consumption. In summary, the configuration proposed with our flow allows obtaining a better EDP than usual architectures (Systolic, 2D-mapping and Tiling) for accelerator-based SoCs. This results in an improvement of the EDP by a factor between 2 and 4 compared to a sequential architecture.

图4中展示的这些架构的不同EDP改进，可以由两个主要原因解释：数据重用，和使用计算资源。Systolic和2D-Mapping架构在功耗上的改进类似。Systolic的延迟比2D-Mapping更高，因为填充处理元素链条的初始化阶段较长。但systolic的数据重用率比2D-Mapping更高，因此对多数workloads，systolic消耗的能量比2D-Mapping要更少。在SoC层次，多数能量都是由数据搬运消耗的，因此如果数据重用增加，EDP就增加。在tiling的情况中，EDP改进很低，因为计算资源利用率很低。其能量效率最低，因为延迟很高，数据重用也很差。我们选定的配置将systolic和2D-Mapping结合到了一起。这对应的配置，其突触并行性最大，增加了数据重用，神经元并行性也很高，平衡了计算资源利用和本地内存load/store消耗。总结起来，我们的工作流提出的配置，对基于加速器的SoCs，比通常的架构(Systolic, 2D-mapping and Tiling)有更好的EDP。与顺序结构相比，EDP的改进因子介于2到4之间。

### 5.2 Hyperopt Convergence Study

We studied the convergence of the hyperparameter optimization algorithm and compared three implementations: random search, conventional TPE and ATPE. LeNet-5 workload is used as a case study. The three implementations are executed in the same search space and the convergence results are illustrated in Figure 5. The total number of possible configuration is 840. The simulation of all possible configurations confirmed the solution obtained with our optimization method. As illustrated in Figure 5, the optimal solution is obtained after a small number of iterations, since 40 are sufficient. The EDP improvement values are distributed into four groups. This distribution can be explained by the complexity of the exploration space, since we try to mix three types of parallelism, as mentioned in 5.1. The ATPE algorithm requires around 30 iterations to converge in the lowest group and achieves the best EDP improvement after 40 iterations. Each iteration requires 30 minutes of CPU time (Intel Xeon E5-2609 at 1.9GHz), considering that gem5-Aladdin represents most of the CPU time. Using this hyperparameter optimization method, and as shown here, we can get a solution faster, which is really useful in the presence of large design spaces.

我们研究了超参数优化算法的收敛性，比较了三种实现：随机搜索，传统的TPE和ATPE。LeNet-5 workload用作案例研究。三种实现在相同的搜索空间中实现，收敛性结果如图5所示。可能配置的总计数量为840。所有可能配置的仿真确认了我们优化方法得到的解。如图5所示，最优解在较少次数的迭代后就可以得到，因为40次就足够了。EDP改进值分布在4组中。这种分布可以由探索空间的复杂度解释，因为我们试图混合三种并行类型，如5.1中提到的。ATPE算法需要大约30次迭代，收敛到最低的组，在40次迭代后得到了最佳的EDP改进。每次迭代需要30分钟CPU时间(Intel Xeon E5-2609 at 1.9GHz)，考虑gem5-Aladdin达标了大多数CPU时间。使用这种超参数优化方法，如这里展示的，我们可以更快的得到一个解，在大型设计空间的探索中，这非常有用。

### 5.3 Coherency Interface Choice Study

The SoC configuration used for evaluating heterogeneous-accelerator architectures is a tiled architecture consisting of one CPU and six accelerator tiles, along with L2 cache controller and main memory controller tiles. The processing units all perform a different task, which means that all the accelerators operate in parallel. Table 3 gives the features of the accelerated-workloads used for the experiment. Two LeNet-5 convolutional neural network layers perform an image classification task. The others correspond to four benchmarks from MachSuite[20]: AES-256, GEMM-nCubed, FFT-Transpose, and Stencil-3D.

用于评估异质加速器架构的SoC配置，是tiled架构，由一个CPU和6个加速器tiles组成，还有L2缓存控制器和主存控制器tiles。处理单元都处理不同的任务，意思是所有加速器并行运算。表3给出了试验中用到的加速的workloads的特征。两个LeNet-5 CNN层进行图像分类任务。其他对应着MachSuite的四种基准测试：AES-256，GEMM-nCubed，FFT-Transpose，和Stencil-3D。

The goal of this experiment is to determine the best coherency interface for each accelerator separately and for the SoC made up of these six accelerators. The performance of each accelerator is affected not only by its computation time and memory access patterns but also by possible conflicts when accessing shared resources. Consequently, the coherency models adapted to each accelerator are difficult to predict at design time. The input space of hyperparameter is six dimensional due to the six accelerators. Each accelerator interface can be either non-coherent, LLC-coherent or fully-coherent, this results in a total of 729 possible configurations. Figure 6 shows the EDP results for each accelerator and for the six-accelerator version. The horizontal axis denotes the different accelerators and the vertical axis denotes EDP normalized with respect to the non-coherent configuration. The columns in one benchmark represent the normalized EDP of the different interfaces.

本试验的目标是，对由6个加速器组成的SoC，和为每个加速器单独确定最佳的一致性接口。每个加速器的性能，不仅受到其计算时间和内存访问模式影响，还受到在访问共享资源时的可能冲突的影响。结果是，对每个加速器修改的一致性模型，在设计时间是很难预测的。超参数的输入空间是6维的，因为有6个加速器。每个加速器接口可以是非一致的，LLC一致的，或完全一致的，这总计有729种可能的配置组合。图6展示了对每个加速器和6个加速器版本的EDP结果。水平轴表示不同的加速器，垂直轴表示对非一致配置归一化的EDP。一个基准测试中的列表示不同接口的归一化EDP。

In most cases, with the exception of FFT-transpose, the full-coherent interface performs worst due to its significant hardware and performance overheads. In particular, for CONV-accelerators such as LeNet-5_C1 and LeNet-5_C3. They access a large amount of data (kernels, inputs, and outputs), which cannot fit in the L1 caches and can therefore lead to significant cache misses, penalizing the overall latency. FFT-transpose performs better with fully-coherent than with non-coherent because only eight bytes per 512 bytes of data are read per iteration whereas with the DMA system almost all the data must be available before the computation starts. Furthermore, LLC-coherent shows a better EDP than non-coherent since the memory requests are first sent to the LLC, and when the LLC hits, this results in much shorter access latency.

在多数情况下，完全一致接口表现最差，因为其硬件和性能代价都很高，但FFT-transpose是一个例外。特别是，对于CONV加速器，如LeNet-5_C1和LeNet-5_C3。它们要访问大量的数据（核，输入和输出），L1缓存是无法适应的，因此会导致显著的缓存misses，使整体延迟变高。FFT-transpose在完全一致接口中比非一致接口表现的更好，因为每次迭代中只读入512 bytes数据中的8 bytes，而用DMA系统的话，在计算开始时要准备好所有数据。而且，LLC一致接口比非一致接口的EDP更好，因为内存访问首先送到LLC中，当LLC hits后，这会得到短的多的访问延迟。

For the six-accelerator version, hybrid selection offers better EDP than systems using a single coherency interface. The solution obtained is the following: LeNet-5_C1 and LeNet-5_C3 use non-coherent while the other accelerators use LLC-coherent interface. This hybrid solution results in an improvement in EDP of 22% and 12% respectively, compared to only non-coherent and LLC-coherent. Although the average (geometric mean) of the EDP improvement over the six accelerators gives the benefit to the only LLC-coherent model, it appears that in the global system view, the hybrid coherent model achieves better EDP improvement. There are many reasons that explain the EDP improvement brought by the hybrid solution. Having a subset of accelerators with non-coherent interfaces reduces pressure at the LLC level. Indeed, if only four accelerators share the last-level cache, the time spent in data movement compared to an all LLC-coherent solution is reduced. In addition, CONV accelerators benefit from the use of a non-coherent interface with streaming data access patterns applications.

对于6加速器版本，混合选择比使用单一一致性接口的选择，给出了更好的EDP。得到的解如下所示：LeNet-5_C1和LeNet-5_C3使用非一致接口，而其他加速器使用LLC-一致接口。余非一致和LLC一致接口比，这种混合的解得到的EDP改进分别为22%和12%。虽然对6加速器的平均EDP改进只倾向于LLC一致模型，但在全局系统的角度来看，混合一致模型获得了更好EDP改进。有很多原因可以解释混合解得到的EDP改进。有一些加速器是非一致接口的，会降低LLC级的压力。确实，如果只有4个加速器共享LLC，与全部都是LLC一致的解相比，在数据移动时所消耗的时间就降低了。而且，CONV加速器在流数据访问模式应用下，会从非一致接口的使用中受益。

## 6 Conclusion and Future Work

In this paper, we described a flow helping in the design of heterogeneous-accelerator SoCs. The flow combines the gem5-Aladdin simulator and a hyperparameter optimization method. It identifies automatically the optimal architecture for heterogeneous-accelerator SoCs. To evaluate our approach, we explored the design space of accelerators for convolutional neural networks including their memory coherency interfaces.

本文中，我们描述了一种工作流，帮助设计异质加速器SoCs。工作流结合了gem5-Aladdin仿真器和一种超参数优化方法。它会对异质加速器SoCs自动找到最优架构。为评估我们的方法，我们探索了CNN加速器设计空间，还包括其内存一致性接口。

In the near future, we plan to add more parameters to expand our design space (scratchpad partitioning, system bus width, cache size, network on chip, etc.). In addition, we are also interested in measuring the efficiency of the optimization algorithms, and are looking to integrate new algorithms into our framework, in order to compare them.

未来我们计划加入更多的参数，以扩大我们的设计空间（便签分割，系统总线宽度，缓存大小，片上网络，等）。另外，我们还可以度量优化算法的效率，将新的算法整合到我们的框架中，以进行比较。