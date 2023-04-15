# A Top-Down Method for Performance Analysis and Counters Architecture

Ahmad Yasin & Intel Corporation

## 0. Abstract

Optimizing an application’s performance for a given microarchitecture has become painfully difficult. Increasing microarchitecture complexity, workload diversity, and the unmanageable volume of data produced by performance tools increase the optimization challenges. At the same time resource and time constraints get tougher with recently emerged segments. This further calls for accurate and prompt analysis methods.

对给定的架构优化一个应用的性能已经变得极度困难。增加的微架构复杂度，workload多样性，和性能工具产生的不可管理的数据量，增大了优化的挑战。同时，资源和时间约束变得越来越严格。这进一步要求准确和迅速的分析方法。

In this paper a Top-Down Analysis is developed – a practical method to quickly identify true bottlenecks in out-of-order processors. The developed method uses designated performance counters in a structured hierarchical approach to quickly and, more importantly, correctly identify dominant performance bottlenecks. The developed method is adopted by multiple in-production tools including VTune. Feedback from VTune average users suggests that the analysis is made easier thanks to the simplified hierarchy which avoids the high-learning curve associated with microarchitecture details. Characterization results of this method are reported for the SPEC CPU2006 benchmarks as well as key enterprise workloads. Field case studies where the method guides software optimization are included, in addition to architectural exploration study for most recent generations of Intel Core™ products.

本文中，我们提出一种自上而下的分析方法，在乱序处理器中迅速找到瓶颈的方法。提出的方法在一种结构化的、层次化的方法中，使用指定的性能计数器，以迅速的、准确的找到主要的性能瓶颈。提出的方法被多个生产中的工具所采用，包括VTune。VTune用户反馈，多亏了这种简化的层次，分析变得简单了，避免了与微架构细节相关的高学习曲线。对SPEC CPU2006 benchmarks以及关键的企业化workloads，给出了这种方法的刻画结果。本文还包含了这种方法引导软件优化的例子，还有对最新的Intel Core产品的架构探索研究。

The insights from this method guide a proposal for a novel performance counters architecture that can determine the true bottlenecks of a general out-of-order processor. Unlike other approaches, our analysis method is low-cost and already featured in in-production systems – it requires just eight simple new performance events to be added to a traditional PMU. It is comprehensive – no restriction to predefined set of performance issues. It accounts for granular bottlenecks in super-scalar cores, missed by earlier approaches.

这种方法的洞见引导提出了一种新的性能计数器架构，可以对通用乱序处理器确定真实的瓶颈。与其他方法不同，我们的分析方法代价很低，已经在生产系统中使用，它只需要8个简单的新的性能事件加入到传统PMU中。我们的方法是很广泛的，对性能事件的集合没有限制。之前的方法无法识别的超标量核中的颗粒瓶颈，本文都可以进行识别。

## 1. Introduction

The primary aim of performance monitoring units (PMUs) is to enable software developers to effectively tune their workload for maximum performance on a given system. Modern processors expose hundreds of performance events, any of which may or may not relate to the bottlenecks of a particular workload. Confronted with a huge volume of data, it is a challenge to determine the true bottlenecks out of these events. A main contributor to this, is the fact that these performance events were historically defined in an ad-doc bottom-up fashion, where PMU designers attempted to cover key issues via “dedicated miss events” [1]. Yet, how does one pin-point performance issues that were not explicitly foreseen at design time?

PMU的主要目标是让软件开发者可以在给定系统中调节其workload到最高性能。现在处理器会展现出数百个性能事件，都可能与特定workload的瓶颈相关或不相关。从这个事件中确定真实的瓶颈，会遇到非常多的数据，从而构成了一个挑战。这些性能事件是在历史上以一种ad-hoc bottom-up的方式定义的，其中PMU的设计者试图通过精细的miss events来覆盖关键的事件。但是，怎样才能确定在设计时没有预见到的性能问题呢？

Bottleneck identification has many applications: computer architects can better understand resource demands of emerging workloads. Workload characterization often uses data of raw event counts. Such unprocessed data may not necessary point to the right bottlenecks the architects should tackle. Compiler writers can determine what Profile Guided Optimization (PGO) suit a workload more effectively and with less overhead. Monitors of virtual systems can improve resource utilization and minimize energy.

识别瓶颈有很多应用：计算机架构师可以更好的理解新workloads的资源需求。Workload刻画通常使用原始事件计数的数据。这种未处理过的数据，不一定指向正确的瓶颈。编译器开发者可以更高效的，确定什么样的PGO更适合一个workload。虚拟系统监控会改进资源利用，并最小化功耗。

In this paper, we present a Top-Down Analysis - a feasible, fast method that identifies critical bottlenecks in out-of-order CPUs. The idea is simple - a structured drill down in a hierarchical manner, guides the user towards the right area to investigate. Weights are assigned to nodes in the tree to guide users to focus their analysis efforts on issues that indeed matter and disregard insignificant issues. For instance, say a given application is significantly hurt by instruction fetch issues; the method categorizes it as Frontend Bound at the uppermost level of the tree. A user/tool is expected to drill down (only) on the Frontend sub-tree of the hierarchy. The drill down is recursively performed until a tree-leaf is reached. A leaf can point to a specific stall of the workload, or it can denote a subset of issues with a common micro-architectural symptom which are likely to limit the application’s performance.

本文中，我们提出了一种自上而下的分析方法，一种可行的，快速的方法，在乱序CPUs中识别关键的瓶颈。思想是简单的，一种结构化的向下展开，层次化的方式，引导用户朝向正确的调查区域。给树中的节点指定权重，引导将其分析聚焦在重要的区域，不要去关心不重要的问题。比如，比如给定的应用明显被取指的问题伤害；我们方法会在树的最上层将其分类为前端bound。用户/工具要只在层次结构的前端子树上进行进一步挖掘。挖掘会迭代进行，直到达到一个叶节点。叶节点会指向workload的特定stall，或表示通用微架构特征的一个问题子集，很可能限制了应用的性能。

We have featured our method with the Intel 3rd generation Core™ codenamed Ivy Bridge. Combined with the hierarchical approach, a small set of Top-Down oriented counters are used to overcome bottleneck identification challenges (detailed in next section). Multiple tools have adopted our method including VTune [2] and an add-on package to the standard Linux perf utility [3]. Field experience with the method has revealed some performance issues that used to be underestimated by traditional methods. Finally, the insights from this method are used to propose a novel performance counters architecture that can determine the true bottlenecks of general out-of-order architecture, in a top down approach.

我们用该方法分析了Intel第3代核，名为Ivy Bridge。与层次化方法结合，我们使用了面向自上而下方法的计数器集合，来克服瓶颈识别的挑战。多种工具都采用了我们的方法，包括VTune，和Linux标准Perf工具的一个插件包。该方法的实战经验揭示了一些性能问题，而传统方法则低估了这些问题。最后，该方法的洞见可用于提出新的性能计数器架构，以自上而下的方法确定一般的乱序架构的瓶颈。

The rest of this paper is organized as follows. Section 2 provides a background and discusses the challenges with bottleneck identification in out-of-order CPUs. The Top-Down Analysis method and its abstracted metrics are introduced in Section 3. In Section 4, novel low-cost counters architecture is proposed to obtain these metrics. Results on popular workloads as well as sample use-cases are presented in Section 5. Related work is discussed in Section 6 and finally, Section 7 concludes and outlines future work.

本文剩余部分组织如下。第2部分给出了背景，讨论了在乱序CPUs中的瓶颈识别的挑战。第3部分给出了自上而下的分析方法及其抽象的度量。在第4部分，提出了新的低代价计数器架构，以得到这些度量。第5部分流行workloads的结果，以及使用案例。第6部分给出了相关工作，第7部分给出了结论和未来工作。

## 2. Background

Modern high-performance CPUs go to great lengths to keep their execution pipelines busy, applying techniques such as large-window out-of-order execution, predictive speculation, and hardware prefetching. Across a broad range of traditional workloads, these high-performance architectures have been largely successful at executing arbitrary code at a high rate of instructions-per-cycle (IPC). However, with these sophisticated super-scalar out-of-order machines attempting to operate so “close to the edge”, even small performance hiccups can limit a workload to perform far below its potential. Unfortunately, identifying true performance limiters from among the many inconsequential issues that can be tolerated by these CPUs has remained an open problem in the field.

现代高性能CPUs努力保持其执行流水线繁忙，使用了很多技术，如大窗口乱序执行，预测投机，和硬件预取。传统workloads类型很多，这些高性能架构大致上都很成功，能以很高的速度IPC执行任意的代码。但是，这些复杂的超标量乱序机器试图的运算，即使是小的性能问题，也会将workload的性能降到最高点很多。不幸的是，从很多非结果性问题中，识别真的性能限制，仍然是这个领域的开放问题。

From a bird’s eye view, the pipeline of modern out-of-order CPU has two main portions: a frontend and a backend. The frontend is responsible for fetching instructions from memory and translating them into micro-operations (uops). These uops are fed to the backend portion. The backend is responsible to schedule, execute and commit (retire) these uops per original program’s order. So as to keep the machine balanced, delivered uops are typically buffered in some “ready-uops-queue” once ready for consumption by the backend. An example block diagram for the Ivy Bridge microarchitecture, with underlying functional units is depicted in Figure 1.

从鸟瞰的视角，现代乱序CPU的流水线有两个主要的部分：前端和后端。前端负责从内存中取指，将其翻译成微操作。这些uops送入后端。后端负责按照原始程序的顺序调度，执行和commit (retire)这些uops。为保持机器均衡，发送过去的uops一旦准备好由后端消耗，一般就在一些ready-uops-queue中缓存。图1给出了Ivy Bridge微架构的一个例子框图，包含潜在的功能单元。

Traditional methods [4][5] do simple estimations of stalls. E.g. the numbers of misses of some cache are multiplied by a pre-defined latency:

传统方法进行简单的stalls估计，如，某cache的misses次数，乘以预定义的延迟：

$$Stall_Cycles = \sum Penalty_i * MissEvent_i$$

While this “naïve-approach” might work for an in-order CPU, surely it is not suitable for modern out-of-order CPUs due to numerous reasons: (1) Stalls overlap, where many units work in parallel. E.g. a data cache miss can be handled, while some future instruction is missing the instruction cache. (2) Speculative execution, when CPU follows an incorrect control-path. Events from incorrect path are less critical than those from correct-path. (3) Penalties are workload-dependent, while naïve-approach assumes a fixed penalty for all workloads. E.g. the distance between branches may add to a misprediction cost. (4) Restriction to a pre-defined set of miss-events, these sophisticated microarchitectures have so many possible hiccups and only the most common subset is covered by dedicated events. (5) Superscalar inaccuracy, a CPU can issue, execute and retire multiple operations in a cycle. Some (e.g. client) applications become limited by the pipeline’s bandwidth as latency is mitigated with more and more techniques.

这种简单方法对顺序CPU可能是可以的，但对于现代乱序CPUs是不适合的，原因如下：(1)stalls会重叠，很多单元会并行进行，如，在处理一个数据cache miss的时候，同时某条未来的指令不在指令cache中；(2)投机执行，CPU进行了不正确的控制流。在非正确路径的事件不是很关键，那些正确路径的事件更重要一些；(3)惩罚是依赖于workload的，而简单方法假设对所有的workloads都是固定的惩罚；(4)局限于预定义的miss events集合，而这些复杂的微架构有非常多的小问题，而这些事件只覆盖了最常见的子集；(5)超标量不准确性，一个CPU可以在一个周期中发射，执行，退役多个运算。一些应用（如客户机）被流水线的带宽所限制，因为延迟有越来越多的技术弥补。

We address those gaps as follows. A major category named “Bad Speculation” (defined later) is placed at the top of the hierarchy. It accounts for stalls due to incorrect predictions as well as resources wasted by execution of incorrect paths. Not only does this bring the issue to user’s first attention, but it also simplifies requisites from hardware counters used elsewhere in the hierarchy. We introduce a dozen truly Top-Down designated counters to let us deal with other points. We found that determining what pipeline stage to look at and “to count when matters”, play a critical role in addressing (1) and (3). For example, instead of total memory access duration, we examine just the sub-duration when execution units are underutilized as a result of pending memory access. Calling for generic events, not tied to “dedicated miss events” let us deal with (4). Some of these are occupancy events (An occupancy event is capable to increment by more than 1 in a given cycle when a certain condition is met for multiple entities) in order to deal with (5).

我们这样处理这些gaps。在层次结构的顶端，放置了一个主要的类别，名为Bad Speculation，它负责的是由于不正确的预测导致的stalls，以及在不正确的路径上的执行所浪费的资源。这是用户首先关注的问题，也简化了在层次结构中其他部分所使用的硬件计数器。我们引入了几十个真正的自上而下指定的计数器，以处理其他点。我们发现，确定查看哪个流水级，和在有问题的时候计数，在处理(1)和(3)的时候非常重要。比如，我们没有检查总计的内存访问耗时，而是只检查由于挂起的内存访问导致的执行单元利用率低的耗时部分。不和专用的miss events绑定，而是和通用events绑定，这样我们可以处理(4)。一些是occupancy events（可以在给定的周期中，如果满足多个实体条件，可以递增超过1），可以处理(5)。

## 3. Top-down Analysis

Top-Down Analysis methodology aims to determine performance bottlenecks correctly and quickly. It guides users to focus on issues that really matter during the performance optimization phase. This phase is typically performed within the time and resources constraints of the overall application development process. Thus, it becomes more important to quickly identify the bottlenecks.

自上而下分析的方法，目标是迅速和正确的确定性能瓶颈。这会引导用户聚焦在性能优化阶段中真正重要的问题上。这个阶段一般是在一些时间和资源约束的情况下进行的，这些约束就是整体应用开发过程的约束。因此，迅速的找到瓶颈，变得更加重要了。

The approach itself is straightforward: Categorize CPU execution time at a high level first. This step flags (reports high fraction value) some domain(s) for possible investigation. Next, the user can drill down into those flagged domains, and can safely ignore all non-flagged domains. The process is repeated in a hierarchical manner until a specific performance issue is determined or at least a small subset of candidate issues is identified for potential investigation.

方法本身是很直接的：首先在高层次上对CPU的执行时间进行分类。这一步骤将一些领域进行标记（给出高分数值），用于可能的研究。下一步，用户对这些标记的区域进行进一步调查，安全的忽略所有非标记的区域。这个过程以层次化的方式重复进行，直到确定特定的性能问题，或对可能的调查确定候选问题的小的子集。

In this section we first overview the hierarchy structure, and then present the heuristics behind the higher levels of the hierarchy.

在本节中，我们首先对层次化结构进行概览，然后给出层次化结构的高层背后的直觉推断。

### 3.1. The Hierarchy

The hierarchy is depicted in Figure 2. First, we assume the user has predefined criteria for analysis. For example, a user might choose to look at an application’s hotspot where at least 20% of execution time is spent. Another example is to analyze why a given hotspot does not show expected speedup from one hardware generation to another. Hotspot can be a software module, function, loop, or a sequence of instructions across basic blocks.

层次化结构如图2所示。首先，我们假设用户已经预定义了分析的规则。比如，用户会选择来查看应用的热点，至少20%的执行时间花在了这个热点上。另一个例子是，分析一个给定的热点，在硬件从一代到另一代时，为什么没有得到期望的加速。热点可以是一个软件模块，函数，循环，或多个basic blocks的指令序列。

Top-Down breakdown is applied to the interesting hotspots where available pipeline slots are split into four basic categories: Retiring, Bad Speculation, Frontend Bound and Backend Bound. These terms are defined in the following subsections. The best way to illustrate this methodology is through an example. Take a workload that is limited by the data cache performance. The method flags Backend Bound, and Frontend Bound will not be flagged. This means the user needs to drill down at the Backend Bound category as next step, leaving alone all Frontend related issues. When drilling down at the Backend, the Memory Bound category would be flagged as the application was assumed cache-sensitive. Similarly, the user can skip looking at non-memory related issues at this point. Next, a drill down inside Memory Bound is performed. L1, L2 and L3-Bound naturally break down the Memory Bound category. Each of them indicates the portion the workload is limited by that cache-level. L1 Bound should be flagged there. Lastly, Loads block due to overlap with earlier stores or cache line split loads might be specific performance issues underneath L1 Bound. The method would eventually recommend the user to focus on this area.

自上而下的分解应用到感兴趣的热点上，其中可能的流水线槽要分解成4个基本的类别：退休，坏的投机，前端限制和后端限制。这些术语在随后的小节中定义。描述这种方法的最好方法是通过一个例子。以一个workload，受到数据cache的性能的限制为例。该方法会标记后端限制，前端限制则不会被标记。这意味着用户下一步需要进一步研究后端限制的类别，而不用管前端相关的问题。当进一步研究后端时，内存限制的类别会被标记，因为应用假设是对cache敏感的。类似的，用户在这一点上会跳过查看非内存相关的问题。下一步，会被内存限制进行进一步的研究。L1，L2和L3限制是内存限制类别很自然的分解。这其中每一个，都表明了workload的这一部分是受这个cache级别限制的。L1限制应当被标记。最后，由于与早期的store或cache line split loads的叠加导致的Loads阻塞，可能是L1限制下的具体性能问题。该方法最后会推荐用户聚焦在这个区域中。

Note that the hierarchical structure adds a natural safety net when looking at counter values. A value of an inner node should be disregarded unless nodes on the path from the root to that particular node are all flagged. For example, a simple code doing some divide operations on a memory-resident buffer may show high values for both Ext. Memory Bound and Divider nodes in Figure 2. Even though the Divider node itself may have high fraction value, it should be ignored assuming the workload is truly memory bound. This is assured as Backend.CoreBound will not be flagged. We refer to this as hierarchical-safety property. Note also that only weights of sibling nodes are comparable. This is due to the fact they are calculated at same pipeline stage. Comparing fractions of non-sibling nodes is not recommended.

注意，层次化结构在查看计数器的值时，很自然的加入了安全网。内部节点的值，只有在从根节点到那个特定节点的路径都被标记时，才会被承认。比如，在一个memory-resident的buffer上进行一些除法操作的代码，在图2中的外部存储限制和Divider节点上都会显示出很高的值。假设workload真是内存限制的，即使Divider节点有很高的值，也应当被忽略。因为Backend.CoreBound不会被标记，所以这一点也会得到确认。我们称这为层次化安全的属性。注意只有sibling节点的权重是可比较的。这是因为，他们是在相同的流水线级进行计算的。不推荐对非sibling节点进行比较。

### 3.2. Top Level breakdown

There is a need for first-order classification of pipeline activity. Given the highly sophisticated microarchitecture, the first interesting question is how and where to do the first level breakdown? We choose the issue point, marked by the asterisk in Figure 1, as it is the natural border that splits the frontend and backend portions of machine. It enables a highly accurate Top-Level classification.

有一种需求，对流水线行为进行一阶分类。由于微架构非常复杂，第一个有趣的问题是，一级分解在哪里进行，怎样进行？我们选择在图1中标为星号的问题点，因为这是将机器分为前端和后端的自然界限。这就可以进行高度准确的高层次分类。

At issue point we classify each pipeline-slot into one of four base categories: Frontend Bound, Backend Bound, Bad Speculation and Retiring, as illustrated by Figure 3. If a uop is issued in a given cycle, it would eventually either get retired or cancelled. Thus it can be attributed to either Retiring or Bad Speculation respectively.

在问题点，我们将每个流水线槽分为四类中的一类：前端限制，后端限制，坏的投机，和退休，如图3所示。如果一个uop在给定的周期中发射，则最终会retire或取消。因此可以归因为retiring或坏的投机。

Otherwise it can be split into whether there was a backend- stall or not. A backend-stall is a backpressure mechanism the Backend asserts upon resource unavailability (e.g. lack of load buffer entries). In such a case we attribute the stall to the Backend, since even if the Frontend was ready with more uops it would not be able to pass them down the pipeline. If there was no backend-stall, it means the Frontend should have delivered some uops while the Backend was ready to accept them; hence we tag it with Frontend Bound. This backend- stall condition is a key one as we outline in FetchBubbles definition in next section.

否则，就可以分为是否有后端的stall。后端的stall是一种反压机制，在资源不可用（如，缺少load buffer entries）时后端会反压。在这种情况下，我们将stall归因到后端，因为即使前端准备好了更多的uops，也不能将其传递到流水线上。如果没有后端stall，这就意味着前端应当发送一些uops，而后端已经准备好接收了；因此我们标记为前端限制。这种后端stall条件是很关键的，我们在下一节中会描述FetchBubbles的定义。

In fact the classification is done at pipeline slots granularity as a superscalar CPU is capable of issuing multiple uops per cycle. This makes the breakdown very accurate and robust which is a necessity at the hierarchy’s top level. This accurate classification distinguishes our method from previous approaches in [1,5,6].

实际上，分类是在流水线槽的粒度进行的，因为超标量CPU是可以每周期发射多个uops的。这就使得分解是非常精确和稳健的，这在层次化结构的最高层是非常必要的。这种准确的分类，使我们与之前的方法有所区分。

### 3.3. Frontend Bound category

Recall that Frontend denotes the first portion of the pipeline where the branch predictor predicts the next address to fetch, cache lines are fetched, parsed into instructions, and decoded into micro-ops that can be executed later by the Backend. Frontend Bound denotes when the frontend of the CPU undersupplies the backend. That is, the latter would have been willing to accept uops.

回忆一下，前端代表流水线的第一部分，分支预测器预测要取的下一个地址，取cache lines，解析成指令，解码成micro-ops，后端可以进一步执行这些micro-ops。前端限制表示CPU的前端不能有效的供应后端。即，后端会处于准备好接收uops的状态。

Dealing with Frontend issues is a bit tricky as they occur at the very beginning of the long and buffered pipeline. This means in many cases transient issues will not dominate the actual performance. Hence, it is rather important to dig into this area only when Frontend Bound is flagged at the Top-Level. With that said, we observe in numerous cases the Frontend supply bandwidth can dominate the performance, especially when high IPC applies. This has led to the addition of dedicated units to hide the fetch pipeline latency and sustain required bandwidth. The Loop Stream Detector as well as Decoded I-cache (i.e. DSB, the Decoded-uop Stream Buffer introduced in Sandy Bridge) are a couple examples from Intel Core [7].

处理前端问题会有些麻烦，因为这发生在很长的buffered流水线的开始。这意味着，在很多情况下，瞬态的问题不会主宰实际的性能，因此，深入研究这个区域，只有在前端限制在最高层被标记的时候，才会非常重要。我们在很多情况下都观察到，前端供应贷款会主宰性能，尤其是在高IPC的时候。这就要加入专用的单元，隐藏fetch流水线的延迟，维护需要的带宽。Intel Core中的几个例子包括，Loop Stream Detector，和Decoded I-cache（即，Sandy Bridge中引入的Decoded-uop Stream Buffer）。

Top-Down further distinguishes between latency and bandwidth stalls. An i-cache miss will be classified under Frontend Latency Bound, while inefficiency in the instruction decoders will be classified under Frontend Bandwidth Bound. Ultimately, we would want these to account for only when the rest of pipeline is likely to get impacted, as discussed earlier.

自上而下的分析，进一步将延迟stalls和带宽stalls区分开来。一个I-cache miss会被分类为前端延迟限制，而指令译码器的效率不高，会被分类为前端带宽限制。最终，我们会在流水线的其他部分很有可能受到影响的时候，才计入这个部分。

Note that these metrics are defined in Top-Down approach; Frontend Latency accounts for cases that lead to fetch starvation (the symptom of no uop delivery) regardless of what has caused that. Familiar i-cache and i-TLB misses fit here, but not only these. For example, [4] has flagged Instruction Length Decoding as a fetch bottleneck. It is CPU- specific, hence not shown in Figure 2. Branch Resteers accounts for delays in the shadow of pipeline flushes e.g. due to branch misprediction. It is tightly coupled with Bad Speculation (where we elaborate on misprediction costs).

注意，这些度量被定义为自上而下的方法；前端延迟负责导致fetch starvation的情况（没有uop delivery的症状），而不管什么导致了这种情况。熟悉的i-cache和i-TLB就是这种情况，但是不止这些。比如，[4]将Instruction Length Decoding标记为fetch瓶颈。这是在特定CPU中才有的，所以没有在图2中展示。Branch Resteers负责流水线冲刷导致的延迟，如，由于分支误预测。这是与坏的投机紧密相关的（这里我们认为是误预测的代价）。

The methodology further classifies bandwidth issues per fetch-unit inserting uops to the uops-ready-queue. Instruction Decoders are commonly used to translate mainstream instructions into uops the rest of machine understands - That would be one fetch unit. Also sophisticated instruction, like CPUID, typically have dedicated unit to supply long uop flows. That would be 2nd fetch unit and so on.

这种方法进一步将每个fetch-unit插入到uops-ready-queue的带宽问题进行分类。指令译码器通常用于将主流指令翻译成剩下的机器理解的uops，这是一个fetch unit。而复杂的指令，如CPUID，一般有专用的单元来提供长的uop flows。这就是2nd fetch unit，等等。

### 3.4. Bad Speculation category

Bad Speculation reflects slots wasted due to incorrect speculations. These include two portions: slots used to issue uops that do not eventually retire; as well as slots in which the issue pipeline was blocked due to recovery from earlier miss- speculations. For example, uops issued in the shadow of a mispredicted branch would be accounted in this category. Note third portion of a misprediction penalty deals with how quick is the fetch from the correct target. This is accounted in Branch Resteers as it may overlap with other frontend stalls.

坏的投机反应的是由于错误的投机浪费的slots。这包含两个部分：用于发射最终没有retire的uops的slots；以及由于从之前的误投机中恢复，导致的发射流水线的阻塞。比如，误预测的分支发射的uops会被归类到这个类别。注意，误预测惩罚的第三部分，处理的是从正确的目标中的fetch的速度。这归类在Branch Resteers，因为这可能与其他的前端stalls会重叠。

Having Bad Speculation category at the Top-Level is a key principle in our Top-Down Analysis. It determines the fraction of the workload under analysis that is affected by incorrect execution paths, which in turn dictates the accuracy of observations listed in other categories. Furthermore, this permits nodes at lower levels to make use of some of the many traditional counters, given that most counters in out-of-order CPUs count speculatively. Hence, a high value in Bad Speculation would be interpreted by the user as a “red flag” that need to be investigated first, before looking at other categories. In other words, assuring Bad Speculation is minor not only improves utilization of the available resources, but also increases confidence in metrics reported throughout the hierarchy.

在我们的自上而下的分析中，在最高层有坏的投机的类别，是一个关键的原则。这确定了分析中的workload，受到不正确的执行路径影响的那部分，然后决定了其他类别中列出的观察的准确率。而且，这使得更底层的节点可以利用很多传统的计数器，在乱续CPUs中，多数计数器是推测计数的。因此，坏的投机的值很高，用户会解释成一个红色标记，需要首先进行研究，然后才进行其他类别的研究。换句话说，确保坏的投机分数很低，不仅可以改进可用资源的利用，而且会增加在整个层次结构中中给出的度量的信心。

The methodology classifies the Bad Speculation slots into Branch Misspredict and Machine Clears. While the former is pretty famous, the latter results in similar symptom where the pipeline is flushed. For example, incorrect data speculation generated Memory Ordering Nukes [7] - a subset of Machine Clears. We make this distinction as the next steps to analyze these issues can be completely different. The first deals with how to make the program control flow friendlier to the branch predictor, while the latter points to typically unexpected situations.

该方法将坏的投机的slots归类到分支误预测和机器清除。前者是很著名的，后者也会得到类似的症状，导致流水线的冲刷。比如，不正确的数据投机会生成Memory Ordering Nukes，这是Machine Clears的一个子集。我们做出这种区分，因为分析这些问题的下面的步骤，是非常不一样的。前者处理的是，使程序控制流对分支预测器更加友好，而后者指向的一般是非预期的情况。

### 3.5. Retiring category

This category reflects slots utilized by “good uops” – issued uops that eventually get retired. Ideally, we would want to see all slots attributed to the Retiring category; that is Retiring of 100% corresponds to hitting the maximal uops retired per cycle of the given microarchitecture. For example, assuming one instruction is decoded into one uop, Retiring of 50% means an IPC of 2 was achieved in a four-wide machine. Hence maximizing Retiring increases IPC.

这个类别反应的是好的uops利用的slots，即最终retire的发射的uops。理想情况下，我们希望所有slots都归属于Retiring类别；这就是100%的Retiring，对应着给定架构下最大的每周期retired uops。比如，假设一条指令解码成一个uop，50%的Retiring意味着，在宽度为4的机器中，可以达到IPC为2。因此，最大化Retiring会增大IPC。

Nevertheless, a high Retiring value does not necessary mean there is no room for more performance. Microcode sequences such as Floating Point (FP) assists typically hurt performance and can be avoided [7]. They are isolated under Micro Sequencer metric in order to bring it to user’s attention.

尽管如此，Retiring值很高，并不一定意味着性能没有改进的空间了。微码序列如浮点(FP)一般会伤害性能，而且是可以避免的[7]。它们孤立在Micro Sequencer度量下，以得到用户的注意。

A high Retiring value for non-vectorized code may be a good hint for user to vectorize the code. Doing so essentially lets more operations to be completed by single instruction/uop; hence improve performance. For more details see Matrix-Multiply use-case in Section 5. Since FP performance is of special interest in HPC land, we further breakdown the base retiring category into FP Arithmetic with Scalar and Vector operations distinction. Note that this is an informative field-originated expansion. Other styles of breakdown on the distribution of retired operations may apply.

对非向量化的代码，高Retiring值很可能意味着用户可以对代码进行向量化。这样做会在一条指令/uop中完成更多操作；因此改进性能。更多细节详见Matrix-Multipy的使用案例。由于FP性能在HPC中具有特殊的兴趣，我们进一步将基准retiring类别分解成FP Arithmetic，并区分Scalar和Vector运算。注意，这是一个informative field-originated expansion。对retired运算，其他类型的分解，也是可行的。

### 3.6. Backend Bound category

Backend Bound reflects slots no uops are being delivered at the issue pipeline, due to lack of required resources for accepting them in the backend. Examples of issues attributed in this category include data-cache misses or stalls due to divider being overloaded.

后端限制反应的是在issue流水线上没有deliver uops的slots，由于缺少在后端接收uops的必须的资源。归属在这个类别的问题的例子包括，data-cache misses，或由于divider过载导致的stalls。

Backend Bound is split into Memory Bound and Core Bound. This is achieved by breaking down backend stalls based on execution units’ occupation at every cycle. Naturally, in order to sustain a maximum IPC, it is necessary to keep execution units busy. For example, in a four-wide machine, if three or less uops are executed in a steady state of some code, this would prevent it to achieve a optimal IPC of 4. These suboptimal cycles are called ExecutionStalls.

后端限制分为Memory限制和Core限制。将后端的stalls，基于执行单元在每个周期的占用情况进行分解，就可以得到。自然的，为维持最大IPC值，有必要保持执行单元繁忙。比如，在宽度为4的机器中，如果在某代码中的稳态执行为3个或更少的uops，这就不能得到4的最佳IPC。这种次优的周期称为ExcutionStalls。

Memory Bound corresponds to execution stalls related to the memory subsystem. These stalls usually manifest with execution units getting starved after a short while, like in the case of a load missing all caches.

内存限制对应着与内存子系统相关的执行stalls。这些stalls通常在执行单元饥饿的状态一小段时间后表现出来，比如load missing all caches的情况。

Core Bound on the other hand, is a bit trickier. Its stalls can manifest either with short execution starvation periods, or with sub-optimal execution ports utilization: A long latency divide operation might serialize execution, while pressure on execution port that serves specific types of uops, might manifest as small number of ports utilized in a cycle. Actual metric calculations is described in Section 4.

另一方面，Core限制更加复杂一点。这些stalls会在短暂的执行饥饿期间表现出来，或在非最优的执行ports利用的情况下表现出来：一个长延迟的除法操作会将执行序列化，而对特定类型的uops服务的执行port的压力，会表现出一个周期中利用的ports数量较小。第4部分描述实际的度量计算。

Core Bound issues often can be mitigated with better code generation. E.g., a sequence of dependent arithmetic operations would be classified as Core Bound. A compiler may relieve that with better instruction scheduling. Vectorization can mitigate Core Bound issues as well; as demonstrated in Section 5.5.

Core限制问题通常会用更好的代码生成缓解。如，互相以来的代数运算的序列，会归类为Core限制。一个编译器会用更好的代码调度进行缓解。向量化也会缓解Core限制；如5.5节所示。

### 3.7 Memory Bound breakdown (within Backend)

Modern CPUs implement three levels of cache hierarchy to hide latency of external memory. In the Intel Core case, the first level has a data cache (L1D). L2 is the second level shared instruction and data cache, which is private to each core. L3 is the last level cache, which is shared among sibling cores. We assume hereby a three-cache-level hierarchy with a unified external memory; even though the metrics are generic-enough to accommodate other cache- and memory- organizations, including NUMA.

现代CPUs实现了三级cache层次结构，以隐藏外部存储的延迟。在Intel Core的情况中，第一级有一个数据cache (L1D)。L2是第二级，数据cache和指令cache共享，是每个核私有的。L3是最后一级cache，是在sibling cores中共享的。我们假设三级cache层次结构，和统一的外部存储；即使这样，度量是非常通用的，可以容纳其他的cache组织和内存组织，包括NUMA。

To deal with the overlapping artifact, we introduce a novel heuristic to determine the actual penalty of memory accesses. A good out-of-order scheduler should be able to hide some of the memory access stalls by keeping the execution units busy with useful uops that do not depends on pending memory accesses. Thus the true penalty for a memory access is when the scheduler has nothing ready to feed the execution units. It is likely that further uops are either waiting for the pending memory access, or depend on other unready uops. Significant ExecutionStalls while no demand-load (Hardware prefetchers are of special treatment. We disregard them as long as they were able to hide the latency from the demand requests) is missing some cache-level, hints execution is likely limited by up to that level itself. Figure 4 also illustrates how to break ExecutionStalls per cache-level.

为处理叠加的效果，我们引入一个新的直觉，来决定内存访问的实际惩罚。一个好的乱序调度器，应该可以通过用有用的uops保持执行单元繁忙，不依赖于悬停的内存访问，来隐藏一些内存访问的stalls。因此，内存访问的真实惩罚是，调度器没有任何东西准备好送入执行单元中。很可能的情况是，更多的uops要么在等待挂起的内存访问，或依赖于其他未准备好的uops。在demand-load都没有miss cache级的时候却有显著的ExecutionStalls，说明执行很可能是由那个级别本身限制住的。图4描述了在每个cache级中分解ExecutionStalls。

For example, L1D cache often has short latency which is comparable to ALU stalls. Yet in certain scenarios, like load blocked to forward data from earlier store to an overlapping address, a load might suffer high latency while eventually being satisfied by L1D. In such scenario, the in-flight load will last for a long period without missing L1D. Hence, it gets tagged under L1 Bound per flowchart in Figure 4. Load blocks due to 4K Aliasing [7] is another scenario with same symptom. Such scenarios of L1 hits and near caches’ misses, are not handled by some approaches [1][5].

比如，L1D cache通常延迟很低，与ALU的stalls可比。在但特定的场景中，比如load被之前的对重叠地址的store所阻塞，这个load可能会有很大的延迟，最终也会被L1D满足。在这种场景中，in-flight load会持续很长时间，而不会miss L1D。因此，在图4中会被标上L1限制。由于4K Aliasing导致的Load blocks，是另一种场景，也有相同的症状。这种L1 hits和接近cache的misses的场景，在一些方法中是没有处理的。

Note performance hiccups, as the mentioned L1 Bound scenarios, would appear as leaf-nodes in the hierarchy in Figure 2. We skipped listing them due to scope limitation.

注意，性能晓问题，如提到的L1限制场景，是图2中的叶节点。由于scope的限制，我们不再列出这些。

So far, load operations of the memory subsystem were treated. Store operations are buffered and executed post- retirement (completion) in out-of-order CPUs due to memory ordering requirements of x86 architecture. For the most part they have small impact on performance (as shown in results section); they cannot be completely neglected though. Top- Down defined Stores Bound metric, as fraction of cycles with low execution ports utilization and high number of stores are buffered. In case both load and store issues apply we will prioritize the loads nodes given the mentioned insight.

迄今，处理了内存子系统的load操作。在乱序CPU中，store操作是buffered，在retire后进行执行，因为x86架构有memory ordering需要。在大部分情况下，其对性能的影响很小；但是，也不能完全被忽视。自上而下定义的Stores限制的度量，因为低执行ports利用和store值很高的部分周期被buffer了。在load和store问题都可以应用的情况下，我们会优先load节点。

Data TLB misses can be categorized under Memory Bound sub-nodes. For example, if a TLB translation is satisfied by L1D, it would be tagged under L1 Bound.

数据TLB misses可以归类到Memory Bound的子节点。比如，如果一个TLB translation被L1D满足，这会被标记为L1 Bound。

Lastly, a simplistic heuristic is used to distinguish MEM Bandwidth and MEM Latency under Ext. Memory Bound. We measure occupancy of requests pending on data return from memory controller. Whenever the occupancy exceeds a certain threshold, say 70% of max number of requests the memory controller can serve simultaneously, we flag that as potentially limited by the memory bandwidth. The remainder fraction will be attributed to memory latency.

最后，在区分外部内存限制下的内存带宽和内存延迟中，我们使用了一个简单的直觉。我们测量了挂起的从内存控制器的数据返回的请求占用。只要这个占用超过了一定的阈值，比如70%的最大数量的请求，我们就标记为很可能是内存带宽的限制。剩余的部分，我们将其归因为内存延迟。

## 4. Counters Architecture

This section describes the hardware support required to feature the described Top-Down Analysis. We assume a baseline PMU commonly available in modern CPU (e.g. x86 or ARM). Such a PMU offers a small set of general counters capable of counting performance events. Nearly a dozen of events are sufficient to feature the key nodes of the hierarchy. In fact, only eight designated new events are required. The rest can be found in the PMU already today – these are marked with asterisk in Table 1. For example, TotalSlots event can be calculated with the basic Clockticks event. Additional PMU legacy events may be used to further expand the hierarchy, thanks to the hierarchical-safety property described in Section 3.

本节描述了要进行上述的自上而下的分析，需要的硬件支持。我们假设在现代CPU（如，x86或ARM）中有基准PMU。这样一个PMU有若干个通用计数器，可以对性能事件进行计数。十几个事件足以满足层次结构中的关键节点。实际上，只需要8个指定的新事件。剩余的可以在已有的PMU中找到，这在表1中标记为星号。比如，TotalSlots事件可以用基本的Clockticks事件来计算得到。额外的PMU传统事件可以用于进一步拓展这个层次结构，第3部分中描述的层次结构安全性质会起到很好的作用。

It is noteworthy that a low-cost hardware support is required. The eight new events are easily implementable. They rely on design local signals, possibly masked with a stall indication. Neither at-retirement tagging is required as in IBM POWER5 [6], nor complex structures with latency counters as in Accurate CPI Stacks proposals [1][8][9].

需要指出的是，需要低代价的硬件支持。8个新事件是很容易实现的。它们依赖于设计的局部信号，可能用stall indication进行掩膜。不需要在retire的时候的标记（就像在IBM POWER5中一样），也不需要传统计数器中的复杂结构（就像在Accurate CPI Stacks建议中）。

### 4.1. Top-Down Events

The basic Top-Down generic events are summarized in Table 1. Please refer to Appendix 1 for the Intel implementation of these events. Notice there, an implementation can provide simpler events and yet get fairly good results.

基本的自上而下的通用事件如表1所总结。参考附录1中的这些事件的Intel实现。注意，一种实现可以给出更简单的事件，同时得到相当好的结果。

### 4.2. Top-Down Metrics

The events in Table 1 can be directly used to calculate the metrics using formulas shown in Table 2. In certain cases, a flavor of the baseline hardware event is used. Italic #-prefixed metric denotes an auxiliary expression.

表1中的事件，可以直接用于计算表2中的公式，得到度量。在特定的情况中，也使用了基准硬件事件。斜体度量表示辅助表达式。

Note ExecutionStall denotes sub-optimal cycles in which no or few uops are executed. A workload is unlikely to hit max IPC in such case. While these thresholds are implementation- specific, our data suggests cycles with 0, 1 or 2 uops executed are well-representing Core Bound scenarios at least for Sandy Bridge-like cores.

注意，ExecutionStall表示非最优的周期，其中没有执行uops，或执行的很少。在这种情况下，一个workload很难得到最大的IPC。这些阈值是实现特定的，我们的数据表明，只执行了0，1，2个uops的周期，可以很好的表示Core Bound的场景，在Sandy Bridge类的cores至少是这样的情况。

## 5. Results

In this section, we present Top-Down Analysis results for the SPEC CPU2006 benchmarks in single-thread (1C) and multi-copy (4C) modes with setup described in Table 3. Then, an across-CPUs study demonstrates an architecture exploration use-case. As Frontend Bound tends to be less of a bottleneck in CPU2006, results for key server workloads are included. Lastly, we share a few use-cases where performance issues are tuned using Top-Down Analysis.

在本节中，我们给出SPEC CPU2006 benchmark中给出自上而下的分析结果，硬件设置如表3所示，模式为单线程(1C)和multi-copy(4C)模式。然后，跨CPU的研究，展示了架构探索的使用案例。由于在CPU2006中的前端限制很少是瓶颈，我们还给出了关键的服务器workloads的结果。最后，我们给出了一些使用案例，其中性能问题是用自上而下的分析来调整的。

### 5.1. SPEC CPU2006 1C

At the Top Level, Figure 5a suggests diverse breakdown of the benchmark’s applications. Performance wise, the Retiring category is close to 50% which aligns with aggregate Instruction-Per-Cycle (IPC) of ~1.7 measured for same set of runs. Recall 100% Retiring means four retired uops-per-cycle while for SPEC CPU2006 an instruction is decoded into slightly more than one uop on average. Note how Retiring correlates well with IPC, included to cross-validate with an established metric.

在最顶层，图5a给出了benchmark应用的各种分解。在性能方面，Retiring类别接近50%，这与累积IPC对其，对相同的运行，大概有～1.7的数值。回忆一下，100%的Retiring意味着每个周期retire 4个uops，而对于SPEC CPU 2006，一条指令平均译码为略多于1条。注意，Retiring是怎样与IPC相关联的，与得到的度量可以互相验证。

Overall Backend Bound is dominant. So we drill down into it in next diagrams in Figure 5. The Backend Level diagram guides the user whether to look at Core or Memory issues next. For example, 456.hmmer is flagged as Backend.CoreBound. Close check of the top hotspots with VTune, indeed points to loops with tight data-dependent arithmetic instructions.

整体上说，后端限制是占主体的。所以我们进行了进一步研究，如图5所示。后端Level图引导用户，下面是查看Core问题，还是Memory问题。比如，456.hmmer标记为Beckend.CoreBound。用VTune仔细检查最高的热点，确实指向了带有数据依赖的代数指令的循环。

The Integer applications are more sensitive to Frontend Bound and Bad Speculation than the FP applications. This aligns with simulations data using a propriety cycle-accurate simulator, as well as prior analysis by Jaleel [11]. For example, Jaleel’s analysis reported that gcc, perlbench, xalancbmk, gobmk, and sjeng have code footprint bigger than 32KB. They are classified as most Frontend Bound workloads. Note how the breakdown eases to assess the relative significance of bottlenecks should multiple apply.

与FP应用比起来，Integer应用对前端限制和坏的投机更加敏感。这与采用自有的周期精确的仿真器的仿真数据是对齐的，与之前的分析[11]也是对齐的。比如，[11]的分析表明，gcc, perlbench, xalancbmk, gobmk, 和sjeng的代码占用超过了32KB。它们被归类为前端限制最多的workloads。注意，自上而下的分解使得评估瓶颈的相对显著性容易了很多。

### 5.2. SPEC CPU2006 4C

Results running 4-copies of these applications are shown in Figure 6. Top Level shows similarity to 1-copy. At a closer look, some applications do exhibit much increased Backend Bound. These are memory-sensitive applications as suggested by bigger Memory Bound fractions in Figure 6b. This is expected as L3 cache is “shared” among cores. Since an identical thread is running alone inside each physical core and given CPU2006 has minor i-cache misses, Frontend Bound and Bad Speculation in 4-copy roughly did not changed over 1-copy.

图6给出了运行这些应用的4份副本的结果。最顶级的特性与1-copy是类似的。近距离观察一下，一些应用展现出了更多的后端限制。图6b表明，有一些内存敏感的应用，内存限制的部分更多。这是符合预期的，因为L3 cache是在多核共享的。由于每个物理核心都在运行相同的线程，而CPU2006的i-cache miss会更少，前端限制和坏的投机在4-copy中与1-copy中是类似的。

For the less-scalable applications, Memory Bound breakdown points to off-core contention when comparing Figure 6c to 5c. The key differences occur in applications that are either (I) sensitive to available memory bandwidth, or (II) impacted by shared cache competition between threads. An example of (I) is 470.lbm which is known for its high memory bandwidth requirements [12]. Its large MEM Bound is the primary change between 1- and 4-copy.

对那些不那么可扩展的应用，Memory Bound更多的将点分解到off-core竞争。关键的差异所涉及的应用，(I)对可用的内存带宽是敏感的，(II)受到线程间的共享cache竞争影响。(I)的一个例子是470.lbm，对内存带宽的需求较高。大的MEM Bound是1-copy和4-copy的主要变化。

A key example of (II) is 482.sphinx3. A close look at Memory Bound breakdown indicates the 4-copy sees reduced L3 Bound, and a greatly increased MEM Bound; capacity contention between threads in the shared L3 cache has forced many more L3 misses. This conclusion can be validated by consulting the working-set of this workload [11]: a single copy demands 8MB (same as LLC capacity) in 1-copy, vs 2MB effective per-core LLC share in 4-copy runs.

(II)的一个关键例子是482.sphinx3。Memory Bound分解的近距离观察说明，4-copy的L3 Bound减少了，而MEM Bound则极大的增加了；在共享L3 cache中的线程间的能力竞争，迫使更多的L3 misses。这个结论可以通过查看这个workload的working-set来验证：一个copy需要8MB（与LLC的容量一样），而在4-copy中，每个core的LLC则只需要2MB。

Figure 7 shows how off-chip resources are utilized for some FP applications, with 1- and 4-copy side-by-side. The bars’ height indicates fraction of run time where the memory controller is serving some request. “MEM Bandwidth” is the relative portion where many requests are being serviced simultanously. Note we could plot these metrics at their native local units, thanks to the hierarchical-safety property. We should consider them carefully though.

图7展示了对一些FP应用来说，怎样利用片外资源。bar的高度表示了运行时间的比例，内存控制器在伺服一些请求。MEM带宽是相对部分，其中很多请求在同时被服务。注意，我们可以在其本地局部单元画这些度量，多亏了层次结构安全的性质。但是我们也应当仔细的考虑这些。

The available 25GB/s bandwidth clearly satisfies demand of 1-copy. The picture changes in 4-copy in different ways. 435.gromacs, 447.dealII, 454.calculix and 465.tonto now spend more memory cycles due to increase of 1.3-3.6x in L3 misses per-kilo instructions as measured by distinct set of performance counters. Note however, they showed on-par Memory- and Core-Bound stall fractions in Figure 6b, likely because the out-of-order could mitigate most of these memory cycles. This aligns with measured IPC in range of 1.7-2.3 in 4- copy. In contrast, 410.bwaves, 433.milc, 437.leslie3d and 470.lbm become much more MEM Bound in 4-copy per Figure 6c. Figure 7 tells us that was due to memory latency in 1-copy which turns into memory bandwidth in 4-copy (4x data demand). Top-Down correctly classifies 470.lbm as MEM Bandwidth limited [12].

可用的25GB/s的带宽，很明显可以满足1-copy的需求。在4-copy中，图有了多种变化。435.gromacs, 447.dealII, 454.calculix和465.tonto现在花费更多的内存周期，因为每千条指令的L3 misses增加了1.3-3.6x，这是不同的性能计数器得到的测量结果。但是，在图6b中，它们展现出了类似的Memory限制和Core限制的stall部分，很可能是因为乱序能够弥补大部分这些内存周期。这与4-copy情况下测量得到的IPC，范围为1.7-2.3，结果是对齐的。对比之下，410.bwaves, 433.milc, 437.leslie3d和470.lbm在图6c中的4-copy情况下变得更加MEM Bound。图7告诉我们，由于1-copy下的内存延迟，编程了4-copy下的内存带宽限制（4x的数据需求）。自上而下的分析正确的将470.lbm归类为MEM Bandwidth限制的。

### 5.3. Microarchitectures comparison

So far we have shown results for the same system. This section demonstrates how Top-Down can assist hardware architects. Figure 8 shows Top Level for Intel Core 3rd and 4th generation CPUs, side-by-side for a subset of CPU2006 integer benchmarks. The newer Intel Core has improved frontend where speculative iTLB and i-cache accesses are supported with better timing to improve the benefits of prefetching [7]. This can be clearly noticed for the benefiting benchmarks with reduction in Frontend Bound. This validation adds to the confidence of underlying heuristics invented two generations earlier.

迄今为止，我们对相同的系统给出了结果。本节中，我们展示一下自上而下的分析怎样帮助硬件架构师。图8展示了Intel Core 3rd和4th代CPUs，运行CPU 2006整数benchmarks的子集的Top Level结果。更新的Intel Core改进了前端，投机性iTLB和i-cache访问的时序支持更好，改进了预取的好处。这可以很清除的看到，前端限制的部分明显减少了。这种验证增加了两代前发明的直觉的信心。

### 5.4. Server workloads

Key server workloads’ results on Sandy Bridge EP are shown in Figure 9. Retiring is lower compared to the SPEC workloads, which conform to the lower IPC domain (a range of 0.4 to 1 is measured). Backend- and Frontend-Bound are more significant given the bigger footprints.

图9给出了在Sandy Bridge EP上的关键的服务器workload的结果。与SPEC workload相比，Retiring要更低，这与低IPC是相符合的（测量结果为0.4到1）。后端限制和前端限制更加明显。

It is interesting to see that the characterization of DBMS workloads generally conforms to [4] who reported these workloads are limited by last-level data cache misses and 1st level i-cache misses a while back.

有趣的是，DBMS workloads的特征刻画与[4]是符合的，其给出的结论是，这些workloads是受到最后一级的data cache misses和L1 i-cache限制的。

Within the Frontend, Latency issues are dominant across all server workloads. This is due to more i-cache and i-TLB misses as expected there, in contrast to client workloads whose Frontend Bound was almost evenly split between Latency and Bandwidth issues (not shown due to paper scope limitation).

在前端，延迟问题在所有服务器workloads中占主要部分。这是因为有更多的i-cache和i-TLB misses，与之相比，客户端workloads的前端限制，基本上在延迟和带宽问题上是等分的。

### 5.5. Case Study 1: Matrix-Multiply

A matrix-multiply textbook kernel is analyzed with Top-Down. It demos the iterative nature of performance tuning. The initial code in multiply1() is extremely MEM Bound as big matrices are traversed in cache-unfriendly manner.

用自上而下的分析方法分析了矩阵乘法的核。这展现出了性能调优的迭代本质。初始代码multiply1()是非常MEM限制的，因为大型矩阵的访问对cache是非常不友好的。

Loop Interchange optimization, applied in multiply2() gives big speedup. The optimized code continues to be Backend Bound though now it shifts from Memory Bound to become Core Bound.

Loop Interchange优化在multiply2()中进行了应用，得到了很大的加速。优化的代码还是后端限制的，但是从Memory限制切换到了Core限制。

Next in multiply3(), Vectorization is attempted as it reduces the port utilization with less net instructions. Another speedup is achieved.

下一步在multiply3()中使用了向量化，因为减少了端口利用率，净指令数减少了，得到了另一种加速。

### 5.6. Case Study 2: False Sharing

A university class educates students on multithreading pitfalls through an example to parallelize a serial compute- bound code. First attempt has no speedup (or, a slowdown) due to False Sharing. False Sharing is a multithreading hiccup, where multiple threads contend on different data-elements mapped into the same cache line. It can be easily avoided by padding to make threads access different lines.

大学课堂在教授学生多线程的pitfalls时，是通过一个并行化顺序的计算限制的代码的例子。第一次尝试因为False Sharing没有得到加速。False Sharing是一个多线程的小问题，其中多个线程会在不同的数据元素映射到相同的cache line中竞争。可以通过padding，使线程访问不同的cache lines，得到很容易的避免。

The single-thread code has modest IPC. Top-Down correctly classifies the first multithreaded code attempt as Backend.Memory.StoresBound (False Sharing must have one thread writing to memory, i.e. a store, to apply). Stores Bound was eliminated in the fixed multithreaded version.

单线程的IPC是普通的。自上而下的方法正确的将第一种多线程代码尝试归类为Backend.Memory.StoresBound。Stores限制在修正的多线程版本中得到了消除。

### 5.7. Case Study 3: Software Prefetch

A customer propriety object-recognition real application is analyzed with Top-Down. The workload is classified as Backend.Memory.ExtMemory.LatencyBound at application scope. Ditto for biggest hotspot function; though the metric fractions are sharper there. This is a symptom of more non- memory bottlenecks in other hotspots.

一个用户自有的目标识别真实应用也用自上而下的方法进行了分析。这个workload被分类为Backend.Memory.ExtMemory.LatencyBound。重复了最大的热点函数；虽然度量部分在那里更加尖锐。在其他热点中，更多的是非内存的瓶颈。

Software Prefetches[10] are planted in the algorithm’s critical loop to prefetch data of next iteration. A speedup of 35% per the algorithm-score is achieved, which is translated to 1.21x at workload scope. Note the optimized version shows higher memory-bandwidth utilization and has become more Backend.CoreBound.

软件预取植入到了算法的关键循环，以为下次迭代预取数据。每个算法分数得到了35%的加速，在workload的层次为1.21x的加速。注意，优化版本展现出了更高的内存带宽利用率，已经成为更多的是Backend.CoreBound。

## 6. Related Work

The widely-used naïve-approach is adopted by [4][5] to name a few. While this might work for in-order CPUs, it is far from being accurate for out-of-order CPUs due to: stalls overlap, speculative misses and workload-dependent penalties as elaborated in Sections 2.

[4,5]采用了广泛使用的naive方法。这对于顺序CPUs是好用的，对于乱序CPUs来说则远远不够：stalls重叠，投机misses，与workload相关的惩罚，如第2部分所示。

IBM POWER5 [6] has dedicated PMU events to aid compute CPI breakdown at retirement (commit) stage. Stall periods with no retirement are counted per type of the next instruction to retire and possibly a miss-event tagged to it. Again this is a predefined set of fixed events picked in a bottom-up way. While a good improvement over naïve- approach, it underestimates frontend misses’ cost as they get accounted after the point where the scheduler’s queue gets emptied. Levinthal [5] presents a Cycle Accounting method for earlier Intel Core implementations. A flat breakdown is performed at execution-stage, to decompose total cycles into retired, non-retired and stall components. Decomposition of stall components then uses the inadequate naïve-approach as author himself indicates.

IBM POWER5有专用的PMU事件来在retirement阶段帮助计算CPI分解。Stall期间没有retirement，会对每种类型要retire的下一条指令进行计数，也可能是一个miss事件，对其进行标记。再一次，这是一个预定义的固定事件集，是以自下而上的方式进行选取的。这是对naive方法的很好的改进，但仍然低估了前端的miss的代价，因为在调度器的队列空了以后，就会有代价。[5]提出了一种Cycle Accounting方法用于早期的Intel Core实现。在执行阶段进行了平坦的分解，将所有周期分解成retired，non-retired和stall组成部分。stall部分的分解，使用了不充分的naive方法。

In contrast, Top-Down does breakdown at issue-stage, at finer granularity (slots) and avoids summing-up all penalties into one flat breakdown. Rather it drills down stalls in a hierarchical manner, where each level zooms into the appropriate portion of the pipeline. Further, designated Top- Down events are utilized; sampling (as opposed to counting) on frontend issues is enabled, as well as breakdown when HT is on. None of these is featured by [5].

对比之下，自上而下的方法当然在issue阶段进行了分解，在更细的粒度进行了分解，避免了将所有惩罚加入到一个平坦的分解中。而是将stalls以层次化的方式进行了挖掘，其中每个级别都可以放大到合适的流水线部分。而且，利用了指定的自上而下的事件；对前端的事件进行了采样（而不是计数），在HT开启时进行分解。[5]中则没有这些。

Some researchers have attempted to accurately classify performance impacts on out-of-order architectures. Eyerman et al. in [1][9] use a simulation-based interval analysis model in order to propose a counter architecture for building accurate CPI stacks. The presented results show improvements over naïve-approach and IBM POWER5 in terms of being closer to the reference simulation-based model. A key drawback of this approach (and its reference model) is that it restricts all stalls to a fixed set of eight predefined miss events. In [1][4][5] there is no consideration of (fetch) bandwidth issues, and short-latency bottlenecks like L1 Bound. Additionally, high hardware cost is implied due to fairly complex tracking structures as authors themselves later state in [8]. While [8] replaces the original structure with smaller FIFO; extra logic is required for penalty calculation and aggregation to new dedicated counters. This is in comparison with the simple events adopted by our method with no additional counters/logic. We have pointed to more drawbacks in previous sections.
More recently, [13] and [12] proposed instrumentation- based tools to analyze data-locality and scalability bottlenecks, respectively. In [13], average memory latency is sampled with a PMU and coupled with reuse distance obtained through combination of Pin and a cache simulator, in order to prioritize optimization efforts. An offline analyzer maps these metrics back to source code and enables the user to explore the data in hierarchal manner starting from main function. [12] presents a method to obtain speedup stacks for a specific type of parallel programs, while accounting for three bottlenecks: cache capacity, external memory bandwidth and synchronization.

一些研究者试图准确的对乱序架构中的性能影响进行分类。[1,9]使用基于仿真的区间分析模型，以提出一个计数器架构，构建准确的CPI stacks。给出的结果表明，对naive方法和IBM POWER5方法都有改进，因为更接近参考的基于仿真的模型。这种方法（及其参考模型）的一个关键缺点是，将所有的stalls限制在了固定的预定义的miss事件集合上。[1,4,5]中，没有考虑fetch带宽问题，和短延迟的瓶颈，如L1限制。额外的，[8]中的作者后来指出，由于相对较复杂的追中结构，导致硬件代价较高。[8]将原始结构替换为更小的FIFO；对新的专用计数器，需要额外的逻辑进行惩罚计算和累积。与之相比，我们的方法使用简单的事件，没有额外的计数器/逻辑。我们在之前的节中指出了更多的缺点。最近，[13,12]提出了基于插桩的工具来分别分析数据局部性和扩展性瓶颈。[13]中，一个PMU对平均内存延迟进行了采样，用Pin和一个cache仿真器结合得到了reuse distance，这两者结合，对优化努力的优先级进行排序。一个离线分析器得到这些度量映射回源码，使用户以层次化的方式从main函数中探索数据。[12]提出了一种方法，对特定类型的并行程序得到加速stacks，考虑了三种瓶颈：cache容量，外部内存带宽，和同步效果。

These can be seen as advanced optimization-specific techniques that may be invoked from Top-Down once Backend.MemoryBound is flagged. Furthermore, better metrics based on our MemStalls.L3Miss event e.g. can be used instead of raw latency value in [13] to quantify when speedup may apply. Examining metrics at higher program scope first, may be applied to our method as already done in VTune’s General Exploration view [2]. While [12] estimates speedups (our method does not), it accounts for subset of scalability bottlenecks. For example, the case in ‎5.6 is not be covered by their three bottlenecks.

这些可以视为高级优化特定的技术，在自上而下的分析中，一旦标记了Backend.MemoryBound，就可以调用这些技术。而且，更好的度量，基于我们的MemStalls.L3Miss事件，可以进行使用，而不是[13]中的原始的延迟值。在更高的程序范围内首先检查度量，可以在我们的方法中应用。[12]估计了加速（我们的方法没有），解释了可扩展性瓶颈的子集。比如，5.6中的情况没有被其3个瓶颈覆盖。

## 7. Summary and Future Work

This paper presented Top-Down Analysis method - a comprehensive, systematic in-production analysis methodology to identify critical performance bottlenecks in out-of-order CPUs. Using designated PMU events in commodity multi-cores, the method adopts a hierarchical classification, enabling the user to zero-in on issues that directly lead to sub-optimal performance. The method was demonstrated to classify critical bottlenecks, across variety of client and server workloads, with multiple microarchitectures’ generations, and targeting both single-threaded and multi-core scenarios.

本文提出了自上而下的分析方法，一种综合性的，系统性的分析方法，可以在乱序CPUs中识别关键的性能瓶颈。在商用多核中，使用指定的PMU事件，该方法采用了层次结构分类，使用户可以校准问题，直接引导到非最优的性能。该方法可以归类关键瓶颈，对各种客户机和服务器workloads，对各种微架构，在单线程和多核的场景中都适用。

The insights from this method are used to propose a novel low-cost performance counters architecture that can determine the true bottlenecks of a general out-of-order processor. Only eight simple new events are required.

本方法的洞见用于提出一种新的低代价性能计数器架构，可以确定通用乱序处理器的真实瓶颈。只需要增加8个新的事件。

The presented method raises few points on PMU architecture and tools front. Breakdown of few levels require multiple events to be collected simultaneously. Some techniques might tolerate this; such as Sandy Bridge’s support of up to eight general-purpose counters [10], or event- multiplexing in the tools [2][3]. Still a better hardware support is desired. Additionally, the ability to pinpoint an identified issue back to the user code can benefit much software developers. While PMU precise mechanisms are a promising direction, some microarchitecture areas are under-covered. Yet, enterprise-class applications impose additional challenges with flat long-tail profiles.

提出的方法在PMU架构和工具中提出了很少的点。很少几个层次的分解，需要多个事件同时进行收集。一些技术会容忍这个；比如Sandy Bridge的支持最多8个通用目标计数器，或在工具[2,3]中的事件-multiplexing。仍然需要更好的硬件支持。另外，将识别的问题定位到用户代码中，会使很多软件开发者受益。PMU精确的机制是一个有希望的方法，一些微架构领域有待覆盖。企业级的应用有平坦的长尾profiles，这提出了更多挑战。

Correctly classifying bottlenecks in the context of hardware hyper-threading (HT) is definitely a challenging front. While it was beyond the scope of this paper, the design of some Top Down events, does take HT into account, letting the Top Level works when HT is enabled; but that is just the start. Lastly, While the goal of our method was to identify critical bottlenecks, it does not gauge the speedup should underlying issues be fixed. Generally, even to determine whether an issue-fix will be translated into speedup (at all) is tricky. A workload often moves to the next critical bottleneck. [12] has done nice progress to that end in scalability bottlenecks context.

在硬件超线程中，正确的分类瓶颈，肯定是一个有挑战的前沿。这超出了本文的范畴，但一些自上而下的事件的设计，确实考虑了HT，当HT开启时，Top Level是会起作用的；但这仅仅是开始。最后，我们的方法的目标是识别关键的瓶颈，但潜在的问题解决掉后，并没有估算加速。一般的，是否解决一个问题会成为加速，这本身都是一个很麻烦的问题。一个workload通常会转移到下一个关键瓶颈上。[12]在可扩展瓶颈的上下文中，已经有了很好的进展。