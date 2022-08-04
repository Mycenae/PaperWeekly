# ICCAD CAD Contest 2022 – “Microarchitecture Design Space Exploration”

Sicheng Li, Xuechao Wei, Bizhao Shi, Yen-Kuang Chen, Yuan Xie

Alibaba Group Inc

## 1. Introduction

Modern chip development requires high cost in design time and workforce. The reason for the expensive chip development cycle lies in two folds. On the one hand, the pre-defined performance, power, and area (PPA) targets of the chip are set aggressively, hoping to deliver the following generation product comparable and superior to market competitors. On the other hand, the design complexity of the chip (number of gates/chip area) is continuously increasing, leaving the improvement in design capability (number of gates/staff × month) behind a considerable margin, causing an imbalance between the required design efforts and the cost of input.

现代芯片开发的设计时间和工作量代价很高。芯片开发周期代价大的原因有两个方面。一方面，芯片预定义的性能、功耗和面积(PPA)目标设定的很高，希望交付与市场上竞争者相似或更好的产品。另一方面，芯片的设计复杂度（门数/芯片面积）持续增加，使设计能力的改进有很大空白，导致需要的设计努力和输入代价有很大的不平衡。

Thanks to the agile design paradigm provided by advanced hardware description languages, e.g., Chisel [1], and flexible and parameterized hardware generators, chip architects and engineers possess the capability to deliver a high-quality chip within a limited time budget.

幸好还有高级硬件描述语言提供的敏捷设计模式，如Chisel，和灵活的、参数化的硬件生成器，芯片架构师和工程师拥有在有限的时间预算内给出高质量的芯片的能力。

To further enhance the chip design ability built on top of the agile development paradigm for industry, exploring a series of chips in a given design space to achieve different degrees of trade-offs w.r.t. performance, power, and area in a short time is necessary.

在工业敏捷开发范式的基础上，为进一步增强芯片设计能力，在给定的设计空间中探索一系列芯片，以在短时间内得到在性能、功耗和面积上不同的折中程度，是必要的。

We look for effective and practical design space exploration algorithms to solve the problem. Specifically, in this contest problem, we focus on microarchitecture exploration of processors, i.e., central processing units (CPU).

我们寻找高效实际的设计空间探索算法，以求解问题。具体的，在这个竞赛问题上，我们聚焦在处理器（即CPU）的微架构探索上。

### 1.1 Microarchitecture

Microarchitecture is an implementation of a given instruction set architecture (ISA). For example, it decides the detailed implementation of different boxes (modules) in processors, such as instruction fetch unit, decoding unit, scheduling and issuing unit, execution unit, load and store unit, etc. Inside each box, queues, buffers, stacks, caches, etc., corporate within different logics to perform pre-determined logic functions. Out-of-order processors mainly leverage a re-order buffer to trace every in-flight instruction in the pipeline. The entries of the re-order buffer needed are relevant to the resources of instruction issuing queues, load and store queues, etc. When a processor is assigned with abundant issuing resources and the load-store queues are highly optimized, we can enlarge entries of the re-order buffer to accommodate higher instruction throughputs. Hence, we can achieve different compromises in adjusting the resources and capacities of critical components to attain various balances between PPA design targets.

微架构是给定的指令集架构(ISA)的一种实现。比如，微架构决定了处理器中不同模块的详细实现，比如，取指令单元，解码单元，调度单元和发射单元，执行单元，LSU等。在每个单元中，队列，缓冲区，堆栈，缓存等，与不同的逻辑合作，进行预先确定的逻辑功能。乱序处理器主要利用了一个重排序缓冲区，来追踪每个流水线中的正在进行的指令。需要的重排序缓冲区的条目，与指令发射队列，LSU等的资源相关。当处理器有充足的发射资源，load-store队列也是高度优化的，我们可以增大重排序缓冲区的条目，以适应更高的指令通量。因此，我们在调整关键组成部分的资源和容量时，可以获得不同的折中，以得到PPA设计目标上不同的均衡。

Figure 1 shows an example model of a processor. Instructions are fetched from I-Cache and packed and stored in a fetch queue. By recording and learning from the history, branch predictors predict the following instruction address given conditional jump instructions. Fetch queue decouples between the front-end and decoders. Micro-ops (Uops) are generated according to instruction. Conflicts in instructions are alleviated by register renaming, bypassing network, etc. Micro-ops trigger execution units to calculate with operands after being issued from issue queues (ISQ). Some memory-related instructions interact with D-Cache via the load and store unit. Across different boxes, parameters w.r.t. components of each box are extracted based on architects’ prior knowledge. Some of the example parameters are also listed in Figure 1. With different combinations of each parameter, a new microarchitecture can be formulated.

图1展示了处理器的一个例子。指令是从指令缓存中取出，打包存储到fetch队列中。在给定条件跳转指令后，通过从历史中记录和学习，分支预测器预测随后的指令地址。取指队列将前端和解码器进行解耦。根据指令生成微码(Uops)。通过寄存器重命名，bypassing网络等，缓解了指令中的冲突。微码触发了执行单元，在从发射队列发射后，用操作数进行计算。一些与内存相关的指令，通过LSU与数据缓存进行互动。在不同的模块中，基于架构师的先验知识，提取每个模块的组成部分的参数。一些参数的例子列在图1中。每个不同的参数组合，都可以得到一个新的微架构。

A very-large-scale integration (VLSI) verification flow is required to estimate the microarchitecture’s performance, power, and area under a technology with an accept fidelity. The VLSI verification flow incorporates different electronic design automation tools, e.g., logic synthesis, physical design, simulation, power estimation, etc., to give PPA values.

需要一个VLSI验证流来估计微架构在一种工艺下在一定的接收可信度下的性能，功耗和面积。VLSI验证流利用了不同的EDA工具，如，逻辑综合，物理设计，仿真，功耗估计，等，来得到PPA值。

### 1.2 Design Space Exploration

In the early stage of the chip development cycle, deciding hardware resources and structures for these components is requisite due to their variate impacts on the performance, power, and area. In previous times, the method to select appropriate logic and hardware resources for these boxes or components mainly relies on the engineering experiences of architects. Nevertheless, it becomes increasingly difficult when processors integrate with more components, causing little prior knowledge transferred to solve it. Therefore, researchers tried different models, analytical or data-driven black-box models, to characterize a suitable parameter combinations [2–7]. As analytical methods become difficult to establish and it is failed to be promising in the compromise between workforce input and accuracy feedback, current solutions rely on machine-learning-driven design space exploration methodologies. Researchers have designed various sampling algorithms and models, e.g., transductive experimental design, orthogonal array sampling, etc., and adopted diverse black-box models, e.g., linear regression with regularization, AdaBoost, Gaussian process, etc. A meaningful question is emerged, i.e., which design space exploration algorithm is more practical and effective in solving the problem? Can we design a more robust and efficient space exploration algorithm that approaches the optimal microarchitecture or predicts the Pareto frontier with higher accuracy in the large design space?

在芯片开发周期的早期阶段，对这些组成部分确定硬件资源和结构是很有必要的，因为对性能，功耗和面积有不同的影响。在之前，对这些模块选择合适的逻辑和硬件资源的方法，主要依赖于架构师的工程经验。尽管如此，当处理器结合了越来越多的组成部分，导致没有解决的这个先验知识，这就变得越来越难了。因此，研究者尝试了不同的模型，解析模型，或数据驱动的黑箱模型，以描述一个合适的参数组合的特征。解析模型越来越难以确立，工作输入和准确率反馈的折中越来越难以得到，目前的解决方案依赖于机器学习驱动的设计空间探索方法。研究者设计了各种采样算法和模型，如，转换试验设计，正交阵列采样，等，采用了各种黑盒模型，如带有正则化的线性回归，AdaBoost，高斯过程，等。一个有意义的问题出现了，即，哪种设计空间探索方法在求解问题中更实际，更有效呢？我们可以设计一种更稳健更高效的设计空间探索吗，达到最优微架构，或者在大型设计空间中用更高的准确率预测Pareto frontier？

### 1.3 Contest Objective

This contest problem aims to develop a practical, efficient, and accurate microarchitecture design space exploration algorithm. In this contest, we provide large-scale microarchitecture benchmarks to evaluate contestants’ solutions. We expect novel ideas to be inspired and applied in industrial product delivery. We also hope that this problem can facilitate innovative researches on microarchitecture design space exploration.

这个比赛问题目标是开发出一种实际的，高效的，和精确的微架构设计空间探索算法。在这个比赛中，我们给出了大规模微架构基准测试，以评估参赛者的解决方案。我们期望出现新思想，应用到工业产品交付上。我们还希望，这个问题可以促进微架构设计空间探索上创新性的研究。

## 2 Problem Formulation

The contest problem is formulated as a microarchitecture design space exploration. Given a fixed design space, find the Pareto optimal set within a short time. The Pareto optimal set is mapped to the Pareto frontier in the PPA objective space.

比赛问题是一个微架构设计空间探索问题。给定一个固定的设计空间，在短时间内找到Pareto最优集。Pareto最优集映射到PPA目标空间中的Pareto frontier。

### 2.1 Problem Description

For better understanding, we utilize an open-source RISC-V out-of-order core, Berkeley-Out-of-Order Machine (BOOM) [8–10], as an example.

为更好的理解，我们利用一个开源的RISC-V乱序核，BOOM，作为例子。

Figure 2 shows an overview of the BOOM pipeline. It is a ten-stage pipeline design, fully compliant with RV64GC ISA. Following Section 1.1, we can extract a microarchitecture design space for the core, listed in Table 1.

图2展示了BOOM流水线的概览。这是一个十级流水线设计，与RV64GC ISA是完全兼容的。根据1.1节，我们可以提取出该核的微架构设计空间，如表1所示。

A combination of parameters determines a microarchitecture implementation. Hence, we use a feature vector to define a microarchitecture embedding (the combination of parameters). Each dimension and element is the component, and selected candidate value, respectively. The encoding is a one-to-one mapping, e.g., a possible microarchitecture embedding (4, 16, 32, 12, 4, 8, 2, 2, 64, 80, 64, 1, 2, 1, 16, 16, 4, 2, 8) denotes the microarchitecture is a two-wide out-of-order core with 4-issue slots, 32-byte fetch width, 4-way I-Cache, 4-way D-Cache, etc. According to Table 1 and the design specifications of BOOM, the design space can be approximately 1.6 × 10^8. Each microarchitecture is evaluated with the same benchmark suite via the same VLSI verification flow to get golden performance, power, area values, and overall running time of the VLSI verification flow. Contestants need to develop a practical, efficient, and accurate microarchitecture design space exploration algorithm for the proposed microarchitecture design space. The algorithm should predict the Pareto optimal set, i.e., a set of microarchitectures formulate the Pareto frontier in the PPA objective space.

参数组合确定了微架构的实现。因此，我们使用一个特征向量来定义一个微架构的embedding（参数的组合）。每个维度和元素分别是组成部分，和选择的候选值。编码是一对一的映射，如，一个可能的微架构embedding (4, 16, 32, 12, 4, 8, 2, 2, 64, 80, 64, 1, 2, 1, 16, 16, 4, 2, 8)表示微架构是一个宽度为2的乱序核，有4发射槽，32-byte取指宽度，4路指令缓存，4路数据缓存，等等。根据表1和BOOM的设计指标，设计空间的大小大约为1.6 × 10^8。每个微架构用相同的基准测试包，通过相同的VLSI验证流进行评估，以得到性能、功耗，面积值，和VLSI验证流的总体运行时间的金标准。参赛者需要对提出的微架构设计空间探索开发一个实际的，高效的，和精确的微架构设计空间探索算法。算法应当预测Pareto最优集，即，微架构的集合，形成PPA目标空间的Pareto frontier。

### 2.2 Input and Output Format

We will provide a design space exploration algorithm benchmarking platform in this contest problem. The design space exploration algorithm benchmarking platform accepts contestants’ submitted solutions and returns their scores according to our pre-defined score functions.

我们会在这个比赛问题中给出一个设计空间探索算法基准测试平台。设计空间探索算法基准测试平台接收参赛者提交的解，根据我们预先定义的评分函数返回其分数。

#### 2.2.1 Input Format

We only release a subset of the design space to contestants, e.g., one possible subset of Table 1. As Section 2.1 introduces, each microarchitecture embedding is a high-dimensional feature vector. Thus, the candidate values are known to contestants. Contestants then implement their design space exploration algorithms to predict the Pareto optimal set. Contestants can only access the PPA values of the subset of the design space and are ignorant of other combinations of parameters outside the design space. Moreover, they are restricted from implementing their algorithms within our provided application program interfaces (APIs).

我们只将一部分设计空间放出给参赛者，如，表1的一个可能的子集。如2.1节介绍的，每个微架构embedding是一个高维特征向量。因此，参赛者是知道候选的值的。参赛者然后实现其设计空间探索算法，以预测Pareto最优集。参赛者只能访问设计空间子集的PPA值，不知道设计空间之外的其他参数组合。而且，参赛者要在我们提供的API之上实现其算法。

The design space exploration algorithm benchmarking platform is based on Bayesmark [11], and we will provide an appropriate version for the contest.

设计空间探索算法基准测试平台是基于Bayesmark，我们会给出比赛的合适的版本。

Most black-box design space exploration algorithms include two critical steps, sample, and update. “Sample” is the step of selecting microarchitectures that need to be evaluated. “Update” is the step to update the black-box model with the already sampled dataset. Design space exploration algorithms iterate over these two steps from time to time and gradually find the Pareto optimal set with the explored dataset, e.g., Bayesian optimization, simulated annealing, genetic algorithm, etc. Our provided benchmarking platform supports these two steps.

多数黑箱设计空间探索算法包括两个关键的步骤，采样，和更新。采样的步骤，是选择要评估的微架构。更新的步骤，是用已经采样的数据集来更新黑箱模型。设计空间探索算法迭代这两个步骤，逐渐发现要探索的数据集的Pareto最优集，如，贝叶斯优化，模拟退火，遗传算法，等。我们给出的基准测试平台支持这两个步骤。

The code snippet listed above demonstrates the input sample of the contest problem. The programming language is restricted to Python. Contestants are required to implement “suggest” and “observe” functions. The function “suggest” is a wrapper for “Sample” and “observe” is a wrapper for “Update”. The benchmarking platform also supports automatic scripts for contestants to familiarize the design space exploration flow and automatic solution codes packing and submission. Random search will serve as a baseline example for contestants to get started quickly. Third-party Python packages are allowed to implement the algorithm, e.g., basic operations with tensors, etc. If contestants leverage third-party Python packages, they should strictly follow submission instructions to make their solutions executed correctly in our servers.

上面列出的代码片段展示了比赛问题的输入样本。编程语言限制为Python。参赛者需要实现suggest和observe函数。函数suggest是采样的一个包装，observe是更新的一个包装。基准测试平台还支持自动脚本，使参赛者熟悉设计空间探索流，和求解代码自动打包和提交。随机搜索是一个基准例子，使参赛者迅速开始。第三方Python包可以用于实现算法，如，对张量的基础操作，等。如果参赛者利用第三方Python包，应当严格遵守提交指示，以使其解在我们的服务器上可以正确的执行。

#### 2.2.2 Output Format

The benchmarking platform only returns scores of their submissions. Contestants can only access to dataset released to them. We leverage hidden datasets to evaluate their submissions. All evaluations are executed on the same independent servers, hidden from all contestants. The requirement is necessary to make the contest fairer. The scores of submissions are calculated based on the Euclidean distance of PPA values mapped from the predicted Pareto optimal set and the golden Pareto optimal set. We will detail the evaluation metrics in Section 3. Each team can submit to their solutions no more than a fixed number of times a day. The rankings regarding scores of their submissions are released to all contestants timely.

基准测试平台只返回其提交的分数。参赛者只能访问释放出来的数据集。我们利用隐藏的数据集来评估其提交。所有评估在相同的独立服务器上进行执行，对所有参赛者都是隐藏的。这些要求使得对参赛者更加公平。提交的分数基于预测的Pareto最优集和金标准Pareto最优集的PPA值的欧式距离计算得到。我们在第3部分详述评估度量标准。每个队伍在一天内提交的次数有限。提交的分数的排名会及时的反馈给所有参赛者。

#### 2.2.3 Notice

The design space is vast, but it does not mean each microarchitecture in the design space is valid. Invalid microarchitectures are failed to simulate. We do not know whether a microarchitecture is invalid until we acquire the simulation reports in the VLSI verification flow. Therefore, the benchmarking platform will return a specified and reasonable number to the function “observe” shown in Section 2.2.1 and increase the running time spent evaluating such an invalid microarchitecture. It incentivizes contestants to design a model in their solutions to learn and circumvent these invalid microarchitectures, saving running time by not pushing them to the VLSI verification flow.

设计空间是巨大的，但并不意味着设计空间中的每个微架构都是有效的。无效的微架构在仿真时会失败。我们只能在VLSI验证流的仿真报告得到后，才能知道一个微架构是否是有效的。因此，基准测试平台会对函数observe返回一个指定的、合理的值，增加评估这样一个无效的微架构的运行的时间。它鼓励参赛者设计一个模型学习并避免这些无效的微架构，不要将其推向VLSI验证流，节省运行时间。

## 3. Evaluation

In this contest problem, we rank contestants’ submissions based on two metrics, Pareto hyper-volume difference and overall running time (ORT).

在这个比赛问题中，我们对参赛者提交的排序是基于两个度量，Pareto超体差异和总计运行时间(ORT)。

Pareto hypervolume, as illustrated in Equation (1), is the Lebesgue measure of the space dominated by the Pareto froniter and bounded by a reference point vref [7].

Pareto超体，如式1所示，是Pareto frontier占优的由参考点vref受限的空间的Lebesgue度量。

$$PVol_{vref} (P(Y)) = \int_y 1[y≽vref] [1-\prod_{y*∈P(Y)} 1[y*⪰̸y]] dy$$(1)

where 1(·) is the indicator function, which outputs 1 if its argument is true and 0 otherwise, P(Y) is the Pareto frontier.

其中1(·)是指示器函数，如果参数值为真则输出1，否则输出0，P(Y)是Pareto frontier。

Figure 3(a) illustrates an overview of the Pareto hypervolume. A better point will be closer to the origin point in the two-dimensional space (f1(x) and f2(x)). Given the point colored in orange as the reference point, the points colored in purple are not dominated by any other, e.g., purple points are not dominated by green. The Pareto hypervolume in Figure 3(a) is the area of the region colored in gray, i.e., a convex polygon bounded by the reference point and purple points. In the contest problem, we deal with three-dimensional space, i.e., performance, power, and area, as Figure 3(b) illustrates. Hence, the Pareto hypervolume is the volume bounded by the reference point (e.g., the original point) and explored objective values, i.e., y1, y2, and so on. The Pareto hypervolume difference is a difference between the golden Pareto hypervolume, formulated from the actual Pareto frontier in the design space, and the predicted Pareto hypervolume, formulated from the predicted Pareto frontier, as shown in Equation (2).

图3给出了Pareto Hypervolume的概览。在二维空间中，一个更好的点会更接近于原点。橘色的点选为参考点，那么紫色的点不会被任何其他点占优，如，紫色的点没有被绿色的点占优。图3a中的Pareto hypervolume是灰色区域的面积，即，由参考点和紫色的点组成的凸多边形。在比赛问题中，我们面对的是三维空间，即，性能，功耗和面积，如图3b所示。因此，Pareto hypervolume是参考点和探索的目标值为边界的体积。Pareto hypervolume差异，是金标准Pareto hypervolume（设计空间中真实的Pareto frontier），和预测的Pareto hypervolume（预测的Pareto frontier）的差异。

$$d_{PVol} = PVol_{vref} (P(y*)) - PVol_{vref}(P(Y))$$(2)

where Y∗ is the set of golden objective values, and Y is the predicted objective values by the optimizer.

其中Y*是金目标值的集合，Y是预测的目标值。

Overall running time measures the total time of algorithms, including the submission of contestants’ algorithms and the time spent on the VLSI verification flow. Since each microarchitecture has been pushed to the VLSI verification flow to get corresponding PPA values before evaluating contestants’ submissions, we can access the time spent on the VLSI verification flow quickly with the dataset, i.e., we do not push the predicted Pareto set to the VLSI verification flow.

总体运行时间(ORT)度量的是算法的总计时间，包括参赛者算法的提交，和在VLSI验证流上花费的时间。由于每个微架构都被推送到VLSI验证流中，以得到对应的PPA值，然后参赛者的提交才会被评估，所以我们可以访问数据集得到在VLSI验证流中所耗费的时间，即，我们不会将预测的Pareto集合推到VLSI验证流中。

The final scores are based on Pareto hypervolume difference and ORT, as Equation (3) shows.

最终的分数是基于Pareto hypervolume差和ORT，如式3所示。

$$score = β/(Pareto hypervolume difference) · (α-(ORT-θ)/θ), ORT≥θ; (α+|ORT-θ|/θ), ORT<θ$$(3)

where β is a constant (will be determined in the real dataset), α is an ORT score baseline, equal to 6, and θ is a pre-defined ORT budget, equivalent to 2625000. The constants are set to align with the de facto microarchitecture design space exploration flow. It is worth noting that if the ORT is six times larger than θ, then the final score will be negative. Hence, a better solution has lower Pareto hypervolume difference and ORT as much as possible.

其中β是一个常数（在真实的数据集中确定），α是一个ORT分数基准，为6，θ是一个预定义的ORT预算，等于262500。这些常数的设置，是为了对齐事实上的微架构设计空间探索流。值得注意的是，如果ORT比θ大6倍，那么最终的分数就是负的。因此，如果Pareto hypervolume difference和ORT值都很小，那么这个解就很好。

All evaluations are executed on the same independent servers, hidden from all contestants. Overall running times are calculated based on nearly the same environment, e.g., servers’ workload, etc.

所有的评估都是在相同的独立服务器上进行的，对所有的参赛者都是隐藏的。总体运行时间的计算，是基于基本相同的环境的，如，服务器的workload，等。

## 4. Benchmark Suite

The benchmark suite includes the microarchitecture design space of several processors, corresponding PPA values, and time consumed in the VLSI verification flow. The processors are open-sourced RISC-V cores, e.g., Rocket [12], BOOM [8–10] , etc., hoping to strive for a research hotspot in the RISC-V community. Each microarchitecture is pushed to the same VLSI verification flow, i.e., electronic design automation tools with the same version and a processing predictive kit (PDK), e.g. ASAP7 [13].

基准测试包包括，几个处理器的微架构设计空间，对应的PPA值，在VLSI验证流中消耗的时间。处理器是开源的RISC-V核，如Rocket，BOOM，等。每个微架构都推到相同的VLSI验证流，即，相同版本的EDA工具，和一个处理预测包(PDK)，如，ASAP7。

## 5. Discussion

Microarchitecture design space exploration is a potential method, highly coupled with the agile design paradigm. Due to the large design space and time-consuming VLSI verification flow, designing such an algorithm that can perform well on average for processors is difficult. For industry, it can save a considerable cost in the long-time chip development cycle with the help of microarchitecture design space exploration and deliver comparable products for customers. We expect a practical, efficient, and accurate microarchitecture design space exploration algorithm to further research in computer architecture and electronic design automation.

微架构设计空间探索是一种有潜力的方法，与敏捷设计范式是耦合在一起的。由于大型设计空间和耗时的VLSI验证流，设计这样一种对处理器可以表现的很好的算法是很难的。对于工业界，在长期的芯片开发周期中，在微架构设计空间探索的帮助下，可以节约相当的代价，交付出不错的产品。我们期望有一个实际的，高效的，和精确的微架构设计空间探索算法，在计算机架构和EDA中得以进一步研究。