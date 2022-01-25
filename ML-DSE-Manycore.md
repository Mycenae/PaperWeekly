# A Machine Learning Framework for Multi-Objective Design Space Exploration and Optimization of Manycore Systems

Biresh Kumar Joardar et. al. @ Washington State University

## 0. Abstract

The growing needs of emerging big data applications has posed significant challenges for the design of optimized manycore systems. Network-on-Chip (NoC) enables the integration of a large number of processing elements (PEs) in a single die. To design optimized manycore systems, we need to establish suitable trade-offs among multiple objectives including power, performance, and thermal. Therefore, we consider multi-objective design space exploration problems arising in the design of NoC-enabled manycore systems: placement of PEs and communication links to optimize two or more objectives (e.g., latency, energy, and throughput). Existing algorithms suffer from scalability and accuracy challenges as size of the design space and the number of objectives grow. In this paper, we propose a novel framework referred as Guided Design Space Exploration (Guided-DSE) that performs adaptive design space exploration using a data-driven model to improve the speed and accuracy of multi-objective design optimization process. We provide two concrete instantiations of guided-DSE and present results to show their efficacy for designing 3D heterogeneous manycore systems.

涌现的大数据应用需求的逐渐增多，为优化众核系统的设计提出了挑战。片上网络(NoC)使得大量处理元素(Processing Elements, PE)可以集成在单个die上。为设计优化的众核系统，我们需要在多个目标中确定合适的折中，包括功耗，性能和散热。因此，我们考虑在NoC众核系统的设计中出现的多目标设计空间探索问题：PEs的布局和通信连接，来优化两个或多个目标（如，延迟，功耗，和吞吐量）。现有的算法在设计空间和目标数量增加时，在可扩展性和准确率上都有问题。本文中，我们提出一个新的框架，称为引导设计空间探索，使用一个数据驱动的模型来进行自适应设计空间探索，来改进多目标设计优化过程的速度和准确率。我们给出引导DSE的两个具体例子，给出的结果表明，在设计3D异质众核系统中是非常有用的。

**Index Terms**—Machine learning, Manycore systems, Heterogeneity, Network-on-chip, Big data computing

## 1. Introduction

There is a rise of exciting big data application areas including deep learning, graph analytics, and scientific computing that demands appropriate design of many core systems by understanding their requirements (e.g.,high performance, energy efficiency, and reliability). Therefore, design space exploration and optimization of many core systems is a fundamental challenge in the field of electronic design automation.

大数据应用领域生在崛起，包括深度学习，图分析，和科学计算，需要合理的设计众核系统，理解其要求（如，高性能，能耗效率高，可靠性等）。因此，众核系统的设计空间探索和优化，是EDA领域的一个基础性挑战。

Designing optimized manycore systems require trading-off multiple objectives including power, performance, and thermal. In general, the objectives are conflicting in nature and all of them cannot be optimized simultaneously (e.g., power and performance). Therefore, we need to find the Pareto optimal set of designs. A design is called Pareto optimal if it cannot be improved in any of the objectives without compromising some other objective. The key challenges for design space exploration can be summarized as follows: 1) large design space that grows in size and complexity with increasing system size and diversity of processing elements; 2) hardness and complexity of design-time optimization problem grows with the number of objectives; and 3) Pareto front of designs is non-convex and highly complex for a large number of objectives.

设计优化的众核系统，需要在多个目标之间折中，包括功耗，性能和散热。一般来说，目标在本质上是相互矛盾的，这些目标不可能同时被优化（如，功耗和性能）。因此，我们需要找到设计的Pareto optimal集。一个设计如果达到了这样一种状态，即想要改进任意一个目标，就需要在一些其他目标上进行折中，我们称之为达到了Pareto最优。设计空间探索的关键挑战可以总结如下：1)设计空间规模很大，规模和复杂度都随着系统规模和处理元素的增加而增加；2)设计时间优化问题的难度和复杂度随着目标数量的增加而增加；3)设计的Pareto front是非凸的，对于大量目标来说是高度复杂的。

Most of the traditional approaches to tackle this problem including methods based on local search heuristics and evolutionary algorithms either generate sub-optimal solutions or require significant computation time making them inefficient for large design spaces. These algorithms do not leverage the knowledge gained from past design space exploration in an explicit manner and do not allow leveraging the training data from simulations to evaluate the quality of the designs. Machine learning (ML) and data-driven algorithms can potentially overcome this challenge by performing intelligent design space exploration to efficiently uncover optimized computing systems. In this paper, we present a general ML framework referred as guided Design Space Exploration (guided-DSE) and two concrete instantiations, namely, MOO-STAGE and MOOS, that performs adaptive design exploration guided by learned knowledge captured explicitly from designs explored in the past. These two algorithms differ in the form of search control knowledge, and how it is learned and used to achieve adaptive design space exploration.

处理这个问题的多数传统方法包括，基于局部搜索启发式的方法，和演化算法，这些方法可能会生成次优的解，或需要大量计算时间，使其对于大型设计空间效率很低。这些算法并没有显式的利用之前的设计空间探索的知识，而且不能利用从仿真中得到的训练数据，来评估设计的质量。机器学习(ML)和数据驱动的算法可能会克服这个挑战，进行智能设计空间探索，以高效的得到优化计算系统。本文中，我们给出了一个通用ML框架，称为引导设计空间探索(guided-DSE)，和两个具体的例子，即，MOO-STAGE和MOOS，进行自适应的设计空间探索，由过去探索的设计中学习到的知识进行引导。这两个算法在搜索控制知识上有所不同，以及在怎样学习的，怎样用于获得自适应设计空间探索上也不同。

In the next section, we formally describe the multi-objective optimization problem along with the physical constraints that need to be satisfied in many core systems. In Section III, we provide a summary of the related work addressing manycore systems design optimization problem and their limitations. Section IV describes the generic guided-DSE framework, which unifies the two main algorithms presented in this paper (MOO-STAGE and MOOS). Finally, in Section V, we demonstrate the efficacy of these methods for the problem of designing 3D Heterogeneous manycore systems.

在下一节，我们在形式上描述了众核系统中的多目标优化问题，与要满足的物理约束。在第3部分中，我们给出了相关工作的总结，处理众核系统设计优化问题，以及其局限性。在第4部分，描述了一般性引导DSE框架，统一了本文中给出的两种主要算法(MOO-STAGE和MOOS)。最后，在第5部分，我们证明了这些方法在设计3D异质众核系统中的有效性。

## 2. Problem Setup

We describe below the key components of a multi-objective design space exploration problem arising in manycore systems. We concretely instantiate each component with an example from NoC-enabled 3D Heterogeneous manycore systems design problem.

我们下面描述在众核系统中出现的多目标设计空间探索问题的关键组成部分。我们对每个组成部分都给了一个例子，来自于NoC 3D异质众核系统设计问题。

**1. Design Space D**. For a fixed system size, we are provided with resources in the form of different types of processing elements (PEs) like CPUs, GPUs and communication links. For N PEs and M communication links, we get a fully-specified design space in the form of all possible manycore design configurations. For example, each NoC-enabled manycore design d ∈ D corresponds to a specific placement of PEs and communication links. The design space grows exponentially with increasing system size (N). Suppose we have T types of PEs with n1, n2, · · · , nT number of PEs for each type such that $N = \sum^T_{i=1} n_i$. The size of the search space for candidate core placements is N!/(n1!·n2!···nT!). Each of these candidate core placements can pick a link placement from the space of all possible link placements, whose size is N choose M. The total size of the design space is a product of the sizes of both (core placement and link placement) search spaces.

设计空间D。对于固定的系统规模，我们给定的资源，是以不同类型的处理单元(PEs)，如CPUs，GPUs和通信连接。对于N个PEs和M个通信连接，我们得到了一个完全指定的设计空间，其形式是所有可能的众核设计配置。比如，每个NoC众核设计d ∈ D，对应着一个特定的PEs的布局和通信连接。设计空间随着系统规模(N)的增加而指数级增加。假设我们有T个类型的PEs，每种类型数量为n1, n2, ..., nT，$N = \sum^T_{i=1} n_i$。候选核布局的搜索空间的规模是N!/(n1!·n2!···nT!)。这些候选核布局的每个，都可以从所有可能的连接布局的空间中选择一个连接布局，这个空间的规模是N choose M。设计空间的总计大小，是两个搜索空间大小之积（核布局和连接布局）。

**2. Objective Set O**. To evaluate each design d ∈ D, we are given a set of k > 1 objectives O = {O1, O2, · · · , Ok}. Some example objectives include power, performance, and thermal.

目标集O。为评估每个设计d ∈ D，我们给定一个目标集合O = {O1, O2, · · · , Ok}，k > 1。一些例子目标包括，功耗，性能和散热。

**3. Physical Design Constraints C**. Generally, each practical and feasible manycore system design needs to satisfy a set of physical constraints C. Some examples constraints for 3D heterogeneous design space are as follows: (a) The overall communication network should have a path between any two tiles; and (b) Restricting a router’s maximum number of inter-router ports so that no router becomes unrealistically large.

物理设计约束C。一般来说，每个实践的和可行的众核系统设计需要满足物理约束集C。3D异质设计空间的一些例子约束如下：(a)总体的通信网络应当在任意两个tiles之间有通路；(b)限制一个路由器的最大数量的路由间端口，这样就不会有路由器变得很大。

Our goal is to find the Pareto set (i.e., non-dominated set of designs) D∗ from DC ⊆ D, where DC is the set of designs that satisfy all the physical constraints C. A design d2 is dominated by design d1 if ∀i, Oi(d1) ≤ Oi(d2); and ∃j, Oj (d1) < Oj (d2). Once the Pareto set D∗ is computed, the designer employs some decision-making criteria (e.g., energy-delay-product from simulations) to select the best design d∗ from the pareto set D∗.

我们的目标是从DC ⊆ D中找到Pareto集（即，设计的非占优集）D*，其中DC是满足所有物理约束C的设计集合。一个设计d2被设计d1占优，如果∀i, Oi(d1) ≤ Oi(d2)，并且∃j, Oj (d1) < Oj (d2)。一旦计算了Pareto set D*，设计者采用一些决策规则（如，仿真中得到的能耗延迟乘积）以从Pareto集D* 中选择最佳设计 d*。

## 3. Related Work

AMOSA [1] and NSGA-II [2] are two common algorithms for solving multi-objective design optimization problem. In [1], the authors demonstrate that AMOSA, which is akin to simulated annealing algorithm can outperform the genetic algorithm based NSGA-II [2] in the number of distinct solutions, time needed, and overall performance. However, both these algorithms do not leverage the knowledge gained from past design space exploration in an explicit manner. Typically, these algorithms execute many unguided and independent searches, from different starting points, to increase the chance of reaching global optima. However, as the number of objectives increases, the complexity of the optimization problem grows, which exponentially increases the time these algorithms takes to find an acceptable solution.

AMOSA[1]和NSGA-II[2]是求解多目标设计优化问题的两种常见算法。在[1]中，作者展示了AMOSA比NSGA-II在不同解的数量，需要的时间，总体性能上都要更优一些，AMOSA是基于模拟退火的，NSGA-II是基于遗传算法的。但是，这两个算法都没有显式的利用之前的设计空间探索得到的知识。一般的，这些算法会执行很多没有引导的和独立的搜索，从不同的起始点，以增加达到全局最优的机会。但是，随着目标数量的增加，优化问题的复杂度也随之增加，这些算法找到一个可接受的解所耗费的时间会指数级增加。

Another body of literature utilizes SAT-decoding [3] for solving constrained combinatorial problem (CCPs) [4] formulation of manycore design space optimization. The key idea behind this approach is to use a SAT solver to generate valid solutions in the design space from the genotypes searched over by an evolutionary algorithm. The valid solutions are represented by a set of Pseudo-Boolean (PB) constraints, formulated using binary variables. Although this approach works well in multiple scenarios, it also doesn’t take into account past exploration and the PB formulation fails to incorporate certain constraints.

还有一些文献利用SAT-decoding来求解众核设计空间优化表述成的约束组合问题(CCP)。这个方法背后的关键思想，是使用SAT求解器来在演化算法搜索的genotypes得到的设计空间中生成可用的解。可用的解是由伪Boolean约束表示的，使用二值变量表述的。虽然这种方法在多个场景中效果都很好，但它并没有考虑到过去的探索，而且伪Boolean表述很难将特定约束纳入进来。

Bayesian optimization has emerged as a widely useful technique in many areas for resource-efficient optimization of expensive black box functions. However, these techniques have not been utilized to full potential in manycore systems because of multiple reasons: a) These methods are only applicable for continuous spaces (i.e., design variables are continuous) [5], [6], whereas most of the problems in manycore systems designs are discrete combinatorial structures [7]; and (b) They do not scale gracefully with large sized systems which is the norm in manycore systems design.

贝叶斯优化已经在很多领域中成为了一个广泛使用的技术，解决昂贵的黑盒函数的资源高效优化。但是，这些技术并没有在众核系统中完全利用，这有几个原因：a)这些方法只对连续空间可应用（即，设计变量是连续的，而众核系统设计中的多数问题是离散的组合结构），b)它们对于大规模系统并不能很好的缩放，而这是众核系统设计中的常态。

## 4. Machine Learning for Multi-Objective Design Space Exploration and Optimization

### 4.1 Motivation and Utility of ML Techniques

Machine Learning (ML) inspired manycore system design is a promising research direction to address the above-mentioned challenges [8], [9]. ML techniques utilize the knowledge from designs explored in the past to guide future design space exploration to improve the computational-efficiency and accuracy of solving the multi-objective optimization problem. This allows searching over candidate solutions in the favourable part of the design space and reject parts of the design space that is expected to perform poorly on the objectives. Recent advances in machine learning also scale gracefully with increase in input dimensions, thereby resulting in higher gains for large system sizes when compared to standard approaches. Indeed, recent work [10]–[12] adapted an online machine learning technique called STAGE to optimize the design of NoC for homogeneous manycore systems, and showed that it is much more efficient than simulated annealing (SA) and genetic algorithm (GA) for a single objective function.

机器学习启发的众核系统设计，是一个有希望的研究方向，可以处理上述提到的挑战。ML技术利用了过去探索的设计的知识，来引导未来设计空间的探索，以改进计算效率和求解多目标优化问题的准确率。这可以在较好的设计空间部分来搜索候选解，拒绝部分被认为不太好的设计空间。最近在机器学习方面的进展，在面对输入维度增加的时候，扩展的比较合理，因此与标准方法比较时，在面对大规模系统时，会得到更大的收益。最近的工作采用了一种在线机器学习技术，称为STAGE，以对同质众核系统来优化NoC的设计，在单目标函数的情况下，比模拟退火(SA)和遗传算法(GA)要高效的多。

### 4.2 Guided Design Space Exploration Framework

We describe a generic framework referred as guided-DSE to solve multi-objective optimization problems arising in the design of manycore systems. The key idea is to learn appropriate search control knowledge from the designs explored in the past and use it to adaptively explore the design space to improve the accuracy and efficiency of over all optimization process. There are three key elements of guided-DSE framework. First, what form of knowledge is learned to perform adaptive design space exploration. Second, how is this knowledge acquired from designs explored in the past. Third, how is this knowledge used to guide future design space exploration in order to quickly uncover high-quality (in terms of multiple objectives) candidate solutions. Figure 1 provides a high-level overview of guided-DSE based on local search procedure. The learned search control knowledge guides the local search procedure by selecting good starting states. We present two concrete algorithmic instantiations of guided-DSE framework below.

我们描述了一个通用框架，称为引导DSE，求解在众核系统设计中出现的多目标优化问题。关键思想是，从过去探索的设计中，学习合适的搜索控制知识，并用于自适应的探索设计空间，以改进整体优化过程的准确率和效率。引导DSE框架有三个关键的元素。第一，学习了什么形式的知识，进行自适应设计空间探索。第二，怎样从过去探索的设计中获取这些知识。第三，这种知识怎样用于引导未来的设计空间探索，以迅速的发现高质量候选解（多目标）。

**MOO-STAGE Algorithm**. MOO-STAGE [13] is a multi-objective extension of the well-known STAGE algorithm that operates in two alternating stages. The first stage executes a local search procedure L from a given starting state, guided by a cost function which captures each primary objective (e.g., energy, latency, thermal). The search trajectories collected from this local search procedure is the form of knowledge utilized by MOO-STAGE algorithm for the subsequent stage called as meta search. There are two key steps performed in the meta search procedure. First, learning an evaluation function E that describes the utility of a starting state in terms of the final solution quality observed by performing local search from this starting state. This is accomplished by learning a regression function from the past local search trajectories: each training example corresponds to a starting design and the quality of final solution from local search. Second, the learned evaluation function E is optimized by executing the local search procedure to find promising starting states that will potentially uncover high-quality solutions for the main optimization problem.

**MOO-STAGE算法**。MOO-STAGE是STAGE算法的多目标拓展，在两个交替的阶段中进行计算。第一个阶段从一个给定的开始状态来执行一个局部搜索过程L，由一个反映每个主要目标（如，能耗，延迟，散热）的代价函数来引导。从这个局部搜索过程得到的搜索轨迹，是MOO-STAGE算法利用用于后续阶段的知识，这个阶段称为meta搜索。在meta搜索过程中，有两个关键的步骤。第一，学习一个评估函数E，以最终解的质量描述了起始状态的作用，从这个起始状态进行局部搜索得到最终解。这是通过从过去的局部搜索轨迹中学习一个回归函数得到的：每个训练样本对应着一个起始设计，和从局部搜索得到的最终解的质量。第二，学习到的评估函数E的优化，是执行这个局部搜索过程，以找到有希望的起始状态，可能对主优化问题发现高质量解。

Using these two computational search processes (local search and meta search), MOO-STAGE progressively learns the structure of the solution space and improves the evaluation function. Essentially, the algorithm attempts to learn a regressor that can predict the quality of the local optima (e.g., PHV of the final Pareto set) from any starting design and explicitly guides the search towards predicted high-quality starting designs.

使用这两个计算搜索过程（局部搜索和元搜索），MOO-STAGE逐渐的学习到了解空间的结构，改进了评估函数。实质上，算法试图学习一个回归器，可以从任何初始设计中预测局部最优值的质量（如，最终Pareto集的PHV），显式的引导搜索朝向预测的高质量初始点。

**MOOS Algorithm**. MOOS [14] is a data-driven algorithm that employs the principle of optimism in the face of uncertainty [15] to improve the speed and accuracy of the design optimization process. The principle of optimism suggests exploring the most favorable region of the design space based on the experience gained from past exploration. MOOS follows an iterative two-stage optimization strategy that acts as an instantiation of the guided-DSE framework described above. In each iteration, it employs a scalarized objective to select the starting solution for a local search procedure. The adaptivity in MOOS is enabled by a tree-based model over the space of scalarization parameters. The tree model is constructed by employing the knowledge learned using the quality of Pareto set obtained by the local search procedure. The parameters of scalarization are chosen adaptively based on this tree-based model. The data-driven tree-based model not only guides the search towards better solutions but also reduces the runtime of MOOS to find acceptable solutions.

MOOS算法。MOOS是一个数据驱动的算法，在面对不确定时，采用乐观的准则，来改进设计优化过程的速度和准确率。乐观的准则意思是，基于过去的探索得到的经验，探索设计空间中最倾向的区域。MOOS按照迭代的两阶段优化策略，是上述引导DSE框架的一个例子。在每次迭代中，都采取了标量化的目标，来对一个局部搜索过程来选择起始解。在MOOS中的自适应性是由一个基于树的模型得到的，在标量化的参数的空间上。树模型的构建，是利用学习到的知识，使用Pareto集的质量，由局部搜索过程得到。参数的标量化，是自适应的选择的，基于这个基于树的模型。数据驱动的基于树的模型，不仅引导搜索朝向更好的解，还减少了MOOS找到可接受的解的运行时间。

MOOS algorithm improves over MOO-STAGE in terms of both speed and accuracy of the optimized designs. The key insight is that MOOS performs efficient search guided by a data-driven tree-based model over scalarization parameters (small and simple search space) when compared to MOO-STAGE that performs data-driven search guided by learned evaluation function over input design space (large and complex search space). Indeed, our experiments demonstrate this phenomenon on real-world manycore design problems.

MOOS算法对MOO-STAGE算法的改进，包括优化的设计的速度和准确率。关键的洞见是，MOOS进行的高效搜索，是由数据驱动的基于树的模型在标量化的参数（小型简单的搜索空间）空间中引导的，而比较的MOO-STAGE，进行的是数据驱动的搜索，是由在输入设计空间的学习的评估函数来引导的（大型复杂的搜索空间）。确实，我们的试验在真实世界众核设计问题上证明了这个现象。

## 5. Experiments and Results

### 5.1 Experimental Setup

We present experimental results on the challenging problems of designing NoC-enabled 3D heterogeneous manycore systems. We consider MO-DSE of 3D heterogeneous many-core architecture design space due to its generality (2D/3D homogeneous and 2D heterogeneous are special cases of this design space) and importance as an upcoming technology. We employ a 64-tile system comprised of 8 CPUs, 16 Last-Level-Caches (LLCs), and 40 GPUs. The tiles have been distributed in a 4 × 4 × 4 3D system with the heat sink at the bottom. The tiles are interconnected via a custom NoC consisting of both vertical and planar links. Vertical links connecting the planar layers are implemented using using Through-Silicon-Via (TSV) technology. Each candidate design in the design space corresponds to a specific placement of PEs (CPUs, GPUs, and LLCs) and input planar links.

我们对设计NoC 3D异质众核系统的有挑战的问题，给出试验结果。我们考虑MO-DSE的3D异质众核架构设计空间，因为这有通用性（2D/3D同质和2D异质是这个设计空间的特殊情况），还非常重要，是一个上升的技术。我们采用了一个64-tile系统，由8个CPUs，16个LLCs，和40个GPUs组成。这些tiles分布成4 × 4 × 4的3D系统，heat sink在底部。这些tiles通过一个定制的NoC互相连接在一起，由竖直的和水平的连接组成。竖直的连接连接了平面层，是由Through-Silicon-Via技术实现的。设计空间中的每个候选设计，对应着PEs(CPUs, GPUs, LLCs)和输入平面连接的一个具体布局。

**Benchmark Applications**. We employ eight different applications from the Rodinia benchmark suite [16]: Back Propagation (BP), Breadth-First search (BFS), Gaussian Elimination (GAU), Hot Spot (HS), k-Nearest Neighbor (k-NN), LU Decomposition (LUD), Needleman-Wunsch (NW), and Path Finder (PF).

基准测试应用。我们采用Rodinia基准测试包中8个不同的应用：Back Propagation (BP), Breadth-First search (BFS), Gaussian Elimination (GAU), Hot Spot (HS), k-Nearest Neighbor (k-NN), LU Decomposition (LUD), Needleman-Wunsch (NW), and Path Finder (PF).

**Baseline and Objectives**. We consider a popular simulated annealing based method called AMOSA as a baseline to compare the performance of MOOS and MOO-STAGE. The objectives are defined below:

基准和目标。我们考虑一个流行的基于模拟退火的方法，称为AMOSA，作为基准，与MOOS和MOO-STAGE的性能进行比较。目标定义如下：

**1. Latency Objective**. Latency is the primary concern for CPUs as higher latencies cause CPUs to stall, leading to poor performance. For C CPUs and M LLCs, we model the average CPU-LLC latency as shown below [11]:

延迟的目标。延迟是CPUs的主要考虑，因为高延迟会导致CPUs停顿，导致性能低下。对于C个CPUs和M个LLCs，我们将平均CPU-LLC延迟建模如下：

$$Latency = \sum_{i=1}^C \sum_{j=1}^M (r·h_{ij} + d_{ij}))·f_{ij} / C / M$$

where r is the number of router stages, $h_{ij}$ is the number of hops from CPU i to LLC j, $d_{ij}$ is the total link delay, and $f_{ij}$ is the frequency of communication between core i and core j.

其中r是路由器阶段数量，$h_{ij}$是从CPU i到LLC j的hops数量，$d_{ij}$是总计连接延迟，$f_{ij}$是核i和核j之间的通信频率。

**2. Throughput Objective**. Unlike CPUs, GPUs rely on high-throughput memory accesses to enable high data parallelism. Load balancing, considering Mean and Standard deviation (STD) of link utilization as objectives, helps improve throughput [13]. The expected utilization Uk of link k is obtained from below:

吞吐量目标。与CPU不同，GPUs依赖于高通量内存访问来得到高数据并行性。负载均衡，考虑将连接利用的均值和标准差作为目标，帮助改进了吞吐量。连接k的期望利用率Uk通过下式得到：

$$U_k = \sum_{i=1}^R \sum_{j=1}^R f_{ij} · p_{ijk}$$

where R is the total number of tiles and $p_{ijk}$ indicates whether a planar/vertical link k is used to communicate between core i and core j respectively. The mean (µ) and standard deviation (σ) of the link utilization can be computed as shown below:

其中R是tile的总共数量，$p_{ijk}$指示的是一个平面/竖直的连接k是否用于在核i和核j之间通信。连接利用率的均值和标准差可以计算如下：

$$µ = \sum_{k=1}^L U_k / L, σ = \sqrt_{\sum_{k=1}^L (U_k-µ)^2 / L}$$

**3. Energy Objective**. Designers often have a fixed energy budget for optimum performance. Therefore, it is necessary to design high-performance systems within given energy budget. The total network energy is the sum of router and link energy

能量的目标。设计者的能量预算通常是固定的，要得到最优的性能。因此，在给定的能量预算下，设计高性能的系统，这是很有必要的。总计网络能量，是路由器和连接能量的和

$$E = \sum_{i=1}^N \sum_{j=1}^N f_{ij} · (\sum_{k=1}^L p_{ijk}·d_k·E_{link} + \sum_{k=1}^R r_{ijk}·E_r·P_k)$$

where N is the number of cores, R and L denote the total number of routers and links respectively, d_k represents the physical length of link k, E_link and Er denote the average link energy per unit length and router logic energy per port respectively and Pk denotes the number of ports at router k. Both pijk and rijk are binary variables that indicate whether a link/router k is used to communicate between core i and core j respectively. fij represents the frequency of communication between core i and core j respectively.

其中N个核的数量，R和L分别表示路由器和连接的总计数量，d_k表示连接k的物理长度，E_link表示单位长度的平均连接能量，Er表示每个端口的路由器逻辑能量，Pk表示在路由器k处的端口数量。pijk和rijk都是二值变量，指示的是一个连接/路由器k是否用于在核i和核j之间通信。fij表示在核i和核j之间的通信频率。

We consider three increasingly difficult MO problems with their corresponding objectives defined as follows: 2-OBJ: Mean and standard deviation (STD) of link utilization; 3-OBJ: Latency objective in addition to Mean and STD of link utilizations; 4-OBJ Energy objective in addition to three objectives in 3 OBJ case.

我们考虑三个越来越难的MO问题，其对应的目标定义如下：2-OBJ，连接利用率的均值和标准差；3-OBJ，除了连接利用率的均值和标准差外，还有延迟的目标；4-OBJ，再加入了能量的目标。

**Metrics**. We compare MOOS and the baseline algorithms in terms of two metrics. **1. Speedup factor** of MOOS over baseline is defined as T_Baseline/T_MOOS, where T_Baseline is the convergence time of the baseline algorithm and T_MOOS is the time taken by MOOS to reach the best solution uncovered by baseline. **2. Percentage quality gain** of MOOS over baseline when comparing the solutions uncovered by MOOS and baseline at the end of maximum time-bound. For performance-aware setting, we measure the quality of a candidate Pareto set by the smallest energy-delay-product (EDP) for the NoC designs in the Pareto set — smaller EDP means better design. Determining the quality of a Pareto set depends on the design requirements and can be evaluated via multiple metrics (e.g., Hypervolume indicator), but we employ EDP since it is the most reliable and commonly used metric for NoC design optimization.

**度量**。我们用两种度量比较了MOOS和基准算法。1.MOOS与基准相比的加速因子，定义为T_Baseline/T_MOOS，其中T_Baseline是基准算法的收敛时间，T_MOOS是MOOS达到基准找到的最佳解的耗费的时间。2.当比较在最大时间限制时，MOOS找到的解，与基准找到的解时，其质量提升百分比。对于感知性能的设置，我们度量候选Pareto集的质量，是用Pareto集中的NoC设计的最小能耗延迟乘积(Energy-delay-product, EDP)，更小的EDP意味着更好的设计。确定一个Pareto集的质量，依赖于设计需求，可以通过多个度量进行评估（如，Hypervolume indicator），但我们采用了EDP，因为这是NoC设计优化最可靠的，最常用的度量。

### 5.2 Results and Discussion

In this section, we demonstrate the advantages of MOOS and MOO-STAGE over AMOSA in terms of normalized energy-delay-product (EDP), speedup factor, and percentage quality gain. Figure 2 shows the normalized energy-delay-product (EDP) of the best design uncovered by MOOS, AMOSA, and MOO-STAGE as a function of time (in hours) for the Back Propagation application as a representative example noting that similar results are observed for other applications. EDP combines two metrics: network latency and energy. Here, latency is a measure of performance. EDP captures the trade-off between performance and power for any design. We can see that both MOOS and MOO-STAGE uncovers designs with lower EDP values significantly faster than AMOSA. Additionally, they find designs with better EDP than AMOSA after the convergence of all agorithms. Table I shows the quality of Pareto fronts in terms of PHV metric obtained by AMOSA and MOO-STAGE normalized with respect to MOOS (i.e., normalized PHV of MOOS is 1). We show the average PHV of all benchmarks for different number of objectives. These results show the advantage of data-driven adaptive design space exploration performed by MOOS and MOO-STAGE.

本节中，我们证明了MOOS和MOO-STAGE算法比AMOSA在多个方面的优越性，包括归一化EDP，加速因子，和质量提升百分比。图2展示了对于反向传播应用，MOOS，AMOSA和MOO-STAGE找到的最佳设计的归一化的EDP，作为时间的函数，这是一个有代表性的例子，对于其他应用也会得到的类似的结果。EDP结合了两个度量：网络延迟和能量。这里，延迟是性能的一种度量。EDP表示的是对于任何设计，其性能和功耗的折中。我们可以看到，MOOS和MOO-STAGE找到更低EDP值的设计时，比AMOSA要快的多。另外，它们在所有算法收敛后，找到的设计其EDP比AMOSA找到的要好。表I展示了Pareto front的质量，用的是AMOSA和MOO-STAGE得到的PHV度量，相对于MOOS进行归一化的结果（即，MOOS的归一化PHV是1）。我们展示了所有基准测试在不同数量的目标时的平均PHV。这些结果表明，MOOS和MOO-STAGE进行的数据驱动的自适应设计空间探索的优势。

Table II and III shows the percentage quality gain (measured in terms of EDP) and speed-up factor of MOOS over MOO-STAGE and AMOSA. There is a significant gap between the quality of solutions obtained by both MOOS and MOO-STAGE over AMOSA. As the number of objectives increase, the speedup factor and percentage quality gain increases showing the benefits of data driven methods.

表II和表III展示了MOOS比MOO-STAGE和AMOSA的质量提升百分比（以EDP进行度量）和加速因子。MOOS和MOO-STAGE获得的解，比AMOSA获得的解，其性能提升非常明显。随着目标数量的提升，加速因子和性能提升百分比也提高了，表明了数据驱动方法的好处。

## 6. Summary and Future Work

The design of manycore systems involve searching a large combinatorial design space to optimize multiple objectives. Prior methods including conventional heuristic-based approaches scale poorly as the design space and number of objectives increases. We proposed a multi-objective design space exploration and optimization framework called guided-DSE to improve the scalability and accuracy of optimization process when compared to state-of-the-art methods. Two concrete instantiations of guided-DSE (MOO-STAGE and MOOS) are studied in the context of 3D heterogeneous manycore systems design. MOO-STAGE and MOOS can adaptively identify and explore better areas of the design space in a data-driven manner. We demonstrated that MOO-STAGE and MOOS improves the speed of finding solutions similar to state-of-the-art methods and uncovers better designs.

众核系统的设计，包括搜索大型组合设计空间，以优化多个目标。之前的方法包括，传统的基于启发式的方法，在设计空间和目标数量增加时，其缩放会比较差。我们提出了一个多目标设计空间探索和优化框架，称为引导DSE，与目前最好的方法比，改进了优化过程的可扩展性和准确率。在3D异质众核系统设计中，研究了2个具体的引导DSE的例子。MOO-STAGE和MOOS可以以数据驱动的方式，自适应的识别和探索设计空间中的更好的区域。我们证明了MOO-STAGE和MOOS改进了找到与目前最好的方法类似的结果的解的速度，也找到了更好的设计。

Future work includes exploring the following questions. 未来的工作包括探索下面的问题。

1) Analytical objective functions are typically not accurate for evaluating designs when compared to their simulation based measurements. How can we synergistically combine the low-fidelity analytical objectives and high-fidelity simulation based objectives to perform efficient design optimization? [17], [18]

解析的目标函数，与基于仿真的度量相比时，在评估设计时一般是不够准确的。我们怎样协同的将低保真度的解析目标，与高保真度的基于仿真的目标，结合到一起，以进行高效的设计优化？

2) Emerging integration technologies pose further challenges for design space exploration and optimization [19]. Monolithic 3D (M3D) presents an attractive alternative to TSV-based designs with many advantages (e.g., improved integration density, total wire length, and communication energy) [12]. However, M3D based systems come with several challenges. During M3D fabrication, each tier is fabricated sequentially from the bottom to the top. One of the key challenges is to create top-tier transistors without impacting the already processed bottom-tier transistors and interconnects. Therefore, performance degradation of the top-tier transistors and increased delay in bottom-tier interconnects must be considered during the design-time optimization process [20]–[22].

出现的集成技术对设计空间探索和优化，提出了进一步的挑战。M3D是基于TSV的设计的替代品，有很多优势（如，改进的集成密度，总计线长，通信能量）。但是，基于M3D的系统有几个挑战。在M3D的制造过程中，每一层是按从下到上的顺序制造的。一个关键的挑战是，创建顶层晶体管时，不影响已经加工的底层晶体管和互联。因此，顶层晶体管的性能恶化，和底层互联增加的延迟，在设计时优化过程中也必须考虑到。

3) Emerging memory technologies such as Resistive RAM (ReRAM) allow us to perform in-memory processing to improve the performance and energy-efficiency of computationally demanding applications [23]–[25]. However, ReRAM’s provide a trade-off between efficiency, accuracy, and reliability. We need to explore novel hardware and software co-design frameworks to improve upon the drawbacks of non-volatile memory to break the power, performance, and memory walls in computing systems [26]–[28].

新出现的存储技术，如Resistive RAM (ReRAM)使我们可以进行存储内处理，以改进对计算需求较高的应用的性能和能耗效率。但是，ReRAM在效率，准确率和可靠性之间提供了一个折中。我们需要探索新的硬件和软件协同设计框架，以改进非易失性存储的缺陷，打破计算系统中的功耗，性能和存储的墙。