# Design Space Exploration using Time and Resource Duality with the Ant Colony Optimization

Gang Wang, et. al. University of California Santa Barbara

## 0. Abstract

Design space exploration during high level synthesis is often conducted through ad-hoc probing of the solution space using some scheduling algorithm. This is not only time consuming but also very dependent on designer’s experience. We propose a novel design exploration method that exploits the duality between the time and resource constrained scheduling problems. Our exploration automatically constructs a high quality time/area tradeoff curve in a fast, effective manner. It uses the MAX-MIN ant colony optimization to solve both the time and resource constrained scheduling problems. We switch between the time and resource constrained algorithms to quickly traverse the design space. Compared to using force directed scheduling exhaustively at every time step, our algorithm provides a significant solution quality savings (average 17.3% reduction of resource counts) with similar run time on a comprehensive benchmark suite constructed with classic and real-life samples. Our algorithms scale well over different applications and problem sizes.

高层次综合中的设计空间探索，通常是使用一些排程算法对解空间进行专门的探索。这耗时很长，而且非常依赖于设计者的经验。我们提出一种新的设计探索方法，探索时间和资源约束下的排程问题的对偶性。我们的探索自动构建了一个高质量的时间/面积折中曲线，非常快速高效。我们使用了MAX-MIN蚁群优化来求解时间和资源约束的排程问题。我们在时间约束和资源约束算法之间切换，来迅速遍历设计空间。与在每一步中都穷举式使用力有向的排程相比，我们的算法在一个综合基准测试上（用经典和真实世界的例子构成）运行，在类似的运行时间中，节省了很多求解质量（平均节约了17.3%的资源）。我们的算法在不同的应用和问题规模上扩展的很好。

**Keywords** Design Space Exploration, Ant Colony Optimization, MAX-MIN Ant System, Instruction Scheduling Algorithms

## 1. Introduction

When building a digital system, designers are faced with a countless number of decisions. Ideally, they must deliver the smallest, fastest, lowest power device that can implement the application at hand. More often than not, these design parameters are contradictory. For example, making the device run faster often makes it larger and more power hungry. With increasingly strict time to market, exploration tools that can quickly survey the design space and report a variety of options are invaluable.

当构建一个数字系统时，设计者会面临无数的决策。理想情况下，他们要给出最小，最快，功耗最低的设备，实现面临的应用。但更多的是，这些设计参数是相互矛盾的。比如，使设备运行的更快，通常会使其更大，能耗更高。随着产品上市时间越来越严格，可以很快探索设计空间并给出多个选项的探索工具，是非常有用的。

Resource and time constrained scheduling are two fundamental high level synthesis problems that are closely related with design space exploration. One possible space exploration method is to vary the constraints and probe for solutions in a point-by-point manner by solving either of these two problems. For instance, you can use some time constrained algorithm iteratively on different input latency. This will give you a number of solutions, and their various resource allocations over a set of time points. Similarly, performing resource constrained algorithm repetitively will provide a latency for each of the given area constraints.

资源和时间受限的排程，是两个基本的高层次综合问题，与设计空间探索紧密相关。一个可能的空间探索方法是，变化约束，求解这两个问题中的一个，以逐点的方式探索解。比如，可以对不同的输入延迟，迭代使用一些时间约束算法。这会给出几个解，和在几个时间点上的不同资源配置。类似的，重复的进行资源约束的算法，会对每个给定的面积约束，给出一个时间延迟。

In this case, the designers are often left with individual tools for tackling either time constrained or resource constrained problems. They must deal with questions such as: Where do we start in the design space? What is the best way to utilize the tools? When do we stop the exploration? Moreover, due to the lack of connection amongst the traditional methods, there is little information shared between time constrained and resource constrained solutions even though the two problems are tightly interwoven. This is unfortunate, as we are essentially throwing away potential solutions since solving one problem should offer more insight to the other problem. An effective design space exploration strategy must understand and exploit the relationship between these seemingly isolated points.

在这种情况下，设计者通常会用单个工具来解决时间约束或资源约束问题。他们必须处理类似下面的问题：我们在设计空间中的哪里开始？怎样最好的利用这些工具？我们怎样停止探索？而且，由于在传统方法之间缺少联系，在时间约束和资源约束的解之间，几乎没有任何共享信息，但这两个问题之间是紧密交织在一起的。这是很不幸的，因为我们正在扔掉可能的解，因为求解一个问题，应该对另外一个问题给出更多洞见。一个高效的设计空间探索策略，必须理解和利用这些看似孤立的点之间的关系。

In this paper, we describe a design space exploration strategy for scheduling and resource allocation. The ant colony optimization (ACO) meta-heuristic lies at the core of our algorithm. We switch between timing and resource constrained ACO heuristics to efficiently traverse the search space. Our algorithms dynamically adjust to the input application and produce a set of high quality Pareto optimal solutions across the design space.

本文中，我们描述了一个排程和资源配置的设计空间探索策略。蚁群优化(ant colony optimization, ACO)的元启发式算法，是我们算法的核心。我们在时间和资源约束的ACO启发式算法之间切换，以高效的遍历搜索空间。我们的算法对输入的应用进行动态的调整，产生了设计空间中高质量的Pareto最优解的集合。

## 2. Related Work

Design space exploration problems involving area cost and execution deadline tradeoffs are closely related with scheduling problems. Although these problems can be formulated with Integer Linear Programming(ILP), it is typically impossible to solve large problem instances in this manner. A lot of research work has been done to cleverly use heuristic approaches for addressing this problem. In [6, 10], genetic algorithms are implemented for design space exploration. In [5], the authors concentrate on providing alternative module bags by heuristically solving clique partitioning problems and using a force directed list scheduler. In the Voyager system [2], scheduling problems are solved by carefully bounding the design space using ILP, and good results are reported on small sized benchmarks. Other methods such as simulated annealing [9] also find their applications in this domain. A survey on design space exploration methodologies can be found in [8].

设计空间探索问题涉及到面积代价和执行的截止时间的折中，与排程问题是紧密相关的。虽然这些问题可以表述成整数线性规划(Integer Linear Programming, ILP)，但是一般不能以这种方式求解大型问题的实例。很多研究工作都很聪明的使用启发式方法来解决这个问题。在[6,10]中，使用了遗传算法来进行设计空间探索。在[5]中，作者聚焦在给出另外的模块包，启发式的求解分团划分问题，使用一个力有向列表排程器。在Voyager系统中[2]，使用ILP仔细的界定设计空间，来解决排程问题，在小规模的基准测试中得到了很好的结果。其他的方法比如模拟退火[9]，在这个领域中也有应用。[8]中给出了设计空间探索方法的综述。

Amongst the existing approaches, the most popular method is perhaps the force directed scheduling (FDS) algorithm [11], where the parallel usage of a resource type (called force) is used as the heuristic. It is a deterministic constructive method. Though it is reported to work well on small sized problems, the algorithm lacks good lookahead function. When the input application gets more complex or the desired deadline is big, collision happens between forces, which leads to inferior solutions. This phenomena is observed in our experiments reported in Section 5.

在现有的方法中，最流行的方法可能是力有向排程(force directed scheduling, FDS)算法[11]，其中使用一种资源方式的并行使用（称为力）作为启发式。这是一种确定性的构建性方法。虽然在小规模问题上效果很好，但算法缺少很好的lookahead函数。当输入的应用变得越来越复杂，或期望的deadline很大时，力之间就会发生冲突，这会导致次优解。我们在第5部分的试验中，会给出这种观察到的现象。

## 3. Time and Resource Constrained Duality

We are concerned with the design problem of making tradeoffs between hardware cost and timing performance. This is still a commonly faced problem in practice, and other system metrics, such as power consumption, are closely related with them. Based on this, our design space can be viewed as a two dimensional space illustrated in Figure 1(a), where the x-axis is the execution deadline and the y-axis is the aggregated hardware cost. In this space, each point represents a specific tradeoff of the two parameters.

我们关心的是设计过程中，硬件代价和计时性能之间的折中。这是实践中面临的一个常见问题，其他的系统衡量指标，比如能耗，与这些也紧密相关。基于此，我们的设计空间可以视为图1(a)中的二维空间，其中x轴是执行deadline，y轴是累积的硬件价值。在这个空间中，每个点都表示这两个参数的一个具体折中。

For a given application, the designer will be given R types of computing resources (e.g. multipliers and adders) to map the application onto. We define a specific design as a configuration, which is simply the number of instances for each resource type. In order to keep the discussion simple, in the rest of the paper we assume there are only two resource types M and A, though our algorithm is not limited to this constraint. Thus, each configuration can be specified by (m, a) where m is the number of resource M and a is the number of A. It is worth noticing that for each point in the design space shown in Figure 1(a), we might have multiple configurations that could realize it. Furthermore, for each specific configuration C with cost c, it covers a horizontal line in the design space starting at (t_min, c), where t_min is the resource constrained minimum scheduling time.

对于一个给定的应用，设计者会被给定R种计算资源（如，乘法器和加法器），将应用映射到这些资源中。我们定义一个具体的设计为一个配置，也就是每个资源类型的实例数量。为保持讨论简单，本文剩下部分，我们假设只有两种资源类型M和A，但我们的算法并没有局限在这种约束中。因此，每个配置可以由(m, a)指定，其中m是资源M的数量，a是资源A的数量。值得注意的是，对图1(a)中的设计空间中的每个点，我们可能有多个配置来实现。而且，对每个具体的配置C，其代价为c，其覆盖了设计空间中的一个水平线，从(t_min, c)开始，其中t_min是资源的最小排程时间约束。

The goal of design space exploration is to help the designer find the optimal tradeoff between the time and area. Theoretically, this can be done by performing time constrained scheduling (TCS) on all t in the interested range. These points form a curve in the design space, as illustrated by curve L in Figure 1(a). This curve divides the design space into two parts, labeled with F and U respectively in Figure 1(a), where all the points in F are feasible to the given application while U contains all the unfeasible time/area pairs. More interestingly, we have the following attribute for curve L (Proof is not given due to space limit):

设计空间探索的目标是，帮助设计者找到时间和面积之间的最优折中。理论上，通过对感兴趣范围内的所有t，都进行时间约束排程(TCS)，就可以完成。这些点在设计空间中形成了一个曲线，如图1a中的曲线L。这个曲线将设计空间分成了两部分，在图1a中分别标为F和U，F中的所有点对给定的应用来说都是可行的，而U包含所有不可行的时间/面积对。更有趣的是，对曲线L，有取下的性质（由于空间限制，就不给出证明了）：

Lemma 3.1 Curve L is monotonically non-increasing as the deadline t increases. 随着deadline t增加，曲线L是单调不增的。

Due to such a monotonically non-increasing property of curve L, there may exist horizontal segments along the curve. Based on our experience, horizontal segments appear frequently in practice. This motivates us to look into potential methods to exploit the duality between RCS and TCS to enhance the design space exploration process. First, we consider the following theorem:

由于曲线L有这样单调不增的性质，沿着曲线可能会存在水平的线段。基于我们的试验，在实践中水平线段会频繁出现。这促使我们去寻找可能的方法，来探索RCS和TCS之间的对偶性，增强设计空间探索的过程。首先，我们考虑下面的定理：

Theorem 3.2 If C is a configuration that provides the minimum cost c at time t1, then the resource constrained scheduling result t2 of C satisfies t2 ⩽ t1. More importantly, there is no configuration C' with a cost c' < c that can produce a minimum execution time
in the range of [t2, t1].

定理3.2 如果C是一个配置，在时间t1处给出最小代价c，那么C的资源约束排程结果t2满足t2 ⩽ t1。更重要的是，代价c' < c的配置C'，其最小执行时间范围在[t2, t1]中，这种情况是不存在的。

PROOF. The first part of the theorem is obvious. Therefore, we focus on the second part. Assuming there is a configuration C' that provides an execution time t3 ∈ [t2, t1], then C' must be able to produce t1. Since C' has a smaller cost, this conflicts with the fact that C is the minimum cost solution (i.e. the TCS solution) at time t1. Thus the statement is true. This is illustrated in Figure 1(b) with configuration (m1, a1) and (m', a').

证明：定理的前一部分是明显的。因此，我们关注在第二部分。假设有一个配置C'，可以给出的执行时间t3 ∈ [t2, t1]，那么C'一定能够产生t1。因为C'的代价更小，这与下面的事实矛盾，即C是时间t1处的最小代价解（即，TCS解）。因此这个表述是正确的。这如图1b中的配置(m1, a1)和(m', a')所示。

This theorem provides a key insight for the design space exploration problem. It says that if we can find a configuration with optimal cost c at time t1, we can move along the horizontal segment from (t1, c) to (t2, c) without losing optimality. Here t2 is the RCS solution for the found configuration. This enables us to efficiently construct the curve L by iteratively using TCS and RCS algorithms and leveraging the fact that such horizontal segments do frequently occur in practice. Based on the above discussion, we propose a new space exploration algorithm as shown in Algorithm 1 that exploits the duality between RCS and TCS solutions. Notice the min function in step 10 is necessary since a practical RCS algorithm may not return the true optimal and could be worse than t_cur.

这个定理对设计空间探索问题给出了一个关键洞见。即，如果我们可以找到一个配置，在时间t1处有最优代价c，我们可以将水平线段从(t1, c)移动到(t2, c)，而不损失最优性。这里t2是对找到的配置的RCS解。这使我们可以迭代的使用TCS和RCS算法，并利用这种水平线段在实践中频繁出现的事实，来高效的构建曲线L。基于上面的讨论，我们提出一个新的空间探索算法，如算法1所示，利用RCS和TCS解的对偶性。注意，在步骤10中的min函数是必须的，因为实际的RCS算法可能不会返回真正的最优结果，可能比t_cur要更差。

Algorithm 1: Iterative Design Space Exploration Algorithm

```
procedure DSE
output: curve L
1. interested time range [t_min, t_max],
2. L = φ
3. t_cur = t_max
4. while t_cur ⩾ t_min do
5.   perform TCS on t_cur to get the optimal configurations Ci.
6.   for configuration Ci do
7.     perform RCS to obtain the minimum time t_rcs^i
8.   end for
9.   t_rcs = min_i(t^i_rcs) /*find the best rcs time*/
10.  t_cur = min(t_cur, t_rcs) - 1
11.  extend L based on TCS and RCS results
12. end while
13. return L
```

## 4. Ant Colony Optimizations

In order to select the suitable TCS and RCS algorithms, we studied different scheduling approaches for the two problems, including the popularly used force directed scheduling (FDS) for the TCS problem [11], various list scheduling heuristics, and the recently proposed ant colony optimization (ACO) based instruction scheduling algorithms [13]. We chose the ACO approach for our design space exploration algorithm. Compared with traditional methods such as FDS and list scheduling, the ACO-based scheduling algorithms offer the following major benefits:

为选择合适的TCS和RCS算法，我们对这两个问题研究了不同的排程方法，包括对TCS问题的流行的力有向排程(FDS)[11]，各种列表排程启发式，和最近提出的基于蚁群优化(ACO)的指令排程算法[13]。我们选择ACO方法作为我们的设计空间探索算法。与传统的方法比如FDS和列表排程算法相比，基于ACO的排程算法有下面的好处：

• It generates better quality scheduling results that are close to the optimal with good stability for both the TCS and RCS problems [13]. 可以生成更高质量的排程结果，接近最优，对TCS和RCS问题的稳定性很好。

• It provides reasonable runtime. It has the same complexity as FDS method for the TCS problem. 运行时间合理。对TCS问题，与FDS方法的复杂度相同。

• More importantly, as a population based method, ACO-based approach naturally provides multiple alternative solutions. This is typically not available for traditional methods, especially for force directed TCS scheduling. This feature provides potential benefit in the iterative process for our algorithm since we can select the largest jump amongst these candidates. 更重要的是，作为一个基于population的方法，基于ACO的方法很自然的给出多个不同的解。对于传统方法来说，这一般是不可能的，尤其是对于TCS的FDS方法。这个特征在迭代过程中对我们的算法给出可能的好处，因为我们可以选择这些候选之间的最大跳。

Ant colony optimization was originally introduced by Dorigo et al. [4]. It is a population based approach inspired by ethological studies on the behavior of ants, in which a collection of agents cooperate together to explore the search space. They communicate via a mechanism imitating the pheromone trails, including auto-catalytic feedback and evaporation. One of the first problems to which ACO was successfully applied was the Traveling Salesman Problem (TSP) [4], for which it gave competitive results compared to traditional methods. Researchers have since formulated ACO methods for a variety of traditional NP-hard problems[3]. Effective algorithms have been constructed to solve time and resource constrained scheduling problems using the MAX-MIN Ant System(MMAS) [12], a variant of the original ACO approach. Space does not permit us to elaborate the algorithms here. Interested readers are encouraged to refer to [14, 13].

Dorigo等[4]最早提出了ACO。这是一个基于population的方法，受蚂蚁等动物行为学研究启发，其中智能体的集合一起合作，来探索搜索空间。它们通过一种模仿信息素轨迹的机制来进行通信，包括自催化的反馈和蒸发。ACO成功应用的第一类问题，是旅行商人问题(Traveling Salesman Problem, TSP)，与传统方法相比，ACO给出了非常有竞争力的结果。研究者因此对很多传统NP难题用ACO进行表述，解决。使用最大-最大小蚁群系统(MAX-MIN Ant System, MMAS)，构建了更有效的算法来求解时间和资源约束的排程问题，这是原始ACO方法的一种变体。限于空间，这里不能详尽的描述这种算法。感兴趣的读者可以参考[14,13]。

## 5. Experiments and Analysis

### 5.1 Benchmarks and Setup

In order to test and evaluate our algorithms, we have constructed a comprehensive set of benchmarks. These benchmarks are taken from one of two sources: (1) Popular benchmarks used in previous literature; (2) Real-life examples generated and selected from the MediaBench suite [7].

为测试和评估我们的算法，我们构建了一个综合基准测试集。这些基准测试是从下面两个源中取得的：(1)之前文献中使用的流行的基准测试；(2)从MediaBench包中生成和选择的真实世界中的例子。

The benefit of having classic samples is that they provide a direct comparison between results generated by our algorithm and results from previously published methods. This is especially helpful when some of the benchmarks have known optimal solutions. In our final testing benchmark set, seven samples widely used in instruction scheduling studies are included.

有经典例子的好处是，可以给出我们算法和之前发表的方法的结果之间的直接比较。当一些基准测试已知有最优解时，这尤其有帮助作用。在我们最终的测试基准测试集中，包含了7个在指令排程研究中广泛使用的7个例子。

However, these samples are typically small to medium in size, and are considered somewhat old. To be representative, it is necessary to create a more comprehensive set with benchmarks of different sizes and complexities. Such benchmarks shall aim to provide challenging samples for instruction scheduling algorithms with regards to larger number of operations, higher level of parallelism and data dependency on more up-to-date testing cases from modern and real-life applications. They should also help us with a wider range of synthesis problems to test the algorithms' scalability.

但是，这些案例一般规模都是小型到中型的，而且都有一些老。为具有代表性，需要创建一个更综合的集合，有不同大小和复杂度的基准测试。这种基准测试的目标，是为指令排程算法给出有挑战性的案例，有更多数量的运算，更高层次的并行度，依赖于更新的测试案例，从现代和真实世界的应用中得到。它们还应当在更广范围内的综合问题上有帮助，可以测试算法的可扩展性。

With the above goals, we investigated the MediaBench suite, which contains a wide range of complete applications for image processing, communications and DSP applications. We analyzed these applications using the SUIF and Machine SUIF and over 14,000 DFGs were extracted as preliminary candidates for our benchmark set. After careful study, 13 samples were selected ranging from matrix operation (the invertmatrix benchmark) to imaging processing algorithm (the jpeg idctifast and smoothcolor benchmarks).

有了上面的目标，我们研究了MediaBench包，包含了大量图像处理，通信和DSP的完整应用。我们用SUIF和Machine SUIF分析了这些应用，提取了超过14000个DFGs作为我们基准测试集的初步候选。在仔细的研究之后，选择了13个案例，从矩阵运算（invertmatrix基准测试），到图像处理算法（jpeg idctifast和smoothcolor基准测试）。

Table 1 lists all twenty benchmarks that were included in our final benchmark set. The “names” column gives the various functions where the basic blocks originated; the “size” column gives the the nodes/edges number pair. This benchmark set, including related statistics, DFG graphs and source code for the all testing samples, is available online [1].

表1列出了所有20个基准测试，组成了我们最终的基准测试集。名称列给出了各种函数，这是基础模块起源的地方；规模列给出了节点/边的数量对。基准测试集，包括所有测试案例的相关的统计，DFG图和源码，都已经开源。

We implemented three different design space exploration algorithms: 1) FDS: exhaustively step through the time range by performing time constrained force directed scheduling at each deadline;2) MMAS-TCS: step through the time range by performing only MMAS-based TCS scheduling at each deadline; and 3)MMAS-D: use the iterative approach proposed in Algorithm 1 by switching between MMAS-based RCS and TCS.

我们实现了三种不同的设计空间探索算法：1) FDS，穷举式的对所有时间范围，对每个deadline进行时间约束FDS；2) MMAS-TCS，对时间范围内的每个点，对每个deadline进行基于MMAS的TCS排程；3) MMAS-D，使用算法1中提出的迭代方法，在基于MMAS的RCS和TCS之间切换。

The MMAS-based TCS and RCS algorithms are similar to those described in Section [14]. Since there is no widely distributed and recognized FDS implementation, we implemented our own. The implementation is based on [11] and has all the applicable refinements proposed in the paper, including multi-cycle instruction support, resource preference control, and look-ahead using second order of displacement in force computation.

基于MMAS的TCS和RCS算法，与[14]中描述的算法类似。因为没有广泛承认和可用的FDS实现，我们实现了自己的FDS算法。这个实现是基于[11]的，包含了文章中所有可应用的精炼，包括multi-cycle instruction support，resource preference control，和look-ahead using second order of displacement in force computation。

For all testing benchmarks, the operations are allocated on two types of computing resources, namely MUL and ALU, where MUL is capable of handling multiplication and division, while ALU is used for other operations such as addition and subtraction. Furthermore, we define the operations running on MUL to take two clock cycles and the ALU operations take one. This definitely is a simplified case from reality, however, it is a close enough approximation and does not change the generality of the results. Other resource mappings can easily be implemented within our framework.

对于所有测试基准，运算分配在两种类型的计算资源中，即MUL和ALU，其中MUL可以处理乘法和出发，ALU用于其他运算，比如加法和减法。而且，我们定义了在MUL上的运算，消耗了2个时钟周期，ALU运算耗时1个时钟周期。这肯定是真实情况的一种简化，但是，这是很接近的一个近似，不会改变结果的一般性。其他资源映射也可以在我们的框架中很容易的实现。

With the assigned resource/operation mapping, ASAP is first performed to find the critical path delay Lc. We then set our predefined deadline range to be [Lc, 2Lc], i.e. from the critical path delay to 2 times of this delay. This results in 263 testing cases in total. Three design space exploration experiments are carried out. For the FDS and MMAS-TCS algorithms, we run force-directed or MMAS-based time constrained scheduling on every deadline and report the best schedule results together with the costs obtained. For the MMAS-D algorithm, we only run MMAS-based TCS on selected deadlines starting from 2Lc and make jumps based on the MMAS RCS results on the TCS results obtained previously.

有了指定的资源/运算映射，首先进行ASAP以找到关键路径延迟Lc。我们然后设预定义的deadline范围为[Lc, 2Lc]，即，从关键路径延迟到延迟的两倍。这是总计263个测试案例的结果。进行了3个设计空间探索的试验。对FDS和MMAS-TCS算法，我们对每个deadline运行FDS或基于MMAS的TCS，给出最优的排程结果，与得到的代价。对于MMAS-D算法，我们只在从2Lc开始的选定的deadlines中运行基于MMAS的TCS，起跳数基于MMAS RCS对之前得到的TCS结果之上。

### 5.2 Quality Assessment

As discussed in Section 5.1, we perform three experiments on all the benchmark samples using different algorithms. First, time constrained FDS scheduling is used at every deadline. The quality of results is used as the baseline for quality performance assessment. Then MMAS-TCS and MMAS-D algorithms are executed; the difference is that MMAS-TCS steps through the design space in the same way as FDS while MMAS-D uses duality between TCS and RCS. Each of these two algorithms are executed five times in order to obtain enough statistics to evaluate their stability.

如5.1节讨论，我们在所有基准测试案例中使用不同的算法进行三个试验。首先，对每个deadline使用时间约束的FDS排程算法。结果的质量用作质量性能评估的基准。然后运行MMAS-TCS和MMAS-D算法；区别是，MMAS-TCS与FDS遍历设计空间的方式一样，而MMAS-D使用TCS和RCS的对偶。这两个算法的每一个都执行5次，以得到足够的统计数据来评估其稳定性。

Detailed design space exploration results for some of benchmark samples are shown in Figure 2, where we compare the curves obtained by MMAS-D and FDS algorithms. Table 1 summarizes the experiment results. In each row, together with the benchmark name, we give the node/edge count, the average resource saving obtained MMAS-TCS and MMAS-D algorithms comparing with FDS. Though we do use different cost weights to bias alternative solutions (for example, solution (3M, 4A) is more favorable than (4M, 3A) as resource M has a large cost weight), we report the saving in percentage of total resource counts. We feel this is more objective and avoids confusion caused by different weight choices. The saving is computed for every deadline used for each benchmark, then the average for a certain benchmark is taken and reported in Table 1. It is easy to see that MMAS-TCS and MMAS-D both outperform the classic FDS method across the board with regard to solution quality, often with significant savings. Overall, MMAS-TCS achieves an average improvement of 16.4% while MMAS-D obtains a 17.3% improvement. Both algorithms scale well for different benchmarks and problem sizes. Moreover, by computing the standard deviation over the 5 different runs, the algorithms are shown to be very stable. For example, the average standard deviation on result quality for MMAS-TCS is only 0.104.

图2给出了对于一些基准测试案例的详细的设计空间探索结果，比较了MMAS-D和FDS算法的曲线结果。表1总结了试验结果。在每一列，与基准测试名称一起，我们给出了节点/边的数量，使用MMAS-TCS和MMAS-D算法与FDS算法相比，得到的平均资源节约率。虽然我们使用了不同的代价权重来对不同的解进行偏置（比如，解(3M, 4A)比(4M, 3A)更受欢迎，因为资源M有更大的代价权重），我们以总资源数量的百分比来给出节约率。我们认为这更加客观，避免了不同的权重选择导致的混淆。节约的百分比对每个基准测试的每个deadline进行计算，然后对特定的基准测试取其平均，并在表1中给出。很容易看出，MMAS-TCS和MMAS-D效果在整个榜上都比经典的FDS要好，通常有显著的节约。总体上，MMAS-TCS可以得到平均16.4%的改进，MMAS-D可以得到17.3%的改进。两种算法对不同的基准测试和问题规模扩展效果很好。而且，通过对5次不同的运行结果计算其标准差，算法表现非常稳定。比如，对MMAS-TCS的结果质量的平均标准差只有0.104。

It is interesting and initially surprising to observe that the MMAS-D always had better performance than MMAS-TCS method. More carefully inspection on the experiments reveals the reason: using the duality between TCS and RCS not only saves us computation but also improves the result quality. To understand this, we recall Theorem 3.2 and Figure 1(b). If we achieve an optimal solution at t1, with MMAS-D we automatically extend this optimality from t1 to t2, while an unperfect MMAS-TCS still have chance to provide worse quality solutions on deadlines between t1 and t2.

观察到MMAS-D比MMAS-TCS方法效果一直很好，这很有趣，而且令人惊讶。更仔细的观察试验结果，会得到其原因：使用TCS和RCS的对偶，不仅节约了计算时间，而且改进了结果质量。为理解这个，我们回忆一下定理3.2和图1b。如果我们在t1上获得了最优解，用MMAS-D，我们自动将这种最优性从t1拓展到t2，而不完美的MMAS-TCS仍然有可能在t1到t2的deadline中给出更坏质量的解。

All of the experiment results are obtained on a Linux box with a 2GHz CPU. Figure 3 diagrams the average execution time comparison for the three design space exploration approaches, ordered by the size of the benchmark. It is easy to see that the all the algorithms have similar run time scale, where MMAS-TCS takes more time, while MMAS-D and FDS have very close run times–especially on larger benchmarks. The major execution time savings come from the fact that MMAS-D exploits the duality and only computes TCS on selected number of deadlines. Over 263 testing cases, we find on average MMAS-D skips about 44% deadlines with the help of RCS. The fact that MMAS-D achieves much better results than FDS with almost the same execution time makes it very attractive in practice.

所有试验结果都是在2GHz CPU的Linux上得到的。图3给出了三个设计空间探索方法的平均执行时间的对比，以基准测试的规模进行排序。很容易看出，所有算法的运行时间尺度类似，其中MMAS-TCS消耗的时间更多一些，而MMAS-D和FDS的运行时间类似，尤其是在更大的基准测试中。主要的执行时间节约是来自，MMAS-D利用了对偶性，只在选定的deadlines中计算TCS。在263个测试案例中，我们发现平均MMAS-D在RCS的帮助下，跳过了44%的deadlines。MMAS-D比FDS获得了更好的结果，而且运行时间类似，这使其在实践中非常具有吸引力。

## 6. Conclusion

We proposed a novel design space exploration method that bridges the time and resource constrained scheduling problems and exploits the duality between them. Combined with the MMAS, our algorithms outperformed the popularly used force directed scheduling method with significant savings (average 17.3% savings on resources) with almost the same run time on comprehensive benchmarks constructed with classic and real-life samples. The algorithms scaled well over different applications and problem sizes.

我们提出了一种新的设计空间探索方法，很好的解决了资源和时间约束的排程问题，利用了它们之间的对偶性。与MMAS相结合，我们的算法超过了流行的FDS方法，节约了很多资源（平均节约了17.3%），运行时间相同，综合基准测试是用经典和真实世界的样本构建得到的。算法在不同的应用和问题规模上扩展非常好。