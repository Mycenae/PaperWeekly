# A tutorial on multiobjective optimization: fundamentals and evolutionary methods

## 0. Abstract

In almost no other field of computer science, the idea of using bio-inspired search paradigms has been so useful as in solving multiobjective optimization problems. The idea of using a population of search agents that collectively approximate the Pareto front resonates well with processes in natural evolution, immune systems, and swarm intelligence. Methods such as NSGA-II, SPEA2, SMS-EMOA, MOPSO, and MOEA/D became standard solvers when it comes to solving multiobjective optimization problems. This tutorial will review some of the most important fundamentals in multiobjective optimization and then introduce representative algorithms, illustrate their working principles, and discuss their application scope. In addition, the tutorial will discuss statistical performance assessment. Finally, it highlights recent important trends and closely related research fields. The tutorial is intended for readers, who want to acquire basic knowledge on the mathematical foundations of multiobjective optimization and state-of-the-art methods in evolutionary multiobjective optimization. The aim is to provide a starting point for researching in this active area, and it should also help the advanced reader to identify open research topics.

在计算机科学中的求解多目标优化问题中，使用生物启发的搜索范式的思想非常有用。使用一族搜索agents，集体近似Pareto front的思想，与自然进化，免疫系统，群体智能非常类似。在求解多目标优化问题时，NSGA-II，SPEA2，SMS-EMOA，MOPSO和MOEA/D这样的方法已经称为标准的求解器。本文会回顾多目标优化中的最重要基础，然后介绍代表性算法，描述其工作原则，讨论其应用范围。另外，本文会讨论统计性能评估。最后，强调了最近的重要趋势和紧密相关的研究领域。本文面向的对象是那些想要知道多目标优化的数学基础，和演化多目标优化的最新方法的。目标是为在这个活跃的领域的研究开个头，也能够为高级读者发现开放的研究课题。

**Keywords** Multiobjective optimization; Multiobjective evolutionary algorithms; Decomposition-based MOEAs; Indicator-based MOEAs; Pareto-based MOEAs; Performance assessment

## 1. Introduction

Consider making investment choices for an industrial process. On the one hand the profit should be maximized and on the other hand environmental emissions should be minimized. Another goal is to improve safety and quality of life of employees. Even in the light of mere economical decision making, just following the legal constraints and minimizing production costs can take a turn for the worse.

考虑对一个工业过程进行投资选择。一方面，利润应当最大化，另一方面，环境排放应当最小化。另一个目标是改进员工生命的安全和质量。即使仅仅按照经济决策，按照法律约束和最小化生产代价，也会使情况恶化。

Another application of multiobjective optimization can be found in the medical field. When searching for new therapeutic drugs, obviously the potency of the drug is to be maximized. But also the minimization of synthesis costs and the minimization of unwanted side effects are much-needed objectives (van der Horst et al. 2012; Rosenthal and Borschbach 2017).

多目标优化的另一个应用，可以在医学领域中找到。当寻找新的治疗药物时，显然药效应当最大化。但是合成代价要最小化，不想要的副作用也要最小化，这些都是很需要的目标。

There are countless other examples where multiobjective optimization has been applied or is recently considered as a promising field of study. Think, for instance, of the minimization of different types of error rates in machine learning (false positives, false negatives) (Yevseyeva et al. 2013; Wang et al. 2015), the optimization of delivery costs and inventory costs in logistics(Geiger and Sevaux 2011), the optimization of building designs with respect to health, energy efficiency, and cost criteria (Hopfe et al. 2012).

多目标优化的应用有无数个其他的例子。比如，机器学习中不同类型的错误率的最小化（假阳性，假阴性），物流中配送代价和库存代价的优化，建筑物设计对健康，能耗和代价的优化。

In the following, we consider a scenario where given the solutions in some space of possible solutions, the so-called decision space which can be evaluated using the so-called objective functions. These are typically based on computable equations but might also be the results of physical experiments. Ultimately, the goal is to find a solution on which the decision maker can agree, and that is optimal in some sense.

下面，我们考虑一个场景，以可能解的一些空间给出解，决策空间可以用所谓的目标函数来评估。这一般是基于可计算的等式的，但也可能是物理试验的结果。最终目标是，找到一个决策者会同意的解，同时要在某种意义上是最优的。

When searching for such solutions, it can be interesting to pre-compute or approximate a set of interesting solutions that reveal the essential trade-offs between the objectives. This strategy implies to avoid so-called Pareto dominated solutions, that is solutions that can improve in one objective without deteriorating the performance in any other objective. The Pareto dominance is named after Vilfredo Pareto, an Italian economist. As it was earlier mentioned by Francis Y. Edgeworth, it is also sometimes called Edgeworth-Pareto dominance (see Ehrgott 2012 for some historical background). To find or to approximate the set of non-dominated solutions and make a selection among them is the main topic of multiobjective optimization and multi-criterion decision making. Moreover, in case the set of non-dominated solutions is known in advance, to aid the decision maker in selecting solutions from this set is the realm of decision analysis (aka decision aiding) which is also part of multi-criterion decision making.

当要搜索这样的解时，预计算或近似这样的有趣的解，揭示这些目标之间的折中关系，就很有趣。这种策略意味着避免所谓的Pareto占优的解，即不需要恶化其他目标的性能，就可以改进一个目标的解。Pareto占优的关系是以一个意大利经济学家Vilfredo Pareto命名的。之前Francis Y. Edgeworth提到过，有时也被称为Edgeworth-Pareto占优。为找到或近似这样的非占优解，做出选择，是多目标优化和多原则决策的主要主题。而且，在非占优解提前已知的情况下，帮助决策者从这个集合中选择解，是决策分析的领域，这也是多原则决策的一部分。

**Definition 1** Multiobjective Optimization. Given m objective functions f1 : X -> R; . . .; fm : X -> R which map a decision space X into R, a multiobjective optimization problem (MOP) is given by the following problem statement:

$$minimize f1(x); . . .; minimize fm(x); x ∈ X$$(1)

**Remark 1** In general, we would demand m > 1 when we talk about multiobjective optimization problems. Moreover, there is the convention to call problems with large m, not multiobjective optimization problems but many-objective optimization problems (see Fleming et al. 2005; Li et al. 2015). The latter problems form a special, albeit important case of multiobjective optimization problems.

**Remark 2** Definition 1 does not explicitly state constraint functions. However, in practical applications constraints have to be handled. Mathematical programming techniques often use linear or quadratic approximations of the feasible space to deal with constraints, whereas in evolutionary multiobjective optimization constraints are handled by penalties that increase the objective function values in proportion to the constraint violation. Typically, penalized objective function values are always higher than objective function values of feasible solutions. As it distracts the attention from particular techniques in MOP solving, we will only consider unconstrained problems. For strategies to handle constraints, see Coello Coello (2013).

Considering the point(s) in time when the decision maker interacts or provides additional preference information, one distinguishes three general approaches to multiobjective optimization (Miettinen 2012):

1. A priori: A total order is defined on the objective space, for instance by defining a utility function Rm -> R and the optimization algorithm finds a minimal point (that is a point in X ) and minimum value concerning this order. The decision maker has to state additional preferences, e.g., weights of the objectives, prior to the optimization.

2. A posteriori: A partial order is defined on the objective space Rm, typically the Pareto order, and the algorithm searches for the minimal set concerning this partial order over the set of all feasible solutions. The user has to state his/her preferences a posteriori, that is after being informed about the trade-offs among non-dominated solutions.

3. Interactive (aka Progressive): The objective functions and constraints and their prioritization are refined by requesting user feedback on preferences at multiple points in time during the execution of an algorithm.

In the sequel, the focus will be on a posteriori approaches to multiobjective optimization. The a priori approach is often supported by classical single-objective optimization algorithms, and we refer to the large body of the literature that exists for such methods. The a posteriori approach, however, requires interesting modifications of theorems and optimization algorithms—in essence due to the use of partial orders and the desire to compute a set of solutions rather than a single solution. Interactive methods are highly interesting in real-world applications, but they typically rely upon algorithmic techniques used in a priori and a posteriori approaches and combine them with intermediate steps of preference elicitation. We will discuss this topic briefly at the end of the tutorial.

## 2. Related Work

There is a multiple of introductory articles that preceded this tutorial:

- In Zitzler et al. (2004) a tutorial on state-of-the-art evolutionary computation methods in 2004 is provided including Strength Pareto Evolutionary Algorithm Version 2 (SPEA2) (Zitzler et al. 2001), Non-dominated Sorting Genetic Algorithm II (NSGA-II) (Deb et al. 2002), Multiobjective Genetic Algorithm (MOGA) (Fonseca and Fleming 1993) and Pareto-Archived Evolution Strategy (PAES) (Knowles and Corne 2000) method. Indicator-based methods and modern variants of decomposition based methods, that our tutorial includes, were not available at that time.

- In Deb (2008) an introduction to earlier multiobjective optimization methods is provided, and also in the form of a tutorial. The article contains references to early books in this field and key articles and also discusses applications.

- Derivative-free methods for multiobjective optimization, including evolutionary and direct search methods, are discussed in Custódio et al. (2012).

- On conferences such as GECCO, PPSN, and EMO there have been regularly tutorials and for some of these slides are available. A very extensive tutorial based on slides is the citable tutorial by Brockhoff(2017).

Our tutorial is based on teaching material and a reader for a course on Multiobjective Optimization and Decision Analysis at Leiden University, The Netherlands (http://moda.liacs.nl). Besides going into details of algorithm design methodology, it also discusses foundations of multiobjective optimization and order theory. In the light of recent developments on hybrid algorithms and links to computational geometry, we considered it valuable to not only cover evolutionary methods but also include the basic principles from deterministic multiobjective optimization and scalarization-based methods in our tutorial.

## 3. Order and Dominance

For the notions we discuss in this section a good reference is Ehrgott (2005).

The concept of Pareto dominance is of fundamental importance to multiobjective optimization, as it allows to compare two objective vectors in a precise sense. That is, they can be compared without adding any additional preference information to the problem definition as stated in Definition 1.

In this section, we first discuss partial orders, pre-orders, and cones. For partial orders on Rm there is an important geometric way of specifying them with cones. We will define the Pareto order (aka Edgeworth-Pareto order) on Rm. The concept of Pareto dominance is of fundamental importance for multiobjective optimization, as it allows to compare two objective vectors in a precise sense (see Definition 5 below). That is, comparisons do not require adding any additional preference information to the problem definition as stated in Definition 1. This way of comparison establishes a pre-order (to be defined below) on the set of possible solutions (i.e., the decision space), and it is possible to search for the set of its minimal elements—the efficient set.

As partial orders and pre-orders are special binary relations, we digress with a discussion on binary relations, orders, and pre-orders.

...

...

...

### 3.1 Cone orders

### 3.2 Time complexity of basic operations on ordered sets

## 4. Scalarization techniques

Classically, multiobjective optimization problems are often solved using scalarization techniques (see, for instance, Miettinen 2012). Also in the theory and practice of evolutionary multiobjective optimization scalarization plays an important role, especially in the so-called decomposition based approaches.

多目标优化问题的经典方法是用标量化技术解决的。在演化多目标优化的理论和实践中，标量化扮演了一个重要的角色，尤其是在所谓的基于分解的方法中。

In brief, scalarization means that the objective functions are aggregated (or reformulated as constraints), and then a constrained single-objective problem is solved. By using different parameters of the constraints and aggregation function, it is possible to obtain different points on the Pareto front. However, when using such techniques, certain caveats have to be considered. In fact, one should always ask the following two questions:

简要来说，标量化意思是，目标函数聚积起来，然后解决一个约束单目标问题。使用约束和聚积函数的不同参数，可能得到Pareto front上不同的点。但是，当使用这样的技术时，必须考虑特定的说明。实际上，应当一直问下面的两个问题：

1. Does the optimization of scalarized problems result in efficient points?

2. Can we obtain all efficient points or vectors on the Pareto front by changing the parameters of the scalarization function or constraints?

标量化问题的优化会得到有效点吗？

通过改变标量化函数或约束的参数，可以得到Pareto front上的所有有效点吗？

We will provide four representative examples of scalarization approaches and analyze whether they have these properties.

我们给出标量化方法的四个典型例子，分析它们是否有这些性质。

### 4.1 Linear weighting

A simple means to scalarize a problem is to attach non-negative weights (at least one of them positive) to each objective function and then to minimize the weighted sum of objective functions. Hence, the multiobjective optimization problem is reformulated to:

标量化一个问题的一种简单方法是，给每个目标函数加上非负权重，最小化目标函数的加权和。因此，多目标优化问题重新表述为：

**Definition 14** Linear Scalarization Problem. The linear scalarization problem (LSP) of an MOP using a weight vector w is given by

$$minimize \sum_{i=1}^m w_if_i(x), x ∈ X$$

**Proposition 8** The solution of an LSP is on the Pareto front, no matter which weights in Rm are chosen.

### 4.2 Chebychev scalarization

Another means of scalarization, that will also uncover points in concave parts of the Pareto front, is to formulate the weighted Chebychev distance to a reference point as an objective function.

**Definition 17** Chebychev Scalarization Problem. The Chebychev scalarization problem (CSP) of an MOP using a weight vector λ ∈ Rm, is given by

$$minimize max_{i∈\{1,...,m\}} λ_i |f_i(x)-z_i^*|, x∈X$$

where z^* is a reference point, i.e., the ideal point defined as z_i^* = inf_{x∈X} f_i(x) with i = 1,...,m.

**Proposition 10** Let us assume a given set of mutually non-dominated solutions in Rm (e.g., a Pareto front). Then for every non-dominated point p there exists a set of weights for a CSP, that makes this point a minimizer of the CSP provided the reference point z* is properly chosen (i.e., the vector p - z* either lies in the positive or negative orthant).

### 4.3 ε-constraint method

A rather straightforward approach to turn a multiobjective optimization problem into a constraint single-objective optimization problem is the ε-constraint method.

**Definition 18** ε–constraint Scalarization. Given a MOP, the ε–constraint scalarization is defined as follows. Given m-1 constants ε1∈R, ..., εm-1∈R,

$$minimize f1(x), subject to g1(x)≤ε1, ..., gm-1(x)≤εm-1$$

where f1, g1, ...; gm-1 constitute the m components of vector function f of the multiobjective optimization problem (see Definition 1).

### 4.4 Boundary intersection methods

Another often suggested way to find an optimizer is to search for intersection points of rays with the attained subset f(X) (Jaszkiewicz and Słowiński 1999). For this method, one needs to choose a reference point in Rm, say r, which, if possible, dominates all points in the Pareto front. Alternatively, in the Normal Boundary Intersection method (Das and Dennis 1998) the rays can emanate from a line (in the bi-objective case) or an m-1 dimensional hyperplane, in which case lines originate from different evenly spaced reference points (Das and Dennis 1998). Then the following problem is solved:

**Definition 19** Boundary Intersection Problem. Let d ∈ Rm denote a direction vector and r ∈ Rm denote the reference vector. Then the boundary intersection problem is formulated as:

$$minimize t, subject to: (a)r+td-f(x) = 0, (b)x ∈ X, (c)t ∈ R$$

## 5. Numerical algorithms

Many of the numerical algorithms for solving multiobjective optimization problems make use of scalarization with varying parameters. It is then possible to use single-objective numerical optimization methods for finding different points on the Pareto front.

Besides these, there are methods that focus on solving the Karush-Kuhn-Tucker conditions. These methods aim for covering all solutions to the typically underdetermined nonlinear equation system given by these condition. Again, for the sake of clarity and brevity, in the following treatment, we will focus on the unconstrained case, noting that the full Karush-Kuhn-Tucker and Fritz-John conditions also feature equality and inequality constraints (Kuhn and Tucker 1951).

## 6. Evolutionary multiobjective optimization

Evolutionary algorithms are a major branch of bio-inspired search heuristics, which originated in the 1960ties and are widely applied to solve combinatorial and non-convex numerical optimization problems. In short, they use paradigms from natural evolution, such as selection, recombination, and mutation to steer a population (set) of individuals (decision vectors) towards optimal or near-optimal solutions (Bäck 1996).

演化算法是生物启发的搜索启发式的主要分支，起源于1960s，广泛应用于求解组合和非凸数值优化问题。简短来说，它们使用的范式来自于自然演化，比如选择，重新组合，和变异，来操纵一族个体（决策向量）朝向最优解或接近最优解。

Multiobjective evolutionary algorithms (MOEAs) generalize this idea, and typically they are designed to gradually approach sets of Pareto optimal solutions that are well-distributed across the Pareto front. As there are—in general—no single-best solutions in multiobjective optimization, the selection schemes of such algorithms differ from those used in single-objective optimization. First MOEAs were developed in the 1990ties—see, e.g., Kursawe (1990) and Fonseca and Fleming (1993), but since around the year 2001, after the first book devoted exclusively to this topic was published by Deb (2001), the number of methods and results in this field grew rapidly.

多目标演化算法(MOEAs)泛化了这个思想，一般设计用于逐渐接近Pareto最优解集，分布于Pareto front附近。因为在多目标优化中，一般并没有单个的最优解，这种算法的选择方案与在单目标优化使用的是不太一样的。第一种MOEAs是在1990s提出的，但自从2001年左右，出版了一本这个领域的专著后，这个领域的方法迅速增多了起来。

With some exceptions, the distinction between different classes of evolutionary multiobjective optimization algorithms is mainly due to the differences in the paradigms used to define the selection operators, whereas the choice of the variation operators is generic and dependent on the problem. As an example, one might consider NSGA-II (see Deb et al. 2002) as a typical evolutionary multiobjective optimization algorithm; NSGA-II can be applied to continuous search spaces as well as to combinatorial search spaces. Whereas the selection operators stay the same, the variation operators (mutation, recombination) must be adapted to the representations of solutions in the decision space.

不同类型的演化多目标优化算法之间的区别，主要是由于定义选择算子的范式有区别，而变化算子的选择是通用的，依赖于问题。一个例子是，可能考虑NSGA-II作为一种典型的演化多目标优化算法；NSGA-II可以用于连续的搜索空间，也可以用于组合搜索空间。尽管选择算子保持一致，变化算子（变异，重组合）必须调整以适应决策空间的解的表示。

There are currently three main paradigms for MOEA designs. These are: 现在主要有三种MOEA设计的范式，如下所述：

1. Pareto based MOEAs: The Pareto based MOEAs use a two-level ranking scheme. The Pareto dominance relation governs the first ranking and contributions of points to diversity is the principle of the second level ranking. The second level ranking applies to points that share the same position in the first ranking. NSGA-II (see Deb et al. 2002) and SPEA2 (see Zitzler and Thiele 1999) are two popular algorithms that fall into this category.

基于Pareto的MOEAs：基于Pareto的MOEAs使用两级ranking的方案。Pareto占优关系主宰了第一种ranking，点对多样性的贡献是第二级ranking的规则。第二级ranking应用的点，与第一级ranking共享相同的位置。NSGA-II和SPEA2是两种流行的这个类别的算法。

2. Indicator based MOEAs: These MOEAs are guided by an indicator that measures the performance of a set, for instance, the hypervolume indicator or the R2 indicator. The MOEAs are designed in a way that improvements concerning this indicator determine the selection procedure or the ranking of individuals.

基于Indicator的MOEAs：这些MOEAs由indicator引导，度量了一个集合的性能，比如，hypervolume indicator或R2 indicator。MOEAs的设计方式是，关于这个indicator的改进，决定了选择过程，或个体的ranking。

3. Decomposition based MOEAs: Here, the algorithm decomposes the problem into several subproblems, each one of them targeting different parts of the Pareto front. For each subproblem, a different parametrization (or weighting) of a scalarization method is used. MOEA/D and NSGA-III are well-known methods in this domain.

基于分解的MOEAs：这里，算法将问题分解成几个子问题，每个的目标是Pareto front的不同部分。对每个子问题，使用一种不同的标量化参数化或加权。MOEA/D和NSGA-III是这个类别的有名方法。

In this tutorial, we will introduce typical algorithms for each of these paradigms: NSGA-II, SMS-EMOA, and MOEA/D. We will discuss important design choices, and how and why other, similar algorithms deviate in these choices.

本文中，我们会介绍每种类别的典型算法：NSGA-II，SMS-EMOA，和MOEA/D。我们会讨论重要的设计选择，其他类似的算法与这些选项怎样不同，以及为什么。

### 6.1 Pareto based algorithms: NSGA-II

The basic loop of NSGA-II (Deb et al. 2002) is given by Algorithm 1.

Firstly, a population of points is initialized. Then the following generational loop is repeated. This loop consists of two parts. In the first, the population undergoes a variation. In the second part, a selection takes place which results in the new generation-population. The generational loop repeats until it meets some termination criterion, which could be convergence detection criterion (cf. Wagner et al. 2009) or the exceedance of a maximal computational budget.

首先，初始化一族点，然后重复下面的代际循环。这个循环包括两部分。在第一部分，种群经历了一个变化。在第二个部分，经过选择，产生新的一代种群。代际循环重复，直到满足某个停止规则，可能是收敛检测规则，或超过了最大的计算代价。

In the variation part of the loop λ offspring are generated. For each offspring, two parents are selected. Each one of them is selected using binary tournament selection, that is drawing randomly two individuals from Pt and selecting the better one concerning its rank in the population. The parents are then recombined using a standard recombination operator. For real-valued problems simulated binary crossover (SBX) is used (see Deb and Argawal 1995). Then the resulting individual is mutated. For real-valued problem polynomial mutation (PM) is used (see Mateo and Alberto 2012). This way, λ individuals are created, which are all combinations or modifications of individuals in Pt. Then the parent and the offspring populations are merged into Pt ∪ Qt.

在变化部分，生成循环的λ子代。对每个子代，选择两个父辈。每个都用二值锦标赛选择来进行选择，即从Pt中随机抽取2个个体，按照其在种群中的rank选择较好的那个。父辈使用标准重组合算子进行重新组合。对于实值问题，使用仿真二值杂交。然后得到的个体进行变异。对于实值问题，使用多项式变异。这样就创建了λ个个体，都是Pt中的个体的组合或改变。然后，父辈和子代种群进行合并Pt ∪ Qt。

In the second part, the selection part, the μ best individuals of Pt ∪ Qt with respect to a multiobjective ranking are selected as the new population Pt+1.

在第二部分，选择部分，在Pt ∪ Qt中，选择按照多目标ranking最好的μ个个体，作为新的种群Pt+1。

Next we digress in order to explain the multiobjective ranking which is used in NSGA-II. The key ingredient of NSGA-II that distinguishes it from genetic algorithms for single-objective optimization, is the way the individuals are ranked. The ranking procedure of NSGA-II consists of two levels. First, non-dominated sorting is performed. This ranking solely depends on the Pareto order and does not depend on diversity. Secondly, individuals which share the same rank after the first ranking are then ranked according to the crowding distance criterion which is a strong reflection of the diversity.

下面，我们解释一下NSGA-II中使用的多目标ranking。NSGA-II与单目标优化的遗传算法有区别的关键部分，是个体分等级的方法。NSGA-II的分等级的过程，包括两个级别。首先，进行非占优排序。这种分等级只依赖于Pareto order，不依赖于diversity。第二，在第一次分等级后，在同一等级的个体，然后会根据拥挤度距离准则来分等级，这是diversity的很强的反应。

Let ND(P) denote the non-dominated solutions in some population. Non-dominated sorting partitions the populations into subsets (layers) based on Pareto non-dominance and it can be specified through recursion as follows.

令ND(P)表示某种群中的非占优解。非占优排序，按照Pareto非占优，将种群分成子集（层），可以按照下式通过递归指定。

$$R1 = ND(P)$$(3)

$$Rk+1 = ND(P\∪_{i=1}^k R_i), k = 1,2,...$$(4)

As in each step of the recursion at least one solution is removed from the population, the maximal number of layers is |P|. We will use the index l to denote the highest non-empty layer. The rank of the solution after non-dominated sorting is given by the subindex k of Rk. It is clear that solutions in the same layer are mutually incomparable. The non-dominated sorting procedure is illustrated in Fig. 5 (upper left). The solutions are ranked as follows R1= {y(1), y(2), y(3), y(4)}, R2 = {y(5), y(6), y(7)}, R3 = {y(8), y(9)}.

在递归的每一步骤中，至少一个解从种群中移除，层的最大数量为|P|。我们使用索引l来表示最高的非空层。在非占优排序后解的等级由Rk的子索引k给出。很清楚，在相同层的解是互相不可比的。非占优排序过程如图5所示。解的等级如下。

Now, if there is more than one solution in a layer, say R, a secondary ranking procedure is used to rank solutions within that layer. This procedure applies the crowding distance criterion. The crowding distance of a solution x ∈ R is computed by a sum over contributions ci of the i-th objective function:

现在，如果在层R中有多于一个解，则用第二个分等级过程，来对层内的解分等级。这个过程应用拥挤度距离准则。一个解x ∈ R的拥挤度距离，通过第i个目标函数的贡献ci的和来计算：

$$li(x):=max({fi(y)|y∈R\{x}∧fi(y)≤fi(x)} ∪ {-∞})$$(5)

$$ui(x):=min({fi(y)|y∈R\{x}∧fi(y)≥fi(x)} ∪ {∞})$$(6)

$$ci(x):=ui-li, i=1,...,m$$(7)

The crowding distance is now given as:

$$c(x):=\sum_{i=1}^m ci(x)/m, x∈R$$(8)

For m = 2 the crowding distances of a set of mutually non-dominated points are illustrated in Fig. 5 (upper right). In this particular case, they are proportional to the perimeter of a rectangle that just is intersecting with the neighboring points (up to a factor of 1/4). Practically speaking, the value of li is determined by the nearest neighbor of x to the left according to the i-coordinate, and li is equal to the i-th coordinate of this nearest neighbor, similarly the value of ui is determined by the nearest neighbor of x to the right according to the i-coordinate, and ui is equal to the i-th coordinate of this right nearest neighbor. The more space there is around a solution, the higher is the crowding distance. Therefore, solutions with a high crowding distance should be ranked better than those with a low crowding distance in order to maintain diversity in the population. This way we establish a second order ranking. If the crowding distance is the same for two points, then it is randomly decided which point is ranked higher.

对于m=2，图5中展示了互相非占优的点的集合的拥挤度距离。在这个特定的情况中，这与和相邻点相交的矩形周长成比例（最多到1/4）。实践来说，li的值是由x的最邻域点到左边的i坐标决定的，li等于这个最邻域点的第i坐标，类似的，ui的值是由x的最邻域点根据i坐标到右边的距离确定的，ui等于最右邻域的第i个坐标。一个解周围的空间越多，拥挤度距离越大。因此，拥挤度距离大的解，等级要比拥挤度距离低的解高，以保持种群中的多样性。这样，我们确立了二阶等级。如果对于两个点的拥挤度距离相同，哪个点等级更高就随机决定。

Now we explain the non-dom_sort procedure in line 13 of Algorithm 1 the role of P is taken over by Pt ∩ Qt : In order to select the μ best members of Pt ∪ Qt according to the above described two level ranking, we proceed as follows. Create the partition R1, R2, ..., Rl of Pt ∪ Qt as described above. For this partition one finds the first index iμ for which the sum of the cardinalities |R1|+...+|Riμ| is for the first time ≥ μ. If |R1|+...+|Riμ| = μ, then set Pt+1 to $∪_{i=1}^iμ Ri$, otherwise determine the set H containing μ - (|R1|+...+|Riμ|) elements from Riμ with the highest crowding distance and set the next generation-population, Pt+1, to $(∪ _{i=1}^{iμ-1} Ri)∪H$.

现在我们解释算法1中第13行的non-dom_sort。P的角色被Pt ∩ Qt替代，以根据上述两级分等级规则来选择Pt ∪ Qt中最好的μ个成员，下面我们进行描述。创建Pt ∪ Qt的分割R1, R2, ..., Rl，对这个分割，找到第一个使下面的基的和R1|+...+|Riμ| ≥ μ的索引iμ。如果|R1|+...+|Riμ| = μ，那么设Pt+1为$∪_{i=1}^iμ Ri$，否则确定集合H包含Riμ中的μ - (|R1|+...+|Riμ|)个元素，有最高的拥挤度距离，设下一代种群Pt+1为$(∪ _{i=1}^{iμ-1} Ri)∪H$。

Pareto-based Algorithms are probably the largest class of MOEAs. They have in common that they combine a ranking criterion based on Pareto dominance with a diversity based secondary ranking. Other common algorithms that belong to this class are as follows. The Multiobjective Genetic Algorithm (MOGA) (Fonseca and Fleming 1993), which was one of the first MOEAs. The PAES (Knowles and Corne 2000), which uses a grid partitioning of the objective space in order to make sure that certain regions of the objective space do not get too crowded. Within a single grid cell, only one solution is selected. The Strength Pareto Evolutionary Algorithm (SPEA) (Zitzler and Thiele 1999) uses a different criterion for ranking based on Pareto dominance. The strength of an individual depends on how many other individuals it dominates and by how many other individuals dominate it. Moreover, clustering serves as a secondary ranking criterion. Both operators have been refined in SPEA2 (Zitzler et al. 2001), and also it features a strategy to maintain an archive of non-dominated solutions. The Multiobjective Micro GA Coello and Pulido (2001) is an algorithm that uses a very small population size in conjunction with an archive. Finally, the Differential Evolution Multiobjective Optimization (DEMO) (Robic and Filipic 2005) algorithm combines concepts from Pareto-based MOEAs with a variation operator from differential evolution, which leads to improved efficiency and more precise results in particular for continuous problems.

基于Pareto的算法是MOEAs的最大类别。它们的共同点是，将基于Pareto占优的分等级准则，与基于多样性的次级分等级结合到了一起。这个类别中其他常见的算法如下。多目标遗传算法(MOGA)，是第一批MOEAs。PAES使用了目标空间的网格划分，以确保目标空间的特定区域不会变得太拥挤。在单个网格单元中，只选择一个解。SPEA使用不同的准则进行基于Pareto占优的分等级。一个个体的强度，依赖于对多少其他的个体占优，有多少其他个体对它占优。而且，聚类作为次级分等级的准则。在SPEA2中，两个算子都得到了改进，这也是维护非占优解档案的一种策略的特色。多目标微GA这种算法，将小种群与档案一起使用。最后，微分演化多目标优化(DEMO)算法，将基于Pareto的MOEAs的概念，与微分演化中的变化算子结合到一起，得到了改进的效率和更精确的结果，特别是对连续问题。

### 6.3 Indicator-based algorithms: SMS-EMOA

A second algorithm that we will discuss is a classical algorithm following the paradigm of indicator-based multiobjective optimization. In the context of MOEAs, by a performance indicator (or just indicator), we denote a scalar measure of the quality of a Pareto front approximation. Indicators can be unary, meaning that they yield an absolute measure of the quality of a Pareto front approximation. They are called binary, whenever they measure how much better one Pareto front approximation is concerning another Pareto front approximation.

我们要讨论的另一类算法是一种经典算法，即基于indicator的多目标优化。在MOEAs的上下文中，性能指示器，或indicator，意思是一种Pareto front近似质量的一个标量度量。Indicators可以是一元的，意思是会对一个Pareto front的近似质量，产生一个绝对的度量。如果衡量一个Pareto front近似比另一个Pareto front近似要好多少，那就是二元的。

The SMS-EMOA (Emmerich et al. 2005) uses the hypervolume indicator as a performance indicator. Theoretical analysis attests that this indicator has some favorable properties, as the maximization of it yields approximations of the Pareto front with points located on the Pareto front and well distributed across the Pareto front. The hypervolume indicator measures the size of the dominated space, bound from above by a reference point.

SMS-EMOA使用hypervolume indicator作为性能指示器。理论分析证实，这种指示器有一些很好的性质，其最大化产生的Pareto front近似，点会在Pareto front上，会在Pareto front附近分布的很好。Hypervolume indicator衡量占优空间的大小，以上面的一个参考点作为界限。

For an approximation set A ∈ Rm it is defined as follows:

$$HI(A) = Vol({y∈Rm: y ≤_ {Pareto} r ∧ ∃a∈A: a ≤_{Pareto} y})$$(9)

Here, Vol(·) denotes the Lebesgue measure of a set in dimension m. This is length for m = 1, area for m = 2, volume for m = 3, and hypervolume for m ≥ 4. Practically speaking, the hypervolume indicator of A measures the size of the space that is dominated by A. The closer points move to the Pareto front, and the more they distribute along the Pareto front, the more space gets dominated. As the size of the dominated space is infinite, it is necessary to bound it. For this reason, the reference point r is introduced.

这里Vol(·)表示m维中一个集合的Lebesgue度量。对m=1为长度，m=2为面积，m=3为体积，m≥4为超体。实际说来，A的hypervolume indicator度量的是A所占优空间的大小。点移动到距离Pareto front越近的地方，它们沿着Pareto front分布的就越多，占优的空间就越大。由于被占优的空间是无限大的，有必要设置一个界限。为此，引入了参考点r。

The SMS-EMOA seeks to maximize the hypervolume indicator of a population which serves as an approximation set. This is achieved by considering the contribution of points to the hypervolume indicator in the selection procedure. Algorithm 2 describes the basic loop of the standard implementation of the SMS-EMOA.

SMS-EMOA对作为近似集的种群的hypervolume indicator进行最大化。通过在选择过程中考虑点对hypervolume indicator的贡献，来完成这个过程。算法2描述了SMS-EMOA标准实现的基本循环。

The algorithm starts with the initialization of a population in the search space. Then it creates only one offspring individual by recombination and mutation. This new individual enters the population, which has now size μ+1. To reduce the population size again to the size of μ, a subset of size μ with maximal hypervolume is selected. This way as long as the reference point for computing the hypervolume remains unchanged, the hypervolume indicator of Pt can only grow or stay equal with an increasing number of iterations t.

算法开始对搜索空间中的种群进行初始化。通过重新组合和变异，产生一个子代个体。这个个体进入种群，种群大小现在是μ+1。为将种群大小重新降低到μ，选择一个大小为μ的子集，其hypervolume最大。这样，只要计算hypervolume的参考点保持不变，随着迭代t的数值不断增加，Pt的hypervolume indicator肯定会增加，或保持不变。

Next, the details of the selection procedure will be discussed. If all solutions in Pt are non-dominated, the selection of a subset of maximal hypervolume is equivalent to deleting the point with the smallest (exclusive) hyper-volume contribution. The hypervolume contribution is defined as:

下一步，讨论选择过程的细节。如果Pt中的所有解都是非占优的，最大hypervolume的子集的选择，等价于将最小hypervolume贡献的点删掉。Hypervolume贡献定义为

$$∆HI(y,Y) = HI(Y) - HI(Y \ {y})$$

An illustration of the hypervolume indicator and hypervolume contributions for m = 2 and, respectively, m = 3 is given in Fig. 6. Efficient computation of all hypervolume contributions of a population can be achieved in time Θ(μlogμ) for m=2 and m=3 (Emmerich and Fonseca 2011). For m=3 or 4, fast implementations are described in Guerreiro and Fonseca (2017). Moreover, for fast logarithmic-time incremental updates for 2-D an algorithm is described in Hupkens and Emmerich (2013). For achieving logarithmic time updates in SMS-EMOA, the non-dominated sorting procedure was replaced by a procedure, that sorts dominated solutions based on age. For m > 2, fast incremental updates of the hypervolume indicator and its contributions were proposed in for more than two dimensions (Guerreiro and Fonseca 2017).

图6中给出了m=2和m=3情况下的hypervolume indicator和hypervolume contribution。一个种群中所有hypervolume contribution的高效计算，在m=2和m=3时为Θ(μlogμ)。对于m=3或4，有快速实现方式。而且，对于2D情况下的log时间增量更新，有文献给出了相应算法。要在SMS-EMOA中给出log时间更新，非占优排序过程替换为，一个基于age的对占优解进行排序的过程。对于m>2，hypervolume indicator及其contribution的快速增量更新，也有文献给出相应的算法。

In case dominated solutions appear the standard implementation of SMS-EMOA partitions the population into layers of equal dominance ranks, just like in NSGA-II. Subsequently, the solution with the smallest hypervolume contribution on the worst ranked layer gets discarded.

在被占优的解出现的情况下，SMS-EMOA的标准实现，将种群分割成相等占优等级的层，与NSGA-II中一样。后续，在最坏等级的层中，最小hypervolume贡献的解被抛弃。

SMS-EMOA typically converges to regularly spaced Pareto front approximations. The density of these approximations depends on the local curvature of the Pareto front. For bi-objective problems, it is highest at points where the slope is equal to -45◦ (Auger et al. 2009). It is possible to influence the distribution of the points in the approximation set by using a generalized cone-based hypervolume indicator. These indicators measure the hypervolume dominated by a cone-order of a given cone, and the resulting optimal distribution gets more uniform if the cones are acute, and more concentrated when using obtuse cones (see Emmerich et al. 2013).

SMS-EMOA一般收敛到均匀间隔的Pareto front近似。这些近似的密度，依赖于Pareto front的局部曲率。对于双目标的问题，在斜率等于-45度的地方，密度最高。要在近似集中影响点的分布，可以使用泛化的基于cone的hypervolume indicator。这些indicator度量的是，给定cone以cone-order占优的hypervolume的大小，如果cones是尖锐的，那么得到的最优分布就会更加均匀，如果cones很迟钝，那么最优分布就会很聚集。

Besides the SMS-EMOA, there are various other indicator-based MOEAs. Some of them also use the hypervolume indicator. The original idea to use the hypervolume indicator in an MOEA was proposed in the context of archiving methods for non-dominated points. Here the hypervolume indicator was used for keeping a bounded-size archive (Knowles et al. 2003). Besides, in an early work hypervolume-based selection which also introduced a novel mutation scheme, which was the focus of the paper (Huband et al. 2003). The term Indicator-based Evolutionary Algorithms (IBEA) (Zitzler and Künzli 2004) was introduced in a paper that proposed an algorithm design, in which the choice of indicators is generic. The hypervolume-based IBEA was discussed as one instance of this class. Its design is however different to SMS-EMOA and makes no specific use of the characteristics of the hypervolume indicator. The Hypervolume Estimation Algorithm (HypE) (Bader and Zitzler 2011) uses a Monte Carlo Estimation for the hypervolume in high dimensions and thus it can be used for optimization with a high number of objectives (so-called many-objective optimization problems). MO-CMA-ES (Igel et al. 2006) is another hypervolume-based MOEA. It uses the covariance-matrix adaptation in its mutation operator, which enables it to adapt its mutation distribution to the local curvature and scaling of the objective functions. Although the hypervolume indicator has been very prominent in IBEAs, there are some algorithms using other indicators, notably this is the R2 indicator (Trautmann et al. 2013), which features an ideal point as a reference point, and the averaged Hausdorff distance (∆p indicator) (Rudolph et al. 2016), which requires an aspiration set or estimation of the Pareto front which is dynamically updated and used as a reference. The idea of aspiration sets for indicators that require knowledge of the 'true' Pareto front also occurred in conjunction with the α-indicator (Wagner et al. 2015), which generalizes the approximation ratio in numerical single-objective optimization. The Portfolio Selection Multiobjective Optimization Algorithm (POSEA) (Yevseyeva et al. 2014) uses the Sharpe Index from financial portfolio theory as an indicator, which applies the hypervolume indicator of singletons as a utility function and a definition of the covariances based on their overlap. The Sharpe index combines the cumulated performance of single individuals with the covariance information (related to diversity), and it has interesting theoretical properties.

除了SMS-EMOA，还有很多其他基于indicator的MOEAs。其中一些也使用hypervolume indicator。在MOEA中使用hypervolume indicator的原始思想，是在对非占优点的归档方法的上下文中提出的。这里，hypervolume indicator用于保持一个有限大小的档案。除此以外，在一个早期工作中，基于hypervolume的选择还引入了一种新的变异方案，是文章的中心。一篇文章提出了术语基于indicator的演化算法(IBEA)，提出了一种算法设计，其中indicator的选择是通用的。基于hypervolume的IBEA作为这种类别中的一个实例来进行讨论。但是其设计与SMS-EMOA不同，没有很特别的利用hypervolume indicator的特征。Hypervolume Estimation Algorithm(HypE)对高维中的hypervolume使用了一种Mente Carlo估计，因此可以用在很多数量目标的优化中（即所谓的众目标优化问题）。MO-CMA-ES是另一个基于hypervolume的MOEA，在变异算子中使用了协方差矩阵的调整，使变异分布调整的随局部曲率变化，也随着目标函数的缩放而变化。虽然在IBEAs中hypervolume indicator非常重要，也有一些算法使用了其他的indicators，即R2 indicator，其特点是用一个理想的点作为参考点，还有∆p indicator，即平均Hausdorff距离，需要一个aspiration集或Pareto front的估计，进行动态更新，并用作参考。用在indicator上的aspiration sets的思想，需要真正的Pareto front的知识，这在α-indicator中也存在这个概念，泛化了数值单目标优化中的近似率。组合选择多目标优化算法(POSEA)使用了金融投资组合理论中的Sharpe Index作为indicator，应用singletons的hypervolume indicator作为一个效用函数，和基于其叠加的协方差的定义。Sharpe index将单个个体的累积性能和协方差信息（与多样性相关）结合到一起，其理论性质非常有趣。

### 6.3 Decomposition-based algorithm: MOEA/D

Decomposition-based algorithms divide the problem into subproblems using scalarizations based on different weights. Each scalarization defines a subproblem. The subproblems are then solved simultaneously by dynamically assigning and re-assigning points to subproblems and exchanging information from solutions to neighboring sub-problems.

基于分解的算法，使用基于不同权重的标量化，将问题分解成子问题。每个标量化定义了一个子问题。子问题同时进行解决，给子问题动态的指定点，重指定点，在相邻的子问题的解之间交换信息。

The method defines neighborhoods on the set of these subproblems based on the distances between their aggregation coefficient vectors. When optimizing a subproblem, information from neighboring subproblems can be exchanged, thereby increasing the efficiency of the search as compared to methods that independently solve the subproblems.

在这些子问题的集合上，基于它们之间聚积系数向量的距离，该方法定义了邻域。在优化一个子问题时，邻近子问题的信息可以进行交换，因此与独立求解这些子问题的方法相比，增加了搜索的效率。

MOEA/D (Zhang and Li 2007) is a very commonly used decomposition based method, that succeeded a couple of preceding algorithms based on the idea of combining decomposition, scalarization and local search(Ishibuchi and Murata 1996; Jin et al. 2001; Jaszkiewicz 2002). Note that even the early proposed algorithms VEGA (Schaffer 1985) and the vector optimization approach of Kursawe (see Kursawe 1990) can be considered as rudimentary decomposition based approaches, where these algorithms obtain a problem decomposition by assigning different members of a population to different objective functions. These early algorithmic designs used subpopulations to solve different scalarized problems. In contrast, in MOEA/D one population with interacting neighboring individuals is applied, which reduces the complexity of the algorithm.

MOEA/D是一种非常常用的基于分解的方法，继承了一些之前的基于分解，标量化和局部搜索的思想组合的算法。注意，即使是早期提出的算法VEGA和Kursawe的向量优化方法，也可以认为是初级的基于分解的方法，这些算法得到问题分解的方法是，将不同成员的种群，指定给不同的目标函数。这些早期的算法设计使用子种群来求解不同的标量化问题。对比起来，在MOEA/D中，应用的是带有相邻个体的相互作用的一个种群，降低了算法的复杂度。

Typically, MOEA/D works with Chebychev scalarizations, but the authors also suggest other scalarization methods, namely scalarization based on linear weighting—which however has problems with approximating non-convex Pareto fronts—and scalarization based on boundary intersection methods—which requires additional parameters and might also obtain strictly dominated points.

一般来说，MOEA/D使用Chebyshev标量化，但是作者也建议使用其他标量化方法，即基于线性加权的标量化，和基于边缘相交方法的标量化。

MOEA/D evolves a population of individuals, each individual x(i) ∈ Pt being associated with a weight vector λ(i). The weight vectors λ(i) are evenly distributed in the search space, e.g., for two objectives a possible choice is: λ(i) = ((λ-i)/λ, i/λ)^T, i = 0, ..., μ.

MOEA/D对一个种群的个体进行演化，每个个体x(i) ∈ Pt关联一个权重向量λ(i)。权重向量λ(i)在搜索空间中均匀分布，如，对两个目标，一个可能的选择是：λ(i) = ((λ-i)/λ, i/λ)^T, i = 0, ..., μ。

The i-th subproblem g(x|λi, z*) is defined by the Chebyshev scalarization function (see also Eq. 2):

第i个子问题g(x|λi, z*)，由Chebyshev标量化函数定义：

$$g(x|λi, z*) = max_{j∈{1,...,m}} {λj(i) |fj(x)-zj*|} + ε\sum_{j=1}^m (fj(x)-zj*)$$(10)

The main idea is that in the creation of a new candidate solution for the i-th individual the neighbors of this individual are considered. A neighbor is an incumbent solution of a subproblem with similar weight vectors. The neighborhood of i-th individual is the set of k subproblems, for so predefined constant k, that is closest to λ(i) in the Euclidean distance, including the i-th subproblem itself. It is denoted with B(i). Given these preliminaries, the MOEA/D algorithm — using Chebychev scalarization — reads as described in Algorithm 3.

主要思想是，在对第i个个体创建一个新的候选解时，会考虑这个个体的邻域。一个邻域是，有相似权重向量的子问题的现任解。第i个个体的邻域是k个子问题的集合，对于预定义的常数k，与λ(i)在欧式距离上最接近，包括第i个子问题本身，这表示为B(i)。给定了这些初步知识，MOEA/D算法如算法3所示，使用的是Chebyshev标量化。

Note the following two remarks about MOEA/D: (1) Many parts of the algorithm are kept generic. Here, generic options are recombination, typically instantiated by standard recombination operators from genetic algorithms, and local improvement heuristic. The local improvement heuristic should find a solution in the vicinity of a given solution that does not violate constraints and has a relatively good performance concerning the objective function values. (2) MOEA/D has additional statements to collect all non-dominated solutions it generates during a run in an external archive. Because this external archive is only used in the final output and does not influence the search dynamics, it can be seen as a generic feature of the algorithm. In principle, an external archive can be used in all EMOAs and could therefore also be done in SMS-EMOA and NSGA-II. To make comparisons to NSGA-II and SMS-EMOA easier, we omitted the archiving strategy in the description.

关于MOEA/D有如下两点要注意：(1)算法的很多部分是通用的。这里，通用的选项是重组合，一般是由遗传算法中的标准重组合算子实例化的，和局部改进的启发式。局部改进启发式应当在给定解的附近找到一个解，不违反约束，关于目标函数值有一个相对较好的性能。(2)MOEA/D有额外的语句，可以收集在一次运行中产生的所有非占优解，保存在外部档案中。因为这个外部档案只用在最终输出中，不影响搜索的过程，可以视为算法的一般特征。理论上，外部档案可以用在所有EMOAs中，因此可以在SMS-EMOA和NSGA-II中完成。为使得与NSGA-II和SMS-EMOA的比较更容易，我们在描述中忽略了存档的策略。

Recently, decomposition-based MOEAs became very popular, also because they scale well to problems with many objective functions. The NSGA-III (Deb and Jain 2014) algorithm is specially designed for many-objective optimization and uses a set of reference points that is dynamically updated in its decomposition. Another decomposition based technique is called Generalized Decomposition (Giagkiozis et al. 2014). It uses a mathematical programming solver to compute updates, and it was shown to perform well on continuous problems. The combination of mathematical programming and decomposition techniques is also explored in other, more novel, hybrid techniques, such as Directed Search (Schütze et al. 2016), which utilizes the Jacobian matrix of the vector-valued objective function (or approximations to it) to find promising directions in the search space, based on desired directions in the objective space.

最近，基于分解的MOEAs变得非常流行，也是因为在众目标函数中缩放的很好。NSGA-III算法就是特别为众目标优化设计的，使用的参考点集合是在其分解中动态更新的。另一种基于分解的技术称为泛化分解(Generalized Decomposition)。它使用数学规划求解器来计算更新，在连续问题上表现良好。数学规划和分解技术的组合在其他更新的混合技术中进行了探索，比如Directed Search，利用了向量值目标函数的Jacobian矩阵来在搜索空间中找到有希望的方向，基于目标空间的期望方向。

## 7. Performance assessment

In order to make a judgement (that is, gain insight into the advantages and disadvantages) of multiobjective evolutionary (or for that matter also deterministic) optimizers we need to take into account (1) the computational resources used, and (2) the quality of the produced result(s).

The current state of the art of multiobjective optimization approaches are mainly compared empirically though theoretical analyses exist (see, for instance, the convergence results described in Rudolph and Agapie (2000), Beume et al. (2011) albeit for rather simple problems as more realistic problems elude mathematical analysis.

The most commonly computational resource which is taken into account is the computation time which is very often measured implicitly by counting fitness function evaluations—in this respect, there is no difference with single-objective optimization. In contrast to single-objective optimization, in multiobjective optimization, a close distance to a (Pareto) optimal solution is not the only thing required but also good coverage of the entire Pareto front. As the results of multiobjective optimization algorithms are (finite) approximation sets to the Pareto front we need to be able to say when one Pareto front approximation is better than another. One good way to define when one approximation set is better than another is as in Definition 22 (see Zitzler et al. 2003).

**Definition 21** Approximation Set of a Pareto Front. A finite subset A of Rm is an approximation set of a Pareto front if and only if A consists of mutually (Pareto) non-dominated points.

**Definition 22** Comparing Approximation Sets of a Pareto Front. Let A and B be approximation sets of a Pareto front in Rm. We say that A is better than B if and only if every b ∈ B is weakly dominated by at least one element a ∈ A and A \neq B.

In Fig. 7 examples are given of the case of one set being better than another and in Fig. 8 examples are given of the case that a set is not better than another.

This relation on sets has been used in Zitzler et al. (2003) to classify performance indicators for Pareto fronts. To do so, they introduced the notion of completeness and compatibility of these indicators with respect to the set relation ‘is better than’.

...

## 8. Recent topics in multiobjective optimization

Recently, there are many new developments in the field of multiobjective optimization. Next we will list some of the most important trends.

### 8.1 Many-objective optimization

Optimization with more than 3 objectives is currently termed many-objective optimization [see, for instance, the survey (Li et al. 2015)]. This is to stress the challenges one meets when dealing with more than 3 objectives. The main reasons are:

1. problems with many objectives have a Pareto front which cannot be visualized in conventional 2D or 3D plots instead other approaches to deal with this are needed;

2. the computation time for many indicators and selection schemes become computationally hard, for instance, time complexity of the hypervolume indicator computation grows super-polynomially with the number of objectives, under the assumption that P \neq NP;

3. last but not least the ratio of non-dominated points tends to increase rapidly with the number of objectives. For instance, the probability that a point is non-dominated in a uniformly distributed set of sample points grows exponentially fast towards 1 with the number of objectives.

In the field of many-objective optimization different techniques are used to deal with these challenges. For the first challenge, various visualization techniques are used such as projection to lower dimensional spaces or parallel coordinate diagrams. In practice, one can, if the dimension is only slightly bigger than 3, express the coordinate values by colors and shape in 3D plots.

Naturally, in many-objective optimization indicators which scale well with the number of objectives (say polynomially) are very much desired. Moreover, decomposition based approaches are typically preferred to indicator based approaches.

The last problem requires, however, more radical deviations from standard approaches. In many cases, the reduction of the search space achieved by reducing it to the efficient set is not sufficiently adequate to allow for subsequent decision making because too many alternatives remain. In such cases, a stricter order than the Pareto order is required which requires additional preference knowledge. To elicit preference knowledge, interactive methods often come to the rescue. Moreover, in some cases, objectives are correlated which allows for grouping of objectives, and in turn, such groups can be aggregated to a single objective. Dimensionality reduction and community detection techniques have been proposed for identifying meaningful aggregation of objective functions.

### 8.2 Preference modeling

The Pareto order is the most applied order in multiobjective optimization. However, different ranking schemes and partial orders have been proposed in the literature for various reasons. Often additional knowledge of user preferences is integrated. For instance, One distinguishes preference modeling according to at what stage of the optimization the preference information is collected (a priori, interactively, and a posteriori). Secondly one can distinguish the type of information requested from the decision maker, for instance, constraints on the trade-offs, relative importance of the objectives, and preferred regions on the Pareto front. Another way to elicit preference information is by ordinal regression; here the user is asked for pairwise comparisons of some of the solutions. From this data, the weights of utility functions are learned (Branke et al. 2015). The interested reader is also referred to interesting work on non-monotonic utility functions, using the Choquet integral (Branke et al. 2016). Notably, the topic of preference elicitation is one of the main topics in Multiple Criteria Decision Analysis (MCDA). In recent years collaboration between MCDA and multiobjective optimization (MOO) brought forward many new useful approaches. A recommended reference for MCDA is Belton and Stewart (2002). For a comprehensive overview of preference modelling in multiobjective optimization we refer to Li et al. (2017) and Hakanen et al. (2016). Moreover Greco et al. (2016) contains an updated collection of state of the art surveys on MCDA. A good reference discussing the integration of MCDA into MOO is Branke et al. (2008).

### 8.3 Optimization with costly function

In industrial optimization very often the evaluation of (an) objective function(s) is achieved by simulation or experiments. Such evaluations are typically time-consuming and expensive. Examples of such costly evaluations occur in the optimization based on crash tests of automobiles, chemical experiments, computational fluid dynamics simulations, and finite element computations of mechanical structures. To deal with such problems techniques that need only a limited number of function evaluations have been devised. A common approach is to learn a surrogate model of the objective functions by using all available past evaluations. This is called surrogate model assisted optimization. One common approach is to optimize on the surrogate model to predict promising locations for evaluation and use these evaluations to further improve the model. In such methods, it is also important to add points for developing the model in under-explored regions of the search space. Some criteria such as expected improvement take both exploitation and exploration into account. Secondly, surrogate models can be used in pre-processing in the selection phase of evolutionary algorithms. To save time, less interesting points can be discarded before they would be evaluated by the costly and precise evaluator. Typically regression methods are used to construct surrogate models; Gaussian processes and neural networks are standard choices. Surrogate modeling has in the last decade been generalized to multiobjective optimization in various ways. Some important early work in this field was on surrogate assisted MOEAs (Emmerich et al. 2006) and ParEGO algorithm (Knowles 2006). A state of the art review can be found in Allmendinger et al. (2017).

### 8.4 New bio-inspired paradigms

Inspiration by nature has been a creative force for dealing with optimization algorithm design. Apart from biological evolution, many other natural phenomena have been considered. While many of these algorithmic ideas have so far remained in a somewhat experimental and immature state, some non-evolutionary bio-inspired optimization algorithms have gained maturation and competitive performance. Among others, this seems to hold for particle swarm optimization (Reyes-Sierra and Coello Coello 2006), ant colony optimization (Barán and Schaerer 2003), and artificial immune systems Coello Coello and Cortés (2005). As with evolutionary algorithms, also these algorithms have first been developed for single-objective optimization, and subsequently, they have been generalized to multiobjective optimization. Moreover, there is some recent research on bio-inspired techniques that are specifically developed for multiobjective optimization. An example of such a development is the Predator-Prey Evolutionary Algorithm, where different objectives are represented by different types of predators to which the prey (solutions) have to adapt (Laumanns et al. 1998; Grimme and Schmitt 2006).

In the field of natural computing, it is also investigated whether algorithms can serve as models for nature. It is an interesting new research direction to view aspects of natural evolution as a multiobjective optimization process, and first such models have been explored in Rueffler (2006) and Sterck et al. (2011).

### 8.5 Set-oriented numerical optimization

Traditionally, numerical techniques for multiobjective optimization are single point techniques: They construct a Pareto front by formulating a series of single-objective optimization problems (with different weights or constraints) or by expanding a Pareto front from a single solution point by point using continuation. In contrast, set-oriented numerical multiobjective optimization operates on the level of solution sets, the points of which are simultaneously improved, and that converge to a well-distributed set on the Pareto front. Only very recently such methods have been developed, and techniques that originated from evolutionary multiobjective optimization have been transferred into deterministic methods. A notable example is the hypervolume indicator gradient ascent method for multi-objective optimization (HIGA-MO) (Wang et al. 2017). In this method a set of say μ points is represented as a single vector of dimension μn, where n is the dimension of the search space. In real-valued decision space the mapping HI: Rμn -> R from the such population vectors to the hypervolume indicator has a well-defined derivative in almost all points. The computation of such derivatives has been described in Emmerich and Deutz (2014). Viewing the vector of partial derivatives as a gradient, conventional gradient methods can be used. It requires, however, some specific adaptations in order to construct robust and practically usable methods for local optimization. On convex problems, fast linear convergence can be achieved. By using second-order derivatives in a hypervolume-based Newton-Raphson method, even quadratic convergence speed has been demonstrated empirically on a set of convex bi-objective problems. The theory of such second-order methods is subject to ongoing research (Hernández et al. 2014).

### 8.6 Advanced performance assessment

Despite significant progress on the topic of performance assessment in recent years, there are still many unanswered questions. A bigger field of research is on performance assessment of interactive and many objective optimization.Moreover, the dependency of performance measures on parameters, such as the reference point of the hypervolume indicator requires further investigation. Some promising work in that direction was recently provided in Ishibuchi et al. (2017).

## 9. How to get started?

In the following, we list some of the main resources for the field of (Evolutionary) Multiobjective Optimization.

Introductory Books:

- Jürgen Branke, Kalyanmoy Deb, Kaisa Miettinen, Roman Slowiński Multiobjective Optimization: Interactive and evolutionary approaches, Springer, 2008

- Carlos Coello Coello et al. Evolutionary Algorithms for Solving Multi-Objective Problems, 2007, Springer

- Kalyanmoy Deb Multi-Objective Optimization using Evolutionary Algorithms, Wiley, 2001

- Matthias Ehrgott Multicriteria Optimization, Springer, 2005

- Joshua Knowles, David Corne, Kalyanmoy Deb Multiobjective Problem Solving from Nature, Springer, 2007

- Kaisa Miettinen Multiobjective Nonlinear Optimization, Kluwer, 2012

Websites:

- EMOO Repository by Carlos Coello Coello http://neo.lcc.uma.es/emoo/

- SIMCO Open Problems http://simco.gforge.inria.fr/doku.php?id=openproblems; a collection of open problems and theoretical results on indicator based approaches and complexity theory.

There are many implementations of multiobjective optimization algorithms available. Table 1 provides a table of MOO Software, including also some packages that include deterministic solvers.

Conferences and Journals:

- Conferences:

- Conference on Evolutionary Computation (CEC), annual, published by IEEE

- Evolutionary Multi-criterion Optimization (EMO) biannual conference, proceedings published by Springer LNCS

- EVOLVE—a Bridge between Probability, Set Oriented Numerics and Evolutionary Computation, annual until 2015, published by Springer Studies in Computational Intelligence, continued as NEO see below

- GECCO with EMO track, annual, published by ACM

- Global Optimization Workshop (GO), biannual, published by diverse publishers (as special issues, and post-proceedings)

- MCDM with EMO track, biannual, published by MCDM International Society

- Numerical and Evolutionary Optimization(NEO), annual, published by Springer Advances in Computational Intelligence

- and others

- Journals: COAP, ECJ, EJOR, IEEE TEVC, JOGO, MCDA Journal, and other Optimization, and Operations Research journals.

Aside from the resources mentioned above, there are many research groups and labs which maintain a repository of software accompanying their published research, e.g., the MODA group at Leiden University http://moda.liacs.nl and the research group of Carlos Fonseca at Coimbra University eden.dei.uc.pt/cmfonsec/software.html.

## 10 Summary and outlook

In this tutorial, we gave an introduction to the field of multiobjective optimization. We covered the topics of order-theoretical foundations, scalarization approaches, and optimality conditions. As solution methods, we discussed homotopy and evolutionary methods. In the context of Evolutionary methods, we discussed three state-of-the-art techniques in detail, namely NSGA-II, SMS-EMOA, and MOEA/D, each representing a key paradigm in evolutionary multiobjective algorithm design. NSGA-II served as a representative of Pareto based approaches, SMS-EMOA as an example of Indicator-based approaches, and MOEA/D as an example of decomposition based approaches. These algorithms have some advantages and disadvantages:

本教程中，我们介绍了多目标优化领域。我们覆盖了order理论基础，标量化方法，最优性条件的主题。至于解的方法，我们讨论了homotopy和演化方法。在演化方法下，我们详细讨论了三种目前最好的技术，即NSGA-II，SMS-EMOA和MOEA/D，每种都代表了演化多目标算法设计的一种关键范式。NSGA-II代表基于Pareto的方法，SMS-EMOA代表基于indicator的方法，MOEA/D代表基于分解的方法。这些算法有一些优势和缺点：

Pareto-based approaches follow a straightforward design principle, that is directly based on Pareto dominance and diversity preservation (using, for instance, crowding distance). Usually, these algorithms require only a few parameters, and larger numbers of objective functions do not cause problems. However, it might be difficult to guarantee and measure convergence and achieve a very regular spacing of solutions.

基于Pareto的方法按照很直接的设计原则，即直接基于Pareto占优性和多样性保持（比如，使用拥挤度距离）。通常，这些算法只需要几个参数，更多数量的目标函数，也不会导致问题。但是，很难确保和度量收敛性，获得间隔很规则的解。

Indicator-based approaches use an indicator for the performance of an approximation set to guide the search. It is possible to assess their convergence behavior online, and they hold the promise to be more amenable to theoretical analysis. However, the computation time often increases rapidly with the number of dimensions and the distribution of points in the approximation sets might depend critically on the settings of the reference point or other parameters of the indicator.

基于indicator的方法使用indicator作为一个近似集的性能，来引导搜索。可以在线评估其收敛行为，更经得起理论分析。但是，计算时间通常随着维度数量迅速增加，近似集中的点的分布非常依赖于参考点的设置，或indicator的其他参数。

Decomposition-based methods provide a very flexible framework for algorithm design, as they can incorporate various scalarization methods. A disadvantage is that they require some a priori knowledge of the position of the Pareto front in the objective space and the number of weight vectors might grow exponentially with the objective space size, even if the Pareto front is of low dimension.

基于分解的方法为算法设计提供了非常灵活的框架，因为可以纳入各种标量化方法。一个劣势是，他们需要一些关于Pareto front在目标空间中位置的先验知识，权重向量的数量可能随着目标空间大小呈指数级增加，即使Pareto front的维度较低。

According to the above, choosing the right method depends much on the dimension of the objective space, the number of solutions one wants to output, the desired distribution of the solutions (knee-point focused or uniformly spread) and the a priori knowledge on the location and shape of the Pareto front.

根据上述，选择正确的方法依赖于目标空间的维度，希望输出的解的数量，解的期望分布（聚集，或均匀散布），和对Pareto front的位置和形状的先验知识。

Due to space constraints, many advanced topics in multiobjective optimization are not covered in depth. We refer for these topics to the literature. For instance, constraint handling (Coello Coello 2013), multimodality (Kerschke et al. 2016), non-convex global optimization (Žilinskas 2013), and combinatorial optimization (Ehrgott and Gandibleux 2000).

由于空间限制，多目标优化中更高级的话题没有深度覆盖。这些话题可以参考相关文献，比如，处理约束，多峰值，非凸全局优化，组合优化，等。

Multiobjective Optimization is a very active field of research. There are still many open, challenging problems in the area. For future development of the research field it will be essential to provide EMO algorithms that are built around a robust notion of performance and, ideally, also can be analyzed more rigorously. Major topics for current research are also uncertainty handling and robustness, many-objective optimization, theoretical foundations and computational complexity, generalizations, for instance, level set approximation, diversity optimization, and set-oriented optimization, customization and integration into multidisciplinary workflows, and scalability to big data, or expensive evaluations.

多目标优化是一个非常活跃的研究领域。这个领域仍然有很多开放的有挑战性的问题。为了这个研究领域未来的发展，提出的EMO算法性能稳健，可以严格分析，这非常关键。当前研究的主要课题仍然是不确定性的处理和稳健性，众目标优化，理论基础和计算复杂度，泛化，比如，水平集的近似，多样性优化，和面向集合的优化，定制化和集成到多学科工作流，扩展到大数据，或昂贵的评估。