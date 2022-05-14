# Pymoo: Multi-Objective Optimization in Python

Julian Blank, Kalyanmoy Deb @ Michigan State University

## 0. Abstract

Python has become the programming language of choice for research and industry projects related to data science, machine learning, and deep learning. Since optimization is an inherent part of these research fields, more optimization related frameworks have arisen in the past few years. Only a few of them support optimization of multiple conflicting objectives at a time, but do not provide comprehensive tools for a complete multi-objective optimization task. To address this issue, we have developed pymoo, a multi-objective optimization framework in Python. We provide a guide to getting started with our framework by demonstrating the implementation of an exemplary constrained multi-objective optimization scenario. Moreover, we give a high-level overview of the architecture of pymoo to show its capabilities followed by an explanation of each module and its corresponding sub-modules. The implementations in our framework are customizable and algorithms can be modified/extended by supplying custom operators. Moreover, a variety of single, multi- and many-objective test problems are provided and gradients can be retrieved by automatic differentiation out of the box. Also, pymoo addresses practical needs, such as the parallelization of function evaluations, methods to visualize low and high-dimensional spaces, and tools for multi-criteria decision making. For more information about pymoo, readers are encouraged to visit: https://pymoo.org.

Python已经成为了研究和工业项目中，与数据科学，机器学习和深度学习相关的编程语言。由于优化是这些研究领域的一个内在部分，在过去这几年，出现了更多与优化相关的框架。只有其中一些支持多个冲突的目标的同时优化，但并没有为完整的多目标优化任务提供综合的工具。为处理这个问题，我们开发了PyMOO，一个Python的多目标优化框架。我们给出了指引，开始使用我们的框架，展示了约束多目标优化场景的实现的例子。而且，我们给出了PyMOO的高层概览，展示了其能力，解释了每个模块，及其对应的子模块。在我们框架中的实现是可定制的，提供了定制算子，可以修改/拓展算法。而且，给出了多个单目标，多目标，众目标测试问题，梯度可以由自动微分获得。同时，PyMOO处理了实际的需求，比如函数评估的并行化，可视化低维和高维空间的方法，以及多规则决策的工具。更多关于PyMOO的信息可以查看网站。

**Index Terms** Customization, genetic algorithm, multi-objective optimization, python

## 1. Introduction

Optimization plays an essential role in many scientific areas, such as engineering, data analytics, and deep learning. These fields are fast-growing and their concepts are employed for various purposes, for instance gaining insights from a large data sets or fitting accurate prediction models. Whenever an algorithm has to handle a significantly large amount of data, an efficient implementation in a suitable programming language is important. Python [1] has become the programming language of choice for the above mentioned research areas over the last few years, not only because it is easy to use but also good community support exists. Python is a high-level, cross-platform, and interpreted programming language that focuses on code readability. A large number of high-quality libraries are available and support for any kind of scientific computation is ensured. These characteristics make Python an appropriate tool for many research and industry projects where the investigations can be rather complex.

优化在很多科学领域都是关键的角色，比如工程，数据分析，和深度学习。这些领域正在快速增长，其概念被各种目的采用，比如从大型数据集中得到洞见，或拟合精确的预测模型。不论什么时候，一个算法要处理大量数据，用合适的编程语言进行高效的实现，是很重要的。Python是上述研究领域在过去几年的编程语言选择，不仅是因为容易使用，而且因为有很好的团体支持。Python是一个高层的，跨平台的，解释编程语言，聚焦在代码的可读性上。大量高质量的库可用，可以支持任何科学计算。这些性质使得Python成为了很多研究和工业项目的合适工具，其中的调查可能非常复杂。

A fundamental principle of research is to ensure reproducibility of studies and to provide access to materials used in the research, whenever possible. In computer science, this translates to a sketch of an algorithm and the implementation itself. However, the implementation of optimization algorithms can be challenging and specifically benchmarking is time-consuming. Having access to either a good collection of different source codes or a comprehensive library is time-saving and avoids an error-prone implementation from scratch.

一个基础的研究准则是，确保研究的可复现性，为研究中使用的材料提供访问。在计算机科学中，这就是算法的更改及其实现。但是，优化算法的实现可能是很有挑战性的，特别是基准测试是非常耗时的。如果可以访问很多不同的源代码，或综合的库，可以节省很多时间，避免从头开始的容易错误的实现。

To address this need for multi-objective optimization in Python, we introduce pymoo. The goal of our framework is not only to provide state of the art optimization algorithms, but also to cover different aspects related to the optimization process itself. We have implemented single, multi and many-objective test problems which can be used as a test-bed for algorithms. In addition to the objective and constraint values of test problems, gradient information can be retrieved through automatic differentiation [2]. Moreover, a parallelized evaluation of solutions can be implemented through vectorized computations, multi-threaded execution, and distributed computing. Further, pymoo provides implementations of performance indicators to measure the quality of results obtained by a multi-objective optimization algorithm. Tools for an explorative analysis through visualization of lower and higher-dimensional data are available and multi-criteria decision making methods guide the selection of a single solution from a solution set based on preferences.

为解决对Python多目标优化的需求，我们引入了pymoo。我们框架的目标不仅是给出目前最好的优化算法，而且还覆盖与优化过程相关的各个部分。我们实现了单目标，多目标和众目标测试问题，可以用于算法的测试。除了测试问题的目标和约束值，还可以通过自动微分来得到梯度信息。而且，通过向量化计算，多线程执行和分布式计算，可以对解进行并行评估。而且，pymoo给出了性能指示器的实现，测量多目标优化算法得到的结果的质量。还有工具可以通过低维和高维数据可视化进行探索性分析，多准则决策方法可以在解集合中，根据偏好指引选择单个解。

Our framework is designed to be extendable through of its modular implementation. For instance, a genetic algorithm is assembled in a plug-and-play manner by making use of specific sub-modules, such as initial sampling, mating selection, crossover, mutation and survival selection. Each sub-module takes care of an aspect independently and, therefore, variants of algorithms can be initiated by passing different combinations of sub-modules. This concept allows end-users to incorporate domain knowledge through custom implementations. For example, in an evolutionary algorithm a biased initial sampling module created with the knowledge of domain experts can guide the initial search.

我们的框架设计是模块化的实现，因此是可拓展的。比如，一个遗传算法是以即插即用的方式组合起来的，利用特定的子模块，比如初始采样，交配选择，交叉，变异和生存选择。每个子模块独立的负责一个方面，因此，可以通过传递不同的子模块组合，来得到算法的变体。这个概念使最终用户通过定制实现来纳入领域知识。比如，在一个演化算法中，一个有偏的初始采样模块，用领域专家知识创建，可以引导初始搜索。

Furthermore, we like to mention that our framework is well-documented with a large number of available code-snippets. We created a starter’s guide for users to become familiar with our framework and to demonstrate its capabilities. As an example, it shows the optimization results of a bi-objective optimization problem with two constraints. An extract from the guide will be presented in this paper. Moreover, we provide an explanation of each algorithm and source code to run it on a suitable optimization problem in our software documentation. Additionally, we show a definition of test problems and provide a plot of their fitness landscapes. The framework documentation is built using Sphinx [3] and correctness of modules is ensured by automatic unit testing [4]. Most algorithms have been developed in collaboration with the second author and have been benchmarked extensively against the original implementations.

而且，我们的框架文档很好，有大量可用的代码片段。我们创建了初次使用的引导，以熟悉我们的框架，展示其能力。例如，展示了带有两个约束的两目标优化问题的优化结果。本文会给出引导的摘录。而且，我们在软件文档中，给出了每个算法的解释，和在合适的优化问题上进行运行的源码。另外，我们展示了测试问题的定义，画出了其适应性图。框架文档用Sphinx构建，模块的正确性由自动单元测试确保。多数算法都是与第二作者合作开发的，和原始实现进行对比进行了广泛的基准测试。

In the remainder of this paper, we first present related existing optimization frameworks in Python and in other programming languages. Then, we provide a guide to getting started with pymoo in Section III which covers the most important steps of our proposed framework. In Section IV we illustrate the framework architecture and the corresponding modules, such as problems, algorithms and related analytics. Each of the modules is then discussed separately in Sections V to VII. Finally, concluding remarks are presented in Section VIII.

本文剩下部分，我们首先给出了相关的现有优化框架，语言为Python或其他编程语言。然后，我们在第3部分给出了开始使用pymoo的引导，这是我们提出的框架的最重要的步骤。在第4部分，我们描述了框架的架构，和对应的模块，比如问题，算法和相关的分析。在第5到第7部分，对每个模块进行了分别讨论。最后，在第8部分给出了结论。

## 2. Related Works

In the last decades, various optimization frameworks in diverse programming languages were developed. However, some of them only partially cover multi-objective optimization. In general, the choice of a suitable framework for an optimization task is a multi-objective problem itself. Moreover, some criteria are rather subjective, for instance, the usability and extendibility of a framework and, therefore, the assessment regarding criteria as well as the decision making process differ from user to user. For example, one might have decided on a programming language first, either because of personal preference or a project constraint, and then search for a suitable framework. One might give more importance to the overall features of a framework, for example parallelization or visualization, over the programming language itself. An overview of some existing multi-objective optimization frameworks in Python is listed in Table 1, each of which is described in the following.

在过去几十年中，用各种编程语言开发了各种优化框架。但是，其中只有一部分覆盖了多目标优化。总体上，对一个优化任务选择合适的框架，本身就是一个多目标问题。而且，一些准则是相当主观的，比如，框架的可用性和可拓展性，因此，关于准则以及决策过程的评估，不同的用户会有不同的标准。比如，一个人可能会首先决定编程语言，可能是因为个人偏好，或项目约束，然后搜索合适的框架。一个人可能会更看中框架的总体特征，比如并行性，或可视化，而对编程语言没有特别的偏好。表1给出了一些现有的多目标优化框架的概览，下面进行逐个描述。

Recently, the well-known multi-objective optimization framework jMetal [5] developed in Java [6] has been ported to a Python version, namely jMetalPy [7]. The authors aim to further extend it and to make use of the full feature set of Python, for instance, data analysis and data visualization. In addition to traditional optimization algorithms, jMetalPy also offers methods for dynamic optimization. Moreover, the post analysis of performance metrics of an experiment with several independent runs is automated.

最近，著名的java多目标优化框架jMetal移植到了Python版，即jMetalPy。作者的目标是进一步进行拓展，以充分使用Python的完整特征，比如，数据分析和数据可视化。除了传统的优化算法外，jMetalPy还给出动态优化的方法。而且，一个试验独立运行几次的性能度量的后分析进行了自动化。

Parallel Global Multiobjective Optimizer, PyGMO [8], is an optimization library for the easy distribution of massive optimization tasks over multiple CPUs. It uses the generalized island-model paradigm for the coarse grained parallelization of optimization algorithms and, therefore, allows users to develop asynchronous and distributed algorithms.

并行全局多目标优化器，即PyGMO，是一个优化库，可以很容易将大型优化任务部署到多个CPUs上。对优化算法的粗糙粒度的并行，使用一般化的岛屿模型范式，因此，用户可以开发异步和分布式算法。

Platypus [9] is a multi-objective optimization framework that offers implementations of state-of-the art algorithms. It enables users to create an experiment with various algorithms and provides post-analysis methods based on metrics and visualization.

Platypus是一个多目标优化框架，给出了目前最好算法的实现，使用户可以用多种算法进行试验，基于度量和可视化给出方法的后分析。

A Distributed Evolutionary Algorithms in Python (DEAP) [10] is novel evolutionary computation framework for rapid prototyping and testing of ideas. Even though, DEAP does not focus on multi-objective optimization, however, due to the modularity and extendibility of the framework multi-objective algorithms can be developed. Moreover, parallelization and load-balancing tasks are supported out of the box.

Python分布式演化算法(DEAP)是新型演化计算框架，可以快速成型并进行测试。虽然这样，DEAP并没有聚焦在多目标优化中，但是，因为框架的模块性和可拓展性，可以开发多目标优化算法。而且，并行化和负载均衡任务可以很容易实现。

Inspyred [11] is a framework for creating bio-inspired computational intelligence algorithms in Python which is not focused on multi-objective algorithms directly, but on evolutionary computation in general. However, an example for NSGA-II [12] is provided and other multi-objective algorithms can be implemented through the modular implementation of the framework.

Inspyred是创建生物启发的计算智能算法的框架，并没有直接聚焦在多目标算法，而是聚焦在演化计算上。但是，给出了NSGA-II算法的例子，其他多目标优化算法可以通过框架的模块化实现来进行实现。

If the search for frameworks is not limited to Python, other popular frameworks should be considered: PlatEMO [13] in Matlab, MOEA [14] and jMetal [5] in Java, jMetal-Cpp [15] and PaGMO [16] in C++. Of course this is not an exhaustive list and readers may search for other available options.

如果不局限在Python的实现上，其他流行的框架也应该考虑：MATLAB中的PlatEMO，Java的jMetal和MOEA，C++的jMetal-Cpp和PaGMO。当然这并不是完整的列表，读者可以搜索其他可用的框架。

## 3. Getting Started

In the following, we provide a starter’s guide for pymoo. It covers the most important steps in an optimization scenario starting with the installation of the framework, defining an optimization problem, and the optimization procedure itself.

下面，我们给出pymoo的使用引导，这覆盖了优化场景最重要的步骤，从安装框架开始，然后是定义优化问题，和优化过程本身。

### 3.1 Installation

Our framework pymoo is available on PyPI [17] which is a central repository to make Python software package easily accessible. The framework can be installed by using the package manager:

$ pip install -U pymoo

Some components are available in Python and additionally in Cython [18]. Cython allows developers to annotate existing Python code which is translated to C or C++ programming languages. The translated files are compiled to a binary executable and can be used to speed up computations. During the installation of pymoo, attempts are made for compilation, however, if unsuccessful due to the lack of a suitable compiler or other reasons, the pure Python version is installed. We would like to emphasize that the compilation is optional and all features are available without it. More detail about the compilation and troubleshooting can be found in our installation guide online.

### 3.2 Problem Definition

In general, multi-objective optimization has several objective functions with subject to inequality and equality constraints to optimize [19]. The goal is to find a set of solutions (variable vectors) that satisfy all constraints and are as good as possible regarding all its objectives values. The problem definition in its general form is given by:

$$min f_m(x), m=1, ..., M, s.t. g_j(x)≤0, j=1, ..., J, h_k(x)=0, k=1, ..., K, x_i^L ≤ x_i ≤ x_i^U, i=1,...,N$$(1)

The formulation above defines a multi-objective optimization problem with N variables, M objectives, J inequality, and K equality constraints. Moreover, for each variable x_i, lower and upper variable boundaries (x^L_i and x^U_i) are also defined.

In the following, we illustrate a bi-objective optimization problem with two constraints.

$$min f1(x)=x_1^2+x_2^2, max f2(x)=-(x_1-1)^2-x_2^2, s.t. g1(x)=2(x_1-0.1)(x_1-0.9)≤0, g2(x)=20(x_1-0.4)(x_1-0.6)≥0, -2≤x_1≤2, -2≤x_2≤2$$(2)

It consists of two objectives (M = 2) where f1(x) is minimized and f2(x) maximized. The optimization is with subject to two inequality constraints (J = 2) where g1(x) is formulated as a less-than-equal-to and g2(x) as a greater-than-equal-to constraint. The problem is defined with respect to two variables (N = 2), x1 and x2, which both are in the range [−2, 2]. The problem does not contain any equality constraints (K = 0). Contour plots of the objective functions are shown in Figure 1. The contours of the objective function f1(x) are represented by solid lines and f2(x) by dashed lines. Constraints g1(x) and g2(x) are parabolas which intersect the x1-axis at (0.1, 0.9) and (0.4, 0.6). The Pareto-optimal set is marked by a thick orange line. Through the combination of both constraints the Pareto-set is split into two parts. Analytically, the Pareto-optimal set is given by PS ={(x1, x2)|(0.1 ≤ x1 ≤ 0.4) ∨ (0.6 ≤ x1 ≤ 0.9) ∧ x2 = 0} and the efficient-front by f2 = (sqrt_f1 − 1)2 where f1 is defined in [0.01, 0.16] and [0.36, 0.81].

In the following, we provide an example implementation of the problem formulation above using pymoo. We assume the reader is familiar with Python and has a fundamental knowledge of NumPy [20] which is utilized to deal with vector and matrix computations.

In pymoo, we consider pure minimization problems for optimization in all our modules. However, without loss of generality an objective which is supposed to be maximized, can be multiplied by −1 and be minimized [19]. Therefore, we minimize −f2(x) instead of maximizing f2(x) in our optimization problem. Furthermore, all constraint functions need to be formulated as a less-than-equal-to constraint. For this reason, g2(x) needs to be multiplied by −1 to flip the ≥ to a ≤ relation. We recommend the normalization of constraints to give equal importance to each of them. For g1(x), the constant ‘resource’ value of the constraint is 2 · (−0.1) · (−0.9) = 0.18 and for g2(x) it is 20 · (−0.4) · (−0.6) = 4.8, respectively. We achieve normalization of constraints by dividing g1(x) and g2(x) by the corresponding constant [21].

Finally, the optimization problem to be optimized using pymoo is defined by:

$$min f1(x)=x_1^2+x_2^2, min f2(x)=(x_1-1)^2+x_2^2, s.t. g1(x)=2(x_1-0.1)(x_1-0.9)/0.18≤0, g2(x)=-20(x_1-0.4)(x_1-0.6)/4.8≤0, -2≤x_1≤2, -2≤x_2≤2$$(3)

Next, the derived problem formulation is implemented in Python. Each optimization problem in pymoo has to inherit from the Problem class. First, by calling the super() function the problem properties such as the number of variables (n_var), objectives (n_obj) and constraints (n_constr) are initialized. Furthermore, lower (xl) and upper variables boundaries (xu) are supplied as a NumPy array. Additionally, the evaluation function _evaluate needs to be overwritten from the superclass. The method takes a two-dimensional NumPy array x with n rows and m columns as an input. Each row represents an individual and each column an optimization variable. After doing the necessary calculations, the objective values are added to the dictionary out with the key F and the constraints with key G.

```
import autograd.numpy as anp
from pymoo.model.problem import Problem

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=anp.array([-2,2]), xu=anp.array([-2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:,0]**2 + x[:,1]**2
        f2 = (x[:,0]-1)**2 + x[:,1]**2

        g1 = 2*(x[:,0] - 0.1) * (x[:,0] - 0.9) / 0.18
        g2 = -20*(x[:,0] - 0.4) * (x[:,0] - 0.6) / 4.8

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2])
```

As mentioned above, pymoo utilizes NumPy [20] for most of its computations. To be able to retrieve gradients through automatic differentiation we are using a wrapper around NumPy called Autograd [22]. Note that this is not obligatory for a problem definition.

### 3.3 Algorithm Initialization

Next, we need to initialize a method to optimize the problem. In pymoo, an algorithm object needs to be created for optimization. For each of the algorithms an API documentation is available and through supplying different parameters, algorithms can be customized in a plug-and-play manner. In general, the choice of a suitable algorithm for optimization problems is a challenge itself. Whenever problem characteristics are known beforehand we recommended using those through customized operators. However, in our case the optimization problem is rather simple, but the aspect of having two objectives and two constraints should be considered. For this reason, we decided to use NSGA-II [12] with its default configuration with minor modifications. We chose a population size of 40, but instead of generating the same number of offsprings, we create only 10 in each generation. This is a steady-state variant of NSGA-II and it is likely to improve the convergence property for rather simple optimization problems without much difficulties, such as the existence of local Pareto-fronts. Moreover, we enable a duplicate check which makes sure that the mating produces offsprings which are different with respect to themselves and also from the existing population regarding their variable vectors. To illustrate the customization aspect, we listed the other unmodified default operators in the code-snippet below. The constructor of NSGA2 is called with the supplied parameters and returns an initialized algorithm object.

```
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

algorithm = NSGA2(pop_size=40, n_offsprings=10, sampling=get_sampling("real_random"), crossover=get_crossover("real_sbx", prob=0.9, eta=15), mutation=get_mutation("real_pm", eta=20), eliminate_duplicate=True)
```

### 3.4 Optimization

Next, we use the initialized algorithm object to optimize the defined problem. Therefore, the minimize function with both instances problem and algorithm as parameters is called. Moreover, we supply the termination criterion of running the algorithm for 40 generations which will result in 40 + 40 × 10 = 440 function evaluations. In addition, we define a random seed to ensure reproducibility and enable the verbose flag to see printouts for each generation.

```
from pymoo.optimize import minimize
res = minimize(MyProblem(), algorithm, ('n_gen', 40), seed=1, verbose=True)
```

The method returns a Result object which contains the non-dominated set of solutions found by the algorithm.

The optimization results are illustrated in Figure 2 where the design space is shown in Figure 2a and in the objective space in Figure 2b. The solid line represents the analytically derived Pareto set and front in the corresponding space and the circles solutions found by the algorithm. It can be observed that the algorithm was able to converge and a set of nearly-optimal solutions was obtained. Some additional post-processing steps and more details about other aspects of the optimization procedure can be found in the remainder of this paper and in our software documentation.

The starters guide showed the steps starting from the installation up to solving an optimization problem. The investigation of a constrained bi-objective problem demonstrated the basic procedure in an optimization scenario.

## 4. Architechure

Software architecture is fundamentally important to keep source code organized. On the one hand, it helps developers and users to get an overview of existing classes, and on the other hand, it allows flexibility and extendibility by adding new modules. Figure 3 visualizes the architecture of pymoo. The first level of abstraction consists of the optimization problems, algorithms and analytics. Each of the modules can be categorized into more detail and consists of multiple submodules.

软件架构是非常重要的，可以使源代码有组织。一方面，可以帮助开发者和用户概览现有的类，另一方面，使软件有灵活性和可拓展性，可以加入新的模块。图3是pymoo的架构图。第一级抽象包含优化问题，算法和分析。每个模块包含多个子模块。

(i) Problems: Optimization problems in our framework are categorized into single, multi, and many-objective test problems. Gradients are available through automatic differentiation and parallelization can be implemented by using a variety of techniques.

问题：框架中的优化问题分类为单目标，多目标和众目标测试问题。通过自动微分可以得到梯度，并行化可以通过多种技术得到实现。

(ii) Optimization: Since most of the algorithms are based on evolutionary computations, operators such as sampling, mating selection, crossover and mutation have to be chosen or implemented. Furthermore, because many problems in practice have one or more constraints, a methodology for handling those must be incorporated. Some algorithms are based on decomposition which splits the multi-objective problem into many single-objective problems. Moreover, when the algorithm is used to solve the problem, a termination criterion must be defined either explicitly or implicitly by the implementation of the algorithm.

优化：由于多数算法是基于演化计算的，所以必须选择或实现采样，配偶选择，交叉和变异。而且，由于很多实践中的问题有一个或多个约束，需要有一种方法将这些约束纳入进来。一些算法是基于decomposition的，将多目标问题分解成多个单目标问题。而且，当算法用于求解问题，必须定义一个停止条件，算法需要显式的或隐式的实现。

(iii) Analytics: During and after an optimization run analytics support the understanding of data. First, intuitively the design space, objective space, or other metrics can be explored through visualization. Moreover, to measure the convergence and/or diversity of a Pareto-optimal set performance indicators can be used. For real-parameter problems, recently proposed theoretical KKT proximity metric [23] computation procedure is included in pymoo to compute the proximity of a solution to the true Pareto-optimal front, despite not knowing its exact location. To support the decision making process either through finding points close to the area of interest in the objective space or high trade-off solutions. This can be applied either during an optimization run to mimic interactive optimization or as a post analysis.

分析：优化的过程中或优化后，需要有分析来支持对数据的理解。首先，直觉上设计空间，目标空间或其他度量，需要通过可视化进行探索。而且，为度量收敛性和/或Pareto最优性集合的多样性，需要用到性能指示器。对于实值参数问题，最近提出的理论KKT接近性度量[23]的计算过程在pymoo中实现了，以计算一个解到真实的Pareto最优性前沿的接近性，尽管不知道其精确的位置。为支持决策过程，要么是通过找到接近目标空间中的感兴趣区域的点，要么是高折中解。这个过程的应用，可以是在优化过程中，以模仿交互式优化，或可以是一个后期分析。

In the remainder of the paper, we will discuss each of the modules mentioned in more detail.

在本文剩余部分，我们会更加详细的讨论提到的模块。

## 5. Problems

It is common practice for researchers to evaluate the performance of algorithms on a variety of test problems. Since we know no single-best algorithm for all arbitrary optimization problems exist [24], this helps to identify problem classes where the algorithm is suitable. Therefore, a collection of test problems with different numbers of variables, objectives or constraints and alternating complexity becomes handy for algorithm development. Moreover, in a multi-objective context, test problems with different Pareto-front shapes or varying variable density close to the optimal region are of interest.

研究者在多个测试问题上评估算法性能，这是常见操作。我们知道，没有哪个算法对所有优化问题是最好的，要识别问题的类别，确定合适的算法。因此，对算法开发来说，有不同数量的变量、目标函数或约束的测试问题集，复杂度也不同，这样会很方便。而且，在多目标的上下文中，有不同的Pareto-front的形状，或不同的变量密度测试问题，是很有趣的。

### 5.1 Implementations

In our framework, we categorize test problems regarding the number of objectives: single-objective (1 objective), multi-objective (2 or 3 objectives) and many-objective (more than 3 objectives). Test problems implemented in pymoo are listed in Table 2. For each problem the number of variables, objectives, and constraints are indicated. If the test problem is scalable to any of the parameters, we label the problem with (s). If the problem is scalable, but a default number was original proposed we indicate that with surrounding brackets. In case the category does not apply, for example because we refer to a test problem family with several functions, we use (·).

在我们的框架中，我们将测试问题按照目标数量进行分类：单目标（1个目标），多目标（2或3个目标），和众目标（多余3个目标）。表2给出了在pymoo中实现的测试问题。对每个问题，给出了变量，目标和约束的数量。如果测试问题可扩展到任意参数数量，我们将问题标记为(s)。如果问题是可扩展的，但是已经提出了一个默认的数量，我们就将其用括号进行指示。在类别不能应用的情况下，比如一个测试问题族有几个函数，我们使用(·)。

The implementations in pymoo let end-users define what values of the corresponding problem should be returned. On an implementation level, the evaluate function of a Problem instance takes a list return_value_of which contains the type of values being returned. By default the objective values “F” and if the problem has constraints the constraint violation “CV” are included. The constraint function values can be returned independently by adding “G”. This gives developers the flexibility to receive the values that are needed for their methods.

pymoo中的实现，使最终用户可以定义对应的问题应当返回什么值。在实现的层次，一个Problem实例的evaluate函数接收return_value_of的列表，包含返回的值的类型。默认情况下，会包含目标值"F"，如果问题有约束，会包含约束违反"CV"。约束函数值可以独立返回，只需要加入"G"。这使开发者会很有灵活性，可以接收其方法需要的值。

### 5.2 Gradients

All our test problems are implemented using Autograd [22]. Therefore, automatic differentiation is supported out of the box. We have shown in Section III how a new optimization problem is defined.

所有测试问题都使用Autograd进行实现。因此，自动微分是支持的。我们在第3部分展示了，怎样定义一个新的优化问题。

If gradients are desired to be calculated the prefix “d” needs to be added to the corresponding value of the return_value_of list. For instance to ask for the objective values and its gradients return_value_of = [“F”, “dF”].

如果需要计算梯度，需要在return_value_of列表中加入对应值的"d"前缀。比如，目标值及其梯度，return_value_of = ["F", "dF"]。

Let us consider the problem we have implemented shown in Equation 3. The derivation of the objective functions F with respect to each variable is given by:

我们考虑式3中实现的问题。目标函数F对每个变量的导数由下式给出：

$$∇F = [2x_1, 2x_2; 2(x_1 - 1), 2x_2]$$(4)

The gradients at the point [0.1, 0.2] are calculated by: 在点[0.1, 0.2]处的梯度由下式计算

```
F, dF = problem.evaluate(np.array([0.1, 0.2]), return_values_of = ["F", "dF"])
```

returns the following output 返回下面的输出

```
F <- [0.05, 0.85]
dF <- [[0.2, 0.4], [-1.8,0.4]]
```

It can easily be verified that the values are matching with the analytic gradient derivation. The gradients for the constraint functions can be calculated accordingly by adding “dG” to the return_value_of list.

可以很容易验证，这些值与分析的梯度偏差是符合的。约束函数的梯度可以相应的计算，只需要将"dG"加入到return_value_of列表中。

### 5.3 Parallelization

If evaluation functions are computationally expensive, a serialized evaluation of a set of solutions can become the bottleneck of the overall optimization procedure. For this reason, parallelization is desired for an use of existing computational resources more efficiently and distribute long-running calculations. In pymoo, the evaluation function receives a set of solutions if the algorithm is utilizing a population. This empowers the user to implement any kind of parallelization as long as the objective values for all solutions are written as an output when the evaluation function terminates. In our framework, a couple of possibilities to implement parallelization exist:

如果评估函数的计算量很大，整个优化过程的瓶颈可能就是一系列解的序列化评估。因此，并行化就很必要了，更高效的使用现有计算资源。在pymoo中，如果算法利用一个族群，评估函数收到解的集合。这使用户可以实现任意并行，只要在评估函数停止时，所有解的目标值都写为输出。在我们的框架中，实现并行有几种可能方法：

(i) Vectorized Evaluation: A common technique to parallelize evaluations is to use matrices where each row represents a solution. Therefore, a vectorized evaluation refers to a column which includes the variables of all solutions. By using vectors the objective values of all solutions are calculated at once. The code-snippet of the example problem in Section III shows such an implementation using NumPy [20]. To run calculations on a GPU, implementing support for PyTorch [25] tensors can be done with little overhead given suitable hardware and correctly installed drivers.

向量化评估：并行化评估的一种常见技术是，使用矩阵，其中每一行表示一个解。因此，一个向量化的评估指的是一列，包含了所有解的变量。使用向量，所有解的目标值都一次性进行计算。第3部分中的问题例子的代码片段，展示了使用NumPy的一个实现。为在GPU上运行计算，实现PyTorch张量的支持的代价是很小的，只要有合适的硬件和正确安装的驱动。

(ii) Threaded Loop-wise Evaluation: If the function evaluation should occur independently, a for loop can be used to set the values. By default the evaluation is serialized and no calculations occur in parallel. By providing a keyword to the evaluation function, pymoo spawns a thread for each evaluation and manages those by using the default thread pool implementation in Python. This behaviour can be implemented out of the box and the number of parallel threads can be modified.

线程逐个循环的评估：如果函数评估要独立的进行，可以用for循环来设置值。默认情况下，评估要顺序进行，没有计算要并行进行。给评估函数一个关键字，pymoo对每个评估产生一个线程，使用Python中默认的线程池来管理。这种行为很容易实现，并行线程数可以进行修改。

(iii) Distributed Evaluation: If the evaluation should not be limited to a single machine, the evaluation itself can be distributed to several workers or a whole cluster. We recommend using Dask [26] which enables distributed computations on different levels. For instance, the matrix operation itself can be distributed or a whole function can be outsourced. Similar to the loop wise evaluation each individual can be evaluate element-wise by sending it to a worker.

分布式评估：如果评估不局限在一台机器上，评估本身可以分布到几台机器或整个集群中。我们推荐使用Dask，可以在不同的级别上实现分布式计算。比如，矩阵运算本身可以分布计算，或整个函数可以外包。与逐个循环评估类似，每个评估可以送到一个worker上进行逐个元素的评估。

## 6. Optimization Module

The optimization module provides different kinds of sub-modules to be used in algorithms. Some of them are more of a generic nature, such as decomposition and termination criterion, and others are more related to evolutionary computing. By assembling those modules together algorithms are built.

优化模块提供了用在算法中的不同子模块。一些更多的具有通用的特征，比如decomposition和终止准则，其他的更多的与演化计算相关。通过将这些模块组装起来，就可以组成算法。

### 6.1 Algorithms

Available algorithm implementations in pymoo are listed in Table 3. Compared to other optimization frameworks the list of algorithms may look rather short, however, each algorithm is customizable and variants can be initialized with different parameters. For instance, a Steady-State NSGA-II [27] can be initialized by setting the number of offspring to 1. This can be achieved by supplying this as a parameter in the initialization method as shown in Section III. Moreover, it is worth mentioning that many-objective algorithms, such as NSGA-III or MOEAD, require reference directions to be provided. The reference directions are commonly desired to be uniform or to have a bias towards a region of interest. Our framework offers an implementation of the Das and Dennis method [28] for a fixed number of points (fixed with respect to a parameter often referred to as partition number) and a recently proposed Riesz-Energy based method which creates a well-spaced point set for an arbitrary number of points and is capable of introducing a bias towards preferred regions in the objective space [29].

表3给出了pymoo中可用的算法实现。与其他优化框架相比，算法列表可能看起来比较短，但是，每个算法都是可定制的，变体可以用不同的参数初始化。比如，稳态NSGA-II[27]可以通过设置子代数量为1进行初始化。通过在初始化方法中设置一个参数，就可以得到这样的效果，如第3部分所示。而且，值得提到的是，众目标算法，如NSGA-III或MOEAD，需要提供参考方向。参考方向通常要均匀分布，或会偏向感兴趣区域。我们的框架给出了Das and Dennis方法[28]的实现，点数固定（相对于一个参数固定，通常称之为分割数），和一个最近提出的基于Riesz-Energy的方法，对任意数量的点创建了well-spaced点集，可以引入偏向于目标空间中的倾向预期的方向。

### 6.2 Operators

The following evolutionary operators are available: 下面的演化算子是可用的：

(i) Sampling: The initial population is mostly based on sampling. In some cases it is created through domain knowledge and/or some solutions are already evaluated, they can directly be used as an initial population. Otherwise, it can be sampled randomly for real, integer, or binary variables. Additionally, Latin-Hypercube Sampling [41] can be used for real variables.

采样：初始种群基本是基于采样的。在一些情况中，是通过领域知识创建的，和/或通过已经评估的解创建的，它们可以直接用作初始种群。否则，对浮点，整数，或二值变量，可以进行随机采样。另外，Latin-Hypercube采样可以用在浮点变量上。

(ii) Crossover: A variety of crossover operators for different type of variables are implemented. In Figure 4 some of them are presented. Figures 4a to 4d help to visualize the information exchange in a crossover with two parents being involved. Each row represents an offspring and each column a variable. The corresponding boxes indicate whether the values of the offspring are inherited from the first or from the second parent. For one and two-point crossovers it can be observed that either one or two cuts in the variable sequence exist. Contrarily, the Uniform Crossover (UX) does not have any clear pattern, because each variable is chosen randomly either from the first or from the second parent. For the Half Uniform Crossover (HUX) half of the variables, which are different, are exchanged. For the purpose of illustration, we have created two parents that have different values in 10 different positions. For real variables, Simulated Binary Crossover [42] is known to be an efficient crossover. It mimics the crossover of binary encoded variables. In Figure 4e the probability distribution when the parents x1 = 0.2 and x2 = 0.8 where xi ∈ [0, 1] with η = 0.8 are recombined is shown. Analogously, in case of integer variables we subtract 0.5 from the lower and add (0.5 − ε) to the upper bound before applying the crossover and round to the nearest integer afterwards (see Figure 4f).

交叉：对不同类型的变量的多种交叉算子进行了实现。图4给出了一些例子。图4a到4d展示了在2个父辈的情况下，交叉的信息交换。每一行代表一个子代，每一列代表一个变量。对应的框表示，子代的值是从第一个父辈那继承得到的，还是从第二个父辈那继承得到的。对于一点和两点交叉，可以观察到，在变量序列中，存在一个或两个buts。对比起来，在均匀交叉(UX)中，不存在清晰的模式，因为每个变量是从第一个或第二个父辈中随机选择的。对于半均匀交叉(HUX)，交换了一半的不同的变量。为进行描述，我们创建了两个父辈，在10个位置有不同的值。对于实值变量，仿真二进制交叉(SBX)是一种有效的交叉，模仿的是二进制编码的变量的交叉。在图4e中，展示了当父辈x1=0.2，x2=0.8，其中xi ∈ [0, 1]，η = 0.8时重新结合时的概率分布。类似的，在整型变量的情况中，我们从下限中减去0.5，在上限上加上(0.5 − ε)，然后应用交叉，最后四舍五入到最近的整数（图4f）。

(iii) Mutation: For real and integer variables Polynomial Mutation [19], [43] and for binary variables Bitflip mutation [44] is provided.

变异：对于实数和整型变量，实现了多项式变异，对于二进制变量，实现了Bitflip变异。

Different problems require different type of operators. In practice, if a problem is supposed to be solved repeatedly and routinely, it makes sense to customize the evolutionary operators to improve the convergence of the algorithm. Moreover, for custom variable types, for instance trees or mixed variables [45], custom operators [46] can be implemented easily and called by algorithm class. Our software documentation contains examples for custom modules, operators and variable types.

不同的问题需要不同类型的算子。在实践中，如果一个问题要重复按惯例进行求解，就可以定制演化算子，以改进算法的收敛性。而且，对于定制的变量类型，比如树或混合变量[45]，定制算子可以很容易的实现，由算法类调用。我们的软件文档包含定制模块，算子和变量类型的例子。

### 6.3 Termination Criterion

For every algorithm it must be determined when it should terminate a run. This can be simply based on a predefined number of function evaluations, iterations, or a more advanced criterion, such as the change of a performance metric over time. For example, we have implemented a termination criterion based on the variable and objective space difference between generations. To make the termination criterion more robust the last k generations are considered. The largest movement from a solution to its closest neighbour is tracked across generation and whenever it is below a certain threshold, the algorithm is considered to have converged. Analogously, the movement in the objective space can also be used. In the objective space, however, normalization is more challenging and has to be addressed carefully. The default termination criterion for multi-objective problems in pymoo keeps track of the boundary points in the objective space and uses them, when they have settled down, for normalization. More details about the proposed termination criterion can be found in [47].

对每个算法，必须要确定什么时候结束运行。这可以是预先定义的函数评估的数量，迭代次数，或更高级的准则，比如随着时间性能度量的变化。比如，我们实现了一个基于变量和目标空间的代差的停止规则。为使停止规则更加文件，考虑最后k代。从一个解到其最接近的邻域解的最大运动，在代与代之间进行追踪，当其小于特定阈值时，算法就可以认为是收敛了。类似的，在目标空间中的运动也可以使用。但是，在目标空间中，归一化更有挑战性，必须要小心处理。Pymoo中对多目标问题，默认的停止规则追踪的是目标空间中的边缘点，当它们稳定下来后就可以停止。关于提出的停止规则，更多的细节可以参见[47]。

### 6.4 Decomposition

Decomposition transforms multi-objective problems into many single-objective optimization problems [48]. Such a technique can be either embedded in a multi-objective algorithm and solved simultaneously or independently using a single-objective optimizer. Some decomposition methods are based on the lp-metrics with different p values. For instance, a naive but frequently used decomposition approach is the Weighted-Sum Method (p = 1), which is known to be not able to converge to the non-convex part of a Pareto-front [19]. Moreover, instead of summing values, Tchebysheff Method (p = ∞) considers only the maximum value of the difference between the ideal point and a solution. Similarly, the Achievement Scalarization Function (ASF) [49] and a modified version Augmented Achievement Scalarization Function (AASF) [50] use the maximum of all differences. Furthermore, Penalty Boundary Intersection (PBI) [40] is calculated by a weighted sum of the norm of the projection of a point onto the reference direction and the perpendicular distance. Also it is worth to note that normalization is essential for any kind of decomposition. All decomposition techniques mentioned above are implemented in pymoo.

Decomposition将多目标问题转换成很多单目标优化问题。这样一种技术可以嵌入到多目标算法中进行同时求解，或使用单目标优化器进行独立求解。一些decomposition方法是基于lp度量的，有不同的p个值。比如，一种朴素但是频繁使用的decomposition方法，是加权和方法(p=1)，不能收敛到Pareto front的非凸部分。而且，Tchebysheff方法(p = ∞)只考虑理想点和一个解之间差异的最大值。类似的，ASF[49]和修正版AASF[50]使用所有差值的最大值。而且，PBI[40]的计算，是一个点到一个参考方向的投影的范数，和其垂直距离的加权和。还值得提出的是，归一化对于任何decomposition都是必须的。所有上面提到的decomposition技术都在pymoo中进行了实现。

## 7. Analytics

### 7.1 Performance Indicators

For single-objective optimization algorithms the comparison regarding performance is rather simple because each optimization run results in a single best solution. In multi-objective optimization, however, each run returns a non-dominated set of solutions. To compare sets of solutions, various performance indicators have been proposed in the past [51]. In pymoo most commonly used performance indicators are described:

对于单目标优化算法，性能的比较是很简单的，因为每次优化运行都得到一个最佳解。但在多目标优化中，每次运行返回一个非占优解集。为比较解集，过去提出了各种性能指示器[51]。在pymoo中常用的性能指示器如下描述。

(i) GD/IGD: Given the Pareto-front PF the deviation between the non-dominated set S found by the algorithm and the optimum can be measured. Following this principle, Generational Distance (GD) indicator [52] calculates the average Euclidean distance in the objective space from each solution in S to the closest solution in PF. This measures the convergence of S, but does not indicate whether a good diversity on the Pareto-front has been reached. Similarly, Inverted Generational Distance (IGD) indicator [52] measures the average Euclidean distance in the objective space from each solution in PF to the closest solution in S. The Pareto-front as a whole needs to be covered by solutions from S to minimize the performance metric. Thus, lower the GD and IGD values, the better is the set. However, IGD is known to be not Pareto compliant [53].

GD/IGD：给定Pareto front PF，最优解和算法找到的非占优解集S之间的偏差，可以进行度量。按照这个原则，代际距离(GD)指示器计算的是，目标空间中，从S中的每个解，到PF中的最接近的解的平均欧式距离。这个度量的是S的收敛性，但并没有指示是否达到了Pareto-front上的好的多样性。类似的，IGD指示器度量的是，目标空间中，PF中的每个解，到S中的最接近的解的平均欧式距离。Pareto-front作为一个整体，需要被S中的解所覆盖，以最小化性能度量。因此，GD和IGD值越小，这个解集越好。但是，IGD并不是Pareto兼容的。

(ii) GD+/IGD+: A variation of GD and IGD has been proposed in [53]. The Euclidean distance is replaced by a distance measure that takes the dominance relation into account. The authors show that IGD+ is weakly Pareto compliant.

GD+/IGD+：[53]提出了一种GD和IGD的变化。欧式距离替换为另一种距离度量，将支配关系纳入了考虑。作者表示，IGD+是弱Pareto兼容的。

(iii) Hypervolume: Moreover, the dominated portion of the objective space can be used to measure the quality of non-dominated solutions [54]. The higher the hypervolume, the better is the set. Instead of the Pareto-front a reference point needs to be provided. It has been shown that Hypervolume is Pareto compliant [55]. Because the performance metric becomes computationally expensive in higher dimensional spaces the exact measure becomes intractable. However, we plan to include some proposed approximation methods in the near future.

Hypervolume：目标空间中被占优的部分，可以用于衡量非占优解的质量。Hypervolume值越大，集合质量越高。除了Pareto front，还需要指定一个参考点。已经证明了Hypervolume是Pareto兼容的。因为性能度量在高维空间中计算量非常大，精确的度量变得很难处理。但是，我们计划在未来包含一些提出的近似方法。

Performance indicators are used to compare existing algorithms. Moreover, the development of new algorithms can be driven by the goodness of different metrics itself.

性能指示器用于比较已经存在的算法。而且，新算法的开发会受到不同度量的优劣的驱动。

### 7.2 Visualization

The visualization of intermediate steps or the final result is inevitable. In multi and many-objective optimization, visualization of the objective space is of interest so that trade-off information among solutions can be easily experienced from the plots. Depending on the dimension of the objective space, different types of plots are suitable to represent a single or a set of solutions. In pymoo the implemented visualizations wrap around the well-known plotting library in Python Matplotlib [56]. Keyword arguments provided by Matplotlib itself are still available which allows to modify for instance the color, thickness, opacity of lines, points or other shapes. Therefore, all visualization techniques are customizable and extendable.

对中间步骤或最终结果的可视化是不可避免的。在多目标和众目标优化中，目标空间的可视化是有兴趣的，这样解与解之间的折中信息可以很容易的从图中看出。依赖于目标空间的维度，有不同类型的图适合于表示单个解或多个解。在pymoo中，实现的可视化包含了Python中著名的画图库Matplotlib。Matplotlib提供的关键字参数仍然是可用的，可以修改点，线段或其他形状的色彩，厚度，和透明度。因此，所有可视化技术是可定制的和可拓展的。

For 2 or 3 objectives, scatter plots (see Figure 5a and 5b) can give a good intuition about the solution set. Trade-offs can be observed by considering the distance between two points. It might be desired to normalize each objective to make sure a comparison between values is based on relative and not absolute values. Pairwise Scatter Plots (see Figure 5c) visualize more than 3 objectives by showing each pair of axes independently. The diagonal is used to label the corresponding objectives.

对于2个或3个目标，散点图（图5a，5b）可以给出解集的很好的直觉。通过两个点之间的距离可以观察到解的折中关系。每个目标要进行归一化，以确保值之间的比较是基于相对值，而不是绝对值的。成对散点图（图5c）可以对超过3个目标进行可视化，独立的展示每对轴。对角线用于标记对应的目标。

Also, high-dimensional data can be illustrated by Parallel Coordinate Plots (PCP) as shown in Figure 5d. All axes are plotted vertically and represent an objective. Each solution is illustrated by a line from the left to the right. The intersection of a line and an axis indicate the value of the solution regarding the corresponding objective. For the purpose of comparison solution(s) can be highlighted by varying color and opacity.

高维数据还可以用并行坐标图(PCP，图5d)进行展示。所有轴都垂直画出，表示一个目标。每个解表示为从左到右的一条线。一条线和一个轴的相交，表示解在对应目标上的值。要比较不同解，可以用不同的色彩和透明度进行强调。

Moreover, a common practice is to project the higher dimensional objective values onto the 2D plane using a transformation function. Radviz (Figure 5e) visualizes all points in a circle and the objective axes are uniformly positioned around on the perimeter. Considering a minimization problem and a set of non-dominated solutions, an extreme point very close to an axis represents the worst solution for that corresponding objective, but is comparably ‘‘good’’ in one or many other objectives. Similarly, Star Coordinate Plots (Figure 5f) illustrate the objective space, except that the transformation function allows solutions outside of the circle.

用一个变换函数，将高维目标值投影到2D平面上，这是很常见的。Radviz（图5e）将所有点在一个圆中进行可视化，目标轴沿着周长均匀分布。考虑一个最小化问题，和一个非占优解集，非常接近一个轴的一个极限点，表示对这个目标来说最坏，但对于一个或多个其他目标相对较好的解。类似的，星坐标图（图5f）描述了目标空间，但变换函数允许解在圆之外。

Heatmaps (Figure 5g) are used to represent the goodness of solutions through colors. Each row represents a solution and each column a variable. We leave the choice to the end-user of what color map to use and whether light or dark colors illustrate better or worse solutions. Also, solutions can be sorted lexicographically by their corresponding objective values.

热力图（图5g）用于通过颜色来表示解的好坏。每一行表示一个解，每个列表示一个变量。用什么样的色图，浓色或淡色表示解的好坏，这个可以由用户来决定。解可以按照对应目标值的词典顺序进行排列。

Instead of visualizing a set of solutions, one solution can be illustrated at a time. The Petal Diagram (Figure 5h) is a pie diagram where the objective value is represented by each piece’s diameter. Colors are used to further distinguish the pieces. Finally, the Spider-Web or Radar Diagram (Figure 5i) shows the objectives values as a point on an axis. The ideal and nadir point [19] is represented by the inner and outer polygon. By definition, the solution lies in between those two extremes. If the objective space ranges are scaled differently, normalization for the purpose of plotting can be enabled and the diagram becomes symmetric. New and emerging methods for visualizing more than three-dimensional efficient solutions, such as 2.5-dimensional PaletteViz plots [57], would be implemented in the future.

还可以不显示解集，而一次只显示一个解。花瓣图（图5h）中，每个目标值都用每一块的半径来表示。色彩也用来区分每一块。最后，蜘蛛网或雷达图（图5i）将目标值展示为轴上的一个点。最理想的和最差的点由内多边形和外多边形表示。所有解都在这两个极值之间。如果目标空间的缩放不同，可以采用归一化，以使图变得对称。超过3维解的新出现可视化方法，比如2.5d的PaletteViz图，会在将来进行实现。

### 7.3 Desicion Making

In practice, after obtaining a set of non-dominated solutions a single solution has to be chosen for implementation. pymoo provides a few ‘a posteriori’ approaches for decision making [19].

实践中，在得到非占优解集后，需要选择一个解来进行实现。pymoo提供了几个后验的方法进行决策。

(i) Compromise Programming: One way of making a decision is to compute value of a scalarized and aggregated function and select one solution based on minimum or maximum value of the function. In pymoo a number of scalarization functions described in Section VI-D can be used to come to a decision regarding desired weights of objectives.

折中规划：一种决策的方法是，计算一个标量化和聚集函数的值，基于函数值的最大值或最小值来选择一个解。在pymoo中，6.4节描述的几种标量化函数可以用于对理想的目标权重进行决策。

(ii) Pseudo-Weights: However, a more intuitive way to chose a solution out of a Pareto-front is the pseudo-weight vector approach proposed in [19]. The pseudo weight wi for the i-th objective function is calculated by:

伪权重：从Pareto front中选择一个解的一个更直观的方法，是[19]中提出的伪权重向量方法。第i个目标函数的伪权重wi计算如下：

$$wi = \frac {(f_i^{max} - f_i(x)) / (f_i^{max} - f_i^{min})} {\sum_{m=1}^M (f_m^{max} - f_m(x)) / (f_m^{max} - f_m^{min})} $$(5)

The normalized distance to the worst solution regarding each objective i is calculated. It is interesting to note that for non-convex Pareto-fronts, the pseudo weight does not correspond to the result of an optimization using the weighted-sum method. A solution having the closest pseudo-weight to a target preference vector of objectives (f1 being preferred twice as important as f2 results in a target preference vector of (0.667, 0.333)) can be chosen as the preferred solution from the efficient set.

计算每个目标i到其最坏解的归一化距离。要提到，对于非凸Pareto fronts，伪权重并不对应着使用加权和方法优化的结果。到目标偏好目标向量最近伪权重的解（在偏好向量(0.667,0.333)中，f1的重要性比f2大2倍），可以从解集中选为偏好解。

(iii) High Trade-Off Solutions: Furthermore, high trade-off solutions are usually of interest, but not straightforward to detect in higher-dimensional objective spaces. We have implemented the procedure proposed in [65]. It was described to be embedded in an algorithm to guide the search; we, however, use it for post-processing. The metric for each solution pair xi and xj in a non-dominated set is given by:

高折中解：在高维目标空间中，高折中解通常是感兴趣的，但不会那么直接检测到。我们实现了[65]中提出的过程，这是一个嵌入到算法中引导搜索的过程；但是，我们将其用在后处理中。对非占优解集中，每个解对xi和xj，其度量为

$$T(xi, xj) = \frac {\sum_{i=1}^M max[0, fm(xj) − fm(xi)]} {\sum_{i=1}^M max[0, fm(xi) − fm(xj)]}$$(6)

where the numerator represents the aggregated sacrifice and the denominator the aggregated gain. The trade-off measure µ(xi, S) for each solution xi with respect to a set of neighboring solutions S is obtained by:

其中分子表示聚积的牺牲，分母表示聚积的收益。每个解xi对邻近解集S的折中度量µ(xi, S)，如下式：

$$µ(xi, S) = min_{xj∈S} T(xi, xj)$$(7)

It finds the minimum T(xi, xj) from xi to all other solutions xj ∈ S. Instead of calculating the metric with respect to all others, we provide the option to only consider the k closest neighbors in the objective space to reduce the computational complexity. Based on circumstances, the ‘min’ operator can be replaced with ‘average’, or ‘max’, or any other suitable operator. Thereafter, the solution having the maximum µ can be chosen as the preferred solution, meaning that this solution causes a maximum sacrifice in one of the objective values for a unit gain in another objective value for it be the most valuable solution for implementation.

这找到的是xi到所有其他解xj∈S的最小值。除了计算对所有其他解的度量，我们还给出了一个选项，只计算对在目标空间中k个最近的邻域解的距离，以降低计算复杂度。基于环境，min算子可以替换为average或max算子，或任何其他的合适的算子。此后，有最大µ值的解被选择为偏好解，意味着这个解对一个目标值有最大的牺牲，对其他目标值有单位的收益，这就是实现起来最宝贵的解。

The above methods are algorithmic, but requires an user interaction to choose a single preferred solution. However, in real practice, a more problem specific decision-making method must be used, such as an interaction EMO method suggested elsewhere [66]. We emphasize here the fact that multi-objective frameworks should include methods for multi-criteria decision making and support end-user further in choosing a solution out of a trade-off solution set.

上述方法是算法性的，但需要用户交互来选择一个偏好解。但是，在实际中，必须使用一个更加对问题特定的决策方法，比如[66]中的互动EMO方法。我们这里强调，多目标框架要包括多准则决策的方法，进一步支持用户从一个折中解集中选择一个解。

## 8. Concluding Remarks

This paper has introduced pymoo, a multi-objective optimization framework in Python. We have walked through our framework beginning with the installation up to the optimization of a constrained bi-objective optimization problem. Moreover, we have presented the overall architecture of the framework consisting of three core modules: Problems, Optimization, and Analytics. Each module has been described in depth and illustrative examples have been provided. We have shown that our framework covers various aspects of multi-objective optimization including the visualization of high-dimensional spaces and multi-criteria decision making to finally select a solution out of the obtained solution set. One distinguishing feature of our framework with other existing ones is that we have provided a few options for various key aspects of a multi-objective optimization task, providing standard evolutionary operators for optimization, standard performance metrics for evaluating a run, standard visualization techniques for showcasing obtained trade-off solutions, and a few approaches for decision-making. Most such implementations were originally suggested and developed by the second author and his collaborators for more than 25 years. Hence, we consider that the implementations of all such ideas are authentic and error-free. Thus, the results from the proposed framework should stay as benchmark results of implemented procedures.

本文介绍了pymoo，一个多目标优化框架。我们开始介绍了安装方法，然后是一个约束双目标优化问题的优化过程。而且，我们给出了框架的整体架构，包含3个核心模块：Problems，Optimization，和Analytics。每个模块都进行了深度介绍，给出了描述性的例子。我们展示了，我们的框架覆盖了多目标优化的各个方面，包括高维空间的可视化，多准则决策，以最终从得到的解集中选择一个解。我们框架的一个重要特征是，我们对多目标优化任务的各个关键方面给出了几个选项，给出了优化的标准演化算子，用于评估一次优化运行的标准性能准则，标准可视化技术，展现得到的折中解，以及几种决策的方法。多数这些实现是由第二作者及其同事在过去25年中提出和开发的。因此，我们认为所有这些思想的实现是真实无错误的。因此，提出的框架应当作为实现过程的基准测试结果。

However, the framework can be definitely extended to make it more comprehensive and we are constantly adding new capabilities based on practicalities learned from our collaboration with industries. In the future, we plan to implement more optimization algorithms and test problems to provide more choices to end-users. Also, we aim to implement some methods from the classical literature on single-objective optimization which can also be used for multi-objective optimization through decomposition or embedded as a local search. So far, we have provided a few basic performance metrics. We plan to extend this by creating a module that runs a list of algorithms on test problems automatically and provides a statistics of different performance indicators.

但是，框架还可以拓展，以使其更综合，我们一直在为其增加新的能力。未来，我们计划实现更多优化算法和测试问题，为用户提供更多选项。我们的目标是实现经典文献中单目标优化的一些方法，通过decomposition或嵌入到局部搜索中，也可以用在多目标优化中。迄今，我们给出了几个基本的性能度量。我们计划拓展，创建一个模块，在测试问题上自动运动一些算法，给出不同的性能指示器的统计值。

Furthermore, we like to mention that any kind of contribution is more than welcome. We see our framework as a collaborative collection from and to the multi-objective optimization community. By adding a method or algorithm to pymoo the community can benefit from a growing comprehensive framework and it can help researchers to advertise their methods. Interested researchers are welcome to contact the authors. In general, different kinds of contributions are possible and more information can be found online. Moreover, we would like to mention that even though we try to keep our framework as bug-free as possible, in case of exceptions during the execution or doubt of correctness, please contact us directly or use our issue tracker.

而且，我们还要提到，任何贡献都是受欢迎的。我们将框架视为多目标优化团体的合作结果。增加一个方法或算法，团体会从不断增长的综合框架中受益，可以帮助研究者宣传其方法。感兴趣的研究者可以联系作者。总体上，可能有不同的贡献，更多信息可以在线上找到。而且，我们还要提到，虽然我们试图让框架没有bug，如果发现执行的过程中可能有问题，请直接联系我们。