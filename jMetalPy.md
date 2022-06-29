# jMetalPy: a Python Framework for Multi-Objective Optimization with Metaheuristics

Antonio Ben´ıtez-Hidalgo et. al. @ University of M´alaga,

## 0. Abstract

This paper describes jMetalPy, an object-oriented Python-based framework for multi-objective optimization with metaheuristic techniques. Building upon our experiences with the well-known jMetal framework, we have developed a new multi-objective optimization software platform aiming not only at replicating the former one in a different programming language, but also at taking advantage of the full feature set of Python, including its facilities for fast prototyping and the large amount of available libraries for data processing, data analysis, data visualization, and high-performance computing. As a result, jMetalPy provides an environment for solving multi-objective optimization problems focused not only on traditional metaheuristics, but also on techniques supporting preference articulation, constrained and dynamic problems, along with a rich set of features related to the automatic generation of statistical data from the results generated, as well as the real-time and interactive visualization of the Pareto front approximations produced by the algorithms. jMetalPy offers additionally support for parallel computing in multicore and cluster systems. We include some use cases to explore the main features of jMetalPy and to illustrate how to work with it.

本文描述了jMetalPy，一种面向对象的基于Python的框架，用元启发技术进行多目标优化。我们在jMetal框架经验的基础之上，开发了新的多目标优化软件平台，其目标不仅是用另一种编程语言来复制之前的框架，而且还利用Python的完整特性，包括可以快速原型开发，和大量用于数据处理，数据分析，数据可视化，和高性能计算的库。结果是，jMetalPy提供了一个求解多目标优化问题的环境，不仅聚焦在传统的元启发式，还关注了支持倾向连接，约束和动态问题的技术，还有对得到的结果的统计数据的自动生成，以及对算法生成的Pareto front近似的互动和实时可视化。jMetalPy还对多核和集群系统的并行计算有额外支持。本文还有几个探索jMetalPy主要特征的使用案例，描述了怎样进行使用。

**Keywords**: Multi-Objective Optimization, Metaheuristics, Software Framework, Python, Statistical Analysis, Visualization

## 1. Introduction

Multi-objective optimization problems are widely found in many disciplines [1, 2], including engineering, economics, logistics, transportation or energy, among others. They are characterized by having two or more conflicting objective functions that have to be maximized or minimized at the same time, with their optimum composed by a set of trade-off solutions known as Pareto optimal set. Besides having several objectives, other factors can make this family of optimization problems particularly difficult to tackle and solve with exact techniques, such as deceptiveness, epistasis, NP-hard complexity, or high dimensionality [3]. As a consequence, the most popular techniques to deal with complex multi-objective optimization problems are metaheuristics [4], a family of non-exact algorithms including evolutionary algorithms and swarm intelligence methods (e.g. ant colony optimization or particle swarm optimization).

多目标优化问题广泛的应用于很多学科，包括工程学，经济学，物流，交通或能源，等等。这个问题的特征是，有两个或多个冲突的目标函数，需要在同时最大化或最小化，其最优值是由一组折中解组成，称之为Pareto最优解。除了有几个目标，其他因素会使得这种优化问题非常难以使用精确技术处理，比如deceptiveness，epistasis，NP难题，和高维。结果是，处理复杂的多目标优化最流行的技术是元启发式，一族不精确的算法，包括演化算法和群体智能方法（如，蚁群优化或粒子群优化）。

An important factor that has ignited the widespread adoption of metaheuristics is the availability of software tools easing their implementation, execution and deployment in practical setups. In the context of multi-objective optimization, one of the most acknowledged frameworks is jMetal [5], a project started in 2006 that has been continuously evolving since then, including a full redesign from scratch in 2015 [6]. jMetal is implemented in Java under the MIT licence, and its source code is publicly available in GitHub.

元启发式的广泛采用，一个重要因素是有软件包可用，使其实现，执行和部署非常容易。在多目标优化的上下文中，最重要的一种框架就是jMetal，该项目在2006年启动，一直在持续演化，在2015年进行了很多细节的重新设计。jMetal用Java实现，代码已开源。

In this paper, we present jMetalPy, a new multi-objective optimization framework written in Python. Our motivation for developing jMetalPy stems from our past experience with jMetal and from the fact that nowadays Python has become a very prominent programming language with a plethora of interesting features, which enables fast prototyping fueled by its large ecosystem of libraries for numerical and scientific computing (NumPy [7], Scipy [8]), data analysis (Pandas), machine learning (Scikit-learn [9]), visualization (Matplotlib [10], Plotly [11]), large-scale processing (Dask [12], PySpark [13]) and so forth. Our goal is not only to rewrite jMetal in Python, but to focus mainly on aspects where Python can help fill the gaps not covered by Java. In particular, we place our interest in the analysis of results provided by the optimization algorithms, realtime and interactive visualization, preference articulation for supporting decision making, and solving constrained and dynamic problems. Furthermore, since Python can be thought of as a more agile programming environment for prototyping new multi-objective solvers, jMetalPy also incorporates a full suite of statistical significance tests and related tools for the sake of a principled comparison among multi-objective metaheuristics.

本文中，我们给出jMetalPy，用Python写的一个新的多目标优化框架。我们开发jMetalPy的动机源自于使用jMetal的经验，实际上，在今天，Python已经成为了非常流行的编程语言，有很多有趣的特征，可以快速原型实现，有数值计算和科学计算的大量生态库（NumPy，SciPy，Pandas，Scikit-learn，Matplotlib，Plotly，Dask，PySpark）等。我们的目标不仅是用Python重写jMetal，而且是聚焦在Python可以帮助覆盖Java不能覆盖的地方。特别是，我们还对优化算法给出的结果进行了分析，进行实时和互动的可视化，对支持的决策算法进行preference articulation，求解约束和动态问题。而且，因为Python可以对新的多目标求解器进行原型实现，jMetal还可以对这些算法进行有效的比较。

jMetalPy has been developed by Computer Science engineers and scientists to support research in multi-objective optimization with metaheuristics, and to utilize the provided algorithms for solving real-word problems.

jMetalPy由于计算机科学工程师和科学家进行开发，以支持多目标优化的研究，利用提供的算法进行真实世界问题求解。

Following the same open source philosophy as in jMetal, jMetalPy is released under the MIT license. The project is in continuous development, with its source code hosted in GitHub, where the last stable and current development versions can be freely obtained.

按照与jMetal相同的开源哲学，jMetalPy也是按照MIT许可开放。项目在进行持续开发，代码放在Github上。

The main features of jMetalPy are summarized as follows: 主要特征总结如下：

• jMetalPy is implemented in Python (version 3.6+), and its object-oriented architecture makes it flexible and extensible.

jMetalPy用Python实现，面向对象的架构使其很灵活，可扩展。

• It provides a set of representative multi-objective metaheuristics of the state-of-the-art (including NSGA-II [14], NSGA-III [15], GDE3 [16], SMPSO [17], OMOPSO [18], MOCell [19], IBEA [20], SPEA2 [21], HypE [22], MOEA/D-DE [23], MOEA/D-DRA [24]) and standard families of problems for benchmarking (ZDT [25], DTLZ [26], WFG [2], and LZ09 [27]).

给出了一些有代表性的多目标优化算法的实现（包括NSGA-II [14], NSGA-III [15], GDE3 [16], SMPSO [17], OMOPSO [18], MOCell [19], IBEA [20], SPEA2 [21], HypE [22], MOEA/D-DE [23], MOEA/D-DRA [24]），和用于基准测试的问题标准库（ZDT [25], DTLZ [26], WFG [2], and LZ09 [27]）。

• Most of the included algorithms can solve classical benchmark constrained problems using the constraint handling approach applied in NSGA-II. Furthermore, recent proposals, such as the MOEA/D-IEpsilon algorithm and the LIR-CMOP test suite [28], are also included.

多数包含的算法可以求解经典的基准测试约束问题，使用NSGA-II中的约束处理方法。而且，最近提出的算法，比如MOEA/D-IEpsilon算法和LIR-CMOP测试包，也包含在内。

• Dynamic multi-objective optimization is supported, including the implementation of dynamic versions of NSGA-II and SMPSO, as well as the FDA [29] problem family.

支持动态多目标优化，包含NSGA-II和SMPSO的动态版本，以及FDA问题族。

• Reference point based preference articulation algorithms, such as SMPSO/RP [30] and versions of NSGA-II and GDE3, are also provided.

给出了基于参考点的preference articulation算法，比如SMPSO/RP，和这些版本的NSGA-II和GDE3。

• It implements quality indicators for multi-objective optimization, such as Hypervolume [31], Additive Epsilon [32] and Inverted Generational Distance [33].

实现了多目标优化算法的quality indicators，比如HyperVolume，Additive Epsilon和IGD。

• It provides visualization components to display the Pareto front approximations when solving problems with two objectives (scatter plot), three objectives (scatter plot 3D), and many-objective problems (parallel coordinates graph and a tailored version of Chord diagrams).

给出了可视化组成部分，以展示Pareto front近似，在求解双目标问题时为散点图，三目标问题时为3D散点图，多目标问题时为并行坐标图和定制的chord图。

• Support for comparative studies, including a wide number of statistical tests and utilities (e.g. non-parametric test, post-hoc tests, boxplots, CD plot), including the automatic generation of LATEX tables (mean, standard deviation, median, interquartile range) and figures in different formats.

支持比较性研究，包括大量统计测试和工具（比如，非参数测试，post-hoc测试，boxplots，CD plot），包括各种不同格式的Latex表格的自动生成（均值，标准差，中值，四分点范围）和图。

• jMetalPy can cooperatively work alongside with jMetal. The latter can be used to run algorithms and compute the quality indicators, while the post-processing data analysis can be carried out with jMetalPy.

jMetalPy可以与jMetal联合使用。后者可以用于运行算法，计算quality indicators，而jMetalPy可以用于数据分析的后处理。

• Parallel computing is supported based on Apache Spark [34] and Dask [12]. This includes an evaluator component that can be used by generational metaheuristics to evaluate solutions in parallel with Spark (synchronous parallelism), as well as a parallel version of NSGA-II based on Dask (asynchronous parallelism).

基于Apache Spark和Dask，可以支持并行计算。这包括一个evaluator组成部分，可以由代际元启发式使用，用Spark并行的评估解（同步并行），以及基于Dask（非同步并行）的并行版本的NSGA-II。

• Supporting documentation. A website is maintained with user manuals and API specification for developers. This site also contains a series of Jupyter notebooks with use cases and examples of experiments and visualizations.

支持文档。维护了一个网站，包含用户手册和API说明。这个网站还包含一系列Jupyter notebooks，是试验和可视化的使用案例和例子。

Our purpose of this paper is to describe jMetalPy, and to illustrate how it can be used by members of the community interested in experimenting with metaheuristics for solving multi-objective optimization problems. To this end, we include some implementation use cases based on NSGA-II to explore the main variants considered in jMetalPy, from standard versions (generational and steady state), to dynamic, reference-point based, parallel and distributed flavors of this solver. A experimental use case is also described to exemplify how the statistical tests and visualization tools included in jMetalPy can be used for post-processing and analyzing the obtained results in depth. For background concepts and formal definitions of multi-objective optimization, we refer to our previous work in [5].

本文的目标是描述jMetalPy，有兴趣用其求解多目标优化问题的人怎么使用。为此，我们包括了一些实现的使用案例，基于NSGA-II，以用jMetalPy探索主要的变体，从标准版，到动态，基于参考点的，并行的和分布式的求解器。描述了一个试验性的使用案例，以说明统计测试和可视化工具怎样进行使用，以对得到的结果进行后处理和分析。对于多目标优化的背景概念，和正式定义，我们推荐参考我们之前的工作[5]。

The remaining of this paper is organized as follows. In Section 2, a review of relevant related algorithmic software platforms is conducted to give an insight and rationale of the main differences and contribution of jMetalPy. Section 3 delves into the jMetalPy architecture and its main components. Section 4 explains a use case of implementation. Visualization facilities are described in Section 5, while a use case of experimentation with statistical procedures is explained in Section 6. Finally, Section 7 presents the conclusions and outlines further related work planned for the near future.

本文剩下的部分组织如下。在第2部分，回顾了相关的算法软件平台，以得到jMetalPy的主要区别和贡献。第3部分给出了jMetalPy架构和其组成部分。第4部分解释了一个使用案例的实现。第5部分描述了可视化部分，第6部分给出了一个使用案例的试验和统计过程。最后，第7部分给出了结论，以及未来工作的打算。

## 2. Related Works

In the last two decades, a number of software frameworks devoted to the implementation of multi-objective metaheuristics has been contributed to the community, such as ECJ [41], EvA [42], JCLEC-MO [43], jMetal [5, 6], MOEA Framework[44], and Opt4J [45], which are written in Java; ParadisEO-MOEO [46], and PISA [47], developed in C/C++; and PlatEMO[48], implemented in Matlab. They all have in common the inclusion of representative algorithms from the the state of the art, benchmark problems and quality indicators for performance assessment.

在过去二十年，有一些实现多目标元启发式优化的软件框架得到了实现，比如ECJ，EvA，JCLEC-MO，jMetal，MOEA框架，Opt4J，这些都是用Java写的；ParadiseEO-MOEO，和PISA，是用C/C++开发的；PlatEMO，是用MATLAB实现的。它们的共同点是，都包含了目标最好的代表性算法，基准测试集，和用于性能评估的quality indicators。

As has been mentioned in the introduction, there is a growing interest within the scientific community in software frameworks implemented in Python, since this language offers a large ecosystem of libraries, most of them devoted to data analysis, data processing and visualization. When it comes to optimization algorithms, a set of representative Python frameworks is listed in Table 1, where they are analyzed according to their algorithmic domains, maintenance status, Python version and licensing, as well as the featured variants, post-processing facilities and algorithms they currently offer. With the exception of the Inspyred framework, they are all active projects (i.e., their public source code have been updated at least one time within the last six months) and work out-of-the-box with a simple pip command. All of these frameworks support Python 3.x.

在介绍中提到过，在科研团体中对用Python实现的软件框架越来越有兴趣，因为这种语言有很大的生态库，多数都用于数据分析，数据处理和可视化。在优化算法中，一些有代表性的Python框架如表1所示，根据算法领域，维护状态，Python版本和许可情况进行了分析，以及包含的有特点的变体，后处理工具和算法。除了Inspyred框架，这些都是活跃的项目（即，在最后6个月中至少有过一次更新），可以用一个简单的pip命令进行工作。所有这些框架都支持Python 3.x。

DEAP and Inspyred are not centered in multi-objective optimization, and they include a shorter number of implemented algorithms. Pagmo/PyGMO, Platypus and Pymoo offer a higher number of features and algorithmic variants, including methods for statistical post-processing and visualization of results. In particular, Pagmo/PyGMO contains implementations of a number of single/multi-objective algorithms, including hybrid variants, with statistical methods for racing algorithms, quality indicators and fitness landscape analysis. Platypus supports parallel processing in solution evaluation phase, whereas Pymoo is rather focused on offering methods for preference articulation based on reference points.

DEAP和Inspyred并不是以多目标优化为中心的，其中所包含的实现的算法较少。Pagmo/PyGMO，Platypus和Pymoo的特征和算法变体更多，包含结果的统计后处理和可视化。特别是，Pagmo/PyGMO包含一些单/多目标算法的实现，包含混合变体，对racing算法有统计方法，有quality indicators和fitness landscape分析。Platypus支持解评估阶段的并行处理，而Pymoo专注于给出基于参考点的preference articulation。

The jMetalPy framework we proposed in this paper is also an active open source project, which is focused mainly on multi-objective optimization (although a number of single-objective algorithms are included) providing an increasing number of algorithms and modern methods for statistical post-processing and visualization of results. It offers algorithmic variants with methods for parallel processing and preference articulation based on reference points to provide decision making support. Moreover, jMetalPy incorporates algorithms and mechanisms for dynamic problem optimization, which is an additional feature not present in the other related frameworks. In this way, the proposed framework attempts at covering as many enhancing features in optimization as possible to support experimentation and decision making in both research and industry communities. Besides these features, an important design goal in jMetalPy has been to make the code easy to understand (in particular, the implementation of the algorithms), to reuse and to extend, as is illustrated in the next two sections.

我们在本文中提出的jMetalPy框架，也是一个活跃的开源项目，主要关注于多目标优化（同时还包含几种单目标优化算法），给出的算法越来越多，还包括对结果的统计后处理和可视化。给出的算法变体，带有并行处理的方法，和基于参考点的preference articulation，以支持决策。而且，jMetalPy包含了动态问题优化的算法和机制，别的框架则没有这个特征。提出的框架包含了尽可能多的强化特征，以支持研究和工业中的决策机制。除了这些特征，jMetalPy中一种重要的设计目标是，让代码尽量容易理解（特别是，在算法实现中），容易重用，容易拓展，如下面两节所示。

## 3. Architecture of jMetalPy

The architecture of jMetalPy has an object-oriented design to make it flexible and extensible (see Figure 1). The core classes define the basic functionality of jMetalPy: an Algorithm solves a Problem by using some Operator entities which manipulate a set of Solution objects. We detail these classes next.

jMetalPy的架构是面向对象的设计，使其灵活可拓展，如图1所示。核心类定义了jMetalPy的基础功能：一个Algorithm求解一个Problem，使用了一些Operator实体，操作Solution目标的集合。下面我们详细叙述这些类。

### 3.1. Core Architecture

Class Algorithm contains a list of solutions (i.e. population in Evolutionary Algorithms or swarm in Swarm Intelligence techniques) and a run() method that implements the behavior of a generic metaheuristic (for the sake of simplicity, full details of the codes are omitted):

类Algorithm包含解的列表（即，演化算法中的种群，或群体智能中的群），和一个run()方法，实现一个通用元启发式的行为（为简单起见，代码完整细节忽略）

```
class Algorithm(ABC):
def __init__(self):
    self.evaluations = 0
    self.solutions = List[]
    self.observable = DefaultObservable()

def run(self):
    self.solutions = self.create_initial_solutions()
    self.solutions = self.evaluate(self.solutions)
    self.init_progress()
    while not self.stopping_condition_is_met():
        self.step()
        self.update_progress()
```

In the above code we note the steps of creating the initial set of solutions, their evaluation, and the main loop of the algorithm, which performs a number of steps until a stopping condition is met. The initialization of state variables of an algorithm and their update at the end of each step are carried out in the init_progress() and update_progress() methods, respectively. In order to allow the communication of the status of an algorithm while running we have adopted the observer pattern [49], so that any algorithm is an observable entity which notifies to registered observers some information specified in advance (e.g., the current evaluation number, running time, or the current solution list), typically in the update_progress() method. In this way we provide a structured method, for example, to display in real-time the current Pareto front approximation or to store it in a file.

在上面的代码中，我们说明一下下面的步骤，创建初始解集，这些解集的评估，算法的主要循环，进行若干次迭代，直到满足停止条件。一个算法的状态变量的初始化以及在每个步骤后的更新，分别是在init_process()和update_process()方法中的。为让算法中的状态在运行时能通信，我们采用了观察者模式，这样任意算法都是一个可观察的实体，会通知给注册的观察者一些事先指定的信息（如，当前的评估数量，运行时间，或当前的解列表），一般是的update_progess()方法中。这样，我们给出了一个结构化的方法，比如，来实时展示当前的Pareto front估计，或将其存储到文件中。

A problem is responsible of creating and evaluating solutions, and it is characterized by its number of decision variables, objectives and constraints. In case of the number of constraints be greater than 0, it is assumed that the evaluate() method also assesses whether the constraints are fulfilled. Subclasses of Problem include additional information depending of the assumed solution encoding; thus, a FloatProblem (for numerical optimization) or an IntegerProblem (for combinatorial optimization) requires the specification of the lower and upper bounds of the decision variables.

一个Problem要负责创建和评估解，其特征是决策变量、目标和约束的数量。在约束数量大于0的情况下，要假设evaluate()方法还会评估约束是否满足。Problem的子类会包含额外的信息，这依赖于假设的解的编码；因此，一个FloatProblem（用于数值优化），或IntegerProblem（用于组合优化），需要决策变量的上界和下界。

Operators such as Mutation, Crossover, and Selection, have an execute(source) method which, given a source object, produces a result. Mutations operate on a solution and return a new one resulting from modifying the original one. On the contrary, crossover operators take a list of solutions (namely, the parents) and produce another list of solutions (correspondingly, the offspring). Selection operators usually receive a list of solutions and returns one of them or a sublist of them.

像Mutation，Crossover和Selection这样的算子，有一个execute(source)方法，在给定source对象时，会产生一个结果。Mutation在一个解上运算，修改原始的解，返回一个新的解。相反的，crossover算子以解的列表为输入（即，父辈），生成解的另一个列表（对应的，子代）。Selection算子通常以解的列表为输入，返回其中的一个，或一个子列表。

The Solution class is a key component in jMetalPy because it is used to represent the available solution encodings, which are linked to the problem type and the operators that can be used to solve it. Every solution is composed by a list of variables, a list of objective values, and a set of attributes implemented as a dictionary of key-value pairs. Attributes can be used to assign, for example, a rank to the solutions of population or a constraint violation degree. Depending on the type of the variables, we have subclasses of Solution such as FloatSolution, IntegerSolution, BinarySolution or PermutationSolution.

Solution类是jMetalPy中的一个关键组成部分，因为它用于表示可用的解的编码，这与问题类型和用于求解之的算子是有关联的。每个解都是由一个变量列表，一个目标值列表，和一个实现为key-value对的字典的属性集合组成的。比如，属性可以用于指定一个rank到解的种群，或一个约束违反程度。依赖于变量类型，我们有Solution的子类，比如FloatSolution，IntegerSolution，BinarySolution，或PermutationSolution。

### 3.2. Classes for Dynamic Optimization

jMetalPy supports dealing with dynamic optimization problems, i.e., problems that change over time. For this purpose, it contains two abstract classes named DynamicProblem and DynamicAlgorithm.

jMetalPy支持处理动态优化问题，即，随着时间而变化的问题。为此，它包含两个抽象类，名为DynamicProblem和DynamicAlgorithm。

A dynamic algorithm is defined as an algorithm with a restarting method, which is called whenever a change in the problem being solved is detected. The code of the DynamicAlgorithm class is as follows:

一个动态算法定义为，带有重启方法的算法，在要求解的问题有所变化时，就进行调用。DynamicAlgorithm类的代码定义如下：

```
class DynamicAlgorithm(Algorithm, ABC):
@abstractmethod
def restart(self) -> None:
    pass
```

The DynamicProblem class extends Problem with methods to query whether the problem has changed whatsoever, and to clear that status:

DynamicProblem类对Problem进行了拓展，有方法来查询问题是否有改变，并清除这个状态：

```
class DynamicProblem(Problem, Observer, ABC):
@abstractmethod
def the_problem_has_changed(self) -> bool:
    pass

@abstractmethod
clear_changed(self) -> None:
pass
```

It is worth mentioning that a dynamic problem is also an observer entity according to the observer pattern. The underlying idea is that in jMetalPy it is assumed that changes in a dynamic problem are produced by external entities, i.e, observable objects where the problem is registered.

值得注意的是，根据观察者模式，一个动态问题仍然是一个观察者实体。潜在的思想是，在jMetalPy中假设的是，在动态问题中的变化，是由外部实体产生的，即，问题所注册的可观察的对象中。

## 4. Implementation Use Case: NSGA-II and Variants

With the aim of illustrating the basic usages of jMetalPy, in this section we describe the implementation of the well-known NSGA-II algorithm [14], as well as some of its variants (steady-state, dynamic, with preference articulation, parallel, and distributed).

本节的目标是描述jMetalPy的基本使用，我们描述了著名的NSGA-II算法的实现，以及该算法的几个变体（稳态，动态，带有preference articulation，并行的，和分布式的）。

NSGA-II is a genetic algorithm, which is a subclass of Evolutionary Algorithms. In jMetalPy we include an abstract class for the latter, and a default implementation for the former. An Evolutionary Algorithm is a metaheuristic where the step() method consists of applying a sequence of selection, reproduction, and replacement methods, as illustrated in the code snippet below:

NSGA-II是一个遗传算法，是演化算法的一个子类。在jMetalPy中，我们有一个演化算法的抽象类，和NSGA-II的一个默认实现。一个演化算法是一个元启发式，其中step()方法包含selection，reproduction和replacement方法的序列，如下面的代码片段所示：

```
class EvolutionaryAlgorithm(Algorithm, ABC):
def __init__(self,
    problem: Problem,
    population_size: int,
    offspring_size: int):
    
    super(EvolutionaryAlgorithm, self).__init__()
    self.problem = problem
    self.population_size = population_size
    self.offspring_size = offspring_size

@abstractmethod
def selection(self, population):
    pass

@abstractmethod
def reproduction(self, population):
    pass

@abstractmethod
def replacement(self, population, offspring):
    pass

def init_progress(self):
    self.evaluations = self.population_size

def step(self):
    mating_pool = self.selection(self.solutions)
    offspring = self.reproduction(mating_pool)
    offspring = self.evaluate(offspring)
    self.solutions = self.replacement(self.solutions, offspring)

def update_progress(self):
    self.evaluations += self.offspring_size
```

On every step, the selection operator is used (line 27) to retrieve the mating pool from the solution list (the population) of the algorithm. Solutions of the mating pool are taken for reproduction (line 28), which yields a new list of solutions called offspring. Solutions of this offspring population must be evaluated (line 29), and thereafter a replacement strategy is applied to update the population (line 30). We can observe that the evaluation counter is initialized and updated in the init_progress() (line 23) and update_progress (line 32), respectively.

在每个step()中，用selection算子来从算法的解列表（种群）中获取配对池（第27行）。配对池的解用作reproduction（第28行），生成了新的解的列表，称为子代。子代种群的解必须进行评估（第29行），后面应用一个replacement策略来更新种群（第30行）。我们可以观察到，评估计数器的初始化和更新分别在第23行的init_process()和第32行的update_progress()进行。

The EvolutionaryAlgorithm class is very generic. We provide a complete implementation of a Genetic Algorithm, which is an evolutionary algorithm where the reproduction is composed by combining a crossover and mutation operator. We partially illustrate this implementation next:

EvolutionaryAlgorithm类是非常通用的。我们给出了遗传算法的一个完整实现，这是一种演化算法，其中reproduction是由一个crossover和Mutation算子结合实现的。下面我们给出其部分实现：

```
class GeneticAlgorithm(EvolutionaryAlgorithm):
def __init__(self,
    problem: Problem[Solution],
    population_size: int,
    offspring_population_size: int,
    mutation: Mutation,
    crossover: Crossover,
    selection: Selection,
    termination_criterion: TerminationCriterion,
    population_generator=RandomGenerator(),
    population_evaluator=SequentialEvaluator()):
...

def create_initial_solutions(self):
    return [self.population_generator.new(self.problem)
        for _ in range(self.population_size)]

def evaluate(self, solutions):
    return self.population_evaluator.evaluate(solutions, self.problem)

def stopping_condition_is_met(self):
    return self.termination_criterion.is_met

def selection(self, population: List[Solution]):
    # select solutions to get the mating pool

def reproduction(self, mating_pool):
    # apply crossover and mutation

def replacement(self, population, offspring):
    # combine the population and offspring populations
```

There are some interesting features to point out here. First, the initial solution list is created from a Generator object (line 14), which, given a problem, returns a number of new solutions according to some strategy implemented in the generator; by default, a RandomGenerator() is chosen to produce a number of solutions uniformly drawn at random from the value range specified for the decision variables. Second, an Evaluator object is used to evaluate all produced solutions (line 19); the default one evaluates the solutions sequentially. Third, a TerminationCriterion object is used to check the stopping condition (line 21), which allows deciding among several stopping criteria when configured. The provided implementations include: stopping after making a maximum number of evaluations, computing for a maximum time, a key has been pressed, or the current population achieves a minimum level of quality according to some indicator. Fourth, the reproduction method applies the crossover and mutation operators over the mating pool to generate the offspring population. Finally, the replacement method combines the population and the offspring population to produce a new population.

有一些有趣的特征需要指出。第一，初始解列表是从一个Generator对象生成的（第14行），在给定一个问题时，会根据generator中实现的某种策略，返回一定数量的新解；默认的会选择RandomGenerator()，来从决策变量指定的值的范围中随机均匀选择，来生成一定数量的解；第二，用一个Evaluator对象，来评估所有产生的解（第19行）；默认的会顺序评估所有解。第三，用一个TerminationCriterion对象来检查停止条件（第21行），如果进行了配置，也可以从几个停止准则中进行决策。提供的实现包括：在最大数量的评估次数后停止，计算达到了最大时间后停止，按下一个键后停止，或当前的种群根据某种indicator获得最小质量级别后停止。第四，reproduction方法对配对池应用crossover和Mutation算子，来生成子代种群。最后，replacement方法将原有种群和子代种群结合到一起，以生成一个新的种群。

Departing from the implemented GeneticAlgorithm class, we are ready to implement the standard NSGA-II algorithm and some variants, which will be described in the next subsections. Computing times will be reported when running the algorithm to solve the ZDT1 benchmark problem [25] on a MacBook Pro with macOS Mojave, 2.2 GHz Intel Core i7 processor (Turbo boost up to 3.4GHz), 16 GB 1600 MHz DDR3 RAM, Python 3.6.7 :: Anaconda.

离开实现的GeneticAlgorithm类，我们准备好实现标准的NSGA-II算法以及几个变体了，下面我们会进行描述。给出的计算时间，是在一个MacBook Pro上运行算法求解ZDT1基准测试问题的时间。

### 4.1. Standard Generational NSGA-II

NSGA-II is a generational genetic algorithm, so the population and the offspring population have the same size. Its main feature is the use of a non-dominated sorting for ranking the solutions in a population to foster convergence, and a crowding distance density estimator to promote diversity [14]. These mechanisms are applied in the replacement method, as shown in the following snippet:

NSGA-II是一个代式遗传算法，所以种群和子代种群的大小是一样的。其主要特征是，使用对种群中解的ranking使用非占优排序，来促进收敛，并用一个拥挤度距离密度估计器，来提升多样性。这些机制是在replacement方法中应用的，如下面的代码片段所示：

```
class NSGAII(GeneticAlgorithm):
def __init__(self,
    problem: Problem,
    population_size,
    offspring_size,
    mutation: Mutation,
    crossover: Crossover,
    selection: Selection,
    termination_criterion: TerminationCriterion,
    population_generator=RandomGenerator(),
    population_evaluator=SequentialEvaluator()
    dominance_comparator=DominanceComparator()):
...

def replacement(self, population, offspring):
    join_population = population + offspring
    return RankingAndCrowdingDistanceSelection(self.population_size, self.dominance_comparator).execute(join_population)
```

No more code is needed. To configure and run the algorithm we include some examples, such as the following code: 不需要更多的代码了。为配置和运行算法，我们包括了一些例子，比如下面的代码：

```
# Standard generational NSGAII runner
problem = ZDT1()
max_evaluations = 25000
algorithm = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=PolynomialMutation(...),
    crossover=SBXCrossover(...),
    selection=BinaryTournamentSelection(...),
    termination_criterion=StoppingByEvaluations(max=max_evaluations),
    dominance_comparator=DominanceComparator()
)

progress_bar = ProgressBarObserver(max=max_evals)
algorithm.observable.register(observer=progress_bar)

real_time = VisualizerObserver()
algorithm.observable.register(observer=real_time)

algorithm.run()
front = algorithm.get_result()

# Save results to file
print_function_values_to_file(front, ‘FUN’)
print_variables_to_file(front, ‘VAR’)
```

This code snippet depicts a standard configuration of NSGAII to solve the ZDT1 benchmark problem. Note that we can define a dominance comparator (line 13), which by default is the one used in the standard implementation of NSGA-II.

这个代码片段描述了NSGA-II的标准配置，以求解ZDT1基准测试问题。注意，我们可以定义一个占优性比较器（第13行），这是在NSGA-II的标准实现中默认使用的。

As commented previously, any algorithm is an observable entity, so observers can register into it. In this code, we register a progress bar observer (shows a bar in the terminal indicating the progress of the algorithm) and a visualizer observer (shows a graph plotting the current population, i.e., the current Pareto front approximation). A screen capture of NSGA-II running in included in Figure 2. The computing time of NSGA-II with this configuration in our target laptop is around 9.2 seconds.

之前评论过，任何算法都是一个可观察的实体，所以观察者可以注册到其中。在这个代码中，我们注册了一个进度条观察者（在终端中展示一个条，表示算法的进度），和一个可视化器观察者（展现一个图，画出了当前的种群，即，当前的Pareto front近似）。图2中展示了NSGA-II运行的一个截图。用这个配置，NSGA-II在当前笔记本上的计算时间为大约9.2s。

### 4.2. Steady-State NSGA-II

A steady-state version of NSGA-II can be configured by resorting to the same code, but just setting the offspring population size to one. This version yielded a better performance in terms of the produced Pareto front approximation compared with the standard NSGA-II as reported in a previous study [50], but at a cost of a higher computing time, which raises up to 190 seconds.

NSGA-II的稳态版可以用相同的代码进行配置，只要将子代种群大小设为1。这个版本得到的Pareto front近似，与标准版NSGA-II相比，会有更好的性能，但代价是计算时间较长，最长会达到190s。

An example of Pareto front approximation found by this version of NSGA-II when solving the ZDT1 benchmark problem is shown in Figure 3-center. As expected given the literature, it compares favorably against the one generated by the standard NSGA-II (Figure 3-left).

这个版本的NSGA-II求解ZDT1基准测试问题，得到的Pareto front近似的一个例子，如图3中间所示。就像文献中给出的一样，其结果与标准NSGA-II的差不多。

### 4.3. NSGA-II with Preference Articulation

The NSGA-II implementation in jMetalPy can be easily extended to incorporate a preference articulation scheme. Concretely, we have developed a g-dominance based comparator considering the g-dominance concept described in [51], where a region of interest can be delimited by defining a reference point. If we desire to focus the search in the interest region delimited by the reference point, say e.g. [f1, f2] = [0.5, 0.5], we can configure NSGA-II with this comparator as follows:

jMetalPy中实现的NSGA-II可以很容易拓展，与preference articulation结合到一起。具体的，我们基于[51]中g-dominance的概念，开发了一个基于g-dominance的比较器，通过定义一个参考点，来限制一个感兴趣区域。如果我们希望将搜索聚焦在参考点限制的感兴趣区域中，比如参考点为[f1, f2] = [0.5, 0.5]，我们可以按照如下配置NSGA-II：

```
reference_point = [0.5, 0.5]
algorithm = NSGAII(... dominance_comparator=GDominanceComparator(reference_point))
```

The resulting front is show in Figure 3-right. 得到的结果如图3右。

### 4.4. Dynamic NSGA-II

The approach adopted in jMetalPy to provide support for dynamic problem solving is as follows: First, we have developed a TimeCounter class (which is an Observable entity) which, given a delay, increments continuously a counter and notifies the registered observers the new counter values; second, we need to define an instance of DynamicProblem, which must implement the methods for checking whether the problem has changed and to clear the changed state. As DynamicProblem inherits from Observer, instances of this class can register in a TimeCounter object. Finally, it is required to extend DynamicAlgorithm with a class defining the restart() method that will be called when the algorithm detects a change in a dynamic problem. The following code snippet shows the implementation of the DynamicNSGAII class:

jMetalPy中采用的用以支持动态问题求解的方法如下：首先，我们开发了一个TimeCounter类（这是一个可观察实体），在给定一个延迟时，会对一个计数器持续的递增，通知注册的观察者新的计数器值；第二，我们需要定义DynamicProblem的一个实例，这个类必须实现检查问题是否改变的方法，清除改变的状态。由DynamicProblem从Observer继承得到，这个类的实例可以在一个TimeCounter对象中注册。最后，要拓展DynamicAlgorithm类，定义restart()方法，在算法检测到动态问题的变化时，就会调用这个方法。下面的代码片段展示了DynamicNSGAII类的实现：

```
class DynamicNSGAII(NSGAII, DynamicAlgorithm):
def __init__(self, ...):
    ...
    self.completed_iterations = 0

def restart(self) -> None
    # restart strategy

def update_progress(self):
    if self.problem.the_problem_has_changed():
        self.restart()
        self.evaluator.evaluate(self.solutions, problem)
        self.problem.clear_changed()
        self.evaluations += self.offspring_size

def stopping_condition_is_met(self):
    if self.termination_criterion.is_met:
        self.restart()
        self.evaluator.evaluate(self.solutions, problem)
        self.init_progress()
        self.completed_iterations += 1
```

As shown above, at the end of each iteration a check is made about a change in the problem. If a change has occurred, the restart method is invoked which, depending on the implemented strategy, will remove some solutions from the population and new ones will be created to replace them. The resulting population will be evaluated and the clear_changed() method of the problem object will be called. As opposed to the standard NSGA-II, the stopping condition method is not invoked to halt the algorithm, but instead to notify registered observers (e.g., a visualizer) that a new resulting population has been produced. Then, the algorithm starts again by invoking the restart() and init_progress() methods. It is worth noting that most of the code of the original NSGA-II implementation is reused and only some methods need to be rewritten.

像上面所示的，在每次迭代的最后，要检查问题是否有变化。如果发生了一个改变，就调用restart方法，按照实现的策略，会从种群中移除掉一些解，创建新的解以进行替换。得到的种群会被评估，会调用Problem对象的clear_changed()方法。至于标准NSGA-II，并没有调用停止条件方法来停止算法，而是通知注册的观察者（如，一个可视化器），生成了一个新的结果种群。然后，通过调用restart()和init_progress()方法，算法可以又开始。值得提到的是，原始NSGA-II的实现的多数代码都可以重用，只有一些方法需要重写。

To illustrate the implementation a dynamic problem, we next show code of the FDA abstract class, which is the base class of the five problems composing the FDA benchmark:

为描述一个动态问题的实现，下面我们展示FDA抽象类的代码，这是FDA基准测试的5个问题的基础类：

```
class FDA(DynamicProblem, FloatProblem, ABC):
def __init__(self):
    super(FDA, self).__init__()
    self.tau_T = 5
    self.nT = 10
    self.time = 1.0
    self.problem_modified = False

def update(self, *args, **kwargs):
    counter = kwargs[’COUNTER’]
    self.time = (1.0 / self.nT) * floor(counter * 1.0 / self.tau_T)
    self.problem_modified = True

def the_problem_has_changed(self) -> bool:
    return self.problem_modified

def clear_changed(self) -> None:
    self.problem_modified = False
```

The key point in this class is the update() method which, when invoked by an observable entity (e.g., an instance of the aforementioned TimeCounter class), sets the problem modified flag to True. We can observe that this flag can be queried and reset.

这个类中的关键点是update()方法，当由一个可观察的实体调用时（如，之前提到的TimeCounter类的一个实例），将问题修改标记设置为True。我们可以观察到，这个flag可以被查询和重置。

The code presented next shows how to configure and run the dynamic NSGA-II algorithm: 下面给出的代码展示了怎样配置和运行动态NSGA-II算法：

```
# Dynamic NSGA-II runner
problem = FDA2()
time_counter = TimeCounter(delay=1))
time_counter.observable.register(problem)
time_counter.start()

algorithm = DynamicNSGAII(
...
termination_criterion=StoppingByEvaluations(max=max_evals)
)
algorithm.run()
```

After creating the instances of the FDA2 benchmark problem [29] and the time counter class, the former is registered in the latter, which runs in a concurrent thread. The dynamic NSGA-II is set with stopping condition which returns a Pareto front approximation every 25,000 function evaluations. An example of running of the dynamic NSGA-II algorithm when solving the FDA2 problem is shown in Figure 4.

在创建了FDA2基准测试问题和时间计时器类的实例后，前者在后者中进行了注册，在并发的线程中进行运行。动态NSGA-II用停止条件进行设置，每25000次函数评估后，返回一次Pareto front近似。求解FDA2问题时，运行动态NSGA-II算法的一个例子，如图4所示。

### 4.5. Parallel NSGA-II with Apache Spark

In order to evaluate a population, NSGA-II (and in general, any generational algorithms in jMetalPy) can use an evaluator object. The default evaluator runs in a sequential fashion but, should the evaluate method of the problem be thread-safe, solutions can be evaluated in parallel. jMetalPy includes an evaluator based on Apache Spark, so the solutions can be evaluated in a variety of parallel systems (multicores, clusters) following the scheme presented in [52]. This evaluator can be used as exemplified next:

为评估一个种群，NSGA-II（更一般的，jMetalPy中的任何代式算法）都可以使用一个evaluator对象。默认的evaluator以顺序的方式运行，但是，如果问题的评估方法是线程安全的，那么解就可以并行评估。jMetalPy包含了一个基于Apache Spark的evaluator，所以解可以按照[52]中给出的方法在不同的并行系统中评估（多核，集群）。这个evaluator可以以下面的方式使用：

```
# NSGAII runner using the Spark evaluator
algorithm = NSGAII(
...
evaluator=SparkEvaluator()
)
```

The resulting parallel NSGA-II algoritm combines parallel with sequential phases, so speed improvements cannot be expected to scale linearly. A pilot test on our target laptop indicates speedup factors in the order of 2.7. However, what is interesting to note here is that no changes are required in NSGA-II, which has the same behavior as its sequential version, so the obtained time reductions are for free.

得到的并行NSGA-II算法将并行和顺序阶段进行了结合，所以速度改进不会是线性缩放的。在我们的笔记本上的小型试验表明，加速因子可以达到2.7。但是，要注意的有趣的地方在于，不需要对NSGA-II进行改变，与顺序版本的行为相同，所以得到的时间缩减是免费的。

### 4.6. Distributed NSGA-II with Dask

The last variant of NSGA-II we present in this paper is a distributed version based on an asynchronous parallel model implemented with Dask [12], a parallel and distributed Python system including a broad set of parallel programming models, including asynchronous parallelism using futures.

我们在本文中给出的NSGA-II的最后的变体，是一个分布式版本，基于用Dask实现的异步并行模型。Dask是一个并行分布式Python系统，包括广泛的并行编程模型，包括异步并行。

The distributed NSGA-II adopts a parallel scheme studied in [50]. The scheme is based on a steady-state NSGA-II and the use of Dask’s futures, in such a way that whenever a new solution has to evaluated, a task is created and submitted to Dask, which returns a future. When a task is completed, its corresponding future returns an evaluated solution, which is inserted into the offspring population. Then, a new solution is produced after performing the replacement, selection, and reproduction stages, to be sent again for evaluation. This way, all the processors/cores of the target cluster will be busy most of the time.

分布式NSGA-II采用了[50]中采用的并行方案。这个方案是基于稳态NSGA-II的，使用Dask的futures，不论什么时候需要评估一个新的解，就创建一个任务，然后提交到Dask中，然后返回一个future。当一个任务完成时，其对应的future返回一个评估的解，插入到子代种群中。然后，一个新的解在进行了replacement，selection和reproduction阶段后，就会产生，然后再送到评估。这样，目标集群的所有处理器/核都会在大部分时间内是忙的。

Preliminary results on our target multicore laptop indicate that speedups around 5.45 can obtained with the 8 cores of the system where simulations were performed. We will discuss on this lack of scalability and other aspects of this use case in the next subsection.

在我们的多核笔记本上的初步结果表明，在8核的系统上，可以达到大约5.45倍的加速。我们在下一节中会讨论缺少扩展性和其他的方面。

### 4.7. Discussion

In this section we have presented five different versions of NSGA-II, most of them (except for the distributed variant) requiring minor changes on the base class implementing NSGA-II. Not all algorithms can be adapted in the same way, but some of the variations of NSGA-II can be implemented in a straightforward manner. Thus, we include in jMetalPy examples of dynamic, preference-based, and parallel versions of some of the included algorithms, such as SMPSO, GDE3, and OMOPSO.

本节中，我们给出了NSGA-II的5个不同版本，多数都需要对实现NSGA-II的基础类进行略微的修改（除了分布式变体）。并不是所有的算法都可以以相同的方式进行修改，NSGA-II的一些变化可以以很直接的方式进行实现。因此，我们在jMetalPy的例子中包括了一些算法的动态版本，基于preference的版本，和并行版本，比如SMPSO，GDE3和OMOPSO。

We would like to again stress on the readability of the codes, by virtue of which all the steps of the algorithms can be clearly identified. Some users may find the class hierarchy EvolutionaryAlgorithm → GeneticAlgorithm → NSGAII cumbersome, and prefer to have all the code of NSGA-II in a single class. However, this alternative design approach would hinder the flexibility of the current implementation, and would require to replicate most of the code when developing algorithmic variants.

我们想再强调一下代码的可读性，算法的所有步骤都可以很清楚的看到。一些用户会发现类的层次关系EvolutionaryAlgorithm → GeneticAlgorithm → NSGAII比较繁琐，想让NSGA-II的所有的代码都在一个类中。但是，这种设计方法会妨碍当前实现的灵活性，在开发算法变体时，会需要复制多数代码。

In the case of parallel algorithms, an exhaustive performance assessment is beyond the scope of this paper. The reported speedups are not remarkable due to the Turbo Boost feature of the processor of the laptop used for performing the experiments, but they give an idea of the time reductions that can be achieved when using a modern multicore computer.

在并行算法的情况下，彻底的性能评估不在本文的范围内。给出的加速结果并没有很显著，因为笔记本处理器的Turbo Boost特点，但给出了这个思想，当使用现代多核计算机时，可以得到时间降低的效果。

## 5. Visualization

An advantage of using Python (instead of Java) is its power related to visualization features thanks to the availability of graphic plotting libraries, such as Matplotlib or Plotly.

使用Python的一个好处，是有关可视化的能力，因为可以使用画图库比如Matplotlib或Plotly。

jMetalPy harnesses these libraries to include three types of visualization charts: static, interactive and streaming. Table 2 summarizes these implementations. Static charts can be shown in the screen, stored in a file, or included in a Jupyter notebook (typically used at the end of the execution of an algorithm). Similarly, interactive charts are generated when an algorithm returns a Pareto front approximation but, unlike the static ones, the user can manipulate them interactively. There are two kinds of interactive charts: those that produce an HTML page including a chart (allowing to apply actions such as zooming, selecting part of the graph, or clicking in a point to see its objective values are allowed) and charts such as the Chord diagram that allows hovering the mouse over the chart and visualizing relationships among objective values. Finally, streaming charts depict graphs in real time, during the execution of the algorithms (and they can also be included in a Jupyter notebook); this can be useful to observe the evolution of the current Pareto front approximation produced by the algorithm.

jMetalPy采用了这些库，有三种可视化图的能力：静态，互动的和流式的。表2总结了这些实现。静态图可以在显示器上展示，存储在文件中，或包含在Jupyter notebook中（一般在算法执行的最后使用）。类似的，互动式图的生成，是在算法返回了一个Pareto front近似，但是，与静态图不同，用户可以互动的进行操作。有两种互动图：产生了一个HTML网页，包含一个图（可以进行放大，选择一部分，或点击一个点查看其目标值），和像Chord图这样的图，允许鼠标在滑过图时，看到目标值之间的关系。最后，流式图以实时的方式展示图，在算法执行的过程中（也可以包含在Jupyter notebook中）；在观察算法产生的Pareto front的演化时，会非常有用。

Figure 5 shows three examples of interactive plots based on Plotly. The target problem is DTLZ1 [26], which is solved with the SMPSO algorithm when the problem is defined with 2, 3 and 5 objectives. For any problem with more than 3 objectives, a parallel coordinates graph is generated. An example of Chord diagram for a problem with 5 objectives is shown in Figure 6; each depicted chord represents a solution of the obtained Pareto front, and ties together its objective values. When hovering over a sector box of a certain objective fi, this chart only renders those solutions whose fi values fall within the value support of this objective delimited by the extremes of the sector box. Finally, the outer partitioned torus of the chart represents a histogram of the values covered in the obtained Pareto front for every objective.

图5展示了基于Plotly的互动式图的3个例子。目标问题是DTLZ1，采用SMPSO算法求解，问题分别为2，3和5个维度。对于任意多余3个目标的问题，会生成一个并行坐标图。5个目标的一个问题的Chord图的例子，如图6所示；每个展示的chord，表示得到的Pareto front的解，与其目标值绑定在一起。当鼠标滑过一部分特定的目标fi，这个图只会渲染出其fi值在特定范围内的解。最后，图中外部分割的环面，表示对每个目标得到的Pareto front的值的直方图。

## 6. Experimental Use Case

In previous sections, we have shown examples of Pareto front approximations produced by some of the metaheuristics included in jMetalPy. In this section, we describe how our framework can be used to carry out rigorous experimental studies based on comparing a number of algorithms to determine which of them presents the best overall performance.

在前面的节中，我们展示了jMetalPy中某个元启发式产生的Pareto front近似的例子。本节中，我们描述一下，我们的框架怎样用在进行严格的试验研究中，比较几种算法，以决定哪个会给出最佳的总体性能。

### 6.1. Experimentation Methodology

An experimental comparison requires a number of steps: 一个试验比较需要几个步骤：

1. Determine the algorithms to be compared and the benchmark problems to be used. 决定要比较的算法，和使用的基准测试问题。

2. Run a number of independent runs per algorithm-problem configuration and get the produced fronts. 对每个算法-问题配置，进行几次独立的运行，以得到产生的fronts。

3. Apply quality indicators to the fronts (e.g., Hypervolume, Epsilon, etc.). 对fronts应用质量indicators（如，Hypervolume，Epsilon，等）。

4. Apply a number of statistical test to assess the statistical significance of the performance differences found among the algorithms considered in the benchmark. 在算法对基准测试得到的结果中，进行几种统计测试，以评估性能差异的统计学显著性。

The first three steps can be done with jMetalPy, but also with jMetal or even manually (e.g., running algorithms using a script). The point where jMetalPy stands out is the fourth one, as it contains a large amount of statistical features to provide the user with a broad set of tools to analyze the results generated by a comparative study. All these functionalities have been programmed from scratch and embedded into the core of jMetalPy. Specifically, the statistical tests included in jMetalPy are listed next:

前三个步骤可以用jMetalPy进行，也可以用jMetal或手工进行（比如，用脚本来运行算法）。jMetalPy独特的地方在于第四点，因为其包含大量统计特征工具，可以分析比较性研究得到的结果。所有这些功能都是从头编程的，嵌入到jMetalPy的核心功能中。具体的，jMetalPy中包含的统计测试功能如下：

• A diverse set of non-parametric null hypothesis significance tests, namely, the Wilcoxon rank sum test, Sign test, Friedman test, Friedman aligned rank test and Quade test. These tests have been traditionally used by the community to shed light on their comparative performance by inspecting a statistic computed from their scores.

多种非参数null假设显著性测试，即，Wilcoxon rank sum test, Sign test, Friedman test, Friedman aligned rank test和Quade test。这些测试的使用是很多的，通过检查从分数中计算得到的统计值，来进行比较性性能研究。

• Bayesian tests (sign test and signed rank test), which have been recently postulated to overcome the shortcomings of null hypothesis significance testing for performance assessment [53]. These tests are complemented by a posterior plot in barycentric coordinates to compare pairs of algorithms under a Bayesian approach by also accounting for possible statistical ties.

贝叶斯测试（sign test和signed rank test），最近提出其可能在性能评估中克服null hypothesis significance testing的缺点。这些测试还有后期的图作为补充。

• Posthoc tests to compare among multiple algorithms, either one-vs-all (Bonferroni-Dunn, Holland, Finner, and Hochberg) or all-vs-all (Li, Holm, Shaffer).

在比较多个算法中的事后测试，可以是one-vs-all或all-vs-all。

The results of these tests are displayed by default in the screen and most of them can be exported to LATEX tables. Furthermore, boxplot diagrams can be also generated. Finally, LATEX tables containing means and medians (and their corresponding standard deviation and interquartile range dispersion measures, respectively) are automatically generated.

这些测试的结果默认是展示在屏幕上的，多数可以导出到LATEX表格中。而且，也可以产生boxplot图。最后，包含均值和中值的LATEX表可以自动生成（以及对应的标准差，和interquartile range dispersion度量）。

### 6.2. Implementation Details

jMetalPy has a laboratory module containing utilities for defining experiments, which require three lists: the algorithms to be compared (which must be properly configured), the benchmark problems to be solved, and the quality indicators to be applied for performance assessment. Additional parameters are the number of independent runs and the output directory.

jMetalPy有一个实验室模块，包含了定义试验的工具，这需要三个列表：要比较的算法（这必须合理的进行配置），要解决的基准测试问题，要对性能评估应用的quality indicators。另外的参数包括，独立运行的数量，和输出目录。

Once the experiment is executed, a summary in the form of a CSV file is generated. This file contains all the information of the quality indicator values, for each configuration and run. Each line of this file has the following schema: Algorithm,Problem, Indicator, ExecutionId, IndicatorValue. An example of its contents follows:

一旦试验执行，会生成CSV格式的摘要。这个文件包括quality indicator值的所有信息，每种配置和运行的都有。这个文件的每一行都有如下格式：Algorithm，Problem，Indicator，执行ID，Indicator值。一个例子如下：

```
Algorithm,Problem,Indicator,ExecutionId,IndicatorValue
NSGAII,ZDT1,EP,0,0.015705992620067832
NSGAII,ZDT1,EP,1,0.012832504015918067
NSGAII,ZDT1,EP,2,0.01071189935186434
...
MOCell,ZDT6,IGD+,22,0.0047265135903854704
MOCell,ZDT6,IGD+,23,0.004496215669027173
MOCell,ZDT6,IGD+,24,0.005483899232523609
```

where we can see the header with the column names, followed by four lines corresponding to the values of the Epsilon indicator of three runs of the NSGA-II algorithm when solving the ZDT1 problem. The end of the file shows the value of the IGD+ indicator for three runs of MOCell when solving the ZDT6 problem. The file contains as many lines as the product of the numbers of algorithms, problems, quality indicators, and independent runs.

在这里我们可以看到，表头是列的名字，下面4行是用NSGA-II算法求解ZDT1问题时3次运行的Epsilon indicator的值。文件最后展示了MOCell算法在求解ZDT6问题时3次运行的IGD+ indicator的值。文件的行数应当等于，算法的数量，问题的数量，quality indicator的数量和独立运行次数的乘积。

The summary file is the input of all the statistical tests, so that they can be applied to any valid file having the proper format. This is particularly interesting to combine jMetal and jMetalPy. The last versions of jMetal generates a summary file after running a set of algorithms in an experimental study, so then we can take advantage of the features of jMetal (providing many algorithms and benchmark problems, faster execution of Java compared with Python) and jMetalPy (better support for data analysis and visualization). We detail an example of combining both frameworks in the next section.

摘要文件是所有统计测试的输入，可以应用到任意有合理格式的有效文件。这在将jMetal和jMetalPy结合到一起时就特别有趣。jMetal的最新版本，在一个试验研究中运行了一些算法后会生成一个摘要文件，所以我们可以利用jMetal的特点（提供了很多算法和基准测试问题，与Python比起来，Java运行速度比较快），和jMetalPy的特点（更好的支持数据分析和可视化）。我们在下一节给出一个将两个框架结合到一起的例子。

### 6.3. Experimental Case Study

Let us consider the following case study. We are interested in assessing the performance of five metaheuristics (GDE3, MOCell, MOEA/D, NSGA-II, and SMPSO) when solving the ZDT suite of continuous problems (ZDT1-4, ZDT6). The quality indicators to calculate are set to the additive Epsilon (EP), Spread (SPREAD), and Hypervolume (HV), which give a measure of convergence, diversity and both properties, respectively. The number of independent runs for every algorithm is set to 25.

让我们考虑下面的案例。我们想要评估5个元启发式(GDE3, MOCell, MOEA/D, NSGA-II, SMPSO)在求解ZDT连续问题包(ZDT1-4, ZDT6)时的性能。要计算的quality indicators包括EP, SPREAD和HV，分别给出了收敛性，多样性和这两种性质的度量。对每个算法的独立运行次数设为25。

We configure an experiment with this information in jMetal and, after running the algorithms and applying the quality indicators, the summary file is obtained. Then, by giving this file as input to the jMetalPy statistical analysis module, we obtain a set of LATEX files and figures in an output directory as well as information displayed in the screen. We analyze next the obtained results.

我们在jMetal中用这些信息配置了一个试验，在运行了算法，应用了quality indicators后，得到了摘要文件。然后，将这个文件输入到jMetalPy统计分析模块，我们在输出文件夹中得到了一系列LATEX文件和图，在屏幕上也展示了信息。下面我们分析了得到的结果。

Tables 3, 4, and 5 show the median and interquartile range of the three selected quality indicators. To facilitate the analysis of the tables, some cells have a grey background. Two grey levels are used, dark and light, to highlight the algorithms yielding the best and second best indicator values, respectively (note that this is automatically performed by jMetalPy). From the tables, we can observe that SMPSO is the overall best performing algorithm, achieving the best indicator values in four problems and one second best value.

表3，4，5展示了3个选择的quality indicators的中值和interquartile range。为方便表格的分析，一些单元有灰色的背景。使用了两个灰度等级，暗的和亮的，以强调算法产生的最佳和第二佳的indicator值（注意，这是jMetalPy自动生成的）。从这个表格中，我们可以观察到，SMPSO是总体上表现最好的算法，在4个问题中获得了最佳indicator值和一个第二佳值。

Nevertheless, it is well known that taking into account only median values for algorithm ranking does not ensure that their differences are statistically significant. Statistical rankings are also needed if we intend to rank the algorithm performance considering all the problems globally. Finally, in studies involving a large number of problems (we have used only five for simplicity), the visual analysis of the medians can be very complicated, so statistical diagrams gathering all the information are needed. This is the reason why jMetalPy can also generate a second set of LATEX tables compiling, in a visually intuitive fashion, the result of non-parametric null hypothesis significance tests run over a certain quality indicator for all algorithms. Tables 6, 7 and 8 are three examples of these tables computed by using the Wilcoxon rank sum test between every pair of algorithms (at the 5% level of significance) for the EP, SPREAD and HV indicators, respectively. In each cell, results for each of the 5 datasets are represented by using three symbols: – if there is not statistical significance between the algorithms represented by the row and column of the cell; ∇ if the approach labeling the column is statistically better than the algorithm in the row; and ∆ if the algorithm in the row significantly outperforms the approach in the column.

尽管如此，大家都知道，只考虑算法排序的中值，不能确保其差异是统计上显著的。如果我们想要考虑所有问题对算法性能排序，就还需要进行统计排序。最后，在涉及到很多问题的研究中（我们只使用了5个），中值的视觉分析就会非常复杂了，所以就需要收集了所有信息的统计图表。这就是为什么jMetalPy为什么可以生成另一个LATEX表格汇编的原因，这是视觉上很直观的方式，是对某一特定的quality indicator对所有算法的非参数null hypothesis significance测试。表6，7，8是这些表格的3个例子，计算的是EP，SPREAD和HV indicators的每一对算法（在5%显著性的级别上），使用Wilcoxon rank sum测试。在每个单元中，这5个数据集每个的结果用3个符号进行表示：–表示对这个单元的行和列上的算法之间，没有统计上的显著性；∇表示列上的比行上的算法统计上要更好；∆表示行上的算法要比列上的算法统计上要更好。

The conclusions drawn from the above tables can be buttressed by inspecting the distribution of the quality indicator values obtained by the algorithms. Figure 7 shows boxplots obtained with jMetalPy by means of the Hypervolume values when solving the ZDT6 problem. Whenever the boxes do not overlap with each other we can state that there should be statistical confidence that the performance gaps are relevant (e.g., between SMPSO and the rest of algorithms), but when they do (as we can see with GDE3 and MOCell) we cannot discern which algorithm performs best.

通过上面的表格可以得出结论，可以通过检查算法得到的quality indicator值的分布，进行支持。图7展示了jMetalPy在求解ZDT6问题时的Hypervolume值得到的boxplot结果。只要这些boxes没有互相重叠，我们就可以说，性能差距是相关的，是具有统计置信度的（如，在SMPSO和其他算法之间），但在有重叠时（如GDE3和MOCell之间），我们就无法区分哪个算法是更好的。

The boxplots and tables described heretofore allow observing the dispersion of the results, as well as the presence of outliers, but they do not allow to get a global vision of the performance of the algorithms in all the problems. This motivates the incorporation of principled methods for comparing multiple techniques over different problem instances, such as those proposed by Demsar [54] in the context of classification problems and machine learning models. As anticipated previously, our developed framework includes Critical Distance (CD) plots (Figure 8, computed for the HV indicator) and Posterior plots (Figure 9, again for the HV indicator). The former plot is used to depict the average ranking of the algorithms computed over the considered problems, so that the chart connects with a bold line those algorithms whose difference in ranks is less than the so-called critical distance. This critical distance is a function of the number of problems, the number of techniques under comparison, and a critical value that results from a Studentized range statistic and a specified confidence level. As shown in Figure 8, SMPSO, MOCell, GDE3 and NSGA-II are reported to perform statistically equivalently, which clashes with the conclusions of the previously discussed table set due to the relatively higher strictness of the statistic from which the critical distance is computed. A higher number of problems would be required to declare statistical significance under this comparison approach.

迄今为止描述的boxplots和表格，可以观察结果的分散程度，以及离群点的存在，但是不能观察到算法在所有问题中的总体性能。这促使我们结合比较多种技术在不同的问题实例中的方法，比如[54]在比较分类问题和机器学习模型的方法。我们开发的框架包括关键距离(Critical Distance, CD)图（图8，计算的是HV indicator），和后验图（图9，还是HV indicator）。前者用于展示算法在所有问题上的表现的平均排序，图中用粗线连接的算法，是距离小于所谓的关键距离的算法。关键距离是问题数量，比较的技术的数量，和一个关键值的函数，这个关键值是从studentized范围统计和指定的置信度水平的结果。如图8所示，SMPSO，MOCell，GDE3和NSGA-II的表现在统计上是等价的，这与之前的表格得到的结论就有冲突了，其原因是之前计算的关键距离得到的统计值较为严格。需要更多数量的问题，才能在这种比较方法中声明统计显著性。

Finally, we end up our experimental use case by showing the Posterior plot that allows comparing pair of algorithms by using Bayesian sign test (Figure 9). When relying on Bayesian hypothesis testing we can directly evaluate the posterior probability of the hypotheses from the available quality indicator values, which enables a more intuitive understanding of the comparative performance of the considered algorithms. Furthermore, a region of practical equivalence (also denoted rope) can be defined to account for ties between the considered multiobjective solvers. The plot in Figure 9 is in essence a barycentric projection of the posterior probabilities: the region at the bottom-right of the chart, for instance, delimits the area where:

最后，我们展示一下试验案例中的后验图，可以使用贝叶斯sign测试来比较算法对（图9）。当依赖于贝叶斯假设检验时，我们可以从可用的quality indicator值中，直接评估假设的后验概率，这可以更直观的理解算法的比较性能。而且，可以定义一个实际等价性的区域（也称为rope），来说明考虑的多目标求解器之间的联系。图9中的图实际上是后验概率的重心投影：比如，图中的右下的区域，实际上是区域

$$θr ≥ max(θe, θl)$$(1)

with θr = P(z > r), θe = P(−r ≤ z ≤ r), θl = P(z < −r), and z denoting the difference between the indicator values of the algorithm on the right and the left (in that order). Based on this notation, the figure exposes, in our case and for r = 0.002, than in most cases z = HV(NSGA-II) − HV(SMPSO) fulfills θl ≥ max(θe, θr), i.e. it is more probable, on average, than the HV values of SMPSO are higher than those of NSGA-II. Particularly these probabilities can be estimated by counting the number of points that fall in every one of the three regions, from which we conclude that in this use case 1) SMPSO is practically better than NSGA-II with probability 0.918; 2) both algorithms perform equivalently with probability 0.021; and 3) NSGA-II is superior than SMPSO with probability 0.061.

有θr = P(z > r), θe = P(−r ≤ z ≤ r), θl = P(z < −r)，z表示左边和右边算法的indicator值的差异。基于这种表示，这个图说明，在我们的案例中，对于r=0.002，比在多数情况中，z = HV(NSGA-II) − HV(SMPSO)满足θl ≥ max(θe, θr)，即，平均来说更可能的是，SMPSO的HV值比NSGA-II的HV值要更高。特别是这些概率可以通过数一下点落在这三个区域中的数量来估计，在这个使用案例中，我们可以得出结论，1)SMPSO比NSGA-II要更好，概率0.918；2)两种算法表现等价的概率为0.021；3)NSGA-II比SMPSO要好的概率为0.061。

## 7. Conclusions and Future Work

In this paper we have presented jMetalPy, a Python-based framework for multi-objective optimization with metaheuristics. It is released under the MIT license and made freely available for the community in GitHub. We have detailed its core architecture and described the implementation of NSGA-II and some of its variants as illustrative examples of how to operate with this framework.

本文中，我们给出了jMetalPy，一个基于Python的元启发式多目标优化框架，代码已开源。我们详述了其核心架构，描述了NSGA-II及其一些变体的实现。

jMetalPy provides support for dynamic optimization, parallelism, and decision making. Other salient features involves visualization (static, streaming, and interactive graphics) for multi- and many-objective problems, and a large set of statistical tests for performance assessment. It is worth noting that jMetalPy is still a young research project, which is intended to be useful for the research community interested in multiobjective optimization with metaheuristics. Thus, it is expected to evolve quickly, incorporating new algorithms and problems by both the development team and by external contributors.

jMetalPy为动态优化，并行化和决策提供了支持。其他特征包括多目标和众目标问题的可视化，性能评估的统计测试功能集合。值得说明的是，jMetalPy仍然是一个年轻的研究项目，想要为元启发式多目标优化做出一些贡献。因此，期待可以迅速演进，通过开发团队和外部贡献者结合新的算法和问题。

Specific lines of future work includes evaluating the performance of parallel and distributed metaheuristics in clusters, as well as applying them to solve new real-world problems. Furthermore, we plan to open the analysis and visualization features of jMetalPy by including different input data formats to foster their use by other external frameworks.

未来工作包括在集群中评估并行和分布式元启发式的性能，以及将其应用以求解新的真实世界的问题。而且，我们计划开放jMetalPy的分析和可视化特征，包含不同的输入数据格式，促进其他外部框架的使用。