# Practical Design Space Exploration

Luigi Nardi et. al. @ Stanford University

## 0. Abstract

Multi-objective optimization is a crucial matter in computer systems design space exploration because real-world applications often rely on a trade-off between several objectives. Derivatives are usually not available or impractical to compute and the feasibility of an experiment can not always be determined in advance. These problems are particularly difficult when the feasible region is relatively small, and it may be prohibitive to even find a feasible experiment, let alone an optimal one.

多目标优化是计算机系统设计空间探索的一个关键问题，因为真实世界的应用通常依赖于多个目标的折中。导数通常是不可用的，或计算起来不实际的，试验的可行性并不一定能提前确定。这些问题在可行区域相对较小时，就尤其困难，甚至不能找到一个可行的试验，更不要说一个最优的试验。

We introduce a new methodology and corresponding software framework, HyperMapper 2.0, which handles multi-objective optimization, unknown feasibility constraints, and categorical/ordinal variables. This new methodology also supports injection of the user prior knowledge in the search when available. All of these features are common requirements in computer systems but rarely exposed in existing design space exploration systems. The proposed methodology follows a white-box model which is simple to understand and interpret (unlike, for example, neural networks) and can be used by the user to better understand the results of the automatic search.

我们提出了一种新的方法论，和对应的软件框架，名为HyperMapper 2.0，处理多目标优化问题，未知的可行性约束和类别/次序变量。这种新方法论还支持将用户先验知识注入到搜索中。所有这些特征都是计算机系统中的常见要求，但并没有暴露在现有的设计空间探索系统中。提出的方法论是一个白盒模型，理解和解释起来都很简单（比如，与神经网络不同），可以被用户使用，以更好的理解自动搜索的结果。

We apply and evaluate the new methodology to the automatic static tuning of hardware accelerators within the recently introduced Spatial programming language, with minimization of design run-time and compute logic under the constraint of the design fitting in a target field-programmable gate array chip. Our results show that HyperMapper 2.0 provides better Pareto fronts compared to state-of-the-art baselines, with better or competitive hypervolume indicator and with 8x improvement in sampling budget for most of the benchmarks explored.

我们将这个方法论，在最近提出的Spatial编程语言中，在硬件加速器的自动静态调节进行应用和评估，在设计适合于目标FPGA芯片的约束下，最小化了设计运行时间和计算逻辑。我们的结果表明，HyperMapper 2.0与目前最好的基准相比，得到了更好的Pareto fronts，有更好的接近的hypervolume指示器，在多数探索过的基准测试中，其采样预算的改进达到8x。

**Index Terms** — Pareto-optimal front, Design space exploration, Hardware design, Performance modeling, Optimizing compilers, Machine learning driven optimization

## 1. Introduction

Design problems are ubiquitous in scientific and industrial achievements. Scientists design experiments to gain insights into physical and social phenomena, and engineers design machines to execute tasks more efficiently. These design problems are fraught with choices which are often complex and high-dimensional and which include interactions that make them difficult for individuals to reason about. In software/hardware co-design, for example, companies develop libraries with tens or hundreds of free choices and parameters that interact in complex ways. In fact, the level of complexity is often so high that it becomes impossible to find domain experts capable of tuning these libraries [10].

设计问题在科学和工业活动中普遍存在。科学家设计试验得到对物理和社会现象的洞见，工程师设计机器来更高效的执行任务。这些设计问题充满了选择，通常是复杂的，高维的，包括很多互动，使其对个人很难进行推理。比如，在软件/硬件协同设计中，公司开发出的库，有数十或上百个自由选项和参数，以复杂的方式来进行互动。实际上，复杂程度通常非常高，不可能找到领域专家能够调节这些库。

Typically, a human developer that wishes to tune a computer system will try some of the options and get an insight of the response surface of the software. They will start to fit a model in their head of how the software responds to the different choices. However, fitting a complex multi-objective function without the use of an automated system is a daunting task. When the response surface is complex, e.g. non-linear, non-convex, discontinuous, or multi-modal, a human designer will hardly be able to model this complex process, ultimately missing the opportunity of delivering high performance products.

典型的，一个希望调节一个计算机系统的人类开发者，会尝试一些选项，得到软件的响应表面的洞见。他们会在他们的脑袋中拟合一个模型，得到软件如何对不同的选项响应。但是，不用自动化系统，来拟合一个复杂的多目标函数，这是一个很恐怖的任务。当响应曲线很复杂时，如，非线性，非凸，不连续，或多模，一个人类设计者就很难对这个复杂的过程建模，最终不能给出高性能的产品。

Mathematically, in the mono-objective formulation, we consider the problem of finding a global minimizer of an unknown (black-box) objective function f:

数学上来说，在单目标的表述下，我们考虑的问题是，对未知（黑盒）的目标函数f，找到一个全局最小化器：

$$x^* = argmin_{x∈X} f(x)$$(1)

where X is some input decision space of interest (also called design space). The problem addressed in this paper is the optimization of a deterministic function f : X → R over a domain of interest that includes lower and upper bound constraints on the problem variables.

其中X是某个感兴趣的输入决策空间（也称为设计空间）。本文处理的问题，是一个确定性函数f : X → R在一个感兴趣领域的优化，这个领域包含了对问题变量上限和下限的约束。

When optimizing a smooth function, it is well known that useful information is contained in the function gradient/derivatives which can be leveraged, for instance, by first order methods. The derivatives are often computed by hand-coding, by automatic differentiation, or by finite differences. However, there are situations where such first-order information is not available or even not well defined. Typically, this is the case for computer systems workloads that include many discrete variables, i.e., either categorical (e.g., boolean) or ordinal (e.g., choice of cache sizes), over which derivatives cannot even be defined. Hence, we assume in our applications of interest that the derivative of f is neither symbolically nor numerically available. This problem is referred to in the literature as DFO [10], [24], also known as black-box optimization [12] and, in the computer systems community, as design space exploration (DSE) [16], [17].

当优化一个平滑函数时，大家都知道，有用的信息是包含在函数的梯度/导数中的，可以被利用，比如，通过一阶方法。导数通常是通过手工编码，通过自动微分，或通过有限差分来计算得到的。但是，有一些情况，这些一阶信息是不可用的，甚至是没有定义的。计算机系统workloads就是这种典型的情况，因为包含了很多离散变量，即，要么是类别的（如，boolean），或序数的（如，cache大小的选择），在这些变量上，导数是无法定义的。因此，我们假设，在我们感兴趣的应用中，f的导数在符号上或数值上都是不可用的。这种问题在文献中称为DFO，也称为黑盒优化，在计算机系统研究圈中，称为设计空间优化。

In addition to objective function evaluations, many optimization programs have similarly expensive evaluations of constraint functions. The set of points where such constraints are satisfied is referred to as the feasibility set. For example, in computer micro-architecture, fine-tuning the particular specifications of a CPU (e.g., L1-Cache size, branch predictor range, and cycle time) need to be carefully balanced to optimize CPU speed, while keeping the power usage strictly within a pre-specified budget. A similar example is in creating hardware designs for field-programmable gate arrays (FPGAs). FPGAs are a type of reconfigurable logic chip with a fixed number of units available to implement circuits. Any generated design must keep the number of units strictly below this resource budget to be implementable on the FPGA. In these examples, feasibility of an experiment cannot be checked prior to termination of the experiment; this is often referred to as unknown feasibility in the literature [14]. Also note that the smaller the feasible region, the harder it is to check if an experiment is feasible (and even more costly to check optimality [14]).

除了目标函数评估，很多优化程序对约束函数的评估的代价也是非常的昂贵。这样的约束被满足的点集，被称为可行集。比如，在计算机微架构中，精调CPU的特殊指标（如，L1-Cache大小，分支预测器范围，和cycle时间）需要进行仔细均衡，以优化CPU速度，同时保持功耗使用严格在预先指定的预算内。一个类似的例子是，创建硬件设计的FPGA。FPGAs是一类可配置的逻辑芯片，用于实现的电路的数量固定。任何生成的设计必须保持单元数量严格小于资源预算之下，即FPGA上可实现的数量。在这些例子中，一个试验的可行性不能在试验完成前被核验；这在文献中被称为未知可行性。还要注意，可行区域越小，越难检查一个试验是否是可行的（要检查最优性，代价更大）。

While the growing demand for sophisticated DFO methods has triggered the development of a wide range of approaches and frameworks, none to date are featured enough to fully address the complexities of design space exploration and optimization in the computer systems domain. To address this problem, we introduce a new methodology and a framework dubbed HyperMapper 2.0. HyperMapper 2.0 is designed for the computer systems community and can handle design spaces consisting of multiple objectives and categorical/ordinal variables. Emphasis is on exploiting user prior knowledge via modeling of the design space parameters distributions. Given the years of hand-tuning experience in optimizing hardware, designers bear a high level of confidence. HyperMapper 2.0 gives means to inject knowledge in the search algorithm. This is achieved by introducing for the first time the use of a Beta distribution for modeling the user belief, i.e., prior knowledge, on how each parameter of the design space influences a response surface. In addition, bearing in mind the feasibility constraints that are common in computer systems workloads we introduce for the first time a model that considers unknown constraints, which is, constraints that are only known after evaluating a system configuration. To aid comparison, we provide a list of existing tools and the corresponding taxonomy in Table I. Our framework uses a model-based algorithm, i.e., construct and utilize a surrogate model of f to guide the search process. A key advantage of having a model, and more specifically a white-box model, is that the final surrogate of f can be analyzed by the user to understand the space and learn fundamental properties of the application of interest.

对复杂的DFO的需求的增长，促进了很多方法和框架的开发，但目前尚没有哪个能够完全处理计算机系统领域的设计空间探索和优化的复杂度。为处理这个问题，我们提出了一种新的方法论和框架，称为HyperMapper 2.0。HyperMapper 2.0是为计算机系统设计，可以处理包含多个目标函数和类别/序数变量的设计空间。强调了通过设计空间参数分布建模，来利用用户先验知识。在优化硬件数年的手工调节的经验上，设计者拥有很高的信息。HyperMapper 2.0给出了将知识注入到搜索算法的方法。第一次使用了一个Beta分布来建模用户对每个设计空间的参数怎样影响响应表面的信念，即，先验知识，来得到这样的功能。另外，还要记得，在计算机系统workloads中常见的可行性约束，我们第一次引入了一个模型，考虑了未知的约束，即，只有在评估了一个系统配置之后，才能知道的约束。为帮助比较，我们在表I中给出了现有工具的列表，和对应的taxonomy。我们的框架使用了一种基于模型的算法，即，构建和利用了f的一个代理模型，来引导搜索过程。。有一个模型，更具体的是一个白盒模型，的关键优势，是f的最终代理可以被用户分析来理解感兴趣应用的空间和学习到其的根本性质。

As shown in Table I, HyperMapper 2.0 is the only framework to provide all the features needed for a practical design space exploration software in computer systems applications. The contributions of this paper are:

如表I所示，HyperMapper 2.0是为计算机系统应用提供了具有所有特征的实际的设计空间探索软件的框架。本文的贡献如下：

- A methodology for multi-objective optimization that deals with categorical and ordinal variables, unknown constraints, and exploitation of the user prior knowledge. 一种多目标优化的方法学，可以处理类别变量，序号变量，未知的约束，利用用户的先验知识。

- An integration and experimental results of our methodology in a full, production-level compiler toolchain for hardware accelerator design. 我们的方法论的集成试验结果，用于硬件加速器设计的完整的产品级的编译器工具链。

- A framework dubbed HyperMapper 2.0 implementing the newly introduced methodology, designed to be simple, user-friendly and application independent. 一个名为HyperMapper 2.0的框架，实现了新引入的方法学，其设计很简单，用户友好，对应用独立。

The remainder of this paper is organized as follows: Section II provides the problem statement and background. In Section III, we describe our methodology and framework. In Section IV we present our experimental evaluation. Section V discusses related work. We conclude in Section VI with a brief discussion of future work.

本文剩余部分组织如下：第II部分给出了问题表述和背景。在第III部分，我们描述了我们的方法学和框架。在第IV部分，我们给出了我们的试验评估。在第V部分，讨论了相关的工作。在第VI部分，给出了结论，并简要讨论了相关工作。

## 2. Background

In this section, we provide the notation and basic concepts used in the paper. We describe the mathematical formulation of the mono-objective optimization problem with feasibility constraints. We then expand this to a definition of the multi-objective optimization problem and provide background on randomized decision forests [8].

本节中，我们给出了本文中使用的符号和基本概念。我们描述了带有可行性约束的单目标优化问题的数学表述。然后我们将其拓展到多目标优化问题，给出了随机决策森林的背景。

### 2.1. Mono-objective Optimization with Unknown Feasibility Constraints

Mathematically, in the mono-objective formulation, we consider the problem of finding a global minimizer (or maximizer) of an unknown (black-box) objective function f under a set of constraint functions c_i:

数学上，在单目标表述中，我们考虑，找到一个未知目标函数f，在一系列约束函数ci下的全局极小值的问题：

$$x^* = argmin_{x∈X} f(x), subject to c_i(x) ≤ b_i, i = 1, . . . , q$$

where X is some input design space of interest and ci are q unknown constraint functions. The problem addressed in this paper is the optimization of a deterministic function f : X → R over a domain of interest that includes lower and upper bounds on the problem variables.

其中X是某个输入感兴趣设计空间，ci是q个未知的约束函数。本文处理的问题是，一个确定性函数f : X → R在一个感兴趣领域中的优化问题，这个领域对问题变量包含下限和上限。

The variables defining the space X can be real (continuous), integer, ordinal, and categorical. Ordinal parameters have a domain of a finite set of values which are either integer and/or real values. For example, the sets {1, 5, 8} and {3.4, 2.5, 6, 9.1} are possible domains of ordinal parameters. Ordinal values must have an ordering by the less-than operator. Ordinal and integer cases are also referred to as discrete variables. Categorical parameters also have domains of a finite set of values but have no ordering requirement. For example, sets of strings describing some property like {true, false} and {car, truck, motorbike} are categorical domains. The primary benefit of encoding a variable as an ordinal is that it can allow better inferences about unseen parameter values. With a categorical parameter, the knowledge of one value does not tell one much about other values, whereas with an ordinal value we would expect closer values (with respect to the ordering) to be more related.

定义了空间X的变量，可以是实数（连续的），整数，序数和类别的。序数参数是一个有限集的值，可以是整数和/或实数值。比如，集合{1, 5, 8}和{3.4, 2.5, 6, 9.1}都是可能的序数参数。序数值必须有排序，由小于算子进行排列。序数和整数的情况也被称为离散变量。类别参数也是有限集的值，但是没有排序的要求。比如，字符串的集合，描述了一些性质，如{true, false}和{car, truck, motorbike}，是类别的领域。将变量编码为序数的主要好处是，可以更好的推理没看到的参数值。有了类别参数，一个值的知识，并不会透露更多其他值的内容，而有了序号值，我们则可以期待更接近的值。

We assume that the derivative of f is not available, and that bounds, such as Lipschitz constants, for the derivative of f is also unavailable. Evaluating feasibility is often in the same order of expense as evaluating the objective function f. As for the objective function, no particular assumptions are made on the constraint functions.

我们假设f的导数是不可用的，因此f的导数的一些界限，如Lipschitz常数都是不可用的。评估可行性与评估目标函数f的代价通常是类似的。至于目标函数，对约束函数不作特殊的假设。

### 2.2. Multi-Objective Optimization: Problem Statement

A pictorial representation of a multi-objective problem is shown in Figure 1. On the left, a three-dimensional design space is composed by one ordinal (p1), one real (p2), and one categorical (p3) variable. The red dots represent samples from this search space. The multi-objective function f maps this input space to the output space on the right, also called the optimization space. The optimization space is composed by two optimization objectives (o1 and o2). The blue dots correspond to the red dots in the left via the application of f. The arrows Min and Max represent the fact that we can minimize or maximize the objectives as a function of the application. Optimization will drive the search of optima points towards either the Min or Max of the right plot.

图1是多目标问题的一个图示。在左边，这个三维设计空间，是由序数(p1)，实值(p2)和类别(p3)变量组成。红点表示这个搜索空间中的样本。多目标函数f将输入空间映射到右边的输出空间，也称为优化空间。优化空间由两个优化目标组成(o1和o2)。蓝色点通过f对应着左边的红色点。箭头Min和Max表示，我们可以将目标作为应用的函数最大化或最小化。优化会将极值点搜索驱赶到邮图的Min或Max处。

Formally, let us consider a multi-objective optimization (minimization) over a design space X ⊆ R^d. We define f : X → R^p as our vector of objective functions f = (f_1, . . . , f_p), taking x as input, and evaluating y = f(x). Our goal is to identify the Pareto frontier of f; that is, the set Γ ⊆ X of points which are not dominated by any other point, i.e., the maximally desirable x which cannot be optimized further for any single objective without making a trade-off. Formally, we consider the partial order in R^p: $y ≺ y' iff ∀i ∈ [p], y_i <= y'_i$ and $∃j, y_j < y'_j$, and define the induced order on X: x ≺ x' iff f(x) ≺ f(x'). The set of minimal points in this order is the Pareto-optimal set Γ = {$x ∈ X : \nexists x' such that x' ≺ x$}.

形式上，让我们考虑一个在设计空间X ⊆ R^d上的多目标优化（最小化）问题。我们定义f : X → R^p作为我们的目标函数向量f = (f_1, . . . , f_p)，以x为输入，计算的是y = f(x)。我们的目标是找到f的Pareto front；即，点集Γ ⊆ X，没有被其他点支配，即，最大期望的x，对任一目标函数的进一步优化，必然导致一个trade-off。形式上，我们考虑R^p中的partial order：$y ≺ y' iff ∀i ∈ [p], y_i <= y'_i$ 并且 $∃j, y_j < y'_j$，并定义在X上的induced order：x ≺ x' iff f(x) ≺ f(x')。Pareto-optimal集是以这个order的最小点集Γ = {$x ∈ X : \nexists x' such that x' ≺ x$}。

We can then introduce a set of inequality constraints c(x) = (c1(x), . . . , cq(x)), b = (b1, . . . , bq) to the optimization, such that we only consider points where all constraints are satisfied (c_i(x) <= b_i). These constraints directly correspond to real-world limitations of the design space under consideration. Applying these constraints gives the constrained Pareto

我们可以对优化引入不等式约束集c(x) = (c1(x), . . . , cq(x)), b = (b1, . . . , bq)，这样我们只考虑所有约束都满足的点(c_i(x) <= b_i)。这些约束直接对应着考虑的设计空间在真实世界的限制。将这些显示应用，给出约束的Pareto

$$Γ = \{ x ∈ X : ∀i <= q, c_i(x) <= b_i \}, where \nexists x' ∈ X such that c_i(x') <= b_i and x' ≺ x$$

Similarly to the mono-objective case in [13], we can define the feasibility indicator function ∆_i(x) ∈ 0, 1 which is 1 if c_i(x) <= b_i, and 0 otherwise. A design point where ∆_i(x) = 1 is termed feasible. Otherwise, it is called infeasible.

与[13]中的单目标的情况类似，我们可以定义可行性指示函数∆_i(x) ∈ 0, 1，如果c_i(x) <= b_i就是1，否则就是0。∆_i(x) = 1的设计点称之为可行，否则称之为不可行。

We aim to identify Γ with the fewest possible function evaluations, solving a sequential decision problem and constructing a strategy $\underline X : f → \{X1, X2, X3, . . . \}$ to iteratively generate the next X_n+1 ∈ X to evaluate. If the evaluation X_i is not very expensive then it is possible to construct a strategy that, for each sequential step, runs multiple evaluations, i.e., a batch of evaluations. In this case it is standard practice to warm-up the strategy with some previously sampled points, using sampling techniques from the design of experiments literature [26].

我们的目标是用最少可能的函数评估找出Γ，求解一个序列决策问题，构建一个策略$\underline X : f → \{X1, X2, X3, . . . \}$，来迭代的生成下一个X_n+1 ∈ X来进行评估。如果评估X_i并不是很昂贵，那么可能构建一个策略，对每个序列步骤，运行多个评估，即，一个批次的评估。在这种情况下，用一些之前采样过的点来预热这个策略，这是标准操作，使用的采样技术是从文献[26]的试验设计中得到的。

It is worth noting that, while infeasible points are never considered our best experiment, they are still useful to add to our set of performed experiments to improve the probabilistic model posteriors. Practically speaking, infeasible samples help to determine the shape and descent directions of c(x), allowing the probabilistic model to discern which regions are more likely to be feasible without actually sampling there. The fact that we do not need to sample in feasible regions to find them is a property that is highly useful in cases where the feasible region is relatively small, and uniform sampling would have difficulty finding these regions.

值得指出的是，不可行点从来不会被考虑为我们最好的试验，但它们仍然可以用于加入到我们的已经进行的试验的集合中，来改进概率模型后验。实践来说，不可行样本可以帮助确定c(x)的形状的下降方向，使概率模型能辨别出哪个区域更可能是可行的，而不用实际的去那里采样。我们不需要在可行区域中采样，以找到它们，这在可行区域相对较小的时候，非常有用，因为均匀采样会很难找到这些区域。

As an example, in this paper, we evaluate the compiler optimization case for targeting FPGAs. In this case, p = 2, q = 1, f_1(x) = Cycles(x) (number of total cycles, i.e., runtime), f_2(x) = Logic(x) (logic utilization, i.e., quantity of logic gates used) in percentage, and ∆_1(x) ∈ 0, 1 represents whether the design point x fits in the target FPGA board.

本文中的一个例子，我们对目标FPGAs评估编译器优化的情况。在这种情况下，p=2，q=1，f_1(x) = Cycles(x)（总计cycles数量，即，运行时间），f_2(x) = Logic(x)（逻辑使用率，即，使用的逻辑门的数量），∆_1(x) ∈ 0, 1表示这个设计点是否在目标FPGA板中。

### 2.3 Randomized Decision Forests

A decision tree is a non-parametric supervised machine learning method widely used to formalize decision making processes across a variety of fields. A randomized decision tree is an analogous machine learning model, which “learns” how to regress (or classify) data points based on randomly selected attributes of a set of training examples. The combination of many weak regressors (binary decisions) allows approximating highly non-linear and multi-modal functions with great accuracy. Randomized decision forests [8], [11] combine many such decorrelated trees based on the randomization at the level of training data points and attributes to yield an even more effective supervised regression and classification model.

一个决策树是一个非参数的有监督机器学习方法，广泛用于很多领域的形式化决策制定过程。随机决策树是一个类似的机器学习模型，学习怎样将数据点回归，基于训练样本集中随机选择的属性。很多弱回归器的组合可以以很高的准确率近似高度非线性的，多模态的函数。随机决策森林，将很多这样的去相关的树组合到一起，基于训练数据点和属性级别的随机化，以生成一个更有效的监督回归和分类模型。

A decision tree represents a recursive binary partitioning of the input space, and uses a simple decision (a one-dimensional decision threshold) at each non-leaf node that aims at maximizing an “information gain” function. Prediction is performed by “dropping” down the test data point from the root, and letting it traverse a path decided by the node decisions, until it reaches a leaf node. Each leaf node has a corresponding function value (or probability distribution on function values), adjusted according to training data, which is predicted as the function value for the test input. During training, randomization is injected into the procedure to reduce variance and avoid overfitting. This is achieved by training each individual tree on randomly selected subsets of the training samples (also called bagging), as well as by randomly selecting the deciding input variable for each tree node to decorrelate the trees.

一个决策树表示输入空间的一个递归二值分割，在每个非叶节点使用一个简单的决策（一个一维决策阈值），目标是最大化一个信息增益函数。预测的进行，是从根节点处抛下测试数据点，让其经过一条路径，由节点决策决定，直到其到达一个叶节点。每个叶节点有一个对应的函数值（或在函数值上的概率分布），根据训练数据进行调整，预测为测试输入的函数值。在训练过程中，随机性注入到这个过程中，以降低方差，避免过拟合。这是通过在随机选择的训练样本子集上训练每个单独的树得到的，还有为每个树节点随机选择决定的输入变量，以解除树之间的相关性。

A regression random forests is built from a set of such decision trees where the leaf nodes output the average of the training data labels and where the output of the whole forest is the average of the predicted results over all trees. In our experiments, we train separate regressors to learn the mapping from our input parameter space to each output variable.

一个回归随机森林是从这样的决策树的集合上构建起来的，其中的叶节点输出的是训练数据标签的平均，整个森林的输出是所有树的预测结果的平均。在我们的试验中，我们训练不同的回归器来学习从我们的输入参数空间到每个输出变量的映射。

It is believed that random forests are a good model for computer systems workloads [7], [15]. In fact, these workloads are often highly discontinuous, multi-modal, and nonlinear [21], all characteristics that can be captured well by the space partitioning behind a decision tree. In addition, random forests naturally deal with categorical and ordinal variables which are important in computer systems optimization. Other popular models like Gaussian processes [23] are less appealing for these type of variables. Additionally, a trained random forests is a “white box” model which is relatively simple for users to understand and to interpret (as compared to, for example, neural network models, which are more difficult to interpret).

随机森林是计算机系统workloads的好模型。实际上，这些workloads通常是高度不连续的，多模态的，非线性的，所有的特征都可以很好的被决策树的空间分割很好的捕获到。此外，随机森林很自然的处理类别和序数变量，在计算机系统优化中，这都是很重要的。其他流行的模型，如高斯过程，对这些变量类型来说不够有吸引力。另外，一个训练好的随机森林是一个白盒模型，用户理解和解释起来相对比较容易（这是与类似神经网络模型进行比较的，比较难以解释）。

## 3. Methodology

### 3.1 Injecting Prior Knowledge to Guide the Search

Here we consider the probability densities and distributions that are useful to model computer systems workloads. In these type of workloads the following should be taken into account:

这里，我们考虑对建模计算机系统workloads有用的概率密度和分布。在这些类型的workloads中，需要将下列限制纳入考虑：

- the range of values for a variable is finite. 一个变量的值的范围是有限的；
- the density mass can be uniform, bell-shaped (Gaussian-like) or J-shaped (decay or exponential-like). 密度质量可以是均匀的，钟形的，或J形的。

For these reasons, in HyperMapper 2.0 we propose the Beta distribution as a model for the search space variables. The following three properties of the Beta distribution make it especially suitable for modeling ordinal, integer and real variables; the Beta distribution:

由于这些原因，HyperMapper 2.0中，我们提出搜索空间变量的模型为Beta分布。Beta分布的下列三个属性使其尤其适合建模序数，整数和实数变量；Beta分布：

1) has a finite domain; 分布范围有限；

2) can flexibly model a wide variety of shapes including a bell-shape (symmetric or skewed), U-shape and J-shape. This is thanks to the parameters α and β (or a and b) of the distribution; 可以灵活对很多形状进行建模，包括钟形，U形和J形。这多亏了分布的α和β参数；

3) has probability density function (PDF) given by: 其概率密度函数(PDF)由下式给出

$$f(x|α, β) = \frac {Γ(α + β)} {Γ(α)Γ(β)} x^{α−1} (1 − x)^{β−1}$$(2)

for x ∈ [0, 1] and α, β > 0, where Γ is the Gamma function. The mean and variance can be computed in closed form.

对于x ∈ [0, 1]和α, β > 0，其中Γ是Gamma函数。其均值和方差可以以闭合形式计算得到。

Note that the Beta distribution has samples that are confined in the interval [0, 1]. For ordinal and integer variables, HyperMapper 2.0 automatically rescales the samples to the range of values of the input variables and then finds the closest allowed value in the ones that define the variables.

注意，Beta分布的样本是限定在区间[0, 1]中的。对于序数和整数值，HyperMapper 2.0自动重新改变样本到输入变量的值的范围，然后在那些定义了变量的值里，找到最近的允许值。

For categorical variables (with K modalities) we use a probability distribution, i.e., instead of a density, that can be easily specified as pairs of (xk, pk), where the set xk represents the k values of the variable and pk is the probability associated to each of them with $\sum^K_{k=1} p_k = 1$.

对于类别变量（有K个模态），我们使用概率分布，而不是密度，即，可以很容易的指定为(xk, pk)对，其中集合xk表示变量的k个值，pk表示与每个相关的概率值，$\sum^K_{k=1} p_k = 1$。

In Figure 2 we show Beta distributions with parameters α and β selected to suit computer systems workloads. We have selected four shapes as follows:

在图2中，我们展示了Beta分布，参数α和β要选定适合计算机系统的workloads。我们选择了4种形状如下：

1) Uniform (α = 1, β = 1): used as a default if the user has no prior knowledge on the variable. 均匀分布：如果用户对变量没有任何先验知识，就用作默认值。

2) Gaussian (α = 3, β = 3): when the user thinks that it is likely that the optimum value for that variable is located in the center but still wants to sample from the whole range of values with lower probability at the borders. This density is reminiscent of an actual Gaussian distribution, though it is finite.

高斯分布：当用户认为，变量的最优值很可能在中间，但仍然希望整个范围内的值以较低的概率成为样本。这个密度会让人想起一个实际的高斯分布，虽然是有限的。

3) Decay (α = 0.5, β = 1.5): used when the optimum is likely located at the beginning of the range of values. This is similar in shape to the log-uniform distribution as in [5], [6].

衰减分布：当最优值很可能分布在范围的开始，就使用这个分布。在形状上，这与[5,6]中的log-uniform分布相似。

4) Exponential (α = 1.5, β = 0.5): used when the optimum is likely located at the end of the range of values. This is similar in shape to the drawn exponentially distribution as in [5].

指数分布：当最优值很可能在范围的最后时使用。这与[5]中的指数分布的形状类似 。

### 3.2 Sampling with Categorical and Discrete Parameters

We first warm-up our model with simple random sampling. In the design of experiments (DoE) literature [26], this is the most commonly used sampling technique to warm-up the search. When prior knowledge is used, samples are drawn from each variable’s prior distribution, or the uniform distribution by default if no prior knowledge is provided.

我们首先用简单随机采样来预热我们的模型。在试验设计(DoE)的文献中，这是预热搜索最常用的采样技术。当先验知识使用时，样本从每个变量的先验分布中进行抽取，或者在没有先验知识时，使用默认的均匀分布。

### 3.3 Unknown Feasibility Constraints

The unknown feasibility constraints algorithm in Hyper-Mapper 2.0 is an adaptation of the constrained Bayesian optimization (cBO) method introduced in [13]. cBO is based on Gaussian Processes (GPs) to model the constraints and it uses these probabilistic models to guide the search, which is, it multiplies the acquisition function of the Bayesian optimization iteration by the constraints represented by the GPs; this leads to a new probabilistic model that is the combination of constraints and surrogate models. We refer to [13] for a more detailed explanation of the cBO algorithm.

在HyperMapper 2.0中的未知可行性约束算法是[13]中引入的约束贝叶斯优化(cBO)方法的调整。cBO基于高斯过程(GP)来建模约束，使用这些概率模型来引导搜索，即，将贝叶斯优化迭代中的获取函数乘以GPs代表的约束；这带来了一个新的概率模型，是约束和代理模型的组合。cBO算法的详细解释，请见[13]。

In HyperMapper 2.0 we implement the constraints with a random forests classification model. The advantage of this choice is that RF is a lightweight model that is interpretable shading light on how the feasibility design space looks like. Experiments in Section IV-C show the effectiveness of the random forests classifier in the context of feasibility constraints. This model can be seen as a filter that is at the core of the search algorithm in HyperMapper 2.0, as explained in Section III-D. The filter instructs the search algorithm on which configurations are likely to be infeasible so that the sampling budget can be used more efficiently.

在HyperMapper 2.0中，我们用随机森林分类模型实现了约束。这个选项的优势是，RF是一个轻量级模型，是可解释的，能让人看清可行设计空间是什么样子。在6.3节中的试验表明，随机森林分类器在可行性约束的上下文中是有效的。这个模型可以视为一个滤波器，是在HyperMapper 2.0中的搜索算法的核心位置，如3.4节所解释。滤波器给搜索算法关于哪个配置很可能是可行的指令，这样采样预算可以更高效的使用。

### 3.4 Active Learning

Active learning is a paradigm in supervised machine learning which uses fewer training examples to achieve better prediction accuracy by iteratively training a predictor, and using the predictor in each iteration to choose the training examples which will increase its accuracy the most. Thus the optimization results are incrementally improved by interleaving exploration and exploitation steps. We use randomized decision forests as our base predictors created from a number of sampled points in the parameter space.

主动学习是有监督机器学习范式的一种，通过迭代训练一个预测器，使用更少的训练样本来获得更好的预测准确率，在每次迭代中使用预测器来选择训练增加准确率最多的样本。因此，优化结果通过交替探索和利用步骤，得到逐步的增加。我们使用随机化的决策树作为我们的基准预测器，从参数空间中的几个采样点创建得到。

The application is evaluated on the sampled points, yielding the labels of the supervised setting given by the multiple objectives. Since our goal is to accurately estimate the points near the Pareto-optimal front, we use the current predictor to provide performance values over the parameter space and thus estimate the Pareto fronts. For the next iteration, only parameter points near the predicted Pareto front are sampled and evaluated, and subsequently used to train new predictors using the entire collection of training points from current and all previous iterations. This process is repeated over a number of iterations forming the active learning loop. Our experiments in Section IV indicate that this guided method of searching for highly informative parameter points in fact yields superior predictors as compared to a baseline that uses randomly sampled points alone. By iterating this process several times in the active learning loop, we are able to discover high-quality design configurations that lead to good performance outcomes.

应用在采样点上进行评估，在给定多个目标的情况下，得到有监督设置下的标签。由于我们的目标是准确的估计在Pareto最优front附近的点，我们使用当前的预测器来参数空间上的性能值，因此估计了Pareto front。对于下一次迭代，只采样在预测的Pareto front附近的参数点，然后进行评估，然后用于训练新的预测器，使用的是当前以及所有其他迭代得到的整个训练点的集合。这个过程重复数次，形成了主动学习的循环。我们在第4部分的试验表明，这种引导的方法搜索信息量很高的参数点，实际上得到了很好的预测器，这是与使用随机采样的点的基准进行比较的。通过在主动学习的循环中迭代这个过程几次，我们可以找到高质量的设计配置，得到很好的性能输出。

Algorithm 1 shows the pseudo-code of the model-based search algorithm used in HyperMapper 2.0. Figure 3 shows a corresponding graphical representation of the algorithm. The while loop on line 9 in Algorithm 1 is the active learning loop, represented by the big loop in the preprocessing box of Figure 3. The user specifies a maximum number of active learning iterations given by the variable maxAL. The function Fit_RF_Regressor() at lines 4, 5, 16 and 17 trains random forests regressors M_{obj1} and M_{obj2} which are the surrogate models to predict the objectives given a parameter vector. We train p separate models, one for each objective (p=2 in Algorithm 1). The random forests regressor is represented by the box ”Regressor” in Figure 3.

算法1展示了HyperMapper 2.0中使用的基于模型的搜索算法的伪代码。图3展示了算法对应的图形表示。算法1中第9行的while循环是主动学习循环，表示为图3中preprocessing框的大循环。用户指定主动学习迭代次数的一个最大值maxAL。第4，5，16，17行的函数Fit_RF_Regressor()从随机森林回归器M_{obj1}和M_{obj2}上训练，这是代理模型，在给定参数向量的时候，预测目标函数。我们训练p个单独的模型，每个目标函数一个（算法1中p=2）。随机森林回归器在图3中表示为Regressor框。

The function Fit_RF_Classifier() on lines 6 and 15 trains a random forests classifier M_f ea to predict if a parameter vector is feasible or infeasible. The classifier becomes increasingly accurate during active learning. Using a classifier to predict the infeasible parameter vectors has proven to be very effective as later shown in Section IV-C. The random forests classifier is represented by the box ”Classifier (Filter)” in Figure 3. The function Predict_Pareto on lines 7 and 19 filters the parameter vectors that are predicted infeasible from X before computing the Pareto, thus dramatically reducing the number of function evaluations. This function is represented by the box ”Compute Valid Predicted Pareto” in Figure 3.

第6和第15行的函数Fit_RF_Classifier()训练一个随机森林分类器M_fea来预测一个参数向量是可行的或不可行的。在主动学习的过程中，分类器变得越来越精确。使用一个分类器来预测不可行的参数向量，已经被证明非常高效，在4.3节会进行展示。随机森林分类器在图3中表示为Classifier框。第7和第19行的函数Predict_Pareto，在计算Pareto之前，过滤了从X预测为不可行的参数向量，因此极大的降低了函数评估的数量。这个函数由图3中的框Compute Valid Predicted Pareto表示。

For sake of space some details are not shown in Algorithm 1. For example, the while loop on line 9 is limited to M evaluations per active learning iteration. When the cardinality |P − X_{out}| > M, a maximum of M samples are selected uniformly at random from the set P − X_{out} for evaluation. In the case where |P − X_{out}| < M, a number of parameter vector samples M − |P − X_{out}| is drawn uniformly at random without repetition. This ensures exploration analogous to the ε-greedy algorithm in the reinforcement learning literature [32]. ε-greedy is known to provide balance between the exploration-exploitation trade-off.

由于空间原因，一些细节没有在算法1中展示。比如，第9行的while循环在每个主动学习迭代中，限制为M次评估。当基数|P − X_{out}| > M，选择最大M个样本，从集合P − X_{out}中随机的均匀选择，以进行评估。在|P − X_{out}| < M的情况下，一些参数向量样本M − |P − X_{out}|是随机均匀抽取的，没有重复。这确保了探索与[32]中强化学习的ε-贪婪算法类似。ε-贪婪能够在探索-利用的折中中给出很好的平衡。

### 3.5 Pareto Wall

In Algorithm 1 lines 7 and 19, the function Predict_Pareto eliminates the X_out samples from X before computing the Pareto front. This means that the newly predicted Pareto front never contains previously evaluated samples and, by consequence, a new layer of Pareto front is considered at each new iteration. We dub this multi-layered approach the Pareto Wall because we consider one Pareto front per active learning iteration, with the result that we are exploring several adjacent Pareto frontiers. Adjacent Pareto frontiers can be seen as a thick Pareto, i.e., a Pareto Wall. The advantage of exploring the Pareto Wall in the active learning loop is that it minimizes the risk of using a surrogate model which is currently inaccurate. At each active learning step, we search previously unexplored samples which, by definition, must be predicted to be worse than the current approximated Pareto front. However, in cases where the predictor is not yet very accurate, some of these unexplored samples will often dominate the approximated Pareto, leading to a better Pareto front and an improved model.

在算法1中的第7和19行，函数Predict_Pareto在计算Pareto front之前，从X中剔除了X_out个样本。这意味着新预测的Pareto front不包含之前评估的样本，结果是，在每次新的迭代中，都考虑了一个新的Pareto front层。我们称这种多层的方法为Pareto Wall，因为我们在每次主动学习迭代中，考虑一个Pareto front，结果是，我们在探索几个相邻的Pareto frontiers。相邻的Pareto frontiers可以被视为一个厚的Pareto，即，一个Pareto Wall。在主动学习的循环中探索Pareto Wall的好处是，这最小化了使用一个目前不准确的代理模型的风险。在每个主动学习步骤中，我们都搜索之前未探索的样本，在定义上，必须要预测为比当前近似的Pareto front要差。但是，在预测器并不十分精确的情况下，一些这些未探索的样本会dominate近似的Pareto，带来更好的Pareto front，和一个改进的模型。

### 3.6 The HyperMapper 2.0 Framework

HyperMapper 2.0 is written in Python and makes use of widely available libraries, e.g., scikit-learn and pyDOE. The HyperMapper 2.0 setup is via a simple json file. A light interface with third party software used for optimization is also necessary: templates for Python and Scala are provided. HyperMapper 2.0 is able to run in parallel on multi-core machines the classifiers and regressors as well as the computation of the Pareto front to accelerate the active learning iterations.

HyperMapper 2.0是用Python写的，使用了很多可用的库，如scikit-learn和pyDOE。HyperMapper 2.0的设置是通过一个简单的json文件。轻接口和第三方的优化软件也是必要的：提供了Python和Scala的模板。HyperMapper 2.0可以在多核机器上并行运行分类器和回归器，以及计算Pareto front，以加速主动学习迭代。

## 4. Evaluation

We run the evaluation on the recently proposed Spatial compiler [19], which implies a full integration of HyperMapper 2.0 on the Spatial production-level compiler toolchain for designing application hardware accelerators on FPGAs. We compare HyperMapper 2.0 with the HyperMapper 1.0 multi-objective auto-tuner to show the effectiveness of the feasibility constraints methodology. Then we compare HyperMapper 2.0 against the real Pareto where exhaustive search is possible, i.e. a total of three benchmarks. This is to give an insight on how the optimizer works in a controlled environment, i.e. when the Pareto front in known and the benchmark is small. Finally a comparison with the previous approach in Spatial, which is a mix of expert programmer pruning and random sampling, is given. The blocking factor to run more comparisons with other auto-tuners is that there is no available framework that has both RIOC variables and multi-objective features as shown in Table I. As an example, the popular OpenTuner supports multiple objectives that are scalarized in one objective or allows optimization of one objective while thresholding a second objective. This means that OpenTuner is inherently single-objective making the comparison with HyperMapper 2.0 not legitimate.

我们在最近提出的Spatial编译器上进行评估，即HyperMapper 2.0与Spatial生产级编译器工具链完全整合到了一起，以在FPGA上设计应用硬件加速器。我们比较了HyperMapper 2.0与HyperMapper 1.0多目标自动调节器比较，以证明可行性约束方法论的有效性。然后我们比较了HyperMapper 2.0与穷举式搜索是可能的真实的Pareto，总计3个基准测试。这是为了看看，优化器在一个受控的环境中工作的如何，即，当Pareto front是已知的，而且基准测试很小的时候。最后，在Spatial中与之前的方法进行比较，是专家程序员修剪和随机采样的混合。阻碍与其他自动调节器运行更多的比较的原因是，没有既有RIOC变量又有多目标特征的框架，如表I所示。作为一个例子，流行的OpenTuner支持多目标，但标量化为一个目标，或允许优化一个目标，同时对第二个目标采用阈值。这意味着，OpenTuner内在的就是单目标的，使得与HyperMapper 2.0的比较不合理。

### 4.1 The Spatial Programming Language

Spatial [19] is a domain-specific language (DSL) and corresponding compiler for the design of application accelerators on reconfigurable architectures. The Spatial frontend is tailored to present programmers with a high level of abstraction for hardware design. Control in Spatial is expressed as nested, parallelizable loops, while data structures are allocated based on their placement in the target hardware’s memory hierarchy. The language also includes support for design parameters to express values which do not change the behavior of the application and which can be changed by the compiler. These parameters can be used to express loop tile sizes, memory sizes, loop unrolling factors, and the like.

Spatial是在可重配置架构上设计应用加速器的领域专用语言(DSL)，也是对应的编译器。Spatial前端是定制的，呈现给程序员的是，硬件设计的高层次抽象。Spatial中的控制表达为嵌套的，可并行化的循环，而数据结构基于其在目标硬件内存的层次结构上的布局来进行分配。这个语言还支持设计参数来表达不改变应用的行为的值，和可以由编译器改变的值。这些参数可以用于表达循环块的大小，内存的大小，循环展开因子，及类似的。

As shown in Figure 4, the Spatial compiler lowers user programs into synthesizable Chisel [2] designs in three phases. In the first phase, it performs basic hardware optimizations and estimates a possible domain for each design parameter in the program. In the second phase, the compiler computes loop pipeline schedules and on-chip memory layouts for some given value for each parameter. It then estimates the amount of hardware resources and the runtime of the application. When targeting an FPGA, the compiler uses a device-specific model to estimate the amount of compute logic (LUTs), dedicated multipliers (DSPs), and on-chip scratchpads (BRAMs) required to instantiate the design. Runtime estimates are performed using similar device-specific models with average and worst case estimates computed for runtime-dependent values. Runtime is typically reported in clock cycles.

如图4所示，Spatial编译器在三个阶段中将用户程序降低到可综合的Chisel设计。在第一阶段，其进行的是基本硬件优化，对程序中的每个设计参数估计一个可能的domain。在第二阶段，编译器对每个参数的给定值，计算循环流程调度和片上内存布局。然后估计硬件资源的数量，和应用的运行时间。当目标是FPGA时，编译器使用一个设备专用的模型，来估计要实例化这个设计所需要的计算逻辑的数量(LUTs)，专用的乘法器(DSPs)，和片上scratchpads(BRAMs)。运行时间的估计是使用类似的设备专用的模型，带有平均情况估计，和最坏情况估计，为依赖于运行时间的值计算得到。运行时间一般以时钟周期数给出。

In the final phase of compilation, the Spatial compiler unrolls parallelized loops, retimes pipelines via register insertion, and performs on-chip memory layout and compute optimizations based on the analyses performed in the previous phase. Finally, the last pass generates a Chisel design which can be synthesized and run on the target FPGA.

在编译的最后阶段，Spatial编译器展开并行化的循环，通过寄存器插入来重新计算流水线的时间，进行片上内存布局，基于在前面的阶段进行的分析来计算优化。最后，最后一轮生成一个Chisel设计，是可以被综合，在目标FPGA上运行。

### 4.2 HyperMapper 2.0 in the Spatial Compiler

The collection of design parameters in a Spatial program, together with their respective domains, yields a hardware design space. The second phase of the compiler gives a way to estimate two cost metrics - performance and FPGA resource utilization - for a given design in this space. Existing work on Spatial has evaluated two methods for design space exploration. The first method heuristically prunes the design space and then performs randomized search with a fixed number of samples. The heuristics, first established by Spatial’s predecessor [20], help to eliminate obviously bad points within the design space prior to random search; the pruning is provided by expert FPGA developers. This is, in essence, a one-time hint to guide search. The second method evaluated the feasibility of using HyperMapper 1.0 [7] to drive exploration, concluding that the tool was promising but still required future development. In some cases, it performed poorly without a feasibility classifier as the search often focused on infeasible regions [19].

在一个Spatial程序中的设计参数集合，与其对应的领域，形成了一个硬件设计空间。编译器的第二阶段给出了一个为这个空间中的给定设计估计两种代价度量的方法，性能和FPGA资源利用率。在Spatial上现有的工作评估了设计空间探索的两种方法。第一种方法启发式的对设计空间修剪，用固定数量的样本进行随机搜索。这种启发式是由Spatial的前任首先确定下来的，帮助在随机搜索之前消除设计空间中明显的坏设计点；这种修剪是由专家FPGA开发者提供的。这实质上是一种一次性线索来引导搜索。第二种方法评估了使用HyperMapper 1.0来驱动探索的过程，得到的结论是这个工具是有希望的，但仍然需要未来的开发。在一些情况中，在没有可行性分类器的情况下，表现比较差，因为搜索通常在不可行区域中进行。

Spatial’s compiler includes hooks at the beginning and end of its second phase to interface with external tools for design space exploration. As shown in Figure 4, the compiler can query at the beginning of this phase for parameter values to evaluate. Similarly, the end of the second phase has hooks to output performance and resource estimates. HyperMapper 2.0 interfaces with these hooks to receive cost estimates, build a surrogate model, and drive search of the space.

Spatial的编译器包括在第二阶段的开始和结束处，与外部工具进行设计空间探索的接口的hooks。如图4所示，编译器可以在这个阶段的开始，查询要评估的参数值。类似的，第二阶段的最后也有hooks，可以输出性能和资源估计。HyperMapper 2.0与这些hoods有接口，以接收代价估计，构建一个代理模型，驱动空间的搜索。

In this work, we evaluate design space exploration when Spatial is targeting an Altera Stratix V FPGA with 48 GB of dedicated DRAM and a peak memory bandwidth of 76.8 GB/sec (an identical approach could be used for any FPGA target). We list the seven benchmarks we evaluate with HyperMapper 2.0 in Table II. These seven benchmarks are a representative subset of those previously used to evaluate the Spatial compiler [19].

本文中，我们评估了设计空间探索，Spatial的目标是一个Altera Stratix V FPGA，有48GB的专用DRAM，峰值内存带宽为76.8GB/s（也可以用于其他任何FPGA）。我们在表II中列出了用HyperMapper 2.0评估的7个基准测试。这7个基准测试是之前用来评估Spatial编译器的代表性子集。

### 4.3 Feasibility Classifier Effectiveness 可行性分类器的效用

We address the question of the effectiveness of the feasibility classifier in the Spatial use case. Of all the hyperparameters defined for binary random forests [22], the parameters that usually have the most impact on the performance of the random forests classifier are: n_estimators, max_depth, max_features and class_weight. The reader can refer to the scikit-learn random forests classifier documentation for more details on these model hyperparameters. We run an exhaustive search to fine-tune the binary random forests classifier hyperparameters and test its performance. The range of values we considered for these parameters is shown in Table III. This defines a comprehensive space of 81 possible choices, small enough that it can be explored using exhaustive search. We dub these choices of parameter vectors as config_1 to config_81 on the x axis.

我们在Spatial用例中，处理可行性分类器的效用问题。在为二值随机森林定义的所有超参数中，对随机森林分类器的性能有最大影响的参数通常是：n_estimators, max_depth, max_features和class_weight。读者可以参考scikit-learn随机森林分类器的文档，查看这些模型超参数的更多细节。我们运行穷举式搜索，来精调二值随机森林分类器的超参数，并测试其性能。对这些参数，我们考虑的值的范围，如表III所示。这定义了一个综合的空间，有81个可能的选项，非常小，可以用穷举式搜索。我们称这些参数向量选择为x轴上的config_1到config_81。

We perform a 5-fold cross-validation using the data collected by HyperMapper 2.0 as training data and report validation recall averaged over the 5 folds. The goal of this optimization procedure is for the binary classification to maximize recall. We want to maximize recall, i.e., true positives/(true positives+false negatives), because it is important to not throw away feasible points that are misclassified as being infeasible and that can potentially be good fits. Precision, i.e., true positives/(true positives+false positives), is less important as there is smaller cost associated with classifying an infeasible parameter vector as feasible. In this case the only downside is that some samples will be wasted because we are evaluating samples that are infeasible, which is not a major drawback.

我们进行5折交叉验证，使用的数据是HyperMapper 2.0收集的，作为训练数据，给出在5个folds上平均的验证recall。这个优化过程的目标是为了二值分类，以最大化recall。我们希望最大化recall，即，true positives/(true positives + false negatives)，因为不要将误分类为不可行的可行点丢弃掉，这可能是很好的拟合，这是很重要的。精度，即true positives/(true positives + false positives)则没那么重要，因为将不可行参数向量误归类为可行的，其代价则较小一些。在这种情况下，唯一的负面是一些样本会被浪费掉，因为我们在评估一些是不可行的样本，这并不是一个主要的缺点。

Figure 5 reports the recall of the random forests classifier across the 7 benchmarks and hyperparameter configurations. For sake of space, we only report the first 25 configurations, but the trend persists across all configurations. Figure 5 (top) shows the recall just after the warm-up sampling and before the first active learning iteration (Algorithm 1 line 6). Figure 5 (bottom) shows the recall after 50 active learning iterations (Algorithm 1 line 15 after 50 iterations of the while loop, where each iteration is evaluating 100 samples). The recall goes up during the active learning loop implying that the feasibility constraint is being predicted more accurately over time. The tables in Figure 5 show this general performance trend with the max mean improving from 0.784 to 0.967.

图5给出了在7个基准测试和各种超参数配置中，随机森林分类器的recall。鉴于空间原因，我们只给出前25个配置，但是趋势在所有配置中都有。图5上展示了在预热采样后、在主动学习第一次迭代（算法1行6）之前的召回。图5下展示了在50次主动学习迭代后的召回（算法1，行15，在50次while循环迭代后，每次迭代都评估了100个样本）。召回在主动学习循环后上升，说明可行性约束随着时间预测的更加准确了。图5中的表格展示了，这种max mean的通用性能趋势从0.784提升到了0.967。

In Figure 5 (top) the recall is low prior to the start of active learning. The configuration that scores best (the maximum score of the minimum scores across the different configurations) has a minimum score of 0.6 on the 7 benchmarks. The configuration is: {’class_weight’:{T:0.75,F:0.25}, ’max_depth’:8, ’max_features’:’auto’, ’n_estimators’:10}. The recall of this configuration ranges from a minimum of 0.6 for TPC-H Q6 to a maximum of 1.0 on BlackScholes with mean and standard deviation of 0.735 and 0.15 respectively.

图5上，召回在主动学习开始前比较低。7个基准测试中，得分最好的配置（在不同配置中，最低分的最高分）其最低分为0.6。其配置是：{’class_weight’:{T:0.75,F:0.25}, ’max_depth’:8, ’max_features’:’auto’, ’n_estimators’:10}。这种配置的召回，其最小值为TPC-H Q6的0.6，到BlackScholes的1.0，其均值和方差分别为0.735和0。15。

In Figure 5 (bottom) the recall is high after 50 iterations of active learning. There are two configurations that score best, with a minimum score of 0.886 on the 7 benchmarks. The configurations are: {’class_weight’:{T:0.75,F:0.25}, ’max_depth’:None, ’max_features’:’0.75’, ’n_estimators’:10} and {’class_weight’:{T:0.9,F:0.1}, ’max_depth’:None, ’max_features’:’0.75’, ’n_estimators’:10}. In general, most of the configurations are very close in terms of recall and the default random forests configuration scores high, perhaps suggesting that the random forests for these kind of workloads does not need a major tuning effort. The statistics of these configurations range from a minimum of 0.886 for DotProduct to a maximum of 1.0 on BlackScholes with mean and standard deviation of 0.964 and 0.04 respectively.

在图5下中，在50次主动学习后，召回就很高了。有两个配置分数最高，7个基准测试中最低分在0.886。配置是：{’class_weight’:{T:0.75,F:0.25}, ’max_depth’:None, ’max_features’:’0.75’, ’n_estimators’:10}和{’class_weight’:{T:0.9,F:0.1}, ’max_depth’:None, ’max_features’:’0.75’, ’n_estimators’:10}。总体上，多数配置在recall上是非常接近的，默认的随机森林配置的分数就很高，可能说明，对于这种workloads的随机森林不需要大量的调节工作。这些配置的统计结果，从DotProduct的最低分0.886，到BlackScholes的最高分1.0，均值和方差分别为0.964和0.04。

Figure 6 compares the predicted Pareto fronts of GDA, the benchmark with the largest design space, using HyperMapper 1.0 and HyperMapper 2.0. HyperMapper 1.0 does not exploit the feasibility constraints feature introduced by HyperMapper 2.0 as shown in Table I. In both cases, we use random sampling to warm-up the optimization with 1,000 samples followed by 5 iterations of active learning. The red dots representing the invalid points for the case without feasibility constraints (HyperMapper 1.0) are spread farther from the corresponding Pareto frontier while the green dots for the case with constraints (HyperMapper 2.0) are close to the respective frontier. This happens because the non-constrained search focuses on seemingly promising but unrealistic points. HyperMapper 2.0 with its constrained search focuses in a region that is more conservative but feasible. The effect of the feasibility constraint is apparent in its improved Pareto front, which almost entirely dominates the approximated Pareto front resulting from unconstrained search. For the sake of brevity, we only show experiments on the biggest design space considered in this evaluation section, i.e., the GDA benchmark, however the results are confirmed in the rest of the benchmarks.

图6比较了GDA预测的Pareto front，有着最大设计空间的基准测试，使用的是HyperMapper 1.0和2.0。HyperMapper 2.0提出了可行性约束特征，而HyperMapper 1.0并没有使用这个特征，如表I所示。在两种情况中，我们使用随机采样来采用1000个样本来预热优化，然后进行5轮主动学习。红点表示无效点（对于没有可行性约束的情况，HyperMapper 1.0），分布在距离Pareto frontier很远的地方，而绿点（带有可行性约束的情况，HyperMapper 2.0）与对应的frontier则很近。这种情况的发生是因为，无约束的搜索聚焦在看起来很有希望，但不实际的点。HyperMapper 2.0带有可行性约束，其搜索聚焦的区域更保守，但可行。可行性约束的效果很明显的，改进了Pareto front，几乎完全dominate了由非约束搜索得到的近似的Pareto front。为简洁，我们只展示了在最大设计空间中的试验，即GDA基准测试，但是，在其他基准测试中，也确认得到了类似的结果。

### 4.4  Optimum vs. Approximated Pareto

We next take the smallest benchmarks, BlackScholes, DotProduct and OuterProduct, and run exhaustive search to compare the approximated Pareto front computed by HyperMapper 2.0 with the true optimal one. This can be achieved only for such small benchmarks as exhaustive search is feasible. However, even on these small spaces, exhaustive search requires 6 to 12 hours when parallelized across 16 CPU cores. In our framework, we use random sampling to warm-up the search with 1000 random samples followed by 5 active learning iterations of about 500 samples total.

下一步，我们用最小的基准测试，BlackScholes，DotProduct和OuterProduct，运行穷举式搜索，来比较HyperMapper 2.0计算得到的近似Pareto front，与真实最优值。这只对这种小型基准测试可以得到结果，因为穷举式搜索是可行的。但是，即使是在这种小型空间，穷举式搜索在16核CPU并行进行时，也需要6-12个小时。在我们的框架中，我们使用随机采样，用1000个随机样本来预热搜索，然后用5次主动学习迭代，利用了500个样本。

Comparisons are synthesized in Figure 7. The optimal Pareto front is very close to the approximated one provided by HyperMapper 2.0, showing our software’s ability to recover the optimal Pareto front. About 1500 total samples are required to recover the Pareto optimum, about the same number of samples for BlackScholes and 66 times fewer for OuterProduct and DotProduct compared to the prior Spatial design space exploration approach using pruning and random sampling.

图7综合了比较结果。最优Pareto front与HyperMapper 2.0给出的近似非常接近，表明我们的软件恢复最优Pareto front的能力。总计大约需要1500个样本来恢复Pareto optimum，与之前使用剪枝和随机采样进行Spatial设计空间探索方法相比，BlackScholes需要的数量大致相同，OuterProduct和DotProduct需要的数量少了66倍。

### 4.5 Hypervolume Indicator

We next show the hypervolume indicator (HVI) [12] function for the whole set of the Spatial benchmarks as a function of the initial number of warm-up samples (for sake of space we omit the smallest benchmark, BlackScholes). For every benchmark, we show 5 repetitions of the experiments and report variability via a line plot with 80% confidence interval. The HVI metric gives the area between the estimated Pareto frontier and the spaces true Pareto front. This metric is the most common to compare multi-objective algorithm performance. Since the true Pareto front is not always known, we use the accumulation of all experiments run on a given benchmark to compute our best approximation of the true Pareto front and use this as a true Pareto. This includes all repetitions across all approaches, e.g., baseline and HyperMapper 2.0. In addition, since logic utilization and cycles have different value ranges by several order of magnitude, we normalize the data by dividing by the standard deviation before computing the HVI. This has the effect of giving the same importance to the two objectives and not skewing the results towards the objective with higher raw values. We set the same number of samples for all the experiments to 100,000 (the default value in the prior work baseline). Based on advice by expert hardware developers, we modify the Spatial compiler to automatically generate the prior knowledge discussed in Section III-A based on design parameter types. For example, on-chip tile sizes have a “decay” prior because increasing memory size initially helps to improve DRAM bandwidth utilization but has diminishing returns after a certain point. This prior information is passed to HyperMapper 2.0 and is used to magnify random sampling. The baseline has no support for prior knowledge.

下面，我们展示一下，Spatial基准测试整个集合的超体指示器(HVI, hypervolume indicator)函数，作为预热样本初始数量的函数（由于空间原因，我们忽略了最小的基准测试，BlackScholes）。对于每个基准测试，我们展示5次重复的试验，通过画在80%置信度间隔的线来给出变化。HVI度量给出了估计的Pareto frontier和空间真正的Pareto点之间的面积。这个度量在比较多目标算法性能中最为常用。由于真正的Pareto front并不是永远已知的，我们使用所有试验在一个给定基准测试上的累积，来计算我们对真正Pareto front的最好约束，使用这个作为一个真正的Pareto。这包含了所有方法的所有重复，如，基准和HyperMapper 2.0。另外，由于逻辑利用和周期有不同的值的范围，相差达到几个数量级，我们对数据进行了归一化，除以了标准差，然后才计算HVI。这个效果是，给两个目标函数相同的重要性，不会将结果偏向于有较高原始值的目标。我们对所有试验都设置相同数量的样本100000（这是在之前的工作基准中的默认值）。基于专家硬件开发者的建议，我们修改Spatial编译器，基于设计参数类型自动生成在3.1节讨论的先验知识。比如，片上tile size有衰减的先验，因为增加内存大小最初会帮助改进DRAM带宽利用，但是在特定点后回报会逐渐消失。这个先验信息传递到HyperMapper 2.0中，用于放大随机采样。这个基准不支持先验知识。

Figure 8 shows the two different approaches: HyperMapper 2.0 using a warm-up sampling phase with the use of the prior and then an active learning phase; Spatial’s previous design space exploration approach (the baseline). The y-axis reports the HVI metric and the x-axis the number of samples in thousands.

图8展示了两种不同的方法：HyperMapper 2.0使用了一种预热采样阶段，使用了先验，然后是主动学习阶段；Spatial之前的设计空间探索方法（基准）。y轴给出了HVI度量，x轴以k为单位给出样本数量。

Table IV quantitatively summarizes the results. We observe the general trend that HyperMapper 2.0 needs far fewer samples to achieve competitive performance compared to the baseline. Additionally, our framework’s variance is generally small, as shown in Figure 8. The total number of samples used by HyperMapper 2.0 is 12,500 on all experiments while the number of samples performed by the baseline varies as a function of the pruning strategy. The number of samples for GEMM, T-PCH Q6, GDA, and DotProduct is 100,000, which leads to an efficiency improvement of 8×, while OuterProduct and K-Means are 31,068 and 18,720, which leads to an improvement of 2.49× and 1.5×, respectively.

表IV量化的总结了结果。我们观察到，一般的趋势是，HyperMapper 2.0只需要少的多的样本，就可以达到基准类似的性能。另外，我们框架的变化一般来说很小，如图8所示。HyperMapper 2.0在所有试验中使用的总计数量的样本是12500，而基准使用的样本数量则随着剪枝策略而变化。GEMM，T-PCH Q6，GDA和DotProduct的样本数量是100000，这带来的性能改进是8x，而OuterProduct和K-Means是31068和18720，其改进分别是2.49x和1.5x。

As a result, the autotuner is robust to randomness and only a reasonable number of random samples are needed for the warm-up and active learning phases.

结果是，自动调节器对随机性是稳健的，只需要一定数量的随机样本来进行预热，并进行主动学习阶段。

### 4.6 Understandability of the Results

HyperMapper 2.0 can be used by domain non-experts to understand more about the domain they are trying to optimize. In particular, users can view feature importance to gain a better understanding of the impact of various parameters on the design objectives. The feature importances for the BlackScholes and OuterProduct benchmarks are given in Table V.

领域非专家可以使用HyperMapper 2.0，来理解要优化的领域更多的东西。特别是，用户可以看到特征重要性，以更好的理解不同的参数对设计目标的影响。BlackScholes和OuterProduct基准测试的特征重要性，如表V所示。

In BlackScholes, innermost loop parallelization (IP) directly determines how fast a single tile of data can be processed. Consequently, as shown in Figure V, IP is highly related to both the design logic utilization and design run-time (cycles). Since BlackScholes is bandwidth bound, changing DRAM utilization with tile sizes directly changes the run-time, but has no impact on the compute logic since larger memories do not require more LUTs. Outer loop parallelization (OP) also duplicates compute logic by making multiple copies of each inner loop, but as shown in Figure V, OP has less importance for run-time than IP.

在BlackScholes中，内层循环并行化(IP)直接决定了单个tile的数据可以处理的速度有多快。结果是，如图V所示，IP与设计逻辑利用率和设计运行时间（周期数）是高度相关的。由于BlackScholes是受到带宽约束的，随着tile sizes直接改变DRAM利用率，会改变运行时间，但对计算逻辑则没有影响，因为更大的内存不需要更多的LUTs。外部循环并行化(OP)还也复制了计算逻辑，即复制了多个内部循环，但如图V所示，OP对运行时间的重要性并没有IP大。

Similarly, in OuterProduct, both tile sizes have roughly even importance on the number of execution cycles, while IP has roughly even importance for both logic utilization and cycles. Unlike BlackScholes, which includes a large amount of floating point compute, OuterProduct has relatively little computation, making the cost of outer loop pipelining relatively impactful on logic utilization but with little importance on cycles. In both cases, this information is taken into account when determining whether to prioritize further optimizing the application for inner loop parallelization or outer loop pipelining.

类似的，在OuterProduct中，两个tile sizes对执行cycles的数量的重要性是大致一样的，而IP对逻辑利用率和周期数的重要性则是类似的。BlackScholes包含大量浮点计算，与之不同的是，OuterProduct的计算量则相对很小，使外部循环流水线的代价，对逻辑利用率要相对更有影响，但对cycles则没有什么重要性。在两个情况中，在确定是否要对应用对内部循环并行还是外部循环流水线化进行优化时，这都进行了考虑。

### 4.7 Optimization Wall-clock Time

Since HyperMapper 1.0 was not optimized for tuning time [21], our framework is already one order of magnitude faster in average. We run on four Intel XEON E7-8890 at 2.50GHz but HyperMapper 2.0 runs mostly sequentially. Optimization wall-clock time varies with the benchmark and is in the same order of magnitude as the Spatial baseline (tens of seconds). This includes the time to evaluate the Spatial samples which is independent from HyperMapper 2.0.

由于HyperMapper并没有对调节时间进行优化，我们的框架已经平均快了一个数量级。我们在四个Intel XEON E7-8890以2.50GHz的速度运行，但HyperMapper 2.0主要是顺序执行。优化wall-clock时间随着基准测试的不同而不同，与Spatial基准的数量级是一样的（几十秒）。这包括了评估独立于HyperMapper 2.0的Spatial样本的时间。

## 5. Related Work

During the last two decades, several design space exploration techniques and frameworks have been used in a variety of different contexts ranging from embedded devices to compiler research to system integration. Table I provides a taxonomy of methodologies and software from both the computer systems and machine learning communities. HyperMapper 2.0 has been inspired by a wide body of work in multiple sub-fields of these communities. The nature of computer systems workloads brings some important features to the design of HyperMapper 2.0 which are often missing in the machine learning community research on design space exploration tools.

在过去二十年中，几种设计空间探索技术和框架已在几种不同的上下文中应用，从嵌入式设备，到编译器研究，到系统集成。表I给出了方法和软件分类，包括计算机系统和机器学习的方面。HyperMapper 2.0得到了很多不同子领域的工作启发。计算机系统workloads的一些特性，给HyperMapper 2.0的设计带来了一些重要特征，这在机器学习团体对设计空间探索工具的研究中是没有的。

In the system community, a popular, state-of-the-art design space exploration tool is OpenTuner [1]. This tool is based on direct approaches (e.g. , differential evolution, Nelder-Mead) and a methodology based on the Area Under the Curve (AUC) and multi-armed bandit techniques to decide what search algorithm deserves to be allocated a higher resource budget. OpenTuner is different from our work in a number of ways. First, our work supports multi-objective optimization. Second, our white-box model-based approach enables the user to understand the results while learning from them. Third, our approach is able to consider unknown feasibility constraints. Lastly, our framework has the ability to inject prior knowledge into the search. The first point in particular does not allow a direct performance comparison of the two tools.

在系统团体中，流行的目前最好的设计空间探索工具是OpenTuner。这个工具是基于直接方法的（如，微分演化，Nelder-Mead）和一种基于AUC和multi-armed bandit技术的方法学，来决定什么搜索算法值得分配更高的资源预算。OpenTuner与我们的工作在几个方面是不同的。第一，我们的工作支持多目标优化。第二，我们基于白盒模型的方法使用户在从中学习的时候理解结果。第三，我们的方法可以考虑未知的可行性约束。最后，我们的框架有将先验知识导入到搜索中的能力。第一点使得这两个工具不能进行直接的性能比较。

Our work is inspired by HyperMapper 1.0 [7], [19], [21], [25]. Bodin et al. [7] introduce HyperMapper 1.0 for autotuning of computer vision applications by considering the full software/hardware stack in the optimization process. Other prior work applies it to computer vision and robotics applications [21], [25]. There has also been preliminary study of applying HyperMapper 1.0 to the Spatial programming language and compiler like in our work [19]. However, it lacks fundamental features that makes it ineffective in the presence of applications with non-feasible designs and prior knowledge.

我们的工作由HyperMapper启发得到。Bodin等[7]提出了HyperMapper 1.0，用于自动调节计算机视觉应用，在优化过程中考虑完整的硬件/软件堆栈。其他之前的工作将其应用到计算机视觉和机器人的应用中。之前也有初步的工作，将HyperMapper 1.0应用到Spatial编程语言和编译器中的，和我们类似。但是，它缺少基本的特征，使其在带有不可行设计和先验知识的应用中，不会奏效。

In [16] the authors use an active learning technique to build an accurate surrogate model by reducing the variance of an ensemble of fully connected neural network models. However, our work is fundamentally different because we are not interested in building a perfect surrogate model, instead we are interested in optimizing the surrogate model (over multiple objectives). So, in our case building a very accurate surrogate model over the entire space would result in a waste of samples.

在[16]中，作者使用了一种主动学习技术，来构建一个准确的代理模型，降低全连接神经网络模型集成的变化。但是，我们的工作在根本上有不同之处，因为我们对构建一个准确的代理模型是不感兴趣的，我们感兴趣的是优化代理模型（在多目标优化中）。所以，在我们的情况中，在整个空间中构建一个非常准确的代理模型，会是非常浪费时间的。

Recent work [9] uses decision trees to automatically tune discrete NVIDIA and SoC ARM GPUs. Norbert et al. tackle the software configurability problem for binary [29] and for both binary and numeric options [28] using a performance-influence model which is based on linear regression. They optimize for execution time on several examples exploring algorithmic and compiler spaces in isolation.

最近的工作使用决策树来自动调节离散的NVIDIA和SoC ARM GPUs。Norbert等对二值选项和二值与数值选项，使用基于线性回归的性能影响模型，处理软件可配置性问题。他们在几个例子上优化了执行时间，独立探索了算法和编译器空间。

In particular, machine learning (ML) techniques have been recently employed in both architectural and compiler research. Khan et al. [18] employed predictive modeling for cross-program design space exploration in multi-core systems. The techniques developed managed to explore a large design space of chip-multiprocessors running parallel applications with low prediction error. In [4] Balaprakash et al. introduce AutoMOMML, an end-to-end, ML-based framework to build predictive models for objectives such as performance, and power. [3] presents the ab-dynaTree active learning parallel algorithm that builds surrogate performance models for scientific kernels and workloads on single-core, multi-core and multi-node architectures. In [34] the authors propose the Pareto Active Learning (PAL) algorithm which intelligently samples the design space to predict the Pareto-optimal set.

特比是，最近在架构研究和编译器研究上采用了机器学习技术。Khan等[18]采用预测建模在多核系统中进行跨程序设计空间建模。开发的技术探索了一个大型设计空间，即芯片多处理器运行并行应用，预测错误很低。在[4]中，Balaprakash等引入了AutoMOMML，一个端到端的，基于ML的框架，对性能和功耗这样的目标构建了预测模型。[3]提出了ab-dynaTree主动学习并行算法，对科学核和workloads，在单核，多核和多节点架构中，构建了代理性能模型。在[34]中，作者提出了Pareto主动学习，智能的采样了设计空间，以预测Pareto-optimal集合。

Our work is similar in nature to the approaches adopted in the Bayesian optimization literature [27]. Example of widely used mono-objective Bayesian DFO software are SMAC [15], SpearMint [30], [31] and the work on tree-structured Parzen estimator (TPE) [6]. These mono-objective methodologies are based on random forests, Gaussian processes and TPEs making the choice of learned models varied.

我们的工作在本质上与[27]的贝叶斯优化工作类似。广泛采用的单目标贝叶斯DFO软件的例子，如SMAC，SpearMint，和树结构Parzen估计器的工作。这些单目标方法是基于随机森林，高斯过程和TPE的，使学习模型的选择有所变化。

## 6. Conclusions and Future Work

HyperMapper 2.0 is inspired by HyperMapper 1.0 [7], by the philosophy behind OpenTuner [1] and SMAC [15]. We have introduced a new DFO methodology and corresponding framework which uses guided search using active learning. This framework, dubbed HyperMapper 2.0, is built for practical, user-friendly design space exploration in computer systems, including support for categorical and ordinal variables, design feasibility constraints, multi-objective optimization, and user input on variable priors. Additionally, HyperMapper 2.0 uses randomized decision forests to model the searched space. This model not only maps well for the discontinuous, non-linear spaces in computer systems, but also gives a “white box” result which the end user can inspect to gain deeper.

HyperMapper 2.0是受HyperMapper 1.0启发，受到OpenTuner和SMAC背后的哲学启发得到的。我们提出了一种新的DFO方法论和对应的框架，使用了采用主动学习的引导搜索。这个框架称为HyperMapper 2.0，是构建用于计算系统中的实践的，用户友好的设计空间探索，支持类别和序数变量，设计可行性约束，多目标优化，和用户输入的可变先验。此外，HyperMapper 2.0使用随机化的决策树来建模搜索空间。这个模型不仅对计算机系统中不连续的非线性空间映射的很好，还给出了一个白盒模型，最终用户可以深入观察到内部。

We have presented the application of HyperMapper 2.0 as a compiler pass of the Spatial language and compiler for generating application accelerators on FPGAs. Our experiments show that, compared to the previously used heuristic random search, our framework finds similar or better approximations of the true Pareto frontier, with significantly fewer samples required, 8x in most of the benchmarks explored.

我们给出了HyperMapper 2.0作为Spatial语言的编译器pass，和生成在FPGAs上的应用加速器的编译器的应用。我们的试验表明，与之前使用的启发式随机搜索比，我们的框架可以找到Pareto frontier类似或更好的近似，需要的样本数量少了很多，在探索过的多数的基准测试中，都少了8x。

Future work will include analysis and incorporation of other DFO strategies. In particular, the use of a full Bayesian approach will help to leverage the prior knowledge by computing a posterior distribution. In our current approach we only exploit the prior distribution at the level of the initial warm-up sampling. Exploration of additional methods to warm-up the search from the design of experiments literature is a promising research venue. In particular the Latin Hypercube sampling technique was recently adapted to work on categorical variables [33] making it suitable for computer systems workloads. Future work will target an extension of this work to the Spatial ASIC back end as well as the Halide programming language.

未来的工作会包括，分析和纳入其他DFO策略。特别是，一个完整贝叶斯方法的使用，会帮助利用先验知识，计算一个后验分布。在我们目前的方法中，我们只在初始预热采样的阶段来利用先验分布。从DoE文献中利用其他方法来预热搜索是一个很有希望的研究路线。特别是，最近Latin Hypercube采样技术被用于类别变量，使其适用于计算机系统workloads。未来的工作会面向本文的拓展，即Spatial ASIC后端，以及Halide编程语言。