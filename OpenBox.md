# OpenBox: A Generalized Black-box Optimization Service

Yang Li, et. al. @ Peking University

## 0. Abstract

Black-box optimization (BBO) has a broad range of applications, including automatic machine learning, engineering, physics, and experimental design. However, it remains a challenge for users to apply BBO methods to their problems at hand with existing software packages, in terms of applicability, performance, and efficiency. In this paper, we build OpenBox, an open-source and general-purpose BBO service with improved usability. The modular design behind OpenBox also facilitates flexible abstraction and optimization of basic BBO components that are common in other existing systems. OpenBox is distributed, fault-tolerant, and scalable. To improve efficiency, OpenBox further utilizes “algorithm agnostic” parallelization and transfer learning. Our experimental results demonstrate the effectiveness and efficiency of OpenBox compared to existing systems.

黑箱优化(BBO)应用非常广泛，包括自动机器学习，工程，物理和试验设计。但是，用户用现有的软件包，将BBO应用到其手头上的问题中，在应用性，性能，和效率上，这仍然是一个问题。本文中，我们构建了OpenBox，一个开源，通用目标的BBO服务，可用性得到了改进。OpenBox的模块化设计也使得在其他已有的系统中通用的基本BBO组成部分的灵活的抽象和优化非常方便。OpenBox是分布式的，可容忍错误的，可扩展的。为改进效率，OpenBox进一步利用了与算法无关的并行化和迁移学习。我们的试验结果表明了OpenBox与其他系统相比，是有效的，而且是高效的。

**Keywords** Bayesian Optimization, Black-box Optimization

## 1. Introduction

Black–box optimization (BBO) is the task of optimizing an objective function within a limited budget for function evaluations. “Black-box” means that the objective function has no analytical form so that information such as the derivative of the objective function is unavailable. Since the evaluation of objective functions is often expensive, the goal of black-box optimization is to find a configuration that approaches the global optimum as rapidly as possible.

黑箱优化(BBO)是在有限的函数评估的预算下，优化一个目标函数的任务。黑箱的意思是，目标函数并没有解析的形式，所以目标函数的梯度这样的信息是不可用的。因为目标函数的评估通常是昂贵的，黑盒优化的目标是，找到一种配置，可以尽快的找到全局最优值。

Traditional BBO with a single objective has many applications: 1) automatic A/B testing, 2) experimental design [15], 3) knobs tuning in database [46, 48, 49], and 4) automatic hyper-parameter tuning [6, 27, 32, 44], one of the most indispensable components in AutoML systems [1, 34] such as Microsoft’s Azure Machine Learning, Google’s Cloud Machine Learning, Amazon Machine Learning [35], and IBM’s Watson Studio AutoAI, where the task is to minimize the validation error of a machine learning algorithm as a function of its hyper-parameters. Recently, generalized BBO emerges and has been applied to many areas such as 1) processor architecture and circuit design [2], 2) resource allocation [18], and 3) automatic chemical design [22], which requires more general functionalities that may not be supported by traditional BBO, such as multiple objectives and constraints. As examples of applications of generalized BBO in the software industry, Microsoft’s Smart Buildings project [36] searches for the best smart building designs by minimizing both energy consumption and construction costs (i.e., BBO with multiple objectives); Amazon Web Service aims to optimize the performance of machine learning models while enforcing fairness constraints [39] (i.e., BBO with constraints).

传统的单目标BBO有很多应用：1)自动A/B测试，2)试验设计，3)数据库中的knobs调节，4)自动超参数调节，在一些AutoML系统中，如微软的Azure机器学习系统，Google云机器学习系统，Amazon的机器学习系统，和IBM的Watson Studio AutoAI系统，这是最不可或缺的一部分，其任务是最小化机器学习算法作为其超参数的函数的验证误差。最近，通用BBO出现了，应用到了很多领域，比如，1)处理器架构和电路设计，2)资源分配，3)自动化学物质设计，这需要更通用的功能，传统的BBO可能并不支持，比如多目标和约束。通用BBO在软件产业中的应用的一个例子是，微软的智慧建筑项目，其搜索最智慧的建筑设计的方法是，最小化能耗和建筑价格（即，多目标的BBO）；Amazon的网页服务的目标是，优化机器学习模型的性能，同时加入一些合理的约束（即，带有约束的BBO）。

Many software packages and platforms have been developed for traditional BBO (see Table 1). Yet, to the best of our knowledge, so far there is no platform that is designed to target generalized BBO. The existing BBO packages have the following three limitations when applied to general BBO scenarios:

对传统BBO，已经开发了很多软件包和平台（见表1）。但是，据我们所知，目前并没有为通用BBO设计任何平台。现有的BBO包在应用到通用BBO场景时，有三个局限：

(1) Restricted scope and applicability. Restricted by the underlying algorithms, most existing BBO implementations cannot handle diverse optimization problems in a unified manner (see Table 1). For example, Hyperopt [6], SMAC3 [27], and HpBandSter [13] can only deal with single-objective problems without constraints. Though BoTorch [3] and GPflowOpt [30] can be used, as a framework, for developers to implement new optimization problems with multi-objectives and constraints; nevertheless, their current off-the-shelf supports are also limited (e.g., the support for non-continuous parameters).

范围和可应用性有限。受到潜在的算法的限制，多数现有BBO的实现不能以统一的方式处理各种优化问题。比如，Hyperopt，SMAC3和HpBandSter只能处理没有约束的单目标优化问题。虽然可以将BoTorch和GPflowOpt用作一个框架，开发者可以实现新的带有多目标和约束的优化算法；尽管如此，他们当前开箱即用的支持也是有限的（如，支持非连续参数）。

(2) Unstable performance across problems. Most existing software packages only implement one or very few BBO algorithms. According to the “no free lunch” theorem [26], no single algorithm can achieve the best performance for all BBO problems. Therefore, existing packages would inevitably suffer from unstable performance when applied to different problems. Figure 1 presents a brief example of hyper-parameter tuning across 25 AutoML tasks, where for each problem we rank the packages according to their performances. We can observe that all packages exhibit unstable performance, and no one consistently outperforms the others. This poses challenges on practitioners to select the best package for a specific problem, which usually requires deep domain knowledge/- expertise and is typically very time-consuming.

在不同问题中的性能不稳定。多数现有的软件包只实现了一个或很少几个BBO算法。根据没有免费午餐的定律，没有那个单独的算法能够在所有BBO问题中获得最好的性能。因此，现有的包在应用到不同的问题时，会不可避免的遇到不稳定的性能。图1给出了超参数调节在25个AutoML任务中的简单例子，对每个问题我们根据性能来对软件包进行排序。我们可以观察到，所有的包都表现出了不稳定的性能，没有哪个会一直比其他的更好。这会使实践者在一个具体问题上选择最好的包时，会遇到挑战，这通常需要很深的领域知识或专业知识，一般是非常耗时的。

(3) Limited scalability and efficiency. Most existing packages execute optimization in a sequential manner, which is inherently inefficient and unscalable. However, extending the sequential algorithm to make it parallelizable is nontrivial and requires significant engineering efforts. Moreover, most existing systems cannot support transfer learning to accelerate the optimization on a similar task.

有限的可扩展性和效率。多数现有的包都以顺序的方式进行优化，这在内在上就不是高效的，也是不可扩展的。但是，将顺序的算法扩展成可并行的，这有很大的工作量，需要很多工程上的努力。而且，多数现有的系统不支持迁移学习，就不能在相似的任务上加速优化过程。

With these challenges, in this paper we propose OpenBox, a system for generalized black-box optimization. The design of OpenBox follows the philosophy of providing “BBO as a service” — instead of developing another software package, we opt to implement OpenBox as a distributed, fault-tolerant, scalable, and efficient service, which addresses the aforementioned challenges in a uniform manner and brings additional advantages such as ease of use, portability, and zero maintenance. In this regard, Google’s Vizier [19] is perhaps the only existing BBO service as far as we know that follows the same design philosophy. Nevertheless, Vizier only supports traditional BBO, and cannot be applied to general scenarios with multiple objectives and constraints that OpenBox aims for. Moreover, unlike Vizier, which remains Google’s internal service as of today, we have open-sourced OpenBox that is available at https://github.com/PKU-DAIR/open-box.

有了这些挑战，本文中，我们提出了OpenBox，这是一种通用黑盒优化的系统。OpenBox的设计哲学，是将BBO作为一种服务来提供，而不是开发另一个软件包，我们将OpenBox实现为一个分布式的，能容错的，可扩展的和高效的服务，统一处理前面提到的挑战，带来了额外的优势，比如容易使用，可移植性，零维护。这方面，Google的Vizier是唯一现有的BBO服务，有着相同的设计哲学。尽管如此，Vizier只支持传统的BBO，不能应用到通用的场景，如多目标，带有约束，这是OpenBox的目标。而且，与Vizier不同的是，我们将OpenBox开源了。

The key novelty of OpenBox lies in both the system implementation and algorithm design. In terms of system implementation, OpenBox allows users to define their tasks and access the generalized BBO service conveniently via a task description language (TDL) along with customized interfaces. OpenBox also introduces a high-level parallel mechanism by decoupling basic components in common optimization algorithms, which is “algorithm agnostic” and enables parallel execution in both synchronous and asynchronous settings. Moreover, OpenBox also provides a general transfer-learning framework for generalized BBO, which can leverage the prior knowledge acquired from previous tasks to improve the efficiency of the current optimization task. In terms of algorithm design, OpenBox can host most of the state-of-the-art optimization algorithms and make their performances more stable via an automatic algorithm selection module, which can choose proper optimization algorithm for a given problem automatically. Furthermore, OpenBox also supports multi-fidelity and early-stopping algorithms for further optimization of algorithm efficiency.

OpenBox的关键创新是系统实现和算法设计。在系统实现上，OpenBox使用户可以通过任务描述语言(TDL)用定制的界面来定义他们的任务，方便的访问通用BBO服务。OpenBox还引入了一种高层次并行机制，将常见的优化算法中的基本组成部分进行解耦，这是算法无关的，可以在同步和异步的设置中进行并行执行。而且，OpenBox还对通用BBO给出了一种通用的迁移学习框架，可以利用从之前的任务获得的先验知识，来改进目前优化任务的效率。在任务设计上，OpenBox包含了多数目前最好的优化算法，通过自动算法选择模块，对给定问题自动选择合适的优化算法，使性能会非常稳定。而且，OpenBox还支持多保真度和早停算法，以进一步优化算法效率。

Contributions. In summary, our main contributions are:

C1. An open-sourced service for generalized BBO. To the best of our knowledge, OpenBox is the first open-sourced service for efficient and general black-box optimization.

通用BBO的开源服务。据我们所知，OpenBox是第一个高效通用黑箱优化算法的开源服务。

C2. Ease of use. OpenBox provides user-friendly interfaces, visualization, resource-aware management, and automatic algorithm selection for consistent performance.

容易使用。OpenBox给出了用户友好的界面，可视化，资源管理，和自动算法选择模块，以得到一致的性能。

C3. High efficiency and scalability. We develop scalable and general frameworks for transfer-learning and distributed parallel execution in OpenBox. These building blocks are properly integrated to handle diverse optimization scenarios efficiently.

很高的效率和可扩展性。在OpenBox中，我们开发了可扩展的通用框架进行迁移学习，和分布式并行执行。这些构建模块得到了合适的集成，以处理各种优化场景。

C4. State-of-the-art performance. Our empirical evaluation demonstrates that OpenBox achieves state-of-the-art performance compared to existing systems over a wide range of BBO tasks.

目前最好的性能。我们的经验评估表明，与现有的系统在很多BBO任务上比较，OpenBox是目前最好的性能。

Moving Forward. With the above advantages and features, OpenBox can be used for optimizing a wide variety of different applications in an industrial setting. We are currently conducting an initial deployment of OpenBox in Kuaishou, one of the most popular “short video” platforms in China, to automate the tedious process of hyperparameter tuning. Initial results have suggested we can outperform human experts.

有了上面的优势和特点，OpenBox可以用于优化很多不同的应用。我们目前正在将OpenBox部署到快手中，以将超参数优化的过程自动化。初步的结果表明，这可以超过人类专家的结果。

## 2. Background and Related Work

**Generalized Black-box Optimization (BBO)**. Black-box optimization makes few assumptions about the problem, and is thus applicable in a wide range of scenarios. We define the generalized BBO problem as follows. The objective function of generalized BBO is a vector-valued black-box function 𝒇(𝒙) : X → R^𝑝, where X is the search space of interest. The goal is to identify the set of Pareto optimal solutions P∗ = {𝒇(𝒙) s.t. \nexists 𝒙′ ∈ X : 𝒇(𝒙′) ≺ 𝒇(𝒙)}, such that any improvement in one objective means deteriorating another. To approximate P∗, we compute the finite Pareto set P from observed data {(𝒙_𝒊,𝒚_𝒊)}^𝑛_{𝑖=1}. When 𝑝 = 1, the problem becomes single-objective BBO, as P = {𝑦_best} where 𝑦_best is defined as the best objective value observed. We also consider the case with black-box inequality constraints. Denote the set of feasible points by C = {𝒙 : 𝑐_1(𝒙) ≤ 0, . . . , 𝑐_𝑞(𝒙) ≤ 0}. Under this setting, we aim to identify the feasible Pareto set P_feas = {𝒇(𝒙) s.t. 𝒙 ∈ C, \nexists 𝒙′ ∈ X : 𝒇(𝒙′) ≺ 𝒇(𝒙), 𝒙′ ∈ C}.

通用BBO。黑箱优化对问题做的假设很少，因此在很广泛的场景中都可以使用。我们将通用BBO问题定义如下。通用BBO的目标函数是一个向量值的黑箱函数𝒇(𝒙) : X → R^𝑝，其中X是感兴趣的搜索空间。目标是找到Pareto最优解的集合P∗ = {𝒇(𝒙) s.t. \nexists 𝒙′ ∈ X : 𝒇(𝒙′) ≺ 𝒇(𝒙)}，这样一个目标的任何改进意味着另一个目标的恶化。为近似P*，我们从观察到的数据{(𝒙_𝒊,𝒚_𝒊)}^𝑛_{𝑖=1}中计算有限Pareto集合P。当p=1时，问题就变成了单目标BBO，其中P={𝑦_best}，这里{𝑦_best}定义为观察到的最好的目标值。我们还考虑有黑箱不等式约束的情况。将可行集的点表示为C = {𝒙 : 𝑐_1(𝒙) ≤ 0, . . . , 𝑐_𝑞(𝒙) ≤ 0}。在这种设置下，我们的目标是找到可行的Pareto集P_feas = {𝒇(𝒙) s.t. 𝒙 ∈ C, \nexists 𝒙′ ∈ X : 𝒇(𝒙′) ≺ 𝒇(𝒙), 𝒙′ ∈ C}。

**Black-box Optimization Methods**. Black-box optimization has been studied extensively in many fields, including derivative-free optimization [42], Bayesian optimization (BO) [43], evolutionaray algorithms [23], multi-armed bandit algorithms [31, 45], etc. To optimize expensive-to-evaluate black-box functions with as few evaluations as possible, OpenBox adopts BO, one of the most prevailing frameworks in BBO, as the basic optimization framework. BO iterates between fitting probabilistic surrogate models and determining which configuration to evaluate next by maximizing an acquisition function. With different choices of acquisition functions, BO can be applied to generalized BBO problems.

黑箱优化方法。黑箱优化在很多领域中得到了广泛的研究，包括无梯度优化，贝叶斯优化，演化算法，多臂老虎机算法，等。为用尽可能少的评估次数，来优化这些评估起来很耗时的黑箱函数，OpenBox采用了BO作为基本的优化框架，这是BBO中最流行的框架之一。BO不断的拟合概率代理模型，通过最大化采集函数来确定哪种配置在下一步进行评估。采集函数的选择不同，BO可以应用到通用BBO问题中。

BBO with Multiple Objectives. Many multi-objective BBO algorithms have been proposed [4, 5, 25, 29, 38]. Couckuyt et. al. [7] propose the Hypervolume Probability of Improvement (HVPOI); Yang et. al. [47] and Daulton et. al. [8] use the Expected Hypervolume Improvement (EHVI) metrics.

多目标BBO。提出了很多多目标BBO算法。Couckuyt等[7]提出了Hypervolume Probability of Improvement (HVPOI)；Yang等[47]和Daulton等[8]使用了Expected Hypervolume Improvement (EHVI)度量。

BBO with Black-box Constraints. Gardner et.al. [16] present Probability of Feasibility (PoF), which uses GP surrogates to model the constraints. In general, multiplying PoF with the unconstrained acquisition function produces the constrained version of it. SCBO [12] employs the trust region method and scales to large batches by extending Thompson sampling to constrained optimization. Other methods handle constraints in different ways [21, 24, 40]. For multiobjective optimization with constraints, PESMOC [17] and MESMOC [5] support constraints by adding the entropy of the conditioned predictive distribution.

带有黑箱约束的BBO。Gardner等[16]提出了Probability of Feasibility (PoF)，使用GP代理来对约束进行建模。一般来说，将PoF与无约束采集函数相乘，生成有约束的版本。SCBO[12]采用信任区域方法，通过将Thompson采样拓展到约束优化中，从而缩放到大的批次中。其他方法以不同的方式处理约束。对于带有约束的多目标优化，PESMOC和MESMOC支持约束，加入了有条件预测分布的熵。

**BBO Systems and Packages**. Many of these algorithms have available open-source implementations. BoTorch, GPflowOpt and HyperMapper implement several BO algorithms to solve mathematical problems in different settings. Within the machine learning community, Hyperopt, Spearmint, SMAC3 and HpBandSter aim to optimize the hyper-parameters of machine learning models. Google’s Vizier is one of the early attempts in building service for BBO. We also note that Facebook Ax provides high-level API for BBO with BoTorch as its Bayesian optimization engine.

BBO系统和包。很多算法都有开源的实现。BoTorch，GPflowOpt和HyperMapper实现了几种BO算法，来在不同的设置中求解数学问题。在机器学习团体中，Hyperopt，Spearmint，SMAC3和HpBandSter的目标是优化机器学习模型中的超参数。Google的Vizier是构建BBO服务的早期尝试。我们还指出，Facebook Ax为BBO提供了高层次API，BoTorch是其贝叶斯优化引擎。

## 3. System Overview

In this section, we provide the basic concepts in the paper, explore the design principles in implementing black-box optimization (BBO) as a service, and describe the system architecture.

本节中，我们给出了本文中的基本概念，探索在实现BBO服务中的设计原则，描述了系统架构。

### 3.1 Definitions

Throughout the paper, we use the following terms to describe the semantics of the system:

在本文中，我们使用下列项来描述系统的语义：

Configuration. Also called suggestion, a vector 𝒙 sampled from the given search space X; each element in 𝒙 is an assignment of a parameter from its domain.

配置。也称为建议，从给定的搜索空间X中采样得到的向量𝒙；𝒙中的每个元素，是在其领域中一个参数的指定值。

Trial. Corresponds to an evaluation of a configuration 𝒙, which has three status: Completed, Running, Ready. Once a trial is completed, we can obtain the evaluation result 𝒇(𝒙).

尝试。对应一个配置𝒙的一次评估，有三种状态：完成，运行，准备好。一旦一次尝试完成了，我们就可以得到评估结果𝒇(𝒙)。

Task. A BBO problem over a search space X. The task type is identified by the number of objectives and constraints.

任务。在一个搜索空间X中的一个BBO问题。任务类型按照目标和约束的数量来确定。

Worker. Refers to a process responsible for executing a trial. 负责执行一次尝试trial的过程。

### 3.2 Goals and Principles

3.2.1 Design Goal. As mentioned before, OpenBox’s design satisfies the following desiderata: 设计目标。前面提到过，OpenBox的设计满足下面的考虑：

• Ease of use. Minimal user effort, and user-friendly visualization for tracking and managing BBO tasks.

容易使用。用户的工作量最小化，用户友好的可视化，可以追踪并管理BBO任务。

• Consistent performance. Host state-of-the-art optimization algorithms; choose the proper algorithm automatically.

一致的性能。包含目前最好的优化算法；自动选择合适的算法。

• Resource-aware management. Give cost-model based advice to users, e.g., minimal workers or time-budget.

资源敏感的管理。给用户基于价格模型的建议，如，最少的workers或时间预算。

• Scalability. Scale to dimensions on the number of input variables, objectives, tasks, trials, and parallel evaluations.

可扩展性。输入变量的数量，目标的数量，任务的数量，trials的数量，和并行评估的数量，都要可以缩放。

• High efficiency. Effective use of parallel resources, system optimization with transfer-learning and multi-fidelities, etc.

高效。并行资源的高效使用，带有迁移学习和多保真度的系统优化，等。

• Fault tolerance, extensibility, and data privacy protection.

容错性，可拓展性，和数据隐私保护。

3.2.2 Design Principles. We present the key principles underlying the design of OpenBox.

设计原则。我们给出OpenBox设计的关键原则。

**P1: Provide convenient service API that abstracts the implementation and execution complexity away from the user**. For ease of use, we adopt the “BBO as a service” paradigm and implement OpenBox as a managed general service for black-box optimization. Users can access this service via REST API conveniently (see Figure 2), and do not need to worry about other issues such as environment setup, software maintenance, programming, and optimization of the execution. Moreover, we also provide a Web UI, through which users can easily track and manage the tasks.

原则1：给出方便的服务API，将实现和执行的复杂度抽象掉，不给用户看到。为容易使用，我们采用了将BBO做成服务的模式，将OpenBox实现为黑箱优化的通用服务。用户可以方便的通过REST API访问这些服务（见图2），不需要担心其他的问题，如环境设置，软件维护性，编程，和执行的优化。而且，我们还提供了一个网页UI，通过这些用户可以很容易的追踪和管理这些任务。

**P2: Separate optimization algorithm selection complexity away from the user**. Users do not need to disturb themselves with choosing the proper algorithm to solve a specific problem via the automatic algorithm selection module. Furthermore, an important decision is to keep our service stateless (see Figure 2), so that we can seamlessly switch algorithms during a task, i.e., dynamically choose the algorithm that is likely to perform the best for a particular task. This enables OpenBox to achieve satisfactory performance once the BBO algorithm is selected properly.

原则2：用户无需担心优化算法选择的复杂性。用户在求解特定问题时，可以通过自动算法选择模块来选择合适的算法。而且，一个重要的决定是，保持我们的服务是无状态的（见图2），所以我们可以无缝的在任务之间切换算法，即，动态的选择对特定的任务看起来是最好的算法。一旦选择了合适的BBO算法，OpenBox就可以获得令人满意的性能。

**P3: Support general distributed parallelization and transfer learning**. We aim to provide users with full potential to improve the efficiency of the BBO service. We design an “algorithm agnostic” mechanism that can parallelize the BBO algorithms (Sec. 5.1), through which we do not need to re-design the parallel version for each algorithm individually. Moreover, if the optimization history over similar tasks is provided, our transfer learning framework can leverage the history to accelerate the current task (Sec. 5.2).

原则3：支持一般的分布式并行化和迁移学习。我们的目标是给用户提供改进BBO服务的效率的完整可能性。我们设计一个算法无关的机制，使BBO算法并行化，通过这个机制，我们不需要重新设计每个算法的并行化版本。而且，如果给出了类似任务的优化历史，我们的迁移学习框架可以利用这个历史来加速当前的任务。

**P4: Offer resource-aware management that saves user expense**. OpenBox implements a resource-aware module and offers advice to users, which can save expense or resources for users especially in the cloud environment. Using performance-resource extrapolation (Sec. 4.4), OpenBox can estimate 1) the minimal number of workers users need to complete the current task within the given time budget, or 2) the minimal time budget to finish the current task given a fixed number of workers. For tasks that involve expensive-to-evaluate functions, low-fidelity or early-stopped evaluations with less cost could help accelerate the convergence of the optimization process (Sec. 5.3).

原则4：给出意识到资源的管理，节省用户的代价。OpenBox实现了一个意识到资源的模块，对用户给出建议，可以节省用户的代价或资源，尤其是在云环境中。使用性能-资源外插，OpenBox可以估计 1)在给定的时间预算下，完成当前任务所需的最少的worker数量；2)在给定数量的workers下，完成当前任务所需的最少时间预算。对于涉及到评估起来很耗时耗资源的函数的任务，低保真度或早停的评估，有较少的代价，会帮助加速优化过程的收敛。

### 3.3 System Architecture

Based on these design principles, we build OpenBox as depicted in Figure 2, which includes five main components. Service Master is responsible for node management, load balance, and fault tolerance. Task Database holds the states of all tasks. Suggestion Service creates new configurations for each task. REST API establishes the bridge between users/workers and suggestion service. Evaluation workers are provided and owned by the users.

基于这些设计原则，我们构建了OpenBox，如图2所示，这包括5个主要组成部分。Service Master负责节点管理，负载均衡，和容错。任务数据库保存所有任务的状态。建议服务对每个任务创建新的配置。REST API确定用户/workers和建议服务之间的桥梁。用户拥有并提供评估workers。

## 4. System Design

In this section, we elaborate on the main features and components of OpenBox from a service perspective.

本节中，我们给出OpenBox的主要特征和组成部分。

### 4.1 Service Interfaces

4.1.1 Task Description Language. For ease of usage, we design a Task Description Language (TDL) to define the optimization task. The essential part of TDL is to define the search space, which includes the type and bound for each parameter and the relationships among them. The parameter types — FLOAT, INTEGER, ORDINAL and CATEGORICAL are supported in OpenBox. In addition, users can add conditions of the parameters to further restrict the search space. Users can also specify the time budget, task type, number of workers, parallel strategy and use of history in TDL. Figure 3 gives an example of TDL. It defines four parameters x1-4 of different types and a condition cdn1, which indicates that x1 is active only if x3 = “a3”. The time budget is three hours, the parallel strategy is async, and transfer learning is enabled.

任务描述语言。为容易使用，我们设计了一种任务描述语言(TDL)来定义优化任务。TDL的基本部分是定义搜索空间，这包括每个参数的类型和界限，和其之间的关系。参数类型有，浮点，整数，序数和类别，在OpenBox中都支持。另外，用户可以对参数加入条件，以进一步限制搜索空间。用户还可以在TDL中指定时间预算，任务类型，workers数量，并行策略和使用历史。图3给出了TDL的一个例子，定义了4个参数x1-4，类型不同，和一个条件cdn1，表明只有在x3="a3"的情况，x1是激活的。时间预算是3个小时，并行策略是非同步的，迁移学习是使能的。

4.1.2 Basic Workflow. Given the TDL for a task, the basic workflow of OpenBox is implemented as follows: 给定了一个任务的TDL，OpenBox的基本工作流实现如下：

```

# Register the worker with a task.

global_task_id = worker. CreateTask(task_tdl)

worker. BindTask(global_task_id)

while not worker. TaskFinished ():

# Obtain a configuration to evaluate.

config = worker. GetSuggestions ()

# Evaluate the objective function.

result = Evaluate (config)

# Report the evaluated results to the server.

worker. UpdateObservations (config, result)

```

Here Evaluate is the evaluation procedure of objective function provided by users. By calling CreateTask, the worker obtains a globally unique identifier global_task_id. All workers registered with the same global_task_id are guaranteed to link with the same task, which enables parallel evaluations. While the task is not finished, the worker continues to call GetSuggestions and UpdateObservations to pull suggestions from the suggestion service and update their corresponding observations.

这里Evaluate是目标函数的评估过程，由用户给出。通过调用CreateTask，worker得到了一个全局唯一的标志符global_task_id。所有注册为相同的global_task_id的worker保证都连接到相同的任务，这确保了并行评估。在任务没有完成时，worker持续调用GetSuggestions和UpdateObservations，以从建议服务中拉取建议，更新其对应的观察。

4.1.3 Interfaces. Users can interact with the OpenBox service via a REST API. We list the most important service calls as follows:

接口。用户可以通过REST API来与OpenBox服务进行交互。我们列出了最重要的服务调用如下：

• Register: It takes as input the global_task_id, which is created when calling CreateTask from workers, and binds the current worker with the corresponding task. This allows for sharing the optimization history across multiple workers.

注册：以global_task_id为输入，这是当从workers调用CreateTask时创建的，将当前的worker与对应的任务绑定。这可以在多个worker中共享优化历史。

• Suggest: It suggests the next configurations to evaluate, given the historical observations of the current task.

建议：在给定当前任务的历史观察下，这建议下一个配置来进行评估。

• Update: This method updates the optimization history with the observations obtained from workers. The observations include three parts: the values of the objectives, the results of constraints, and the evaluation information.

更新：这个方法用workers得到的观察更新优化历史。观察包括三部分：目标的值，约束的结果，评估信息。

• StopEarly: It returns a boolean value that indicates whether the current evaluation should be stopped early.

早停：返回一个bool值，表示当前的评估是否应当早停。

• Extrapolate: It uses performance-resource extrapolation, and interactively gives resource-aware advice to users.

外插：使用性能资源外插，互动的向用户给出资源敏感的建议。

### 4.2 Automatic Algorithm Selection

OpenBox implements a wide range of optimization algorithms to achieve high performance in various BBO problems. Unlike the existing software packages that use the same algorithm for each task and the same setting for each algorithm, OpenBox chooses the proper algorithm and setting according to the characteristic of the incoming task. We use the classic EI [37] for single-objective optimization task. For multi-objective problems, we select EHVI [11] when the number of objectives is less than 5; we use MESMO [4] algorithm for problems with a larger number of objectives, since EHVI’s complexity increases exponentially as the number of objectives increases, which not only incurs a large computational overhead but also accumulates floating-point errors. We select the surrogate models in BO depending on the configuration space and the number of trials: If the input space has conditions, such as one parameter must be less than another parameter, or there are over 50 parameters in the input space, or the number of trials exceeds 500, we choose the Probabilistic Random Forest proposed in [27] instead of Gaussian Process (GP) as the surrogate to avoid incompatibility or high computational complexity of GP. Otherwise, we use GP [10]. In addition, OpenBox will use the L-BFGS-B algorithm to optimize the acquisition function if the search space only contains FLOAT and INTEGER parameters; it applies an interleaved local and random search when some of the parameters are not numerical. More details about the algorithms implemented in OpenBox are discussed in Appendix A.2.

OpenBox实现了很多优化算法，以在各种BBO问题中获得很好的性能。现有的软件包对每个任务使用相同的算法，对每个算法使用相同的设置，与此不同的是，OpenBox根据任务的特点选择合适的算法和配置。我们对单目标优化任务使用经典的EI[37]。对多目标问题，在目标数量少于5时，我们选择EHVI[11]；对于目标数量更大的问题，我们使用MESMO[4]算法，因为随着目标数量的增长，EHVI的复杂度呈指数级增长，不仅带来大量计算代价，而且会累积浮点误差。我们根据配置空间和trials的数量，在BO中选择代理模型：如果输入空间有条件，比如一个参数必须小于另一个参数，或在输入空间中有超过50个参数，或trials的数量超过了500，我们选择[27]中提出的概率随机森林，替换高斯过程作为代理，以避免与GP不兼容，或复杂度太高。否则，我们使用GP。另外，如果搜索空间包含浮点和整数参数，OpenBox会使用L-BFGS-B算法来优化采集函数；在一些参数不是数值参数时，会实行一个交叉的局部和随机搜索。关于在OpenBox中实现的算法的更多的细节，我们在附录A.2中讨论。

### 4.3 Parallel Infrastructure

OpenBox is designed to generate suggestions for a large number of tasks concurrently, and a single machine would be insufficient to handle the workload. Our suggestion service is therefore deployed across several machines, called suggestion servers. Each suggestion server generates suggestions for several tasks in parallel, giving us a massively scalable suggestion infrastructure. Another main component is service master, which is responsible for managing the suggestion servers and balancing the workload. It serves as the unified endpoint, and accepts the requests from workers; in this way, each worker does not need to know the dispatching details. The worker requests new configurations from the suggestion server and the suggestion server generates these configurations based on an algorithm determined by the automatic algorithm selection module. Concretely, in this process, the suggestion server utilizes the local penalization based parallelization mechanism (Sec. 5.1) and transfer-learning framework (Sec. 5.2) to improve the sample efficiency.

OpenBox设计用于同时对大量任务来生成建议，一台机器不足以用于处理workload。我们的建议服务因此是在几台机器中部署的，称为建议服务器。每个建议服务器对几个任务并行生成建议，给了我们一个巨大的可扩展的建议基础设施。另一个主要组成部分是，服务master，负责管理建议服务器，均衡workload。它的作用是统一的endpoint，接受workers的请求；这样，每个worker不需要知道调度细节。worker从建议服务器上请求新的配置，建议服务器生成这些配置基于自动算法选择模块确定的算法。具体的，在这个过程中，建议服务器利用基于局部惩罚的并行机制和迁移学习框架，来改进采样效率。

One main design consideration is to maintain a fault-tolerant production system, as machine crash happens inevitably. In OpenBox, the service master monitors the status of each server and preserves a table of active servers. When a new task comes, the service master will assign it to an active server and record this binding information. If one server is down, its tasks will be dispatched to a new server by the master, along with the related optimization history stored in the task database. Load balance is one of the most important guidelines to make such task assignments. In addition, the snapshot of service master is stored in the remote database service; if the master is down, we can recover it by restarting the node and fetching the snapshot from the database.

一个主要的设计考虑是，维护一个容错生产系统，因为机器不可避免的会崩溃。在OpenBox中，服务器master监控每个服务器的状态，保留活跃服务器的表格。当新任务到来时，服务master会将其指定到一个活跃的服务器，记录这个绑定信息。如果一个服务器down机，其任务会由master指派到一个新的服务器，以及相关的优化历史，存储在任务数据库中。负载均衡是一个最重要的指引来进行这样的任务指定。另外，服务master的快照存储在远程数据库服务中；如果master down机，我们可以通过重启这个节点，从数据库中取回快照，来进行恢复。

### 4.4 Performance-Resource Extrapolation

In the setting of parallel infrastructure with cloud computing, saving expense is one of the most important concerns from users. OpenBox can guide users to configure their resources, e.g., the minimal number of workers or time budget, which further saves expense for users. Concretely, we use a weighted cost model to extrapolate the performance vs. trial curve. It uses several parametric decreasing saturating function families as base models, and we apply MCMC inference to estimate the parameters of the model. Given the existing observations, OpenBox trains a cost model as above and uses it to predict the number of trials at which the curve approaches the optimum. Based on this prediction and the cost of each evaluation, OpenBox estimates the minimal resource needed to reach satisfactory performance (more details in Appendix A.1).

在云计算的并行基础设施的设置下，节约花费是用户一个最重要的关切。OpenBox可以引导用户来配置其资源，如，workers或时间预算的最小值，这进一步为用户节约了代价。具体的，我们使用了一个加权代价模型，来外插性能vs trial曲线。它使用几个参数化的下降的饱和函数族，作为基础模型，我们使用MCMC推理来估计模型的参数。给定现有的观察，OpenBox训练了一个代价模型，用于预测曲线达到最优的数量时trials的数量。基于这些预测和每次评估的代价，OpenBox估计需要的最小资源来达到另一满意的性能。

Application Example. Two interesting applications that save expense for users are listed as follows:

应用例子。两个有趣的为用户节约代价的应用如下：

Case 1. Given a fixed number of workers, OpenBox outputs a minimal time budget 𝐵min to finish this task based on the estimated evaluation cost of workers. With this estimation, users can stop the task in advance if the given time budget 𝐵task > 𝐵min; otherwise, users should increase the time budget to 𝐵min.

案例1。给定固定数量的workers，OpenBox输出了一个最小时间预算Bmin，来完成这个任务，基于估计的workers的代价评估。有了这个估计，用户可以提前停止这个任务，如果给定的时间预算Btask>Bmin；否则，用户应当增加时间预算到Bmin。

Case 2. Given a fixed time budget 𝐵task and initial number of workers, OpenBox can suggest the minimal number of workers 𝑁min to finish the current task within 𝐵task by adjusting the number of workers to 𝑁min dynamically.

案例2。给定固定的时间预算Btask和初始数量的workers，OpenBox可以建议workers的最小数量Nmin来在Btask内完成目前的任务，动态的调整workers的数量到Nmin。

### 4.5 Augmented Components in OpenBox

Extensibility and Benchmark Support. OpenBox’s modular design allows users to define their suggestion algorithms easily by inheriting and implementing an abstract Advisor. The key abstraction method of Advisor is GetSuggestions, which receives the observations of the current task and suggests the next configurations to evaluate based on the user-defined policy. In addition, OpenBox provides a benchmark suite of various BBO problems to benchmark the optimization algorithms.

可拓展性和基准测试支持。OpenBox的模块化设计，使用户可以很容易的定义其建议，只要继承和实现一个抽象的Advisor。Advisor的关键抽象方法是GetSuggestions，接收当前任务的观察，根据用户定义的策略来建议要评估的下一个配置。此外，OpenBox给出了各种BBO问题的一个基准测试包，以测试优化算法。

Data Privacy Protection. In some scenarios, the names and ranges of parameters are sensitive, e.g., in hyper-parameter tuning, the parameter names may reveal the architecture details of neural networks. To protect data privacy, the REST API applies a transformation to anonymize the parameter-related information before sending it to the service. This transformation involves 1) converting the parameter names to some regular ones like “param1” and 2) rescaling each parameter to a default range that has no semantic. The workers can perform an inverse transformation when receiving an anonymous configuration from the service.

数据隐私保护。在一些场景中，参数的名称和范围是敏感的，如，在超参数调节中，参数名称可能会暴露神经网络的架构细节。为保护数据隐私，REST API应用了一个变换，来将参数相关的信息匿名化，然后再送到服务中去。这种变换包换包括，1)将参数名称变换到一些常规名称，如param1，2)将每个参数重新缩放到一个默认的范围，没有任何语义。workers在从服务中接收到匿名配置后，进行一个逆变换。

Visualization. OpenBox provides an online dashboard based on TensorBoardX which enables users to monitor the optimization process and check the evaluation info of the current task. Figure 4 visualizes the evaluation results in a hyper-parameter tuning task.

可视化。OpenBox给出了一个基于TensorFlowBoardX的在线的仪表盘，使用户可以监控优化过程，检查当前任务的评估信息。图4对超参数调节任务的评估结果进行了可视化。

## 5. System Optimizations

### 5.1 Local Penalization based Parallelization

Most proposed Bayesian optimization (BO) approaches only allow the exploration of the parameter space to occur sequentially. To fully utilize the computing resources in a parallel infrastructure, we provide a mechanism for distributed parallelization, where multiple configurations can be evaluated concurrently across workers. Two parallel settings are considered (see Figure 5):

多数提出的贝叶斯优化方法只允许顺序探索参数空间。为完全利用计算资源的并行基础设施，我们提供了一种机制进行分布式并行化，其中可以在多个workers中同时评估多个配置。考虑了两种并行的设置（见图5）：

1) Synchronous parallel setting. The worker pulls new configuration from suggestion server to evaluate until all the workers have finished their last evaluations.

同步并行设置。workers从建议服务器上拉下来新的配置进行评估，直到所有workers完成其最后的评估。

2) Asynchronous parallel setting. The worker pulls a new configuration when the previous evaluation is completed.

异步并行设置。worker在前一个评估完成后，就拉下来新的配置。

Our main concern is to design an algorithm-agnostic mechanism that can parallelize the optimization algorithms under the sync and async settings easily, so we do not need to implement the parallel version for each algorithm individually. To this end, we propose a local penalization based parallelization mechanism, the goal of which is to sample new configurations that are promising and far enough from the configurations being evaluated by other workers. This mechanism can handle the well-celebrated exploration vs. exploitation trade-off, and meanwhile prevent workers from exploring similar configurations. Algorithm 1 gives the pseudo-code of sampling a new configuration under the sync/async settings. More discussion about this is provided in Appendix A.4.

我们的主要考虑是，设计一个算法无关的机制，可以在sync和async的设置下，很容易的并行化优化算法，这样我们就不需要单独实现每个算法的并行版本。为此，我们提出了一种基于局部惩罚的并行机制，其目的是，采样新的有希望的配置，与其他workers评估的配置距离很远。这种机制可以处理著名的探索vs利用折中，同时防止workers探索类似的配置。算法1给出了在sync/async的配置下采样一个新配置的伪代码。更多的讨论在附录A.4中给出。

### 5.2 General Transfer-Learning Framework

When performing BBO, users often run tasks that are similar to previous ones. This fact can be used to speed up the current task. Compared with Vizier, which only provides limited transfer learning functionality for single-objective BBO problems, OpenBox employs a general transfer learning framework with the following advantages: 1) support for the generalized black-box optimization problems, and 2) compatibility with most BO methods.

当进行BBO时，用户运行的任务通常会与之前的任务类似。这个事实可以用于加速当前的任务。Vizier对单目标BBO问题提供的迁移学习功能很有限，OpenBox采用了一种通用的迁移学习框架，有以下优势：1)支持通用BBO问题，2)与多数BO方法兼容。

OpenBox takes as input observations from 𝐾 + 1 tasks: 𝐷^1, ..., 𝐷^𝐾 for 𝐾 previous tasks and 𝐷^𝑇 for the current task. Each 𝐷^𝑖 ={(𝒙^𝑖_𝑗, 𝒚^𝑖_𝑗)}^{𝑛_𝑖}_{𝑗=1}, 𝑖 = 1, ..., 𝐾, includes a set of observations. Note that, 𝒚 is an array, including multiple objectives for configuration 𝒙.

OpenBox以K+1个任务的观察为输入：𝐷^1, ..., 𝐷^𝐾，这是K个之前的任务，𝐷^𝑇是当前的任务。每个𝐷^𝑖 ={(𝒙^𝑖_𝑗, 𝒚^𝑖_𝑗)}^{𝑛_𝑖}_{𝑗=1}, 𝑖 = 1, ..., 𝐾，包含一个观察集合。注意，𝒚是一个阵列，包含配置𝒙的多个目标。

For multi-objective problems with 𝑝 objectives, we propose to transfer the knowledge about 𝑝 objectives individually. Thus, the transfer learning of multiple objectives is turned into 𝑝 single-objective transfer learning processes. For each dimension of the objectives, we take RGPE [14] as the base method. 1) We first train a surrogate model 𝑀^𝑖 on 𝐷^𝑖 for the 𝑖-𝑡ℎ prior task and 𝑀^𝑇 on 𝐷^𝑇; based on 𝑀^{1:𝐾} and 𝑀^𝑇, 2) we then build a transfer learning surrogate by combining all base surrogates:

对𝑝个目标的多目标问题，我们提出逐个迁移p个目标的知识。因此，多目标迁移学习就转变成了p个单目标的迁移学习过程。对目标的每个维度，我们以RGPE[14]为基准方法。1)我们首先在𝐷^𝑖上对第i个先验任务训练一个代理模型𝑀^𝑖，并在𝐷^𝑇上训练𝑀^𝑇，基于𝑀^{1:𝐾}和𝑀^𝑇，2)我们然后构建一个迁移学习代理，将所有这些代理结合到一起

$$𝑀^{TL} = agg({𝑀^1, ..., 𝑀^𝐾, 𝑀^𝑇}; w)$$

3) the surrogate 𝑀^{TL} is used to guide the configuration search, instead of the original 𝑀^𝑇. Concretely, we combine the multiple base surrogates (agg) linearly, and the parameters w are calculated based on the ranking of configurations, which reflects the similarity between the source and target task (see details in Appendix A.3).

代理𝑀^{TL}用于替代原始的𝑀^𝑇引导配置搜索。具体的，我们将多个基础代理agg线性的结合起来，参数w是基于配置的ranking来计算得到的，反应了源任务和目标任务的相似性。

Scalability discussion A more intuitive alternative is to obtain a transfer learning surrogate by using all observations from 𝐾 + 1 tasks, and this incurs a complexity of O(𝑘^3𝑛^3) for 𝑘 tasks with 𝑛 trials each (since GP has O(𝑛^3) complexity). Therefore, it is hard to scale to a larger number of source tasks (a large 𝑘). By training base surrogates individually, the proposed framework is a more computation-efficient solution that has O(𝑘𝑛^3) complexity.

可扩展性讨论。一个更直观的替代是，使用K+1个任务的所有观察，得到一个迁移学习代理，对于k个任务，每个任务n次trial，这带来的复杂度为O(𝑘^3𝑛^3)。因此，很难对大量的源任务进行缩放。通过单独训练基础代理，提出的框架在计算上更加高效，复杂度为O(𝑘𝑛^3)。

### 5.3 Additional Optimizations

OpenBox also includes two additional optimizations that can be applied to improve the efficiency of black-box optimizations.

OpenBox还有两个额外的优化，可以用于改进BBO的效率。

5.3.1 Multi-Fidelity Support and Applications. During each evaluation in the multi-fidelity setting [33, 41], the worker receives an additional parameter, indicating how many resources are used to evaluate this configuration. The resource type needs to be specified by users. For example, in hyper-parameter tuning, it can be the number of iterations for an iterative algorithm and the size of dataset subset. The trial with partial resource returns a low-fidelity result with a cheap evaluation cost. Though not as precise as high-fidelity results, the low-fidelity results can provide some useful information to guide the configuration search. In OpenBox, we have implemented several multi-fidelity algorithms, such as MFES-HB [33].

多保真度支持和应用。在多保真度设置下，在每次评估过程中，worker收到一个额外的参数，表明用了多少资源来评估这个配置。资源类型要由用户指定。比如，在超参数调节中，可以是迭代算法的迭代次数，和数据集子集的大小。用部分资源的trial，会返回一个低保真度的结果，评估代价也是较廉价的。虽然不像高保真度的结果那么精确，低保真度的结果会提供一些有用的信息，来引导配置搜索。在OpenBox中，我们实现了几个多保真度算法，如MFES-HB。

5.3.2 Early-Stopping Strategy. Orthogonal to the above optimization, early-stopping strategies aim to stop a poor trial in advance based on its intermediate results. In practice, a worker can periodically ask suggestion service whether it should terminate the current evaluation early. In OpenBox, we provide two early-stopping strategies: 1) learning curve extrapolation based methods [9, 28] that stop the poor configurations by estimating the future performance, and 2) mean or median termination rules based on comparing the current result with previous ones.

早停策略。与上面的优化同时，早停的策略的目标是，基于其中间结果，提前停止一个很差的trial。在实践中，worker会周期性的向建议服务中查询，是否应当早停当前的评估。在OpenBox中，我们提供了两个早停策略：1)学习基于曲线外插方法，通过估计未来的性能，来停止很差的配置；2)将当前的结果与之前的进行对比，平均或中值停止准则。

## 6. Experimental Evaluation

In this section, we compare the performance and efficiency of OpenBox against existing software packages on multiple kinds of blackbox optimization tasks, including tuning tasks in AutoML.

在本节中，我们与现有的软件包进行对比，在多种黑箱优化任务上，比较了OpenBox的性能和效率，包括AutoML中的调节任务。

### 6.1 Experimental Setup

6.1.1 Baselines. Besides the systems mentioned in Table 1, we also use CMA-ES [23], Random Search and 2×Random Search (Random Search with double budgets) as baselines. To evaluate transfer learning, we compare OpenBox with Google Vizier. For multi-fidelity experiments, we compare OpenBox against HpBandSter and BOHB, the details of which are in Appendix A.5.

基准。除了表1中描述的系统，我们还使用CMA-ES，随机搜索和2×随机搜索（双倍预算的随机搜索）作为基准。为评估迁移学习，我们将OpenBox与Google Vizier进行了比较。对多保真度试验，我们比较将OpenBox与HpBandSter和BOHB进行了比较，详见附录A.5。

6.1.2 Problems. We use 12 black-box problems (mathematical functions) from [50] and two AutoML optimization problems on 25 OpenML datasets. In particular, 2d-Branin, 2d-Beale, 6d-Hartmann and (2d, 4d, 8d, 16d, 32d)-Ackley are used for single-objective optimization; 2d-Townsend, 2d-Mishra, 4d-Ackley and 10d-Keane are used for constrained single-objective optimization; 3d-ZDT2 with two objectives and 6d-DTLZ1 with five objectives are used for multi-objective optimization; 2d-CONSTR and 2d-SRN with two objectives are used for constrained multi-objective optimization. All the parameters for mathematical problems are of the FLOAT type and the maximum trials of each problem depend on its difficulty, which ranges from 80 to 500. For AutoML problems on 25 datasets, we split each dataset and search for the configuration with the best validation performance. Specifically, we tune LightGBM and LibSVM with the linear kernel, where the parameters of LightGBM are of the FLOAT type while LibSVM contains CATEGORICAL and conditioned parameters.

问题。我们使用[50]中的12个黑箱问题（数学函数），和在25个OpenML数据集中的两个AutoML优化问题。特别是，2d-Branin, 2d-Beale, 6d-Hartmann和(2d, 4d, 8d, 16d, 32d)-Ackley用于单目标优化；2d-Townsend, 2d-Mishra, 4d-Ackley和10d-Keane用于约束单目标优化；带有2个目标的3d-ZDT2，和5个目标的6d-DTLZ1，用于多目标优化；两个目标的2d-CONSTR和2d-SRN用于约束多目标优化。数学问题的所有参数都是FLOAT类型，每个问题的最大trials次数依赖于其难度，范围在80到500。对于在25个数据集上的AutoML问题，我们将每个数据集分割，搜索最佳验证性能的配置。具体的，我们用线性核来调节LightGBM和LibSVM，其中LightGBM的参数是FLOAT类型的，而LibSVM包含CATEGORICAL和有条件的参数。

6.1.3 Metrics. We employ the three metrics as follows. 度量。我们采用下面三种度量：

1. Optimality gap is used for single-objective mathematical problem. That is, if 𝑥∗ optimizes 𝑓, and \hat 𝑥 is the best configuration found by the method, then |𝑓(\hat 𝑥) − 𝑓(𝑥∗)| measures the success of the method on that function. In rare cases, we report the objective value if the ground-truth optimal 𝑥∗ is extremely hard to obtain.

优化性空白，用于单目标数学问题。即，如果𝑥∗使f最优化，\hat x是算法发现的最优配置，那么|𝑓(\hat 𝑥) − 𝑓(𝑥∗)|度量的是该算法在这个函数上的成功程度。在很少的情况中，如果真值最优𝑥∗极其难以得到，我们给出目标值。

2. Hypervolume indicator given a reference point 𝒓 measures the quality of a Pareto front in multi-objective problems. We report the difference between the hypervolume of the ideal Pareto front P∗ and that of the estimated Pareto front P by a given algorithm, which is 𝐻𝑉 (P∗,𝒓) − 𝐻𝑉 (P,𝒓).

Hypervolume indicator，在给定一个参考点𝒓，度量的是多目标问题中的Pareto front的质量。我们用一个给定的算法，衡量出理想Pareto front P*和估计的Pareto front P之间的hypervolume的差异，即𝐻𝑉 (P∗,𝒓) − 𝐻𝑉 (P,𝒓)。

3. Metric for AutoML. For single-objective AutoML problems, we report the validation error. To measure the results across different datasets, we use Rank as the metric.

AutoML的度量。对于单目标AutoML问题，我们给出验证误差。为度量不同数据集之间的结果，我们使用Rand作为度量。

6.1.4 Parameter Settings. For both OpenBox and the considered baselines, we use the default setting. Each experiment is repeated 10 times, and we compute the mean and variance for visualization.

参数设置。对于OpenBox和考虑的基准，我们使用默认设置。每个试验重复10次，我们计算出均值和方差，以进行可视化。

### 6.2 Results and Analysis

6.2.1 Single-Objective Problems without Constraints. Figure 6 illustrates the results of OpenBox on different single-objective problems compared with competitive baselines while Figure 7 displays the performance with the growth of input dimensions. In particular, Figure 6 shows that OpenBox, HyperMapper and BoTorch are capable of optimizing these low-dimensional functions stably. However, when the dimensions of the parameter space grow larger, as shown in Figure 7, only OpenBox achieves consistent and excellent results while the other baselines fail, which demonstrates its scalability on input dimensions. Note that, OpenBox achieves more than 10-fold speedups over the baselines when solving Ackley with 16 and 32-dimensional inputs.

没有约束的单目标问题。图6给出了OpenBox在不同的单目标问题上的结果，并与一些基准进行了对比；图7给出了随着输入维度增长，性能的变化。特别是，图6展示了OpenBox，HyperMapper，和BoTorch可以很稳定的优化低维问题。但是，当参数空间的维度变得越来越大，如图7所示，只有OpenBox获得了一致的优秀结果，而其他基准则没有，这说明了OpenBox对输入维度的可扩展性。注意，OpenBox在处理16维和32维的Ackley问题时，比基准获得了超过10倍的加速。

6.2.2 Single-Objective Problems with Constraints. Figure 8 shows the results of OpenBox along with the baselines on four constrained single-objective problems. Besides Random Search, we compare OpenBox with three of the software packages that support constraints. OpenBox surpasses all the considered baselines on the convergence result. Note that on the 10-dimensional Keane problem in which the ground-truth optimal value is hard to locate, OpenBox is the only method that successfully optimizes this function while the other methods fail to suggest sufficient feasible configurations.

有约束的单目标问题。图8展示了OpenBox与基准在4个有约束单目标问题上的结果。除了随机搜索，我们比较OpenBox与其他三个支持约束的软件包。OpenBox在收敛结果上超过了所有考虑的基准。注意在10维Keane问题上，其中真值最优值很难定位，OpenBox是唯一的一种成功的最优化了这个函数的方法，而其他方法没有得到足够可行的配置。

6.2.3 Multi-Objective Problems without Constraints. We compare OpenBox with three baselines that support multiple objectives and the results are depicted in Figure 9(a) and 9(b). In Figure 9(a), the hypervolume difference of GPflowOpt and Hypermapper decreases slowly as the number of trials grow, while BoTorch and OpenBox obtain a satisfactory Pareto Front quickly within 50 trials. In Figure 9(b) where the number of objectives is 5, BoTorch meets the bottleneck of optimizing the Pareto front while OpenBox tackles this problem easily by switching its inner algorithm from EHVI to MESMO; GPflowOpt is missing due to runtime errors.

没有约束的多目标问题。我们比较OpenBox与三个支持多目标的基准，结果如图9a和9b所示。在图9a中，GPflowOpt和HyperMapper的hypervolume差异随着trials的数量增加，逐步缓慢降低，而BoTorch和OpenBox则很快的在50个trials以内得到了一个满意的Pareto Front。在图9b中，目标的数量为5，BoTorch遇到了优化Pareto front的瓶颈，而OpenBox则很容易的处理这个问题，将其内部算法从EVHI切换到了MESMO；GPflowOpt由于有运行时错误，没有画出。

6.2.4 Multi-Objective Problems with Constraints. We compare OpenBox with Hypermapper and BoTorch on constrained multi-objective problems (See Figure 9(c) and 9(d)). Figure 9(c) demonstrates the performance on a simple problem, in which the convergence result of OpenBox is slightly better than the other two baselines. However, in Figure 9(d) where the constraints are strict, BoTorch and Hypermapper fail to suggest sufficient feasible configurations to update the Pareto Front. Compared with BoTorch and Hypermapper, OpenBox has more stable performance when solving multi-objective problems with constraints.

带有约束的多目标问题。我们在带约束的多目标问题上比较OpenBox与HyperMapper和BoTorch，如图9c和9d所示。图9c在简单问题上展示了性能，其中OpenBox的收敛结果比其他两个基准要略微好一些。但是，在图9d中，其中约束更加严格，BoTorch和HyperMapper没有给出足够可行的配置以更新Pareto front。与BoTorch和HyperMapper相比，OpenBox在求解带约束的多目标问题中时，有更稳定的性能。

### 6.3 Results on AutoML Tuning Tasks

6.3.1 AutoML Tuning on 25 OpenML datasets. Figure 11 demonstrates the universality and stability of OpenBox in 25 AutoML tuning tasks. We compare OpenBox with SMAC3 and Hyperopt on LibSVM since only these two baselines support CATEGORICAL parameters with conditions. In general, OpenBox is capable of handling different types of input parameters while achieving the best median performance among the baselines considered.

在25个OpenML数据集上的AutoML Tuning。图11展示了OpenBox在25个AutoML tuning任务上的统一性和稳定性。我们将OpenBox与SMAC3和Hyperopt在LibSVM上进行比较，因为只有这两个基准支持带条件的CATEGORICAL参数。一般来说，OpenBox可以处理不同的输入参数类型，同时在考虑的基准上获得最好的中值性能。

6.3.2 Parallel Experiments. To evaluate OpenBox with parallel settings, we conduct an experiment to tune the hyper-parameters of LightGBM on Optdigits with a budget of 600 seconds. Figure 11(a) shows the average validation error with different parallel settings. We observe that the asynchronous mode with 8 workers achieves the best results and outperforms Random Search with 8 workers by a wide margin. It brings a speedup of 8× over the sequential mode, which is close to the ideal speedup. In addition, although the synchronous mode brings a certain improvement over the sequential mode in the beginning, the convergence result is usually worse than the asynchronous mode due to stragglers.

并行试验。为在并行的设置下评估OpenBox，我们进行了一个试验，在Optdigits上用600s的预算上对LightGBM进行超参数调节。图11a是在不同的并行设置下的平均验证误差。我们观察到，有8个workers的异步模式获得了最好的结果，比8个workers的随机搜索好了很多。比顺序模式带来了8x的加速，与理想的加速接近。此外，虽然同步模式在开始时比顺序模式有一定的改进，收敛结果通常比异步模式要差，因为有stragglers。

6.3.3 Transfer Learning Experiment. In this experiment, we remove all baselines except Vizier, which provides the transfer learning functionality for the traditional black-box optimization. We also add SMAC3 that provides a non-transfer reference. In addition, this experiment involves tuning LightGBM on 25 OpenML datasets, and it is performed in a leave-one-out fashion, i.e, we tune the hyperparameters of LightGBM on a dataset (target problem), while taking the tuning history on the remaining datasets as prior observations. Figure 11(b) shows the average rank for each baseline. We observe that 1) Vizier and OpenBox show improved sample efficiency relative to SMAC3 that cannot use prior knowledge from source problems, and 2) the proposed transfer learning framework in OpenBox performs better than the transfer learning algorithm used in Vizier. Furthermore, it is worth mentioning that OpenBox also supports transfer learning for the generalized black-box optimization, while Vizier does not.

迁移学习的试验。在这个试验中，我们去除了所有的基准，只留下了Vizier，因为它对传统的BBO提供了迁移学习的功能。我们还加入了SMAC3，给出了一个非迁移的参考。此外，这个试验是在25个OpenML数据集上调节LightGBM，而且是以leave-one-out的方式进行的试验，即，我们调节LightGBM在一个数据集上的超参数（目标问题），而以在其余数据集上的调节历史作为先验观察。图11b展示了每个baseline的平均rank。我们观察到，1)Vizier和OpenBox，比SMAC3，采样效率得到了改进，因为SMAC3不能利用先验知识，2)提出的在OpenBox中的迁移学习框架，比Vizier中的迁移学习算法效果要好。而且，值得注意的是，OpenBox还支持通用BBO中的迁移学习，而Vizier则不支持。

## 7 Conclusion

In this paper, we have introduced a service that aims for solving generalized BBO problems – OpenBox, which is open-sourced and highly efficient. We have presented new principles from a service perspective that drive the system design, and we have proposed efficient frameworks for accelerating BBO tasks by leveraging local-penalization based parallelization and transfer learning. OpenBox hosts lots of state-of-the-art optimization algorithms with consistent performance, via adaptive algorithm selection. It also offers a set of advanced features, such as performance-resource extrapolation, multi-fidelity optimization, automatic early stopping, and data privacy protection. Our experimental evaluations have also showcased the performance and efficiency of OpenBox on a wide range of BBO tasks.

本文中，我们提出了一种服务OpenBox，求解通用BBO问题，开源而且高效。我们从服务的角度提出了新的原则，驱动了系统的设计，我们提出了高效的框架以加速BBO任务，利用了基于local-penalization的并行化和迁移学习。OpenBox包含了大量目前最好的优化算法，通过自动算法选择，达到一直很好的性能。它还支持很多高级特征，比如性能资源外插，多可信度优化，自动早停，数据隐私保护。我们的试验评估在很多BBO任务上展示了OpenBox的性能和效率。