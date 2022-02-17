# Google Vizier: A Service for Black-Box Optimization

Daniel Golovin et. al. @ Google Research

## 0. Abstract

Any sufficiently complex system acts as a black box when it becomes easier to experiment with than to understand. Hence, black-box optimization has become increasingly important as systems have become more complex. In this paper we describe Google Vizier, a Google-internal service for performing black-box optimization that has become the de facto parameter tuning engine at Google. Google Vizier is used to optimize many of our machine learning models and other systems, and also provides core capabilities to Google’s Cloud Machine Learning HyperTune subsystem. We discuss our requirements, infrastructure design, underlying algorithms, and advanced features such as transfer learning and automated early stopping that the service provides.

任何充分复杂的系统，在其试验比理解起来更容易的时候，都会成为一个黑箱。因此，当系统变得越来越复杂，黑箱优化就变得越来越重要。本文中，我们描述了进行黑箱优化的Google Vizier内部服务，这已经成为了Google实际上的调参引擎。Google Vizier用于优化很多机器学习模型和其他系统，为Google的云机器学习HyperTune子系统提供了核心能力。我们讨论了我们的需求，基础设施设计，潜在的算法，和高级特征，比如迁移学习和自动早停，这些服务都提供了这样的功能。

**KEYWORDS** Black-Box Optimization, Bayesian Optimization, Gaussian Processes, Hyperparameters, Transfer Learning, Automated Stopping

## 1. Introduction

Black–box optimization is the task of optimizing an objective function f : X → R with a limited budget for evaluations. The adjective “black–box” means that while we can evaluate f(x) for any x ∈ X, we have no access to any other information about f , such as gradients or the Hessian. When function evaluations are expensive, it makes sense to carefully and adaptively select values to evaluate; the overall goal is for the system to generate a sequence of x_t that approaches the global optimum as rapidly as possible.

黑箱优化的任务是在有限的评估预算下，优化目标函数f : X → R。形容词黑箱的意思是，我们可以对任何x ∈ X评估得到f(x)，但是我们没有任何关于f的其他信息，比如梯度，或Hessian矩阵。当函数评估很昂贵时，仔细的、自适应的选择一些值来评估，就变得有意义了；系统的总体目标是，生成x_t的序列，尽可能快的达到全局最优。

Black box optimization algorithms can be used to find the best operating parameters for any system whose performance can be measured as a function of adjustable parameters. It has many important applications, such as automated tuning of the hyperparameters of machine learning systems (e.g., learning rates, or the number of hidden layers in a deep neural network), optimization of the user interfaces of web services (e.g. optimizing colors and fonts to maximize reading speed), and optimization of physical systems (e.g., optimizing airfoils in simulation).

黑箱优化算法可以用于对任何系统寻找最佳操作参数，其中系统的性能可能用一些可调节的参数的函数来度量。这有很多重要应用，比如自动调节机器学习系统的超参数（如，学习速率，深度神经网络中隐藏层的数量），网络服务的用户界面的优化（如，优化颜色和字体，来最大化阅读速度），物理系统的优化（如，在仿真中优化翼型）。

In this paper we discuss a state-of-the-art system for black–box optimization developed within Google, called Google Vizier, named after a high official who offers advice to rulers. It is a service for black-box optimization that supports several advanced algorithms. The system has a convenient Remote Procedure Call (RPC) interface, along with a dashboard and analysis tools. Google Vizier is a research project, parts of which supply core capabilities to our Cloud Machine Learning HyperTune subsystem. We discuss the architecture of the system, design choices, and some of the algorithms used.

本文中，我们讨论了Google内部开发的黑箱优化目前最好的系统，称为Google Vizier。这是一个黑箱优化的服务，支持几种高级算法。系统有方便的RPC接口，和控制面板，分析工具。Google Vizier是一个研究项目，其中的一部分为我们的云机器学习HyperTune子系统提供了核心的能力。我们讨论了系统的架构，设计选择，和一些使用的算法。

### 1.1 Related Work

Black–box optimization makes minimal assumptions about the problem under consideration, and thus is broadly applicable across many domains and has been studied in multiple scholarly fields under names including Bayesian Optimization [2, 25, 26], Derivative–free optimization [7, 24], Sequential Experimental Design [5], and assorted variants of the multiarmed bandit problem [13, 20, 29].

黑箱优化对考虑的问题所做的假设是最小的，因此在很多领域中都可以应用，在多个学术领域中都进行了研究，有不同的学术称呼，包括贝叶斯优化，无导数优化，顺序试验设计，和多臂老虎机问题的各种变体。

Several classes of algorithms have been proposed for the problem. The simplest of these are non-adaptive procedures such as Random Search, which selects x_t uniformly at random from X at each time step t independent of the previous points selected, {x_τ : 1 ≤ τ < t}, and Grid Search, which selects along a grid (i.e., the Cartesian product of finite sets of feasible values for each parameter). Classic algorithms such as SimulatedAnnealing and assorted genetic algorithms have also been investigated, e.g., Covariance Matrix Adaptation [16].

对这个问题提出了几类算法。这些算法中最简单的，是非自适应过程，比如随机搜索，在每个步骤中从X中随机均匀选择x_t，与之前所选的点是独立的，{x_τ : 1 ≤ τ < t}，还有网格搜索，沿着一个网络进行选择（即，对每个参数的可行值的有限集的笛卡尔积）。经典算法如模拟退火，和各种遗传算法也进行了研究，如协方差矩阵自适应。

Another class of algorithms performs a local search by selecting points that maintain a search pattern, such as a simplex in the case of the classic Nelder–Mead algorithm [22]. More modern variants of these algorithms maintain simple models of the objective f within a subset of the feasible regions (called the trust region), and select a point x_t to improve the model within the trust region [7].

另一类算法进行的是局部搜索，选择的点会形成一个搜索模式，比如在经典的Nelder-Mead算法中形成的单纯形。这些算法的更多的现代变体，维护了目标f在可行区域的子集（称为信任区域）中的简单模型，选择了一个点x_t，来在信任区域中改进模型。

More recently, some researchers have combined powerful techniques for modeling the objective f over the entire feasible region, using ideas developed for multiarmed bandit problems for managing explore / exploit trade-offs. These approaches are fundamentally Bayesian in nature, hence this literature goes under the name Bayesian Optimization. Typically, the model for f is a Gaussian process (as in [26, 29]), a deep neural network (as in [27, 31]), or a regression forest (as in [2, 19]).

最近，一些研究者将在整个可行区域中对目标函数f进行建模的技术结合到了一起，使用的思想是为多臂老虎机问题开发的，管理explore/exploit的折中。这些方法在本质上是贝叶斯的，因此这些文献的，名称都是贝叶斯优化。典型的，f的模型是一个高斯过程，或一个深度神经网络，或一个回归树。

Many of these algorithms have open-source implementations available. Within the machine learning community, examples include, e.g., HyperOpt, MOE, Spearmint, and AutoWeka, among many others. In contrast to such software packages, which require practitioners to set them up and run them locally, we opted to develop a managed service for black–box optimization, which is more convenient for users but involves additional design considerations.

很多这些算法都有开源的实现。在机器学习的群体中，例子包括HyperOpt，MOE，Spearmint，和AutoWeka，等等。这些软件包需要人们对其进行设置，在本地进行运行，而我们则选择来开发黑箱优化的服务，对用户来说更方便，涉及到额外的设计考虑。

### 1.2 Definitions

Throughout the paper, we use to the following terms to describe the semantics of the system: 在本文中，我们使用下面的术语来描述系统的语义：

A Trial is a list of parameter values, x, that will lead to a single evaluation of f(x). A trial can be “Completed”, which means that it has been evaluated and the objective value f(x) has been assigned to it, otherwise it is “Pending”. 一次尝试(Trial)，是参数值x的一个列表，可以得到f(x)的一次结果评估。一次尝试可以完成，意思是已经评估结束了，目标值f(x)已经得到了赋值，否则就是在pending中。

A Study represents a single optimization run over a feasible space. Each Study contains a configuration describing the feasible space, as well as a set of Trials. It is assumed that f(x) does not change in the course of a Study. 一个study表示在可行空间中的一次优化运行。每个study包含一个配置，描述了可行空间，以及尝试的集合。假设在一个study的过程中，f(x)是不变的。

A Worker refers to a process responsible for evaluating a Pending Trial and calculating its objective value. 一个worker表示一个进程，负责评估一个pending的尝试，计算其目标函数值。

## 2. System Overview

This section explores the design considerations involved in implementing black-box optimization as a service. 本节探索在将黑箱优化实现为一个服务时涉及到的设计考虑。

### 2.1 Design Goals and Constraints

Vizier’s design satisfies the following desiderata: Vizier的设计满足下列需求：

- Ease of use. Minimal user configuration and setup. 容易使用。用户配置和设置最小化。

- Hosts state-of-the-art black-box optimization algorithms. 包含目前最好的黑箱优化算法。

- High availability. 高可用性。

- Scalable to millions of trials per study, thousands of parallel trial evaluations per study, and billions of studies. 可以扩展到每个study有上百万次trial，每个study中有数千个并行的trail评估，并有数十亿个studies。

- Easy to experiment with new algorithms. 新算法很容易进行试验。

- Easy to change out algorithms deployed in production. 容易改变在生产中部署的算法。

For ease of use, we implemented Vizier as a managed service that stores the state of each optimization. This approach drastically reduces the effort a new user needs to get up and running; and a managed service with a well-documented and stable RPC API allows us to upgrade the service without user effort. We provide a default configuration for our managed service that is good enough to ensure that most users need never concern themselves with the underlying optimization algorithms.

为使用起来更容易，我们将Vizier实现为一个服务，存储了每次优化的状态。这种方法极大的降低了一个新用户运行的工作量；一个管理起来的服务，带有良好的文档，和稳定的RPC API，使我们可以升级服务，而不需要用户进行努力。我们对服务提供了默认的配置，对于多数用户来说已经很好了，不需要去考虑底层的优化算法。

The default option allows the service to dynamically select a recommended black–box algorithm along with low–level settings based on the study configuration. We choose to make our algorithms stateless, so that we can seamlessly switch algorithms during a study, dynamically choosing the algorithm that is likely to perform better for a particular trial of a given study. For example, Gaussian Process Bandits [26, 29] provide excellent result quality, but naive implementations scale as O(n^3) with the number of training points. Thus, once we’ve collected a large number of completed Trials, we may want to switch to using a more scalable algorithm.

默认的选项使服务可以动态的选择推荐的黑箱算法，以及基于study配置的底层设置。我们使算法无状态，这样我们可以在一个study中无缝的切换算法，动态的选择可以在给定的study中的特定trial中可能会表现更好的算法。比如，高斯过程老虎机可以给出优秀的结果质量，但朴素的实现随着训练点数量的增加增加为O(n^3)。因此，一旦我们收集了大量完成的trial，我们可能会想切换到一个更加可扩展的算法。

At the same time, we want to allow ourselves (and advanced users) the freedom to experiment with new algorithms or special-case modifications of the supported algorithms in a manner that is safe, easy, and fast. Hence, we’ve built Google Vizier as a modular system consisting of four cooperating processes (see Figure 1) that update the state of Studies in the central database. The processes themselves are modular with several clean abstraction layers that allow us to experiment with and apply different algorithms easily.

同时，我们希望我们自己和高级用户，可以自由的用新算法进行试验，或对支持的算法进行特殊情况的修改，而且要安全，容易，快速。因此，我们将Google Vizier构建为一个模块化的系统，由四个合作的过程构成（见图1），在一个中央数据库中更新studies的状态。这个过程本身就是模块化的，有几个干净的抽象层，使我们可以很容易的进行试验和应用不同的算法。

Finally we want to allow multiple trials to be evaluated in parallel, and allow for the possibility that evaluating the objective function for each trial could itself be a distributed process. To this end we define Workers, responsible for evaluating suggestions, and identify each worker by a name (a worker_handle) that persists across process preemptions or crashes.

最后，我们希望可以并行评估多个trial，对每个trial都评估目标函数，这种可能性本身就是一个分布式的过程。为此，我们定义workers，负责评估建议，每个worker都有一个名称(workder_handle)，在整个进程的preemptions或crashes的过程中都保持不变。

### 2.2 Basic User Workflow

To use Vizier, a developer may use one of our client libraries (currently implemented in C++, Python, Golang), which will generate service requests encoded as protocol buffers [15]. The basic workflow is extremely simple. Users specify a study configuration which includes:

为使用Vizier，开发者要使用我们的一个客户端库（目前用C++，Python，Golang实现的），生成的服务请求编码为protocol buffers。基本的工作流是非常简单的。用户指定一个study的配置，这包括：

- Identifying characteristics of the study (e.g. name, owner, permissions). 识别study的特性（如，名称，所有者，权限）。

- The set of parameters along with feasible sets for each (c.f., Section 2.3.1 for details); Vizier does constrained optimization over the feasible set. 参数集与可行集；Vizier将优化约束在可行集中。

Given this configuration, basic use of the service (with each trial being evaluated by a single process) can be implemented as follows: 给定这些配置，这些服务的基本使用（每个trial用单个进程进行评估）可以实现如下：

```

# Register this client with the Study, creating it if necessary.

client.LoadStudy(study_config, worker_handle)

while (not client.StudyIsDone()):

# Obtain a trial to evaluate.

trial = client.GetSuggestion()

# Evaluate the objective function at the trial parameters.

metrics = RunTrial(trial)

# Report back the results.

client.CompleteTrial(trial, metrics)

```

Here RunTrial is the problem–specific evaluation of the objective function f. Multiple named metrics may be reported back to Vizier, however one must be distinguished as the objective value f(x) for trial x. Note that multiple processes working on a study should share the same worker_handle if and only if they are collaboratively evaluating the same trial. All processes registered with a given study with the same worker_handle are guaranteed to receive the same trial upon request, which enables distributed trial evaluation.

这里，RunTrial是目标函数f的问题专用的评估。多个名称的度量会汇报回给Vizier，但一个人必须对trial x区别出目标值f(x)。注意在一个study上工作的多个进程，如果其在共同评估相同的trial，且只在这种情况时，需要共享相同的worker_handle。对一个给定的study用相同的worker_handle注册的所有进程，要确保在请求时收到相同的trial，使分布式trial评估成为可能。

### 2.3 Interfaces

2.3.1 Configuring a Study. To configure a study, the user provides a study name, owner, optional access permissions, an optimization goal from {MAXIMIZE, MINIMIZE}, and specifies the feasible region X via a set of ParameterConfigs, each of which declares a parameter name along with its values. We support the following parameter types:

配置一个study。为配置一个study，用户需要给出一个study的名称，所有人，可选的访问权限，一个优化目标，为{Maximize, Minimize}中的一个，通过ParameterConfigs集指定可行区域X，其中每个都声明了一个参数名称，与对应的值。我们支持下列参数类型：

- DOUBLE: The feasible region is a closed interval [a,b] for some real values a ≤ b. 可行区域是一个封闭区间[a, b]，其中a≤b是实值。

- INTEGER: The feasible region has the form [a,b] ∩ Z for some integers a ≤ b. 可行区域为[a,b] ∩ Z，其中a≤b都是整数。

- DISCRETE: The feasible region is an explicitly specified, ordered set of real numbers. 可行区域是显式指定的，实值的有序集。

- CATEGORICAL: The feasible region is an explicitly specified, unordered set of strings. 可行区域是显式指定的，无序的字符串集合。

Users may also suggest recommended scaling, e.g., logarithmic scaling for parameters for which the objective may depend only on the order of magnitude of a parameter value. 用户还会建议推荐的缩放，如，对数缩放，对于目标可能只依赖于一个参数值的数量级的参数。

2.3.2 API Definition. Workers and end users can make calls to the Vizier Service using either a REST API or using Google’s internal RPC protocol [15]. The most important service calls are:

API定义。Workers和终端用户调用Vizier服务是使用一个REST API，或使用Google的内部RPC协议。最重要的服务调用是：

- CreateStudy: Given a Study configuration, this creates an optimization Study and returns a globally unique identifier (“guid”) which is then used for all future service calls. If a Study with a matching name exists, the guid for that Study is returned. This allows parallel workers to call this method and all register with the same Study.

给定一个Study配置，这创建了一个优化study，返回一个guid，用于所有未来的服务调用。如果存在一个名称相同的Study，那么就返回这个study的guid。这允许并行的worker调用这个方法，都注册为相同的study。

- SuggestTrials: This method takes a “worker handle” as input, and immediately returns a globally unique handle for a “long-running operation” that represents the work of generating Trial suggestions. The user can then poll the API periodically to check the status of the operation. Once the operation is completed, it will contain the suggested Trials. This design ensures that all service calls are made with low latency, while allowing for the fact that the generation of Trials can take longer.

这个方法以worker handle为输入，立刻对一个长期运行的操作返回一个全局唯一的handle，表示生成Trial建议的工作。用户然后可以周期性的调用这个API，以检查这个操作的状态。一旦操作完成，就会包含建议的Trials。这个设计确保了所有的服务调用延迟都很低，还允许了Trials的生成可以耗时更长。

- AddMeasurementToTrial: This method allows clients to provide intermediate metrics during the evaluation of a Trial. These metrics are then used by the Automated Stopping rules to determine which Trials should be stopped early.

这个方法使客户在评估一个Trial时能提供中间度量。这些度量然后用于Automated Stopping规则来确定哪个trial应当早停。

- CompleteTrial: This method changes a Trial’s status to “Completed”, and provides a final objective value that is then used to inform the suggestions provided by future calls to SuggestTrials.

这个方法将一个Trial的状态变为完成，然后提供一个最终的目标值，用于通知未来的调用给SuggestTrials的建议。

- ShouldTrialStop: This method returns a globally unique handle for a long-running operation that represents the work of determining whether a Pending Trial should be stopped.

这个方法对一个长期运行的操作返回一个全局唯一的handle，表示确定一个Pending Trial是否应当停止的工作。

### 2.4 Infrastructure

2.4.1 Parallel Processing of Suggestion Work. As the de facto parameter tuning engine of Google, Vizier is constantly working on generating suggestions for a large number of Studies concurrently. As such, a single machine would be insufficient for handling the workload. Our Suggestion Service is therefore partitioned across several Google datacenters, with a number of machines being used in each one. Each instance of the Suggestion Service potentially can generate suggestions for several Studies in parallel, giving us a massively scalable suggestion infrastructure. Google’s load balancing infrastructure is then used to allow clients to make calls to a unified endpoint, without needing to know which instance is doing the work.

建议工作的并行处理。作为Google内部实际的参数调节引擎，Vizier一直工作在为大量Studies同时生成建议上。这样，一台机器不能处理这些workload。我们的Suggestion服务因此分割到了几个Google数据中心中，每个数据中心中都用了一定量的机器。Suggestion服务的每个实例，都可以给几个studies并行生成建议，给出大量可缩放的建议基础设施。Google的负载均衡基础设施然后用于使客户可以调用统一的终点，不需要知道哪个实例在进行这个工作。

When a request is received by a Suggestion Service instance to generate suggestions, the instance first places a distributed lock on the Study. This lock is acquired for a fixed period of time, and is periodically extended by a separate thread running on the instance. In other words, the lock will be held until either the instance fails, or it decides it’s done working on the Study. If the instance fails (due to e.g. hardware failure, job preemption, etc), the lock soon expires, making it eligible to be picked up by a separate process (called the “DanglingWorkFinder”) which then reassigns the Study to a different Suggestion Service instance.

当一个Suggestion服务实例接收到一个请求，要生成建议，这个实例首先为这个study放上一个分布式锁。这个锁在固定的时间段内存在，由一个其他的线程周期性的在这个实例上拓展。换句话说，这个锁会直到实例失败时失效，或决定了在这个study上的工作结束了才失效。如果这个实例失败了（由于硬件错误，job preemption，等），这个锁很快就过期了，使其有资格被另一个进程pick up（称为DanglingWorkFinder），然后将这个study重新指派给一个不同的Suggestion服务实例。

One consideration in maintaining a production system is that bugs are inevitably introduced as our code matures. Occasionally, a new algorithmic change, however well tested, will lead to instances of the Suggestion Service failing for particular Studies. If a Study is picked up by the DanglingWorkFinder too many times, it will temporarily halt the Study and alert us. This prevents subtle bugs that only affect a few Studies from causing crash loops that affect the overall stability of the system.

在维护一个生产系统中的一个考虑是，当代码成熟后，不可避免的会出现bugs。偶尔的，一个新的算法变化，不论测试的多好，会导致Suggestion服务的实例在特定studies上失败。如果一个study由DanglingWorkFinder pick up很多次，会临时的停止这个study，并警告我们。这避免了只影响几个studies的不明显的bugs导致崩溃的loops，从而影响系统的整体稳定性。

### 2.5 The Algorithm Playground

Vizier’s algorithm playground provides a mechanism for advanced users to easily, quickly, and safely replace Vizier’s core optimization algorithms with arbitrary algorithms. Vizier的算法运动场，为高级用户提供了一个机制，以容易的，快速的，安全的替换Vizier的核心优化算法为任意算法。

The playground serves a dual purpose; it allows rapid prototyping of new algorithms, and it allows power-users to easily customize Vizier with advanced or exotic capabilities that are particular to their use-case. In all cases, users of the playground benefit from all of Vizier’s infrastructure aside from the core algorithms, such as access to a persistent database of Trials, the dashboard, and visualizations.

运动场有两重目的；可以进行新算法的快速原型，允许高级用户很容易的定制Vizier，对其使用案例增加其特有的高级功能或特别功能。在所有情况中，除了核心算法意外，运动场的用户从所有Vizier的基础设施中受益，比如访问Trials的永久数据库，仪表盘，和可视化。

At the core of the playground is the ability to inject Trials into a Study. Vizier allows the user or other authorized processes to request one or more particular Trials be evaluated. In Playground mode, Vizier does not suggest Trials for evaluation, but relies on an external binary to generate Trials, which are then pushed to the service for later distribution to the workers.

运动场的核心，是将Trials注射到Study的的能力。Vizier允许用户或其他有权限的进程，请求评估一个或更多的特定Trials。在Playground模式，Vizier并不并不建议Trials进行评估，而是依赖于外部程序来生成Trials，然后推送到服务中，后续分布到各种worker中。

More specifically, the architecture of the Playground involves the following key components: (1) Abstract Policy (2) Playground Binary, (3) Vizier Service and (4) Evaluation Workers. See Figure 2 for an illustration.

更具体的，Playground的架构涉及到下面的关键组成部分：(1) Abstract Policy (2) Playground Binary, (3) Vizier Service (4) Evaluation Workers。见图2的描述。

The Abstract Policy contains two abstract methods: 抽象政策包含两个抽象方法：

(1) GetNewSuggestions(trials, num_suggestions)

(2) GetEarlyStoppingTrials(trials)

which should be implemented by the user’s custom policy. Both these methods are passed the full state of all Trials in the Study, so algorithms may be implemented in a stateless fashion if desired. GetNewSuggestions is expected to generate num_suggestions new trials, while the GetEarlyStoppingTrials method is expected to return a list of Pending Trials that should be stopped early. The custom policy is registered with the Playground Binary which periodically polls the Vizier Service. The Evaluation Workers maintain the service abstraction and are unaware of the existence of the Playground.

这两者应当由用户的定制策略来实现。这些方法在study中都接收了所有trials的完整状态，所以如果期望的话，算法可能以一种无状态的方式实现。GetNewSuggestions的期望是生成num_suggestions个新的trials，而GetEarlyStoppingTrials方法要返回Pending Trials的一个列表，这个列表中的应当在早期就停止掉。定制策略注册到Playground Binary中，周期性的访问Vizier服务。Evaluation Workers维护服务抽象，并不会意识到Playground的存在。

### 2.6 Benchmarking Suite

Vizier has an integrated framework that allows us to efficiently benchmark our algorithms on a variety of objective functions. Many of the objective functions come from the Black-Box Optimization Benchmarking Workshop [10], but the framework allows for any function to be modeled by implementing an abstract Experimenter class, which has a virtual method responsible for calculating the objective value for a given Trial, and a second virtual method that returns the optimal solution for that benchmark.

Vizier有一个集成框架，使我们可以高效在各种目标函数中对算法进行基准测试。很多目标函数都来自黑箱优化基准测试工作室，但这个框架可以对任何函数进行建模，只要实现一个抽象的Experimenter类，有一个虚拟方法负责计算对给定Trials的目标值，还要有一个虚拟方法返回对这个基准测试的最优解。

Users configure a set of benchmark runs by providing a set of algorithm configurations and a set of objective functions. The benchmarking suite will optimize each function with each algorithm k times (where k is configurable), producing a series of performance-over-time metrics which are then formatted after execution. The individual runs are distributed over multiple threads and multiple machines, so it is easy to have thousands of benchmark runs executed in parallel.

用户配置基准测试运行集合，提供算法配置集合和目标函数集合。基准测试包会用每个算法优化每个函数k次（k是可配置的），产生一系列performance-over-time的度量，在执行后以一定格式组织好。单次运行会分布到多个线程和多个机器中，所以很容易有数千次基准测试并行运行。

### 2.7 Dashboard and Visualizations

Vizier has a web dashboard which is used for both monitoring and changing the state of Vizier studies. The dashboard is fully featured and implements the full functionality of the Vizier API. The dashboard is commonly used for: (1) Tracking the progress of a study; (2) Interactive visualizations; (3) Creating, updating and deleting a study; (4) Requesting new suggestions, early stopping, activating/deactivating a study. See Figure 3 for a section of the dashboard. In addition to monitoring and visualizations, the dashboard contains action buttons such as Get Suggestions.

Vizier有一个网页版仪表盘，用于监控和改变Vizier studies的状态。仪表盘很有特点，实现了Vizier API的所有功能。仪表盘一般用于：(1)追踪一个study的进程；(2)互动的可视化；(3)创建，更新和删除一个study；(4)请求新的建议，早停，激活/停用一个study。见图3中仪表盘的一部分。除了监控和可视化，仪表盘包含行为的按钮，比如Get Suggestions。

The dashboard uses a translation layer which converts between JSON and protocol buffers [15] when talking with backend servers. The dashboard is built with Polymer [14] an open source web framework supported by Google and uses material design principles. It contains interactive visualizations for analyzing the parameters in your study. In particular, we use the parallel coordinates visualization [18] which has the benefit of scaling to high dimensional spaces (∼15 dimensions) and works with both numerical and categorical parameters. See Figure 4 for an example. Each vertical axis is a dimension corresponding to a parameter, whereas each horizontal line is an individual trial. The point at which the horizontal line intersects the vertical axis gives the value of the parameter in that dimension. This can be used for examining how the dimensions co-vary with each other and also against the objective function value (left most axis). The visualizations are built using d3.js [4].

仪表盘使用一个翻译层，在与后端服务器进行交互时，在JSON和protobuf之间进行转换。仪表盘是用Polymer构建的，这是一个开源网页框架，由Google支持，使用材质设计的原则。包含交互的可视化，以分析study中的参数。特别是，我们使用并行的协调可视化，可以缩放到高维空间（~15维度），对数值参数和类别参数都适用。见图4中的例子。每个纵轴是对应一个参数的维度，而每个横线是一个单独的trial。横线与纵轴相交的点，给出了这个维度中的值。这可以用于检查各个维度怎样互相变化，怎样随着目标函数值变化。可视化是用d3.js构建的。

## 3. The Vizier Algorithms

Vizier’s modular design allows us to easily support multiple algorithms. For studies with under a thousand trials, Vizier defaults to using Batched Gaussian Process Bandits [8]. We use a Matérn kernel with automatic relevance determination (see e.g. section 5.1 of Rasmussen and Williams [23] for a discussion) and the expected improvement acquisition function [21]. We search for and find local maxima of the acquisition function with a proprietary gradient-free hill climbing algorithm, with random starting points.

Vizier的模块化设计，使我们可以很容易的支持多个算法。对于在1000个trials以下的studies，Vizier默认使用批量化高斯过程老虎机算法。我们使用一种Matérn核，可以自动确定相关性，有期望的改进获取函数。我们用获得函数，用无梯度爬坡算法来寻找局部最大值，起始点是随机的。

We implement discrete parameters by embedding them in R. Categorical parameters with k feasible values are represented via one-hot encoding, i.e., embedded in [0,1]^k. In both cases, the Gaussian Process regressor gives us a continuous and differentiable function upon which we can walk uphill, then when the walk has converged, round to the nearest feasible point.

我们实现离散参数的方式是将其嵌入到实数域R中。类别参数有k个可行值，是通过独热码表示的，即，嵌入到[0,1]^k中。在两种情况下，高斯过程回归器给了我们一个连续的可微分的函数，在其上，我们可以进行爬坡，当优化收敛时，四舍五入到最近的可行点上。

While some authors recommend using Bayesian deep learning models in lieu of Gaussian processes for scalability [27, 31], in our experience they are too sensitive to their own hyperparameters and do not reliably perform well. Other researchers have recognized this problem as well, and are working to address it [28].

一些作者推荐使用贝叶斯深度学习模型代替高斯过程，可以获得可扩展性，在我们的经验中，它们对其本身的超参数过于敏感，不能很可靠的得到很好的性能。其他的研究者也发现了这个问题，正在进行处理。

For studies with tens of thousands of trials or more, other algorithms may be used. Though RandomSearch and GridSearch are supported as first–class choices and may be used in this regime, and many other published algorithms are supported through the algorithm playground, we currently recommend a proprietary local–search algorithm under these conditions.

对于超过上万次trials的studies，可能会使用其他算法。虽然RandomSearch和GridSearch是第一类支持的选择，可能在这个领域中应用，很多其他发表的算法也通过算法演练场进行支持，我们目前推荐一个局部搜索算法。

For all of these algorithms we support data normalization, which maps numeric parameter values into [0,1] and objective values onto [−0.5,0.5]. Depending on the problem, a one-to-one nonlinear mapping may be used for some of the parameters, and is typically used for the objective. Data normalization is handled before trials are presented to the trial suggestion algorithms, and its suggestions are transparently mapped back to the user-specified scaling.

对所有算法，我们都支持数据归一化，将数值参数值映射到[0, 1]中，目标值映射到[-0.5, 0.5]中。依赖于问题，对一些参数可能会应用一对一的非线性映射，一般会用于目标值。数据归一化是在trials之前进行处理，然后送入trial Suggestion算法，其Suggestion透明的映射回用户指定的缩放中。

### 3.1 Automated Early Stopping

In some important applications of black–box optimization, information related to the performance of a trial may become available during trial evaluation. Perhaps the best example of such a performance curve occurs when tuning machine learning hyperparameters for models trained progressively (e.g., via some version of stochastic gradient descent). In this case, the model typically becomes more accurate as it trains on more data, and the accuracy of the model is available at the end of each training epoch. Using these accuracy vs. training step curves, it is often possible to determine that a trial’s parameter settings are unpromising well before evaluation is finished. In this case we can terminate trial evaluation early, freeing those evaluation resources for more promising trial parameters. When done algorithmically, this is referred to as automated early stopping.

在一些重要的黑盒优化应用中，在trial评估时，与一个trial相关的性能信息就会可用。这样一个性能曲线的最佳例子，可能就是在调节机器学习模型超参数时，逐步训练的例子（如，通过随机梯度下降）。在这种情况中，随着在更多的数据上训练，模型一般会变得更加精确，模型的准确率在每个训练epoch后就会可用。使用这些准确率vs训练步骤曲线，通常可能在评估结束之前确定一个trial的参数设置是不是好。在这种情况中，我们就可以早一些结束trial评估，释放这些评估资源，用于更有希望的trial参数。当用算法进行处理时，这通常称为自动早停。

Vizier supports automated early stopping via an API call to a ShouldTrialStop method. Analogously to the Suggestion Service, there is an Automated Stopping Service that accepts requests from the Vizier API to analyze a study and determine the set of trials that should be stopped, according to the configured early stopping algorithm. As with suggestion algorithms, several automated early stopping algorithms are supported, and rapid prototyping can be done via the algorithm playground.

Vizier通过API调用ShouldTrialStop方法支持自动早停。与Suggestion服务类似，有一个自动停止服务，接收来自Vizier API的请求，分析一个study，根据配置的早停算法，确定需要停止的trials集合。至于Suggestion算法，支持几种自动早停算法，通过算法运动场可以支持快速原型。

### 3.2 Automated Stopping Algorithms

Vizier supports the following automated stopping algorithms. These are meant to work in a stateless fashion i.e. they are given the full state of all trials in the Vizier study when determining which trials should stop.

Vizier支持下列自动停止算法。这意味着以一种无状态的方式进行工作，即，在确定哪些trials需要停止时，在Vizier study中确定了所有trials的完整状态。

**3.2.1 Performance Curve Stopping Rule**. This stopping rule performs regression on the performance curves to make a prediction of the final objective value of a Trial given a set of Trials that are already Completed, and a partial performance curve (i.e., a set of measurements taken during Trial evaluation). Given this prediction, if the probability of exceeding the optimal value found thus far is sufficiently low, early stopping is requested for the Trial.

性能曲线停止准则。停止准则对性能曲线进行回归，在给定已经完成的trials，和部分性能曲线时（即，在Trial评估时，获得的度量集合），预测最终的目标值。给定这个预测，如果超出最优值的概率足够低，则对这个trial要求早停。

While prior work on automated early stopping used Bayesian parametric regression [9, 30], we opted for a Bayesian non-parametric regression, specifically a Gaussian process model with a carefully designed kernel that measures similarity between performance curves. Our motivation in this was to be robust to many kinds of performance curves, including those coming from applications other than tuning machine learning hyperparameters in which the performance curves may have very different semantics. Notably, this stopping rule still works well even when the performance curve is not measuring the same quantity as the objective value, but is merely predictive of it.

之前在自动早停上的工作使用贝叶斯参数回归，而我们则选择贝叶斯非参数回归，具体的，是一个高斯过程模型，带有仔细设计的核，测量性能曲线之间的相似性。我们的动机是，对多种性能曲线要稳健，包括那些不是调节机器学习超参数的性能曲线，这些曲线会非常不同。值得注意的是，这种停止规则在性能曲线并不是度量目标值的时候，而只是预测的时候，效果也很好。

**3.2.2 Median Stopping Rule**. The median stopping rule stops a pending trial x_t at step s if the trial’s best objective value by step s is strictly worse than the median value of the running averages $\hat o^τ_{1:s}$ of all completed trials’ objectives x_τ reported up to step s. Here, we calculate the running average of a trial x_τ up to step s as $\hat o^τ_{1:s} = Σ^s_{i=1} o^τ_i/s$, where $o^τ_i$ is the objective value of $x_τ$ at step i. As with the performance curve stopping rule, the median stopping rule does not depend on a parametric model, and is applicable to a wide range of performance curves. In fact, the median stopping rule is model–free, and is more reminiscent of a bandit-based approach such as HyperBand [20].

中值停止规则。中值停止规则在满足如下条件时会在步骤s时停止一个pending trial x_t，即在步骤s时的trial的最好的目标值，比之前所有值的中值都要小时。这里，我们计算trial x_τ到步骤s时的平均为$\hat o^τ_{1:s} = Σ^s_{i=1} o^τ_i/s$，其中$o^τ_i$是$x_τ$在步骤i时的目标值。至于性能曲线停止规则，中值停止规则不依赖于参数模型，可以应用到很广范围内的性能曲线中。实际上，中值停止规则是不需要模型的，更能让人想起基于bandit的方法，比如HyperBand。

### 3.3 Transfer Learning

When doing black-box optimization, users often run studies that are similar to studies they have run before, and we can use this fact to minimize repeated work. Vizier supports a form of Transfer Learning which leverages data from prior studies to guide and accelerate the current study. For instance, one might tune the learning rate and regularization of a machine learning system, then use that Study as a prior to tune the same ML system on a different data set.

当进行黑盒优化时，用户运行的studies与其之前运行的studies通常类似，我们可以用这个事实，来最小化重复的工作。Vizier支持迁移学习的形态，利用之前的studies的数据，来加速当前的study。比如，一个人可能调节一个机器学习系统的学习速率和规则化，然后使用这个study作为一个先验，在不同的数据集上调节相同的ML系统。

Vizier’s current approach to transfer learning is relatively simple, yet robust to changes in objective across studies. We designed our transfer learning approach with these goals in mind:

Vizier当前的迁移学习方法是相对简单的，但对不同studies之间的目标的变化是稳健的。我们设计我们的迁移学习的方法时，有以下目标：

(1) Scale well to situations where there are many prior studies. 对于有很多先验studies的情况，适应的很好；

(2) Accelerate studies (i.e., achieve better results with fewer trials) when the priors are good, particularly in cases where the location of the optimum, x∗, doesn’t change much. 当先验很好的时，加速studies（即，用更少的trials得到更好的结果），尤其是最优位置x*变化不大的时候。

(3) Be robust against poorly chosen prior studies (i.e., a bad prior should give only a modest deceleration). 对选择的不太好的先验studies要稳健（即，不好的先验会给出中等的减速）。

(4) Share information even when there is no formal relationship between the prior and current Studies. 在先验和当前的studies没有关系时，会共享之间的信息。

In previous work on transfer learning in the context of hyperparameter optimization, Bardenet et al. [1] discuss the difficulty in transferring knowledge across different datasets especially when the observed metrics and the sampling of the datasets are different. They use a ranking approach for constructing a surrogate model for the response surface. This approach suffers from the computational overhead of running a ranking algorithm. Yogatama and Mann [32] propose a more efficient approach, which scales as Θ(kn + n^3) for k studies of n trials each, where the cubic term comes from using a Gaussian process in their acquisition function.

在之前超参数优化的迁移学习的工作中，Bardenet等[1]讨论了在不同的数据集中迁移知识的难度，尤其是观察到的度量，和数据集的采样是不同的情况。他们使用了排序方法来为响应表面构建代理模型。这个方法的问题在于运行排序算法的计算代价比较高。Yogatama and Mann [32]提出了一种更高效的方法，对于k个studies中每个n个trials，计算复杂度为Θ(kn + n^3)，其中三次方项来自于在获得函数中使用了一个高斯过程。

Vizier typically uses Gaussian Process regressors, so one natural approach to implementing transfer learning might be to build a larger Gaussian Process regressor that is trained on both the prior(s) and the current Study. However that approach fails to satisfy design goal 1: for k studies with n trials each it would require Ω(k^3n^3) time. Such an approach also requires one to specify or learn kernel functions that bridge between the prior(s) and current Study, violating design goal 4.

Vizier一般使用高斯过程回归器，所以一种自然的实现迁移学习的方法是，构建一个更大的高斯过程回归器，在先验和当前的study中进行训练。但是，这个方法不满足目标1：对于k个studies，每个有n次trial，会需要Ω(k^3n^3)耗时。这样一种方法还需要指定或学习核函数，弥补先验和当前study之间的差异，这违反了目标4。

Instead, our strategy is to build a stack of Gaussian Process regressors, where each regressor is associated with a study, and where each level is trained on the residuals relative to the regressor below it. Our model is that the studies were performed in a linear sequence, each study using the studies before it as priors.

我们的策略是，构建一组高斯过程回归器，其中每个回归器都和一个study相关，每个层次都是在其之下的回归器的残差上训练的。我们的模型是，这些studies是在一个线性序列中进行的，每个study都使用之前的studies作为先验。

The bottom of the stack contains a regressor built using data from the oldest study in the stack. The regressor above it is associated with the 2nd oldest study, and regresses on the residual of its objective relative to the predictions of the regressor below it. Similarly, the regressor associated with the i-th study is built using the data from that study, and regresses on the residual of the objective with respect to the predictions of the regressor below it.

这一组的第一个回归器，是用最老的数据进行构建的。之上的回归器是与第二老的study相关，在与之前的回归器预测的残差上进行回归得到。类似的，与第i个study相关的回归器是用这个study的数据构建得到的，是在之前的回归器的残差上回归得到的。

More formally, we have a sequence of studies {S_i}^k_{i=1} on unknown objective functions {f_i}^k_i=1, where the current study is S_k, and we build two sequences of regressors {R_i}^k_{i=1} and {R'_i}^k _{i=1} having posterior mean functions {µ_i}^k_{i=1} and {µ'_i}^k _{i=1} respectively, and posterior standard deviation functions {σ_i}^k_{i=1} and {σ'_i}^k _{i=1}, respectively. Our final predictions will be µ_k and σ_k.

更正式一点，我们有一个studies序列{S_i}^k_{i=1} ，未知的目标函数{f_i}^k_i=1，当前study是S_k，我们构建两个回归器序列{R_i}^k_{i=1}和{R'_i}^k _{i=1}，其后验均值函数分别为{µ_i}^k_{i=1}和{µ'_i}^k _{i=1}，后验标准差函数分别为{σ_i}^k_{i=1}和{σ'_i}^k _{i=1}。我们的最终预测为µ_k和σ_k。

Let D_i ={(x^i_t, y^i_t)}_t be the dataset for study S_i. Let R'_i be a regressor trained using data {((x^i_t,y^i_t) − µ_{i−1} (x^i_t))}_t which computes µ'_i and σ'_i. Then we define as our posterior means at level i as µ_i(x) := µ'_ i(x) + µ_{i−1} (x). We take our posterior standard deviations at level i, σ_i(x), to be a weighted geometric mean of σ'_ i(x) and σ_{i−1} (x), where the weights are a function of the amount of data (i.e., completed trials) in S_i and S_{i−1}. The exact weighting function depends on a constant α ≈ 1 sets the relative importance of old and new standard deviations.

令D_i ={(x^i_t, y^i_t)}_t是study S_i的数据集。令R'_i是用数据{((x^i_t,y^i_t) − µ_{i−1} (x^i_t))}_t训练的，计算了µ'_i和σ'_i。然后我们定义在层次i上的后验均值为µ_i(x) := µ'_ i(x) + µ_{i−1} (x)。在层次i上的后验标准方差σ_i(x)为，σ'_ i(x)和σ_{i−1} (x)的加权几何平均，权值是S_i和S_{i−1}上的数据总量（即，完全的trials）的函数。严格的加权函数依赖于一个常数α ≈ 1，设定了老和新的标准差的相对重要性。

This approach has nice properties when the prior regressors are densely supported (i.e. has many well-spaced data points), but the top-level regressor has relatively little training data: (1) fine structure in the priors carries through to µ_k, even if the top-level regressor gives a low-resolution model of the objective function residual; (2) since the estimate for σ'_ k is inaccurate, averaging it with σ_{k−1} can lead to an improved estimate. Further, when the top-level regressor has dense support, β → 1 and the σ_k → σ'_k, as one might desire.

在先验的回归器是密集支撑的时候（即，有很多分布很好的数据点），这种方法的性质很好，但顶层的回归器的训练数据较少：(1)即使顶层回归器给出的目标函数残差模型是低分辨率的，在先验中的精细结构一直坚持到了µ_k；(2)由于对σ'_ k的估计是不精确的，用σ_{k−1}进行平均，会带来改进的估计。进一步，当顶层回归器有密集支持时，β → 1，σ_k → σ'_k，符合我们的期望。

We provide details in the pseudocode in Algorithm 1, and illustrate the regressors in Figure 5. 我们在算法1中的伪代码给出了细节，在图5中描述了回归器。

Algorithm 1 is then used in the Batched Gaussian Process Bandits [8] algorithm. Algorithm 1 has the property that for a sufficiently dense sampling of the feasible region in the training data for the current study, the predictions converge to those of a regressor trained only on the current study data. This ensures a certain degree of robustness: badly chosen priors will eventually be overwhelmed (design goal 3).

算法1然后用于批量高斯过程老虎机算法。算法1有下面的性质，对于当前study的训练数据的一个充分密集采样的可行区域，预测收敛到了只在当前study数据中训练的回归器。这确保了一定程度的稳健性：选择不好的先验最终会被克服（设计目标3）。

In production settings, transfer learning is often particularly valuable when the number of trials per study is relatively small, but there are many such studies. For example, certain production machine learning systems may be very expensive to train, limiting the number of trials that can be run for hyperparameter tuning, yet are mission critical for a business and are thus worked on year after year. Over time, the total number of trials spanning several small hyperparameter tuning runs can be quite informative. Our transfer learning scheme is particularly well-suited to this case, as illustrated in section 4.3.

在生产设置中，在每个study中的trials数量相对较小时，迁移学习通常特别宝贵，但有很多这样的studies。比如，特定的生产机器学习系统在训练时会非常昂贵，限制了运行进行超参数调整的trials的数量，但是对于business来说是很关键的，是一年一年都在运行的。随着时间的进行，支撑几个小型的超参数调整的trials的总数量，信息量会非常大。我们的迁移学习方案非常适合于这种情况，如4.3节所示。

## 4 Results

### 4.1 Performance Evaluation

To evaluate the performance of Google Vizier we require functions that can be used to benchmark the results. These are pre-selected, easily calculated functions with known optimal points that have proven challenging for black-box optimization algorithms. We can measure the success of an optimizer on a benchmark function f by its final optimality gap. That is, if x∗ minimizes f , and \hat x is the best solution found by the optimizer, then |f(\hat x) − f(x∗)| measures the success of that optimizer on that function. If, as is frequently the case, the optimizer has a stochastic component, we then calculate the average optimality gap by averaging over multiple runs of the optimizer on the same benchmark function.

为评估Google Vizier的性能，我们需要可以用于基准测试结果的函数。这些是预先选择的，很容易的计算的函数，有已知的最优值点，对于黑箱优化算法已经证明是很有挑战性的。我们可以用其最终最优性差异，度量一个优化器在一个基准测试函数f上的成功。即，如果x*使f最小化，而且\hat x是优化器找到的最优解，那么|f(\hat x) − f(x∗)|度量了优化器在这个函数上的成功性。如果优化器有一个随机的组成部分，而且通常都是这种情况，我们然后通过将优化器在相同的基准测试函数上运行多次，计算平均的优化空间。

Comparing between benchmarks is a more difficult given that the different benchmark functions have different ranges and difficulties. For example, a good black-box optimizer applied to the Rastrigin function might achieve an optimality gap of 160, while simple random sampling of the Beale function can quickly achieve an optimality gap of 60 [10]. We normalize for this by taking the ratio of the optimality gap to the optimality gap of Random Search on the same function under the same conditions. Once normalized, we average over the benchmarks to get a single value representing an optimizer’s performance.

在基准测试之间的比较是更困难的，因为不同的基准测试函数有着不同的范围和难度。比如，一个好的黑箱优化器应用到Rastrigin函数，会得到的优化性差距是160，而Beale函数的简单的随机采样会迅速的获得优化性差异60。我们将其进行了归一化，将优化性差异除以在同样函数在同样条件下的Random Search上的优化性差异。一旦归一化之后，我们在基准测试之间进行平均，以得到单个值，表示一个优化器的性能。

The benchmarks selected were primarily taken from the BlackBox Optimization Benchmarking Workshop [10] (an academic competition for black–box optimizers), and include the Beale, Branin, Ellipsoidal, Rastrigin, Rosenbrock, Six Hump Camel, Sphere, and Styblinski benchmark functions.

选择的基准测试是从黑箱优化基准测试工作室中得到的（这是一个黑箱优化器的学术比赛），包括Beale, Branin, Ellipsoidal, Rastrigin, Rosenbrock, Six Hump Camel, Sphere, and Styblinski基准测试函数。

### 4.2 Empirical Results

In Figures 6 we look at result quality for four optimization algorithms currently implemented in the Vizier framework: a multiarmed bandit technique using a Gaussian process regressor [29], the SMAC algorithm [19], the Covariance Matrix Adaption Evolution Strategy (CMA-ES) [16], and a probabilistic search method of our own. For a given dimension d, we generalized each benchmark function into a d dimensional space, ran each optimizer on each benchmark 100 times, and recorded the intermediate results (averaging these over the multiple runs). Figure 6 shows their improvement over Random Search; the horizontal axis represents the number of trials have been evaluated, while the vertical axis indicates each optimality gap as a fraction of the Random Search optimality gap at the same point. The 2×Random Search curve is the Random Search algorithm when it was allowed to sample two points for each point the other algorithms evaluated. While some authors have claimed that 2×Random Search is highly competitive with Bayesian Optimization methods [20], our data suggests this is only true when the dimensionality of the problem is sufficiently high (e.g., over 16).

图6中，我们看到了4种优化算法的结果质量，在Vizier框架中进行了实现：使用了高斯过程回归器的多臂老虎机技术，SMAC算法，协方差矩阵自适应演化策略(CMA-ES)，和我们自己的概率搜索方法。对于给定的维度d，我们将每个基准测试函数泛化到了一个d维空间，将每个优化器在每个基准测试上运行100次，记录中间结果（在多次运行中平均这些结果）。图6展示的是对随机搜索方法的改进；水平轴表示的是评估的trials数量，竖直轴表示每个优化性差距在同一点上除以随机搜索的优化性差距。2×随机搜索曲线是可以对每个其他算法评估的点都可以采样两个点，再使用随机搜索算法的结果。一些作者声称2×Random Search与贝叶斯优化方法相比很有竞争力，我们的数据说明，只有当问题的维度足够高时，这才是真的。

### 4.3 Transfer Learning

We display the value of transfer learning in Figure 7 with a series of short studies; each study is just six trials long. Even so, one can see that transfer learning from one study to the next leads to steady progress towards the optimum, as the stack of regressors gradually builds up information about the shape of the objective function.

我们在图7中用一系列短的studies来展示了迁移学习的值；每个study只有6个trials那么长。即使如此，我们可以看到，从一个study到下一个的迁移学习，会带来向最优值的稳定改进，回归器族逐渐构建起了向目标函数形状的信息。

This experiment is conducted in a 10 dimensional space, using the 8 black-box functions described in section 4.1. We run 30 studies (180 trials) and each study uses transfer learning from all previous studies.

这个试验在10维空间进行，使用了4.1节所述的8个黑箱函数。我们运行30个studies（180 trials），每个study使用从其他之前的studies的迁移学习。

As one might hope, transfer learning causes the GP bandit algorithm to show a strong systematic decrease in the optimality gap from study to study, with its final average optimality gap 37% the size of Random Search’s. As expected, Random Search shows no systematic improvement in its optimality gap from study to study.

一个人会希望，迁移学习导致GP bandit算法从一个study到另一个study会展现出很优化性差距的系统性的降低，其最终的平均优化性差距是随机搜索37%的大小。就像期待的一样，随机搜索从一个study到另一个study并没有展现出优化性差距的系统性改进。

Note that a systematic improvement in the optimality gap is a difficult task since each study gets a budget of only 6 trials whilst operating in a 10 dimensional space, and the GP regressor is optimizing 8 internal hyperparameters for each study. By any reasonable measure, a single study’s data is insufficient for the regressor to learn much about the shape of the objective function.

注意，优化性差距的系统性改进是一个很困难的任务，因为每个study只有6个trials，而在10维空间进行操作，而GP回归器对每个study优化8个内部超参数。单个study的数据不足以让回归器学习到目标函数的形状的很多信息。

### 4.4 Automated Stopping

4.4.1 Performance Curve Stopping Rule. In our experiments, we found that the use of the performance curve stopping rule resulted in achieving optimality gaps comparable to those achieved without the stopping rule, while using approximately 50% fewer CPU-hours when tuning hyperparameter for deep neural networks. Our result is in line with figures reported by other researchers, while using a more flexible non-parametric model (e.g., Domhan et al. [9] report reductions in the 40% to 60% range on three ML hyperparameter tuning benchmarks).

性能曲线停止准则。在我们的试验中，我们发现，在深度神经网络应用中，使用性能曲线停止准则，得到的优化性差距，与那些不使用停止准则得到的类似，而使用的CPU-小时数则大约只有50%。我们的结果与其他研究者给出的图标结果是一致的，而使用了更灵活的非参数模型。

4.4.2 Median Automated Stopping Rule. We evaluated the Median Stopping Rule for several hyperparameter search problems, including a state-of-the-art residual network architecture based on [17] for image classification on CIFAR10 with 16 tunable hyperparameters, and an LSTM architecture [33] for language modeling on the Penn TreeBank data set with 12 tunable hyperparameters. We observed that in all cases the stopping rule consistently achieved a factor two to three speedup over random search, while always finding the best performing Trial. Li et al. [20] argued that “2X random search”, i.e., random search at twice the speed, is competitive with several state-of-the-art black-box optimization methods on a broad range of benchmarks. The robustness of the stopping rule was also evaluated by running repeated simulations on a large set of completed random search trials under random permutation, which showed that the algorithm almost never decided to stop the ultimately-best-performing trial early.

中值自动停止准则。我们对几个超参数搜索问题评估了中值停止准则，包括一个目前最好的残差网络架构，基于[17]，在CIFAR10上进行图像分类，有16个可调节的超参数，和一个进行语言建模的LSTM架构，有12个可调节超参数。我们观察到，在所有情况中，停止准则都会随机搜索一直有2-3倍的加速，而总会找到表现最好的trial。Li等[20]认为，2X随机搜索，即两倍速的随机搜索，与几个目前最好的黑箱优化方法，在很多基准测试中的效果都是类似的。停止准则的稳健性，通过在很大的完成的随机搜索trials在随机排列下的重复的仿真，也进行了评估，展现出算法几乎没有决定早一些停止那些最终表现最好的trial。

## 5 Use Cases

Vizier is used for a number of different application domains. Vizier用于几种不同的应用领域。

### 5.1 Hyperparameter tuning and HyperTune

Vizier is used across Google to optimize hyperparameters of machine learning models, both for research and production models. Our implementation scales to service the entire hyperparameter tuning workload across Alphabet, which is extensive. As one (admittedly extreme) example, Collins et al. [6] used Vizier to perform hyperparameter tuning studies that collectively contained millions of trials for a research project investigating the capacity of different recurrent neural network architectures. In this context, a single trial involved training a distinct machine learning model using different hyperparameter values. That research project would not be possible without effective black–box optimization. For other research projects, automating the arduous and tedious task of hyperparameter tuning accelerates their progress.

Vizier在Google中用于优化机器学习模型的超参数，包括研究和生产。我们的实现服务了Alphabet中的所有超参数调节workload，这个任务非常重。一个极端的例子是，Collins等[6]使用Vizier进行超参数调节研究，共计有上百万次trials，研究了不同的RNN架构的能力。在这个上下文中，单次trial涉及到训练一个独特的机器学习模型，使用不同的超参数值。如果没有高效的黑盒优化工具，这个研究项目就不会可能。对于其他的研究项目，自动调节超参数也加速了其进展。

Perhaps even more importantly, Vizier has made notable improvements to production models underlying many Google products, resulting in measurably better user experiences for over a billion people. External researchers and developers can achieve the same benefits using Google Cloud Machine Learning HyperTune subsystem, which benefits from our experience and technology.

可能更重要的是，Vizier对很多Google产品中使用的生产模型进行了很大改进，对超过十亿人得到了更好的用户体验。外部研究者和开发者在使用Google云机器学习HyperTune子系统时，也可以得到相同的好处。

### 5.2 Automated A/B testing

In addition to tuning hyperparameters, Vizier has a number of other uses. It is used for automated A/B testing of Google web properties, for example tuning user–interface parameters such as font and thumbnail sizes, color schema, and spacing, or traffic-serving parameters such as the relative importance of various signals in determining which items to show to a user. An example of the latter would be “how should the search results returned from Google Maps trade off search-relevance for distance from the user?”

除了调节超参数，Vizier还有几个其他的用处。可以用于Google网页的自动A/B测试，比如调节用户界面参数，比如字体和缩略图大小，色彩方案，和间距，或流量服务的参数，比如各种信号的相对重要性，确定要展示给用户的那些项。后者的一个例子是，Google地图返回的搜索结果，与距离用户的搜索相关性，应当怎样折中？

### 5.3 Delicious Chocolate Chip Cookies

Vizier is also used to solve complex black–box optimization problems arising from physical design or logistical problems. Here we present an example that highlights some additional capabilities of the system: finding the most delicious chocolate chip cookie recipe from a parameterized space of recipes.

Vizier还可以用于求解复杂的物理设计或逻辑问题带来的黑箱优化问题。这里我们给出一个例子，强调系统的一些额外能力：从参数化的配方空间中，找到最可口的巧克力饼干配方。

Parameters included baking soda, brown sugar, white sugar, butter, vanilla, egg, flour, chocolate, chip type, salt, cayenne, orange extract, baking time, and baking temperature. We provided recipes to contractors responsible for providing desserts for Google employees. The head chefs among the contractors were given discretion to alter parameters if (and only if) they strongly believed it to be necessary, but would carefully note what alterations were made. The cookies were baked, and distributed to the cafes for taste–testing. Cafe goers tasted the cookies and provided feedback via a survey. Survey results were aggregated and the results were sent back to Vizier. The “machine learning cookies” were provided about twice a week over several weeks.

参数包括baking soda, brown sugar, white sugar, butter, vanilla, egg, flour, chocolate, chip type, salt, cayenne, orange extract, baking time, and baking temperature。我们给承包商配方，承包商给谷歌雇员提供甜品。承包商中的主厨，如果觉得必要性非常强的话，可以改变参数，但必须小心的标出进行了什么变化。饼干烘焙后，分发到咖啡馆去尝试味道。去咖啡馆的人尝试饼干的味道，通过调查给出反馈。调查结果汇集到一起，结果发回到Vizier。这些机器学习饼干一周提供2次，一共几个星期。

The cookies improved significantly over time; later rounds were extremely well-rated and, in the authors’ opinions, delicious. However, we wish to highlight the following capabilities of Vizier the cookie design experiment exercised:

饼干随着时间改善明显；后面几轮的评分非常好，在作者的观念中，非常好吃。但是，我们希望强调Vizier在饼干设计试验中的如下能力：

- Infeasible trials: In real applications, some trials may be infeasible, meaning they cannot be evaluated for reasons that are intrinsic to the parameter settings. Very high learning rates may cause training to diverge, leading to garbage models. In this example: very low levels of butter may make your cookie dough impossibly crumbly and incohesive.

不可行的尝试：在真实的应用中，一些trials是不可行的，意思是不能进行评估，这是参数设置所内在的。很高的学习速率会导致训练发散，得到垃圾模型。在这个例子中，butter太少，会导致饼干面团易碎。

- Manual overrides of suggested trials: Sometimes you cannot evaluate the suggested trial or else mistakenly evaluate a different trial than the one asked for. For example, when baking you might be running low on an ingredient and have to settle for less than the recommended amount.

建议的trials的手工覆盖：有时候不能评估建议的trial，或错误评估的不同的trial，而不是要求的那个。比如，在烘焙时，可能会用完一种成分，不得不比推荐的量要少。

- Transfer learning: Before starting to bake at large scale, we baked some recipes in a smaller scale run-through. This provided useful data that we could transfer learn from when baking at scale. Conditions were not identical, however, resulting in some unexpected consequences. For example, the dough was allowed to sit longer in large-scale production which unexpectedly, and somewhat dramatically, increased the subjective spiciness of the cookies for trials that involved cayenne. Fortunately, our transfer learning scheme is relatively robust to such shifts.

迁移学习：在进行大规模烘焙之前，我们以小规模烘焙一些配方。这会提供有用的数据，迁移到我们在大规模烘焙的时候。但是条件并不是一致的，会得到一些不在期望中的结果。比如，面团在大规模生产中可以sit更长的时间，这会增加饼干的主观辣度。幸运的是，我们的迁移学习方案对这种变化是相对稳健的。

Vizier supports marking trials as infeasible, in which case they do not receive an objective value. In the case of Bayesian Optimization, previous work either assigns them a particularly bad objective value, attempts to incorporate a probability of infeasibility into the acquisition function to penalize points that are likely to be infeasible [3], or tries to explicitly model the shape of the infeasible region [11, 12]. We take the first approach, which is simple and fairly effective for the applications we consider. Regarding manual overrides, Vizier’s stateless design makes it easy to support updating or deleting trials; we simply update the trial state on the database. For details on transfer learning, refer to section 3.3.

Vizier支持将trials标记为不可行，在这些情况中，他们不会得到一个目标值。在贝叶斯优化的情况中，之前的工作要么指定了一个特别坏的目标值，试图将不可行的概率纳入到获得函数中，以惩罚很可能是不可行的点，或试图显式的对不可行区域的形状进行建模。我们采取第一种方法，这非常简单，对我们考虑的应用非常有效。至于手工覆盖，Vizier的无状态设计使其很容易支持更新或删除trials；我们只需要在数据库中更新trial的状态。对于迁移学习的细节，请参考3.3节。

## 6 Conclusion

We have presented our design for Vizier, a scalable, state-of-the-art internal service for black–box optimization within Google, explained many of its design choices, and described its use cases and benefits. It has already proven to be a valuable platform for research and development, and we expect it will only grow more so as the area of black–box optimization grows in importance. Also, it designs excellent cookies, which is a very rare capability among computational systems.

我们给出Vizier的设计思路，这是一个可扩展的目前最好的黑箱优化Google内部服务，解释了很多其设计选择，描述了其使用案例和好处。已经证明了这是一个研究和开发的宝贵平台，我们期望其会不断发展，黑箱优化这个领域的重要性也会不断增长。同时，它还可以设计很好的饼干，这是计算系统中很少具有这种能力的。