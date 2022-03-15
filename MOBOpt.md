# MOBOpt — multi-objective Bayesian optimization

Paulo Paneque Galuzio et. al. @ Brazil

## 0. Abstract

This work presents a new software, programmed as a Python class, that implements a multi-objective Bayesian optimization algorithm. The proposed method is able to calculate the Pareto front approximation of optimization problems with fewer objective functions evaluations than other methods, which makes it appropriate for costly objectives. The software was extensively tested on benchmark functions for optimization, and it was able to obtain Pareto Function approximations for the benchmarks with as many as 20 objective function evaluations, those results were obtained for problems with different dimensionalities and constraints.

本文给出了一款新的软件，一个Python类，实现了多目标贝叶斯优化算法。提出的方法可以计算优化问题的Pareto front近似，比其他方法所用的目标函数评估要少，非常适用于昂贵的目标函数。软件在优化用的基准测试函数上进行了广泛测试，用最多20次目标函数评估，就得到了基准测试的Pareto函数近似，对于各种不同维度和约束的问题，都可以得到这样的结果。

## 1. 1. Motivation and significance

Optimization of designs and processes constitutes an ubiquitous open problem in science and engineering. Even though there are efficient optimization methods available that work reasonably well for complex problems (for example, see [1]), if the function to be optimized is very costly, most of these methods become undesirable as they rely on a very large number of evaluations of the objective functions [2–4]. Mathematically, the problem of optimization can be simply stated as finding the argument values that return the minimal (or maximal) value of a given function, which is referred to as objective function or objective. In most real-world problems there are more than one, usually conflicting, objective functions to be simultaneously optimized, which leads to the following categorization for the problem: single-objective optimization when there is only one objective function; multi-objective optimization for up to four objectives; and many-objective optimization for more than four.

设计和过程的优化，是科学和技术中的一个到处存在的问题。对于复杂的问题，虽然现在有一些高效的优化方法，效果还不错（如[1]），如果要优化的函数计算起来很昂贵，多数这些方法就变得不那么好用了，因为它们通常都依赖于对目标函数大量的评估。数学上来说，优化问题可以简单的表述为找到的参数值可以得到给定函数的最大或最小值，这个函数就称为目标函数或目标。在多数真实世界问题中，要同时优化的目标函数都不止一个，通常还是有冲突的，这就是下面的问题分类：当只有一个目标函数时，是单目标优化；不超过4个目标函数时，称为多目标优化；多于4个目标函数时，是众目标优化。

To optimize many simultaneous objectives is challenging since, in most situations, they are conflicting, i.e., the optimal values are in different regions of the search space, and optimizing one objective function means that other objectives will be non-optimal. In this scenario, the solution to the optimization problem is a set that represents the compromise among the many objectives, called the Pareto front of the problem. To address this problem there are a number of different optimization methods proposed in the literature, of which a significant fraction are heuristics or metaheuristics [5,6], many inspired by biological problems [7–11]. Most of these approaches rely on a large number of evaluations of the objectives, which are then interpreted as a fitness or cost function assigned to an individual, later subjected to some biologically inspired selection and evolution criteria. Although efficient, these methods need to probe the values of the objectives several times, which becomes prohibitive if the objective functions are very costly, as the result of some time consuming experiment or of some numerically expensive simulation[3].

同时优化多个目标是很有挑战的，因为在多数情况中，它们都是有冲突的，即，最优值是在搜索空间的不同区域，优化一个目标函数会意味着另一个目标就不是最优的。在这种情况下，优化问题的解是一个集合，表示了众多目标之间的折中，称为问题的Pareto front。为处理这个问题，在文献中有很多不同的优化方法，很多是启发式或元启发式，很多是受生物问题的启发。多数方法依赖于目标函数的大量评估，然后翻译为单个点的适应度或代价函数，后面会根据一些生物启发的选择和进化规则进行操作。虽然很高效，这些方法需要计算目标的值若干次，如果目标函数计算起来很昂贵，就变得不太可行了，因为有时候会涉及到试验，或一些数值计算很昂贵的仿真。

In order to optimize this class of continuous real-valued costly functions, Bayesian optimization methods were proposed as an alternative for the existing ones [12–16]. Substantiated by the Bayes theorem, they present two major advantages over other conventional methods [2,17]: (i) they are highly efficient regarding the number of objective function evaluations; (ii) they do not require any analytical knowledge of the objectives, allowing the methods to perform well with black-box functions. Also, Bayesian methods work well even when the objective functions are multimodal or non-convex [17,18].

为优化这类昂贵的连续的实值函数，贝叶斯优化方法是现有方法之外的另一个选择。与其他方法相比，有两个主要的优势，这一点贝叶斯定理证实了：(1)在目标函数评估上，它们非常高效；(2)不需要知道目标的任何解析知识，使这种方法对黑箱函数效果很好。同时，在目标函数多峰或非凸的情况下，贝叶斯方法的效果也很好。

For expensive objective functions, the simulation bottleneck lies on the number of evaluations of the objectives required by optimization algorithms, in which case one looks for a method that needs fewer iterations [19]. A popular approach consists of building a model of the objectives, which are evaluated as few as possible, and through successively minimizing a loss function built on top of such model, obtain an approximation to the Pareto front. The choice of the loss function is where most existing Bayesian algorithms differ [20–27]. The reduced number of evaluations of the objectives lead to undersampled solutions of the optimization problem, which is observed in current implementations of Bayesian algorithms [28,29]. The method proposed in this paper addresses this problem by taking advantage of well established multi-objective optimization methods that are known to obtain qualitatively good Pareto fronts [30].

对于昂贵的目标函数，仿真瓶颈在于，优化算法需要对目标函数评估若干次，这样就要找到一种需要尽可能少次数迭代的方法。一种流行的方法对目标进行建模，评估的次数尽可能的少，通过在这个模型的基础上的损失函数的后续最小化，得到Pareto front的近似。损失函数的选择，是多数现有的贝叶斯方法不一样的地方。目标函数评估次数的减少，得到优化问题解的降采样，这在贝叶斯算法的当前实现中可以观察到。本文提出的方法，通过利用公认的可以得到很好的Pareto front的多目标优化方法，来解决这个问题。

### 1.1 Mathemtical description

An optimization problem can be represented by a pair (Ω, ⃗f) [31], where Ω ⊆ Rn is the search space, and n is its dimension. ⃗f is a m-dimensional real valued vector function of Ω, such as:

一个优化问题可以表示为一个对(Ω, ⃗f)，其中Ω ⊆ Rn是搜索空间，n是其维度。 ⃗f是Ω上的m维的实值向量函数，有

$$ ⃗f : Ω ↦→ Y$$(1)

where Y ⊆ Rm is the objective space. The n components of ⃗x ∈ Ω are also called design variables. To solve the multi-objective optimization problem means finding the set of values {⃗x⋆} ⊂ Ω, called the Pareto Set, which elements satisfy the condition:

其中Y ⊆ Rm是目标空间。 ⃗x ∈ Ω中的n个组成元素也称为设计变量。求解多目标优化问题，意味着找到集合值{⃗x⋆} ⊂ Ω，称为Pareto集，其中的元素满足下列条件

$$ ⃗x⋆ = argmax_{s.t. ⃗x∈Ω} (f1(⃗x), f2(⃗x), . . . , fm(⃗x))$$(2)

However, for most typical problems, there is no ⃗x ∈ Ω that simultaneously maximize (or minimize) all the m objective functions, as these are conflicting.

但是，对于多数典型的问题，不存在可以同时最大化或最小化所有m个目标函数的 ⃗x ∈ Ω，因为它们是互相冲突的。

Two objective functions fi(⃗x) and fj(⃗x) are said to be conflicting if the values of ⃗ ⃗x that lead to the maximum value of fi are different from the values that maximize fj, in a way that these two objectives cannot be maximized with the same value of ⃗x. In this context, the Pareto set {⃗x⋆} represents a compromise between the conflicting objectives, which is represented through the concept of dominance, defined as follows:

两个目标函数fi(⃗x)和fj(⃗x)是冲突的，即fi得到最大值的 ⃗x，与fj得到最大值的 ⃗x是不一样的，这两个目标函数不能用同一个 ⃗x得到最大值。在这个上下文中，Pareto集{⃗x⋆}表示互相冲突的目标之间的折中，这是通过占优的概念表示的，定义如下：

a⃗⃗ ≻ b⃗ (a⃗ dominates b⃗) if ⃗f (a⃗) ≥ ⃗f (b⃗), where fi(a⃗) > fi(b⃗) for at least one component i of ⃗f.

The comparisons between vectors are made elementwise, i.e., for a⃗ ∈ Rm and b⃗ ∈ Rm, we say that a⃗ > b⃗ if and only if aµ > bµ for all µ ∈ {1, 2, 3, . . . , m}. Therefore, the Pareto Set is given by

$${⃗x⋆} = {⃗x∗ ∈ Ω | ⃗x∗ ≻ ⃗x, ∀⃗x ∈ Ω}$$(3)

which represents the subset of search space that contains only non-dominated points. The image of {⃗x⋆} in objective space is called the Pareto front of the problem and it is represented by the set F [3]

$$F = {⃗f (⃗x) | ⃗x ∈ {⃗x⋆}}$$(4)

To generalize the optimization problem, one can impose further restrictions on the search space on the form of Ng inequality constraints [32], which can be arbitrarily represented by:

$$gk(⃗x) ≤ 0, k = 1, . . . , Ng$$(5)

where gk(⃗x) are functions on the search space. Regions of Ω in which the constraints are satisfied are called valid. When there are constraints in the problem, the Pareto set definition must be updated:

$${⃗x⋆} = {⃗x∗ ∈ Ω | ⃗x∗ ≻ ⃗x, ∀⃗x ∈ Ω ∧ gk(⃗x) ≤ 0, k = 1, . . . , Ng }$$(6)

## 2. Software description

A general description of the algorithm behind the Bayesian method is given in Section 2.1. Following, details of the current implementation are provided in Section 2.2.

2.1节给出了贝叶斯方法算法的一般性描述，2.2节给出了当前实现的细节。

### 2.1. Software architecture

The Bayesian method is built upon the idea of constructing models for the objective functions, so that the models can be optimized instead of the objectives themselves [2]. In this way, the total number of objective functions evaluations is expected to be reduced to the minimal necessary to build the model accurately in the vicinity of the Pareto front of the problem. A schematic description of the algorithm proposed is depicted in Fig. 1, where each major logic step of it is labeled by a number from 1 to 8.

贝叶斯方法的基础思想是，为目标函数构建模型，这样模型可以被优化，而不是目标函数本身。这样，目标函数评估的次数，是在问题的Pareto front的附近准确构建模型所必须的最小次数。图1给出了算法的简单描述，每个主要的逻辑步骤用数字1到8进行了标注。

In order to build the model of the objectives it is necessary to sample them at different regions of the search space, preferably we would like the samples to be as close as possible to the Pareto set of the problem, so as to increase the accuracy of the model in this region and consequently reduce the total number of functions evaluations. However, in the first steps of the algorithm, as nothing is yet known about the Pareto set, it is necessary to evaluate the objectives N_init times, at randomly selected points. The information about our observations of each objective fi(⃗x) is stored in the set Di(t), defined as:

为构建目标函数的模型，必须要在搜索空间的不同区域对其进行采样，我们希望样本尽可能接近问题的Pareto front，这样可以增加模型在这个区域中的准确性，最后降低函数评估的总次数。但是，在算法的开始步骤中，因为不知道关于Pareto集的任何信息，需要在随机选取的点上评估目标函数N_init次。每个目标函数fi(⃗x)上我们观察到的信息都存储在集合Di(t)中，定义为：

$$Di(t) = {⃗x^(1), fi(⃗x^(1)); ⃗x^(2), fi(⃗x^(2));. . .; ⃗x^(t), fi(⃗x^(t))}$$(7)

which is just a collection of pairs of points in the search space ⃗x(t) and its image of the ith objective function fi(⃗x(t)). In Eq. (7), t stands for the number of points sampled so far in the algorithm, such that, at the start of the method t = N_init. This is the first step represented in Fig. 1.

这是在搜索空间中的点 ⃗x^(t)和其第i个目标函数fi(⃗x^(t))的映射的点形成的点对的集合。在式(7)中，t表示算法中目前采样的点的数量，这样，在算法的开始t=N_init。这是图1中的步骤1。

Given Di(t), it is possible to construct a model ηi(⃗x) for the ith objective function fi(⃗x):

给定Di(t)，可能对第i个目标函数fi(⃗x)构建一个模型ηi(⃗x)：

$$ηi(⃗x;t) = G[Di(t); C_{3/2}](⃗x) − ζ \sum_{k=1}^{N_g} Θ(g_k(⃗x))$$(8)

where ζ is a penalty constant factor, Θ(·) is the Heaviside step function and gk(⃗x) are the constraints defined in Eq. (5); G[D; K] is a Gaussian Process (GP), where the first argument was defined in Eq. (7), and the second argument (K) is the kernel or covariance function used in the GP. Observe that in the valid region, the penalty term vanishes and ηi ≈ fi. Also, as no model is built for gk(⃗x), expensive constraints are prohibitive, which is a limitation of the method. The construction of the model corresponds to the third step of the algorithm, as depicted in Fig. 1.

其中ζ是一个惩罚常数因子，Θ(·)是Heaviside阶跃函数，gk(⃗x)是式(5)中定义的约束；G[D; K]是高斯过程(GP)，其中第一个参数在式(7)中定义，第二个参数K是GP中的核或协方差函数。在有效区域，惩罚项消失，ηi ≈ fi。同时，因为对gk(⃗x)并没有构建模型，所以昂贵的约束计算起来就很麻烦，这是这种方法的一个局限。模型的构建对应着算法的第三步，如图1所示。

Gaussian Processes are distributions over functions [33], completely specified by their mean and covariance function [2]. They can be understood as a random function that, for each given value ⃗x, returns the mean and variance of a Gaussian distribution that better describes fi(⃗x), given our knowledge of fi contained in Di, and our estimate of how these observations correlate to each other represented by the kernel.

高斯过程是函数的分布，完全由其均值和协方差函数指定。这可以理解为一个随机函数，给每个给定的值 ⃗x，返回的是更好的描述fi(⃗x)的高斯过程的均值和方差，给定在Di中fi的知识，以及这些观察怎样相互关联的估计，由核来表示。

In this work we used the Matérn covariance function Cν (⃗x, ⃗x′)[33,34], given by: 在本文中，我们使用Matérn协方差函数Cν (⃗x, ⃗x′)，由下式给定

$$Cν (⃗x, ⃗x′) = \frac {2^{1-ν}}{Γ(ν)} (\frac {\sqrt_{2ν}r}{ℓ})^ν K_ν \frac {\sqrt_{2ν}r}{ℓ}$$(9)

where ℓ is a typical length scale, Γ(·) is the Gamma function, Kν (·) is the modified Bessel function of the second kind, r = ∥⃗x − ⃗x′∥ is the distance between the two arguments of the kernel and ν is a positive parameter, such that the process η(⃗x) is k-times mean-square differentiable if and only if ν > k [33]. In this work, we adopted ν = 3/2, so we require only once differentiability in our models for the objective functions.

其中ℓ是一个典型的长度尺度，Γ(·)是gamma函数，Kν (·)是第二类修正Bessel函数，r = ∥⃗x − ⃗x′∥是核中两个参数的距离，ν是正参数，这样只有在条件ν > k时，过程η(⃗x)是k次均方可微的。在本文中，我们采用ν = 3/2，这样我们只需要模型中的目标函数一次可微性。

Pareto front approximation. After t observations of the objective functions, a model ηi(⃗x;t) is obtained by applying Eq. (8), which can then be used to calculate an approximation to the Pareto front of the objectives (F). The Pareto front approximation (PFA) at time t is denoted Φt, and the Pareto set approximation (PSA) that generates Φt is denoted χt, i.e., ξ⃗^(t)_ι ∈ χ_t ⊂ Ω ⇐⇒ ⃗f(ξ⃗^(t)_ι) ∈ Φ_t. The PFA can be obtained by employing a well established multi-objective optimization algorithm to the models ηi, which are much faster to evaluate than the actual objectives. If the models are accurate enough in the vicinity of the Pareto front of the problem, then the PFA is a good approximation of the Pareto front, i.e., Φt ≈ F. In this work, we selected the non-dominated sorting genetic algorithm II (NSGAII) [30] to find Φt. This is step 4 in Fig. 1. It is important to note that the NSGA-II is implemented in the method without any particular adaptation, from its perspective the approximate objectives ηi(⃗x;t) are optimized as if they were the actual objectives of the optimization problem being solved.

Pareto front近似。在对目标函数有了t次观察之后，通过应用式(8)可以得到一个模型ηi(⃗x;t)，然后可以用于计算目标函数F的Pareto front的近似。在t时刻的Pareto front近似(PFA)记为Φt，生成Φt的Pareto集近似(PSA)记为χt，即ξ⃗^(t)_ι ∈ χ_t ⊂ Ω ⇐⇒ ⃗f(ξ⃗^(t)_ι) ∈ Φ_t。PFA的得到，可以通过对模型ηi采用公认的多目标优化算法，这比实际的目标函数评估起来要快的多。如果模型在问题的Pareto front的附近是足够精确的，那么PFA就是Pareto front的一个很好的近似，即Φt ≈ F。本文中，我们用NSGAII算法来找到Φt。这是图1中的步骤4。

The method works iteratively, meaning that it needs a rule for selecting the next point in search space. To improve the quality of the PFA, i.e., to reduce the difference between Φt and F, it is of interest to select points close to {⃗x⋆} in the search space, in such a way as to make the models ηi(⃗x) more representative of their respective objective functions fi(⃗x) in the vicinity of F. However, at the same time it is important to evaluate undersampled regions of the search space Ω so as to guarantee that no region of F is overlooked and the algorithm gets trapped with an under-representation of the Pareto front. These two contradicting approaches – sampling closer to {⃗x⋆} or sampling Ω uniformly – represent the trade-off between exploitation and exploration.

本方法是迭代进行的，意思是需要一条规则来选择搜索空间中的下一个点。为改进PFA的质量，即，减少Φt和F之间的差异，需要在搜索空间中选择接近于{⃗x⋆}的点，要让模型ηi(⃗x)在F的附近对目标函数fi(⃗x)更有代表性。但是，同时也很重要的是，评估搜索空间Ω的降采样的区域，以确保没有忽视F的任何区域，算法陷入了Pareto front的降表示中。这两种矛盾的方法 - 在接近{⃗x⋆}处进行采样，或对Ω进行均匀的采样 - 表示了挖掘和探索之间的折中。

In order to pursue exploitation, we assume that our current PFA is a good approximation to the true Pareto front at current time, then the corresponding PSA approximates the actual Pareto set with the information available at time t, i.e., χt ≈ {⃗x⋆}, therefore points taken from χt are good candidates to be the next iteration point in the algorithm, i.e., ⃗x^(t+1) = ξ⃗^(t)_{ι_next} ∈ χt, where:

为追求挖掘，我们假设我们当前的PFA是在当前时刻是真实的Pareto front的一个很好的估计，那么对应的PSA用在时刻t可用的信息近似了真实的Pareto集，即，χt ≈ {⃗x⋆}，因此，从χt中取的点，是算法中下一次迭代点的很好的候选，即 ⃗x^(t+1) = ξ⃗^(t)_{ι_next} ∈ χt，其中

$$ι_{next} = argmax_{ι∈1,...,N} [q (\frac {d_ι,f − µ_f} {σ_f})+ (1-q) (\frac {d_ι,x − µ_x} {σ_x})]$$(10)

In Eq. (10), dι,x corresponds to the least distance from the ι-th point in χt to all other t points {⃗x^(1), . . . ⃗x^(t)} probed by the algorithm; µ_x is the average of all d_ι,x and σ_x their standard deviation. The same quantities are calculated in objective space and are represented by the index f. The method selects as the next iteration point the one that belongs to χ_t and is the farthest away from all previously observed points. This analysis can be made at the search or the objective space, the parameter q in Eq. (10) quantifies that choice, if q = 1 or q = 0 only distances in objective space or search space are considered, respectively. Intermediate values of q are also accepted and weight both spaces accordingly. This is step 5 in Fig. 1.

在式(10)中，dι,x对应从χt中的第ι个点到算法探索过的所有其他t个点 {⃗x^(1), . . . ⃗x^(t)}的最小距离；µ_x是所有d_ι,x的平均值，σ_x是标准差。同样的量在目标空间中也计算了一次，用索引f进行表示。这个方法选择的下一个迭代点，是属于χ_t的，而且是与所有之前观察过的点距离最远的。这种分析可以在搜索空间或目标函数空间进行，式(10)中的参数q量化了这个选项，如果q=1或q=0，分别只考虑目标函数空间或搜索空间中的距离。q的中间值也是接受的，对两个空间进行相应的加权。这是图1中的步骤5。

At every step, with probability given by the parameter r ∈ [0, 1], a randomly selected component of ⃗x^(t+1) is replaced by R(x_µ,min, x_µ,max), where R(a, b) is a uniform random number in the interval [a, b) and x_µ,min and x_µ,max are the limits to the µ-th component of ⃗x. This is step 6 in Fig. 1, and is implemented to increase the exploration of the search space by the method. The iteration stops when the set number of iterations was achieved.

在每一步中，以参数r ∈ [0, 1]给定的概率，随机选择 ⃗x^(t+1)的一个元素替换成R(x_µ,min, x_µ,max)，其中R(a, b)是区间[a,b)中的一个均匀随机数，x_µ,min和x_µ,max是 ⃗x的第µ个元素。这是图1中的步骤6，其实现是用来增加方法对搜索空间的探索。当达到设定的迭代次数时，迭代停止。

### 2.2. Software functionalities

The Multi-Objective Bayesian optimization algorithm is implemented as a Python class in the MOBOpt package. Its usage is centered around the MOBayesianOpt class, which can be instantiated as:

多目标贝叶斯优化算法在MOBOpt包中实现为一个Python类。其使用是围绕MOBayesianOpt类的，可以实例化为


```

import mobopt as mo

Optimizer = mo.MOBayesianopt(target=objective, NObj=n_obj, pbounds=bounds)

```

Where target is the function to be optimized, NObj is the number of objective functions, pbounds is a numpy array [35] specifying the bounds of the variables in search space. The function objective to be passed to the optimizer must be declared as:

其中target是要优化的函数，Nobj是目标函数的数量，pbounds是一个numpy阵列，指定了在搜索空间中变量的界限。要给优化器传递的函数目标要声明为：

```

def objective(x):

...

return numpy.array(f_{1}(x), f_{2}(x), ..., f_{n_obj}(x))

```

The initialization of the algorithm is called by the method initialize: 算法的初始化由方法initialize调用

```

Optimizer.initialize(init_points=N_init)

```

where the parameter init_points corresponds to the parameter Ninit, explained in Section 2.1. Finally, the main method of the class is called by: 其中参数init_points对应参数Ninit，在2.1节中进行了解释。最后，类的主要方法由下面调用：

```

front, pop = Optimizer.maximize(n_iter=N_iter)

```

where n_iter corresponds to the number of iteration of the method; the outputs front and pop correspond to the Pareto front and Pareto Set of the problem, respectively. Optional arguments for the methods, as well as other methods of the class are explained in details in the repository wiki page. The implementation of the algorithm was based on available open source packages [34–37].

其中n_iter对应方法的迭代次数；输出front和pop分别对应问题的Pareto front和Pareto Set。类的这个方法和其他方法的可选参数，在仓库wiki页进行了详细解释。算法的实现是基于开源包的。

## 3. Illustrative examples

To assess the efficiency of the proposed algorithm, it was tested on a set of benchmark functions [30]. These were chosen because they are well established in the field of optimization, and they have distinct characteristics from each other: their Pareto fronts have different topologies, which allows us to determine the best convergence scenarios; their search space have different dimensionalities; and their constraint conditions are different. For all these benchmark functions the True Pareto front is known, allowing to access the convergence of the method.

为评估提出的算法的效率，在一些基准测试函数集合上进行了测试。选择这些是因为它们是在优化领域公认的，它们互相之间都有非常不同的特征：它们的Pareto fronts有不同的拓扑，使我们可以确定最好的收敛场景；它们的搜索空间维度不同；它们的约束条件也是不一样的。对于这些基准测试函数，真实的Pareto front是已知的，使我们可以得到这些方法的收敛性。

Four different benchmarks were used as an example of Pareto front convergence, namely: Schaffer’s, Fonseca and Flemming’s, Poloni’s and Tanaka’s. Also, the 2-D ZDT1, from Zitzler, Deb and Thiele’s [38], function was used to infer parameter dependency of the method. All of the benchmarks were taken from reference [30] and are minimization problems, they are detailed in Table 1. Tanaka’s benchmark was chosen as an example of constrained problem, and it is subjected to the following constraint:

用了四种不同的基准测试，来作为Pareto front收敛性的例子，即：Schaffer's, Fonseca and Flemming’s, Poloni’s and Tanaka’s。同时，用了Zitzler中的2D ZDT1和Deb and Thiele’s [38]函数，来推断方法的参数依赖性。所有的基准测试都是从参考[30]中取的，是最小化问题，表1中给出了细节。Tanaka基准测试被选为约束问题的一个例子，有如下约束

$$g_1(x) = -x_1^2-x_2^2+1+0.1cos(16arctan(x_1/x_2))≤ 0$$(11)

$$g_2(x) = (x_1-0.5)^2+(x_2-0.5)^2≤ 0.5$$(12)

In Fig. 2 the Pareto front approximations obtained for each of the benchmark functions are represented for different number of function evaluations (denoted N). In all figures the black dots correspond to the True Pareto front of the problem. It can be seen that even for as few as 20 objective function evaluation, for the Schaffer benchmark (Fig. 2(a)), the algorithm is able to find accurate approximations for the Pareto fronts. Even for the most complicated functions, such as the Poloni benchmark with a discontinuous Pareto front, with few evaluations the method converges to a satisfactory solution. Also, the imposition of constraint does not limit the efficiency of the method, as can be seen in Fig. 2(d).

在图2中，对每个基准测试函数得到的Pareto front近似，对不同数量的函数评估进行了表示（表示为N）。在所有的图中，黑点表示问题的真实的Pareto front。可以看到，即使只有20次目标函数评估，对于Schaffer基准测试（图2a），算法仍然可以找到Pareto front的精确的近似。即使对于最复杂的函数，比如Poloni基准测试，其Pareto front是不连续的，在只有很少评估的情况下，收敛到了一个令人满意的解。同时，施加了约束也不会限制方法的效率，如图2d所示。

In order to provide further quantification of the convergence of the code, there are several metrics that can be used, one of the simplest is the generational distance (∆) defined as:

为对代码收敛进行进一步的量化，可以使用几种度量，一种最简单的是代际距离∆，定义为：

$$∆(Φ, F) = \frac {\sqrt_{\sum_{i=1}^{|Φ|} d_i^2}} {|Φ|}$$(13)

where di = min⃗_{f∈F} |F (φ⃗_i) − F (⃗f)| is the smallest distance from φ⃗_i ∈ Φ to the closest solution in F [39], where Φ is the optimal solution set found by the algorithm, and F is the True Pareto front of the problem.

其中di = min⃗_{f∈F} |F (φ⃗_i) − F (⃗f)|是从φ⃗_i ∈ Φ到F中的最近解的最小距离，其中Φ是算法找到的最优解集，F是问题的真实Pareto front。

In Fig. 3, ∆ is represented as a function of the method parameters q and r, in order to provide some general guide for the parameter choice, which can be problem dependent. Fig. 3 was calculated with 10 simulations for the 2-D ZDT1 benchmark, for different values of q and r. For each simulation there were Ninit = 5 initial points and 50 iterations. Generational distance was calculated and averaged for the 20 last iterations in every simulation.

在图3中，∆表示为参数q和r的函数，以对参数选择给出一些通用的指引，这可能是不依赖于问题的。图3是对2D ZDT1基准测试进行了10次仿真的结果，都用了不同的q和r值。对每次仿真，Ninit = 5，50次迭代。计算了代际距离，对每次仿真的最后20次迭代进行平均。

## 4. Impact

Optimization problems are not restricted to academic questions, but in fact have applications throughout different fields of science and technology. Even though there are many available methods that address this problem, most of them rely on many objective functions evaluations, which can become prohibitive whenever the objectives are numerically costly or are the outcome of experiments. In these situations, it is desirable to obtain an approximation to the Pareto front with as few evaluations of the objectives as possible.

优化问题不会局限在学术问题中，而是实际上在不同的科学技术领域有应用。即使有很多可用的解决这个问题的方法，它们多数依赖于很多次目标函数的评估，当目标函数在数值计算上很昂贵，或是试验的结果时，这就会变得不可行。在这些情况中，用尽可能少的目标函数评估次数，来得到Pareto front的近似，是很理想的。

The current software, by using a Bayesian algorithm, is able to calculate the Pareto front of the objectives efficiently in terms of objective functions evaluations. It does so through a simple user interface, written in Python, which allows it to be easily interfaced with other programming languages through the use of numpy arrays [35]. The user interface is straightforward and easy of use, structured around one simple class and few control parameters.

目前的软件使用了一种贝叶斯算法，可以很高效的计算目标函数的Pareto front，目标函数评估的次数很少。通过用Python写的简单的用户接口实现了这个功能，通过使用Numpy阵列，可以很容易的与其他编程语言进行接口。用户接口很直观，容易使用，只有一个简单的类，和几个控制参数。

Also, the software provides the user with many metrics so that convergence quantification can be assessed. In cases of longer simulations, there are methods implemented that allow the user to easily save the status of the simulation so that it can be resumed without the need of reinitialization.

同时，软件还为用户提供了很多度量，可以评估收敛的量化。在更长的仿真中，实现了一些方法，使用户可以很容易的保存仿真的状态，这样不需要重新初始化，就可以恢复状态。

This Bayesian method was tested on a more complicated and realistic problem, a thermal-hydraulic plate heat exchanger (HE) [40], where each simulation takes around ∼ 15 min on a Intel(R) Core(TM) i5-8265U CPU with 1.60 GHz and 8 GB of RAM. Other applications are, for example, optimizing refrigerator evaporators or hyperparameters of machine learning techniques, problems prohibitive for traditional multi-objective optimization algorithms requiring thousands of objectives evaluations.

这种贝叶斯方法在一个更复杂更实际的问题中进行了测试，a thermal-hydraulic plate heat exchanger (HE)，在一个Intel(R) Core(TM) i5-8265U CPU with 1.60 GHz and 8 GB of RAM机器上，每次仿真要大约15min。其他的应用还有，优化冷冻机的蒸发器，或机器学习系统的超参数，这对于需要几千次目标函数评估的传统多目标优化算法，是不可行的。

## 5. Conclusions

A multi-objective Bayesian optimization software was outlined. The software is able to calculate the Pareto front of optimization problems with fewer objective functions evaluations than most currently available optimization algorithms. The implementation is made in Python, so that it is easy of use and can be interfaced with many other programming languages. A series of benchmark functions were tested and the software was able to find reliable Pareto front approximations with only a few evaluations of the objectives.

给出了一个多目标贝叶斯优化软件。与现有的大多数优化算法相比，软件可以用更少的目标函数评估，计算优化问题的Pareto front。实现用了Python语言，这样很容易使用，与其他很多语言进行接口。测试了一系列基准测试函数，软件只用少数的目标函数评估，就可以找到可靠的Pareto front近似。