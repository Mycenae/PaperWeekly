# Bayesian Optimization for Adaptive Experimental Design: A Review

Steward Greenhill et. al. @ Deakin University, Australia

## 0. Abstract

Bayesian optimisation is a statistical method that efficiently models and optimises expensive "black-box" functions. This review considers the application of Bayesian optimisation to experimental design, in comparison to existing Design of Experiments (DOE) methods. Solutions are surveyed for a range of core issues in experimental design including: the incorporation of prior knowledge, high dimensional optimisation, constraints, batch evaluation, multiple objectives, multi-fidelity data, and mixed variable types.

贝叶斯优化是一种统计方法，可以对昂贵的黑箱函数进行高效的建模和优化。本回顾考虑了贝叶斯优化在试验设计中的应用，比较了已有的DOE方法。在试验设计中的核心问题的解进行了回顾，包括：纳入先验知识，高维优化，约束，批评估，多目标，多可信度数据，混合变量类型。

**Index terms** Bayesian methods, design for experiments, design optimization, machine learning algorithms.

## 1. Introduction

Experiments are fundamental to scientific and engineering practice. A well-designed experiment yields an empirical model of a process, which facilitates understanding and prediction of its behaviour. Experiments are often costly, so formal Design of Experiments methods (or DOE) [1]–[3] optimise measurement of the design space to give the best model from the fewest observations.

试验对于科学和工程实践来说是非常基础的。设计良好的试验会产生一个过程的经验模型，有利于其行为的理解和预测。试验通常很昂贵，所以正式的DOE优化设计空间的测量，来从最少的观测中给出最好的模型。

Models are important decision tools for design engineers. Understanding of design problems is enhanced when the design space can be explored cheaply and rapidly, allowing adjustment of the number and range of design variables, identification of ineffective constraints, balancing multiple design objectives, and optimisation [4]. Industrial processes must be robust to environmental conditions, component variation, and variability around a target [3]. Robust Parameter Design (RPD) [5]–[7] systematically characterises the influence of uncontrollable variables and noise. The number of observations required to build a model increases rapidly with the number of variables, making it challenging to investigate systems with many variables. Screening experiments can identify subsets of important variables to be later investigated in more detail [8], [9]. Optimisation is important in most industrial applications, and there are often multiple objectives which must balanced including yield, robustness, and cost. In classical experimental design, modelling and optimisation are separate processes, but newer model-based approaches can potentially sample more efficiently by adapting to the response surface, and can incorporate optimisation into the modelling process.

对于设计工程师来说，模型是很重要的决策工具。当设计空间可以廉价快速的被探索，可以调整设计变量的数值和范围，识别无效的约束，平衡多个设计目标和优化时，对设计问题的理解会得到加强。工业过程对于环境条件，组成部分的变化和围绕目标的变化，必须是稳健的。稳健参数设计(RPD)系统性的描述了不可控变量和噪声的影响的特征。为构建一个模型需要的观测的数量，随着变量的数量的增加迅速增加，这样对于有很多变量的系统，要进行研究就非常有挑战性。筛选试验可以识别重要变量子集，后面更详细的研究。优化在多数工业应用中都很重要，通常有多个目标需要平衡，包括产出，稳健性，和代价。在经典的试验设计中，建模和优化是分离的过程，但更新的基于模型的方法可以更高效的采样，对响应表面自适应，将优化纳入到建模过程中。

Machine learning has made great strides in the recent past, and we present here a machine learning approach to experimental design. Bayesian Optimisation (BO) [19], [20]is a powerful method for efficient global optimisation of expensive black-box functions. The experimental method introduces specific challenges: how to handle constraints, high dimensionality, mixed variable types, multiple objectives, parallel (batch) evaluation, and the transfer of prior knowledge. Several reviews have presented BO for a technical audience [20]–[22]. Our review surveys recent methods for systematically handling these challenges within BO framework, with an emphasis on applications in science and engineering, and in the context of modern experimental design.

机器学习最近有了很大进展，我们这里对试验设计给出一种机器学习的方法。贝叶斯优化是一种强力的方法，可以对昂贵的黑箱函数进行高效的全局优化。试验方法引入了具体的挑战：怎样处理约束，高维，混合变量类型，多目标，并行（批次）评估，先验知识的传递。几篇回顾文章将BO进行了介绍。我们回顾了在BO框架中系统性的处理这些挑战的方法，强调了在科学和工程中的应用，在现代试验设计的上下文中。

Bayesian optimisation is a sample efficient optimisation algorithm and thus suits optimisation of expensive, black-box systems. By “black-box” we mean that the objective function does not have a closed-form representation, does not provide function derivatives, and only allows point-wise evaluation. Several optimisation algorithms can handle optimisation of black-box functions such as multi-start derivative free local optimiser e.g. COBYLA [36], or evolutionary algorithms e.g. ISRES [37], or Lipschitzian methods such as DIRECT [34]. However, none of these are designed to be sample efficient, and all need to evaluate a function many times to perform optimisation. In contrast, Bayesian optimisation uses a model based approach with an adaptive sampling strategy to minimise the number of function evaluations.

贝叶斯优化是一种样本高效的优化算法，因此适合于优化昂贵的黑箱系统。黑箱的意思是，目标函数并没有闭合形式的表示，没有函数导数，只允许逐点的评估。几种优化算法可以进行黑箱函数的优化，如多起始无导数局部优化器，如COBYLA[36]，或演化算法，如ISRES[37]，或Lipschitzian方法，如DIRECT[34]。但是，这些方法都不是样本高效的，都需要评估一个函数很多次，以进行优化。比较起来，贝叶斯优化使用基于模型的方法，有自适应的采样策略，来最小化函数评估的数量。

Past approaches to experimental design have closely coupled sampling and modelling. Factorial designs assume a linear model and sample at orthogonal corners of the design space (see Figure 1). For more complex non-linear models, general purpose space-filling designs such as Latin hyper-cubes offer a more uniform coverage of the design space. For N sample points in k dimensions, there are (N!)^{k−1} possible Latin hypercube designs, and finding a suitable design involves balancing space-filling (e.g. via entropy, or potential energy) with other desirable properties such as orthogonality. Much literature exists on the design of Latin hypercubes, and many research issues remain open [10], [11] such as: mixing of discrete and continuous variables, incorporation of global sensitivity information, and sequential sampling.

过去的试验设计方法将采样和建模进行紧耦合。因子设计假设是一个线性模型，在设计空间的正交角点进行采样（见图1）。对于给复杂的非线性模型，通用目的的空间填充设计，比如Latin hyper-cubes，给出了设计空间的更加均匀的覆盖。对于k维中的N个采样点，有(N!)^{k−1}种可能的Latin hypercube设计，找到一个合适的设计，要平衡空间填充（如，通过熵，或势能）和其他想要的性质，如正交性。有很多关于Latin hypercube设计的文献，很多研究问题仍然是开放的，比如，离散和连续变量的混合，纳入全局敏感性信息，顺序采样。

Response Surface Methodology (RSM) [3], [12] is a sequential approach which has become the primary method for industrial experimentation. In its original form, response surfaces are second order polynomials which are determined using central composite factorial experiments, and a path of steepest ascent is used to seek an optimal point. For robust design, replication is used to estimate noise factors, and optimisation must consider dual responses for process mean and variance. Approaches for handling multiple objectives include “split-plot” techniques, “desirability functions” and Pareto fronts [13]. Non-parametric RSM can be more general than second-order polynomials, and uses techniques such as Gaussian processes, thin-plate splines, and neural networks. Alternative optimisation approaches include simulated annealing, branch-and-bound and genetic algorithms [14].

响应表面方法(RSM)是一种顺序方法，是工业试验的主要方法。其原始的形式，响应空间是二次多项式，使用central composite factorial试验来确定，最速下降路径用于寻找嘴有点。对于文件设计，用复制来估计噪声因素，优化要考虑对偶响应，以处理均值和方差。处理多目标的方法包括split-plot技术，desirability functions和Pareto fronts。非参数RSM比二次多项式更加一般化，使用高斯过程，薄板样条和神经网络这种技术。其他优化方法包括，模拟退火，分支界定，和基因算法。

In many areas, experiments are performed with detailed computer simulations of physical systems. Aerospace designers frequently work with expensive CFD (computational fluid dynamic) and FEA (finite element analysis) simulations. Multi-agent simulations are used to model how actor behaviour determines the outcome of group interactions in areas such as defence, networking, transportation, and logistics. Design and Analysis of Computer Experiments (or DACE, after [15]) differs from DOE in several ways. Simulations are generally deterministic, without random effects and uncontrolled variables, so less emphasis is placed on dealing with measurement noise. Simulations often include many variables, so there is more need to handle high dimensionality and mixed variable types. Where the response is complex, non-parametric models are used, including Gaussian Processes, Multivariate Adaptive Regression Splines, and Support Vector Regression [4], [16], [17].

在很多领域中，试验的进行，是对物理系统用详细的计算机模型进行的。航空设计者频繁的使用昂贵的计算流体力学(CFD)和有限元分析(FEA)仿真。多agent仿真用于建模actor的行为怎样确定group interaction的输出，应用领域包括defence，networking，交通和物流。计算机试验的设计与分析(DACE)与DOE有几个方面的不同。仿真通常是确定性的，没有随机效果和不可控的变量，所以在处理测量噪声上工作不多。仿真通常包括很多变量，所以更需要处理高维和混合变量类型。当响应很复杂时，就使用非参数模型，包括高斯过程，多变量自适应回归样条，支持矢量回归。

A problem with classical DOE and space-filling designs is that the sampling pattern is determined before measurements are made, and cannot adapt to features that appear during the experiment. In contrast, adaptive sampling [16], [18] is a sequential process that decides the location of the next sample by balancing two criteria. Firstly, it samples in areas that have not been previously explored (e.g. based on distance from previous samples). Secondly, it samples more densely in areas where interesting behaviour is observed, such as rapid change or non-linearity. This can be detected using local gradients, prediction variance (e.g. where uncertainty is modelled), by checking agreement between the model and data (cross-validation), or agreement between an ensemble of models. BO is a form of model-based global optimisation (MBGO [16]), which uses adaptive sampling to guide the experiment towards a global optimum. Unlike pure adaptive sampling, MBGO considers the optimum of the modelled objective when deciding where to sample.

传统DOE和空间填充设计的一个问题是，采样模式是在测量进行之前确定的，不能随着在试验中出现的特征自适应变化。对比起来，自适应采样是一个顺序过程，通过平衡两个准则来决定下一个采样的位置。首先，它会在之前没探索的区域中进行采样（即，基于之前采样的距离）。第二，在观察到有趣行为的区域，采样的更加密集，比如快速变化区域或非线性区域。这可以使用局部梯度，预测方差来检测到（如，对不确定性进行建模的地方），通过检测模型和数据的一致性得到（交叉检验），或模型集成之间的一致性。BO是基于模型的全局优化的一种形式(MBGO)，使用自适应采样来引导试验，朝向全局最优。与纯自适应采样不同的是，当决定哪里去采样时，MBGO考虑建模的目标的最优性。

Recently, there has been a surge in applying Bayesian optimisation to design problems involving physical products and processes. In [23], Bayesian optimisation is applied in combination with a density functional theory (DFT) based computational tool to design low thermal hysteresis NiTi-based shape memory alloys. Similarly, in [24] Bayesian optimisation is used to optimise both the alloy composition and the associated heat treatment schedule to improve the performance of Al-7xxx series alloys. In [25], Bayesian optimisation is applied for high-quality nano-fibre design meeting a required specification of fibre length and diameter within few tens of iterations, greatly accelerating the production process. It has also been applied in other diverse fields including optimisation of nano-structures for optimal phonon transport [26], optimisation for maximum power point tracking in photovoltaic power plants [27], optimisation for efficient determination of metal oxide grain boundary structures [28], and for optimisation of computer game design to maximise engagement [29]. It has also been used in a recent neuroscience study [30] in designing cognitive tasks that maximally segregate ventral and dorsal FPN activity.

最近，有很多工作应用贝叶斯优化来设计涉及到物理产品和过程的问题。在[23]中，贝叶斯优化与基于密度泛函理论(DFT)的计算工具一起应用，来设计基于低热磁滞NiTi的形状记忆合金。类似的，在[24]中，用贝叶斯优化来优化合金组成和相关的热处理调度，来改进Al-7xxx系列合金的性能。在[25]中，贝叶斯优化用于设计高质量纳米纤维，满足需要的纤维长度和半径指标，在几十次迭代之内完成，极大的加速了生产过程。还在很多其他领域中应用，包括优化最优光子传输的纳米结构，优化光伏发电厂中的最大能量点跟踪，优化高效确定金属氧化物晶界结构，优化计算游戏设计的最大参与。还在最近的神经科学研究中，用于设计认知任务，最大化的分离腹侧和背侧FPN行为。

The recent advances in both the theory and practice of Bayesian optimisation has led to a plethora of techniques. In most parts, each advance is applicable to a sub-set of experimental conditions. What is lacking is both an overview of these methods and a methodology to adapt these techniques to a particular experimental design context. We fill this gap and provide a comprehensive study of the state-of-the-art Bayesian optimisation algorithms in terms of their applicability in experimental optimisation. Further, we provide a template of how disparate algorithms can be connected to create a fit-for-purpose solution. This thus provides an overview of the capability and increases the reach of these powerful methods. We conclude by discussion where further research is needed.

贝叶斯优化最近的理论和实践的进展，带来了很多技术。在多数部分，每个进展对试验条件的一部分都是可应用的。缺少的是这些方法的概览，和对特定试验设计上下文调整这些技术的方法学。我们填补这些空白，给出了就其在试验优化上应用性目前最好的贝叶斯优化算法的综合研究。而且，我们还给出了模板，算法怎样结合以创建符合目标的解。这就给出了这些方法能力的概览，增加了对这些方法的可能应用。我们讨论了进一步需要那些研究，作为结论。

## 2. Bayesian Optimization

Bayesian optimisation incorporates two main ideas: 贝叶斯优化利用了以下两个主要思想：

- A Gaussian process (GP) is used to maintain a belief over the design space. This simultaneously models the predicted mean μt(x) and the epistemic uncertainty σt(x) at any point x in the input space, given a set of observations D_{1:t} = {(x1, y1), (x2, y2), ...(xt, yt)}, where xt is the process input, and yt is the corresponding output at time t.

高斯过程用于维护关于设计空间的信念。在给定观测集合D_{1:t} = {(x1, y1), (x2, y2), ...(xt, yt)}下，这同时对输入空间中任意点x处的预测均值μt(x)和认知不确定性σt(x)进行建模，其中xt是过程输入，yt是对应在t时刻的输出。

- An acquisition function expresses the most promising setting for the next experiment, based on the predicted mean μt(x) and the uncertainty σt(x).

采集函数表达对下一个试验最有希望的设置，基于预测均值μt(x)和不确定性σt(x)。

A GP is completely specified by its mean function m(x) and covariance function k(x, x'):

一个高斯过程完全由其均值函数m(x)和协方差函数k(x,x')指定：

$$f(x) ∼ GP(m(x), k(x, x'))$$(1)

The covariance function k(x, x') is also called the "kernel", and expresses the "smoothness" of the process. We expect that if two points x and x' are "close", then the corresponding process outputs y and y' will also be "close", and that the closeness depends on the distance between the points, and not the absolute location or direction of separation. A popular choice for the covariance function is the squared exponential (SE) function, also known as radial basis function (RBF):

协方差函数k(x,x')也称为核，表达了过程的平滑性。我们期望，如果两个点x和x'很接近，那么对应的过程输出y和y'也应当很接近，接近程度依赖于两个点之间的距离，而不是绝对位置或分离的方向。对协方差函数的一个流行的选择是指数平方函数，也称为径向基函数(RBF)：

$$k(x, x') = exp(-||x-x'||^2 /2θ^2)$$(2)

Equation 2 says that the correlation decreases with the square of the distance between points, and includes a parameter θ to define the length scale over which this happens. Specialised kernel functions are sometimes used to express pre-existing knowledge about the function (e.g. if something is known about the shape of f).

式2表明，关联程度随着两点之间的距离的平方增加而下降，包含一个参数θ来定义一个长度尺度。专用的核函数有时候也用于表达函数预先存在的知识（如，关于f的形状已知一部分内容）。

In an experimental setting, observations include a term for normally distributed noise ε ∼ N (0, σ_{noise}^2), and the observation model is:

在一个试验设置中，观察中包含正态分布的噪声ε ∼ N (0, σ_{noise}^2)，观测模型为：

$$y = f(x) + ε$$

Gaussian process regression (or "kriging") can predict the value of the objective function f(·) at time t + 1 for any location x. The result is a normal distribution with mean μt(x) and uncertainty σt(x).

高斯过程回归（或kriging）可以预测目标函数f(·)在时刻t+1在任意位置x处的值。结果是一个正态分布，均值μt(x)，方差σt(x)。

$$P(f_{t+1} | D_{1:t}, x) = N(μ_t(x), σ_t^2(x))$$(3)

where 其中

$$μ_t(x) = k^T [K + σ_{noise}^2 I]^{−1} y_{1:t}, σ_t(x) = k(x, x) − k^T [K + σ_{noise}^2 I]^{−1} k$$(4)

$$k = [k(x, x1), k(x, x2), ..., k(x, xt)], K = \left[ \begin{matrix} k(x1,x1) & ... & k(x1,xt) \\ ... & ... & ... \\ k(xt,x1) & ... & k(xt,xt) \end{matrix} \right]$$(5)

Using the Gaussian process model, an acquisition function is constructed to represent the most promising setting for the next experiment. Acquisition functions are mainly derived from the μ(x) and σ(x) of the GP model, and are hence cheap to compute. The acquisition function allows a balance between exploitation (sampling where the objective mean μ(·) is high) and exploration (sampling where the uncertainty σ(·) is high), and its global maximiser is used as the next experimental setting.

使用高斯过程模型，构建一个采集函数来表示下一个试验最有希望的设置。采集函数主要是从高斯过程模型的μ(x)和σ(x)推导得到，因此计算量很小。采集函数允许在挖掘（在目标均值μ(·)很高的地方采样）和探索（在不确定性σ(·)很高的地方采样）之间进行平衡，其全局最大化器用作下一个试验设置。

Acquisition functions are designed to be large near potentially high values of the objective function. Figure 3 shows commonly used acquisition functions: PI, EI, and GP-UCB. PI prefers areas where improvement over the current maximum f(x+) is most likely. EI considers not only probability of improvement, but also the expected magnitude of improvement. GP-UCB maximises f(·) while minimising regret, the difference between the average utility and the ideal utility. Regret bounds are important for theoretically proving convergence. Unlike the original function, the acquisition function can be cheaply sampled, and may be optimised using a derivative-free global optimisation method like DIRECT [34] or using multi-start method with a derivative based local optimiser such as L-BFGS [35]. Details can be found in [19], [21].

采集函数被设计在目标函数值很可能很大的地方很大。图3展示了常用的采集函数：PI，EI和GP-UCB。PI倾向于在当前最大值f(x+)最有可能改进的区域。EI不仅考虑改进的概率，还考虑改进的期望幅度。GP-UCB最大化f(·)，同时最小化regret，这是平均utility和理想utility之间的差。Regret界对理论证明收敛是很重要的。与原始函数不一样的是，采集函数的采样要很廉价，可能用一个无梯度全局优化方法如DIRECT进行优化，或使用多起始方法，带有基于梯度的局部优化器如L-BFGS进行优化。详见[19,21]。

## 3. Experimental Design with Bayesian Optimization

BO has been influential in computer science for hyperparameter tuning [38]–[42], combinatorial optimisation [43], [44], and reinforcement learning [21]. Recent years have seen new applications in areas such as robotics [45], [46], neuroscience [47], [48], and materials discovery [49]–[55].

BO在计算机科学中的下面领域很有影响力，超参数调节，组合优化，强化学习。近几年在机器人学，神经科学，和材料发现领域看到了新应用。

Bayesian optimisation is an iterative process outlined in Figure 2, which can be applied to experiments where inputs are unconstrained and the objective is a scalarised function of measured outputs. Examples of this kind include material design using physical models [56], or laboratory experiments [25]. However, experiments often involve complicating factors such as constraints, batches, and multiple objectives. For example, in the alloy design process the composition of each sample follows a set of mixture constraints (see Figure 4). Batches of samples then undergo heat treatment for up to 70 hours, exposed to the same temperatures but with possible variation in duration between samples [24]. The optimiser must produce a batch of experimental settings, obeying inequality constraints, with some factors varying and others fixed within each batch. This impacts the design of the optimiser, through the formulation of the model, acquisition functions, and the search strategy. These are active areas of research, and recent developments are surveyed in the following discussion.

BO是一个迭代过程，如图2所示，可以应用到输入被约束，输出是测量的输出的标量化函数的试验中。这种例子包括，使用物理模型的材料设计，或实验室试验。但是，试验通常涉及到复杂的因素，如约束，批次，和多目标。比如，在合金设计过程中，每个样本的组成满足混合物约束的集合（见图4）。批次样本进行热处理，最多可达70小时，样本在相同的温度下进行，但是可能会有持续时间的不同。优化器要产生试验设置的批次，满足不等式的约束，在每个批次中一些因素变化，其他的固定。这会通过模型、采集函数和搜索策略的表述，影响优化器的设计。这些都是活跃的研究领域，在下面的讨论中回顾了最近的进展。

### 3.1 Incorporating Prior Knowledge

Where successive experiments are sufficiently similar to previous ones, it may be desirable to transfer knowledge from previous outcomes. Prior knowledge about the function or data can be used to reduce the search complexity and accelerate optimisation. Table 1 outlines some approaches. (1) Knowledge may be transferred from past (source) experiments to new (target) experiments where there are known or learnable similarities between the domains. For example, the source and target may be loosely similar, or have similar trends. (2) Where something is known about the influence of particular variables on the objective function, this can be imposed on the GP model. This could include monotonicity, function shape, or the probable location of the optimum or other features. (3) Where dependency structures exist in the design space, these can exploited to constrain the GP, or to handle high dimensionality via embedding.

连续的试验与之前的试验是足够相似的，将之前的输出中的知识进行传递，就非常理想。关于功能或数据之前的知识可以用于降低搜索复杂度和加速优化。表1列出了一些方法。(1)在两个领域之间有已知的或可学习的相似性时，知识可能从过去的（源）试验传递到新的（目的）试验。比如，源和目的可能松散的类似，或有相似的趋势。(2)已知特定变量对目标函数有一些影响时，这可以加入到GP模型中。这可能包括单调性，函数形状，或最优质的可能位置，或其他特征。(3)在设计空间中存在依赖性结构时，这可以被用作约束GP，或通过嵌入处理高维度问题。

### 3.2 High Dimensional Optimization

The acquisition function must be optimised to find the next best suggestion for evaluating the objective. In continuous domain the acquisition functions can be extremely sharp in high dimensions, having only a few peaks marooned in a large terrain of almost flat surface. Global optimisation algorithms such as DIRECT [34] are infeasible above about 10 dimensions, and gradient-dependent methods cannot move if initialised in the flat terrain.

采集函数要被优化，以找到下一个最优的建议，以评估目标函数。在连续域中，采集函数在高维情况下，可能非常尖锐，在一大片几乎平坦的平面上有几个非常峰值。全局优化算法如DIRECT在高于10维时就不能处理了，依赖于梯度的方法如果在平坦区域初始化，就无法移动了。

General strategies for tackling high-dimensionality include [103]: reducing the design space, screening important variables, decomposing the design into simpler sub-problems, mapping into a lower-dimensional space, and visualisation. Table 1(4) outlines approaches that have been reported for high dimensional BO, including: using coarse-to-fine approximations, projection into a lower-dimensional space, and approximation through low-rank matrices or additive structures. Choice of a method depends on whether the objective function has an intrinsic low dimensional structure (4B) or not (4A).

处理高维的一般策略包括：缩减设计空间，筛选重要变量，将设计分解成简单的子问题，映射到更低维的空间，和可视化。表1(4)列出了高维BO的方法，包括：使用粗糙到精细的近似，投影到更低维的空间，通过低秩矩阵或加性结构近似。一种方法的选择，依赖于目标函数是否有内蕴的低维结构(4A, 4B)。

Standard BO is known to perform well in low dimensions, but performance degrades above about 15-20 dimensions. High dimensional BO has been demonstrated for 25-34 intrinsic dimensions on "real world" data, and up to 50 dimensions for synthetic functions [73], [77]. Projection methods have been shown to work independently of the number of extrinsic dimensions [43], [79], [81], whereas special kernels are shown to work in hundreds of dimensions [75].

标准BO在低维时表现良好，但在高于15-20维时，性能就逐渐下降。高维BO在真实世界数据上对25-34内蕴维上展示过结果，对合成的函数高于50维的也展示过。投影方法可以有效果，与外在维度无关，而特殊的核可以在上百维中有作用。

### 3.3 Multi-objective Optimization

Design problems often include multiple objectives which can be challenging to optimise. For example [104] demonstrates multiple objectives for discovery of new materials. Scalarisation by weighted sum of objectives can be done, but may not work when objectives have strong conflicts. In that setting a Pareto set of optimal points can be found [105]. For a point in a Pareto set, any one of the objectives cannot be improved without penalising another objective.

设计问题通常包括多个目标，优化起来非常有挑战性。比如，[104]展示了发现新材料的多目标情况。目标函数加权和的标量化方法可以进行，但当目标有很强的冲突时就不能工作了。在这个设置中，可以发现最优点的Pareto set。对于Pareto set中的一个点，任一目标的改进，必须要牺牲另一个目标。

Many methods have been proposed for using Bayesian optimisation for multi-objective optimisation [106]–[109], but these suffer from computational limitations because the acquisition function generally requires computation for all objective functions and as the number of objective functions grow the computational cost grows exponentially.

提出了很多方法对多目标优化使用贝叶斯优化，但存在计算限制问题，因为采集函数一般需要计算所有目标函数，随着目标函数数量的增加，计算代价以指数速度增加。

Moving away from EI, the method of [109] allows the optimisation of multiple objectives without rank modelling for conflicting objectives, while also remaining scale-invariant toward different objectives. The method performs better than [107], but suffers in high dimensions and can be computationally expensive. Predictive entropy search is used by [110], allowing the different objectives to be decoupled, computing acquisition for subsets of objectives when required. The computational cost increases linearly with the number of objectives. The method of [111] can be used for single- or multiple-objective optimisation, including in multiple inequality constraints and has been shown to be robust in highly constrained settings where the feasible design space is small.

除了EI以外，[109]的方法允许优化多个目标，而不需要对有冲突的目标进行rank建模，同时对不同的目标保持尺度不变。该方法比[107]的效果要好，但在高维时有问题，计算代价也很大。[110]使用了预测性熵搜索，使不同的目标可以解耦，需要时可以计算目标子集的采集函数。其计算代价随着目标的数量线性增长。[111]的方法可以用于单目标或多目标优化，包括多个不等式约束，已经证明在高度约束的设置中也很稳健，其中可行的设计空间很小。

### 3.4 Constraints

Table 1(5) outlines some approaches to handling constraints. If constraints are known, they can be handled during optimisation of the acquisition function by limiting the search. More difficult are "black box" constraints that can be evaluated but have unknown form. If the constraint is cheap to evaluate, this is not a problem. Methods for expensive constraint functions include a weighted EI function [83], [84], and weighted predictive entropy search [86]. A lookahead strategy for unknown constraints is described by [88]. A different formulation for the unknown is proposed by [85], handling expensive constraints using ADMM solver of [112].

表1(5)列出了处理约束的一些方法。如果约束已知，可以通过限制搜索，来在采集函数的优化中进行处理。更困难的是黑箱约束，可以被评估，但形式未知。如果约束评估起来很容易，这并不是个问题。对于昂贵约束函数的方法包括，加权EI函数，加权预测性熵搜索。[88]是对未知约束的有预见力的策略。[85]提出了对未知的不同表述，使用[112]的ADMM求解器处理昂贵的约束。

The above methods deal with inequality constraints. In [89] both inequality and equality constraints are handled, using slack variables to convert inequality constraints to equality constraints, and Augmented Lagrangian (AL) to convert these inequality constraints into a sequence of simpler sub-problems.

上面的方法处理的是不等式约束。在[89]中，处理了不等式和等式约束，使用松弛变量来将不等式约束转化成等式约束，使用Augmented Lagrangian(AL)将这些不等式约束转化到一系列更简单的子问题中。

The concept of weighted predictive entropy search has been extended for multi-objective problems [87] for inequality constraints which are both unknown and expensive to evaluate. A different type of constraint specifically for multiple objectives is investigated by [90] where between all the objectives, there exists a rank order preference on which objective is important. The algorithm developed therein can preferentially sample the Pareto set such that Pareto samples are more varied for the more important objectives.

加权预测熵搜索的概念被拓展到了未知且评估昂贵的不等式约束的多目标问题[87]。[90]研究了多目标的不同类型的约束，在所有目标之间，都有rank顺序倾向，表示哪个目标更重要。那里开发的算法可以采样Pareto集，这样Pareto样本对更重要的目标来说变化更多。

### 3.5 Parallel (Batch) Optmization

In some experiments it can be efficient to evaluate several settings in parallel. For example, during alloy design batches of different mixtures undergo similar heat treatment phases, so the optimiser must recommend multiple settings before receiving any new results. Sequential algorithms can be used to find the point that maximises the acquisition function, and then move on to find the next point in the batch after suppressing this point. Suppression can be achieved by temporarily updating the GP with a hypothetical value for the point (e.g. based on a recent posterior mean), or by applying a penalty in the acquisition function. Table 1(6) outlines some approaches that have been reported. Most methods are for unconstrained batches, though recent work has handled constraints on selected variables within a batch [102].

在一些试验中，并行评估几个设置是非常高效的。比如，在不同混合物的合金设计批次中，会经历类似的热处理阶段，所以优化器要推荐多个设置，然后才能得到新的结果。顺序算法可以用于找到最大化采集函数的点，然后在抑制这个点之后继续寻找批次中的下一个点。抑制可以通过临时用这个点的假设值来更新GP（如，基于最近的后验均值），或在采集函数中应用一个惩罚。。表1(6)列出了一些给出的方法。多数方法都是无约束批次，最近的工作在一个批次中选定的变量中处理了约束。

### 3.6 Multi-fidelity Optimization

When function evaluations are prohibitively expensive, cheap approximations may be useful. In such situations high fidelity data obtained through experimentation might be augmented by low fidelity data obtained through running a simulation. For example, during alloy design, simulation software can predict the alloy strength but results may be less accurate than measurements obtained from casting experiments. Multi-fidelity Bayesian optimisation has been demonstrated in [113], [114]. Recently, [115] proposed BO for an optimisation problem with multi-fidelity data. Although multi-fidelity approach has been applied in problem-specific context or non-optimisation related tasks [41], [116]–[120], the method of [115] generalises well for BO problems.

当函数评估太过于昂贵时，廉价的近似就很有用。在这种情况下，通过试验得到的高保真度数据，可以通过用仿真得到的低保真度数据进行扩充。比如，在合金设计的过程中，仿真软件可以预测合金的强度，但结果的准确度要比铸造试验得到的测量结果低。多保真度贝叶斯优化在[113,114]中得到了展示。最近，[115]提出了用多保真度数据用BO进行优化。虽然多保真度方法在一些特定问题或非优化相关的任务进行了应用，但[115]的方法对BO问题泛化的很好。

### 3.7 Mixed-type Input

Experimental parameters are often combinations of different types: continuous, discrete, categorical, and binary. Incorporation of mixed type input is challenging across the domains, including simpler methods such as Latin hypercube sampling [11]. Non-continuous variables are problematic in BO because the objective function approximation with GP assumes continuous input space, with covariance functions defining the relationship between these continuous variables. One common way to deal with discrete variables is to round the value to a close integer [40], but this approach leads to sub-optimal optimisation [121].

试验参数通常是不同类型数据的组合：连续的，离散的，类别的，和二值的。混合类型输入在不同领域中是很有挑战的，包括更简单的方法，比如Latin hypercube采样。非连续变量在BO中是有问题的，因为用GP的目标函数近似假设连续的输入空间，协方差函数定义了这些连续变量之间的关系。处理离散变量的一种常见方法是将值四舍五入到接近的整数，但这种方法带来次优的优化。

Two options for handling mixed-type inputs are: (1) designing kernels that are suitable for different variables, and (2) subsampling of data for maximising the objective function, which is especially useful in higher dimensional space. For integer variables the problem can be solved through kernel transformation, by assuming the objective function to be flat for the region where two continuous variables would be rounded to the same integer [121]. In [67] categorical variables are included by one-hot-encoding alongside numerical variables. A specialised kernel for categorical variables is proposed in [122].

处理混合类型输入的方法有两种选择：(1)设计适用于不同变量的核，(2)对数据进行下采样，以最大化目标函数，这在高维空间中尤其有用。对整数变量，可以通过核变换解决问题，假设目标函数在两个连续变量四舍五入到相同整数的区域是平坦的。在[67]中通过独热码包含了类别变量。[122]中对类别变量提出了专门的核。

Random forest regression is a good alternative to GP for regression in a sequential model-based algorithm configuration (SMAC, [44]). Random forests are good at exploitation but don’t perform well for exploration as they may not predict well at points that are distant from observations. Additionally, a non-differentiable response surface renders it unsuitable for gradient-based optimisation.

对于基于顺序模型的算法配置，随机森林回归是GP的一个很好的替代。随机森林擅长挖掘，但对于探索并不擅长，因为对于距离观测比较远的点，预测并不太好。另外，不可微分的响应表面使其不适用于基于梯度的优化。

## 4. Discussion

Machine-learning methods through Bayesian optimisation offer a powerful way to deal with many problems of experimental optimization that have not been previously addressed. While techniques exist for different issues (high dimensionality, multi-objective, etc.), few works solve multiple issues in a general way. Methods are likely to be composable where no incompatible changes are required to the BO process.Figure 5 outlines composability based on the current repertoire of Bayesian optimisation algorithms. When a design problem is single objective, has single fidelity measurement, and all the variables are continuous then it offers the greatest flexibility in terms of adding specific capability such as transfer learning or high dimensional optimisation. Other cases require careful selection of algorithms to add desired capabilities. For example, the method of [111] handles multiple objectives with constraints, and the method of [43] handles parallel evaluation in high dimensions with mixed type inputs. Some combinations may not even be possible, for example, Random Forest based algorithm such as [44] would not admit many capabilities. Note that this graph does not portray any theoretical limitations, but merely presents a gist of the current capability through the lens of composability.

通过贝叶斯优化的机器学习方法，可以有效的处理很多试验优化的问题，之前并未得到很好的解决。对不同的问题都有解决方法（高维，多目标，等），很少有工作可以以通用的方式解决多个问题。在BO过程需要的变化没有不兼容的情况下，这些方法很可能是模块化的。图5列出了基于目前贝叶斯优化算法的全部技能的可组合性。当设计问题是单目标，有单保真度的测量结果，所有变量都是连续的，那么就有最大的灵活性，可以加入特定的能力，比如迁移学习，或高维优化的能力。其他的情况需要仔细选择算法来加入期望的能力。比如，[111]的方法处理带约束的多目标优化，[43]的方法处理的是混合输入的高维问题的并行评估。一些组合可能是不可能的，比如，基于随机森林的算法，如[44]，不会允许更多的能力。注意，这个图没有描述任何理论限制，但只是给出了当前能力的要点。

Several open-source libraries are available for incorporating BO into computer programs. Depending on the application, computation speed may be an issue. A common operation in most algorithms is Cholesky decomposition which is used to invert the kernel matrix and is generally O(n^3) for n data points, but with care this can be calculated incrementally as new points arrive, reducing the complexity to O(n^2) [123]. Several algorithms gain speed-up by implementing part of the algorithm on a GPU, which can be up to 100 times faster than the equivalent single-threaded code [124].

一些开源库可以将BO纳入到计算机程序中。依赖于应用，计算速度可能是一个问题。多数算法中常见的运算是Cholesky分解，用于求核矩阵的逆，对于n个数据点，复杂度是O(n^3)，但可以在新数据点到来的时候，可以逐渐计算，将计算复杂度降低到O(n^2)。一些算法可以通过将算法在GPU上实现来得到加速，这会比单线程代码快100倍。

- GPyOpt (https://github.com/SheffieldML/GPyOpt) is a Bayesian optimisation framework, written in Python and supporting parallel optimisation, mixed factor types (continuous, discrete, and categorical), and inequality constraints.

GPyOpt是一个贝叶斯优化框架，用Python写，支持并行优化，混合因子类型（连续，离散和类别），和不等式约束。

- GPflowOpt (https://github.com/GPflow/GPflowOpt) is written in Python and uses TensorFlow (https://www.tensorflow.org) to accelerate computation on GPU hardware. It supports multi-objective acquisition functions, and black-box constraints [125].

GPflowOpt是用Python写的，使用TensorFlow在GPU硬件上来加速计算。支持多目标采集函数，和黑箱约束。

- DiceOptim (https://cran.r-project.org/web/packages/DiceOptim/index.html) is a BO package written in R. Mixed equality and inequality constraints are implemented using the method of [89], and parallel optimisation is via multipoint EI [91], however parallel and constraints cannot be mixed in a single optimisation.

DiceOptim是一个用R写的BO包。使用[89]的方法实现了等式和不等式混合约束，通过多点EI[91]来实现并行优化，但是并行和约束不能在单个优化中混合实现。

- MOE (https://github.com/Yelp/MOE) supports parallel optimisation via multi-point stochastic gradient ascent [124]. Interfaces are provided for Python and C++, and optimisation can be accelerated on GPU hardware.

MOE通过多点随机梯度下降支持并行优化。接口是用Python和C++给出的，优化可以在GPU硬件上加速。

- SigOpt (http://sigopt.com) offers Bayesian optimisation as a web service. The implementation is based on MOE, but includes some enhancements such as mixed factor types (continuous, discrete, categorical), and automatic hyperparameter tuning.

SigOpt将贝叶斯优化提供为网络服务。实现是基于MOE的，但加入了一些强化，比如混合因子类型（连续，离散，类别），和自动超参数调节。

- BayesOpt (https://github.com/rmcantin/bayesopt) is written in C++, and includes common interfaces for C, C++, Python, Matlab, and Octave [123].

BayesOpt用C++编写，有C，C++，Python，MATLAB和Octave的接口。

## 5. Conclusion

This review has presented an overview of Bayesian optimisation (BO) with application to experimental design. BO was introduced in relation to existing Design of Experiments (DOE) methods such as factorial designs, response surface methodology, and adaptive sampling. A brief discussion of the theory highlighted the roles of the Gaussian process, kernel, and acquisition function. A set of seven core issues was identified as being important in practical experimental designs, and some detailed solutions were reviewed. These core issues are: (1) the incorporation of prior knowledge, (2) high dimensional optimisation, (3) constraints, (4) batch evaluation, (5) multiple objectives, (6) multi-fidelity data, and (7) mixed variable types.

本文给出了贝叶斯优化在试验设计中的应用的综述。BO的介绍是与现有的DOE方法相关的，如分解设计，响应表面方法学，和自适应采样。理论的简单讨论，强调了高斯过程，核和采集函数的角色。7个核心问题对实践中的试验设计非常重要，回顾了一些细节解决方法。这些核心问题是：(1)纳入先验知识，(2)高维优化，(3)约束，(4)批次评估，(5)多目标，(6)多保真度数据，(7)混合变量类型。

Recent works have shown the potential of Bayesian optimisation in fields such as robotics, neuroscience, and materials discovery. As the range of potential applications expands, it is increasingly unlikely that "vanilla" optimisation approaches for small numbers of unconstrained, continuous variables will be appropriate. This is particularly true in DACE simulation applications where high dimensional mixed-type inputs are typical.

最近的工作证明了贝叶斯优化在一些领域的潜力，如机器人学，神经科学，材料发现。随着潜在应用范围的扩展，传统的优化方法，无约束的，数量较少的连续变量就不太合适了。在DACE仿真应用中，一般都是高维的混合类型的输入，都是这种情况。

Bayesian optimisation offers a powerful and rigorous framework for exploring and optimising expensive "black box" functions. While solutions exist for the core issues in experimental design, each approach has strengths and weaknesses that could potentially be improved, and the combination of the individual solutions is not necessarily straightforward. Thus there is a need for ongoing work in this area to: (1) improve the efficiency, generality, and scalability of approaches to the core issues, (2) develop designs that allow easy combination of multiple approaches, and (3) develop theoretical guarantees on the performance of solutions.

贝叶斯优化对探索和优化昂贵的黑箱函数提供了强有力的严格的框架。对试验设计的核心问题，解是存在的，但每个方法都有优势和可以改进的劣势，单个解的组合并不是直接的。因此，这个领域正在进行的工作有必要去：(1)改进核心问题方法的效率，泛化性和扩展性，(2)提出一些设计，使多个方法容易组合，(3)对解的性能提出理论保证。