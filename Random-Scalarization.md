# A Flexible Framework for Multi-Objective Bayesian Optimization using Random Scalarizations

Biswajit Paria et. al @ CMU & UC Berkeley

## 0. Abstract

Many real world applications can be framed as multi-objective optimization problems, where we wish to simultaneously optimize for multiple criteria. Bayesian optimization techniques for the multi-objective setting are pertinent when the evaluation of the functions in question are expensive. Traditional methods for multi-objective optimization, both Bayesian and otherwise, are aimed at recovering the Pareto front of these objectives. However, in certain cases a practitioner might desire to identify Pareto optimal points only in a subset of the Pareto front due to external considerations. In this work, we propose a strategy based on random scalarizations of the objectives that addresses this problem. Our approach is able to flexibly sample from desired regions of the Pareto front and, computationally, is considerably cheaper than most approaches for MOO. We also study a notion of regret in the multi-objective setting and show that our strategy achieves sublinear regret. We experiment with both synthetic and real-life problems, and demonstrate superior performance of our proposed algorithm in terms of flexibility, scalability and regret.

很多现实世界应用可以表述为多目标优化问题，需要同时对多个准则进行优化。多目标设置下的贝叶斯优化技术，在问题中函数评估很昂贵时就非常合适。多目标优化的传统方法，包括贝叶斯方法和其他方法，其目标是恢复这些目标的Pareto front。但是，在特定情况下，因为外部考虑，可能想要找到Pareto front的一个子集中的Pareto最优解。在本文中，我们提出一种基于目标函数随机标量化的策略，来解决这个问题。我们的方法可以从Pareto front中期望区域进行灵活的采样，计算上比多数MOO方法要少很多。我们研究了多目标设置下regret的概念，表明我们的策略可以得到亚线性的regret。我们用合成问题和真实生活中的问题进行了试验，证明了我们提出的算法在灵活性，可扩展性和regret上都有非常好的性能。

# 1. Introduction

Bayesian optimization (BO) is a popular recipe for optimizing expensive black-box functions where the goal is to find a global maximizer of the function. Bayesian optimization has been used for a variety of practical optimization tasks such as hyperparameter tuning for machine learning algorithms, experiment design, online advertising, and scientific discovery (Snoek et al., 2012; Hernandez-Lobato et al., 2017; Martinez-Cantin et al., 2007; Gonzalez et al., 2015; Kandasamy et al., 2017).

贝叶斯优化是优化昂贵的黑箱函数的流行方法，其目标是找到函数的全局最大化值。贝叶斯优化曾用于很多实际优化任务，比如机器学习算法的超参数调节，试验设计，在线广告，和科学发现。

In many practical applications however, we are required to optimize multiple objectives, and moreover, these objectives tend to be competing in nature. For instance, consider drug discovery, where each evaluation of the functions is an in-vitro experiment and as the output of the experiment, we measure the solubility, toxicity and potency of a candidate example. A chemist wishes to find a molecule that has high solubility and potency, but low toxicity. This is an archetypal example for Bayesian optimization as the lab experiment is expensive. Further, drugs that are very potent are also likely to be toxic, so these two objectives are typically competing. Other problems include creating fast but accurate neural networks. While smaller neural networks are faster to evaluate, they suffer in terms of accuracy.

但是，在很多实际的应用中，我们需要优化多个目标，而且，这些目标一般在本质上就是互相竞争的。比如，在药物发现中，函数的每次评估都是试管内试验，作为试验的输出，我们测量的是候选样本的溶解性，毒性和效力。一个药剂师想要找到的分子要具有高可溶性和效力，但是毒性要低。这是贝叶斯优化应用的一个典型例子，因为实验室试验是非常昂贵的。而且，效力很强的药物，其毒性通常也会很强，所以这两个目标一般是竞争性的。其他问题包括，创建快速但准确的神经网络。小型神经网络评估起来很快，但是在准确率上也通常有问题。

Due to their conflicting nature, all the objectives cannot be optimized simultaneously. As a result, most multiobjective optimization (MOO) approaches aim to recover the Pareto front, defined as the set of Pareto optimal points. A point is Pareto optimal if it cannot be improved in any of the objectives without degrading some other objective. More formally, given K objectives f(x) = (f1(x), . . . , fK(x)) : X → R^K over a compact domain X ⊂ R^d, a point x1 ∈ X is Pareto dominated by another point x2 ∈ X iff fk(x1) ≤ fk(x2) ∀k ∈ [K] and ∃k ∈ [K] s.t. fk(x1) < fk(x2), where we use the notation [K] throughout the paper to denote the set {1, ..., K}. We denote this by f(x1) ≺ f(x2). A point is Pareto optimal if it is not Pareto dominated by any other point. We use X^* _f to denote the Pareto front for a multi-objective function f, and f(X^ *_f) to denote the set of Pareto optimal values, where f(X) = {f(x) | x ∈ X} for any X ⊆ X. The traditional goal in the MOO optimization regime is to approximate the set of Pareto optimal points (Hernandez-Lobato et al., 2016; Knowles, 2006; Ponweiser et al., 2008; Zuluaga et al., 2013).

由于其冲突性的本质，不能同时对所有目标进行最小化。结果是，多数多目标优化(MOO)方法的目标都是得到Pareto front，也就是Pareto最优点的集合。如果一个点在任何目标函数上的改进，都需要降低其他某个目标函数的质量，那么就称这个点为Pareto最优的。更正式的，给定K个目标f(x) = (f1(x), . . . , fK(x)) : X → R^K，定义域为紧凑域X ⊂ R^d，一个点x1 ∈ X被另一个点x2 ∈ X Pareto支配的意思是，∀k ∈ [K]有fk(x1) ≤ fk(x2)，而且∃k ∈ [K]，s.t. fk(x1) < fk(x2)，我们在本文中使用[K]表示集合{1, ..., K}。我们将Pareto支配关系表示为f(x1) ≺ f(x2)。一个点是Pareto最优的，意思是没有被任何其他点Pareto支配。我们使用X^* _f表示一个多目标函数f的Pareto front，使用f(X^ *_f)表示Pareto最优值的集合，其中f(X) = {f(x) | x ∈ X} for any X ⊆ X。MOO优化领域中的传统目标是，近似Pareto最优点的集合。

However, in certain scenarios, it is preferable to explore only a part of the Pareto front. For example, consider the drug discovery application described above. A method which aims to find the Pareto front, might also invest its budget to discover drugs that are potent, but too toxic to administer to a human. Such scenarios arise commonly in many practical applications. Therefore, we need flexible methods for MOO that can steer the sampling strategy towards regions of the Pareto front that a domain expert may be interested in. Towards this end, we propose a Bayesian approach based on random-scalarizations in which the practitioner encodes their preferences as a prior on a set of scalarization functions.

但是，在特定的场景中，可能只需要探索Pareto front的一部分。比如，考虑上面所述的药物发现应用。一种方法的目标是找到Pareto front，很可能会找到效用很强，但是毒性也很强的药物。这种场景在很多实际应用中都会出现。因此，对MOO我们需要灵活的方法，可以将采样策略定向到领域专家感兴趣的区域中的Pareto front。为此，我们提出了一种基于随机标量化的贝叶斯方法，可以将这种倾向编码为标量化函数集合，作为先验。

A common approach to multi-objective optimization is to use scalarization functions s_λ(y) : R^K → R (Roijers et al., 2013), parameterized by λ belonging to a set Λ, and y ∈ R^K denoting K-dimensional objective values. Scalarizations are often used to convert multi-objective values to scalars, and standard Bayesian optimization methods for scalar functions are applied. Since our goal is to sample points from the Pareto front, we need additional assumptions to ensure that the utility functions are maximized for y ∈ f(X^* _f). Following Roijers et al. (2013) and Zintgraf et al. (2015) we assume that s_λ(y) are monotonically increasing in all coordinates. Optimizing for a single fixed scalarization amounts to the following maximization problem, which returns a single optimal point lying on the Pareto front.

多目标优化的一种常见方法是使用标量化函数s_λ(y) : R^K → R，参数λ属于集合Λ，y ∈ R^K表示K维目标值。标量化也经常用于将多目标值转换成标量，然后应用标量函数的标准贝叶斯优化方法。由于我们的目标是要从Pareto front上采样一些点，我们需要额外的假设，来确保效用函数对y ∈ f(X^* _f)是最大化的。我们假设s_λ(y)对所有坐标都是单调递增的。优化单个固定的标量会得到下面的最大化问题，会得到在Pareto front上的单个最优点。

$$x^*_ λ = argmax_{x∈X} s_λ(f(x))$$(1)

One can verify that x^* _λ ∈ Pf follows from the monotonicity of the scalarization. In this work, we are interested in a set of points X ^* = {x^* _i}^T _{i=1} of size at most T, spanning a specified region of the Pareto front rather than a single point. To achieve this we take a Bayesian approach and assume a prior p(λ) with support on Λ, which intuitively translates to a prior on the set of scalarizations S_Λ = {s_λ | λ ∈ Λ}. Thus, in place of optimizing a single scalarization, we aim to optimize over a set of scalarizations weighted by the prior p(λ). Each λ ∈ Λ maps to a pareto optimal value f(x^* _λ) ∈ f(X^* _f). Thus, the prior p(λ) defines a probability distribution over the set of Pareto optimal values, and hence encodes user preference, which is depicted in Figure 1.

可以验证，x^* _λ ∈ Pf也符合标量化的单调性。本文中，我们感兴趣的是点X ^* = {x^* _i}^T _{i=1}的集合，集合大小最多为T，支撑了特定区域的Pareto front，而不是单个点。为获得这个目标，我们采用贝叶斯方法，假设在Λ上的先验p(λ)，很直观的得到在标量化集合S_Λ = {s_λ | λ ∈ Λ}上的先验。因此，我们并不是优化单个标量化，而是优化由先验p(λ)加权的标量化集合。每个λ ∈ Λ映射到一个Pareto最优值f(x^* _λ) ∈ f(X^* _f)。因此，先验p(λ)在Pareto最优值集合上定义了一个概率分布，因此包含了用户的偏好，如图1所示。

In this paper, we propose to minimize a Bayes regret which incorporates user preference through the prior and scalarization specified by the user. We propose multiobjective extensions of classical BO algorithms: upper confidence bound (UCB) (Auer, 2002), and Thompson sampling (TS) (Thompson, 1933) to minimize our proposed regret. At each step the algorithm computes the next point to evaluate by randomly sampling a scalarization s_λ using the prior p(λ), and optimizes it to get x^* _λ. Our algorithm is fully amenable to changing priors in an interactive setting, and hence can also be used with other interactive strategies in the literature. The complete algorithm is presented in Algorithm 1 and discussed in detail in Section 3. While random scalarizations have been previously explored by Knowles (2006) and Zhang et al. (2010), our approach is different in terms of the underlying algorithm. Furthermore, we study a more general class of scalarizations and also prove regret bounds. As we shall see, this formulation fortunately gives rise to an extremely flexible framework that is much simpler than the existing work for MOO and computationally less expensive. Our contributions can summarized as follows:

本文中，我们提出对一个贝叶斯regret进行最小化，通过用户指定的先验和标量化，包含了用户的偏好。我们提出了经典BO算法的多目标拓展：UCB，和Thompson采样(TS)，以最下化我们提出的regret。在算法的每一步中，计算下一个要评估的点，通过使用先验p(λ)来随机采样一个标量化s_λ，对其进行优化以得到x^* _λ。我们的算法可以完全适应以互动方式变化的先验，因此也可以用于文献中其他的互动策略。完整的算法如算法1，在第3部分中进行详细讨论。随机标量化其他作者进行过探索，但我们的方法是不一样的。而且，我们研究了更一般类别的标量化，也证明了regret界限。我们将会看到，这种表述很幸运的得到非常灵活的框架，比现有的MOO工作要更简单，计算量也要小很多。我们的贡献总结如下：

1. We propose a flexible framework for MOO using the notion of random scalarizations. Our algorithm is flexible enough to sample from the entire Pareto front or an arbitrary region specified by the user. It is also naturally capable of sampling from non-convex regions of the Pareto front. While other competing approaches can be modified to sample from such complex regions, this seamlessly fits into our framework. In contrast to the prior work on MOBO, we consider more general scalarizations that are only required to be Lipschitz and monotonic.

我们提出了一个MOO的灵活框架，使用了随机标量化的概念。我们的算法非常灵活，可以从整个Pareto front，或用户指定的任意区域中进行采样，也可以很自然的从Pareto front的非凸区域中进行采样。其他方法经过修改也可以从这样的复杂区域进行采样，但这无缝的吻合我们的框架。与之前MOBO的工作相比，我们考虑更一般化的标量化，只需要是Lipschitz和单调的。

2. We prove sublinear regret bounds making only assumptions of Lipschitzness and monotonicity of the scalarization function. To our knowledge the only prior work discussing theoretical guarantees for MOO algorithms is Pareto Active Learning (Zuluaga et al., 2013) with sample complexity bounds.

我们只假设了标量化函数的Lipschitzness和单调性，证明了亚线性的regret界限。据我们所知，之前唯一讨论过MOO算法的理论保证的工作，是Pareto主动学习，有样本的复杂的界限。

3. We compare our algorithm to other existing MOO approaches on synthetic and real-life tasks. We demonstrate that our algorithm achieves the said flexibility and superior performance in terms of the proposed regret, while being computationally inexpensive.

我们将我们的算法与其他MOO方法，在合成任务和真实生活任务上进行了比较。我们证明了，算法灵活性很高，以regret计性能非常好，同时计算量也不大。

### 1.1 Related Work

Most multi-objective bayesian optimization approaches aim at approximating the whole Pareto front. Predictive Entropy Search (PESMO) by Hernandez-Lobato et al. (2016) is based on reducing the posterior entropy of the Pareto front. SMSego by Ponweiser et al. (2008) uses an optimistic estimate of the function in an UCB fashion, and chooses the point with the maximum hypervolume improvement. Pareto Active Learning (PAL) (Zuluaga et al., 2013) and ε-PAL (Zuluaga et al., 2016) are similar to SMSego, and with theoretical guarantees. Campigotto et al. (2014) introduce another active learning approach that approximates the surface of the Pareto front. Expected hypervolume improvement (EHI) (Emmerich and Klinkenberg, 2008) and Sequential uncertainty reduction (SUR) (Picheny, 2015) are two similar approaches based on maximizing the expected hypervolume. Computing the expected hypervolume is an expensive process that renders EHI and SUR computationally intractable in practice when there are several objectives.

多数多目标贝叶斯优化方法的目标是，近似整个Pareto front。预测熵搜索(PESMO)是基于降低Pareto front的后验熵。SMSego使用了函数的UCB式的乐观估计，选择了有最大超体积改进的点。Pareto主动学习(PAL)和ε-PAL与SMSego类似，都带有理论保证。Campigotto等提出了另一种主动学习方法，近似了Pareto front的表面。期望超体积改进(EHI)和顺序不确定性消减(SUR)是两种类似的方法，基于期望超体积的最大化。计算期望的超体积是一个昂贵的过程，当有几个目标的时候，EHI和SUR的计算量非常大。

The idea of random scalarizations has been previously explored in the following works aimed at recovering the whole Pareto front: ParEGO (Knowles, 2006) which uses random scalarizations to explore the whole Pareto front; MOEA/D (Zhang and Li, 2007), an evolutionary computing approach to MOO; and MOEA/D-EGO (Zhang et al., 2010), an extension of MOEA/D using Gaussian processes that evaluates batches of points at a time instead of a single point. At each iteration, both ParEGO and MOEA/D-EGO sample a weight vector uniformly from the K−1 simplex, which is used to compute a scalar objective. The next candidate point is chosen by maximizing an off-the-shelf acquisition function over the GP fitted on the scalar objective. Our algorithm on the other hand, maintains K different GPs, one for each objective. Furthermore, our approach necessitates using acquisitions specially designed for the multi-objective setting for any general scalarization; more specifically, they are generalizations of single-objective acquisitions for multiple objectives (see Table 1). These differences with ParEGO are not merely superficial – our approach gives rise to a theoretical regret bound, while no such bound exists for the above methods.

随机标量化的思想在下面的工作中探索过，其目标是得到整个Pareto front：ParEGO使用随机标量化来探索整个Pareto front；MOEA/D，这是一种用进化计算的方法解决MOO问题；MOEA/D-EGO，这是MOEA/D算法的拓展，使用高斯过程来同时评估多个点。在每次迭代中，ParEGO和MOEA/D-EGO从K-1单纯形中均匀采样一个权值向量，用于计算得到一个标量目标。下一个候选点的选择，是要最大化采集函数，即在标量目标函数上拟合得到的GP。我们的算法则是维护K个不同的GPs，每个目标函数一个。而且，我们的方法需要使用专门为多目标设置对任何通用标量化设计的采集函数；更具体的，是单目标采集函数在多目标情况下的泛化（见表1）。这些与ParEGO相比的不同，并不是表面的不同，我们的方法可以得到理论上的regret界限，而对于上述方法则不存在这样的界限。

Another line of work involving scalarizations include utility function based approaches. Roijers et al. (2013); Zintgraf et al. (2015) propose scalar utility functions as an evaluation criteria. Zintgraf et al. (2018); Roijers et al. (2018, 2017) propose interactive strategies to maximize an unknown utility. In contrast to our approach the utility in these works is assumed to be fixed.

另一条涉及到标量化的研究线，包括基于效用函数的方法。有工作提出标量效用函数作为评估准则。有工作提出交互式策略来最大化未知的效用。与我们的方法相比，这些工作中的效用假设是固定的。

While there has been ample work on incorporating preferences in multi-objective optimization using evolutionary techniques (Deb and Sundar, 2006; Thiele et al., 2009; Kim et al., 2012; Branke and Deb, 2005; Branke, 2008), there has been fewer on using preferences for optimization, when using surrogate functions. Surrogate functions are essential for expensive black-box optimization. PESC (Garrido-Merchan and Hern andez-Lobato, 2016) is an extension of PESM allowing to specify preferences as constraints. Hakanen and Knowles (2017) propose an extension of ParEGO in an interactive setting, where users provide feedback on the observations by specifying constraints on the objectives in an online fashion. Yang et al. (2016) propose another way to take preferences into account by using truncated functions. An interesting idea proposed by Sato et al. (2007) uses a modified notion of Pareto dominance to prevent one or more objectives from being too small. The survey by Ishibuchi et al. (2008) on evolutionary approaches to MOO can be referred for a more extensive review.

在多目标优化中，采用进化技术结合偏好，有很多工作，但在使用代理函数时，使用偏好来进行优化的工作就少了很多。代理函数对于昂贵的黑箱优化是必不可少的。PESC是PESM的一种拓展，允许将偏好指定为约束。Hakanen和 Knowles提出了ParEGO在交互设置下的拓展，其中用户对观察提出反馈，以在线的方式对目标函数指定约束。Yang等提出另一种考虑偏好的方法，使用了截断函数。Sato等提出了一种有趣的想法，使用修正的Pareto支配概念，防止一个或多个目标函数太小。还可以参考Ishibuchi等对MOO的进化方法的综述。

When compared to existing work for MOO, our approach enjoys the following advantages. 与现有的MOO方法进行比较时，我们的方法有以下优势。

1. Flexibility: Our approach allows the flexibility to specify any region of the Pareto front including non-connected regions of the Pareto front, which is not an advantage enjoyed by other methods. Furthermore, the approach is flexible enough to recover the entire Pareto front when necessary. Our approach is not restricted to linear scalarization and extends to a much larger class of scalarizations.

灵活性：我们的方法非常灵活，可以指定Pareto front的任何区域，包括Pareto front的非连接区域，其他方法则无法实现这样的功能。而且，当需要时，方法还可以得到整个Pareto front。我们的方法不局限于线性标量化方法，可以拓展到更大类别的标量化方法。

2. Theoretical guarantees: Our approach seamlessly lends itself to analysis using our proposed notion of regret, and achieves sub-linear regret bounds.

理论保证：我们的方法使用我们提出的regret的概念，可以很自然的进行分析，得到亚线性的regret界限。

3. Computational simplicity: The computational complexity of our approach scales linearly with the number of objectives K. This is in contrast to EHI and SUR, whose complexity scales exponentially with K. Our method is also computationally cheaper than other entropy based methods such as PESMO.

计算简单性：我们的方法的计算复杂度，随着目标K的数量而线性增长。EHI和SUR则是随着K呈指数增长，这形成了鲜明的对比。我们的方法与其他基于熵的方法比计算量也更小，如PESMO。

## 2. Background

Most BO approaches make use of a probabilistic model acting as a surrogate to the unknown function. Gaussian processes (GPs) Rasmussen and Williams (2006) are a popular choice for their ability to model well calibrated uncertainty at unknown points. We will begin with a brief review of GPs and single objective BO.

多数BO方法使用概率模型作为对未知函数的代理。高斯过程(GPs)是一种流行的选择，因为可以在未知的点上很好的对标定的不确定性进行建模。我们先简单介绍下GPs和单目标BO。

**Gaussian Processes**. A Gaussian process (GP) defines a prior distribution over functions defined on some input space X. GPs are characterized by a mean function µ: X → R and a kernel κ: X × X → R. For any function f ∼ GP(µ, κ) and some finite set of points x1, ..., xn ∈ X, the function values f(x1), ..., f(xn) follow a multivariate Gaussian distribution with mean µ and covariance Σ given by µi = µ(xi), Σij = κ(xi, xj) ∀1≤i,j≤n. Examples of popular kernels include the squared exponential and the Matern kernel. The mean function is often assumed to be 0 without any loss of generality. The posterior process, given observations D = {(xi, yi)}^{t−1}_{i=1} where yi = f(xi) + εi ∈ R, εi ∼ N (µ, σ2), is also a GP with the mean and kernel function given by

高斯过程。一个高斯过程(GP)定义了在某输入空间X上定义的函数的先验分布。GPs的特征是均值函数µ: X → R，和核函数κ: X × X → R。对任何函数f ∼ GP(µ, κ)，和某有限点集x1, ..., xn ∈ X，函数值f(x1), ..., f(xn)遵循多变量高斯分布，均值µ和协方差Σ分别为µi = µ(xi), Σij = κ(xi, xj) ∀1≤i,j≤n。核函数的一些例子包括，平方幂和Matern核。不失一般性，均值函数一般假设为0。给定观察D = {(xi, yi)}^{t−1}_{i=1}，其中yi = f(xi) + εi ∈ R, εi ∼ N (µ, σ2)，后验过程也是一个GP，其均值和核函数为

$$µ_t(x) = k^T (Σ + σ^2I)^{−1}Y, κ_t(x, x') = κ(x, x') − k^T(Σ + σ^2I)^{−1}k'$$(2)

where Y = [yi]^t_{i=1} is the vector of observed values, Σ = [κ(xi, xj)]^t_{i,j=1} is the Gram matrix, k = [κ(x, xi)]^t_{i=1}, and k' = [κ(x', xi)]^t_{i=1}. Further details on GPs can be found in Rasmussen and Williams (2006).

**Bayesian Optimization**. BO procedures operate sequentially, using past observations {(xi, yi)}^{t−1}_ {i=1} to determine the next point xt. Given t-1 observations Thompson Sampling (TS) (Thompson, 1933) draws a sample gt from the posterior GP. The next candidate xt is choosen as xt = argmax gt(x). Gaussian Process UCB (Srinivas et al., 2010) constructs an upper confidence bound Ut as Ut(x) = µ_{t−1}(x) + \sqrt_{β_t}σ_{t−1}(x). Here µ_{t−1} and σ_{t−1} are the posterior mean and covariances according to equations 2. β_t is a function of t and the dimensionality of the input domain X. GP-UCB stipulates that we choose xt = argmax_{x∈X} U_t(x).

贝叶斯优化。BO过程是按顺序操作的，使用过去的观察{(xi, yi)}^{t−1}_ {i=1}来确定下一个点xt。给定t-1个观察，Thompson采样(TS)从后验GP中抽取一个样本gt。下一个候选xt选择为xt = argmax gt(x)。高斯过程UCB构建了一个置信度上限Ut为Ut(x) = µ_{t−1}(x) + \sqrt_{β_t}σ_{t−1}(x)。这里µ_{t−1}和σ_{t−1}为根据式2得到的后验均值和方差。β_t是t，和输入域X的维度的函数。GP-UCB要求，我们选择xt = argmax_{x∈X} U_t(x)。

In this paper, we assume that the K objectives f1, ..., fK are sampled from known GP priors GP(0, κ_k), (k ∈ [K]), with a common compact domain X ⊂ R^d. Without loss of generality, we assume X ⊆ [0, 1]^d. The feasible region is defined as f(X). We further assume that the observations are noisy, that is, y_k = f_k(x) + ε_k, where ε_k ∼ N (0, σ_k^2), ∀k ∈ [K].

本文中，我们假设K个目标函数f1, ..., fK，从已知的GP先验GP(0, κ_k), (k ∈ [K])中采样得到，有常见的紧凑域X ⊂ R^d。不失一般性，我们假设X ⊆ [0, 1]^d。可行区域定义为f(X)。我们进一步假设，观察是含有噪声的，即，y_k = f_k(x) + ε_k, 其中ε_k ∼ N (0, σ_k^2), ∀k ∈ [K]。

## 3. Our Approach

We first provide a formal description of random scalarizations, then we formulate a regret minimization problem, and finally propose multi-objective extensions of the classical UCB and TS algorithms to optimize it.

我们首先给出随机标量化的正式描述，然后我们表述出regret最小化问题，最终给出经典UCB和TS算法的多目标拓展，进行优化。

### 3.1 Random Scalarizations

As discussed earlier in Section 1 in this paper we consider a set of scalarizations s_λ parameterized by λ ∈ Λ. We assume a prior p(λ) with support Λ. We further assumed that, for all λ ∈ Λ, s_λ is L_λ-Lipschitz in the l1-norm and monotonically increasing in all the coordinates. More formally,

本文第1部分已经讨论过，我们标量化s_λ的集合，参数为λ ∈ Λ。我们假设先验p(λ)，支持为Λ。我们进一步假设，对于所有的λ ∈ Λ，s_λ是在l1-norm中是L_λ-Lipschitz的，在所有坐标中是单调递增的。更正式的

$$s_λ(y_1) − s_λ(y_2) ≤ L_λ||y1 − y2||_1, ∀λ ∈ Λ, y_1, y_2 ∈ R^d, and, s_λ(y_1) < s_λ(y_2), whenever y1 ≺ y2$$

The Lipschitz condition can also be generalized to lp-norms using the fact that ||y||_1 ≥ K^{1 − 1/p} ||y||_p for any p ∈ [1, ∞] and y ∈ R^K. Monotonicity ensures that

由于下式||y||_1 ≥ K^{1 − 1/p} ||y||_p对于任意的p ∈ [1, ∞]和y ∈ R^K是成立的，Lipschitz条件也可以泛化到lp范数中。单调性确保了

$$x^* _λ = argmax _{x∈X} s_λ(f(x)) ∈ X^*_f$$

since otherwise, if f(x^* _λ) ≺ f(x) for some $x /neq x^* _λ$, then we have s_λ(f(x^* _λ)) < s_λ(f(x)), leading to a contradiction. Each λ ∈ Λ maps to an x^* _λ ∈ X^* _f and a y^* = f(x^* _λ) ∈ f(X^* _f). Assuming the required measure theoretic regularity conditions hold, the prior p(λ) imposes a probability distribution on f(X^* _f) through the above mapping as depicted in Figure 1.

### 3.2 Bayes Regret

In contrast to (1), which returns a single optimal point, in this work, we aim to return a set of points from the user specified region. Our goal is to compute a subset X ⊂ X such that f(X) spans the high probability region of f(X^* _f). This can be achieved by minimizing the following Bayes regret denoted by R_B,

式(1)返回的是单个最优点，本文中，我们的目标是从用户指定的区域中返回一个点集。我们的目标是计算一个子集X ⊂ X，这样f(X)是f(X^* _f)的高概率区域。可以通过最小化下面的Bayes regret R_B得到，

$$R_B(X) = E_{λ∼p(λ)} (max_{x∈X} s_λ(f(x)) - max_{x∈X} s_λ(f(x))), X^* = argmin_{X⊂X ,|X|≤T} R_B(X)$$(4)

We now elaborate on the above expression. The pointwise regret max_{x∈X} s_λ(f(x)) − max_{x∈X} s_λ(f(x)) quantifies the regret for a particular λ and is analogous to the simple regret in the standard bandit setting (Bubeck et al., 2012). R_B(T) similarly corresponds to the Bayes simple regret in a bandit setting. The pointwise is minimized when x^* _λ = argmax _{x∈X} s_λ(f(x)) belongs to X. Since X is finite, the minimum may not be achieved for all λ, as the set of optimial points can be potentially infinite. However, the regret can be small when ∃x ∈ X such that f(x), f(x^* _λ) are close, from which it follows using the Lipschitz assumption that s_λ(f(x^* _λ)) − s_λ(f(x)) is small. Therefore, roughly speaking, the Bayes regret is minimized when the points in X are Pareto optimal and f(X) well approximates the high probability regions of f(X^* _f). In this case, s_λ(f(x^* _λ)) − s_λ(f(x)) is small for λs with high probabilities. Even though the rest of the regions are not well approximated, it does not affect the Bayes regret since those regions do not dominate the expectation by virtue of their low probability. This is what was desired from the beginning, that is, to compute a set of points with the majority of them spanning the desired region of interest. This is also illustrated in Figure 2 showing three scenarios which can incur a high regret.

我们现在解释一下上面的表达式。逐点regret max_{x∈X} s_λ(f(x)) − max_{x∈X} s_λ(f(x))对特定的λ计算regret，可以类比为标准bandit设置下的简单regret。R_B(T)类似的对应着bandit设置中的Bayes简单regret。逐点regret在x^* _λ = argmax _{x∈X} s_λ(f(x))处最小化。由于X是有限的，最小值不会在所有λ处得到，因为最优点的集合可能是无限的。但是，当∃x ∈ X，使得f(x), f(x^* _λ)很接近时，regret可能会非常小，因为会遵循Lipschitz假设，即s_λ(f(x^* _λ)) − s_λ(f(x))是很小的。因此，大致来说，Bayes regret在X中的点为Pareto最优值时，f(X)很好的近似了f(X^* _f)的高概率区域时，会最小化。在这种情况下，s_λ(f(x^* _λ)) − s_λ(f(x))对于高概率的λs是小的。即使剩下的区域没有得到很好的近似，但并没有影响Bayes regret，因为那些区域并不主宰期望，因为其概率很低。这是最初就期望的样子，即，计算点集，点集的主要部分支撑起期望的兴趣区域。这是图2描述的样子，表明3种可能会带来高regret的场景。

It is interesting to ask, why cannot one simply maximize 那么为什么不对下式进行最大化呢

$$max_{x∈X} E_{λ∼p(λ)} [s_λ (f(x))]$$

The above expression can be maximized using a single point x which is not the purpose of our approach. On the other hand, our proposed Bayes regret is not minimized by a single point or multiple points clustered in a small region of the Pareto front. Minimizing the pointwise regret for a single λ does not minimize the Bayes regret, as illustrated in Figure 2. Our proposed regret has some resemblance to the expected utility metric in Zintgraf et al. (2015). However, the authors present it as an evaluation criteria, whereas we propose an optimization algorithm for minimizing it and also prove regret bounds on it.

上面的表达式可以使用一个点x来最大化，这并不是我们的方法的目的。另一方面，我们提出的Bayes regret的最小化，并不是一个点，或在Pareto front的很小区域内的多个点。对单个λ最小化逐点regret，并不会最小化Bayes regret，如图2所示。我们提出的regret与Zintgraf等提出的期望效用度量有所类似。但是，作者提出是作为评估准则，而我们提出了一种优化算法来对其最小化，而且在其上证明了regret界限。

### 3.3 Scalarized Thompson Sampling and UCB

In this section we introduce Thompson Sampling and UCB based algorithms for minimizing the Bayes regret. In contrast to other methods based on random scalarizations (Knowles, 2006; Nakayama et al., 2009), our algorithm does not convert each observation to a scalar value and fit a GP on them, but instead models them separately by maintaining a GP for each objective separately. In each iteration, we first fit a GP for each objective using the previous observations. Then we sample a λ ∼ p(λ), which is used to compute a multi-objective acquisition function based on the scalarization s_λ. The next candidate point is chosen to be the maximizer of the acquisition function. The complete algorithm is presented in Algorithm 1 and the acquisition functions are presented in Table 1. The acquisition function for UCB is a scalarization of the individual upper bounds of each of the objectives. Similarly, the acquisition function for TS is a scalarization of posterior samples of the K objectives.

本节中，我们介绍基于Thompson Sampling和UCB的最小化Bayes regret算法。与其他基于随机标量化的方法相比，我们的算法并不将每个观察转化成一个标量值，然后拟合得到一个GP，而是分别进行建模，对每个目标维护一个GP。在每次迭代中，我们首先使用之前的观察对每个目标拟合一个GP。然后我们采样一个λ ∼ p(λ)，用于计算一个基于标量化s_λ的多目标采集函数。下一个候选点选择为采集函数的最大值点。完整算法如算法1所示，采集函数如表1所示。UCB的采集函数是每个目标函数的单个上限的标量化。类似的，TS的采集函数是K个目标的后验采样的标量化。

The intuition behind our approach is to choose the xt that minimizes the pointwise regret for the particular λt sampled in that iteration. Looking at the expression of the Bayes regret, at a high level, it seems that it can be minimized by sampling a λ from the prior and choosing an xt that minimizes the regret for the sampled λ. We prove regret bounds for both TS and UCB in Section 4 and show that this idea is indeed true.

在我们方法后面的直觉是，选择的xt，要对那次迭代中采样得到的特定λt的逐点regret进行最小化。在高层观察Bayes regret的表达式，似乎其最小化可以是，从先验中采样一个λ，选择一个xt对采样的λ的regret进行最小化。我们对TS和UCB在第4部分中证明了regret界限，表明这个想法是确实正确的。

**Practical Considerations**. In practice, our method requires the prior and class of scalarization functions to be specified by the user. These would typically be domain dependent. In practice, a user would also interactively update their prior based on the observations, as done in Hakanen and Knowles (2017); Roijers et al. (2017, 2018). Our approach is fully amenable to changing the prior interactively, and changing regions of interest. In this paper we do not propose any general methods for choosing or updating the prior, as it is not possible to do so for any general class of scalarizations. The interested readers can refer to the literature on interactive methods for MOBO. However, for the sake of demonstration we propose a simple heuristic in the experimental section.

实际的考虑。在实际中，我们的方法需要用户指定先验和标量化函数的类别。这通常是领域相关的。在实践中，用户还会互动的基于观察更新其先验。我们的方法还可以互动的改变先验，改变感兴趣区域。在本文中，我们并没有提出任何一般性方法来选择或更新先验，因为不能对任何一般性标量化类别这样做。感兴趣的读者可以参考MOBO互动方法的文献。但是，为进行展示，我们在试验部分给出一个简单的启发式。

### 3.4 Computational Complexity

At each step all algorithms incur a cost of at most O(KT^3), for fitting K GPs, except for ParEGO, which fits a single GP at each time step with a cost of O(T^3). The next step of maximizing the acquisition function differs widely across the algorithms. Computing the acquisition function at each point x costs O(T) for ParEGO, and O(KT) for our approach. The additional factor K is the price one must pay when maintaining K GPs.

在所有算法的每一步中，计算复杂度最多为O(KT^3)，以拟合K个GPs，除了ParEGO，在每个步骤中拟合一个GP，代价为O(T^3)。下一步最大化采集函数，会随着不同的函数而有很大不同。在每个点x处计算采集函数，对ParEGO的代价为O(T)，对我们的方法为O(KT)。当维护K个GPs时，就必须有额外的稀疏K。

Apart from fitting the K GPs, SMSEgo requires computing the expected hypervolume gain at each point which is much more expensive than computing the acquisitions for UCB or TS. Computing the expected hypervolume improvement in EHI is expensive and grows exponentially with K. PESM has a cost that is linear in K. However the computation involves performing expensive steps of expectation-propagation and MC estimates, which results in a large constant factor.

除了拟合K个GPs，SMSEgo需要在每个点处计算期望超体收益，这比计算UBC或TS的采集函数要更加昂贵的多。计算EHI中的期望超体改进是很昂贵的，随着K以指数增长。PESM的代价是随K线性增长的。但是计算涉及到期望传播和MC估计的昂贵步骤，会得到很大的常数系数。

## 4. Regret Bounds

In this section we provide formal guarantees to prove upper bounds on the Bayes regret RB which goes to zero as T → ∞. We also show that our upper bound is able to recover regret bounds for single objectives when K = 1.

本节中，我们给出正式保证，证明了在T → ∞时，Bayes regret RB的上限趋向于0。我们还证明了，当K=1时，我们的上限可以给出单目标的regret限。

Analogous to the notion of regret in the single-objective setting (Bubeck et al., 2012), we first define the instantaneous and cumulative regrets for the multi-objective optimization. The instantaneous regret incurred by our algorithm in step t is defined as,

与单目标设置中的regret概念类似，我们首先给出多目标优化中的瞬时regret和累积regret。我们算法在步骤t时的瞬时regret定义为

$$r(xt,λt) = max_{x∈X} s_{λ_t}(f(x)) − s_{λ_t}(f(x_t))$$(5)

where λt and xt are the same as in Algorithm 1. The cumulative regret till step T is defined as, 其中λt和xt和算法1中一样。到步骤T时的累积regret定义为

$$R_C(T) = \sum_{t=1}^T r(x_t,λ_t)$$(6)

For convenience, we do not explicitly mention the dependency of RC (T) on {xt}^T_{t=1} and {λt}^T_{t=1}. Next, we will make a slight abuse of notation here and define RB(T), the Bayes regret incurred till step T, as RB(XT) (See Eqn. 4), where XT = {xt}^T_{t=1}.

为方便，我们没有显式的提及RC(T)对{xt}^T_{t=1}和{λt}^T_{t=1}的依赖性。下一步，我们略微滥用一下概念，定义到步骤T时引入的Bayes regret RB(T)，为RB(XT)，其中XT = {xt}^T_{t=1}。

We further define the expected Bayes regret as ERB(T), where the expectation is taken over the random process f, noise ε and any other randomness occurring in the algorithm. Similarly, we also define the expected cumulative regret as ERC(T), where the expectation is taken over all the aforementioned random variables and additionally {λt}^T_{t=1}. We will show that the expected Bayes regret can be upper bounded by the expected cumulative regret, which can be further upper bounded using the maximum information gain.

我们进一步定义期望Bayes regret为ERB(T)，其中对随机过程f，噪声ε和算法中其他的随机性取期望。类似的，我们还定义了期望累积regret为ERC(T)，对所有之前提到的随机变量和{λt}^T_{t=1}取期望。我们还证明了，期望Bayes regret的上限为期望累积regret，也就是最大信息收益。

**Maximum Information Gain**. The maximum information gain (MIG) captures the notion of information gained about a random process f given a set of observations. For any subset A ⊂ X define yA = {ya = f(a) + εa|a ∈ A}. The reduction in uncertainty about a random process can be quantified using the notion of information gain given by I(yA; f) = H(yA)−H(yA|f), where H denotes the Shannon entropy. The maximum information gain after T observations is defined as

$$γT = max_{A⊂X :|A|=T} I(yA; f)$$(7)

**Regret Bounds**. We assume that ∀k ∈ [K], t ∈ [T], x ∈ X , fk(x) follows a Gaussian distribution with marginal variances upper bounded by 1, and the observation noise εtk ∼ N (0, σ^2_k) is drawn independently of everything else. Assume upper bounds Lλ ≤ L, σ^2_k ≤ σ^2, γTk ≤ γT , where γTk is the MIG for the k th objective. When X ⊆ [0, 1]^d, the cumulative regret after T observations can be bounded as follows.

**Theorem 1**. The expected cumulative regret for MOBO-RS after T observations can be upper bounded for both UCB and TS as,

$$ER_C(T) = O(L[\frac {K^2TdγTlnT} {ln(1+σ^{-2})}]^{1/2})$$(8)

The proof follows from Theorem 2 in the appendix. The bound for single-objective BO can be recovered by setting K = 1, which matches the bound of O(\sqrt_{TdγTln T}) shown in Russo and Van Roy (2014); Srinivas et al. (2010). The proof is build on ideas for single objective analyses for TS and UCB (Russo and Van Roy, 2014; Kandasamy et al., 2018).

Under further assumption of the space Λ being a bounded subset of a normed linear space, and the scalarizations s_λ being Lipschitz in λ, it can be shown that ERB(T) ≤ ERC(T)/T + o(1), which combined with Theorem 1 shows that the Bayes regret converges to zero as T → ∞. A complete proof can be found in Appendix B.3

## 5. Experimental Results

We experiment with both synthetic and real world problems. We compare our methods to the other existing MOO approaches in the literature: PESM, EHI, SMSego, ParEGO, and MOEA/D-EGO. EHI being computationally expensive is not feasible for more than two objectives. Other than visually comparing the results for three or lesser objectives we also compare them in terms of the Bayes regret defined in Eqn. 4.

我们用合成和真实世界问题进行试验。我们将我们的方法与其他现有的MOO方法进行比较：PESM，EHI，SMSego，ParEGO和MOEA/D-EGO。EHI是一个计算量很大的方法，不适用于多于2个目标的问题。除了视觉上比较3个或更少目标的问题的结果，我们还以Bayes regret对其进行了比较。

While our method is valid for any scalarization satisfying the Lipschitz and monotonicity conditions, we demonstrate the performance of our algorithm on two commonly used scalarizations, the linear and the Tchebyshev scalarizations (Nakayama et al., 2009) defined as,

我们的方法对任何满足Lipschitz和单调条件的标量化都是有效的，我们用两种常用的标量化方法证明了我们算法的性能，即线性标量化和TChebyshev标量化

$$s_λ^{lin} (y) = \sum_{k=1}^K λ_k y_k, s_λ^{tch} (y) = min_{k=1}^K λ_k (y_k - z_k)$$(9)

where z is some reference point. In both cases, Λ = {λ > 0 | ||λ||_1 = 1}. It can be verified that the Lipschitz constant in both cases is upper bounded by 1.

其中z是某参考点。在两种情况中，Λ = {λ > 0 | ||λ||_1 = 1}。可以验证，两种情况下的Lipschitz常数上限都是1。

**Choosing the weight distribution p(λ)**. While the user has the liberty to choose any distribution best suited for the application at hand, for demonstration we show one possible way. A popular way of specifying user preferences is by using bounding boxes (Hakanen and Knowles, 2017), where the goal is to satisfy fk(x) ∈ [ak, bk], ∀1 ≤ k ≤ K. We convert bounding boxes to a weight distribution using a heuristic described below.

选择权重分布p(λ)。用户可以自由选择任何适合于应用的分布，但为了进行展示，我们展示了一种可能的方式。指定用户偏好的一种流行的方法是，使用边界框，其中目标是满足fk(x) ∈ [ak, bk], ∀1 ≤ k ≤ K。我们将边界框转化成一个使用启发式的权重分布。

For the linear scalarization, it can be verified that the regret is minimized when y is pareto optimal, and the normal vector at the surface of the Pareto front at y has the same direction as λ. This is illustrated using a simple example in Figure 3. Consider two simple objectives f1(x, y) = xy, f2(x, y) = y·\sqrt_{1-x^2}. Sampling λ = [u/(u+1), 1/(u+1)] where u ∼ Unif (0, 0.3), results in the first figure. In this example we have λ1 smaller than λ2, resulting in exploration of the region where f2(x, y) is high. Whereas sampling λ = [u/(u+v), v/(u+v)] where u, v ∼ Unif (0.5, 0.7) results in the second figure since both components of λ have similar magnitudes. This idea leads to the following heuristic to convert bounding boxes to a sampling strategy. We sample as λ = u/||u||_1 where uk ∼ Unif (ak, bk), k ∈ [K]. The intuition behind this approach is shown in Figure 4. Such a weight distribution roughly samples points from inside the bounding box.

对于线性标量化，可以验证在y是Pareto最优，Pareto front在y处的表面的法向量与λ方向相同的时候，regret得到最小化。这如图3中的例子所示。考虑两个简单的目标函数f1(x, y) = xy, f2(x, y) = y·\sqrt_{1-x^2}。采样λ = [u/(u+1), 1/(u+1)] 其中u ∼ Unif (0, 0.3)，得到图3中的第一幅图。在这个例子中，我们有λ1小于λ2，得到的探索区域中f2(x,y)要更大。而采样λ = [u/(u+v), v/(u+v)] 其中u, v ∼ Unif (0.5, 0.7)，得到第二幅图，因为λ的两个分量有类似的幅度。这个思想可以得到下面的启发式，即将边界框转换成采样策略。我们采样λ = u/||u||_1 其中uk ∼ Unif (ak, bk), k ∈ [K]。这种方法背后的直觉如图4所示。这样一个权重分布采样的点大致是在边界框之内的。

For the Tchebychev scalarization, at the optimum, y − z is inversely proportional to λ. For the purpose of demonstration and comparison we would like both the scalarization to obtain similar objective values. Therefore, we reuse the λ sampled for the linear scalarization to get λ_{tch} = λ'/||λ'||_1 where λ' = (1/λ1, ..., 1/λK). We have normalized the vector so that it lies in Λ.

对于TChebyshev标量化，在最优点处，y-z与λ是成反比的。为了展示和比较，我们希望两种标量化得到类似的目标值。因此，我们重用线性标量化采样的λ，得到λ_{tch} = λ'/||λ'||_1 其中λ' = (1/λ1, ..., 1/λK)。我们对向量进行归一化，这样其在Λ中。

In order to explore the whole Pareto front, one can also specify a flat distribution. For instance consider the Dirichlet distribution on the simplex {x ∈ R^K | \sum^K_{k=1} x_k = 1, x > 0}. One can sample from the Dirichlet distribution as λ ∼ Dir(1, ..., 1), which roughly provides equal weight to all the objectives leading to exploration of the whole Pareto front. Other strategies include λ = |λ'|/||λ'||_1 where λ' ∼ N(0, I).

为探索整个Pareto front，也可以指定一个平坦的分布。比如，考虑在单纯形{x ∈ R^K | \sum^K_{k=1} x_k = 1, x > 0}上的Dirichlet分布。可以从Dirichlet分布中采样得到λ ∼ Dir(1, ..., 1)，对所有目标给出大致相同的权重，得到的是探索整个Pareto front。其他策略包括λ = |λ'|/||λ'||_1 其中 λ' ∼ N(0, I)。

Other possible ways of choosing the weight vector includes learning the distribution of the weight vector from interactive user feedback. In fact, our framework also allows us to perform a joint posterior inference on the GP model and the weight distribution, thus learning the weight distribution in a more principled manner. While we leave these methods to future work, this demonstrates the flexibility of our framework.

其他选择权重向量的可能方式包括，从互动用户反馈中学习权重向量分布。实际上，我们的框架也允许我们在GP模型和权重分布上进行联合后验推理，因此以一种更有原则的方式学习权重分布。我们将这些方法留给未来的工作，这也证明了我们的框架的灵活性。

**Experimental Setup**. For all our experiments, we use the squared exponential function as the GP kernel (in practice, this is a hyperparameter that must be specified by the user), given by κ(x1, x2) = s exp(−||x1 − x2||^2/(2σ^2)), where s and σ are parameters that are estimated during optimization. We perform experiments with both TS and UCB using both kinds of scalarizations. In Eqn. 4, we observe that the term E_λ max_{x∈X} s_λ(f(x)) is independent of the algorithm, hence it is sufficient to plot −E_λ max_{x∈X} s_λ(f(x)). In all our experiments, we plot this expression, thus avoiding computing the global maximum of an unknown function. For the purposes of computing the Bayes simple regret, we linearly map the objective values to [0, 1] so that the values are of reasonable magnitude. This however is not a requirement of our algorithm. Further experimental details can be found in the Appendix. The implementation can be found in Dragonfly, a publicly available python library for scalable Bayesian optimization (Kandasamy et al., 2019).

试验设置。在我们所有的试验中，我们使用高斯核作为GP核（实践中，有一个超参数需要用户指定），即κ(x1, x2) = s exp(−||x1 − x2||^2/(2σ^2))，其中s和σ是参数，在优化的过程中进行估计。我们用TS和UCB，使用两种标量化进行试验。在式4中，我们观察到，E_λ max_{x∈X} s_λ(f(x))是与算法无关的，因此只要画出−E_λ max_{x∈X} s_λ(f(x))就足够了。在我们的所有试验中，我们画出这个表达式，因此避免了计算一个未知函数的全局最大值。为计算Bayes简单regret，我们将目标函数值线性映射到[0, 1]，这样值的幅度就会很合理。但这并不是算法的要求。更多的试验细节可以在附录中找到。实现可以在一个公开可用的Python库Dragonfly中找到，这个库是可扩展贝叶斯优化库。

**Synthetic two-objective function**. We construct a synthetic two-objective optimization problem using the Branin-4 and CurrinExp-4 functions as the two objectives respectively. These are the 4-dimensional counterparts of the Branin and CurrinExp functions (Lizotte, 2008), each mapping [0, 1]^4 → R. For this experiment we specify the bounding boxes [(a1, b1),(a2, b2)]. We sample from three different regions, which we label as top: [(−110, −95),(23, 27)], mid:[(−80, −70),(16, 22)], and flat: where we sample from a flat distribution. We also sample from a mixture of the top and mid distributions denoted by top/mid, thus demonstrating sampling from non-connected regions in the Pareto front. Figure 5 shows a scatter plot of the sampled values for the various methods. The simple regret plots are shown in Figure 6.

合成两目标函数。我们使用Branin-4和CurrinExp-4函数作为两个目标函数，构建双目标优化问题。Branin和CurrinExp函数有对应的4维函数，每个都映射[0, 1]^4 → R。这个试验中，我们指定边界框[(a1, b1),(a2, b2)]。我们从三个不同的区域采样，我们标记为top: [(−110, −95),(23, 27)], mid:[(−80, −70),(16, 22)], 和flat: 我们从一个平坦分布中采样。我们还从top和mid的混合分布中采样，记为top/mid，这表示从Pareto front中的非连接区域中采样。图5展示了不同方法的采样值的散点图。图6给出了简单regret图。

**Synthetic six-objective function**. To show the viability of our method in high-dimensions, we sample six random functions fk: R^6 → R, fk ∼ GP(0, κ), k ∈ [6] where κ is the squared exponential kernel. Devoid of any domain knowledge about this random function, we linearly transform the objectives values to [0, 1] for simplicity. We specify the bounding box as [ak, bk] = [2/3, 1], ∀k ∈ [6] and denote it as the mid region, as the weight samples are of similar magnitude. The simple regret plot for this experiment is shown in Figure 7.

合成六目标函数。为展示我们方法在高维时的可行性，我们采样6个随机函数fk: R^6 → R, fk ∼ GP(0, κ), k ∈ [6] 其中κ是高斯核。关于这个随机函数没有任何领域知识，我们线性的将目标函数值变换到[0, 1]。我们指定边界框为[ak, bk] = [2/3, 1], ∀k ∈ [6]，将其表示为mid区域，因为权重样本具有类似的幅度。这个试验的简单regret图如图7。

**Locality Sensitive Hashing**. Locality Sensitive Hashing (LSH) (Andoni et al., 2015) is a randomized algorithm for computing the k-nearest neighbours. LSH involves a number of tunable parameters: the number of hash tables, number of hash bits, and the number of probes to make for each query. The parameters affect the average query time, precision and memory usage. While increasing the number of hash tables results in smaller query times, it leads to an increase in the memory footprint. Similarly, while increasing the number of probes leads to a higher precision, it increases the query time. We explore the trade-offs between these three objectives.

局部敏感哈希。LSH是一个随机算法，计算k最近邻。LSH涉及几个可调节的参数：哈希表的个数，哈希bit的数量，进行每次查询的probes的数量。参数会影响平均查询时间，精度和内存使用。增加哈希表的数量，会得到更小的查询次数，但会增加使用的内存。类似的，增加probes的数量，会得到更高的精度，但会增加查询时间。我们探索这三个目标之间的折中关系。

We run LSH using the publicly available FALCONN library on Glove word embeddings (Pennington et al., 2014). We use the Glove Wikipedia-Gigaword dataset trained on 6B tokens with a vocabulary size of 400K and 300-d embeddings. Given a word embedding, finding the nearest word embedding from a dictionary of word embeddings is a common task in NLP applications. We consider the following three objectives to minimize with their respective bounding boxes: Time (s) [0.0, 0.65], 1−Precision [0.0, 0.35], and the Memory (MB) [0, 1600]. The SR plots are shown in Figure 7 and the sampled objective values in Figure 8.

我们使用公开可用的FALCONN库在Glove word embeddings上运行LSH。我们使用Glove Wikipedia-Gigaword数据集，在6B token上训练，词典大小为400K和300-d embeddings。给定一个word embedding，从词典中找到最近的word embedding，这是一个常见的NLP任务。我们考虑下面三个目标进行最小化，有其对应的边界框：Time (s) [0.0, 0.65], 1−Precision [0.0, 0.35], and the Memory (MB) [0, 1600]。SR plots如图7所示，采样的目标值如图8所示。

**Viola Jones**. The Viola Jones algorithm (Viola and Jones, 2001) is a fast stagewise face detection algorithm. At each stage a simple feature detector is run over the image producing a real value. If the value is smaller than a threshold the algorithm exits with a decision, otherwise the image is processed by the next stage and so on. The Viola Jones pipeline has 27 tunable thresholds. We treat these thresholds as inputs and optimize for Sensitivity, Specificity, and the Time per query. We consider the following three objectives to minimize with their bounding boxes: 1−Sensitivity [0, 0.3], 1−Specificity [0, 0.13], and Time per query [0, 0.07]. Figure 7 shows the regret plot for this experiment.

Viola Jones算法是一个快速多阶段人脸检测算法。在每个阶段在图像上运行一个简单的特征检测器，产生一个实值。如果该值比阈值小，算法就退出得到一个决策，否则图像就继续进入下一个阶段处理。Viola Jones流程有27个可调节的阈值。我们将这个阈值作为输入，优化敏感性，专用性和每个查询的时间。我们考虑下面三个目标进行最小化，其边界框为：1−Sensitivity [0, 0.3], 1−Specificity [0, 0.13], 和Time per query [0, 0.07]。图7展示了这个试验的regret图。

**Results and Discussion**. Figures 5 and 8 show the sampling patterns of our proposed approach for the synthetic 2-d and the LSH glove experiment. We observe that our approach successfully samples from the specified region after some initial exploration, leading to a high concentration of points in the desired part of the Pareto front in the later iterations.

结果和讨论。图5和图8展示了我们提出的方法对合成2d和LSH glove试验的采样模式。我们观察到，我们的方法在进行了一些初始探索后，成功的从指定的区域进行采样，在后续的迭代中，在Pareto front的期望部分聚焦。

In Figures 6 and 7 we observe that the proposed approach achieves a smaller or comparable regret compared to the other baselines. We notice that the improvement is most significant for the high dimensional experiments. A plausible explanation for this could be that learning high dimensional surfaces have a much higher sample complexity. However, our since our approach learns only a part of the Pareto front, it is able to achieve a small regret in a few number of samples, thus demonstrating the effectiveness of our approach.

在图6和7中，我们观察到，提出的方法与其他基准对比，获得了更小的，或类似的regret。我们观察到，对高维试验来说，改进最为明显。这可能是因为，学习高维表面的采样复杂度更高。但是，由于我们的方法只学习了一部分Pareto front，只用几个样本就可以得到很小的regret，因此表现出了我们方法的有效性。

## 6. Conclusion

In this paper we proposed a MOBO algorithm for efficient exploration of specific parts of the Pareto front. We experimentally showed that our algorithm can successfully sample from a specified region of the Pareto front as is required in many applications, but is still flexible enough to sample from the whole Pareto front. Furthermore, our algorithm is computationally cheap and scales linearly with the number of objectives.

本文中，我们提出了一种MOBO算法，可以有效探索Pareto front的特定部分。我们通过试验表明，我们的算法可以成功的从Pareto front的指定区域进行采样，很多应用都有这样的需求，算法还非常灵活，可以从整个Pareto front中进行采样。而且，我们的算法计算量很小，与目标数量呈线性关系增加。

Our approach also lends itself to a notion of regret in the MO setting that also captures user preferences; with the regret being high if not sampled from the specified region or sampled outside of it. We provided a theoretical proof of the fact that our algorithm achieves a zero regret in the limit under necessary regularity assumptions. We experimentally showed that our approach leads to a smaller or comparable regret compared to the baselines.

我们的方法还带来了多目标设置下的regret概念，还可以捕获用户的偏好；如果不从指定区域采样，那么regret就会很高。我们给出了理论证明，只要给出必要的假设，算法的极限为零regret。我们通过试验表明，我们的方法与基准相比，得到的regret更小，或类似。