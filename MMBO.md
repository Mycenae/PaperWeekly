# Bayesian Optimization for Multi-objective Optimization and Multi-point Search

Takashi Wada, Hideitsu Hino

## 0. Abstract

Bayesian optimization is an effective method to efficiently optimize unknown objective functions with high evaluation costs. Traditional Bayesian optimization algorithms select one point per iteration for single objective function, whereas in recent years, Bayesian optimization for multi-objective optimization or multi-point search per iteration have been proposed. However, Bayesian optimization that can deal with them at the same time in non-heuristic way is not known at present. We propose a Bayesian optimization algorithm that can deal with multi-objective optimization and multi-point search at the same time. First, we define an acquisition function that considers both multi-objective and multi-point search problems. It is difficult to analytically maximize the acquisition function as the computational cost is prohibitive even when approximate calculations such as sampling approximation are performed; therefore, we propose an accurate and computationally efficient method for estimating gradient of the acquisition function, and develop an algorithm for Bayesian optimization with multi-objective and multi-point search. It is shown via numerical experiments that the performance of the proposed method is comparable or superior to those of heuristic methods.

贝叶斯优化是优化高评估代价的未知目标函数的有效方法。传统贝叶斯优化算法为单个目标函数在每次迭代中选一个点，而最近几年，提出了多目标优化的贝叶斯优化，或每次迭代多点搜索的贝叶斯优化。但是，目前还没有以非启发式的方式同时处理这两种情况的贝叶斯优化。我们提出一种贝叶斯优化算法，可以同时处理多目标优化和多点搜索。首先，我们定义了一个采集函数，同时考虑了多目标问题和多点搜索问题。要解析的最大化采集函数很难，因为计算代价即使在进行近似计算的情况下，比如采样近似，仍然太高；因此，我们提出了一种准确的，计算高效的方法，估计采集函数的梯度，提出了一种多目标多点搜索的贝叶斯优化算法。通过数值试验证明了，提出的方法的性能超过了启发式方法，或是可比的。

## 1. Introduction

Performance requirements for industrial products are getting stricter, and to develop a product that satisfies the required industrial standards, it is necessary to identify optimal design conditions by repetitively evaluating performance of products through prototyping or simulation. However, expenses and time for trial productions and simulations are limited, thus it is necessary to identify optimal design conditions within a few trials.

工业产品的性能要求变得越来越严格，为开发满足工业标准的产品，必须要通过原型试验或仿真来重复的评估产品性能，来找到最优的设计条件。但是，尝试生产和仿真的代价和时间是有限的，因此必须在几次试验中找到最优的设计条件。

Bayesian optimization (BO) (Shahriari et al., 2016; Brochu et al., 2010) is an efficient approach for optimizing unknown functions with high evaluation costs. BO can efficiently be used to search for a globally optimal solution x⋆ = argmin_{x∈X} f(x) with respect to the unknown function f(x), which represents the relation between objective variable and explanatory variable x ∈ X ⊂ R^{d_x}, where X is the feasible region of x. BO consists of steps of learning probability models and for determining points to be evaluated next, based on a certain evaluation criteria called acquisition function J(x), and the global optimal solution x⋆ is searched by repeating each step.

贝叶斯优化是优化高评估代价未知函数的有效方法。BO可以对未知函数f(x)高效的搜索全局最优解，x⋆ = argmin_{x∈X} f(x)，表示了目标变量和解释下变量x ∈ X ⊂ R^{d_x}之间的关系，X是x的可行区域。BO的步骤为，首先学习概率模型，基于特定的评估准则，称为采集函数J(x)，确定下一步评估的点，重复每个步骤，搜索得到全局最优解x⋆。

First, in learning probability model step, a model of the unknown function f(x) is learned based on the currently available dataset Dn = {(x1, f(x1)), ···, (xn, f(xn))}. A typical model for f(x) is the Gaussian process (GP) (Rasmussen & Williams, 2005). Next, in the step for determining the next evaluation point, the point xn+1 = argmax_{x∈X} J(x) at which the acquisition function J(x) is maximized is determined as the next point to be evaluated based on the learned probability model. Several methods for designing the acquisition function such as probability of improvement (PI) (Torn & Zilinskas, 1989), expected improvement (EI) (Jones et al., 1998), upper confidence bound (UCB) (Srinivas et al., 2010), entropy search (ES) (Hernández-Lobato et al., 2016), stepwise uncertainty reduction (SUR) (Picheny, 2015), and knowledge gradient (KG) (Frazier et al., 2009) have been proposed.

首先，在学习概率模型步骤，基于目前可用的数据集Dn = {(x1, f(x1)), ···, (xn, f(xn))}学习未知函数f(x)的一个模型。f(x)的典型模型是高斯过程。然后，在确定下一个评估点的步骤中，用xn+1 = argmax_{x∈X} J(x)来确定下一个评估点，即采集函数J(x)最大化的点，基于学习的概率模型。有几种设计采集函数的方法，如，改进概率(PI)，期望改进(EI)，置信度上界(UCB)，熵搜索(ES)，逐步不确定性降低(SUR)和知识梯度(KG)。

A simple BO is the optimization of a single objective variable, that is modeled as an objective function f(x). Also, when performing the iterative search, unknown function f is evaluated in succession. However, in reality, optimizing multiple objectives in a trade-off relationship, such as strength and weight as product performance, may be required. Also, in the evaluation phase, it is more efficient to simultaneously evaluate multiple points when performing the iterative search if it is possible to perform prototyping or simulation under multiple conditions in parallel. Owing to these requirements and circumstances, recently, BOs that can be employed in handling multi-objective optimization (Emmerich et al., 2011; Svenson & Santner, 2016; Ponweiser et al., 2008; Picheny, 2015; Hernández-Lobato et al., 2016) and multi-point searches (Ginsbourger et al., 2010; Chevalier & Ginsbourger, 2013; Marmin et al., 2015; Wu & Frazier, 2016; Shah & Ghahramani, 2015; Desautels et al., 2014; Chaudhuri & Haftka, 2014; Li et al., 2016) have been proposed. However, a BO method that can simultaneously handle both multi-objective optimization and multi-point search problems has not yet been proposed.

简单的BO是单目标优化，建模为目标函数f(x)。同时，当进行迭代搜索时，接着要评估未知函数f。但是，在实际中，需要优化有折中关系的多个目标，比如产品性能的强度的重要。同时，在评估阶段，在迭代搜索时，同时评估多个点会更高效，即在多个条件下并行进行原型试验或仿真。由于这些要求和条件，最近，采用BO来处理多目标优化和多点搜索。但是，还没有能同时进行多目标优化和多点搜索的BO。

Hence, this paper proposes a BO method that can simultaneously handle both multi-objective optimization and multi-point searches. In the proposed method, we define an acquisition function by extending the existing multi-objective optimization and multi-point search methods. Subsequently we consider an optimization problem to find maximum of the acquisition function. The acquisition function defined here involves multivariate integration. The approximate calculation using Monte Carlo sampling is often adopted, but generally it is computationally demanding. Furthermore, naı̈ve Monte Carlo sampling is not suitable for evaluating the acquisition function in BO because in a multi-point search problem the dimension of the variables to be optimized tend to significantly increase, e.g., when we consider q points in a simultaneous search, it implies we have to estimate the integral with respect to dx × q dimensional variables. Moreover, when the gradient of the objective function with a high-dimensional variable is estimated using sampling approximation, the approximated gradient tends to become a zero vector (vanishing gradient problem).

因此，本文提出了一种BO方法，可以同时处理多目标优化和多点搜索。在提出的方法中，我们定义了一个采集函数，拓展了现有的多目标优化和多点搜索方法。然后，我们考虑一个优化问题，找到采集函数的最大值。这里定义的采集函数涉及到多变量积分。通常采用使用蒙特卡洛采样的近似计算，但一般来说，计算量都很大。而且，朴素蒙特卡洛采样不适用于在Bo中评估采集函数，因为在多点搜索问题中，待优化的变量的维度会显著增加，如，当我们在同时搜索中考虑q个点，这意味着我们需要对dx × q维变量估计积分。而且，当高维变量目标函数的梯度是使用采样近似来进行估计的，近似的梯度倾向于是一个零向量（梯度消失的问题）。

Major contributions of this study are as follows: 本文的主要贡献如下：

- To the best of our knowledge, BO method for multi-objective and multi-point searches using non-heuristic approach is proposed for the first time in this paper.

据我们所知，本文第一次提出了多目标和多点的非启发式BO方法。

- We propose a computationally efficient algorithm for the proposed BO method based on Monte Carlo approximation of the gradient of the acquisition function. We empirically showed the newly designed acquisition function can effectively avoid the vanishing gradient problem.

我们对提出的BO方法，提出了计算上很高效的算法，基于对采集函数梯度的蒙特卡洛近似。我们通过经验表明，新设计的采集函数可以高效的避免梯度消失问题。

## 2. Related Work

### 2.1 Gaussian Process

In this work, we will use Gaussian Process (GP) as the probabilistic model for Bayesian optimization. GP is a stochastic process of a function values f(x) that follows prior distribution with mean λ and covariance between two points f(x) and f(x') is defined by a positive definite kernel function κ(x, x'). We introduce the covariance matrix Kn having elements [Kn]_{i,j} = κ(xi, xj). We adopted a well-known kernel function for GP is a Gaussian kernel of automatic relevance determination type (Rasmussen & Williams, 2005) for implementing the proposed method. Kernel parameters are determined by evidence maximization.

本文中，我们会使用高斯过程(GP)作为概率模型进行贝叶斯优化。GP是一个函数f(x)的随机过程，先验分布均值为λ，两个点f(x)和f(x')的协方差由正定核函数κ(x, x')来定义。我们引入协方差矩阵Kn，其元素为[Kn]_{i,j} = κ(xi, xj)。我们采用了一个著名的核函数，是一个高斯核，自动确定相关类型，实现提出的方法。核的参数是由证据最大化来确定的。

Given the observed dataset Dn, the posterior distribution of f(x) is defined as follows:

给定观察到的数据集Dn，f(x)的后验分布定义如下：

$$p(f(x)|Dn) = N(μn(x), σn^2(x)), μn(x) = λ + kn(x)^⊤ Kn^{−1}(f − λ1), σn^2(x) =κ(x, x) − kn(x)^⊤ Kn^{−1}kn(x)$$

where kn(x) = [κ(x, x1), ···, κ(x, xn)]^⊤, f = [f(x1), ···, f(xn)]^⊤ , and 1 is a vector with all ones. In an iterative search using BO, hyper parameters are estimated every time a new observation datum is added, and the posterior distribution p(f(x)|Dn) is updated.

在使用BO的迭代搜索中，在每次加入了新的观察后，都要进行超参数估计，然后更新后验分布p(f(x)|Dn)。

### 2.2 Multi-objective Bayesian Optimization

Here we describe BO dealing with multi-objective optimization that minimizes df objective variables. Hereinafter, unknown functions expressing the relation between each objective and explanatory variable are denoted as f^(1)(x), ···, f^(df)(x). They can be collectively represented as F(x) = [f^(1)(x), ···, f^(df)(x)]^⊤. In the objective variable in the trade-off relationship, there is no single solution that minimizes each objective variable. For a multi-objective optimization, a solution S1 is said to dominate a solution S2 if all the objective values of S1 are better than the corresponding objective values of solution S2. In this case, S2 is said to be dominated by S1. A non-dominated solution S is a solution that is not dominated by any other solution, and is also referred to as the Pareto solution. Generally, it is not a single point but a set X⋆ ⊂ X. Also, a plane on the range of object variable formed by a set of Pareto solutions are referred to as the Pareto front. We aim to search a finite number of solutions that closely approximate Pareto front because the Pareto solution is not a finite set. However, since the Pareto front is unknown, at the time of actual optimization, a new point is added to the surface formed by the non-dominated solution set Dn^⋆ defined by a set satisfying the following condition

这里我们描述处理多目标优化的BO，最小化df个目标变量。下面，表示每个目标和解释性变量的关系的未知函数表示为f^(1)(x), ···, f^(df)(x)，也可以集体表示为F(x) = [f^(1)(x), ···, f^(df)(x)]^⊤。在有折中关系的目标变量中，没有哪个解可以最小化每个目标变量。对多目标优化，一个解S1对解S2占优，就是说S1的所有目标值都要比S2的对应目标值要好。非占优解S，是没有被其他解占优的解，也被称为pareto解。一般来说，这并不是一个点，而是一个集合X⋆ ⊂ X。Pareto解的集合形成的目标值的平面称为Pareto front。我们的目标是，搜索有限数量的解，能够近似Pareto front，因为Pareto解不是有限集合。但是，由于Pareto front是未知的，在实际优化的时候，新的点加入到非占优解集合Dn^⋆的表面，由满足下面条件的集合定义

$$∀x, (x, f(x)) ∈ Dn^⋆ ⊂ Dn, ∀x', (x', f(x')) ∈ Dn, ∃k ∈ {1, · · · , df} such that f^(k)(x) ≤ f^(k)(x')$$(1)

based on the current observation data set Dn = {(x1, F(x1)), ···, (xn, F(xn))}. The solutions are searched based on the amount of improvement when adding a new point to update the observation dataset as Dn+1 = {Dn, (xn+1 , F(xn+1))}.

基于现有的观察数据集Dn = {(x1, F(x1)), ···, (xn, F(xn))}。在将新的点加入以更新观察数据集的时候，Dn+1 = {Dn, (xn+1 , F(xn+1))}，解的搜索，是基于改进的程度。

Hypervolume improvement (HVI) and additive epsilon (Zitzler et al., 2003) shown in Fig. 1 are often used as measures of improvement. Suppose HV(Dn⋆) is the Lebesgue measure (hypervolume) of the region dominated by Dn⋆ with a reference point (a user-defined parameter to specify the upper limit of the Pareto solution to be searched) as an upper bound. Then, HVI is defined as

图1中的HVI和加性epsilon，通常用于改进度量。假设HV(Dn⋆)是Dn*占优的区域的Lebesgue度量(hypervolume)，有一个参考点要指定为上界（一个用户定义的参数，指定要搜索的pareto解的上限）。这时，HVI定义为

$$HVI(F(x_{n+1})) = HV(Dn+1*) - HV(Dn*)$$(2)

Various BO methods for multi-objective optimization have been proposed. Expected hypervolume improvement (EHI) (Emmerich et al., 2011) is an extension of EI’s idea to the HVI, and the acquisition function is defined as

提出了多目标优化的各种BO方法。EHI是EI的思想在HVI上的拓展，采集函数定义为：

$$J(x) = E_{p(F(x)|Dn)} [HVI(F(x))]$$(3)

where p(F(x)|Dn) denotes the posterior distribution of F(x). To make the computation tractable, F(x) is often assumed to be independent as p(F(x)|Dn) = p(f(1)(x)|Dn) · · · p(f(df)(x)|Dn). Multiple output Gaussian process (Álvarez et al., 2010) and BO with correlated outputs (Shah & Ghahramani, 2016) have been proposed and our proposed method can be improved by introducing the correlation, with additional computational cost.

其中p(F(x)|Dn)表示F(x)的后验分布。为使计算容易处理，F(x)通常假设为独立的，即p(F(x)|Dn) = p(f(1)(x)|Dn) · · · p(f(df)(x)|Dn)。文章提出了多输出高斯过程，和带有相关输出的BO，我们提出的方法可以由引入相关来改进，计算量也有有所增加。

Expected maximum improvement (EMI) is proposed in (Svenson & Santner, 2016) by extending the idea of EI to the additive epsilon. Ponweiser et al. (2008) proposed the S-metric selection (SMS) method, which extends the concept of UCB to HVI. Picheny (2015) proposed the SUR method, which can be used to calculate the expected value of HVI in the entire feasible region X and search for a point that minimizes the expected value.

将EI拓展到加性epsilon中，文章提出了期望最大改进(EMI)。文章提出了S度量选择(SMS)方法，将UCB的概念拓展到了HVI中。文章提出了SUR方法，可以用于计算HVI在整个可行区域X中的期望值，搜索一个最大化期望值的点。

### 2.3 Bayesian Optimization with Multi-point Search

Herein we present a BO method for multi-point search which searches for q candidate points at each iterative search. Thereafter, q candidate points x(1), · · ·, x(q) are arranged to a vector X = [x(1)^T, · · · , x(q)^T]^T ∈ R^{qdx}.

下面我们提出了多点搜索的BO方法，在每次迭代搜索中搜索q个候选点。这里，q个候选点x(1), · · ·, x(q)组合成一个向量X = [x(1)^T, · · · , x(q)^T]^T ∈ R^{qdx}。

The existing BO methods for multi-point search can be divided into two categories: The first category is a non-greedy search approach with the acquisition function J(X) designed for q candidate points, and the point where Xn+1 = argmax_{X∈Xq} J(X) is determined as the next point to be evaluated. Here, Xq represents a feasible region of X. Concerning the design of acquisition function J(X), q-EI (Ginsbourger et al., 2010; Chevalier & Ginsbourger, 2013; Marmin et al., 2015) as an extension of EI, q-KG (Wu & Frazier, 2016) as an extension of KG, and PPES (Shah & Ghahramani, 2015) as an extension of ES have been proposed.

现有的多点搜索BO方法可以分为两类：第一类是非探索搜索方法，采集函数是为q个候选点设计的，满足Xn+1 = argmax_{X∈Xq} J(X)的点是下一个要评估的点。这里Xq表示X的可行区域。关于采集函数J(X)的设计，q-EI是EI的拓展，q-KG是KG的拓展，PPES是ES的拓展。

The second category is greedy search approach. Kriging believer constant liar proposed in (Ginsbourger et al., 2010) decides the first point x(1) using EI. The objective variable corresponding to the determined point x(1) can be calculated using the predictive mean μn(x) of GP and the next point is found using EI as if a new data point (x(1), μn(x))is obtained. BUCB (Desautels et al., 2014) is also a greedy search method based on UCB. It can sequentially determine candidate points while updating only the GP prediction variance σn(x). There is also a technique of imposing a penalty to exclude neighbor points already determined in greedy search from the candidate points. For example, Chaudhuri & Haftka (2014) proposed to exclude a region whose distance from the already determined points is equal to or less than a certain value. Li et al. (2016) proposed to impose a penalty so as to increase the mutual information between already decided points and the next candidate.

第二类是贪婪搜索方法。文章提出了Kriging believer constant liar，决定第一个点x(1)使用EI。对应确定点x(1)的目标变量可以使用GP的预测均值μn(x)来计算，下一个点使用EI来发现，就像得到了一个新数据(x(1), μn(x))。BUCB是一个基于UCB的贪婪搜索方法。可以在只更新GP预测方差σn(x)的同时，来顺序的决定候选点。也有一种技术，加入惩罚项，来从候选点中排除已经在贪婪搜索中确定的邻域点。比如，文章提出排除一个区域，其距离与已经确定的点等于或小于一定的值。文章提出加入一个惩罚项，以在已经决定的点和候选点之间增加互信息。

### 2.4 Heuristic Multi-objective multi-point Bayesian Optimization

In considering BO that can handle multi-objective optimization and multi-point searching at the same time, we will present a non-greedy method in Section 3. There are no other existing non-heuristic method for multi-objective and multi-point search method. Before developing a new method, we briefly discuss two naı̈ve extensions of the conventional methods for multi-objective BO to multi-point search based on greedy approach, and a multi-point search method, which individually decides candidate points in succession.

在考虑同时处理多目标优化和多点搜索的BO时，我们在第3部分提出一种非贪婪方法。对多目标和多点搜索方法，没有其他现有的非启发式方法。在提出一种新方法之前，我们简要的讨论传统的基于贪婪方法多目标BO多点搜索的两种朴素拓展，和一种多点搜索方法，独自确定后续的候选点。

The first method is based on multi-point search method proposed by Chaudhuri & Haftka (2014), which is denoted as DC (distance constraints). In the sequential determination of the candidate points, the explanatory variables are normalized to the range of [0, 1], and the Euclidean distance from the point already determined is 0.1*\sqrt_{dx} or less is excluded from the candidate region X. The algorithmic description of the multi-objective and multi-point search DC algorithm is presented in the supplementary material.

第一种方法是基于文章提出的多点搜索方法，表示为DC（距离约束）。在候选点的顺序确定中，解释性变量归一化到[0, 1]的范围内，从已经确定的点到候选点的欧式距离为0.1*\sqrt_{dx}，或更少的，从候选区域X中排除掉了。D多目标和多点搜索DC算法的算法描述，在附录中给出。

The second method is based on the method proposed by Ginsbourger et al. (2010), which is denoted as KB (knowledge believer). For sequential determination of the candidate points, we can substitute the value of the objective variable for the already decided point with the predicted mean μn(x) of GP, and temporarily add it to the observation data Dn for use in searching for the next candidate point. The algorithmic description of the multi-objective and milt-point search KB algorithm is presented in the supplementary material.

第二种方法，是基于文章提出的方法，表示为KB(knowledge believer)。对候选点的顺序确定，我们将已经确定的点的目标变量的值替换为GP预测均值μn(x)，临时加到观测值Dn上，在搜索下一个候选点时使用。KB多目标和多点搜索算法的描述在附录中给出。

In both these methods, the optimization of the acquisition function is done by using a conventional BO method for multi-objective functions.

在这两种方法中，采集函数的优化，使用传统多目标函数的BO进行。

## 3. Proposed Method

In this section, we present the proposed non-greedy multi-point search method based on the concept of EHI in multi-objective optimization.

本节中，我们给出提出的基于多目标优化中EHI概念的非贪婪多点搜索方法。

### 3.1 Design of Acquisition Function

As a simple extension of the acquisition function of EHI in Eq. (3) to a non-greedy multi-point search, we define the following acquisition function

作为式(3)中采集函数到非贪婪多点搜索的拓展，我们定义了如下的采集函数

$$q-EHI(X) = E_{p(Fq(X)|Dn)} [HVI(Fq(X))] $$(4)

where Fq(X) = [F(x(1))^⊤, · · ·, F(x(q))^⊤]^⊤ ∈ R^{qdf}, HVI(Fq(X)) is the improvement of the hypervolume when q candidates F(x(1)), · · ·, F(x(q)) are added to Dn⋆. p(Fq(X)|Dn) is the posterior of Fq(X) given the observation dataset Dn. Each of the objective functions f(k)(x) are assumed to follow independent Gaussian processes as p(Fq(X)|Dn) = p(f(1)(x(1)), · · ·, f(1)(x(q))|Dn) · · ·, p(f(df)(x(1)), ..., f(df)(x(q))|Dn).

Since the acqusition function (4) requires an integral with respect to multivariate distribution for calculating expectation, it is difficult to calculate it analytically. Indeed, it is pointed out the by Hernández-Lobato et al. (2016) that computation of EHI for multi-objective problem is feasible at most for two or three objectives, and when we consider multi-point search, the difficulty of the computation would exponentially grow. Therefore, approximation using Monte Carlo sampling is performed:

$$q-EHI_{MC}(X) = \sum_{m=1}^M HVI(\tilde F_{q,m}(X))/M$$(5)

where M is the number of Monte Carlo samplings, F̃q,m(X) is the m-th sample point that follows the distribution p(Fq(X)|Dn). As an example, let us consider a case df = 2, q = 2 as shown in Fig. 2. In this case, the elements of F̃q,m(X) are F̃2,m(X) = [F̃m(x(1))^⊤, F̃m(x(2))^⊤]^⊤ = [fm(1)(x(1)), fm(2)(x(1)), fm(1)(x(2)), fm(2)(x(2))]^⊤.

To find the maximizer of the acquisition function, it is possible to use metaheuristics such as a genetic algorithm. However, because an approximate evaluation of the acquisition function itself is computationally intensive, and the dimension of variable to be optimized can be very high when multiple point search is considered, metaheuristics which requires evaluating the objective function many times is prohibitive. Therefore, we derive a method for approximating the gradient of the acquisition function for efficient optimization based on the gradient method.

### 3.2 Gradient of the Acquisition Function

The gradient of the acquisition function (5) is given by

$$∇q-EHI_{MC}(X) = \sum_{m=1}^M \frac {∂HVI(F̃q,m(X))} {∂X} /M$$(6)

where

$$\frac {∂HVI(F̃q,m(X))} {∂X} = [\frac {∂HVI(F̃q,m(X))} {∂x(1)}, ..., \frac {∂HVI(F̃q,m(X))} {∂x(q)}]^T$$(7)

$$\frac {∂HVI(F̃q,m(X))} {∂x(i)} = \frac {∂F̃q,m(x(i))} {∂x(i)} \frac {∂HVI(F̃q,m(X))} {∂F̃q,m(x(i))} = \sum_{k=1}^{df} \frac {∂HVI(F̃q,m(X))} {∂fm(k)(x(i))} \frac {∂fm(k)(x(i))} {∂x(i)}$$

In order to calculate this gradient, we need to calculate the partial derivatives $\frac {∂HVI(F̃q,m(X))} {∂fm(k)(x(i))}$ and $\frac {∂fm(k)(x(i))} {∂x(i)}$.

Concerning the derivative $\frac {∂HVI(F̃q,m(X))} {∂fm(k)(x(i))}$, in actual problems requiring multi-objective optimization, df tends to be small, thus numerical derivation approach is computationally tractable in such scenario. Let δk be a vector with a small positive number at the k-th element and zero otherwise. We can approximate the infinitesimal change in HVI(F̃q,m(X)) by HVI(F̃q,m(X) + δk), k = 1, ..., df caused by an infinitesimal change in F̃q,m(X), and perform numerical differentiation using df + 1-times evaluation of the HVI.

On the other hand, $\frac {∂fm(k)(x(i))} {∂x(i)}$ cannot be estimated by numerical derivation in the same manner as $\frac {∂HVI(F̃q,m(X))} {∂fm(k)(x(i))}$, because for each q × dx element of X, we need to sample perturbed points according to the probability p(Fq (X)|Dn). In general, dx > df and the number of points to be searched q is set to a maximum allowable number by the prototyping or simulation system. The computational cost of estimating the probability distribution for a new point given small perturbation to an element of X is of order O(n2) and it grows in the process of BO.

One of the major contributions of this work is in deriving a computationally efficient method for estimating $\frac {∂fm(k)(x(i))} {∂x(i)}$, by generating sample gradients of fm(k)(x) that follows the distribution p(f(k)(x)|Dn). Because fm(k)(x) cannot be described as an explicit function of x, we introduce an approximation method. For a shift-invariant kernel κ, from the Bochner’s theorem (Bochner, 1959), there exists a Fourier dual s(w) as the spectral density of κ and for the normalized probability density function p(w) = s(w)/α and a realization b of uniform random variable B ∼ U [0, 2π], we have

$$κ(x, x') = αE_{p(w)} [exp(-jw^T (x-x'))] = 2αE_{p(w,b)} [cos(w^Tx+b) cos(w^Tx'+b)]$$

Let W ∈ R^{r×dx}, b ∈ R^r be the r realizations sampled from p(w, b), and consider a basis function φ(x) = \sqrt_{2α/r}cos(Wx+b) ∈ R^r, where cos acts element-wise. Then, the value of the kernel function is approximated as κ(x, x') ≃ φ(x)^⊤φ(x'). Now the sample drawn from probability distribution p(f(k)(x)|Dn) is approximated by a linear model

$$g(k)(x) = φ(x)^⊤ θ_φ + λ(k)$$(8)

$$θ_φ = (Φ^t Φ)^{-1} Φ^T(f(k)-λ(k)1)$$(9)

where Φ = [φ(x1), ..., φ(xn)]^T, λ(k) is the expectation of the prior distribution of the k-th objective variable. It is guaranteed that probability distribution of g(k)(x) converges to p(f(k)(x)|Dn) as r → ∞ (Neal, 1996).

Equation (8) is a function of x. Let [W]:,i be the i-th column vector of W. The gradient of g(k) is given by

$$\frac {∂g(k)(x)} {∂x[i]} = (\frac {∂φ(x)} {∂x[i]})^T θ_φ =(-sqrt_{2α/r} diag([W]_{:,i}) sin(Wx+b))^T θ_φ$$(10)

Now we can efficiently estimate the gradient of the acquisition function (6).

### 3.3 Dealing with the Vanishing Gradient Problem

Even when the candidate point X has a small probability of improvement, the value of the acquisition function of Eq. (4) can have a positive value q-EHI > 0, and an improvement direction of q-EHI exists. However, when calculating the acquisition function of Eq. (5) approximately by Monte Carlo sampling, all sample points tend to be dominated and q-EHI_MC (X) = 0. As a result, ∇q-EHI_MC (X) = 0, and the gradient-based optimization could get disrupted. Here, this problem is referred to as the vanishing gradient problem. To solve this problem, we introduce the idea of regret as shown in Fig. 3.

First we define the regret HVR_{MC}^{(q')} at a cadidate point x^{(q')} as

$$HVR_{MC}^{(q')} = \sum_{m=1}^M HVR(F̃m(x^{(q')}))/M; (F̃m(x^{(q')}), ∀m are dominated$); 0, otherwise$(11)

In this definition, HVR(F̃m(x^{(q')})) is the hypervolume dominated by the non-dominated solution set Dn* with a reference point F̃m(x^{(q')}) and the predefined limit points. The regret is an index indicating the extent the candidate point is dominated, and it becomes larger as the probability of improvement is smaller. Now we introduce a novel acquisition function by subtracting regret (11) from (5) as

$$q-EHIR_{MC}(X) = \sum_{m=1}^M HVI(F̃q,m(X))/M - \sum_{q'=1}^q HVR_{MC}^{(q')}$$(12)

As shown in Fig. 3, when there exists q' such that $HVR_{MC}^{(q')} \neq 0$, i.e., when there are candidate points with a very low probability of improvement, this value decreases due to the penalty of regret with respect to the candidate point $x^{(q')}$. Accordingly, when maximizing the acquisition function, the effect of reducing the amount of regret is imposed, and it is expected that the candidate point x(q') will move towards improvement. It should be noted that when $HVR_{MC}^{(q')} = 0$ at all candidate points, this novel acquisition function reduces to Eq. (5).

The gradient of the acquisition function (12) is

$$∇q-EHIR_{MC}(X) = \sum_{m=1}^M \frac {∂HVI(F̃q,m(X))} {∂X}/M - \sum_{q'=1}^q \frac {∂HVR_{MC}^{(q')}} {∂X}$$(13)

The first term is nothing but Eq. (6). The second term is, as in Subsection 3.2, divided into the partial derivative of HV R with respect to the objective variable and that with respect to the explanatory variable. The former is calculated by using numerical differentiation while the latter is estimated using Eq. (10). We summarized the proposed method, which is referred to as multi-objective multi-point Bayesian optimization (MMBO), in Algorithm 1.

## 4. Experimental Results

To show the effectiveness of MMBO, we will first show its calculation cost and approximation accuracy in the gradient calculation of the acquisition function as discussed in Section 3.2. Subsequently, we compare MMBO and the heuristic method discussed in Subsection 2.4 on the test functions introduced in (Huband et al., 2006).

### 4.1 Evaluation of the Gradient Approximation

We evaluate the computational cost and approximation accuracy of gradient of acquisition function by MMBO on the test function ZDT1 (Huband et al., 2006). ZDT1 is a test function that can set the number of arbitrary explanatory variables with two objective variables (df = 2), and dx = 3, 6. We also varied the number of multi-points to be searched as q from 2 to 8.

The number of samples for approximating the acquisition function (12) is set to M = 200, and the number of bases φ(x) for approximating the posterior of GP is set to r = 300. The number of initial observation was set to n = 30 and we considered the approximate computation of the gradient of the acquisition function at a random point.

Figure 4 shows a comparison result of the computational costs between the proposed method and those by using numerical differentiation of the acquisition function. The computational cost increases as the dimension dx and the number of search points q increase. The increase in computational cost for a numerical differentiation-based method with the increase in dx and q is rapid while that of MMBO is relatively low. It is noteworthy that an increase in the dimension of the explanatory variable has a minor impact on MMBO. This is because the computational cost of the proposed gradient approximation method is mainly associated with the evaluation of the numerical derivative of the acquisition function by the objective function, which is independent of the dimension of the explanatory variable.

Next we evaluate the accuracy of the gradient approximation. Since it is difficult to calculate the ground truth of the gradient, we performed numerical derivative with M = 1, 000 and regard the result as the (nearly) ground truth gradient. Then, we calculate the angle θ between the ground truth and the estimated gradient using the proposed method. Figure 5 shows the accuracy of the gradient estimation in θ [rad] when the dimension of the explanatory variable and the number of points to be searched are varied. For each combination of the dimension of the explanatory variable and number of points to be searched, we randomly sampled 100 points and gradients were evaluated at these points (Fig. 5). From this figure, it can be seen that lower the dimension and the smaller the search points, higher the accuracy of the gradient estimation method. It can also be seen that even when dx = 6, q = 8, the average of the angle between the true and estimated gradients is θ̄ = 0.30[rad], indicating high accuracy of the MMBO method.

From these results, we can conclude that the proposed method can efficiently approximate the gradient of the acquisition function and is computationally efficient.

### 4.2 Evaluation of the Search Efficiency

We evaluated the search efficiency of the multi-point search method with multi-objective optimization methods using three test functions. Each of these test functions is characterized by the dimension of explanatory variable dx, the number of objective functions df, the shape of the objective functions (unimodal/multi-modal), and the shape of Pareto front (concave/convex/disconnected), as shown in Table 1.

The parameters of the MMBO are the same as those discussed in Subsection 4.1. Quasi-Newton method is used for optimizing the acquisition function with the approximated gradient using the proposed method. The number of randomly selected initial data points is n = 3dx . The quality of the search is evaluated by the hypervolume dominated by the observed data at each iteration of the optimization process. We compared MMBO with the two heuristic methods discussed in Subsection 2.4. In these two heuristic methods, the acquisition function was derived from EHI in the same manner as in our proposed method, and we used the genetic algorithm for optimizing the acquisition function.

By changing the random number for sampling the initial observation points 20 times, we evaluate the hypervolumes at each step of iteration. Figure 6 shows the search results for each test function when the number of search points q was set at q = 2, 4, 8. For test functions ZDT1 and DTLZ2, the hypervolumes converged to a large value in the early stage of the optimization in MMBO, and shown the best search efficiency among all the methods. Conversely, for ZDT3, the value to which the final hypervolume converged was inferior to those obtained in the case of heuristic methods. This result can be attributed to the fact that the test function ZDT3 is multi-modal and is difficult for gradient-based methods to find the global optimal solution. We discovered that at the early stage of the optimization for ZDT3, MMBO was comparable to other heuristic methods, and for q = 8 in which the number of search points is large, MMBO outperformed the others.

## 5. Conclusion

We proposed a novel non-heuristic Bayesian optimization method that can simultaneously handle multi-objective optimization and multi-point searches. We defined an acquisition function that can simultaneously handle multi-objective optimization and multi-point search problems, and proposed an efficient and accurate method for calculating the gradient of the acquisition function. The effectiveness of the proposed method was validated using different test functions and it was confirmed that it can be searched more efficiently than the heuristic methods for the unimodal objective function. Conversely, we found that the proposed method is subject to local solution problem. Our future work will focus on better ways of finding a good initial value in the gradient method and improving the acquisition function itself so as not to fall into the local solution. Furthermore, we proposed an approximation method for calculating the gradient of the acquisition function. Our future work will include combining the proposed approximation method with the recently developed quasi-Monte Carlo sequence-based methods (Leobacher & Pillichshammer, 2014; Buchholz et al., 2018) or quadrature Fourier feature-based method (Mutny & Krause, 2018) for stochastic gradients to improve its convergence speed.