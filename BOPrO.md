# Bayesian Optimization with a Prior for the Optimum

Artur Souza et. al. @ Lund University

## 0. Abstract

While Bayesian Optimization (BO) is a very popular method for optimizing expensive black-box functions, it fails to leverage the experience of domain experts. This causes BO to waste function evaluations on bad design choices (e.g., machine learning hyperparameters) that the expert already knows to work poorly. To address this issue, we introduce Bayesian Optimization with a Prior for the Optimum (BOPrO). BOPrO allows users to inject their knowledge into the optimization process in the form of priors about which parts of the input space will yield the best performance, rather than BO’s standard priors over functions, which are much less intuitive for users. BOPrO then combines these priors with BO’s standard probabilistic model to form a pseudo-posterior used to select which points to evaluate next. We show that BOPrO is around 6.67× faster than state-of-the-art methods on a common suite of benchmarks, and achieves a new state-of-the-art performance on a real-world hardware design application. We also show that BOPrO converges faster even if the priors for the optimum are not entirely accurate and that it robustly recovers from misleading priors.

贝叶斯优化是优化昂贵的黑箱函数的一种流行方法，但不能利用领域专家的经验。这导致BO在不好的设计选项中浪费了函数评估（如，机器学习超参数），而专家已经知道这些选项效果是很差的。为解决这个问题，我们提出了带有向最优的先验的贝叶斯优化(BOPrO)。BOPrO使用户可以向优化过程中注入知识，即一些关于哪部分输入空间会产出最佳性能的先验知识，而不是BO关于函数的标准先验，这对于用户非常不直观。BOPrO然后将这些先验与BO的标准概率模型结合起来，形成伪后验，用于选择下面来评估哪个点。我们证明了，BOPrO比标准方法在常见的基准测试包上快了大约6.67倍，在真实世界硬件设计应用中获得了新的目前最好的性能。我们还证明了，即使关于最优的先验知识并不是完全准确的，BOPrO收敛的更快，还可以从有误导的先验中很稳健的恢复出来。

## 1. Introduction

Bayesian Optimization (BO) is a data-efficient method for the joint optimization of design choices that has gained great popularity in recent years. It is impacting a wide range of areas, including hyperparameter optimization [43,11], AutoML [21],robotics [6], computer vision [32,4], Computer Go [7], hardware design [33,25], and many others. It promises greater automation so as to increase both product quality and human productivity. As a result, BO is also established in large tech companies, e.g., Google [14] and Facebook [1].

贝叶斯优化(BO)是一种设计选项联合优化的一种数据高效方法，最近几年流行度非常高，正在影响非常多的领域，包括超参数优化，AutoML，机器人学，计算机视觉，计算机围棋，硬件设计，和很多其他领域。采用这种方法可以得到更大的自动化程度，可以提高产品质量和生产率。结果是，BO在大型科技公司是流行的技术，如Google和Facebook。

Nevertheless, domain experts often have substantial prior knowledge that standard BO cannot easily incorporate so far [46]. Users can incorporate prior knowledge by narrowing the search space; however, this type of hard prior can lead to poor performance by missing important regions. BO also supports a prior over functions p(f), e.g., via a kernel function. However, this is not the prior domain experts have: they often know which ranges of hyperparameters tend to work best [38], and are able to specify a probability distribution p_{best}(x) to quantify these priors; e.g., many users of the Adam optimizer [23] know that its best learning rate is often in the vicinity of 1 × 10^{−3} (give or take an order of magnitude), yet one may not know the accuracy one may achieve in a new application. Similarly, Clarke et al. [8] derived neural network hyperparameter priors for image datasets based on their experience with five datasets. In these cases, users know potentially good values for a new application, but cannot be certain about them.

尽管如此，领域专家有一些关键的先验知识，目前标准BO方法不能很容易的集成。用户可以通过缩小搜索空间来集成先验知识；但是，这种硬先验可能导致较差的性能，错失重要的区域。BO还支持函数p(f)的先验，如，通过核函数。但是，这不是领域专家所用于的先验：他们通常知道的是，哪些范围的超参数可能效果最好，可以指定概率分布p_{best}(x)来量化这种先验；如，很多Adam优化器的用户知道，最佳学习速率通常是在1 × 10^{−3}附近（最多一个数量级的差别），但是不知道在新应用中可能得到的准确率。类似的，Clarke等[8]基于在5个数据集上的经验，推导出了图像数据集的神经网络超参数先验。在这些情况中，用户知道新应用的可能较好的值，但对其不确定。

As a result, many competent users instead revert to manual search, which can fully incorporate their prior knowledge. A recent survey showed that most NeurIPS 2019 and ICLR 2020 papers reported having tuned hyperparameters used manual search, with only a very small fraction using BO [5]. In order for BO to be adopted widely, and help facilitate faster progress in the ML community by tuning hyperparameters faster and better, it is therefore crucial to devise a method that fully incorporates expert knowledge about the location of high performance areas into BO. In this paper, we introduce Bayesian Optimization with a Prior for the Optimum (BOPrO), a novel BO variant that combines priors for the optimum with a probabilistic model of the observations made. Our technical contributions are:

结果是，很多能干的用户退回到了手工搜索，这可以完全利用其先验知识。最近的调查表明，多数NeuralIPS 2019和ICLR 2020的论文在调节超参数时使用的是手工搜索，只有一小部分使用BO。为使BO广泛采用，并促进ML的更快发展，更快更好的调节超参数，设计一种方法完全集成高性能区域的位置的领域知识到BO中，就非常关键。本文中，我们提出带有对最优位置先验的BO(BOPrO)，一种新的BO变体，可以结合最优位置先验的概率模型。我们的贡献如下：

- We introduce Bayesian Optimization with a Prior over the Optimum, short BOPrO, which allows users to inject priors that were previously difficult to inject into BO, such as Gaussian, exponential, multimodal, and multivariate priors for the location of the optimum. To ensure robustness against misleading priors, BOPrO gives more importance to the data-driven model as iterations progress, gradually forgetting the prior.

我们提出了BOPrO，用户可以将先验知识注入到BO中，之前是很难这样做的，比如最优位置的先验为高斯的，指数分布，多模分布的，和多元分布。为确保对误导的先验有稳健性，BOPrO在迭代进行的过程中，给数据驱动模型越来越多的重要性，逐渐忘记了先验。

- BOPrO’s model bridges the gap between the well-established Tree-structured Parzen Estimator (TPE) methodology, which is based on Parzen kernel density estimators, and standard BO probabilistic models, such as Gaussian Processes (GPs). This is made possible by using the Probability of Improvement (PI) criterion to derive from BO’s standard posterior over functions p(f|(xi, yi)^t_{i=1}) the probability of an input x leading to good function values.

树状Parzen估计器(TPE)的方法是基于Parzen核密度估计器的，标准BO是基于概率模型的，如高斯过程，BOPrO模型则弥合了这两者之间的空白。我们使用改进概率(PI)的原则来从BO的函数p(f|(xi, yi)^t_{i=1})标准后验的推导得出，这是输入x得到好的函数值的概率。

- We demonstrate the effectiveness of BOPrO on a comprehensive set of synthetic benchmarks and real-world applications, showing that knowledge about the locality of an optimum helps BOPrO to achieve similar performance to current state-of-the-art on average 6.67× faster on synthetic benchmarks and 1.49× faster on a real-world application. BOPrO also achieves similar or better final performance on all benchmarks.

我们在很多合成基准测试和真实世界应用中证明了BOPrO的有效性，表明关于最优性能的位置的知识，帮助BOPrO得到了与目前最好的性能，在合成基准测试上速度快了平均6.67x，在真实应用中快了1.49x。BOPrO在所有基准测试上，都得到了类似的或更好的最终性能。

BOPrO is publicly available as part of the HyperMapper optimization framework. BOPrO是HyperMapper优化框架的一部分。

## 2 Background

### 2.1 Bayesian Optimization

Bayesian Optimization (BO) is an approach for optimizing an unknown function f: X → R that is expensive to evaluate over an input space X. In this paper, we aim to minimize f, i.e., find x∗ ∈ argmin_{x∈X} f(x). BO approximates x∗ with a sequence of evaluations x1, x2, . . . ∈ X that maximizes an utility metric, with each new x_{t+1}depending on the previous function values y1, y2, . . . , yt at x1, . . . , xt. BO achieves this by building a posterior on f based on the set of evaluated points. At each iteration, a new point is selected and evaluated based on the posterior, and the posterior is updated to include the new point (x_{t+1}, y_{t+1}).

贝叶斯优化(BO)是一种优化未知函数f: X → R的方法，函数评估起来一般很昂贵，输入空间为X。本文中，我们的目标是最小化f，即，找到x∗ ∈ argmin_{x∈X} f(x)。BO用一系列评估x1, x2, . . . ∈ X最大化一个效用度量，来近似x*，每个新的x_{t+1}依赖于在x1, . . . , xt处的函数值y1, y2, . . . , yt。BO基于已经评估的点，构建了f的一个后验，得到最终的结果。在每次迭代中，基于这个后验来选择一个新的点进行评估，然后将新的点(x_{t+1}, y_{t+1})包括进来来得到新的后验。

The points explored by BO are dictated by the acquisition function, which attributes an utility to each x ∈ X by balancing the predicted value and uncertainty of the prediction for each x [41]. In this work, as the acquisition function we choose Expected Improvement (EI) [31], which quantifies the expected improvement over the best function value found so far:

BO探索过的点由采集函数来决定，对每个x ∈ X指定了一个效用，对在每个x上的预测的值和预测的不确定性之间进行均衡。本文中，我们选择期望改进EI作为采集函数，量化了在迄今为止发现的最佳函数值之上的期望改进：

$$EI_{y_{inc}}(x) := \int_{-∞}^{+∞} max(y_{inc} − y, 0)p(y|x)dy$$(1)

where yinc is the incumbent function value, i.e., the best objective function value found so far, and p(y|x) is given by a probabilistic model, e.g., a GP. Alternatives to EI would be Probability of Improvement (PI) [22,26], upper-confidence bounds (UCB) [44], entropy-based methods (e.g. Hernández-Lobato et al. [18]), and knowledge gradient [47].

其中yinc是目前的函数值，即，迄今为止发现的最佳目标函数值，p(y|x)由一个概率模型给出，如，一个高斯过程。EI之外的选择可能是，改进概率(PI)，置信度上限(UCB)，基于熵的方法，和知识梯度。

### 2.2 Tree-structured Parzen Estimator

The Tree-structured Parzen Estimator (TPE) method is a BO approach introduced by Bergstra et al. [3]. Whereas the standard probabilistic model in BO directly models p(y|x), the TPE approach models p(x|y) and p(y) instead. This is done by constructing two parametric densities, g(x) and b(x), which are computed using the observations with function value below and above a given threshold, respectively. The separating threshold y∗ is defined as a quantile of the observed function values. TPE uses the densities g(x) and b(x) to define p(x|y) as:

树结构Parzen估计器(TPE)是Bergstra等[3]提出的一种BO方法。BO中的标准概率模型对p(y|x)进行建模，而TPE方法对p(x|y)和p(y)进行建模。通过构建两个参数化密度来进行，即g(x)和b(x)，这是在给定阈值的情况下，通过使用大于和小于该阈值观察的函数值来计算的。TPE使用密度g(x)和b(x)来定义p(x|y)如下：

$$p(x|y) = g(x)I(y < y∗) + b(x)(1 − I(y < y∗))$$(2)

where I(y < y∗) is 1 when y < y∗ and 0 otherwise. Bergstra et al. [3] show that the parametrization of the generative model p(x, y) = p(x|y)p(y) facilitates the computation of EI as it leads to EI_{y∗} (x) ∝ g(x)/b(x) and, thus, argmax_{x∈X} EI_{y∗} (x) = argmax_{x∈X} g(x)/b(x).

其中I(y < y∗)在y < y*时为1，其余处为0。Bergstra等[3]表明，生成式模型的参数化p(x, y) = p(x|y)p(y)，促进了EI的计算，因为这带来了EI_{y∗} (x) ∝ g(x)/b(x)，因此，argmax_{x∈X} EI_{y∗} (x) = argmax_{x∈X} g(x)/b(x)。

## 3 BO with a Prior for the Optimum

We now describe our BOPrO approach, which allows domain experts to inject user knowledge about the locality of an optimum into the optimization in the form of priors. BOPrO combines this user-defined prior with a probabilistic model that captures the likelihood of the observed data Dt = (xi, yi)^t_{i=1}. BOPrO is independent of the probabilistic model being used; it can be freely combined with, e.g., Gaussian processes (GPs), random forests, or Bayesian NNs.

我们现在描述BOPrO方法，使领域专家可以将最优值的位置以先验的形式注入到优化过程中。BOPrO将这种用户定义的先验与捕获观测数据Dt = (xi, yi)^t_{i=1}的似然的概率模型结合到一起。BOPrO与使用的概率模型是独立的；可以与概率模型自由组合，如，高斯过程(GPs)，随机森林，或贝叶斯NNs。

### 3.1 BOPrO Priors

BOPrO allows users to inject prior knowledge w.r.t. promising areas into BO. This is done via a prior distribution that informs where in the input space X we expect to find good f(x) values. A point is considered “good” if it leads to low function values, and potentially to a global optimum. We denote the prior distribution Pg(x), where g denotes that this is a prior on good points and x ∈ X is a given point. Examples of priors are shown in Figures 2 and 3, additional examples of continuous and discrete priors are shown in Appendices A and D, respectively. Similarly, we define a prior on where in the input space we expect to have “bad” points. Although we could have a user-defined probability distribution Pb(x), we aim to keep the decision-making load on users low and thus, for simplicity, only require the definition of Pg(x) and compute Pb(x) = 1 − Pg(x). Pg(x) is normalized to [0, 1] by min-max scaling before computing Pb(x).

BOPrO使用户可以将先验知识，即有希望的区域，注入到BO过程中。我们用一个先验分布，表明输入空间X中的哪些地方可能找到好的f(x)值，来完成注入。一个点是好的，如果可能得到低的函数值，潜在可能是一个全局最优点。我们将先验分布表示为Pg(x)，其中g表示，这是一个关于好的点的先验，x ∈ X是一个给定的点。先验的例子如图2和3表示，连续先验和离散先验的例子如附录A和D所示。类似的，我们还可以定义，输入空间中期望有坏的点的先验。虽然我们可能有用户定义的概率分布Pb(x)，我们想要让用户方面的决策任务较低，因此，为简化，只需要Pg(x)的定义，计算Pb(x)=1-Pg(x)。Pg(x)在计算Pb(x)前通过min-max scaling来进行归一化。

In practice, x contains several dimensions but it is difficult for domain experts to provide a joint prior distribution Pg(x) for all of them. However, users can typically easily specify, e.g., sketch out, a univariate or bivariate prior distribution for continuous dimensions or provide a list of probabilities for discrete dimensions. In BOPrO, users are free to define a complex multivariate distribution, but we expect the standard use case to be that users mainly want to specify univariate distributions, implicitly assuming a prior that factors as Pg(x) = \prod_{i=1}^D Pg(xi), where D is the number of dimensions in X and xi is the i-th input dimension of x. To not assume unrealistically complex priors and to mimic what we expect most users will provide, in our experiments we use factorized priors; in Appendix E we show that these factorized priors can in fact lead to similar BO performance as multivariate priors.

在实践中，x包含几个维度，领域专家很难对所有维度给出一个联合先验分布Pg(x)。但是，用户一般可以很容易的对连续维度给出一个单变量或双变量的先验分布，或对离散维度给出概率列表。在BOPrO中，用户可以自由定义一个复杂的多变量分布，但是我们期望标准的使用情况是，用户主要想要指定单变量分布，隐式的假设一个先验可以分解成，Pg(x) = \prod_{i=1}^D Pg(xi)，其中D是X中的维度数量，xi是x的第i个输入维度。在我们的试验中，我们使用分解先验，而不假设不现实的复杂先验，模仿我们期望的多数用户会给出的；在附录E中，我们展示了这些分解先验实际上会与多变量先验一样得到类似的BO性能。

### 3.2 Model

Whereas the standard probabilistic model in BO, e.g., a GP, quantifies p(y|x) directly, that model is hard to combine with the prior Pg(x). We therefore introduce a method to translate the standard probabilistic model p(y|x) into a model that is easier to combine with this prior. Similar to the TPE work described in Section 2.2, our generative model combines p(x|y) and p(y) instead of directly modeling p(y|x).

BO中的标准概率模型，如GP，直接量化p(y|x)，这个模型很难与先验Pg(x)结合到一起。因此我们提出了一种方法，将标准概率模型p(y|x)翻译成一个模型，更容易与这个先验进行结合。与2.2节中描述的TPE工作类似，我们的生成式模型将p(x|y)与p(y)结合到一起，而不是直接建模p(y|x)。

The computation we perform for this translation is to quantify the probability that a given input x is “good” under our standard probabilistic model p(y|x). As in TPE, we define configurations as “good” if their observed y-value is below a certain quantile γ of the observed function values (so that p(y < f_γ) = γ). We in addition exploit the fact that our standard probabilistic model p(y|x) has a Gaussian form, and under this Gaussian prediction we can compute the probability Mg(x) of the function value lying below a certain quantile using the standard closed-form formula for PI [26]:

我们为这个翻译进行的计算，是在我们的标准概率模型p(y|x)下，将给定的输入x是好的概率进行量化。如同在TPE中一样，如果观察到的y值是在观察到的所有函数值的特定分位数γ之下(p(y < f_γ) = γ)，那么我们定义这个配置为好。我们还利用了，标准概率模型p(y|x)的形式为高斯的，在这个高斯预测之下，我们可以计算，函数值低于特定分位数的概率Mg(x)，使用PI的标准的闭合形式公式：

$$M_g(x) = p(f(x) < f_γ|x,Dt) = Φ(\frac {f_γ − µ_x} {σ_x})$$(3)

where Dt = (xi, yi)^t_{i=1} are the evaluated configurations, µx and σx are the predictive mean and standard deviation of the probabilistic model at x, and Φ is the standard normal CDF, see Figure 1. Note that there are two probabilistic models here:

其中Dt = (xi, yi)^t_{i=1}是评估过的配置，µx和σx是概率模型在x处的预测性的均值和标准差，Φ是标准正态CDF，如图1所示。注意，这里有2个概率模型：

1. The standard probabilistic model of BO, with a structural prior over functions p(f), updated by data Dt to yield a posterior over functions p(f|Dt), allowing us to quantify the probability Mg(x) = p(f(x) < fγ|x, Dt) in Eq. (3).

BO的标准概率模型，函数p(f)有一个结构先验，由数据Dt进行更新，得到函数p(f|Dt)的一个后验，使我们量化式3中的概率Mg(x) = p(f(x) < fγ|x, Dt)。

2. The TPE-like generative model that combines p(y) and p(x|y) instead of directly modelling p(y|x).

TPE式的生成式模型，将p(y)和p(x|y)结合到一起，而不是直接建模p(y|x)。

Eq. (3) bridges these two models by using the probability of improvement from BO’s standard probabilistic model as the probability Mg(x) in TPE’s model. Ultimately, this is a heuristic since there is no formal connection between the two probabilistic models. However, we believe that the use of BO’s familiar, theoretically sound framework of probabilistic modelling of p(y|x), followed by the computation of the familiar PI formula is an intuitive choice for obtaining the probability of an input achieving at least a given performance threshold – exactly the term we need for TPE’s Mg(x). Similarly, we also define a probability Mb(x) of x being bad as Mb(x) = 1 − Mg(x).

式3将这两种模型连接了起来，使用BO标准概率模型的改进概率PI作为TPE模型中的概率Mg(x)。最终，这是一个启发式，因为在这两种概率模型没有任何正式的连接。但是，我们相信，使用BO的熟悉的理论上合理的概率建模p(y|x)框架，然后计算熟悉的PI公式，这是得到输入获得至少给定的性能阈值的概率的一个直观选择，这就是我们所需要的计算TPE的Mg(x)的项。类似的，我们还定义了x是坏的一个概率Mb(x)，为Mb(x) = 1 − Mg(x)。

### 3.3 Pseudo-posterior

BOPrO combines the prior Pg(x) in Section (3.1) and the model Mg(x) in Eq. (3) into a pseudo-posterior on “good” points. This pseudo-posterior represents the updated beliefs on where we can find good points, based on the prior and data that has been observed. The pseudo-posterior is computed as the product:

BOPrO将3.1节中的先验Pg(x)和式3中的模型Mg(x)结合成一个关于好的点的伪后验。这个伪后验表示我们在哪里可以找到好的点的更新的信念，基于先验和我们观察到的数据。这个伪后验计算为下面的乘积：

$$g(x) ∝ Pg(x)Mg(x)^{t/β}$$(4)

where t is the current optimization iteration, β is an optimization hyperparameter, Mg(x) is defined in Eq. (3), and Pg(x) is the prior defined in Sec 3.1, rescaled to [0, 1] using min-max scaling. We note that this pseudo-posterior is not normalized, but this suffices for BOPrO to determine the next xt as the normalization constant cancels out (c.f. Section 3.5). Since g(x) is not normalized and we include the exponent t/β in Eq. (4), we refer to g(x) as a pseudo-posterior, to emphasize that it is not a standard posterior probability distribution.

这里t和当前的优化迭代次数，β是一个优化超参数，Mg(x)如式3定义，Pg(x)是3.1节定义的先验，使用min-max scaling重新缩放到[0, 1]。我们注意到，这种伪后验并不是归一化的，但足以让BOPrO来确定下一个xt，因为归一化常数会互相抵消。由于g(x)并未归一化，我们在式4中包括了指数t/β，我们称g(x)为伪后验，以强调这并不是一个标准后验概率分布。

The t/β fraction in Eq. (4) controls how much weight is given to Mg(x). As the optimization progresses, more weight is given to Mg(x) over Pg(x). Intuitively, we put more emphasis on Mg(x) as it observes more data and becomes more accurate. We do this under the assumption that the model Mg(x) will eventually be better than the user at predicting where to find good points. This also allows to recover from misleading priors as we show in Section 4.1; similar to, and inspired by Bayesian models, the data ultimately washes out the prior. The β hyperparameter defines the balance between prior and model, with higher β values giving more importance to the prior and requiring more data to overrule it.

式4中的t/β控制对Mg(x)给与了多少权重。随着优化过程的进行，越来越多的权重在给与Mg(x)，而不是Pg(x)。直觉上来说，我们越来越强调Mg(x)，因为观察到越来越多的数据，并变得越来越精确。我们这样做有一个假设，模型Mg(x)在预测哪里可以找到好的点上，最终会被用户更加擅长。这也会从误导性的先验中恢复出来，如4.1节所示；类似的，也是受贝叶斯模型启发，数据最终会冲刷掉先验。超参数β定义了先验与模型之间的均衡，β值越高，先验的重要性就越高，需要更多的数据来推翻先验。

We note that, directly computing Eq (4) can lead to numerical issues. Namely, the pseudo-posterior can reach extremely low values if the Pg(x) and Mg(x) probabilities are low, especially as t/β grows. To prevent this, in practice, BOPrO uses the logarithm of the pseudo-posterior instead:

我们注意到，直接计算式4会带来数值问题。即，伪后验在Pg(x)和Mg(x)概率都很小的时候，会达到非常低的值，尤其是随着t/β的情况下。为避免这种情况，在实践中，BOPrO使用伪后验的log来替代：

$$log(g(x)) ∝ log(Pg(x)) + t/β· log(Mg(x))$$(5)

Once again, we also define an analogous pseudo-posterior distribution on bad x: b(x) ∝ Pb(x)Mb(x)^{t/β}. We then use these quantities to define a density model p(x|y) as follows:

再一次，我们还定义对坏的x的类似的伪后验分布：b(x) ∝ Pb(x)Mb(x)^{t/β}。然后我们使用这些量来定义一个密度模型p(x|y)如下：

$$p(x|y) ∝ g(x), if y < f_γ; b(x), if y ≥ f_γ$$(6)

### 3.4 Model and Pseudo-posterior Visualization

We visualize the prior Pg(x), the model Mg(x), and the pseudo-posterior g(x) and their evolution over the optimization iterations for a 1D-Branin function. We define the 1D-Branin by setting the second dimension of the function to the global optimum x2 = 2.275 and optimizing the first dimension. We use a Beta distribution prior Pg(x) = B(3, 3), which resembles a truncated Gaussian centered close to the global optimum, and a GP as predictive model. We perform an initial design of D + 1 = 2 random points sampled from the prior and then run BOPrO for 20 iterations.

我们对1D-Branin函数的优化迭代的演化中的先验Pg(x)，模型Mg(x)，及其伪后验g(x)进行可视化。我们定义1D-Branin，设置函数的第二个维度的全局最优值为x2=2.275，优化其第一个维度。我们使用一个Beta分布先验Pg(x)=B(3,3)，类似于一个截断的高斯函数，中心接近于全局最优值，高斯过程作为预测模型。我们进行了D+1=2个随机点的初始设计，从先验中进行采样，然后运行BOPrO 20次迭代。

Figure 2 shows the optimization at different stages. Red crosses denote the initial design and blue/green crosses denote BOPrO samples, with green samples denoting later iterations. Figure 2a shows the initialization phase (bottom) and the Beta prior (top). After 5 BO iterations, in Figure 2b (top), the pseudo-posterior is high near the global minimum, around x = π, where both the prior Pg(x) and the model Mg(x) agree there are good points. After 10 BO iterations in Figure 2c (top), there are three regions with high pseudo-posterior. The middle region, where BOPrO is exploiting until the optimum is found, and two regions to the right and left, which will lead to future exploration as shown in Figure 2d (bottom) on the right and left of the global optimum in light green crosses. After 20 iterations, see Figure 2d (top), the pseudo-posterior vanishes where the model Mg(x) is certain there will be no improvement, but it is high wherever there is uncertainty in the GP.

图2展示了在不同阶段的优化。红叉表示初始设计，蓝色/绿色叉表示BOPrO样本，绿色样本表示较后期的迭代。图2a展示了初始阶段（下面）和Beta先验（上面）。在5次BO迭代后，在图2b（上面）中，伪后验在全局最优附近是很高的，约在x = π处，这里先验Pg(x)和模型Mg(x)都表明这里有很好的点。在10次迭代后，如图2c（上面），有3个区域有很高的伪后验。中间区域，这里BOPrO不断挖掘直到找到最优点，还有两个区域位于左边和右边，这会带来未来的探索，如图2d（底部）所示，在全局最优的左边和右边的浅绿色叉。在20次迭代后，见图2d（上面），伪后验在Mg(x)很确定不会有任何改进的地方就消失了，但在GP中不确定的地方，值仍然很高。

### 3.5 Acquisition Function

We adopt the EI formulation used in Bergstra et al. [3] by replacing their Adaptive Parzen Estimators with our pseudo-posterior from Eq. (4), i.e.:

我们采用Bergstra等[3]中使用的EI，并将其的自适应Parzen估计器，替换为我们的式4中的伪后验，即

$$EI_{f_γ}(x) := \int_{-∞}^{+∞} max(f_γ − y, 0)p(y|x)dy = \int_{-∞}^{f_γ} (f_γ-y) p(x|y)p(y)/p(x) dy ∝ (γ + b(x)/g(x) (1 − γ))^{-1}$$(7)

The full derivation of Eq. (7) is shown in Appendix B. Eq. (7) shows that to maximize improvement we would like points x with high probability under g(x) and low probability under b(x), i.e., minimizing the ratio b(x)/g(x). We note that the point that minimizes the ratio for our unnormalized pseudo-posteriors will be the same that minimizes the ratio for the normalized pseudo-posterior and, thus, computing the normalized pseudo-posteriors is unnecessary.

式7的完整推导如附录B所示。式7展示了，要最大化改进，我们要g(x)概率高，和b(x)概率低的点，即，最小化比率b(x)/g(x)。我们注意到，对我们的未归一化伪后验，最小化这个比率的点，将会同样的最小化归一化伪后验的比率，因此，计算归一化伪后验是没有必要的。

The dynamics of the BOPrO algorithm can be understood in terms of the following proposition (proof in Appendix B):

BOPrO算法的动态，可以根据下面的命题理解：

**Proposition 1**. Given fγ, Pg(x), Pb(x), Mg(x), Mb(x), g(x), b(x), p(x|y), and β as above, then

$$lim_{t→∞} argmax_{x∈X} EI_{fγ}(x) = lim_{t→∞} argmax_{x∈X} Mg(x)$$

where EIfγ is the Expected Improvement acquisition function as defined in Eq. (7) and Mg(x) is as defined in Eq. (3)

In early BO iterations the prior for the optimum will have a predominant role, but in later BO iterations the model will grow more important, and as Proposition 1 shows, if BOPrO is run long enough the prior washes out and BOPrO only trusts the model Mg(x) informed by the data. Since Mg(x) is the Probability of Improvement (PI) on the probabilistic model p(y|x) then, in the limit, maximizing the acquisition function EIfγ(x) is equivalent to maximizing the PI acquisition function on the probabilistic model p(y|x). In other words, for high values of t, BOPrO converges to standard BO with a PI acquisition function.

在早期的BO迭代中，最优点的先验会占主要角色，但在后期的BO迭代中，模型会变得越来越重要，就像命题1所展示的，如果BOPrO运行的足够长，先验会冲刷掉，BOPrO只信任数据带来的模型Mg(x)。由于Mg(x)是概率模型p(y|x)上的改进概率PI，在极限上，最大化采集函数EIfγ(x)，等价于最大化在概率模型p(y|x)上的PI采集函数。换句话说，t值越高，BOPrO收敛到PI采集函数的标准BO问题。

### 3.6 Putting It All Together

Algorithm 1 shows the BOPrO algorithm. In Line 3, BOPrO starts with a design of experiments (DoE) phase, where it randomly samples a number of points from the user-defined prior Pg(x). After initialization, the BO loop starts at Line 4. In each loop iteration, BOPrO fits the models Mg(x) and Mb(x) on the previously evaluated points (Lines 5 and 6) and computes the pseudo-posteriors g(x) and b(x) (Lines 7 and 8 respectively). The EI acquisition function is computed next, using the pseudo-posteriors, and the point that maximizes EI is selected as the next point to evaluate at Line 9. The black-box function evaluation is performed at Line 10. This BO loop is repeated for a predefined number of iterations, according to the user-defined budget B.

算法1展示了BOPrO算法。在第3行，BOPrO以DoE阶段开始，从用户定义的先验Pg(x)中随机采样一定数量的点。在初始化之后，BO循环从第4行开始。在每个循环迭代中，BOPrO在之前计算的点上拟合模型Mg(x)和Mb(x)（第5行第6行），并计算伪后验g(x)和b(x)（第7行第8行）。下一步计算EI采集函数，使用伪后验，最大化EI的点选为下面评估的点，如第9行所示。黑箱函数评估在第10行进行。这个BO循环重复达到预先定义的迭代次数，根据用户定义的预算B。

## 4. Experiments

We implement both Gaussian processes (GPs) and random forests (RFs) as predictive models and use GPs in all experiments, except for our real-world experiments (Section 4.3), where we use RFs for a fair comparison. We set the model weight β = 10 and the model quantile to γ = 0.05, see our sensitivity hyperparameter study in Appendices I and J. Before starting the main BO loop in BOPrO, we randomly sample D + 1 points from the prior as an initial design consistently on all benchmarks. We optimize our EI acquisition function using a combination of multi-start local search [20] and CMA-ES [16]. We consider four synthetic benchmarks: Branin, SVM, FC-Net, and XGBoost, which are 2, 2, 6, and 8 dimensional, respectively. The last three are part of the Profet benchmarks [24], generated by a generative model built using performance data on OpenML or UCI datasets. See Appendix C for more details.

我们实现了高斯过程(GPs)和随机森林(RFs)作为预测模型，在所有试验中使用GPs，除了真实世界试验（4.3节），那里我们使用RFs以进行公平比较。我们设置模型权重β = 10，模型分位数为γ = 0.05，见我们在附录I和J中的敏感度超参数研究。在开始BOPrO中的主BO循环之前，我们从先验中随机采样D+1个点，作为初始设计，这在所有基准测试中都是一样的。我们优化EI采集函数，使用的是多起始局部搜索[20]和CMA-ES的组合。我们考虑4个合成基准测试：Branin，SVM，FC-Net和XGBoost，分别是2，2，6，8维。后面3个是Profet基准测试的一部分，使用在OpenML或UCI数据集中的性能数据构建的生成式模型来生成。见附录C中的更多细节。

### 4.1 Prior Forgetting

We first show that BOPrO can recover from a misleading prior, thanks to our model Mg(x) and the t/β parameter in the pseudo-posterior computation in Eq. (4). As BO progresses, the model Mg(x) becomes more accurate and receives more weight, guiding optimization away from the wrong prior and towards better values of the function.

我们首先展示BOPrO可以从误导的先验中恢复，这是由式4中伪后验计算中我们的模型Mg(x)和t/β参数。在BO的迭代过程中，模型Mg(x)变得越来越精确，收到更多的权重，将迭代从错误的先验处引导开，朝向函数更好的值。

Figure 3 shows BOPrO on the 1D Branin function with an exponential prior. Columns (b), (c), and (d) show BOPrO after D + 1 = 2 initial samples and 0, 10, 20 BO iterations, respectively. After initialization, as shown in Column (b), the pseudo-posterior is nearly identical to the exponential prior and guides BOPrO towards the region of the space on the right, which is towards the local optimum. This happens until the model Mg(x) becomes certain there will be no more improvement from sampling that region (Columns (c) and (d)). After that, Mg(x) guides the pseudo-posterior towards exploring regions with high uncertainty. Once the global minimum region is found, the pseudo-posterior starts balancing exploiting the global minimum and exploring regions with high uncertainty, as shown in Figure 3d (bottom). Notably, the pseudo-posterior after x > 4 falls to 0 in Figure 3d (top), as the model Mg(x) is certain there will be no improvement from sampling the region of the local optimum. We provide additional examples of forgetting in Appendix A, and a comparison of BOPrO with misleading priors, no prior, and correct priors in Appendix F.

图3展示了BOPrO在1D Branin函数上的结果，有一个指数分布的先验。列b，c，d展示了BOPrO在D+1=2个初始样本，和0，10，20次BO迭代后的结果。在初始化后，如列b所示，伪后验与指数先验基本一致，将BOPrO引导到空间右边的区域，也就是趋向于局部最优。直到模型Mg(x)变得很确定，在那个区域中采样不会有任何改进（列c和d），才停止这样。此后，Mg(x)将伪后验引导向探索不确定性很高的区域。一旦找到了全局最小区域，伪后验开始在挖掘全局最小和探索高不确定性区域之间进行均衡，如图3d（下）。值得说明的是，在x > 4之后，伪后验落到图3d（上）的0处，因为模型Mg(x)非常确定 ，在局部最优区域采样，不会有任何改进了。我们在附录A中给出了忘记的其他例子，在附录F中比较了有误导的先验，没有先验和正确的先验的BOPrO。

### 4.2 Comparison Against Strong Baselines

We build two priors for the optimum in a controlled way and evaluate BOPrO’s performance with these different prior strengths. We emphasize that in practice, manual priors would be based on the domain experts’ expertise on their applications; here, we only use artificial priors to guarantee that our prior is not biased by our own expertise for the benchmarks we used. In practice, users will manually define these priors like in our real-world experiments (Section 4.3).

我们对最优点以可控的方式构建了2个先验，用这些不同的先验强度来评估BOPrO的性能。我们强调，在实践中，手工先验应当是基于领域专家对应用的经验上的；这里，我们只是使用人造先验来保证，我们的先验并没有被我们对使用的基准测试的专业知识所偏向。在实践中，用户可以手工定义其先验，就像在我们的真实世界试验中一样（4.3节）。

Our synthetic priors take the form of Gaussian distributions centered near the optimum. For each input x ∈ X, we inject a prior of the form N(µx, σx^2), where µx is sampled from a Gaussian centered at the optimum value x_opt for that parameter µx ∼ N(xopt, σx^2), and σx is a hyperparameter of our experimental setup determining the prior’s strength. For each run of BOPrO, we sample new µx’s. This setup provides us with a synthetic prior that is close to the optimum, but not exactly centered at it, and allows us to control the strength of the prior by σx. We use two prior strengths in our experiments: a strong prior, computed with σx = 0.01, and a weak prior, computed with σx = 0.1.

我们的合成先验，是以高斯分布的形式，中心在最优值点附近。对每个输入x ∈ X，我们注入的先验的形式为N(µx, σx^2)，其中µx是从一个高斯函数中采样得到的，以最优值x_opt为中心，参数为µx ∼ N(xopt, σx^2)，而σx是我们试验设置的一个超参数，确定先验的强度。对BOPrO的每次运行，我们采样新的µx。这种设置给我们提供了接近于最优值的一种合成先验，但并不是以其为中心的，使我们可以用σx来控制先验的强度。我们在实验中使用两个先验强度：一个很强的先验，用σx=0.01计算得到，一个较弱的先验，用σx=0.1计算得到。

Figure 4 compares BOPrO to other optimizers using the log simple regret on 5 runs (mean and std error reported) on the synthetic benchmarks. We compare the results of BOPrO with weak and strong priors to 10,000× random search (RS, i.e., for each BO sample we draw 10,000 uniform random samples), sampling from the strong prior only, and Spearmint [43], a well-adopted BO approach using GPs and EI. In Appendix G, we also show a comparison of BOPrO with TPE, SMAC, and TuRBO [20,28,10]. Also, in Appendix H, we compare BOPrO to other baselines with the same prior initialization and show that the performance of the baselines remains similar.

图4比较了BOPrO和其他优化器，在合成基准测试上，运行5次，使用的是log simple regret。我们比较了带有强和弱先验的BOPrO的结果，与10000x随机搜索（RS，即，对每个BO样本，我们抽取10000次均匀随机样本），只从强先验中采样，和Spearmint，采用的很多的使用GPs和IE的BO方法。在附录G中，我们还展示了BOPrO和TPE，SMAC和TuRBO的比较。在附录H中，我们将BOPrO与其他基准进行了比较，但有相同的先验初始化，展示了基准的性能是类似的。

BOPrO with a strong prior for the optimum beats 10,000× RS and BOPrO with a weak prior on all benchmarks. It also outperforms the performance of sampling from the strong prior; this is expected because the prior sampling cannot focus on the real location of the optimum. The two methods are identical during the initialization phase because they both sample from the same prior in that phase.

在所有基准测试中，带有最佳点强先验的BOPrO，比10000x RS和带有弱先验的BOPrO都要好，还超过了从强先验中采样的性能；这是符合预期的，因为先验采样不会聚焦到最优点的真实位置上。这两种方法在初始化阶段是一样的，因为它们在这个阶段都从相同的先验中进行采样。

BOPrO with a strong prior is also more sample efficient and finds better or similar results to Spearmint on all benchmarks. Importantly, in all our experiments, BOPrO with a good prior consistently shows tremendous speedups in the early phases of the optimization process, requiring on average only 15 iterations to reach the performance that Spearmint reaches after 100 iterations (6.67× faster). Thus, in comparison to other traditional BO approaches, BOPrO makes use of the best of both worlds, leveraging prior knowledge and efficient optimization based on BO.

带有强先验的BOPrO的样本效率也更高，与Spearmint相比，在所有基准测试中都找到了更好或类似的结果。重要的是，在我们所有的试验中，带有很好先验的BOPrO，在优化过程的早期阶段，都展现出了极大的加速，平均只需要15次迭代就达到了Spearmint需要100次迭代达到的效果（6.67x加速）。因此，与其他传统的BO方法比较，BOPrO对两种知识进行了最佳利用，即先验知识，和基于BO的高效优化。

### 4.3 The Spatial Use-case

We next apply BOPrO to the Spatial [25] real-world application. Spatial is a programming language and corresponding compiler for the design of application accelerators, i.e., FPGAs. We apply BOPrO to three Spatial benchmarks, namely, 7D shallow and deep CNNs, and a 10D molecular dynamics grid application (MD Grid). We compare the performance of BOPrO to RS, manual optimization, and HyperMapper [33], the current state-of-the-art BO solution for Spatial. For a fair comparison between BOPrO and HyperMapper, since HyperMapper uses RFs as its surrogate model, here, we also use RFs in BOPrO. The manual optimization and the prior for BOPrO were provided by an unbiased Spatial developer, who is not an author of this paper. The priors were provided once and kept unchanged for the whole project. More details on the setup, including the priors used, are presented in Appendix D.

下一步，我们将BOPrO应用到Spatial的真实世界应用中。Spatial是一种编程语言和对应的编译器，用于应用加速器的设计，即FPGAs。我们将BOPrO应用到3个Spatial基准测试中，即，7D浅层CNNs和深层CNNs，和一个分子动力学网格应用。我们将BOPrO与RS，手工优化和HyperMapper进行了比较，HyperMapper是Spatial目前最好的BO解决方案。为将BOPrO和HyperMapper公平比较，由于HyperMapper使用RFs作为其代理模型，这里，我们在BOPrO中也使用RFs。手工优化和BOPrO的先验是由一位无偏的Spatial开发者提供的，不是本文的作者。先验只提供了一次，在整个项目中保持不变。在设置上的更多细节，包括使用的先验，请参考附录D。

Figure 5 shows the log regret on the Spatial benchmarks. BOPrO vastly outperforms RS in all benchmarks; notably, RS does not improve over the default configuration in MD Grid. BOPrO is also able to leverage the expert’s prior and outperforms the expert’s configuration in all benchmarks (2.68×, 1.06×, and 10.4× speedup for shallow CNN, deep CNN, and MD Grid, respectively). In the MD Grid benchmark, BOPrO achieves better performance than HyperMapper in the early stages of optimization (up to 1.73× speedup between iterations 25 and 40, see the plot inset), and achieves better final performance (1.28× speedup). For context, this is a significant improvement in the FPGA field, where a 10% improvement could qualify for acceptance in a top-tier conference. In the CNN benchmarks, BOPrO converges to the minima regions faster than HyperMapper (1.58× and 1.4× faster for shallow and deep, respectively). Thus, BOPrO leverages both the expert’s prior knowledge and BO to provide a new state of the art for Spatial.

图5展示了在Spatial基准测试上的log regret。在所有基准测试中，BOPrO都大幅超过了RS；值得注意的是，RS在MD Grid中并没有在默认配置基础上进行改进。BOPrO还可以利用专家的先验，在所有基准测试上都超过了专家的配置的性能（对浅层CNN，深层CNN和MD Grid，分别有2.68x，1.06x和10.4x的加速）。在MD Grid基准测试中，BOPrO比HyperMapper在优化的初级阶段获得了更好的性能（在迭代25到40次之间获得了1.73x的加速），最终性能上也有提升（1.28x加速）。这是FPGA领域的一个显著改进，其中10%的改进可以被顶级会议所接收。在CNN基准测试中，BOPrO收敛到极小值区域比HyperMapper要快（浅层和深层CNN分别快了1.58x和1.4x）。因此，BOPrO利用了专家的先验知识和BO，对Spatial给出了新的目前最好的结果。

## 5. Related Work

TPE by Bergstra et al. [3], the default optimizer in the popular HyperOpt package [2], supports limited hand-designed priors in the form of normal or log-normal distributions. We make three technical contributions that make BOPrO more flexible than TPE. First, we generalize over the TPE approach by allowing more flexible priors; second, BOPrO is agnostic to the probabilistic model used, allowing the use of more sample-efficient models than TPE’s kernel density estimators (e.g., we use GPs and RFs in our experiments); and third, BOPrO is inspired by Bayesian models that give more importance to the data as iterations progress. We also show that BOPrO outperforms HyperOpt’s TPE in Appendix G.

Bergstra等提出的TPE，是流行的HyperOpt包的默认优化器，支持有限的手工设计的先验，形式是正态分布或对数正态分布。我们有3个贡献，使BOPrO比TPE更加灵活。第一，我们对TPE方法进行了泛化，可以使用更灵活的先验；第二，BOPrO对使用的概率模型是无关的，可以使用比TPE的核密度估计器更加样本高效的模型（如，我们在试验中使用GPs和RFs）；第三，BOPrO受贝叶斯模型启发，在迭代进行的过程中，不断给与数据越来越重要的地位。在附录G中，我们还展示了，BOPrO比HyperOpt中的TPE性能要好。

In parallel work, Li et al. [27] allow users to specify priors via a probability distribution. Their two-level approach samples a number of configurations by maximizing samples from a GP posterior and then chooses the configuration with the highest prior as the next to evaluate. In contrast, BOPrO leverages the information from the prior more directly; is agnostic to the probabilistic model used, which is important for applications with many discrete variables like our real-world application, where RFs outperform GPs; and provably recovers from misspecified priors, while in their approach the prior never gets washed out.

在并行的工作中，Li等[27]使用户可以通过一个概率分布来指定先验。他们的两级方法，通过从GP后验中最大化样本，来采样一些配置，然后选择有最高先验的配置作为下一个要评估的。对比起来，BOPrO利用先验中的信息更加直接；与使用的概率模型无关，对很多带有离散变量的应用来说，这非常重要，就像我们的真实世界应用，其中RFs超过了GPs；而且可以从错误指定的先验中恢复回来，而在他们的方法中，先验一直没有被冲刷掉。

The work of Ramachandran et al. [39] also supports priors in the form of probability distributions. Their work uses the probability integral transform to warp the search space, stretching regions where the prior has high probability, and shrinking others. Once again, compared to their approach, BOPrO is agnostic to the probabilistic model used and directly controls the balance between prior and model via the β hyperparameter. Additionally, BOPrO’s probabilistic model is fitted independently from the prior, which ensures it is not biased by the prior, while their approach fits the model to a warped version of the space, transformed by the prior, making it difficult to recover from misleading priors.

Ramachandran等[39]的工作也支持先验，形式是概率分布。他们的工作使用概率积分变换来使搜索空间变形，将有高概率先验的区域进行伸展，并收缩其他区域。再一次，与他们的方法相比，BOPrO对使用的概率模型是无关的，直接通过超参数β控制先验和模型之间的均衡。另外，BOPrO的概率模型的拟合，与先验是无关的，这确保了并没有被先验弄偏，而他们的方法将模型拟合到空间的形变版本，由先验变换而来，使其很难从误导的先验中恢复出来。

Black-box optimization tools, such as SMAC [20] or iRace [30] also support simple hand-designed priors, e.g. log-transformations. However, these are not properly reflected in the predictive models and both cannot explicitly recover from bad priors.

黑箱优化工具，如SMAC或iRace也支持简单的手工设计的先验，如log-变换。但是，它们并没有在预测模型中合适的反应出来，也都不能从坏的先验中恢复出来。

Oh et al. [35] and Siivola et al. [42] propose structural priors for high-dimensional problems. They assume that users always place regions they expect to be good at the center of the search space and then develop BO approaches that favor configurations near the center. However, this is a rigid assumption about optimum locality, which does not allow users to freely specify their priors. Similarly, Shahriari et al. [40] focus on unbounded search spaces. The priors in their work are not about good regions of the space, but rather a regularization function that penalizes configurations based on their distance to the center of the user-defined search space. The priors are automatically derived from the search space and not provided by users.

Oh等[35]和Siivola等[42]对高维问题提出了结构先验。他们假设用户将期待是好的区域会放在搜索空间的中央，然后提出更喜欢在中央的配置的BO方法。但是，这是对最优性位置的一个刚性假设，不允许用户自由的指定其先验。类似的，Shahriari等[40]聚焦在无界的搜索空间。他们工作中的先验并不是关于空间中好的区域，而是一个正则化函数，基于到用户定义的搜索空间的中心的距离，来对配置进行惩罚。先验是从搜索空间中自动推导出来的，而并不是由用户指定的。

Our work also relates to meta-learning for BO [45], where BO is applied to many similar optimization problems in a sequence such that knowledge about the general problem structure can be exploited in future optimization problems. In contrast to meta-learning, BOPrO allows human experts to explicitly specify their priors. Furthermore, BOPrO does not depend on any meta-features [12] and incorporates the human’s prior instead of information gained from different experiments [29].

我们的工作也有BO的元学习有关，这里BO应用到很多类似的顺序优化问题中，这样关于一般性问题结构的知识可以在未来的优化问题中进行利用。与元学习进行对比，BOPrO使人类专家可以显式的指定其先验。而且，BOPrO不依赖于任何元特征[12]，结合了人类的先验，而不是从不同的试验中得到的信息。

## 6. Conclusions and Future Work

We have proposed a novel BO variant, BOPrO, that allows users to inject their expert knowledge into the optimization in the form of priors about which parts of the input space will yield the best performance. These are different than standard priors over functions which are much less intuitive for users. So far, BO failed to leverage the experience of domain experts, not only causing inefficiency but also driving users away from applying BO approaches because they could not exploit their years of knowledge in optimizing their black-box functions. BOPrO addresses this issue and we therefore expect it to facilitate the adoption of BO. We showed that BOPrO is 6.67× more sample efficient than strong BO baselines, and 10,000× faster than random search, on a common suite of benchmarks and achieves a new state-of-the-art performance on a real-world hardware design application. We also showed that BOPrO converges faster and robustly recovers from misleading priors.

我们提出了一种新的BO变体，BOPrO，使用户可以将专家知识以先验的形式注入到优化中，即输入空间的哪部分会得到最佳的性能。这与关于函数的标准先验是不一样的，对于用户非常不直观。目前，BO不能利用领域专家的经验，不仅导致效率低下，而且用户不愿意使用BO方法，因为不能利用其数年的经验在优化黑箱函数中。BOPrO解决了这个问题，我们因此期待其能促进BO的采用。我们证明了，在常见的基准测试包上，BOPrO比强BO基线的样本效率提高了6.67x，比随机搜索要快10000x，在真实世界硬件设计应用中获得了新的目前最好的结果。我们还证明了，BOPrO收敛的更快，可以从误导性的先验中稳健的恢复出来。

In future work, we will study how our approach can be used to leverage prior knowledge from meta-learning. Bringing these two worlds together will likely boost the performance of BO even further.

在未来的工作中，我们会研究我们的方法可以怎样用于利用元学习中的先验知识。将这两个世界融合到一起，很可能进一步提升BO的性能。